//! renderer.rs — owns GPU resources and records a frame:
//!   upload SceneParams -> FieldPass (compute) -> barrier -> ParticlePass (HDR, additive)
//!   -> barrier -> TonemapPass (8-bit) -> (offscreen readback | present).
//!
//! Synchronization model (deliberately simple, one frame in flight):
//! * The CPU writes the uniform buffer, records one command buffer, submits, and — before the
//!   next frame touches the uniform or the command buffer again — waits on `frame_fence`.
//! * Inside the command buffer, pipeline barriers order the passes (see `record_frame`).
//! * Windowed: `image_available` (acquire -> render) and `Swapchain::render_done[i]`
//!   (render -> present) are GPU-GPU semaphores; the fence is the GPU -> CPU signal.
use anyhow::Context;
use ash::vk;
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::scene::SceneParams;

use super::{
    buffers::Buffer,
    passes::{FieldPass, ParticlePass, SceneBindings, TonemapPass, TonemapPush, HDR_FORMAT},
    target::{Offscreen, Swapchain, OFFSCREEN_FORMAT},
    GpuContext,
};

/// Bytes per Droplet (3 × vec4) — must match `struct Droplet` in common.glsl.
const DROPLET_BYTES: u64 = 48;

/// Particle cloud configuration (fixed for a run).
#[derive(Clone, Copy, Debug)]
pub struct CloudSpec {
    pub count: u32,
    /// Particles are uniformly scattered in the cube [-half, half]^3.
    pub half_extent: f32,
    pub seed: u64,
}

/// Jittered grid: split the cube into n³ cells (n = ceil(∛count)) and put one particle at a
/// random spot inside each cell, in cell order, until `count` are placed. Compared to pure
/// random points this has no clumps or holes, so a thin 3D slice of a 4D surface looks evenly
/// "sampled" instead of blotchy. w = a per-particle random in [0,1) for shader flourishes.
pub fn jittered_grid(spec: &CloudSpec) -> Vec<[f32; 4]> {
    let n = (spec.count as f64).cbrt().ceil().max(1.0) as u32;
    let cell = 2.0 * spec.half_extent / n as f32;
    let mut rng = SmallRng::seed_from_u64(spec.seed);
    // Visit cells in a scrambled order so a count < n³ still covers the whole cube evenly
    // (stride by a number coprime to n³ — a cheap permutation).
    let total = n as u64 * n as u64 * n as u64;
    let mut stride = (total as f64 * 0.618_033_988_7) as u64 | 1;
    while gcd(stride, total) != 1 {
        stride += 2;
    }
    (0..spec.count as u64)
        .map(|i| {
            let c = (i * stride) % total;
            let (ix, iy, iz) = ((c % n as u64) as f32, ((c / n as u64) % n as u64) as f32, (c / (n as u64 * n as u64)) as f32);
            let j = |k: f32, rng: &mut SmallRng| -spec.half_extent + (k + rng.gen::<f32>()) * cell;
            [j(ix, &mut rng), j(iy, &mut rng), j(iz, &mut rng), rng.gen::<f32>()]
        })
        .collect()
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

pub struct Renderer {
    pub ctx: GpuContext,
    pub cloud: CloudSpec,
    pub scene_ubo: Buffer,
    pub seeds: Buffer,
    pub droplets: Buffer,
    pub bindings: SceneBindings,
    pub field_pass: FieldPass,
    pub particle_pass: ParticlePass,
    pub tonemap: TonemapPass,
    /// Tonemap knobs: x = exposure, y = vignette. Free to tweak between frames.
    pub tonemap_knobs: [f32; 4],
    pub pool: vk::CommandPool,
    pub cmd: vk::CommandBuffer,
    /// Signalled when the frame's GPU work completes; the CPU waits on it (starts signalled so
    /// the first wait is a no-op).
    pub frame_fence: vk::Fence,
    /// Windowed only: signalled when the acquired swapchain image is ready to be drawn into.
    pub image_available: vk::Semaphore,
    /// The view currently bound to the tonemap input (re-bound when the HDR image changes).
    tonemap_input: vk::ImageView,
}

impl Renderer {
    fn new(ctx: GpuContext, cloud: CloudSpec, out_format: vk::Format) -> anyhow::Result<Renderer> {
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let scene_ubo = Buffer::new(&ctx, std::mem::size_of::<SceneParams>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, host)?;
        // Seeds never change: written once by the CPU. On unified memory a host-visible
        // storage buffer is as fast to read as a "device-local" one.
        let seeds = Buffer::new(&ctx, cloud.count as u64 * 16, vk::BufferUsageFlags::STORAGE_BUFFER, host)?;
        seeds.write(&jittered_grid(&cloud));
        // Droplets are GPU-only: written by compute, read by the vertex shader.
        let droplets = Buffer::new(&ctx, cloud.count as u64 * DROPLET_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let bindings = SceneBindings::new(&ctx, scene_ubo.buffer, seeds.buffer, droplets.buffer)?;
        let field_pass = FieldPass::new(&ctx, &bindings)?;
        let particle_pass = ParticlePass::new(&ctx, &bindings, HDR_FORMAT)?;
        let tonemap = TonemapPass::new(&ctx, out_format)?;
        let d = &ctx.device;
        let (pool, cmd, frame_fence, image_available) = unsafe {
            let pool = d.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
            let fence = d.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?;
            let sem = d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            (pool, cmd, fence, sem)
        };
        Ok(Renderer {
            ctx, cloud, scene_ubo, seeds, droplets, bindings, field_pass, particle_pass, tonemap,
            tonemap_knobs: [1.0, 0.35, 0.0, 0.0], pool, cmd, frame_fence, image_available,
            tonemap_input: vk::ImageView::null(),
        })
    }

    /// Headless renderer for `keep render`.
    pub fn new_offscreen(cloud: CloudSpec, width: u32, height: u32) -> anyhow::Result<(Renderer, Offscreen)> {
        let ctx = GpuContext::new(&[])?;
        let r = Renderer::new(ctx, cloud, OFFSCREEN_FORMAT)?;
        let target = Offscreen::new(&r.ctx, width, height)?;
        Ok((r, target))
    }

    /// Windowed renderer for `keep run`.
    pub fn new_windowed(cloud: CloudSpec, window: &winit::window::Window) -> anyhow::Result<(Renderer, Swapchain)> {
        use raw_window_handle::HasDisplayHandle;
        let exts = ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?;
        let ctx = GpuContext::new(exts)?;
        let sc = Swapchain::new(&ctx, window)?;
        // The tonemap pipeline bakes the output format, so build it after the swapchain.
        let r = Renderer::new(ctx, cloud, sc.format)?;
        Ok((r, sc))
    }

    /// CPU side of a frame: wait for the previous frame, upload params, begin the command buffer.
    fn begin_frame(&mut self, params: &SceneParams) -> anyhow::Result<u32> {
        let d = &self.ctx.device;
        unsafe {
            // Wait until the GPU finished the previous frame: only then may we overwrite the
            // uniform buffer and re-record the command buffer it was using.
            d.wait_for_fences(&[self.frame_fence], true, u64::MAX)?;
            d.reset_fences(&[self.frame_fence])?;
        }
        let mut p = *params;
        p.counts[0] = p.counts[0].min(self.cloud.count);
        if p.counts[0] == 0 {
            p.counts[0] = self.cloud.count;
        }
        self.scene_ubo.write(std::slice::from_ref(&p));
        unsafe {
            d.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        }
        Ok(p.counts[0])
    }

    /// Record compute -> particles(HDR) -> tonemap(out). Leaves `out_image` in
    /// COLOR_ATTACHMENT_OPTIMAL with the tonemap's writes done.
    fn record_frame(&mut self, hdr: (vk::Image, vk::ImageView), out_image: vk::Image, out_view: vk::ImageView, extent: vk::Extent2D, count: u32) {
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        let ctx = &self.ctx;
        let cmd = self.cmd;
        let (hdr_image, hdr_view) = hdr;
        if self.tonemap_input != hdr_view {
            // Safe: we waited on the fence, so no in-flight frame uses the descriptor set.
            self.tonemap.set_input(ctx, hdr_view);
            self.tonemap_input = hdr_view;
        }

        // (1) Compute: evaluate the 4D field per particle, write Droplets.
        //     Previous frame's vertex shader read the droplets; the fence already ordered that
        //     on the CPU side, but a barrier is the honest GPU-side statement of the hazard
        //     (write-after-read), and it is free.
        ctx.memory_barrier(cmd, S::VERTEX_SHADER, A::SHADER_STORAGE_READ, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE);
        self.field_pass.record(ctx, cmd, &self.bindings, count);

        // (2) Compute writes -> vertex shader reads (read-after-write on the Droplets buffer).
        //     Without this the vertex shader could fetch stale/partial droplets.
        ctx.memory_barrier(cmd, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, S::VERTEX_SHADER, A::SHADER_STORAGE_READ);

        // (3) HDR image -> COLOR_ATTACHMENT_OPTIMAL. old = UNDEFINED: we clear it anyway, so the
        //     previous contents can be discarded. Src = last frame's tonemap read of it.
        ctx.image_barrier(
            cmd, hdr_image, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL,
            S::FRAGMENT_SHADER, A::SHADER_STORAGE_READ,
            S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE | A::COLOR_ATTACHMENT_READ,
        );
        self.particle_pass.record(ctx, cmd, &self.bindings, hdr_view, extent, count);

        // (4) HDR: attachment writes -> fragment-shader storage reads, layout -> GENERAL
        //     (the layout storage images must be in).
        ctx.image_barrier(
            cmd, hdr_image, L::COLOR_ATTACHMENT_OPTIMAL, L::GENERAL,
            S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE,
            S::FRAGMENT_SHADER, A::SHADER_STORAGE_READ,
        );
        // (5) Output image -> COLOR_ATTACHMENT_OPTIMAL (contents discarded; fully overwritten).
        //     For a swapchain image the src stage matches the acquire semaphore's wait stage.
        ctx.image_barrier(
            cmd, out_image, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL,
            S::COLOR_ATTACHMENT_OUTPUT, A::NONE,
            S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE,
        );
        self.tonemap.record(ctx, cmd, out_view, extent, TonemapPush { knobs: self.tonemap_knobs });
    }

    /// Render one frame into the offscreen target and block until its RGBA pixels are readable.
    pub fn render_offscreen<'a>(&mut self, target: &'a Offscreen, params: &SceneParams) -> anyhow::Result<&'a [u8]> {
        let count = self.begin_frame(params)?;
        self.record_frame((target.hdr.image, target.hdr.view), target.color.image, target.color.view, target.color.extent, count);
        target.record_readback(&self.ctx, self.cmd);
        let d = &self.ctx.device;
        unsafe {
            d.end_command_buffer(self.cmd)?;
            let cmds = [self.cmd];
            d.queue_submit(self.ctx.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], self.frame_fence)?;
            // Block: the caller wants the pixels now (ffmpeg pipe). Fence + the HOST barrier in
            // record_readback = the bytes in the mapped buffer are final.
            d.wait_for_fences(&[self.frame_fence], true, u64::MAX)?;
        }
        Ok(target.pixels())
    }

    /// Render + present one frame to the window. Returns false if the swapchain needs recreating.
    pub fn render_present(&mut self, swapchain: &mut Swapchain, params: &SceneParams) -> anyhow::Result<bool> {
        let count = self.begin_frame(params)?;
        // Acquire: which swapchain image is ours? `image_available` is signalled once the
        // presentation engine is done reading it.
        let idx = match unsafe { swapchain.loader.acquire_next_image(swapchain.swapchain, u64::MAX, self.image_available, vk::Fence::null()) } {
            Ok((i, false)) => i,
            Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // Nothing was submitted: end the empty recording and re-signal the fence so the
                // next begin_frame doesn't wait forever.
                unsafe {
                    self.ctx.device.end_command_buffer(self.cmd)?;
                    self.ctx.device.queue_submit(self.ctx.queue, &[], self.frame_fence)?;
                }
                return Ok(false);
            }
            Err(e) => return Err(e).context("acquire_next_image"),
        };
        let (image, view) = (swapchain.images[idx as usize], swapchain.views[idx as usize]);
        self.record_frame((swapchain.hdr.image, swapchain.hdr.view), image, view, swapchain.extent, count);
        // Output -> PRESENT_SRC: the layout the presentation engine expects.
        self.ctx.image_barrier(
            self.cmd, image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE, vk::AccessFlags2::NONE,
        );
        let done = swapchain.render_done[idx as usize];
        let d = &self.ctx.device;
        unsafe {
            d.end_command_buffer(self.cmd)?;
            let (cmds, waits, stages, signals) = ([self.cmd], [self.image_available], [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT], [done]);
            // Wait for the acquire only at COLOR_ATTACHMENT_OUTPUT: compute + particle pass may
            // run before the image is even available.
            let submit = vk::SubmitInfo::default().command_buffers(&cmds).wait_semaphores(&waits).wait_dst_stage_mask(&stages).signal_semaphores(&signals);
            d.queue_submit(self.ctx.queue, &[submit], self.frame_fence)?;
            let (scs, idxs) = ([swapchain.swapchain], [idx]);
            let present = vk::PresentInfoKHR::default().wait_semaphores(&signals).swapchains(&scs).image_indices(&idxs);
            match swapchain.loader.queue_present(self.ctx.queue, &present) {
                Ok(false) => Ok(true),
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(false),
                Err(e) => Err(e).context("queue_present"),
            }
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Runs before the fields drop, i.e. before GpuContext destroys the device.
        let ctx = &self.ctx;
        unsafe {
            let _ = ctx.device.device_wait_idle();
            ctx.device.destroy_semaphore(self.image_available, None);
            ctx.device.destroy_fence(self.frame_fence, None);
            ctx.device.destroy_command_pool(self.pool, None);
        }
        self.tonemap.destroy(ctx);
        self.particle_pass.destroy(ctx);
        self.field_pass.destroy(ctx);
        self.bindings.destroy(ctx);
        self.droplets.destroy(ctx);
        self.seeds.destroy(ctx);
        self.scene_ubo.destroy(ctx);
    }
}
