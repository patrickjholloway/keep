//! glitch.rs (gpu) — the image-plane glitch chain: a sequence of compute passes that run on the
//! HDR image after the particle pass and before the tonemap. See src/glitch.rs for the patch bay
//! and shaders/glitch_*.comp for the math of each module.
//!
//! Resources:
//!   * `ubo`     GlitchParams (host-visible, rewritten every frame)
//!   * `fft_a/b` spectral-gate work buffers, 2048×1024 cells × (re.rgb, im.rgb) = 64 MiB each
//!     (the 1024×512 working image plus its mirror extension, so the FFT sees no edge seam)
//!   * `stats`   16 bytes written by glitch_stats.comp, read by the CPU after the fence
//!   * two descriptor sets that differ only in which image is `src` and which is `dst`:
//!     sets[0] = HDR -> scratch, sets[1] = scratch -> HDR. Each active module uses the set for
//!     the current direction and flips it, so modules chain by ping-ponging two images.
//!
//! Every module is a separate pipeline over one shared pipeline layout (set 0 + a 16-byte push
//! constant). A module whose depth is 0 is simply not recorded: zero GPU cost when bypassed.
use ash::vk;

use super::{buffers::Buffer, passes::shader_module, shaders, GpuContext};
use crate::glitch::GlitchParams;

/// Spectral-gate FFT grid (must match FN / FM in glitch_spectral.comp): twice the working
/// image in each axis, which LOAD fills with the image's mirror extension.
pub const FFT_N: u32 = 2048;
pub const FFT_M: u32 = 1024;
/// Working image the frame is resampled to (IW / IH in glitch_spectral.comp).
pub const FFT_IW: u32 = FFT_N / 2;
pub const FFT_IH: u32 = FFT_M / 2;
const CPLX_BYTES: u64 = 32;

pub struct GlitchPass {
    pub ubo: Buffer,
    pub fft_a: Buffer,
    pub fft_b: Buffer,
    pub stats: Buffer,
    pub set_layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub sets: [vk::DescriptorSet; 2],
    pub layout: vk::PipelineLayout,
    sync: vk::Pipeline,
    ring: vk::Pipeline,
    smear: vk::Pipeline,
    disp: vk::Pipeline,
    warp: vk::Pipeline,
    spectral: vk::Pipeline,
    crush: vk::Pipeline,
    stats_pipe: vk::Pipeline,
    bound: (vk::ImageView, vk::ImageView),
}

impl GlitchPass {
    pub fn new(ctx: &GpuContext) -> anyhow::Result<GlitchPass> {
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let ubo = Buffer::new(ctx, std::mem::size_of::<GlitchParams>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, host)?;
        let fft_bytes = FFT_N as u64 * FFT_M as u64 * CPLX_BYTES;
        let fft_a = Buffer::new(ctx, fft_bytes, vk::BufferUsageFlags::STORAGE_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let fft_b = Buffer::new(ctx, fft_bytes, vk::BufferUsageFlags::STORAGE_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let stats = Buffer::new(ctx, 16, vk::BufferUsageFlags::STORAGE_BUFFER, host)?;
        stats.write(&[0f32; 4]);
        let d = &ctx.device;
        unsafe {
            let cs = vk::ShaderStageFlags::COMPUTE;
            let b = |i: u32, ty: vk::DescriptorType| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(cs);
            use vk::DescriptorType as T;
            let bindings = [b(0, T::UNIFORM_BUFFER), b(1, T::STORAGE_IMAGE), b(2, T::STORAGE_IMAGE), b(3, T::STORAGE_BUFFER), b(4, T::STORAGE_BUFFER), b(5, T::STORAGE_BUFFER)];
            let set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let sizes = [
                vk::DescriptorPoolSize { ty: T::UNIFORM_BUFFER, descriptor_count: 2 },
                vk::DescriptorPoolSize { ty: T::STORAGE_IMAGE, descriptor_count: 4 },
                vk::DescriptorPoolSize { ty: T::STORAGE_BUFFER, descriptor_count: 6 },
            ];
            let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&sizes), None)?;
            let sl = [set_layout, set_layout];
            let v = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sl))?;
            let sets = [v[0], v[1]];
            // Buffers never change: write them once into both sets.
            for set in sets {
                let infos = [ubo.buffer, fft_a.buffer, fft_b.buffer, stats.buffer].map(|b| [vk::DescriptorBufferInfo { buffer: b, offset: 0, range: vk::WHOLE_SIZE }]);
                let writes = [
                    vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(T::UNIFORM_BUFFER).buffer_info(&infos[0]),
                    vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(T::STORAGE_BUFFER).buffer_info(&infos[1]),
                    vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4).descriptor_type(T::STORAGE_BUFFER).buffer_info(&infos[2]),
                    vk::WriteDescriptorSet::default().dst_set(set).dst_binding(5).descriptor_type(T::STORAGE_BUFFER).buffer_info(&infos[3]),
                ];
                d.update_descriptor_sets(&writes, &[]);
            }
            let push = [vk::PushConstantRange { stage_flags: cs, offset: 0, size: 16 }];
            let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]).push_constant_ranges(&push), None)?;
            let pipe = |spv: &[u8]| -> anyhow::Result<vk::Pipeline> {
                let module = shader_module(ctx, spv)?;
                let stage = vk::PipelineShaderStageCreateInfo::default().stage(cs).module(module).name(c"main");
                let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
                let p = d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e)?[0];
                d.destroy_shader_module(module, None);
                Ok(p)
            };
            Ok(GlitchPass {
                sync: pipe(shaders::GLITCH_SYNC)?,
                ring: pipe(shaders::GLITCH_RING)?,
                smear: pipe(shaders::GLITCH_SMEAR)?,
                disp: pipe(shaders::GLITCH_DISP)?,
                warp: pipe(shaders::GLITCH_WARP)?,
                spectral: pipe(shaders::GLITCH_SPECTRAL)?,
                crush: pipe(shaders::GLITCH_CRUSH)?,
                stats_pipe: pipe(shaders::GLITCH_STATS)?,
                ubo, fft_a, fft_b, stats, set_layout, pool, sets, layout,
                bound: (vk::ImageView::null(), vk::ImageView::null()),
            })
        }
    }

    /// Forget which views the sets point at. Call after the images are recreated (resize):
    /// a new view can reuse a destroyed one's handle value, which would defeat the cache.
    pub fn invalidate_images(&mut self) {
        self.bound = (vk::ImageView::null(), vk::ImageView::null());
    }

    /// Point the two sets at (hdr, scratch). Only call when the GPU is not using the sets
    /// (the renderer calls it right after the frame fence wait).
    fn bind_images(&mut self, ctx: &GpuContext, hdr: vk::ImageView, scratch: vk::ImageView) {
        if self.bound == (hdr, scratch) {
            return;
        }
        let img = |v: vk::ImageView| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: v, image_layout: vk::ImageLayout::GENERAL }];
        let (h, s) = (img(hdr), img(scratch));
        fn w<'a>(set: vk::DescriptorSet, b: u32, info: &'a [vk::DescriptorImageInfo; 1]) -> vk::WriteDescriptorSet<'a> {
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(info)
        }
        let writes = [w(self.sets[0], 1, &h), w(self.sets[0], 2, &s), w(self.sets[1], 1, &s), w(self.sets[1], 2, &h)];
        unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };
        self.bound = (hdr, scratch);
    }

    /// Mean luminance (tone-compressed, 0..1) of the last completed frame, before glitches.
    pub fn mean_luma(&self) -> f32 {
        let b = self.stats.bytes();
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// Record the chain. Both images must be in GENERAL and visible to compute. Returns true if
    /// the final result is in `scratch` (odd number of active modules), false if in `hdr`.
    pub fn record(&mut self, ctx: &GpuContext, cmd: vk::CommandBuffer, p: &GlitchParams, hdr: vk::ImageView, scratch: vk::ImageView, extent: vk::Extent2D) -> bool {
        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        self.ubo.write(std::slice::from_ref(p));
        self.bind_images(ctx, hdr, scratch);
        let d = &ctx.device;
        let rw = A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE;
        // compute -> compute: each pass sees the previous pass's image/buffer writes
        let barrier = || ctx.memory_barrier(cmd, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, S::COMPUTE_SHADER, rw);
        let groups2d = (extent.width.div_ceil(16), extent.height.div_ceil(16));
        let mut dir = 0usize; // 0: hdr -> scratch, 1: scratch -> hdr
        let bind = |pipe: vk::Pipeline, set: vk::DescriptorSet, push: [u32; 4]| unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[set], &[]);
            d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push));
        };

        // Image stats of the clean frame (one invocation; read by the CPU next frame).
        bind(self.stats_pipe, self.sets[0], [0; 4]);
        unsafe { d.cmd_dispatch(cmd, 1, 1, 1) };

        let image_pass = |pipe: vk::Pipeline, dir: &mut usize| {
            bind(pipe, self.sets[*dir], [0; 4]);
            unsafe { d.cmd_dispatch(cmd, groups2d.0, groups2d.1, 1) };
            barrier();
            *dir ^= 1;
        };
        if p.sync_on() { image_pass(self.sync, &mut dir); }
        if p.ring_on() { image_pass(self.ring, &mut dir); }
        if p.smear_on() {
            // one thread per row (IIR is sequential along x)
            bind(self.smear, self.sets[dir], [0; 4]);
            unsafe { d.cmd_dispatch(cmd, extent.height.div_ceil(64), 1, 1) };
            barrier();
            dir ^= 1;
        }
        if p.disp_on() { image_pass(self.disp, &mut dir); }
        if p.warp_on() { image_pass(self.warp, &mut dir); }
        if p.spec_on() {
            let set = self.sets[dir];
            let cells = (FFT_N * FFT_M).div_ceil(256);
            let butterflies = (FFT_N * FFT_M / 2).div_ceil(256);
            let mut in_b = 0u32; // which FFT buffer holds the data (flag bit 2)
            bind(self.spectral, set, [0, 0, 0, 0]);
            unsafe { d.cmd_dispatch(cmd, cells, 1, 1) };
            barrier();
            let stages = |inverse: u32, in_b: &mut u32| {
                // forward: rows then columns; inverse: columns then rows (any order works,
                // the 2D DFT is separable)
                let order = if inverse == 0 { [(0u32, FFT_N), (1, FFT_M)] } else { [(1, FFT_M), (0, FFT_N)] };
                for (axis, len) in order {
                    let mut ns = 1;
                    while ns < len {
                        bind(self.spectral, set, [1, ns, axis | (inverse << 1) | (*in_b << 2), 0]);
                        unsafe { d.cmd_dispatch(cmd, butterflies, 1, 1) };
                        barrier();
                        *in_b ^= 1;
                        ns *= 2;
                    }
                }
            };
            stages(0, &mut in_b);
            bind(self.spectral, set, [2, 0, in_b << 2, 0]);
            unsafe { d.cmd_dispatch(cmd, cells, 1, 1) };
            barrier();
            stages(1, &mut in_b);
            bind(self.spectral, set, [3, 0, in_b << 2, 0]);
            unsafe { d.cmd_dispatch(cmd, (extent.width * extent.height).div_ceil(256), 1, 1) };
            barrier();
            dir ^= 1;
        }
        if p.crush_on() { image_pass(self.crush, &mut dir); }
        // stats buffer -> host (read after the fence)
        ctx.memory_barrier(cmd, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, S::HOST, A::HOST_READ);
        dir == 1
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        unsafe {
            for p in [self.sync, self.ring, self.smear, self.disp, self.warp, self.spectral, self.crush, self.stats_pipe] {
                ctx.device.destroy_pipeline(p, None);
            }
            ctx.device.destroy_pipeline_layout(self.layout, None);
            ctx.device.destroy_descriptor_pool(self.pool, None);
            ctx.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
        self.ubo.destroy(ctx);
        self.fft_a.destroy(ctx);
        self.fft_b.destroy(ctx);
        self.stats.destroy(ctx);
    }
}
