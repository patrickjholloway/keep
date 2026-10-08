//! surface.rs (gpu) — the GPU → CPU surface descriptor that feeds the sonification voice.
//!
//! One compute pass (`shaders/surface.wgsl`) right after field.comp: it reads the Droplets and
//! atomically bins the visible ones into a small buffer (16 B header + 1024 bins × 16 B =
//! 16 KiB). What this file teaches:
//!
//! * **A host-visible storage buffer as a readback channel.** The buffer is HOST_VISIBLE |
//!   HOST_COHERENT and mapped once for its lifetime ("persistent mapping"). No staging copy: on
//!   Apple silicon (unified memory) the GPU writes the very bytes the CPU then reads.
//! * **vkCmdFillBuffer** zeroes it on the GPU timeline at the start of the pass — the CPU never
//!   touches it, so there is no CPU/GPU race on the clear. A TRANSFER → COMPUTE barrier orders
//!   the fill before the shader's atomics.
//! * **A COMPUTE → HOST barrier** at the end makes the shader's writes available to host reads;
//!   the CPU then reads only after the frame fence signals (fence + barrier = the bytes are
//!   final). In `keep run` that happens at the next frame's fence wait (one frame latency);
//!   in `keep render`, right after the frame (zero latency, frame-synchronous).
//! * **Push constants for feedback.** The pass bins directions around the PREVIOUS frame's
//!   centroid (read back from the header), pushed as 16 bytes: no extra uniform buffer.
//! * **Its own pipeline layout over the shared set.** Set 0 is the scene set (binding 0 Scene,
//!   2 Droplets, 3 this buffer); this pipeline's layout adds the push-constant range, which the
//!   field / particle pipelines do not need. Layouts are "compatible for set 0" because the set
//!   layout is identical, so the same descriptor set binds to both.
use ash::vk;

use super::{buffers::Buffer, passes::{shader_module, SceneBindings}, shaders, GpuContext};
use crate::sonify::{SurfaceDescriptor, SURFACE_BYTES};

pub struct SurfacePass {
    pub buffer: Buffer,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// Centroid used for the next dispatch (xyz; w unused).
    center: [f32; 4],
}

impl SurfacePass {
    /// The readback buffer; created before the scene descriptor set, which points binding 3 at it.
    pub fn create_buffer(ctx: &GpuContext) -> anyhow::Result<Buffer> {
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // TRANSFER_DST: vkCmdFillBuffer needs it.
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let b = Buffer::new(ctx, SURFACE_BYTES as u64, usage, host)?;
        b.write(&vec![0u32; SURFACE_BYTES / 4]); // a valid "empty slice" before the first frame
        Ok(b)
    }

    pub fn new(ctx: &GpuContext, bindings: &SceneBindings, buffer: Buffer) -> anyhow::Result<SurfacePass> {
        let d = &ctx.device;
        unsafe {
            let push = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 16 }];
            let sl = [bindings.layout];
            let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&push), None)?;
            let module = shader_module(ctx, shaders::SURFACE_COMP)?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e)?[0];
            d.destroy_shader_module(module, None);
            Ok(SurfacePass { buffer, layout, pipeline, center: [0.0; 4] })
        }
    }

    /// Record: clear → (barrier) → bin the droplets → (barrier to host). The droplets must
    /// already be visible to compute reads (renderer.rs barrier after field.comp).
    pub fn record(&self, ctx: &GpuContext, cmd: vk::CommandBuffer, bindings: &SceneBindings, particle_count: u32) {
        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        let d = &ctx.device;
        unsafe {
            // (a) Zero the whole buffer on the GPU timeline.
            d.cmd_fill_buffer(cmd, self.buffer.buffer, 0, vk::WHOLE_SIZE, 0);
        }
        // (b) The fill is a TRANSFER write; the shader's atomics are storage reads + writes.
        ctx.memory_barrier(cmd, S::CLEAR | S::TRANSFER, A::TRANSFER_WRITE, S::COMPUTE_SHADER, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE);
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[bindings.set], &[]);
            d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&self.center));
            // (c) One invocation per particle, 256 per workgroup (= one shared histogram each).
            d.cmd_dispatch(cmd, particle_count.div_ceil(256), 1, 1);
        }
        // (d) Shader writes -> host reads (performed after the fence).
        ctx.memory_barrier(cmd, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, S::HOST, A::HOST_READ);
    }

    /// Decode the last completed frame's descriptor into `out` (no allocation) and adopt its
    /// centroid for the next dispatch. Only call after the frame fence has signalled.
    pub fn read(&mut self, out: &mut SurfaceDescriptor) {
        let words: &[u32] = bytemuck::cast_slice(&self.buffer.bytes()[..SURFACE_BYTES]);
        out.decode_from(words);
        if out.total > 0 {
            self.center = [out.centroid[0], out.centroid[1], out.centroid[2], 0.0];
        }
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        unsafe {
            ctx.device.destroy_pipeline(self.pipeline, None);
            ctx.device.destroy_pipeline_layout(self.layout, None);
        }
        self.buffer.destroy(ctx);
    }
}
