//! passes.rs — the two GPU passes.
//! FieldPass:    compute, dispatch ceil(N/256) groups of field.comp. Writes Droplets.
//! ParticlePass: graphics, 6 verts × N instances, additive blending, no depth (emissive cloud),
//!               dynamic rendering (VK_KHR_dynamic_rendering, core in 1.3 / supported by MoltenVK).
//! A buffer barrier (COMPUTE_SHADER write -> VERTEX_SHADER read) sits between them.
use ash::vk;

use super::GpuContext;

/// Shared descriptor set layout + pool + set for bindings 0..=2 (see gpu/mod.rs).
pub struct SceneBindings {
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
    pub pipeline_layout: vk::PipelineLayout,
}

impl SceneBindings {
    pub fn new(ctx: &GpuContext, scene_ubo: vk::Buffer, seeds: vk::Buffer, droplets: vk::Buffer) -> anyhow::Result<SceneBindings> {
        let _ = (ctx, scene_ubo, seeds, droplets);
        todo!()
    }
}

pub struct FieldPass {
    pub pipeline: vk::Pipeline,
}

impl FieldPass {
    pub fn new(ctx: &GpuContext, bindings: &SceneBindings) -> anyhow::Result<FieldPass> {
        let _ = (ctx, bindings);
        todo!()
    }
    pub fn record(&self, ctx: &GpuContext, cmd: vk::CommandBuffer, bindings: &SceneBindings, particle_count: u32) {
        let _ = (ctx, cmd, bindings, particle_count);
        todo!()
    }
}

pub struct ParticlePass {
    pub pipeline: vk::Pipeline,
    pub color_format: vk::Format,
}

impl ParticlePass {
    pub fn new(ctx: &GpuContext, bindings: &SceneBindings, color_format: vk::Format) -> anyhow::Result<ParticlePass> {
        let _ = (ctx, bindings, color_format);
        todo!()
    }
    /// Clear `view` to black and draw all droplets into it.
    pub fn record(&self, ctx: &GpuContext, cmd: vk::CommandBuffer, bindings: &SceneBindings, view: vk::ImageView, extent: vk::Extent2D, particle_count: u32) {
        let _ = (ctx, cmd, bindings, view, extent, particle_count);
        todo!()
    }
}
