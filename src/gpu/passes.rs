//! passes.rs — the three GPU passes of a frame.
//! FieldPass:    compute, dispatch ceil(N/256) groups of field.comp. Seeds -> Droplets.
//! ParticlePass: graphics, 6 verts × N instances, additive blending, no depth (emissive cloud),
//!               into an RGBA16F HDR image via dynamic rendering (VK_KHR_dynamic_rendering).
//! TonemapPass:  graphics, one fullscreen triangle: HDR image -> 8-bit target (offscreen or
//!               swapchain), ACES curve + sRGB encode.
//! Barriers between them are recorded by renderer.rs (it knows the images and the order).
//!
//! Pipeline anatomy (worth knowing once): a VkPipeline bakes shaders + fixed-function state;
//! a VkPipelineLayout says which descriptor-set layouts / push constants it may see; a
//! VkDescriptorSet is the actual list of buffers/images bound at those slots.
use std::ffi::CStr;

use ash::vk;

use super::{shaders, GpuContext};

const MAIN: &CStr = c"main";

/// HDR accumulation format: 16-bit float per channel so additive blending of millions of
/// faint particles neither clips at 1.0 nor bands.
pub const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

pub fn shader_module(ctx: &GpuContext, spv: &[u8]) -> anyhow::Result<vk::ShaderModule> {
    let words = shaders::words(spv);
    Ok(unsafe { ctx.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)? })
}

/// Shared descriptor set layout + pool + set for bindings 0..=2 (see gpu/mod.rs).
pub struct SceneBindings {
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
    pub pipeline_layout: vk::PipelineLayout,
}

impl SceneBindings {
    pub fn new(ctx: &GpuContext, scene_ubo: vk::Buffer, seeds: vk::Buffer, droplets: vk::Buffer) -> anyhow::Result<SceneBindings> {
        let d = &ctx.device;
        let all = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let binding = |b: u32, ty: vk::DescriptorType| {
            vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(ty).descriptor_count(1).stage_flags(all)
        };
        let bindings = [
            binding(0, vk::DescriptorType::UNIFORM_BUFFER),
            binding(1, vk::DescriptorType::STORAGE_BUFFER),
            binding(2, vk::DescriptorType::STORAGE_BUFFER),
        ];
        unsafe {
            let layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 },
            ];
            let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)?;
            let layouts = [layout];
            let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?[0];
            // Point the set's slots at our buffers (whole range each).
            let infos = [scene_ubo, seeds, droplets].map(|b| [vk::DescriptorBufferInfo { buffer: b, offset: 0, range: vk::WHOLE_SIZE }]);
            let writes = [
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&infos[0]),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&infos[1]),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&infos[2]),
            ];
            d.update_descriptor_sets(&writes, &[]);
            let pipeline_layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None)?;
            Ok(SceneBindings { layout, pool, set, pipeline_layout })
        }
    }
    pub fn destroy(&mut self, ctx: &GpuContext) {
        unsafe {
            ctx.device.destroy_pipeline_layout(self.pipeline_layout, None);
            ctx.device.destroy_descriptor_pool(self.pool, None);
            ctx.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

pub struct FieldPass {
    pub pipeline: vk::Pipeline,
}

impl FieldPass {
    pub fn new(ctx: &GpuContext, bindings: &SceneBindings) -> anyhow::Result<FieldPass> {
        let module = shader_module(ctx, shaders::FIELD_COMP)?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(MAIN);
        let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(bindings.pipeline_layout);
        let pipeline = unsafe { ctx.device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e)?[0] };
        unsafe { ctx.device.destroy_shader_module(module, None) };
        Ok(FieldPass { pipeline })
    }
    pub fn record(&self, ctx: &GpuContext, cmd: vk::CommandBuffer, bindings: &SceneBindings, particle_count: u32) {
        unsafe {
            ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            ctx.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, bindings.pipeline_layout, 0, &[bindings.set], &[]);
            // local_size_x = 256 in field.comp; the shader early-outs for id >= count.
            ctx.device.cmd_dispatch(cmd, particle_count.div_ceil(256), 1, 1);
        }
    }
    pub fn destroy(&mut self, ctx: &GpuContext) {
        unsafe { ctx.device.destroy_pipeline(self.pipeline, None) };
    }
}

/// Build a "no vertex buffers, one color attachment, dynamic viewport" graphics pipeline.
/// `additive` = blend ONE + ONE (light accumulates) vs. plain overwrite.
fn graphics_pipeline(
    ctx: &GpuContext, layout: vk::PipelineLayout, vert: &[u8], frag: &[u8], color_format: vk::Format, additive: bool,
) -> anyhow::Result<vk::Pipeline> {
    let (vs, fs) = (shader_module(ctx, vert)?, shader_module(ctx, frag)?);
    let stages = [
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(MAIN),
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(MAIN),
    ];
    // Vertices come from gl_VertexIndex / gl_InstanceIndex in the shader, so no input bindings.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    // Viewport/scissor are dynamic so the same pipeline survives window resizes.
    let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .line_width(1.0);
    let msaa = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let attach = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(additive)
        // result = src*1 + dst*1 : order-independent, so no sorting and no depth buffer needed.
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attach);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    // Dynamic rendering: instead of a VkRenderPass, tell the pipeline the attachment formats.
    let formats = [color_format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&msaa)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    let pipeline = unsafe { ctx.device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e)?[0] };
    unsafe {
        ctx.device.destroy_shader_module(vs, None);
        ctx.device.destroy_shader_module(fs, None);
    }
    Ok(pipeline)
}

/// Begin dynamic rendering into one color view (clear or keep), set viewport + scissor.
fn begin(ctx: &GpuContext, cmd: vk::CommandBuffer, view: vk::ImageView, extent: vk::Extent2D, clear: bool) {
    let attach = [vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(if clear { vk::AttachmentLoadOp::CLEAR } else { vk::AttachmentLoadOp::DONT_CARE })
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } })];
    let area = vk::Rect2D { offset: vk::Offset2D::default(), extent };
    let info = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&attach);
    unsafe {
        ctx.dynamic_rendering.cmd_begin_rendering(cmd, &info);
        let vp = vk::Viewport { x: 0.0, y: 0.0, width: extent.width as f32, height: extent.height as f32, min_depth: 0.0, max_depth: 1.0 };
        ctx.device.cmd_set_viewport(cmd, 0, &[vp]);
        ctx.device.cmd_set_scissor(cmd, 0, &[area]);
    }
}

pub struct ParticlePass {
    pub pipeline: vk::Pipeline,
    pub color_format: vk::Format,
}

impl ParticlePass {
    pub fn new(ctx: &GpuContext, bindings: &SceneBindings, color_format: vk::Format) -> anyhow::Result<ParticlePass> {
        let pipeline = graphics_pipeline(ctx, bindings.pipeline_layout, shaders::PARTICLE_VERT, shaders::PARTICLE_FRAG, color_format, true)?;
        Ok(ParticlePass { pipeline, color_format })
    }
    /// Clear `view` to black and draw all droplets into it. `view` must be in
    /// COLOR_ATTACHMENT_OPTIMAL and the droplets visible to the vertex stage (renderer barriers).
    pub fn record(&self, ctx: &GpuContext, cmd: vk::CommandBuffer, bindings: &SceneBindings, view: vk::ImageView, extent: vk::Extent2D, particle_count: u32) {
        begin(ctx, cmd, view, extent, true);
        unsafe {
            ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            ctx.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, bindings.pipeline_layout, 0, &[bindings.set], &[]);
            // 6 vertices (two triangles) per billboard, one instance per particle.
            ctx.device.cmd_draw(cmd, 6, particle_count, 0, 0);
            ctx.dynamic_rendering.cmd_end_rendering(cmd);
        }
    }
    pub fn destroy(&mut self, ctx: &GpuContext) {
        unsafe { ctx.device.destroy_pipeline(self.pipeline, None) };
    }
}

/// Tonemap knobs pushed as a push constant (tiny per-draw data, no buffer needed).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TonemapPush {
    /// x = exposure multiplier, y = vignette strength, zw unused.
    pub knobs: [f32; 4],
}

/// HDR -> 8-bit. Owns its own descriptor set (binding 0 = HDR storage image) so it is
/// independent of the scene bindings; `set_input` re-points it when the HDR image changes.
pub struct TonemapPass {
    pub set_layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl TonemapPass {
    pub fn new(ctx: &GpuContext, out_format: vk::Format) -> anyhow::Result<TonemapPass> {
        let d = &ctx.device;
        unsafe {
            let b = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            let set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&b), None)?;
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1 }];
            let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)?;
            let sl = [set_layout];
            let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sl))?[0];
            let push = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::FRAGMENT, offset: 0, size: 16 }];
            let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&push), None)?;
            let pipeline = graphics_pipeline(ctx, layout, shaders::TONEMAP_VERT, shaders::TONEMAP_FRAG, out_format, false)?;
            Ok(TonemapPass { set_layout, pool, set, layout, pipeline })
        }
    }
    /// Point binding 0 at `hdr_view` (expected in layout GENERAL when read).
    pub fn set_input(&self, ctx: &GpuContext, hdr_view: vk::ImageView) {
        let info = [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: hdr_view, image_layout: vk::ImageLayout::GENERAL }];
        let w = [vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&info)];
        unsafe { ctx.device.update_descriptor_sets(&w, &[]) };
    }
    /// Draw the fullscreen triangle into `view` (COLOR_ATTACHMENT_OPTIMAL).
    pub fn record(&self, ctx: &GpuContext, cmd: vk::CommandBuffer, view: vk::ImageView, extent: vk::Extent2D, push: TonemapPush) {
        begin(ctx, cmd, view, extent, false); // every pixel is overwritten, no clear needed
        unsafe {
            ctx.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            ctx.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[self.set], &[]);
            ctx.device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
            ctx.device.cmd_draw(cmd, 3, 1, 0, 0);
            ctx.dynamic_rendering.cmd_end_rendering(cmd);
        }
    }
    pub fn destroy(&mut self, ctx: &GpuContext) {
        unsafe {
            ctx.device.destroy_pipeline(self.pipeline, None);
            ctx.device.destroy_pipeline_layout(self.layout, None);
            ctx.device.destroy_descriptor_pool(self.pool, None);
            ctx.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
