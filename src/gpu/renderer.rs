//! renderer.rs — owns GPU resources and records a frame: upload SceneParams -> FieldPass ->
//! barrier -> ParticlePass -> (offscreen readback | present).
use crate::scene::SceneParams;

use super::{
    buffers::Buffer,
    passes::{FieldPass, ParticlePass, SceneBindings},
    target::{Offscreen, Swapchain},
    GpuContext,
};

/// Particle cloud configuration (fixed for a run).
#[derive(Clone, Copy, Debug)]
pub struct CloudSpec {
    pub count: u32,
    /// Particles are uniformly scattered in the cube [-half, half]^3.
    pub half_extent: f32,
    pub seed: u64,
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
}

impl Renderer {
    /// Headless renderer for `keep render`.
    pub fn new_offscreen(cloud: CloudSpec, width: u32, height: u32) -> anyhow::Result<(Renderer, Offscreen)> {
        let _ = (cloud, width, height);
        todo!()
    }

    /// Windowed renderer for `keep run`.
    pub fn new_windowed(cloud: CloudSpec, window: &winit::window::Window) -> anyhow::Result<(Renderer, Swapchain)> {
        let _ = (cloud, window);
        todo!()
    }

    /// Render one frame into the offscreen target and block until its RGBA pixels are readable.
    pub fn render_offscreen<'a>(&mut self, target: &'a Offscreen, params: &SceneParams) -> anyhow::Result<&'a [u8]> {
        let _ = (target, params);
        todo!()
    }

    /// Render + present one frame to the window. Returns false if the swapchain needs recreating.
    pub fn render_present(&mut self, swapchain: &mut Swapchain, params: &SceneParams) -> anyhow::Result<bool> {
        let _ = (swapchain, params);
        todo!()
    }
}
