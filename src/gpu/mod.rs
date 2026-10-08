//! gpu — Vulkan (ash over MoltenVK). Read in this order:
//!   context.rs   instance (portability enumeration), physical device, logical device, queue
//!   buffers.rs   buffer/image allocation helpers (host-visible uniform, device-local SSBOs)
//!   target.rs    render targets: `Offscreen` (color image + readback) and `Swapchain` (window)
//!   passes.rs    `FieldPass` (compute: seeds -> droplets) and `ParticlePass` (billboard draw)
//!   shaders.rs   SPIR-V produced by build.rs
//!   glitch.rs    `GlitchPass`: image-plane glitch chain (compute) between particles and tonemap
//!   renderer.rs  `Renderer`: owns all of the above and records one frame
//!
//! Descriptor set 0 (shared by every shader, see shaders/common.glsl):
//!   binding 0  uniform  Scene      (SceneParams)       compute + vertex + fragment
//!   binding 1  storage  Seeds      (vec4 per particle)  compute
//!   binding 2  storage  Droplets   (Droplet per particle) compute (write) + vertex (read)
pub mod buffers;
pub mod context;
pub mod glitch;
pub mod passes;
pub mod renderer;
pub mod shaders;
pub mod target;

pub use context::GpuContext;
pub use renderer::Renderer;
pub mod png;

#[cfg(test)]
mod tests;
