//! target.rs — where a frame lands.
//! `Offscreen`: RGBA8 color image + host-visible staging buffer for readback (render mode).
//! `Swapchain`: window surface images (run mode). Both expose `RenderTarget` to the renderer.
use ash::vk;

use super::{buffers::{Buffer, Image}, GpuContext};

/// What the particle pass needs to draw into one target image.
pub trait RenderTarget {
    fn extent(&self) -> vk::Extent2D;
    fn format(&self) -> vk::Format;
}

pub struct Offscreen {
    pub color: Image,
    /// width*height*4 bytes, HOST_VISIBLE|HOST_COHERENT.
    pub readback: Buffer,
}

impl Offscreen {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> anyhow::Result<Offscreen> {
        let _ = (ctx, width, height);
        todo!()
    }
    /// Record: transition color -> TRANSFER_SRC, copy to `readback`.
    pub fn record_readback(&self, ctx: &GpuContext, cmd: vk::CommandBuffer) {
        let _ = (ctx, cmd);
        todo!()
    }
    /// After the frame's fence: borrow the tightly packed RGBA8 pixels.
    pub fn pixels(&self) -> &[u8] {
        todo!()
    }
}

impl RenderTarget for Offscreen {
    fn extent(&self) -> vk::Extent2D { self.color.extent }
    fn format(&self) -> vk::Format { self.color.format }
}

pub struct Swapchain {
    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::khr::surface::Instance,
    pub loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    /// Create surface (ash_window) + swapchain for `window`.
    pub fn new(ctx: &GpuContext, window: &winit::window::Window) -> anyhow::Result<Swapchain> {
        let _ = (ctx, window);
        todo!()
    }
    /// Rebuild after resize / VK_ERROR_OUT_OF_DATE_KHR.
    pub fn recreate(&mut self, ctx: &GpuContext, width: u32, height: u32) -> anyhow::Result<()> {
        let _ = (ctx, width, height);
        todo!()
    }
}

impl RenderTarget for Swapchain {
    fn extent(&self) -> vk::Extent2D { self.extent }
    fn format(&self) -> vk::Format { self.format }
}
