//! buffers.rs — minimal allocation: one VkDeviceMemory per buffer/image (fine for a handful).
use ash::vk;

use super::GpuContext;

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    /// Persistently mapped pointer when host-visible.
    pub mapped: Option<*mut u8>,
}

impl Buffer {
    pub fn new(ctx: &GpuContext, size: vk::DeviceSize, usage: vk::BufferUsageFlags, flags: vk::MemoryPropertyFlags) -> anyhow::Result<Buffer> {
        let _ = (ctx, size, usage, flags);
        todo!()
    }
    /// memcpy into a mapped buffer.
    pub fn write<T: bytemuck::Pod>(&self, data: &[T]) {
        let _ = data;
        todo!()
    }
    pub fn destroy(&mut self, ctx: &GpuContext) {
        let _ = ctx;
        todo!()
    }
}

pub struct Image {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Image {
    pub fn new_color(ctx: &GpuContext, extent: vk::Extent2D, format: vk::Format, usage: vk::ImageUsageFlags) -> anyhow::Result<Image> {
        let _ = (ctx, extent, format, usage);
        todo!()
    }
    pub fn destroy(&mut self, ctx: &GpuContext) {
        let _ = ctx;
        todo!()
    }
}
