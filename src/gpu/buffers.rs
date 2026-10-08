//! buffers.rs — minimal allocation: one VkDeviceMemory per buffer/image (fine for a handful).
//!
//! Vulkan separates *objects* (VkBuffer/VkImage: a description) from *memory* (VkDeviceMemory:
//! bytes). The dance is always: create object -> ask its memory requirements -> allocate a
//! matching memory type -> bind. A real engine sub-allocates big blocks (VMA); we have ~6
//! allocations, so one each is the clearest.
//!
//! On Apple silicon all memory is unified, so HOST_VISIBLE|DEVICE_LOCAL types exist and
//! "uploading" is a memcpy into a mapped pointer.
use anyhow::Context;
use ash::vk;

use super::{context::color_range, GpuContext};

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    /// Persistently mapped pointer when host-visible.
    pub mapped: Option<*mut u8>,
}

impl Buffer {
    pub fn new(ctx: &GpuContext, size: vk::DeviceSize, usage: vk::BufferUsageFlags, flags: vk::MemoryPropertyFlags) -> anyhow::Result<Buffer> {
        let d = &ctx.device;
        unsafe {
            let buffer = d.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
            let req = d.get_buffer_memory_requirements(buffer);
            let mt = ctx.memory_type(req.memory_type_bits, flags)?;
            let memory = d
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt), None)
                .with_context(|| format!("allocate {size} bytes"))?;
            d.bind_buffer_memory(buffer, memory, 0)?;
            // Map once and keep it mapped for the buffer's lifetime ("persistent mapping").
            let mapped = if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                Some(d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? as *mut u8)
            } else {
                None
            };
            Ok(Buffer { buffer, memory, size, mapped })
        }
    }
    /// memcpy into a mapped buffer. Memory is HOST_COHERENT, so no flush is needed; the queue
    /// submit that follows makes host writes visible to the GPU (implicit host->device barrier).
    pub fn write<T: bytemuck::Pod>(&self, data: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        assert!(bytes.len() as u64 <= self.size, "write {} > buffer {}", bytes.len(), self.size);
        let dst = self.mapped.expect("Buffer::write on non-host-visible buffer");
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
    }
    /// Borrow the mapped bytes (caller guarantees the GPU is done writing them).
    pub fn bytes(&self) -> &[u8] {
        let p = self.mapped.expect("Buffer::bytes on non-host-visible buffer");
        unsafe { std::slice::from_raw_parts(p, self.size as usize) }
    }
    pub fn destroy(&mut self, ctx: &GpuContext) {
        unsafe {
            if self.mapped.take().is_some() {
                ctx.device.unmap_memory(self.memory);
            }
            ctx.device.destroy_buffer(self.buffer, None);
            ctx.device.free_memory(self.memory, None);
        }
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
    /// 2D, 1 mip, 1 layer, optimal tiling, device-local; starts in layout UNDEFINED.
    pub fn new_color(ctx: &GpuContext, extent: vk::Extent2D, format: vk::Format, usage: vk::ImageUsageFlags) -> anyhow::Result<Image> {
        let d = &ctx.device;
        unsafe {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = d.create_image(&info, None)?;
            let req = d.get_image_memory_requirements(image);
            let mt = ctx.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let memory = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt), None)?;
            d.bind_image_memory(image, memory, 0)?;
            let view = d.create_image_view(
                &vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(color_range()),
                None,
            )?;
            Ok(Image { image, memory, view, format, extent })
        }
    }
    pub fn destroy(&mut self, ctx: &GpuContext) {
        unsafe {
            ctx.device.destroy_image_view(self.view, None);
            ctx.device.destroy_image(self.image, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}
