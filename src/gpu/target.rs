//! target.rs — where a frame lands.
//! `Offscreen`: HDR image + RGBA8 color image + host-visible staging buffer for readback.
//! `Swapchain`: HDR image + window surface images (run mode).
//! Both carry their own HDR (RGBA16F) accumulation image, sized like the output, because the
//! particle pass needs float blending and the tonemap pass then converts to 8 bits.
use anyhow::Context;
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::{
    buffers::{Buffer, Image},
    context::color_range,
    passes::HDR_FORMAT,
    GpuContext,
};

/// Output format of the offscreen path: tightly packed RGBA8, ready for ffmpeg `-pix_fmt rgba`.
pub const OFFSCREEN_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// HDR target usage: rendered to (COLOR_ATTACHMENT), then read by the tonemap (STORAGE).
fn new_hdr(ctx: &GpuContext, extent: vk::Extent2D) -> anyhow::Result<Image> {
    Image::new_color(ctx, extent, HDR_FORMAT, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
}

/// What the particle pass needs to draw into one target image.
pub trait RenderTarget {
    fn extent(&self) -> vk::Extent2D;
    fn format(&self) -> vk::Format;
}

pub struct Offscreen {
    pub hdr: Image,
    pub color: Image,
    /// width*height*4 bytes, HOST_VISIBLE|HOST_COHERENT.
    pub readback: Buffer,
}

impl Offscreen {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> anyhow::Result<Offscreen> {
        let extent = vk::Extent2D { width, height };
        let hdr = new_hdr(ctx, extent)?;
        let color = Image::new_color(ctx, extent, OFFSCREEN_FORMAT, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)?;
        let readback = Buffer::new(
            ctx,
            width as u64 * height as u64 * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        Ok(Offscreen { hdr, color, readback })
    }
    /// Record: transition color COLOR_ATTACHMENT -> TRANSFER_SRC, copy to `readback`, and make
    /// the transfer write visible to the host (the fence wait then guarantees it's done).
    pub fn record_readback(&self, ctx: &GpuContext, cmd: vk::CommandBuffer) {
        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        ctx.image_barrier(
            cmd, self.color.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE,
            S::COPY, A::TRANSFER_READ,
        );
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,   // 0 = tightly packed rows (width pixels)
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: self.color.extent.width, height: self.color.extent.height, depth: 1 },
        };
        unsafe {
            ctx.device.cmd_copy_image_to_buffer(cmd, self.color.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.readback.buffer, &[region]);
        }
        ctx.memory_barrier(cmd, S::COPY, A::TRANSFER_WRITE, S::HOST, A::HOST_READ);
    }
    /// After the frame's fence: borrow the tightly packed RGBA8 pixels.
    pub fn pixels(&self) -> &[u8] {
        self.readback.bytes()
    }
    pub fn destroy(&mut self, ctx: &GpuContext) {
        self.hdr.destroy(ctx);
        self.color.destroy(ctx);
        self.readback.destroy(ctx);
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
    pub hdr: Image,
    /// One "rendering finished" semaphore per swapchain image (present waits on it). Indexed
    /// by image so a semaphore is never re-signalled while its present may still be pending.
    pub render_done: Vec<vk::Semaphore>,
}

impl Swapchain {
    /// Create surface (ash_window) + swapchain for `window`.
    pub fn new(ctx: &GpuContext, window: &winit::window::Window) -> anyhow::Result<Swapchain> {
        let surface = unsafe {
            ash_window::create_surface(
                &ctx.entry, &ctx.instance,
                window.display_handle()?.as_raw(), window.window_handle()?.as_raw(), None,
            )?
        };
        let surface_loader = ash::khr::surface::Instance::new(&ctx.entry, &ctx.instance);
        let ok = unsafe { surface_loader.get_physical_device_surface_support(ctx.physical, ctx.queue_family, surface)? };
        anyhow::ensure!(ok, "queue family {} cannot present to this window", ctx.queue_family);
        let loader = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let size = window.inner_size();
        let placeholder = vk::Extent2D { width: 1, height: 1 };
        let mut sc = Swapchain {
            surface, surface_loader, loader,
            swapchain: vk::SwapchainKHR::null(), images: vec![], views: vec![],
            format: vk::Format::UNDEFINED, extent: placeholder,
            hdr: new_hdr(ctx, placeholder)?, render_done: vec![],
        };
        sc.recreate(ctx, size.width, size.height)?;
        Ok(sc)
    }

    /// Rebuild after resize / VK_ERROR_OUT_OF_DATE_KHR. Caller must ensure the GPU is idle
    /// (or at least done with the old images); we device_wait_idle to be safe.
    pub fn recreate(&mut self, ctx: &GpuContext, width: u32, height: u32) -> anyhow::Result<()> {
        unsafe {
            ctx.device.device_wait_idle()?;
            let caps = self.surface_loader.get_physical_device_surface_capabilities(ctx.physical, self.surface)?;
            let formats = self.surface_loader.get_physical_device_surface_formats(ctx.physical, self.surface)?;
            // UNORM, not SRGB: tonemap.frag writes already-sRGB-encoded values (same shader as
            // the offscreen RGBA8 path), so the hardware must not encode a second time.
            let fmt = formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
                .or_else(|| formats.iter().find(|f| f.format == vk::Format::R8G8B8A8_UNORM))
                .or(formats.first())
                .context("surface has no formats")?;
            // currentExtent = u32::MAX means "you choose"; otherwise the window decides.
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };
            let mut count = caps.min_image_count + 1;
            if caps.max_image_count > 0 {
                count = count.min(caps.max_image_count);
            }
            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(count)
                .image_format(fmt.format)
                .image_color_space(fmt.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO) // vsync; always supported
                .clipped(true)
                .old_swapchain(self.swapchain);
            let new = self.loader.create_swapchain(&info, None)?;
            self.destroy_views(ctx);
            if self.swapchain != vk::SwapchainKHR::null() {
                self.loader.destroy_swapchain(self.swapchain, None);
            }
            self.swapchain = new;
            self.images = self.loader.get_swapchain_images(new)?;
            self.views = self
                .images
                .iter()
                .map(|&img| {
                    ctx.device.create_image_view(
                        &vk::ImageViewCreateInfo::default().image(img).view_type(vk::ImageViewType::TYPE_2D).format(fmt.format).subresource_range(color_range()),
                        None,
                    )
                })
                .collect::<Result<_, _>>()?;
            self.render_done = (0..self.images.len())
                .map(|_| ctx.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
                .collect::<Result<_, _>>()?;
            self.format = fmt.format;
            self.extent = extent;
            self.hdr.destroy(ctx);
            self.hdr = new_hdr(ctx, extent)?;
        }
        Ok(())
    }

    fn destroy_views(&mut self, ctx: &GpuContext) {
        unsafe {
            for v in self.views.drain(..) {
                ctx.device.destroy_image_view(v, None);
            }
            for s in self.render_done.drain(..) {
                ctx.device.destroy_semaphore(s, None);
            }
        }
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        unsafe {
            let _ = ctx.device.device_wait_idle();
            self.destroy_views(ctx);
            self.hdr.destroy(ctx);
            self.loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

impl RenderTarget for Swapchain {
    fn extent(&self) -> vk::Extent2D { self.extent }
    fn format(&self) -> vk::Format { self.format }
}
