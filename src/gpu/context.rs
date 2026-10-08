//! context.rs — Vulkan bootstrap for MoltenVK.
//!
//! Vulkan start-up is always the same four steps:
//!   1. Entry     — the function table of the Vulkan *implementation* (here: libMoltenVK.dylib,
//!                  dlopened from $KEEP_VULKAN_LIB; there is no Khronos loader in between).
//!   2. Instance  — "I am an app using Vulkan"; instance extensions (surface, portability).
//!   3. Physical device — pick a GPU (on this Mac: exactly one, the Apple M5 Pro).
//!   4. Logical device + queue — "open" the GPU with the device extensions/features we use.
//!
//! MoltenVK is a *portability* implementation (Vulkan layered on Metal):
//! * the instance enables VK_KHR_portability_enumeration + ENUMERATE_PORTABILITY *only when
//!   offered* — it is a loader extension, and MoltenVK loaded directly does NOT list it;
//!   enabling it unconditionally fails with VK_ERROR_EXTENSION_NOT_PRESENT.
//! * the device MUST enable VK_KHR_portability_subset (MoltenVK advertises it).
//!
//! MoltenVK reports API 1.2 here, so the two "modern Vulkan" features we rely on are enabled as
//! KHR extensions (they became core in 1.3, same semantics):
//! * VK_KHR_dynamic_rendering — draw into image views directly, no VkRenderPass/VkFramebuffer.
//! * VK_KHR_synchronization2  — clearer barrier structs (stage+access per side, per barrier).
use std::ffi::CStr;

use anyhow::{bail, Context};
use ash::vk;

pub struct GpuContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    /// One queue family that supports graphics + compute (+ present when windowed).
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    /// Extension function tables (KHR versions of what is core in Vulkan 1.3).
    pub dynamic_rendering: ash::khr::dynamic_rendering::Device,
    pub sync2: ash::khr::synchronization2::Device,
    /// True when created with window surface extensions (VK_KHR_swapchain is then enabled).
    pub windowed: bool,
}

fn has_ext(list: &[vk::ExtensionProperties], name: &CStr) -> bool {
    list.iter().any(|e| e.extension_name_as_c_str() == Ok(name))
}

impl GpuContext {
    /// Load $KEEP_VULKAN_LIB, create instance + device. `window_exts` = surface extensions
    /// from ash_window::enumerate_required_extensions (empty for headless).
    pub fn new(window_exts: &[*const std::ffi::c_char]) -> anyhow::Result<GpuContext> {
        let lib = std::env::var("KEEP_VULKAN_LIB").context("KEEP_VULKAN_LIB unset; run via `mise exec --`")?;
        // SAFETY: dlopen of a Vulkan ICD; the returned Entry keeps the library loaded.
        let entry = unsafe { ash::Entry::load_from(&lib)? };

        // ---- 2. instance -------------------------------------------------------------------
        let app = vk::ApplicationInfo::default()
            .application_name(c"keep")
            .api_version(vk::API_VERSION_1_2);
        let available = unsafe { entry.enumerate_instance_extension_properties(None)? };
        let portability = has_ext(&available, ash::khr::portability_enumeration::NAME);
        let mut inst_exts: Vec<*const std::ffi::c_char> = window_exts.to_vec();
        if portability {
            inst_exts.push(ash::khr::portability_enumeration::NAME.as_ptr());
        }
        // Needed to query portability_subset features through vkGetPhysicalDeviceFeatures2 on 1.0
        // instances; harmless (and core) on 1.2, but MoltenVK still lists it — enable if present.
        if has_ext(&available, ash::khr::get_physical_device_properties2::NAME) {
            inst_exts.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }
        let mut info = vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&inst_exts);
        if portability {
            info = info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }
        let instance = unsafe { entry.create_instance(&info, None)? };

        // ---- 3. physical device + queue family ---------------------------------------------
        // Prefer a discrete/integrated GPU over a CPU implementation; need a queue family that
        // does both GRAPHICS and COMPUTE so one queue (and simple ordering) serves every pass.
        let mut best: Option<(vk::PhysicalDevice, u32, i32)> = None;
        for pd in unsafe { instance.enumerate_physical_devices()? } {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                _ => 1,
            };
            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let want = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
            if let Some(qf) = families.iter().position(|f| f.queue_flags.contains(want)) {
                if best.map_or(true, |b| score > b.2) {
                    best = Some((pd, qf as u32, score));
                }
            }
        }
        let (physical, queue_family, _) = best.context("no Vulkan device with a graphics+compute queue")?;

        // ---- 4. logical device ---------------------------------------------------------------
        let dev_available = unsafe { instance.enumerate_device_extension_properties(physical)? };
        let mut dev_exts = vec![
            ash::khr::dynamic_rendering::NAME.as_ptr(),
            ash::khr::synchronization2::NAME.as_ptr(),
            // dynamic_rendering depends on these two (core in 1.2, listed for 1.1-style deps).
            ash::khr::depth_stencil_resolve::NAME.as_ptr(),
            ash::khr::create_renderpass2::NAME.as_ptr(),
        ];
        for need in [ash::khr::dynamic_rendering::NAME, ash::khr::synchronization2::NAME] {
            if !has_ext(&dev_available, need) {
                bail!("device lacks {need:?}");
            }
        }
        if has_ext(&dev_available, ash::khr::portability_subset::NAME) {
            dev_exts.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        let windowed = !window_exts.is_empty();
        if windowed {
            dev_exts.push(ash::khr::swapchain::NAME.as_ptr());
        }
        dev_exts.retain(|&p| has_ext(&dev_available, unsafe { CStr::from_ptr(p) }));

        // Features are switched on by chaining structs into DeviceCreateInfo via p_next.
        let mut dyn_feat = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut sync2_feat = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let prio = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&prio)];
        let dinfo = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&dev_exts)
            .push_next(&mut dyn_feat)
            .push_next(&mut sync2_feat);
        let device = unsafe { instance.create_device(physical, &dinfo, None)? };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let dynamic_rendering = ash::khr::dynamic_rendering::Device::new(&instance, &device);
        let sync2 = ash::khr::synchronization2::Device::new(&instance, &device);
        Ok(GpuContext { entry, instance, physical, device, queue_family, queue, mem_props, dynamic_rendering, sync2, windowed })
    }

    /// Pick a memory type index matching `type_bits` and `flags`.
    /// `type_bits` comes from vkGet*MemoryRequirements: bit i set = memory type i is allowed.
    pub fn memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> anyhow::Result<u32> {
        (0..self.mem_props.memory_type_count)
            .find(|&i| type_bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(flags))
            .with_context(|| format!("no memory type for bits {type_bits:#b} flags {flags:?}"))
    }

    /// Record commands into a throwaway command buffer, submit, and wait (uploads, readback).
    pub fn one_shot(&self, pool: vk::CommandPool, f: impl FnOnce(vk::CommandBuffer)) -> anyhow::Result<()> {
        unsafe {
            let alloc = vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&alloc)?[0];
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            f(cmd);
            self.device.end_command_buffer(cmd)?;
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            self.device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(pool, &cmds);
        }
        Ok(())
    }

    /// Image layout transition / memory dependency for one image (synchronization2 style).
    /// Reads as: "wait for `src_stage` writes of kind `src_access` to finish and be visible,
    /// move the image from `old` to `new` layout, before `dst_stage` does `dst_access`".
    #[allow(clippy::too_many_arguments)]
    pub fn image_barrier(
        &self, cmd: vk::CommandBuffer, image: vk::Image,
        old: vk::ImageLayout, new: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags2, src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2, dst_access: vk::AccessFlags2,
    ) {
        let b = [vk::ImageMemoryBarrier2::default()
            .image(image)
            .old_layout(old)
            .new_layout(new)
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(color_range())];
        unsafe { self.sync2.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
    }

    /// Global memory dependency (buffers): same reading as `image_barrier`, no layouts.
    pub fn memory_barrier(
        &self, cmd: vk::CommandBuffer,
        src_stage: vk::PipelineStageFlags2, src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2, dst_access: vk::AccessFlags2,
    ) {
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)];
        unsafe { self.sync2.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b)) };
    }
}

/// The whole (single-mip, single-layer) color aspect of an image.
pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
