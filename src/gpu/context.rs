//! context.rs — Vulkan bootstrap for MoltenVK.
//! MoltenVK is a *portability* implementation: the instance must enable
//! VK_KHR_portability_enumeration + ENUMERATE_PORTABILITY *when offered* (it is a loader
//! extension; MoltenVK loaded directly via ash::Entry::load_from does NOT list it — enabling it
//! unconditionally fails with VK_ERROR_EXTENSION_NOT_PRESENT). The device MUST enable
//! VK_KHR_portability_subset (MoltenVK advertises it).
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
}

impl GpuContext {
    /// Load $KEEP_VULKAN_LIB, create instance + device. `window_exts` = surface extensions
    /// from ash_window::enumerate_required_extensions (empty for headless).
    pub fn new(window_exts: &[*const std::ffi::c_char]) -> anyhow::Result<GpuContext> {
        let _ = window_exts;
        todo!("instance + device with portability")
    }

    /// Pick a memory type index matching `type_bits` and `flags`.
    pub fn memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> anyhow::Result<u32> {
        let _ = (type_bits, flags);
        todo!()
    }

    /// Record commands into a throwaway command buffer, submit, and wait (uploads, readback).
    pub fn one_shot(&self, pool: vk::CommandPool, f: impl FnOnce(vk::CommandBuffer)) -> anyhow::Result<()> {
        let _ = (pool, f);
        todo!()
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
