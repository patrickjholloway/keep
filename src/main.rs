use ash::vk;

fn main() -> anyhow::Result<()> {
    let lib = std::env::var("KEEP_VULKAN_LIB")?;
    let entry = unsafe { ash::Entry::load_from(&lib)? };
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    let instance = unsafe {
        entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)?
    };
    for pd in unsafe { instance.enumerate_physical_devices()? } {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = props.device_name_as_c_str()?.to_string_lossy();
        println!("vulkan device: {name} (api {}.{})",
            vk::api_version_major(props.api_version), vk::api_version_minor(props.api_version));
    }
    unsafe { instance.destroy_instance(None) };
    Ok(())
}
