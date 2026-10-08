//! keep — 4D implicit-field slice renderer.
//!
//!   keep run    scripts/scene.lua [--audio a.wav | --mic] [--frames N]   (N = exit after N frames; smoke test)
//!   keep render scripts/scene.lua --audio out.wav --seconds N --fps 30 --size 1920x1080 --out out.mp4 [--particles N]
//!   keep probe                       (list Vulkan devices; smoke test)
//!
//! See docs/ARCHITECTURE.md for the SceneParams contract and module ownership.
mod app;
mod audio;
mod camera;
mod capture;
mod field;
mod glitch;
mod live;
mod gpu;
mod math4d;
mod offline;
mod scene;
mod script;

use std::path::PathBuf;

use anyhow::{bail, Context};
use ash::vk;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => {
            let script = PathBuf::from(args.get(1).context("usage: keep run <scene.lua> [--audio a.wav | --mic] [--frames N]")?);
            let (mut frames, mut wav, mut mic) = (None, None, false);
            let mut it = args[2..].iter();
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--frames" => frames = Some(it.next().context("--frames N")?.parse()?),
                    "--audio" => wav = Some(PathBuf::from(it.next().context("--audio a.wav")?)),
                    "--mic" => mic = true,
                    f => bail!("unknown flag {f}"),
                }
            }
            let audio = if mic {
                app::AudioSource::Mic(live::MicAnalyzer::start()?)
            } else {
                // Default: the exemplar track (75 s, as in the README render), cached in out/.
                let wav = wav.unwrap_or_else(|| PathBuf::from("out/keep-exemplar.wav"));
                let track = live::prepare(&wav, 75.0)?;
                let player = live::Player::start(&wav)?;
                eprintln!("[run] playing {} ({:.1}s loop)", wav.display(), player.duration());
                app::AudioSource::File { player, track }
            };
            app::run(&script, frames, audio)
        }
        Some("render") => offline::render(&parse_render(&args[1..])?),
        Some("probe") => probe(),
        _ => bail!("usage: keep run <scene.lua> | keep render <scene.lua> --audio a.wav --seconds N --fps 30 --size WxH --out out.mp4 | keep probe"),
    }
}

fn parse_render(a: &[String]) -> anyhow::Result<offline::RenderArgs> {
    let script = PathBuf::from(a.first().context("render: missing script path")?);
    let mut r = offline::RenderArgs {
        script, audio: None, seconds: 30.0, fps: 30, width: 1920, height: 1080,
        out: "out.mp4".into(), particles: 750_000,
    };
    let mut it = a[1..].iter();
    while let Some(flag) = it.next() {
        let v = it.next().with_context(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--audio" => r.audio = Some(v.into()),
            "--seconds" => r.seconds = v.parse()?,
            "--fps" => r.fps = v.parse()?,
            "--out" => r.out = v.into(),
            "--particles" => r.particles = v.parse()?,
            "--size" => {
                let (w, h) = v.split_once('x').context("--size WxH")?;
                r.width = w.parse()?;
                r.height = h.parse()?;
            }
            _ => bail!("unknown flag {flag}"),
        }
    }
    Ok(r)
}

/// Original smoke test: enumerate Vulkan devices through MoltenVK.
fn probe() -> anyhow::Result<()> {
    let lib = std::env::var("KEEP_VULKAN_LIB").context("KEEP_VULKAN_LIB unset; run via `mise exec --`")?;
    let entry = unsafe { ash::Entry::load_from(&lib)? };
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    // Loading MoltenVK directly (no Khronos loader) means VK_KHR_portability_enumeration is
    // usually NOT offered — it is a loader extension. Enable it only when present.
    let available = unsafe { entry.enumerate_instance_extension_properties(None)? };
    let has_portability = available.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::khr::portability_enumeration::NAME));
    let exts: Vec<_> = if has_portability { vec![ash::khr::portability_enumeration::NAME.as_ptr()] } else { vec![] };
    let mut info = vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&exts);
    if has_portability { info = info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR); }
    let instance = unsafe { entry.create_instance(&info, None)? };
    for pd in unsafe { instance.enumerate_physical_devices()? } {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = props.device_name_as_c_str()?.to_string_lossy();
        println!("vulkan device: {name} (api {}.{})",
            vk::api_version_major(props.api_version), vk::api_version_minor(props.api_version));
    }
    unsafe { instance.destroy_instance(None) };
    Ok(())
}
