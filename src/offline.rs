//! offline.rs — `keep render`: deterministic headless render to video.
//! 1. ensure audio (synthesize if --audio path missing)   audio::synthesize
//! 2. per-frame features                                   audio::analyze
//! 3. for frame i: t = i/fps; desc = script.update(t, f[i]); params = desc.to_params(..)
//! 4. renderer.render_offscreen -> encoder.push_frame;     capture::Encoder (muxes audio)
use std::path::PathBuf;

use glam::Vec3;

use crate::{audio::{self, TrackSpec}, capture, gpu::{renderer::CloudSpec, Renderer}, scene::Camera, script::ScriptHost};

#[derive(Clone, Debug)]
pub struct RenderArgs {
    pub script: PathBuf,
    /// WAV to use; synthesized here if the file does not exist. None = `<out>.wav`.
    pub audio: Option<PathBuf>,
    pub seconds: f32,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub out: PathBuf,
    pub particles: u32,
}

/// Camera used when the script returns none: a fixed three-quarter view of the cloud.
pub fn default_camera() -> Camera {
    Camera { eye: Vec3::new(0.0, 0.8, 4.2), target: Vec3::ZERO, fov_y: 48f32.to_radians() }
}

pub fn render(args: &RenderArgs) -> anyhow::Result<()> {
    anyhow::ensure!(capture::ffmpeg_available(), "ffmpeg not found; run via `mise exec --`");
    // 1. Audio: reuse an existing WAV, otherwise synthesize an original track of the right length.
    let wav = args.audio.clone().unwrap_or_else(|| args.out.with_extension("wav"));
    if !wav.exists() {
        if let Some(d) = wav.parent() { if !d.as_os_str().is_empty() { std::fs::create_dir_all(d)?; } }
        let spec = TrackSpec { seconds: args.seconds, ..TrackSpec::default() };
        eprintln!("[render] synthesizing {:.1}s track -> {}", args.seconds, wav.display());
        audio::synthesize(&spec, &wav)?;
    }
    // 2. One Features per video frame. Pass the known BPM only if we generated it ourselves.
    let features = audio::analyze(&wav, args.fps as f32, None)?;
    let frames = (args.seconds * args.fps as f32).round() as usize;

    // 3. Script + GPU.
    let mut script = ScriptHost::load(&args.script)?;
    let cloud = CloudSpec { count: args.particles, half_extent: 1.5, seed: 1 };
    let (mut renderer, mut target) = Renderer::new_offscreen(cloud, args.width, args.height)?;
    let mut enc = capture::Encoder::start(capture::EncodeSpec {
        width: args.width, height: args.height, fps: args.fps, out: args.out.clone(), audio: Some(wav), crf: 16,
    })?;
    let aspect = args.width as f32 / args.height as f32;
    let started = std::time::Instant::now();
    // Rendering is deterministic: time comes from the frame index, never the wall clock.
    let result = (|| -> anyhow::Result<()> {
        for i in 0..frames {
            let t = i as f32 / args.fps as f32;
            let f = features.get(i).copied().unwrap_or_default();
            let desc = script.update(t, &f)?;
            let cam = desc.camera.unwrap_or_else(default_camera);
            let params = desc.to_params(t, f.as_vec4(), cam, aspect, args.particles);
            let px = renderer.render_offscreen(&target, &params)?;
            enc.push_frame(px)?;
            if i % args.fps as usize == 0 {
                eprintln!("[render] frame {i}/{frames} ({:.1}s elapsed)", started.elapsed().as_secs_f32());
            }
        }
        Ok(())
    })();
    target.destroy(&renderer.ctx);
    result?;
    let out = enc.finish()?;
    eprintln!("[render] wrote {} in {:.1}s", out.display(), started.elapsed().as_secs_f32());
    Ok(())
}
