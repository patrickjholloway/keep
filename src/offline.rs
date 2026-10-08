//! offline.rs — `keep render`: deterministic headless render to video.
//! 1. ensure audio (synthesize if --audio path missing)   audio::synthesize
//! 2. per-frame features                                   audio::analyze
//! 3. for frame i: t = i/fps; desc = script.update(t, f[i]); params = desc.to_params(..)
//!    renderer.glitch = glitch::resolve(desc.glitch, t, f[i], mean_luma of frame i-1)
//! 4. renderer.render_offscreen -> encoder.push_frame;     capture::Encoder (video only)
//! 5. sonify: after each frame, renderer.surface (that frame's descriptor, zero latency) sets
//!    the voice's targets and the voice renders exactly that frame's samples
//!    [i·sr/fps, (i+1)·sr/fps) — frame-synchronous and deterministic.
//! 6. if the script ever enabled `sonify`: mix voice + input audio into ONE stereo track
//!    (`<out>.mix.wav`; the voice alone is kept as `<out>.sonify.wav`), scaled down only if
//!    the sum would clip; then mux video + audio (capture::mux). A single mixed track rather
//!    than a second audio stream, because most players only ever play the first stream.
use std::path::PathBuf;

use glam::Vec3;

use crate::{audio::{self, TrackSpec}, capture, glitch, gpu::{renderer::CloudSpec, Renderer}, scene::Camera, script::ScriptHost, sonify::Sonifier};

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
    Camera { eye: Vec3::new(0.0, 0.8, 4.2), target: Vec3::ZERO, fov_y: 48f32.to_radians(), roll: 0.0 }
}

pub fn render(args: &RenderArgs) -> anyhow::Result<()> {
    anyhow::ensure!(capture::ffmpeg_available(), "ffmpeg not found; run via `mise exec --`");
    // 1. Audio: reuse an existing WAV, otherwise synthesize an original track of the right length.
    let wav = args.audio.clone().unwrap_or_else(|| args.out.with_extension("wav"));
    if !wav.exists() {
        anyhow::ensure!(wav.extension().map_or(false, |e| e.eq_ignore_ascii_case("wav")),
            "{}: not found (only a missing .wav path is synthesized)", wav.display());
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
    // Video goes to a temporary file; the audio (original or mixed with the voice) is muxed at the end.
    let video_tmp = args.out.with_extension("video.mp4");
    let mut enc = capture::Encoder::start(capture::EncodeSpec {
        width: args.width, height: args.height, fps: args.fps, out: video_tmp.clone(), audio: None, crf: 16,
    })?;
    // Sonify voice at the input's own sample rate (so mixing needs no resampling).
    let (input, in_ch, in_rate) = audio::decode(&wav)?;
    let in_frames = input.len() / in_ch.max(1);
    let mut voice = Sonifier::new(in_rate as f32);
    let voice_len = in_frames.max(((frames as f64) * in_rate as f64 / args.fps as f64).ceil() as usize);
    let (mut vl, mut vr) = (vec![0f32; voice_len], vec![0f32; voice_len]);
    let mut voice_used = false;
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
            // Glitch patch bay: resolve this frame's cables (mean_luma = the previous frame's).
            renderer.glitch = glitch::resolve(&desc.glitch, t, &f, renderer.mean_luma(), i as u32);
            let px = renderer.render_offscreen(&target, &params)?;
            enc.push_frame(px)?;
            // This frame's surface -> the voice, for exactly this frame's span of samples.
            voice_used |= desc.sonify.enable;
            voice.set_targets(&renderer.surface, &desc.sonify);
            let s0 = (i as f64 * in_rate as f64 / args.fps as f64).round() as usize;
            let s1 = (((i + 1) as f64 * in_rate as f64 / args.fps as f64).round() as usize).min(voice_len);
            if s1 > s0 { voice.render(&mut vl[s0..s1], &mut vr[s0..s1]); }
            if i % args.fps as usize == 0 {
                eprintln!("[render] frame {i}/{frames} ({:.1}s elapsed)", started.elapsed().as_secs_f32());
            }
        }
        Ok(())
    })();
    target.destroy(&renderer.ctx);
    result?;
    enc.finish()?;
    // Mix only the rendered span (the input may be longer than --seconds).
    let span = (((frames as f64) * in_rate as f64 / args.fps as f64).round() as usize).min(in_frames);
    let track = if voice_used { mix_voice(&args.out, &input[..span * in_ch], in_ch, in_rate, &vl, &vr)? } else { wav };
    capture::mux(&video_tmp, &track, &args.out)?;
    std::fs::remove_file(&video_tmp).ok();
    eprintln!("[render] wrote {} in {:.1}s", args.out.display(), started.elapsed().as_secs_f32());
    Ok(())
}

/// Write the voice stem (`<out>.sonify.wav`) and the music + voice mix (`<out>.mix.wav`, 32-bit
/// float stereo at the input rate). Returns the mix path.
fn mix_voice(out: &std::path::Path, input: &[f32], ch: usize, rate: u32, vl: &[f32], vr: &[f32]) -> anyhow::Result<PathBuf> {
    let spec = hound::WavSpec { channels: 2, sample_rate: rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let frames = input.len() / ch.max(1);
    let stem = out.with_extension("sonify.wav");
    let mut w = hound::WavWriter::create(&stem, spec)?;
    for (l, r) in vl.iter().zip(vr).take(frames) { w.write_sample(*l)?; w.write_sample(*r)?; }
    w.finalize()?;
    let mix: Vec<[f32; 2]> = (0..frames).map(|i| {
        let c = &input[i * ch..i * ch + ch];
        let (l, r) = (c[0], *c.get(1).unwrap_or(&c[0]));
        [l + vl.get(i).copied().unwrap_or(0.0), r + vr.get(i).copied().unwrap_or(0.0)]
    }).collect();
    let peak = mix.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
    let vpeak = vl.iter().chain(vr).fold(0f32, |m, v| m.max(v.abs()));
    // Never clip: if music + voice would exceed -0.1 dBFS, scale the whole mix down (no limiter
    // pumping, the balance is preserved).
    let scale = if peak > 0.989 { 0.989 / peak } else { 1.0 };
    let path = out.with_extension("mix.wav");
    let mut w = hound::WavWriter::create(&path, spec)?;
    for s in &mix { w.write_sample(s[0] * scale)?; w.write_sample(s[1] * scale)?; }
    w.finalize()?;
    eprintln!("[render] sonify voice peak {:.1} dBFS; mix peak {:.1} dBFS{} -> {}", 20.0 * vpeak.max(1e-9).log10(),
        20.0 * (peak * scale).max(1e-9).log10(), if scale < 1.0 { format!(" (scaled by {scale:.3} to avoid clipping)") } else { String::new() }, path.display());
    Ok(path)
}
