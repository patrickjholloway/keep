//! offline.rs — `keep render`: deterministic headless render to video.
//! 1. ensure audio (synthesize if --audio path missing)   audio::synthesize
//! 2. per-frame features                                   audio::analyze
//! 3. for frame i: t = i/fps; desc = script.update(t, f[i]); params = desc.to_params(..)
//! 4. renderer.render_offscreen -> encoder.push_frame;     capture::Encoder (muxes audio)
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct RenderArgs {
    pub script: PathBuf,
    pub audio: PathBuf,
    pub seconds: f32,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub out: PathBuf,
    pub particles: u32,
}

pub fn render(args: &RenderArgs) -> anyhow::Result<()> {
    let _ = args;
    todo!("offline render loop")
}
