//! capture.rs — stream raw RGBA frames into an ffmpeg child process (stdin pipe), then mux
//! the WAV in. ffmpeg comes from mise (`conda:ffmpeg`); run keep under `mise exec --`.
use std::{path::PathBuf, process::Child};

#[derive(Clone, Debug)]
pub struct EncodeSpec {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub out: PathBuf,
    /// Audio to mux in (AAC); None = silent video.
    pub audio: Option<PathBuf>,
    /// x264 CRF (lower = better; 16 is visually lossless-ish for particles).
    pub crf: u32,
}

pub struct Encoder {
    spec: EncodeSpec,
    child: Child,
    frames: u64,
}

impl Encoder {
    /// Spawn `ffmpeg -f rawvideo -pix_fmt rgba -s WxH -r fps -i - [-i audio] -c:v libx264 -pix_fmt yuv420p ... out`.
    pub fn start(spec: EncodeSpec) -> anyhow::Result<Encoder> {
        let _ = spec;
        todo!("spawn ffmpeg")
    }

    /// Write one tightly packed RGBA8 frame (width*height*4 bytes).
    pub fn push_frame(&mut self, rgba: &[u8]) -> anyhow::Result<()> {
        let _ = (rgba, &self.spec, &mut self.child, self.frames);
        todo!("write to stdin")
    }

    /// Close stdin, wait for ffmpeg, error if it failed. Returns the output path.
    pub fn finish(self) -> anyhow::Result<PathBuf> {
        todo!("close + wait")
    }
}

/// True if `ffmpeg` is runnable on PATH.
pub fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

