//! capture.rs — stream raw RGBA frames into an ffmpeg child process (stdin pipe), then mux
//! the WAV in. ffmpeg comes from mise (`conda:ffmpeg`); run keep under `mise exec --`.
use std::{path::PathBuf, process::Child};

use anyhow::Context;

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
        use std::process::{Command, Stdio};
        if let Some(dir) = spec.out.parent() {
            if !dir.as_os_str().is_empty() { std::fs::create_dir_all(dir)?; }
        }
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error", "-y"])
            // Input 0: raw frames on stdin, exactly as the GPU read them back.
            .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-s", &format!("{}x{}", spec.width, spec.height)])
            .args(["-r", &spec.fps.to_string(), "-i", "-"]);
        if let Some(a) = &spec.audio {
            cmd.arg("-i").arg(a);
        }
        // yuv420p = what every player expects; +faststart moves the index to the front.
        cmd.args(["-c:v", "libx264", "-preset", "medium", "-crf", &spec.crf.to_string(), "-pix_fmt", "yuv420p"])
            .args(["-movflags", "+faststart"]);
        if spec.audio.is_some() {
            // Map video from stdin and audio from the WAV; -shortest trims to the video length.
            cmd.args(["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "256k", "-shortest"]);
        }
        cmd.arg(&spec.out).stdin(Stdio::piped());
        let child = cmd.spawn().context("spawning ffmpeg (run keep via `mise exec --`)")?;
        Ok(Encoder { spec, child, frames: 0 })
    }

    /// Write one tightly packed RGBA8 frame (width*height*4 bytes).
    pub fn push_frame(&mut self, rgba: &[u8]) -> anyhow::Result<()> {
        use std::io::Write;
        let want = (self.spec.width * self.spec.height * 4) as usize;
        anyhow::ensure!(rgba.len() == want, "frame is {} bytes, expected {want}", rgba.len());
        self.child.stdin.as_mut().context("ffmpeg stdin closed")?.write_all(rgba).context("writing frame to ffmpeg")?;
        self.frames += 1;
        Ok(())
    }

    /// Close stdin, wait for ffmpeg, error if it failed. Returns the output path.
    pub fn finish(mut self) -> anyhow::Result<PathBuf> {
        drop(self.child.stdin.take()); // EOF tells ffmpeg the stream is over.
        let status = self.child.wait()?;
        anyhow::ensure!(status.success(), "ffmpeg exited with {status} after {} frames", self.frames);
        Ok(self.spec.out)
    }
}

/// True if `ffmpeg` is runnable on PATH.
pub fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

