//! audio.rs — original procedural music + feature extraction. No samples, no copyrighted audio.
//!
//! `synthesize` writes a mono/stereo 48 kHz WAV of a generated track (kick, bass, pads, hats
//! built from oscillators + noise, seeded so it is reproducible).
//! `analyze` reads a WAV and produces one `Features` per video frame via STFT (realfft).
use std::path::Path;

/// Audio features for one video frame. All roughly normalized to [0, 1].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Features {
    /// Energy 20–150 Hz.
    pub bass: f32,
    /// Energy 150–2000 Hz.
    pub mid: f32,
    /// Energy 2–12 kHz.
    pub high: f32,
    /// Spectral-flux onset strength (spikes on hits), decays quickly.
    pub onset: f32,
    /// RMS envelope (smoothed loudness).
    pub rms: f32,
    /// Position within the current beat, 0 at the beat, → 1 just before the next.
    pub beat_phase: f32,
}

impl Features {
    /// Packed for SceneParams.audio: (bass, mid, high, onset).
    pub fn as_vec4(&self) -> glam::Vec4 {
        glam::Vec4::new(self.bass, self.mid, self.high, self.onset)
    }
}

/// Parameters of the generated track.
#[derive(Clone, Copy, Debug)]
pub struct TrackSpec {
    pub seconds: f32,
    pub bpm: f32,
    pub sample_rate: u32,
    pub seed: u64,
}

impl Default for TrackSpec {
    fn default() -> Self { TrackSpec { seconds: 60.0, bpm: 122.0, sample_rate: 48_000, seed: 7 } }
}

/// Generate an original track and write it as 16-bit stereo WAV at `out`.
pub fn synthesize(spec: &TrackSpec, out: &Path) -> anyhow::Result<()> {
    let _ = (spec, out);
    todo!("procedural synthesis")
}

/// Analyze `wav` into `ceil(duration * fps)` frames of features. `bpm` (if known) drives beat_phase.
pub fn analyze(wav: &Path, fps: f32, bpm: Option<f32>) -> anyhow::Result<Vec<Features>> {
    let _ = (wav, fps, bpm);
    todo!("STFT feature extraction")
}
