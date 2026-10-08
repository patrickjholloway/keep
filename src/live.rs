//! live.rs — audio for `keep run`.
//!
//! * `Track`: the precomputed analysis (same `audio::analyze` as the offline render), looked up
//!   and interpolated at an arbitrary playback time.
//! * `Player`: loops a WAV through the default output device (cpal). The playback position is
//!   the scene clock, so visuals stay locked to what you hear. Pause / seek / mute.
//!   Without an output device it falls back to a silent wall clock with the same transport.
//! * `MicAnalyzer`: live input via cpal, analyzed in fixed 60 Hz hops with a streaming version of
//!   the offline features (adaptive peak normalization instead of the 99th percentile).
use std::{
    collections::VecDeque,
    path::Path,
    sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex},
    time::Instant,
};

use anyhow::{bail, Context};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::audio::{self, Features};

/// Analysis rate for live lookup (features per second of audio).
pub const ANALYSIS_FPS: f32 = 60.0;

/// Wrap a time into [0, duration).
pub fn wrap_time(t: f64, duration: f64) -> f64 {
    if duration <= 0.0 { return 0.0; }
    let w = t.rem_euclid(duration);
    if w >= duration { 0.0 } else { w }
}

fn lerp(a: f32, b: f32, x: f32) -> f32 { a + (b - a) * x }

/// Precomputed per-frame features of a looping track.
pub struct Track {
    pub feats: Vec<Features>,
    pub fps: f32,
}

impl Track {
    pub fn duration(&self) -> f64 { self.feats.len() as f64 / self.fps as f64 }

    /// Features at playback time `t` (wrapped into the track), linearly interpolated between
    /// analysis frames. `beat_phase` interpolates across its 1→0 wrap.
    pub fn at(&self, t: f64) -> Features {
        let n = self.feats.len();
        if n == 0 { return Features::default(); }
        let x = wrap_time(t, self.duration()) * self.fps as f64;
        let i = (x.floor() as usize).min(n - 1);
        let fr = (x - i as f64) as f32;
        let (a, b) = (self.feats[i], self.feats[(i + 1) % n]);
        let mut bp1 = b.beat_phase;
        if bp1 < a.beat_phase - 0.5 { bp1 += 1.0; }
        Features {
            bass: lerp(a.bass, b.bass, fr),
            mid: lerp(a.mid, b.mid, fr),
            high: lerp(a.high, b.high, fr),
            onset: lerp(a.onset, b.onset, fr),
            rms: lerp(a.rms, b.rms, fr),
            beat_phase: lerp(a.beat_phase, bp1, fr).rem_euclid(1.0),
        }
    }
}

/// Transport shared between the audio callback and the UI thread.
struct Transport {
    /// Position in source frames.
    pos: f64,
    /// When `pos` was last advanced (for smooth extrapolation between callbacks).
    stamp: Instant,
    paused: bool,
}

pub struct Player {
    shared: Arc<Mutex<Transport>>,
    muted: Arc<AtomicBool>,
    /// Source frames (per channel) and source rate.
    frames: usize,
    rate: f64,
    /// Max extrapolation past the last callback (one buffer is plenty).
    max_ahead: f64,
    _stream: Option<cpal::Stream>,
}

impl Player {
    /// Load `wav` and start looping it. Falls back to a silent clock if no output device works.
    pub fn start(wav: &Path) -> anyhow::Result<Player> {
        let (samples, rate) = read_stereo(wav)?;
        let frames = samples.len() / 2;
        anyhow::ensure!(frames > 0, "{}: empty audio", wav.display());
        let shared = Arc::new(Mutex::new(Transport { pos: 0.0, stamp: Instant::now(), paused: false }));
        let muted = Arc::new(AtomicBool::new(false));
        let mut p = Player { shared, muted, frames, rate: rate as f64, max_ahead: 0.05, _stream: None };
        match open_output(Arc::new(samples), rate, p.shared.clone(), p.muted.clone()) {
            Ok((stream, buf_secs)) => { p.max_ahead = buf_secs.max(0.01) * 2.0; p._stream = Some(stream); }
            Err(e) => {
                eprintln!("[run] no audio output ({e:#}); visuals follow a silent clock");
                p.max_ahead = f64::INFINITY; // pure wall clock
            }
        }
        Ok(p)
    }

    pub fn duration(&self) -> f64 { self.frames as f64 / self.rate }

    /// Current playback time in seconds, in [0, duration).
    pub fn time(&self) -> f64 {
        let s = self.shared.lock().unwrap();
        let mut secs = s.pos / self.rate;
        if !s.paused { secs += s.stamp.elapsed().as_secs_f64().min(self.max_ahead); }
        wrap_time(secs, self.duration())
    }

    pub fn toggle_pause(&self) -> bool {
        let now = self.time();
        let mut s = self.shared.lock().unwrap();
        s.paused = !s.paused;
        s.pos = now * self.rate;
        s.stamp = Instant::now();
        s.paused
    }

    /// Seek by `delta` seconds, wrapping around the loop.
    pub fn seek(&self, delta: f64) {
        let t = wrap_time(self.time() + delta, self.duration());
        let mut s = self.shared.lock().unwrap();
        s.pos = t * self.rate;
        s.stamp = Instant::now();
    }

    pub fn toggle_mute(&self) -> bool { !self.muted.fetch_xor(true, Ordering::Relaxed) }
}

/// Read a WAV as interleaved stereo f32 (mono is duplicated, >2 channels keep the first two).
fn read_stereo(wav: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut rd = hound::WavReader::open(wav).with_context(|| format!("open {}", wav.display()))?;
    let s = rd.spec();
    let ch = s.channels as usize;
    let inter: Vec<f32> = match s.sample_format {
        hound::SampleFormat::Float => rd.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (s.bits_per_sample - 1)) as f32;
            rd.samples::<i32>().map(|x| x.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
    };
    let out = inter.chunks(ch).flat_map(|c| [c[0], *c.get(1).unwrap_or(&c[0])]).collect();
    Ok((out, s.sample_rate))
}

fn open_output(samples: Arc<Vec<f32>>, src_rate: u32, shared: Arc<Mutex<Transport>>, muted: Arc<AtomicBool>)
    -> anyhow::Result<(cpal::Stream, f64)> {
    let dev = cpal::default_host().default_output_device().context("no default output device")?;
    let cfg = dev.default_output_config()?;
    if cfg.sample_format() != cpal::SampleFormat::F32 { bail!("output format {:?} unsupported", cfg.sample_format()); }
    let out_rate = cfg.sample_rate().0 as f64;
    let chans = cfg.channels() as usize;
    let step = src_rate as f64 / out_rate;
    let frames = samples.len() / 2;
    let stream = dev.build_output_stream(
        &cfg.into(),
        move |data: &mut [f32], _| {
            let mut s = shared.lock().unwrap();
            if s.paused { data.fill(0.0); return; }
            let mute = muted.load(Ordering::Relaxed);
            for frame in data.chunks_mut(chans) {
                let i = s.pos as usize % frames;
                let fr = (s.pos - s.pos.floor()) as f32;
                let j = (i + 1) % frames;
                let l = lerp(samples[2 * i], samples[2 * j], fr);
                let r = lerp(samples[2 * i + 1], samples[2 * j + 1], fr);
                for (c, o) in frame.iter_mut().enumerate() {
                    *o = if mute { 0.0 } else if c % 2 == 0 { l } else { r };
                }
                s.pos = wrap_time(s.pos + step, frames as f64);
            }
            s.stamp = Instant::now();
        },
        |e| eprintln!("[audio] output error: {e}"),
        None,
    )?;
    stream.play()?;
    // Rough buffer length for clock extrapolation: assume ~1024 frames if unknown.
    Ok((stream, 1024.0 / out_rate))
}

/// Ensure the WAV exists (synthesizing the default track if not), then analyze it for lookup.
pub fn prepare(wav: &Path, synth_seconds: f32) -> anyhow::Result<Track> {
    if !wav.exists() {
        if let Some(d) = wav.parent() { if !d.as_os_str().is_empty() { std::fs::create_dir_all(d)?; } }
        let spec = audio::TrackSpec { seconds: synth_seconds, ..Default::default() };
        eprintln!("[run] synthesizing {synth_seconds:.0}s track -> {}", wav.display());
        audio::synthesize(&spec, wav)?;
    }
    let feats = audio::analyze(wav, ANALYSIS_FPS, None)?;
    Ok(Track { feats, fps: ANALYSIS_FPS })
}

// ───────────────────────────── live microphone ─────────────────────────────

/// Streaming analyzer state (pure, testable without a device).
pub struct LiveFeatures {
    sr: f32,
    hop: usize,
    fft: Arc<dyn realfft::RealToComplex<f32>>,
    hann: Vec<f32>,
    prev_log: Vec<f32>,
    /// Adaptive peaks for bass, mid, high, flux, low-flux, rms.
    peaks: [f32; 6],
    onset_hist: VecDeque<f32>,
    hops: u64,
    beat: Option<(f32, f32)>,
    pub state: Features,
}

impl LiveFeatures {
    pub fn new(sr: f32) -> Self {
        let fft = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(audio::WIN);
        let w = audio::WIN;
        LiveFeatures {
            sr, hop: (sr / ANALYSIS_FPS).round() as usize, fft,
            hann: (0..w).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / w as f32).cos()).collect(),
            prev_log: vec![0.0; w / 2 + 1], peaks: [1e-6; 6], onset_hist: VecDeque::new(), hops: 0, beat: None,
            state: Features::default(),
        }
    }

    pub fn hop(&self) -> usize { self.hop }

    /// Analyze one window (the latest WIN samples) and advance the state by one hop.
    pub fn push_window(&mut self, win: &[f32]) {
        let w = audio::WIN;
        let mut buf = self.fft.make_input_vec();
        let mut spec = self.fft.make_output_vec();
        let mut sumsq = 0.0;
        for (j, b) in buf.iter_mut().enumerate() {
            let x = win.get(j).copied().unwrap_or(0.0);
            sumsq += x * x;
            *b = x * self.hann[j];
        }
        self.fft.process(&mut buf, &mut spec).expect("fft length");
        let mag: Vec<f32> = spec.iter().map(|c| c.norm()).collect();
        let bin_hz = self.sr / w as f32;
        let band = |lo: f32, hi: f32| ((lo / bin_hz).ceil() as usize).max(1)..((hi / bin_hz) as usize).min(w / 2);
        let energy = |r: std::ops::Range<usize>| { let n = r.len().max(1) as f32; (r.map(|k| mag[k] * mag[k]).sum::<f32>() / n).sqrt() };
        let hi = band(2000.0, 12000.0);
        let low = band(20.0, 200.0);
        let (mut flux, mut lflux) = (0.0, 0.0);
        for k in 1..hi.end {
            let lg = (1.0 + 10.0 * mag[k]).ln();
            let d = (lg - self.prev_log[k]).max(0.0);
            flux += d;
            if low.contains(&k) { lflux += d; }
            self.prev_log[k] = lg;
        }
        let raw = [energy(band(20.0, 150.0)), energy(band(150.0, 2000.0)), energy(hi), flux, lflux, (sumsq / w as f32).sqrt()];
        // Adaptive normalization: peak follows up instantly, decays over ~10 s.
        let decay = (-1.0 / (10.0 * ANALYSIS_FPS)).exp();
        let mut n = [0f32; 6];
        for k in 0..6 {
            self.peaks[k] = raw[k].max(self.peaks[k] * decay).max(1e-6);
            n[k] = (raw[k] / self.peaks[k]).clamp(0.0, 1.0);
        }
        let onset = (0.65 * n[4] + 0.35 * n[3]).clamp(0.0, 1.0);

        let dt = 1.0 / ANALYSIS_FPS;
        let kf = |tau: f32| 1.0 - (-dt / tau).exp();
        let (att, rel) = (kf(0.015), kf(0.12));
        let follow = |p: f32, x: f32| p + if x > p { att } else { rel } * (x - p);
        let s = &mut self.state;
        s.bass = follow(s.bass, n[0]);
        s.mid = follow(s.mid, n[1]);
        s.high = follow(s.high, n[2]);
        s.onset = onset.max(s.onset * (-dt / 0.08).exp());
        s.rms += kf(0.08) * (n[5] - s.rms);

        // Tempo: re-estimate once a second over the last 8 s of onsets.
        self.onset_hist.push_back(onset);
        if self.onset_hist.len() > (8.0 * ANALYSIS_FPS) as usize { self.onset_hist.pop_front(); }
        self.hops += 1;
        if self.hops % ANALYSIS_FPS as u64 == 0 {
            let h: Vec<f32> = self.onset_hist.iter().copied().collect();
            let base = self.hops as f32 - h.len() as f32;
            self.beat = audio::estimate_beat(&h, ANALYSIS_FPS).map(|(p, off)| (p, off + base));
        }
        s.beat_phase = self.beat.map_or(0.0, |(p, off)| ((self.hops as f32 - 1.0 - off) / p).rem_euclid(1.0));
    }
}

/// Microphone capture + `LiveFeatures`, pumped from the render loop.
pub struct MicAnalyzer {
    ring: Arc<Mutex<VecDeque<f32>>>,
    window: VecDeque<f32>,
    feats: LiveFeatures,
    _stream: cpal::Stream,
}

impl MicAnalyzer {
    pub fn start() -> anyhow::Result<MicAnalyzer> {
        let dev = cpal::default_host().default_input_device().context("no default input device")?;
        let cfg = dev.default_input_config()?;
        if cfg.sample_format() != cpal::SampleFormat::F32 { bail!("input format {:?} unsupported", cfg.sample_format()); }
        let sr = cfg.sample_rate().0 as f32;
        let chans = cfg.channels() as usize;
        let ring = Arc::new(Mutex::new(VecDeque::new()));
        let r2 = ring.clone();
        let stream = dev.build_input_stream(
            &cfg.into(),
            move |data: &[f32], _| {
                let mut q = r2.lock().unwrap();
                q.extend(data.chunks(chans).map(|c| c.iter().sum::<f32>() / chans as f32));
                let cap = sr as usize * 2;
                while q.len() > cap { q.pop_front(); }
            },
            |e| eprintln!("[audio] input error: {e}"),
            None,
        )?;
        stream.play()?;
        eprintln!("[run] listening on {}", dev.name().unwrap_or_default());
        Ok(MicAnalyzer { ring, window: VecDeque::from(vec![0.0; audio::WIN]), feats: LiveFeatures::new(sr), _stream: stream })
    }

    /// Consume captured audio in whole hops and return the latest features.
    pub fn poll(&mut self) -> Features {
        let mut q = self.ring.lock().unwrap();
        let hop = self.feats.hop();
        while q.len() >= hop {
            self.window.extend(q.drain(..hop));
            while self.window.len() > audio::WIN { self.window.pop_front(); }
            let w: Vec<f32> = self.window.iter().copied().collect();
            self.feats.push_window(&w);
        }
        self.feats.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: f32, bp: f32) -> Features { Features { bass: x, mid: x, high: x, onset: x, rms: x, beat_phase: bp } }

    #[test]
    fn seek_wraps() {
        assert_eq!(wrap_time(3.0, 10.0), 3.0);
        assert_eq!(wrap_time(12.0, 10.0), 2.0);
        assert!((wrap_time(-5.0 + 2.0, 10.0) - 7.0).abs() < 1e-12); // seek back past 0
        assert_eq!(wrap_time(10.0, 10.0), 0.0);
        assert_eq!(wrap_time(5.0, 0.0), 0.0);
    }

    #[test]
    fn lookup_interpolates_and_wraps() {
        let t = Track { feats: vec![f(0.0, 0.0), f(1.0, 0.5), f(0.5, 0.9), f(0.2, 0.1)], fps: 2.0 };
        assert_eq!(t.duration(), 2.0);
        assert_eq!(t.at(0.5), f(1.0, 0.5)); // exact frame
        let m = t.at(0.25);
        assert!((m.bass - 0.5).abs() < 1e-6 && (m.beat_phase - 0.25).abs() < 1e-6);
        // 0.9 → 0.1 crosses the beat: halfway is 0.0 (mod 1), not 0.5.
        let w = t.at(1.25);
        assert!(w.beat_phase < 1e-5 || w.beat_phase > 1.0 - 1e-5, "{}", w.beat_phase);
        // Past the end wraps to the start; the last frame interpolates toward frame 0.
        assert_eq!(t.at(2.5), t.at(0.5));
        assert!((t.at(1.75).bass - 0.1).abs() < 1e-6);
    }

    #[test]
    fn live_features_respond_to_a_kick() {
        let sr = 48_000.0;
        let mut lf = LiveFeatures::new(sr);
        let hop = lf.hop();
        let mut sig = vec![0.0f32; sr as usize * 3];
        for beat in 0..6 { // 60 Hz thump every 0.5 s
            let s0 = (beat as f32 * 0.5 * sr) as usize;
            for i in 0..4800 { sig[s0 + i] = (std::f32::consts::TAU * 60.0 * i as f32 / sr).sin() * (-(i as f32) / 1200.0).exp(); }
        }
        let mut max_onset: f32 = 0.0;
        let mut end = audio::WIN;
        while end <= sig.len() {
            lf.push_window(&sig[end - audio::WIN..end]);
            max_onset = max_onset.max(lf.state.onset);
            end += hop;
        }
        assert!(max_onset > 0.5, "onset {max_onset}");
        assert!(lf.state.bass >= 0.0 && lf.state.bass <= 1.0);
    }
}
