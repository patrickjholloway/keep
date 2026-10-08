//! audio.rs — original procedural music + feature extraction. No samples, no copyrighted audio.
//!
//! `synthesize` writes a mono/stereo 48 kHz WAV of a generated track (kick, bass, pads, hats
//! built from oscillators + noise, seeded so it is reproducible).
//! `analyze` reads a WAV and produces one `Features` per video frame via STFT (realfft).
//! Run `keep render --audio ...` to hear it; see the section map just below `TrackSpec`.
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

// ───────────────────────────── synthesis ─────────────────────────────
//
// Song form (in 4-beat bars), repeating every 36 bars:
//   0..4   intro   pads + soft arp + sparse hats
//   4..12  groove  + kick, offbeat sub bass, 8th hats
//   12..20 build   kick thins out, snare roll accelerates, noise riser, filter opens
//   20..36 drop    everything, rolling bass, 16th hats, crash on the downbeat
// Every voice is a closed-form function of "time since its last trigger", which keeps the
// code stateless (and deterministic) apart from noise and the one-pole filters.

const INTRO: u32 = 4;
const GROOVE: u32 = 12;
const BUILD: u32 = 20;
const FORM: u32 = 36;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Section { Intro, Groove, Build, Drop }

fn section_of(bar: u32) -> Section {
    match bar % FORM {
        b if b < INTRO => Section::Intro,
        b if b < GROOVE => Section::Groove,
        b if b < BUILD => Section::Build,
        _ => Section::Drop,
    }
}

/// Tiny deterministic PRNG (xorshift64*). We avoid `rand` so the stream is stable forever.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1, 1).
    fn noise(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0 }
}

/// Stateless hash → [0,1) for per-step "humanization" (velocity jitter etc.).
fn hash01(seed: u64, n: u64) -> f32 {
    let mut r = Rng::new(seed ^ n.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    r.next_u64();
    (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32
}

fn midi_hz(n: f32) -> f32 { 440.0 * 2f32.powf((n - 69.0) / 12.0) }

/// One-pole low-pass: y += a (x - y). `a` from cutoff via the usual exp mapping.
#[derive(Default, Clone, Copy)]
struct OnePole { y: f32 }
impl OnePole {
    fn lp(&mut self, x: f32, cutoff_hz: f32, sr: f32) -> f32 {
        let a = 1.0 - (-std::f32::consts::TAU * cutoff_hz / sr).exp();
        self.y += a * (x - self.y);
        self.y
    }
}

/// Chord progressions: (root midi note, triad intervals). One is chosen by the seed.
const PROGRESSIONS: [[(i32, [i32; 3]); 4]; 3] = [
    // Am – F – C – G
    [(57, [0, 3, 7]), (53, [0, 4, 7]), (48, [0, 4, 7]), (55, [0, 4, 7])],
    // Dm – Bb – F – C
    [(50, [0, 3, 7]), (46, [0, 4, 7]), (53, [0, 4, 7]), (48, [0, 4, 7])],
    // Em – C – G – D
    [(52, [0, 3, 7]), (48, [0, 4, 7]), (55, [0, 4, 7]), (50, [0, 4, 7])],
];

/// Generate an original track and write it as 16-bit stereo WAV at `out`.
pub fn synthesize(spec: &TrackSpec, out: &Path) -> anyhow::Result<()> {
    let (l, r) = render(spec);
    let wav = hound::WavSpec {
        channels: 2,
        sample_rate: spec.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(out, wav)?;
    for (a, b) in l.iter().zip(&r) {
        w.write_sample((a.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        w.write_sample((b.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

/// Render the track to two f32 channels in [-1, 1].
fn render(spec: &TrackSpec) -> (Vec<f32>, Vec<f32>) {
    use std::f32::consts::TAU;
    let sr = spec.sample_rate as f32;
    let n = (spec.seconds * sr).ceil() as usize;
    let spb = 60.0 / spec.bpm; // seconds per beat
    let prog = &PROGRESSIONS[(spec.seed % 3) as usize];
    // Seeded 8-step arp pattern over chord-tone indices (0..7 = 3 tones × octaves + top root).
    let arp: Vec<usize> = (0..8).map(|i| {
        let base = [0usize, 1, 2, 3, 4, 5, 6, 4][i];
        if hash01(spec.seed, 100 + i as u64) < 0.35 { (base + 2) % 7 } else { base }
    }).collect();

    let mut rng = Rng::new(spec.seed);
    let (mut pad_l, mut pad_r) = (OnePole::default(), OnePole::default());
    let (mut hat_lp, mut riser_lp) = (OnePole::default(), OnePole::default());
    let mut out_l = vec![0f32; n];
    let mut out_r = vec![0f32; n];

    for i in 0..n {
        let t = i as f32 / sr;
        let beat_f = t / spb;
        let beat = beat_f.floor() as u32;
        let tb = (beat_f - beat as f32) * spb; // seconds since the beat
        let bar = beat / 4;
        let bar_pos = (beat_f / 4.0).fract(); // 0..1 within the bar
        let sec = section_of(bar);
        let bar_in_form = bar % FORM;
        let (root, triad) = prog[(bar % 4) as usize];
        let chord: [f32; 3] = triad.map(|iv| (root + iv) as f32);

        // ── kick: sine with a fast downward pitch sweep (150 → 45 Hz) + tiny click.
        let kick_on = match sec {
            Section::Intro => false,
            Section::Build => bar_in_form < BUILD - 2, // drop out for the last 2 bars: tension
            _ => true,
        };
        let mut kick = 0.0;
        let mut duck = 1.0; // sidechain gain applied to bass/pads
        if kick_on {
            // Phase is the integral of f(t) = 45 + 110 e^{-30t}.
            let ph = 45.0 * tb + 110.0 * (1.0 - (-30.0 * tb).exp()) / 30.0;
            kick = (TAU * ph).sin() * (-tb * 6.0).exp() * 0.9 + rng.noise() * (-tb * 400.0).exp() * 0.15;
            duck = 1.0 - 0.7 * (-tb * 9.0).exp();
        }

        // ── sub bass: offbeat 8ths (groove/build), rolling 16ths minus the downbeat (drop).
        let mut bass = 0.0;
        if sec != Section::Intro {
            // (seconds since this bass note started) or None when the bass is silent.
            let sixteenth = spb / 4.0;
            let since = match sec {
                Section::Drop => {
                    let step = (tb / sixteenth) as u32; // 0..3 within the beat
                    (step != 0).then(|| tb - step as f32 * sixteenth)
                }
                _ => (tb >= spb / 2.0).then(|| tb - spb / 2.0),
            };
            if let Some(ts) = since {
                let f = midi_hz((root - 24) as f32);
                let env = (1.0 - (-ts * 300.0).exp()) * (-ts * 7.0).exp();
                bass = ((TAU * f * t).sin() * 1.8).tanh() * env * 0.55;
            }
        }

        // ── pads: chord tones as band-limited saws (4 harmonics), detuned L/R, one-pole LPF.
        let pad_env = (bar_pos / 0.15).min(1.0) * (1.0 - ((bar_pos - 0.85) / 0.15).clamp(0.0, 1.0));
        let (mut pl, mut pr) = (0.0, 0.0);
        for (k, &note) in chord.iter().chain(std::iter::once(&(root as f32 - 12.0))).enumerate() {
            let f = midi_hz(note);
            for h in 1..=4 {
                let hf = h as f32;
                pl += (TAU * f * 1.003 * hf * t + k as f32).sin() / hf;
                pr += (TAU * f * 0.997 * hf * t + 2.0 * k as f32).sin() / hf;
            }
        }
        let cutoff = match sec {
            Section::Intro => 700.0,
            Section::Groove => 1100.0,
            Section::Build => 1100.0 + 4000.0 * ((bar_in_form - GROOVE) as f32 + bar_pos) / 8.0,
            Section::Drop => 3500.0,
        };
        let pad_gain = 0.06 * pad_env;
        let pl = pad_l.lp(pl, cutoff, sr) * pad_gain;
        let pr = pad_r.lp(pr, cutoff, sr) * pad_gain;

        // ── arp: 16th-note plucks walking the seeded pattern over the chord, ping-ponged.
        let step16 = (beat_f * 4.0).floor() as u64;
        let ts = (beat_f * 4.0).fract() * spb / 4.0;
        let tones = [chord[0], chord[1], chord[2], chord[0] + 12.0, chord[1] + 12.0, chord[2] + 12.0, chord[0] + 24.0];
        let note = tones[arp[(step16 % 8) as usize]] + 12.0;
        let f = midi_hz(note);
        let vel = 0.8 + 0.2 * hash01(spec.seed, step16);
        let arp_level = match sec {
            Section::Intro => 0.10,
            Section::Groove => 0.08,
            Section::Build => 0.08 + 0.06 * ((bar_in_form - GROOVE) as f32 / 8.0),
            Section::Drop => 0.14,
        };
        let pluck = ((TAU * f * t).sin() + 0.3 * (2.0 * TAU * f * t).sin() + 0.1 * (3.0 * TAU * f * t).sin())
            * (-ts * 14.0).exp() * (1.0 - (-ts * 2000.0).exp()) * vel * arp_level;
        let pan = if step16 % 2 == 0 { 0.3 } else { 0.7 };

        // ── hats: high-passed noise (noise minus its low-pass), short decay.
        let hp = { let x = rng.noise(); x - hat_lp.lp(x, 6000.0, sr) };
        let eighth = spb / 2.0;
        let th8 = tb % eighth;
        let off8 = tb >= eighth;
        let hat_env = match sec {
            Section::Intro => if off8 && beat % 2 == 1 { (-th8 * 50.0).exp() * 0.5 } else { 0.0 },
            Section::Groove | Section::Build => if off8 { (-th8 * 45.0).exp() } else { 0.0 },
            Section::Drop => (-ts * 60.0).exp() * if (step16 % 2) == 1 { 1.0 } else { 0.55 },
        };
        let hat = hp * hat_env * 0.22 * (0.8 + 0.2 * hash01(spec.seed ^ 77, step16));

        // ── build FX: accelerating snare roll + noise riser through an opening LPF.
        let mut fx = 0.0;
        if sec == Section::Build {
            let p = ((bar_in_form - GROOVE) as f32 + bar_pos) / 8.0; // 0..1 across the build
            let div = if p < 0.5 { 1.0 } else if p < 0.75 { 2.0 } else { 4.0 }; // hits per beat
            let tr = (beat_f * div).fract() * spb / div;
            let snare = (rng.noise() * 0.7 + (TAU * 190.0 * tr).sin() * 0.4) * (-tr * 30.0).exp();
            let riser = riser_lp.lp(rng.noise(), 300.0 + 8000.0 * p * p, sr);
            fx = snare * 0.25 * (0.3 + 0.7 * p) + riser * 0.35 * p * p;
        }
        // Crash on the first downbeat of the drop: long noise tail.
        if sec == Section::Drop && bar_in_form == BUILD {
            let tc = bar_pos * 4.0 * spb;
            fx += rng.noise() * (-tc * 2.5).exp() * 0.2;
        }

        // ── mix: duck the sustained parts under the kick, soft-clip, fade in/out.
        let mono = kick + bass * duck + hat + fx;
        let mut l = mono + pl * duck + pluck * (1.0 - pan) * 2.0;
        let mut r = mono + pr * duck + pluck * pan * 2.0;
        let fade = (t / 0.5).min(1.0) * ((spec.seconds - t) / 1.5).clamp(0.0, 1.0);
        l = (l * 1.1).tanh() * fade;
        r = (r * 1.1).tanh() * fade;
        out_l[i] = l;
        out_r[i] = r;
    }

    // Normalize to a -1 dBFS peak so every seed lands at the same loudness.
    let peak = out_l.iter().chain(&out_r).fold(1e-9f32, |m, x| m.max(x.abs()));
    let g = 0.89 / peak;
    out_l.iter_mut().chain(out_r.iter_mut()).for_each(|x| *x *= g);
    (out_l, out_r)
}

// ───────────────────────────── analysis ─────────────────────────────
//
// One analysis window (2048 samples, Hann) is centred on each video frame's timestamp.
// From its magnitude spectrum we take band energies and spectral flux (sum of positive
// log-magnitude increases vs the previous frame = "something new just started").
// Raw values are normalized by their 99th percentile over the whole track, then smoothed
// with an attack/release envelope follower so visuals react fast and relax gracefully.

const WIN: usize = 2048;

/// Read any 16/24/32-bit int or float WAV and mix it to mono f32.
fn read_mono(wav: &Path) -> anyhow::Result<(Vec<f32>, f32)> {
    let mut rd = hound::WavReader::open(wav)?;
    let s = rd.spec();
    let ch = s.channels as usize;
    let interleaved: Vec<f32> = match s.sample_format {
        hound::SampleFormat::Float => rd.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (s.bits_per_sample - 1)) as f32;
            rd.samples::<i32>().map(|x| x.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
    };
    let mono = interleaved.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect();
    Ok((mono, s.sample_rate as f32))
}

/// Divide by the 99th percentile and clamp to [0, 1].
fn normalize(v: &mut [f32]) {
    let mut s: Vec<f32> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = s.get(((s.len() as f32 * 0.99) as usize).min(s.len().saturating_sub(1))).copied().unwrap_or(1.0);
    let p = if p > 1e-9 { p } else { 1.0 };
    v.iter_mut().for_each(|x| *x = (*x / p).clamp(0.0, 1.0));
}

/// Raw (unsmoothed, normalized) per-frame features: (bass, mid, high, onset, rms).
fn raw_features(mono: &[f32], sr: f32, fps: f32) -> [Vec<f32>; 5] {
    let frames = ((mono.len() as f32 / sr) * fps).ceil() as usize;
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(WIN);
    let hann: Vec<f32> = (0..WIN).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / WIN as f32).cos()).collect();
    let mut buf = fft.make_input_vec();
    let mut spec = fft.make_output_vec();
    let bin_hz = sr / WIN as f32;
    let band = |lo: f32, hi: f32| ((lo / bin_hz).ceil() as usize).max(1)..((hi / bin_hz) as usize).min(WIN / 2);
    let (b_bass, b_mid, b_high) = (band(20.0, 150.0), band(150.0, 2000.0), band(2000.0, 12000.0));
    let b_low_flux = band(20.0, 200.0);

    let mut out: [Vec<f32>; 5] = Default::default();
    let mut low_flux = Vec::with_capacity(frames);
    let mut prev_log = vec![0f32; WIN / 2 + 1];
    for f in 0..frames {
        let centre = (f as f32 / fps * sr) as isize;
        let start = centre - WIN as isize / 2;
        let mut sumsq = 0.0;
        for (j, b) in buf.iter_mut().enumerate() {
            let x = mono.get((start + j as isize) as usize).copied().filter(|_| start + j as isize >= 0).unwrap_or(0.0);
            sumsq += x * x;
            *b = x * hann[j];
        }
        fft.process(&mut buf, &mut spec).expect("fft length");
        let mag: Vec<f32> = spec.iter().map(|c| c.norm()).collect();
        let energy = |r: std::ops::Range<usize>| {
            let n = r.len().max(1) as f32;
            (r.map(|k| mag[k] * mag[k]).sum::<f32>() / n).sqrt()
        };
        out[0].push(energy(b_bass.clone()));
        out[1].push(energy(b_mid.clone()));
        out[2].push(energy(b_high.clone()));
        // Spectral flux on log magnitudes (log makes it loudness-independent-ish).
        let (mut flux, mut lflux) = (0.0, 0.0);
        for k in 1..b_high.end {
            let lg = (1.0 + 10.0 * mag[k]).ln();
            let d = (lg - prev_log[k]).max(0.0);
            flux += d;
            if b_low_flux.contains(&k) { lflux += d; }
            prev_log[k] = lg;
        }
        out[3].push(flux);
        low_flux.push(lflux);
        out[4].push((sumsq / WIN as f32).sqrt());
    }
    for v in out.iter_mut() { normalize(v); }
    normalize(&mut low_flux);
    // Kicks drive the visuals, so the onset leans on low-frequency flux.
    for (o, l) in out[3].iter_mut().zip(&low_flux) { *o = (0.65 * l + 0.35 * *o).clamp(0.0, 1.0); }
    out
}

/// Estimate (period in frames, phase offset in frames) from the onset curve by autocorrelation.
fn estimate_beat(onset: &[f32], fps: f32) -> Option<(f32, f32)> {
    let lag_lo = (fps * 60.0 / 180.0).floor() as usize;
    let lag_hi = (fps * 60.0 / 70.0).ceil() as usize;
    if onset.len() < lag_hi * 4 { return None; }
    let mean = onset.iter().sum::<f32>() / onset.len() as f32;
    let c: Vec<f32> = onset.iter().map(|x| x - mean).collect();
    let (mut best, mut best_lag) = (f32::MIN, lag_lo);
    for lag in lag_lo.max(1)..=lag_hi {
        let s: f32 = c.iter().zip(&c[lag..]).map(|(a, b)| a * b).sum::<f32>() / (c.len() - lag) as f32;
        if s > best { best = s; best_lag = lag; }
    }
    // Phase: the offset whose comb of beats collects the most onset energy.
    let period = best_lag as f32;
    let (mut best_off, mut best_s) = (0.0, f32::MIN);
    for off in 0..best_lag {
        let s: f32 = (0..).map(|k| off + k * best_lag).take_while(|&i| i < onset.len()).map(|i| onset[i]).sum();
        if s > best_s { best_s = s; best_off = off as f32; }
    }
    Some((period, best_off))
}

/// Analyze `wav` into `ceil(duration * fps)` frames of features. `bpm` (if known) drives beat_phase;
/// otherwise the tempo is estimated from the onset curve.
pub fn analyze(wav: &Path, fps: f32, bpm: Option<f32>) -> anyhow::Result<Vec<Features>> {
    let (mono, sr) = read_mono(wav)?;
    let [bass, mid, high, onset, rms] = raw_features(&mono, sr, fps);
    let beat = match bpm {
        Some(b) => Some((fps * 60.0 / b, 0.0)),
        None => estimate_beat(&onset, fps),
    };
    // Envelope follower coefficients: tau → per-frame blend factor.
    let dt = 1.0 / fps;
    let k = |tau: f32| 1.0 - (-dt / tau).exp();
    let (att, rel, rms_k, onset_decay) = (k(0.015), k(0.12), k(0.08), (-dt / 0.08).exp());
    let follow = |prev: f32, x: f32| prev + if x > prev { att } else { rel } * (x - prev);

    let mut s = Features::default();
    let mut out = Vec::with_capacity(bass.len());
    for i in 0..bass.len() {
        s.bass = follow(s.bass, bass[i]);
        s.mid = follow(s.mid, mid[i]);
        s.high = follow(s.high, high[i]);
        s.onset = onset[i].max(s.onset * onset_decay); // instant attack, quick exponential fall
        s.rms += rms_k * (rms[i] - s.rms);
        s.beat_phase = beat.map_or(0.0, |(p, off)| ((i as f32 - off) / p).rem_euclid(1.0));
        out.push(s);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("keep-audio-{}-{name}.wav", std::process::id()))
    }

    #[test]
    fn synthesis_is_deterministic() {
        let spec = TrackSpec { seconds: 3.0, bpm: 110.0, sample_rate: 22_050, seed: 42 };
        let (a, b, c) = (tmp("d1"), tmp("d2"), tmp("d3"));
        synthesize(&spec, &a).unwrap();
        synthesize(&spec, &b).unwrap();
        synthesize(&TrackSpec { seed: 43, ..spec }, &c).unwrap();
        let (ba, bb, bc) = (std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap(), std::fs::read(&c).unwrap());
        assert_eq!(ba, bb, "same seed must give identical bytes");
        assert_ne!(ba, bc, "different seed should change the track");
        for p in [a, b, c] { let _ = std::fs::remove_file(p); }
    }

    #[test]
    fn onsets_align_with_kicks() {
        // 120 BPM → a beat every 0.5 s = 30 frames at 60 fps. Kicks start at bar 4 (t = 8 s).
        let spec = TrackSpec { seconds: 14.0, bpm: 120.0, sample_rate: 48_000, seed: 7 };
        let p = tmp("onset");
        synthesize(&spec, &p).unwrap();
        let feats = analyze(&p, 60.0, Some(120.0)).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(feats.len(), 14 * 60);

        let window_max = |c: usize, r: std::ops::Range<isize>| {
            r.map(|d| feats[(c as isize + d) as usize].onset).fold(0.0f32, f32::max)
        };
        let (mut on_kick, mut off_kick) = (0.0, 0.0);
        let beats = 17..27;
        for b in beats.clone() {
            let f = b * 30;
            let k = window_max(f, -1..4);
            assert!(k > 0.4, "beat {b}: onset {k} too weak at the kick");
            on_kick += k;
            off_kick += window_max(f + 15, -1..4); // the offbeat (hats + bass live here)
            assert!(feats[f + 1].beat_phase < 0.1, "beat_phase should wrap at the beat");
        }
        let n = beats.len() as f32;
        assert!(on_kick / n > 1.5 * off_kick / n, "kick onsets {on_kick} vs offbeats {off_kick}");
    }

    #[test]
    fn tempo_estimate_finds_the_beat() {
        // Synthetic onset train at 120 BPM / 60 fps: impulse every 30 frames starting at 7.
        let onset: Vec<f32> = (0..1200).map(|i| if i % 30 == 7 { 1.0 } else { 0.0 }).collect();
        let (p, off) = estimate_beat(&onset, 60.0).unwrap();
        assert_eq!((p, off), (30.0, 7.0));
    }
}
