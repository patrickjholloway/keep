//! glitch.rs — the image-plane "glitch patch bay" (CPU side).
//!
//! The glitch modules only touch the rasterized 2D HDR image (after the particles are drawn,
//! before the tonemap). Nothing here changes the 4D field or the particles.
//!
//! Two views of one frame:
//!   * a **signal**: the scanlines concatenated, sample index s = y·W + x (like analog video);
//!     sync slip, ring mod and smear operate along s.
//!   * a **plane**: pixels as points in R² / C; dispersion, the conformal (Möbius) warp,
//!     the spectral gate (2D FFT) and bitcrush operate on it.
//!
//! Patch bay, modular-synth style:
//!   value(param) = base(param) + Σ_patches(param) [ gain · source + offset ]     (then clamped)
//! where `base` comes from `glitch.set(...)` (default = the module's identity / bypass value) and
//! sources are the audio features, LFOs, the previous frame's mean luminance and script-defined
//! sources. Resolution happens per frame in Rust (`resolve`) into the `GlitchParams` uniform.
//!
//! Contract: `GlitchParams` here ⇄ `uniform Glitch` in shaders/glitch_common.glsl (std140, all
//! vec4, size asserted). Change both together.
use std::collections::BTreeMap;

use anyhow::bail;
use bytemuck::{Pod, Zeroable};

use crate::audio::Features;

/// GPU uniform for every glitch pass (`uniform Glitch` in shaders/glitch_common.glsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GlitchParams {
    /// x = depth (max delay, fraction of a line), y = freq (bands per frame height),
    /// z = speed (band drift, Hz), w = block (fraction of bands that tear, 0..1).
    pub sync: [f32; 4],
    /// x = depth (0..1), y = freq (carrier cycles per scanline), z = speed (carrier drift, Hz),
    /// w = chroma (carrier phase offset between R, G, B in radians).
    pub ring: [f32; 4],
    /// x = depth (0..1 trail mix), y = feedback a (0..0.999), z = direction (+1 →, -1 ←), w = unused.
    pub smear: [f32; 4],
    /// x = depth (pixels of red↔blue split at 1080p, at the frame edge for radial),
    /// y = mode (0 radial, 1 directional), z = angle (radians, directional mode), w = minimum
    /// split (px at 1080p) whenever depth > 0, so a kick always reads as a colour split.
    pub disp: [f32; 4],
    /// Möbius coefficients a, b (complex, re/im) — already blended with identity by `warp.amount`.
    pub warp_ab: [f32; 4],
    /// Möbius coefficients c, d (complex, re/im).
    pub warp_cd: [f32; 4],
    /// x = amount (0 = bypass), y = zoom (output z scale), z = twist (radians of swirl at the
    /// centre, scaled by amount), w = twist radius (Gaussian falloff, frame-height units).
    pub warp_m: [f32; 4],
    /// x = mix (0 = bypass), y = band lo, z = band hi (radial frequency, cycles/pixel of the
    /// working grid, 0..0.5·√2), w = attenuation in the band (0..1, can exceed 1 to invert).
    pub spec: [f32; 4],
    /// x = phase (ghost-echo strength in the band, 0..1 → up to ±π), y = seed (echo direction), z = band softness, w = gain
    /// outside the band (1 = untouched).
    pub spec2: [f32; 4],
    /// x = depth (0 = off .. 1 = 2 brightness levels), y = hold (sample-and-hold block, pixels),
    /// zw unused.
    pub crush: [f32; 4],
    /// x = time (s), y = mean luminance of the previous frame, z = frame index, w = unused.
    pub misc: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<GlitchParams>() == 11 * 16);

impl Default for GlitchParams {
    fn default() -> Self {
        resolve(&GlitchDesc::default(), 0.0, &Features::default(), 0.0, 0)
    }
}

impl GlitchParams {
    pub fn sync_on(&self) -> bool { self.sync[0] > 0.0 }
    pub fn ring_on(&self) -> bool { self.ring[0] > 0.0 }
    pub fn smear_on(&self) -> bool { self.smear[0] > 0.0 }
    pub fn disp_on(&self) -> bool { self.disp[0] > 0.0 }
    pub fn warp_on(&self) -> bool { self.warp_m[0] > 0.0 }
    pub fn spec_on(&self) -> bool { self.spec[0] > 0.0 }
    pub fn crush_on(&self) -> bool { self.crush[0] > 0.0 || self.crush[1] > 1.0 }
}

// --------------------------------------------------------------------------- parameter table

/// Every patchable destination: name, default (the identity / bypass value) and clamp range.
/// `warp.a` etc. are complex; Lua's `glitch.set("warp.a", re, im)` writes `warp.a.re/.im`.
pub const PARAMS: &[(&str, f32, f32, f32)] = &[
    ("sync.depth", 0.0, 0.0, 1.0),
    ("sync.freq", 24.0, 1.0, 400.0),
    ("sync.speed", 1.5, -50.0, 50.0),
    ("sync.block", 0.35, 0.0, 1.0),
    ("ring.depth", 0.0, 0.0, 1.0),
    ("ring.freq", 3.37, 0.0, 400.0),
    ("ring.speed", 0.5, -50.0, 50.0),
    ("ring.chroma", 0.6, -6.3, 6.3),
    ("smear.depth", 0.0, 0.0, 1.0),
    ("smear.feedback", 0.95, 0.0, 0.999),
    ("smear.dir", 1.0, -1.0, 1.0),
    ("smear.axis", 0.0, 0.0, 1.0),
    ("dispersion.depth", 0.0, 0.0, 200.0),
    ("dispersion.mode", 0.0, 0.0, 1.0),
    ("dispersion.angle", 0.0, -100.0, 100.0),
    ("dispersion.min", 0.0, 0.0, 64.0),
    ("warp.amount", 0.0, 0.0, 1.0),
    ("warp.zoom", 1.0, 0.05, 20.0),
    ("warp.twist", 0.0, -12.0, 12.0),
    ("warp.radius", 0.6, 0.05, 4.0),
    ("warp.a.re", 1.0, -100.0, 100.0),
    ("warp.a.im", 0.0, -100.0, 100.0),
    ("warp.b.re", 0.0, -100.0, 100.0),
    ("warp.b.im", 0.0, -100.0, 100.0),
    ("warp.c.re", 0.0, -100.0, 100.0),
    ("warp.c.im", 0.0, -100.0, 100.0),
    ("warp.d.re", 1.0, -100.0, 100.0),
    ("warp.d.im", 0.0, -100.0, 100.0),
    ("spectral.mix", 0.0, 0.0, 1.0),
    ("spectral.lo", 0.02, 0.0, 0.8),
    ("spectral.hi", 0.12, 0.0, 0.8),
    ("spectral.atten", 1.0, -4.0, 4.0),
    ("spectral.phase", 0.0, 0.0, 1.0),
    ("spectral.seed", 0.0, -1e6, 1e6),
    ("spectral.soft", 0.01, 0.0005, 0.5),
    ("spectral.outside", 1.0, 0.0, 4.0),
    ("crush.depth", 0.0, 0.0, 1.0),
    ("crush.hold", 1.0, 1.0, 256.0),
];

pub fn param_index(name: &str) -> Option<usize> {
    PARAMS.iter().position(|p| p.0 == name)
}

/// Built-in source names (plus any LFO or script source the Lua side defines).
pub const BUILTIN_SOURCES: &[&str] = &["bass", "mid", "high", "onset", "rms", "beat_phase", "mean_luma", "one"];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LfoShape { Sine, Tri, Saw, Square }

impl LfoShape {
    pub fn parse(s: &str) -> anyhow::Result<LfoShape> {
        Ok(match s {
            "sine" => LfoShape::Sine,
            "tri" | "triangle" => LfoShape::Tri,
            "saw" => LfoShape::Saw,
            "square" => LfoShape::Square,
            _ => bail!("unknown lfo shape `{s}` (sine|tri|saw|square)"),
        })
    }
}

/// A low-frequency oscillator, unipolar 0..1 (so `gain`/`offset` read the same as for audio).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lfo {
    pub shape: LfoShape,
    /// Hz.
    pub rate: f32,
    /// Phase offset in cycles.
    pub phase: f32,
}

impl Lfo {
    pub fn eval(&self, t: f32) -> f32 {
        let p = (t * self.rate + self.phase).rem_euclid(1.0);
        match self.shape {
            LfoShape::Sine => 0.5 - 0.5 * (std::f32::consts::TAU * p).cos(), // starts at 0
            LfoShape::Tri => 1.0 - (2.0 * p - 1.0).abs(),
            LfoShape::Saw => p,
            LfoShape::Square => if p < 0.5 { 1.0 } else { 0.0 },
        }
    }
}

/// One cable: dst += gain · src + offset.
#[derive(Clone, Debug, PartialEq)]
pub struct Patch {
    pub src: String,
    pub dst: usize,
    pub gain: f32,
    pub offset: f32,
}

/// What the script asked for (parsed from the Lua `glitch` table after `update`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GlitchDesc {
    /// Base values by PARAMS index (missing = default).
    pub sets: BTreeMap<usize, f32>,
    pub patches: Vec<Patch>,
    pub lfos: BTreeMap<String, Lfo>,
    /// Script-computed sources (`glitch.source(name, v)`).
    pub sources: BTreeMap<String, f32>,
}

impl GlitchDesc {
    /// Look up a source's current value. Unknown names read 0 (validated at parse time).
    pub fn source(&self, name: &str, t: f32, f: &Features, mean_luma: f32) -> f32 {
        match name {
            "bass" => f.bass,
            "mid" => f.mid,
            "high" => f.high,
            "onset" => f.onset,
            "rms" => f.rms,
            "beat_phase" => f.beat_phase,
            "mean_luma" => mean_luma,
            "one" => 1.0,
            _ => self
                .sources
                .get(name)
                .copied()
                .or_else(|| self.lfos.get(name).map(|l| l.eval(t)))
                .unwrap_or(0.0),
        }
    }

    pub fn is_known_source(&self, name: &str) -> bool {
        BUILTIN_SOURCES.contains(&name) || self.lfos.contains_key(name) || self.sources.contains_key(name)
    }
}

/// Resolve every parameter for this frame: base + Σ(gain·src + offset), clamped to its range.
pub fn resolve_values(desc: &GlitchDesc, t: f32, f: &Features, mean_luma: f32) -> Vec<f32> {
    let mut v: Vec<f32> = PARAMS.iter().enumerate().map(|(i, p)| desc.sets.get(&i).copied().unwrap_or(p.1)).collect();
    for p in &desc.patches {
        v[p.dst] += p.gain * desc.source(&p.src, t, f, mean_luma) + p.offset;
    }
    for (x, p) in v.iter_mut().zip(PARAMS) {
        *x = if x.is_finite() { x.clamp(p.2, p.3) } else { p.1 };
    }
    v
}

/// Blend a Möbius coefficient set with the identity (a=d=1, b=c=0) by `amount` in 0..1.
///
/// The plain coefficient lerp I + (M − I)·t is NOT always a Möbius map: its determinant
/// ad − bc is a quadratic in t and can pass through 0 (a = −1, b = c = 0 at t = ½ gives
/// a = 0: every pixel would sample one point). It is what the exemplar's dipole maps were
/// tuned with, and for them (a = d = 1) the determinant 1 − t²·bc never nears 0, so it is kept
/// whenever the whole lerp path stays well away from singular (relative |det| > 0.2 along it;
/// a unitary map scores 0.5). Otherwise the blend uses the
/// matrix power M^t (a geodesic from I to M), which is invertible for every t: a rotation
/// preset then rotates by t·θ instead of shrinking through 0. The choice depends only on M,
/// so ramping `amount` never switches branch mid-ramp. A singular M gives the identity.
pub fn blend_with_identity(m: [C; 4], amount: f32) -> [C; 4] {
    let id = [C(1.0, 0.0), C(0.0, 0.0), C(0.0, 0.0), C(1.0, 0.0)];
    let lerp = |t: f32| {
        let mut out = id;
        for i in 0..4 {
            out[i] = C(id[i].0 + (m[i].0 - id[i].0) * t, id[i].1 + (m[i].1 - id[i].1) * t);
        }
        out
    };
    let scale = |q: [C; 4]| q.iter().map(|c| c.0 * c.0 + c.1 * c.1).sum::<f32>().max(1e-12);
    // relative singularity: |det| / (|a|²+|b|²+|c|²+|d|²) is ½ for a unitary map, 0 if singular
    let rel = |q: [C; 4]| det(q).abs() / scale(q);
    if !(rel(m) > 1e-6) {
        return id;
    }
    let lerp_safe = (0..=64).all(|i| rel(lerp(i as f32 / 64.0)) > 0.2);
    if lerp_safe { lerp(amount) } else { mat_pow(m, amount) }
}

fn det(m: [C; 4]) -> C { m[0].mul(m[3]).sub(m[1].mul(m[2])) }

/// M^t for an invertible complex 2×2 matrix [[a, b], [c, d]] (principal branch).
/// Distinct eigenvalues λ1 ≠ λ2 (Sylvester):  M^t = (λ1^t (M − λ2 I) − λ2^t (M − λ1 I)) / (λ1 − λ2)
/// Repeated eigenvalue λ (M = λ(I + N), N nilpotent):  M^t = λ^t (I + t N)
fn mat_pow(m: [C; 4], t: f32) -> [C; 4] {
    let [a, b, c, d] = m;
    let half_tr = C(0.5 * (a.0 + d.0), 0.5 * (a.1 + d.1));
    let disc = half_tr.mul(half_tr).sub(det(m)).sqrt();
    let (l1, l2) = (half_tr.add(disc), half_tr.sub(disc));
    let one = C(1.0, 0.0);
    if disc.abs() < 1e-4 * half_tr.abs().max(1e-6) {
        let lt = half_tr.pow(t);
        let ti = C(t, 0.0);
        let n = [a.div(half_tr).sub(one), b.div(half_tr), c.div(half_tr), d.div(half_tr).sub(one)];
        return [lt.mul(one.add(ti.mul(n[0]))), lt.mul(ti.mul(n[1])), lt.mul(ti.mul(n[2])), lt.mul(one.add(ti.mul(n[3])))];
    }
    let (p1, p2) = (l1.pow(t), l2.pow(t));
    let den = l1.sub(l2);
    let f = |m_ii: C, lam_other: C, lam: C, is_diag: bool| -> C {
        let (x, y) = if is_diag { (m_ii.sub(lam_other), m_ii.sub(lam)) } else { (m_ii, m_ii) };
        p1.mul(x).sub(p2.mul(y)).div(den)
    };
    [f(a, l2, l1, true), f(b, l2, l1, false), f(c, l2, l1, false), f(d, l2, l1, true)]
}

pub fn resolve(desc: &GlitchDesc, t: f32, f: &Features, mean_luma: f32, frame: u32) -> GlitchParams {
    let v = resolve_values(desc, t, f, mean_luma);
    let g = |n: &str| v[param_index(n).expect("param name")];
    let amount = g("warp.amount");
    let m = blend_with_identity(
        [C(g("warp.a.re"), g("warp.a.im")), C(g("warp.b.re"), g("warp.b.im")), C(g("warp.c.re"), g("warp.c.im")), C(g("warp.d.re"), g("warp.d.im"))],
        amount,
    );
    GlitchParams {
        sync: [g("sync.depth"), g("sync.freq"), g("sync.speed"), g("sync.block")],
        ring: [g("ring.depth"), g("ring.freq"), g("ring.speed"), g("ring.chroma")],
        smear: [g("smear.depth"), g("smear.feedback"), if g("smear.dir") < 0.0 { -1.0 } else { 1.0 }, if g("smear.axis") >= 0.5 { 1.0 } else { 0.0 }],
        disp: [g("dispersion.depth"), g("dispersion.mode").round(), g("dispersion.angle"), g("dispersion.min")],
        warp_ab: [m[0].0, m[0].1, m[1].0, m[1].1],
        warp_cd: [m[2].0, m[2].1, m[3].0, m[3].1],
        warp_m: [amount, g("warp.zoom"), g("warp.twist"), g("warp.radius")],
        spec: [g("spectral.mix"), g("spectral.lo"), g("spectral.hi"), g("spectral.atten")],
        spec2: [g("spectral.phase"), g("spectral.seed"), g("spectral.soft"), g("spectral.outside")],
        crush: [g("crush.depth"), g("crush.hold").round(), 0.0, 0.0],
        misc: [t, mean_luma, frame as f32, 0.0],
    }
}

// --------------------------------------------------------------------------- CPU references
// These mirror the shader math one-to-one; the tests pin them (unused by the binary itself).

/// Minimal complex number (re, im).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct C(pub f32, pub f32);

#[cfg_attr(not(test), allow(dead_code))]
impl C {
    pub fn add(self, o: C) -> C { C(self.0 + o.0, self.1 + o.1) }
    pub fn sub(self, o: C) -> C { C(self.0 - o.0, self.1 - o.1) }
    pub fn mul(self, o: C) -> C { C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0) }
    pub fn div(self, o: C) -> C {
        let d = o.0 * o.0 + o.1 * o.1;
        C((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d)
    }
    pub fn neg(self) -> C { C(-self.0, -self.1) }
    pub fn abs(self) -> f32 { self.0.hypot(self.1) }
    pub fn expi(theta: f32) -> C { C(theta.cos(), theta.sin()) }
    pub fn sqrt(self) -> C { let r = self.abs().sqrt(); C::expi(0.5 * self.1.atan2(self.0)).mul(C(r, 0.0)) }
    /// Principal power z^t = e^{t·ln z} (z ≠ 0).
    pub fn pow(self, t: f32) -> C { C::expi(t * self.1.atan2(self.0)).mul(C(self.abs().powf(t), 0.0)) }
}

/// Forward Möbius map f(z) = (a z + b) / (c z + d).
#[cfg_attr(not(test), allow(dead_code))]
pub fn mobius(m: [C; 4], z: C) -> C {
    m[0].mul(z).add(m[1]).div(m[2].mul(z).add(m[3]))
}

/// Inverse map f⁻¹(w) = (d w − b) / (−c w + a). The shader uses this: each OUTPUT pixel w asks
/// "which input point z maps onto me?" and samples the image there (backward mapping, no holes).
#[cfg_attr(not(test), allow(dead_code))]
pub fn mobius_inverse(m: [C; 4], w: C) -> C {
    m[3].mul(w).sub(m[1]).div(m[2].neg().mul(w).add(m[0]))
}

/// One step of the smear filter: a one-pole IIR low-pass y[n] = (1 − a)·x[n] + a·y[n−1].
/// Impulse response (1 − a)·aⁿ: unit DC gain, time constant −1/ln(a) samples.
#[cfg_attr(not(test), allow(dead_code))]
pub fn iir_run(x: &[f32], a: f32) -> Vec<f32> {
    let mut y = 0.0;
    x.iter().map(|&v| { y = (1.0 - a) * v + a * y; y }).collect()
}

/// Radix-2 Stockham autosort FFT, the exact algorithm of shaders/glitch_fft.comp.
/// `sign` = −1 forward, +1 inverse (unnormalized; divide by n yourself). n must be 2^k.
///
/// Stage with sub-transform size Ns = 1, 2, 4, …, n/2; for each j in 0..n/2:
///   k = j mod Ns;  v0 = x[j];  v1 = x[j + n/2] · e^{sign·iπ·k/Ns}
///   y[(j − k)·2 + k] = v0 + v1;   y[(j − k)·2 + k + Ns] = v0 − v1
/// Stockham ping-pongs between two buffers and lands in natural order (no bit reversal).
#[cfg_attr(not(test), allow(dead_code))]
pub fn fft_stockham(data: &mut Vec<C>, sign: f32) {
    let n = data.len();
    assert!(n.is_power_of_two());
    let mut src = data.clone();
    let mut dst = vec![C(0.0, 0.0); n];
    let mut ns = 1;
    while ns < n {
        for j in 0..n / 2 {
            let k = j % ns;
            let w = C::expi(sign * std::f32::consts::PI * k as f32 / ns as f32);
            let v0 = src[j];
            let v1 = src[j + n / 2].mul(w);
            let o = (j - k) * 2 + k;
            dst[o] = v0.add(v1);
            dst[o + ns] = v0.sub(v1);
        }
        std::mem::swap(&mut src, &mut dst);
        ns *= 2;
    }
    *data = src;
}

/// 2D FFT (rows then columns) of a w×h grid, row-major.
#[cfg_attr(not(test), allow(dead_code))]
pub fn fft2d(data: &mut [C], w: usize, h: usize, sign: f32) {
    for y in 0..h {
        let mut row = data[y * w..(y + 1) * w].to_vec();
        fft_stockham(&mut row, sign);
        data[y * w..(y + 1) * w].copy_from_slice(&row);
    }
    for x in 0..w {
        let mut col: Vec<C> = (0..h).map(|y| data[y * w + x]).collect();
        fft_stockham(&mut col, sign);
        for y in 0..h { data[y * w + x] = col[y]; }
    }
}

/// Radial frequency of FFT bin (kx, ky) in cycles/pixel (bins above n/2 are negative freqs).
#[cfg_attr(not(test), allow(dead_code))]
pub fn bin_freq(kx: usize, ky: usize, w: usize, h: usize) -> f32 {
    let fx = if kx <= w / 2 { kx as f32 } else { kx as f32 - w as f32 } / w as f32;
    let fy = if ky <= h / 2 { ky as f32 } else { ky as f32 - h as f32 } / h as f32;
    fx.hypot(fy)
}

/// Spectral gate transfer: 1 − atten·band(f), band = soft-edged [lo, hi]. DC (f = 0) passes.
#[cfg_attr(not(test), allow(dead_code))]
pub fn gate_gain(f: f32, lo: f32, hi: f32, soft: f32, atten: f32, outside: f32) -> f32 {
    if f == 0.0 { return 1.0; }
    let s = |e0: f32, e1: f32, x: f32| { let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) };
    let band = s(lo - soft, lo + soft, f) * (1.0 - s(hi - soft, hi + soft, f));
    band * (1.0 - atten) + (1.0 - band) * outside
}

/// Smear output for one scanline (glitch_smear.comp): only the boosted history ABOVE the input
/// is added, so flat input maps to itself.
#[cfg_attr(not(test), allow(dead_code))]
pub fn smear_run(x: &[f32], a: f32, depth: f32) -> Vec<f32> {
    let boost = (1.0 / (1.0 - a).sqrt()).min(8.0);
    let mut acc = 0.0;
    x.iter().map(|&v| {
        let o = v + depth * (boost * a * (acc - v)).max(0.0);
        acc = (1.0 - a) * v + a * acc;
        o
    }).collect()
}

/// Bitcrush quantizer (glitch_crush.comp): mid-tread L levels in the tone-compressed domain.
#[cfg_attr(not(test), allow(dead_code))]
pub fn crush_quantize(c: f32, levels: f32) -> f32 {
    let v = c / (1.0 + c);
    let v = ((v * (levels - 1.0)).round() / (levels - 1.0)).min((levels - 0.5) / levels);
    v / (1.0 - v)
}

/// Bitcrush a pixel (glitch_crush.comp): quantize its LUMINANCE and rescale the RGB by the
/// same factor, so the hue (channel ratios) survives — posterized bands in the frame's own
/// palette instead of per-channel rounding that throws out random saturated primaries.
#[cfg_attr(not(test), allow(dead_code))]
pub fn crush_pixel(c: [f32; 3], levels: f32) -> [f32; 3] {
    const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
    let l = c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2];
    if l <= 0.0 { return [0.0; 3]; }
    let k = crush_quantize(l, levels) / l;
    [c[0] * k, c[1] * k, c[2] * k]
}

/// Mirror (even) extension of a w×h image into a 2w×2h grid, as LOAD in glitch_spectral.comp
/// does: the periodic extension has no seam, so the FFT does not see a torus edge.
#[cfg_attr(not(test), allow(dead_code))]
pub fn mirror_extend(img: &[f32], w: usize, h: usize) -> Vec<C> {
    let m = |x: usize, n: usize| if x < n { x } else { 2 * n - 1 - x };
    (0..4 * w * h).map(|i| C(img[m(i / (2 * w), h) * w + m(i % (2 * w), w)], 0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() <= eps }

    #[test]
    fn patch_resolution_sums_and_clamps() {
        let mut d = GlitchDesc::default();
        let depth = param_index("sync.depth").unwrap();
        d.sets.insert(depth, 0.1);
        d.patches.push(Patch { src: "bass".into(), dst: depth, gain: 0.5, offset: 0.0 });
        d.patches.push(Patch { src: "onset".into(), dst: depth, gain: 0.2, offset: 0.05 });
        let f = Features { bass: 0.4, onset: 1.0, ..Default::default() };
        let v = resolve_values(&d, 0.0, &f, 0.0);
        assert!(close(v[depth], 0.1 + 0.5 * 0.4 + 0.2 + 0.05, 1e-6), "{}", v[depth]);
        // clamp to [0, 1]
        d.patches.push(Patch { src: "one".into(), dst: depth, gain: 5.0, offset: 0.0 });
        assert_eq!(resolve_values(&d, 0.0, &f, 0.0)[depth], 1.0);
        // defaults are the identity: nothing on, warp = identity
        let p = resolve(&GlitchDesc::default(), 1.0, &Features::default(), 0.3, 0);
        assert!(!p.sync_on() && !p.ring_on() && !p.smear_on() && !p.disp_on() && !p.warp_on() && !p.spec_on() && !p.crush_on());
        assert_eq!(p.misc[1], 0.3);
    }

    #[test]
    fn lfo_and_custom_sources() {
        let mut d = GlitchDesc::default();
        d.lfos.insert("wob".into(), Lfo { shape: LfoShape::Square, rate: 2.0, phase: 0.0 });
        d.lfos.insert("s".into(), Lfo { shape: LfoShape::Sine, rate: 1.0, phase: 0.0 });
        d.sources.insert("env".into(), 0.7);
        let f = Features::default();
        assert_eq!(d.source("wob", 0.1, &f, 0.0), 1.0);
        assert_eq!(d.source("wob", 0.3, &f, 0.0), 0.0);
        assert!(close(d.source("s", 0.5, &f, 0.0), 1.0, 1e-6));
        assert!(close(d.source("s", 0.0, &f, 0.0), 0.0, 1e-6));
        assert_eq!(d.source("env", 0.0, &f, 0.0), 0.7);
        assert_eq!(d.source("mean_luma", 0.0, &f, 0.42), 0.42);
        let tri = Lfo { shape: LfoShape::Tri, rate: 1.0, phase: 0.0 };
        assert!(close(tri.eval(0.25), 0.5, 1e-6) && close(tri.eval(0.5), 1.0, 1e-6));
        let saw = Lfo { shape: LfoShape::Saw, rate: 0.5, phase: 0.0 };
        assert!(close(saw.eval(1.0), 0.5, 1e-6));
    }

    #[test]
    fn mobius_inverse_roundtrips() {
        let m = [C(0.9, 0.3), C(0.2, -0.1), C(0.15, 0.25), C(1.1, -0.2)];
        for &z in &[C(0.0, 0.0), C(0.5, -0.3), C(-1.2, 0.8), C(0.01, 1.5)] {
            let w = mobius(m, z);
            let back = mobius_inverse(m, w);
            assert!(back.sub(z).abs() < 1e-4, "{z:?} -> {w:?} -> {back:?}");
            let fwd = mobius(m, mobius_inverse(m, z));
            assert!(fwd.sub(z).abs() < 1e-4);
        }
        // identity blend: amount 0 is exactly the identity map
        let id = blend_with_identity(m, 0.0);
        assert_eq!(mobius(id, C(0.3, 0.7)), C(0.3, 0.7));
        // a = −1 (rotation by π): the plain lerp would hit a = 0 (singular) at ½; the blend must
        // stay invertible and rotate by π/2 there instead
        let half = blend_with_identity([C(-1.0, 0.0), C(0.0, 0.0), C(0.0, 0.0), C(1.0, 0.0)], 0.5);
        assert!(det(half).abs() > 0.5, "{half:?}");
        let w = mobius(half, C(1.0, 0.0));
        assert!(close(w.abs(), 1.0, 1e-4) && close(w.0, 0.0, 1e-4), "{w:?}");
        for i in 0..=20 {
            let q = blend_with_identity([C::expi(3.0), C(0.2, 0.0), C(0.0, 0.1), C(1.0, 0.0)], i as f32 / 20.0);
            assert!(det(q).abs() > 0.3, "{i} {q:?}");
        }
        // endpoints exact on the power branch too
        let rot_pi = [C(-1.0, 0.0), C(0.0, 0.0), C(0.0, 0.0), C(1.0, 0.0)];
        let full = blend_with_identity(rot_pi, 1.0);
        assert!(mobius(full, C(0.3, 0.4)).sub(C(-0.3, -0.4)).abs() < 1e-4);
        // the exemplar's dipole map (a = d = 1) keeps the plain lerp (unchanged look)
        let dip = [C(1.0, 0.0), C(0.0, -0.4), C(0.0, -0.9), C(1.0, 0.0)];
        assert_eq!(blend_with_identity(dip, 0.5), [C(1.0, 0.0), C(0.0, -0.2), C(0.0, -0.45), C(1.0, 0.0)]);
        // a pure rotation (a = e^{iθ}, d = 1) preserves |z|: conformal + isometric
        let rot = [C::expi(0.7), C(0.0, 0.0), C(0.0, 0.0), C(1.0, 0.0)];
        assert!(close(mobius(rot, C(0.6, 0.8)).abs(), 1.0, 1e-6));
    }

    #[test]
    fn iir_dc_gain_and_decay() {
        // step response converges to the input (unit DC gain)
        let y = iir_run(&vec![1.0; 2000], 0.95);
        assert!(close(*y.last().unwrap(), 1.0, 1e-4));
        // impulse response is (1-a)·a^n: decays by a per sample, time constant -1/ln(a)
        let mut imp = vec![0.0; 100];
        imp[0] = 1.0;
        let y = iir_run(&imp, 0.9);
        assert!(close(y[0], 0.1, 1e-6));
        for n in 1..50 { assert!(close(y[n] / y[n - 1], 0.9, 1e-4)); }
        // longer feedback = longer trail: energy past 20 samples grows with a
        let tail = |a: f32| iir_run(&imp, a)[20..].iter().sum::<f32>();
        assert!(tail(0.97) > tail(0.9) && tail(0.9) > tail(0.5));
        // a = 0 is a passthrough
        assert_eq!(iir_run(&[0.3, 0.7], 0.0), vec![0.3, 0.7]);
    }

    #[test]
    fn stockham_matches_naive_dft() {
        let n = 16;
        let x: Vec<C> = (0..n).map(|i| C((i as f32 * 0.7).sin() + 0.1 * i as f32, (i as f32 * 1.3).cos())).collect();
        let mut f = x.clone();
        fft_stockham(&mut f, -1.0);
        for k in 0..n {
            let mut acc = C(0.0, 0.0);
            for (j, v) in x.iter().enumerate() {
                acc = acc.add(v.mul(C::expi(-std::f32::consts::TAU * (j * k) as f32 / n as f32)));
            }
            assert!(acc.sub(f[k]).abs() < 1e-3, "bin {k}: {acc:?} vs {:?}", f[k]);
        }
    }

    #[test]
    fn fft2d_roundtrip_identity_on_test_image() {
        let (w, h) = (64, 32);
        // test image: a disc plus stripes
        let img: Vec<C> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let disc = if (x - 30.0).hypot(y - 15.0) < 8.0 { 1.0 } else { 0.0 };
                C(disc + 0.3 * (x * 0.5).sin().max(0.0), 0.0)
            })
            .collect();
        let mut d = img.clone();
        fft2d(&mut d, w, h, -1.0);
        fft2d(&mut d, w, h, 1.0);
        let scale = 1.0 / (w * h) as f32;
        for (a, b) in img.iter().zip(&d) {
            assert!(close(a.0, b.0 * scale, 1e-4) && close(0.0, b.1 * scale, 1e-4));
        }
        // a gate that removes everything but DC returns the mean everywhere
        let mut d = img.clone();
        fft2d(&mut d, w, h, -1.0);
        for ky in 0..h { for kx in 0..w {
            let g = gate_gain(bin_freq(kx, ky, w, h), 0.0, 1.0, 1e-4, 1.0, 1.0);
            d[ky * w + kx] = d[ky * w + kx].mul(C(g, 0.0));
        }}
        fft2d(&mut d, w, h, 1.0);
        let mean = img.iter().map(|c| c.0).sum::<f32>() / (w * h) as f32;
        assert!(d.iter().all(|c| close(c.0 * scale, mean, 1e-3)));
    }

    #[test]
    fn gate_band_shape() {
        assert_eq!(gate_gain(0.0, 0.0, 0.5, 0.01, 1.0, 1.0), 1.0, "DC always passes");
        assert!(close(gate_gain(0.1, 0.05, 0.2, 0.01, 1.0, 1.0), 0.0, 1e-6), "inside the band is removed");
        assert!(close(gate_gain(0.4, 0.05, 0.2, 0.01, 1.0, 1.0), 1.0, 1e-6), "outside passes");
        assert!(close(gate_gain(0.1, 0.05, 0.2, 0.01, 0.5, 1.0), 0.5, 1e-6));
        assert!(close(bin_freq(48, 0, 64, 32), 16.0 / 64.0, 1e-6), "negative frequency wraps");
    }

    #[test]
    fn smear_flat_input_is_identity_and_trail_is_boosted() {
        for &a in &[0.5, 0.95, 0.99] {
            let y = smear_run(&vec![0.7; 500], a, 1.0);
            assert!(y.iter().all(|&o| close(o, 0.7, 1e-4)), "flat run brightened at a={a}");
        }
        let mut x = vec![0.1; 200];
        x[50] = 5.0;
        let y = smear_run(&x, 0.95, 1.0);
        assert!(close(y[50], 5.0, 1e-5), "bead stays crisp");
        assert!(y[51] > 0.1 + 0.5, "trail behind the bead");
        assert!(close(y[10], 0.1, 1e-3), "background before the bead untouched");
    }

    #[test]
    fn crush_top_band_stays_proportionate() {
        // L = 2: top level (v >= 0.5, c >= 1) clamps to v = 0.75 -> c = 3, not infinity
        assert!(close(crush_quantize(3.0, 2.0), 3.0, 1e-4));
        assert!(close(crush_quantize(1000.0, 2.0), 3.0, 1e-3));
        // black stays black at every level count (mid-tread), dim values snap down to 0
        for &l in &[2.0f32, 4.0, 6.7, 256.0] {
            assert_eq!(crush_quantize(0.0, l), 0.0);
        }
        assert_eq!(crush_quantize(0.2, 2.0), 0.0);
        // L = 4: levels v = 0, 1/3, 2/3 -> c = 0, 0.5, 2
        assert!(close(crush_quantize(0.45, 4.0), 0.5, 1e-4));
        // fine quantization is close to the identity
        for &c in &[0.05f32, 0.5, 2.0] {
            assert!((crush_quantize(c, 256.0) - c).abs() < 0.02 * (1.0 + c) * (1.0 + c));
        }
    }

    #[test]
    fn crush_pixel_keeps_hue() {
        // a dim orange at L = 4: per-channel rounding would split it into pure red; luminance
        // quantization keeps the R:G:B ratios
        let c = [0.9f32, 0.45, 0.1];
        let q = crush_pixel(c, 4.0);
        assert!(q[0] > 0.0);
        assert!(close(q[1] / q[0], 0.5, 1e-4) && close(q[2] / q[0], c[2] / c[0], 1e-4));
        assert_eq!(crush_pixel([0.0; 3], 2.0), [0.0; 3]);
    }

    #[test]
    fn mirror_extension_removes_the_edge_seam() {
        // Vertical ramp (dark top, bright bottom): on a torus the bottom row abuts the top row,
        // a step of ~1. A band-stop then rings at the top/bottom rows. Mirrored, the periodic
        // extension is continuous and the edge rows barely move.
        let (w, h) = (32usize, 32usize);
        let img: Vec<f32> = (0..w * h).map(|i| (i / w) as f32 / (h - 1) as f32).collect();
        let band_stop = |grid: &mut Vec<C>, gw: usize, gh: usize| {
            fft2d(grid, gw, gh, -1.0);
            for ky in 0..gh {
                for kx in 0..gw {
                    let g = gate_gain(bin_freq(kx, ky, gw, gh), 0.1, 0.4, 0.01, 1.0, 1.0);
                    let v = grid[ky * gw + kx];
                    grid[ky * gw + kx] = C(v.0 * g, v.1 * g);
                }
            }
            fft2d(grid, gw, gh, 1.0);
            let n = (gw * gh) as f32;
            for v in grid.iter_mut() { *v = C(v.0 / n, v.1 / n); }
        };
        let edge_err = |out: &dyn Fn(usize, usize) -> f32| {
            (0..w).map(|x| (out(x, 0) - img[x]).abs().max((out(x, h - 1) - img[(h - 1) * w + x]).abs())).fold(0.0f32, f32::max)
        };
        let mut torus: Vec<C> = img.iter().map(|&v| C(v, 0.0)).collect();
        band_stop(&mut torus, w, h);
        let mut mir = mirror_extend(&img, w, h);
        band_stop(&mut mir, 2 * w, 2 * h);
        let e_torus = edge_err(&|x, y| torus[y * w + x].0);
        let e_mir = edge_err(&|x, y| mir[y * 2 * w + x].0);
        assert!(e_torus > 0.1, "torus seam should ring ({e_torus})");
        assert!(e_mir < 0.25 * e_torus, "mirror {e_mir} vs torus {e_torus}");
    }
}
