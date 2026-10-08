//! sonify.rs — the visible slice, played as an additive-synthesis voice (sound → 4D form → sound).
//!
//! # The loop
//! `shaders/surface.wgsl` bins every visible bead by its direction from the slice centroid into
//! 1024 octahedral bins (32 × 32) and hands the CPU a [`SurfaceDescriptor`]: per bin the bead
//! count, mean heat, mean roughness and mean radius. [`Sonifier::set_targets`] turns that into
//! one sine partial per bin (control rate, once per video frame), and [`Bank`] renders the
//! partials sample by sample (audio rate).
//!
//! # Mapping (per bin i; all knobs are `sonify.*` in Lua, see [`SonifyParams`])
//! | surface                                 | sound                                         |
//! |-----------------------------------------|-----------------------------------------------|
//! | bead count `c_i`                        | amplitude `a_i = sqrt(c_i / Σc)` (Σ a² = 1: loudness is independent of how many bins are lit) |
//! | bin height (direction's y) `h_i`, radius relative to the mean `ρ_i`, roughness | pitch position `x = 0.45 h + 0.45 ρ + 0.1 rough` in 0..1 → `base + span·x` semitones, optionally snapped to `scale` in `key` |
//! | mean heat (cold inner shell .. hot edge)| brightness: self phase-modulation index `β = 0.35 · brightness · heat` cycles (`sin(φ + β sin φ)`: more upper partials) |
//! | bin direction's x                       | pan (constant power), scaled by `spread`       |
//! | total visible beads                     | presence `min(1, Σc / 1500)` (an empty slice fades out) |
//!
//! So a sphere-like slice is a balanced chord across the span (bins at the top sing high,
//! bulges sing sharper than dents), creased / pinched forms push partials up, a hot shell
//! buzzes, and the stereo image follows the form left/right.
//!
//! # The SIMD pattern (after <https://mitchellh.com/writing/everyone-should-know-simd>)
//! The bank is **struct of arrays** (`phase[]`, `inc[]`, `beta[]`, `al[]`, `ar[]` + targets), so
//! W adjacent oscillators sit in one register. [`render_block`] is written once, generic over a
//! [`Lanes`] type, and instantiated for `f32` (scalar reference), `wide::f32x4` / `f32x8`
//! (portable, stable Rust) and [`neon::F32x4`] (raw `std::arch::aarch64` intrinsics):
//! 1. **broadcast** constants: `V::splat(k)`, polynomial coefficients;
//! 2. **loop by vector width**: `for i in (0..full).step_by(V::W)`, oscillators in lanes;
//! 3. **lane-parallel ops**: smoothing, phase wrap, fast sine — no branches (`abs`, `round`,
//!    `copysign` instead of `if`), so every lane does identical work;
//! 4. **reduce / store**: per-sample lane accumulators are summed across lanes once
//!    (`reduce_add`) at the end of the block; oscillator state is stored back;
//! 5. **scalar remainder**: `n % W` leftover oscillators go through the same code with `V = f32`.
//!
//! Per-sample parameter smoothing (one-pole `x += k (target − x)`) runs inside the hot loop, so
//! control-rate jumps never zipper. Targets are floored at `TINY` so decays never reach
//! subnormal floats (which are ~100× slower on many CPUs).
use glam::Vec3;
use std::f32::consts::TAU;

// ─────────────────────────────── surface descriptor ───────────────────────────────

/// Octahedral grid side (must match SIDE in shaders/surface.wgsl).
pub const SIDE: usize = 32;
pub const NBINS: usize = SIDE * SIDE;
/// Fixed-point scales of the GPU accumulators (must match surface.wgsl).
pub const FX_HEAT: f32 = 1024.0;
pub const FX_ROUGH: f32 = 1024.0;
pub const FX_RADIUS: f32 = 1024.0;
pub const FX_POS: f32 = 512.0;
/// Bytes of the GPU buffer: header (4 × i32) + NBINS × 4 × u32.
pub const SURFACE_BYTES: usize = 16 + NBINS * 16;

/// One bin's statistics (means over the beads in it).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bin {
    pub count: u32,
    /// 0 (cold inner shell) .. 1 (hot outer edge).
    pub heat: f32,
    /// 1 − n·r̂: ~0 smooth outward bulge, ~1 crease / saddle, up to 2 inward-facing.
    pub rough: f32,
    /// Mean distance from the centroid.
    pub radius: f32,
}

/// What the GPU tells the voice about the visible slice each frame.
#[derive(Clone, Debug)]
pub struct SurfaceDescriptor {
    pub total: u32,
    pub centroid: [f32; 3],
    pub bins: Vec<Bin>,
}

impl Default for SurfaceDescriptor {
    fn default() -> Self { SurfaceDescriptor { total: 0, centroid: [0.0; 3], bins: vec![Bin::default(); NBINS] } }
}

impl SurfaceDescriptor {
    /// Decode the raw GPU words (layout of `Surface` in surface.wgsl) in place. No allocation.
    pub fn decode_from(&mut self, words: &[u32]) {
        assert!(words.len() >= 4 + 4 * NBINS);
        let total = words[0] as i32;
        self.total = total.max(0) as u32;
        let n = total.max(1) as f32;
        for k in 0..3 { self.centroid[k] = (words[1 + k] as i32) as f32 / FX_POS / n; }
        for (i, b) in self.bins.iter_mut().enumerate() {
            let w = &words[4 + 4 * i..8 + 4 * i];
            let c = w[0];
            let cn = c.max(1) as f32;
            *b = Bin { count: c, heat: w[1] as f32 / FX_HEAT / cn, rough: w[2] as f32 / FX_ROUGH / cn, radius: w[3] as f32 / FX_RADIUS / cn };
        }
    }

    /// Copy without allocating (both have NBINS bins).
    pub fn copy_from(&mut self, o: &SurfaceDescriptor) {
        self.total = o.total;
        self.centroid = o.centroid;
        self.bins.copy_from_slice(&o.bins);
    }

    /// CPU twin of surface.wgsl's accumulation (tests, and documentation by example).
    /// `beads` = (position, unit normal, heat, roughness is derived) relative to `center`.
    pub fn from_beads(beads: &[(Vec3, Vec3, f32)], center: Vec3) -> SurfaceDescriptor {
        let mut words = vec![0u32; 4 + 4 * NBINS];
        let mut head = [0i32; 4];
        for &(p, n, heat) in beads {
            let r = p - center;
            let rl = r.length();
            let rh = if rl > 1e-6 { r / rl } else { Vec3::Z };
            let b = bin_of(rh);
            let rough = (1.0 - n.dot(rh)).clamp(0.0, 2.0);
            words[4 + 4 * b] += 1;
            words[5 + 4 * b] += (heat * FX_HEAT + 0.5) as u32;
            words[6 + 4 * b] += (rough * FX_ROUGH + 0.5) as u32;
            words[7 + 4 * b] += (rl * FX_RADIUS + 0.5) as u32;
            head[0] += 1;
            for k in 0..3 { head[k + 1] += (p[k] * FX_POS).round() as i32; }
        }
        for k in 0..4 { words[k] = head[k] as u32; }
        let mut d = SurfaceDescriptor::default();
        d.decode_from(&words);
        d
    }
}

/// Octahedral encode (mirror of surface.wgsl): unit vector → [-1, 1]².
pub fn oct_encode(d: Vec3) -> [f32; 2] {
    let d = d / (d.x.abs() + d.y.abs() + d.z.abs());
    if d.z >= 0.0 { return [d.x, d.y]; }
    let s = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    [(1.0 - d.y.abs()) * s(d.x), (1.0 - d.x.abs()) * s(d.y)]
}

/// Octahedral decode: [-1, 1]² → unit vector.
pub fn oct_decode(e: [f32; 2]) -> Vec3 {
    let mut v = Vec3::new(e[0], e[1], 1.0 - e[0].abs() - e[1].abs());
    if v.z < 0.0 {
        let s = |x: f32| if x >= 0.0 { 1.0 } else { -1.0 };
        let (x, y) = (v.x, v.y);
        v.x = (1.0 - y.abs()) * s(x);
        v.y = (1.0 - x.abs()) * s(y);
    }
    v.normalize()
}

/// Bin index of a unit direction (same arithmetic as the shader).
pub fn bin_of(d: Vec3) -> usize {
    let e = oct_encode(d);
    let c = |v: f32| ((v * 0.5 + 0.5) * SIDE as f32).clamp(0.0, SIDE as f32 - 0.5) as usize;
    c(e[1]) * SIDE + c(e[0])
}

/// Direction through the centre of bin `i`.
pub fn bin_dir(i: usize) -> Vec3 {
    let (cx, cy) = (i % SIDE, i / SIDE);
    let f = |c: usize| (c as f32 + 0.5) / SIDE as f32 * 2.0 - 1.0;
    oct_decode([f(cx), f(cy)])
}

// ─────────────────────────────── parameters (Lua `sonify.*`) ───────────────────────────────

/// A musical scale as a 12-bit pitch-class mask relative to the key (bit 0 = the key itself).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale(pub u16);

impl Scale {
    pub const NAMES: [(&'static str, &'static [i32]); 9] = [
        ("chromatic", &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        ("major", &[0, 2, 4, 5, 7, 9, 11]),
        ("minor", &[0, 2, 3, 5, 7, 8, 10]),
        ("dorian", &[0, 2, 3, 5, 7, 9, 10]),
        ("harmonic_minor", &[0, 2, 3, 5, 7, 8, 11]),
        ("pentatonic", &[0, 2, 4, 7, 9]),
        ("minor_pentatonic", &[0, 3, 5, 7, 10]),
        ("whole_tone", &[0, 2, 4, 6, 8, 10]),
        ("fifths", &[0, 7]),
    ];
    pub fn named(name: &str) -> Option<Scale> {
        Self::NAMES.iter().find(|(n, _)| *n == name).map(|(_, d)| Scale::from_degrees(d))
    }
    pub fn from_degrees(d: &[i32]) -> Scale { Scale(d.iter().fold(0u16, |m, &k| m | 1 << k.rem_euclid(12))) }
    fn has(self, n: i32) -> bool { self.0 & (1 << n.rem_euclid(12)) != 0 }
    /// Nearest in-scale semitone to `n` (relative to the key); ties go down.
    pub fn snap(self, n: i32) -> i32 {
        if self.0 & 0xfff == 0 { return n; }
        for d in 0..=6 {
            if self.has(n - d) { return n - d; }
            if self.has(n + d) { return n + d; }
        }
        n
    }
}

/// The voice's knobs. Lua: `sonify.enable = true; sonify.gain = 0.1; ...` (see README).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonifyParams {
    pub enable: bool,
    /// Output level of the voice before its soft limiter (which bounds it below 1.0).
    pub gain: f32,
    /// Lowest pitch, MIDI note number (45 = A2, 110 Hz).
    pub base: f32,
    /// Pitch range in semitones above `base`.
    pub span: f32,
    /// Key (pitch class 0 = C .. 11 = B) the scale is built on.
    pub key: i32,
    pub scale: Scale,
    /// Snap partials to the scale (false = continuous glissando pitch).
    pub quantize: bool,
    /// 0..1: how much heat turns into phase-modulation brightness.
    pub brightness: f32,
    /// 0..1: stereo width.
    pub spread: f32,
    /// Smoothing time constant (seconds) of every per-sample parameter.
    pub smoothing: f32,
}

impl Default for SonifyParams {
    fn default() -> Self {
        SonifyParams {
            enable: false, gain: 0.12, base: 45.0, span: 36.0, key: 0,
            scale: Scale::named("pentatonic").unwrap(), quantize: true,
            brightness: 0.5, spread: 0.8, smoothing: 0.05,
        }
    }
}

/// Pitch class from a name like "D", "f#", "Bb" (or None).
pub fn key_from_name(s: &str) -> Option<i32> {
    let mut c = s.chars();
    let base = match c.next()?.to_ascii_uppercase() {
        'C' => 0, 'D' => 2, 'E' => 4, 'F' => 5, 'G' => 7, 'A' => 9, 'B' => 11, _ => return None,
    };
    let acc = match c.as_str() { "" => 0, "#" | "s" => 1, "b" => -1, _ => return None };
    Some((base + acc as i32).rem_euclid(12))
}

// ─────────────────────────────── the SIMD lane abstraction ───────────────────────────────

/// What the kernel needs from a "vector of W floats". Implemented for `f32` (W = 1),
/// `wide::f32x4`, `wide::f32x8` and the raw-NEON [`neon::F32x4`].
pub trait Lanes: Copy + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self> {
    const W: usize;
    fn splat(x: f32) -> Self;
    /// Load `W` floats from the front of `s`.
    fn load(s: &[f32]) -> Self;
    fn store(self, s: &mut [f32]);
    /// Round half away from zero (f32::round semantics).
    fn round(self) -> Self;
    fn abs(self) -> Self;
    /// self * a + b (fused where the hardware has it).
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// Per bit: `self` (a mask) selects `if_one`, else `if_zero`. With a sign-bit mask this is
    /// copysign; the mask is a broadcast constant passed in, so it is never re-splatted.
    fn bitselect(self, if_one: Self, if_zero: Self) -> Self;
    /// Horizontal sum of the lanes.
    fn reduce_add(self) -> f32;
}

impl Lanes for f32 {
    const W: usize = 1;
    #[inline(always)] fn splat(x: f32) -> Self { x }
    #[inline(always)] fn load(s: &[f32]) -> Self { s[0] }
    #[inline(always)] fn store(self, s: &mut [f32]) { s[0] = self }
    #[inline(always)] fn round(self) -> Self { f32::round(self) }
    #[inline(always)] fn abs(self) -> Self { f32::abs(self) }
    #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self { f32::mul_add(self, a, b) }
    #[inline(always)] fn bitselect(self, a: Self, b: Self) -> Self { f32::from_bits((self.to_bits() & a.to_bits()) | (!self.to_bits() & b.to_bits())) }
    #[inline(always)] fn reduce_add(self) -> f32 { self }
}

macro_rules! impl_wide {
    ($t:ty, $w:expr) => {
        impl Lanes for $t {
            const W: usize = $w;
            #[inline(always)] fn splat(x: f32) -> Self { <$t>::splat(x) }
            #[inline(always)] fn load(s: &[f32]) -> Self { <$t>::from(<[f32; $w]>::try_from(&s[..$w]).unwrap()) }
            #[inline(always)] fn store(self, s: &mut [f32]) { s[..$w].copy_from_slice(&self.to_array()) }
            #[inline(always)] fn round(self) -> Self { <$t>::round(self) }
            #[inline(always)] fn abs(self) -> Self { <$t>::abs(self) }
            #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self { <$t>::mul_add(self, a, b) }
            #[inline(always)] fn bitselect(self, a: Self, b: Self) -> Self { <$t>::bitselect(self, a, b) }
            #[inline(always)] fn reduce_add(self) -> f32 { <$t>::reduce_add(self) }
        }
    };
}
impl_wide!(wide::f32x4, 4);
impl_wide!(wide::f32x8, 8);

/// The same 4-lane vector, spelled in raw AArch64 NEON intrinsics (what `wide` does for you).
#[cfg(target_arch = "aarch64")]
pub mod neon {
    use std::arch::aarch64::*;
    /// NEON is mandatory on AArch64, so these intrinsics are always available there.
    #[derive(Clone, Copy)]
    pub struct F32x4(pub float32x4_t);
    impl std::ops::Add for F32x4 { type Output = Self; #[inline(always)] fn add(self, o: Self) -> Self { unsafe { F32x4(vaddq_f32(self.0, o.0)) } } }
    impl std::ops::Sub for F32x4 { type Output = Self; #[inline(always)] fn sub(self, o: Self) -> Self { unsafe { F32x4(vsubq_f32(self.0, o.0)) } } }
    impl std::ops::Mul for F32x4 { type Output = Self; #[inline(always)] fn mul(self, o: Self) -> Self { unsafe { F32x4(vmulq_f32(self.0, o.0)) } } }
    impl super::Lanes for F32x4 {
        const W: usize = 4;
        #[inline(always)] fn splat(x: f32) -> Self { unsafe { F32x4(vdupq_n_f32(x)) } }               // DUP
        #[inline(always)] fn load(s: &[f32]) -> Self { assert!(s.len() >= 4); unsafe { F32x4(vld1q_f32(s.as_ptr())) } } // LD1
        #[inline(always)] fn store(self, s: &mut [f32]) { assert!(s.len() >= 4); unsafe { vst1q_f32(s.as_mut_ptr(), self.0) } } // ST1
        #[inline(always)] fn round(self) -> Self { unsafe { F32x4(vrndaq_f32(self.0)) } }             // FRINTA: ties away
        #[inline(always)] fn abs(self) -> Self { unsafe { F32x4(vabsq_f32(self.0)) } }                // FABS
        // vfmaq_f32(acc, x, y) = acc + x*y  (FMLA)
        #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self { unsafe { F32x4(vfmaq_f32(b.0, self.0, a.0)) } }
        // BSL: bits of `a` where the mask is 1, of `b` where it is 0.
        #[inline(always)] fn bitselect(self, a: Self, b: Self) -> Self { unsafe { F32x4(vbslq_f32(vreinterpretq_u32_f32(self.0), a.0, b.0)) } }
        #[inline(always)] fn reduce_add(self) -> f32 { unsafe { vaddvq_f32(self.0) } }               // FADDP across lanes
    }
}

// ─────────────────────────────── fast sine ───────────────────────────────

const C3: f32 = -1.0 / 6.0;
const C5: f32 = 1.0 / 120.0;
const C7: f32 = -1.0 / 5040.0;
const C9: f32 = 1.0 / 362_880.0;

/// Polynomial constants, broadcast ONCE per block (SIMD step 1). Splatting inside the hot loop
/// looks free but is not: for `wide::f32x8` on AArch64 (two NEON halves) LLVM materialized each
/// `splat` as a `memset_pattern16` call per sample, which made the "SIMD" kernel slower than
/// scalar until these were hoisted.
#[derive(Clone, Copy)]
pub struct SinConsts<V> { quarter: V, tau: V, one: V, c3: V, c5: V, c7: V, c9: V, sign: V }

impl<V: Lanes> SinConsts<V> {
    #[inline(always)]
    pub fn new() -> Self {
        SinConsts { quarter: V::splat(0.25), tau: V::splat(TAU), one: V::splat(1.0), c3: V::splat(C3), c5: V::splat(C5), c7: V::splat(C7), c9: V::splat(C9), sign: V::splat(-0.0) }
    }
}

/// sin(2π·x) for x in *cycles* (any range), branch-free, for any lane width.
/// 1. wrap: z = x − round(x) ∈ [−½, ½]
/// 2. fold: sin is symmetric about ¼ cycle, so m = ¼ − |(|z| − ¼)| ∈ [0, ¼] has the same |sin|
/// 3. odd Taylor polynomial to θ⁹ on θ = 2πm ∈ [0, π/2] (Horner form, 4 FMAs)
/// 4. restore the sign of z: copysign as a bit select on the sign bit.
/// Error bound: the Taylor remainder at θ = π/2 is (π/2)¹¹/11! ≈ 3.6e-6 (−109 dB re full
/// scale); `fast_sin_error_bound` measures ≤ 4e-6 including f32 rounding.
#[inline(always)]
pub fn fast_sin<V: Lanes>(x: V, c: &SinConsts<V>) -> V {
    let z = x - x.round();
    let m = c.quarter - (z.abs() - c.quarter).abs();
    let t = m * c.tau;
    let t2 = t * t;
    let p = t2.mul_add(c.c9, c.c7);
    let p = t2.mul_add(p, c.c5);
    let p = t2.mul_add(p, c.c3);
    let p = t2.mul_add(p, c.one);
    c.sign.bitselect(z, t * p)
}

/// Soft limiter applied to the summed voice: x / sqrt(1 + x²). Linear for small x (≈ x − x³/2),
/// strictly bounded by ±1: the voice can never exceed full scale whatever the targets.
#[inline(always)]
fn soft_limit(x: f32) -> f32 { x / (1.0 + x * x).sqrt() }

// ─────────────────────────────── the oscillator bank ───────────────────────────────

/// Floor for decaying targets: keeps state out of the subnormal range (−240 dB).
const TINY: f32 = 1e-12;
/// Samples per block (per-lane accumulators live on the stack: BLOCK × 2 × W floats).
pub const BLOCK: usize = 128;

/// N oscillators, struct-of-arrays. Current values chase their `*_t` targets per sample.
pub struct Bank {
    pub n: usize,
    /// Phase in cycles, kept in [−½, ½].
    pub phase: Vec<f32>,
    /// Phase increment (cycles per sample) = f / sample_rate.
    pub inc: Vec<f32>,
    /// Self phase-modulation depth in cycles (brightness).
    pub beta: Vec<f32>,
    /// Left / right amplitude (gain · amplitude · pan).
    pub al: Vec<f32>,
    pub ar: Vec<f32>,
    pub inc_t: Vec<f32>,
    pub beta_t: Vec<f32>,
    pub al_t: Vec<f32>,
    pub ar_t: Vec<f32>,
    /// One-pole smoothing coefficient per sample.
    pub k: f32,
}

/// The kernels the bench compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel { Scalar, Wide4, Wide8, #[cfg(target_arch = "aarch64")] Neon }

impl Kernel {
    pub fn all() -> Vec<Kernel> {
        #[allow(unused_mut)]
        let mut v = vec![Kernel::Scalar, Kernel::Wide4, Kernel::Wide8];
        #[cfg(target_arch = "aarch64")] v.push(Kernel::Neon);
        v
    }
    /// The one used for real audio.
    pub fn best() -> Kernel {
        #[cfg(target_arch = "aarch64")] { Kernel::Neon }
        #[cfg(not(target_arch = "aarch64"))] { Kernel::Wide8 }
    }
}

/// One group of W oscillators held in registers for a block.
struct Osc<V> { phase: V, inc: V, beta: V, al: V, ar: V, inc_t: V, beta_t: V, al_t: V, ar_t: V }

impl<V: Lanes> Osc<V> {
    #[inline(always)]
    fn load(b: &Bank, i: usize) -> Self {
        Osc {
            phase: V::load(&b.phase[i..]), inc: V::load(&b.inc[i..]), beta: V::load(&b.beta[i..]),
            al: V::load(&b.al[i..]), ar: V::load(&b.ar[i..]),
            inc_t: V::load(&b.inc_t[i..]), beta_t: V::load(&b.beta_t[i..]),
            al_t: V::load(&b.al_t[i..]), ar_t: V::load(&b.ar_t[i..]),
        }
    }
    #[inline(always)]
    fn store(&self, b: &mut Bank, i: usize) {
        self.phase.store(&mut b.phase[i..]);
        self.inc.store(&mut b.inc[i..]);
        self.beta.store(&mut b.beta[i..]);
        self.al.store(&mut b.al[i..]);
        self.ar.store(&mut b.ar[i..]);
    }
    /// One sample for W oscillators at once. Returns the oscillator output y in −1..1.
    #[inline(always)]
    fn step(&mut self, k: V, c: &SinConsts<V>) -> V {
        // one-pole smoothing toward the targets: x += k (t − x)
        self.inc = k.mul_add(self.inc_t - self.inc, self.inc);
        self.beta = k.mul_add(self.beta_t - self.beta, self.beta);
        self.al = k.mul_add(self.al_t - self.al, self.al);
        self.ar = k.mul_add(self.ar_t - self.ar, self.ar);
        // phase accumulator, wrapped to [−½, ½] without a branch
        self.phase = self.phase + self.inc;
        self.phase = self.phase - self.phase.round();
        // y = sin(φ + β sin φ): self phase modulation adds upper harmonics as β grows
        let s1 = fast_sin(self.phase, c);
        fast_sin(self.beta.mul_add(s1, self.phase), c)
    }
}

/// Render `len ≤ BLOCK` samples of the whole bank with lane type V and ADD them into out_l/out_r.
/// This is the whole SIMD pattern in one function (see the module docs).
#[inline(always)]
fn render_block<V: Lanes>(b: &mut Bank, out_l: &mut [f32], out_r: &mut [f32]) {
    let len = out_l.len().min(out_r.len()).min(BLOCK);
    let k = V::splat(b.k);                                   // (1) broadcast, outside every loop
    let c = SinConsts::<V>::new();
    let c1 = SinConsts::<f32>::new();
    let zero = V::splat(0.0);
    let mut acc_l = [zero; BLOCK];                           //     per-lane, per-sample sums
    let mut acc_r = [zero; BLOCK];
    let full = b.n / V::W * V::W;
    for i in (0..full).step_by(V::W) {                       // (2) loop by vector width
        let mut o = Osc::<V>::load(b, i);
        for s in 0..len {
            let y = o.step(k, &c);                           // (3) lane-parallel
            acc_l[s] = o.al.mul_add(y, acc_l[s]);
            acc_r[s] = o.ar.mul_add(y, acc_r[s]);
        }
        o.store(b, i);                                       // (4) store state back
    }
    let mut rem_l = [0f32; BLOCK];                           // (5) scalar remainder
    let mut rem_r = [0f32; BLOCK];
    for i in full..b.n {
        let mut o = Osc::<f32>::load(b, i);
        for s in 0..len {
            let y = o.step(b.k, &c1);
            rem_l[s] = o.al.mul_add(y, rem_l[s]);
            rem_r[s] = o.ar.mul_add(y, rem_r[s]);
        }
        o.store(b, i);
    }
    for s in 0..len {                                        // (4) reduce across lanes, once per sample
        out_l[s] += soft_limit(acc_l[s].reduce_add() + rem_l[s]);
        out_r[s] += soft_limit(acc_r[s].reduce_add() + rem_r[s]);
    }
}

impl Bank {
    pub fn new(n: usize) -> Bank {
        let z = || vec![TINY; n];
        // Spread initial phases (golden-ratio sequence) so partials that share a pitch do not
        // start in phase and spike.
        let phase = (0..n).map(|i| (i as f32 * 0.618_034).fract() - 0.5).collect();
        Bank { n, phase, inc: z(), beta: z(), al: z(), ar: z(), inc_t: z(), beta_t: z(), al_t: z(), ar_t: z(), k: 0.001 }
    }

    /// Add the next `out_l.len()` samples of the bank into out_l / out_r, with `kernel`.
    pub fn render(&mut self, kernel: Kernel, out_l: &mut [f32], out_r: &mut [f32]) {
        assert_eq!(out_l.len(), out_r.len());
        for (l, r) in out_l.chunks_mut(BLOCK).zip(out_r.chunks_mut(BLOCK)) {
            match kernel {
                Kernel::Scalar => render_block::<f32>(self, l, r),
                Kernel::Wide4 => render_block::<wide::f32x4>(self, l, r),
                Kernel::Wide8 => render_block::<wide::f32x8>(self, l, r),
                #[cfg(target_arch = "aarch64")]
                Kernel::Neon => render_block::<neon::F32x4>(self, l, r),
            }
        }
    }
}

// ─────────────────────────────── control rate: descriptor → targets ───────────────────────────────

/// The voice: a bank of NBINS partials plus the mapping from surface to targets.
pub struct Sonifier {
    pub bank: Bank,
    pub sr: f32,
    pub kernel: Kernel,
    dirs: Vec<Vec3>,
    on: bool,
    primed: bool,
    /// Samples rendered since the voice was switched off (to stop computing once silent).
    off_samples: f32,
    tau: f32,
}

impl Sonifier {
    pub fn new(sr: f32) -> Sonifier {
        Sonifier {
            bank: Bank::new(NBINS), sr, kernel: Kernel::best(), dirs: (0..NBINS).map(bin_dir).collect(),
            on: false, primed: false, off_samples: 0.0, tau: 0.05,
        }
    }

    /// Map a surface descriptor + knobs to per-oscillator targets (once per video frame). No
    /// allocation, so it is safe to call from the audio callback.
    pub fn set_targets(&mut self, d: &SurfaceDescriptor, p: &SonifyParams) {
        let b = &mut self.bank;
        self.tau = p.smoothing.max(1e-4);
        b.k = 1.0 - (-1.0 / (self.tau * self.sr)).exp();
        let on = p.enable && d.total > 0 && p.gain > 0.0;
        if on != self.on { self.off_samples = 0.0; }
        self.on = on;
        if !on {
            b.al_t.fill(TINY);
            b.ar_t.fill(TINY);
            return;
        }
        let total = d.total as f32;
        let mean_r = d.bins.iter().map(|x| x.count as f32 * x.radius).sum::<f32>() / total;
        let presence = (total / 1500.0).min(1.0);
        let gain = p.gain.max(0.0) * presence;
        let nyq = 0.45 * self.sr;
        for (i, bin) in d.bins.iter().enumerate() {
            let dir = self.dirs[i];
            let amp = (bin.count as f32 / total).sqrt();
            let h = 0.5 * (dir.y + 1.0);
            let rho = if bin.count > 0 { (0.5 + 1.5 * (bin.radius / mean_r.max(1e-6) - 1.0)).clamp(0.0, 1.0) } else { 0.5 };
            let x = (0.45 * h + 0.45 * rho + 0.1 * bin.rough.min(1.0)).clamp(0.0, 1.0);
            let mut note = p.base + p.span * x;
            if p.quantize {
                let n = note.round() as i32;
                note = (p.key + p.scale.snap(n - p.key)) as f32;
            }
            let hz = (440.0 * ((note - 69.0) / 12.0).exp2()).min(nyq);
            b.inc_t[i] = hz / self.sr;
            b.beta_t[i] = (0.35 * p.brightness.clamp(0.0, 1.0) * bin.heat).max(TINY);
            let pan = (dir.x * p.spread.clamp(0.0, 1.0)).clamp(-1.0, 1.0);
            let th = (pan + 1.0) * 0.25 * std::f32::consts::PI;
            let a = gain * amp;
            b.al_t[i] = (a * th.cos()).max(TINY);
            b.ar_t[i] = (a * th.sin()).max(TINY);
        }
        if !self.primed {
            // First real frame: start at the right pitch / colour (amplitude still fades in).
            b.inc.copy_from_slice(&b.inc_t);
            b.beta.copy_from_slice(&b.beta_t);
            self.primed = true;
        }
    }

    /// True once the voice has been off long enough (10 τ, −87 dB) that rendering is skipped.
    pub fn idle(&self) -> bool { !self.on && self.off_samples > 10.0 * self.tau * self.sr }

    /// Add the voice into out_l / out_r.
    pub fn render(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        if !self.on { self.off_samples += out_l.len() as f32; }
        if self.idle() { return; }
        self.bank.render(self.kernel, out_l, out_r);
    }
}

// ─────────────────────────────── bench ───────────────────────────────

/// `keep bench-sonify [--osc N] [--seconds S]`: oscillators × samples per second per kernel.
pub fn bench(n: usize, seconds: f32) {
    let sr = 48_000.0;
    let samples = (seconds * sr) as usize;
    println!("sonify bench: {n} oscillators x {samples} samples ({seconds} s at 48 kHz), block {BLOCK}");
    println!("{:<8} {:>10} {:>16} {:>9} {:>10}", "kernel", "time", "osc*samples/s", "speedup", "realtime");
    let mut base = None;
    for kernel in Kernel::all() {
        let mut bank = Bank::new(n);
        for i in 0..n {
            bank.inc_t[i] = (110.0 + 3.0 * i as f32) / sr;
            bank.beta_t[i] = 0.1;
            bank.al_t[i] = 0.5 / n as f32;
            bank.ar_t[i] = 0.5 / n as f32;
        }
        bank.k = 0.002;
        let (mut l, mut r) = (vec![0f32; samples], vec![0f32; samples]);
        bank.render(kernel, &mut l[..4800], &mut r[..4800]); // warm up
        let t0 = std::time::Instant::now();
        bank.render(kernel, &mut l, &mut r);
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(&l);
        let rate = n as f64 * samples as f64 / dt;
        let b = *base.get_or_insert(rate);
        println!("{:<8} {:>8.1}ms {:>16.3e} {:>8.2}x {:>9.0}x", format!("{kernel:?}"), dt * 1e3, rate, rate / b, seconds as f64 / dt);
    }
}

// ─────────────────────────────── tests ───────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn random_bank(n: usize, seed: u32) -> Bank {
        let mut b = Bank::new(n);
        let mut s = seed;
        let mut rnd = || { s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); (s >> 8) as f32 / (1u32 << 24) as f32 };
        for i in 0..n {
            b.inc_t[i] = (50.0 + 4000.0 * rnd()) / 48_000.0;
            b.beta_t[i] = 0.35 * rnd();
            b.al_t[i] = rnd() / n as f32;
            b.ar_t[i] = rnd() / n as f32;
            b.inc[i] = rnd() * 0.05;
        }
        b.k = 0.01;
        b
    }

    fn run(kernel: Kernel, n: usize, samples: usize) -> (Vec<f32>, Vec<f32>, Bank) {
        let mut b = random_bank(n, 7);
        let (mut l, mut r) = (vec![0f32; samples], vec![0f32; samples]);
        // odd chunk sizes exercise partial blocks too
        let mut at = 0;
        for chunk in [1usize, 37, 128, 300, 999].iter().cycle() {
            if at >= samples { break; }
            let e = (at + chunk).min(samples);
            b.render(kernel, &mut l[at..e], &mut r[at..e]);
            at = e;
        }
        (l, r, b)
    }

    #[test]
    fn fast_sin_error_bound() {
        let mut worst = 0f32;
        for i in 0..1_000_000 {
            let x = -3.0 + 6.0 * i as f32 / 1e6;
            let exact = (std::f64::consts::TAU * x as f64).sin();
            worst = worst.max((fast_sin(x, &SinConsts::new()) as f64 - exact).abs() as f32);
        }
        assert!(worst < 4e-6, "max error {worst}");
    }

    #[test]
    fn simd_kernels_match_scalar_with_remainders() {
        // 1027 = 128·8 + 3: not a multiple of 4 or 8, so every SIMD kernel runs a scalar tail.
        for n in [1usize, 3, 7, 8, 13, 1027] {
            let (sl, sr, sb) = run(Kernel::Scalar, n, 3000);
            for k in Kernel::all().into_iter().skip(1) {
                let (l, r, b) = run(k, n, 3000);
                let err = sl.iter().zip(&l).chain(sr.iter().zip(&r)).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(err < 2e-5, "{k:?} n={n}: max sample diff {err}");
                let perr = sb.phase.iter().zip(&b.phase).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(perr < 1e-3, "{k:?} n={n}: phase drift {perr}");
                assert!(l.iter().any(|&v| v != 0.0), "{k:?} n={n}: silent");
            }
        }
    }

    #[test]
    fn output_bounded_finite_and_no_denormals() {
        // Absurd targets: everything loud, then everything off for a long time.
        let n = 1024;
        let mut b = Bank::new(n);
        for i in 0..n { b.inc_t[i] = 440.0 / 48_000.0; b.al_t[i] = 1.0; b.ar_t[i] = 1.0; b.beta_t[i] = 0.35; }
        b.k = 0.01;
        for k in Kernel::all() {
            let (mut l, mut r) = (vec![0f32; 4800], vec![0f32; 4800]);
            b.render(k, &mut l, &mut r);
            assert!(l.iter().chain(&r).all(|v| v.is_finite() && v.abs() < 1.0), "{k:?} exceeds the limiter bound");
        }
        b.al_t.fill(TINY); b.ar_t.fill(TINY); b.beta_t.fill(TINY);
        let (mut l, mut r) = (vec![0f32; 48_000 * 4], vec![0f32; 48_000 * 4]);
        b.render(Kernel::best(), &mut l, &mut r);
        for v in [&b.al, &b.ar, &b.beta, &b.phase, &b.inc] {
            assert!(v.iter().all(|x| x.is_finite() && !x.is_subnormal()), "subnormal or non-finite state");
        }
        assert!(l.iter().chain(&r).all(|v| v.is_finite() && !v.is_subnormal()));
    }

    #[test]
    fn octahedral_roundtrip_and_bins() {
        for i in 0..NBINS {
            let d = bin_dir(i);
            assert!((d.length() - 1.0).abs() < 1e-5);
            assert_eq!(bin_of(d), i, "bin centre maps back to its bin");
        }
        // every bin gets a share of uniformly spread directions (no pole pinch)
        let mut hits = vec![0u32; NBINS];
        let n = 200_000;
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let a = i as f32 * 2.399_963;
            let rr = (1.0 - z * z).sqrt();
            hits[bin_of(Vec3::new(rr * a.cos(), rr * a.sin(), z))] += 1;
        }
        let (lo, hi) = (*hits.iter().min().unwrap() as f32, *hits.iter().max().unwrap() as f32);
        assert!(lo > 0.0 && hi / lo < 6.0, "bin solid angles {lo}..{hi}");
    }

    #[test]
    fn scale_snap() {
        let pent = Scale::named("minor_pentatonic").unwrap(); // 0 3 5 7 10
        assert_eq!(pent.snap(0), 0);
        assert_eq!(pent.snap(1), 0);
        assert_eq!(pent.snap(2), 3);
        assert_eq!(pent.snap(4), 3); // tie 3/5 goes down
        assert_eq!(pent.snap(11), 10);
        assert_eq!(pent.snap(-1), -2); // -2 ≡ 10
        assert_eq!(key_from_name("D"), Some(2));
        assert_eq!(key_from_name("Bb"), Some(10));
        assert_eq!(key_from_name("f#"), Some(6));
    }

    #[test]
    fn sphere_descriptor_maps_to_a_scale_chord() {
        // Beads on a sphere of radius 0.8 around (0.1, 0, 0), normals outward, medium heat.
        let c = Vec3::new(0.1, 0.0, 0.0);
        let mut beads = Vec::new();
        let n = 20_000;
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let a = i as f32 * 2.399_963;
            let rr = (1.0 - z * z).sqrt();
            let d = Vec3::new(rr * a.cos(), rr * a.sin(), z);
            beads.push((c + 0.8 * d, d, 0.5));
        }
        let desc = SurfaceDescriptor::from_beads(&beads, c);
        assert_eq!(desc.total, n as u32);
        assert!((Vec3::from(desc.centroid) - c).length() < 0.01, "{:?}", desc.centroid);
        let lit: Vec<&Bin> = desc.bins.iter().filter(|b| b.count > 0).collect();
        assert!(lit.len() > 900);
        assert!(lit.iter().all(|b| (b.radius - 0.8).abs() < 0.01 && b.rough < 0.01 && (b.heat - 0.5).abs() < 0.01));

        let p = SonifyParams { enable: true, key: 2, scale: Scale::named("minor_pentatonic").unwrap(), ..Default::default() };
        let mut s = Sonifier::new(48_000.0);
        s.set_targets(&desc, &p);
        for i in 0..NBINS {
            if desc.bins[i].count == 0 { continue; }
            let midi = 69.0 + 12.0 * (s.bank.inc_t[i] * 48_000.0 / 440.0).log2();
            let pc = (midi.round() as i32 - 2).rem_euclid(12);
            assert!([0, 3, 5, 7, 10].contains(&pc), "bin {i}: pitch class {pc} outside D minor pentatonic");
            assert!((midi - midi.round()).abs() < 1e-3);
        }
        // amplitudes are power-normalized: Σ (al² + ar²) = gain²
        let pow: f32 = (0..NBINS).map(|i| s.bank.al_t[i].powi(2) + s.bank.ar_t[i].powi(2)).sum();
        assert!((pow - p.gain * p.gain).abs() < 1e-3, "power {pow}");
        // a top bin sings higher than a bottom bin
        let top = bin_of(Vec3::Y);
        let bottom = bin_of(-Vec3::Y);
        assert!(s.bank.inc_t[top] > s.bank.inc_t[bottom]);
        // the voice renders, bounded
        let (mut l, mut r) = (vec![0f32; 9600], vec![0f32; 9600]);
        s.render(&mut l, &mut r);
        let peak = l.iter().chain(&r).fold(0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.01 && peak < 1.0, "peak {peak}");
        // switching off fades out and then goes idle
        s.set_targets(&desc, &SonifyParams { enable: false, ..p });
        let (mut l, mut r) = (vec![0f32; 48_000], vec![0f32; 48_000]);
        s.render(&mut l[..24_000], &mut r[..24_000]);
        s.render(&mut l[24_000..], &mut r[24_000..]);
        assert!(s.idle());
        assert!(l[23_000..].iter().all(|v| v.abs() < 1e-3));
    }
}
