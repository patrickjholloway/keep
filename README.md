# keep

A 4D audio-reactive visualizer. A Lua script describes a shape in **four** dimensions (hyperspheres,
tesseracts, duocylinders, Clifford tori, combined with smooth unions) and moves a 3D "slice" through
it along the w axis. What you see is that 3D cross-section, drawn as hundreds of thousands of glowing
beads, while 4D rotations (xw / yw / zw planes) make it bulge, split and re-form in ways no 3D
object can. An original procedurally synthesized track (no copyrighted music) drives the motion.

Rendering is Vulkan (via `ash` on MoltenVK), shaders are GLSL compiled to SPIR-V at build time by
`build.rs` (naga), and offline renders go straight to H.264 + AAC through ffmpeg.

## Run

```sh
mise install                          # rust, ffmpeg, MoltenVK; sets KEEP_VULKAN_LIB
mise exec -- cargo build --release
mise exec -- cargo test               # 35 tests (GPU tests skip if no Vulkan)

# live window with music, hot-reloads the script on save. Without --audio it plays the
# exemplar track (synthesized to out/keep-exemplar.wav on first run, 75 s, looped).
mise exec -- ./target/release/keep run scripts/exemplar.lua [--audio a.flac] [--frames N]
# or react to the default microphone instead (realtime FFT, same features)
mise exec -- ./target/release/keep run scripts/exemplar.lua --mic

# --audio accepts WAV, FLAC, MP3, OGG/Vorbis and M4A/AAC (decoded in-process by symphonia;
# analysis uses a mono mixdown, playback keeps stereo and resamples to the device rate).
# Only a missing .wav path is synthesized; any other missing or unknown file is an error.
# offline render (synthesizes <out>.wav if --audio is missing or absent; ffmpeg muxes the
# original --audio file as AAC)
mise exec -- ./target/release/keep render scripts/exemplar.lua \
    --seconds 75 --fps 60 --size 1920x1080 --out out/keep-exemplar.mp4 [--audio a.wav] [--particles 750000]
```

The exemplar (75 s, 1080p60, ~95 s to render on an M5 Pro, glitch chain included) is at `renders/keep-exemplar.mp4`
(not committed); stills are in `renders/stills/`.

## Controls (`keep run`)

| Key | Action |
| --- | --- |
| WASD / Q E | move camera (Shift = fast) |
| mouse drag | look |
| 1 / 2 / 3 (+Shift reverses) | add extra XW / YW / ZW rotation on top of the script |
| [ / ] | nudge the w slice |
| R | reset nudges |
| Space | pause / resume the music (visuals freeze with it) |
| ← / → | seek −5 s / +5 s (wraps around the loop) |
| M | mute (visuals keep reacting) |
| Esc | quit |

In `keep run` the scene clock `t` and the features `f` come from the audio playback position:
the track is analyzed up front exactly like `keep render` (at 60 features/s) and looked up,
interpolated, at the current playback time, so the shape and the glitch chain stay locked to the
sound through pauses and seeks. With `--mic` there is no transport; `t` is wall-clock time and the
features come from a streaming analyzer (levels normalized against a ~10 s running peak, tempo
re-estimated every second from the last 8 s of onsets).

## Lua API

A script defines `update(t, f)` and returns a scene table each frame.
`f` holds audio features, all roughly 0..1: `bass`, `mid`, `high`, `onset`, `rms`, `beat_phase`
(runs 0..1 across each beat).

The `hyper` prelude provides:

- shapes: `hyper.hypersphere{r}`, `hyper.tesseract{half}`, `hyper.duocylinder{r1, r2}`,
  `hyper.clifford{...}`; each takes `op` (`union`, `smooth_union`, `subtract`, `intersect`), `k`
  (smooth-union softness) and `transform`.
- transforms (lists applied in order): `hyper.rotate("xw", a)` (any of xy xz xw yz yw zw),
  `hyper.translate(x,y,z,w)`, `hyper.scale(...)`, `hyper.shear(dst, src, k)`.
- helpers: `clamp`, `lerp`, `smoothstep`, `pulse(phase, sharp)`, `ease`, `tri(t, period)`, `tau`.

Scene keys returned: `field` (list of shapes), `transform` (whole-object 4D transform), `w_slice`,
`surface_width`, `particle_radius`, `temperature` (`{cold, hot}` in kelvin), `exposure`,
`reflectivity`, `palette` (-1 amber .. 0 teal .. +1 magenta), `beat_rise` (beat ring trigger),
`camera` (`eye`, `target`, `fov`, `roll`).

### Glitch patch bay (`glitch`)

Image-plane effects on the rendered frame (see [Glitch modules](#glitch-modules)), patched like a
modular synth. State persists across frames until changed; Rust resolves it every frame as

    value(param) = base(param) + Σ over cables into param of (gain · source + offset)   -- then clamped

- `glitch.set(param, v)` — base value (default = bypass / identity). Complex params take two
  numbers: `glitch.set("warp.a", re, im)` (or set `warp.a.re` / `warp.a.im` separately).
- `glitch.patch(src, param, gain, offset)` — add a cable; patching the same `src`→`param`
  again replaces it. `glitch.unpatch(src, param)`, `glitch.clear()`.
- `glitch.lfo(name, shape, rate_hz [, phase])` — a unipolar 0..1 oscillator source;
  shape `sine` | `tri` | `saw` | `square`.
- `glitch.source(name, v)` — a script-computed source (envelopes, section gates, ...).
- Built-in sources: `bass mid high onset rms beat_phase` (the audio features), `mean_luma`
  (mean tone-compressed luminance of the previous frame, an envelope follower on the
  picture itself) and `one` (constant 1).

Unknown parameter or source names are script errors (the last good frame is kept).

| param | default | meaning |
| --- | --- | --- |
| `sync.depth` · `freq` · `speed` · `block` | 0 · 24 · 1.5 · 0.35 | max delay (fraction of a line) · bands per frame · re-roll rate (Hz) · fraction of bands that tear |
| `ring.depth` · `freq` · `speed` · `chroma` | 0 · 3.37 · 0.5 · 0.6 | 0..1 · carrier cycles per scanline · drift (Hz) · R/G/B carrier phase offset (rad) |
| `smear.depth` · `feedback` · `dir` · `axis` | 0 · 0.95 · 1 · 0 | trail mix · IIR coefficient a · +1 forward / −1 back · 0 rows, 1 columns |
| `dispersion.depth` · `mode` · `angle` · `min` | 0 · 0 · 0 · 0 | px of spread at 1080p (radial: peaks near the silhouette) · 0 radial / 1 directional · direction (rad) · minimum split (px) |
| `warp.amount` · `zoom` · `twist` · `radius` · `a b c d` (complex) | 0 · 1 · 0 · 0.6 · 1, 0, 0, 1 | blend toward the Möbius map · output-plane scale · swirl at the centre (rad, × amount) · swirl falloff radius · coefficients |
| `spectral.mix` · `lo` · `hi` · `atten` · `phase` · `seed` · `soft` · `outside` | 0 · .02 · .12 · 1 · 0 · 0 · .01 · 1 | wet mix · band (cycles/px) · gain cut in band · ghost-echo strength · echo direction · band edge · gain outside band |
| `crush.depth` · `hold` | 0 · 1 | quantization (0 off .. 1 = 2 levels) · sample-and-hold block (px) |

```lua
glitch.lfo("wobble", "sine", 0.25)
glitch.patch("onset", "sync.depth", 0.06)          -- tear on hits
glitch.patch("wobble", "ring.depth", 0.3, 0.05)    -- breathing carrier
glitch.patch("mean_luma", "smear.depth", -1.5)     -- bright frames smear less
glitch.set("warp.c", 0, -0.4); glitch.set("warp.amount", 0.5)
```

## How the audio drives it (scripts/exemplar.lua)

| Audio | Drives |
| --- | --- |
| bass × beat envelope ("kick") | extra XW and ZW rotation — the shape lurches through hyperspace on the beat; bubble size |
| onset (strong) | slice jumps ±0.15 in w for 0.3 s (the shape snaps to a different cross-section); hotter blue-white highlights; camera push-in |
| beat rise | expanding indigo ring in the tonemap pass |
| beat_phase | smooth-union `k`: bubbles snap apart on the beat and melt together between beats |
| mid | YW rotation, scaled by the section energy |
| high | hot colour temperature |
| rms | small wobble of the slice position |
| bass | outline brightness and back-hemisphere dimming |

Time alone drives the 9 s slice sweep (shapes are born, swell and vanish), the slow camera orbit
and the palette drift, which follows the slice position.

## Glitch modules

After the particles are drawn into the HDR image and before the tonemap, an optional chain of
compute passes (`shaders/glitch_*.comp`, recorded by `src/gpu/glitch.rs`) treats the frame
two ways: as an **analog video signal** — all scanlines concatenated, sample index
`s = y·W + x` — and as a **2D plane**. The 4D math and the particles are never touched. Each
module ping-pongs between the HDR image and a scratch image of the same size; a module whose
depth / mix / amount is 0 records no work at all. Order: sync → ring → smear → dispersion →
warp → spectral → crush. Each shader opens with a longer explanation of its math.

1. **Sync slip** (`glitch_sync.comp`) — horizontal tearing as a *time delay on the signal*:
   `out(s) = in(s − d(y))`. Because `s` continues across line ends, a delay larger than `x`
   pulls pixels from the end of the previous line — true raster wrap-around. `d` is constant
   per band of lines: bands whose hash exceeds `1 − block` jump by a random fraction of a line,
   re-rolled `speed` times per second, plus a slow hsync wobble; R and B get ±4% different
   delays (composite colour misregistration).
2. **Ring mod** (`glitch_ring.comp`) — multiply by a carrier running along the signal,
   `c(s) = cos(2π·freq·s/W + 2π·speed·t + φ_ch)`. A product in space is a convolution in
   frequency, so every spatial frequency `f` moves to `f ± f_c` (sidebands): fine structure
   becomes moiré stripes. A non-integer `freq` starts each line at a new carrier phase, so the
   stripes lean diagonally. Light can't be negative, so the gain is `max(mix(1, c, depth), 0)`:
   depth 1 is true ring mod, less is amplitude modulation. Scaling RGB by one scalar scales
   luminance and keeps hue; `chroma` offsets the carrier per channel into colour fringes.
3. **Smear** (`glitch_smear.comp`) — a one-pole IIR low-pass along each scanline (or column, `axis` 1),
   `y[n] = (1 − a)·x[n] + a·y[n−1]`. Impulse response `(1 − a)·aⁿ`: unit DC gain, trail time
   constant `−1/ln a` pixels (a = 0.95 → 20 px, 0.99 → 100 px). Being recursive it is
   inherently serial, so it is one compute thread per row (or column). Only the history above the input is
   boosted (by `1/√(1−a)`), so flat areas pass unchanged, beads stay crisp and grow comet tails.
4. **Dispersion** (`glitch_disp.comp`) — a prism: refractive index follows Cauchy's
   `n(λ) ≈ A + B/λ²`, so the displacement is `k(λ) = (1/λ² − 1/550²)/(1/400² − 1/700²)` times
   `depth` along either the radial direction (lateral chromatic aberration: nothing at the
   centre, most at the edges) or a fixed angle. Wavelengths 400–700 nm are sampled evenly in
   displacement (7 to 64 taps, enough to keep neighbouring taps ≤ 1.5 px apart), jittered per
   pixel, and weighted by Gaussian R/G/B responses (normalized per channel, so white stays white): edges
   spread into continuous rainbows rather than three ghost copies.
5. **Conformal warp** (`glitch_warp.comp`) — each pixel is a complex `z` (centred, aspect
   correct, ±1 over the height) and the image is pushed through the Möbius map
   `f(z) = (az + b)/(cz + d)`. Möbius maps are conformal (angles preserved: beads stay round)
   and send circles to circles. The shader maps *backward*: each output pixel `w` samples the
   input at `f⁻¹(w) = (dw − b)/(−cw + a)`, so there are no holes. `warp.amount` blends
   `a, b, c, d` with the identity (a plain lerp when that stays invertible, else the matrix
   power `M^amount`, so e.g. a rotation preset turns by `amount·θ` instead of collapsing). The exemplar uses an *elliptic* map with fixed points `±p`
   (a rotation conjugated by `(z − p)/(z + p)`): `a = d = 1`, `b = −i p tan(θ/2)`,
   `c = −i tan(θ/2)/p` — the picture swirls around two still points; since the centre is
   magnified by `1 + tan²(θ/2)`, the same envelope is also patched into `warp.zoom` (capped at 1.15). On top, `warp.twist` rotates the lookup by `twist·amount·exp(−r²/radius²)`, a swirl that wrings the form around its centre.
6. **Spectral gate** (`glitch_spectral.comp`) — the frame is box-filtered down to a 1024×512 grid,
   mirrored into a 2048×1024 grid (even extension: no wrap seam at the frame edges), and
   2D-FFT'd with a radix-2 **Stockham** FFT (one dispatch per butterfly stage, ping-ponging two
   buffers, output in natural order: no bit reversal; 11 + 10 stages each way). Per bin, the
   radial frequency `|f|` (cycles per grid pixel, aspect-corrected so the band is a circle in
   the image) selects a soft band `[lo, hi]`; the band is scaled by
   `1 − atten` and its phase bent by `θ(f) = phase·π·sin(2π f·D)`. By the Jacobi–Anger
   expansion `e^{iA sin x} = Σ Jₙ(A) e^{inx}`, that phase makes copies of the band's content
   shifted by `n·D` with Bessel weights — multipath ghost echoes of fine detail only, while the
   broad form is untouched; `seed` picks the echo direction. Gain is even and phase odd in `f`,
   so the inverse stays real. Only the *change* is upsampled and added back
   (`out = orig + mix·(up(filtered) − up(down(orig)))`), so the downsampling never blurs the
   picture; that change is gated by local brightness, so echoes stay on or near the form
   instead of rippling across empty black. Because the FFT sees the mirrored frame, echoes reflect at the frame edges
   instead of wrapping to the opposite edge. `src/glitch.rs` has a CPU twin of the algorithm,
   tested against a naive DFT.
7. **Bitcrush / sample-and-hold** (`glitch_crush.comp`) — decimation holds one sample per
   `hold × hold` block (a lower sample rate, in 2D); quantization snaps brightness to
   `2^(8 − 7·depth)` mid-tread levels (black stays 0) (a lower bit depth); luminance is quantized and RGB rescaled with it, so the hue survives, in the tone-compressed domain `c/(1 + c)`, so
   unbounded HDR values posterize evenly.

The `mean_luma` source comes from `glitch_stats.comp`: one invocation averages a 64×36 grid of
the clean frame, the CPU reads it after the fence and feeds it to the next frame — one frame of
latency, like an envelope follower patched back into a synth.

How the exemplar plays them (`choreograph_glitch` in `scripts/exemplar.lua`): the intro is
clean; in the groove, strong onsets fire ~0.15 s sync-slip bursts and kicks push radial
dispersion; the build grows smear and a rising ring-mod carrier and ends with an 8th-note
bitcrush/sample-and-hold stutter; in the drop every kick also fires a short vertical smear burst; the drop downbeat opens the Möbius twist and the spectral
ghosting, which relax over two bars and return smaller every 4 bars; every 8 bars the drop ends
in a half-bar stutter; bars 28–30 are a clean breather, and the outro is clean.

## What each Vulkan file teaches

Read `src/gpu/` in this order:

- `context.rs` — instance creation with `VK_KHR_portability_enumeration` (needed for MoltenVK),
  physical device pick, logical device with `portability_subset`, queue.
- `buffers.rs` — Vulkan's object-vs-memory split: create a buffer/image, query requirements,
  pick a memory type, bind. Host-visible uniforms vs device-local storage buffers.
- `target.rs` — where frames land: an offscreen RGBA16F HDR image + RGBA8 image + staging buffer
  for readback (render mode), or a swapchain (run mode).
- `passes.rs` — the three passes: a compute pass (`field.comp`, one invocation per particle),
  an additive instanced billboard pass with no vertex buffers (dynamic rendering), and a
  fullscreen tonemap pass.
- `glitch.rs` — a chain of compute passes over storage images: two descriptor sets that differ
  only in which image is source vs destination (ping-pong), a shared pipeline layout with a
  push constant, compute→compute barriers, and conditional recording (bypass = no commands).
- `renderer.rs` — recording a frame: upload params, dispatch, pipeline barriers between compute
  and graphics, glitch chain, tonemap, then readback or present.
- `shaders.rs` / `build.rs` — GLSL to SPIR-V at build time, with a tiny `#include` inliner.
- `png.rs`, `tests.rs` — a dependency-free PNG writer and GPU smoke tests.

Shaders (`shaders/`): `common.glsl` is the std140 contract mirrored by `src/scene.rs`;
`field.glsl` is the 4D signed-distance library mirrored by `src/field.rs` (tests pin both);
`field.comp` places each seed at `(x, y, z, w_slice)`, maps it back through the inverse 4D
transform and keeps it if it lies near the surface; `particle.vert/.frag` draw each bead as an
analytic sphere; `glitch_*.comp` are the image-plane glitch modules (contract in `glitch_common.glsl` mirrored
by `src/glitch.rs`); `tonemap.frag` does bloom, hue-preserving tonemapping, grading and grain.
4D math lives in `src/math4d.rs` (rotations in the six planes, affine 4D transforms).

## Known limitations

- The cloud is a point-sampled cross-section, so very thin slices (near the end of each sweep)
  briefly go almost empty; the exemplar has a couple of near-blank moments (~12 s, ~30 s) by design
  of the drop.
- The climax shapes are still fairly rounded blobs rather than crisply faceted forms.
- The glitch chain is not audio-reactive in `keep run` (live mode has no audio input yet), but
  LFOs, `mean_luma` and script sources work there.
- MoltenVK only has been tested (Apple GPU); other Vulkan drivers should work but are untried.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
