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
mise exec -- cargo test               # 18 tests (GPU smoke tests skip if no Vulkan)

# live window, hot-reloads the script on save
mise exec -- ./target/release/keep run scripts/exemplar.lua

# offline render (synthesizes <out>.wav if --audio is missing or absent)
mise exec -- ./target/release/keep render scripts/exemplar.lua \
    --seconds 75 --fps 60 --size 1920x1080 --out out/keep-exemplar.mp4 [--audio a.wav] [--particles 750000]
```

The exemplar (75 s, 1080p60, ~55 s to render on an M5 Pro) is at `renders/keep-exemplar.mp4`
(not committed); stills are in `renders/stills/`.

## Controls (`keep run`)

| Key | Action |
| --- | --- |
| WASD / Q E | move camera (Shift = fast) |
| mouse drag | look |
| 1 / 2 / 3 (+Shift reverses) | add extra XW / YW / ZW rotation on top of the script |
| [ / ] | nudge the w slice |
| R | reset nudges |
| Esc | quit |

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
- `renderer.rs` — recording a frame: upload params, dispatch, pipeline barriers between compute
  and graphics, tonemap, then readback or present.
- `shaders.rs` / `build.rs` — GLSL to SPIR-V at build time, with a tiny `#include` inliner.
- `png.rs`, `tests.rs` — a dependency-free PNG writer and GPU smoke tests.

Shaders (`shaders/`): `common.glsl` is the std140 contract mirrored by `src/scene.rs`;
`field.glsl` is the 4D signed-distance library mirrored by `src/field.rs` (tests pin both);
`field.comp` places each seed at `(x, y, z, w_slice)`, maps it back through the inverse 4D
transform and keeps it if it lies near the surface; `particle.vert/.frag` draw each bead as an
analytic sphere; `tonemap.frag` does bloom, hue-preserving tonemapping, grading and grain.
4D math lives in `src/math4d.rs` (rotations in the six planes, affine 4D transforms).

## Known limitations

- The cloud is a point-sampled cross-section, so very thin slices (near the end of each sweep)
  briefly go almost empty; the exemplar has a couple of near-blank moments (~12 s, ~30 s) by design
  of the drop.
- The climax shapes are still fairly rounded blobs rather than crisply faceted forms.
- MoltenVK only has been tested (Apple GPU); other Vulkan drivers should work but are untried.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
