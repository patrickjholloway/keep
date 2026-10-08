# keep — architecture

## The contract: `SceneParams` (src/scene.rs ⇄ shaders/common.glsl)

One `#[repr(C)]`, bytemuck-`Pod` struct, uploaded each frame as uniform `Scene`
(set 0, binding 0, std140). Every member is a vec4/mat4/uvec4, so Rust and GLSL layouts
match with no padding; a compile-time size assert guards drift. **Change both files together.**

| field | meaning |
|---|---|
| `inv_a: mat4`, `t: vec4` | global object transform, shader does `q = inv_a * (p4 - t)` (inv_a = A⁻¹, t = forward t) |
| `slice` | x w_slice · y surface_width (ε) · z particle_radius · w time |
| `material` | x temp_lo K · y temp_hi K · z reflectivity · w exposure |
| `audio` | x bass · y mid · z high · w onset (for shader flourishes) |
| `view`, `proj`, `cam_pos` | camera (proj is Vulkan-flipped y, depth 0..1); cam_pos.w = fov_y |
| `counts` | x particle_count · y prim_count |
| `prims[8]: Primitive` | 4D field: `kind_op` (PRIM_*, OP_*), `params`, local `inv_a`, `t` |

Data flow per frame:

```
audio::analyze ─► Features ─► Lua update(t, f) ─► SceneDesc ─► to_params(camera) ─► SceneParams ─► GPU
                                   (script.rs)      (scene.rs)                         (renderer.rs)
```

GPU buffers (set 0): binding 0 `Scene` UBO · binding 1 `Seeds` (vec4 per particle, fixed
cloud) · binding 2 `Droplets` (compute writes pos/size, normal/d, emission/visibility;
vertex reads). Frame = field.comp dispatch → buffer barrier → billboard draw (6 verts ×
N instances, additive, no depth) → readback (render) or present (run).

## Lua scene table (returned from `update(t, f)`)

```lua
{ transform = { {"rotate","xw",a}, {"shear","w","x",k}, {"translate",x,y,z,w}, {"scale",x,y,z,w} }, -- in order
  field = { {shape="hypersphere", r=1}, {shape="tesseract", half={x,y,z,w}},
            {shape="duocylinder", r1=1, r2=.6}, {shape="clifford", r1=1, r2=1, thickness=.1},
            -- every entry after the first may take op="union"|"smooth_union"(k=)|"intersect"|"subtract"
            -- and its own transform={...} },
  w_slice=0, surface_width=0.02, particle_radius=0.004, temperature={lo,hi},
  reflectivity=0.6, exposure=1, camera={eye={..}, target={..}, fov=deg} -- camera optional }
```
`f` = `{bass, mid, high, onset, rms, beat_phase}`. Script errors keep the last good scene.

## File ownership

| path | owns |
|---|---|
| `src/scene.rs` | **the contract**: SceneParams, GpuPrimitive, SceneDesc, Camera, packing |
| `shaders/common.glsl` | GLSL mirror of the contract + Droplet struct + PRIM_/OP_ ids |
| `src/math4d.rs` | Affine4 (A, t), Plane (6 rotation planes), rotation/shear/scale/translate/compose/inverse |
| `src/field.rs` | Shape / Op / Prim / Field; CPU SDF eval (tests); pack to GpuPrimitive |
| `src/audio.rs` | original procedural track → WAV (hound); STFT analysis → Features per frame (realfft) |
| `src/script.rs` | ScriptHost: mlua state, `hyper` helpers, notify hot reload, table → SceneDesc |
| `src/gpu/context.rs` | instance/device on MoltenVK (portability), queue, memory types, one-shot submit |
| `src/gpu/buffers.rs` | Buffer / Image allocation helpers |
| `src/gpu/target.rs` | Offscreen (RGBA8 + readback) and Swapchain targets |
| `src/gpu/passes.rs` | descriptor bindings, FieldPass (compute), ParticlePass (dynamic rendering) |
| `src/gpu/renderer.rs` | Renderer: resources + frame recording, `render_offscreen` / `render_present` |
| `shaders/field.comp` | per-particle 4D field eval, gradient normal, blackbody emission |
| `shaders/particle.{vert,frag}` | billboards shaded as analytic spheres (specular + fresnel + emission) |
| `src/capture.rs` | Encoder: raw RGBA → ffmpeg stdin (libx264 yuv420p) + AAC audio mux |
| `src/camera.rs` | FlyCamera (WASD/QE + mouse) |
| `src/app.rs` | `keep run`: winit loop, hot reload, camera, XW/YW/ZW + w_slice nudges |
| `src/offline.rs` | `keep render`: synth/analyze audio, deterministic frame loop, encode |
| `src/main.rs` | CLI: `run`, `render`, `probe` |
| `build.rs` | GLSL → SPIR-V via naga (inlines `#include "common.glsl"`) |

## Toolchain notes
- Always `mise exec -- cargo …` / `mise exec -- ffmpeg …` (Rust 1.90, MoltenVK 1.4.2, conda:ffmpeg 9.0.2 with libx264/videotoolbox).
- MoltenVK loaded directly does **not** offer `VK_KHR_portability_enumeration` (loader ext): enable only if listed. Device must enable `VK_KHR_portability_subset`.
- naga GLSL frontend: no `#include` (build.rs inlines), no `writeonly` storage buffers.
