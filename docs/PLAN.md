# keep — 4D slice renderer

"keep" is a working name only. Standalone creative-computing repo: no dependency
on Bonsii, mission-control or any agent infrastructure.

## Intent
Part of an artistic practice around formal systems (repetition, variation,
recursion, cross-modal expression): invent a constrained mathematical system,
then explore what it can express. At once a learning project, an instrument, and
a possible substrate for installable works. Not a general art platform: make one
phenomenon compelling first; let reusable primitives emerge from that work.

Long-range method (NOT in scope yet): corpus/source → capture → formalization/
transduction → resonate/amplify/filter/transform/modulate/feedback → projection
(image, sound, animation, text, physical). Don't let it shape spike 1.

## Stack decisions
- Rust: intentional — depth in math, graphics, DSP, simulation, systems.
- Lua: the creative topcoat. Artistic iteration must never need a Rust rebuild.
- Vulkan (ash), chosen over wgpu deliberately: learning compute, buffers,
  sync and shaders is part of the point. Tradeoff: more code and MoltenVK quirks
  before the first image. If raw Vulkan creates friction that teaches nothing,
  raise it as an explicit decision — never switch silently.
- Aesthetic emerges from the math; no reference look to imitate.

A 4D implicit field F(x,y,z,w)=0 is a 3-manifold. Its intersection with our 3D
slice (w = w_slice) is a 2D surface, sampled by a fixed cloud of droplet-like
particles. As the 4D object is rotated, sheared and translated, the surface
appears, splits and vanishes inside the cloud.

## Per-particle (GPU compute)
p4 = (x, y, z, w_slice); q = A⁻¹(p4 − t); d = F(q)
- |d| < ε → visible; gradient → normal; curvature → roughness
- field value → temperature → blackbody RGB (emission, separate from reflection)

## Math
4D affine = Mat4 A + Vec4 t (no 5×5). Compose (A2,t2)∘(A1,t1) = (A2A1, A2t1+t2).
Six rotation planes: XY XZ XW YZ YW ZW. glam on the hot path.

## Split
Lua (mlua): what — shapes, transforms, animation, parameters, hot reload.
Rust: 4D math, scene state, bindings, Vulkan (ash).
Shaders: field eval (compute) + billboard droplets shaded as analytic spheres.

## Spike 1 (only this)
real-time window, free-fly camera through the particle volume, 1 cloud, hypersphere + one asymmetric primitive (duocylinder/tesseract),
4D affine transforms (6 plane rotations, shear, translate), fixed slice, black background, specular + blackbody emission,
Lua hot reload. No editor, timeline, audio, scene graph, refraction.

Done when: rotating/shearing/translating an asymmetric 4D object through XW/YW/ZW shows its 3D
cross-section emerging from a dark field of reflective/emissive particles.

## Later
CSG (union/intersection/difference/smooth_union), shear4/scale4, tesseract,
simplex, Clifford torus, 4D→3D projection mode, offline/video capture, live input, corpus/audio mapping.

## Platform
macOS via MoltenVK (Vulkan portability: enable VK_KHR_portability_enumeration).
