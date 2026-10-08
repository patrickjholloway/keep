# keep — 4D slice renderer

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
1 camera, 1 cloud, hypersphere + one asymmetric primitive (duocylinder/tesseract),
6 plane rotations, fixed slice, black background, specular + blackbody emission,
Lua hot reload. No editor, timeline, audio, scene graph, refraction.

Done when: rotating an asymmetric 4D object through XW/YW/ZW shows its 3D
cross-section emerging from a dark field of reflective/emissive particles.

## Later
CSG (union/intersection/difference/smooth_union), shear4/scale4, tesseract,
simplex, Clifford torus, 4D→3D projection mode, capture to video.

## Platform
macOS via MoltenVK (Vulkan portability: enable VK_KHR_portability_enumeration).
