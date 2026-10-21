# Render regression references

Test data for `src/gpu/regression.rs` (a `#[cfg(test)]` module of the `keep` binary; there is
no `.rs` file here, so cargo builds no extra test target from this folder).

## What the frames are

`regression.lua` is a fixed scene: a duocylinder smooth-unioned with an offset hypersphere
that moves in w, rotated in the xw, yw and yz planes, a w-slice sweeping with time, and two
glitch modules on at fixed settings (radial dispersion with a `mean_luma` cable, ring
modulator). It uses no `math.random` and no clock. The test feeds it fixed features (constant
levels, `beat_phase` cycling every 15 frames; no audio) and renders frames 0..90 at 30 fps,
in order, through `offline::render_frame` — the same per-frame step `keep render` runs — with
`keep render`'s particle cloud (1,000,000 particles, seed 1) and default camera.

| file | frame | t |
|---|---|---|
| `frame-000.png` | 0 | 0.0 s |
| `frame-045.png` | 45 | 1.5 s |
| `frame-090.png` | 90 | 3.0 s |

Format: 320×180 RGBA8 PNG as written by `src/gpu/png.rs` (stored deflate blocks, filter 0;
~230 KB each). The test reads them with a small reader that accepts exactly that form and
refuses anything else, so do not re-save them with an image editor or optimiser.

## The comparison

Per frame, in 8-bit levels over R, G, B and A:

- mean absolute error must be ≤ 0.25, and
- at most 0.1% of pixels may have a channel that differs by more than 8 levels.

Every frame must also have 2–50% lit pixels (R+G+B > 150) before it is compared or blessed,
so a black frame can never become a reference.

Why these numbers (measured on an Apple M5 Pro through MoltenVK 1.4.2): two renders of the
same frames, in one process and across six separate `cargo test` runs, are byte-identical
(noise floor 0). The smallest effect of the named wrong implementations the suite must catch
was a mean error of 0.42–0.69 (a swapped ring-modulator parameter, frames 45 and 90) and
0.63 with 0.20% of pixels over 8 levels (the tonemap's teal shadow lift applied in the wrong
channel order). The tolerance sits below both with a 1.7–2.5× margin.

What it does not cover: changes smaller than that are let through — for example moving the
ring modulator's default chroma from 0.6 to 0.5 rad changes these frames by a mean of 0.02
levels (max 5) and passes. Another Apple GPU generation, macOS or MoltenVK version may differ
by small float error; the tolerance leaves room for scattered single-level dither changes but
was not measured on any other machine. Other GPU vendors are out of scope. Without MoltenVK
(`KEEP_VULKAN_LIB` unset or pointing at a missing file) the GPU tests print `skipping: …` and
pass, so CI without a GPU checks nothing here.

## Running and blessing

```sh
mise exec -- cargo test gpu::regression     # compare (and the determinism test)
KEEP_BLESS=1 mise exec -- cargo test render_regression   # rewrite the references, then compare
```

A missing reference fails and prints the bless command. A failing frame writes
`target/render-regression/frame-NNN.actual.png` and `frame-NNN.diff.png` (|difference| × 8,
gitignored) and names the frame, both metric values and the tolerances. Bless only after
looking at the actual frame, and say in the commit why the picture changed. Changing
`regression.lua`, the frame list or the tolerances means re-blessing.
