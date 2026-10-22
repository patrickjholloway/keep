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

- no pixel may have a channel that differs by more than 8 levels (`PIXEL_LEVELS` = 8,
  `DIFFERING_TOLERANCE` = 0 pixels), and
- the mean absolute error must be ≤ 0.25 (`MAE_TOLERANCE`).

Every frame must also have 2–50% lit pixels (R+G+B > 150) before it is compared or blessed,
so a black frame can never become a reference.

Why these numbers (measured on an Apple M5 Pro through MoltenVK 1.4.2): two renders of the
same frames, in one process and across separate `cargo test` runs, are byte-identical (noise
floor 0), so on this machine one pixel more than 8 levels off is a changed picture, not noise.

- The pixel check catches small, local changes. Swapping the dispersion module's `angle` and
  `min` parameters moves 1, 3 and 1 pixels (frames 0, 45, 90) by up to 15, 31 and 17 levels
  at a mean error of only 0.0009–0.0013. The earlier tolerance (0.1% of pixels, ~57 per
  frame) let that through.
- The mean error catches broad, faint changes. Doubling the tonemap's grain moves every pixel
  by at most 1 level and is caught only by the mean: 0.2786–0.2802 against 0.25, a 1.1×
  margin, so the mean tolerance cannot be raised past ~0.27 without losing it.
- Every other named wrong implementation in the suite also has pixels over 8 levels. The
  faintest, the tonemap's teal shadow lift in the wrong channel order, has 0.20–0.25% of its
  pixels off by up to 10 levels at a mean error of 0.63.

What it does not cover: a change that keeps every pixel within 8 levels and the mean under
0.25 is let through. For example, moving the ring modulator's default chroma from 0.6 to
0.5 rad changes these frames by a mean of 0.02 levels (max 5) and passes. In the other
direction there is no headroom for float error: a ±1-level change over the whole frame, such
as a grain hash that rounds differently, already exceeds the mean-error tolerance.

**The references are per-machine.** They were blessed on an Apple M5 Pro through MoltenVK
1.4.2. On another GPU, driver, macOS or MoltenVK version the test is expected to fail. To move
them, run the test once without `KEEP_BLESS` so each failing frame writes its actual and diff
images to `target/render-regression/`, look at those, then re-bless with
`KEEP_BLESS=1 mise exec -- cargo test render_regression` and review the new references before
committing them. Other GPU vendors are out of scope. Without MoltenVK (`KEEP_VULKAN_LIB` unset
or pointing at a missing file) the GPU tests print `skipping: …` and pass, so CI without a GPU
checks nothing here.

## Running and blessing

```sh
mise exec -- cargo test gpu::regression     # compare (and the determinism test)
KEEP_BLESS=1 mise exec -- cargo test render_regression   # rewrite the references, then compare
```

A missing reference fails and prints the bless command. A failing frame writes
`target/render-regression/frame-NNN.actual.png` and `frame-NNN.diff.png` (|difference| × 8,
gitignored) and names the frame, every metric value and the tolerances. Bless only after
looking at the actual frame, and say in the commit why the picture changed. Changing
`regression.lua`, the frame list or the tolerances means re-blessing.
