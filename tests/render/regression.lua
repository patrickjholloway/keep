-- regression.lua — the fixed scene the render regression tests (src/gpu/regression.rs) drive
-- through `keep render`'s per-frame path. Test data, not a showcase: change it and the
-- references in this folder must be re-blessed (see README.md).
--
-- Deterministic by construction: no math.random, no clock; everything is a function of t and
-- of the features the test passes in (fixed values per frame index, no audio).
-- What it exercises:
--   * two shapes, smooth-unioned: a duocylinder and a hypersphere offset in x and moving in w
--     (the offset breaks the x-mirror symmetry, so a flipped rotation sign is visible)
--   * a non-zero 4D rotation in the xw and yw planes, plus a little ordinary yz spin
--   * a w-slice that sweeps with t
--   * two glitch modules on at fixed settings: radial dispersion and the ring modulator;
--     dispersion also takes a mean_luma cable, so each frame depends on the one before it
--   * no camera: the default camera (offline::default_camera)

local H = hyper
local G = glitch

G.set("dispersion.depth", 24)
G.set("dispersion.mode", 0)          -- radial
G.set("dispersion.min", 4)
G.patch("mean_luma", "dispersion.depth", 40)
G.set("ring.depth", 0.35)            -- ring.freq / speed / chroma stay at their defaults

function update(t, f)
  return {
    transform = {
      H.rotate("xw", 0.6 + 0.4 * t),
      H.rotate("yw", 0.3 + 0.15 * t),
      H.rotate("yz", 0.1 * t),
    },
    field = {
      H.duocylinder { r1 = 0.85, r2 = 0.5 },
      H.hypersphere {
        r = 0.55, op = "smooth_union", k = 0.25,
        transform = { H.translate(0.45, 0, 0, 0.6 * math.sin(1.3 * t)) },
      },
    },
    w_slice         = -0.35 + 0.3 * t,
    surface_width   = 0.04 + 0.01 * f.bass,
    particle_radius = 0.016,
    temperature     = { 1500, 9000 + 3000 * f.high },
    reflectivity    = 0.6,
    exposure        = 1.1,
    palette         = 0.5 * math.sin(t),
    beat_rise       = 0.5 * math.exp(-4 * f.beat_phase),
  }
end
