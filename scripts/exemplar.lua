-- exemplar.lua — "Clifford Bloom": the showcase choreography rendered to video.
--
-- A duocylinder (product of two discs) smooth-unioned with a Clifford torus (the flat torus
-- |xy| = r1, |zw| = r2 living on a 3-sphere). Neither shape fits in 3D; we only ever see the
-- 3D cross-section at w = w_slice, rendered as a cloud of glowing droplets.
--
-- Choreography over a ~60 s track:
--   * XW / YW / ZW rotations (the "impossible" ones that turn the object through the 4th axis)
--     advance steadily and get a kick on every beat, scaled by bass.
--   * w_slice slowly sweeps through the object in long "builds" (triangle wave), so the slice
--     opens, morphs and closes like a flower.
--   * temperature spikes (hotter / bluer) on onsets, decays between them.
--   * the camera slowly orbits and bobs.
--
-- update(t, f): t seconds; f = { bass, mid, high, onset, rms, beat_phase } (all ~0..1)

local H = hyper

-- slow sections: 0 = calm intro, 1 = full energy; a 16 s cosine breath
local function section(t) return H.ease(H.tri(t + 4, 32)) end

function update(t, f)
  local energy = section(t)
  local kick   = H.pulse(f.beat_phase, 7) * f.bass   -- spike on the beat, weighted by bass

  -- 4D rotation angles: steady drift + beat kicks. Kicks are small so motion stays smooth.
  local xw = t * 0.21 + 0.35 * kick
  local yw = t * 0.13 + 0.25 * f.mid * energy
  local zw = t * 0.09 + 0.20 * kick
  local yz = t * 0.05

  -- slice sweep: long build through the object, plus a breathing wobble with rms
  local sweep = H.lerp(-0.9, 0.9, H.ease(H.tri(t, 24)))
  local w_slice = sweep + 0.08 * math.sin(t * 1.7) * f.rms

  -- the clifford torus drifts along w relative to the duocylinder so the blend morphs
  local cliff_w = 0.4 * math.sin(t * 0.31)

  -- temperature: cool deep reds normally, white-blue flashes on onsets
  local hot = 6500 + 9000 * f.onset + 2500 * f.high
  local cold = 900 + 600 * energy

  -- camera: slow orbit (one revolution per ~90 s), gentle vertical bob
  local orbit = t * H.tau / 90
  local dist = 5.2 - 0.6 * energy
  local eye = { dist * math.sin(orbit), 1.2 * math.sin(t * 0.13), dist * math.cos(orbit) }

  return {
    transform = {
      H.rotate("xw", xw),
      H.rotate("yw", yw),
      H.rotate("zw", zw),
      H.rotate("yz", yz),
      H.shear("x", "w", 0.18 * f.onset),        -- onsets smear w into x for a split second
      H.scale(1.0 + 0.08 * kick),               -- pulse in size on the beat
    },
    field = {
      H.duocylinder { r1 = 1.05, r2 = 0.55 + 0.1 * energy },
      H.clifford {
        r1 = 0.9, r2 = 0.9, thickness = 0.12 + 0.06 * f.bass,
        op = "smooth_union", k = 0.25 + 0.25 * energy,
        transform = { H.rotate("xz", t * 0.11), H.translate(0, 0, 0, cliff_w) },
      },
      -- a small hypersphere carved out of the centre on loud passages: opens a void
      H.hypersphere {
        r = 0.25 + 0.35 * energy * f.rms,
        op = "subtract",
      },
    },
    w_slice         = w_slice,
    surface_width   = 0.016 + 0.025 * f.rms + 0.01 * kick,
    particle_radius = 0.0035 * (1 + 0.8 * f.onset),
    temperature     = { cold, hot },
    reflectivity    = 0.55 + 0.3 * f.high,
    exposure        = 0.9 + 0.5 * energy + 0.4 * f.onset,
    camera          = { eye = eye, target = { 0, 0, 0 }, fov = 48 - 4 * energy },
  }
end
