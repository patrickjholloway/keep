-- exemplar.lua — "Clifford Bloom": the showcase choreography rendered to video.
--
-- A duocylinder (product of two discs) smooth-unioned with a Clifford torus, plus three small
-- hyperspheres orbiting through w. Nothing here fits in 3D; we only see the 3D cross-section at
-- w = w_slice, drawn as a cloud of glowing droplets.
--
-- What the viewer should read:
--   * the slice sweeps fully OUT of the object (empty frame) and back in: the shape is born,
--     morphs, pinches apart and dies — that is the 4th dimension, not a 3D tumble.
--   * the "bubbles" are hyperspheres moving along w: their slices appear as points, swell,
--     merge with the main body and vanish.
--   * rotation is mostly in XW / ZW (through the 4th axis); YZ (ordinary 3D spin) is slow.
--   * on beats: shapes snap apart (small smooth-k) then melt together (large k); onsets flash.
--
-- update(t, f): t seconds; f = { bass, mid, high, onset, rms, beat_phase } (all ~0..1)

local H = hyper

-- slow sections: 0 = calm intro, 1 = full energy
local function section(t) return H.ease(H.tri(t + 4, 32)) end

-- State kept across frames (offline rendering calls update() in frame order).
local jump_target, jump, last_onset, last_t = 0, 0, 0, 0
local push_t, last_push = -10, 0

function update(t, f)
  local energy = section(t)
  local dt = math.max(t - last_t, 0); last_t = t
  -- beat envelope: ~150 ms decay assuming ~0.5 s beats (beat_phase runs 0..1 per beat)
  local kick = H.pulse(f.beat_phase, 3.3) * f.bass

  -- bass-triggered slice jumps: each strong onset flips the offset to +-0.15, eased toward it.
  if f.onset > 0.5 and last_onset <= 0.5 then
    jump_target = (jump_target > 0) and -0.15 or 0.15
  end
  last_onset = f.onset
  jump = jump + (jump_target - jump) * (1 - math.exp(-dt * 10))

  -- 4D rotation: through-w planes dominate, ordinary 3D spin (yz) is slow.
  local xw = t * 0.32 + 0.5 * kick
  local yw = t * 0.10 + 0.25 * f.mid * energy
  local zw = t * 0.26 + 0.35 * kick
  local yz = t * 0.015

  -- slice sweep: 9 s triangle over +-1.05 so it only grazes the edge. Phase-shifted so frame 0
  -- is already at w~-0.7 (a shape being born). The single near-empty "drop" is the bass hit
  -- jump at the +1.05 end, which pushes the slice briefly past the edge.
  local sweep = H.lerp(-1.05, 1.05, H.ease(H.tri(t + 1.6, 9)))
  local w_slice = sweep + jump + 0.05 * math.sin(t * 1.7) * f.rms

  local cliff_w = 0.4 * math.sin(t * 0.31)

  -- smooth-union k: snap apart on the beat (0.05), melt together between beats (0.6)
  local k = H.lerp(0.05, 0.6, H.smoothstep(0.0, 0.6, f.beat_phase))

  -- bubbles: three hyperspheres circling in the xw / yw / zw planes so they cross the slice
  local bubbles = {}
  for i = 0, 2 do
    local a = t * (0.7 + 0.15 * i) + i * H.tau / 3
    local R = 0.95
    local pos = { 0, 0, 0, R * math.sin(a) }
    pos[i + 1] = R * math.cos(a)
    bubbles[#bubbles + 1] = H.hypersphere {
      r = 0.32 + 0.08 * f.bass,
      op = "smooth_union", k = k,
      transform = { H.translate(pos[1], pos[2], pos[3], pos[4]) },
    }
  end

  -- temperature: wide spread — deep orange-red cold, blue-white hot; onsets flash hotter
  local cold = 1200
  local hot = 12500 + 8000 * f.onset + 3000 * f.high

  -- camera: one orbit per ~25 s. Distance follows the cross-section's size: |w_slice| near 0
  -- is the fattest slice, near +-1 the smallest. Clamped to 3.0..3.8 for consistent framing.
  if f.onset > 0.5 and last_push <= 0.5 then push_t = t end
  last_push = f.onset
  local slice_extent = math.max(0, 1 - (w_slice / 1.05) ^ 2)        -- ~0..1
  local dist = math.min(3.8, math.max(3.0, 2.6 + 0.8 * slice_extent + 0.5))
  -- onset push: quick 0.3 dolly, eased out over 0.4 s
  local pa = math.min(math.max((t - push_t) / 0.4, 0), 1)
  dist = dist - 0.3 * (1 - pa) ^ 3
  local orbit = t * H.tau / 25
  local eye = { dist * math.sin(orbit), 0.9 * math.sin(t * 0.21), dist * math.cos(orbit) }
  local roll = math.rad(8) * math.sin(t * 0.4)                     -- slow +-8 degree roll

  local field = {
    H.duocylinder { r1 = 0.8, r2 = 0.45 + 0.1 * energy },
    H.clifford {
      r1 = 0.75, r2 = 0.75, thickness = 0.1 + 0.06 * f.bass,
      op = "smooth_union", k = k,
      transform = { H.rotate("xz", t * 0.11), H.translate(0, 0, 0, cliff_w) },
    },
  }
  for _, b in ipairs(bubbles) do field[#field + 1] = b end

  return {
    transform = {
      H.rotate("xw", xw),
      H.rotate("yw", yw),
      H.rotate("zw", zw),
      H.rotate("yz", yz),
      H.shear("x", "w", 0.25 * f.onset),
      H.scale(1.0 + 0.15 * kick),               -- ~15% pulse, ~150 ms decay
    },
    field = field,
    w_slice         = w_slice,
    surface_width   = 0.012 + 0.022 * f.bass,   -- thin shell: a thick one stacks into a white blob
    particle_radius = 0.014 * (1 + 0.4 * f.onset),
    temperature     = { cold, hot },
    reflectivity    = 0.6 + 0.3 * f.high,
    exposure        = (0.9 + 0.4 * energy) * (1 - 0.35 * f.bass) + 0.35 * f.onset,
    -- the field is centred at the origin (all prims orbit it), so the origin is its centre of mass
    camera          = { eye = eye, target = { 0, 0, 0 }, fov = 46 - 4 * energy, roll = roll },
  }
end
