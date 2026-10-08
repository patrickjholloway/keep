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
local G = glitch

-- ---------------------------------------------------------------- glitch choreography
-- The song (src/audio.rs, 122 BPM, 4-beat bars of 240/122 s): intro bars 0-4, groove 4-12,
-- build 12-20, drop 20-36, then the intro again. Glitches punctuate it:
--   intro            clean — the 4D form alone, so the contrast lands later
--   groove           sync-slip tears on strong onsets, prism dispersion on kicks
--   build            smear (comet trails) and a rising ring-mod carrier grow with the riser;
--                    the last bar stutters (bitcrush + sample-and-hold on 8th notes)
--   drop downbeat    Möbius warp + spectral phase scramble that relax over ~2 bars
--   drop             kicks disperse, every 4 bars the warp swirls back, every 8 bars a
--                    half-bar bitcrush stutter; bar 28-30 is left clean as a breather
--   outro            clean again
local BAR = 240 / 122
local DROP_T = 20 * BAR
G.lfo("swirl", "sine", 0.11)        -- slow rotation of the warp's pole
G.lfo("shimmer", "tri", 0.5)        -- carrier drift during the build
-- static cables: the patch bay does the multiply-add, the script only shapes envelopes
G.patch("kick", "dispersion.depth", 1)      -- gains/offsets set per section via glitch.source
G.patch("tear", "sync.depth", 1)
G.patch("smear_env", "smear.depth", 1)
G.patch("mean_luma", "smear.depth", -1.5)   -- envelope follower: bright frames smear less
G.patch("ring_env", "ring.depth", 1)
G.patch("drop_env", "warp.amount", 1)
G.patch("drop_env", "spectral.mix", 1)
-- swirl LFO adds a little extra twist on top of the base warp (both b and c, see below)
G.patch("swirl", "warp.b.im", -0.06)
G.patch("swirl", "warp.c.im", -0.2)
G.patch("shimmer", "ring.speed", 2)
G.patch("crush_env", "crush.depth", 1)
G.set("dispersion.mode", 0)                 -- radial: lens-like chromatic aberration
G.set("sync.speed", 14)                     -- tears re-roll fast inside a burst
G.set("smear.dir", 1)
-- Warp: an ELLIPTIC Möbius map with fixed points ±p. Conjugating a rotation by
-- g(z) = (z - p)/(z + p) gives, normalized to a = d = 1:
--   b = -i p tan(θ/2),  c = -i tan(θ/2) / p
-- so the picture swirls around two still points (a "dipole"). warp.amount blends b, c
-- toward 0, which is the same as shrinking θ: amount 0.5 = half the twist.
local P, THETA = 0.55, math.rad(70)   -- centre magnification |f'(0)| = 1 + tan²(θ/2) ≈ 1.5
local TT = math.tan(THETA / 2)
G.set("warp.a", 1, 0)
G.set("warp.b", 0, -P * TT)
G.set("warp.c", 0, -TT / P)
G.set("warp.d", 1, 0)
-- the map magnifies the centre by 1 + tan²(θ/2); zoom the output plane by the same amount
-- (another cable from the same envelope) so the form keeps its size and only the swirl shows
G.patch("drop_env", "warp.zoom", TT * TT)
G.set("spectral.lo", 0.035)
G.set("spectral.hi", 0.22)
G.set("spectral.atten", 0.35)
G.set("spectral.phase", 0.8)
G.set("spectral.soft", 0.02)

local tear_t, last_tear_onset = -10, 0
local function choreograph_glitch(t, f, kick)
  local bar = t / BAR
  local form = bar % 36
  local groove = (form >= 4 and form < 12) and 1 or 0
  local build = (form >= 12 and form < 20) and 1 or 0
  local drop = (form >= 20) and 1 or 0
  local breather = (form >= 28 and form < 30) and 1 or 0
  local active = (groove + build + drop) * (1 - breather)

  -- sync slip: a ~0.15 s burst of tearing on each strong onset (groove + drop, rarer in build)
  if f.onset > 0.6 and last_tear_onset <= 0.6 then tear_t = t; G.set("sync.freq", 10 + (math.floor(t * 7) % 5) * 9) end
  last_tear_onset = f.onset
  local tear = math.exp(-(t - tear_t) / 0.08)
  G.source("tear", active * tear * (0.05 * groove + 0.02 * build + 0.06 * drop))
  G.set("sync.block", 0.25 + 0.3 * drop)

  -- dispersion on kicks: pixels of rainbow split at the frame edge
  G.source("kick", active * kick * (40 * groove + 10 * build + 45 * drop))

  -- build: smear and ring-mod carrier grow with the riser
  local prog = build * (form - 12) / 8                     -- 0..1 across the build
  G.source("smear_env", 0.3 * prog * prog + 0.2 * prog)
  G.set("smear.feedback", 0.9 + 0.065 * prog)
  G.source("ring_env", 0.55 * prog * prog)
  G.set("ring.freq", 1.37 + 9 * prog * prog)               -- non-integer: stripes lean

  -- drop downbeat: warp + spectral scramble, relaxing over ~2 bars; every 4 bars a smaller swirl
  local since_drop = (form - 20) * BAR
  local big = drop * math.exp(-math.max(since_drop, 0) / (1.2 * BAR))
  local phrase = (form - 20) % 4
  local swirl = drop * (1 - breather) * (form >= 24 and 1 or 0) * math.exp(-phrase * BAR / 0.5) * 0.45
  G.source("drop_env", math.min(1, 0.75 * big + swirl))
  G.set("spectral.seed", math.floor(t * 122 / 60))          -- new echo direction every beat

  -- bitcrush stutters: last bar of the build on 8th notes, last half-bar of every 8 in the drop
  local eighth = math.floor(bar * 8) % 2
  local stutter = 0
  if form >= 19 and form < 20 then stutter = 1 end
  if drop == 1 and (form - 20) % 8 >= 7.5 then stutter = 1 end
  G.source("crush_env", stutter * (0.45 + 0.3 * eighth))
  G.set("crush.hold", stutter * (eighth == 1 and 14 or 5) + 1)
end

-- slow sections: 0 = calm intro, 1 = full energy
local function section(t) return H.ease(H.tri(t + 4, 32)) end

-- State kept across frames (offline rendering calls update() in frame order).
local jump_target, jump, last_onset, last_t = 0, 0, 0, 0
local push_t, last_push = -10, 0
local jump_t = -10                 -- time of the last slice jump (it bounces back after 0.3 s)
local beat_prev, rise_t = 0, -10   -- previous beat level, time of the last beat RISE

function update(t, f)
  local energy = section(t)
  local dt = math.max(t - last_t, 0); last_t = t
  -- beat envelope: ~150 ms decay assuming ~0.5 s beats (beat_phase runs 0..1 per beat)
  local kick = H.pulse(f.beat_phase, 3.3) * f.bass
  choreograph_glitch(t, f, kick)

  -- bass-triggered slice jumps: each strong onset flips the offset to +-0.15, eased toward it.
  if f.onset > 0.5 and last_onset <= 0.5 then
    jump_target = (jump_target > 0) and -0.15 or 0.15
    jump_t = t
  end
  -- bounce back: a jump only holds for 0.3 s, so the slice never parks outside the object
  local target_now = (t - jump_t < 0.3) and jump_target or 0
  last_onset = f.onset
  jump = jump + (target_now - jump) * (1 - math.exp(-dt * 10))

  -- 4D rotation: through-w planes dominate, ordinary 3D spin (yz) is slow.
  local xw = t * 0.32 + 0.5 * kick
  local yw = t * 0.10 + 0.25 * f.mid * energy
  local zw = t * 0.26 + 0.35 * kick
  local yz = t * 0.015

  -- slice sweep: 9 s triangle over +-1.05 so it only grazes the edge. Phase-shifted so frame 0
  -- is already at w~-0.7 (a shape being born). The single near-empty "drop" is the bass hit
  -- jump at the +1.05 end, which pushes the slice briefly past the edge.
  local sweep = H.lerp(-0.95, 0.95, H.ease(H.tri(t + 1.6, 9)))
  -- clamp so a bass jump never carries the slice fully past the shape (was blanking ~12s/~30s)
  local w_slice = math.max(-0.97, math.min(0.97, sweep + jump + 0.05 * math.sin(t * 1.7) * f.rms))

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
  -- is the fattest slice, near +-1 the smallest. Clamped to 2.4..3.2 so the shape fills ~55% of the frame height.
  -- Beat ring: send only the RISE of the beat level (max(beat - beat_prev, 0)), so the ring
  -- fires at the start of each beat instead of glowing for as long as the level stays high.
  -- A rise of >0.15 restarts the envelope; it then decays with a ~0.25 s time constant.
  local beat = f.onset
  if beat - beat_prev > 0.15 then rise_t = t end
  beat_prev = beat
  local beat_rise = math.exp(-(t - rise_t) / 0.25)

  if f.onset > 0.5 and last_push <= 0.5 then push_t = t end
  last_push = f.onset
  local slice_extent = math.max(0, 1 - (w_slice / 1.05) ^ 2)        -- ~0..1
  local dist = math.min(3.2, math.max(2.4, 2.0 + 0.8 * slice_extent + 0.5))
  -- onset push: quick 0.4 dolly, eased out over 0.4 s
  local pa = math.min(math.max((t - push_t) / 0.4, 0), 1)
  dist = dist - 0.4 * (1 - pa) ^ 3
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
    -- thin shell: a thick one stacks into a white blob. Near/past the edge of the object
    -- (|w| > 0.9) the shell widens so a sparse spray of beads lingers instead of an empty frame.
    surface_width   = 0.012 + 0.022 * f.bass + 0.03 * H.smoothstep(0.9, 1.15, math.abs(w_slice)),
    particle_radius = 0.014 * (1 + 0.4 * f.onset),
    temperature     = { cold, hot },
    reflectivity    = 0.6 + 0.3 * f.high,
    exposure        = (0.9 + 0.4 * energy) * (1 - 0.35 * f.bass) + 0.35 * f.onset,
    -- the field is centred at the origin (all prims orbit it), so the origin is its centre of mass
    -- colour journey tied to the slice position: -1 amber, 0 teal/amber, +1 magenta-amber
    palette         = math.max(-1, math.min(1, w_slice)),
    beat_rise       = beat_rise,
    camera          = { eye = eye, target = { 0, 0, 0 }, fov = 46 - 4 * energy, roll = roll },
  }
end
