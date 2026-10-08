-- scene.lua — the creative topcoat. Hot-reloaded on save (keep run) and called once per
-- video frame (keep render). Return plain tables; see docs/ARCHITECTURE.md "Lua scene table".
--
-- update(t, f):  t = seconds;  f = { bass, mid, high, onset, rms, beat_phase }  (all ~0..1)

function update(t, f)
  return {
    -- Global 4D transform, applied in list order (first entry happens first).
    transform = {
      { "rotate", "xw", t * 0.17 + f.bass * 0.6 },
      { "rotate", "yw", t * 0.11 + f.mid * 0.4 },
      { "rotate", "zw", t * 0.07 },
      { "rotate", "yz", t * 0.08 },
      { "shear", "w", "x", 0.25 * f.onset },
    },
    field = {
      { shape = "duocylinder", r1 = 1.0, r2 = 0.6 },
      { shape = "hypersphere", r = 0.7, op = "smooth_union", k = 0.3,
        transform = { { "translate", 0, 0, 0, math.sin(t * 0.5) } } },
    },
    w_slice         = 0.3 * math.sin(t * 0.23),
    surface_width   = 0.018 + 0.03 * f.rms,
    particle_radius = 0.004 * (1 + f.onset),
    temperature     = { 1200, 9000 * (0.6 + 0.4 * f.high) },
    reflectivity    = 0.6,
    exposure        = 1.0,
    -- camera = { eye = { 0, 0, 6 }, target = { 0, 0, 0 }, fov = 50 },  -- optional; omitted in `run`
  }
end
