-- Target API sketch for the first spike; not wired up yet.
local shape = hyper.duocylinder { r1 = 1.0, r2 = 0.6 }
shape:rotate("xw", time * 0.17)
shape:rotate("yz", time * 0.08)

scene {
  particles = cloud { count = 500000, bounds = { -3, 3 }, radius = 0.004 },
  field = shape,
  material = { surface_width = 0.018, temperature = { 1200, 9000 }, reflectivity = 0.6 },
  background = "black",
}
