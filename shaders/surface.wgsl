// surface.wgsl — GPU -> CPU "surface descriptor" for the sonification voice (src/sonify.rs).
//
// Why WGSL when every other shader is GLSL: naga's GLSL frontend has no atomics, and this pass
// is all about atomics. naga's WGSL frontend has them (atomic<u32> in workgroup and storage
// memory), and both frontends emit the same SPIR-V 1.3 for Vulkan, so build.rs simply picks the
// frontend by file extension. The Scene uniform below mirrors the prefix of shaders/common.glsl
// that this pass reads (a shader may declare a block smaller than the bound buffer).
//
// After field.comp has written one Droplet per particle, this pass bins every *visible* droplet
// (billboard radius > 0, i.e. |d| < surface_width: the beads you actually see) by its
// DIRECTION from the slice centroid, and accumulates per bin
//     count, Σ heat, Σ roughness, Σ radius            (means are taken on the CPU: Σ / count)
// plus a header with the total count and Σ position (-> the next frame's centroid).
//
// Bin mapping: OCTAHEDRAL. A unit direction d is projected onto the octahedron |x|+|y|+|z| = 1,
// the lower half (z < 0) is folded out over the diagonals, and the result is the square
// [-1,1]^2 cut into SIDE x SIDE cells. Unlike latitude/longitude there is no pole pinch: cells
// cover similar solid angles, and encode/decode are a handful of ALU ops. src/sonify.rs mirrors
// `oct_encode` (tests pin the CPU twin of this whole pass against the GPU).
//
//   heat      = (0.5 + 0.5 d/eps)^2 clamped to 0..1: the exponent field.comp uses to place a
//               bead between temp_lo and temp_hi (0 = cold inner shell, 1 = hot outer edge).
//   roughness = 1 - n.r_hat, n = the bead's 3D surface normal, r_hat = direction from the
//               centroid. ~0 on a smooth bulge facing outward, ~1 on creases / saddles /
//               pinch-off necks, up to 2 on inward-facing (concave) shell. A cheap stand-in for
//               |curvature| that needs no second derivatives.
//   radius    = |p - centroid|.
//
// Reduction strategy (the interesting part):
//   1. PRIVATIZE: each 256-thread workgroup owns a histogram in workgroup (on-chip, "shared")
//      memory, 16 KiB. Threads atomicAdd into it; contention stays inside one workgroup.
//   2. FLUSH: after a barrier, the workgroup walks its histogram and adds only the NON-EMPTY
//      bins to the global buffer, one global atomicAdd per word. A thin shell touches a few
//      dozen bins per workgroup, so global atomic traffic drops ~100x versus every particle
//      hitting global memory.
// Core Vulkan only guarantees INTEGER atomics (float atomics need VK_EXT_shader_atomic_float),
// so everything accumulates in FIXED POINT (value * scale, rounded). The scales keep 1M beads
// inside 32 bits (see FX_* below).
//
// The global buffer is zeroed by vkCmdFillBuffer before this dispatch and read by the CPU after
// the frame fence (host-visible, persistently mapped, HOST_COHERENT): see src/gpu/surface.rs.

const SIDE: u32 = 32u;
const NBINS: u32 = 1024u;          // SIDE * SIDE
const FX_HEAT: f32 = 1024.0;       // heat 0..1      -> <= 1024 per bead (1M beads: 1.0e9 < 2^32)
const FX_ROUGH: f32 = 1024.0;      // rough 0..2     -> <= 2048 per bead (1M beads: 2.1e9 < 2^32)
const FX_RADIUS: f32 = 1024.0;     // radius 0..2.6  -> <= 2700 per bead (1M beads: 2.7e9 < 2^32)
const FX_POS: f32 = 512.0;         // pos +-1.5      -> +-768 per bead   (1M beads: 7.7e8 < 2^31)

struct Scene {
    inv_a: mat4x4<f32>,
    t: vec4<f32>,
    slice: vec4<f32>,      // x = w_slice, y = surface_width (eps), z = particle_radius, w = time
    material: vec4<f32>,
    audio: vec4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    counts: vec4<u32>,     // x = particle_count
};

struct Droplet {
    pos_size: vec4<f32>,   // xyz = position, w = billboard radius (0 => culled)
    normal_d: vec4<f32>,   // xyz = 3D normal, w = field value d
    emission: vec4<f32>,
};

// header = (visible count, Σx, Σy, Σz) in FX_POS fixed point; bins[4i..4i+4] = (count, Σheat, Σrough, Σradius).
struct Surface {
    header: array<atomic<i32>, 4>,
    bins: array<atomic<u32>>,
};

struct Push { center: vec4<f32> };  // centroid of the PREVIOUS frame's visible beads

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(2) var<storage, read> droplets: array<Droplet>;
@group(0) @binding(3) var<storage, read_write> surf: Surface;
var<push_constant> pc: Push;

var<workgroup> s_bins: array<atomic<u32>, 4096>;   // NBINS * 4
var<workgroup> s_head: array<atomic<i32>, 4>;

// Octahedral encode: unit vector -> [-1,1]^2.
fn oct_encode(d_in: vec3<f32>) -> vec2<f32> {
    let d = d_in / (abs(d_in.x) + abs(d_in.y) + abs(d_in.z));
    var e = d.xy;
    if (d.z < 0.0) {
        let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), e >= vec2<f32>(0.0));
        e = (vec2<f32>(1.0) - abs(e.yx)) * s;
    }
    return e;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    // (1) Workgroup memory starts zeroed in WGSL (naga emits the zero-init), but we clear
    //     explicitly anyway so the cost is visible: 4096 words / 256 threads = 16 stores each.
    for (var k = lid; k < NBINS * 4u; k += 256u) { atomicStore(&s_bins[k], 0u); }
    if (lid < 4u) { atomicStore(&s_head[lid], 0); }
    workgroupBarrier();   // every clear lands before any thread accumulates

    let id = gid.x;
    if (id < scene.counts.x) {
        let o = droplets[id];
        if (o.pos_size.w > 0.0) {                       // visible bead
            let p = o.pos_size.xyz;
            let r = p - pc.center.xyz;
            let rl = length(r);
            var rh = vec3<f32>(0.0, 0.0, 1.0);
            if (rl > 1e-6) { rh = r / rl; }
            let e = oct_encode(rh);
            let c = vec2<u32>(clamp((e * 0.5 + 0.5) * f32(SIDE), vec2<f32>(0.0), vec2<f32>(f32(SIDE) - 0.5)));
            let b = (c.y * SIDE + c.x) * 4u;
            let eps = scene.slice.y;
            let sx = clamp(0.5 + 0.5 * o.normal_d.w / eps, 0.0, 1.0);
            let heat = sx * sx;
            let rough = clamp(1.0 - dot(o.normal_d.xyz, rh), 0.0, 2.0);
            atomicAdd(&s_bins[b + 0u], 1u);
            atomicAdd(&s_bins[b + 1u], u32(heat * FX_HEAT + 0.5));
            atomicAdd(&s_bins[b + 2u], u32(rough * FX_ROUGH + 0.5));
            atomicAdd(&s_bins[b + 3u], u32(rl * FX_RADIUS + 0.5));
            let q = vec3<i32>(round(p * FX_POS));
            atomicAdd(&s_head[0], 1);
            atomicAdd(&s_head[1], q.x);
            atomicAdd(&s_head[2], q.y);
            atomicAdd(&s_head[3], q.z);
        }
    }
    workgroupBarrier();   // the whole workgroup's contributions are in workgroup memory

    // (2) Flush the non-empty bins: one global atomic per word.
    for (var bin = lid; bin < NBINS; bin += 256u) {
        let n = atomicLoad(&s_bins[bin * 4u]);
        if (n != 0u) {
            atomicAdd(&surf.bins[bin * 4u + 0u], n);
            atomicAdd(&surf.bins[bin * 4u + 1u], atomicLoad(&s_bins[bin * 4u + 1u]));
            atomicAdd(&surf.bins[bin * 4u + 2u], atomicLoad(&s_bins[bin * 4u + 2u]));
            atomicAdd(&surf.bins[bin * 4u + 3u], atomicLoad(&s_bins[bin * 4u + 3u]));
        }
    }
    if (lid == 0u) {
        let n = atomicLoad(&s_head[0]);
        if (n != 0) {
            atomicAdd(&surf.header[0], n);
            atomicAdd(&surf.header[1], atomicLoad(&s_head[1]));
            atomicAdd(&surf.header[2], atomicLoad(&s_head[2]));
            atomicAdd(&surf.header[3], atomicLoad(&s_head[3]));
        }
    }
}
