// common.glsl — THE GPU side of the SceneParams contract.
// Must match `SceneParams` / `GpuField` / `GpuPrimitive` in src/scene.rs byte-for-byte
// (std140; every member is a vec4/mat4/uvec4 so there is no hidden padding).
// build.rs textually inlines this file wherever a shader says `#include "common.glsl"`.

#define MAX_PRIMS 8

// Primitive kinds (Field::kind_id in src/field.rs).
#define PRIM_HYPERSPHERE  0u  // params.x = radius
#define PRIM_TESSERACT    1u  // params.xyzw = half extents (rounded by nothing)
#define PRIM_DUOCYLINDER  2u  // params.x = r1 (xy disc), params.y = r2 (zw disc)
#define PRIM_CLIFFORD     3u  // params.x = R1, params.y = R2, params.z = tube thickness

// Combine ops (how prim i joins the accumulated field).
#define OP_UNION        0u
#define OP_SMOOTH_UNION 1u  // smooth min with k = params.w
#define OP_INTERSECT    2u
#define OP_SUBTRACT     3u

struct Primitive {
    uvec4 kind_op;    // x = kind, y = op, zw unused
    vec4  params;     // shape parameters (see PRIM_*), w = smooth k for OP_SMOOTH_UNION
    mat4  inv_a;      // local inverse 4D linear part (A^-1), column-major like glam
    vec4  t;          // local translation; local q = inv_a * (p - t)
};

layout(std140, set = 0, binding = 0) uniform Scene {
    // --- global 4D object transform: q = inv_a * (p4 - t) ---
    mat4  inv_a;
    vec4  t;
    // --- slice + material ---
    vec4  slice;      // x = w_slice, y = surface_width (eps), z = particle_radius, w = time
    vec4  material;   // x = temp_lo (K), y = temp_hi (K), z = reflectivity, w = exposure
    vec4  audio;      // x = bass, y = mid, z = high, w = onset   (for shader-side flourishes)
    // --- camera ---
    mat4  view;
    mat4  proj;
    vec4  cam_pos;    // xyz = eye, w = fov_y (radians)
    // --- particle cloud ---
    uvec4 counts;     // x = particle_count, y = prim_count, zw unused
    // --- 4D field ---
    Primitive prims[MAX_PRIMS];
} scene;

// Per-particle output written by field.comp, read by particle.vert.
struct Droplet {
    vec4 pos_size;    // xyz = world position, w = billboard radius (0 => culled)
    vec4 normal_d;    // xyz = 3D normal from the 4D gradient, w = field value d
    vec4 emission;    // rgb = blackbody emission (linear), a = visibility weight
};
