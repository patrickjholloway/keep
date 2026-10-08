#version 450
// particle.vert — camera-facing billboard per droplet; no vertex buffers.
// Draw with vertexCount = 6, instanceCount = particle_count.
#include "common.glsl"
layout(std430, set = 0, binding = 2) readonly buffer Drops { Droplet droplets[]; };

layout(location = 0) out vec2 v_uv;        // [-1,1]^2 across the quad
layout(location = 1) out vec3 v_center;    // droplet center (world)
layout(location = 2) out vec3 v_emission;
layout(location = 3) out float v_vis;
layout(location = 4) out float v_radius;

void main() {
    const vec2 corners[6] = vec2[6](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,-1), vec2(1,1), vec2(-1,1));
    Droplet dr = droplets[gl_InstanceIndex];
    vec2 c = corners[gl_VertexIndex];
    // Per-droplet size jitter (+-50%), seeded by a hash of the instance index so it is stable
    // from frame to frame (no shimmering). PCG-style integer hash -> [0,1).
    uint hsh = uint(gl_InstanceIndex) * 747796405u + 2891336453u;
    hsh = ((hsh >> ((hsh >> 28u) + 4u)) ^ hsh) * 277803737u;
    hsh = (hsh >> 22u) ^ hsh;
    float jitter = 0.5 + float(hsh & 0xffffu) / 65535.0;          // 0.5 .. 1.5
    float r = dr.pos_size.w * jitter;
    // Camera right/up are the first two rows of the view matrix's rotation.
    vec3 right = vec3(scene.view[0][0], scene.view[1][0], scene.view[2][0]);
    vec3 up    = vec3(scene.view[0][1], scene.view[1][1], scene.view[2][1]);
    vec3 world = dr.pos_size.xyz + (right * c.x + up * c.y) * r;
    gl_Position = (r > 0.0) ? scene.proj * scene.view * vec4(world, 1.0) : vec4(2.0, 2.0, 2.0, 1.0); // culled => off-screen
    v_uv = c; v_center = dr.pos_size.xyz; v_emission = dr.emission.rgb; v_vis = dr.emission.a; v_radius = r;
}
