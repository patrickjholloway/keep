#version 450
// particle.frag — shade the billboard as an analytic sphere: specular + blackbody emission.
#include "common.glsl"
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_center;
layout(location = 2) in vec3 v_emission;
layout(location = 3) in float v_vis;
layout(location = 4) in float v_radius;
layout(location = 0) out vec4 o_color;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;                                  // outside the sphere's silhouette
    vec3 right = vec3(scene.view[0][0], scene.view[1][0], scene.view[2][0]);
    vec3 up    = vec3(scene.view[0][1], scene.view[1][1], scene.view[2][1]);
    vec3 fwd   = normalize(scene.cam_pos.xyz - v_center);
    vec3 n = normalize(right * v_uv.x + up * v_uv.y + fwd * sqrt(1.0 - r2));   // sphere normal
    vec3 l = normalize(vec3(0.4, 0.8, 0.5));
    vec3 h = normalize(l + fwd);
    float spec = pow(max(dot(n, h), 0.0), 64.0) * scene.material.z;
    float fres = pow(1.0 - max(dot(n, fwd), 0.0), 4.0);
    vec3 col = v_emission * (0.35 + 0.65 * n.z * n.z) + vec3(spec + 0.3 * fres * scene.material.z);
    o_color = vec4(col * v_vis, 1.0);                       // additive blend in the pipeline
}
