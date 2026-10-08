#version 450
// particle.frag — shade the billboard as an analytic sphere: emission + sharp glint + sky reflection.
// Output is ADDED into the HDR image, so every term here is "light this droplet contributes".
// Keep per-droplet light small (EMIT_SCALE): thousands of droplets stack on dense surfaces and
// anything too bright gets its colour bleached by the tonemapper.
#include "common.glsl"
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_center;
layout(location = 2) in vec3 v_emission;
layout(location = 3) in float v_vis;
layout(location = 4) in float v_radius;
layout(location = 0) out vec4 o_color;

const float EMIT_SCALE = 0.15;

// Two-tone "sky" the droplets reflect: deep indigo below, pale teal above.
vec3 sky(vec3 dir) {
    return mix(vec3(0.05, 0.02, 0.12), vec3(0.25, 0.6, 0.7), smoothstep(-0.3, 0.8, dir.y));
}

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;                                  // outside the sphere's silhouette
    vec3 right = vec3(scene.view[0][0], scene.view[1][0], scene.view[2][0]);
    vec3 up    = vec3(scene.view[0][1], scene.view[1][1], scene.view[2][1]);
    vec3 fwd   = normalize(scene.cam_pos.xyz - v_center);  // droplet -> eye
    vec3 n = normalize(right * v_uv.x + up * v_uv.y + fwd * sqrt(1.0 - r2));   // sphere normal
    float ndv = max(dot(n, fwd), 0.0);                     // 1 at the droplet centre, 0 at rim

    // Emission: view-relative falloff (bright core, darker rim) so droplets read as round beads.
    float ev = max(max(v_emission.r, v_emission.g), v_emission.b);
    vec3 hue = v_emission / max(ev, 1e-4);                 // normalized emission colour
    vec3 col = v_emission * EMIT_SCALE * (0.25 + 0.75 * ndv);

    // Glint: very sharp highlight, tinted halfway toward the droplet's own hue.
    vec3 l = normalize(vec3(0.4, 0.8, 0.5));
    vec3 h = normalize(l + fwd);
    float spec = pow(max(dot(n, h), 0.0), 220.0) * scene.material.z * 2.5;
    col += spec * mix(vec3(1.0), hue, 0.5);

    // Environment reflection (Fresnel weighted) + faint rim.
    vec3 refl = reflect(-fwd, n);
    float fres = pow(1.0 - ndv, 4.0);
    col += sky(refl) * scene.material.z * (0.04 + 0.3 * fres);
    col += hue * 0.08 * fres;
    o_color = vec4(col * v_vis, 1.0);                       // additive blend in the pipeline
}
