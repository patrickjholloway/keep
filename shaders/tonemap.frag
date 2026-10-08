#version 450
// tonemap.frag — HDR (RGBA16F, linear, unbounded) -> display (8-bit, sRGB-encoded).
// The particle pass *adds* light, so dense surfaces can sum far above 1.0. A tonemap curve
// rolls those highlights off smoothly instead of clipping them to flat white.
// The HDR image is read as a storage image (imageLoad at this pixel) — same resolution as the
// output, so no sampler/filtering is needed.
layout(set = 0, binding = 0, rgba16f) uniform readonly image2D hdr;
layout(push_constant) uniform Push { vec4 knobs; } pc;   // x = exposure, y = vignette strength
layout(location = 0) out vec4 o_color;

// ACES filmic fit (Krzysztof Narkowicz): cheap, punchy, pleasant highlight desaturation.
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
// Linear -> sRGB transfer function (the output images are UNORM, so we encode ourselves).
vec3 to_srgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec3 c = imageLoad(hdr, px).rgb * pc.knobs.x;
    vec2 uv = gl_FragCoord.xy / vec2(imageSize(hdr));
    float vig = 1.0 - pc.knobs.y * dot(uv - 0.5, uv - 0.5) * 2.0;   // gentle corner darkening
    o_color = vec4(to_srgb(aces(c * vig)), 1.0);
}
