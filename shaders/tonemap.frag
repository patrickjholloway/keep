#version 450
// tonemap.frag — HDR (RGBA16F, linear, unbounded) -> display (8-bit, sRGB-encoded).
// Steps: bloom gather -> background glow -> exposure -> luminance tonemap (keeps hue) ->
// colour grade -> sRGB -> film grain.
// The HDR image is read as a storage image (imageLoad), so "blur" here is a sparse gather of
// explicit texel loads rather than filtered sampling.
layout(set = 0, binding = 0, rgba16f) uniform readonly image2D hdr;
layout(push_constant) uniform Push { vec4 knobs; } pc;   // x = exposure, y = vignette, z = time, w = bass
layout(location = 0) out vec4 o_color;

// ACES filmic fit (Narkowicz). Applied to LUMINANCE only, below, so it can't bleach hue.
float aces(float x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 to_srgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}
float hash12(vec2 p) {                     // cheap white noise for grain
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Bloom: a single-pass, wide, sparse "Gaussian" — 3 rings x 12 taps, weights fall off with ring
// radius. Not a true separable blur (that would need extra images + passes); at 1080p this is
// cheap and gives the soft halo we want. Radii scale with image height.
vec3 bloom(ivec2 px, ivec2 size) {
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    float base = float(size.y) / 1080.0;
    for (int ring = 1; ring <= 3; ring++) {
        float rad = base * 6.0 * float(ring * ring);            // 6, 24, 54 px
        float w = exp(-0.5 * float(ring * ring) / 2.0);
        for (int k = 0; k < 12; k++) {
            float a = 6.2831853 * (float(k) + 0.5 * float(ring)) / 12.0;
            ivec2 q = clamp(px + ivec2(vec2(cos(a), sin(a)) * rad), ivec2(0), size - 1);
            acc += imageLoad(hdr, q).rgb * w;
            wsum += w;
        }
    }
    return acc / wsum;
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    ivec2 size = imageSize(hdr);
    vec2 uv = gl_FragCoord.xy / vec2(size);
    vec3 c = imageLoad(hdr, px).rgb + 0.15 * 4.0 * bloom(px, size);   // 4x: the gather averages mostly-dark taps

    // Faint radial background glow tinted by the low band (deep red-violet, warmer with bass).
    float rd = length((uv - 0.5) * vec2(float(size.x) / float(size.y), 1.0));
    vec3 lowcol = mix(vec3(0.04, 0.01, 0.06), vec3(0.12, 0.02, 0.04), clamp(pc.knobs.w, 0.0, 1.0));
    c += lowcol * exp(-rd * rd * 3.0);

    c *= pc.knobs.x;
    float vig = 1.0 - pc.knobs.y * dot(uv - 0.5, uv - 0.5) * 2.0;
    c *= vig;

    // Saturation-preserving tonemap: compress luminance, keep the RGB ratios. A gentle
    // desaturation only for very bright pixels (so hot cores still go a little white).
    float L = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float Lt = aces(L);
    vec3 m = c * (Lt / max(L, 1e-5));
    m = mix(m, vec3(Lt), smoothstep(0.85, 1.0, Lt) * 0.5);
    m = clamp(m, 0.0, 1.0);

    // Colour grade: lift shadows toward teal, a touch of contrast.
    float lum = dot(m, vec3(0.2126, 0.7152, 0.0722));
    // (weighted by lum so true black stays black; peaks in the dim mid-shadows)
    m += vec3(-0.01, 0.06, 0.08) * lum * (1.0 - smoothstep(0.0, 0.35, lum)) * 2.0;
    m = clamp(m, 0.0, 1.0);

    vec3 s = to_srgb(m);
    s += (hash12(gl_FragCoord.xy + fract(pc.knobs.z * 7.13) * 431.0) - 0.5) * (3.0 / 255.0);   // grain/dither
    o_color = vec4(s, 1.0);
}
