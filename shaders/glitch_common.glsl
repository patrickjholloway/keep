// glitch_common.glsl — shared declarations for the image-plane glitch passes (glitch_*.comp).
// Mirrors `GlitchParams` in src/glitch.rs byte-for-byte (std140, every member a vec4).
//
// The glitch chain runs AFTER the particles are rasterized into the HDR image and BEFORE the
// tonemap. Each module reads `src` and writes `dst`; the renderer ping-pongs two descriptor
// sets (HDR -> scratch, scratch -> HDR) so a module never reads the pixels it is writing.
// A module whose depth/mix is 0 is not dispatched at all (zero cost when bypassed).

layout(std140, set = 0, binding = 0) uniform Glitch {
    vec4 sync;     // depth (fraction of a line), freq (bands / frame), speed (Hz), block (0..1)
    vec4 ring;     // depth, freq (carrier cycles per scanline), speed (Hz), chroma (rad)
    vec4 smear;    // depth, feedback a, direction (+1 / -1), -
    vec4 disp;     // depth (px @1080p), mode (0 radial, 1 directional), angle (rad), -
    vec4 warp_ab;  // Mobius a (re, im), b (re, im)
    vec4 warp_cd;  // Mobius c (re, im), d (re, im)
    vec4 warp_m;   // amount, zoom, -, -
    vec4 spec;     // mix, band lo, band hi (cycles/pixel), attenuation
    vec4 spec2;    // phase scramble, seed, band softness, gain outside the band
    vec4 crush;    // depth (levels), hold (block px), -, -
    vec4 misc;     // time (s), mean luminance of previous frame, frame index, -
} g;

layout(set = 0, binding = 1, rgba16f) uniform readonly image2D src;
layout(set = 0, binding = 2, rgba16f) uniform writeonly image2D dst;

// FFT work buffers (spectral gate): one complex number per working-grid cell and colour
// channel, stored as (re.rgb, im.rgb). Stockham ping-pongs between A and B.
struct Cplx { vec4 re; vec4 im; };
layout(std430, set = 0, binding = 3) buffer FftA { Cplx fa[]; };
layout(std430, set = 0, binding = 4) buffer FftB { Cplx fb[]; };
// Image statistics written by glitch_stats.comp, read back by the CPU next frame.
layout(std430, set = 0, binding = 5) buffer Stats { vec4 stats; };

// x = mode / sub-pass, y = stage parameter, z = flags, w = spare (see each shader).
layout(push_constant) uniform Push { uvec4 p; } pc;

const float PI = 3.14159265358979;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);   // Rec.709 luminance weights

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Bilinear sample of `src` at continuous pixel coordinates (pixel centres at integer + 0).
// Storage images have no sampler, so we do the 4-tap lerp by hand. Outside -> black.
vec3 fetch(ivec2 q, ivec2 size) {
    if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size))) return vec3(0.0);
    return imageLoad(src, q).rgb;
}
vec3 bilinear(vec2 p, ivec2 size) {
    vec2 f = floor(p);
    vec2 t = p - f;
    ivec2 i = ivec2(f);
    vec3 a = mix(fetch(i, size), fetch(i + ivec2(1, 0), size), t.x);
    vec3 b = mix(fetch(i + ivec2(0, 1), size), fetch(i + ivec2(1, 1), size), t.x);
    return mix(a, b, t.y);
}
