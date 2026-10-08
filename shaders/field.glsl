// field.glsl — 4D field library, the GPU mirror of src/field.rs (Shape::sdf, smin, Field::eval).
// Keep these two in lockstep: every formula here has a twin in field.rs, and the Rust tests
// pin values at known points. Requires common.glsl (Primitive, PRIM_*, OP_*, `scene`) first.
//
// Coordinates: p4 is a world point on the slice hyperplane, q = A^-1 (p4 - t) is object space,
// l = local_inv_a (q - local_t) is primitive-local space where the canonical shapes live.

// --- 4D signed distance primitives (in primitive-local coordinates) ---
float sd_hypersphere(vec4 q, float r) { return length(q) - r; }
float sd_tesseract(vec4 q, vec4 b) {
    vec4 d = abs(q) - b;
    return length(max(d, vec4(0.0))) + min(max(max(d.x, d.y), max(d.z, d.w)), 0.0);
}
float sd_duocylinder(vec4 q, float r1, float r2) {
    vec2 d = vec2(length(q.xy) - r1, length(q.zw) - r2);
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
}
float sd_clifford(vec4 q, float r1, float r2, float th) {
    vec2 d = vec2(length(q.xy) - r1, length(q.zw) - r2);
    return length(d) - th;
}
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-5), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float prim_eval(uint i, vec4 q) {
    Primitive p = scene.prims[i];
    vec4 l = p.inv_a * (q - p.t);
    uint k = p.kind_op.x;
    if (k == PRIM_HYPERSPHERE) return sd_hypersphere(l, p.params.x);
    if (k == PRIM_TESSERACT)   return sd_tesseract(l, p.params);
    if (k == PRIM_DUOCYLINDER) return sd_duocylinder(l, p.params.x, p.params.y);
    return sd_clifford(l, p.params.x, p.params.y, p.params.z);
}

float field(vec4 p4) {
    vec4 q = scene.inv_a * (p4 - scene.t);
    float d = 1e9;
    for (uint i = 0u; i < scene.counts.y; i++) {
        float di = prim_eval(i, q);
        uint op = scene.prims[i].kind_op.y;
        if (i == 0u || op == OP_UNION) d = min(d, di);
        else if (op == OP_SMOOTH_UNION) d = smin(d, di, scene.prims[i].params.w);
        else if (op == OP_INTERSECT) d = max(d, di);
        else d = max(d, -di);
    }
    return d;
}
