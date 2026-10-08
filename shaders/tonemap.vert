#version 450
// tonemap.vert — one fullscreen triangle, no vertex buffers. Draw with vertexCount = 3.
// The triangle (-1,-1) (3,-1) (-1,3) covers the whole [-1,1]^2 clip square; the excess is clipped.
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
