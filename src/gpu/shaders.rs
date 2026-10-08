//! SPIR-V blobs compiled from shaders/*.glsl by build.rs (naga).
pub const FIELD_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/field.comp.spv"));
pub const PARTICLE_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/particle.vert.spv"));
pub const PARTICLE_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/particle.frag.spv"));

/// SPIR-V must be consumed as u32 words; include_bytes! is only byte-aligned, so copy.
pub fn words(spv: &[u8]) -> Vec<u32> {
    spv.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
