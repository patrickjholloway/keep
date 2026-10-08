//! SPIR-V blobs compiled from shaders/*.glsl by build.rs (naga).
pub const FIELD_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/field.comp.spv"));
pub const PARTICLE_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/particle.vert.spv"));
pub const PARTICLE_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/particle.frag.spv"));

/// SPIR-V must be consumed as u32 words; include_bytes! is only byte-aligned, so copy.
pub fn words(spv: &[u8]) -> Vec<u32> {
    spv.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
pub const TONEMAP_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.vert.spv"));
pub const TONEMAP_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.frag.spv"));
pub const GLITCH_SYNC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glitch_sync.comp.spv"));
pub const GLITCH_RING: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glitch_ring.comp.spv"));
pub const GLITCH_SMEAR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glitch_smear.comp.spv"));
pub const GLITCH_DISP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glitch_disp.comp.spv"));
pub const GLITCH_WARP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glitch_warp.comp.spv"));
pub const GLITCH_SPECTRAL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glitch_spectral.comp.spv"));
pub const GLITCH_CRUSH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glitch_crush.comp.spv"));
pub const GLITCH_STATS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glitch_stats.comp.spv"));
