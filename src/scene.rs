//! scene.rs — THE contract between Lua, CPU and GPU.
//!
//! `SceneParams` is uploaded verbatim (bytemuck) into the `Scene` uniform block declared in
//! `shaders/common.glsl`. Layout rules: std140, every field is a 16-byte vec4/mat4/uvec4, so
//! the Rust `#[repr(C)]` layout and the GLSL layout are identical with no padding surprises.
//! If you change one side, change the other (the size assertion below catches drift).
//!
//! Flow per frame:  Features (audio.rs) -> Lua update(t, features) -> SceneDesc (script.rs)
//!                  -> SceneDesc::to_params(camera) -> SceneParams -> GPU uniform.
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

use crate::{field::Field, math4d::Affine4};

/// Must equal `MAX_PRIMS` in common.glsl.
pub const MAX_PRIMS: usize = 8;

/// One 4D primitive as the GPU sees it (`struct Primitive` in common.glsl). 112 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuPrimitive {
    /// x = kind (PRIM_*), y = op (OP_*).
    pub kind_op: [u32; 4],
    /// Shape params; w = smooth-union k.
    pub params: [f32; 4],
    /// Local inverse linear part A⁻¹ (column-major).
    pub inv_a: [[f32; 4]; 4],
    /// Local translation t.
    pub t: [f32; 4],
}

/// The whole per-frame uniform (`uniform Scene` in common.glsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SceneParams {
    /// Global object transform, inverted: q = inv_a * (p4 - t).
    pub inv_a: [[f32; 4]; 4],
    pub t: [f32; 4],
    /// x = w_slice, y = surface_width, z = particle_radius, w = time (s).
    pub slice: [f32; 4],
    /// x = temp_lo K, y = temp_hi K, z = reflectivity, w = exposure.
    pub material: [f32; 4],
    /// x = bass, y = mid, z = high, w = onset (copied from audio Features).
    pub audio: [f32; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    /// xyz = eye, w = fov_y radians.
    pub cam_pos: [f32; 4],
    /// x = particle_count, y = prim_count,
    /// z = palette drift (f32 bits, -1..1: amber -> teal -> magenta; read with uintBitsToFloat),
    /// w = beat-rise envelope (f32 bits, 1 at the start of a beat, decays ~0.25 s; drives the ring).
    /// Floats ride in the uvec4 as raw bits so the std140 layout stays unchanged.
    pub counts: [u32; 4],
    pub prims: [GpuPrimitive; MAX_PRIMS],
}
const _: () = assert!(std::mem::size_of::<SceneParams>() == 64 + 16 * 4 + 128 + 16 + 16 + 112 * MAX_PRIMS);

/// Camera as decided either by Lua (render mode) or the free-fly controller (run mode).
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub fov_y: f32,
    /// Roll (radians) about the view axis: tilts the camera's "up" vector.
    pub roll: f32,
}

/// What Lua's `update(t, features)` returns, in Rust types. Friendly to build, then packed.
#[derive(Clone, Debug)]
pub struct SceneDesc {
    /// Global 4D affine applied to the whole field (A, t — NOT inverted; packing inverts).
    pub object: Affine4,
    pub field: Field,
    pub w_slice: f32,
    pub surface_width: f32,
    pub particle_radius: f32,
    pub temperature: (f32, f32),
    pub reflectivity: f32,
    pub exposure: f32,
    /// Slow colour journey across the 4D sweep: -1 amber, 0 teal/amber, +1 magenta-amber.
    pub palette: f32,
    /// Beat-start envelope (1 right at a beat's rise, decaying to 0). Tonemap ring age = 1 - this.
    pub beat_rise: f32,
    /// Lua-driven camera; `None` = leave to the interactive controller / default.
    pub camera: Option<Camera>,
    /// Image-plane glitch patch bay state (resolved per frame by `glitch::resolve`).
    pub glitch: crate::glitch::GlitchDesc,
}

impl SceneDesc {
    /// Pack into the GPU contract. `aspect` = width / height.
    pub fn to_params(&self, time: f32, audio: Vec4, camera: Camera, aspect: f32, particle_count: u32) -> SceneParams {
        let inv = self.object.inverse();
        let mut prims = [GpuPrimitive::default(); MAX_PRIMS];
        let n = self.field.pack(&mut prims);
        // Roll = rotate the world-up vector about the viewing direction before building the view.
        let fwd = (camera.target - camera.eye).normalize_or_zero();
        let up = glam::Quat::from_axis_angle(fwd, camera.roll) * Vec3::Y;
        let view = Mat4::look_at_rh(camera.eye, camera.target, up);
        // Vulkan clip space: y down, z in [0,1]. perspective_rh already gives [0,1] depth; flip y.
        let mut proj = Mat4::perspective_rh(camera.fov_y, aspect, 0.01, 100.0);
        proj.y_axis.y *= -1.0;
        SceneParams {
            inv_a: inv.a.to_cols_array_2d(),
            t: inv_t_for_shader(&self.object),
            slice: [self.w_slice, self.surface_width, self.particle_radius, time],
            material: [self.temperature.0, self.temperature.1, self.reflectivity, self.exposure],
            audio: audio.to_array(),
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            cam_pos: [camera.eye.x, camera.eye.y, camera.eye.z, camera.fov_y],
            counts: [particle_count, n as u32, self.palette.to_bits(), self.beat_rise.to_bits()],
            prims,
        }
    }
}

/// The shader computes q = inv_a * (p - t) with the *forward* t, which equals A⁻¹p + (−A⁻¹t).
fn inv_t_for_shader(obj: &Affine4) -> [f32; 4] {
    obj.t.to_array()
}
