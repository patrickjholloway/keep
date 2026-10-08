//! camera.rs — free-fly camera for `keep run` (WASD + mouse look, Q/E down/up, shift = fast).
use glam::Vec3;

use crate::scene::Camera;

#[derive(Clone, Debug)]
pub struct FlyCamera {
    pub pos: Vec3,
    /// Radians; yaw around +Y, pitch clamped to ±89°.
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
    pub fov_y: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        FlyCamera { pos: Vec3::new(0.0, 0.0, 6.0), yaw: -std::f32::consts::FRAC_PI_2, pitch: 0.0, speed: 2.0, fov_y: 50f32.to_radians() }
    }
}

impl FlyCamera {
    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.yaw.cos() * self.pitch.cos(), self.pitch.sin(), self.yaw.sin() * self.pitch.cos())
    }
    /// `mv` = (right, up, forward) intent in [-1,1]; dt seconds.
    pub fn step(&mut self, mv: Vec3, dt: f32) {
        let f = self.forward();
        let r = f.cross(Vec3::Y).normalize();
        self.pos += (r * mv.x + Vec3::Y * mv.y + f * mv.z) * self.speed * dt;
    }
    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.003;
        self.pitch = (self.pitch - dy * 0.003).clamp(-1.55, 1.55);
    }
    pub fn camera(&self) -> Camera {
        Camera { eye: self.pos, target: self.pos + self.forward(), fov_y: self.fov_y }
    }
}
