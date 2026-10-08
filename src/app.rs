//! app.rs — `keep run`: winit window, live Lua hot reload, free-fly camera.
//! Keys: WASD/QE move, mouse-drag look, 1/2/3 (+shift = reverse) nudge extra XW/YW/ZW rotation
//! on top of whatever Lua sets, [ ] nudge w_slice, R reset nudges, Esc quit.
use std::path::Path;

use glam::Vec3;

/// Interactive extras layered over the Lua scene (radians / slice units).
#[derive(Clone, Copy, Debug, Default)]
pub struct Nudges {
    pub xw: f32,
    pub yw: f32,
    pub zw: f32,
    pub w_slice: f32,
    /// Current WASD/QE intent, (right, up, forward).
    pub movement: Vec3,
}

/// Open the window and run until closed. Features are zero (or live input later).
pub fn run(script: &Path) -> anyhow::Result<()> {
    let _ = script;
    todo!("winit ApplicationHandler: Renderer::new_windowed, ScriptHost, FlyCamera, Nudges")
}
