//! app.rs — `keep run`: winit window, live Lua hot reload, free-fly camera.
//! Keys: WASD/QE move, mouse-drag look, 1/2/3 (+shift = reverse) nudge extra XW/YW/ZW rotation
//! on top of whatever Lua sets, [ ] nudge w_slice, R reset nudges, Esc quit.
use std::{path::Path, sync::Arc, time::Instant};

use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::{
    audio::Features,
    camera::FlyCamera,
    gpu::{renderer::CloudSpec, target::Swapchain, Renderer},
    math4d::{Affine4, Plane},
    script::ScriptHost,
};

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

/// Open the window and run until closed (or after `max_frames` frames, for smoke tests).
/// Features are zero for now (live input later); `t` is wall-clock seconds since start.
pub fn run(script: &Path, max_frames: Option<u64>) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    // Poll = redraw continuously (an animation, not a document editor).
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        script: ScriptHost::load(script)?, max_frames, gpu: None, cam: FlyCamera::default(),
        nudges: Nudges::default(), start: Instant::now(), last: Instant::now(), frames: 0,
        dragging: false, last_cursor: None, fast: false, error: None,
    };
    event_loop.run_app(&mut app)?;
    if let Some(e) = app.error { return Err(e); }
    eprintln!("[run] rendered {} frames", app.frames);
    Ok(())
}

/// GPU objects exist only once winit hands us a window (`resumed`).
struct Gpu {
    // Field order = drop order: the swapchain is destroyed manually in `exiting`, then the
    // renderer (device) drops, then the window.
    renderer: Renderer,
    swapchain: Swapchain,
    window: Arc<Window>,
}

struct App {
    script: ScriptHost,
    max_frames: Option<u64>,
    gpu: Option<Gpu>,
    cam: FlyCamera,
    nudges: Nudges,
    start: Instant,
    last: Instant,
    frames: u64,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    fast: bool,
    error: Option<anyhow::Error>,
}

impl App {
    fn fail(&mut self, el: &ActiveEventLoop, e: anyhow::Error) {
        self.error = Some(e);
        el.exit();
    }

    fn frame(&mut self) -> anyhow::Result<()> {
        let Some(gpu) = self.gpu.as_mut() else { return Ok(()) };
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().min(0.1);
        self.last = now;
        let t = (now - self.start).as_secs_f32();

        self.script.poll_reload();
        let mut desc = self.script.update(t, &Features::default())?;
        // Layer the keyboard nudges *after* the script's own transform (applied last).
        let n = self.nudges;
        let extra = Affine4::rotation(Plane::ZW, n.zw)
            .compose(&Affine4::rotation(Plane::YW, n.yw))
            .compose(&Affine4::rotation(Plane::XW, n.xw));
        desc.object = extra.compose(&desc.object);
        desc.w_slice += n.w_slice;

        self.cam.speed = if self.fast { 6.0 } else { 2.0 };
        self.cam.step(n.movement, dt);
        let size = gpu.window.inner_size();
        if size.width == 0 || size.height == 0 { return Ok(()); } // minimized
        let aspect = size.width as f32 / size.height as f32;
        let params = desc.to_params(t, glam::Vec4::ZERO, self.cam.camera(), aspect, gpu.renderer.cloud.count);
        if !gpu.renderer.render_present(&mut gpu.swapchain, &params)? {
            // Out of date (resize): rebuild at the window's current size and try next frame.
            unsafe { gpu.renderer.ctx.device.device_wait_idle()? };
            gpu.swapchain.recreate(&gpu.renderer.ctx, size.width, size.height)?;
        }
        self.frames += 1;
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gpu.is_some() { return; }
        let attrs = Window::default_attributes().with_title("keep").with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let res = (|| -> anyhow::Result<Gpu> {
            let window = Arc::new(el.create_window(attrs)?);
            let cloud = CloudSpec { count: 1_000_000, half_extent: 1.5, seed: 1 };
            let (renderer, swapchain) = Renderer::new_windowed(cloud, &window)?;
            Ok(Gpu { renderer, swapchain, window })
        })();
        match res {
            Ok(g) => { g.window.request_redraw(); self.gpu = Some(g); }
            Err(e) => self.fail(el, e),
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.frame() { return self.fail(el, e); }
                if self.max_frames.is_some_and(|m| self.frames >= m) { return el.exit(); }
                if let Some(g) = &self.gpu { g.window.request_redraw(); }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = state == ElementState::Pressed;
                self.last_cursor = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    if let Some((x, y)) = self.last_cursor { self.cam.look((position.x - x) as f32, (position.y - y) as f32); }
                    self.last_cursor = Some((position.x, position.y));
                }
            }
            WindowEvent::ModifiersChanged(m) => self.fast = m.state().shift_key(),
            WindowEvent::KeyboardInput { event, .. } => self.key(el, event),
            _ => {}
        }
    }

    fn exiting(&mut self, _el: &ActiveEventLoop) {
        if let Some(mut g) = self.gpu.take() {
            unsafe { let _ = g.renderer.ctx.device.device_wait_idle(); }
            g.swapchain.destroy(&g.renderer.ctx);
        }
    }
}

impl App {
    fn key(&mut self, el: &ActiveEventLoop, ev: KeyEvent) {
        let down = ev.state == ElementState::Pressed;
        let v = if down { 1.0 } else { 0.0 };
        let PhysicalKey::Code(code) = ev.physical_key else { return };
        let step = if self.fast { -0.05 } else { 0.05 }; // shift reverses nudges
        let m = &mut self.nudges.movement;
        match code {
            KeyCode::KeyD => m.x = v, KeyCode::KeyA => m.x = -v,
            KeyCode::KeyE => m.y = v, KeyCode::KeyQ => m.y = -v,
            KeyCode::KeyW => m.z = v, KeyCode::KeyS => m.z = -v,
            _ if !down => {}
            KeyCode::Digit1 => self.nudges.xw += step,
            KeyCode::Digit2 => self.nudges.yw += step,
            KeyCode::Digit3 => self.nudges.zw += step,
            KeyCode::BracketLeft => self.nudges.w_slice -= 0.02,
            KeyCode::BracketRight => self.nudges.w_slice += 0.02,
            KeyCode::KeyR => self.nudges = Nudges::default(),
            KeyCode::Escape => el.exit(),
            _ => {}
        }
    }
}
