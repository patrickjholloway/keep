//! script.rs — the Lua topcoat. Loads a scene script, hot-reloads it on save, and each frame
//! calls its global `update(t, features)` which returns a scene table (see docs/ARCHITECTURE.md
//! "Lua scene table"). Errors in Lua never crash the app: the last good SceneDesc is kept.
use std::{path::{Path, PathBuf}, sync::mpsc::Receiver};

use crate::{audio::Features, scene::SceneDesc};

pub struct ScriptHost {
    lua: mlua::Lua,
    path: PathBuf,
    /// Kept alive so file events keep arriving.
    _watcher: notify::RecommendedWatcher,
    events: Receiver<notify::Result<notify::Event>>,
    /// Last successfully produced scene (fallback after a script error).
    last_good: Option<SceneDesc>,
}

impl ScriptHost {
    /// Load `path`, install the `hyper` helper API (shapes, rotate, shear, ...) and start watching.
    pub fn load(path: &Path) -> anyhow::Result<ScriptHost> {
        let _ = path;
        todo!("create Lua state, register API, exec file, start notify watcher")
    }

    /// Re-execute the script if it changed on disk. Returns true if a reload happened.
    pub fn poll_reload(&mut self) -> bool {
        todo!("drain self.events; on modify, re-exec self.path into self.lua")
    }

    /// Call Lua `update(t, features)` and convert the returned table into a `SceneDesc`.
    /// On Lua error, logs it and returns the last good scene (or the error if there is none).
    pub fn update(&mut self, t: f32, features: &Features) -> anyhow::Result<SceneDesc> {
        let _ = (t, features, &self.lua, &self.path, &self.last_good);
        todo!("call update, parse table into SceneDesc")
    }
}
