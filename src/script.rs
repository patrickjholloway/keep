//! script.rs — the Lua topcoat. Loads a scene script, hot-reloads it on save, and each frame
//! calls its global `update(t, features)` which returns a scene table (see docs/ARCHITECTURE.md
//! "Lua scene table"). Errors in Lua never crash the app: the last good SceneDesc is kept.
//!
//! The host installs a small `hyper` library before running the script (see `PRELUDE`):
//!   * constructors that just build the plain tables the parser understands
//!     (`hyper.rotate("xw", a)`, `hyper.duocylinder{r1=1, r2=.6}`, ...), so scripts can use
//!     either style;
//!   * helper math for choreography (`lerp`, `smoothstep`, `clamp`, `pulse`, `ease`, `tau`).
use std::{
    path::{Path, PathBuf},
    sync::mpsc::{channel, Receiver},
};

use anyhow::{anyhow, bail, Context};
use glam::{Vec3, Vec4};
use mlua::{Lua, Table, Value};
use notify::{RecursiveMode, Watcher};

use crate::{
    audio::Features,
    field::{Field, Op, Prim, Shape},
    glitch::{self, GlitchDesc, Lfo, LfoShape, Patch},
    math4d::{Affine4, Plane},
    scene::{Camera, SceneDesc},
};

/// Lua source installed before every (re)load. Pure Lua: easy to read, easy to extend.
const PRELUDE: &str = r#"
hyper = {}
hyper.tau = 2 * math.pi
-- transform steps (use inside `transform = { ... }` lists)
function hyper.rotate(plane, a)        return { "rotate", plane, a } end
function hyper.shear(dst, src, k)      return { "shear", dst, src, k } end
function hyper.translate(x, y, z, w)   return { "translate", x or 0, y or 0, z or 0, w or 0 } end
function hyper.scale(x, y, z, w)
  if y == nil then y, z, w = x, x, x end
  return { "scale", x, y, z, w }
end
-- shapes: pass a table of options (op, k, transform, plus shape params)
local function shape(name) return function(o) o = o or {}; o.shape = name; return o end end
hyper.hypersphere = shape("hypersphere")
hyper.tesseract   = shape("tesseract")
hyper.duocylinder = shape("duocylinder")
hyper.clifford    = shape("clifford")
-- choreography math
function hyper.clamp(x, a, b)    return math.max(a, math.min(b, x)) end
function hyper.lerp(a, b, s)     return a + (b - a) * s end
function hyper.smoothstep(e0, e1, x)
  local s = hyper.clamp((x - e0) / (e1 - e0), 0, 1); return s * s * (3 - 2 * s)
end
-- sharp spike at phase 0 decaying over the beat (phase in 0..1)
function hyper.pulse(phase, sharp) return math.exp(-(sharp or 6) * phase) end
-- cosine ease 0..1
function hyper.ease(s) s = hyper.clamp(s, 0, 1); return 0.5 - 0.5 * math.cos(math.pi * s) end
-- triangle wave in 0..1 with given period
function hyper.tri(t, period) local p = (t / period) % 1; return 1 - math.abs(2 * p - 1) end

-- glitch patch bay (image-plane effects; see README "Glitch modules"). State persists across
-- frames until changed; Rust reads it after every update() and resolves it per frame.
glitch = { _set = {}, _patch = {}, _lfo = {}, _src = {} }
-- base value; complex params take (re, im): glitch.set("warp.a", 0.9, 0.3)
function glitch.set(name, v, im)
  if im ~= nil then glitch._set[name .. ".re"] = v; glitch._set[name .. ".im"] = im
  else glitch._set[name] = v end
end
-- cable: dst += gain * src + offset (one cable per src/dst pair; re-patching replaces it)
function glitch.patch(src, dst, gain, offset)
  glitch._patch[src .. "->" .. dst] = { src = src, dst = dst, gain = gain or 1, offset = offset or 0 }
end
function glitch.unpatch(src, dst) glitch._patch[src .. "->" .. dst] = nil end
-- unipolar 0..1 oscillator usable as a source: glitch.lfo("wobble", "sine", 0.5 [, phase])
function glitch.lfo(name, shape, rate, phase) glitch._lfo[name] = { shape = shape, rate = rate, phase = phase or 0 } end
-- script-computed source value (e.g. an envelope): glitch.source("drop", x)
function glitch.source(name, v) glitch._src[name] = v end
function glitch.clear() glitch._set, glitch._patch, glitch._src = {}, {}, {} end
"#;

pub struct ScriptHost {
    lua: Lua,
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
        let path = path.canonicalize().with_context(|| format!("script {}", path.display()))?;
        let lua = Lua::new();
        exec_file(&lua, &path)?;
        // Watch the *directory*: editors often save by writing a temp file and renaming it,
        // which would orphan a watch on the file inode itself.
        let (tx, events) = channel();
        let mut watcher = notify::recommended_watcher(tx)?;
        let dir = path.parent().unwrap_or(Path::new("."));
        watcher.watch(dir, RecursiveMode::NonRecursive)?;
        Ok(ScriptHost { lua, path, _watcher: watcher, events, last_good: None })
    }

    /// Re-execute the script if it changed on disk. Returns true if a reload happened.
    /// A broken edit is reported and the previous definitions stay active.
    pub fn poll_reload(&mut self) -> bool {
        let name = self.path.file_name();
        let mut changed = false;
        while let Ok(ev) = self.events.try_recv() {
            if let Ok(ev) = ev {
                let relevant = matches!(ev.kind, notify::EventKind::Modify(_) | notify::EventKind::Create(_))
                    && ev.paths.iter().any(|p| p.file_name() == name);
                changed |= relevant;
            }
        }
        if !changed {
            return false;
        }
        match exec_file(&self.lua, &self.path) {
            Ok(()) => { eprintln!("[script] reloaded {}", self.path.display()); true }
            Err(e) => { eprintln!("[script] reload failed, keeping previous script: {e:#}"); false }
        }
    }

    /// Call Lua `update(t, features)` and convert the returned table into a `SceneDesc`.
    /// On Lua error, logs it and returns the last good scene (or the error if there is none).
    pub fn update(&mut self, t: f32, features: &Features) -> anyhow::Result<SceneDesc> {
        match self.try_update(t, features) {
            Ok(desc) => { self.last_good = Some(desc.clone()); Ok(desc) }
            Err(e) => match &self.last_good {
                Some(good) => { eprintln!("[script] t={t:.2}s: {e:#} (keeping last good scene)"); Ok(good.clone()) }
                None => Err(e),
            },
        }
    }

    fn try_update(&self, t: f32, f: &Features) -> anyhow::Result<SceneDesc> {
        let update: mlua::Function = self.lua.globals().get("update").map_err(lua_err)
            .context("script defines no global function `update(t, f)`")?;
        let ft = self.lua.create_table().map_err(lua_err)?;
        for (k, v) in [("bass", f.bass), ("mid", f.mid), ("high", f.high), ("onset", f.onset),
                       ("rms", f.rms), ("beat_phase", f.beat_phase)] {
            ft.set(k, v).map_err(lua_err)?;
        }
        let ret: Table = update.call((t, ft)).map_err(lua_err).context("update(t, f)")?;
        let mut desc = parse_scene(&ret)?;
        if let Some(g) = self.lua.globals().get::<Option<Table>>("glitch").map_err(lua_err)? {
            desc.glitch = parse_glitch(&g).context("glitch")?;
        }
        Ok(desc)
    }
}

/// mlua errors aren't Send+Sync-friendly for anyhow in every config; stringify them.
fn lua_err(e: mlua::Error) -> anyhow::Error { anyhow!("{e}") }

fn exec_file(lua: &Lua, path: &Path) -> anyhow::Result<()> {
    let src = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    lua.load(PRELUDE).set_name("hyper-prelude").exec().map_err(lua_err)?;
    lua.load(&src).set_name(path.display().to_string()).exec().map_err(lua_err)
        .with_context(|| format!("executing {}", path.display()))
}

// ---------------------------------------------------------------- table -> SceneDesc parsing

fn num(t: &Table, k: &str, default: f32) -> anyhow::Result<f32> {
    match t.get::<Value>(k).map_err(lua_err)? {
        Value::Nil => Ok(default),
        Value::Integer(i) => Ok(i as f32),
        Value::Number(n) => Ok(n as f32),
        other => bail!("field `{k}` must be a number, got {}", other.type_name()),
    }
}

fn idx_num(t: &Table, i: usize) -> anyhow::Result<f32> {
    match t.get::<Value>(i).map_err(lua_err)? {
        Value::Integer(v) => Ok(v as f32),
        Value::Number(v) => Ok(v as f32),
        Value::Nil => Ok(0.0),
        other => bail!("element {i} must be a number, got {}", other.type_name()),
    }
}

fn vec_n(t: &Table) -> anyhow::Result<Vec<f32>> {
    (1..=t.raw_len()).map(|i| idx_num(t, i)).collect()
}

fn axis(s: &str) -> anyhow::Result<usize> {
    Ok(match s { "x" => 0, "y" => 1, "z" => 2, "w" => 3, _ => bail!("unknown axis `{s}`") })
}

/// A transform list `{ {"rotate","xw",a}, ... }` composed in order (first entry applies first).
fn parse_transform(list: Option<Table>) -> anyhow::Result<Affine4> {
    let mut acc = Affine4::IDENTITY;
    let Some(list) = list else { return Ok(acc) };
    for (i, step) in list.sequence_values::<Table>().enumerate() {
        let step = step.map_err(lua_err).with_context(|| format!("transform[{}] must be a table", i + 1))?;
        let op: String = step.get(1).map_err(lua_err)?;
        let s = |j: usize| -> anyhow::Result<String> { step.get(j).map_err(lua_err) };
        let m = match op.as_str() {
            "rotate" => {
                let p = s(2)?;
                Affine4::rotation(Plane::parse(&p).ok_or_else(|| anyhow!("unknown plane `{p}`"))?, idx_num(&step, 3)?)
            }
            "shear" => Affine4::shear(axis(&s(2)?)?, axis(&s(3)?)?, idx_num(&step, 4)?),
            "translate" => Affine4::translation(Vec4::new(idx_num(&step, 2)?, idx_num(&step, 3)?, idx_num(&step, 4)?, idx_num(&step, 5)?)),
            "scale" => Affine4::scale(Vec4::new(idx_num(&step, 2)?, idx_num(&step, 3)?, idx_num(&step, 4)?, idx_num(&step, 5)?)),
            other => bail!("transform[{}]: unknown op `{other}`", i + 1),
        };
        acc = m.compose(&acc);
    }
    Ok(acc)
}

fn parse_prim(p: &Table, first: bool) -> anyhow::Result<Prim> {
    let name: String = p.get("shape").map_err(lua_err).context("missing `shape`")?;
    let shape = match name.as_str() {
        "hypersphere" => Shape::Hypersphere { r: num(p, "r", 1.0)? },
        "tesseract" => {
            let h = match p.get::<Option<Table>>("half").map_err(lua_err)? {
                Some(t) => { let v = vec_n(&t)?; Vec4::new(v.first().copied().unwrap_or(0.5), v.get(1).copied().unwrap_or(0.5), v.get(2).copied().unwrap_or(0.5), v.get(3).copied().unwrap_or(0.5)) }
                None => Vec4::splat(num(p, "size", 0.5)?),
            };
            Shape::Tesseract { half: h }
        }
        "duocylinder" => Shape::Duocylinder { r1: num(p, "r1", 1.0)?, r2: num(p, "r2", 0.6)? },
        "clifford" => Shape::Clifford { r1: num(p, "r1", 1.0)?, r2: num(p, "r2", 1.0)?, thickness: num(p, "thickness", 0.1)? },
        other => bail!("unknown shape `{other}`"),
    };
    let op_name: Option<String> = p.get("op").map_err(lua_err)?;
    let op = match op_name.as_deref() {
        None | Some("union") => Op::Union,
        Some("smooth_union") => Op::SmoothUnion { k: num(p, "k", 0.2)? },
        Some("intersect") => Op::Intersect,
        Some("subtract") => Op::Subtract,
        Some(o) => bail!("unknown op `{o}`"),
    };
    let _ = first; // the first prim's op is ignored by the fold; accepted for symmetry
    Ok(Prim { shape, op, local: parse_transform(p.get("transform").map_err(lua_err)?)? })
}

fn parse_vec3(t: &Table, k: &str) -> anyhow::Result<Option<Vec3>> {
    Ok(match t.get::<Option<Table>>(k).map_err(lua_err)? {
        Some(v) => Some(Vec3::new(idx_num(&v, 1)?, idx_num(&v, 2)?, idx_num(&v, 3)?)),
        None => None,
    })
}

/// Convert the table returned by `update` into Rust types. Every key is optional.
pub fn parse_scene(t: &Table) -> anyhow::Result<SceneDesc> {
    let object = parse_transform(t.get("transform").map_err(lua_err)?).context("transform")?;
    let mut prims = Vec::new();
    if let Some(list) = t.get::<Option<Table>>("field").map_err(lua_err)? {
        for (i, p) in list.sequence_values::<Table>().enumerate() {
            let p = p.map_err(lua_err)?;
            prims.push(parse_prim(&p, i == 0).with_context(|| format!("field[{}]", i + 1))?);
        }
    }
    if prims.is_empty() {
        prims.push(Prim { shape: Shape::Hypersphere { r: 1.0 }, op: Op::Union, local: Affine4::IDENTITY });
    }
    if prims.len() > crate::scene::MAX_PRIMS {
        eprintln!("[script] field has {} prims; only the first {} are used", prims.len(), crate::scene::MAX_PRIMS);
    }
    let temperature = match t.get::<Option<Table>>("temperature").map_err(lua_err)? {
        Some(tt) => (idx_num(&tt, 1)?, idx_num(&tt, 2)?),
        None => (1200.0, 9000.0),
    };
    let camera = match t.get::<Option<Table>>("camera").map_err(lua_err)? {
        Some(c) => Some(Camera {
            eye: parse_vec3(&c, "eye")?.unwrap_or(Vec3::new(0.0, 0.0, 6.0)),
            target: parse_vec3(&c, "target")?.unwrap_or(Vec3::ZERO),
            fov_y: num(&c, "fov", 50.0)?.to_radians(),
            roll: num(&c, "roll", 0.0)?,
        }),
        None => None,
    };
    Ok(SceneDesc {
        object,
        field: Field { prims },
        w_slice: num(t, "w_slice", 0.0)?,
        surface_width: num(t, "surface_width", 0.02)?,
        particle_radius: num(t, "particle_radius", 0.004)?,
        temperature,
        reflectivity: num(t, "reflectivity", 0.6)?,
        exposure: num(t, "exposure", 1.0)?,
        palette: num(t, "palette", 0.0)?,
        beat_rise: num(t, "beat_rise", 0.0)?,
        camera,
        glitch: GlitchDesc::default(),
    })
}

/// Read the prelude's `glitch` state tables into a GlitchDesc, validating names.
pub fn parse_glitch(g: &Table) -> anyhow::Result<GlitchDesc> {
    let mut d = GlitchDesc::default();
    let tbl = |k: &str| -> anyhow::Result<Table> { g.get::<Table>(k).map_err(lua_err) };
    for pair in tbl("_lfo")?.pairs::<String, Table>() {
        let (name, l) = pair.map_err(lua_err)?;
        let shape: String = l.get("shape").map_err(lua_err)?;
        d.lfos.insert(name.clone(), Lfo { shape: LfoShape::parse(&shape).with_context(|| format!("lfo `{name}`"))?, rate: num(&l, "rate", 1.0)?, phase: num(&l, "phase", 0.0)? });
    }
    for pair in tbl("_src")?.pairs::<String, f32>() {
        let (name, v) = pair.map_err(lua_err)?;
        d.sources.insert(name, v);
    }
    for pair in tbl("_set")?.pairs::<String, f32>() {
        let (name, v) = pair.map_err(lua_err)?;
        let i = glitch::param_index(&name).ok_or_else(|| anyhow!("glitch.set: unknown parameter `{name}`"))?;
        d.sets.insert(i, v);
    }
    let mut patches: Vec<(String, Patch)> = Vec::new();
    for pair in tbl("_patch")?.pairs::<String, Table>() {
        let (key, p) = pair.map_err(lua_err)?;
        let src: String = p.get("src").map_err(lua_err)?;
        let dst: String = p.get("dst").map_err(lua_err)?;
        let di = glitch::param_index(&dst).ok_or_else(|| anyhow!("glitch.patch: unknown destination `{dst}`"))?;
        if !d.is_known_source(&src) { bail!("glitch.patch: unknown source `{src}` (define it with glitch.lfo / glitch.source)"); }
        patches.push((key, Patch { src, dst: di, gain: num(&p, "gain", 1.0)?, offset: num(&p, "offset", 0.0)? }));
    }
    patches.sort_by(|a, b| a.0.cmp(&b.0)); // Lua table order is unspecified; keep it deterministic
    d.patches = patches.into_iter().map(|p| p.1).collect();
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scripts_dir() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts") }

    fn feats(bass: f32, onset: f32, phase: f32) -> Features {
        Features { bass, mid: 0.3, high: 0.4, onset, rms: 0.5, beat_phase: phase }
    }

    #[test]
    fn scene_lua_loads_and_reacts() {
        let mut h = ScriptHost::load(&scripts_dir().join("scene.lua")).unwrap();
        let a = h.update(1.0, &feats(0.0, 0.0, 0.0)).unwrap();
        let b = h.update(1.0, &feats(1.0, 1.0, 0.0)).unwrap();
        assert_eq!(a.field.prims.len(), 2);
        assert_ne!(a.object.a, b.object.a, "bass should change the rotation");
        assert!(b.particle_radius > a.particle_radius);
    }

    #[test]
    fn exemplar_loads_and_reacts() {
        let mut h = ScriptHost::load(&scripts_dir().join("exemplar.lua")).unwrap();
        let quiet = h.update(10.0, &feats(0.0, 0.0, 0.5)).unwrap();
        let loud = h.update(10.0, &feats(1.0, 1.0, 0.0)).unwrap();
        assert!(quiet.camera.is_some());
        assert!(loud.temperature.1 > quiet.temperature.1, "onset should heat");
        assert_ne!(quiet.object.a, loud.object.a);
        // sweep over time: w_slice must move
        let later = h.update(25.0, &feats(0.0, 0.0, 0.5)).unwrap();
        assert_ne!(quiet.w_slice, later.w_slice);
        for t in 0..120 { h.update(t as f32 * 0.5, &feats(0.5, 0.2, 0.3)).unwrap(); }
    }

    #[test]
    fn error_keeps_last_good_and_reload_works() {
        let dir = std::env::temp_dir().join(format!("keep-script-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("s.lua");
        std::fs::write(&p, "function update(t,f) if t > 1 then error('boom') end \
            return { field = { hyper.hypersphere{ r = 0.5 } }, w_slice = t } end").unwrap();
        let mut h = ScriptHost::load(&p).unwrap();
        let good = h.update(0.5, &Features::default()).unwrap();
        let after = h.update(2.0, &Features::default()).unwrap();
        assert_eq!(good.w_slice, after.w_slice);
        // a broken file on reload must not replace the working script
        std::fs::write(&p, "function update(").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));
        h.poll_reload();
        assert_eq!(h.update(0.25, &Features::default()).unwrap().w_slice, 0.25);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn glitch_api_parses_and_validates() {
        let lua = Lua::new();
        lua.load(PRELUDE).exec().unwrap();
        lua.load(r#"glitch.set("sync.depth", 0.1); glitch.set("warp.a", 0.5, 0.25)
            glitch.lfo("w", "tri", 2); glitch.patch("w", "ring.depth", 0.5, 0.1)
            glitch.patch("bass", "sync.depth", 0.3); glitch.patch("bass", "sync.depth", 0.4)"#).exec().unwrap();
        let g: Table = lua.globals().get("glitch").unwrap();
        let d = parse_glitch(&g).unwrap();
        assert_eq!(d.patches.len(), 2, "re-patching the same cable replaces it");
        assert_eq!(d.sets[&glitch::param_index("warp.a.im").unwrap()], 0.25);
        let f = Features { bass: 1.0, ..Default::default() };
        let v = glitch::resolve_values(&d, 0.25, &f, 0.0);
        assert!((v[glitch::param_index("sync.depth").unwrap()] - 0.5).abs() < 1e-6);
        assert!((v[glitch::param_index("ring.depth").unwrap()] - (0.5 * 1.0 + 0.1)).abs() < 1e-6);
        lua.load(r#"glitch.patch("nope", "sync.depth", 1)"#).exec().unwrap();
        assert!(parse_glitch(&g).is_err(), "unknown source must be rejected");
        lua.load(r#"glitch.clear(); glitch.set("sync.bogus", 1)"#).exec().unwrap();
        assert!(parse_glitch(&g).is_err(), "unknown parameter must be rejected");
    }

    #[test]
    fn transform_order_first_applies_first() {
        let lua = Lua::new();
        lua.load(PRELUDE).exec().unwrap();
        let t: Table = lua.load("return { transform = { hyper.translate(1,0,0,0), hyper.scale(2) } }").eval().unwrap();
        let d = parse_scene(&t).unwrap();
        assert_eq!(d.object.apply(Vec4::ZERO), Vec4::new(2.0, 0.0, 0.0, 0.0));
    }
}
