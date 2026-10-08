//! GPU smoke tests. They need MoltenVK ($KEEP_VULKAN_LIB, set by `mise exec --`); without it
//! they print a note and pass, so `cargo test` stays green on machines without a GPU.
//! Output image: target/gpu-test/hypersphere.png
use glam::{Mat4, Vec3};

use super::{renderer::{jittered_grid, CloudSpec}, Renderer};
use crate::scene::{GpuPrimitive, SceneParams, MAX_PRIMS};

/// GPU tests run one at a time: several MoltenVK devices created concurrently by the parallel
/// test harness is not something we need to support.
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn have_gpu() -> bool {
    std::env::var("KEEP_VULKAN_LIB").map(|p| std::path::Path::new(&p).exists()).unwrap_or(false)
}

/// Build SceneParams by hand (independent of the Lua/field modules): a unit hypersphere
/// sliced at w = 0.4 -> a 3D sphere of radius sqrt(1 - 0.16) ≈ 0.92.
fn hypersphere_params(width: u32, height: u32, count: u32) -> SceneParams {
    let eye = Vec3::new(0.0, 0.6, 3.2);
    let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    let fov = 45f32.to_radians();
    let mut proj = Mat4::perspective_rh(fov, width as f32 / height as f32, 0.01, 100.0);
    proj.y_axis.y *= -1.0;
    let mut prims = [GpuPrimitive::default(); MAX_PRIMS];
    prims[0] = GpuPrimitive { kind_op: [0, 0, 0, 0], params: [1.0, 0.0, 0.0, 0.0], inv_a: Mat4::IDENTITY.to_cols_array_2d(), t: [0.0; 4] };
    SceneParams {
        inv_a: Mat4::IDENTITY.to_cols_array_2d(),
        t: [0.0; 4],
        slice: [0.4, 0.03, 0.01, 0.0],
        material: [1800.0, 7000.0, 0.6, 0.6],
        audio: [0.0; 4],
        view: view.to_cols_array_2d(),
        proj: proj.to_cols_array_2d(),
        cam_pos: [eye.x, eye.y, eye.z, fov],
        counts: [count, 1, 0, 0],
        prims,
    }
}

#[test]
fn jittered_grid_fills_cube() {
    let spec = CloudSpec { count: 10_000, half_extent: 1.5, seed: 7 };
    let pts = jittered_grid(&spec);
    assert_eq!(pts.len(), 10_000);
    assert!(pts.iter().all(|p| p[..3].iter().all(|c| c.abs() <= 1.5) && (0.0..1.0).contains(&p[3])));
    // Every octant gets roughly 1/8 of the points (the scrambled cell order covers the cube).
    let pos = pts.iter().filter(|p| p[0] > 0.0 && p[1] > 0.0 && p[2] > 0.0).count();
    assert!((1000..1500).contains(&pos), "octant count {pos}");
}

#[test]
fn png_roundtrip_header() {
    let png = super::png::encode_rgba(2, 1, &[255, 0, 0, 255, 0, 255, 0, 255]);
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
}

#[test]
fn render_hypersphere_slice_headless() {
    if !have_gpu() {
        eprintln!("skipping: KEEP_VULKAN_LIB not set (run via `mise exec -- cargo test`)");
        return;
    }
    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (w, h, n) = (640u32, 480u32, 1_000_000u32);
    let (mut r, target) = Renderer::new_offscreen(CloudSpec { count: n, half_extent: 1.5, seed: 1 }, w, h).unwrap();
    let params = hypersphere_params(w, h, n);
    // Two frames: the second exercises fence reuse + descriptor/state reuse.
    r.render_offscreen(&target, &params).unwrap();
    let px = r.render_offscreen(&target, &params).unwrap().to_vec();
    assert_eq!(px.len(), (w * h * 4) as usize);

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/gpu-test");
    std::fs::create_dir_all(&dir).unwrap();
    let mut target = target;
    target.destroy(&r.ctx);
    super::png::write_rgba(&dir.join("hypersphere.png"), w, h, &px).unwrap();

    // Sanity: corners are (near) black, and a ring around the image centre is lit.
    let at = |x: u32, y: u32| { let i = ((y * w + x) * 4) as usize; px[i] as u32 + px[i + 1] as u32 + px[i + 2] as u32 };
    assert!(at(2, 2) < 30, "corner not dark: {}", at(2, 2));
    let lit = px.chunks_exact(4).filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 60).count();
    let frac = lit as f32 / (w * h) as f32;
    eprintln!("lit fraction {frac:.3}");
    assert!(frac > 0.05 && frac < 0.8, "lit fraction {frac}");
}

/// Render the hypersphere twice — clean and with `g` — at w×h; returns (clean, glitched) RGBA8.
fn render_pair(w: u32, h: u32, g: crate::glitch::GlitchParams) -> (Vec<u8>, Vec<u8>) {
    let n = 400_000u32;
    let (mut r, mut target) = Renderer::new_offscreen(CloudSpec { count: n, half_extent: 1.5, seed: 1 }, w, h).unwrap();
    let params = hypersphere_params(w, h, n);
    let clean = r.render_offscreen(&target, &params).unwrap().to_vec();
    r.glitch = g;
    let glitched = r.render_offscreen(&target, &params).unwrap().to_vec();
    assert!(r.mean_luma() > 0.0, "stats pass should see the lit sphere");
    target.destroy(&r.ctx);
    (clean, glitched)
}

fn set(d: &mut crate::glitch::GlitchDesc, name: &str, v: f32) {
    d.sets.insert(crate::glitch::param_index(name).unwrap(), v);
}

#[test]
fn spectral_gate_identity_roundtrips_on_gpu() {
    // At exactly the FFT working-image resolution (1024×512) the composite subtracts the same
    // down(orig) it filtered, so a spectral gate that keeps every band (atten 0, gain 1 outside, no phase scramble) at mix 1
    // must give back the clean frame: forward 2D FFT -> inverse 2D FFT == identity.
    if !have_gpu() { eprintln!("skipping: no KEEP_VULKAN_LIB"); return; }
    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut d = crate::glitch::GlitchDesc::default();
    set(&mut d, "spectral.mix", 1.0);
    set(&mut d, "spectral.atten", 0.0);
    let g = crate::glitch::resolve(&d, 0.0, &Default::default(), 0.0, 0);
    assert!(g.spec_on());
    let (clean, out) = render_pair(super::glitch::FFT_IW, super::glitch::FFT_IH, g);
    let max = clean.iter().zip(&out).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
    eprintln!("fft roundtrip max 8-bit diff {max}");
    assert!(max <= 3, "GPU FFT roundtrip differs by {max}");
    // and removing a band really changes the picture
    set(&mut d, "spectral.atten", 1.0);
    let g = crate::glitch::resolve(&d, 0.0, &Default::default(), 0.0, 0);
    let (clean, out) = render_pair(super::glitch::FFT_IW, super::glitch::FFT_IH, g);
    let diff: u64 = clean.iter().zip(&out).map(|(a, b)| (*a as i64 - *b as i64).unsigned_abs()).sum();
    assert!(diff > 10_000, "band-stop should alter the frame ({diff})");
}

#[test]
fn every_glitch_module_runs_and_changes_the_frame() {
    if !have_gpu() { eprintln!("skipping: no KEEP_VULKAN_LIB"); return; }
    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (w, h) = (480u32, 270u32);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/gpu-test");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, v) in [("sync.depth", 0.2), ("ring.depth", 1.0), ("smear.depth", 1.0), ("dispersion.depth", 40.0),
                      ("warp.amount", 1.0), ("spectral.mix", 1.0), ("crush.depth", 0.9)] {
        let mut d = crate::glitch::GlitchDesc::default();
        set(&mut d, name, v);
        set(&mut d, "sync.block", 1.0);
        set(&mut d, "warp.c.re", 0.4);
        set(&mut d, "warp.a.im", 0.5);
        let g = crate::glitch::resolve(&d, 0.5, &Default::default(), 0.0, 0);
        let (clean, out) = render_pair(w, h, g);
        let diff: u64 = clean.iter().zip(&out).map(|(a, b)| (*a as i64 - *b as i64).unsigned_abs()).sum();
        let lit = out.chunks_exact(4).filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 60).count();
        eprintln!("{name}: diff {diff}, lit {lit}");
        super::png::write_rgba(&dir.join(format!("glitch-{name}.png")), w, h, &out).unwrap();
        assert!(diff > 5_000, "{name} did not change the frame");
        assert!(lit > 200, "{name} blacked out the frame");
    }
}

/// The surface pass sees the hypersphere slice: a sphere of radius sqrt(1 - 0.4²) ≈ 0.917
/// around the origin, smooth (low roughness), covering nearly every octahedral bin.
#[test]
fn surface_descriptor_reads_the_slice() {
    if !have_gpu() { return; }
    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (w, h, n) = (320u32, 240u32, 1_000_000u32);
    let (mut r, mut target) = Renderer::new_offscreen(CloudSpec { count: n, half_extent: 1.5, seed: 1 }, w, h).unwrap();
    let params = hypersphere_params(w, h, n);
    // Frame 1 bins around (0,0,0); frame 2 around frame 1's centroid (also ~0).
    r.render_offscreen(&target, &params).unwrap();
    r.render_offscreen(&target, &params).unwrap();
    let s = r.surface.clone();
    target.destroy(&r.ctx);
    let expect = (1.0f32 - 0.16).sqrt();
    let lit = s.bins.iter().filter(|b| b.count > 0).count();
    let mean_r = s.bins.iter().map(|b| b.count as f32 * b.radius).sum::<f32>() / s.total as f32;
    let mean_rough = s.bins.iter().map(|b| b.count as f32 * b.rough).sum::<f32>() / s.total as f32;
    let binned: u32 = s.bins.iter().map(|b| b.count).sum();
    eprintln!("surface: total {} lit bins {lit} mean r {mean_r:.3} (expect {expect:.3}) rough {mean_rough:.3} centroid {:?}", s.total, s.centroid);
    assert!(s.total > 10_000, "visible beads {}", s.total);
    assert_eq!(binned, s.total, "every visible bead lands in exactly one bin");
    assert!(lit > 950, "lit bins {lit}");
    assert!((mean_r - expect).abs() < 0.03, "mean radius {mean_r}");
    assert!(mean_rough < 0.1, "roughness {mean_rough}");
    assert!(s.centroid.iter().all(|c| c.abs() < 0.02), "centroid {:?}", s.centroid);
}
