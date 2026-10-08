//! GPU smoke tests. They need MoltenVK ($KEEP_VULKAN_LIB, set by `mise exec --`); without it
//! they print a note and pass, so `cargo test` stays green on machines without a GPU.
//! Output image: target/gpu-test/hypersphere.png
use glam::{Mat4, Vec3};

use super::{renderer::{jittered_grid, CloudSpec}, Renderer};
use crate::scene::{GpuPrimitive, SceneParams, MAX_PRIMS};

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
