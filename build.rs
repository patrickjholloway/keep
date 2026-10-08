//! build.rs — compile shaders/*.{comp,vert,frag} (GLSL 450) to SPIR-V with naga.
//! `.wgsl` files go through naga's WGSL frontend instead (needed for atomics: surface.wgsl).
//! naga has no `#include`, so `#include "x.glsl"` lines are inlined textually here.
//! Output: $OUT_DIR/<name>.spv, pulled in by src/gpu/shaders.rs via include_bytes!.
use std::{env, fs, path::Path};

fn inline_includes(src: &str, dir: &Path) -> String {
    src.lines()
        .map(|line| match line.trim().strip_prefix("#include \"").and_then(|s| s.strip_suffix('"')) {
            Some(f) => fs::read_to_string(dir.join(f)).unwrap_or_else(|e| panic!("include {f}: {e}")),
            None => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    let dir = Path::new("shaders");
    let out = env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed=shaders");
    for (file, stage) in [
        ("field.comp", naga::ShaderStage::Compute),
        ("particle.vert", naga::ShaderStage::Vertex),
        ("particle.frag", naga::ShaderStage::Fragment),
        ("tonemap.vert", naga::ShaderStage::Vertex),
        ("tonemap.frag", naga::ShaderStage::Fragment),
        ("glitch_sync.comp", naga::ShaderStage::Compute),
        ("glitch_ring.comp", naga::ShaderStage::Compute),
        ("glitch_smear.comp", naga::ShaderStage::Compute),
        ("glitch_disp.comp", naga::ShaderStage::Compute),
        ("glitch_warp.comp", naga::ShaderStage::Compute),
        ("glitch_spectral.comp", naga::ShaderStage::Compute),
        ("glitch_crush.comp", naga::ShaderStage::Compute),
        ("glitch_stats.comp", naga::ShaderStage::Compute),
    ] {
        let src = inline_includes(&fs::read_to_string(dir.join(file)).unwrap(), dir);
        let module = naga::front::glsl::Frontend::default()
            .parse(&naga::front::glsl::Options::from(stage), &src)
            .unwrap_or_else(|e| panic!("{file}: {}", e.emit_to_string(&src)));
        write_spv(&module, &src, file, &out);
    }
    // WGSL shaders (naga's GLSL frontend has no atomics; its WGSL frontend does).
    for file in ["surface.wgsl"] {
        let src = fs::read_to_string(dir.join(file)).unwrap();
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{file}: {}", e.emit_to_string(&src)));
        write_spv(&module, &src, file, &out);
    }
}

/// Validate a parsed module and write `$OUT_DIR/<file>.spv`.
fn write_spv(module: &naga::Module, src: &str, file: &str, out: &str) {
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(module)
        .unwrap_or_else(|e| panic!("{file}: {}", e.emit_to_string(src)));
    let opts = naga::back::spv::Options { lang_version: (1, 3), ..Default::default() };
    let words = naga::back::spv::write_vec(module, &info, &opts, None).expect("spv");
    fs::write(Path::new(out).join(format!("{file}.spv")), bytemuck_cast(&words)).unwrap();
}

fn bytemuck_cast(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
