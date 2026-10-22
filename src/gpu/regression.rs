//! Render regression tests: a fixed scene (`tests/render/regression.lua`) driven through
//! `keep render`'s own per-frame step (`offline::render_frame`) on the real GPU, headless, and
//! compared against small reference frames committed in `tests/render/`.
//!
//! * Inputs are fixed: features are a function of the frame index (no audio), the particle cloud
//!   is `keep render`'s (seed 1), 320×180, frames rendered in order from t = 0 at 30 fps (a frame
//!   reads the previous frame's mean luminance, so frames cannot be rendered out of order).
//! * Comparison, per reference frame, in 8-bit levels over R, G, B and A: the number of pixels
//!   whose largest channel difference exceeds `PIXEL_LEVELS` (8), which must be 0, and the mean
//!   absolute error, which must be ≤ `MAE_TOLERANCE` (0.25); either over its tolerance fails.
//!   `tests/render/README.md` has the numbers behind the tolerances.
//! * On failure the actual frame and an amplified diff are written to
//!   `target/render-regression/` (gitignored) and the message names the frame, every metric
//!   value, the tolerances and those paths.
//! * A frame must also be lit (`LIT_BAND`), checked before any comparison and before a bless,
//!   so an all-black render can never become a reference.
//! * Bless: `KEEP_BLESS=1 mise exec -- cargo test render_regression` rewrites the references,
//!   then compares against what it wrote. Without it a missing reference fails and says so.
//! * Without MoltenVK (`KEEP_VULKAN_LIB` unset or pointing at a missing file) the GPU tests
//!   print `skipping: …` and pass, like the other GPU tests.
//!
//! References are per-machine: blessed on an Apple M5 Pro through MoltenVK 1.4.2, where two
//! renders of a frame are byte-identical. On another GPU, driver, macOS or MoltenVK version the
//! test is expected to fail; re-bless there and look at the diff and actual frames before
//! committing. The README says what the comparison does and does not catch.
use std::path::{Path, PathBuf};

use super::{png, renderer::CloudSpec, tests::have_gpu, tests::GPU_LOCK, Renderer};
use crate::{audio::Features, offline, script::ScriptHost};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;
const FPS: u32 = 30;
const PARTICLES: u32 = 1_000_000;
/// Same cloud seed as `offline::render`.
const SEED: u64 = 1;
/// Frames compared against references, spread over the script's first three seconds.
const FRAMES: [usize; 3] = [0, 45, 90];
/// Mean absolute error over every channel of every pixel, 8-bit levels.
const MAE_TOLERANCE: f64 = 0.25;
/// A pixel "differs" when its largest channel difference exceeds this many 8-bit levels.
const PIXEL_LEVELS: u8 = 8;
/// Most differing pixels a frame may have: none. Two renders on the blessing machine are
/// byte-identical, so one pixel over `PIXEL_LEVELS` is a changed picture, not noise.
const DIFFERING_TOLERANCE: usize = 0;
/// Fraction of lit pixels (R+G+B > `LIT_SUM`) every compared frame must have. Not the other GPU
/// tests' 60: the tonemap's background glow alone puts ~15% of this frame's pixels over 60, so
/// a frame with no form at all would pass; it stays under 150.
const LIT_SUM: u32 = 150;
const LIT_BAND: (f32, f32) = (0.02, 0.5);
const SKIP: &str =
    "skipping: KEEP_VULKAN_LIB not set or missing (run via `mise exec -- cargo test`)";
const BLESS_CMD: &str = "KEEP_BLESS=1 mise exec -- cargo test render_regression";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn reference_path(frame: usize) -> PathBuf {
    root().join(format!("tests/render/frame-{frame:03}.png"))
}

fn blessing() -> bool {
    std::env::var("KEEP_BLESS").is_ok_and(|v| v == "1")
}

/// The fixed features of frame `i`: steady levels, a beat every half second.
fn features(i: usize) -> Features {
    Features {
        bass: 0.5,
        mid: 0.3,
        high: 0.4,
        onset: 0.0,
        rms: 0.3,
        beat_phase: ((i + 5) % 15) as f32 / 15.0,
    }
}

/// Render frames 0..=max(keep) of the regression script in order, with a fresh script host and
/// renderer; returns the RGBA8 pixels of each frame listed in `keep`.
fn render_frames(seed: u64, keep: &[usize]) -> Vec<(usize, Vec<u8>)> {
    let last = *keep.iter().max().unwrap();
    let mut script = ScriptHost::load(&root().join("tests/render/regression.lua")).unwrap();
    let cloud = CloudSpec {
        count: PARTICLES,
        half_extent: 1.5,
        seed,
    };
    let (mut renderer, mut target) = Renderer::new_offscreen(cloud, WIDTH, HEIGHT).unwrap();
    let mut out = Vec::new();
    for i in 0..=last {
        let t = i as f32 / FPS as f32;
        let f = features(i);
        let (_, px) = offline::render_frame(&mut script, &mut renderer, &target, t, &f, i).unwrap();
        if keep.contains(&i) {
            out.push((i, px.to_vec()));
        }
    }
    target.destroy(&renderer.ctx);
    out
}

fn lit_fraction(px: &[u8]) -> f32 {
    let lit = px
        .chunks_exact(4)
        .filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > LIT_SUM)
        .count();
    lit as f32 / (px.len() / 4) as f32
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Metric {
    /// Mean absolute error over all channels, 8-bit levels.
    mae: f64,
    /// Number of pixels whose largest channel difference exceeds `PIXEL_LEVELS`.
    differing: usize,
    /// Largest single channel difference.
    max: u8,
}

fn compare(a: &[u8], b: &[u8]) -> Metric {
    assert_eq!(a.len(), b.len(), "frame sizes differ");
    let mut sum = 0u64;
    let mut max = 0u8;
    let mut differing = 0usize;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let mut pmax = 0u8;
        for c in 0..4 {
            let d = pa[c].abs_diff(pb[c]);
            sum += d as u64;
            pmax = pmax.max(d);
        }
        max = max.max(pmax);
        differing += (pmax > PIXEL_LEVELS) as usize;
    }
    Metric {
        mae: sum as f64 / a.len() as f64,
        differing,
        max,
    }
}

/// Fraction of a `WIDTH`×`HEIGHT` frame's pixels that differ.
fn differing_fraction(m: &Metric) -> f64 {
    m.differing as f64 / (WIDTH * HEIGHT) as f64
}

fn fails(m: &Metric) -> bool {
    m.mae > MAE_TOLERANCE || m.differing > DIFFERING_TOLERANCE
}

/// |a − b| per colour channel, amplified ×8 so small drift is visible; alpha opaque.
fn diff_image(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(a.len());
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            out.push(pa[c].abs_diff(pb[c]).saturating_mul(8));
        }
        out.push(255);
    }
    out
}

// The reader keeps its own CRC-32 and Adler-32 (not png.rs's), so a writer bug cannot vouch
// for itself.
fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = (c >> 1) ^ (0xedb8_8320 & (c & 1).wrapping_neg());
        }
    }
    !c
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut s) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        s = (s + a) % 65521;
    }
    (s << 16) | a
}

/// Decode exactly what `png::encode_rgba` writes: IHDR, IDAT, IEND and nothing after; 8-bit RGBA,
/// no interlace; one zlib stream of stored deflate blocks; filter 0 on every row. Anything else is
/// refused, not guessed at.
fn decode_png(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let rest = bytes
        .strip_prefix(b"\x89PNG\r\n\x1a\n")
        .ok_or("not a PNG signature")?;
    let (mut pos, mut ihdr, mut idat, mut ended) = (0usize, None, Vec::new(), false);
    while pos < rest.len() {
        let take = |at: usize, n: usize| rest.get(at..at + n).ok_or("truncated chunk");
        let len = u32::from_be_bytes(take(pos, 4)?.try_into().unwrap()) as usize;
        let ty = take(pos + 4, 4)?;
        let data = take(pos + 8, len)?;
        let crc = u32::from_be_bytes(take(pos + 8 + len, 4)?.try_into().unwrap());
        if crc32(&rest[pos + 4..pos + 8 + len]) != crc {
            return Err(format!("bad CRC on {}", String::from_utf8_lossy(ty)));
        }
        // png.rs writes IHDR, then IDAT, then IEND, and nothing after it.
        match ty {
            b"IHDR" if ihdr.is_none() => ihdr = Some(data),
            b"IDAT" if ihdr.is_some() => idat.extend_from_slice(data),
            b"IEND" if pos + 12 == rest.len() => {
                ended = true;
                break;
            }
            other => {
                return Err(format!(
                    "unexpected chunk {} here",
                    String::from_utf8_lossy(other)
                ))
            }
        }
        pos += 12 + len;
    }
    if !ended {
        return Err("no IEND".into());
    }
    let ihdr = ihdr.ok_or("no IHDR")?;
    if ihdr.len() != 13 || ihdr[8..] != [8, 6, 0, 0, 0] {
        return Err("only 8-bit RGBA, no interlace".into());
    }
    let w = u32::from_be_bytes(ihdr[0..4].try_into().unwrap());
    let h = u32::from_be_bytes(ihdr[4..8].try_into().unwrap());
    if idat.get(..2) != Some(&[0x78, 0x01][..]) {
        return Err("unexpected zlib header".into());
    }
    let mut raw = Vec::new();
    let mut p = 2usize;
    loop {
        let head = *idat.get(p).ok_or("truncated deflate stream")?;
        if head & 0b110 != 0 {
            return Err("compressed deflate block (only stored blocks are read)".into());
        }
        if head > 1 {
            return Err("non-zero padding bits in a stored block header".into());
        }
        let lens = idat.get(p + 1..p + 5).ok_or("truncated stored block")?;
        let len = u16::from_le_bytes([lens[0], lens[1]]);
        if !len != u16::from_le_bytes([lens[2], lens[3]]) {
            return Err("stored block length check failed".into());
        }
        let block = idat
            .get(p + 5..p + 5 + len as usize)
            .ok_or("truncated stored block")?;
        raw.extend_from_slice(block);
        p += 5 + len as usize;
        if head & 1 == 1 {
            break;
        }
    }
    let sum = idat.get(p..p + 4).ok_or("missing adler32")?;
    if p + 4 != idat.len() {
        return Err("bytes after the zlib stream".into());
    }
    if u32::from_be_bytes(sum.try_into().unwrap()) != adler32(&raw) {
        return Err("adler32 mismatch".into());
    }
    let row = w as usize * 4 + 1;
    if raw.len() != row * h as usize {
        return Err(format!("{} bytes of scanlines for {w}×{h}", raw.len()));
    }
    let mut rgba = Vec::with_capacity(w as usize * h as usize * 4);
    for line in raw.chunks_exact(row) {
        if line[0] != 0 {
            return Err(format!("row filter {} (only filter 0 is read)", line[0]));
        }
        rgba.extend_from_slice(&line[1..]);
    }
    Ok((w, h, rgba))
}

fn read_reference(path: &Path) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (w, h, rgba) = decode_png(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    if (w, h) != (WIDTH, HEIGHT) {
        return Err(format!(
            "{}: {w}×{h}, expected {WIDTH}×{HEIGHT}",
            path.display()
        ));
    }
    Ok(rgba)
}

#[test]
fn render_regression_matches_references() {
    if !have_gpu() {
        eprintln!("{SKIP}");
        return;
    }
    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let frames = render_frames(SEED, &FRAMES);
    for (i, px) in &frames {
        let lit = lit_fraction(px);
        eprintln!("render_regression: frame {i} lit fraction {lit:.3}");
        assert!(
            lit > LIT_BAND.0 && lit < LIT_BAND.1,
            "frame {i}: lit fraction {lit:.4} outside {LIT_BAND:?}; refusing to compare or bless it"
        );
    }
    if blessing() {
        for (i, px) in &frames {
            let path = reference_path(*i);
            png::write_rgba(&path, WIDTH, HEIGHT, px).unwrap();
            eprintln!("render_regression: blessed {}", path.display());
        }
    }
    let out = root().join("target/render-regression");
    let mut failures = Vec::new();
    for (i, px) in &frames {
        let path = reference_path(*i);
        let actual = out.join(format!("frame-{i:03}.actual.png"));
        let diff = out.join(format!("frame-{i:03}.diff.png"));
        // A failure image left by an earlier run would outlive the fix that made it go away.
        std::fs::remove_file(&actual).ok();
        std::fs::remove_file(&diff).ok();
        if !path.exists() {
            failures.push(format!(
                "frame {i}: no reference at {}; bless with `{BLESS_CMD}`",
                path.display()
            ));
            continue;
        }
        let reference = read_reference(&path).unwrap_or_else(|e| panic!("frame {i}: {e}"));
        let m = compare(px, &reference);
        eprintln!(
            "render_regression: frame {i} mae {:.4} (tolerance {MAE_TOLERANCE}) differing {} px \
             {:.5} (tolerance {DIFFERING_TOLERANCE} px, >{PIXEL_LEVELS} levels) max {}",
            m.mae,
            m.differing,
            differing_fraction(&m),
            m.max
        );
        if fails(&m) {
            std::fs::create_dir_all(&out).unwrap();
            png::write_rgba(&actual, WIDTH, HEIGHT, px).unwrap();
            png::write_rgba(&diff, WIDTH, HEIGHT, &diff_image(px, &reference)).unwrap();
            failures.push(format!(
                "frame {i}: mae {:.4} (tolerance {MAE_TOLERANCE}), differing pixels {} \
                 ({:.5} of the frame; tolerance {DIFFERING_TOLERANCE}, >{PIXEL_LEVELS} levels), \
                 max channel diff {}; actual {}, diff {}; \
                 if the change is intended, re-bless with `{BLESS_CMD}`",
                m.mae,
                m.differing,
                differing_fraction(&m),
                m.max,
                actual.display(),
                diff.display()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "render regression:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn render_is_deterministic() {
    if !have_gpu() {
        eprintln!("{SKIP}");
        return;
    }
    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let a = render_frames(SEED, &FRAMES);
    let b = render_frames(SEED, &FRAMES);
    for ((i, pa), (_, pb)) in a.iter().zip(&b) {
        let m = compare(pa, pb);
        let bytes = pa.iter().zip(pb).filter(|(x, y)| x != y).count();
        eprintln!(
            "determinism: frame {i} differing bytes {bytes} max {} mae {:.4}",
            m.max, m.mae
        );
        assert_eq!(bytes, 0, "frame {i}: two identical renders differ ({m:?})");
    }
}

#[test]
fn one_pixel_over_pixel_levels_fails_the_frame() {
    let reference = vec![100u8; (WIDTH * HEIGHT * 4) as usize];
    let mut px = reference.clone();
    // One channel of one pixel off by exactly `PIXEL_LEVELS`: within the tolerance.
    px[4 * 1000 + 1] += PIXEL_LEVELS;
    let m = compare(&px, &reference);
    assert_eq!((m.differing, m.max), (0, PIXEL_LEVELS));
    assert!(!fails(&m), "{m:?}");
    // One level more, one pixel of 57,600: the frame fails although its MAE is ~0.00004.
    px[4 * 1000 + 1] += 1;
    let m = compare(&px, &reference);
    assert_eq!(m.differing, 1);
    assert!(m.mae < MAE_TOLERANCE && fails(&m), "{m:?}");
    // R, G and B of every pixel off by one level: no pixel over `PIXEL_LEVELS`, but MAE 0.75.
    let all: Vec<u8> = reference
        .iter()
        .enumerate()
        .map(|(k, &v)| if k % 4 == 3 { v } else { v + 1 })
        .collect();
    let m = compare(&all, &reference);
    assert_eq!((m.differing, m.max), (0, 1));
    assert!(fails(&m), "{m:?}");
}

/// Wrap one chunk the way png.rs does (length, type, data, CRC).
fn chunk(out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(ty);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

#[test]
fn png_reader_roundtrips_the_writer() {
    // 320×180 RGBA is several stored blocks; the pattern makes every byte value appear.
    let (w, h) = (WIDTH, HEIGHT);
    let px: Vec<u8> = (0..w * h * 4).map(|i| (i * 7 + i / 13) as u8).collect();
    let (dw, dh, back) = decode_png(&png::encode_rgba(w, h, &px)).unwrap();
    assert_eq!((dw, dh), (w, h));
    assert!(back == px, "decoded pixels differ");
}

#[test]
fn png_reader_refuses_what_the_writer_never_writes() {
    let good = png::encode_rgba(2, 1, &[255, 0, 0, 255, 0, 255, 0, 255]);
    // Layout: 8 signature, IHDR chunk at 8 (data 16..29), IDAT chunk at 33 (data from 41).
    let ihdr = &good[16..29];
    let idat_len = u32::from_be_bytes(good[33..37].try_into().unwrap()) as usize;
    let idat = &good[41..41 + idat_len];
    // Re-wrap modified chunks with valid CRCs, so each refusal is about the content.
    let png_of = |ihdr: &[u8], idat: &[u8]| {
        let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
        chunk(&mut out, b"IHDR", ihdr);
        chunk(&mut out, b"IDAT", idat);
        chunk(&mut out, b"IEND", &[]);
        out
    };
    assert_eq!(png_of(ihdr, idat), good);
    assert!(decode_png(&good).is_ok());

    // Row filter 1 (Sub), with the Adler-32 fixed up: refused.
    let mut filtered = idat.to_vec();
    filtered[7] = 1;
    let n = filtered.len();
    let sum = adler32(&filtered[7..n - 4]);
    filtered[n - 4..].copy_from_slice(&sum.to_be_bytes());
    let e = decode_png(&png_of(ihdr, &filtered)).unwrap_err();
    assert!(e.contains("row filter"), "{e}");

    // A fixed-Huffman (compressed) deflate block: refused.
    let mut fixed = idat.to_vec();
    fixed[2] = 0b011;
    let e = decode_png(&png_of(ihdr, &fixed)).unwrap_err();
    assert!(e.contains("compressed"), "{e}");

    // Colour type 2 (RGB): refused.
    let mut rgb = ihdr.to_vec();
    rgb[9] = 2;
    let e = decode_png(&png_of(&rgb, idat)).unwrap_err();
    assert!(e.contains("RGBA"), "{e}");

    // A second IHDR, and IHDR after IDAT: refused.
    let mut twice = b"\x89PNG\r\n\x1a\n".to_vec();
    chunk(&mut twice, b"IHDR", ihdr);
    chunk(&mut twice, b"IHDR", ihdr);
    chunk(&mut twice, b"IDAT", idat);
    chunk(&mut twice, b"IEND", &[]);
    let e = decode_png(&twice).unwrap_err();
    assert!(e.contains("unexpected chunk IHDR"), "{e}");

    // Bytes after IEND: refused.
    let mut trailing = good.clone();
    trailing.push(0);
    let e = decode_png(&trailing).unwrap_err();
    assert!(e.contains("unexpected chunk IEND"), "{e}");

    // Non-zero padding bits in a stored block's header byte: refused.
    let mut padded = idat.to_vec();
    padded[2] |= 0b1000;
    let e = decode_png(&png_of(ihdr, &padded)).unwrap_err();
    assert!(e.contains("padding"), "{e}");

    // Bytes after the zlib stream's Adler-32: refused.
    let mut longer = idat.to_vec();
    longer.push(0);
    let e = decode_png(&png_of(ihdr, &longer)).unwrap_err();
    assert!(e.contains("after the zlib stream"), "{e}");

    // One flipped bit in a chunk's CRC: refused.
    let mut bad = good.clone();
    bad[41 + idat_len] ^= 1;
    let e = decode_png(&bad).unwrap_err();
    assert!(e.contains("CRC"), "{e}");
}
