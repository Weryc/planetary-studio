//! The inputs of the sharpening network (`ps_core::curve_net::features`) for simulated stacks,
//! exactly as the app computes them, twice: on the stack as it is, and with its sky clipped to
//! black as some programs leave it (a black point set just above the sky's noise), so the
//! network learns to do without the sky and the clipped sky is checked to be noticed.
//!
//! `cargo run --release -p ps-train --bin curve_features -- out.json run…`
//! (also accepts a file of run folders, one per line, with `@list.txt`).
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::analysis::analyze;
use ps_core::curve_net::{features, measure};
use rayon::prelude::*;
use std::path::PathBuf;

/// The image with everything below the sky's 99th percentile set to black (each channel
/// lowered by that level and clipped at 0).
fn black_sky(img: &Image) -> Image {
    let a = analyze(img);
    let lum = ml::lum(img);
    let (w, h) = (img.width(), img.height());
    let (cx, cy, r) = a.disk.map_or((w as f32 / 2.0, h as f32 / 2.0, 0.3 * w as f32), |d| (d.cx, d.cy, d.radius));
    let mut sky: Vec<f32> = (0..w * h).filter(|&i| ((i % w) as f32 - cx).hypot((i / w) as f32 - cy) > 1.3 * r).map(|i| lum[i]).collect();
    sky.sort_by(f32::total_cmp);
    let level = sky.get(sky.len() * 99 / 100).copied().unwrap_or(0.0);
    Image::new(w, h, img.channels().iter().map(|p| p.iter().map(|v| (v - level).max(0.0)).collect()).collect(), 16)
}

fn arr(v: &[f32]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{x:.5}")).collect::<Vec<_>>().join(","))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(&args[0]);
    let mut runs: Vec<PathBuf> = Vec::new();
    for a in &args[1..] {
        if let Some(list) = a.strip_prefix('@') {
            runs.extend(std::fs::read_to_string(list).unwrap().lines().filter(|l| !l.trim().is_empty()).map(PathBuf::from));
        } else {
            runs.push(PathBuf::from(a));
        }
    }
    let rows: Vec<String> = runs
        .par_iter()
        .map(|run| {
            let name = run.file_name().unwrap().to_string_lossy().to_string();
            let raw = ml::load_any(&run.join("grezza.tif")).unwrap();
            let one = |img: &Image| {
                let a = analyze(img);
                let planet = a.planet.map_or("null".to_string(), |p| format!("\"{p:?}\""));
                match measure(img, &a) {
                    Some(m) => format!("{{\"x\": {}, \"cielo\": {}, \"nota\": {}, \"pianeta\": {planet}}}", arr(&features(&m)), m.sky.is_some(), m.sky_note.map_or("null".to_string(), |n| format!("\"{n}\""))),
                    None => format!("{{\"x\": null, \"pianeta\": {planet}}}"),
                }
            };
            format!("  {{\"nome\": \"{name}\", \"normale\": {}, \"nero\": {}}}", one(&raw), one(&black_sky(&raw)))
        })
        .collect();
    std::fs::write(&out, format!("[\n{}\n]\n", rows.join(",\n"))).unwrap();
    eprintln!("{} grezze → {}", rows.len(), out.display());
}
