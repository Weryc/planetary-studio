//! For the noise-reduction comparison: each simulated stack's measured square (as
//! `ps_core::curve_net::frame` takes it), with its truth moved exactly onto it, the app's own
//! noise reductions of it, and the subject's pixels, as `.npy` files a Python script compares
//! standard algorithms on.
//!
//! `cargo run --release -p ps-train --bin noise_bench_export -- out_dir @runs.txt`
//! → out_dir/<run>/{grezza,verita,app_pianeti,app_appresa,dentro}.npy
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::align::measure_shift;
use ps_core::analysis::analyze;
use ps_core::enhance::{NoiseMethod, NoiseParams, denoise};
use ps_core::fft::FftConvolver;
use rayon::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};

fn fourier_shift(plane: &[f32], w: usize, h: usize, d: [f32; 2]) -> Vec<f32> {
    let plan = FftConvolver::plan(w, h);
    let mut spec = plan.forward(plane);
    for (idx, c) in spec.iter_mut().enumerate() {
        let (kx, ky) = (idx / h, idx % h);
        let fy = if ky <= h / 2 { ky as f32 } else { ky as f32 - h as f32 } / h as f32;
        let phase = std::f32::consts::TAU * (kx as f32 / w as f32 * d[0] + fy * d[1]);
        let (sn, cs) = phase.sin_cos();
        let (re, im) = (c.re, c.im);
        c.re = re * cs - im * sn;
        c.im = re * sn + im * cs;
    }
    plan.inverse(spec)
}

/// A little-endian float32 array as NumPy's `.npy` (version 1.0).
fn npy(path: &Path, shape: &[usize], data: &[f32]) {
    let dims = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}{}), }}", if shape.len() == 1 { "," } else { "" });
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    for v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn save(dir: &Path, name: &str, img: &Image) {
    let data: Vec<f32> = img.channels().iter().flatten().copied().collect();
    npy(&dir.join(format!("{name}.npy")), &[img.channels().len(), img.height(), img.width()], &data);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(&args[0]);
    // Real stacks (no truth): `--reali image…` exports the raw, the app's reductions and the
    // subject's pixels, plus the noise network's inputs.
    if args[1] == "--reali" {
        for path in args[2..].iter().map(PathBuf::from) {
            let raw = ps_io::load(&path).unwrap();
            let a = analyze(&raw);
            let Some((m, f)) = ps_core::noise_net::measure(&raw, &a) else { continue };
            let dir = out.join(path.file_stem().unwrap());
            std::fs::create_dir_all(&dir).unwrap();
            let crop = |img: &Image| img.crop(f.x0, f.y0, f.side, f.side);
            save(&dir, "grezza", &f.crop);
            for (file, method) in [("app_pianeti", NoiseMethod::Planetary), ("app_appresa", NoiseMethod::Learned), ("app_rete", NoiseMethod::Guided)] {
                let chroma = if method == NoiseMethod::Guided { 0 } else { 2 };
                let d = denoise(&raw, &NoiseParams { method, strength: 1.0, detail: 0.3, chroma, sky: 0.8 });
                save(&dir, file, &crop(&d));
            }
            for k in [2.0f32, 3.0] {
                let d = denoise(&raw, &NoiseParams { method: NoiseMethod::Guided, strength: k, detail: 0.3, chroma: 0, sky: 0.8 });
                save(&dir, &format!("app_rete{k:.0}"), &crop(&d));
            }
            let mut mask = vec![0.0f32; f.side * f.side];
            f.inside.iter().for_each(|&i| mask[i] = 1.0);
            npy(&dir.join("dentro.npy"), &[f.side, f.side], &mask);
            let x = ps_core::noise_net::features(&m);
            npy(&dir.join("ingressi.npy"), &[x.len()], &x);
            eprintln!("{}: esportata", path.display());
        }
        return;
    }
    let runs: Vec<PathBuf> = std::fs::read_to_string(args[1].trim_start_matches('@')).unwrap().lines().filter(|l| !l.trim().is_empty()).map(PathBuf::from).collect();
    let done: usize = runs
        .par_iter()
        .map(|run| {
            let name = run.file_name().unwrap().to_string_lossy().to_string();
            let raw = ml::load_any(&run.join("grezza.tif")).unwrap();
            let truth = ml::load_any(&run.join("verita_campionata.tif")).unwrap();
            let (w, h) = (raw.width(), raw.height());
            let s = measure_shift(&ml::lum(&truth), &ml::lum(&raw), w, h);
            let truth = Image::new(w, h, truth.channels().iter().map(|c| fourier_shift(c, w, h, s)).collect(), 16);
            let a = analyze(&raw);
            let Some(f) = ps_core::curve_net::frame(&raw, &a) else { return 0 };
            let dir = out.join(&name);
            std::fs::create_dir_all(&dir).unwrap();
            let crop = |img: &Image| img.crop(f.x0, f.y0, f.side, f.side);
            save(&dir, "grezza", &f.crop);
            save(&dir, "verita", &crop(&truth));
            for (file, method) in [("app_pianeti", NoiseMethod::Planetary), ("app_appresa", NoiseMethod::Learned)] {
                let d = denoise(&raw, &NoiseParams { method, strength: 1.0, detail: 0.3, chroma: 2, sky: 0.8 });
                save(&dir, file, &crop(&d));
            }
            let mut mask = vec![0.0f32; f.side * f.side];
            f.inside.iter().for_each(|&i| mask[i] = 1.0);
            npy(&dir.join("dentro.npy"), &[f.side, f.side], &mask);
            1
        })
        .sum();
    eprintln!("{done} stack esportati in {}", out.display());
}
