//! Learned noise reduction, step 3: look at it on real stacks.
//!
//! `cargo run --release -p ps-train --bin denoise_compare -- weights.bin out_dir image…`
//!
//! For every image, a 192² crop (shown at 2×) of the luminance around the subject, side by side: original,
//! learned, starlet thresholds (Pianeti), bilateral (Luna e Sole), then the learned method's
//! removed part (×8 around grey), all with the same levels and a mild wavelet sharpening
//! (×4 and ×3 on the two finest scales) so the noise left is easy to see, as after processing.
//! Prints the noise measured and the time the network took on the whole image.
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::denoise_net::{Net, denoise};
use ps_core::enhance::{NoiseMethod, NoiseParams};
use ps_core::wavelet::{ScaleMode, WaveletLayer, WaveletParams, estimate_noise};
use std::path::Path;

const SIDE: usize = 192;

fn sharpen(p: &[f32], w: usize, h: usize) -> Vec<f32> {
    let params = WaveletParams { layers: vec![WaveletLayer { gain: 3.0, ..Default::default() }, WaveletLayer { gain: 2.0, ..Default::default() }], scale_mode: ScaleMode::Dyadic, ..Default::default() };
    let d = ps_core::wavelet::decompose(p, w, h, &params, &|| false).unwrap();
    ps_core::wavelet::recompose(&d, &params)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let net = Net::from_bytes(&std::fs::read(args.next().expect("weights")).unwrap()).expect("pesi non validi");
    let out = args.next().expect("out dir");
    std::fs::create_dir_all(&out).unwrap();
    for path in args {
        let path = Path::new(&path);
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let Some(img) = ml::load_any(path) else { continue };
        let (w, h) = (img.width(), img.height());
        let y = ml::lum(&img);
        let sigma = estimate_noise(&y, w, h);
        let t = std::time::Instant::now();
        let learned = denoise(&net, &y, w, h, sigma, 1.0, &|| false).unwrap();
        let took = t.elapsed().as_secs_f32();
        let mono = Image::new(w, h, vec![y.clone()], 16);
        let classic = |method| ps_core::enhance::denoise(&mono, &NoiseParams { method, strength: 1.0, detail: 0.3, chroma: 0, sky: 0.0 }).channels()[0].clone();
        let (planet, lunar) = (classic(NoiseMethod::Planetary), classic(NoiseMethod::Lunar));
        let a = ps_core::analysis::analyze(&img);
        let (cx, cy) = ml::centre(&img, &a);
        let side = SIDE.min(w).min(h);
        let x0 = (cx as usize).saturating_sub(side / 2).min(w - side);
        let y0 = (cy as usize).saturating_sub(side / 2).min(h - side);
        let crop = |p: &[f32]| sharpen(&ml::crop(p, w, (x0, y0, side)), side, side);
        let panels = [crop(&y), crop(&learned), crop(&planet), crop(&lunar)];
        let mut sorted: Vec<f32> = panels[0].clone();
        sorted.sort_by(f32::total_cmp);
        let (lo, hi) = (sorted[side * side / 200], sorted[side * side * 998 / 1000].max(sorted[side * side / 200] + 1e-4));
        let removed: Vec<f32> = ml::crop(&y, w, (x0, y0, side)).iter().zip(ml::crop(&learned, w, (x0, y0, side))).map(|(a, b)| 0.5 + 8.0 * (a - b) / (hi - lo)).collect();
        let n = panels.len() + 1;
        let mut montage = vec![0.0f32; n * side * side];
        for (k, p) in panels.iter().map(|p| p.iter().map(|v| (v - lo) / (hi - lo)).collect::<Vec<f32>>()).chain([removed]).enumerate() {
            for yy in 0..side {
                for xx in 0..side {
                    montage[yy * n * side + k * side + xx] = p[yy * side + xx].clamp(0.0, 1.0);
                }
            }
        }
        // Twice the size, so the grain is visible.
        let big: Vec<f32> = (0..4 * n * side * side).map(|i| montage[(i / (2 * n * side)) / 2 * n * side + (i % (2 * n * side)) / 2]).collect();
        ps_io::save(Path::new(&format!("{out}/{stem}.png")), &Image::new(2 * n * side, 2 * side, vec![big], 16)).unwrap();
        println!("{stem:<60} {w}×{h}  σ {:.5}  rete {took:.2} s", sigma);
    }
}
