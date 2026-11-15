//! Learned noise reduction on raw stacks, with the smart sharpening at 100 %.
//!
//! `cargo run --release -p ps-train --bin denoise_real -- out_dir image…`
//!
//! For every stack, a strip around the subject (luminance, same levels): raw | sharpening only |
//! denoise ×1 → sharpening (the app's order) | denoise ×2 → sharpening | sharpening → denoise ×1 |
//! denoise at half resolution → sharpening. All on the luminance.
//! Prints the noise the app measures (σ), the same measured on the 2×2-binned image (×2: equal
//! for white noise, larger when the grain is coarser than a pixel), and the fine grain left in
//! the subject relative to sharpening only.
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::enhance::{NoiseMethod, NoiseParams};
use ps_core::wavelet::estimate_noise;
use std::path::Path;

fn denoise(img: &Image, strength: f32) -> Image {
    ps_core::enhance::denoise(img, &NoiseParams { method: NoiseMethod::Learned, strength, detail: 0.1, chroma: 0, sky: 0.0 })
}

/// σ of the 2×2-binned plane, ×2 (so white noise gives the same as on the full plane).
fn binned_sigma(y: &[f32], w: usize, h: usize) -> f32 {
    let (bw, bh) = (w / 2, h / 2);
    let b: Vec<f32> = (0..bw * bh).map(|i| (i % bw * 2, i / bw * 2)).map(|(x, yy)| (y[yy * w + x] + y[yy * w + x + 1] + y[(yy + 1) * w + x] + y[(yy + 1) * w + x + 1]) / 4.0).collect();
    2.0 * estimate_noise(&b, bw, bh)
}

/// The learned denoise at half resolution: the 2×2-binned luminance cleaned, the noise found
/// there brought back up (bilinear) and taken from the luminance.
fn denoise_half(img: &Image, strength: f32) -> Image {
    let (w, h) = (img.width(), img.height());
    let y = ml::lum(img);
    let (bw, bh) = (w / 2, h / 2);
    let b: Vec<f32> = (0..bw * bh).map(|i| (i % bw * 2, i / bw * 2)).map(|(x, yy)| (y[yy * w + x] + y[yy * w + x + 1] + y[(yy + 1) * w + x] + y[(yy + 1) * w + x + 1]) / 4.0).collect();
    let net = ps_core::denoise_net::model().expect("pesi");
    let clean = ps_core::denoise_net::denoise(net, &b, bw, bh, estimate_noise(&b, bw, bh), strength, &|| false).unwrap();
    let noise: Vec<f32> = b.iter().zip(&clean).map(|(a, c)| a - c).collect();
    let at = |x: f32, yy: f32| {
        let (x, yy) = (x.clamp(0.0, (bw - 1) as f32), yy.clamp(0.0, (bh - 1) as f32));
        let (x0, y0) = (x as usize, yy as usize);
        let (x1, y1) = ((x0 + 1).min(bw - 1), (y0 + 1).min(bh - 1));
        let (fx, fy) = (x - x0 as f32, yy - y0 as f32);
        let r0 = noise[y0 * bw + x0] * (1.0 - fx) + noise[y0 * bw + x1] * fx;
        let r1 = noise[y1 * bw + x0] * (1.0 - fx) + noise[y1 * bw + x1] * fx;
        r0 * (1.0 - fy) + r1 * fy
    };
    let out: Vec<f32> = (0..w * h).map(|i| y[i] - 0.9 * at((i % w) as f32 / 2.0 - 0.25, (i / w) as f32 / 2.0 - 0.25)).collect();
    Image::new(w, h, vec![out], 16)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("out dir");
    std::fs::create_dir_all(&out).unwrap();
    println!("{:<44} {:>8} {:>8} {:>6}   grana fine nel soggetto (solo nitidezza = 1)", "immagine", "σ", "σ 2×2", "rapp.");
    for path in args {
        let Ok(src) = ml::source(Path::new(&path)) else { continue };
        let mono = Image::new(src.img.width(), src.img.height(), vec![ml::lum(&src.img)], 16);
        let img = &mono;
        let (w, h) = (img.width(), img.height());
        let a = &src.analysis;
        let mut sp = ps_core::preset::auto_smart(a.kind, a, ps_core::preset::AutoStrength::Normal);
        sp.strength = 1.0;
        let subject = ps_core::mask::subject_and_moons(img);
        let sharpen = |i: &Image| ps_core::smart::sharpen(i, None, &sp, Some(&subject));
        let only = sharpen(img);
        let variants = [img.clone(), only.clone(), sharpen(&denoise(img, 1.0)), sharpen(&denoise(img, 2.0)), denoise(&only, 1.0), sharpen(&denoise_half(img, 1.0))];
        let y0 = ml::lum(img);
        let (s1, s2) = (estimate_noise(&y0, w, h), binned_sigma(&y0, w, h));
        // Fine grain: what is above a 1.5 px blur, RMS inside the subject.
        let inside: Vec<usize> = (0..w * h).filter(|&i| subject[i] > 0.5).collect();
        let grain = |o: &Image| -> f32 {
            let l = ml::lum(o);
            let s = ml::lum(&ml::blur(&Image::new(w, h, vec![l.clone()], 16), 1.5));
            (inside.iter().map(|&i| (l[i] - s[i]).powi(2)).sum::<f32>() / inside.len().max(1) as f32).sqrt()
        };
        let g_only = grain(&only);
        let rel: Vec<String> = variants[2..].iter().map(|v| format!("{:.2}", grain(v) / g_only.max(1e-9))).collect();
        println!("{:<44} {s1:>8.5} {s2:>8.5} {:>6.2}   rumore×1→nit {}  rumore×2→nit {}  nit→rumore×1 {}  rumore½→nit {}", src.stem.chars().take(44).collect::<String>(), s2 / s1, rel[0], rel[1], rel[2], rel[3]);

        // Strip: square around the subject, 2× when small.
        let (cx, cy) = ml::centre(img, a);
        let side = 240.min(w).min(h);
        let x0 = (cx as usize).saturating_sub(side / 2).min(w - side);
        let yy0 = (cy as usize).saturating_sub(side / 2).min(h - side);
        let crops: Vec<Vec<f32>> = variants.iter().map(|v| ml::crop(&ml::lum(v), w, (x0, yy0, side))).collect();
        let mut sorted = crops[1].clone();
        sorted.sort_by(f32::total_cmp);
        let (lo, hi) = (sorted[sorted.len() / 200], sorted[sorted.len() * 998 / 1000].max(sorted[sorted.len() / 200] + 1e-4));
        let z = 2;
        let t = side * z;
        let gap = 6;
        let ow = crops.len() * (t + gap) - gap;
        let mut plane = vec![0.1f32; ow * t];
        for (k, c) in crops.iter().enumerate() {
            for yy in 0..t {
                for xx in 0..t {
                    plane[yy * ow + k * (t + gap) + xx] = ((c[(yy / z) * side + xx / z] - lo) / (hi - lo)).clamp(0.0, 1.0);
                }
            }
        }
        ps_io::save(Path::new(&format!("{out}/{}.png", src.stem.replace(' ', "_"))), &Image::new(ow, t, vec![plane], 16)).unwrap();
    }
}
