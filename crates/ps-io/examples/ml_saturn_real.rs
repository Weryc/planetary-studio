//! Saturn's globe on real stacks: how much fine grain the automatic learned sharpening adds on
//! the globe, against BlurXTerminator, with and without a light cleaning of the globe's grain.
//! The grain is the fine part (above a 1.5 px blur) of what the sharpening added, RMS on the
//! globe. Writes strips around the globe: stack | automatic | automatic + cleaning (each
//! amount) | BXT.
//!
//! `cargo run --release -p ps-io --example ml_saturn_real -- out_dir bxt_cache image… [clean=0.3,0.6]`
#[path = "ml_common/mod.rs"]
mod ml;

use ml::*;
use ps_core::Image;
use ps_core::mask::{AutoRegion, MaskShape, MaskSpec};
use std::path::Path;

/// The globe's grain blended toward a lightly blurred copy, by `amount` inside the globe.
fn clean_globe(img: &Image, globe: &[f32], amount: f32) -> Image {
    let smooth = blur(img, 0.8);
    let ch = img.channels().iter().zip(smooth.channels()).map(|(a, b)| a.iter().zip(b).zip(globe).map(|((x, y), m)| x + amount * m * (y - x)).collect()).collect();
    Image::new(img.width(), img.height(), ch, 16)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("out dir");
    let cache = args.next().expect("bxt cache");
    std::fs::create_dir_all(&out).unwrap();
    let (paths, opts): (Vec<String>, Vec<String>) = args.partition(|a| !a.starts_with("clean="));
    let amounts: Vec<f32> = opts.first().map_or(vec![0.3, 0.6], |o| o[6..].split(',').filter_map(|v| v.parse().ok()).collect());
    for path in paths {
        let Ok(src) = source(Path::new(&path)) else { continue };
        let img = &src.img;
        let (w, h) = (img.width(), img.height());
        let a = &src.analysis;
        let p = ps_core::preset::auto_smart(a.kind, a, ps_core::preset::AutoStrength::Normal);
        let subject = ps_core::mask::subject_and_moons(img);
        let auto = ps_core::smart::sharpen(img, None, &p, Some(&subject));
        let globe_hard = ps_core::mask::render(&MaskSpec::new(MaskShape::Auto(AutoRegion::SaturnGlobe), 0.0), img);
        let globe = ps_core::mask::gaussian_approx(&globe_hard, w, h, 2.0);
        let inside: Vec<usize> = (0..w * h).filter(|&i| globe_hard[i] > 0.5).collect();
        if inside.is_empty() {
            eprintln!("{}: globo non trovato", src.stem);
            continue;
        }
        let l0 = lum(img);
        // Fine grain added on the globe.
        let grain = |o: &Image| -> f32 {
            let d: Vec<f32> = lum(o).iter().zip(&l0).map(|(a, b)| a - b).collect();
            let s = lum(&blur(&Image::new(w, h, vec![d.clone()], 16), 1.5));
            (inside.iter().map(|&i| (d[i] - s[i]).powi(2)).sum::<f32>() / inside.len() as f32).sqrt()
        };
        let bxt = ps_io::load(Path::new(&format!("{cache}/{}_bxt.tif", src.stem.replace(' ', "_")))).ok().filter(|b| b.width() == w && b.height() == h);
        let reference = bxt.as_ref().map(grain);
        let rel = |g: f32| reference.map_or(format!("{g:.5}"), |r| format!("{:.2}", g / r.max(1e-9)));
        let mut panels = vec![img.clone(), auto.clone()];
        let mut line = format!("{:<48} auto {}", src.stem.chars().take(48).collect::<String>(), rel(grain(&auto)));
        for &k in &amounts {
            let c = clean_globe(&auto, &globe, k);
            line += &format!("  pulizia {k}: {}", rel(grain(&c)));
            panels.push(c);
        }
        println!("{line}{}", if reference.is_some() { "  (BXT = 1)" } else { "  (senza BXT)" });
        if let Some(b) = bxt {
            panels.push(b);
        }
        // Strip around the globe, 2× enlarged when small.
        let e = ps_core::mask::saturn_globe(img).unwrap();
        let side = ((e.rx * 2.4) as usize).min(w).min(h);
        let x0 = (e.cx as usize).saturating_sub(side / 2).min(w - side);
        let y0 = (e.cy as usize).saturating_sub(side / 2).min(h - side);
        let zoom = (360 / side.max(1)).clamp(1, 3);
        let t = side * zoom;
        let ow = panels.len() * (t + 6) - 6;
        let mut sorted = crop(&l0, w, (x0, y0, side));
        sorted.sort_by(f32::total_cmp);
        let (lo, hi) = (sorted[sorted.len() / 200], sorted[sorted.len() * 998 / 1000].max(1e-4));
        let mut planes = vec![vec![0.08f32; ow * t]; 3];
        for (k, im) in panels.iter().enumerate() {
            for (c, plane) in planes.iter_mut().enumerate() {
                let s = &im.channels()[c.min(im.channels().len() - 1)];
                for y in 0..t {
                    for x in 0..t {
                        plane[y * ow + k * (t + 6) + x] = ((s[(y0 + y / zoom) * w + x0 + x / zoom] - lo) / (hi - lo)).clamp(0.0, 1.0);
                    }
                }
            }
        }
        ps_io::save(&Path::new(&out).join(format!("{}_globo.png", src.stem.chars().take(24).collect::<String>())), &Image::new(ow, t, planes, 16)).unwrap();
    }
}
