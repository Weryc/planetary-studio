//! The learned sharpening on real stacks near the limb and the poles: the stack, the automatic
//! learned sharpening and BlurXTerminator when cached, around the top of the disk. Also the
//! fine-scale energy added in the outer ring of the disk over the centre's (grain where there is
//! little detail).
//!
//! `cargo run --release -p ps-io --example ml_zone_real -- out_dir bxt_cache image… [Kind:width:protection…]`
#[path = "ml_common/mod.rs"]
mod ml;

use ml::*;
use ps_core::Image;
use std::path::Path;

fn sharpen(img: &Image, limb: &std::collections::HashMap<String, (f32, f32)>) -> Image {
    let a = ps_core::analysis::analyze(img);
    let mut p = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
    if let Some((w, pr)) = limb.get(&format!("{:?}", a.kind)) {
        (p.limb_width, p.limb_protection) = (*w, *pr);
    }
    let subject = ps_core::mask::subject_and_moons(img);
    ps_core::smart::sharpen(img, None, &p, Some(&subject))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("out dir");
    let cache = args.next().expect("bxt cache");
    std::fs::create_dir_all(&out).unwrap();
    // Paths, and `Kind:width:protection` overrides of the limb protection.
    let (paths, limb): (Vec<String>, Vec<String>) = args.partition(|a| a.contains('/'));
    let limb = limb_overrides(limb.into_iter());
    for path in paths {
        let Ok(src) = source(Path::new(&path)) else { continue };
        let Some(e) = outline(&src.img, &src.analysis) else { continue };
        let model = sharpen(&src.img, &limb);
        let bxt = ps_io::load(Path::new(&format!("{cache}/{}_bxt.tif", src.stem.replace(' ', "_")))).ok();
        let w = src.img.width();
        let l0 = lum(&src.img);
        // Fine-scale energy added (difference to the stack, high-passed by a 1.5 px blur), outer ring over centre.
        let added = |img: &Image| -> f32 {
            let d: Vec<f32> = lum(img).iter().zip(&l0).map(|(a, b)| a - b).collect();
            let dimg = Image::new(w, src.img.height(), vec![d.clone()], 16);
            let smooth = lum(&blur(&dimg, 1.5));
            let hp: Vec<f32> = d.iter().zip(&smooth).map(|(a, b)| a - b).collect();
            let energy = |r0: f32, r1: f32| { let v: Vec<f32> = (0..hp.len()).filter(|&i| { let r = e.radius((i % w) as f32, (i / w) as f32); r >= r0 && r < r1 }).map(|i| hp[i] * hp[i]).collect(); (v.iter().sum::<f32>() / v.len().max(1) as f32).sqrt() };
            energy(0.6, 0.92) / energy(0.0, 0.5).max(1e-9)
        };
        println!("{:<60} modello {:.2} bxt {}", src.stem.chars().take(60).collect::<String>(), added(&model), bxt.as_ref().map_or("-".into(), |b| format!("{:.2}", added(b))));
        // Strip around the top of the disk.
        let mut panels = vec![&src.img, &model];
        if let Some(b) = &bxt {
            panels.push(b);
        }
        let side = (e.rx.max(e.ry) * 0.9) as usize;
        let (cx, cy) = (e.cx as usize, (e.cy - e.ry * 0.55).max(side as f32 / 2.0) as usize);
        let x0 = cx.saturating_sub(side / 2).min(w - side);
        let y0 = cy.saturating_sub(side / 2).min(src.img.height() - side);
        let zoom = (300 / side.max(1)).clamp(1, 4);
        let t = side * zoom;
        let ow = panels.len() * (t + 6) - 6;
        let mut planes = vec![vec![0.08f32; ow * t]; 3];
        let mut sorted: Vec<f32> = crop(&l0, w, (x0, y0, side));
        sorted.sort_by(f32::total_cmp);
        let (lo, hi) = (sorted[sorted.len() / 200], sorted[sorted.len() * 998 / 1000].max(1e-4));
        for (k, img) in panels.iter().enumerate() {
            for (c, plane) in planes.iter_mut().enumerate() {
                let p = &img.channels()[c.min(img.channels().len() - 1)];
                for y in 0..t {
                    for x in 0..t {
                        plane[y * ow + k * (t + 6) + x] = ((p[(y0 + y / zoom) * w + x0 + x / zoom] - lo) / (hi - lo)).clamp(0.0, 1.0);
                    }
                }
            }
        }
        ps_io::save(&Path::new(&out).join(format!("{}_poli.png", src.stem.replace(' ', "_"))), &Image::new(ow, t, planes, 16)).unwrap();
    }
}
