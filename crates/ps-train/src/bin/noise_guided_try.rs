//! The app's noise reduction guided by the noise network on real stacks: how much it removes
//! (RMS of the change on the disk, over its mean level) and a view of before / removed / after.
//!
//! `cargo run --release -p ps-train --bin noise_guided_try -- out_dir image…`
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::enhance::{NoiseMethod, NoiseParams, denoise};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&out).unwrap();
    for path in args[1..].iter().map(PathBuf::from) {
        let img = ps_io::load(&path).unwrap();
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let started = std::time::Instant::now();
        let p = NoiseParams { method: NoiseMethod::Guided, strength: 1.0, detail: 0.0, chroma: 0, sky: 0.0 };
        let cleaned = denoise(&img, &p);
        let ms = started.elapsed().as_millis();
        let a = ps_core::analysis::analyze(&img);
        let guided = ps_core::noise_net::guided_denoise(&img, 1.0).is_some();
        let (l0, l1) = (ml::lum(&img), ml::lum(&cleaned));
        let disk: Vec<usize> = match a.disk {
            Some(d) => (0..l0.len()).filter(|&i| ((i % img.width()) as f32 - d.cx).hypot((i / img.width()) as f32 - d.cy) < 0.8 * d.radius).collect(),
            None => (0..l0.len()).collect(),
        };
        let mean = disk.iter().map(|&i| l0[i] as f64).sum::<f64>() / disk.len() as f64;
        let rms = (disk.iter().map(|&i| ((l0[i] - l1[i]) as f64).powi(2)).sum::<f64>() / disk.len() as f64).sqrt() / mean;
        println!("{stem}: {} in {ms} ms, tolto {:.3} % del livello del disco", if guided { "rete del rumore" } else { "ripiego sul metodo Appreso" }, 100.0 * rms);
        // Before | removed (×20 on grey) | after, the disk's middle.
        let c = ml::centre(&img, &a);
        let side = 160usize.min(img.width()).min(img.height());
        let (b, af) = (ml::crop_around(&img, c, side), ml::crop_around(&cleaned, c, side));
        let mut top: Vec<f32> = ml::lum(&b);
        top.sort_by(f32::total_cmp);
        let t = top[top.len() * 995 / 1000].max(1e-6);
        let planes = (0..b.channels().len())
            .map(|k| {
                let (p0, p1) = (&b.channels()[k], &af.channels()[k]);
                (0..side * 3 * side)
                    .map(|i| {
                        let (x, y) = (i % (3 * side), i / (3 * side));
                        let j = y * side + x % side;
                        match x / side {
                            0 => p0[j] / t,
                            1 => 0.5 + 20.0 * (p0[j] - p1[j]) / t,
                            _ => p1[j] / t,
                        }
                        .clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();
        ps_io::save(&out.join(format!("{stem}.png")), &Image::new(3 * side, side, planes, 16)).unwrap();
    }
}
