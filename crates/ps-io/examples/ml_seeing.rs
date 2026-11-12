//! Look at the simulated lucky-imaging seeing: per night, the stacked PSF's core and halo,
//! and its transfer (MTF) by frequency over the diffraction cut-off.
//!
//! `cargo run --release -p ps-io --example ml_seeing`
#[path = "ml_common/mod.rs"]
mod ml;

use ml::seeing::*;

fn main() {
    for night in NIGHTS {
        let t = std::time::Instant::now();
        let psfs = stacked(&night, &LAMBDA_RGB, &[0.3 * night.dispersion, 0.55 * night.dispersion, night.dispersion], 1);
        let (fwhm, core) = describe(&psfs[1]);
        // MTF of the green kernel at k = 0.2 λ/D per px (cut-off at 0.2 c/px).
        let k = 0.2;
        let kern = to_pixels(&psfs[1], 1.0, k);
        let s = kern.size();
        let mtf = |f: f32| -> f32 {
            // Radially averaged |OTF| at f c/px.
            let mut acc = 0.0f64;
            for a in 0..16 {
                let th = a as f32 * std::f32::consts::PI / 16.0;
                let (u, v) = (f * th.cos(), f * th.sin());
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for y in 0..s {
                    for x in 0..s {
                        let ph = std::f32::consts::TAU * (u * (x as f32 - kern.radius as f32) + v * (y as f32 - kern.radius as f32));
                        re += (kern.data[y * s + x] * ph.cos()) as f64;
                        im += (kern.data[y * s + x] * ph.sin()) as f64;
                    }
                }
                acc += re.hypot(im);
            }
            (acc / 16.0) as f32
        };
        let line: Vec<String> = [0.05f32, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0].iter().map(|r| format!("{r}:{:.3}", mtf(r * k))).collect();
        eprintln!("{:<11} FWHM {fwhm:.2} λ/D, core {:.0} %, kernel r {} px, {:.1} s | MTF {}", night.label, core * 100.0, kern.radius, t.elapsed().as_secs_f32(), line.join(" "));
    }
}
