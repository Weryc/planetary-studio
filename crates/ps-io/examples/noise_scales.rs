//! How much noise each wavelet layer really holds, three ways: scaled from the finest plane as
//! if the noise were white (the old estimate), measured on the dark sky far from the subject
//! (pure noise, when the stacker didn't clip the sky), and the quietest blocks of the subject.
//!
//! `cargo run --release -p ps-io --example noise_scales -- a.tif b.tif …`
use ps_core::color::{ColorMode, working_planes};
use ps_core::preset::wavelet_structure;
use ps_core::wavelet::{ScaleMode, decompose};

fn median_abs(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mid = v.len() / 2;
    v.select_nth_unstable_by(mid, f32::total_cmp);
    v[mid] / 0.6745
}

fn main() {
    for path in std::env::args().skip(1) {
        let img = ps_io::load(path.as_ref()).unwrap();
        let a = ps_core::analysis::analyze(&img);
        let (w, h) = (img.width(), img.height());
        let plane = working_planes(&img, ColorMode::Luminance).swap_remove(0);
        let p = wavelet_structure(a.kind, &a, ScaleMode::Free);
        let d = decompose(&plane, w, h, &p, &|| false).unwrap();
        let n = w * h;
        let mut sorted: Vec<f32> = d.residual.iter().step_by(7).copied().collect();
        sorted.sort_by(f32::total_cmp);
        let (sky, bright) = (sorted[sorted.len() / 5], sorted[sorted.len() * 99 / 100]);
        let name: String = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().chars().take(28).collect();
        println!("\n{name}  {:?}  {w}×{h}  first {:.2}  sky level {sky:.4} bright {bright:.3}", a.kind, p.first_scale);
        // Coarse levels c_k = residual + layers from k on.
        let mut coarse = vec![d.residual.clone()];
        for l in d.layers.iter().rev() {
            let prev = coarse.last().unwrap();
            coarse.push(prev.iter().zip(l).map(|(c, v)| c + v).collect());
        }
        coarse.reverse(); // coarse[k] = c_k (k = 0 … J), c_0 = plane without grain
        let grain_sigma = d.grain.as_ref().map(|(g, s)| (median_abs(g.iter().step_by(3).map(|v| v.abs()).collect()), *s));
        println!("grain: all-pixels MAD {:?}", grain_sigma);
        let limit0 = sky + 0.02 * (bright - sky);
        let far0 = &coarse[1.min(d.layers.len())];
        let sky_grain = d.grain.as_ref().map(|(g, _)| median_abs((0..n).step_by(3).filter(|&i| far0[i] < limit0).map(|i| g[i].abs()).collect())).unwrap_or(f32::NAN);
        let white_grain = d.grain.as_ref().map(|g| g.1).unwrap_or(f32::NAN);
        println!("grain sky σ {sky_grain:.6}  (signal σ {white_grain:.6})");
        println!("layer scale   white σ    sky σ (n)          quiet-block σ   ratio sky/white  block/white  excess-vs-grain  model");
        for (j, layer) in d.layers.iter().enumerate() {
            let far = &coarse[(j + 2).min(d.layers.len())];
            let limit = sky + 0.02 * (bright - sky);
            let sky_vals: Vec<f32> = (0..n).step_by(3).filter(|&i| far[i] < limit).map(|i| layer[i].abs()).collect();
            let sky_sigma = median_abs(sky_vals.clone());
            // Quiet blocks on the subject.
            let b = ((4.0 * p.scale(j)).ceil() as usize).max(16);
            let thr = sky + 0.3 * (bright - sky);
            let mut rms = vec![];
            for by in (0..h - b).step_by(b) {
                for bx in (0..w - b).step_by(b) {
                    let mut ok = true;
                    let mut s = 0.0f64;
                    for y in by..by + b {
                        for x in bx..bx + b {
                            let i = y * w + x;
                            if coarse[0][i] < thr {
                                ok = false;
                            }
                            s += (layer[i] as f64).powi(2);
                        }
                    }
                    if ok {
                        rms.push((s / (b * b) as f64).sqrt() as f32);
                    }
                }
            }
            rms.sort_by(f32::total_cmp);
            let block = if rms.len() >= 8 { rms[rms.len() / 10] } else { f32::NAN };
            println!(
                "  {}   {:5.2}  {:9.6}  {:9.6} ({:7})  {:9.6} ({:4})   {:6.2}          {:6.2}   {:6.2}   {:5.2}",
                j + 1,
                p.scale(j),
                d.white_noise[j],
                sky_sigma,
                sky_vals.len(),
                block,
                rms.len(),
                sky_sigma / d.white_noise[j],
                block / d.white_noise[j],
                (sky_sigma / sky_grain) / (d.white_noise[j] / white_grain),
                d.noise[j] / d.white_noise[j]
            );
        }
    }
}
