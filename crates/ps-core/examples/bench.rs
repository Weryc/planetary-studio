//! Rough timings: `cargo run --release -p ps-core --example bench [width height]`

use ps_core::Image;
use ps_core::analysis::analyze;
use ps_core::deconv::{DeconvParams, deconvolve_plane};
use ps_core::pipeline::{Aux, Processor, next_generation};
use ps_core::preset::auto_pipeline;
use ps_core::wavelet::{WaveletParams, decompose, recompose};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (w, h) = (
        *args.first().unwrap_or(&4000),
        *args.get(1).unwrap_or(&3000),
    );
    let plane: Vec<f32> = (0..w * h)
        .map(|i| (((i % w) as f32 * 0.01).sin() * ((i / w) as f32 * 0.013).cos() + 1.0) * 0.4)
        .collect();
    println!(
        "{w}×{h} ({:.1} MP), {} threads",
        (w * h) as f64 / 1e6,
        rayon::current_num_threads()
    );

    let mut params = WaveletParams::default();
    params.layers[0].gain = 2.0;
    params.layers[1].denoise = 1.0;
    let t = Instant::now();
    let d = decompose(&plane, w, h, &params, &|| false).unwrap();
    println!(
        "wavelet decompose (6 layers): {:>7.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    let t = Instant::now();
    let _ = recompose(&d, &params);
    println!(
        "wavelet recompose:            {:>7.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    let gpu = ps_core::gpu::get().map(|g| g.name.clone());
    println!("GPU: {}", gpu.as_deref().unwrap_or("nessuna"));
    for on_gpu in [false, true] {
        ps_core::gpu::set_enabled(on_gpu);
        if on_gpu && gpu.is_none() {
            continue;
        }
        for iterations in [1, 20] {
            let p = DeconvParams {
                iterations,
                ..Default::default()
            };
            let t = Instant::now();
            let _ = deconvolve_plane(&plane, w, h, &p, &|| false, &|_| {});
            println!(
                "RL {iterations:>2} it ({}):                {:>7.1} ms",
                if on_gpu { "GPU" } else { "CPU" },
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    // Whole automatic pipeline on an RGB image, stage by stage, plus tone and colour.
    let rgb = Image::new(
        w,
        h,
        vec![
            plane.clone(),
            plane.iter().map(|v| v * 0.9).collect(),
            plane.iter().map(|v| v * 0.7).collect(),
        ],
        16,
    );
    let t = Instant::now();
    let a = analyze(&rgb);
    println!(
        "analysis:                     {:>7.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    let mut p = auto_pipeline(a.kind, &a);
    p.tone.enabled = true;
    p.tone.params.current.gamma = 1.2;
    p.color.params.saturation = 1.3;
    p.dering.enabled = true;
    let rgb = Arc::new(rgb);
    let mut planetary = p.clone();
    planetary.denoise.params.method = ps_core::enhance::NoiseMethod::Planetary;
    let mut last = None;
    for on_gpu in [false, true] {
        ps_core::gpu::set_enabled(on_gpu);
        for (name, p) in [("lunare", &p), ("planetaria", &planetary)] {
            let t = Instant::now();
            let out = Processor::default()
                .run(
                    Arc::clone(&rgb),
                    next_generation(),
                    p,
                    &Aux::default(),
                    &|| false,
                    &|_, _| {},
                )
                .unwrap();
            println!(
                "pipeline {name}, {}: {:>7.1} ms",
                if on_gpu { "GPU" } else { "CPU" },
                t.elapsed().as_secs_f64() * 1e3
            );
            for s in &out.timings {
                println!(
                    "  {:<28}{:>7.1} ms",
                    s.stage.name(),
                    s.elapsed.as_secs_f64() * 1e3
                );
            }
            last = Some(out);
        }
    }
    let out = last.unwrap();
    let t = Instant::now();
    let _ = out.image.to_rgba8();
    println!(
        "display conversion:           {:>7.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    // Deringing on its own, without the pipeline's caches around.
    drop(out);
    let reference = Image::new(w, h, vec![plane.clone(), plane.clone(), plane.clone()], 16);
    let sharp = reference.clone();
    let d = ps_core::dering::DeringParams::for_blur(1.6);
    let _ = ps_core::dering::dering(&sharp, &reference, &d, None, None);
    let t = Instant::now();
    let _ = ps_core::dering::dering(&sharp, &reference, &d, None, None);
    println!(
        "dering alone (RGB):           {:>7.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    for sigma in [1.5, 40.0] {
        let _ = ps_core::mask::gaussian_approx(&plane, w, h, sigma);
        let t = Instant::now();
        for _ in 0..3 {
            let _ = ps_core::mask::gaussian_approx(&plane, w, h, sigma);
        }
        println!(
            "gaussian blur σ {sigma:<4}:          {:>7.1} ms",
            t.elapsed().as_secs_f64() * 1e3 / 3.0
        );
    }
}
