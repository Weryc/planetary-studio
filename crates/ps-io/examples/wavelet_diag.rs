//! Why the wavelets look the way they do on one image: for several settings (and an optional
//! BlurXTerminator result), how much each size of detail is amplified (output over input
//! spectrum, by period), plus what the automatic settings do inside: each layer's noise
//! estimate, the average share of gain left by the noise-adaptive factor and by the edge
//! protection. Writes a strip of crops for each variant.
//!
//! `cargo run --release -p ps-io --example wavelet_diag -- out_dir image.tif [bxt.tif]`
use ps_core::Image;
use ps_core::color::{ColorMode, working_planes};
use ps_core::pipeline::{Aux, Pipeline, Processor, next_generation};
use ps_core::preset::{AutoStrength, auto_pipeline_with};
use ps_core::wavelet::{ScaleMode, WaveletLayer, WaveletParams, decompose};
use std::path::Path;
use std::sync::Arc;

fn lum(img: &Image) -> Vec<f32> {
    working_planes(img, ColorMode::Luminance).swap_remove(0)
}

fn only(p: &Pipeline, deconv: bool, wavelet: bool) -> Pipeline {
    let mut q = p.clone();
    (q.align.enabled, q.bxt.enabled, q.denoise.enabled, q.contrast.enabled, q.dering.enabled) = (false, false, false, false, false);
    (q.deconv.enabled, q.wavelet.enabled, q.smart.enabled) = (deconv, wavelet, false);
    q
}

/// "Nitidezza intelligente" alone (or after the deconvolution) at `strength`.
fn smart(p: &Pipeline, a: &ps_core::analysis::Analysis, strength: f32, deconv: bool) -> Pipeline {
    let mut q = only(p, deconv, false);
    q.smart = ps_core::pipeline::Module::on(ps_core::smart::SmartParams { strength, ..ps_core::preset::smart_structure(a.kind, a) });
    q
}

fn run(img: &Arc<Image>, p: &Pipeline) -> Image {
    // As the app does: with the masks the pipeline needs (the subject, for smart sharpening).
    let masks = ps_core::pipeline::masks_for(p, img);
    Processor::default().run(Arc::clone(img), next_generation(), p, &Aux { masks: &masks, ..Default::default() }, &|| false, &|_, _| {}).unwrap().image.as_ref().clone()
}

const PERIODS: [f32; 10] = [32.0, 16.0, 11.0, 8.0, 6.0, 5.0, 4.0, 3.3, 2.7, 2.2];

/// Amplitude gain by period: √(P_out / P_in) averaged in a band around each period.
fn gain_curve(input: &[f32], output: &[f32], w: usize, h: usize, cx: f32, cy: f32) -> Vec<f32> {
    let a = ps_core::spectrum::radial_power(input, w, h, cx, cy).unwrap_or_default();
    let b = ps_core::spectrum::radial_power(output, w, h, cx, cy).unwrap_or_default();
    PERIODS
        .iter()
        .map(|&p| {
            let f = 1.0 / p;
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for ((fa, pa), (_, pb)) in a.iter().zip(&b) {
                if (fa / f).ln().abs() < 0.12 {
                    (num, den) = (num + *pb as f64, den + *pa as f64);
                }
            }
            if den > 0.0 { (num / den).sqrt() as f32 } else { f32::NAN }
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("output directory");
    let path = args.next().expect("image");
    let bxt = args.next().map(|p| ps_io::load(p.as_ref()).unwrap());
    let img = Arc::new(ps_io::load(path.as_ref()).unwrap());
    let a = ps_core::analysis::analyze(&img);
    let (w, h) = (img.width(), img.height());
    let (cx, cy) = a.disk.map_or((w as f32 / 2.0, h as f32 / 2.0), |d| (d.cx, d.cy));
    println!("{:?} cutoff {:?} (period {:?}) limbσ {:?} snr {:.0}", a.kind, a.spectral.map(|s| s.cutoff), a.spectral.map(|s| 1.0 / s.cutoff), a.psf_sigma, a.snr);

    let free = auto_pipeline_with(a.kind, &a, AutoStrength::Normal, ScaleMode::Free);
    let classic = auto_pipeline_with(a.kind, &a, AutoStrength::Normal, ScaleMode::Dyadic);
    // What a RegiStax user typically does: strong finest layer, a bit on the second, nothing
    // adaptive or protected, mild denoise.
    let mut registax = classic.clone();
    registax.wavelet.params = WaveletParams { scale_mode: ScaleMode::Dyadic, noise_adaptive: 0.0, edge_protection: 0.0, ..classic.wavelet.params.clone() };
    for (j, l) in registax.wavelet.params.layers.iter_mut().enumerate() {
        *l = WaveletLayer { gain: [4.0, 1.5, 0.3][..].get(j).copied().unwrap_or(0.0), denoise: if j == 0 { 1.0 } else { 0.0 } };
    }
    let mut free_plain = free.clone();
    (free_plain.wavelet.params.noise_adaptive, free_plain.wavelet.params.edge_protection) = (0.0, 0.0);
    // Free scales shaped like BlurXTerminator's response: most gain just above the cut-off,
    // the grain cleaned.
    let mut shaped = free.clone();
    shaped.wavelet.params.grain_denoise = 3.0;
    for (l, g) in shaped.wavelet.params.layers.iter_mut().zip([5.0, 3.0, 0.8, 0.2]) {
        l.gain = g;
    }

    let smart_variants = [
        ("intelligente 35 %", smart(&free, &a, 0.35, false)),
        ("intelligente 60 %", smart(&free, &a, 0.6, false)),
        ("deconv + intelligente 35 %", smart(&free, &a, 0.35, true)),
        ("intelligente 60 % uniforme", {
            let mut q = smart(&free, &a, 0.6, false);
            (q.smart.params.adaptive, q.smart.params.limb_protection) = (false, 0.0);
            q
        }),
    ];
    let input_plane = lum(&img);
    let mut variants: Vec<(String, Image)> = vec![("originale".into(), (*img).clone())];
    for (name, p, deconv) in [
        ("wavelet auto libere", &free, false),
        ("libere a forma BXT", &shaped, false),
        ("libere senza adattiva/bordi", &free_plain, false),
        ("wavelet auto classiche", &classic, false),
        ("stile RegiStax (classiche)", &registax, false),
        ("auto completo (deconv+wav)", &free, true),
    ] {
        let q = only(p, deconv, true);
        if std::env::var_os("SMART_ONLY").is_some() && !name.contains("libere") {
            continue;
        }
        let gains: Vec<f32> = q.wavelet.params.layers.iter().map(|l| l.gain).collect();
        println!(
            "{name}: scale {:?} first {:.2} gains {gains:?} adaptive {} edge {} denoise {:?} grain {:.2}",
            q.wavelet.params.scale_mode,
            q.wavelet.params.first_scale,
            q.wavelet.params.noise_adaptive,
            q.wavelet.params.edge_protection,
            q.wavelet.params.layers.iter().map(|l| l.denoise).collect::<Vec<_>>(),
            q.wavelet.params.grain_denoise
        );
        variants.push((name.into(), run(&img, &q)));
    }
    for (name, q) in &smart_variants {
        variants.push((name.to_string(), run(&img, q)));
    }
    if let Some(b) = bxt {
        variants.push(("BlurXTerminator".into(), b));
    }

    // Inside the automatic free wavelets: noise per layer and how much gain survives.
    let p = &free.wavelet.params;
    let d = decompose(&input_plane, w, h, p, &|| false).unwrap();
    let signal: Vec<usize> = (0..w * h).filter(|&i| input_plane[i] > 0.15 * input_plane.iter().copied().fold(0.0, f32::max)).collect();
    println!("layer  scale  noise σ   |coef| median  adaptive share  (edge share below)");
    for (j, layer) in d.layers.iter().enumerate() {
        let noise = d.noise[j];
        let a = p.noise_adaptive * noise;
        let mut coefs: Vec<f32> = signal.iter().map(|&i| layer[i].abs()).collect();
        coefs.sort_by(f32::total_cmp);
        let median = coefs.get(coefs.len() / 2).copied().unwrap_or(0.0);
        let share = signal.iter().map(|&i| if a > 0.0 { let r2 = (layer[i] / a).powi(2); r2 / (1.0 + r2) } else { 1.0 }).sum::<f32>() / signal.len() as f32;
        println!("  {}    {:.2}   {:.5}   {:.5}        {:.2}", j + 1, p.scale(j), noise, median, share);
    }

    // Measured inside the disk: the limb's overshoot, clipped against black sky, would read as
    // fine detail.
    let inner = |plane: &[f32]| -> (Vec<f32>, usize) {
        let side = a.disk.map_or(w.min(h), |d| ((d.radius * 1.3) as usize).max(96)).min(w).min(h);
        let (x0, y0) = ((cx as usize).saturating_sub(side / 2).min(w - side), (cy as usize).saturating_sub(side / 2).min(h - side));
        ((0..side).flat_map(|y| plane[(y0 + y) * w + x0..(y0 + y) * w + x0 + side].to_vec()).collect(), side)
    };
    let (inner_in, side) = inner(&input_plane);
    println!("\ngain by period (px), inside the disk ({side} px): {}", PERIODS.map(|p| format!("{p:>5}")).join(" "));
    for (name, v) in &variants[1..] {
        let (inner_out, _) = inner(&lum(v));
        let g = gain_curve(&inner_in, &inner_out, side, side, side as f32 / 2.0, side as f32 / 2.0);
        println!("{name:30} {}", g.iter().map(|x| format!("{x:5.2}")).collect::<Vec<_>>().join(" "));
    }

    // Strip: crop around the subject, 3×, stretched like the original.
    let half = a.disk.map_or(160.0, |d| d.radius * 0.55);
    let (x0, y0) = ((cx - half).max(0.0) as usize, (cy - half).max(0.0) as usize);
    let side = ((2.0 * half) as usize).min(w - x0).min(h - y0);
    let zoom = (330 / side).clamp(1, 3);
    let tile = side * zoom;
    let cols = 4;
    let rows = variants.len().div_ceil(cols);
    let (ow, oh) = (cols * (tile + 6), rows * (tile + 6));
    let mut sample: Vec<f32> = input_plane.clone();
    sample.sort_by(f32::total_cmp);
    let (lo, hi) = (sample[sample.len() / 100], sample[sample.len() * 999 / 1000].max(1e-4));
    let mut plane = vec![0.0f32; ow * oh];
    for (k, (_, v)) in variants.iter().enumerate() {
        let src = lum(v);
        let (ox, oy) = ((k % cols) * (tile + 6), (k / cols) * (tile + 6));
        for y in 0..tile {
            for x in 0..tile {
                plane[(oy + y) * ow + ox + x] = ((src[(y0 + y / zoom) * w + x0 + x / zoom] - lo) / (hi - lo)).clamp(0.0, 1.0);
            }
        }
    }
    let stem = Path::new(&path).file_stem().unwrap().to_string_lossy().replace(' ', "_");
    ps_io::save(Path::new(&format!("{out}/{stem}_diag.png")), &Image::new(ow, oh, vec![plane], 16)).unwrap();
    println!("\ntiles: {}", variants.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(" | "));
}
