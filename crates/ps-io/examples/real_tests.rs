//! Side-by-side strips of real images under the automatic processing, to see how deconvolution
//! and wavelets behave: original | automatic (free scales) | automatic (classic scales) |
//! deconvolution only | wavelets only. Each strip is cropped around the subject, enlarged
//! for small subjects, and stretched like the original. Prints the settings used and a few
//! numbers per variant: dark rim outside the limb (ringing) and grain in the sky.
//!
//! `cargo run --release -p ps-io --example real_tests -- out_dir a.tif b.tif …`
use ps_core::Image;
use ps_core::pipeline::{Aux, Pipeline, Processor, next_generation};
use ps_core::preset::{AutoStrength, auto_pipeline_with};
use ps_core::wavelet::ScaleMode;
use std::path::Path;
use std::sync::Arc;

fn run(img: &Arc<Image>, p: &Pipeline) -> Image {
    // As the app does: with the masks the pipeline needs (the subject, for smart sharpening).
    let masks = ps_core::pipeline::masks_for(p, img);
    Processor::default().run(Arc::clone(img), next_generation(), p, &Aux { masks: &masks, ..Default::default() }, &|| false, &|_, _| {}).unwrap().image.as_ref().clone()
}

fn luminance(img: &Image) -> Vec<f32> {
    ps_core::color::working_planes(img, ps_core::color::ColorMode::Luminance).swap_remove(0)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("output directory");
    for path in args {
        let img = Arc::new(ps_io::load(path.as_ref()).unwrap());
        let a = ps_core::analysis::analyze(&img);
        let stem: String = Path::new(&path).file_stem().unwrap().to_string_lossy().chars().take(20).collect::<String>().replace(' ', "_");
        let (w, h) = (img.width(), img.height());

        let free = auto_pipeline_with(a.kind, &a, AutoStrength::Normal, ScaleMode::Free);
        let classic = auto_pipeline_with(a.kind, &a, AutoStrength::Normal, ScaleMode::Dyadic);
        let mut deconv_only = free.clone();
        (deconv_only.denoise.enabled, deconv_only.wavelet.enabled, deconv_only.dering.enabled) = (false, false, false);
        let mut wavelets_only = free.clone();
        (wavelets_only.deconv.enabled, wavelets_only.denoise.enabled, wavelets_only.dering.enabled) = (false, false, false);
        println!(
            "{stem}: {:?} {}×{} limbσ {:?} cut {:?} | psf {:?} it {} | free first {:.2} gains {:?} | classic gains {:?}",
            a.kind,
            w,
            h,
            a.psf_sigma.map(|v| (v * 100.0).round() / 100.0),
            a.spectral.map(|s| (s.cutoff * 1000.0).round() / 1000.0),
            free.deconv.params.psf,
            free.deconv.params.iterations,
            free.wavelet.params.first_scale,
            free.wavelet.params.layers.iter().map(|l| l.gain).collect::<Vec<_>>(),
            classic.wavelet.params.layers.iter().map(|l| l.gain).collect::<Vec<_>>()
        );
        let variants: Vec<(String, Image)> = if std::env::var_os("PSF").is_some() {
            // Deconvolution alone with the automatic PSF and with Gaussians from the spectrum:
            // the fitted fall-off σ, and the σ whose MTF is down to 10 % at the cut-off.
            let mut list = vec![("originale".to_string(), (*img).clone()), (format!("auto {:?}", free.deconv.params.psf), run(&img, &deconv_only))];
            let its = |p: &Pipeline| Processor::default().run(Arc::clone(&img), next_generation(), p, &Aux::default(), &|| false, &|_, _| {}).unwrap().deconv.map(|r| r.iterations);
            println!("   auto: iterations {:?}", its(&deconv_only));
            if let Some(s) = a.spectral {
                let core = 2.3548 * 0.3416 / s.cutoff;
                for (label, psf, stop) in [
                    ("Moffat β1.5 auto", ps_core::psf::Psf::Moffat { fwhm: core, beta: 1.5 }, true),
                    ("Moffat β1.5 fisso", ps_core::psf::Psf::Moffat { fwhm: core, beta: 1.5 }, false),
                    ("Moffat β2.5 fisso", ps_core::psf::Psf::Moffat { fwhm: core, beta: 2.5 }, false),
                ] {
                    let sigma = core / 2.3548;
                    let mut p = deconv_only.clone();
                    p.deconv.params.psf = psf;
                    p.deconv.params.auto_stop = stop;
                    println!("   {label}: σ {sigma:.2} iterations {:?}", its(&p));
                    list.push((format!("{label} {sigma:.2}"), run(&img, &p)));
                }
            }
            list
        } else {
            vec![
                ("originale".to_string(), (*img).clone()),
                ("auto libere".to_string(), run(&img, &free)),
                ("auto classiche".to_string(), run(&img, &classic)),
                ("solo deconv".to_string(), run(&img, &deconv_only)),
                ("solo wavelet".to_string(), run(&img, &wavelets_only)),
            ]
        };

        // Grain: RMS of the finest detail where the original is smooth sky or flat surface;
        // rim: how far below the sky the darkest ring just outside the limb goes.
        let lum0 = luminance(&img);
        for (name, v) in &variants[1..] {
            let lum = luminance(v);
            let fine = |p: &[f32]| -> f64 {
                let mut acc = (0.0f64, 0usize);
                for y in 1..h - 1 {
                    for x in 1..w - 1 {
                        let i = y * w + x;
                        let lap = 4.0 * p[i] - p[i - 1] - p[i + 1] - p[i - w] - p[i + w];
                        acc = (acc.0 + (lap as f64).powi(2), acc.1 + 1);
                    }
                }
                (acc.0 / acc.1.max(1) as f64).sqrt()
            };
            let rim = a.disk.map(|d| {
                let mut lowest = f32::MAX;
                for k in 0..180 {
                    let t = k as f32 * std::f32::consts::TAU / 180.0;
                    for dr in 1..(d.radius * 0.15).max(4.0) as i32 {
                        let r = d.radius + dr as f32;
                        let (x, y) = ((d.cx + r * t.cos()) as isize, (d.cy + r * t.sin()) as isize);
                        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                            let i = y as usize * w + x as usize;
                            lowest = lowest.min(lum[i] - lum0[i]);
                        }
                    }
                }
                lowest
            });
            println!("   {name:15} detail energy ×{:.2}  rim {:?}", fine(&lum) / fine(&lum0), rim.map(|r| (r * 1000.0).round() / 1000.0));
        }

        // Crop around the subject (whole frame for surfaces), at least 2× for small ones.
        let (cx, cy, half) = match a.disk {
            Some(d) => (d.cx, d.cy, (d.radius * 1.35).max(40.0)),
            None if a.kind == ps_core::analysis::ImageKind::Saturn => (w as f32 / 2.0, h as f32 / 2.0, a.subject_size * 0.6),
            None => (w as f32 / 2.0, h as f32 / 2.0, (w.min(h) as f32 / 2.0).min(200.0)),
        };
        let (x0, y0) = ((cx - half).max(0.0) as usize, (cy - half).max(0.0) as usize);
        let side = ((2.0 * half) as usize).min(w - x0).min(h - y0);
        let zoom = (360 / side).clamp(1, 4);
        let tile = side * zoom;
        let channels = img.channels().len();
        let ow = variants.len() * (tile + 6);
        let mut planes = vec![vec![0.0f32; ow * tile]; channels];
        // Stretch from the original: black at its 1st percentile, white at its 99.9th.
        let mut sample: Vec<f32> = lum0.iter().step_by(3).copied().collect();
        sample.sort_by(f32::total_cmp);
        let (lo, hi) = (sample[sample.len() / 100], sample[sample.len() * 999 / 1000].max(1e-4));
        for (k, (_, v)) in variants.iter().enumerate() {
            for (plane, src) in planes.iter_mut().zip(v.channels()) {
                for y in 0..tile {
                    for x in 0..tile {
                        let value = src[(y0 + y / zoom) * w + x0 + x / zoom];
                        plane[y * ow + k * (tile + 6) + x] = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
                    }
                }
            }
        }
        ps_io::save(Path::new(&format!("{out}/{stem}.png")), &Image::new(ow, tile, planes, 16)).unwrap();
    }
}
