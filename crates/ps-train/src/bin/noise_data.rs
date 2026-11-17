//! The noise dataset: for each simulated stack, at each stage of the app's processing, what the
//! noise estimator sees (`ps_core::noise_net::features`) and the noise there really is, known
//! from the sharp original (`ps_core::noise_net::incoherent`), for the grain (luminance) and
//! the colour (R − Y, B − Y).
//!
//! Stages: the app's learned noise reduction at strengths 0 (none), 0.5, 1 and 1.5 (colour
//! noise reduction 0, 1, 2 and 3 levels), each as it is and after the smart sharpening with the
//! subject network's curve at 100 %, as in the app (noise reduction comes first).
//!
//! `cargo run --release -p ps-train --bin noise_data -- out.json @runs.txt`
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::align::measure_shift;
use ps_core::analysis::analyze;
use ps_core::enhance::{NoiseMethod, NoiseParams, denoise};
use ps_core::fft::FftConvolver;
use ps_core::noise_net::{features, incoherent, measure, planes};
use ps_core::smart::{SmartCurve, SmartParams};
use rayon::prelude::*;
use std::path::PathBuf;

/// `plane` moved by `d` px exactly (a phase ramp; same direction as `align::shift`).
fn fourier_shift(plane: &[f32], w: usize, h: usize, d: [f32; 2]) -> Vec<f32> {
    let plan = FftConvolver::plan(w, h);
    let mut spec = plan.forward(plane);
    for (idx, c) in spec.iter_mut().enumerate() {
        let (kx, ky) = (idx / h, idx % h);
        let fy = if ky <= h / 2 { ky as f32 } else { ky as f32 - h as f32 } / h as f32;
        let phase = std::f32::consts::TAU * (kx as f32 / w as f32 * d[0] + fy * d[1]);
        let (sn, cs) = phase.sin_cos();
        let (re, im) = (c.re, c.im);
        c.re = re * cs - im * sn;
        c.im = re * sn + im * cs;
    }
    plan.inverse(spec)
}

fn arr(v: &[f32]) -> String {
    format!("[{}]", v.iter().map(|x| if x.is_finite() { format!("{x:.4}") } else { "null".into() }).collect::<Vec<_>>().join(","))
}

/// RMS error of `x` against `t` over `px` after the best scale, over the truth's spread there.
fn error(x: &[f32], t: &[f32], px: &[usize]) -> f32 {
    let (mut xt, mut xx, mut tt, mut st) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &i in px {
        xt += x[i] as f64 * t[i] as f64;
        xx += (x[i] as f64).powi(2);
        tt += (t[i] as f64).powi(2);
        st += t[i] as f64;
    }
    let var = tt - st * st / px.len().max(1) as f64;
    ((tt - xt * xt / xx.max(1e-30)).max(0.0) / var.max(1e-30)).sqrt() as f32
}

const STRENGTHS: [(f32, u32); 4] = [(0.0, 0), (0.5, 1), (1.0, 2), (1.5, 3)];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(&args[0]);
    let runs: Vec<PathBuf> = std::fs::read_to_string(args[1].trim_start_matches('@')).unwrap().lines().filter(|l| !l.trim().is_empty()).map(PathBuf::from).collect();
    let rows: Vec<String> = runs
        .par_iter()
        .flat_map(|run| {
            let name = run.file_name().unwrap().to_string_lossy().to_string();
            let raw = ml::load_any(&run.join("grezza.tif")).unwrap();
            let truth = ml::load_any(&run.join("verita_campionata.tif")).unwrap();
            let (w, h) = (raw.width(), raw.height());
            let s = measure_shift(&ml::lum(&truth), &ml::lum(&raw), w, h);
            let truth = Image::new(w, h, truth.channels().iter().map(|c| fourier_shift(c, w, h, s)).collect(), 16);
            let a = analyze(&raw);
            let Ok(net) = ps_core::curve_net::suggestion(&raw, &a, ps_core::capture::Body::Jupiter) else {
                eprintln!("{name}: nessuna curva della rete");
                return Vec::new();
            };
            let mut a_net = a.clone();
            a_net.curve_net = Some(net);
            let base = ps_core::preset::auto_smart(a.kind, &a_net, ps_core::preset::AutoStrength::Normal);
            let sharp = SmartParams { curve: SmartCurve::Network, strength: 1.0, color: ps_core::preset::learned_colour(&a), ..base };
            let mut out = Vec::new();
            for (strength, levels) in STRENGTHS {
                let cleaned = if strength > 0.0 { denoise(&raw, &NoiseParams { method: NoiseMethod::Learned, strength, detail: 0.3, chroma: levels, sky: 0.8 }) } else { raw.clone() };
                let subject = ps_core::mask::subject_and_moons(&cleaned);
                let sharpened = ps_core::smart::sharpen(&cleaned, None, &sharp, Some(&subject));
                for (stage, img) in [("ridotta", &cleaned), ("nitida", &sharpened)] {
                    // Its own measures, but the disk of the image as opened (as the app knows
                    // it): a sharpened limb can hide it from a new analysis.
                    let mut ai = analyze(img);
                    ai.disk = a.disk;
                    let Some((m, f)) = measure(img, &ai) else {
                        let fr = ps_core::curve_net::frame(img, &ai);
                        eprintln!("{name} {strength} {stage}: non misurabile (disco {:?}, tipo {:?}, cornice {})", ai.disk.map(|d| d.radius), ai.kind, fr.map_or("no".into(), |f| format!("{} dentro", f.inside.len())));
                        continue;
                    };
                    let t = truth.crop(f.x0, f.y0, f.side, f.side);
                    let ((xl, xc), (tl, tc)) = (planes(&f.crop), planes(&t));
                    let lum = incoherent(&xl, &tl, f.side, &f.inside, f.mean);
                    let chroma = match (xc, tc) {
                        (Some([xr, xb]), Some([tr, tb])) => {
                            let (r, b) = (incoherent(&xr, &tr, f.side, &f.inside, f.mean), incoherent(&xb, &tb, f.side, &f.inside, f.mean));
                            Some(r.iter().zip(&b).map(|(x, y)| (10f32.powf(*x) + 10f32.powf(*y)).log10()).collect::<Vec<f32>>())
                        }
                        _ => None,
                    };
                    let err = error(&xl, &tl, &f.inside);
                    out.push(format!(
                        "  {{\"nome\": \"{name}\", \"forza_riduzione\": {strength}, \"colore_riduzione\": {levels}, \"fase\": \"{stage}\", \"x\": {}, \"rumore_grana\": {}, \"rumore_colore\": {}, \"errore\": {err:.4}, \"rumore_app\": {:.6}, \"media\": {:.6}, \"cielo\": {}}}",
                        arr(&features(&m)),
                        arr(&lum),
                        chroma.as_ref().map_or("null".to_string(), |c| arr(c)),
                        ai.noise,
                        f.mean,
                        m.lum.sky.is_some(),
                    ));
                }
            }
            eprintln!("{name}: {} fasi", out.len());
            out
        })
        .collect();
    std::fs::write(&out, format!("[\n{}\n]\n", rows.join(",\n"))).unwrap();
    eprintln!("{} righe → {}", rows.len(), out.display());
}
