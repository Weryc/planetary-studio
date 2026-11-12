//! Image-level check of the learned sharpening.
//!
//! `cargo run --release -p ps-io --example ml_eval_ref -- ref_dir heldout.tsv out_dir [bxt_cache image…]`
//!
//! For every truth `ml_ref` saved, each degraded variant is rebuilt exactly and sharpened with:
//! today's automatic fixed curve, the model (held out: trained without that reference) on the
//! luminance and on R, G, B, the model pushed ×1.5, the truth's own best curve; BlurXTerminator
//! where it ran. The error is RMS against the truth inside the subject, over the error with no
//! sharpening: on the luminance, and for colour truths on the chroma (R − L, B − L) too.
//! Writes `out_dir/eval.json` and strips. With the optional arguments, the real stacks are
//! shown side by side too: original | today | model | model on R, G, B | BXT.
#[path = "ml_common/mod.rs"]
mod ml;

use ml::*;
use ps_core::Image;
use ps_core::analysis::{Analysis, ImageKind};
use ps_core::color::ColorMode;
use ps_core::learned::STEPS;
use ps_core::smart::{SmartCurve, SmartParams};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

fn today(a: &Analysis) -> SmartParams {
    let s = ps_core::preset::smart_structure(a.kind, a);
    let strength = if matches!(a.kind, ImageKind::Surface | ImageKind::FullDisk) { 0.4 } else { 0.35 };
    SmartParams { curve: SmartCurve::Standard, learned: Vec::new(), strength, ..s }
}

fn pushed(a: &Analysis, ln: [f32; STEPS], factor: f32) -> Option<SmartParams> {
    let (strength, table) = ps_core::learned::to_smart(ln)?;
    Some(SmartParams { curve: SmartCurve::Learned, learned: table, strength: (strength * factor).min(ps_core::smart::MAX_STRENGTH), ..today(a) })
}

fn sharpen(img: &Image, p: &SmartParams) -> Image {
    let subject = (!p.include_sky).then(|| ps_core::mask::subject_and_moons(img));
    ps_core::smart::sharpen(img, None, p, subject.as_deref())
}

/// Chroma planes R − L and B − L (none for mono).
fn chroma(img: &Image) -> Option<[Vec<f32>; 2]> {
    let [r, _, b] = img.channels() else { return None };
    let l = lum(img);
    Some([r.iter().zip(&l).map(|(c, l)| c - l).collect(), b.iter().zip(&l).map(|(c, l)| c - l).collect()])
}

fn rmse(a: &[f32], b: &[f32], w: usize, sq: (usize, usize, usize)) -> f32 {
    let (x, y) = (crop(a, w, sq), crop(b, w, sq));
    let (ma, mb) = (x.iter().sum::<f32>() / x.len() as f32, y.iter().sum::<f32>() / y.len() as f32);
    (x.iter().zip(&y).map(|(p, q)| ((p - ma) - (q - mb)).powi(2)).sum::<f32>() / x.len() as f32).sqrt()
}

/// Panels side by side around the subject, zoomed, stretched alike (colour kept).
fn strip(panels: &[&Image], centre: (f32, f32), radius: f32, gamma: f32, file: &Path) {
    let first = panels[0];
    let (w, h) = (first.width(), first.height());
    let half = radius.clamp(40.0, 170.0);
    let x0 = ((centre.0 - half).max(0.0) as usize).min(w.saturating_sub(1));
    let y0 = ((centre.1 - half).max(0.0) as usize).min(h.saturating_sub(1));
    let s = ((2.0 * half) as usize).min(w - x0).min(h - y0);
    let zoom = (340 / s.max(1)).clamp(1, 4);
    let t = s * zoom;
    let gap = 6;
    let (ow, oh) = (panels.len() * (t + gap) - gap, t);
    let mut sorted: Vec<f32> = crop(&lum(first), w, (x0, y0, s)).to_vec();
    sorted.sort_by(f32::total_cmp);
    let (lo, hi) = (sorted[sorted.len() / 200], sorted[sorted.len() * 998 / 1000].max(1e-4));
    let mut planes = vec![vec![0.08f32; ow * oh]; 3];
    for (k, img) in panels.iter().enumerate() {
        for (c, plane) in planes.iter_mut().enumerate() {
            let src = &img.channels()[c.min(img.channels().len() - 1)];
            for y in 0..t {
                for x in 0..t {
                    let v = src[(y0 + y / zoom) * w + x0 + x / zoom];
                    plane[y * ow + k * (t + gap) + x] = ((v - lo) / (hi - lo)).clamp(0.0, 1.0).powf(1.0 / gamma);
                }
            }
        }
    }
    ps_io::save(file, &Image::new(ow, oh, planes, 16)).unwrap();
}

fn answers(path: &str) -> HashMap<(String, String), [f32; STEPS]> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            ((c[0].to_string(), c[1].to_string()), std::array::from_fn(|i| c[2 + i].parse().unwrap_or(0.0)))
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("ref dir");
    let held = answers(&args.next().expect("heldout.tsv"));
    let out = args.next().expect("out dir");
    std::fs::create_dir_all(&out).unwrap();
    let rows = read_rows(Path::new(&format!("{dir}/dataset.tsv")));
    let nights = Nights::new();
    let mut json = String::from("{\"results\": [\n");
    let mut first = true;
    for line in std::fs::read_to_string(format!("{dir}/pairs.tsv")).unwrap().lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let (name, key) = (c[0], c[1]);
        let sq: (usize, usize, usize) = (c[2].parse().unwrap(), c[3].parse().unwrap(), c[4].parse().unwrap());
        let k: f32 = c[5].parse().unwrap();
        let Ok(truth) = ps_io::load(Path::new(&format!("{dir}/pairs/{key}.png"))) else { continue };
        let truth = if truth.channels().len() == 3 && truth.channels()[0] == truth.channels()[1] { Image::new(truth.width(), truth.height(), vec![truth.channels()[0].clone()], 16) } else { truth };
        let started = std::time::Instant::now();
        let lt = lum(&truth);
        let ct = chroma(&truth);
        let w = truth.width();
        for i in 0..ml::seeing::NIGHTS.len() {
            for j in 0..SNRS.len() {
                let variant = variant_name(i, j);
                let Some(row) = rows.iter().find(|r| r.name == name && r.variant == variant) else { continue };
                let Some(model) = held.get(&(name.to_string(), variant.clone())) else { continue };
                let (_, degraded) = degrade(&truth, name, &nights, k, i, j);
                let a = ps_core::analysis::analyze(&degraded);
                let base = rmse(&lum(&degraded), &lt, w, sq);
                let score = |img: &Image| rmse(&lum(img), &lt, w, sq) / base.max(1e-9);
                // The colour's error: both chroma planes, over the error with no sharpening.
                let chroma_base = ct.as_ref().zip(chroma(&degraded)).map(|(t, d)| rmse(&d[0], &t[0], w, sq) + rmse(&d[1], &t[1], w, sq));
                let colour = |img: &Image| -> Option<f32> {
                    let (t, b) = (ct.as_ref()?, chroma_base?);
                    let c = chroma(img)?;
                    Some((rmse(&c[0], &t[0], w, sq) + rmse(&c[1], &t[1], w, sq)) / b.max(1e-9))
                };
                let fixed = sharpen(&degraded, &today(&a));
                let learned = pushed(&a, *model, 1.0);
                let lum_mode = learned.as_ref().map(|p| sharpen(&degraded, p));
                let rgb_mode = learned.as_ref().filter(|_| !degraded.is_mono()).map(|p| sharpen(&degraded, &SmartParams { color: ColorMode::Rgb, ..p.clone() }));
                let more = pushed(&a, *model, 1.5).map(|p| score(&sharpen(&degraded, &p)));
                let oracle = row.wiener.and_then(|t| pushed(&a, t, 1.0)).map(|p| score(&sharpen(&degraded, &p)));
                let bxt = ps_io::load(Path::new(&format!("{dir}/bxt/{key}_{variant}_bxt.tif"))).ok();
                let num = |v: Option<f32>| v.map_or("null".into(), |v| format!("{v:.4}"));
                if !first {
                    json.push_str(",\n");
                }
                first = false;
                write!(
                    json,
                    "  {{\"name\": \"{name}\", \"group\": \"{}\", \"variant\": \"{variant}\", \"night\": \"{}\", \"kind\": \"{:?}\", \"fixed\": {:.4}, \"model\": {}, \"model_rgb\": {}, \"model_x1.5\": {}, \"oracle\": {}, \"bxt\": {}, \"c_fixed\": {}, \"c_model\": {}, \"c_model_rgb\": {}, \"c_bxt\": {}, \"model_strength\": {}}}",
                    row.group,
                    ml::seeing::NIGHTS[i].label,
                    a.kind,
                    score(&fixed),
                    num(lum_mode.as_ref().map(score)),
                    num(rgb_mode.as_ref().map(score)),
                    num(more),
                    num(oracle),
                    num(bxt.as_ref().map(score)),
                    num(colour(&fixed)),
                    num(lum_mode.as_ref().and_then(colour)),
                    num(rgb_mode.as_ref().and_then(colour)),
                    num(bxt.as_ref().and_then(colour)),
                    num(ps_core::learned::to_smart(*model).map(|s| s.0))
                )
                .unwrap();
                if (i, j) == (1, 1)
                    && let Some(l) = &lum_mode
                {
                    let (cx, cy, r) = outline(&degraded, &a).map_or(((sq.0 + sq.2 / 2) as f32, (sq.1 + sq.2 / 2) as f32, sq.2 as f32 * 0.6), |e| (e.cx, e.cy, e.rx.max(e.ry) * 1.1));
                    let mut panels = vec![&truth, &degraded, &fixed, l];
                    if let Some(r) = &rgb_mode {
                        panels.push(r);
                    }
                    if let Some(b) = &bxt {
                        panels.push(b);
                    }
                    // The references were made linear: shown with a display gamma.
                    strip(&panels, (cx, cy), r, 2.2, &Path::new(&out).join(format!("{key}_synth.png")));
                }
            }
        }
        eprintln!("{name:<40} {:.0} s", started.elapsed().as_secs_f32());
    }
    json.push_str("\n]}\n");
    std::fs::write(Path::new(&out).join("eval.json"), json).unwrap();

    // The real stacks, side by side, with the model as the app has it (trained on everything).
    let Some(cache) = args.next() else { return };
    for path in args {
        let path = Path::new(&path);
        let Ok(src) = source(path) else { continue };
        let a = &src.analysis;
        let Some(ln) = ps_core::learned::predict(a) else { continue };
        let Some(p) = pushed(a, ln, 1.0) else { continue };
        let fixed = sharpen(&src.img, &today(a));
        let model = sharpen(&src.img, &p);
        let rgb = (!src.img.is_mono()).then(|| sharpen(&src.img, &SmartParams { color: ColorMode::Rgb, ..p.clone() }));
        let file_key = src.stem.replace(' ', "_");
        let bxt = ps_io::load(Path::new(&format!("{cache}/{file_key}_bxt.tif"))).ok();
        let mut panels = vec![&src.img, &fixed, &model];
        if let Some(r) = &rgb {
            panels.push(r);
        }
        if let Some(b) = &bxt {
            panels.push(b);
        }
        let (c, r) = outline(&src.img, a).map_or(((src.img.width() as f32 / 2.0, src.img.height() as f32 / 2.0), 160.0), |e| ((e.cx, e.cy), e.rx.max(e.ry) * 1.1));
        strip(&panels, c, r, 1.0, &Path::new(&out).join(format!("{file_key}_real.png")));
        eprintln!("{} (vera)", src.stem);
    }
}
