//! Quick image-level check of learned sharpening models, a few minutes per run: only the kinds
//! of subject asked for and a few variants per truth, the models applied exactly as the app's
//! automatic does (strength capped, colour mode by subject, limb protection by subject). The
//! error is RMS against the truth over the error with no sharpening (1 = no better than
//! nothing), inside the measuring square and, on Saturn, on the globe and on the rings apart.
//!
//! `cargo run --release -p ps-io --example ml_fast -- ref_dir [kinds=Saturn,Jupiter,Mars,Moon,Sun]
//!     [variants=medio_snr450,…] [label=heldout.tsv…] [saturn=label:globe_factor:ring_factor:denoise…]`
//!
//! `label=path` adds a model (held-out answers from `ml_train`); `saturn=` adds a split Saturn:
//! that model's curve with its strength scaled on the globe and on the rings, and the globe's
//! grain smoothed (`denoise`, 0‥1) where the sharpening made it too strong.
#[path = "ml_common/mod.rs"]
mod ml;

use ml::*;
use ps_core::Image;
use ps_core::analysis::Analysis;
use ps_core::learned::STEPS;
use ps_core::mask::{AutoRegion, MaskShape, MaskSpec};
use ps_core::smart::{SmartCurve, SmartParams};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

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

/// The learned curve `ln` as the automatic applies it.
fn as_app(a: &Analysis, ln: [f32; STEPS]) -> Option<SmartParams> {
    let (strength, table) = ps_core::learned::to_smart(ln)?;
    let base = ps_core::preset::smart_structure(a.kind, a);
    Some(SmartParams { curve: SmartCurve::Learned, learned: table, learned_strength: strength, strength: strength.min(ps_core::preset::LEARNED_CAP), color: ps_core::preset::learned_colour(a), ..base })
}

/// Today's fixed curve, as before the model.
fn fixed(a: &Analysis) -> SmartParams {
    let s = ps_core::preset::smart_structure(a.kind, a);
    SmartParams { curve: SmartCurve::Standard, learned: Vec::new(), strength: 0.35, color: ps_core::color::ColorMode::Luminance, ..s }
}

fn sharpen(img: &Image, p: &SmartParams) -> Image {
    let subject = (!p.include_sky).then(|| ps_core::mask::subject_and_moons(img));
    ps_core::smart::sharpen(img, None, p, subject.as_deref())
}

fn region(img: &Image, r: AutoRegion) -> Vec<f32> {
    ps_core::mask::render(&MaskSpec::new(MaskShape::Auto(r), 0.0), img)
}

/// Saturn split: globe and rings sharpened with the strength scaled apart, the globe's added
/// grain then smoothed by `denoise`.
fn split_saturn(img: &Image, p: &SmartParams, globe: &[f32], (gf, rf, denoise): (f32, f32, f32)) -> Image {
    let on = |f: f32| sharpen(img, &SmartParams { strength: (p.strength * f).min(ps_core::smart::MAX_STRENGTH), ..p.clone() });
    let (g, r) = (on(gf), if rf == gf { None } else { Some(on(rf)) });
    let g = if denoise > 0.0 {
        // The grain the sharpening added on the globe, blended back toward a smoothed copy.
        let smooth = blur(&g, 0.8);
        let ch = g.channels().iter().zip(smooth.channels()).map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + denoise * (y - x)).collect()).collect();
        Image::new(g.width(), g.height(), ch, 16)
    } else {
        g
    };
    let Some(r) = r else { return g };
    let ch = g.channels().iter().zip(r.channels()).map(|(a, b)| a.iter().zip(b).zip(globe).map(|((x, y), m)| m * x + (1.0 - m) * y).collect()).collect();
    Image::new(g.width(), g.height(), ch, 16)
}

/// RMS error against the truth over the pixels `idx` (means removed), over the unsharpened's.
fn err(out: &[f32], deg: &[f32], truth: &[f32], idx: &[usize]) -> f32 {
    let e = |x: &[f32]| {
        let n = idx.len().max(1) as f32;
        let m = idx.iter().map(|&i| x[i] - truth[i]).sum::<f32>() / n;
        (idx.iter().map(|&i| (x[i] - truth[i] - m).powi(2)).sum::<f32>() / n).sqrt()
    };
    e(out) / e(deg).max(1e-9)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("ref dir");
    let mut by_group = false;
    let (mut kinds, mut variants, mut models, mut splits) = (vec!["Saturn".to_string(), "Jupiter".into(), "Mars".into()], vec!["medio_snr450".to_string(), "turbolento_snr150".into(), "calmo_snr1300".into()], Vec::new(), Vec::new());
    for a in args {
        match a.split_once('=') {
            Some(("kinds", v)) => kinds = v.split(',').map(str::to_string).collect(),
            Some(("by", "group")) => by_group = true,
            Some(("variants", v)) => variants = v.split(',').map(str::to_string).collect(),
            Some(("saturn", v)) => {
                let c: Vec<&str> = v.split(':').collect();
                splits.push((c[0].to_string(), (c[1].parse::<f32>().unwrap(), c[2].parse::<f32>().unwrap(), c[3].parse::<f32>().unwrap())));
            }
            Some((label, path)) => models.push((label.to_string(), answers(path))),
            None => panic!("argomento {a}"),
        }
    }
    let nights = Nights::new();
    // (kind, method, zone) → errors.
    let mut res: BTreeMap<(String, String, &str), Vec<f32>> = BTreeMap::new();
    let started = std::time::Instant::now();
    let mut cases = 0;
    for line in std::fs::read_to_string(format!("{dir}/pairs.tsv")).unwrap().lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let (name, key) = (c[0], c[1]);
        let sq: (usize, usize, usize) = (c[2].parse().unwrap(), c[3].parse().unwrap(), c[4].parse().unwrap());
        let k: f32 = c[5].parse().unwrap();
        let Ok(truth) = ps_io::load(Path::new(&format!("{dir}/pairs/{key}.png"))) else { continue };
        let truth = if truth.channels().len() == 3 && truth.channels()[0] == truth.channels()[1] { Image::new(truth.width(), truth.height(), vec![truth.channels()[0].clone()], 16) } else { truth };
        // By what the reference really shows: the analysis can mistake one subject for another.
        let subject = subject_of(name.split('@').next().unwrap_or(name));
        if !kinds.iter().any(|k| k == subject) {
            continue;
        }
        let (w, lt) = (truth.width(), lum(&truth));
        // Zones, from the truth: the measuring square; on Saturn the globe and the rings.
        let square: Vec<usize> = (sq.1..sq.1 + sq.2).flat_map(|y| (sq.0..sq.0 + sq.2).map(move |x| y * w + x)).collect();
        let saturn = subject == "Saturn";
        let globe_mask = if saturn { region(&truth, AutoRegion::SaturnGlobe) } else { Vec::new() };
        let ring_mask = if saturn { region(&truth, AutoRegion::SaturnRings) } else { Vec::new() };
        let pick = |m: &[f32]| (0..m.len()).filter(|&i| m[i] > 0.5).collect::<Vec<usize>>();
        let mut zones: Vec<(&str, Vec<usize>)> = vec![("disco", square)];
        if saturn {
            zones.push(("globo", pick(&globe_mask)));
            zones.push(("anelli", pick(&ring_mask)));
        }
        for i in 0..ml::seeing::NIGHTS.len() {
            for j in 0..SNRS.len() {
                let variant = variant_name(i, j);
                if !variants.contains(&variant) {
                    continue;
                }
                // Only cases every model answers, so they are compared on the same images.
                let id = (name.to_string(), variant.clone());
                if models.iter().any(|(_, held)| !held.contains_key(&id)) {
                    continue;
                }
                let (_, degraded) = degrade(&truth, name, &nights, k, i, j);
                let a = ps_core::analysis::analyze(&degraded);
                // With `by=group`, results are kept per reference instead of per kind.
                let kind = if by_group { name.split('@').next().unwrap_or(name).to_string() } else { subject.to_string() };
                let ld = lum(&degraded);
                let mut record = |method: String, img: &Image| {
                    let lo = lum(img);
                    for (z, idx) in &zones {
                        res.entry((kind.clone(), method.clone(), z)).or_default().push(err(&lo, &ld, &lt, idx));
                    }
                };
                record("curva fissa".into(), &sharpen(&degraded, &fixed(&a)));
                let mut params: HashMap<&str, SmartParams> = HashMap::new();
                for (label, held) in &models {
                    let Some(p) = as_app(&a, held[&id]) else { continue };
                    record(label.clone(), &sharpen(&degraded, &p));
                    params.insert(label, p);
                }
                if saturn {
                    // The globe mask on the degraded image, feathered so the seam doesn't show.
                    let g = region(&degraded, AutoRegion::SaturnGlobe);
                    let g = ps_core::mask::gaussian_approx(&g, w, degraded.height(), 2.0);
                    for (label, f) in &splits {
                        let Some(p) = params.get(label.as_str()) else { continue };
                        record(format!("{label} globo×{} anelli×{} pulizia {}", f.0, f.1, f.2), &split_saturn(&degraded, p, &g, *f));
                    }
                }
                cases += 1;
            }
        }
    }
    let med = |v: &[f32]| {
        let mut v = v.to_vec();
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    println!("{cases} casi in {:.0} s (mediane; 1 = come non fare niente; solo i casi a cui rispondono tutti i modelli)", started.elapsed().as_secs_f32());
    let mut last = String::new();
    for ((kind, method, zone), v) in &res {
        if *kind != last {
            println!("== {kind}");
            last = kind.clone();
        }
        println!("  {method:<48} {zone:<7} {:.3}  ({} casi)", med(v), v.len());
    }
}
