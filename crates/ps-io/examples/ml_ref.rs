//! Learned sharpening: examples made from SHARP references (finished Hubble, Cassini, SDO
//! images and NASA's LRO-based Moon renders), so the truth is really sharp.
//!
//! `cargo run --release -p ps-io --example ml_ref -- out_dir reference…`
//!
//! Each reference is made linear (the files are gamma-encoded), shrunk to the sizes amateur
//! stacks have (disk spans by body; for the Moon and the Sun, tiles of detailed zones and the
//! whole disk), and that is the truth. Each truth is then seen on three simulated nights
//! (`ml_common/seeing.rs`: lucky imaging through Kolmogorov turbulence, best frames aligned and
//! stacked, per colour, with atmospheric dispersion) at the pixel scale a 280 mm telescope
//! would have for that size, and given stack-like noise at three levels; every other truth is
//! turned to mono, like a filtered mono camera.
//!
//! Writes `out_dir/dataset.tsv` (same columns as `ml_dataset`: the answer is the best linear
//! gain back to the truth; BlurXTerminator's gain on one variant per truth), the pairs as
//! 16-bit PNGs in `out_dir/pairs/` and their measuring squares and pixel scales in
//! `out_dir/pairs.tsv`, for
//! `ml_eval_ref`. BlurXTerminator results are cached in `out_dir/bxt/` (`NO_BXT=1` skips it).
#[path = "ml_common/mod.rs"]
mod ml;

use ml::*;
use ps_core::Image;
use ps_core::analysis::ImageKind;
use ps_core::learned::{FEATURE_NAMES, features};
use ps_core::preset::{AutoStrength, auto_pipeline_with};
use ps_core::resample::{Method, ResizeParams, resize};
use ps_core::wavelet::ScaleMode;
use std::fmt::Write as _;
use std::path::Path;

/// The variant BlurXTerminator also runs on.
const TEACHER: (usize, usize) = (1, 1);

/// λ/D at 550 nm for a 280 mm aperture, arcseconds.
const RESOLUTION: f32 = 0.405;

fn scaled(img: &Image, factor: f32) -> Image {
    let (w, h) = (((img.width() as f32 * factor).round() as usize).max(1), ((img.height() as f32 * factor).round() as usize).max(1));
    resize(img, &ResizeParams { width: w, height: h, method: Method::Mitchell, anti_ringing: 0.0 })
}

/// sRGB-ish encoded → linear, brightest part of the subject at about 0.85.
fn linear(img: &Image) -> Image {
    let planes: Vec<Vec<f32>> = img.channels().iter().map(|p| p.iter().map(|v| v.max(0.0).powf(2.2)).collect()).collect();
    let mut all: Vec<f32> = planes.iter().flat_map(|p| p.iter().step_by(7).copied()).collect();
    all.sort_by(f32::total_cmp);
    let top = all[all.len() * 999 / 1000].max(1e-4);
    Image::new(img.width(), img.height(), planes.iter().map(|p| p.iter().map(|v| (v * 0.85 / top).min(1.0)).collect()).collect(), 16)
}

fn mono(img: &Image) -> Image {
    Image::new(img.width(), img.height(), vec![lum(img)], 16)
}

/// The truths made from one reference: (tag, image, pixel in λ/D).
fn truths(stem: &str, reference: &Image) -> Vec<(String, Image, f32)> {
    // Big references are first brought under 3500 px: still far sharper than any target.
    let reference = if reference.width().max(reference.height()) > 3500 { scaled(reference, 3500.0 / reference.width().max(reference.height()) as f32) } else { reference.clone() };
    let reference = linear(&reference);
    let a = ps_core::analysis::analyze(&reference);
    let lower = stem.to_lowercase();
    let mut out = Vec::new();
    let is_surface = lower.starts_with("moon") || lower.starts_with("sdo");
    if !is_surface {
        // A planet: the subject at the spans amateur stacks have.
        // Spans in pixels and the subject's apparent size (arcsec: Saturn with its rings).
        let (spans, arcsec): (&[f32], f32) = match a.kind {
            ImageKind::Saturn => (&[450.0, 650.0, 900.0], 42.0),
            _ if lower.contains("1609") || lower.contains("1814c") => (&[150.0, 250.0, 350.0], 20.0),
            _ => (&[300.0, 450.0, 650.0], 44.0),
        };
        let centre = centre(&reference, &a);
        for &span in spans {
            let f = span / a.subject_size.max(1.0);
            if f >= 0.95 {
                continue;
            }
            let small = scaled(&reference, f);
            let side = ((span * 1.35) as usize).max(256);
            out.push((format!("d{}", span as usize), crop_around(&small, (centre.0 * f, centre.1 * f), side), arcsec / span / RESOLUTION));
        }
    } else {
        // The Moon and the Sun: the whole disk, and tiles of the most detailed zones.
        let disk = a.disk.map_or(a.subject_size, |d| 2.0 * d.radius).max(1.0);
        let whole = scaled(&reference, 1000.0 / disk);
        // The whole disk with a small telescope; the close-ups at an amateur's planetary scale.
        out.push(("disco".to_string(), whole, 0.5));
        for &f in &[0.7f32, 0.5] {
            let small = scaled(&reference, f);
            for (k, centre) in busiest(&small, 512, 3).into_iter().enumerate() {
                out.push((format!("z{}_{k}", (f * 10.0) as usize), crop_around(&small, centre, 512), if f > 0.6 { 0.22 } else { 0.3 }));
            }
        }
    }
    out
}

/// Centres of the `n` 512² zones with the most fine detail, apart from each other.
fn busiest(img: &Image, side: usize, n: usize) -> Vec<(f32, f32)> {
    let (w, h) = (img.width(), img.height());
    if w < side || h < side {
        return vec![(w as f32 / 2.0, h as f32 / 2.0)];
    }
    let l = lum(img);
    let mut scored = Vec::new();
    let step = side / 4;
    for y0 in (0..=h - side).step_by(step) {
        for x0 in (0..=w - side).step_by(step) {
            let (mut detail, mut lit) = (0.0f64, 0usize);
            for y in (y0 + 1..y0 + side - 1).step_by(3) {
                for x in (x0 + 1..x0 + side - 1).step_by(3) {
                    let v = l[y * w + x];
                    if v > 0.05 {
                        lit += 1;
                    }
                    detail += ((4.0 * v - l[y * w + x - 1] - l[y * w + x + 1] - l[(y - 1) * w + x] - l[(y + 1) * w + x]).abs()) as f64;
                }
            }
            // Mostly lit: shadows and sky don't teach much.
            if lit * 9 * 10 > side * side * 7 {
                scored.push((detail, (x0 + side / 2) as f32, (y0 + side / 2) as f32));
            }
        }
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut picked: Vec<(f32, f32)> = Vec::new();
    for (_, x, y) in scored {
        if picked.iter().all(|p| (p.0 - x).hypot(p.1 - y) > side as f32 * 0.9) {
            picked.push((x, y));
        }
        if picked.len() == n {
            break;
        }
    }
    picked
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("out dir");
    let (pairs, cache) = (format!("{out}/pairs"), format!("{out}/bxt"));
    std::fs::create_dir_all(&pairs).unwrap();
    std::fs::create_dir_all(&cache).unwrap();
    let exe = if std::env::var_os("NO_BXT").is_some() { None } else { ps_core::bxt::find_executable() };
    let mut text = String::new();
    let mut squares = String::new();
    let nights = Nights::new();
    let grid = |p: &str| (0..ps_core::learned::STEPS).map(|i| format!("{p}{}", ps_core::learned::GRID[i])).collect::<Vec<_>>().join("\t");
    writeln!(text, "name\tgroup\tvariant\tkind\tbody\tcutoff\t{}\t{}\t{}", FEATURE_NAMES.join("\t"), grid("bxt_"), grid("wiener_")).unwrap();
    let mut alternate = 0usize;
    for path in args {
        let path = Path::new(&path);
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let Some(reference) = load_any(path) else {
            eprintln!("{stem}: illeggibile");
            continue;
        };
        // The same observation released twice is one group.
        let group = stem.trim_end_matches("_pub").to_string();
        let started = std::time::Instant::now();
        let mut rows = 0;
        for (tag, truth, k) in truths(&stem, &reference) {
            alternate += 1;
            let truth = if alternate.is_multiple_of(2) && !truth.is_mono() { mono(&truth) } else { truth };
            let name = format!("{stem}@{tag}{}", if truth.is_mono() { "_mono" } else { "" });
            let at = ps_core::analysis::analyze(&truth);
            let Some(sq) = inner_square(truth.width(), truth.height(), outline(&truth, &at), 64) else {
                eprintln!("  {name}: soggetto troppo piccolo");
                continue;
            };
            let truth_key = name.replace([' ', '@'], "_");
            ps_io::save(Path::new(&format!("{pairs}/{truth_key}.png")), &truth).unwrap();
            writeln!(squares, "{name}\t{truth_key}\t{}\t{}\t{}\t{k:.4}", sq.0, sq.1, sq.2).unwrap();
            for i in 0..ml::seeing::NIGHTS.len() {
                for j in 0..SNRS.len() {
                    let (blurred, degraded) = degrade(&truth, &name, &nights, k, i, j);
                    let variant = variant_name(i, j);
                    let ad = ps_core::analysis::analyze(&degraded);
                    let (Some(s), Some(f)) = (ad.spectral, features(&ad)) else { continue };
                    let best = on_grid(&wiener_with(&truth, &blurred, &degraded, sq), s.cutoff);
                    let key = format!("{}_{variant}", name.replace([' ', '@'], "_"));
                    let teacher = if (i, j) == TEACHER {
                        let cached = format!("{cache}/{key}_bxt.tif");
                        let out = if Path::new(&cached).exists() {
                            ps_io::load(cached.as_ref()).ok()
                        } else {
                            exe.as_ref().and_then(|exe| {
                                let params = auto_pipeline_with(ad.kind, &ad, AutoStrength::Normal, ScaleMode::Free).bxt.params;
                                let o = ps_core::bxt::run(&degraded, &params, exe, &|| false).ok().flatten()?;
                                ps_io::save(cached.as_ref(), &o).ok();
                                Some(o)
                            })
                        };
                        out.map(|o| on_grid(&gain(&lum(&degraded), &lum(&o), degraded.width(), sq), s.cutoff))
                    } else {
                        None
                    };
                    let curve = |v: Option<[f32; ps_core::learned::STEPS]>| v.map_or_else(|| ["nan"; ps_core::learned::STEPS].join("\t"), |g| g.map(|x| format!("{x:.4}")).join("\t"));
                    writeln!(
                        text,
                        "{name}\t{group}\t{variant}\t{:?}\t{}\t{:.4}\t{}\t{}\t{}",
                        ad.kind,
                        ad.planet.map_or("-".into(), |b| format!("{b:?}")),
                        s.cutoff,
                        f.map(|x| format!("{x:.4}")).join("\t"),
                        curve(teacher),
                        curve(Some(best))
                    )
                    .unwrap();
                    rows += 1;
                }
            }
        }
        eprintln!("{stem:<24} {rows} esempi in {:.0} s", started.elapsed().as_secs_f32());
        std::fs::write(format!("{out}/dataset.tsv"), &text).unwrap();
        std::fs::write(format!("{out}/pairs.tsv"), &squares).unwrap();
    }
}
