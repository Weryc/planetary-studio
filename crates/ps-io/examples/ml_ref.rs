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
use ml::truths::{mono, truths};
use ps_core::learned::{FEATURE_NAMES, features};
use ps_core::preset::{AutoStrength, auto_pipeline_with};
use ps_core::wavelet::ScaleMode;
use std::fmt::Write as _;
use std::path::Path;

/// The variant BlurXTerminator also runs on.
const TEACHER: (usize, usize) = (1, 1);

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
