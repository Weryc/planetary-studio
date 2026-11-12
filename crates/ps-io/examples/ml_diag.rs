//! Why does our sharpening look flatter than BlurXTerminator's? On real stacks with a cached
//! BXT result, compares: the stack, the learned curve as the app applies it (luminance), the
//! same curve on R, G and B, and BXT. Inside the subject it measures:
//! - the luminance gain by size of detail (multiples of the cut-off, 0.06 = broad bands);
//! - the colour: saturation (chroma over luminance) and the gain of the chroma planes by size;
//! - the tone: mean and spread of the luminance.
//!
//! `cargo run --release -p ps-io --example ml_diag -- bxt_cache out.tsv image…`
#[path = "ml_common/mod.rs"]
mod ml;

use ml::*;
use ps_core::Image;
use ps_core::color::ColorMode;
use ps_core::smart::{SmartCurve, SmartParams};
use std::fmt::Write as _;
use std::path::Path;

/// Sizes of detail measured, as multiples of the cut-off.
const AT: [f32; 8] = [0.06, 0.1, 0.15, 0.25, 0.5, 1.0, 1.5, 2.0];

fn sharpen(img: &Image, p: &SmartParams) -> Image {
    let subject = (!p.include_sky).then(|| ps_core::mask::subject_and_moons(img));
    ps_core::smart::sharpen(img, None, p, subject.as_deref())
}

/// Chroma planes: R − L and B − L.
fn chroma(img: &Image) -> Option<[Vec<f32>; 2]> {
    let [r, _, b] = img.channels() else { return None };
    let l = lum(img);
    Some([r.iter().zip(&l).map(|(c, l)| c - l).collect(), b.iter().zip(&l).map(|(c, l)| c - l).collect()])
}

/// Saturation inside the square: RMS chroma over mean luminance.
fn saturation(img: &Image, sq: (usize, usize, usize)) -> f32 {
    let w = img.width();
    let Some([cr, cb]) = chroma(img) else { return 0.0 };
    let (cr, cb, l) = (crop(&cr, w, sq), crop(&cb, w, sq), crop(&lum(img), w, sq));
    let c = (cr.iter().zip(&cb).map(|(a, b)| a * a + b * b).sum::<f32>() / cr.len() as f32).sqrt();
    c / (l.iter().sum::<f32>() / l.len() as f32).max(1e-6)
}

fn mean_sd(v: &[f32]) -> (f32, f32) {
    let m = v.iter().sum::<f32>() / v.len() as f32;
    (m, (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32).sqrt())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cache = args.next().expect("bxt cache");
    let out = args.next().expect("out.tsv");
    let mut tsv = String::from("image\tkind\tversion\tsat\tmean\tsd");
    AT.iter().for_each(|r| write!(tsv, "\tL{r}").unwrap());
    AT[..6].iter().for_each(|r| write!(tsv, "\tC{r}").unwrap());
    tsv.push('\n');
    for path in args {
        let path = Path::new(&path);
        let Ok(src) = source(path) else { continue };
        let Ok(bxt) = ps_io::load(Path::new(&format!("{cache}/{}_bxt.tif", src.stem.replace(' ', "_")))) else { continue };
        if (bxt.width(), bxt.height(), bxt.channels().len()) != (src.img.width(), src.img.height(), src.img.channels().len()) {
            continue;
        }
        let a = &src.analysis;
        let Some(cutoff) = ps_core::preset::detail_cutoff(a) else { continue };
        let base = ps_core::preset::smart_structure(a.kind, a);
        let Some((strength, table)) = ps_core::learned::smart_curve(a) else { continue };
        // As the automatic sharpening applies it: strength capped, colour as the app picks.
        let ours = SmartParams { curve: SmartCurve::Learned, learned: table, learned_strength: strength, strength: strength.min(ps_core::preset::LEARNED_CAP), color: ps_core::preset::learned_colour(a), ..base.clone() };
        let mut versions: Vec<(&str, Image)> = vec![("stack", src.img.clone()), ("nostra", sharpen(&src.img, &ours))];
        if src.img.channels().len() == 3 {
            versions.push(("nostra_rgb", sharpen(&src.img, &SmartParams { color: ColorMode::Rgb, ..ours.clone() })));
        }
        versions.push(("bxt", bxt));
        let (w, sq) = (src.img.width(), src.square);
        let l0 = lum(&src.img);
        let c0 = chroma(&src.img);
        let (sat0, (m0, sd0)) = (saturation(&src.img, sq), mean_sd(&crop(&l0, w, sq)));
        for (name, img) in &versions {
            let l = lum(img);
            let g = gain(&l0, &l, w, sq);
            let (m, sd) = mean_sd(&crop(&l, w, sq));
            let sat = if sat0 > 0.0 { saturation(img, sq) / sat0 } else { f32::NAN };
            write!(tsv, "{}\t{:?}\t{name}\t{sat:.3}\t{:.3}\t{:.3}", src.stem, a.kind, m / m0, sd / sd0).unwrap();
            AT.iter().for_each(|r| write!(tsv, "\t{:.3}", at(&g, r * cutoff)).unwrap());
            match (&c0, chroma(img)) {
                (Some(c0), Some(c)) => {
                    // Both chroma planes together: their power summed.
                    let p0: Vec<(f32, f32)> = spectrum_of(&c0[0], w, sq).iter().zip(spectrum_of(&c0[1], w, sq)).map(|(a, b)| (a.0, a.1 + b.1)).collect();
                    let p1: Vec<(f32, f32)> = spectrum_of(&c[0], w, sq).iter().zip(spectrum_of(&c[1], w, sq)).map(|(a, b)| (a.0, a.1 + b.1)).collect();
                    let gc = smooth(&p0.iter().zip(&p1).map(|(a, b)| (a.0, (b.1 / a.1.max(1e-30)).sqrt())).collect::<Vec<_>>());
                    AT[..6].iter().for_each(|r| write!(tsv, "\t{:.3}", at(&gc, r * cutoff)).unwrap());
                }
                _ => AT[..6].iter().for_each(|_| tsv.push_str("\tnan")),
            }
            tsv.push('\n');
        }
        eprintln!("{} (sat {:.3}, cut-off {:.3}, forza {:.0} %)", src.stem, sat0, cutoff, strength * 100.0);
    }
    std::fs::write(out, tsv).unwrap();
}
