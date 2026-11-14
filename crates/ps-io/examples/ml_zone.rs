//! Where the learned sharpening leaves grain: error against the truth by zone of the disk
//! (centre, outer ring towards the limb and poles), on the simulated pairs of `ml_ref`, for the
//! sharpening as the app applies it (automatic strength).
//!
//! Per kind of subject; `kind:width:protection` arguments (e.g. `Saturn:4:0.6`) override the
//! limb protection for that kind, to compare settings.
//!
//! `cargo run --release -p ps-io --example ml_zone -- ref_dir [Kind:width:protection…]`
#[path = "ml_common/mod.rs"]
mod ml;

use ml::*;
use ps_core::Image;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("ref dir");
    let limb = limb_overrides(args);
    let nights = Nights::new();
    let mut zones: std::collections::BTreeMap<String, (Vec<f32>, Vec<f32>)> = Default::default();
    for line in std::fs::read_to_string(format!("{dir}/pairs.tsv")).unwrap().lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let (name, key, k): (&str, &str, f32) = (c[0], c[1], c[5].parse().unwrap());
        let Ok(truth) = ps_io::load(Path::new(&format!("{dir}/pairs/{key}.png"))) else { continue };
        let truth = if truth.channels().len() == 3 && truth.channels()[0] == truth.channels()[1] { Image::new(truth.width(), truth.height(), vec![truth.channels()[0].clone()], 16) } else { truth };
        let at = ps_core::analysis::analyze(&truth);
        let Some(e) = outline(&truth, &at) else { continue };
        let lt = lum(&truth);
        let w = truth.width();
        for j in [0usize, 1] {
            let (_, degraded) = degrade(&truth, name, &nights, k, 1, j);
            let a = ps_core::analysis::analyze(&degraded);
            let mut p = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
            if let Some((w, pr)) = limb.get(&format!("{:?}", a.kind)) {
                (p.limb_width, p.limb_protection) = (*w, *pr);
            }
            if p.curve != ps_core::smart::SmartCurve::Learned {
                continue;
            }
            let subject = ps_core::mask::subject_and_moons(&degraded);
            let out = ps_core::smart::sharpen(&degraded, None, &p, Some(&subject));
            let (lo, ld) = (lum(&out), lum(&degraded));
            // Per zone: RMS error after over before, mean removed.
            let zone = |inner: f32, outer_r: f32| -> f32 {
                let idx: Vec<usize> = (0..lt.len()).filter(|&i| { let r = e.radius((i % w) as f32, (i / w) as f32); r >= inner && r < outer_r }).collect();
                let err = |x: &[f32]| { let m = idx.iter().map(|&i| x[i] - lt[i]).sum::<f32>() / idx.len() as f32; (idx.iter().map(|&i| (x[i] - lt[i] - m).powi(2)).sum::<f32>() / idx.len() as f32).sqrt() };
                err(&lo) / err(&ld).max(1e-9)
            };
            let z = zones.entry(format!("{:?}", a.kind)).or_default();
            z.0.push(zone(0.0, 0.55));
            z.1.push(zone(0.55, 0.93));
        }
    }
    let med = |v: &mut Vec<f32>| { v.sort_by(f32::total_cmp); v[v.len() / 2] };
    for (kind, (mut centre, mut outer)) in zones {
        println!("{kind:<8} {:>4} casi · centro {:.3} · anello esterno {:.3}", centre.len(), med(&mut centre), med(&mut outer));
    }
}
