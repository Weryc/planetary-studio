//! The noise estimator's inputs on real stacks, at each stage of the app's processing, with
//! views of each stage for a report: the app's learned noise reduction at 0, 0.5, 1, 1.5
//! (colour reduction 0‥3 levels), each as it is, after the smart sharpening with the subject
//! network's curve at 100 %, and after the app's current automatic smart sharpening.
//!
//! `cargo run --release -p ps-train --bin noise_try -- out_dir image…` → out_dir/reali.json
//! and JPEG crops.
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::analysis::analyze;
use ps_core::enhance::{NoiseMethod, NoiseParams, denoise};
use ps_core::noise_net::{features, measure};
use ps_core::smart::{SmartCurve, SmartParams};
use std::path::PathBuf;

const STRENGTHS: [(f32, u32); 4] = [(0.0, 0), (0.5, 1), (1.0, 2), (1.5, 3)];

fn arr(v: &[f32]) -> String {
    format!("[{}]", v.iter().map(|x| if x.is_finite() { format!("{x:.4}") } else { "null".into() }).collect::<Vec<_>>().join(","))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&out).unwrap();
    let mut rows = Vec::new();
    for (n, path) in args[1..].iter().map(PathBuf::from).enumerate() {
        let img = ps_io::load(&path).unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let a = analyze(&img);
        let Ok(net) = ps_core::curve_net::suggestion(&img, &a, ps_core::capture::Body::Jupiter) else {
            eprintln!("{stem}: nessuna curva della rete");
            continue;
        };
        let mut a_net = a.clone();
        a_net.curve_net = Some(net.clone());
        let app = ps_core::preset::auto_smart(a.kind, &a_net, ps_core::preset::AutoStrength::Normal);
        let rete = SmartParams { curve: SmartCurve::Network, strength: 1.0, color: ps_core::preset::learned_colour(&a), ..app.clone() };
        // One display for every stage: the image as it is (real stacks are already set for
        // display), white at its brightest.
        let mut l = ml::lum(&img);
        l.sort_by(f32::total_cmp);
        let top = l[l.len() * 999 / 1000].max(1e-6);
        let disk = a.disk.map_or(300.0, |d| 2.0 * d.radius);
        let c = ml::centre(&img, &a);
        let side = ((0.45 * disk) as usize).max(96).min(img.width()).min(img.height());
        let view = |x: &Image| -> Image {
            let crop = ml::crop_around(x, c, side);
            let k = (360 / side).max(1);
            let s = side * k;
            let planes = crop.channels().iter().map(|p| (0..s * s).map(|i| (p[(i / s / k) * side + (i % s) / k] / top).clamp(0.0, 1.0)).collect()).collect();
            Image::new(s, s, planes, 16)
        };
        let id = format!("r{n:02}");
        for (strength, levels) in STRENGTHS {
            let cleaned = if strength > 0.0 { denoise(&img, &NoiseParams { method: NoiseMethod::Learned, strength, detail: 0.3, chroma: levels, sky: 0.8 }) } else { img.clone() };
            let subject = ps_core::mask::subject_and_moons(&cleaned);
            let with_net = ps_core::smart::sharpen(&cleaned, None, &rete, Some(&subject));
            let with_app = ps_core::smart::sharpen(&cleaned, None, &app, Some(&subject));
            for (stage, x) in [("ridotta", &cleaned), ("rete100", &with_net), ("app", &with_app)] {
                let mut ai = analyze(x);
                ai.disk = a.disk;
                let Some((m, _)) = measure(x, &ai) else { continue };
                let file = format!("{id}_{strength}_{stage}.jpg");
                ps_io::save(&out.join(&file), &view(x)).unwrap();
                rows.push(format!(
                    "  {{\"id\": \"{id}\", \"nome\": \"{stem}\", \"forza_riduzione\": {strength}, \"colore_riduzione\": {levels}, \"fase\": \"{stage}\", \"x\": {}, \"cielo\": {}, \"nota_cielo\": {}, \"colore\": {}, \"vista\": \"{file}\", \"disco_px\": {disk:.0}, \"rete_picco_px\": {:.2}, \"rete_forza\": {:.2}, \"app_curva\": \"{:?}\", \"app_forza\": {:.2}}}",
                    arr(&features(&m)),
                    m.lum.sky.is_some(),
                    m.lum.sky_note.map_or("null".to_string(), |s| format!("\"{s}\"")),
                    ps_core::noise_net::chroma_planes(&img).is_some(),
                    1.0 / net.cutoff,
                    net.strength,
                    app.curve,
                    app.strength,
                ));
            }
        }
        eprintln!("{stem}: fatto");
    }
    std::fs::write(out.join("reali.json"), format!("[\n{}\n]\n", rows.join(",\n"))).unwrap();
}
