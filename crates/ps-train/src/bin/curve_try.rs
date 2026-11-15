//! The subject's network on real stacks: what it measures, what it suggests, and the image
//! sharpened with the app's current automatic curve and with the network's, side by side.
//!
//! `cargo run --release -p ps-train --bin curve_try -- out_dir [--body Jupiter] image…`
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::analysis::analyze;
use ps_core::capture::Body;
use ps_core::smart::{SmartCurve, SmartParams};
use std::path::PathBuf;

/// Display: the luminance's 99.9th percentile at white, gamma 2.2.
fn view(img: &Image, top: f32) -> Image {
    let planes = img.channels().iter().map(|p| p.iter().map(|v| (v / top).clamp(0.0, 1.0).powf(1.0 / 2.2)).collect()).collect();
    Image::new(img.width(), img.height(), planes, 16)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&out).unwrap();
    let mut body = Body::Jupiter;
    let mut images = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--body" {
            body = Body::ALL.into_iter().find(|b| format!("{b:?}") == args[i + 1]).expect("soggetto sconosciuto");
            i += 2;
        } else {
            images.push(PathBuf::from(&args[i]));
            i += 1;
        }
    }
    for path in images {
        let img = ps_io::load(&path).unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
        let mut a = analyze(&img);
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        match ps_core::curve_net::suggestion(&img, &a, body) {
            Ok(n) => {
                println!(
                    "{stem}: picco su {:.1} px ({:.3} cicli/px), forza {:.2}, disaccordo {:.2}, cielo {}",
                    1.0 / n.cutoff,
                    n.cutoff,
                    n.strength,
                    n.spread,
                    n.sky_note.unwrap_or("usato")
                );
                a.curve_net = Some(n);
            }
            Err(why) => {
                println!("{stem}: nessuna curva della rete ({why})");
                continue;
            }
        }
        let app = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
        let net = SmartParams { curve: SmartCurve::Network, strength: app.network_strength, color: ps_core::preset::learned_colour(&a), ..app.clone() };
        println!("  app: curva {:?}, forza {:.2}, taglio {:.3}", app.curve, app.strength, app.cutoff);
        let subject = ps_core::mask::subject_and_moons(&img);
        let (s_app, s_net) = rayon::join(|| ps_core::smart::sharpen(&img, None, &app, Some(&subject)), || ps_core::smart::sharpen(&img, None, &net, Some(&subject)));
        let mut l = ml::lum(&img);
        l.sort_by(f32::total_cmp);
        let top = l[l.len() * 999 / 1000].max(1e-6);
        let c = ml::centre(&img, &a);
        let side = (a.disk.map_or(400.0, |d| 2.2 * d.radius) as usize).min(img.width()).min(img.height());
        let crop = |x: &Image| view(&ml::crop_around(x, c, side), top);
        let (g, ap, ne) = (crop(&img), crop(&s_app), crop(&s_net));
        // Side by side: raw, app, network.
        let w = g.width();
        let planes = (0..3)
            .map(|k| {
                let pick = |im: &Image| im.channels()[k.min(im.channels().len() - 1)].clone();
                let (p0, p1, p2) = (pick(&g), pick(&ap), pick(&ne));
                (0..g.height()).flat_map(|y| p0[y * w..(y + 1) * w].iter().chain(&p1[y * w..(y + 1) * w]).chain(&p2[y * w..(y + 1) * w]).copied().collect::<Vec<_>>()).collect()
            })
            .collect();
        let sheet = Image::new(3 * w, g.height(), planes, 16);
        ps_io::save(&out.join(format!("{stem}_grezza_app_rete.jpg")), &sheet).unwrap();
    }
}
