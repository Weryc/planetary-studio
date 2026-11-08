//! One line per image: size, kind, noise, limb blur, spectral cut-off; with `SCALE=1` the image
//! scale from the date in the name and the subject's apparent diameter instead.
//! `cargo run --release -p ps-io --example survey -- a.tif b.tif …`
fn main() {
    for path in std::env::args().skip(1) {
        let Ok(img) = ps_io::load(path.as_ref()) else {
            println!("{path}: unreadable");
            continue;
        };
        let a = ps_core::analysis::analyze(&img);
        let name: String = path.rsplit('/').next().unwrap_or(&path).chars().take(44).collect();
        if std::env::var_os("SCALE").is_some() {
            let stem = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
            let info = ps_core::capture::from_name(&stem);
            let scale = match (info.body, info.time) {
                (Some(b), Some(t)) => match ps_core::analysis::image_scale(&img, &a, b, &t) {
                    Some(s) => {
                        let d = ps_core::ephemeris::apparent_diameter(b, &t);
                        let f = |p: f32| 206.265 * p / s;
                        format!("{b:?} {d:.1}\" → {s:.3}\"/px (focale {:.0} mm @2.9µm, {:.0} @3.75µm, {:.0} @5.6µm)", f(2.9), f(3.75), f(5.6))
                    }
                    None => format!("{b:?}: nessun disco intero"),
                },
                _ => format!("oggetto {:?}, data {:?}", info.body, info.time.map(|t| (t.year, t.month, t.day))),
            };
            println!("{name:44} filtro {:?} | {scale}", info.filter);
            continue;
        }
        let s = a.spectral.map_or("-".to_string(), |s| format!("cut {:.3} (period {:.1} px) σfit {:.2}", s.cutoff, 1.0 / s.cutoff, s.sigma));
        println!(
            "{name:44} {:>4}×{:<4} {} ch {:?}/{:?} snr {:.0} limbσ {:?} size {:.0} | {s}",
            img.width(),
            img.height(),
            img.channels().len(),
            a.kind,
            a.confidence,
            a.snr,
            a.psf_sigma.map(|v| (v * 100.0).round() / 100.0),
            a.subject_size
        );
    }
}
