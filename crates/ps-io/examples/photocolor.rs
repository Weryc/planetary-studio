//! Photometric colour on a Jupiter picture: the fit for every reference, and a side-by-side
//! (original | each reference): `cargo run --release -p ps-io --example photocolor -- out.png image.tif [yyyy-mm-dd-hhmm]`
//! The planet's tilt comes from the capture time (file name, or the argument) when there is one.
use ps_core::photocolor::{PhotoColorParams, Reference, analyse, apply_fit};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = std::path::Path::new(&args[2]);
    let img = ps_io::load(path).unwrap();
    let when = args.get(3).and_then(|t| ps_core::capture::from_name(t).time).or_else(|| ps_core::capture::from_name(path.file_stem().unwrap().to_str().unwrap()).time);
    let tilt = when.map_or(0.0, |w| ps_core::moons::jupiter_tilt(&w)) as f32;
    println!("tilt {tilt:+.2}° ({})", if when.is_some() { "from the capture time" } else { "no time: 0" });
    let (w, h) = (img.width(), img.height());
    let mut cols = vec![img.clone()];
    for r in Reference::ALL {
        let params = PhotoColorParams { reference: r, strength: 1.0, tilt };
        let t = std::time::Instant::now();
        match analyse(&img, &params) {
            Some(fit) => {
                println!("{:<13} {} bands, distance {:.3} -> {:.3}, north at +b {}, spot {}, gains {:?}, in {:?}", r.label(), fit.bands, fit.before, fit.after, fit.north_at_plus_b, fit.spot, fit.gains.map(|g| (g * 100.0).round() / 100.0), t.elapsed());
                cols.push(apply_fit(&img, &fit, 1.0));
            }
            None => {
                println!("{}: no fit", r.label());
                cols.push(img.clone());
            }
        }
    }
    let n = cols.len();
    let mut out = vec![vec![0.0f32; w * n * h]; 3];
    for (k, c) in cols.iter().enumerate() {
        for ch in 0..3 {
            for y in 0..h {
                out[ch][y * w * n + k * w..y * w * n + (k + 1) * w].copy_from_slice(&c.channels()[ch][y * w..(y + 1) * w]);
            }
        }
    }
    ps_io::save(args[1].as_ref(), &ps_core::Image::new(w * n, h, out, 16)).unwrap();
}
