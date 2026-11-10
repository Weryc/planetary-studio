//! What the ephemerides say is on an image, with the names drawn on a copy:
//! `cargo run --release -p ps-io --example identify -- out.png image.tif [Body] [yyyy-mm-dd-hhmm]`
//! The body and time default to what the file name says (or the planet the pixels suggest).
use ps_core::capture::{Body, Moment};
use ps_core::identify::{LabelKind, identify};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (out, path) = (&args[1], std::path::Path::new(&args[2]));
    let img = ps_io::load(path).unwrap();
    let info = ps_core::capture::from_name(path.file_stem().unwrap().to_str().unwrap());
    let body = args.get(3).and_then(|b| Body::from_word(b)).or(info.body).or_else(|| ps_core::analysis::analyze(&img).planet).expect("which body?");
    let when = args
        .get(4)
        .and_then(|t| ps_core::capture::from_name(t).time)
        .or(info.time)
        .unwrap_or(Moment { year: 2026, month: 1, day: 1, hour: 0, minute: 0, second: 0.0 });
    let start = std::time::Instant::now();
    let id = identify(&img, body, &when, None);
    println!("{} in {:?}", body.label(), start.elapsed());
    for (k, v) in &id.facts {
        println!("  {k}: {v}");
    }
    for n in &id.notes {
        println!("  · {n}");
    }
    println!("  orientamento {:?} (trovato: {}), disco {:?}", id.orientation, id.orientation_found, id.disk);
    // Draw: a cross per label (white found, grey predicted), the disk outline.
    let (w, h) = (img.width(), img.height());
    let lum = ps_core::color::working_planes(&img, ps_core::color::ColorMode::Luminance).swap_remove(0);
    let mut rgb: Vec<Vec<f32>> = vec![lum.clone(), lum.clone(), lum];
    let mut dot = |x: f32, y: f32, c: [f32; 3]| {
        if x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h {
            let i = y as usize * w + x as usize;
            for k in 0..3 {
                rgb[k][i] = c[k];
            }
        }
    };
    if let Some((cx, cy, r)) = id.disk {
        for k in 0..720 {
            let a = k as f32 / 720.0 * std::f32::consts::TAU;
            dot(cx + r * a.cos(), cy + r * a.sin(), [0.0, 0.8, 1.0]);
        }
    }
    for l in &id.labels {
        let c = match l.kind {
            LabelKind::MoonFound => [0.2, 1.0, 0.2],
            LabelKind::MoonPredicted => [1.0, 0.6, 0.0],
            LabelKind::Crater => [1.0, 1.0, 0.0],
            _ => [1.0, 0.3, 0.3],
        };
        for d in -6..=6 {
            dot(l.x + d as f32, l.y, c);
            dot(l.x, l.y + d as f32, c);
        }
        for k in 0..90 {
            let a = k as f32 / 90.0 * std::f32::consts::TAU;
            dot(l.x + l.radius * a.cos(), l.y + l.radius * a.sin(), c);
        }
        println!("  {:<22} {:>7.1} {:>7.1} {:?}{}", l.name, l.x, l.y, l.kind, if l.on_terminator { " (terminatore)" } else { "" });
    }
    ps_io::save(out.as_ref(), &ps_core::Image::new(w, h, rgb, 16)).unwrap();
}
