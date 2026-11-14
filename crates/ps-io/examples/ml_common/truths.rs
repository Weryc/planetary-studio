//! The truths of the reference-based studies: sharp references made linear and brought to the
//! sizes amateur stacks have (`ml_ref` for the learned sharpening, `ps-train` for the learned
//! noise reduction).

use super::*;
use ps_core::resample::{Method, ResizeParams, resize};

/// λ/D at 550 nm for a 280 mm aperture, arcseconds.
pub const RESOLUTION: f32 = 0.405;

pub fn scaled(img: &Image, factor: f32) -> Image {
    let (w, h) = (((img.width() as f32 * factor).round() as usize).max(1), ((img.height() as f32 * factor).round() as usize).max(1));
    resize(img, &ResizeParams { width: w, height: h, method: Method::Mitchell, anti_ringing: 0.0 })
}

/// sRGB-ish encoded → linear, brightest part of the subject at about 0.85.
pub fn linear(img: &Image) -> Image {
    let planes: Vec<Vec<f32>> = img.channels().iter().map(|p| p.iter().map(|v| v.max(0.0).powf(2.2)).collect()).collect();
    let mut all: Vec<f32> = planes.iter().flat_map(|p| p.iter().step_by(7).copied()).collect();
    all.sort_by(f32::total_cmp);
    let top = all[all.len() * 999 / 1000].max(1e-4);
    Image::new(img.width(), img.height(), planes.iter().map(|p| p.iter().map(|v| (v * 0.85 / top).min(1.0)).collect()).collect(), 16)
}

pub fn mono(img: &Image) -> Image {
    Image::new(img.width(), img.height(), vec![lum(img)], 16)
}

/// The truths made from one reference: (tag, image, pixel in λ/D).
pub fn truths(stem: &str, reference: &Image) -> Vec<(String, Image, f32)> {
    // Big references are first brought under 3500 px: still far sharper than any target.
    let reference = if reference.width().max(reference.height()) > 3500 { scaled(reference, 3500.0 / reference.width().max(reference.height()) as f32) } else { reference.clone() };
    let reference = linear(&reference);
    let a = ps_core::analysis::analyze(&reference);
    let lower = stem.to_lowercase();
    let mut out = Vec::new();
    let is_surface = lower.starts_with("moon") || lower.starts_with("sdo") || lower.starts_with("sun");
    if !is_surface {
        // A planet: the subject at the spans amateur stacks have.
        // Spans in pixels and the subject's apparent size (arcsec: Saturn with its rings).
        let (spans, arcsec): (&[f32], f32) = match a.kind {
            ImageKind::Saturn => (&[450.0, 650.0, 900.0], 42.0),
            _ if lower.contains("1609") || lower.contains("1814c") || lower.contains("opo0322") => (&[150.0, 250.0, 350.0], 20.0),
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
pub fn busiest(img: &Image, side: usize, n: usize) -> Vec<(f32, f32)> {
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
