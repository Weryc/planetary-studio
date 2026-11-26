//! Changing the image's size with classical, reproducible interpolation: the same input and
//! settings always give the same pixels (no learned model, nothing invented).
//!
//! Each method is a separable kernel: every output pixel is a weighted sum of the input
//! pixels around its position, first along the rows, then along the columns. When shrinking,
//! the kernel is stretched by the reduction so it also filters what the smaller image could
//! not hold (no aliasing).
//!
//! Sharp kernels (Catmull-Rom, Lanczos) have negative lobes: next to a hard edge — a
//! planet's limb against the black sky, a shadow on the Moon — they draw a thin dark/bright
//! halo. "Anti-alone" clamps each output pixel between the darkest and brightest of the 2×2
//! input pixels around it, which removes the halo and keeps the sharpness elsewhere.

use crate::i18n::tl;
use crate::Image;
use crate::analysis::ImageKind;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Method {
    /// Each output pixel copies the nearest input pixel: big visible squares.
    Nearest,
    /// Linear between the 2×2 nearest pixels: smooth, a little soft.
    Bilinear,
    /// Cubic B-spline: very smooth, never overshoots, softer than the others.
    BSpline,
    /// Mitchell–Netravali (B = C = ⅓): the classic compromise, almost no halo.
    Mitchell,
    /// Catmull-Rom cubic (the "bicubic" of most programs): sharper, slight halo.
    CatmullRom,
    /// Windowed sinc over 2 lobes.
    Lanczos2,
    /// Windowed sinc over 3 lobes: the sharpest that stays practical.
    Lanczos3,
    /// Windowed sinc over 4 lobes: a little crisper still, more halo.
    Lanczos4,
}

impl Method {
    pub const ALL: [Method; 8] = [Method::Nearest, Method::Bilinear, Method::BSpline, Method::Mitchell, Method::CatmullRom, Method::Lanczos2, Method::Lanczos3, Method::Lanczos4];

    pub fn label(self) -> &'static str {
        match self {
            Method::Nearest => tl("Vicino più prossimo"),
            Method::Bilinear => tl("Bilineare"),
            Method::BSpline => tl("B-spline cubica"),
            Method::Mitchell => "Mitchell-Netravali",
            Method::CatmullRom => tl("Bicubica (Catmull-Rom)"),
            Method::Lanczos2 => "Lanczos 2",
            Method::Lanczos3 => "Lanczos 3",
            Method::Lanczos4 => "Lanczos 4",
        }
    }

    /// One sentence on what it does to the image.
    pub fn description(self) -> &'static str {
        match self {
            Method::Nearest => tl("Ripete i pixel, a quadretti netti e senza nulla di inventato (utile per vedere i pixel originali)."),
            Method::Bilinear => tl("Media dei 4 pixel vicini, morbida e senza aloni ma perde un po' di dettaglio."),
            Method::BSpline => tl("La più liscia, senza aloni né rumore accentuato, ma ammorbidisce."),
            Method::Mitchell => tl("Compromesso classico tra nitidezza e morbidezza, quasi senza aloni."),
            Method::CatmullRom => tl("La bicubica dei programmi di fotoritocco, nitida con un piccolo alone sui bordi netti."),
            Method::Lanczos2 => tl("Sinc su 2 lobi, nitida come la bicubica con transizioni più pulite."),
            Method::Lanczos3 => tl("Sinc su 3 lobi, conserva il dettaglio fine meglio di tutte ma disegna un alone sui bordi netti."),
            Method::Lanczos4 => tl("Sinc su 4 lobi, leggermente più incisiva di Lanczos 3 con un alone un po' più marcato."),
        }
    }

    /// Half-width of the kernel at scale 1, in input pixels.
    fn support(self) -> f32 {
        match self {
            Method::Nearest => 0.5,
            Method::Bilinear => 1.0,
            Method::BSpline | Method::Mitchell | Method::CatmullRom | Method::Lanczos2 => 2.0,
            Method::Lanczos3 => 3.0,
            Method::Lanczos4 => 4.0,
        }
    }

    fn weight(self, x: f32) -> f32 {
        let x = x.abs();
        match self {
            Method::Nearest => {
                if x < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Method::Bilinear => (1.0 - x).max(0.0),
            Method::BSpline => cubic(x, 1.0, 0.0),
            Method::Mitchell => cubic(x, 1.0 / 3.0, 1.0 / 3.0),
            Method::CatmullRom => cubic(x, 0.0, 0.5),
            Method::Lanczos2 => lanczos(x, 2.0),
            Method::Lanczos3 => lanczos(x, 3.0),
            Method::Lanczos4 => lanczos(x, 4.0),
        }
    }

    /// Whether the kernel has negative lobes (and so can draw halos).
    pub fn overshoots(self) -> bool {
        matches!(self, Method::Mitchell | Method::CatmullRom | Method::Lanczos2 | Method::Lanczos3 | Method::Lanczos4)
    }
}

/// Mitchell–Netravali family of cubics.
fn cubic(x: f32, b: f32, c: f32) -> f32 {
    let (x2, x3) = (x * x, x * x * x);
    let v = if x < 1.0 {
        (12.0 - 9.0 * b - 6.0 * c) * x3 + (-18.0 + 12.0 * b + 6.0 * c) * x2 + (6.0 - 2.0 * b)
    } else if x < 2.0 {
        (-b - 6.0 * c) * x3 + (6.0 * b + 30.0 * c) * x2 + (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)
    } else {
        0.0
    };
    v / 6.0
}

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        let px = std::f32::consts::PI * x;
        px.sin() / px
    }
}

fn lanczos(x: f32, a: f32) -> f32 {
    if x < a { sinc(x) * sinc(x / a) } else { 0.0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResizeParams {
    pub width: usize,
    pub height: usize,
    pub method: Method,
    /// 0 = off, 1 = output kept within the 2×2 input neighbours (no halo). Only acts when enlarging.
    pub anti_ringing: f32,
}

/// The method suited to what is being enlarged, with the reason.
pub fn method_for(kind: ImageKind, sun: bool) -> (Method, f32, &'static str) {
    match kind {
        ImageKind::Planet | ImageKind::Saturn => (
            Method::Lanczos3,
            1.0,
            tl("Lanczos 3 con anti-alone pieno, che tiene il dettaglio fine di bande e divisioni senza l'anello scuro sul bordo contro il cielo nero."),
        ),
        ImageKind::FullDisk if sun => (
            Method::Mitchell,
            0.0,
            tl("Mitchell-Netravali, che rende la granulazione senza esaltarne il rumore e senza aloni sul bordo luminoso."),
        ),
        ImageKind::FullDisk => (
            Method::Mitchell,
            0.0,
            tl("Mitchell-Netravali, il compromesso classico con quasi niente alone su bordo e terminatore e senza esaltare il rumore."),
        ),
        ImageKind::Surface => (
            Method::Lanczos3,
            0.5,
            tl("Lanczos 3 con mezzo anti-alone, che conserva la trama fine (craterini, granulazione) senza bordi chiari accanto alle ombre nere."),
        ),
    }
}

/// One output position: the input pixels involved and their weights.
struct Taps {
    start: usize,
    weights: Vec<f32>,
}

/// Contributions for resampling `n_in` samples into `n_out`.
fn taps(method: Method, n_in: usize, n_out: usize) -> Vec<Taps> {
    let scale = n_out as f32 / n_in as f32;
    // Shrinking: the kernel is widened so it averages what is lost.
    let stretch = if scale < 1.0 { 1.0 / scale } else { 1.0 };
    let support = method.support() * stretch;
    (0..n_out)
        .map(|o| {
            // Pixel centres line up: output centre o + ½ maps to input (o + ½)/scale.
            let centre = (o as f32 + 0.5) / scale - 0.5;
            if method == Method::Nearest {
                let i = (centre.round().max(0.0) as usize).min(n_in - 1);
                return Taps { start: i, weights: vec![1.0] };
            }
            let lo = (centre - support).ceil() as isize;
            let hi = (centre + support).floor() as isize;
            let mut weights: Vec<(isize, f32)> = (lo..=hi).map(|i| (i, method.weight((i as f32 - centre) / stretch))).collect();
            // Samples beyond the edge repeat the edge pixel.
            let mut folded = vec![0.0f32; n_in];
            let (mut first, mut last) = (n_in, 0usize);
            for (i, w) in weights.drain(..) {
                let j = i.clamp(0, n_in as isize - 1) as usize;
                folded[j] += w;
                first = first.min(j);
                last = last.max(j);
            }
            let mut ws: Vec<f32> = folded[first..=last].to_vec();
            let sum: f32 = ws.iter().sum();
            if sum.abs() > 1e-9 {
                ws.iter_mut().for_each(|w| *w /= sum);
            }
            Taps { start: first, weights: ws }
        })
        .collect()
}

fn resize_plane(plane: &[f32], w: usize, h: usize, p: &ResizeParams, tx: &[Taps], ty: &[Taps]) -> Vec<f32> {
    let (ow, oh) = (p.width, p.height);
    // Along the rows.
    let mut rows = vec![0.0f32; ow * h];
    rows.par_chunks_mut(ow).enumerate().for_each(|(y, out)| {
        let src = &plane[y * w..(y + 1) * w];
        for (o, t) in out.iter_mut().zip(tx) {
            *o = t.weights.iter().zip(&src[t.start..]).map(|(a, b)| a * b).sum();
        }
    });
    // Along the columns.
    let mut out = vec![0.0f32; ow * oh];
    out.par_chunks_mut(ow).enumerate().for_each(|(oy, line)| {
        let t = &ty[oy];
        for (k, wgt) in t.weights.iter().enumerate() {
            let src = &rows[(t.start + k) * ow..(t.start + k + 1) * ow];
            for (o, s) in line.iter_mut().zip(src) {
                *o += wgt * s;
            }
        }
    });
    let strength = p.anti_ringing.clamp(0.0, 1.0);
    if strength > 0.0 && ow >= w && oh >= h {
        let (sx, sy) = (w as f32 / ow as f32, h as f32 / oh as f32);
        out.par_chunks_mut(ow).enumerate().for_each(|(oy, line)| {
            let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
            let y0 = (fy.floor() as usize).min(h - 1);
            let y1 = (y0 + 1).min(h - 1);
            for (ox, v) in line.iter_mut().enumerate() {
                let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
                let x0 = (fx.floor() as usize).min(w - 1);
                let x1 = (x0 + 1).min(w - 1);
                let n = [plane[y0 * w + x0], plane[y0 * w + x1], plane[y1 * w + x0], plane[y1 * w + x1]];
                let (lo, hi) = n.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &s| (lo.min(s), hi.max(s)));
                let clamped = v.clamp(lo, hi);
                *v += (clamped - *v) * strength;
            }
        });
    }
    out
}

/// The image at `p.width`×`p.height`. Values may slightly leave 0‥1 with sharp kernels; they
/// are clipped.
pub fn resize(img: &Image, p: &ResizeParams) -> Image {
    let (w, h) = (img.width(), img.height());
    let p = ResizeParams { width: p.width.max(1), height: p.height.max(1), ..*p };
    if p.width == w && p.height == h {
        return img.clone();
    }
    let (tx, ty) = (taps(p.method, w, p.width), taps(p.method, h, p.height));
    let channels = img
        .channels()
        .iter()
        .map(|c| {
            let mut out = resize_plane(c, w, h, &p, &tx, &ty);
            out.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
            out
        })
        .collect();
    Image::new(p.width, p.height, channels, img.source_bit_depth)
}

/// Print size: pixels from centimetres at a resolution in dots per inch, and back.
pub fn px_from_cm(cm: f32, dpi: f32) -> f32 {
    cm / 2.54 * dpi
}

pub fn cm_from_px(px: f32, dpi: f32) -> f32 {
    px / dpi * 2.54
}
