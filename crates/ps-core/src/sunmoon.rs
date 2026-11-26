//! The Moon or the Sun, from the pixels: for a close-up of a surface and for a whole disk, which
//! the analysis tells apart from a planet by size and framing but not from each other.
//!
//! What gives each away:
//! - a lunar close-up is relief lit from one side: its brightness changes mostly along the
//!   light's direction, with sharp shadow edges on one side and gentle lit slopes on the other
//!   (a strongly one-directional, lopsided gradient), and its shadows are deep;
//! - a solar close-up (granulation, spots, H-alpha fibrils) has texture in every direction,
//!   evenly, and no deep shadows besides a spot's umbra;
//! - a lunar disk is about as bright at the limb as at the centre, with the large dark maria
//!   inside, often in phase (not round);
//! - a solar disk darkens towards the limb, is otherwise even, and is round.
//!
//! In H-alpha or through a red filter a colour image is strongly red: also the Sun.

use crate::Image;
use crate::analysis::{Analysis, ImageKind};
use crate::capture::Body;
use crate::color::{ColorMode, working_planes};

/// What [`classify`] looks at (also printed by `ps-train`'s `sunmoon_eval`).
#[derive(Clone, Debug, Default)]
pub struct Features {
    /// How one-directional the gradient is (0 = the same in every direction, 1 = all along one).
    pub anisotropy: f32,
    /// Skewness of the gradient along that direction: sharp shadow edges against gentle slopes.
    pub skew: f32,
    /// Share of the frame much darker than its typical brightness (shadows, a spot's umbra).
    pub dark: f32,
    /// Red over blue of a colour image (`None` for mono).
    pub red: Option<f32>,
    /// Spread of the brightness (5th to 95th percentile) over its median, away from the sky.
    pub contrast: f32,
    /// Share of the ground, away from the sky, darker than half its median (shadows).
    pub shadows: f32,
    /// Fine detail's spread over the median brightness.
    pub texture: f32,
    /// A disk: brightness near the limb over the centre's.
    pub limb: Option<f32>,
    /// A disk: spread of its large-scale brightness inside (the maria), over its mean.
    pub patches: Option<f32>,
}

/// Longest side the measures are taken at (px).
const SIDE: usize = 512;

/// Measure `img` (its analysis `a`).
pub fn features(img: &Image, a: &Analysis) -> Features {
    let factor = img.width().max(img.height()).div_ceil(SIDE).max(1);
    let small = if factor > 1 { img.downscaled(factor) } else { img.clone() };
    let (w, h) = (small.width(), small.height());
    let lum = working_planes(&small, ColorMode::Luminance).swap_remove(0);
    let smooth = crate::mask::gaussian_approx(&lum, w, h, 1.5);
    // The relief's own detail, without the large-scale light (a limb, the sweet spot of an
    // H-alpha filter, the terminator) whose single direction would pass for one light source.
    let broad = crate::mask::gaussian_approx(&lum, w, h, 6.0);
    let detail: Vec<f32> = smooth.iter().zip(&broad).map(|(s, b)| s - b).collect();
    // Where to look: the disk's inside (85 % of its radius), else the frame less its borders
    // (a stack's edges are often black or smeared).
    let disk = a.disk.filter(|_| a.kind != ImageKind::Surface).map(|d| (d.cx / factor as f32, d.cy / factor as f32, d.radius / factor as f32));
    let inside = |x: usize, y: usize| -> bool {
        match disk {
            Some((cx, cy, r)) => (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) < 0.85 * r,
            None => x >= w / 10 && x < w - w / 10 && y >= h / 10 && y < h - h / 10,
        }
    };
    // Away from the sky (and from a lunar night side, which looks the same): its edge, the limb
    // or the terminator, is one long edge in one direction.
    let near_sky = near_sky(&smooth, w, h);
    let px: Vec<usize> = (1..h - 1).flat_map(|y| (1..w - 1).map(move |x| (x, y))).filter(|&(x, y)| inside(x, y)).map(|(x, y)| y * w + x).filter(|&i| !near_sky[i]).collect();
    let mut f = Features::default();
    if px.len() < 100 {
        return f;
    }
    // Gradient structure: its main direction, how dominant it is, and how lopsided along it.
    // The strongest gradients are capped: a limb against the sky is a few strong pixels all in
    // one direction, and must not outweigh the texture all over the frame.
    let grads: Vec<(f32, f32)> = px.iter().map(|&i| (0.5 * (detail[i + 1] - detail[i - 1]), 0.5 * (detail[i + w] - detail[i - w]))).collect();
    let mut sizes: Vec<f32> = grads.iter().map(|g| g.0.hypot(g.1)).collect();
    let k = (sizes.len() * 9 / 10).min(sizes.len() - 1);
    let cap = sizes.select_nth_unstable_by(k, f32::total_cmp).1.max(1e-12);
    let (mut xx, mut yy, mut xy) = (0.0f64, 0.0f64, 0.0f64);
    for &(gx, gy) in &grads {
        let m = gx.hypot(gy);
        let s = if m > cap { cap / m } else { 1.0 };
        let (gx, gy) = (gx * s, gy * s);
        (xx, yy, xy) = (xx + (gx * gx) as f64, yy + (gy * gy) as f64, xy + (gx * gy) as f64);
    }
    let (mean, diff) = ((xx + yy) / 2.0, (((xx - yy) / 2.0).powi(2) + xy * xy).sqrt());
    f.anisotropy = (diff / mean.max(1e-30)) as f32;
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let (ux, uy) = (angle.cos() as f32, angle.sin() as f32);
    // Lopsidedness from the tails (3rd and 97th percentiles): sharp shadow edges one way,
    // gentle slopes the other; robust to the few pixels of a limb.
    let mut along: Vec<f32> = grads.iter().map(|(gx, gy)| gx * ux + gy * uy).collect();
    let n = along.len();
    let lo = *along.select_nth_unstable_by(n * 3 / 100, f32::total_cmp).1;
    let hi = *along.select_nth_unstable_by((n * 97 / 100).min(n - 1), f32::total_cmp).1;
    f.skew = ((hi + lo) / (hi - lo).max(1e-12)).abs();
    let mut values: Vec<f32> = px.iter().map(|&i| smooth[i]).collect();
    let n = values.len();
    let q = |v: &mut Vec<f32>, f: f32| *v.select_nth_unstable_by(((n as f32 * f) as usize).min(n - 1), f32::total_cmp).1;
    let (q05, q50, q95) = (q(&mut values, 0.05), q(&mut values, 0.5), q(&mut values, 0.95));
    f.contrast = (q95 - q05) / q50.max(1e-9);
    f.shadows = px.iter().filter(|&&i| smooth[i] < 0.5 * q50).count() as f32 / n as f32;
    let fine: Vec<f32> = px.iter().map(|&i| lum[i] - smooth[i] + detail[i]).collect();
    let mean_fine = fine.iter().sum::<f32>() / n as f32;
    f.texture = (fine.iter().map(|v| (v - mean_fine).powi(2)).sum::<f32>() / n as f32).sqrt() / q50.max(1e-9);
    // Deep darks against the typical brightness.
    let mut values: Vec<f32> = px.iter().map(|&i| smooth[i]).collect();
    let k = (values.len() * 3 / 4).min(values.len() - 1);
    let typical = *values.select_nth_unstable_by(k, f32::total_cmp).1;
    f.dark = px.iter().filter(|&&i| smooth[i] < 0.3 * typical).count() as f32 / px.len() as f32;
    if let [r, _, b] = small.channels() {
        let (sr, sb) = px.iter().fold((0.0f64, 0.0f64), |(sr, sb), &i| (sr + r[i] as f64, sb + b[i] as f64));
        f.red = (sb > 0.0).then(|| (sr / sb) as f32);
    }
    if let Some((cx, cy, r)) = disk {
        let ring = |lo: f32, hi: f32| -> Option<f32> {
            let (mut sum, mut n) = (0.0f32, 0usize);
            for (i, v) in lum.iter().enumerate() {
                let d = ((i % w) as f32 + 0.5 - cx).hypot((i / w) as f32 + 0.5 - cy) / r;
                if (lo..hi).contains(&d) {
                    (sum, n) = (sum + v, n + 1);
                }
            }
            (n > 20).then(|| sum / n as f32)
        };
        if let (Some(edge), Some(centre)) = (ring(0.8, 0.92), ring(0.0, 0.5)) {
            f.limb = Some(edge / centre.max(1e-9));
        }
        let broad = crate::mask::gaussian_approx(&lum, w, h, (0.04 * r).max(2.0));
        let v: Vec<f32> = (0..w * h).filter(|&i| ((i % w) as f32 + 0.5 - cx).hypot((i / w) as f32 + 0.5 - cy) < 0.7 * r).map(|i| broad[i]).collect();
        if v.len() > 20 {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
            f.patches = Some(s / m.max(1e-9));
        }
    }
    f
}

/// Pixels within [`SKY_REACH`] px of the sky: dark (below 12 % of the bright level) and connected
/// to the frame's edge over a large area.
fn near_sky(lum: &[f32], w: usize, h: usize) -> Vec<bool> {
    let mut sample: Vec<f32> = lum.to_vec();
    let k = (sample.len() * 99 / 100).min(sample.len().saturating_sub(1));
    let top = *sample.select_nth_unstable_by(k, f32::total_cmp).1;
    let dark: Vec<bool> = lum.iter().map(|&v| v < 0.12 * top).collect();
    let mut sky = vec![false; w * h];
    let mut stack: Vec<usize> = (0..w).flat_map(|x| [x, (h - 1) * w + x]).chain((0..h).flat_map(|y| [y * w, y * w + w - 1])).filter(|&i| dark[i]).collect();
    stack.iter().for_each(|&i| sky[i] = true);
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        let next = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
        for j in next.into_iter().flatten() {
            if dark[j] && !sky[j] {
                sky[j] = true;
                stack.push(j);
            }
        }
    }
    if sky.iter().filter(|&&s| s).count() < w * h / 100 {
        return vec![false; w * h];
    }
    let plane: Vec<f32> = sky.iter().map(|&s| s as u8 as f32).collect();
    let (_, grown) = crate::dering::min_max(&plane, w, h, SKY_REACH);
    grown.iter().map(|&v| v > 0.5).collect()
}

/// How far from the sky the measures start (px, at the working size).
const SKY_REACH: usize = 10;

/// The Moon or the Sun, for a close-up of a surface or a whole disk (`None` for other kinds, or
/// when the pixels don't say).
pub fn classify(img: &Image, a: &Analysis) -> Option<Body> {
    if !matches!(a.kind, ImageKind::Surface | ImageKind::FullDisk) {
        return None;
    }
    let f = features(img, a);
    Some(decide(a.kind, a.disk.is_some(), &f))
}

/// [`classify`] from the measures.
pub fn decide(kind: ImageKind, round: bool, f: &Features) -> Body {
    // Strongly red: H-alpha or a red filter on the Sun.
    if f.red.is_some_and(|r| r > 2.5) {
        return Body::Sun;
    }
    match kind {
        ImageKind::FullDisk if !round => Body::Moon,
        ImageKind::FullDisk => {
            // Darker at the limb and even inside: the Sun; as bright at the limb, with maria: the Moon.
            let sun = f.limb.is_some_and(|l| l < 0.85) && f.patches.is_none_or(|p| p < 0.08);
            if sun { Body::Sun } else { Body::Moon }
        }
        _ => {
            // Relief lit from one side, strong contrast, marked texture: two of the three make
            // the Moon (on 68 lunar and 8 solar close-ups, 67 and 8 right). Light from no
            // direction at all is never the Moon's relief.
            let votes = (f.anisotropy > LIGHT_FROM_ONE_SIDE) as u8 + (f.contrast > LUNAR_CONTRAST) as u8 + (f.texture > LUNAR_TEXTURE) as u8;
            if votes >= 2 && f.anisotropy > NO_LIGHT_DIRECTION { Body::Moon } else { Body::Sun }
        }
    }
}

/// [`Features::anisotropy`] above which the light comes from one side.
const LIGHT_FROM_ONE_SIDE: f32 = 0.1;
/// Below it the light has no direction at all.
const NO_LIGHT_DIRECTION: f32 = 0.04;
/// [`Features::contrast`] of a lunar close-up (solar ones: 0.26–0.34, a few up to 0.57).
const LUNAR_CONTRAST: f32 = 0.5;
/// [`Features::texture`] of a lunar close-up (solar ones: 0.02–0.04, a few up to 0.14).
const LUNAR_TEXTURE: f32 = 0.1;
