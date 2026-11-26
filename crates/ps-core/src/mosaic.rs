//! Mosaics: overlapping frames of the same subject taken at the same focal length (the Moon,
//! the Sun, a large planet), joined into one image that is then processed like any other.
//!
//! - **Registration**, pair by pair: phase correlation of reduced copies, padded so frames that
//!   only partly overlap still give one unambiguous peak, checked by the normalised
//!   correlation of the overlap. Then, at full resolution, patches across the overlap are
//!   correlated one by one and a rotation + translation fitted to them (least squares, outliers
//!   dropped), a few times over: field rotation between frames is small but real.
//! - **Arrangement**: the best-overlapping pairs form a tree from the reference frame (the
//!   first); each frame's placement is its parent's composed with their relative placement.
//! - **Brightness**: transparency and exposure change between frames, so each gets per-channel
//!   gains from the overlap with the frames already placed, then a gentle brightness *plane*
//!   (vignetting and gradients differ from frame to frame, a gain alone leaves a step).
//! - **Edges**: each frame's dirty border (zero rows, darkened rim of the stack) is trimmed.
//! - **Blending**: multiband. Each frame is split in three frequency bands; the coarse band is
//!   mixed over the whole feather width, the fine band over a few pixels, so brightness
//!   differences fade slowly while detail changes over a short span and never shows a line.
//!   The sharper frame only wins in the fine band. Every channel uses the same weights, so
//!   colours (Mineral Moon tints) are not mixed up along the join.
//!
//! The frames and their placements are kept ([`Mosaic`]): the mosaic can be rebuilt with a frame
//! left out or nudged, and the processing on top of it stays as it is.

use crate::Image;
use crate::color::{ColorMode, working_planes};
use crate::fft::{FftConvolver, next_fast_size};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Where a frame sits in the mosaic: frame point p goes to R(angle)·p + (x, y).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub x: f32,
    pub y: f32,
    /// Radians.
    pub angle: f32,
}

impl Placement {
    pub const IDENTITY: Placement = Placement { x: 0.0, y: 0.0, angle: 0.0 };

    pub fn apply(&self, [px, py]: [f32; 2]) -> [f32; 2] {
        let (s, c) = self.angle.sin_cos();
        [c * px - s * py + self.x, s * px + c * py + self.y]
    }

    pub fn invert(&self, [qx, qy]: [f32; 2]) -> [f32; 2] {
        let (s, c) = self.angle.sin_cos();
        let (dx, dy) = (qx - self.x, qy - self.y);
        [c * dx + s * dy, -s * dx + c * dy]
    }

    /// `self` after `inner`: a point of `inner`'s frame placed by `inner`, then by `self`.
    pub fn then(&self, inner: &Placement) -> Placement {
        let [x, y] = self.apply([inner.x, inner.y]);
        Placement { x, y, angle: self.angle + inner.angle }
    }
}

/// One frame of a mosaic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MosaicTile {
    pub path: PathBuf,
    pub enabled: bool,
    /// `None` until registered, or when it overlaps no other frame.
    pub placement: Option<Placement>,
    /// Per-channel brightness factors.
    pub gain: [f32; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MosaicParams {
    /// Width (px) over which a frame fades out towards its border.
    pub feather: f32,
    /// Even out the frames' brightness.
    pub match_brightness: bool,
    /// Where frames overlap, let the sharper one count more.
    pub prefer_sharp: bool,
}

impl Default for MosaicParams {
    fn default() -> Self {
        Self { feather: 80.0, match_brightness: true, prefer_sharp: true }
    }
}

/// A mosaic as saved: its frames (by file) and how they are joined.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Mosaic {
    pub tiles: Vec<MosaicTile>,
    pub params: MosaicParams,
}

impl Mosaic {
    pub fn new(paths: Vec<PathBuf>) -> Self {
        Self { tiles: paths.into_iter().map(|path| MosaicTile { path, enabled: true, placement: None, gain: [1.0; 3] }).collect(), params: MosaicParams::default() }
    }
}

/// How two frames overlap: the placement of the second in the first's coordinates.
#[derive(Clone, Copy, Debug)]
pub struct PairMatch {
    pub placement: Placement,
    /// Normalised correlation of the overlap (‥1).
    pub score: f32,
}

/// Lowest normalised correlation of a pair's overlap to count as overlapping.
const MIN_SCORE: f32 = 0.5;
/// Smallest overlap, as a fraction of the smaller frame.
const MIN_OVERLAP: f32 = 0.03;
/// Side of the reduced copies used to find the overlap.
const COARSE_SIDE: usize = 640;
/// Border of a frame left out (stacking software leaves ragged edges).
const INSET: f32 = 4.0;

/// A luminance plane with its size.
pub struct Plane {
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

impl Plane {
    pub fn luminance(img: &Image) -> Plane {
        Plane { data: working_planes(img, ColorMode::Luminance).swap_remove(0), w: img.width(), h: img.height() }
    }

    fn reduced(&self, f: usize) -> Plane {
        let data = crate::image::downscale_plane(&self.data, self.w, self.h, f);
        Plane { data, w: (self.w / f).max(1), h: (self.h / f).max(1) }
    }

    /// Without its smooth part: what is left after subtracting a blurred copy.
    fn detail(&self) -> Plane {
        let smooth = crate::mask::gaussian_approx(&self.data, self.w, self.h, 3.0);
        Plane { data: self.data.iter().zip(&smooth).map(|(v, s)| v - s).collect(), w: self.w, h: self.h }
    }

    /// Without slow brightness variation (vignetting, haze): the plane minus a wide blur,
    /// where the subject is (the dark sky stays flat so the outline keeps its edge).
    fn flattened(&self) -> Plane {
        let sigma = (self.w.min(self.h) as f32 / 12.0).max(4.0);
        let smooth = crate::mask::gaussian_approx(&self.data, self.w, self.h, sigma);
        Plane { data: self.data.iter().zip(&smooth).map(|(v, s)| v - 0.9 * s).collect(), w: self.w, h: self.h }
    }

    /// Bilinear sample; `None` outside.
    fn sample(&self, x: f32, y: f32) -> Option<f32> {
        let (x, y) = (x - 0.5, y - 0.5);
        if x < 0.0 || y < 0.0 || x > (self.w - 1) as f32 || y > (self.h - 1) as f32 {
            return None;
        }
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        let at = |x: usize, y: usize| self.data[y * self.w + x];
        let top = at(x0, y0) + tx * (at(x1, y0) - at(x0, y0));
        let bottom = at(x0, y1) + tx * (at(x1, y1) - at(x0, y1));
        Some(top + ty * (bottom - top))
    }
}

/// Where `b` sits on `a` (both luminance, same scale), or `None` if they don't overlap.
pub fn match_pair(a: &Plane, b: &Plane) -> Option<PairMatch> {
    let f = a.w.max(a.h).max(b.w).max(b.h).div_ceil(COARSE_SIDE).max(1);
    // Vignetting and haze differ between frames: the coarse search sees them flattened.
    let (sa, sb) = (a.reduced(f).flattened(), b.reduced(f).flattened());
    let coarse = coarse_offset(&sa, &sb)?;
    let mut placement = Placement { x: coarse.x * f as f32, y: coarse.y * f as f32, angle: coarse.angle };
    for _ in 0..4 {
        placement = refine(a, b, placement).unwrap_or(placement);
    }
    // Checked on the fine detail only: large smooth shapes (the limb, a mare) correlate
    // passably even where frames don't match.
    let g = (f / 2).max(1);
    let (da, db) = (a.reduced(g).detail(), b.reduced(g).detail());
    let scaled = Placement { x: placement.x / g as f32, y: placement.y / g as f32, angle: placement.angle };
    let score = overlap_correlation(&da, &db, &scaled, 1)?;
    (score >= MIN_SCORE).then_some(PairMatch { placement, score })
}

/// Placement of `b` on `a` (reduced copies): the best of the strongest phase-correlation
/// peaks, with `b` turned by a few small angles (field rotation between frames blurs the peak).
fn coarse_offset(a: &Plane, b: &Plane) -> Option<Placement> {
    let (pw, ph) = (next_fast_size(a.w + b.w), next_fast_size(a.h + b.h));
    let plan = FftConvolver::plan(pw, ph);
    let fa = plan.forward(&padded(a, pw, ph));
    let centre = [b.w as f32 / 2.0, b.h as f32 / 2.0];
    let mut best: Option<(Placement, f32)> = None;
    let readings = |p: usize, n: usize, lo: f32, hi: f32| -> Vec<f32> {
        let (p, n) = (p as f32, n as f32);
        [p, p - n, -p, n - p].into_iter().filter(|&v| v > lo && v < hi).collect()
    };
    for degrees in [0.0f32, -1.0, 1.0, -2.0, 2.0] {
        let angle = degrees.to_radians();
        // `b` turned about its centre: u = R(p − c) + c.
        let turn = Placement { x: 0.0, y: 0.0, angle };
        let turned = if degrees == 0.0 {
            Plane { data: b.data.clone(), w: b.w, h: b.h }
        } else {
            let mean = b.data.iter().sum::<f32>() / b.data.len() as f32;
            let data = (0..b.w * b.h)
                .map(|i| {
                    let u = [(i % b.w) as f32 + 0.5 - centre[0], (i / b.w) as f32 + 0.5 - centre[1]];
                    let [x, y] = turn.invert(u);
                    b.sample(x + centre[0], y + centre[1]).unwrap_or(mean)
                })
                .collect();
            Plane { data, w: b.w, h: b.h }
        };
        let fb = plan.forward(&padded(&turned, pw, ph));
        // Pure phase correlation gives the sharpest peaks for detail, a partial one copes better
        // with a narrow overlap: peaks of both are tried.
        for power in [1.0f32, 0.5] {
            let cross = fb
                .par_iter()
                .zip(&fa)
                .map(|(m, r)| {
                    let c = r * m.conj();
                    let mag = c.norm();
                    if mag > 1e-12 { c / mag.powf(power) } else { Default::default() }
                })
                .collect();
            let mut surface = plan.inverse(cross);
            // A peak at p means an offset of ±p modulo the padded size; every reading that keeps
            // the frames overlapping is checked on the overlap itself, for a few peaks (repeated
            // texture, the limb).
            for _ in 0..8 {
                let (peak, _) = surface.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m });
                let (px, py) = (peak % pw, peak / pw);
                for tx in readings(px, pw, -(b.w as f32), a.w as f32) {
                    for ty in readings(py, ph, -(b.h as f32), a.h as f32) {
                        // Frame point p lands at t + R(p − c) + c.
                        let [rx, ry] = turn.apply(centre);
                        let candidate = Placement { x: tx + centre[0] - rx, y: ty + centre[1] - ry, angle };
                        if let Some(score) = overlap_correlation(a, b, &candidate, 2)
                            && best.is_none_or(|(_, s)| score > s)
                        {
                            best = Some((candidate, score));
                        }
                    }
                }
                for dy in -8i64..=8 {
                    for dx in -8i64..=8 {
                        let (x, y) = ((px as i64 + dx).rem_euclid(pw as i64) as usize, (py as i64 + dy).rem_euclid(ph as i64) as usize);
                        surface[y * pw + x] = f32::MIN;
                    }
                }
            }
        }
    }
    best.map(|(p, _)| p)
}

/// `p` tapered towards its border, its weighted mean removed (otherwise the frames' outlines
/// themselves correlate, best at no offset at all), in a `pw`×`ph` zero canvas.
fn padded(p: &Plane, pw: usize, ph: usize) -> Vec<f32> {
    let ramp = |i: usize, n: usize| {
        let edge = (n as f32 * 0.08).max(4.0);
        (i.min(n - 1 - i) as f32 / edge).min(1.0)
    };
    let weight = |x: usize, y: usize| ramp(x, p.w) * ramp(y, p.h);
    let (mut sum, mut total) = (0.0f64, 0.0f64);
    for y in 0..p.h {
        for x in 0..p.w {
            (sum, total) = (sum + (weight(x, y) * p.data[y * p.w + x]) as f64, total + weight(x, y) as f64);
        }
    }
    let mean = (sum / total.max(1e-9)) as f32;
    let mut out = vec![0.0f32; pw * ph];
    for y in 0..p.h {
        for x in 0..p.w {
            out[y * pw + x] = (p.data[y * p.w + x] - mean) * weight(x, y);
        }
    }
    out
}

/// Normalised correlation of `a` and `b` placed on it, over their overlap (every `step` px of
/// `a`); `None` if the overlap is too small.
fn overlap_correlation(a: &Plane, b: &Plane, placement: &Placement, step: usize) -> Option<f32> {
    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in (0..a.h).step_by(step) {
        for x in (0..a.w).step_by(step) {
            let q = [x as f32 + 0.5, y as f32 + 0.5];
            let [bx, by] = placement.invert(q);
            if let Some(vb) = b.sample(bx, by) {
                let (va, vb) = (a.data[y * a.w + x] as f64, vb as f64);
                (n, sa, sb) = (n + 1.0, sa + va, sb + vb);
                (saa, sbb, sab) = (saa + va * va, sbb + vb * vb, sab + va * vb);
            }
        }
    }
    let smaller = (a.w * a.h).min(b.w * b.h) as f64 / (step * step) as f64;
    if n < MIN_OVERLAP as f64 * smaller || n < 64.0 {
        return None;
    }
    let cov = sab / n - (sa / n) * (sb / n);
    let (va, vb) = (saa / n - (sa / n).powi(2), sbb / n - (sb / n).powi(2));
    (va > 1e-12 && vb > 1e-12).then(|| (cov / (va * vb).sqrt()) as f32)
}

/// Better placement of `b` on `a` from patches across the overlap, each correlated after
/// resampling `b` onto `a` with the current placement; a rotation and translation is fitted
/// to them. `None` when too few patches have detail.
fn refine(a: &Plane, b: &Plane, current: Placement) -> Option<Placement> {
    // The overlap's bounding box in `a`.
    let corners = [[0.0, 0.0], [b.w as f32, 0.0], [0.0, b.h as f32], [b.w as f32, b.h as f32]].map(|p| current.apply(p));
    let (x0, x1) = (corners.iter().map(|c| c[0]).fold(f32::MAX, f32::min).max(0.0), corners.iter().map(|c| c[0]).fold(f32::MIN, f32::max).min(a.w as f32));
    let (y0, y1) = (corners.iter().map(|c| c[1]).fold(f32::MAX, f32::min).max(0.0), corners.iter().map(|c| c[1]).fold(f32::MIN, f32::max).min(a.h as f32));
    // Patches as large as a narrow overlap allows (at least 32 px), at most 256.
    let narrow = (x1 - x0).min(y1 - y0);
    let side = ((narrow / 2.5).max((narrow * 0.8).min(64.0)) as usize).min(256) & !1;
    if side < 32 {
        return None;
    }
    let plan = FftConvolver::plan(side, side);
    let half = side as f32 / 2.0;
    let centres: Vec<[f32; 2]> =
        (0..4).flat_map(|j| (0..4).map(move |i| (i, j))).map(|(i, j)| [x0 + half + (x1 - x0 - side as f32) * i as f32 / 3.0, y0 + half + (y1 - y0 - side as f32) * j as f32 / 3.0]).collect();
    let pairs: Vec<([f32; 2], [f32; 2])> = centres
        .par_iter()
        .filter_map(|&c| {
            let mut pa = Vec::with_capacity(side * side);
            let mut pb = Vec::with_capacity(side * side);
            for v in 0..side {
                for u in 0..side {
                    let q = [c[0] - half + u as f32 + 0.5, c[1] - half + v as f32 + 0.5];
                    pa.push(a.sample(q[0], q[1])?);
                    let [bx, by] = current.invert(q);
                    pb.push(b.sample(bx, by)?);
                }
            }
            // Skip featureless patches (sky, a flat mare).
            let mean = pa.iter().sum::<f32>() / pa.len() as f32;
            let var = pa.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / pa.len() as f32;
            if var.sqrt() < 0.01 {
                return None;
            }
            let [dx, dy] = crate::align::phase_correlate(&plan, &crate::align::taper(&pa, side, side), &crate::align::taper(&pb, side, side), side);
            // a(c) shows what b-on-a shows at c − d.
            Some((current.invert([c[0] - dx, c[1] - dy]), c))
        })
        .collect();
    if pairs.len() < 3 {
        return None;
    }
    let mut fit = rigid_fit(&pairs)?;
    // Drop patches that disagree (moving shadows near the terminator, noise) and fit again.
    let residual = |p: &([f32; 2], [f32; 2]), t: &Placement| {
        let q = t.apply(p.0);
        (q[0] - p.1[0]).hypot(q[1] - p.1[1])
    };
    let kept: Vec<_> = pairs.iter().copied().filter(|p| residual(p, &fit) < 1.5).collect();
    if kept.len() >= 3 && kept.len() < pairs.len() {
        fit = rigid_fit(&kept)?;
    }
    Some(fit)
}

/// Rotation and translation taking the first points onto the second (least squares).
fn rigid_fit(pairs: &[([f32; 2], [f32; 2])]) -> Option<Placement> {
    let n = pairs.len() as f64;
    let (mut px, mut py, mut qx, mut qy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (p, q) in pairs {
        (px, py, qx, qy) = (px + p[0] as f64, py + p[1] as f64, qx + q[0] as f64, qy + q[1] as f64);
    }
    let (px, py, qx, qy) = (px / n, py / n, qx / n, qy / n);
    let (mut sxx, mut sxy) = (0.0f64, 0.0f64);
    for (p, q) in pairs {
        let (ax, ay, bx, by) = (p[0] as f64 - px, p[1] as f64 - py, q[0] as f64 - qx, q[1] as f64 - qy);
        sxx += ax * bx + ay * by;
        sxy += ax * by - ay * bx;
    }
    if sxx.abs() + sxy.abs() < 1e-9 {
        return None;
    }
    let angle = sxy.atan2(sxx);
    let (s, c) = angle.sin_cos();
    Some(Placement { x: (qx - (c * px - s * py)) as f32, y: (qy - (s * px + c * py)) as f32, angle: angle as f32 })
}

/// Placements of all frames relative to the first enabled one; `None` for those that overlap
/// none of the others. `report(done, total)` follows the pairs.
pub fn arrange(planes: &[Option<&Plane>], report: &(dyn Fn(usize, usize) + Sync)) -> Vec<Option<Placement>> {
    let enabled: Vec<bool> = planes.iter().map(Option::is_some).collect();
    tree(&enabled, &match_all(planes, report))
}

/// Every overlapping pair (first index lower) with the second frame's placement on the first.
pub fn match_all(planes: &[Option<&Plane>], report: &(dyn Fn(usize, usize) + Sync)) -> Vec<((usize, usize), PairMatch)> {
    match_from(planes, 0, report)
}

/// Like `match_all`, but only the pairs whose second frame is `from` or later: the overlaps of
/// frames just added, with each other and with those already there.
pub fn match_from(planes: &[Option<&Plane>], from: usize, report: &(dyn Fn(usize, usize) + Sync)) -> Vec<((usize, usize), PairMatch)> {
    let n = planes.len();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (from.max(i + 1)..n).map(move |j| (i, j))).filter(|&(i, j)| planes[i].is_some() && planes[j].is_some()).collect();
    let done = std::sync::atomic::AtomicUsize::new(0);
    pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let m = match_pair(planes[i]?, planes[j]?);
            report(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1, pairs.len());
            m.map(|m| ((i, j), m))
        })
        .collect()
}

/// Placements of the `enabled` frames from their pairwise `matches`: the best overlaps form a
/// tree from the first enabled frame (maximum spanning tree, Prim).
pub fn tree(enabled: &[bool], matches: &[((usize, usize), PairMatch)]) -> Vec<Option<Placement>> {
    extend(enabled, matches, vec![None; enabled.len()])
}

/// Places the enabled frames still without a placement from those that have one, following
/// the best overlaps; frames already placed stay where they are. With none placed, the first
/// enabled frame is the reference.
pub fn extend(enabled: &[bool], matches: &[((usize, usize), PairMatch)], mut placed: Vec<Option<Placement>>) -> Vec<Option<Placement>> {
    if !placed.iter().zip(enabled).any(|(p, &e)| e && p.is_some()) {
        let Some(root) = enabled.iter().position(|&e| e) else {
            return placed;
        };
        placed[root] = Some(Placement::IDENTITY);
    }
    loop {
        let next = matches
            .iter()
            .filter(|((i, j), _)| enabled[*i] && enabled[*j])
            .filter_map(|((i, j), m)| match (placed[*i], placed[*j]) {
                (Some(pi), None) => Some((*j, pi.then(&m.placement), m.score)),
                (None, Some(pj)) => Some((*i, pj.then(&inverse(&m.placement)), m.score)),
                _ => None,
            })
            .max_by(|a, b| a.2.total_cmp(&b.2));
        match next {
            Some((k, p, _)) => placed[k] = Some(p),
            None => break,
        }
    }
    placed
}

fn inverse(p: &Placement) -> Placement {
    let [x, y] = Placement { x: 0.0, y: 0.0, angle: -p.angle }.apply([-p.x, -p.y]);
    Placement { x, y, angle: -p.angle }
}

/// Per-channel gains evening out the frames' brightness: each frame against all the frames
/// already evened out where they overlap, from the reference outwards.
pub fn brightness(images: &[Option<&Image>], placements: &[Option<Placement>]) -> Vec<[f32; 3]> {
    let n = images.len();
    let mut gains = vec![[1.0f32; 3]; n];
    let order: Vec<usize> = (0..n).filter(|&i| images[i].is_some() && placements[i].is_some()).collect();
    let margins: Vec<[f32; 4]> = images.iter().map(|i| i.map_or([INSET; 4], margins)).collect();
    for (k, &i) in order.iter().enumerate().skip(1) {
        let (Some(img), Some(pi)) = (images[i], placements[i]) else {
            continue;
        };
        let (mut own, mut others) = ([0.0f64; 3], [0.0f64; 3]);
        let step = (img.width().max(img.height()) / 300).max(1);
        for y in (0..img.height()).step_by(step) {
            for x in (0..img.width()).step_by(step) {
                let q = pi.apply([x as f32 + 0.5, y as f32 + 0.5]);
                let v = pixel(img, x, y);
                // Bright enough to measure (not the sky), not near saturation.
                if crate::color::luma(v) < 0.08 || v.iter().any(|&c| c > 0.97) {
                    continue;
                }
                for &j in &order[..k] {
                    let (Some(other), Some(pj)) = (images[j], placements[j]) else {
                        continue;
                    };
                    let [ox, oy] = pj.invert(q);
                    if edge_distance(&margins[j], other.width() as f32, other.height() as f32, ox, oy) <= 0.0 {
                        continue;
                    }
                    let o = pixel(other, ox as usize, oy as usize);
                    for c in 0..3 {
                        own[c] += v[c] as f64;
                        others[c] += (o[c] * gains[j][c]) as f64;
                    }
                }
            }
        }
        if own[1] > 0.0 {
            gains[i] = std::array::from_fn(|c| if own[c] > 1e-9 { (others[c] / own[c]).clamp(0.25, 4.0) as f32 } else { 1.0 });
        }
    }
    gains
}

fn pixel(img: &Image, x: usize, y: usize) -> [f32; 3] {
    let i = y * img.width() + x;
    match img.channels() {
        [r, g, b] => [r[i], g[i], b[i]],
        [m] => [m[i]; 3],
        _ => [0.0; 3],
    }
}

/// Bounding box (x, y, w, h) of the placed frames, in the reference frame's coordinates.
pub fn bounds(sizes: &[(usize, usize)], placements: &[Option<Placement>]) -> Option<(f32, f32, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (&(w, h), p) in sizes.iter().zip(placements) {
        let Some(p) = p else { continue };
        for c in [[0.0, 0.0], [w as f32, 0.0], [0.0, h as f32], [w as f32, h as f32]] {
            let [x, y] = p.apply(c);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
    }
    (x1 > x0).then(|| (x0.floor(), y0.floor(), (x1.ceil() - x0.floor()) as usize, (y1.ceil() - y0.floor()) as usize))
}

/// Distance (px) from (x, y) to the border of the usable part of a `w`×`h` frame; ≤ 0 outside.
fn edge_distance(m: &[f32; 4], w: f32, h: f32, x: f32, y: f32) -> f32 {
    (x - m[0]).min(y - m[1]).min(w - m[2] - x).min(h - m[3] - y)
}

/// Margins (left, top, right, bottom) of a frame to leave out: [`INSET`] plus any dirty rim of
/// the stack: rows/columns of exact zeros, or markedly darker than the ones a few pixels in.
pub fn margins(img: &Image) -> [f32; 4] {
    let (w, h) = (img.width(), img.height());
    let lum = Plane::luminance(img);
    // Mean of the stripe `k` pixels in from `side` (0 left, 1 top, 2 right, 3 bottom); and whether it is all zero.
    let stripe = |side: usize, k: usize| -> Option<(f32, bool)> {
        let (len, depth) = if side % 2 == 0 { (h, w) } else { (w, h) };
        if k >= depth {
            return None;
        }
        let (mut sum, mut zero) = (0.0f64, true);
        for i in 0..len {
            let (x, y) = match side {
                0 => (k, i),
                1 => (i, k),
                2 => (w - 1 - k, i),
                _ => (i, h - 1 - k),
            };
            let v = lum.data[y * w + x];
            sum += v as f64;
            zero &= v.abs() < 1e-6;
        }
        Some(((sum / len as f64) as f32, zero))
    };
    std::array::from_fn(|side| {
        let mut k = 0;
        while k < TRIM_CAP {
            let Some((mean, zero)) = stripe(side, k) else { break };
            let inner = stripe(side, k + TRIM_LOOK).map(|(m, _)| m).unwrap_or(0.0);
            if zero || (inner > 0.1 && mean < 0.75 * inner) {
                k += 1;
            } else {
                break;
            }
        }
        INSET + k as f32
    })
}

/// Largest dirty rim trimmed from a frame (px).
const TRIM_CAP: usize = 48;
/// How far inside a rim stripe is compared with.
const TRIM_LOOK: usize = 10;

/// Brightness plane of one frame: factor = a + b·u + c·v per channel, u, v ∈ [-1, 1] across the frame.
type Shade = [[f32; 3]; 3];
const NO_SHADE: Shade = [[1.0; 3], [0.0; 3], [0.0; 3]];

fn shade_at(s: &Shade, c: usize, u: f32, v: f32) -> f32 {
    (s[0][c] + s[1][c] * u + s[2][c] * v).clamp(0.7, 1.4)
}

struct Tile<'a> {
    img: &'a Image,
    place: Placement,
    gain: [f32; 3],
    quality: f32,
    margin: [f32; 4],
    shade: Shade,
}

impl Tile<'_> {
    fn size(&self) -> (f32, f32) {
        (self.img.width() as f32, self.img.height() as f32)
    }

    /// Frame coordinates of canvas point `q`, and the distance to the usable border (≤ 0 outside).
    fn locate(&self, q: [f32; 2]) -> ([f32; 2], f32) {
        let t = self.place.invert(q);
        let (tw, th) = self.size();
        (t, edge_distance(&self.margin, tw, th, t[0], t[1]))
    }

    /// Corrected value of every channel at frame point `t`.
    fn value(&self, t: [f32; 2]) -> [f32; 3] {
        let (tw, th) = self.size();
        let (u, v) = (2.0 * t[0] / tw - 1.0, 2.0 * t[1] / th - 1.0);
        let raw = bilinear(self.img, t[0], t[1]);
        let mono = self.img.channels().len() != 3;
        std::array::from_fn(|c| raw[if mono { 0 } else { c }] * self.gain[c] * shade_at(&self.shade, c, u, v))
    }
}

/// Smooth planes absorbing what a constant gain cannot: each frame, from the second, is fitted
/// to the frames before it on their overlap (block means, so detail cancels), with a plane
/// a + b·u + c·v per channel. The tilt is shrunk towards zero where the overlap says little
/// about it (a thin strip), and the factor is bounded, so it never extrapolates wildly.
fn fit_shading(tiles: &mut [Tile]) {
    const CELLS: usize = 24;
    for k in 1..tiles.len() {
        let (before, rest) = tiles.split_at_mut(k);
        let me = &mut rest[0];
        let (tw, th) = me.size();
        let step = (tw.max(th) / 320.0).ceil().max(1.0) as usize;
        // Per cell: sums of own and other values per channel, and the sample count.
        let mut cells = vec![([0.0f64; 3], [0.0f64; 3], 0u32); CELLS * CELLS];
        let mut y = 0;
        while (y as f32) < th {
            let mut x = 0;
            while (x as f32) < tw {
                let t = [x as f32 + 0.5, y as f32 + 0.5];
                let q = me.place.apply(t);
                let own = bilinear(me.img, t[0], t[1]);
                let mono = me.img.channels().len() != 3;
                let own: [f32; 3] = std::array::from_fn(|c| own[if mono { 0 } else { c }] * me.gain[c]);
                // Measurable: on the subject, away from saturation, inside the usable area.
                if me.locate(q).1 > 0.0 && crate::color::luma(own) > 0.08 && own.iter().all(|&c| c < 0.97) {
                    for other in before.iter() {
                        let (to, d) = other.locate(q);
                        if d <= 0.0 {
                            continue;
                        }
                        let o = other.value(to);
                        let cell = &mut cells[((y as f32 / th * CELLS as f32) as usize).min(CELLS - 1) * CELLS + ((x as f32 / tw * CELLS as f32) as usize).min(CELLS - 1)];
                        for c in 0..3 {
                            cell.0[c] += own[c] as f64;
                            cell.1[c] += o[c] as f64;
                        }
                        cell.2 += 1;
                    }
                }
                x += step;
            }
            y += step;
        }
        let mut shade = NO_SHADE;
        for c in 0..3 {
            // Weighted least squares of ratio = a + b·u + c·v, ridge on the tilt.
            let (mut m, mut r, mut wsum, mut n) = ([[0.0f64; 3]; 3], [0.0f64; 3], 0.0f64, 0);
            for (i, cell) in cells.iter().enumerate() {
                if cell.2 < 12 || cell.0[c] < 1e-6 {
                    continue;
                }
                let ratio = cell.1[c] / cell.0[c];
                let (u, v) = (((i % CELLS) as f64 + 0.5) / CELLS as f64 * 2.0 - 1.0, ((i / CELLS) as f64 + 0.5) / CELLS as f64 * 2.0 - 1.0);
                let wgt = cell.0[c];
                let basis = [1.0, u, v];
                for a in 0..3 {
                    for b in 0..3 {
                        m[a][b] += wgt * basis[a] * basis[b];
                    }
                    r[a] += wgt * basis[a] * ratio;
                }
                wsum += wgt;
                n += 1;
            }
            if n < 6 {
                continue;
            }
            for a in 1..3 {
                m[a][a] += 0.15 * wsum;
            }
            if let Some(sol) = solve3(m, r) {
                shade[0][c] = sol[0] as f32;
                shade[1][c] = sol[1] as f32;
                shade[2][c] = sol[2] as f32;
            }
        }
        me.shade = shade;
    }
}

fn solve3(mut m: [[f64; 3]; 3], mut r: [f64; 3]) -> Option<[f64; 3]> {
    for i in 0..3 {
        let p = (i..3).max_by(|&a, &b| m[a][i].abs().total_cmp(&m[b][i].abs()))?;
        if m[p][i].abs() < 1e-12 {
            return None;
        }
        m.swap(i, p);
        r.swap(i, p);
        for j in i + 1..3 {
            let f = m[j][i] / m[i][i];
            for k in i..3 {
                m[j][k] -= f * m[i][k];
            }
            r[j] -= f * r[i];
        }
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        x[i] = (r[i] - (i + 1..3).map(|k| m[i][k] * x[k]).sum::<f64>()) / m[i][i];
    }
    Some(x)
}

/// Box blur (running sums, edges treated as empty) of a plane, along rows.
fn box_rows(data: &mut [f32], w: usize, r: usize) {
    data.par_chunks_mut(w).for_each(|row| {
        let src = row.to_vec();
        let mut sum = src[..r.min(w - 1) + 1].iter().map(|&v| v as f64).sum::<f64>() - 0.0;
        for x in 0..w {
            row[x] = (sum / (2 * r + 1) as f64) as f32;
            if x + r + 1 < w {
                sum += src[x + r + 1] as f64;
            }
            if x >= r {
                sum -= src[x - r] as f64;
            }
        }
    });
}

fn transpose(data: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0; w * h];
    out.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
        for (y, v) in col.iter_mut().enumerate() {
            *v = data[y * w + x];
        }
    });
    out
}

/// Gaussian-like blur (three box passes each way) of a `w`×`h` plane.
fn blur(data: &mut Vec<f32>, w: usize, h: usize, sigma: f32) {
    if sigma < 0.7 {
        return;
    }
    let r = (((4.0 * sigma * sigma + 1.0).sqrt() - 1.0) / 2.0).round().max(1.0) as usize;
    for _ in 0..3 {
        box_rows(data, w, r);
    }
    let mut t = transpose(data, w, h);
    for _ in 0..3 {
        box_rows(&mut t, h, r);
    }
    *data = transpose(&t, h, w);
}

/// Blur of the valid part only: `data` is zero where `mask` is 0, and the blurred mask
/// renormalises it, so the invalid surround does not darken the border.
fn blur_masked(data: &[f32], mask: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let mut d = data.to_vec();
    let mut m = mask.to_vec();
    blur(&mut d, w, h, sigma);
    blur(&mut m, w, h, sigma);
    for (v, &wt) in d.iter_mut().zip(&m) {
        *v = if wt > 1e-4 { *v / wt } else { 0.0 };
    }
    d
}

/// Number of frequency bands blended separately: coarse, middle, fine.
const BANDS: usize = 3;

/// Weight of a frame at distance `d` from its border, for `band`. Falls from 1 deep inside to
/// ~0 at the border over a width that is `feather` for the coarse band and narrower for finer ones.
fn band_weight(d: f32, feather: f32, band: usize) -> f32 {
    let width = match band {
        0 => feather,
        1 => feather / 3.0,
        _ => (feather / 10.0).clamp(4.0, 24.0),
    };
    let t = (d / width).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t) + 1e-4
}

/// The frames joined: each placed, scaled by its gains and brightness plane, and mixed band by
/// band (see the module notes); `feather` px is the width of the coarse mix. `scale` < 1 renders a
/// reduced copy (for previews).
pub fn compose(images: &[Option<&Image>], placements: &[Option<Placement>], gains: &[[f32; 3]], params: &MosaicParams, scale: f32) -> Option<Image> {
    let used: Vec<usize> = (0..images.len()).filter(|&i| images[i].is_some() && placements[i].is_some()).collect();
    let sizes: Vec<(usize, usize)> = images.iter().map(|i| i.map_or((0, 0), |i| (i.width(), i.height()))).collect();
    let masked: Vec<Option<Placement>> = (0..images.len()).map(|i| if used.contains(&i) { placements[i] } else { None }).collect();
    let (ox, oy, w, h) = bounds(&sizes, &masked)?;
    let (w, h) = (((w as f32 * scale).ceil() as usize).max(1), ((h as f32 * scale).ceil() as usize).max(1));
    let colour = used.iter().any(|&i| images[i].is_some_and(|img| img.channels().len() == 3));
    let channels = if colour { 3 } else { 1 };
    let feather = params.feather.max(1.0);
    let sharpness: Vec<f32> = used.iter().map(|&i| if params.prefer_sharp { sharpness(images[i].expect("used")) } else { 1.0 }).collect();
    let sharpest = sharpness.iter().copied().fold(1e-9, f32::max);
    // Twice as sharp counts sixteen times as much (in the fine band): a soft frame only fills in.
    let mut tiles: Vec<Tile> = used
        .iter()
        .zip(&sharpness)
        .map(|(&i, &q)| {
            let img = images[i].expect("used");
            Tile { img, place: placements[i].expect("used"), gain: if params.match_brightness { gains[i] } else { [1.0; 3] }, quality: (q / sharpest).powi(4).max(1e-3), margin: margins(img), shade: NO_SHADE }
        })
        .collect();
    if params.match_brightness {
        fit_shading(&mut tiles);
    }
    let to_canvas = |x: f32, y: f32| [ox + (x + 0.5) / scale, oy + (y + 0.5) / scale];
    // Pass 1: total weight of every band at every canvas pixel.
    let mut wsum = vec![[0.0f32; BANDS]; w * h];
    for t in &tiles {
        wsum.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, acc) in row.iter_mut().enumerate() {
                let d = t.locate(to_canvas(x as f32, y as f32)).1;
                if d > 0.0 {
                    for (b, a) in acc.iter_mut().enumerate() {
                        *a += band_weight(d, feather, b) * if b == BANDS - 1 { t.quality } else { 1.0 };
                    }
                }
            }
        });
    }
    // Pass 2: each frame's bands, weighted, into the canvas.
    let mut planes = vec![vec![0.0f32; w * h]; channels];
    let sigma = [feather * scale / 6.0, (feather * scale / 24.0).max(1.0)];
    for t in &tiles {
        let (tw, th) = t.size();
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for c in [[0.0, 0.0], [tw, 0.0], [0.0, th], [tw, th]] {
            let [x, y] = t.place.apply(c);
            (x0, y0, x1, y1) = (x0.min((x - ox) * scale), y0.min((y - oy) * scale), x1.max((x - ox) * scale), y1.max((y - oy) * scale));
        }
        let (bx, by) = ((x0.floor().max(0.0)) as usize, (y0.floor().max(0.0)) as usize);
        let (ex, ey) = ((x1.ceil().max(0.0) as usize + 1).min(w), (y1.ceil().max(0.0) as usize + 1).min(h));
        if ex <= bx || ey <= by {
            continue;
        }
        let (bw, bh) = (ex - bx, ey - by);
        // The frame on its canvas patch: values (zero outside), validity mask, distance to border.
        let mut mask = vec![0.0f32; bw * bh];
        let mut dist = vec![0.0f32; bw * bh];
        let mut vals = vec![vec![0.0f32; bw * bh]; channels];
        {
            let rows: Vec<(Vec<f32>, Vec<f32>, Vec<[f32; 3]>)> = (0..bh)
                .into_par_iter()
                .map(|y| {
                    let (mut m, mut dd, mut vv) = (vec![0.0f32; bw], vec![0.0f32; bw], vec![[0.0f32; 3]; bw]);
                    for x in 0..bw {
                        let (tp, d) = t.locate(to_canvas((bx + x) as f32, (by + y) as f32));
                        if d > 0.0 {
                            (m[x], dd[x], vv[x]) = (1.0, d, t.value(tp));
                        }
                    }
                    (m, dd, vv)
                })
                .collect();
            for (y, (m, dd, vv)) in rows.into_iter().enumerate() {
                mask[y * bw..(y + 1) * bw].copy_from_slice(&m);
                dist[y * bw..(y + 1) * bw].copy_from_slice(&dd);
                for c in 0..channels {
                    for x in 0..bw {
                        vals[c][y * bw + x] = vv[x][c];
                    }
                }
            }
        }
        // Blurred copies (renormalised over the valid part): bands are I−G(fine), G(fine)−G(coarse), G(coarse).
        let weights_wsum = |i: usize| -> [f32; BANDS] {
            let (x, y) = (i % bw + bx, i / bw + by);
            let d = dist[i];
            let ws = wsum[y * w + x];
            std::array::from_fn(|b| if ws[b] > 0.0 { band_weight(d, feather, b) * if b == BANDS - 1 { t.quality } else { 1.0 } / ws[b] } else { 0.0 })
        };
        for c in 0..channels {
            let coarse = blur_masked(&vals[c], &mask, bw, bh, sigma[0]);
            let fine = blur_masked(&vals[c], &mask, bw, bh, sigma[1]);
            planes[c].par_chunks_mut(w).enumerate().skip(by).take(bh).for_each(|(y, row)| {
                for x in 0..bw {
                    let i = (y - by) * bw + x;
                    if mask[i] == 0.0 {
                        continue;
                    }
                    let wt = weights_wsum(i);
                    let bands = [coarse[i], fine[i] - coarse[i], vals[c][i] - fine[i]];
                    row[bx + x] += (0..BANDS).map(|b| bands[b] * wt[b]).sum::<f32>();
                }
            });
        }
    }
    let depth = tiles.iter().map(|t| t.img.source_bit_depth).max().unwrap_or(16);
    Some(Image::new(w, h, planes, depth))
}

/// How much fine detail a frame holds: the spread of its luminance minus a blurred copy, on
/// the subject (sky left out). Comparable between frames of the same scene.
pub fn sharpness(img: &Image) -> f32 {
    let f = img.width().max(img.height()).div_ceil(1600).max(1);
    let plane = Plane::luminance(img).reduced(f);
    let detail = plane.detail();
    let (mut n, mut sum) = (0.0f64, 0.0f64);
    for (d, v) in detail.data.iter().zip(&plane.data) {
        if *v > 0.08 {
            (n, sum) = (n + 1.0, sum + (*d as f64).powi(2));
        }
    }
    if n > 0.0 { (sum / n).sqrt() as f32 } else { 0.0 }
}

/// Bilinear sample of every channel at (x, y) (pixel centres at +0.5), edges clamped.
fn bilinear(img: &Image, x: f32, y: f32) -> [f32; 3] {
    let (w, h) = (img.width(), img.height());
    let mut out = [0.0f32; 3];
    for (o, plane) in out.iter_mut().zip(img.channels()) {
        *o = crate::image::bilinear_clamped(plane, w, h, x, y);
    }
    out
}
