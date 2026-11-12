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
//!   gains from the overlap with its parent.
//! - **Blending**: weights falling off smoothly towards each frame's border (the stacking
//!   software's ragged edges are left out), so no seam shows.
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
    let (sa, sb) = (a.reduced(f), b.reduced(f));
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
                    if ox < INSET || oy < INSET || ox >= other.width() as f32 - INSET || oy >= other.height() as f32 - INSET {
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

/// The frames joined: each placed, scaled by its gains, faded out towards its border over
/// `feather` px; where they overlap, the weighted average. `scale` < 1 renders a reduced copy
/// (for previews).
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
    // Twice as sharp counts sixteen times as much: a soft frame only fills in.
    let tiles: Vec<(&Image, Placement, [f32; 3], f32)> = used
        .iter()
        .zip(&sharpness)
        .map(|(&i, &q)| (images[i].expect("used"), placements[i].expect("used"), if params.match_brightness { gains[i] } else { [1.0; 3] }, (q / sharpest).powi(4).max(1e-3)))
        .collect();
    let mut planes = vec![vec![0.0f32; w * h]; channels];
    let rows: Vec<Vec<Vec<f32>>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![vec![0.0f32; w]; channels];
            let mut weight = vec![0.0f32; w];
            for (img, p, gain, quality) in &tiles {
                let (tw, th) = (img.width() as f32, img.height() as f32);
                for x in 0..w {
                    let q = [ox + (x as f32 + 0.5) / scale, oy + (y as f32 + 0.5) / scale];
                    let [tx, ty] = p.invert(q);
                    let d = (tx - INSET).min(ty - INSET).min(tw - INSET - tx).min(th - INSET - ty);
                    if d <= 0.0 {
                        continue;
                    }
                    let t = (d / feather).min(1.0);
                    let wgt = (t * t * (3.0 - 2.0 * t) + 1e-4) * quality;
                    let v = bilinear(img, tx, ty);
                    for c in 0..channels {
                        let src = if img.channels().len() == 3 { c } else { 0 };
                        row[c][x] += wgt * v[src] * gain[c];
                    }
                    weight[x] += wgt;
                }
            }
            for plane in &mut row {
                for (v, &wgt) in plane.iter_mut().zip(&weight) {
                    if wgt > 0.0 {
                        *v /= wgt;
                    }
                }
            }
            row
        })
        .collect();
    for (y, row) in rows.into_iter().enumerate() {
        for (c, values) in row.into_iter().enumerate() {
            planes[c][y * w..(y + 1) * w].copy_from_slice(&values);
        }
    }
    let depth = tiles.iter().map(|(img, ..)| img.source_bit_depth).max().unwrap_or(16);
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
    let (x, y) = ((x - 0.5).clamp(0.0, (w - 1) as f32), (y - 0.5).clamp(0.0, (h - 1) as f32));
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let mut out = [0.0f32; 3];
    for (c, plane) in img.channels().iter().enumerate().take(3) {
        let at = |x: usize, y: usize| plane[y * w + x];
        let top = at(x0, y0) + tx * (at(x1, y0) - at(x0, y0));
        let bottom = at(x0, y1) + tx * (at(x1, y1) - at(x0, y1));
        out[c] = top + ty * (bottom - top);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cratered "surface" with a limb, from which overlapping frames are cut.
    fn scene(w: usize, h: usize) -> Vec<f32> {
        let mut seed = 12345u32;
        let mut rand = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let craters: Vec<(f32, f32, f32, f32)> = (0..600).map(|_| (rand() * w as f32, rand() * h as f32, 3.0 + 25.0 * rand().powi(3), rand() - 0.5)).collect();
        let mut out = vec![0.45f32; w * h];
        for &(cx, cy, r, depth) in &craters {
            let (x0, x1) = ((cx - 2.0 * r).max(0.0) as usize, ((cx + 2.0 * r) as usize).min(w));
            let (y0, y1) = ((cy - 2.0 * r).max(0.0) as usize, ((cy + 2.0 * r) as usize).min(h));
            for y in y0..y1 {
                for x in x0..x1 {
                    let d = (x as f32 - cx).hypot(y as f32 - cy) / r;
                    out[y * w + x] += 0.15 * depth * (-d * d).exp();
                }
            }
        }
        for (i, v) in out.iter_mut().enumerate() {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            if (x - w as f32 * 0.6).hypot(y - h as f32 * 0.55) > w as f32 * 0.55 {
                *v = 0.0;
            }
        }
        out
    }

    /// Frame of `scene` (`sw` wide) whose pixel p shows scene point placement(p).
    fn cut(scene: &[f32], sw: usize, placement: Placement, w: usize, h: usize, gain: f32) -> Image {
        let src = Plane { data: scene.to_vec(), w: sw, h: scene.len() / sw };
        let data = (0..w * h)
            .map(|i| {
                let [x, y] = placement.apply([(i % w) as f32 + 0.5, (i / w) as f32 + 0.5]);
                gain * src.sample(x, y).unwrap_or(0.0)
            })
            .collect();
        Image::new(w, h, vec![data], 16)
    }

    #[test]
    fn finds_shift_and_small_rotation() {
        let (sw, sh) = (900, 700);
        let s = scene(sw, sh);
        let a_place = Placement { x: 50.0, y: 60.0, angle: 0.0 };
        let b_place = Placement { x: 380.0, y: 180.0, angle: 0.02 };
        let a = cut(&s, sw, a_place, 480, 400, 1.0);
        let b = cut(&s, sw, b_place, 480, 400, 1.0);
        let m = match_pair(&Plane::luminance(&a), &Plane::luminance(&b)).expect("they overlap");
        // b in a's frame: a⁻¹ ∘ b.
        let want = inverse(&a_place).then(&b_place);
        assert!((m.placement.x - want.x).abs() < 0.3 && (m.placement.y - want.y).abs() < 0.3, "{:?} vs {want:?}", m.placement);
        assert!((m.placement.angle - want.angle).abs() < 0.002, "{:?} vs {want:?}", m.placement);
        assert!(m.score > 0.9, "{}", m.score);
    }

    #[test]
    fn frames_that_do_not_overlap_are_left_out() {
        let (sw, sh) = (1400, 700);
        let s = scene(sw, sh);
        let a = cut(&s, sw, Placement { x: 10.0, y: 100.0, angle: 0.0 }, 400, 400, 1.0);
        let b = cut(&s, sw, Placement { x: 900.0, y: 100.0, angle: 0.0 }, 400, 400, 1.0);
        let c = cut(&s, sw, Placement { x: 300.0, y: 150.0, angle: 0.0 }, 400, 400, 1.0);
        let planes = [a, b, c].map(|i| Plane::luminance(&i));
        let placed = arrange(&[Some(&planes[0]), Some(&planes[1]), Some(&planes[2])], &|_, _| {});
        assert!(placed[0].is_some() && placed[2].is_some());
        assert!((placed[2].unwrap().x - 290.0).abs() < 0.5, "{:?}", placed[2]);
        assert!(placed[1].is_none(), "b overlaps neither: {:?}", placed[1]);
    }

    #[test]
    fn joins_without_seams_and_evens_out_brightness() {
        let (sw, sh) = (900, 600);
        let s = scene(sw, sh);
        let a = cut(&s, sw, Placement { x: 100.0, y: 100.0, angle: 0.0 }, 420, 380, 1.0);
        let b = cut(&s, sw, Placement { x: 380.0, y: 120.0, angle: 0.0 }, 420, 380, 0.8);
        let planes = [Plane::luminance(&a), Plane::luminance(&b)];
        let placed = arrange(&[Some(&planes[0]), Some(&planes[1])], &|_, _| {});
        let images = [Some(&a), Some(&b)];
        let gains = brightness(&images, &placed);
        assert!((gains[1][1] - 1.25).abs() < 0.03, "{:?}", gains[1]);
        let out = compose(&images, &placed, &gains, &MosaicParams::default(), 1.0).expect("a mosaic");
        assert!(out.width() >= 698 && out.width() <= 702, "{}", out.width());
        // Across the overlap, the mosaic matches the scene.
        let plane = &out.channels()[0];
        let mut worst = 0.0f32;
        for y in 50..330 {
            for x in 250..450 {
                let want = s[(y + 100) * sw + x + 100];
                if want > 0.0 {
                    worst = worst.max((plane[y * out.width() + x] - want).abs());
                }
            }
        }
        assert!(worst < 0.02, "{worst}");
    }
}
