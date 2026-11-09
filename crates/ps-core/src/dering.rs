//! Deringing: lifts the dark ring that diffraction or miscollimation leaves along a sharp edge —
//! just inside a planet's limb, in the sky around it, beside a crater's rim or shadow on the
//! Moon — and that sharpening makes plain.
//!
//! Finding such a ring on its own proved unreliable (belts, spots and the limb's own curve look
//! alike), so the user paints roughly over it with the brush, about as wide as it is. Under the
//! painted area:
//! - **how wide** the ring is: a morphological *closing* (a max filter, then a min filter of the
//!   same size) fills every dark valley narrower than its window up to the lower of its two
//!   sides, and leaves alone what is not a valley: a step edge such as the limb, a broad belt, a
//!   convex disk. Growing the window, the fill on the ring grows until the whole valley is filled,
//!   then stops: the smallest window reaching almost the full fill is the ring's width (never
//!   more than the painted band allows);
//! - **where** it is: where that fill stands clearly out of the noise, softly;
//! - **the correction**: the fill, computed on a lightly smoothed copy so the noise's own dips
//!   are not lifted, is added to the image: the ring rises to its surroundings and the grain
//!   stays. Each colour channel is filled on its own.
//!
//! [`weight`] is a separate, automatic area along sharp edges: where deconvolution's
//! anti-ringing holds its estimate within the observed image's range.

use crate::Image;
use crate::color::{ColorMode, working_planes};
use crate::mask::{self, Ellipse, Stroke, foreground, main_body};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeringParams {
    /// 0 = off, 1 = the ring fully lifted.
    pub strength: f32,
    /// Where the user painted over the ring.
    pub strokes: Vec<Stroke>,
    /// The ring's width (px); 0 = measured under the brush.
    pub width: f32,
    /// Width of the halos (px) of deconvolution's anti-ringing area ([`weight`]).
    pub radius: u32,
    /// How sharp an edge must be to enter that area: its local range as a fraction of the
    /// subject's brightness.
    pub edges: f32,
    /// Where that area lies (on Saturn, the globe or the rings).
    pub region: DeringRegion,
}

/// Where deconvolution's anti-ringing area lies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeringRegion {
    /// Every sharp edge of the image (on a whole planet, a band along the limb).
    #[default]
    Subject,
    /// Saturn's globe (with the rings crossing it) and its limb.
    Globe,
    /// Saturn's rings outside the globe.
    Rings,
}

/// Everything the anti-ringing area depends on, to cache it.
#[derive(Clone, Debug, PartialEq)]
pub struct AreaKey {
    radius: u32,
    edges: f32,
    region: DeringRegion,
}

impl Default for DeringParams {
    fn default() -> Self {
        Self { strength: 1.0, strokes: Vec::new(), width: 0.0, radius: 4, edges: 0.1, region: DeringRegion::Subject }
    }
}

impl DeringParams {
    /// Anti-ringing halo width from the blur σ (px): the bands of a sharpened edge lie within a
    /// few σ of it.
    pub fn for_blur(sigma: f32) -> Self {
        Self { radius: (2.5 * sigma + 2.0).round().clamp(2.0, 12.0) as u32, ..Default::default() }
    }

    /// Nothing to do: off, or nothing painted.
    pub fn is_neutral(&self) -> bool {
        self.strength <= 0.0 || self.strokes.iter().all(|s| s.erase)
    }

    pub fn area_key(&self) -> AreaKey {
        AreaKey { radius: self.radius, edges: self.edges, region: self.region }
    }
}

/// Where the user painted over the ring (0‥1), at `w` × `h`.
pub fn painted(strokes: &[Stroke], w: usize, h: usize) -> Vec<f32> {
    let mut area = vec![0.0f32; w * h];
    for stroke in strokes {
        mask::paint(&mut area, w, h, stroke);
    }
    area
}

/// The ring found under the painted area.
#[derive(Clone, Debug)]
pub struct Ring {
    /// Per channel of the image, how much each pixel is lifted at full strength (≥ 0; zero
    /// away from the ring).
    pub lift: Vec<Vec<f32>>,
    /// The same on the luminance: what "Mostra gli aloni" shows.
    pub map: Vec<f32>,
    /// Half-width of the closing that fills it (px): about the ring's width.
    pub reach: usize,
}

/// Smoothing (σ, px) along the band of the copy the ring is filled on: the noise's own dips
/// are not lifted, and the ring, running along the band, keeps its depth.
const SMOOTH: f32 = 1.5;
/// The widest closing tried (half-width, px).
const MAX_REACH: usize = 16;

/// Finds and measures the ring under `area` (the painted coverage, 0‥1, at the image's size)
/// in `img`. `width` (px, 0 = measured) forces its width. `None` when nothing is painted.
pub fn find_ring(img: &Image, area: &[f32], width: f32) -> Option<Ring> {
    let (w, h) = (img.width(), img.height());
    if area.len() != w * h {
        return None;
    }
    // The painted band's width: twice its area over its outline's length.
    let (mut inside, mut outline) = (0usize, 0usize);
    let mut bbox = (w, h, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if area[i] <= 1e-3 {
                continue;
            }
            bbox = (bbox.0.min(x), bbox.1.min(y), bbox.2.max(x + 1), bbox.3.max(y + 1));
            if area[i] >= 0.5 {
                inside += 1;
                let edge = x == 0 || y == 0 || x + 1 == w || y + 1 == h || [i - 1, i + 1, i - w, i + w].iter().any(|&j| area[j] < 0.5);
                outline += edge as usize;
            }
        }
    }
    if inside == 0 {
        return None;
    }
    let band = 2.0 * inside as f32 / outline.max(1) as f32;
    let max_reach = if width > 0.0 { width.round().max(1.0) as usize } else { (band / 2.0).ceil().clamp(2.0, MAX_REACH as f32) as usize };
    // Work on the painted part only, with room for the widest closing and the smoothing.
    let margin = 2 * max_reach + 4;
    let (x0, y0) = (bbox.0.saturating_sub(margin), bbox.1.saturating_sub(margin));
    let (x1, y1) = ((bbox.2 + margin).min(w), (bbox.3 + margin).min(h));
    let (cw, ch) = (x1 - x0, y1 - y0);
    let crop = |plane: &[f32]| -> Vec<f32> { (y0..y1).flat_map(|y| plane[y * w + x0..y * w + x1].iter().copied()).collect() };
    let a = crop(area);
    let lum = crop(&working_planes(img, ColorMode::Luminance).swap_remove(0));
    let across = across_band(&a, &lum, cw, ch, band);
    let smooth = along_blur(&lum, cw, ch, &across, SMOOTH);
    // What the closing of each width adds to the ring: its strongest fill under the brush.
    let painted_px: Vec<usize> = (0..cw * ch).filter(|&i| a[i] >= 0.5).collect();
    let strongest = |fill: &[f32]| -> f32 {
        let mut v: Vec<f32> = painted_px.iter().map(|&i| fill[i]).collect();
        let k = (v.len() * 95 / 100).min(v.len() - 1);
        *v.select_nth_unstable_by(k, f32::total_cmp).1
    };
    let reach = if width > 0.0 {
        max_reach
    } else {
        let curve: Vec<f32> = (1..=max_reach).map(|r| strongest(&fill(&smooth, cw, ch, &across, r))).collect();
        let full = curve.iter().copied().fold(0.0f32, f32::max);
        // The smallest window that fills almost all of it (wider ones reach other structures).
        curve.iter().position(|&q| q >= 0.95 * full).map_or(max_reach, |k| k + 1)
    };
    let lum_fill = fill(&smooth, cw, ch, &across, reach);
    // The noise, from what the smoothing took away, and how strong the ring's fill is: where
    // the fill stands clearly out of both is the ring.
    let mut residual: Vec<f32> = painted_px.iter().map(|&i| (lum[i] - smooth[i]).abs()).collect();
    let mid = residual.len() / 2;
    let noise = 1.4826 * *residual.select_nth_unstable_by(mid, f32::total_cmp).1;
    let top = strongest(&lum_fill);
    let (lo, hi) = ((0.6 * noise).max(0.1 * top), (1.5 * noise).max(0.3 * top));
    // The ring runs along the band: what is not there along most of it nearby (a belt or a
    // spot crossing the band) is not the ring.
    let along = along_quantile(&lum_fill, cw, ch, &across, (2 * reach).max(8), 0.5);
    // The brush only says where the ring is: once found, it is lifted across its whole width,
    // also where it runs a little outside the painted band.
    let a = reach_across(&a, cw, ch, &across, reach);
    let ring: Vec<f32> = (0..cw * ch)
        .map(|i| {
            let kept = along[i].min(2.0 * lum_fill[i]);
            let t = ((kept - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
            a[i] * t * t * (3.0 - 2.0 * t) * (kept / lum_fill[i].max(1e-9)).min(1.0)
        })
        .collect();
    let ring = blur(&ring, cw, ch, 0.7);
    // Back to the image's size.
    let place = |part: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        for (y, row) in part.chunks(cw).enumerate() {
            out[(y0 + y) * w + x0..(y0 + y) * w + x1].copy_from_slice(row);
        }
        out
    };
    // Smoothed a little along the ring, so it rises evenly.
    let lift_of = |filled: &[f32]| place(&along_blur(&filled.iter().zip(&ring).map(|(f, r)| f * r).collect::<Vec<_>>(), cw, ch, &across, 1.0));
    let map = lift_of(&lum_fill);
    let lift = if img.channels().len() == 1 {
        vec![map.clone()]
    } else {
        img.channels().par_iter().map(|c| lift_of(&fill(&along_blur(&crop(c), cw, ch, &across, SMOOTH), cw, ch, &across, reach))).collect()
    };
    Some(Ring { lift, map, reach })
}

/// Gaussian smoothing (σ px) of `plane` along the band (across `across`).
fn along_blur(plane: &[f32], w: usize, h: usize, across: &[[f32; 2]], sigma: f32) -> Vec<f32> {
    let reach = (3.0 * sigma).ceil() as i32;
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let [dx, dy] = across[i];
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let (mut sum, mut total) = (0.0f32, 0.0f32);
            for s in -reach..=reach {
                // Bilinear, so the line along a curved band stays on it.
                if let Some(v) = bilinear(plane, w, h, x + 0.5 - s as f32 * dy, y + 0.5 + s as f32 * dx) {
                    let g = (-(s * s) as f32 / (2.0 * sigma * sigma)).exp();
                    (sum, total) = (sum + g * v, total + g);
                }
            }
            if total > 0.0 { sum / total } else { plane[i] }
        })
        .collect()
}

/// `plane` at (x, y), pixel centres at half-integers; `None` outside.
fn bilinear(plane: &[f32], w: usize, h: usize, x: f32, y: f32) -> Option<f32> {
    let (x, y) = (x - 0.5, y - 0.5);
    if x < 0.0 || y < 0.0 || x > (w - 1) as f32 || y > (h - 1) as f32 {
        return None;
    }
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let top = plane[y0 * w + x0] * (1.0 - tx) + plane[y0 * w + x1] * tx;
    let bottom = plane[y1 * w + x0] * (1.0 - tx) + plane[y1 * w + x1] * tx;
    Some(top * (1.0 - ty) + bottom * ty)
}

/// Least-squares parabola v = a + b·t + c·t² through `points` (t, v): (a, b, c, rms misfit).
fn parabola(points: &[(f32, f32)]) -> (f32, f32, f32, f32) {
    let mut m = [[0.0f64; 3]; 3];
    let mut y = [0.0f64; 3];
    for &(t, v) in points {
        let p = [1.0, t as f64, (t * t) as f64];
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += p[r] * p[c];
            }
            y[r] += p[r] * v as f64;
        }
    }
    let det = |m: &[[f64; 3]; 3]| m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let d = det(&m);
    if d.abs() < 1e-12 {
        return (0.0, 0.0, 0.0, f32::MAX);
    }
    let solve = |k: usize| {
        let mut mk = m;
        (0..3).for_each(|r| mk[r][k] = y[r]);
        (det(&mk) / d) as f32
    };
    let (a, b, c) = (solve(0), solve(1), solve(2));
    let misfit = (points.iter().map(|&(t, v)| (v - (a + b * t + c * t * t)).powi(2)).sum::<f32>() / points.len() as f32).sqrt();
    (a, b, c, misfit)
}

/// The strongest value of `plane` within `reach` px across the band.
fn reach_across(plane: &[f32], w: usize, h: usize, across: &[[f32; 2]], reach: usize) -> Vec<f32> {
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let [dx, dy] = across[i];
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            (-(reach as i32)..=reach as i32).fold(0.0f32, |m, t| {
                let (sx, sy) = ((x + t as f32 * dx).round(), (y + t as f32 * dy).round());
                if sx < 0.0 || sy < 0.0 || sx >= w as f32 || sy >= h as f32 { m } else { m.max(plane[sy as usize * w + sx as usize]) }
            })
        })
        .collect()
}

/// The `q` quantile of `plane` along the band (across `across`), `reach` px either way: a low
/// one keeps only what is there along most of it.
fn along_quantile(plane: &[f32], w: usize, h: usize, across: &[[f32; 2]], reach: usize, q: f32) -> Vec<f32> {
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let [dx, dy] = across[i];
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let mut v: Vec<f32> = (-(reach as i32)..=reach as i32)
                .filter_map(|s| {
                    let (sx, sy) = ((x - s as f32 * dy).round(), (y + s as f32 * dx).round());
                    (sx >= 0.0 && sy >= 0.0 && sx < w as f32 && sy < h as f32).then(|| plane[sy as usize * w + sx as usize])
                })
                .collect();
            let k = ((v.len() as f32 * q) as usize).min(v.len() - 1);
            *v.select_nth_unstable_by(k, f32::total_cmp).1
        })
        .collect()
}

/// The direction across the ring at each pixel (unit vector). A ring follows the sharp edge
/// that made it (the limb, a crater's rim), so this is mostly the direction of the image's
/// gradient over about the band's width (a structure tensor); the painted band's own direction
/// (its outline faces across it on both sides) helps where the image has no clear edge. A
/// single dab of the brush has no direction of its own.
fn across_band(area: &[f32], image: &[f32], w: usize, h: usize, band: f32) -> Vec<[f32; 2]> {
    let reach = (0.6 * band).max(2.0);
    let tensor = |plane: &[f32], pre: f32| -> [Vec<f32>; 3] {
        let soft = blur(plane, w, h, pre);
        let at = |x: usize, y: usize| soft[y.min(h - 1) * w + x.min(w - 1)];
        let (mut xx, mut yy, mut xy) = (vec![0.0f32; w * h], vec![0.0f32; w * h], vec![0.0f32; w * h]);
        for y in 0..h {
            for x in 0..w {
                let gx = 0.5 * (at(x + 1, y) - at(x.saturating_sub(1), y));
                let gy = 0.5 * (at(x, y + 1) - at(x, y.saturating_sub(1)));
                let i = y * w + x;
                (xx[i], yy[i], xy[i]) = (gx * gx, gy * gy, gx * gy);
            }
        }
        [blur(&xx, w, h, reach), blur(&yy, w, h, reach), blur(&xy, w, h, reach)]
    };
    let [ixx, iyy, ixy] = tensor(image, 1.0);
    let [axx, ayy, axy] = tensor(area, 1.0);
    // The image's direction counts fully on a clear edge, little where it is flat (there its
    // direction is the noise's); the band's counts for its direction only.
    let mut traces: Vec<f32> = (0..w * h).map(|i| ixx[i] + iyy[i]).collect();
    let mid = traces.len() / 2;
    let typical = traces.select_nth_unstable_by(mid, f32::total_cmp).1.max(1e-12);
    (0..w * h)
        .map(|i| {
            let (ni, na) = (1.0 / (ixx[i] + iyy[i] + typical), 0.5 / (axx[i] + ayy[i]).max(1e-12));
            let (xx, yy, xy) = (ixx[i] * ni + axx[i] * na, iyy[i] * ni + ayy[i] * na, ixy[i] * ni + axy[i] * na);
            let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
            [angle.cos(), angle.sin()]
        })
        .collect()
}

/// How much a dark valley running along the band, about `r` px wide on either side, sits below
/// its surroundings at each pixel of `plane` (≥ 0), looking only along `across`. A belt crossing
/// the band is not a valley along this direction.
///
/// Two estimates of what lies under the valley:
/// - the lower of its two sides (a closing): always safe — a step edge such as the limb, a
///   shadow's border, a convex disk are not raised — but on a slope (the limb darkening) it fills
///   only up to the lower side;
/// - a parabola through the surroundings on both sides (from `r` to `2r` px away): it follows
///   a slope and its bend (the limb darkening, the fading tail of an edge, which is then not
///   mistaken for a valley), and is trusted only where the two sides continue one another
///   (beside a step edge they do not, and it would raise the edge's foot).
fn fill(plane: &[f32], w: usize, h: usize, across: &[[f32; 2]], r: usize) -> Vec<f32> {
    let offset = |i: usize, t: f32| -> Option<usize> {
        let [dx, dy] = across[i];
        let (sx, sy) = (((i % w) as f32 + t * dx).round(), ((i / w) as f32 + t * dy).round());
        (sx >= 0.0 && sy >= 0.0 && sx < w as f32 && sy < h as f32).then(|| sy as usize * w + sx as usize)
    };
    let extreme = |src: &[f32], i: usize, pick: fn(f32, f32) -> f32, start: f32| -> f32 { (-(r as i32)..=r as i32).filter_map(|t| offset(i, t as f32)).fold(start, |acc, j| pick(acc, src[j])) };
    let dilated: Vec<f32> = (0..w * h).into_par_iter().map(|i| extreme(plane, i, f32::max, f32::MIN)).collect();
    let closing: Vec<f32> = (0..w * h).into_par_iter().map(|i| (extreme(&dilated, i, f32::min, f32::MAX) - plane[i]).max(0.0)).collect();
    // At each pixel taken as the valley's centre: the parabola through the surroundings (value
    // here, slope and bend along `across`) and how much to trust it (0‥1) times the dip under it.
    let far = 2 * r as i32;
    let lines: Vec<(f32, f32, f32, f32)> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let points: Vec<(f32, f32)> = (-far..=far).filter(|t| t.unsigned_abs() as usize >= r).filter_map(|t| offset(i, t as f32).map(|j| (t as f32, plane[j]))).collect();
            if points.iter().filter(|p| p.0 < 0.0).count() < 2 || points.iter().filter(|p| p.0 > 0.0).count() < 2 {
                return (0.0, 0.0, 0.0, 0.0);
            }
            let fit = |pts: &[(f32, f32)]| -> (f32, f32, f32) {
                let n = pts.len() as f32;
                let (mt, mv) = pts.iter().fold((0.0, 0.0), |(a, b), p| (a + p.0 / n, b + p.1 / n));
                let (stt, stv) = pts.iter().fold((0.0, 0.0), |(a, b), p| (a + (p.0 - mt).powi(2), b + (p.0 - mt) * (p.1 - mv)));
                let slope = stv / stt.max(1e-6);
                let here = mv - slope * mt;
                let misfit = (pts.iter().map(|p| (p.1 - (here + slope * p.0)).powi(2)).sum::<f32>() / n).sqrt();
                (here, slope, misfit)
            };
            let (here, slope, curve, misfit) = parabola(&points);
            let split = points.iter().position(|p| p.0 > 0.0).unwrap_or(points.len());
            let ((_, left, left_misfit), (_, right, right_misfit)) = (fit(&points[..split]), fit(&points[split..]));
            // The surroundings must continue one another: on both sides the same slope (a ring on
            // the limb darkening), or both flat. Beside a step edge hidden under the valley they
            // do not (the planet's side falls, the sky's is flat), and the line would raise it.
            let noise = 2.0 * left_misfit.max(right_misfit) / r as f32;
            let (gentle, steep) = (left.abs().min(right.abs()), left.abs().max(right.abs()));
            let agree = if steep < noise {
                ((2.0 * noise - slope.abs()) / noise).clamp(0.0, 1.0)
            } else if left * right > 0.0 {
                ((gentle / steep - 0.15) / 0.2).clamp(0.0, 1.0) * ((1.5 * steep + noise - slope.abs()) / noise).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let step = 1.0 - agree;
            let dip = here - plane[i];
            let t = ((dip / misfit.max(1e-6) - 2.0) / 2.0).clamp(0.0, 1.0) * (1.0 - step);
            (here, slope, curve, t * t * (3.0 - 2.0 * t) * dip.max(0.0))
        })
        .collect();
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let v = plane[i];
            // The best-centred trusted parabola nearby, carried to this pixel.
            let best = (-(r as i32)..=r as i32).filter_map(|t| offset(i, t as f32).map(|j| (t as f32, j))).max_by(|a, b| lines[a.1].3.total_cmp(&lines[b.1].3));
            let line = best.filter(|&(_, j)| lines[j].3 > 0.0).map_or(0.0, |(t, j)| {
                let (here, slope, curve, score) = lines[j];
                let dip = here - plane[j];
                // This pixel lies -t·cos(angle between the two directions) from it along its own.
                let along = -t * (across[i][0] * across[j][0] + across[i][1] * across[j][1]);
                // Never above the brightest of its surroundings.
                (score / dip.max(1e-6)) * ((here + slope * along + curve * along * along).min(dilated[i]) - v)
            });
            closing[i].max(line)
        })
        .collect()
}

/// `sharp` with the ring lifted: `ring` is [`find_ring`] on it (computed here if `None`),
/// `area` the painted coverage.
pub fn dering(sharp: &Image, p: &DeringParams, area: &[f32], ring: Option<&Ring>) -> Image {
    let own;
    let ring = match ring {
        Some(r) => Some(r),
        None if !p.is_neutral() => {
            own = find_ring(sharp, area, p.width);
            own.as_ref()
        }
        None => None,
    };
    let Some(ring) = ring.filter(|r| !p.is_neutral() && r.lift.len() == sharp.channels().len() && r.map.len() == sharp.width() * sharp.height()) else {
        return sharp.clone();
    };
    let channels = sharp.channels().par_iter().zip(&ring.lift).map(|(plane, lift)| plane.iter().zip(lift).map(|(v, l)| v + p.strength * l).collect()).collect();
    Image::new(sharp.width(), sharp.height(), channels, sharp.source_bit_depth)
}

/// Gaussian blur (σ px) of a small plane, the borders extended.
fn blur(plane: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let rows: Vec<f32> = plane.par_chunks(w).flat_map_iter(|row| gaussian_1d(row, sigma)).collect();
    let mut cols = vec![0.0f32; w * h];
    let done: Vec<Vec<f32>> = (0..w).into_par_iter().map(|x| gaussian_1d(&(0..h).map(|y| rows[y * w + x]).collect::<Vec<_>>(), sigma)).collect();
    for (x, col) in done.iter().enumerate() {
        for (y, v) in col.iter().enumerate() {
            cols[y * w + x] = *v;
        }
    }
    cols
}

/// The disk deconvolution's anti-ringing area follows: Saturn's globe (from the rings'
/// geometry), or the planet's outline as an ellipse. `None` for the rings, or when the subject isn't a whole disk
/// (a close-up running off the frame).
pub fn find_globe(image: &Image, region: DeringRegion) -> Option<Ellipse> {
    match region {
        DeringRegion::Rings => None,
        DeringRegion::Globe => {
            let globe = mask::saturn_globe(image)?;
            let lum = working_planes(image, ColorMode::Luminance).swap_remove(0);
            Some(fit_limb(&lum, image.width(), image.height(), globe))
        }
        DeringRegion::Subject => {
            let (w, h) = (image.width(), image.height());
            let lum = working_planes(image, ColorMode::Luminance).swap_remove(0);
            let mut body = main_body(&foreground(&lum, w, h), w, h);
            // The subject mask reaches into the blurred limb: the disk ends at half its
            // brightness.
            let mut inside: Vec<f32> = body.iter().zip(&lum).filter(|(b, _)| **b).map(|(_, &v)| v).collect();
            if inside.len() < 100 {
                return None;
            }
            let mid = inside.len() / 2;
            let half = 0.5 * *inside.select_nth_unstable_by(mid, f32::total_cmp).1;
            body.iter_mut().zip(&lum).for_each(|(b, &v)| *b &= v > half);
            let (mut n, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
            for (i, _) in body.iter().enumerate().filter(|(_, b)| **b) {
                (n, sx, sy) = (n + 1.0, sx + (i % w) as f64 + 0.5, sy + (i / w) as f64 + 0.5);
            }
            if n < 100.0 {
                return None;
            }
            let (cx, cy) = (sx / n, sy / n);
            let (mut xx, mut yy, mut xy) = (0.0f64, 0.0f64, 0.0f64);
            for (i, _) in body.iter().enumerate().filter(|(_, b)| **b) {
                let (dx, dy) = ((i % w) as f64 + 0.5 - cx, (i / w) as f64 + 0.5 - cy);
                (xx, yy, xy) = (xx + dx * dx, yy + dy * dy, xy + dx * dy);
            }
            let (xx, yy, xy) = (xx / n, yy / n, xy / n);
            let (mean, diff) = ((xx + yy) / 2.0, (((xx - yy) / 2.0).powi(2) + xy * xy).sqrt());
            // A filled ellipse with semi-axes a, b has variances a²/4 and b²/4.
            let (a, b) = (2.0 * (mean + diff).sqrt(), 2.0 * (mean - diff).max(0.0).sqrt());
            let ellipse = Ellipse { cx: cx as f32, cy: cy as f32, rx: a as f32, ry: b as f32, angle: (0.5 * (2.0 * xy).atan2(xx - yy)) as f32 };
            // Only a whole disk: the ellipse must match the subject.
            let mismatch = body.iter().enumerate().filter(|(i, b)| **b != (ellipse.radius((i % w) as f32 + 0.5, (i / w) as f32 + 0.5) < 1.0)).count() as f64;
            (mismatch / n < 0.08 && a > 8.0).then_some(ellipse)
        }
    }
}

/// `globe` scaled so its outline sits where the disk falls to half its brightness: the median
/// over rays that reach the sky (the others cross the rings).
fn fit_limb(lum: &[f32], w: usize, h: usize, globe: Ellipse) -> Ellipse {
    let (sin, cos) = globe.angle.sin_cos();
    let at = |rho: f32, phi: f32| -> Option<f32> {
        let (u, v) = (rho * globe.rx * phi.cos(), rho * globe.ry * phi.sin());
        let (x, y) = (globe.cx + u * cos - v * sin, globe.cy + u * sin + v * cos);
        (x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h).then(|| lum[y as usize * w + x as usize])
    };
    let mut found: Vec<f32> = (0..180)
        .filter_map(|k| {
            let phi = k as f32 * std::f32::consts::TAU / 180.0;
            let mut inner: Vec<f32> = (0..30).filter_map(|j| at(0.5 + 0.01 * j as f32, phi)).collect();
            if inner.len() < 30 {
                return None;
            }
            let half = 0.5 * *inner.select_nth_unstable_by(15, f32::total_cmp).1;
            // Beyond the limb this ray must reach the sky.
            if (0..10).any(|j| at(1.25 + 0.02 * j as f32, phi).is_none_or(|v| v > 0.5 * half)) {
                return None;
            }
            (0..300).map(|j| 0.7 + 0.002 * j as f32).find(|&rho| at(rho, phi).is_some_and(|v| v < half))
        })
        .collect();
    if found.len() < 20 {
        return globe;
    }
    let mid = found.len() / 2;
    let scale = *found.select_nth_unstable_by(mid, f32::total_cmp).1;
    Ellipse { rx: globe.rx * scale, ry: globe.ry * scale, ..globe }
}

/// Where deconvolution's anti-ringing holds its estimate (0‥1): near the image's sharp edges,
/// within the region (on a whole planet, a band along its limb).
pub fn weight(image: &Image, p: &DeringParams) -> Vec<f32> {
    let (w, h) = (image.width(), image.height());
    let lum = working_planes(image, ColorMode::Luminance).swap_remove(0);
    // An edge counts from a range of `edges` of the subject's brightness, fully from about
    // twice that, as far away as the halos reach.
    let (lo, hi) = min_max(&lum, w, h, p.radius.max(1) as usize);
    let stride = lum.len().div_ceil(1 << 16).max(1);
    let mut sample: Vec<f32> = lum.iter().step_by(stride).copied().collect();
    let n = sample.len();
    let bright = sample.select_nth_unstable_by((n * 995 / 1000).min(n - 1), f32::total_cmp).1.max(1e-3);
    let (from, to) = (p.edges.max(0.0), 2.0 * p.edges.max(0.0) + 0.05);
    let globe = match p.region {
        // A whole planet: its limb.
        DeringRegion::Subject => find_globe(image, DeringRegion::Subject),
        _ => find_globe(image, DeringRegion::Globe),
    };
    // The globe and its limb's halo (Globe), or what is outside it (Rings).
    let margin = globe.map_or(0.0, |g| (p.radius as f32 + 2.0) / g.rx.min(g.ry));
    // On a whole planet only a band along the limb, `radius` px deep inside it (the dark ring
    // sharpening draws there) and the sky beside it: belts and spots further in, whose edges can
    // be as marked, are not ringing.
    let band = p.radius as f32 + 2.0;
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let allowed = match globe {
                Some(g) => {
                    let rho = g.radius((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                    match p.region {
                        DeringRegion::Globe => ((1.0 + margin - rho) / margin.max(1e-3)).clamp(0.0, 1.0),
                        DeringRegion::Rings => ((rho - 1.0) / 0.03).clamp(0.0, 1.0),
                        DeringRegion::Subject => ((band - (1.0 - rho) * g.rx.min(g.ry)) / 2.0).clamp(0.0, 1.0),
                    }
                }
                None => 1.0,
            };
            let t = (((hi[i] - lo[i]) / bright - from) / (to - from)).clamp(0.0, 1.0);
            allowed * t * t * (3.0 - 2.0 * t)
        })
        .collect()
}

/// Minimum and maximum over the (2r+1)² square around each pixel. Both passes combine whole
/// shifted rows element by element (no transposes, vectorisable).
pub(crate) fn min_max(plane: &[f32], w: usize, h: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    if let Some(out) = crate::gpu::for_plane(w, h, crate::gpu::Work::Light).and_then(|g| g.min_max(plane, w, h, r).ok()) {
        return out;
    }
    min_max_cpu(plane, w, h, r)
}

pub(crate) fn min_max_cpu(plane: &[f32], w: usize, h: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut row_min, mut row_max) = (plane.to_vec(), plane.to_vec());
    row_min.par_chunks_mut(w).zip(row_max.par_chunks_mut(w)).zip(plane.par_chunks(w)).for_each(|((lo, hi), row)| {
        for k in 1..=r.min(w - 1) {
            for (x, &v) in row[..w - k].iter().enumerate() {
                // row[x] seen from x + k, and row[x + k] seen from x.
                let (a, b) = (x + k, x);
                lo[a] = lo[a].min(v);
                hi[a] = hi[a].max(v);
                lo[b] = lo[b].min(row[x + k]);
                hi[b] = hi[b].max(row[x + k]);
            }
        }
    });
    let (mut lo, mut hi) = (vec![0.0f32; w * h], vec![0.0f32; w * h]);
    lo.par_chunks_mut(w).zip(hi.par_chunks_mut(w)).enumerate().for_each(|(y, (lo, hi))| {
        let (y0, y1) = (y.saturating_sub(r), (y + r).min(h - 1));
        lo.copy_from_slice(&row_min[y0 * w..(y0 + 1) * w]);
        hi.copy_from_slice(&row_max[y0 * w..(y0 + 1) * w]);
        for yy in y0 + 1..=y1 {
            lo.iter_mut().zip(&row_min[yy * w..(yy + 1) * w]).for_each(|(o, &v)| *o = o.min(v));
            hi.iter_mut().zip(&row_max[yy * w..(yy + 1) * w]).for_each(|(o, &v)| *o = o.max(v));
        }
    });
    (lo, hi)
}

/// Gaussian smoothing of a row (σ in samples), the ends extended.
fn gaussian_1d(row: &[f32], sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil() as i32;
    let kernel: Vec<f32> = (-r..=r).map(|k| (-(k as f32).powi(2) / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = kernel.iter().sum();
    let n = row.len() as i32;
    (0..n).map(|i| (-r..=r).zip(&kernel).map(|(k, g)| g * row[(i + k).clamp(0, n - 1) as usize]).sum::<f32>() / total).collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    use crate::fft::FftConvolver;
    use crate::psf::Psf;

    const S: usize = 260;
    const RD: f32 = 100.0;

    fn at(i: usize) -> (f32, f32) {
        let c = S as f32 / 2.0;
        ((i % S) as f32 + 0.5 - c, (i / S) as f32 + 0.5 - c)
    }

    /// A blurred limb-darkened disk with belts 40 px apart (parallel to the equator) and noise;
    /// `sharpen` adds an unsharp mask (a bright rim on the limb); `ring` a dark ring that is a
    /// shifted copy of the limb (7–11 px inside, σ 1.3 px), on the right half only. Returns
    /// (image, the ring's darkening).
    fn planet(ring: bool, sharpen: bool, colour: bool) -> (Image, Vec<f32>) {
        let noise = |i: usize| ((i.wrapping_mul(2_654_435_761) >> 7) % 1000) as f32 / 1000.0 - 0.5;
        let plane: Vec<f32> = (0..S * S)
            .map(|i| {
                let (dx, dy) = at(i);
                let r = dx.hypot(dy) / RD;
                let belts = 0.05 * (dy / 40.0 * std::f32::consts::TAU).sin();
                if r < 1.0 { 0.75 * (1.0 - r * r).powf(0.25) + belts } else { 0.02 }
            })
            .collect();
        let blur = |p: &[f32], s: f32| FftConvolver::new(S, S, &Psf::Gaussian { sigma: s }.kernel()).convolve(p);
        let reference = blur(&plane, 1.5);
        let soft = blur(&reference, 2.0);
        let darkening: Vec<f32> = (0..S * S)
            .map(|i| {
                let (dx, dy) = at(i);
                if !ring || dx.hypot(dy) > RD {
                    return 0.0;
                }
                let strength = 0.08 * dx.max(0.0) / dx.hypot(dy).max(1.0);
                let r = (dx + 2.0).hypot(dy);
                -strength * (-((r - (RD - 9.0)) / 1.3).powi(2) / 2.0).exp()
            })
            .collect();
        let gain = if sharpen { 1.2 } else { 0.0 };
        let out: Vec<f32> = (0..S * S).map(|i| reference[i] + gain * (reference[i] - soft[i]) + darkening[i] + 0.008 * noise(i)).collect();
        let image = if colour { Image::new(S, S, vec![out.iter().map(|v| v * 1.1).collect(), out.clone(), out.iter().map(|v| v * 0.8).collect()], 16) } else { Image::new(S, S, vec![out], 16) };
        (image, darkening)
    }

    /// A rough stroke along the right half of the ring, wobbling ±2 px around it.
    fn rough_stroke(radius: f32) -> Stroke {
        let c = S as f32 / 2.0;
        let points = (0..=40)
            .map(|k| {
                let theta = (-80.0 + 4.0 * k as f32).to_radians();
                let r = RD - 9.0 + 2.0 * (k as f32 * 1.7).sin();
                [c - 2.0 + r * theta.cos(), c + r * theta.sin()]
            })
            .collect();
        Stroke { points, radius, hardness: 0.5, erase: false }
    }

    fn painted_params(radius: f32) -> DeringParams {
        DeringParams { strokes: vec![rough_stroke(radius)], ..Default::default() }
    }

    /// Painted roughly over it, the ring goes; the limb's bright rim, the belts and what lies
    /// outside the brush stay.
    #[test]
    fn lifts_the_painted_ring_and_nothing_else() {
        let (sharp, ring) = planet(true, true, false);
        let (clean, _) = planet(false, true, false);
        let p = painted_params(6.0);
        let area = painted(&p.strokes, S, S);
        let found = find_ring(&sharp, &area, 0.0).expect("painted");
        let out = dering(&sharp, &p, &area, Some(&found));
        let (out, sharp, clean) = (&out.channels()[0], &sharp.channels()[0], &clean.channels()[0]);
        let (mut before, mut after, mut rim, mut beside, mut beside_n) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0usize);
        for i in 0..S * S {
            let (dx, dy) = at(i);
            let r = dx.hypot(dy);
            if ring[i] < -0.03 && area[i] > 0.9 {
                before += (sharp[i] - clean[i]).abs();
                after += (out[i] - clean[i]).abs();
            }
            if (r - RD).abs() < 2.0 {
                rim = rim.max((out[i] - sharp[i]).abs());
            }
            if dx > 0.0 && ring[i] > -0.001 && area[i] > 0.0 {
                beside += (out[i] - sharp[i]).abs();
                beside_n += 1;
            }
            if area[i] == 0.0 && (r - (RD - 9.0)).abs() > 20.0 {
                assert_eq!(out[i], sharp[i], "outside the brush at {dx} {dy}");
            }
        }
        assert!(after < 0.25 * before, "ring {before} → {after} (reach {})", found.reach);
        assert!(rim < 0.01, "limb rim changed by {rim}");
        assert!(beside / (beside_n as f32) < 0.002, "around the ring changed by {}", beside / beside_n as f32);
    }

    /// Painted where there is no ring (along the limb, and a dab across its edge), almost
    /// nothing is lifted.
    #[test]
    fn no_ring_no_change() {
        let (image, _) = planet(false, false, false);
        for stroke in [rough_stroke(6.0), Stroke { points: vec![[222.0, 130.0], [234.0, 130.0]], radius: 6.0, hardness: 0.5, erase: false }] {
            let p = DeringParams { strokes: vec![stroke], ..Default::default() };
            let out = dering(&image, &p, &painted(&p.strokes, S, S), None);
            let worst = out.channels()[0].iter().zip(&image.channels()[0]).map(|(a, b)| a - b).fold(0.0f32, f32::max);
            assert!(worst < 0.01, "lifted {worst} where there is no ring");
        }
    }

    /// In colour each channel is lifted by its own ring.
    #[test]
    fn colour_channels_are_lifted_on_their_own() {
        let (sharp, ring) = planet(true, true, true);
        let p = painted_params(6.0);
        let area = painted(&p.strokes, S, S);
        let out = dering(&sharp, &p, &area, None);
        let deepest = (0..S * S).min_by(|&a, &b| ring[a].total_cmp(&ring[b])).unwrap();
        let lifts: Vec<f32> = (0..3).map(|c| out.channels()[c][deepest] - sharp.channels()[c][deepest]).collect();
        assert!(lifts[0] > lifts[1] && lifts[1] > lifts[2] && lifts[2] > 0.03, "{lifts:?}");
    }

    #[test]
    fn nothing_painted_or_zero_strength_is_identity() {
        let (sharp, _) = planet(true, true, false);
        let none = DeringParams::default();
        assert!(none.is_neutral());
        assert_eq!(dering(&sharp, &none, &vec![0.0; S * S], None).channels(), sharp.channels());
        let p = DeringParams { strength: 0.0, ..painted_params(6.0) };
        assert_eq!(dering(&sharp, &p, &painted(&p.strokes, S, S), None).channels(), sharp.channels());
    }

    #[test]
    fn finds_the_disk() {
        let (reference, _) = planet(false, false, false);
        let g = find_globe(&reference, DeringRegion::Subject).expect("a whole disk");
        assert!((g.cx - S as f32 / 2.0).abs() < 1.0 && (g.cy - S as f32 / 2.0).abs() < 1.0, "{g:?}");
        assert!((g.rx - RD).abs() < 2.5 && (g.ry - RD).abs() < 2.5, "{g:?}");
    }

    #[test]
    fn min_max_matches_brute_force() {
        let (w, h, r) = (23, 17, 2);
        let plane: Vec<f32> = (0..w * h).map(|i| ((i * 7919) % 101) as f32).collect();
        let (lo, hi) = min_max(&plane, w, h, r);
        for y in 0..h {
            for x in 0..w {
                let window = (y.saturating_sub(r)..=(y + r).min(h - 1)).flat_map(|yy| (x.saturating_sub(r)..=(x + r).min(w - 1)).map(move |xx| (xx, yy)));
                let values: Vec<f32> = window.map(|(xx, yy)| plane[yy * w + xx]).collect();
                assert_eq!(lo[y * w + x], values.iter().copied().fold(f32::MAX, f32::min));
                assert_eq!(hi[y * w + x], values.iter().copied().fold(f32::MIN, f32::max));
            }
        }
    }

    /// On Saturn the area can follow the globe's limb and leave the rings alone.
    #[test]
    fn globe_area_leaves_the_rings_out() {
        let (w, h, r) = (700usize, 400usize, 100.0f32);
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 + 0.5 - 350.0, (i / w) as f32 + 0.5 - 200.0);
                let globe = (x / r).powi(2) + (y / (0.9 * r)).powi(2) < 1.0;
                let e = ((x / (2.27 * r)).powi(2) + (y / (0.6 * r)).powi(2)).sqrt();
                if globe {
                    0.7
                } else if (0.68..1.0).contains(&e) {
                    0.5
                } else {
                    0.0
                }
            })
            .collect();
        let img = Image::new(w, h, vec![plane], 16);
        let p = DeringParams { region: DeringRegion::Globe, ..DeringParams::for_blur(1.5) };
        let area = weight(&img, &p);
        let at = |x: usize, y: usize| area[y * w + x];
        assert!(at(350, 200 - 90) > 0.5, "on the globe's limb");
        assert!(at(350 + 227, 200) < 0.05, "out on the rings");
        assert!(at(350, 200) < 0.05, "centre of the globe, no edge there");
        let rings = weight(&img, &DeringParams { region: DeringRegion::Rings, ..p });
        assert!(rings[200 * w + 350 + 226] > 0.5, "rings area follows their outer edge");
    }

}
