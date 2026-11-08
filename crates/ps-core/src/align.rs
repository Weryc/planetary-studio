//! RGB channel alignment: atmospheric dispersion and imperfect stacking leave R and B shifted
//! relative to G by fractions of a pixel. Shifts are applied with Catmull–Rom interpolation and
//! measured in one of two ways:
//! - a subject entirely inside the frame (planet, Saturn, whole disk): the channels' centroids
//!   over the same region. Thousands of pixels are averaged, so it stays precise on soft,
//!   blurred frames where correlation peaks are broad and wander by tenths of a pixel; and a
//!   point-symmetric subject has no colour asymmetry to bias it;
//! - an extended surface running off the frame: (partially whitened) phase correlation, refined
//!   iteratively.

use crate::Image;
use crate::fft::FftConvolver;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Largest window (pixels per side) used for measuring the shift.
const WINDOW: usize = 1024;
/// Border kept around the window so it can be re-sampled at a shifted position.
const MARGIN: usize = 16;
const ITERATIONS: usize = 4;

/// Shift (dx, dy) in pixels applied to the red and blue channels; green is the reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AlignParams {
    pub red: [f32; 2],
    pub blue: [f32; 2],
}

impl AlignParams {
    pub fn is_identity(&self) -> bool {
        self.red == [0.0; 2] && self.blue == [0.0; 2]
    }
}

/// Shifts aligning R and B onto G by their centroids over the subject, for a subject on a
/// dark sky. `None` for mono images or when there is no clear subject.
pub fn estimate_by_centroid(img: &Image) -> Option<AlignParams> {
    let [r, g, b] = img.channels() else {
        return None;
    };
    let (w, h) = (img.width(), img.height());
    // One region for all three channels (from green, widened so every channel's blurred limb
    // is inside), or each centroid would move with its own threshold.
    let smooth = crate::mask::gaussian_approx(g, w, h, 2.0);
    let stride = smooth.len().div_ceil(1 << 18).max(1);
    let mut sample: Vec<f32> = smooth.iter().step_by(stride).copied().collect();
    let n = sample.len();
    let sky = *sample.select_nth_unstable_by(n / 5, f32::total_cmp).1;
    let bright = *sample.select_nth_unstable_by(n * 995 / 1000, f32::total_cmp).1;
    if bright - sky <= 0.0 {
        return None;
    }
    let core: Vec<f32> = smooth.iter().map(|&v| (v > sky + 0.15 * (bright - sky)) as u8 as f32).collect();
    let region: Vec<bool> = crate::mask::gaussian_approx(&core, w, h, 4.0).iter().map(|&v| v > 0.01).collect();
    let centroid = |plane: &[f32]| -> Option<[f64; 2]> {
        // Sky level of this channel, from outside the region.
        let mut outside: Vec<f32> = plane.iter().zip(&region).step_by(stride).filter(|(_, r)| !**r).map(|(v, _)| *v).collect();
        let level = if outside.is_empty() {
            0.0
        } else {
            let mid = outside.len() / 2;
            *outside.select_nth_unstable_by(mid, f32::total_cmp).1
        };
        let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
        for (i, (&v, _)) in plane.iter().zip(&region).enumerate().filter(|(_, (_, r))| **r) {
            let weight = (v - level).max(0.0) as f64;
            sx += weight * (i % w) as f64;
            sy += weight * (i / w) as f64;
            sw += weight;
        }
        (sw > 0.0).then(|| [sx / sw, sy / sw])
    };
    let (cr, cg, cb) = (centroid(r)?, centroid(g)?, centroid(b)?);
    let towards = |c: [f64; 2]| [(cg[0] - c[0]) as f32, (cg[1] - c[1]) as f32];
    Some(AlignParams { red: towards(cr), blue: towards(cb) })
}

/// Measure the shifts that align R and B onto G by phase correlation. `None` for mono images.
pub fn estimate(img: &Image) -> Option<AlignParams> {
    let [r, g, b] = img.channels() else {
        return None;
    };
    let (w, h) = (img.width(), img.height());
    Some(AlignParams { red: measure_shift(g, r, w, h), blue: measure_shift(g, b, w, h) })
}

/// Shift to apply to `moving` so it lines up with `reference` (both `w`×`h`).
///
/// A blurred image has only noise at high frequencies, which widens the correlation peak and
/// biases any single sub-pixel estimate. So the estimate is refined iteratively: shift the
/// moving window by the current estimate, measure the (smaller, less biased) residual, repeat.
pub fn measure_shift(reference: &[f32], moving: &[f32], w: usize, h: usize) -> [f32; 2] {
    let side = WINDOW.min(w.saturating_sub(2 * MARGIN)).min(h.saturating_sub(2 * MARGIN)) & !1;
    if side < 32 {
        return [0.0; 2];
    }
    let outer = side + 2 * MARGIN;
    let (x0, y0) = ((w - outer) / 2, (h - outer) / 2);
    let crop = |plane: &[f32]| -> Vec<f32> { (0..outer).flat_map(|y| plane[(y0 + y) * w + x0..(y0 + y) * w + x0 + outer].iter().copied()).collect() };
    let plan = FftConvolver::plan(side, side);
    let reference = taper(&crop(reference), outer, side);
    let moving = crop(moving);
    let mut total = [0.0f32; 2];
    for _ in 0..ITERATIONS {
        let shifted = shift(&moving, outer, outer, total);
        let d = phase_correlate(&plan, &reference, &taper(&shifted, outer, side), side);
        total = [(total[0] + d[0]).clamp(-15.0, 15.0), (total[1] + d[1]).clamp(-15.0, 15.0)];
        if d[0].abs() < 0.005 && d[1].abs() < 0.005 {
            break;
        }
    }
    total
}

/// Central `side`² of an `outer`² crop, mean-subtracted and tapered with a Hann window so the
/// window edges don't dominate the correlation.
pub(crate) fn taper(crop: &[f32], outer: usize, side: usize) -> Vec<f32> {
    let m = (outer - side) / 2;
    let mut out: Vec<f32> = (0..side).flat_map(|y| crop[(m + y) * outer + m..(m + y) * outer + m + side].iter().copied()).collect();
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    let hann: Vec<f32> = (0..side).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / (side - 1) as f32).cos()).collect();
    for (i, v) in out.iter_mut().enumerate() {
        *v = (*v - mean) * hann[i % side] * hann[i / side];
    }
    out
}

/// Shift to apply to `moving` so it lines up with `reference`.
pub(crate) fn phase_correlate(plan: &FftConvolver, reference: &[f32], moving: &[f32], side: usize) -> [f32; 2] {
    let a = plan.forward(reference);
    let mut cross = plan.forward(moving);
    // Partial whitening (|C|^0.5): pure phase correlation weighs every frequency equally, and
    // in a blurred image the high ones are only noise, which makes the peak weak and erratic.
    cross.par_iter_mut().zip(&a).for_each(|(m, r)| {
        let c = r * m.conj();
        let mag = c.norm();
        *m = if mag > 1e-12 { c / mag.sqrt() } else { Default::default() };
    });
    let surface = plan.inverse(cross);
    let (peak, _) = surface.iter().enumerate().fold((0, f32::MIN), |best, (i, &v)| if v > best.1 { (i, v) } else { best });
    let (px, py) = (peak % side, peak / side);
    let at = |x: isize, y: isize| surface[(y.rem_euclid(side as isize) as usize) * side + x.rem_euclid(side as isize) as usize];
    // Three-point Gaussian fit; its bias shrinks with the offset, which the iterations drive to 0.
    let refine = |m: f32, c: f32, p: f32| {
        let (lm, lc, lp) = (m.max(1e-9).ln(), c.max(1e-9).ln(), p.max(1e-9).ln());
        let d = lm - 2.0 * lc + lp;
        if d < -1e-9 { (0.5 * (lm - lp) / d).clamp(-0.5, 0.5) } else { 0.0 }
    };
    let (ix, iy) = (px as isize, py as isize);
    let c = at(ix, iy);

    let fx = refine(at(ix - 1, iy), c, at(ix + 1, iy));
    let fy = refine(at(ix, iy - 1), c, at(ix, iy + 1));
    let wrap = |p: usize| {
        if p > side / 2 { p as f32 - side as f32 } else { p as f32 }
    };
    [wrap(px) + fx, wrap(py) + fy]
}

/// Apply the shifts (only to colour images).
pub fn apply(img: &Image, p: &AlignParams) -> Image {
    let [r, g, b] = img.channels() else {
        return img.clone();
    };
    let (w, h) = (img.width(), img.height());
    let channels = vec![shift(r, w, h, p.red), g.clone(), shift(b, w, h, p.blue)];
    Image::new(w, h, channels, img.source_bit_depth)
}

/// `out(x, y) = in(x − dx, y − dy)`, separable Catmull–Rom, edges clamped.
pub fn shift(plane: &[f32], w: usize, h: usize, [dx, dy]: [f32; 2]) -> Vec<f32> {
    if dx == 0.0 && dy == 0.0 {
        return plane.to_vec();
    }
    let (ox, wx) = taps(dx);
    let mut rows = vec![0.0f32; w * h];
    rows.par_chunks_mut(w).zip(plane.par_chunks(w)).for_each(|(out, src)| {
        for (x, o) in out.iter_mut().enumerate() {
            *o = (0..4).map(|k| wx[k] * src[(x as isize + ox + k as isize - 1).clamp(0, w as isize - 1) as usize]).sum();
        }
    });
    let (oy, wy) = taps(dy);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (k, &weight) in wy.iter().enumerate() {
            let sy = (y as isize + oy + k as isize - 1).clamp(0, h as isize - 1) as usize;
            for (o, &v) in row.iter_mut().zip(&rows[sy * w..(sy + 1) * w]) {
                *o += weight * v;
            }
        }
    });
    out
}

/// Integer offset and the four Catmull–Rom weights for sampling at `x − d`.
fn taps(d: f32) -> (isize, [f32; 4]) {
    let s = -d;
    let o = s.floor();
    let t = s - o;
    let (t2, t3) = (t * t, t * t * t);
    let weights = [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2];
    (o as isize, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(w: usize, h: usize) -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let r = ((x - 150.0).powi(2) + (y - 120.0).powi(2)).sqrt();
                (if r < 80.0 { 0.6 } else { 0.05 }) + 0.2 * (x * 0.21).sin() * (y * 0.17).cos()
            })
            .collect()
    }

    /// Planet-like scene: a limb-darkened, banded disk on a black sky, with colour.
    fn planet(w: usize, h: usize) -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - 150.0, (i / w) as f32 - 120.0);
                let d = x.hypot(y) / 80.0;
                if d < 1.0 { 0.6 * (1.0 - d * d).powf(0.2) * (0.85 + 0.15 * (y * 0.15).sin()) } else { 0.0 }
            })
            .collect()
    }

    #[test]
    fn centroids_measure_dispersion_on_a_blurred_planet() {
        let (w, h) = (300, 240);
        let blur = |p: Vec<f32>| crate::fft::FftConvolver::new(w, h, &crate::psf::Psf::Gaussian { sigma: 3.0 }.kernel()).convolve(&p);
        let g = blur(planet(w, h));
        let r: Vec<f32> = shift(&g, w, h, [0.4, -1.3]).iter().map(|v| 0.9 * v).collect();
        let b: Vec<f32> = shift(&g, w, h, [-0.2, 0.9]).iter().map(|v| 0.6 * v + 0.002).collect();
        let p = estimate_by_centroid(&Image::new(w, h, vec![r, g, b], 16)).unwrap();
        for (got, want) in p.red.iter().chain(&p.blue).zip([-0.4, 1.3, 0.2, -0.9]) {
            assert!((got - want).abs() < 0.05, "{p:?}");
        }
    }

    #[test]
    fn integer_shift_moves_content() {
        let plane: Vec<f32> = (0..20).map(|v| v as f32).collect();
        let out = shift(&plane, 5, 4, [1.0, 0.0]);
        assert_eq!(&out[0..5], &[0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn measures_subpixel_shifts() {
        let (w, h) = (300, 240);
        let g = scene(w, h);
        let r = shift(&g, w, h, [1.6, -0.7]);
        let b = shift(&g, w, h, [-2.3, 0.4]);
        let img = Image::new(w, h, vec![r, g, b], 16);
        let p = estimate(&img).unwrap();
        for (got, want) in p.red.iter().chain(&p.blue).zip([-1.6, 0.7, 2.3, -0.4]) {
            assert!((got - want).abs() < 0.15, "{p:?}");
        }
        // Applying the estimate realigns the channels.
        let fixed = apply(&img, &p);
        let [r2, g2, _] = fixed.channels() else { panic!() };
        let err: f32 = r2.iter().zip(g2).skip(w * 20).take(w * 200).map(|(a, b)| (a - b).abs()).sum::<f32>() / (w * 200) as f32;
        assert!(err < 0.01, "residual {err}");
    }

    #[test]
    fn no_spurious_shift_on_blurred_noisy_channels() {
        let (w, h) = (300, 240);
        let sharp = scene(w, h);
        let blurred = crate::fft::FftConvolver::new(w, h, &crate::psf::Psf::Gaussian { sigma: 2.5 }.kernel()).convolve(&sharp);
        let mut seed = 99u32;
        let mut noisy = |gain: f32| -> Vec<f32> {
            blurred
                .iter()
                .map(|v| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    gain * v + 0.01 * (seed as f32 / u32::MAX as f32 - 0.5)
                })
                .collect()
        };
        let img = Image::new(w, h, vec![noisy(1.0), noisy(0.9), noisy(0.7)], 16);
        let p = estimate(&img).unwrap();
        // Independent noise on blurred channels limits the precision to about a tenth of a pixel.
        assert!(p.red.iter().chain(&p.blue).all(|v| v.abs() < 0.15), "{p:?}");
    }
}
