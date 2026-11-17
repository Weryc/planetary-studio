//! The stacker, as AutoStakkert works: grade the frames, keep the best, build a reference
//! from the very best, align each kept frame on it (globally, then on a grid of alignment
//! points whose shifts are blended into a smooth warp), resample with Catmull–Rom and average.
//! Nothing here knows the true tilts: the residual errors are whatever the measurement makes.
use crate::camera::Sensor;
use ps_core::align::{measure_shift, shift};
use rayon::prelude::*;

/// Alignment point: patch side and spacing (sensor px).
const AP: usize = 64;
const AP_MARGIN: usize = 16;
const AP_SPACING: usize = 40;
/// Coarse grid of the warp field (px).
const STEP: usize = 8;

pub struct Stacked {
    pub planes: Vec<Vec<f32>>,
    /// The same with the even and the odd frames only: half their difference is the stack's
    /// random noise, as two half stacks of a real recording give it.
    pub halves: [Vec<Vec<f32>>; 2],
    pub kept: usize,
    pub points: usize,
    /// Mean |local shift| over the alignment points (px): how much the warp did.
    pub mean_local: f32,
}

pub fn luminance(planes: &[Vec<f32>]) -> Vec<f32> {
    if planes.len() == 1 {
        return planes[0].clone();
    }
    planes[0].iter().zip(&planes[1]).zip(&planes[2]).map(|((r, g), b)| 0.25 * r + 0.5 * g + 0.25 * b).collect()
}

/// Sharpness as a stacker grades it: gradient energy of the lightly smoothed luminance over the
/// lit part (smoothing keeps the noise from counting as detail).
pub fn quality(raw: &[u16], sensor: &Sensor) -> f32 {
    let (w, h) = (sensor.w / 2, sensor.h / 2);
    // 2×2 bins: one full Bayer cell, so colour doesn't matter.
    let half: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = (2 * (i % w), 2 * (i / w));
            let s = sensor.w;
            (raw[y * s + x] as f32 + raw[y * s + x + 1] as f32 + raw[(y + 1) * s + x] as f32 + raw[(y + 1) * s + x + 1] as f32) / 4.0
        })
        .collect();
    let smooth = ps_core::mask::gaussian_approx(&half, w, h, 1.2);
    let top = smooth.iter().copied().fold(0.0f32, f32::max);
    let mut sum = 0.0f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            if smooth[i] > 0.15 * top {
                let gx = smooth[i + 1] - smooth[i - 1];
                let gy = smooth[i + w] - smooth[i - w];
                sum += (gx * gx + gy * gy) as f64;
            }
        }
    }
    (sum / (top as f64).powi(2).max(1e-9)) as f32
}

fn cubic(t: f32) -> [f32; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [0.5 * (-t3 + 2.0 * t2 - t), 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0), 0.5 * (-3.0 * t3 + 4.0 * t2 + t), 0.5 * (t3 - t2)]
}

/// Catmull–Rom sample at (x, y), edges clamped.
pub fn sample(p: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (wx, wy) = (cubic(x - x0), cubic(y - y0));
    let mut acc = 0.0;
    for (j, wyj) in wy.iter().enumerate() {
        let sy = (y0 as isize + j as isize - 1).clamp(0, h as isize - 1) as usize;
        let row = &p[sy * w..(sy + 1) * w];
        let mut r = 0.0;
        for (i, wxi) in wx.iter().enumerate() {
            r += wxi * row[(x0 as isize + i as isize - 1).clamp(0, w as isize - 1) as usize];
        }
        acc += wyj * r;
    }
    acc
}

/// Centre of light of the lit part (above 15 % of the brightest): a first, coarse alignment
/// that works however far the planet has wandered, as AutoStakkert's "planet" mode does.
fn centroid(l: &[f32], w: usize) -> [f32; 2] {
    let top = l.iter().copied().fold(0.0f32, f32::max);
    let (mut sx, mut sy, mut s) = (0.0f64, 0.0f64, 0.0f64);
    for (i, &v) in l.iter().enumerate() {
        if v > 0.15 * top {
            sx += (v * (i % w) as f32) as f64;
            sy += (v * (i / w) as f32) as f64;
            s += v as f64;
        }
    }
    [(sx / s.max(1e-12)) as f32, (sy / s.max(1e-12)) as f32]
}

/// Global shift taking `moving` onto `reference`: centroids first, then phase correlation.
fn global_shift(reference: &[f32], ref_centre: Option<[f32; 2]>, moving: &[f32], w: usize, h: usize) -> [f32; 2] {
    let coarse = ref_centre.map_or([0.0; 2], |r| {
        let c = centroid(moving, w);
        [(r[0] - c[0]).round(), (r[1] - c[1]).round()]
    });
    let fine = measure_shift(reference, &shift(moving, w, h, coarse), w, h);
    [coarse[0] + fine[0], coarse[1] + fine[1]]
}

fn patch(p: &[f32], w: usize, cx: usize, cy: usize, side: usize) -> Vec<f32> {
    let (x0, y0) = (cx - side / 2, cy - side / 2);
    (0..side).flat_map(|y| p[(y0 + y) * w + x0..(y0 + y) * w + x0 + side].iter().copied()).collect()
}

/// Stack the frames at `order` (best first), `keep` of them.
/// `surface`: a field filled by the subject (the Moon), aligned by correlation alone.
pub fn stack(raws: &[Vec<u16>], order: &[usize], keep: usize, sensor: &Sensor, surface: bool) -> Stacked {
    let (w, h) = (sensor.w, sensor.h);
    let kept = &order[..keep.min(order.len()).max(1)];
    // Reference: the best frame, then the average of the best 10 % aligned on it.
    let best = luminance(&sensor.develop(&raws[kept[0]]));
    let n_ref = (kept.len() / 10).max(5).min(kept.len());
    let refs: Vec<Vec<f32>> = kept[..n_ref]
        .par_iter()
        .map(|&i| {
            let l = luminance(&sensor.develop(&raws[i]));
            let s = global_shift(&best, (!surface).then(|| centroid(&best, w)), &l, w, h);
            shift(&l, w, h, s)
        })
        .collect();
    let reference: Vec<f32> = (0..w * h).map(|i| refs.iter().map(|r| r[i]).sum::<f32>() / n_ref as f32).collect();
    let ref_centre = (!surface).then(|| centroid(&reference, w));
    // Alignment points: on the lit subject, whole patch inside the frame.
    let top = reference.iter().copied().fold(0.0f32, f32::max);
    let outer = AP + 2 * AP_MARGIN;
    let mut points = Vec::new();
    for cy in (outer / 2..h - outer / 2).step_by(AP_SPACING) {
        for cx in (outer / 2..w - outer / 2).step_by(AP_SPACING) {
            let p = patch(&reference, w, cx, cy, AP);
            let mean = p.iter().sum::<f32>() / p.len() as f32;
            if mean > 0.12 * top {
                points.push((cx, cy, patch(&reference, w, cx, cy, outer)));
            }
        }
    }
    let (gw, gh) = (w.div_ceil(STEP) + 1, h.div_ceil(STEP) + 1);
    let sigma = 0.8 * AP_SPACING as f32;
    let channels = if sensor.bayer { 3 } else { 1 };
    let zero = || (vec![vec![vec![0.0f32; w * h]; channels]; 3], 0.0f32, [0usize; 3]);
    let (sums, local_sum, count) = kept
        .par_iter()
        .map(|&i| {
            let planes = sensor.develop(&raws[i]);
            let lum = luminance(&planes);
            let g = global_shift(&reference, ref_centre, &lum, w, h);
            let lum_g = shift(&lum, w, h, g);
            let local: Vec<(f32, f32, [f32; 2])> = points
                .iter()
                .filter_map(|(cx, cy, rp)| {
                    let d = measure_shift(rp, &patch(&lum_g, w, *cx, *cy, outer), outer, outer);
                    (d[0].hypot(d[1]) < 6.0).then_some((*cx as f32, *cy as f32, d))
                })
                .collect();
            let mean_local = local.iter().map(|l| l.2[0].hypot(l.2[1])).sum::<f32>() / local.len().max(1) as f32;
            // The warp on the coarse grid: Gaussian-weighted blend of the points' shifts, fading
            // to the global shift alone where there are none.
            let field: Vec<[f32; 2]> = (0..gw * gh)
                .map(|k| {
                    let (x, y) = (((k % gw) * STEP) as f32, ((k / gw) * STEP) as f32);
                    let (mut sx, mut sy, mut sw) = (0.0f32, 0.0f32, 0.05f32);
                    for (px, py, d) in &local {
                        let r2 = (x - px).powi(2) + (y - py).powi(2);
                        if r2 < 9.0 * sigma * sigma {
                            let wgt = (-r2 / (2.0 * sigma * sigma)).exp();
                            sx += wgt * d[0];
                            sy += wgt * d[1];
                            sw += wgt;
                        }
                    }
                    [g[0] + sx / sw, g[1] + sy / sw]
                })
                .collect();
            let at = |x: usize, y: usize| -> [f32; 2] {
                let (fx, fy) = (x as f32 / STEP as f32, y as f32 / STEP as f32);
                let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
                let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
                let f = |xx: usize, yy: usize| field[yy.min(gh - 1) * gw + xx.min(gw - 1)];
                let (a, b, c, d) = (f(x0, y0), f(x0 + 1, y0), f(x0, y0 + 1), f(x0 + 1, y0 + 1));
                std::array::from_fn(|k| (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty)
            };
            let aligned = planes
                .iter()
                .map(|p| {
                    let mut out = vec![0.0f32; w * h];
                    for y in 0..h {
                        for x in 0..w {
                            let s = at(x, y);
                            out[y * w + x] = sample(p, w, h, x as f32 - s[0], y as f32 - s[1]);
                        }
                    }
                    out
                })
                .collect();
            (aligned, mean_local, i % 2)
        })
        .fold(zero, |mut acc, (aligned, ml, parity): (Vec<Vec<f32>>, f32, usize)| {
            for slot in [0, 1 + parity] {
                for (a, p) in acc.0[slot].iter_mut().zip(&aligned) {
                    a.iter_mut().zip(p).for_each(|(s, v)| *s += v);
                }
                acc.2[slot] += 1;
            }
            (acc.0, acc.1 + ml, acc.2)
        })
        .reduce(zero, |mut a, b| {
            for slot in 0..3 {
                for (x, y) in a.0[slot].iter_mut().zip(&b.0[slot]) {
                    x.iter_mut().zip(y).for_each(|(s, v)| *s += v);
                }
                a.2[slot] += b.2[slot];
            }
            (a.0, a.1 + b.1, a.2)
        });
    let mut stacks: Vec<Vec<Vec<f32>>> = sums.into_iter().zip(count).map(|(p, n)| p.into_iter().map(|c| c.into_iter().map(|v| v / n.max(1) as f32).collect()).collect()).collect();
    let (odd, even, planes) = (stacks.pop().unwrap(), stacks.pop().unwrap(), stacks.pop().unwrap());
    Stacked { planes, halves: [even, odd], kept: count[0], points: points.len(), mean_local: local_sum / count[0] as f32 }
}
