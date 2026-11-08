//! Colour adjustments: saturation, vibrance, white balance, and the "Mineral Moon" tools
//! (strong saturation with chroma noise reduction, hue rotation, tone curve and LRGB).
//!
//! Colour is split into luminance Y (Rec. 709) and chroma C − Y, so saturation and chroma
//! smoothing never change brightness.

use crate::Image;
use crate::color::{LUMA, luma};
use crate::conv::convolve_separable;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorParams {
    /// 1 = unchanged, 0 = greyscale.
    pub saturation: f32,
    /// Saturates muted colours more than already vivid ones; 0 = off.
    pub vibrance: f32,
    /// Per-channel gains (white balance).
    pub balance: [f32; 3],
}

impl Default for ColorParams {
    fn default() -> Self {
        Self { saturation: 1.0, vibrance: 0.0, balance: [1.0; 3] }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MineralParams {
    /// Saturation multiplier (Mineral Moon images typically need 3–8×).
    pub saturation: f32,
    /// Chroma smoothing: number of starlet scales averaged away (0 = off). Strong saturation
    /// amplifies colour noise, so this goes hand in hand with it.
    pub chroma_smoothing: u32,
    /// Hue rotation in degrees.
    pub hue: f32,
    /// Tone curve on luminance: points (x, y) in 0‥1, sorted by x, including both ends.
    pub curve: Vec<[f32; 2]>,
    /// LRGB: how much of the external luminance replaces the colour image's own (0‥1).
    pub lrgb_amount: f32,
}

impl Default for MineralParams {
    fn default() -> Self {
        Self { saturation: 4.0, chroma_smoothing: 3, hue: 0.0, curve: vec![[0.0, 0.0], [1.0, 1.0]], lrgb_amount: 1.0 }
    }
}

impl ColorParams {
    pub fn is_neutral(&self) -> bool {
        *self == Self::default()
    }
}

impl MineralParams {
    /// Settings that change nothing (an external luminance still does, see the pipeline).
    pub fn neutral() -> Self {
        Self { saturation: 1.0, chroma_smoothing: 0, hue: 0.0, ..Default::default() }
    }

    pub fn is_neutral(&self) -> bool {
        self.saturation == 1.0 && self.chroma_smoothing == 0 && self.hue == 0.0 && self.curve == Self::default().curve
    }
}

pub fn apply_color(img: &Image, p: &ColorParams) -> Image {
    let [r, g, b] = img.channels() else {
        return img.clone();
    };
    let (w, h) = (img.width(), img.height());
    let mut out = vec![vec![0.0f32; w * h]; 3];
    let [or, og, ob] = out.as_mut_slice() else { unreachable!() };
    or.par_iter_mut().zip(og.par_iter_mut()).zip(ob.par_iter_mut()).enumerate().for_each(|(i, ((ro, go), bo))| {
        let c = [r[i] * p.balance[0], g[i] * p.balance[1], b[i] * p.balance[2]];
        let y = luma(c);
        let (max, min) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
        let sat = if max > 1e-6 { (max - min) / max } else { 0.0 };
        let k = shadow_safe(p.saturation * (1.0 + p.vibrance * (1.0 - sat)), y);
        *ro = y + k * (c[0] - y);
        *go = y + k * (c[1] - y);
        *bo = y + k * (c[2] - y);
    });
    Image::new(w, h, out, img.source_bit_depth)
}

/// Grey-world white balance on the subject (pixels brighter than a fifth of the 99th
/// percentile), with green as the reference.
pub fn auto_balance(img: &Image) -> [f32; 3] {
    let [r, g, b] = img.channels() else {
        return [1.0; 3];
    };
    let stride = r.len().div_ceil(1 << 18).max(1);
    let mut lum: Vec<f32> = (0..r.len()).step_by(stride).map(|i| luma([r[i], g[i], b[i]])).collect();
    let k = (lum.len() * 99 / 100).min(lum.len().saturating_sub(1));
    let threshold = 0.2 * *lum.select_nth_unstable_by(k, f32::total_cmp).1;
    let mut sums = [0.0f64; 3];
    for i in (0..r.len()).step_by(stride) {
        if luma([r[i], g[i], b[i]]) > threshold {
            sums[0] += r[i] as f64;
            sums[1] += g[i] as f64;
            sums[2] += b[i] as f64;
        }
    }
    if sums.iter().any(|&s| s <= 0.0) {
        return [1.0; 3];
    }
    [(sums[1] / sums[0]) as f32, 1.0, (sums[1] / sums[2]) as f32]
}

/// Mineral Moon finishing. `luminance` (same size, 0‥1) enables LRGB.
pub fn apply_mineral(img: &Image, p: &MineralParams, luminance: Option<&[f32]>) -> Image {
    let (w, h) = (img.width(), img.height());
    let lut = curve_lut(&p.curve);
    let [r, g, b] = img.channels() else {
        // Mono: only the tone curve applies.
        let plane = img.channels()[0].par_iter().map(|&v| lookup(&lut, v)).collect();
        return Image::new(w, h, vec![plane], img.source_bit_depth);
    };
    let n = w * h;
    let y: Vec<f32> = (0..n).into_par_iter().map(|i| luma([r[i], g[i], b[i]])).collect();
    let mut cr: Vec<f32> = r.par_iter().zip(&y).map(|(c, y)| c - y).collect();
    let mut cb: Vec<f32> = b.par_iter().zip(&y).map(|(c, y)| c - y).collect();
    if p.chroma_smoothing > 0 {
        cr = smooth_plane(&cr, w, h, p.chroma_smoothing);
        cb = smooth_plane(&cb, w, h, p.chroma_smoothing);
    }
    let (sin, cos) = p.hue.to_radians().sin_cos();
    let mut out = vec![vec![0.0f32; n]; 3];
    let [or, og, ob] = out.as_mut_slice() else { unreachable!() };
    or.par_iter_mut().zip(og.par_iter_mut()).zip(ob.par_iter_mut()).enumerate().for_each(|(i, ((ro, go), bo))| {
        // Rotate the (Cb, Cr) chroma vector, then scale it.
        let (cb_i, cr_i) = (cb[i] * cos - cr[i] * sin, cb[i] * sin + cr[i] * cos);
        let k = shadow_safe(p.saturation, y[i]);
        let (cb_i, cr_i) = (cb_i * k, cr_i * k);
        let mut yy = lookup(&lut, y[i]);
        if let Some(l) = luminance {
            yy += p.lrgb_amount * (l[i] - yy);
        }
        let rr = yy + cr_i;
        let bb = yy + cb_i;
        *ro = rr;
        *bo = bb;
        *go = (yy - LUMA[0] * rr - LUMA[2] * bb) / LUMA[1];
    });
    Image::new(w, h, out, img.source_bit_depth)
}

/// A saturation boost `k` for a pixel of luminance `y`, faded out in the deepest shadows: the
/// colour of a nearly black sky is only noise, and boosting it gives coloured dots.
fn shadow_safe(k: f32, y: f32) -> f32 {
    if k <= 1.0 {
        return k;
    }
    let t = ((y - 0.01) / 0.05).clamp(0.0, 1.0);
    1.0 + (k - 1.0) * t * t * (3.0 - 2.0 * t)
}

/// Low-pass: the B3 starlet smooth plane after `levels` scales.
pub(crate) fn smooth_plane(plane: &[f32], w: usize, h: usize, levels: u32) -> Vec<f32> {
    let taps = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let steps: Vec<usize> = (0..levels.min(8)).map(|j| 1 << j).collect();
    if let Some((_, smooth)) = crate::gpu::for_plane(w, h, crate::gpu::Work::Light).and_then(|g| g.starlet(plane, w, h, &taps, &steps, false).ok()) {
        return smooth;
    }
    let mut current = plane.to_vec();
    let mut next = vec![0.0f32; plane.len()];
    let mut tmp = vec![0.0f32; plane.len()];
    for j in 0..levels.min(8) {
        convolve_separable(&current, &mut next, &mut tmp, w, h, &taps, 1 << j);
        std::mem::swap(&mut current, &mut next);
    }
    current
}

const LUT_SIZE: usize = 4096;

/// Monotone cubic (Fritsch–Carlson) interpolation through the curve points, tabulated.
pub fn curve_lut(points: &[[f32; 2]]) -> Vec<f32> {
    let mut pts: Vec<[f32; 2]> = points.to_vec();
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-6);
    if pts.len() < 2 {
        return (0..LUT_SIZE).map(|i| i as f32 / (LUT_SIZE - 1) as f32).collect();
    }
    let n = pts.len();
    let delta: Vec<f32> = pts.windows(2).map(|p| (p[1][1] - p[0][1]) / (p[1][0] - p[0][0])).collect();
    let mut m = vec![0.0f32; n];
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for i in 1..n - 1 {
        m[i] = if delta[i - 1] * delta[i] <= 0.0 { 0.0 } else { (delta[i - 1] + delta[i]) / 2.0 };
    }
    for i in 0..n - 1 {
        if delta[i] == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let (a, b) = (m[i] / delta[i], m[i + 1] / delta[i]);
        let s = a * a + b * b;
        if s > 9.0 {
            let t = 3.0 / s.sqrt();
            m[i] = t * a * delta[i];
            m[i + 1] = t * b * delta[i];
        }
    }
    (0..LUT_SIZE)
        .map(|k| {
            let x = k as f32 / (LUT_SIZE - 1) as f32;
            if x <= pts[0][0] {
                return pts[0][1];
            }
            if x >= pts[n - 1][0] {
                return pts[n - 1][1];
            }
            let i = pts.windows(2).position(|p| x <= p[1][0]).unwrap();
            let (x0, x1, y0, y1) = (pts[i][0], pts[i + 1][0], pts[i][1], pts[i + 1][1]);
            let hh = x1 - x0;
            let t = (x - x0) / hh;
            let (t2, t3) = (t * t, t * t * t);
            (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * hh * m[i] + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * hh * m[i + 1]
        })
        .collect()
}

pub fn lookup(lut: &[f32], v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;
    let i = (x as usize).min(LUT_SIZE - 2);
    let t = x - i as f32;
    lut[i] * (1.0 - t) + lut[i + 1] * t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(r: f32, g: f32, b: f32) -> Image {
        Image::new(1, 1, vec![vec![r], vec![g], vec![b]], 16)
    }

    fn px(img: &Image) -> [f32; 3] {
        [img.channels()[0][0], img.channels()[1][0], img.channels()[2][0]]
    }

    #[test]
    fn saturation_keeps_luminance() {
        let img = rgb(0.6, 0.4, 0.3);
        let out = apply_color(&img, &ColorParams { saturation: 2.0, ..Default::default() });
        assert!((luma(px(&out)) - luma(px(&img))).abs() < 1e-6);
        assert!(px(&out)[0] > 0.6);
        let grey = apply_color(&img, &ColorParams { saturation: 0.0, ..Default::default() });
        let p = px(&grey);
        assert!((p[0] - p[1]).abs() < 1e-6 && (p[1] - p[2]).abs() < 1e-6);
    }

    #[test]
    fn identity_curve_and_neutral_mineral_change_nothing() {
        let img = rgb(0.6, 0.4, 0.3);
        let p = MineralParams { saturation: 1.0, chroma_smoothing: 0, ..Default::default() };
        let out = apply_mineral(&img, &p, None);
        for (a, b) in px(&out).iter().zip(px(&img)) {
            assert!((a - b).abs() < 1e-4, "{:?}", px(&out));
        }
    }

    #[test]
    fn curve_passes_through_points_and_is_monotone() {
        let lut = curve_lut(&[[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]]);
        assert!((lookup(&lut, 0.5) - 0.7).abs() < 1e-3);
        assert!(lut.windows(2).all(|w| w[1] >= w[0] - 1e-6));
    }

    #[test]
    fn lrgb_replaces_luminance() {
        let img = rgb(0.6, 0.4, 0.3);
        let p = MineralParams { saturation: 1.0, chroma_smoothing: 0, ..Default::default() };
        let out = apply_mineral(&img, &p, Some(&[0.8]));
        assert!((luma(px(&out)) - 0.8).abs() < 1e-4);
    }

    #[test]
    fn saturation_leaves_the_black_sky_alone() {
        let sky = rgb(0.001, 0.0015, 0.004);
        let out = apply_color(&sky, &ColorParams { saturation: 2.0, ..Default::default() });
        assert_eq!(px(&out), px(&sky));
    }

    #[test]
    fn auto_balance_neutralises_a_cast() {
        let n = 100;
        let img = Image::new(n, 1, vec![vec![0.5; n], vec![0.4; n], vec![0.25; n]], 16);
        let gains = auto_balance(&img);
        let out = apply_color(&img, &ColorParams { balance: gains, ..Default::default() });
        let p = px(&out);
        assert!((p[0] - p[1]).abs() < 1e-4 && (p[2] - p[1]).abs() < 1e-4, "{p:?}");
    }
}
