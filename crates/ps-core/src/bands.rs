//! The belts and zones of a banded planet (Jupiter, Saturn's globe), measured on the image: how
//! wide the dark belts and the bright zones are, in pixels. Shown with the learned sharpening to
//! say which sizes of detail the curve lifts (a band of width `w` is mostly detail of period
//! `2w`).
//!
//! The brightness is averaged across the central strip of the disk (away from the limb), along
//! the direction from pole to pole; the limb darkening is divided out with a smooth fit (a
//! quartic), and the runs darker and brighter than that are the belts and the zones. The
//! direction of the bands is Saturn's rings'; for a disk it is searched: the one with the most
//! contrast across it.

use crate::mask::Ellipse;

/// Widths in pixels (pole to pole direction), sorted, of the belts and zones that were clear.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bands {
    pub dark: Vec<f32>,
    pub bright: Vec<f32>,
}

/// The bands of the globe `globe` on the luminance `lum`: with `search`, in whichever direction
/// shows them best, else along `globe.angle` (the equator). `None` when the globe is too small
/// or shows no clear banding.
pub fn measure(lum: &[f32], w: usize, h: usize, globe: Ellipse, search: bool) -> Option<Bands> {
    if globe.ry < 40.0 {
        return None;
    }
    let angles: Vec<f32> = if search { (0..36).map(|k| (k as f32 * 5.0).to_radians()).collect() } else { vec![globe.angle] };
    let rel = angles
        .into_iter()
        .filter_map(|angle| profile(lum, w, h, Ellipse { angle, ..globe }).map(|p| relative(&p)))
        .max_by(|a, b| rms(a).total_cmp(&rms(b)))?;
    let spread = rms(&rel);
    if spread < 0.01 {
        return None;
    }
    let threshold = 0.3 * spread;
    let mut bands = Bands::default();
    let mut start = 0;
    let sign = |r: f32| if r > threshold { 1 } else if r < -threshold { -1 } else { 0 };
    for i in 1..=rel.len() {
        if i < rel.len() && sign(rel[i]) == sign(rel[start]) {
            continue;
        }
        // A whole run, not cut by the ends of the strip, at least 3 px wide.
        let width = (i - start) as f32;
        if start > 0 && i < rel.len() && width >= 3.0 {
            match sign(rel[start]) {
                -1 => bands.dark.push(width),
                1 => bands.bright.push(width),
                _ => {}
            }
        }
        start = i;
    }
    bands.dark.sort_by(f32::total_cmp);
    bands.bright.sort_by(f32::total_cmp);
    (bands.dark.len() >= 2).then_some(bands)
}

/// Mean brightness by distance from the equator (1 px bins) on the central strip, out to 80 %
/// of the radius from pole to pole.
fn profile(lum: &[f32], w: usize, h: usize, globe: Ellipse) -> Option<Vec<f32>> {
    let (along, across) = (globe.rx, globe.ry);
    let half = (0.8 * across).floor() as usize;
    let (mut sum, mut n) = (vec![0.0f64; 2 * half + 1], vec![0u32; 2 * half + 1]);
    let (s, c) = globe.angle.sin_cos();
    let reach = along.max(across).ceil() as isize;
    let (x0, y0) = (globe.cx.round() as isize, globe.cy.round() as isize);
    for y in (y0 - reach).max(0)..(y0 + reach).min(h as isize) {
        for x in (x0 - reach).max(0)..(x0 + reach).min(w as isize) {
            let (dx, dy) = (x as f32 - globe.cx, y as f32 - globe.cy);
            let (u, v) = (dx * c + dy * s, -dx * s + dy * c);
            if u.abs() > 0.3 * along {
                continue;
            }
            let bin = v.round() as isize + half as isize;
            if (0..sum.len() as isize).contains(&bin) {
                sum[bin as usize] += lum[y as usize * w + x as usize] as f64;
                n[bin as usize] += 1;
            }
        }
    }
    if n.iter().any(|&k| k < 3) {
        return None;
    }
    Some(sum.iter().zip(&n).map(|(s, &k)| (s / k as f64) as f32).collect())
}

/// The profile over its smooth trend (a least-squares quartic: the limb darkening), minus 1,
/// lightly smoothed against the noise.
fn relative(p: &[f32]) -> Vec<f32> {
    let n = p.len();
    let x = |i: usize| 2.0 * i as f64 / (n - 1).max(1) as f64 - 1.0;
    let mut a = [[0.0f64; 5]; 5];
    let mut b = [0.0f64; 5];
    for (i, &v) in p.iter().enumerate() {
        let powers: [f64; 5] = std::array::from_fn(|k| x(i).powi(k as i32));
        for r in 0..5 {
            for c in 0..5 {
                a[r][c] += powers[r] * powers[c];
            }
            b[r] += powers[r] * v as f64;
        }
    }
    let coef = solve5(a, b);
    let trend = |i: usize| (0..5).map(|k| coef[k] * x(i).powi(k as i32)).sum::<f64>() as f32;
    let rel: Vec<f32> = p.iter().enumerate().map(|(i, &v)| v / trend(i).max(1e-6) - 1.0).collect();
    smooth(&rel, 1.0)
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|r| r * r).sum::<f32>() / v.len().max(1) as f32).sqrt()
}

/// Gauss–Jordan with partial pivoting for the 5×5 normal equations.
fn solve5(mut a: [[f64; 5]; 5], mut b: [f64; 5]) -> [f64; 5] {
    for c in 0..5 {
        let p = (c..5).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap_or(c);
        a.swap(c, p);
        b.swap(c, p);
        let d = a[c][c];
        if d.abs() < 1e-12 {
            continue;
        }
        let pivot = a[c];
        for r in 0..5 {
            if r != c {
                let f = a[r][c] / d;
                a[r][c..].iter_mut().zip(&pivot[c..]).for_each(|(x, p)| *x -= f * p);
                b[r] -= f * b[c];
            }
        }
    }
    std::array::from_fn(|i| if a[i][i].abs() < 1e-12 { 0.0 } else { b[i] / a[i][i] })
}

/// Gaussian smoothing of a profile (edges mirrored).
fn smooth(v: &[f32], sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f32> = (-r..=r).map(|k| (-(k * k) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let n = v.len() as isize;
    (0..n)
        .map(|i| {
            let (mut s, mut wsum) = (0.0, 0.0);
            for (k, &kw) in (-r..=r).zip(&kernel) {
                let mut j = i + k;
                if j < 0 {
                    j = -j;
                }
                if j >= n {
                    j = 2 * n - 2 - j;
                }
                let j = j.clamp(0, n - 1) as usize;
                s += kw * v[j];
                wsum += kw;
            }
            s / wsum
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A disk with alternating belts 12 px and zones 20 px wide is read back with those widths.
    #[test]
    fn it_reads_back_the_widths_of_synthetic_bands() {
        let (w, h) = (400, 400);
        let (cx, cy, r) = (200.0f32, 200.0f32, 150.0f32);
        let lum: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - cx, (i / w) as f32 - cy);
                let d = (x * x + y * y).sqrt() / r;
                if d >= 1.0 {
                    return 0.0;
                }
                let phase = (y + 300.0).rem_euclid(32.0);
                let band = if phase < 12.0 { 0.75 } else { 1.0 };
                band * (0.4 + 0.6 * (1.0 - d * d).sqrt())
            })
            .collect();
        let b = measure(&lum, w, h, Ellipse { cx, cy, rx: r, ry: r, angle: 0.0 }, true).expect("bands");
        let median = |v: &[f32]| v[v.len() / 2];
        assert!((median(&b.dark) - 12.0).abs() <= 3.0, "{:?}", b.dark);
        assert!((median(&b.bright) - 20.0).abs() <= 3.0, "{:?}", b.bright);
    }
}
