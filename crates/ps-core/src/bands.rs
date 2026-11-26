//! What a planet shows, measured on the image: on a banded globe (Jupiter, Saturn) how wide the
//! dark belts and the bright zones are and how much of the detail breaks across them (storms,
//! festoons, ovals); Jupiter's Great Red Spot; Saturn's rings and how deep the Cassini division
//! comes out. Shown with the learned sharpening and used by its model (a band of width `w` is
//! mostly detail of period `2w`).
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
    /// Direction of the bands (radians).
    pub angle: f32,
    /// Share of the medium-size detail that runs across the bands rather than along them
    /// (0 = bands only; storms, festoons and ovals raise it).
    pub storms: f32,
}

/// Saturn's rings as measured: outer semi-major axis (px) and how dark the Cassini division
/// comes out against the A and B rings beside it (0 = invisible, 1 = black).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rings {
    pub outer: f32,
    pub cassini: f32,
}

/// The bands of the globe `globe` on the luminance `lum`: with `search`, in whichever direction
/// shows them best, else along `globe.angle` (the equator). `None` when the globe is too small
/// or shows no clear banding.
pub fn measure(lum: &[f32], w: usize, h: usize, globe: Ellipse, search: bool) -> Option<Bands> {
    if globe.ry < 40.0 {
        return None;
    }
    let angles: Vec<f32> = if search { (0..36).map(|k| (k as f32 * 5.0).to_radians()).collect() } else { vec![globe.angle] };
    let (angle, rel) = angles
        .into_iter()
        .filter_map(|angle| profile(lum, w, h, Ellipse { angle, ..globe }).map(|p| (angle, relative(&p))))
        .max_by(|a, b| rms(&a.1).total_cmp(&rms(&b.1)))?;
    let spread = rms(&rel);
    if spread < 0.01 {
        return None;
    }
    let threshold = 0.3 * spread;
    let mut bands = Bands { angle, storms: across_share(lum, w, h, Ellipse { angle, ..globe }), ..Default::default() };
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

/// How much of the medium-size detail (a band-pass between a 60th and a 20th of the globe)
/// varies along the bands rather than across them, on the central part of the disk.
fn across_share(lum: &[f32], w: usize, h: usize, globe: Ellipse) -> f32 {
    let r = globe.ry;
    let fine = crate::mask::gaussian_approx(lum, w, h, (r / 60.0).max(1.0));
    let coarse = crate::mask::gaussian_approx(lum, w, h, (r / 20.0).max(2.0));
    let band: Vec<f32> = fine.iter().zip(&coarse).map(|(a, b)| a - b).collect();
    let (s, c) = globe.angle.sin_cos();
    let (mut along, mut across) = (0.0f64, 0.0f64);
    let reach = (0.6 * r) as isize;
    let (x0, y0) = (globe.cx.round() as isize, globe.cy.round() as isize);
    for y in (y0 - reach).max(1)..(y0 + reach).min(h as isize - 1) {
        for x in (x0 - reach).max(1)..(x0 + reach).min(w as isize - 1) {
            let (dx, dy) = (x as f32 - globe.cx, y as f32 - globe.cy);
            if dx.hypot(dy) > 0.6 * r {
                continue;
            }
            let i = y as usize * w + x as usize;
            let (gx, gy) = (0.5 * (band[i + 1] - band[i - 1]), 0.5 * (band[i + w] - band[i - w]));
            // Along the bands (u) and across them (v).
            let (gu, gv) = (gx * c + gy * s, -gx * s + gy * c);
            along += (gu * gu) as f64;
            across += (gv * gv) as f64;
        }
    }
    (along / (along + across).max(1e-30)) as f32
}

/// Jupiter's Great Red Spot: the reddest compact oval on the disk (belts can be reddish too, but
/// they are long). Its diameter in pixels, `None` on a mono image or when nothing stands out.
pub fn red_spot(img: &crate::Image, cx: f32, cy: f32, radius: f32) -> Option<f32> {
    let [r, g, b] = img.channels() else { return None };
    let w = img.width();
    let h = img.height();
    let inside = |x: usize, y: usize| (x as f32 - cx).hypot(y as f32 - cy) < 0.85 * radius;
    let redness = |i: usize| (r[i] - b[i]) / (r[i] + g[i] + b[i]).max(1e-4);
    let mut values: Vec<f32> = (0..w * h).filter(|&i| inside(i % w, i / w)).map(redness).collect();
    if values.len() < 500 {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let (median, top) = (values[values.len() / 2], values[values.len() * 995 / 1000]);
    // The spot stands clearly above the disk's usual redness.
    let threshold = median + 0.6 * (top - median);
    if top - median < 0.03 {
        return None;
    }
    let mask: Vec<bool> = (0..w * h).map(|i| inside(i % w, i / w) && redness(i) > threshold).collect();
    let mut best: Option<(f32, f32)> = None;
    for comp in crate::mask::components(&mask, w, h) {
        if comp.len() < 12 {
            continue;
        }
        let n = comp.len() as f32;
        let (mx, my) = comp.iter().fold((0.0, 0.0), |(sx, sy), &i| (sx + (i % w) as f32, sy + (i / w) as f32));
        let (mx, my) = (mx / n, my / n);
        let (mut xx, mut yy, mut xy) = (0.0f32, 0.0f32, 0.0f32);
        for &i in &comp {
            let (dx, dy) = ((i % w) as f32 - mx, (i / w) as f32 - my);
            xx += dx * dx;
            yy += dy * dy;
            xy += dx * dy;
        }
        let (a, d) = ((xx + yy) / (2.0 * n), (((xx - yy) / (2.0 * n)).powi(2) + (xy / n).powi(2)).sqrt());
        let elongation = ((a + d) / (a - d).max(1e-6)).sqrt();
        // An oval, not a stretch of belt; the largest such.
        if elongation < 2.5 && best.is_none_or(|q| n > q.0) {
            best = Some((n, 2.0 * (n / std::f32::consts::PI).sqrt()));
        }
    }
    best.map(|q| q.1).filter(|&d| d > 0.03 * radius && d < 0.4 * radius)
}

/// Saturn's rings from the globe found by [`crate::mask::saturn_globe`]: their outer extent and
/// how deep the Cassini division is, on brightness profiles along the ring's major axis (both
/// ansae). The division sits at 87.6 % of the outer radius of the A ring, between the A ring
/// (outside) and the brighter B ring (inside).
pub fn rings(lum: &[f32], w: usize, h: usize, globe: Ellipse) -> Option<Rings> {
    let outer = globe.rx * 2.27;
    if outer < 60.0 {
        return None;
    }
    let (s, c) = globe.angle.sin_cos();
    // Mean brightness at `r` (fraction of the outer radius) on both ansae, across ±1.5 px.
    let at = |r: f32| -> Option<f32> {
        let (mut sum, mut n) = (0.0f32, 0);
        for side in [-1.0f32, 1.0] {
            for k in -1i32..=1 {
                let (u, v) = (side * r * outer, 1.5 * k as f32);
                let (x, y) = (globe.cx + u * c - v * s, globe.cy + u * s + v * c);
                let (xi, yi) = (x.round() as isize, y.round() as isize);
                if (0..w as isize).contains(&xi) && (0..h as isize).contains(&yi) {
                    sum += lum[yi as usize * w + xi as usize];
                    n += 1;
                }
            }
        }
        (n >= 4).then(|| sum / n as f32)
    };
    let gap = at(0.876)?;
    let (a_ring, b_ring) = (at(0.94)?, at(0.80)?);
    let around = 0.5 * (a_ring + b_ring);
    (around > 1e-4).then(|| Rings { outer, cassini: ((around - gap) / around).clamp(0.0, 1.0) })
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
