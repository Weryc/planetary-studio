//! The automatic "Nitidezza intelligente" on simulated raws, measured against their truths and
//! against the best linear sharpening there could have been.
//!
//! `cargo run --release -p ps-train --bin sim_eval -- out_dir run… [--real raw.tif]`
//!
//! Each `run` is a folder written by `sim_stack` (with `grezza.tif` and `meta.json`); its parent
//! holds that telescope's truths (`verita_campionata.tif`, `limite_diffrazione.tif`). The
//! sharpening is the app's one-button setting (`preset::auto_smart`, normal strength, subject
//! only). Everything is measured on the luminance, inside a square on the disk, after a global
//! sub-pixel registration on the truth. By frequency (log bins):
//! - response: the part of an image in phase with the truth over the truth (1 = all of its
//!   contrast at that size);
//! - the best linear gain ("Wiener", from the data): Re(X·T*) / |X|², what multiplying the raw's
//!   spectrum should have done to come closest to the truth — next to the gain the sharpening
//!   actually gave, Re(S·X*) / |X|²;
//! - SNR of the raw: power in phase with the truth over the rest (noise, misplacement);
//! - the rest after sharpening and after the best gain (the noise each one leaves).
//!
//! The best gain is also applied to the raw, as the bound of what linear sharpening could do.
//! Writes JPEG views and `metrics.json` into `out_dir`.
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::align::measure_shift;
use ps_core::analysis::{Analysis, analyze};
use ps_core::fft::FftConvolver;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn load(path: &Path) -> Image {
    ml::load_any(path).unwrap_or_else(|| panic!("{}: illeggibile", path.display()))
}

fn sharpen(img: &Image) -> (Image, Analysis, ps_core::smart::SmartParams) {
    let a = analyze(img);
    let p = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
    let subject = ps_core::mask::subject_and_moons(img);
    (ps_core::smart::sharpen(img, None, &p, Some(&subject)), a, p)
}

/// Best gain and offset taking `x` onto `t` over the square: (α, β).
fn fit(x: &[f32], t: &[f32], w: usize, sq: (usize, usize, usize)) -> (f32, f32) {
    let (xs, ts) = (ml::crop(x, w, sq), ml::crop(t, w, sq));
    let n = xs.len() as f64;
    let (mx, mt) = (xs.iter().map(|&v| v as f64).sum::<f64>() / n, ts.iter().map(|&v| v as f64).sum::<f64>() / n);
    let (mut cov, mut var) = (0.0f64, 0.0f64);
    for (a, b) in xs.iter().zip(&ts) {
        cov += (*a as f64 - mx) * (*b as f64 - mt);
        var += (*a as f64 - mx).powi(2);
    }
    let alpha = cov / var.max(1e-30);
    (alpha as f32, (mt - alpha * mx) as f32)
}

/// RMS error over the truth's standard deviation, after the fit.
fn rel_error(x: &[f32], t: &[f32], w: usize, sq: (usize, usize, usize)) -> f32 {
    let (a, b) = fit(x, t, w, sq);
    let (xs, ts) = (ml::crop(x, w, sq), ml::crop(t, w, sq));
    let n = ts.len() as f32;
    let mt = ts.iter().sum::<f32>() / n;
    let sd = (ts.iter().map(|v| (v - mt).powi(2)).sum::<f32>() / n).sqrt();
    let err = (xs.iter().zip(&ts).map(|(x, t)| (a * x + b - t).powi(2)).sum::<f32>() / n).sqrt();
    err / sd.max(1e-12)
}

/// Mean SSIM (Gaussian windows, σ 1.5 px) over the square, after the fit.
fn ssim(x: &[f32], t: &[f32], w: usize, sq: (usize, usize, usize)) -> f32 {
    let (a, b) = fit(x, t, w, sq);
    let s = sq.2;
    let xs: Vec<f32> = ml::crop(x, w, sq).iter().map(|v| a * v + b).collect();
    let ts = ml::crop(t, w, sq);
    let g = |p: &[f32]| ps_core::mask::gaussian_approx(p, s, s, 1.5);
    let (mx, mt) = (g(&xs), g(&ts));
    let sxx = g(&xs.iter().map(|v| v * v).collect::<Vec<_>>());
    let stt = g(&ts.iter().map(|v| v * v).collect::<Vec<_>>());
    let sxt = g(&xs.iter().zip(&ts).map(|(p, q)| p * q).collect::<Vec<_>>());
    let range = ts.iter().copied().fold(0.0f32, f32::max) - ts.iter().copied().fold(f32::MAX, f32::min);
    let (c1, c2) = ((0.01 * range).powi(2), (0.03 * range).powi(2));
    let m = 8;
    let (mut sum, mut n) = (0.0f64, 0usize);
    for y in m..s - m {
        for x in m..s - m {
            let i = y * s + x;
            let (vx, vt, cxt) = (sxx[i] - mx[i] * mx[i], stt[i] - mt[i] * mt[i], sxt[i] - mx[i] * mt[i]);
            sum += (((2.0 * mx[i] * mt[i] + c1) * (2.0 * cxt + c2)) / ((mx[i] * mx[i] + mt[i] * mt[i] + c1) * (vx + vt + c2))) as f64;
            n += 1;
        }
    }
    (sum / n as f64) as f32
}

const BINS: usize = 22;
const F_LO: f32 = 0.012;
const F_HI: f32 = 0.5;

fn bin_of(f: f32) -> Option<usize> {
    if !(F_LO..=F_HI).contains(&f) {
        return None;
    }
    let (lo, hi) = (F_LO.ln(), F_HI.ln());
    Some((((f.ln() - lo) / (hi - lo)) * BINS as f32).clamp(0.0, BINS as f32 - 1.0) as usize)
}

fn bin_centre(k: usize) -> f32 {
    let (lo, hi) = (F_LO.ln(), F_HI.ln());
    (lo + (k as f32 + 0.5) / BINS as f32 * (hi - lo)).exp()
}

/// Per bin, sums over the square's coefficients of the cross products of the images.
struct Spectra {
    /// [i][j] = Σ Re(A_i · A_j*), images in the order given.
    cross: Vec<Vec<Vec<f64>>>,
}

fn spectra(planes: &[&[f32]], w: usize, sq: (usize, usize, usize)) -> Spectra {
    let c: Vec<Vec<[f32; 4]>> = planes.iter().map(|p| ml::coefs(&ml::crop(p, w, sq), sq.2)).collect();
    let n = planes.len();
    let mut cross = vec![vec![vec![0.0f64; BINS]; n]; n];
    for idx in 0..c[0].len() {
        let Some(k) = bin_of(c[0][idx][0].hypot(c[0][idx][1])) else { continue };
        for i in 0..n {
            for j in i..n {
                let (a, b) = (c[i][idx], c[j][idx]);
                let v = (a[2] * b[2] + a[3] * b[3]) as f64;
                cross[i][j][k] += v;
                if i != j {
                    cross[j][i][k] += v;
                }
            }
        }
    }
    Spectra { cross }
}

impl Spectra {
    /// Response of image `i` against the truth `t`: Re(I·T*) / |T|².
    fn response(&self, i: usize, t: usize) -> Vec<f32> {
        (0..BINS).map(|k| if self.cross[t][t][k] > 0.0 { (self.cross[i][t][k] / self.cross[t][t][k]) as f32 } else { f32::NAN }).collect()
    }
    /// What in image `i` isn't the truth seen through its response, over |T|² (amplitude).
    fn rest(&self, i: usize, t: usize) -> Vec<f32> {
        (0..BINS)
            .map(|k| {
                let tt = self.cross[t][t][k];
                if tt <= 0.0 {
                    return f32::NAN;
                }
                let it = self.cross[i][t][k];
                ((self.cross[i][i][k] - it * it / tt).max(0.0) / tt).sqrt() as f32
            })
            .collect()
    }
    /// Gain from image `x` to image `y`: Re(Y·X*) / |X|².
    fn gain(&self, y: usize, x: usize) -> Vec<f32> {
        (0..BINS).map(|k| if self.cross[x][x][k] > 0.0 { (self.cross[y][x][k] / self.cross[x][x][k]) as f32 } else { f32::NAN }).collect()
    }
    /// SNR of image `x`: power in phase with `t` over the rest.
    fn snr(&self, x: usize, t: usize) -> Vec<f32> {
        (0..BINS)
            .map(|k| {
                if self.cross[t][t][k] <= 0.0 {
                    return f32::NAN;
                }
                let (xx, xt, tt) = (self.cross[x][x][k], self.cross[x][t][k], self.cross[t][t][k]);
                let coherent = xt * xt / tt;
                (coherent / (xx - coherent).max(1e-30)) as f32
            })
            .collect()
    }
    /// Total power of the rest of image `i` (all bins): the noise it carries, in its units.
    fn rest_power(&self, i: usize, t: usize) -> f64 {
        (0..BINS)
            .map(|k| {
                let tt = self.cross[t][t][k].max(1e-30);
                let it = self.cross[i][t][k];
                (self.cross[i][i][k] - it * it / tt).max(0.0)
            })
            .sum()
    }
}

/// Normalise a curve by its mean over the bins below 0.025 cycles/px that hold coefficients
/// (photometric scale).
fn normalised(v: &[f32]) -> Vec<f32> {
    let low: Vec<f32> = v.iter().enumerate().filter(|(k, x)| bin_centre(*k) < 0.025 && x.is_finite()).map(|(_, x)| *x).collect();
    let base = if low.is_empty() { v.iter().copied().find(|x| x.is_finite()).unwrap_or(1.0) } else { low.iter().sum::<f32>() / low.len() as f32 };
    v.iter().map(|x| x / base).collect()
}

/// Band of a frequency for the best filter: 0 below F_LO (with the mean), 1..=BINS the log
/// bins, BINS + 1 the corners beyond 0.5.
fn band_of(f: f32) -> usize {
    if f < F_LO {
        0
    } else if f > F_HI {
        BINS + 1
    } else {
        1 + bin_of(f).unwrap()
    }
}

const BANDS: usize = BINS + 2;

/// The plane split into its frequency bands (they add up to it).
fn bands(plane: &[f32], side: usize) -> Vec<Vec<f32>> {
    let plan = FftConvolver::plan(side, side);
    let spec = plan.forward(plane);
    let which: Vec<usize> = (0..spec.len())
        .map(|idx| {
            let (kx, ky) = (idx / side, idx % side);
            let fy = (if ky <= side / 2 { ky as f32 } else { ky as f32 - side as f32 }) / side as f32;
            band_of((kx as f32 / side as f32).hypot(fy))
        })
        .collect();
    (0..BANDS)
        .map(|b| {
            let masked = spec.iter().zip(&which).map(|(c, &k)| if k == b { *c } else { Default::default() }).collect();
            plan.inverse(masked)
        })
        .collect()
}

/// The best radial filter taking `x` onto `t` over the pixels `on` (the whole subject): a gain
/// per band (and an offset), least squares. By construction no single sharpening curve applied
/// to the subject gets closer to the truth there; fitted on the whole subject (rings and limb
/// included) it can't chase the noise of a small patch.
fn best_filter(x: &[f32], t: &[f32], side: usize, on: &[usize]) -> Vec<f32> {
    // No offset: sky is zero in both, so the relation goes through the origin, and an offset
    // fitted on a bright zone would lift the dark ones (Saturn's ring-globe gap turned grey).
    let b = bands(x, side);
    let k = BANDS;
    let mut m = vec![0.0f64; k * k];
    let mut r = vec![0.0f64; k];
    let mut row = vec![0.0f64; k];
    for &i in on {
        for (j, band) in b.iter().enumerate() {
            row[j] = band[i] as f64;
        }
        for a in 0..k {
            r[a] += row[a] * t[i] as f64;
            for c in 0..k {
                m[a * k + c] += row[a] * row[c];
            }
        }
    }
    // A touch of ridge towards gain 1 keeps empty or noise-only bands sane.
    let ridge = 1e-6 * (0..k).map(|a| m[a * k + a]).sum::<f64>() / k as f64;
    for a in 0..k {
        m[a * k + a] += ridge;
        r[a] += ridge;
    }
    solve(m, r, k).into_iter().map(|v| v as f32).collect()
}

/// The error of any band filter `g` applied to `x`, against `t` over `on`, as a quadratic form:
/// Σ(Σ_b g_b·x_b − t)² = gᵀMg − 2rᵀg + tt. Saved per run, it lets a curve predictor be trained
/// and scored on the true image error without touching an image again.
fn quadratic(x: &[f32], t: &[f32], side: usize, on: &[usize]) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let b = bands(x, side);
    let k = BANDS;
    let (mut m, mut r, mut tt) = (vec![0.0f64; k * k], vec![0.0f64; k], 0.0f64);
    let mean = on.iter().map(|&i| t[i] as f64).sum::<f64>() / on.len().max(1) as f64;
    let mut tvar = 0.0f64;
    let mut row = vec![0.0f64; k];
    for &i in on {
        for (j, band) in b.iter().enumerate() {
            row[j] = band[i] as f64;
        }
        let ti = t[i] as f64;
        tt += ti * ti;
        tvar += (ti - mean).powi(2);
        for a in 0..k {
            r[a] += row[a] * ti;
            for c in 0..k {
                m[a * k + c] += row[a] * row[c];
            }
        }
    }
    (m, r, tt, tvar)
}

/// RMS error over `on` after the best scale (no offset: sky is zero in both), over the truth's
/// spread there.
fn region_error(x: &[f32], t: &[f32], on: &[usize], tvar: f64) -> (f32, f64) {
    let (mut xt, mut xx, mut tt) = (0.0f64, 0.0f64, 0.0f64);
    for &i in on {
        xt += x[i] as f64 * t[i] as f64;
        xx += (x[i] as f64).powi(2);
        tt += (t[i] as f64).powi(2);
    }
    let alpha = xt / xx.max(1e-30);
    (((tt - xt * alpha).max(0.0) / tvar.max(1e-30)).sqrt() as f32, alpha)
}

/// Colour error over `on`: the two chroma planes (R−G, B−G) after the luminance's scale, RMS
/// over the truth's chroma spread. NaN for mono.
fn colour_error(img: &Image, truth: &Image, on: &[usize], alpha: f64) -> f32 {
    if img.is_mono() || truth.is_mono() {
        return f32::NAN;
    }
    let (x, t) = (img.channels(), truth.channels());
    let (mut err, mut spread) = (0.0f64, 0.0f64);
    for c in [0usize, 2] {
        let ct: Vec<f64> = on.iter().map(|&i| (t[c][i] - t[1][i]) as f64).collect();
        let mean = ct.iter().sum::<f64>() / ct.len().max(1) as f64;
        for (k, &i) in on.iter().enumerate() {
            let cx = alpha * (x[c][i] - x[1][i]) as f64;
            err += (cx - ct[k]).powi(2);
            spread += (ct[k] - mean).powi(2);
        }
    }
    (err / spread.max(1e-30)).sqrt() as f32
}

fn solve(mut m: Vec<f64>, mut r: Vec<f64>, k: usize) -> Vec<f64> {
    for c in 0..k {
        let p = (c..k).max_by(|&a, &b| m[a * k + c].abs().total_cmp(&m[b * k + c].abs())).unwrap();
        if p != c {
            for j in 0..k {
                m.swap(c * k + j, p * k + j);
            }
            r.swap(c, p);
        }
        let d = m[c * k + c];
        if d.abs() < 1e-30 {
            continue;
        }
        for q in c + 1..k {
            let f = m[q * k + c] / d;
            for j in c..k {
                m[q * k + j] -= f * m[c * k + j];
            }
            r[q] -= f * r[c];
        }
    }
    let mut x = vec![0.0; k];
    for c in (0..k).rev() {
        let d = m[c * k + c];
        x[c] = if d.abs() < 1e-30 { 0.0 } else { (r[c] - (c + 1..k).map(|j| m[c * k + j] * x[j]).sum::<f64>()) / d };
    }
    x
}

/// Local shifts of `x` against `t` on 64-px patches across the lit disk: (RMS, max) px.
fn deformation(x: &[f32], t: &[f32], w: usize, h: usize) -> (f32, f32, usize) {
    let top = t.iter().copied().fold(0.0f32, f32::max);
    let outer = 96;
    let mut d = Vec::new();
    for cy in (outer / 2..h - outer / 2).step_by(32) {
        for cx in (outer / 2..w - outer / 2).step_by(32) {
            let p = |q: &[f32]| -> Vec<f32> { (0..outer).flat_map(|y| q[(cy - outer / 2 + y) * w + cx - outer / 2..(cy - outer / 2 + y) * w + cx + outer / 2].to_vec()).collect() };
            let pt = p(t);
            if pt.iter().copied().fold(f32::MAX, f32::min) < 0.2 * top {
                continue;
            }
            let s = measure_shift(&pt, &p(x), outer, outer);
            d.push(s[0].hypot(s[1]));
        }
    }
    let rms = (d.iter().map(|v| v * v).sum::<f32>() / d.len().max(1) as f32).sqrt();
    (rms, d.iter().copied().fold(0.0, f32::max), d.len())
}

/// Luminance mapped by (α, β), `top` at white, gamma 2.2, or the colour planes by the same map.
fn view(img: &Image, (a, b): (f32, f32), top: f32) -> Image {
    let mut planes: Vec<Vec<f32>> = img.channels().iter().map(|p| p.iter().map(|v| ((a * v + b) / top).clamp(0.0, 1.0).powf(1.0 / 2.2)).collect()).collect();
    if planes.len() == 1 {
        planes = vec![planes[0].clone(), planes[0].clone(), planes[0].clone()];
    }
    Image::new(img.width(), img.height(), planes, 16)
}

/// A `side`² crop around (cx, cy), enlarged to about `px` (nearest).
fn zoom(img: &Image, cx: usize, cy: usize, side: usize, px: usize) -> Image {
    let side = side.min(img.width()).min(img.height());
    let z = px.div_ceil(side).max(1);
    let (x0, y0) = (cx.saturating_sub(side / 2).min(img.width() - side), cy.saturating_sub(side / 2).min(img.height() - side));
    let s = side * z;
    let planes = img.channels().iter().map(|p| (0..s * s).map(|i| p[(y0 + (i / s) / z) * img.width() + x0 + (i % s) / z]).collect()).collect();
    Image::new(s, s, planes, 16)
}

fn percentile_top(l: &[f32]) -> f32 {
    let mut v: Vec<f32> = l.iter().step_by(3).copied().collect();
    v.sort_by(f32::total_cmp);
    v[v.len() * 999 / 1000].max(1e-6)
}

fn arr(v: &[f32]) -> String {
    format!("[{}]", v.iter().map(|x| if x.is_finite() { format!("{x:.4}") } else { "null".into() }).collect::<Vec<_>>().join(","))
}

/// The plane with each band multiplied by its gain.
fn apply_bands(plane: &[f32], side: usize, gains: &[f32]) -> Vec<f32> {
    let b = bands(plane, side);
    (0..plane.len()).map(|i| b.iter().zip(gains).map(|(band, g)| band[i] * g).sum()).collect()
}

/// The filter applied as the app sharpens: to the luminance only and only on the subject
/// (`weight`, 0‥1), merged back into colour as the app does (`color::merge_planes`: brightening
/// adds, darkening scales the channels, so dark zones keep their hue). Gains are taken relative
/// to band 0's (the mean and the broadest shapes): an even zone stays exactly as it was, so the
/// raw's brightness and its black are kept and only the detail changes.
fn image_with_gains(img: &Image, lum: &[f32], fitted: &[f32], weight: &[f32]) -> Image {
    let side = img.width();
    let rel: Vec<f32> = fitted.iter().map(|g| g / fitted[0]).collect();
    let sharp = apply_bands(lum, side, &rel);
    let new_lum: Vec<f32> = sharp.iter().zip(lum).zip(weight).map(|((s, l), w)| l + w * (s - l)).collect();
    if img.is_mono() {
        return Image::new(side, img.height(), vec![new_lum], 16);
    }
    ps_core::color::merge_planes(img, ps_core::color::ColorMode::Luminance, &[lum.to_vec()], vec![new_lum])
}

/// The stack's own PSF, known from the simulation: the transfer from the truth to the raw
/// (Re(R·T*) / |T|² over the whole frame, in rings one frequency step wide, normalised at
/// zero frequency), turned back into a kernel. It holds everything the raw went through:
/// optics, the kept frames' residual seeing, the stacker's misalignments, the debayer.
fn stack_psf(raw: &[f32], truth: &[f32], side: usize, radius: usize) -> ps_core::psf::Kernel2D {
    let plan = FftConvolver::plan(side, side);
    let (r, t) = (plan.forward(raw), plan.forward(truth));
    let ring = |idx: usize| {
        let (kx, ky) = (idx / side, idx % side);
        let fy = if ky <= side / 2 { ky as f32 } else { ky as f32 - side as f32 };
        (kx as f32).hypot(fy).round() as usize
    };
    let rings = side;
    let (mut num, mut den) = (vec![0.0f64; rings], vec![0.0f64; rings]);
    for idx in 0..r.len() {
        let k = ring(idx);
        if k < rings {
            num[k] += (r[idx].re * t[idx].re + r[idx].im * t[idx].im) as f64;
            den[k] += (t[idx].re * t[idx].re + t[idx].im * t[idx].im) as f64;
        }
    }
    let h0 = num[0] / den[0].max(1e-30);
    let mtf: Vec<f32> = (0..rings).map(|k| if den[k] > 0.0 { ((num[k] / den[k]) / h0).clamp(0.0, 1.0) as f32 } else { 0.0 }).collect();
    let mut spec = r.clone();
    for (idx, c) in spec.iter_mut().enumerate() {
        c.re = mtf.get(ring(idx)).copied().unwrap_or(0.0);
        c.im = 0.0;
    }
    let k = plan.inverse(spec);
    let size = 2 * radius + 1;
    let mut data: Vec<f32> = (0..size * size)
        .map(|i| {
            let (dx, dy) = ((i % size) as isize - radius as isize, (i / size) as isize - radius as isize);
            k[(dy.rem_euclid(side as isize) as usize) * side + dx.rem_euclid(side as isize) as usize].max(0.0)
        })
        .collect();
    let sum: f32 = data.iter().sum();
    data.iter_mut().for_each(|v| *v /= sum.max(1e-12));
    ps_core::psf::Kernel2D { radius, data }
}

/// The app's deconvolution on the luminance, merged back as the app does.
fn deconvolve(img: &Image, p: &ps_core::deconv::DeconvParams) -> Image {
    let p = ps_core::deconv::DeconvParams { color: ps_core::color::ColorMode::Luminance, ..p.clone() };
    let planes = ps_core::color::working_planes(img, p.color);
    let out = ps_core::deconv::deconvolve_plane(&planes[0], img.width(), img.height(), &p, &|| false, &|_| {}).unwrap();
    ps_core::color::merge_planes(img, p.color, &planes, vec![out])
}

/// `plane` moved by `d` px (same direction as `ps_core::align::shift`), exactly: a phase ramp in
/// frequency, so nothing is softened.
fn fourier_shift(plane: &[f32], w: usize, h: usize, d: [f32; 2]) -> Vec<f32> {
    let plan = FftConvolver::plan(w, h);
    let mut spec = plan.forward(plane);
    // Transposed layout: index = kx·h + ky, kx ∈ 0..=w/2.
    for (idx, c) in spec.iter_mut().enumerate() {
        let (kx, ky) = (idx / h, idx % h);
        let fy = if ky <= h / 2 { ky as f32 } else { ky as f32 - h as f32 } / h as f32;
        let fx = kx as f32 / w as f32;
        let phase = std::f32::consts::TAU * (fx * d[0] + fy * d[1]);
        let (sn, cs) = phase.sin_cos();
        let (re, im) = (c.re, c.im);
        c.re = re * cs - im * sn;
        c.im = re * sn + im * cs;
    }
    plan.inverse(spec)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(&args[0]);
    let mut runs = Vec::new();
    let mut real = None;
    let mut deconv_test = false;
    let mut net_app = false;
    let mut predicted: std::collections::HashMap<String, Vec<f32>> = Default::default();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--gains" {
            // Predicted curves: "name<TAB>g0 … g23" per line, gains relative to band 0.
            for line in std::fs::read_to_string(&args[i + 1]).unwrap().lines() {
                let mut f = line.split('\t');
                if let Some(n) = f.next() {
                    let g: Vec<f32> = f.filter_map(|v| v.parse().ok()).collect();
                    if g.len() == BANDS {
                        predicted.insert(n.to_string(), g);
                    }
                }
            }
            i += 2;
        } else if args[i] == "--net-app" {
            net_app = true;
            i += 1;
        } else if args[i] == "--deconv" {
            deconv_test = true;
            i += 1;
        } else if args[i] == "--real" {
            real = args.get(i + 1).cloned();
            i += 2;
        } else {
            runs.push(PathBuf::from(&args[i]));
            i += 1;
        }
    }
    std::fs::create_dir_all(&out).unwrap();
    let save = |name: &str, img: &Image| ps_io::save(&out.join(format!("{name}.jpg")), img).unwrap();
    let freqs: Vec<f32> = (0..BINS).map(bin_centre).collect();
    let mut json = format!("{{\n  \"frequenze\": {},\n  \"versioni\": [\n", arr(&freqs));
    let mut truths_done = std::collections::HashSet::new();
    let mut deconv_tsv = String::from("nome\tfwhm_psf\tforza_nitida\tforza_dopo_vera\tgrezza\tnitida\tdeconv_app\tdeconv_app+nitida\tdeconv_vera\tdeconv_vera+nitida\tdeconv_vera_lunga+nitida\tottima\tottima_dopo_deconv_vera\tssim_nitida\tssim_vera+nitida\tssim_ottima\tbruciati_nitida\tbruciati_vera+nitida\n");
    let mut net_tsv = String::from("nome\tpicco_rete\tforza_rete\tapp_errore\tapp_colore\tapp_bruciati\trete_rgb_adattiva_errore\tcolore\tbruciati\trete_rgb_uniforme_errore\tcolore\tbruciati\trete_lum_adattiva_errore\tcolore\tbruciati\n");
    let mut burnt_tsv = String::from("nome\tverita\tgrezza\tnitida\tottima\tpredetta\n");
    for (k, run) in runs.iter().enumerate() {
        let name = run.file_name().unwrap().to_string_lossy().to_string();
        let base = run.parent().unwrap();
        let setup = format!("{}_{}", base.parent().unwrap().file_name().unwrap().to_string_lossy(), base.file_name().unwrap().to_string_lossy());
        // The run's own truths (older runs kept them one level up, shared by the setup).
        let own = |f: &str| if run.join(f).exists() { run.join(f) } else { base.join(f) };
        let truth = load(&own("verita_campionata.tif"));
        let diffraction = load(&own("limite_diffrazione.tif"));
        let (w, h) = (truth.width(), truth.height());
        // The truths are moved onto the raw, not the raw onto the truths: an interpolated raw
        // loses some of its finest detail, and everything measured here (the network's inputs,
        // the best curve) must be about the raw exactly as the app gets it. An exact shift in
        // frequency softens nothing (the frame's edges are black sky, so wrapping is harmless).
        let raw = load(&run.join("grezza.tif"));
        let s = measure_shift(&ml::lum(&truth), &ml::lum(&raw), w, h);
        let onto_raw = |img: &Image| Image::new(w, h, img.channels().iter().map(|c| fourier_shift(c, w, h, s)).collect(), 16);
        let (truth, diffraction) = (onto_raw(&truth), onto_raw(&diffraction));
        let left = measure_shift(&ml::lum(&truth), &ml::lum(&raw), w, h);
        if left[0].hypot(left[1]) > 0.05 {
            eprintln!("  attenzione: dopo lo spostamento della verità restano {:.3}, {:.3} px", left[0], left[1]);
        }
        let ta = analyze(&truth);
        let disk = ta.disk.map_or(ta.subject_size, |d| 2.0 * d.radius);
        // Inside the subject (Saturn: the globe); small globes get a smaller square.
        let sq = ml::inner_square(w, h, ml::outline(&truth, &ta), 32).unwrap_or_else(|| {
            let (cx, cy) = ml::centre(&truth, &ta);
            let side = 32usize;
            ((cx as usize).saturating_sub(side / 2).min(w - side), (cy as usize).saturating_sub(side / 2).min(h - side), side)
        });
        let (lt, ld) = (ml::lum(&truth), ml::lum(&diffraction));
        let top = percentile_top(&lt);
        let zside = ((disk * 0.36) as usize).max(48).min(w);
        // From the simulation's own record: the analysis can take a lunar field for a disk.
        let meta_text = std::fs::read_to_string(run.join("meta.json")).unwrap_or_default();
        let surface = meta_text.contains("\"soggetto\": \"Moon\"") || (matches!(ta.kind, ps_core::analysis::ImageKind::Surface | ps_core::analysis::ImageKind::FullDisk) && ta.disk.is_none());
        // Zoom on the most detailed zone of the truth (the subject's centre if none qualifies).
        let grs = {
            let c = ml::centre(&truth, &ta);
            let found = ml::truths::busiest(&truth, zside, 1).first().copied().filter(|p| surface || (p.0 - c.0).hypot(p.1 - c.1) < 0.35 * disk);
            let p = found.unwrap_or(c);
            (p.0 as usize, p.1 as usize)
        };
        // Per run: each has its own truth (the disk's size changes between runs).
        if truths_done.insert(name.clone()) {
            save(&format!("{name}_verita_zoom"), &zoom(&view(&truth, (1.0, 0.0), top), grs.0, grs.1, zside, 360));
            save(&format!("{name}_diffrazione_zoom"), &zoom(&view(&diffraction, (1.0, 0.0), top), grs.0, grs.1, zside, 360));
        }
        let (sharp, a, p) = sharpen(&raw);
        let (rawr, sharpr) = (raw.clone(), sharp);
        let (lr, ls) = (ml::lum(&rawr), ml::lum(&sharpr));
        // Images: 0 raw, 1 sharpened, 2 truth, 3 diffraction.
        let sp = spectra(&[&lr, &ls, &lt, &ld], w, sq);
        // The best filter, fitted on the pixels; as a curve, the log bins over the lowest ones.
        // Where the best filter is fitted and applied: the subject (rings and moons included,
        // from the truth), or the whole field but its edges for a surface.
        let weight: Vec<f32> = if surface {
            vec![1.0; w * h]
        } else {
            let m = ps_core::mask::subject_and_moons(&truth);
            ps_core::mask::gaussian_approx(&m, w, h, 2.0)
        };
        // Fitted inside, a few pixels from the limb and ring edges (the app protects them too):
        // there the sharp edge against the sky would decide the whole curve.
        let border = 12;
        let inner = if surface { weight.clone() } else { ps_core::mask::gaussian_approx(&ps_core::mask::subject_and_moons(&truth), w, h, 4.0) };
        // A surface reaches the frame's edges, where the periodic bands wrap: there the big
        // central square is used instead.
        let saturn = meta_text.contains("\"soggetto\": \"Saturn\"");
        // Too small a globe (under 3000 px) leaves too few pixels for 25 parameters: the curve
        // would chase the noise; the whole subject is used instead.
        let globe = if saturn { ps_core::mask::saturn_globe(&truth).filter(|e| std::f32::consts::PI * 0.81 * e.rx * e.ry > 3000.0) } else { None };
        let on: Vec<usize> = if surface {
            (sq.1..sq.1 + sq.2).flat_map(|y| (sq.0..sq.0 + sq.2).map(move |x| y * w + x)).collect()
        } else if let Some(e) = globe {
            // Saturn: one curve can't suit both the rings' sharp edges and the soft globe; it is
            // fitted where it's measured, the globe (90 % of its outline).
            let (c, si) = (e.angle.cos(), e.angle.sin());
            (0..w * h)
                .filter(|&i| {
                    let (dx, dy) = ((i % w) as f32 - e.cx, (i / w) as f32 - e.cy);
                    let (u, v) = (dx * c + dy * si, -dx * si + dy * c);
                    (u / (0.9 * e.rx)).powi(2) + (v / (0.9 * e.ry)).powi(2) <= 1.0
                })
                .collect()
        } else {
            (0..w * h).filter(|&i| inner[i] > 0.97 && (border..w - border).contains(&(i % w)) && (border..h - border).contains(&(i / w))).collect()
        };
        let gains = best_filter(&lr, &lt, w, &on);
        let gains_diff = best_filter(&lr, &ld, w, &on);
        // What the app could know from the raw alone: the power of each band over the raw's own
        // subject interior (log10, over the squared mean level), and the analysis.
        let features = {
            let m = ps_core::mask::gaussian_approx(&ps_core::mask::subject_and_moons(&rawr), w, h, 4.0);
            let on_raw: Vec<usize> = (0..w * h).filter(|&i| m[i] > 0.97 && (border..w - border).contains(&(i % w)) && (border..h - border).contains(&(i / w))).collect();
            let on_raw = if on_raw.len() > 500 { on_raw } else { on.clone() };
            let mean = on_raw.iter().map(|&i| lr[i] as f64).sum::<f64>() / on_raw.len() as f64;
            let b = bands(&lr, w);
            let pw: Vec<f32> = b.iter().map(|band| ((on_raw.iter().map(|&i| (band[i] as f64).powi(2)).sum::<f64>() / on_raw.len() as f64) / mean.powi(2).max(1e-30)).max(1e-14).log10() as f32).collect();
            (pw, on_raw.len())
        };
        let rel_gains: Vec<f32> = gains.iter().map(|g| g / gains[0]).collect();
        // Noise as the sky shows it: the power of each band well outside the subject (1.5 radii,
        // away from the frame's edges), on the same scale as the subject's bands.
        let noise_bands: Vec<f32> = {
            let (cx, cy) = ml::centre(&truth, &ta);
            let sky_px: Vec<usize> = (0..w * h)
                .filter(|&i| ((i % w) as f32 - cx).hypot((i / w) as f32 - cy) > 0.75 * disk && (16..w - 16).contains(&(i % w)) && (16..h - 16).contains(&(i / w)))
                .collect();
            let m = ps_core::mask::gaussian_approx(&ps_core::mask::subject_and_moons(&rawr), w, h, 4.0);
            let inside: Vec<usize> = (0..w * h).filter(|&i| m[i] > 0.97).collect();
            let mean = inside.iter().map(|&i| lr[i] as f64).sum::<f64>() / inside.len().max(1) as f64;
            if surface || sky_px.len() < 500 {
                vec![f32::NAN; BANDS]
            } else {
                bands(&lr, w).iter().map(|band| ((sky_px.iter().map(|&i| (band[i] as f64).powi(2)).sum::<f64>() / sky_px.len() as f64) / mean.powi(2).max(1e-30)).max(1e-14).log10() as f32).collect()
            }
        };
        // Headroom: the subject's 99.9th percentile in the raw's own units (its white is 1).
        let peak = {
            let mut v: Vec<f32> = on.iter().map(|&i| lr[i]).collect();
            v.sort_by(f32::total_cmp);
            v.get(v.len() * 999 / 1000).copied().unwrap_or(f32::NAN)
        };
        let (qm, qr, qtt, qvar) = quadratic(&lr, &lt, w, &on);
        let pred_img = predicted.get(&name).map(|g| image_with_gains(&rawr, &lr, g, &weight));
        let curve = |g: &[f32]| normalised(&g[1..=BINS]);
        let best = curve(&gains);
        let best_diff = curve(&gains_diff);
        let smart_gain = normalised(&sp.gain(1, 0));
        let wiener_img = image_with_gains(&rawr, &lr, &gains, &weight);
        let lw = ml::lum(&wiener_img);
        let sp2 = spectra(&[&lr, &ls, &lt, &lw], w, sq);
        let noise_raw = sp2.rest_power(0, 2);
        let amp = |i: usize| ((sp2.rest_power(i, 2) / noise_raw.max(1e-30)).sqrt() as f32, sp2.rest(i, 2));
        let (amp_smart, _) = amp(1);
        let (amp_best, _) = amp(3);
        // Rest over the truth, in the units of the normalised response (truth = image 2).
        let rest_norm = |s: &Spectra, i: usize| -> Vec<f32> {
            let r = s.response(i, 2);
            let n = normalised(&r);
            let b = r.iter().zip(&n).find(|(a, _)| a.is_finite()).map_or(1.0, |(a, n)| a / n);
            s.rest(i, 2).iter().map(|v| v / b).collect()
        };
        // Sky: median outside the disk, raw and sharpened, and the display fits' offsets.
        let sky = |l: &[f32]| {
            let (cx, cy) = ml::centre(&truth, &ta);
            let mut v: Vec<f32> = (0..w * h).filter(|&i| ((i % w) as f32 - cx).hypot((i / w) as f32 - cy) > 0.62 * disk).map(|i| l[i]).collect();
            v.sort_by(f32::total_cmp);
            (v[v.len() / 2], v[v.len() * 99 / 100])
        };
        let (fr, fs) = (fit(&lr, &lt, w, sq), fit(&ls, &lt, w, sq));
        eprintln!(
            "  cielo grezza mediana {:.5} p99 {:.5} | nitida {:.5} p99 {:.5} | adattamento vista grezza α {:.3} β {:+.4}, nitida α {:.3} β {:+.4} (bianco a {top:.3})",
            sky(&lr).0, sky(&lr).1, sky(&ls).0, sky(&ls).1, fr.0, fr.1, fs.0, fs.1
        );
        let (def_rms, def_max, def_n) = deformation(&lr, &ld, w, h);
        let meta = std::fs::read_to_string(run.join("meta.json")).unwrap_or_default();
        let metrics = |l: &[f32]| (rel_error(l, &lt, w, sq), ssim(l, &lt, w, sq), rel_error(l, &ld, w, sq), ssim(l, &ld, w, sq));
        let (mr, ms, mw) = (metrics(&lr), metrics(&ls), metrics(&lw));
        let mp = pred_img.as_ref().map(|img| metrics(&ml::lum(img)));
        // Over the fitted region (the same pixels as the quadratic form), and the colour.
        let (er_raw, al_raw) = region_error(&lr, &lt, &on, qvar);
        let (er_smart, al_smart) = region_error(&ls, &lt, &on, qvar);
        let (er_best, al_best) = region_error(&lw, &lt, &on, qvar);
        let ce = [colour_error(&rawr, &truth, &on, al_raw), colour_error(&sharpr, &truth, &on, al_smart), colour_error(&wiener_img, &truth, &on, al_best)];
        {
            let dir = out.join("quad");
            std::fs::create_dir_all(&dir).unwrap();
            let f = |v: &[f64]| format!("[{}]", v.iter().map(|x| format!("{x:.6e}")).collect::<Vec<_>>().join(","));
            std::fs::write(dir.join(format!("{name}.json")), format!("{{\"m\": {}, \"r\": {}, \"tt\": {qtt:.6e}, \"tvar\": {qvar:.6e}, \"n\": {}}}\n", f(&qm), f(&qr), on.len())).unwrap();
        }
        let _ = write!(
            json,
            "    {{ \"nome\": \"{name}\", \"setup\": \"{setup}\", \"meta\": {}, \"disco_px\": {disk:.0},\n      \"analisi\": {{ \"taglio\": {:.4}, \"rumore\": {:.6}, \"snr\": {:.0}, \"sigma_bordo\": {:.2} }}, \"nitidezza\": {{ \"forza\": {:.3}, \"curva\": \"{:?}\", \"picco\": {:.3} }},\n      \"errore\": {{ \"grezza\": [{:.4},{:.4},{:.4},{:.4}], \"nitida\": [{:.4},{:.4},{:.4},{:.4}], \"ottima\": [{:.4},{:.4},{:.4},{:.4}] }},\n      \"risposta_grezza\": {}, \"risposta_nitida\": {}, \"risposta_ottima\": {},\n      \"guadagno_ottimo\": {}, \"guadagno_ottimo_diffrazione\": {}, \"guadagno_nitida\": {},\n      \"snr_grezza\": {}, \"resto_grezza\": {}, \"resto_nitida\": {}, \"resto_ottima\": {},\n      \"rumore_amplificato\": {{ \"nitida\": {amp_smart:.3}, \"ottima\": {amp_best:.3} }},\n      \"deformazione\": {{ \"rms_px\": {def_rms:.3}, \"max_px\": {def_max:.3}, \"zone\": {def_n} }},\n      \"potenza_bande\": {}, \"pixel_soggetto\": {}, \"guadagni_bande\": {}, \"predetta\": {},\n      \"rumore_bande\": {}, \"picco_soggetto\": {peak:.4}, \"errore_regione\": [{er_raw:.4},{er_smart:.4},{er_best:.4}], \"errore_colore\": {}, \"forma\": {{ \"canali_sigma\": {}, \"moffat\": {}, \"rgb_shift\": {} }} }}",
            meta.trim(),
            a.spectral.as_ref().map_or(0.0, |s| s.cutoff),
            a.noise,
            a.snr,
            a.psf_sigma.unwrap_or(f32::NAN),
            p.strength,
            p.curve,
            p.peak,
            mr.0, mr.1, mr.2, mr.3,
            ms.0, ms.1, ms.2, ms.3,
            mw.0, mw.1, mw.2, mw.3,
            arr(&normalised(&sp.response(0, 2))),
            arr(&normalised(&sp.response(1, 2))),
            arr(&normalised(&sp2.response(3, 2))),
            arr(&best),
            arr(&best_diff),
            arr(&smart_gain),
            arr(&sp.snr(0, 2)),
            arr(&rest_norm(&sp, 0)),
            arr(&rest_norm(&sp2, 1)),
            arr(&rest_norm(&sp2, 3)),
            arr(&features.0),
            features.1,
            arr(&rel_gains),
            mp.map_or("null".to_string(), |m| format!("[{:.4},{:.4},{:.4},{:.4}]", m.0, m.1, m.2, m.3)),
            arr(&noise_bands),
            arr(&ce),
            a.channel_sigma.map_or("null".to_string(), |c| arr(&c)),
            a.psf_moffat.map_or("null".to_string(), |(f, b)| format!("[{f:.3},{b:.3}]")),
            a.rgb_shift.map_or("null".to_string(), |s| format!("\"{s:?}\"")),
        );
        json.push_str(if k + 1 < runs.len() { ",\n" } else { "\n" });
        // Displayed with the truth's scale but the sky kept black: only the gain is fitted, the
        // offset puts the sky's median at zero (a fitted offset would lift a sky that the
        // sharpening never touched).
        // One display mapping for the three, the raw's: both sharpenings keep the raw's scale, so
        // differences are what each did (a mapping fitted per image on a small square would
        // amplify any of them differently).
        // The raw's own photometry (from the simulation: raw ≈ fotometria × truth ⊛ PSF), sky at
        // black: the truth's white stays white. A gain fitted on a blurred square comes out too
        // high (the blur lowers the raw's variance) and burnt the bright zones, shifting their
        // colour as single channels clipped.
        let photometry: f32 = meta_text
            .split("\"fotometria\":")
            .nth(1)
            .and_then(|r| r.split([',', '\n', '}']).next())
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1.0);
        let mapping = {
            let (a, b) = fit(&lr, &lt, w, sq);
            if surface { (a, b) } else { (1.0 / photometry, -sky(&lr).0 / photometry) }
        };
        // Burnt whites: share of the subject brighter than the truth's white (its 99.9th
        // percentile), on the same scale.
        let over = |l: &[f32], map: (f32, f32)| {
            let inside: Vec<usize> = (0..w * h).filter(|&i| weight[i] > 0.5).collect();
            inside.iter().filter(|&&i| map.0 * l[i] + map.1 > top).count() as f32 / inside.len().max(1) as f32
        };
        let over_white = |l: &[f32]| over(l, mapping);
        // The truth is already on its own scale.
        let burnt = [over(&lt, (1.0, 0.0)), over_white(&lr), over_white(&ls), over_white(&lw), pred_img.as_ref().map_or(f32::NAN, |img| over_white(&ml::lum(img)))];
        eprintln!("  bianchi bruciati: verità {:.4} grezza {:.4} nitida {:.4} ottima {:.4} predetta {:.4}", burnt[0], burnt[1], burnt[2], burnt[3], burnt[4]);
        let _ = writeln!(burnt_tsv, "{name}\t{}", burnt.iter().map(|v| format!("{v:.5}")).collect::<Vec<_>>().join("\t"));
        // The predicted curve applied as the app applies it: the smart sharpening with the
        // network's curve (R, G, B on a colour planet), with and without "Segui le zone", and on
        // the luminance only for comparison.
        if net_app {
            if let Some(g) = predicted.get(&name) {
                let (fc, st, table) = ps_core::curve_net::smart_curve(&ps_core::curve_net::Prediction { gains: g.clone(), spread: 0.0 });
                let base = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
                let subject = ps_core::mask::subject_and_moons(&raw);
                let mut cells = vec![format!("{fc:.4}"), format!("{st:.3}"), format!("{er_smart:.4}"), format!("{:.4}", ce[1]), format!("{:.5}", over_white(&ls))];
                for (colour, adaptive) in [(ps_core::preset::learned_colour(&a), true), (ps_core::preset::learned_colour(&a), false), (ps_core::color::ColorMode::Luminance, true)] {
                    let params = ps_core::smart::SmartParams {
                        curve: ps_core::smart::SmartCurve::Network,
                        network: table.clone(),
                        network_strength: st,
                        network_cutoff: fc,
                        strength: st,
                        color: colour,
                        adaptive,
                        ..base.clone()
                    };
                    let img = ps_core::smart::sharpen(&raw, None, &params, Some(&subject));
                    let l = ml::lum(&img);
                    let (e, al) = region_error(&l, &lt, &on, qvar);
                    cells.extend([format!("{e:.4}"), format!("{:.4}", colour_error(&img, &truth, &on, al)), format!("{:.5}", over_white(&l))]);
                    if adaptive && colour != ps_core::color::ColorMode::Luminance {
                        save(&format!("{name}_rete_app_zoom"), &zoom(&view(&img, mapping, top), grs.0, grs.1, zside, 360));
                    }
                }
                let _ = writeln!(net_tsv, "{name}\t{}", cells.join("\t"));
            }
        }
        let view_of = |img: &Image, _l: &[f32]| view(img, mapping, top);
        if deconv_test {
            let psf = stack_psf(&lr, &lt, w, 20);
            let fwhm = ps_core::psf::kernel_fwhm(&psf);
            let app = ps_core::preset::auto_deconv(a.kind, &a, ps_core::preset::AutoStrength::Normal);
            let truep = ps_core::deconv::DeconvParams { psf: ps_core::psf::Psf::Measured { fwhm }, measured: Some(psf), ..app.clone() };
            // A longer run stopped by the noise (discrepancy principle), as one would with a PSF
            // one trusts.
            let longp = ps_core::deconv::DeconvParams { iterations: 60, auto_stop: true, ..truep.clone() };
            let d_app = deconvolve(&rawr, &app);
            let d_true = deconvolve(&rawr, &truep);
            let d_long = deconvolve(&rawr, &longp);
            let (s_app, ..) = sharpen(&d_app);
            let (s_true, _, p_true) = sharpen(&d_true);
            let (s_long, ..) = sharpen(&d_long);
            let lt_true = ml::lum(&d_true);
            let best_after = image_with_gains(&d_true, &lt_true, &best_filter(&lt_true, &lt, w, &on), &weight);
            let e = |img: &Image| metrics(&ml::lum(img));
            let all = [e(&rawr), e(&sharpr), e(&d_app), e(&s_app), e(&d_true), e(&s_true), e(&s_long), e(&wiener_img), e(&best_after)];
            let _ = writeln!(
                deconv_tsv,
                "{name}\t{fwhm:.2}\t{:.3}\t{:.3}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.5}\t{:.5}",
                p.strength,
                p_true.strength,
                all.iter().map(|m| format!("{:.4}", m.0)).collect::<Vec<_>>().join("\t"),
                all[1].1,
                all[5].1,
                all[7].1,
                over_white(&ls),
                over_white(&ml::lum(&s_true)),
            );
            eprintln!("  deconv: psf fwhm {fwhm:.2} px, iterazioni app {} | errore nitida {:.3}, deconv app+nitida {:.3}, deconv vera {:.3}, vera+nitida {:.3}, vera lunga+nitida {:.3}, ottima {:.3}, ottima dopo deconv {:.3}", app.iterations, all[1].0, all[3].0, all[4].0, all[5].0, all[6].0, all[7].0, all[8].0);
            save(&format!("{name}_deconv_nitida_zoom"), &zoom(&view_of(&s_true, &lr), grs.0, grs.1, zside, 360));
            save(&format!("{name}_deconv_app_nitida_zoom"), &zoom(&view_of(&s_app, &lr), grs.0, grs.1, zside, 360));
        }
        {
            // Dark zones inside the subject (Saturn's ring-globe gap, shadows): their mean level.
            let dark: Vec<usize> = (0..w * h).filter(|&i| weight[i] > 0.5 && lt[i] < 0.03 * top).collect();
            let mean = |l: &[f32]| dark.iter().map(|&i| l[i]).sum::<f32>() / dark.len().max(1) as f32;
            eprintln!("  buio nel soggetto ({} px): grezza {:.4} nitida {:.4} ottima {:.4}, cielo {:.4}", dark.len(), mean(&lr), mean(&ls), mean(&lw), sky(&lr).0);
        }
        save(&format!("{name}_grezza_zoom"), &zoom(&view_of(&rawr, &lr), grs.0, grs.1, zside, 360));
        save(&format!("{name}_nitida_zoom"), &zoom(&view_of(&sharpr, &ls), grs.0, grs.1, zside, 360));
        save(&format!("{name}_ottima_zoom"), &zoom(&view_of(&wiener_img, &lw), grs.0, grs.1, zside, 360));
        if let Some(img) = &pred_img {
            save(&format!("{name}_predetta_zoom"), &zoom(&view_of(img, &lr), grs.0, grs.1, zside, 360));
        }
        save(&format!("{name}_grezza"), &view_of(&rawr, &lr));
        save(&format!("{name}_nitida"), &view_of(&sharpr, &ls));
        eprintln!(
            "{name:<34} {setup:<20} errore {:.3} → nitida {:.3} / ottima {:.3}   SSIM {:.3} → {:.3} / {:.3}   rumore ×{amp_smart:.2} / ×{amp_best:.2}   taglio {:.3}",
            mr.0,
            ms.0,
            mw.0,
            mr.1,
            ms.1,
            mw.1,
            a.spectral.as_ref().map_or(0.0, |s| s.cutoff)
        );
    }
    json.push_str("  ]");
    if let Some(path) = real {
        let raw = ps_io::load(Path::new(&path)).unwrap();
        let (sharp, a, p) = sharpen(&raw);
        let l = ml::lum(&raw);
        let t = percentile_top(&l);
        let c = ml::centre(&raw, &a);
        save("vera_grezza", &view(&raw, (1.0, 0.0), t));
        save("vera_nitida", &view(&sharp, (1.0, 0.0), t));
        save("vera_grezza_zoom", &zoom(&view(&raw, (1.0, 0.0), t), c.0 as usize, c.1 as usize, 160, 360));
        save("vera_nitida_zoom", &zoom(&view(&sharp, (1.0, 0.0), t), c.0 as usize, c.1 as usize, 160, 360));
        let _ = write!(
            json,
            ",\n  \"vera\": {{ \"file\": \"{}\", \"taglio\": {:.4}, \"forza\": {:.3} }}",
            Path::new(&path).file_name().unwrap().to_string_lossy(),
            a.spectral.as_ref().map_or(0.0, |s| s.cutoff),
            p.strength
        );
    }
    json.push_str("\n}\n");
    std::fs::write(out.join("metrics.json"), json).unwrap();
    std::fs::write(out.join("bianchi.tsv"), burnt_tsv).unwrap();
    if net_app {
        std::fs::write(out.join("rete_app.tsv"), net_tsv).unwrap();
    }
    if deconv_test {
        std::fs::write(out.join("deconv.tsv"), deconv_tsv).unwrap();
    }
}
