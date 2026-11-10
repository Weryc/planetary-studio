//! Parametric point-spread functions, sampled into normalised 2-D kernels.

use crate::i18n::tl;
use serde::{Deserialize, Serialize};

const MAX_RADIUS: usize = 64;
const SUPERSAMPLE: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Psf {
    /// Gaussian with standard deviation `sigma` (pixels). A good first guess for seeing.
    Gaussian { sigma: f32 },
    /// Moffat profile (Moffat 1969): `(1 + r²/α²)^-β`. Heavier wings than a Gaussian, closer to
    /// real seeing; β ≈ 2.5–4.5 is typical, β → ∞ tends to a Gaussian.
    Moffat { fwhm: f32, beta: f32 },
    /// Diffraction pattern of a circular aperture with optional central obstruction (fraction of
    /// the aperture diameter).
    Airy { aperture_mm: f32, wavelength_nm: f32, arcsec_per_px: f32, obstruction: f32 },
    /// Measured on the image (a star or small moon); the samples live in
    /// `DeconvParams::measured`. `fwhm` is for display and the fallback.
    Measured { fwhm: f32 },
}

impl Default for Psf {
    fn default() -> Self {
        Psf::Gaussian { sigma: 1.5 }
    }
}

/// Square kernel of side `2·radius + 1`, row-major, summing to 1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Kernel2D {
    pub radius: usize,
    pub data: Vec<f32>,
}

impl Kernel2D {
    pub fn size(&self) -> usize {
        2 * self.radius + 1
    }
}

impl Psf {
    /// The same shape `factor` times wider (a measured PSF can't be rescaled).
    pub fn scaled(self, factor: f32) -> Psf {
        match self {
            Psf::Gaussian { sigma } => Psf::Gaussian { sigma: sigma * factor },
            Psf::Moffat { fwhm, beta } => Psf::Moffat { fwhm: fwhm * factor, beta },
            Psf::Airy { aperture_mm, wavelength_nm, arcsec_per_px, obstruction } => Psf::Airy { aperture_mm, wavelength_nm, arcsec_per_px: arcsec_per_px / factor.max(1e-3), obstruction },
            Psf::Measured { .. } => self,
        }
    }

    pub fn kernel(&self) -> Kernel2D {
        match *self {
            // Fallback when the measured samples are missing.
            Psf::Measured { fwhm } => Psf::Gaussian { sigma: fwhm / 2.3548 }.kernel(),
            Psf::Gaussian { sigma } => {
                let s = sigma.max(0.1) as f64;
                let radius = (3.5 * s).ceil() as usize;
                sample(radius, |r| (-r * r / (2.0 * s * s)).exp())
            }
            Psf::Moffat { fwhm, beta } => {
                let beta = beta.max(1.01) as f64;
                let alpha = moffat_alpha(fwhm as f64, beta);
                // Extend until the profile drops below 0.1 % of the peak.
                let radius = (alpha * (1e-3f64.powf(-1.0 / beta) - 1.0).sqrt()).ceil() as usize;
                sample(radius, |r| (1.0 + r * r / (alpha * alpha)).powf(-beta))
            }
            Psf::Airy { aperture_mm, wavelength_nm, arcsec_per_px, obstruction } => {
                let first_null = self.airy_first_null_px().unwrap_or(1.0) as f64;
                let rad_per_px = arcsec_per_px.max(1e-4) as f64 * std::f64::consts::PI / (180.0 * 3600.0);
                let d = aperture_mm.max(1.0) as f64 * 1e-3;
                let lambda = wavelength_nm.max(100.0) as f64 * 1e-9;
                let eps = obstruction.clamp(0.0, 0.9) as f64;
                let radius = (3.0 * first_null).ceil() as usize;
                sample(radius, |r| {
                    let x = std::f64::consts::PI * d * r * rad_per_px / lambda;
                    let a = (airy_amplitude(x) - eps * eps * airy_amplitude(eps * x)) / (1.0 - eps * eps);
                    a * a
                })
            }
        }
    }

    /// The same shape with full width at half maximum `fwhm` (px). A measured PSF can't be
    /// resized and is returned as it is.
    pub fn with_fwhm(self, fwhm: f32) -> Psf {
        let current = self.fwhm_px();
        if current <= 0.0 || matches!(self, Psf::Measured { .. }) {
            return self;
        }
        self.scaled(fwhm.max(0.2) / current)
    }

    /// Full width at half maximum in pixels, for display.
    pub fn fwhm_px(&self) -> f32 {
        match *self {
            Psf::Gaussian { sigma } => 2.3548 * sigma,
            Psf::Moffat { fwhm, .. } => fwhm,
            // FWHM of an unobstructed Airy disk is 1.029 λ/D; the first null is at 1.22 λ/D.
            Psf::Airy { .. } => self.airy_first_null_px().unwrap_or(0.0) * 1.029 / 1.22,
            Psf::Measured { fwhm } => fwhm,
        }
    }

    fn airy_first_null_px(&self) -> Option<f32> {
        match *self {
            Psf::Airy { aperture_mm, wavelength_nm, arcsec_per_px, .. } => {
                let rad = 1.22 * wavelength_nm.max(100.0) as f64 * 1e-9 / (aperture_mm.max(1.0) as f64 * 1e-3);
                let arcsec = rad * 180.0 * 3600.0 / std::f64::consts::PI;
                Some((arcsec / arcsec_per_px.max(1e-4) as f64) as f32)
            }
            _ => None,
        }
    }
}

impl Psf {
    /// A sum of Gaussians (weight, σ) close to this PSF, so blurring with it can go through fast
    /// separable passes: a Gaussian is one term; a Moffat's wide wings take a few wider ones,
    /// fitted to its radial profile. `None` for shapes that aren't radially smooth (Airy rings,
    /// a measured PSF) or when the fit is poor.
    pub fn gaussian_mixture(&self) -> Option<Vec<(f32, f32)>> {
        match *self {
            Psf::Gaussian { sigma } => Some(vec![(1.0, sigma.max(0.1))]),
            Psf::Moffat { fwhm, beta } => {
                let beta = beta.max(1.01) as f64;
                let alpha = moffat_alpha(fwhm as f64, beta);
                let core = fwhm as f64 / 2.3548;
                // A ladder of widths from inside the core out to the wings.
                let sigmas: Vec<f64> = [0.55, 0.9, 1.5, 2.6, 4.5].iter().map(|f| f * core).collect();
                let reach = (alpha * (1e-3f64.powf(-1.0 / beta) - 1.0).sqrt()).max(3.0 * core);
                let samples: Vec<f64> = (0..200).map(|i| reach * i as f64 / 199.0).collect();
                // Fit the 2-D profiles (each term normalised to unit volume), weighting by r: the
                // volume at radius r is what blurring moves around.
                let moffat_norm = (beta - 1.0) / (std::f64::consts::PI * alpha * alpha);
                let target: Vec<f64> = samples.iter().map(|&r| moffat_norm * (1.0 + r * r / (alpha * alpha)).powf(-beta)).collect();
                let basis: Vec<Vec<f64>> = sigmas.iter().map(|&sg| samples.iter().map(|&r| (-(r * r) / (2.0 * sg * sg)).exp() / (2.0 * std::f64::consts::PI * sg * sg)).collect()).collect();
                let weights = nonnegative_least_squares(&basis, &target, &samples)?;
                let total: f64 = weights.iter().sum();
                if total <= 0.0 {
                    return None;
                }
                // Relative error of the fitted profile, by volume.
                let (mut err, mut norm) = (0.0, 0.0);
                for (i, &r) in samples.iter().enumerate() {
                    let fit: f64 = weights.iter().zip(&basis).map(|(w, b)| w * b[i]).sum();
                    err += r * (fit - target[i]).abs();
                    norm += r * target[i];
                }
                (err / norm < 0.05).then(|| weights.iter().zip(&sigmas).filter(|(w, _)| **w > 1e-4 * total).map(|(w, s)| ((w / total) as f32, *s as f32)).collect())
            }
            Psf::Airy { .. } | Psf::Measured { .. } => None,
        }
    }
}

/// min Σ rᵢ (Σₖ wₖ bₖ(rᵢ) − tᵢ)² with w ≥ 0, by projected coordinate descent.
fn nonnegative_least_squares(basis: &[Vec<f64>], target: &[f64], r: &[f64]) -> Option<Vec<f64>> {
    let k = basis.len();
    let gram = |a: &[f64], b: &[f64]| a.iter().zip(b).zip(r).map(|((x, y), w)| x * y * w).sum::<f64>();
    let g: Vec<Vec<f64>> = (0..k).map(|i| (0..k).map(|j| gram(&basis[i], &basis[j])).collect()).collect();
    let c: Vec<f64> = (0..k).map(|i| gram(&basis[i], target)).collect();
    let mut w = vec![0.0f64; k];
    for _ in 0..2000 {
        for i in 0..k {
            if g[i][i] <= 0.0 {
                return None;
            }
            let rest: f64 = (0..k).filter(|&j| j != i).map(|j| g[i][j] * w[j]).sum();
            w[i] = ((c[i] - rest) / g[i][i]).max(0.0);
        }
    }
    Some(w)
}

/// The PSF as an unresolved source (a star, a small moon) shows it: the patch around the
/// brightest pixel within 12 px of (`x`, `y`), sky level removed, centred on its centroid to a
/// fraction of a pixel and normalised. `None` if there is no clear point source there.
pub fn measure(plane: &[f32], w: usize, h: usize, x: usize, y: usize) -> Option<Kernel2D> {
    let at = |x: isize, y: isize| plane[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    // Brightest pixel near the click.
    let (mut px, mut py, mut peak) = (x as isize, y as isize, f32::MIN);
    for dy in -12..=12 {
        for dx in -12..=12 {
            let v = at(x as isize + dx, y as isize + dy);
            if v > peak {
                (px, py, peak) = (x as isize + dx, y as isize + dy, v);
            }
        }
    }
    measure_at(plane, w, h, px, py).map(|(k, _)| k)
}

/// What [`measure_in`] found.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    pub kernel: Kernel2D,
    /// Centre of the source in the image (px).
    pub x: f32,
    pub y: f32,
    pub fwhm: f32,
    /// Ratio of the narrowest to the widest axis of the core (1 = round).
    pub roundness: f32,
    /// Peak height above the sky, relative to the brightest pixel in the image.
    pub brightness: f32,
}

/// Measure the PSF on the brightest compact source inside the rectangle `x0..x1`, `y0..y1`:
/// local maxima are tried from the brightest, skipping anything too extended to be a point
/// (a planet's limb, a crater). The error says why nothing could be used.
pub fn measure_in(plane: &[f32], w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Result<Measurement, &'static str> {
    let (x1, y1) = (x1.min(w), y1.min(h));
    if x1 <= x0 + 2 || y1 <= y0 + 2 {
        return Err(tl("Il riquadro è troppo piccolo."));
    }
    let image_peak = plane.iter().copied().fold(f32::MIN, f32::max).max(1e-6);
    let mut maxima: Vec<(f32, usize, usize)> = Vec::new();
    for y in y0.max(1)..(y1).min(h - 1) {
        for x in x0.max(1)..(x1).min(w - 1) {
            let v = plane[y * w + x];
            let around = [plane[y * w + x - 1], plane[y * w + x + 1], plane[(y - 1) * w + x], plane[(y + 1) * w + x]];
            // A peak, not a flat stretch (a planet's disk has plenty of those).
            if around.iter().all(|&a| v >= a) && around.iter().any(|&a| v > a) {
                maxima.push((v, x, y));
            }
        }
    }
    maxima.sort_by(|a, b| b.0.total_cmp(&a.0));
    // One try per object: nearby maxima belong to the same one.
    let mut tried: Vec<(usize, usize)> = Vec::new();
    let mut extended = false;
    for &(_, x, y) in &maxima {
        if tried.len() >= 300 {
            break;
        }
        if tried.iter().any(|&(tx, ty)| tx.abs_diff(x) < 10 && ty.abs_diff(y) < 10) {
            continue;
        }
        tried.push((x, y));
        match measure_at(plane, w, h, x as isize, y as isize) {
            // Wider than any blur: a bright patch of a planet or of the Moon.
            Some((kernel, _)) if kernel_fwhm(&kernel) > 16.0 => extended = true,
            Some((kernel, (cx, cy, height, roundness))) => {
                let fwhm = kernel_fwhm(&kernel);
                return Ok(Measurement { kernel, x: cx, y: cy, fwhm, roundness, brightness: height / image_peak });
            }
            None => extended = true,
        }
    }
    Err(if extended {
        tl("Nel riquadro non c'è un oggetto piccolo e isolato: solo bordi o zone estese. Prova attorno a una stella o a una piccola luna, con un po' di cielo intorno.")
    } else {
        tl("Nel riquadro non c'è niente di più luminoso del fondo.")
    })
}

/// The patch around the peak at (`px`, `py`), with its centre, height above the sky and
/// roundness; `None` if it isn't a compact source.
fn measure_at(plane: &[f32], w: usize, h: usize, px: isize, py: isize) -> Option<(Kernel2D, (f32, f32, f32, f32))> {
    let at = |x: isize, y: isize| plane[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    let peak = at(px, py);
    // Sky level: median of a ring well outside the source.
    let ring: Vec<f32> = {
        let mut v: Vec<f32> = (0..64)
            .map(|k| {
                let a = k as f32 * std::f32::consts::TAU / 64.0;
                at(px + (40.0 * a.cos()) as isize, py + (40.0 * a.sin()) as isize)
            })
            .collect();
        v.sort_by(f32::total_cmp);
        v
    };
    let sky = ring[ring.len() / 2];
    let height = peak - sky;
    if height <= 0.0 {
        return None;
    }
    // Extent: where the profile falls to 2 % of the peak, in the four directions.
    let mut reach = 0isize;
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let r = (1..40).find(|&r| at(px + dx * r, py + dy * r) - sky < 0.02 * height).unwrap_or(40);
        reach = reach.max(r);
    }
    if reach >= 40 {
        return None; // Not a compact source (an edge, a disk…).
    }
    let radius = ((reach as f32 * 1.5).ceil() as usize).clamp(3, MAX_RADIUS.min(32));
    // Sub-pixel centroid of the core, and its second moments for the roundness.
    let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for dy in -(reach)..=reach {
        for dx in -(reach)..=reach {
            let v = ((at(px + dx, py + dy) - sky) as f64).max(0.0);
            let (fx, fy) = (dx as f64, dy as f64);
            (sx, sy, sw) = (sx + v * fx, sy + v * fy, sw + v);
            (sxx, syy, sxy) = (sxx + v * fx * fx, syy + v * fy * fy, sxy + v * fx * fy);
        }
    }
    let (cx, cy) = (px as f32 + (sx / sw) as f32, py as f32 + (sy / sw) as f32);
    let (mx, my) = (sx / sw, sy / sw);
    let (vxx, vyy, vxy) = (sxx / sw - mx * mx, syy / sw - my * my, sxy / sw - mx * my);
    let (mean, diff) = ((vxx + vyy) / 2.0, (((vxx - vyy) / 2.0).powi(2) + vxy * vxy).sqrt());
    let roundness = ((mean - diff).max(0.0) / (mean + diff).max(1e-12)).sqrt() as f32;
    let bilinear = |x: f32, y: f32| {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as isize, y0 as isize);
        at(x0, y0) * (1.0 - fx) * (1.0 - fy) + at(x0 + 1, y0) * fx * (1.0 - fy) + at(x0, y0 + 1) * (1.0 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy
    };
    let size = 2 * radius + 1;
    let r = radius as f32;
    let mut data: Vec<f32> = (0..size * size)
        .map(|i| {
            let (dx, dy) = ((i % size) as f32 - r, (i / size) as f32 - r);
            // A round window: the corners hold only sky.
            if dx.hypot(dy) > r + 0.5 { 0.0 } else { (bilinear(cx + dx, cy + dy) - sky).max(0.0) }
        })
        .collect();
    let sum: f32 = data.iter().sum();
    if sum <= 0.0 {
        return None;
    }
    data.iter_mut().for_each(|v| *v /= sum);
    Some((Kernel2D { radius, data }, (cx, cy, height, roundness)))
}

/// Full width at half maximum of a kernel, from the area above half its peak (px).
pub fn kernel_fwhm(k: &Kernel2D) -> f32 {
    let peak = k.data.iter().copied().fold(0.0f32, f32::max);
    let above = k.data.iter().filter(|v| **v >= 0.5 * peak).count() as f32;
    2.0 * (above / std::f32::consts::PI).sqrt()
}

fn moffat_alpha(fwhm: f64, beta: f64) -> f64 {
    fwhm.max(0.2) / (2.0 * (2f64.powf(1.0 / beta) - 1.0).sqrt())
}

/// Sample a radial profile over each pixel's area (SUPERSAMPLE² points) and normalise.
fn sample(radius: usize, profile: impl Fn(f64) -> f64) -> Kernel2D {
    let radius = radius.clamp(1, MAX_RADIUS);
    let size = 2 * radius + 1;
    let mut data = vec![0.0f32; size * size];
    let n = SUPERSAMPLE as f64;
    for ky in 0..size {
        for kx in 0..size {
            let mut acc = 0.0;
            for sy in 0..SUPERSAMPLE {
                for sx in 0..SUPERSAMPLE {
                    let dx = kx as f64 - radius as f64 + (sx as f64 + 0.5) / n - 0.5;
                    let dy = ky as f64 - radius as f64 + (sy as f64 + 0.5) / n - 0.5;
                    acc += profile((dx * dx + dy * dy).sqrt());
                }
            }
            data[ky * size + kx] = acc as f32;
        }
    }
    let sum: f32 = data.iter().sum();
    data.iter_mut().for_each(|v| *v /= sum);
    Kernel2D { radius, data }
}

/// `2·J1(x)/x`, the Airy amplitude, → 1 at x = 0.
fn airy_amplitude(x: f64) -> f64 {
    if x.abs() < 1e-8 { 1.0 } else { 2.0 * bessel_j1(x) / x }
}

/// Bessel J1 via the rational/asymptotic approximations of Numerical Recipes (|error| < 1e-8).
fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let n = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606)))));
        let d = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
        n / d
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 3.0 * std::f64::consts::FRAC_PI_4;
        let p = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
        let q = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let ans = (std::f64::consts::FRAC_2_PI / ax).sqrt() * (xx.cos() * p - z * xx.sin() * q);
        if x < 0.0 { -ans } else { ans }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A planet's disk and a small moon in the box: the moon is measured, not the limb; a box
    /// holding only the disk's edge says why nothing could be used.
    #[test]
    fn measures_the_moon_in_a_box_and_not_the_limb() {
        let (w, h) = (200, 120);
        let moon = Psf::Gaussian { sigma: 1.6 }.kernel();
        let mut plane = vec![0.01f32; w * h];
        for y in 0..h {
            for x in 0..w {
                if (x as f32 - 60.0).hypot(y as f32 - 60.0) < 40.0 {
                    plane[y * w + x] = 0.8;
                }
            }
        }
        let r = moon.radius as isize;
        for ky in -r..=r {
            for kx in -r..=r {
                let i = ((70 + ky) as usize) * w + (150 + kx) as usize;
                plane[i] += 2.0 * moon.data[((ky + r) as usize) * moon.size() + (kx + r) as usize];
            }
        }
        let m = measure_in(&plane, w, h, 0, 0, w, h).expect("the moon");
        assert!((m.x - 150.0).abs() < 1.0 && (m.y - 70.0).abs() < 1.0, "{} {}", m.x, m.y);
        assert!((m.fwhm - 2.3548 * 1.6).abs() < 0.8, "fwhm {}", m.fwhm);
        assert!(m.roundness > 0.9);
        assert!(measure_in(&plane, w, h, 80, 40, 110, 80).is_err());
    }

    #[test]
    fn kernels_are_normalised_and_peak_in_centre() {
        let psfs = [Psf::Gaussian { sigma: 1.5 }, Psf::Moffat { fwhm: 3.0, beta: 3.0 }, Psf::Airy { aperture_mm: 235.0, wavelength_nm: 550.0, arcsec_per_px: 0.1, obstruction: 0.33 }];
        for psf in psfs {
            let k = psf.kernel();
            let sum: f32 = k.data.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "{psf:?} sums to {sum}");
            let centre = k.data[k.radius * k.size() + k.radius];
            assert!(k.data.iter().all(|&v| v <= centre + 1e-7), "{psf:?} peak off-centre");
        }
    }

    /// The Gaussian sum blurs like the Moffat it stands for.
    #[test]
    fn gaussian_mixture_matches_moffat() {
        for (fwhm, beta) in [(4.0, 2.0), (7.0, 1.75), (3.0, 4.0)] {
            let psf = Psf::Moffat { fwhm, beta };
            let mix = psf.gaussian_mixture().expect("fits");
            let exact = psf.kernel();
            let size = exact.size();
            let r = exact.radius as f32;
            // Rebuild the mixture's kernel on the same grid (pixel centres, no supersampling).
            let approx: Vec<f32> = (0..size * size)
                .map(|i| {
                    let (x, y) = ((i % size) as f32 - r, (i / size) as f32 - r);
                    mix.iter().map(|(w, s)| w * (-(x * x + y * y) / (2.0 * s * s)).exp() / (2.0 * std::f32::consts::PI * s * s)).sum()
                })
                .collect();
            let diff: f32 = exact.data.iter().zip(&approx).map(|(a, b)| (a - b).abs()).sum();
            assert!(diff < 0.08, "fwhm {fwhm}, β {beta}: L1 difference {diff} with {} terms", mix.len());
        }
    }

    #[test]
    fn measures_a_star() {
        let (w, h) = (120, 100);
        let truth = Psf::Gaussian { sigma: 2.0 }.kernel();
        let mut plane = vec![0.02f32; w * h];
        let (sx, sy) = (60usize, 50usize);
        let r = truth.radius;
        for ky in 0..truth.size() {
            for kx in 0..truth.size() {
                plane[(sy + ky - r) * w + sx + kx - r] += 3.0 * truth.data[ky * truth.size() + kx];
            }
        }
        let k = measure(&plane, w, h, sx + 5, sy - 4).expect("a star");
        assert!((kernel_fwhm(&k) - 2.3548 * 2.0).abs() < 0.8, "fwhm {}", kernel_fwhm(&k));
        let c = k.radius * k.size() + k.radius;
        assert!(k.data.iter().all(|&v| v <= k.data[c] + 1e-6), "peak in the centre");
        assert!(measure(&vec![0.3; w * h], w, h, 60, 50).is_none(), "nothing to measure on a flat field");
    }

    #[test]
    fn bessel_first_zero() {
        assert!(bessel_j1(3.8317).abs() < 1e-4);
        assert!((bessel_j1(1.0) - 0.4400506).abs() < 1e-6);
    }

    #[test]
    fn airy_fwhm_for_c9_25() {
        // 235 mm at 550 nm: λ/D ≈ 0.483"; FWHM ≈ 0.497" → ≈ 5 px at 0.1"/px.
        let psf = Psf::Airy { aperture_mm: 235.0, wavelength_nm: 550.0, arcsec_per_px: 0.1, obstruction: 0.0 };
        assert!((psf.fwhm_px() - 4.97).abs() < 0.05, "{}", psf.fwhm_px());
    }
}
