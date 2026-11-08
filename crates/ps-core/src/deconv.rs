//! Deconvolution: Richardson–Lucy (with Biggs–Andrews acceleration, TV regularisation and
//! White's damping), Wiener and Van Cittert. All run on a mirror-padded plane. Blurring with
//! the PSF goes through [`FftConvolver`], except for a Gaussian PSF in the iterative methods:
//! a Gaussian is separable, so two 1-D passes do the same job several times faster.

use crate::color::ColorMode;
use crate::conv::convolve_separable;
use crate::fft::{FftConvolver, Padded};
use crate::psf::{Kernel2D, Psf};
use crate::wavelet::estimate_noise;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

const EPS: f32 = 1e-7;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeconvAlgorithm {
    /// Van Cittert was dropped: it amplified noise early and was never the better choice;
    /// saved settings that used it open with Richardson–Lucy.
    #[serde(alias = "VanCittert")]
    RichardsonLucy,
    Wiener,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeconvParams {
    pub algorithm: DeconvAlgorithm,
    pub psf: Psf,
    pub iterations: u32,
    /// Biggs & Andrews (1997) vector extrapolation for RL.
    pub accelerate: bool,
    /// Total-variation weight λ (Dey et al. 2006); 0 = off, ≈ 0.002 typical.
    pub tv_lambda: f32,
    /// White (1994) damping threshold in units of the noise σ; 0 = off.
    pub damping: f32,
    /// Wiener noise-to-signal ratio K.
    pub wiener_k: f32,
    pub color: ColorMode,
    /// Treat `iterations` as a maximum and stop once the model explains the data down to the
    /// measured noise (Morozov's discrepancy principle): going on would only fit noise.
    #[serde(default)]
    pub auto_stop: bool,
    /// Anti-ringing during the iterations (0‥1): in the deringing area the estimate may not
    /// leave the local range of the observed image (PixInsight's local deringing idea).
    #[serde(default)]
    pub deringing: f32,
    /// Wiener: K from the measured noise and signal instead of `wiener_k`.
    #[serde(default)]
    pub auto_k: bool,
    /// R·G·B mode: PSF size factor for each channel (seeing blurs blue more than red).
    #[serde(default = "ones")]
    pub channel_scale: [f32; 3],
    /// The samples of a PSF measured on the image, for `Psf::Measured`.
    #[serde(default)]
    pub measured: Option<Kernel2D>,
}

fn ones() -> [f32; 3] {
    [1.0; 3]
}

impl Default for DeconvParams {
    fn default() -> Self {
        Self {
            algorithm: DeconvAlgorithm::RichardsonLucy,
            psf: Psf::default(),
            iterations: 20,
            accelerate: true,
            tv_lambda: 0.0,
            damping: 0.0,
            wiener_k: 0.01,
            color: ColorMode::Luminance,
            auto_stop: false,
            deringing: 0.0,
            auto_k: false,
            channel_scale: [1.0; 3],
            measured: None,
        }
    }
}

impl DeconvParams {
    /// No iterations: nothing to do (Wiener has no such setting and is never neutral).
    pub fn is_neutral(&self) -> bool {
        self.iterations == 0 && self.algorithm != DeconvAlgorithm::Wiener
    }

    /// The PSF's samples, `scale` times wider (a measured PSF is used as it is).
    pub fn kernel(&self, scale: f32) -> Kernel2D {
        match (&self.psf, &self.measured) {
            (Psf::Measured { .. }, Some(k)) => k.clone(),
            (psf, _) => psf.scaled(scale).kernel(),
        }
    }
}

/// Per-plane extras for [`deconvolve_plane_with`].
#[derive(Clone, Copy)]
pub struct PlaneOptions<'a> {
    /// PSF size factor for this plane (R·G·B mode).
    pub psf_scale: f32,
    /// Where anti-ringing may act (the deringing area, same size as the plane); computed from
    /// the plane when missing.
    pub area: Option<&'a [f32]>,
}

impl Default for PlaneOptions<'_> {
    fn default() -> Self {
        Self { psf_scale: 1.0, area: None }
    }
}

/// What a deconvolution reports besides the image.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DeconvReport {
    /// Iterations actually run (fewer than asked when stopped automatically).
    pub iterations: u32,
}

/// Safeguards applied inside Richardson–Lucy (shared with the GPU version).
pub(crate) struct Guards {
    /// Stop once the mean squared residual over the subject (data above `signal`) reaches
    /// `target` (the noise variance).
    pub stop: Option<(f32, f64)>,
    /// Anti-ringing: allowed range (`lo`, `hi`, already widened by the tolerance) and where
    /// and how much to hold the estimate in it.
    pub clamp: Option<Clamp>,
}

pub(crate) struct Clamp {
    pub lo: Vec<f32>,
    pub hi: Vec<f32>,
    /// Strength × area, per pixel.
    pub weight: Vec<f32>,
}

impl Clamp {
    fn apply(&self, x: &mut [f32]) {
        x.par_iter_mut().zip(self.lo.par_iter().zip(&self.hi)).zip(&self.weight).for_each(|((v, (&lo, &hi)), &k)| {
            if k > 0.0 {
                *v += k * (v.clamp(lo, hi) - *v);
            }
        });
    }
}

/// Deconvolve one plane. `progress` receives 0‥1; returns `None` if cancelled.
pub fn deconvolve_plane(plane: &[f32], w: usize, h: usize, p: &DeconvParams, cancel: &dyn Fn() -> bool, progress: &dyn Fn(f32)) -> Option<Vec<f32>> {
    deconvolve_plane_with(plane, w, h, p, PlaneOptions::default(), cancel, progress).map(|(out, _)| out)
}

/// [`deconvolve_plane`] with per-plane options, also reporting what was done.
pub fn deconvolve_plane_with(plane: &[f32], w: usize, h: usize, p: &DeconvParams, options: PlaneOptions, cancel: &dyn Fn() -> bool, progress: &dyn Fn(f32)) -> Option<(Vec<f32>, DeconvReport)> {
    let kernel = p.kernel(options.psf_scale);
    let margin = (2 * kernel.radius).max(16);
    let padded = Padded::new(plane, w, h, margin);
    let observed: Vec<f32> = padded.data.par_iter().map(|v| v.max(0.0)).collect();
    // Gaussians (and Moffats, as sums of Gaussians) take fast separable passes; other shapes FFT.
    let blur = || match p.psf {
        Psf::Gaussian { .. } => Blur::separable(&kernel, padded.w, padded.h),
        psf => match psf.scaled(options.psf_scale).gaussian_mixture() {
            Some(terms) => Blur::mixture(&terms, padded.w, padded.h),
            None => Blur::Fft(FftConvolver::new(padded.w, padded.h, &kernel)),
        },
    };
    let needs_noise = p.damping > 0.0 || p.auto_stop || (p.auto_k && p.algorithm == DeconvAlgorithm::Wiener);
    let noise = if needs_noise { estimate_noise(plane, w, h) } else { 0.0 };

    let (out, iterations) = match p.algorithm {
        DeconvAlgorithm::RichardsonLucy => {
            let damp = p.damping * noise;
            let guards = Guards {
                stop: p.auto_stop.then(|| (signal_level(plane), (noise as f64).powi(2))),
                clamp: (p.deringing > 0.0).then(|| anti_ringing(plane, w, h, &padded, &observed, p, &kernel, options.area)),
            };
            let blur = blur();
            match gpu_richardson_lucy(&observed, &blur, padded.w, padded.h, p, damp, &guards, cancel, progress) {
                Some(result) => result?,
                None => richardson_lucy(&observed, &blur, padded.w, padded.h, p, damp, &guards, cancel, progress)?,
            }
        }
        DeconvAlgorithm::Wiener => {
            let k = if p.auto_k { wiener_k(plane, noise) } else { p.wiener_k };
            (wiener(&observed, &FftConvolver::new(padded.w, padded.h, &kernel), k), 0)
        }
    };
    progress(1.0);
    Some((padded.crop(&out), DeconvReport { iterations }))
}

/// A level separating the subject from the sky: the faint sky would otherwise make the mean
/// residual look small long before the subject is fitted.
fn signal_level(plane: &[f32]) -> f32 {
    let stride = plane.len().div_ceil(1 << 18).max(1);
    let mut sample: Vec<f32> = plane.iter().step_by(stride).copied().collect();
    let n = sample.len();
    let sky = *sample.select_nth_unstable_by(n / 5, f32::total_cmp).1;
    let bright = *sample.select_nth_unstable_by(n * 99 / 100, f32::total_cmp).1;
    sky + 0.1 * (bright - sky)
}

/// Wiener's K = noise power / signal power, both measured on the plane.
fn wiener_k(plane: &[f32], noise: f32) -> f32 {
    let n = plane.len() as f64;
    let mean = plane.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = plane.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let noise_var = (noise as f64).powi(2);
    (noise_var / (var - noise_var).max(1e-12)).clamp(1e-4, 0.5) as f32
}

/// Ranges and weights for anti-ringing, in the padded layout: the observed image's local
/// minimum and maximum over about the blur's size, widened by a tenth of their span, held in
/// the deringing area (near sharp edges).
#[allow(clippy::too_many_arguments)]
fn anti_ringing(plane: &[f32], w: usize, h: usize, padded: &Padded, observed: &[f32], p: &DeconvParams, kernel: &Kernel2D, area: Option<&[f32]>) -> Clamp {
    let sigma = (crate::psf::kernel_fwhm(kernel) / 2.3548).max(0.5);
    let r = sigma.round().clamp(1.0, 8.0) as usize;
    let (mut lo, mut hi) = crate::dering::min_max(observed, padded.w, padded.h, r);
    lo.par_iter_mut().zip(hi.par_iter_mut()).for_each(|(l, u)| {
        let margin = 0.1 * (*u - *l);
        (*l, *u) = (*l - margin, *u + margin);
    });
    let area = match area.filter(|a| a.len() == w * h) {
        Some(a) => a.to_vec(),
        None => crate::dering::weight(&crate::Image::new(w, h, vec![plane.to_vec()], 16), &crate::dering::DeringParams::for_blur(sigma)),
    };
    let strength = p.deringing.clamp(0.0, 1.0);
    let weight = Padded::new(&area, w, h, padded.margin()).data.into_iter().map(|a| a * strength).collect();
    Clamp { lo, hi, weight }
}

/// Rows with `rows`, then columns with `cols` (equal for a plain Gaussian).
fn separable_pass(img: &[f32], dst: &mut [f32], tmp: &mut [f32], w: usize, h: usize, rows: &[f32], cols: &[f32]) {
    if rows == cols {
        convolve_separable(img, dst, tmp, w, h, rows, 1);
    } else {
        crate::conv::convolve_rows_then_cols(img, dst, tmp, w, h, rows, cols);
    }
}

/// Richardson–Lucy on the GPU when there is one and the PSF is separable. `None`: not
/// possible here (or the GPU failed), use the CPU; `Some(None)`: cancelled.
#[allow(clippy::too_many_arguments)]
fn gpu_richardson_lucy(d: &[f32], blur: &Blur, w: usize, h: usize, p: &DeconvParams, damp: f32, guards: &Guards, cancel: &dyn Fn() -> bool, progress: &dyn Fn(f32)) -> Option<Option<(Vec<f32>, u32)>> {
    let Blur::Separable { terms, .. } = blur else {
        return None;
    };
    let gpu = crate::gpu::get()?;
    gpu.richardson_lucy(d, w, h, terms, p, damp, guards, cancel, progress).ok()
}

/// Blurring with the PSF (`convolve`) and with its mirror image (`correlate`).
enum Blur {
    Fft(FftConvolver),
    /// A sum of separable terms, each given by its row and column taps (mirror edges): one
    /// for a Gaussian, a few for a Moffat approximated by Gaussians.
    Separable {
        terms: Vec<SeparableTerm>,
        w: usize,
        h: usize,
        scratch: Mutex<Vec<f32>>,
    },
}

/// Row taps and column taps of one separable term (the weight is in the column taps).
pub(crate) type SeparableTerm = (Vec<f32>, Vec<f32>);

/// The kernel's column sums: for a separable kernel K(x, y) = a(x)·a(y), exactly a.
fn column_sums(kernel: &Kernel2D) -> Vec<f32> {
    let size = kernel.size();
    (0..size).map(|x| (0..size).map(|y| kernel.data[y * size + x]).sum()).collect()
}

impl Blur {
    fn separable(kernel: &Kernel2D, w: usize, h: usize) -> Self {
        let taps = column_sums(kernel);
        Blur::Separable { terms: vec![(taps.clone(), taps)], w, h, scratch: Mutex::new(Vec::new()) }
    }

    /// Σ weight · Gaussian(σ), each Gaussian sampled like [`Psf::kernel`] does.
    fn mixture(terms: &[(f32, f32)], w: usize, h: usize) -> Self {
        let terms = terms
            .iter()
            .map(|&(weight, sigma)| {
                let taps = column_sums(&Psf::Gaussian { sigma }.kernel());
                let weighted = taps.iter().map(|t| t * weight).collect();
                (taps, weighted)
            })
            .collect();
        Blur::Separable { terms, w, h, scratch: Mutex::new(Vec::new()) }
    }

    #[cfg(test)]
    fn convolve(&self, img: &[f32]) -> Vec<f32> {
        let (mut out, mut tmp) = (vec![0.0f32; img.len()], vec![0.0f32; img.len()]);
        self.convolve_into(img, &mut out, &mut tmp);
        out
    }

    /// `dst = img ⊛ psf`; `tmp` is scratch of the same size.
    fn convolve_into(&self, img: &[f32], dst: &mut [f32], tmp: &mut [f32]) {
        match self {
            Blur::Fft(conv) => dst.copy_from_slice(&conv.convolve(img)),
            Blur::Separable { terms, w, h, scratch } => {
                let (rows, cols) = &terms[0];
                separable_pass(img, dst, tmp, *w, *h, rows, cols);
                if terms.len() > 1 {
                    let mut part = scratch.lock().expect("not poisoned");
                    part.resize(img.len(), 0.0);
                    for (rows, cols) in &terms[1..] {
                        separable_pass(img, &mut part, tmp, *w, *h, rows, cols);
                        dst.par_iter_mut().zip(part.par_iter()).for_each(|(d, p)| *d += p);
                    }
                }
            }
        }
    }

    /// `dst = img ⋆ psf`; `tmp` is scratch of the same size.
    fn correlate_into(&self, img: &[f32], dst: &mut [f32], tmp: &mut [f32]) {
        match self {
            Blur::Fft(conv) => dst.copy_from_slice(&conv.correlate(img)),
            // Symmetric taps: the mirror image is the PSF itself.
            Blur::Separable { .. } => self.convolve_into(img, dst, tmp),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn richardson_lucy(d: &[f32], conv: &Blur, w: usize, h: usize, p: &DeconvParams, damp_threshold: f32, guards: &Guards, cancel: &dyn Fn() -> bool, progress: &dyn Fn(f32)) -> Option<(Vec<f32>, u32)> {
    let iterations = p.iterations.max(1);
    let n = d.len();
    // Every buffer is allocated once: at 12 MP each is ~50 MB, and fresh memory is slow.
    let mut x: Vec<f32> = d.iter().map(|v| v.max(EPS)).collect();
    let mut x_prev = x.clone();
    let (mut y, mut ratio, mut correction, mut tmp) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    // Last two RL corrections g_{k-1} (g1) and g_{k-2} (g2) for the Biggs–Andrews step length.
    let (mut g1, mut g2) = if p.accelerate { (vec![0.0f32; n], vec![0.0f32; n]) } else { (Vec::new(), Vec::new()) };
    let mut corrections = 0;
    let mut tv = (p.tv_lambda > 0.0).then(|| TvScratch::new(n));

    for k in 0..iterations {
        if cancel() {
            return None;
        }
        let alpha = if p.accelerate && corrections >= 2 {
            let (num, den) = g1.par_iter().zip(&g2).map(|(&a, &b)| ((a * b) as f64, (b * b) as f64)).reduce(|| (0.0, 0.0), |l, r| (l.0 + r.0, l.1 + r.1));
            // α < 1 keeps the extrapolation stable (Biggs & Andrews cap it just below 1).
            if den > 0.0 { (num / den).clamp(0.0, 0.95) as f32 } else { 0.0 }
        } else {
            0.0
        };
        y.par_iter_mut().zip(x.par_iter().zip(&x_prev)).for_each(|(y, (&xc, &xp))| *y = (xc + alpha * (xc - xp)).max(EPS));

        conv.convolve_into(&y, &mut ratio, &mut tmp);
        if let Some((signal, target)) = guards.stop
            && k > 0
        {
            // Fitted down to the noise: what is left to explain is noise.
            let (sum, count) = ratio.par_iter().zip(d).filter(|(_, d)| **d > signal).map(|(&m, &d)| (((d - m) as f64).powi(2), 1u64)).reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            if count > 0 && sum / count as f64 <= target {
                progress(1.0);
                return Some((x, k));
            }
        }
        ratio.par_iter_mut().zip(d).for_each(|(r, &d)| {
            let m = r.max(EPS);
            *r = if damp_threshold > 0.0 { damped_ratio(d, m, damp_threshold) } else { d / m };
        });
        conv.correlate_into(&ratio, &mut correction, &mut tmp);
        if let Some(tv) = &mut tv {
            tv.divergence(&y, w, h);
        }

        // The new estimate goes where x_prev was; x becomes x_prev below.
        x_prev.par_iter_mut().zip(y.par_iter().zip(&correction)).enumerate().for_each(|(i, (new, (&yv, &c)))| {
            let denom = tv.as_ref().map_or(1.0, |tv| (1.0 - p.tv_lambda * tv.div[i]).max(0.2));
            *new = (yv * c / denom).max(EPS);
        });
        if let Some(clamp) = &guards.clamp {
            clamp.apply(&mut x_prev);
        }
        if p.accelerate {
            g2.par_iter_mut().zip(x_prev.par_iter().zip(&y)).for_each(|(g, (n, y))| *g = n - y);
            std::mem::swap(&mut g1, &mut g2);
            corrections += 1;
        }
        std::mem::swap(&mut x, &mut x_prev);
        progress((k + 1) as f32 / iterations as f32);
    }
    Some((x, iterations))
}

/// White (1994) damped RL ratio: residuals within ~`t` of the model (i.e. noise) are pushed
/// towards a ratio of 1 so RL stops fitting them.
fn damped_ratio(d: f32, m: f32, t: f32) -> f32 {
    const N: f32 = 10.0;
    let d64 = d as f64;
    let m64 = m as f64;
    let log_term = if d > 0.0 { d64 * (m64 / d64).ln() } else { 0.0 };
    let u = (-2.0 / (t as f64 * t as f64) * (log_term - m64 + d64)).min(1.0) as f32;
    let factor = u.powf(N - 1.0) * (N - (N - 1.0) * u);
    1.0 + factor * (d / m - 1.0)
}

/// Buffers for `div(∇u / |∇u|)`, reused across iterations.
struct TvScratch {
    nx: Vec<f32>,
    ny: Vec<f32>,
    div: Vec<f32>,
}

impl TvScratch {
    fn new(n: usize) -> Self {
        Self { nx: vec![0.0; n], ny: vec![0.0; n], div: vec![0.0; n] }
    }

    /// Forward differences for the gradient, backward for the divergence.
    fn divergence(&mut self, u: &[f32], w: usize, h: usize) {
        const EPS_TV: f32 = 1e-4;
        self.nx.par_chunks_mut(w).zip(self.ny.par_chunks_mut(w)).enumerate().for_each(|(y, (rx, ry))| {
            for x in 0..w {
                let i = y * w + x;
                let gx = if x + 1 < w { u[i + 1] - u[i] } else { 0.0 };
                let gy = if y + 1 < h { u[i + w] - u[i] } else { 0.0 };
                let norm = (gx * gx + gy * gy + EPS_TV * EPS_TV).sqrt();
                rx[x] = gx / norm;
                ry[x] = gy / norm;
            }
        });
        let (nx, ny) = (&self.nx, &self.ny);
        self.div.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                let i = y * w + x;
                let dx = nx[i] - if x > 0 { nx[i - 1] } else { 0.0 };
                let dy = ny[i] - if y > 0 { ny[i - w] } else { 0.0 };
                *v = dx + dy;
            }
        });
    }
}

/// `U = conj(H)·D / (|H|² + K)`
fn wiener(d: &[f32], conv: &FftConvolver, k: f32) -> Vec<f32> {
    let k = k.max(1e-6);
    let mut spec = conv.forward(d);
    spec.par_iter_mut().zip(conv.otf()).for_each(|(s, h)| *s = *s * h.conj() / (h.norm_sqr() + k));
    conv.inverse(spec).into_iter().map(|v| v.max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blurred pair of point sources should get sharper (higher peak) after deconvolution.
    fn blurred_points(w: usize, h: usize, psf: Psf) -> Vec<f32> {
        let mut img = vec![0.05f32; w * h];
        img[20 * w + 20] = 1.0;
        img[24 * w + 27] = 0.8;
        FftConvolver::new(w, h, &psf.kernel()).convolve(&img)
    }

    fn peak(v: &[f32]) -> f32 {
        v.iter().copied().fold(0.0, f32::max)
    }

    #[test]
    fn every_algorithm_sharpens() {
        let (w, h) = (48, 48);
        let psf = Psf::Gaussian { sigma: 1.5 };
        let blurred = blurred_points(w, h, psf);
        let variants = [
            DeconvParams { psf, ..Default::default() },
            DeconvParams { psf, accelerate: false, ..Default::default() },
            DeconvParams { psf, tv_lambda: 0.002, ..Default::default() },
            DeconvParams { psf, damping: 2.0, ..Default::default() },
            DeconvParams { psf, algorithm: DeconvAlgorithm::Wiener, wiener_k: 0.001, ..Default::default() },
        ];
        for p in variants {
            let out = deconvolve_plane(&blurred, w, h, &p, &|| false, &|_| {}).unwrap();
            assert!(peak(&out) > 1.5 * peak(&blurred), "{:?}: {} vs {}", p.algorithm, peak(&out), peak(&blurred));
        }
    }

    /// The separable path gives the FFT path's result. A single blur matches to float
    /// precision; over many iterations Richardson–Lucy amplifies rounding, so the two paths
    /// must differ less than the FFT path differs from itself when the input moves by 1e-7.
    #[test]
    fn separable_gaussian_matches_fft() {
        let (w, h) = (256, 256);
        let psf = Psf::Gaussian { sigma: 1.7 };
        let mut plane = vec![0.05f32; w * h];
        for (x, y, v) in [(120, 128, 1.0), (131, 125, 0.8), (128, 140, 0.6)] {
            plane[y * w + x] = v;
        }
        let plane = FftConvolver::new(w, h, &psf.kernel()).convolve(&plane);
        let kernel = psf.kernel();
        let padded = Padded::new(&plane, w, h, (2 * kernel.radius).max(16));
        let (fft, separable) = (Blur::Fft(FftConvolver::new(padded.w, padded.h, &kernel)), Blur::separable(&kernel, padded.w, padded.h));
        let pw = padded.w;
        let inner = |v: &[f32]| -> Vec<f32> { (40..padded.h - 40).flat_map(|y| v[y * pw + 40..y * pw + pw - 40].to_vec()).collect() };
        let max_diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(max_diff(&inner(&fft.convolve(&padded.data)), &inner(&separable.convolve(&padded.data))) < 1e-6, "single blur");

        let p = DeconvParams { psf, iterations: 15, tv_lambda: 0.002, ..Default::default() };
        let run = |data: &[f32], blur: &Blur| padded.crop(&richardson_lucy(data, blur, padded.w, padded.h, &p, 0.0, &NO_GUARDS, &|| false, &|_| {}).unwrap().0);
        let nudged: Vec<f32> = padded.data.iter().enumerate().map(|(i, v)| v + if i % 7 == 0 { 1e-7 } else { 0.0 }).collect();
        let reference = run(&padded.data, &fft);
        let noise_floor = max_diff(&reference, &run(&nudged, &fft));
        let err = max_diff(&reference, &run(&padded.data, &separable));
        assert!(err < noise_floor, "paths differ by {err}, numerical noise of R–L is {noise_floor}");
    }

    /// The GPU gives the CPU's result, to within Richardson–Lucy's own numerical noise (see
    /// above). Skipped on machines without a usable GPU.
    #[test]
    fn gpu_matches_cpu() {
        let Some(gpu) = crate::gpu::get() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let (w, h) = (300, 220);
        let psf = Psf::Gaussian { sigma: 1.6 };
        let mut plane = vec![0.05f32; w * h];
        for (i, v) in plane.iter_mut().enumerate() {
            let (x, y) = ((i % w) as f32 - 150.0, (i / w) as f32 - 110.0);
            if x.hypot(y) < 70.0 {
                *v = 0.5 + 0.2 * (y * 0.3).sin();
            }
        }
        let plane = FftConvolver::new(w, h, &psf.kernel()).convolve(&plane);
        let kernel = psf.kernel();
        let blur = Blur::separable(&kernel, w, h);
        let Blur::Separable { terms, .. } = &blur else { unreachable!() };
        let max_diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        for p in [
            DeconvParams { psf, iterations: 12, ..Default::default() },
            DeconvParams { psf, iterations: 12, accelerate: false, tv_lambda: 0.002, ..Default::default() },
            DeconvParams { psf, iterations: 12, damping: 1.0, ..Default::default() },
        ] {
            let damp = p.damping * 0.01;
            let cpu = richardson_lucy(&plane, &blur, w, h, &p, damp, &NO_GUARDS, &|| false, &|_| {}).unwrap().0;
            let nudged: Vec<f32> = plane.iter().enumerate().map(|(i, v)| v + if i % 7 == 0 { 1e-7 } else { 0.0 }).collect();
            let noise_floor = max_diff(&cpu, &richardson_lucy(&nudged, &blur, w, h, &p, damp, &NO_GUARDS, &|| false, &|_| {}).unwrap().0);
            let on_gpu = gpu.richardson_lucy(&plane, w, h, terms, &p, damp, &NO_GUARDS, &|| false, &|_| {}).unwrap().unwrap().0;
            let err = max_diff(&cpu, &on_gpu);
            assert!(err <= noise_floor.max(1e-5), "{p:?}: GPU differs by {err}, numerical noise {noise_floor}");
        }
    }

    const NO_GUARDS: Guards = Guards { stop: None, clamp: None };

    /// A noisy, blurred disk with a little texture, on black.
    fn noisy_disk(w: usize, h: usize, sigma: f32, noise: f32) -> Vec<f32> {
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - w as f32 / 2.0, (i / w) as f32 - h as f32 / 2.0);
                if x.hypot(y) < 0.3 * w as f32 { 0.5 + 0.1 * (x * 0.4).sin() * (y * 0.3).cos() } else { 0.0 }
            })
            .collect();
        let mut seed = 99u32;
        FftConvolver::new(w, h, &Psf::Gaussian { sigma }.kernel())
            .convolve(&plane)
            .into_iter()
            .map(|v| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (v + noise * 3.46 * (seed as f32 / u32::MAX as f32 - 0.5)).max(0.0)
            })
            .collect()
    }

    /// The noisier the data, the sooner fitting it to the noise is reached.
    #[test]
    fn automatic_stop_depends_on_the_noise() {
        let (w, h) = (128, 128);
        let p = DeconvParams { psf: Psf::Gaussian { sigma: 2.0 }, iterations: 100, auto_stop: true, ..Default::default() };
        let used = |noise| deconvolve_plane_with(&noisy_disk(w, h, 2.0, noise), w, h, &p, PlaneOptions::default(), &|| false, &|_| {}).unwrap().1.iterations;
        let (clean, noisy) = (used(0.002), used(0.03));
        assert!(noisy < clean, "noisy {noisy}, clean {clean}");
        assert!((1..100).contains(&noisy), "{noisy}");
    }

    /// Anti-ringing keeps Richardson–Lucy's overshoot at the limb down.
    #[test]
    fn anti_ringing_reduces_limb_overshoot() {
        let (w, h) = (160, 160);
        let plane = noisy_disk(w, h, 2.0, 0.0);
        let wide = Psf::Gaussian { sigma: 2.8 };
        let run = |deringing| deconvolve_plane(&plane, w, h, &DeconvParams { psf: wide, iterations: 40, deringing, ..Default::default() }, &|| false, &|_| {}).unwrap();
        let peak = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
        let (plain, held) = (peak(&run(0.0)), peak(&run(1.0)));
        assert!(held < plain - 0.05, "peak {plain} → {held}");
    }

    #[test]
    fn automatic_wiener_k_follows_the_noise() {
        let (w, h) = (96, 96);
        let k = |noise| wiener_k(&noisy_disk(w, h, 2.0, noise), noise);
        assert!(k(0.03) > 5.0 * k(0.003), "{} vs {}", k(0.03), k(0.003));
    }

    #[test]
    fn channel_scale_widens_the_psf() {
        let p = DeconvParams { psf: Psf::Gaussian { sigma: 1.5 }, ..Default::default() };
        assert!(p.kernel(1.4).radius > p.kernel(1.0).radius);
    }

    /// With both safeguards on, the GPU still gives the CPU's result.
    #[test]
    fn gpu_matches_cpu_with_safeguards() {
        let Some(gpu) = crate::gpu::get() else { return };
        let (w, h) = (200, 180);
        let plane = noisy_disk(w, h, 2.0, 0.01);
        let p = DeconvParams { psf: Psf::Gaussian { sigma: 2.0 }, iterations: 60, auto_stop: true, deringing: 1.0, accelerate: false, ..Default::default() };
        let kernel = p.kernel(1.0);
        let padded = Padded::new(&plane, w, h, (2 * kernel.radius).max(16));
        let observed: Vec<f32> = padded.data.iter().map(|v| v.max(0.0)).collect();
        let noise = estimate_noise(&plane, w, h);
        let guards = || Guards { stop: Some((signal_level(&plane), (noise as f64).powi(2))), clamp: Some(anti_ringing(&plane, w, h, &padded, &observed, &p, &kernel, None)) };
        let blur = Blur::separable(&kernel, padded.w, padded.h);
        let Blur::Separable { terms, .. } = &blur else { unreachable!() };
        let (cpu, cpu_it) = richardson_lucy(&observed, &blur, padded.w, padded.h, &p, 0.0, &guards(), &|| false, &|_| {}).unwrap();
        let (on_gpu, gpu_it) = gpu.richardson_lucy(&observed, padded.w, padded.h, terms, &p, 0.0, &guards(), &|| false, &|_| {}).unwrap().unwrap();
        assert!((cpu_it as i32 - gpu_it as i32).abs() <= 1, "iterations: CPU {cpu_it}, GPU {gpu_it}");
        if cpu_it == gpu_it {
            let err = cpu.iter().zip(&on_gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-3, "differ by {err}");
        }
    }

    /// A Moffat blurred as a sum of Gaussians is close to the exact FFT blur, on the CPU and
    /// the GPU alike.
    #[test]
    fn moffat_as_gaussians() {
        let (w, h) = (160, 140);
        let plane = noisy_disk(w, h, 1.0, 0.0);
        let psf = Psf::Moffat { fwhm: 5.0, beta: 2.0 };
        let exact = FftConvolver::new(w, h, &psf.kernel()).convolve(&plane);
        let mixture = Blur::mixture(&psf.gaussian_mixture().unwrap(), w, h);
        let approx = mixture.convolve(&plane);
        let inner = |v: &[f32]| -> Vec<f32> { (30..h - 30).flat_map(|y| v[y * w + 30..y * w + w - 30].to_vec()).collect() };
        let err = inner(&exact).iter().zip(&inner(&approx)).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 0.01, "blur differs by {err}");

        let Some(gpu) = crate::gpu::get() else { return };
        let p = DeconvParams { psf, iterations: 10, accelerate: false, ..Default::default() };
        let Blur::Separable { terms, .. } = &mixture else { unreachable!() };
        assert!(terms.len() > 1);
        let cpu = richardson_lucy(&plane, &mixture, w, h, &p, 0.0, &NO_GUARDS, &|| false, &|_| {}).unwrap().0;
        let on_gpu = gpu.richardson_lucy(&plane, w, h, terms, &p, 0.0, &NO_GUARDS, &|| false, &|_| {}).unwrap().unwrap().0;
        let err = cpu.iter().zip(&on_gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-3, "GPU differs by {err}");
    }

    #[test]
    fn acceleration_converges_faster() {
        let (w, h) = (48, 48);
        let psf = Psf::Gaussian { sigma: 2.0 };
        let blurred = blurred_points(w, h, psf);
        let run = |accelerate| {
            let p = DeconvParams { psf, iterations: 10, accelerate, ..Default::default() };
            peak(&deconvolve_plane(&blurred, w, h, &p, &|| false, &|_| {}).unwrap())
        };
        assert!(run(true) > run(false));
    }

    #[test]
    fn damped_ratio_is_one_inside_noise() {
        assert!((damped_ratio(0.501, 0.5, 0.05) - 1.0).abs() < 1e-3);
        assert!((damped_ratio(0.9, 0.5, 0.01) - 1.8).abs() < 1e-3);
    }
}
