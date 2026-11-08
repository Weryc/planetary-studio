//! Isotropic undecimated wavelet ("starlet" / à trous) sharpening and denoising.
//!
//! Starck, Murtagh & Bertero, "Starlet Transform in Astronomical Data Processing";
//! Starck, Fadili & Murtagh, IEEE TIP 16(2), 2007.
//!
//! The work is split into an expensive [`decompose`] (depends only on the input and the
//! transform shape) and a cheap [`recompose`] (gains and thresholds), so moving a gain slider
//! only re-runs the latter.

use crate::conv::convolve_separable;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::color::ColorMode;

pub const MAX_LAYERS: usize = 8;
/// Below this σ (px) the grain of free scales is too faint to set apart.
const GRAIN_MIN: f32 = 0.3;

/// Gaussian blur of `sigma` px as an à trous pass: taps spaced so the kernel stays short (the
/// plane it smooths is already blurred more than the spacing, so the holes miss nothing).
/// Narrow blurs use the three-tap kernel with the same variance: sampling a Gaussian narrower
/// than a pixel gets its width wrong.
fn gaussian_pass(sigma: f32) -> (Vec<f32>, usize) {
    let step = ((sigma / 2.0).floor() as usize).max(1);
    let s = sigma / step as f32;
    if s < 0.75 {
        let a = 0.5 * s * s;
        return (vec![a, 1.0 - 2.0 * a, a], step);
    }
    let r = (3.0 * s).ceil() as i32;
    let taps: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * s * s)).exp()).collect();
    let sum: f32 = taps.iter().sum();
    (taps.into_iter().map(|t| t / sum).collect(), step)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum WaveletKernel {
    /// B3-spline `[1 4 6 4 1]/16`, Starck's default: smooth, isotropic.
    B3Spline,
    /// Linear `[1 2 1]/4`: narrower support, crisper fine-scale response.
    Linear,
    /// Sampled Gaussian with the given σ (in units of the tap spacing).
    Gaussian { sigma: f32 },
}

impl WaveletKernel {
    pub fn taps(&self) -> Vec<f32> {
        match *self {
            WaveletKernel::B3Spline => {
                vec![1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
            }
            WaveletKernel::Linear => vec![0.25, 0.5, 0.25],
            WaveletKernel::Gaussian { sigma } => {
                let s = sigma.clamp(0.3, 4.0);
                let r = (3.0 * s).ceil() as i32;
                let taps: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * s * s)).exp()).collect();
                let sum: f32 = taps.iter().sum();
                taps.into_iter().map(|t| t / sum).collect()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleMode {
    /// Tap spacing doubles each layer (1, 2, 4, 8 …): the classic starlet.
    Dyadic,
    /// Tap spacing grows by one each layer (1, 2, 3, 4 …): finer control over small scales.
    Linear,
    /// Gaussian scales of any size, doubling each layer from `first_scale`: the first layer
    /// can sit right where the image's finest real detail is, instead of on 1 px (often only
    /// noise) or 2 px (already too coarse). What is finer than half the first scale is kept
    /// apart as grain: never sharpened, only cleaned (`grain_denoise`).
    Free,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdMode {
    Soft,
    Hard,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WaveletLayer {
    /// Extra amplification of this layer: 0 = unchanged, 1 = doubled, −1 = removed.
    pub gain: f32,
    /// Denoise threshold in units of this layer's noise σ (0 = off).
    pub denoise: f32,
}

impl Default for WaveletLayer {
    fn default() -> Self {
        Self { gain: 0.0, denoise: 0.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WaveletParams {
    pub kernel: WaveletKernel,
    pub scale_mode: ScaleMode,
    /// Tap spacing of the first layer.
    pub base_step: u32,
    /// [`ScaleMode::Free`]: size of the first layer (px), on the same footing as the classic
    /// layers' 1, 2, 4 … px.
    #[serde(default = "default_first_scale")]
    pub first_scale: f32,
    /// [`ScaleMode::Free`]: denoise threshold of the grain, in units of its noise σ (0 = off).
    #[serde(default)]
    pub grain_denoise: f32,
    /// [`ScaleMode::Free`]: size ratio between neighbouring layers (2 = an octave each; √2
    /// gives finer control over the frequencies for twice the layers).
    #[serde(default = "two")]
    pub ratio: f32,
    pub layers: Vec<WaveletLayer>,
    pub threshold: ThresholdMode,
    pub color: ColorMode,
    /// Multiplies every layer's gain, for when the sliders' range isn't enough.
    #[serde(default = "one")]
    pub amplify: f32,
    /// Noise-adaptive sharpening: a coefficient gets the full gain only when it stands out
    /// from the layer's noise by more than this many σ, and less and less below (0 = off).
    #[serde(default)]
    pub noise_adaptive: f32,
    /// Less sharpening where large-scale edges are strong (limb, rings), so halos don't form
    /// (0 = off, 1 = no extra gain on the strongest edges).
    #[serde(default)]
    pub edge_protection: f32,
}

fn one() -> f32 {
    1.0
}

fn two() -> f32 {
    2.0
}

fn default_first_scale() -> f32 {
    1.5
}

impl Default for WaveletParams {
    fn default() -> Self {
        Self {
            kernel: WaveletKernel::B3Spline,
            scale_mode: ScaleMode::Free,
            base_step: 1,
            first_scale: default_first_scale(),
            grain_denoise: 0.0,
            ratio: 2.0,
            layers: vec![WaveletLayer::default(); 6],
            threshold: ThresholdMode::Soft,
            color: ColorMode::Luminance,
            amplify: 1.0,
            noise_adaptive: 0.0,
            edge_protection: 0.0,
        }
    }
}

/// Everything the expensive decomposition depends on (besides the input image).
#[derive(Clone, Debug, PartialEq)]
pub struct DecompositionKey {
    kernel: WaveletKernel,
    scale_mode: ScaleMode,
    base_step: u32,
    first_scale: f32,
    ratio: f32,
    levels: usize,
    color: ColorMode,
}

impl WaveletParams {
    pub fn decomposition_key(&self) -> DecompositionKey {
        DecompositionKey { kernel: self.kernel, scale_mode: self.scale_mode, base_step: self.base_step, first_scale: self.first_scale, ratio: self.ratio, levels: self.layers.len(), color: self.color }
    }

    /// Tap spacing used to build layer `j` (0-based).
    pub fn step(&self, j: usize) -> usize {
        let base = self.base_step.max(1) as usize;
        match self.scale_mode {
            ScaleMode::Dyadic | ScaleMode::Free => base << j,
            ScaleMode::Linear => base * (j + 1),
        }
    }

    /// The free scales' ratio, kept to something sensible.
    fn ratio(&self) -> f32 {
        self.ratio.clamp(1.2, 3.0)
    }

    /// Size of layer `j` in pixels, for display.
    pub fn scale(&self, j: usize) -> f32 {
        match self.scale_mode {
            ScaleMode::Free => self.first_scale * self.ratio().powi(j as i32),
            _ => self.step(j) as f32,
        }
    }

    /// The smoothing passes building the layers: (taps, tap spacing) each. With `true`, the
    /// first pass sets the grain apart and isn't a layer.
    ///
    /// Free scales: smoothing c_k = image ⊛ G(σ_k) with σ_0 = first/2 and σ doubling, so layer
    /// j = c_j − c_{j+1} is a band around the scale `first·2^j` (the classic B3 layers 1, 2, 4 px
    /// fall between σ 0–1, 1–2.2, 2.2–4.6 px: about the same footing). Each pass adds the blur
    /// that is missing, √(σ_{k+1}² − σ_k²) = √3·σ_k.
    fn passes(&self) -> (bool, Vec<(Vec<f32>, usize)>) {
        match self.scale_mode {
            ScaleMode::Free => {
                let r = self.ratio();
                let half = self.first_scale.max(0.4) / r;
                let grain = half >= GRAIN_MIN;
                let mut passes = Vec::with_capacity(self.layers.len() + 1);
                if grain {
                    passes.push(gaussian_pass(half));
                }
                // From σ to r·σ: add √(r² − 1)·σ of blur.
                passes.extend((0..self.layers.len()).map(|j| gaussian_pass((r * r - 1.0).sqrt() * half * r.powi(j as i32))));
                (grain, passes)
            }
            _ => (false, (0..self.layers.len()).map(|j| (self.kernel.taps(), self.step(j))).collect()),
        }
    }

    pub fn is_neutral(&self) -> bool {
        self.layers.iter().all(|l| l.gain == 0.0 && l.denoise == 0.0) && (self.scale_mode != ScaleMode::Free || self.grain_denoise == 0.0)
    }
}

pub struct Decomposition {
    /// Detail layers w_1 … w_J, finest first.
    pub layers: Vec<Vec<f32>>,
    /// Free scales: what is finer than the first layer, and its noise σ.
    pub grain: Option<(Vec<f32>, f32)>,
    /// Smooth residual c_J.
    pub residual: Vec<f32>,
    /// Estimated noise σ of each detail layer: stack noise is correlated, so it falls more
    /// slowly with the scale than white noise would ([`correlated_excess`]).
    pub noise: Vec<f32>,
    /// The same, scaled from the finest plane as if the noise were white: modules whose
    /// thresholds were tuned on it (noise reduction, smart sharpening) keep using it.
    pub white_noise: Vec<f32>,
    pub width: usize,
    /// Where large-scale edges are (0‥1), computed the first time edge protection asks.
    edges: OnceLock<Vec<f32>>,
}

impl Decomposition {
    /// Strength of the large-scale edges at each pixel (0‥1): the gradient of the image without
    /// its two finest layers, relative to its strongest values, widened a little; only where
    /// the edge meets the dark sky (a limb, the edge of the rings). Belts and festoons have
    /// strong edges too, but that is the detail to sharpen, and halos form against the sky.
    pub(crate) fn edges(&self) -> &[f32] {
        self.edges.get_or_init(|| {
            let (w, n) = (self.width, self.residual.len());
            let h = n / w.max(1);
            if w < 3 || h < 3 {
                return vec![0.0; n];
            }
            let coarse: Vec<f32> = (0..n).into_par_iter().map(|i| self.residual[i] + self.layers.iter().skip(2).map(|l| l[i]).sum::<f32>()).collect();
            let mut grad = vec![0.0f32; n];
            grad.par_chunks_mut(w).enumerate().skip(1).take(h - 2).for_each(|(y, row)| {
                for (x, g) in row.iter_mut().enumerate().take(w - 1).skip(1) {
                    let i = y * w + x;
                    *g = (coarse[i + 1] - coarse[i - 1]).hypot(coarse[i + w] - coarse[i - w]);
                }
            });
            let stride = n.div_ceil(1 << 18).max(1);
            let mut sample: Vec<f32> = grad.iter().step_by(stride).copied().collect();
            let k = (sample.len() * 995 / 1000).min(sample.len() - 1);
            let top = sample.select_nth_unstable_by(k, f32::total_cmp).1.max(1e-9);
            let strength: Vec<f32> = grad
                .par_iter()
                .map(|g| {
                    let t = ((g / top - 0.25) / 0.5).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                })
                .collect();
            // Darkness nearby: the local minimum against the bright level.
            let (low, _) = crate::dering::min_max(&coarse, w, h, 6);
            let mut sample: Vec<f32> = coarse.iter().step_by(stride).copied().collect();
            let k = (sample.len() * 99 / 100).min(sample.len() - 1);
            let bright = sample.select_nth_unstable_by(k, f32::total_cmp).1.max(1e-9);
            let near_sky: Vec<f32> = strength.par_iter().zip(&low).map(|(s, &l)| s * (1.0 - ((l / bright - 0.08) / 0.12).clamp(0.0, 1.0))).collect();
            let widened = crate::mask::gaussian_approx(&near_sky, w, h, 2.0);
            widened.into_iter().map(|v| (2.0 * v).min(1.0)).collect()
        })
    }
}

/// Frequency response of each detail plane at (`fx`, `fy`) cycles/px: the grain (free scales)
/// and the layers, finest first. Each plane is c_k − c_{k+1}, and the smoothing passes are
/// separable and symmetric, so its response is a difference of products of cosine sums.
pub(crate) fn responses(params: &WaveletParams, fx: f32, fy: f32) -> (Option<f32>, Vec<f32>) {
    let (has_grain, passes) = params.passes();
    let tau = std::f32::consts::TAU;
    let one_d = |taps: &[f32], step: usize, f: f32| -> f32 {
        let r = (taps.len() / 2) as isize;
        taps.iter().enumerate().map(|(i, t)| t * (tau * f * step as f32 * (i as isize - r) as f32).cos()).sum()
    };
    let mut before = 1.0f32;
    let mut planes = Vec::with_capacity(passes.len());
    for (taps, step) in &passes {
        let after = before * one_d(taps, *step, fx) * one_d(taps, *step, fy);
        planes.push(before - after);
        before = after;
    }
    let grain = has_grain.then(|| planes.remove(0));
    (grain, planes)
}

/// Gains for the layers of `to` sharpening detail coarser than `up_to` cycles/px as much as
/// the gains of `from` do: least squares on the overall response 1 + Σ g·H over the frequency
/// plane below `up_to`. Finer than that is only noise: there `to` (with free scales, whose
/// grain is never sharpened) adds less than `from`. Gains stay within 0‥8.
///
/// With noise-adaptive sharpening a layer only gets part of its gain, the less the more of it
/// is noise; `snr(f)` (signal over noise power at f cycles/px) tells how much of each layer is
/// noise, and the gains are matched on what they really do.
pub fn equivalent_gains(from: &WaveletParams, to: &WaveletParams, up_to: f32, snr: &dyn Fn(f32) -> f32) -> Vec<f32> {
    const GRID: usize = 64;
    let n = to.layers.len();
    let limit = up_to.clamp(0.02, 0.5);
    let effective_from = adaptive_factors(from, snr);
    let effective_to = adaptive_factors(to, snr);
    let (mut a, mut b) = (vec![vec![0.0f64; n]; n], vec![0.0f64; n]);
    for iy in 0..GRID {
        for ix in 0..GRID {
            let (fx, fy) = ((ix as f32 + 0.5) * limit / GRID as f32, (iy as f32 + 0.5) * limit / GRID as f32);
            if fx.hypot(fy) > limit {
                continue;
            }
            let (_, hf) = responses(from, fx, fy);
            let want: f64 = from.layers.iter().zip(&hf).zip(&effective_from).map(|((l, h), e)| (from.amplify * l.gain * h * e) as f64).sum();
            let (_, h) = responses(to, fx, fy);
            let h: Vec<f32> = h.iter().zip(&effective_to).map(|(h, e)| h * e).collect();
            for j in 0..n {
                b[j] += want * h[j] as f64;
                for k in 0..n {
                    a[j][k] += (h[j] * h[k]) as f64;
                }
            }
        }
    }
    // A little smoothness between neighbouring gains: layers overlap, so least squares alone
    // happily trades one against the next and gives zigzag sliders for the same response.
    let trace = (0..n).map(|j| a[j][j]).sum::<f64>() / n.max(1) as f64;
    let smooth = SMOOTHNESS * trace;
    for j in 0..n {
        a[j][j] += 1e-4 * trace;
        if j + 1 < n {
            a[j][j] += smooth;
            a[j + 1][j + 1] += smooth;
            a[j][j + 1] -= smooth;
            a[j + 1][j] -= smooth;
        }
    }
    bounded_least_squares(a, b, 0.0, 8.0).into_iter().map(|g| ((g as f32) * 100.0).round() / 100.0).collect()
}

/// Average share of its gain each layer of `params` gets from noise-adaptive sharpening, for an
/// image with signal-to-noise power `snr(f)`. A layer's coefficients are about Gaussian, with
/// variance S + N (its signal and noise power); one of size v gets r²/(1 + r²) of the gain,
/// r = v / (a·σ), σ² = N.
fn adaptive_factors(params: &WaveletParams, snr: &dyn Fn(f32) -> f32) -> Vec<f32> {
    let n = params.layers.len();
    if params.noise_adaptive <= 0.0 {
        return vec![1.0; n];
    }
    const GRID: usize = 48;
    let (mut signal, mut noise) = (vec![0.0f64; n], vec![0.0f64; n]);
    for iy in 0..GRID {
        for ix in 0..GRID {
            let (fx, fy) = ((ix as f32 + 0.5) / (2 * GRID) as f32, (iy as f32 + 0.5) / (2 * GRID) as f32);
            let ratio = snr(fx.hypot(fy)).min(1e6) as f64;
            let (_, h) = responses(params, fx, fy);
            for j in 0..n {
                let p = (h[j] * h[j]) as f64;
                signal[j] += p * ratio;
                noise[j] += p;
            }
        }
    }
    (0..n)
        .map(|j| {
            // t² = (S + N) / (a² N): E[x²t² / (1 + x²t²)] for x ~ N(0, 1), by quadrature.
            let t2 = (signal[j] + noise[j]) / (noise[j].max(1e-30) * (params.noise_adaptive as f64).powi(2));
            let (mut sum, mut weight) = (0.0, 0.0);
            for k in -120..=120 {
                let x = k as f64 * 0.05;
                let g = (-0.5 * x * x).exp();
                sum += g * x * x * t2 / (1.0 + x * x * t2);
                weight += g;
            }
            (sum / weight) as f32
        })
        .collect()
}

const SMOOTHNESS: f64 = 0.02;

/// Minimise ½xᵀAx − bᵀx with every x within `lo..=hi`: out-of-range values are pinned to their
/// bound one at a time (the worst first) and the rest solved again.
fn bounded_least_squares(a: Vec<Vec<f64>>, b: Vec<f64>, lo: f64, hi: f64) -> Vec<f64> {
    let n = b.len();
    let mut pinned: Vec<Option<f64>> = vec![None; n];
    loop {
        let free: Vec<usize> = (0..n).filter(|&j| pinned[j].is_none()).collect();
        let sub_a: Vec<Vec<f64>> = free.iter().map(|&j| free.iter().map(|&k| a[j][k]).collect()).collect();
        let sub_b: Vec<f64> = free.iter().map(|&j| b[j] - (0..n).filter_map(|k| pinned[k].map(|g| a[j][k] * g)).sum::<f64>()).collect();
        let x = solve(sub_a, sub_b);
        let mut out: Vec<f64> = pinned.iter().map(|p| p.unwrap_or(0.0)).collect();
        free.iter().zip(&x).for_each(|(&j, &g)| out[j] = g);
        let worst = free.iter().map(|&j| (j, (lo - out[j]).max(out[j] - hi))).filter(|&(_, over)| over > 1e-9).max_by(|p, q| p.1.total_cmp(&q.1));
        match worst {
            Some((j, _)) => pinned[j] = Some(out[j].clamp(lo, hi)),
            None => return out,
        }
    }
}

/// `a·x = b` for a small system (Gaussian elimination with partial pivoting).
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap_or(c);
        a.swap(c, p);
        b.swap(c, p);
        if a[c][c].abs() < 1e-300 {
            continue;
        }
        let pivot = a[c].clone();
        for r in c + 1..n {
            let f = a[r][c] / pivot[c];
            a[r][c..].iter_mut().zip(&pivot[c..]).for_each(|(v, p)| *v -= f * p);
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = if a[r][r].abs() < 1e-300 { 0.0 } else { (b[r] - s) / a[r][r] };
    }
    x
}

/// `c_{j+1} = c_j ⊛ h_j`, `w_{j+1} = c_j − c_{j+1}`. Returns `None` if cancelled.
pub fn decompose(plane: &[f32], w: usize, h: usize, params: &WaveletParams, cancel: &dyn Fn() -> bool) -> Option<Decomposition> {
    if cancel() {
        return None;
    }
    let (has_grain, passes) = params.passes();
    let borrowed: Vec<(&[f32], usize)> = passes.iter().map(|(t, s)| (t.as_slice(), *s)).collect();
    let gpu = crate::gpu::for_plane(w, h, crate::gpu::Work::Light).and_then(|g| g.cascade(plane, w, h, &borrowed, true).ok());
    let (mut layers, residual) = match gpu {
        Some(result) => result,
        None => {
            let n = w * h;
            let mut current = plane.to_vec();
            let mut next = vec![0.0f32; n];
            let mut tmp = vec![0.0f32; n];
            let mut layers = Vec::with_capacity(passes.len());
            for (taps, step) in &passes {
                if cancel() {
                    return None;
                }
                convolve_separable(&current, &mut next, &mut tmp, w, h, taps, *step);
                let mut detail = vec![0.0f32; n];
                detail.par_iter_mut().zip(current.par_iter().zip(&next)).for_each(|(d, (c, s))| *d = c - s);
                layers.push(detail);
                std::mem::swap(&mut current, &mut next);
            }
            (layers, current)
        }
    };

    // The noise is measured on the finest plane (the grain when there is one) and scaled to
    // the others with white noise's ratios.
    let factors = noise_factors(params);
    let sigma = layers.first().map(|w0| signal_mad_sigma(w0, plane) / factors[0]).unwrap_or(0.0);
    let mut white: Vec<f32> = factors.iter().map(|f| sigma * f).collect();
    let excess = correlated_excess(&layers, &residual, &white, has_grain, params);
    let mut noise: Vec<f32> = white.iter().zip(&excess).map(|(s, e)| s * e).collect();
    // Not measured on the first layer itself: it holds the finest real detail, which its spread
    // would count as noise (20× too much on a real Jupiter), and the noise-adaptive gain would
    // then leave that layer almost unsharpened.
    let grain = has_grain.then(|| {
        white.remove(0);
        (layers.remove(0), noise.remove(0))
    });
    Some(Decomposition { layers, grain, residual, noise, white_noise: white, width: w, edges: OnceLock::new() })
}

/// Correlated-noise growth when the sky can't be measured: the first plane above the reference
/// one, and each octave of scale after it. Measured on the test stacks whose sky wasn't
/// clipped (×1.3–2.1 and ×1.25–1.5 per octave): the low end, so a clean stack isn't held back.
const DEFAULT_FIRST_EXCESS: [f32; 2] = [1.25, 1.5];
const DEFAULT_OCTAVE_EXCESS: f32 = 1.25;

/// How much more noise each plane holds than white noise would give it, measured from the
/// finest plane (the reference, 1 by definition). Stacks are made of aligned, resampled and
/// blended frames, so their noise is correlated: on the test images it falls with the scale
/// as about scale^−0.65 instead of scale^−1, and a medium layer holds 2–4× the noise of the
/// white-noise estimate.
///
/// Measured on the dark sky far from the subject (pure noise) when the stacker left it noisy,
/// on the first three layers, and extended to the coarser ones as a constant growth per octave
/// (their sky picks up the subject's glow). With a clipped sky, typical values.
fn correlated_excess(planes: &[Vec<f32>], residual: &[f32], white: &[f32], has_grain: bool, params: &WaveletParams) -> Vec<f32> {
    let count = planes.len();
    if count < 2 {
        return vec![1.0; count];
    }
    // Scale of each plane: the grain sits an octave below the first layer.
    let scale = |m: usize| if has_grain { if m == 0 { params.scale(0) / params.ratio() } else { params.scale(m - 1) } } else { params.scale(m) };
    let octaves = |m: usize| (scale(m) / scale(1)).log2();
    let measured = sky_excess(planes, residual, white, has_grain);
    let (first, growth) = match measured {
        Some(e) => {
            let first = e[0].clamp(1.0, 3.0);
            // Least squares on log e(x) = log first + x log k, through the first layer.
            let (num, den) = e.iter().enumerate().skip(1).fold((0.0, 0.0), |(n, d), (i, &v)| {
                let x = octaves(i + 1);
                (n + (v.max(1e-3) / first).ln() * x, d + x * x)
            });
            let k = if den > 0.0 { (num / den).exp() } else { DEFAULT_OCTAVE_EXCESS };
            (first, k.clamp(1.0, 1.6))
        }
        None => (DEFAULT_FIRST_EXCESS[has_grain as usize], DEFAULT_OCTAVE_EXCESS),
    };
    (0..count).map(|m| if m == 0 { 1.0 } else { (first * growth.powf(octaves(m))).min(8.0) }).collect()
}

/// [`correlated_excess`] measured on the sky, for the planes after the reference (at most 3),
/// or `None` when there isn't enough unclipped sky.
fn sky_excess(planes: &[Vec<f32>], residual: &[f32], white: &[f32], has_grain: bool) -> Option<Vec<f32>> {
    let n = residual.len();
    let stride = n.div_ceil(1 << 20).max(1);
    // "Far from the subject": dark even after blurring to about four layers' scale.
    let k = planes.len().min(4 + has_grain as usize);
    let far = |i: usize| residual[i] + planes[k..].iter().map(|p| p[i]).sum::<f32>();
    let mut sample: Vec<f32> = (0..n).step_by(stride).map(far).collect();
    if sample.len() < 100 {
        return None;
    }
    let len = sample.len();
    let sky = *sample.select_nth_unstable_by(len / 5, f32::total_cmp).1;
    let bright = *sample.select_nth_unstable_by(len * 99 / 100, f32::total_cmp).1;
    let limit = sky + 0.02 * (bright - sky);
    let picked: Vec<usize> = (0..n).step_by(stride).filter(|&i| far(i) < limit).collect();
    if picked.len() < 2000 {
        return None;
    }
    let sigma = |m: usize| mad_sigma(&picked.iter().map(|&i| planes[m][i]).collect::<Vec<_>>());
    let reference = sigma(0);
    // A clipped sky has (almost) no noise left, or only quantisation steps.
    if reference < 3e-5 || reference < 0.02 * white[0] {
        return None;
    }
    let last = (planes.len() - 1).min(3);
    Some((1..=last).map(|m| (sigma(m) / reference) / (white[m] / white[0])).collect())
}

/// `out = c_J + Σ_j (1 + a·g_j·m) · T_j(w_j)`, with T_j the soft/hard threshold at
/// `denoise_j · σ_j`, a the multiplier and m ≤ 1 the noise-adaptive and edge-protection
/// factors (1 when those are off; they only temper extra sharpening, never removal).
pub fn recompose(d: &Decomposition, params: &WaveletParams) -> Vec<f32> {
    const CHUNK: usize = 16 * 1024;
    let mut out = d.residual.clone();
    // The grain is only ever cleaned.
    if let Some((grain, sigma)) = &d.grain {
        let t = params.grain_denoise.max(0.0) * sigma;
        let soft = params.threshold == ThresholdMode::Soft;
        out.par_iter_mut().zip(grain).for_each(|(o, &v)| {
            *o += if t <= 0.0 {
                v
            } else if soft {
                v.signum() * (v.abs() - t).max(0.0)
            } else if v.abs() > t {
                v
            } else {
                0.0
            };
        });
    }
    // (extra gain, threshold, noise scale for the adaptive factor) per layer.
    let layer_ops: Vec<(f32, f32, f32)> = params.layers.iter().zip(&d.noise).map(|(l, sigma)| (params.amplify * l.gain, l.denoise.max(0.0) * sigma, params.noise_adaptive * sigma)).collect();
    let soft = params.threshold == ThresholdMode::Soft;
    let protect = params.edge_protection.clamp(0.0, 1.0);
    let edges = (protect > 0.0 && layer_ops.iter().any(|o| o.0 > 0.0)).then(|| d.edges());
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, chunk)| {
        let start = ci * CHUNK;
        let edge = edges.map(|e| &e[start..start + chunk.len()]);
        for (layer, &(extra, t, noise)) in d.layers.iter().zip(&layer_ops) {
            let src = &layer[start..start + chunk.len()];
            for (k, (o, &v)) in chunk.iter_mut().zip(src).enumerate() {
                let kept = if t <= 0.0 {
                    v
                } else if soft {
                    v.signum() * (v.abs() - t).max(0.0)
                } else if v.abs() > t {
                    v
                } else {
                    0.0
                };
                let mut boost = extra;
                if boost > 0.0 {
                    if noise > 0.0 {
                        // r²/(1 + r²): ≈ 0 inside the noise, → 1 well above it.
                        let r2 = (v / noise) * (v / noise);
                        boost *= r2 / (1.0 + r2);
                    }
                    if let Some(e) = edge {
                        boost *= 1.0 - protect * e[k];
                    }
                }
                *o += (1.0 + boost) * kept;
            }
        }
    });
    out
}

/// Robust noise σ from the median absolute deviation (Gaussian noise: σ = MAD / 0.6745).
fn mad_sigma(values: &[f32]) -> f32 {
    // A strided sample of ≤ 1 M values is plenty for a median and keeps 50 MP images fast.
    let stride = values.len().div_ceil(1 << 20).max(1);
    let mut abs: Vec<f32> = values.iter().step_by(stride).map(|v| v.abs()).collect();
    if abs.is_empty() {
        return 0.0;
    }
    let mid = abs.len() / 2;
    let (_, median, _) = abs.select_nth_unstable_by(mid, f32::total_cmp);
    *median / 0.6745
}

/// [`mad_sigma`] of the finest layer `layer`, over the pixels of `plane` that carry signal.
/// Stacking software often clips the sky to (almost) pure black, where there is no noise left:
/// in a planetary frame that is most of the pixels, and the estimate would describe the sky
/// instead of the planet, where the noise actually is.
///
/// Pixels are picked on the first smooth level (`plane − layer`): picking them on the noisy
/// values would favour the ones that noise pushed up, and bias the estimate.
fn signal_mad_sigma(layer: &[f32], plane: &[f32]) -> f32 {
    let stride = plane.len().div_ceil(1 << 20).max(1);
    let smooth = |i: usize| plane[i] - layer[i];
    let mut sample: Vec<f32> = (0..plane.len()).step_by(stride).map(smooth).collect();
    if sample.len() < 100 {
        return mad_sigma(layer);
    }
    let n = sample.len();
    let sky = *sample.select_nth_unstable_by(n / 5, f32::total_cmp).1;
    let bright = *sample.select_nth_unstable_by(n * 99 / 100, f32::total_cmp).1;
    let threshold = sky + 0.1 * (bright - sky);
    let mut abs: Vec<f32> = (0..plane.len()).step_by(stride).filter(|&i| smooth(i) > threshold).map(|i| layer[i].abs()).collect();
    // Too little signal to measure on (or nothing but sky): use everything.
    if abs.len() < 1000.max(n / 50) {
        return mad_sigma(layer);
    }
    let mid = abs.len() / 2;
    *abs.select_nth_unstable_by(mid, f32::total_cmp).1 / 0.6745
}

/// Noise σ of a plane, estimated from the finest B3 starlet layer.
pub fn estimate_noise(plane: &[f32], w: usize, h: usize) -> f32 {
    let params = WaveletParams { scale_mode: ScaleMode::Dyadic, layers: vec![WaveletLayer::default(); 1], ..Default::default() };
    let d = decompose(plane, w, h, &params, &|| false).expect("never cancelled");
    d.noise[0] / noise_factors(&params)[0]
}

/// σ of each detail plane (the grain first, if any) when the input is unit-variance white
/// noise. Published values exist for the dyadic B3 starlet (0.8908, 0.2007, 0.0856 …);
/// computing them numerically covers every kernel and scale mode. Cached per transform shape.
fn noise_factors(params: &WaveletParams) -> Vec<f32> {
    static CACHE: OnceLock<Mutex<HashMap<String, Vec<f32>>>> = OnceLock::new();
    let key = format!("{:?}", params.decomposition_key());
    let cache = CACHE.get_or_init(Default::default);
    if let Some(f) = cache.lock().unwrap().get(&key) {
        return f.clone();
    }

    const SIZE: usize = 256;
    let noise = white_noise(SIZE * SIZE);
    let mut current = noise;
    let mut next = vec![0.0f32; SIZE * SIZE];
    let mut tmp = vec![0.0f32; SIZE * SIZE];
    let (_, passes) = params.passes();
    let mut factors = Vec::with_capacity(passes.len());
    for (taps, step) in &passes {
        convolve_separable(&current, &mut next, &mut tmp, SIZE, SIZE, taps, *step);
        let var = current.iter().zip(&next).map(|(c, s)| ((c - s) as f64).powi(2)).sum::<f64>() / (SIZE * SIZE) as f64;
        factors.push(var.sqrt().max(1e-6) as f32);
        std::mem::swap(&mut current, &mut next);
    }
    cache.lock().unwrap().insert(key, factors.clone());
    factors
}

/// Deterministic unit-variance Gaussian noise (xorshift + Box–Muller).
pub(crate) fn white_noise(n: usize) -> Vec<f32> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (uniform(), uniform());
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_plane(w: usize, h: usize) -> Vec<f32> {
        (0..w * h).map(|i| (((i % w) as f32 * 0.3).sin() * ((i / w) as f32 * 0.2).cos() + 1.0) * 0.4).collect()
    }

    #[test]
    fn neutral_parameters_reconstruct_exactly() {
        let (w, h) = (64, 48);
        let plane = test_plane(w, h);
        for scale_mode in [ScaleMode::Dyadic, ScaleMode::Linear, ScaleMode::Free] {
            let params = WaveletParams { scale_mode, ..Default::default() };
            let d = decompose(&plane, w, h, &params, &|| false).unwrap();
            let out = recompose(&d, &params);
            let err = plane.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(err < 1e-5, "{scale_mode:?}: max error {err}");
        }
    }

    #[test]
    fn b3_noise_factors_match_published_values() {
        let f = noise_factors(&WaveletParams { scale_mode: ScaleMode::Dyadic, ..Default::default() });
        for (got, want) in f.iter().zip([0.8908, 0.2007, 0.0856, 0.0413]) {
            assert!((got - want).abs() / want < 0.06, "{got} vs {want}");
        }
    }

    #[test]
    fn noise_estimate_recovers_sigma() {
        let (w, h) = (256, 256);
        let sigma = 0.02;
        let plane: Vec<f32> = white_noise(w * h).iter().map(|n| 0.5 + sigma * n).collect();
        let est = estimate_noise(&plane, w, h);
        assert!((est - sigma).abs() / sigma < 0.1, "estimated {est}");
    }

    /// A planet on a sky clipped to black: the noise is the planet's, not the sky's (≈ 0).
    #[test]
    fn noise_is_measured_on_the_subject() {
        let (w, h) = (256, 256);
        let sigma = 0.01;
        let noise = white_noise(w * h);
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - 128.0, (i / w) as f32 - 128.0);
                if x.hypot(y) < 50.0 { 0.5 + sigma * noise[i] } else { 0.0 }
            })
            .collect();
        let est = estimate_noise(&plane, w, h);
        assert!((est - sigma).abs() / sigma < 0.25, "estimated {est}");
    }

    #[test]
    fn positive_gain_increases_fine_detail() {
        let (w, h) = (64, 64);
        let plane = test_plane(w, h);
        let mut params = WaveletParams::default();
        params.layers[0].gain = 2.0;
        let d = decompose(&plane, w, h, &params, &|| false).unwrap();
        let out = recompose(&d, &params);
        let energy = |v: &[f32]| v.windows(2).map(|p| (p[1] - p[0]).powi(2)).sum::<f32>();
        assert!(energy(&out) > energy(&plane));
    }

    #[test]
    fn noise_adaptive_spares_the_grain() {
        // Pure noise on a flat field: plain sharpening boosts it, the adaptive one much less.
        let (w, h) = (128, 128);
        let plane: Vec<f32> = white_noise(w * h).iter().map(|n| 0.5 + 0.01 * n).collect();
        let mut plain = WaveletParams { scale_mode: ScaleMode::Dyadic, ..Default::default() };
        plain.layers[0].gain = 3.0;
        let adaptive = WaveletParams { noise_adaptive: 3.0, ..plain.clone() };
        let d = decompose(&plane, w, h, &plain, &|| false).unwrap();
        let rms = |v: &[f32]| (v.iter().map(|x| (x - 0.5).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        let (a, b) = (rms(&recompose(&d, &plain)), rms(&recompose(&d, &adaptive)));
        assert!(b < 0.6 * a, "plain {a}, adaptive {b}");
    }

    #[test]
    fn edge_protection_tames_the_limb() {
        // A sharp disk edge with fine texture inside: protection lowers the overshoot at the
        // edge much more than the texture's gain.
        let (w, h) = (160, 160);
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - 80.0, (i / w) as f32 - 80.0);
                if x.hypot(y) < 50.0 { 0.6 + 0.03 * (x * 1.3).sin() * (y * 1.1).cos() } else { 0.0 }
            })
            .collect();
        let mut plain = WaveletParams::default();
        plain.layers[0].gain = 3.0;
        plain.layers[1].gain = 2.0;
        let protected = WaveletParams { edge_protection: 1.0, ..plain.clone() };
        let d = decompose(&plane, w, h, &plain, &|| false).unwrap();
        let (a, b) = (recompose(&d, &plain), recompose(&d, &protected));
        let overshoot = |v: &[f32]| (0..w).map(|x| (v[80 * w + x] - plane[80 * w + x]).abs()).fold(0.0f32, f32::max);
        let texture = |v: &[f32]| (60..100).map(|x| (v[80 * w + x] - plane[80 * w + x]).abs()).sum::<f32>();
        assert!(overshoot(&b) < 0.5 * overshoot(&a), "edge {} → {}", overshoot(&a), overshoot(&b));
        assert!(texture(&b) > 0.7 * texture(&a), "texture {} → {}", texture(&a), texture(&b));
    }

    /// Detail just coarser than the blur, under fine noise: the free first layer holds far more
    /// of the detail for the noise it carries than the classic 1 px layer.
    #[test]
    fn free_first_layer_sits_on_the_real_detail() {
        let side = 256;
        let blurred = crate::spectrum::tests::degraded(side, 1.8, 0.0);
        let noise: Vec<f32> = white_noise(side * side).iter().map(|n| 0.004 * n).collect();
        let energy = |plane: &[f32], p: &WaveletParams| decompose(plane, side, side, p, &|| false).unwrap().layers[0].iter().map(|v| (v * v) as f64).sum::<f64>();
        let snr = |p: &WaveletParams| energy(&blurred, p) / energy(&noise, p);
        let classic = WaveletParams { scale_mode: ScaleMode::Dyadic, ..Default::default() };
        let free = WaveletParams { scale_mode: ScaleMode::Free, first_scale: 1.6, ..Default::default() };
        assert!(snr(&free) > 3.0 * snr(&classic), "free {} vs classic {}", snr(&free), snr(&classic));
    }

    #[test]
    fn grain_is_only_cleaned() {
        let (w, h) = (128, 128);
        let plane: Vec<f32> = white_noise(w * h).iter().map(|n| 0.5 + 0.01 * n).collect();
        let mut p = WaveletParams { scale_mode: ScaleMode::Free, first_scale: 2.0, ..Default::default() };
        p.layers.iter_mut().for_each(|l| l.gain = 4.0);
        let d = decompose(&plane, w, h, &p, &|| false).unwrap();
        let (grain, _) = d.grain.as_ref().expect("grain set apart");
        // The grain goes through unchanged, whatever the gains…
        let out = recompose(&d, &p);
        let rest: Vec<f32> = (0..w * h).map(|i| out[i] - grain[i]).collect();
        let again = recompose(&Decomposition { grain: None, layers: d.layers.clone(), residual: d.residual.clone(), noise: d.noise.clone(), white_noise: d.white_noise.clone(), width: w, edges: OnceLock::new() }, &p);
        assert!(rest.iter().zip(&again).all(|(a, b)| (a - b).abs() < 1e-5));
        // …and its noise setting removes it.
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        let cleaned = recompose(&d, &WaveletParams { grain_denoise: 5.0, ..p.clone() });
        let left: Vec<f32> = (0..w * h).map(|i| cleaned[i] - again[i]).collect();
        assert!(rms(&left) < 0.05 * rms(grain), "grain left {}", rms(&left));
    }

    /// Without noise-adaptive sharpening, free-scale gains equivalent to classic ones give the
    /// same overall response below the cut-off.
    #[test]
    fn equivalent_gains_keep_the_response() {
        let mut classic = WaveletParams { scale_mode: ScaleMode::Dyadic, ..Default::default() };
        for (l, g) in classic.layers.iter_mut().zip([1.2, 1.0, 0.5, 0.1]) {
            l.gain = g;
        }
        let free = WaveletParams { scale_mode: ScaleMode::Free, first_scale: 1.6, ..Default::default() };
        let gains = equivalent_gains(&classic, &free, 0.2, &|_| 100.0);
        let response = |p: &WaveletParams, gains: &[f32], f: f32| 1.0 + responses(p, f, 0.0).1.iter().zip(gains).map(|(h, g)| h * g).sum::<f32>();
        let classic_gains: Vec<f32> = classic.layers.iter().map(|l| l.gain).collect();
        for f in [0.03, 0.08, 0.12, 0.17] {
            let (want, got) = (response(&classic, &classic_gains, f), response(&free, &gains, f));
            assert!((got - want).abs() < 0.12 * want, "f {f}: {got} vs {want} (gains {gains:?})");
        }
    }

    /// Correlated noise (white + blurred white, like a stack's): the noise of each layer is
    /// measured on the sky, where the white-noise estimate falls short at medium scales.
    #[test]
    fn correlated_noise_is_measured_per_layer() {
        let (w, h) = (384, 384);
        let fine = white_noise(w * h);
        let mut coarse: Vec<f32> = fine.iter().rev().copied().collect();
        let mut tmp = vec![0.0; w * h];
        let mut out = vec![0.0; w * h];
        let (taps, step) = gaussian_pass(2.0);
        convolve_separable(&coarse, &mut out, &mut tmp, w, h, &taps, step);
        std::mem::swap(&mut coarse, &mut out);
        // A disk in the middle, noise everywhere (an unclipped sky).
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - 192.0, (i / w) as f32 - 192.0);
                let disk = if x.hypot(y) < 60.0 { 0.5 } else { 0.05 };
                disk + 0.002 * fine[i] + 0.004 * coarse[i]
            })
            .collect();
        let p = WaveletParams { first_scale: 1.5, layers: vec![WaveletLayer::default(); 4], ..Default::default() };
        let d = decompose(&plane, w, h, &p, &|| false).unwrap();
        // Truth: the layers of the noise alone.
        let noise: Vec<f32> = (0..w * h).map(|i| 0.002 * fine[i] + 0.004 * coarse[i]).collect();
        let truth = decompose(&noise, w, h, &WaveletParams { noise_adaptive: 0.0, ..p.clone() }, &|| false).unwrap();
        for j in 1..3 {
            let real = mad_sigma(&truth.layers[j]);
            assert!((d.noise[j] - real).abs() < 0.3 * real, "layer {j}: estimated {} vs real {real}", d.noise[j]);
        }
        let real = mad_sigma(&truth.layers[2]);
        assert!(d.white_noise[2] < 0.75 * real, "white {} should fall short of {real}", d.white_noise[2]);
    }

    /// Clipped sky: nothing to measure on, the typical growth is used.
    #[test]
    fn clipped_sky_uses_typical_correlation() {
        let (w, h) = (256, 256);
        let n = white_noise(w * h);
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - 128.0, (i / w) as f32 - 128.0);
                if x.hypot(y) < 50.0 { 0.5 + 0.01 * n[i] } else { 0.0 }
            })
            .collect();
        let d = decompose(&plane, w, h, &WaveletParams::default(), &|| false).unwrap();
        for j in 0..d.noise.len() {
            assert!(d.noise[j] > d.white_noise[j] * 1.2, "layer {j}");
        }
        assert!(d.noise[2] / d.white_noise[2] > d.noise[0] / d.white_noise[0]);
    }

    #[test]
    fn cancellation_stops_early() {
        let plane = test_plane(32, 32);
        assert!(decompose(&plane, 32, 32, &WaveletParams::default(), &|| true).is_none());
    }
}
