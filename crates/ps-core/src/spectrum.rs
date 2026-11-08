//! What the image holds at each scale, from its radially averaged power spectrum.
//!
//! A blurred, noisy image has a recognisable spectrum: the subject's power falls off as a
//! power law, the blur (MTF²) makes it collapse at some frequency, and below that only the
//! floor of the noise is left. Fitting `P(f) = A·f^−α·e^(−4π²σ²f²) + N·e^(−4π²τ²f²)` gives
//! the noise floor and the frequency where the signal sinks into it: the finest detail the
//! image really holds. The floor is flat for white noise; stacking with resampling (drizzle,
//! sub-pixel alignment) smooths the noise a little, τ, and the floor slopes down.
//!
//! The signal-to-noise ratio at each frequency doesn't change under linear filtering, so the
//! cut-off measured on the raw stack still holds after deconvolution.

use crate::fft::{FftConvolver, next_fast_size};
use rayon::prelude::*;

/// Largest square analysed: big enough for a smooth spectrum, small enough to stay quick.
const MAX_SIDE: usize = 1536;
/// Below this side the spectrum is too coarse to fit.
const MIN_SIDE: usize = 96;
/// Lowest frequency fitted (cycles/px): coarser scales describe the subject's shape, not a
/// power law.
const F_LOW: f32 = 0.02;
const BINS: usize = 48;
/// Smoothing of the noise tried (px).
const NOISE_BLURS: [f32; 5] = [0.0, 0.3, 0.45, 0.6, 0.8];
/// Worse fits than this (relative RMS error) aren't trusted.
const MAX_MISFIT: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spectral {
    /// Frequency (cycles/px) where the subject's power drops to the noise floor. 0.5 when the
    /// detail reaches the single pixel.
    pub cutoff: f32,
    /// Gaussian blur σ (px) of the fall-off.
    pub sigma: f32,
    /// Smoothing of the noise (px): 0 for white noise.
    pub noise_blur: f32,
    /// Power-law slope of the subject.
    pub slope: f32,
    /// Signal amplitude and noise floor of the fitted model (power per frequency bin, in the
    /// analysed crop's own units: only their ratio means something).
    amplitude: f32,
    floor: f32,
    /// Relative RMS error of the fit in the fitted band (0 = perfect).
    pub misfit: f32,
}

impl Spectral {
    /// Fitted signal power over noise power at frequency `f` (cycles/px).
    pub fn snr(&self, f: f32) -> f32 {
        self.signal(f) / self.noise(f).max(1e-30)
    }

    /// Fitted noise power at `f`.
    pub fn noise(&self, f: f32) -> f32 {
        self.floor * model(f.max(1e-4), self.noise_blur, 0.0)
    }

    /// Fitted signal power at `f`.
    pub fn signal(&self, f: f32) -> f32 {
        model(f.max(1e-4), self.sigma, self.slope) * self.amplitude
    }
}

fn model(f: f32, sigma: f32, slope: f32) -> f32 {
    f.powf(-slope) * (-4.0 * std::f32::consts::PI.powi(2) * sigma * sigma * f * f).exp()
}

/// Radially averaged power of a `side`² crop centred near (`cx`, `cy`): (frequency, power)
/// per logarithmic bin from [`F_LOW`] to 0.5 cycles/px.
pub fn radial_power(plane: &[f32], w: usize, h: usize, cx: f32, cy: f32) -> Option<Vec<(f32, f32)>> {
    let mut side = w.min(h).min(MAX_SIDE);
    while next_fast_size(side) != side {
        side -= 1;
    }
    if side < MIN_SIDE {
        return None;
    }
    let x0 = ((cx - side as f32 / 2.0).round().max(0.0) as usize).min(w - side);
    let y0 = ((cy - side as f32 / 2.0).round().max(0.0) as usize).min(h - side);
    let mut crop: Vec<f32> = (0..side).flat_map(|y| plane[(y0 + y) * w + x0..(y0 + y) * w + x0 + side].iter().copied()).collect();
    let mean = crop.iter().map(|&v| v as f64).sum::<f64>() as f32 / crop.len() as f32;
    let hann: Vec<f32> = (0..side).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * (i as f32 + 0.5) / side as f32).cos()).collect();
    crop.par_chunks_mut(side).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * hann[x] * hann[y];
        }
    });
    let spec = FftConvolver::plan(side, side).forward(&crop);

    // Transposed layout: index = kx·side + ky, kx ∈ 0..=side/2.
    let (lo, hi) = (F_LOW.ln(), 0.5f32.ln());
    let bin_of = |f: f32| ((f.ln() - lo) / (hi - lo) * BINS as f32) as isize;
    let (sum, count) = spec
        .par_chunks(side)
        .enumerate()
        .fold(
            || (vec![0.0f64; BINS], vec![0u32; BINS]),
            |(mut sum, mut count), (kx, col)| {
                let fx = kx as f32 / side as f32;
                for (ky, c) in col.iter().enumerate() {
                    let fy = (if ky <= side / 2 { ky as f32 } else { ky as f32 - side as f32 }) / side as f32;
                    let f = fx.hypot(fy);
                    if f < F_LOW || f > 0.5 {
                        continue;
                    }
                    let b = bin_of(f).clamp(0, BINS as isize - 1) as usize;
                    sum[b] += c.norm_sqr() as f64;
                    count[b] += 1;
                }
                (sum, count)
            },
        )
        .reduce(
            || (vec![0.0f64; BINS], vec![0u32; BINS]),
            |(mut s, mut c), (s2, c2)| {
                s.iter_mut().zip(s2).for_each(|(a, b)| *a += b);
                c.iter_mut().zip(c2).for_each(|(a, b)| *a += b);
                (s, c)
            },
        );
    let bins: Vec<(f32, f32)> = (0..BINS)
        .filter(|&b| count[b] > 0)
        .map(|b| {
            let f = (lo + (b as f32 + 0.5) / BINS as f32 * (hi - lo)).exp();
            (f, (sum[b] / count[b] as f64) as f32)
        })
        .filter(|&(_, p)| p > 0.0)
        .collect();
    (bins.len() >= BINS / 2).then_some(bins)
}

/// Fit the blur-and-noise model to a radial spectrum. Every (σ, α) on a grid is tried with
/// the best amplitude and floor for it (linear least squares on relative errors, so each
/// decade of power counts the same).
pub fn fit(bins: &[(f32, f32)]) -> Option<Spectral> {
    let mut best: Option<Spectral> = None;
    let slopes = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let mut sigma = 0.2f32;
    while sigma < 8.0 {
        // The noise can't be smoother than the detail it rides on.
        for noise_blur in NOISE_BLURS.into_iter().filter(|&t| t < sigma) {
            for &slope in &slopes {
                if let Some((amplitude, floor, misfit)) = solve(bins, sigma, slope, noise_blur)
                    && best.is_none_or(|b| misfit < b.misfit)
                {
                    best = Some(Spectral { cutoff: 0.5, sigma, noise_blur, slope, amplitude, floor, misfit });
                }
            }
        }
        sigma *= 1.03;
    }
    let mut s = best?;
    s.cutoff = cutoff(&s);
    Some(s)
}

/// Amplitude, floor and relative RMS misfit for one (σ, α, τ).
fn solve(bins: &[(f32, f32)], sigma: f32, slope: f32, noise_blur: f32) -> Option<(f32, f32, f32)> {
    // Minimise Σ (A·u + N·v − 1)² with u = g/P, v = n/P (g, n: signal and noise shapes).
    let (mut uu, mut uv, mut vv, mut u1, mut v1) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let rows: Vec<(f64, f64)> = bins.iter().map(|&(f, p)| (model(f, sigma, slope) as f64 / p as f64, model(f, noise_blur, 0.0) as f64 / p as f64)).collect();
    for &(u, v) in &rows {
        (uu, uv, vv, u1, v1) = (uu + u * u, uv + u * v, vv + v * v, u1 + u, v1 + v);
    }
    let det = uu * vv - uv * uv;
    let (mut a, mut n) = if det.abs() > 1e-300 { ((u1 * vv - v1 * uv) / det, (v1 * uu - u1 * uv) / det) } else { (-1.0, -1.0) };
    if a <= 0.0 || n < 0.0 {
        // One of the two terms alone.
        (a, n) = if uu > 0.0 { (u1 / uu, 0.0) } else { return None };
        if a <= 0.0 {
            return None;
        }
    }
    let err = rows.iter().map(|&(u, v)| (a * u + n * v - 1.0).powi(2)).sum::<f64>() / rows.len() as f64;
    Some((a as f32, n as f32, err.sqrt() as f32))
}

/// First frequency where the fitted signal power falls to the floor.
fn cutoff(s: &Spectral) -> f32 {
    if s.floor <= 0.0 {
        return 0.5;
    }
    let mut f = F_LOW;
    while f < 0.5 {
        if s.snr(f) <= 1.0 {
            return f;
        }
        f *= 1.01;
    }
    0.5
}

/// [`radial_power`] + [`fit`], keeping only believable fits.
pub fn analyze(plane: &[f32], w: usize, h: usize, cx: f32, cy: f32) -> Option<Spectral> {
    measure(plane, w, h, cx, cy).0
}

/// The believable fit, if any, and the measured spectrum (empty when too small to measure).
pub fn measure(plane: &[f32], w: usize, h: usize, cx: f32, cy: f32) -> (Option<Spectral>, Vec<(f32, f32)>) {
    let Some(bins) = radial_power(plane, w, h, cx, cy) else { return (None, Vec::new()) };
    (fit(&bins).filter(|s| s.misfit < MAX_MISFIT), bins)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::fft::FftConvolver;
    use crate::psf::Psf;

    /// Random craters (sharp-edged disks of every size): a power-law scene.
    pub(crate) fn scene(side: usize) -> Vec<f32> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32
        };
        let mut img = vec![0.5f32; side * side];
        for _ in 0..1500 {
            let (cx, cy) = (rnd() * side as f32, rnd() * side as f32);
            let r = 1.5 + 40.0 * rnd().powi(3);
            let depth = 0.08 * (rnd() - 0.5);
            let (x0, x1) = ((cx - r).max(0.0) as usize, ((cx + r) as usize + 1).min(side));
            let (y0, y1) = ((cy - r).max(0.0) as usize, ((cy + r) as usize + 1).min(side));
            for y in y0..y1 {
                for x in x0..x1 {
                    if (x as f32 - cx).hypot(y as f32 - cy) < r {
                        img[y * side + x] += depth;
                    }
                }
            }
        }
        img
    }

    pub(crate) fn degraded(side: usize, sigma: f32, noise: f32) -> Vec<f32> {
        let sharp = scene(side);
        let blurred = FftConvolver::new(side, side, &Psf::Gaussian { sigma }.kernel()).convolve(&sharp);
        let n = crate::wavelet::white_noise(side * side);
        blurred.iter().zip(&n).map(|(b, n)| b + noise * n).collect()
    }

    #[test]
    fn measures_the_blur_and_the_cutoff() {
        let side = 512;
        let mut last_cutoff = 1.0;
        for sigma in [1.0, 1.8, 2.6] {
            let img = degraded(side, sigma, 0.004);
            let s = analyze(&img, side, side, 256.0, 256.0).expect("fit");
            assert!((s.sigma - sigma).abs() < 0.25 * sigma, "σ {sigma}: fitted {s:?}");
            // More blur → the detail ends sooner.
            assert!(s.cutoff < last_cutoff, "σ {sigma}: cutoff {} after {last_cutoff}", s.cutoff);
            last_cutoff = s.cutoff;
        }
    }

    #[test]
    fn more_noise_lowers_the_cutoff() {
        let side = 512;
        let clean = analyze(&degraded(side, 1.6, 0.001), side, side, 256.0, 256.0).unwrap();
        let noisy = analyze(&degraded(side, 1.6, 0.01), side, side, 256.0, 256.0).unwrap();
        assert!(noisy.cutoff < 0.85 * clean.cutoff, "clean {clean:?}, noisy {noisy:?}");
    }
}
