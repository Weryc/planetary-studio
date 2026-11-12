//! Simulated seeing as a lucky-imaging stack really sees it.
//!
//! Each frame looks through its own random atmosphere: a Kolmogorov phase screen over the
//! telescope's pupil (central obstruction included), whose far-field image is that frame's PSF
//! (speckles, tilt). As a stacker does, only the sharpest frames are kept, each is re-centred
//! on its own tilt (alignment), with a small residual error (alignment points are never
//! perfect), and they are averaged. The result has a sharp core and the wide halo of the
//! seeing, which takes contrast from the broad structures too.
//!
//! Blurring a truth with the stacked PSF is exactly what stacking blurred frames does (the sum
//! of convolutions is the convolution with the summed PSF), so the frames needn't be images.
//!
//! Colour: the same air delays every wavelength by the same path, so the phase scales as
//! 1/λ and the PSF's size as λ: blue is blurred by the seeing more, red by diffraction more.
//! Atmospheric dispersion spreads each colour band into a short vertical streak (longer in
//! blue); the stacker's RGB alignment removes the shift between bands, not the streaks.
use ps_core::Image;
use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

/// Pupil grid and aperture across it (px): the PSF's fine pixel is DP/N λ/D wide.
const N: usize = 512;
const DP: f32 = 48.0;
const OBSTRUCTION: f32 = 0.33;
/// Wavelengths over 550 nm: R, G, B, and mono (a red/IR filter).
pub const LAMBDA_RGB: [f32; 3] = [620.0 / 550.0, 1.0, 460.0 / 550.0];
pub const LAMBDA_MONO: f32 = 680.0 / 550.0;

/// One kind of night.
#[derive(Clone, Copy, Debug)]
pub struct Seeing {
    pub label: &'static str,
    /// Aperture over Fried's r0 at 550 nm (larger = worse).
    pub d_r0: f32,
    /// Share of frames the stacker keeps.
    pub keep: f32,
    /// Residual alignment error, λ/D (RMS per axis).
    pub jitter: f32,
    /// Dispersion streak in the blue band, λ/D (green ×0.55, red ×0.3, mono ×0.25).
    pub dispersion: f32,
}

/// Calm, average and turbulent nights.
pub const NIGHTS: [Seeing; 3] = [
    Seeing { label: "calmo", d_r0: 2.5, keep: 0.3, jitter: 0.1, dispersion: 0.4 },
    Seeing { label: "medio", d_r0: 4.5, keep: 0.2, jitter: 0.15, dispersion: 0.6 },
    Seeing { label: "turbolento", d_r0: 7.5, keep: 0.1, jitter: 0.2, dispersion: 0.8 },
];

const FRAMES: usize = 400;

struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f32 {
        ((-2.0 * self.uniform().max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()) as f32
    }
}

/// A Kolmogorov phase screen (radians at 550 nm) on the N×N grid, for r0 = `r0` grid px. Made by
/// filtering white noise with the κ^(-11/6) amplitude spectrum, then scaled so its structure
/// function at small separations matches 6.88 (r / r0)^(5/3).
fn screen(r0: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut field: Vec<Complex32> = (0..N * N)
        .map(|i| {
            let (kx, ky) = (i % N, i / N);
            let fx = if kx <= N / 2 { kx as f32 } else { kx as f32 - N as f32 };
            let fy = if ky <= N / 2 { ky as f32 } else { ky as f32 - N as f32 };
            let k = fx.hypot(fy);
            let amp = if k == 0.0 { 0.0 } else { k.powf(-11.0 / 6.0) };
            Complex32::new(rng.gauss() * amp, rng.gauss() * amp)
        })
        .collect();
    fft2(&mut field, false);
    let phase: Vec<f32> = field.iter().map(|c| c.re).collect();
    // Structure function at separations of 2 and 4 px, averaged over the grid.
    let d = |s: usize| {
        let mut sum = 0.0f64;
        for y in 0..N {
            for x in 0..N - s {
                let a = phase[y * N + x] - phase[y * N + x + s];
                let b = phase[x * N + y] - phase[(x + s) * N + y];
                sum += (a * a + b * b) as f64;
            }
        }
        sum / (2 * N * (N - s)) as f64
    };
    let wanted = |s: f32| 6.88 * (s / r0).powf(5.0 / 3.0);
    let scale = ((wanted(2.0) + wanted(4.0)) as f64 / (d(2) + d(4))).sqrt() as f32;
    phase.into_iter().map(|v| v * scale).collect()
}

/// In-place 2-D FFT of an N×N grid.
fn fft2(data: &mut [Complex32], inverse: bool) {
    let mut planner = FftPlanner::<f32>::new();
    let fft = if inverse { planner.plan_fft_inverse(N) } else { planner.plan_fft_forward(N) };
    data.par_chunks_mut(N).for_each(|row| fft.process(row));
    let mut t = vec![Complex32::default(); N * N];
    for y in 0..N {
        for x in 0..N {
            t[x * N + y] = data[y * N + x];
        }
    }
    t.par_chunks_mut(N).for_each(|row| fft.process(row));
    for y in 0..N {
        for x in 0..N {
            data[x * N + y] = t[y * N + x];
        }
    }
}

/// The pupil: 1 inside the annulus, centred on the grid.
fn pupil() -> Vec<(usize, usize)> {
    let c = N as f32 / 2.0;
    let mut v = Vec::new();
    for y in 0..N {
        for x in 0..N {
            let r = (x as f32 + 0.5 - c).hypot(y as f32 + 0.5 - c) / (DP / 2.0);
            if (OBSTRUCTION..=1.0).contains(&r) {
                v.push((x, y));
            }
        }
    }
    v
}

/// The far-field image of the pupil with phase `phase` (radians, at this wavelength), centred.
fn frame_psf(pupil: &[(usize, usize)], phase: &dyn Fn(usize, usize) -> f32) -> Vec<f32> {
    let mut field = vec![Complex32::default(); N * N];
    for &(x, y) in pupil {
        let p = phase(x, y);
        field[y * N + x] = Complex32::new(p.cos(), p.sin());
    }
    fft2(&mut field, false);
    // FFT-shifted so the centre is at (N/2, N/2).
    let mut out = vec![0.0f32; N * N];
    for y in 0..N {
        for x in 0..N {
            out[((y + N / 2) % N) * N + (x + N / 2) % N] = field[y * N + x].norm_sqr();
        }
    }
    let total: f32 = out.iter().sum();
    out.iter_mut().for_each(|v| *v /= total);
    out
}

fn centroid(psf: &[f32]) -> (f32, f32) {
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    for (i, &v) in psf.iter().enumerate() {
        sx += v as f64 * (i % N) as f64;
        sy += v as f64 * (i / N) as f64;
    }
    (sx as f32 - (N / 2) as f32, sy as f32 - (N / 2) as f32)
}

/// The stacked PSF at each wavelength (over 550 nm), on the fine grid (pixel DP/N · λ/D at that
/// wavelength), centred, with the dispersion streaks (blue first by wavelength: `streaks` λ/D).
pub fn stacked(night: &Seeing, lambdas: &[f32], streaks: &[f32], seed: u64) -> Vec<Vec<f32>> {
    let pupil = pupil();
    let r0 = DP / night.d_r0;
    // Many frames come from one screen: each looks through a different patch of it.
    let per_screen = 40;
    let screens: Vec<Vec<f32>> = (0..FRAMES.div_ceil(per_screen)).into_par_iter().map(|k| screen(r0, seed * 1000 + k as u64)).collect();
    let frames: Vec<(f32, Vec<Vec<f32>>)> = (0..FRAMES)
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng((seed * 7919 + i as u64).wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
            let s = &screens[i / per_screen];
            let (ox, oy) = ((rng.uniform() * N as f64) as usize, (rng.uniform() * N as f64) as usize);
            let psfs: Vec<Vec<f32>> = lambdas.iter().map(|&l| frame_psf(&pupil, &|x, y| s[((y + oy) % N) * N + (x + ox) % N] / l)).collect();
            // Sharpness the stacker sees: the peak of the green (or only) frame.
            let reference = &psfs[psfs.len() / 2];
            let peak = reference.iter().copied().fold(0.0f32, f32::max);
            (peak, psfs)
        })
        .collect();
    let mut order: Vec<usize> = (0..FRAMES).collect();
    order.sort_by(|&a, &b| frames[b].0.total_cmp(&frames[a].0));
    let kept = &order[..((FRAMES as f32 * night.keep) as usize).max(1)];
    let fine = DP / N as f32; // λ/D per fine pixel at its own wavelength
    let mut rng = Rng(seed ^ 0xABCD_EF01);
    let mut sums = vec![vec![0.0f32; N * N]; lambdas.len()];
    let reference = lambdas.len() / 2;
    for &i in kept {
        let psfs = &frames[i].1;
        // Aligned on the frame's tilt (the same angle at every wavelength), plus the residual.
        let (cx, cy) = centroid(&psfs[reference]);
        let (jx, jy) = (rng.gauss() * night.jitter / fine, rng.gauss() * night.jitter / fine);
        for (c, psf) in psfs.iter().enumerate() {
            let k = lambdas[reference] / lambdas[c];
            let (dx, dy) = (((cx + jx) * k).round() as isize, ((cy + jy) * k).round() as isize);
            for y in 0..N {
                let sy = y as isize + dy;
                if !(0..N as isize).contains(&sy) {
                    continue;
                }
                for x in 0..N {
                    let sx = x as isize + dx;
                    if (0..N as isize).contains(&sx) {
                        sums[c][y * N + x] += psf[sy as usize * N + sx as usize];
                    }
                }
            }
        }
    }
    sums.into_iter()
        .zip(streaks)
        .map(|(sum, &streak)| {
            // The dispersion streak: a vertical box of that length.
            let len = (streak / fine).round() as usize;
            let mut out = sum.clone();
            if len > 0 {
                for x in 0..N {
                    for y in 0..N {
                        let mut acc = 0.0;
                        for t in 0..=len {
                            let sy = y as isize + t as isize - (len / 2) as isize;
                            if (0..N as isize).contains(&sy) {
                                acc += sum[sy as usize * N + x];
                            }
                        }
                        out[y * N + x] = acc / (len + 1) as f32;
                    }
                }
            }
            let total: f32 = out.iter().sum();
            out.into_iter().map(|v| v / total).collect()
        })
        .collect()
}

/// A fine-grid PSF at wavelength `lambda` (over 550 nm) as a kernel in image pixels of `k` λ/D
/// (at 550 nm) each, trimmed where 99.7 % of the light is in.
pub fn to_pixels(fine_psf: &[f32], lambda: f32, k: f32) -> ps_core::psf::Kernel2D {
    let fine = DP / N as f32 * lambda; // λ550/D per fine pixel
    let per = fine / k; // image px per fine px
    let max_r = 160usize;
    let full = ((N as f32 / 2.0 * per).ceil() as usize).min(max_r);
    let size = 2 * full + 1;
    let mut data = vec![0.0f32; size * size];
    let c = (N / 2) as f32;
    for y in 0..N {
        for x in 0..N {
            let v = fine_psf[y * N + x];
            if v <= 0.0 {
                continue;
            }
            let (px, py) = ((x as f32 - c) * per + full as f32, (y as f32 - c) * per + full as f32);
            let (x0, y0) = (px.floor(), py.floor());
            let (fx, fy) = (px - x0, py - y0);
            for (dx, dy, w) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
                let (ix, iy) = (x0 as isize + dx, y0 as isize + dy);
                if (0..size as isize).contains(&ix) && (0..size as isize).contains(&iy) {
                    data[iy as usize * size + ix as usize] += v * w;
                }
            }
        }
    }
    // Trim to the radius holding 99.7 % of the light.
    let total: f32 = data.iter().sum();
    let mut radius = full;
    for r in 4..full {
        let inside: f32 = (full - r..=full + r).flat_map(|y| (full - r..=full + r).map(move |x| (x, y))).map(|(x, y)| data[y * size + x]).sum();
        if inside >= 0.997 * total {
            radius = r;
            break;
        }
    }
    let side = 2 * radius + 1;
    let mut trimmed: Vec<f32> = (0..side * side).map(|i| data[(full - radius + i / side) * size + full - radius + i % side]).collect();
    let t: f32 = trimmed.iter().sum();
    trimmed.iter_mut().for_each(|v| *v /= t);
    ps_core::psf::Kernel2D { radius, data: trimmed }
}

/// Blur each channel with its own kernel (one for mono).
pub fn blur_with(img: &Image, kernels: &[ps_core::psf::Kernel2D]) -> Image {
    let (w, h) = (img.width(), img.height());
    let planes = img
        .channels()
        .iter()
        .enumerate()
        .map(|(c, plane)| {
            let k = &kernels[c.min(kernels.len() - 1)];
            let padded = ps_core::fft::Padded::new(plane, w, h, k.radius + 2);
            let conv = ps_core::fft::FftConvolver::new(padded.w, padded.h, k);
            padded.crop(&conv.convolve(&padded.data))
        })
        .collect();
    Image::new(w, h, planes, 16)
}

/// FWHM (fine px) and share of light within 1 λ/D of the centre: for the log.
pub fn describe(fine_psf: &[f32]) -> (f32, f32) {
    let c = N / 2;
    let peak = fine_psf[c * N + c];
    let mut fwhm = 0.0;
    for r in 0..N / 2 {
        if fine_psf[c * N + c + r] < peak / 2.0 {
            fwhm = 2.0 * r as f32;
            break;
        }
    }
    let rad = N as f32 / DP;
    let mut core = 0.0;
    for y in 0..N {
        for x in 0..N {
            if (x as f32 - c as f32).hypot(y as f32 - c as f32) <= rad {
                core += fine_psf[y * N + x];
            }
        }
    }
    (fwhm * DP / N as f32, core)
}
