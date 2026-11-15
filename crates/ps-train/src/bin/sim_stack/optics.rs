//! Telescopes, the atmosphere above them and the PSF of one frame.
//!
//! The atmosphere has two frozen layers: one at the ground, which every point of the field
//! sees through the same air, and one high up (kilometres), which a point at field angle θ
//! sees displaced by h·θ. Over Jupiter's 45" the high layer is shifted by several apertures,
//! so the blur and the tilt change across the disk (anisoplanatism), as in real videos.
//!
//! Each frame is split in two parts: the tilt (the mean slope of the wavefront, which only
//! moves the image) is taken out of each PSF and applied as a smooth warp of the image;
//! what is left (speckles, halo) is the PSF, computed on a 3×3 grid and blended.
use crate::rng::Rng;
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub const ARCSEC_PER_RAD: f32 = 206_264.8;

#[derive(Clone, Copy, Debug)]
pub struct Telescope {
    pub name: &'static str,
    pub label: &'static str,
    pub aperture_mm: f32,
    /// Secondary diameter over the aperture.
    pub obstruction: f32,
    /// Effective focal length (Barlow included).
    pub focal_mm: f32,
}

pub const TELESCOPES: [Telescope; 7] = [
    Telescope { name: "apo102", label: "Rifrattore apo 102 a f/16 (Barlow)", aperture_mm: 102.0, obstruction: 0.0, focal_mm: 1630.0 },
    Telescope { name: "mak127", label: "Maksutov 127 f/12", aperture_mm: 127.0, obstruction: 0.30, focal_mm: 1500.0 },
    Telescope { name: "newton200", label: "Newton 200 a f/15 (Barlow 2.5×)", aperture_mm: 200.0, obstruction: 0.23, focal_mm: 3000.0 },
    Telescope { name: "c8", label: "Schmidt-Cassegrain 203 f/10", aperture_mm: 203.0, obstruction: 0.34, focal_mm: 2032.0 },
    Telescope { name: "sc235", label: "Schmidt-Cassegrain 235 f/10", aperture_mm: 235.0, obstruction: 0.36, focal_mm: 2350.0 },
    Telescope { name: "sc356", label: "Schmidt-Cassegrain 356 f/11", aperture_mm: 356.0, obstruction: 0.32, focal_mm: 3910.0 },
    // The setup of the real raw used for calibration (disk 443 px on 49.4" → 0.111"/px).
    Telescope { name: "edgehd11", label: "EdgeHD 11 a f/13 (calibrazione)", aperture_mm: 279.4, obstruction: 0.34, focal_mm: 3700.0 },
];

pub fn telescope(name: &str) -> Option<Telescope> {
    TELESCOPES.iter().copied().find(|t| t.name == name)
}

/// The turbulence of one night.
#[derive(Clone, Copy, Debug)]
pub struct Atmosphere {
    /// Aperture over Fried's r0 at 550 nm, both layers together (larger = worse).
    pub d_r0: f32,
    /// Share of the turbulence (in r0^-5/3) in the ground layer.
    pub ground_share: f32,
    /// Height of the high layer, km.
    pub high_km: f32,
    /// Outer scale of the turbulence, m.
    pub outer_m: f32,
}

/// Pupil grid for the PSFs (px): the PSF comes out on the same N×N grid, in image pixels.
pub const N: usize = 512;
/// Phase-screen pixels per aperture.
const SCREEN_PER_D: f32 = 64.0;

/// What one run needs about the optics, made once.
pub struct Optics {
    /// Image (fine) pixel in λ550/D.
    pub pix_d: f32,
    /// Wavelengths over 550 nm, one per channel.
    pub lambdas: Vec<f32>,
    /// Per wavelength: the pupil's cells (grid index, u, v in apertures from the centre).
    pupils: Vec<Vec<(u32, f32, f32)>>,
    fwd: Arc<dyn Fft<f32>>,
}

impl Optics {
    pub fn new(scope: Telescope, pix_arcsec: f32, lambdas: Vec<f32>) -> Self {
        let lambda_d = 0.55e-6 / (scope.aperture_mm * 1e-3) * ARCSEC_PER_RAD;
        let pix_d = pix_arcsec / lambda_d;
        let pupils = lambdas
            .iter()
            .map(|&l| {
                // The PSF's pixel at wavelength l is (DP/N)·l λ550/D: DP chosen so it is pix_d.
                let dp = N as f32 * pix_d / l;
                assert!(dp <= N as f32 / 2.0, "PSF sottocampionata: aumentare il sovracampionamento");
                let c = N as f32 / 2.0;
                let mut cells = Vec::new();
                for y in 0..N {
                    for x in 0..N {
                        let (u, v) = ((x as f32 + 0.5 - c) / dp, (y as f32 + 0.5 - c) / dp);
                        let r = 2.0 * u.hypot(v);
                        if r <= 1.0 && r >= scope.obstruction {
                            cells.push(((y * N + x) as u32, u, v));
                        }
                    }
                }
                cells
            })
            .collect();
        let fwd = FftPlanner::<f32>::new().plan_fft_forward(N);
        Optics { pix_d, lambdas, pupils, fwd }
    }

    /// The PSF at channel `c` through `phase` (radians at 550 nm, over u, v in apertures),
    /// centred at (N/2, N/2), summing to 1.
    pub fn psf(&self, c: usize, phase: &dyn Fn(f32, f32) -> f32) -> Vec<f32> {
        let l = self.lambdas[c];
        let mut field = vec![Complex32::default(); N * N];
        for &(i, u, v) in &self.pupils[c] {
            let p = phase(u, v) / l;
            field[i as usize] = Complex32::new(p.cos(), p.sin());
        }
        fft2(&self.fwd, &mut field);
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
}

/// Sequential 2-D FFT of an N×N grid (frames already run in parallel).
fn fft2(fft: &Arc<dyn Fft<f32>>, data: &mut [Complex32]) {
    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
    for row in data.chunks_mut(N) {
        fft.process_with_scratch(row, &mut scratch);
    }
    let mut t = vec![Complex32::default(); N * N];
    for y in 0..N {
        for x in 0..N {
            t[x * N + y] = data[y * N + x];
        }
    }
    for row in t.chunks_mut(N) {
        fft.process_with_scratch(row, &mut scratch);
    }
    for y in 0..N {
        for x in 0..N {
            data[x * N + y] = t[y * N + x];
        }
    }
}

/// A square von Kármán phase screen (radians at 550 nm), `n` px, r0 = `r0` px, outer scale
/// `l0` px: white noise filtered by the (κ² + κ0²)^(-11/12) amplitude spectrum, plus
/// subharmonics (Lane et al.) for the scales larger than the screen, which carry most of the
/// tilt; scaled so its structure function at small separations is 6.88 (r / r0)^(5/3).
fn screen(n: usize, r0: f32, l0: f32, rng: &mut Rng) -> Vec<f32> {
    let k0 = n as f32 / l0;
    let amp = |fx: f32, fy: f32| (fx * fx + fy * fy + k0 * k0).powf(-11.0 / 12.0);
    let mut field: Vec<Complex32> = (0..n * n)
        .map(|i| {
            let (kx, ky) = (i % n, i / n);
            let fx = if kx <= n / 2 { kx as f32 } else { kx as f32 - n as f32 };
            let fy = if ky <= n / 2 { ky as f32 } else { ky as f32 - n as f32 };
            let a = if kx == 0 && ky == 0 { 0.0 } else { amp(fx, fy) };
            Complex32::new(rng.gauss() * a, rng.gauss() * a)
        })
        .collect();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
    for row in field.chunks_mut(n) {
        fft.process_with_scratch(row, &mut scratch);
    }
    let mut t = vec![Complex32::default(); n * n];
    for y in 0..n {
        for x in 0..n {
            t[x * n + y] = field[y * n + x];
        }
    }
    for row in t.chunks_mut(n) {
        fft.process_with_scratch(row, &mut scratch);
    }
    let mut phase: Vec<f32> = t.iter().map(|c| c.re).collect();
    // Subharmonics: three levels of 3×3 cells, each a third of the one before.
    for level in 1..=3 {
        let step = 3f32.powi(-level);
        for j in -1i32..=1 {
            for i in -1i32..=1 {
                if i == 0 && j == 0 {
                    continue;
                }
                let (fx, fy) = (i as f32 * step, j as f32 * step);
                let c = Complex32::new(rng.gauss(), rng.gauss()) * (amp(fx, fy) * step);
                let row: Vec<Complex32> = (0..n).map(|x| Complex32::from_polar(1.0, std::f32::consts::TAU * fx * x as f32 / n as f32)).collect();
                for y in 0..n {
                    let cy = c * Complex32::from_polar(1.0, std::f32::consts::TAU * fy * y as f32 / n as f32);
                    for (x, r) in row.iter().enumerate() {
                        phase[y * n + x] += (cy * r).re;
                    }
                }
            }
        }
    }
    // Structure function at 2 and 4 px, on every 3rd row and column.
    let d = |s: usize| {
        let (mut sum, mut count) = (0.0f64, 0usize);
        for y in (0..n).step_by(3) {
            for x in 0..n - s {
                let a = phase[y * n + x] - phase[y * n + x + s];
                let b = phase[x * n + y] - phase[(x + s) * n + y];
                sum += (a * a + b * b) as f64;
                count += 2;
            }
        }
        sum / count as f64
    };
    let wanted = |s: f32| 6.88 * (s / r0).powf(5.0 / 3.0);
    let scale = ((wanted(2.0) + wanted(4.0)) as f64 / (d(2) + d(4))).sqrt() as f32;
    phase.into_iter().map(|v| v * scale).collect()
}

fn bilinear_wrap(s: &[f32], n: usize, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let at = |dx: i64, dy: i64| s[((y0 as i64 + dy).rem_euclid(n as i64) as usize) * n + (x0 as i64 + dx).rem_euclid(n as i64) as usize];
    (at(0, 0) * (1.0 - fx) + at(1, 0) * fx) * (1.0 - fy) + (at(0, 1) * (1.0 - fx) + at(1, 1) * fx) * fy
}

/// The air in front of the telescope during one frame.
pub struct Turbulence {
    ground: Vec<f32>,
    ng: usize,
    high: Vec<f32>,
    nh: usize,
    /// High-layer screen pixels per arcsec of field angle.
    high_px_per_arcsec: f32,
}

impl Turbulence {
    /// Fresh screens for one frame, wide enough for a field of `field_arcsec` across.
    pub fn new(atm: &Atmosphere, scope: &Telescope, field_arcsec: f32, rng: &mut Rng) -> Self {
        let d = scope.aperture_mm * 1e-3;
        let r0 = SCREEN_PER_D / atm.d_r0;
        // r0^-5/3 adds over layers.
        let r0_ground = r0 * atm.ground_share.max(1e-3).powf(-0.6);
        let r0_high = r0 * (1.0 - atm.ground_share).max(1e-3).powf(-0.6);
        let high_px_per_arcsec = atm.high_km * 1e3 / ARCSEC_PER_RAD / d * SCREEN_PER_D;
        let ng = ps_core::fft::next_fast_size((4.0 * SCREEN_PER_D) as usize);
        let nh = ps_core::fft::next_fast_size(((field_arcsec * 1.2 * high_px_per_arcsec) + 3.0 * SCREEN_PER_D) as usize);
        let l0 = atm.outer_m / d * SCREEN_PER_D;
        Turbulence { ground: screen(ng, r0_ground, l0, rng), ng, high: screen(nh, r0_high, l0, rng), nh, high_px_per_arcsec }
    }

    /// Phase (radians at 550 nm) at pupil point (u, v) (apertures) for field angle θ (arcsec).
    pub fn phase(&self, u: f32, v: f32, theta: (f32, f32)) -> f32 {
        let g = bilinear_wrap(&self.ground, self.ng, u * SCREEN_PER_D + self.ng as f32 / 2.0, v * SCREEN_PER_D + self.ng as f32 / 2.0);
        let hx = u * SCREEN_PER_D + self.nh as f32 / 2.0 + theta.0 * self.high_px_per_arcsec;
        let hy = v * SCREEN_PER_D + self.nh as f32 / 2.0 + theta.1 * self.high_px_per_arcsec;
        g + bilinear_wrap(&self.high, self.nh, hx, hy)
    }

    /// Least-squares plane φ ≈ c + a·u + b·v over the pupil: (c, a, b), radians at 550 nm.
    pub fn plane(&self, theta: (f32, f32), obstruction: f32) -> (f32, f32, f32) {
        const S: usize = 24;
        let (mut m, mut r) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
        for j in 0..S {
            for i in 0..S {
                let (u, v) = ((i as f32 + 0.5) / S as f32 - 0.5, (j as f32 + 0.5) / S as f32 - 0.5);
                let rad = 2.0 * u.hypot(v);
                if rad > 1.0 || rad < obstruction {
                    continue;
                }
                let p = self.phase(u, v, theta) as f64;
                let row = [1.0, u as f64, v as f64];
                for a in 0..3 {
                    for b in 0..3 {
                        m[a][b] += row[a] * row[b];
                    }
                    r[a] += row[a] * p;
                }
            }
        }
        let x = solve3(m, r);
        (x[0] as f32, x[1] as f32, x[2] as f32)
    }
}

fn solve3(mut m: [[f64; 3]; 3], mut r: [f64; 3]) -> [f64; 3] {
    for c in 0..3 {
        let p = (c..3).max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs())).unwrap();
        m.swap(c, p);
        r.swap(c, p);
        for k in c + 1..3 {
            let f = m[k][c] / m[c][c];
            for j in c..3 {
                m[k][j] -= f * m[c][j];
            }
            r[k] -= f * r[c];
        }
    }
    let mut x = [0.0; 3];
    for c in (0..3).rev() {
        x[c] = (r[c] - (c + 1..3).map(|j| m[c][j] * x[j]).sum::<f64>()) / m[c][c];
    }
    x
}

/// An N×N centred PSF wrapped onto a `w`×`h` canvas with its centre at the origin.
pub fn wrap_onto(psf: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let c = (N / 2) as isize;
    for y in 0..N {
        let dy = y as isize - c;
        if dy.unsigned_abs() >= h / 2 {
            continue;
        }
        for x in 0..N {
            let dx = x as isize - c;
            let v = psf[y * N + x];
            if v == 0.0 || dx.unsigned_abs() >= w / 2 {
                continue;
            }
            out[dy.rem_euclid(h as isize) as usize * w + dx.rem_euclid(w as isize) as usize] += v;
        }
    }
    out
}

/// Without any air: the diffraction pattern of the obstructed aperture, per channel.
pub fn airy(optics: &Optics) -> Vec<Vec<f32>> {
    (0..optics.lambdas.len()).into_par_iter().map(|c| optics.psf(c, &|_, _| 0.0)).collect()
}
