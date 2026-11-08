//! 2-D real FFT convolution with a fixed PSF, the workhorse of every deconvolution algorithm.
//!
//! Rows go through a real-to-complex FFT (half the work of a complex one), the result is
//! transposed and the columns transformed as contiguous rows. The spectrum is kept in that
//! transposed layout: pointwise products don't care, and it saves a transpose each way.

use crate::conv::reflect;
use crate::psf::Kernel2D;
use rayon::prelude::*;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub struct FftConvolver {
    w: usize,
    h: usize,
    wc: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    col_fwd: Arc<dyn Fft<f32>>,
    col_inv: Arc<dyn Fft<f32>>,
    otf: Vec<Complex32>,
}

impl FftConvolver {
    /// `w`×`h` is the (already padded) working size; the PSF is centred at the origin.
    pub fn new(w: usize, h: usize, psf: &Kernel2D) -> Self {
        let mut s = Self::plan(w, h);
        let mut shifted = vec![0.0f32; w * h];
        let (size, r) = (psf.size(), psf.radius as isize);
        for ky in 0..size {
            for kx in 0..size {
                let y = (ky as isize - r).rem_euclid(h as isize) as usize;
                let x = (kx as isize - r).rem_euclid(w as isize) as usize;
                shifted[y * w + x] += psf.data[ky * size + kx];
            }
        }
        s.otf = s.forward(&shifted);
        s
    }

    /// Transforms only (no PSF): for [`Self::forward`] / [`Self::inverse`].
    pub fn plan(w: usize, h: usize) -> Self {
        let mut real = RealFftPlanner::<f32>::new();
        let mut complex = FftPlanner::<f32>::new();
        Self { w, h, wc: w / 2 + 1, r2c: real.plan_fft_forward(w), c2r: real.plan_fft_inverse(w), col_fwd: complex.plan_fft_forward(h), col_inv: complex.plan_fft_inverse(h), otf: Vec::new() }
    }

    /// Optical transfer function (FFT of the PSF), in the transposed spectrum layout.
    pub fn otf(&self) -> &[Complex32] {
        &self.otf
    }

    pub fn forward(&self, img: &[f32]) -> Vec<Complex32> {
        let (w, h, wc) = (self.w, self.h, self.wc);
        let mut rows = vec![Complex32::default(); h * wc];
        rows.par_chunks_mut(wc).zip(img.par_chunks(w)).for_each_init(
            || (self.r2c.make_input_vec(), self.r2c.make_scratch_vec()),
            |(input, scratch), (out, src)| {
                input.copy_from_slice(src);
                self.r2c.process_with_scratch(input, out, scratch).expect("row FFT sizes match the plan");
            },
        );
        let mut cols = transpose(&rows, h, wc);
        let scratch_len = self.col_fwd.get_inplace_scratch_len();
        cols.par_chunks_mut(h).for_each_init(|| vec![Complex32::default(); scratch_len], |scratch, col| self.col_fwd.process_with_scratch(col, scratch));
        cols
    }

    pub fn inverse(&self, mut spec: Vec<Complex32>) -> Vec<f32> {
        let (w, h, wc) = (self.w, self.h, self.wc);
        let scratch_len = self.col_inv.get_inplace_scratch_len();
        spec.par_chunks_mut(h).for_each_init(|| vec![Complex32::default(); scratch_len], |scratch, col| self.col_inv.process_with_scratch(col, scratch));
        let mut rows = transpose(&spec, wc, h);
        let norm = 1.0 / (w * h) as f32;
        let mut out = vec![0.0f32; w * h];
        out.par_chunks_mut(w).zip(rows.par_chunks_mut(wc)).for_each_init(
            || self.c2r.make_scratch_vec(),
            |scratch, (dst, src)| {
                // Rounding leaves tiny imaginary parts on the DC/Nyquist bins; a real signal has none.
                src[0].im = 0.0;
                if w % 2 == 0 {
                    src[wc - 1].im = 0.0;
                }
                self.c2r.process_with_scratch(src, dst, scratch).expect("row IFFT sizes match the plan");
                dst.iter_mut().for_each(|v| *v *= norm);
            },
        );
        out
    }

    /// `img ⊛ psf`
    pub fn convolve(&self, img: &[f32]) -> Vec<f32> {
        let mut spec = self.forward(img);
        spec.par_iter_mut().zip(&self.otf).for_each(|(s, h)| *s *= h);
        self.inverse(spec)
    }

    /// `img ⋆ psf` (correlation, i.e. convolution with the flipped PSF).
    pub fn correlate(&self, img: &[f32]) -> Vec<f32> {
        let mut spec = self.forward(img);
        spec.par_iter_mut().zip(&self.otf).for_each(|(s, h)| *s *= h.conj());
        self.inverse(spec)
    }
}

/// Cache-blocked transpose: each task fills a band of TILE output rows, reading TILE×TILE tiles
/// so both source and destination stay in cache.
fn transpose(src: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
    const TILE: usize = 32;
    let mut dst = vec![Complex32::default(); rows * cols];
    dst.par_chunks_mut(rows * TILE).enumerate().for_each(|(band, out)| {
        let c0 = band * TILE;
        let band_cols = out.len() / rows;
        for r0 in (0..rows).step_by(TILE) {
            for r in r0..(r0 + TILE).min(rows) {
                let src_row = &src[r * cols + c0..r * cols + c0 + band_cols];
                for (dc, &v) in src_row.iter().enumerate() {
                    out[dc * rows + r] = v;
                }
            }
        }
    });
    dst
}

/// Smallest size ≥ n whose only prime factors are 2, 3 and 5 (fast FFT lengths).
pub fn next_fast_size(n: usize) -> usize {
    (n.max(1)..)
        .find(|&m| {
            let mut m = m;
            for p in [2, 3, 5] {
                while m % p == 0 {
                    m /= p;
                }
            }
            m == 1
        })
        .unwrap()
}

/// A plane mirror-padded to a fast FFT size, remembering where the original sits.
pub struct Padded {
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
    margin: usize,
    orig_w: usize,
    orig_h: usize,
}

impl Padded {
    /// Mirror padding keeps edges continuous; `margin` should be at least the PSF radius so the
    /// FFT's wrap-around seam lands in padding rather than on image data.
    pub fn new(plane: &[f32], w: usize, h: usize, margin: usize) -> Self {
        let pw = next_fast_size(w + 2 * margin);
        let ph = next_fast_size(h + 2 * margin);
        let mut data = vec![0.0f32; pw * ph];
        data.par_chunks_mut(pw).enumerate().for_each(|(y, row)| {
            let sy = reflect(y as isize - margin as isize, h);
            for (x, v) in row.iter_mut().enumerate() {
                *v = plane[sy * w + reflect(x as isize - margin as isize, w)];
            }
        });
        Self { data, w: pw, h: ph, margin, orig_w: w, orig_h: h }
    }

    /// Padding on each side of the original.
    pub fn margin(&self) -> usize {
        self.margin
    }

    pub fn crop(&self, padded: &[f32]) -> Vec<f32> {
        let (m, w) = (self.margin, self.orig_w);
        (0..self.orig_h).flat_map(|y| &padded[(y + m) * self.w + m..(y + m) * self.w + m + w]).copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::psf::Psf;

    #[test]
    fn fast_sizes() {
        assert_eq!(next_fast_size(7), 8);
        assert_eq!(next_fast_size(1001), 1024);
        assert_eq!(next_fast_size(97), 100);
    }

    #[test]
    fn fft_convolution_matches_direct_convolution() {
        let (w, h) = (30, 20);
        let img: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 11) as f32 / 10.0).collect();
        let psf = Psf::Gaussian { sigma: 1.2 }.kernel();
        let conv = FftConvolver::new(w, h, &psf);
        let got = conv.convolve(&img);

        let (size, r) = (psf.size(), psf.radius as isize);
        for (y, x) in [(10, 15), (0, 0), (19, 29)] {
            let mut want = 0.0;
            for ky in 0..size {
                for kx in 0..size {
                    // Circular convolution, as the FFT computes.
                    let sy = (y as isize - (ky as isize - r)).rem_euclid(h as isize) as usize;
                    let sx = (x as isize - (kx as isize - r)).rem_euclid(w as isize) as usize;
                    want += psf.data[ky * size + kx] * img[sy * w + sx];
                }
            }
            assert!((got[y * w + x] - want).abs() < 1e-4, "({x},{y}): {} vs {want}", got[y * w + x]);
        }
    }

    #[test]
    fn pad_then_crop_is_identity() {
        let plane: Vec<f32> = (0..35).map(|v| v as f32).collect();
        let p = Padded::new(&plane, 7, 5, 4);
        assert_eq!(p.crop(&p.data), plane);
    }
}
