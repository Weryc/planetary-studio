//! Separable convolution with mirror boundaries, used by the starlet transform.

use rayon::prelude::*;

/// Mirror an out-of-range index back into `0..n` without repeating the edge sample
/// (… 2 1 0 1 2 …), the boundary convention used for the starlet transform.
#[inline]
pub fn reflect(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * n as isize - 2;
    let m = i.rem_euclid(period);
    (if m >= n as isize { period - m } else { m }) as usize
}

/// Convolve `src` (w×h) with a symmetric 1-D kernel along both axes. Kernel taps are spaced
/// `step` pixels apart, which implements the "holes" of the à trous algorithm.
pub fn convolve_separable(src: &[f32], dst: &mut [f32], tmp: &mut [f32], w: usize, h: usize, kernel: &[f32], step: usize) {
    tmp.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(|(out, row)| convolve_row(row, out, kernel, step));

    // Columns are processed row-by-row (adding whole shifted rows) so memory access stays sequential.
    let r = (kernel.len() / 2) as isize;
    dst.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        out.fill(0.0);
        for (k, &c) in kernel.iter().enumerate() {
            let sy = reflect(y as isize + (k as isize - r) * step as isize, h);
            for (o, &v) in out.iter_mut().zip(&tmp[sy * w..(sy + 1) * w]) {
                *o += c * v;
            }
        }
    });
}

/// Like [`convolve_separable`] (step 1) but with different taps along rows and columns.
pub fn convolve_rows_then_cols(src: &[f32], dst: &mut [f32], tmp: &mut [f32], w: usize, h: usize, rows: &[f32], cols: &[f32]) {
    tmp.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(|(out, row)| convolve_row(row, out, rows, 1));
    let r = (cols.len() / 2) as isize;
    dst.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        out.fill(0.0);
        for (k, &c) in cols.iter().enumerate() {
            let sy = reflect(y as isize + k as isize - r, h);
            for (o, &v) in out.iter_mut().zip(&tmp[sy * w..(sy + 1) * w]) {
                *o += c * v;
            }
        }
    });
}

fn convolve_row(row: &[f32], out: &mut [f32], kernel: &[f32], step: usize) {
    let w = row.len() as isize;
    let r = (kernel.len() / 2) as isize;
    out.fill(0.0);
    for (k, &c) in kernel.iter().enumerate() {
        let off = (k as isize - r) * step as isize;
        // Interior span where x + off stays inside the row: a plain shifted multiply-add.
        let lo = (-off).clamp(0, w) as usize;
        let hi = (w - off).clamp(0, w) as usize;
        if lo < hi {
            let src = &row[(lo as isize + off) as usize..(hi as isize + off) as usize];
            for (o, &v) in out[lo..hi].iter_mut().zip(src) {
                *o += c * v;
            }
        }
        for x in (0..lo).chain(hi.max(lo)..w as usize) {
            out[x] += c * row[reflect(x as isize + off, w as usize)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_mirrors_without_repeating_edge() {
        let got: Vec<usize> = (-3..7).map(|i| reflect(i, 4)).collect();
        assert_eq!(got, vec![3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
    }

    #[test]
    fn constant_image_is_preserved() {
        let (w, h) = (17, 9);
        let src = vec![0.5; w * h];
        let mut dst = vec![0.0; w * h];
        let mut tmp = vec![0.0; w * h];
        let k = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
        for step in [1, 4, 32] {
            convolve_separable(&src, &mut dst, &mut tmp, w, h, &k, step);
            assert!(dst.iter().all(|v| (v - 0.5).abs() < 1e-6), "step {step}");
        }
    }
}
