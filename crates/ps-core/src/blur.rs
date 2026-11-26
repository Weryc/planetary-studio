//! Gaussian blur of a plane, shared by the modules that need a plain smoothing.

use rayon::prelude::*;

/// What lies beyond the plane's edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// The outermost row/column repeated.
    Clamp,
    /// The plane mirrored at its edge.
    Reflect,
}

impl Edge {
    fn index(self, i: isize, n: usize) -> usize {
        let n = n as isize;
        match self {
            Edge::Clamp => i.clamp(0, n - 1) as usize,
            Edge::Reflect => {
                let m = i.rem_euclid(2 * n);
                (if m < n { m } else { 2 * n - 1 - m }) as usize
            }
        }
    }
}

/// Sampled Gaussian of σ samples, out to 3σ, summing to 1.
pub fn gaussian_taps(sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil() as i32;
    let taps: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = taps.iter().sum();
    taps.into_iter().map(|t| t / sum).collect()
}

/// Separable Gaussian blur of σ px.
pub fn gaussian_blur(plane: &[f32], w: usize, h: usize, sigma: f32, edge: Edge) -> Vec<f32> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let k = gaussian_taps(sigma);
    let r = (k.len() / 2) as isize;
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).zip(plane.par_chunks(w)).for_each(|(out, row)| {
        for (x, o) in out.iter_mut().enumerate() {
            *o = k.iter().enumerate().map(|(t, kv)| kv * row[edge.index(x as isize + t as isize - r, w)]).sum();
        }
    });
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (t, kv) in k.iter().enumerate() {
            let src = &tmp[edge.index(y as isize + t as isize - r, h) * w..][..w];
            row.iter_mut().zip(src).for_each(|(o, s)| *o += kv * s);
        }
    });
    out
}
