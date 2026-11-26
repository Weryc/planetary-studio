//! The learned noise reduction: a small U-Net that sees the luminance of a stack and the noise
//! level measured on it, and answers with the noise to take away.
//!
//! It was trained outside the app (`crates/ps-train`) on sharp references (Hubble, Cassini, SDO,
//! NASA's LRO Moon) brought to amateur sizes and blurred by a simulated lucky-imaging seeing,
//! the clean stack being the truth; noise is added on the fly with random level, grain size
//! (white to smoothed by the stacker's resampling) and dependence on brightness, sometimes
//! after a sharpening of both (the coloured noise a deconvolution leaves). So it has learnt
//! what planetary, lunar and solar detail looks like under stack noise, not photographs.
//!
//! Network (every convolution 3×3, ReLU after all but the last):
//! full size 3→16→16; half 16→32 (stride 2) →32; quarter 32→64 (stride 2) →64→64; back up by
//! nearest-neighbour doubling and a convolution, adding the skip of that size, then one more
//! convolution; last 16→1, the noise in units of σ (so strength 0 changes nothing). About
//! 130 000 weights, ~28 GMAC per megapixel.
//!
//! The inputs are normalised the same way in training and here ([`normalise`]): luminance over
//! its 99.5th percentile; its fine detail ([`detail`]: minus a 1 px Gaussian blur) in units of
//! the noise σ, where the noise has about unit size whatever the signal-to-noise ratio (in the
//! luminance alone it is a few thousandths and the network could not learn to see it); and a
//! plane holding [`SIGMA_SCALE`] × the noise σ measured by [`crate::wavelet::estimate_noise`]
//! on the whole image, over the same scale. The "strength"
//! multiplies that σ: the network cleans as if the noise were that much stronger or weaker.
//!
//! Inference runs on the CPU by tiles (512² with a 48 px margin, wider than what the network
//! sees), so memory stays small and the result has no seams.

use rayon::prelude::*;
use std::sync::OnceLock;

/// Convolutions in order: (input channels, output channels, stride).
pub const LAYERS: [(usize, usize, usize); 13] = [
    (3, 16, 1),  // 0  e0a
    (16, 16, 1), // 1  e0b  → skip 0
    (16, 32, 2), // 2  d1
    (32, 32, 1), // 3  e1   → skip 1
    (32, 64, 2), // 4  d2
    (64, 64, 1), // 5  m2a
    (64, 64, 1), // 6  m2b
    (64, 32, 1), // 7  u1 (after doubling), + skip 1
    (32, 32, 1), // 8  c1
    (32, 16, 1), // 9  u0 (after doubling), + skip 0
    (16, 16, 1), // 10 c0
    (16, 16, 1), // 11 c0b
    (16, 1, 1),  // 12 out (no ReLU): the noise
];

/// The noise plane holds this × σ (normalised), so typical values are around 0.1.
pub const SIGMA_SCALE: f32 = 30.0;

/// Percentile of the luminance that becomes 1.
pub const NORM_PERCENTILE: f32 = 0.995;

/// Sides must be multiples of this (two halvings).
pub const MULTIPLE: usize = 4;

const MAGIC: &[u8; 8] = b"PSDN0001";

/// The trained weights (`ps-train`'s `denoise_train` writes them).
static WEIGHTS: &[u8] = include_bytes!("denoise_weights.bin");

struct Conv {
    cin: usize,
    cout: usize,
    stride: usize,
    /// [cout][cin][3][3]
    w: Vec<f32>,
    b: Vec<f32>,
}

pub struct Net {
    layers: Vec<Conv>,
}

/// Number of weights the file must hold.
pub fn weight_count() -> usize {
    LAYERS.iter().map(|&(i, o, _)| o * i * 9 + o).sum()
}

impl Net {
    /// `MAGIC`, then for every layer its weights [cout][cin][3][3] and biases, little-endian f32.
    pub fn from_bytes(bytes: &[u8]) -> Option<Net> {
        let body = bytes.strip_prefix(MAGIC.as_slice())?;
        if body.len() != weight_count() * 4 {
            return None;
        }
        let mut values = body.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c));
        let layers = LAYERS
            .iter()
            .map(|&(cin, cout, stride)| {
                let w = values.by_ref().take(cout * cin * 9).collect();
                let b = values.by_ref().take(cout).collect();
                Conv { cin, cout, stride, w, b }
            })
            .collect();
        Some(Net { layers })
    }

    /// The inverse of [`Net::from_bytes`], for the trainer: weights and biases per layer.
    pub fn to_bytes(layers: &[(Vec<f32>, Vec<f32>)]) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        for (w, b) in layers {
            out.extend(w.iter().chain(b).flat_map(|v| v.to_le_bytes()));
        }
        out
    }

    /// Noise predicted for a normalised tile (`w`, `h` multiples of [`MULTIPLE`]).
    pub fn noise(&self, y: &[f32], noise_level: f32, w: usize, h: usize) -> Vec<f32> {
        let mut input = Act::zeros(3, w, h);
        input.set_plane(0, y);
        input.set_plane(1, &detail(y, w, h, noise_level / SIGMA_SCALE));
        input.set_plane(2, &vec![noise_level; w * h]);
        let l = &self.layers;
        let e0 = conv(&conv(&input, &l[0], true), &l[1], true);
        let e1 = conv(&conv(&e0, &l[2], true), &l[3], true);
        let m = conv(&conv(&conv(&e1, &l[4], true), &l[5], true), &l[6], true);
        let mut u1 = conv(&double(&m), &l[7], true);
        u1.add(&e1);
        let c1 = conv(&u1, &l[8], true);
        let mut u0 = conv(&double(&c1), &l[9], true);
        u0.add(&e0);
        let c0 = conv(&conv(&u0, &l[10], true), &l[11], true);
        // The network answers in units of the noise σ.
        let sigma = noise_level / SIGMA_SCALE;
        conv(&c0, &l[12], false).plane(0).into_iter().map(|v| v * sigma).collect()
    }
}

/// Fine detail in units of `sigma`: the plane minus its Gaussian blur of 1 px (edges clamped).
pub fn detail(y: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    const K: [f32; 7] = [0.004433, 0.053991, 0.241971, 0.398943, 0.241971, 0.053991, 0.004433];
    let norm: f32 = K.iter().sum();
    let at = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
    let rows: Vec<f32> = (0..w * h).into_par_iter().map(|i| (0..7).map(|k| K[k] * y[(i / w) * w + at((i % w) as isize + k as isize - 3, w)]).sum::<f32>() / norm).collect();
    let inv = 1.0 / sigma.max(1e-6);
    (0..w * h).into_par_iter().map(|i| (y[i] - (0..7).map(|k| K[k] * rows[at((i / w) as isize + k as isize - 3, h) * w + i % w]).sum::<f32>() / norm) * inv).collect()
}

/// The trained network, if the weights are present and match the architecture.
pub fn model() -> Option<&'static Net> {
    static NET: OnceLock<Option<Net>> = OnceLock::new();
    NET.get_or_init(|| Net::from_bytes(WEIGHTS)).as_ref()
}

/// Whether the learned method can run (weights shipped with this build).
pub fn available() -> bool {
    model().is_some()
}

/// The luminance scale: its [`NORM_PERCENTILE`] (from a sample), never below 1e-3.
pub fn normalise(y: &[f32]) -> f32 {
    let stride = y.len().div_ceil(1 << 18).max(1);
    let mut sample: Vec<f32> = y.iter().step_by(stride).copied().collect();
    if sample.is_empty() {
        return 1.0;
    }
    let k = ((sample.len() - 1) as f32 * NORM_PERCENTILE) as usize;
    sample.select_nth_unstable_by(k, f32::total_cmp).1.max(1e-3)
}

/// Clean a luminance plane: `sigma` is the noise measured on it, `strength` scales it.
pub fn denoise(net: &Net, y: &[f32], w: usize, h: usize, sigma: f32, strength: f32, cancel: &(dyn Fn() -> bool + Sync)) -> Option<Vec<f32>> {
    const CORE: usize = 512;
    const MARGIN: usize = 48;
    let scale = normalise(y);
    let level = SIGMA_SCALE * sigma.max(0.0) * strength.max(0.0) / scale;
    let mut out = y.to_vec();
    let tiles: Vec<(usize, usize)> = (0..h.div_ceil(CORE)).flat_map(|ty| (0..w.div_ceil(CORE)).map(move |tx| (tx * CORE, ty * CORE))).collect();
    // Tiles one after another (each is parallel inside), so cancelling is prompt.
    for (x0, y0) in tiles {
        if cancel() {
            return None;
        }
        let (cw, ch) = (CORE.min(w - x0), CORE.min(h - y0));
        // The tile with its margin, clipped to the image, then padded by mirroring to a multiple.
        let (ax, ay) = (x0.saturating_sub(MARGIN), y0.saturating_sub(MARGIN));
        let (bx, by) = ((x0 + cw + MARGIN).min(w), (y0 + ch + MARGIN).min(h));
        let (tw, th) = (bx - ax, by - ay);
        let (pw, ph) = (tw.next_multiple_of(MULTIPLE), th.next_multiple_of(MULTIPLE));
        let mirror = |i: usize, n: usize| if i < n { i } else { (2 * n).saturating_sub(i + 2).min(n - 1) };
        let tile: Vec<f32> = (0..pw * ph).map(|i| y[(ay + mirror(i / pw, th)) * w + ax + mirror(i % pw, tw)] / scale).collect();
        let noise = net.noise(&tile, level, pw, ph);
        for yy in y0..y0 + ch {
            for xx in x0..x0 + cw {
                out[yy * w + xx] -= noise[(yy - ay) * pw + xx - ax] * scale;
            }
        }
    }
    Some(out)
}

/// Activations: `c` planes of (w+2)×(h+2) with a zero border, so 3×3 kernels need no bounds checks.
struct Act {
    c: usize,
    w: usize,
    h: usize,
    data: Vec<f32>,
}

impl Act {
    fn zeros(c: usize, w: usize, h: usize) -> Act {
        Act { c, w, h, data: vec![0.0; c * (w + 2) * (h + 2)] }
    }

    fn stride(&self) -> usize {
        (self.w + 2) * (self.h + 2)
    }

    fn row(&self, c: usize, padded_y: usize) -> &[f32] {
        let start = c * self.stride() + padded_y * (self.w + 2);
        &self.data[start..start + self.w + 2]
    }

    fn set_plane(&mut self, c: usize, v: &[f32]) {
        let (w, s) = (self.w, self.stride());
        for y in 0..self.h {
            let start = c * s + (y + 1) * (w + 2) + 1;
            self.data[start..start + w].copy_from_slice(&v[y * w..(y + 1) * w]);
        }
    }

    fn plane(&self, c: usize) -> Vec<f32> {
        (0..self.h).flat_map(|y| self.row(c, y + 1)[1..self.w + 1].iter().copied()).collect()
    }

    fn add(&mut self, other: &Act) {
        // Borders are zero in both, so adding everything keeps them zero.
        self.data.par_iter_mut().zip(&other.data).for_each(|(a, b)| *a += b);
    }
}

/// 3×3 convolution with zero padding (as in training), optional ReLU.
fn conv(input: &Act, l: &Conv, relu: bool) -> Act {
    debug_assert_eq!(input.c, l.cin);
    let (ow, oh) = (input.w / l.stride, input.h / l.stride);
    let rows: Vec<Vec<f32>> = (0..oh)
        .into_par_iter()
        .map(|oy| {
            let mut acc = vec![0.0f32; l.cout * ow];
            for (co, b) in l.b.iter().enumerate() {
                acc[co * ow..(co + 1) * ow].fill(*b);
            }
            let mut seg = vec![0.0f32; ow];
            for ci in 0..l.cin {
                for ky in 0..3 {
                    let src = input.row(ci, oy * l.stride + ky);
                    for kx in 0..3 {
                        let seg: &[f32] = if l.stride == 1 {
                            &src[kx..kx + ow]
                        } else {
                            for (x, s) in seg.iter_mut().enumerate() {
                                *s = src[x * l.stride + kx];
                            }
                            &seg
                        };
                        for co in 0..l.cout {
                            let wv = l.w[((co * l.cin + ci) * 3 + ky) * 3 + kx];
                            for (a, s) in acc[co * ow..(co + 1) * ow].iter_mut().zip(seg) {
                                *a += wv * s;
                            }
                        }
                    }
                }
            }
            if relu {
                acc.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            acc
        })
        .collect();
    let mut out = Act::zeros(l.cout, ow, oh);
    let s = out.stride();
    for (oy, acc) in rows.iter().enumerate() {
        for co in 0..l.cout {
            let start = co * s + (oy + 1) * (ow + 2) + 1;
            out.data[start..start + ow].copy_from_slice(&acc[co * ow..(co + 1) * ow]);
        }
    }
    out
}

/// Nearest-neighbour doubling.
fn double(input: &Act) -> Act {
    let (w, h) = (input.w * 2, input.h * 2);
    let mut out = Act::zeros(input.c, w, h);
    for c in 0..input.c {
        let plane: Vec<f32> = (0..w * h).map(|i| input.row(c, i / w / 2 + 1)[(i % w) / 2 + 1]).collect();
        out.set_plane(c, &plane);
    }
    out
}
