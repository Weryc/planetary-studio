//! What the noise estimator looks at: the brightness and the colour of an image, band by band,
//! inside the subject and in the sky around it.
//!
//! The estimator itself (trained in `ps-train` on simulated stacks, where the noise is known
//! because the sharp original is) predicts from these, for the image's subject:
//! - the grain: the luminance's noise, in 24 frequency bands;
//! - the colour noise: the noise of the two colour differences (R − Y and B − Y, as the app's
//!   colour noise reduction sees them), in the same bands.
//!
//! The bands and the framing are those of [`crate::curve_net`], so both networks see the
//! image the same way; nothing depends on the image's brightness scale.

use crate::Image;
use crate::analysis::Analysis;
use crate::color::{ColorMode, LUMA, working_planes};
use crate::curve_net::{BANDS, Frame, Measures, SKY_FROM};

/// The two colour differences of a colour image (R − Y, B − Y). `None` for a mono image, also
/// when it is saved as three identical channels (a grey picture has no colour noise to find).
pub fn chroma_planes(img: &Image) -> Option<[Vec<f32>; 2]> {
    let [r, g, b] = img.channels() else { return None };
    let top = g.iter().copied().fold(0.0f32, f32::max).max(1e-12);
    let grey = r.iter().zip(g).zip(b).step_by(7).all(|((r, g), b)| (r - g).abs() <= 2e-3 * top && (b - g).abs() <= 2e-3 * top);
    if grey {
        return None;
    }
    let y: Vec<f32> = (0..r.len()).map(|i| LUMA[0] * r[i] + LUMA[1] * g[i] + LUMA[2] * b[i]).collect();
    Some([r.iter().zip(&y).map(|(c, y)| c - y).collect(), b.iter().zip(&y).map(|(c, y)| c - y).collect()])
}

/// Everything the estimator is shown.
pub struct NoiseMeasures {
    /// The luminance's, as the curve network measures it.
    pub lum: Measures,
    /// Colour differences, both added: inside the subject, and in the sky (when the sky is
    /// usable). `None` for a mono image.
    pub chroma_subject: Option<Vec<f32>>,
    pub chroma_sky: Option<Vec<f32>>,
}

/// Log10 of the summed power of two planes' bands (each already log10).
fn add_log(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| (10f32.powf(*x) + 10f32.powf(*y)).log10()).collect()
}

/// Measure `img` (its analysis `a`), with the frame it was measured in.
pub fn measure(img: &Image, a: &Analysis) -> Option<(NoiseMeasures, Frame)> {
    let lum = crate::curve_net::measure(img, a)?;
    let f = crate::curve_net::frame(img, a)?;
    let (chroma_subject, chroma_sky) = match chroma_planes(&f.crop) {
        Some([cr, cb]) => {
            let inside = add_log(&f.power(&cr, &f.inside), &f.power(&cb, &f.inside));
            let sky = lum.sky.is_some().then(|| add_log(&f.power(&cr, &f.sky), &f.power(&cb, &f.sky)));
            (Some(inside), sky)
        }
        None => (None, None),
    };
    Some((NoiseMeasures { lum, chroma_subject, chroma_sky }, f))
}

/// The estimator's inputs: the curve network's, then the colour's (zeros and a flag when
/// there is no colour or no sky: missing, never "no noise").
pub fn features(m: &NoiseMeasures) -> Vec<f32> {
    let mut f = crate::curve_net::features(&m.lum);
    match &m.chroma_subject {
        Some(c) => {
            f.extend_from_slice(&c[1..]);
            f.push(1.0);
            match &m.chroma_sky {
                Some(s) => {
                    f.extend_from_slice(&s[SKY_FROM..]);
                    f.extend(c[SKY_FROM..].iter().zip(&s[SKY_FROM..]).map(|(a, b)| a - b));
                    f.push(1.0);
                }
                None => {
                    f.extend_from_slice(&c[SKY_FROM..]);
                    f.extend(std::iter::repeat_n(0.0, BANDS - SKY_FROM));
                    f.push(0.0);
                }
            }
        }
        None => {
            f.extend(std::iter::repeat_n(0.0, BANDS - 1 + 1 + 2 * (BANDS - SKY_FROM) + 1));
        }
    }
    f
}

/// The noise of `x` against the clean `truth` (same frame, same size), band by band over `px`:
/// what of each band the truth's same band can't explain (its energy less the part in step
/// with the truth), log10 over the mean squared. What the training measures; `mean` is the
/// frame's.
pub fn incoherent(x: &[f32], truth: &[f32], side: usize, px: &[usize], mean: f64) -> Vec<f32> {
    let (bx, bt) = (crate::curve_net::bands(x, side), crate::curve_net::bands(truth, side));
    bx.iter()
        .zip(&bt)
        .map(|(a, t)| {
            let (mut xx, mut xt, mut tt) = (0.0f64, 0.0f64, 0.0f64);
            for &i in px {
                let (u, v) = (a[i] as f64, t[i] as f64);
                xx += u * u;
                xt += u * v;
                tt += v * v;
            }
            let rest = (xx - xt * xt / tt.max(1e-30)).max(0.0) / px.len().max(1) as f64;
            (rest / mean.powi(2)).max(1e-14).log10() as f32
        })
        .collect()
}

/// The luminance and colour-difference planes of an image (colour `None` for mono).
pub fn planes(img: &Image) -> (Vec<f32>, Option<[Vec<f32>; 2]>) {
    (working_planes(img, ColorMode::Luminance).swap_remove(0), chroma_planes(img))
}

/// The noise estimator: an average of small networks, with a calibration on real stacks.
pub struct NoiseModel {
    pub inputs: usize,
    pub hidden: usize,
    /// Grain bands 2‥=23, then colour bands 2‥=23.
    pub outputs: usize,
    pub mean: &'static [f32],
    pub scale: &'static [f32],
    pub z_min: &'static [f32],
    pub z_max: &'static [f32],
    pub nets: &'static [crate::curve_net::Net],
    /// Added to each output (log10): the gap measured between the networks and the real noise
    /// of half stacks.
    pub calibration: &'static [f32],
}

/// First band the estimator answers for (band 0 is the mean, band 1 holds almost no
/// frequencies at planetary sizes).
pub const FIRST_BAND: usize = 2;

impl NoiseModel {
    /// The estimated noise (log10 of its power over the subject's mean squared) of each band
    /// from [`FIRST_BAND`]: grain, then colour.
    pub fn predict(&self, x: &[f32]) -> Option<(Vec<f32>, Vec<f32>)> {
        if x.len() != self.inputs {
            return None;
        }
        let z: Vec<f32> = x.iter().enumerate().map(|(i, &v)| ((v - self.mean[i]) / self.scale[i]).clamp(self.z_min[i] - 0.5, self.z_max[i] + 0.5)).collect();
        let n = self.nets.len() as f32;
        let mut out = vec![0.0f32; self.outputs];
        for net in self.nets {
            let h: Vec<f32> = (0..self.hidden).map(|j| (net.b1[j] + z.iter().enumerate().map(|(i, &v)| v * net.w1[i * self.hidden + j]).sum::<f32>()).tanh()).collect();
            for (k, o) in out.iter_mut().enumerate() {
                *o += (net.b2[k] + h.iter().enumerate().map(|(j, &v)| v * net.w2[j * self.outputs + k]).sum::<f32>()) / n;
            }
        }
        out.iter_mut().zip(self.calibration).for_each(|(o, c)| *o += c);
        let half = self.outputs / 2;
        Some((out[..half].to_vec(), out[half..].to_vec()))
    }
}

/// The estimator for a subject, if there is one (Jupiter for now; the app uses it for any
/// planet's disk, as the closest it has).
pub fn model() -> &'static NoiseModel {
    &crate::noise_net_data::JUPITER
}

/// Multiply each frequency band of `plane` (whole image) by its gain (bands as
/// [`crate::curve_net::band_of`]).
fn filter_bands(plane: &[f32], w: usize, h: usize, gains: &[f32]) -> Vec<f32> {
    let plan = crate::fft::FftConvolver::plan(w, h);
    let mut spec = plan.forward(plane);
    // Transposed layout: index = kx·h + ky, kx ∈ 0..=w/2.
    for (idx, c) in spec.iter_mut().enumerate() {
        let (kx, ky) = (idx / h, idx % h);
        let fy = if ky <= h / 2 { ky as f32 } else { ky as f32 - h as f32 } / h as f32;
        let g = gains[crate::curve_net::band_of((kx as f32 / w as f32).hypot(fy))];
        c.re *= g;
        c.im *= g;
    }
    plan.inverse(spec)
}

/// Noise reduction guided by the estimator: each frequency band of the brightness and of the
/// colour keeps the share of it that is not noise (a Wiener gain from the band's measured power
/// inside the subject and the estimated noise power). `strength` multiplies the estimated noise
/// (1 = as estimated, 0 = nothing removed). `None` without a disk to measure on.
pub fn guided_denoise(img: &Image, strength: f32) -> Option<Image> {
    let a = crate::analysis::analyze(img);
    let (m, _) = measure(img, &a)?;
    let (noise_lum, noise_col) = model().predict(&features(&m))?;
    let gains = |power: &[f32], noise: &[f32]| -> Vec<f32> {
        let mut g = vec![1.0f32; BANDS];
        for (j, b) in (FIRST_BAND..BANDS).enumerate() {
            let ratio = 10f32.powf(noise[j] - power[b]);
            g[b] = (1.0 - strength.max(0.0) * ratio).clamp(0.0, 1.0);
        }
        g
    };
    let (w, h) = (img.width(), img.height());
    let (y, chroma) = planes(img);
    let y2 = filter_bands(&y, w, h, &gains(&m.lum.subject, &noise_lum));
    let (Some([cr, cb]), Some(cs)) = (chroma, m.chroma_subject.as_ref()) else {
        return Some(if img.is_mono() { Image::new(w, h, vec![y2], img.source_bit_depth) } else { crate::color::merge_planes(img, ColorMode::Luminance, &[y], vec![y2]) });
    };
    let gc = gains(cs, &noise_col);
    let (cr2, cb2) = (filter_bands(&cr, w, h, &gc), filter_bands(&cb, w, h, &gc));
    let r: Vec<f32> = y2.iter().zip(&cr2).map(|(y, c)| y + c).collect();
    let b: Vec<f32> = y2.iter().zip(&cb2).map(|(y, c)| y + c).collect();
    let g: Vec<f32> = (0..w * h).map(|i| (y2[i] - LUMA[0] * r[i] - LUMA[2] * b[i]) / LUMA[1]).collect();
    Some(Image::new(w, h, vec![r, g, b], img.source_bit_depth))
}
