//! The sharpening curve predicted by a small network, one per subject (Jupiter for now).
//!
//! Trained on simulated stacks (`ps-train`: `sim_stack` simulates hundreds of frames through
//! seeing and a sensor and stacks them for real, `sim_eval` measures them, `curve_train.py`
//! trains), the network looks at what the image itself shows and answers with the gain each
//! size of detail should get: the curve that brings the stack closest to the sharp original.
//! What it looks at:
//! - the power of 24 frequency bands inside the subject (log, over the subject's mean level
//!   squared, so the image's brightness doesn't matter);
//! - the same bands in the sky around it, where there is only noise: they say where the
//!   detail sinks into it. A sky made black by an earlier step (clipped, no noise left) says
//!   nothing and is flagged as missing, never read as "no noise";
//! - the analysis' cut-off, limb blur, noise over the subject's level and SNR, and the disk's
//!   size.
//!
//! Nothing it looks at depends on the image's brightness scale (exposure, stretch, how the
//! stacker normalised its output): in the simulated stacks the absolute brightness happened to
//! tell how much the seeing had flattened the highlights, a cue real images don't share.
//!
//! Its answer becomes a shape for [`crate::smart`] (gain relative to its peak, by frequency over
//! the peak's) with a strength and a position, so the "Nitidezza" and "Forma della risposta"
//! sliders act on it as on every other curve, and the pixels are computed as always.

use crate::Image;
use crate::analysis::Analysis;
use crate::color::{ColorMode, working_planes};
use crate::fft::FftConvolver;

/// Log-spaced bands between these frequencies (cycles/px); band 0 is everything coarser, band
/// `BINS + 1` the corners past 0.5.
pub const BINS: usize = 22;
pub const F_LO: f32 = 0.012;
pub const F_HI: f32 = 0.5;
pub const BANDS: usize = BINS + 2;
/// Bands from here up are measured in the sky (coarser ones there are the limb's spill).
pub const SKY_FROM: usize = 8;

/// Band of a frequency.
pub fn band_of(f: f32) -> usize {
    if f < F_LO {
        0
    } else if f > F_HI {
        BINS + 1
    } else {
        let (lo, hi) = (F_LO.ln(), F_HI.ln());
        1 + (((f.ln() - lo) / (hi - lo)) * BINS as f32).clamp(0.0, BINS as f32 - 1.0) as usize
    }
}

/// Centre frequency of band `k` (1‥=BINS; band 0 and the corners get nominal ones).
pub fn band_centre(k: usize) -> f32 {
    let (lo, hi) = (F_LO.ln(), F_HI.ln());
    match k {
        0 => 0.5 * F_LO,
        k if k > BINS => 0.6,
        k => (lo + (k as f32 - 0.5) / BINS as f32 * (hi - lo)).exp(),
    }
}

/// The plane split into its frequency bands (they add up to it). Square planes.
pub fn bands(plane: &[f32], side: usize) -> Vec<Vec<f32>> {
    let plan = FftConvolver::plan(side, side);
    let spec = plan.forward(plane);
    let which: Vec<usize> = (0..spec.len())
        .map(|idx| {
            let (kx, ky) = (idx / side, idx % side);
            let fy = (if ky <= side / 2 { ky as f32 } else { ky as f32 - side as f32 }) / side as f32;
            band_of((kx as f32 / side as f32).hypot(fy))
        })
        .collect();
    (0..BANDS)
        .map(|b| {
            let masked = spec.iter().zip(&which).map(|(c, &k)| if k == b { *c } else { Default::default() }).collect();
            plan.inverse(masked)
        })
        .collect()
}

/// What the network is shown, measured on the image alone.
#[derive(Clone, Debug, PartialEq)]
pub struct Measures {
    /// Log10 power of each band inside the subject, over its mean level squared.
    pub subject: Vec<f32>,
    /// The same in the sky, when there is a sky with its noise (not clipped to black).
    pub sky: Option<Vec<f32>>,
    /// Why the sky was left out, if it was.
    pub sky_note: Option<&'static str>,
    pub cutoff: f32,
    pub edge_sigma: Option<f32>,
    pub disk_px: f32,
    /// The analysis' noise over the subject's mean level, and its SNR (both independent of the
    /// image's brightness).
    pub noise_rel: f32,
    pub snr: f32,
}

/// Side of the square measured: the disk with a margin of sky, framed exactly as the simulated
/// stacks the network learned on (1.35 diameters, at least 192 px, a fast FFT size), within
/// the image.
fn measured_side(w: usize, h: usize, diameter: f32) -> usize {
    crate::fft::next_fast_size(((1.35 * diameter) as usize).max(192)).min(w).min(h)
}

/// Measure `img` (its analysis `a`). `None` without a disk to measure on.
pub fn measure(img: &Image, a: &Analysis) -> Option<Measures> {
    let disk = a.disk?;
    let diameter = 2.0 * disk.radius;
    let (w, h) = (img.width(), img.height());
    let side = measured_side(w, h, diameter);
    if side < 64 {
        return None;
    }
    let x0 = ((disk.cx - side as f32 / 2.0).round().max(0.0) as usize).min(w - side);
    let y0 = ((disk.cy - side as f32 / 2.0).round().max(0.0) as usize).min(h - side);
    let crop = img.crop(x0, y0, side, side);
    let lum = working_planes(&crop, ColorMode::Luminance).swap_remove(0);
    let (cx, cy) = (disk.cx - x0 as f32, disk.cy - y0 as f32);
    // Inside the subject, a few pixels from its edges (the limb against the sky would weigh on
    // every band).
    let border = 12;
    let inside_mask = crate::mask::gaussian_approx(&crate::mask::subject_and_moons(&crop), side, side, 4.0);
    let inside: Vec<usize> = (0..side * side).filter(|&i| inside_mask[i] > 0.97 && (border..side - border).contains(&(i % side)) && (border..side - border).contains(&(i / side))).collect();
    if inside.len() < 500 {
        return None;
    }
    let mean = inside.iter().map(|&i| lum[i] as f64).sum::<f64>() / inside.len() as f64;
    if mean <= 0.0 {
        return None;
    }
    let b = bands(&lum, side);
    let power = |px: &[usize]| -> Vec<f32> { b.iter().map(|band| ((px.iter().map(|&i| (band[i] as f64).powi(2)).sum::<f64>() / px.len() as f64) / mean.powi(2)).max(1e-14).log10() as f32).collect() };
    let subject = power(&inside);
    // The sky: well away from the disk (1.5 radii) and from the crop's edges.
    let sky_px: Vec<usize> = (0..side * side).filter(|&i| ((i % side) as f32 - cx).hypot((i / side) as f32 - cy) > 0.75 * diameter && (16..side - 16).contains(&(i % side)) && (16..side - 16).contains(&(i / side))).collect();
    let (sky, sky_note) = if sky_px.len() < 500 {
        (None, Some("poco cielo attorno al disco"))
    } else if let Some(note) = sky_without_noise(&lum, &sky_px, mean) {
        (None, Some(note))
    } else {
        (Some(power(&sky_px)), None)
    };
    Some(Measures {
        subject,
        sky,
        sky_note,
        cutoff: a.spectral.as_ref().map_or(0.0, |s| s.cutoff),
        edge_sigma: a.psf_sigma,
        disk_px: diameter,
        noise_rel: (a.noise as f64 / mean) as f32,
        snr: a.snr,
    })
}

/// Whether the sky has had its noise removed (a black point set above it, a mask, a
/// denoiser): many pixels at the very same lowest value, or almost no variation at all.
fn sky_without_noise(lum: &[f32], sky: &[usize], subject_mean: f64) -> Option<&'static str> {
    let mut v: Vec<f32> = sky.iter().map(|&i| lum[i]).collect();
    v.sort_by(f32::total_cmp);
    let low = v[0];
    let tiny = 1e-6 * subject_mean as f32;
    let at_floor = v.iter().take_while(|&&x| x <= low + tiny).count() as f32 / v.len() as f32;
    if at_floor > 0.2 {
        return Some("cielo nero senza rumore (tagliato a zero)");
    }
    // Spread between the 10th and 90th percentiles against the subject's level: a stack's sky
    // always keeps some noise.
    let spread = v[v.len() * 9 / 10] - v[v.len() / 10];
    if spread <= 1e-5 * subject_mean as f32 {
        return Some("cielo uniforme senza rumore");
    }
    None
}

/// The network's inputs from the measures (the same function builds the training set).
pub fn features(m: &Measures) -> Vec<f32> {
    let mut f = m.subject.clone();
    match &m.sky {
        Some(sky) => {
            f.extend_from_slice(&sky[SKY_FROM..]);
            f.extend(m.subject[SKY_FROM..].iter().zip(&sky[SKY_FROM..]).map(|(s, n)| s - n));
            f.push(1.0);
        }
        None => {
            // Missing, not zero: the subject's own power stands in, and the flag says so.
            f.extend_from_slice(&m.subject[SKY_FROM..]);
            f.extend(std::iter::repeat_n(0.0, BANDS - SKY_FROM));
            f.push(0.0);
        }
    }
    f.push(m.cutoff);
    f.push(if m.cutoff > 0.0 { 1.0 } else { 0.0 });
    f.push(m.edge_sigma.unwrap_or(5.0));
    f.push(if m.edge_sigma.is_some() { 1.0 } else { 0.0 });
    f.push(m.disk_px.max(1.0).log10());
    f.push(m.noise_rel.max(1e-7).log10());
    f.push(m.snr.max(1.0).log10());
    f
}

/// One trained network: standardisation, a hidden tanh layer, log gains of bands 1‥.
pub struct Net {
    pub w1: &'static [f32],
    pub b1: &'static [f32],
    pub w2: &'static [f32],
    pub b2: &'static [f32],
}

/// The networks of one subject (an average of a few trained from different starts).
pub struct Model {
    pub inputs: usize,
    pub hidden: usize,
    pub mean: &'static [f32],
    pub scale: &'static [f32],
    /// Standardised inputs are held within the range seen in training, widened by a margin.
    pub z_min: &'static [f32],
    pub z_max: &'static [f32],
    pub nets: &'static [Net],
    pub note: &'static str,
}

/// What a model answers for an image.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    /// Gain of each band, band 0 = 1.
    pub gains: Vec<f32>,
    /// Disagreement between the networks: standard deviation of their log gain at the peak.
    pub spread: f32,
}

impl Model {
    pub fn predict(&self, x: &[f32]) -> Option<Prediction> {
        if x.len() != self.inputs {
            return None;
        }
        let z: Vec<f32> = x.iter().enumerate().map(|(i, &v)| ((v - self.mean[i]) / self.scale[i]).clamp(self.z_min[i] - 0.5, self.z_max[i] + 0.5)).collect();
        let outs: Vec<Vec<f32>> = self
            .nets
            .iter()
            .map(|n| {
                let h: Vec<f32> = (0..self.hidden).map(|j| (n.b1[j] + z.iter().enumerate().map(|(i, &v)| v * n.w1[i * self.hidden + j]).sum::<f32>()).tanh()).collect();
                (0..BANDS - 1).map(|k| (n.b2[k] + h.iter().enumerate().map(|(j, &v)| v * n.w2[j * (BANDS - 1) + k]).sum::<f32>()).clamp(-3.0, 5.0)).collect()
            })
            .collect();
        let n = outs.len() as f32;
        let mean: Vec<f32> = (0..BANDS - 1).map(|k| outs.iter().map(|o| o[k]).sum::<f32>() / n).collect();
        let peak = (0..BINS).max_by(|&a, &b| mean[a].total_cmp(&mean[b])).unwrap_or(0);
        let spread = (outs.iter().map(|o| (o[peak] - mean[peak]).powi(2)).sum::<f32>() / n).sqrt();
        let mut gains = vec![1.0];
        gains.extend(mean.iter().map(|v| v.exp()));
        Some(Prediction { gains, spread })
    }
}

/// The model for a subject, if there is one.
pub fn model_for(body: crate::capture::Body) -> Option<&'static Model> {
    match body {
        crate::capture::Body::Jupiter => Some(&crate::curve_net_data::JUPITER),
        _ => None,
    }
}

/// A predicted curve as the smart sharpening takes it: (cut-off, strength, shape table).
/// The cut-off is placed at the curve's peak (with the response's position at 100 %), the
/// strength is its height, the table its shape (log gain over the peak's, by frequency over
/// the peak's; below 0 where the network lowers the grain).
pub fn smart_curve(p: &Prediction) -> (f32, f32, Vec<(f32, f32)>) {
    let peak = (1..=BINS).max_by(|&a, &b| p.gains[a].total_cmp(&p.gains[b])).unwrap_or(BINS / 2);
    let fp = band_centre(peak);
    let top = p.gains[peak].max(1.02);
    let ln_top = top.ln();
    let mut table = vec![(band_centre(1) / fp * 0.8, 0.0)];
    table.extend((1..=BINS).map(|k| (band_centre(k) / fp, (p.gains[k].max(0.05).ln() / ln_top).clamp(-2.0, 1.0))));
    // Past the last band the table ends: gain 1 beyond.
    let last = table.last().map_or(2.0, |q| q.0);
    table.push((last * 1.15, 0.0));
    let strength = ((top - 1.0) / (crate::smart::PEAK_AT_FULL - 1.0)).clamp(0.0, crate::smart::MAX_STRENGTH);
    (fp, strength, table)
}

/// The curve a subject's network suggests for an image, as the smart sharpening takes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    pub body: crate::capture::Body,
    /// Where its peak falls, cycles/px.
    pub cutoff: f32,
    pub strength: f32,
    /// Log-gain over the peak's, by frequency over the peak's.
    pub table: Vec<(f32, f32)>,
    /// How much the networks disagree (log gain at the peak; above ~0.3 the answer is shaky).
    pub spread: f32,
    /// Why the sky was not used, if it wasn't.
    pub sky_note: Option<&'static str>,
}

/// The curve the subject's network suggests for `img` (its analysis `a`), or why there is none.
pub fn suggestion(img: &Image, a: &Analysis, body: crate::capture::Body) -> Result<Suggestion, &'static str> {
    let model = model_for(body).ok_or("nessuna rete per questo soggetto")?;
    let m = measure(img, a).ok_or("serve un disco intero da misurare")?;
    let p = model.predict(&features(&m)).ok_or("misure incompatibili con la rete")?;
    let (cutoff, strength, table) = smart_curve(&p);
    Ok(Suggestion { body, cutoff, strength, table, spread: p.spread, sky_note: m.sky_note })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_cover_the_spectrum_once() {
        assert_eq!(band_of(0.001), 0);
        assert_eq!(band_of(0.6), BINS + 1);
        for k in 1..=BINS {
            assert_eq!(band_of(band_centre(k)), k);
        }
        let plane: Vec<f32> = (0..64 * 64).map(|i| ((i * 7919) % 101) as f32).collect();
        let b = bands(&plane, 64);
        for i in [0, 100, 2000, 4095] {
            let s: f32 = b.iter().map(|band| band[i]).sum();
            assert!((s - plane[i]).abs() < 1e-2, "{s} vs {}", plane[i]);
        }
    }

    #[test]
    fn black_sky_is_noticed() {
        let lum: Vec<f32> = (0..1000).map(|i| if i % 3 == 0 { 0.0 } else { 0.0 }).collect();
        let px: Vec<usize> = (0..1000).collect();
        assert!(sky_without_noise(&lum, &px, 0.5).is_some());
        let noisy: Vec<f32> = (0..1000).map(|i| 0.001 * ((i * 7919) % 13) as f32).collect();
        assert!(sky_without_noise(&noisy, &px, 0.5).is_none());
    }

    /// The exported networks answer as in training: `CURVE_NET_CHECK` names the file
    /// `curve_train.py --export` writes beside the evaluation (inputs and expected gains).
    #[test]
    fn exported_networks_match_training() {
        let Ok(path) = std::env::var("CURVE_NET_CHECK") else { return };
        let text = std::fs::read_to_string(path).unwrap();
        // A tiny reader for [{"x": [...], "guadagni": [...]}, ...].
        let nums = |key: &str, from: usize| -> (Vec<f32>, usize) {
            let start = text[from..].find(key).unwrap() + from;
            let open = text[start..].find('[').unwrap() + start;
            let close = text[open..].find(']').unwrap() + open;
            (text[open + 1..close].split(',').map(|v| v.trim().parse().unwrap()).collect(), close)
        };
        let (mut at, mut checked) = (0, 0);
        while text[at..].contains("\"x\"") {
            let (x, a1) = nums("\"x\"", at);
            let (g, a2) = nums("\"guadagni\"", a1);
            let p = crate::curve_net_data::JUPITER.predict(&x).unwrap();
            for (a, b) in p.gains.iter().zip(&g) {
                assert!((a / b - 1.0).abs() < 1e-3, "{a} vs {b}");
            }
            at = a2;
            checked += 1;
        }
        assert!(checked > 0);
    }

    #[test]
    fn curve_peaks_at_one() {
        let mut gains = vec![1.0; BANDS];
        for (k, g) in gains.iter_mut().enumerate().take(BINS + 1).skip(1) {
            *g = 1.0 + 9.0 * (-((k as f32 - 15.0) / 4.0).powi(2)).exp();
        }
        let (fp, strength, table) = smart_curve(&Prediction { gains, spread: 0.0 });
        assert!((fp - band_centre(15)).abs() < 1e-6);
        assert!((strength - 9.0 / 11.0).abs() < 1e-3);
        let top = table.iter().copied().fold((0.0, f32::MIN), |a, q| if q.1 > a.1 { q } else { a });
        assert!((top.0 - 1.0).abs() < 1e-5 && (top.1 - 1.0).abs() < 1e-5);
        assert!(table.windows(2).all(|w| w[1].0 > w[0].0));
    }
}
