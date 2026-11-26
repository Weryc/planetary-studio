//! Moon close-ups (crater fields, wide stretches of the surface): what the learned sharpening and
//! noise estimate see there.
//!
//! There is no disk to frame: the surface fills the picture, with at most some sky past the limb
//! and black shadows near the terminator. The measures are taken on a square in the middle of the
//! image, split into the lit surface (where the detail is) and the dark part (sky and deep
//! shadows, where there is only noise), with the bands of [`crate::curve_net`].
//!
//! The networks (`moon_data.rs`, from `ps-train/scripts/moon_train.py`) were trained on
//! simulated stacks of LROC close-ups, some with the limb and black sky, some soft as many real
//! stacks are. Applied as the app applies them, against the app's curve for a surface: 0.91 of
//! the way from the raw to the best possible curve against 0.81; the noise estimate spares the
//! medium detail where the learned reduction ate it (10 stacks out of 136 against 70).

use crate::Image;
use crate::analysis::Analysis;
use crate::color::{ColorMode, working_planes};
use crate::curve_net::{BANDS, SKY_FROM};

/// Largest side of the measured square (px).
pub const MAX_SIDE: usize = 768;
/// Fewer lit pixels than this and nothing is measured.
const MIN_LIT: usize = 2000;
/// Fewer dark pixels than this and the dark part is not measured.
pub const MIN_DARK: usize = 500;

/// The square measured on a close-up and its two pixel sets.
pub struct Frame {
    pub crop: Image,
    pub x0: usize,
    pub y0: usize,
    pub side: usize,
    /// The lit surface (above 15 % of the bright level, a few pixels in from the square's edges)
    /// and the dark part (below 6 %, at least a few pixels from anything lit).
    pub lit: Vec<usize>,
    pub dark: Vec<usize>,
    /// The lit surface's mean luminance: every power is relative to its square.
    pub mean: f64,
    /// The lit surface's spread over its mean (how contrasted the field is).
    pub contrast: f32,
}

impl Frame {
    /// Log10 power of each band of `plane` over `px`, over the lit mean squared.
    pub fn power(&self, plane: &[f32], px: &[usize]) -> Vec<f32> {
        crate::curve_net::bands(plane, self.side).iter().map(|band| ((px.iter().map(|&i| (band[i] as f64).powi(2)).sum::<f64>() / px.len().max(1) as f64) / self.mean.powi(2)).max(1e-14).log10() as f32).collect()
    }
}

/// Frame a close-up: the largest fast square (up to [`MAX_SIDE`]) in the middle of the image.
pub fn frame(img: &Image) -> Option<Frame> {
    let (w, h) = (img.width(), img.height());
    let mut side = w.min(h).min(MAX_SIDE);
    while side > 64 && crate::fft::next_fast_size(side) != side {
        side -= 1;
    }
    if side < 128 {
        return None;
    }
    let (x0, y0) = ((w - side) / 2, (h - side) / 2);
    let crop = img.crop(x0, y0, side, side);
    let lum = working_planes(&crop, ColorMode::Luminance).swap_remove(0);
    let smooth = crate::mask::gaussian_approx(&lum, side, side, 2.0);
    let mut sample: Vec<f32> = smooth.iter().step_by(7).copied().collect();
    sample.sort_by(f32::total_cmp);
    let top = sample[sample.len() * 99 / 100].max(1e-9);
    let within = |i: usize, b: usize| (b..side - b).contains(&(i % side)) && (b..side - b).contains(&(i / side));
    let lit_mask: Vec<f32> = smooth.iter().map(|&v| if v > 0.15 * top { 1.0 } else { 0.0 }).collect();
    let near_lit = crate::mask::gaussian_approx(&lit_mask, side, side, 3.0);
    let lit: Vec<usize> = (0..side * side).filter(|&i| within(i, 8) && lit_mask[i] > 0.5).collect();
    let dark: Vec<usize> = (0..side * side).filter(|&i| within(i, 12) && smooth[i] < 0.06 * top && near_lit[i] < 0.02).collect();
    if lit.len() < MIN_LIT {
        return None;
    }
    let n = lit.len() as f64;
    let mean = lit.iter().map(|&i| lum[i] as f64).sum::<f64>() / n;
    if mean <= 0.0 {
        return None;
    }
    let var = lit.iter().map(|&i| (lum[i] as f64 - mean).powi(2)).sum::<f64>() / n;
    Some(Frame { crop, x0, y0, side, lit, dark, mean, contrast: (var.sqrt() / mean) as f32 })
}

/// What the sharpening network sees on a close-up.
pub struct Measures {
    pub lit: Vec<f32>,
    pub dark: Option<Vec<f32>>,
    pub dark_note: Option<&'static str>,
    pub dark_share: f32,
    pub contrast: f32,
    pub cutoff: f32,
    pub edge_sigma: Option<f32>,
    pub noise_rel: f32,
    pub snr: f32,
}

/// Measure a close-up (its analysis `a`), with the frame it was measured in.
pub fn measure(img: &Image, a: &Analysis) -> Option<(Measures, Frame)> {
    let f = frame(img)?;
    let lum = working_planes(&f.crop, ColorMode::Luminance).swap_remove(0);
    let probe = crate::curve_net::Frame { crop: f.crop.clone(), x0: f.x0, y0: f.y0, side: f.side, diameter: f.side as f32, inside: f.lit.clone(), sky: f.dark.clone(), mean: f.mean };
    let dark_note = if f.dark.len() < MIN_DARK { Some("poco cielo o ombra") } else { probe.sky_problem(&lum) };
    let m = Measures {
        lit: f.power(&lum, &f.lit),
        dark: dark_note.is_none().then(|| f.power(&lum, &f.dark)),
        dark_note,
        dark_share: f.dark.len() as f32 / (f.side * f.side) as f32,
        contrast: f.contrast,
        cutoff: a.spectral.as_ref().map_or(0.0, |s| s.cutoff),
        edge_sigma: a.psf_sigma,
        noise_rel: (a.noise as f64 / f.mean) as f32,
        snr: a.snr,
    };
    Some((m, f))
}

/// The sharpening network's inputs (the same function builds the training set).
pub fn features(m: &Measures) -> Vec<f32> {
    let mut f = m.lit.clone();
    match &m.dark {
        Some(d) => {
            f.extend_from_slice(&d[SKY_FROM..]);
            f.extend(m.lit[SKY_FROM..].iter().zip(&d[SKY_FROM..]).map(|(s, n)| s - n));
            f.push(1.0);
        }
        None => {
            f.extend_from_slice(&m.lit[SKY_FROM..]);
            f.extend(std::iter::repeat_n(0.0, BANDS - SKY_FROM));
            f.push(0.0);
        }
    }
    f.push(m.cutoff);
    f.push(if m.cutoff > 0.0 { 1.0 } else { 0.0 });
    f.push(m.edge_sigma.unwrap_or(5.0));
    f.push(if m.edge_sigma.is_some() { 1.0 } else { 0.0 });
    f.push(m.dark_share.max(1e-3).log10());
    f.push(m.contrast.max(1e-3).log10());
    f.push(m.noise_rel.max(1e-7).log10());
    f.push(m.snr.max(1.0).log10());
    f
}

/// Number of sharpening inputs.
pub const INPUTS: usize = BANDS + 2 * (BANDS - SKY_FROM) + 1 + 8;

/// The noise estimator's inputs: the sharpening's, then the colour differences (R − Y plus
/// B − Y) of the lit surface and of the dark part, and whether there is colour.
pub fn noise_features(m: &Measures, f: &Frame) -> Vec<f32> {
    let mut x = features(m);
    match crate::noise_net::chroma_planes(&f.crop) {
        Some([cr, cb]) => {
            let both = |px: &[usize]| -> Vec<f32> { f.power(&cr, px).iter().zip(f.power(&cb, px)).map(|(a, b)| (10f32.powf(*a) + 10f32.powf(b)).log10()).collect() };
            let p = both(&f.lit);
            x.extend_from_slice(&p[1..]);
            match m.dark.is_some() {
                true => x.extend_from_slice(&both(&f.dark)[SKY_FROM..]),
                false => x.extend_from_slice(&p[SKY_FROM..]),
            }
            x.push(1.0);
        }
        None => x.extend(std::iter::repeat_n(0.0, (BANDS - 1) + (BANDS - SKY_FROM) + 1)),
    }
    x
}

/// Number of noise inputs.
pub const NOISE_INPUTS: usize = INPUTS + (BANDS - 1) + (BANDS - SKY_FROM) + 1;

/// The sharpening curve the Moon's network suggests for a close-up, as the smart sharpening
/// takes it (cut-off, strength, shape), with the networks' disagreement and why the dark part
/// was not used, if it wasn't. A full disk is not a close-up: the network never saw one.
pub fn suggestion(img: &Image, a: &Analysis) -> Result<(crate::saturn::Curve, f32, Option<&'static str>), &'static str> {
    if a.kind == crate::analysis::ImageKind::FullDisk {
        return Err("la rete della Luna è per i primi piani, non per il disco intero");
    }
    let (m, _) = measure(img, a).ok_or("troppa poca superficie illuminata da misurare")?;
    let p = crate::moon_data::SHARPEN.predict(&features(&m)).ok_or("misure incompatibili con la rete")?;
    Ok((crate::curve_net::smart_curve(&p), p.spread, m.dark_note))
}

/// What the guided noise reduction measures on a close-up: the lit surface's band power and
/// the Moon estimator's noise there (see [`crate::noise_net::NoisePlan`]).
pub fn noise_plan(img: &Image, a: &Analysis) -> Option<crate::noise_net::NoisePlan> {
    let (m, f) = measure(img, a)?;
    let (lum, colour) = crate::moon_data::NOISE.predict(&noise_features(&m, &f))?;
    let (y, chroma) = crate::noise_net::planes(&f.crop);
    let chroma_power = chroma.as_ref().map(|[cr, cb]| f.power(cr, &f.lit).iter().zip(f.power(cb, &f.lit)).map(|(a, b)| (10f32.powf(*a) + 10f32.powf(b)).log10()).collect::<Vec<f32>>());
    crate::log::info("rumore", &format!("Luna: immagine {}×{}, riquadro {} px, illuminati {} scuri {}{}", img.width(), img.height(), f.side, f.lit.len(), f.dark.len(), m.dark_note.map_or(String::new(), |n| format!(" ({n})"))));
    // Colour bands the estimator gave up on (no colour in training there) leave colour alone.
    let colour = chroma_power.filter(|_| colour.iter().all(|v| v.is_finite() && *v > -13.0)).map(|c| (c, colour));
    Some(crate::noise_net::NoisePlan { zones: vec![crate::noise_net::ZoneNoise { name: "Luna, superficie".into(), lum: (f.power(&y, &f.lit), lum), colour }], globe: None, sky_level: None })
}
