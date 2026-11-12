//! A small learned model that picks the shape and height of the "Nitidezza intelligente" boost
//! from what the analysis measured on the image. It only chooses settings: the pixels still
//! come from the wavelets, never from the model.
//!
//! The model looks at a handful of numbers ([`features`]): where the detail ends, how the
//! spectrum falls towards it, the signal-to-noise ratio at several sizes of detail, the size of
//! the subject, colour or mono, the kind of subject and the body. It answers with the wanted
//! gain at fixed multiples of the cut-off ([`GRID`]), in log units, from the broad structures
//! (bands, a tenth of the cut-off) to past the finest detail.
//!
//! It was trained on sharp references (Hubble, Cassini, SDO, NASA's LRO-based Moon) brought to
//! amateur sizes and seen through a simulated lucky-imaging seeing: hundreds of frames through
//! random turbulence, the best kept, aligned and stacked, plus stack noise. Training happens
//! outside the app (`ps-io/examples/ml_ref.rs` makes the examples, `ml_train.rs` fits the model
//! and writes its weights); see docs/ARCHITETTURA.md. Being linear, its answer can be explained
//! ([`explain`]): how much each thing it measured pushed the curve up or down.
//!
//! How much halo the seeing left is hard to tell from the image, so besides the main model there
//! is one per kind of night ([`Night`]: calm, average, turbulent), each trained only on that
//! night's examples: three hypotheses the user can compare ([`nights`]). A night model that finds
//! the image unlike its examples has no answer (that hypothesis is unlikely).
//!
//! The curve also goes below 1 past the cut-off: the grain there is cleaned by as much as the
//! truth asked for (down to ×[`MIN_GAIN`]).

use crate::analysis::{Analysis, ImageKind};
use crate::capture::Body;
use serde::{Deserialize, Serialize};
use crate::learned_data as d;

/// Whether the automatic sharpening starts from the learned curve (else from the fixed
/// Standard one, the learned curve being offered in the card's curve menu).
pub const IN_USE: bool = true;

/// Where the response is described: frequency over the cut-off.
pub const GRID: [f32; STEPS] = [0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.4];

/// Number of points of [`GRID`].
pub const STEPS: usize = 14;

/// Number of [`features`].
pub const FEATURES: usize = 26;

/// Names of the features, in order (for the study tools).
pub const FEATURE_NAMES: [&str; FEATURES] = [
    "ln_cutoff", "slope", "sigma_x_cutoff", "noise_blur", "misfit", "snr_0.5", "snr_0.7", "snr_1.0", "snr_1.4", "snr_2.0", "pow_0.25", "pow_1.0", "pow_1.4", "pow_2.0", "ln_elements", "ln_snr",
    "colour", "planet", "saturn", "full_disk", "surface", "jupiter", "saturn_body", "mars", "moon", "sun",
];

/// What the model looks at, from the analysis alone (the body is the one the pixels suggest),
/// so the app and the training see the same numbers. `None` when the spectrum couldn't be
/// fitted (the model then has nothing to go on).
pub fn features(analysis: &Analysis) -> Option<[f32; FEATURES]> {
    let s = analysis.spectral?;
    let c = s.cutoff.clamp(0.01, 0.5);
    let at = |r: f32| (r * c).min(0.5);
    let snr = |r: f32| s.snr(at(r)).max(1e-3).log10().clamp(-3.0, 6.0);
    let power = |f: f32| analysis.power.iter().min_by(|a, b| (a.0 / f).ln().abs().total_cmp(&(b.0 / f).ln().abs())).map_or(1.0, |p| p.1.max(1e-30));
    let reference = power(at(0.5));
    let relative = |r: f32| (power(at(r)) / reference).log10().clamp(-8.0, 2.0);
    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    let body = analysis.planet;
    Some([
        c.ln(),
        s.slope,
        s.sigma * c,
        s.noise_blur,
        s.misfit,
        snr(0.5),
        snr(0.7),
        snr(1.0),
        snr(1.4),
        snr(2.0),
        relative(0.25),
        relative(1.0),
        relative(1.4),
        relative(2.0),
        (analysis.subject_size.max(1.0) * c).ln(),
        analysis.snr.max(1.0).ln(),
        flag(analysis.colour),
        flag(analysis.kind == ImageKind::Planet),
        flag(analysis.kind == ImageKind::Saturn),
        flag(analysis.kind == ImageKind::FullDisk),
        flag(analysis.kind == ImageKind::Surface),
        flag(body == Some(Body::Jupiter)),
        flag(body == Some(Body::Saturn)),
        flag(body == Some(Body::Mars)),
        flag(body == Some(Body::Moon)),
        flag(body == Some(Body::Sun)),
    ])
}

/// Largest gain the model may ask for (ln), as in training.
const MAX_LN: f32 = 3.4;
/// Smallest gain (most cleaning) the curve may ask for past the cut-off.
pub const MIN_GAIN: f32 = 0.3;

/// One model's numbers (written by `ml_train.rs` in `learned_data.rs`).
pub(crate) struct Weights {
    pub mean: [f32; FEATURES],
    pub scale: [f32; FEATURES],
    pub bias: [f32; STEPS],
    pub weights: [[f32; FEATURES]; STEPS],
    pub z_min: [f32; FEATURES],
    pub z_max: [f32; FEATURES],
}

/// The kinds of night the seeing simulation knows (as in training).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Night {
    Calm,
    Average,
    Turbulent,
}

impl Night {
    pub const ALL: [Night; 3] = [Night::Calm, Night::Average, Night::Turbulent];

    fn weights(self) -> &'static Weights {
        &d::NIGHTS[self as usize]
    }
}

/// The model's answer for the image: ln(gain) at [`GRID`]. `None` without a fitted spectrum or
/// when the image is unlike every training example in some respect (the model would be
/// guessing: the fixed curve is safer then).
pub fn predict(analysis: &Analysis) -> Option<[f32; STEPS]> {
    predict_with(&features(analysis)?, &d::ALL)
}

/// The features standardised as model `m` was trained.
fn standard(x: &[f32; FEATURES], m: &Weights) -> [f32; FEATURES] {
    std::array::from_fn(|j| (x[j] - m.mean[j]) * m.scale[j])
}

fn predict_with(x: &[f32; FEATURES], m: &Weights) -> Option<[f32; STEPS]> {
    let z = standard(x, m);
    // A little beyond what training saw is fine; far beyond it is not.
    if z.iter().enumerate().any(|(j, &v)| v < m.z_min[j] - 1.0 || v > m.z_max[j] + 1.0) {
        return None;
    }
    Some(std::array::from_fn(|k| (m.bias[k] + m.weights[k].iter().zip(&z).map(|(w, v)| w * v).sum::<f32>()).clamp(MIN_GAIN.ln(), MAX_LN)))
}

/// The learned boost in the terms of the smart sharpening: its strength (1 = ×12 at the peak)
/// and its shape table (frequency over the cut-off, log-gain over the peak's). `None` when the
/// model has no answer or sees nothing worth sharpening.
pub fn smart_curve(analysis: &Analysis) -> Option<(f32, Vec<(f32, f32)>)> {
    to_smart(predict(analysis)?)
}

/// The learned curve under one hypothesis on the night: (night, strength, shape table).
pub type NightCurve = (Night, f32, Vec<(f32, f32)>);

/// The curve under each hypothesis on the night, for those that have an answer.
pub fn nights(analysis: &Analysis) -> Vec<NightCurve> {
    let Some(x) = features(analysis) else { return Vec::new() };
    Night::ALL.iter().filter_map(|&n| predict_with(&x, n.weights()).and_then(to_smart).map(|(s, t)| (n, s, t))).collect()
}

/// Whether the night hypotheses give curves different enough to be worth comparing: somewhere
/// up to the cut-off two of them differ by more than 35 %.
pub fn nights_differ(curves: &[NightCurve]) -> bool {
    let ln = |s: f32, t: &[(f32, f32)], r: f32| (1.0 + 11.0 * s).ln() * shape_at(t, r);
    curves.len() >= 2
        && GRID.iter().filter(|&&r| r <= 1.0).any(|&r| {
            let v: Vec<f32> = curves.iter().map(|(_, s, t)| ln(*s, t, r)).collect();
            v.iter().copied().fold(f32::MIN, f32::max) - v.iter().copied().fold(f32::MAX, f32::min) > 0.3
        })
}

/// A shape table's value at `r` (the table's own points: exact on [`GRID`]).
fn shape_at(t: &[(f32, f32)], r: f32) -> f32 {
    t.iter().min_by(|a, b| (a.0 / r).ln().abs().total_cmp(&(b.0 / r).ln().abs())).map_or(0.0, |q| q.1)
}

/// ln(gain) at [`GRID`] as a smart-sharpening strength and shape table. Past the peak the
/// shape may go below 0: gains under 1, the grain cleaned (it follows the strength like the
/// boost does).
pub fn to_smart(ln: [f32; STEPS]) -> Option<(f32, Vec<(f32, f32)>)> {
    let ln = ln.map(|v| v.clamp(MIN_GAIN.ln(), MAX_LN));
    let peak = ln.iter().copied().fold(0.0f32, f32::max);
    if peak < 0.05 {
        return None;
    }
    let top = (0..STEPS).max_by(|&a, &b| ln[a].total_cmp(&ln[b])).unwrap_or(0);
    let mut table = vec![(0.05, 0.0)];
    // Before the peak never below 1 (the broad structures are only lifted).
    table.extend(GRID.iter().zip(ln).enumerate().map(|(k, (&r, v))| (r, if k < top { v.max(0.0) } else { v } / peak)));
    match table.last().copied() {
        // Back to 1 a little past the last point, as the fixed curves do.
        Some(q) if q.1 > 0.0 => table.push((2.8, 0.0)),
        // Still cleaning: so up to the finest grain.
        Some(q) => table.push((8.0, q.1)),
        None => {}
    }
    let strength = ((peak.exp() - 1.0) / 11.0).min(crate::smart::MAX_STRENGTH);
    Some((strength, table))
}

/// What the model took into account, grouped as a person would say it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Factor {
    /// Where the detail ends and how soft it is there.
    Cutoff,
    /// Signal against noise, at every size of detail.
    Noise,
    /// How the detail fades from the broad structures to the finest.
    Spectrum,
    /// Size of the subject in resolution elements.
    Size,
    Colour,
    /// Kind of subject and body.
    Subject,
}

impl Factor {
    fn of(feature: usize) -> Factor {
        match feature {
            0 | 2 => Factor::Cutoff,
            3 | 5..=9 | 15 => Factor::Noise,
            1 | 4 | 10..=13 => Factor::Spectrum,
            14 => Factor::Size,
            16 => Factor::Colour,
            _ => Factor::Subject,
        }
    }

    pub const ALL: [Factor; 6] = [Factor::Cutoff, Factor::Noise, Factor::Spectrum, Factor::Size, Factor::Colour, Factor::Subject];
}

/// Why the curve is what it is: the model's answer split, against an average training image,
/// into what each [`Factor`] added (ln units, + = more boost).
#[derive(Clone, Debug, PartialEq)]
pub struct Explanation {
    /// ln(gain) at [`GRID`] for this image and for the average image.
    pub answer: [f32; STEPS],
    pub average: [f32; STEPS],
    /// Per factor: what it added at the peak of the curve and on the broad structures
    /// (up to a third of the cut-off).
    pub factors: Vec<(Factor, f32, f32)>,
}

/// Where the broad structures end on [`GRID`] (inclusive).
const BROAD: usize = 3;

/// The model's answer for the image with its reasons; `None` when it has no answer.
pub fn explain(analysis: &Analysis) -> Option<Explanation> {
    let x = features(analysis)?;
    let answer = predict_with(&x, &d::ALL)?;
    let z = standard(&x, &d::ALL);
    let peak = (0..STEPS).max_by(|&a, &b| answer[a].total_cmp(&answer[b])).unwrap_or(7);
    let factors = Factor::ALL
        .iter()
        .map(|&f| {
            let part = |k: usize| (0..FEATURES).filter(|&j| Factor::of(j) == f).map(|j| d::ALL.weights[k][j] * z[j]).sum::<f32>();
            (f, part(peak), (0..=BROAD).map(part).sum::<f32>() / (BROAD + 1) as f32)
        })
        .collect();
    Some(Explanation { answer, average: d::ALL.bias.map(|v| v.clamp(MIN_GAIN.ln(), MAX_LN)), factors })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For a typical image (every feature at the training average) every model answers with a
    /// boost peaking near the cut-off, which the smart sharpening takes as a curve.
    #[test]
    fn the_model_answers_with_a_usable_curve() {
        for m in std::iter::once(&d::ALL).chain(&d::NIGHTS) {
            let ln = predict_with(&m.mean, m).expect("answer");
            assert!(ln.iter().all(|v| v.is_finite()));
            let top = (0..STEPS).max_by(|&i, &j| ln[i].total_cmp(&ln[j])).unwrap();
            assert!((0.5..=1.5).contains(&GRID[top]), "peak at {}× the cut-off", GRID[top]);
            let (strength, table) = to_smart(ln).expect("curve");
            assert!(strength > 0.0 && table.len() >= 3);
            let p = crate::smart::SmartParams { strength, curve: crate::smart::SmartCurve::Learned, learned: table, cutoff: 0.15, ..Default::default() };
            assert!(p.target(0.15, 0.15) > 1.0);
        }
    }

    /// An image unlike any in training gets no answer: the fixed curve is used instead.
    #[test]
    fn it_declines_what_it_never_saw() {
        let mut x = d::ALL.mean;
        x[0] += 40.0 / d::ALL.scale[0].max(1e-3);
        assert!(predict_with(&x, &d::ALL).is_none());
    }

    /// Past the peak the curve may clean (gain under 1), before it it only lifts.
    #[test]
    fn the_curve_cleans_only_past_its_peak() {
        let mut ln = [0.0f32; STEPS];
        ln[0] = -0.5;
        ln[7] = 2.5;
        ln[12] = -1.0;
        ln[13] = -1.0;
        let (_, table) = to_smart(ln).unwrap();
        assert!(table.iter().take_while(|q| q.0 < 1.0).all(|q| q.1 >= 0.0));
        assert!(table.last().unwrap().1 < 0.0);
    }
}
