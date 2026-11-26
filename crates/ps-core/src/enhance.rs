//! Finishing tools: noise reduction (one method for planets, one for the Moon and Sun, and a
//! learned one for all),
//! contrast (unsharp mask and local contrast) and tone (levels and curve).
//!
//! All of them work on luminance Y and add the change back to R, G and B, so colours keep
//! their hue; chroma noise is handled separately.

use crate::Image;
use crate::adjust::{curve_lut, lookup, smooth_plane};
use crate::color::{ColorMode, LUMA, luma, merge_planes, working_planes};
use crate::mask::gaussian_approx;
use crate::wavelet::{ScaleMode, ThresholdMode, WaveletLayer, WaveletParams, decompose, estimate_noise, recompose};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---- noise reduction ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseMethod {
    /// Multiscale soft thresholding of the finest starlet scales. Planet images have fine,
    /// grainy noise over soft detail: removing small coefficients cleans the grain without
    /// touching belts and festoons, which live on larger scales.
    Planetary,
    /// Edge-preserving (bilateral) smoothing. Lunar and solar surfaces are full of sharp
    /// edges (crater rims, shadows, granulation): they must survive while flat areas are cleaned.
    Lunar,
    /// A small network trained on simulated planetary, lunar and solar stacks
    /// ([`crate::denoise_net`]): it tells noise from detail by what that detail looks like.
    /// Falls back to [`NoiseMethod::Planetary`] if this build has no weights.
    Learned,
    /// Guided by the noise estimator ([`crate::noise_net`]): each size of detail, in brightness
    /// and in colour, keeps the share the estimator says is not noise. Handles colour noise
    /// itself (the colour levels are not applied on top). Falls back to
    /// [`NoiseMethod::Learned`] without a disk to measure on.
    Guided,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseParams {
    pub method: NoiseMethod,
    /// Overall strength, in units of the measured noise (1 = standard).
    pub strength: f32,
    /// Lunar and learned methods: share of fine detail given back after cleaning (0‥1).
    pub detail: f32,
    /// Chroma noise reduction (starlet scales averaged away from the colour, 0 = off).
    pub chroma: u32,
    /// Extra smoothing of the black sky around the subject (0‥1).
    pub sky: f32,
}

impl NoiseParams {
    /// Nothing cleaned: no strength, no colour noise reduction, no sky cleaning.
    pub fn is_neutral(&self) -> bool {
        self.strength <= 0.0 && self.chroma == 0 && self.sky <= 0.0
    }
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self { method: NoiseMethod::Planetary, strength: 1.0, detail: 0.3, chroma: 2, sky: 0.7 }
    }
}

pub fn denoise(img: &Image, p: &NoiseParams) -> Image {
    denoise_planned(img, p, None)
}

/// [`denoise`], the guided method using `plan` (measured on an image `img` sits in, at the
/// offset given) instead of measuring `img` itself: a zoomed region then gets the whole
/// image's estimate.
pub fn denoise_planned(img: &Image, p: &NoiseParams, plan: Option<(&crate::noise_net::NoisePlan, (usize, usize))>) -> Image {
    denoise_measured(img, p, plan, None).0
}

/// What the noise reductions other than the guided one measure on the image: the noise σ, the
/// sky's level and, for the wavelet method, each scale's white noise. A zoomed region uses the
/// whole image's (on a region inside the subject its own would be the subject's detail).
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseLevels {
    pub sigma: f32,
    pub sky_level: f32,
    pub white: Vec<f32>,
}

/// [`denoise_planned`] with given `levels` (the whole image's, for a zoomed region), also
/// returning the levels used (`None` for the guided method, which has its plan).
pub fn denoise_measured(img: &Image, p: &NoiseParams, plan: Option<(&crate::noise_net::NoisePlan, (usize, usize))>, levels: Option<&NoiseLevels>) -> (Image, Option<NoiseLevels>) {
    if p.method == NoiseMethod::Guided {
        let guided = match plan {
            Some((plan, offset)) => Some(crate::noise_net::apply_plan(img, plan, offset, p.strength)),
            None => crate::noise_net::guided_denoise(img, p.strength),
        };
        if let Some(guided) = guided {
            return (finish_guided(img, guided, p, plan.and_then(|(plan, _)| plan.sky_level)), None);
        }
        crate::log::warn("rumore", &format!("rete del rumore non applicabile su {}×{} (nessun disco intero): si usa il metodo appreso", img.width(), img.height()));
        return denoise_measured(img, &NoiseParams { method: NoiseMethod::Learned, ..p.clone() }, None, levels);
    }
    let (w, h) = (img.width(), img.height());
    let y = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let sigma = levels.map_or_else(|| estimate_noise(&y, w, h).max(1e-5), |l| l.sigma);
    let mut white = Vec::new();

    let learned = (p.method == NoiseMethod::Learned).then(crate::denoise_net::model).flatten();
    let mut cleaned = match (p.method, learned) {
        (NoiseMethod::Learned, Some(net)) => {
            let clean = crate::denoise_net::denoise(net, &y, w, h, sigma, p.strength, &|| false).expect("never cancelled");
            clean.par_iter().zip(&y).map(|(&s, &o)| s + p.detail.clamp(0.0, 1.0) * (o - s)).collect()
        }
        (NoiseMethod::Planetary | NoiseMethod::Learned | NoiseMethod::Guided, _) => {
            // Thresholds per scale in units of that scale's noise σ (finest first).
            let k = [2.2, 1.4, 0.7].map(|v| v * p.strength);
            let layers = k.iter().map(|&denoise| WaveletLayer { gain: 0.0, denoise, ..Default::default() }).collect();
            let params = WaveletParams { layers, threshold: ThresholdMode::Soft, scale_mode: ScaleMode::Dyadic, ..Default::default() };
            let mut d = decompose(&y, w, h, &params, &|| false).expect("never cancelled");
            if let Some(l) = levels.filter(|l| l.white.len() == d.white_noise.len()) {
                d.white_noise.clone_from(&l.white);
            }
            white.clone_from(&d.white_noise);
            // The thresholds above were tuned on the white-noise estimate.
            d.noise = d.white_noise.clone();
            recompose(&d, &params)
        }
        (NoiseMethod::Lunar, _) => {
            let smooth = bilateral(&y, w, h, 1.5, 2.5 * sigma * p.strength.max(0.05));
            smooth.par_iter().zip(&y).map(|(&s, &o)| s + p.detail.clamp(0.0, 1.0) * (o - s)).collect()
        }
    };

    let sky_level = levels.map_or_else(|| sky_threshold(&y, sigma), |l| l.sky_level);
    if p.sky > 0.0 {
        // Below the sky threshold, blend towards a strong blur of the cleaned image.
        let blurred = gaussian_approx(&cleaned, w, h, 3.0);
        let weight = sky_weight(&y, w, h, sky_level);
        cleaned.par_iter_mut().zip(&blurred).zip(&weight).for_each(|((c, &b), &m)| *c += p.sky * m * (b - *c));
    }

    let merged = merge_planes(img, ColorMode::Luminance, std::slice::from_ref(&y), vec![cleaned]);
    let out = if p.chroma > 0 && !img.is_mono() { smooth_chroma(&merged, p.chroma) } else { merged };
    (out, Some(NoiseLevels { sigma, sky_level, white }))
}

/// The guided reduction's result with the usual finishing: part of the fine detail given back
/// (`detail`) and the sky smoothed (`sky`), on the luminance; its colour is kept as it is.
/// `sky_level`: the whole image's (see [`crate::noise_net::NoisePlan::sky_level`]), else measured.
fn finish_guided(img: &Image, guided: Image, p: &NoiseParams, sky_level: Option<f32>) -> Image {
    let (w, h) = (img.width(), img.height());
    let y = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let yg = working_planes(&guided, ColorMode::Luminance).swap_remove(0);
    let detail = p.detail.clamp(0.0, 1.0) / p.strength.max(1.0);
    let mut cleaned: Vec<f32> = yg.par_iter().zip(&y).map(|(&s, &o)| s + detail * (o - s)).collect();
    if p.sky > 0.0 {
        let sky_level = sky_level.unwrap_or_else(|| sky_threshold(&y, estimate_noise(&y, w, h).max(1e-5)));
        let blurred = gaussian_approx(&cleaned, w, h, 3.0);
        let weight = sky_weight(&y, w, h, sky_level);
        cleaned.par_iter_mut().zip(&blurred).zip(&weight).for_each(|((c, &b), &m)| *c += p.sky * m * (b - *c));
    }
    merge_planes(&guided, ColorMode::Luminance, std::slice::from_ref(&yg), vec![cleaned])
}

/// Where the sky smoothing acts (0‥1): below `sky_level` and with nothing brighter within
/// [`SKY_CLEAR`] px. A narrow dark gap between bright parts (Saturn's Cassini division, a ring's
/// shadow on the globe, a dark belt's edge) is detail, not sky: smoothed with the sky it lost
/// contrast, and the sharpening after it could not bring that back.
fn sky_weight(y: &[f32], w: usize, h: usize, sky_level: f32) -> Vec<f32> {
    let (_, bright) = crate::dering::min_max(y, w, h, SKY_CLEAR);
    gaussian_approx(&bright.par_iter().map(|&v| (v < sky_level) as u8 as f32).collect::<Vec<_>>(), w, h, 4.0)
}

/// How far from anything brighter than the sky a pixel must be to be smoothed as sky (px).
const SKY_CLEAR: usize = 4;

/// A level just above the sky background: its median plus a few noise σ.
pub(crate) fn sky_threshold(y: &[f32], sigma: f32) -> f32 {
    let stride = y.len().div_ceil(1 << 18).max(1);
    let mut sample: Vec<f32> = y.iter().step_by(stride).copied().collect();
    let k = sample.len() / 5;
    let low = *sample.select_nth_unstable_by(k, f32::total_cmp).1;
    low + 4.0 * sigma
}

/// Bilateral filter: neighbours are weighted by distance (σs) and by similarity (σr), so
/// averaging stops at edges.
fn bilateral(plane: &[f32], w: usize, h: usize, sigma_s: f32, sigma_r: f32) -> Vec<f32> {
    if let Some(out) = crate::gpu::for_plane(w, h, crate::gpu::Work::Heavy).and_then(|g| g.bilateral(plane, w, h, sigma_s, sigma_r).ok()) {
        return out;
    }
    bilateral_cpu(plane, w, h, sigma_s, sigma_r)
}

pub(crate) fn bilateral_cpu(plane: &[f32], w: usize, h: usize, sigma_s: f32, sigma_r: f32) -> Vec<f32> {
    let r = (2.0 * sigma_s).ceil() as isize;
    let spatial: Vec<(isize, isize, f32)> =
        (-r..=r).flat_map(|dy| (-r..=r).map(move |dx| (dx, dy))).map(|(dx, dy)| (dx, dy, (-((dx * dx + dy * dy) as f32) / (2.0 * sigma_s * sigma_s)).exp())).collect();
    let inv_r = -1.0 / (2.0 * sigma_r * sigma_r);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let c = plane[y * w + x];
            let (mut sum, mut norm) = (0.0f32, 0.0f32);
            for &(dx, dy, ws) in &spatial {
                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                let v = plane[sy * w + sx];
                let weight = ws * ((v - c) * (v - c) * inv_r).exp();
                sum += weight * v;
                norm += weight;
            }
            *o = sum / norm;
        }
    });
    out
}

/// Smooth only the colour (Cb, Cr), keeping luminance untouched.
fn smooth_chroma(img: &Image, levels: u32) -> Image {
    let [r, g, b] = img.channels() else {
        return img.clone();
    };
    let (w, h) = (img.width(), img.height());
    let y: Vec<f32> = (0..w * h).into_par_iter().map(|i| luma([r[i], g[i], b[i]])).collect();
    let cr = smooth_plane(&r.par_iter().zip(&y).map(|(c, y)| c - y).collect::<Vec<_>>(), w, h, levels);
    let cb = smooth_plane(&b.par_iter().zip(&y).map(|(c, y)| c - y).collect::<Vec<_>>(), w, h, levels);
    let rr: Vec<f32> = y.par_iter().zip(&cr).map(|(y, c)| y + c).collect();
    let bb: Vec<f32> = y.par_iter().zip(&cb).map(|(y, c)| y + c).collect();
    let gg: Vec<f32> = (0..w * h).into_par_iter().map(|i| (y[i] - LUMA[0] * rr[i] - LUMA[2] * bb[i]) / LUMA[1]).collect();
    Image::new(w, h, vec![rr, gg, bb], img.source_bit_depth)
}

// ---- contrast ----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContrastParams {
    /// Unsharp mask amount (0 = off).
    pub amount: f32,
    /// Unsharp mask radius (Gaussian σ, pixels).
    pub radius: f32,
    /// Differences smaller than this (0‥1 brightness) are left alone, so noise isn't sharpened.
    pub threshold: f32,
    /// Local contrast: an unsharp mask with a large radius (0 = off).
    pub local_amount: f32,
    pub local_radius: f32,
}

impl Default for ContrastParams {
    fn default() -> Self {
        Self { amount: 0.5, radius: 1.5, threshold: 0.0, local_amount: 0.0, local_radius: 40.0 }
    }
}

impl ContrastParams {
    pub fn is_neutral(&self) -> bool {
        self.amount == 0.0 && self.local_amount == 0.0
    }
}

pub fn contrast(img: &Image, p: &ContrastParams) -> Image {
    let (w, h) = (img.width(), img.height());
    let y = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let mut out = y.clone();
    if p.amount != 0.0 {
        let blur = gaussian_approx(&y, w, h, p.radius.max(0.3));
        let t = p.threshold.max(0.0);
        out.par_iter_mut().zip(y.par_iter().zip(&blur)).for_each(|(o, (&v, &b))| {
            let d = v - b;
            *o += p.amount * d.signum() * (d.abs() - t).max(0.0);
        });
    }
    if p.local_amount != 0.0 {
        let blur = gaussian_approx(&y, w, h, p.local_radius.max(2.0));
        out.par_iter_mut().zip(y.par_iter().zip(&blur)).for_each(|(o, (&v, &b))| *o += p.local_amount * (v - b));
    }
    merge_planes(img, ColorMode::Luminance, std::slice::from_ref(&y), vec![out])
}

// ---- tone --------------------------------------------------------------------------------

/// One tone adjustment: levels and a curve applied to every channel, then each channel's own
/// levels and curve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToneStep {
    pub black: f32,
    pub white: f32,
    /// Midtone gamma: > 1 brightens.
    pub gamma: f32,
    pub curve: Vec<[f32; 2]>,
    /// Red, green and blue curves (colour images).
    #[serde(default = "straight_curves")]
    pub channel_curves: [Vec<[f32; 2]>; 3],
    /// Red, green and blue levels: black, white, gamma.
    #[serde(default = "neutral_levels")]
    pub channel_levels: [[f32; 3]; 3],
}

/// Levels that change nothing: black 0, white 1, gamma 1.
pub const NEUTRAL_LEVELS: [f32; 3] = [0.0, 1.0, 1.0];

fn neutral_levels() -> [[f32; 3]; 3] {
    [NEUTRAL_LEVELS; 3]
}

/// Black point, white point and midtone gamma (> 1 brightens) on a value.
fn levels(v: f32, [black, white, gamma]: [f32; 3]) -> f32 {
    let span = (white - black).max(1e-4);
    ((v - black) / span).clamp(0.0, 1.0).powf(1.0 / gamma.max(0.05))
}

/// The identity curve.
pub fn straight() -> Vec<[f32; 2]> {
    vec![[0.0, 0.0], [1.0, 1.0]]
}

fn straight_curves() -> [Vec<[f32; 2]>; 3] {
    [straight(), straight(), straight()]
}

impl Default for ToneStep {
    fn default() -> Self {
        Self { black: 0.0, white: 1.0, gamma: 1.0, curve: straight(), channel_curves: straight_curves(), channel_levels: neutral_levels() }
    }
}

impl ToneStep {
    pub fn is_neutral(&self) -> bool {
        *self == Self::default()
    }

    /// The step as a function of one value (`curve` is this step's [`curve_lut`]).
    fn map(&self, curve: &[f32], v: f32) -> f32 {
        lookup(curve, levels(v, [self.black, self.white, self.gamma]))
    }

    /// A channel's own levels and curve (`curve` is its [`curve_lut`]) after the common part.
    fn map_channel(&self, c: usize, curve: &[f32], v: f32) -> f32 {
        let v = if self.channel_levels[c] == NEUTRAL_LEVELS { v } else { levels(v, self.channel_levels[c]) };
        lookup(curve, v)
    }

    /// The whole step for one channel (0‥2) of a value, or for a mono image (`None`): for
    /// previews that need it value by value (e.g. a histogram as it will look).
    pub fn apply_value(&self, channel: Option<usize>, v: f32) -> f32 {
        self.value_map()(channel, v)
    }

    /// [`Self::apply_value`] for many values: the curves' tables are built once, not per value.
    pub fn value_map(&self) -> impl Fn(Option<usize>, f32) -> f32 + '_ {
        let master = curve_lut(&self.curve);
        let channels: [Vec<f32>; 3] = std::array::from_fn(|c| curve_lut(&self.channel_curves[c]));
        move |channel, v| {
            let common = self.map(&master, v);
            match channel {
                Some(c) => self.map_channel(c, &channels[c], common),
                None => common,
            }
        }
    }

    fn has_channel_parts(&self) -> bool {
        self.channel_curves.iter().any(|c| *c != straight()) || self.channel_levels.iter().any(|l| *l != NEUTRAL_LEVELS)
    }
}

/// Tone adjustments: the steps already applied ("Applica"), then the one being edited.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToneParams {
    #[serde(flatten)]
    pub current: ToneStep,
    #[serde(default)]
    pub applied: Vec<ToneStep>,
}

impl ToneParams {
    pub fn is_neutral(&self) -> bool {
        self.current.is_neutral() && self.applied.iter().all(ToneStep::is_neutral)
    }

    /// `first`, then `then`, as one chain of steps.
    pub fn chain(first: &ToneParams, then: &ToneParams) -> ToneParams {
        let mut steps: Vec<ToneStep> = first.applied.iter().chain([&first.current]).chain(&then.applied).cloned().collect();
        steps.retain(|s| !s.is_neutral());
        ToneParams { current: then.current.clone(), applied: steps }
    }

    /// Freeze the current adjustment and start a new one on top of it.
    pub fn commit(&mut self) {
        if !self.current.is_neutral() {
            self.applied.push(std::mem::take(&mut self.current));
        }
    }
}

/// Entries of the table a chain of tone steps is folded into. Linear interpolation stays
/// within half an 8-bit level everywhere; the worst case is right at a black point raised by a
/// strong gamma, where the curve is steepest.
const TONE_LUT: usize = 1 << 16;

/// Every step maps values one by one, so the whole chain is a single function of the input:
/// tabulate it once and apply it with one lookup per sample.
pub fn tone(img: &Image, p: &ToneParams) -> Image {
    let steps: Vec<&ToneStep> = p.applied.iter().chain(std::iter::once(&p.current)).filter(|s| !s.is_neutral()).collect();
    if steps.is_empty() {
        return img.clone();
    }
    // One table, or one per channel when there are channel curves.
    let per_channel = img.channels().len() == 3 && steps.iter().any(|s| s.has_channel_parts());
    let luts: Vec<(Vec<f32>, [Vec<f32>; 3])> = steps.iter().map(|s| (curve_lut(&s.curve), s.channel_curves.clone().map(|c| curve_lut(&c)))).collect();
    let table = |channel: Option<usize>| -> Vec<f32> {
        (0..TONE_LUT)
            .into_par_iter()
            .map(|k| {
                steps.iter().zip(&luts).fold(k as f32 / (TONE_LUT - 1) as f32, |v, (step, (master, own))| {
                    let v = step.map(master, v);
                    match channel {
                        Some(c) => step.map_channel(c, &own[c], v),
                        None => v,
                    }
                })
            })
            .collect()
    };
    let tables: Vec<Vec<f32>> = if per_channel { (0..3).map(|c| table(Some(c))).collect() } else { vec![table(None)] };
    let at = |table: &[f32], v: f32| {
        // Every step starts by clamping to 0‥1, so the table covers all inputs.
        let x = v.clamp(0.0, 1.0) * (TONE_LUT - 1) as f32;
        let i = (x as usize).min(TONE_LUT - 2);
        let t = x - i as f32;
        table[i] + t * (table[i + 1] - table[i])
    };
    let channels = img
        .channels()
        .par_iter()
        .enumerate()
        .map(|(c, plane)| {
            let table = &tables[c.min(tables.len() - 1)];
            plane.par_iter().map(|&v| at(table, v)).collect()
        })
        .collect();
    Image::new(img.width(), img.height(), channels, img.source_bit_depth)
}

/// Image after the applied tone steps: what the current step (and its histogram) works on.
pub fn tone_base(img: &Image, p: &ToneParams) -> Image {
    tone(img, &ToneParams { current: ToneStep::default(), applied: p.applied.clone() })
}

/// Black point at the sky level, white point at the brightest 0.05 % of the luminance.
pub fn auto_levels(img: &Image) -> (f32, f32) {
    let y = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let stride = y.len().div_ceil(1 << 18).max(1);
    let mut sample: Vec<f32> = y.iter().step_by(stride).copied().collect();
    let n = sample.len();
    let low = *sample.select_nth_unstable_by(n / 100, f32::total_cmp).1;
    let high = *sample.select_nth_unstable_by((n * 9995 / 10000).min(n - 1), f32::total_cmp).1;
    (low.clamp(0.0, 0.5), high.clamp(low + 0.05, 1.0))
}

/// Levels for the automatic first look: the black point just below the darkest tones (the sky)
/// and the white point just above the brightest (the subject), each a few percent of the range
/// away from them, so the picture fills the range without anything being clipped.
pub fn gentle_levels(img: &Image) -> (f32, f32) {
    let y = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let stride = y.len().div_ceil(1 << 18).max(1);
    let mut sample: Vec<f32> = y.iter().step_by(stride).copied().collect();
    let n = sample.len();
    let low = *sample.select_nth_unstable_by(n / 1000, f32::total_cmp).1;
    let high = *sample.select_nth_unstable_by((n * 99995 / 100000).min(n - 1), f32::total_cmp).1;
    let span = (high - low).max(0.05);
    let black = (low - 0.04 * span).clamp(0.0, 0.5);
    let white = (high + 0.03 * span).clamp(black + 0.05, 1.0);
    (black, white)
}
