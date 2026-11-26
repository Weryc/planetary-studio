//! "Nitidezza intelligente": one control, sharpening shaped by the image.
//!
//! What the wavelet sliders do by hand is set here from a target response, the one
//! BlurXTerminator gives real stacks (measured with `ps-io/examples/cutoff_study.rs` on eight
//! planetary and lunar stacks, normalised by our measured cut-off): the boost peaks at the
//! cut-off, is still strong at 1.5× it, back to 1 at 2× and the grain beyond is cleaned;
//! towards coarser detail it falls off slowly (×1.7 at a quarter of the cut-off).
//!
//! The free-scale wavelet layers only say WHERE to sharpen: each layer is weighted, pixel by
//! pixel, by how much of it is real detail and not noise (a Wiener factor S / (S + N), with
//! noise growing with brightness: photon noise) and by the limb protection. Their weighted sum
//! is the detail to add. HOW MUCH each size of detail gets is then set exactly in frequency,
//! on that added detail only (the layers are far too broad to shape the response by
//! themselves: one half-octave layer spreads over two octaves). Zones the weights hold back
//! are therefore left as they are, instead of being reshaped with the rest.
//!
//! By default only the subject is sharpened (planet, disk, rings, moons): the sky holds only
//! noise, and the halo a sharpened limb leaves on it is never wanted. Whites are rolled off
//! before they clip, so the rest can be pushed further.
//!
//! If sharpening came before (deconvolution), what it already did is measured (spectrum after
//! over spectrum before) and only what is missing is added; its noise past the cut-off is
//! cleaned with the rest.
//!
//! What doesn't depend on the strength or the limb protection ([`Prepared`]) can be kept, so
//! dragging those sliders only recombines the layers.

use crate::i18n::tl;
use crate::Image;
use crate::color::{ColorMode, LUMA, merge_planes, working_planes};
use crate::wavelet::{ScaleMode, WaveletLayer, WaveletParams, decompose};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SmartParams {
    /// 0‥[`MAX_STRENGTH`]: height of the boost (1 = BlurXTerminator's typical ×12 at its peak).
    pub strength: f32,
    /// Where the detail ends, cycles/px (from the analysis); 0 = measured on the image.
    pub cutoff: f32,
    /// Peak of the boost as a fraction of the cut-off (by kind of subject).
    pub peak: f32,
    /// Less boost where edges meet the sky (limb, rings), 0‥1.
    pub limb_protection: f32,
    /// Width of the limb protection's band, as a multiple of the halo a boost at the peak
    /// leaves (1 = just that halo). Wider on Saturn, whose limb and ring edges grain up after
    /// stacking (alignment is worse there).
    pub limb_width: f32,
    /// Sharpen each zone by its own detail and noise (else uniformly).
    pub adaptive: bool,
    /// Shape of the boost across sizes of detail.
    pub curve: SmartCurve,
    pub color: ColorMode,
    /// Sharpen the sky too; by default only the subject (planet, disk, rings, moons) is.
    pub include_sky: bool,
    /// Roll the brightest values off before they reach white, so they never clip.
    pub protect_highlights: bool,
    /// Moon close-ups: along the sky past the limb, where the image is darker than this share
    /// of its bright level, the result never falls below the darkest of the original around
    /// it, so the limb leaves no black band beside it. 0 = off.
    pub sky_floor: f32,
    /// With [`Self::sky_floor`]: there the result also never rises above the brightest of the
    /// original around it, so no glow of amplified sky noise is left beside the edges (Saturn's
    /// rings and limb).
    pub sky_ceiling: bool,
    /// The shape picked for this image by the learned model ([`crate::learned`]), used by
    /// [`SmartCurve::Learned`]: (frequency over the cut-off, log-gain over the peak's).
    pub learned: Vec<(f32, f32)>,
    /// The strength the learned model suggests with its shape.
    pub learned_strength: f32,
    /// The learned curve under each hypothesis on the seeing (strength, shape), for those the
    /// models answered; `night` picks one of them instead of the main model's.
    pub learned_nights: Vec<crate::learned::NightCurve>,
    pub night: Option<crate::learned::Night>,
    /// The shape the subject's network ([`crate::curve_net`]) predicts for this image, used by
    /// [`SmartCurve::Network`]: (frequency over its peak's, log-gain over the peak's).
    pub network: Vec<(f32, f32)>,
    /// The height the network predicts (as a strength) and where its peak falls (cycles/px):
    /// with the curve chosen, the response's position (`peak`) moves around this frequency.
    pub network_strength: f32,
    pub network_cutoff: f32,
    /// Saturn: the rings' own network curve (shape, strength, peak), applied on the rings while
    /// the one above goes to the globe, blended by a soft mask of the globe. Empty elsewhere.
    pub network_rings: Vec<(f32, f32)>,
    pub network_rings_strength: f32,
    pub network_rings_cutoff: f32,
    /// The subject the network was trained for, which names the curve ("Specifica per Saturno").
    pub network_body: Option<crate::capture::Body>,
}

/// Shape of the boost across sizes of detail, relative to the cut-off. The default is the one
/// measured on BlurXTerminator; the others are variations of it designed for particular cases
/// (not measured). `Learned` is the one the learned model picks for the image (Standard when
/// there is none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmartCurve {
    /// Earlier versions also offered curves for poor and good seeing, surfaces and smooth
    /// globes (dropped); projects saved with any of them open with this one.
    #[default]
    #[serde(alias = "PoorSeeing", alias = "GoodSeeing", alias = "Surface", alias = "SmoothGlobe")]
    Standard,
    /// Earlier versions also had a model trained on the stacks themselves (dropped: its truth
    /// wasn't sharp); projects saved with either open with this one.
    #[serde(alias = "LearnedRef")]
    Learned,
    /// Predicted by the subject's network (Jupiter; Saturn's globe and rings), trained on simulated stacks.
    Network,
}

impl SmartCurve {
    pub const ALL: [SmartCurve; 1] = [Self::Standard];

    pub fn label(self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::Learned => tl("Appresa dai riferimenti"),
            Self::Network => tl("Appresa (rete)"),
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Standard => tl("Come BlurXTerminator sulle immagini planetarie: il massimo sul dettaglio più fine presente, un po' anche sulle strutture grandi."),
            Self::Learned => tl("Forma e forza scelte da un piccolo modello che ha imparato da immagini nitide di Hubble, Cassini, SDO e della NASA, riportate a dimensioni amatoriali e viste attraverso un seeing simulato come in uno stack vero: centinaia di fotogrammi con turbolenza, i migliori allineati e sommati. Decide solo le impostazioni: i pixel li calcolano sempre le wavelet."),
            Self::Network => tl("Curva prevista da una piccola rete addestrata apposta per questo soggetto (Giove; Saturno, con una curva per il globo e una per gli anelli; i primi piani della Luna) su stack simulati: centinaia di fotogrammi attraverso seeing e sensore, allineati e sommati davvero, confrontati con l'originale nitido. Guarda lo spettro del soggetto e il rumore nel cielo (se il cielo è stato tagliato a nero lo ignora) e sceglie quanto spingere ogni dimensione di dettaglio. Nitidezza e forma della risposta restano regolabili."),
        }
    }

    /// Log-gain over the peak's, by frequency over the cut-off (linear in log frequency).
    fn table(self) -> &'static [(f32, f32)] {
        match self {
            // BlurXTerminator's response on real stacks (median of eight, see the module notes).
            Self::Standard => &[(0.10, 0.0), (0.18, 0.10), (0.25, 0.21), (0.35, 0.35), (0.5, 0.55), (0.7, 0.80), (0.85, 0.93), (1.0, 1.0), (1.15, 0.91), (1.3, 0.76), (1.5, 0.48), (1.7, 0.24), (2.0, 0.03), (2.2, 0.0)],
            Self::Learned | Self::Network => Self::Standard.table(),
        }
    }
}

/// Value of a shape table at `r` (log-linear between points, 0 outside).
fn shape_in(t: &[(f32, f32)], r: f32) -> f32 {
    let (first, last) = (t[0], t[t.len() - 1]);
    if r <= first.0 || r >= last.0 {
        return 0.0;
    }
    let i = t.iter().position(|q| q.0 >= r).unwrap_or(t.len() - 1).max(1);
    let (a, b) = (t[i - 1], t[i]);
    a.1 + (b.1 - a.1) * (r / a.0).ln() / (b.0 / a.0).ln()
}

impl Default for SmartParams {
    fn default() -> Self {
        Self {
            strength: 0.0,
            cutoff: 0.0,
            peak: 1.0,
            limb_protection: 0.3,
            limb_width: 1.0,
            adaptive: true,
            curve: SmartCurve::Standard,
            color: ColorMode::Luminance,
            include_sky: false,
            protect_highlights: true,
            sky_floor: 0.0,
            sky_ceiling: false,
            learned: Vec::new(),
            learned_strength: 0.0,
            learned_nights: Vec::new(),
            night: None,
            network: Vec::new(),
            network_strength: 0.0,
            network_cutoff: 0.0,
            network_rings: Vec::new(),
            network_rings_strength: 0.0,
            network_rings_cutoff: 0.0,
            network_body: None,
        }
    }
}

/// Peak gain at strength 1: BlurXTerminator's typical peak on real stacks (median ×10–12).
pub const PEAK_AT_FULL: f32 = 12.0;
/// The strength can go past BlurXTerminator's typical result.
pub const MAX_STRENGTH: f32 = 2.0;

/// The name of a network's curve: the subject it was trained for.
pub fn network_label(body: Option<crate::capture::Body>) -> String {
    use crate::capture::Body;
    match body {
        Some(Body::Moon) => tl("Specifica per la Luna").to_string(),
        Some(Body::Sun) => tl("Specifica per il Sole").to_string(),
        Some(b) => crate::tr!("Specifica per {}", "{}-specific", b.label()),
        None => SmartCurve::Network.label().to_string(),
    }
}

impl SmartParams {
    /// The name shown for `curve`: a network's says which subject it was trained for.
    pub fn curve_label(&self, curve: SmartCurve) -> String {
        match curve {
            SmartCurve::Network => network_label(self.network_body),
            other => other.label().to_string(),
        }
    }

    /// The shape table in use: the learned one when chosen and present.
    fn table(&self) -> &[(f32, f32)] {
        match self.curve {
            SmartCurve::Learned => match self.night.and_then(|n| self.learned_nights.iter().find(|q| q.0 == n)) {
                Some(q) => &q.2,
                None if self.learned.len() >= 2 => &self.learned,
                None => SmartCurve::Standard.table(),
            },
            SmartCurve::Network if self.network.len() >= 2 => &self.network,
            curve => curve.table(),
        }
    }

    /// Whether the network's curve is the one in use (chosen and present).
    fn uses_network(&self) -> bool {
        self.curve == SmartCurve::Network && self.network.len() >= 2 && self.network_cutoff > 0.0
    }

    /// Saturn's rings, when their own curve is in use: the same settings with the rings'
    /// curve, the strength moved as the user moved the globe's (same ratio to the suggestion).
    pub fn rings_params(&self) -> Option<SmartParams> {
        if !self.uses_network() || self.network_rings.len() < 2 || self.network_rings_cutoff <= 0.0 {
            return None;
        }
        let ratio = if self.network_strength > 0.0 { self.strength / self.network_strength } else { 1.0 };
        Some(SmartParams {
            network: self.network_rings.clone(),
            network_strength: self.network_rings_strength,
            network_cutoff: self.network_rings_cutoff,
            strength: (self.network_rings_strength * ratio).clamp(0.0, MAX_STRENGTH),
            // The rings are edges against the sky: held back there, their curve lost a seventh
            // of what it gains (simulated stacks, 2026-11-17), with no halo to save.
            limb_protection: 0.0,
            network_rings: Vec::new(),
            ..self.clone()
        })
    }

    /// The frequency the shape is placed by (before the response's position): the network's
    /// peak with its curve, else the cut-off set from the analysis (0 = measured on the image).
    pub fn base_cutoff(&self) -> f32 {
        if self.uses_network() { self.network_cutoff } else { self.cutoff }
    }

    fn shape(&self, r: f32) -> f32 {
        shape_in(self.table(), r)
    }

    /// Past this multiple of the cut-off there is only grain: it starts being cleaned. A learned
    /// table says it itself: where it first goes below 1 past its peak.
    fn grain_from(&self) -> f32 {
        let t = self.table();
        if self.cleans_itself() {
            let top = t.iter().enumerate().max_by(|a, b| a.1.1.total_cmp(&b.1.1)).map_or(0, |q| q.0);
            if let Some(q) = t[top..].iter().find(|q| q.1 <= 0.0) {
                return q.0;
            }
        }
        t.last().map_or(1.9, |q| 0.86 * q.0)
    }

    /// The learned shape carries its own cleaning of the grain (values below 0 past the peak).
    fn cleans_itself(&self) -> bool {
        matches!(self.curve, SmartCurve::Learned | SmartCurve::Network) && self.table().iter().any(|q| q.1 < 0.0)
    }

    /// Where the finest boosted layer should peak, over the cut-off.
    fn finest(&self) -> f32 {
        if self.cleans_itself() {
            return 0.95 * self.grain_from();
        }
        self.table().last().map_or(1.8, |q| 0.82 * q.0)
    }

    /// Share of the grain past the cut-off that is removed: follows the strength.
    pub fn effective_cleaning(&self) -> f32 {
        0.6 * self.strength.clamp(0.0, 1.0)
    }

    pub fn is_neutral(&self) -> bool {
        self.strength <= 0.0
    }

    /// What [`prepare`] depends on (besides the images).
    pub fn prepare_key(&self) -> PrepareKey {
        let learned = self.table().iter().chain(self.rings_params().as_ref().map_or(&[][..], |r| &r.network[..])).fold(0u64, |h, &(r, v)| h.rotate_left(7) ^ (u64::from(r.to_bits()) << 32 | u64::from(v.to_bits())));
        let learned = learned ^ u64::from(self.network_rings_cutoff.to_bits()).rotate_left(17);
        PrepareKey { cutoff: self.base_cutoff().to_bits(), peak: self.peak.to_bits(), curve: self.curve, learned, adaptive: self.adaptive, color: self.color, limb_width: self.limb_width.to_bits() }
    }

    /// The overall response wanted at `f` given what earlier steps did (`done`): never below 1
    /// where there is detail (what they sharpened beyond the target stays), below 1 past the
    /// detail (grain, and noise they amplified, is cleaned).
    fn wanted(&self, f: f32, cutoff: f32, done: &Done) -> f32 {
        let want = self.target(f, cutoff) / done.at(f);
        if f <= self.grain_from() * self.peak * cutoff { want.max(1.0) } else { want }
    }

    /// Target overall response at `f` cycles/px for the cut-off `cutoff`.
    pub fn target(&self, f: f32, cutoff: f32) -> f32 {
        let r = f.max(1e-4) / (self.peak * cutoff).max(1e-3);
        let peak = 1.0 + (PEAK_AT_FULL - 1.0) * self.strength.clamp(0.0, MAX_STRENGTH);
        let t = ((r - self.grain_from()) / 0.6).clamp(0.0, 1.0);
        let beyond = t * t * (3.0 - 2.0 * t);
        if self.cleans_itself() {
            return peak.powf(self.shape(r));
        }
        peak.powf(self.shape(r)) * (1.0 - 0.9 * self.effective_cleaning() * beyond)
    }
}

/// Peak frequency (cycles/px) of a free-scale layer of size `scale` with neighbours `ratio`
/// apart: the layer is a difference of Gaussians σ and ratio·σ, σ = scale / ratio, peaking at
/// √(ln r² / (2π² (r² − 1))) / σ.
fn layer_peak(scale: f32, ratio: f32) -> f32 {
    let r2 = ratio * ratio;
    (r2.ln() / (2.0 * std::f32::consts::PI.powi(2) * (r2 - 1.0))).sqrt() * ratio / scale
}

/// Free-scale transform with its first layer peaking at `peak` cycles/px, half an octave
/// apart up to ~50 px (the broad belts and zones of a large planet need the coarse layers).
/// Everything finer is grain.
fn transform(peak: f32) -> WaveletParams {
    let ratio = std::f32::consts::SQRT_2;
    let first = (layer_peak(1.0, ratio) / peak.max(0.04)).clamp(1.0, 5.0);
    let levels = (((50.0 / first).ln() / ratio.ln()).ceil() as usize + 1).clamp(4, 12);
    WaveletParams { scale_mode: ScaleMode::Free, first_scale: first, ratio, layers: vec![WaveletLayer::default(); levels], noise_adaptive: 0.0, edge_protection: 0.0, ..Default::default() }
}

/// The cut-off to use: the given one, else measured on the plane, else a typical one.
fn cutoff_of(p: &SmartParams, plane: &[f32], w: usize, h: usize) -> f32 {
    if p.base_cutoff() > 0.0 {
        return p.base_cutoff().min(0.5);
    }
    crate::spectrum::analyze(plane, w, h, w as f32 / 2.0, h as f32 / 2.0).map_or(0.2, |s| s.cutoff)
}

/// Bands averaged on each side by [`Done::measure`].
const DONE_SMOOTHING: usize = 2;

/// How much sharpening already happened, by frequency: √(P_now / P_before), 1 if unknown.
#[derive(Clone, Debug, Default, PartialEq)]
struct Done(Vec<(f32, f32)>);

impl Done {
    fn measure(now: &[f32], before: &[f32], w: usize, h: usize) -> Done {
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        match (crate::spectrum::radial_power(now, w, h, cx, cy), crate::spectrum::radial_power(before, w, h, cx, cy)) {
            (Some(a), Some(b)) => {
                // Only what earlier steps added counts: what they took away on purpose (the noise
                // reduction, a wavelet's grain cleaning) must not be put back. Counting it, the
                // sharpening boosted the cleaned bands up to ×3 more: noise reduction then made
                // the result noisier and dug rings beside the edges.
                let raw: Vec<(f32, f32)> = a.iter().zip(&b).map(|(&(f, pa), &(_, pb))| (f, (pa / pb.max(1e-30)).sqrt().clamp(1.0, 20.0))).collect();
                // Averaged over neighbouring bands: band by band the ratio is noisy, and a jagged
                // response rings far across the image.
                let k = DONE_SMOOTHING;
                let ratio: Vec<(f32, f32)> = (0..raw.len())
                    .map(|i| {
                        let part = &raw[i.saturating_sub(k)..(i + k + 1).min(raw.len())];
                        (raw[i].0, (part.iter().map(|r| r.1.ln()).sum::<f32>() / part.len() as f32).exp())
                    })
                    .collect();
                if ratio.iter().all(|r| (r.1 - 1.0).abs() < 1e-3) { Done::default() } else { Done(ratio) }
            }
            _ => Done::default(),
        }
    }

    fn at(&self, f: f32) -> f32 {
        let bands = &self.0;
        let (Some(first), Some(last)) = (bands.first(), bands.last()) else { return 1.0 };
        if f <= first.0 {
            return first.1;
        }
        if f >= last.0 {
            return last.1;
        }
        // Between the two bands around `f`, in log frequency: a step from one band to the next
        // would make the filter ring far across the image (and differ on a zoomed region).
        let i = bands.iter().position(|b| b.0 >= f).unwrap_or(bands.len() - 1).max(1);
        let (a, b) = (bands[i - 1], bands[i]);
        let t = ((f / a.0).ln() / (b.0 / a.0).ln().max(1e-9)).clamp(0.0, 1.0);
        a.1 + t * (b.1 - a.1)
    }
}

/// Brightness relative to the subject's, per pixel: photon noise grows with it.
fn brightness(plane: &[f32], w: usize, h: usize) -> Vec<f32> {
    brightness_with(plane, w, h, brightness_reference(plane))
}

/// The level [`brightness`] is relative to: the median of the lit part of `plane`.
fn brightness_reference(plane: &[f32]) -> f32 {
    let mut sample: Vec<f32> = plane.iter().step_by(plane.len().div_ceil(1 << 16).max(1)).copied().collect();
    sample.sort_by(f32::total_cmp);
    let top = sample.get(sample.len() * 99 / 100).copied().unwrap_or(1.0).max(1e-6);
    let lit: Vec<f32> = sample.iter().copied().filter(|&v| v > 0.15 * top).collect();
    lit.get(lit.len() / 2).copied().unwrap_or(top)
}

/// [`brightness`] relative to a given level.
fn brightness_with(plane: &[f32], w: usize, h: usize, reference: f32) -> Vec<f32> {
    let floor = 0.1 * reference;
    let smooth = crate::mask::gaussian_approx(plane, w, h, 2.0);
    smooth.into_par_iter().map(|v| (v.max(0.0) + floor) / (reference + floor)).collect()
}

/// How much of a layer is real detail at each pixel (0‥1): its local energy against the noise
/// expected there (`sigma` scaled by the brightness).
fn layer_share(layer: &[f32], sigma: f32, brightness: &[f32], w: usize, h: usize, scale: f32) -> Vec<f32> {
    let noise = sigma * sigma;
    let energy: Vec<f32> = layer.par_iter().map(|v| v * v).collect();
    let local = crate::mask::gaussian_approx(&energy, w, h, (2.0 * scale).max(2.0));
    local
        .into_par_iter()
        .zip(brightness)
        .map(|(e, &b)| {
            let n = noise * b;
            let s = (e - n).max(0.0);
            s / (s + n).max(1e-30)
        })
        .collect()
}

/// Where the limb protection acts (0‥1): the edges against the sky widened, inwards and
/// outwards, by the size of the halo a boost peaking at `fp` cycles/px leaves.
fn limb_band(edges: &[f32], w: usize, h: usize, fp: f32, width: f32) -> Vec<f32> {
    let width = width.clamp(0.5, 8.0);
    let r = (1.2 / fp.max(1e-3)).clamp(3.0, 20.0) * width;
    let (_, grown) = crate::dering::min_max(edges, w, h, r.round() as usize);
    crate::mask::gaussian_approx(&grown, w, h, r / 3.0).into_iter().map(|v| v.clamp(0.0, 1.0)).collect()
}

/// What sharpening needs that doesn't depend on the strength or the limb protection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrepareKey {
    cutoff: u32,
    peak: u32,
    curve: SmartCurve,
    learned: u64,
    adaptive: bool,
    color: ColorMode,
    limb_width: u32,
}

/// What [`prepare`] measures on the whole of a plane: a zoomed region reuses it, so it is
/// sharpened as the whole image is (its own noise and brightness would differ).
#[derive(Clone, Debug, PartialEq)]
struct PlaneMeasures {
    cutoff: f32,
    /// The noise of each layer, as white noise measured on the finest.
    white_noise: Vec<f32>,
    /// The level the brightness is relative to.
    light: f32,
    /// What earlier steps did (measured over the whole image).
    done: Done,
    /// What the limb protection measures edges against (see
    /// [`crate::wavelet::Decomposition::edge_levels`]).
    edges: Option<(f32, f32)>,
}

/// [`prepare`]'s measures of a whole image ([`Prepared::reference`]), for a zoomed region of it
/// ([`prepare_with`]).
#[derive(Clone, Debug, PartialEq)]
pub struct SmartReference {
    key: PrepareKey,
    planes: Vec<PlaneMeasures>,
    /// Saturn with two curves: the rings' measures and the globe found on the whole image (a
    /// region cut through it would find another, or none).
    rings: Option<(Vec<PlaneMeasures>, crate::mask::Ellipse)>,
    /// The bright level [`SmartParams::sky_floor`] is a share of.
    sky_top: Option<f32>,
}

/// One working plane, ready to be sharpened at any strength.
struct PreparedPlane {
    measures: PlaneMeasures,
    cutoff: f32,
    transform: WaveletParams,
    /// (layer index, layer weighted by its share of real detail): only layers the curve boosts.
    layers: Vec<(usize, Vec<f32>)>,
    /// Where the limb protection acts.
    band: Vec<f32>,
    done: Done,
}

/// The decomposed, weighted image: [`apply`] turns it into the sharpened one at any strength
/// and limb protection.
pub struct Prepared {
    planes: Vec<PreparedPlane>,
    /// Saturn with two curves: the rings' prepared planes, the globe's weight per pixel and the
    /// globe it comes from.
    rings: Option<(Vec<PreparedPlane>, Vec<f32>, crate::mask::Ellipse)>,
    /// The image before any earlier sharpening, kept for [`SmartParams::sky_floor`]: the
    /// deconvolution and the wavelets may already have dug the band along the limb, so the floor
    /// is the original's.
    floor: Option<Image>,
    /// The bright level of the image the floor is held against (the whole image's, for a region).
    sky_top: Option<f32>,
}

impl Prepared {
    /// What was measured, for a zoomed region of the same image prepared with `p`.
    pub fn reference(&self, p: &SmartParams) -> SmartReference {
        let rings = self.rings.as_ref().map(|(planes, _, globe)| (planes.iter().map(|q| q.measures.clone()).collect(), *globe));
        SmartReference { key: p.prepare_key(), planes: self.planes.iter().map(|q| q.measures.clone()).collect(), rings, sky_top: self.sky_top }
    }

    /// Memory held (bytes).
    pub fn bytes(&self) -> usize {
        let of = |planes: &[PreparedPlane]| planes.iter().map(|p| 4 * (p.band.len() + p.layers.iter().map(|l| l.1.len()).sum::<usize>())).sum::<usize>();
        of(&self.planes) + self.rings.as_ref().map_or(0, |(r, m, _)| of(r) + 4 * m.len()) + self.floor.as_ref().map_or(0, |f| 4 * f.width() * f.height() * f.channels().len())
    }
}

/// Decompose and weight `img`; `before` is the image before any earlier sharpening.
pub fn prepare(img: &Image, before: Option<&Image>, p: &SmartParams) -> Prepared {
    prepare_with(img, before, p, None, (0, 0))
}

/// [`prepare`] a zoomed region with the measures of the whole image (`reference`, used when
/// prepared with the same settings); `offset` is where the region sits in the whole image.
pub fn prepare_with(img: &Image, before: Option<&Image>, p: &SmartParams, reference: Option<&SmartReference>, offset: (usize, usize)) -> Prepared {
    let (w, h) = (img.width(), img.height());
    let planes = working_planes(img, p.color);
    let before_planes = before.filter(|b| b.width() == w && b.height() == h).map(|b| working_planes(b, p.color));
    let reference = reference.filter(|r| r.key == p.prepare_key() && r.planes.len() == planes.len());
    // One plane at a time: each is already processed in parallel, and memory stays low.
    let prep = |q: &SmartParams, measured: Option<&SmartReference>| {
        planes.iter().enumerate().map(|(k, plane)| prepare_plane(plane, before_planes.as_ref().map(|b| b[k].as_slice()), w, h, q, measured.map(|r| &r.planes[k]))).collect::<Vec<_>>()
    };
    let rings = p.rings_params().and_then(|r| match reference.and_then(|q| q.rings.as_ref()) {
        Some((measures, globe)) if measures.len() == planes.len() => {
            let measured = SmartReference { key: p.prepare_key(), planes: measures.clone(), rings: None, sky_top: None };
            Some((prep(&r, Some(&measured)), crate::saturn::globe_weight(globe, w, h, offset), *globe))
        }
        _ => crate::saturn::zones(before.unwrap_or(img)).map(|z| (prep(&r, None), z.planet, z.globe)),
    });
    let floor = before.filter(|b| p.sky_floor > 0.0 && b.width() == w && b.height() == h && b.channels().len() == img.channels().len()).cloned();
    let sky_top = (p.sky_floor > 0.0).then(|| reference.and_then(|r| r.sky_top).or_else(|| bright_level(floor.as_ref().unwrap_or(img)))).flatten();
    Prepared { planes: prep(p, reference), rings, floor, sky_top }
}

fn prepare_plane(plane: &[f32], before: Option<&[f32]>, w: usize, h: usize, p: &SmartParams, measured: Option<&PlaneMeasures>) -> PreparedPlane {
    let cutoff = measured.map_or_else(|| cutoff_of(p, before.unwrap_or(plane), w, h), |m| m.cutoff);
    let fp = p.peak * cutoff;
    let done = match (measured, before) {
        (Some(m), Some(_)) => m.done.clone(),
        (None, Some(b)) => Done::measure(plane, b, w, h),
        (_, None) => Done::default(),
    };
    let transform = transform(p.finest() * fp);
    let Some(mut d) = decompose(plane, w, h, &transform, &|| false) else {
        let measures = PlaneMeasures { cutoff, white_noise: Vec::new(), light: 1.0, done: done.clone(), edges: None };
        return PreparedPlane { measures, cutoff, transform, layers: Vec::new(), band: vec![0.0; plane.len()], done };
    };
    let edges = measured.map_or_else(|| Some(d.measure_edge_levels()), |m| m.edges);
    d.edge_levels = edges;
    let band = limb_band(d.edges(), w, h, fp, p.limb_width);
    let white_noise = measured.map(|m| m.white_noise.clone()).filter(|v| v.len() == d.white_noise.len()).unwrap_or_else(|| d.white_noise.clone());
    let light_ref = measured.map_or_else(|| brightness_reference(plane), |m| m.light);
    let measures = PlaneMeasures { cutoff, white_noise: white_noise.clone(), light: light_ref, done: done.clone(), edges };
    let light = if p.adaptive { Some(brightness_with(plane, w, h, light_ref)) } else { None };
    let mut layers = Vec::new();
    for (j, mut layer) in std::mem::take(&mut d.layers).into_iter().enumerate() {
        // Layers the curve never boosts, at any strength, aren't needed.
        if p.shape(layer_peak(transform.scale(j), transform.ratio) / fp) <= 0.0 {
            continue;
        }
        if let Some(light) = &light {
            let share = layer_share(&layer, white_noise[j], light, w, h, transform.scale(j));
            layer.par_iter_mut().zip(share).for_each(|(v, s)| *v *= s);
        }
        layers.push((j, layer));
    }
    PreparedPlane { measures, cutoff, transform, layers, band, done }
}

/// The sharpened image from [`prepare`]d `img`; `subject` (0‥1 per pixel, same size) is where
/// it applies when the sky is left out.
pub fn apply(prepared: &Prepared, img: &Image, p: &SmartParams, subject: Option<&[f32]>) -> Image {
    let (w, h) = (img.width(), img.height());
    let subject = subject.filter(|m| !p.include_sky && m.len() == w * h);
    let planes = working_planes(img, p.color);
    let mut processed: Vec<Vec<f32>> = planes.iter().zip(&prepared.planes).map(|(plane, pp)| apply_plane(plane, pp, w, h, p, subject)).collect();
    // Saturn: the rings' curve where the globe's weight falls to 0.
    if let (Some((rings, globe, _)), Some(rp)) = (&prepared.rings, p.rings_params())
        && globe.len() == w * h
    {
        for ((out, plane), pp) in processed.iter_mut().zip(&planes).zip(rings) {
            let r = apply_plane(plane, pp, w, h, &rp, subject);
            out.par_iter_mut().zip(r).zip(globe.par_iter()).for_each(|((o, r), t)| *o = r + t * (*o - r));
        }
    }
    if p.color == ColorMode::Rgb {
        smooth_added_colour(&planes, &mut processed, w, h);
    }
    let mut out = merge_planes(img, p.color, &planes, processed);
    if p.sky_floor > 0.0 {
        let original = prepared.floor.as_ref().filter(|f| f.width() == w && f.height() == h && f.channels().len() == img.channels().len());
        hold_sky(&mut out, original.unwrap_or(img), p.sky_floor, p.sky_ceiling, prepared.sky_top);
    }
    if p.protect_highlights {
        out.channels_mut().par_iter_mut().zip(img.channels()).for_each(|(o, before)| o.iter_mut().zip(before).for_each(|(v, &b)| *v = roll_off(*v, b)));
    }
    out
}

/// Window of the floor: the darkest original value within this many pixels.
const FLOOR_RADIUS: usize = 2;
/// The sky: below this share of the bright level (smoothed).
const SKY_SHARE: f32 = 0.06;
/// How far from the sky the floor reaches (px): the band a strong boost darkens beside a limb.
const FLOOR_REACH: usize = 8;

/// Along the sky (past the Moon's limb), where `img` is darker than `share` of its bright level,
/// keep `out` at or above the darkest original value nearby: the black band a strong boost
/// leaves beside a bright edge is cut, the boost on the lit side is kept. Shadows away from the
/// sky (crater floors) are left to the sharpening.
/// With `ceiling`, there it also never rises above the brightest original value nearby. `top`:
/// the bright level (the whole image's, for a zoomed region), else measured on `img`.
fn hold_sky(out: &mut Image, img: &Image, share: f32, ceiling: bool, top: Option<f32>) {
    let (w, h) = (img.width(), img.height());
    let smooth = smooth_brightness(img);
    let Some(top) = top.or_else(|| bright_level(img)) else { return };
    if top <= 0.0 {
        return;
    }
    let sky: Vec<f32> = smooth.iter().map(|&v| if v < SKY_SHARE * top { 1.0 } else { 0.0 }).collect();
    if sky.iter().sum::<f32>() < (w * h) as f32 * 0.002 {
        return;
    }
    let (_, near_sky) = crate::dering::min_max(&sky, w, h, FLOOR_REACH);
    // Softly (a hard on/off would leave steps where the band ends), but in full on all of it:
    // grown by a few pixels before the blur, full from half the blurred weight up.
    let held: Vec<f32> = smooth.iter().zip(&near_sky).map(|(&v, &s)| if s > 0.5 && v < share * top { 1.0 } else { 0.0 }).collect();
    let (_, held) = crate::dering::min_max(&held, w, h, 3);
    let held = crate::mask::gaussian_approx(&held, w, h, 1.5);
    out.channels_mut().par_iter_mut().zip(img.channels()).for_each(|(o, before)| {
        let (lo, hi) = crate::dering::min_max(before, w, h, FLOOR_RADIUS);
        o.iter_mut().zip(lo.iter().zip(&hi)).zip(&held).for_each(|((v, (&floor, &top)), &k)| {
            let k = (2.0 * k).clamp(0.0, 1.0);
            if *v < floor {
                *v += k * (floor - *v);
            } else if ceiling && *v > top {
                *v -= k * (*v - top);
            }
        });
    });
}

/// The mean of the channels, lightly smoothed.
fn smooth_brightness(img: &Image) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let n = img.channels().len().max(1) as f32;
    let lum: Vec<f32> = (0..w * h).map(|i| img.channels().iter().map(|c| c[i]).sum::<f32>() / n).collect();
    crate::mask::gaussian_approx(&lum, w, h, 2.0)
}

/// The bright level [`SmartParams::sky_floor`] is a share of: the 99th percentile of
/// [`smooth_brightness`]. `None` on an empty image.
fn bright_level(img: &Image) -> Option<f32> {
    let smooth = smooth_brightness(img);
    let mut sample: Vec<f32> = smooth.iter().step_by(7).copied().collect();
    if sample.is_empty() {
        return None;
    }
    sample.sort_by(f32::total_cmp);
    Some(sample[sample.len() * 99 / 100])
}

/// Where whites start being rolled off.
const KNEE: f32 = 0.85;

/// A value raised by sharpening approaches white without reaching it; what was already there
/// is kept as it was.
fn roll_off(v: f32, before: f32) -> f32 {
    let start = before.max(KNEE);
    if v <= start {
        return v;
    }
    if start >= 1.0 {
        return v.min(before);
    }
    let room = 1.0 - start;
    start + room * (1.0 - (-(v - start) / room).exp())
}

fn apply_plane(plane: &[f32], pp: &PreparedPlane, w: usize, h: usize, p: &SmartParams, subject: Option<&[f32]>) -> Vec<f32> {
    const CHUNK: usize = 16 * 1024;
    let cutoff = pp.cutoff;
    let want = |f: f32| towards_nyquist(p.wanted(f, cutoff, &pp.done), f);
    let gains: Vec<f32> = pp.layers.iter().map(|(j, _)| (want(layer_peak(pp.transform.scale(*j), pp.transform.ratio)) - 1.0).max(0.0)).collect();
    let protect = p.limb_protection.clamp(0.0, 1.0);
    // The detail to add: each layer by its gain, where it is real detail, less at the limb.
    let mut added = vec![0.0f32; plane.len()];
    added.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, chunk)| {
        let (start, len) = (ci * CHUNK, chunk.len());
        for ((_, layer), &g) in pp.layers.iter().zip(&gains) {
            if g > 0.0 {
                chunk.iter_mut().zip(&layer[start..start + len]).for_each(|(a, &v)| *a += g * v);
            }
        }
        if protect > 0.0 {
            chunk.iter_mut().zip(&pp.band[start..start + len]).for_each(|(a, &b)| *a *= 1.0 - protect * b);
        }
    });
    // What the layers add at full weight, by frequency.
    let mut all = vec![0.0f32; pp.transform.layers.len()];
    pp.layers.iter().zip(&gains).for_each(|((j, _), &g)| all[*j] = g);
    let extra = |f: f32| crate::wavelet::responses(&pp.transform, f, 0.0).1.iter().zip(&all).map(|(r, g)| g * r).sum::<f32>();
    // The added detail shaped to the target exactly (so at full weight the image gets the
    // target response); the grain past the detail cleaned everywhere it is applied.
    let shape = |f: f32| {
        let (t, e) = (want(f), extra(f));
        if t > 1.0 && e > 1e-3 { ((t - 1.0) / e).min(MAX_SHAPE) } else { 0.0 }
    };
    let clean = |f: f32| want(f).min(1.0) - 1.0;
    let mut out = add_filtered(plane, &added, w, h, &shape, &clean);
    if let Some(m) = subject {
        out.par_iter_mut().zip(plane).zip(m).for_each(|((o, &v), &m)| *o = v + m.clamp(0.0, 1.0) * (*o - v));
    }
    out
}

/// Most the added layers are multiplied by at one frequency to reach the target: where they
/// hold little of it (next to the pixel's limit) a larger factor only amplifies their grain.
const MAX_SHAPE: f32 = 3.0;
/// Past this frequency (cycles/px) a boost fades out, to none at the pixel's limit: there a
/// stack holds almost only noise, and a boost that stops abruptly at the limit draws a fine
/// grid (the frequencies just below it are kept, those on it are not).
const NYQUIST_FADE_FROM: f32 = 0.36;
/// The added colour (a channel's boost apart from the brightness's) is kept only for detail
/// coarser than this (Gaussian σ, px): the colour the seeing blurred comes back, the colour
/// noise of each channel is not amplified.
const CHROMA_SIGMA: f32 = 1.2;

/// A boost (`want` > 1 at `f` cycles/px) faded out towards the pixel's limit.
fn towards_nyquist(want: f32, f: f32) -> f32 {
    let from = NYQUIST_FADE_FROM;
    if want <= 1.0 || f <= from {
        return want;
    }
    let t = ((f - from) / (0.5 - from)).clamp(0.0, 1.0);
    let keep = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
    1.0 + (want - 1.0) * keep
}

/// Red, green, blue sharpened each on its own: keep what they add to the brightness, and only
/// the coarser part of what they add to the colour.
fn smooth_added_colour(before: &[Vec<f32>], after: &mut [Vec<f32>], w: usize, h: usize) {
    let sigma = CHROMA_SIGMA;
    if before.len() != 3 || after.len() != 3 || sigma <= 0.0 {
        return;
    }
    let n = w * h;
    let added: Vec<Vec<f32>> = (0..3).map(|k| (0..n).map(|i| after[k][i] - before[k][i]).collect()).collect();
    let lum: Vec<f32> = (0..n).map(|i| LUMA[0] * added[0][i] + LUMA[1] * added[1][i] + LUMA[2] * added[2][i]).collect();
    for k in 0..3 {
        let colour: Vec<f32> = added[k].iter().zip(&lum).map(|(a, l)| a - l).collect();
        let colour = crate::mask::gaussian_approx(&colour, w, h, sigma);
        after[k].par_iter_mut().enumerate().for_each(|(i, v)| *v = before[k][i] + lum[i] + colour[i]);
    }
}

/// `plane + F⁻¹[shape(f)·F(added) + clean(f)·F(plane)]`, mirror-padded so the edges don't wrap.
fn add_filtered(plane: &[f32], added: &[f32], w: usize, h: usize, shape: &dyn Fn(f32) -> f32, clean: &dyn Fn(f32) -> f32) -> Vec<f32> {
    // Tabulated: the closures needn't be shareable between threads.
    const STEPS: usize = 512;
    let tab = |r: &dyn Fn(f32) -> f32| -> Vec<f32> { (0..=STEPS).map(|k| r(0.5 * k as f32 / STEPS as f32)).collect() };
    let (shape, clean) = (tab(shape), tab(clean));
    let cleaning = clean.iter().any(|c| c.abs() > 1e-4);
    if !cleaning && shape.iter().all(|s| *s == 0.0) {
        return plane.to_vec();
    }
    let padded = crate::fft::Padded::new(added, w, h, 16);
    let (pw, ph) = (padded.w, padded.h);
    let fft = crate::fft::FftConvolver::plan(pw, ph);
    let mut spec = fft.forward(&padded.data);
    let base = cleaning.then(|| fft.forward(&crate::fft::Padded::new(plane, w, h, 16).data));
    let index = |kx: usize, ky: usize| {
        let fx = kx as f32 / pw as f32;
        let fy = (if ky <= ph / 2 { ky as f32 } else { ky as f32 - ph as f32 }) / ph as f32;
        ((fx.hypot(fy).min(0.5) / 0.5) * STEPS as f32).round() as usize
    };
    // Transposed layout: index = kx·ph + ky, kx ∈ 0..=pw/2.
    spec.par_chunks_mut(ph).enumerate().for_each(|(kx, col)| {
        for (ky, c) in col.iter_mut().enumerate() {
            let k = index(kx, ky);
            *c *= shape[k];
            if let Some(b) = &base {
                *c += b[kx * ph + ky] * clean[k];
            }
        }
    });
    let mut out = padded.crop(&fft.inverse(spec));
    out.par_iter_mut().zip(plane).for_each(|(o, &v)| *o += v);
    out
}

/// Sharpen `img`; `before` is the image before any earlier sharpening (deconvolution), whose
/// effect is taken into account; `subject`: see [`apply`].
pub fn sharpen(img: &Image, before: Option<&Image>, p: &SmartParams, subject: Option<&[f32]>) -> Image {
    apply(&prepare(img, before, p), img, p, subject)
}

/// Where the sharpening is held back, per pixel (0‥1 = share of the boost removed): by "Segui
/// le zone" (little detail against the noise there) and by the limb protection. Weighted by
/// how much each layer is boosted.
pub fn hold_back_map(plane: &[f32], w: usize, h: usize, p: &SmartParams) -> (Vec<f32>, Vec<f32>) {
    let cutoff = cutoff_of(p, plane, w, h);
    let fp = p.peak * cutoff;
    let probe = SmartParams { strength: p.strength.max(0.35), ..p.clone() };
    let transform = transform(p.finest() * fp);
    let Some(d) = decompose(plane, w, h, &transform, &|| false) else { return (vec![0.0; plane.len()], vec![0.0; plane.len()]) };
    let protect = p.limb_protection.clamp(0.0, 1.0);
    let edges = if protect > 0.0 { limb_band(d.edges(), w, h, fp, p.limb_width).into_iter().map(|e| protect * e).collect() } else { vec![0.0; plane.len()] };
    if !p.adaptive {
        return (vec![0.0; plane.len()], edges);
    }
    let light = brightness(plane, w, h);
    let mut held = vec![0.0f32; plane.len()];
    let mut total = 0.0f32;
    for (j, layer) in d.layers.iter().enumerate() {
        let g = (probe.wanted(layer_peak(transform.scale(j), transform.ratio), cutoff, &Done::default()) - 1.0).max(0.0);
        if g <= 0.0 {
            continue;
        }
        total += g;
        let share = layer_share(layer, d.white_noise[j], &light, w, h, transform.scale(j));
        held.par_iter_mut().zip(share).for_each(|(o, s)| *o += g * s);
    }
    let zones = held.into_par_iter().map(|v| if total > 0.0 { 1.0 - v / total } else { 0.0 }).collect();
    (zones, edges)
}

/// The size of the detail present at each point (px): the periods of the layers averaged (in
/// log) by how far each one stands above the subject's smoothest zones there, so sharp zones,
/// whose excess sits on the fine layers, come out small, soft ones large; it varies smoothly.
/// 0 where nothing stands out (flat zones), negative on the limb (an edge against the sky,
/// not detail).
///
/// The reference is measured on the subject itself (`subject`, else the whole image), scale by
/// scale, on the energy over the brightness: a stack's noise is correlated, far stronger at
/// medium scales than white noise scaled from the finest plane would say, and photon noise
/// grows with brightness.
pub fn resolution_map(plane: &[f32], w: usize, h: usize, p: &SmartParams, subject: Option<&[f32]>) -> Vec<f32> {
    let cutoff = cutoff_of(p, plane, w, h);
    // Starting finer than the cut-off: some zones hold finer detail than the image overall.
    let transform = transform((1.5 * cutoff).min(0.5));
    let Some(d) = decompose(plane, w, h, &transform, &|| false) else { return vec![0.0; plane.len()] };
    // The limb against the sky holds energy at every scale without being detail: left out.
    let limb = limb_band(d.edges(), w, h, cutoff, 1.0);
    let stride = plane.len().div_ceil(1 << 16).max(1);
    let inside: Vec<usize> = match subject.filter(|m| m.len() == plane.len()) {
        Some(m) => (0..plane.len()).step_by(stride).filter(|&i| m[i] > 0.5 && limb[i] < 0.3).collect(),
        None => Vec::new(),
    };
    let sample: Vec<usize> = if inside.len() >= 64 { inside } else { (0..plane.len()).step_by(stride).collect() };
    let light = brightness(plane, w, h);
    // Σ excess and Σ excess · log period, per pixel.
    let (mut weight, mut sum) = (vec![0.0f32; plane.len()], vec![0.0f32; plane.len()]);
    for (j, layer) in d.layers.iter().enumerate() {
        let log_period = -layer_peak(transform.scale(j), transform.ratio).ln();
        let energy: Vec<f32> = layer.par_iter().zip(&light).map(|(v, b)| v * v / b).collect();
        let local = crate::mask::gaussian_approx(&energy, w, h, (2.0 * transform.scale(j)).max(2.0));
        let mut values: Vec<f32> = sample.iter().map(|&i| local[i]).collect();
        let k = values.len() / 5;
        let floor = values.select_nth_unstable_by(k, f32::total_cmp).1.max(1e-12);
        weight.par_iter_mut().zip(sum.par_iter_mut()).zip(local).for_each(|((wt, s), e)| {
            let excess = (e / floor - 2.0).max(0.0);
            *wt += excess;
            *s += excess * log_period;
        });
    }
    weight
        .into_par_iter()
        .zip(sum)
        .zip(limb)
        .map(|((wt, s), l)| if l >= 0.3 { -1.0 } else if wt < 0.5 { 0.0 } else { (s / wt).exp() })
        .collect()
}

/// What can be shown over the image about the smart sharpening.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SmartView {
    #[default]
    None,
    /// Where it holds back ([`hold_back_map`]) and the sky it leaves out.
    HoldBack,
    /// The finest detail present ([`resolution_map`]), on the subject only.
    Resolution,
}

/// A map computed for [`SmartView`], at the size of the image it was computed on.
pub enum SmartMap {
    HoldBack {
        zones: Vec<f32>,
        edges: Vec<f32>,
        /// Left out (sky), 0‥1; `None` when the sky is sharpened too.
        excluded: Option<Vec<f32>>,
    },
    /// Period of the finest detail (px), 0 where there is none, negative off the subject and on
    /// the limb.
    Resolution(Vec<f32>),
}

/// The map for `view` of `img` (the image entering the sharpening), with `subject` from
/// [`crate::mask::subject_and_moons`] when known.
pub fn view_map(view: SmartView, img: &Image, p: &SmartParams, subject: Option<&[f32]>) -> Option<SmartMap> {
    let (w, h) = (img.width(), img.height());
    let subject = subject.filter(|m| m.len() == w * h);
    let plane = working_planes(img, ColorMode::Luminance).swap_remove(0);
    match view {
        SmartView::None => None,
        SmartView::HoldBack => {
            let (zones, edges) = hold_back_map(&plane, w, h, p);
            let excluded = subject.filter(|_| !p.include_sky).map(|m| m.iter().map(|v| 1.0 - v.clamp(0.0, 1.0)).collect());
            Some(SmartMap::HoldBack { zones, edges, excluded })
        }
        SmartView::Resolution => {
            let mut map = resolution_map(&plane, w, h, p, subject);
            if let Some(m) = subject {
                map.par_iter_mut().zip(m).for_each(|(v, &s)| {
                    if s < 0.5 {
                        *v = -1.0;
                    }
                });
            }
            Some(SmartMap::Resolution(map))
        }
    }
}
