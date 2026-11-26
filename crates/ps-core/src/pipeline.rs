//! Non-destructive processing pipeline: a fixed sequence of optional modules plus masked
//! layers, each stage cached on (input, parameters) so only what changed is recomputed.
//!
//! ```text
//! source → RGB align → BlurXTerminator → deconvolution ─┬→ finishing ──────────┐
//!                                                       └→ layer k: finishing_k ┤ blend by mask
//!                                                                               └→ Mineral Moon → solar colour → out
//! finishing = noise reduction → wavelets → contrast → dering → levels/curves → colour
//! ```
//! Deconvolution is shared: the blur is the same everywhere. Each layer re-runs the finishing
//! chain on the deconvolved image, with its own settings for the modules it overrides and the
//! global ones for the rest, and is blended over the base through its mask. Deringing lifts the
//! ring the user painted over ([`RING_MASK`]) and uses the global settings.

use crate::i18n::tl;
use crate::Image;
use crate::adjust::{ColorParams, MineralParams, apply_color, apply_mineral};
use crate::align::{self, AlignParams};
use crate::blend::{self, BlendMode};
use crate::bxt::{self, BxtParams};
use crate::color::{merge_planes, working_planes};
use crate::deconv::{DeconvParams, DeconvReport, PlaneOptions, deconvolve_plane_with};
use crate::dering::{self, DeringParams};
use crate::enhance::{self, ContrastParams, NoiseParams, ToneParams};
use crate::mask::MaskSpec;
use crate::photocolor::PhotoColorParams;
use crate::psf::Psf;
use crate::smart::SmartParams;
use crate::suncolor::SunColorParams;
use crate::wavelet::{Decomposition, DecompositionKey, ScaleMode, WaveletParams, decompose, recompose};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Id under which [`Aux::masks`] carries deconvolution's anti-ringing area
/// ([`dering::weight`]), computed on the whole image like the layers' masks.
pub const DERING_MASK: u64 = u64::MAX;
/// Id under which [`Aux::masks`] carries where the user painted over the ring to lift
/// ([`dering::painted`]).
pub const RING_MASK: u64 = u64::MAX - 2;
/// Id under which [`Aux::masks`] carries where "Nitidezza intelligente" works when it leaves
/// the sky out ([`crate::mask::subject_and_moons`]).
pub const SMART_MASK: u64 = u64::MAX - 1;

/// Decompositions kept in memory (base wavelets + one layer with different transform settings).
const DECOMPOSITION_CACHE: usize = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Module<T> {
    pub enabled: bool,
    pub params: T,
}

impl<T: Default> Default for Module<T> {
    fn default() -> Self {
        Self::off(T::default())
    }
}

impl<T> Module<T> {
    pub fn on(params: T) -> Self {
        Self { enabled: true, params }
    }
    pub fn off(params: T) -> Self {
        Self { enabled: false, params }
    }
}

/// Layers with a dedicated card in the panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerRole {
    Rings,
    Moons,
}

/// A region of the image with its own finishing settings.
///
/// Two ways to work, by [`Layer::additive`]. A mask made by the user is additive, like an
/// adjustment layer: it starts with nothing and what is switched on is applied on top of the
/// result so far, laid over it in the layer's [`BlendMode`]. The older kind (the rings' and
/// moons' masks, and files saved before) redoes the finishing with its own settings instead of
/// the whole image's: there, a disabled sub-module means "same as the whole image".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub id: u64,
    pub name: String,
    pub enabled: bool,
    /// Set for the layers behind the "Anelli" and "Lune" cards.
    #[serde(default)]
    pub role: Option<LayerRole>,
    pub mask: MaskSpec,
    /// Works on top of the result so far instead of replacing the whole image's settings.
    #[serde(default)]
    pub additive: bool,
    /// How the adjusted region is laid over what is below it.
    #[serde(default)]
    pub blend: BlendMode,
    #[serde(default)]
    pub denoise: Module<NoiseParams>,
    pub wavelet: Module<WaveletParams>,
    #[serde(default)]
    pub contrast: Module<ContrastParams>,
    /// The final ("smart") sharpening of the layer's region.
    #[serde(default)]
    pub smart: Module<SmartParams>,
    #[serde(default)]
    pub tone: Module<ToneParams>,
    pub color: Module<ColorParams>,
}

impl Layer {
    /// Whether the layer changes anything compared with the whole image.
    pub fn overrides_any(&self) -> bool {
        self.denoise.enabled || self.wavelet.enabled || self.contrast.enabled || self.smart.enabled || self.tone.enabled || self.color.enabled
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Pipeline {
    pub align: Module<AlignParams>,
    pub bxt: Module<BxtParams>,
    pub deconv: Module<DeconvParams>,
    pub denoise: Module<NoiseParams>,
    pub wavelet: Module<WaveletParams>,
    /// "Nitidezza intelligente": sharpening shaped by the image, one or two controls.
    pub smart: Module<SmartParams>,
    pub contrast: Module<ContrastParams>,
    pub dering: Module<DeringParams>,
    /// Levels: black and white points, midtones (the steps' curves stay straight).
    pub tone: Module<ToneParams>,
    /// Curves, applied after the levels (the steps' levels stay neutral).
    pub curves: Module<ToneParams>,
    /// Jupiter's band colours carried onto a NASA map's (only when the subject is Jupiter).
    pub photocolor: Module<PhotoColorParams>,
    pub color: Module<ColorParams>,
    pub layers: Vec<Layer>,
    pub mineral: Module<MineralParams>,
    /// False colour for solar images (yellows, whites, H-alpha orange…).
    pub suncolor: Module<SunColorParams>,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self {
            align: Module::off(AlignParams::default()),
            bxt: Module::off(BxtParams::default()),
            deconv: Module::on(DeconvParams::default()),
            denoise: Module::on(NoiseParams::default()),
            wavelet: Module::on(WaveletParams::default()),
            smart: Module::on(SmartParams::default()),
            contrast: Module::off(ContrastParams::default()),
            dering: Module::off(DeringParams::default()),
            tone: Module::on(ToneParams::default()),
            curves: Module::on(ToneParams::default()),
            photocolor: Module::off(PhotoColorParams::default()),
            color: Module::on(ColorParams::default()),
            layers: Vec::new(),
            mineral: Module::off(MineralParams::default()),
            suncolor: Module::off(SunColorParams::neutral()),
        }
    }
}

impl Pipeline {
    /// New additive layer: no adjustment switched on, and each one starts from nothing (the
    /// wavelets keep the whole image's scales, with every amount at zero).
    pub fn add_layer(&mut self, name: &str, mask: MaskSpec) -> u64 {
        let id = self.layers.iter().map(|l| l.id).max().map_or(1, |m| m + 1);
        let mut wavelet = self.wavelet.params.clone();
        wavelet.layers.iter_mut().for_each(|l| *l = Default::default());
        (wavelet.amplify, wavelet.grain_denoise) = (1.0, 0.0);
        self.layers.push(Layer {
            id,
            name: name.to_string(),
            enabled: true,
            role: None,
            mask,
            additive: true,
            blend: BlendMode::Normal,
            denoise: Module::off(NoiseParams { strength: 0.0, chroma: 0, sky: 0.0, ..self.denoise.params.clone() }),
            wavelet: Module::off(wavelet),
            contrast: Module::off(ContrastParams { amount: 0.0, local_amount: 0.0, ..self.contrast.params.clone() }),
            smart: Module::off(SmartParams { strength: 0.0, ..self.smart.params.clone() }),
            tone: Module::off(ToneParams::default()),
            color: Module::off(ColorParams::default()),
        });
        id
    }

    pub fn layer_mut(&mut self, id: u64) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    /// The same processing for the image shrunk `factor` times (a power of two), for a quick
    /// approximate preview: every size in pixels shrinks with it. Wavelet scales move down a
    /// level per halving; detail finer than the reduced image can show is dropped.
    pub fn downscaled(&self, factor: usize) -> Pipeline {
        let s = 1.0 / factor as f32;
        let halvings = factor.max(1).trailing_zeros();
        let mut p = self.clone();
        p.align.params.red = p.align.params.red.map(|v| v * s);
        p.align.params.blue = p.align.params.blue.map(|v| v * s);
        p.bxt.params.psf_diameter *= s;
        p.deconv.params.psf = match p.deconv.params.psf.scaled(s) {
            Psf::Gaussian { sigma } => Psf::Gaussian { sigma: sigma.max(0.3) },
            Psf::Moffat { fwhm, beta } => Psf::Moffat { fwhm: fwhm.max(0.5), beta },
            other => other,
        };
        let wavelet = |w: &mut WaveletParams| {
            for _ in 0..halvings {
                if w.scale_mode == ScaleMode::Free {
                    w.first_scale /= 2.0;
                    if w.first_scale < 1.0 && w.layers.len() > 1 {
                        w.first_scale *= 2.0;
                        w.layers.remove(0);
                    }
                } else if w.base_step >= 2 {
                    w.base_step /= 2;
                } else if w.layers.len() > 1 {
                    w.layers.remove(0);
                }
            }
        };
        let contrast = |c: &mut ContrastParams| {
            c.radius = (c.radius * s).max(0.3);
            c.local_radius = (c.local_radius * s).max(2.0);
        };
        wavelet(&mut p.wavelet.params);
        contrast(&mut p.contrast.params);
        p.dering.params.radius = ((p.dering.params.radius as f32 * s).round() as u32).max(1);
        p.dering.params.width *= s;
        // Frequencies in cycles per pixel grow as the pixels do.
        if p.smart.params.cutoff > 0.0 {
            p.smart.params.cutoff = (p.smart.params.cutoff * factor as f32).min(0.5);
        }
        for layer in &mut p.layers {
            wavelet(&mut layer.wavelet.params);
            contrast(&mut layer.contrast.params);
            if layer.smart.params.cutoff > 0.0 {
                layer.smart.params.cutoff = (layer.smart.params.cutoff * factor as f32).min(0.5);
            }
        }
        p
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stage {
    Align,
    Bxt,
    Deconv,
    Denoise,
    Wavelet,
    Smart,
    Contrast,
    Dering,
    /// Levels (the name predates the split from the curves; kept for saved preferences).
    Tone,
    Curves,
    /// Jupiter's colours from a NASA map, before the general colour controls.
    PhotoColor,
    Color,
    Layers,
    Mineral,
    /// Panel cards for the dedicated Saturn-rings and moons layers (processed as layers).
    Rings,
    Moons,
    /// False colour for solar images.
    SunColor,
}

impl Stage {
    /// Every stage, in processing order.
    pub const ALL: [Stage; 17] = [
        Stage::Align,
        Stage::Bxt,
        Stage::Deconv,
        Stage::Denoise,
        Stage::Wavelet,
        Stage::Contrast,
        Stage::Smart,
        Stage::Dering,
        Stage::Tone,
        Stage::Curves,
        Stage::PhotoColor,
        Stage::Color,
        Stage::Layers,
        Stage::Rings,
        Stage::Moons,
        Stage::Mineral,
        Stage::SunColor,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Align => tl("Allineamento RGB"),
            Stage::Bxt => "BlurXTerminator",
            Stage::Deconv => tl("Deconvoluzione"),
            Stage::Denoise => tl("Riduzione rumore"),
            Stage::Wavelet => "Wavelet",
            Stage::Smart => tl("Nitidezza intelligente"),
            Stage::Contrast => tl("Contrasto"),
            Stage::Dering => "Dering",
            Stage::Tone => tl("Livelli tonali"),
            Stage::Curves => tl("Curve"),
            Stage::PhotoColor => tl("Colore fotometrico"),
            Stage::Color => tl("Colore"),
            Stage::Layers => tl("Maschere"),
            Stage::Mineral => "Mineral Moon",
            Stage::Rings => tl("Anelli di Saturno"),
            Stage::Moons => tl("Lune"),
            Stage::SunColor => tl("Colorazione solare"),
        }
    }
}

/// Data the pipeline needs besides the image, already cropped to the processed region.
#[derive(Default)]
pub struct Aux<'a> {
    /// Masks of the layers (by layer id), each with a generation identifying its content.
    pub masks: &'a [(u64, Arc<Vec<f32>>, u64)],
    /// External luminance for LRGB, with its generation.
    pub luminance: Option<(Arc<Vec<f32>>, u64)>,
    pub bxt_executable: Option<&'a Path>,
    /// Also return the ring found under the brush ([`dering::Ring::map`]).
    pub ringing_map: bool,
    /// Also return this map of the smart sharpening, computed on the image entering it.
    pub smart_view: crate::smart::SmartView,
    /// Text and marks laid over the finished image (premultiplied red, green, blue and coverage
    /// planes, each with a generation).
    pub overlay: Option<OverlayPlanes>,
    /// The guided noise reduction's measures of the whole image and where the source sits in it
    /// (a zoomed region), so the region is reduced as the whole image is. Without it they are
    /// measured on the source.
    pub noise_plan: Option<(Arc<crate::noise_net::NoisePlan>, (usize, usize))>,
    /// The smart sharpening's measures of the whole image ([`RunOutput::smart_reference`] of a
    /// whole-image run with the same earlier steps), with a generation, for a zoomed region.
    pub smart_reference: Option<(Arc<RegionReference>, u64)>,
    /// Where the source sits in the whole image (a zoomed region; (0, 0) for the whole image).
    pub offset: (usize, usize),
}

/// What a whole-image run measured that a zoomed region of it should reuse, so it is processed
/// as the whole image is: the noise of each wavelet layer (on a region the finest layer holds
/// the subject's detail, which would count as noise), and the smart sharpening's measures.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RegionReference {
    /// Per working plane: the deconvolution's iterations and measured noise.
    pub deconv: Option<Vec<(u32, f32)>>,
    /// The noise reduction's levels (not for the guided one, which has its plan).
    pub denoise: Option<enhance::NoiseLevels>,
    /// Per working plane: each layer's noise, its white-noise version and the grain's noise.
    pub wavelet: Option<Vec<WaveletNoise>>,
    pub smart: Option<crate::smart::SmartReference>,
}

/// One plane's wavelet noise (see [`crate::wavelet::Decomposition`]).
#[derive(Clone, Debug, PartialEq)]
pub struct WaveletNoise {
    pub noise: Vec<f32>,
    pub white_noise: Vec<f32>,
    pub grain: Option<f32>,
    /// What its edge protection measured edges against ([`crate::wavelet::Decomposition::edge_levels`]).
    pub edges: Option<(f32, f32)>,
}

/// Premultiplied red, green, blue and coverage planes of a layer of marks, at the image's size.
pub type OverlayPlanes = [(Arc<Vec<f32>>, u64); 4];

/// `img` with the marks laid over it: `out = img · (1 − coverage) + colour`. A coloured layer
/// over a mono picture makes it RGB (otherwise the colour would be averaged to grey).
pub fn apply_overlay(img: &Image, o: &OverlayPlanes) -> Image {
    let a = &o[3].0;
    let n = a.len();
    let coloured = (0..n).any(|i| a[i] > 0.0 && ((o[0].0[i] - o[1].0[i]).abs() > 0.004 || (o[1].0[i] - o[2].0[i]).abs() > 0.004));
    let mut out = if img.is_mono() && coloured {
        let plane = img.channels()[0].clone();
        Image::new(img.width(), img.height(), vec![plane.clone(), plane.clone(), plane], img.source_bit_depth)
    } else {
        img.clone()
    };
    let mono = out.is_mono();
    for (c, ch) in out.channels_mut().iter_mut().enumerate() {
        for (i, v) in ch.iter_mut().enumerate().take(n) {
            let p = if mono { (o[0].0[i] + o[1].0[i] + o[2].0[i]) / 3.0 } else { o[c.min(2)].0[i] };
            *v = *v * (1.0 - a[i]) + p;
        }
    }
    out
}

/// Whether the pipeline uses deconvolution's anti-ringing area.
pub fn needs_dering_area(p: &Pipeline) -> bool {
    p.deconv.enabled && !p.deconv.params.is_neutral() && p.deconv.params.deringing > 0.0
}

/// Whether the pipeline lifts a painted ring.
pub fn needs_ring_area(p: &Pipeline) -> bool {
    p.dering.enabled && !p.dering.params.is_neutral()
}

/// Whether the smart sharpening needs the subject's mask: it is on and leaves the sky out.
pub fn needs_smart_mask(p: &Pipeline) -> bool {
    p.smart.enabled && !p.smart.params.is_neutral() && !p.smart.params.include_sky
}

/// The full-size masks a pipeline needs for `source` (for [`Aux::masks`]): the enabled layers'
/// and, when used, the anti-ringing area and the painted ring; the subject when the smart
/// sharpening leaves the sky out.
pub fn masks_for(p: &Pipeline, source: &Image) -> Vec<(u64, Arc<Vec<f32>>, u64)> {
    let mut masks: Vec<_> = p.layers.iter().filter(|l| l.enabled).map(|l| (l.id, Arc::new(crate::mask::render(&l.mask, source)), next_generation())).collect();
    if needs_dering_area(p) {
        masks.push((DERING_MASK, Arc::new(dering::weight(source, &p.dering.params)), next_generation()));
    }
    if needs_ring_area(p) {
        masks.push((RING_MASK, Arc::new(dering::area(&p.dering.params, source)), next_generation()));
    }
    if needs_smart_mask(p) {
        masks.push((SMART_MASK, Arc::new(crate::mask::subject_and_moons(source)), next_generation()));
    }
    masks
}

/// Whether the guided noise reduction may run (on the whole image or in a layer): it needs
/// [`Aux::noise_plan`] for a zoomed region.
pub fn uses_guided_denoise(p: &Pipeline) -> bool {
    let guided = |d: &NoiseParams| d.method == crate::enhance::NoiseMethod::Guided;
    guided(&p.denoise.params) || p.layers.iter().any(|l| guided(&l.denoise.params))
}

/// Unique identity of an image flowing through the pipeline; equal generations mean equal pixels.
pub fn next_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Debug)]
pub struct StageTiming {
    pub stage: Stage,
    pub elapsed: Duration,
    pub cached: bool,
}

pub struct RunOutput {
    pub image: Arc<Image>,
    pub timings: Vec<StageTiming>,
    /// Problems that did not stop processing (the stage was skipped).
    pub warnings: Vec<(Stage, String)>,
    /// What the levels' current step works on (for its histogram and the black/white-point
    /// picker).
    pub tone_input: Arc<Image>,
    /// What the curves' current step works on (for the histogram behind the curve).
    pub curves_input: Arc<Image>,
    /// What the deconvolution did (when it ran).
    pub deconv: Option<DeconvReport>,
    /// The ring found under the brush, when asked for ([`Aux::ringing_map`]).
    pub ringing: Option<Arc<Vec<f32>>>,
    /// The smart sharpening's map asked for ([`Aux::smart_view`]).
    pub smart_map: Option<Arc<crate::smart::SmartMap>>,
    /// What the wavelets and the smart sharpening measured, when they ran without
    /// [`Aux::smart_reference`].
    pub smart_reference: Option<Arc<RegionReference>>,
}

struct Slot<K> {
    input_gen: u64,
    key: K,
    output: Arc<Image>,
    output_gen: u64,
}

type Frame = (Arc<Image>, u64);

fn lookup<K: PartialEq>(slot: &Option<Slot<K>>, input_gen: u64, key: &K) -> Option<Frame> {
    slot.as_ref().filter(|s| s.input_gen == input_gen && s.key == *key).map(|s| (Arc::clone(&s.output), s.output_gen))
}

fn store<K>(slot: &mut Option<Slot<K>>, input_gen: u64, key: K, image: Image) -> Frame {
    let output = Arc::new(image);
    let output_gen = next_generation();
    *slot = Some(Slot { input_gen, key, output: Arc::clone(&output), output_gen });
    (output, output_gen)
}

/// The smart sharpening's preparation, reused while only its strength or limb protection change.
struct SmartCache {
    input_gen: u64,
    before_gen: u64,
    /// Generation of the whole image's measures used (0: measured on the input).
    reference_gen: u64,
    key: crate::smart::PrepareKey,
    prepared: Arc<crate::smart::Prepared>,
}

/// What a smart sharpening map depends on: view, input and subject generations, settings.
type SmartMapKey = (crate::smart::SmartView, u64, u64, SmartParams);

struct WaveletCache {
    input_gen: u64,
    /// Generation of the noise measures given in place of the decomposition's own (0: its own).
    noise_gen: u64,
    key: DecompositionKey,
    planes: Vec<Vec<f32>>,
    decompositions: Vec<Decomposition>,
}

/// What the ring under the brush depends on: image and painted-area generations, forced width
/// bits.
type RingKey = (u64, u64, u32);

/// The ring found under the brush, with its generation (`None`: nothing painted).
type FoundRing = Option<(Arc<dering::Ring>, u64)>;

/// Effective finishing settings of a layer (`None` = module not applied).
#[derive(Clone, PartialEq)]
struct LayerKey {
    denoise: Option<NoiseParams>,
    wavelet: Option<WaveletParams>,
    contrast: Option<ContrastParams>,
    /// With the generation of the subject mask it leaves the sky out with (0: none).
    smart: Option<(SmartParams, u64)>,
    /// With the generation of the painted area.
    dering: Option<(DeringParams, u64)>,
    tone: Option<ToneParams>,
    color: Option<ColorParams>,
}

/// The layer's own settings if it overrides the module, the whole image's otherwise.
fn own_or<T: Clone>(own: &Module<T>, base: &Option<T>) -> Option<T> {
    if own.enabled { Some(own.params.clone()) } else { base.clone() }
}

/// What a blend step was computed from: the base's and the layer's generations, and the mode.
type BlendKey = (u64, u64, BlendMode);

/// Runs pipelines, reusing every stage whose input and parameters did not change.
#[derive(Default)]
pub struct Processor {
    /// Keep only the expensive stages cached (deconvolution, BlurXTerminator, noise
    /// reduction, one wavelet decomposition, layers); the quick ones are recomputed.
    pub memory_saving: bool,
    align: Option<Slot<AlignParams>>,
    bxt: Option<Slot<BxtParams>>,
    deconv: Option<Slot<(DeconvParams, u64, u64)>>,
    /// The last deconvolution's iterations and noise of each plane, when measured on its own.
    deconv_planes: Option<Vec<(u32, f32)>>,
    /// What the last deconvolution computed reported.
    deconv_report: Option<DeconvReport>,
    /// The ring found under the brush ([`dering::find_ring`]), with a generation.
    ring: Option<(RingKey, FoundRing)>,
    denoise: Option<Slot<(NoiseParams, u64)>>,
    /// What the last noise reduction measured on its own (see [`enhance::NoiseLevels`]).
    denoise_levels: Option<enhance::NoiseLevels>,
    wavelet: Option<Slot<(WaveletParams, u64)>>,
    smart: Option<Slot<(SmartParams, u64, u64, u64)>>,
    smart_cache: Option<SmartCache>,
    smart_map: Option<(SmartMapKey, Arc<crate::smart::SmartMap>)>,
    contrast: Option<Slot<ContrastParams>>,
    dering: Option<Slot<(DeringParams, u64)>>,
    tone: Option<Slot<ToneParams>>,
    color: Option<Slot<ColorParams>>,
    photocolor: Option<Slot<PhotoColorParams>>,
    layers: HashMap<u64, Option<Slot<LayerKey>>>,
    blend: Option<Slot<Vec<BlendKey>>>,
    /// The additive layers: their adjusted image, and that laid over what is below.
    add_layers: HashMap<u64, Option<Slot<LayerKey>>>,
    add_blend: HashMap<u64, Option<Slot<BlendKey>>>,
    mineral: Option<Slot<(MineralParams, u64)>>,
    suncolor: Option<Slot<SunColorParams>>,
    gamut: Option<Slot<()>>,
    decompositions: Vec<WaveletCache>,
    /// The guided noise reduction's measures of the source (by its generation).
    noise_plan: Option<(u64, Option<Arc<crate::noise_net::NoisePlan>>)>,
}

impl Processor {
    /// `progress(stage, fraction)`. Returns `None` if cancelled; finished stages stay cached.
    pub fn run(&mut self, source: Arc<Image>, source_gen: u64, p: &Pipeline, aux: &Aux, cancel: &dyn Fn() -> bool, progress: &dyn Fn(Stage, f32)) -> Option<RunOutput> {
        let mut timings = Vec::new();
        let mut warnings = Vec::new();
        let mut frame: Frame = (source, source_gen);
        // Deconvolution's anti-ringing area and the painted ring, if provided.
        let dering_area = aux.masks.iter().find(|(id, ..)| *id == DERING_MASK).map(|(_, plane, generation)| (Arc::clone(plane), *generation));
        let ring_area = aux.masks.iter().find(|(id, ..)| *id == RING_MASK).map(|(_, plane, generation)| (Arc::clone(plane), *generation));
        let colour = !frame.0.is_mono();
        // The guided noise reduction's measures: given (a zoomed region's whole image), or of
        // the source, measured once.
        let noise_plan = if !uses_guided_denoise(p) {
            None
        } else if let Some((plan, offset)) = &aux.noise_plan {
            Some((Arc::clone(plan), *offset))
        } else {
            if !matches!(&self.noise_plan, Some((g, _)) if *g == source_gen) {
                self.noise_plan = Some((source_gen, crate::noise_net::plan(&frame.0).map(Arc::new)));
            }
            self.noise_plan.as_ref().and_then(|(_, plan)| plan.clone()).map(|plan| (plan, (0, 0)))
        };
        let noise_plan = noise_plan.as_ref().map(|(plan, offset)| (plan.as_ref(), *offset));

        // Look the stage up in its slot; compute and store it on a miss.
        macro_rules! stage {
            ($stage:expr, $slot:expr, $key:expr, $compute:expr) => {{
                let start = Instant::now();
                let key = $key;
                frame = match lookup(&$slot, frame.1, &key) {
                    Some(hit) => {
                        timings.push(StageTiming { stage: $stage, elapsed: start.elapsed(), cached: true });
                        hit
                    }
                    None => {
                        let input_gen = frame.1;
                        let image: Image = $compute(&*frame.0)?;
                        timings.push(StageTiming { stage: $stage, elapsed: start.elapsed(), cached: false });
                        store(&mut $slot, input_gen, key, image)
                    }
                };
            }};
        }

        if p.align.enabled && colour && !p.align.params.is_identity() {
            stage!(Stage::Align, self.align, p.align.params, |img: &Image| Some(align::apply(img, &p.align.params)));
        }
        // What deringing compares with: the image before any sharpening.
        let unsharpened = frame.clone();

        if p.bxt.enabled {
            match aux.bxt_executable {
                None => warnings.push((Stage::Bxt, tl("eseguibile rc-astro non configurato nelle Impostazioni").to_string())),
                Some(exe) => {
                    let start = Instant::now();
                    if let Some(hit) = lookup(&self.bxt, frame.1, &p.bxt.params) {
                        timings.push(StageTiming { stage: Stage::Bxt, elapsed: start.elapsed(), cached: true });
                        frame = hit;
                    } else {
                        let report = |f: f32| progress(Stage::Bxt, f);
                        match bxt::run_passes(&frame.0, &p.bxt.params, exe, cancel, &report) {
                            Ok(Some(out)) => {
                                timings.push(StageTiming { stage: Stage::Bxt, elapsed: start.elapsed(), cached: false });
                                frame = store(&mut self.bxt, frame.1, p.bxt.params.clone(), out);
                            }
                            Ok(None) => return None,
                            // Not cached, so it is retried once the configuration is fixed.
                            Err(e) => warnings.push((Stage::Bxt, e)),
                        }
                    }
                }
            }
        }

        let mut deconv_measures: Option<Vec<(u32, f32)>> = None;
        if p.deconv.enabled && !p.deconv.params.is_neutral() {
            let report = |f: f32| progress(Stage::Deconv, f);
            let area_gen = dering_area.as_ref().filter(|_| p.deconv.params.deringing > 0.0).map_or(0, |a| a.1);
            let area = dering_area.as_ref().map(|(plane, _)| plane.as_slice());
            let reported = &mut self.deconv_report;
            let measured = aux.smart_reference.as_ref().and_then(|(r, g)| r.deconv.as_ref().map(|m| (m.as_slice(), *g)));
            let measured_gen = measured.map_or(0, |m| m.1);
            let each_plane = &mut self.deconv_planes;
            stage!(Stage::Deconv, self.deconv, (p.deconv.params.clone(), area_gen, measured_gen), |img: &Image| {
                let (out, report, each) = run_deconvolution(img, &p.deconv.params, area, measured.map(|m| m.0), cancel, &report)?;
                *reported = Some(report);
                *each_plane = (measured.is_none()).then_some(each);
                Some(out)
            });
            if aux.smart_reference.is_none() {
                deconv_measures = self.deconv_planes.clone();
            }
        }
        let deconvolved = frame.clone();

        let mut denoise_measures: Option<enhance::NoiseLevels> = None;
        let base_denoise = Some(p.denoise.params.clone()).filter(|d| p.denoise.enabled && !d.is_neutral());
        if let Some(d) = &base_denoise {
            let measured = aux.smart_reference.as_ref().and_then(|(r, g)| r.denoise.as_ref().map(|l| (l, *g)));
            let measured_gen = measured.map_or(0, |m| m.1);
            let levels = &mut self.denoise_levels;
            stage!(Stage::Denoise, self.denoise, (d.clone(), measured_gen), |img: &Image| {
                let (out, used) = enhance::denoise_measured(img, d, noise_plan, measured.map(|m| m.0));
                *levels = used.filter(|_| measured.is_none());
                Some(out)
            });
            if aux.smart_reference.is_none() {
                denoise_measures = self.denoise_levels.clone();
            }
        }
        let denoised = frame.clone();

        let base_wavelet = Some(p.wavelet.params.clone()).filter(|w| p.wavelet.enabled && !w.is_neutral());
        let decompositions = if self.memory_saving { 1 } else { DECOMPOSITION_CACHE };
        let mut wavelet_measures: Option<Vec<WaveletNoise>> = None;
        if let Some(w) = &base_wavelet {
            let caches = &mut self.decompositions;
            let measured = aux.smart_reference.as_ref().and_then(|(r, g)| r.wavelet.as_ref().map(|n| (n.as_slice(), *g)));
            let noise_gen = measured.map_or(0, |m| m.1);
            stage!(Stage::Wavelet, self.wavelet, (w.clone(), noise_gen), |img: &Image| { apply_wavelet(caches, decompositions, img, denoised.1, w, measured, cancel) });
            if aux.smart_reference.is_none() {
                wavelet_measures = self.decompositions.iter().rev().find(|c| c.input_gen == denoised.1 && c.key == w.decomposition_key() && c.noise_gen == 0).map(|c| {
                    c.decompositions.iter().map(|d| WaveletNoise { noise: d.noise.clone(), white_noise: d.white_noise.clone(), grain: d.grain.as_ref().map(|g| g.1), edges: (w.edge_protection > 0.0).then(|| d.measure_edge_levels()) }).collect()
                });
            }
        }

        // Before the smart sharpening, which works on everything sharpened so far.
        let base_contrast = Some(p.contrast.params.clone()).filter(|c| p.contrast.enabled && !c.is_neutral());
        if let Some(c) = &base_contrast {
            stage!(Stage::Contrast, self.contrast, c.clone(), |img: &Image| { Some(enhance::contrast(img, c)) });
        }

        // Shaped by what earlier steps already did: compared with the image before them.
        let smart_input = frame.clone();
        let mut smart_measures = None;
        let subject = aux.masks.iter().find(|(id, ..)| *id == SMART_MASK).map(|(_, plane, generation)| (Arc::clone(plane), *generation));
        let base_smart = Some(p.smart.params.clone()).filter(|s| p.smart.enabled && !s.is_neutral());
        if let Some(s) = base_smart.clone() {
            // Nothing sharpened before it: no need to compare.
            let before = (unsharpened.1 != frame.1).then(|| Arc::clone(&unsharpened.0));
            let mask = subject.as_ref().filter(|_| !s.include_sky);
            let mask_gen = mask.map_or(0, |m| m.1);
            let (cache, keep) = (&mut self.smart_cache, !self.memory_saving);
            let (input_gen, before_gen) = (frame.1, unsharpened.1);
            let reference = aux.smart_reference.as_ref();
            let reference_gen = reference.map_or(0, |r| r.1);
            let mut made = None;
            stage!(Stage::Smart, self.smart, (s.clone(), unsharpened.1, mask_gen, reference_gen), |img: &Image| {
                let key = s.prepare_key();
                let prepared = match cache.as_ref().filter(|c| c.input_gen == input_gen && c.before_gen == before_gen && c.reference_gen == reference_gen && c.key == key) {
                    Some(c) => Arc::clone(&c.prepared),
                    None => {
                        *cache = None;
                        let prepared = Arc::new(crate::smart::prepare_with(img, before.as_deref(), &s, reference.and_then(|r| r.0.smart.as_ref()), aux.offset));
                        if keep {
                            *cache = Some(SmartCache { input_gen, before_gen, reference_gen, key, prepared: Arc::clone(&prepared) });
                        }
                        prepared
                    }
                };
                made = Some(Arc::clone(&prepared));
                Some(crate::smart::apply(&prepared, img, &s, mask.map(|m| m.0.as_slice())))
            });
            if reference.is_none() {
                let prepared = made.or_else(|| self.smart_cache.as_ref().filter(|c| c.input_gen == input_gen && c.before_gen == before_gen && c.reference_gen == 0).map(|c| Arc::clone(&c.prepared)));
                smart_measures = prepared.map(|pr| pr.reference(&s));
            }
        }
        if aux.smart_view != crate::smart::SmartView::None {
            // The resolution doesn't depend on the settings that only say how much to sharpen.
            let settings = match aux.smart_view {
                crate::smart::SmartView::Resolution => SmartParams { cutoff: p.smart.params.cutoff, ..Default::default() },
                _ => p.smart.params.clone(),
            };
            let key = (aux.smart_view, smart_input.1, subject.as_ref().map_or(0, |m| m.1), settings);
            if self.smart_map.as_ref().is_none_or(|(k, _)| *k != key) {
                let map = crate::smart::view_map(aux.smart_view, &smart_input.0, &p.smart.params, subject.as_ref().map(|m| m.0.as_slice()));
                self.smart_map = map.map(|m| (key, Arc::new(m)));
            }
        }

        let base_dering = Some(p.dering.params.clone()).filter(|d| p.dering.enabled && !d.is_neutral());
        // The ring under the brush, found once for the deringing and its map.
        let ring = match ring_area.as_ref().filter(|_| base_dering.is_some() || aux.ringing_map) {
            Some((area, area_gen)) => {
                let key: RingKey = (frame.1, *area_gen, p.dering.params.width.to_bits());
                if self.ring.as_ref().is_none_or(|(k, _)| *k != key) {
                    let found = dering::find_ring(&frame.0, area, p.dering.params.width).map(|r| (Arc::new(r), next_generation()));
                    self.ring = Some((key, found));
                }
                self.ring.as_ref().and_then(|(_, r)| r.clone())
            }
            None => None,
        };
        let ring_map = ring.as_ref().filter(|_| aux.ringing_map).map(|(r, _)| Arc::new(r.map.clone()));
        let area = ring_area.as_ref().map(|(plane, _)| plane.as_slice());
        if let (Some(d), Some(area)) = (&base_dering, area) {
            let ring_gen = ring.as_ref().map_or(0, |r| r.1);
            let found = ring.as_ref().map(|(r, _)| r.as_ref());
            stage!(Stage::Dering, self.dering, (d.clone(), ring_gen), |img: &Image| Some(dering::dering(img, d, area, found)));
        }

        let levels = Some(p.tone.params.clone()).filter(|t| p.tone.enabled && !t.is_neutral());
        let curves = Some(p.curves.params.clone()).filter(|t| p.curves.enabled && !t.is_neutral());
        let tone_input = match &levels {
            Some(l) if !l.applied.is_empty() => Arc::new(enhance::tone_base(&frame.0, l)),
            _ => Arc::clone(&frame.0),
        };
        let curves_input = match (&levels, &curves) {
            (None, None) => Arc::clone(&frame.0),
            (l, c) => {
                let before = ToneParams::chain(l.as_ref().unwrap_or(&ToneParams::default()), &ToneParams { current: Default::default(), ..c.clone().unwrap_or_default() });
                if before.is_neutral() { Arc::clone(&frame.0) } else { Arc::new(enhance::tone(&frame.0, &before)) }
            }
        };
        // Levels then curves, folded into one table.
        let base_tone = match (&levels, &curves) {
            (None, None) => None,
            (l, c) => Some(ToneParams::chain(l.as_ref().unwrap_or(&ToneParams::default()), c.as_ref().unwrap_or(&ToneParams::default()))),
        };
        if let Some(t) = &base_tone {
            stage!(Stage::Tone, self.tone, t.clone(), |img: &Image| Some(enhance::tone(img, t)));
        }

        if p.photocolor.enabled && colour && !p.photocolor.params.is_neutral() {
            let params = p.photocolor.params.clone();
            match crate::photocolor::analyse(&frame.0, &params) {
                Some(fit) => stage!(Stage::PhotoColor, self.photocolor, params.clone(), |img: &Image| Some(crate::photocolor::apply_fit(img, &fit, params.strength))),
                None => warnings.push((Stage::PhotoColor, tl("non trovo Giove con le sue fasce: lascio i colori com'erano").to_string())),
            }
        }

        let base_color = Some(p.color.params.clone()).filter(|c| p.color.enabled && colour && !c.is_neutral());
        if let Some(c) = &base_color {
            stage!(Stage::Color, self.color, c.clone(), |img: &Image| Some(apply_color(img, c)));
        }

        // ---- layers ----
        let start = Instant::now();
        let mut recomputed = false;
        let mut parts: Vec<(Frame, Arc<Vec<f32>>, u64, BlendMode)> = Vec::new();
        self.layers.retain(|id, _| p.layers.iter().any(|l| l.id == *id));
        self.add_layers.retain(|id, _| p.layers.iter().any(|l| l.id == *id));
        self.add_blend.retain(|id, _| p.layers.iter().any(|l| l.id == *id));
        for layer in p.layers.iter().filter(|l| l.enabled && !l.additive && l.overrides_any()) {
            let Some((_, mask, mask_gen)) = aux.masks.iter().find(|(id, ..)| *id == layer.id) else {
                continue;
            };
            let key = LayerKey {
                denoise: own_or(&layer.denoise, &base_denoise),
                wavelet: own_or(&layer.wavelet, &base_wavelet),
                contrast: own_or(&layer.contrast, &base_contrast),
                smart: own_or(&layer.smart, &base_smart).filter(|s| !s.is_neutral()).map(|s| {
                    let mask_gen = subject.as_ref().filter(|_| !s.include_sky).map_or(0, |m| m.1);
                    (s, mask_gen)
                }),
                dering: base_dering.clone().filter(|_| area.is_some()).map(|d| (d, ring_area.as_ref().map_or(0, |a| a.1))),
                tone: own_or(&layer.tone, &base_tone),
                color: if colour { own_or(&layer.color, &base_color) } else { None },
            };
            let slot = self.layers.entry(layer.id).or_default();
            let out = match lookup(slot, deconvolved.1, &key) {
                Some(hit) => hit,
                None => {
                    recomputed = true;
                    // Reuse the whole image's denoised frame (and its wavelet decomposition) when
                    // the layer doesn't change the noise reduction.
                    let (mut img, generation) = if key.denoise == base_denoise {
                        ((*denoised.0).clone(), denoised.1)
                    } else {
                        let img = key.denoise.as_ref().map_or_else(|| (*deconvolved.0).clone(), |d| enhance::denoise_planned(&deconvolved.0, d, noise_plan));
                        (img, next_generation())
                    };
                    if let Some(w) = &key.wavelet {
                        img = apply_wavelet(&mut self.decompositions, decompositions, &img, generation, w, None, cancel)?;
                    }
                    if let Some(c) = &key.contrast {
                        img = enhance::contrast(&img, c);
                    }
                    if let Some((sp, _)) = &key.smart {
                        let before = (img.width(), img.height()) == (unsharpened.0.width(), unsharpened.0.height());
                        let prepared = crate::smart::prepare(&img, before.then_some(&*unsharpened.0), sp);
                        let mask = subject.as_ref().filter(|m| !sp.include_sky && m.0.len() == img.width() * img.height());
                        img = crate::smart::apply(&prepared, &img, sp, mask.map(|m| m.0.as_slice()));
                    }
                    if let (Some((d, _)), Some(area)) = (&key.dering, area) {
                        img = dering::dering(&img, d, area, None);
                    }
                    if let Some(t) = &key.tone {
                        img = enhance::tone(&img, t);
                    }
                    if let Some(c) = &key.color {
                        img = apply_color(&img, c);
                    }
                    store(slot, deconvolved.1, key, img)
                }
            };
            parts.push((out, Arc::clone(mask), *mask_gen, layer.blend));
        }
        let mut any_layer = !parts.is_empty();
        if !parts.is_empty() {
            let key: Vec<(u64, u64, BlendMode)> = parts.iter().map(|(f, _, mask_gen, mode)| (f.1, *mask_gen, *mode)).collect();
            frame = match lookup(&self.blend, frame.1, &key) {
                Some(hit) => hit,
                None => {
                    recomputed = true;
                    let input_gen = frame.1;
                    store(&mut self.blend, input_gen, key, blend(&frame.0, &parts))
                }
            };
        }
        // The additive layers, in the order of the list, each over what is below it.
        for layer in p.layers.iter().filter(|l| l.enabled && l.additive && l.overrides_any()) {
            let Some((_, mask, mask_gen)) = aux.masks.iter().find(|(id, ..)| *id == layer.id) else {
                continue;
            };
            let own = |m: bool, neutral: bool| m && !neutral;
            let key = LayerKey {
                denoise: Some(layer.denoise.params.clone()).filter(|d| own(layer.denoise.enabled, d.is_neutral())),
                wavelet: Some(layer.wavelet.params.clone()).filter(|w| own(layer.wavelet.enabled, w.is_neutral())),
                contrast: Some(layer.contrast.params.clone()).filter(|c| own(layer.contrast.enabled, c.is_neutral())),
                smart: Some(layer.smart.params.clone()).filter(|s| own(layer.smart.enabled, s.is_neutral())).map(|s| {
                    let mask_gen = subject.as_ref().filter(|_| !s.include_sky).map_or(0, |m| m.1);
                    (s, mask_gen)
                }),
                dering: None,
                tone: Some(layer.tone.params.clone()).filter(|t| own(layer.tone.enabled, t.is_neutral())),
                color: Some(layer.color.params.clone()).filter(|c| colour && own(layer.color.enabled, c.is_neutral())),
            };
            if key.denoise.is_none() && key.wavelet.is_none() && key.contrast.is_none() && key.smart.is_none() && key.tone.is_none() && key.color.is_none() {
                continue;
            }
            any_layer = true;
            let slot = self.add_layers.entry(layer.id).or_default();
            let adjusted = match lookup(slot, frame.1, &key) {
                Some(hit) => hit,
                None => {
                    recomputed = true;
                    let (mut img, generation) = match &key.denoise {
                        Some(d) => (enhance::denoise_planned(&frame.0, d, noise_plan), next_generation()),
                        None => ((*frame.0).clone(), frame.1),
                    };
                    if let Some(w) = &key.wavelet {
                        img = apply_wavelet(&mut self.decompositions, decompositions, &img, generation, w, None, cancel)?;
                    }
                    if let Some(c) = &key.contrast {
                        img = enhance::contrast(&img, c);
                    }
                    if let Some((sp, _)) = &key.smart {
                        let prepared = crate::smart::prepare(&img, None, sp);
                        let subject_mask = subject.as_ref().filter(|m| !sp.include_sky && m.0.len() == img.width() * img.height());
                        img = crate::smart::apply(&prepared, &img, sp, subject_mask.map(|m| m.0.as_slice()));
                    }
                    if let Some(t) = &key.tone {
                        img = enhance::tone(&img, t);
                    }
                    if let Some(c) = &key.color {
                        img = apply_color(&img, c);
                    }
                    store(slot, frame.1, key, img)
                }
            };
            let blend_key = (adjusted.1, *mask_gen, layer.blend);
            let slot = self.add_blend.entry(layer.id).or_default();
            frame = match lookup(slot, frame.1, &blend_key) {
                Some(hit) => hit,
                None => {
                    recomputed = true;
                    let input_gen = frame.1;
                    store(slot, input_gen, blend_key, blend::composite(&frame.0, &adjusted.0, mask, layer.blend))
                }
            };
        }
        if any_layer {
            timings.push(StageTiming { stage: Stage::Layers, elapsed: start.elapsed(), cached: !recomputed });
        }

        if p.mineral.enabled && (!p.mineral.params.is_neutral() || aux.luminance.is_some()) {
            let (lum, lum_gen) = match &aux.luminance {
                Some((plane, g)) if plane.len() == frame.0.width() * frame.0.height() => (Some(plane.as_slice()), *g),
                Some(_) => {
                    warnings.push((Stage::Mineral, tl("la luminanza LRGB ha dimensioni diverse dall'immagine").to_string()));
                    (None, 0)
                }
                None => (None, 0),
            };
            stage!(Stage::Mineral, self.mineral, (p.mineral.params.clone(), lum_gen), |img: &Image| Some(apply_mineral(img, &p.mineral.params, lum)));
        }

        if p.suncolor.enabled && !p.suncolor.params.is_neutral() {
            stage!(Stage::SunColor, self.suncolor, p.suncolor.params.clone(), |img: &Image| Some(crate::suncolor::apply(img, &p.suncolor.params)));
        }

        // Colours pushed outside 0‥1 are brought back by desaturating, not by clipping channels.
        if crate::color::out_of_gamut(&frame.0) {
            frame = match lookup(&self.gamut, frame.1, &()) {
                Some(hit) => hit,
                None => {
                    let input_gen = frame.1;
                    store(&mut self.gamut, input_gen, (), crate::color::fit_gamut_image(&frame.0))
                }
            };
        }
        // The names and marks go on last, over everything.
        if let Some(o) = aux.overlay.as_ref().filter(|o| o[3].0.len() == frame.0.width() * frame.0.height()) {
            frame = (Arc::new(apply_overlay(&frame.0, o)), next_generation());
        }
        if self.memory_saving {
            self.drop_quick_stages();
        }
        let deconv = if p.deconv.enabled && !p.deconv.params.is_neutral() { self.deconv_report } else { None };
        let ringing = ring_map;
        let smart_map = if aux.smart_view != crate::smart::SmartView::None { self.smart_map.as_ref().map(|(_, m)| Arc::clone(m)) } else { None };
        let smart_reference = (aux.smart_reference.is_none() && (deconv_measures.is_some() || denoise_measures.is_some() || wavelet_measures.is_some() || smart_measures.is_some())).then(|| Arc::new(RegionReference { deconv: deconv_measures, denoise: denoise_measures, wavelet: wavelet_measures, smart: smart_measures }));
        Some(RunOutput { image: frame.0, timings, warnings, tone_input, curves_input, deconv, ringing, smart_map, smart_reference })
    }

    /// Forget the results that take only milliseconds to recompute.
    fn drop_quick_stages(&mut self) {
        self.align = None;
        self.wavelet = None;
        self.smart = None;
        self.smart_cache = None;
        self.smart_map = None;
        self.contrast = None;
        self.dering = None;
        self.ring = None;
        self.tone = None;
        self.color = None;
        self.photocolor = None;
        self.blend = None;
        self.add_blend.clear();
        self.mineral = None;
        self.suncolor = None;
        self.gamut = None;
    }
}

/// `keep`: how many decompositions to hold on to.
/// `measured`: the whole image's noise of each plane's layers, with its generation, for a
/// zoomed region.
fn apply_wavelet(caches: &mut Vec<WaveletCache>, keep: usize, img: &Image, input_gen: u64, p: &WaveletParams, measured: Option<(&[WaveletNoise], u64)>, cancel: &dyn Fn() -> bool) -> Option<Image> {
    if p.is_neutral() {
        return Some(img.clone());
    }
    let key = p.decomposition_key();
    let noise_gen = measured.map_or(0, |m| m.1);
    let index = match caches.iter().position(|c| c.input_gen == input_gen && c.key == key && c.noise_gen == noise_gen) {
        Some(i) => i,
        None => {
            let planes = working_planes(img, p.color);
            let mut decompositions = Vec::with_capacity(planes.len());
            for (k, plane) in planes.iter().enumerate() {
                let mut d = decompose(plane, img.width(), img.height(), p, cancel)?;
                if let Some(m) = measured.and_then(|(m, _)| m.get(k)).filter(|m| m.noise.len() == d.noise.len() && m.white_noise.len() == d.white_noise.len()) {
                    d.noise.clone_from(&m.noise);
                    d.white_noise.clone_from(&m.white_noise);
                    if let (Some(g), Some(n)) = (d.grain.as_mut(), m.grain) {
                        g.1 = n;
                    }
                    d.edge_levels = m.edges;
                }
                decompositions.push(d);
            }
            caches.push(WaveletCache { input_gen, noise_gen, key, planes, decompositions });
            while caches.len() > keep {
                caches.remove(0);
            }
            caches.len() - 1
        }
    };
    let cache = &caches[index];
    let processed = cache.decompositions.par_iter().map(|d| recompose(d, p)).collect();
    Some(merge_planes(img, p.color, &cache.planes, processed))
}

/// `out = base·(1 − m) + layer·m` (or the layer's blend mode instead of `layer`), one layer
/// after the other.
fn blend(base: &Image, parts: &[(Frame, Arc<Vec<f32>>, u64, BlendMode)]) -> Image {
    let mut channels = base.channels().to_vec();
    for ((layer, _), mask, _, mode) in parts {
        blend::composite_into(&mut channels, layer.channels(), mask, *mode);
    }
    Image::new(base.width(), base.height(), channels, base.source_bit_depth)
}

/// Deconvolve the working planes; in R·G·B mode each channel gets its own PSF size.
/// Each working plane's deconvolution iterations and measured noise.
type DeconvPlanes = Vec<(u32, f32)>;

/// `measured`: the whole image's iterations and noise of each plane, for a zoomed region. Also
/// returns each plane's.
fn run_deconvolution(img: &Image, p: &DeconvParams, area: Option<&[f32]>, measured: Option<&[(u32, f32)]>, cancel: &dyn Fn() -> bool, progress: &dyn Fn(f32)) -> Option<(Image, DeconvReport, DeconvPlanes)> {
    let planes = working_planes(img, p.color);
    let count = planes.len() as f32;
    let measured = measured.filter(|m| m.len() == planes.len());
    let mut processed = Vec::with_capacity(planes.len());
    let mut iterations = 0;
    let mut each = Vec::with_capacity(planes.len());
    for (k, plane) in planes.iter().enumerate() {
        let report = |f: f32| progress((k as f32 + f) / count);
        let psf_scale = if planes.len() == 3 { p.channel_scale[k].max(0.1) } else { 1.0 };
        let options = PlaneOptions { psf_scale, area, measured: measured.map(|m| m[k]) };
        let (out, r) = deconvolve_plane_with(plane, img.width(), img.height(), p, options, cancel, &report)?;
        iterations = iterations.max(r.iterations);
        each.push(r.measured);
        processed.push(out);
    }
    Some((merge_planes(img, p.color, &planes, processed), DeconvReport { iterations, measured: (iterations, 0.0) }, each))
}
