//! Automatic processing settings for each kind of image, adapted to the measured noise, blur
//! and subject size.
//!
//! Two tiers:
//! - **structural** settings (wavelet kernel, number of levels, first step, PSF shape and size,
//!   noise-reduction method) follow from the analysis and are always chosen automatically;
//! - **simple** settings (how much sharpening, how much noise reduction) are left to the user,
//!   or filled in on request by [`auto_pipeline`].

use crate::adjust::ColorParams;
use crate::analysis::{Analysis, ImageKind};
use crate::bxt::BxtParams;
use crate::color::ColorMode;
use crate::deconv::DeconvParams;
use crate::dering::{DeringParams, DeringRegion};
use crate::enhance::{ContrastParams, ToneParams, ToneStep};
use crate::enhance::{NoiseMethod, NoiseParams};
use crate::mask::{AutoRegion, MaskShape, MaskSpec};
use crate::pipeline::LayerRole;
use crate::pipeline::{Module, Pipeline};
use crate::psf::Psf;
use crate::wavelet::{MAX_LAYERS, ScaleMode, WaveletKernel, WaveletLayer, WaveletParams};
use serde::{Deserialize, Serialize};

struct Base {
    psf_sigma: f32,
    iterations: u32,
    tv_lambda: f32,
    damping: f32,
    gains: [f32; 6],
    denoise: [f32; 6],
    noise: NoiseParams,
}

fn base(kind: ImageKind) -> Base {
    match kind {
        ImageKind::Planet => Base {
            psf_sigma: 1.5,
            iterations: 15,
            tv_lambda: 0.0015,
            damping: 1.0,
            gains: [1.6, 1.2, 0.6, 0.2, 0.0, 0.0],
            denoise: [1.2, 0.6, 0.0, 0.0, 0.0, 0.0],
            // Fine grain over soft belts: multiscale thresholding, clean colour, black sky.
            noise: NoiseParams { method: NoiseMethod::Planetary, strength: 1.0, detail: 0.3, chroma: 2, sky: 0.8 },
        },
        // Rings and the Cassini division ring easily: gentler on both steps.
        ImageKind::Saturn => Base {
            psf_sigma: 1.3,
            iterations: 12,
            tv_lambda: 0.002,
            damping: 1.2,
            gains: [1.2, 1.0, 0.5, 0.1, 0.0, 0.0],
            denoise: [1.4, 0.7, 0.0, 0.0, 0.0, 0.0],
            // Thin ring divisions are small-scale detail: a little gentler.
            noise: NoiseParams { method: NoiseMethod::Planetary, strength: 0.8, detail: 0.3, chroma: 2, sky: 0.8 },
        },
        // The sharp limb against black sky rings first; keep deconvolution moderate.
        ImageKind::FullDisk => Base {
            psf_sigma: 1.2,
            iterations: 10,
            tv_lambda: 0.001,
            damping: 1.0,
            gains: [1.0, 0.9, 0.5, 0.25, 0.0, 0.0],
            denoise: [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            // Crater rims and limb must stay crisp; the sky around the disk can be cleaned hard.
            noise: NoiseParams { method: NoiseMethod::Lunar, strength: 1.0, detail: 0.3, chroma: 1, sky: 0.9 },
        },
        // Craters and granulation span many scales; no limb to protect.
        ImageKind::Surface => Base {
            psf_sigma: 1.6,
            iterations: 20,
            tv_lambda: 0.0015,
            damping: 1.0,
            gains: [1.4, 1.1, 0.6, 0.3, 0.1, 0.0],
            denoise: [1.2, 0.6, 0.0, 0.0, 0.0, 0.0],
            // Texture everywhere and no sky: edge-preserving, keep more fine detail.
            noise: NoiseParams { method: NoiseMethod::Lunar, strength: 0.8, detail: 0.4, chroma: 1, sky: 0.0 },
        },
    }
}

/// Measured blur for `kind`, or the kind's typical value. The limb estimate is only trusted
/// for the kind it was measured on, and is scaled down: an overestimated PSF causes ringing,
/// an underestimated one merely under-sharpens.
fn psf_sigma(kind: ImageKind, analysis: &Analysis) -> f32 {
    analysis.psf_sigma.filter(|_| kind == analysis.kind).map_or(base(kind).psf_sigma, |s| (0.85 * s).clamp(0.6, 3.5))
}

/// Core of the blur, as a Gaussian σ (px): from the frequency where the detail ends (the σ
/// whose MTF is down to 10 % there), else the limb or typical value of [`psf_sigma`].
///
/// On real stacks the limb's profile is much wider than the blur of the detail: limb darkening
/// and the seeing's wide wings widen it. A PSF that wide over-deconvolves the detail into
/// blotches and outlines; the core from the spectrum, with Moffat wings, doesn't.
fn blur_core(kind: ImageKind, analysis: &Analysis) -> f32 {
    match analysis.spectral.and(detail_cutoff(analysis)) {
        Some(cutoff) => (0.3416 / cutoff).clamp(0.6, 4.0),
        None => psf_sigma(kind, analysis),
    }
}

/// Moffat β of the deconvolution PSF with a core from the spectrum: the limb's when it showed
/// wide wings, else 2 (stacks of short exposures through turbulence have wings well beyond a
/// Gaussian's).
fn moffat_beta(kind: ImageKind, analysis: &Analysis) -> f32 {
    analysis.psf_moffat.filter(|_| kind == analysis.kind).map_or(2.0, |(_, beta)| beta).clamp(1.5, 3.0)
}

/// Noisier data → stronger denoising and damping; very clean data → lighter.
fn noise_factor(analysis: &Analysis) -> f32 {
    (60.0 / analysis.snr.max(1.0)).clamp(0.6, 2.0)
}

/// Size of the first free-scale layer: the layer peaks at 0.3/size cycles/px, so it sits on
/// the frequency where the detail sinks into the noise (on test images whose sharp version is
/// known, that is where the best results were). Without a spectrum, from the blur.
pub fn first_scale(kind: ImageKind, analysis: &Analysis) -> f32 {
    let size = match detail_cutoff(analysis) {
        Some(cutoff) => 0.3 / cutoff.max(0.05),
        None => 0.8 * psf_sigma(kind, analysis),
    };
    (size.clamp(1.0, 4.0) * 20.0).round() / 20.0
}

/// Finest frequency with real detail (cycles/px): where the spectrum sinks into the noise, and
/// never past what the telescope can resolve. With the equipment but no spectrum, a little
/// below the diffraction limit (the seeing always takes some).
pub fn detail_cutoff(analysis: &Analysis) -> Option<f32> {
    let optical = analysis.optics.map(|o| o.cutoff);
    match (analysis.spectral.map(|s| s.cutoff), optical) {
        (Some(s), Some(o)) => Some(s.min(o)),
        (Some(s), None) => Some(s),
        (None, Some(o)) => Some(0.8 * o),
        (None, None) => None,
    }
}

/// Signal over noise power at `f` cycles/px: measured, or assumed well above the noise up to
/// `cutoff` and absent beyond; never any past the diffraction limit.
fn signal_to_noise(analysis: &Analysis, cutoff: f32) -> impl Fn(f32) -> f32 + '_ {
    move |f: f32| {
        if analysis.optics.is_some_and(|o| f > o.cutoff) {
            return 0.0;
        }
        analysis.spectral.map_or(if f < cutoff { 100.0 } else { 0.0 }, |s| s.snr(f))
    }
}

/// Wavelet transform shape from the image, with `scales` free or classic (the user's choice;
/// [`ScaleMode::Linear`] counts as classic): enough levels for the coarsest to reach about a
/// tenth of the subject. Free scales start at [`first_scale`]. Classic ones (whose settings are
/// filled in either way, for a later switch) start with a step of 2 px when the image is
/// oversampled (σ > 2.6, the finest scale then holds only noise) and use the narrower linear
/// kernel when it is undersampled.
pub fn wavelet_structure(kind: ImageKind, analysis: &Analysis, scales: ScaleMode) -> WaveletParams {
    let sigma = psf_sigma(kind, analysis);
    let base_step: u32 = if sigma > 2.6 { 2 } else { 1 };
    let first = first_scale(kind, analysis);
    let free = scales == ScaleMode::Free;
    let coarsest = (analysis.subject_size / 10.0).max(4.0) / if free { first } else { base_step as f32 };
    let levels = (coarsest.log2().floor() as usize + 1).clamp(4, MAX_LAYERS);
    WaveletParams {
        kernel: if sigma < 1.0 { WaveletKernel::Linear } else { WaveletKernel::B3Spline },
        scale_mode: if free { ScaleMode::Free } else { ScaleMode::Dyadic },
        base_step,
        first_scale: first,
        layers: vec![WaveletLayer::default(); levels],
        // Detail must clear the noise by ~2σ to get the full gain.
        noise_adaptive: 2.0,
        // A disk on black sky halos at the limb; a surface has no such edge.
        edge_protection: match kind {
            ImageKind::Saturn => 0.6,
            ImageKind::Planet | ImageKind::FullDisk => 0.5,
            ImageKind::Surface => 0.2,
        },
        ..Default::default()
    }
}

/// Deringing neighbourhood and limb band from the blur; Saturn gets a wider tolerance so the
/// Cassini division and the ring edges, genuine sharp detail, keep their contrast.
pub fn dering_structure(kind: ImageKind, analysis: &Analysis) -> DeringParams {
    let mut d = DeringParams::for_blur(blur_core(kind, analysis));
    d.tolerance = if kind == ImageKind::Saturn { 0.15 } else { 0.1 };
    // On Saturn the ring edges and divisions are real sharp detail: the globe's limb only.
    d.region = if kind == ImageKind::Saturn { DeringRegion::Globe } else { DeringRegion::Subject };
    d
}

/// Limb rings only form where there is a limb.
fn has_limb(kind: ImageKind) -> bool {
    kind != ImageKind::Surface
}

/// How hard the automatic settings push.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutoStrength {
    /// Natural, almost free of artefacts.
    Gentle,
    #[default]
    Normal,
    /// As much detail as the data holds.
    Strong,
}

impl AutoStrength {
    pub fn label(self) -> &'static str {
        match self {
            AutoStrength::Gentle => "Delicato",
            AutoStrength::Normal => "Normale",
            AutoStrength::Strong => "Forte",
        }
    }

    /// (maximum deconvolution iterations, wavelet gains, noise reduction) relative to the
    /// kind's typical settings. Iterations stop on their own at the noise in every case.
    fn factors(self) -> (f32, f32, f32) {
        match self {
            AutoStrength::Gentle => (0.8, 0.6, 1.25),
            AutoStrength::Normal => (2.0, 1.0, 1.0),
            AutoStrength::Strong => (4.0, 1.5, 0.85),
        }
    }
}

/// Deconvolution's structure from the analysis: PSF shape and size (a Moffat when the limb
/// shows wide wings), one PSF size per channel when the channels' blur differs, automatic
/// stop and anti-ringing on a disk's limb. Amounts are left at the kind's typical values.
pub fn deconv_structure(kind: ImageKind, analysis: &Analysis) -> DeconvParams {
    let b = base(kind);
    let sigma = psf_sigma(kind, analysis);
    // Measurements are only trusted for the kind they were made on.
    let ours = kind == analysis.kind;
    let psf = if analysis.spectral.is_some() {
        Psf::Moffat { fwhm: 2.3548 * blur_core(kind, analysis), beta: moffat_beta(kind, analysis) }
    } else {
        match analysis.psf_moffat.filter(|_| ours) {
            Some((fwhm, beta)) => Psf::Moffat { fwhm: (0.85 * fwhm).clamp(1.0, 9.0), beta },
            None => Psf::Gaussian { sigma },
        }
    };
    let per_channel = analysis.channel_sigma.filter(|_| ours).zip(analysis.psf_sigma).map(|(c, lum)| c.map(|v| (v / lum).clamp(0.7, 1.4)));
    let (color, channel_scale) = match per_channel {
        // Worth three deconvolutions only when the channels' blur clearly differs.
        Some(scale) if scale.iter().any(|v| (v - 1.0).abs() > 0.12) => (ColorMode::Rgb, scale),
        _ => (ColorMode::Luminance, [1.0; 3]),
    };
    DeconvParams {
        psf,
        iterations: b.iterations,
        tv_lambda: b.tv_lambda,
        damping: b.damping,
        color,
        channel_scale,
        auto_stop: true,
        deringing: if has_limb(kind) { 0.5 } else { 0.0 },
        ..Default::default()
    }
}

/// "Nitidezza intelligente" for the image: the cut-off from the analysis and the equipment,
/// where to put the peak, how much to spare the limb and whether there is a sky to leave out,
/// by kind of subject. Planets' belts
/// and festoons, the Moon's crater rims and the Sun's filaments carry real detail almost up to
/// the cut-off; a disk on black sky halos at its limb, a surface has no limb.
pub fn smart_structure(kind: ImageKind, analysis: &Analysis) -> crate::smart::SmartParams {
    let (peak, limb_protection) = match kind {
        // Low: inside the disk it changes nothing and the limb rarely needs much; the map in
        // the card shows where it acts, so it can be raised where halos appear.
        ImageKind::Planet => (1.0, 0.25),
        ImageKind::Saturn => (1.0, 0.3),
        ImageKind::FullDisk => (1.0, 0.2),
        ImageKind::Surface => (1.0, 0.0),
    };
    // A surface has no sky: its dark parts are shadows, part of the subject.
    let include_sky = kind == ImageKind::Surface;
    crate::smart::SmartParams { cutoff: detail_cutoff(analysis).unwrap_or(0.0), peak, limb_protection, include_sky, ..Default::default() }
}

/// The automatic "Nitidezza intelligente": the structure with a strength by kind and by how
/// hard the user wants it pushed (it isn't part of the full automatic processing, which uses
/// the wavelets: the two are alternatives).
pub fn auto_smart(kind: ImageKind, analysis: &Analysis, strength: AutoStrength) -> crate::smart::SmartParams {
    let base: f32 = match kind {
        ImageKind::Surface | ImageKind::FullDisk => 0.4,
        _ => 0.35,
    };
    let factor = match strength {
        AutoStrength::Gentle => 0.7,
        AutoStrength::Normal => 1.0,
        AutoStrength::Strong => 1.4,
    };
    crate::smart::SmartParams { strength: (base * factor).min(1.0), ..smart_structure(kind, analysis) }
}

/// The automatic deconvolution: [`deconv_structure`] with as many iterations at most as
/// `strength` says, and damping following the noise.
pub fn auto_deconv(kind: ImageKind, analysis: &Analysis, strength: AutoStrength) -> DeconvParams {
    let b = base(kind);
    let (iterations, ..) = strength.factors();
    DeconvParams { iterations: (b.iterations as f32 * iterations).round() as u32, damping: b.damping * noise_factor(analysis), ..deconv_structure(kind, analysis) }
}

/// Everything automatic, simple settings included: what the "Automatico" buttons apply.
pub fn auto_pipeline(kind: ImageKind, analysis: &Analysis) -> Pipeline {
    auto_pipeline_with(kind, analysis, AutoStrength::Normal, ScaleMode::Free)
}

/// [`auto_pipeline`] pushing as hard as `strength` says.
pub fn auto_pipeline_with(kind: ImageKind, analysis: &Analysis, strength: AutoStrength, scales: ScaleMode) -> Pipeline {
    let b = base(kind);
    let noise = noise_factor(analysis);
    let sigma = blur_core(kind, analysis);
    let (_, gains, cleaning) = strength.factors();
    let deconv = auto_deconv(kind, analysis, strength);
    // The tables are tuned for the classic scales, with single-pixel first steps (shifted with
    // the first step).
    let mut classic = wavelet_structure(kind, analysis, ScaleMode::Dyadic);
    for (j, layer) in classic.layers.iter_mut().enumerate() {
        let k = j + classic.base_step as usize - 1;
        // The noise-reduction module does most of the work; wavelets keep a lighter touch.
        *layer = WaveletLayer { gain: gains * b.gains.get(k).copied().unwrap_or(0.0), denoise: 0.5 * b.denoise.get(k).copied().unwrap_or(0.0) * noise };
    }
    // Free scales sharpen the real detail as much as the classic ones would.
    let wavelet = if scales == ScaleMode::Free { convert_scales(kind, analysis, &classic, ScaleMode::Free) } else { classic };
    // Below ~0.25 px the measurement is within its noise and the dispersion invisible.
    let shift = analysis.rgb_shift.unwrap_or_default();
    let worth_aligning = shift.red.iter().chain(&shift.blue).any(|v| v.abs() > 0.25);
    // BlurXTerminator's nonstellar PSF diameter is a FWHM: 2.355 σ.
    let bxt = BxtParams { psf_diameter: (2.3548 * sigma).clamp(0.5, 8.0), ..Default::default() };
    Pipeline {
        align: Module { enabled: worth_aligning, params: shift },
        bxt: Module::off(bxt),
        deconv: Module::on(deconv),
        denoise: Module::on(NoiseParams { strength: b.noise.strength * noise * cleaning, ..b.noise }),
        wavelet: Module::on(wavelet),
        dering: Module { enabled: has_limb(kind), params: dering_structure(kind, analysis) },
        ..Default::default()
    }
}

/// `current` with the other kind of scales (`to`; [`ScaleMode::Linear`] counts as classic):
/// the transform derived from the image again, gains giving the same sharpening of the real
/// detail (finer than the cut-off there is only noise, which free scales leave alone), noise
/// thresholds moved to the matching layer (the classic finest layer's ↔ the grain). The
/// user's other choices stay.
pub fn convert_scales(kind: ImageKind, analysis: &Analysis, current: &WaveletParams, to: ScaleMode) -> WaveletParams {
    let structure = wavelet_structure(kind, analysis, to);
    let mut out = WaveletParams {
        kernel: structure.kernel,
        scale_mode: structure.scale_mode,
        base_step: structure.base_step,
        first_scale: structure.first_scale,
        grain_denoise: 0.0,
        layers: structure.layers.clone(),
        ..current.clone()
    };
    let cutoff = detail_cutoff(analysis).unwrap_or(0.3 / structure.first_scale);
    let snr = signal_to_noise(analysis, cutoff);
    let unit = WaveletParams { amplify: 1.0, ..current.clone() };
    let gains = crate::wavelet::equivalent_gains(&unit, &out, cutoff, &snr);
    // Thresholds by scale: finest first, the grain counting as the finest of free scales.
    let mut thresholds: Vec<f32> = current.layers.iter().map(|l| l.denoise).collect();
    if current.scale_mode == ScaleMode::Free {
        thresholds.insert(0, current.grain_denoise);
    }
    if out.scale_mode == ScaleMode::Free && !thresholds.is_empty() {
        out.grain_denoise = thresholds.remove(0);
    }
    for (j, layer) in out.layers.iter_mut().enumerate() {
        *layer = WaveletLayer { gain: gains[j], denoise: thresholds.get(j).copied().unwrap_or(0.0) };
    }
    out
}

/// Starting point after opening an image: structural settings chosen automatically, every
/// module switched on but set to change nothing, so moving any control acts at once. BXT, an
/// external program, stays off; so does the deconvolution, which has no amount to start from
/// zero (its PSF and iterations are automatic): switching it on applies it as set up. The user
/// (or an "Automatico" button) does the rest.
pub fn initial_pipeline(kind: ImageKind, analysis: &Analysis, scales: ScaleMode) -> Pipeline {
    let auto = auto_pipeline(kind, analysis);
    nothing_applied(Pipeline {
        bxt: Module::off(auto.bxt.params),
        deconv: Module::on(auto.deconv.params),
        denoise: Module::on(auto.denoise.params),
        wavelet: Module::on(wavelet_structure(kind, analysis, scales)),
        smart: Module::on(smart_structure(kind, analysis)),
        dering: Module::on(dering_structure(kind, analysis)),
        ..Default::default()
    })
}

/// Generic starting values, nothing taken from the image: what "Rimuovi impostazioni
/// automatiche" goes back to.
pub fn neutral_pipeline() -> Pipeline {
    nothing_applied(Pipeline::default())
}

/// `p` with every module on and every amount at "no change" (structure kept).
fn nothing_applied(p: Pipeline) -> Pipeline {
    let d = Pipeline::default();
    Pipeline {
        align: Module::on(Default::default()),
        bxt: Module::off(p.bxt.params),
        // Deconvolution has no amount: it waits switched off, already set up.
        deconv: Module::off(p.deconv.params),
        denoise: Module::on(NoiseParams { strength: 0.0, chroma: 0, sky: 0.0, ..p.denoise.params }),
        wavelet: Module::on(WaveletParams { layers: vec![WaveletLayer::default(); p.wavelet.params.layers.len()], amplify: 1.0, grain_denoise: 0.0, ..p.wavelet.params }),
        smart: Module::on(crate::smart::SmartParams { strength: 0.0, ..p.smart.params }),
        contrast: Module::on(ContrastParams { amount: 0.0, local_amount: 0.0, ..d.contrast.params }),
        dering: Module::on(DeringParams { strength: 0.0, ..p.dering.params }),
        tone: Module::on(ToneParams::default()),
        curves: Module::on(ToneParams::default()),
        color: Module::on(ColorParams::default()),
        mineral: Module::on(crate::adjust::MineralParams::neutral()),
        layers: p.layers,
    }
}

/// Automatic white balance for the kind of image. The Moon is nearly grey, so its average
/// colour is made neutral; planets have colours of their own (Saturn's pale yellow, Mars'
/// orange), so only half of the correction is applied in log terms: the camera's cast goes,
/// the planet's hue stays.
pub fn auto_color(kind: ImageKind, img: &crate::Image) -> [f32; 3] {
    let gains = crate::adjust::auto_balance(img);
    match kind {
        ImageKind::FullDisk | ImageKind::Surface => gains,
        ImageKind::Planet | ImageKind::Saturn => gains.map(f32::sqrt),
    }
}

/// Add the layer behind the "Anelli di Saturno" or "Lune" card, set up for its job.
pub fn add_role_layer(p: &mut Pipeline, role: LayerRole) -> u64 {
    let (name, region, feather) = match role {
        LayerRole::Rings => ("Anelli di Saturno", AutoRegion::SaturnRings, 2.0),
        LayerRole::Moons => ("Lune", AutoRegion::Moons, 1.5),
    };
    let id = p.add_layer(name, MaskSpec::new(MaskShape::Auto(region), feather));
    let base_wavelet = p.wavelet.params.clone();
    let layer = p.layer_mut(id).expect("just added");
    layer.role = Some(role);
    let mut wavelet = base_wavelet;
    let gains: &[f32] = match role {
        // Ring divisions and ringlets sit at 2–4 px: favour those scales.
        LayerRole::Rings => &[1.0, 1.6, 0.9, 0.2],
        // Moons are a few pixels wide: a little crispness, nothing coarse.
        LayerRole::Moons => &[0.8, 0.5],
    };
    for (j, l) in wavelet.layers.iter_mut().enumerate() {
        l.gain = gains.get(j).copied().unwrap_or(0.0);
    }
    layer.wavelet = Module::on(wavelet);
    match role {
        LayerRole::Rings => layer.contrast = Module::on(ContrastParams { amount: 0.4, radius: 1.5, ..Default::default() }),
        // Moons are much fainter than the planet: lift their midtones.
        LayerRole::Moons => layer.tone = Module::on(ToneParams { current: ToneStep { gamma: 1.5, ..Default::default() }, applied: vec![] }),
    }
    id
}

/// Why the automatic (structural) settings are what they are, for this image: one short
/// paragraph per point, shown under "Avanzate".
pub struct Explanation {
    pub deconv: Vec<String>,
    pub wavelet: Vec<String>,
    pub denoise: Vec<String>,
    pub bxt: Vec<String>,
    pub dering: Vec<String>,
}

pub fn explain(kind: ImageKind, analysis: &Analysis, scales: ScaleMode) -> Explanation {
    let b = base(kind);
    let sigma = psf_sigma(kind, analysis);
    let what = match kind {
        ImageKind::Planet => "il pianeta",
        ImageKind::Saturn => "Saturno con gli anelli",
        ImageKind::FullDisk => "il disco",
        ImageKind::Surface => "l'immagine",
    };
    let measured = analysis.psf_sigma.filter(|_| kind == analysis.kind);
    let core = analysis.spectral.and(detail_cutoff(analysis));
    let mut deconv = vec![match (core, measured) {
        (Some(cutoff), limb) => {
            let (fwhm, beta) = (2.3548 * blur_core(kind, analysis), moffat_beta(kind, analysis));
            let limb = limb.map_or(String::new(), |m| {
                format!(" Il bordo del disco darebbe σ {m:.1} px, ma è allargato dall'oscuramento al bordo e dalle ali del seeing: una PSF così larga trasformerebbe i dettagli in macchie e contorni.")
            });
            format!(
                "PSF di Moffat, FWHM {fwhm:.1} px, β {beta:.1}: il nucleo viene dallo spettro (il dettaglio finisce a un periodo di {:.1} px), le ali larghe sono quelle tipiche del seeing.{limb}",
                1.0 / cutoff
            )
        }
        (None, measured) => match measured {
            Some(m) => format!(
                "PSF gaussiana σ {sigma:.2} px: l'85 % della sfocatura misurata sul bordo del disco (σ {m:.2} px). È un po' più piccola apposta: una PSF troppo grande crea aloni, una troppo piccola rende solo meno nitido."
            ),
            None => format!("PSF gaussiana σ {sigma:.2} px: valore tipico per questo tipo di immagine. La sfocatura non si è potuta misurare (serve un disco rotondo con il bordo netto)."),
        },
    }];
    let structure = deconv_structure(kind, analysis);
    if let (None, Psf::Moffat { fwhm, beta }) = (core, structure.psf) {
        deconv.push(format!(
            "PSF di Moffat (FWHM {fwhm:.1} px, β {beta:.1}): il profilo del bordo ha ali più larghe di una gaussiana, tipiche del seeing turbolento; una gaussiana le correggerebbe poco."
        ));
    }
    if structure.color == ColorMode::Rgb {
        let [r, g, b] = structure.channel_scale;
        deconv.push(format!("Una PSF per canale (rosso ×{r:.2}, verde ×{g:.2}, blu ×{b:.2}): la sfocatura misurata sul bordo è diversa nei tre colori."));
    }
    deconv.push("Richardson–Lucy con accelerazione Biggs–Andrews: il metodo di riferimento per le immagini planetarie; raggiunge la stessa qualità con molte meno iterazioni.".into());
    deconv.push("Arresto automatico: le iterazioni sono un massimo; si ferma quando l'immagine è ricostruita fino al livello del rumore misurato, oltre il quale amplificherebbe solo rumore.".into());
    if structure.deringing > 0.0 {
        deconv.push("Anti-ringing a metà: sul bordo e nel cielo la stima non può allontanarsi troppo dall'immagine osservata, così gli anelli non si formano durante le iterazioni.".into());
    }
    deconv.push(match kind {
        ImageKind::Saturn => format!("Regolarizzazione TV {:.4}, un po' più alta: la divisione di Cassini e i bordi degli anelli creano facilmente aloni.", b.tv_lambda),
        ImageKind::FullDisk => format!("Regolarizzazione TV {:.4}, bassa: il bordo netto del disco sul cielo nero va trattato con prudenza ma senza appiattire i crateri.", b.tv_lambda),
        _ => format!("Regolarizzazione TV {:.4}: limita gli aloni sui bordi senza appiattire i dettagli.", b.tv_lambda),
    });

    let w = wavelet_structure(kind, analysis, scales);
    let coarsest = w.scale(w.layers.len() - 1);
    let mut wavelet = vec![format!(
        "{} livelli: la scala più grossa ({coarsest:.0} px) è circa un decimo della dimensione del soggetto ({:.0} px). Livelli più grandi cambierebbero solo la forma generale di {what}.",
        w.layers.len(),
        analysis.subject_size
    )];
    if w.scale_mode == ScaleMode::Free {
        let optical = analysis.optics.map(|o| o.cutoff);
        wavelet.push(match (analysis.spectral, optical) {
            (Some(sp), Some(o)) if o < sp.cutoff => format!(
                "Primo livello {:.2} px: lo strumento non può risolvere dettagli più fini di un periodo di {:.1} px (limite di diffrazione), anche se lo spettro arriverebbe a {:.1} px: quello che c'è oltre è rumore. Più fine del primo livello c'è solo grana, che viene soltanto ripulita.",
                w.first_scale,
                1.0 / o,
                1.0 / sp.cutoff
            ),
            (Some(sp), _) => format!(
                "Primo livello {:.2} px: lo spettro dell'immagine mostra che il dettaglio vero finisce intorno a un periodo di {:.1} px, dove il segnale scende sotto il rumore. Il primo livello è centrato lì; quello che è più fine è solo grana e viene soltanto ripulito, mai reso più nitido.",
                w.first_scale,
                1.0 / sp.cutoff
            ),
            (None, Some(o)) => format!("Primo livello {:.2} px, dallo strumento: il limite di diffrazione è un periodo di {:.1} px; lo spettro dell'immagine non si è potuto misurare.", w.first_scale, 1.0 / o),
            (None, None) => format!("Primo livello {:.2} px, dalla sfocatura (σ {sigma:.1} px): lo spettro dell'immagine non si è potuto misurare.", w.first_scale),
        });
        wavelet.push("Scale libere: ogni livello raddoppia la scala del precedente partendo da una dimensione qualsiasi, non per forza da 1 px. Sulle immagini di prova danno un risultato più fedele delle scale classiche 1·2·4·8.".into());
    } else {
        wavelet.push(if w.base_step == 2 {
            format!("Passo iniziale 2 px: l'immagine è sovracampionata (σ {sigma:.1} px), quindi un primo livello da 1 px conterrebbe solo rumore.")
        } else {
            format!("Passo iniziale 1 px: con una sfocatura di σ {sigma:.1} px il dettaglio più fine arriva quasi al singolo pixel.")
        });
        wavelet.push(match w.kernel {
            crate::wavelet::WaveletKernel::Linear => "Kernel lineare: l'immagine è sottocampionata (σ < 1 px) e serve una risposta più stretta.".into(),
            _ => "Kernel B3-spline: liscio e isotropo, non introduce artefatti in nessuna direzione (Starck & Murtagh).".into(),
        });
        wavelet.push("Progressione diadica (1·2·4·8…): ogni livello raddoppia la scala, così pochi livelli coprono tutte le dimensioni dei dettagli.".into());
    }

    let mut denoise = vec![match b.noise.method {
        NoiseMethod::Planetary => "Metodo Pianeti: il rumore è una grana fine sopra dettagli morbidi; toglierlo dalle 3 scale più fini lascia intatte bande e festoni.".to_string(),
        NoiseMethod::Lunar => "Metodo Luna e Sole: crateri, ombre e granulazione hanno bordi netti; il filtro bilaterale media solo pixel simili e non li ammorbidisce.".to_string(),
    }];
    denoise.push(match kind {
        ImageKind::Saturn => "Intensità ridotta: anche le divisioni sottili degli anelli sono dettaglio fine e non vanno cancellate.".into(),
        ImageKind::Surface => "Pulizia del cielo a zero: in un'immagine di superficie non c'è cielo.".into(),
        _ => "Pulizia del cielo alta: lo sfondo intorno al soggetto è nero e si può pulire a fondo.".into(),
    });
    denoise.push(format!("L'intensità tiene conto del rapporto segnale/rumore misurato (SNR {:.0}): più l'immagine è rumorosa, più la pulizia è forte.", analysis.snr));

    let bxt = vec![format!(
        "Diametro PSF {:.1} px: è la FWHM della sfocatura misurata (2,355 × σ). La stima automatica di BlurXTerminator usa le stelle, che in queste immagini non ci sono.",
        (2.3548 * blur_core(kind, analysis)).clamp(0.5, 8.0)
    )];
    let d = dering_structure(kind, analysis);
    let mut dering = vec![format!(
        "Larghezza aloni {} px, dalla sfocatura (σ {:.1} px): le bande chiare e scure che la nitidezza crea accanto a un bordo netto stanno entro pochi σ da esso.",
        d.radius,
        blur_core(kind, analysis)
    )];
    dering.push("Agisce solo vicino ai bordi netti dell'originale (bordo del pianeta, anelli): bande e dettagli fini hanno poco contrasto locale e non vengono toccati.".into());
    if kind == ImageKind::Saturn {
        dering.push("Solo sul globo, compresi gli anelli che gli passano davanti: i bordi degli anelli all'esterno e la divisione di Cassini sono dettagli netti veri e verrebbero smorzati. Si può cambiare in \"Dove agisce\".".into());
    }
    dering.push(if kind == ImageKind::Saturn {
        format!("Tolleranza {:.0} %, più alta: la divisione di Cassini e i bordi degli anelli sono dettagli netti veri e non vanno smorzati.", d.tolerance * 100.0)
    } else {
        format!("Tolleranza {:.0} %: la deconvoluzione può superare di poco i valori originali vicini; oltre è quasi sempre ringing.", d.tolerance * 100.0)
    });
    Explanation { deconv, wavelet, denoise, bxt, dering }
}

/// Apply the structural settings for `kind` to `current`, keeping the user's simple settings
/// (sharpening amounts, strengths), on/off choices and kind of wavelet scales.
pub fn restructure(kind: ImageKind, analysis: &Analysis, current: &Pipeline) -> Pipeline {
    let auto = auto_pipeline(kind, analysis);
    let wavelet = wavelet_structure(kind, analysis, current.wavelet.params.scale_mode);
    let structure = Pipeline {
        deconv: Module::on(deconv_structure(kind, analysis)),
        wavelet: Module::on(wavelet),
        smart: Module::on(smart_structure(kind, analysis)),
        dering: Module::on(dering_structure(kind, analysis)),
        ..auto
    };
    with_structure(current, &structure)
}

/// The structural settings back to generic values (nothing taken from the image), keeping the
/// user's amounts and on/off choices: "impostazioni avanzate automatiche" switched off.
pub fn generic_structure(current: &Pipeline) -> Pipeline {
    with_structure(current, &Pipeline::default())
}

/// `current` with the structural settings of `structure` (the kind of wavelet scales stays the
/// user's).
fn with_structure(current: &Pipeline, structure: &Pipeline) -> Pipeline {
    let mut p = current.clone();
    let (d, s) = (&mut p.deconv.params, &structure.deconv.params);
    (d.algorithm, d.psf, d.accelerate, d.tv_lambda, d.color, d.channel_scale) = (s.algorithm, s.psf, s.accelerate, s.tv_lambda, s.color, s.channel_scale);
    let (w, s) = (&mut p.wavelet.params, &structure.wavelet.params);
    (w.kernel, w.base_step, w.first_scale) = (s.kernel, s.base_step, s.first_scale);
    (w.noise_adaptive, w.edge_protection) = (s.noise_adaptive, s.edge_protection);
    w.layers.resize(s.layers.len(), WaveletLayer::default());
    p.denoise.params.method = structure.denoise.params.method;
    p.bxt.params.psf_diameter = structure.bxt.params.psf_diameter;
    let (m, s) = (&mut p.smart.params, &structure.smart.params);
    (m.cutoff, m.peak, m.limb_protection, m.include_sky) = (s.cutoff, s.peak, s.limb_protection, s.include_sky);
    p.align.params = structure.align.params;
    let (own, d) = (&mut p.dering.params, &structure.dering.params);
    (own.radius, own.edges, own.tolerance, own.region) = (d.radius, d.edges, d.tolerance, d.region);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Confidence;

    fn analysis(kind: ImageKind, snr: f32, psf_sigma: Option<f32>) -> Analysis {
        Analysis {
            kind,
            confidence: Confidence::High,
            disk: None,
            noise: 0.01,
            snr,
            psf_sigma,
            psf_moffat: None,
            channel_sigma: None,
            rgb_shift: None,
            subject_size: 450.0,
            spectral: None,
            power: Vec::new(),
            optics: None,
        }
    }

    fn deconv(p: &Pipeline) -> &DeconvParams {
        &p.deconv.params
    }

    #[test]
    fn measured_blur_is_used_for_the_detected_kind_only() {
        let a = analysis(ImageKind::Planet, 100.0, Some(2.0));
        assert_eq!(deconv(&auto_pipeline(ImageKind::Planet, &a)).psf, Psf::Gaussian { sigma: 1.7 });
        assert_eq!(deconv(&auto_pipeline(ImageKind::Surface, &a)).psf, Psf::Gaussian { sigma: 1.6 });
    }

    #[test]
    fn noisy_images_get_more_damping() {
        let clean = auto_pipeline(ImageKind::Planet, &analysis(ImageKind::Planet, 200.0, None));
        let noisy = auto_pipeline(ImageKind::Planet, &analysis(ImageKind::Planet, 20.0, None));
        assert!(deconv(&noisy).damping > deconv(&clean).damping);
    }

    #[test]
    fn wavelet_levels_follow_subject_size() {
        let mut a = analysis(ImageKind::Planet, 100.0, Some(1.5));
        a.subject_size = 300.0;
        let small = wavelet_structure(ImageKind::Planet, &a, ScaleMode::Free).layers.len();
        a.subject_size = 2400.0;
        let large = wavelet_structure(ImageKind::Planet, &a, ScaleMode::Free).layers.len();
        assert!(small < large, "{small} vs {large}");
        assert!((4..=MAX_LAYERS).contains(&small) && (4..=MAX_LAYERS).contains(&large));
    }

    #[test]
    fn free_scales_start_where_the_detail_ends() {
        let side = 512;
        let mut a = analysis(ImageKind::Surface, 100.0, None);
        let sharp = crate::spectrum::analyze(&crate::spectrum::tests::degraded(side, 1.2, 0.004), side, side, 256.0, 256.0).unwrap();
        let soft = crate::spectrum::analyze(&crate::spectrum::tests::degraded(side, 2.4, 0.004), side, side, 256.0, 256.0).unwrap();
        a.spectral = Some(sharp);
        let fine = wavelet_structure(ImageKind::Surface, &a, ScaleMode::Free);
        a.spectral = Some(soft);
        let coarse = wavelet_structure(ImageKind::Surface, &a, ScaleMode::Free);
        assert_eq!(fine.scale_mode, ScaleMode::Free);
        assert!(coarse.first_scale > fine.first_scale + 0.3, "{} vs {}", fine.first_scale, coarse.first_scale);
        assert!((fine.first_scale - 0.3 / sharp.cutoff).abs() < 0.05);
    }

    /// Switching scales and back gives about the same gains, and the thresholds come back to
    /// their layers.
    #[test]
    fn switching_scales_keeps_the_settings() {
        let a = analysis(ImageKind::Planet, 100.0, Some(1.8));
        let classic = auto_pipeline_with(ImageKind::Planet, &a, AutoStrength::Normal, ScaleMode::Dyadic).wavelet.params;
        let free = convert_scales(ImageKind::Planet, &a, &classic, ScaleMode::Free);
        assert_eq!(free.scale_mode, ScaleMode::Free);
        assert_eq!(free.grain_denoise, classic.layers[0].denoise);
        assert!(free.layers[0].gain > 0.0);
        let back = convert_scales(ImageKind::Planet, &a, &free, ScaleMode::Dyadic);
        for (b, c) in back.layers.iter().zip(&classic.layers).take(3) {
            assert!((b.gain - c.gain).abs() < 0.35, "{:?} vs {:?}", back.layers, classic.layers);
            assert_eq!(b.denoise, c.denoise);
        }
    }

    /// With a spectrum the PSF is a Moffat whose core comes from where the detail ends, not
    /// the limb's (much wider) profile; the telescope's diffraction limit caps that frequency.
    #[test]
    fn the_spectrum_sets_the_psf_core_and_the_optics_cap_it() {
        let side = 512;
        let mut a = analysis(ImageKind::Planet, 100.0, Some(6.0));
        let spectral = crate::spectrum::analyze(&crate::spectrum::tests::degraded(side, 1.4, 0.004), side, side, 256.0, 256.0).unwrap();
        a.spectral = Some(spectral);
        let Psf::Moffat { fwhm, .. } = deconv_structure(ImageKind::Planet, &a).psf else { panic!("a Moffat PSF") };
        assert!((fwhm - 2.3548 * 0.3416 / spectral.cutoff).abs() < 0.01 && fwhm < 2.3548 * 6.0 * 0.6, "fwhm {fwhm}");
        let free = first_scale(ImageKind::Planet, &a);
        let setup = crate::equipment::Setup {
            telescope: Some(crate::equipment::Telescope { name: "x".into(), aperture_mm: 60.0, focal_mm: 3000.0, obstruction: 0.0 }),
            camera: Some(crate::equipment::Camera { name: "y".into(), pixel_um: 2.9, colour: false }),
            ..Default::default()
        };
        a.optics = setup.optics();
        assert!(a.optics.unwrap().cutoff < spectral.cutoff, "{:?} vs {spectral:?}", a.optics);
        assert!(first_scale(ImageKind::Planet, &a) > free + 0.2);
        assert!(explain(ImageKind::Planet, &a, ScaleMode::Free).wavelet.iter().any(|t| t.contains("diffrazione")));
    }

    #[test]
    fn oversampled_images_start_at_step_two() {
        let a = analysis(ImageKind::Planet, 100.0, Some(3.5));
        assert_eq!(wavelet_structure(ImageKind::Planet, &a, ScaleMode::Dyadic).base_step, 2);
    }

    /// Everything that would change the image, in [`initial_pipeline`] and [`neutral_pipeline`].
    fn changes_something(p: &Pipeline) -> bool {
        let on = |enabled: bool, neutral: bool| enabled && !neutral;
        p.bxt.enabled
            || on(p.align.enabled, p.align.params.is_identity())
            || on(p.deconv.enabled, p.deconv.params.is_neutral())
            || on(p.denoise.enabled, p.denoise.params.is_neutral())
            || on(p.wavelet.enabled, p.wavelet.params.is_neutral())
            || on(p.contrast.enabled, p.contrast.params.is_neutral())
            || on(p.dering.enabled, p.dering.params.is_neutral())
            || on(p.tone.enabled, p.tone.params.is_neutral())
            || on(p.curves.enabled, p.curves.params.is_neutral())
            || on(p.color.enabled, p.color.params.is_neutral())
            || on(p.mineral.enabled, p.mineral.params.is_neutral())
    }

    #[test]
    fn initial_pipeline_leaves_the_image_untouched() {
        let p = initial_pipeline(ImageKind::Planet, &analysis(ImageKind::Planet, 100.0, Some(2.0)), ScaleMode::Free);
        assert!(!changes_something(&p));
        // Modules are on, so moving a control acts at once…
        assert!(p.denoise.enabled && p.wavelet.enabled && p.curves.enabled);
        // …but the structure is already chosen from the analysis. The deconvolution, with no
        // amount, waits switched off but ready, iterations included.
        assert!(!p.deconv.enabled && p.deconv.params.iterations > 0);
        assert_eq!(p.deconv.params.psf, Psf::Gaussian { sigma: 1.7 });
    }

    #[test]
    fn strength_orders_the_automatic_settings() {
        let a = analysis(ImageKind::Planet, 100.0, Some(2.0));
        let p = |s| auto_pipeline_with(ImageKind::Planet, &a, s, ScaleMode::Free);
        let (g, n, s) = (p(AutoStrength::Gentle), p(AutoStrength::Normal), p(AutoStrength::Strong));
        assert!(g.deconv.params.iterations < n.deconv.params.iterations && n.deconv.params.iterations < s.deconv.params.iterations);
        assert!(g.wavelet.params.layers[0].gain < n.wavelet.params.layers[0].gain && n.wavelet.params.layers[0].gain < s.wavelet.params.layers[0].gain);
        assert!(n.deconv.params.auto_stop, "iterations stop on their own");
    }

    #[test]
    fn neutral_pipeline_changes_nothing() {
        assert!(!changes_something(&neutral_pipeline()));
    }

    #[test]
    fn colour_balance_keeps_a_planets_hue() {
        // Saturn-like pale yellow: the Moon setting makes it grey, the planet setting keeps a tint.
        let n = 400;
        let img = crate::Image::new(n, 1, vec![vec![0.5; n], vec![0.45; n], vec![0.3; n]], 16);
        let moon = auto_color(ImageKind::FullDisk, &img);
        let planet = auto_color(ImageKind::Saturn, &img);
        assert!((moon[2] * 0.3 - 0.45).abs() < 1e-3, "{moon:?}");
        assert!(planet[2] * 0.3 < 0.44 && planet[2] > 1.0, "{planet:?}");
    }

    #[test]
    fn dering_follows_the_blur_and_skips_surfaces() {
        let a = analysis(ImageKind::Planet, 100.0, Some(3.0));
        let d = auto_pipeline(ImageKind::Planet, &a).dering;
        assert!(d.enabled && d.params.radius >= 2);
        assert!(!auto_pipeline(ImageKind::Surface, &a).dering.enabled);
        assert!(initial_pipeline(ImageKind::Planet, &a, ScaleMode::Free).dering.params.is_neutral());
        assert_eq!(initial_pipeline(ImageKind::Saturn, &a, ScaleMode::Free).dering.params.region, DeringRegion::Globe);
    }

    #[test]
    fn generic_structure_keeps_amounts_and_forgets_the_image() {
        let a = analysis(ImageKind::Saturn, 100.0, Some(2.0));
        let mut p = initial_pipeline(ImageKind::Saturn, &a, ScaleMode::Free);
        p.deconv.params.iterations = 33;
        let g = generic_structure(&p);
        assert_eq!(g.deconv.params.iterations, 33);
        assert_eq!(g.deconv.params.psf, Psf::default());
        assert_eq!(g.dering.params.region, DeringRegion::Subject);
        assert_eq!(restructure(ImageKind::Saturn, &a, &g).deconv.params.psf, p.deconv.params.psf, "and back");
    }

    #[test]
    fn restructure_keeps_the_users_amounts() {
        let a = analysis(ImageKind::Planet, 100.0, Some(2.0));
        let mut p = initial_pipeline(ImageKind::Planet, &a, ScaleMode::Free);
        p.wavelet.params.layers[0].gain = 3.0;
        p.deconv.params.iterations = 42;
        let q = restructure(ImageKind::Saturn, &a, &p);
        assert_eq!(q.wavelet.params.layers[0].gain, 3.0);
        assert_eq!(q.deconv.params.iterations, 42);
    }
}
