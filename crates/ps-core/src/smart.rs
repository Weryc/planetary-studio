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

use crate::Image;
use crate::color::{ColorMode, merge_planes, working_planes};
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
    /// Sharpen each zone by its own detail and noise (else uniformly).
    pub adaptive: bool,
    /// Shape of the boost across sizes of detail.
    pub curve: SmartCurve,
    pub color: ColorMode,
    /// Sharpen the sky too; by default only the subject (planet, disk, rings, moons) is.
    pub include_sky: bool,
    /// Roll the brightest values off before they reach white, so they never clip.
    pub protect_highlights: bool,
}

/// Shape of the boost across sizes of detail, relative to the cut-off. The default is the one
/// measured on BlurXTerminator; the others are variations of it designed for particular cases
/// (not measured).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmartCurve {
    #[default]
    Standard,
    PoorSeeing,
    GoodSeeing,
    Surface,
    SmoothGlobe,
}

impl SmartCurve {
    pub const ALL: [SmartCurve; 5] = [Self::Standard, Self::PoorSeeing, Self::GoodSeeing, Self::Surface, Self::SmoothGlobe];

    pub fn label(self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::PoorSeeing => "Seeing scarso",
            Self::GoodSeeing => "Seeing ottimo",
            Self::Surface => "Luna e superfici",
            Self::SmoothGlobe => "Globi lisci",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Standard => "Come BlurXTerminator sulle immagini planetarie: il massimo sul dettaglio più fine presente, un po' anche sulle strutture grandi.",
            Self::PoorSeeing => "Sposta la spinta sulle strutture più grandi (bande, macchie) e lascia stare il dettaglio più fine, che con seeing scarso è soprattutto rumore.",
            Self::GoodSeeing => "Spinge fino al dettaglio più fine e un po' oltre, tocca meno le strutture grandi. Per immagini nitide e poco rumorose.",
            Self::Surface => "Spinta larga su tutte le dimensioni: crateri piccoli e grandi, montagne, bordi di ombre. Per la Luna e le riprese ravvicinate.",
            Self::SmoothGlobe => "Più dolce sul dettaglio finissimo, dove un globo liscio (Saturno, Urano, Venere) mostra solo grana: valorizza bande e divisioni senza macchiettare.",
        }
    }

    /// Log-gain over the peak's, by frequency over the cut-off (linear in log frequency).
    fn table(self) -> &'static [(f32, f32)] {
        match self {
            // BlurXTerminator's response on real stacks (median of eight, see the module notes).
            Self::Standard => &[(0.10, 0.0), (0.18, 0.10), (0.25, 0.21), (0.35, 0.35), (0.5, 0.55), (0.7, 0.80), (0.85, 0.93), (1.0, 1.0), (1.15, 0.91), (1.3, 0.76), (1.5, 0.48), (1.7, 0.24), (2.0, 0.03), (2.2, 0.0)],
            Self::PoorSeeing => &[(0.08, 0.0), (0.15, 0.15), (0.25, 0.35), (0.35, 0.52), (0.5, 0.76), (0.7, 1.0), (0.85, 0.95), (1.0, 0.8), (1.15, 0.55), (1.3, 0.3), (1.5, 0.1), (1.7, 0.0)],
            Self::GoodSeeing => &[(0.15, 0.0), (0.25, 0.12), (0.35, 0.22), (0.5, 0.4), (0.7, 0.65), (0.85, 0.85), (1.0, 0.97), (1.15, 1.0), (1.3, 0.9), (1.5, 0.68), (1.7, 0.42), (2.0, 0.15), (2.3, 0.0)],
            Self::Surface => &[(0.07, 0.0), (0.15, 0.22), (0.25, 0.4), (0.35, 0.55), (0.5, 0.72), (0.7, 0.88), (0.85, 0.97), (1.0, 1.0), (1.15, 0.92), (1.3, 0.75), (1.5, 0.45), (1.7, 0.2), (2.0, 0.02), (2.2, 0.0)],
            Self::SmoothGlobe => &[(0.10, 0.0), (0.18, 0.15), (0.25, 0.3), (0.35, 0.48), (0.5, 0.72), (0.7, 0.93), (0.85, 1.0), (1.0, 0.92), (1.15, 0.72), (1.3, 0.48), (1.5, 0.22), (1.7, 0.05), (1.9, 0.0)],
        }
    }

    fn shape(self, r: f32) -> f32 {
        let t = self.table();
        let (first, last) = (t[0], t[t.len() - 1]);
        if r <= first.0 || r >= last.0 {
            return 0.0;
        }
        let i = t.iter().position(|q| q.0 >= r).unwrap_or(t.len() - 1).max(1);
        let (a, b) = (t[i - 1], t[i]);
        a.1 + (b.1 - a.1) * (r / a.0).ln() / (b.0 / a.0).ln()
    }

    /// Past this multiple of the cut-off there is only grain: it starts being cleaned.
    fn grain_from(self) -> f32 {
        self.table().last().map_or(1.9, |q| 0.86 * q.0)
    }

    /// Where the finest boosted layer should peak, over the cut-off.
    fn finest(self) -> f32 {
        self.table().last().map_or(1.8, |q| 0.82 * q.0)
    }
}

impl Default for SmartParams {
    fn default() -> Self {
        Self {
            strength: 0.0,
            cutoff: 0.0,
            peak: 1.0,
            limb_protection: 0.3,
            adaptive: true,
            curve: SmartCurve::Standard,
            color: ColorMode::Luminance,
            include_sky: false,
            protect_highlights: true,
        }
    }
}

/// Peak gain at strength 1: BlurXTerminator's typical peak on real stacks (median ×10–12).
const PEAK_AT_FULL: f32 = 12.0;
/// The strength can go past BlurXTerminator's typical result.
pub const MAX_STRENGTH: f32 = 1.9;

impl SmartParams {
    /// Share of the grain past the cut-off that is removed: follows the strength.
    pub fn effective_cleaning(&self) -> f32 {
        0.6 * self.strength.clamp(0.0, 1.0)
    }

    pub fn is_neutral(&self) -> bool {
        self.strength <= 0.0
    }

    /// What [`prepare`] depends on (besides the images).
    pub fn prepare_key(&self) -> PrepareKey {
        PrepareKey { cutoff: self.cutoff.to_bits(), peak: self.peak.to_bits(), curve: self.curve, adaptive: self.adaptive, color: self.color }
    }

    /// The overall response wanted at `f` given what earlier steps did (`done`): never below 1
    /// where there is detail (what they sharpened beyond the target stays), below 1 past the
    /// detail (grain, and noise they amplified, is cleaned).
    fn wanted(&self, f: f32, cutoff: f32, done: &Done) -> f32 {
        let want = self.target(f, cutoff) / done.at(f);
        if f <= self.curve.grain_from() * self.peak * cutoff { want.max(1.0) } else { want }
    }

    /// Target overall response at `f` cycles/px for the cut-off `cutoff`.
    pub fn target(&self, f: f32, cutoff: f32) -> f32 {
        let r = f.max(1e-4) / (self.peak * cutoff).max(1e-3);
        let peak = 1.0 + (PEAK_AT_FULL - 1.0) * self.strength.clamp(0.0, MAX_STRENGTH);
        let t = ((r - self.curve.grain_from()) / 0.6).clamp(0.0, 1.0);
        let beyond = t * t * (3.0 - 2.0 * t);
        peak.powf(self.curve.shape(r)) * (1.0 - 0.9 * self.effective_cleaning() * beyond)
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
/// apart up to ~25 px. Everything finer is grain.
fn transform(peak: f32) -> WaveletParams {
    let ratio = std::f32::consts::SQRT_2;
    let first = (layer_peak(1.0, ratio) / peak.max(0.04)).clamp(1.0, 5.0);
    let levels = (((25.0 / first).ln() / ratio.ln()).ceil() as usize + 1).clamp(4, 10);
    WaveletParams { scale_mode: ScaleMode::Free, first_scale: first, ratio, layers: vec![WaveletLayer::default(); levels], noise_adaptive: 0.0, edge_protection: 0.0, ..Default::default() }
}

/// The cut-off to use: the given one, else measured on the plane, else a typical one.
fn cutoff_of(p: &SmartParams, plane: &[f32], w: usize, h: usize) -> f32 {
    if p.cutoff > 0.0 {
        return p.cutoff.min(0.5);
    }
    crate::spectrum::analyze(plane, w, h, w as f32 / 2.0, h as f32 / 2.0).map_or(0.2, |s| s.cutoff)
}

/// How much sharpening already happened, by frequency: √(P_now / P_before), 1 if unknown.
#[derive(Clone, Debug, Default)]
struct Done(Vec<(f32, f32)>);

impl Done {
    fn measure(now: &[f32], before: &[f32], w: usize, h: usize) -> Done {
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        match (crate::spectrum::radial_power(now, w, h, cx, cy), crate::spectrum::radial_power(before, w, h, cx, cy)) {
            (Some(a), Some(b)) => {
                let ratio: Vec<(f32, f32)> = a.iter().zip(&b).map(|(&(f, pa), &(_, pb))| (f, (pa / pb.max(1e-30)).sqrt().clamp(0.3, 20.0))).collect();
                if ratio.iter().all(|r| (r.1 - 1.0).abs() < 1e-3) { Done::default() } else { Done(ratio) }
            }
            _ => Done::default(),
        }
    }

    fn at(&self, f: f32) -> f32 {
        // Nearest band (they are fine, logarithmic).
        self.0.iter().min_by(|a, b| (a.0 / f).ln().abs().total_cmp(&(b.0 / f).ln().abs())).map_or(1.0, |r| r.1)
    }
}

/// Brightness relative to the subject's, per pixel: photon noise grows with it.
fn brightness(plane: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut sample: Vec<f32> = plane.iter().step_by(plane.len().div_ceil(1 << 16).max(1)).copied().collect();
    sample.sort_by(f32::total_cmp);
    let top = sample.get(sample.len() * 99 / 100).copied().unwrap_or(1.0).max(1e-6);
    let lit: Vec<f32> = sample.iter().copied().filter(|&v| v > 0.15 * top).collect();
    let reference = lit.get(lit.len() / 2).copied().unwrap_or(top);
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
fn limb_band(edges: &[f32], w: usize, h: usize, fp: f32) -> Vec<f32> {
    let r = (1.2 / fp.max(1e-3)).clamp(3.0, 20.0);
    let (_, grown) = crate::dering::min_max(edges, w, h, r.round() as usize);
    crate::mask::gaussian_approx(&grown, w, h, r / 3.0).into_iter().map(|v| v.clamp(0.0, 1.0)).collect()
}

/// What sharpening needs that doesn't depend on the strength or the limb protection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrepareKey {
    cutoff: u32,
    peak: u32,
    curve: SmartCurve,
    adaptive: bool,
    color: ColorMode,
}

/// One working plane, ready to be sharpened at any strength.
struct PreparedPlane {
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
}

impl Prepared {
    /// Memory held (bytes).
    pub fn bytes(&self) -> usize {
        self.planes.iter().map(|p| 4 * (p.band.len() + p.layers.iter().map(|l| l.1.len()).sum::<usize>())).sum()
    }
}

/// Decompose and weight `img`; `before` is the image before any earlier sharpening.
pub fn prepare(img: &Image, before: Option<&Image>, p: &SmartParams) -> Prepared {
    let (w, h) = (img.width(), img.height());
    let planes = working_planes(img, p.color);
    let before_planes = before.filter(|b| b.width() == w && b.height() == h).map(|b| working_planes(b, p.color));
    // One plane at a time: each is already processed in parallel, and memory stays low.
    let planes = planes.iter().enumerate().map(|(k, plane)| prepare_plane(plane, before_planes.as_ref().map(|b| b[k].as_slice()), w, h, p)).collect();
    Prepared { planes }
}

fn prepare_plane(plane: &[f32], before: Option<&[f32]>, w: usize, h: usize, p: &SmartParams) -> PreparedPlane {
    let cutoff = cutoff_of(p, before.unwrap_or(plane), w, h);
    let fp = p.peak * cutoff;
    let done = before.map_or_else(Done::default, |b| Done::measure(plane, b, w, h));
    let transform = transform(p.curve.finest() * fp);
    let Some(mut d) = decompose(plane, w, h, &transform, &|| false) else {
        return PreparedPlane { cutoff, transform, layers: Vec::new(), band: vec![0.0; plane.len()], done };
    };
    let band = limb_band(d.edges(), w, h, fp);
    let light = if p.adaptive { Some(brightness(plane, w, h)) } else { None };
    let mut layers = Vec::new();
    for (j, mut layer) in std::mem::take(&mut d.layers).into_iter().enumerate() {
        // Layers the curve never boosts, at any strength, aren't needed.
        if p.curve.shape(layer_peak(transform.scale(j), transform.ratio) / fp) <= 0.0 {
            continue;
        }
        if let Some(light) = &light {
            let share = layer_share(&layer, d.white_noise[j], light, w, h, transform.scale(j));
            layer.par_iter_mut().zip(share).for_each(|(v, s)| *v *= s);
        }
        layers.push((j, layer));
    }
    PreparedPlane { cutoff, transform, layers, band, done }
}

/// The sharpened image from [`prepare`]d `img`; `subject` (0‥1 per pixel, same size) is where
/// it applies when the sky is left out.
pub fn apply(prepared: &Prepared, img: &Image, p: &SmartParams, subject: Option<&[f32]>) -> Image {
    let (w, h) = (img.width(), img.height());
    let subject = subject.filter(|m| !p.include_sky && m.len() == w * h);
    let planes = working_planes(img, p.color);
    let processed = planes.iter().zip(&prepared.planes).map(|(plane, pp)| apply_plane(plane, pp, w, h, p, subject)).collect();
    let mut out = merge_planes(img, p.color, &planes, processed);
    if p.protect_highlights {
        out.channels_mut().par_iter_mut().zip(img.channels()).for_each(|(o, before)| o.iter_mut().zip(before).for_each(|(v, &b)| *v = roll_off(*v, b)));
    }
    out
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
    let want = |f: f32| p.wanted(f, cutoff, &pp.done);
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
        if t > 1.0 && e > 1e-3 { ((t - 1.0) / e).min(8.0) } else { 0.0 }
    };
    let clean = |f: f32| want(f).min(1.0) - 1.0;
    let mut out = add_filtered(plane, &added, w, h, &shape, &clean);
    if let Some(m) = subject {
        out.par_iter_mut().zip(plane).zip(m).for_each(|((o, &v), &m)| *o = v + m.clamp(0.0, 1.0) * (*o - v));
    }
    out
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
    let transform = transform(p.curve.finest() * fp);
    let Some(d) = decompose(plane, w, h, &transform, &|| false) else { return (vec![0.0; plane.len()], vec![0.0; plane.len()]) };
    let protect = p.limb_protection.clamp(0.0, 1.0);
    let edges = if protect > 0.0 { limb_band(d.edges(), w, h, fp).into_iter().map(|e| protect * e).collect() } else { vec![0.0; plane.len()] };
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
    let limb = limb_band(d.edges(), w, h, cutoff);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::tests::degraded;

    #[test]
    fn the_target_peaks_below_the_cutoff_and_cleans_beyond() {
        let p = SmartParams { strength: 1.0, ..Default::default() };
        let c = 0.15;
        let at = |f: f32| p.target(f, c);
        assert!((at(c) - PEAK_AT_FULL).abs() < 0.01, "peak {}", at(c));
        assert!(at(c) > at(0.5 * c) && at(0.5 * c) > at(0.25 * c) && at(0.25 * c) > 1.0);
        // Like BlurXTerminator: still strong at 1.5× the cut-off, the grain cut past 2×.
        assert!(at(1.5 * c) > 2.5, "1.5× cut-off: {}", at(1.5 * c));
        assert!(at(0.45) < 0.5, "beyond the cut-off: {}", at(0.45));
        assert!(SmartParams::default().is_neutral());
    }

    /// On a blurred, noisy scene: gentle sharpening brings it closer to the sharp scene;
    /// following the local detail does better than sharpening everywhere; fine noise on a flat
    /// field isn't amplified, and is cleaned when pushed hard.
    #[test]
    fn sharpens_the_detail_and_cleans_the_grain() {
        let side = 256;
        let sharp = crate::spectrum::tests::scene(side);
        let blurred = degraded(side, 1.8, 0.0);
        let noisy = degraded(side, 1.8, 0.004);
        let img = Image::new(side, side, vec![noisy.clone()], 16);
        let rms = |a: &[f32], b: &[f32]| (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32).sqrt();
        let error = |p: SmartParams| rms(&sharpen(&img, None, &p, None).channels()[0], &sharp);
        let gentle = SmartParams { strength: 0.05, ..Default::default() };
        assert!(error(gentle.clone()) < rms(&noisy, &sharp));
        let stronger = SmartParams { strength: 0.15, ..Default::default() };
        assert!(error(stronger.clone()) < error(SmartParams { adaptive: false, ..stronger.clone() }));
        let flat: Vec<f32> = noisy.iter().zip(&blurred).map(|(n, b)| 0.5 + n - b).collect();
        let flat_img = Image::new(side, side, vec![flat.clone()], 16);
        let spread = |v: &[f32]| rms(v, &vec![0.5; v.len()]);
        // Gentle: the grain is left about as it is (not amplified where there is no detail)…
        let gentle = sharpen(&flat_img, None, &SmartParams { cutoff: 0.2, ..stronger }, None);
        assert!(spread(&gentle.channels()[0]) < 1.03 * spread(&flat), "grain {} → {}", spread(&flat), spread(&gentle.channels()[0]));
        // …strong: the grain past twice the cut-off is cleaned.
        let full = sharpen(&flat_img, None, &SmartParams { cutoff: 0.12, strength: 1.0, ..Default::default() }, None);
        assert!(spread(&full.channels()[0]) < 0.9 * spread(&flat), "grain {} → {}", spread(&flat), spread(&full.channels()[0]));
    }

    /// Every curve peaks near the cut-off at the chosen height, leaves coarse structure gently
    /// boosted and cleans the grain; "Seeing scarso" favours coarser detail than "Seeing ottimo".
    #[test]
    fn the_curves_are_sane() {
        let c = 0.15;
        for curve in SmartCurve::ALL {
            let p = SmartParams { strength: 1.0, curve, ..Default::default() };
            let best = (40..=300).map(|k| p.target(k as f32 * 0.001, c)).fold(0.0f32, f32::max);
            assert!((best - PEAK_AT_FULL).abs() < 0.5, "{curve:?} peak {best}");
            assert!(p.target(0.25 * c, c) >= 1.0 && p.target(0.25 * c, c) < 4.0, "{curve:?} coarse {}", p.target(0.25 * c, c));
            assert!(p.target(0.49, c) < 0.6, "{curve:?} grain {}", p.target(0.49, c));
        }
        let at = |curve, f: f32| SmartParams { strength: 1.0, curve, ..Default::default() }.target(f, c);
        assert!(at(SmartCurve::PoorSeeing, 0.5 * c) > at(SmartCurve::GoodSeeing, 0.5 * c));
        assert!(at(SmartCurve::PoorSeeing, 1.4 * c) < at(SmartCurve::GoodSeeing, 1.4 * c));
    }

    /// On a disk against black sky: the edge protection holds back at the limb only, "Segui le
    /// zone" in the sky more than inside the detailed disk.
    #[test]
    fn the_hold_back_map_shows_limb_and_sky() {
        let side = 256;
        let scene = degraded(side, 1.5, 0.003);
        let r = 80.0;
        let plane: Vec<f32> = (0..side * side)
            .map(|i| {
                let (x, y) = ((i % side) as f32 - 128.0, (i / side) as f32 - 128.0);
                if x.hypot(y) < r { 0.3 + 0.5 * scene[i] } else { 0.01 + (scene[i] - 0.5) * 0.02 }
            })
            .collect();
        let (zones, edges) = hold_back_map(&plane, side, side, &SmartParams { strength: 0.5, limb_protection: 1.0, cutoff: 0.2, ..Default::default() });
        let at = |v: &[f32], x: usize, y: usize| v[y * side + x];
        assert!(at(&edges, 128 + 80, 128) > 0.3, "limb {}", at(&edges, 128 + 80, 128));
        // Also just outside the limb, where a sharpened limb leaves its halo, and just inside.
        assert!(at(&edges, 128 + 86, 128) > 0.2, "halo zone {}", at(&edges, 128 + 86, 128));
        assert!(at(&edges, 128 + 75, 128) > 0.2, "inside the limb {}", at(&edges, 128 + 75, 128));
        assert!(at(&edges, 128, 128) < 0.05 && at(&edges, 128 + 40, 128) < 0.05);
        let mean = |v: &[f32], pts: &[(usize, usize)]| pts.iter().map(|&(x, y)| at(v, x, y)).sum::<f32>() / pts.len() as f32;
        let sky = mean(&zones, &[(10, 10), (240, 20), (20, 240), (245, 245)]);
        let disk = mean(&zones, &[(128, 128), (110, 140), (150, 110), (130, 160)]);
        assert!(sky > disk + 0.2, "held back in the sky {sky}, in the disk {disk}");
    }

    #[test]
    fn finer_detail_shows_on_the_map() {
        let side = 256;
        let fine = degraded(side, 1.0, 0.002);
        let coarse = degraded(side, 3.0, 0.002);
        // Left half fine, right half coarse.
        let plane: Vec<f32> = (0..side * side).map(|i| if i % side < side / 2 { fine[i] } else { coarse[i] }).collect();
        let map = resolution_map(&plane, side, side, &SmartParams { cutoff: 0.3, ..Default::default() }, None);
        let mean = |range: std::ops::Range<usize>| {
            let v: Vec<f32> = (32..side - 32).flat_map(|y| range.clone().map(move |x| (x, y))).map(|(x, y)| map[y * side + x]).filter(|&v| v > 0.0).collect();
            v.iter().sum::<f32>() / v.len().max(1) as f32
        };
        let (left, right) = (mean(32..side / 2 - 32), mean(side / 2 + 32..side - 32));
        assert!(left < right, "finest detail {left} px (sharp half) vs {right} px (blurred half)");
    }

    /// A disk on a noisy sky: with the subject's mask the sky is left exactly as it was.
    #[test]
    fn the_sky_is_left_alone() {
        let side = 256;
        let scene = degraded(side, 1.5, 0.003);
        let plane: Vec<f32> = (0..side * side)
            .map(|i| {
                let (x, y) = ((i % side) as f32 - 128.0, (i / side) as f32 - 128.0);
                if x.hypot(y) < 70.0 { 0.3 + 0.5 * scene[i] } else { 0.02 + (scene[i] - 0.5) * 0.05 }
            })
            .collect();
        let img = Image::new(side, side, vec![plane.clone()], 16);
        let subject = crate::mask::subject_and_moons(&img);
        let p = SmartParams { strength: 1.0, cutoff: 0.2, ..Default::default() };
        let out = sharpen(&img, None, &p, Some(&subject));
        let o = &out.channels()[0];
        for (i, (&m, (&a, &b))) in subject.iter().zip(o.iter().zip(&plane)).enumerate() {
            if m == 0.0 {
                assert_eq!(a, b, "sky changed at {i}");
            }
        }
        // Inside, it is sharpened.
        assert!(o.iter().zip(&plane).zip(&subject).any(|((a, b), &m)| m == 1.0 && (a - b).abs() > 0.01));
        // Asked to, the sky is sharpened too.
        let all = sharpen(&img, None, &SmartParams { include_sky: true, ..p }, Some(&subject));
        assert!(all.channels()[0].iter().zip(&plane).zip(&subject).any(|((a, b), &m)| m == 0.0 && a != b));
    }

    /// Bright detail pushed hard never reaches white; without the protection it would.
    #[test]
    fn whites_never_clip() {
        let side = 192;
        let scene = degraded(side, 1.5, 0.002);
        let plane: Vec<f32> = scene.iter().map(|v| (0.55 + 1.2 * (v - 0.5)).clamp(0.0, 0.97)).collect();
        let img = Image::new(side, side, vec![plane.clone()], 16);
        let p = SmartParams { strength: MAX_STRENGTH, cutoff: 0.2, adaptive: false, ..Default::default() };
        let top = |img: &Image| img.channels()[0].iter().copied().fold(0.0f32, f32::max);
        let unprotected = sharpen(&img, None, &SmartParams { protect_highlights: false, ..p.clone() }, None);
        assert!(top(&unprotected) > 1.0, "test needs clipping: {}", top(&unprotected));
        let protected = sharpen(&img, None, &p, None);
        assert!(top(&protected) < 1.0, "white {}", top(&protected));
        // Below the knee nothing changes.
        assert_eq!(roll_off(0.5, 0.4), 0.5);
        assert!(roll_off(1.4, 0.8) < 1.0 && roll_off(1.4, 0.8) > roll_off(1.1, 0.8));
    }

    /// Where "Segui le zone" holds back (smooth, noisy zone beside detailed ones), the image is
    /// left about as it was: its large structures keep their contrast instead of being
    /// flattened by a correction meant for the boosted zones.
    #[test]
    fn held_back_zones_keep_their_structure() {
        let side = 256;
        let detail = degraded(side, 1.2, 0.002);
        let noise: Vec<f32> = degraded(side, 1.2, 0.004).iter().zip(&detail).map(|(n, d)| n - d).collect();
        // Right half: a smooth wave (period 64 px) and noise only.
        let plane: Vec<f32> = (0..side * side)
            .map(|i| {
                let x = i % side;
                if x < side / 2 { detail[i] } else { 0.5 + 0.03 * (std::f32::consts::TAU * x as f32 / 64.0).sin() + noise[i] }
            })
            .collect();
        let img = Image::new(side, side, vec![plane.clone()], 16);
        let out = sharpen(&img, None, &SmartParams { strength: 1.0, cutoff: 0.15, ..Default::default() }, None);
        let smooth = |v: &[f32]| crate::mask::gaussian_approx(v, side, side, 4.0);
        let (a, b) = (smooth(&plane), smooth(&out.channels()[0]));
        let spread = |v: &[f32]| {
            let pts: Vec<f32> = (40..side - 40).flat_map(|y| (side / 2 + 32..side - 32).map(move |x| y * side + x)).map(|i| v[i]).collect();
            let m = pts.iter().sum::<f32>() / pts.len() as f32;
            (pts.iter().map(|p| (p - m).powi(2)).sum::<f32>() / pts.len() as f32).sqrt()
        };
        let ratio = spread(&b) / spread(&a);
        assert!((0.9..1.5).contains(&ratio), "large structure in the held-back zone ×{ratio}");
    }
}
