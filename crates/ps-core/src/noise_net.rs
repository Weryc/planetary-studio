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

/// Detail coarser than this (cycles/px, a period of about 14 px) is never reduced.
pub const SMALLEST_NOISE_FREQUENCY: f32 = 0.07;

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
/// [`crate::curve_net::band_of`]). With `margin` > 0 the plane is mirror-padded first: an image
/// cut through the subject (a zoomed region) would otherwise ring, in stripes, from its border
/// wrapping onto the opposite one.
fn filter_bands(plane: &[f32], w: usize, h: usize, gains: &[f32], margin: usize) -> Vec<f32> {
    let padded = (margin > 0).then(|| crate::fft::Padded::new(plane, w, h, margin));
    let (data, pw, ph) = padded.as_ref().map_or((plane, w, h), |p| (p.data.as_slice(), p.w, p.h));
    let plan = crate::fft::FftConvolver::plan(pw, ph);
    let mut spec = plan.forward(data);
    // Transposed layout: index = kx·h + ky, kx ∈ 0..=w/2.
    for (idx, c) in spec.iter_mut().enumerate() {
        let (kx, ky) = (idx / ph, idx % ph);
        let fy = if ky <= ph / 2 { ky as f32 } else { ky as f32 - ph as f32 } / ph as f32;
        let g = gains[crate::curve_net::band_of((kx as f32 / pw as f32).hypot(fy))];
        c.re *= g;
        c.im *= g;
    }
    let out = plan.inverse(spec);
    match padded {
        Some(p) => p.crop(&out),
        None => out,
    }
}

/// Mirror padding for [`filter_bands`]: [`PAD`] px when the subject reaches the border of
/// `lum` (its border holds light near the subject's), none when the border is sky (as on a
/// whole frame: there padding changes nothing that matters).
fn border_margin(lum: &[f32], w: usize, h: usize) -> usize {
    let mut border: Vec<f32> = (0..w * h).filter(|&i| i % w < 2 || i % w >= w - 2 || i / w < 2 || i / w >= h - 2).map(|i| lum[i]).collect();
    let mut all: Vec<f32> = lum.iter().step_by(7).copied().collect();
    if border.is_empty() || all.is_empty() {
        return 0;
    }
    let pick = |v: &mut Vec<f32>, q: f32| {
        let k = ((v.len() - 1) as f32 * q) as usize;
        *v.select_nth_unstable_by(k, f32::total_cmp).1
    };
    let (edge, top) = (pick(&mut border, 0.95), pick(&mut all, 0.995));
    if edge > 0.15 * top { PAD.min(w.max(h)) } else { 0 }
}

/// Each band's Wiener gain from its measured power and estimated noise (log10, noise from
/// [`FIRST_BAND`]), `strength` multiplying the noise.
fn band_gains(strength: f32, power: &[f32], noise: &[f32]) -> Vec<f32> {
    let mut g = vec![1.0f32; BANDS];
    for (j, b) in (FIRST_BAND..BANDS).enumerate() {
        // Larger detail is left alone: there two stacks of the same recording differ by the
        // seeing's leftover deformation, not by noise a reduction could remove.
        if crate::curve_net::band_centre(b) < SMALLEST_NOISE_FREQUENCY {
            continue;
        }
        let ratio = 10f32.powf(noise[j] - power[b]);
        g[b] = (1.0 - strength.max(0.0) * ratio).clamp(0.0, 1.0);
    }
    // Never more of a finer band than of a coarser one: a band removed between two kept
    // ones (a notch) rings around edges, and the finest bands are mostly noise anyway.
    for b in FIRST_BAND + 1..BANDS {
        g[b] = g[b].min(g[b - 1]);
    }
    g
}

/// Mirror padding of [`filter_bands`] on each side (px), when needed.
const PAD: usize = 32;

/// What the guided noise reduction measures on an image, kept apart from applying it so that a
/// zoomed region is reduced as the whole image is (with the whole image's estimate): each
/// zone's band power and estimated noise, and for Saturn the globe that splits the two zones.
#[derive(Clone, Debug)]
pub struct NoisePlan {
    /// One zone (a disk), or Saturn's globe then its rings.
    pub zones: Vec<ZoneNoise>,
    /// Saturn's globe, in the measured image's pixels: the zones are blended by its soft mask.
    pub globe: Option<crate::mask::Ellipse>,
}

/// A zone's measures: log10 band power (all bands) and estimated noise (from [`FIRST_BAND`]),
/// for the luminance and, on a colour image, the colour differences.
#[derive(Clone, Debug)]
pub struct ZoneNoise {
    pub name: String,
    pub lum: (Vec<f32>, Vec<f32>),
    pub colour: Option<(Vec<f32>, Vec<f32>)>,
}

/// Measure `img` for the guided noise reduction. `None` without a disk (or Saturn) to measure.
pub fn plan(img: &Image) -> Option<NoisePlan> {
    let a = crate::analysis::analyze(img);
    // Saturn: the globe and the rings each with their own estimate (when trained).
    if a.kind == crate::analysis::ImageKind::Saturn
        && let Some(p) = crate::saturn::noise_plan(img)
    {
        return Some(p);
    }
    let (m, _) = measure(img, &a)?;
    let (noise_lum, noise_col) = model().predict(&features(&m))?;
    let colour = m.chroma_subject.clone().map(|c| (c, noise_col));
    Some(NoisePlan { zones: vec![ZoneNoise { name: format!("disco ({:?})", a.kind), lum: (m.lum.subject.clone(), noise_lum), colour }], globe: None })
}

/// Reduce the noise of `img` with a plan measured on an image `img` sits in at `offset`
/// (x, y): the whole image itself (0, 0), or the whole image around a zoomed region.
pub fn apply_plan(img: &Image, plan: &NoisePlan, offset: (usize, usize), strength: f32) -> Image {
    let reduce = |z: &ZoneNoise| denoise_with(img, strength, (&z.lum.0, &z.lum.1), z.colour.as_ref().map(|(c, n)| (c.as_slice(), n.as_slice())), &z.name);
    match (plan.zones.as_slice(), plan.globe) {
        ([globe_zone, rings_zone], Some(globe)) => {
            let weight = crate::saturn::globe_weight(&globe, img.width(), img.height(), offset);
            crate::saturn::blend(&reduce(globe_zone), &reduce(rings_zone), &weight)
        }
        (zones, _) => reduce(&zones[0]),
    }
}

/// Noise reduction guided by the estimator: each frequency band of the brightness and of the
/// colour keeps the share of it that is not noise (a Wiener gain from the band's measured power
/// inside the subject and the estimated noise power). `strength` multiplies the estimated noise
/// (1 = as estimated, 0 = nothing removed). `None` without a disk to measure on.
pub fn guided_denoise(img: &Image, strength: f32) -> Option<Image> {
    Some(apply_plan(img, &plan(img)?, (0, 0), strength))
}

/// The band Wiener reduction of [`guided_denoise`] from given measures: each band's power where
/// the estimate holds and the estimated noise there (from [`FIRST_BAND`]), for the luminance
/// and, on a colour image, for the colour differences.
pub fn denoise_with(img: &Image, strength: f32, lum: (&[f32], &[f32]), colour: Option<(&[f32], &[f32])>, zone: &str) -> Image {
    let gains = |power: &[f32], noise: &[f32]| band_gains(strength, power, noise);
    let (w, h) = (img.width(), img.height());
    let (y, chroma) = planes(img);
    let gl = gains(lum.0, lum.1);
    let margin = border_margin(&y, w, h);
    let gc_log = colour.map(|(c, n)| gains(c, n));
    let list = |g: &[f32]| g[FIRST_BAND..].iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" ");
    crate::log::info("rumore", &format!("{zone}: immagine {w}×{h}{}, forza {strength:.2}, guadagni grana (bande {FIRST_BAND}‥) {} | colore {}", if margin > 0 { " (soggetto sul bordo)" } else { "" }, list(&gl), gc_log.as_deref().map_or("-".into(), list)));
    let y2 = filter_bands(&y, w, h, &gl, margin);
    let (Some([cr, cb]), Some((cs, noise_col))) = (chroma, colour) else {
        return if img.is_mono() { Image::new(w, h, vec![y2], img.source_bit_depth) } else { crate::color::merge_planes(img, ColorMode::Luminance, &[y], vec![y2]) };
    };
    let gc = gains(cs, noise_col);
    let (cr2, cb2) = (filter_bands(&cr, w, h, &gc, margin), filter_bands(&cb, w, h, &gc, margin));
    let r: Vec<f32> = y2.iter().zip(&cr2).map(|(y, c)| y + c).collect();
    let b: Vec<f32> = y2.iter().zip(&cb2).map(|(y, c)| y + c).collect();
    let g: Vec<f32> = (0..w * h).map(|i| (y2[i] - LUMA[0] * r[i] - LUMA[2] * b[i]) / LUMA[1]).collect();
    Image::new(w, h, vec![r, g, b], img.source_bit_depth)
}

#[cfg(test)]
mod tests {

    /// Unit gains change nothing, also on an image cut through a bright subject.
    #[test]
    fn unit_gains_keep_the_image() {
        let (w, h) = (90, 60);
        let plane: Vec<f32> = (0..w * h).map(|i| if i % w > 60 { 1.0 } else { 0.05 + 0.01 * ((i * 7) % 5) as f32 }).collect();
        let out = super::filter_bands(&plane, w, h, &[1.0; super::BANDS], super::border_margin(&plane, w, h));
        assert!(plane.iter().zip(&out).all(|(a, b)| (a - b).abs() < 1e-4));
    }

    /// Removing fine detail on an image whose right side is bright (a region cut through the
    /// globe) leaves no stripes far from the edge: the wrap-around seam is in the padding.
    #[test]
    fn no_stripes_from_the_border() {
        let (w, h) = (120, 80);
        let plane: Vec<f32> = (0..w * h).map(|i| if i % w >= 100 { 1.0 } else { 0.0 }).collect();
        let mut gains = [1.0f32; super::BANDS];
        gains[12..].iter_mut().for_each(|g| *g = 0.0);
        assert!(super::border_margin(&plane, w, h) > 0);
        let out = super::filter_bands(&plane, w, h, &gains, super::border_margin(&plane, w, h));
        // The left edge, which the FFT wraps next to the bright right edge, stays dark.
        let worst = (0..h).flat_map(|y| (0..40).map(move |x| y * w + x)).map(|i| out[i].abs()).fold(0.0f32, f32::max);
        assert!(worst < 0.01, "{worst}");
    }

    /// The gains never rise again toward finer bands.
    #[test]
    fn gains_fall_with_frequency() {
        let power = vec![-6.0f32; super::BANDS];
        // Noise above the power in some middle bands only.
        let noise: Vec<f32> = (super::FIRST_BAND..super::BANDS).map(|b| if (16..20).contains(&b) { -5.0 } else { -8.0 }).collect();
        let g = super::band_gains(1.0, &power, &noise);
        assert!(g.windows(2).all(|p| p[1] <= p[0]), "{g:?}");
        assert_eq!(g[23], 0.0);
    }
    /// The exported networks answer as in Python: `NOISE_NET_CHECK` names a JSON list of
    /// {"x": inputs, "p": expected outputs}.
    #[test]
    fn exported_networks_match_training() {
        let Ok(path) = std::env::var("NOISE_NET_CHECK") else { return };
        let text = std::fs::read_to_string(path).unwrap();
        let mut checked = 0;
        for item in text.split("{\"x\": [").skip(1) {
            let (xs, rest) = item.split_once("], \"p\": [").unwrap();
            let ps = rest.split(']').next().unwrap();
            let parse = |s: &str| s.split(',').map(|v| v.trim().parse::<f32>().unwrap()).collect::<Vec<_>>();
            let (x, p) = (parse(xs), parse(ps));
            let (l, c) = super::model().predict(&x).unwrap();
            for (a, b) in l.iter().chain(&c).zip(&p) {
                assert!((a - b).abs() < 1e-3, "{a} vs {b}");
            }
            checked += 1;
        }
        assert!(checked > 0);
    }
}
