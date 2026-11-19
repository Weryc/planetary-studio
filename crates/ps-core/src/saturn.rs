//! Saturn in two zones, the globe and the rings: masks, measures, the sharpening curve and a
//! noise estimate for each zone.
//!
//! The globe is soft (belts, limb darkening) while the rings are sharp edges and thin divisions
//! against the sky, and they are not equally bright. Everything here works in the background:
//! the app shows one "learned" curve and one "network" noise reduction as for Jupiter.
//!
//! Measured on simulated stacks (2026-11-18, 102 stacks from 13 originals), applied as the app
//! applies them: the globe's network gives the best curve on the globe and, with no edge
//! protection, also on the rings (better than a rings-only network or one trained on both), so
//! the sharpening uses that one curve everywhere. The noise is estimated per zone, blended by a
//! soft mask of the globe, and reaches what the true noise would allow. A rings curve of its
//! own stays supported ([`crate::smart::SmartParams::network_rings`]) for a future network.
//!
//! What the networks see (the bands of [`crate::curve_net`]), in a square framing the whole ring
//! system:
//! - the power of each band inside the globe and on the rings (log, over the globe's mean level
//!   squared, so brightness doesn't matter);
//! - the sky around the rings, where there is only noise (missing when clipped to black);
//! - the analysis' cut-off and limb blur, noise and SNR, the ring span, how open the rings are
//!   and how bright against the globe.

use crate::Image;
use crate::analysis::Analysis;
use crate::color::{ColorMode, working_planes};
use crate::curve_net::{BANDS, SKY_FROM};
use crate::mask::Ellipse;

/// Ring system's span over the globe's equatorial diameter (A ring's outer edge).
const RING_SPAN_OVER_GLOBE: f32 = 2.27;

/// The globe, and how much of each pixel belongs to it (the planet's curve) rather than to the
/// rings (the rings' curve): 1 inside, 0 outside, a soft step of a few pixels at the limb.
pub struct Zones {
    pub globe: Ellipse,
    /// The ring system's major extent, px.
    pub span: f32,
    pub planet: Vec<f32>,
}

/// Where the globe's weight falls from 1 to 0, as a normalised ellipse radius.
const STEP: (f32, f32) = (0.97, 1.05);

/// The two zones of a Saturn image. `None` when no ring system is found.
pub fn zones(img: &Image) -> Option<Zones> {
    let globe = crate::mask::saturn_globe(img)?;
    if globe.rx < 4.0 {
        return None;
    }
    let planet = globe_weight(&globe, img.width(), img.height(), (0, 0));
    Some(Zones { globe, span: 2.0 * RING_SPAN_OVER_GLOBE * globe.rx, planet })
}

/// The globe's soft weight (see [`Zones::planet`]) over a `w`×`h` image lying at `offset` in the
/// one `globe` was found on.
pub fn globe_weight(globe: &Ellipse, w: usize, h: usize, offset: (usize, usize)) -> Vec<f32> {
    let (ox, oy) = (offset.0 as f32, offset.1 as f32);
    (0..w * h)
        .map(|i| {
            let r = globe.radius((i % w) as f32 + 0.5 + ox, (i / w) as f32 + 0.5 + oy);
            let t = ((STEP.1 - r) / (STEP.1 - STEP.0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })
        .collect()
}

/// The square measured on a Saturn image and its three pixel sets.
pub struct Frame {
    pub crop: Image,
    pub x0: usize,
    pub y0: usize,
    pub side: usize,
    pub span: f32,
    /// Inside the globe (88 % of its outline), on the rings (the lit subject outside the globe,
    /// a pixel or two in from its edges), and the sky well away from both.
    pub planet: Vec<usize>,
    pub rings: Vec<usize>,
    pub sky: Vec<usize>,
    /// The globe's mean luminance: every power is relative to its square.
    pub mean: f64,
    /// The rings' mean luminance over the globe's.
    pub ring_level: f32,
}

impl Frame {
    /// Log10 power of each band of `plane` over `px`, over the globe's mean squared.
    pub fn power(&self, plane: &[f32], px: &[usize]) -> Vec<f32> {
        crate::curve_net::bands(plane, self.side).iter().map(|band| ((px.iter().map(|&i| (band[i] as f64).powi(2)).sum::<f64>() / px.len().max(1) as f64) / self.mean.powi(2)).max(1e-14).log10() as f32).collect()
    }
}

/// Frame a Saturn image: a square around the whole ring system.
pub fn frame(img: &Image) -> Option<Frame> {
    let z = zones(img)?;
    let (w, h) = (img.width(), img.height());
    let side = crate::fft::next_fast_size(((1.25 * z.span) as usize).max(128)).min(w).min(h);
    if side < 64 {
        return None;
    }
    let x0 = ((z.globe.cx - side as f32 / 2.0).round().max(0.0) as usize).min(w - side);
    let y0 = ((z.globe.cy - side as f32 / 2.0).round().max(0.0) as usize).min(h - side);
    let crop = img.crop(x0, y0, side, side);
    let lum = working_planes(&crop, ColorMode::Luminance).swap_remove(0);
    let globe = Ellipse { cx: z.globe.cx - x0 as f32, cy: z.globe.cy - y0 as f32, ..z.globe };
    let subject = crate::mask::subject_and_moons(&crop);
    // The rings where they are lit: the subject's mask fills the dark gap between globe and
    // rings (and edge-on rings' shadow), which is sky, not ring.
    let lit: Vec<f32> = crate::mask::foreground(&lum, side, side).into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect();
    let lit = crate::mask::gaussian_approx(&lit, side, side, 1.0);
    let smooth = crate::mask::gaussian_approx(&lum, side, side, 1.5);
    let border = 8;
    let within = |i: usize, b: usize| (b..side - b).contains(&(i % side)) && (b..side - b).contains(&(i / side));
    let at = |i: usize| ((i % side) as f32 + 0.5, (i / side) as f32 + 0.5);
    let planet: Vec<usize> = (0..side * side).filter(|&i| within(i, border) && globe.radius(at(i).0, at(i).1) < 0.88).collect();
    let planet_mean = planet.iter().map(|&i| lum[i] as f64).sum::<f64>() / planet.len().max(1) as f64;
    let rings: Vec<usize> = (0..side * side).filter(|&i| within(i, border) && lit[i] > 0.99 && smooth[i] as f64 > 0.12 * planet_mean && globe.radius(at(i).0, at(i).1) > 1.08).collect();
    let sky: Vec<usize> = (0..side * side)
        .filter(|&i| within(i, 16) && subject[i] < 0.01 && (at(i).0 - globe.cx).hypot(at(i).1 - globe.cy) > 0.62 * z.span)
        .collect();
    if planet.len() < 200 {
        return None;
    }
    let mean_of = |px: &[usize]| px.iter().map(|&i| lum[i] as f64).sum::<f64>() / px.len().max(1) as f64;
    let mean = mean_of(&planet);
    if mean <= 0.0 {
        return None;
    }
    let ring_level = (mean_of(&rings) / mean) as f32;
    Some(Frame { crop, x0, y0, side, span: z.span, planet, rings, sky, mean, ring_level })
}

/// Fewer ring pixels than this (rings edge-on, a thin line) and the rings' bands are not
/// measured: the globe's stand in, and a flag says so.
pub const MIN_RING_PIXELS: usize = 300;

/// What the sharpening networks see on a Saturn image.
pub struct Measures {
    pub planet: Vec<f32>,
    pub rings: Option<Vec<f32>>,
    pub sky: Option<Vec<f32>>,
    pub sky_note: Option<&'static str>,
    pub cutoff: f32,
    pub edge_sigma: Option<f32>,
    pub span: f32,
    /// Ring pixels over globe pixels (how open the rings are) and their brightness over the globe's.
    pub ring_share: f32,
    pub ring_level: f32,
    pub noise_rel: f32,
    pub snr: f32,
}

/// Measure a Saturn image (its analysis `a`), with the frame it was measured in.
pub fn measure(img: &Image, a: &Analysis) -> Option<(Measures, Frame)> {
    let f = frame(img)?;
    let lum = working_planes(&f.crop, ColorMode::Luminance).swap_remove(0);
    let planet = f.power(&lum, &f.planet);
    let rings = (f.rings.len() >= MIN_RING_PIXELS).then(|| f.power(&lum, &f.rings));
    let probe = crate::curve_net::Frame { crop: f.crop.clone(), x0: f.x0, y0: f.y0, side: f.side, diameter: f.span, inside: f.planet.clone(), sky: f.sky.clone(), mean: f.mean };
    let sky_note = probe.sky_problem(&lum);
    let sky = sky_note.is_none().then(|| f.power(&lum, &f.sky));
    let m = Measures {
        planet,
        rings,
        sky,
        sky_note,
        cutoff: a.spectral.as_ref().map_or(0.0, |s| s.cutoff),
        edge_sigma: a.psf_sigma,
        span: f.span,
        ring_share: f.rings.len() as f32 / f.planet.len() as f32,
        ring_level: f.ring_level,
        noise_rel: (a.noise as f64 / f.mean) as f32,
        snr: a.snr,
    };
    Some((m, f))
}

/// The networks' inputs (the same function builds the training set).
pub fn features(m: &Measures) -> Vec<f32> {
    let mut f = m.planet.clone();
    match &m.rings {
        Some(r) => {
            f.extend_from_slice(r);
            f.push(1.0);
        }
        None => {
            f.extend_from_slice(&m.planet);
            f.push(0.0);
        }
    }
    match &m.sky {
        Some(sky) => {
            f.extend_from_slice(&sky[SKY_FROM..]);
            f.extend(m.planet[SKY_FROM..].iter().zip(&sky[SKY_FROM..]).map(|(s, n)| s - n));
            f.push(1.0);
        }
        None => {
            f.extend_from_slice(&m.planet[SKY_FROM..]);
            f.extend(std::iter::repeat_n(0.0, BANDS - SKY_FROM));
            f.push(0.0);
        }
    }
    f.push(m.cutoff);
    f.push(if m.cutoff > 0.0 { 1.0 } else { 0.0 });
    f.push(m.edge_sigma.unwrap_or(5.0));
    f.push(if m.edge_sigma.is_some() { 1.0 } else { 0.0 });
    f.push(m.span.max(1.0).log10());
    f.push(m.ring_share.max(1e-3).log10());
    f.push(m.ring_level.max(1e-3).log10());
    f.push(m.noise_rel.max(1e-7).log10());
    f.push(m.snr.max(1.0).log10());
    f
}

/// Number of sharpening inputs.
pub const INPUTS: usize = 2 * BANDS + 1 + 2 * (BANDS - SKY_FROM) + 1 + 9;

/// The noise estimator's inputs: the sharpening's, then the colour differences (R − Y plus
/// B − Y) of the globe, the rings and the sky (zeros and flags when missing).
pub fn noise_features(m: &Measures, f: &Frame) -> Vec<f32> {
    let mut x = features(m);
    match crate::noise_net::chroma_planes(&f.crop) {
        Some([cr, cb]) => {
            let both = |px: &[usize]| -> Vec<f32> { f.power(&cr, px).iter().zip(f.power(&cb, px)).map(|(a, b)| (10f32.powf(*a) + 10f32.powf(b)).log10()).collect() };
            let p = both(&f.planet);
            x.extend_from_slice(&p[1..]);
            match m.rings.is_some() {
                true => x.extend_from_slice(&both(&f.rings)[1..]),
                false => x.extend_from_slice(&p[1..]),
            }
            x.push(1.0);
            match m.sky.is_some() {
                true => x.extend_from_slice(&both(&f.sky)[SKY_FROM..]),
                false => x.extend_from_slice(&p[SKY_FROM..]),
            }
            x.push(if m.sky.is_some() { 1.0 } else { 0.0 });
        }
        None => x.extend(std::iter::repeat_n(0.0, 2 * (BANDS - 1) + 1 + (BANDS - SKY_FROM) + 1)),
    }
    x
}

/// Number of noise inputs.
pub const NOISE_INPUTS: usize = INPUTS + 2 * (BANDS - 1) + 1 + (BANDS - SKY_FROM) + 1;

/// The two sharpening curves the Saturn networks suggest, as the smart sharpening takes them:
/// (cut-off, strength, shape) for the globe and for the rings.
pub type Curve = (f32, f32, Vec<(f32, f32)>);

/// The globe's curve for a Saturn image and the rings' own when there is a rings network, or
/// why there is none. Without a rings network the globe's curve goes everywhere.
pub fn suggestion(img: &Image, a: &Analysis) -> Result<(Curve, Option<Curve>, f32, Option<&'static str>), &'static str> {
    let (planet_model, rings_model) = crate::saturn_data::sharpen_models().ok_or("nessuna rete per Saturno")?;
    let (m, _) = measure(img, a).ok_or("anelli e globo non trovati")?;
    let x = features(&m);
    let p = planet_model.predict(&x).ok_or("misure incompatibili con la rete")?;
    let r = match rings_model {
        Some(model) => Some(model.predict(&x).ok_or("misure incompatibili con la rete")?),
        None => None,
    };
    let spread = r.as_ref().map_or(p.spread, |r| p.spread.max(r.spread));
    Ok((crate::curve_net::smart_curve(&p), r.as_ref().map(crate::curve_net::smart_curve), spread, m.sky_note))
}

/// The two zones' noise estimates from [`noise_features`]: (globe grain, globe colour, rings
/// grain, rings colour), log10 power of bands from [`crate::noise_net::FIRST_BAND`].
pub fn noise_estimates(x: &[f32]) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let (planet_model, rings_model) = crate::saturn_data::noise_models()?;
    let (pl, pc) = planet_model.predict(x)?;
    let (rl, rc) = rings_model.predict(x)?;
    Some((pl, pc, rl, rc))
}

/// Blend two versions of an image by the globe's weight (`planet` 1 = the first).
pub fn blend(planet_version: &Image, rings_version: &Image, planet: &[f32]) -> Image {
    let (w, h) = (planet_version.width(), planet_version.height());
    let planes = planet_version.channels().iter().zip(rings_version.channels()).map(|(a, b)| a.iter().zip(b).zip(planet).map(|((a, b), t)| b + t * (a - b)).collect()).collect();
    Image::new(w, h, planes, planet_version.source_bit_depth)
}

/// What the noise reduction measures on a Saturn image: the globe's and the rings' band power
/// and estimated noise, and the globe that blends them (see [`crate::noise_net::NoisePlan`]).
/// `None` without models or zones.
pub fn noise_plan(img: &Image) -> Option<crate::noise_net::NoisePlan> {
    let (planet_model, rings_model) = crate::saturn_data::noise_models()?;
    let a = crate::analysis::analyze(img);
    let (m, f) = measure(img, &a)?;
    let x = noise_features(&m, &f);
    let (pl, pc) = planet_model.predict(&x)?;
    let (rl, rc) = rings_model.predict(&x)?;
    let (y, chroma) = crate::noise_net::planes(&f.crop);
    let chroma_power = |px: &[usize]| chroma.as_ref().map(|[cr, cb]| f.power(cr, px).iter().zip(f.power(cb, px)).map(|(a, b)| (10f32.powf(*a) + 10f32.powf(b)).log10()).collect::<Vec<f32>>());
    let ring_px: &[usize] = if m.rings.is_some() { &f.rings } else { &f.planet };
    let globe = crate::mask::saturn_globe(img)?;
    crate::log::info(
        "rumore",
        &format!(
            "Saturno: immagine {}×{}, globo centro ({:.0}, {:.0}) raggio {:.0}, riquadro {} px, pixel globo {} anelli {}{} cielo {}{}",
            img.width(), img.height(), globe.cx, globe.cy, globe.rx, f.side, f.planet.len(), f.rings.len(),
            if m.rings.is_none() { " (troppo pochi: si usa il globo)" } else { "" },
            f.sky.len(), m.sky_note.map_or(String::new(), |n| format!(" ({n})")),
        ),
    );
    let zone = |name: &str, px: &[usize], noise: Vec<f32>, colour: Vec<f32>| crate::noise_net::ZoneNoise {
        name: name.to_string(),
        lum: (f.power(&y, px), noise),
        colour: chroma_power(px).map(|c| (c, colour)),
    };
    Some(crate::noise_net::NoisePlan { zones: vec![zone("Saturno, globo", &f.planet, pl, pc), zone("Saturno, anelli", ring_px, rl, rc)], globe: Some(globe) })
}

/// Noise reduction guided by the Saturn estimators, one for the globe and one for the rings:
/// each zone's bands keep the share that is not noise there (as [`crate::noise_net::guided_denoise`]),
/// and the two results are blended by the globe's weight. `None` without models or zones.
pub fn guided_denoise(img: &Image, strength: f32) -> Option<Image> {
    Some(crate::noise_net::apply_plan(img, &noise_plan(img)?, (0, 0), strength))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic Saturn: an oblate globe and a ring annulus around it, tilted.
    fn saturn(side: usize) -> Image {
        let (cx, cy, rg) = (side as f32 / 2.0, side as f32 / 2.0, side as f32 / 9.0);
        let p = (0..side * side)
            .map(|i| {
                let (x, y) = ((i % side) as f32 - cx, (i / side) as f32 - cy);
                let globe = (x / rg).powi(2) + (y / (0.9 * rg)).powi(2) < 1.0;
                let rr = (x / rg).powi(2) + (y / (0.35 * rg)).powi(2);
                let ring = (1.5 * 1.5..2.27 * 2.27).contains(&rr);
                if globe { 0.8 } else if ring { 0.6 } else { 0.01 + 0.002 * ((i * 7919 % 13) as f32) }
            })
            .collect();
        Image::new(side, side, vec![p], 16)
    }

    /// A zoomed region reduced with the whole image's plan matches the whole image reduced, away
    /// from the region's edges (the app's ROI preview).
    #[test]
    fn region_matches_the_whole_image() {
        let clean = saturn(512);
        let noisy: Vec<f32> = clean.channels()[0].iter().enumerate().map(|(i, v)| v + 0.03 * ((((i as u64).wrapping_mul(2654435761) >> 7) % 1000) as f32 / 1000.0 - 0.5)).collect();
        let img = Image::new(512, 512, vec![noisy], 16);
        let plan = noise_plan(&img).expect("plan");
        let whole = crate::noise_net::apply_plan(&img, &plan, (0, 0), 2.0);
        let (x0, y0, w, h) = (180, 200, 220, 160);
        let part = crate::noise_net::apply_plan(&img.crop(x0, y0, w, h), &plan, (x0, y0), 2.0);
        let (a, b) = (&whole.channels()[0], &part.channels()[0]);
        let mut worst = 0.0f32;
        for y in 32..h - 32 {
            for x in 32..w - 32 {
                worst = worst.max((a[(y + y0) * 512 + x + x0] - b[y * w + x]).abs());
            }
        }
        assert!(worst < 0.01, "{worst}");
    }

    #[test]
    fn finds_the_globe_and_the_rings() {
        let img = saturn(512);
        let z = zones(&img).expect("zones");
        let (cx, cy) = (256usize, 256usize);
        assert!(z.planet[cy * 512 + cx] > 0.99);
        // On the ring, along the major axis, well past the globe.
        let ring_x = cx + (2.0 * 512.0 / 9.0) as usize;
        assert!(z.planet[cy * 512 + ring_x] < 0.01, "{}", z.planet[cy * 512 + ring_x]);
        let f = frame(&img).expect("frame");
        assert!(f.planet.len() > 1000 && f.rings.len() > MIN_RING_PIXELS && f.sky.len() > 500, "{} {} {}", f.planet.len(), f.rings.len(), f.sky.len());
        // The ring's edges in a sharp synthetic image add some sky to its mean.
        assert!((0.55..0.85).contains(&f.ring_level), "{}", f.ring_level);
        let a = crate::analysis::analyze(&img);
        let (m, _) = measure(&img, &a).unwrap();
        assert_eq!(features(&m).len(), INPUTS);
    }
}

#[cfg(test)]
mod export_check {
    /// The exported networks answer as in training: `SATURN_NET_CHECK` names the
    /// `reti_controllo.json` that `saturn_train.py --export` writes beside the evaluation.
    #[test]
    fn exported_networks_match_training() {
        let Ok(path) = std::env::var("SATURN_NET_CHECK") else { return };
        let Some((planet, rings)) = crate::saturn_data::sharpen_models() else { return };
        let models: Vec<(&crate::curve_net::Model, &str)> = std::iter::once((planet, "pianeta")).chain(rings.map(|r| (r, "anelli"))).collect();
        let (noise_planet, _) = crate::saturn_data::noise_models().unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        let list = |item: &str, key: &str| -> Vec<f32> {
            let at = item.find(&format!("\"{key}\": [")).unwrap() + key.len() + 5;
            let end = item[at..].find(']').unwrap() + at;
            item[at..end].split(',').map(|v| v.trim().parse().unwrap()).collect()
        };
        let mut checked = 0;
        for item in text.split("{\"x\"").skip(1) {
            let item = format!("{{\"x\"{item}");
            let (x, xn) = (list(&item, "x"), list(&item, "xn"));
            for &(model, key) in &models {
                let p = model.predict(&x).unwrap();
                for (a, b) in p.gains.iter().zip(list(&item, key)) {
                    assert!((a / b - 1.0).abs() < 1e-3, "{key}: {a} vs {b}");
                }
            }
            let (l, c) = noise_planet.predict(&xn).unwrap();
            for (a, b) in l.iter().chain(&c).zip(list(&item, "rumore_pianeta")) {
                assert!((a - b).abs() < 1e-3, "rumore: {a} vs {b}");
            }
            checked += 1;
        }
        assert!(checked > 0);
    }
}
