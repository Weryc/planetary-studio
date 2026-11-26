//! Derotazione di Giove e Saturno: più riprese della stessa sera, fatte a qualche minuto
//! l'una dall'altra, portate allo stesso istante e sommate.
//!
//! Ogni immagine mostra il pianeta ruotato di `360° · Δt / periodo` rispetto all'istante di
//! riferimento. Per ogni pixel del risultato si trova il punto del globo che vi si vede
//! (sferoide schiacciato, inclinato verso di noi di B), lo si sposta in longitudine di quanto
//! il pianeta è girato fra le due ore, lo si riporta sul piano dell'immagine di partenza e la
//! si campiona (Catmull–Rom). Fuori dal globo (cielo, anelli di Saturno, che sono simmetrici
//! e non sembrano girare) le immagini sono solo allineate. Le parti che all'ora della ripresa
//! erano dietro il bordo non esistono in quell'immagine: lì pesano le altre.
//!
//! Il disco di ogni immagine (centro e raggio dall'analisi, orientamento dalle bande o dagli
//! anelli) dà l'allineamento iniziale; [`refine_centres`] lo corregge con la correlazione di
//! fase delle immagini già derotate. La somma è una media pesata, con sigma-clipping
//! facoltativo per scartare i valori anomali (un difetto in una sola ripresa).
//!
//! Limiti: tutte le immagini condividono raggio e orientamento (stessa sera, stessa
//! strumentazione); la rotazione è rigida (un solo periodo: i Sistemi I e II di Giove differiscono
//! di 5 minuti, che in 10 minuti di intervallo fanno 0,1°); su Saturno gli anelli davanti al
//! globo e la loro ombra seguono il globo nella derotazione.

use crate::analysis::{Analysis, ImageKind};
use crate::capture::{Body, Moment};
use crate::image::Image;
use rayon::prelude::*;
use std::f64::consts::TAU;

/// Jupiter's System I (equatorial region), hours: 9h 50m 30s.
pub const JUPITER_SYSTEM_I_HOURS: f64 = 9.0 + 50.0 / 60.0 + 30.0 / 3600.0;
/// Jupiter's System II (the rest of the globe), hours: 9h 55m 40.6s.
pub const JUPITER_SYSTEM_II_HOURS: f64 = 9.0 + 55.0 / 60.0 + 40.6 / 3600.0;
/// Saturn (System III), hours: 10h 33m 38s.
pub const SATURN_HOURS: f64 = 10.0 + 33.0 / 60.0 + 38.0 / 3600.0;

/// Mars (sidereal day), hours: 24h 37m 22.66s.
pub const MARS_HOURS: f64 = 24.0 + 37.0 / 60.0 + 22.66 / 3600.0;

/// Which of Jupiter's rotation systems to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum JupiterSystem {
    /// The equatorial zone, a few minutes faster.
    I,
    /// Belts and the Great Red Spot: most of the globe.
    #[default]
    II,
}

/// Rotation period in hours of a planet that can be derotated (Jupiter, Saturn, Mars).
pub fn rotation_period(body: Body, system: JupiterSystem) -> Option<f64> {
    match (body, system) {
        (Body::Jupiter, JupiterSystem::I) => Some(JUPITER_SYSTEM_I_HOURS),
        (Body::Jupiter, JupiterSystem::II) => Some(JUPITER_SYSTEM_II_HOURS),
        (Body::Saturn, _) => Some(SATURN_HOURS),
        (Body::Mars, _) => Some(MARS_HOURS),
        _ => None,
    }
}

/// The planet's tilt towards the Earth at `when` (sub-Earth latitude B), degrees: positive
/// when the north pole is turned towards us.
pub fn tilt(body: Body, when: &Moment) -> f64 {
    match body {
        Body::Jupiter => crate::moons::jupiter_tilt(when),
        Body::Saturn => crate::identify::saturn_ring_tilt(when).unwrap_or(0.0),
        _ => crate::ephemeris::sub_earth_latitude(body, when).unwrap_or(0.0),
    }
}

/// Hours from `b` to `a` (positive when `a` comes later).
pub fn hours_between(a: &Moment, b: &Moment) -> f64 {
    (a.julian() - b.julian()) * 24.0
}

/// The weighted mean of some moments.
pub fn mean_moment(times: &[(Moment, f32)]) -> Option<Moment> {
    let total: f64 = times.iter().map(|(_, w)| *w as f64).sum();
    if times.is_empty() || total <= 0.0 {
        return None;
    }
    // Relative to the first, so the Julian dates' large integer part doesn't eat the precision.
    let base = times[0].0.julian();
    let offset: f64 = times.iter().map(|(t, w)| (t.julian() - base) * *w as f64).sum::<f64>() / total;
    Some(Moment::from_julian(base + offset))
}

pub use crate::globe::Globe;

/// The globe as the analysis found it: Saturn's from its rings (whose direction is the
/// equator's), another planet's disk with the equator along its bands (horizontal when they
/// are not clear). `None` without a whole disk.
pub fn find_globe(img: &Image, analysis: &Analysis) -> Option<Globe> {
    if analysis.kind == ImageKind::Saturn {
        let e = crate::mask::saturn_globe(img)?;
        return Some(Globe { cx: e.cx, cy: e.cy, radius: e.rx, angle: fold_angle(e.angle) });
    }
    let d = analysis.disk?;
    let angle = analysis.bands.as_ref().map_or(0.0, |b| fold_angle(b.angle));
    Some(Globe { cx: d.cx, cy: d.cy, radius: d.radius, angle })
}

/// A direction (a line has two) folded into (−90°, 90°].
fn fold_angle(a: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let a = a.rem_euclid(pi);
    if a > pi / 2.0 { a - pi } else { a }
}

/// How the planet turns and is seen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Rotation period, hours.
    pub period_hours: f64,
    /// Period of the equatorial zone when it differs (Jupiter's System I), hours; only the
    /// automatic derotation ([`auto`]) uses it.
    pub equator_period_hours: Option<f64>,
    /// Sub-Earth latitude B, degrees.
    pub tilt_deg: f64,
    /// Polar radius / equatorial radius ([`crate::moons::flattening`]).
    pub oblateness: f64,
    /// The image is mirrored (a diagonal or a Newtonian's odd number of reflections): the
    /// surface seems to turn the other way.
    pub mirrored: bool,
}

impl Params {
    pub fn for_body(body: Body, system: JupiterSystem, when: Option<&Moment>) -> Option<Params> {
        Some(Params {
            period_hours: rotation_period(body, system)?,
            equator_period_hours: None,
            tilt_deg: when.map_or(0.0, |w| tilt(body, w)),
            oblateness: crate::moons::flattening(body),
            mirrored: false,
        })
    }

    /// Longitude the planet turns through in `hours`, radians, in the direction the surface
    /// moves on the image (left to right with north up, unmirrored).
    fn turn(&self, hours: f64) -> f64 {
        let a = TAU * hours / self.period_hours;
        if self.mirrored { -a } else { a }
    }
}

use crate::globe::{Projection, turn};

pub mod auto;

/// One image to sum.
#[derive(Clone, Copy, Debug)]
pub struct Frame<'a> {
    pub image: &'a Image,
    pub globe: Globe,
    /// Hours from the reference instant to the capture (negative when taken before it).
    pub hours: f64,
    /// Weight in the sum (> 0).
    pub weight: f32,
    /// Brightness factor per channel.
    pub gain: [f32; 3],
}

/// Where in `frame` the output pixel (x, y) of the reference globe `reference` was, and how
/// much it counts there (1 off the globe; less towards the limb; almost nothing where the
/// point was hidden, then the plain alignment is used).
fn source(frame: &Frame, reference: &Globe, proj: &Projection, params: &Params, x: f32, y: f32) -> ((f32, f32), f32) {
    let (u, v) = reference.to_plane(x, y);
    let (su, sv, vis) = match proj.to_globe(u, v) {
        Some(p) => {
            // Where the feature seen at the reference instant was at the capture.
            let ((su, sv), mu) = proj.to_plane(turn(p, params.turn(frame.hours)));
            if mu > 0.0 { (su, sv, smoothstep(mu / 0.2).max(1e-4)) } else { (u, v, 1e-4) }
        }
        None => (u, v, 1.0),
    };
    (frame.globe.to_pixel(su, sv), vis)
}

pub(crate) fn smoothstep(t: f64) -> f32 {
    let t = t.clamp(0.0, 1.0);
    (t * t * (3.0 - 2.0 * t)) as f32
}

/// Sum the frames brought to the reference instant on the reference globe's grid
/// (`size` pixels, sampled at `scale` for a preview). `clip`: sigma-clipping threshold, in
/// standard deviations (from 3 frames up). `None` without frames.
pub fn stack(frames: &[Frame], reference: &Globe, params: &Params, size: (usize, usize), scale: f32, clip: Option<f32>) -> Option<Image> {
    let first = frames.first()?;
    let nc = first.image.channels().len();
    let scale = scale.clamp(0.01, 1.0);
    let (w, h) = (((size.0 as f32 * scale).round() as usize).max(1), ((size.1 as f32 * scale).round() as usize).max(1));
    let proj = Projection::new(params.tilt_deg, params.oblateness);
    let total: f32 = frames.iter().map(|f| f.weight.max(0.0)).sum();
    if total <= 0.0 {
        return None;
    }
    let mut out = vec![0.0f32; w * h * nc];
    out.par_chunks_mut(w * nc).enumerate().for_each(|(j, row)| {
        let mut samples: Vec<([f32; 3], f32)> = Vec::with_capacity(frames.len());
        for i in 0..w {
            let (x, y) = ((i as f32 + 0.5) / scale, (j as f32 + 0.5) / scale);
            samples.clear();
            for f in frames.iter().filter(|f| f.weight > 0.0) {
                let ((px, py), vis) = source(f, reference, &proj, params, x, y);
                let chans = f.image.channels();
                let mut v = [0.0f32; 3];
                for (c, slot) in v.iter_mut().enumerate().take(nc) {
                    let plane = &chans[c.min(chans.len() - 1)];
                    *slot = catmull_rom(plane, f.image.width(), f.image.height(), px, py) * f.gain[c];
                }
                samples.push((v, f.weight * vis));
            }
            for c in 0..nc {
                row[i * nc + c] = combine(&samples, c, clip);
            }
        }
    });
    let channels = (0..nc).map(|c| out.iter().skip(c).step_by(nc).copied().collect()).collect();
    Some(Image::new(w, h, channels, first.image.source_bit_depth))
}

/// Weighted mean of channel `c` of the samples, after dropping those further than `clip`
/// standard deviations from it.
pub(crate) fn combine(samples: &[([f32; 3], f32)], c: usize, clip: Option<f32>) -> f32 {
    let mean = |keep: &dyn Fn(f32) -> bool| {
        let (mut sw, mut sv) = (0.0f64, 0.0f64);
        for (v, w) in samples.iter().filter(|(v, _)| keep(v[c])) {
            sw += *w as f64;
            sv += (*w * v[c]) as f64;
        }
        (sw > 0.0).then(|| (sv / sw) as f32)
    };
    let Some(m) = mean(&|_| true) else { return 0.0 };
    let Some(k) = clip.filter(|_| samples.len() >= 3) else { return m };
    // Weighted standard deviation; the median is the centre, which an outlier doesn't drag.
    let mut values: Vec<f32> = samples.iter().map(|(v, _)| v[c]).collect();
    values.sort_by(f32::total_cmp);
    let centre = values[values.len() / 2];
    let (mut sw, mut ss) = (0.0f64, 0.0f64);
    for (v, w) in samples {
        sw += *w as f64;
        ss += (*w as f64) * ((v[c] - m) as f64).powi(2);
    }
    let sigma = ((ss / sw.max(1e-12)).sqrt() as f32).max(1e-5);
    mean(&|v| (v - centre).abs() <= k * sigma).unwrap_or(m)
}

/// Catmull–Rom sample of a plane at (x, y), pixel centres at +0.5, edges clamped; held within
/// the four nearest pixels so sharp edges don't ring.
pub(crate) fn catmull_rom(plane: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let at = |x: isize, y: isize| plane[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize];
    let weights = |t: f32| {
        let (t2, t3) = (t * t, t * t * t);
        [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2]
    };
    let (wx, wy) = (weights(tx), weights(ty));
    let mut v = 0.0;
    for (j, wyj) in wy.iter().enumerate() {
        let mut r = 0.0;
        for (i, wxi) in wx.iter().enumerate() {
            r += wxi * at(x0 - 1 + i as isize, y0 - 1 + j as isize);
        }
        v += wyj * r;
    }
    let near = [at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1)];
    let (lo, hi) = near.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    v.clamp(lo, hi)
}

fn luminance(img: &Image) -> Vec<f32> {
    let chans = img.channels();
    let n = chans.len() as f32;
    (0..img.width() * img.height()).map(|i| chans.iter().map(|c| c[i]).sum::<f32>() / n).collect()
}

/// Correct each frame's centre (the reference frame's stays) by the shift left between its
/// derotated image and the reference frame's, both on the grid of `reference` (`size` pixels).
pub fn refine_centres(frames: &mut [Frame], reference_index: usize, reference: &Globe, params: &Params, size: (usize, usize)) {
    let (w, h) = size;
    if w < 64 || h < 64 || reference_index >= frames.len() {
        return;
    }
    let single = |f: &Frame| stack(&[Frame { weight: 1.0, ..*f }], reference, params, size, 1.0, None).map(|i| luminance(&i));
    let Some(target) = single(&frames[reference_index]) else { return };
    let shifts: Vec<Option<[f32; 2]>> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| if i == reference_index { None } else { single(f).map(|l| crate::align::measure_shift(&target, &l, w, h)) })
        .collect();
    for (f, s) in frames.iter_mut().zip(shifts) {
        if let Some([dx, dy]) = s {
            let k = f.globe.radius / reference.radius;
            f.globe.cx -= dx * k;
            f.globe.cy -= dy * k;
        }
    }
}

/// Mean of each channel inside 90 % of the globe (for matching brightness).
pub fn disk_means(img: &Image, globe: &Globe) -> [f32; 3] {
    let (w, h) = (img.width(), img.height());
    let mut sums = [0.0f64; 3];
    let mut n = 0usize;
    let r2 = (0.9 * globe.radius).powi(2);
    let y0 = (globe.cy - globe.radius).floor().max(0.0) as usize;
    let y1 = ((globe.cy + globe.radius).ceil().max(0.0) as usize).min(h);
    let x0 = (globe.cx - globe.radius).floor().max(0.0) as usize;
    let x1 = ((globe.cx + globe.radius).ceil().max(0.0) as usize).min(w);
    for y in y0..y1 {
        for x in x0..x1 {
            if (x as f32 + 0.5 - globe.cx).powi(2) + (y as f32 + 0.5 - globe.cy).powi(2) <= r2 {
                for (c, ch) in img.channels().iter().enumerate().take(3) {
                    sums[c] += ch[y * w + x] as f64;
                }
                n += 1;
            }
        }
    }
    let nc = img.channels().len().min(3);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        out[c] = if n == 0 { 0.0 } else { (sums[c.min(nc - 1)] / n as f64) as f32 };
    }
    out
}

/// Brightness factors bringing each image's globe to the reference one's level.
pub fn gains(means: &[[f32; 3]], reference: usize) -> Vec<[f32; 3]> {
    let Some(r) = means.get(reference) else { return vec![[1.0; 3]; means.len()] };
    means.iter().map(|m| std::array::from_fn(|c| if m[c] > 1e-6 && r[c] > 1e-6 { r[c] / m[c] } else { 1.0 })).collect()
}

/// How much fine detail the globe shows: mean squared Laplacian inside 80 % of it, relative to
/// its brightness. Comparable between frames of one session (same noise and scale).
pub fn sharpness(img: &Image, globe: &Globe) -> f32 {
    let (w, h) = (img.width(), img.height());
    let lum = luminance(img);
    let r2 = (0.8 * globe.radius).powi(2);
    let (mut sum, mut level, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if (x as f32 + 0.5 - globe.cx).powi(2) + (y as f32 + 0.5 - globe.cy).powi(2) > r2 {
                continue;
            }
            let k = y * w + x;
            let lap = 4.0 * lum[k] - lum[k - 1] - lum[k + 1] - lum[k - w] - lum[k + w];
            sum += (lap as f64).powi(2);
            level += lum[k] as f64;
            n += 1;
        }
    }
    if n == 0 || level <= 0.0 {
        return 0.0;
    }
    let mean = level / n as f64;
    (sum / n as f64 / (mean * mean)) as f32
}
