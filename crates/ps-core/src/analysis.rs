//! Automatic recognition of what an image shows (planet, Saturn, full lunar/solar disk or
//! extended surface), its noise level and, when a clean disk edge exists, the blur.

use crate::i18n::tl;
use crate::Image;
use crate::align::{self, AlignParams};
use crate::color::{ColorMode, working_planes};
use crate::equipment::Optics;
use crate::mask::Ellipse;
use crate::spectrum::{self, Spectral};
use crate::wavelet::estimate_noise;
use serde::{Deserialize, Serialize};

/// Disks wider than this (pixels) are taken to be the Moon or the Sun rather than a planet.
const FULL_DISK_PX: f32 = 900.0;
/// Segmentation runs on a copy downsampled to at most this side: faster and less noisy.
const WORK_SIDE: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageKind {
    Planet,
    Saturn,
    FullDisk,
    Surface,
}

impl ImageKind {
    pub const ALL: [ImageKind; 4] = [ImageKind::Planet, ImageKind::Saturn, ImageKind::FullDisk, ImageKind::Surface];

    pub fn label(self) -> &'static str {
        match self {
            ImageKind::Planet => tl("Pianeta"),
            ImageKind::Saturn => tl("Saturno"),
            ImageKind::FullDisk => tl("Disco intero"),
            ImageKind::Surface => tl("Superficie"),
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            ImageKind::Planet => tl("Giove, Marte, Venere…"),
            ImageKind::Saturn => tl("Con gli anelli"),
            ImageKind::FullDisk => tl("Luna o Sole interi"),
            ImageKind::Surface => tl("Dettaglio o mosaico"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confidence {
    High,
    Medium,
    Low,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Disk {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
}

#[derive(Clone, Debug)]
pub struct Analysis {
    pub kind: ImageKind,
    pub confidence: Confidence,
    pub disk: Option<Disk>,
    /// Noise σ of the luminance (0‥1 units).
    pub noise: f32,
    /// Median subject brightness divided by the noise σ.
    pub snr: f32,
    /// Gaussian PSF σ (pixels) measured on the limb, when the subject is a clean round disk.
    pub psf_sigma: Option<f32>,
    /// When the limb's blur has the wide wings of a Moffat profile rather than a Gaussian's
    /// (turbulent seeing): its FWHM (px) and β.
    pub psf_moffat: Option<(f32, f32)>,
    /// Blur σ of the R, G and B channels on the limb (colour images).
    pub channel_sigma: Option<[f32; 3]>,
    /// Shifts aligning R and B on G (colour images only).
    pub rgb_shift: Option<AlignParams>,
    /// Size of the subject in pixels: the disk (or ring system) span, or the frame for surfaces.
    pub subject_size: f32,
    /// What the image holds at each scale: where the detail sinks into the noise.
    pub spectral: Option<Spectral>,
    /// The measured spectrum behind it: (frequency in cycles/px, power) per band.
    pub power: Vec<(f32, f32)>,
    /// What the equipment means for the image (set from the user's setup or the capture log,
    /// not measured here).
    pub optics: Option<Optics>,
    /// Which planet the pixels suggest, when they say anything (see [`guess_planet`]): Saturn
    /// for `Saturn`, Mars from its colour, Jupiter from its flattening. A name in the file or
    /// in the capture log is more reliable and wins.
    pub planet: Option<crate::capture::Body>,
    /// Colour image (else mono).
    pub colour: bool,
    /// The belts and zones of a banded globe (Jupiter, Saturn), when they are clear.
    pub bands: Option<crate::bands::Bands>,
    /// Jupiter's Great Red Spot, diameter in pixels (colour images).
    pub red_spot: Option<f32>,
    /// Saturn's rings: outer size and how deep the Cassini division is.
    pub rings: Option<crate::bands::Rings>,
    /// The sharpening curve the subject's network suggests ([`crate::curve_net::suggestion`]):
    /// set by the caller once the subject is known (the file name or the capture log can name
    /// it better than the pixels).
    pub curve_net: Option<crate::curve_net::Suggestion>,
}

pub fn analyze(img: &Image) -> Analysis {
    let (w, h) = (img.width(), img.height());
    let lum = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let noise = estimate_noise(&lum, w, h);

    let (small, sw, sh, scale) = downsample(&lum, w, h, WORK_SIDE);
    let threshold = otsu(&small);
    let mask: Vec<bool> = small.iter().map(|&v| v > threshold).collect();
    let mut foreground: Vec<f32> = small.iter().zip(&mask).filter(|(_, m)| **m).map(|(v, _)| *v).collect();
    let signal = median(&mut foreground).unwrap_or(0.0);
    let snr = signal / noise.max(1e-6);
    let frame = w.min(h) as f32;
    // The channels' shift: measured by centroids and by correlation, and kept only if it makes
    // the edges of the three channels coincide better than no shift.
    let rgb_shift = align::estimate_verified(img);
    let (psf_sigma, psf_moffat, channel_sigma, spectral, power, optics) = (None, None, None, None, Vec::new(), None);
    let base = Analysis { kind: ImageKind::Planet, confidence: Confidence::Low, disk: None, noise, snr, psf_sigma, psf_moffat, channel_sigma, rgb_shift, subject_size: frame, spectral, power, optics, planet: None, colour: !img.is_mono(), bands: None, red_spot: None, rings: None, curve_net: None };

    // Subject running off the frame edges: a close-up or mosaic of an extended surface. Lit
    // ground is told from the sky by a low level (a fraction of the brightest), not by the
    // two-level split above: on a lunar close-up that split falls between maria and highlands,
    // or between lit ground and shadows, and the darker ground would pass for sky.
    let border = border_fraction(&mask, sw, sh);
    let coverage = foreground.len() as f32 / mask.len() as f32;
    let lit = lit_mask(&small);
    let lit_border = border_fraction(&lit, sw, sh);
    let lit_coverage = lit.iter().filter(|&&m| m).count() as f32 / lit.len().max(1) as f32;
    if border > 0.25 || coverage > 0.8 || lit_border > LIT_BORDER || lit_coverage > 0.85 {
        let confidence = if border > 0.5 || coverage > 0.9 || lit_border > 0.6 { Confidence::High } else { Confidence::Medium };
        let (spectral, power) = spectrum::measure(&lum, w, h, w as f32 / 2.0, h as f32 / 2.0);
        let surface = Analysis { kind: ImageKind::Surface, confidence, spectral, power, ..base };
        // The Moon's relief or the Sun's surface.
        return Analysis { planet: crate::sunmoon::classify(img, &surface), ..surface };
    }

    let Some(blob) = largest_component(&mask, sw, sh) else {
        let (spectral, power) = spectrum::measure(&lum, w, h, w as f32 / 2.0, h as f32 / 2.0);
        return Analysis { spectral, power, ..base };
    };
    let (bw, bh) = (blob.bbox_w(), blob.bbox_h());
    let aspect = blob.elongation;
    let (bcx, bcy) = blob.bbox_centre();
    // Saturn is point-symmetric; a crescent's mass sits off its bounding-box centre.
    let off_centre = ((blob.cx - bcx) / bw).abs().max(((blob.cy - bcy) / bh).abs());
    // The longest extent whatever the orientation (a ring system at 45° spans its bounding box's diagonal).
    let diameter = blob.length.max(bw.max(bh)) * scale;
    // A disk fills π/4 ≈ 0.785 of its bounding box.
    let is_round = |b: &Blob| b.elongation < 1.15 && (0.68..0.9).contains(&(b.area as f32 / (b.bbox_w() * b.bbox_h())));
    // The two-level split can fall inside the subject: on Mars its large dark areas and darker
    // limb drop out, and what is left looks elongated (1.2–1.35 on simulated stacks of round
    // Hubble globes). The lit disk at a low level is then the subject's outline.
    let lit_disk = (!is_round(&blob)).then(|| largest_component(&lit, sw, sh)).flatten().filter(|b| is_round(b) && b.area >= blob.area);
    let round = is_round(&blob) || lit_disk.is_some();

    let (kind, confidence) = if aspect > 1.6 && off_centre < 0.08 && diameter < FULL_DISK_PX {
        (ImageKind::Saturn, if aspect > 1.9 { Confidence::High } else { Confidence::Medium })
    } else if diameter >= FULL_DISK_PX {
        (ImageKind::FullDisk, if round { Confidence::High } else { Confidence::Medium })
    } else {
        (ImageKind::Planet, if round { Confidence::High } else { Confidence::Medium })
    };
    let disk = match &lit_disk {
        Some(b) => {
            let (cx, cy) = b.bbox_centre();
            Some(Disk { cx: cx * scale, cy: cy * scale, radius: b.length.max(b.bbox_w().max(b.bbox_h())) * scale / 2.0 })
        }
        None => round.then(|| Disk { cx: bcx * scale, cy: bcy * scale, radius: diameter / 2.0 }),
    };
    let limb = match (kind, disk) {
        // Saturn's globe, in the directions away from the rings.
        (ImageKind::Saturn, _) => crate::mask::saturn_globe(img).map(|g| (g, Some(g.angle))),
        (_, Some(d)) => Some((Ellipse { cx: d.cx, cy: d.cy, rx: d.radius, ry: d.radius, angle: 0.0 }, None)),
        _ => None,
    };
    let fit = limb.and_then(|(e, avoid)| limb_psf(&lum, w, h, e, avoid));
    let channel_sigma = match (limb, img.channels()) {
        (Some((e, avoid)), [r, g, b]) => {
            let sigma = |plane: &[f32]| limb_psf(plane, w, h, e, avoid).map(|f| f.sigma);
            sigma(r).zip(sigma(g)).zip(sigma(b)).map(|((r, g), b)| [r, g, b])
        }
        _ => None,
    };
    let (spectral, power) = spectrum::measure(&lum, w, h, bcx * scale, bcy * scale);
    let planet = match kind {
        ImageKind::Saturn => Some(crate::capture::Body::Saturn),
        ImageKind::Planet => guess_planet(img, &mask, sw, scale, aspect, round),
        // The Moon or the Sun, whole.
        ImageKind::FullDisk => crate::sunmoon::classify(img, &Analysis { kind, disk, ..base.clone() }),
        ImageKind::Surface => None,
    };
    // The globe: Saturn's with the equator along its rings; a disk's bands are searched.
    let globe = match (kind, disk) {
        (ImageKind::Saturn, _) => limb.map(|l| (l.0, false)),
        (ImageKind::Planet, Some(d)) => Some((Ellipse { cx: d.cx, cy: d.cy, rx: d.radius, ry: d.radius, angle: 0.0 }, true)),
        _ => None,
    };
    let bands = globe.and_then(|(g, search)| crate::bands::measure(&lum, w, h, g, search));
    let red_spot = match (planet, disk) {
        (Some(crate::capture::Body::Jupiter), Some(d)) => crate::bands::red_spot(img, d.cx, d.cy, d.radius),
        _ => None,
    };
    let rings = match (kind, globe) {
        (ImageKind::Saturn, Some((g, _))) => crate::bands::rings(&lum, w, h, g),
        _ => None,
    };
    Analysis { kind, confidence, disk, planet, psf_sigma: fit.map(|f| f.sigma), psf_moffat: fit.and_then(|f| f.moffat), channel_sigma, subject_size: diameter, spectral, power, bands, red_spot, rings, ..base }
}

/// Which planet a lone disk probably is, from the pixels alone, or `None` when they don't
/// say. Mars is the only clearly red planet (R/B well above 1.5 in any reasonable colour
/// balance); Jupiter is the only visibly flattened round one (1/15: an elongation of about
/// 1.07, where Mars, Venus and Mercury are round within a few thousandths). Venus and Mercury,
/// usually in phase and nearly colourless, are left to the file name or the user.
fn guess_planet(img: &Image, small_mask: &[bool], sw: usize, scale: f32, elongation: f32, round: bool) -> Option<crate::capture::Body> {
    use crate::capture::Body;
    if let [r, g, b] = img.channels() {
        let (w, h) = (img.width(), img.height());
        let mut sum = [0.0f64; 3];
        let mut n = 0usize;
        for (i, _) in small_mask.iter().enumerate().filter(|(_, m)| **m) {
            let (x, y) = (((i % sw) as f32 + 0.5) * scale, ((i / sw) as f32 + 0.5) * scale);
            let (x, y) = ((x as usize).min(w - 1), (y as usize).min(h - 1));
            let k = y * w + x;
            sum[0] += r[k] as f64;
            sum[1] += g[k] as f64;
            sum[2] += b[k] as f64;
            n += 1;
        }
        if n > 0 && sum[2] > 0.0 && sum[1] > 0.0 && sum[0] / sum[2] > 1.7 && sum[0] / sum[1] > 1.25 {
            return Some(Body::Mars);
        }
    }
    (round && (1.04..1.11).contains(&elongation)).then_some(Body::Jupiter)
}

/// Image scale (arcsec per pixel) from the subject's size in pixels and its apparent diameter
/// at `when`: Saturn's globe (its equator, without the rings), a planet's, the Moon's or the
/// Sun's disk. `None` when no whole disk was found.
pub fn image_scale(img: &Image, analysis: &Analysis, body: crate::capture::Body, when: &crate::capture::Moment) -> Option<f32> {
    use crate::capture::Body;
    let diameter_px = match body {
        Body::Saturn => crate::mask::saturn_globe(img).map(|g| 2.0 * g.rx.max(g.ry))?,
        _ => 2.0 * analysis.disk?.radius,
    };
    (diameter_px > 8.0).then(|| crate::ephemeris::apparent_diameter(body, when) as f32 / diameter_px)
}

/// Box-average to at most `max_side`; returns the image, its size and the scale factor.
pub(crate) fn downsample(plane: &[f32], w: usize, h: usize, max_side: usize) -> (Vec<f32>, usize, usize, f32) {
    let f = w.max(h).div_ceil(max_side).max(1);
    (crate::image::downscale_plane(plane, w, h, f), (w / f).max(1), (h / f).max(1), f as f32)
}

/// Otsu's threshold: the level that best separates the histogram into two classes.
pub(crate) fn otsu(values: &[f32]) -> f32 {
    const BINS: usize = 256;
    let mut hist = [0u32; BINS];
    for &v in values {
        hist[(v.clamp(0.0, 1.0) * (BINS - 1) as f32) as usize] += 1;
    }
    let total = values.len() as f64;
    let sum_all: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let (mut w0, mut sum0, mut best, mut best_t) = (0.0, 0.0, -1.0, 0);
    for (t, &c) in hist.iter().enumerate() {
        w0 += c as f64;
        sum0 += t as f64 * c as f64;
        let w1 = total - w0;
        if w0 == 0.0 || w1 == 0.0 {
            continue;
        }
        let (m0, m1) = (sum0 / w0, (sum_all - sum0) / w1);
        let between = w0 * w1 * (m0 - m1).powi(2);
        if between > best {
            best = between;
            best_t = t;
        }
    }
    (best_t as f32 + 0.5) / (BINS - 1) as f32
}

fn median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mid = values.len() / 2;
    Some(*values.select_nth_unstable_by(mid, f32::total_cmp).1)
}

/// Share of the frame's edge that must be lit ground for a close-up of a surface.
const LIT_BORDER: f32 = 0.3;

/// Lit ground or subject: above 12 % of the brightest (99th percentile), well above any sky.
fn lit_mask(small: &[f32]) -> Vec<bool> {
    let mut sample: Vec<f32> = small.to_vec();
    if sample.is_empty() {
        return Vec::new();
    }
    let k = (sample.len() * 99 / 100).min(sample.len() - 1);
    let top = *sample.select_nth_unstable_by(k, f32::total_cmp).1;
    small.iter().map(|&v| v > 0.12 * top).collect()
}

fn border_fraction(mask: &[bool], w: usize, h: usize) -> f32 {
    let edge = (0..w).flat_map(|x| [x, (h - 1) * w + x]).chain((0..h).flat_map(|y| [y * w, y * w + w - 1]));
    let (on, all) = edge.fold((0, 0), |(on, all), i| (on + mask[i] as usize, all + 1));
    on as f32 / all.max(1) as f32
}

struct Blob {
    pub area: usize,
    /// Ratio of the principal axes (√ of the covariance eigenvalues): 1 for a disk, whatever
    /// the orientation; a tilted ring system is elongated even when its bounding box is square.
    pub elongation: f32,
    /// Extent along the principal (longest) axis, pixels.
    pub length: f32,
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub cx: f32,
    pub cy: f32,
}

impl Blob {
    fn bbox_w(&self) -> f32 {
        (self.x1 - self.x0 + 1) as f32
    }
    fn bbox_h(&self) -> f32 {
        (self.y1 - self.y0 + 1) as f32
    }
    fn bbox_centre(&self) -> (f32, f32) {
        ((self.x0 + self.x1 + 1) as f32 / 2.0, (self.y0 + self.y1 + 1) as f32 / 2.0)
    }
}

/// Largest 4-connected foreground region (moons and stars are smaller and ignored).
fn largest_component(mask: &[bool], w: usize, h: usize) -> Option<Blob> {
    // The first of equally large regions, as a scan in reading order meets them.
    let members = crate::mask::components(mask, w, h).into_iter().rev().max_by_key(Vec::len).filter(|m| m.len() >= 16)?;
    let mut b = Blob { area: members.len(), elongation: 1.0, length: 0.0, x0: usize::MAX, y0: usize::MAX, x1: 0, y1: 0, cx: 0.0, cy: 0.0 };
    for &i in &members {
        let (x, y) = (i % w, i / w);
        (b.x0, b.y0, b.x1, b.y1) = (b.x0.min(x), b.y0.min(y), b.x1.max(x), b.y1.max(y));
        b.cx += x as f32 + 0.5;
        b.cy += y as f32 + 0.5;
    }
    b.cx /= b.area as f32;
    b.cy /= b.area as f32;
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for &i in &members {
        let (dx, dy) = ((i % w) as f64 + 0.5 - b.cx as f64, (i / w) as f64 + 0.5 - b.cy as f64);
        (sxx, syy, sxy) = (sxx + dx * dx, syy + dy * dy, sxy + dx * dy);
    }
    let (mean, diff) = ((sxx + syy) / 2.0, (((sxx - syy) / 2.0).powi(2) + sxy * sxy).sqrt());
    b.elongation = ((mean + diff) / (mean - diff).max(1e-9)).sqrt() as f32;
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (s, c) = (angle.sin(), angle.cos());
    let along = |i: usize| ((i % w) as f64 + 0.5 - b.cx as f64) * c + ((i / w) as f64 + 0.5 - b.cy as f64) * s;
    let (lo, hi) = members.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &i| (lo.min(along(i)), hi.max(along(i))));
    b.length = (hi - lo + 1.0) as f32;
    Some(b)
}

/// Estimate the blur from the limb: a sharp edge blurred by a Gaussian PSF has a Gaussian
/// derivative across it, whose FWHM is 2.355 σ. Measured along many rays and combined with
/// a median; rays across a terminator (soft, low contrast) are discarded. `limb` is the
/// disk's outline (a circle, or Saturn's oblate globe); rays within 40° of `avoid` (the
/// direction of Saturn's rings) are skipped.
///
/// The outer side of the averaged edge profile (towards the sky, which is flat, unlike the
/// limb-darkened disk) is also compared with a Moffat profile's: wide wings mean turbulent
/// seeing, which a Gaussian PSF under-corrects.
fn limb_psf(lum: &[f32], w: usize, h: usize, limb: Ellipse, avoid: Option<f32>) -> Option<LimbPsf> {
    let radius_at = |a: f32| {
        let phi = a - limb.angle;
        1.0 / ((phi.cos() / limb.rx).powi(2) + (phi.sin() / limb.ry).powi(2)).sqrt()
    };
    let disk = Disk { cx: limb.cx, cy: limb.cy, radius: limb.rx.max(limb.ry) };
    const RAYS: usize = 96;
    const STEP: f32 = 0.25;
    const SMOOTH: f32 = 0.5; // px, Gaussian smoothing applied to each profile
    let span = (disk.radius * 0.15).clamp(12.0, 60.0);
    let n = (2.0 * span / STEP) as usize;
    let sample = |x: f32, y: f32| -> Option<f32> {
        if x < 0.0 || y < 0.0 || x >= (w - 1) as f32 || y >= (h - 1) as f32 {
            return None;
        }
        let (x0, y0) = (x as usize, y as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |xx: usize, yy: usize| lum[yy * w + xx];
        Some(p(x0, y0) * (1.0 - fx) * (1.0 - fy) + p(x0 + 1, y0) * fx * (1.0 - fy) + p(x0, y0 + 1) * (1.0 - fx) * fy + p(x0 + 1, y0 + 1) * fx * fy)
    };
    let kernel: Vec<f32> = {
        let r = (3.0 * SMOOTH / STEP) as i32;
        let k: Vec<f32> = (-r..=r).map(|i| (-((i as f32 * STEP).powi(2)) / (2.0 * SMOOTH * SMOOTH)).exp()).collect();
        let s: f32 = k.iter().sum();
        k.into_iter().map(|v| v / s).collect()
    };

    let mut estimates: Vec<(f32, f32, Vec<f32>)> = Vec::new(); // (σ, contrast, outer profile)
    for k in 0..RAYS {
        let a = k as f32 * std::f32::consts::TAU / RAYS as f32;
        if let Some(axis) = avoid {
            // Angle to the ring axis, either way along it.
            let off = (a - axis).rem_euclid(std::f32::consts::PI);
            if off.min(std::f32::consts::PI - off) < 40f32.to_radians() {
                continue;
            }
        }
        let (dx, dy) = (a.cos(), a.sin());
        let limb_r = radius_at(a);
        let Some(profile) = (0..n)
            .map(|i| {
                let r = limb_r - span + i as f32 * STEP;
                sample(disk.cx + r * dx, disk.cy + r * dy)
            })
            .collect::<Option<Vec<f32>>>()
        else {
            continue;
        };
        let half = kernel.len() / 2;
        let smooth: Vec<f32> = (half..n - half).map(|i| kernel.iter().enumerate().map(|(j, c)| c * profile[i + j - half]).sum()).collect();
        // Outward intensity drop per sample.
        let drop: Vec<f32> = smooth.windows(2).map(|p| p[0] - p[1]).collect();
        let (peak, &peak_v) = drop.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
        if peak_v <= 0.0 {
            continue;
        }
        let half_max = peak_v / 2.0;
        let crossing = |range: &mut dyn Iterator<Item = usize>| -> Option<f32> {
            for i in range {
                let j = if i > peak { i - 1 } else { i + 1 };
                if drop[i] < half_max {
                    // Linear interpolation between i and its neighbour towards the peak.
                    let t = (half_max - drop[i]) / (drop[j] - drop[i]);
                    return Some(i as f32 + t * (j as f32 - i as f32));
                }
            }
            None
        };
        let (Some(left), Some(right)) = (crossing(&mut (0..peak).rev()), crossing(&mut (peak + 1..drop.len()))) else {
            continue;
        };
        let fwhm = (right - left) * STEP;
        let sigma = (fwhm / 2.3548).powi(2) - SMOOTH * SMOOTH;
        let contrast = smooth[..peak].iter().copied().fold(f32::MIN, f32::max) - smooth[peak..].iter().copied().fold(f32::MAX, f32::min);
        if sigma > 0.0 {
            let outer: Vec<f32> = drop[peak..].iter().map(|v| v / peak_v).collect();
            estimates.push((sigma.sqrt(), contrast, outer));
        }
    }
    if estimates.len() < RAYS / 5 {
        return None;
    }
    let mut contrasts: Vec<f32> = estimates.iter().map(|e| e.1).collect();
    let min_contrast = 0.6 * median(&mut contrasts)?;
    let kept: Vec<&(f32, f32, Vec<f32>)> = estimates.iter().filter(|e| e.1 >= min_contrast).collect();
    let mut sigmas: Vec<f32> = kept.iter().map(|e| e.0).collect();
    let sigma = median(&mut sigmas)?;
    // Average outer profile, over the length every ray has.
    let len = kept.iter().map(|e| e.2.len()).min()?;
    let outer: Vec<f32> = (0..len).map(|j| kept.iter().map(|e| e.2[j]).sum::<f32>() / kept.len() as f32).collect();
    Some(LimbPsf { sigma, moffat: fit_moffat(&outer, STEP) })
}

/// What the limb tells about the PSF.
#[derive(Clone, Copy)]
struct LimbPsf {
    sigma: f32,
    /// (FWHM, β) when a Moffat fits the wings clearly better than a Gaussian.
    moffat: Option<(f32, f32)>,
}

/// Compare the outer half of a line-spread profile (peak = 1 at index 0, samples `step` px
/// apart) with a Gaussian and with Moffats of β 1.5‥6, all matched at half maximum. The line
/// spread of a Moffat (1 + r²/α²)^−β is (1 + x²/α²)^−(β−½).
fn fit_moffat(outer: &[f32], step: f32) -> Option<(f32, f32)> {
    // Half width at half maximum.
    let j = outer.iter().position(|&v| v < 0.5)?;
    let t = (outer[j - 1] - 0.5) / (outer[j - 1] - outer[j]);
    let hw = (j as f32 - 1.0 + t) * step;
    let reach = ((4.0 * hw / step) as usize).min(outer.len());
    if hw <= 0.0 || reach < j + 4 {
        return None;
    }
    let error = |model: &dyn Fn(f32) -> f32| (0..reach).map(|i| (outer[i] - model(i as f32 * step)).powi(2)).sum::<f32>();
    let gaussian = error(&|x| (-(x * x) * std::f32::consts::LN_2 / (hw * hw)).exp());
    let (beta, moffat) = (6..=24)
        .map(|k| k as f32 * 0.25)
        .map(|beta| {
            let alpha = hw / (2f32.powf(1.0 / (beta - 0.5)) - 1.0).sqrt();
            (beta, error(&|x| (1.0 + x * x / (alpha * alpha)).powf(-(beta - 0.5))))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    // Only when the wings are clearly not Gaussian.
    if moffat > 0.7 * gaussian {
        return None;
    }
    let alpha = hw / (2f32.powf(1.0 / (beta - 0.5)) - 1.0).sqrt();
    Some((2.0 * alpha * (2f32.powf(1.0 / beta) - 1.0).sqrt(), beta))
}
