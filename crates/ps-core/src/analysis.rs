//! Automatic recognition of what an image shows (planet, Saturn, full lunar/solar disk or
//! extended surface), its noise level and, when a clean disk edge exists, the blur.

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
            ImageKind::Planet => "Pianeta",
            ImageKind::Saturn => "Saturno",
            ImageKind::FullDisk => "Disco intero",
            ImageKind::Surface => "Superficie",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            ImageKind::Planet => "Giove, Marte, Venere…",
            ImageKind::Saturn => "Con gli anelli",
            ImageKind::FullDisk => "Luna o Sole interi",
            ImageKind::Surface => "Dettaglio o mosaico",
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
    // A subject inside the frame is aligned by centroids; a surface by correlation.
    let rgb_shift = align::estimate_by_centroid(img);
    let (psf_sigma, psf_moffat, channel_sigma, spectral, power, optics) = (None, None, None, None, Vec::new(), None);
    let base = Analysis { kind: ImageKind::Planet, confidence: Confidence::Low, disk: None, noise, snr, psf_sigma, psf_moffat, channel_sigma, rgb_shift, subject_size: frame, spectral, power, optics };

    // Subject running off the frame edges: a close-up or mosaic of an extended surface.
    let border = border_fraction(&mask, sw, sh);
    let coverage = foreground.len() as f32 / mask.len() as f32;
    if border > 0.25 || coverage > 0.8 {
        let confidence = if border > 0.5 || coverage > 0.9 { Confidence::High } else { Confidence::Medium };
        let (spectral, power) = spectrum::measure(&lum, w, h, w as f32 / 2.0, h as f32 / 2.0);
        return Analysis { kind: ImageKind::Surface, confidence, rgb_shift: align::estimate(img), spectral, power, ..base };
    }

    let Some(blob) = largest_component(&mask, sw, sh) else {
        let (spectral, power) = spectrum::measure(&lum, w, h, w as f32 / 2.0, h as f32 / 2.0);
        return Analysis { spectral, power, ..base };
    };
    let (bw, bh) = (blob.bbox_w(), blob.bbox_h());
    let aspect = blob.elongation;
    let fill = blob.area as f32 / (bw * bh);
    let (bcx, bcy) = blob.bbox_centre();
    // Saturn is point-symmetric; a crescent's mass sits off its bounding-box centre.
    let off_centre = ((blob.cx - bcx) / bw).abs().max(((blob.cy - bcy) / bh).abs());
    // The longest extent whatever the orientation (a ring system at 45° spans its bounding box's diagonal).
    let diameter = blob.length.max(bw.max(bh)) * scale;
    // A disk fills π/4 ≈ 0.785 of its bounding box.
    let round = aspect < 1.15 && (0.68..0.9).contains(&fill);

    let (kind, confidence) = if aspect > 1.6 && off_centre < 0.08 && diameter < FULL_DISK_PX {
        (ImageKind::Saturn, if aspect > 1.9 { Confidence::High } else { Confidence::Medium })
    } else if diameter >= FULL_DISK_PX {
        (ImageKind::FullDisk, if round { Confidence::High } else { Confidence::Medium })
    } else {
        (ImageKind::Planet, if round { Confidence::High } else { Confidence::Medium })
    };
    let disk = round.then(|| Disk { cx: bcx * scale, cy: bcy * scale, radius: diameter / 2.0 });
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
    Analysis { kind, confidence, disk, psf_sigma: fit.map(|f| f.sigma), psf_moffat: fit.and_then(|f| f.moffat), channel_sigma, subject_size: diameter, spectral, power, ..base }
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
    let (dw, dh) = (w / f, h / f);
    let mut out = vec![0.0f32; dw * dh];
    for y in 0..dh {
        for x in 0..dw {
            let mut acc = 0.0;
            for sy in 0..f {
                let row = &plane[(y * f + sy) * w + x * f..(y * f + sy) * w + x * f + f];
                acc += row.iter().sum::<f32>();
            }
            out[y * dw + x] = acc / (f * f) as f32;
        }
    }
    (out, dw, dh, f as f32)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::FftConvolver;
    use crate::psf::Psf;

    fn blur(plane: Vec<f32>, w: usize, h: usize, sigma: f32) -> Vec<f32> {
        FftConvolver::new(w, h, &Psf::Gaussian { sigma }.kernel()).convolve(&plane)
    }

    fn noisy(mut v: Vec<f32>, sigma: f32) -> Vec<f32> {
        let mut s = 1234567u32;
        for x in &mut v {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *x += sigma * (s as f32 / u32::MAX as f32 - 0.5) * 3.46;
        }
        v
    }

    /// Ellipse with optional ring, on black.
    fn scene(w: usize, h: usize, radius: f32, ring: bool) -> Vec<f32> {
        tilted_scene(w, h, radius, ring, 0.0)
    }

    /// The same, with the ring plane rotated by `degrees`.
    fn tilted_scene(w: usize, h: usize, radius: f32, ring: bool, degrees: f32) -> Vec<f32> {
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let (s, c) = degrees.to_radians().sin_cos();
        (0..w * h)
            .map(|i| {
                let (u, v) = ((i % w) as f32 + 0.5 - cx, (i / w) as f32 + 0.5 - cy);
                let (x, y) = (u * c + v * s, -u * s + v * c);
                let r = (x * x + y * y).sqrt();
                let disk = if r < radius { 0.6 + 0.2 * (y * 0.05).sin() } else { 0.0 };
                let e = ((x / (radius * 2.2)).powi(2) + (y / (radius * 0.6)).powi(2)).sqrt();
                let ring_v = if ring && (0.75..1.0).contains(&e) { 0.5 } else { 0.0 };
                f32::max(disk, ring_v)
            })
            .collect()
    }

    fn analyze_plane(plane: Vec<f32>, w: usize, h: usize) -> Analysis {
        analyze(&Image::new(w, h, vec![plane], 16))
    }

    #[test]
    fn recognises_a_planet_and_measures_its_blur() {
        let (w, h) = (600, 500);
        for sigma in [1.2, 2.0, 3.0] {
            let a = analyze_plane(noisy(blur(scene(w, h, 150.0, false), w, h, sigma), 0.005), w, h);
            assert_eq!(a.kind, ImageKind::Planet);
            assert_eq!(a.confidence, Confidence::High);
            let est = a.psf_sigma.expect("limb estimate");
            assert!((est - sigma).abs() / sigma < 0.2, "σ {sigma}: estimated {est}");
        }
    }

    #[test]
    fn recognises_saturn_and_measures_its_blur() {
        // Saturn's proportions: globe flattened to 0.9, rings out to 2.27 globe radii.
        let (w, h, r) = (800, 500, 110.0f32);
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 + 0.5 - 400.0, (i / w) as f32 + 0.5 - 250.0);
                let globe = (x / r).powi(2) + (y / (0.9 * r)).powi(2) < 1.0;
                let e = ((x / (2.27 * r)).powi(2) + (y / (0.42 * r)).powi(2)).sqrt();
                if globe && !(y > 0.0 && (0.52..0.98).contains(&e)) {
                    0.65
                } else if (0.52..0.98).contains(&e) {
                    0.5
                } else {
                    0.0
                }
            })
            .collect();
        for sigma in [1.5, 3.0] {
            let a = analyze_plane(blur(plane.clone(), w, h, sigma), w, h);
            assert_eq!(a.kind, ImageKind::Saturn);
            let est = a.psf_sigma.expect("measured on the globe");
            assert!((est - sigma).abs() / sigma < 0.25, "σ {sigma}: estimated {est}");
        }
    }

    /// Regression: with the rings at 45° the bounding box is nearly square, and Saturn was
    /// taken for a plain planet.
    #[test]
    fn recognises_a_tilted_saturn() {
        let (w, h) = (700, 700);
        for degrees in [30.0, 45.0, 60.0, 90.0] {
            let a = analyze_plane(blur(tilted_scene(w, h, 90.0, true, degrees), w, h, 1.5), w, h);
            assert_eq!(a.kind, ImageKind::Saturn, "rings at {degrees}°");
            assert!(a.subject_size > 2.0 * 2.2 * 90.0 * 0.9, "span {} at {degrees}°", a.subject_size);
        }
    }

    /// Wide-winged (Moffat) blur is recognised as such; Gaussian blur is not.
    #[test]
    fn tells_moffat_wings_from_gaussian_blur() {
        let (w, h) = (600, 500);
        let disk = scene(w, h, 150.0, false);
        let moffat = FftConvolver::new(w, h, &Psf::Moffat { fwhm: 5.0, beta: 2.0 }.kernel()).convolve(&disk);
        let a = analyze_plane(moffat, w, h);
        let (fwhm, beta) = a.psf_moffat.expect("Moffat wings");
        assert!((beta - 2.0).abs() < 0.8 && (fwhm - 5.0).abs() < 1.5, "fwhm {fwhm}, β {beta}");
        assert!(analyze_plane(blur(disk, w, h, 2.0), w, h).psf_moffat.is_none(), "Gaussian blur");
    }

    #[test]
    fn measures_each_channels_blur() {
        let (w, h) = (600, 500);
        let disk = scene(w, h, 150.0, false);
        let img = Image::new(w, h, vec![blur(disk.clone(), w, h, 1.6), blur(disk.clone(), w, h, 2.0), blur(disk, w, h, 2.6)], 16);
        let s = analyze(&img).channel_sigma.expect("per channel");
        assert!(s[0] < s[1] && s[1] < s[2], "{s:?}");
    }

    #[test]
    fn recognises_a_full_disk() {
        let (w, h) = (1400, 1300);
        let a = analyze_plane(blur(scene(w, h, 600.0, false), w, h, 1.5), w, h);
        assert_eq!(a.kind, ImageKind::FullDisk);
    }

    #[test]
    fn recognises_a_surface() {
        let (w, h) = (700, 500);
        let plane = (0..w * h).map(|i| 0.4 + 0.2 * ((i % w) as f32 * 0.07).sin() * ((i / w) as f32 * 0.05).cos()).collect();
        assert_eq!(analyze_plane(plane, w, h).kind, ImageKind::Surface);
    }

    #[test]
    fn crescent_is_not_saturn() {
        let (w, h) = (600, 600);
        let r = 200.0;
        let plane = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - 300.0, (i / w) as f32 - 300.0);
                let lit = x * x + y * y < r * r && (x - 90.0).powi(2) + y * y > r * r;
                if lit { 0.7 } else { 0.0 }
            })
            .collect();
        assert_ne!(analyze_plane(plane, w, h).kind, ImageKind::Saturn);
    }
}
