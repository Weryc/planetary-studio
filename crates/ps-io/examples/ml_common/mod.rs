//! Shared by the learned-sharpening study tools (`ml_ref`, `ml_train`, `ml_eval_ref`, `ml_diag`).
#![allow(dead_code)]

use ps_core::Image;
use ps_core::analysis::{Analysis, ImageKind};
use ps_core::color::{ColorMode, working_planes};
use ps_core::fft::{FftConvolver, next_fast_size};
use ps_core::mask::Ellipse;
use ps_core::learned::STEPS;
use std::path::Path;

pub mod seeing;

/// Load an image; also reads the uncompressed 32-bit float TIFFs written by ImageJ (any
/// range), which the app's loader refuses. Values are scaled to 0‥1.
pub fn load_any(path: &Path) -> Option<Image> {
    let img = ps_io::load(path).ok().or_else(|| float_tiff(path))?;
    let max = img.channels().iter().flatten().fold(0.0f32, |m, &v| m.max(v));
    if max > 1.0 {
        let planes = img.channels().iter().map(|p| p.iter().map(|v| v / max).collect()).collect();
        return Some(Image::new(img.width(), img.height(), planes, 16));
    }
    Some(img)
}

fn float_tiff(path: &Path) -> Option<Image> {
    let b = std::fs::read(path).ok()?;
    let big = match &b[..4] {
        b"MM\0*" => true,
        b"II*\0" => false,
        _ => return None,
    };
    let u16_at = |o: usize| if big { u16::from_be_bytes([b[o], b[o + 1]]) } else { u16::from_le_bytes([b[o], b[o + 1]]) };
    let u32_at = |o: usize| if big { u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) } else { u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) };
    let ifd = u32_at(4) as usize;
    let n = u16_at(ifd) as usize;
    let (mut w, mut h, mut bps, mut offset, mut format, mut spp, mut compression) = (0usize, 0usize, 0u32, 0usize, 1u32, 1u32, 1u32);
    for e in 0..n {
        let o = ifd + 2 + 12 * e;
        let (tag, typ, count) = (u16_at(o), u16_at(o + 2), u32_at(o + 4));
        let value = if typ == 3 && count == 1 { u16_at(o + 8) as u32 } else { u32_at(o + 8) };
        match tag {
            256 => w = value as usize,
            257 => h = value as usize,
            258 => bps = value,
            259 => compression = value,
            273 => offset = if count == 1 { value as usize } else if typ == 3 { u16_at(value as usize) as usize } else { u32_at(value as usize) as usize },
            277 => spp = value,
            339 => format = value,
            _ => {}
        }
    }
    if bps != 32 || format != 3 || spp != 1 || compression != 1 || w == 0 || h == 0 || offset + 4 * w * h > b.len() {
        return None;
    }
    let plane: Vec<f32> = (0..w * h).map(|i| f32::from_bits(u32_at(offset + 4 * i))).map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 }).collect();
    Some(Image::new(w, h, vec![plane], 32))
}

/// Images from the same night, telescope or author are not independent: they are kept
/// together when validating (never one in training and its sibling in the test).
pub fn group_of(stem: &str) -> String {
    let parts: Vec<&str> = stem.split('_').collect();
    if (stem.starts_with("CN_") || stem.starts_with("LV_")) && parts.len() >= 3 {
        // LV_saturn_20080312_…: by date; CN_Saturno_Dob16_…: by setup.
        return parts[..3].join("_");
    }
    if stem.starts_with("IIS_") {
        return "IIS".into();
    }
    let words: Vec<&str> = stem.split(' ').collect();
    if words.len() >= 2 && words[1].len() == 8 && words[1].chars().filter(|&c| c == '-').count() == 2 {
        return words[1].to_string();
    }
    if stem.len() >= 10 && stem.as_bytes()[4] == b'-' && stem.as_bytes()[7] == b'-' {
        return stem[..10].to_string();
    }
    stem.to_string()
}

pub fn lum(img: &Image) -> Vec<f32> {
    working_planes(img, ColorMode::Luminance).swap_remove(0)
}

pub fn fast_below(mut n: usize) -> usize {
    while next_fast_size(n) != n {
        n -= 1;
    }
    n
}

/// Subject outline: the disk, Saturn's globe, or nothing (surface).
pub fn outline(img: &Image, a: &Analysis) -> Option<Ellipse> {
    match (a.kind, a.disk) {
        (ImageKind::Saturn, _) => ps_core::mask::saturn_globe(img),
        (_, Some(d)) => Some(Ellipse { cx: d.cx, cy: d.cy, rx: d.radius, ry: d.radius, angle: 0.0 }),
        _ => None,
    }
}

/// Square fully inside the subject (or centred, for a surface): (x0, y0, side).
pub fn inner_square(w: usize, h: usize, e: Option<Ellipse>, min_side: usize) -> Option<(usize, usize, usize)> {
    let (cx, cy, half) = match e {
        Some(e) => (e.cx, e.cy, 0.62 * e.rx.min(e.ry)),
        None => (w as f32 / 2.0, h as f32 / 2.0, (w.min(h).min(1024) / 2) as f32),
    };
    let side = fast_below(((2.0 * half) as usize).min(w).min(h));
    if side < min_side {
        return None;
    }
    let x0 = ((cx - side as f32 / 2.0).max(0.0) as usize).min(w - side);
    let y0 = ((cy - side as f32 / 2.0).max(0.0) as usize).min(h - side);
    Some((x0, y0, side))
}

pub fn crop(plane: &[f32], w: usize, (x0, y0, side): (usize, usize, usize)) -> Vec<f32> {
    (0..side).flat_map(|y| plane[(y0 + y) * w + x0..(y0 + y) * w + x0 + side].to_vec()).collect()
}

/// Mean removed, Hann window: (fx, fy, re, im) per coefficient of the half plane.
pub fn coefs(plane: &[f32], side: usize) -> Vec<[f32; 4]> {
    let mean = plane.iter().map(|&v| v as f64).sum::<f64>() as f32 / plane.len() as f32;
    let hann: Vec<f32> = (0..side).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * (i as f32 + 0.5) / side as f32).cos()).collect();
    let win: Vec<f32> = (0..side * side).map(|i| (plane[i] - mean) * hann[i % side] * hann[i / side]).collect();
    FftConvolver::plan(side, side)
        .forward(&win)
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            let (kx, ky) = (idx / side, idx % side);
            let fy = (if ky <= side / 2 { ky as f32 } else { ky as f32 - side as f32 }) / side as f32;
            [kx as f32 / side as f32, fy, c.re, c.im]
        })
        .collect()
}

/// Radial power in log bins from 0.01 to 0.5: (f, power).
pub fn radial(c: &[[f32; 4]]) -> Vec<(f32, f32)> {
    const BINS: usize = 70;
    let (lo, hi) = (0.01f32.ln(), 0.5f32.ln());
    let (mut sum, mut n) = (vec![0.0f64; BINS], vec![0u32; BINS]);
    for k in c {
        let f = k[0].hypot(k[1]);
        if !(0.01..=0.5).contains(&f) {
            continue;
        }
        let b = (((f.ln() - lo) / (hi - lo)) * BINS as f32).clamp(0.0, BINS as f32 - 1.0) as usize;
        sum[b] += (k[2] * k[2] + k[3] * k[3]) as f64;
        n[b] += 1;
    }
    (0..BINS).filter(|&b| n[b] > 0).map(|b| ((lo + (b as f32 + 0.5) / BINS as f32 * (hi - lo)).exp(), (sum[b] / n[b] as f64) as f32)).collect()
}

pub fn spectrum_of(plane: &[f32], w: usize, sq: (usize, usize, usize)) -> Vec<(f32, f32)> {
    radial(&coefs(&crop(plane, w, sq), sq.2))
}

pub fn smooth(v: &[(f32, f32)]) -> Vec<(f32, f32)> {
    (0..v.len()).map(|i| (v[i].0, (v[i.saturating_sub(1)].1 + v[i].1 + v[(i + 1).min(v.len() - 1)].1) / 3.0)).collect()
}

/// Value of a (frequency, value) curve at the nearest frequency (log distance).
pub fn at(curve: &[(f32, f32)], f: f32) -> f32 {
    let f = f.clamp(0.01, 0.5);
    curve.iter().min_by(|a, b| (a.0 / f).ln().abs().total_cmp(&(b.0 / f).ln().abs())).map_or(f32::NAN, |q| q.1)
}

/// Amplitude gain by frequency from `input` to `output` inside the square, smoothed.
pub fn gain(input: &[f32], output: &[f32], w: usize, sq: (usize, usize, usize)) -> Vec<(f32, f32)> {
    let (a, b) = (spectrum_of(input, w, sq), spectrum_of(output, w, sq));
    smooth(&a.iter().zip(&b).map(|(p, q)| (p.0, (q.1 / p.1.max(1e-30)).sqrt())).collect::<Vec<_>>())
}

/// A curve sampled at the learned grid (multiples of `cutoff`), as ln(gain).
pub fn on_grid(curve: &[(f32, f32)], cutoff: f32) -> [f32; STEPS] {
    ps_core::learned::GRID.map(|r| at(curve, r * cutoff).clamp(0.05, 50.0).ln())
}

/// Exact Gaussian blur of every channel.
pub fn blur(img: &Image, sigma: f32) -> Image {
    if sigma < 0.05 {
        return img.clone();
    }
    let r = (4.0 * sigma).ceil() as isize;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    let (w, h) = (img.width(), img.height());
    let planes = img
        .channels()
        .iter()
        .map(|p| {
            let (mut dst, mut tmp) = (vec![0.0; w * h], vec![0.0; w * h]);
            ps_core::conv::convolve_rows_then_cols(p, &mut dst, &mut tmp, w, h, &k, &k);
            dst
        })
        .collect();
    Image::new(w, h, planes, 16)
}

/// Mean of `d`×`d` blocks (the truth for the synthetic examples: sharper than the stack).
pub fn shrink(img: &Image, d: usize) -> Image {
    let (w, h) = (img.width() / d, img.height() / d);
    let planes = img
        .channels()
        .iter()
        .map(|p| {
            (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    let mut s = 0.0;
                    for yy in 0..d {
                        for xx in 0..d {
                            s += p[(y * d + yy) * img.width() + x * d + xx];
                        }
                    }
                    s / (d * d) as f32
                })
                .collect()
        })
        .collect();
    Image::new(w, h, planes, 16)
}

/// Gaussian noise of standard deviation `sigma`, smoothed by `corr` px (like a stack's), one
/// independent field per channel.
pub fn noise(w: usize, h: usize, channels: usize, sigma: f32, corr: f32, seed: u64) -> Image {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let planes: Vec<Vec<f32>> = (0..channels)
        .map(|_| (0..w * h).map(|_| ((-2.0 * next().max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * next()).cos()) as f32).collect())
        .collect();
    let field = blur(&Image::new(w, h, planes, 16), corr);
    let planes = field
        .channels()
        .iter()
        .map(|p| {
            let sd = (p.iter().map(|v| (v * v) as f64).sum::<f64>() / p.len() as f64).sqrt() as f32;
            p.iter().map(|v| v * sigma / sd.max(1e-12)).collect()
        })
        .collect();
    Image::new(w, h, planes, 16)
}

pub fn add(a: &Image, b: &Image) -> Image {
    let planes = a.channels().iter().zip(b.channels()).map(|(p, q)| p.iter().zip(q).map(|(x, y)| x + y).collect()).collect();
    Image::new(a.width(), a.height(), planes, 16)
}

/// Where the subject is: its centre, for cropping big frames.
pub fn centre(img: &Image, a: &Analysis) -> (f32, f32) {
    outline(img, a).map_or((img.width() as f32 / 2.0, img.height() as f32 / 2.0), |e| (e.cx, e.cy))
}

/// At most `side`² around (`cx`, `cy`).
pub fn crop_around(img: &Image, (cx, cy): (f32, f32), side: usize) -> Image {
    let (w, h) = (side.min(img.width()), side.min(img.height()));
    let x0 = ((cx - w as f32 / 2.0).max(0.0) as usize).min(img.width() - w);
    let y0 = ((cy - h as f32 / 2.0).max(0.0) as usize).min(img.height() - h);
    img.crop(x0, y0, w, h)
}

/// Read the dataset written by `ml_ref`.
pub struct Row {
    pub name: String,
    pub group: String,
    pub variant: String,
    pub kind: String,
    pub body: String,
    pub cutoff: f32,
    pub features: Vec<f32>,
    pub bxt: Option<[f32; STEPS]>,
    pub wiener: Option<[f32; STEPS]>,
}

pub fn read_rows(path: &Path) -> Vec<Row> {
    let text = std::fs::read_to_string(path).expect("dataset");
    let nf = ps_core::learned::FEATURES;
    text.lines()
        .skip(1)
        .map(|line| {
            let c: Vec<&str> = line.split('\t').collect();
            let num = |s: &str| s.parse::<f32>().unwrap_or(f32::NAN);
            let curve = |from: usize| -> Option<[f32; STEPS]> {
                let v: [f32; STEPS] = std::array::from_fn(|i| num(c[from + i]));
                v.iter().all(|x| x.is_finite()).then_some(v)
            };
            let f0 = 6;
            Row {
                name: c[0].into(),
                group: c[1].into(),
                variant: c[2].into(),
                kind: c[3].into(),
                body: c[4].into(),
                cutoff: num(c[5]),
                features: c[f0..f0 + nf].iter().map(|s| num(s)).collect(),
                bxt: curve(f0 + nf),
                wiener: curve(f0 + nf + STEPS),
            }
        })
        .collect()
}

/// One image ready for the study: cropped to at most 1024² around its subject and analysed.
pub struct Source {
    pub stem: String,
    pub img: Image,
    pub analysis: Analysis,
    pub body: Option<ps_core::capture::Body>,
    /// Square inside the subject where responses are measured.
    pub square: (usize, usize, usize),
}

pub fn source(path: &Path) -> Result<Source, String> {
    let stem = path.file_stem().unwrap().to_string_lossy().to_string();
    let full = load_any(path).ok_or("illeggibile")?;
    let img = if full.width() * full.height() > 1_200_000 {
        let a = ps_core::analysis::analyze(&full);
        crop_around(&full, centre(&full, &a), 1024)
    } else {
        full
    };
    let analysis = ps_core::analysis::analyze(&img);
    let body = ps_core::capture::from_name(&stem).body.or(analysis.planet);
    let square = inner_square(img.width(), img.height(), outline(&img, &analysis), 64).ok_or("soggetto troppo piccolo")?;
    Ok(Source { stem, img, analysis, body, square })
}

/// The best linear gain, by frequency, from `degraded` back to the signal of `truth`
/// (Wiener): B·S / (B²·(S + N_truth) + N_added), each measured on the square.
pub fn wiener(truth: &Image, blurred_by: f32, degraded: &Image, sq: (usize, usize, usize)) -> Vec<(f32, f32)> {
    wiener_with(truth, &blur(truth, blurred_by), degraded, sq)
}

/// [`wiener`] with the blurred truth given (any blur, not only a Gaussian).
pub fn wiener_with(truth: &Image, blurred: &Image, degraded: &Image, sq: (usize, usize, usize)) -> Vec<(f32, f32)> {
    let w = truth.width();
    let (lt, lb) = (lum(truth), lum(blurred));
    let p_truth = spectrum_of(&lt, w, sq);
    let p_blur = spectrum_of(&lb, w, sq);
    let added: Vec<f32> = lum(degraded).iter().zip(&lb).map(|(d, b)| d - b).collect();
    let p_noise = spectrum_of(&added, w, sq);
    // The truth's own noise: the fitted floor, else its highest frequencies.
    let floor = {
        let mut hi: Vec<f32> = p_truth.iter().filter(|q| q.0 > 0.42).map(|q| q.1).collect();
        hi.sort_by(f32::total_cmp);
        hi.get(hi.len() / 2).copied().unwrap_or(0.0)
    };
    let fit = ps_core::spectrum::fit(&p_truth);
    let n_truth = |f: f32| fit.map_or(floor, |s| s.noise(f).min(floor * 4.0));
    let curve: Vec<(f32, f32)> = p_truth
        .iter()
        .zip(&p_blur)
        .zip(&p_noise)
        .map(|((t, b), n)| {
            let f = t.0;
            let b2 = (b.1 / t.1.max(1e-30)).clamp(0.0, 1.0);
            let s = (t.1 - n_truth(f)).max(0.0);
            (f, b2.sqrt() * s / (b2 * (s + n_truth(f)) + n.1).max(1e-30))
        })
        .collect();
    smooth(&curve)
}


/// Subject brightness over the noise σ, as in real stacks (median ≈ 400).
pub const SNRS: [f32; 3] = [150.0, 450.0, 1300.0];

/// The stacked lucky-imaging PSFs of every night ([`seeing::NIGHTS`]), on the fine grid: for
/// colour (R, G, B) and for mono. Made once; the same for every truth.
pub struct Nights {
    colour: Vec<Vec<Vec<f32>>>,
    mono: Vec<Vec<f32>>,
}

impl Nights {
    pub fn new() -> Self {
        let started = std::time::Instant::now();
        let colour = seeing::NIGHTS.iter().enumerate().map(|(i, n)| seeing::stacked(n, &seeing::LAMBDA_RGB, &[0.3 * n.dispersion, 0.55 * n.dispersion, n.dispersion], i as u64 + 1)).collect();
        let mono = seeing::NIGHTS.iter().enumerate().map(|(i, n)| seeing::stacked(n, &[seeing::LAMBDA_MONO], &[0.25 * n.dispersion], i as u64 + 11).swap_remove(0)).collect();
        eprintln!("seeing simulato in {:.0} s", started.elapsed().as_secs_f32());
        Nights { colour, mono }
    }

    /// Kernels for night `i` at `k` λ/D per pixel: three for colour, one for mono.
    pub fn kernels(&self, i: usize, mono: bool, k: f32) -> Vec<ps_core::psf::Kernel2D> {
        if mono {
            vec![seeing::to_pixels(&self.mono[i], seeing::LAMBDA_MONO, k)]
        } else {
            self.colour[i].iter().zip(seeing::LAMBDA_RGB).map(|(p, l)| seeing::to_pixels(p, l, k)).collect()
        }
    }
}

/// Median of the lit part of the luminance (the subject's brightness).
pub fn lit_median(img: &Image) -> f32 {
    let mut v: Vec<f32> = lum(img).into_iter().step_by(3).collect();
    v.sort_by(f32::total_cmp);
    let top = v[v.len() * 99 / 100];
    let lit: Vec<f32> = v.into_iter().filter(|&x| x > 0.15 * top).collect();
    lit[lit.len() / 2]
}

/// A truth seen on night `i` at `k` λ/D per pixel with noise level `j` (deterministic: the
/// same every time). Returns (blurred truth, degraded).
pub fn degrade(truth: &Image, name: &str, nights: &Nights, k: f32, i: usize, j: usize) -> (Image, Image) {
    let blurred = seeing::blur_with(truth, &nights.kernels(i, truth.is_mono(), k));
    let seed = (i * 7 + j) as u64 + name.len() as u64 * 131;
    let n = noise(truth.width(), truth.height(), truth.channels().len(), lit_median(truth) / SNRS[j], 0.6, seed);
    let pedestal = Image::new(truth.width(), truth.height(), vec![vec![0.01; truth.width() * truth.height()]; truth.channels().len()], 16);
    let degraded = add(&add(&blurred, &n), &pedestal);
    (blurred, degraded)
}

pub fn variant_name(i: usize, j: usize) -> String {
    format!("{}_snr{}", seeing::NIGHTS[i].label, SNRS[j] as usize)
}
