//! Classic (1·2·4·8 px) vs free wavelet scales on images whose sharp version is known.
//!
//! Each test image is shrunk 3× (the truth: sharper and cleaner than the stack), then blurred
//! and given noise again; a scene of random craters (no blur at all) comes first. Reported, as
//! RMS error relative to the unsharpened input (lower is better):
//! - the best possible linear gains for each kind of scales (least squares against the truth:
//!   the scales themselves compared, not two ways of guessing the sliders), with all layers,
//!   the best two neighbouring ones and the best single one;
//! - the automatic wavelets of each kind alone, at their own amount (k = 1) and at the best
//!   overall amount k; and the whole automatic processing (`FULL=1`: also its best k).
//!
//! The truth is still a little blurred, so anything sharpening past it counts as error: the
//! numbers compare the two kinds of scales, they don't say how sharp the result should be.
//! `SWEEP=1` prints every first scale tried.
//!
//! `cargo run --release -p ps-io --example wavelet_bench -- Immagini_test/*.tif`
use ps_core::Image;
use ps_core::color::{ColorMode, working_planes};
use ps_core::fft::FftConvolver;
use ps_core::pipeline::{Aux, Processor, next_generation};
use ps_core::preset::{AutoStrength, auto_pipeline_with};
use ps_core::psf::Psf;
use ps_core::spectrum;
use ps_core::wavelet::{ScaleMode, WaveletLayer, WaveletParams, decompose};
use std::sync::Arc;

const SHRINK: usize = 3;
const MARGIN: usize = 24;

fn shrink(plane: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (dw, dh) = (w / SHRINK, h / SHRINK);
    let mut out = vec![0.0f32; dw * dh];
    for y in 0..dh {
        for x in 0..dw {
            let mut acc = 0.0;
            for sy in 0..SHRINK {
                for sx in 0..SHRINK {
                    acc += plane[(y * SHRINK + sy) * w + x * SHRINK + sx];
                }
            }
            out[y * dw + x] = acc / (SHRINK * SHRINK) as f32;
        }
    }
    (out, dw, dh)
}

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n).map(|_| ((-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos()) as f32).collect()
}

/// Solve the small symmetric system `a·x = b` (Gaussian elimination with pivoting).
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, p);
        b.swap(c, p);
        let d = a[c][c];
        if d.abs() < 1e-18 {
            continue;
        }
        let pivot = a[c].clone();
        for r in c + 1..n {
            let f = a[r][c] / d;
            a[r][c..].iter_mut().zip(&pivot[c..]).for_each(|(v, p)| *v -= f * p);
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = if a[r][r].abs() < 1e-18 { 0.0 } else { (b[r] - s) / a[r][r] };
    }
    x
}

/// RMS error against the truth with the best gains on `planes` (out = input + Σ g·plane).
fn best(truth: &[f32], input: &[f32], planes: &[&[f32]], mask: &[usize]) -> (f64, Vec<f64>) {
    let n = planes.len();
    let mut a = vec![vec![0.0f64; n]; n];
    let mut b = vec![0.0f64; n];
    for &i in mask {
        let r = (truth[i] - input[i]) as f64;
        for j in 0..n {
            let pj = planes[j][i] as f64;
            b[j] += pj * r;
            for k in j..n {
                a[j][k] += pj * planes[k][i] as f64;
            }
        }
    }
    for j in 0..n {
        let column: Vec<f64> = (0..j).map(|k| a[k][j]).collect();
        a[j][..j].copy_from_slice(&column);
    }
    let g = if n > 0 { solve(a, b) } else { Vec::new() };
    let err = mask
        .iter()
        .map(|&i| {
            let out = input[i] as f64 + planes.iter().zip(&g).map(|(p, g)| g * p[i] as f64).sum::<f64>();
            (out - truth[i] as f64).powi(2)
        })
        .sum::<f64>();
    ((err / mask.len() as f64).sqrt(), g)
}

struct Scores {
    all: f64,
    one: f64,
    two: f64,
}

/// Best error with every layer free, with only one layer, with only two adjacent layers.
fn scores(truth: &[f32], input: &[f32], layers: &[&[f32]], mask: &[usize]) -> Scores {
    let all = best(truth, input, layers, mask).0;
    let one = (0..layers.len()).map(|j| best(truth, input, &layers[j..=j], mask).0).fold(f64::MAX, f64::min);
    let two = (0..layers.len() - 1).map(|j| best(truth, input, &layers[j..j + 2], mask).0).fold(f64::MAX, f64::min);
    Scores { all, one, two }
}

/// The automatic wavelets (with their noise-adaptive gain and thresholds; no deconvolution,
/// noise reduction or deringing) of each kind of scales, at the best overall amount: every
/// gain multiplied by k, k searched. "classic best (k) / free best (k), error at k = 1".
fn auto_pipelines(truth: &[f32], input: &[f32], w: usize, h: usize, mask: &[usize], base: f64) -> String {
    let img = Arc::new(Image::new(w, h, vec![input.to_vec()], 16));
    let a = ps_core::analysis::analyze(&img);
    let rms = |out: &[f32]| (mask.iter().map(|&i| ((out[i] - truth[i]) as f64).powi(2)).sum::<f64>() / mask.len() as f64).sqrt() / base;
    let mut parts = Vec::new();
    for scales in [ScaleMode::Dyadic, ScaleMode::Free] {
        let mut p = auto_pipeline_with(a.kind, &a, AutoStrength::Normal, scales);
        let mut proc = Processor::default();
        let generation = next_generation();
        let mut run = |p: &ps_core::pipeline::Pipeline| {
            let out = proc.run(Arc::clone(&img), generation, p, &Aux::default(), &|| false, &|_, _| {}).unwrap();
            rms(&out.image.channels()[0])
        };
        let full = run(&p);
        let (mut full_best, mut full_k) = (f64::MAX, 0.0);
        if std::env::var_os("FULL").is_some() {
            for step in 0..=16 {
                let k = step as f32 * 0.125;
                let mut q = p.clone();
                q.wavelet.params.layers.iter_mut().for_each(|l| l.gain *= k);
                let e = run(&q);
                if e < full_best {
                    (full_best, full_k) = (e, k);
                }
            }
        }
        (p.deconv.enabled, p.denoise.enabled, p.dering.enabled) = (false, false, false);
        let at_one = run(&p);
        let (mut best, mut best_k) = (f64::MAX, 0.0);
        for step in 0..=24 {
            let k = step as f32 * 0.125;
            let mut q = p.clone();
            q.wavelet.params.layers.iter_mut().for_each(|l| l.gain *= k);
            let e = run(&q);
            if e < best {
                (best, best_k) = (e, k);
            }
        }
        let label = if scales == ScaleMode::Free { format!("free@{:.2}", p.wavelet.params.first_scale) } else { "classic".into() };
        parts.push(format!(
            "{label} full auto {full:.3} best {full_best:.3} (k {full_k:.2}) | wavelets best {best:.3} (k {best_k:.2}) k=1 {at_one:.3} gains {:?}",
            p.wavelet.params.layers.iter().map(|l| l.gain).collect::<Vec<_>>()
        ));
    }
    format!("{:?} | {}", a.kind, parts.join(" · "))
}

/// Random craters (sharp-edged disks of every size): a power-law scene with no blur at all.
fn craters(side: usize) -> Vec<f32> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };
    let mut img = vec![0.4f32; side * side];
    for _ in 0..4000 {
        let (cx, cy) = (rnd() * side as f32, rnd() * side as f32);
        let r = 1.0 + 40.0 * rnd().powi(3);
        let depth = 0.12 * (rnd() - 0.5);
        let (x0, x1) = ((cx - r).max(0.0) as usize, ((cx + r) as usize + 1).min(side));
        let (y0, y1) = ((cy - r).max(0.0) as usize, ((cy + r) as usize + 1).min(side));
        for y in y0..y1 {
            for x in x0..x1 {
                if (x as f32 - cx).hypot(y as f32 - cy) < r {
                    img[y * side + x] += depth;
                }
            }
        }
    }
    img
}

fn main() {
    let mut rows = Vec::new();
    for path in std::iter::once("craters".to_string()).chain(std::env::args().skip(1)) {
        let name: String = path.rsplit('/').next().unwrap().chars().take(28).collect();
        let (truth, w, h) = if path == "craters" {
            (craters(768), 768, 768)
        } else {
            let img = ps_io::load(path.as_ref()).unwrap();
            let lum = working_planes(&img, ColorMode::Luminance).swap_remove(0);
            shrink(&lum, img.width(), img.height())
        };
        // The subject, away from the borders (the blur wraps around).
        let top = {
            let mut s: Vec<f32> = truth.iter().step_by(7).copied().collect();
            let k = s.len() * 99 / 100;
            *s.select_nth_unstable_by(k, f32::total_cmp).1
        };
        let mask: Vec<usize> = (0..w * h).filter(|&i| (MARGIN..w - MARGIN).contains(&(i % w)) && (MARGIN..h - MARGIN).contains(&(i / w)) && truth[i] > 0.1 * top).collect();
        for sigma in [1.4f32, 2.2] {
            for level in [0.0015f32, 0.004] {
                let blurred = FftConvolver::new(w, h, &Psf::Gaussian { sigma }.kernel()).convolve(&truth);
                let n = noise(w * h, 0x9E37_79B9_7F4A_7C15 ^ (sigma.to_bits() as u64) ^ ((level.to_bits() as u64) << 20));
                let input: Vec<f32> = blurred.iter().zip(&n).map(|(b, n)| b + level * n).collect();
                let spec = spectrum::analyze(&input, w, h, w as f32 / 2.0, h as f32 / 2.0);
                let cutoff = spec.map_or(f32::NAN, |s| s.cutoff);
                let base = best(&truth, &input, &[], &mask).0;

                let classic = WaveletParams { scale_mode: ScaleMode::Dyadic, layers: vec![WaveletLayer::default(); 6], ..Default::default() };
                let d = decompose(&input, w, h, &classic, &|| false).unwrap();
                let refs: Vec<&[f32]> = d.layers.iter().map(|l| l.as_slice()).collect();
                let c = scores(&truth, &input, &refs, &mask);

                let mut free_best: Option<(f32, Scores, Scores)> = None;
                let mut sweep = Vec::new();
                let mut first = 1.0f32;
                while first <= 4.01 {
                    let p = WaveletParams { scale_mode: ScaleMode::Free, first_scale: first, layers: vec![WaveletLayer::default(); 6], ..Default::default() };
                    let d = decompose(&input, w, h, &p, &|| false).unwrap();
                    let mut refs: Vec<&[f32]> = d.layers.iter().map(|l| l.as_slice()).collect();
                    let fixed = scores(&truth, &input, &refs, &mask);
                    // Also with the grain given a gain of its own (as noise reduction would).
                    if let Some((g, _)) = &d.grain {
                        refs.insert(0, g.as_slice());
                    }
                    let with_grain = best(&truth, &input, &refs, &mask).0;
                    sweep.push(format!("{first:.2}: all {:.3} +g {:.3}", fixed.all / base, with_grain / base));
                    if free_best.as_ref().is_none_or(|b| fixed.two < b.1.two) {
                        free_best = Some((first, fixed, Scores { all: with_grain, one: 0.0, two: 0.0 }));
                    }
                    first += 0.25;
                }
                let (first, f, g) = free_best.unwrap();
                if std::env::var_os("SWEEP").is_some() {
                    println!("{name} σ{sigma} n{level}: cutoff {cutoff:.3} fit {spec:?}\n   {}", sweep.join("\n   "));
                }
                let pipeline = auto_pipelines(&truth, &input, w, h, &mask, base);
                rows.push(format!(
                    "{name:28} σ{sigma:.1} n{level:.4} cut {cutoff:.3} | classic all {:.3} two {:.3} one {:.3} | free@{first:.2} all {:.3} (+grain {:.3}) two {:.3} one {:.3} | 0.306/first {:.3} | auto {pipeline}",
                    c.all / base,
                    c.two / base,
                    c.one / base,
                    f.all / base,
                    g.all / base,
                    f.two / base,
                    f.one / base,
                    0.306 / first,
                ));
            }
        }
    }
    println!("\nRMS error relative to the unsharpened input (lower is better):");
    for r in rows {
        println!("{r}");
    }
}
