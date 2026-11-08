//! Where does the detail end? Several ways of measuring it, side by side with what
//! BlurXTerminator does to the same image (how much it amplifies each frequency, inside the
//! subject). Study tool, not used by the app.
//!
//! `cargo run --release -p ps-io --example cutoff_study -- cache_dir image…`
//!
//! BlurXTerminator results are cached in `cache_dir` (`<stem>_bxt.tif`). Frequencies are in
//! cycles/px (period = 1/f px). Methods:
//! - `attuale`: what the app uses today (`preset::detail_cutoff`, spectrum of a square centred
//!   on the subject, sky and limb included)
//! - `disco`: same model fit, on a square entirely inside the disk
//! - `plateau`: disk spectrum falls to 2× its own high-frequency floor
//! - `cielo`: noise spectrum SHAPE from the sky (keeps its correlation from stacking), LEVEL
//!   from the disk's highest frequencies; detail ends where the disk falls to 2× that
//! - `bordo`: limb edge-spread → MTF (MTF50 / MTF10)
//! - `decorr`: Descloux et al. 2019 decorrelation analysis on the disk square
//! - `FRC`: single-image Fourier ring correlation, checkerboard split (max 0.25)
use ps_core::Image;
use ps_core::analysis::{Analysis, ImageKind};
use ps_core::color::{ColorMode, working_planes};
use ps_core::fft::{FftConvolver, next_fast_size};
use ps_core::mask::Ellipse;
use ps_core::preset::{AutoStrength, auto_pipeline_with};
use ps_core::wavelet::ScaleMode;
use std::path::Path;

fn lum(img: &Image) -> Vec<f32> {
    working_planes(img, ColorMode::Luminance).swap_remove(0)
}

fn fast_below(mut n: usize) -> usize {
    while next_fast_size(n) != n {
        n -= 1;
    }
    n
}

/// Subject outline: the disk, Saturn's globe, or nothing (surface).
fn outline(img: &Image, a: &Analysis) -> Option<Ellipse> {
    match (a.kind, a.disk) {
        (ImageKind::Saturn, _) => ps_core::mask::saturn_globe(img),
        (_, Some(d)) => Some(Ellipse { cx: d.cx, cy: d.cy, rx: d.radius, ry: d.radius, angle: 0.0 }),
        _ => None,
    }
}

/// Square fully inside the subject (or centred, for a surface): (x0, y0, side).
fn inner_square(w: usize, h: usize, e: Option<Ellipse>) -> Option<(usize, usize, usize)> {
    let (cx, cy, half) = match e {
        // Inscribed square of a circle has half-side r/√2 ≈ 0.707 r; keep a margin from the limb.
        Some(e) => (e.cx, e.cy, 0.62 * e.rx.min(e.ry)),
        None => (w as f32 / 2.0, h as f32 / 2.0, (w.min(h).min(1024) / 2) as f32),
    };
    let side = fast_below(((2.0 * half) as usize).min(w).min(h));
    if side < 64 {
        return None;
    }
    let x0 = ((cx - side as f32 / 2.0).max(0.0) as usize).min(w - side);
    let y0 = ((cy - side as f32 / 2.0).max(0.0) as usize).min(h - side);
    Some((x0, y0, side))
}

fn crop(plane: &[f32], w: usize, (x0, y0, side): (usize, usize, usize)) -> Vec<f32> {
    (0..side).flat_map(|y| plane[(y0 + y) * w + x0..(y0 + y) * w + x0 + side].to_vec()).collect()
}

/// Mean removed, Hann window: (fx, fy, re, im) per coefficient of the half plane.
fn coefs(plane: &[f32], side: usize) -> Vec<[f32; 4]> {
    let mean = plane.iter().map(|&v| v as f64).sum::<f64>() as f32 / plane.len() as f32;
    let hann: Vec<f32> = (0..side).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * (i as f32 + 0.5) / side as f32).cos()).collect();
    let win: Vec<f32> = (0..side * side).map(|i| (plane[i] - mean) * hann[i % side] * hann[i / side]).collect();
    // Transposed layout: index = kx·side + ky.
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

/// Radial power in log bins from 0.02 to 0.5: (f, power).
fn radial(c: &[[f32; 4]]) -> Vec<(f32, f32)> {
    const BINS: usize = 60;
    let (lo, hi) = (0.02f32.ln(), 0.5f32.ln());
    let (mut sum, mut n) = (vec![0.0f64; BINS], vec![0u32; BINS]);
    for k in c {
        let f = k[0].hypot(k[1]);
        if !(0.02..=0.5).contains(&f) {
            continue;
        }
        let b = (((f.ln() - lo) / (hi - lo)) * BINS as f32).clamp(0.0, BINS as f32 - 1.0) as usize;
        sum[b] += (k[2] * k[2] + k[3] * k[3]) as f64;
        n[b] += 1;
    }
    (0..BINS).filter(|&b| n[b] > 0).map(|b| ((lo + (b as f32 + 0.5) / BINS as f32 * (hi - lo)).exp(), (sum[b] / n[b] as f64) as f32)).collect()
}

fn smooth(v: &[(f32, f32)]) -> Vec<(f32, f32)> {
    (0..v.len()).map(|i| (v[i].0, (v[i.saturating_sub(1)].1 + v[i].1 + v[(i + 1).min(v.len() - 1)].1) / 3.0)).collect()
}

/// First frequency (above 0.04) where `p` falls below `floor(f)`.
fn first_below(p: &[(f32, f32)], floor: impl Fn(f32) -> f32) -> f32 {
    p.iter().filter(|q| q.0 > 0.04).find(|q| q.1 < floor(q.0)).map_or(0.5, |q| q.0)
}

fn plateau(p: &[(f32, f32)]) -> f32 {
    let mut hi: Vec<f32> = p.iter().filter(|q| q.0 > 0.42).map(|q| q.1).collect();
    hi.sort_by(f32::total_cmp);
    hi.get(hi.len() / 2).copied().unwrap_or(0.0)
}

/// A dark square far from the subject, for the noise spectrum.
fn sky_square(plane: &[f32], w: usize, h: usize, e: Ellipse, side: usize) -> Option<(usize, usize, usize)> {
    let side = fast_below(side.min(w / 3).min(h / 3).max(64));
    let reach = e.rx.max(e.ry) * 2.3; // Saturn's rings reach ~2.3 equatorial radii.
    let mut best: Option<((usize, usize, usize), f32)> = None;
    let step = (side / 4).max(8);
    for y0 in (0..=h - side).step_by(step) {
        for x0 in (0..=w - side).step_by(step) {
            let (qx, qy) = (x0 as f32 + side as f32 / 2.0, y0 as f32 + side as f32 / 2.0);
            let corner_dist = ((qx - e.cx).abs() - side as f32 * 0.71).max(0.0).hypot(((qy - e.cy).abs() - side as f32 * 0.71).max(0.0));
            if corner_dist < reach * 0.6 && (qx - e.cx).hypot(qy - e.cy) < reach + side as f32 * 0.71 {
                continue;
            }
            let c = crop(plane, w, (x0, y0, side));
            let mut s = c.clone();
            s.sort_by(f32::total_cmp);
            let p99 = s[s.len() * 99 / 100];
            if best.as_ref().is_none_or(|b| p99 < b.1) {
                best = Some(((x0, y0, side), p99));
            }
        }
    }
    best.map(|b| b.0)
}

/// Edge-spread along the limb → MTF; returns (MTF50, MTF10) in cycles/px.
fn limb_mtf(plane: &[f32], w: usize, h: usize, e: Ellipse, saturn: bool) -> Option<(f32, f32, usize)> {
    let at = |x: f32, y: f32| -> f32 {
        if x < 0.0 || y < 0.0 || x >= (w - 1) as f32 || y >= (h - 1) as f32 {
            return f32::NAN;
        }
        let (x0, y0) = (x as usize, y as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |xx: usize, yy: usize| plane[yy * w + xx];
        p(x0, y0) * (1.0 - fx) * (1.0 - fy) + p(x0 + 1, y0) * fx * (1.0 - fy) + p(x0, y0 + 1) * (1.0 - fx) * fy + p(x0 + 1, y0 + 1) * fx * fy
    };
    const OS: f32 = 4.0; // ESF bins per px
    const HALF: f32 = 12.0; // px either side of the edge
    let nb = (2.0 * HALF * OS) as usize;
    let (mut sum, mut cnt) = (vec![0.0f64; nb], vec![0u32; nb]);
    let mut used = 0;
    for k in 0..1440 {
        let theta = k as f32 / 1440.0 * std::f32::consts::TAU;
        let phi = theta - e.angle;
        if saturn {
            // Only around the poles, away from the rings: within ±35° of the minor axis.
            let off = (phi.sin().abs()).acos(); // 0 at the poles
            if off > 35f32.to_radians() {
                continue;
            }
        }
        let r = e.rx * e.ry / ((e.ry * phi.cos()).powi(2) + (e.rx * phi.sin()).powi(2)).sqrt();
        let (dx, dy) = (theta.cos(), theta.sin());
        let prof: Vec<(f32, f32)> = (0..((2.0 * HALF + 16.0) * OS) as usize)
            .map(|i| {
                let t = r - HALF - 8.0 + i as f32 / OS;
                (t, at(e.cx + t * dx, e.cy + t * dy))
            })
            .collect();
        if prof.iter().any(|p| p.1.is_nan()) {
            continue;
        }
        let n = prof.len();
        let mut inside: Vec<f32> = prof[..n / 5].iter().map(|p| p.1).collect();
        let mut sky: Vec<f32> = prof[n * 4 / 5..].iter().map(|p| p.1).collect();
        inside.sort_by(f32::total_cmp);
        sky.sort_by(f32::total_cmp);
        let (vin, vsky) = (inside[inside.len() / 2], sky[sky.len() / 2]);
        if vin - vsky < 0.05 {
            continue;
        }
        let mid = 0.5 * (vin + vsky);
        let crossings: Vec<usize> = (1..n).filter(|&i| (prof[i - 1].1 - mid) * (prof[i].1 - mid) <= 0.0).collect();
        if crossings.len() != 1 {
            continue;
        }
        let i = crossings[0];
        let (a, b) = (prof[i - 1], prof[i]);
        let edge = a.0 + (mid - a.1) / (b.1 - a.1) * (b.0 - a.0);
        for p in &prof {
            let rel = p.0 - edge;
            let bin = ((rel + HALF) * OS).floor();
            if bin >= 0.0 && (bin as usize) < nb {
                sum[bin as usize] += ((p.1 - vsky) / (vin - vsky)) as f64;
                cnt[bin as usize] += 1;
            }
        }
        used += 1;
    }
    if used < 20 || cnt.iter().any(|&c| c == 0) {
        return None;
    }
    let esf: Vec<f32> = sum.iter().zip(&cnt).map(|(s, &c)| (s / c as f64) as f32).collect();
    // LSF (inside → sky is a falling edge; sign flipped), Hann window.
    let lsf: Vec<f32> = (0..nb - 1)
        .map(|i| {
            let x = (i as f32 + 1.0) / (nb - 1) as f32;
            (esf[i] - esf[i + 1]) * (0.5 - 0.5 * (std::f32::consts::TAU * x).cos())
        })
        .collect();
    let mtf = |f: f32| -> f32 {
        // f in cycles/px, sample spacing 1/OS px
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (i, &v) in lsf.iter().enumerate() {
            let ph = std::f32::consts::TAU * f * i as f32 / OS;
            re += v * ph.cos();
            im += v * ph.sin();
        }
        re.hypot(im)
    };
    let m0 = mtf(0.0).max(1e-9);
    let find = |level: f32| (1..=500).map(|i| i as f32 * 0.001).find(|&f| mtf(f) / m0 < level).unwrap_or(0.5);
    Some((find(0.5), find(0.1), used))
}

/// Descloux, Grußmayer & Radenovic 2019: decorrelation analysis. Returns cut-off in cycles/px.
fn decorrelation(c: &[[f32; 4]]) -> f32 {
    // Only the half plane is stored; ratios are unaffected.
    let r_of = |k: &[f32; 4]| k[0].hypot(k[1]) / 0.5; // 1 = Nyquist
    let mags: Vec<(f32, f32)> = c.iter().map(|k| (r_of(k), (k[2] * k[2] + k[3] * k[3]).sqrt())).filter(|m| m.0 > 0.0 && m.0 <= 1.0).collect();
    let curve = |weight: &dyn Fn(f32) -> f32| -> Vec<(f32, f32)> {
        // d(r) = Σ_{k<r} |I_g| / sqrt(Σ |I_g|² · N(r)),  In = I/|I| (|In| = 1)
        let total: f64 = mags.iter().map(|m| ((m.1 * weight(m.0)) as f64).powi(2)).sum();
        let mut sorted: Vec<(f32, f32)> = mags.iter().map(|m| (m.0, m.1 * weight(m.0))).collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let steps = 100;
        let (mut acc, mut n, mut j) = (0.0f64, 0usize, 0usize);
        (1..=steps)
            .map(|s| {
                let r = s as f32 / steps as f32;
                while j < sorted.len() && sorted[j].0 <= r {
                    acc += sorted[j].1 as f64;
                    n += 1;
                    j += 1;
                }
                (r, if n > 0 && total > 0.0 { (acc / (total * n as f64).sqrt()) as f32 } else { 0.0 })
            })
            .collect()
    };
    let peak = |d: &[(f32, f32)]| -> Option<(f32, f32)> {
        // Last local maximum that isn't at the end, with a meaningful amplitude.
        let mut best = None;
        for i in 1..d.len() - 1 {
            if d[i].1 >= d[i - 1].1 && d[i].1 > d[i + 1].1 && d[i].1 > 0.05 {
                best = Some(d[i]);
            }
        }
        best
    };
    let mut kc = peak(&curve(&|_| 1.0)).map_or(0.0, |p| p.0);
    // High-pass filtered versions (Gaussian high-pass of decreasing width).
    for g in 0..15 {
        let sigma = 0.6 * (0.03f32 / 0.6).powf(g as f32 / 14.0); // in Nyquist units
        if let Some(p) = peak(&curve(&|r| 1.0 - (-r * r / (2.0 * sigma * sigma)).exp())) {
            kc = kc.max(p.0);
        }
    }
    kc * 0.5
}

/// Checkerboard single-image FRC, threshold 1/7. Returns cut-off in cycles/px (≤ 0.25).
fn frc(plane: &[f32], side: usize) -> f32 {
    let half = fast_below(side / 2);
    let sub = |ox: usize, oy: usize| -> Vec<f32> { (0..half * half).map(|i| plane[(2 * (i / half) + oy) * side + 2 * (i % half) + ox]).collect() };
    let (a, b) = (coefs(&sub(0, 0), half), coefs(&sub(1, 1), half));
    let rings = (half / 2).min(64);
    let mut best = 0.0f32;
    for sign in [-1.0f32, 1.0] {
        let (mut num, mut da, mut db) = (vec![0.0f64; rings], vec![0.0f64; rings], vec![0.0f64; rings]);
        for (p, q) in a.iter().zip(&b) {
            let f = p[0].hypot(p[1]);
            let ring = (f / 0.5 * rings as f32) as usize;
            if ring >= rings {
                continue;
            }
            // b is shifted half a sub-pixel diagonally: undo its phase.
            let ph = sign * std::f32::consts::PI * (p[0] + p[1]);
            let (qr, qi) = (q[2] * ph.cos() - q[3] * ph.sin(), q[2] * ph.sin() + q[3] * ph.cos());
            num[ring] += (p[2] * qr + p[3] * qi) as f64;
            da[ring] += (p[2] * p[2] + p[3] * p[3]) as f64;
            db[ring] += (qr * qr + qi * qi) as f64;
        }
        let curve: Vec<f32> = (0..rings).map(|r| (num[r] / (da[r] * db[r]).sqrt().max(1e-30)) as f32).collect();
        let cut = (2..rings).find(|&r| curve[r] < 1.0 / 7.0).unwrap_or(rings);
        // sub-image frequency (cycles/sub-px) → original cycles/px: ÷2
        let f = (cut as f32 / rings as f32) * 0.5 / 2.0;
        best = best.max(f);
    }
    best
}

/// BlurXTerminator's amplitude gain by frequency inside the square, its peak and where it
/// falls back below 1 (and below 1.5).
fn bxt_response(input: &[f32], output: &[f32], side: usize) -> (Vec<(f32, f32)>, f32, f32, f32, f32) {
    let (pa, pb) = (radial(&coefs(input, side)), radial(&coefs(output, side)));
    let g: Vec<(f32, f32)> = smooth(&pa.iter().zip(&pb).map(|(a, b)| (a.0, (b.1 / a.1.max(1e-30)).sqrt())).collect::<Vec<_>>());
    let (ipk, &(fpk, gpk)) = g.iter().enumerate().max_by(|a, b| a.1.1.total_cmp(&b.1.1)).unwrap();
    let after = |level: f32| g[ipk..].iter().find(|q| q.1 < level).map_or(0.5, |q| q.0);
    (g.clone(), fpk, gpk, after(1.5), after(1.0))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cache = args.next().expect("cache dir");
    std::fs::create_dir_all(&cache).unwrap();
    let exe = ps_core::bxt::find_executable();
    println!(
        "{:<34} {:>6} {:>5} | {:>13} {:>5} {:>5} | {:>6} {:>6} {:>6} {:>6} {:>13} {:>6} {:>6} | {:>5}",
        "immagine", "tipo", "lato", "BXT picco(×)", "→1.5", "→1", "attual", "disco", "platea", "cielo", "bordo 50/10", "decorr", "FRC", "rumC"
    );
    for path in args {
        let name = Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
        let mut img = ps_io::load(path.as_ref()).unwrap();
        // Big surfaces: a central 1024² crop (BlurXTerminator on 8 MP+ is slow here).
        if img.width() * img.height() > 2_500_000 {
            let s = 1024.min(img.width()).min(img.height());
            img = img.crop((img.width() - s) / 2, (img.height() - s) / 2, s, s);
        }
        let a = ps_core::analysis::analyze(&img);
        let (w, h) = (img.width(), img.height());
        let plane = lum(&img);
        let e = outline(&img, &a);
        let Some(sq) = inner_square(w, h, e) else {
            println!("{name:<34} soggetto troppo piccolo");
            continue;
        };
        let disk = crop(&plane, w, sq);
        let side = sq.2;

        // BlurXTerminator, as the app would run it (PSF diameter from the limb).
        let cached = format!("{cache}/{}_bxt.tif", name.replace(' ', "_"));
        let bxt = if Path::new(&cached).exists() {
            Some(ps_io::load(cached.as_ref()).unwrap())
        } else if let Some(exe) = &exe {
            let params = auto_pipeline_with(a.kind, &a, AutoStrength::Normal, ScaleMode::Free).bxt.params;
            let out = ps_core::bxt::run(&img, &params, exe, &|| false).ok().flatten();
            if let Some(o) = &out {
                ps_io::save(cached.as_ref(), o).unwrap();
            }
            out
        } else {
            None
        };

        let current = ps_core::preset::detail_cutoff(&a).unwrap_or(f32::NAN);
        let c = coefs(&disk, side);
        let p = smooth(&radial(&c));
        let fit = ps_core::spectrum::fit(&radial(&c)).map_or(f32::NAN, |s| s.cutoff);
        let floor = plateau(&p);
        let plat = first_below(&p, |_| 2.0 * floor);
        let (sky, noise_ratio) = match e.and_then(|e| sky_square(&plane, w, h, e, side)) {
            Some(ssq) => {
                let n = smooth(&radial(&coefs(&crop(&plane, w, ssq), ssq.2)));
                // noise shape from the sky, level matched to the disk's highest frequencies
                let level = floor / plateau(&n).max(1e-30);
                let at = |f: f32| n.iter().min_by(|a, b| (a.0 / f).ln().abs().total_cmp(&(b.0 / f).ln().abs())).map_or(0.0, |q| q.1);
                // correlation of the noise: power at 0.1 vs 0.45 cycles/px
                (first_below(&p, |f| 2.0 * level * at(f)), at(0.1) / at(0.45).max(1e-30))
            }
            None => (f32::NAN, f32::NAN),
        };
        let limb = e.and_then(|e| limb_mtf(&plane, w, h, e, a.kind == ImageKind::Saturn));
        let dec = decorrelation(&c);
        let fr = frc(&disk, side);
        let (bx, curve) = match &bxt {
            Some(b) => {
                let (g, fpk, gpk, f15, f1) = bxt_response(&disk, &crop(&lum(b), w, sq), side);
                (format!("{fpk:.3}({gpk:4.1}) {f15:5.3} {f1:5.3}"), Some(g))
            }
            None => (format!("{:>13} {:>5} {:>5}", "-", "-", "-"), None),
        };
        let kind = format!("{:?}", a.kind);
        println!(
            "{:<34} {:>6} {:>5} | {bx} | {current:6.3} {fit:6.3} {plat:6.3} {sky:6.3} {:>13} {dec:6.3} {fr:6.3} | {noise_ratio:5.1}",
            &name[..name.len().min(34)],
            &kind[..kind.len().min(6)],
            side,
            limb.map_or("-".into(), |(m50, m10, n)| format!("{m50:.3}/{m10:.3}({n})"))
        );
        // Our "Nitidezza intelligente" at full strength, adaptive and uniform: realised gain.
        let ours = |adaptive: bool, limb: Option<f32>| -> Vec<(f32, f32)> {
            let mut sp = ps_core::preset::smart_structure(a.kind, &a);
            sp.strength = 1.0;
            sp.adaptive = adaptive;
            if let Some(l) = limb {
                sp.limb_protection = l;
            }
            let out = ps_core::smart::sharpen(&img, None, &sp, None);
            bxt_response(&disk, &crop(&lum(&out), w, sq), side).0
        };
        let rel = [0.25f32, 0.35, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.4];
        let at_rel = |g: &[(f32, f32)]| -> String {
            rel.iter()
                .map(|&r| {
                    let f = (r * current).min(0.5);
                    let v = g.iter().min_by(|a, b| (a.0 / f).ln().abs().total_cmp(&(b.0 / f).ln().abs())).unwrap().1;
                    format!("{v:5.1}")
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        if std::env::var_os("OURS").is_some() && current.is_finite() {
            println!("    f/limite            {}", rel.map(|r| format!("{r:5.2}")).join(" "));
            if let Some(g) = &curve {
                println!("    BXT                 {}", at_rel(g));
            }
            println!("    nostro adattivo     {}", at_rel(&ours(true, None)));
            println!("    nostro uniforme     {}", at_rel(&ours(false, None)));
            println!("    uniforme, no bordo  {}", at_rel(&ours(false, Some(0.0))));
        }
        // Strip: original | ours (35 %, 100 %) | BlurXTerminator, crop around the subject, zoomed.
        if let (Some(dir), Some(b)) = (std::env::var_os("STRIP"), &bxt) {
            let mut sp = ps_core::preset::smart_structure(a.kind, &a);
            let tiles: Vec<Vec<f32>> = [0.35f32, 1.0]
                .iter()
                .map(|&st| {
                    sp.strength = st;
                    lum(&ps_core::smart::sharpen(&img, None, &sp, None))
                })
                .collect();
            let tiles = [vec![plane.clone()], tiles, vec![lum(b)]].concat();
            let (cx, cy, rad) = e.map_or((w as f32 / 2.0, h as f32 / 2.0, 120.0), |e| (e.cx, e.cy, e.rx.max(e.ry) * 0.75));
            let half = rad.min(160.0);
            let (x0, y0) = ((cx - half).max(0.0) as usize, (cy - half).max(0.0) as usize);
            let s = ((2.0 * half) as usize).min(w - x0).min(h - y0);
            let zoom = (360 / s).clamp(1, 3);
            let t = s * zoom;
            let (ow, oh) = (tiles.len() * (t + 6), t);
            let mut sorted = plane.clone();
            sorted.sort_by(f32::total_cmp);
            let (lo, hi) = (sorted[sorted.len() / 100], sorted[sorted.len() * 999 / 1000].max(1e-4));
            let mut out = vec![0.0f32; ow * oh];
            for (k, src) in tiles.iter().enumerate() {
                for y in 0..t {
                    for x in 0..t {
                        out[y * ow + k * (t + 6) + x] = ((src[(y0 + y / zoom) * w + x0 + x / zoom] - lo) / (hi - lo)).clamp(0.0, 1.0);
                    }
                }
            }
            let file = format!("{}/{}.png", dir.to_string_lossy(), name.replace(' ', "_"));
            ps_io::save(Path::new(&file), &Image::new(ow, oh, vec![out], 16)).unwrap();
        }
        if let Some(g) = curve {
            let pts: Vec<String> = [0.04f32, 0.06, 0.08, 0.1, 0.125, 0.15, 0.18, 0.22, 0.27, 0.33, 0.4, 0.48]
                .iter()
                .map(|&f| {
                    let v = g.iter().min_by(|a, b| (a.0 / f).ln().abs().total_cmp(&(b.0 / f).ln().abs())).unwrap().1;
                    format!("{f}:{v:.1}")
                })
                .collect();
            println!("    BXT guadagno per f: {}", pts.join(" "));
        }
    }
}
