//! Saturn in two zones on the simulated stacks: what the globe and the rings each need, against
//! one curve for both.
//!
//! `cargo run --release -p ps-train --bin saturn_eval -- out_dir run…|@runs.txt [--apply preds.json]`
//!
//! Without `--apply`, for each run (a `sim_stack` folder with grezza, pari, dispari and its
//! truths) writes `out_dir/<run>.json`:
//! - the error of any band filter over the globe and over the rings as quadratic forms (as
//!   `sim_eval` does for the whole subject), so curves can be trained and scored per zone or for
//!   both together without touching an image again;
//! - the error over each zone of the raw and of the app's sharpening as it is now;
//! - the inputs of the Saturn networks (`ps_core::saturn::features` and `noise_features`), on the
//!   stack as it is and with its sky clipped to black;
//! - the true random noise in each zone: half the difference of the even- and odd-frame half
//!   stacks (realigned), band by band, brightness and colour, over the globe, the rings and both.
//!
//! With `--apply preds.json` (from `saturn_train.py --preds`: for each run the curves and the
//! noise of one network for both zones and of the two zone networks, never trained on that
//! original) the predictions are applied as the app applies them: the smart sharpening with the
//! network's curve (two curves blended by the globe's mask), and the guided noise reduction
//! (one estimate, or one per zone blended). Writes `out_dir/applicati.json` and views.
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::align::measure_shift;
use ps_core::analysis::analyze;
use ps_core::curve_net::{BANDS, band_centre, bands};
use ps_core::fft::FftConvolver;
use ps_core::saturn;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

fn load(p: &Path) -> Image {
    ml::load_any(p).unwrap_or_else(|| panic!("{}: illeggibile", p.display()))
}

fn fourier_shift(plane: &[f32], w: usize, h: usize, d: [f32; 2]) -> Vec<f32> {
    let plan = FftConvolver::plan(w, h);
    let mut spec = plan.forward(plane);
    for (idx, c) in spec.iter_mut().enumerate() {
        let (kx, ky) = (idx / h, idx % h);
        let fy = if ky <= h / 2 { ky as f32 } else { ky as f32 - h as f32 } / h as f32;
        let phase = std::f32::consts::TAU * (kx as f32 / w as f32 * d[0] + fy * d[1]);
        let (sn, cs) = phase.sin_cos();
        let (re, im) = (c.re, c.im);
        c.re = re * cs - im * sn;
        c.im = re * sn + im * cs;
    }
    plan.inverse(spec)
}

fn shifted(img: &Image, d: [f32; 2]) -> Image {
    let (w, h) = (img.width(), img.height());
    Image::new(w, h, img.channels().iter().map(|c| fourier_shift(c, w, h, d)).collect(), 16)
}

fn arr(v: &[f32]) -> String {
    format!("[{}]", v.iter().map(|x| if x.is_finite() { format!("{x:.5}") } else { "null".into() }).collect::<Vec<_>>().join(","))
}

fn arr64(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{x:.6e}")).collect::<Vec<_>>().join(","))
}

/// Σ(Σ_b g_b·x_b − t)² = gᵀMg − 2rᵀg + tt over `on`, and the truth's variance there.
fn quadratic(xb: &[Vec<f32>], t: &[f32], on: &[usize]) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let k = xb.len();
    let (mut m, mut r, mut tt) = (vec![0.0f64; k * k], vec![0.0f64; k], 0.0f64);
    let mean = on.iter().map(|&i| t[i] as f64).sum::<f64>() / on.len().max(1) as f64;
    let mut tvar = 0.0f64;
    let mut row = vec![0.0f64; k];
    for &i in on {
        for (j, band) in xb.iter().enumerate() {
            row[j] = band[i] as f64;
        }
        let ti = t[i] as f64;
        tt += ti * ti;
        tvar += (ti - mean).powi(2);
        for a in 0..k {
            r[a] += row[a] * ti;
            for c in a..k {
                m[a * k + c] += row[a] * row[c];
            }
        }
    }
    for a in 0..k {
        for c in 0..a {
            m[a * k + c] = m[c * k + a];
        }
    }
    (m, r, tt, tvar)
}

/// RMS error over `on` after the best scale (no offset: the sky is zero in both), over the
/// truth's spread there.
fn region_error(x: &[f32], t: &[f32], on: &[usize]) -> f32 {
    let (mut xt, mut xx, mut tt, mut s) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &i in on {
        xt += x[i] as f64 * t[i] as f64;
        xx += (x[i] as f64).powi(2);
        tt += (t[i] as f64).powi(2);
        s += t[i] as f64;
    }
    let n = on.len().max(1) as f64;
    let tvar = tt - s * s / n;
    (((tt - xt * xt / xx.max(1e-30)).max(0.0) / tvar.max(1e-30)).sqrt()) as f32
}

/// The stack with its sky clipped to black (everything below the sky's 99th percentile, far
/// from the rings).
fn black_sky(img: &Image) -> Image {
    let (w, h) = (img.width(), img.height());
    let Some(z) = saturn::zones(img) else { return img.clone() };
    let lum = ml::lum(img);
    let mut sky: Vec<f32> = (0..w * h).filter(|&i| ((i % w) as f32 - z.globe.cx).hypot((i / w) as f32 - z.globe.cy) > 0.65 * z.span).map(|i| lum[i]).collect();
    sky.sort_by(f32::total_cmp);
    let level = sky.get(sky.len() * 99 / 100).copied().unwrap_or(0.0);
    Image::new(w, h, img.channels().iter().map(|p| p.iter().map(|v| (v - level).max(0.0)).collect()).collect(), 16)
}

/// The zones scored against the truth (from the truth's own geometry): inside the globe (90 %
/// of its outline) and the rings with their edges (where the truth, lightly blurred, is above
/// 3 % of the globe's level, outside the globe), away from the frame's edges.
fn scored_zones(truth: &Image) -> Option<(Vec<usize>, Vec<usize>)> {
    let (w, h) = (truth.width(), truth.height());
    let z = saturn::zones(truth)?;
    let lt = ml::lum(truth);
    let at = |i: usize| ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
    let within = |i: usize| (12..w - 12).contains(&(i % w)) && (12..h - 12).contains(&(i / w));
    let planet: Vec<usize> = (0..w * h).filter(|&i| within(i) && z.globe.radius(at(i).0, at(i).1) < 0.9).collect();
    let level = planet.iter().map(|&i| lt[i] as f64).sum::<f64>() / planet.len().max(1) as f64;
    let soft = ps_core::mask::gaussian_approx(&lt, w, h, 2.0);
    let rings: Vec<usize> = (0..w * h).filter(|&i| within(i) && soft[i] as f64 > 0.03 * level && z.globe.radius(at(i).0, at(i).1) > 1.05).collect();
    Some((planet, rings))
}

/// Band powers (log10 over `mean`²) of a plane's bands over `px`.
fn power(b: &[Vec<f32>], px: &[usize], mean: f64) -> Vec<f32> {
    b.iter().map(|band| ((px.iter().map(|&i| (band[i] as f64).powi(2)).sum::<f64>() / px.len().max(1) as f64) / mean.powi(2)).max(1e-14).log10() as f32).collect()
}

fn add_log(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| (10f32.powf(*x) + 10f32.powf(*y)).log10()).collect()
}

fn union(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut u: Vec<usize> = a.iter().chain(b).copied().collect();
    u.sort_unstable();
    u.dedup();
    u
}

struct Run {
    name: String,
    raw: Image,
    truth: Image,
    even: Image,
    odd: Image,
}

fn open(run: &Path) -> Run {
    let name = run.file_name().unwrap().to_string_lossy().to_string();
    let raw = load(&run.join("grezza.tif"));
    let truth = load(&run.join("verita_campionata.tif"));
    let (w, h) = (raw.width(), raw.height());
    // The truth moved exactly onto the raw (an interpolated raw would lose its finest detail).
    let s = measure_shift(&ml::lum(&truth), &ml::lum(&raw), w, h);
    let truth = shifted(&truth, s);
    let (even, odd) = (load(&run.join("pari.tif")), load(&run.join("dispari.tif")));
    // The halves can sit a few hundredths of a pixel apart: not noise, each moves half of it.
    let o = measure_shift(&ml::lum(&even), &ml::lum(&odd), w, h);
    Run { name, raw, truth, even: shifted(&even, [-0.5 * o[0], -0.5 * o[1]]), odd: shifted(&odd, [0.5 * o[0], 0.5 * o[1]]) }
}

/// The app's sharpening as it is now (one-button setting, normal strength, subject only).
fn app_sharpen(img: &Image) -> Image {
    let a = analyze(img);
    let p = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
    let subject = ps_core::mask::subject_and_moons(img);
    ps_core::smart::sharpen(img, None, &p, Some(&subject))
}

fn measure_run(run: &Path) -> Option<String> {
    let r = open(run);
    let (w, h) = (r.raw.width(), r.raw.height());
    assert_eq!(w, h, "quadrati attesi");
    let (planet_on, rings_on) = scored_zones(&r.truth)?;
    let (lr, lt) = (ml::lum(&r.raw), ml::lum(&r.truth));
    let xb = bands(&lr, w);
    let (pm, pr, ptt, pvar) = quadratic(&xb, &lt, &planet_on);
    let (rm, rr, rtt, rvar) = quadratic(&xb, &lt, &rings_on);
    let app = ml::lum(&app_sharpen(&r.raw));
    let errors = |l: &[f32]| (region_error(l, &lt, &planet_on), region_error(l, &lt, &rings_on));
    let (e_raw, e_app) = (errors(&lr), errors(&app));

    let a = analyze(&r.raw);
    let (m, f) = saturn::measure(&r.raw, &a)?;
    let black = black_sky(&r.raw);
    let ab = analyze(&black);
    let (xs, nx) = (saturn::features(&m), saturn::noise_features(&m, &f));
    let (xs_b, nx_b) = saturn::measure(&black, &ab).map_or((xs.clone(), nx.clone()), |(mb, fb)| (saturn::features(&mb), saturn::noise_features(&mb, &fb)));

    // The true random noise per zone, framed as the networks measure (on the raw).
    let half: Image = Image::new(w, h, r.even.channels().iter().zip(r.odd.channels()).map(|(e, o)| e.iter().zip(o).map(|(a, b)| 0.5 * (a - b)).collect()).collect(), 16);
    let d = half.crop(f.x0, f.y0, f.side, f.side);
    let (dl, dc) = ps_core::noise_net::planes(&d);
    let (cl, cc) = ps_core::noise_net::planes(&f.crop);
    let both = union(&f.planet, &f.rings);
    let ring_px: &[usize] = if f.rings.len() >= saturn::MIN_RING_PIXELS { &f.rings } else { &f.planet };
    let bl = bands(&dl, f.side);
    let bcl = bands(&cl, f.side);
    let noise_l = |px: &[usize]| power(&bl, px, f.mean);
    let image_l = |px: &[usize]| power(&bcl, px, f.mean);
    let (noise_c, image_c): (Box<dyn Fn(&[usize]) -> Option<Vec<f32>>>, Box<dyn Fn(&[usize]) -> Option<Vec<f32>>>) = match (dc, cc) {
        (Some([a, b]), Some([c, e])) => {
            let (ba, bb, bc, be) = (bands(&a, f.side), bands(&b, f.side), bands(&c, f.side), bands(&e, f.side));
            let mean = f.mean;
            (Box::new(move |px: &[usize]| Some(add_log(&power(&ba, px, mean), &power(&bb, px, mean)))), Box::new(move |px: &[usize]| Some(add_log(&power(&bc, px, mean), &power(&be, px, mean)))))
        }
        _ => (Box::new(|_: &[usize]| None), Box::new(|_: &[usize]| None)),
    };
    let opt = |v: Option<Vec<f32>>| v.map_or("null".to_string(), |v| arr(&v));
    let zone = |px: &[usize]| format!("{{\"grana\": {}, \"colore\": {}, \"potenza\": {}, \"potenza_colore\": {}, \"pixel\": {}}}", arr(&noise_l(px)), opt(noise_c(px)), arr(&image_l(px)), opt(image_c(px)), px.len());
    // The two masks the app blends by, over the raw: globe in warm, rings in cool.
    if let Some(z) = saturn::zones(&r.raw) {
        let mut v: Vec<f32> = lr.iter().step_by(3).copied().collect();
        v.sort_by(f32::total_cmp);
        let top = v[v.len() * 999 / 1000].max(1e-6);
        let subject = ps_core::mask::subject_and_moons(&r.raw);
        let g: Vec<f32> = lr.iter().map(|x| (x / top).clamp(0.0, 1.0).powf(1.0 / 2.2)).collect();
        let tint = |k: usize| -> Vec<f32> {
            (0..w * h)
                .map(|i| {
                    let (p, q) = (z.planet[i], (subject[i] - z.planet[i]).max(0.0));
                    let c = [[1.0, 0.55, 0.2], [0.25, 0.6, 1.0]];
                    g[i] * (1.0 - 0.5 * (p + q)) + 0.5 * g[i].max(0.25) * (p * c[0][k] + q * c[1][k])
                })
                .collect()
        };
        let dir = run.parent().unwrap().parent().unwrap().parent().unwrap().join("maschere");
        std::fs::create_dir_all(&dir).ok();
        let _ = ps_io::save(&dir.join(format!("{}.jpg", r.name)), &Image::new(w, h, vec![tint(0), tint(1), tint(2)], 16));
    }
    let meta = std::fs::read_to_string(run.join("meta.json")).unwrap_or_else(|_| "{}".into());
    eprintln!("{}: globo {} px, anelli {} px | errore grezza {:.3}/{:.3} app {:.3}/{:.3}", r.name, planet_on.len(), rings_on.len(), e_raw.0, e_raw.1, e_app.0, e_app.1);
    Some(format!(
        "{{\"nome\": \"{}\", \"meta\": {},\n \"quad_pianeta\": {{\"m\": {}, \"r\": {}, \"tt\": {ptt:.6e}, \"tvar\": {pvar:.6e}, \"n\": {}}},\n \"quad_anelli\": {{\"m\": {}, \"r\": {}, \"tt\": {rtt:.6e}, \"tvar\": {rvar:.6e}, \"n\": {}}},\n \"errore_grezza\": [{:.5},{:.5}], \"errore_app\": [{:.5},{:.5}],\n \"x\": {}, \"x_nero\": {}, \"xn\": {}, \"xn_nero\": {},\n \"rumore\": {{\"pianeta\": {}, \"anelli\": {}, \"insieme\": {}}}, \"anelli_misurati\": {}, \"cielo\": {}}}\n",
        r.name,
        meta.trim(),
        arr64(&pm),
        arr64(&pr),
        planet_on.len(),
        arr64(&rm),
        arr64(&rr),
        rings_on.len(),
        e_raw.0,
        e_raw.1,
        e_app.0,
        e_app.1,
        arr(&xs),
        arr(&xs_b),
        arr(&nx),
        arr(&nx_b),
        zone(&f.planet),
        zone(ring_px),
        zone(&both),
        f.rings.len() >= saturn::MIN_RING_PIXELS,
        m.sky.is_some()
    ))
}

// ---------------------------------------------------------------- applying predictions

/// Predictions for one run: band gains (24, band 0 = 1) and noise (44: grain then colour,
/// bands 2‥23) for one network on both zones and for each zone's own.
struct Preds {
    single: Vec<f32>,
    planet: Vec<f32>,
    rings: Vec<f32>,
    noise_single: Vec<f32>,
    noise_planet: Vec<f32>,
    noise_rings: Vec<f32>,
}

/// Tiny reader of `{"run": {"key": [numbers], …}, …}`.
fn read_preds(path: &Path) -> std::collections::HashMap<String, Preds> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut out = std::collections::HashMap::new();
    for chunk in text.split("\"nome\": \"").skip(1) {
        let name = chunk[..chunk.find('"').unwrap()].to_string();
        let get = |key: &str| -> Vec<f32> {
            let at = chunk.find(&format!("\"{key}\": [")).unwrap_or_else(|| panic!("{key} mancante per {name}"));
            let open = at + key.len() + 4;
            let close = chunk[open..].find(']').unwrap() + open;
            chunk[open + 1..close].split(',').map(|v| v.trim().parse().unwrap()).collect()
        };
        out.insert(name.clone(), Preds { single: get("unica"), planet: get("pianeta"), rings: get("anelli"), noise_single: get("rumore_unica"), noise_planet: get("rumore_pianeta"), noise_rings: get("rumore_anelli") });
    }
    out
}

/// The smart sharpening with a network curve (band gains), as the app applies it.
fn network_params(base: &ps_core::smart::SmartParams, gains: &[f32], colour: ps_core::color::ColorMode) -> ps_core::smart::SmartParams {
    let (fc, st, table) = ps_core::curve_net::smart_curve(&ps_core::curve_net::Prediction { gains: gains.to_vec(), spread: 0.0 });
    ps_core::smart::SmartParams { curve: ps_core::smart::SmartCurve::Network, network: table, network_strength: st, network_cutoff: fc, strength: st, color: colour, ..base.clone() }
}

/// Noise-reduction score over a zone and a range of bands: 1 − error after / error before,
/// against the noise-free stack (each band of the truth scaled onto the raw's, in step with it).
fn denoise_score(raw: &[f32], out: &[f32], truth: &[f32], side: usize, px: &[usize], range: (f32, f32)) -> f32 {
    let (bx, bo, bt) = (bands(raw, side), bands(out, side), bands(truth, side));
    let (mut before, mut after) = (0.0f64, 0.0f64);
    for k in 2..BANDS {
        let c = band_centre(k);
        if c < range.0 || c >= range.1 {
            continue;
        }
        let (mut xt, mut tt) = (0.0f64, 0.0f64);
        for &i in px {
            xt += bx[k][i] as f64 * bt[k][i] as f64;
            tt += (bt[k][i] as f64).powi(2);
        }
        let alpha = xt / tt.max(1e-30);
        for &i in px {
            let clean = alpha * bt[k][i] as f64;
            before += (bx[k][i] as f64 - clean).powi(2);
            after += (bo[k][i] as f64 - clean).powi(2);
        }
    }
    (1.0 - after / before.max(1e-30)) as f32
}

fn chroma_sum(img: &Image) -> Option<Vec<f32>> {
    // Both colour differences' scores are computed on R − Y and B − Y separately and averaged.
    ps_core::noise_net::chroma_planes(img).map(|[a, b]| a.into_iter().chain(b).collect())
}

fn apply_run(run: &Path, preds: &Preds, out: &Path) -> Option<String> {
    let r = open(run);
    let (w, h) = (r.raw.width(), r.raw.height());
    let (planet_on, rings_on) = scored_zones(&r.truth)?;
    let lt = ml::lum(&r.truth);
    let a = analyze(&r.raw);
    let base = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
    let colour = ps_core::preset::learned_colour(&a);
    let subject = ps_core::mask::subject_and_moons(&r.raw);
    let zones = saturn::zones(&r.raw)?;
    let sharpen = |g: &[f32]| ps_core::smart::sharpen(&r.raw, None, &network_params(&base, g, colour), Some(&subject));
    let single = match std::env::var("SATURN_SINGLE_LIMB").ok().and_then(|v| v.parse::<f32>().ok()) {
        Some(l) => ps_core::smart::sharpen(&r.raw, None, &ps_core::smart::SmartParams { limb_protection: l, ..network_params(&base, &preds.single, colour) }, Some(&subject)),
        None => sharpen(&preds.single),
    };
    // Test: SATURN_RINGS_LIMB=x sets the rings' limb protection (the app's default otherwise).
    let rings_limb: Option<f32> = std::env::var("SATURN_RINGS_LIMB").ok().and_then(|v| v.parse().ok());
    let rings_sharp = match rings_limb {
        Some(l) => ps_core::smart::sharpen(&r.raw, None, &ps_core::smart::SmartParams { limb_protection: l, adaptive: std::env::var("SATURN_RINGS_ADAPTIVE").map_or(true, |v| v != "0"), ..network_params(&base, &preds.rings, colour) }, Some(&subject)),
        None => sharpen(&preds.rings),
    };
    let planet_sharp = match std::env::var("SATURN_PLANET_LIMB").ok().and_then(|v| v.parse::<f32>().ok()) {
        Some(l) => ps_core::smart::sharpen(&r.raw, None, &ps_core::smart::SmartParams { limb_protection: l, ..network_params(&base, &preds.planet, colour) }, Some(&subject)),
        None => sharpen(&preds.planet),
    };
    let dual = saturn::blend(&planet_sharp, &rings_sharp, &zones.planet);
    // Hybrid: the globe's own curve, the single network's on the rings (with the rings' edge setting).
    let single_on_rings = match rings_limb {
        Some(l) => ps_core::smart::sharpen(&r.raw, None, &ps_core::smart::SmartParams { limb_protection: l, ..network_params(&base, &preds.single, colour) }, Some(&subject)),
        None => sharpen(&preds.single),
    };
    let hybrid = saturn::blend(&planet_sharp, &single_on_rings, &zones.planet);
    // The app's final choice: the globe's curve everywhere (with the rings' edge setting there).
    let globe_on_rings = match rings_limb {
        Some(l) => ps_core::smart::sharpen(&r.raw, None, &ps_core::smart::SmartParams { limb_protection: l, ..network_params(&base, &preds.planet, colour) }, Some(&subject)),
        None => sharpen(&preds.planet),
    };
    let globe_all = saturn::blend(&planet_sharp, &globe_on_rings, &zones.planet);
    let app = app_sharpen(&r.raw);
    let err = |img: &Image| {
        let l = ml::lum(img);
        (region_error(&l, &lt, &planet_on), region_error(&l, &lt, &rings_on))
    };
    let (e_raw, e_app, e_single, e_dual, e_hybrid, e_globe) = (err(&r.raw), err(&app), err(&single), err(&dual), err(&hybrid), err(&globe_all));
    // The globe's limb against the sky (where the edge protection acts and halos would show):
    // a band around the outline, rings left out.
    let limb_on: Vec<usize> = {
        let z = saturn::zones(&r.truth)?;
        let ring: std::collections::HashSet<usize> = rings_on.iter().copied().collect();
        (0..w * h).filter(|&i| !ring.contains(&i) && (0.92..1.15).contains(&z.globe.radius((i % w) as f32 + 0.5, (i / w) as f32 + 0.5))).collect()
    };
    let limb_err = |img: &Image| region_error(&ml::lum(img), &lt, &limb_on);
    let e_limb = [limb_err(&r.raw), limb_err(&app), limb_err(&single), limb_err(&dual), limb_err(&hybrid)];

    // Noise: one estimate for both zones, or one per zone (blended); the true noise as ceiling.
    let (m, f) = saturn::measure(&r.raw, &a)?;
    let (y, chroma) = ps_core::noise_net::planes(&f.crop);
    let cpow = |px: &[usize]| chroma.as_ref().map(|[cr, cb]| add_log(&f.power(cr, px), &f.power(cb, px)));
    let both = union(&f.planet, &f.rings);
    let ring_px: Vec<usize> = if m.rings.is_some() { f.rings.clone() } else { f.planet.clone() };
    let split = |n: &[f32]| (n[..22].to_vec(), n[22..].to_vec());
    let reduce = |px: &[usize], n: &[f32]| {
        let (l, c) = split(n);
        let cp = cpow(px);
        ps_core::noise_net::denoise_with(&r.raw, 1.0, (&f.power(&y, px), &l), cp.as_deref().map(|cp| (cp, c.as_slice())), "")
    };
    let n_single = reduce(&both, &preds.noise_single);
    let n_dual = saturn::blend(&reduce(&f.planet, &preds.noise_planet), &reduce(&ring_px, &preds.noise_rings), &zones.planet);
    // The true noise of each zone (half stacks), framed the same way.
    let half: Image = Image::new(w, h, r.even.channels().iter().zip(r.odd.channels()).map(|(e, o)| e.iter().zip(o).map(|(a, b)| 0.5 * (a - b)).collect()).collect(), 16);
    let d = half.crop(f.x0, f.y0, f.side, f.side);
    let (dl, dc) = ps_core::noise_net::planes(&d);
    let true_noise = |px: &[usize]| -> Vec<f32> {
        let mut v = f.power(&dl, px)[2..].to_vec();
        match &dc {
            Some([a, b]) => v.extend(add_log(&f.power(a, px), &f.power(b, px))[2..].iter()),
            None => v.extend(std::iter::repeat_n(-14.0, 22)),
        }
        v
    };
    let n_true = saturn::blend(&reduce(&f.planet, &true_noise(&f.planet)), &reduce(&ring_px, &true_noise(&ring_px)), &zones.planet);
    let (lr, tl) = (ml::lum(&r.raw), lt.clone());
    let colour_truth = chroma_sum(&r.truth);
    let scores = |img: &Image| -> String {
        let l = ml::lum(img);
        let mut cells = Vec::new();
        for on in [&planet_on, &rings_on] {
            cells.push(denoise_score(&lr, &l, &tl, w, on, (0.2, 1.0)));
            cells.push(denoise_score(&lr, &l, &tl, w, on, (0.07, 0.2)));
            let c = match (chroma_sum(&r.raw), chroma_sum(img), &colour_truth) {
                (Some(cr), Some(co), Some(ct)) => {
                    // The two planes stacked as one taller image: score each, average.
                    let n = w * h;
                    let s1 = denoise_score(&cr[..n], &co[..n], &ct[..n], w, on, (0.07, 1.0));
                    let s2 = denoise_score(&cr[n..], &co[n..], &ct[n..], w, on, (0.07, 1.0));
                    0.5 * (s1 + s2)
                }
                _ => f32::NAN,
            };
            cells.push(c);
        }
        arr(&cells)
    };
    // Views: the whole system, raw / app / one network / two networks, and the noise versions.
    let lum_top = {
        let mut v: Vec<f32> = lr.iter().step_by(3).copied().collect();
        v.sort_by(f32::total_cmp);
        v[v.len() * 999 / 1000].max(1e-6)
    };
    let view = |img: &Image| Image::new(w, h, img.channels().iter().map(|p| p.iter().map(|v| (v / lum_top).clamp(0.0, 1.0).powf(1.0 / 2.2)).collect()).collect(), 16);
    let dir = out.join("viste");
    std::fs::create_dir_all(&dir).ok();
    for (tag, img) in [("grezza", &r.raw), ("app", &app), ("unica", &single), ("doppia", &dual), ("globo", &globe_all), ("verita", &r.truth), ("rumore_unica", &n_single), ("rumore_doppia", &n_dual)] {
        let _ = ps_io::save(&dir.join(format!("{}_{tag}.jpg", r.name)), &view(img));
    }
    eprintln!(
        "{}: nitidezza globo grezza {:.3} app {:.3} unica {:.3} doppia {:.3} | anelli {:.3} {:.3} {:.3} {:.3}",
        r.name, e_raw.0, e_app.0, e_single.0, e_dual.0, e_raw.1, e_app.1, e_single.1, e_dual.1
    );
    Some(format!(
        "{{\"nome\": \"{}\", \"nitidezza\": {{\"grezza\": [{:.5},{:.5}], \"app\": [{:.5},{:.5}], \"unica\": [{:.5},{:.5}], \"doppia\": [{:.5},{:.5}], \"ibrida\": [{:.5},{:.5}], \"globo\": [{:.5},{:.5}], \"bordo\": {}}}, \"rumore\": {{\"unica\": {}, \"doppia\": {}, \"vero\": {}}}}}",
        r.name,
        e_raw.0,
        e_raw.1,
        e_app.0,
        e_app.1,
        e_single.0,
        e_single.1,
        e_dual.0,
        e_dual.1,
        e_hybrid.0,
        e_hybrid.1,
        e_globe.0,
        e_globe.1,
        arr(&e_limb),
        scores(&n_single),
        scores(&n_dual),
        scores(&n_true)
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&out).unwrap();
    let mut runs: Vec<PathBuf> = Vec::new();
    let mut preds = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--app" || args[i] == "--giove" {
            i += 1;
            continue;
        }
        if args[i] == "--apply" {
            preds = Some(read_preds(Path::new(&args[i + 1])));
            i += 2;
            continue;
        }
        if let Some(list) = args[i].strip_prefix('@') {
            runs.extend(std::fs::read_to_string(list).unwrap().lines().filter(|l| !l.trim().is_empty()).map(PathBuf::from));
        } else {
            runs.push(PathBuf::from(&args[i]));
        }
        i += 1;
    }
    if let Some(k) = args.iter().position(|a| a == "--real") {
        // Real stacks: the app's automatic sharpening and noise reduction before (standard
        // curve) and now (Saturn's networks), side by side.
        args[k + 1..].par_iter().for_each(|path| {
            let raw = ps_io::load(Path::new(path)).unwrap();
            let mut a = analyze(&raw);
            let before = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
            a.curve_net = ps_core::curve_net::suggestion(&raw, &a, ps_core::capture::Body::Saturn).ok();
            let now = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
            let subject = ps_core::mask::subject_and_moons(&raw);
            let old = ps_core::smart::sharpen(&raw, None, &before, Some(&subject));
            let denoised = ps_core::noise_net::guided_denoise(&raw, 0.9).unwrap_or_else(|| raw.clone());
            let new = ps_core::smart::sharpen(&denoised, None, &now, Some(&subject));
            let lum = ml::lum(&raw);
            let mut v: Vec<f32> = lum.iter().step_by(3).copied().collect();
            v.sort_by(f32::total_cmp);
            let top = v[v.len() * 999 / 1000].max(1e-6);
            let (w, h) = (raw.width(), raw.height());
            let view = |img: &Image| -> Vec<Vec<f32>> {
                let ch: Vec<Vec<f32>> = img.channels().iter().map(|p| p.iter().map(|x| (x / top).clamp(0.0, 1.0).powf(1.0 / 2.2)).collect()).collect();
                if ch.len() == 1 { vec![ch[0].clone(), ch[0].clone(), ch[0].clone()] } else { ch }
            };
            let parts = [view(&raw), view(&old), view(&new)];
            let mw = 3 * w + 16;
            let planes: Vec<Vec<f32>> = (0..3).map(|c| {
                let mut o = vec![0.12f32; mw * h];
                for (k, p) in parts.iter().enumerate() {
                    for y in 0..h {
                        o[y * mw + k * (w + 8)..y * mw + k * (w + 8) + w].copy_from_slice(&p[c][y * w..(y + 1) * w]);
                    }
                }
                o
            }).collect();
            let name = Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
            ps_io::save(&out.join(format!("{name}.jpg")), &Image::new(mw, h, planes, 16)).unwrap();
            eprintln!("{name}: {:?} forza {:.2} → {:?} forza {:.2} (anelli: {})", before.curve, before.strength, now.curve, now.strength, now.network_rings.len() >= 2);
        });
        return;
    }
    if args.iter().any(|a| a == "--giove") {
        // Jupiter's sharpening network on Saturn's stacks (measured as for Jupiter: Saturn's
        // globe as the disk): its curve per run, scored later on Saturn's own quadratic forms.
        let model = ps_core::curve_net::model_for(ps_core::capture::Body::Jupiter).unwrap();
        let rows: Vec<String> = runs
            .par_iter()
            .filter_map(|run| {
                let raw = load(&run.join("grezza.tif"));
                let mut a = analyze(&raw);
                // Saturn's globe as the disk (the analysis often finds none behind the rings).
                let g = saturn::zones(&raw)?.globe;
                a.disk = Some(ps_core::analysis::Disk { cx: g.cx, cy: g.cy, radius: g.rx });
                let m = ps_core::curve_net::measure(&raw, &a)?;
                let p = model.predict(&ps_core::curve_net::features(&m))?;
                Some(format!("{{\"nome\": \"{}\", \"giove\": {}}}", run.file_name().unwrap().to_string_lossy(), arr(&p.gains)))
            })
            .collect();
        std::fs::write(out.join("giove_su_saturno.json"), format!("[\n{}\n]\n", rows.join(",\n"))).unwrap();
        return;
    }
    if args.iter().any(|a| a == "--app") {
        // The app's own path, as wired: analysis, the subject's suggestion, the learned curve
        // chosen with its strength, the smart sharpening; and the guided noise reduction.
        let rows: Vec<String> = runs
            .par_iter()
            .filter_map(|run| {
                let r = open(run);
                let (planet_on, rings_on) = scored_zones(&r.truth)?;
                let lt = ml::lum(&r.truth);
                let mut a = analyze(&r.raw);
                a.curve_net = ps_core::curve_net::suggestion(&r.raw, &a, ps_core::capture::Body::Saturn).ok();
                let n = a.curve_net.as_ref()?;
                let auto = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
                let p = ps_core::smart::SmartParams { curve: ps_core::smart::SmartCurve::Network, strength: n.strength, color: ps_core::preset::learned_colour(&a), ..auto };
                let subject = ps_core::mask::subject_and_moons(&r.raw);
                let both = ps_core::smart::sharpen(&r.raw, None, &p, Some(&subject));
                let globe_only = ps_core::smart::sharpen(&r.raw, None, &ps_core::smart::SmartParams { network_rings: Vec::new(), ..p.clone() }, Some(&subject));
                let e = |img: &Image| {
                    let l = ml::lum(img);
                    format!("[{:.4},{:.4}]", region_error(&l, &lt, &planet_on), region_error(&l, &lt, &rings_on))
                };
                let dn = ps_core::noise_net::guided_denoise(&r.raw, 1.0).map(|d| denoise_score(&ml::lum(&r.raw), &ml::lum(&d), &lt, r.raw.width(), &planet_on, (0.2, 1.0)));
                Some(format!("{{\"nome\": \"{}\", \"due_curve\": {}, \"solo_globo\": {}, \"rumore_fine_globo\": {:.4}}}", r.name, e(&both), e(&globe_only), dn.unwrap_or(f32::NAN)))
            })
            .collect();
        std::fs::write(out.join("app.json"), format!("[\n{}\n]\n", rows.join(",\n"))).unwrap();
        return;
    }
    match preds {
        None => runs.par_iter().for_each(|run| {
            let name = run.file_name().unwrap().to_string_lossy().to_string();
            let path = out.join(format!("{name}.json"));
            if path.exists() {
                return;
            }
            match measure_run(run) {
                Some(json) => std::fs::write(path, json).unwrap(),
                None => eprintln!("{name}: zone di Saturno non trovate"),
            }
        }),
        Some(p) => {
            let rows: Vec<String> = runs
                .par_iter()
                .filter_map(|run| {
                    let name = run.file_name().unwrap().to_string_lossy().to_string();
                    apply_run(run, p.get(&name)?, &out)
                })
                .collect();
            std::fs::write(out.join("applicati.json"), format!("[\n{}\n]\n", rows.join(",\n"))).unwrap();
        }
    }
}
