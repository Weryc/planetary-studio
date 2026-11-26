//! Derotazione automatica sul globo: più riprese dello stesso pianeta (Giove, Saturno, Marte),
//! dello stesso filtro o tutte a colori, portate allo stesso istante e sommate.
//!
//! Come WinJUPOS, ma senza cornici da posare a mano:
//! 1. **Lembo** di ogni ripresa ([`crate::globe::fit_limb`]): centro, raggio e direzione
//!    dell'equatore *per ogni ripresa* (con una montatura altazimutale il campo ruota fra una
//!    ripresa e l'altra). Il raggio comune è la mediana.
//! 2. **Luce e dettaglio** ([`crate::globe::Shading`]): la luce (scurimento al bordo, orli
//!    lasciati dalla nitidezza, fase) non gira con il pianeta; viene tolta prima di derotare e
//!    rimessa alla fine. Così immagini grezze ed elaborate si confrontano allo stesso modo e
//!    l'orlo chiaro del bordo non viene trascinato dentro il disco.
//! 3. **Rotazione dalle immagini**: per ogni ripresa si cerca la rotazione (e una piccola
//!    correzione di centro e di rotazione di campo) che fa coincidere meglio il suo dettaglio,
//!    proiettato sul globo, con quello della ripresa di riferimento (correlazione normalizzata
//!    del dettaglio filtrato). Prima una ricerca ampia a bassa risoluzione, poi un affinamento.
//! 4. **Tempi**: con le ore delle riprese la rotazione attesa è `360°·Δt/periodo`; il confronto
//!    con quella misurata dà il verso (immagine specchiata o no) e scopre le ore sbagliate, che
//!    vengono corrette. Le ore giuste sono più precise della misura e vengono usate quelle.
//!    Senza ore basta la misura (le ore vengono stimate a partire da una ripresa datata, se c'è).
//! 5. **Somma**: per ogni pixel del risultato (la geometria della ripresa più vicina all'istante
//!    di riferimento) il punto del globo viene cercato in ogni ripresa, girato della sua
//!    rotazione (Giove: la zona equatoriale con il Sistema I), e i valori si mediano pesati per
//!    nitidezza della ripresa e angolo di vista (vicino al bordo conta chi lo vede di fronte).
//!    Fuori dal globo e sugli anelli di Saturno davanti al globo le riprese sono solo
//!    allineate; le lune si muovono: restano quelle della ripresa più vicina all'istante.
//!
//! [`planet_map`] dà la mappa cilindrica del pianeta dalle stesse riprese (per un globo che ruota).

use super::{JUPITER_SYSTEM_I_HOURS, Params, catmull_rom, combine, hours_between, mean_moment, rotation_period, smoothstep};
use crate::capture::{Body, Moment};
use crate::globe::{self, Globe, PlanetMap, Projection, Shading, lon_lat, point, turn};
use crate::image::{Image, bilinear_clamped};
use crate::tr;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Una ripresa da sommare.
#[derive(Clone, Copy, Debug)]
pub struct Capture<'a> {
    pub image: &'a Image,
    /// L'ora della ripresa (UT), se nota.
    pub time: Option<Moment>,
    /// Il globo posato a mano: non viene più misurato (solo affinato di poco se `refine`).
    pub globe: Option<Globe>,
}

/// L'istante a cui portare le riprese.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Reference {
    /// La media (pesata) delle riprese.
    #[default]
    Mean,
    /// Quello di una ripresa.
    Frame(usize),
}

/// Come derotare.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub body: Body,
    pub reference: Reference,
    /// Le riprese più nitide contano di più.
    pub weigh_sharpness: bool,
    /// Sigma-clipping sul globo (soglia in deviazioni standard), da 3 riprese.
    pub clip: Option<f32>,
    /// Usa le ore dichiarate (quando concordano con le immagini).
    pub use_times: bool,
    /// Misura la rotazione sulle immagini (altrimenti solo le ore).
    pub measure: bool,
    /// Immagini specchiate; `None` = dal confronto fra ore e immagini.
    pub mirrored: Option<bool>,
    /// Latitudine del centro del disco (B), gradi; `None` = dalle effemeridi.
    pub tilt_deg: Option<f64>,
    /// Giove: la zona equatoriale gira con il Sistema I.
    pub zonal: bool,
}

impl Settings {
    pub fn new(body: Body) -> Settings {
        Settings { body, reference: Reference::Mean, weigh_sharpness: true, clip: None, use_times: true, measure: true, mirrored: None, tilt_deg: None, zonal: true }
    }
}

/// Cosa non va.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Il pianeta non si derota (solo Giove, Saturno, Marte).
    Body(Body),
    /// Nessuna ripresa.
    Empty,
    /// Il disco non si trova nella ripresa `i`.
    NoDisk(usize),
}

impl Error {
    pub fn message(&self) -> String {
        match self {
            Error::Body(b) => tr!("La derotazione vale per Giove, Saturno e Marte, non per {}.", "Derotation applies to Jupiter, Saturn and Mars, not {}.", b.label()),
            Error::Empty => crate::i18n::tl("Nessuna ripresa da sommare").to_string(),
            Error::NoDisk(i) => tr!("Il disco del pianeta non si trova nella ripresa {}.", "The planet's disk was not found in capture {}.", i + 1),
        }
    }
}

/// Una ripresa preparata: il globo misurato, la luce, il dettaglio e i piani per il confronto.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// Il globo da cui si parte (misurato sul lembo o posato a mano).
    pub globe: Globe,
    /// Com'è andata la misura del lembo (`None` se posato a mano o non riuscita).
    pub limb: Option<globe::LimbFit>,
    pub shading: Shading,
    /// Quanto dettaglio fine mostra (confrontabile fra riprese della stessa sera).
    pub sharpness: f32,
    /// Rumore del cielo (deviazione standard robusta).
    pub noise: f32,
    width: usize,
    height: usize,
    /// Immagine / luce, per canale (fuori dal disco: l'immagine sul livello del bordo).
    detail: Vec<Vec<f32>>,
    /// Dettaglio filtrato e normalizzato per il confronto: fine e grossolano.
    fine: Vec<f32>,
    coarse: Vec<f32>,
}

/// L'oblatezza e la geometria di un pianeta.
fn oblateness(body: Body) -> f64 {
    match body {
        Body::Mars => 0.9941,
        _ => crate::moons::flattening(body),
    }
}

/// Il globo iniziale dall'analisi automatica.
fn first_guess(img: &Image) -> Option<Globe> {
    let analysis = crate::analysis::analyze(img);
    super::find_globe(img, &analysis)
}

/// Il globo di una ripresa: dato a mano, o misurato sul lembo partendo dall'analisi.
fn locate_globe(img: &Image, lum: &[f32], given: Option<Globe>, body: Body, proj: &Projection) -> Option<(Globe, Option<globe::LimbFit>)> {
    if let Some(g) = given {
        return Some((g, None));
    }
    let start = first_guess(img)?;
    // Saturn's angle comes from the rings (the limb is half hidden); Jupiter's from its
    // flattening; Mars is round.
    let fit_angle = body == Body::Jupiter;
    Some(match globe::fit_limb(lum, img.width(), img.height(), start, proj, fit_angle) {
        Some(f) => (f.globe, Some(f)),
        None => (start, None),
    })
}

/// Saturn's rings are left out of the light's fit.
fn shading_of(img: &Image, globe: &Globe, body: Body, proj: &Projection, slope: Option<&[[f32; 2]]>) -> Shading {
    let flipped = Projection::new(-proj.tilt_deg(), proj.k);
    let rings = body == Body::Saturn;
    Shading::fit_with(img, globe, proj, &|u, v| rings && ring_any(proj, &flipped, u, v), slope)
}

/// The rings in front of the globe or around it, for either sign of the tilt (the sign is
/// found later).
fn ring_any(proj: &Projection, flipped: &Projection, u: f64, v: f64) -> bool {
    let inside = |r: Option<f64>| r.is_some_and(|r| r > globe::SATURN_RINGS.0 - 0.05 && r < globe::SATURN_RINGS.1 + 0.05);
    inside(proj.ring_radius(u, v)) || inside(flipped.ring_radius(u, v))
}

fn finish(img: &Image, lum: &[f32], globe: Globe, limb: Option<globe::LimbFit>, body: Body, proj: &Projection, shading: Shading) -> Prepared {
    let (w, h) = (img.width(), img.height());
    let flipped = Projection::new(-proj.tilt_deg(), proj.k);
    let rings = body == Body::Saturn;
    let skip = |u: f64, v: f64| rings && ring_any(proj, &flipped, u, v);
    let detail: Vec<Vec<f32>> = img
        .channels()
        .par_iter()
        .enumerate()
        .map(|(c, plane)| {
            let floor = 0.08 * shading.peak(c).max(1e-6);
            (0..w * h)
                .map(|i| {
                    let (u, v) = globe.to_plane((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                    plane[i] / shading.at(c, proj.mu(u, v), u, v).max(floor)
                })
                .collect()
        })
        .collect();
    // The disk (where the detail is compared) and the rest (Saturn's rings).
    let disk: Vec<f32> = (0..w * h)
        .map(|i| {
            let (u, v) = globe.to_plane((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            if proj.rho(u, v) < 0.97 && !skip(u, v) { 1.0 } else { 0.0 }
        })
        .collect();
    let dlum: Vec<f32> = (0..w * h).map(|i| detail.iter().map(|d| d[i]).sum::<f32>() / detail.len() as f32).collect();
    let r = globe.radius;
    // On the disk the detail (which turns); around it the image itself: the limb's edge and
    // Saturn's rings, which don't turn and pin the centre and the field rotation.
    let off: Vec<f32> = disk.iter().map(|d| 1.0 - d).collect();
    let band = |sigma: f32| -> Vec<f32> {
        let (on_disk, around) = rayon::join(|| band_pass(&dlum, Some(&disk), w, h, sigma), || band_pass(lum, Some(&off), w, h, sigma));
        (0..w * h).map(|i| if disk[i] > 0.0 { on_disk[i] } else { around[i] }).collect()
    };
    let (fine, coarse) = rayon::join(|| band(fine_sigma(r)), || band(coarse_sigma(r)));
    // Sky noise beyond the planet (and Saturn's rings).
    let reach = if rings { 2.5 } else { 1.25 };
    let mut sky: Vec<f32> = (0..w * h)
        .step_by(3)
        .filter(|&i| {
            let (u, v) = globe.to_plane((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            proj.rho(u, v) > reach
        })
        .map(|i| lum[i])
        .collect();
    let noise = if sky.len() > 50 {
        let m = sky.len() / 2;
        let med = *sky.select_nth_unstable_by(m, f32::total_cmp).1;
        let mut dev: Vec<f32> = sky.iter().map(|v| (v - med).abs()).collect();
        let m = dev.len() / 2;
        1.4826 * *dev.select_nth_unstable_by(m, f32::total_cmp).1
    } else {
        0.01
    };
    let quality = detail_quality(&dlum, lum, &shading, &globe, proj, w, h, rings);
    Prepared { globe, limb, shading, sharpness: quality, noise: noise.max(1e-4), width: w, height: h, detail, fine, coarse }
}

/// How much fine detail the globe shows, above the noise: the energy of the fine band of the
/// detail (light removed, so the brightness of the capture doesn't count) inside 80 % of the
/// globe, minus what the same band gives on the sky (scaled to the disk's level). Comparable
/// between captures of one evening, even at different exposures.
#[allow(clippy::too_many_arguments)]
fn detail_quality(dlum: &[f32], lum: &[f32], shading: &Shading, globe: &Globe, proj: &Projection, w: usize, h: usize, rings: bool) -> f32 {
    let band = |p: &[f32]| {
        let a = globe::blur(p, w, h, 0.8);
        let b = globe::blur(p, w, h, 2.5);
        a.iter().zip(&b).map(|(x, y)| x - y).collect::<Vec<f32>>()
    };
    let (bd, bs) = rayon::join(|| band(dlum), || band(lum));
    let level = (0..shading.profile.len()).map(|c| shading.level(c)).sum::<f32>() / shading.profile.len().max(1) as f32;
    let (mut disk, mut nd, mut sky, mut ns) = (0.0f64, 0usize, 0.0f64, 0usize);
    let reach = if rings { 2.5 } else { 1.3 };
    for i in (0..w * h).step_by(2) {
        let (u, v) = globe.to_plane((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
        let rho = proj.rho(u, v);
        // Saturn: well clear of the rings crossing the globe (a bright line when edge-on).
        let margin = 5.0 / globe.radius as f64;
        let ring = rings && [-margin, 0.0, margin].iter().any(|d| proj.ring_radius(u, v + d).is_some());
        if rho < 0.8 && !ring {
            disk += (bd[i] as f64).powi(2);
            nd += 1;
        } else if rho > reach {
            sky += (bs[i] as f64 / level.max(1e-6) as f64).powi(2);
            ns += 1;
        }
    }
    if nd == 0 {
        return 0.0;
    }
    let signal = disk / nd as f64 - if ns > 0 { sky / ns as f64 } else { 0.0 };
    signal.max(1e-12).sqrt() as f32
}

/// Scale of the detail compared in the fine search, px.
fn fine_sigma(radius: f32) -> f32 {
    (radius / 50.0).max(2.0)
}

/// Scale of the detail compared in the wide search, px.
fn coarse_sigma(radius: f32) -> f32 {
    (radius / 16.0).max(4.0)
}

/// Band-pass (difference of Gaussians from `sigma/4` to `sigma`) normalised to unit local
/// contrast, inside `mask` (normalised convolution: the edges of the mask don't leak in).
fn band_pass(plane: &[f32], mask: Option<&[f32]>, w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let g = |p: &[f32], s: f32| globe::blur(p, w, h, s);
    let ones;
    let m = match mask {
        Some(m) => m,
        None => {
            ones = vec![1.0f32; w * h];
            &ones
        }
    };
    let xm: Vec<f32> = plane.iter().zip(m).map(|(x, m)| x * m).collect();
    let small = (sigma / 4.0).max(0.6);
    let (gm_s, gx_s) = rayon::join(|| g(m, small), || g(&xm, small));
    let (gm_l, gx_l) = rayon::join(|| g(m, sigma), || g(&xm, sigma));
    let r: Vec<f32> = (0..w * h)
        .map(|i| {
            if m[i] <= 0.0 || gm_s[i] < 1e-3 || gm_l[i] < 1e-3 {
                0.0
            } else {
                gx_s[i] / gm_s[i] - gx_l[i] / gm_l[i]
            }
        })
        .collect();
    let r2: Vec<f32> = r.iter().map(|v| v * v).collect();
    let var = g(&r2, 2.0 * sigma);
    let gm_v = g(m, 2.0 * sigma);
    let mean_var = {
        let (s, n) = var.iter().zip(&gm_v).filter(|(_, g)| **g > 0.2).fold((0.0f64, 0usize), |(s, n), (v, g)| (s + (*v / *g) as f64, n + 1));
        if n > 0 { (s / n as f64) as f32 } else { 1.0 }
    };
    // Gentle normalisation: busy areas don't drown quiet ones, but a quiet area's faint
    // leftovers are not blown up to the level of real detail.
    let eps = 0.5 * mean_var + 1e-12;
    (0..w * h).map(|i| if m[i] <= 0.0 { 0.0 } else { r[i] / (var[i] / gm_v[i].max(1e-3) + eps).sqrt() }).collect()
}

/// The rotation of the planet at planetographic latitude `lat` (radians) for a rotation
/// `turn` of the main system: Jupiter's equatorial zone (System I) runs ahead.
fn zonal_turn(params: &Params, zonal: bool, lat: f64, turn: f64) -> f64 {
    let Some(eq) = params.equator_period_hours.filter(|_| zonal) else { return turn };
    let a = lat.abs().to_degrees();
    // System I within 7°, System II beyond 12°.
    let t = ((12.0 - a) / 5.0).clamp(0.0, 1.0);
    let t = t * t * (3.0 - 2.0 * t);
    turn * (1.0 + t * (params.period_hours / eq - 1.0))
}

/// One point of the reference to compare.
struct Sample {
    /// Body point (globe samples) or `None` (rigid: Saturn's rings).
    p: Option<[f64; 3]>,
    lat: f64,
    u: f64,
    v: f64,
    mu: f32,
    value: f32,
}

/// Points of the reference frame on a grid of `step` pixels.
fn samples(prep: &Prepared, proj: &Projection, body: Body, fine: bool, step: f32) -> Vec<Sample> {
    let g = prep.globe;
    let plane = if fine { &prep.fine } else { &prep.coarse };
    let (w, h) = (prep.width, prep.height);
    let reach = if body == Body::Saturn { globe::SATURN_RINGS.1 + 0.05 } else { 1.12 };
    let r = g.radius * reach as f32 * 1.05;
    let mut out = Vec::new();
    let mut y = (g.cy - r).max(1.0);
    while y < (g.cy + r).min(h as f32 - 1.0) {
        let mut x = (g.cx - r).max(1.0);
        while x < (g.cx + r).min(w as f32 - 1.0) {
            let (u, v) = g.to_plane(x, y);
            let rho = proj.rho(u, v);
            let ring = body == Body::Saturn && proj.ring_radius(u, v).is_some_and(|r| r > globe::SATURN_RINGS.0 && r < globe::SATURN_RINGS.1);
            let value = bilinear_clamped(plane, w, h, x, y);
            if rho < 0.93 && !ring {
                if let Some(p) = proj.to_globe(u, v) {
                    let (_, mu) = proj.to_plane(p);
                    let (_, lat) = lon_lat(p, proj.k);
                    out.push(Sample { p: Some(p), lat, u, v, mu: mu as f32, value });
                }
            } else if (ring || (0.97..1.12).contains(&rho)) && value != 0.0 {
                out.push(Sample { p: None, lat: 0.0, u, v, mu: 1.0, value });
            }
            x += step;
        }
        y += step;
    }
    out
}

/// The frame's geometry moved by the parameters (turn, field rotation, shift).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Pose {
    /// Rotation of the planet from the reference to this frame, radians (main system).
    turn: f64,
    /// Field rotation, radians.
    spin: f64,
    dx: f64,
    dy: f64,
}

impl Pose {
    fn apply(&self, g: Globe) -> Globe {
        Globe { cx: g.cx + self.dx as f32, cy: g.cy + self.dy as f32, angle: g.angle + self.spin as f32, ..g }
    }
}

/// Normalised cross-correlation of the reference samples with `frame` seen through `pose`.
fn score(samples: &[Sample], frame: &Prepared, proj: &Projection, params: &Params, zonal: bool, fine: bool, pose: &Pose) -> f64 {
    let g = pose.apply(frame.globe);
    let plane = if fine { &frame.fine } else { &frame.coarse };
    let (w, h) = (frame.width, frame.height);
    let (mut sw, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for s in samples {
        let (u, v, wt) = match s.p {
            Some(p) => {
                let q = turn(p, zonal_turn(params, zonal, s.lat, pose.turn));
                let ((u, v), mu) = proj.to_plane(q);
                if mu < 0.2 {
                    continue;
                }
                (u, v, (s.mu * mu as f32) as f64)
            }
            None => (s.u, s.v, 0.7),
        };
        let (x, y) = g.to_pixel(u, v);
        if x < 1.0 || y < 1.0 || x > w as f32 - 1.0 || y > h as f32 - 1.0 {
            continue;
        }
        let b = bilinear_clamped(plane, w, h, x, y) as f64;
        let a = s.value as f64;
        sw += wt;
        sa += wt * a;
        sb += wt * b;
        saa += wt * a * a;
        sbb += wt * b * b;
        sab += wt * a * b;
    }
    if sw <= 0.0 {
        return -1.0;
    }
    let (ma, mb) = (sa / sw, sb / sw);
    let va = saa / sw - ma * ma;
    let vb = sbb / sw - mb * mb;
    if va <= 1e-12 || vb <= 1e-12 {
        return -1.0;
    }
    (sab / sw - ma * mb) / (va * vb).sqrt()
}

/// Coordinate search with halving steps on the active parameters within bounds.
fn pattern_search(eval: &dyn Fn(&Pose) -> f64, start: Pose, steps: [f64; 4], bounds: [f64; 4], centre: Pose, min_step: [f64; 4]) -> (Pose, f64) {
    let mut best = start;
    let mut best_v = eval(&best);
    let mut steps = steps;
    let get = |p: &Pose, k: usize| [p.turn, p.spin, p.dx, p.dy][k];
    let set = |p: &mut Pose, k: usize, v: f64| match k {
        0 => p.turn = v,
        1 => p.spin = v,
        2 => p.dx = v,
        _ => p.dy = v,
    };
    for _ in 0..60 {
        let mut improved = false;
        for k in 0..4 {
            if steps[k] <= 0.0 {
                continue;
            }
            for dir in [-1.0, 1.0] {
                let mut p = best;
                let v = get(&best, k) + dir * steps[k];
                if (v - get(&centre, k)).abs() > bounds[k] {
                    continue;
                }
                set(&mut p, k, v);
                let val = eval(&p);
                if val > best_v {
                    best = p;
                    best_v = val;
                    improved = true;
                    break;
                }
            }
        }
        if !improved {
            let mut any = false;
            for k in 0..4 {
                if steps[k] > 0.0 {
                    steps[k] *= 0.5;
                    if steps[k] < min_step[k] {
                        steps[k] = 0.0;
                    } else {
                        any = true;
                    }
                }
            }
            if !any {
                break;
            }
        }
    }
    (best, best_v)
}

/// What was measured for one frame against the reference.
#[derive(Clone, Copy, Debug)]
struct Measure {
    pose: Pose,
    /// Correlation after the fine search.
    score: f64,
    /// How distinct the peak is (0 = flat: the turn is not measurable).
    confidence: f64,
}

/// Search the frame's turn (within `range` radians around `centre`) on the coarse planes, then
/// refine turn, field rotation and centre on the fine ones.
#[allow(clippy::too_many_arguments)]
fn measure(reference: &Prepared, frame: &Prepared, proj: &Projection, params: &Params, body: Body, zonal: bool, centre: f64, range: f64, fixed_turn: bool) -> Measure {
    let r = reference.globe.radius as f64;
    let sig_c = coarse_sigma(r as f32) as f64;
    let coarse = samples(reference, proj, body, false, (sig_c / 2.5).max(1.0) as f32);
    let fine_step = (fine_sigma(r as f32) / 1.5).max(1.0);
    let fine = samples(reference, proj, body, true, fine_step);
    // Coarse scan of the turn.
    let step = (0.4 * sig_c / r).max(0.2f64.to_radians());
    let n = (range / step).ceil() as i64;
    let scan: Vec<(f64, f64)> = if fixed_turn || n == 0 {
        vec![(centre, score(&coarse, frame, proj, params, zonal, false, &Pose { turn: centre, ..Default::default() }))]
    } else {
        (-n..=n).into_par_iter().map(|k| {
            let t = centre + k as f64 * step;
            (t, score(&coarse, frame, proj, params, zonal, false, &Pose { turn: t, ..Default::default() }))
        }).collect()
    };
    let (best_t, best_v) = scan.iter().copied().fold((centre, f64::MIN), |a, b| if b.1 > a.1 { b } else { a });
    // Fine search: turn, field rotation, centre.
    let eval = |p: &Pose| score(&fine, frame, proj, params, zonal, true, p);
    let start = Pose { turn: best_t, ..Default::default() };
    let px = 1.0 / r;
    let bound_shift = 0.03 * r + 1.5;
    let steps = [if fixed_turn { 0.0 } else { step / 2.0 }, 0.5f64.to_radians(), 0.5, 0.5];
    let bounds = [if fixed_turn { 0.0 } else { 2.0 * step }, 4f64.to_radians(), bound_shift, bound_shift];
    let (pose, value) = pattern_search(&eval, start, steps, bounds, start, [0.02 * px, 0.01f64.to_radians(), 0.02, 0.02]);
    // Confidence: how far the peak stands over the correlation a detail's width away on either
    // side (on the coarse planes: zonal bands alone, which don't turn, give a flat curve).
    let confidence = if fixed_turn {
        0.0
    } else {
        let away = 1.5 * sig_c / r;
        let at = |t: f64| score(&coarse, frame, proj, params, zonal, false, &Pose { turn: t, ..pose });
        let peak = at(pose.turn);
        (peak - at(pose.turn - away).max(at(pose.turn + away))).max(0.0)
    };
    if std::env::var_os("PS_DEROT_DEBUG").is_some() {
        eprintln!("measure: samples {}/{} scan best {:.2}° {best_v:.3} conf {confidence:.4} → fine {:.3}° spin {:.2}° shift ({:.2}, {:.2}) score {value:.3}", coarse.len(), fine.len(), best_t.to_degrees(), pose.turn.to_degrees(), pose.spin.to_degrees(), pose.dx, pose.dy);
    }
    Measure { pose, score: value, confidence }
}

/// Come è stata trovata la rotazione di ogni ripresa.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnSource {
    /// Dalle ore dichiarate (confermate dalle immagini, se misurate).
    Times,
    /// Dalle immagini (senza ore, o con ore corrette).
    Images,
    /// Né ore né dettaglio misurabile: le riprese sono solo allineate.
    None,
}

/// La soluzione per una ripresa.
#[derive(Clone, Copy, Debug)]
pub struct FrameFit {
    /// Il globo usato (dopo l'affinamento).
    pub globe: Globe,
    /// Rotazione dall'istante di riferimento a quello della ripresa, gradi (sistema principale;
    /// positiva = più tardi).
    pub turn_deg: f64,
    /// La stessa misurata sulle immagini, se misurata.
    pub image_turn_deg: Option<f64>,
    /// Quanto il dettaglio coincide con il riferimento dopo la derotazione (−1…1).
    pub score: f32,
    /// Quanto la misura è affidabile (0 = dettaglio insufficiente).
    pub confidence: f32,
    pub source: TurnSource,
    /// L'ora usata (dichiarata, corretta o stimata).
    pub time: Option<Moment>,
    /// Correzione dell'ora dichiarata suggerita dalle immagini, minuti.
    pub time_shift_min: Option<f64>,
    /// Peso nella somma.
    pub weight: f32,
}

/// La soluzione completa.
#[derive(Clone, Debug)]
pub struct Solution {
    pub frames: Vec<FrameFit>,
    /// La ripresa sulla cui griglia (dimensioni e globo) si disegna il risultato.
    pub grid: usize,
    /// L'istante a cui sono portate le riprese, se noto.
    pub moment: Option<Moment>,
    pub params: Params,
    pub body: Body,
    pub zonal: bool,
    /// Le osservazioni (ore corrette, verso trovato…), nella lingua corrente.
    pub notes: Vec<String>,
}

fn period(body: Body) -> Option<f64> {
    rotation_period(body, super::JupiterSystem::II)
}

/// Il nord e la specchiatura dalle lune (Giove, Saturno), se il riconoscimento le trova:
/// (il nord è verso `+v` del globo, l'immagine è specchiata).
fn moons_orientation(img: &Image, globe: &Globe, body: Body, when: &Moment) -> Option<(bool, bool)> {
    if !matches!(body, Body::Jupiter | Body::Saturn) {
        return None;
    }
    let id = crate::identify::identify(img, body, when, None);
    let o = id.orientation.filter(|_| id.orientation_found)?;
    // North on the image against our +v; west (where the surface goes as the planet turns)
    // against our +u, the way an unmirrored capture moves.
    let (x0, y0) = globe.to_pixel(0.0, 0.0);
    let along = |(dx, dy): (f64, f64), (x1, y1): (f32, f32)| dx * (x1 - x0) as f64 + dy * (y1 - y0) as f64;
    let north_is_v = along(o.to_image(0.0, 1.0), globe.to_pixel(0.0, 1.0)) > 0.0;
    let west_is_u = along(o.to_image(1.0, 0.0), globe.to_pixel(1.0, 0.0)) > 0.0;
    Some((north_is_v, !west_is_u))
}

/// Parametri del pianeta: periodo, Sistema I per l'equatore di Giove, inclinazione.
pub fn params_for(body: Body, tilt_deg: f64, mirrored: bool) -> Option<Params> {
    Some(Params {
        period_hours: period(body)?,
        equator_period_hours: (body == Body::Jupiter).then_some(JUPITER_SYSTEM_I_HOURS),
        tilt_deg,
        oblateness: oblateness(body),
        mirrored,
    })
}

/// The tilt to start from: settings, else the ephemeris at the first known time.
fn base_tilt(captures: &[Capture], settings: &Settings) -> f64 {
    if let Some(t) = settings.tilt_deg {
        return t;
    }
    captures.iter().find_map(|c| c.time.filter(Moment::valid)).map_or(0.0, |t| super::tilt(settings.body, &t))
}

/// Prepara tutte le riprese (in parallelo). `progress` riceve un messaggio per ripresa.
pub fn prepare_all(captures: &[Capture], settings: &Settings, progress: &(dyn Fn(usize) + Sync)) -> Result<Vec<Prepared>, Error> {
    period(settings.body).ok_or(Error::Body(settings.body))?;
    if captures.is_empty() {
        return Err(Error::Empty);
    }
    let tilt = base_tilt(captures, settings);
    let proj = Projection::new(tilt, oblateness(settings.body));
    let found: Vec<(Vec<f32>, Option<(Globe, Option<globe::LimbFit>)>)> = captures
        .par_iter()
        .map(|c| {
            let lum = globe::luminance(c.image);
            let g = locate_globe(c.image, &lum, c.globe, settings.body, &proj);
            progress(1);
            (lum, g)
        })
        .collect();
    if let Some(i) = found.iter().position(|f| f.1.is_none()) {
        return Err(Error::NoDisk(i));
    }
    let mut globes: Vec<(Globe, Option<globe::LimbFit>)> = found.iter().map(|f| f.1.expect("checked")).collect();
    // A common radius (same evening, same optics): the median of those measured.
    let mut radii: Vec<f32> = globes.iter().zip(captures).filter(|(_, c)| c.globe.is_none()).map(|(g, _)| g.0.radius).collect();
    if !radii.is_empty() {
        radii.sort_by(f32::total_cmp);
        let r = radii[radii.len() / 2];
        // Captures at another image scale (a different Barlow) keep their own radius.
        for (g, c) in globes.iter_mut().zip(captures) {
            if c.globe.is_none() && (g.0.radius / r - 1.0).abs() < 0.04 {
                g.0.radius = r;
            }
        }
    }
    // The light is the same in every capture of the evening: one slope for all.
    let own: Vec<Shading> = captures.par_iter().zip(&globes).map(|(c, g)| shading_of(c.image, &g.0, settings.body, &proj, None)).collect();
    let nc = own.iter().map(|s| s.slope.len()).max().unwrap_or(1);
    let slope: Vec<[f32; 2]> = (0..nc)
        .map(|c| {
            let mut a: Vec<f32> = own.iter().map(|s| s.slope[c.min(s.slope.len() - 1)][0]).collect();
            let mut b: Vec<f32> = own.iter().map(|s| s.slope[c.min(s.slope.len() - 1)][1]).collect();
            a.sort_by(f32::total_cmp);
            b.sort_by(f32::total_cmp);
            [a[a.len() / 2], b[b.len() / 2]]
        })
        .collect();
    let prepared: Vec<Option<Prepared>> = captures
        .par_iter()
        .zip(&found)
        .zip(&globes)
        .map(|((c, f), g)| {
            let shading = shading_of(c.image, &g.0, settings.body, &proj, Some(&slope));
            Some(finish(c.image, &f.0, g.0, g.1, settings.body, &proj, shading))
        })
        .collect();
    let prepared: Vec<Prepared> = prepared.into_iter().map(|p| p.expect("checked")).collect();
    if std::env::var_os("PS_DEROT_DEBUG").is_some() {
        for p in &prepared {
            eprintln!("prepared: {:?} limb rms {:?} sharp {:.4} noise {:.4}", p.globe, p.limb.map(|l| l.rms), p.sharpness, p.noise);
        }
    }
    Ok(prepared)
}

/// Trova la rotazione, il centro e la rotazione di campo di ogni ripresa e l'istante di
/// riferimento. `prepared` viene da [`prepare_all`] con le stesse riprese.
pub fn solve(captures: &[Capture], prepared: &[Prepared], settings: &Settings) -> Result<Solution, Error> {
    let body = settings.body;
    let n = captures.len();
    if n == 0 || prepared.len() != n {
        return Err(Error::Empty);
    }
    let period_h = period(body).ok_or(Error::Body(body))?;
    let omega = 360.0 / period_h; // degrees per hour
    let mut notes = Vec::new();
    let times: Vec<Option<Moment>> = captures.iter().map(|c| c.time.filter(|t| t.valid() && settings.use_times)).collect();
    let tilt = base_tilt(captures, settings);
    let quality: Vec<f32> = prepared.iter().map(|p| p.sharpness).collect();
    // A limb that fits much worse than the others': the globe is uncertain. Such a capture is
    // neither the reference nor summed (placed by hand, it counts again).
    let suspect: Vec<bool> = {
        let mut rms: Vec<f32> = prepared.iter().filter_map(|p| p.limb.map(|l| l.rms)).collect();
        rms.sort_by(f32::total_cmp);
        let median = rms.get(rms.len() / 2).copied().unwrap_or(0.0);
        prepared.iter().map(|p| rms.len() >= 3 && p.limb.is_some_and(|l| l.rms > (3.0 * median).max(1.5))).collect()
    };
    let quality: Vec<f32> = quality.iter().zip(&suspect).map(|(&q, &s)| if s { 0.0 } else { q }).collect();
    // Relative to the upper quartile: one odd capture can't flatten all the others.
    let qtop = {
        let mut q = quality.clone();
        q.sort_by(f32::total_cmp);
        q[(q.len() * 3 / 4).min(q.len() - 1)]
    };
    let weights: Vec<f32> = quality.iter().map(|&q| if settings.weigh_sharpness && qtop > 0.0 { (q / qtop).clamp(0.05, 1.0).powi(2) } else { 1.0 }).collect();

    // The frame the others are measured against: a sharp one near the middle of the series.
    let order_key = |i: usize| times[i].map(|t| t.julian());
    // The timed captures of the main group: the most within 150° of rotation of each other
    // (another evening's captures are left for later, they can't be the reference).
    let timed: Vec<usize> = {
        let mut all: Vec<usize> = (0..n).filter(|&i| times[i].is_some() && !suspect[i]).collect();
        all.sort_by(|&a, &b| order_key(a).unwrap().total_cmp(&order_key(b).unwrap()));
        let window = 150.0 / omega / 24.0;
        let mut best = (0usize, 0usize);
        for a in 0..all.len() {
            let mut b = a;
            while b + 1 < all.len() && order_key(all[b + 1]).unwrap() - order_key(all[a]).unwrap() <= window {
                b += 1;
            }
            if b + 1 - a > best.1 - best.0 {
                best = (a, b + 1);
            }
        }
        all[best.0..best.1].to_vec()
    };
    let reference = if timed.len() >= 2 {
        let (lo, hi) = timed.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &i| (lo.min(order_key(i).unwrap()), hi.max(order_key(i).unwrap())));
        let mid = 0.5 * (lo + hi);
        let half = 0.5 * (hi - lo);
        *timed
            .iter()
            .max_by(|&&a, &&b| {
                let s = |i: usize| quality[i] as f64 * (1.0 - 0.6 * ((order_key(i).unwrap() - mid).abs() / half.max(1e-9)));
                s(a).total_cmp(&s(b))
            })
            .expect("timed")
    } else {
        (0..n).max_by(|&a, &b| quality[a].total_cmp(&quality[b])).expect("frames")
    };

    // Tilt sign: north may be down on the image. Tried on the frame farthest in time (or any).
    let mut tilt = tilt;
    let base = params_for(body, tilt, false).ok_or(Error::Body(body))?;
    let hours_to = |i: usize| match (times[i], times[reference]) {
        (Some(a), Some(b)) => Some(hours_between(&a, &b)),
        _ => None,
    };
    let range_for = |i: usize| -> (f64, f64) {
        // Around what the times predict (either direction unless known), else ±50°.
        match (hours_to(i), settings.mirrored) {
            (Some(hh), Some(m)) => {
                let pred = omega * hh * if m { -1.0 } else { 1.0 };
                (pred.to_radians(), (3.0 + 0.3 * pred.abs()).to_radians())
            }
            (Some(hh), None) => (0.0, (omega * hh.abs() * 1.3 + 3.0).to_radians()),
            _ => (0.0, 50f64.to_radians()),
        }
    };
    // The moons, when found, tell where north is and whether the image is mirrored.
    let moons = times[reference].and_then(|t| moons_orientation(captures[reference].image, &prepared[reference].globe, body, &t));
    if let Some((north_is_v, _)) = moons
        && settings.tilt_deg.is_none()
        && !north_is_v
    {
        tilt = -tilt;
        if tilt.abs() > 1.0 {
            notes.push(tr!("Dalle lune: il nord è in basso nell'immagine (inclinazione {:+.1}°).", "From the moons: north is at the bottom of the image (tilt {:+.1}°).", tilt));
        }
    }
    if moons.is_none() && body != Body::Jupiter && tilt.abs() > 3.0 && settings.tilt_deg.is_none() && settings.measure && n > 1 {
        let probe = (0..n).filter(|&i| i != reference).max_by(|&a, &b| hours_to(a).unwrap_or(0.0).abs().total_cmp(&hours_to(b).unwrap_or(0.0).abs())).expect("n > 1");
        let (c, r) = range_for(probe);
        let try_sign = |t: f64| {
            let proj = Projection::new(t, base.oblateness);
            let p = Params { tilt_deg: t, ..base };
            measure(&prepared[reference], &prepared[probe], &proj, &p, body, settings.zonal, c, r, false).score
        };
        let (a, b) = rayon::join(|| try_sign(tilt), || try_sign(-tilt));
        if b > a + 0.002 {
            tilt = -tilt;
            notes.push(tr!("Il polo visibile è quello in basso nell'immagine (inclinazione {:+.1}°).", "The visible pole is at the bottom of the image (tilt {:+.1}°).", tilt));
        }
    }
    let params0 = Params { tilt_deg: tilt, ..base };
    let proj = Projection::new(tilt, params0.oblateness);

    // Measure every frame against the reference.
    let measured: Vec<Option<Measure>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if i == reference {
                return Some(Measure { pose: Pose::default(), score: 1.0, confidence: 1.0 });
            }
            if !settings.measure {
                // Only the centre and field rotation, at the turn the times give.
                let pred = hours_to(i).map(|hh| omega * hh * if settings.mirrored == Some(true) { -1.0 } else { 1.0 })?;
                return Some(measure(&prepared[reference], &prepared[i], &proj, &params0, body, settings.zonal, pred.to_radians(), 0.0, true));
            }
            let (c, r) = range_for(i);
            Some(measure(&prepared[reference], &prepared[i], &proj, &params0, body, settings.zonal, c, r, false))
        })
        .collect();
    let image_turn: Vec<Option<f64>> = measured.iter().map(|m| m.map(|m| m.pose.turn.to_degrees())).collect();
    let confident: Vec<bool> = (0..n).map(|i| i == reference || measured[i].is_some_and(|m| m.confidence > 0.03 && m.score > 0.15)).collect();

    // Combine with the times.
    let mut mirrored = settings.mirrored.unwrap_or(false);
    let mut time_shift: Vec<Option<f64>> = vec![None; n];
    let mut eff_turn = vec![0.0f64; n]; // relative to the reference frame, degrees
    let mut source = vec![TurnSource::None; n];
    let timed_confident: Vec<usize> = timed.iter().copied().filter(|&i| confident[i] && settings.measure).collect();
    let have_times = times[reference].is_some() && timed.len() >= 2;
    if have_times {
        let rel = |i: usize| hours_to(i).expect("timed") * omega;
        if settings.mirrored.is_none() {
            // The direction from the measured turns when there are any, else from the moons.
            let agree: f64 = timed_confident.iter().map(|&i| image_turn[i].unwrap_or(0.0) * rel(i)).sum();
            if agree.abs() > 1e-6 {
                mirrored = agree < 0.0;
            } else if let Some((_, m)) = moons {
                mirrored = m;
            }
            if mirrored {
                notes.push(crate::i18n::tl("Le immagini sono specchiate: la superficie gira da destra a sinistra.").to_string());
            }
        }
        let s = if mirrored { -1.0 } else { 1.0 };
        // Residuals of the measured turns from the times; their median is the reference's own error.
        let mut res: Vec<f64> = timed_confident.iter().map(|&i| image_turn[i].unwrap_or(0.0) - s * rel(i)).collect();
        res.sort_by(f64::total_cmp);
        let offset = if res.len() >= 3 { res[res.len() / 2] } else { 0.0 };
        let threshold = 1.0;
        let offset = if offset.abs() > threshold { offset } else { 0.0 };
        for i in 0..n {
            match times[i] {
                // The reference is where the turns are measured from.
                Some(_) if i == reference => source[i] = TurnSource::Times,
                Some(_) => {
                    let pred = s * rel(i) + offset;
                    let dev = image_turn[i].map(|m| m - pred);
                    if confident[i] && settings.measure && i != reference && dev.is_some_and(|d| d.abs() > threshold) {
                        eff_turn[i] = image_turn[i].expect("measured");
                        source[i] = TurnSource::Images;
                        time_shift[i] = Some((eff_turn[i] - s * rel(i)) * s / omega * 60.0 - offset * s / omega * 60.0);
                    } else {
                        eff_turn[i] = pred;
                        source[i] = TurnSource::Times;
                    }
                }
                None => {
                    if confident[i] {
                        eff_turn[i] = image_turn[i].unwrap_or(0.0);
                        source[i] = TurnSource::Images;
                    }
                }
            }
        }
        if offset != 0.0 {
            time_shift[reference] = Some(-offset * s / omega * 60.0);
        }
        for i in 0..n {
            if let Some(m) = time_shift[i].filter(|m| m.abs() > 0.05) {
                notes.push(tr!("L'ora della ripresa {} sembra sbagliata di {:+.1} min: corretta con la rotazione misurata.", "The time of capture {} seems off by {:+.1} min: corrected with the measured rotation.", i + 1, m));
            }
        }
    } else {
        for i in 0..n {
            if confident[i] {
                eff_turn[i] = image_turn[i].unwrap_or(0.0);
                source[i] = TurnSource::Images;
            }
        }
        if settings.measure && n > 1 {
            notes.push(crate::i18n::tl("Senza le ore delle riprese la rotazione è misurata sulle immagini.").to_string());
        }
    }
    let unmeasured: Vec<String> = (0..n).filter(|&i| i != reference && source[i] == TurnSource::None).map(|i| (i + 1).to_string()).collect();
    match unmeasured.len() {
        0 => {}
        1 => notes.push(tr!("Ripresa {}: rotazione non misurabile (poco dettaglio), solo allineata.", "Capture {}: rotation not measurable (too little detail), only aligned.", unmeasured[0])),
        _ => notes.push(tr!("Riprese {}: rotazione non misurabile (poco dettaglio), solo allineate. Con le ore delle riprese verrebbero derotate.", "Captures {}: rotation not measurable (too little detail), only aligned. With the capture times they would be derotated.", unmeasured.join(", "))),
    }
    let s = if mirrored { -1.0 } else { 1.0 };
    // Times: declared, corrected or estimated from the reference's.
    let ref_time = times[reference].map(|t| Moment::from_julian(t.julian() + time_shift[reference].unwrap_or(0.0) / 1440.0));
    let eff_time: Vec<Option<Moment>> = (0..n)
        .map(|i| match (ref_time, source[i]) {
            (Some(rt), TurnSource::Times | TurnSource::Images) => Some(Moment::from_julian(rt.julian() + eff_turn[i] * s / omega / 24.0)),
            _ => times[i],
        })
        .collect();
    // Captures from another evening (or hours away) can't be derotated with these: the largest
    // group within 150° of rotation is kept.
    let keep = {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| eff_turn[a].total_cmp(&eff_turn[b]));
        let mut best = (0usize, 0usize, 0.0f32);
        for a in 0..n {
            let mut b = a;
            while b + 1 < n && eff_turn[order[b + 1]] - eff_turn[order[a]] <= 150.0 {
                b += 1;
            }
            let total: f32 = order[a..=b].iter().map(|&i| weights[i]).sum();
            if b - a + 1 > best.1 - best.0 || (b - a + 1 == best.1 - best.0 && total > best.2) {
                best = (a, b + 1, total);
            }
        }
        let mut keep = vec![false; n];
        for &i in &order[best.0..best.1] {
            keep[i] = true;
        }
        keep
    };
    let mut weights = weights;
    for i in 0..n {
        if suspect[i] {
            weights[i] = 0.0;
            notes.push(tr!("Ripresa {}: il bordo del globo non si misura bene, esclusa (si può posare il globo a mano).", "Capture {}: the globe's edge can't be measured well, left out (the globe can be placed by hand).", i + 1));
        }
    }
    for i in 0..n {
        if !keep[i] {
            weights[i] = 0.0;
            notes.push(tr!("Ripresa {}: troppo lontana nel tempo dalle altre ({:+.0}° di rotazione), esclusa.", "Capture {}: too far in time from the others ({:+.0}° of rotation), left out.", i + 1, eff_turn[i]));
        }
    }
    // The reference instant, as a turn from the reference frame.
    let ref_turn = match settings.reference {
        Reference::Frame(j) if j < n && keep[j] => eff_turn[j],
        _ => {
            let total: f64 = weights.iter().map(|&w| w as f64).sum();
            (0..n).map(|i| eff_turn[i] * weights[i] as f64).sum::<f64>() / total.max(1e-9)
        }
    };
    let moment = match (settings.reference, ref_time) {
        (Reference::Frame(j), _) if j < n && keep[j] => eff_time[j],
        (_, Some(rt)) => Some(Moment::from_julian(rt.julian() + ref_turn * s / omega / 24.0)),
        _ => None,
    };
    let grid = match settings.reference {
        Reference::Frame(j) if j < n && keep[j] => j,
        _ => (0..n).filter(|&i| keep[i]).min_by(|&a, &b| (eff_turn[a] - ref_turn).abs().total_cmp(&(eff_turn[b] - ref_turn).abs())).expect("frames"),
    };
    // The tilt at the reference instant, with the sign found.
    let flipped = settings.tilt_deg.is_none() && tilt != base_tilt(captures, settings);
    let tilt = match (settings.tilt_deg, moment) {
        (None, Some(m)) => super::tilt(body, &m) * if flipped { -1.0 } else { 1.0 },
        _ => tilt,
    };
    let params = Params { tilt_deg: tilt, mirrored, ..params0 };

    // Final poses: where the times gave a turn different from the measured one, refine the
    // centre and field rotation again at that turn.
    let frames: Vec<FrameFit> = (0..n)
        .into_par_iter()
        .map(|i| {
            let m = measured[i].expect("measured or reference");
            let mut pose = m.pose;
            let mut score = m.score;
            if i != reference && (pose.turn.to_degrees() - eff_turn[i]).abs() > 0.01 {
                let again = measure(&prepared[reference], &prepared[i], &proj, &params, body, settings.zonal, eff_turn[i].to_radians(), 0.0, true);
                pose = again.pose;
                score = again.score;
            }
            let globe = Pose { turn: 0.0, ..pose }.apply(prepared[i].globe);
            FrameFit {
                globe,
                turn_deg: eff_turn[i] - ref_turn,
                image_turn_deg: (settings.measure && i != reference).then(|| image_turn[i].map(|t| t - ref_turn)).flatten(),
                score: score as f32,
                confidence: measured[i].map_or(0.0, |m| m.confidence as f32),
                source: if i == reference && have_times { TurnSource::Times } else if i == reference { TurnSource::Images } else { source[i] },
                time: eff_time[i],
                time_shift_min: time_shift[i],
                weight: weights[i],
            }
        })
        .collect();
    Ok(Solution { frames, grid, moment, params, body, zonal: settings.zonal, notes })
}

/// Where the point of the globe seen at (u, v) of the reference falls in frame `f`, and how well
/// it is seen there (μ).
fn locate(p: [f64; 3], lat: f64, fit: &FrameFit, proj: &Projection, params: &Params, zonal: bool) -> Option<((f32, f32), f64)> {
    let q = turn(p, zonal_turn(params, zonal, lat, fit.turn_deg.to_radians()));
    let ((u, v), mu) = proj.to_plane(q);
    (mu > 0.0).then(|| (fit.globe.to_pixel(u, v), mu))
}

/// La somma derotata, sulla griglia della ripresa `solution.grid` (a `scale` per un'anteprima).
/// `clip`: sigma-clipping sul globo (da 3 riprese).
pub fn render(captures: &[Capture], prepared: &[Prepared], solution: &Solution, scale: f32, clip: Option<f32>) -> Option<Image> {
    let grid = solution.grid;
    let base = captures.get(grid)?.image;
    let nc = base.channels().len();
    let scale = scale.clamp(0.01, 1.0);
    let (w, h) = (((base.width() as f32 * scale).round() as usize).max(1), ((base.height() as f32 * scale).round() as usize).max(1));
    let params = &solution.params;
    let proj = Projection::new(params.tilt_deg, params.oblateness);
    let out_globe = solution.frames[grid].globe;
    let shading = Shading::mean(&prepared.iter().zip(&solution.frames).map(|(p, f)| (&p.shading, f.weight)).collect::<Vec<_>>())?;
    let rings = solution.body == Body::Saturn;
    let r_px = out_globe.radius as f64;
    let mut out = vec![0.0f32; w * h * nc];
    out.par_chunks_mut(w * nc).enumerate().for_each(|(j, row)| {
        let mut samples: Vec<([f32; 3], f32)> = Vec::with_capacity(captures.len());
        for i in 0..w {
            let (x, y) = ((i as f32 + 0.5) / scale, (j as f32 + 0.5) / scale);
            let (u, v) = out_globe.to_plane(x, y);
            let rho = proj.rho(u, v);
            let ring = rings && proj.ring_radius(u, v).is_some_and(|r| r > globe::SATURN_RINGS.0 && r < globe::SATURN_RINGS.1);
            // How much of the pixel is globe (a pixel-wide edge).
            let on_globe = if ring { 0.0 } else { ((1.0 - rho) * r_px / 1.5).clamp(0.0, 1.0) as f32 };
            let mut globe_value = [0.0f32; 3];
            let mut have_globe = false;
            if on_globe > 0.0
                && let Some(p) = proj.to_globe(u, v.clamp(-proj.kp * 0.999_999, proj.kp * 0.999_999))
            {
                let (_, lat) = lon_lat(p, proj.k);
                samples.clear();
                for ((cap, prep), fit) in captures.iter().zip(prepared).zip(&solution.frames) {
                    if fit.weight <= 0.0 {
                        continue;
                    }
                    let Some(((px, py), mu)) = locate(p, lat, fit, &proj, params, solution.zonal) else { continue };
                    let vis = smoothstep(mu / 0.08) * (mu * mu) as f32 + 1e-6;
                    let mut val = [0.0f32; 3];
                    for (c, slot) in val.iter_mut().enumerate().take(nc) {
                        *slot = catmull_rom(&prep.detail[c.min(prep.detail.len() - 1)], cap.image.width(), cap.image.height(), px, py);
                    }
                    samples.push((val, fit.weight * vis));
                }
                if !samples.is_empty() {
                    have_globe = true;
                    for c in 0..nc {
                        globe_value[c] = combine(&samples, c, clip) * shading.at(c, proj.to_plane(p).1.max(0.0), u, v);
                    }
                }
            }
            // Off the globe (and the rings): only aligned; moons and anything that moves stay
            // as in the grid frame.
            let mut rigid = [0.0f32; 3];
            if on_globe < 1.0 || !have_globe {
                let anchor: Vec<f32> = {
                    let (px, py) = solution.frames[grid].globe.to_pixel(u, v);
                    let img = captures[grid].image;
                    (0..nc).map(|c| catmull_rom(&img.channels()[c], img.width(), img.height(), px, py)).collect()
                };
                for c in 0..nc {
                    let (mut sw, mut sv) = (0.0f32, 0.0f32);
                    for ((cap, prep), fit) in captures.iter().zip(prepared).zip(&solution.frames) {
                        if fit.weight <= 0.0 {
                            continue;
                        }
                        let (px, py) = fit.globe.to_pixel(u, v);
                        let img = cap.image;
                        let val = catmull_rom(&img.channels()[c.min(img.channels().len() - 1)], img.width(), img.height(), px, py);
                        if (val - anchor[c]).abs() <= 4.0 * prep.noise + 0.08 * anchor[c].abs() {
                            sw += fit.weight;
                            sv += fit.weight * val;
                        }
                    }
                    rigid[c] = if sw > 0.0 { sv / sw } else { anchor[c] };
                }
            }
            let t = if have_globe { on_globe } else { 0.0 };
            for c in 0..nc {
                row[i * nc + c] = t * globe_value[c] + (1.0 - t) * rigid[c];
            }
        }
    });
    let channels = (0..nc).map(|c| out.iter().skip(c).step_by(nc).copied().collect()).collect();
    Some(Image::new(w, h, channels, base.source_bit_depth))
}

/// Una sola ripresa derotata all'istante di riferimento (per confronti e per l'interfaccia).
pub fn render_one(captures: &[Capture], prepared: &[Prepared], solution: &Solution, index: usize, scale: f32) -> Option<Image> {
    let mut only = solution.clone();
    for (i, f) in only.frames.iter_mut().enumerate() {
        f.weight = if i == index { 1.0 } else { 1e-9 };
    }
    render(captures, prepared, &only, scale, None)
}

/// La mappa cilindrica del pianeta dalle riprese derotate: `width` colonne per 360° (e metà
/// righe per 180°). Le longitudini mai viste restano con copertura 0.
pub fn planet_map(captures: &[Capture], prepared: &[Prepared], solution: &Solution, width: usize) -> Option<PlanetMap> {
    let width = width.clamp(64, 8192) & !1;
    let height = width / 2;
    let params = &solution.params;
    let proj = Projection::new(params.tilt_deg, params.oblateness);
    let nc = captures.first()?.image.channels().len();
    let k = proj.k;
    let mut data = vec![0.0f32; width * height * (nc + 1)];
    data.par_chunks_mut(width * (nc + 1)).enumerate().for_each(|(j, row)| {
        let lat = PI / 2.0 - PI * (j as f64 + 0.5) / height as f64;
        for i in 0..width {
            let alpha = -PI + 2.0 * PI * (i as f64 + 0.5) / width as f64;
            let p = point(alpha, lat, k);
            let (mut sw, mut sv) = (0.0f32, [0.0f32; 3]);
            for ((cap, prep), fit) in captures.iter().zip(prepared).zip(&solution.frames) {
                let Some(((px, py), mu)) = locate(p, lat, fit, &proj, params, solution.zonal) else { continue };
                if mu < 0.05 || fit.weight <= 0.0 {
                    continue;
                }
                let wt = fit.weight * smoothstep(mu / 0.15) * (mu * mu) as f32;
                for c in 0..nc {
                    sv[c] += wt * catmull_rom(&prep.detail[c.min(prep.detail.len() - 1)], cap.image.width(), cap.image.height(), px, py);
                }
                sw += wt;
            }
            let o = i * (nc + 1);
            for c in 0..nc {
                row[o + c] = if sw > 0.0 { sv[c] / sw } else { 0.0 };
            }
            row[o + nc] = sw;
        }
    });
    let channels: Vec<Vec<f32>> = (0..nc).map(|c| data.iter().skip(c).step_by(nc + 1).copied().collect()).collect();
    let coverage: Vec<f32> = data.iter().skip(nc).step_by(nc + 1).copied().collect();
    let zonal = globe::zonal_means(&channels, &coverage, width, height);
    let shading = Shading::mean(&prepared.iter().zip(&solution.frames).map(|(p, f)| (&p.shading, f.weight)).collect::<Vec<_>>())?;
    let central_longitude = match solution.moment {
        Some(m) if params.equator_period_hours.is_some() => Some(crate::moons::jupiter_central_meridian(&m).1),
        _ => None,
    };
    Some(PlanetMap {
        width,
        height,
        channels,
        coverage,
        zonal,
        shading,
        oblateness: params.oblateness,
        tilt_deg: params.tilt_deg,
        period_hours: params.period_hours,
        central_longitude,
        moment: solution.moment,
        mirrored: params.mirrored,
    })
}

/// Tutto in una volta: prepara, risolve, somma a piena risoluzione.
pub fn derotate(captures: &[Capture], settings: &Settings) -> Result<(Image, Solution), Error> {
    let prepared = prepare_all(captures, settings, &|_| {})?;
    let solution = solve(captures, &prepared, settings)?;
    let image = render(captures, &prepared, &solution, 1.0, settings.clip).ok_or(Error::Empty)?;
    Ok((image, solution))
}

/// L'istante medio delle ore note (per chi non usa [`solve`]).
pub fn mean_time(captures: &[Capture]) -> Option<Moment> {
    mean_moment(&captures.iter().filter_map(|c| c.time.filter(Moment::valid).map(|t| (t, 1.0f32))).collect::<Vec<_>>())
}
