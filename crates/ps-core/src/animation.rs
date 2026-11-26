//! Animazione da poche riprese: le immagini della stessa sera, ordinate per ora, allineate sul
//! disco, con la luminosità uniformata, diventano una sequenza fluida di molti fotogrammi
//! (50–60 al secondo) interpolando fra una ripresa e la successiva.
//!
//! Tre modi di interpolare:
//! - **Globo** (Giove, Saturno, con l'ora di ogni ripresa): lo sfasamento fra due riprese è una
//!   rotazione nota dal periodo, quindi il fotogramma all'istante `t` si ottiene girando le due
//!   riprese vicine fino a `t` (stessa proiezione della derotazione) e mescolandole. I dettagli
//!   si muovono davvero, invece di dissolversi.
//! - **Flusso** (Luna, Sole, o senza orari): flusso ottico denso (Horn–Schunck a più scale)
//!   fra riprese vicine; i pixel si spostano lungo il flusso e le due immagini si mescolano.
//! - **Dissolvenza**: solo la miscela, il minimo.
//!
//! Tutto il lavoro geometrico (allineamento sul disco, ritaglio, ridimensionamento, guadagno di
//! luminosità) passa da [`crate::derotation::stack`], così l'animazione e la derotazione
//! condividono la stessa proiezione. L'uscita sono [`Image`] in virgola mobile: l'esportatore
//! (`ps_io::video`) prende fotogrammi RGB a 8 bit qualsiasi, anche quelli di un altro
//! generatore (il globo 3D che ruota).

use crate::analysis::{Analysis, ImageKind};
use crate::capture::{Body, Moment};
use crate::derotation::{self, Frame, Globe, JupiterSystem, Params};
use crate::image::{Image, bilinear_clamped};
use rayon::prelude::*;
use std::sync::Arc;

/// Come interpolare fra le riprese.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Il globo se è Giove o Saturno con gli orari, altrimenti il flusso ottico.
    #[default]
    Auto,
    Globe,
    Flow,
    Fade,
}

/// Una ripresa.
#[derive(Clone)]
pub struct Shot {
    pub image: Arc<Image>,
    pub time: Option<Moment>,
    /// Il globo (centro, raggio, direzione dell'equatore), se l'analisi lo ha trovato.
    pub globe: Option<Globe>,
    /// Mezza estensione del soggetto in pixel (con gli anelli di Saturno è più del raggio).
    pub extent: f32,
    pub means: [f32; 3],
}

impl Shot {
    /// Analizza l'immagine per trovarne il disco.
    pub fn from_image(image: Arc<Image>, time: Option<Moment>) -> Shot {
        let analysis = crate::analysis::analyze(&image);
        Shot::from_analysis(image, time, &analysis)
    }

    /// Come [`Shot::from_image`] con l'analisi già fatta.
    pub fn from_analysis(image: Arc<Image>, time: Option<Moment>, analysis: &Analysis) -> Shot {
        let globe = derotation::find_globe(&image, analysis);
        let extent = match (&globe, analysis.kind) {
            (Some(g), ImageKind::Saturn) => (analysis.subject_size / 2.0).max(g.radius),
            (Some(g), _) => g.radius,
            _ => 0.0,
        };
        let means = globe.as_ref().map_or([0.0; 3], |g| derotation::disk_means(&image, g));
        Shot { image, time, globe, extent, means }
    }
}

/// Le scelte dell'utente.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub interpolation: Interpolation,
    pub body: Option<Body>,
    pub system: JupiterSystem,
    /// Verso di rotazione: `None` = si sceglie quello che fa coincidere meglio le riprese.
    pub mirrored: Option<bool>,
    /// Durata di tutto il ciclo, secondi.
    pub seconds: f32,
    pub fps: f32,
    /// Avanti e indietro invece di ricominciare dal primo.
    pub ping_pong: bool,
    /// Ritaglia attorno al pianeta (altrimenti il fotogramma intero).
    pub crop: bool,
    /// Lato maggiore dell'uscita in pixel (0 = la dimensione originale).
    pub size: usize,
    /// Uniforma la luminosità del disco fra le riprese.
    pub match_brightness: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { interpolation: Interpolation::Auto, body: None, system: JupiterSystem::II, mirrored: None, seconds: 6.0, fps: 50.0, ping_pong: true, crop: true, size: 600, match_brightness: true }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Servono almeno due riprese utilizzabili.
    TooFew,
    /// Le riprese non hanno la stessa dimensione.
    Sizes,
    /// Il modo "globo" non si può usare: non è Giove o Saturno, mancano gli orari, o il disco.
    NotGlobe,
}

/// Tutto quanto serve a disegnare un fotogramma qualsiasi, già preparato.
pub struct Plan {
    /// Le riprese in ordine di tempo.
    shots: Vec<Shot>,
    /// Il modo usato davvero.
    mode: Interpolation,
    /// Le posizioni delle riprese sulla linea del tempo (ore dalla prima in modo Globo,
    /// altrimenti 0, 1, 2…).
    timeline: Vec<f64>,
    /// Le riprese, con globo corretto dal riallineamento (modi Globo e, a zero ore, gli altri).
    frames: Vec<FrameData>,
    reference: Globe,
    params: Params,
    window: (usize, usize),
    scale: f32,
    /// Le riprese già allineate e ritagliate all'uscita (modi Flusso e Dissolvenza).
    aligned: Vec<Image>,
    flows: Vec<Flow>,
    total: usize,
    ping_pong: bool,
    mirrored: Option<bool>,
}

#[derive(Clone)]
struct FrameData {
    globe: Globe,
    gain: [f32; 3],
}

impl Plan {
    pub fn new(shots: &[Shot], opts: &Options) -> Result<Plan, Error> {
        // In order of time (those without one keep their place after the others).
        let mut shots: Vec<Shot> = shots.to_vec();
        shots.sort_by(|a, b| match (a.time, b.time) {
            (Some(x), Some(y)) => x.julian().total_cmp(&y.julian()),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });
        if shots.len() < 2 {
            return Err(Error::TooFew);
        }
        let size0 = (shots[0].image.width(), shots[0].image.height());
        if shots.iter().any(|s| (s.image.width(), s.image.height()) != size0) {
            return Err(Error::Sizes);
        }

        let all_times = shots.iter().all(|s| s.time.is_some_and(|t| t.valid()));
        let all_globes = shots.iter().all(|s| s.globe.is_some());
        let params_base = opts.body.and_then(|b| Params::for_body(b, opts.system, shots[0].time.as_ref()));
        let globe_ok = params_base.is_some() && all_times && all_globes;
        let mode = match opts.interpolation {
            Interpolation::Auto => {
                if globe_ok {
                    Interpolation::Globe
                } else {
                    Interpolation::Flow
                }
            }
            Interpolation::Globe if !globe_ok => return Err(Error::NotGlobe),
            m => m,
        };

        // Brightness factors and the common geometry: radius and direction of the equator are
        // the median of those measured; a shot without a globe is centred on the frame.
        let median = |mut v: Vec<f32>, fallback: f32| {
            v.sort_by(f32::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(fallback)
        };
        let fallback_radius = size0.0.min(size0.1) as f32 / 3.0;
        let radius = median(shots.iter().filter_map(|s| s.globe.map(|g| g.radius)).collect(), fallback_radius);
        let angle = median(shots.iter().filter_map(|s| s.globe.map(|g| g.angle)).collect(), 0.0);
        let ratio = median(shots.iter().filter_map(|s| s.globe.map(|g| s.extent / g.radius.max(1.0))).collect(), 1.0).max(1.0);
        let mut globes: Vec<Globe> = shots.iter().map(|s| s.globe.map_or(Globe { cx: size0.0 as f32 / 2.0, cy: size0.1 as f32 / 2.0, radius, angle }, |g| Globe { angle, ..g })).collect();

        let mid = shots.len() / 2;
        let means: Vec<[f32; 3]> = shots.iter().map(|s| s.means).collect();
        let gains = if opts.match_brightness && shots.iter().all(|s| s.globe.is_some()) { derotation::gains(&means, mid) } else { vec![[1.0; 3]; shots.len()] };

        // The window on the reference grid and the scale to the output.
        let (window, reference) = if opts.crop {
            let half = (ratio * radius * 1.12).max(8.0);
            let side = (2.0 * half).ceil() as usize;
            ((side, side), Globe { cx: side as f32 / 2.0, cy: side as f32 / 2.0, radius, angle })
        } else {
            (size0, Globe { radius, angle, ..globes[0] })
        };
        // Never above 1: `stack` samples at most at the reference grid's resolution.
        let scale = if opts.size == 0 { 1.0 } else { (opts.size as f32 / window.0.max(window.1) as f32).clamp(0.01, 1.0) };

        // Time line: real hours for the globe, else the order.
        let hours_from_first: Vec<f64> = match shots[0].time {
            Some(t0) if all_times => shots.iter().map(|s| derotation::hours_between(&s.time.expect("checked"), &t0)).collect(),
            _ => (0..shots.len()).map(|i| i as f64).collect(),
        };
        let timeline = if mode == Interpolation::Globe { hours_from_first.clone() } else { (0..shots.len()).map(|i| i as f64).collect() };

        // Turning direction and the centres refined on the markings (globe) or the disc.
        let ref_hours: Vec<f64> = if mode == Interpolation::Globe { hours_from_first.iter().map(|h| h - hours_from_first[mid]).collect() } else { vec![0.0; shots.len()] };
        let mut params = params_base.unwrap_or(Params { period_hours: 24.0, equator_period_hours: None, tilt_deg: 0.0, oblateness: 1.0, mirrored: false });
        if mode != Interpolation::Globe {
            // Without a planet's rotation the frames are only registered, never turned.
            params.oblateness = 1.0;
        }
        let frames_for = |gl: &[Globe]| -> Vec<Frame> { shots.iter().enumerate().map(|(i, s)| Frame { image: &s.image, globe: gl[i], hours: ref_hours[i], weight: 1.0, gain: gains[i] }).collect() };
        let refine = |mirrored: bool, gl: &[Globe]| -> (Vec<Globe>, f32) {
            let p = Params { mirrored, ..params };
            let mut frames = frames_for(gl);
            derotation::refine_centres(&mut frames, mid, &reference, &p, window);
            // How well the derotated frames agree with the middle one (smaller is better).
            let single = |f: &Frame| derotation::stack(std::slice::from_ref(f), &reference, &p, window, 0.5, None);
            let spread = match single(&frames[mid]) {
                Some(target) => {
                    let t = &target.channels()[0];
                    frames
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != mid)
                        .filter_map(|(_, f)| single(f))
                        .map(|im| im.channels()[0].iter().zip(t).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / t.len() as f32)
                        .sum::<f32>()
                }
                None => f32::MAX,
            };
            (frames.iter().map(|f| f.globe).collect(), spread)
        };
        let mut used_mirrored = None;
        if mode == Interpolation::Globe {
            let (g, m) = match opts.mirrored {
                Some(m) => (refine(m, &globes).0, m),
                None => {
                    let (a, b) = rayon::join(|| refine(false, &globes), || refine(true, &globes));
                    if b.1 < a.1 { (b.0, true) } else { (a.0, false) }
                }
            };
            globes = g;
            used_mirrored = Some(m);
            params.mirrored = m;
        } else {
            globes = refine(false, &globes).0;
        }

        let frames: Vec<FrameData> = globes.iter().zip(&gains).map(|(&globe, &gain)| FrameData { globe, gain }).collect();
        let mut plan = Plan { shots, mode, timeline, frames, reference, params, window, scale, aligned: Vec::new(), flows: Vec::new(), total: 0, ping_pong: opts.ping_pong, mirrored: used_mirrored };

        // The cycle's length in frames.
        let pass = (opts.seconds.max(0.2) * opts.fps.clamp(1.0, 120.0)).round().max(2.0) as usize;
        plan.total = if opts.ping_pong { (pass + 2).div_ceil(2) * 2 - 2 } else { pass };

        if mode != Interpolation::Globe {
            let still = Params { period_hours: 1e9, ..plan.params };
            plan.aligned = (0..plan.shots.len())
                .map(|i| {
                    let f = Frame { image: &plan.shots[i].image, globe: plan.frames[i].globe, hours: 0.0, weight: 1.0, gain: plan.frames[i].gain };
                    derotation::stack(&[f], &plan.reference, &still, plan.window, plan.scale, None).ok_or(Error::TooFew)
                })
                .collect::<Result<_, _>>()?;
            if mode == Interpolation::Flow {
                plan.flows = plan.aligned.windows(2).collect::<Vec<_>>().par_iter().map(|w| Flow::between(&w[0], &w[1])).collect();
            }
        }
        Ok(plan)
    }

    /// Fotogrammi in tutto (un ciclo).
    pub fn frames(&self) -> usize {
        self.total
    }

    /// Dimensione dei fotogrammi.
    pub fn size(&self) -> (usize, usize) {
        ((self.window.0 as f32 * self.scale).round().max(1.0) as usize, (self.window.1 as f32 * self.scale).round().max(1.0) as usize)
    }

    pub fn mode(&self) -> Interpolation {
        self.mode
    }

    /// Il verso di rotazione usato (solo in modo Globo).
    pub fn mirrored(&self) -> Option<bool> {
        self.mirrored
    }

    /// Le riprese in ordine di tempo.
    pub fn shots(&self) -> &[Shot] {
        &self.shots
    }

    /// Minuti fra la prima e l'ultima ripresa, se gli orari ci sono.
    pub fn span_minutes(&self) -> Option<f64> {
        let (a, b) = (self.shots.first()?.time?, self.shots.last()?.time?);
        Some(derotation::hours_between(&b, &a) * 60.0)
    }

    /// Posizione 0‥1 del fotogramma `k` lungo l'andata.
    fn position(&self, k: usize) -> f64 {
        let k = k.min(self.total.saturating_sub(1));
        if self.ping_pong {
            let half = self.total / 2; // forward frames 0..=half, back half+1..total
            let i = if k <= half { k } else { 2 * half - k };
            i as f64 / half as f64
        } else {
            k as f64 / (self.total.max(2) - 1) as f64
        }
    }

    /// (ripresa precedente, ripresa successiva, peso della successiva) alla posizione `p`.
    fn locate(&self, p: f64) -> (usize, usize, f32, f64) {
        let t = self.timeline[0] + p * (self.timeline[self.timeline.len() - 1] - self.timeline[0]);
        let n = self.timeline.len();
        let b = (1..n).find(|&i| self.timeline[i] >= t).unwrap_or(n - 1);
        let a = b - 1;
        let span = self.timeline[b] - self.timeline[a];
        let s = if span > 1e-12 { ((t - self.timeline[a]) / span).clamp(0.0, 1.0) as f32 } else { 0.0 };
        (a, b, s, t)
    }

    /// L'istante del fotogramma `k`, se le riprese hanno l'ora.
    pub fn moment(&self, k: usize) -> Option<Moment> {
        let (t0, tn) = (self.shots.first()?.time?, self.shots.last()?.time?);
        let p = self.position(k);
        // The line of the globe is in real hours; the others run through the shots in order.
        let (a, b, s, t) = self.locate(p);
        let jd = if self.mode == Interpolation::Globe {
            t0.julian() + t / 24.0
        } else {
            let (ta, tb) = (self.shots[a].time?.julian(), self.shots[b].time?.julian());
            ta + (tb - ta) * s as f64
        };
        let _ = tn;
        Some(Moment::from_julian(jd))
    }

    /// Il fotogramma `k` (0 ≤ k < [`Plan::frames`]).
    pub fn render(&self, k: usize) -> Image {
        let (a, b, s, t) = self.locate(self.position(k));
        match self.mode {
            Interpolation::Globe => {
                let make = |i: usize, weight: f32| Frame { image: &self.shots[i].image, globe: self.frames[i].globe, hours: self.timeline[i] - t, weight, gain: self.frames[i].gain };
                let pair = [make(a, 1.0 - s), make(b, s)];
                derotation::stack(&pair, &self.reference, &self.params, self.window, self.scale, None).unwrap_or_else(|| self.aligned_or_blank())
            }
            Interpolation::Flow => blend_flow(&self.aligned[a], &self.aligned[b], &self.flows[a], s),
            _ => blend(&self.aligned[a], &self.aligned[b], s),
        }
    }

    fn aligned_or_blank(&self) -> Image {
        let (w, h) = self.size();
        Image::new(w, h, vec![vec![0.0; w * h]; 3], 16)
    }
}

/// `(1 − s)·a + s·b`.
fn blend(a: &Image, b: &Image, s: f32) -> Image {
    let channels = a.channels().iter().zip(b.channels()).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p + (q - p) * s).collect()).collect();
    Image::new(a.width(), a.height(), channels, a.source_bit_depth)
}

// ───────────────────────────── flusso ottico ─────────────────────────────

/// Movimento denso da A a B: un punto in `p` di A è in `p + (u, v)` in B.
struct Flow {
    u: Vec<f32>,
    v: Vec<f32>,
}

impl Flow {
    fn between(a: &Image, b: &Image) -> Flow {
        let (w, h) = (a.width(), a.height());
        let lum = |im: &Image| -> Vec<f32> {
            let c = im.channels();
            (0..w * h).map(|i| 255.0 * c.iter().map(|p| p[i]).sum::<f32>() / c.len() as f32).collect()
        };
        estimate_flow(&lum(a), &lum(b), w, h)
    }
}

/// Horn–Schunck on a pyramid, with warping at each level.
fn estimate_flow(a: &[f32], b: &[f32], w: usize, h: usize) -> Flow {
    let mut pyramid = vec![(a.to_vec(), b.to_vec(), w, h)];
    while pyramid.len() < 5 {
        let (pa, pb, pw, ph) = pyramid.last().expect("not empty");
        if (*pw).min(*ph) < 40 {
            break;
        }
        let (na, nb) = (crate::image::downscale_plane(pa, *pw, *ph, 2), crate::image::downscale_plane(pb, *pw, *ph, 2));
        let (nw, nh) = (*pw / 2, *ph / 2);
        pyramid.push((na, nb, nw, nh));
    }
    let (mut u, mut v): (Vec<f32>, Vec<f32>) = {
        let (_, _, cw, ch) = pyramid.last().expect("not empty");
        (vec![0.0; cw * ch], vec![0.0; cw * ch])
    };
    let (mut fw, mut fh) = {
        let (_, _, cw, ch) = pyramid.last().expect("not empty");
        (*cw, *ch)
    };
    for (level, (la, lb, lw, lh)) in pyramid.iter().enumerate().rev() {
        let (lw, lh) = (*lw, *lh);
        if (lw, lh) != (fw, fh) {
            // Up from the coarser level: sizes double (the odd row or column is repeated).
            let up = |p: &[f32]| -> Vec<f32> { (0..lw * lh).map(|i| bilinear_clamped(p, fw, fh, ((i % lw) as f32 + 0.5) * fw as f32 / lw as f32, ((i / lw) as f32 + 0.5) * fh as f32 / lh as f32) * (lw as f32 / fw as f32)).collect() };
            u = up(&u);
            v = up(&v);
        }
        (fw, fh) = (lw, lh);
        let alpha2 = 100.0f32;
        for _ in 0..if level == 0 { 2 } else { 3 } {
            // B brought back by the flow so far; what is left is a small step.
            let warped: Vec<f32> = (0..lw * lh).map(|i| bilinear_clamped(lb, lw, lh, (i % lw) as f32 + 0.5 + u[i], (i / lw) as f32 + 0.5 + v[i])).collect();
            let at = |p: &[f32], x: isize, y: isize| p[y.clamp(0, lh as isize - 1) as usize * lw + x.clamp(0, lw as isize - 1) as usize];
            let (mut ix, mut iy, mut it) = (vec![0.0f32; lw * lh], vec![0.0f32; lw * lh], vec![0.0f32; lw * lh]);
            for y in 0..lh {
                for x in 0..lw {
                    let (xi, yi) = (x as isize, y as isize);
                    let k = y * lw + x;
                    ix[k] = 0.25 * (at(la, xi + 1, yi) - at(la, xi - 1, yi) + at(&warped, xi + 1, yi) - at(&warped, xi - 1, yi));
                    iy[k] = 0.25 * (at(la, xi, yi + 1) - at(la, xi, yi - 1) + at(&warped, xi, yi + 1) - at(&warped, xi, yi - 1));
                    it[k] = warped[k] - la[k];
                }
            }
            let (mut du, mut dv) = (vec![0.0f32; lw * lh], vec![0.0f32; lw * lh]);
            for _ in 0..40 {
                let (mut nu, mut nv) = (vec![0.0f32; lw * lh], vec![0.0f32; lw * lh]);
                nu.par_chunks_mut(lw).zip(nv.par_chunks_mut(lw)).enumerate().for_each(|(y, (ru, rv))| {
                    for x in 0..lw {
                        let (xi, yi) = (x as isize, y as isize);
                        let mean = |p: &[f32]| 0.25 * (at(p, xi - 1, yi) + at(p, xi + 1, yi) + at(p, xi, yi - 1) + at(p, xi, yi + 1));
                        let (ub, vb) = (mean(&du), mean(&dv));
                        let k = y * lw + x;
                        let common = (ix[k] * ub + iy[k] * vb + it[k]) / (alpha2 + ix[k] * ix[k] + iy[k] * iy[k]);
                        ru[x] = ub - ix[k] * common;
                        rv[x] = vb - iy[k] * common;
                    }
                });
                du = nu;
                dv = nv;
                // A step of more than a pixel and a half is the linearisation failing.
                du.iter_mut().chain(dv.iter_mut()).for_each(|d| *d = d.clamp(-1.5, 1.5));
            }
            for i in 0..lw * lh {
                u[i] += du[i];
                v[i] += dv[i];
            }
        }
    }
    Flow { u, v }
}

/// The flow seen from the instant `s` between the frames: each pixel of A carries its motion
/// to where it will be then (bright pixels count most, so the sky doesn't wash out the
/// planet's motion), and the gaps are filled from their neighbours.
fn flow_at(a: &Image, flow: &Flow, s: f32) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = (a.width(), a.height());
    let c = a.channels();
    let lum = |i: usize| c.iter().map(|p| p[i]).sum::<f32>() / c.len() as f32;
    let (mut su, mut sv, mut sw) = (vec![0.0f32; w * h], vec![0.0f32; w * h], vec![0.0f32; w * h]);
    for i in 0..w * h {
        let (x, y) = ((i % w) as f32 + s * flow.u[i], (i / w) as f32 + s * flow.v[i]);
        let weight = 0.002 + lum(i).max(0.0);
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        for (dx, dy, k) in [(0, 0, (1.0 - tx) * (1.0 - ty)), (1, 0, tx * (1.0 - ty)), (0, 1, (1.0 - tx) * ty), (1, 1, tx * ty)] {
            let (px, py) = (x0 as isize + dx, y0 as isize + dy);
            if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                let j = py as usize * w + px as usize;
                su[j] += weight * k * flow.u[i];
                sv[j] += weight * k * flow.v[i];
                sw[j] += weight * k;
            }
        }
    }
    let mut known: Vec<bool> = sw.iter().map(|&k| k > 1e-4).collect();
    for i in 0..w * h {
        if known[i] {
            su[i] /= sw[i];
            sv[i] /= sw[i];
        } else {
            su[i] = 0.0;
            sv[i] = 0.0;
        }
    }
    // Holes take the mean of the known neighbours, a ring at a time.
    for _ in 0..24 {
        if known.iter().all(|&k| k) {
            break;
        }
        let mut next = known.clone();
        let (mut nu, mut nv) = (su.clone(), sv.clone());
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if known[i] {
                    continue;
                }
                let (mut u, mut v, mut n) = (0.0, 0.0, 0.0);
                for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let (px, py) = (x as isize + dx, y as isize + dy);
                    if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h && known[py as usize * w + px as usize] {
                        let j = py as usize * w + px as usize;
                        u += su[j];
                        v += sv[j];
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    nu[i] = u / n;
                    nv[i] = v / n;
                    next[i] = true;
                }
            }
        }
        (su, sv, known) = (nu, nv, next);
    }
    (su, sv)
}

/// The frame at fraction `s` of the way from `a` to `b`: both are carried along the flow to
/// that instant and mixed.
fn blend_flow(a: &Image, b: &Image, flow: &Flow, s: f32) -> Image {
    if s <= 0.0 {
        return a.clone();
    }
    if s >= 1.0 {
        return b.clone();
    }
    let (w, h) = (a.width(), a.height());
    let (fu, fv) = flow_at(a, flow, s);
    let channels: Vec<Vec<f32>> = a
        .channels()
        .iter()
        .zip(b.channels())
        .map(|(pa, pb)| {
            (0..w * h)
                .into_par_iter()
                .map(|i| {
                    let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                    let (u, v) = (fu[i], fv[i]);
                    let from_a = bilinear_clamped(pa, w, h, x - s * u, y - s * v);
                    let from_b = bilinear_clamped(pb, w, h, x + (1.0 - s) * u, y + (1.0 - s) * v);
                    from_a + (from_b - from_a) * s
                })
                .collect()
        })
        .collect();
    Image::new(w, h, channels, a.source_bit_depth)
}

// ───────────────────────────── uscita ─────────────────────────────

/// A frame as 8-bit RGB with the capture time written in the corner (when `label`).
pub fn to_rgb8(img: &Image, label: Option<&str>) -> Vec<u8> {
    let (w, h) = (img.width(), img.height());
    let rgba = img.to_rgba8();
    let mut rgb: Vec<u8> = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    if let Some(text) = label {
        draw_label(&mut rgb, w, h, text);
    }
    rgb
}

/// "2019-07-05 22:16:11 UT".
pub fn format_moment(m: &Moment) -> String {
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} UT", m.year, m.month, m.day, m.hour, m.minute, m.second.floor() as u32)
}

/// 5×7 glyphs of the few characters a time label needs.
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        '-' => [0, 0, 0, 0x1F, 0, 0, 0],
        ':' => [0, 0x04, 0, 0, 0, 0x04, 0],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        _ => [0; 7],
    }
}

/// Text at the bottom left, white with a dark outline so it reads on any sky.
fn draw_label(rgb: &mut [u8], w: usize, h: usize, text: &str) {
    let k = (h / 220).max(1);
    let (cw, ch) = (6 * k, 7 * k);
    let (x0, y0) = (k * 4, h.saturating_sub(ch + 4 * k));
    let put = |rgb: &mut [u8], x: isize, y: isize, v: u8| {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            let i = (y as usize * w + x as usize) * 3;
            rgb[i..i + 3].fill(v);
        }
    };
    for pass in 0..2 {
        for (n, c) in text.chars().enumerate() {
            let g = glyph(c);
            for (row, bits) in g.iter().enumerate() {
                for col in 0..5 {
                    if bits >> (4 - col) & 1 == 0 {
                        continue;
                    }
                    let (px, py) = ((x0 + n * cw + col * k) as isize, (y0 + row * k) as isize);
                    for dy in 0..k as isize {
                        for dx in 0..k as isize {
                            if pass == 0 {
                                // The outline: one step around.
                                for oy in -(k as isize)..=(k as isize) {
                                    for ox in -(k as isize)..=(k as isize) {
                                        put(rgb, px + dx + ox, py + dy + oy, 0);
                                    }
                                }
                            } else {
                                put(rgb, px + dx, py + dy, 235);
                            }
                        }
                    }
                }
            }
        }
    }
}
