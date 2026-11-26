//! Il globo di un pianeta su un'immagine: geometria, misura del lembo, luce, mappa.
//!
//! - **Geometria** ([`Globe`], [`Projection`]): il pianeta è uno sferoide schiacciato
//!   (raggio polare `k` volte quello equatoriale), inclinato verso di noi della latitudine B
//!   del centro del disco, proiettato ortograficamente. Sul piano dell'immagine si usano le
//!   coordinate `(u, v)` in raggi equatoriali: `u` lungo l'equatore, `v` verso il polo nord.
//!   Sul globo: `α` è l'angolo dal meridiano centrale (positivo verso `+u`, cioè nel verso in
//!   cui la superficie si muove sulle immagini), `φ` la latitudine planetografica.
//! - **Lembo** ([`fit_limb`]): il bordo è il punto di massima discesa della luminosità lungo
//!   tanti raggi dal centro; un'ellisse con lo schiacciamento noto si adatta ai punti con pesi
//!   robusti. Il massimo della discesa è al bordo sia su un'immagine grezza (morbida) sia su
//!   una elaborata (orlo chiaro dentro e anello scuro fuori); i raggi che incontrano gli anelli
//!   di Saturno o il terminatore di Marte cadono lontano dall'ellisse e vengono scartati.
//! - **Luce** ([`Shading`]): la luminosità media in funzione della distanza dal bordo (limb
//!   darkening, orli lasciati dalla nitidezza, alone) più una pendenza lineare (la fase di
//!   Marte). Non gira con il pianeta: si separa dal dettaglio prima di derotare e si riapplica
//!   alla geometria finale, come la compensazione del lembo di WinJUPOS ma misurata
//!   sull'immagine stessa.
//! - **Mappa** ([`PlanetMap`]): proiezione cilindrica (longitudine × latitudine) del dettaglio
//!   senza la luce; [`PlanetMap::render_globe`] la riproietta su un globo girato di un angolo
//!   qualsiasi (per un'animazione del pianeta che ruota).

use crate::capture::Moment;
use crate::image::{Image, bilinear_clamped};
use rayon::prelude::*;
use std::f64::consts::{PI, TAU};

/// Il globo su un'immagine: centro e raggio equatoriale in pixel (coordinate continue, il
/// centro del pixel `i` è a `i + 0,5`), e la direzione dell'equatore (radianti, assi
/// dell'immagine con y verso il basso). Con `angle` = 0 il nord è in alto.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Globe {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
    pub angle: f32,
}

impl Globe {
    /// Pixel → piano dell'immagine in raggi equatoriali (`u` lungo l'equatore, `v` verso nord).
    pub fn to_plane(&self, x: f32, y: f32) -> (f64, f64) {
        let (s, c) = (self.angle as f64).sin_cos();
        let (dx, dy) = ((x - self.cx) as f64, (y - self.cy) as f64);
        let r = self.radius as f64;
        ((dx * c + dy * s) / r, (dx * s - dy * c) / r)
    }

    /// Piano dell'immagine → pixel.
    pub fn to_pixel(&self, u: f64, v: f64) -> (f32, f32) {
        let (s, c) = (self.angle as f64).sin_cos();
        let r = self.radius as f64;
        (self.cx + ((u * c + v * s) * r) as f32, self.cy + ((u * s - v * c) * r) as f32)
    }
}

/// Lo sferoide visto con inclinazione B. Assi del corpo: X lungo l'equatore verso `+u`, Y
/// lungo l'asse verso nord, Z verso il meridiano centrale; l'osservatore guarda lungo
/// −(0, sin B, cos B).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    sb: f64,
    cb: f64,
    /// Raggio polare / equatoriale.
    pub k: f64,
    inv_k2: f64,
    /// Semiasse polare apparente del contorno: √(k² cos² B + sin² B).
    pub kp: f64,
}

impl Projection {
    pub fn new(tilt_deg: f64, oblateness: f64) -> Projection {
        let (sb, cb) = tilt_deg.to_radians().sin_cos();
        let k = oblateness.clamp(0.5, 1.0);
        Projection { sb, cb, k, inv_k2: 1.0 / (k * k), kp: (k * k * cb * cb + sb * sb).sqrt() }
    }

    pub fn tilt_deg(&self) -> f64 {
        self.sb.atan2(self.cb).to_degrees()
    }

    /// Il punto del globo visto in (u, v), sul lato rivolto verso di noi.
    pub fn to_globe(&self, u: f64, v: f64) -> Option<[f64; 3]> {
        let (sb, cb, ik) = (self.sb, self.cb, self.inv_k2);
        // P = u·(1,0,0) + v·(0,cos B,−sin B) + t·(0,sin B,cos B) su X² + Y²/k² + Z² = 1.
        let a = sb * sb * ik + cb * cb;
        let b = v * sb * cb * (ik - 1.0);
        let c = u * u + v * v * (cb * cb * ik + sb * sb) - 1.0;
        let disc = b * b - a * c;
        if disc < 0.0 {
            return None;
        }
        let t = (-b + disc.sqrt()) / a;
        Some([u, v * cb + t * sb, -v * sb + t * cb])
    }

    /// Punto del globo → piano dell'immagine, e il coseno dell'angolo fra la sua superficie e
    /// la linea di vista (μ ≤ 0: nascosto dietro il bordo).
    pub fn to_plane(&self, [x, y, z]: [f64; 3]) -> ((f64, f64), f64) {
        let n = [x, y * self.inv_k2, z];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        let mu = (n[1] * self.sb + n[2] * self.cb) / len;
        ((x, y * self.cb - z * self.sb), mu)
    }

    /// Distanza dal centro in unità del contorno apparente: 1 sul bordo.
    pub fn rho(&self, u: f64, v: f64) -> f64 {
        (u * u + (v / self.kp).powi(2)).sqrt()
    }

    /// Il coseno dell'angolo di vista del punto visto in (u, v) (0 fuori dal globo).
    pub fn mu(&self, u: f64, v: f64) -> f64 {
        self.to_globe(u, v).map_or(0.0, |p| self.to_plane(p).1.max(0.0))
    }

    /// Quanto è vicino all'osservatore il punto del globo (per l'ordine davanti/dietro).
    fn depth(&self, p: [f64; 3]) -> f64 {
        p[1] * self.sb + p[2] * self.cb
    }

    /// Gli anelli di Saturno davanti al globo o fuori da esso in (u, v): raggio nel piano degli
    /// anelli in raggi equatoriali, se la linea di vista lo attraversa davanti al globo.
    pub fn ring_radius(&self, u: f64, v: f64) -> Option<f64> {
        if self.sb.abs() < 1e-3 {
            return None;
        }
        // Y = v cos B + t sin B = 0.
        let t = -v * self.cb / self.sb;
        let z = -v * self.sb + t * self.cb;
        if let Some(p) = self.to_globe(u, v)
            && self.depth(p) > t
        {
            return None;
        }
        Some((u * u + z * z).sqrt())
    }
}

/// Raggi interno ed esterno degli anelli visibili di Saturno (anello C – anello A), in raggi
/// equatoriali.
pub const SATURN_RINGS: (f64, f64) = (1.22, 2.29);

/// Gira un punto del globo attorno all'asse di `angle` (verso +X).
pub fn turn([x, y, z]: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, c) = angle.sin_cos();
    [x * c + z * s, y, z * c - x * s]
}

/// Angolo dal meridiano centrale e latitudine planetografica (radianti) di un punto del globo.
pub fn lon_lat([x, y, z]: [f64; 3], k: f64) -> (f64, f64) {
    let alpha = x.atan2(z);
    let centric = y.atan2((x * x + z * z).sqrt());
    (alpha, (centric.tan() / (k * k)).atan())
}

/// Il punto del globo con angolo `alpha` dal meridiano centrale e latitudine planetografica `lat`.
pub fn point(alpha: f64, lat: f64, k: f64) -> [f64; 3] {
    let centric = (lat.tan() * k * k).atan();
    let (sc, cc) = centric.sin_cos();
    let r = 1.0 / (cc * cc + sc * sc / (k * k)).sqrt();
    let (sa, ca) = alpha.sin_cos();
    [r * cc * sa, r * sc, r * cc * ca]
}

/// La luminosità di un'immagine (media dei canali).
pub fn luminance(img: &Image) -> Vec<f32> {
    let chans = img.channels();
    let n = chans.len() as f32;
    (0..img.width() * img.height()).into_par_iter().map(|i| chans.iter().map(|c| c[i]).sum::<f32>() / n).collect()
}

/// Il risultato della misura del lembo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LimbFit {
    pub globe: Globe,
    /// Scarto quadratico medio dei punti del bordo tenuti, pixel.
    pub rms: f32,
    /// Raggi con un bordo tenuto, e raggi in tutto.
    pub used: usize,
    pub total: usize,
}

/// Misura il bordo del globo partendo da `start` (che deve esserne vicino entro ~15 % del
/// raggio): centro e raggio, e anche la direzione dell'equatore se `fit_angle` (utile solo con
/// un globo schiacciato). `proj` dà lo schiacciamento apparente del contorno. `exclude` dice
/// quali direzioni (angolo sul piano `(u, v)`, radianti) non usare.
pub fn fit_limb(lum: &[f32], w: usize, h: usize, start: Globe, proj: &Projection, fit_angle: bool) -> Option<LimbFit> {
    let r0 = start.radius as f64;
    if r0 < 8.0 || w < 16 || h < 16 {
        return None;
    }
    let smooth = blur(lum, w, h, (r0 as f32 / 250.0).max(0.7));
    let kp = proj.kp;
    let rays = ((4.0 * r0) as usize).clamp(180, 1440);
    // Edge points: the steepest descent along each ray, from 0.8 to 1.2 of the expected radius.
    let points: Vec<(f64, f64, f64)> = (0..rays)
        .into_par_iter()
        .filter_map(|k| {
            let t = TAU * (k as f64 + 0.5) / rays as f64;
            // Direction in the (u, v) plane and the expected outline radius along it.
            let (du, dv) = (t.cos(), t.sin());
            let expected = 1.0 / (du * du + (dv / kp).powi(2)).sqrt();
            let (x0, y0) = start.to_pixel(0.0, 0.0);
            let (x1, y1) = start.to_pixel(du, dv);
            let (ex, ey) = ((x1 - x0) as f64, (y1 - y0) as f64);
            let len = (ex * ex + ey * ey).sqrt();
            let (ex, ey) = (ex / len, ey / len);
            let (a, b) = (0.8 * expected * r0, 1.2 * expected * r0);
            let step = 0.25;
            let n = ((b - a) / step) as usize;
            let sample = |r: f64| bilinear_clamped(&smooth, w, h, (x0 as f64 + ex * r) as f32, (y0 as f64 + ey * r) as f32) as f64;
            let mut best = (0.0f64, 0usize);
            let vals: Vec<f64> = (0..=n + 8).map(|i| sample(a - 4.0 * step + i as f64 * step)).collect();
            for i in 4..=n + 4 {
                // Derivative over one pixel.
                let d = vals[i + 2] - vals[i - 2];
                if -d > best.0 {
                    best = (-d, i);
                }
            }
            if best.0 <= 0.0 {
                return None;
            }
            // Sub-sample peak of the descent.
            let i = best.1;
            let d = |i: usize| vals[(i + 2).min(vals.len() - 1)] - vals[i.saturating_sub(2)];
            let (dm, d0, dp) = (-d(i - 1), -d(i), -d(i + 1));
            let denom = dm - 2.0 * d0 + dp;
            let off = if denom.abs() > 1e-12 { (0.5 * (dm - dp) / denom).clamp(-0.5, 0.5) } else { 0.0 };
            let r = a + (i as f64 - 4.0 + off) * step;
            let x = x0 as f64 + ex * r;
            let y = y0 as f64 + ey * r;
            Some((x, y, best.0))
        })
        .collect();
    if points.len() < 24 {
        return None;
    }
    // Only edges with a clear descent: at least a third of the median strength.
    let mut strengths: Vec<f64> = points.iter().map(|p| p.2).collect();
    strengths.sort_by(f64::total_cmp);
    let median = strengths[strengths.len() / 2];
    let points: Vec<(f64, f64, f64)> = points.into_iter().filter(|p| p.2 > median / 3.0).collect();

    // Robust Gauss–Newton on (cx, cy, R[, angle]).
    let mut p = [start.cx as f64, start.cy as f64, r0, start.angle as f64];
    let nparams = if fit_angle { 4 } else { 3 };
    let residual = |p: &[f64; 4], x: f64, y: f64| {
        let (s, c) = p[3].sin_cos();
        let (dx, dy) = (x - p[0], y - p[1]);
        let (u, v) = (dx * c + dy * s, dx * s - dy * c);
        (u * u + (v / kp).powi(2)).sqrt() - p[2]
    };
    let mut weights = vec![1.0f64; points.len()];
    let mut scale = 0.05 * r0;
    for iter in 0..25 {
        let res: Vec<f64> = points.iter().map(|q| residual(&p, q.0, q.1)).collect();
        // Robust scale from the median absolute residual.
        let mut abs: Vec<f64> = res.iter().map(|r| r.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let mad = 1.4826 * abs[abs.len() / 2];
        scale = if iter < 3 { (0.05 * r0).max(3.0 * mad) } else { (2.5 * mad).max(0.35) };
        for (wt, (r, q)) in weights.iter_mut().zip(res.iter().zip(&points)) {
            let t = r / scale;
            *wt = if t.abs() < 1.0 { (1.0 - t * t).powi(2) * q.2.sqrt() } else { 0.0 };
        }
        // Normal equations with a numerical Jacobian.
        let mut ata = [[0.0f64; 4]; 4];
        let mut atb = [0.0f64; 4];
        for ((q, &r), &wt) in points.iter().zip(&res).zip(&weights) {
            if wt == 0.0 {
                continue;
            }
            let mut j = [0.0f64; 4];
            for (k, jk) in j.iter_mut().enumerate().take(nparams) {
                let mut pp = p;
                let hstep = if k == 3 { 1e-4 } else { 1e-3 };
                pp[k] += hstep;
                *jk = (residual(&pp, q.0, q.1) - r) / hstep;
            }
            for a in 0..nparams {
                atb[a] -= wt * j[a] * r;
                for b in 0..nparams {
                    ata[a][b] += wt * j[a] * j[b];
                }
            }
        }
        let Some(delta) = solve(&ata, &atb, nparams) else { break };
        for k in 0..nparams {
            p[k] += delta[k];
        }
        if delta.iter().take(3).all(|d| d.abs() < 1e-4) && delta[3].abs() < 1e-6 {
            break;
        }
    }
    let kept: Vec<f64> = points.iter().map(|q| residual(&p, q.0, q.1)).filter(|r| r.abs() < scale).collect();
    if kept.len() < 12 || !(0.7 * r0..1.3 * r0).contains(&p[2]) {
        return None;
    }
    let rms = (kept.iter().map(|r| r * r).sum::<f64>() / kept.len() as f64).sqrt() as f32;
    Some(LimbFit { globe: Globe { cx: p[0] as f32, cy: p[1] as f32, radius: p[2] as f32, angle: p[3] as f32 }, rms, used: kept.len(), total: rays })
}

/// Gaussian blur (three box blurs) on the CPU: these planes are many and small, and run from
/// several threads at once, where a round trip to the GPU for each would queue up.
pub(crate) fn blur(plane: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = ((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() / 2.0).round().max(1.0) as usize;
    crate::mask::box_blur3(plane, w, h, r)
}

/// Solve the `n`×`n` system (Gaussian elimination with pivoting).
fn solve(a: &[[f64; 4]; 4], b: &[f64; 4], n: usize) -> Option<[f64; 4]> {
    let mut m = [[0.0f64; 5]; 4];
    for i in 0..n {
        m[i][..n].copy_from_slice(&a[i][..n]);
        m[i][4] = b[i];
    }
    for c in 0..n {
        let piv = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[piv][c].abs() < 1e-12 {
            return None;
        }
        m.swap(c, piv);
        for r in 0..n {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..5 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    let mut x = [0.0f64; 4];
    for i in 0..n {
        x[i] = m[i][4] / m[i][i];
    }
    Some(x)
}

/// La luce sul disco: luminosità media in funzione dell'angolo di vista μ (un valore per ogni
/// anello di uguale area, `1 − μ²` a passi uguali, fino al bordo) per ogni canale, per una
/// pendenza lineare `1 + a·u + b·v` (la fase). Non dipende dalla rotazione del pianeta.
#[derive(Clone, Debug, PartialEq)]
pub struct Shading {
    /// `[canale][anello]`: l'anello `i` copre 1 − μ² ∈ [i/n, (i+1)/n).
    pub profile: Vec<Vec<f32>>,
    /// `[canale]`: (a, b).
    pub slope: Vec<[f32; 2]>,
}

impl Shading {
    /// La luce del canale `c` in (u, v), dove μ è il coseno dell'angolo di vista ([`Projection::mu`]).
    pub fn at(&self, c: usize, mu: f64, u: f64, v: f64) -> f32 {
        let prof = &self.profile[c.min(self.profile.len() - 1)];
        let n = prof.len();
        let x = (1.0 - mu * mu).clamp(0.0, 1.0) as f32 * n as f32 - 0.5;
        let i = (x.floor().max(0.0) as usize).min(n - 1);
        let j = (i + 1).min(n - 1);
        let t = (x - i as f32).clamp(0.0, 1.0);
        let base = prof[i] + t * (prof[j] - prof[i]);
        let [a, b] = self.slope[c.min(self.slope.len() - 1)];
        base * (1.0 + a * u as f32 + b * v as f32).max(0.05)
    }

    /// Il livello più alto del profilo del canale `c` (per i minimi delle divisioni).
    pub fn peak(&self, c: usize) -> f32 {
        self.profile[c.min(self.profile.len() - 1)].iter().copied().fold(0.0, f32::max)
    }

    /// Il livello medio del disco nel canale `c`.
    pub fn level(&self, c: usize) -> f32 {
        let p = &self.profile[c.min(self.profile.len() - 1)];
        p.iter().sum::<f32>() / p.len().max(1) as f32
    }

    /// Misura la luce di un'immagine con il suo globo; `skip(u, v)` esclude dei punti (gli
    /// anelli di Saturno davanti al globo).
    pub fn fit(img: &Image, globe: &Globe, proj: &Projection, skip: &(dyn Fn(f64, f64) -> bool + Sync)) -> Shading {
        Shading::fit_with(img, globe, proj, skip, None)
    }

    /// Come [`Shading::fit`], con la pendenza data (per canale) invece che misurata: la luce
    /// di più riprese della stessa sera è la stessa, e la pendenza di una sola risente delle
    /// macchie che vi si trovano.
    pub fn fit_with(img: &Image, globe: &Globe, proj: &Projection, skip: &(dyn Fn(f64, f64) -> bool + Sync), fixed_slope: Option<&[[f32; 2]]>) -> Shading {
        let (w, h) = (img.width(), img.height());
        let n = ((2.0 * globe.radius) as usize).clamp(48, 1500);
        let reach = globe.radius * 1.02 + 2.0;
        let (x0, x1) = (((globe.cx - reach).floor().max(0.0)) as usize, ((globe.cx + reach).ceil().max(0.0) as usize).min(w));
        let (y0, y1) = (((globe.cy - reach).floor().max(0.0)) as usize, ((globe.cy + reach).ceil().max(0.0) as usize).min(h));
        // Disk pixels: (index, 1 − μ², u, v).
        let mut pix: Vec<(usize, f64, f64, f64)> = Vec::new();
        for y in y0..y1 {
            for x in x0..x1 {
                let (u, v) = globe.to_plane(x as f32 + 0.5, y as f32 + 0.5);
                if proj.rho(u, v) < 1.0 && !skip(u, v) {
                    let mu = proj.mu(u, v);
                    pix.push((y * w + x, 1.0 - mu * mu, u, v));
                }
            }
        }
        let mut profile = Vec::new();
        let mut slope = Vec::new();
        for (c, plane) in img.channels().iter().enumerate() {
            let given = fixed_slope.map(|s| s[c.min(s.len() - 1)]);
            let mut ab = given.unwrap_or([0.0f32; 2]);
            let mut prof = vec![0.0f32; n];
            for _ in 0..2 {
                // Median per equal-area ring of the image divided by the slope.
                let mut bins: Vec<Vec<f32>> = vec![Vec::new(); n];
                for &(i, s2, u, v) in &pix {
                    let tilt = (1.0 + ab[0] * u as f32 + ab[1] * v as f32).max(0.05);
                    bins[((s2 * n as f64) as usize).min(n - 1)].push(plane[i] / tilt);
                }
                for (b, out) in bins.iter_mut().zip(prof.iter_mut()) {
                    *out = if b.is_empty() {
                        f32::NAN
                    } else {
                        let m = b.len() / 2;
                        *b.select_nth_unstable_by(m, f32::total_cmp).1
                    };
                }
                fill_gaps(&mut prof);
                smooth_profile(&mut prof, globe.radius);
                smooth_law(&mut prof);
                if given.is_some() {
                    continue;
                }
                // Linear slope of what is left, inside 90 % of the disk.
                let (mut suu, mut suv, mut svv, mut sur, mut svr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for &(i, s2, u, v) in &pix {
                    if s2 > 0.81 {
                        continue;
                    }
                    let base = prof[((s2 * n as f64) as usize).min(n - 1)] as f64;
                    if base <= 1e-6 {
                        continue;
                    }
                    let r = plane[i] as f64 / base - 1.0;
                    suu += u * u;
                    suv += u * v;
                    svv += v * v;
                    sur += u * r;
                    svr += v * r;
                }
                let det = suu * svv - suv * suv;
                if det.abs() > 1e-9 {
                    ab = [((sur * svv - svr * suv) / det) as f32, ((svr * suu - sur * suv) / det) as f32];
                    ab = ab.map(|c| c.clamp(-1.5, 1.5));
                }
            }
            profile.push(prof);
            slope.push(ab);
        }
        Shading { profile, slope }
    }

    /// La media pesata di più luci (tutte con lo stesso numero di canali).
    pub fn mean(list: &[(&Shading, f32)]) -> Option<Shading> {
        let (first, _) = list.first()?;
        let total: f32 = list.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            return None;
        }
        let n = first.profile.first()?.len();
        let channels = first.profile.len();
        let mut profile = vec![vec![0.0f32; n]; channels];
        let mut slope = vec![[0.0f32; 2]; channels];
        for (s, w) in list {
            for c in 0..channels {
                let p = &s.profile[c.min(s.profile.len() - 1)];
                for (i, out) in profile[c].iter_mut().enumerate() {
                    // Resample if the ring count differs.
                    let j = (i * p.len() / n).min(p.len() - 1);
                    *out += w * p[j] / total;
                }
                let sl = s.slope[c.min(s.slope.len() - 1)];
                slope[c][0] += w * sl[0] / total;
                slope[c][1] += w * sl[1] / total;
            }
        }
        Some(Shading { profile, slope })
    }
}

/// Inside 90 % of the disk the light follows a smooth law: a cubic in μ fitted (robustly) to
/// the ring medians. Near the centre a ring holds few pixels and its median would learn a
/// feature that sits there (and erase it, or ring it with a halo). The outer rings, long and
/// narrow, keep their own values: the rim left by sharpening is there.
fn smooth_law(p: &mut [f32]) {
    let n = p.len();
    let mu = |i: usize| (1.0 - (i as f64 + 0.5) / n as f64).max(0.0).sqrt();
    let fit_end = (0.85 * n as f64) as usize;
    let mut keep: Vec<bool> = vec![true; fit_end];
    let mut coef = [0.0f64; 4];
    for _ in 0..3 {
        let mut ata = [[0.0f64; 4]; 4];
        let mut atb = [0.0f64; 4];
        for i in (0..fit_end).filter(|&i| keep[i]) {
            let m = mu(i);
            let row = [1.0, m, m * m, m * m * m];
            for a in 0..4 {
                atb[a] += row[a] * p[i] as f64;
                for b in 0..4 {
                    ata[a][b] += row[a] * row[b];
                }
            }
        }
        let Some(c) = solve(&ata, &atb, 4) else { return };
        coef = c;
        let res: Vec<f64> = (0..fit_end).map(|i| p[i] as f64 - law(&coef, mu(i))).collect();
        let mut abs: Vec<f64> = res.iter().map(|r| r.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let mad = 1.4826 * abs[abs.len() / 2] + 1e-9;
        keep = res.iter().map(|r| r.abs() < 3.0 * mad).collect();
    }
    for (i, out) in p.iter_mut().enumerate() {
        let s = (i as f64 + 0.5) / n as f64;
        // Law inside s < 0.8 (ρ ≈ 0.9), own values beyond 0.9 (ρ ≈ 0.95).
        let t = ((s - 0.8) / 0.1).clamp(0.0, 1.0) as f32;
        *out = (1.0 - t) * law(&coef, mu(i)) as f32 + t * *out;
    }
}

fn law(c: &[f64; 4], m: f64) -> f64 {
    c[0] + m * (c[1] + m * (c[2] + m * c[3]))
}

/// Smooth a profile over ~2 px of radius inside the disk, down to ~0.4 px at the edge (where
/// the rim left by sharpening is): the ring medians are noisy, and their ripples would be a
/// fixed pattern on the disk (which doesn't turn with the planet).
fn smooth_profile(p: &mut [f32], radius: f32) {
    let n = p.len();
    let raw = p.to_vec();
    for (i, out) in p.iter_mut().enumerate() {
        let s = ((i as f64 + 0.5) / n as f64).clamp(1e-4, 1.0);
        // Pixels of radius per bin (ρ ≈ √s on a sphere).
        let px_per_bin = radius as f64 / (2.0 * s.sqrt() * n as f64);
        let edge_px = (1.0 - s.sqrt()) * radius as f64;
        let sigma_px = 0.4 + 1.6 * (edge_px / 6.0).clamp(0.0, 1.0);
        let sigma = sigma_px / px_per_bin;
        if sigma < 0.3 {
            continue;
        }
        let reach = (3.0 * sigma).ceil() as isize;
        let (mut sw, mut sv) = (0.0f64, 0.0f64);
        for d in -reach..=reach {
            let j = i as isize + d;
            if j < 0 || j >= n as isize {
                continue;
            }
            let wgt = (-(d as f64 / sigma).powi(2) / 2.0).exp();
            sw += wgt;
            sv += wgt * raw[j as usize] as f64;
        }
        *out = (sv / sw) as f32;
    }
}

fn fill_gaps(p: &mut [f32]) {
    let n = p.len();
    for i in 0..n {
        if p[i].is_nan() {
            let before = (0..i).rev().find(|&j| !p[j].is_nan());
            let after = (i + 1..n).find(|&j| !p[j].is_nan());
            p[i] = match (before, after) {
                (Some(a), Some(b)) => p[a] + (p[b] - p[a]) * (i - a) as f32 / (b - a) as f32,
                (Some(a), None) => p[a],
                (None, Some(b)) => p[b],
                (None, None) => 0.0,
            };
        }
    }
}

/// La mappa cilindrica del pianeta (equirettangolare): il dettaglio senza la luce, cioè
/// circa 1 dove la superficie ha la luminosità media del disco, più chiaro o più scuro altrove.
///
/// Colonna `i` = angolo `α = −180° + 360°·(i + ½)/width` dal meridiano centrale all'istante di
/// riferimento (positivo nel verso in cui la superficie si muove sulle immagini: la mappa ha
/// lo stesso verso del pianeta sulle riprese); riga `j` = latitudine planetografica
/// `90° − 180°·(j + ½)/height`. La longitudine del pianeta nel suo sistema (se nota) è
/// `central_longitude − α` per un'immagine non specchiata.
#[derive(Clone, Debug)]
pub struct PlanetMap {
    pub width: usize,
    pub height: usize,
    /// `[canale][j·width + i]`.
    pub channels: Vec<Vec<f32>>,
    /// Quanto è stato visto ogni punto (somma dei pesi, 0 = mai visto).
    pub coverage: Vec<f32>,
    /// La media di ogni riga (dove è stata vista), per riempire le longitudini mai viste.
    pub zonal: Vec<Vec<f32>>,
    /// La luce media delle riprese, per riaccendere il globo.
    pub shading: Shading,
    /// Raggio polare / equatoriale.
    pub oblateness: f64,
    /// Latitudine del centro del disco all'istante di riferimento, gradi.
    pub tilt_deg: f64,
    /// Periodo di rotazione usato, ore (per [`PlanetMap::turn_for_hours`]).
    pub period_hours: f64,
    /// La longitudine (gradi) del meridiano centrale all'istante di riferimento, nel sistema
    /// del pianeta (Giove: Sistema II), quando l'ora è nota.
    pub central_longitude: Option<f64>,
    /// L'istante di riferimento.
    pub moment: Option<Moment>,
    /// Le riprese erano specchiate (la superficie gira da destra a sinistra).
    pub mirrored: bool,
}

/// Come mostrare il globo con [`PlanetMap::render_globe`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobeView {
    pub width: usize,
    pub height: usize,
    /// Posizione, raggio equatoriale e orientamento nel risultato.
    pub globe: Globe,
    /// Latitudine del centro del disco, gradi (di solito quella della mappa).
    pub tilt_deg: f64,
    /// Di quanto il pianeta è girato dall'istante di riferimento, gradi, nel verso della
    /// rotazione (positivo = più tardi).
    pub turn_deg: f64,
    /// Riapplica la luce delle riprese (bordo, fase); altrimenti il dettaglio puro.
    pub lit: bool,
    /// Le longitudini mai viste prendono la media della loro latitudine (altrimenti il fondo).
    pub fill_unseen: bool,
    /// Il valore del cielo.
    pub background: f32,
}

impl PlanetMap {
    /// Gradi di rotazione in `hours` ore.
    pub fn turn_for_hours(&self, hours: f64) -> f64 {
        360.0 * hours / self.period_hours
    }

    fn column_of(&self, alpha: f64) -> f64 {
        ((alpha + PI) / TAU).rem_euclid(1.0) * self.width as f64 - 0.5
    }

    fn row_of(&self, lat: f64) -> f64 {
        (0.5 - lat / PI) * self.height as f64 - 0.5
    }

    /// Il valore della mappa (canale `c`) in (α, φ) radianti, bilineare con il giro in
    /// longitudine; con la copertura.
    pub fn sample(&self, c: usize, alpha: f64, lat: f64) -> (f32, f32) {
        let (w, h) = (self.width, self.height);
        let x = self.column_of(alpha);
        let y = self.row_of(lat).clamp(0.0, (h - 1) as f64);
        let (x0, y0) = (x.floor(), y.floor() as usize);
        let (tx, ty) = ((x - x0) as f32, (y - y0 as f64) as f32);
        let xa = (x0 as isize).rem_euclid(w as isize) as usize;
        let xb = (xa + 1) % w;
        let y1 = (y0 + 1).min(h - 1);
        let plane = &self.channels[c.min(self.channels.len() - 1)];
        let mix = |p: &[f32]| {
            let top = p[y0 * w + xa] + tx * (p[y0 * w + xb] - p[y0 * w + xa]);
            let bottom = p[y1 * w + xa] + tx * (p[y1 * w + xb] - p[y1 * w + xa]);
            top + ty * (bottom - top)
        };
        (mix(plane), mix(&self.coverage))
    }

    /// La mappa come immagine (dettaglio × livello medio del disco), le parti mai viste
    /// riempite con la media della latitudine se `fill_unseen`.
    pub fn to_image(&self, fill_unseen: bool) -> Image {
        let (w, h) = (self.width, self.height);
        let channels = self
            .channels
            .iter()
            .enumerate()
            .map(|(c, plane)| {
                let level = self.shading.level(c);
                (0..w * h)
                    .map(|i| {
                        let cov = self.coverage[i];
                        let fill = if fill_unseen { self.zonal[c][i / w] } else { 0.0 };
                        let t = (cov / 0.05).min(1.0);
                        (t * plane[i] + (1.0 - t) * fill) * level
                    })
                    .collect()
            })
            .collect();
        Image::new(w, h, channels, 16)
    }

    /// Il globo visto come dice `view`: per ogni pixel il punto della superficie che vi si
    /// vede, girato di `turn_deg`, preso dalla mappa e (se `lit`) riacceso con la luce delle
    /// riprese. Per un'animazione: lo stesso `view` con `turn_deg` che cresce.
    pub fn render_globe(&self, view: &GlobeView) -> Image {
        let (w, h) = (view.width.max(1), view.height.max(1));
        let proj = Projection::new(view.tilt_deg, self.oblateness);
        // The map is in the captures' own frame (α grows the way the surface moves on them).
        let turn_rad = view.turn_deg.to_radians();
        let nc = self.channels.len();
        let mut out = vec![view.background; w * h * nc];
        out.par_chunks_mut(w * nc).enumerate().for_each(|(j, row)| {
            for i in 0..w {
                let (u, v) = view.globe.to_plane(i as f32 + 0.5, j as f32 + 0.5);
                let rho = proj.rho(u, v);
                // A pixel-wide soft edge.
                let edge = ((1.0 - rho) * view.globe.radius as f64 + 0.5).clamp(0.0, 1.0) as f32;
                if edge <= 0.0 {
                    continue;
                }
                let Some(p) = proj.to_globe(u, v.clamp(-proj.kp * 0.999_999, proj.kp * 0.999_999)) else { continue };
                let mu = proj.to_plane(p).1.max(0.0);
                // What is seen at α at this moment was at α − turn at the reference instant
                // (with the map's own direction; turning forward moves the surface towards +u).
                let (alpha, lat) = lon_lat(turn(p, -turn_rad), self.oblateness);
                let row_index = self.row_of(lat).round().clamp(0.0, (self.height - 1) as f64) as usize;
                for c in 0..nc {
                    let (value, cov) = self.sample(c, alpha, lat);
                    let t = (cov / 0.05).min(1.0);
                    let fill = if view.fill_unseen { self.zonal[c][row_index] } else { 0.0 };
                    let mut x = t * value + (1.0 - t) * fill;
                    x *= if view.lit { self.shading.at(c, mu, u, v) } else { self.shading.level(c) };
                    row[i * nc + c] = edge * x + (1.0 - edge) * view.background;
                }
            }
        });
        let channels = (0..nc).map(|c| out.iter().skip(c).step_by(nc).copied().collect()).collect();
        Image::new(w, h, channels, 16)
    }
}

/// Media di ogni riga dove è stata vista.
pub(crate) fn zonal_means(channels: &[Vec<f32>], coverage: &[f32], w: usize, h: usize) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = channels
        .iter()
        .map(|plane| {
            let mut z: Vec<f32> = (0..h)
                .map(|j| {
                    let (mut s, mut n) = (0.0f64, 0.0f64);
                    for i in 0..w {
                        let c = coverage[j * w + i] as f64;
                        s += c * plane[j * w + i] as f64;
                        n += c;
                    }
                    if n > 1e-6 { (s / n) as f32 } else { f32::NAN }
                })
                .collect();
            fill_gaps(&mut z);
            z
        })
        .collect();
    for z in &mut out {
        if z.iter().all(|v| *v == 0.0) {
            z.iter_mut().for_each(|v| *v = 1.0);
        }
    }
    out
}
