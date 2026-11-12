//! Where on the Moon a picture was taken, found by comparing it with a *drawing of the Moon made
//! for that very moment*: the LOLA height map (NASA, public domain) is lit from the Sun's real
//! position, seen from the Earth with the real libration, and the picture is fitted onto it.
//!
//! The comparison is on the relief's shading only (log brightness, band-passed, locally
//! normalised), so the Moon's dark plains and bright rays — which a height map does not have —
//! mostly drop out. The picture is unknown in four numbers (where on the disk, how big, how turned)
//! plus the mirror; the Sun's direction on the disk is known, so the picture's own light axis gives
//! the turn up to a small window, which keeps the search small:
//!
//! 1. coarse: a grid of places × sizes × turns, scored by normalised correlation on a tiny copy;
//! 2. fine: the best few are polished by a pattern search on a larger copy.
//!
//! This is a prototype: the height map is read from a PDS file (`ldem_16.img`, 16 pixels/degree).

use crate::capture::Moment;
use crate::identify::{Orientation, Placement};
use crate::selene::{self, MoonView};
use rayon::prelude::*;

const RAD: f64 = std::f64::consts::PI / 180.0;
const DEM_W: usize = 5760;
const DEM_H: usize = 2880;

/// The height map: 16 pixels per degree, longitude 0–360 east, latitude +90 at the top.
pub struct Dem {
    data: Vec<i16>,
}

impl Dem {
    pub fn load_pds(path: &std::path::Path) -> std::io::Result<Dem> {
        let bytes = std::fs::read(path)?;
        if bytes.len() != DEM_W * DEM_H * 2 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "ldem_16.img has an unexpected size"));
        }
        Ok(Dem { data: bytes.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect() })
    }

    /// Height (m) at latitude, east longitude (degrees), bilinear.
    fn height(&self, lat: f64, lon: f64) -> f64 {
        let row = ((90.0 - lat) * 16.0 - 0.5).clamp(0.0, (DEM_H - 1) as f64);
        let col = (lon.rem_euclid(360.0) * 16.0 - 0.5).rem_euclid(DEM_W as f64);
        let (r0, c0) = (row.floor() as usize, col.floor() as usize);
        let (fr, fc) = (row - r0 as f64, col - c0 as f64);
        let r1 = (r0 + 1).min(DEM_H - 1);
        let c1 = (c0 + 1) % DEM_W;
        let at = |r: usize, c: usize| self.data[r * DEM_W + c] as f64 * 0.5;
        (at(r0, c0) * (1.0 - fc) + at(r0, c1) * fc) * (1.0 - fr) + (at(r1, c0) * (1.0 - fc) + at(r1, c1) * fc) * fr
    }
}

/// The Moon's disk as the Earth sees it, `n`×`n` pixels (disk radius n/2), lit from the Sun's
/// position: brightness 0–1 (Lommel–Seeliger on the DEM's slopes, no cast shadows) and the disk mask.
pub fn render(dem: &Dem, v: &MoonView, n: usize) -> (Vec<f32>, Vec<bool>) {
    let (b0, l0) = (v.earth_lat, v.earth_lon);
    let unit = |lat: f64, lon: f64| [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let sun = unit(v.sun_lat, v.sun_lon);
    let earth = unit(b0, l0);
    let step = 1.0 / 16.0; // one DEM pixel, degrees
    let r_m = selene::RADIUS_KM * 1000.0;
    let mut out = vec![0.0f32; n * n];
    let mut mask = vec![false; n * n];
    out.par_chunks_mut(n).zip(mask.par_chunks_mut(n)).enumerate().for_each(|(j, (row, mrow))| {
        for i in 0..n {
            let x = (i as f64 + 0.5) / n as f64 * 2.0 - 1.0;
            let y = 1.0 - (j as f64 + 0.5) / n as f64 * 2.0;
            let zz = 1.0 - x * x - y * y;
            if zz <= 0.0 {
                continue;
            }
            let z = zz.sqrt();
            let lat = (y * b0.cos() + z * b0.sin()).clamp(-1.0, 1.0).asin();
            let lon = l0 + x.atan2(z * b0.cos() - y * b0.sin());
            let (lat_d, lon_d) = (lat / RAD, lon / RAD);
            let cos_lat = lat.cos().max(0.05);
            let de = r_m * cos_lat * step * RAD;
            let dn = r_m * step * RAD;
            let slope_e = (dem.height(lat_d, lon_d + step) - dem.height(lat_d, lon_d - step)) / (2.0 * de);
            let slope_n = (dem.height(lat_d + step, lon_d) - dem.height(lat_d - step, lon_d)) / (2.0 * dn);
            let up = unit(lat, lon);
            let east = [-lon.sin(), lon.cos(), 0.0];
            let north = [-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()];
            let mut normal = [up[0] - slope_e * east[0] - slope_n * north[0], up[1] - slope_e * east[1] - slope_n * north[1], up[2] - slope_e * east[2] - slope_n * north[2]];
            let len = dot(normal, normal).sqrt();
            normal.iter_mut().for_each(|c| *c /= len);
            let (mu0, mu) = (dot(normal, sun), dot(normal, earth).max(0.05));
            row[i] = if mu0 > 0.0 { (mu0 / (mu0 + mu)) as f32 } else { 0.0 };
            mrow[i] = true;
        }
    });
    (out, mask)
}

/// Gaussian blur (separable, reflected edges).
fn blur(img: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let norm: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / norm).collect();
    let refl = |i: isize, n: usize| -> usize {
        let n = n as isize;
        let m = i.rem_euclid(2 * n);
        (if m < n { m } else { 2 * n - 1 - m }) as usize
    };
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = k.iter().enumerate().map(|(t, kv)| kv * img[y * w + refl(x as isize + t as isize - r, w)]).sum();
        }
    });
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = k.iter().enumerate().map(|(t, kv)| kv * tmp[refl(y as isize + t as isize - r, h) * w + x]).sum();
        }
    });
    out
}

/// The relief's shading only: log brightness, band-passed, divided by its local size.
fn shading(img: &[f32], w: usize, h: usize, mask: &[bool]) -> Vec<f32> {
    let lg: Vec<f32> = img.iter().map(|v| (v.max(0.0) + 0.03).ln()).collect();
    let (fine, coarse) = (blur(&lg, w, h, 0.8), blur(&lg, w, h, 4.0));
    let band: Vec<f32> = fine.iter().zip(&coarse).map(|(a, b)| a - b).collect();
    let mag: Vec<f32> = band.iter().map(|v| v.abs()).collect();
    let local = blur(&mag, w, h, 10.0);
    band.iter().zip(&local).zip(mask).map(|((b, l), m)| if *m { b / (l + 0.01) } else { 0.0 }).collect()
}

/// The disk drawn once and kept at several sizes, each already shaded for comparison.
pub struct Reference {
    levels: Vec<Level>,
    /// The plain brightness at the finest size, for drawing the overlay.
    pub plain: (Vec<f32>, usize),
    /// The Sun's direction in the disk's frame: angle from +x (east) towards +y (north).
    sun_axis: f64,
}

struct Level {
    n: usize,
    data: Vec<f32>,
    mask: Vec<bool>,
}

impl Reference {
    /// Draw the disk for `when` (finest level `top`×`top`, a power of two) and its smaller copies.
    pub fn new(dem: &Dem, when: &Moment, top: usize) -> Reference {
        let v = selene::view(when);
        let (img, mask) = render(dem, &v, top);
        let plain = (img.clone(), top);
        let mut levels = Vec::new();
        let (mut cur, mut cmask, mut n) = (img, mask, top);
        loop {
            levels.push(Level { n, data: shading(&cur, n, n, &cmask), mask: cmask.clone() });
            if n <= 64 {
                break;
            }
            let half = n / 2;
            let mut next = vec![0.0f32; half * half];
            let mut nmask = vec![false; half * half];
            for j in 0..half {
                for i in 0..half {
                    let idx = [(2 * j) * n + 2 * i, (2 * j) * n + 2 * i + 1, (2 * j + 1) * n + 2 * i, (2 * j + 1) * n + 2 * i + 1];
                    let inside = idx.iter().filter(|&&k| cmask[k]).count();
                    nmask[j * half + i] = inside == 4;
                    next[j * half + i] = idx.iter().map(|&k| cur[k]).sum::<f32>() / 4.0;
                }
            }
            (cur, cmask, n) = (next, nmask, half);
        }
        let unit = |lat: f64, lon: f64| [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
        let (b0, l0) = (v.earth_lat, v.earth_lon);
        let east = [-l0.sin(), l0.cos(), 0.0];
        let north = [-b0.sin() * l0.cos(), -b0.sin() * l0.sin(), b0.cos()];
        let s = unit(v.sun_lat, v.sun_lon);
        let d = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        Reference { levels, plain, sun_axis: d(s, north).atan2(d(s, east)) }
    }

    /// The level whose pixel is closest to `1/rho` of the disk's radius (`rho`: disk radius in
    /// template pixels), so reference and template have the same sharpness.
    fn level_for(&self, rho: f64) -> &Level {
        let want = 2.0 * rho;
        self.levels.iter().min_by(|a, b| ((a.n as f64) / want).ln().abs().total_cmp(&((b.n as f64) / want).ln().abs())).expect("levels")
    }
}

/// A picture reduced to the shading the comparison uses.
struct Template {
    w: usize,
    h: usize,
    data: Vec<f32>,
    valid: Vec<bool>,
    /// Axis of the picture's brightness gradients (radians in 0..π, image axes, y down).
    axis: f64,
    /// Full-picture pixels per template pixel.
    factor: f64,
}

fn template(lum: &[f32], w: usize, h: usize, width: usize) -> Template {
    let f = (w as f64 / width as f64).max(1.0);
    let fi = f.round().max(1.0) as usize;
    let small = crate::image::downscale_plane(lum, w, h, fi);
    let (sw, sh) = ((w / fi).max(1), (h / fi).max(1));
    let mut sorted: Vec<f32> = small.iter().copied().step_by((small.len() / 50_000).max(1)).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let top = sorted[(sorted.len() as f32 * 0.99) as usize];
    let valid: Vec<bool> = small.iter().map(|v| *v > 0.04 * top).collect();
    let norm: Vec<f32> = small.iter().map(|v| (v / top.max(1e-9)).clamp(0.0, 1.5)).collect();
    let data = shading(&norm, sw, sh, &valid);
    // Where the light comes from: the doubled-angle mean of the gradients, weighted by their strength.
    let lg: Vec<f32> = norm.iter().map(|v| (v + 0.03).ln()).collect();
    let sm = blur(&lg, sw, sh, 1.2);
    let (mut c, mut s) = (0.0f64, 0.0f64);
    for y in 1..sh - 1 {
        for x in 1..sw - 1 {
            if !valid[y * sw + x] {
                continue;
            }
            let gx = (sm[y * sw + x + 1] - sm[y * sw + x - 1]) as f64;
            let gy = (sm[(y + 1) * sw + x] - sm[(y - 1) * sw + x]) as f64;
            let m2 = gx * gx + gy * gy;
            c += (gx * gx - gy * gy) * m2.sqrt();
            s += 2.0 * gx * gy * m2.sqrt();
        }
    }
    let axis = (s.atan2(c) / 2.0).rem_euclid(std::f64::consts::PI);
    Template { w: sw, h: sh, data, valid, axis, factor: w as f64 / sw as f64 }
}

/// Pattern search on (u, v, rho, angle) from `h0`: the fit and its correlation.
fn polish(fine: &Template, r: &Reference, h0: Hyp) -> (f64, Hyp) {
    let mut h = h0;
    let (mut best, _) = score(fine, r, &h);
    let mut step = [2.0 / h.rho, 2.0 / h.rho, 0.06, 2.0 * RAD];
    for _ in 0..60 {
        let mut improved = false;
        for p in 0..4 {
            for sign in [-1.0, 1.0] {
                let mut t = h;
                match p {
                    0 => t.u += sign * step[0],
                    1 => t.v += sign * step[1],
                    2 => t.rho *= (sign * step[2]).exp(),
                    _ => t.angle += sign * step[3],
                }
                let (s, cover) = score(fine, r, &t);
                if s > best && cover > 0.6 {
                    (best, h, improved) = (s, t, true);
                }
            }
        }
        if !improved {
            step.iter_mut().for_each(|s| *s *= 0.6);
            if step[0] * h.rho < 0.05 {
                break;
            }
        }
    }
    (best, h)
}

/// Where a fit puts the disk on the full picture.
fn to_placement(fine: &Template, best: Hyp) -> Placement {
    let or = best.orientation();
    let (ox, oy) = or.to_image(-best.u * best.rho, -best.v * best.rho);
    let (tcx, tcy) = ((fine.w as f64 - 1.0) / 2.0, (fine.h as f64 - 1.0) / 2.0);
    let k = fine.factor;
    Placement { cx: (((tcx + ox) + 0.5) * k - 0.5) as f32, cy: (((tcy + oy) + 0.5) * k - 0.5) as f32, r: (best.rho * k) as f32, orientation: or }
}

/// Polish a placement already roughly known (a disk found by its limb, say): the fit onto the relief
/// and its correlation, and the correlation of the starting placement.
pub fn refine(lum: &[f32], w: usize, h: usize, r: &Reference, start: Placement) -> (Solution, f64) {
    refine_scan(lum, w, h, r, start, false)
}

/// Like [`refine`]; with `scan`, the disk's centre and radius are kept and every turn (both
/// mirrors) is tried first, so the starting turn doesn't matter.
pub fn refine_scan(lum: &[f32], w: usize, h: usize, r: &Reference, start: Placement, scan: bool) -> (Solution, f64) {
    let fine = template(lum, w, h, 240);
    let mid = template(lum, w, h, 120);
    let k = fine.factor;
    let (tcx, tcy) = ((fine.w as f64 - 1.0) / 2.0, (fine.h as f64 - 1.0) / 2.0);
    // The template's centre in full pixels, from the disk's centre.
    let (fx, fy) = ((tcx + 0.5) * k - 0.5, (tcy + 0.5) * k - 0.5);
    let (bx, by) = start.orientation.body_offset(fx - start.cx as f64, fy - start.cy as f64);
    let h0 = Hyp { u: bx / start.r as f64, v: by / start.r as f64, rho: start.r as f64 / k, angle: start.orientation.angle as f64 * RAD, mirrored: start.orientation.mirrored };
    let before = score(&fine, r, &h0).0;
    if scan {
        // Hypotheses on the disk found: for each turn the template's centre sits at a different place.
        let (mfx, mfy) = (((mid.w as f64 - 1.0) / 2.0 + 0.5) * mid.factor - 0.5, ((mid.h as f64 - 1.0) / 2.0 + 0.5) * mid.factor - 0.5);
        let mut scored: Vec<(f64, Hyp)> = Vec::new();
        for mirrored in [false, true] {
            for a in 0..180 {
                let angle = a as f64 * 2.0;
                let or = Orientation { angle: angle as f32, mirrored, scale: 1.0 };
                let (bx, by) = or.body_offset(mfx - start.cx as f64, mfy - start.cy as f64);
                let hyp = Hyp { u: bx / start.r as f64, v: by / start.r as f64, rho: start.r as f64 / mid.factor, angle: angle * RAD, mirrored };
                let (s, cover) = score(&mid, r, &hyp);
                if cover > 0.5 {
                    scored.push((s, hyp));
                }
            }
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut starts: Vec<(f64, Hyp)> = Vec::new();
        for (s, hy) in scored {
            if starts.iter().all(|(_, k)| k.mirrored != hy.mirrored || ((k.angle - hy.angle + 3.0 * std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI).abs() > 12.0 * RAD) {
                starts.push((s, hy));
                if starts.len() >= 4 {
                    break;
                }
            }
        }
        let mut best: Option<(f64, Hyp)> = None;
        for (_, hy) in starts {
            let hy_fine = Hyp { rho: start.r as f64 / fine.factor, ..hy };
            // The template centre differs between the two sizes by a fraction of a pixel; polish absorbs it.
            let (s, p) = polish(&fine, r, hy_fine);
            if best.is_none_or(|(b, _)| s > b) {
                best = Some((s, p));
            }
        }
        let (s, p) = best.unwrap_or_else(|| polish(&fine, r, h0));
        return (Solution { placement: to_placement(&fine, p), score: s, runner_up: 0.0 }, before);
    }
    let (s, best) = polish(&fine, r, h0);
    (Solution { placement: to_placement(&fine, best), score: s, runner_up: 0.0 }, before)
}

/// A fit: disk centre `(u, v)` in the disk frame under the template's centre, the disk radius `rho`
/// in template pixels and the orientation.
#[derive(Clone, Copy, Debug)]
struct Hyp {
    u: f64,
    v: f64,
    rho: f64,
    angle: f64,
    mirrored: bool,
}

impl Hyp {
    fn orientation(&self) -> Orientation {
        Orientation { angle: (self.angle / RAD) as f32, mirrored: self.mirrored, scale: 1.0 }
    }
}

/// Normalised correlation of the template with the reference under `hyp`, and the share of the
/// template's valid pixels that fell on the lit disk.
fn score(t: &Template, r: &Reference, hyp: &Hyp) -> (f64, f64) {
    let level = r.level_for(hyp.rho);
    let n = level.n as f64;
    let or = hyp.orientation();
    let (cx, cy) = ((t.w as f64 - 1.0) / 2.0, (t.h as f64 - 1.0) / 2.0);
    let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut cnt, mut valid_total) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0f64, 0.0f64);
    for py in 0..t.h {
        for px in 0..t.w {
            if !t.valid[py * t.w + px] {
                continue;
            }
            valid_total += 1.0;
            let (bx, by) = or.body_offset(px as f64 - cx, py as f64 - cy);
            let (x, y) = (hyp.u + bx / hyp.rho, hyp.v + by / hyp.rho);
            if x * x + y * y >= 0.96 {
                continue;
            }
            let (fx, fy) = ((x + 1.0) / 2.0 * n - 0.5, (1.0 - y) / 2.0 * n - 0.5);
            let (x0, y0) = (fx.floor(), fy.floor());
            if x0 < 0.0 || y0 < 0.0 || x0 + 1.0 >= n || y0 + 1.0 >= n {
                continue;
            }
            let (ix, iy) = (x0 as usize, y0 as usize);
            let nn = level.n;
            if !(level.mask[iy * nn + ix] && level.mask[iy * nn + ix + 1] && level.mask[(iy + 1) * nn + ix] && level.mask[(iy + 1) * nn + ix + 1]) {
                continue;
            }
            let (ax, ay) = ((fx - x0) as f32, (fy - y0) as f32);
            let d = &level.data;
            let rv = (d[iy * nn + ix] * (1.0 - ax) + d[iy * nn + ix + 1] * ax) * (1.0 - ay) + (d[(iy + 1) * nn + ix] * (1.0 - ax) + d[(iy + 1) * nn + ix + 1] * ax) * ay;
            let (a, b) = (rv as f64, t.data[py * t.w + px] as f64);
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
            cnt += 1.0;
        }
    }
    if cnt < 30.0 || valid_total < 30.0 {
        return (-1.0, 0.0);
    }
    let cov = sab / cnt - sa * sb / cnt / cnt;
    let (va, vb) = (saa / cnt - (sa / cnt).powi(2), sbb / cnt - (sb / cnt).powi(2));
    if va <= 1e-9 || vb <= 1e-9 {
        return (-1.0, 0.0);
    }
    (cov / (va * vb).sqrt(), cnt / valid_total)
}

#[derive(Clone, Copy, Debug)]
pub struct Solution {
    pub placement: Placement,
    /// Correlation of the best fit, and of the best one far from it.
    pub score: f64,
    pub runner_up: f64,
}

/// Fit `lum` (w×h) onto the reference. The disk of a full-disk picture is the easy case; a close-up
/// of a few hundred kilometres is the one this is for.
pub fn solve(lum: &[f32], w: usize, h: usize, r: &Reference, log: bool) -> Option<Solution> {
    let coarse = template(lum, w, h, 80);
    let fine = template(lum, w, h, 240);
    let km_per_px_coarse = |fw_km: f64| fw_km / coarse.w as f64;
    // Field widths (km) tried: from a crater's neighbourhood to the whole disk.
    let mut widths = Vec::new();
    let mut fw = 90.0f64;
    while fw < 3600.0 {
        widths.push(fw);
        fw *= 1.3;
    }
    // Places on the disk (disk frame, unit radius).
    let mut places = Vec::new();
    let steps = 44;
    for j in 0..steps {
        for i in 0..steps {
            let (u, v) = ((i as f64 + 0.5) / steps as f64 * 2.0 - 1.0, 1.0 - (j as f64 + 0.5) / steps as f64 * 2.0);
            if u * u + v * v < 0.97 {
                places.push((u, v));
            }
        }
    }
    // The light axis in the picture for each mirror: angle of the Sun's direction on the disk
    // carried by the orientation, minus the turn, must equal the measured axis (mod π).
    let mut jobs: Vec<(f64, f64, f64, f64, bool)> = Vec::new();
    for &mirrored in &[false, true] {
        let (ux, uy) = (if mirrored { -r.sun_axis.cos() } else { r.sun_axis.cos() }, -r.sun_axis.sin());
        let base = uy.atan2(ux);
        // image axis = base + turn  =>  turn = measured − base (mod π)
        let turn0 = (coarse.axis - base).rem_euclid(std::f64::consts::PI);
        for half in 0..2 {
            for k in -4..=4 {
                let angle = (turn0 + half as f64 * std::f64::consts::PI + k as f64 * 8.0 * RAD).rem_euclid(std::f64::consts::TAU);
                for &fw in &widths {
                    let rho = selene::RADIUS_KM / km_per_px_coarse(fw);
                    for &(u, v) in &places {
                        jobs.push((u, v, rho, angle, mirrored));
                    }
                }
            }
        }
    }
    if log {
        eprintln!("coarse: {} hypotheses, light axis {:.0}°, predicted base axes {:.0}° / {:.0}° (sun on disk {:.0}°)", jobs.len(), coarse.axis / RAD, (-r.sun_axis.sin()).atan2(r.sun_axis.cos()) / RAD, (-r.sun_axis.sin()).atan2(-r.sun_axis.cos()) / RAD, r.sun_axis / RAD);
        if let Ok(spec) = std::env::var("LOLA_TRUE") {
            let q: Vec<f64> = spec.split(',').map(|v| v.parse().unwrap()).collect();
            let hyp = Hyp { u: q[0], v: q[1], rho: q[2] / coarse.factor, angle: q[3] * RAD, mirrored: q[4] > 0.5 };
            eprintln!("  truth scores {:?}", score(&coarse, r, &hyp));
            let hf = Hyp { rho: q[2] / fine.factor, ..hyp };
            eprintln!("  truth fine {:?} (factor {:.2}, fine {}x{})", score(&fine, r, &hf), fine.factor, fine.w, fine.h);
            for d in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                let hd = Hyp { u: hf.u + d / hf.rho, ..hf };
                eprintln!("    shift {d:+.1} px: {:.3}", score(&fine, r, &hd).0);
            }
        }
    }
    let mut scored: Vec<(f64, Hyp)> = jobs
        .par_iter()
        .filter_map(|&(u, v, rho, angle, mirrored)| {
            let hyp = Hyp { u, v, rho, angle, mirrored };
            let (s, cover) = score(&coarse, r, &hyp);
            (cover > 0.7).then_some((s, hyp))
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    // Distinct candidates.
    let mut cands: Vec<(f64, Hyp)> = Vec::new();
    for (s, h) in scored {
        let near = cands.iter().any(|(_, k)| {
            k.mirrored == h.mirrored && (k.rho / h.rho).ln().abs() < 0.5 && ((k.u - h.u).powi(2) + (k.v - h.v).powi(2)).sqrt() < 0.12 + 60.0 / k.rho.max(60.0) * 0.0 && ((k.angle - h.angle + 3.0 * std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI).abs() < 25.0 * RAD
        });
        if !near {
            cands.push((s, h));
            if cands.len() >= 12 {
                break;
            }
        }
    }
    if log {
        for (s, h) in &cands {
            eprintln!("  coarse {s:.3}: u={:.2} v={:.2} fw={:.0} km angle={:.0} mirror={}", h.u, h.v, 2.0 * selene::RADIUS_KM * 0.0 + coarse.w as f64 * selene::RADIUS_KM / h.rho, h.angle / RAD, h.mirrored);
        }
    }
    // Polish: the fine template, pattern search on (u, v, rho, angle).
    let ratio = coarse.factor / fine.factor; // template pixels fine/coarse
    let mut polished: Vec<(f64, Hyp)> = cands.iter().map(|(_, h0)| polish(&fine, r, Hyp { rho: h0.rho * ratio, ..*h0 })).collect();
    polished.sort_by(|a, b| b.0.total_cmp(&a.0));
    if log {
        for (s, h) in &polished {
            eprintln!("  fine {s:.3}: u={:.3} v={:.3} rho={:.0} angle={:.1} mirror={}", h.u, h.v, h.rho, h.angle / RAD, h.mirrored);
        }
    }
    let (best_score, best) = *polished.first()?;
    let runner_up = polished.iter().skip(1).find(|(_, h)| h.mirrored != best.mirrored || (h.u - best.u).hypot(h.v - best.v) > 0.1 || (h.rho / best.rho).ln().abs() > 0.4).map_or(0.0, |(s, _)| *s);
    Some(Solution { placement: to_placement(&fine, best), score: best_score, runner_up })
}
