//! Deringing: removes the dark and bright bands that sharpening draws along sharp edges — a
//! planet's limb, Saturn's rings, the rings crossing in front of the globe.
//!
//! Deconvolution and wavelet sharpening amplify the Gibbs oscillations at a strong edge: a
//! bright rind on its bright side, a dark band next to it, a halo in the sky. Diffraction (the
//! first Airy ring) and miscollimation (an asymmetric ring) put a faint ring in the data that
//! sharpening then strengthens.
//!
//! Two corrections:
//! - **Halos along sharp edges**: the anti-ringing clamp of image resampling, the same idea as
//!   PixInsight's deringing: ringing creates values *outside the range the unsharpened image
//!   has around that pixel*, so the sharpened value is held within the local minimum/maximum
//!   of the reference over about the halo's width, plus a tolerance. It acts only where the
//!   reference has a sharp edge nearby (its local range is a good part of the subject's
//!   brightness): belts and fine detail, whose range is small, are left alone. That area is a
//!   mask ([`weight`]) the user can see and touch up with the brush; on Saturn it can be limited
//!   to the globe or to the rings.
//! - **Concentric rings inside the disk** (a planet, Saturn's globe), often on one side only:
//!   what sharpening changed is taken along arcs at each depth below the limb (the median over
//!   an angular sector — belts and spots cross the arc and don't survive it), and the dark
//!   bands of that radial profile are lifted, varying smoothly from sector to sector. The
//!   brightening of the limb, which is sharpening, stays.

use crate::Image;
use crate::color::{ColorMode, working_planes};
use crate::mask::{self, Ellipse, Stroke, foreground, main_body};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeringParams {
    /// 0 = off, 1 = values fully held within the local range.
    pub strength: f32,
    /// Width of the halos (px): how far from a sharp edge values are held.
    pub radius: u32,
    /// Allowed excess beyond the local range, as a fraction of it (genuine detail recovered
    /// by deconvolution does go a little beyond).
    pub tolerance: f32,
    /// How sharp an edge must be to be treated: its local range as a fraction of the
    /// subject's brightness, from where the correction starts.
    pub edges: f32,
    /// Where the halos are treated.
    pub region: DeringRegion,
    /// Brush strokes adding to or removing from the area.
    pub strokes: Vec<Stroke>,
    /// How deep below the limb concentric rings are looked for, as a fraction of the radius
    /// (0 = not looked for).
    pub depth: f32,
    /// How much of the bright rind along the limb goes (0‥1): its brightening down to the
    /// level deeper inside.
    pub rind: f32,
}

/// Where halos are treated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeringRegion {
    /// Every sharp edge of the image.
    #[default]
    Subject,
    /// Saturn's globe (with the rings crossing it) and its limb.
    Globe,
    /// Saturn's rings outside the globe.
    Rings,
}

/// Everything the area depends on (strength and tolerance don't), to cache it.
#[derive(Clone, Debug, PartialEq)]
pub struct AreaKey {
    radius: u32,
    edges: f32,
    region: DeringRegion,
    strokes: Vec<Stroke>,
}

impl Default for DeringParams {
    fn default() -> Self {
        Self { strength: 0.8, radius: 4, tolerance: 0.1, edges: 0.1, region: DeringRegion::Subject, strokes: Vec::new(), depth: 0.35, rind: 0.5 }
    }
}

impl DeringParams {
    /// Halo width from the blur σ (px): the bands of a sharpened edge lie within a few σ of it,
    /// and wavelet sharpening's coarser layers reach a few pixels further.
    pub fn for_blur(sigma: f32) -> Self {
        Self { radius: (2.5 * sigma + 2.0).round().clamp(2.0, 12.0) as u32, ..Default::default() }
    }

    pub fn is_neutral(&self) -> bool {
        self.strength <= 0.0
    }

    /// Half-size of the neighbourhood whose range a value is held within: a fraction of the
    /// halo's width, so the range stays that of the pixel's own side of the edge.
    pub fn window(&self) -> usize {
        (self.radius as usize).div_ceil(4).max(1)
    }

    pub fn area_key(&self) -> AreaKey {
        AreaKey { radius: self.radius, edges: self.edges, region: self.region, strokes: self.strokes.clone() }
    }
}

/// `sharp` is the processed image, `reference` the same region before sharpening. `area` is
/// [`weight`] for this region and `globe` the disk ([`find_globe`]) in its coordinates, when
/// already computed on the whole image (the app does, so a ROI is treated like the rest).
pub fn dering(sharp: &Image, reference: &Image, p: &DeringParams, area: Option<&[f32]>, globe: Option<Ellipse>) -> Image {
    let (w, h) = (sharp.width(), sharp.height());
    if p.is_neutral() || (reference.width(), reference.height()) != (w, h) {
        return sharp.clone();
    }
    let computed;
    let weight = match area.filter(|a| a.len() == w * h) {
        Some(a) => a,
        None => {
            computed = weight(reference, p);
            &computed
        }
    };
    // Inside the disk: concentric rings, removed as such.
    let globe = if p.region == DeringRegion::Rings || p.depth <= 0.0 { None } else { globe.or_else(|| find_globe(reference, p.region)) };
    let rings = globe.map(|g| {
        let s = working_planes(sharp, ColorMode::Luminance).swap_remove(0);
        let r = working_planes(reference, ColorMode::Luminance).swap_remove(0);
        concentric_rings(&s, &r, w, h, &g, p.depth, p.rind)
    });
    // How much of the concentric correction applies: none on the limb, all of it a few pixels
    // in, so where Saturn's rings cross the limb no seam is left.
    let lift = |i: usize| match (globe, &rings) {
        (Some(g), Some(r)) => {
            let depth = (1.0 - g.radius((i % w) as f32 + 0.5, (i / w) as f32 + 0.5)) * g.rx.min(g.ry);
            (depth / BLEND_PX).clamp(0.0, 1.0) * r[i]
        }
        _ => 0.0,
    };
    let r = p.window();
    let channels = sharp
        .channels()
        .par_iter()
        .enumerate()
        .map(|(c, plane)| {
            // A mono reference (LRGB) serves every channel.
            let refp = &reference.channels()[c.min(reference.channels().len() - 1)];
            let (lo, hi) = min_max(refp, w, h, r);
            plane
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let s = s - p.strength * lift(i);
                    // The area (sharp edges, with the brush strokes) says where halos are held.
                    let k = p.strength * weight[i];
                    if k <= 0.0 {
                        return s;
                    }
                    let margin = p.tolerance * (hi[i] - lo[i]);
                    let held = s.clamp(lo[i] - margin, hi[i] + margin);
                    s + k * (held - s)
                })
                .collect()
        })
        .collect();
    Image::new(w, h, channels, sharp.source_bit_depth)
}

/// The disk rings are removed from inside: Saturn's globe (from the rings' geometry), or the
/// planet's outline as an ellipse. `None` for the rings, or when the subject isn't a whole disk
/// (a close-up running off the frame).
pub fn find_globe(image: &Image, region: DeringRegion) -> Option<Ellipse> {
    match region {
        DeringRegion::Rings => None,
        DeringRegion::Globe => {
            let globe = mask::saturn_globe(image)?;
            let lum = working_planes(image, ColorMode::Luminance).swap_remove(0);
            Some(fit_limb(&lum, image.width(), image.height(), globe))
        }
        DeringRegion::Subject => {
            let (w, h) = (image.width(), image.height());
            let lum = working_planes(image, ColorMode::Luminance).swap_remove(0);
            let mut body = main_body(&foreground(&lum, w, h), w, h);
            // The subject mask reaches into the blurred limb: the disk ends at half its
            // brightness.
            let mut inside: Vec<f32> = body.iter().zip(&lum).filter(|(b, _)| **b).map(|(_, &v)| v).collect();
            if inside.len() < 100 {
                return None;
            }
            let mid = inside.len() / 2;
            let half = 0.5 * *inside.select_nth_unstable_by(mid, f32::total_cmp).1;
            body.iter_mut().zip(&lum).for_each(|(b, &v)| *b &= v > half);
            let (mut n, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
            for (i, _) in body.iter().enumerate().filter(|(_, b)| **b) {
                (n, sx, sy) = (n + 1.0, sx + (i % w) as f64 + 0.5, sy + (i / w) as f64 + 0.5);
            }
            if n < 100.0 {
                return None;
            }
            let (cx, cy) = (sx / n, sy / n);
            let (mut xx, mut yy, mut xy) = (0.0f64, 0.0f64, 0.0f64);
            for (i, _) in body.iter().enumerate().filter(|(_, b)| **b) {
                let (dx, dy) = ((i % w) as f64 + 0.5 - cx, (i / w) as f64 + 0.5 - cy);
                (xx, yy, xy) = (xx + dx * dx, yy + dy * dy, xy + dx * dy);
            }
            let (xx, yy, xy) = (xx / n, yy / n, xy / n);
            let (mean, diff) = ((xx + yy) / 2.0, (((xx - yy) / 2.0).powi(2) + xy * xy).sqrt());
            // A filled ellipse with semi-axes a, b has variances a²/4 and b²/4.
            let (a, b) = (2.0 * (mean + diff).sqrt(), 2.0 * (mean - diff).max(0.0).sqrt());
            let ellipse = Ellipse { cx: cx as f32, cy: cy as f32, rx: a as f32, ry: b as f32, angle: (0.5 * (2.0 * xy).atan2(xx - yy)) as f32 };
            // Only a whole disk: the ellipse must match the subject.
            let mismatch = body.iter().enumerate().filter(|(i, b)| **b != (ellipse.radius((i % w) as f32 + 0.5, (i / w) as f32 + 0.5) < 1.0)).count() as f64;
            (mismatch / n < 0.08 && a > 8.0).then_some(ellipse)
        }
    }
}

/// `globe` scaled so its outline sits where the disk falls to half its brightness: the median
/// over rays that reach the sky (the others cross the rings).
fn fit_limb(lum: &[f32], w: usize, h: usize, globe: Ellipse) -> Ellipse {
    let (sin, cos) = globe.angle.sin_cos();
    let at = |rho: f32, phi: f32| -> Option<f32> {
        let (u, v) = (rho * globe.rx * phi.cos(), rho * globe.ry * phi.sin());
        let (x, y) = (globe.cx + u * cos - v * sin, globe.cy + u * sin + v * cos);
        (x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h).then(|| lum[y as usize * w + x as usize])
    };
    let mut found: Vec<f32> = (0..180)
        .filter_map(|k| {
            let phi = k as f32 * std::f32::consts::TAU / 180.0;
            let mut inner: Vec<f32> = (0..30).filter_map(|j| at(0.5 + 0.01 * j as f32, phi)).collect();
            if inner.len() < 30 {
                return None;
            }
            let half = 0.5 * *inner.select_nth_unstable_by(15, f32::total_cmp).1;
            // Beyond the limb this ray must reach the sky.
            if (0..10).any(|j| at(1.25 + 0.02 * j as f32, phi).is_none_or(|v| v > 0.5 * half)) {
                return None;
            }
            (0..300).map(|j| 0.7 + 0.002 * j as f32).find(|&rho| at(rho, phi).is_some_and(|v| v < half))
        })
        .collect();
    if found.len() < 20 {
        return globe;
    }
    let mid = found.len() / 2;
    let scale = *found.select_nth_unstable_by(mid, f32::total_cmp).1;
    Ellipse { rx: globe.rx * scale, ry: globe.ry * scale, ..globe }
}

/// Pixels below the limb over which the concentric correction comes in.
const BLEND_PX: f32 = 3.0;

/// Angular sectors the limb is divided into.
const SECTORS: usize = 36;
/// Sectors on each side whose pixels join a sector's median: an arc of 70°, long enough that
/// belts and spots cross it while rings run along it.
const ARC: i32 = 3;

/// How much of what sharpening changed (`sharp` − `reference`, luminance) is concentric dark
/// rings: the median change along an arc around each sector at each depth below the limb
/// (1 px steps, down to `depth` of the radius) gives a radial profile per sector; its dips
/// and whatever falls below the reference are the rings. Smoothed across neighbouring
/// sectors, faded out towards the deepest band; zero outside the disk and below that depth.
/// `rind`: how much of the bright rind along the limb also counts (0‥1).
pub fn concentric_rings(sharp: &[f32], reference: &[f32], w: usize, h: usize, globe: &Ellipse, depth: f32, rind: f32) -> Vec<f32> {
    let max_depth = (depth.clamp(0.05, 0.9) * globe.rx.min(globe.ry)).max(3.0);
    let bins = max_depth.ceil() as usize;
    let (sin, cos) = globe.angle.sin_cos();
    // Sector (fractional) and depth (px) of a pixel, if it lies in the band.
    let locate = |x: usize, y: usize| -> Option<(f32, f32)> {
        let (dx, dy) = (x as f32 + 0.5 - globe.cx, y as f32 + 0.5 - globe.cy);
        let (u, v) = (dx * cos + dy * sin, -dx * sin + dy * cos);
        let rho = ((u / globe.rx).powi(2) + (v / globe.ry).powi(2)).sqrt();
        if !(0.05..1.0).contains(&rho) {
            return None;
        }
        let depth_px = (1.0 - rho) * u.hypot(v) / rho;
        if depth_px >= max_depth {
            return None;
        }
        let theta = (v / globe.ry).atan2(u / globe.rx);
        Some(((theta + std::f32::consts::PI) / std::f32::consts::TAU * SECTORS as f32, depth_px))
    };
    // The band's bounding box.
    let reach_x = (globe.rx.max(globe.ry) + 2.0) as usize;
    let (x0, x1) = ((globe.cx as usize).saturating_sub(reach_x), (globe.cx as usize + reach_x).min(w));
    let (y0, y1) = ((globe.cy as usize).saturating_sub(reach_x), (globe.cy as usize + reach_x).min(h));
    let mut cells: Vec<Vec<f32>> = vec![Vec::new(); SECTORS * bins];
    for y in y0..y1 {
        for x in x0..x1 {
            if let Some((sector, d)) = locate(x, y) {
                let i = y * w + x;
                cells[(sector as usize % SECTORS) * bins + (d as usize).min(bins - 1)].push(sharp[i] - reference[i]);
            }
        }
    }
    let medians: Vec<Option<f32>> = (0..SECTORS * bins)
        .into_par_iter()
        .map(|k| {
            let (s, b) = (k / bins, k % bins);
            let mut arc: Vec<f32> = (-ARC..=ARC).flat_map(|o| &cells[(s as i32 + o).rem_euclid(SECTORS as i32) as usize * bins + b]).copied().collect();
            (arc.len() >= 5).then(|| {
                let mid = arc.len() / 2;
                *arc.select_nth_unstable_by(mid, f32::total_cmp).1
            })
        })
        .collect();
    // What of each sector's radial profile is ringing. Dark rings: the profile should be its
    // closing (narrow dips filled) and never below the reference. The bright rind: a boost
    // near the limb above the level deeper in, which makes the plainer band behind it look
    // dark, is brought down towards that level by `rind`.
    let reach = (0.08 * globe.rx.min(globe.ry)).max(4.0) as usize;
    let mut artifacts: Vec<Option<f32>> = vec![None; SECTORS * bins];
    for s in 0..SECTORS {
        let d = &medians[s * bins..(s + 1) * bins];
        let window = |b: usize| d[b.saturating_sub(reach)..(b + reach + 1).min(bins)].iter().flatten().copied();
        let dilated: Vec<Option<f32>> = (0..bins).map(|b| window(b).reduce(f32::max)).collect();
        let closed: Vec<Option<f32>> = (0..bins).map(|b| d[b].map(|v| dilated[b.saturating_sub(reach)..(b + reach + 1).min(bins)].iter().flatten().copied().fold(v, f32::min).max(0.0))).collect();
        for b in 0..bins {
            if let (Some(v), Some(c)) = (d[b], closed[b]) {
                let deeper = closed[b..].iter().flatten().copied().fold(c, f32::min);
                artifacts[s * bins + b] = Some(v - (c - rind * (c - deeper)));
            }
        }
    }
    let medians = artifacts;
    // Smooth across sectors (rings change slowly around the limb), fade out at the deepest bins.
    let profile: Vec<f32> = (0..SECTORS * bins)
        .map(|k| {
            let (s, b) = (k / bins, k % bins);
            let (mut sum, mut weight) = (0.0f32, 0.0f32);
            for (offset, wgt) in [(-1i32, 1.0f32), (0, 2.0), (1, 1.0)] {
                let neighbour = (s as i32 + offset).rem_euclid(SECTORS as i32) as usize;
                if let Some(m) = medians[neighbour * bins + b] {
                    (sum, weight) = (sum + wgt * m, weight + wgt);
                }
            }
            let fade = ((bins as f32 - b as f32) / (0.25 * bins as f32)).clamp(0.0, 1.0);
            if weight > 0.0 { fade * sum / weight } else { 0.0 }
        })
        .collect();
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if !(y0..y1).contains(&y) {
            return;
        }
        for (x, o) in row.iter_mut().enumerate().take(x1).skip(x0) {
            if let Some((sector, d)) = locate(x, y) {
                // Bilinear in (sector, depth).
                let s0 = (sector - 0.5).floor();
                let t = sector - 0.5 - s0;
                let sa = (s0 as i32).rem_euclid(SECTORS as i32) as usize;
                let sb = (sa + 1) % SECTORS;
                // Bins are centred half a pixel in.
                let d = (d - 0.5).max(0.0);
                let b0 = (d.floor() as usize).min(bins - 1);
                let b1 = (b0 + 1).min(bins - 1);
                let u = d - d.floor();
                let at = |s: usize, b: usize| profile[s * bins + b];
                let near = at(sa, b0) * (1.0 - u) + at(sa, b1) * u;
                let far = at(sb, b0) * (1.0 - u) + at(sb, b1) * u;
                *o = near * (1.0 - t) + far * t;
            }
        }
    });
    out
}

/// Where sharpening left rings: how far (in brightness) the luminance of `sharp` lies outside
/// the range `reference` has around each pixel, with the same radius and tolerance as
/// deringing. Zero where there is nothing to correct.
pub fn ringing_map(sharp: &Image, reference: &Image, p: &DeringParams) -> Vec<f32> {
    let (w, h) = (sharp.width(), sharp.height());
    if (reference.width(), reference.height()) != (w, h) {
        return vec![0.0; w * h];
    }
    let s = working_planes(sharp, ColorMode::Luminance).swap_remove(0);
    let r = working_planes(reference, ColorMode::Luminance).swap_remove(0);
    let (lo, hi) = min_max(&r, w, h, p.window());
    s.par_iter()
        .enumerate()
        .map(|(i, &v)| {
            let margin = p.tolerance * (hi[i] - lo[i]);
            (lo[i] - margin - v).max(v - hi[i] - margin).max(0.0)
        })
        .collect()
}

/// Where halos are held (0‥1): near the reference's sharp edges, within the region, plus the
/// brush strokes.
pub fn weight(image: &Image, p: &DeringParams) -> Vec<f32> {
    let (w, h) = (image.width(), image.height());
    let lum = working_planes(image, ColorMode::Luminance).swap_remove(0);
    // An edge counts from a range of `edges` of the subject's brightness, fully from about
    // twice that, as far away as the halos reach.
    let (lo, hi) = min_max(&lum, w, h, p.radius.max(1) as usize);
    let stride = lum.len().div_ceil(1 << 16).max(1);
    let mut sample: Vec<f32> = lum.iter().step_by(stride).copied().collect();
    let n = sample.len();
    let bright = sample.select_nth_unstable_by((n * 995 / 1000).min(n - 1), f32::total_cmp).1.max(1e-3);
    let (from, to) = (p.edges.max(0.0), 2.0 * p.edges.max(0.0) + 0.05);
    let globe = match p.region {
        DeringRegion::Subject => None,
        _ => find_globe(image, DeringRegion::Globe),
    };
    // The globe and its limb's halo (Globe), or what is outside it (Rings).
    let margin = globe.map_or(0.0, |g| (p.radius as f32 + 2.0) / g.rx.min(g.ry));
    let mut area: Vec<f32> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let inside = globe.map(|g| g.radius((i % w) as f32 + 0.5, (i / w) as f32 + 0.5));
            let allowed = match (p.region, inside) {
                (DeringRegion::Globe, Some(rho)) => ((1.0 + margin - rho) / margin.max(1e-3)).clamp(0.0, 1.0),
                (DeringRegion::Rings, Some(rho)) => ((rho - 1.0) / 0.03).clamp(0.0, 1.0),
                _ => 1.0,
            };
            let t = (((hi[i] - lo[i]) / bright - from) / (to - from)).clamp(0.0, 1.0);
            allowed * t * t * (3.0 - 2.0 * t)
        })
        .collect();
    for stroke in &p.strokes {
        mask::paint(&mut area, w, h, stroke);
    }
    area
}

/// Minimum and maximum over the (2r+1)² square around each pixel. Both passes combine whole
/// shifted rows element by element (no transposes, vectorisable).
pub(crate) fn min_max(plane: &[f32], w: usize, h: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    if let Some(out) = crate::gpu::for_plane(w, h, crate::gpu::Work::Light).and_then(|g| g.min_max(plane, w, h, r).ok()) {
        return out;
    }
    min_max_cpu(plane, w, h, r)
}

pub(crate) fn min_max_cpu(plane: &[f32], w: usize, h: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut row_min, mut row_max) = (plane.to_vec(), plane.to_vec());
    row_min.par_chunks_mut(w).zip(row_max.par_chunks_mut(w)).zip(plane.par_chunks(w)).for_each(|((lo, hi), row)| {
        for k in 1..=r.min(w - 1) {
            for (x, &v) in row[..w - k].iter().enumerate() {
                // row[x] seen from x + k, and row[x + k] seen from x.
                let (a, b) = (x + k, x);
                lo[a] = lo[a].min(v);
                hi[a] = hi[a].max(v);
                lo[b] = lo[b].min(row[x + k]);
                hi[b] = hi[b].max(row[x + k]);
            }
        }
    });
    let (mut lo, mut hi) = (vec![0.0f32; w * h], vec![0.0f32; w * h]);
    lo.par_chunks_mut(w).zip(hi.par_chunks_mut(w)).enumerate().for_each(|(y, (lo, hi))| {
        let (y0, y1) = (y.saturating_sub(r), (y + r).min(h - 1));
        lo.copy_from_slice(&row_min[y0 * w..(y0 + 1) * w]);
        hi.copy_from_slice(&row_max[y0 * w..(y0 + 1) * w]);
        for yy in y0 + 1..=y1 {
            lo.iter_mut().zip(&row_min[yy * w..(yy + 1) * w]).for_each(|(o, &v)| *o = o.min(v));
            hi.iter_mut().zip(&row_max[yy * w..(yy + 1) * w]).for_each(|(o, &v)| *o = o.max(v));
        }
    });
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::FftConvolver;
    use crate::psf::Psf;

    const N: usize = 160;
    const R: f32 = 50.0;

    /// A blurred, slightly limb-darkened disk, and an over-sharpened copy that rings.
    fn disk() -> (Image, Image) {
        let c = N as f32 / 2.0;
        let plane: Vec<f32> = (0..N * N)
            .map(|i| {
                let d = (((i % N) as f32 - c).powi(2) + ((i / N) as f32 - c).powi(2)).sqrt() / R;
                if d < 1.0 { 0.7 * (1.0 - d * d).powf(0.15) } else { 0.0 }
            })
            .collect();
        let blur = |p: &[f32], s: f32| FftConvolver::new(N, N, &Psf::Gaussian { sigma: s }.kernel()).convolve(p);
        let reference = blur(&plane, 2.0);
        // Strong unsharp mask: overshoot on the edge, a dark ring inside, a halo outside.
        let soft = blur(&reference, 3.0);
        let sharp: Vec<f32> = reference.iter().zip(&soft).map(|(r, s)| r + 2.5 * (r - s)).collect();
        (Image::new(N, N, vec![sharp], 16), Image::new(N, N, vec![reference], 16))
    }

    /// Along the row through the centre: (how far the brightest point near the limb rises
    /// above the reference's, largest excursion in the sky, centre value).
    fn profile(img: &Image, reference: &Image) -> (f32, f32, f32) {
        let row = N / 2;
        let at = |im: &Image, x: usize| im.channels()[0][row * N + x];
        let c = N / 2;
        let limb = c + R as usize;
        let peak = |im: &Image| (limb - 12..limb).map(|x| at(im, x)).fold(f32::MIN, f32::max);
        let overshoot = peak(img) - peak(reference);
        let sky = (limb + 3..N - 4).map(|x| at(img, x).abs()).fold(f32::MIN, f32::max);
        (overshoot, sky, at(img, c))
    }

    #[test]
    fn removes_the_rings_and_keeps_the_disk() {
        let (sharp, reference) = disk();
        let p = DeringParams::for_blur(2.0);
        let out = dering(&sharp, &reference, &DeringParams { strength: 1.0, ..p }, None, None);
        let (over_before, sky_before, centre_before) = profile(&sharp, &reference);
        let (over_after, sky_after, centre_after) = profile(&out, &reference);
        assert!(over_before > 0.03 && sky_before > 0.02, "the test image must ring: {over_before} {sky_before}");
        assert!(over_after < 0.5 * over_before, "bright rind {over_before} → {over_after}");
        assert!(sky_after < 0.3 * sky_before, "halo {sky_before} → {sky_after}");
        assert_eq!(centre_after, centre_before, "the inside of the disk is not touched");
    }

    /// Richardson–Lucy with a PSF 40 % too large rings at the limb; deringing removes most
    /// of it.
    #[test]
    fn tames_richardson_lucy_ringing() {
        use crate::deconv::{DeconvParams, deconvolve_plane};
        let (_, reference) = disk();
        let p = DeconvParams { psf: Psf::Gaussian { sigma: 2.8 }, iterations: 40, tv_lambda: 0.0, damping: 0.0, ..Default::default() };
        let plane = deconvolve_plane(&reference.channels()[0], N, N, &p, &|| false, &|_| {}).unwrap();
        let sharp = Image::new(N, N, vec![plane], 16);
        let out = dering(&sharp, &reference, &DeringParams { strength: 1.0, ..DeringParams::for_blur(2.0) }, None, None);
        let (over_before, sky_before, _) = profile(&sharp, &reference);
        let (over_after, sky_after, _) = profile(&out, &reference);
        assert!(over_after < 0.5 * over_before.max(1e-3), "rind {over_before} → {over_after}");
        assert!(sky_after <= sky_before, "halo {sky_before} → {sky_after}");
    }

    /// The ringing seen on real Saturns: a dark band concentric with the limb, a few pixels
    /// inside, on one side only. It goes; the belts sharpening brought out stay.
    #[test]
    fn removes_a_one_sided_concentric_band_and_keeps_the_belts() {
        let c = N as f32 / 2.0;
        let polar = |i: usize| {
            let (dx, dy) = ((i % N) as f32 + 0.5 - c, (i / N) as f32 + 0.5 - c);
            (dx.hypot(dy), dy.atan2(dx), dy)
        };
        let belts = |dy: f32| 0.05 * (dy / 7.0).sin();
        let reference: Vec<f32> = (0..N * N).map(|i| if polar(i).0 < R { 0.6 + belts(polar(i).2) } else { 0.0 }).collect();
        // The band: about 3 px wide, 8 px below the limb, strongest on the left and gone by the
        // top and bottom.
        let band = |i: usize| {
            let (r, theta, _) = polar(i);
            let side = (-theta.cos()).max(0.0);
            -0.08 * side * (-((r - (R - 8.0)) / 1.5).powi(2)).exp()
        };
        let sharp: Vec<f32> = (0..N * N).map(|i| if polar(i).0 < R { reference[i] + belts(polar(i).2) + band(i) } else { 0.0 }).collect();
        let (sharp, reference) = (Image::new(N, N, vec![sharp], 16), Image::new(N, N, vec![reference], 16));
        let out = dering(&sharp, &reference, &DeringParams { strength: 1.0, rind: 0.0, ..Default::default() }, None, None);
        let (out, sharp, reference) = (&out.channels()[0], &sharp.channels()[0], &reference.channels()[0]);
        let (mut band_before, mut band_after, mut belt_before, mut belt_after) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for i in 0..N * N {
            let (r, theta, _) = polar(i);
            if band(i) < -0.04 {
                band_before += (sharp[i] - reference[i] - belts(polar(i).2)).abs();
                band_after += (out[i] - reference[i] - belts(polar(i).2)).abs();
            } else if r < R - 2.0 && theta.cos() < -0.7 {
                // The belts on the band's side, where they cross the limb's arcs.
                belt_before += (sharp[i] - reference[i]).abs();
                belt_after += (out[i] - reference[i]).abs();
            }
        }
        assert!(band_after < 0.3 * band_before, "band {band_before} → {band_after}");
        assert!(belt_after > 0.8 * belt_before, "belts {belt_before} → {belt_after}");
    }

    #[test]
    fn finds_the_disk() {
        let (_, reference) = disk();
        let g = find_globe(&reference, DeringRegion::Subject).expect("a whole disk");
        assert!((g.cx - N as f32 / 2.0).abs() < 1.0 && (g.cy - N as f32 / 2.0).abs() < 1.0, "{g:?}");
        assert!((g.rx - R).abs() < 2.5 && (g.ry - R).abs() < 2.5, "{g:?}");
    }

    #[test]
    fn min_max_matches_brute_force() {
        let (w, h, r) = (23, 17, 2);
        let plane: Vec<f32> = (0..w * h).map(|i| ((i * 7919) % 101) as f32).collect();
        let (lo, hi) = min_max(&plane, w, h, r);
        for y in 0..h {
            for x in 0..w {
                let window = (y.saturating_sub(r)..=(y + r).min(h - 1)).flat_map(|yy| (x.saturating_sub(r)..=(x + r).min(w - 1)).map(move |xx| (xx, yy)));
                let values: Vec<f32> = window.map(|(xx, yy)| plane[yy * w + xx]).collect();
                assert_eq!(lo[y * w + x], values.iter().copied().fold(f32::MAX, f32::min));
                assert_eq!(hi[y * w + x], values.iter().copied().fold(f32::MIN, f32::max));
            }
        }
    }

    /// On Saturn the area can follow the globe's limb and leave the rings alone.
    #[test]
    fn globe_area_leaves_the_rings_out() {
        let (w, h, r) = (700usize, 400usize, 100.0f32);
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 + 0.5 - 350.0, (i / w) as f32 + 0.5 - 200.0);
                let globe = (x / r).powi(2) + (y / (0.9 * r)).powi(2) < 1.0;
                let e = ((x / (2.27 * r)).powi(2) + (y / (0.6 * r)).powi(2)).sqrt();
                if globe {
                    0.7
                } else if (0.68..1.0).contains(&e) {
                    0.5
                } else {
                    0.0
                }
            })
            .collect();
        let img = Image::new(w, h, vec![plane], 16);
        let p = DeringParams { region: DeringRegion::Globe, ..DeringParams::for_blur(1.5) };
        let area = weight(&img, &p);
        let at = |x: usize, y: usize| area[y * w + x];
        assert!(at(350, 200 - 90) > 0.5, "on the globe's limb");
        assert!(at(350 + 227, 200) < 0.05, "out on the rings");
        assert!(at(350, 200) < 0.05, "centre of the globe, no edge there");
        let rings = weight(&img, &DeringParams { region: DeringRegion::Rings, ..p });
        assert!(rings[200 * w + 350 + 226] > 0.5, "rings area follows their outer edge");
    }

    /// The dark band sharpening leaves next to a bright edge crossing the disk (Saturn's rings
    /// in front of the globe) is held back; a soft belt, without a sharp edge, is not touched.
    #[test]
    fn holds_the_halo_beside_an_edge_and_leaves_belts() {
        let (w, h) = (160usize, 120usize);
        let plane: Vec<f32> = (0..w * h)
            .map(|i| {
                let y = (i / w) as f32;
                let ring = if (55.0..62.0).contains(&y) { 0.25 } else { 0.0 };
                let belt = 0.04 * (-((y - 95.0) / 6.0).powi(2)).exp();
                0.5 + ring - belt
            })
            .collect();
        let blur = |p: &[f32], s: f32| FftConvolver::new(w, h, &Psf::Gaussian { sigma: s }.kernel()).convolve(p);
        let reference = blur(&plane, 1.5);
        let soft = blur(&reference, 3.0);
        let sharp: Vec<f32> = reference.iter().zip(&soft).map(|(r, s)| r + 2.0 * (r - s)).collect();
        let (sharp, reference) = (Image::new(w, h, vec![sharp], 16), Image::new(w, h, vec![reference], 16));
        let out = dering(&sharp, &reference, &DeringParams { strength: 1.0, depth: 0.0, ..DeringParams::for_blur(1.5) }, None, None);
        let column = |img: &Image, y: usize| img.channels()[0][y * w + w / 2];
        // Below the ring: how far under the globe's level the darkest point goes.
        let under = |img: &Image| (62..80).map(|y| 0.5 - column(img, y)).fold(0.0f32, f32::max);
        assert!(under(&sharp) > 0.03, "the test image must ring: {}", under(&sharp));
        assert!(under(&out) < 0.5 * under(&sharp), "halo {} → {}", under(&sharp), under(&out));
        let belt = |img: &Image| column(img, 80) - column(img, 95);
        assert!(belt(&out) > 0.95 * belt(&sharp), "belt {} → {}", belt(&sharp), belt(&out));
    }

    #[test]
    fn ringing_map_finds_the_rings_and_not_the_inside() {
        let (sharp, reference) = disk();
        let map = ringing_map(&sharp, &reference, &DeringParams::for_blur(2.0));
        let row = N / 2;
        let limb = N / 2 + R as usize;
        let near = (limb - 6..limb + 6).map(|x| map[row * N + x]).fold(0.0f32, f32::max);
        assert!(near > 0.02, "rings at the limb: {near}");
        assert!(map[row * N + N / 2] < 0.1 * near, "next to nothing in the middle of the disk: {}", map[row * N + N / 2]);
    }

    #[test]
    fn zero_strength_is_identity() {
        let (sharp, reference) = disk();
        let out = dering(&sharp, &reference, &DeringParams { strength: 0.0, ..Default::default() }, None, None);
        assert_eq!(out.channels(), sharp.channels());
    }
}
