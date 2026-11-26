//! Layer masks: which part of the image a layer's own settings apply to (0‥1 per pixel).

use crate::i18n::tl;
use crate::Image;
use crate::analysis::{downsample, otsu};
use crate::color::{ColorMode, working_planes};
use crate::wavelet::estimate_noise;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Outer edge of Saturn's A ring over the planet's equatorial radius.
const SATURN_RING_TO_GLOBE: f32 = 2.27;
/// Polar / equatorial radius of Saturn.
const SATURN_OBLATENESS: f32 = 0.90;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutoRegion {
    /// The planet, the lunar/solar disk, or whatever stands out from the sky.
    Subject,
    Background,
    SaturnGlobe,
    SaturnRings,
    /// Small bright bodies apart from the main planet (Galilean moons, Titan…).
    Moons,
}

impl AutoRegion {
    pub fn label(self) -> &'static str {
        match self {
            AutoRegion::Subject => tl("Soggetto"),
            AutoRegion::Background => tl("Sfondo"),
            AutoRegion::SaturnGlobe => tl("Globo di Saturno"),
            AutoRegion::SaturnRings => tl("Anelli di Saturno"),
            AutoRegion::Moons => tl("Lune"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MaskShape {
    Auto(AutoRegion),
    /// Centre, semi-axes (pixels) and rotation (degrees).
    Ellipse {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        angle: f32,
    },
    Rectangle {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
    },
    /// Nothing: the mask is made only of brush strokes.
    Empty,
}

impl MaskShape {
    pub fn label(&self) -> &'static str {
        match self {
            MaskShape::Auto(r) => r.label(),
            MaskShape::Ellipse { .. } => tl("Ellisse"),
            MaskShape::Rectangle { .. } => tl("Rettangolo"),
            MaskShape::Empty => tl("Pennello"),
        }
    }

    /// Move a hand-drawn shape (e.g. after cropping); automatic regions are unaffected.
    pub fn translate(&mut self, dx: f32, dy: f32) {
        match self {
            MaskShape::Ellipse { cx, cy, .. } => (*cx, *cy) = (*cx + dx, *cy + dy),
            MaskShape::Rectangle { x, y, .. } => (*x, *y) = (*x + dx, *y + dy),
            MaskShape::Auto(_) | MaskShape::Empty => {}
        }
    }

    /// Carry a hand-drawn shape through a rigid motion of the image (see [`MaskSpec::remap`]).
    /// A rectangle becomes the box of its moved corners.
    pub fn remap(&mut self, f: &impl Fn([f32; 2]) -> [f32; 2]) {
        match self {
            MaskShape::Ellipse { cx, cy, rx, ry, angle } => {
                let (s, c) = angle.to_radians().sin_cos();
                let centre = f([*cx, *cy]);
                let along = |dx: f32, dy: f32, len: f32| {
                    let p = f([*cx + len * dx, *cy + len * dy]);
                    (p[0] - centre[0], p[1] - centre[1])
                };
                let (ax, ay) = along(c, s, *rx);
                let (bx, by) = along(-s, c, *ry);
                (*cx, *cy, *rx, *ry, *angle) = (centre[0], centre[1], ax.hypot(ay), bx.hypot(by), ay.atan2(ax).to_degrees());
            }
            MaskShape::Rectangle { x, y, w, h } => {
                let corners = [f([*x, *y]), f([*x + *w, *y]), f([*x, *y + *h]), f([*x + *w, *y + *h])];
                let min = |i: usize| corners.iter().map(|p| p[i]).fold(f32::INFINITY, f32::min);
                let max = |i: usize| corners.iter().map(|p| p[i]).fold(f32::NEG_INFINITY, f32::max);
                (*x, *y, *w, *h) = (min(0), min(1), max(0) - min(0), max(1) - min(1));
            }
            MaskShape::Auto(_) | MaskShape::Empty => {}
        }
    }
}

/// Move the points of `strokes` with `f` (a rigid motion: the brush radius stays).
pub fn remap_strokes(strokes: &mut [Stroke], f: &impl Fn([f32; 2]) -> [f32; 2]) {
    for p in strokes.iter_mut().flat_map(|s| s.points.iter_mut()) {
        *p = f(*p);
    }
}

/// A brush stroke painted over a mask, in image pixels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stroke {
    pub points: Vec<[f32; 2]>,
    pub radius: f32,
    /// 0 = soft edge across the whole radius, 1 = hard edge.
    pub hardness: f32,
    /// Remove from the mask instead of adding.
    pub erase: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskSpec {
    pub shape: MaskShape,
    /// Edge softness (Gaussian σ, pixels) of the shape.
    pub feather: f32,
    pub invert: bool,
    /// Brush strokes applied on top of the (feathered, inverted) shape.
    #[serde(default)]
    pub strokes: Vec<Stroke>,
}

impl MaskSpec {
    pub fn new(shape: MaskShape, feather: f32) -> Self {
        Self { shape, feather, invert: false, strokes: Vec::new() }
    }

    /// Move hand-drawn parts (shapes and strokes), e.g. after cropping.
    pub fn translate(&mut self, dx: f32, dy: f32) {
        self.shape.translate(dx, dy);
        for p in self.strokes.iter_mut().flat_map(|s| s.points.iter_mut()) {
            p[0] += dx;
            p[1] += dy;
        }
    }

    /// Carry hand-drawn parts through a turn, mirror or rotation of the image: `f` says where
    /// a point of the old image lies in the new one.
    pub fn remap(&mut self, f: &impl Fn([f32; 2]) -> [f32; 2]) {
        self.shape.remap(f);
        remap_strokes(&mut self.strokes, f);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ellipse {
    pub cx: f32,
    pub cy: f32,
    pub rx: f32,
    pub ry: f32,
    /// Radians.
    pub angle: f32,
}

impl Ellipse {
    /// Normalised radius: < 1 inside.
    pub fn radius(&self, x: f32, y: f32) -> f32 {
        let (s, c) = self.angle.sin_cos();
        let (dx, dy) = (x - self.cx, y - self.cy);
        let (u, v) = (dx * c + dy * s, -dx * s + dy * c);
        ((u / self.rx).powi(2) + (v / self.ry).powi(2)).sqrt()
    }
}

pub fn render(spec: &MaskSpec, img: &Image) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let mut mask = match spec.shape {
        MaskShape::Auto(region) => auto_region(img, region),
        MaskShape::Ellipse { cx, cy, rx, ry, angle } => {
            let e = Ellipse { cx, cy, rx: rx.max(1.0), ry: ry.max(1.0), angle: angle.to_radians() };
            raster(w, h, |x, y| (e.radius(x, y) < 1.0) as u8 as f32)
        }
        MaskShape::Rectangle { x, y, w: rw, h: rh } => raster(w, h, |px, py| ((x..x + rw).contains(&px) && (y..y + rh).contains(&py)) as u8 as f32),
        MaskShape::Empty => vec![0.0; w * h],
    };
    if spec.feather > 0.0 && spec.shape != MaskShape::Empty {
        mask = gaussian_approx(&mask, w, h, spec.feather);
    }
    if spec.invert {
        mask.par_iter_mut().for_each(|v| *v = 1.0 - *v);
    }
    for stroke in &spec.strokes {
        paint(&mut mask, w, h, stroke);
    }
    mask
}

/// Stamp a stroke: coverage is the brush profile at the distance from the stroke's path.
/// Coverage only falls with distance, so taking the strongest coverage segment by segment
/// (each over its own small box) equals using the distance to the whole path.
pub(crate) fn paint(mask: &mut [f32], w: usize, h: usize, s: &Stroke) {
    let Some(first) = s.points.first() else {
        return;
    };
    let r = s.radius.max(0.5);
    let core = r * s.hardness.clamp(0.0, 0.99);
    let segments: Vec<([f32; 2], [f32; 2])> = if s.points.len() == 1 { vec![(*first, *first)] } else { s.points.windows(2).map(|p| (p[0], p[1])).collect() };
    for (a, b) in segments {
        let bx0 = ((a[0].min(b[0]) - r).floor().max(0.0) as usize).min(w);
        let by0 = ((a[1].min(b[1]) - r).floor().max(0.0) as usize).min(h);
        let bx1 = ((a[0].max(b[0]) + r).ceil().max(0.0) as usize + 1).min(w);
        let by1 = ((a[1].max(b[1]) + r).ceil().max(0.0) as usize + 1).min(h);
        if bx0 >= bx1 || by0 >= by1 {
            continue;
        }
        mask[by0 * w..by1 * w].par_chunks_mut(w).enumerate().for_each(|(dy, row)| {
            let py = (by0 + dy) as f32 + 0.5;
            for (x, v) in row.iter_mut().enumerate().take(bx1).skip(bx0) {
                let d = segment_distance(x as f32 + 0.5, py, a, b);
                if d >= r {
                    continue;
                }
                let cover = if d <= core { 1.0 } else { 1.0 - (d - core) / (r - core) };
                // Smoothstep for a soft, round falloff.
                let cover = cover * cover * (3.0 - 2.0 * cover);
                *v = if s.erase { v.min(1.0 - cover) } else { v.max(cover) };
            }
        });
    }
}

fn segment_distance(px: f32, py: f32, a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { (((px - a[0]) * dx + (py - a[1]) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    ((px - a[0] - t * dx).powi(2) + (py - a[1] - t * dy).powi(2)).sqrt()
}

fn raster(w: usize, h: usize, inside: impl Fn(f32, f32) -> f32 + Sync) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = inside(x as f32 + 0.5, y as f32 + 0.5);
        }
    });
    out
}

fn auto_region(img: &Image, region: AutoRegion) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let lum = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let foreground = foreground(&lum, w, h);
    let as_mask = |m: Vec<bool>| m.into_par_iter().map(|v| v as u8 as f32).collect();
    match region {
        AutoRegion::Subject => as_mask(main_body(&foreground, w, h)),
        AutoRegion::Background => as_mask(main_body(&foreground, w, h).into_iter().map(|v| !v).collect()),
        AutoRegion::Moons => moons(&lum, w, h),
        AutoRegion::SaturnGlobe | AutoRegion::SaturnRings => {
            let Some(globe) = saturn_globe(img) else {
                return as_mask(main_body(&foreground, w, h));
            };
            if region == AutoRegion::SaturnGlobe {
                raster(w, h, |x, y| (globe.radius(x, y) < 1.0) as u8 as f32)
            } else {
                // A slightly larger globe keeps the limb out of the ring mask.
                let grown = Ellipse { rx: globe.rx * 1.03, ry: globe.ry * 1.03, ..globe };
                raster(w, h, |x, y| (foreground[(y as usize) * w + x as usize] && grown.radius(x, y) >= 1.0) as u8 as f32)
            }
        }
    }
}

/// Where "Nitidezza intelligente" works when the sky is left out (0‥1): the subject (planet,
/// disk, ring system) and the moons, widened by ~2 px so the blurred limb is in, with a soft
/// edge.
pub fn subject_and_moons(img: &Image) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let lum = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let body = main_body(&foreground(&lum, w, h), w, h);
    let moons = moons(&lum, w, h);
    let solid: Vec<f32> = body.into_par_iter().zip(moons).map(|(b, m)| if b { 1.0 } else { m }).collect();
    gaussian_approx(&solid, w, h, 2.0).into_par_iter().map(|v| (2.0 * v).min(1.0)).collect()
}

/// Pixels brighter than the sky. The threshold sits halfway between the sky level and Otsu's
/// threshold, so limb-darkened belts still count as subject; a light blur keeps sky noise out.
pub(crate) fn foreground(lum: &[f32], w: usize, h: usize) -> Vec<bool> {
    let (small, ..) = downsample(lum, w, h, 512);
    let otsu_t = otsu(&small);
    let mut sky: Vec<f32> = small.iter().copied().filter(|&v| v <= otsu_t).collect();
    let sky = if sky.is_empty() {
        0.0
    } else {
        let mid = sky.len() / 2;
        *sky.select_nth_unstable_by(mid, f32::total_cmp).1
    };
    let threshold = sky + 0.5 * (otsu_t - sky);
    gaussian_approx(lum, w, h, 1.5).into_par_iter().map(|v| v > threshold).collect()
}

/// The largest bright region, made solid (moons and stars are left out).
pub(crate) fn main_body(foreground: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut only = vec![false; foreground.len()];
    if let Some(largest) = components(foreground, w, h).into_iter().max_by_key(Vec::len) {
        for i in largest {
            only[i] = true;
        }
    }
    fill_holes(&only, w, h)
}

/// Discs around every compact bright body other than the main one. Moons are much fainter
/// than the planet's threshold allows (Titan, Europa in shadow…), so detection uses the noise
/// level instead: anything 8σ above the sky that is not the planet.
fn moons(lum: &[f32], w: usize, h: usize) -> Vec<f32> {
    let sigma = estimate_noise(lum, w, h).max(1e-4);
    let (small, ..) = downsample(lum, w, h, 512);
    let mut sky: Vec<f32> = small.clone();
    let mid = sky.len() / 4;
    let sky = *sky.select_nth_unstable_by(mid, f32::total_cmp).1;
    let smooth = gaussian_approx(lum, w, h, 1.0);
    let bright: Vec<bool> = smooth.par_iter().map(|&v| v > sky + 8.0 * sigma).collect();
    let mut parts = components(&bright, w, h);
    parts.sort_by_key(|c| std::cmp::Reverse(c.len()));
    let Some(planet) = parts.first().map(Vec::len) else {
        return vec![0.0; w * h];
    };
    let mut out = vec![0.0f32; w * h];
    for part in parts.iter().skip(1).filter(|c| c.len() >= 3 && c.len() * 20 < planet) {
        let n = part.len() as f32;
        let (sx, sy) = part.iter().fold((0.0, 0.0), |(sx, sy), &i| (sx + (i % w) as f32 + 0.5, sy + (i / w) as f32 + 0.5));
        let stroke = Stroke { points: vec![[sx / n, sy / n]], radius: (n / std::f32::consts::PI).sqrt() * 1.6 + 3.0, hardness: 0.6, erase: false };
        paint(&mut out, w, h, &stroke);
    }
    out
}

/// 4-connected regions of `mask`, as pixel indices.
pub(crate) fn components(mask: &[bool], w: usize, h: usize) -> Vec<Vec<usize>> {
    let mut seen = vec![false; mask.len()];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || seen[start] {
            continue;
        }
        let mut members = Vec::new();
        seen[start] = true;
        stack.push(start);
        while let Some(i) = stack.pop() {
            members.push(i);
            let (x, y) = (i % w, i / w);
            let neighbours = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
            for nb in neighbours.into_iter().flatten() {
                if mask[nb] && !seen[nb] {
                    seen[nb] = true;
                    stack.push(nb);
                }
            }
        }
        out.push(members);
    }
    out
}

/// Make the subject solid: everything not connected to the image border through background
/// pixels belongs to it (dark belts, craters in shadow, the Cassini division…).
fn fill_holes(foreground: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut outside = vec![false; foreground.len()];
    let mut stack: Vec<usize> = (0..w).flat_map(|x| [x, (h - 1) * w + x]).chain((0..h).flat_map(|y| [y * w, y * w + w - 1])).filter(|&i| !foreground[i]).collect();
    for &i in &stack {
        outside[i] = true;
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        let neighbours = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
        for n in neighbours.into_iter().flatten() {
            if !foreground[n] && !outside[n] {
                outside[n] = true;
                stack.push(n);
            }
        }
    }
    outside.into_iter().map(|o| !o).collect()
}

/// Saturn's globe from the ring system's geometry: the rings' major axis gives the
/// orientation and, through the ring/globe size ratio, the globe's radius.
pub fn saturn_globe(img: &Image) -> Option<Ellipse> {
    let (w, h) = (img.width(), img.height());
    let lum = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let (small, sw, sh, scale) = downsample(&lum, w, h, 512);
    // Halfway between the sky and Otsu's threshold, as for the subject's mask: Otsu's alone,
    // on a sharp image, can leave the fainter rings out and split the globe from them, and the
    // globe alone would then be taken for the ring system.
    let otsu_t = otsu(&small);
    let mut sky: Vec<f32> = small.iter().copied().filter(|&v| v <= otsu_t).collect();
    let sky = if sky.is_empty() { 0.0 } else { let mid = sky.len() / 2; *sky.select_nth_unstable_by(mid, f32::total_cmp).1 };
    let threshold = sky + 0.5 * (otsu_t - sky);
    let member = largest_component_members(&small.iter().map(|&v| v > threshold).collect::<Vec<_>>(), sw, sh)?;

    let n = member.len() as f32;
    let (mx, my) = member.iter().fold((0.0, 0.0), |(sx, sy), &(x, y)| (sx + x, sy + y));
    let (mx, my) = (mx / n, my / n);
    let (mut cxx, mut cyy, mut cxy) = (0.0, 0.0, 0.0);
    for &(x, y) in &member {
        let (dx, dy) = (x - mx, y - my);
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    let angle = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    let (s, c) = angle.sin_cos();
    let half_length = member.iter().map(|&(x, y)| ((x - mx) * c + (y - my) * s).abs()).fold(0.0f32, f32::max) + 0.5;
    let half_width = member.iter().map(|&(x, y)| (-(x - mx) * s + (y - my) * c).abs()).fold(0.0f32, f32::max) + 0.5;
    // Rings too faint to reach the threshold (near edge-on, or dark as in 1995–96 and 2025):
    // the region is the globe alone, nearly round, where the ring system, even fully open as
    // seen from Earth, is at most about half as tall as it is wide.
    let ring_to_globe = if half_width > 0.75 * half_length { 1.0 } else { SATURN_RING_TO_GLOBE };
    let rx = half_length * scale / ring_to_globe;
    Some(Ellipse { cx: mx * scale, cy: my * scale, rx, ry: rx * SATURN_OBLATENESS, angle })
}

/// Pixel centres of the largest 4-connected foreground region.
fn largest_component_members(mask: &[bool], w: usize, h: usize) -> Option<Vec<(f32, f32)>> {
    let best = components(mask, w, h).into_iter().max_by_key(Vec::len)?;
    (best.len() >= 16).then(|| best.into_iter().map(|i| ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5)).collect())
}

/// Gaussian blur approximated by three box blurs (O(1) per pixel whatever the radius).
pub fn gaussian_approx(plane: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = ((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() / 2.0).round().max(1.0) as usize;
    if let Some(out) = crate::gpu::for_plane(w, h, crate::gpu::Work::Light).and_then(|g| g.box_blur3(plane, w, h, r).ok()) {
        return out;
    }
    box_blur3(plane, w, h, r)
}

/// Three passes of a box blur of radius `r` along rows and columns (CPU).
pub(crate) fn box_blur3(plane: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut a = plane.to_vec();
    for _ in 0..3 {
        a = box_cols(&box_rows(&a, w, r), w, h, r);
    }
    a
}

/// Running box average down the columns, edges clamped. Whole rows are added and removed, so
/// memory is walked in order; blocks of rows run in parallel, each starting its own sum.
fn box_cols(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    // Each block first sums 2r + 1 rows: keep blocks large next to that.
    let block = (4 * r).max(64);
    let mut out = vec![0.0f32; src.len()];
    let norm = 1.0 / (2 * r + 1) as f32;
    let row = |y: isize| &src[y.clamp(0, h as isize - 1) as usize * w..][..w];
    out.par_chunks_mut(w * block).enumerate().for_each(|(b, dst)| {
        let y0 = (b * block) as isize;
        let mut acc = vec![0.0f32; w];
        for y in y0 - r as isize..=y0 + r as isize {
            acc.iter_mut().zip(row(y)).for_each(|(a, &v)| *a += v);
        }
        for (k, out_row) in dst.chunks_mut(w).enumerate() {
            let y = y0 + k as isize;
            out_row.iter_mut().zip(&acc).for_each(|(o, &a)| *o = a * norm);
            let (add, remove) = (row(y + r as isize + 1), row(y - r as isize));
            acc.iter_mut().zip(add.iter().zip(remove)).for_each(|(a, (&p, &m))| *a += p - m);
        }
    });
    out
}

/// Running box average along rows, edges clamped.
fn box_rows(src: &[f32], w: usize, r: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    out.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(|(o, row)| {
        let at = |i: isize| row[i.clamp(0, w as isize - 1) as usize];
        let norm = 1.0 / (2 * r + 1) as f32;
        let mut sum: f32 = (-(r as isize)..=r as isize).map(at).sum();
        // Clamped reads only where the window leaves the row.
        let (lo, hi) = (r.min(w), w.saturating_sub(r + 1).max(r.min(w)));
        for (x, v) in o[..lo].iter_mut().enumerate() {
            *v = sum * norm;
            sum += at(x as isize + r as isize + 1) - at(x as isize - r as isize);
        }
        for x in lo..hi {
            o[x] = sum * norm;
            sum += row[x + r + 1] - row[x - r];
        }
        for (x, v) in o.iter_mut().enumerate().skip(hi) {
            *v = sum * norm;
            sum += at(x as isize + r as isize + 1) - at(x as isize - r as isize);
        }
    });
    out
}
