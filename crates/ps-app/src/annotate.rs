//! The names printed onto the exported image: a layer of text, circles and a compass drawn at
//! the image's own resolution (not the screen's), so the picture can be shared as it is.
//!
//! Text is drawn from the font outlines the interface already ships (Ubuntu Light), with an
//! exact-area coverage rasteriser; every mark gets a dark halo so it reads on bright and dark
//! parts alike.

use ps_core::i18n::tl;
use ps_core::Image;
use ps_core::identify::{Label, LabelKind};
use serde::{Deserialize, Serialize};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{FontRef, MetadataProvider};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tint {
    White,
    Yellow,
    Green,
    Orange,
}

impl Tint {
    pub const ALL: [Tint; 4] = [Tint::White, Tint::Yellow, Tint::Green, Tint::Orange];

    pub fn label(self) -> &'static str {
        match self {
            Tint::White => tl("Bianco"),
            Tint::Yellow => tl("Giallo"),
            Tint::Green => tl("Verde"),
            Tint::Orange => tl("Arancio"),
        }
    }

    pub fn rgb(self) -> [f32; 3] {
        match self {
            Tint::White => [1.0, 1.0, 1.0],
            Tint::Yellow => [1.0, 0.92, 0.45],
            Tint::Green => [0.5, 0.95, 0.55],
            Tint::Orange => [1.0, 0.7, 0.3],
        }
    }
}

/// Where the caption sits; the compass takes the opposite side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Corner {
    BottomLeft,
    BottomRight,
    TopLeft,
    TopRight,
}

impl Corner {
    pub const ALL: [Corner; 4] = [Corner::TopLeft, Corner::TopRight, Corner::BottomLeft, Corner::BottomRight];

    pub fn label(self) -> &'static str {
        match self {
            Corner::TopLeft => tl("In alto a sinistra"),
            Corner::TopRight => tl("In alto a destra"),
            Corner::BottomLeft => tl("In basso a sinistra"),
            Corner::BottomRight => tl("In basso a destra"),
        }
    }

    fn right(self) -> bool {
        matches!(self, Corner::TopRight | Corner::BottomRight)
    }

    fn bottom(self) -> bool {
        matches!(self, Corner::BottomLeft | Corner::BottomRight)
    }

    fn mirrored(self) -> Corner {
        match self {
            Corner::TopLeft => Corner::TopRight,
            Corner::TopRight => Corner::TopLeft,
            Corner::BottomLeft => Corner::BottomRight,
            Corner::BottomRight => Corner::BottomLeft,
        }
    }
}

/// What goes on the picture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LabelOptions {
    /// A ring round each moon.
    pub circles: bool,
    /// Moons whose spot was found in the image.
    pub moons: bool,
    /// Moons only predicted by the ephemerides (no spot found).
    pub predicted: bool,
    pub craters: bool,
    pub seas: bool,
    pub object: bool,
    pub filter: bool,
    pub time: bool,
    pub scale: bool,
    /// Telescope and camera, as the capture log says.
    pub equipment: bool,
    pub compass: bool,
    /// The north direction and the turn of the image, in words.
    pub angles: bool,
    /// Text height, percent of the image's shorter side.
    pub size: f32,
    /// Colour of the names on the picture.
    pub colour: [f32; 3],
    /// Colour of the caption and the compass.
    pub caption_colour: [f32; 3],
    /// Free lines, printed when not empty: a title (larger), who took the picture, where, a note.
    pub title: String,
    pub author: String,
    pub location: String,
    pub note: String,
    pub corner: Corner,
}

impl Default for LabelOptions {
    fn default() -> Self {
        Self { circles: false, moons: true, predicted: false, craters: true, seas: true, object: true, filter: true, time: true, scale: false, equipment: false, compass: true, angles: false, size: 2.4, colour: Tint::White.rgb(), caption_colour: Tint::White.rgb(), title: String::new(), author: String::new(), location: String::new(), note: String::new(), corner: Corner::BottomLeft }
    }
}

/// Everything the layer needs, in the coordinates of the image that was identified.
#[derive(Clone, Debug)]
pub struct Plan {
    pub labels: Vec<Label>,
    /// Size of the image the label positions refer to.
    pub from: (usize, usize),
    /// The title, drawn larger above the caption.
    pub heading: Option<String>,
    /// Lines of the caption (object, filter, time…), already chosen.
    pub caption: Vec<String>,
    /// Where north points, degrees clockwise from up.
    pub north: Option<f32>,
}

const FONT: &[u8] = epaint_default_fonts::UBUNTU_LIGHT;

/// Coverage of a run of text, with the pen origin inside it.
struct Run {
    w: usize,
    h: usize,
    cover: Vec<f32>,
    /// Distance from the top to the baseline.
    baseline: f32,
}

struct Accumulator<'a> {
    stride: usize,
    h: usize,
    acc: &'a mut [f32],
    offset: (f32, f32),
    start: (f32, f32),
    at: (f32, f32),
}

impl Accumulator<'_> {
    fn point(&self, x: f32, y: f32) -> (f32, f32) {
        // Outlines are y-up.
        (self.offset.0 + x, self.offset.1 - y)
    }

    /// Signed-area accumulation of one edge (exact coverage once summed along the row).
    fn line(&mut self, p0: (f32, f32), p1: (f32, f32)) {
        if p0.1 == p1.1 {
            return;
        }
        let (dir, p0, p1) = if p0.1 < p1.1 { (1.0, p0, p1) } else { (-1.0, p1, p0) };
        let dxdy = (p1.0 - p0.0) / (p1.1 - p0.1);
        let mut x = p0.0;
        if p0.1 < 0.0 {
            x -= p0.1 * dxdy;
        }
        let y0 = p0.1.max(0.0) as usize;
        let max_x = (self.stride - 2) as f32;
        for y in y0..self.h.min(p1.1.ceil().max(0.0) as usize) {
            let row = y * self.stride;
            let dy = ((y + 1) as f32).min(p1.1) - (y as f32).max(p0.1);
            let xnext = x + dxdy * dy;
            let d = dy * dir;
            let (a, b) = if x < xnext { (x, xnext) } else { (xnext, x) };
            let (a, b) = (a.clamp(0.0, max_x), b.clamp(0.0, max_x));
            let a_floor = a.floor();
            let ai = a_floor as usize;
            let b_ceil = b.ceil();
            let bi = b_ceil as usize;
            if bi <= ai + 1 {
                let xmf = 0.5 * (a + b) - a_floor;
                self.acc[row + ai] += d - d * xmf;
                self.acc[row + ai + 1] += d * xmf;
            } else {
                let s = 1.0 / (b - a);
                let af = a - a_floor;
                let a0 = 0.5 * s * (1.0 - af) * (1.0 - af);
                let bf = b - b_ceil + 1.0;
                let am = 0.5 * s * bf * bf;
                self.acc[row + ai] += d * a0;
                if bi == ai + 2 {
                    self.acc[row + ai + 1] += d * (1.0 - a0 - am);
                } else {
                    let a1 = s * (1.5 - af);
                    self.acc[row + ai + 1] += d * (a1 - a0);
                    for xi in ai + 2..bi - 1 {
                        self.acc[row + xi] += d * s;
                    }
                    let a2 = a1 + (bi - ai - 3) as f32 * s;
                    self.acc[row + bi - 1] += d * (1.0 - a2 - am);
                }
                self.acc[row + bi] += d * am;
            }
            x = xnext;
        }
    }
}

impl OutlinePen for Accumulator<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.point(x, y);
        (self.start, self.at) = (p, p);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.point(x, y);
        self.line(self.at, p);
        self.at = p;
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (c, p1, p0) = (self.point(cx, cy), self.point(x, y), self.at);
        let n = 8;
        for k in 1..=n {
            let t = k as f32 / n as f32;
            let u = 1.0 - t;
            let q = (u * u * p0.0 + 2.0 * u * t * c.0 + t * t * p1.0, u * u * p0.1 + 2.0 * u * t * c.1 + t * t * p1.1);
            self.line(self.at, q);
            self.at = q;
        }
    }

    fn curve_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
        let (c1, c2, p1, p0) = (self.point(c1x, c1y), self.point(c2x, c2y), self.point(x, y), self.at);
        let n = 12;
        for k in 1..=n {
            let t = k as f32 / n as f32;
            let u = 1.0 - t;
            let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
            let q = (a * p0.0 + b * c1.0 + c * c2.0 + d * p1.0, a * p0.1 + b * c1.1 + c * c2.1 + d * p1.1);
            self.line(self.at, q);
            self.at = q;
        }
    }

    fn close(&mut self) {
        self.line(self.at, self.start);
        self.at = self.start;
    }
}

/// The coverage of `text` at `px` pixels tall, or `None` when the font can't be read.
fn rasterise(text: &str, px: f32) -> Option<Run> {
    let font = FontRef::new(FONT).ok()?;
    let size = Size::new(px);
    let metrics = font.metrics(size, LocationRef::default());
    let (charmap, outlines, gm) = (font.charmap(), font.outline_glyphs(), font.glyph_metrics(size, LocationRef::default()));
    let pad = 2.0f32;
    let glyphs: Vec<(skrifa::GlyphId, f32)> = text
        .chars()
        .map(|c| {
            let gid = charmap.map(c).or_else(|| charmap.map('?')).unwrap_or_default();
            (gid, gm.advance_width(gid).unwrap_or(px * 0.5))
        })
        .collect();
    let width: f32 = glyphs.iter().map(|g| g.1).sum();
    let baseline = metrics.ascent + pad;
    let (w, h) = ((width + 2.0 * pad).ceil() as usize + 2, (metrics.ascent - metrics.descent + 2.0 * pad).ceil() as usize + 1);
    let stride = w + 2;
    let mut acc = vec![0.0f32; stride * h];
    let mut x = pad;
    for (gid, advance) in glyphs {
        if let Some(glyph) = outlines.get(gid) {
            let mut pen = Accumulator { stride, h, acc: &mut acc, offset: (x, baseline), start: (0.0, 0.0), at: (0.0, 0.0) };
            glyph.draw(DrawSettings::unhinted(size, LocationRef::default()), &mut pen).ok()?;
        }
        x += advance;
    }
    let mut cover = vec![0.0f32; w * h];
    for y in 0..h {
        let mut sum = 0.0f32;
        for x in 0..w {
            sum += acc[y * stride + x];
            cover[y * w + x] = sum.abs().min(1.0);
        }
    }
    Some(Run { w, h, cover, baseline })
}

/// The picture being drawn on.
struct Canvas<'a> {
    img: &'a mut Image,
}

impl Canvas<'_> {
    fn blend(&mut self, x: i64, y: i64, alpha: f32, rgb: [f32; 3]) {
        let (w, h) = (self.img.width() as i64, self.img.height() as i64);
        if alpha <= 0.0 || x < 0 || y < 0 || x >= w || y >= h {
            return;
        }
        let i = (y * w + x) as usize;
        let mono = self.img.is_mono();
        for (c, plane) in self.img.channels_mut().iter_mut().enumerate() {
            let target = if mono { 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] } else { rgb[c] };
            plane[i] += (target - plane[i]) * alpha;
        }
    }

    /// Text with its top-left at `at`, a dark halo underneath.
    fn text(&mut self, text: &str, at: (f32, f32), px: f32, rgb: [f32; 3]) {
        let Some(run) = rasterise(text, px) else { return };
        let (ox, oy) = (at.0.round() as i64, at.1.round() as i64);
        let reach = (px / 16.0).round().max(1.0) as i64;
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                if dx * dx + dy * dy > reach * reach + 1 {
                    continue;
                }
                for y in 0..run.h {
                    for x in 0..run.w {
                        let c = run.cover[y * run.w + x];
                        if c > 0.0 {
                            self.blend(ox + x as i64 + dx, oy + y as i64 + dy, c * 0.55, [0.0; 3]);
                        }
                    }
                }
            }
        }
        for y in 0..run.h {
            for x in 0..run.w {
                let c = run.cover[y * run.w + x];
                if c > 0.0 {
                    self.blend(ox + x as i64, oy + y as i64, c, rgb);
                }
            }
        }
    }

    fn text_size(&self, text: &str, px: f32) -> (f32, f32, f32) {
        rasterise(text, px).map_or((0.0, 0.0, 0.0), |r| (r.w as f32, r.h as f32, r.baseline))
    }

    /// A segment `width` wide, anti-aliased, with a dark halo.
    fn segment(&mut self, a: (f32, f32), b: (f32, f32), width: f32, rgb: [f32; 3]) {
        for (extra, colour, strength) in [(width * 1.2, [0.0; 3], 0.55), (0.0, rgb, 1.0)] {
            let half = (width + extra) / 2.0;
            let (x0, x1) = ((a.0.min(b.0) - half - 1.0).floor() as i64, (a.0.max(b.0) + half + 1.0).ceil() as i64);
            let (y0, y1) = ((a.1.min(b.1) - half - 1.0).floor() as i64, (a.1.max(b.1) + half + 1.0).ceil() as i64);
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let len2 = (dx * dx + dy * dy).max(1e-6);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f32 + 0.5 - a.0, y as f32 + 0.5 - a.1);
                    let t = ((px * dx + py * dy) / len2).clamp(0.0, 1.0);
                    let d = ((px - t * dx).powi(2) + (py - t * dy).powi(2)).sqrt();
                    self.blend(x, y, (half + 0.5 - d).clamp(0.0, 1.0) * strength, colour);
                }
            }
        }
    }

    fn ring(&mut self, c: (f32, f32), r: f32, width: f32, rgb: [f32; 3]) {
        for (extra, colour, strength) in [(width * 1.2, [0.0; 3], 0.55), (0.0, rgb, 1.0)] {
            let half = (width + extra) / 2.0;
            let reach = (r + half + 2.0).ceil() as i64;
            for y in -reach..=reach {
                for x in -reach..=reach {
                    let (px, py) = (x as f32 + c.0.floor() + 0.5 - c.0, y as f32 + c.1.floor() + 0.5 - c.1);
                    let d = ((px * px + py * py).sqrt() - r).abs();
                    self.blend(x + c.0.floor() as i64, y + c.1.floor() as i64, (half + 0.5 - d).clamp(0.0, 1.0) * strength, colour);
                }
            }
        }
    }
}

/// `img` with the plan's layer printed onto it. Positions scale with the image's size, so the
/// layer can be added after an enlargement.
pub fn burn_in(img: &Image, plan: &Plan, opt: &LabelOptions) -> Image {
    let mut out = img.clone();
    let rgb = opt.colour;
    let grey = |c: [f32; 3]| (c[0] - c[1]).abs() < 0.02 && (c[1] - c[2]).abs() < 0.02;
    // A coloured layer needs three channels.
    if out.is_mono() && !(grey(opt.colour) && grey(opt.caption_colour)) {
        let plane = out.channels()[0].clone();
        out = Image::new(out.width(), out.height(), vec![plane.clone(), plane.clone(), plane], out.source_bit_depth);
    }
    let (w, h) = (out.width() as f32, out.height() as f32);
    let k = w / plan.from.0.max(1) as f32;
    let px = (opt.size / 100.0 * w.min(h)).clamp(11.0, 400.0);
    let margin = px * 0.9;
    let mut canvas = Canvas { img: &mut out };

    for l in &plan.labels {
        let (x, y) = ((l.x + 0.5) * k, (l.y + 0.5) * k);
        if x < 0.0 || y < 0.0 || x >= w || y >= h {
            continue;
        }
        match l.kind {
            LabelKind::MoonFound | LabelKind::MoonPredicted => {
                if (l.kind == LabelKind::MoonFound && !opt.moons) || (l.kind == LabelKind::MoonPredicted && !opt.predicted) {
                    continue;
                }
                let r = if opt.circles { px * 0.55 } else { 0.0 };
                if opt.circles {
                    canvas.ring((x, y), r, (px / 14.0).max(1.2), rgb);
                }
                let (tw, th, _) = canvas.text_size(&l.name, px);
                let gap = r + px * 0.3;
                // Beside the moon, on the side with room.
                let left = if x + gap + tw < w { x + gap } else { x - gap - tw };
                canvas.text(&l.name, (left, y - th / 2.0), px, rgb);
            }
            LabelKind::Crater | LabelKind::Feature if opt.craters => {
                let (tw, th, _) = canvas.text_size(&l.name, px * 0.8);
                canvas.text(&l.name, (x - tw / 2.0, y - th / 2.0), px * 0.8, rgb);
            }
            LabelKind::Mare if opt.seas => {
                let (tw, th, _) = canvas.text_size(&l.name, px);
                canvas.text(&l.name, (x - tw / 2.0, y - th / 2.0), px, rgb);
            }
            _ => {}
        }
    }

    // The caption in its corner (the last line nearest a bottom edge, the first nearest a top one),
    // the title larger above it.
    let cap = opt.caption_colour;
    let mut lines: Vec<(&str, f32)> = Vec::new();
    if let Some(t) = &plan.heading {
        lines.push((t, 1.35));
    }
    lines.extend(plan.caption.iter().map(|l| (l.as_str(), 1.0)));
    let heights: Vec<f32> = lines.iter().map(|(_, k)| px * k * 1.3).collect();
    let total: f32 = heights.iter().sum();
    let mut y = if opt.corner.bottom() { h - margin - total } else { margin };
    for ((line, k), step) in lines.iter().zip(&heights) {
        let (tw, _, _) = canvas.text_size(line, px * k);
        let x = if opt.corner.right() { w - margin - tw } else { margin };
        canvas.text(line, (x, y), px * k, cap);
        y += step;
    }

    // The compass on the other side: north is the arrow, south the opposite end.
    if let (true, Some(north)) = (opt.compass, plan.north) {
        let side = opt.corner.mirrored();
        let r = px * 2.6;
        let cx = if side.right() { w - margin - r - px } else { margin + r + px };
        let cy = if side.bottom() { h - margin - r - px * 0.5 } else { margin + r + px * 1.5 };
        let c = (cx, cy);
        let a = north.to_radians();
        let dir = (a.sin(), -a.cos());
        let tip = (c.0 + dir.0 * r, c.1 + dir.1 * r);
        let tail = (c.0 - dir.0 * r, c.1 - dir.1 * r);
        let lw = (px / 12.0).max(1.5);
        canvas.segment(tail, tip, lw, cap);
        // Arrow head.
        for side in [-1.0f32, 1.0] {
            let back = (tip.0 - dir.0 * px * 0.7 - side * dir.1 * px * 0.35, tip.1 - dir.1 * px * 0.7 + side * dir.0 * px * 0.35);
            canvas.segment(tip, back, lw, cap);
        }
        let place = |canvas: &Canvas, s: &str, p: (f32, f32), d: (f32, f32)| {
            let (tw, th, _) = canvas.text_size(s, px);
            (p.0 + d.0 * (tw / 2.0 + px * 0.25) - tw / 2.0, p.1 + d.1 * (th / 2.0 + px * 0.1) - th / 2.0)
        };
        let n_at = place(&canvas, "N", tip, dir);
        canvas.text("N", n_at, px, cap);
        let s_at = place(&canvas, "S", tail, (-dir.0, -dir.1));
        canvas.text("S", s_at, px * 0.85, cap);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps_core::identify::Label;

    #[test]
    fn text_has_coverage_and_grows_with_size() {
        let small = rasterise("Io", 20.0).expect("font");
        let big = rasterise("Io", 40.0).expect("font");
        assert!(small.cover.iter().sum::<f32>() > 20.0);
        // Twice the height: about four times the ink.
        let ratio = big.cover.iter().sum::<f32>() / small.cover.iter().sum::<f32>();
        assert!((3.0..5.0).contains(&ratio), "{ratio}");
        assert!(small.cover.iter().all(|c| (0.0..=1.0).contains(c)));
    }

    #[test]
    fn burn_in_marks_the_picture_and_leaves_the_rest() {
        let img = Image::new(400, 300, vec![vec![0.1; 400 * 300]], 16);
        let plan = Plan {
            labels: vec![Label { name: "Europa".into(), x: 100.0, y: 100.0, radius: 0.0, kind: LabelKind::MoonFound, on_terminator: false }],
            from: (400, 300),
            heading: Some("Giove".into()),
            caption: vec!["Giove".into(), "2026-11-09 20:10 UT".into()],
            north: Some(30.0),
        };
        let out = burn_in(&img, &plan, &LabelOptions { circles: true, ..Default::default() });
        let changed = img.channels()[0].iter().zip(&out.channels()[0]).filter(|(a, b)| (**a - **b).abs() > 0.01).count();
        assert!(changed > 400, "{changed}");
        // The far corner is untouched.
        assert_eq!(out.channels()[0][5 * 400 + 395], 0.1);
        // A coloured layer on a mono image makes it RGB.
        let tinted = burn_in(&img, &plan, &LabelOptions { colour: Tint::Green.rgb(), ..Default::default() });
        assert!(!tinted.is_mono());
    }

    /// `PS_SNAPSHOT_IMAGE=<abs path> PS_SNAPSHOT_OUT=dir cargo test -p ps-app -- --ignored annotate_sample`
    #[test]
    #[ignore]
    fn annotate_sample() {
        let input = std::path::PathBuf::from(std::env::var("PS_SNAPSHOT_IMAGE").expect("PS_SNAPSHOT_IMAGE"));
        let out = std::path::PathBuf::from(std::env::var("PS_SNAPSHOT_OUT").expect("PS_SNAPSHOT_OUT"));
        let img = ps_io::load(&input).expect("image");
        let (w, h) = (img.width() as f32, img.height() as f32);
        let plan = Plan {
            labels: vec![
                Label { name: "Io".into(), x: w * 0.7, y: h * 0.45, radius: 0.0, kind: LabelKind::MoonFound, on_terminator: false },
                Label { name: "Ganimede".into(), x: w * 0.12, y: h * 0.3, radius: 0.0, kind: LabelKind::MoonFound, on_terminator: false },
            ],
            from: (img.width(), img.height()),
            heading: Some("Giove, 5 luglio 2019".into()),
            caption: vec!["Foto: Mario Rossi".into(), "Filtro R".into(), "2019-07-05 22:27 UT".into()],
            north: Some(321.0),
        };
        for (name, circles) in [("annotated_plain.png", false), ("annotated_circles.png", true)] {
            let done = burn_in(&img, &plan, &LabelOptions { circles, ..Default::default() });
            ps_io::save(&out.join(name), &done).expect("save");
        }
    }
}
