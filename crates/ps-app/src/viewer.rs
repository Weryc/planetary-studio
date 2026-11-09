//! Central image canvas: zoom towards the cursor, pan, before/after/split comparison, and
//! rectangle editing for the ROI, the crop and hand-drawn mask shapes, and mask painting.

use crate::theme::{ACCENT, BG, TEXT};
use crate::worker::Roi;
use eframe::egui::{
    self, Color32, ColorImage, CornerRadius, CursorIcon, FontId, PointerButton, Pos2, Rect, Sense, Shape, Stroke, StrokeKind,
    TextureHandle, TextureOptions, Vec2, pos2,
};
use std::sync::Arc;

const MIN_ZOOM: f32 = 0.02;
const MAX_ZOOM: f32 = 32.0;
const MIN_RECT: f32 = 16.0;
const DEFAULT_ROI: f32 = 512.0;
/// Screen distance (points) within which a corner or the split line can be grabbed.
const GRAB: f32 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    After,
    Before,
    Split,
    /// Split view with the kept version ("A") on the left instead of the original.
    Ab,
    /// The kept version alone.
    OnlyA,
}

/// Information drawn over the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overlay {
    /// Rings left by sharpening.
    Ringing,
    /// Pixels burnt to white or crushed to black.
    Clipping,
    /// What "Nitidezza intelligente" does: where it holds back, or the finest detail present.
    SmartMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Pan,
    Roi,
    Crop,
    /// Edit the selected layer's hand-drawn mask shape.
    Shape,
    /// Paint on the selected layer's mask.
    Brush,
    /// Click to sample a pixel (black/white point).
    Pick,
    /// Draw a box around something to measure (a star or moon for the PSF).
    Probe,
}

/// Brush as shown on the canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushView {
    /// Radius in image pixels.
    pub radius: f32,
    pub erase: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShapeKind {
    Ellipse { angle_deg: f32 },
    Rectangle,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CropAction {
    Apply,
    Cancel,
    /// Quarter turn: +1 clockwise, −1 counter-clockwise.
    Rotate(i8),
    Flip { horizontal: bool },
    /// Fine rotation (degrees), sent when the slider is released.
    Straighten(f32),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    Roi,
    Crop,
    Shape,
    Probe,
}

struct Layer {
    texture: TextureHandle,
    image: Arc<ColorImage>,
}

impl Layer {
    fn new(ctx: &egui::Context, name: &str, image: Arc<ColorImage>, nearest: bool) -> Self {
        let texture = ctx.load_texture(name, Arc::clone(&image), filter(nearest));
        Self { texture, image }
    }

    fn replace(&mut self, image: Arc<ColorImage>, nearest: bool) {
        self.texture.set(Arc::clone(&image), filter(nearest));
        self.image = image;
    }
}

fn filter(nearest: bool) -> TextureOptions {
    if nearest { TextureOptions::NEAREST } else { TextureOptions::LINEAR }
}

enum Drag {
    Pan,
    Split,
    Create { target: Target, start: Pos2 },
    Move { target: Target, grab: Vec2 },
    Resize { target: Target, anchor: Pos2 },
    Paint,
}

#[derive(Default)]
pub struct ViewerOutput {
    /// Image pixel under the cursor.
    pub hover_pixel: Option<(usize, usize)>,
    pub crop_action: Option<CropAction>,
    /// A finished brush stroke (image coordinates).
    pub stroke: Option<Vec<[f32; 2]>>,
    /// Pixel clicked with the picker.
    pub picked: Option<(usize, usize)>,
    /// Box drawn with the probe (image coordinates).
    pub probed: Option<Rect>,
    pub hovered: bool,
    /// The button under the ROI was clicked: done editing it (back to moving the view), or
    /// edit it again.
    pub roi_toggle: bool,
}

pub struct Viewer {
    original: Layer,
    result: Option<(Layer, Option<Roi>)>,
    mask: Option<(Layer, u64)>,
    /// Ringing and clipping overlays, by [`Overlay`] index.
    overlays: [Option<(Layer, Option<Roi>)>; 3],
    /// The A/B snapshot.
    reference: Option<(Layer, Option<Roi>)>,
    image_size: Vec2,
    zoom: f32,
    /// Offset of the image centre from the canvas centre, in screen points.
    pan: Vec2,
    fit_pending: bool,
    nearest: bool,
    /// Split position as a fraction of the canvas width (stays on screen at any zoom).
    split: f32,
    pub roi: Option<Roi>,
    /// Crop being edited, in image coordinates.
    pub crop: Option<Rect>,
    /// Hand-drawn mask shape of the selected layer: its unrotated bounding box.
    pub shape: Option<(Rect, ShapeKind)>,
    /// Box being drawn with the probe.
    probe: Option<Rect>,
    drag: Option<Drag>,
    last_canvas: Rect,
    /// Points of the stroke being painted.
    painting: Vec<[f32; 2]>,
    /// Straightening angle shown in the crop bar.
    pub straighten: f32,
}

impl Viewer {
    pub fn new(ctx: &egui::Context, display: Arc<ColorImage>, width: usize, height: usize) -> Self {
        Self {
            original: Layer::new(ctx, "original", display, false),
            result: None,
            mask: None,
            overlays: [None, None, None],
            reference: None,
            image_size: Vec2::new(width as f32, height as f32),
            zoom: 1.0,
            pan: Vec2::ZERO,
            fit_pending: true,
            nearest: false,
            split: 0.5,
            roi: None,
            crop: None,
            probe: None,
            shape: None,
            drag: None,
            last_canvas: Rect::NOTHING,
            painting: Vec::new(),
            straighten: 0.0,
        }
    }

    pub fn set_result(&mut self, ctx: &egui::Context, display: Arc<ColorImage>, roi: Option<Roi>) {
        match &mut self.result {
            Some((layer, r)) => {
                layer.replace(display, self.nearest);
                *r = roi;
            }
            None => self.result = Some((Layer::new(ctx, "result", display, self.nearest), roi)),
        }
    }

    pub fn set_overlay(&mut self, ctx: &egui::Context, which: Overlay, image: Option<(Arc<ColorImage>, Option<Roi>)>) {
        let slot = &mut self.overlays[which as usize];
        *slot = image.map(|(img, roi)| match slot.take() {
            Some((mut layer, _)) => {
                if !Arc::ptr_eq(&layer.image, &img) {
                    layer.replace(img, self.nearest);
                }
                (layer, roi)
            }
            None => (Layer::new(ctx, "overlay", img, self.nearest), roi),
        });
    }

    /// The snapshot shown as "A" in the A/B comparison.
    pub fn set_reference(&mut self, ctx: &egui::Context, image: Option<(Arc<ColorImage>, Option<Roi>)>) {
        self.reference = image.map(|(img, roi)| (Layer::new(ctx, "reference", img, self.nearest), roi));
    }

    pub fn set_mask_overlay(&mut self, ctx: &egui::Context, overlay: Option<(u64, Arc<ColorImage>)>) {
        match (overlay, &mut self.mask) {
            (Some((id, img)), Some((layer, current))) => {
                if !Arc::ptr_eq(&layer.image, &img) {
                    layer.replace(img, self.nearest);
                }
                *current = id;
            }
            (Some((id, img)), None) => self.mask = Some((Layer::new(ctx, "mask", img, self.nearest), id)),
            (None, _) => self.mask = None,
        }
    }

    #[cfg(test)]
    pub fn canvas(&self) -> Rect {
        self.last_canvas
    }

    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    pub fn request_fit(&mut self) {
        self.fit_pending = true;
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        let zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.pan *= zoom / self.zoom;
        self.zoom = zoom;
    }

    /// Place a default ROI in the middle of what is currently visible.
    pub fn ensure_roi(&mut self) {
        if self.roi.is_some() {
            return;
        }
        let centre = if self.last_canvas.is_positive() { self.to_image(self.last_canvas, self.last_canvas.center()) } else { (self.image_size / 2.0).to_pos2() };
        let half = Vec2::splat(DEFAULT_ROI / 2.0);
        self.roi = self.clamp_to_image(Rect::from_min_max(centre - half, centre + half)).map(to_roi);
    }

    /// Start editing a crop covering the whole image.
    pub fn begin_crop(&mut self) {
        self.crop = Some(Rect::from_min_size(Pos2::ZERO, self.image_size));
    }

    fn origin(&self, canvas: Rect) -> Pos2 {
        canvas.center() + self.pan - self.image_size * self.zoom / 2.0
    }

    fn to_screen(&self, canvas: Rect, p: Pos2) -> Pos2 {
        self.origin(canvas) + p.to_vec2() * self.zoom
    }

    /// The part of the image on screen (image coordinates).
    pub fn visible_rect(&self) -> Rect {
        let c = self.last_canvas;
        Rect::from_two_pos(self.to_image(c, c.min), self.to_image(c, c.max)).intersect(Rect::from_min_size(Pos2::ZERO, self.image_size))
    }

    fn to_image(&self, canvas: Rect, p: Pos2) -> Pos2 {
        ((p - self.origin(canvas)) / self.zoom).to_pos2()
    }

    fn rect_to_screen(&self, canvas: Rect, r: Rect) -> Rect {
        Rect::from_min_max(self.to_screen(canvas, r.min), self.to_screen(canvas, r.max))
    }

    fn clamp_to_image(&self, r: Rect) -> Option<Rect> {
        let bounds = Rect::from_min_size(Pos2::ZERO, self.image_size);
        let r = r.intersect(bounds);
        (r.width() >= MIN_RECT && r.height() >= MIN_RECT).then_some(r)
    }

    fn rect_of(&self, target: Target) -> Option<Rect> {
        match target {
            Target::Roi => self.roi.map(|r| Rect::from_min_size(pos2(r.x as f32, r.y as f32), Vec2::new(r.w as f32, r.h as f32))),
            Target::Crop => self.crop,
            Target::Shape => self.shape.map(|(r, _)| r),
            Target::Probe => self.probe,
        }
    }

    fn set_rect(&mut self, target: Target, r: Rect) {
        match target {
            Target::Roi => {
                if let Some(r) = self.clamp_to_image(r) {
                    self.roi = Some(to_roi(r));
                }
            }
            Target::Crop => {
                if let Some(r) = self.clamp_to_image(r) {
                    self.crop = Some(Rect::from_min_max(r.min.round(), r.max.round()));
                }
            }
            Target::Probe => self.probe = Some(r.intersect(Rect::from_min_size(Pos2::ZERO, self.image_size))),
            Target::Shape => {
                if r.width() >= MIN_RECT / 2.0
                    && r.height() >= MIN_RECT / 2.0
                    && let Some((rect, _)) = &mut self.shape
                {
                    *rect = r;
                }
            }
        }
    }

    /// Move `r` keeping its size, staying inside the image for ROI and crop.
    fn moved(&self, target: Target, r: Rect, min: Pos2) -> Rect {
        let mut min = min;
        if target != Target::Shape {
            min = min.clamp(Pos2::ZERO, (self.image_size - r.size()).max(Vec2::ZERO).to_pos2());
        }
        Rect::from_min_size(min, r.size())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn show(&mut self, ui: &mut egui::Ui, compare: Compare, tool: Tool, roi_mode: bool, busy: bool, show_mask: bool, brush: Option<BrushView>) -> ViewerOutput {
        let mut output = ViewerOutput::default();
        let (canvas, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        self.last_canvas = canvas;
        let painter = ui.painter_at(canvas);
        painter.rect_filled(canvas, 0.0, BG);

        if self.fit_pending && canvas.is_positive() {
            let fit = (canvas.size() / self.image_size).min_elem() * 0.92;
            self.zoom = fit.clamp(MIN_ZOOM, MAX_ZOOM);
            self.pan = Vec2::ZERO;
            self.fit_pending = false;
        }

        // A stroke in progress belongs to the brush: if the tool changed under it, or the drag
        // ended where we didn't see it (button released outside the window), drop it.
        let painting_now = matches!(self.drag, Some(Drag::Paint)) && tool == Tool::Brush && (ui.input(|i| i.pointer.primary_down()) || response.drag_stopped());
        if !painting_now {
            if matches!(self.drag, Some(Drag::Paint)) {
                self.drag = None;
            }
            self.painting.clear();
        }

        let target = match tool {
            Tool::Roi if roi_mode => Some(Target::Roi),
            Tool::Crop => Some(Target::Crop),
            Tool::Shape if self.shape.is_some() => Some(Target::Shape),
            // Always a new box: the last one isn't kept for editing.
            Tool::Probe => {
                if response.drag_started() {
                    self.probe = None;
                }
                Some(Target::Probe)
            }
            _ => None,
        };
        let split_x = canvas.left() + self.split * canvas.width();
        let near_split = |p: Pos2| matches!(compare, Compare::Split | Compare::Ab) && (p.x - split_x).abs() < GRAB;

        // ---- interaction -------------------------------------------------------------------
        if response.drag_started() {
            // Decide from where the button went down, not where the pointer is after the drag threshold.
            let origin = ui.input(|i| i.pointer.press_origin()).or(response.interact_pointer_pos());
            if let Some(p) = origin {
                let primary = response.dragged_by(PointerButton::Primary);
                let img_p = self.to_image(canvas, p);
                self.drag = Some(if !primary {
                    Drag::Pan
                } else if tool == Tool::Brush && brush.is_some() {
                    self.painting = vec![[img_p.x, img_p.y]];
                    Drag::Paint
                } else if near_split(p) {
                    Drag::Split
                } else if let Some(t) = target {
                    match self.rect_of(t) {
                        Some(r) => {
                            let screen = self.rect_to_screen(canvas, r);
                            let corners = [screen.left_top(), screen.right_top(), screen.left_bottom(), screen.right_bottom()];
                            match corners.iter().position(|c| c.distance(p) < GRAB) {
                                // Anchor at the opposite corner.
                                Some(i) => Drag::Resize { target: t, anchor: [r.right_bottom(), r.left_bottom(), r.right_top(), r.left_top()][i] },
                                None if screen.contains(p) => Drag::Move { target: t, grab: img_p - r.min },
                                None => Drag::Create { target: t, start: img_p },
                            }
                        }
                        None => Drag::Create { target: t, start: img_p },
                    }
                } else {
                    Drag::Pan
                });
            }
        }
        let pointer = response.interact_pointer_pos().or(response.hover_pos());
        if response.dragged()
            && let Some(p) = pointer
        {
            let img_p = self.to_image(canvas, p);
            match self.drag {
                Some(Drag::Pan) => self.pan += response.drag_delta(),
                Some(Drag::Split) => self.split = ((p.x - canvas.left()) / canvas.width()).clamp(0.0, 1.0),
                Some(Drag::Create { target, start }) | Some(Drag::Resize { target, anchor: start }) => {
                    self.set_rect(target, Rect::from_two_pos(start, img_p));
                }
                Some(Drag::Move { target, grab }) => {
                    if let Some(r) = self.rect_of(target) {
                        let r = self.moved(target, r, img_p - grab);
                        self.set_rect(target, r);
                    }
                }
                Some(Drag::Paint) => {
                    // Keep points about a quarter of the brush apart: smooth, but not wasteful.
                    let spacing = brush.map_or(1.0, |b| (b.radius * 0.25).max(0.5));
                    if self.painting.last().is_none_or(|l| Pos2::new(l[0], l[1]).distance(img_p) >= spacing) {
                        self.painting.push([img_p.x, img_p.y]);
                    }
                }
                None => {}
            }
        }
        if response.drag_stopped() {
            if matches!(self.drag, Some(Drag::Paint)) && !self.painting.is_empty() {
                output.stroke = Some(std::mem::take(&mut self.painting));
            }
            if tool == Tool::Probe {
                output.probed = self.probe.take().filter(|r| r.width() >= 3.0 && r.height() >= 3.0);
            }
            self.drag = None;
        }
        // A plain click with the probe: a box of 48 px around the point.
        if tool == Tool::Probe
            && response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            output.probed = Some(Rect::from_center_size(self.to_image(canvas, p), Vec2::splat(48.0)));
        }
        if tool == Tool::Pick
            && response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            let q = self.to_image(canvas, p);
            if q.x >= 0.0 && q.y >= 0.0 && q.x < self.image_size.x && q.y < self.image_size.y {
                output.picked = Some((q.x as usize, q.y as usize));
            }
        }
        if tool == Tool::Brush
            && brush.is_some()
            && response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            let q = self.to_image(canvas, p);
            output.stroke = Some(vec![[q.x, q.y]]);
        }
        if response.double_clicked() && target.is_none() && tool != Tool::Brush {
            self.fit_pending = true;
        }
        if let Some(cursor) = response.hover_pos() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (scroll * 0.0025).exp() * pinch;
            if factor != 1.0 {
                let new_zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
                // Keep the image point under the cursor fixed.
                let from_centre = cursor - canvas.center() - self.pan;
                self.pan -= from_centre * (new_zoom / self.zoom - 1.0);
                self.zoom = new_zoom;
            }
        }

        let hover_cursor = |p: Pos2| -> CursorIcon {
            if near_split(p) {
                return CursorIcon::ResizeHorizontal;
            }
            let Some(t) = target else { return CursorIcon::Grab };
            match self.rect_of(t).map(|r| self.rect_to_screen(canvas, r)) {
                Some(s) if [s.left_top(), s.right_bottom()].iter().any(|c| c.distance(p) < GRAB) => CursorIcon::ResizeNwSe,
                Some(s) if [s.right_top(), s.left_bottom()].iter().any(|c| c.distance(p) < GRAB) => CursorIcon::ResizeNeSw,
                Some(s) if s.contains(p) => CursorIcon::Move,
                _ => CursorIcon::Crosshair,
            }
        };
        let cursor = match (&self.drag, pointer) {
            (Some(Drag::Pan), _) => CursorIcon::Grabbing,
            (Some(Drag::Split), _) => CursorIcon::ResizeHorizontal,
            (Some(Drag::Move { .. }), _) => CursorIcon::Move,
            (Some(Drag::Create { .. } | Drag::Resize { .. }), _) => CursorIcon::Crosshair,
            // The brush outline is drawn instead.
            (Some(Drag::Paint), _) => CursorIcon::None,
            (None, Some(_)) if tool == Tool::Brush && brush.is_some() => CursorIcon::None,
            (None, Some(_)) if matches!(tool, Tool::Pick | Tool::Probe) => CursorIcon::Crosshair,
            (None, Some(p)) => hover_cursor(p),
            (None, None) => CursorIcon::Default,
        };
        if response.hovered() || self.drag.is_some() {
            ui.ctx().set_cursor_icon(cursor);
        }

        // Above ~300 % switch to nearest sampling so individual pixels are visible. Texture
        // options are fixed at upload, so re-upload only when crossing the threshold.
        let nearest = self.zoom >= 3.0;
        if nearest != self.nearest {
            self.nearest = nearest;
            let layers = std::iter::once(&mut self.original)
                .chain(self.result.as_mut().map(|(l, _)| l))
                .chain(self.mask.as_mut().map(|(l, _)| l))
                .chain(self.overlays.iter_mut().flatten().map(|(l, _)| l))
                .chain(self.reference.as_mut().map(|(l, _)| l));
            for layer in layers {
                layer.texture.set(Arc::clone(&layer.image), filter(nearest));
            }
        }

        // ---- drawing -----------------------------------------------------------------------
        let image_rect = Rect::from_min_size(self.origin(canvas), self.image_size * self.zoom);
        let full_uv = Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0));
        painter.image(self.original.texture.id(), image_rect, full_uv, Color32::WHITE);
        let place = |roi: &Option<Roi>| roi.map_or(image_rect, |r| self.rect_to_screen(canvas, self.rect_of_roi(r)));
        if matches!(compare, Compare::Ab | Compare::OnlyA)
            && let Some((layer, roi)) = &self.reference
        {
            painter.image(layer.texture.id(), place(roi), full_uv, Color32::WHITE);
        }

        let right_of_split = canvas.intersect(Rect::from_x_y_ranges(split_x..=canvas.right(), canvas.y_range()));
        let clip = if matches!(compare, Compare::Split | Compare::Ab) { right_of_split } else { canvas };
        let shows_result = !matches!(compare, Compare::Before | Compare::OnlyA);
        if let Some((layer, roi)) = &self.result
            && shows_result
        {
            painter.with_clip_rect(clip).image(layer.texture.id(), place(roi), full_uv, Color32::WHITE);
        }
        if shows_result {
            for (layer, roi) in self.overlays.iter().flatten() {
                painter.with_clip_rect(clip).image(layer.texture.id(), place(roi), full_uv, Color32::WHITE);
            }
        }
        if show_mask && let Some((layer, _)) = &self.mask {
            painter.image(layer.texture.id(), image_rect, full_uv, Color32::WHITE);
        }

        if roi_mode && let Some(r) = self.roi {
            let rect = self.rect_to_screen(canvas, self.rect_of_roi(r));
            shade_outside(&painter, image_rect.intersect(canvas), rect, 120);
            painter.rect_stroke(rect, 0.0, Stroke::new(1.5, ACCENT), StrokeKind::Outside);
            let editing = tool == Tool::Roi;
            if editing {
                corner_handles(&painter, rect, ACCENT);
            }
            pill(&painter, rect.left_top() - Vec2::new(0.0, 22.0), &format!("ROI {} × {}", r.w, r.h), ACCENT, Color32::BLACK);
            // Below the box's bottom-right corner (inside it when that would leave the canvas):
            // finish editing and go back to moving the view, or edit again.
            if matches!(tool, Tool::Roi | Tool::Pan) && self.drag.is_none() {
                let (icon, text, tip) = if editing {
                    (egui_material_icons::icons::ICON_CHECK, "Fatto", "Fine modifica della ROI: torna a spostare la vista")
                } else {
                    (egui_material_icons::icons::ICON_EDIT, "Modifica ROI", "Sposta o ridimensiona la ROI")
                };
                let size = Vec2::new(if editing { 82.0 } else { 124.0 }, 24.0);
                let mut at = rect.right_bottom() + Vec2::new(-size.x, 8.0);
                if at.y + size.y > canvas.bottom() - 4.0 {
                    at.y = rect.bottom() - size.y - 8.0;
                }
                at.x = at.x.max(canvas.left() + 4.0);
                let (fill, colour) = if editing { (ACCENT, Color32::BLACK) } else { (Color32::from_black_alpha(190), TEXT) };
                let button = egui::Button::new((icon.rich_text().color(colour), egui::RichText::new(text).strong().color(colour))).fill(fill).corner_radius(CornerRadius::same(12));
                if ui.put(Rect::from_min_size(at, size), button).on_hover_text(tip).clicked() {
                    output.roi_toggle = true;
                }
            }
        }

        if let Some((r, kind)) = self.shape {
            let stroke = Stroke::new(1.5, Color32::from_rgb(255, 110, 90));
            match kind {
                ShapeKind::Rectangle => {
                    painter.rect_stroke(self.rect_to_screen(canvas, r), 0.0, stroke, StrokeKind::Middle);
                }
                ShapeKind::Ellipse { angle_deg } => {
                    let (s, c) = angle_deg.to_radians().sin_cos();
                    let (centre, radii) = (r.center(), r.size() / 2.0);
                    let points = (0..96)
                        .map(|k| {
                            let t = k as f32 / 96.0 * std::f32::consts::TAU;
                            let (u, v) = (radii.x * t.cos(), radii.y * t.sin());
                            self.to_screen(canvas, centre + Vec2::new(u * c - v * s, u * s + v * c))
                        })
                        .collect();
                    painter.add(Shape::closed_line(points, stroke));
                }
            }
            if tool == Tool::Shape {
                corner_handles(&painter, self.rect_to_screen(canvas, r), stroke.color);
            }
        }

        if tool == Tool::Crop && let Some(r) = self.crop {
            let rect = self.rect_to_screen(canvas, r);
            shade_outside(&painter, image_rect.intersect(canvas), rect, 170);
            let thirds = Stroke::new(1.0, Color32::from_white_alpha(60));
            for k in 1..3 {
                let x = rect.left() + rect.width() * k as f32 / 3.0;
                let y = rect.top() + rect.height() * k as f32 / 3.0;
                painter.line_segment([pos2(x, rect.top()), pos2(x, rect.bottom())], thirds);
                painter.line_segment([pos2(rect.left(), y), pos2(rect.right(), y)], thirds);
            }
            painter.rect_stroke(rect, 0.0, Stroke::new(1.5, Color32::WHITE), StrokeKind::Outside);
            corner_handles(&painter, rect, Color32::WHITE);
            pill(&painter, rect.left_top() - Vec2::new(0.0, 22.0), &format!("Ritaglio {:.0} × {:.0}", r.width(), r.height()), Color32::WHITE, Color32::BLACK);
            output.crop_action = crop_bar(ui, canvas, &mut self.straighten);
        }

        if let Some(b) = brush.filter(|_| tool == Tool::Brush) {
            let colour = if b.erase { Color32::from_rgb(255, 90, 80) } else { Color32::from_rgb(90, 230, 140) };
            let r = b.radius * self.zoom;
            if !self.painting.is_empty() {
                let points: Vec<Pos2> = self.painting.iter().map(|p| self.to_screen(canvas, pos2(p[0], p[1]))).collect();
                let fill = colour.gamma_multiply(0.35);
                for p in &points {
                    painter.circle_filled(*p, r, fill);
                }
            }
            if let Some(p) = response.hover_pos().or(pointer) {
                painter.circle_stroke(p, r, Stroke::new(1.5, Color32::BLACK));
                painter.circle_stroke(p, r, Stroke::new(1.0, colour));
                painter.circle_filled(p, 1.5, colour);
            }
        }

        if matches!(compare, Compare::Split | Compare::Ab) {
            painter.line_segment([pos2(split_x, canvas.top()), pos2(split_x, canvas.bottom())], Stroke::new(1.5, Color32::WHITE));
            let handle = pos2(split_x, canvas.center().y);
            painter.circle_filled(handle, 13.0, Color32::from_black_alpha(170));
            painter.circle_stroke(handle, 13.0, Stroke::new(1.5, Color32::WHITE));
            for dir in [-1.0f32, 1.0] {
                let tip = handle + Vec2::new(dir * 7.0, 0.0);
                let base = handle + Vec2::new(dir * 2.5, 0.0);
                painter.add(Shape::convex_polygon(vec![tip, base + Vec2::new(0.0, -4.0), base + Vec2::new(0.0, 4.0)], Color32::WHITE, Stroke::NONE));
            }
            let (left, right) = if compare == Compare::Ab { ("A · FISSATA", "B · IN MODIFICA") } else { ("PRIMA", "DOPO") };
            for (text, x, align) in [(left, split_x - 10.0, egui::Align2::RIGHT_TOP), (right, split_x + 10.0, egui::Align2::LEFT_TOP)] {
                // Below the view bar floating at the top of the canvas.
                painter.text(pos2(x, canvas.top() + 56.0), align, text, FontId::proportional(10.0), Color32::from_white_alpha(200));
            }
        }

        if compare == Compare::Before {
            // Make it obvious that this is the untouched original (Space held, or "Prima").
            pill(&painter, egui::pos2(canvas.center().x - 30.0, canvas.top() + 56.0), "PRIMA · originale", Color32::WHITE, Color32::BLACK);
        }
        if compare == Compare::OnlyA {
            pill(&painter, egui::pos2(canvas.center().x - 40.0, canvas.top() + 56.0), "A · versione fissata", ACCENT, Color32::BLACK);
        }
        if tool == Tool::Pick {
            pill(&painter, egui::pos2(canvas.left() + 12.0, canvas.top() + 56.0), "Clicca un punto dell'immagine", ACCENT, Color32::BLACK);
        }
        if tool == Tool::Probe {
            if let Some(r) = self.probe {
                painter.rect_stroke(self.rect_to_screen(canvas, r), 0.0, Stroke::new(1.5, ACCENT), egui::StrokeKind::Middle);
            }
            pill(&painter, egui::pos2(canvas.left() + 12.0, canvas.top() + 56.0), "Disegna un riquadro attorno a una stella o a una luna · Esc per annullare", ACCENT, Color32::BLACK);
        }

        if busy {
            let pill_rect = Rect::from_min_size(canvas.right_top() + Vec2::new(-130.0, 12.0), Vec2::new(118.0, 26.0));
            painter.rect_filled(pill_rect, CornerRadius::same(13), Color32::from_black_alpha(180));
            ui.put(Rect::from_center_size(pill_rect.left_center() + Vec2::new(16.0, 0.0), Vec2::splat(14.0)), egui::Spinner::new().size(14.0).color(ACCENT));
            painter.text(pill_rect.left_center() + Vec2::new(30.0, 0.0), egui::Align2::LEFT_CENTER, "Elaborazione…", FontId::proportional(12.0), TEXT);
        }

        output.hovered = response.hovered();
        output.hover_pixel = response.hover_pos().and_then(|p| {
            let q = self.to_image(canvas, p);
            (q.x >= 0.0 && q.y >= 0.0 && q.x < self.image_size.x && q.y < self.image_size.y).then_some((q.x as usize, q.y as usize))
        });
        output
    }

    fn rect_of_roi(&self, r: Roi) -> Rect {
        Rect::from_min_size(pos2(r.x as f32, r.y as f32), Vec2::new(r.w as f32, r.h as f32))
    }
}

fn to_roi(r: Rect) -> Roi {
    Roi { x: r.min.x.round() as usize, y: r.min.y.round() as usize, w: r.width().round() as usize, h: r.height().round() as usize }
}

fn shade_outside(painter: &egui::Painter, outside: Rect, inside: Rect, alpha: u8) {
    let shade = Color32::from_black_alpha(alpha);
    for band in [
        Rect::from_min_max(outside.min, pos2(outside.max.x, inside.top())),
        Rect::from_min_max(pos2(outside.min.x, inside.bottom()), outside.max),
        Rect::from_min_max(pos2(outside.min.x, inside.top()), pos2(inside.left(), inside.bottom())),
        Rect::from_min_max(pos2(inside.right(), inside.top()), pos2(outside.max.x, inside.bottom())),
    ] {
        let band = band.intersect(outside);
        if band.is_positive() {
            painter.rect_filled(band, 0.0, shade);
        }
    }
}

fn corner_handles(painter: &egui::Painter, rect: Rect, color: Color32) {
    for c in [rect.left_top(), rect.right_top(), rect.left_bottom(), rect.right_bottom()] {
        painter.rect_filled(Rect::from_center_size(c, Vec2::splat(8.0)), CornerRadius::same(2), color);
    }
}

fn pill(painter: &egui::Painter, pos: Pos2, text: &str, fill: Color32, text_color: Color32) {
    let galley = painter.layout_no_wrap(text.to_string(), FontId::proportional(11.0), text_color);
    let rect = Rect::from_min_size(pos, galley.size() + Vec2::new(12.0, 6.0));
    painter.rect_filled(rect, CornerRadius::same(4), fill);
    painter.galley(rect.min + Vec2::new(6.0, 3.0), galley, text_color);
}

/// Floating bar while cropping: orientation tools, straightening, cancel/apply.
fn crop_bar(ui: &mut egui::Ui, canvas: Rect, straighten: &mut f32) -> Option<CropAction> {
    use egui_material_icons::icons::*;
    let size = Vec2::new(640.0, 40.0);
    let area = Rect::from_center_size(pos2(canvas.center().x, canvas.bottom() - 36.0), size);
    let mut action = None;
    ui.scope_builder(egui::UiBuilder::new().max_rect(area).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
        egui::Frame::new().fill(Color32::from_black_alpha(210)).corner_radius(CornerRadius::same(10)).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
            let tool = |ui: &mut egui::Ui, icon: egui_material_icons::MaterialIcon, tip: &str| ui.add(egui::Button::new(icon.rich_text().size(18.0)).frame(false)).on_hover_text(tip).clicked();
            if tool(ui, ICON_ROTATE_LEFT, "Ruota di 90° in senso antiorario") {
                action = Some(CropAction::Rotate(-1));
            }
            if tool(ui, ICON_ROTATE_RIGHT, "Ruota di 90° in senso orario") {
                action = Some(CropAction::Rotate(1));
            }
            if tool(ui, ICON_SWAP_HORIZ, "Specchia orizzontalmente") {
                action = Some(CropAction::Flip { horizontal: true });
            }
            if tool(ui, ICON_SWAP_VERT, "Specchia verticalmente") {
                action = Some(CropAction::Flip { horizontal: false });
            }
            ui.add(egui::Separator::default().vertical());
            ui.label(ICON_STRAIGHTEN.rich_text().size(16.0)).on_hover_text("Raddrizza");
            ui.spacing_mut().slider_width = 130.0;
            let slider = ui.add(egui::Slider::new(straighten, -45.0..=45.0).fixed_decimals(1).suffix("°"));
            if slider.drag_stopped() || (slider.changed() && !slider.dragged()) {
                action = Some(CropAction::Straighten(*straighten));
            }
            ui.add(egui::Separator::default().vertical());
            if ui.add(egui::Button::new("Annulla").min_size(Vec2::new(80.0, 26.0))).on_hover_text("Esc").clicked() {
                action = Some(CropAction::Cancel);
            }
            let apply = egui::Button::new(egui::RichText::new("Applica ritaglio").color(Color32::BLACK).strong()).fill(ACCENT).min_size(Vec2::new(130.0, 26.0));
            if ui.add(apply).on_hover_text("Invio").clicked() {
                action = Some(CropAction::Apply);
            }
        });
    });
    action
}
