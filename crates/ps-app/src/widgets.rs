//! Small custom widgets that give the app its look: switches, segmented controls,
//! labelled sliders with reset, and the histogram.

use crate::theme::{self, ACCENT, BG, BORDER, SURFACE_HOVER, TEXT, TEXT_WEAK};
use eframe::egui::{self, Color32, CornerRadius, Pos2, Rect, Response, RichText, Sense, Stroke, Ui, Vec2, emath};
use egui_material_icons::MaterialIcon;
use std::ops::RangeInclusive;

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

/// iOS-style animated on/off switch.
pub fn toggle(ui: &mut Ui, on: &mut bool) -> Response {
    let (rect, mut response) = ui.allocate_exact_size(Vec2::new(30.0, 17.0), Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    let t = ui.ctx().animate_bool_responsive(response.id, *on);
    let radius = rect.height() / 2.0;
    let painter = ui.painter();
    painter.rect_filled(rect, radius, lerp_color(BORDER, ACCENT, t));
    let x = egui::lerp(rect.left() + radius..=rect.right() - radius, t);
    painter.circle_filled(Pos2::new(x, rect.center().y), radius - 2.5, Color32::WHITE);
    response
}

/// A pill of mutually exclusive options. Returns true if the value changed.
pub fn segmented<T: PartialEq + Copy>(ui: &mut Ui, value: &mut T, options: &[(T, Option<MaterialIcon>, &str)]) -> bool {
    let mut changed = false;
    egui::Frame::new()
        .fill(BG)
        .corner_radius(CornerRadius::same(7))
        .inner_margin(egui::Margin::same(2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.spacing_mut().button_padding = Vec2::new(8.0, 3.0);
            // In a right-to-left parent the row inherits that direction; add the options in
            // reverse so they still read left to right.
            let reversed = ui.layout().main_dir() == egui::Direction::RightToLeft;
            ui.horizontal(|ui| {
                let ordered: Vec<_> = if reversed { options.iter().rev().collect() } else { options.iter().collect() };
                for &&(option, icon, label) in &ordered {
                    let selected = *value == option;
                    let color = if selected { ACCENT } else { TEXT_WEAK };
                    let text = RichText::new(label).color(color);
                    let button = match icon {
                        Some(icon) => egui::Button::new((icon.rich_text().color(color).size(15.0), text)),
                        None => egui::Button::new(text),
                    };
                    let button = button
                        .fill(if selected { SURFACE_HOVER } else { Color32::TRANSPARENT })
                        .stroke(Stroke::NONE)
                        .corner_radius(CornerRadius::same(5));
                    if ui.add(button).clicked() && !selected {
                        *value = option;
                        changed = true;
                    }
                }
            });
        });
    changed
}

/// Frameless icon button with tooltip.
pub fn icon_button(ui: &mut Ui, icon: MaterialIcon, tooltip: &str) -> Response {
    ui.add(egui::Button::new(icon.rich_text().size(17.0)).frame(false)).on_hover_text(tooltip)
}

/// Frameless toolbar button: icon and label, in the accent colour while its mode is on.
pub fn tool_button(ui: &mut Ui, icon: MaterialIcon, label: &str, on: bool) -> Response {
    let colour = if on { ACCENT } else { TEXT };
    ui.add(egui::Button::new((icon.rich_text().size(theme::size::ICON).color(colour), RichText::new(label).size(theme::size::BODY).color(colour))).frame(false))
}

/// One row: label, slider, editable value and a reset button, shown when the value differs
/// from `default` (double-clicking the slider resets too).
pub struct ParamSlider<'a, N: emath::Numeric> {
    label: &'a str,
    value: &'a mut N,
    range: RangeInclusive<N>,
    default: N,
    decimals: usize,
    logarithmic: bool,
    suffix: &'a str,
    help: Option<&'a str>,
}

impl<'a, N: emath::Numeric> ParamSlider<'a, N> {
    pub fn new(label: &'a str, value: &'a mut N, range: RangeInclusive<N>, default: N) -> Self {
        Self { label, value, range, default, decimals: 2, logarithmic: false, suffix: "", help: None }
    }

    pub fn decimals(mut self, decimals: usize) -> Self {
        self.decimals = decimals;
        self
    }

    pub fn logarithmic(mut self) -> Self {
        self.logarithmic = true;
        self
    }

    pub fn suffix(mut self, suffix: &'a str) -> Self {
        self.suffix = suffix;
        self
    }

    pub fn help(mut self, help: &'a str) -> Self {
        self.help = Some(help);
        self
    }

    /// Returns true if the value changed.
    pub fn show(self, ui: &mut Ui) -> bool {
        const VALUE_W: f32 = 64.0;
        const RESET_W: f32 = 16.0;
        let mut changed = false;
        let modified = self.value.to_f64() != self.default.to_f64();
        ui.horizontal(|ui| {
            let label_w = (ui.available_width() * 0.34).min(120.0);
            let row_h = ui.spacing().interact_size.y;
            ui.allocate_ui_with_layout(Vec2::new(label_w, row_h), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.set_width(label_w);
                let text = RichText::new(self.label).color(if modified { TEXT } else { TEXT_WEAK });
                let label = ui.add(egui::Label::new(text).truncate());
                if let Some(help) = self.help {
                    label.on_hover_text(help);
                }
            });
            // Leave a few points of slack: in a resizable panel, content even 1 pt wider than
            // the panel makes it grow every frame.
            ui.spacing_mut().slider_width = (ui.available_width() - VALUE_W - RESET_W - 3.0 * ui.spacing().item_spacing.x - 6.0).max(40.0);
            let mut slider = egui::Slider::new(&mut *self.value, self.range.clone()).show_value(false);
            if self.logarithmic {
                slider = slider.logarithmic(true).smallest_positive(1e-4);
            }
            let response = ui.add(slider);
            changed |= response.changed();
            if response.double_clicked() {
                *self.value = self.default;
                changed = true;
            }
            let speed = (self.range.end().to_f64() - self.range.start().to_f64()) / 300.0;
            let drag = egui::DragValue::new(self.value).range(self.range).speed(speed).fixed_decimals(self.decimals).suffix(self.suffix);
            changed |= ui.add_sized([VALUE_W, row_h - 4.0], drag).changed();
            // The space is always taken, so rows don't shift when the button appears.
            let (rect, response) = ui.allocate_exact_size(Vec2::new(RESET_W, row_h), if modified { Sense::click() } else { Sense::hover() });
            if modified {
                let icon = egui_material_icons::icons::ICON_RESTART_ALT;
                let color = if response.hovered() { ACCENT } else { TEXT_WEAK };
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, icon.codepoint, egui::FontId::new(14.0, icon.font_family()), color);
                if response.on_hover_text(format!("Ripristina ({})", fmt_default(self.default.to_f64(), self.decimals))).clicked() {
                    *self.value = self.default;
                    changed = true;
                }
            }
        });
        changed
    }
}

fn fmt_default(v: f64, decimals: usize) -> String {
    format!("{v:.decimals$}").replace('.', ",")
}

/// How the large histograms (levels, curves) are drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HistogramMode {
    /// Luminance, in grey.
    #[default]
    Luminance,
    /// Red, green and blue, overlaid.
    Rgb,
}

const BINS: usize = 256;

/// Per-channel 256-bin histogram (`channels`, each bin normalised to 0‥1, for the small one) and
/// the raw counts, per channel and of the luminance, that the large one redraws and remaps.
#[derive(Clone, Default)]
pub struct Histogram {
    pub channels: Vec<Vec<f32>>,
    counts: Vec<Vec<f32>>,
    luma: Vec<f32>,
}

/// √count scaled to the tallest bin away from the ends (the black sky and clipped highlights
/// would dwarf everything else).
fn normalised(series: &[&Vec<f32>]) -> Vec<Vec<f32>> {
    let max = series.iter().flat_map(|c| c[1..BINS - 1].iter().copied()).fold(1.0, f32::max).sqrt();
    series.iter().map(|c| c.iter().map(|v| (v.sqrt() / max).min(1.0)).collect()).collect()
}

impl Histogram {
    pub fn compute(img: &ps_core::Image) -> Self {
        // A strided sample keeps this under a few ms even on 50 MP images.
        let stride = (img.width() * img.height()).div_ceil(1 << 20).max(1);
        let bin = |v: f32| (v.clamp(0.0, 1.0) * (BINS - 1) as f32) as usize;
        let counts: Vec<Vec<f32>> = img
            .channels()
            .iter()
            .map(|plane| {
                let mut bins = vec![0.0f32; BINS];
                for &v in plane.iter().step_by(stride) {
                    bins[bin(v)] += 1.0;
                }
                bins
            })
            .collect();
        let mut luma = vec![0.0f32; BINS];
        match img.channels() {
            [r, g, b] => {
                for i in (0..r.len()).step_by(stride) {
                    luma[bin(ps_core::color::luma([r[i], g[i], b[i]]))] += 1.0;
                }
            }
            _ => luma.clone_from(&counts[0]),
        }
        Self::from_counts(counts, luma)
    }

    fn from_counts(counts: Vec<Vec<f32>>, luma: Vec<f32>) -> Self {
        let channels = normalised(&counts.iter().collect::<Vec<_>>());
        Self { channels, counts, luma }
    }

    /// The histogram after passing every value through `map(channel, v)` (`None` for the
    /// luminance and for mono images): how it will look once an adjustment is applied, without
    /// waiting for the image to be processed.
    pub fn mapped(&self, map: impl Fn(Option<usize>, f32) -> f32) -> Histogram {
        let colour = self.counts.len() == 3;
        let remap = |series: &[f32], channel: Option<usize>| {
            let mut out = vec![0.0f32; BINS];
            for (i, &n) in series.iter().enumerate() {
                if n > 0.0 {
                    let v = map(channel, i as f32 / (BINS - 1) as f32);
                    out[(v.clamp(0.0, 1.0) * (BINS - 1) as f32).round() as usize] += n;
                }
            }
            out
        };
        let counts = self.counts.iter().enumerate().map(|(c, series)| remap(series, colour.then_some(c))).collect();
        Self::from_counts(counts, remap(&self.luma, None))
    }

    /// Share of the pixels at pure black and at pure white (luminance).
    pub fn clipped(&self) -> (f32, f32) {
        let total: f32 = self.luma.iter().sum::<f32>().max(1.0);
        (self.luma.first().copied().unwrap_or(0.0) / total, self.luma.last().copied().unwrap_or(0.0) / total)
    }

    /// The small histogram under the panel.
    pub fn show(&self, ui: &mut Ui) {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 84.0), Sense::hover());
        let painter = ui.painter_at(rect);
        frame(&painter, rect);
        let colors: &[Color32] = match self.channels.len() {
            3 => &CHANNEL_COLOURS,
            _ => &[Color32::from_rgba_unmultiplied(210, 214, 222, 150)],
        };
        for (bins, &color) in self.channels.iter().zip(colors) {
            bars(&painter, rect.shrink(4.0), bins, color);
        }
    }

    /// The large histogram of the levels card. `ghost`: the histogram before the adjustment,
    /// drawn faintly behind; `channel`: draw only that channel. Below it, how much of the image
    /// is pushed to pure black and pure white.
    pub fn show_large(&self, ui: &mut Ui, mode: HistogramMode, ghost: Option<&Histogram>, channel: Option<usize>) {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 150.0), Sense::hover());
        let painter = ui.painter_at(rect);
        frame(&painter, rect);
        self.paint(&painter, rect.shrink(4.0), mode, ghost, channel, 1.0);
        self.clip_labels(ui);
    }

    /// Bars of this histogram over the outline of `ghost`, on the same scale so the change is a
    /// true comparison. `alpha` dims both (behind a curve).
    fn paint(&self, painter: &egui::Painter, inner: Rect, mode: HistogramMode, ghost: Option<&Histogram>, channel: Option<usize>, alpha: f32) {
        let series = |h: &Histogram| -> Vec<(Vec<f32>, Color32)> {
            match (channel.filter(|_| h.counts.len() == 3), mode, h.counts.len()) {
                (Some(c), _, _) => vec![(h.counts[c].clone(), CHANNEL_COLOURS[c])],
                (None, HistogramMode::Rgb, 3) => h.counts.iter().cloned().zip(CHANNEL_COLOURS).collect(),
                _ => vec![(h.luma.clone(), Color32::from_rgba_unmultiplied(215, 218, 226, 170))],
            }
        };
        let now = series(self);
        let before = ghost.map(series).unwrap_or_default();
        let all: Vec<&Vec<f32>> = now.iter().chain(&before).map(|(s, _)| s).collect();
        let scaled = normalised(&all);
        let (now_scaled, before_scaled) = scaled.split_at(now.len());
        for (bins, (_, colour)) in before_scaled.iter().zip(&before) {
            outline(painter, inner, bins, colour.gamma_multiply(0.5 * alpha));
        }
        for (bins, (_, colour)) in now_scaled.iter().zip(&now) {
            bars(painter, inner, bins, colour.gamma_multiply(alpha));
        }
    }

    /// How much of the image is at pure black and pure white.
    fn clip_labels(&self, ui: &mut Ui) {
        let (black, white) = self.clipped();
        let label = |frac: f32| if frac >= 0.001 { format!("{:.1} %", frac * 100.0) } else { "0 %".into() };
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("neri pieni {}", label(black))).size(theme::size::SMALL).color(TEXT_WEAK));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(format!("bianchi pieni {}", label(white))).size(theme::size::SMALL).color(TEXT_WEAK));
            });
        });
    }
}

/// The histogram drawn behind a curve: as the image will look (`now`), over how it is
/// (`before`).
pub struct Backdrop<'a> {
    pub now: &'a Histogram,
    pub before: &'a Histogram,
    pub mode: HistogramMode,
    pub channel: Option<usize>,
}

const CHANNEL_COLOURS: [Color32; 3] = [
    Color32::from_rgba_premultiplied(160, 60, 60, 160),
    Color32::from_rgba_premultiplied(70, 150, 85, 160),
    Color32::from_rgba_premultiplied(60, 95, 170, 160),
];

fn frame(painter: &egui::Painter, rect: Rect) {
    painter.rect_filled(rect, CornerRadius::same(theme::RADIUS), BG);
    for q in 1..4 {
        let x = rect.left() + rect.width() * q as f32 / 4.0;
        painter.line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(1.0, Color32::from_white_alpha(8)));
    }
}

fn bars(painter: &egui::Painter, inner: Rect, bins: &[f32], color: Color32) {
    let dx = inner.width() / bins.len() as f32;
    for (i, &v) in bins.iter().enumerate() {
        if v > 0.0 {
            let x = inner.left() + i as f32 * dx;
            painter.rect_filled(Rect::from_min_max(Pos2::new(x, inner.bottom() - v * inner.height()), Pos2::new(x + dx.max(1.0), inner.bottom())), 0.0, color);
        }
    }
}

fn outline(painter: &egui::Painter, inner: Rect, bins: &[f32], color: Color32) {
    let dx = inner.width() / bins.len() as f32;
    let points: Vec<Pos2> = bins.iter().enumerate().map(|(i, &v)| Pos2::new(inner.left() + (i as f32 + 0.5) * dx, inner.bottom() - v * inner.height())).collect();
    painter.add(egui::Shape::line(points, Stroke::new(1.0, color)));
}

/// Tone-curve editor: click to add a point, drag to move it, double-click to remove it.
/// The end points move only vertically. `backdrop`: the histogram drawn behind it, with the
/// black/white clipping below. Returns true if the curve changed.
/// `colour`: of the curve (the accent for the master curve, the channel's otherwise).
pub fn curve_editor(ui: &mut Ui, points: &mut Vec<[f32; 2]>, backdrop: Option<Backdrop>, colour: Color32) -> bool {
    const HIT: f32 = 9.0;
    let height = if backdrop.is_some() { ui.available_width().min(380.0) * 0.8 } else { ui.available_width().min(260.0) * 0.75 };
    let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::click_and_drag());
    let plot = rect.shrink(6.0);
    let to_screen = |p: [f32; 2]| Pos2::new(plot.left() + p[0] * plot.width(), plot.bottom() - p[1] * plot.height());
    let to_curve = |s: Pos2| [((s.x - plot.left()) / plot.width()).clamp(0.0, 1.0), ((plot.bottom() - s.y) / plot.height()).clamp(0.0, 1.0)];
    let drag_id = response.id.with("dragging");
    let mut changed = false;

    let nearest = |pos: Pos2, points: &[[f32; 2]]| {
        points.iter().enumerate().map(|(i, &p)| (i, to_screen(p).distance(pos))).filter(|(_, d)| *d < HIT).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(i, _)| i)
    };

    if response.drag_started()
        && let Some(pos) = ui.input(|i| i.pointer.press_origin())
    {
        let index = nearest(pos, points).unwrap_or_else(|| {
            points.push(to_curve(pos));
            points.sort_by(|a, b| a[0].total_cmp(&b[0]));
            changed = true;
            points.iter().position(|p| *p == to_curve(pos)).unwrap_or(0)
        });
        ui.data_mut(|d| d.insert_temp(drag_id, index));
    }
    if response.dragged()
        && let (Some(index), Some(pos)) = (ui.data(|d| d.get_temp::<usize>(drag_id)), response.interact_pointer_pos())
        && index < points.len()
    {
        let mut p = to_curve(pos);
        let last = points.len() - 1;
        p[0] = match index {
            0 => 0.0,
            i if i == last => 1.0,
            i => p[0].clamp(points[i - 1][0] + 0.01, points[i + 1][0] - 0.01),
        };
        if points[index] != p {
            points[index] = p;
            changed = true;
        }
    }
    if response.drag_stopped() {
        ui.data_mut(|d| d.remove::<usize>(drag_id));
    }
    if response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
        && nearest(pos, points).is_none()
    {
        points.push(to_curve(pos));
        points.sort_by(|a, b| a[0].total_cmp(&b[0]));
        changed = true;
    }
    if response.double_clicked()
        && let Some(pos) = response.interact_pointer_pos()
        && let Some(i) = nearest(pos, points)
        && i != 0
        && i != points.len() - 1
    {
        points.remove(i);
        changed = true;
    }

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::same(theme::RADIUS), BG);
    let grid = Stroke::new(1.0, Color32::from_white_alpha(10));
    for k in 1..4 {
        let t = k as f32 / 4.0;
        painter.line_segment([Pos2::new(plot.left() + t * plot.width(), plot.top()), Pos2::new(plot.left() + t * plot.width(), plot.bottom())], grid);
        painter.line_segment([Pos2::new(plot.left(), plot.top() + t * plot.height()), Pos2::new(plot.right(), plot.top() + t * plot.height())], grid);
    }
    if let Some(b) = &backdrop {
        b.now.paint(&painter, plot, b.mode, Some(b.before), b.channel, 0.45);
    }
    painter.line_segment([plot.left_bottom(), plot.right_top()], Stroke::new(1.0, Color32::from_white_alpha(25)));
    let lut = ps_core::adjust::curve_lut(points);
    let line: Vec<Pos2> = (0..=64).map(|k| k as f32 / 64.0).map(|x| to_screen([x, ps_core::adjust::lookup(&lut, x)])).collect();
    painter.add(egui::Shape::line(line, Stroke::new(2.0, colour)));
    for &p in points.iter() {
        painter.circle_filled(to_screen(p), 4.5, colour);
        painter.circle_stroke(to_screen(p), 4.5, Stroke::new(1.0, Color32::BLACK));
    }
    if let Some(b) = &backdrop {
        b.now.clip_labels(ui);
    }
    changed
}
