//! Dark, low-glare theme with a warm amber accent (easy on night-adapted eyes).

use eframe::egui::{self, Color32, CornerRadius, FontFamily, FontId, Margin, Shadow, Stroke, TextStyle, Vec2};

pub const ACCENT: Color32 = Color32::from_rgb(240, 168, 72);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(120, 84, 38);
pub const BG: Color32 = Color32::from_rgb(12, 13, 16);
pub const PANEL: Color32 = Color32::from_rgb(20, 22, 26);
pub const SURFACE: Color32 = Color32::from_rgb(28, 31, 37);
pub const SURFACE_HOVER: Color32 = Color32::from_rgb(36, 40, 47);
pub const BORDER: Color32 = Color32::from_rgb(40, 44, 52);
pub const TEXT: Color32 = Color32::from_rgb(222, 225, 231);
pub const TEXT_WEAK: Color32 = Color32::from_rgb(132, 139, 152);
pub const SUCCESS: Color32 = Color32::from_rgb(110, 196, 140);
pub const WARNING: Color32 = Color32::from_rgb(230, 190, 90);
pub const DANGER: Color32 = Color32::from_rgb(230, 105, 95);

pub const RADIUS: u8 = radius::M;

/// Corner radii (points). Pills (radius = half their height) and hand-drawn shapes keep their own.
pub mod radius {
    /// Thumbnails, small badges.
    pub const S: u8 = 4;
    /// Controls, boxes inside a card.
    pub const M: u8 = 6;
    /// Cards, nested groups, stand-alone buttons.
    pub const L: u8 = 8;
    /// Windows and dialogs.
    pub const WINDOW: u8 = 10;
}

/// The one spacing scale of the interface (points): every gap is one of these.
pub mod space {
    /// Between a label and what it labels.
    pub const XS: f32 = 2.0;
    /// Between rows of controls.
    pub const S: f32 = 4.0;
    /// Between groups of controls inside a card.
    pub const M: f32 = 8.0;
    /// Between cards and sections.
    pub const L: f32 = 12.0;
}

/// The text sizes of the interface (points).
pub mod size {
    /// Window and dialog titles.
    pub const HEADING: f32 = 15.0;
    /// Card titles.
    pub const TITLE: f32 = 14.0;
    /// Controls and values.
    pub const BODY: f32 = 13.0;
    /// Hints, section labels, units.
    pub const SMALL: f32 = 11.0;
    /// Small icons: in segmented controls, beside hints and status lines.
    pub const ICON_SMALL: f32 = 14.0;
    /// Icons next to body text.
    pub const ICON: f32 = 17.0;
}

/// Name of the Alt/Option key as the platform writes it.
pub fn alt_key() -> &'static str {
    if cfg!(target_os = "macos") { "⌥" } else { "Alt" }
}

/// egui's bundled fonts lack symbols such as ⌘ ⇧ ⌥ ″ (used by shortcut labels on macOS):
/// borrow them from a system symbol font, as a last-resort fallback.
fn add_symbol_fallback(ctx: &egui::Context) {
    use egui::epaint::text::{FontData, FontInsert, FontPriority, InsertFontFamily};
    const CANDIDATES: [&str; 3] = [
        "/System/Library/Fonts/Apple Symbols.ttf",
        r"C:\Windows\Fonts\seguisym.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ];
    if let Some(bytes) = CANDIDATES.iter().find_map(|p| std::fs::read(p).ok()) {
        let families = [FontFamily::Proportional, FontFamily::Monospace].map(|family| InsertFontFamily { family, priority: FontPriority::Lowest });
        ctx.add_font(FontInsert::new("system-symbols", FontData::from_owned(bytes), families.to_vec()));
    }
}

pub fn apply(ctx: &egui::Context) {
    egui_material_icons::initialize(ctx);
    add_symbol_fallback(ctx);
    ctx.set_theme(egui::Theme::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.text_styles = [
            (TextStyle::Heading, FontId::new(size::HEADING, FontFamily::Proportional)),
            (TextStyle::Body, FontId::new(13.0, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(13.0, FontFamily::Proportional)),
            (TextStyle::Small, FontId::new(11.0, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(12.0, FontFamily::Monospace)),
        ]
        .into();

        let s = &mut style.spacing;
        s.item_spacing = Vec2::new(8.0, 6.0);
        s.button_padding = Vec2::new(10.0, 5.0);
        s.interact_size.y = 26.0;
        s.slider_width = 220.0;
        s.menu_margin = Margin::same(6);
        s.window_margin = Margin::same(12);

        let v = &mut style.visuals;
        v.dark_mode = true;
        v.override_text_color = None;
        v.panel_fill = PANEL;
        v.window_fill = PANEL;
        v.window_stroke = Stroke::new(1.0, BORDER);
        v.window_corner_radius = CornerRadius::same(radius::WINDOW);
        v.menu_corner_radius = CornerRadius::same(radius::L);
        v.window_shadow = Shadow { offset: [0, 6], blur: 24, spread: 0, color: Color32::from_black_alpha(140) };
        v.popup_shadow = Shadow { offset: [0, 4], blur: 16, spread: 0, color: Color32::from_black_alpha(120) };
        v.extreme_bg_color = BG;
        v.faint_bg_color = SURFACE;
        v.code_bg_color = SURFACE;
        v.hyperlink_color = ACCENT;
        v.selection.bg_fill = ACCENT_DIM;
        v.selection.stroke = Stroke::new(1.0, ACCENT);
        v.slider_trailing_fill = true;
        v.handle_shape = egui::style::HandleShape::Circle;

        let r = CornerRadius::same(RADIUS);
        let w = &mut v.widgets;
        w.noninteractive.bg_fill = PANEL;
        w.noninteractive.weak_bg_fill = PANEL;
        w.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
        w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        w.noninteractive.corner_radius = r;

        w.inactive.bg_fill = SURFACE_HOVER;
        w.inactive.weak_bg_fill = SURFACE;
        w.inactive.bg_stroke = Stroke::NONE;
        w.inactive.fg_stroke = Stroke::new(1.0, TEXT);
        w.inactive.corner_radius = r;

        w.hovered.bg_fill = Color32::from_rgb(52, 57, 66);
        w.hovered.weak_bg_fill = SURFACE_HOVER;
        w.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(64, 70, 80));
        w.hovered.fg_stroke = Stroke::new(1.5, Color32::WHITE);
        w.hovered.corner_radius = r;
        w.hovered.expansion = 0.0;

        w.active.bg_fill = ACCENT;
        w.active.weak_bg_fill = Color32::from_rgb(46, 50, 58);
        w.active.bg_stroke = Stroke::new(1.0, ACCENT);
        w.active.fg_stroke = Stroke::new(1.5, Color32::WHITE);
        w.active.corner_radius = r;
        w.active.expansion = 0.0;

        w.open = w.hovered;
    });
}

/// Mix of two opaque colours, `t` = 0 → `a`, 1 → `b`.
pub fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    egui::Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

/// Rounded card used for pipeline steps and parameter groups.
pub fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE)
        .corner_radius(CornerRadius::same(radius::L))
        .inner_margin(Margin::same(10))
}
