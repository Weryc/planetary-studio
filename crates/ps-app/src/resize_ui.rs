//! The "Ingrandisci" tab: how big the exported image is, and how it is interpolated.
//!
//! The size is kept as one factor (output / input, the aspect ratio is locked) and shown in the
//! unit the user picks: percent, pixels, or centimetres at a print resolution. The method is
//! chosen for the kind of subject unless the user picks one ([`ps_core::resample::method_for`]).
//! Enlarging happens on export only: the processing and the brush strokes stay at the
//! original's size.

use ps_core::i18n::tl;
use ps_core::tr;
use crate::panels::hint;
use crate::theme::{self, TEXT, TEXT_WEAK};
use crate::widgets::{ParamSlider, segmented, toggle};
use eframe::egui::{self, ColorImage, RichText, TextureHandle, TextureOptions, Ui, Vec2};
use ps_core::Image;
use ps_core::analysis::ImageKind;
use ps_core::resample::{self, Method, ResizeParams};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SizeUnit {
    Percent,
    Pixels,
    Centimetres,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResizeSettings {
    pub enabled: bool,
    /// Output size over input size.
    pub scale: f32,
    pub unit: SizeUnit,
    /// Print resolution for sizes in centimetres.
    pub dpi: f32,
    /// `None`: chosen for the subject.
    pub method: Option<Method>,
    pub anti_ringing: Option<f32>,
}

impl Default for ResizeSettings {
    fn default() -> Self {
        Self { enabled: false, scale: 2.0, unit: SizeUnit::Percent, dpi: 300.0, method: None, anti_ringing: None }
    }
}

const MIN_SCALE: f32 = 0.1;
const MAX_SCALE: f32 = 16.0;
/// No side longer than this (memory: 30000² RGB floats would be 10 GB).
const MAX_SIDE: f32 = 24_000.0;

impl ResizeSettings {
    fn clamp(&mut self, w: usize, h: usize) {
        let largest = w.max(h).max(1) as f32;
        self.scale = self.scale.clamp(MIN_SCALE, MAX_SCALE.min(MAX_SIDE / largest));
    }

    pub fn output_size(&self, w: usize, h: usize) -> (usize, usize) {
        (((w as f32 * self.scale).round() as usize).max(1), ((h as f32 * self.scale).round() as usize).max(1))
    }

    /// Method and anti-halo amount actually used, and why.
    pub fn method(&self, kind: ImageKind, sun: bool) -> (Method, f32, &'static str) {
        let (m, a, why) = resample::method_for(kind, sun);
        (self.method.unwrap_or(m), self.anti_ringing.unwrap_or(if self.method.is_some() && !self.method.unwrap().overshoots() { 0.0 } else { a }), why)
    }

    /// What export applies, if anything.
    pub fn params(&self, w: usize, h: usize, kind: ImageKind, sun: bool) -> Option<ResizeParams> {
        let (ow, oh) = self.output_size(w, h);
        if !self.enabled || (ow == w && oh == h) {
            return None;
        }
        let (method, anti_ringing, _) = self.method(kind, sun);
        Some(ResizeParams { width: ow, height: oh, method, anti_ringing })
    }
}

/// The two preview tiles (original pixels, chosen method), remade when what they show changes.
/// Image (by address), scale, method, anti-halo, centre.
type PreviewKey = (usize, u32, Method, u32, (u32, u32), u32, (u32, u32));

pub struct ResizePreview {
    key: Option<PreviewKey>,
    tiles: Option<[TextureHandle; 2]>,
    /// Magnification of the preview over the export's own pixels.
    zoom: f32,
    /// Where the preview looks, from the subject's centre (source pixels).
    pan: (f32, f32),
}

impl Default for ResizePreview {
    fn default() -> Self {
        Self { key: None, tiles: None, zoom: 1.0, pan: (0.0, 0.0) }
    }
}

const MAX_ZOOM: f32 = 8.0;

pub struct ResizeView<'a> {
    pub kind: ImageKind,
    pub sun: bool,
    /// The processed image (or the original while nothing is processed yet).
    pub image: Option<Arc<Image>>,
    /// Where to look in the preview: the subject's centre.
    pub centre: (f32, f32),
    pub preview: &'a mut ResizePreview,
}

pub fn resize_body(ui: &mut Ui, s: &mut ResizeSettings, view: ResizeView) {
    let ResizeView { kind, sun, image, centre, preview } = view;
    let Some(image) = image else { return };
    let (w, h) = (image.width(), image.height());
    s.clamp(w, h);
    // Drawn inside the export tab's own card.
    egui::Frame::new().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            toggle(ui, &mut s.enabled);
            ui.label(RichText::new(tl("Cambia la dimensione")).color(TEXT)).on_hover_text(tl("Metodi matematici classici: stessa immagine e stesse impostazioni danno sempre gli stessi pixel. Nessun dettaglio viene inventato."));
            if !s.enabled {
                ui.label(RichText::new(tl("spento: dimensioni originali")).size(theme::size::SMALL).color(TEXT_WEAK));
            }
        });
        ui.add_enabled_ui(s.enabled, |ui| {
            let mut percent = s.scale * 100.0;
            if ParamSlider::new(tl("Dimensione"), &mut percent, MIN_SCALE * 100.0..=800.0, 200.0).decimals(0).suffix(" %").show(ui) {
                s.scale = percent / 100.0;
            }
            ui.horizontal(|ui| {
                for f in [0.5f32, 1.5, 2.0, 3.0, 4.0] {
                    if ui.selectable_label((s.scale - f).abs() < 1e-3, format!("×{f}")).clicked() {
                        s.scale = f;
                    }
                }
            });
            s.clamp(w, h);
            let (ow, oh) = s.output_size(w, h);
            ui.label(RichText::new(format!("{w} × {h} → {ow} × {oh} px")).color(TEXT).strong());
            let (method, _, _) = s.method(kind, sun);
            ui.label(RichText::new(tr!("Metodo: {}{}", "Method: {}{}", method.label(), if s.method.is_none() { tl(" (scelto per il soggetto)") } else { "" })).size(theme::size::SMALL).color(TEXT_WEAK));
            ui.add_space(theme::space::XS);
            egui::CollapsingHeader::new(RichText::new(tl("Avanzate")).size(theme::size::SMALL).color(TEXT_WEAK)).id_salt("resize-advanced").default_open(false).show(ui, |ui| advanced(ui, s, (w, h), kind, sun, &image, centre, preview));
        });
    });
}

#[allow(clippy::too_many_arguments)]
fn advanced(ui: &mut Ui, s: &mut ResizeSettings, (w, h): (usize, usize), kind: ImageKind, sun: bool, image: &Arc<Image>, centre: (f32, f32), preview: &mut ResizePreview) {
    let section = |ui: &mut Ui, title: &str| {
        ui.add_space(theme::space::S);
        ui.label(RichText::new(title).size(theme::size::SMALL).strong().color(TEXT_WEAK));
    };
    section(ui, tl("DIMENSIONE IN"));
    segmented(ui, &mut s.unit, &[(SizeUnit::Percent, None, tl("Percentuale")), (SizeUnit::Pixels, None, "Pixel"), (SizeUnit::Centimetres, None, tl("Centimetri"))]);
    match s.unit {
        SizeUnit::Percent => {}
        SizeUnit::Pixels => {
            let (mut ow, mut oh) = s.output_size(w, h);
            ui.horizontal(|ui| {
                ui.label(tl("Larghezza"));
                if ui.add(egui::DragValue::new(&mut ow).range(1..=MAX_SIDE as usize).suffix(" px")).changed() {
                    s.scale = ow as f32 / w as f32;
                }
                ui.label(tl("Altezza"));
                if ui.add(egui::DragValue::new(&mut oh).range(1..=MAX_SIDE as usize).suffix(" px")).changed() {
                    s.scale = oh as f32 / h as f32;
                }
            });
            hint(ui, tl("Le proporzioni restano quelle dell'immagine."));
        }
        SizeUnit::Centimetres => {
            let (ow, oh) = s.output_size(w, h);
            let (mut cw, mut ch) = (resample::cm_from_px(ow as f32, s.dpi), resample::cm_from_px(oh as f32, s.dpi));
            ui.horizontal(|ui| {
                ui.label(tl("Larghezza"));
                if ui.add(egui::DragValue::new(&mut cw).range(0.1..=500.0).speed(0.1).max_decimals(1).suffix(" cm")).changed() {
                    s.scale = resample::px_from_cm(cw, s.dpi) / w as f32;
                }
                ui.label(tl("Altezza"));
                if ui.add(egui::DragValue::new(&mut ch).range(0.1..=500.0).speed(0.1).max_decimals(1).suffix(" cm")).changed() {
                    s.scale = resample::px_from_cm(ch, s.dpi) / h as f32;
                }
            });
            ui.horizontal(|ui| {
                ui.label(tl("Stampa a"));
                // Changing the resolution keeps the size in centimetres.
                let before = s.dpi;
                let mut dpi = s.dpi;
                if ui.add(egui::DragValue::new(&mut dpi).range(72.0..=1200.0).speed(1.0).suffix(" dpi")).changed() {
                    s.dpi = dpi;
                }
                for d in [150.0, 300.0, 600.0] {
                    if ui.selectable_label(s.dpi == d, format!("{d:.0}")).clicked() {
                        s.dpi = d;
                    }
                }
                if s.dpi != before {
                    s.scale *= s.dpi / before;
                }
            });
        }
    }
    s.clamp(w, h);

    section(ui, tl("METODO"));
    let (auto_method, auto_anti, why) = resample::method_for(kind, sun);
    let mut automatic = s.method.is_none();
    ui.horizontal(|ui| {
        if toggle(ui, &mut automatic).changed() {
            s.method = if automatic { None } else { Some(auto_method) };
            s.anti_ringing = None;
        }
        ui.label(RichText::new(tr!("Scelto per {}", "Chosen for {}", kind_label(kind, sun))).color(TEXT));
    });
    if automatic {
        hint(ui, why);
    } else {
        let mut current = s.method.unwrap_or(auto_method);
        egui::ComboBox::from_id_salt("resize-method").width(ui.available_width() - 8.0).selected_text(current.label()).show_ui(ui, |ui| {
            for m in Method::ALL {
                ui.selectable_value(&mut current, m, m.label()).on_hover_text(m.description());
            }
        });
        if Some(current) != s.method {
            s.method = Some(current);
            s.anti_ringing = None;
        }
        hint(ui, current.description());
    }
    let (method, anti, _) = s.method(kind, sun);
    if method.overshoots() {
        let mut a = anti;
        if ParamSlider::new(tl("Anti-alone"), &mut a, 0.0..=1.0, if automatic { auto_anti } else { 0.0 }).help(tl("Toglie il bordino scuro o chiaro che i metodi nitidi disegnano lungo i bordi netti (il lembo contro il cielo, le ombre)")).show(ui) {
            s.anti_ringing = Some(a);
        }
    }

    section(ui, tl("ANTEPRIMA"));
    let shown_scale = s.scale.max(1.0);
    let width = ui.available_width().floor().clamp(120.0, 520.0);
    let height = (width * 0.62).floor();
    let at = (centre.0 + preview.pan.0, centre.1 + preview.pan.1);
    let key = (Arc::as_ptr(image) as usize, shown_scale.to_bits(), method, anti.to_bits(), (at.0 as u32, at.1 as u32), preview.zoom.to_bits(), (width as u32, height as u32));
    if preview.key != Some(key) {
        preview.key = Some(key);
        preview.tiles = make_tiles(ui.ctx(), image, at, shown_scale, method, anti, preview.zoom, (width as usize, height as usize));
    }
    if preview.tiles.is_some() {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Zoom").size(theme::size::SMALL).color(TEXT_WEAK));
            ui.spacing_mut().slider_width = (ui.available_width() - 120.0).max(60.0);
            ui.add(egui::Slider::new(&mut preview.zoom, 1.0..=MAX_ZOOM).logarithmic(true).show_value(false)).on_hover_text(tl("Avvicina o allontana l'anteprima"));
            ui.label(RichText::new(format!("×{:.1}", preview.zoom)).size(theme::size::SMALL).color(TEXT));
            if ui.small_button(tl("Centra")).on_hover_text(tl("Torna alla vista iniziale")).clicked() {
                preview.zoom = 1.0;
                preview.pan = (0.0, 0.0);
            }
        });
    }
    let mut pan = egui::Vec2::ZERO;
    if let Some(tiles) = &preview.tiles {
        for (tex, label) in tiles.iter().zip([tl("Pixel originali"), method.label()]) {
            let r = ui.add(egui::Image::new(tex).fit_to_exact_size(Vec2::new(width, height)).corner_radius(4).sense(egui::Sense::hover()));
            if r.hovered() {
                // The wheel (or two-finger scroll) explores the image; zoom is the slider's.
                pan -= ui.input(|i| i.smooth_scroll_delta);
            }
            ui.label(RichText::new(label).size(theme::size::SMALL).color(TEXT_WEAK));
        }
        hint(ui, if s.scale < 1.0 { tl("In riduzione l'anteprima mostra il metodo al 100 %.") } else { tl("Rotella o trackpad: sposta l'anteprima su altre zone.") });
    }
    if pan != Vec2::ZERO {
        let k = 1.0 / (shown_scale * preview.zoom);
        preview.pan.0 = (preview.pan.0 + pan.x * k).clamp(-(w as f32), w as f32);
        preview.pan.1 = (preview.pan.1 + pan.y * k).clamp(-(h as f32), h as f32);
    }
}

fn kind_label(kind: ImageKind, sun: bool) -> &'static str {
    match kind {
        ImageKind::Planet => tl("un pianeta"),
        ImageKind::Saturn => tl("Saturno"),
        ImageKind::FullDisk if sun => tl("il disco solare"),
        ImageKind::FullDisk => tl("un disco intero"),
        ImageKind::Surface => tl("una superficie (Luna o Sole da vicino)"),
    }
}

/// A crop around `centre` (`zoom` times closer than the export's own pixels), enlarged to the
/// tile with nearest pixels and with `method`.
#[allow(clippy::too_many_arguments)]
fn make_tiles(ctx: &egui::Context, image: &Image, centre: (f32, f32), scale: f32, method: Method, anti: f32, zoom: f32, tile: (usize, usize)) -> Option<[TextureHandle; 2]> {
    let (w, h) = (image.width(), image.height());
    // Source pixels covered by the tile; the tile is later stretched `zoom` times by the screen.
    let cover = |side: usize| (((side as f32 / (scale * zoom)).ceil()) as usize).clamp(4, w.min(h));
    let (cw, ch) = (cover(tile.0).min(w), cover(tile.1).min(h));
    let x = (centre.0 as isize - cw as isize / 2).clamp(0, (w - cw) as isize) as usize;
    let y = (centre.1 as isize - ch as isize / 2).clamp(0, (h - ch) as isize) as usize;
    let crop = image.crop(x, y, cw, ch);
    let (ow, oh) = (((cw as f32 * scale).round() as usize).max(1), ((ch as f32 * scale).round() as usize).max(1));
    let make = |m: Method, a: f32, name: &str| {
        let img = resample::resize(&crop, &ResizeParams { width: ow, height: oh, method: m, anti_ringing: a });
        let colour = ColorImage::from_rgba_unmultiplied([ow, oh], &img.to_rgba8());
        ctx.load_texture(name, colour, TextureOptions::NEAREST)
    };
    Some([make(Method::Nearest, 0.0, "resize-before"), make(method, anti, "resize-after")])
}
