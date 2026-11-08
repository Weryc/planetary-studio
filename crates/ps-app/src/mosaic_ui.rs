//! The mosaic window: the frames to join, how they line up (found automatically, nudged by
//! dragging), a preview; then the mosaic becomes the image being processed.

use crate::theme::{self, ACCENT, BORDER, TEXT, TEXT_WEAK};
use crate::widgets::{ParamSlider, toggle};
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, RichText, Sense, Stroke, TextureHandle, Vec2};
use egui_material_icons::icons::*;
use ps_core::Image;
use ps_core::mosaic::{self, Mosaic, PairMatch, Placement, Plane};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

/// What the window asks the app to do.
pub enum MosaicOutcome {
    /// Join the frames at full size and open the result. `keep_settings`: it replaces the open
    /// mosaic, whose processing settings stay.
    Create { mosaic: Mosaic, images: Vec<Option<Arc<Image>>>, keep_settings: bool },
}

/// Results of the background work.
enum Job {
    Loaded(Vec<Result<Arc<Image>, String>>),
    Progress(String),
    Matched(Vec<((usize, usize), PairMatch)>),
}

pub struct MosaicEditor {
    pub open: bool,
    mosaic: Mosaic,
    images: Vec<Option<Arc<Image>>>,
    /// Pairwise overlaps, from which placements follow (empty for a saved mosaic until
    /// realigned).
    matches: Vec<((usize, usize), PairMatch)>,
    /// What is being done in the background.
    busy: Option<String>,
    errors: Vec<String>,
    tx: Sender<Job>,
    rx: Receiver<Job>,
    /// The preview, its scale and the mosaic coordinates of its top-left corner.
    preview: Option<(TextureHandle, f32, [f32; 2])>,
    dirty: bool,
    selected: Option<usize>,
    /// The tile being dragged and where it started.
    drag: Option<(usize, Placement)>,
    /// The mosaic being edited is the open document.
    editing_open: bool,
}

impl Default for MosaicEditor {
    fn default() -> Self {
        let (tx, rx) = channel();
        Self {
            open: false,
            mosaic: Mosaic::default(),
            images: Vec::new(),
            matches: Vec::new(),
            busy: None,
            errors: Vec::new(),
            tx,
            rx,
            preview: None,
            dirty: false,
            selected: None,
            drag: None,
            editing_open: false,
        }
    }
}

impl MosaicEditor {
    /// Loading or aligning, or the preview not drawn yet.
    #[cfg(test)]
    pub fn is_busy(&self) -> bool {
        self.busy.is_some() || self.dirty
    }

    /// A new mosaic of these frames, aligned automatically.
    pub fn start(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        *self = Self { open: true, mosaic: Mosaic::new(paths), ..Default::default() };
        self.load(ctx, true);
    }

    /// Edit a mosaic (the open one, or a saved one) as it was joined.
    pub fn edit(&mut self, ctx: &egui::Context, mosaic: Mosaic, editing_open: bool) {
        *self = Self { open: true, mosaic, editing_open, ..Default::default() };
        self.load(ctx, false);
    }

    fn load(&mut self, ctx: &egui::Context, align: bool) {
        let paths: Vec<PathBuf> = self.mosaic.tiles.iter().map(|t| t.path.clone()).collect();
        self.busy = Some(format!("Caricamento di {} immagini…", paths.len()));
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let images: Vec<Result<Arc<Image>, String>> =
                paths.iter().map(|p| ps_io::load(p).map(Arc::new).map_err(|e| format!("{}: {e}", p.file_name().and_then(|n| n.to_str()).unwrap_or("?")))).collect();
            let loaded: Vec<Option<Arc<Image>>> = images.iter().map(|r| r.as_ref().ok().cloned()).collect();
            tx.send(Job::Loaded(images)).ok();
            ctx.request_repaint();
            if align {
                Self::align_in_background(&tx, &ctx, &loaded);
            }
        });
    }

    fn realign(&mut self, ctx: &egui::Context) {
        self.busy = Some("Allineamento…".into());
        let (tx, ctx, images) = (self.tx.clone(), ctx.clone(), self.images.clone());
        std::thread::spawn(move || Self::align_in_background(&tx, &ctx, &images));
    }

    fn align_in_background(tx: &Sender<Job>, ctx: &egui::Context, images: &[Option<Arc<Image>>]) {
        tx.send(Job::Progress("Allineamento…".into())).ok();
        let planes: Vec<Option<Plane>> = images.iter().map(|i| i.as_deref().map(Plane::luminance)).collect();
        let refs: Vec<Option<&Plane>> = planes.iter().map(Option::as_ref).collect();
        let report = |done: usize, total: usize| {
            tx.send(Job::Progress(format!("Allineamento: coppia {done} di {total}…"))).ok();
            ctx.request_repaint();
        };
        let matches = mosaic::match_all(&refs, &report);
        tx.send(Job::Matched(matches)).ok();
        ctx.request_repaint();
    }

    /// Placements from the overlaps for the enabled frames, then their brightness.
    fn arrange(&mut self) {
        if !self.matches.is_empty() {
            let enabled: Vec<bool> = self.mosaic.tiles.iter().zip(&self.images).map(|(t, i)| t.enabled && i.is_some()).collect();
            for (tile, p) in self.mosaic.tiles.iter_mut().zip(mosaic::tree(&enabled, &self.matches)) {
                tile.placement = p;
            }
        }
        self.update_gains();
    }

    fn update_gains(&mut self) {
        let placements: Vec<Option<Placement>> = self.mosaic.tiles.iter().map(|t| t.placement).collect();
        let gains = mosaic::brightness(&self.used_images(), &placements);
        for (tile, g) in self.mosaic.tiles.iter_mut().zip(gains) {
            tile.gain = g;
        }
        self.dirty = true;
    }

    /// The enabled, loaded frames.
    fn used_images(&self) -> Vec<Option<&Image>> {
        self.mosaic.tiles.iter().zip(&self.images).map(|(t, i)| i.as_deref().filter(|_| t.enabled)).collect()
    }

    fn receive(&mut self) {
        while let Ok(job) = self.rx.try_recv() {
            match job {
                Job::Loaded(images) => {
                    self.errors = images.iter().filter_map(|r| r.as_ref().err().cloned()).collect();
                    self.images = images.into_iter().map(Result::ok).collect();
                    self.busy = None;
                    self.update_gains();
                }
                Job::Progress(text) => self.busy = Some(text),
                Job::Matched(matches) => {
                    self.matches = matches;
                    self.busy = None;
                    self.arrange();
                }
            }
        }
    }

    fn refresh_preview(&mut self, ctx: &egui::Context) {
        self.dirty = false;
        let images = self.used_images();
        let placements: Vec<Option<Placement>> = self.mosaic.tiles.iter().map(|t| t.placement).collect();
        let gains: Vec<[f32; 3]> = self.mosaic.tiles.iter().map(|t| t.gain).collect();
        let sizes: Vec<(usize, usize)> = images.iter().map(|i| i.map_or((0, 0), |i| (i.width(), i.height()))).collect();
        let used: Vec<Option<Placement>> = placements.iter().zip(&images).map(|(p, i)| p.filter(|_| i.is_some())).collect();
        let Some((ox, oy, w, h)) = mosaic::bounds(&sizes, &used) else {
            self.preview = None;
            return;
        };
        let scale = (1400.0 / w.max(h) as f32).min(1.0);
        let Some(img) = mosaic::compose(&images, &placements, &gains, &self.mosaic.params, scale) else {
            self.preview = None;
            return;
        };
        let pixels = ColorImage::from_rgba_unmultiplied([img.width(), img.height()], &img.to_rgba8());
        let texture = ctx.load_texture("mosaic-preview", pixels, egui::TextureOptions::LINEAR);
        self.preview = Some((texture, scale, [ox, oy]));
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<MosaicOutcome> {
        if !self.open {
            return None;
        }
        self.receive();
        if self.dirty && self.drag.is_none() {
            self.refresh_preview(ctx);
        }
        let mut outcome = None;
        let mut open = self.open;
        egui::Window::new(RichText::new("Mosaico").strong())
            .open(&mut open)
            .default_size([1000.0, 640.0])
            .min_size([700.0, 420.0])
            .collapsible(false)
            .frame(egui::Frame::window(&ctx.global_style()).fill(theme::PANEL).stroke(Stroke::new(1.0, BORDER)))
            .show(ctx, |ui| {
                egui::Panel::right("mosaic-side").resizable(false).exact_size(300.0).frame(egui::Frame::NONE.inner_margin(egui::Margin { left: 12, ..Default::default() })).show(ui, |ui| {
                    outcome = self.side(ui);
                });
                egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| self.canvas(ui));
            });
        // Closed with ×, or by creating the mosaic.
        self.open &= open;
        outcome
    }

    fn side(&mut self, ui: &mut egui::Ui) -> Option<MosaicOutcome> {
        let mut outcome = None;
        ui.label(RichText::new("IMMAGINI").size(theme::size::SMALL).strong().color(TEXT_WEAK));
        crate::panels::hint(ui, "Fotogrammi dello stesso soggetto con la stessa focale, che si sovrappongono un po'. Trascina un riquadro nell'anteprima per ritoccarne la posizione.");
        let mut changed = false;
        egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
            for (i, tile) in self.mosaic.tiles.iter_mut().enumerate() {
                let loaded = self.images.get(i).is_some_and(Option::is_some);
                ui.horizontal(|ui| {
                    if ui.add_enabled(loaded, egui::Checkbox::without_text(&mut tile.enabled)).changed() {
                        changed = true;
                    }
                    let name = tile.path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
                    let selected = self.selected == Some(i);
                    let text = RichText::new(format!("{} · {name}", i + 1)).size(theme::size::SMALL).color(if selected { ACCENT } else { TEXT });
                    if ui.add(egui::Label::new(text).truncate().sense(Sense::click())).on_hover_text(tile.path.display().to_string()).clicked() {
                        self.selected = Some(i);
                    }
                });
                let (status, colour) = match (loaded, tile.enabled, tile.placement) {
                    (false, ..) if self.busy.is_some() => ("in caricamento", TEXT_WEAK),
                    (false, ..) => ("non si apre", theme::DANGER),
                    (_, false, _) => ("esclusa", TEXT_WEAK),
                    (_, true, None) if self.busy.is_some() => ("in allineamento", TEXT_WEAK),
                    (_, true, None) => ("non si sovrappone alle altre", theme::WARNING),
                    (_, true, Some(p)) if p == Placement::IDENTITY => ("riferimento", theme::SUCCESS),
                    (_, true, Some(_)) => ("allineata", theme::SUCCESS),
                };
                ui.label(RichText::new(format!("      {status}")).size(theme::size::SMALL).color(colour));
            }
        });
        if changed {
            self.arrange();
        }
        for e in &self.errors {
            ui.label(RichText::new(e).size(theme::size::SMALL).color(theme::DANGER));
        }
        ui.add_space(theme::space::S);
        ui.horizontal(|ui| {
            if ui.button((ICON_ADD.rich_text(), "Aggiungi…")).clicked()
                && let Some(more) = rfd::FileDialog::new().set_title("Aggiungi immagini al mosaico").add_filter("Immagini", ps_io::READ_EXTENSIONS).pick_files()
            {
                let mut paths: Vec<PathBuf> = self.mosaic.tiles.iter().map(|t| t.path.clone()).collect();
                let new: Vec<PathBuf> = more.into_iter().filter(|p| !paths.contains(p)).collect();
                paths.extend(new);
                let editing_open = self.editing_open;
                let params = self.mosaic.params.clone();
                self.start(ui.ctx(), paths);
                (self.editing_open, self.mosaic.params) = (editing_open, params);
            }
            if ui.add_enabled(self.busy.is_none() && !self.images.is_empty(), egui::Button::new((ICON_AUTO_FIX_HIGH.rich_text(), "Riallinea"))).on_hover_text("Ritrova automaticamente le posizioni (i ritocchi a mano si perdono)").clicked() {
                self.realign(ui.ctx());
            }
        });

        ui.add_space(theme::space::M);
        ui.label(RichText::new("UNIONE").size(theme::size::SMALL).strong().color(TEXT_WEAK));
        let p = &mut self.mosaic.params;
        let before = p.clone();
        ParamSlider::new("Sfumatura bordi", &mut p.feather, 0.0..=400.0, 80.0).decimals(0).suffix(" px").help("Su quanti pixel un'immagine sfuma nell'altra: più ampia nasconde meglio le giunture").show(ui);
        ui.horizontal(|ui| {
            toggle(ui, &mut p.match_brightness);
            ui.label("Uniforma la luminosità").on_hover_text("Trasparenza ed esposizione cambiano da un'immagine all'altra: le porta allo stesso livello dove si sovrappongono");
        });
        ui.horizontal(|ui| {
            toggle(ui, &mut p.prefer_sharp);
            ui.label("Preferisci la più nitida").on_hover_text("Dove le immagini si sovrappongono conta di più quella con più dettaglio");
        });
        if *p != before {
            self.dirty = true;
        }

        ui.add_space(theme::space::L);
        if let Some(text) = &self.busy {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(text).size(theme::size::SMALL).color(TEXT_WEAK));
            });
        }
        let ready = self.busy.is_none() && self.mosaic.tiles.iter().zip(&self.images).filter(|(t, i)| t.enabled && i.is_some() && t.placement.is_some()).count() >= 1;
        let black = Color32::BLACK;
        let label = if self.editing_open { "Aggiorna il mosaico" } else { "Crea il mosaico" };
        let create = egui::Button::new((ICON_CHECK.rich_text().color(black), RichText::new(label).color(black).strong())).fill(ACCENT).min_size(Vec2::new(ui.available_width(), 32.0));
        if ui.add_enabled(ready, create).on_hover_text("Unisce le immagini a piena risoluzione e le apre per l'elaborazione; il mosaico resta modificabile (File › Modifica mosaico)").clicked() {
            outcome = Some(MosaicOutcome::Create { mosaic: self.mosaic.clone(), images: self.images.clone(), keep_settings: self.editing_open });
            self.open = false;
        }
        ui.add_space(theme::space::S);
        if ui.add_enabled(ready, egui::Button::new((ICON_SAVE.rich_text(), "Salva mosaico…")).min_size(Vec2::new(ui.available_width(), 26.0))).on_hover_text("Salva l'elenco delle immagini e le loro posizioni, per riaprire il mosaico in seguito").clicked() {
            save(&self.mosaic);
        }
        outcome
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 8.0, theme::BG);
        let Some((texture, scale, [ox, oy])) = &self.preview else {
            let text = if self.busy.is_some() { "Preparazione dell'anteprima…" } else { "Nessuna immagine allineata" };
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(14.0), TEXT_WEAK);
            return;
        };
        let size = texture.size_vec2();
        let fit = ((rect.width() - 24.0) / size.x).min((rect.height() - 24.0) / size.y);
        let shown = Rect::from_center_size(rect.center(), size * fit);
        painter.image(texture.id(), shown, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        // Mosaic coordinates ↔ screen.
        let px = scale * fit;
        let (ox, oy) = (*ox, *oy);
        let to_screen = |[x, y]: [f32; 2]| Pos2::new(shown.left() + (x - ox) * px, shown.top() + (y - oy) * px);
        let sizes: Vec<Option<(f32, f32)>> = self.images.iter().map(|i| i.as_ref().map(|i| (i.width() as f32, i.height() as f32))).collect();
        let outline = |i: usize, p: &Placement| -> Option<[Pos2; 4]> {
            let (w, h) = (*sizes.get(i)?)?;
            Some([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].map(|c| to_screen(p.apply(c))))
        };
        // Click to select (the last drawn on top), drag to move.
        let inside = |q: Pos2, c: &[Pos2; 4]| (0..4).all(|k| {
            let (a, b) = (c[k], c[(k + 1) % 4]);
            (b - a).x * (q - a).y - (b - a).y * (q - a).x >= 0.0
        });
        if (response.clicked() || response.drag_started())
            && let Some(pos) = response.interact_pointer_pos()
        {
            let hit = self.mosaic.tiles.iter().enumerate().rev().find(|(i, t)| t.enabled && t.placement.and_then(|p| outline(*i, &p)).is_some_and(|c| inside(pos, &c)));
            self.selected = hit.map(|(i, _)| i);
            if response.drag_started()
                && let Some((i, t)) = hit
            {
                self.drag = t.placement.map(|p| (i, p));
            }
        }
        if let Some((i, start)) = self.drag {
            let delta = response.drag_delta() / px;
            if let Some(p) = &mut self.mosaic.tiles[i].placement {
                p.x += delta.x;
                p.y += delta.y;
            }
            if response.drag_stopped() {
                self.drag = None;
                if self.mosaic.tiles[i].placement != Some(start) {
                    self.update_gains();
                }
            }
        }
        for (i, tile) in self.mosaic.tiles.iter().enumerate() {
            let (Some(p), true) = (tile.placement, tile.enabled) else { continue };
            let Some(c) = outline(i, &p) else { continue };
            let selected = self.selected == Some(i);
            let stroke = if selected { Stroke::new(2.0, ACCENT) } else { Stroke::new(1.0, Color32::from_white_alpha(90)) };
            painter.add(egui::Shape::closed_line(c.to_vec(), stroke));
            let label_at = c[0] + Vec2::new(8.0, 6.0);
            painter.circle_filled(label_at + Vec2::new(5.0, 7.0), 10.0, Color32::from_black_alpha(170));
            painter.text(label_at + Vec2::new(5.0, 7.0), egui::Align2::CENTER_CENTER, format!("{}", i + 1), egui::FontId::proportional(12.0), if selected { ACCENT } else { TEXT });
        }
        if self.drag.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        } else if response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
    }
}

/// Ask where and save the mosaic's description (`.psmosaic`, JSON).
fn save(mosaic: &Mosaic) {
    let stem = mosaic.tiles.first().and_then(|t| t.path.file_stem()).and_then(|s| s.to_str()).unwrap_or("mosaico");
    let dir = mosaic.tiles.first().and_then(|t| t.path.parent().map(PathBuf::from));
    let mut dialog = rfd::FileDialog::new().set_title("Salva mosaico").set_file_name(format!("{stem}_mosaico.psmosaic")).add_filter("Mosaico Planetary Studio", &["psmosaic"]);
    if let Some(dir) = dir {
        dialog = dialog.set_directory(dir);
    }
    if let Some(path) = dialog.save_file()
        && let Ok(json) = serde_json::to_string_pretty(mosaic)
    {
        std::fs::write(path, json).ok();
    }
}

/// A saved mosaic (`.psmosaic`).
pub fn read(path: &std::path::Path) -> Result<Mosaic, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| format!("file di mosaico non valido: {e}"))
}
