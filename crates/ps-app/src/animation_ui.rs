//! The animation window: several captures of one subject (typically five or six shots of
//! Jupiter from one evening) become a smooth GIF or MP4. The captures are aligned on the disk,
//! brought to the same brightness, ordered by time and interpolated up to 50–60 frames per
//! second ([`ps_core::animation`]); the preview plays the result and the export writes it
//! ([`ps_io::video`]). The window saves the file itself: nothing opens in the main view.

use crate::instrument::CaptureState;
use crate::theme::{self, TEXT, TEXT_WEAK};
use crate::widgets::{ParamSlider, caps, primary_button, toggle};
use eframe::egui::{self, ColorImage, RichText, TextureHandle, Vec2};
use egui_material_icons::icons::*;
use ps_core::animation::{self, Interpolation, Options, Plan, Shot};
use ps_core::capture::Body;
use ps_core::derotation::JupiterSystem;
use ps_core::tr;
use ps_io::video::{self, FrameSource, RgbFrame};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

/// Side of the preview, pixels.
const PREVIEW_SIDE: usize = 320;
/// Frame rate of the preview (the machine's drawing, not the file's).
const PREVIEW_FPS: f32 = 25.0;

/// Frames of a plan as an exporter source.
struct PlanSource {
    plan: Plan,
    label: bool,
}

impl FrameSource for PlanSource {
    fn len(&self) -> usize {
        self.plan.frames()
    }
    fn size(&self) -> (usize, usize) {
        self.plan.size()
    }
    fn frame(&self, k: usize) -> RgbFrame {
        let img = self.plan.render(k);
        let text = if self.label { self.plan.moment(k).map(|m| animation::format_moment(&m)) } else { None };
        RgbFrame::new(img.width(), img.height(), animation::to_rgb8(&img, text.as_deref()))
    }
}

/// One capture.
struct Item {
    path: PathBuf,
    name: String,
    shot: Option<Shot>,
    time_from: &'static str,
    enabled: bool,
    error: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Gif,
    Mp4,
}

enum Msg {
    Progress(String),
    Loaded { items: Vec<Item>, body: Option<Body> },
    Preview { generation: u64, result: Result<PreviewFrames, String> },
    Exporting { done: usize, total: usize },
    Exported(Result<String, String>),
}

struct PreviewFrames {
    frames: Vec<ColorImage>,
    mode: Interpolation,
    mirrored: Option<bool>,
    span_minutes: Option<f64>,
}

pub struct AnimationEditor {
    pub open: bool,
    items: Vec<Item>,
    body: Option<Body>,
    interpolation: Interpolation,
    system: JupiterSystem,
    auto_mirror: bool,
    mirrored: bool,
    seconds: f32,
    fps: f32,
    size: usize,
    ping_pong: bool,
    crop: bool,
    label: bool,
    brightness: bool,
    busy: Option<String>,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    // The preview: what it was drawn with, its frames, the texture shown.
    generation: u64,
    preview_running: bool,
    dirty: bool,
    preview_key: String,
    preview: Option<PreviewFrames>,
    texture: Option<TextureHandle>,
    shown: Option<usize>,
    started: f64,
    warning: Option<String>,
    // The export.
    ffmpeg: Option<std::path::PathBuf>,
    exporting: Option<(usize, usize)>,
    cancel: Arc<AtomicBool>,
    message: Option<(bool, String)>,
}

impl Default for AnimationEditor {
    fn default() -> Self {
        let (tx, rx) = channel();
        Self {
            open: false,
            items: Vec::new(),
            body: None,
            interpolation: Interpolation::Auto,
            system: JupiterSystem::II,
            auto_mirror: true,
            mirrored: false,
            seconds: 6.0,
            fps: 50.0,
            size: 600,
            ping_pong: true,
            crop: true,
            label: true,
            brightness: true,
            busy: None,
            tx,
            rx,
            generation: 0,
            preview_running: false,
            dirty: false,
            preview_key: String::new(),
            preview: None,
            texture: None,
            shown: None,
            started: 0.0,
            warning: None,
            ffmpeg: None,
            exporting: None,
            cancel: Arc::new(AtomicBool::new(false)),
            message: None,
        }
    }
}

impl AnimationEditor {

    /// Open the window on these captures.
    pub fn start(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>, body: Option<Body>) {
        *self = Self { open: true, body, ffmpeg: video::find_ffmpeg(), ..Default::default() };
        self.busy = Some(tr!("Caricamento di {} immagini…", "Loading {} images…", paths.len()));
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let total = paths.len();
            let mut detected = body;
            let mut items = Vec::with_capacity(total);
            for (k, path) in paths.into_iter().enumerate() {
                tx.send(Msg::Progress(tr!("Analisi dell'immagine {} di {total}…", "Analysing image {} of {total}…", k + 1))).ok();
                ctx.request_repaint();
                let capture = CaptureState::read(&path);
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
                let mut item = Item { path, name, shot: None, time_from: capture.time_from, enabled: false, error: None };
                match ps_io::load(&item.path) {
                    Ok(image) => {
                        let analysis = ps_core::analysis::analyze(&image);
                        if detected.is_none() {
                            detected = capture.body.or(match analysis.kind {
                                ps_core::analysis::ImageKind::Saturn => Some(Body::Saturn),
                                _ => analysis.planet,
                            });
                        }
                        item.shot = Some(Shot::from_analysis(Arc::new(image), capture.info.time, &analysis));
                        item.enabled = true;
                    }
                    Err(e) => item.error = Some(format!("{}: {e}", item.name)),
                }
                items.push(item);
            }
            items.sort_by(|a, b| match (a.shot.as_ref().and_then(|s| s.time), b.shot.as_ref().and_then(|s| s.time)) {
                (Some(x), Some(y)) => x.julian().total_cmp(&y.julian()),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.path.cmp(&b.path),
            });
            tx.send(Msg::Loaded { items, body: detected }).ok();
            ctx.request_repaint();
        });
    }

    fn shots(&self) -> Vec<Shot> {
        self.items.iter().filter(|i| i.enabled).filter_map(|i| i.shot.clone()).collect()
    }

    /// The options as set, for a file at `fps` and `size` pixels.
    fn options(&self, fps: f32, size: usize) -> Options {
        Options {
            interpolation: self.interpolation,
            body: self.body,
            system: self.system,
            mirrored: if self.auto_mirror { None } else { Some(self.mirrored) },
            seconds: self.seconds,
            fps,
            ping_pong: self.ping_pong,
            crop: self.crop,
            size,
            match_brightness: self.brightness,
        }
    }

    /// What the preview depends on.
    fn key(&self) -> String {
        let enabled: Vec<bool> = self.items.iter().map(|i| i.enabled).collect();
        format!("{:?}|{:?}|{}|{}|{}|{}|{}|{}|{:?}|{:?}", self.options(PREVIEW_FPS, PREVIEW_SIDE), self.label, self.items.len(), self.auto_mirror, self.mirrored, self.seconds, self.ping_pong, self.crop, enabled, self.system)
    }

    fn receive(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Progress(text) => self.busy = Some(text),
                Msg::Loaded { items, body } => {
                    self.items = items;
                    self.body = body;
                    self.busy = None;
                    self.dirty = true;
                }
                Msg::Preview { generation, result } => {
                    self.preview_running = false;
                    if generation != self.generation {
                        continue;
                    }
                    match result {
                        Ok(p) => {
                            self.warning = None;
                            self.preview = Some(p);
                            self.shown = None;
                            self.started = ctx.input(|i| i.time);
                        }
                        Err(why) => {
                            self.warning = Some(why);
                            self.preview = None;
                        }
                    }
                }
                Msg::Exporting { done, total } => self.exporting = Some((done, total)),
                Msg::Exported(result) => {
                    self.exporting = None;
                    self.message = Some(match result {
                        Ok(text) => (true, text),
                        Err(why) => (false, why),
                    });
                }
            }
        }
    }

    /// Draw the preview again when something it depends on changed (one at a time).
    fn refresh_preview(&mut self, ctx: &egui::Context) {
        let key = self.key();
        if key != self.preview_key {
            self.preview_key = key;
            self.dirty = true;
            self.generation += 1;
        }
        if !self.dirty || self.preview_running || self.busy.is_some() {
            return;
        }
        self.dirty = false;
        let shots = self.shots();
        let generation = self.generation;
        if shots.len() < 2 {
            self.preview = None;
            self.warning = Some(tr!("Servono almeno due immagini.", "At least two images are needed."));
            return;
        }
        self.preview_running = true;
        let (opts, label, tx, ctx) = (self.options(PREVIEW_FPS, PREVIEW_SIDE), self.label, self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let result = Plan::new(&shots, &opts).map_err(plan_error).map(|plan| {
                let frames: Vec<ColorImage> = (0..plan.frames())
                    .into_par_iter()
                    .map(|k| {
                        let img = plan.render(k);
                        let text = if label { plan.moment(k).map(|m| animation::format_moment(&m)) } else { None };
                        ColorImage::from_rgb([img.width(), img.height()], &animation::to_rgb8(&img, text.as_deref()))
                    })
                    .collect();
                PreviewFrames { frames, mode: plan.mode(), mirrored: plan.mirrored(), span_minutes: plan.span_minutes() }
            });
            tx.send(Msg::Preview { generation, result }).ok();
            ctx.request_repaint();
        });
    }

    /// Ask where to save, then write the file off the UI thread.
    fn export(&mut self, ctx: &egui::Context, format: Format) {
        let first = self.items.iter().find(|i| i.enabled).map(|i| i.path.clone());
        let stem = first.as_ref().and_then(|p| p.file_stem()).and_then(|s| s.to_str()).unwrap_or("animazione");
        let (ext, name) = match format {
            Format::Gif => ("gif", tr!("GIF animata", "Animated GIF")),
            Format::Mp4 => ("mp4", tr!("Video MP4", "MP4 video")),
        };
        let mut dialog = crate::dialog::FileDialog::new().set_title(tr!("Salva l'animazione", "Save the animation")).add_filter(name, &[ext]).set_file_name(format!("{stem}_{}.{ext}", tr!("animazione", "animation")));
        if let Some(dir) = first.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.save_file() else { return };
        let shots = self.shots();
        let fps = match format {
            Format::Gif => self.fps.min(video::GIF_MAX_FPS),
            Format::Mp4 => self.fps,
        };
        let opts = self.options(fps, self.size);
        let (label, tx, ctx, cancel) = (self.label, self.tx.clone(), ctx.clone(), Arc::clone(&self.cancel));
        let ffmpeg = self.ffmpeg.clone();
        self.cancel.store(false, Ordering::Relaxed);
        self.exporting = Some((0, 0));
        self.message = None;
        std::thread::spawn(move || {
            let result = (|| -> Result<String, String> {
                let plan = Plan::new(&shots, &opts).map_err(plan_error)?;
                let (w, h, n) = (plan.size().0, plan.size().1, plan.frames());
                let source = PlanSource { plan, label };
                let mut progress = |done: usize, total: usize| {
                    tx.send(Msg::Exporting { done, total }).ok();
                    ctx.request_repaint();
                    !cancel.load(Ordering::Relaxed)
                };
                match format {
                    Format::Gif => {
                        let mut file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
                        let stats = video::write_gif(&source, &video::GifOptions { fps, ..Default::default() }, &mut file, &mut progress).map_err(|e| e.to_string());
                        if stats.is_err() {
                            std::fs::remove_file(&path).ok();
                        }
                        let stats = stats?;
                        Ok(tr!("GIF salvata: {n} fotogrammi {w}×{h} a {fps:.0} fps, {:.1} MB", "GIF saved: {n} frames {w}×{h} at {fps:.0} fps, {:.1} MB", stats.bytes as f32 / 1e6))
                    }
                    Format::Mp4 => {
                        let ffmpeg = ffmpeg.ok_or_else(no_ffmpeg)?;
                        video::write_mp4(&source, &video::Mp4Options { fps, ..Default::default() }, &ffmpeg, &path, &mut progress).map_err(|e| e.to_string())?;
                        Ok(tr!("Video salvato: {n} fotogrammi {w}×{h} a {fps:.0} fps", "Video saved: {n} frames {w}×{h} at {fps:.0} fps"))
                    }
                }
            })();
            tx.send(Msg::Exported(result)).ok();
            ctx.request_repaint();
        });
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        self.receive(ctx);
        self.refresh_preview(ctx);
        let mut open = self.open;
        crate::widgets::window(ctx, &tr!("Animazione", "Animation")).open(&mut open).default_size([780.0, 560.0]).min_size([640.0, 420.0]).show(ctx, |ui| {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(300.0);
                    egui::ScrollArea::vertical().id_salt("animation-side").show(ui, |ui| self.side(ui, ctx));
                });
                ui.vertical(|ui| self.canvas(ui, ctx));
            });
        });
        self.open &= open;
        if !self.open {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn side(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label(caps(&tr!("RIPRESE", "SHOTS")));
        if self.items.is_empty() {
            if let Some(text) = &self.busy {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new(text).size(theme::size::SMALL).color(TEXT_WEAK));
                });
            }
        }
        egui::ScrollArea::vertical().id_salt("animation-items").max_height(130.0).show(ui, |ui| {
            for item in &mut self.items {
                ui.horizontal(|ui| {
                    if item.shot.is_some() {
                        ui.checkbox(&mut item.enabled, "");
                    }
                    let detail = match (&item.error, item.shot.as_ref()) {
                        (Some(e), _) => e.clone(),
                        (None, Some(s)) => match s.time {
                            Some(t) => format!("{:02}:{:02}:{:02} UT", t.hour, t.minute, t.second.floor() as u32),
                            None => tr!("senza ora", "no time"),
                        },
                        _ => String::new(),
                    };
                    let hover = if item.time_from.is_empty() { item.path.display().to_string() } else { format!("{} · {}", item.path.display(), item.time_from) };
                    ui.add(egui::Label::new(RichText::new(format!("{}   {detail}", item.name)).size(theme::size::SMALL).color(if item.enabled { TEXT } else { TEXT_WEAK })).truncate()).on_hover_text(hover);
                });
            }
        });

        ui.add_space(theme::space::M);
        ui.label(caps(&tr!("INTERPOLAZIONE", "INTERPOLATION")));
        let (auto, globe, flow, fade) = (tr!("Auto", "Auto"), tr!("Globo", "Globe"), tr!("Flusso", "Flow"), tr!("Dissolvenza", "Fade"));
        crate::widgets::segmented(ui, &mut self.interpolation, &[(Interpolation::Auto, None, auto.as_str()), (Interpolation::Globe, None, globe.as_str()), (Interpolation::Flow, None, flow.as_str()), (Interpolation::Fade, None, fade.as_str())]);
        crate::panels::hint(
            ui,
            &match self.interpolation {
                Interpolation::Auto => tr!("Giove e Saturno con l'ora di ogni ripresa: la rotazione del pianeta. Altrimenti il flusso ottico.", "Jupiter and Saturn with the time of each shot: the planet's rotation. Otherwise optical flow."),
                Interpolation::Globe => tr!("Fra due riprese il pianeta gira davvero, di quanto dice il periodo di rotazione: serve l'ora di ogni ripresa.", "Between two shots the planet really turns, by what the rotation period says: the time of each shot is needed."),
                Interpolation::Flow => tr!("Il movimento dei dettagli viene stimato fra le immagini e seguito: per la Luna, il Sole, o riprese senza ora.", "The motion of the details is estimated between images and followed: for the Moon, the Sun, or shots without a time."),
                Interpolation::Fade => tr!("Solo una dissolvenza fra le immagini.", "Just a cross-fade between the images."),
            },
        );
        if matches!(self.interpolation, Interpolation::Auto | Interpolation::Globe) && self.body == Some(Body::Jupiter) {
            let (s2, s1) = (tr!("Sistema II", "System II"), tr!("Sistema I", "System I"));
            crate::widgets::segmented(ui, &mut self.system, &[(JupiterSystem::II, None, s2.as_str()), (JupiterSystem::I, None, s1.as_str())]);
        }
        if matches!(self.interpolation, Interpolation::Auto | Interpolation::Globe) {
            ui.horizontal(|ui| {
                toggle(ui, &mut self.auto_mirror);
                ui.label(tr!("Verso di rotazione automatico", "Automatic rotation direction")).on_hover_text(tr!("Prova i due versi e tiene quello che fa coincidere meglio le riprese", "Tries both directions and keeps the one that makes the shots agree best"));
            });
            ui.add_enabled_ui(!self.auto_mirror, |ui| {
                ui.horizontal(|ui| {
                    toggle(ui, &mut self.mirrored);
                    ui.label(tr!("Immagine specchiata", "Mirrored image"));
                });
            });
        }

        ui.add_space(theme::space::M);
        ui.label(caps(&tr!("USCITA", "OUTPUT")));
        ParamSlider::new(&tr!("Durata", "Duration"), &mut self.seconds, 1.0..=30.0, 6.0).decimals(1).suffix(" s").help(&tr!("Durata di un ciclo completo (andata e ritorno se è attivo)", "Length of one whole cycle (there and back when that is on)")).show(ui);
        ParamSlider::new(&tr!("Fotogrammi/s", "Frames/s"), &mut self.fps, 10.0..=60.0, 50.0).decimals(0).help(&tr!("La GIF arriva a 50 al secondo, il formato non permette di più; l'MP4 usa il valore scelto", "A GIF goes up to 50 a second, the format allows no more; the MP4 uses the chosen value")).show(ui);
        let mut size = self.size as f32;
        if ParamSlider::new(&tr!("Dimensione", "Size"), &mut size, 200.0..=1200.0, 600.0).decimals(0).suffix(" px").help(&tr!("Lato maggiore: non supera quello delle immagini originali", "Longest side: it never exceeds the original images'")).show(ui) {
            self.size = size.round() as usize;
        }
        for (flag, text, help) in [
            (&mut self.ping_pong, tr!("Avanti e indietro", "Ping-pong"), tr!("Dopo l'ultima ripresa torna indietro invece di ricominciare dalla prima", "After the last shot it goes back instead of starting again from the first")),
            (&mut self.crop, tr!("Ritaglia sul pianeta", "Crop on the planet"), tr!("Centra il pianeta e taglia il cielo attorno", "Centres the planet and trims the sky around it")),
            (&mut self.brightness, tr!("Uniforma la luminosità", "Even out the brightness"), tr!("Porta il disco di ogni ripresa allo stesso livello", "Brings each shot's disk to the same level")),
            (&mut self.label, tr!("Data e ora in sovrimpressione", "Date and time overlay"), tr!("Scrive l'ora (UT) della ripresa in basso a sinistra", "Writes the capture time (UT) at the bottom left")),
        ] {
            ui.horizontal(|ui| {
                toggle(ui, flag);
                ui.label(text).on_hover_text(help);
            });
        }

        ui.add_space(theme::space::L);
        let ready = self.shots().len() >= 2 && self.preview.is_some() && self.exporting.is_none();
        let gif = primary_button(Some(ICON_GIF), &tr!("Salva GIF…", "Save GIF…")).min_size(Vec2::new(ui.available_width(), 32.0));
        if ui.add_enabled(ready, gif).on_hover_text(tr!("Animazione ciclica fino a 50 fotogrammi al secondo", "A looping animation of up to 50 frames a second")).clicked() {
            self.export(ctx, Format::Gif);
        }
        ui.add_space(theme::space::XS);
        let mp4 = egui::Button::new((ICON_MOVIE.rich_text(), tr!("Salva MP4…", "Save MP4…"))).min_size(Vec2::new(ui.available_width(), 32.0));
        let response = ui.add_enabled(ready && self.ffmpeg.is_some(), mp4);
        if self.ffmpeg.is_none() {
            response.on_disabled_hover_text(no_ffmpeg());
            ui.label(RichText::new(no_ffmpeg()).size(theme::size::SMALL).color(theme::WARNING));
        } else if response.on_hover_text(tr!("Video H.264, fluido a 60 fotogrammi al secondo", "H.264 video, smooth at 60 frames a second")).clicked() {
            self.export(ctx, Format::Mp4);
        }
        if let Some((done, total)) = self.exporting {
            ui.add_space(theme::space::S);
            ui.add(egui::ProgressBar::new(if total == 0 { 0.0 } else { done as f32 / total as f32 }).text(tr!("Scrittura {done}/{total}", "Writing {done}/{total}")));
            if ui.button(tr!("Annulla", "Cancel")).clicked() {
                self.cancel.store(true, Ordering::Relaxed);
            }
        }
        if let Some((ok, text)) = &self.message {
            ui.add_space(theme::space::S);
            ui.add(egui::Label::new(RichText::new(text).size(theme::size::SMALL).color(if *ok { TEXT } else { theme::DANGER })).wrap());
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let side = (ui.available_width().min(ui.available_height())).clamp(240.0, 480.0);
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(side), egui::Sense::hover());
        ui.painter().rect_filled(rect, egui::CornerRadius::same(theme::radius::M), egui::Color32::BLACK);
        if let Some(p) = &self.preview {
            let n = p.frames.len().max(1);
            let t = ctx.input(|i| i.time) - self.started;
            let k = ((t * PREVIEW_FPS as f64) as usize) % n;
            if self.shown != Some(k) {
                match &mut self.texture {
                    Some(tex) => tex.set(p.frames[k].clone(), egui::TextureOptions::LINEAR),
                    None => self.texture = Some(ctx.load_texture("animation-preview", p.frames[k].clone(), egui::TextureOptions::LINEAR)),
                }
                self.shown = Some(k);
            }
            if let Some(tex) = &self.texture {
                let [w, h] = p.frames[k].size;
                let scale = (side / w as f32).min(side / h as f32);
                let r = egui::Rect::from_center_size(rect.center(), Vec2::new(w as f32 * scale, h as f32 * scale));
                ui.painter().image(tex.id(), r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        } else if self.busy.is_some() || self.preview_running || self.dirty {
            ui.put(egui::Rect::from_center_size(rect.center(), Vec2::splat(24.0)), egui::Spinner::new());
        }
        ui.add_space(theme::space::S);
        if let Some(why) = &self.warning {
            ui.add(egui::Label::new(RichText::new(why).size(theme::size::SMALL).color(theme::WARNING)).wrap());
        } else if let Some(p) = &self.preview {
            let mode = match p.mode {
                Interpolation::Globe => tr!("rotazione del globo", "globe rotation"),
                Interpolation::Flow => tr!("flusso ottico", "optical flow"),
                _ => tr!("dissolvenza", "cross-fade"),
            };
            let mut text = tr!("{} riprese · {mode}", "{} shots · {mode}", self.shots().len());
            if let Some(m) = p.span_minutes {
                text += &tr!(" · {m:.0} min", " · {m:.0} min");
            }
            if let Some(mirrored) = p.mirrored {
                text += &if mirrored { tr!(" · rotazione da destra a sinistra", " · turning right to left") } else { String::new() };
            }
            ui.label(RichText::new(text).size(theme::size::SMALL).color(TEXT_WEAK));
            let fps = if self.fps > video::GIF_MAX_FPS { format!("{} / {}", video::GIF_MAX_FPS, self.fps.round()) } else { format!("{}", self.fps.round()) };
            let frames = (self.seconds * self.fps).round() as usize;
            ui.label(RichText::new(tr!("≈ {frames} fotogrammi a {fps} fps (GIF / MP4)", "≈ {frames} frames at {fps} fps (GIF / MP4)")).size(theme::size::SMALL).color(TEXT_WEAK));
            ui.label(RichText::new(tr!("L'anteprima è a bassa risoluzione e a 25 fotogrammi al secondo.", "The preview is low resolution at 25 frames a second.")).size(theme::size::SMALL).color(TEXT_WEAK));
        }
    }
}

fn no_ffmpeg() -> String {
    tr!("Per il video MP4 serve ffmpeg, che non è installato (per esempio: brew install ffmpeg). La GIF funziona comunque.", "ffmpeg is needed for MP4 video and is not installed (for example: brew install ffmpeg). The GIF works anyway.")
}

fn plan_error(e: animation::Error) -> String {
    match e {
        animation::Error::TooFew => tr!("Servono almeno due immagini.", "At least two images are needed."),
        animation::Error::Sizes => tr!("Le immagini hanno dimensioni diverse: servono tutte della stessa.", "The images differ in size: they all need the same."),
        animation::Error::NotGlobe => tr!("La rotazione del globo vale per Giove e Saturno, con l'ora di ogni ripresa: scegli un'altra interpolazione.", "Globe rotation applies to Jupiter and Saturn, with the time of each shot: choose another interpolation."),
    }
}
