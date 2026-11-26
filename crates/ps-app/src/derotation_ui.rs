//! The derotation window: several captures of Jupiter, Saturn or Mars from one evening, brought
//! to the same instant on the planet's globe and summed (the mosaic stays for the Moon and the
//! Sun). Everything is automatic ([`ps_core::derotation::auto`]): the globe of each capture,
//! its turn (from the images and the times), the reference instant. The window shows the
//! globe found on each capture so it can be checked and corrected (drag to move, the handle on
//! the equator to resize, the one over the pole to turn), the time of each capture (editable),
//! the result and the planet's map; then the sum is opened as a normal image.

use crate::instrument::CaptureState;
use crate::theme::{self, ACCENT, TEXT, TEXT_WEAK};
use crate::widgets::{ParamSlider, caps, primary_button, toggle};
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, RichText, Sense, Stroke, TextureHandle, Vec2};
use egui_material_icons::icons::*;
use ps_core::Image;
use ps_core::capture::{Body, Moment};
use ps_core::derotation::Globe;
use ps_core::derotation::auto::{self, Capture, Prepared, Reference, Settings, Solution, TurnSource};
use ps_core::globe::{Projection, point};
use ps_core::tr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

/// What the window asks the app to do.
pub enum DerotationOutcome {
    /// Sum at full size and open the result.
    Create(Box<StackJob>),
}

/// What the opened sum came from (for its title, its name and its capture time).
#[derive(Clone, Debug)]
pub struct Derotated {
    pub frames: usize,
    /// The instant the planet was brought to, when the times were known.
    pub moment: Option<Moment>,
    /// Built from monochrome channels (`RGB`, `LRGB`) instead of summed.
    pub mono: Option<&'static str>,
    /// Whether the planet's rotation was compensated (for a colour image from channels).
    pub rotated: bool,
}

/// Everything the sum needs, away from the window: the captures already prepared and solved,
/// so creating the image only draws it at full size.
#[derive(Clone)]
pub struct StackJob {
    pub images: Vec<Arc<Image>>,
    /// Hours from the reference instant to each capture (from the turns found).
    #[allow(dead_code)]
    pub hours: Vec<f64>,
    times: Vec<Option<Moment>>,
    prepared: Arc<Vec<Prepared>>,
    solution: Solution,
    clip: Option<f32>,
    pub path: PathBuf,
    pub derotated: Derotated,
}

impl StackJob {
    fn captures(&self) -> Vec<Capture<'_>> {
        self.images.iter().zip(&self.times).zip(&self.solution.frames).map(|((image, time), f)| Capture { image, time: *time, globe: Some(f.globe) }).collect()
    }

    /// The sum at `scale`, and the globe used for each capture.
    pub fn run(&self, scale: f32) -> Option<(Image, Vec<Globe>)> {
        let image = auto::render(&self.captures(), &self.prepared, &self.solution, scale, self.clip)?;
        Some((image, self.solution.frames.iter().map(|f| f.globe).collect()))
    }
}

/// One capture.
struct Item {
    path: PathBuf,
    image: Option<Arc<Image>>,
    /// A small copy for the screen.
    thumb: Option<TextureHandle>,
    time: Option<Moment>,
    time_from: &'static str,
    /// The time being typed.
    time_text: String,
    time_bad: bool,
    /// The globe placed by hand (else measured).
    globe: Option<Globe>,
    enabled: bool,
    error: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    Capture,
    Result,
    Map,
}

/// Which part of the overlay is being dragged.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Grip {
    Move,
    Resize,
    Turn,
}

enum Job {
    Progress(String),
    Loaded { items: Vec<LoadedItem>, body: Option<Body> },
    Solved { generation: u64, key: String, prepared: Arc<Vec<Prepared>>, result: Result<(Solution, Image, Option<Image>), String> },
}

struct LoadedItem {
    path: PathBuf,
    image: Option<Arc<Image>>,
    thumb: Option<ColorImage>,
    time: Option<Moment>,
    time_from: &'static str,
    error: Option<String>,
}

pub struct DerotationEditor {
    pub open: bool,
    body: Body,
    items: Vec<Item>,
    reference: Option<usize>,
    weigh_sharpness: bool,
    use_times: bool,
    /// `None` = found automatically.
    mirrored: Option<bool>,
    clip: bool,
    kappa: f32,
    selected: usize,
    view: View,
    busy: Option<String>,
    tx: Sender<Job>,
    rx: Receiver<Job>,
    dirty: bool,
    generation: u64,
    running: bool,
    /// The captures prepared for a key (which captures, their globes, the planet).
    prepared: Option<(String, Arc<Vec<Prepared>>)>,
    /// The last solution, for the enabled captures (indices into `items`).
    solution: Option<(Vec<usize>, Solution)>,
    error: Option<String>,
    preview: Option<TextureHandle>,
    map: Option<TextureHandle>,
    drag: Option<(Grip, Globe)>,
}

impl Default for DerotationEditor {
    fn default() -> Self {
        let (tx, rx) = channel();
        Self {
            open: false,
            body: Body::Jupiter,
            items: Vec::new(),
            reference: None,
            weigh_sharpness: true,
            use_times: true,
            mirrored: None,
            clip: false,
            kappa: 2.5,
            selected: 0,
            view: View::Result,
            busy: None,
            tx,
            rx,
            dirty: false,
            generation: 0,
            running: false,
            prepared: None,
            solution: None,
            error: None,
            preview: None,
            map: None,
            drag: None,
        }
    }
}

/// The body of several captures as their names (or logs, or folder) say, when one is named.
pub fn named_body(paths: &[PathBuf]) -> Option<Body> {
    paths.iter().find_map(|p| CaptureState::read(p).body)
}

/// Whether a body is derotated (else several images make a mosaic).
pub fn derotates(body: Body) -> bool {
    matches!(body, Body::Jupiter | Body::Saturn | Body::Mars)
}

/// The body the pixels of an image suggest, for the derotation: Saturn from its rings,
/// Jupiter from its flattening.
pub fn body_from_pixels(img: &Image) -> Option<Body> {
    let analysis = ps_core::analysis::analyze(img);
    match analysis.kind {
        ps_core::analysis::ImageKind::Saturn => Some(Body::Saturn),
        _ => analysis.planet,
    }
}

fn format_time(t: &Moment) -> String {
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:04.1}", t.year, t.month, t.day, t.hour, t.minute, t.second)
}

/// `AAAA-MM-GG hh:mm[:ss[.s]]` (also with `T` or `/`).
fn parse_time(text: &str) -> Option<Moment> {
    let cleaned: String = text.trim().chars().map(|c| if c == 'T' || c == '/' || c == '_' { ' ' } else { c }).collect();
    let mut parts = cleaned.split_whitespace();
    let date: Vec<&str> = parts.next()?.split('-').collect();
    let time: Vec<&str> = parts.next()?.split(':').collect();
    if date.len() != 3 || !(2..=3).contains(&time.len()) {
        return None;
    }
    let m = Moment {
        year: date[0].parse().ok()?,
        month: date[1].parse().ok()?,
        day: date[2].parse().ok()?,
        hour: time[0].parse().ok()?,
        minute: time[1].parse().ok()?,
        second: time.get(2).map_or(Some(0.0), |s| s.parse().ok())?,
    };
    m.valid().then_some(m)
}

/// A copy of the image for the screen, at most `side` pixels, stretched to its brightest part.
fn thumbnail(img: &Image, side: usize) -> ColorImage {
    let f = img.width().max(img.height()).div_ceil(side).max(1);
    let small = if f > 1 { img.downscaled(f) } else { img.clone() };
    let mut values: Vec<f32> = small.channels().iter().flat_map(|c| c.iter().step_by(7).copied()).collect();
    let k = ((values.len() as f32 * 0.999) as usize).min(values.len().saturating_sub(1));
    let peak = if values.is_empty() { 1.0 } else { *values.select_nth_unstable_by(k, f32::total_cmp).1 };
    let gain = if peak > 1e-6 { 1.0 / peak } else { 1.0 };
    let scaled = Image::new(small.width(), small.height(), small.channels().iter().map(|c| c.iter().map(|v| (v * gain).min(1.0)).collect()).collect(), 16);
    ColorImage::from_rgba_unmultiplied([scaled.width(), scaled.height()], &scaled.to_rgba8())
}

impl DerotationEditor {



    /// The images of these captures, of `body` when known (else as the first one looks).
    pub fn start(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>, body: Option<Body>) {
        *self = Self { open: true, body: body.unwrap_or(Body::Jupiter), ..Default::default() };
        self.busy = Some(tr!("Caricamento di {} immagini…", "Loading {} images…", paths.len()));
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let total = paths.len();
            let mut detected = body;
            let mut items = Vec::with_capacity(total);
            for (k, path) in paths.into_iter().enumerate() {
                tx.send(Job::Progress(tr!("Lettura dell'immagine {} di {total}…", "Reading image {} of {total}…", k + 1))).ok();
                ctx.request_repaint();
                let capture = CaptureState::read(&path);
                let mut item = LoadedItem { path, image: None, thumb: None, time: capture.info.time, time_from: capture.time_from, error: None };
                match ps_io::load(&item.path) {
                    Ok(image) => {
                        if detected.is_none() {
                            detected = capture.body.or_else(|| body_from_pixels(&image));
                        }
                        item.thumb = Some(thumbnail(&image, 900));
                        item.image = Some(Arc::new(image));
                    }
                    Err(e) => item.error = Some(format!("{}: {e}", item.path.file_name().and_then(|n| n.to_str()).unwrap_or("?"))),
                }
                items.push(item);
            }
            // In the order they were taken (those without a time last).
            items.sort_by(|a, b| match (a.time, b.time) {
                (Some(x), Some(y)) => x.julian().total_cmp(&y.julian()),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.path.cmp(&b.path),
            });
            tx.send(Job::Loaded { items, body: detected }).ok();
            ctx.request_repaint();
        });
    }

    fn receive(&mut self, ctx: &egui::Context) {
        while let Ok(job) = self.rx.try_recv() {
            match job {
                Job::Progress(text) => self.busy = Some(text),
                Job::Loaded { items, body } => {
                    self.items = items
                        .into_iter()
                        .enumerate()
                        .map(|(k, it)| Item {
                            thumb: it.thumb.map(|t| ctx.load_texture(format!("derotation-capture-{k}"), t, egui::TextureOptions::LINEAR)),
                            time_text: it.time.as_ref().map(format_time).unwrap_or_default(),
                            time_bad: false,
                            enabled: it.image.is_some(),
                            globe: None,
                            path: it.path,
                            image: it.image,
                            time: it.time,
                            time_from: it.time_from,
                            error: it.error,
                        })
                        .collect();
                    if let Some(b) = body.filter(|b| derotates(*b)) {
                        self.body = b;
                    }
                    self.busy = None;
                    self.dirty = true;
                }
                Job::Solved { generation, key, prepared, result } => {
                    self.running = false;
                    self.prepared = Some((key, prepared));
                    if generation != self.generation {
                        continue;
                    }
                    match result {
                        Ok((solution, preview, map)) => {
                            let used = self.usable();
                            self.error = None;
                            let pixels = ColorImage::from_rgba_unmultiplied([preview.width(), preview.height()], &preview.to_rgba8());
                            self.preview = Some(ctx.load_texture("derotation-preview", pixels, egui::TextureOptions::LINEAR));
                            self.map = map.map(|m| ctx.load_texture("derotation-map", thumbnail(&m, 1024), egui::TextureOptions::LINEAR));
                            self.solution = Some((used, solution));
                        }
                        Err(e) => {
                            self.error = Some(e);
                            self.solution = None;
                            self.preview = None;
                        }
                    }
                }
            }
        }
    }

    /// The enabled captures that can be summed.
    fn usable(&self) -> Vec<usize> {
        (0..self.items.len()).filter(|&i| self.items[i].enabled && self.items[i].image.is_some()).collect()
    }

    fn settings(&self, used: &[usize]) -> Settings {
        let mut s = Settings::new(self.body);
        s.reference = match self.reference.and_then(|r| used.iter().position(|&i| i == r)) {
            Some(k) => Reference::Frame(k),
            None => Reference::Mean,
        };
        s.weigh_sharpness = self.weigh_sharpness;
        s.use_times = self.use_times;
        s.mirrored = self.mirrored;
        s.clip = self.clip.then_some(self.kappa);
        s
    }

    /// What the preparation depends on: the captures, their globes (and times, for the tilt), the planet.
    fn prepare_key(&self, used: &[usize]) -> String {
        let parts: Vec<String> = used.iter().map(|&i| format!("{}|{:?}|{:?}", self.items[i].path.display(), self.items[i].globe, self.items[i].time.map(|t| (t.year, t.month, t.day)))).collect();
        format!("{:?}|{}", self.body, parts.join(";"))
    }

    fn snapshot(&self) -> (Vec<usize>, Vec<Arc<Image>>, Vec<Option<Moment>>, Vec<Option<Globe>>) {
        let used = self.usable();
        let images = used.iter().map(|&i| Arc::clone(self.items[i].image.as_ref().expect("usable"))).collect();
        let times = used.iter().map(|&i| self.items[i].time).collect();
        let globes = used.iter().map(|&i| self.items[i].globe).collect();
        (used, images, times, globes)
    }

    /// The sum as it is set now (once the captures are solved).
    pub(crate) fn job(&self) -> Option<StackJob> {
        let (used, solution) = self.solution.as_ref()?;
        if *used != self.usable() {
            return None;
        }
        let (key, prepared) = self.prepared.as_ref()?;
        if *key != self.prepare_key(used) {
            return None;
        }
        let (_, images, times, _) = self.snapshot();
        let period = solution.params.period_hours;
        let hours = solution.frames.iter().map(|f| f.turn_deg / 360.0 * period * if solution.params.mirrored { -1.0 } else { 1.0 }).collect();
        let grid_item = used[solution.grid];
        Some(StackJob {
            images,
            hours,
            times,
            prepared: Arc::clone(prepared),
            solution: solution.clone(),
            clip: self.clip.then_some(self.kappa),
            path: self.items[grid_item].path.clone(),
            derotated: Derotated { frames: used.len(), moment: solution.moment, mono: None, rotated: true },
        })
    }

    /// Prepare (when the captures changed), solve and draw a preview, off the UI thread.
    fn refresh(&mut self, ctx: &egui::Context) {
        self.dirty = false;
        let (used, images, times, globes) = self.snapshot();
        if used.is_empty() {
            self.solution = None;
            self.preview = None;
            return;
        }
        let settings = self.settings(&used);
        let key = self.prepare_key(&used);
        let cached = self.prepared.as_ref().filter(|(k, _)| *k == key).map(|(_, p)| Arc::clone(p));
        self.generation += 1;
        self.running = true;
        let (tx, ctx, generation) = (self.tx.clone(), ctx.clone(), self.generation);
        let total = used.len();
        std::thread::spawn(move || {
            let debug = std::env::var_os("PS_DEROT_DEBUG").is_some();
            let t0 = std::time::Instant::now();
            let caps: Vec<Capture> = images.iter().zip(&times).zip(&globes).map(|((image, time), globe)| Capture { image, time: *time, globe: *globe }).collect();
            let prepared = match cached {
                Some(p) => Ok(p),
                None => {
                    tx.send(Job::Progress(tr!("Misura del globo su {total} riprese…", "Measuring the globe on {total} captures…"))).ok();
                    ctx.request_repaint();
                    auto::prepare_all(&caps, &settings, &|_| {}).map(Arc::new)
                }
            };
            let (prepared, result) = match prepared {
                Ok(p) => {
                    let result = auto::solve(&caps, &p, &settings).map_err(|e| e.message()).and_then(|solution| {
                        let grid = &images[solution.grid];
                        let scale = (720.0 / grid.width().max(grid.height()) as f32).min(1.0);
                        let preview = auto::render(&caps, &p, &solution, scale, settings.clip).ok_or_else(|| tr!("Nessuna ripresa da sommare", "No captures to sum"))?;
                        let map = auto::planet_map(&caps, &p, &solution, 1024).map(|m| m.to_image(true));
                        Ok((solution, preview, map))
                    });
                    (p, result)
                }
                Err(e) => (Arc::new(Vec::new()), Err(e.message())),
            };
            if debug {
                eprintln!("derotation refresh {generation}: {:.1} s", t0.elapsed().as_secs_f32());
            }
            tx.send(Job::Solved { generation, key, prepared, result }).ok();
            ctx.request_repaint();
        });
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<DerotationOutcome> {
        if !self.open {
            return None;
        }
        self.receive(ctx);
        if self.running {
            // The worker's progress replaces the "loading" text.
        } else if self.busy.as_deref().is_some_and(|_| !self.items.is_empty()) {
            self.busy = None;
        }
        if self.dirty && !self.running && self.busy.is_none() && self.drag.is_none() {
            self.refresh(ctx);
        }
        let mut outcome = None;
        let mut open = self.open;
        crate::widgets::window(ctx, &tr!("Derotazione", "Derotation"))
            .open(&mut open)
            .default_size([1080.0, 700.0])
            .min_size([760.0, 460.0])
            .show(ctx, |ui| {
                egui::Panel::right("derotation-side").resizable(false).exact_size(340.0).frame(egui::Frame::NONE.inner_margin(egui::Margin { left: 12, ..Default::default() })).show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| outcome = self.side(ui));
                });
                egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| self.canvas(ui));
            });
        self.open &= open;
        outcome
    }

    /// The solution for item `i`, if solved.
    fn fit_of(&self, i: usize) -> Option<&auto::FrameFit> {
        let (used, sol) = self.solution.as_ref()?;
        used.iter().position(|&k| k == i).map(|k| &sol.frames[k])
    }

    fn side(&mut self, ui: &mut egui::Ui) -> Option<DerotationOutcome> {
        let mut changed = false;
        ui.label(caps(&tr!("RIPRESE", "CAPTURES")));
        crate::panels::hint(ui, &tr!(
            "Immagini dello stesso pianeta e dello stesso filtro (o tutte a colori), riprese a pochi minuti l'una dall'altra. Il globo e la rotazione di ognuna si trovano da soli; l'ora (dal nome, dal log o dall'intestazione) serve a verificarli e si può correggere.",
            "Images of the same planet with the same filter (or all in colour), taken a few minutes apart. The globe and turn of each are found automatically; the time (from the name, the log or the header) checks them and can be corrected."
        ));
        let moment = self.solution.as_ref().and_then(|(_, s)| s.moment);
        egui::ScrollArea::vertical().id_salt("derotation-frames").max_height(300.0).show(ui, |ui| {
            for i in 0..self.items.len() {
                let fit = self.fit_of(i).copied();
                let item = &mut self.items[i];
                ui.horizontal(|ui| {
                    if ui.add_enabled_ui(item.image.is_some(), |ui| toggle(ui, &mut item.enabled)).inner.changed() {
                        changed = true;
                    }
                    let name = item.path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
                    let selected = self.selected == i;
                    let text = RichText::new(format!("{} · {name}", i + 1)).size(theme::size::SMALL).color(if selected { ACCENT } else { TEXT });
                    if ui.add(egui::Label::new(text).truncate().sense(Sense::click())).on_hover_text(item.path.display().to_string()).clicked() {
                        self.selected = i;
                        self.view = View::Capture;
                    }
                });
                if let Some(e) = &item.error {
                    ui.label(RichText::new(format!("      {e}")).size(theme::size::SMALL).color(theme::DANGER));
                    continue;
                }
                if self.selected == i {
                    // The time, editable.
                    ui.horizontal(|ui| {
                        ui.add_space(30.0);
                        let edit = egui::TextEdit::singleline(&mut item.time_text).desired_width(178.0).hint_text("AAAA-MM-GG hh:mm:ss").font(egui::TextStyle::Monospace);
                        let r = ui.add(edit).on_hover_text(tr!("Ora UT della ripresa (metà del video). Invio per applicare.", "UT time of the capture (middle of the video). Enter to apply."));
                        if r.lost_focus() {
                            let text = item.time_text.trim();
                            if text.is_empty() {
                                item.time_bad = false;
                                if item.time.is_some() {
                                    item.time = None;
                                    changed = true;
                                }
                            } else if let Some(t) = parse_time(text) {
                                item.time_bad = false;
                                if item.time != Some(t) {
                                    item.time = Some(t);
                                    item.time_from = "";
                                    changed = true;
                                }
                            } else {
                                item.time_bad = true;
                            }
                        }
                        if item.time_bad {
                            ui.label(RichText::new(ICON_ERROR.codepoint).color(theme::DANGER)).on_hover_text(tr!("Formato: 2026-09-12 21:30:15", "Format: 2026-09-12 21:30:15"));
                        }
                        if item.globe.is_some() && ui.small_button(tr!("Globo automatico", "Automatic globe")).clicked() {
                            item.globe = None;
                            changed = true;
                        }
                    });
                }
                let (status, colour) = if !item.enabled {
                    (tr!("esclusa", "excluded"), TEXT_WEAK)
                } else if fit.is_some_and(|f| f.weight <= 0.0) {
                    (tr!("esclusa (vedi le note sotto)", "left out (see the notes below)"), theme::WARNING)
                } else if let Some(f) = fit {
                    let when = match (f.time, moment) {
                        (Some(t), Some(m)) => {
                            let minutes = ps_core::derotation::hours_between(&t, &m) * 60.0;
                            format!("{:02}:{:02}:{:02} UT ({minutes:+.1} min)", t.hour, t.minute, t.second.floor() as u32)
                        }
                        _ => tr!("ora non nota", "time unknown"),
                    };
                    let source = match f.source {
                        TurnSource::Times => tr!("dall'ora", "from the time"),
                        TurnSource::Images => tr!("dalle immagini", "from the images"),
                        TurnSource::None => tr!("solo allineata", "only aligned"),
                    };
                    let weight = if self.weigh_sharpness { tr!(" · peso {:.0} %", " · weight {:.0} %", f.weight * 100.0) } else { String::new() };
                    let colour = match f.source {
                        TurnSource::None => theme::WARNING,
                        _ if f.time_shift_min.is_some() => theme::WARNING,
                        _ => theme::SUCCESS,
                    };
                    (format!("{when} · {:+.1}° {source}{weight}", f.turn_deg), colour)
                } else {
                    match item.time {
                        Some(t) => (format!("{:02}:{:02}:{:02} UT {}", t.hour, t.minute, t.second.floor() as u32, item.time_from), TEXT_WEAK),
                        None => (tr!("ora sconosciuta: la rotazione si misura sulle immagini", "time unknown: the turn is measured on the images"), theme::WARNING),
                    }
                };
                ui.label(RichText::new(format!("      {status}")).size(theme::size::SMALL).color(colour));
                if let Some(f) = fit
                    && let (Some(shift), Some(t)) = (f.time_shift_min, item.time)
                {
                    ui.horizontal(|ui| {
                        ui.add_space(30.0);
                        ui.label(RichText::new(tr!("l'ora sembra sbagliata di {shift:+.1} min", "the time seems off by {shift:+.1} min")).size(theme::size::SMALL).color(theme::WARNING));
                        if ui.small_button(tr!("Correggi", "Correct")).on_hover_text(tr!("Scrive l'ora stimata dalla rotazione misurata", "Writes the time estimated from the measured turn")).clicked() {
                            let fixed = Moment::from_julian(t.julian() + shift / 1440.0);
                            item.time = Some(fixed);
                            item.time_text = format_time(&fixed);
                            item.time_from = "";
                            changed = true;
                        }
                    });
                }
            }
        });

        ui.add_space(theme::space::M);
        ui.label(caps(&tr!("PIANETA", "PLANET")));
        ui.horizontal(|ui| {
            for body in [Body::Jupiter, Body::Saturn, Body::Mars] {
                if ui.selectable_label(self.body == body, body.label()).clicked() && self.body != body {
                    self.body = body;
                    changed = true;
                }
            }
        });
        if let Some((_, s)) = &self.solution {
            let p = &s.params;
            let span = self.span_minutes();
            let turned = span / 60.0 / p.period_hours * 360.0;
            ui.label(RichText::new(tr!(
                "Periodo {}h {:02}m{} · inclinazione {:+.1}° · le riprese coprono {span:.1} min ({turned:.1}° di rotazione)",
                "Period {}h {:02}m{} · tilt {:+.1}° · the captures span {span:.1} min ({turned:.1}° of rotation)",
                p.period_hours.floor(),
                (p.period_hours.fract() * 60.0).round(),
                if p.equator_period_hours.is_some() { tr!(" (Sistema II; equatore Sistema I)", " (System II; equator System I)") } else { String::new() },
                p.tilt_deg
            )).size(theme::size::SMALL).color(TEXT_WEAK));
            if turned > 40.0 {
                ui.label(RichText::new(tr!("Oltre 40° di rotazione il bordo che entra e quello che esce restano coperti da poche riprese.", "Beyond 40° of rotation the incoming and outgoing limbs are covered by few captures.")).size(theme::size::SMALL).color(theme::WARNING));
            }
            for note in &s.notes {
                ui.label(RichText::new(note).size(theme::size::SMALL).color(theme::WARNING));
            }
        }
        if let Some(e) = &self.error {
            ui.label(RichText::new(e).size(theme::size::SMALL).color(theme::DANGER));
        }

        ui.add_space(theme::space::M);
        ui.label(caps(&tr!("ISTANTE DI RIFERIMENTO", "REFERENCE INSTANT")));
        let label = |r: Option<usize>, items: &[Item]| match r {
            None => tr!("Media delle riprese", "Mean of the captures"),
            Some(i) => {
                let name = items[i].path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
                format!("{} · {name}", i + 1)
            }
        };
        let before = self.reference;
        egui::ComboBox::from_id_salt("derotation-reference").width(ui.available_width() - 8.0).selected_text(label(self.reference, &self.items)).show_ui(ui, |ui| {
            ui.selectable_value(&mut self.reference, None, label(None, &self.items));
            for i in 0..self.items.len() {
                if self.items[i].image.is_some() {
                    ui.selectable_value(&mut self.reference, Some(i), label(Some(i), &self.items));
                }
            }
        });
        if self.reference != before {
            if let Some(i) = self.reference {
                self.items[i].enabled = true;
            }
            changed = true;
        }
        match moment {
            Some(m) => ui.label(RichText::new(tr!("Il pianeta è mostrato com'era alle {:02}:{:02}:{:02} UT del {}/{}/{}", "The planet is shown as it was at {:02}:{:02}:{:02} UT on {}/{}/{}", m.hour, m.minute, m.second.floor() as u32, m.day, m.month, m.year)).size(theme::size::SMALL).color(TEXT_WEAK)),
            None => ui.label(RichText::new(tr!("Senza ore: il riferimento è la rotazione media delle riprese.", "Without times: the reference is the captures' mean turn.")).size(theme::size::SMALL).color(TEXT_WEAK)),
        };

        ui.add_space(theme::space::M);
        ui.label(caps(&tr!("OPZIONI", "OPTIONS")));
        let mut row = |ui: &mut egui::Ui, on: &mut bool, label: String, help: String| {
            ui.horizontal(|ui| {
                if toggle(ui, on).changed() {
                    changed = true;
                }
                ui.label(label).on_hover_text(help);
            });
        };
        row(ui, &mut self.use_times, tr!("Usa le ore delle riprese", "Use the capture times"), tr!("Le ore confermate dalle immagini danno la rotazione; quelle in disaccordo vengono corrette. Spento: la rotazione si misura solo sulle immagini.", "Times confirmed by the images give the turn; those that disagree are corrected. Off: the turn is measured on the images only."));
        row(ui, &mut self.weigh_sharpness, tr!("Pesa per nitidezza", "Weigh by sharpness"), tr!("Le riprese con più dettaglio contano di più nella media", "Captures with more detail count more in the mean"));
        row(ui, &mut self.clip, tr!("Scarta i valori anomali", "Reject outliers"), tr!("Sigma-clipping sul globo: in ogni pixel ignora i valori troppo lontani dagli altri (un difetto in una sola ripresa). Serve almeno tre riprese.", "Sigma clipping on the globe: in each pixel ignore values too far from the others (a defect in one capture). Needs at least three captures."));
        if self.clip {
            let label = tr!("Soglia", "Threshold");
            let help = tr!("Quante deviazioni standard dal valore centrale prima di scartare: più bassa scarta di più", "How many standard deviations from the central value before rejecting: lower rejects more");
            if ParamSlider::new(&label, &mut self.kappa, 1.0..=4.0, 2.5).decimals(1).suffix(" σ").help(&help).show(ui) {
                changed = true;
            }
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new(tr!("Verso di rotazione", "Turning direction")).size(theme::size::SMALL));
            let (auto_label, normal, mirrored) = (tr!("Automatico", "Automatic"), tr!("Normale", "Normal"), tr!("Specchiato", "Mirrored"));
            let options = [(None, None, auto_label.as_str()), (Some(false), None, normal.as_str()), (Some(true), None, mirrored.as_str())];
            if crate::widgets::segmented(ui, &mut self.mirrored, &options) {
                changed = true;
            }
        })
        .response
        .on_hover_text(tr!("Con un diagonale o un riflettore l'immagine può essere specchiata: i dettagli girano da destra a sinistra. Automatico: dal confronto fra ore e immagini, o dalle lune.", "With a diagonal or a reflector the image may be mirrored: the details turn right to left. Automatic: from comparing times and images, or from the moons."));
        if changed {
            self.dirty = true;
        }

        ui.add_space(theme::space::L);
        if let Some(text) = &self.busy {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(text).size(theme::size::SMALL).color(TEXT_WEAK));
            });
        } else if self.running || self.dirty {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(tr!("Calcolo…", "Computing…")).size(theme::size::SMALL).color(TEXT_WEAK));
            });
        }
        let job = if self.busy.is_none() && !self.running && !self.dirty { self.job() } else { None };
        let create = primary_button(Some(ICON_CHECK), &tr!("Crea l'immagine sommata", "Create the summed image")).min_size(Vec2::new(ui.available_width(), 32.0));
        let mut outcome = None;
        if ui.add_enabled(job.is_some(), create).on_hover_text(tr!("Deruota e somma le riprese a piena risoluzione e apre il risultato per l'elaborazione", "Derotates and sums the captures at full resolution and opens the result for processing")).clicked()
            && let Some(job) = job
        {
            outcome = Some(DerotationOutcome::Create(Box::new(job)));
            self.open = false;
        }
        outcome
    }

    /// Minutes between the first and the last capture summed (with the times found).
    fn span_minutes(&self) -> f64 {
        let Some((_, s)) = &self.solution else { return 0.0 };
        let turns: Vec<f64> = s.frames.iter().filter(|f| f.weight > 0.0).map(|f| f.turn_deg).collect();
        let (lo, hi) = turns.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &t| (lo.min(t), hi.max(t)));
        if turns.len() < 2 { 0.0 } else { (hi - lo) / 360.0 * s.params.period_hours * 60.0 }
    }

    /// The globe shown on capture `i`: dragged, placed by hand, or found.
    fn shown_globe(&self, i: usize) -> Option<Globe> {
        if let Some((_, g)) = self.drag.filter(|_| self.selected == i) {
            return Some(g);
        }
        self.items[i].globe.or_else(|| self.fit_of(i).map(|f| f.globe))
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let (a, b, c) = (tr!("Ripresa", "Capture"), tr!("Risultato", "Result"), tr!("Mappa", "Map"));
            let options = [(View::Capture, Some(ICON_IMAGE), a.as_str()), (View::Result, Some(ICON_AUTORENEW), b.as_str()), (View::Map, Some(ICON_PUBLIC), c.as_str())];
            crate::widgets::segmented(ui, &mut self.view, &options);
            if self.view == View::Capture && !self.items.is_empty() {
                if ui.small_button(ICON_CHEVRON_LEFT.codepoint).clicked() {
                    self.selected = (self.selected + self.items.len() - 1) % self.items.len();
                }
                if ui.small_button(ICON_CHEVRON_RIGHT.codepoint).clicked() {
                    self.selected = (self.selected + 1) % self.items.len();
                }
                ui.add(egui::Label::new(RichText::new(tr!("Trascina: il globo lo sposta, il punto sull'equatore cambia il raggio, quello sul polo lo ruota", "Drag: the globe moves it, the dot on the equator sets the radius, the one on the pole turns it")).size(theme::size::SMALL).color(TEXT_WEAK)).truncate());
            }
        });
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 8.0, theme::BG);
        let placeholder = |text: String| painter.text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(14.0), TEXT_WEAK);
        match self.view {
            View::Result => match (&self.preview, self.solution.as_ref()) {
                (Some(texture), Some((used, s))) => {
                    let size = texture.size_vec2();
                    let fit = ((rect.width() - 24.0) / size.x).min((rect.height() - 24.0) / size.y);
                    let shown = Rect::from_center_size(rect.center(), size * fit);
                    painter.image(texture.id(), shown, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                    if let Some(img) = self.items.get(used[s.grid]).and_then(|it| it.image.as_ref()) {
                        let k = shown.width() / img.width() as f32;
                        self.overlay(&painter, shown.min, k, s.frames[s.grid].globe, false);
                    }
                }
                _ => {
                    placeholder(if self.busy.is_some() || self.running || self.dirty { tr!("Preparazione dell'anteprima…", "Preparing the preview…") } else { tr!("Nessuna ripresa da sommare", "No captures to sum") });
                }
            },
            View::Map => match &self.map {
                Some(texture) => {
                    let size = texture.size_vec2();
                    let fit = ((rect.width() - 24.0) / size.x).min((rect.height() - 24.0) / size.y);
                    let shown = Rect::from_center_size(rect.center(), size * fit);
                    painter.image(texture.id(), shown, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                    painter.text(shown.center_bottom() + Vec2::new(0.0, 4.0), egui::Align2::CENTER_TOP, tr!("Mappa cilindrica: al centro il meridiano centrale all'istante di riferimento; le longitudini non viste hanno la media della loro latitudine.", "Cylindrical map: the central meridian at the reference instant in the middle; unseen longitudes take the mean of their latitude."), egui::FontId::proportional(11.0), TEXT_WEAK);
                }
                None => {
                    placeholder(tr!("Nessuna mappa", "No map"));
                }
            },
            View::Capture => {
                let i = self.selected.min(self.items.len().saturating_sub(1));
                let Some(item) = self.items.get(i) else {
                    placeholder(tr!("Nessuna ripresa", "No captures"));
                    return;
                };
                let (Some(texture), Some(img)) = (&item.thumb, &item.image) else {
                    placeholder(item.error.clone().unwrap_or_default());
                    return;
                };
                let (w, h) = (img.width() as f32, img.height() as f32);
                let fit = ((rect.width() - 24.0) / w).min((rect.height() - 24.0) / h);
                let shown = Rect::from_center_size(rect.center(), Vec2::new(w, h) * fit);
                painter.image(texture.id(), shown, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                let Some(globe) = self.shown_globe(i) else {
                    placeholder(if self.running || self.dirty { tr!("Ricerca del globo…", "Finding the globe…") } else { tr!("Globo non trovato", "Globe not found") });
                    return;
                };
                self.overlay(&painter, shown.min, fit, globe, true);
                self.drag_globe(&response, shown.min, fit, globe, i);
            }
        }
    }

    fn projection(&self) -> Projection {
        let (tilt, k) = self.solution.as_ref().map_or((0.0, ps_core::moons::flattening(self.body)), |(_, s)| (s.params.tilt_deg, s.params.oblateness));
        Projection::new(tilt, k)
    }

    /// Screen positions of the grips: centre, the end of the equator, above the pole.
    fn grips(&self, origin: Pos2, k: f32, g: Globe) -> [Pos2; 3] {
        let to_screen = |(x, y): (f32, f32)| Pos2::new(origin.x + x * k, origin.y + y * k);
        let kp = self.projection().kp;
        [to_screen(g.to_pixel(0.0, 0.0)), to_screen(g.to_pixel(1.0, 0.0)), to_screen(g.to_pixel(0.0, kp * 1.18))]
    }

    /// The globe's outline, equator and central meridian, and the grips when editable.
    fn overlay(&self, painter: &egui::Painter, origin: Pos2, k: f32, g: Globe, editable: bool) {
        let proj = self.projection();
        let to_screen = |(x, y): (f32, f32)| Pos2::new(origin.x + x * k, origin.y + y * k);
        let colour = if editable { Color32::from_rgba_unmultiplied(240, 168, 72, 200) } else { Color32::from_white_alpha(70) };
        let stroke = Stroke::new(1.2, colour);
        let outline: Vec<Pos2> = (0..120).map(|i| {
            let t = i as f64 / 120.0 * std::f64::consts::TAU;
            to_screen(g.to_pixel(t.cos(), proj.kp * t.sin()))
        }).collect();
        painter.add(egui::Shape::closed_line(outline, stroke));
        // Visible arcs of the equator and of the central meridian.
        let arc = |points: Vec<[f64; 3]>| -> Vec<Pos2> {
            points.into_iter().filter_map(|p| {
                let ((u, v), mu) = proj.to_plane(p);
                (mu >= 0.0).then(|| to_screen(g.to_pixel(u, v)))
            }).collect()
        };
        let n = 60;
        let equator = arc((0..=n).map(|i| point(-std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * i as f64 / n as f64, 0.0, proj.k)).collect());
        let meridian = arc((0..=n).map(|i| point(0.0, -std::f64::consts::FRAC_PI_2 * 0.999 + std::f64::consts::PI * 0.999 * i as f64 / n as f64, proj.k)).collect());
        painter.add(egui::Shape::line(equator, Stroke::new(1.0, colour)));
        painter.add(egui::Shape::dashed_line(&meridian, Stroke::new(1.0, colour), 5.0, 4.0));
        let [_, edge, pole] = self.grips(origin, k, g);
        painter.text(pole + Vec2::new(0.0, -15.0), egui::Align2::CENTER_CENTER, "N", egui::FontId::proportional(11.0), colour);
        if editable {
            for p in [edge, pole] {
                painter.circle(p, 5.0, theme::BG, Stroke::new(1.5, ACCENT));
            }
        }
    }

    /// Dragging the globe of capture `i`: moving, resizing, turning; on release it is kept as
    /// placed by hand and everything is computed again.
    fn drag_globe(&mut self, response: &egui::Response, origin: Pos2, k: f32, globe: Globe, i: usize) {
        let Some(pointer) = response.interact_pointer_pos().or(response.hover_pos()) else { return };
        let [centre, edge, pole] = self.grips(origin, k, globe);
        if response.drag_started() {
            let grip = if pointer.distance(edge) < 10.0 {
                Some(Grip::Resize)
            } else if pointer.distance(pole) < 10.0 {
                Some(Grip::Turn)
            } else if pointer.distance(centre) < globe.radius * k {
                Some(Grip::Move)
            } else {
                None
            };
            self.drag = grip.map(|g| (g, globe));
        }
        if let Some((grip, mut g)) = self.drag {
            let d = response.drag_delta() / k;
            match grip {
                Grip::Move => {
                    g.cx += d.x;
                    g.cy += d.y;
                }
                Grip::Resize => {
                    let to_image = |p: Pos2| ((p.x - origin.x) / k, (p.y - origin.y) / k);
                    let (x, y) = to_image(pointer);
                    g.radius = (x - g.cx).hypot(y - g.cy).max(8.0);
                }
                Grip::Turn => {
                    let (x, y) = ((pointer.x - origin.x) / k, (pointer.y - origin.y) / k);
                    // The pole points along (sin a, −cos a) on the image.
                    g.angle = (x - g.cx).atan2(-(y - g.cy));
                }
            }
            self.drag = Some((grip, g));
            if response.drag_stopped() {
                self.items[i].globe = Some(g);
                self.drag = None;
                self.dirty = true;
            }
        }
        if response.hovered() {
            let cursor = if pointer.distance(edge) < 10.0 || pointer.distance(pole) < 10.0 { egui::CursorIcon::Crosshair } else if pointer.distance(centre) < globe.radius * k { egui::CursorIcon::Grab } else { egui::CursorIcon::Default };
            response.ctx.set_cursor_icon(cursor);
        }
    }
}
