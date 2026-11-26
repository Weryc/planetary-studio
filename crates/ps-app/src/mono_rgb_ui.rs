//! The "Mono → RGB" window: monochrome captures taken through R, G and B filters (and
//! optionally L) become one colour image, either lined up as they are (classic) or, for Jupiter
//! and Saturn, each brought to the same instant by the planet's rotation (derotated). The
//! checks live in `ps_core::monocolor`; here the files are listed, their channels and times
//! assigned (from the file name or the log, else by hand), and the first problem is said in
//! words, with the button disabled until there is none.

use crate::instrument::CaptureState;
use crate::theme::{self, TEXT, TEXT_WEAK};
use crate::widgets::{caps, primary_button, toggle};
use eframe::egui::{self, RichText, Vec2};
use egui_material_icons::icons::*;
use ps_core::Image;
use ps_core::capture::{Body, Moment};
use ps_core::derotation::JupiterSystem;
use ps_core::i18n::tl;
use ps_core::monocolor::{self, Input, Options, Output, Role};
use ps_core::tr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

/// What the window asks the app to do.
pub enum MonoRgbOutcome {
    /// Build the colour image and open it.
    Create(Box<MonoJob>),
}

/// One image of the job.
#[derive(Clone)]
pub struct JobItem {
    pub name: String,
    pub image: Arc<Image>,
    pub role: Role,
    pub time: Option<Moment>,
}

/// Everything the merge needs, away from the window.
#[derive(Clone)]
pub struct MonoJob {
    pub items: Vec<JobItem>,
    /// Set for the derotated version.
    pub derotate: Option<Body>,
    pub options: Options,
    /// The file the result is named after.
    pub path: PathBuf,
}

impl MonoJob {
    pub fn lrgb(&self) -> bool {
        self.items.iter().any(|i| i.role == Role::L)
    }

    /// Merge, returning the colour image with the instant its channels refer to: the one they
    /// were brought to, else the mean of the times known.
    pub fn run(&self) -> Result<Output, monocolor::Error> {
        let inputs: Vec<Input> = self.items.iter().map(|i| Input { name: i.name.clone(), image: &i.image, role: i.role, time: i.time, globe: None }).collect();
        let mut out = match self.derotate {
            Some(body) => monocolor::compose_derotated(&inputs, Some(body), &self.options)?,
            None => monocolor::compose(&inputs, &self.options)?,
        };
        if out.moment.is_none() {
            let times: Vec<(Moment, f32)> = self.items.iter().filter_map(|i| i.time.map(|t| (t, 1.0))).collect();
            // Only when every capture has one: a mean of some would be a half-truth.
            if times.len() == self.items.len() {
                out.moment = ps_core::derotation::mean_moment(&times);
            }
        }
        Ok(out)
    }
}

struct Row {
    path: PathBuf,
    name: String,
    image: Option<Arc<Image>>,
    error: Option<String>,
    role: Option<Role>,
    /// `AAAA-MM-DD HH:MM:SS`, or only the time when the date is the one of the other captures.
    time_text: String,
    time_from: &'static str,
}

enum Msg {
    Progress(String),
    Loaded { rows: Vec<Row>, body: Option<Body> },
}

pub struct MonoRgbEditor {
    pub open: bool,
    derotated: bool,
    rows: Vec<Row>,
    body: Option<Body>,
    system: JupiterSystem,
    mirrored: bool,
    /// Pick the turning direction automatically.
    auto_mirror: bool,
    align: bool,
    resize: bool,
    busy: Option<String>,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
}

impl Default for MonoRgbEditor {
    fn default() -> Self {
        let (tx, rx) = channel();
        Self { open: false, derotated: false, rows: Vec::new(), body: None, system: JupiterSystem::II, mirrored: false, auto_mirror: true, align: true, resize: false, busy: None, tx, rx }
    }
}

pub fn format_time(t: &Moment) -> String {
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", t.year, t.month, t.day, t.hour, t.minute, t.second.floor() as u32)
}

/// A time as typed: `AAAA-MM-DD HH:MM[:SS]`, or `HH:MM[:SS]` with the date of `date`.
pub fn parse_time(text: &str, date: Option<(i32, u32, u32)>) -> Option<Moment> {
    let mut parts = text.split_whitespace();
    let (first, second) = (parts.next()?, parts.next());
    if parts.next().is_some() {
        return None;
    }
    let (day, clock) = match second {
        Some(c) => (Some(first), c),
        None => (None, first),
    };
    let (year, month, d) = match day {
        Some(d) => {
            let n: Vec<i32> = d.split(['-', '/']).map(|p| p.parse().ok()).collect::<Option<_>>()?;
            if n.len() != 3 {
                return None;
            }
            (n[0], n[1] as u32, n[2] as u32)
        }
        None => date?,
    };
    let c: Vec<&str> = clock.split(':').collect();
    if !(2..=3).contains(&c.len()) {
        return None;
    }
    let hour: u32 = c[0].parse().ok()?;
    let minute: u32 = c[1].parse().ok()?;
    let second: f32 = c.get(2).map_or(Some(0.0), |s| s.replace(',', ".").parse().ok())?;
    let m = Moment { year, month, day: d, hour, minute, second };
    (hour < 24 && minute < 60 && (0.0..60.0).contains(&second) && (1..=12).contains(&month) && (1..=31).contains(&d)).then_some(m)
}

impl MonoRgbEditor {
    /// The images of these captures; `derotated` starts the window in that mode, `body` is the
    /// planet when the names say it.
    pub fn start(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>, derotated: bool, body: Option<Body>) {
        *self = Self { open: true, derotated, body: body.filter(|b| ps_core::derotation::rotation_period(*b, JupiterSystem::II).is_some()), ..Default::default() };
        self.busy = Some(tr!("Caricamento di {} immagini…", "Loading {} images…", paths.len()));
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let total = paths.len();
            let mut named = None;
            let mut rows = Vec::with_capacity(total);
            for (k, path) in paths.into_iter().enumerate() {
                tx.send(Msg::Progress(tr!("Analisi dell'immagine {} di {total}…", "Analysing image {} of {total}…", k + 1))).ok();
                ctx.request_repaint();
                let capture = CaptureState::read(&path);
                named = named.or(capture.body);
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                let role = monocolor::guess_role(capture.info.filter.as_deref(), stem);
                let (image, error) = match ps_io::load(&path) {
                    Ok(img) => (Some(Arc::new(img)), None),
                    Err(e) => (None, Some(tr!("{name}: non si apre ({e})", "{name}: cannot be opened ({e})"))),
                };
                let time_text = capture.info.time.as_ref().map(format_time).unwrap_or_default();
                rows.push(Row { path, name, image, error, role, time_text, time_from: capture.time_from });
            }
            // The same channel named twice is not a guess to keep.
            for role in Role::ALL {
                if rows.iter().filter(|r| r.role == Some(role)).count() > 1 {
                    rows.iter_mut().filter(|r| r.role == Some(role)).for_each(|r| r.role = None);
                }
            }
            let body = named.or_else(|| rows.iter().find_map(|r| r.image.as_deref()).and_then(crate::derotation_ui::body_from_pixels));
            tx.send(Msg::Loaded { rows, body }).ok();
            ctx.request_repaint();
        });
    }

    fn receive(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Progress(text) => self.busy = Some(text),
                Msg::Loaded { rows, body } => {
                    self.rows = rows;
                    self.body = self.body.or(body.filter(|b| crate::derotation_ui::derotates(*b)));
                    self.busy = None;
                }
            }
        }
    }

    /// The date typed times without a date take: the first complete one.
    fn default_date(&self) -> Option<(i32, u32, u32)> {
        self.rows.iter().find_map(|r| parse_time(&r.time_text, None)).map(|m| (m.year, m.month, m.day))
    }

    fn options(&self) -> Options {
        Options { resize: self.resize, align: self.align, system: self.system, mirrored: if self.auto_mirror { None } else { Some(self.mirrored) } }
    }

    /// The job as set now, or the first thing that is wrong, in words.
    pub(crate) fn job(&self) -> Result<MonoJob, String> {
        if self.busy.is_some() {
            return Err(tl("Caricamento in corso…").to_string());
        }
        if let Some(e) = self.rows.iter().find_map(|r| r.error.clone()) {
            return Err(e);
        }
        // A colour image is wrong whatever its channel.
        if let Some(r) = self.rows.iter().find(|r| r.image.as_ref().is_some_and(|i| !i.is_mono())) {
            return Err(monocolor::Error::NotMono(r.name.clone()).message());
        }
        let date = self.default_date();
        let mut items = Vec::new();
        for r in &self.rows {
            let (Some(role), Some(image)) = (r.role, &r.image) else { continue };
            let time = if r.time_text.trim().is_empty() {
                None
            } else {
                Some(parse_time(&r.time_text, date).ok_or_else(|| tr!("L'ora di {} non è valida: scrivi AAAA-MM-DD HH:MM:SS (o solo HH:MM:SS se la data è quella delle altre).", "The time of {} is not valid: write YYYY-MM-DD HH:MM:SS (or just HH:MM:SS if the date is the others').", r.name))?)
            };
            items.push(JobItem { name: r.name.clone(), image: Arc::clone(image), role, time });
        }
        let options = self.options();
        let job = MonoJob { items, derotate: if self.derotated { self.body } else { None }, options, path: PathBuf::new() };
        let inputs: Vec<Input> = job.items.iter().map(|i| Input { name: i.name.clone(), image: &i.image, role: i.role, time: i.time, globe: None }).collect();
        let checked = if self.derotated { monocolor::check_derotated(&inputs, self.body, &options).map(|_| ()) } else { monocolor::validate(&inputs, options.resize).map(|_| ()) };
        checked.map_err(|e| e.message())?;
        // The result is named after the green capture (or the luminance one).
        let named = job.items.iter().find(|i| i.role == Role::G).or(job.items.first()).map(|i| i.name.clone()).unwrap_or_default();
        let path = self.rows.iter().find(|r| r.name == named).map(|r| r.path.clone()).unwrap_or_default();
        Ok(MonoJob { path, ..job })
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<MonoRgbOutcome> {
        if !self.open {
            return None;
        }
        self.receive();
        let mut outcome = None;
        let mut open = self.open;
        crate::widgets::window(ctx, tl("Mono → RGB")).open(&mut open).default_size([640.0, 520.0]).min_size([520.0, 360.0]).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| outcome = self.body_ui(ui));
        });
        self.open &= open;
        outcome
    }

    fn body_ui(&mut self, ui: &mut egui::Ui) -> Option<MonoRgbOutcome> {
        ui.label(caps(tl("MODO")));
        let options = [(false, None, tl("Classico")), (true, None, tl("Derotato (Giove, Saturno)"))];
        crate::widgets::segmented(ui, &mut self.derotated, &options);
        crate::panels::hint(
            ui,
            if self.derotated {
                tl("Ogni canale è portato allo stesso istante compensando la rotazione del pianeta: serve l'ora di ogni ripresa. Per Giove e Saturno, riprese a minuti di distanza.")
            } else {
                tl("I canali vengono allineati fra loro e fusi. Con un canale L si ottiene un LRGB: il colore viene da R, G e B, il dettaglio da L.")
            },
        );

        ui.add_space(theme::space::M);
        ui.label(caps(tl("IMMAGINI")));
        let date = self.default_date();
        let derotated = self.derotated;
        egui::Grid::new("mono-rgb-rows").num_columns(if derotated { 4 } else { 3 }).spacing([10.0, 6.0]).show(ui, |ui| {
            for r in &mut self.rows {
                ui.add(egui::Label::new(RichText::new(&r.name).size(theme::size::SMALL).color(TEXT)).truncate()).on_hover_text(r.path.display().to_string());
                let size = r.image.as_ref().map(|i| format!("{}×{}{}", i.width(), i.height(), if i.is_mono() { "" } else { " · colore" }));
                ui.label(RichText::new(size.unwrap_or_default()).size(theme::size::SMALL).color(if r.image.as_ref().is_some_and(|i| !i.is_mono()) { theme::DANGER } else { TEXT_WEAK }));
                egui::ComboBox::from_id_salt(("mono-rgb-role", &r.path))
                    .width(90.0)
                    .selected_text(r.role.map_or(tl("— esclusa").to_string(), |x| x.label().to_string()))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut r.role, None, tl("— esclusa"));
                        for role in [Role::R, Role::G, Role::B, Role::L] {
                            ui.selectable_value(&mut r.role, Some(role), match role {
                                Role::R => tl("R · rosso"),
                                Role::G => tl("G · verde"),
                                Role::B => tl("B · blu"),
                                Role::L => tl("L · luminanza"),
                            });
                        }
                    });
                if derotated {
                    let valid = r.time_text.trim().is_empty() || parse_time(&r.time_text, date).is_some();
                    let mut edit = egui::TextEdit::singleline(&mut r.time_text).desired_width(150.0).hint_text(tl("AAAA-MM-DD HH:MM:SS"));
                    if !valid {
                        edit = edit.text_color(theme::DANGER);
                    }
                    let response = ui.add(edit);
                    if !r.time_from.is_empty() && !r.time_text.is_empty() {
                        response.on_hover_text(r.time_from);
                    }
                }
                ui.end_row();
            }
        });
        if self.busy.is_none() && self.rows.iter().any(|r| r.role.is_none()) {
            ui.label(RichText::new(tl("Le immagini «esclusa» non vengono usate.")).size(theme::size::SMALL).color(TEXT_WEAK));
        }

        ui.add_space(theme::space::M);
        ui.label(caps(tl("OPZIONI")));
        if self.derotated {
            ui.horizontal(|ui| {
                ui.label(RichText::new(tl("Pianeta")).size(theme::size::SMALL));
                for body in [Body::Jupiter, Body::Saturn] {
                    if ui.selectable_label(self.body == Some(body), body.label()).clicked() {
                        self.body = Some(body);
                    }
                }
                if self.body == Some(Body::Jupiter) {
                    let options = [(JupiterSystem::II, None, tl("Sistema II")), (JupiterSystem::I, None, tl("Sistema I"))];
                    crate::widgets::segmented(ui, &mut self.system, &options);
                }
            });
            ui.horizontal(|ui| {
                toggle(ui, &mut self.auto_mirror);
                ui.label(tl("Verso di rotazione automatico")).on_hover_text(tl("Prova i due versi e tiene quello che fa coincidere meglio i dischi dei canali"));
            });
            ui.add_enabled_ui(!self.auto_mirror, |ui| {
              ui.horizontal(|ui| {
                toggle(ui, &mut self.mirrored);
                ui.label(tl("Immagine specchiata")).on_hover_text(tl("Con un diagonale o un riflettore l'immagine può essere specchiata: i dettagli sembrano girare da destra a sinistra"));
              });
            });
        } else {
            ui.horizontal(|ui| {
                toggle(ui, &mut self.align);
                ui.label(tl("Allinea i canali")).on_hover_text(tl("Corregge gli spostamenti fra le immagini con la correlazione di fase, usando L (o G) come riferimento"));
            });
        }
        let sizes: Vec<(usize, usize)> = self.rows.iter().filter_map(|r| r.image.as_ref().map(|i| (i.width(), i.height()))).collect();
        if sizes.windows(2).any(|w| w[0] != w[1]) {
            ui.horizontal(|ui| {
                toggle(ui, &mut self.resize);
                ui.label(tl("Ricampiona alla dimensione del riferimento")).on_hover_text(tl("Le immagini più piccole o più grandi vengono portate alla dimensione di L (o di G) con un'interpolazione bilineare"));
            });
        }

        ui.add_space(theme::space::L);
        if let Some(text) = &self.busy {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(text).size(theme::size::SMALL).color(TEXT_WEAK));
            });
        }
        let job = self.job();
        let lrgb = self.rows.iter().any(|r| r.role == Some(Role::L));
        let label = match (lrgb, self.derotated) {
            (true, true) => tl("Crea l'LRGB derotato"),
            (true, false) => tl("Crea l'LRGB"),
            (false, true) => tl("Crea l'RGB derotato"),
            (false, false) => tl("Crea l'RGB"),
        };
        if let Err(why) = &job
            && self.busy.is_none()
        {
            ui.label(RichText::new(why).size(theme::size::SMALL).color(theme::WARNING));
        }
        let create = primary_button(Some(ICON_CHECK), label).min_size(Vec2::new(ui.available_width(), 32.0));
        let mut outcome = None;
        if ui.add_enabled(job.is_ok(), create).on_hover_text(tl("Unisce i canali a piena risoluzione e apre il risultato per l'elaborazione")).clicked()
            && let Ok(job) = job
        {
            outcome = Some(MonoRgbOutcome::Create(Box::new(job)));
            self.open = false;
        }
        outcome
    }
}
