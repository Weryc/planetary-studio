//! Several images opened together: a window asks what to do with them (a mosaic, a derotated
//! sum, a colour image from monochrome channels, the same derotated) instead of guessing from
//! the subject. An action that cannot work with these files is greyed out with the reason.

use crate::instrument::CaptureState;
use crate::theme::{self, TEXT, TEXT_WEAK};
use eframe::egui::{self, RichText, Vec2};
use egui_material_icons::icons::*;
use ps_core::capture::Body;
use ps_core::i18n::tl;
use ps_core::tr;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

/// What to do with the files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    Mosaic,
    Derotation,
    MonoRgb,
    MonoRgbDerotated,
    Animation,
}

/// What the window decided.
pub struct Chosen {
    pub choice: Choice,
    pub paths: Vec<PathBuf>,
    /// The subject, when the names or the pixels say it.
    pub body: Option<Body>,
}

/// What is known of one file.
struct Facts {
    name: String,
    size: Option<(usize, usize)>,
    mono: bool,
    error: Option<String>,
}

enum Msg {
    Facts { facts: Vec<Facts>, body: Option<Body> },
}

pub struct ChooseEditor {
    pub open: bool,
    paths: Vec<PathBuf>,
    facts: Vec<Facts>,
    body: Option<Body>,
    loading: bool,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
}

impl Default for ChooseEditor {
    fn default() -> Self {
        let (tx, rx) = channel();
        Self { open: false, paths: Vec::new(), facts: Vec::new(), body: None, loading: false, tx, rx }
    }
}

/// Whether `choice` can be done with these files, else why not. `None` for the facts while
/// they are being read.
fn availability(choice: Choice, facts: &[Facts], body: Option<Body>) -> Result<(), String> {
    if let Some(f) = facts.iter().find(|f| f.error.is_some()) {
        return Err(tr!("{} non si apre", "{} cannot be opened", f.name));
    }
    let not_derotatable = body.filter(|b| !crate::derotation_ui::derotates(*b));
    let planet_only = |not: Option<Body>| match not {
        Some(b) => Err(tr!("Il soggetto è {}: la derotazione vale per Giove, Saturno e Marte", "The subject is {}: derotation applies to Jupiter, Saturn and Mars", b.label())),
        None => Ok(()),
    };
    let channels = || {
        if let Some(f) = facts.iter().find(|f| !f.mono) {
            return Err(tr!("{} è a colori: servono immagini monocromatiche", "{} is a colour image: monochrome images are needed", f.name));
        }
        if !(3..=4).contains(&facts.len()) {
            return Err(tr!("Servono 3 immagini (R, G, B) o 4 (con L): ne hai scelte {}", "3 images (R, G, B) or 4 (with L) are needed: you chose {}", facts.len()));
        }
        Ok(())
    };
    match choice {
        Choice::Mosaic => Ok(()),
        Choice::Animation => Ok(()),
        Choice::Derotation => planet_only(not_derotatable),
        Choice::MonoRgb => channels(),
        Choice::MonoRgbDerotated => planet_only(not_derotatable).and_then(|_| channels()),
    }
}

impl ChooseEditor {
    pub fn start(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        let ctx = ctx.clone();
        let list = paths.clone();
        // A fresh channel: the reply of this thread must reach the window that replaces `self`.
        *self = Self { open: true, loading: true, paths, ..Default::default() };
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let mut named = None;
            let mut facts = Vec::with_capacity(list.len());
            let mut pixels = None;
            for path in &list {
                named = named.or(CaptureState::read(path).body);
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
                match ps_io::load(path) {
                    Ok(img) => {
                        if named.is_none() && pixels.is_none() {
                            pixels = Some(crate::derotation_ui::body_from_pixels(&img));
                        }
                        facts.push(Facts { name, size: Some((img.width(), img.height())), mono: img.is_mono(), error: None });
                    }
                    Err(e) => facts.push(Facts { name, size: None, mono: true, error: Some(e.to_string()) }),
                }
            }
            tx.send(Msg::Facts { facts, body: named.or(pixels.flatten()) }).ok();
            ctx.request_repaint();
        });
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<Chosen> {
        if !self.open {
            return None;
        }
        while let Ok(Msg::Facts { facts, body }) = self.rx.try_recv() {
            self.facts = facts;
            self.body = body;
            self.loading = false;
        }
        let mut chosen = None;
        let mut open = self.open;
        crate::widgets::window(ctx, tl("Più immagini: cosa vuoi fare?")).open(&mut open).resizable(false).default_width(520.0).show(ctx, |ui| {
            ui.label(RichText::new(tr!("{} immagini aperte insieme", "{} images opened together", self.paths.len())).color(TEXT));
            egui::ScrollArea::vertical().id_salt("choose-files").max_height(110.0).show(ui, |ui| {
                for (k, path) in self.paths.iter().enumerate() {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                    let detail = match self.facts.get(k) {
                        Some(f) => match (&f.error, f.size) {
                            (Some(_), _) => tl("non si apre").to_string(),
                            (None, Some((w, h))) => format!("{w}×{h} · {}", if f.mono { tl("mono") } else { tl("colore") }),
                            _ => String::new(),
                        },
                        None => String::new(),
                    };
                    ui.label(RichText::new(format!("{name}   {detail}")).size(theme::size::SMALL).color(TEXT_WEAK));
                }
            });
            if let Some(b) = self.body
                && !self.loading
            {
                ui.label(RichText::new(tr!("Soggetto riconosciuto: {}", "Subject recognised: {}", b.label())).size(theme::size::SMALL).color(TEXT_WEAK));
            }
            ui.add_space(theme::space::M);
            if self.loading {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new(tl("Lettura delle immagini…")).size(theme::size::SMALL).color(TEXT_WEAK));
                });
                return;
            }
            let options = [
                (Choice::Mosaic, ICON_GRID_VIEW, tl("Mosaico"), tl("Unisce immagini sovrapposte dello stesso soggetto (Luna, Sole…) in una sola")),
                (Choice::Derotation, ICON_AUTORENEW, tl("Derotazione (somma)"), tl("Somma più riprese di Giove o Saturno portandole allo stesso istante")),
                (Choice::MonoRgb, ICON_PALETTE, tl("Mono → RGB"), tl("Unisce le immagini monocromatiche R, G, B (e L) in una a colori, allineandole")),
                (Choice::MonoRgbDerotated, ICON_PALETTE, tl("Mono → RGB derotato"), tl("Come Mono → RGB, ma ogni canale è portato allo stesso istante con la rotazione del pianeta (Giove, Saturno)")),
                (Choice::Animation, ICON_MOVIE, tl("Animazione"), tl("Una GIF o un video fluido dalle riprese, allineate e interpolate fra l'una e l'altra")),
            ];
            for (choice, icon, title, help) in options {
                let available = availability(choice, &self.facts, self.body);
                ui.horizontal(|ui| {
                    let button = egui::Button::new((icon.rich_text(), title)).min_size(Vec2::new(210.0, 30.0));
                    if ui.add_enabled(available.is_ok(), button).on_hover_text(help).clicked() {
                        chosen = Some(Chosen { choice, paths: self.paths.clone(), body: self.body });
                    }
                    let (text, colour) = match &available {
                        Ok(()) => (help.to_string(), TEXT_WEAK),
                        Err(why) => (why.clone(), theme::WARNING),
                    };
                    ui.add(egui::Label::new(RichText::new(text).size(theme::size::SMALL).color(colour)).wrap());
                });
                ui.add_space(theme::space::XS);
            }
        });
        self.open &= open;
        if chosen.is_some() {
            self.open = false;
        }
        chosen
    }
}
