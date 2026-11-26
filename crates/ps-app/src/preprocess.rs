//! Opening an image raw or already pre-processed: what "pre-processed" means, the question asked
//! the first time an image is opened (with a before/after preview) and the window that asks which
//! planet it is when the pixels did not say.

use crate::panels::PLANETS;
use crate::theme::{self, ACCENT, BORDER, TEXT, TEXT_WEAK};
use crate::widgets::primary_button;
use eframe::egui::{self, ColorImage, RichText, Stroke, TextureHandle, TextureOptions};
use ps_core::Image;
use ps_core::analysis::{Analysis, ImageKind};
use ps_core::capture::Body;
use ps_core::pipeline::{Aux, Pipeline, Processor, next_generation};
use ps_core::preset::AutoStrength;
use ps_core::tr;
use ps_core::wavelet::ScaleMode;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// Larger images get no preview in the question (it would take too long).
const PREVIEW_PIXELS: usize = 3_000_000;

/// The pre-processed settings (see [`ps_core::preset::preprocessed`]).
pub fn pipeline(kind: ImageKind, analysis: &Analysis, source: &Image, body: Option<Body>, strength: AutoStrength, scales: ScaleMode) -> Pipeline {
    ps_core::preset::preprocessed(kind, analysis, source, body, strength, scales)
}

/// The answer to the first-open question.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    Raw,
    Preprocessed,
}

type Preview = Option<(ColorImage, ColorImage)>;

/// The question and its before/after preview, which is computed in the background.
pub struct Dialog {
    rx: Option<Receiver<Preview>>,
    textures: Option<(TextureHandle, TextureHandle)>,
    /// No preview: too large, or the processing failed.
    unavailable: bool,
}

impl Dialog {
    pub fn start(ctx: &egui::Context, source: Arc<Image>, analysis: Analysis, kind: ImageKind, body: Option<Body>, scales: ScaleMode) -> Self {
        if source.width() * source.height() > PREVIEW_PIXELS {
            return Self { rx: None, textures: None, unavailable: true };
        }
        let (tx, rx) = channel();
        let (ctx, side) = (ctx.clone(), ctx.input(|i| i.max_texture_side));
        std::thread::spawn(move || {
            let p = pipeline(kind, &analysis, &source, body, AutoStrength::Normal, scales);
            let masks = ps_core::pipeline::masks_for(&p, &source);
            let aux = Aux { masks: &masks, ..Default::default() };
            let out = Processor::default().run(Arc::clone(&source), next_generation(), &p, &aux, &|| false, &|_, _| {});
            tx.send(out.map(|o| (crate::worker::display_image(&source, side), crate::worker::display_image(&o.image, side)))).ok();
            ctx.request_repaint();
        });
        Self { rx: Some(rx), textures: None, unavailable: false }
    }

    /// Shows the question; the answer once given.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Choice> {
        if let Some(rx) = &self.rx
            && let Ok(done) = rx.try_recv()
        {
            self.rx = None;
            match done {
                Some((before, after)) => {
                    let options = TextureOptions::LINEAR;
                    self.textures = Some((ctx.load_texture("preprocess-before", before, options), ctx.load_texture("preprocess-after", after, options)));
                }
                None => self.unavailable = true,
            }
        }
        let mut answer = None;
        let frame = egui::Frame::new().fill(theme::PANEL).corner_radius(theme::radius::WINDOW).inner_margin(20).stroke(Stroke::new(1.0, BORDER));
        egui::Modal::new(egui::Id::unique("preprocess-question")).frame(frame).show(ctx, |ui| {
            ui.set_width(600.0);
            ui.label(RichText::new(tr!("Come vuoi aprire le immagini?", "How do you want images to open?")).size(theme::size::HEADING).strong().color(TEXT));
            ui.add_space(theme::space::S);
            ui.label(RichText::new(tr!(
                "Planetary Studio può aprire l'immagine così com'è, oppure già migliorata: nitidezza intelligente al 100 %, rete del rumore a metà intensità, colore fotometrico (dove c'è un riferimento), livelli con il nero e il bianco vicini agli estremi e immagine raddrizzata con il nord in alto. Potrai sempre cambiare ogni regolazione.",
                "Planetary Studio can open the image as it is, or already improved: smart sharpening at 100 %, noise network at half strength, photometric colour (where there is a reference), levels with black and white close to the ends and the image straightened north up. Every adjustment stays yours to change."
            ))
            .color(TEXT_WEAK));
            ui.add_space(theme::space::M);
            ui.columns(2, |cols| {
                for (i, caption) in [tr!("Grezza", "Raw"), tr!("Preelaborata", "Pre-processed")].into_iter().enumerate() {
                    let ui = &mut cols[i];
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new(caption).strong().color(TEXT));
                        match &self.textures {
                            Some((before, after)) => {
                                let tex = if i == 0 { before } else { after };
                                let size = tex.size_vec2();
                                let scale = (270.0 / size.x).min(270.0 / size.y).min(1.0);
                                ui.image((tex.id(), size * scale));
                            }
                            None if self.unavailable => {
                                ui.add_space(theme::space::L);
                                ui.label(RichText::new(tr!("Anteprima non disponibile", "No preview available")).size(theme::size::SMALL).color(TEXT_WEAK));
                            }
                            None => {
                                ui.add_space(theme::space::L);
                                ui.spinner();
                            }
                        }
                    });
                }
            });
            ui.add_space(theme::space::M);
            ui.label(RichText::new(tr!("La scelta vale per le prossime immagini; si cambia in Impostazioni › Apertura delle immagini. Non verrà più chiesta.", "The choice applies to the next images; change it in Settings › Opening images. It is not asked again.")).size(theme::size::SMALL).color(TEXT_WEAK));
            ui.add_space(theme::space::M);
            ui.horizontal(|ui| {
                if ui.add(primary_button(None, &tr!("Apri preelaborate (consigliato)", "Open pre-processed (recommended)"))).clicked() {
                    answer = Some(Choice::Preprocessed);
                }
                if ui.button(tr!("Apri grezze", "Open raw")).clicked() {
                    answer = Some(Choice::Raw);
                }
            });
        });
        answer
    }
}

/// The window that asks what the image shows when it was not recognised. `current` is the
/// image's kind; the answer is the kind and body picked (`Some(None)` = "other / not sure").
pub fn subject_prompt(ctx: &egui::Context, current: ImageKind) -> Option<Option<(ImageKind, Body)>> {
    let mut answer = None;
    let frame = egui::Frame::new().fill(theme::PANEL).corner_radius(theme::radius::WINDOW).inner_margin(20).stroke(Stroke::new(1.0, BORDER));
    egui::Modal::new(egui::Id::unique("subject-question")).frame(frame).show(ctx, |ui| {
        ui.set_width(420.0);
        ui.label(RichText::new(tr!("Che cosa mostra l'immagine?", "What does the image show?")).size(theme::size::HEADING).strong().color(TEXT));
        ui.add_space(theme::space::S);
        ui.label(RichText::new(tr!("Non l'ho riconosciuto dall'aspetto. Dimmelo e ti propongo la nitidezza e il rumore giusti per quel soggetto.", "I could not tell from the look of it. Tell me and I will offer the sharpening and noise reduction that fit that subject.")).color(TEXT_WEAK));
        ui.add_space(theme::space::M);
        for (body, kind, _) in PLANETS {
            if ui.add_sized([ui.available_width(), 28.0], egui::Button::new(RichText::new(body.label()).color(TEXT))).clicked() {
                answer = Some(Some((kind, body)));
            }
        }
        let disk_kind = if matches!(current, ImageKind::FullDisk | ImageKind::Surface) { current } else { ImageKind::FullDisk };
        for body in [Body::Moon, Body::Sun] {
            if ui.add_sized([ui.available_width(), 28.0], egui::Button::new(RichText::new(body.label()).color(TEXT))).clicked() {
                answer = Some(Some((disk_kind, body)));
            }
        }
        ui.add_space(theme::space::S);
        if ui.add(egui::Button::new(RichText::new(tr!("Altro / non so", "Other / not sure")).color(ACCENT)).frame(false)).clicked() {
            answer = Some(None);
        }
    });
    answer
}
