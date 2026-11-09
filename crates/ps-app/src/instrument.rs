//! The "Strumento" tab: what the image shows and when, the equipment it was taken with, and
//! what both say about the image (scale, sampling, the finest detail it can and does hold).

use crate::panels::hint;
use crate::theme::{self, ACCENT, BORDER, SUCCESS, SURFACE, TEXT, TEXT_WEAK, WARNING};
use crate::widgets::ParamSlider;
use eframe::egui::{self, Color32, RichText, Sense, Stroke, Ui, Vec2};
use egui_material_icons::icons::*;
use ps_core::Image;
use ps_core::analysis::{Analysis, image_scale};
use ps_core::capture::{self, Body, CaptureInfo};
use ps_core::equipment::{self, Barlow, Camera, Filter, Setup, Telescope};
use ps_core::wavelet::{ScaleMode, WaveletParams};
use std::path::{Path, PathBuf};

/// What is known about the capture of the open image.
#[derive(Clone, Debug, Default)]
pub struct CaptureState {
    /// From the file name and the log.
    pub info: CaptureInfo,
    /// The log read, if any.
    pub log: Option<PathBuf>,
    /// The subject: the one named, unless the user picked another.
    pub body: Option<Body>,
    /// Scale measured from the subject's size and its apparent diameter (arcsec per pixel).
    pub scale: Option<f32>,
}

impl CaptureState {
    /// From the file name and the first log found beside the image (the log wins).
    pub fn read(path: &Path) -> CaptureState {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let mut state = CaptureState { info: capture::from_name(stem), ..Default::default() };
        if let Some((log, text)) = capture::logs_beside(path).into_iter().find_map(|p| std::fs::read_to_string(&p).ok().map(|t| (p, t))) {
            state.info = capture::from_log(&text).or(state.info);
            state.log = Some(log);
        }
        state.body = state.info.body;
        state
    }

    /// A log chosen by the user: what it says replaces what was known.
    pub fn add_log(&mut self, path: &Path, text: &str) {
        let info = capture::from_log(text);
        self.body = info.body.or(self.body);
        self.info = info.or(std::mem::take(&mut self.info));
        self.log = Some(path.to_path_buf());
    }

    /// Measure the scale again (subject or time changed).
    pub fn measure(&mut self, img: &Image, analysis: &Analysis) {
        self.scale = match (self.body, self.info.time) {
            (Some(body), Some(when)) => image_scale(img, analysis, body, &when),
            _ => None,
        };
    }
}

pub enum InstrumentAction {
    /// The equipment changed: the optics and the structural settings follow.
    SetupChanged,
    /// The subject changed: the scale is measured again.
    BodyChanged,
    /// Pick a capture log to read.
    LoadLog,
}

/// A short name for a setup, for the top bar: "C8 · ASI462MC", or what is missing.
pub fn setup_label(setup: &Setup) -> String {
    let short = |name: &str| name.split(" (").next().unwrap_or(name).to_string();
    let parts: Vec<String> = [setup.telescope.as_ref().map(|t| short(&t.name)), setup.camera.as_ref().map(|c| short(&c.name))].into_iter().flatten().collect();
    if parts.is_empty() { "Strumento non indicato".into() } else { parts.join(" · ") }
}

pub struct InstrumentView<'a> {
    pub setup: &'a mut Setup,
    pub capture: &'a mut CaptureState,
    pub analysis: &'a Analysis,
    pub wavelet: &'a WaveletParams,
}

pub fn instrument_body(ui: &mut Ui, view: InstrumentView, actions: &mut Vec<InstrumentAction>) {
    let InstrumentView { setup, capture, analysis, wavelet } = view;
    let section = |ui: &mut Ui, title: &str| {
        ui.add_space(theme::space::L);
        ui.label(RichText::new(title).size(theme::size::SMALL).strong().color(TEXT_WEAK));
        ui.add_space(theme::space::XS);
    };

    section(ui, "OGGETTO");
    let mut body = capture.body;
    let source = if capture.body.is_some() && capture.body == capture.info.body {
        if capture.log.is_some() { " · dal log" } else { " · dal nome del file" }
    } else if capture.body.is_some() && capture.body == analysis.planet {
        " · dall'aspetto"
    } else {
        ""
    };
    choose(ui, "Soggetto", &mut body, Body::ALL.iter().map(|b| (Some(*b), b.label().to_string())).collect(), "Non indicato", source);
    if body != capture.body {
        capture.body = body;
        actions.push(InstrumentAction::BodyChanged);
    }
    let when = capture.info.time.map_or("data e ora non indicate nel nome del file".to_string(), |t| format!("{:02}/{:02}/{} {:02}:{:02} UT", t.day, t.month, t.year, t.hour, t.minute));
    hint(ui, &format!("Ripresa: {when}. Con oggetto e data si calcola la scala dell'immagine dal diametro apparente."));

    section(ui, "STRUMENTAZIONE");
    let before = setup.clone();
    choose_custom(
        ui,
        "Telescopio",
        &mut setup.telescope,
        equipment::telescopes(),
        |t| &t.name,
        || Telescope { name: "Personalizzato".into(), aperture_mm: 200.0, focal_mm: 2000.0, obstruction: 0.33 },
    );
    if let Some(t) = setup.telescope.as_mut().filter(|t| t.name == "Personalizzato") {
        ParamSlider::new("Apertura", &mut t.aperture_mm, 50.0..=1000.0, 200.0).decimals(0).suffix(" mm").show(ui);
        ParamSlider::new("Focale", &mut t.focal_mm, 200.0..=10000.0, 2000.0).decimals(0).suffix(" mm").show(ui);
        ParamSlider::new("Ostruzione", &mut t.obstruction, 0.0..=0.6, 0.33).help("Diametro dello specchio secondario diviso per l'apertura (0 per i rifrattori)").show(ui);
    }
    choose_custom(ui, "Camera", &mut setup.camera, equipment::cameras(), |c| &c.name, || Camera { name: "Personalizzata".into(), pixel_um: 2.9, colour: true });
    if let Some(c) = setup.camera.as_mut().filter(|c| c.name == "Personalizzata") {
        ParamSlider::new("Pixel", &mut c.pixel_um, 1.0..=10.0, 2.9).suffix(" µm").show(ui);
    }
    choose_custom(ui, "Barlow", &mut setup.barlow, equipment::barlows(), |b| &b.name, || Barlow { name: "Personalizzata".into(), factor: 2.0 });
    if let Some(b) = setup.barlow.as_mut().filter(|b| b.name == "Personalizzata") {
        ParamSlider::new("Ingrandimento", &mut b.factor, 0.5..=8.0, 2.0).suffix("×").help("L'ingrandimento reale dipende dalla distanza tra Barlow e sensore").show(ui);
    }
    choose_custom(ui, "Filtro", &mut setup.filter, equipment::filters(), |f| &f.name, || Filter { name: "Personalizzato".into(), wavelength_nm: 550.0 });
    if let Some(f) = setup.filter.as_mut().filter(|f| f.name == "Personalizzato") {
        ParamSlider::new("Lunghezza d'onda", &mut f.wavelength_nm, 350.0..=1000.0, 550.0).decimals(0).suffix(" nm").help("Lunghezza d'onda media della luce che arriva al sensore").show(ui);
    }
    ParamSlider::new("Binning", &mut setup.binning, 1..=4, 1).decimals(0).show(ui);
    ParamSlider::new("Drizzle", &mut setup.drizzle, 1.0..=3.0, 1.0).suffix("×").help("Drizzle o ricampionamento di AutoStakkert: 1,5× rende i pixel 1,5 volte più piccoli").show(ui);
    if let Some(focal) = setup.focal_mm {
        ui.horizontal(|ui| {
            hint(ui, &format!("Focale effettiva {focal:.0} mm, dal log."));
            if ui.small_button("Usa telescopio × Barlow").clicked() {
                setup.focal_mm = None;
            }
        });
    }
    if *setup != before {
        actions.push(InstrumentAction::SetupChanged);
    }
    ui.add_space(theme::space::S);
    ui.horizontal(|ui| {
        if ui
            .add(egui::Button::new((ICON_UPLOAD_FILE.rich_text().size(15.0), RichText::new("Carica log…"))))
            .on_hover_text("FireCapture, SharpCap o AutoStakkert: camera, filtro, focale, data e drizzle")
            .clicked()
        {
            actions.push(InstrumentAction::LoadLog);
        }
        if let Some(log) = &capture.log {
            ui.label(RichText::new(log.file_name().and_then(|n| n.to_str()).unwrap_or("")).size(theme::size::SMALL).color(TEXT_WEAK));
        }
    });
    hint(ui, "La strumentazione viene ricordata per le prossime immagini. Un log accanto all'immagine, con lo stesso nome, viene letto da solo.");

    section(ui, "DIAGNOSI");
    diagnosis(ui, setup, capture, analysis);
    ui.add_space(theme::space::M);
    spectrum_graph(ui, analysis, wavelet);
}

/// A row "label: combo" choosing among `options` (or none).
fn choose<T: PartialEq + Clone>(ui: &mut Ui, label: &str, value: &mut T, options: Vec<(T, String)>, none: &str, note: &str) {
    ui.horizontal(|ui| {
        label_cell(ui, label, 90.0);
        let current = options.iter().find(|(v, _)| v == value).map_or(none.to_string(), |(_, n)| n.clone());
        egui::ComboBox::from_id_salt(label).width((ui.available_width() - 8.0).max(80.0)).selected_text(format!("{current}{note}")).show_ui(ui, |ui| {
            for (v, name) in options {
                if ui.selectable_label(*value == v, name).clicked() {
                    *value = v;
                }
            }
        });
    });
}

/// [`choose`] for a catalogue with "Non indicato" and a custom entry.
fn choose_custom<T: PartialEq + Clone>(ui: &mut Ui, label: &str, value: &mut Option<T>, catalogue: Vec<T>, name_of: impl Fn(&T) -> &String, custom: impl Fn() -> T) {
    ui.horizontal(|ui| {
        label_cell(ui, label, 90.0);
        let current = value.as_ref().map_or("Non indicato".to_string(), |v| name_of(v).clone());
        egui::ComboBox::from_id_salt(label).width((ui.available_width() - 8.0).max(80.0)).height(360.0).selected_text(current).show_ui(ui, |ui| {
            if ui.selectable_label(value.is_none(), "Non indicato").clicked() {
                *value = None;
            }
            for item in catalogue {
                let selected = value.as_ref() == Some(&item);
                if ui.selectable_label(selected, name_of(&item).as_str()).clicked() {
                    *value = Some(item);
                }
            }
            let custom = custom();
            let is_custom = value.as_ref().is_some_and(|v| name_of(v) == name_of(&custom));
            if ui.selectable_label(is_custom, format!("{}…", name_of(&custom))).clicked() && !is_custom {
                *value = Some(custom);
            }
        });
    });
}

/// A label in a left-aligned column of fixed width.
fn label_cell(ui: &mut Ui, label: &str, width: f32) {
    let height = ui.spacing().interact_size.y;
    ui.allocate_ui_with_layout(Vec2::new(width, height), egui::Layout::left_to_right(egui::Align::Center), |ui| {
        ui.set_width(width);
        ui.label(RichText::new(label).color(TEXT_WEAK));
    });
}

/// "label  value" with an optional note under it.
fn fact(ui: &mut Ui, label: &str, value: &str, colour: Color32, note: Option<&str>) {
    ui.horizontal(|ui| {
        label_cell(ui, label, 150.0);
        ui.label(RichText::new(value).color(colour));
    });
    if let Some(note) = note {
        hint(ui, note);
    }
}

fn period(cutoff: f32) -> String {
    format!("{:.1} px", 1.0 / cutoff.max(1e-3))
}

fn diagnosis(ui: &mut Ui, setup: &Setup, capture: &CaptureState, analysis: &Analysis) {
    let optics = setup.optics();
    // Scale: from the equipment and from the subject's size.
    match (optics, capture.scale) {
        (Some(o), Some(m)) => {
            let off = (o.arcsec_per_px / m - 1.0).abs();
            let note = if off > 0.08 {
                format!("Dal diametro del soggetto risultano {m:.3}″/px, il {:.0} % di differenza. Controlla Barlow, distanza dal sensore o drizzle.", off * 100.0)
            } else {
                format!("Concorda con quanto misurato sul diametro del soggetto ({m:.3}″/px).")
            };
            fact(ui, "Scala", &format!("{:.3}″/px · f/{:.1}", o.arcsec_per_px, o.focal_ratio), if off > 0.08 { WARNING } else { TEXT }, Some(&note));
        }
        (Some(o), None) => fact(ui, "Scala", &format!("{:.3}″/px · f/{:.1}", o.arcsec_per_px, o.focal_ratio), TEXT, None),
        (None, Some(m)) => {
            let focal = setup.pixel().map(|p| 206.265 * p / m);
            let note = match (focal, setup.telescope.as_ref()) {
                (Some(f), Some(t)) => Some(format!("Focale effettiva ≈ {f:.0} mm, cioè f/{:.1} con questo telescopio (Barlow ≈ {:.1}×).", f / t.aperture_mm, f / t.focal_mm)),
                (Some(f), None) => Some(format!("Focale effettiva ≈ {f:.0} mm con i pixel di questa camera.")),
                _ => Some("Indica camera e telescopio per sapere focale e limite di diffrazione.".to_string()),
            };
            fact(ui, "Scala misurata", &format!("{m:.3}″/px"), TEXT, note.as_deref());
        }
        (None, None) => hint(ui, "Indica telescopio, camera e Barlow (o oggetto e data) per conoscere la scala dell'immagine."),
    }
    if let Some(o) = optics {
        fact(ui, "Limite di diffrazione", &format!("periodo {} a {:.0} nm", period(o.cutoff), o.wavelength_nm), TEXT, None);
        let (verdict, colour) = match o.oversampling {
            s if s < 0.8 => ("Sottocampionata, i dettagli più fini si perdono. Una Barlow più forte aiuterebbe.".to_string(), WARNING),
            s if s < 1.3 => ("Ideale, al limite di Nyquist.".to_string(), SUCCESS),
            s if s < 2.3 => ("Un po' sovracampionata, normale per avere margine con il seeing.".to_string(), TEXT),
            s => (format!("Molto sovracampionata (×{s:.1}). Con una Barlow più corta avresti più luce e meno rumore senza perdere dettaglio."), WARNING),
        };
        fact(ui, "Campionamento", &format!("×{:.1}", o.oversampling), colour, Some(&verdict));
        fact(ui, "Disco di Airy", &format!("FWHM {:.1} px", o.airy_fwhm_px), TEXT, None);
    }
    match (analysis.spectral, optics) {
        (Some(s), Some(o)) => {
            let reached = (s.cutoff / o.cutoff).min(1.0);
            let note = if s.cutoff > o.cutoff * 1.05 {
                "Lo spettro va oltre il limite dello strumento. Quella parte è rumore, oppure lo strumento indicato non è quello giusto.".to_string()
            } else {
                format!("Il dettaglio vero finisce a un periodo di {}, il {:.0} % della risoluzione teorica (il resto lo toglie il seeing).", period(s.cutoff), reached * 100.0)
            };
            fact(ui, "Risoluzione raggiunta", &format!("{:.0} %", reached * 100.0), if reached > 0.7 { SUCCESS } else { TEXT }, Some(&note));
        }
        (Some(s), None) => fact(ui, "Dettaglio più fine", &format!("periodo {}", period(s.cutoff)), TEXT, Some("Dove lo spettro dell'immagine scende sotto il rumore.")),
        (None, _) => hint(ui, "Spettro non misurabile su questa immagine (soggetto troppo piccolo o rumore non riconoscibile)."),
    }
    if let Some(sigma) = analysis.psf_sigma {
        let arcsec = optics.map(|o| o.arcsec_per_px).or(capture.scale).map_or(String::new(), |s| format!(" · {:.2}″", 2.3548 * sigma * s));
        fact(ui, "Sfocatura sul bordo", &format!("FWHM {:.1} px{arcsec}", 2.3548 * sigma), TEXT, None);
    }
    fact(ui, "Segnale / rumore", &format!("{:.0}", analysis.snr), TEXT, None);
}

/// The measured spectrum, the fitted signal and noise, where the detail ends, the diffraction
/// limit and where the wavelet layers sit.
fn spectrum_graph(ui: &mut Ui, analysis: &Analysis, wavelet: &WaveletParams) {
    if analysis.power.is_empty() {
        return;
    }
    let width = ui.available_width() - 4.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 150.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 6.0, SURFACE);
    let plot = rect.shrink2(Vec2::new(8.0, 14.0));
    let (f_lo, f_hi) = (0.02f32.ln(), 0.5f32.ln());
    let x = |f: f32| plot.left() + (f.max(0.02).ln() - f_lo) / (f_hi - f_lo) * plot.width();
    let powers: Vec<f32> = analysis.power.iter().map(|(_, p)| p.max(1e-30).log10()).collect();
    let (p_lo, p_hi) = (powers.iter().copied().fold(f32::MAX, f32::min) - 0.3, powers.iter().copied().fold(f32::MIN, f32::max) + 0.2);
    let y = |p: f32| plot.bottom() - (p.max(1e-30).log10() - p_lo) / (p_hi - p_lo).max(1e-3) * plot.height();

    // Period labels along the bottom.
    for px in [32.0f32, 16.0, 8.0, 4.0, 2.0] {
        let xx = x(1.0 / px);
        painter.line_segment([egui::pos2(xx, plot.top()), egui::pos2(xx, plot.bottom())], Stroke::new(1.0, BORDER));
        painter.text(egui::pos2(xx, rect.bottom() - 2.0), egui::Align2::CENTER_BOTTOM, format!("{px:.0} px"), egui::FontId::proportional(9.0), TEXT_WEAK);
    }
    if let Some(s) = analysis.spectral {
        let curve =
            |f: &dyn Fn(f32) -> f32| -> Vec<egui::Pos2> { (0..=80).map(|i| 0.02 * (25.0f32).powf(i as f32 / 80.0)).map(|fr| egui::pos2(x(fr), y(f(fr)).clamp(plot.top(), plot.bottom()))).collect() };
        painter.add(egui::Shape::line(curve(&|f| s.noise(f)), Stroke::new(1.0, TEXT_WEAK)));
        painter.add(egui::Shape::line(curve(&|f| s.signal(f)), Stroke::new(1.5, ACCENT)));
        let xx = x(s.cutoff);
        painter.line_segment([egui::pos2(xx, plot.top()), egui::pos2(xx, plot.bottom())], Stroke::new(1.5, ACCENT));
        painter.text(egui::pos2(xx + 3.0, plot.top()), egui::Align2::LEFT_TOP, "dettaglio", egui::FontId::proportional(9.0), ACCENT);
    }
    if let Some(o) = analysis.optics {
        let xx = x(o.cutoff.min(0.5));
        painter.line_segment([egui::pos2(xx, plot.top()), egui::pos2(xx, plot.bottom())], Stroke::new(1.0, WARNING));
        painter.text(egui::pos2(xx - 3.0, plot.top() + 12.0), egui::Align2::RIGHT_TOP, "diffrazione", egui::FontId::proportional(9.0), WARNING);
    }
    for (&(f, _), p) in analysis.power.iter().zip(&powers) {
        painter.circle_filled(egui::pos2(x(f), y(10f32.powf(*p))), 1.6, TEXT);
    }
    // Wavelet layers: each peaks at about 0.3 / its size (free scales) or 0.25 / step.
    for j in 0..wavelet.layers.len() {
        let peak = if wavelet.scale_mode == ScaleMode::Free { 0.3 / wavelet.scale(j) } else { 0.25 / wavelet.scale(j) };
        if (0.02..=0.5).contains(&peak) {
            let xx = x(peak);
            painter.text(egui::pos2(xx, plot.bottom() - 2.0), egui::Align2::CENTER_BOTTOM, format!("{}", j + 1), egui::FontId::proportional(10.0), TEXT);
        }
    }
    hint(ui, "Spettro dell'immagine: puntini misurati, in arancio il segnale, in grigio il rumore. I numeri sono i livelli wavelet.");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A FireCapture log beside the stack is found by name and wins over the file name.
    #[test]
    fn reads_the_log_beside_the_image() {
        let dir = std::env::temp_dir().join(format!("ps-log-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("2026-08-29-2009_9-U-RGB-Jup.txt"), "Camera=ZWO ASI462MC\nFilter=IR685\nProfile=Jupiter\nFocalLength=5080mm\nDate=290826\nMid(UT)=201000\nDate_format=ddMMyy\n")
            .unwrap();
        let state = CaptureState::read(&dir.join("2026-08-29-2009_9-U-RGB-Jup_AS_P20_lapl4_ap116.tif"));
        std::fs::remove_dir_all(&dir).ok();
        assert!(state.log.is_some());
        assert_eq!(state.body, Some(Body::Jupiter));
        assert_eq!(state.info.filter.as_deref(), Some("IR685"));
        let setup = Setup::from_capture(&state.info, &Setup::default());
        assert_eq!(setup.camera.unwrap().pixel_um, 2.9);
        assert_eq!(setup.focal_mm, Some(5080.0));
        assert_eq!(setup.filter.unwrap().name, "IR 685 nm");
    }
}
