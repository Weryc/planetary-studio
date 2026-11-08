//! The right-hand panel: image type, then one collapsible card per module. Each card shows
//! the few controls that matter; everything else lives under a collapsed "Avanzate" section.

use crate::prefs::{Group, stage_icon};
use crate::theme::{self, ACCENT, ACCENT_DIM, BG, BORDER, SUCCESS, SURFACE_HOVER, TEXT, TEXT_WEAK};
use crate::naming::NameKeyword;
use crate::widgets::{Backdrop, Histogram, HistogramMode, ParamSlider, curve_editor, icon_button, segmented, toggle};
use eframe::egui::{self, Color32, CornerRadius, RichText, Sense, Stroke, TextureHandle, TextureOptions, Ui, Vec2};
use egui_material_icons::MaterialIcon;
use egui_material_icons::icons::*;
use ps_core::adjust::{ColorParams, MineralParams};
use ps_core::align::AlignParams;
use ps_core::analysis::{Analysis, Confidence, ImageKind};
use ps_core::preset::AutoStrength;
use ps_core::bxt::BxtParams;
use ps_core::color::ColorMode;
use ps_core::deconv::{DeconvAlgorithm, DeconvParams, DeconvReport};
use ps_core::dering::{DeringParams, DeringRegion};
use ps_core::enhance::{ContrastParams, NoiseMethod, NoiseParams, ToneParams};
use ps_core::mask::{AutoRegion, MaskShape};
use ps_core::pipeline::{Layer, LayerRole, Stage, StageTiming};
use ps_core::psf::{Kernel2D, Psf};
use ps_core::smart::{SmartCurve, SmartParams, SmartView};
use ps_core::wavelet::{MAX_LAYERS, ScaleMode, ThresholdMode, WaveletKernel, WaveletLayer, WaveletParams};

pub fn format_duration(d: std::time::Duration) -> String {
    let ms = d.as_secs_f64() * 1e3;
    if ms < 1000.0 { format!("{ms:.0} ms") } else { format!("{:.2} s", ms / 1e3) }
}

/// A grey hint of one sentence; anything after it is shown on hover, so cards stay short.
pub fn hint(ui: &mut Ui, text: &str) {
    let (first, rest) = match text.find(". ").or_else(|| text.find(": ")) {
        Some(i) if i + 2 < text.len() => (&text[..=i], Some(text)),
        _ => (text, None),
    };
    let label = ui.add(egui::Label::new(RichText::new(first).size(theme::size::SMALL).color(TEXT_WEAK)).wrap());
    if let Some(full) = rest {
        label.on_hover_text(full);
    }
}

/// Requests from the panel that the app carries out.
pub enum PanelAction {
    SelectKind(ImageKind),
    /// Show the PSF shapes and algorithms side by side.
    CompareDeconvolution,
    /// Every automatic setting, simple ones included.
    AutoAll,
    /// Automatic settings for one module.
    AutoModule(Stage),
    /// A module's settings back to the starting point.
    ResetModule(Stage),
    /// Drop every setting the analysis chose and go back to generic values.
    ClearAuto,
    /// Structural settings from the image (true) or generic (false).
    SetAutoStructure(bool),
    /// Export next to the original, in the chosen format.
    Export,
    /// Export choosing where.
    ExportAs,
    SavePreset,
    LoadPreset,
    OpenSettings,
    LoadLuminance,
    ClearLuminance,
    AddLayer(LayerTemplate),
    /// Automatic black/white points, for the whole image (`None`) or a layer.
    AutoLevels(Option<u64>),
    /// Click on the image to set the black or white point.
    Pick(PickTarget),
    /// Create the dedicated layer behind the "Anelli" / "Lune" card.
    CreateRole(LayerRole),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickTarget {
    Black,
    White,
    /// A star or small moon whose shape becomes the PSF.
    Psf,
}

#[derive(Clone, Copy)]
pub enum LayerTemplate {
    Auto(AutoRegion),
    Ellipse,
    Rectangle,
    /// Empty mask, painted with the brush.
    Brush,
}

// ---- image type --------------------------------------------------------------------------

fn kind_icon(kind: ImageKind) -> MaterialIcon {
    match kind {
        ImageKind::Planet => ICON_PUBLIC,
        ImageKind::Saturn => ICON_PLANET,
        ImageKind::FullDisk => ICON_BRIGHTNESS_1,
        ImageKind::Surface => ICON_TERRAIN,
    }
}

/// The panel's first rows: what the image is (detected, or chosen from the menu) and the
/// automatic settings, with their strength and "remove" in the button's arrow menu.
/// `customised` = settings differ from the automatic ones.
pub fn start_row(ui: &mut Ui, current: ImageKind, analysis: &Analysis, customised: bool, auto_structure: bool, strength: &mut AutoStrength, actions: &mut Vec<PanelAction>) {
    ui.horizontal(|ui| {
        ui.label(kind_icon(current).rich_text().size(theme::size::ICON).color(ACCENT));
        egui::ComboBox::from_id_salt("image-kind")
            .width(140.0)
            .selected_text(RichText::new(current.label()).size(theme::size::BODY))
            .show_ui(ui, |ui| {
                for kind in ImageKind::ALL {
                    let icon = kind_icon(kind);
                    let text = (icon.rich_text().size(theme::size::ICON), RichText::new(kind.label()).size(theme::size::BODY), RichText::new(format!("  {}", kind.description())).size(theme::size::SMALL).color(TEXT_WEAK));
                    if ui.add(egui::Button::selectable(kind == current, text)).clicked() && kind != current {
                        actions.push(PanelAction::SelectKind(kind));
                    }
                }
            });
        let (mark, text) = if current == analysis.kind { (ICON_AUTO_AWESOME, "riconosciuto") } else { (ICON_TOUCH_APP, "scelto a mano") };
        let confidence = match analysis.confidence {
            Confidence::High => "alta",
            Confidence::Medium => "media",
            Confidence::Low => "bassa",
        };
        let mut facts = vec![format!("Rilevato: {} (affidabilità {confidence})", analysis.kind.label()), format!("SNR {:.0}", analysis.snr)];
        if let Some(s) = analysis.psf_sigma {
            facts.push(format!("Sfocatura misurata sul bordo: σ {s:.2} px"));
        }
        if let Some((fwhm, beta)) = analysis.psf_moffat {
            facts.push(format!("Ali di seeing (Moffat): FWHM {fwhm:.1} px, β {beta:.1}"));
        }
        ui.label(mark.rich_text().size(theme::size::SMALL).color(TEXT_WEAK));
        ui.label(RichText::new(text).size(theme::size::SMALL).color(TEXT_WEAK)).on_hover_text(facts.join("\n"));
    });
    ui.add_space(theme::space::M);

    // Whether the advanced settings follow the image: a switch with its state spelled out.
    let mut on = auto_structure;
    let frame = egui::Frame::new()
        .fill(if on { theme::SURFACE } else { theme::PANEL })
        .stroke(Stroke::new(1.0, if on { ACCENT_DIM } else { BORDER }))
        .corner_radius(CornerRadius::same(7))
        .inner_margin(egui::Margin::symmetric(10, 6));
    frame
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                if toggle(ui, &mut on).changed() {
                    actions.push(PanelAction::SetAutoStructure(on));
                }
                ui.label(RichText::new("Impostazioni avanzate automatiche").size(theme::size::BODY).color(TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (text, colour) = if on { ("accese", ACCENT) } else { ("spente", TEXT_WEAK) };
                    ui.label(RichText::new(text).size(theme::size::BODY).strong().color(colour));
                });
            });
        })
        .response
        .on_hover_text(if on {
            "PSF, livelli wavelet, metodo antirumore e area del dering sono scelti in base all'immagine (vedi \"Perché queste impostazioni\" in Avanzate). Spegnendo si torna a valori generici; le tue regolazioni restano."
        } else {
            "Le impostazioni avanzate hanno valori generici. Accendendo vengono scelte in base all'immagine; le tue regolazioni restano."
        });
    ui.add_space(theme::space::S);

    // Applying the automatic amounts is an action, not a state: an ordinary button.
    ui.horizontal(|ui| {
        let arrow_w = 17.0 + 2.0 * ui.spacing().button_padding.x + 4.0;
        let label = format!("Regola in automatico · {}", strength.label());
        let auto = egui::Button::new((ICON_AUTO_FIX_HIGH.rich_text().color(ACCENT), RichText::new(label).color(TEXT)))
            .min_size(Vec2::new(ui.available_width() - arrow_w - ui.spacing().item_spacing.x - 2.0, 28.0))
            .corner_radius(CornerRadius::same(7));
        let tip = if customised {
            "Imposta nitidezza (deconvoluzione, wavelet, dering) e riduzione del rumore con i valori scelti dall'analisi. Si annulla con ⌘Z / Ctrl+Z."
        } else {
            "Le regolazioni automatiche sono già quelle applicate"
        };
        if ui.add(auto).on_hover_text(tip).clicked() {
            actions.push(PanelAction::AutoAll);
        }
        // Wide enough for the icon and the button's padding: a row even a point wider than the
        // panel makes the resizable panel grow every frame.
        let arrow = egui::Button::new(ICON_EXPAND_MORE.rich_text().color(TEXT)).min_size(Vec2::new(arrow_w, 28.0)).corner_radius(CornerRadius::same(7));
        egui::containers::menu::MenuButton::from_button(arrow).ui(ui, |ui| {
            ui.label(RichText::new("Intensità").size(theme::size::SMALL).color(TEXT_WEAK));
            for (s, what) in [
                (AutoStrength::Gentle, "naturale, quasi senza artefatti"),
                (AutoStrength::Normal, "il compromesso consigliato"),
                (AutoStrength::Strong, "tutto il dettaglio che i dati contengono"),
            ] {
                if ui.add(egui::Button::selectable(*strength == s, (RichText::new(s.label()), RichText::new(format!("  {what}")).size(theme::size::SMALL).color(TEXT_WEAK)))).clicked() {
                    *strength = s;
                }
            }
            ui.separator();
            if ui.button((ICON_AUTO_FIX_OFF.rich_text(), "Togli tutte le regolazioni")).on_hover_text("Torna a nulla applicato e a impostazioni avanzate generiche. I livelli restano. Si annulla con ⌘Z / Ctrl+Z.").clicked() {
                actions.push(PanelAction::ClearAuto);
            }
        });
    });
}

/// One tab per area: icon over name, the open one underlined in the accent colour, a count of
/// the modules acting in each.
pub fn area_tabs(ui: &mut Ui, areas: &[(Group, usize)], current: &mut Group) {
    let gap = 2.0;
    let n = areas.len().max(1) as f32;
    let width = ((ui.available_width() - gap * (n - 1.0)) / n).floor();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = gap;
        for (k, (group, count)) in areas.iter().enumerate() {
            let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 46.0), Sense::click());
            let selected = *group == *current;
            let painter = ui.painter_at(rect);
            let fill = if selected { theme::SURFACE } else if response.hovered() { theme::SURFACE_HOVER } else { theme::PANEL };
            painter.rect_filled(rect, CornerRadius::same(6), fill);
            if selected {
                painter.rect_filled(egui::Rect::from_min_max(egui::pos2(rect.left() + 8.0, rect.bottom() - 3.0), egui::pos2(rect.right() - 8.0, rect.bottom())), CornerRadius::same(2), ACCENT);
            }
            let colour = if selected { TEXT } else { TEXT_WEAK };
            let icon = group.icon();
            // Names too long for the tab go on two lines (split at the space nearest the middle).
            let font = egui::FontId::proportional(theme::size::SMALL);
            let name = group.name();
            let fits = ui.fonts_mut(|f| f.layout_no_wrap(name.to_string(), font.clone(), colour).size().x) <= width - 6.0;
            let lines: Vec<&str> = match name.char_indices().filter(|(_, c)| *c == ' ').min_by_key(|(i, _)| i.abs_diff(name.len() / 2)) {
                Some((i, _)) if !fits => vec![&name[..i], &name[i + 1..]],
                _ => vec![name],
            };
            let icon_y = if lines.len() == 1 { 15.0 } else { 11.0 };
            painter.text(rect.center_top() + Vec2::new(0.0, icon_y), egui::Align2::CENTER_CENTER, icon.codepoint, egui::FontId::new(18.0, icon.font_family()), if selected { ACCENT } else { colour });
            for (k, line) in lines.iter().enumerate() {
                let y = if lines.len() == 1 { 34.0 } else { 27.0 + 11.0 * k as f32 };
                painter.text(rect.center_top() + Vec2::new(0.0, y), egui::Align2::CENTER_CENTER, *line, font.clone(), colour);
            }
            if *count > 0 {
                let badge = rect.right_top() + Vec2::new(-11.0, 10.0);
                painter.circle_filled(badge, 7.0, theme::SUCCESS);
                painter.text(badge, egui::Align2::CENTER_CENTER, count.to_string(), egui::FontId::proportional(9.5), Color32::BLACK);
            }
            let tip = format!("{} ({}){}", group.name(), k + 1, if *count > 0 { format!(" · {count} attivi") } else { String::new() });
            if response.on_hover_text(tip).clicked() {
                *current = *group;
            }
        }
    });
}

// ---- module card -------------------------------------------------------------------------

/// What a card shows about the last run of its stage.
#[derive(Default, Clone, Copy)]
pub struct StageStatus<'a> {
    pub timing: Option<&'a StageTiming>,
    pub running: Option<f32>,
    pub warning: Option<&'a str>,
}

/// Header buttons of a card: offered (`auto`/`reset` on entry) and clicked (on return).
#[derive(Clone, Copy, Default)]
pub struct CardButtons {
    pub auto: bool,
    pub reset: bool,
}

/// Collapsible card for a pipeline module, with "automatic" and "reset" buttons in the header.
/// `active`: the module currently changes the image (only then is the card highlighted).
/// Returns which of the buttons was clicked.
#[allow(clippy::too_many_arguments)]
pub fn module_card(ui: &mut Ui, stage: Stage, enabled: Option<&mut bool>, active: bool, collapsed: &mut bool, status: StageStatus, body: impl FnOnce(&mut Ui)) -> CardButtons {
    if stage == Stage::Layers {
        card(ui, stage_icon(stage), stage.name(), false, enabled, active, collapsed, status, None, body);
        return CardButtons::default();
    }
    let mut buttons = CardButtons { auto: true, reset: true };
    card(ui, stage_icon(stage), stage.name(), false, enabled, active, collapsed, status, Some(&mut buttons), body);
    buttons
}

/// Collapsible card: chevron + icon + title (click to fold, double click to switch off), then
/// timing, the automatic and reset buttons and the on/off switch. Folded cards show only the
/// header; the body is dimmed when off. The accent marks `active` cards, the ones changing the
/// image. In `nested` cards (sections of a layer) the switch means "own settings instead of the
/// whole image's".
#[allow(clippy::too_many_arguments)]
fn card(
    ui: &mut Ui,
    icon: MaterialIcon,
    title: &str,
    nested: bool,
    enabled: Option<&mut bool>,
    active: bool,
    collapsed: &mut bool,
    status: StageStatus,
    buttons: Option<&mut CardButtons>,
    body: impl FnOnce(&mut Ui),
) {
    let on = enabled.as_ref().is_none_or(|e| **e);
    let active = active && on;
    let stroke = Stroke::new(1.0, if active { ACCENT_DIM } else { BORDER });
    let frame = if nested {
        egui::Frame::new().fill(BG).corner_radius(CornerRadius::same(7)).inner_margin(egui::Margin::same(8))
    } else {
        theme::card().inner_margin(egui::Margin::symmetric(12, 6))
    };
    #[cfg(test)]
    {
        // Lets tests watch for layout jumps.
        let top = ui.cursor().top();
        ui.ctx().data_mut(|d| d.insert_temp(egui::Id::unique(("card-top", title)), top));
    }
    frame.stroke(stroke).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            let chevron = if *collapsed { ICON_KEYBOARD_ARROW_RIGHT } else { ICON_KEYBOARD_ARROW_DOWN };
            let size = if nested { theme::size::BODY } else { theme::size::TITLE };
            let title = egui::Button::new((
                chevron.rich_text().size(16.0).color(TEXT_WEAK),
                icon.rich_text().size(size + 5.0).color(if active { ACCENT } else { TEXT_WEAK }),
                RichText::new(title).size(size).strong().color(if on { TEXT } else { TEXT_WEAK }),
            ))
            .frame(false);
            let tip = if nested || enabled.is_none() { "Clic: apri o chiudi" } else { "Clic: apri o chiudi · doppio clic: escludi o riattiva" };
            let title = ui.add(title).on_hover_text(tip);
            let mut enabled = enabled;
            if title.double_clicked()
                && !nested
                && let Some(e) = enabled.as_deref_mut()
            {
                // The first click of the two folded or unfolded the card: put it back.
                *e = !*e;
                *collapsed = !*collapsed;
            } else if title.clicked() {
                *collapsed = !*collapsed;
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(enabled) = enabled {
                    let tip = if nested {
                        "Impostazioni proprie del livello invece di quelle dell'immagine intera"
                    } else if *enabled {
                        "Acceso: clic per spegnerlo e vedere l'immagine senza"
                    } else {
                        "Spento: clic per accenderlo"
                    };
                    toggle(ui, enabled).on_hover_text(tip);
                }
                if let Some(b) = buttons {
                    let offered = *b;
                    *b = CardButtons::default();
                    if offered.auto {
                        b.auto = icon_button(ui, ICON_AUTO_FIX_HIGH, "Impostazioni automatiche per questo modulo").clicked();
                    }
                    if offered.reset {
                        b.reset = icon_button(ui, ICON_RESTART_ALT, "Ripristina le impostazioni iniziali di questo modulo").clicked();
                    }
                }
                if on
                    && status.running.is_none()
                    && let Some(t) = status.timing
                {
                    let (text, color) = if t.cached { ("in cache".to_string(), SUCCESS) } else { (format_duration(t.elapsed), TEXT_WEAK) };
                    ui.label(RichText::new(text).size(theme::size::SMALL).color(color));
                }
            });
        });
        // The progress strip always takes its space, so nothing below moves while processing.
        let (strip, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 3.0), Sense::hover());
        if let Some(f) = status.running.filter(|_| on) {
            ui.painter().rect_filled(strip, CornerRadius::same(1), theme::SURFACE_HOVER);
            let done = egui::Rect::from_min_size(strip.min, Vec2::new(strip.width() * f.clamp(0.0, 1.0), strip.height()));
            ui.painter().rect_filled(done, CornerRadius::same(1), ACCENT);
        }
        if let Some(w) = status.warning.filter(|_| on) {
            ui.horizontal_wrapped(|ui| {
                ui.label(ICON_WARNING.rich_text().color(ui.visuals().warn_fg_color));
                ui.label(RichText::new(w).size(theme::size::SMALL).color(ui.visuals().warn_fg_color));
            });
        }
        if !*collapsed {
            ui.add_space(if nested { theme::space::S } else { theme::space::M });
            ui.add_enabled_ui(on, body);
        }
    });
    ui.add_space(if nested { theme::space::S } else { theme::space::M });
}

/// "Why these settings": the reasoning behind the automatic choices for this image.
fn why_box(ui: &mut Ui, lines: &[String]) {
    if lines.is_empty() {
        return;
    }
    ui.add_space(theme::space::S);
    egui::Frame::new().fill(theme::SURFACE).corner_radius(CornerRadius::same(6)).inner_margin(egui::Margin::same(8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(ICON_INFO.rich_text().size(14.0).color(ACCENT));
            ui.label(RichText::new("Perché queste impostazioni").size(theme::size::SMALL).strong().color(TEXT));
        });
        for line in lines {
            ui.add(egui::Label::new(RichText::new(format!("• {line}")).size(theme::size::SMALL).color(TEXT_WEAK)).wrap());
        }
    });
}

/// Sub-section of a card, boxed so it reads as nested inside it.
fn advanced(ui: &mut Ui, id: &str, body: impl FnOnce(&mut Ui)) {
    ui.add_space(theme::space::M);
    egui::Frame::new()
        .fill(BG)
        .corner_radius(CornerRadius::same(6))
        .inner_margin(egui::Margin { left: 6, right: 8, top: 2, bottom: 4 })
        .stroke(Stroke::new(1.0, BORDER))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let title = RichText::new("AVANZATE").size(theme::size::SMALL).strong().extra_letter_spacing(0.8).color(TEXT_WEAK);
            egui::CollapsingHeader::new(title)
                .icon(|ui, openness, response| {
                    let chevron = if openness > 0.5 { ICON_KEYBOARD_ARROW_DOWN } else { ICON_KEYBOARD_ARROW_RIGHT };
                    let color = ui.style().interact(response).fg_stroke.color;
                    ui.painter().text(response.rect.center(), egui::Align2::CENTER_CENTER, chevron.codepoint, egui::FontId::new(14.0, chevron.font_family()), color);
                })
            .id_salt(id)
            .default_open(false)
            .show(ui, |ui| {
                ui.add_space(theme::space::S);
                body(ui);
            });
        });
}

fn field_label(ui: &mut Ui, text: &str) {
    ui.add_space(theme::space::XS);
    ui.label(RichText::new(text).color(TEXT_WEAK));
}

fn color_mode(ui: &mut Ui, mode: &mut ColorMode) {
    field_label(ui, "Colore");
    segmented(ui, mode, &[(ColorMode::Luminance, None, "Luminanza"), (ColorMode::Rgb, None, "R · G · B")]);
}

// ---- RGB alignment -----------------------------------------------------------------------

pub fn align_body(ui: &mut Ui, p: &mut AlignParams, mono: bool) {
    if mono {
        hint(ui, "Disponibile solo per immagini a colori.");
        return;
    }
    ui.horizontal(|ui| {
        hint(ui, "Spostamento dei canali rispetto al verde");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if icon_button(ui, ICON_RESTART_ALT, "Azzera").clicked() {
                *p = AlignParams::default();
            }
        });
    });
    ui.add_space(theme::space::XS);
    let red = Color32::from_rgb(235, 100, 100);
    let blue = Color32::from_rgb(110, 150, 245);
    for (name, color, shift) in [("Rosso", red, &mut p.red), ("Blu", blue, &mut p.blue)] {
        ui.label(RichText::new(name).size(theme::size::SMALL).strong().color(color));
        ParamSlider::new("Orizzontale", &mut shift[0], -8.0..=8.0, 0.0).suffix(" px").show(ui);
        ParamSlider::new("Verticale", &mut shift[1], -8.0..=8.0, 0.0).suffix(" px").show(ui);
    }
}

// ---- BlurXTerminator ---------------------------------------------------------------------

pub fn bxt_body(ui: &mut Ui, p: &mut BxtParams, configured: bool, running: Option<f32>, why: &[String], actions: &mut Vec<PanelAction>) {
    if !configured {
        hint(ui, "Indica dove si trova lo strumento rc-astro di RC Astro.");
        if ui.button((ICON_SETTINGS.rich_text(), "Apri impostazioni")).clicked() {
            actions.push(PanelAction::OpenSettings);
        }
        return;
    }
    hint(ui, "Deconvoluzione IA di RC Astro, con il programma esterno rc-astro. Più lenta: ogni modifica richiede qualche secondo per passaggio.");
    // What is happening right now, since the work runs in another program.
    let passes = p.passes.clamp(1, 5);
    let status = match running {
        Some(f) => format!("In corso: passaggio {} di {passes}…", ((f * passes as f32) as u32 + 1).min(passes)),
        None => "Pronto: il risultato è in cache finché non cambi le impostazioni.".to_string(),
    };
    ui.horizontal(|ui| {
        if running.is_some() {
            ui.add(egui::Spinner::new().size(12.0).color(ACCENT));
        }
        ui.label(RichText::new(status).size(theme::size::SMALL).color(if running.is_some() { ACCENT } else { TEXT_WEAK }));
    });
    ui.add_space(theme::space::XS);
    ui.horizontal(|ui| {
        toggle(ui, &mut p.lunar_planetary);
        ui.label("Modalità Luna e pianeti").on_hover_text("--lunar-planetary: modello dedicato (rc-astro 2.6 o successivo)");
    });
    ui.add_enabled_ui(!p.correct_only, |ui| {
        ParamSlider::new("Nitidezza", &mut p.sharpen_nonstellar, 0.0..=1.0, 0.7).help("--sharpen-nonstellar: quanto rendere nitidi i dettagli estesi").show(ui);
    });
    ParamSlider::new("Passaggi", &mut p.passes, 1..=3, 1)
        .decimals(0)
        .help("Esegue BlurXTerminator più volte di seguito, ognuna sul risultato della precedente")
        .show(ui);
    if !p.auto_psf {
        ParamSlider::new("Diametro PSF", &mut p.psf_diameter, 0.0..=8.0, 3.0)
            .suffix(" px")
            .help("--nonstellar-diameter: impostato dalla sfocatura misurata sul bordo del disco")
            .show(ui);
    }
    advanced(ui, "bxt-advanced", |ui| {
        ui.horizontal(|ui| {
            toggle(ui, &mut p.auto_psf);
            ui.label("PSF stimata da BlurXTerminator").on_hover_text("La stima usa le stelle: utile solo se nell'immagine ce ne sono");
        });
        ui.horizontal(|ui| {
            toggle(ui, &mut p.correct_only);
            ui.label("Solo correzione (senza aumentare la nitidezza)");
        });
        ui.add_enabled_ui(!p.correct_only, |ui| {
            ParamSlider::new("Nitidezza stelle", &mut p.sharpen_stars, 0.0..=0.7, 0.0).show(ui);
        });
        why_box(ui, why);
    });
}

// ---- deconvolution -----------------------------------------------------------------------

/// `gpu`: a graphics card is available and allowed. `report`: what the last run did.
#[allow(clippy::too_many_arguments)]
/// What the deconvolution card needs besides its settings.
pub struct DeconvView<'a> {
    /// The automatic settings for this image: what the ↺ buttons go back to.
    pub auto: &'a DeconvParams,
    /// The equipment's optics, for the Airy PSF.
    pub optics: Option<ps_core::equipment::Optics>,
    pub preview: &'a mut PsfPreview,
    pub why: &'a [String],
    pub gpu: bool,
    pub report: Option<DeconvReport>,
    /// What the last measurement on a star or moon found (or why it failed).
    pub probe: Option<&'a str>,
}

/// One control, the width of the blur to remove, with the PSF drawn beside it; how many
/// iterations and when to stop are automatic (advanced settings).
pub fn deconvolution_body(ui: &mut Ui, p: &mut DeconvParams, view: DeconvView, actions: &mut Vec<PanelAction>) {
    let DeconvView { auto, optics, preview, why, gpu, report, probe } = view;
    let algorithm = match p.algorithm {
        DeconvAlgorithm::RichardsonLucy => "Richardson–Lucy",
        DeconvAlgorithm::Wiener => "Wiener",
    };
    let status = match (p.algorithm, report.filter(|r| r.iterations > 0)) {
        (DeconvAlgorithm::Wiener, _) => " · in un passo".to_string(),
        (_, Some(r)) if p.auto_stop && r.iterations < p.iterations => format!(" · fermata da sola dopo {} iterazioni", r.iterations),
        (_, Some(r)) => format!(" · {} iterazioni", r.iterations),
        _ => String::new(),
    };
    let on_gpu = gpu && p.algorithm == DeconvAlgorithm::RichardsonLucy && matches!(p.psf, Psf::Gaussian { .. });
    hint(ui, &format!("{algorithm} · {}{status}{}", psf_name(&p.psf), if on_gpu { " · su GPU" } else { "" }));
    ui.add_space(theme::space::S);
    match p.psf {
        Psf::Measured { fwhm } => {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("PSF misurata sull'immagine, larga {fwhm:.1} px")).color(TEXT));
                if ui.small_button("Torna alla PSF automatica").clicked() {
                    (p.psf, p.measured) = (auto.psf, None);
                }
            });
        }
        psf => {
            let mut fwhm = psf.fwhm_px();
            let changed = ParamSlider::new("Sfocatura da togliere", &mut fwhm, 0.5..=12.0, auto.psf.fwhm_px())
                .suffix(" px")
                .help("Quanto è larga la sfocatura che la deconvoluzione toglie (FWHM della PSF). Il valore automatico viene dallo spettro dell'immagine e ↺ lo ripristina. Più larga = più nitidezza; se esagerata compaiono macchie e aloni.")
                .show(ui);
            if changed {
                p.psf = psf.with_fwhm(fwhm);
            }
        }
    }
    ui.horizontal(|ui| {
        preview.show(ui, &p.kernel(1.0), 48.0);
        ui.vertical(|ui| {
            ui.set_width(ui.available_width());
            hint(ui, psf_shape_note(&p.psf));
        });
    });
    ui.add_space(theme::space::S);
    ui.horizontal_wrapped(|ui| {
        if ui.add(egui::Button::new((ICON_GRID_VIEW.rich_text(), "Confronta"))).on_hover_text("Le tre forme di PSF e i due algoritmi affiancati, sulla zona visibile dell'immagine").clicked() {
            actions.push(PanelAction::CompareDeconvolution);
        }
        if ui.add(egui::Button::new((ICON_HIGHLIGHT_ALT.rich_text(), "Misura su stella o luna"))).on_hover_text("Disegna un riquadro attorno a una stella o a una piccola luna: la sua forma diventa la PSF").clicked() {
            actions.push(PanelAction::Pick(PickTarget::Psf));
        }
        let automatic = p.psf == auto.psf && p.algorithm == auto.algorithm && p.iterations == auto.iterations && p.auto_stop == auto.auto_stop;
        if !automatic && ui.add(egui::Button::new((ICON_RESTART_ALT.rich_text(), "Valori automatici"))).on_hover_text("PSF, algoritmo e iterazioni come li sceglie il programma per questa immagine").clicked() {
            let measured = None;
            *p = DeconvParams { measured, ..auto.clone() };
        }
    });
    if let Some(note) = probe {
        hint(ui, note);
    }
    advanced(ui, "deconv-advanced", |ui| {
        deconvolution_advanced(ui, p, auto, optics, preview, report);
        why_box(ui, why);
    });
}

fn psf_name(psf: &Psf) -> &'static str {
    match psf {
        Psf::Gaussian { .. } => "PSF gaussiana",
        Psf::Moffat { .. } => "PSF di Moffat",
        Psf::Airy { .. } => "PSF di Airy",
        Psf::Measured { .. } => "PSF misurata",
    }
}

/// What a PSF shape is, in a sentence.
fn psf_shape_note(psf: &Psf) -> &'static str {
    match psf {
        Psf::Gaussian { .. } => "Gaussiana, una campana senza ali. Va bene con seeing calmo.",
        Psf::Moffat { .. } => "Moffat, nucleo stretto e ali larghe come il seeing reale degli stack. È la scelta automatica.",
        Psf::Airy { .. } => "Airy, il disco di diffrazione del telescopio con i suoi anelli. Ha senso solo con seeing eccellente.",
        Psf::Measured { .. } => "Misurata, la forma vera presa da una stella o da una luna dell'immagine.",
    }
}

fn deconvolution_advanced(ui: &mut Ui, p: &mut DeconvParams, auto: &DeconvParams, optics: Option<ps_core::equipment::Optics>, preview: &mut PsfPreview, report: Option<DeconvReport>) {
    field_label(ui, "Algoritmo");
    segmented(ui, &mut p.algorithm, &[(DeconvAlgorithm::RichardsonLucy, None, "Richardson–Lucy"), (DeconvAlgorithm::Wiener, None, "Wiener")]);
    hint(
        ui,
        match p.algorithm {
            DeconvAlgorithm::RichardsonLucy => {
                "Consigliato per pianeti, Luna e Sole. Ricostruisce l'immagine a passi e si ferma quando ha recuperato il dettaglio che sta sopra il rumore; conserva la luminosità e non crea valori negativi."
            }
            DeconvAlgorithm::Wiener => {
                "Un solo passo, istantaneo. Utile per provare velocemente una PSF o su immagini molto pulite; con il rumore crea più facilmente grana e aloni e non sa quando fermarsi."
            }
        },
    );

    field_label(ui, "Forma della PSF");
    let mut kind = match p.psf {
        Psf::Gaussian { .. } => 0,
        Psf::Moffat { .. } => 1,
        Psf::Airy { .. } => 2,
        Psf::Measured { .. } => 3,
    };
    let mut options = vec![(0, None, "Gaussiana"), (1, None, "Moffat"), (2, None, "Airy")];
    if p.measured.is_some() {
        options.push((3, None, "Misurata"));
    }
    if segmented(ui, &mut kind, &options) {
        p.psf = match (kind, &p.measured) {
            (3, Some(k)) => Psf::Measured { fwhm: ps_core::psf::kernel_fwhm(k) },
            _ => convert_psf(p.psf, kind, optics),
        };
    }
    hint(ui, psf_shape_note(&p.psf));
    ui.add_space(theme::space::S);
    let kernel = p.kernel(1.0);
    ui.horizontal(|ui| {
        preview.show(ui, &kernel, 76.0);
        ui.vertical(|ui| {
            ui.label(RichText::new("FWHM").size(theme::size::SMALL).color(TEXT_WEAK));
            ui.label(RichText::new(format!("{:.2} px", p.psf.fwhm_px())).size(17.0).color(TEXT));
            ui.label(RichText::new(format!("kernel {0}×{0}", kernel.size())).size(theme::size::SMALL).color(TEXT_WEAK));
        });
    });
    match &mut p.psf {
        Psf::Gaussian { .. } | Psf::Measured { .. } => {}
        Psf::Moffat { beta, .. } => {
            let default = match auto.psf {
                Psf::Moffat { beta, .. } => beta,
                _ => 2.0,
            };
            ParamSlider::new("Ali (β)", beta, 1.5..=10.0, default).help("Basso = ali larghe, seeing turbolento; alto = quasi una gaussiana").show(ui);
        }
        Psf::Airy { aperture_mm, wavelength_nm, arcsec_per_px, obstruction } => {
            if let Some(o) = optics
                && ui.small_button("Dallo strumento").on_hover_text("Apertura, filtro, scala e ostruzione indicati nella scheda Strumento").clicked()
                && let Psf::Airy { aperture_mm: a, wavelength_nm: l, arcsec_per_px: s, obstruction: ob } = o.psf
            {
                (*aperture_mm, *wavelength_nm, *arcsec_per_px, *obstruction) = (a, l, s, ob);
            }
            ParamSlider::new("Apertura", aperture_mm, 50.0..=1000.0, 235.0).decimals(0).suffix(" mm").show(ui);
            ParamSlider::new("Lunghezza d'onda", wavelength_nm, 350.0..=1000.0, 550.0).decimals(0).suffix(" nm").show(ui);
            ParamSlider::new("Campionamento", arcsec_per_px, 0.02..=1.0, 0.1).decimals(3).logarithmic().suffix(" ″/px").show(ui);
            ParamSlider::new("Ostruzione", obstruction, 0.0..=0.6, 0.33).help("Diametro dell'ostruzione centrale / apertura").show(ui);
        }
    }

    match p.algorithm {
        DeconvAlgorithm::RichardsonLucy => {
            field_label(ui, "Iterazioni");
            ui.horizontal(|ui| {
                toggle(ui, &mut p.auto_stop);
                ui.label("Fermati da sola").on_hover_text("Si ferma quando l'immagine è ricostruita fino al livello del rumore: oltre si amplificherebbe solo rumore");
            });
            let label = if p.auto_stop { "Al massimo" } else { "Iterazioni" };
            ParamSlider::new(label, &mut p.iterations, 1..=150, auto.iterations).decimals(0).suffix(" it").help("Con l'arresto automatico è solo il limite; senza, è il numero di passi").show(ui);
            if let Some(r) = report.filter(|r| r.iterations > 0) {
                let text = if p.auto_stop && r.iterations < p.iterations {
                    format!("Fermata dopo {} iterazioni: oltre avrebbe amplificato il rumore.", r.iterations)
                } else if p.auto_stop {
                    format!("Usate tutte le {} iterazioni: si può alzare il massimo.", r.iterations)
                } else {
                    format!("{} iterazioni.", r.iterations)
                };
                hint(ui, &text);
            }
            ParamSlider::new("Anti-rumore", &mut p.damping, 0.0..=5.0, auto.damping).suffix(" σ").help("Smette di \"inseguire\" variazioni piccole come il rumore (damped RL)").show(ui);
            ParamSlider::new("Anti-ringing", &mut p.deringing, 0.0..=1.0, auto.deringing)
                .help("Durante le iterazioni, sul bordo e nel cielo (l'area del Dering), la stima non può uscire dall'intervallo dell'immagine osservata: gli anelli non si formano. 0 = spento")
                .show(ui);
            ui.add_space(theme::space::S);
            ui.horizontal(|ui| {
                toggle(ui, &mut p.accelerate);
                ui.label("Accelerazione Biggs–Andrews").on_hover_text("Stessa qualità con molte meno iterazioni");
            });
            ui.add_space(theme::space::S);
            ParamSlider::new("Regolarizzazione TV", &mut p.tv_lambda, 0.0..=0.02, auto.tv_lambda)
                .decimals(4)
                .help("Total Variation (Dey 2006): riduce ringing e rumore sui bordi. Tipico 0.001–0.003")
                .show(ui);
        }
        DeconvAlgorithm::Wiener => {
            field_label(ui, "Rumore");
            ui.horizontal(|ui| {
                toggle(ui, &mut p.auto_k);
                ui.label("Rapporto rumore automatico").on_hover_text("Calcolato dal rumore misurato sull'immagine");
            });
            ui.add_enabled_ui(!p.auto_k, |ui| {
                ParamSlider::new("Rapporto rumore", &mut p.wiener_k, 0.0001..=0.5, 0.01).decimals(4).logarithmic().help("Più alto = più morbido e meno rumore").show(ui);
            });
        }
    }
    color_mode(ui, &mut p.color);
    if p.color == ColorMode::Rgb && !matches!(p.psf, Psf::Measured { .. }) {
        field_label(ui, "Sfocatura per canale");
        hint(ui, "Quanto è più larga la PSF di ogni canale (il seeing sfoca di più il blu).");
        for (k, name) in ["Rosso", "Verde", "Blu"].into_iter().enumerate() {
            ParamSlider::new(name, &mut p.channel_scale[k], 0.5..=2.0, auto.channel_scale[k]).suffix("×").show(ui);
        }
    }
}

/// Switch PSF family keeping the FWHM (Airy from the equipment when known, else a typical
/// telescope, then widened or narrowed to the same FWHM).
pub fn convert_psf(psf: Psf, kind: u8, optics: Option<ps_core::equipment::Optics>) -> Psf {
    let fwhm = psf.fwhm_px();
    match kind {
        0 => Psf::Gaussian { sigma: (fwhm / 2.3548).max(0.3) },
        1 => Psf::Moffat { fwhm: fwhm.max(0.5), beta: if let Psf::Moffat { beta, .. } = psf { beta } else { 2.0 } },
        _ => optics.map_or(Psf::Airy { aperture_mm: 235.0, wavelength_nm: 550.0, arcsec_per_px: 0.1, obstruction: 0.33 }, |o| o.psf).with_fwhm(fwhm),
    }
}

/// Texture cache for the PSF thumbnail.
#[derive(Default)]
pub struct PsfPreview {
    kernel: Option<Kernel2D>,
    texture: Option<TextureHandle>,
}

impl PsfPreview {
    fn show(&mut self, ui: &mut Ui, k: &Kernel2D, side: f32) {
        if self.kernel.as_ref() != Some(k) || self.texture.is_none() {
            let size = k.size();
            let max = k.data.iter().copied().fold(f32::MIN, f32::max).max(1e-12);
            let [r, g, b, _] = ACCENT.to_array();
            let pixels = k
                .data
                .iter()
                .map(|&v| {
                    // Square root lifts the faint wings and Airy rings into view.
                    let t = (v / max).max(0.0).sqrt();
                    Color32::from_rgb((r as f32 * t) as u8, (g as f32 * t) as u8, (b as f32 * t) as u8)
                })
                .collect();
            let image = egui::ColorImage::new([size, size], pixels);
            self.texture = Some(ui.ctx().load_texture("psf", image, TextureOptions::NEAREST));
            self.kernel = Some(k.clone());
        }
        let texture = self.texture.as_ref().unwrap();
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(side), Sense::hover());
        ui.painter().rect_filled(rect, CornerRadius::same(6), Color32::BLACK);
        // Show at most ±12 px around the centre so small PSFs stay legible.
        let size = texture.size()[0] as f32;
        let visible = size.min(25.0) / size;
        let uv = egui::Rect::from_center_size(egui::pos2(0.5, 0.5), Vec2::splat(visible));
        ui.painter().image(texture.id(), rect.shrink(4.0), uv, Color32::WHITE);
    }
}

// ---- smart sharpening ---------------------------------------------------------------------

/// Colour of the finest detail present, over the periods `range` spans: from white-yellow
/// (finest) through orange and red to violet (coarsest). `None` where there is no detail.
pub fn resolution_colour(period: f32, range: (f32, f32)) -> Option<Color32> {
    if period <= 0.0 {
        return None;
    }
    let (lo, hi) = (range.0.max(0.1).ln(), range.1.max(range.0 * 1.01).ln());
    let t = ((period.ln() - lo) / (hi - lo)).clamp(0.0, 1.0);
    let stops = [(255.0, 250.0, 200.0), (250.0, 190.0, 70.0), (225.0, 90.0, 60.0), (150.0, 60.0, 170.0), (60.0, 70.0, 160.0)];
    let x = t * (stops.len() - 1) as f32;
    let (i, f) = ((x.floor() as usize).min(stops.len() - 2), x - x.floor());
    let (a, b) = (stops[i], stops[i + 1]);
    let mix = |p: f32, q: f32| (p + (q - p) * f) as u8;
    Some(Color32::from_rgb(mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2)))
}

/// Set while the control choosing what the smart sharpening shows over the image is on screen.
pub const SMART_ZONES_SWITCH_SHOWN: &str = "smart-zones-switch-shown";

/// `view`: what is shown over the image; `legend`: the periods (px) the resolution colours span.
pub fn smart_body(ui: &mut Ui, p: &mut SmartParams, auto: &SmartParams, view: &mut SmartView, legend: Option<(f32, f32)>) {
    let mut strength = p.strength * 100.0;
    if ParamSlider::new("Nitidezza", &mut strength, 0.0..=ps_core::smart::MAX_STRENGTH * 100.0, 0.0)
        .decimals(0)
        .suffix(" %")
        .help("Quanto rinforzare il dettaglio vero. 100 % = quanto fa di solito BlurXTerminator; oltre si spinge di più. Più è alta, più pulisce anche la grana oltre il dettaglio più fine.")
        .show(ui)
    {
        p.strength = strength / 100.0;
    }
    let mut peak = p.peak * 100.0;
    if ParamSlider::new("Dove spingere", &mut peak, 60.0..=140.0, auto.peak * 100.0)
        .decimals(0)
        .suffix(" %")
        .help("Su quale dimensione di dettaglio cade la spinta massima, rispetto al dettaglio più fine misurato: più alto = più fine (più nitido ma più rumore), più basso = strutture più grandi")
        .show(ui)
    {
        p.peak = peak / 100.0;
    }
    ui.add_space(theme::space::S);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Curva").size(theme::size::BODY));
        egui::ComboBox::from_id_salt("smart-curve").width(170.0).selected_text(RichText::new(p.curve.label()).size(theme::size::BODY)).show_ui(ui, |ui| {
            for curve in SmartCurve::ALL {
                let text = (RichText::new(curve.label()).size(theme::size::BODY), RichText::new(if curve == SmartCurve::Standard { "  predefinita" } else { "" }).size(theme::size::SMALL).color(TEXT_WEAK));
                if ui.add(egui::Button::selectable(curve == p.curve, text)).on_hover_text(curve.description()).clicked() {
                    p.curve = curve;
                }
            }
        });
    });
    // The whole explanation (a hint would show only its first sentence).
    ui.add(egui::Label::new(RichText::new(p.curve.description()).size(theme::size::SMALL).color(TEXT_WEAK)).wrap());
    ui.add_space(theme::space::S);
    let mut leave_sky = !p.include_sky;
    ui.horizontal(|ui| {
        toggle(ui, &mut leave_sky);
        ui.label("Escludi il cielo").on_hover_text("Rende nitidi solo pianeta, anelli e lune. Il cielo contiene solo rumore, e la nitidezza lì farebbe solo grana e aloni accanto al bordo");
    });
    p.include_sky = !leave_sky;
    hint(ui, if leave_sky { "Il cielo resta com'è, perché lì c'è solo rumore." } else { "La nitidezza agisce anche sul cielo e sulle zone scure." });
    ui.add_space(theme::space::S);
    let period = if p.cutoff > 0.0 { format!("{:.1} px", 1.0 / p.cutoff) } else { "quello misurato sull'immagine".to_string() };
    hint(ui, &format!("Agisce sull'immagine già elaborata (dopo deconvoluzione e wavelet), aggiungendo solo la nitidezza che manca. Dettaglio più fine misurato: {period}."));
    ui.add_space(theme::space::S);
    response_graph(ui, p);
    ui.add_space(theme::space::S);
    field_label(ui, "Mostra sull'immagine");
    ui.ctx().data_mut(|d| d.insert_temp(egui::Id::unique(SMART_ZONES_SWITCH_SHOWN), true));
    segmented(ui, view, &[(SmartView::None, None, "Niente"), (SmartView::HoldBack, None, "Dove frena"), (SmartView::Resolution, None, "Risoluzione")]);
    match *view {
        SmartView::None => {}
        SmartView::HoldBack => {
            ui.horizontal_wrapped(|ui| {
                let mut items = vec![(ZONES_COLOUR, "Segui le zone: solo rumore"), (EDGE_COLOUR, "Protezione del bordo")];
                if !p.include_sky {
                    items.push((SKY_COLOUR, "Cielo escluso"));
                }
                for (colour, text) in items {
                    swatch(ui, Color32::from_rgb(colour[0], colour[1], colour[2]));
                    ui.label(RichText::new(text).size(theme::size::SMALL).color(TEXT_WEAK));
                }
            });
            hint(ui, "Più il colore è intenso, più la nitidezza è ridotta lì. Dove l'immagine resta com'è, la nitidezza agisce piena.");
        }
        SmartView::Resolution => match legend {
            Some(range) => {
                resolution_legend(ui, range);
                hint(ui, "La dimensione del dettaglio presente in ogni zona del soggetto: più chiaro = più fine. Grigio scuro: niente sopra le zone più lisce. Cielo e bordo del disco non sono colorati.");
            }
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new("Calcolo della mappa…").size(theme::size::SMALL).color(TEXT_WEAK));
                });
            }
        },
    }
    advanced(ui, "smart-advanced", |ui| {
        ParamSlider::new("Protezione del bordo", &mut p.limb_protection, 0.0..=1.0, auto.limb_protection)
            .help("Meno nitidezza dove il soggetto confina con il cielo (bordo del disco, anelli), dentro e subito fuori dal bordo, dove si formano gli aloni. Il resto del disco non cambia")
            .show(ui);
        ui.horizontal(|ui| {
            toggle(ui, &mut p.adaptive);
            ui.label("Segui le zone").on_hover_text("Rinforza ogni scala solo dove contiene dettaglio vero e non rumore. Spento = la stessa nitidezza ovunque");
        });
        ui.horizontal(|ui| {
            toggle(ui, &mut p.protect_highlights);
            ui.label("Proteggi i bianchi").on_hover_text("Le zone più luminose si avvicinano al bianco senza mai bruciarsi, così si può spingere la nitidezza del resto");
        });
        color_mode(ui, &mut p.color);
    });
}

fn swatch(ui: &mut Ui, colour: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(12.0, 12.0), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, colour);
}

/// The resolution map's colour bar, with the periods (px) at a few points.
fn resolution_legend(ui: &mut Ui, range: (f32, f32)) {
    let width = ui.available_width() - 4.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 30.0), Sense::hover());
    let painter = ui.painter_at(rect);
    let bar = egui::Rect::from_min_size(rect.min, Vec2::new(width, 12.0));
    const STEPS: usize = 48;
    for k in 0..STEPS {
        let t = (k as f32 + 0.5) / STEPS as f32;
        let period = range.0 * (range.1 / range.0).powf(t);
        let x0 = bar.left() + k as f32 / STEPS as f32 * bar.width();
        let cell = egui::Rect::from_min_max(egui::pos2(x0, bar.top()), egui::pos2(x0 + bar.width() / STEPS as f32 + 0.5, bar.bottom()));
        painter.rect_filled(cell, 0.0, resolution_colour(period, range).unwrap_or(Color32::BLACK));
    }
    for k in 0..5 {
        let t = k as f32 / 4.0;
        let period = range.0 * (range.1 / range.0).powf(t);
        let x = bar.left() + t * bar.width();
        let align = match k {
            0 => egui::Align2::LEFT_TOP,
            4 => egui::Align2::RIGHT_TOP,
            _ => egui::Align2::CENTER_TOP,
        };
        let text = if period < 10.0 { format!("{period:.1} px") } else { format!("{period:.0} px") };
        painter.text(egui::pos2(x, bar.bottom() + 3.0), align, text, egui::FontId::proportional(theme::size::SMALL), TEXT_WEAK);
    }
}

/// Colours of the map of where the smart sharpening holds back.
pub const ZONES_COLOUR: [u8; 3] = [70, 150, 255];
pub const EDGE_COLOUR: [u8; 3] = [255, 150, 40];
pub const SKY_COLOUR: [u8; 3] = [120, 120, 135];

/// The target response: how much each size of detail is amplified (1 = unchanged).
fn response_graph(ui: &mut Ui, p: &SmartParams) {
    let cutoff = if p.cutoff > 0.0 { p.cutoff } else { 0.2 };
    let width = ui.available_width() - 4.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 90.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 6.0, BG);
    let plot = rect.shrink2(Vec2::new(8.0, 12.0));
    let (lo, hi) = (0.02f32.ln(), 0.5f32.ln());
    let x = |f: f32| plot.left() + (f.ln() - lo) / (hi - lo) * plot.width();
    let peak = (0..=80).map(|i| p.target(0.02 * 25f32.powf(i as f32 / 80.0), cutoff)).fold(1.0f32, f32::max);
    let top = (peak * 1.15).max(4.0);
    let y = |g: f32| plot.bottom() - g.clamp(0.0, top) / top * plot.height();
    painter.line_segment([egui::pos2(plot.left(), y(1.0)), egui::pos2(plot.right(), y(1.0))], Stroke::new(1.0, BORDER));
    for px in [32.0f32, 16.0, 8.0, 4.0, 2.0] {
        painter.text(egui::pos2(x(1.0 / px), rect.bottom() - 1.0), egui::Align2::CENTER_BOTTOM, format!("{px:.0} px"), egui::FontId::proportional(9.0), TEXT_WEAK);
    }
    let points: Vec<egui::Pos2> = (0..=80).map(|i| 0.02 * 25f32.powf(i as f32 / 80.0)).map(|f| egui::pos2(x(f), y(p.target(f, cutoff)))).collect();
    painter.add(egui::Shape::line(points, Stroke::new(1.8, ACCENT)));
    let xc = x(cutoff);
    painter.line_segment([egui::pos2(xc, plot.top()), egui::pos2(xc, plot.bottom())], Stroke::new(1.0, TEXT_WEAK));
    painter.text(egui::pos2(xc + 3.0, plot.top()), egui::Align2::LEFT_TOP, "limite del dettaglio", egui::FontId::proportional(9.0), TEXT_WEAK);
    painter.text(egui::pos2(plot.left(), plot.top()), egui::Align2::LEFT_TOP, format!("picco ×{peak:.1}"), egui::FontId::proportional(9.0), ACCENT);
}

// ---- wavelets ----------------------------------------------------------------------------

fn layer_name(j: usize) -> &'static str {
    match j {
        0 => "Dettagli finissimi",
        1 => "Dettagli fini",
        2 => "Dettagli medi",
        3 => "Strutture medie",
        4 => "Strutture grandi",
        _ => "Strutture molto grandi",
    }
}

/// `why`: show the advanced section, with the reasons for the automatic choices (`None` for
/// layers, which share the global transform settings).
pub fn wavelet_body(ui: &mut Ui, p: &mut WaveletParams, why: Option<&[String]>) {
    ui.horizontal(|ui| {
        hint(ui, "Nitidezza e rumore per ogni scala, dalla più fine");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if icon_button(ui, ICON_RESTART_ALT, "Azzera tutti i livelli").clicked() {
                p.layers.iter_mut().for_each(|l| *l = WaveletLayer::default());
                (p.amplify, p.grain_denoise) = (1.0, 0.0);
            }
        });
    });
    ui.add_space(theme::space::XS);
    ParamSlider::new("Moltiplicatore", &mut p.amplify, 1.0..=5.0, 1.0)
        .suffix("×")
        .help("Moltiplica la nitidezza di tutti i livelli: quando un cursore a fondo scala non basta")
        .show(ui);
    ParamSlider::new("Sopra il rumore", &mut p.noise_adaptive, 0.0..=5.0, 0.0)
        .suffix(" σ")
        .help("Nitidezza adattiva: un dettaglio riceve tutta la nitidezza solo se supera il rumore di questo numero di σ, meno se è più debole. Si può spingere di più senza far crescere la grana. 0 = spento")
        .show(ui);
    ParamSlider::new("Protezione bordi", &mut p.edge_protection, 0.0..=1.0, 0.0)
        .help("Meno nitidezza sui bordi molto forti (lembo, anelli), dove si formano gli aloni. 0 = spenta, 1 = nessuna nitidezza in più sui bordi più forti")
        .show(ui);
    if why.is_some() {
        field_label(ui, "Scale");
        let mut free = p.scale_mode == ScaleMode::Free;
        // The app re-derives the transform (and carries the gains over) when this changes.
        if segmented(ui, &mut free, &[(true, None, "Libere"), (false, None, "Classiche 1·2·4·8")]) {
            p.scale_mode = if free { ScaleMode::Free } else { ScaleMode::Dyadic };
        }
        hint(
            ui,
            if free {
                "Il primo livello parte da dove finisce il dettaglio vero dell'immagine, misurato sul suo spettro. Quello che è più fine è solo grana: non viene mai reso più nitido."
            } else {
                "Le scale classiche: ogni livello raddoppia partendo da un passo intero (1 o 2 px). Il primo livello spesso contiene quasi solo rumore e il secondo è già un po' grosso."
            },
        );
    }
    ui.add_space(theme::space::XS);
    if p.scale_mode == ScaleMode::Free {
        grain_row(ui, p.first_scale / 2.0, &mut p.grain_denoise);
    }
    for j in 0..p.layers.len() {
        let scale = p.scale(j);
        layer_row(ui, j, scale, &mut p.layers[j]);
    }
    if let Some(why) = why {
        advanced(ui, "wavelet-advanced", |ui| {
            wavelet_advanced(ui, p);
            why_box(ui, why);
        });
    }
}

/// What is finer than the first free-scale layer: only ever cleaned.
fn grain_row(ui: &mut Ui, below: f32, denoise: &mut f32) {
    let active = *denoise != 0.0;
    egui::Frame::new()
        .fill(BG)
        .corner_radius(CornerRadius::same(7))
        .inner_margin(egui::Margin::symmetric(10, 8))
        .stroke(Stroke::new(1.0, if active { ACCENT_DIM } else { Color32::TRANSPARENT }))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(RichText::new("Grana").strong().color(if active { TEXT } else { TEXT_WEAK }));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(format!("sotto {} px", scale_label(below))).size(theme::size::SMALL).color(TEXT_WEAK));
                });
            });
            ParamSlider::new("Rumore", denoise, 0.0..=5.0, 0.0)
                .suffix(" σ")
                .help("Più fine del primo livello c'è solo rumore: qui si può solo ripulire. Elimina le variazioni più piccole di N volte il rumore stimato")
                .show(ui);
        });
    ui.add_space(theme::space::S);
}

/// "1", "1.5", "3.2": sizes in pixels without needless decimals.
fn scale_label(px: f32) -> String {
    if (px - px.round()).abs() < 0.05 { format!("{px:.0}") } else { format!("{px:.1}") }
}

fn layer_row(ui: &mut Ui, j: usize, scale: f32, layer: &mut WaveletLayer) {
    let active = layer.gain != 0.0 || layer.denoise != 0.0;
    ui.push_id(j, |ui| {
        egui::Frame::new()
            .fill(BG)
            .corner_radius(CornerRadius::same(7))
            .inner_margin(egui::Margin::symmetric(10, 8))
            .stroke(Stroke::new(1.0, if active { ACCENT_DIM } else { Color32::TRANSPARENT }))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(20.0), Sense::hover());
                    ui.painter().circle_filled(rect.center(), 10.0, if active { ACCENT } else { SURFACE_HOVER });
                    let number_color = if active { Color32::BLACK } else { TEXT };
                    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, format!("{}", j + 1), egui::FontId::proportional(11.0), number_color);
                    ui.label(RichText::new(layer_name(j)).strong().color(if active { TEXT } else { TEXT_WEAK }));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if active && icon_button(ui, ICON_RESTART_ALT, "Azzera livello").clicked() {
                            *layer = WaveletLayer::default();
                        }
                        ui.label(RichText::new(format!("{} px", scale_label(scale))).size(theme::size::SMALL).color(TEXT_WEAK));
                    });
                });
                // Linear: a logarithmic scale spent three quarters of the track below a gain of 1,
                // where nothing visible happens.
                ParamSlider::new("Nitidezza", &mut layer.gain, -1.0..=8.0, 0.0).help("0 = invariato, 1 = dettaglio raddoppiato, −1 = livello rimosso").show(ui);
                ParamSlider::new("Rumore", &mut layer.denoise, 0.0..=5.0, 0.0)
                    .suffix(" σ")
                    .help("Elimina le variazioni più piccole di N volte il rumore stimato su questo livello")
                    .show(ui);
            });
    });
    ui.add_space(theme::space::S);
}

fn wavelet_advanced(ui: &mut Ui, p: &mut WaveletParams) {
    let mut levels = p.layers.len();
    if ParamSlider::new("Numero di livelli", &mut levels, 1..=MAX_LAYERS, 6).decimals(0).show(ui) {
        p.layers.resize(levels, WaveletLayer::default());
    }
    if p.scale_mode == ScaleMode::Free {
        ParamSlider::new("Primo livello", &mut p.first_scale, 1.0..=4.0, 1.5)
            .suffix(" px")
            .help("Dimensione del livello più fine; ogni livello successivo la raddoppia. Scelta in automatico dallo spettro dell'immagine")
            .show(ui);
        p.first_scale = (p.first_scale * 20.0).round() / 20.0;
    } else {
        classic_scales(ui, p);
    }
    field_label(ui, "Soglia del rumore");
    hint(ui, "Morbida riduce tutti i coefficienti. Netta azzera solo quelli sotto soglia: più dettaglio, più artefatti.");
    segmented(ui, &mut p.threshold, &[(ThresholdMode::Soft, None, "Morbida"), (ThresholdMode::Hard, None, "Netta")]);
    color_mode(ui, &mut p.color);
}

fn classic_scales(ui: &mut Ui, p: &mut WaveletParams) {
    field_label(ui, "Kernel");
    let mut kernel = match p.kernel {
        WaveletKernel::B3Spline => 0,
        WaveletKernel::Linear => 1,
        WaveletKernel::Gaussian { .. } => 2,
    };
    hint(ui, "Il filtro tra una scala e l'altra. Decide quanto sono \"morbidi\" i livelli.");
    if segmented(ui, &mut kernel, &[(0, None, "B3-spline"), (1, None, "Lineare"), (2, None, "Gaussiano")]) {
        p.kernel = match kernel {
            0 => WaveletKernel::B3Spline,
            1 => WaveletKernel::Linear,
            _ => WaveletKernel::Gaussian { sigma: 1.0 },
        };
    }
    if let WaveletKernel::Gaussian { sigma } = &mut p.kernel {
        ParamSlider::new("Sigma kernel", sigma, 0.3..=4.0, 1.0).show(ui);
    }
    field_label(ui, "Progressione delle scale");
    hint(ui, "Come cresce la dimensione dei dettagli da un livello al successivo.");
    segmented(ui, &mut p.scale_mode, &[(ScaleMode::Dyadic, None, "Diadica 1·2·4·8"), (ScaleMode::Linear, None, "Lineare 1·2·3·4")]);
    ui.add_space(theme::space::S);
    ParamSlider::new("Passo iniziale", &mut p.base_step, 1..=4, 1).decimals(0).suffix(" px").help("Distanza tra i tap del primo livello").show(ui);
}

// ---- colour ------------------------------------------------------------------------------

pub fn color_body(ui: &mut Ui, p: &mut ColorParams, mono: bool) {
    if mono {
        hint(ui, "Disponibile solo per immagini a colori.");
        return;
    }
    ParamSlider::new("Saturazione", &mut p.saturation, 0.0..=3.0, 1.0).help("1 = invariata, 0 = bianco e nero").show(ui);
    ParamSlider::new("Vividezza", &mut p.vibrance, -1.0..=1.0, 0.0).help("Satura di più i colori tenui e meno quelli già vivaci").show(ui);
    ui.add_space(theme::space::XS);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Bilanciamento").size(theme::size::SMALL).strong().color(TEXT_WEAK));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if icon_button(ui, ICON_RESTART_ALT, "Azzera").clicked() {
                p.balance = [1.0; 3];
            }

        });
    });
    for (i, name) in ["Rosso", "Verde", "Blu"].into_iter().enumerate() {
        ParamSlider::new(name, &mut p.balance[i], 0.5..=2.0, 1.0).decimals(3).show(ui);
    }
}

// ---- layers ------------------------------------------------------------------------------

/// What dragging on the image does while a layer is selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerTool {
    None,
    /// Move/resize the hand-drawn shape.
    Shape,
    Brush,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushSettings {
    /// Radius in image pixels.
    pub radius: f32,
    pub hardness: f32,
    pub erase: bool,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self { radius: 20.0, hardness: 0.5, erase: false }
    }
}

/// UI state of the layers card that lives in the app.
pub struct LayerUi<'a> {
    pub selected: &'a mut Option<u64>,
    pub show_mask: &'a mut bool,
    pub tool: &'a mut LayerTool,
    pub brush: &'a mut BrushSettings,
    pub collapsed: &'a mut std::collections::HashSet<String>,
    pub kind: ImageKind,
    /// Offer only the regions that make sense for the image.
    pub adapt: bool,
    pub mono: bool,
    /// Opening a layer section closes the others.
    pub accordion: bool,
}

/// Sections of a layer editor, by key.
const LAYER_SECTIONS: [&str; 6] = ["mask", "wavelet", "contrast", "noise", "tone", "color"];

pub fn layers_body(ui: &mut Ui, layers: &mut Vec<Layer>, mut state: LayerUi, actions: &mut Vec<PanelAction>) {
    hint(ui, "Elabora una parte dell'immagine con impostazioni proprie.");
    ui.add_space(theme::space::S);
    let mut delete = None;
    for layer in layers.iter_mut().filter(|l| l.role.is_none()) {
        let selected = *state.selected == Some(layer.id);
        ui.push_id(layer.id, |ui| {
            egui::Frame::new()
                .fill(if selected { SURFACE_HOVER } else { BG })
                .corner_radius(CornerRadius::same(7))
                .stroke(Stroke::new(1.0, if selected { ACCENT } else { Color32::TRANSPARENT }))
                .inner_margin(egui::Margin::symmetric(8, 6))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        toggle(ui, &mut layer.enabled);
                        let label = egui::Button::new((RichText::new(&layer.name).color(TEXT), RichText::new(format!("  {}", layer.mask.shape.label())).size(theme::size::SMALL).color(TEXT_WEAK))).frame(false);
                        if ui.add(label.truncate()).clicked() {
                            *state.selected = if selected { None } else { Some(layer.id) };
                            *state.tool = LayerTool::None;
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if icon_button(ui, ICON_DELETE, "Elimina livello").clicked() {
                                delete = Some(layer.id);
                            }
                        });
                    });
                });
            if selected {
                ui.add_space(theme::space::S);
                layer_editor(ui, layer, &mut state, actions);
            }
        });
        ui.add_space(theme::space::S);
    }
    if let Some(id) = delete {
        layers.retain(|l| l.id != id);
        if *state.selected == Some(id) {
            *state.selected = None;
            *state.tool = LayerTool::None;
        }
    }

    ui.menu_button((ICON_ADD.rich_text().color(ACCENT), RichText::new("Aggiungi livello").color(ACCENT)), |ui| {
        let mut entry = |ui: &mut Ui, icon: MaterialIcon, text: &str, t: LayerTemplate| {
            if ui.button((icon.rich_text(), text)).clicked() {
                actions.push(PanelAction::AddLayer(t));
            }
        };
        let saturn = state.kind == ImageKind::Saturn || !state.adapt;
        let planet = matches!(state.kind, ImageKind::Planet | ImageKind::Saturn) || !state.adapt;
        let disk = state.kind != ImageKind::Surface || !state.adapt;
        if saturn {
            entry(ui, ICON_PLANET, "Globo di Saturno", LayerTemplate::Auto(AutoRegion::SaturnGlobe));
            entry(ui, ICON_ORBIT, "Anelli di Saturno", LayerTemplate::Auto(AutoRegion::SaturnRings));
        }
        if disk {
            let subject = if planet { "Pianeta" } else { "Disco" };
            entry(ui, ICON_AUTO_AWESOME, subject, LayerTemplate::Auto(AutoRegion::Subject));
        }
        if planet {
            entry(ui, ICON_BUBBLE_CHART, "Lune", LayerTemplate::Auto(AutoRegion::Moons));
        }
        if disk {
            entry(ui, ICON_GRADIENT, "Sfondo", LayerTemplate::Auto(AutoRegion::Background));
        }
        ui.separator();
        entry(ui, ICON_BRUSH, "Pennello (da dipingere)", LayerTemplate::Brush);
        entry(ui, ICON_PANORAMA_FISH_EYE, "Ellisse", LayerTemplate::Ellipse);
        entry(ui, ICON_CROP_SQUARE, "Rettangolo", LayerTemplate::Rectangle);
    });
}

/// A layer section, collapsible like the main cards; its fold state is remembered per section.
#[allow(clippy::too_many_arguments)]
fn layer_section(
    ui: &mut Ui,
    folded: &mut std::collections::HashSet<String>,
    accordion: bool,
    key: &str,
    icon: MaterialIcon,
    title: &str,
    own: Option<&mut bool>,
    body: impl FnOnce(&mut Ui),
) {
    let id = format!("layer:{key}");
    let mut collapsed = folded.contains(&id);
    let overrides = own.as_ref().is_none_or(|o| **o);
    card(ui, icon, title, true, own, overrides, &mut collapsed, StageStatus::default(), None, |ui| {
        if overrides {
            body(ui);
        }
    });
    if !overrides && !collapsed {
        // Not overridden: the body is hidden; say where the settings come from.
        ui.label(RichText::new("    come nell'immagine intera").size(theme::size::SMALL).color(TEXT_WEAK));
    }
    if collapsed {
        folded.insert(id);
    } else if folded.remove(&id) && accordion {
        // Just opened: fold the other sections.
        folded.extend(LAYER_SECTIONS.iter().filter(|k| **k != key).map(|k| format!("layer:{k}")));
    }
}

/// Body of the "Anelli di Saturno" / "Lune" cards: the dedicated layer's sections.
pub fn role_body(ui: &mut Ui, layer: Option<&mut Layer>, role: LayerRole, mut state: LayerUi, actions: &mut Vec<PanelAction>) {
    let Some(layer) = layer else {
        hint(
            ui,
            match role {
                LayerRole::Rings => "Elabora gli anelli a parte dal globo. La maschera si crea da sola: clicca il pallino in alto per iniziare.",
                LayerRole::Moons => "Elabora a parte le lune, di solito più deboli del pianeta. La maschera si crea da sola: clicca il pallino in alto per iniziare.",
            },
        );
        return;
    };
    layer_editor(ui, layer, &mut state, actions);
}

fn layer_editor(ui: &mut Ui, layer: &mut Layer, state: &mut LayerUi, actions: &mut Vec<PanelAction>) {
    egui::Frame::new().inner_margin(egui::Margin { left: 6, right: 0, top: 0, bottom: 0 }).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Nome").color(TEXT_WEAK));
            ui.add(egui::TextEdit::singleline(&mut layer.name).desired_width(ui.available_width() - 4.0));
        });
        ui.add_space(theme::space::S);

        // Ordered by importance: what the region is, then detail, contrast, noise, tone, colour.
        let LayerUi { selected, show_mask, tool, brush, collapsed, mono, accordion, .. } = state;
        let (mono, id, acc) = (*mono, layer.id, *accordion);
        layer_section(ui, collapsed, acc, "mask", ICON_GRADIENT, "Maschera", None, |ui| mask_editor(ui, &mut layer.mask, id, selected, show_mask, tool, brush));
        layer_section(ui, collapsed, acc, "wavelet", ICON_WAVES, "Wavelet", Some(&mut layer.wavelet.enabled), |ui| {
            wavelet_body(ui, &mut layer.wavelet.params, None)
        });
        layer_section(ui, collapsed, acc, "contrast", ICON_CONTRAST, "Contrasto", Some(&mut layer.contrast.enabled), |ui| contrast_body(ui, &mut layer.contrast.params));
        layer_section(ui, collapsed, acc, "noise", ICON_GRAIN, "Rumore", Some(&mut layer.denoise.enabled), |ui| denoise_body(ui, &mut layer.denoise.params, mono, None));
        layer_section(ui, collapsed, acc, "tone", ICON_SHOW_CHART, "Curve e livelli", Some(&mut layer.tone.enabled), |ui| {
            tone_body(ui, &mut layer.tone.params, ToneParts::Both, Some(id), None, &mut HistogramMode::default(), mono, actions)
        });
        layer_section(ui, collapsed, acc, "color", ICON_PALETTE, "Colore", Some(&mut layer.color.enabled), |ui| color_body(ui, &mut layer.color.params, mono));
    });
}

/// Mask tools. The overlay and the drag tool belong to the selected layer: using them here
/// selects this one.
#[allow(clippy::too_many_arguments)]
fn mask_editor(
    ui: &mut Ui,
    mask: &mut ps_core::mask::MaskSpec,
    id: u64,
    selected: &mut Option<u64>,
    show_mask: &mut bool,
    tool: &mut LayerTool,
    brush: &mut BrushSettings,
) {
    let manual = matches!(mask.shape, MaskShape::Ellipse { .. } | MaskShape::Rectangle { .. });
    mask_tools(ui, id, manual, &mut mask.strokes, selected, show_mask, tool, brush);
    ui.add_space(theme::space::S);
    if let MaskShape::Ellipse { angle, .. } = &mut mask.shape {
        ParamSlider::new("Rotazione", angle, -90.0..=90.0, 0.0).decimals(0).suffix("°").show(ui);
    }
    if mask.shape != MaskShape::Empty {
        ParamSlider::new("Sfumatura", &mut mask.feather, 0.0..=40.0, 4.0).decimals(1).suffix(" px").help("Morbidezza del bordo della forma").show(ui);
    }
    ui.horizontal(|ui| {
        toggle(ui, &mut mask.invert);
        ui.label("Inverti");
    });
}

/// Temporary memory flag: the selected mask's tools were on screen this frame.
pub const MASK_TOOLS_SHOWN: &str = "mask-tools-shown";

/// Show-on-image switch, what dragging on the image does (pan, move the shape, paint) and the
/// brush and its strokes, for the mask with id `id`. Using them selects that mask.
#[allow(clippy::too_many_arguments)]
fn mask_tools(
    ui: &mut Ui,
    id: u64,
    has_shape: bool,
    strokes: &mut Vec<ps_core::mask::Stroke>,
    selected: &mut Option<u64>,
    show_mask: &mut bool,
    tool: &mut LayerTool,
    brush: &mut BrushSettings,
) {
    let is_selected = *selected == Some(id);
    if is_selected {
        ui.ctx().data_mut(|d| d.insert_temp(egui::Id::unique(MASK_TOOLS_SHOWN), true));
    }
    let mut show = is_selected && *show_mask;
    let mut current_tool = if is_selected { *tool } else { LayerTool::None };
    ui.horizontal(|ui| {
        toggle(ui, &mut show);
        ui.label("Mostra sull'immagine");
    });
    ui.add_space(theme::space::XS);
    let mut options = vec![(LayerTool::None, Some(ICON_PAN_TOOL), "Sposta vista")];
    if has_shape {
        options.push((LayerTool::Shape, Some(ICON_OPEN_WITH), "Forma"));
    }
    options.push((LayerTool::Brush, Some(ICON_BRUSH), "Pennello"));
    segmented(ui, &mut current_tool, &options);
    if show != (is_selected && *show_mask) || current_tool != (if is_selected { *tool } else { LayerTool::None }) {
        (*selected, *show_mask, *tool) = (Some(id), show, current_tool);
    }

    if current_tool == LayerTool::Brush {
        ui.add_space(theme::space::S);
        segmented(ui, &mut brush.erase, &[(false, Some(ICON_BRUSH), "Aggiungi"), (true, Some(ICON_INK_ERASER), "Togli")]);
        ParamSlider::new("Dimensione", &mut brush.radius, 1.0..=300.0, 20.0).decimals(0).logarithmic().suffix(" px").help("Tasti [ e ] per cambiarla").show(ui);
        ParamSlider::new("Durezza", &mut brush.hardness, 0.0..=1.0, 0.5).show(ui);
        hint(ui, &format!("Tieni premuto {} per togliere al volo.", theme::alt_key()));
    }
    if !strokes.is_empty() {
        ui.horizontal(|ui| {
            hint(ui, &format!("{} pennellate", strokes.len()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(egui::Button::new((ICON_DELETE.rich_text(), "Cancella")).frame(false)).clicked() {
                    strokes.clear();
                }
                if ui.add(egui::Button::new((ICON_UNDO.rich_text(), "Ultima")).frame(false)).on_hover_text("Togli l'ultima pennellata").clicked() {
                    strokes.pop();
                }
            });
        });
    }
}

// ---- noise, contrast, tone -----------------------------------------------------------------

pub fn denoise_body(ui: &mut Ui, p: &mut NoiseParams, mono: bool, why: Option<&[String]>) {
    segmented(ui, &mut p.method, &[(NoiseMethod::Planetary, Some(ICON_PUBLIC), "Pianeti"), (NoiseMethod::Lunar, Some(ICON_NIGHTLIGHT), "Luna e Sole")]);
    hint(
        ui,
        match p.method {
            NoiseMethod::Planetary => "Wavelet multiscala: toglie la grana fine senza intaccare bande e dettagli morbidi.",
            NoiseMethod::Lunar => "Filtro che rispetta i bordi: pulisce le zone piatte e lascia netti crateri e granulazione.",
        },
    );
    ParamSlider::new("Intensità", &mut p.strength, 0.0..=3.0, 0.0).help("In multipli del rumore misurato sull'immagine (1 = standard)").show(ui);
    if p.method == NoiseMethod::Lunar {
        ParamSlider::new("Dettaglio", &mut p.detail, 0.0..=1.0, 0.3).help("Quanta trama fine restituire dopo la pulizia").show(ui);
    }
    if !mono {
        ParamSlider::new("Rumore colore", &mut p.chroma, 0..=4, 0).decimals(0).help("Pulisce le macchie di colore senza toccare la luminosità").show(ui);
    }
    ParamSlider::new("Cielo", &mut p.sky, 0.0..=1.0, 0.0).help("Pulizia extra dello sfondo nero intorno al soggetto").show(ui);
    if let Some(why) = why {
        advanced(ui, "denoise-why", |ui| why_box(ui, why));
    }
}

pub fn contrast_body(ui: &mut Ui, p: &mut ContrastParams) {
    ui.label(RichText::new("Maschera di contrasto").size(theme::size::SMALL).strong().color(TEXT_WEAK));
    ParamSlider::new("Intensità", &mut p.amount, 0.0..=3.0, 0.0).show(ui);
    ParamSlider::new("Raggio", &mut p.radius, 0.3..=10.0, 1.5).suffix(" px").show(ui);
    ParamSlider::new("Soglia", &mut p.threshold, 0.0..=0.05, 0.0).decimals(3).help("Le differenze più piccole non vengono accentuate (protegge dal rumore)").show(ui);
    ui.add_space(theme::space::XS);
    ui.label(RichText::new("Contrasto locale").size(theme::size::SMALL).strong().color(TEXT_WEAK));
    ParamSlider::new("Intensità", &mut p.local_amount, 0.0..=1.0, 0.0).show(ui);
    ParamSlider::new("Raggio", &mut p.local_radius, 5.0..=200.0, 40.0).decimals(0).logarithmic().suffix(" px").show(ui);
}

/// Temporary memory flag: the switch of the ringing map was on screen this frame.
pub const RINGING_SWITCH_SHOWN: &str = "ringing-switch-shown";

/// `saturn`: offer the globe/rings choice. `state`: the mask tools' shared state.
/// `show_ringing`: draw where sharpening left rings.
pub fn dering_body(ui: &mut Ui, p: &mut DeringParams, why: &[String], saturn: bool, show_ringing: &mut bool, state: LayerUi) {
    hint(ui, "Toglie le bande chiare e scure che la nitidezza crea accanto ai bordi netti, e gli anelli concentrici dentro il disco.");
    ParamSlider::new("Intensità", &mut p.strength, 0.0..=1.0, 0.0).help("0 = spento, 1 = correzione piena. Tipico 0,8").show(ui);
    ui.ctx().data_mut(|d| d.insert_temp(egui::Id::unique(RINGING_SWITCH_SHOWN), true));
    ui.horizontal(|ui| {
        toggle(ui, show_ringing);
        ui.label("Mostra gli aloni").on_hover_text("In magenta, dove la nitidezza ha spinto i valori oltre quelli vicini dell'originale: così vedi se e dove serve");
    });

    let section = |ui: &mut Ui, title: &str, text: &str| {
        ui.add_space(theme::space::M);
        ui.label(RichText::new(title).size(theme::size::SMALL).strong().color(TEXT_WEAK));
        hint(ui, text);
    };
    section(ui, "ALONI SUI BORDI NETTI", "Accanto al bordo del pianeta e agli anelli: i valori vengono tenuti vicini a quelli dell'originale intorno. Le bande e i dettagli fini non vengono toccati.");
    ParamSlider::new("Larghezza", &mut p.radius, 1..=16, 4)
        .decimals(0)
        .suffix(" px")
        .help("Quanto si allontanano dal bordo le bande da togliere. Se resta una banda scura staccata dal bordo, aumentala")
        .show(ui);
    ParamSlider::new("Contrasto lasciato", &mut p.tolerance, 0.0..=0.5, 0.1)
        .help("Quanto il risultato può andare oltre i valori originali vicini: più alto conserva più contrasto, più basso toglie più alone")
        .show(ui);
    ParamSlider::new("Bordi trattati", &mut p.edges, 0.02..=0.4, 0.1)
        .help("Quanto deve essere netto un bordo per essere trattato: più basso agisce anche su bordi deboli")
        .show(ui);
    if saturn {
        segmented(ui, &mut p.region, &[(DeringRegion::Globe, Some(ICON_PUBLIC), "Globo"), (DeringRegion::Rings, Some(ICON_ORBIT), "Anelli"), (DeringRegion::Subject, None, "Tutto")]);
        hint(
            ui,
            match p.region {
                DeringRegion::Globe => "Sul globo e sul suo bordo, compresi gli anelli che gli passano davanti. I bordi esterni degli anelli e la divisione di Cassini restano nitidi.",
                DeringRegion::Rings => "Solo sugli anelli fuori dal globo.",
                DeringRegion::Subject => "Su tutti i bordi netti di Saturno.",
            },
        );
    }
    let LayerUi { selected, show_mask, tool, brush, .. } = state;
    mask_tools(ui, ps_core::pipeline::DERING_MASK, false, &mut p.strokes, selected, show_mask, tool, brush);

    if p.region != DeringRegion::Rings {
        section(ui, "ANELLI DENTRO IL DISCO", "Bande concentriche al bordo del pianeta, anche solo su un lato: un orlo chiaro lungo il bordo e la banda più scura dietro.");
        ParamSlider::new("Orlo chiaro", &mut p.rind, 0.0..=1.0, 0.5)
            .help("Quanto si abbassa l'orlo chiaro lungo il bordo, che fa sembrare scura la zona subito dietro. 1 = alla luminosità più interna")
            .show(ui);
        ParamSlider::new("Profondità", &mut p.depth, 0.0..=0.9, 0.35)
            .help("Fin dove, dal bordo verso il centro, si cercano le bande: frazione del raggio. 0 = spento")
            .show(ui);
    }
    advanced(ui, "dering-advanced", |ui| why_box(ui, why));
}

/// Which parts of a tone step an editor shows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToneParts {
    Levels,
    Curve,
    /// Both, as in a layer's section.
    Both,
}

/// `histogram`: of what the current step works on; `mode`: how the large histogram is drawn
/// (shared by every tone card).
#[allow(clippy::too_many_arguments)]
pub fn tone_body(ui: &mut Ui, p: &mut ToneParams, parts: ToneParts, layer: Option<u64>, histogram: Option<&Histogram>, mode: &mut HistogramMode, mono: bool, actions: &mut Vec<PanelAction>) {
    let (levels, curve) = (parts != ToneParts::Curve, parts != ToneParts::Levels);
    // Steps already applied, newest last; each can be taken back.
    if !p.applied.is_empty() {
        ui.horizontal(|ui| {
            ui.label(ICON_CHECK_CIRCLE.rich_text().size(14.0).color(theme::SUCCESS));
            let n = p.applied.len();
            ui.label(RichText::new(if n == 1 { "1 regolazione applicata".to_string() } else { format!("{n} regolazioni applicate") }).size(theme::size::SMALL).color(TEXT));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(egui::Button::new((ICON_DELETE.rich_text(), "Tutte")).frame(false)).on_hover_text("Togli tutte le regolazioni applicate").clicked() {
                    p.applied.clear();
                }
                if ui.add(egui::Button::new((ICON_UNDO.rich_text(), "Indietro")).frame(false)).on_hover_text("Riprendi l'ultima regolazione applicata per modificarla").clicked()
                    && let Some(last) = p.applied.pop()
                {
                    p.current = last;
                }
            });
        });
        ui.add_space(theme::space::XS);
    }
    // Which channel the levels and curve edit: all of them, or red, green or blue.
    let channel_id = ui.scope_id().with("tone-channel");
    let mut channel: Option<usize> = ui.data(|d| d.get_temp(channel_id)).unwrap_or(None);
    if mono {
        channel = None;
    }
    ui.horizontal(|ui| {
        if !mono {
            segmented(ui, &mut channel, &[(None, None, "RGB"), (Some(0), None, "R"), (Some(1), None, "G"), (Some(2), None, "B")]);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if histogram.is_some() && !mono && channel.is_none() {
                segmented(ui, mode, &[(HistogramMode::Luminance, None, "B/N"), (HistogramMode::Rgb, None, "Colori")]);
            }
        });
    });
    ui.add_space(theme::space::XS);

    // One histogram, as the image will look with the current adjustment (the input's counts
    // redistributed through it as you drag, the input itself faint behind): above the levels,
    // or behind the curve.
    let step = p.current.clone();
    let after = histogram.map(|h| {
        h.mapped(|c, v| {
            let mut s = step.clone();
            if !curve {
                s.curve = ps_core::enhance::straight();
                s.channel_curves = std::array::from_fn(|_| ps_core::enhance::straight());
            }
            s.apply_value(c, v)
        })
    });
    if levels {
        if let (Some(h), Some(after)) = (histogram, &after) {
            after.show_large(ui, *mode, Some(h), channel);
            ui.add_space(theme::space::S);
        }
        levels_editor(ui, p, channel, layer, actions);
    }
    if curve {
        ui.add_space(theme::space::S);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Curva").size(theme::size::SMALL).strong().color(TEXT_WEAK));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let target = match channel {
                    Some(c) => &mut p.current.channel_curves[c],
                    None => &mut p.current.curve,
                };
                if icon_button(ui, ICON_RESTART_ALT, "Rendi lineare questa curva").clicked() {
                    *target = ps_core::enhance::straight();
                }
            });
        });
        let (points, colour) = match channel {
            Some(c) => (&mut p.current.channel_curves[c], [Color32::from_rgb(235, 100, 100), Color32::from_rgb(110, 210, 125), Color32::from_rgb(110, 150, 245)][c]),
            None => (&mut p.current.curve, ACCENT),
        };
        // With the levels above, their histogram is enough.
        let backdrop = match (histogram, &after) {
            (Some(before), Some(now)) if !levels => Some(Backdrop { now, before, mode: *mode, channel }),
            _ => None,
        };
        curve_editor(ui, points, backdrop, colour);
        hint(ui, "Clic per aggiungere un punto, trascina per spostarlo, doppio clic per toglierlo.");
    }
    ui.data_mut(|d| d.insert_temp(channel_id, channel));
    ui.add_space(theme::space::S);
    let can_apply = !p.current.is_neutral();
    let apply = egui::Button::new((ICON_CHECK.rich_text(), "Applica e continua")).min_size(Vec2::new(ui.available_width(), 26.0));
    if ui.add_enabled(can_apply, apply).on_hover_text("Fissa questa regolazione e iniziane un'altra sopra").clicked() {
        p.commit();
    }
}

/// Black and white points (with pickers on the whole image) and midtones, of every channel
/// or of one (`channel`).
fn levels_editor(ui: &mut Ui, p: &mut ToneParams, channel: Option<usize>, layer: Option<u64>, actions: &mut Vec<PanelAction>) {
    if let Some(c) = channel {
        let [black, white, gamma] = &mut p.current.channel_levels[c];
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Livelli · {}", ["rosso", "verde", "blu"][c])).size(theme::size::SMALL).strong().color(TEXT_WEAK));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if icon_button(ui, ICON_RESTART_ALT, "Azzera i livelli di questo canale").clicked() {
                    (*black, *white, *gamma) = (0.0, 1.0, 1.0);
                }
            });
        });
        ParamSlider::new("Nero", black, 0.0..=0.5, 0.0).decimals(3).show(ui);
        ParamSlider::new("Bianco", white, 0.5..=1.0, 1.0).decimals(3).show(ui);
        ParamSlider::new("Toni medi", gamma, 0.3..=3.0, 1.0).help("Gamma del canale: più alto lo schiarisce").show(ui);
        return;
    }
    ui.horizontal(|ui| {
        ui.label(RichText::new("Livelli").size(theme::size::SMALL).strong().color(TEXT_WEAK));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if icon_button(ui, ICON_RESTART_ALT, "Azzera nero, bianco e toni medi").clicked() {
                (p.current.black, p.current.white, p.current.gamma) = (0.0, 1.0, 1.0);
            }
            // The whole image has the button in the card header.
            if layer.is_some() && ui.button((ICON_AUTO_FIX_HIGH.rich_text(), "Automatico")).on_hover_text("Nero sul cielo, bianco sulle luci più forti").clicked() {
                actions.push(PanelAction::AutoLevels(layer));
            }
        });
    });
    let picker = |ui: &mut Ui, target: PickTarget, actions: &mut Vec<PanelAction>| {
        let tip = match target {
            PickTarget::Black => "Clicca sull'immagine un punto che deve diventare nero",
            PickTarget::White => "Clicca sull'immagine un punto che deve diventare bianco",
            PickTarget::Psf => "Disegna un riquadro attorno a una stella o a una piccola luna",
        };
        if layer.is_none() && icon_button(ui, ICON_COLORIZE, tip).clicked() {
            actions.push(PanelAction::Pick(target));
        }
    };
    ui.horizontal(|ui| {
        picker(ui, PickTarget::Black, actions);
        ParamSlider::new("Nero", &mut p.current.black, 0.0..=0.5, 0.0).decimals(3).show(ui);
    });
    ui.horizontal(|ui| {
        picker(ui, PickTarget::White, actions);
        ParamSlider::new("Bianco", &mut p.current.white, 0.5..=1.0, 1.0).decimals(3).show(ui);
    });
    ParamSlider::new("Toni medi", &mut p.current.gamma, 0.3..=3.0, 1.0).help("Gamma: più alto schiarisce i toni medi").show(ui);
}

// ---- export ------------------------------------------------------------------------------

/// What the exported file is called: an editable base name, keywords added after it, the
/// user's suffix, and the resulting name (`stem`, without extension).
pub struct ExportName<'a> {
    pub base: &'a mut String,
    pub keywords: &'a mut Vec<NameKeyword>,
    pub suffix: &'a mut String,
    pub stem: &'a str,
}

fn keyword_help(k: NameKeyword) -> &'static str {
    match k {
        NameKeyword::Object => "Luna, Saturno, Giove…: dal nome del file di ripresa, o dal tipo d'immagine",
        NameKeyword::Date => "Data di ripresa dal nome del file (SharpCap, FireCapture), o del file",
        NameKeyword::Time => "Ora di ripresa (hhmm), allo stesso modo",
        NameKeyword::Modules => "Le elaborazioni che cambiano l'immagine: deconv, wav, dering, livelli…",
        NameKeyword::Suffix => "Un testo fisso, per riconoscere i file elaborati",
    }
}

/// The "Esporta" tab: format, name and place, then the export buttons and the processing
/// settings' file. `source`: the original file; `last`: the last file exported.
pub fn export_body(ui: &mut Ui, format: &mut String, name: &mut ExportName, source: &std::path::Path, mono: bool, last: Option<&std::path::Path>, actions: &mut Vec<PanelAction>) {
    let section = |ui: &mut Ui, title: &str| {
        ui.add_space(theme::space::M);
        ui.label(RichText::new(title).size(theme::size::SMALL).strong().color(TEXT_WEAK));
        ui.add_space(theme::space::XS);
    };
    egui::Frame::new().fill(theme::SURFACE).corner_radius(CornerRadius::same(8)).inner_margin(egui::Margin::same(12)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        section(ui, "FORMATO");
        let mut picked = match format.as_str() {
            "png" => "png",
            "jpg" | "jpeg" => "jpg",
            _ => "tif",
        };
        if segmented(ui, &mut picked, &[("tif", None, "TIFF 16 bit"), ("png", None, "PNG 16 bit"), ("jpg", None, "JPEG")]) {
            *format = picked.to_string();
        }
        hint(
            ui,
            match format.as_str() {
                "jpg" => "8 bit, compresso: per condividere. Perde dettaglio nelle ombre.",
                "png" => "16 bit senza perdite. Va bene per il web e per altri programmi.",
                _ => "16 bit senza perdite. Il formato da usare per ulteriori elaborazioni.",
            },
        );
        section(ui, "NOME");
        let original = source.file_stem().and_then(|s| s.to_str()).unwrap_or("immagine");
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(name.base).desired_width(ui.available_width() - 30.0).hint_text(original));
            if icon_button(ui, ICON_RESTART_ALT, "Torna al nome dell'originale").clicked() {
                *name.base = original.to_string();
            }
        });
        ui.add_space(theme::space::XS);
        ui.label(RichText::new("Aggiungi al nome").size(theme::size::SMALL).color(TEXT_WEAK));
        ui.horizontal_wrapped(|ui| {
            for k in NameKeyword::ALL {
                let mut on = name.keywords.contains(&k);
                if ui.toggle_value(&mut on, k.label()).on_hover_text(keyword_help(k)).changed() {
                    if on {
                        name.keywords.push(k);
                    } else {
                        name.keywords.retain(|x| *x != k);
                    }
                }
            }
        });
        if name.keywords.contains(&NameKeyword::Suffix) {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Suffisso").size(theme::size::SMALL).color(TEXT_WEAK));
                ui.add(egui::TextEdit::singleline(name.suffix).desired_width(90.0));
            });
        }
        ui.add_space(theme::space::XS);
        // The resulting name, wrapped rather than widening the panel.
        ui.add(egui::Label::new(RichText::new(format!("→ {}.{format}", name.stem)).size(theme::size::BODY).color(TEXT).monospace()).wrap());
        hint(ui, &format!("Nella cartella dell'originale, {}. Se il file esiste, si aggiunge un numero: niente viene sovrascritto.", source.parent().map(|p| p.display().to_string()).unwrap_or_default()));
        if mono && *format == "jpg" {
            hint(ui, "L'immagine è in bianco e nero: il JPEG sarà in scala di grigi.");
        }
        ui.add_space(theme::space::L);
        let black = Color32::BLACK;
        let export = egui::Button::new((ICON_IOS_SHARE.rich_text().color(black), RichText::new("Esporta").color(black).strong()))
            .fill(ACCENT)
            .min_size(Vec2::new(ui.available_width(), 32.0))
            .corner_radius(CornerRadius::same(7));
        if ui.add(export).on_hover_text("⌘⇧E / Ctrl+Shift+E").clicked() {
            actions.push(PanelAction::Export);
        }
        ui.add_space(theme::space::S);
        if ui.add(egui::Button::new((ICON_SAVE_AS.rich_text(), "Esporta come…")).min_size(Vec2::new(ui.available_width(), 28.0))).on_hover_text("Scegli cartella e nome (⌘E / Ctrl+E)").clicked() {
            actions.push(PanelAction::ExportAs);
        }
        if let Some(path) = last {
            ui.add_space(theme::space::S);
            ui.horizontal_wrapped(|ui| {
                ui.label(ICON_CHECK_CIRCLE.rich_text().size(14.0).color(theme::SUCCESS));
                ui.label(RichText::new(format!("Ultima esportazione: {}", path.display())).size(theme::size::SMALL).color(TEXT_WEAK));
            });
        }
    });
    ui.add_space(theme::space::L);
    egui::Frame::new().fill(theme::SURFACE).corner_radius(CornerRadius::same(8)).inner_margin(egui::Margin::same(12)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("IMPOSTAZIONI DI ELABORAZIONE").size(theme::size::SMALL).strong().color(TEXT_WEAK));
        hint(ui, "Salva tutte le regolazioni in un file per riusarle su altre immagini.");
        ui.horizontal(|ui| {
            if ui.button((ICON_SAVE.rich_text(), "Salva…")).clicked() {
                actions.push(PanelAction::SavePreset);
            }
            if ui.button((ICON_UPLOAD_FILE.rich_text(), "Carica…")).clicked() {
                actions.push(PanelAction::LoadPreset);
            }
        });
    });
}

// ---- Mineral Moon ------------------------------------------------------------------------

pub fn mineral_body(ui: &mut Ui, p: &mut MineralParams, mono: bool, luminance: Option<&str>, actions: &mut Vec<PanelAction>) {
    if mono {
        hint(ui, "Mineral Moon richiede un'immagine a colori; qui si applica solo la curva.");
    } else {
        hint(ui, "Esalta i colori dei minerali lunari. Il bilanciamento è nella scheda Colore.");
        ParamSlider::new("Saturazione", &mut p.saturation, 1.0..=12.0, 1.0).decimals(1).suffix("×").show(ui);
        ParamSlider::new("Pulizia colore", &mut p.chroma_smoothing, 0..=6, 0)
            .decimals(0)
            .help("Riduce il rumore cromatico, che la saturazione forte rende evidente (scale wavelet sfumate)")
            .show(ui);
        ParamSlider::new("Tonalità", &mut p.hue, -180.0..=180.0, 0.0).decimals(0).suffix("°").show(ui);
    }
    ui.add_space(theme::space::S);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Curva").size(theme::size::SMALL).strong().color(TEXT_WEAK));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if icon_button(ui, ICON_RESTART_ALT, "Curva lineare").clicked() {
                p.curve = MineralParams::default().curve;
            }
        });
    });
    curve_editor(ui, &mut p.curve, None, ACCENT);
    hint(ui, "Clic per aggiungere un punto, trascina per spostarlo, doppio clic per toglierlo.");

    if !mono {
        ui.add_space(theme::space::M);
        ui.label(RichText::new("LUMINANZA (LRGB)").size(theme::size::SMALL).strong().color(TEXT_WEAK));
        match luminance {
            Some(name) => {
                ui.horizontal(|ui| {
                    ui.label(ICON_IMAGE.rich_text().color(ACCENT));
                    ui.add(egui::Label::new(RichText::new(name).size(theme::size::SMALL)).truncate());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if icon_button(ui, ICON_CLOSE, "Rimuovi luminanza").clicked() {
                            actions.push(PanelAction::ClearLuminance);
                        }
                    });
                });
                ParamSlider::new("Intensità", &mut p.lrgb_amount, 0.0..=1.0, 1.0).show(ui);
            }
            None => {
                hint(ui, "Combina il colore con un'immagine mono più nitida della stessa ripresa.");
                if ui.button((ICON_ADD_PHOTO_ALTERNATE.rich_text(), "Carica immagine mono…")).clicked() {
                    actions.push(PanelAction::LoadLuminance);
                }
            }
        }
    }
}
