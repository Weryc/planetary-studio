//! User preferences (persisted between sessions) and the Settings window (⌘, / Ctrl+,).

use ps_core::i18n::{Lang, tl};
use ps_core::tr;
use crate::theme::{self, ACCENT, BORDER, SUCCESS, TEXT, TEXT_WEAK};
use crate::widgets::{caps, note, segmented, toggle};
use eframe::egui::{self, RichText, Stroke};
use egui_material_icons::MaterialIcon;
use egui_material_icons::icons::*;
use ps_core::analysis::ImageKind;
use ps_core::pipeline::{Pipeline, Stage};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    /// Shown at the top of the panel.
    Favorite,
    Visible,
    /// Not shown, and not applied.
    Hidden,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModulePref {
    pub stage: Stage,
    pub visibility: Visibility,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Preferences {
    /// In processing order.
    pub modules: Vec<ModulePref>,
    /// What the open picture shows, as far as it is known (set from the document, not saved): the
    /// cards that exist for one subject only (Jupiter's photometric colour) appear when it matches.
    #[serde(skip)]
    pub subject: Option<ps_core::capture::Body>,
    pub bxt_path: Option<PathBuf>,
    /// Also apply the automatic sharpening and noise settings when opening a file (the
    /// structural ones are always automatic).
    pub auto_settings: bool,
    /// Choose the structural settings (PSF, wavelet levels, noise method…) from the image
    /// when opening it; off = generic starting values.
    #[serde(default = "yes")]
    pub auto_structure: bool,
    /// Show only the modules that make sense for the image (no colour tools for mono images,
    /// Mineral Moon only for the Moon…).
    #[serde(default = "yes")]
    pub adapt_to_image: bool,
    /// Opening a card closes the others of its area.
    #[serde(default = "yes")]
    pub one_group_open: bool,
    /// Open module cards (all closed by default).
    pub expanded: HashSet<Stage>,
    /// The area shown in the panel, by name.
    pub current_area: String,
    /// File format of quick exports (extension), the last one chosen.
    pub export_format: String,
    /// Added to the original's name when exporting next to it.
    pub export_suffix: String,
    /// The equipment last used (or read from a log), for the next images.
    #[serde(default)]
    pub setup: ps_core::equipment::Setup,
    /// The last few equipment setups used, newest first, offered in the top bar.
    #[serde(default)]
    pub recent_setups: Vec<ps_core::equipment::Setup>,
    /// Keywords added to exported files' names.
    #[serde(default = "default_keywords")]
    pub export_keywords: Vec<crate::naming::NameKeyword>,
    /// How the levels and curves histograms are drawn.
    pub histogram_mode: crate::widgets::HistogramMode,
    /// Folded thematic groups and layer sections, by name.
    pub collapsed_sections: HashSet<String>,
    /// Keep only the expensive processing steps in memory (off: keep as much as possible).
    pub memory_saving: bool,
    /// While a slider is dragged, preview at reduced resolution (approximate, but fluid).
    pub fast_preview: bool,
    /// Free or classic wavelet scales, as last chosen in the wavelet panel.
    #[serde(default = "free_scales")]
    pub wavelet_scales: ps_core::wavelet::ScaleMode,
    /// How hard the automatic settings push.
    pub auto_strength: ps_core::preset::AutoStrength,
    /// Use the graphics card for the heavy processing when there is a suitable one.
    #[serde(default = "yes")]
    pub use_gpu: bool,
    /// Enlarging (or reducing) the exported image.
    #[serde(default)]
    pub resize: crate::resize_ui::ResizeSettings,
    /// Open the Identifica e annota tab when an image is opened (off: only on request).
    pub identify_tab_on_open: bool,
    /// What the Identifica e annota tab prints onto the exported image.
    pub identify: crate::annotate::LabelOptions,
    /// Interface language (the first run follows the system's).
    pub language: ps_core::i18n::Lang,
    /// Format version, to migrate settings saved by older releases. Field-level default (0):
    /// the container-level default would fill in the *current* version and skip migrations.
    #[serde(default)]
    pub version: u32,
}

const VERSION: u32 = 4;

/// English unless the system is set to Italian (read once: it may spawn a process).
fn system_language() -> ps_core::i18n::Lang {
    use ps_core::i18n::Lang;
    // Tests look for the Italian labels.
    if cfg!(test) {
        return Lang::It;
    }
    static LANG: std::sync::OnceLock<Lang> = std::sync::OnceLock::new();
    *LANG.get_or_init(|| {
        let from_env = ["LC_ALL", "LC_MESSAGES", "LANGUAGE", "LANG"].iter().find_map(|v| std::env::var(v).ok().filter(|s| !s.is_empty() && s != "C" && s != "POSIX"));
        // A GUI app on macOS has no LANG: ask the system for the preferred languages.
        let preferred = from_env.or_else(|| {
            let out = std::process::Command::new("defaults").args(["read", "-g", "AppleLanguages"]).output().ok()?;
            out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
        });
        match preferred {
            Some(s) if s.to_lowercase().trim_start_matches(['(', ' ', '\n', '"']).starts_with("en") => Lang::En,
            Some(s) if s.to_lowercase().trim_start_matches(['(', ' ', '\n', '"']).starts_with("it") => Lang::It,
            Some(_) => Lang::En,
            None => Lang::It,
        }
    })
}

fn yes() -> bool {
    true
}

/// Thematic areas of the side panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Group {
    Favorites,
    Sharpness,
    /// Fixing what the processing left behind (rings, halos).
    Corrections,
    Noise,
    Tone,
    Color,
    Regions,
    /// Not a group of modules: what the image shows (moons, craters), from the ephemerides.
    Identify,
    /// Not a group of modules: the export tab.
    Export,
}

impl Group {
    pub fn name(self) -> &'static str {
        match self {
            Group::Favorites => "Preferiti",
            Group::Sharpness => "Nitidezza",
            Group::Corrections => "Correzione artefatti",
            Group::Noise => "Rumore",
            Group::Tone => "Curve e livelli",
            Group::Color => "Colore",
            Group::Regions => "Maschere",
            Group::Identify => "Identifica e annota",
            Group::Export => "Esporta",
        }
    }

    /// The name shown on the tab (the Italian `name` stays the stable key saved in preferences).
    pub fn label(self) -> &'static str {
        match self {
            Group::Favorites => tl("Preferiti"),
            Group::Sharpness => tl("Nitidezza"),
            Group::Corrections => tl("Correzione artefatti"),
            Group::Noise => tl("Rumore"),
            Group::Tone => tl("Curve e livelli"),
            Group::Color => tl("Colore"),
            Group::Regions => tl("Maschere"),
            Group::Identify => tl("Identifica e annota"),
            Group::Export => tl("Esporta"),
        }
    }

    pub fn icon(self) -> MaterialIcon {
        match self {
            Group::Favorites => ICON_STAR,
            Group::Sharpness => ICON_DEBLUR,
            Group::Corrections => ICON_HEALING,
            Group::Noise => ICON_GRAIN,
            Group::Tone => ICON_SHOW_CHART,
            Group::Color => ICON_PALETTE,
            Group::Regions => ICON_LAYERS,
            Group::Identify => ICON_TRAVEL_EXPLORE,
            Group::Export => ICON_IOS_SHARE,
        }
    }

    fn of(stage: Stage) -> Group {
        match stage {
            Stage::Bxt | Stage::Deconv | Stage::Wavelet | Stage::Smart | Stage::Contrast => Group::Sharpness,
            Stage::Dering => Group::Corrections,
            Stage::Denoise => Group::Noise,
            Stage::Tone | Stage::Curves => Group::Tone,
            Stage::Align | Stage::PhotoColor | Stage::Color | Stage::Mineral | Stage::SunColor => Group::Color,
            Stage::Rings | Stage::Moons | Stage::Layers => Group::Regions,
        }
    }
}

/// Processing order: sharpen, clean, then tone and colour, then local work.
pub const GROUP_ORDER: [Group; 6] = [Group::Sharpness, Group::Corrections, Group::Noise, Group::Tone, Group::Color, Group::Regions];

/// Whether a module makes sense for this image.
pub fn relevant(stage: Stage, kind: ImageKind, mono: bool) -> bool {
    match stage {
        Stage::Align | Stage::Color => !mono,
        // Jupiter's bands: needs the subject to be known as Jupiter too (see `Preferences::subject`).
        Stage::PhotoColor => !mono && kind == ImageKind::Planet,
        Stage::Mineral => !mono && matches!(kind, ImageKind::FullDisk | ImageKind::Surface),
        // Solar frames are full disks or close-ups.
        Stage::SunColor => matches!(kind, ImageKind::FullDisk | ImageKind::Surface),
        Stage::Rings => kind == ImageKind::Saturn,
        Stage::Moons => matches!(kind, ImageKind::Planet | ImageKind::Saturn),
        _ => true,
    }
}

impl Default for Preferences {
    fn default() -> Self {
        use Visibility::*;
        let modules = [
            (Stage::Align, Visible),
            (Stage::Bxt, Hidden),
            (Stage::Deconv, Visible),
            (Stage::Denoise, Visible),
            (Stage::Wavelet, Visible),
            (Stage::Contrast, Visible),
            (Stage::Smart, Visible),
            (Stage::Dering, Visible),
            (Stage::Tone, Visible),
            (Stage::Curves, Visible),
            (Stage::PhotoColor, Visible),
            (Stage::Color, Visible),
            (Stage::Layers, Visible),
            (Stage::Mineral, Visible),
            (Stage::SunColor, Visible),
        ]
        .into_iter()
        .map(|(stage, visibility)| ModulePref { stage, visibility })
        .collect();
        Self {
            modules,
            subject: None,
            bxt_path: ps_core::bxt::find_executable(),
            auto_settings: false,
            auto_structure: true,
            adapt_to_image: true,
            one_group_open: true,
            expanded: HashSet::new(),
            current_area: Group::Sharpness.name().to_string(),
            export_format: "tif".into(),
            export_suffix: "_ps".into(),
            export_keywords: default_keywords(),
            setup: Default::default(),
            recent_setups: Vec::new(),
            histogram_mode: Default::default(),
            collapsed_sections: HashSet::new(),
            memory_saving: false,
            fast_preview: false,
            auto_strength: Default::default(),
            wavelet_scales: free_scales(),
            use_gpu: true,
            resize: Default::default(),
            identify_tab_on_open: false,
            identify: Default::default(),
            language: system_language(),
            version: VERSION,
        }
    }
}

pub fn stage_icon(stage: Stage) -> MaterialIcon {
    match stage {
        Stage::Align => ICON_CENTER_FOCUS_STRONG,
        Stage::Bxt => ICON_AUTO_AWESOME,
        Stage::Deconv => ICON_DEBLUR,
        Stage::Denoise => ICON_GRAIN,
        Stage::Wavelet => ICON_WAVES,
        Stage::Smart => ICON_MAGIC_BUTTON,
        Stage::Contrast => ICON_CONTRAST,
        Stage::Dering => ICON_FLARE,
        Stage::Tone => ICON_TUNE,
        Stage::Curves => ICON_SHOW_CHART,
        Stage::PhotoColor => ICON_COLORIZE,
        Stage::Color => ICON_PALETTE,
        Stage::Layers => ICON_LAYERS,
        Stage::Mineral => ICON_NIGHTLIGHT,
        Stage::SunColor => ICON_WB_SUNNY,
        Stage::Rings => ICON_ORBIT,
        Stage::Moons => ICON_BUBBLE_CHART,
    }
}

impl Preferences {
    /// Preferences saved by an older version lack newer modules: add them, in processing order.
    pub fn upgraded(mut self) -> Self {
        if self.version < 2 {
            // Version 1 applied every automatic setting on open; now that is opt-in.
            self.auto_settings = false;
            self.adapt_to_image = true;
        }
        if self.version < 3 {
            // Deconvolution and wavelets were favourites; they now live in "Nitidezza".
            self.modules.iter_mut().filter(|m| m.visibility == Visibility::Favorite).for_each(|m| m.visibility = Visibility::Visible);
            if let Some(m) = self.modules.iter_mut().find(|m| m.stage == Stage::Mineral) {
                // Shown only for the Moon now, so it no longer clutters other images.
                m.visibility = Visibility::Visible;
            }
            self.one_group_open = true;
        }
        self.version = VERSION;
        // Rings and moons are masks in the "Maschere" list now, no longer modules of their own:
        // a stale "hidden" entry must not switch their masks off.
        self.modules.retain(|m| !matches!(m.stage, Stage::Rings | Stage::Moons));
        let defaults = Preferences::default().modules;
        let missing: Vec<ModulePref> = defaults.iter().filter(|d| !self.modules.iter().any(|m| m.stage == d.stage)).cloned().collect();
        self.modules.extend(missing);
        let order = |s: Stage| defaults.iter().position(|d| d.stage == s).unwrap_or(usize::MAX);
        self.modules.sort_by_key(|m| order(m.stage));
        self
    }

    /// Shown in the panel for this image: not hidden and, if adapting, relevant.
    fn shown(&self, stage: Stage, kind: ImageKind, mono: bool) -> bool {
        // Only for a colour picture of Jupiter, whatever the "adapt to the image" setting says.
        if stage == Stage::PhotoColor && (mono || self.subject != Some(ps_core::capture::Body::Jupiter)) {
            return false;
        }
        self.visibility(stage) != Visibility::Hidden && (!self.adapt_to_image || relevant(stage, kind, mono))
    }

    pub fn visibility(&self, stage: Stage) -> Visibility {
        self.modules.iter().find(|m| m.stage == stage).map_or(Visibility::Visible, |m| m.visibility)
    }

    fn set_visibility(&mut self, stage: Stage, v: Visibility) {
        if let Some(m) = self.modules.iter_mut().find(|m| m.stage == stage) {
            m.visibility = v;
        }
    }

    /// Panel layout: favourites first, then the other visible modules by thematic group,
    /// each in processing order. Empty groups are left out.
    pub fn panel_layout(&self, kind: ImageKind, mono: bool) -> Vec<(Group, Vec<Stage>)> {
        let shown = |m: &&ModulePref| self.shown(m.stage, kind, mono);
        let favorites: Vec<Stage> = self.modules.iter().filter(shown).filter(|m| m.visibility == Visibility::Favorite).map(|m| m.stage).collect();
        let mut out = Vec::new();
        if !favorites.is_empty() {
            out.push((Group::Favorites, favorites));
        }
        for group in GROUP_ORDER {
            let stages: Vec<Stage> = self.modules.iter().filter(shown).filter(|m| m.visibility == Visibility::Visible && Group::of(m.stage) == group).map(|m| m.stage).collect();
            if !stages.is_empty() {
                out.push((group, stages));
            }
        }
        out
    }

    /// The pipeline actually processed: modules not shown are switched off.
    pub fn effective(&self, p: &Pipeline, kind: ImageKind, mono: bool) -> Pipeline {
        let mut p = p.clone();
        for m in self.modules.iter().filter(|m| !self.shown(m.stage, kind, mono)) {
            match m.stage {
                Stage::Align => p.align.enabled = false,
                Stage::Bxt => p.bxt.enabled = false,
                Stage::Deconv => p.deconv.enabled = false,
                Stage::Denoise => {
                    p.denoise.enabled = false;
                    p.layers.iter_mut().for_each(|l| l.denoise.enabled = false);
                }
                Stage::Wavelet => {
                    p.wavelet.enabled = false;
                    p.layers.iter_mut().for_each(|l| l.wavelet.enabled = false);
                }
                Stage::Smart => p.smart.enabled = false,
                Stage::Contrast => {
                    p.contrast.enabled = false;
                    p.layers.iter_mut().for_each(|l| l.contrast.enabled = false);
                }
                Stage::Tone => {
                    p.tone.enabled = false;
                    p.layers.iter_mut().for_each(|l| l.tone.enabled = false);
                }
                Stage::Curves => p.curves.enabled = false,
                Stage::Color => {
                    p.color.enabled = false;
                    p.layers.iter_mut().for_each(|l| l.color.enabled = false);
                }
                Stage::Layers => p.layers.iter_mut().for_each(|l| l.enabled = false),
                Stage::PhotoColor => p.photocolor.enabled = false,
                Stage::Mineral => p.mineral.enabled = false,
                Stage::SunColor => p.suncolor.enabled = false,
                Stage::Dering => p.dering.enabled = false,
                // Listed among the masks, not as modules (see `upgraded`).
                Stage::Rings | Stage::Moons => {}
            }
        }
        p
    }

    pub fn bxt_executable(&self) -> Option<PathBuf> {
        self.bxt_path.clone().filter(|p| p.is_file())
    }
}

/// State of the open Settings window.
#[derive(Default)]
pub struct SettingsWindow {
    pub open: bool,
    check: Option<Receiver<Result<String, String>>>,
    check_result: Option<Result<String, String>>,
}

impl SettingsWindow {
    pub fn show(&mut self, ctx: &egui::Context, prefs: &mut Preferences) {
        if !self.open {
            return;
        }
        if let Some(rx) = &self.check
            && let Ok(result) = rx.try_recv()
        {
            self.check_result = Some(result);
            self.check = None;
        }
        // Same corners, border and title size as the other windows; modal, unlike them.
        let frame = egui::Frame::new().fill(crate::theme::PANEL).corner_radius(theme::radius::WINDOW).inner_margin(20).stroke(Stroke::new(1.0, BORDER));
        let response = egui::Modal::new(egui::Id::unique("settings")).frame(frame).show(ctx, |ui| {
            ui.set_width(520.0);
            ui.horizontal(|ui| {
                ui.label(ICON_SETTINGS.rich_text().size(theme::size::ICON).color(ACCENT));
                ui.label(RichText::new(tl("Impostazioni")).size(theme::size::HEADING).strong().color(TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if crate::widgets::icon_button(ui, ICON_CLOSE, tl("Chiudi (Esc)")).clicked() {
                        self.open = false;
                    }
                });
            });
            ui.add_space(10.0);
            // Taller than small windows: scroll, keeping the title in view.
            let max_height = (ctx.content_rect().height() - 140.0).max(200.0);
            egui::ScrollArea::vertical().max_height(max_height).min_scrolled_height(max_height.min(720.0)).auto_shrink([false, true]).show(ui, |ui| {
                // Shown in both languages, so it can be found whichever one is on.
                section(ui, "LINGUA · LANGUAGE");
                if crate::widgets::segmented(ui, &mut prefs.language, &[(Lang::It, None, Lang::It.native_name()), (Lang::En, None, Lang::En.native_name())]) {
                    ps_core::i18n::set_lang(prefs.language);
                }
                ui.add_space(14.0);
                self.modules_section(ui, prefs);
                ui.add_space(14.0);
                self.bxt_section(ui, prefs);
                ui.add_space(14.0);
                section(ui, tl("APERTURA DELLE IMMAGINI"));
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.auto_structure);
                    ui.label(tl("Scegli le impostazioni avanzate (PSF, livelli, kernel…) in base all'immagine"));
                });
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.auto_settings);
                    ui.label(tl("Applica subito anche nitidezza e riduzione del rumore automatiche"));
                });
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.adapt_to_image);
                    ui.label(tl("Mostra solo i moduli adatti all'immagine aperta"));
                });
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.identify_tab_on_open);
                    ui.label(tl("Apri subito la scheda «Identifica e annota» se conosco oggetto e data"));
                });
                ui.add_space(14.0);
                section(ui, tl("PRESTAZIONI"));
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.use_gpu);
                    ui.label(tl("Usa la scheda grafica (GPU)"));
                });
                note(
                    ui,
                    &match ps_core::gpu::status() {
                        None => tl("Ricerca della GPU in corso…").to_string(),
                        Some(None) => tl("Nessuna GPU utilizzabile: si usa il processore.").to_string(),
                        Some(Some(gpu)) if gpu.shared_memory => tr!("{}: deconvoluzione, riduzione del rumore, wavelet, contrasto, dering e maschere. Il risultato è lo stesso del processore.", "{}: deconvolution, noise reduction, wavelets, contrast, dering and masks. The result is the same as on the processor.",
                            gpu.name
                        ),
                        Some(Some(gpu)) => tr!("{}: deconvoluzione e riduzione del rumore Luna e Sole. Le operazioni leggere restano sul processore: su una scheda separata trasferire l'immagine costerebbe quanto calcolarla. Il risultato è lo stesso.", "{}: deconvolution and Moon and Sun noise reduction. Light operations stay on the processor: on a separate card, transferring the image would cost as much as computing it. The result is the same.",
                            gpu.name
                        ),
                    },
                );
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.fast_preview);
                    ui.label(tl("Anteprima rapida mentre muovi un cursore"));
                });
                note(ui, tl("Durante il trascinamento elabora l'immagine ridotta (approssimata); al rilascio torna a piena risoluzione."));
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.memory_saving);
                    ui.label(tl("Risparmia memoria"));
                });
                note(ui, tl("Tiene in memoria solo i passi lenti (deconvoluzione, rumore, wavelet…) e ricalcola quelli rapidi. Utile con immagini grandi e poca RAM."));
                ui.add_space(14.0);
                section(ui, tl("PANNELLO"));
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.one_group_open);
                    ui.label(tl("Tieni aperta una sola scheda alla volta"));
                });
            });
        });
        if response.should_close() {
            self.open = false;
        }
    }

    fn modules_section(&mut self, ui: &mut egui::Ui, prefs: &mut Preferences) {
        section(ui, tl("MODULI DEL PANNELLO"));
        ui.label(RichText::new(tl("I preferiti stanno in cima; i moduli nascosti non vengono né mostrati né applicati.")).small().color(TEXT_WEAK));
        ui.add_space(6.0);
        let stages: Vec<Stage> = prefs.modules.iter().map(|m| m.stage).collect();
        ui.spacing_mut().item_spacing.y = 4.0;
        for stage in stages {
            ui.horizontal(|ui| {
                ui.label(stage_icon(stage).rich_text().size(theme::size::ICON).color(TEXT_WEAK));
                ui.label(RichText::new(stage.name()).color(TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut v = prefs.visibility(stage);
                    let options = [
                        (Visibility::Favorite, Some(ICON_STAR), tl("Preferito")),
                        (Visibility::Visible, Some(ICON_VISIBILITY), tl("Visibile")),
                        (Visibility::Hidden, Some(ICON_VISIBILITY_OFF), tl("Nascosto")),
                    ];
                    if segmented(ui, &mut v, &options) {
                        prefs.set_visibility(stage, v);
                    }
                });
            });
        }
    }

    fn bxt_section(&mut self, ui: &mut egui::Ui, prefs: &mut Preferences) {
        section(ui, "BLURXTERMINATOR");
        let mut enabled = prefs.visibility(Stage::Bxt) != Visibility::Hidden;
        ui.horizontal(|ui| {
            if toggle(ui, &mut enabled).changed() {
                prefs.set_visibility(Stage::Bxt, if enabled { Visibility::Visible } else { Visibility::Hidden });
            }
            ui.label(tl("Usa BlurXTerminator (strumento a riga di comando rc-astro)"));
        });
        ui.label(
            RichText::new(tl("Richiede il tool stand-alone di RC Astro e una licenza attiva di BlurXTerminator."))
                .small()
                .color(TEXT_WEAK),
        );
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let path = prefs.bxt_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| tl("non trovato").into());
            ui.label(RichText::new(tl("Eseguibile")).color(TEXT_WEAK));
            ui.add(egui::Label::new(RichText::new(path).monospace().small()).truncate());
        });
        ui.horizontal(|ui| {
            if ui.button((ICON_SEARCH.rich_text(), tl("Cerca automaticamente"))).clicked() {
                prefs.bxt_path = ps_core::bxt::find_executable();
                self.check_result = prefs.bxt_path.is_none().then(|| Err(tl("rc-astro non trovato: indicalo con Sfoglia").to_string()));
            }
            if ui.button((ICON_FOLDER_OPEN.rich_text(), tl("Sfoglia…"))).clicked()
                && let Some(p) = rfd::FileDialog::new().set_title(tl("Seleziona l'eseguibile rc-astro")).pick_file()
            {
                prefs.bxt_path = Some(p);
                self.check_result = None;
            }
            let can_check = self.check.is_none() && prefs.bxt_executable().is_some();
            if ui.add_enabled(can_check, egui::Button::new((ICON_VERIFIED.rich_text(), tl("Verifica")))).clicked()
                && let Some(exe) = prefs.bxt_executable()
            {
                let (tx, rx) = channel();
                let ctx = ui.ctx().clone();
                std::thread::spawn(move || {
                    tx.send(ps_core::bxt::check(&exe)).ok();
                    ctx.request_repaint();
                });
                self.check = Some(rx);
                self.check_result = None;
            }
            if self.check.is_some() {
                ui.add(egui::Spinner::new().size(14.0).color(ACCENT));
            }
        });
        match &self.check_result {
            Some(Ok(text)) => {
                ui.horizontal(|ui| {
                    ui.label(ICON_CHECK_CIRCLE.rich_text().color(SUCCESS));
                    ui.label(RichText::new(first_lines(text)).small().color(SUCCESS));
                });
            }
            Some(Err(text)) => {
                ui.label(RichText::new(first_lines(text)).small().color(ui.visuals().error_fg_color));
            }
            None => {}
        }
    }
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.label(caps(title));
    ui.add_space(4.0);
}

fn first_lines(text: &str) -> String {
    text.lines().take(3).collect::<Vec<_>>().join("\n")
}

fn free_scales() -> ps_core::wavelet::ScaleMode {
    ps_core::wavelet::ScaleMode::Free
}

fn default_keywords() -> Vec<crate::naming::NameKeyword> {
    vec![crate::naming::NameKeyword::Suffix]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shown(p: &Preferences, kind: ImageKind, mono: bool) -> Vec<Stage> {
        p.panel_layout(kind, mono).into_iter().flat_map(|(_, s)| s).collect()
    }

    #[test]
    fn tab_names_are_translated_but_the_saved_key_is_not() {
        let all = [Group::Favorites, Group::Sharpness, Group::Corrections, Group::Noise, Group::Tone, Group::Color, Group::Regions, Group::Identify, Group::Export];
        ps_core::i18n::with_lang(Lang::En, || {
            for g in all {
                assert_eq!(g.label(), ps_core::i18n::tl(g.name()), "{:?} label matches its key's translation", g);
                assert_ne!(g.label(), "", "{g:?}");
            }
            assert_eq!(Group::Corrections.label(), "Artefact correction");
            assert_eq!(Group::Identify.name(), "Identifica e annota");
        });
    }

    /// Jupiter's photometric colour exists only for a colour picture whose subject is Jupiter.
    #[test]
    fn photometric_colour_needs_jupiter_in_colour() {
        use ps_core::capture::Body;
        let mut p = Preferences::default();
        p.modules.iter_mut().for_each(|m| m.visibility = Visibility::Visible);
        assert!(!shown(&p, ImageKind::Planet, false).contains(&Stage::PhotoColor), "no subject known");
        p.subject = Some(Body::Mars);
        assert!(!shown(&p, ImageKind::Planet, false).contains(&Stage::PhotoColor));
        p.subject = Some(Body::Jupiter);
        assert!(shown(&p, ImageKind::Planet, false).contains(&Stage::PhotoColor));
        assert!(!shown(&p, ImageKind::Planet, true).contains(&Stage::PhotoColor), "mono");
        // Even with "adapt to the image" off.
        p.adapt_to_image = false;
        assert!(!shown(&p, ImageKind::Planet, true).contains(&Stage::PhotoColor));
        // Switched on it is dropped from the processed pipeline when the subject isn't Jupiter.
        let mut pipeline = Pipeline::default();
        pipeline.photocolor.enabled = true;
        p.subject = Some(Body::Saturn);
        assert!(!p.effective(&pipeline, ImageKind::Saturn, false).photocolor.enabled);
        p.subject = Some(Body::Jupiter);
        assert!(p.effective(&pipeline, ImageKind::Planet, false).photocolor.enabled);
    }

    /// Settings saved by the first versions of the photometric colour still read.
    #[test]
    fn old_photometric_references_still_read() {
        use ps_core::photocolor::{PhotoColorParams, Reference};
        let old: PhotoColorParams = serde_json::from_str(r#"{"reference":"HubbleNatural2019","strength":0.5}"#).unwrap();
        assert_eq!((old.reference, old.strength, old.tilt), (Reference::Hubble2019, 0.5, 0.0));
        let older: PhotoColorParams = serde_json::from_str(r#"{"reference":"Hubble2019"}"#).unwrap();
        assert_eq!(older.reference, Reference::Hubble2019);
        assert_eq!(older.strength, 1.0, "switched on, it starts at full strength");
    }

    #[test]
    fn panel_adapts_to_the_image() {
        let mut p = Preferences::default();
        p.modules.iter_mut().for_each(|m| m.visibility = Visibility::Visible);
        let mono = shown(&p, ImageKind::Planet, true);
        assert!(!mono.contains(&Stage::Color) && !mono.contains(&Stage::Align) && !mono.contains(&Stage::Mineral));
        assert!(mono.contains(&Stage::Tone), "levels and curves still make sense in mono");
        assert!(!shown(&p, ImageKind::Saturn, false).contains(&Stage::Mineral));
        assert!(shown(&p, ImageKind::FullDisk, false).contains(&Stage::Mineral));
        p.adapt_to_image = false;
        assert!(shown(&p, ImageKind::Planet, true).contains(&Stage::Color));
    }

    #[test]
    fn groups_follow_processing_order() {
        let mut p = Preferences::default();
        p.modules.iter_mut().for_each(|m| m.visibility = Visibility::Visible);
        let layout = p.panel_layout(ImageKind::Planet, false);
        assert_eq!(layout[0].0, Group::Sharpness, "sharpening comes first");
        let color = layout.iter().find(|(g, _)| *g == Group::Color).unwrap();
        assert!(color.1.contains(&Stage::Align) && !color.1.contains(&Stage::Tone), "tone and colour are separate");
    }

    /// Rings and moons used to be modules: hiding one switched its mask off.
    #[test]
    fn old_ring_and_moon_modules_are_dropped() {
        let mut old = Preferences::default();
        old.modules.push(ModulePref { stage: Stage::Rings, visibility: Visibility::Hidden });
        let p = old.upgraded();
        assert!(!p.modules.iter().any(|m| matches!(m.stage, Stage::Rings | Stage::Moons)));
        let mut pipeline = Pipeline::default();
        ps_core::preset::add_role_layer(&mut pipeline, ps_core::pipeline::LayerRole::Rings);
        assert!(p.effective(&pipeline, ImageKind::Saturn, false).layers[0].enabled);
        assert!(!p.panel_layout(ImageKind::Saturn, false).iter().flat_map(|(_, s)| s).any(|s| *s == Stage::Rings));
    }

    #[test]
    fn old_preferences_stop_auto_applying() {
        let old = Preferences { auto_settings: true, version: 0, ..Default::default() };
        assert!(!old.upgraded().auto_settings);
    }

    /// The bug behind "everything was already on": a saved file without `version` must read
    /// as version 0, not as the current one.
    #[test]
    fn missing_version_reads_as_old() {
        let json = r#"{"auto_settings": true}"#;
        let p: Preferences = serde_json::from_str(json).unwrap();
        assert_eq!(p.version, 0);
        assert!(!p.upgraded().auto_settings);
    }
}
