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

/// An image opened before: where it is and a small RGB picture of it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecentFile {
    pub path: PathBuf,
    pub thumb: Vec<u8>,
    pub size: [usize; 2],
}

const THUMB_SIDE: usize = 48;
const RECENT_FILES: usize = 8;

/// `display` reduced to fit a `THUMB_SIDE` square, as packed RGB with its size.
pub fn thumbnail(display: &eframe::egui::ColorImage) -> (Vec<u8>, [usize; 2]) {
    let [w, h] = display.size;
    let scale = (w.max(h) as f32 / THUMB_SIDE as f32).max(1.0);
    let (tw, th) = (((w as f32 / scale).round() as usize).max(1), ((h as f32 / scale).round() as usize).max(1));
    let mut rgb = Vec::with_capacity(tw * th * 3);
    for ty in 0..th {
        for tx in 0..tw {
            let (x0, x1) = (tx * w / tw, ((tx + 1) * w / tw).max(tx * w / tw + 1));
            let (y0, y1) = (ty * h / th, ((ty + 1) * h / th).max(ty * h / th + 1));
            let (mut sum, mut n) = ([0u32; 3], 0u32);
            // At most 4 × 4 samples per block.
            for y in (y0..y1).step_by(((y1 - y0) / 4).max(1)) {
                for x in (x0..x1).step_by(((x1 - x0) / 4).max(1)) {
                    let c = display.pixels[y * w + x];
                    sum = [sum[0] + u32::from(c.r()), sum[1] + u32::from(c.g()), sum[2] + u32::from(c.b())];
                    n += 1;
                }
            }
            rgb.extend(sum.map(|v| (v / n.max(1)) as u8));
        }
    }
    (rgb, [tw, th])
}

impl Preferences {
    /// `path` opened now: first in the list of recent images.
    pub fn remember_file(&mut self, path: PathBuf, display: &eframe::egui::ColorImage) {
        let (thumb, size) = thumbnail(display);
        self.recent_files.retain(|r| r.path != path);
        self.recent_files.insert(0, RecentFile { path, thumb, size });
        self.recent_files.truncate(RECENT_FILES);
    }
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
    /// Open images already pre-processed (sharpening, noise, colour, levels, north up) instead
    /// of raw. The structural settings (PSF, wavelet levels…) are always automatic.
    pub auto_settings: bool,
    /// The images opened last, newest first, with a small picture of each.
    #[serde(default)]
    pub recent_files: Vec<RecentFile>,
    /// The first-open question (raw or pre-processed) was answered: it is asked only once.
    #[serde(default)]
    pub preprocess_asked: bool,
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
    /// The equipment configurations the user saved, by name.
    #[serde(default)]
    pub saved_setups: Vec<SavedSetup>,
    /// The saved configuration every new image starts with (none = no equipment until chosen).
    #[serde(default)]
    pub favorite_setup: Option<String>,
    /// The user's own details, for the printed annotations.
    #[serde(default)]
    pub profile: Profile,
    /// Keywords added to exported files' names.
    #[serde(default = "default_keywords")]
    pub export_keywords: Vec<crate::naming::NameKeyword>,
    /// How the levels and curves histograms are drawn.
    /// (Renamed so that settings saved while grey was the default start in colour.)
    #[serde(rename = "histogram_colour_mode")]
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
    Noise,
    Tone,
    Color,
    Regions,
    /// Not a group of modules: what the image shows (moons, seas), from the ephemerides.
    Identify,
    /// Not a group of modules: the export tab.
    Export,
}

impl Group {
    pub fn name(self) -> &'static str {
        match self {
            Group::Favorites => "Preferiti",
            Group::Sharpness => "Nitidezza",
            Group::Noise => "Rumore e artefatti",
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
            Group::Noise => tl("Rumore e artefatti"),
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
            Stage::Dering | Stage::Denoise => Group::Noise,
            Stage::Tone | Stage::Curves => Group::Tone,
            Stage::Align | Stage::PhotoColor | Stage::Color | Stage::Mineral | Stage::SunColor => Group::Color,
            Stage::Rings | Stage::Moons | Stage::Layers => Group::Regions,
        }
    }
}

/// Processing order: sharpen, clean, then tone and colour, then local work.
pub const GROUP_ORDER: [Group; 5] = [Group::Sharpness, Group::Noise, Group::Tone, Group::Color, Group::Regions];

/// Whether a module makes sense for this image.
pub fn relevant(stage: Stage, kind: ImageKind, mono: bool) -> bool {
    match stage {
        Stage::Align | Stage::Color => !mono,
        // Jupiter's or Mars's disk, Saturn: needs the subject to be known too (see `Preferences::subject`).
        Stage::PhotoColor => !mono && matches!(kind, ImageKind::Planet | ImageKind::Saturn),
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
            preprocess_asked: false,
            recent_files: Vec::new(),
            adapt_to_image: true,
            one_group_open: true,
            expanded: HashSet::new(),
            current_area: Group::Sharpness.name().to_string(),
            export_format: "tif".into(),
            export_suffix: "_ps".into(),
            export_keywords: default_keywords(),
            saved_setups: Vec::new(),
            favorite_setup: None,
            profile: Profile::default(),
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
        // Only for a colour picture of a subject with references (Jupiter, Saturn, Mars), whatever
        // the "adapt to the image" setting says.
        if stage == Stage::PhotoColor && (mono || self.subject.is_none_or(|b| ps_core::photocolor::Reference::for_body(b).is_empty())) {
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
            ui.add_space(theme::space::M);
            // Taller than small windows: scroll, keeping the title in view.
            let max_height = (ctx.content_rect().height() - 140.0).max(200.0);
            egui::ScrollArea::vertical().max_height(max_height).min_scrolled_height(max_height.min(720.0)).auto_shrink([false, true]).show(ui, |ui| {
                // Shown in both languages, so it can be found whichever one is on.
                section(ui, "LINGUA · LANGUAGE");
                if crate::widgets::segmented(ui, &mut prefs.language, &[(Lang::It, None, Lang::It.native_name()), (Lang::En, None, Lang::En.native_name())]) {
                    ps_core::i18n::set_lang(prefs.language);
                }
                ui.add_space(theme::space::L);
                self.modules_section(ui, prefs);
                ui.add_space(theme::space::L);
                self.bxt_section(ui, prefs);
                ui.add_space(theme::space::L);
                section(ui, tl("APERTURA DELLE IMMAGINI"));
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.auto_settings);
                    ui.label(tl("Apri le immagini già preelaborate (nitidezza, rumore, colore, livelli, nord in alto)"));
                });
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.adapt_to_image);
                    ui.label(tl("Mostra solo i moduli adatti all'immagine aperta"));
                });
                ui.horizontal(|ui| {
                    toggle(ui, &mut prefs.identify_tab_on_open);
                    ui.label(tl("Apri subito la scheda «Identifica e annota» se conosco oggetto e data"));
                });
                ui.add_space(theme::space::L);
                section(ui, tl("I TUOI DATI"));
                note(ui, tl("Compaiono nelle annotazioni stampate sull'immagine, quando i campi della scheda «Identifica e annota» sono vuoti."));
                for (label, value, hint) in [
                    (tl("Nome"), &mut prefs.profile.author, tl("Nome e cognome")),
                    (tl("Luogo"), &mut prefs.profile.location, tl("Città, paese")),
                    (tl("Contatto"), &mut prefs.profile.contact, tl("E-mail, sito o profilo")),
                    (tl("Riga fissa"), &mut prefs.profile.note, tl("© anno, crediti di elaborazione…")),
                ] {
                    ui.horizontal(|ui| {
                        ui.add_sized([90.0, 20.0], egui::Label::new(label));
                        ui.add(egui::TextEdit::singleline(value).desired_width(260.0).hint_text(hint));
                    });
                }
                ui.add_space(theme::space::L);
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
                ui.add_space(theme::space::L);
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
        ui.add_space(theme::space::M);
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
        ui.add_space(theme::space::M);
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
                && let Some(p) = crate::dialog::FileDialog::new().set_title(tl("Seleziona l'eseguibile rc-astro")).pick_file()
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
    ui.add_space(theme::space::S);
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

/// An equipment configuration with a name of the user's choosing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedSetup {
    pub name: String,
    pub setup: ps_core::equipment::Setup,
}

/// Who took the pictures: printed in the annotations when the fields there are left empty.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Profile {
    pub author: String,
    pub location: String,
    /// E-mail, web page or social account.
    pub contact: String,
    /// A line added to every caption (copyright, processing credits…).
    pub note: String,
}

impl Preferences {
    /// The equipment a new image starts with: the favourite configuration, else nothing.
    pub fn favorite_setup_value(&self) -> ps_core::equipment::Setup {
        self.favorite_setup.as_ref().and_then(|n| self.saved_setups.iter().find(|s| &s.name == n)).map(|s| s.setup.clone()).unwrap_or_default()
    }

    /// The annotation options with the user's details filling the empty fields.
    pub fn label_options(&self) -> crate::annotate::LabelOptions {
        let mut o = self.identify.clone();
        if o.use_profile {
            let p = &self.profile;
            for (field, mine) in [(&mut o.author, &p.author), (&mut o.location, &p.location), (&mut o.contact, &p.contact), (&mut o.note, &p.note)] {
                if field.trim().is_empty() {
                    field.clone_from(mine);
                }
            }
        }
        o
    }
}
