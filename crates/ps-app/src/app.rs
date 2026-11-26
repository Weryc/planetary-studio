use ps_core::i18n::tl;
use ps_core::tr;
use crate::history::History;
use crate::animation_ui::AnimationEditor;
use crate::choose_ui::{Choice, ChooseEditor};
use crate::mono_rgb_ui::{MonoJob, MonoRgbEditor, MonoRgbOutcome};
use crate::mosaic_ui::{MosaicEditor, MosaicOutcome};
use crate::derotation_ui::{Derotated, DerotationEditor, DerotationOutcome, StackJob};
use crate::panels::{self, BrushSettings, LayerTemplate, LayerTool, LayerUi, MaskThumbView, PanelAction, PickTarget, PsfPreview, StageStatus, ToneParts, format_duration};
use crate::prefs::{self as prefs_group, Preferences, SettingsWindow};
use crate::theme::{self, ACCENT, BORDER, TEXT, TEXT_WEAK};
use crate::viewer::{BrushView, Compare, CropAction, Overlay, ShapeKind, Tool, Viewer};
use crate::widgets::{Histogram, caps, icon_button, primary_button, segmented, tool_button};
use crate::worker::{self, Output, Request, Roi, Worker};
use eframe::egui::{self, ColorImage, Key, KeyboardShortcut, Modifiers, Pos2, Rect, RichText, Stroke, Vec2};
use egui_material_icons::icons::*;
use ps_core::Image;
use ps_core::analysis::{Analysis, ImageKind, analyze};
use ps_core::color::{ColorMode, working_planes};
use ps_core::mask::{MaskShape, MaskSpec, Stroke as BrushStroke};
use ps_core::mosaic::{Mosaic, Placement};
use ps_core::geometry::{self, Geometry};
use ps_core::dering::DeringParams;
use ps_core::pipeline::{Aux, LayerRole, RING_MASK, Module, Pipeline, Processor, Stage, StageTiming, next_generation};
use ps_core::smart::SmartView;
use ps_core::preset::{add_role_layer, auto_pipeline_with, explain, initial_pipeline, restructure};
use ps_core::wavelet::{ScaleMode, WaveletParams};
use ps_core::equipment::Setup;
use ps_core::deconv::{DeconvAlgorithm, DeconvParams};
use crate::instrument::{CaptureState, InstrumentAction, InstrumentView, instrument_body, setup_label};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

const OPEN: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::O);
pub(crate) const EXPORT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::E);
pub(crate) const UNDO: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Z);
const REDO: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::Z);
/// The platform's standard shortcut for preferences (⌘, on macOS, Ctrl+, elsewhere).
const HELP: KeyboardShortcut = KeyboardShortcut::new(Modifiers::NONE, Key::Questionmark);
const SETTINGS: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Comma);
const FIT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::NONE, Key::F);
const ACTUAL_SIZE: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Num1);
/// Export next to the original, in the last format used.
pub(crate) const QUICK_EXPORT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::E);
/// Keys 1‥5 pick the panel's areas.
const AREA_KEYS: [Key; 6] = [Key::Num1, Key::Num2, Key::Num3, Key::Num4, Key::Num5, Key::Num6];
const TOGGLE_ROI: KeyboardShortcut = KeyboardShortcut::new(Modifiers::NONE, Key::R);
const SPLIT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::NONE, Key::S);
const CROP: KeyboardShortcut = KeyboardShortcut::new(Modifiers::NONE, Key::C);

const STORAGE_PREFS: &str = "preferences";
/// Saved mosaics.
const MOSAIC_EXTENSION: &str = "psmosaic";

/// A mosaic with its frames loaded (`None` for those left out).
type MosaicFrames = (Mosaic, Vec<Option<Arc<Image>>>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreviewMode {
    Full,
    Roi,
}

/// Results posted back to the UI thread by file I/O threads.
enum Event {
    /// `mosaic`: the image was joined from these frames; `keep_settings`: it replaces the open
    /// mosaic, whose processing stays.
    /// `derotated`: the image is the sum of these captures of Jupiter or Saturn.
    Loaded { path: PathBuf, image: Image, display: ColorImage, analysis: Box<Analysis>, elapsed: Duration, mosaic: Option<Box<Mosaic>>, keep_settings: bool, derotated: Option<Derotated> },
    LuminanceLoaded { name: String, plane: Vec<f32> },
    Exported { path: PathBuf, elapsed: Duration },
    Failed(String),
}

/// The question asked before a series is processed.
struct BatchDialog {
    files: Vec<PathBuf>,
    dir: Option<PathBuf>,
    mode: crate::batch::Mode,
}

struct Document {
    path: PathBuf,
    /// As loaded.
    original: Arc<Image>,
    /// Rotation and mirroring applied to `original`…
    geometry: Geometry,
    /// …giving this; `source` is this, cropped.
    oriented: Arc<Image>,
    /// Top-left of the crop inside `oriented`.
    crop_offset: (usize, usize),
    source: Arc<Image>,
    source_gen: u64,
    viewer: Viewer,
    analysis: Analysis,
    /// Current type: the detected one unless the user picked another.
    kind: ImageKind,
    luminance: Option<Luminance>,
    /// The names and marks from «Identifica», laid over the finished image.
    annotation: Option<Annotation>,
    /// Base name of exported files (the original's unless edited).
    export_name: String,
    /// The frames this image was joined from, and how.
    mosaic: Option<Mosaic>,
    /// Subject, time and log of the capture.
    capture: CaptureState,
    /// The equipment it was taken with.
    setup: Setup,
    /// The pre-processed settings it was opened with, while they are still the current ones.
    preprocessed: Option<Pipeline>,
}

/// The deconvolution comparison: six tiles computed in the background.
struct DeconvCompare {
    tiles: Vec<CompareTile>,
    pending: Option<std::sync::mpsc::Receiver<(usize, ColorImage)>>,
    /// Width of the blur used, px.
    fwhm: f32,
}

struct CompareTile {
    label: String,
    /// `None` for the original.
    params: Option<DeconvParams>,
    texture: Option<egui::TextureHandle>,
}

/// "Moffat", "Airy"… for tile labels.
fn short_psf(psf: &ps_core::psf::Psf) -> &'static str {
    match psf {
        ps_core::psf::Psf::Gaussian { .. } => tl("Gaussiana"),
        ps_core::psf::Psf::Moffat { .. } => "Moffat",
        ps_core::psf::Psf::Airy { .. } => "Airy",
        ps_core::psf::Psf::Measured { .. } => tl("misurata"),
    }
}

/// The planes of an annotation being made in the background.
type AnnotationAnswer = std::sync::mpsc::Receiver<[(Arc<Vec<f32>>, u64); 4]>;

/// Names and marks laid over the finished image: premultiplied red, green, blue and coverage
/// planes for the working source they were made for.
struct Annotation {
    source_gen: u64,
    /// What it was made from (names, options, image): to know when it is out of date.
    key: u64,
    planes: [(Arc<Vec<f32>>, u64); 4],
}

impl Document {
    /// The names to lay over the image, if they were made for the image as it is now.
    fn overlay(&self) -> Option<[(Arc<Vec<f32>>, u64); 4]> {
        self.annotation.as_ref().filter(|a| a.source_gen == self.source_gen).map(|a| a.planes.clone())
    }
}

/// LRGB luminance loaded from a mono file.
struct Luminance {
    name: String,
    /// At the original's size and orientation, registered onto it.
    full: Arc<Image>,
    /// Cropped like `source`, with its generation.
    cropped: (Arc<Vec<f32>>, u64),
}

/// What the Identifica tab needs to know about the open image.
fn identify_input(doc: &Document) -> crate::identify_ui::IdentifyInput<'_> {
    let body = doc.capture.body.or(doc.analysis.planet).or((doc.kind == ImageKind::Saturn).then_some(ps_core::capture::Body::Saturn));
    crate::identify_ui::IdentifyInput {
        path: &doc.path,
        source: &doc.source,
        source_gen: doc.source_gen,
        body,
        time: doc.capture.info.time,
        time_from: doc.capture.time_from,
        filter: doc.capture.info.filter.clone(),
        // The instrument chosen in the top bar; the capture's own words when none is.
        equipment: {
            let short = |name: &str| name.split(" (").next().unwrap_or(name).to_string();
            match (doc.setup.telescope.as_ref().map(|t| short(&t.name)).or_else(|| doc.capture.info.telescope.clone()), doc.setup.camera.as_ref().map(|c| short(&c.name)).or_else(|| doc.capture.info.camera.clone())) {
                (Some(t), Some(c)) => Some(format!("{t} · {c}")),
                (Some(x), None) | (None, Some(x)) => Some(x),
                _ => None,
            }
        },
    }
}

impl Document {
    fn crop_luminance(&mut self) {
        if let Some(Luminance { full, cropped, .. }) = &mut self.luminance {
            let oriented = geometry::apply(full, &self.geometry);
            let (ox, oy) = self.crop_offset;
            let plane = oriented.crop(ox, oy, self.source.width(), self.source.height()).channels()[0].clone();
            *cropped = (Arc::new(plane), next_generation());
        }
    }
}

/// What was last sent to the worker, to avoid resubmitting an identical request.
#[derive(PartialEq)]
struct RequestKey {
    source_gen: u64,
    pipeline: Pipeline,
    roi: Option<Roi>,
    luminance_gen: Option<u64>,
    overlay_gen: Option<u64>,
    bxt: Option<PathBuf>,
    mask_overlay: Option<u64>,
    memory_saving: bool,
    proxy: usize,
    ringing_map: bool,
    clip_warning: bool,
    smart_view: SmartView,
}

/// Version "A" of the comparison: the settings that made the result kept by the viewer, while
/// editing goes on as "B".
struct Snapshot {
    source_gen: u64,
    pipeline: Pipeline,
}

/// How much the quick preview shrinks an image: enough for big lunar and solar frames to
/// stay fluid; small planetary frames aren't worth it.
fn proxy_factor(img: &Image) -> usize {
    match img.width().max(img.height()) {
        0..600 => 1,
        600..3000 => 2,
        _ => 4,
    }
}

pub struct PlanetaryStudio {
    document: Option<Document>,
    pipeline: Pipeline,
    history: History,
    prefs: Preferences,
    settings: SettingsWindow,
    about_open: bool,
    /// The automatic identification's moon view: when it started and when its search settled.
    /// It stays on only a few seconds, then the view is switched off.
    auto_identify: Option<(Instant, Option<Instant>)>,
    /// The levels were set from the source: refit them on what the sharpening and noise steps
    /// produce (it raises the whites), once the first full result arrives.
    levels_pending: bool,
    /// The window title's base (the file's name) and the title last sent to the window.
    doc_title: String,
    shown_title: String,
    /// The brief notice after an export: its text, the file and when it appeared.
    toast: Option<(String, PathBuf, Instant)>,
    /// The list of keyboard shortcuts, in a window (the `?` key).
    shortcuts_open: bool,
    /// Pictures of the recent images, by path (made when the menu first shows them).
    recent_textures: HashMap<PathBuf, egui::TextureHandle>,
    /// The first-open question (raw or pre-processed), while it is up.
    preprocess_dialog: Option<crate::preprocess::Dialog>,
    /// Ask which planet the open image shows (it was not recognised).
    subject_prompt: bool,
    /// The subject the Identifica tab had last frame (a change there is the user's choice).
    last_identify_body: Option<ps_core::capture::Body>,
    /// The pipeline as last exported or saved (or as the image was opened): closing with a
    /// different one asks first.
    clean_pipeline: Pipeline,
    /// The pipeline of the export under way; it becomes the clean one when that succeeds.
    exporting: Option<Pipeline>,
    confirm_close: bool,
    close_confirmed: bool,
    mosaic_editor: MosaicEditor,
    /// A saved mosaic's frames, being loaded.
    pending_mosaic: Option<Receiver<MosaicFrames>>,
    derotation_editor: DerotationEditor,
    /// Several images opened together, while the first one's pixels say what they show: the
    /// paths and the body (Jupiter and Saturn are derotated, the rest makes a mosaic).
    /// Several images opened together: what to do with them.
    choose_editor: ChooseEditor,
    animation_editor: AnimationEditor,
    mono_rgb_editor: MonoRgbEditor,
    preview: PreviewMode,
    compare: Compare,
    /// Tool chosen in the toolbar (crop and shape editing override it while active).
    tool: Tool,
    cropping: bool,
    selected_layer: Option<u64>,
    show_mask: bool,
    layer_tool: LayerTool,
    brush: BrushSettings,
    /// Miniatures of the layer masks, by layer id (kept while a layer is off).
    mask_thumbs: HashMap<u64, MaskThumbView>,
    /// Width of the side panel's content last frame (watched by a test for runaway growth).
    side_width: f32,
    /// Waiting for a click on the image to set the black or white point.
    picking: Option<PickTarget>,
    /// What the last PSF measurement found, for the deconvolution card.
    psf_note: Option<String>,
    /// The PSF shapes and algorithms side by side.
    deconv_compare: Option<DeconvCompare>,
    /// The pointer was over the image last frame (Space then means "show the original").
    canvas_hovered: bool,
    /// What the canvas showed last frame (Space can override the chosen comparison).
    shown_compare: Compare,
    /// Show where sharpening left rings (from the Dering card).
    show_ringing: bool,
    /// What "Nitidezza intelligente" shows over the image (from its card), and the periods the
    /// resolution map's colours span.
    smart_view: SmartView,
    smart_legend: Option<(f32, f32)>,
    /// The learned sharpening under the three hypotheses on the seeing.
    smart_compare: crate::smart_compare::SmartCompare,
    /// The series being chosen (files, folder, settings) and the one being processed.
    batch_dialog: Option<BatchDialog>,
    batch: Option<crate::batch::Run>,
    /// The saved-configurations window (from the equipment menu) and the name being typed.
    setups_open: bool,
    new_setup_name: String,
    /// The equipment window (from the top bar).
    instrument_open: bool,
    /// A levels slider or curve point was being dragged in the last panel pass.
    tone_dragging: bool,
    /// Show burnt and crushed pixels, and how many there are.
    show_clipping: bool,
    clipped: Option<usize>,
    snapshot: Option<Snapshot>,
    /// "Identifica": names from the ephemerides over the image.
    identify: crate::identify_ui::IdentifyState,
    /// The annotation being made in the background: (what from, image generation, answer).
    annotating: Option<(u64, u64, AnnotationAnswer)>,
    /// The "Ingrandisci" tab's preview tiles.
    resize_preview: crate::resize_ui::ResizePreview,
    /// The last file exported, shown in the export tab.
    last_export: Option<PathBuf>,
    worker: Worker,
    last_request: Option<RequestKey>,
    /// Latest result on screen (full image or ROI).
    last_output: Option<Output>,
    /// Latest full-image result, used for instant export.
    full_output: Option<Output>,
    timings: HashMap<Stage, StageTiming>,
    warnings: HashMap<Stage, String>,
    histogram: Option<Histogram>,
    psf_preview: PsfPreview,
    io_busy: Option<String>,
    status: String,
    hover: Option<(usize, usize)>,
    backend: String,
    events_tx: Sender<Event>,
    events_rx: Receiver<Event>,
}

impl PlanetaryStudio {
    pub fn new(cc: &eframe::CreationContext<'_>, initial_file: Option<PathBuf>) -> Self {
        theme::apply(&cc.egui_ctx);
        let backend = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| {
                let info = rs.adapter.get_info();
                format!("{:?} · {}", info.backend, info.name)
            })
            .unwrap_or_default();
        let prefs = cc.storage.and_then(|s| eframe::get_value::<Preferences>(s, STORAGE_PREFS)).unwrap_or_default().upgraded();
        // Always start on the sharpening tab, whatever was open last time.
        let prefs = Preferences { current_area: prefs_group::Group::Sharpness.name().to_string(), ..prefs };
        ps_core::i18n::set_lang(prefs.language);
        let pipeline = Pipeline::default();
        let (events_tx, events_rx) = channel();
        let mut app = Self {
            history: History::new(&pipeline),
            pipeline,
            prefs,
            settings: SettingsWindow::default(),
            about_open: false,
            toast: None,
            doc_title: String::new(),
            shown_title: String::new(),
            auto_identify: None,
            levels_pending: false,
            batch_dialog: None,
            batch: None,
            setups_open: false,
            new_setup_name: String::new(),
            shortcuts_open: false,
            recent_textures: HashMap::new(),
            preprocess_dialog: None,
            subject_prompt: false,
            last_identify_body: None,
            clean_pipeline: Pipeline::default(),
            exporting: None,
            confirm_close: false,
            close_confirmed: false,
            mosaic_editor: MosaicEditor::default(),
            pending_mosaic: None,
            derotation_editor: DerotationEditor::default(),
            choose_editor: ChooseEditor::default(),
            animation_editor: AnimationEditor::default(),
            mono_rgb_editor: MonoRgbEditor::default(),
            document: None,
            preview: PreviewMode::Full,
            compare: Compare::After,
            tool: Tool::Pan,
            cropping: false,
            selected_layer: None,
            show_mask: false,
            layer_tool: LayerTool::None,
            brush: BrushSettings::default(),
            mask_thumbs: HashMap::new(),
            side_width: 0.0,
            picking: None,
            psf_note: None,
            deconv_compare: None,
            canvas_hovered: false,
            shown_compare: Compare::After,
            show_ringing: false,
            smart_view: SmartView::None,
            smart_legend: None,
            smart_compare: Default::default(),
            instrument_open: false,
            tone_dragging: false,
            show_clipping: false,
            clipped: None,
            snapshot: None,
            last_export: None,
            resize_preview: Default::default(),
            identify: Default::default(),
            annotating: None,
            worker: Worker::new(cc.egui_ctx.clone()),
            last_request: None,
            last_output: None,
            full_output: None,
            timings: HashMap::new(),
            warnings: HashMap::new(),
            histogram: None,
            psf_preview: PsfPreview::default(),
            io_busy: None,
            status: String::new(),
            hover: None,
            backend,
            events_tx,
            events_rx,
        };
        // Finding and opening the GPU takes a moment: do it now, off the interface thread.
        ps_core::gpu::set_enabled(app.prefs.use_gpu);
        std::thread::spawn(|| {
            ps_core::gpu::probe();
        });
        if let Some(path) = initial_file {
            app.load(&cc.egui_ctx, path);
        }
        app
    }

    /// Adjustments made since the image was opened, exported or saved.
    fn has_unsaved_changes(&self) -> bool {
        self.document.is_some() && self.pipeline != self.clean_pipeline
    }

    /// Closing the window with adjustments neither exported nor saved asks first.
    fn close_guard(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.viewport().close_requested()) && self.has_unsaved_changes() && !self.close_confirmed {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.confirm_close = true;
        }
        if !self.confirm_close {
            return;
        }
        let frame = egui::Frame::new().fill(theme::PANEL).corner_radius(theme::radius::WINDOW).inner_margin(20).stroke(egui::Stroke::new(1.0, BORDER));
        let response = egui::Modal::new(egui::Id::unique("confirm-close")).frame(frame).show(ctx, |ui| {
            ui.set_width(380.0);
            ui.label(RichText::new(tl("Uscire senza esportare?")).size(theme::size::HEADING).strong().color(TEXT));
            ui.add_space(theme::space::M);
            ui.label(tl("Le regolazioni fatte su questa immagine non sono state né esportate né salvate."));
            ui.add_space(theme::space::L);
            let mut quit = false;
            ui.horizontal(|ui| {
                if ui.add(primary_button(None, tl("Torna all'immagine"))).clicked() {
                    self.confirm_close = false;
                }
                if ui.button(tl("Esci senza esportare")).clicked() {
                    quit = true;
                }
            });
            quit
        });
        if response.inner {
            self.close_confirmed = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if response.should_close() {
            self.confirm_close = false;
        }
    }

    /// «Informazioni»: version, license, where the crash log goes.
    /// The questions asked on opening: raw or pre-processed (the first time), then which planet
    /// it is when it was not recognised.
    fn opening_questions(&mut self, ctx: &egui::Context) {
        if let Some(dialog) = &mut self.preprocess_dialog {
            if let Some(choice) = dialog.show(ctx) {
                self.preprocess_dialog = None;
                self.prefs.preprocess_asked = true;
                self.prefs.auto_settings = choice == crate::preprocess::Choice::Preprocessed;
                if self.prefs.auto_settings {
                    self.apply_preprocess();
                }
            }
            return;
        }
        if self.subject_prompt {
            let Some(kind) = self.document.as_ref().map(|d| d.kind) else {
                self.subject_prompt = false;
                return;
            };
            if let Some(answer) = crate::preprocess::subject_prompt(ctx, kind) {
                self.subject_prompt = false;
                if let Some((kind, body)) = answer {
                    self.apply_actions(ctx, vec![PanelAction::SelectSubject(panels::Subject { kind, body: Some(body) })]);
                }
            }
        }
    }

    fn about_window(&mut self, ctx: &egui::Context) {
        if !self.about_open {
            return;
        }
        let mut open = true;
        crate::widgets::window(ctx, tl("Informazioni")).open(&mut open).resizable(false).collapsible(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.label(ICON_PLANET.rich_text().size(48.0).color(theme::ACCENT_DIM));
                ui.label(RichText::new("Planetary Studio").size(theme::size::HEADING).strong().color(TEXT));
                ui.label(RichText::new(tr!("Versione {}", "Version {}", env!("CARGO_PKG_VERSION"))).size(theme::size::SMALL).color(TEXT_WEAK));
            });
            ui.add_space(theme::space::M);
            ui.label(tl("Elaborazione di immagini planetarie, lunari e solari già impilate."));
            ui.add_space(theme::space::S);
            crate::panels::hint(ui, tl("Licenza PolyForm Noncommercial 1.0.0. I componenti di terze parti mantengono le loro licenze (THIRD_PARTY.md)."));
            crate::panels::hint(ui, &tr!("In caso di errore il rapporto è in {}", "After an error the report is in {}", crate::crash::log_dir().display()));
        });
        self.about_open = open;
    }

    // ---- file I/O ----------------------------------------------------------------------

    /// One file operation at a time: a second one would finish out of order and clear the busy
    /// indicator of the first. Says so in the status line and returns true while one runs.
    fn busy_with_io(&mut self) -> bool {
        let busy = self.io_busy.is_some();
        if busy {
            self.status = tl("Un'altra operazione sui file è in corso: attendi che finisca.").to_string();
        }
        busy
    }

    fn open_dialog(&mut self, ctx: &egui::Context) {
        if self.busy_with_io() {
            return;
        }
        let picked = crate::dialog::FileDialog::new()
            .set_title(tl("Apri immagine"))
            .add_filter(tl("Immagini astronomiche e mosaici"), &[ps_io::READ_EXTENSIONS, &[MOSAIC_EXTENSION]].concat())
            .pick_files()
            .unwrap_or_default();
        self.open_paths(ctx, picked);
    }

    /// One image (or saved mosaic) opens; several ask what to do with them (mosaic, derotated
    /// sum, RGB from monochrome channels…) rather than guess.
    fn open_paths(&mut self, ctx: &egui::Context, mut paths: Vec<PathBuf>) {
        if paths.len() > 1 {
            paths.retain(|p| !p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(MOSAIC_EXTENSION)));
        }
        match paths.len() {
            0 => {}
            1 => self.load(ctx, paths.remove(0)),
            _ => self.choose_editor.start(ctx, paths),
        }
    }

    /// What the user chose to do with several images.
    fn do_choice(&mut self, ctx: &egui::Context, choice: Choice, paths: Vec<PathBuf>, body: Option<ps_core::capture::Body>) {
        match choice {
            Choice::Mosaic => self.mosaic_editor.start(ctx, paths),
            Choice::Derotation => self.derotation_editor.start(ctx, paths, body.filter(|b| crate::derotation_ui::derotates(*b))),
            Choice::MonoRgb => self.mono_rgb_editor.start(ctx, paths, false, body),
            Choice::MonoRgbDerotated => self.mono_rgb_editor.start(ctx, paths, true, body),
            Choice::Animation => self.animation_editor.start(ctx, paths, body),
        }
    }

    /// Several captures of one subject: the window that turns them into a GIF or a video.
    fn new_animation_dialog(&mut self, ctx: &egui::Context) {
        if let Some(paths) = crate::dialog::FileDialog::new().set_title(tr!("Riprese per l'animazione", "Shots for the animation")).add_filter(tl("Immagini"), ps_io::READ_EXTENSIONS).pick_files()
            && !paths.is_empty()
        {
            let body = crate::derotation_ui::named_body(&paths);
            self.animation_editor.start(ctx, paths, body);
        }
    }

    /// Pick the monochrome channels, then the window that assigns and checks them.
    fn new_mono_rgb_dialog(&mut self, ctx: &egui::Context, derotated: bool) {
        if let Some(paths) = crate::dialog::FileDialog::new().set_title(tl("Immagini monocromatiche (R, G, B ed eventualmente L)")).add_filter(tl("Immagini"), ps_io::READ_EXTENSIONS).pick_files()
            && !paths.is_empty()
        {
            let body = crate::derotation_ui::named_body(&paths).filter(|b| crate::derotation_ui::derotates(*b));
            self.mono_rgb_editor.start(ctx, paths, derotated, body);
        }
    }

    /// Merge the channels at full size off the UI thread, then open the result like a file.
    fn create_mono_rgb(&mut self, ctx: &egui::Context, job: MonoJob) {
        if self.busy_with_io() {
            return;
        }
        self.io_busy = Some(if job.derotate.is_some() {
            tr!("Derotazione e unione di {} canali…", "Derotating and merging {} channels…", job.items.len())
        } else {
            tr!("Unione di {} canali…", "Merging {} channels…", job.items.len())
        });
        let (tx, ctx) = (self.events_tx.clone(), ctx.clone());
        let max_side = ctx.input(|i| i.max_texture_side);
        std::thread::spawn(move || {
            let start = Instant::now();
            let event = match job.run() {
                Ok(out) => {
                    let label = if out.lrgb { "LRGB" } else { "RGB" };
                    let derotated = Derotated { frames: job.items.len(), moment: out.moment, mono: Some(label), rotated: job.derotate.is_some() };
                    let image = out.image;
                    let (display, analysis) = rayon::join(|| worker::display_image(&image, max_side), || Box::new(analyze(&image)));
                    Event::Loaded { path: job.path.clone(), image, display, analysis, elapsed: start.elapsed(), mosaic: None, keep_settings: false, derotated: Some(derotated) }
                }
                Err(e) => Event::Failed(e.message()),
            };
            tx.send(event).ok();
            ctx.request_repaint();
        });
    }

    /// Several captures of Jupiter or Saturn, whatever their names say.
    fn new_derotation_dialog(&mut self, ctx: &egui::Context) {
        if let Some(paths) = crate::dialog::FileDialog::new().set_title(tl("Riprese di Giove o Saturno da deruotare")).add_filter(tl("Immagini"), ps_io::READ_EXTENSIONS).pick_files()
            && !paths.is_empty()
        {
            let body = crate::derotation_ui::named_body(&paths).filter(|b| crate::derotation_ui::derotates(*b));
            self.derotation_editor.start(ctx, paths, body);
        }
    }

    /// Derotate and sum at full size off the UI thread, then open the result like a file.
    fn create_derotation(&mut self, ctx: &egui::Context, job: StackJob) {
        if self.busy_with_io() {
            return;
        }
        self.io_busy = Some(tr!("Derotazione e somma di {} immagini…", "Derotating and summing {} images…", job.images.len()));
        let (tx, ctx) = (self.events_tx.clone(), ctx.clone());
        let max_side = ctx.input(|i| i.max_texture_side);
        std::thread::spawn(move || {
            let start = Instant::now();
            let event = match job.run(1.0) {
                Some((image, _)) => {
                    let (display, analysis) = rayon::join(|| worker::display_image(&image, max_side), || Box::new(analyze(&image)));
                    Event::Loaded { path: job.path.clone(), image, display, analysis, elapsed: start.elapsed(), mosaic: None, keep_settings: false, derotated: Some(job.derotated.clone()) }
                }
                None => Event::Failed(tl("Nessuna ripresa da sommare").into()),
            };
            tx.send(event).ok();
            ctx.request_repaint();
        });
    }

    /// Several images: a new mosaic of them.
    fn new_mosaic_dialog(&mut self, ctx: &egui::Context) {
        if let Some(paths) = crate::dialog::FileDialog::new().set_title(tl("Immagini da unire in un mosaico")).add_filter(tl("Immagini"), ps_io::READ_EXTENSIONS).pick_files()
            && !paths.is_empty()
        {
            self.mosaic_editor.start(ctx, paths);
        }
    }

    /// Join the frames at full size off the UI thread, then open the result like a file.
    fn create_mosaic(&mut self, ctx: &egui::Context, mosaic: Mosaic, images: Vec<Option<Arc<Image>>>, keep_settings: bool) {
        if self.busy_with_io() {
            return;
        }
        let Some(first) = mosaic.tiles.iter().zip(&images).find(|(t, i)| t.enabled && i.is_some()).map(|(t, _)| t.path.clone()) else { return };
        self.io_busy = Some(tr!("Creazione del mosaico di {} immagini…", "Building the mosaic of {} images…", mosaic.tiles.iter().filter(|t| t.enabled).count()));
        let (tx, ctx) = (self.events_tx.clone(), ctx.clone());
        let max_side = ctx.input(|i| i.max_texture_side);
        std::thread::spawn(move || {
            let start = Instant::now();
            let used: Vec<Option<&Image>> = mosaic.tiles.iter().zip(&images).map(|(t, i)| i.as_deref().filter(|_| t.enabled)).collect();
            let placements: Vec<Option<Placement>> = mosaic.tiles.iter().map(|t| t.placement).collect();
            let gains: Vec<[f32; 3]> = mosaic.tiles.iter().map(|t| t.gain).collect();
            let joined = ps_core::mosaic::compose(&used, &placements, &gains, &mosaic.params, 1.0);
            drop(used);
            drop(images);
            let event = match joined {
                Some(image) => {
                    let (display, analysis) = rayon::join(|| worker::display_image(&image, max_side), || Box::new(analyze(&image)));
                    Event::Loaded { path: first, image, display, analysis, elapsed: start.elapsed(), mosaic: Some(Box::new(mosaic)), keep_settings, derotated: None }
                }
                None => Event::Failed(tl("Nessuna immagine allineata da unire").into()),
            };
            tx.send(event).ok();
            ctx.request_repaint();
        });
    }

    /// A saved mosaic: its frames joined as they were.
    fn open_saved_mosaic(&mut self, ctx: &egui::Context, path: &Path) {
        let mosaic = match crate::mosaic_ui::read(path) {
            Ok(m) => m,
            Err(e) => {
                self.status = tr!("Impossibile aprire {}: {e}", "Cannot open {}: {e}", file_name(path));
                return;
            }
        };
        self.io_busy = Some(tr!("Apertura del mosaico {}…", "Opening mosaic {}…", file_name(path)));
        let (tx, ctx) = (self.events_tx.clone(), ctx.clone());
        let (frames_tx, frames_rx) = channel();
        std::thread::spawn(move || {
            let images: Result<Vec<Option<Arc<Image>>>, String> =
                mosaic.tiles.iter().map(|t| if t.enabled { ps_io::load(&t.path).map(|i| Some(Arc::new(i))).map_err(|e| format!("{}: {e}", file_name(&t.path))) } else { Ok(None) }).collect();
            match images {
                Ok(images) => {
                    frames_tx.send((mosaic, images)).ok();
                }
                Err(e) => {
                    tx.send(Event::Failed(tr!("Mosaico incompleto: {e}", "Incomplete mosaic: {e}"))).ok();
                }
            }
            ctx.request_repaint();
        });
        self.pending_mosaic = Some(frames_rx);
    }

    /// Decode, analyse and prepare the display texture off the UI thread.
    fn load(&mut self, ctx: &egui::Context, path: PathBuf) {
        if self.busy_with_io() {
            return;
        }
        if path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(MOSAIC_EXTENSION)) {
            self.open_saved_mosaic(ctx, &path);
            return;
        }
        self.io_busy = Some(tr!("Apertura di {}…", "Opening {}…", file_name(&path)));
        let tx = self.events_tx.clone();
        let ctx = ctx.clone();
        let max_side = ctx.input(|i| i.max_texture_side);
        std::thread::spawn(move || {
            let start = Instant::now();
            let event = match ps_io::load(&path) {
                Ok(image) => {
                    let (display, analysis) = rayon::join(|| worker::display_image(&image, max_side), || Box::new(analyze(&image)));
                    Event::Loaded { path, image, display, analysis, elapsed: start.elapsed(), mosaic: None, keep_settings: false, derotated: None }
                }
                Err(e) => Event::Failed(tr!("Impossibile aprire {}: {e}", "Cannot open {}: {e}", file_name(&path))),
            };
            tx.send(event).ok();
            ctx.request_repaint();
        });
    }

    /// Load a mono image as LRGB luminance and register it onto the colour image.
    fn load_luminance(&mut self, ctx: &egui::Context) {
        if self.busy_with_io() {
            return;
        }
        let Some(doc) = &self.document else { return };
        let Some(path) = crate::dialog::FileDialog::new().set_title(tl("Immagine mono per la luminanza")).add_filter(tl("Immagini"), ps_io::READ_EXTENSIONS).pick_file() else {
            return;
        };
        self.io_busy = Some(tr!("Caricamento di {}…", "Loading {}…", file_name(&path)));
        let original = Arc::clone(&doc.original);
        let tx = self.events_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let event = match ps_io::load(&path) {
                Ok(img) if img.width() != original.width() || img.height() != original.height() => Event::Failed(tr!("La luminanza deve avere le stesse dimensioni dell'immagine ({}×{}), non {}×{}", "The luminance must be the same size as the image ({}×{}), not {}×{}",
                    original.width(),
                    original.height(),
                    img.width(),
                    img.height()
                )),
                Ok(img) => {
                    let (w, h) = (img.width(), img.height());
                    let plane = working_planes(&img, ColorMode::Luminance).swap_remove(0);
                    let reference = working_planes(&original, ColorMode::Luminance).swap_remove(0);
                    let shift = ps_core::align::measure_shift(&reference, &plane, w, h);
                    Event::LuminanceLoaded { name: file_name(&path), plane: ps_core::align::shift(&plane, w, h, shift) }
                }
                Err(e) => Event::Failed(tr!("Impossibile aprire {}: {e}", "Cannot open {}: {e}", file_name(&path))),
            };
            tx.send(event).ok();
            ctx.request_repaint();
        });
    }

    /// "Elabora una serie…": choose the files, then what to do with them.
    fn batch_pick(&mut self) {
        let picked = crate::dialog::FileDialog::new().set_title(tl("Immagini da elaborare")).add_filter(tl("Immagini"), &["tif", "tiff", "png", "jpg", "jpeg", "fit", "fits"]).pick_files();
        let Some(files) = picked.filter(|f| !f.is_empty()) else { return };
        let dir = files[0].parent().map(|p| p.join("elaborate"));
        self.batch_dialog = Some(BatchDialog { files, dir, mode: if self.document.is_some() { crate::batch::Mode::Current } else { crate::batch::Mode::Preprocessed } });
    }

    /// The series window: what to do, where to save; then the progress.
    fn batch_windows(&mut self, ctx: &egui::Context) {
        let mut start = false;
        let mut close = false;
        if let Some(d) = &mut self.batch_dialog {
            let has_doc = self.document.is_some();
            crate::widgets::window(ctx, tl("Elabora una serie")).collapsible(false).default_width(440.0).show(ctx, |ui| {
                ui.label(tr!("{} immagini scelte.", "{} images chosen.", d.files.len()));
                ui.add_space(theme::space::S);
                ui.label(caps(tl("IMPOSTAZIONI")));
                ui.radio_value(&mut d.mode, crate::batch::Mode::Preprocessed, tl("Preelaborate (nitidezza, rumore, colore, livelli, calcolate per ognuna)"));
                ui.add_enabled_ui(has_doc, |ui| {
                    ui.radio_value(&mut d.mode, crate::batch::Mode::Current, tl("Quelle dell'immagine aperta (senza maschere e pennellate)"));
                });
                ui.add_space(theme::space::S);
                ui.label(caps(tl("CARTELLA DI DESTINAZIONE")));
                ui.horizontal(|ui| {
                    ui.label(d.dir.as_ref().map_or(String::new(), |p| p.display().to_string()));
                    if ui.button(tl("Scegli…")).clicked()
                        && let Some(p) = crate::dialog::FileDialog::new().set_title(tl("Cartella di destinazione")).pick_folder()
                    {
                        d.dir = Some(p);
                    }
                });
                ui.label(RichText::new(tr!("Formato {} e suffisso «{}», come l'esportazione rapida; i file esistenti non vengono sovrascritti.", "Format {} and suffix “{}”, as in quick export; existing files are never overwritten.", self.prefs.export_format, self.prefs.export_suffix)).small().color(TEXT_WEAK));
                ui.add_space(theme::space::M);
                ui.horizontal(|ui| {
                    if ui.add_enabled(d.dir.is_some(), egui::Button::new(tl("Avvia"))).clicked() {
                        start = true;
                    }
                    if ui.button(tl("Annulla")).clicked() {
                        close = true;
                    }
                });
            });
        }
        if start && let Some(d) = self.batch_dialog.take() {
            let job = crate::batch::Job {
                files: d.files,
                dir: d.dir.expect("start needs a folder"),
                extension: self.prefs.export_format.clone(),
                suffix: self.prefs.export_suffix.clone(),
                mode: d.mode,
                strength: self.prefs.auto_strength,
                scales: self.prefs.wavelet_scales,
                setup: self.prefs.favorite_setup_value(),
                current: self.pipeline.clone(),
            };
            let _ = std::fs::create_dir_all(&job.dir);
            self.batch = Some(crate::batch::Run::start(ctx, job));
        }
        if close {
            self.batch_dialog = None;
        }
        let mut dismiss = false;
        if let Some(run) = &mut self.batch {
            run.poll();
            let mut open = true;
            crate::widgets::window(ctx, tl("Elaborazione della serie")).collapsible(false).default_width(420.0).open(&mut open).show(ctx, |ui| {
                let shown = run.done.min(run.total);
                ui.add(egui::ProgressBar::new(shown as f32 / run.total.max(1) as f32).text(format!("{shown} / {}", run.total)));
                if run.finished {
                    let ok = run.done - run.failed.len();
                    ui.label(if run.cancelled { tr!("Interrotta: {ok} immagini salvate.", "Stopped: {ok} images saved.") } else { tr!("Fatto: {ok} immagini salvate.", "Done: {ok} images saved.") });
                    for why in &run.failed {
                        ui.label(RichText::new(why).small().color(theme::DANGER));
                    }
                    ui.horizontal(|ui| {
                        if ui.button(tl("Mostra la cartella")).clicked() {
                            reveal(run.last.as_deref().unwrap_or(run.dir()));
                        }
                        if ui.button(tl("Chiudi")).clicked() {
                            dismiss = true;
                        }
                    });
                } else {
                    ui.label(tr!("In corso: {}", "Working on: {}", run.current));
                    if ui.button(tl("Interrompi")).clicked() {
                        run.cancel();
                    }
                    ctx.request_repaint_after(Duration::from_millis(300));
                }
            });
            dismiss |= !open && run.finished;
            if !open && !run.finished {
                run.cancel();
            }
        }
        if dismiss {
            self.batch = None;
        }
    }

    /// Name of exported files (without extension): the base name and the chosen keywords.
    fn export_stem(&self) -> String {
        let Some(doc) = &self.document else { return tl("immagine").into() };
        let mut modules: Vec<&'static str> = Stage::ALL.into_iter().filter(|&s| self.stage_active(s)).map(crate::naming::module_tag).collect();
        if self.resize_params().is_some() {
            modules.push(if self.prefs.resize.scale >= 1.0 { tl("ingrandita") } else { tl("ridotta") });
        }
        let source = crate::naming::NameSource { base: &doc.export_name, original: &doc.path, kind: doc.kind, modules, suffix: &self.prefs.export_suffix };
        crate::naming::stem(&source, &self.prefs.export_keywords)
    }

    /// "Esporta come…": pick the file; its format becomes the one quick export uses.
    fn export_dialog(&mut self, ctx: &egui::Context) {
        if self.busy_with_io() {
            return;
        }
        let Some(doc) = &self.document else { return };
        let stem = self.export_stem();
        let picked = crate::dialog::FileDialog::new()
            .set_title(tl("Esporta immagine elaborata"))
            .set_directory(doc.path.parent().unwrap_or(Path::new(".")))
            .set_file_name(format!("{stem}.{}", self.prefs.export_format))
            .add_filter(tl("TIFF 16 bit"), &["tif", "tiff"])
            .add_filter(tl("PNG 16 bit"), &["png"])
            .add_filter("JPEG", &["jpg", "jpeg"])
            .save_file();
        let Some(path) = picked else { return };
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            self.prefs.export_format = ext.to_ascii_lowercase();
        }
        self.export_to(ctx, path);
    }

    /// Export next to the original under [`Self::export_stem`], never overwriting a file.
    fn quick_export(&mut self, ctx: &egui::Context) {
        if self.busy_with_io() {
            return;
        }
        let Some(doc) = &self.document else { return };
        let (dir, stem) = (doc.path.parent().map(Path::to_path_buf).unwrap_or_default(), self.export_stem());
        let ext = self.prefs.export_format.clone();
        let path = (1..)
            .map(|k| dir.join(if k == 1 { format!("{stem}.{ext}") } else { format!("{stem}-{k}.{ext}") }))
            .find(|p| !p.exists())
            .expect("some name is free");
        self.export_to(ctx, path);
    }

    /// Keep the «Annotazione» layer in step with the Identifica tab: made when switched on,
    /// remade (in the background) whenever the names, the options or the identification change,
    /// dropped when switched off.
    fn update_annotation(&mut self, ctx: &egui::Context) {
        let Some(doc) = &mut self.document else { return };
        if !self.identify.annotate {
            doc.annotation = None;
            self.annotating = None;
            return;
        }
        if let Some((key, source_gen, rx)) = &self.annotating
            && let Ok(planes) = rx.try_recv()
        {
            doc.annotation = Some(Annotation { source_gen: *source_gen, key: *key, planes });
            self.annotating = None;
            ctx.request_repaint();
        }
        let doc = self.document.as_ref().expect("checked above");
        let Some(plan) = self.identify.plan(&identify_input(doc), &self.prefs.label_options()) else { return };
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (format!("{plan:?}{:?}", self.prefs.label_options()), doc.source_gen).hash(&mut hasher);
        let key = hasher.finish();
        let current = doc.annotation.as_ref().is_some_and(|a| a.key == key && a.source_gen == doc.source_gen) || self.annotating.as_ref().is_some_and(|(k, g, _)| *k == key && *g == doc.source_gen);
        if current {
            return;
        }
        let (w, h, depth, source_gen) = (doc.source.width(), doc.source.height(), doc.source.source_bit_depth, doc.source_gen);
        let opt = self.prefs.label_options();
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let flat = |v: f32| Image::new(w, h, vec![vec![v; w * h]; 3], depth);
            // The same marks over black and over white give the colour (premultiplied) and the coverage.
            let (on_black, on_white) = (crate::annotate::burn_in(&flat(0.0), &plan, &opt), crate::annotate::burn_in(&flat(1.0), &plan, &opt));
            let mut coverage = vec![0.0f32; w * h];
            for (i, a) in coverage.iter_mut().enumerate() {
                let lost: f32 = (0..3).map(|c| on_white.channels()[c][i] - on_black.channels()[c][i]).sum::<f32>() / 3.0;
                *a = (1.0 - lost).clamp(0.0, 1.0);
            }
            let plane = |v: Vec<f32>| (Arc::new(v), next_generation());
            let [r, g, b] = [0, 1, 2].map(|c| plane(on_black.channels()[c].iter().zip(&coverage).map(|(p, a)| p.min(*a)).collect()));
            tx.send([r, g, b, plane(coverage)]).ok();
            ctx.request_repaint();
        });
        self.annotating = Some((key, source_gen, rx));
    }

    /// The enlargement export applies, if switched on.
    /// The pipeline actually processed: modules not shown are off, and the things only the capture
    /// knows are filled in (Jupiter's tilt towards us, from the time, for the photometric colour).
    fn processed_pipeline(&self, doc: &Document) -> Pipeline {
        let mut p = self.prefs.effective(&self.pipeline, doc.kind, doc.source.is_mono());
        let when = self.identify.chosen_time().or(doc.capture.info.time);
        // The photometric colour: the subject's own references, its tilt from the time (Saturn's
        // is otherwise measured on its rings).
        let body = doc.capture.body;
        if let Some(b) = body
            && p.photocolor.params.reference.body() != b
            && let Some(first) = ps_core::photocolor::Reference::for_body(b).first()
        {
            p.photocolor.params.reference = *first;
        }
        p.photocolor.params.tilt = match body {
            Some(ps_core::capture::Body::Saturn) => when.and_then(|t| ps_core::identify::saturn_ring_tilt(&t)).unwrap_or(0.0) as f32,
            _ => when.map_or(0.0, |t| ps_core::moons::jupiter_tilt(&t)) as f32,
        };
        p
    }

    fn resize_params(&self) -> Option<ps_core::resample::ResizeParams> {
        let doc = self.document.as_ref()?;
        let sun = doc.capture.body == Some(ps_core::capture::Body::Sun);
        self.prefs.resize.params(doc.source.width(), doc.source.height(), doc.kind, sun)
    }

    fn export_to(&mut self, ctx: &egui::Context, path: PathBuf) {
        if self.busy_with_io() {
            return;
        }
        let resize = self.resize_params();
        let Some(doc) = &self.document else { return };
        self.exporting = Some(self.pipeline.clone());
        let overlay = doc.overlay();
        let pipeline = self.processed_pipeline(doc);
        // Usually already rendered (in the foreground or in the background while on a ROI).
        let ready = self.full_output.as_ref().filter(|o| o.source_gen == doc.source_gen && o.pipeline == pipeline && o.overlay_gen == overlay.as_ref().map(|p| p[3].1)).map(|o| Arc::clone(&o.image));
        let source = Arc::clone(&doc.source);
        let luminance = doc.luminance.as_ref().map(|l| l.cropped.clone());
        let bxt = self.prefs.bxt_executable();

        self.io_busy = Some(tr!("Esportazione di {}…", "Exporting {}…", file_name(&path)));
        let tx = self.events_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let start = Instant::now();
            let image = ready.unwrap_or_else(|| {
                let masks = ps_core::pipeline::masks_for(&pipeline, &source);
                let aux = Aux { masks: &masks, luminance, overlay, bxt_executable: bxt.as_deref(), ..Default::default() };
                let out = Processor::default().run(source, next_generation(), &pipeline, &aux, &|| false, &|_, _| {});
                out.expect("never cancelled").image
            });
            let image = match &resize {
                Some(r) => Arc::new(ps_core::resample::resize(&image, r)),
                None => image,
            };
            let event = match ps_io::save(&path, &image) {
                Ok(()) => Event::Exported { path, elapsed: start.elapsed() },
                Err(e) => Event::Failed(tr!("Esportazione fallita: {e}", "Export failed: {e}")),
            };
            tx.send(event).ok();
            ctx.request_repaint();
        });
    }

    fn save_preset(&mut self) {
        let picked = crate::dialog::FileDialog::new().set_title(tl("Salva impostazioni")).set_file_name("impostazioni.pspipe").add_filter(tl("Impostazioni"), &["pspipe"]).save_file();
        if let Some(path) = picked {
            let json = serde_json::to_string_pretty(&self.pipeline).expect("pipeline serialises");
            self.status = match std::fs::write(&path, json) {
                Ok(()) => {
                    self.clean_pipeline = self.pipeline.clone();
                    tr!("impostazioni salvate in {}", "settings saved to {}", file_name(&path))
                }
                Err(e) => tr!("salvataggio fallito: {e}", "saving failed: {e}"),
            };
        }
    }

    fn load_preset(&mut self) {
        let picked = crate::dialog::FileDialog::new().set_title(tl("Carica impostazioni")).add_filter(tl("Impostazioni"), &["pspipe"]).pick_file();
        let Some(path) = picked else { return };
        let parsed = std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str::<Pipeline>(&s).map_err(|e| e.to_string()));
        self.status = match parsed {
            Ok(p) => {
                self.pipeline = p;
                tr!("impostazioni {} caricate", "settings {} loaded", file_name(&path))
            }
            Err(e) => tr!("file di impostazioni non valido: {e}", "invalid settings file: {e}"),
        };
    }

    // ---- automatic settings ------------------------------------------------------------

    /// Every automatic setting for sharpening, noise and alignment; tone, colour, layers and
    /// Mineral Moon are left as they are.
    fn auto_all(&self, kind: ImageKind, analysis: &Analysis) -> Pipeline {
        let auto = auto_pipeline_with(kind, analysis, self.prefs.auto_strength, self.prefs.wavelet_scales);
        Pipeline {
            align: auto.align,
            bxt: Module { enabled: self.pipeline.bxt.enabled, params: auto.bxt.params },
            deconv: auto.deconv,
            denoise: auto.denoise,
            wavelet: auto.wavelet,
            smart: auto.smart,
            dering: auto.dering,
            ..self.pipeline.clone()
        }
    }

    /// The type was changed by hand: re-derive the structural settings, keep the amounts.
    fn apply_kind(&mut self, kind: ImageKind) {
        let Some(doc) = &mut self.document else { return };
        doc.kind = kind;
        self.pipeline = restructure(kind, &doc.analysis, &self.pipeline);
    }

    /// What a freshly opened image starts from, and what a module's reset button goes back to.
    fn starting_point(&self, kind: ImageKind, analysis: &Analysis) -> Pipeline {
        initial_pipeline(kind, analysis, self.prefs.wavelet_scales)
    }

    /// Back to the starting point: nothing applied, the structure still the image's (layers stay).
    fn clear_auto(&mut self) {
        let Some(doc) = &self.document else { return };
        let start = self.starting_point(doc.kind, &doc.analysis);
        let layers = std::mem::take(&mut self.pipeline.layers);
        self.pipeline = Pipeline { layers, ..start };
    }

    /// A new image was just opened: the first-open question the first time, else the chosen
    /// way (pre-processed or raw), then the question about the subject if it is unknown.
    fn opened_fresh(&mut self, ctx: &egui::Context) {
        let Some(doc) = &self.document else { return };
        if !self.prefs.preprocess_asked {
            self.preprocess_dialog = Some(crate::preprocess::Dialog::start(ctx, Arc::clone(&doc.source), doc.analysis.clone(), doc.kind, doc.capture.body, self.prefs.wavelet_scales));
        } else if self.prefs.auto_settings {
            self.apply_preprocess();
        }
        let doc = self.document.as_ref();
        self.subject_prompt = doc.is_some_and(|d| d.capture.body.is_none() && d.kind != ImageKind::Saturn);
    }

    /// The pre-processed settings for the open image (layers stay); the identification is
    /// switched on when there is a subject and a time to compute from.
    fn apply_preprocess(&mut self) {
        let Some(doc) = &mut self.document else { return };
        let pre = crate::preprocess::pipeline(doc.kind, &doc.analysis, &doc.source, doc.capture.body, self.prefs.auto_strength, self.prefs.wavelet_scales);
        self.pipeline = Pipeline { layers: std::mem::take(&mut self.pipeline.layers), ..pre };
        doc.preprocessed = Some(self.pipeline.clone());
        if doc.capture.body.is_some() && doc.capture.info.time.is_some() {
            self.identify.switch_on();
            self.auto_identify = Some((Instant::now(), None));
        }
        self.levels_pending = self.pipeline.tone.enabled;
    }

    /// Whether the settings are the pre-processed ones, the raw starting point, or neither.
    fn opening_state(&self) -> panels::Opening {
        let Some(doc) = &self.document else { return panels::Opening::Raw };
        let bare = |p: &Pipeline| Pipeline { layers: Vec::new(), ..p.clone() };
        if doc.preprocessed.as_ref().is_some_and(|p| bare(p) == bare(&self.pipeline)) {
            panels::Opening::Preprocessed
        } else if bare(&self.starting_point(doc.kind, &doc.analysis)) == bare(&self.pipeline) {
            panels::Opening::Raw
        } else {
            panels::Opening::Modified
        }
    }

    /// The subject was set by hand (or read from a log): the sharpening network for it, then
    /// the settings that depend on it. Settings that were the automatic ones follow.
    fn refresh_subject(&mut self, follows: (bool, bool)) {
        let (was_pre, _) = follows;
        let Some(doc) = &mut self.document else { return };
        let suggestion = doc.capture.body.map(|b| ps_core::curve_net::suggestion(&doc.original, &doc.analysis, b));
        ps_core::log::info(
            "soggetto",
            &format!("scelto {:?}: rete nitidezza {}", doc.capture.body, match &suggestion {
                Some(Ok(s)) => format!("picco {:.4} forza {:.2}", s.cutoff, s.strength),
                Some(Err(why)) => format!("nessuna ({why})"),
                None => "nessun soggetto".into(),
            }),
        );
        doc.analysis.curve_net = suggestion.and_then(Result::ok);
        self.pipeline = restructure(doc.kind, &doc.analysis, &self.pipeline);
        if was_pre {
            self.apply_preprocess();
        } 
    }

    /// Whether the settings are still the pre-processed ones, or the automatic ones.
    fn follows_auto(&self) -> (bool, bool) {
        (self.document.as_ref().is_some_and(|d| d.preprocessed.as_ref() == Some(&self.pipeline)), !self.is_customised())
    }

    /// The Subject for `body`: the image type it is processed as.
    fn subject_for(&self, body: ps_core::capture::Body) -> panels::Subject {
        use ps_core::capture::Body;
        let kind = match body {
            Body::Moon | Body::Sun => self.document.as_ref().map(|d| d.kind).filter(|k| matches!(k, ImageKind::FullDisk | ImageKind::Surface)).unwrap_or(ImageKind::FullDisk),
            _ => panels::PLANETS.iter().find(|(b, ..)| *b == body).map_or(ImageKind::Planet, |(_, k, _)| *k),
        };
        panels::Subject { kind, body: Some(body) }
    }

    fn instrument_action(&mut self, action: InstrumentAction) {
        match action {
            InstrumentAction::SetupChanged => {
                self.equipment_changed();
            }
            InstrumentAction::BodyChanged => {
                let follows = self.follows_auto();
                if let Some(doc) = &mut self.document {
                    doc.capture.measure(&doc.original, &doc.analysis);
                }
                self.refresh_subject(follows);
            }
            InstrumentAction::LoadLog => {
                let picked = crate::dialog::FileDialog::new().set_title(tl("Log di ripresa")).add_filter("Log (FireCapture, SharpCap, AutoStakkert)", &["txt", "log", "as3", "as4"]).pick_file();
                if let Some(path) = picked {
                    self.read_log(&path);
                }
            }
        }
    }

    /// The equipment selector of the top bar: the saved configurations, the details.
    fn instrument_menu(&mut self, ui: &mut egui::Ui) {
        let Some(doc) = &self.document else { return };
        let label = setup_label(&doc.setup);
        let button = egui::Button::new((ICON_PHOTO_CAMERA.rich_text().color(ACCENT), RichText::new(label).color(TEXT))).corner_radius(egui::CornerRadius::same(theme::radius::L));
        let mut picked = None;
        let (mut details, mut manage) = (false, false);
        let current = doc.setup.clone();
        let saved = self.prefs.saved_setups.clone();
        let favorite = self.prefs.favorite_setup.clone();
        egui::containers::menu::MenuButton::from_button(button).ui(ui, |ui| {
            ui.set_min_width(240.0);
            ui.label(caps(tl("CONFIGURAZIONI")));
            if ui.add(egui::Button::selectable(current == Setup::default(), tl("Nessuno strumento"))).clicked() {
                picked = Some(Setup::default());
            }
            for s in saved {
                let star = if favorite.as_ref() == Some(&s.name) { ICON_STAR.rich_text().color(ACCENT) } else { RichText::new("") };
                if ui.add(egui::Button::selectable(s.setup == current, (star, s.name.clone()))).on_hover_text(setup_label(&s.setup)).clicked() {
                    picked = Some(s.setup);
                }
            }
            ui.separator();
            if ui.button((ICON_TUNE.rich_text(), tl("Dettagli strumento…"))).on_hover_text(tl("Oggetto, telescopio, camera, Barlow, filtro, log di ripresa e diagnosi")).clicked() {
                details = true;
            }
            if ui.button(tl("Salva e gestisci configurazioni…")).on_hover_text(tl("Salva lo strumento attuale con un nome, rinominalo, scegli il preferito")).clicked() {
                manage = true;
            }
        })
        .0
        .on_hover_text(tl("Lo strumento dell'immagine: scegli una configurazione salvata o apri i dettagli"));
        if let Some(setup) = picked
            && let Some(doc) = &mut self.document
        {
            doc.setup = setup;
            self.instrument_action(InstrumentAction::SetupChanged);
        }
        if details {
            self.instrument_open = true;
        }
        if manage {
            self.setups_open = true;
        }
    }

    /// Saved equipment configurations: save the current one, rename, pick the favourite (the one
    /// every new image starts with), use, update, delete.
    fn setups_window(&mut self, ctx: &egui::Context) {
        if !self.setups_open {
            return;
        }
        let mut open = true;
        let current = self.document.as_ref().map(|d| d.setup.clone());
        let mut use_it = None;
        crate::widgets::window(ctx, tl("Configurazioni dello strumento")).open(&mut open).default_width(460.0).show(ctx, |ui| {
            ui.label(RichText::new(tl("Il preferito (la stella) è lo strumento con cui si apre ogni nuova immagine; senza preferito non viene inserito nessuno strumento.")).small().color(TEXT_WEAK));
            ui.add_space(theme::space::S);
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.new_setup_name).desired_width(200.0).hint_text(tl("Nome, per esempio C8 + ASI462")));
                let name = self.new_setup_name.trim().to_string();
                let can = current.is_some() && !name.is_empty();
                if ui.add_enabled(can, egui::Button::new(tl("Salva lo strumento attuale"))).clicked()
                    && let Some(setup) = current.clone()
                {
                    match self.prefs.saved_setups.iter_mut().find(|s| s.name == name) {
                        Some(existing) => existing.setup = setup,
                        None => self.prefs.saved_setups.push(crate::prefs::SavedSetup { name, setup }),
                    }
                    self.new_setup_name.clear();
                }
            });
            ui.add_space(theme::space::S);
            if self.prefs.saved_setups.is_empty() {
                ui.label(RichText::new(tl("Nessuna configurazione salvata.")).color(TEXT_WEAK));
            }
            let mut remove = None;
            let mut renamed = None;
            let mut favorite = self.prefs.favorite_setup.clone();
            for (i, s) in self.prefs.saved_setups.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    let is_fav = favorite.as_ref() == Some(&s.name);
                    if ui.selectable_label(is_fav, ICON_STAR.rich_text().color(if is_fav { ACCENT } else { TEXT_WEAK })).on_hover_text(tl("Preferito: si apre con ogni nuova immagine")).clicked() {
                        favorite = if is_fav { None } else { Some(s.name.clone()) };
                    }
                    let before = s.name.clone();
                    ui.add(egui::TextEdit::singleline(&mut s.name).desired_width(170.0));
                    if s.name != before {
                        renamed = Some((before, s.name.clone()));
                    }
                    if ui.button(tl("Usa")).on_hover_text(setup_label(&s.setup)).clicked() {
                        use_it = Some(s.setup.clone());
                    }
                    if ui.add_enabled(current.as_ref().is_some_and(|c| *c != s.setup), egui::Button::new(tl("Aggiorna"))).on_hover_text(tl("Sostituisce questa configurazione con lo strumento attuale")).clicked()
                        && let Some(c) = &current
                    {
                        s.setup = c.clone();
                    }
                    if ui.button(ICON_CLOSE.rich_text()).on_hover_text(tl("Elimina")).clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some((from, to)) = renamed
                && favorite.as_ref() == Some(&from)
            {
                favorite = Some(to);
            }
            if let Some(i) = remove {
                let gone = self.prefs.saved_setups.remove(i);
                if favorite.as_ref() == Some(&gone.name) {
                    favorite = None;
                }
            }
            self.prefs.favorite_setup = favorite;
        });
        self.setups_open = open;
        if let Some(setup) = use_it
            && let Some(doc) = &mut self.document
        {
            doc.setup = setup;
            self.instrument_action(InstrumentAction::SetupChanged);
        }
    }

    /// The equipment and diagnosis, in a window of its own.
    fn instrument_window(&mut self, ctx: &egui::Context) {
        if !self.instrument_open {
            return;
        }
        let mut open = true;
        let mut actions = Vec::new();
        if let Some(doc) = &mut self.document {
            crate::widgets::window(ctx, tl("Strumento"))
                .open(&mut open)
                .default_pos(egui::pos2(200.0, 56.0))
                .default_width(440.0)
                .default_height(640.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        let view = InstrumentView { setup: &mut doc.setup, capture: &mut doc.capture, analysis: &doc.analysis, wavelet: &self.pipeline.wavelet.params };
                        instrument_body(ui, view, &mut actions);
                    });
                });
        } else {
            open = false;
        }
        self.instrument_open = open;
        for action in actions {
            self.instrument_action(action);
        }
    }

    /// A capture log for the open image: its equipment, subject and time.
    fn read_log(&mut self, path: &Path) {
        let follows = self.follows_auto();
        let Some(doc) = &mut self.document else { return };
        match std::fs::read_to_string(path) {
            Ok(text) => {
                doc.capture.add_log(path, &text);
                doc.capture.measure(&doc.original, &doc.analysis);
                doc.setup = Setup::from_capture(&doc.capture.info, &doc.setup);
                self.status = tr!("letto il log {}", "read log {}", file_name(path));
                self.refresh_subject(follows);
                self.equipment_changed();
            }
            Err(e) => self.status = tr!("impossibile leggere {}: {e}", "cannot read {}: {e}", file_name(path)),
        }
    }

    /// The optics follow the equipment, and the structural settings the optics.
    fn equipment_changed(&mut self) {
        let Some(doc) = &mut self.document else { return };
        doc.analysis.optics = doc.setup.optics();
        self.pipeline = restructure(doc.kind, &doc.analysis, &self.pipeline);
    }

    /// The wavelet scales were switched in the panel (from `from`): the transform for the new
    /// kind, the same sharpening carried over, and the choice kept for the next images.
    fn switch_scales(&mut self, from: ScaleMode) {
        let Some(doc) = &self.document else { return };
        let to = self.pipeline.wavelet.params.scale_mode;
        let current = WaveletParams { scale_mode: from, ..self.pipeline.wavelet.params.clone() };
        self.pipeline.wavelet.params = ps_core::preset::convert_scales(doc.kind, &doc.analysis, &current, to);
        self.prefs.wavelet_scales = to;
    }

    /// The reset button of one module's card: its settings go back to the starting point,
    /// its switch stays as it is.
    fn reset_module(&mut self, stage: Stage) {
        let Some(doc) = &self.document else { return };
        let start = self.starting_point(doc.kind, &doc.analysis);
        let p = &mut self.pipeline;
        match stage {
            Stage::Align => p.align.params = start.align.params,
            Stage::Bxt => p.bxt.params = start.bxt.params,
            Stage::Deconv => p.deconv.params = start.deconv.params,
            Stage::Denoise => p.denoise.params = start.denoise.params,
            Stage::Wavelet => p.wavelet.params = start.wavelet.params,
            Stage::Smart => p.smart.params = start.smart.params,
            Stage::Contrast => p.contrast.params = start.contrast.params,
            // The strokes are the user's drawing, not a setting ("Cancella" removes them).
            Stage::Dering => p.dering.params = DeringParams { strokes: std::mem::take(&mut p.dering.params.strokes), ..start.dering.params },
            Stage::Tone => p.tone.params = start.tone.params,
            Stage::Curves => p.curves.params = start.curves.params,
            Stage::PhotoColor => p.photocolor.params = start.photocolor.params,
            Stage::Color => p.color.params = start.color.params,
            Stage::Mineral => p.mineral.params = start.mineral.params,
            Stage::SunColor => p.suncolor.params = start.suncolor.params,
            // Masks, not cards: nothing to reset here.
            Stage::Layers | Stage::Rings | Stage::Moons => {}
        }
    }

    fn apply_auto_all(&mut self) {
        let Some(doc) = &self.document else { return };
        self.pipeline = self.auto_all(doc.kind, &doc.analysis);
    }

    /// The "automatic" button of one module's card.
    fn apply_auto_module(&mut self, stage: Stage) {
        let Some(doc) = &self.document else { return };
        let auto = auto_pipeline_with(doc.kind, &doc.analysis, self.prefs.auto_strength, self.prefs.wavelet_scales);
        let p = &mut self.pipeline;
        match stage {
            Stage::Align => p.align = Module::on(auto.align.params),
            Stage::Bxt => p.bxt = Module::on(auto.bxt.params),
            Stage::Deconv => p.deconv = auto.deconv,
            Stage::Denoise => p.denoise = auto.denoise,
            Stage::Wavelet => p.wavelet = auto.wavelet,
            Stage::Smart => p.smart = Module::on(ps_core::preset::auto_smart(doc.kind, &doc.analysis, self.prefs.auto_strength)),
            Stage::Contrast => {
                // A light unsharp mask at the scale of the blur.
                let radius = match auto.deconv.params.psf {
                    ps_core::psf::Psf::Gaussian { sigma } => sigma,
                    other => other.fwhm_px() / 2.355,
                };
                p.contrast = Module::on(ps_core::enhance::ContrastParams { amount: 0.4, radius: radius.max(0.8), threshold: 2.0 * doc.analysis.noise, ..Default::default() });
            }
            Stage::Tone => {
                let (black, white) = ps_core::enhance::auto_levels(&doc.source);
                let mut tone = p.tone.params.clone();
                (tone.current.black, tone.current.white) = (black, white);
                p.tone = Module::on(tone);
            }
            Stage::Color => {
                let balance = ps_core::preset::auto_color(doc.kind, &doc.source);
                p.color = Module::on(ps_core::adjust::ColorParams { balance, ..p.color.params.clone() });
            }
            // An S-curve: a little more contrast in the midtones, ends untouched.
            Stage::Curves => {
                let mut curves = p.curves.params.clone();
                curves.current.curve = vec![[0.0, 0.0], [0.25, 0.22], [0.75, 0.79], [1.0, 1.0]];
                p.curves = Module::on(curves);
            }
            // Jupiter's colours carried onto the Hubble map, in full.
            Stage::PhotoColor => p.photocolor = Module::on(ps_core::photocolor::PhotoColorParams::default()),
            Stage::Mineral => p.mineral = Module::on(Default::default()),
            // The yellow the Sun is usually pictured in, at full strength.
            Stage::SunColor => p.suncolor = Module::on(ps_core::suncolor::SunColorParams::default()),
            // The automatic settings keep the area touched up by hand.
            Stage::Dering => p.dering = Module::on(DeringParams { strength: 1.0, strokes: p.dering.params.strokes.clone(), width: p.dering.params.width, ..auto.dering.params }),
            // Masks, not cards: their recommended settings come back from the mask's editor.
            Stage::Layers | Stage::Rings | Stage::Moons => {}
        }
    }

    fn is_customised(&self) -> bool {
        self.document.as_ref().is_some_and(|doc| self.auto_all(doc.kind, &doc.analysis) != self.pipeline)
    }

    // ---- crop --------------------------------------------------------------------------

    fn start_crop(&mut self) {
        if let Some(doc) = &mut self.document {
            self.cropping = true;
            self.layer_tool = LayerTool::None;
            self.picking = None;
            doc.viewer.begin_crop();
            doc.viewer.straighten = doc.geometry.angle;
        }
    }

    fn finish_crop(&mut self, ctx: &egui::Context, action: CropAction) {
        // Orientation changes apply at once and keep the crop tool open on the new image.
        let geometry_change = |g: &mut Geometry| match action {
            CropAction::Rotate(1) => g.rotate_clockwise(),
            CropAction::Rotate(_) => g.rotate_counterclockwise(),
            CropAction::Flip { horizontal: true } => g.flip_horizontal = !g.flip_horizontal,
            CropAction::Flip { horizontal: false } => g.flip_vertical = !g.flip_vertical,
            CropAction::Straighten(angle) => g.angle = angle,
            CropAction::Apply | CropAction::Cancel => {}
        };
        if !matches!(action, CropAction::Apply | CropAction::Cancel) {
            if let Some(mut g) = self.document.as_ref().map(|d| d.geometry) {
                geometry_change(&mut g);
                self.set_geometry(ctx, g);
                self.start_crop();
            }
            return;
        }
        self.cropping = false;
        let Some(doc) = &mut self.document else { return };
        let Some(r) = doc.viewer.crop.take() else { return };
        if action == CropAction::Cancel {
            return;
        }
        let (x, y) = (r.min.x as usize, r.min.y as usize);
        let (w, h) = (r.width() as usize, r.height() as usize);
        if (w, h) == (doc.source.width(), doc.source.height()) {
            return;
        }
        let offset = (doc.crop_offset.0 + x, doc.crop_offset.1 + y);
        self.set_crop(ctx, offset, w, h);
    }

    fn remove_crop(&mut self, ctx: &egui::Context) {
        if let Some(doc) = &self.document {
            let (w, h) = (doc.oriented.width(), doc.oriented.height());
            self.set_crop(ctx, (0, 0), w, h);
        }
    }

    /// New orientation: the crop is reset (it referred to the old orientation).
    fn set_geometry(&mut self, ctx: &egui::Context, g: Geometry) {
        // The orientation referred to the old image.
        self.identify.orientation_reset();
        let Some(doc) = &mut self.document else { return };
        // Hand-drawn masks and brush strokes stay on the same pixels of the subject: from the
        // old cropped frame to the original, and from there to the new orientation.
        let (ow, oh, from, (cx, cy)) = (doc.original.width(), doc.original.height(), doc.geometry, doc.crop_offset);
        let carry = |p: [f32; 2]| geometry::remap(ow, oh, &from, &g, [p[0] + cx as f32, p[1] + cy as f32]);
        for layer in &mut self.pipeline.layers {
            layer.mask.remap(&carry);
        }
        ps_core::mask::remap_strokes(&mut self.pipeline.dering.params.strokes, &carry);
        doc.geometry = g;
        doc.oriented = Arc::new(geometry::apply(&doc.original, &g));
        doc.crop_offset = (0, 0);
        let (w, h) = (doc.oriented.width(), doc.oriented.height());
        self.set_crop(ctx, (0, 0), w, h);
    }

    fn rotate(&mut self, ctx: &egui::Context, change: impl FnOnce(&mut Geometry)) {
        if let Some(mut g) = self.document.as_ref().map(|d| d.geometry) {
            change(&mut g);
            self.set_geometry(ctx, g);
        }
    }

    /// Re-derive the working source from the original, shifting hand-drawn masks with it.
    fn set_crop(&mut self, ctx: &egui::Context, offset: (usize, usize), w: usize, h: usize) {
        let Some(doc) = &mut self.document else { return };
        let (dx, dy) = (doc.crop_offset.0 as f32 - offset.0 as f32, doc.crop_offset.1 as f32 - offset.1 as f32);
        doc.crop_offset = offset;
        doc.source = Arc::new(doc.oriented.crop(offset.0, offset.1, w, h));
        doc.source_gen = next_generation();
        let display = worker::display_image(&doc.source, ctx.input(|i| i.max_texture_side));
        doc.viewer = Viewer::new(ctx, Arc::new(display), w, h);
        doc.crop_luminance();
        for layer in &mut self.pipeline.layers {
            layer.mask.translate(dx, dy);
        }
        for p in self.pipeline.dering.params.strokes.iter_mut().flat_map(|s| s.points.iter_mut()) {
            (p[0], p[1]) = (p[0] + dx, p[1] + dy);
        }
        self.last_output = None;
        self.full_output = None;
        self.mask_thumbs.clear();
        self.timings.clear();
        if self.preview == PreviewMode::Roi
            && let Some(doc) = &mut self.document
        {
            doc.viewer.ensure_roi();
        }
    }

    // ---- per-frame plumbing ------------------------------------------------------------

    fn handle_events(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.events_rx.try_recv() {
            self.io_busy = None;
            match event {
                Event::Loaded { path, image, display, analysis, elapsed, mosaic, keep_settings, derotated } => {
                    let done = if mosaic.is_some() {
                        tl("mosaico pronto")
                    } else if let Some(d) = &derotated {
                        if d.mono.is_some() { tl("immagine a colori pronta") } else { tl("derotazione pronta") }
                    } else {
                        tl("aperta")
                    };
                    self.status = format!("{done} in {}", format_duration(elapsed));
                    let title = match (&mosaic, &derotated) {
                        (Some(_), _) => tr!("Mosaico da {}", "Mosaic from {}", file_name(&path)),
                        (None, Some(Derotated { mono: Some(label), frames, rotated, .. })) => {
                            if *rotated {
                                tr!("{label} derotato da {frames} immagini mono ({})", "Derotated {label} from {frames} mono images ({})", file_name(&path))
                            } else {
                                tr!("{label} da {frames} immagini mono ({})", "{label} from {frames} mono images ({})", file_name(&path))
                            }
                        }
                        (None, Some(d)) => tr!("Derotazione di {} riprese da {}", "Derotation of {} captures from {}", d.frames, file_name(&path)),
                        (None, None) => file_name(&path),
                    };
                    self.doc_title = title;
                    if mosaic.is_none() && derotated.is_none() {
                        self.prefs.remember_file(path.clone(), &display);
                    }
                    let viewer = Viewer::new(ctx, Arc::new(display), image.width(), image.height());
                    // An updated mosaic keeps its processing, its type and its export name.
                    let kept = self.document.as_ref().filter(|d| keep_settings && d.mosaic.is_some()).map(|d| (d.kind, d.export_name.clone()));
                    let kept_none = kept.is_none();
                    self.selected_layer = None;
                    (self.layer_tool, self.show_mask, self.picking, self.cropping) = (LayerTool::None, false, None, false);
                    (self.psf_note, self.deconv_compare) = (None, None);
                    self.smart_legend = None;
                    // What the name and any log beside the image say, and the equipment: the log's,
                    // else the one used last.
                    let mut analysis = analysis;
                    let mut capture = CaptureState::read(&path);
                    // A derotated sum shows the planet at the instant it was brought to.
                    if let Some(m) = derotated.as_ref().and_then(|d| d.moment) {
                        capture.info.time = Some(m);
                        capture.time_from = tl("dall'istante della derotazione");
                    }
                    // A colour image made of channels: the filter of the first one no longer holds.
                    if let Some(label) = derotated.as_ref().and_then(|d| d.mono) {
                        capture.info.filter = Some(label.to_string());
                    }
                    // No subject named: what the pixels suggest (Saturn, Mars, Jupiter).
                    capture.body = capture.body.or(analysis.planet);
                    capture.measure(&image, &analysis);
                    // The sharpening curve of the subject's network, when there is one for it.
                    let suggestion = capture.body.map(|b| ps_core::curve_net::suggestion(&image, &analysis, b));
                    ps_core::log::info(
                        "apertura",
                        &format!(
                            "{} ({}×{}, {}) tipo {:?} soggetto {:?} disco {:?} rumore {:.5} SNR {:.0} taglio {:.4} | rete nitidezza: {}",
                            path.display(),
                            image.width(),
                            image.height(),
                            if image.is_mono() { "mono" } else { "colore" },
                            analysis.kind,
                            capture.body,
                            analysis.disk.map(|d| (d.cx.round(), d.cy.round(), d.radius.round())),
                            analysis.noise,
                            analysis.snr,
                            analysis.spectral.as_ref().map_or(0.0, |s| s.cutoff),
                            match &suggestion {
                                Some(Ok(s)) => format!("picco {:.4} forza {:.2} disaccordo {:.2} anelli {} cielo {}", s.cutoff, s.strength, s.spread, s.rings.is_some(), s.sky_note.unwrap_or("usato")),
                                Some(Err(why)) => format!("nessuna ({why})"),
                                None => "nessun soggetto".into(),
                            }
                        ),
                    );
                    analysis.curve_net = suggestion.and_then(Result::ok);
                    // A subject and a time to compute from: open the Identifica tab, which sets
                    // itself up (the moons' fit included).
                    if self.prefs.identify_tab_on_open && !self.prefs.auto_settings && !keep_settings && capture.body.is_some() && capture.info.time.is_some() {
                        self.prefs.current_area = prefs_group::Group::Identify.name().to_string();
                    }
                    let setup = Setup::from_capture(&capture.info, &self.prefs.favorite_setup_value());
                    analysis.optics = setup.optics();
                    if kept.is_none() {
                        // A fresh image: nothing of the previous one stays on (layers, the
                        // Identifica tab and its subject, a question still open).
                        self.identify.new_image();
                        self.auto_identify = None;
                        self.last_identify_body = None;
                        self.preprocess_dialog = None;
                        self.subject_prompt = false;
                        // Structural settings follow the analysis; amounts only when the user
                        // chose to open images pre-processed.
                        self.pipeline = self.starting_point(analysis.kind, &analysis);
                        self.history = History::new(&self.pipeline);
                        self.clean_pipeline = self.pipeline.clone();
                    }
                    let image = Arc::new(image);
                    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(tl("immagine")).to_string();
                    let (kind, export_name) = kept.unwrap_or_else(|| (analysis.kind, if mosaic.is_some() { tr!("{stem}_mosaico", "{stem}_mosaic") } else if let Some(d) = &derotated { if let Some(label) = d.mono { format!("{stem}_{label}") } else { tr!("{stem}_derotata", "{stem}_derotated") } } else { stem }));
                    self.document = Some(Document {
                        export_name,
                        path,
                        original: Arc::clone(&image),
                        geometry: Geometry::default(),
                        oriented: Arc::clone(&image),
                        crop_offset: (0, 0),
                        source: image,
                        source_gen: next_generation(),
                        viewer,
                        analysis: *analysis,
                        kind,
                        luminance: None,
                        annotation: None,
                        mosaic: mosaic.map(|m| *m),
                        capture,
                        setup,
                        preprocessed: None,
                    });
                    self.last_output = None;
                    self.full_output = None;
                    self.mask_thumbs.clear();
                    self.histogram = None;
                    self.timings.clear();
                    self.warnings.clear();
                    if kept_none {
                        self.opened_fresh(ctx);
                    }
                    if self.preview == PreviewMode::Roi
                        && let Some(doc) = &mut self.document
                    {
                        doc.viewer.ensure_roi();
                    }
                }
                Event::LuminanceLoaded { name, plane } => {
                    if let Some(doc) = &mut self.document {
                        let full = Arc::new(Image::new(doc.original.width(), doc.original.height(), vec![plane], 16));
                        doc.luminance = Some(Luminance { name, full, cropped: (Arc::new(Vec::new()), 0) });
                        doc.crop_luminance();
                        self.pipeline.mineral.enabled = true;
                    }
                }
                Event::Exported { path, elapsed } => {
                    self.status = tr!("esportata {} in {}", "exported {} in {}", file_name(&path), format_duration(elapsed));
                    self.toast = Some((tr!("Esportata {} in {}", "Exported {} in {}", file_name(&path), format_duration(elapsed)), path.clone(), Instant::now()));
                    self.last_export = Some(path);
                    if let Some(exported) = self.exporting.take() {
                        self.clean_pipeline = exported;
                    }
                }
                Event::Failed(message) => {
                    ps_core::log::error("app", &message);
                    self.exporting = None;
                    self.status = message;
                }
            }
        }

        for out in self.worker.poll() {
            let Some(doc) = &mut self.document else { continue };
            if out.source_gen != doc.source_gen {
                continue;
            }
            if out.roi.is_none() && !out.proxy {
                self.full_output = Some(out.clone());
            }
            for t in &out.mask_thumbs {
                if self.mask_thumbs.get(&t.layer).is_none_or(|v| v.generation != t.generation) {
                    let texture = ctx.load_texture(format!("mask-thumb-{}", t.layer), (*t.image).clone(), egui::TextureOptions::LINEAR);
                    self.mask_thumbs.insert(t.layer, MaskThumbView { texture, generation: t.generation, empty: t.empty });
                }
            }
            self.mask_thumbs.retain(|id, _| self.pipeline.layers.iter().any(|l| l.id == *id));
            if out.background {
                continue;
            }
            doc.viewer.set_result(ctx, Arc::clone(&out.display), out.roi);
            doc.viewer.set_mask_overlay(ctx, out.mask_overlay.clone());
            doc.viewer.set_overlay(ctx, Overlay::Ringing, out.ringing_overlay.clone().map(|o| (o, out.roi)));
            doc.viewer.set_overlay(ctx, Overlay::Clipping, out.clip_overlay.as_ref().map(|(o, _)| (Arc::clone(o), out.roi)));
            doc.viewer.set_overlay(ctx, Overlay::SmartMap, out.smart_overlay.as_ref().map(|(o, _)| (Arc::clone(o), out.roi)));
            if let Some((_, legend)) = &out.smart_overlay {
                self.smart_legend = *legend;
            }
            self.clipped = out.clip_overlay.as_ref().map(|(_, n)| *n);
            self.timings = out.timings.iter().map(|t| (t.stage, t.clone())).collect();
            self.warnings = out.warnings.iter().cloned().collect();
            log_render(&out);
            self.histogram = Some(out.histogram.clone());
            // A reduced preview is only for looking at: pixel values, the black/white picker
            // and export keep using the last full-resolution result.
            if !out.proxy {
                if self.levels_pending && out.roi.is_none() && self.pipeline.tone.params.applied.is_empty() {
                    self.levels_pending = false;
                    let (black, white) = ps_core::enhance::gentle_levels(&out.tone_input);
                    let tone = &mut self.pipeline.tone;
                    (tone.params.current.black, tone.params.current.white) = (black, white);
                    if let Some(pre) = &mut doc.preprocessed {
                        (pre.tone.params.current.black, pre.tone.params.current.white) = (black, white);
                    }
                }
                self.last_output = Some(out);
            }
        }
    }

    fn submit_if_changed(&mut self, ctx: &egui::Context) {
        let Some(doc) = &self.document else { return };
        let roi = match self.preview {
            PreviewMode::Full => None,
            PreviewMode::Roi => match doc.viewer.roi {
                Some(r) => Some(r),
                None => return,
            },
        };
        let mask_overlay = self.selected_layer.filter(|_| self.show_mask);
        let luminance = doc.luminance.as_ref().map(|l| l.cropped.clone());
        let overlay = doc.overlay();
        let pipeline = self.processed_pipeline(doc);
        // Quick preview while a control in the panel is being dragged and the settings change.
        let dragging = self.prefs.fast_preview && roi.is_none() && !self.canvas_hovered && ctx.input(|i| i.pointer.any_down());
        let changing = self.last_request.as_ref().is_none_or(|k| k.proxy > 1 || k.pipeline != pipeline);
        let proxy = if dragging && changing { proxy_factor(&doc.source) } else { 1 };
        let key = RequestKey {
            source_gen: doc.source_gen,
            pipeline,
            roi,
            luminance_gen: luminance.as_ref().map(|l| l.1),
            overlay_gen: overlay.as_ref().map(|o| o[3].1),
            bxt: self.prefs.bxt_executable(),
            mask_overlay,
            memory_saving: self.prefs.memory_saving,
            proxy,
            ringing_map: self.show_ringing,
            clip_warning: self.clipping_shown(),
            smart_view: self.smart_view,
        };
        if self.last_request.as_ref() == Some(&key) {
            return;
        }
        self.worker.submit(Request {
            source: Arc::clone(&doc.source),
            source_gen: doc.source_gen,
            pipeline: key.pipeline.clone(),
            roi,
            max_texture_side: ctx.input(|i| i.max_texture_side),
            luminance,
            overlay,
            bxt_executable: key.bxt.clone(),
            mask_overlay,
            memory_saving: key.memory_saving,
            proxy,
            ringing_map: key.ringing_map,
            clip_warning: key.clip_warning,
            smart_view: key.smart_view,
        });
        self.last_request = Some(key);
    }

    /// Start comparing (or move "A" to now): the current result becomes version A, editing
    /// goes on as B.
    fn take_snapshot(&mut self, ctx: &egui::Context) {
        let (Some(doc), Some(out)) = (&mut self.document, &self.last_output) else { return };
        if out.source_gen != doc.source_gen {
            return;
        }
        doc.viewer.set_reference(ctx, Some((Arc::clone(&out.display), out.roi)));
        self.snapshot = Some(Snapshot { source_gen: doc.source_gen, pipeline: self.pipeline.clone() });
        self.compare = Compare::Ab;
        self.status = tl("Versione A fissata: continua a modificare (B), poi scegli quale tenere").into();
    }

    /// End the comparison keeping A (its settings come back) or B (A is forgotten).
    fn end_comparison(&mut self, ctx: &egui::Context, keep_a: bool) {
        let Some(snap) = self.snapshot.take() else { return };
        if keep_a {
            self.pipeline = snap.pipeline;
            self.status = tl("Tenuta la versione A").into();
        } else {
            self.status = tl("Tenuta la versione B").into();
        }
        if let Some(doc) = &mut self.document {
            doc.viewer.set_reference(ctx, None);
        }
        self.compare = Compare::After;
    }

    /// Forget the snapshot when it no longer matches the image (new file, crop, rotation).
    fn drop_stale_snapshot(&mut self, ctx: &egui::Context) {
        let current = self.document.as_ref().map(|d| d.source_gen);
        if self.snapshot.as_ref().is_some_and(|s| Some(s.source_gen) != current) {
            self.snapshot = None;
            if let Some(doc) = &mut self.document {
                doc.viewer.set_reference(ctx, None);
            }
        }
        if self.snapshot.is_none() && matches!(self.compare, Compare::Ab | Compare::OnlyA) {
            self.compare = Compare::After;
        }
    }

    fn set_preview(&mut self, mode: PreviewMode) {
        self.preview = mode;
        self.tool = if mode == PreviewMode::Roi { Tool::Roi } else { Tool::Pan };
        if mode == PreviewMode::Roi
            && let Some(doc) = &mut self.document
        {
            doc.viewer.ensure_roi();
        }
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.input_mut(|i| i.consume_shortcut(&SETTINGS)) {
            self.settings.open = !self.settings.open;
        }
        // Several images dropped together: a window asks what to do with them.
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        // A capture log dropped on an open image is read for it.
        let is_log = |p: &PathBuf| p.extension().and_then(|e| e.to_str()).is_some_and(|e| ["txt", "log", "as3", "as4"].contains(&e.to_ascii_lowercase().as_str()));
        let (logs, dropped): (Vec<PathBuf>, Vec<PathBuf>) = dropped.into_iter().partition(|p| is_log(p) && self.document.is_some());
        for log in logs {
            self.read_log(&log);
        }
        self.open_paths(ctx, dropped);
        if self.settings.open {
            return;
        }
        if ctx.input_mut(|i| i.consume_shortcut(&HELP)) {
            self.shortcuts_open = !self.shortcuts_open;
        }
        if ctx.input_mut(|i| i.consume_shortcut(&OPEN)) {
            self.open_dialog(ctx);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&QUICK_EXPORT)) {
            self.quick_export(ctx);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&EXPORT)) {
            self.export_dialog(ctx);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&ACTUAL_SIZE))
            && let Some(doc) = &mut self.document
        {
            doc.viewer.set_zoom(1.0);
        }
        // A focused text field undoes its own typing: leave ⌘Z / ⇧⌘Z to it.
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        if ctx.input_mut(|i| i.consume_shortcut(&REDO)) {
            self.history.redo(&mut self.pipeline);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&UNDO)) {
            self.history.undo(&mut self.pipeline);
        }
        if self.cropping {
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter)) {
                self.finish_crop(ctx, CropAction::Apply);
            }
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
                self.finish_crop(ctx, CropAction::Cancel);
            }
        }
        if self.picking.is_some() && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
            self.picking = None;
        }
        if self.layer_tool == LayerTool::Brush {
            let (bigger, smaller) = ctx.input(|i| (i.key_pressed(Key::CloseBracket), i.key_pressed(Key::OpenBracket)));
            if bigger {
                self.brush.radius = (self.brush.radius * 1.25).min(300.0);
            }
            if smaller {
                self.brush.radius = (self.brush.radius / 1.25).max(1.0);
            }
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
                self.layer_tool = LayerTool::None;
            }
        }
        // 1‥6: the panel's areas, in the order shown.
        if let Some(doc) = &self.document {
            let areas: Vec<String> = self.prefs.panel_layout(doc.kind, doc.source.is_mono()).iter().map(|(g, _)| g.name().to_string()).collect();
            for (key, name) in AREA_KEYS.iter().zip(areas) {
                if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, *key)) {
                    self.prefs.current_area = name;
                }
            }
        }
        if ctx.input_mut(|i| i.consume_shortcut(&CROP)) && !self.cropping {
            self.start_crop();
        }
        if ctx.input_mut(|i| i.consume_shortcut(&TOGGLE_ROI)) {
            self.set_preview(if self.preview == PreviewMode::Full { PreviewMode::Roi } else { PreviewMode::Full });
        }
        if ctx.input_mut(|i| i.consume_shortcut(&SPLIT)) {
            self.compare = if self.compare == Compare::Split { Compare::After } else { Compare::Split };
        }
        if let Some(doc) = &mut self.document
            && ctx.input_mut(|i| i.consume_shortcut(&FIT))
        {
            doc.viewer.request_fit();
        }
    }

    fn apply_actions(&mut self, ctx: &egui::Context, actions: Vec<PanelAction>) {
        for action in actions {
            match action {
                PanelAction::SelectSubject(subject) => {
                    let follows = self.follows_auto();
                    if self.document.as_ref().is_some_and(|d| d.kind != subject.kind) {
                        self.apply_kind(subject.kind);
                    }
                    if let Some(doc) = &mut self.document {
                        doc.capture.body = subject.body;
                        doc.capture.measure(&doc.original, &doc.analysis);
                    }
                    self.identify.set_body(subject.body);
                    self.last_identify_body = subject.body;
                    self.subject_prompt = false;
                    self.refresh_subject(follows);
                }
                PanelAction::RemoveAuto => self.clear_auto(),
                PanelAction::AutoAll => self.apply_preprocess(),
                PanelAction::AutoModule(stage) => self.apply_auto_module(stage),
                PanelAction::ResetModule(stage) => self.reset_module(stage),
                PanelAction::ClearAuto => {
                    self.clear_auto();
                }
                PanelAction::Export => self.quick_export(ctx),
                PanelAction::OpenAnnotation => {
                    self.prefs.current_area = prefs_group::Group::Identify.name().to_string();
                    self.identify.annotate_collapsed = false;
                }
                PanelAction::NorthUp { mirror, degrees } => {
                    self.rotate(ctx, |g| *g = g.then(mirror, degrees));
                }
                PanelAction::ExportAs => self.export_dialog(ctx),
                PanelAction::SavePreset => self.save_preset(),
                PanelAction::LoadPreset => self.load_preset(),
                PanelAction::OpenSettings => self.settings.open = true,
                PanelAction::LoadLuminance => self.load_luminance(ctx),
                PanelAction::ClearLuminance => {
                    if let Some(doc) = &mut self.document {
                        doc.luminance = None;
                    }
                }
                PanelAction::AddLayer(template) => self.add_layer(template),
                PanelAction::AutoLevels(layer) => {
                    let Some(doc) = &self.document else { continue };
                    let (black, white) = ps_core::enhance::auto_levels(&doc.source);
                    let tone = match layer.and_then(|id| self.pipeline.layer_mut(id)) {
                        Some(l) => &mut l.tone,
                        None => &mut self.pipeline.tone,
                    };
                    (tone.params.current.black, tone.params.current.white) = (black, white);
                    tone.enabled = true;
                }
                PanelAction::CompareDeconvolution => self.start_deconv_compare(ctx),
                PanelAction::Pick(target) => {
                    self.picking = Some(target);
                    self.cropping = false;
                    self.layer_tool = LayerTool::None;
                }
                PanelAction::CreateRole(role) => {
                    if !self.pipeline.layers.iter().any(|l| l.role == Some(role)) {
                        let id = add_role_layer(&mut self.pipeline, role);
                        (self.selected_layer, self.show_mask, self.layer_tool) = (Some(id), true, LayerTool::None);
                    }
                }
                PanelAction::RestoreRole(id) => self.restore_role(id),
            }
        }
    }

    /// Back to the recommended settings of the rings' or moons' mask, keeping its place in the
    /// list, its name, its switch and the strokes painted on it.
    fn restore_role(&mut self, id: u64) {
        let Some(index) = self.pipeline.layers.iter().position(|l| l.id == id) else { return };
        let Some(role) = self.pipeline.layers[index].role else { return };
        let fresh_id = add_role_layer(&mut self.pipeline, role);
        let mut fresh = self.pipeline.layers.pop().expect("just added");
        debug_assert_eq!(fresh.id, fresh_id);
        let old = &self.pipeline.layers[index];
        (fresh.id, fresh.name, fresh.enabled) = (old.id, old.name.clone(), old.enabled);
        fresh.mask.strokes = old.mask.strokes.clone();
        self.pipeline.layers[index] = fresh;
    }

    fn add_layer(&mut self, template: LayerTemplate) {
        let Some(doc) = &self.document else { return };
        let (w, h) = (doc.source.width() as f32, doc.source.height() as f32);
        let shape = match template {
            LayerTemplate::Auto(region) => MaskShape::Auto(region),
            LayerTemplate::Ellipse => MaskShape::Ellipse { cx: w / 2.0, cy: h / 2.0, rx: w / 4.0, ry: h / 4.0, angle: 0.0 },
            LayerTemplate::Rectangle => MaskShape::Rectangle { x: w / 4.0, y: h / 4.0, w: w / 2.0, h: h / 2.0 },
            LayerTemplate::Brush => MaskShape::Empty,
        };
        let id = self.pipeline.add_layer(shape.label(), MaskSpec::new(shape, 4.0));
        self.selected_layer = Some(id);
        self.show_mask = true;
        // Its mask tools must be on screen, or they would be put away at once.
        self.prefs.collapsed_sections.remove("layer:mask");
        self.layer_tool = match shape {
            MaskShape::Empty => LayerTool::Brush,
            MaskShape::Ellipse { .. } | MaskShape::Rectangle { .. } => LayerTool::Shape,
            MaskShape::Auto(_) => LayerTool::None,
        };
    }

    /// Set the black or white point from the brightness under the click (3×3 average of what
    /// the current tone step receives).
    fn pick_point(&mut self, target: PickTarget, (x, y): (usize, usize)) {
        self.picking = None;
        if target == PickTarget::Psf {
            self.measure_psf(Rect::from_center_size(Pos2::new(x as f32, y as f32), Vec2::splat(48.0)));
            return;
        }
        let Some(out) = &self.last_output else { return };
        let (ox, oy) = out.roi.map_or((0, 0), |r| (r.x, r.y));
        let img = &out.tone_input;
        let (Some(px), Some(py)) = (x.checked_sub(ox), y.checked_sub(oy)) else { return };
        if px >= img.width() || py >= img.height() {
            self.status = tl("Il punto scelto è fuori dalla regione elaborata (ROI)").into();
            return;
        }
        let lum = working_planes(img, ColorMode::Luminance).swap_remove(0);
        let (w, h) = (img.width(), img.height());
        let mut sum = 0.0;
        let mut n = 0.0;
        for yy in py.saturating_sub(1)..(py + 2).min(h) {
            for xx in px.saturating_sub(1)..(px + 2).min(w) {
                sum += lum[yy * w + xx];
                n += 1.0;
            }
        }
        let v = (sum / n).clamp(0.0, 1.0);
        let tone = &mut self.pipeline.tone;
        match target {
            PickTarget::Black => tone.params.current.black = v.min(tone.params.current.white - 0.02).max(0.0),
            PickTarget::White => tone.params.current.white = v.max(tone.params.current.black + 0.02).min(1.0),
            PickTarget::Psf => unreachable!("handled above"),
        }
        tone.enabled = true;
    }

    /// The selected mask must still exist (an undo can remove its layer).
    fn drop_stale_selection(&mut self) {
        if let Some(id) = self.selected_layer
            && id != RING_MASK
            && !self.pipeline.layers.iter().any(|l| l.id == id)
        {
            (self.selected_layer, self.layer_tool, self.show_mask) = (None, LayerTool::None, false);
        }
    }

    /// After the panel is drawn: mask tools whose controls are no longer on screen (card or
    /// area closed) are put away, so the brush and the red overlay don't linger; and picking a
    /// mask tool ends the black/white picker and the crop.
    /// `before`: tool and selected mask at the start of the frame (a tool chosen during this
    /// frame, e.g. by adding a layer, gets its controls drawn only from the next one).
    fn settle_tools(&mut self, ctx: &egui::Context, before: (LayerTool, Option<u64>)) {
        let shown = ctx.data(|d| d.get_temp::<bool>(egui::Id::unique(panels::MASK_TOOLS_SHOWN))).unwrap_or(false);
        // Same for the ringing map: only while its switch is in view.
        if !ctx.data(|d| d.get_temp::<bool>(egui::Id::unique(panels::RINGING_SWITCH_SHOWN))).unwrap_or(false) {
            self.show_ringing = false;
        }
        if !ctx.data(|d| d.get_temp::<bool>(egui::Id::unique(panels::SMART_ZONES_SWITCH_SHOWN))).unwrap_or(false) {
            self.smart_view = SmartView::None;
        }
        let unchanged = (self.layer_tool, self.selected_layer) == before;
        if !shown && unchanged {
            (self.layer_tool, self.show_mask) = (LayerTool::None, false);
        }
        if self.layer_tool != before.0 && self.layer_tool != LayerTool::None {
            self.picking = None;
            if self.cropping {
                self.finish_crop(ctx, CropAction::Cancel);
            }
        }
    }

    /// The PSF from the star or moon inside `area` (image pixels of the oriented, cropped
    /// source), and a note saying what was found.
    fn measure_psf(&mut self, area: Rect) {
        self.picking = None;
        let Some(doc) = &self.document else { return };
        let lum = working_planes(&doc.source, ColorMode::Luminance).swap_remove(0);
        let (x0, y0) = (area.min.x.max(0.0) as usize, area.min.y.max(0.0) as usize);
        let (x1, y1) = (area.max.x.max(0.0) as usize, area.max.y.max(0.0) as usize);
        match ps_core::psf::measure_in(&lum, doc.source.width(), doc.source.height(), x0, y0, x1, y1) {
            Ok(m) => {
                let shape = if m.roundness > 0.85 { tl("rotondo") } else if m.roundness > 0.65 { tl("un po' allungato") } else { tl("allungato: forse mosso o non puntiforme") };
                let faint = if m.brightness < 0.05 { tl(" Molto debole rispetto al resto dell'immagine: la forma è incerta.") } else { "" };
                self.psf_note = Some(tr!("Misurata su un oggetto in ({:.0}, {:.0}): largo {:.1} px, {shape}.{faint} Se è una luna, la sua dimensione si somma alla sfocatura.", "Measured on an object at ({:.0}, {:.0}): {:.1} px wide, {shape}.{faint} If it is a moon, its own size adds to the blur.",
                    m.x, m.y, m.fwhm
                ));
                let d = &mut self.pipeline.deconv;
                d.params.psf = ps_core::psf::Psf::Measured { fwhm: m.fwhm };
                d.params.measured = Some(m.kernel);
                d.enabled = true;
            }
            Err(why) => self.psf_note = Some(why.to_string()),
        }
    }

    /// Deconvolve the visible part of the image with each PSF shape and each algorithm, off the
    /// UI thread, one tile after the other.
    fn start_deconv_compare(&mut self, ctx: &egui::Context) {
        let Some(doc) = &self.document else { return };
        let current = self.pipeline.deconv.params.clone();
        let fwhm = current.psf.fwhm_px();
        let optics = doc.analysis.optics;
        let shape = |kind: u8| DeconvParams { psf: panels::convert_psf(current.psf, kind, optics), algorithm: DeconvAlgorithm::RichardsonLucy, measured: None, ..current.clone() };
        let algorithm = |algorithm: DeconvAlgorithm| DeconvParams { algorithm, auto_k: true, ..current.clone() };
        let tiles = vec![
            (tl("Gaussiana").to_string(), Some(shape(0))),
            ("Moffat".to_string(), Some(shape(1))),
            ("Airy".to_string(), Some(shape(2))),
            (tl("Originale").to_string(), None),
            (format!("Richardson–Lucy · {}", short_psf(&current.psf)), Some(algorithm(DeconvAlgorithm::RichardsonLucy))),
            (format!("Wiener · {}", short_psf(&current.psf)), Some(algorithm(DeconvAlgorithm::Wiener))),
        ];
        // The visible part, at most 384 px a side around its centre.
        let visible = doc.viewer.visible_rect();
        let (w, h) = (doc.source.width(), doc.source.height());
        let side_w = (visible.width() as usize).clamp(64, 384).min(w);
        let side_h = (visible.height() as usize).clamp(64, 384).min(h);
        let x0 = ((visible.center().x as usize).saturating_sub(side_w / 2)).min(w - side_w);
        let y0 = ((visible.center().y as usize).saturating_sub(side_h / 2)).min(h - side_h);
        let crop = Arc::new(doc.source.crop(x0, y0, side_w, side_h));
        let (tx, rx) = std::sync::mpsc::channel();
        let jobs: Vec<Option<DeconvParams>> = tiles.iter().map(|(_, p)| p.clone()).collect();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            for (k, job) in jobs.into_iter().enumerate() {
                let image = match job {
                    Some(params) => {
                        let mut p = Pipeline::default();
                        (p.align.enabled, p.bxt.enabled, p.denoise.enabled, p.wavelet.enabled, p.contrast.enabled) = (false, false, false, false, false);
                        (p.dering.enabled, p.tone.enabled, p.curves.enabled, p.color.enabled, p.mineral.enabled, p.suncolor.enabled, p.photocolor.enabled) = (false, false, false, false, false, false, false);
                        p.deconv = Module::on(params);
                        match Processor::default().run(Arc::clone(&crop), next_generation(), &p, &Aux::default(), &|| false, &|_, _| {}) {
                            Some(out) => out.image,
                            None => continue,
                        }
                    }
                    None => Arc::clone(&crop),
                };
                if tx.send((k, worker::display_image(&image, 512))).is_err() {
                    return;
                }
                ctx.request_repaint();
            }
        });
        self.deconv_compare = Some(DeconvCompare { tiles: tiles.into_iter().map(|(label, params)| CompareTile { label, params, texture: None }).collect(), pending: Some(rx), fwhm });
    }

    /// The comparison window: tiles as they arrive, "Usa questa" to adopt one.
    fn deconv_compare_window(&mut self, ctx: &egui::Context) {
        let Some(compare) = &mut self.deconv_compare else { return };
        if let Some(rx) = &compare.pending {
            while let Ok((k, image)) = rx.try_recv() {
                compare.tiles[k].texture = Some(ctx.load_texture(format!("deconv-compare-{k}"), image, egui::TextureOptions::LINEAR));
            }
        }
        let current = &self.pipeline.deconv.params;
        let mut open = true;
        let mut chosen = None;
        crate::widgets::window(ctx, tl("Confronto deconvoluzione")).default_size([960.0, 700.0]).open(&mut open).show(ctx, |ui| {
            panels::hint(ui, &tr!("Zona visibile dell'immagine, sfocatura {:.1} px. Sopra: le tre forme di PSF con Richardson–Lucy. Sotto: l'originale e i due algoritmi con la forma attuale.", "Visible area of the image, blur {:.1} px. Top: the three PSF shapes with Richardson–Lucy. Bottom: the original and the two algorithms with the current shape.", compare.fwhm));
            let avail = ui.available_size();
            let side = ((avail.x - 24.0) / 3.0).min((avail.y - 90.0) / 2.0).max(120.0);
            for row in compare.tiles.chunks(3).enumerate() {
                ui.horizontal(|ui| {
                    for (j, tile) in row.1.iter().enumerate() {
                        let k = row.0 * 3 + j;
                        let selected = tile.params.as_ref().is_some_and(|p| p.psf == current.psf && p.algorithm == current.algorithm);
                        egui::Frame::new().stroke(Stroke::new(1.5, if selected { ACCENT } else { egui::Color32::TRANSPARENT })).inner_margin(egui::Margin::same(3)).show(ui, |ui| {
                            ui.vertical(|ui| {
                                match &tile.texture {
                                    Some(t) => {
                                        let size = t.size_vec2();
                                        let scale = side / size.x.max(size.y);
                                        ui.image((t.id(), size * scale));
                                    }
                                    None => {
                                        ui.allocate_ui(Vec2::splat(side), |ui| {
                                            ui.centered_and_justified(|ui| ui.spinner());
                                        });
                                    }
                                }
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(&tile.label).strong());
                                    if tile.params.is_some() && ui.button((ICON_CHECK.rich_text(), tl("Usa questa"))).clicked() {
                                        chosen = Some(k);
                                    }
                                });
                            });
                        });
                    }
                });
            }
        });
        if let Some(k) = chosen
            && let Some(params) = compare.tiles[k].params.clone()
        {
            self.pipeline.deconv = Module::on(params);
            open = false;
        }
        if !open {
            self.deconv_compare = None;
        }
    }

    fn add_stroke(&mut self, points: Vec<[f32; 2]>, erase: bool) {
        let stroke = BrushStroke { points, radius: self.brush.radius, hardness: self.brush.hardness, erase };
        match self.selected_layer {
            Some(RING_MASK) => self.pipeline.dering.params.strokes.push(stroke),
            Some(id) => {
                if let Some(layer) = self.pipeline.layer_mut(id) {
                    layer.mask.strokes.push(stroke);
                }
            }
            None => {}
        }
    }

    /// Mirror the selected layer's hand-drawn shape into the viewer for editing.
    fn sync_shape_to_viewer(&mut self) {
        let Some(doc) = &mut self.document else { return };
        let layer = self.selected_layer.and_then(|id| self.pipeline.layers.iter().find(|l| l.id == id));
        doc.viewer.shape = layer.and_then(|l| match l.mask.shape {
            MaskShape::Ellipse { cx, cy, rx, ry, angle } => Some((Rect::from_center_size(Pos2::new(cx, cy), Vec2::new(2.0 * rx, 2.0 * ry)), ShapeKind::Ellipse { angle_deg: angle })),
            MaskShape::Rectangle { x, y, w, h } => Some((Rect::from_min_size(Pos2::new(x, y), Vec2::new(w, h)), ShapeKind::Rectangle)),
            MaskShape::Auto(_) | MaskShape::Empty => None,
        });
        if doc.viewer.shape.is_none() && self.layer_tool == LayerTool::Shape {
            self.layer_tool = LayerTool::None;
        }
    }

    fn sync_shape_from_viewer(&mut self) {
        let Some(doc) = &self.document else { return };
        let (Some((r, kind)), Some(id)) = (doc.viewer.shape, self.selected_layer) else { return };
        let Some(layer) = self.pipeline.layer_mut(id) else { return };
        let shape = match kind {
            ShapeKind::Ellipse { angle_deg } => MaskShape::Ellipse { cx: r.center().x, cy: r.center().y, rx: r.width() / 2.0, ry: r.height() / 2.0, angle: angle_deg },
            ShapeKind::Rectangle => MaskShape::Rectangle { x: r.min.x, y: r.min.y, w: r.width(), h: r.height() },
        };
        if layer.mask.shape != shape {
            layer.mask.shape = shape;
        }
    }

    // ---- layout ------------------------------------------------------------------------

    /// The list of shortcuts, as shown in the Help menu and in its own window.
    fn shortcut_rows(ui: &mut egui::Ui, ctx: &egui::Context) {
                let f = |k: &KeyboardShortcut| ctx.format_shortcut(k);
                let alt = tr!("{} (tieni)", "{} (hold)", theme::alt_key());
                let open_export = format!("{} / {}", f(&OPEN), f(&EXPORT));
                let quick_export = f(&QUICK_EXPORT);
                let fit_actual = format!("F / {}", f(&ACTUAL_SIZE));
                let undo_redo = format!("{} / {}", f(&UNDO), f(&REDO));
                let settings = f(&SETTINGS);
                for (keys, what) in [
                    (open_export.as_str(), tl("Apri / esporta come…")),
                    (quick_export.as_str(), tl("Esporta accanto all'originale")),
                    ("1 … 6", tl("Aree del pannello")),
                    (undo_redo.as_str(), tl("Annulla / ripeti")),
                    (settings.as_str(), tl("Impostazioni")),
                    (tl("Spazio (tieni)"), tl("Mostra l'originale")),
                    ("S", tl("Vista divisa prima/dopo")),
                    ("R", tl("Anteprima solo su una regione (ROI)")),
                    ("C", tl("Ritaglia (Invio applica, Esc annulla)")),
                    ("[ / ]", tl("Pennello più piccolo / più grande")),
                    (alt.as_str(), tl("Il pennello toglie invece di aggiungere")),
                    (fit_actual.as_str(), tl("Adatta / 100 %")),
                    (tl("Doppio clic su un titolo"), tl("Escludi o riattiva il modulo")),
                    (tl("Rotella, pinch"), "Zoom"),
                    (tl("Tasto destro, trascina"), tl("Sposta la vista")),
                    (tl("Doppio clic su un cursore"), tl("Valore predefinito")),
                ] {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(keys).monospace().color(ACCENT));
                        ui.label(what);
                    });
                }
    }

    /// The shortcuts in a window of their own, opened with `?`.
    fn shortcuts_window(&mut self, ctx: &egui::Context) {
        if !self.shortcuts_open {
            return;
        }
        let mut open = true;
        crate::widgets::window(ctx, tl("Scorciatoie")).open(&mut open).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            Self::shortcut_rows(ui, ctx);
        });
        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            open = false;
        }
        self.shortcuts_open = open;
    }

    /// A thin accent border while a file is dragged over the window.
    fn drop_hint(ctx: &egui::Context) {
        if ctx.input(|i| i.raw.hovered_files.is_empty()) {
            return;
        }
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("drop-hint")));
        painter.rect_stroke(ctx.content_rect().shrink(2.0), 6.0, Stroke::new(3.0, ACCENT), egui::StrokeKind::Inside);
    }

    /// The notice after an export: what was written and a way to see it, gone after a few seconds.
    fn export_toast(&mut self, ctx: &egui::Context) {
        const SECONDS: f32 = 8.0;
        let Some((text, path, since)) = &self.toast else { return };
        let left = SECONDS - since.elapsed().as_secs_f32();
        if left <= 0.0 {
            self.toast = None;
            return;
        }
        let (text, path) = (text.clone(), path.clone());
        let mut close = false;
        egui::Area::new(egui::Id::unique("export-toast")).order(egui::Order::Foreground).anchor(egui::Align2::RIGHT_BOTTOM, [-16.0, -40.0]).show(ctx, |ui| {
            egui::Frame::new().fill(theme::PANEL).stroke(Stroke::new(1.0, ACCENT)).corner_radius(theme::radius::L).inner_margin(egui::Margin::symmetric(12, 8)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(ICON_CHECK_CIRCLE.rich_text().color(ACCENT));
                    ui.label(RichText::new(text).color(TEXT));
                    if ui.button(if cfg!(target_os = "macos") { tr!("Mostra nel Finder", "Show in Finder") } else { tr!("Mostra nella cartella", "Show in folder") }).clicked() {
                        reveal(&path);
                        close = true;
                    }
                    if icon_button(ui, ICON_CLOSE, tl("Chiudi")).clicked() {
                        close = true;
                    }
                });
            });
        });
        if close {
            self.toast = None;
        } else {
            ctx.request_repaint_after(Duration::from_secs_f32(left.min(1.0)));
        }
    }

    /// The "Apri recente" submenu of the File menu.
    fn recent_menu(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let recents = self.prefs.recent_files.clone();
        let free = self.io_busy.is_none();
        let mut picked = None;
        ui.add_enabled_ui(!recents.is_empty(), |ui| {
            ui.menu_button((ICON_HISTORY.rich_text(), tr!("Apri recente", "Open recent")), |ui| {
                for r in &recents {
                    let tex = self.recent_textures.entry(r.path.clone()).or_insert_with(|| {
                        let image = ColorImage::from_rgb(r.size, &r.thumb);
                        ctx.load_texture(format!("recent-{}", r.path.display()), image, egui::TextureOptions::LINEAR)
                    });
                    ui.horizontal(|ui| {
                        ui.image((tex.id(), Vec2::new(r.size[0] as f32, r.size[1] as f32) * 0.75));
                        if ui.add_enabled(free, egui::Button::new(file_name(&r.path)).frame(false)).on_hover_text(r.path.display().to_string()).clicked() {
                            picked = Some(r.path.clone());
                        }
                    });
                }
            });
        });
        if let Some(path) = picked {
            ui.close();
            if path.exists() {
                self.load(ctx, path);
            } else {
                self.prefs.recent_files.retain(|r| r.path != path);
                self.status = tr!("{} non c'è più", "{} is gone", file_name(&path));
            }
        }
    }

    /// One bar: application menus on the left, view controls (found in no menu) on the right.
    fn top_bar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.label(ICON_PLANET.rich_text().size(theme::size::ICON).color(ACCENT));
            ui.add_space(theme::space::S);
            ui.menu_button("File", |ui| {
                let io_free = self.io_busy.is_none();
                if ui.add_enabled(io_free, egui::Button::new((ICON_FOLDER_OPEN.rich_text(), tl("Apri…"))).shortcut_text(ctx.format_shortcut(&OPEN))).clicked() {
                    self.open_dialog(ctx);
                }
                self.recent_menu(ctx, ui);
                let quick = egui::Button::new((ICON_IOS_SHARE.rich_text(), tl("Esporta"))).shortcut_text(ctx.format_shortcut(&QUICK_EXPORT));
                if ui.add_enabled(io_free && self.document.is_some(), quick).on_hover_text(tl("Accanto all'originale, nel formato dell'ultima volta")).clicked() {
                    self.quick_export(ctx);
                }
                let export = egui::Button::new((ICON_SAVE_AS.rich_text(), tl("Esporta come…"))).shortcut_text(ctx.format_shortcut(&EXPORT));
                if ui.add_enabled(io_free && self.document.is_some(), export).clicked() {
                    self.export_dialog(ctx);
                }
                if ui.add_enabled(self.batch.is_none(), egui::Button::new((ICON_PHOTO_LIBRARY.rich_text(), tl("Elabora una serie…")))).on_hover_text(tl("Più immagini con le stesse impostazioni, salvate in una cartella")).clicked() {
                    self.batch_pick();
                }
                ui.separator();
                if ui.add_enabled(io_free, egui::Button::new((ICON_GRID_VIEW.rich_text(), tl("Nuovo mosaico…")))).on_hover_text(tl("Unisci più immagini sovrapposte dello stesso soggetto (Luna, Sole…) in una sola")).clicked() {
                    self.new_mosaic_dialog(ctx);
                }
                if ui.add_enabled(io_free, egui::Button::new((ICON_AUTORENEW.rich_text(), tl("Derotazione…")))).on_hover_text(tl("Somma più riprese di Giove o Saturno portandole allo stesso istante con la rotazione del pianeta")).clicked() {
                    self.new_derotation_dialog(ctx);
                }
                if ui.add_enabled(io_free, egui::Button::new((ICON_PALETTE.rich_text(), tl("Mono → RGB…")))).on_hover_text(tl("Unisci le immagini monocromatiche R, G, B (ed eventualmente L) in una a colori")).clicked() {
                    self.new_mono_rgb_dialog(ctx, false);
                }
                if ui.add_enabled(io_free, egui::Button::new((ICON_PALETTE.rich_text(), tl("Mono → RGB derotato…")))).on_hover_text(tl("Come Mono → RGB, ma ogni canale è portato allo stesso istante con la rotazione di Giove o Saturno")).clicked() {
                    self.new_mono_rgb_dialog(ctx, true);
                }
                let mosaic = self.document.as_ref().and_then(|d| d.mosaic.clone());
                if ui.add_enabled(mosaic.is_some(), egui::Button::new((ICON_EDIT.rich_text(), tl("Modifica mosaico…")))).on_hover_text(tl("Cambia le immagini o le posizioni del mosaico aperto: le regolazioni restano")).clicked()
                    && let Some(m) = mosaic
                {
                    self.mosaic_editor.edit(ctx, m, true);
                }
                ui.separator();
                if ui.button((ICON_UPLOAD_FILE.rich_text(), tl("Carica impostazioni…"))).clicked() {
                    self.load_preset();
                }
                if ui.button((ICON_SAVE.rich_text(), tl("Salva impostazioni…"))).clicked() {
                    self.save_preset();
                }
            });
            ui.menu_button(tl("Modifica"), |ui| {
                let undo = egui::Button::new((ICON_UNDO.rich_text(), tl("Annulla"))).shortcut_text(ctx.format_shortcut(&UNDO));
                let undo_tip = self.history.undo_name().map(|n| tr!("Annulla: {n}", "Undo: {n}"));
                if ui.add_enabled(self.history.can_undo(), undo).on_hover_text(undo_tip.unwrap_or_default()).clicked() {
                    self.history.undo(&mut self.pipeline);
                }
                let redo = egui::Button::new((ICON_REDO.rich_text(), tl("Ripeti"))).shortcut_text(ctx.format_shortcut(&REDO));
                let redo_tip = self.history.redo_name().map(|n| tr!("Ripeti: {n}", "Redo: {n}"));
                if ui.add_enabled(self.history.can_redo(), redo).on_hover_text(redo_tip.unwrap_or_default()).clicked() {
                    self.history.redo(&mut self.pipeline);
                }
                ui.separator();
                let has_doc = self.document.is_some();
                if ui.add_enabled(has_doc && !self.cropping, egui::Button::new((ICON_CROP.rich_text(), tl("Ritaglia"))).shortcut_text("C")).clicked() {
                    self.start_crop();
                }
                if ui.add_enabled(has_doc, egui::Button::new((ICON_ROTATE_RIGHT.rich_text(), tl("Ruota di 90° in senso orario")))).clicked() {
                    self.rotate(ctx, Geometry::rotate_clockwise);
                }
                if ui.add_enabled(has_doc, egui::Button::new((ICON_ROTATE_LEFT.rich_text(), tl("Ruota di 90° in senso antiorario")))).clicked() {
                    self.rotate(ctx, Geometry::rotate_counterclockwise);
                }
                if ui.add_enabled(has_doc, egui::Button::new((ICON_SWAP_HORIZ.rich_text(), tl("Specchia orizzontalmente")))).clicked() {
                    self.rotate(ctx, |g| g.flip_horizontal = !g.flip_horizontal);
                }
                if ui.add_enabled(has_doc, egui::Button::new((ICON_SWAP_VERT.rich_text(), tl("Specchia verticalmente")))).clicked() {
                    self.rotate(ctx, |g| g.flip_vertical = !g.flip_vertical);
                }
                let cropped = self.document.as_ref().is_some_and(|d| d.source.width() != d.oriented.width() || d.source.height() != d.oriented.height());
                if ui.add_enabled(cropped, egui::Button::new((ICON_CROP_FREE.rich_text(), tl("Rimuovi ritaglio")))).clicked() {
                    self.remove_crop(ctx);
                }
                ui.separator();
                let auto = egui::Button::new((ICON_AUTO_FIX_HIGH.rich_text(), tl("Impostazioni automatiche")));
                if ui.add_enabled(self.document.is_some(), auto).clicked() {
                    self.apply_auto_all();
                }
            });
            ui.menu_button(tl("Aiuto"), |ui| {
                ui.label(RichText::new(tl("Scorciatoie")).strong());
                Self::shortcut_rows(ui, ctx);
                if ui.button((ICON_KEYBOARD.rich_text(), tr!("Mostra in una finestra (?)", "Show in a window (?)"))).clicked() {
                    self.shortcuts_open = true;
                }
                ui.separator();
                if ui.button((ICON_INFO.rich_text(), tl("Informazioni su Planetary Studio"))).clicked() {
                    self.about_open = true;
                }
            });

            if self.document.is_some() {
                ui.add(egui::Separator::default().vertical().spacing(14.0));
                self.instrument_menu(ui);
            }

            // Actions only (the view options float over the image, export has its tab).
            // Right-to-left: added from the far right (settings) towards the menus.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let separator = |ui: &mut egui::Ui| ui.add(egui::Separator::default().vertical().spacing(14.0));
                if icon_button(ui, ICON_SETTINGS, &tr!("Impostazioni ({})", "Settings ({})", ctx.format_shortcut(&SETTINGS))).clicked() {
                    self.settings.open = true;
                }
                separator(ui);
                if self.document.is_some() {
                    if tool_button(ui, ICON_WARNING_AMBER, tl("Bruciati"), self.show_clipping)
                        .on_hover_text(tl("Mostra i pixel bruciati: in rosso quelli diventati bianco pieno, in blu il dettaglio schiacciato a nero"))
                        .clicked()
                    {
                        self.show_clipping = !self.show_clipping;
                    }
                    let comparing = self.snapshot.is_some();
                    let can_compare = self.last_output.is_some();
                    let tip = if comparing {
                        tl("Chiudi il confronto tenendo la versione B (quella che stai modificando)")
                    } else {
                        tl("Fissa il risultato attuale come versione A, continua a modificare (B) e confrontale: poi tieni quella che preferisci")
                    };
                    let tip = format!("{tip} ({})", ctx.format_shortcut(&SPLIT));
                    if ui.add_enabled_ui(can_compare, |ui| tool_button(ui, ICON_COMPARE, tl("Confronta"), comparing)).inner.on_hover_text(tip).clicked() {
                        if comparing {
                            self.end_comparison(ctx, false);
                        } else {
                            self.take_snapshot(ctx);
                        }
                    }
                    if tool_button(ui, ICON_CROP, tl("Ritaglia"), self.cropping).on_hover_text(tl("Ritaglia e ruota (C)")).clicked() && !self.cropping {
                        self.start_crop();
                    }
                    // On by itself, or while its tab is the one showing (like the burnt pixels while a curve is dragged).
                    let in_tab = self.prefs.current_area == prefs_group::Group::Identify.name();
                    if tool_button(ui, ICON_TRAVEL_EXPLORE, tl("Identifica"), self.identify.active || in_tab)
                        .on_hover_text(tl("Riconosce l'oggetto e scrive i nomi sull'immagine: lune di Giove e Saturno, mari della Luna. Usa data e informazioni della ripresa. Le impostazioni sono nella scheda «Identifica e annota»"))
                        .clicked()
                    {
                        if in_tab {
                            // Off: the tab closes with it.
                            self.identify.active = false;
                            self.prefs.current_area = prefs_group::Group::Sharpness.name().to_string();
                        } else if self.identify.active {
                            self.identify.active = false;
                        } else {
                            self.identify.switch_on();
                            self.prefs.current_area = prefs_group::Group::Identify.name().to_string();
                        }
                    }
                }
            });
        });
    }

    /// Burnt pixels on screen: switched on in the top bar, or while a levels slider or a curve
    /// point is dragged.
    fn clipping_shown(&self) -> bool {
        self.show_clipping || self.tone_dragging
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(busy) = &self.io_busy {
                ui.add(egui::Spinner::new().size(12.0).color(ACCENT));
                ui.label(RichText::new(busy).small());
            } else if let Some(doc) = &self.document {
                let img = &doc.source;
                let info = format!(
                    "{} · {} × {} · {} bit {}",
                    file_name(&doc.path),
                    img.width(),
                    img.height(),
                    img.source_bit_depth,
                    if img.is_mono() { "mono" } else { "RGB" }
                );
                ui.label(RichText::new(info).small().color(TEXT_WEAK));
                if !self.status.is_empty() {
                    ui.separator();
                    ui.label(RichText::new(&self.status).small().color(TEXT_WEAK));
                }
            } else {
                ui.label(RichText::new(&self.status).small().color(TEXT_WEAK));
            }
            if let Some(p) = self.worker.progress() {
                ui.separator();
                ui.label(RichText::new(p.stage.name()).small());
                ui.add(egui::ProgressBar::new(p.fraction).desired_width(120.0).desired_height(5.0).fill(ACCENT));
            } else if self.worker.is_background_busy() {
                ui.separator();
                ui.add(egui::Spinner::new().size(10.0).color(TEXT_WEAK));
                ui.label(RichText::new(tl("Elaborazione dell'immagine intera in background…")).small().color(TEXT_WEAK));
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(&self.backend).small().color(TEXT_WEAK));
                if let Some(out) = &self.last_output {
                    ui.separator();
                    ui.label(RichText::new(tr!("elaborazione {}", "processing {}", format_duration(out.elapsed))).small().color(TEXT_WEAK));
                }
                if let Some(n) = self.clipped.filter(|_| self.clipping_shown()) {
                    ui.separator();
                    let colour = if n > 0 { ui.visuals().warn_fg_color } else { TEXT_WEAK };
                    ui.label(RichText::new(if n == 1 { tl("1 pixel bruciato").to_string() } else { tr!("{n} pixel bruciati", "{n} clipped pixels") }).small().color(colour));
                }
                if let (Some((x, y)), Some(doc)) = (self.hover, &self.document) {
                    ui.separator();
                    let value = self.pixel_value(doc, x, y);
                    ui.label(RichText::new(format!("x {x}  y {y}  ·  {value}")).small().monospace().color(TEXT_WEAK));
                }
            });
        });
    }

    /// Processed value under the cursor if available, otherwise the original.
    fn pixel_value(&self, doc: &Document, x: usize, y: usize) -> String {
        // Only an output at the source's own scale: the quick preview while a slider is dragged
        // is reduced (reading it at the source's coordinates ran past its end).
        let (sw, sh) = (doc.source.width(), doc.source.height());
        let processed = self.last_output.as_ref().filter(|o| o.source_gen == doc.source_gen).and_then(|o| match o.roi {
            None if (o.image.width(), o.image.height()) == (sw, sh) => Some((&o.image, x, y)),
            Some(r) if (r.x..r.x + r.w).contains(&x) && (r.y..r.y + r.h).contains(&y) && (o.image.width(), o.image.height()) == (r.w, r.h) => Some((&o.image, x - r.x, y - r.y)),
            _ => None,
        });
        let (img, px, py) = processed.unwrap_or((&doc.source, x, y));
        if px >= img.width() || py >= img.height() {
            return String::new();
        }
        let i = py * img.width() + px;
        let values: Vec<String> = img.channels().iter().map(|c| format!("{:.4}", c[i])).collect();
        values.join(" ")
    }

    /// Whether a module currently changes the image (for the group headers' counts).
    fn stage_active(&self, stage: Stage) -> bool {
        let p = &self.pipeline;
        let role = |r: LayerRole| p.layers.iter().any(|l| l.role == Some(r) && l.enabled);
        match stage {
            Stage::Align => p.align.enabled && !p.align.params.is_identity(),
            Stage::Bxt => p.bxt.enabled,
            Stage::Deconv => p.deconv.enabled && !p.deconv.params.is_neutral(),
            Stage::Denoise => p.denoise.enabled && !p.denoise.params.is_neutral(),
            Stage::Wavelet => p.wavelet.enabled && !p.wavelet.params.is_neutral(),
            Stage::Smart => p.smart.enabled && !p.smart.params.is_neutral(),
            Stage::Contrast => p.contrast.enabled && !p.contrast.params.is_neutral(),
            Stage::Dering => p.dering.enabled && !p.dering.params.is_neutral(),
            Stage::Tone => p.tone.enabled && !p.tone.params.is_neutral(),
            Stage::Curves => p.curves.enabled && !p.curves.params.is_neutral(),
            Stage::PhotoColor => p.photocolor.enabled && !p.photocolor.params.is_neutral(),
            Stage::Color => p.color.enabled && !p.color.params.is_neutral(),
            Stage::Mineral => p.mineral.enabled && (!p.mineral.params.is_neutral() || self.document.as_ref().is_some_and(|d| d.luminance.is_some())),
            Stage::SunColor => p.suncolor.enabled && !p.suncolor.params.is_neutral(),
            Stage::Layers => p.layers.iter().any(|l| l.enabled && l.overrides_any()),
            Stage::Rings => role(LayerRole::Rings),
            Stage::Moons => role(LayerRole::Moons),
        }
    }

    fn side_panel(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        egui::Panel::bottom("histogram").frame(egui::Frame::NONE.inner_margin(egui::Margin::symmetric(0, 10))).show(ui, |ui| {
            ui.label(caps(tl("ISTOGRAMMA")));
            ui.add_space(theme::space::S);
            match &self.histogram {
                Some(h) => h.show(ui),
                None => Histogram::default().show(ui),
            }
        });

        let mut actions = Vec::new();
        let opening = self.opening_state();
        let customised = opening != panels::Opening::Preprocessed;
        let untouched = opening == panels::Opening::Raw;
        let running = self.worker.progress();
        let scales_before = self.pipeline.wavelet.params.scale_mode;
        let Some((kind, mono)) = self.document.as_ref().map(|d| (d.kind, d.source.is_mono())) else { return };
        let layout = self.prefs.panel_layout(kind, mono);
        let active: Vec<(prefs_group::Group, usize)> = layout.iter().map(|(g, stages)| (*g, stages.iter().filter(|s| self.stage_active(**s)).count())).collect();
        let adapt = self.prefs.adapt_to_image;
        let one_open = self.prefs.one_group_open;
        let bxt_configured = self.prefs.bxt_executable().is_some();
        let deconv_report = self.last_output.as_ref().and_then(|o| o.deconv);
        let active_stages: std::collections::HashSet<Stage> = layout.iter().flat_map(|(_, s)| s.iter().copied()).filter(|s| self.stage_active(*s)).collect();
        let gpu = self.prefs.use_gpu && matches!(ps_core::gpu::status(), Some(Some(_)));
        let last_export = self.last_export.clone();
        // Only the export tab shows the name (building it may read the file's date).
        let export_stem = if self.prefs.current_area == prefs_group::Group::Export.name() { self.export_stem() } else { String::new() };
        let resize_image = self.document.as_ref().map(|d| self.full_output.as_ref().filter(|o| o.source_gen == d.source_gen && o.image.width() == d.source.width()).map_or_else(|| Arc::clone(&d.source), |o| Arc::clone(&o.image)));
        let Self { document, pipeline, prefs, psf_preview, psf_note, timings, warnings, selected_layer, show_mask, layer_tool, brush, mask_thumbs, side_width, show_ringing, smart_view, smart_legend, smart_compare, resize_preview, identify, last_output, .. } = self;
        let tone_histogram = last_output.as_ref().map(|o| &o.tone_histogram);
        let curves_histogram = last_output.as_ref().map(|o| &o.curves_histogram);
        let Some(doc) = document else { return };
        smart_compare.update(ui.ctx(), &doc.source, doc.source_gen, &pipeline.smart.params, &doc.analysis);
        let why = explain(doc.kind, &doc.analysis, pipeline.wavelet.params.scale_mode);
        let auto_deconv = ps_core::preset::auto_deconv(doc.kind, &doc.analysis, prefs.auto_strength);
        let auto_smart = ps_core::preset::smart_structure(doc.kind, &doc.analysis);
        // What the wavelet controls' resets go back to: the image's starting point.
        let scales = pipeline.wavelet.params.scale_mode;
        let wavelet_start = ps_core::preset::wavelet_structure(doc.kind, &doc.analysis, scales);
        let detail_cutoff = ps_core::preset::detail_cutoff(&doc.analysis);
        let luminance_name = doc.luminance.as_ref().map(|l| l.name.clone());
        // What the final sharpening works on: the sharpening steps shown before it.
        let upstream: Vec<(Stage, bool)> = layout
            .iter()
            .flat_map(|(_, stages)| stages.iter().copied())
            .take_while(|s| *s != Stage::Smart)
            .filter(|s| matches!(s, Stage::Bxt | Stage::Deconv | Stage::Wavelet | Stage::Contrast))
            .map(|s| (s, active_stages.contains(&s)))
            .collect();

        // Fixed at the top: what the image is, the automatic settings, the areas.
        ui.add_space(theme::space::L);
        ui.set_max_width(ui.available_width());
        let start_view = panels::StartView {
            subject: panels::Subject { kind: doc.kind, body: doc.capture.body },
            analysis: &doc.analysis,
            named: doc.capture.info.body,
            customised,
            untouched,
        };
        panels::start_row(ui, start_view, &mut prefs.auto_strength, &mut actions);
        ui.add_space(theme::space::L);
        // The module areas, then the export tab.
        let mut tabs = active.clone();
        tabs.push((prefs_group::Group::Identify, 0));
        tabs.push((prefs_group::Group::Export, 0));
        let mut area = tabs.iter().map(|(g, _)| *g).find(|g| g.name() == prefs.current_area).or(tabs.first().map(|(g, _)| *g));
        if let Some(current) = &mut area {
            panels::area_tabs(ui, &tabs, current);
            prefs.current_area = current.name().to_string();
        }
        ui.add_space(theme::space::M);

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            // Nothing inside may ask for more than the panel's width, or the panel grows.
            ui.set_max_width(ui.available_width());
            *side_width = ui.available_width();
            if area == Some(prefs_group::Group::Identify) {
                let input = identify_input(doc);
                identify.tab(ui, &input, &mut prefs.identify, &prefs.profile, &mut actions);
            }
            if area == Some(prefs_group::Group::Export) {
                let sun = doc.capture.body == Some(ps_core::capture::Body::Sun);
                let centre = doc.analysis.disk.as_ref().map_or((doc.source.width() as f32 / 2.0, doc.source.height() as f32 / 2.0), |d| (d.cx, d.cy));
                let view = crate::resize_ui::ResizeView { kind: doc.kind, sun, image: resize_image.clone(), centre, preview: resize_preview };
                let resize = &mut prefs.resize;
                let mut name = panels::ExportName { base: &mut doc.export_name, keywords: &mut prefs.export_keywords, suffix: &mut prefs.export_suffix, stem: &export_stem };
                panels::export_body(ui, &mut prefs.export_format, &mut name, &doc.path, mono, last_export.as_deref(), &mut actions, |ui| crate::resize_ui::resize_body(ui, resize, view));
            }
            for (_, stages) in layout.into_iter().filter(|(g, _)| Some(*g) == area) {
                let stages_in_group = stages.clone();
                {
                    for stage in stages {
                        let status = StageStatus {
                            timing: timings.get(&stage),
                            running: running.filter(|p| p.stage == stage).map(|p| p.fraction),
                            warning: warnings.get(&stage).map(String::as_str),
                        };
                        let bxt_running = status.running;
                        let mut collapsed = !prefs.expanded.contains(&stage);
                        let was_collapsed = collapsed;
                        let is_active = active_stages.contains(&stage);
                        let layer_ui = |selected, show_mask, tool, brush, folded| LayerUi { selected, show_mask, tool, brush, collapsed: folded, kind: doc.kind, adapt, mono, wavelet_start: &wavelet_start, smart_start: &auto_smart, thumbs: &*mask_thumbs, annotation: None };
                        let buttons = match stage {
                            Stage::Align => panels::module_card(ui, stage, Some(&mut pipeline.align.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::align_body(ui, &mut pipeline.align.params, mono)
                            }),
                            Stage::Bxt => panels::module_card(ui, stage, Some(&mut pipeline.bxt.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::bxt_body(ui, &mut pipeline.bxt.params, bxt_configured, bxt_running, &why.bxt, &mut actions)
                            }),
                            Stage::Deconv => panels::module_card(ui, stage, Some(&mut pipeline.deconv.enabled), is_active, &mut collapsed, status, |ui| {
                                let (auto, optics, probe) = (&auto_deconv, doc.analysis.optics, psf_note.as_deref());
                                let view = panels::DeconvView { auto, optics, preview: psf_preview, why: &why.deconv, gpu, report: deconv_report, probe };
                                panels::deconvolution_body(ui, &mut pipeline.deconv.params, view, &mut actions)
                            }),
                            Stage::Denoise => panels::module_card(ui, stage, Some(&mut pipeline.denoise.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::denoise_body(ui, &mut pipeline.denoise.params, mono, Some(&why.denoise), Some(&ps_core::noise_net::network_label(doc.kind, doc.analysis.planet)))
                            }),
                            Stage::Wavelet => panels::module_card(ui, stage, Some(&mut pipeline.wavelet.enabled), is_active, &mut collapsed, status, |ui| {
                                let view = panels::WaveletView { start: &wavelet_start, why: Some(&why.wavelet), cutoff: detail_cutoff };
                                panels::wavelet_body(ui, &mut pipeline.wavelet.params, view)
                            }),
                            Stage::Smart => panels::module_card(ui, stage, Some(&mut pipeline.smart.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::smart_body(ui, &mut pipeline.smart.params, &auto_smart, &mut *smart_view, *smart_legend, &upstream, panels::LearnedView { analysis: &doc.analysis, thumbs: smart_compare.thumbs(), busy: smart_compare.busy() })
                            }),
                            Stage::Contrast => panels::module_card(ui, stage, Some(&mut pipeline.contrast.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::contrast_body(ui, &mut pipeline.contrast.params)
                            }),
                            Stage::Dering => panels::module_card(ui, stage, Some(&mut pipeline.dering.enabled), is_active, &mut collapsed, status, |ui| {
                                let state = layer_ui(&mut *selected_layer, &mut *show_mask, &mut *layer_tool, &mut *brush, &mut prefs.collapsed_sections);
                                panels::dering_body(ui, &mut pipeline.dering.params, &why.dering, &mut *show_ringing, state)
                            }),
                            Stage::Tone => panels::module_card(ui, stage, Some(&mut pipeline.tone.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::tone_body(ui, &mut pipeline.tone.params, ToneParts::Levels, None, tone_histogram, &mut prefs.histogram_mode, mono, &mut actions)
                            }),
                            Stage::Curves => panels::module_card(ui, stage, Some(&mut pipeline.curves.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::tone_body(ui, &mut pipeline.curves.params, ToneParts::Curve, None, curves_histogram, &mut prefs.histogram_mode, mono, &mut actions)
                            }),
                            Stage::PhotoColor => panels::module_card(ui, stage, Some(&mut pipeline.photocolor.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::photocolor_body(ui, &mut pipeline.photocolor.params, doc.capture.body)
                            }),
                            Stage::Color => panels::module_card(ui, stage, Some(&mut pipeline.color.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::color_body(ui, &mut pipeline.color.params, mono)
                            }),
                            // Listed among the masks now (see `Preferences::upgraded`).
                            Stage::Rings | Stage::Moons => panels::CardButtons::default(),
                            // The list of masks, then the selected one's editor: no card to fold.
                            Stage::Layers => {
                                let mut state = layer_ui(&mut *selected_layer, &mut *show_mask, &mut *layer_tool, &mut *brush, &mut prefs.collapsed_sections);
                                state.annotation = Some(&mut identify.annotate);
                                panels::masks_area(ui, &mut pipeline.layers, state, &mut actions);
                                panels::CardButtons::default()
                            }
                            Stage::Mineral => panels::module_card(ui, stage, Some(&mut pipeline.mineral.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::mineral_body(ui, &mut pipeline.mineral.params, mono, luminance_name.as_deref(), &mut actions)
                            }),
                            Stage::SunColor => panels::module_card(ui, stage, Some(&mut pipeline.suncolor.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::suncolor_body(ui, &mut pipeline.suncolor.params, doc.capture.body == Some(ps_core::capture::Body::Sun))
                            }),
                        };
                        if buttons.auto {
                            actions.push(PanelAction::AutoModule(stage));
                        }
                        if buttons.reset {
                            actions.push(PanelAction::ResetModule(stage));
                        }
                        // Dering: opening its card is to paint, so the brush comes up with the painted
                        // area shown; closing it puts the brush away.
                        if stage == Stage::Dering && was_collapsed != collapsed {
                            if !collapsed {
                                (*selected_layer, *show_mask, *layer_tool) = (Some(RING_MASK), true, LayerTool::Brush);
                                pipeline.dering.enabled = true;
                            } else if *selected_layer == Some(RING_MASK) {
                                (*selected_layer, *show_mask, *layer_tool) = (None, false, LayerTool::None);
                            }
                        }
                        if collapsed {
                            prefs.expanded.remove(&stage);
                        } else if prefs.expanded.insert(stage) && one_open {
                            // Just opened: one card at a time in an area.
                            prefs.expanded.retain(|s| *s == stage || !stages_in_group.contains(s));
                        }
                    }
                }
            }
            ui.add_space(theme::space::L);
        });
        if self.pipeline.wavelet.params.scale_mode != scales_before {
            self.switch_scales(scales_before);
        }
        self.apply_actions(ctx, actions);
    }

    fn empty_state(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let mut action: Option<Choice> = None;
        let mut picked: Option<PathBuf> = None;
        ui.vertical_centered(|ui| {
            let width = ui.available_width().min(640.0);
            ui.add_space((ui.available_height() * 0.5 - 250.0).max(16.0));
            ui.allocate_ui_with_layout(egui::vec2(width, 0.0), egui::Layout::top_down(egui::Align::Center), |ui| {
                // Title and the main action.
                ui.label(ICON_PLANET.rich_text().size(theme::size::SPLASH).color(theme::ACCENT_DIM));
                ui.add_space(theme::space::M);
                ui.label(RichText::new(tl("Apri un'immagine planetaria, lunare o solare")).size(theme::size::LEAD).color(TEXT));
                ui.add_space(theme::space::XS);
                ui.label(RichText::new(tl("Il tipo viene riconosciuto e le impostazioni scelte automaticamente")).color(TEXT_WEAK));
                ui.add_space(theme::space::L + theme::space::M);
                if ui.add(primary_button(Some(ICON_FOLDER_OPEN), tl("Apri immagine…")).min_size(egui::vec2(200.0, 40.0))).clicked() {
                    self.open_dialog(ctx);
                }
                ui.add_space(theme::space::M);
                ui.label(RichText::new(tl("oppure trascinala qui · TIFF, FITS, PNG, JPEG")).small().color(TEXT_WEAK));

                // Several images: four ways to combine them, as cards.
                ui.add_space(24.0);
                ui.label(caps(&tr!("PIÙ IMMAGINI INSIEME", "SEVERAL IMAGES TOGETHER")));
                ui.label(RichText::new(tr!("Trascinale qui e scegli cosa farne, oppure parti da una di queste", "Drop them here and choose what to do, or start from one of these")).small().color(TEXT_WEAK));
                ui.add_space(theme::space::L);
                let cards = [
                    (Choice::Mosaic, ICON_GRID_VIEW, tl("Mosaico"), tl("Unisce immagini sovrapposte dello stesso soggetto (Luna, Sole…) in una sola")),
                    (Choice::Derotation, ICON_AUTORENEW, tl("Derotazione (somma)"), tl("Somma più riprese di Giove o Saturno portandole allo stesso istante")),
                    (Choice::MonoRgb, ICON_PALETTE, tl("Mono → RGB"), tl("Unisce le immagini monocromatiche R, G, B (e L) in una a colori, allineandole")),
                    (Choice::MonoRgbDerotated, ICON_PALETTE, tl("Mono → RGB derotato"), tl("Come Mono → RGB, ma ogni canale è portato allo stesso istante con la rotazione del pianeta (Giove, Saturno)")),
                    (Choice::Animation, ICON_MOVIE, tl("Animazione"), tl("Una GIF o un video fluido dalle riprese, allineate e interpolate fra l'una e l'altra")),
                ];
                let gap = 10.0;
                let card_w = (width - gap) / 2.0;
                for row in cards.chunks(2) {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = gap;
                        for &(choice, icon, title, help) in row {
                            let (rect, response) = ui.allocate_exact_size(egui::vec2(card_w, 76.0), egui::Sense::click());
                            let hovered = response.hovered();
                            ui.painter().rect(
                                rect,
                                egui::CornerRadius::same(theme::radius::L),
                                if hovered { theme::SURFACE_HOVER } else { theme::SURFACE },
                                egui::Stroke::new(1.0, if hovered { theme::ACCENT_DIM } else { theme::BORDER }),
                                egui::StrokeKind::Inside,
                            );
                            let mut inner = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink2(egui::vec2(14.0, 10.0))).layout(egui::Layout::left_to_right(egui::Align::Center)));
                            inner.label(icon.rich_text().size(26.0).color(if hovered { theme::ACCENT } else { theme::ACCENT_DIM }));
                            inner.add_space(theme::space::L);
                            inner.vertical(|ui| {
                                ui.spacing_mut().item_spacing.y = 2.0;
                                ui.add(egui::Label::new(RichText::new(title).size(theme::size::BODY).strong().color(TEXT)).truncate());
                                ui.add(egui::Label::new(RichText::new(help).size(theme::size::SMALL).color(TEXT_WEAK)));
                            });
                            if response.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                                action = Some(choice);
                            }
                        }
                    });
                    ui.add_space(gap);
                }

                // The images opened last, a click away.
                let recents: Vec<_> = self.prefs.recent_files.iter().take(6).cloned().collect();
                if !recents.is_empty() {
                    ui.add_space(theme::space::L);
                    ui.label(caps(&tr!("RECENTI", "RECENT")));
                    ui.add_space(theme::space::M);
                    let rgap = 8.0;
                    let rw = (width - 2.0 * rgap) / 3.0;
                    for row in recents.chunks(3) {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = rgap;
                            for r in row {
                                let tex = self.recent_textures.entry(r.path.clone()).or_insert_with(|| ctx.load_texture(format!("recent-{}", r.path.display()), ColorImage::from_rgb(r.size, &r.thumb), egui::TextureOptions::LINEAR));
                                let (rect, response) = ui.allocate_exact_size(egui::vec2(rw, 60.0), egui::Sense::click());
                                let hovered = response.hovered();
                                ui.painter().rect(
                                    rect,
                                    egui::CornerRadius::same(theme::radius::M),
                                    if hovered { theme::SURFACE_HOVER } else { theme::PANEL },
                                    egui::Stroke::new(1.0, if hovered { theme::ACCENT_DIM } else { theme::BORDER }),
                                    egui::StrokeKind::Inside,
                                );
                                // The thumbnail in a fixed square, the name beside it.
                                let square = egui::Rect::from_min_size(rect.min + egui::vec2(6.0, 6.0), egui::vec2(48.0, 48.0));
                                ui.painter().rect_filled(square, egui::CornerRadius::same(theme::radius::S), egui::Color32::BLACK);
                                let (tw, th) = (r.size[0] as f32, r.size[1] as f32);
                                let k = (48.0 / tw.max(1.0)).min(48.0 / th.max(1.0)).min(1.0);
                                let img_rect = egui::Rect::from_center_size(square.center(), egui::vec2(tw * k, th * k));
                                ui.painter().image(tex.id(), img_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                                let mut inner = ui.new_child(egui::UiBuilder::new().max_rect(egui::Rect::from_min_max(square.right_top() + egui::vec2(8.0, 0.0), rect.right_bottom() - egui::vec2(6.0, 6.0))).layout(egui::Layout::left_to_right(egui::Align::Center)));
                                inner.add(egui::Label::new(RichText::new(file_name(&r.path)).size(theme::size::SMALL).color(TEXT)).wrap());
                                if response.on_hover_text(r.path.display().to_string()).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                                    picked = Some(r.path.clone());
                                }
                            }
                        });
                        ui.add_space(rgap);
                    }
                }

                // A short tip that changes every few seconds.
                let tips = [
                    tr!("Tieni premuto Spazio per vedere l'originale mentre lavori", "Hold Space to see the original while you work"),
                    tr!("Doppio clic su un cursore per tornare al suo valore predefinito", "Double-click a slider to return to its default value"),
                    tr!("Premi ? per l'elenco di tutte le scorciatoie", "Press ? for the list of all shortcuts"),
                ];
                let tip = (ctx.input(|i| i.time) / 6.0) as usize % tips.len();
                ui.add_space(theme::space::L);
                ui.label(RichText::new(format!("{} {}", ICON_LIGHTBULB.codepoint, tips[tip])).small().color(TEXT_WEAK));
                ctx.request_repaint_after(Duration::from_secs(1));
                if !self.status.is_empty() {
                    ui.add_space(theme::space::L);
                    ui.label(RichText::new(&self.status).small().color(TEXT_WEAK));
                }
            });
        });
        match action {
            Some(Choice::Mosaic) => self.new_mosaic_dialog(ctx),
            Some(Choice::Derotation) => self.new_derotation_dialog(ctx),
            Some(Choice::MonoRgb) => self.new_mono_rgb_dialog(ctx, false),
            Some(Choice::MonoRgbDerotated) => self.new_mono_rgb_dialog(ctx, true),
            Some(Choice::Animation) => self.new_animation_dialog(ctx),
            None => {}
        }
        if let Some(path) = picked {
            if path.exists() {
                self.load(ctx, path);
            } else {
                self.prefs.recent_files.retain(|r| r.path != path);
                self.status = tr!("{} non c'è più", "{} is gone", file_name(&path));
            }
        }
    }
}

impl eframe::App for PlanetaryStudio {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = &ui.ctx().clone();
        self.handle_events(ctx);
        self.handle_shortcuts(ctx);
        // Quit asked for while a file dialog was open: close now, through the usual path.
        if crate::dialog::take_quit_request() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        self.close_guard(ctx);
        // The title carries an asterisk while there are adjustments not yet exported or saved.
        let title = if self.document.is_none() {
            "Planetary Studio".to_string()
        } else {
            format!("{}{} — Planetary Studio", self.doc_title, if self.has_unsaved_changes() { " *" } else { "" })
        };
        if title != self.shown_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.shown_title = title;
        }
        // The subject (the Identifica tab's choice over the file name, log or pixels) decides which
        // subject-specific cards exist.
        self.prefs.subject = self.document.as_ref().and_then(|d| self.identify.chosen_body().or(identify_input(d).body));
        self.drop_stale_selection();
        self.drop_stale_snapshot(ctx);
        let tool_before = (self.layer_tool, self.selected_layer);
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::unique(panels::MASK_TOOLS_SHOWN), false);
            d.insert_temp(egui::Id::unique(panels::RINGING_SWITCH_SHOWN), false);
            d.insert_temp(egui::Id::unique(panels::SMART_ZONES_SWITCH_SHOWN), false);
            d.insert_temp(egui::Id::unique(panels::TONE_DRAGGING), false);
        });
        self.settings.show(ctx, &mut self.prefs);
        self.about_window(ctx);
        self.opening_questions(ctx);
        self.shortcuts_window(ctx);
        self.export_toast(ctx);
        Self::drop_hint(ctx);
        ps_core::gpu::set_enabled(self.prefs.use_gpu);
        if let Some(MosaicOutcome::Create { mosaic, images, keep_settings }) = self.mosaic_editor.show(ctx) {
            self.create_mosaic(ctx, mosaic, images, keep_settings);
        }
        if let Some(rx) = &self.pending_mosaic
            && let Ok((mosaic, images)) = rx.try_recv()
        {
            self.pending_mosaic = None;
            self.create_mosaic(ctx, mosaic, images, false);
        }
        if let Some(DerotationOutcome::Create(job)) = self.derotation_editor.show(ctx) {
            self.create_derotation(ctx, *job);
        }
        if let Some(MonoRgbOutcome::Create(job)) = self.mono_rgb_editor.show(ctx) {
            self.create_mono_rgb(ctx, *job);
        }
        self.animation_editor.show(ctx);
        if let Some(c) = self.choose_editor.show(ctx) {
            self.do_choice(ctx, c.choice, c.paths, c.body);
        }

        let bar = |fill| egui::Frame::new().fill(fill).stroke(Stroke::new(1.0, BORDER));
        egui::Panel::top("top").frame(bar(theme::PANEL).inner_margin(egui::Margin::symmetric(12, 6))).show(ui, |ui| self.top_bar(ctx, ui));
        egui::Panel::bottom("status").frame(bar(theme::BG).inner_margin(egui::Margin::symmetric(12, 4))).show(ui, |ui| self.status_bar(ui));
        if self.document.is_some() {
            let side = egui::Frame::new().fill(theme::PANEL).inner_margin(egui::Margin::symmetric(14, 4));
            let max = (ctx.content_rect().width() * 0.45).clamp(360.0, 640.0);
            egui::Panel::right("side").frame(side).resizable(true).default_size(420.0).min_size(340.0).max_size(max).show(ui, |ui| self.side_panel(ctx, ui));
            self.settle_tools(ctx, tool_before);
            self.tone_dragging = ctx.data(|d| d.get_temp::<bool>(egui::Id::unique(panels::TONE_DRAGGING))).unwrap_or(false);
        }

        // Hold Space to peek at the original. With the pointer over the image the key is ours
        // even if a value box has focus; and it must not "press" a focused button or switch.
        let space = ctx.input(|i| i.key_down(Key::Space));
        let peek = space && (self.canvas_hovered || !ctx.egui_wants_keyboard_input());
        if peek && ctx.input(|i| i.key_pressed(Key::Space)) {
            ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Space));
            ctx.memory_mut(|m| {
                if let Some(id) = m.focused() {
                    m.surrender_focus(id);
                }
            });
        }
        let compare = if peek { Compare::Before } else { self.compare };
        self.shown_compare = compare;
        let busy = self.worker.is_busy();
        let roi_mode = self.preview == PreviewMode::Roi;
        self.sync_shape_to_viewer();
        let layer_selected = self.selected_layer.is_some();
        let tool = match self.layer_tool {
            _ if self.cropping => Tool::Crop,
            _ if self.picking == Some(PickTarget::Psf) => Tool::Probe,
            _ if self.picking.is_some() => Tool::Pick,
            LayerTool::Shape if layer_selected => Tool::Shape,
            LayerTool::Brush if layer_selected => Tool::Brush,
            _ => self.tool,
        };
        // ⌥/Alt flips the brush between adding and removing.
        let erase = self.brush.erase ^ ctx.input(|i| i.modifiers.alt);
        let brush = (tool == Tool::Brush).then_some(BrushView { radius: self.brush.radius, erase });
        let show_mask = self.show_mask && self.selected_layer.is_some();
        let (mut crop_action, mut stroke, mut picked, mut new_preview) = (None, None, None, None);
        let mut probed = None;
        let (mut end_comparison, mut retake) = (None, false);
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(theme::BG)).show(ui, |ui| {
            if let Some(doc) = &mut self.document {
                let out = doc.viewer.show(ui, compare, tool, roi_mode, busy, show_mask, brush);
                {
                    let viewer = &doc.viewer;
                    let painter = ui.painter_at(viewer.canvas());
                    // Once the names are part of the image they are no longer drawn on top.
                    if doc.overlay().is_none() {
                        self.identify.paint(&painter, |p| viewer.image_to_screen(p), viewer.zoom(), &self.prefs.identify);
                    }
                }
                // Floating over the image: how it is shown (top) and the zoom (bottom right).
                let canvas = ui.max_rect();
                // A bar of its own size, with its `pivot` corner at `at`, above the image.
                // Not over the mosaic window.
                let covered = self.mosaic_editor.open || self.derotation_editor.open || self.mono_rgb_editor.open || self.choose_editor.open || self.animation_editor.open || self.deconv_compare.is_some();
                let float = |ui: &mut egui::Ui, id: &str, at: egui::Pos2, pivot: egui::Align2, add: &mut dyn FnMut(&mut egui::Ui)| {
                    if covered {
                        return;
                    }
                    egui::Area::new(egui::Id::unique(id)).order(egui::Order::Foreground).fixed_pos(at).pivot(pivot).show(ui.ctx(), |ui| {
                        egui::Frame::new().fill(egui::Color32::from_black_alpha(190)).corner_radius(theme::radius::WINDOW).inner_margin(egui::Margin::symmetric(8, 5)).show(ui, |ui| {
                            ui.horizontal(|ui| add(ui));
                        });
                    });
                };
                let mut preview = self.preview;
                float(ui, "view-bar", canvas.center_top() + Vec2::new(0.0, 8.0), egui::Align2::CENTER_TOP, &mut |ui| {
                    segmented(ui, &mut preview, &[(PreviewMode::Full, Some(ICON_IMAGE), tl("Intera")), (PreviewMode::Roi, Some(ICON_CROP_FREE), "ROI")]);
                    if roi_mode {
                        segmented(ui, &mut self.tool, &[(Tool::Pan, Some(ICON_PAN_TOOL), ""), (Tool::Roi, Some(ICON_HIGHLIGHT_ALT), "")]);
                    }
                    ui.add(egui::Separator::default().vertical().spacing(10.0));
                    // While comparing, the versions take the place of before/after.
                    let options = if self.snapshot.is_some() {
                        vec![(Compare::OnlyA, None, "A"), (Compare::Ab, Some(ICON_VERTICAL_SPLIT), "A | B"), (Compare::After, None, "B")]
                    } else {
                        vec![(Compare::After, None, tl("Dopo")), (Compare::Before, None, tl("Prima")), (Compare::Split, Some(ICON_VERTICAL_SPLIT), tl("Divisa"))]
                    };
                    segmented(ui, &mut self.compare, &options);
                });
                if self.snapshot.is_some() {
                    float(ui, "compare-bar", canvas.center_bottom() - Vec2::new(0.0, 12.0), egui::Align2::CENTER_BOTTOM, &mut |ui| {
                        ui.label(ICON_COMPARE.rich_text().color(ACCENT));
                        ui.label(RichText::new(tl("Confronto")).strong().color(TEXT));
                        ui.add(egui::Separator::default().vertical().spacing(10.0));
                        if ui.button(tl("Tieni A")).on_hover_text(tl("Torna alle impostazioni della versione A e chiudi il confronto")).clicked() {
                            end_comparison = Some(true);
                        }
                        if ui.button(tl("Tieni B")).on_hover_text(tl("Tieni le modifiche attuali e dimentica A")).clicked() {
                            end_comparison = Some(false);
                        }
                        if ui.add(egui::Button::new((ICON_RESTART_ALT.rich_text(), tl("A = ora"))).frame(false)).on_hover_text(tl("Fissa il risultato attuale come nuova versione A")).clicked() {
                            retake = true;
                        }
                    });
                }
                if preview != self.preview {
                    new_preview = Some(preview);
                }
                let viewer = &mut doc.viewer;
                float(ui, "zoom-bar", canvas.right_bottom() - Vec2::new(12.0, 12.0), egui::Align2::RIGHT_BOTTOM, &mut |ui| {
                    if icon_button(ui, ICON_ZOOM_OUT, tl("Riduci")).clicked() {
                        viewer.set_zoom(viewer.zoom() / 1.25);
                    }
                    let zoom_label = RichText::new(format!("{:.0}%", viewer.zoom() * 100.0)).monospace().color(TEXT);
                    if ui.add(egui::Button::new(zoom_label).frame(false)).on_hover_text(tr!("Dimensione reale ({})", "Actual size ({})", ctx.format_shortcut(&ACTUAL_SIZE))).clicked() {
                        viewer.set_zoom(1.0);
                    }
                    if icon_button(ui, ICON_ZOOM_IN, tl("Ingrandisci")).clicked() {
                        viewer.set_zoom(viewer.zoom() * 1.25);
                    }
                    if tool_button(ui, ICON_FIT_SCREEN, tl("Adatta"), false).on_hover_text(tl("Adatta alla finestra (F)")).clicked() {
                        viewer.request_fit();
                    }
                });
                self.hover = out.hover_pixel;
                crop_action = out.crop_action;
                stroke = out.stroke;
                picked = out.picked;
                probed = out.probed;
                if out.roi_toggle {
                    self.tool = if self.tool == Tool::Roi { Tool::Pan } else { Tool::Roi };
                }
                self.canvas_hovered = out.hovered;
            } else {
                self.empty_state(ctx, ui);
            }
        });
        if let Some(mode) = new_preview {
            self.set_preview(mode);
        }
        if let Some(keep_a) = end_comparison {
            self.end_comparison(ctx, keep_a);
        }
        if retake {
            self.take_snapshot(ctx);
        }
        self.sync_shape_from_viewer();
        if let Some(points) = stroke {
            self.add_stroke(points, erase);
        }
        if let (Some(target), Some(pixel)) = (self.picking, picked) {
            self.pick_point(target, pixel);
        }
        if let Some(area) = probed {
            self.measure_psf(area);
        }
        self.deconv_compare_window(ctx);
        self.instrument_window(ctx);
        self.setups_window(ctx);
        self.batch_windows(ctx);
        // The search runs and the names show while the button is on or its tab is the one showing.
        self.identify.open = self.document.is_some() && (self.identify.active || self.prefs.current_area == prefs_group::Group::Identify.name());
        self.update_annotation(ctx);
        if let Some((mirror, degrees)) = self.identify.take_upright() {
            self.rotate(ctx, |g| *g = g.then(mirror, degrees));
        }
        // The subject chosen in the Identifica tab is the image's subject too.
        let tab_body = self.identify.chosen_body();
        if tab_body != self.last_identify_body {
            self.last_identify_body = tab_body;
            if let Some(body) = tab_body
                && self.document.as_ref().is_some_and(|d| d.capture.body != Some(body))
            {
                let subject = self.subject_for(body);
                self.apply_actions(ctx, vec![PanelAction::SelectSubject(subject)]);
            }
        }
        if let Some(doc) = &self.document {
            let input = identify_input(doc);
            self.identify.update(ctx, &input);
            // The automatic identification shows the moons only for a few seconds.
            if let Some((started, settled)) = &mut self.auto_identify {
                let busy = self.identify.busy();
                if !self.identify.active {
                    self.auto_identify = None;
                } else if busy {
                    *settled = None;
                } else if settled.is_none() && started.elapsed() > Duration::from_millis(1500) {
                    *settled = Some(Instant::now());
                }
                if let Some((started, settled)) = self.auto_identify
                    && (settled.is_some_and(|t| t.elapsed() > Duration::from_secs(4)) || started.elapsed() > Duration::from_secs(40))
                {
                    self.identify.active = false;
                    self.auto_identify = None;
                }
                ctx.request_repaint_after(Duration::from_millis(500));
            }
            if self.identify.busy() {
                ctx.request_repaint_after(Duration::from_millis(50));
            }
        }
        if let Some(action) = crop_action {
            self.finish_crop(ctx, action);
        }

        let interacting = ctx.input(|i| i.pointer.any_down());
        self.history.track(&self.pipeline, interacting);
        self.submit_if_changed(ctx);
        if busy || self.io_busy.is_some() || self.worker.is_background_busy() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, STORAGE_PREFS, &self.prefs);
    }
}

/// Show `path` in the system's file manager.
fn reveal(path: &Path) {
    let mut cmd;
    if cfg!(target_os = "macos") {
        cmd = std::process::Command::new("open");
        cmd.arg("-R").arg(path);
    } else if cfg!(target_os = "windows") {
        cmd = std::process::Command::new("explorer");
        cmd.arg(format!("/select,{}", path.display()));
    } else {
        cmd = std::process::Command::new("xdg-open");
        cmd.arg(path.parent().unwrap_or(path));
    }
    if let Err(e) = cmd.spawn() {
        ps_core::log::error("app", &format!("impossibile mostrare {}: {e}", path.display()));
    }
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// One line per finished render (ROI or full, not the reduced previews): how long, what went
/// wrong, and the settings when they changed since the last one logged.
fn log_render(out: &crate::worker::Output) {
    use std::sync::Mutex;
    static LAST: Mutex<String> = Mutex::new(String::new());
    if out.proxy {
        return;
    }
    let total: f64 = out.timings.iter().map(|t| t.elapsed.as_secs_f64()).sum();
    let what = match out.roi {
        Some(r) => format!("anteprima di una zona {}×{} a ({}, {})", r.w, r.h, r.x, r.y),
        None => "immagine intera".to_string(),
    };
    ps_core::log::info("elaborazione", &format!("{what} in {total:.2} s"));
    for (stage, message) in &out.warnings {
        ps_core::log::warn("elaborazione", &format!("{stage:?}: {message}"));
    }
    if let Ok(json) = serde_json::to_string(&out.pipeline)
        && let Ok(mut last) = LAST.lock()
        && *last != json
    {
        ps_core::log::info("impostazioni", &json);
        *last = json;
    }
}
