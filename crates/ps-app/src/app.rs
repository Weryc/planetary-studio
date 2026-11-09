use crate::history::History;
use crate::mosaic_ui::{MosaicEditor, MosaicOutcome};
use crate::panels::{self, BrushSettings, LayerTemplate, LayerTool, LayerUi, MaskThumbView, PanelAction, PickTarget, PsfPreview, StageStatus, ToneParts, format_duration};
use crate::prefs::{self as prefs_group, Preferences, SettingsWindow};
use crate::theme::{self, ACCENT, BORDER, TEXT, TEXT_WEAK};
use crate::viewer::{BrushView, Compare, CropAction, Overlay, ShapeKind, Tool, Viewer};
use crate::widgets::{Histogram, icon_button, segmented, tool_button};
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
use ps_core::preset::{add_role_layer, auto_pipeline_with, explain, initial_pipeline, neutral_pipeline, restructure};
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
const EXPORT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::E);
const UNDO: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Z);
const REDO: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::Z);
/// The platform's standard shortcut for preferences (⌘, on macOS, Ctrl+, elsewhere).
const SETTINGS: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Comma);
const FIT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::NONE, Key::F);
const ACTUAL_SIZE: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Num1);
/// Export next to the original, in the last format used.
const QUICK_EXPORT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::E);
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
    Loaded { path: PathBuf, image: Image, display: ColorImage, analysis: Box<Analysis>, elapsed: Duration, mosaic: Option<Box<Mosaic>>, keep_settings: bool },
    LuminanceLoaded { name: String, plane: Vec<f32> },
    Exported { path: PathBuf, elapsed: Duration },
    Failed(String),
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
    /// Base name of exported files (the original's unless edited).
    export_name: String,
    /// The frames this image was joined from, and how.
    mosaic: Option<Mosaic>,
    /// Subject, time and log of the capture.
    capture: CaptureState,
    /// The equipment it was taken with.
    setup: Setup,
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
        ps_core::psf::Psf::Gaussian { .. } => "Gaussiana",
        ps_core::psf::Psf::Moffat { .. } => "Moffat",
        ps_core::psf::Psf::Airy { .. } => "Airy",
        ps_core::psf::Psf::Measured { .. } => "misurata",
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
    mosaic_editor: MosaicEditor,
    /// A saved mosaic's frames, being loaded.
    pending_mosaic: Option<Receiver<MosaicFrames>>,
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
    /// The equipment window (from the top bar).
    instrument_open: bool,
    /// A levels slider or curve point was being dragged in the last panel pass.
    tone_dragging: bool,
    /// Show burnt and crushed pixels, and how many there are.
    show_clipping: bool,
    clipped: Option<usize>,
    snapshot: Option<Snapshot>,
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
        let pipeline = Pipeline::default();
        let (events_tx, events_rx) = channel();
        let mut app = Self {
            history: History::new(&pipeline),
            pipeline,
            prefs,
            settings: SettingsWindow::default(),
            mosaic_editor: MosaicEditor::default(),
            pending_mosaic: None,
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
            instrument_open: false,
            tone_dragging: false,
            show_clipping: false,
            clipped: None,
            snapshot: None,
            last_export: None,
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

    // ---- file I/O ----------------------------------------------------------------------

    fn open_dialog(&mut self, ctx: &egui::Context) {
        let picked = rfd::FileDialog::new()
            .set_title("Apri immagine")
            .add_filter("Immagini astronomiche e mosaici", &[ps_io::READ_EXTENSIONS, &[MOSAIC_EXTENSION]].concat())
            .pick_file();
        if let Some(path) = picked {
            self.load(ctx, path);
        }
    }

    /// Several images: a new mosaic of them.
    fn new_mosaic_dialog(&mut self, ctx: &egui::Context) {
        if let Some(paths) = rfd::FileDialog::new().set_title("Immagini da unire in un mosaico").add_filter("Immagini", ps_io::READ_EXTENSIONS).pick_files()
            && !paths.is_empty()
        {
            self.mosaic_editor.start(ctx, paths);
        }
    }

    /// Join the frames at full size off the UI thread, then open the result like a file.
    fn create_mosaic(&mut self, ctx: &egui::Context, mosaic: Mosaic, images: Vec<Option<Arc<Image>>>, keep_settings: bool) {
        let Some(first) = mosaic.tiles.iter().zip(&images).find(|(t, i)| t.enabled && i.is_some()).map(|(t, _)| t.path.clone()) else { return };
        self.io_busy = Some(format!("Creazione del mosaico di {} immagini…", mosaic.tiles.iter().filter(|t| t.enabled).count()));
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
                    Event::Loaded { path: first, image, display, analysis, elapsed: start.elapsed(), mosaic: Some(Box::new(mosaic)), keep_settings }
                }
                None => Event::Failed("Nessuna immagine allineata da unire".into()),
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
                self.status = format!("Impossibile aprire {}: {e}", file_name(path));
                return;
            }
        };
        self.io_busy = Some(format!("Apertura del mosaico {}…", file_name(path)));
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
                    tx.send(Event::Failed(format!("Mosaico incompleto: {e}"))).ok();
                }
            }
            ctx.request_repaint();
        });
        self.pending_mosaic = Some(frames_rx);
    }

    /// Decode, analyse and prepare the display texture off the UI thread.
    fn load(&mut self, ctx: &egui::Context, path: PathBuf) {
        if path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(MOSAIC_EXTENSION)) {
            self.open_saved_mosaic(ctx, &path);
            return;
        }
        self.io_busy = Some(format!("Apertura di {}…", file_name(&path)));
        let tx = self.events_tx.clone();
        let ctx = ctx.clone();
        let max_side = ctx.input(|i| i.max_texture_side);
        std::thread::spawn(move || {
            let start = Instant::now();
            let event = match ps_io::load(&path) {
                Ok(image) => {
                    let (display, analysis) = rayon::join(|| worker::display_image(&image, max_side), || Box::new(analyze(&image)));
                    Event::Loaded { path, image, display, analysis, elapsed: start.elapsed(), mosaic: None, keep_settings: false }
                }
                Err(e) => Event::Failed(format!("Impossibile aprire {}: {e}", file_name(&path))),
            };
            tx.send(event).ok();
            ctx.request_repaint();
        });
    }

    /// Load a mono image as LRGB luminance and register it onto the colour image.
    fn load_luminance(&mut self, ctx: &egui::Context) {
        let Some(doc) = &self.document else { return };
        let Some(path) = rfd::FileDialog::new().set_title("Immagine mono per la luminanza").add_filter("Immagini", ps_io::READ_EXTENSIONS).pick_file() else {
            return;
        };
        self.io_busy = Some(format!("Caricamento di {}…", file_name(&path)));
        let original = Arc::clone(&doc.original);
        let tx = self.events_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let event = match ps_io::load(&path) {
                Ok(img) if img.width() != original.width() || img.height() != original.height() => Event::Failed(format!(
                    "La luminanza deve avere le stesse dimensioni dell'immagine ({}×{}), non {}×{}",
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
                Err(e) => Event::Failed(format!("Impossibile aprire {}: {e}", file_name(&path))),
            };
            tx.send(event).ok();
            ctx.request_repaint();
        });
    }

    /// Name of exported files (without extension): the base name and the chosen keywords.
    fn export_stem(&self) -> String {
        let Some(doc) = &self.document else { return "immagine".into() };
        let modules = Stage::ALL.into_iter().filter(|&s| self.stage_active(s)).map(crate::naming::module_tag).collect();
        let source = crate::naming::NameSource { base: &doc.export_name, original: &doc.path, kind: doc.kind, modules, suffix: &self.prefs.export_suffix };
        crate::naming::stem(&source, &self.prefs.export_keywords)
    }

    /// "Esporta come…": pick the file; its format becomes the one quick export uses.
    fn export_dialog(&mut self, ctx: &egui::Context) {
        let Some(doc) = &self.document else { return };
        let stem = self.export_stem();
        let picked = rfd::FileDialog::new()
            .set_title("Esporta immagine elaborata")
            .set_directory(doc.path.parent().unwrap_or(Path::new(".")))
            .set_file_name(format!("{stem}.{}", self.prefs.export_format))
            .add_filter("TIFF 16 bit", &["tif", "tiff"])
            .add_filter("PNG 16 bit", &["png"])
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
        let Some(doc) = &self.document else { return };
        let (dir, stem) = (doc.path.parent().map(Path::to_path_buf).unwrap_or_default(), self.export_stem());
        let ext = self.prefs.export_format.clone();
        let path = (1..)
            .map(|k| dir.join(if k == 1 { format!("{stem}.{ext}") } else { format!("{stem}-{k}.{ext}") }))
            .find(|p| !p.exists())
            .expect("some name is free");
        self.export_to(ctx, path);
    }

    fn export_to(&mut self, ctx: &egui::Context, path: PathBuf) {
        let Some(doc) = &self.document else { return };
        let pipeline = self.prefs.effective(&self.pipeline, doc.kind, doc.source.is_mono());
        // Usually already rendered (in the foreground or in the background while on a ROI).
        let ready = self.full_output.as_ref().filter(|o| o.source_gen == doc.source_gen && o.pipeline == pipeline).map(|o| Arc::clone(&o.image));
        let source = Arc::clone(&doc.source);
        let luminance = doc.luminance.as_ref().map(|l| l.cropped.clone());
        let bxt = self.prefs.bxt_executable();

        self.io_busy = Some(format!("Esportazione di {}…", file_name(&path)));
        let tx = self.events_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let start = Instant::now();
            let image = ready.unwrap_or_else(|| {
                let masks = ps_core::pipeline::masks_for(&pipeline, &source);
                let aux = Aux { masks: &masks, luminance, bxt_executable: bxt.as_deref(), ..Default::default() };
                let out = Processor::default().run(source, next_generation(), &pipeline, &aux, &|| false, &|_, _| {});
                out.expect("never cancelled").image
            });
            let event = match ps_io::save(&path, &image) {
                Ok(()) => Event::Exported { path, elapsed: start.elapsed() },
                Err(e) => Event::Failed(format!("Esportazione fallita: {e}")),
            };
            tx.send(event).ok();
            ctx.request_repaint();
        });
    }

    fn save_preset(&mut self) {
        let picked = rfd::FileDialog::new().set_title("Salva impostazioni").set_file_name("impostazioni.pspipe").add_filter("Impostazioni", &["pspipe"]).save_file();
        if let Some(path) = picked {
            let json = serde_json::to_string_pretty(&self.pipeline).expect("pipeline serialises");
            self.status = match std::fs::write(&path, json) {
                Ok(()) => format!("impostazioni salvate in {}", file_name(&path)),
                Err(e) => format!("salvataggio fallito: {e}"),
            };
        }
    }

    fn load_preset(&mut self) {
        let picked = rfd::FileDialog::new().set_title("Carica impostazioni").add_filter("Impostazioni", &["pspipe"]).pick_file();
        let Some(path) = picked else { return };
        let parsed = std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str::<Pipeline>(&s).map_err(|e| e.to_string()));
        self.status = match parsed {
            Ok(p) => {
                self.pipeline = p;
                format!("impostazioni {} caricate", file_name(&path))
            }
            Err(e) => format!("file di impostazioni non valido: {e}"),
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
        if self.prefs.auto_structure { initial_pipeline(kind, analysis, self.prefs.wavelet_scales) } else { self.with_scales(neutral_pipeline()) }
    }

    /// Back to generic values, dropping everything the analysis chose (layers stay).
    fn clear_auto(&mut self) {
        let layers = std::mem::take(&mut self.pipeline.layers);
        self.pipeline = Pipeline { layers, ..self.with_scales(neutral_pipeline()) };
    }

    fn instrument_action(&mut self, action: InstrumentAction) {
        match action {
            InstrumentAction::SetupChanged => {
                if let Some(doc) = &self.document {
                    self.prefs.setup = doc.setup.clone();
                    self.remember_setup();
                }
                self.equipment_changed();
            }
            InstrumentAction::BodyChanged => {
                if let Some(doc) = &mut self.document {
                    doc.capture.measure(&doc.original, &doc.analysis);
                }
            }
            InstrumentAction::LoadLog => {
                let picked = rfd::FileDialog::new().set_title("Log di ripresa").add_filter("Log (FireCapture, SharpCap, AutoStakkert)", &["txt", "log", "as3", "as4"]).pick_file();
                if let Some(path) = picked {
                    self.read_log(&path);
                }
            }
        }
    }

    /// The current equipment first among the recent ones (when anything of it is known).
    fn remember_setup(&mut self) {
        let setup = self.prefs.setup.clone();
        if setup.telescope.is_none() && setup.camera.is_none() {
            return;
        }
        let recent = &mut self.prefs.recent_setups;
        // Telescope and camera identify a setup; Barlow and filter changes replace it.
        recent.retain(|s| (s.telescope.as_ref().map(|t| &t.name), s.camera.as_ref().map(|c| &c.name)) != (setup.telescope.as_ref().map(|t| &t.name), setup.camera.as_ref().map(|c| &c.name)));
        recent.insert(0, setup);
        recent.truncate(5);
    }

    /// The equipment selector of the top bar: the current setup, the recent ones, the details.
    fn instrument_menu(&mut self, ui: &mut egui::Ui) {
        let Some(doc) = &self.document else { return };
        let label = setup_label(&doc.setup);
        let button = egui::Button::new((ICON_PHOTO_CAMERA.rich_text().color(ACCENT), RichText::new(label).color(TEXT))).corner_radius(egui::CornerRadius::same(7));
        let mut picked = None;
        let mut details = false;
        let current = doc.setup.clone();
        let recent = self.prefs.recent_setups.clone();
        egui::containers::menu::MenuButton::from_button(button).ui(ui, |ui| {
            ui.set_min_width(240.0);
            ui.label(RichText::new("STRUMENTI RECENTI").small().strong().color(TEXT_WEAK));
            if recent.is_empty() {
                ui.label(RichText::new("Nessuno: scegli telescopio e camera nei dettagli").small().color(TEXT_WEAK));
            }
            for setup in recent {
                if ui.add(egui::Button::selectable(setup == current, setup_label(&setup))).clicked() {
                    picked = Some(setup);
                }
            }
            ui.separator();
            if ui.button((ICON_TUNE.rich_text(), "Dettagli strumento…")).on_hover_text("Oggetto, telescopio, camera, Barlow, filtro, log di ripresa e diagnosi").clicked() {
                details = true;
            }
        })
        .0
        .on_hover_text("Lo strumento dell'immagine: scegline uno recente o apri i dettagli");
        if let Some(setup) = picked
            && let Some(doc) = &mut self.document
        {
            doc.setup = setup;
            self.instrument_action(InstrumentAction::SetupChanged);
        }
        if details {
            self.instrument_open = true;
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
            egui::Window::new(RichText::new("Strumento").strong())
                .open(&mut open)
                .default_pos(egui::pos2(200.0, 56.0))
                .default_width(440.0)
                .default_height(640.0)
                .resizable(true)
                .collapsible(false)
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
        let Some(doc) = &mut self.document else { return };
        match std::fs::read_to_string(path) {
            Ok(text) => {
                doc.capture.add_log(path, &text);
                doc.capture.measure(&doc.original, &doc.analysis);
                doc.setup = Setup::from_capture(&doc.capture.info, &doc.setup);
                self.status = format!("letto il log {}", file_name(path));
                self.equipment_changed();
            }
            Err(e) => self.status = format!("impossibile leggere {}: {e}", file_name(path)),
        }
    }

    /// The optics follow the equipment, and the structural settings the optics.
    fn equipment_changed(&mut self) {
        let Some(doc) = &mut self.document else { return };
        doc.analysis.optics = doc.setup.optics();
        if self.prefs.auto_structure {
            self.pipeline = restructure(doc.kind, &doc.analysis, &self.pipeline);
        }
    }

    /// `p` with the kind of wavelet scales the user prefers (generic transform settings).
    fn with_scales(&self, mut p: Pipeline) -> Pipeline {
        if self.prefs.wavelet_scales != ScaleMode::Free {
            p.wavelet.params.scale_mode = ScaleMode::Dyadic;
        }
        p
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
            Stage::Color => p.color.params = start.color.params,
            Stage::Mineral => p.mineral.params = start.mineral.params,
            Stage::Rings | Stage::Moons => self.apply_auto_module(stage),
            Stage::Layers => {}
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
            Stage::Mineral => p.mineral = Module::on(Default::default()),
            // The automatic settings keep the area touched up by hand.
            Stage::Dering => p.dering = Module::on(DeringParams { strength: 1.0, strokes: p.dering.params.strokes.clone(), width: p.dering.params.width, ..auto.dering.params }),
            Stage::Rings | Stage::Moons => {
                // Start the dedicated layer over with its default settings.
                let role = if stage == Stage::Rings { LayerRole::Rings } else { LayerRole::Moons };
                p.layers.retain(|l| l.role != Some(role));
                add_role_layer(p, role);
            }
            Stage::Layers => {}
        }
    }

    /// No automatic amount applied: the fields the automatic switch sets are at the starting point.
    fn is_untouched(&self) -> bool {
        let Some(doc) = &self.document else { return true };
        let (start, p) = (self.starting_point(doc.kind, &doc.analysis), &self.pipeline);
        (&p.align, &p.deconv, &p.denoise, &p.wavelet, &p.dering, &p.bxt.params) == (&start.align, &start.deconv, &start.denoise, &start.wavelet, &start.dering, &start.bxt.params)
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
        let Some(doc) = &mut self.document else { return };
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
                Event::Loaded { path, image, display, analysis, elapsed, mosaic, keep_settings } => {
                    self.status = format!("{} in {}", if mosaic.is_some() { "mosaico pronto" } else { "aperta" }, format_duration(elapsed));
                    let title = if mosaic.is_some() { format!("Mosaico da {}", file_name(&path)) } else { file_name(&path) };
                    ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!("{title} — Planetary Studio")));
                    let viewer = Viewer::new(ctx, Arc::new(display), image.width(), image.height());
                    // An updated mosaic keeps its processing, its type and its export name.
                    let kept = self.document.as_ref().filter(|d| keep_settings && d.mosaic.is_some()).map(|d| (d.kind, d.export_name.clone()));
                    self.selected_layer = None;
                    (self.layer_tool, self.show_mask, self.picking, self.cropping) = (LayerTool::None, false, None, false);
                    (self.psf_note, self.deconv_compare) = (None, None);
                    self.smart_legend = None;
                    // What the name and any log beside the image say, and the equipment: the log's,
                    // else the one used last.
                    let mut analysis = analysis;
                    let mut capture = CaptureState::read(&path);
                    // No subject named: what the pixels suggest (Saturn, Mars, Jupiter).
                    capture.body = capture.body.or(analysis.planet);
                    capture.measure(&image, &analysis);
                    let setup = Setup::from_capture(&capture.info, &self.prefs.setup);
                    analysis.optics = setup.optics();
                    if kept.is_none() {
                        // A fresh pipeline: layers belonged to the previous image. Structural
                        // settings follow the analysis (unless turned off); amounts only if asked to.
                        let (strength, scales) = (self.prefs.auto_strength, self.prefs.wavelet_scales);
                        self.pipeline = if self.prefs.auto_settings { auto_pipeline_with(analysis.kind, &analysis, strength, scales) } else { self.starting_point(analysis.kind, &analysis) };
                        self.history = History::new(&self.pipeline);
                    }
                    let image = Arc::new(image);
                    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("immagine").to_string();
                    let (kind, export_name) = kept.unwrap_or_else(|| (analysis.kind, if mosaic.is_some() { format!("{stem}_mosaico") } else { stem }));
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
                        mosaic: mosaic.map(|m| *m),
                        capture,
                        setup,
                    });
                    self.cropping = false;
                    self.layer_tool = LayerTool::None;
                    self.last_output = None;
                    self.full_output = None;
                    self.mask_thumbs.clear();
                    self.histogram = None;
                    self.timings.clear();
                    self.warnings.clear();
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
                    self.status = format!("esportata {} in {}", file_name(&path), format_duration(elapsed));
                    self.last_export = Some(path);
                }
                Event::Failed(message) => self.status = message,
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
            self.histogram = Some(out.histogram.clone());
            // A reduced preview is only for looking at: pixel values, the black/white picker
            // and export keep using the last full-resolution result.
            if !out.proxy {
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
        let pipeline = self.prefs.effective(&self.pipeline, doc.kind, doc.source.is_mono());
        // Quick preview while a control in the panel is being dragged and the settings change.
        let dragging = self.prefs.fast_preview && roi.is_none() && !self.canvas_hovered && ctx.input(|i| i.pointer.any_down());
        let changing = self.last_request.as_ref().is_none_or(|k| k.proxy > 1 || k.pipeline != pipeline);
        let proxy = if dragging && changing { proxy_factor(&doc.source) } else { 1 };
        let key = RequestKey {
            source_gen: doc.source_gen,
            pipeline,
            roi,
            luminance_gen: luminance.as_ref().map(|l| l.1),
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
        self.status = "Versione A fissata: continua a modificare (B), poi scegli quale tenere".into();
    }

    /// End the comparison keeping A (its settings come back) or B (A is forgotten).
    fn end_comparison(&mut self, ctx: &egui::Context, keep_a: bool) {
        let Some(snap) = self.snapshot.take() else { return };
        if keep_a {
            self.pipeline = snap.pipeline;
            self.status = "Tenuta la versione A".into();
        } else {
            self.status = "Tenuta la versione B".into();
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
        if self.settings.open {
            return;
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
        if ctx.input_mut(|i| i.consume_shortcut(&REDO)) {
            self.history.redo(&mut self.pipeline);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&UNDO)) {
            self.history.undo(&mut self.pipeline);
        }
        // Several images dropped together: a mosaic of them.
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        // A capture log dropped on an open image is read for it.
        let is_log = |p: &PathBuf| p.extension().and_then(|e| e.to_str()).is_some_and(|e| ["txt", "log", "as3", "as4"].contains(&e.to_ascii_lowercase().as_str()));
        let (logs, dropped): (Vec<PathBuf>, Vec<PathBuf>) = dropped.into_iter().partition(|p| is_log(p) && self.document.is_some());
        for log in logs {
            self.read_log(&log);
        }
        match dropped.len() {
            0 => {}
            1 => self.load(ctx, dropped.into_iter().next().expect("one")),
            _ => self.mosaic_editor.start(ctx, dropped),
        }
        if ctx.egui_wants_keyboard_input() {
            return;
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
                    if self.document.as_ref().is_some_and(|d| d.kind != subject.kind) {
                        self.apply_kind(subject.kind);
                    }
                    if let Some(doc) = &mut self.document {
                        doc.capture.body = subject.body;
                        doc.capture.measure(&doc.original, &doc.analysis);
                    }
                }
                PanelAction::RemoveAuto => {
                    if let Some(doc) = &self.document {
                        let start = self.starting_point(doc.kind, &doc.analysis);
                        let p = &mut self.pipeline;
                        let strokes = std::mem::take(&mut p.dering.params.strokes);
                        (p.align, p.deconv, p.denoise, p.wavelet, p.dering) = (start.align, start.deconv, start.denoise, start.wavelet, start.dering);
                        p.dering.params.strokes = strokes;
                        p.bxt.params = start.bxt.params;
                    }
                }
                PanelAction::AutoAll => self.apply_auto_all(),
                PanelAction::AutoModule(stage) => self.apply_auto_module(stage),
                PanelAction::ResetModule(stage) => self.reset_module(stage),
                PanelAction::ClearAuto => {
                    self.clear_auto();
                    self.prefs.auto_structure = false;
                }
                PanelAction::Export => self.quick_export(ctx),
                PanelAction::ExportAs => self.export_dialog(ctx),
                PanelAction::SavePreset => self.save_preset(),
                PanelAction::LoadPreset => self.load_preset(),
                PanelAction::SetAutoStructure(on) => {
                    self.prefs.auto_structure = on;
                    if let Some(doc) = &self.document {
                        self.pipeline = if on { restructure(doc.kind, &doc.analysis, &self.pipeline) } else { ps_core::preset::generic_structure(&self.pipeline) };
                    }
                }
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
            self.status = "Il punto scelto è fuori dalla regione elaborata (ROI)".into();
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

    /// The star or small moon near (`x`, `y`) becomes the deconvolution's PSF.
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
                let shape = if m.roundness > 0.85 { "rotondo" } else if m.roundness > 0.65 { "un po' allungato" } else { "allungato: forse mosso o non puntiforme" };
                let faint = if m.brightness < 0.05 { " Molto debole rispetto al resto dell'immagine: la forma è incerta." } else { "" };
                self.psf_note = Some(format!(
                    "Misurata su un oggetto in ({:.0}, {:.0}): largo {:.1} px, {shape}.{faint} Se è una luna, la sua dimensione si somma alla sfocatura.",
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
            ("Gaussiana".to_string(), Some(shape(0))),
            ("Moffat".to_string(), Some(shape(1))),
            ("Airy".to_string(), Some(shape(2))),
            ("Originale".to_string(), None),
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
                        (p.dering.enabled, p.tone.enabled, p.curves.enabled, p.color.enabled, p.mineral.enabled) = (false, false, false, false, false);
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
        egui::Window::new("Confronto deconvoluzione").collapsible(false).resizable(true).default_size([960.0, 700.0]).open(&mut open).show(ctx, |ui| {
            panels::hint(ui, &format!("Zona visibile dell'immagine, sfocatura {:.1} px. Sopra: le tre forme di PSF con Richardson–Lucy. Sotto: l'originale e i due algoritmi con la forma attuale.", compare.fwhm));
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
                                    if tile.params.is_some() && ui.small_button("Usa questa").clicked() {
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

    /// One bar: application menus on the left, view controls (found in no menu) on the right.
    fn top_bar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.label(ICON_PLANET.rich_text().size(20.0).color(ACCENT));
            ui.add_space(4.0);
            ui.menu_button("File", |ui| {
                if ui.add(egui::Button::new((ICON_FOLDER_OPEN.rich_text(), "Apri…")).shortcut_text(ctx.format_shortcut(&OPEN))).clicked() {
                    self.open_dialog(ctx);
                }
                let quick = egui::Button::new((ICON_IOS_SHARE.rich_text(), "Esporta")).shortcut_text(ctx.format_shortcut(&QUICK_EXPORT));
                if ui.add_enabled(self.document.is_some(), quick).on_hover_text("Accanto all'originale, nel formato dell'ultima volta").clicked() {
                    self.quick_export(ctx);
                }
                let export = egui::Button::new((ICON_SAVE_AS.rich_text(), "Esporta come…")).shortcut_text(ctx.format_shortcut(&EXPORT));
                if ui.add_enabled(self.document.is_some(), export).clicked() {
                    self.export_dialog(ctx);
                }
                ui.separator();
                if ui.button((ICON_GRID_VIEW.rich_text(), "Nuovo mosaico…")).on_hover_text("Unisci più immagini sovrapposte dello stesso soggetto (Luna, Sole…) in una sola").clicked() {
                    self.new_mosaic_dialog(ctx);
                }
                let mosaic = self.document.as_ref().and_then(|d| d.mosaic.clone());
                if ui.add_enabled(mosaic.is_some(), egui::Button::new((ICON_EDIT.rich_text(), "Modifica mosaico…"))).on_hover_text("Cambia le immagini o le posizioni del mosaico aperto: le regolazioni restano").clicked()
                    && let Some(m) = mosaic
                {
                    self.mosaic_editor.edit(ctx, m, true);
                }
                ui.separator();
                if ui.button((ICON_UPLOAD_FILE.rich_text(), "Carica impostazioni…")).clicked() {
                    self.load_preset();
                }
                if ui.button((ICON_SAVE_AS.rich_text(), "Salva impostazioni…")).clicked() {
                    self.save_preset();
                }
            });
            ui.menu_button("Modifica", |ui| {
                let undo = egui::Button::new((ICON_UNDO.rich_text(), "Annulla")).shortcut_text(ctx.format_shortcut(&UNDO));
                if ui.add_enabled(self.history.can_undo(), undo).clicked() {
                    self.history.undo(&mut self.pipeline);
                }
                let redo = egui::Button::new((ICON_REDO.rich_text(), "Ripeti")).shortcut_text(ctx.format_shortcut(&REDO));
                if ui.add_enabled(self.history.can_redo(), redo).clicked() {
                    self.history.redo(&mut self.pipeline);
                }
                ui.separator();
                let has_doc = self.document.is_some();
                if ui.add_enabled(has_doc && !self.cropping, egui::Button::new((ICON_CROP.rich_text(), "Ritaglia")).shortcut_text("C")).clicked() {
                    self.start_crop();
                }
                if ui.add_enabled(has_doc, egui::Button::new((ICON_ROTATE_RIGHT.rich_text(), "Ruota di 90° in senso orario"))).clicked() {
                    self.rotate(ctx, Geometry::rotate_clockwise);
                }
                if ui.add_enabled(has_doc, egui::Button::new((ICON_ROTATE_LEFT.rich_text(), "Ruota di 90° in senso antiorario"))).clicked() {
                    self.rotate(ctx, Geometry::rotate_counterclockwise);
                }
                if ui.add_enabled(has_doc, egui::Button::new((ICON_SWAP_HORIZ.rich_text(), "Specchia orizzontalmente"))).clicked() {
                    self.rotate(ctx, |g| g.flip_horizontal = !g.flip_horizontal);
                }
                if ui.add_enabled(has_doc, egui::Button::new((ICON_SWAP_VERT.rich_text(), "Specchia verticalmente"))).clicked() {
                    self.rotate(ctx, |g| g.flip_vertical = !g.flip_vertical);
                }
                let cropped = self.document.as_ref().is_some_and(|d| d.source.width() != d.oriented.width() || d.source.height() != d.oriented.height());
                if ui.add_enabled(cropped, egui::Button::new((ICON_CROP_FREE.rich_text(), "Rimuovi ritaglio"))).clicked() {
                    self.remove_crop(ctx);
                }
                ui.separator();
                let auto = egui::Button::new((ICON_AUTO_FIX_HIGH.rich_text(), "Impostazioni automatiche"));
                if ui.add_enabled(self.document.is_some(), auto).clicked() {
                    self.apply_auto_all();
                }
            });
            ui.menu_button("Aiuto", |ui| {
                ui.label(RichText::new("Scorciatoie").strong());
                let f = |k: &KeyboardShortcut| ctx.format_shortcut(k);
                let alt = format!("{} (tieni)", theme::alt_key());
                let open_export = format!("{} / {}", f(&OPEN), f(&EXPORT));
                let quick_export = f(&QUICK_EXPORT);
                let fit_actual = format!("F / {}", f(&ACTUAL_SIZE));
                let undo_redo = format!("{} / {}", f(&UNDO), f(&REDO));
                let settings = f(&SETTINGS);
                for (keys, what) in [
                    (open_export.as_str(), "Apri / esporta come…"),
                    (quick_export.as_str(), "Esporta accanto all'originale"),
                    ("1 … 6", "Aree del pannello"),
                    (undo_redo.as_str(), "Annulla / ripeti"),
                    (settings.as_str(), "Impostazioni"),
                    ("Spazio (tieni)", "Mostra l'originale"),
                    ("S", "Vista divisa prima/dopo"),
                    ("R", "Anteprima solo su una regione (ROI)"),
                    ("C", "Ritaglia (Invio applica, Esc annulla)"),
                    ("[ / ]", "Pennello più piccolo / più grande"),
                    (alt.as_str(), "Il pennello toglie invece di aggiungere"),
                    (fit_actual.as_str(), "Adatta / 100 %"),
                    ("Doppio clic su un titolo", "Escludi o riattiva il modulo"),
                    ("Rotella, pinch", "Zoom"),
                    ("Tasto destro, trascina", "Sposta la vista"),
                    ("Doppio clic su un cursore", "Valore predefinito"),
                ] {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(keys).monospace().color(ACCENT));
                        ui.label(what);
                    });
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
                if icon_button(ui, ICON_SETTINGS, &format!("Impostazioni ({})", ctx.format_shortcut(&SETTINGS))).clicked() {
                    self.settings.open = true;
                }
                separator(ui);
                if self.document.is_some() {
                    if tool_button(ui, ICON_WARNING_AMBER, "Bruciati", self.show_clipping)
                        .on_hover_text("Mostra i pixel bruciati: in rosso quelli diventati bianco pieno, in blu il dettaglio schiacciato a nero")
                        .clicked()
                    {
                        self.show_clipping = !self.show_clipping;
                    }
                    let comparing = self.snapshot.is_some();
                    let can_compare = self.last_output.is_some();
                    let tip = if comparing {
                        "Chiudi il confronto tenendo la versione B (quella che stai modificando)"
                    } else {
                        "Fissa il risultato attuale come versione A, continua a modificare (B) e confrontale: poi tieni quella che preferisci"
                    };
                    if ui.add_enabled_ui(can_compare, |ui| tool_button(ui, ICON_COMPARE, "Confronta", comparing)).inner.on_hover_text(tip).clicked() {
                        if comparing {
                            self.end_comparison(ctx, false);
                        } else {
                            self.take_snapshot(ctx);
                        }
                    }
                    if tool_button(ui, ICON_CROP, "Ritaglia", self.cropping).on_hover_text("Ritaglia e ruota (C)").clicked() && !self.cropping {
                        self.start_crop();
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
                ui.label(RichText::new("Elaborazione dell'immagine intera in background…").small().color(TEXT_WEAK));
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(&self.backend).small().color(TEXT_WEAK));
                if let Some(out) = &self.last_output {
                    ui.separator();
                    ui.label(RichText::new(format!("elaborazione {}", format_duration(out.elapsed))).small().color(TEXT_WEAK));
                }
                if let Some(n) = self.clipped.filter(|_| self.clipping_shown()) {
                    ui.separator();
                    let colour = if n > 0 { ui.visuals().warn_fg_color } else { TEXT_WEAK };
                    ui.label(RichText::new(if n == 1 { "1 pixel bruciato".to_string() } else { format!("{n} pixel bruciati") }).small().color(colour));
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
        let processed = self.last_output.as_ref().filter(|o| o.source_gen == doc.source_gen).and_then(|o| match o.roi {
            None => Some((&o.image, x, y)),
            Some(r) if (r.x..r.x + r.w).contains(&x) && (r.y..r.y + r.h).contains(&y) => Some((&o.image, x - r.x, y - r.y)),
            _ => None,
        });
        let (img, px, py) = processed.unwrap_or((&doc.source, x, y));
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
            Stage::Color => p.color.enabled && !p.color.params.is_neutral(),
            Stage::Mineral => p.mineral.enabled && (!p.mineral.params.is_neutral() || self.document.as_ref().is_some_and(|d| d.luminance.is_some())),
            Stage::Layers => p.layers.iter().any(|l| l.enabled && l.overrides_any()),
            Stage::Rings => role(LayerRole::Rings),
            Stage::Moons => role(LayerRole::Moons),
        }
    }

    fn side_panel(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        egui::Panel::bottom("histogram").frame(egui::Frame::NONE.inner_margin(egui::Margin::symmetric(0, 10))).show(ui, |ui| {
            ui.label(RichText::new("ISTOGRAMMA").small().strong().color(TEXT_WEAK));
            ui.add_space(4.0);
            self.histogram.clone().unwrap_or_default().show(ui);
        });

        let mut actions = Vec::new();
        let customised = self.is_customised();
        let untouched = self.is_untouched();
        let running = self.worker.progress();
        let scales_before = self.pipeline.wavelet.params.scale_mode;
        let Some((kind, mono)) = self.document.as_ref().map(|d| (d.kind, d.source.is_mono())) else { return };
        let layout = self.prefs.panel_layout(kind, mono);
        let active: Vec<(prefs_group::Group, usize)> = layout.iter().map(|(g, stages)| (*g, stages.iter().filter(|s| self.stage_active(**s)).count())).collect();
        let adapt = self.prefs.adapt_to_image;
        let one_open = self.prefs.one_group_open;
        let bxt_configured = self.prefs.bxt_executable().is_some();
        let tone_histogram = self.last_output.as_ref().map(|o| o.tone_histogram.clone());
        let curves_histogram = self.last_output.as_ref().map(|o| o.curves_histogram.clone());
        let deconv_report = self.last_output.as_ref().and_then(|o| o.deconv);
        let active_stages: std::collections::HashSet<Stage> = layout.iter().flat_map(|(_, s)| s.iter().copied()).filter(|s| self.stage_active(*s)).collect();
        let gpu = self.prefs.use_gpu && matches!(ps_core::gpu::status(), Some(Some(_)));
        let last_export = self.last_export.clone();
        let export_stem = self.export_stem();
        let Self { document, pipeline, prefs, psf_preview, psf_note, timings, warnings, selected_layer, show_mask, layer_tool, brush, mask_thumbs, side_width, show_ringing, smart_view, smart_legend, .. } = self;
        let Some(doc) = document else { return };
        let why = explain(doc.kind, &doc.analysis, pipeline.wavelet.params.scale_mode);
        let auto_deconv = ps_core::preset::auto_deconv(doc.kind, &doc.analysis, prefs.auto_strength);
        let auto_smart = ps_core::preset::smart_structure(doc.kind, &doc.analysis);
        // What the wavelet controls' resets go back to: the image's starting point.
        let scales = pipeline.wavelet.params.scale_mode;
        let wavelet_start = if prefs.auto_structure {
            ps_core::preset::wavelet_structure(doc.kind, &doc.analysis, scales)
        } else {
            ps_core::wavelet::WaveletParams { scale_mode: scales, ..Default::default() }
        };
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
            auto_structure: prefs.auto_structure,
        };
        panels::start_row(ui, start_view, &mut prefs.auto_strength, &mut actions);
        ui.add_space(theme::space::L);
        // The module areas, then the export tab.
        let mut tabs = active.clone();
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
            if area == Some(prefs_group::Group::Export) {
                let mut name = panels::ExportName { base: &mut doc.export_name, keywords: &mut prefs.export_keywords, suffix: &mut prefs.export_suffix, stem: &export_stem };
                panels::export_body(ui, &mut prefs.export_format, &mut name, &doc.path, mono, last_export.as_deref(), &mut actions);
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
                        let is_active = active_stages.contains(&stage);
                        let layer_ui = |selected, show_mask, tool, brush, folded| LayerUi { selected, show_mask, tool, brush, collapsed: folded, kind: doc.kind, adapt, mono, wavelet_start: &wavelet_start, smart_start: &auto_smart, thumbs: &*mask_thumbs };
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
                                panels::denoise_body(ui, &mut pipeline.denoise.params, mono, Some(&why.denoise))
                            }),
                            Stage::Wavelet => panels::module_card(ui, stage, Some(&mut pipeline.wavelet.enabled), is_active, &mut collapsed, status, |ui| {
                                let view = panels::WaveletView { start: &wavelet_start, why: Some(&why.wavelet), cutoff: detail_cutoff };
                                panels::wavelet_body(ui, &mut pipeline.wavelet.params, view)
                            }),
                            Stage::Smart => panels::module_card(ui, stage, Some(&mut pipeline.smart.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::smart_body(ui, &mut pipeline.smart.params, &auto_smart, &mut *smart_view, *smart_legend, &upstream)
                            }),
                            Stage::Contrast => panels::module_card(ui, stage, Some(&mut pipeline.contrast.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::contrast_body(ui, &mut pipeline.contrast.params)
                            }),
                            Stage::Dering => panels::module_card(ui, stage, Some(&mut pipeline.dering.enabled), is_active, &mut collapsed, status, |ui| {
                                let state = layer_ui(&mut *selected_layer, &mut *show_mask, &mut *layer_tool, &mut *brush, &mut prefs.collapsed_sections);
                                panels::dering_body(ui, &mut pipeline.dering.params, &why.dering, &mut *show_ringing, state)
                            }),
                            Stage::Tone => panels::module_card(ui, stage, Some(&mut pipeline.tone.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::tone_body(ui, &mut pipeline.tone.params, ToneParts::Levels, None, tone_histogram.as_ref(), &mut prefs.histogram_mode, mono, &mut actions)
                            }),
                            Stage::Curves => panels::module_card(ui, stage, Some(&mut pipeline.curves.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::tone_body(ui, &mut pipeline.curves.params, ToneParts::Curve, None, curves_histogram.as_ref(), &mut prefs.histogram_mode, mono, &mut actions)
                            }),
                            Stage::Color => panels::module_card(ui, stage, Some(&mut pipeline.color.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::color_body(ui, &mut pipeline.color.params, mono)
                            }),
                            // Listed among the masks now (see `Preferences::upgraded`).
                            Stage::Rings | Stage::Moons => panels::CardButtons::default(),
                            // The list of masks, then the selected one's editor: no card to fold.
                            Stage::Layers => {
                                let state = layer_ui(&mut *selected_layer, &mut *show_mask, &mut *layer_tool, &mut *brush, &mut prefs.collapsed_sections);
                                panels::masks_area(ui, &mut pipeline.layers, state, &mut actions);
                                panels::CardButtons::default()
                            }
                            Stage::Mineral => panels::module_card(ui, stage, Some(&mut pipeline.mineral.enabled), is_active, &mut collapsed, status, |ui| {
                                panels::mineral_body(ui, &mut pipeline.mineral.params, mono, luminance_name.as_deref(), &mut actions)
                            }),
                        };
                        if buttons.auto {
                            actions.push(PanelAction::AutoModule(stage));
                        }
                        if buttons.reset {
                            actions.push(PanelAction::ResetModule(stage));
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
            ui.add_space(12.0);
        });
        if self.pipeline.wavelet.params.scale_mode != scales_before {
            self.switch_scales(scales_before);
        }
        self.apply_actions(ctx, actions);
    }

    fn empty_state(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space((ui.available_height() * 0.5 - 120.0).max(20.0));
            ui.label(ICON_PLANET.rich_text().size(72.0).color(theme::ACCENT_DIM));
            ui.add_space(10.0);
            ui.label(RichText::new("Apri un'immagine planetaria, lunare o solare").size(19.0).color(TEXT));
            ui.add_space(2.0);
            ui.label(RichText::new("Il tipo viene riconosciuto e le impostazioni scelte automaticamente").color(TEXT_WEAK));
            ui.add_space(16.0);
            let black = egui::Color32::BLACK;
            let open = egui::Button::new((ICON_FOLDER_OPEN.rich_text().color(black), RichText::new("Apri immagine…").color(black).strong()))
                .fill(ACCENT)
                .min_size(egui::vec2(180.0, 36.0))
                .corner_radius(egui::CornerRadius::same(8));
            if ui.add(open).clicked() {
                self.open_dialog(ctx);
            }
            ui.add_space(8.0);
            ui.label(RichText::new("oppure trascinala qui · TIFF, FITS, PNG, JPEG").small().color(TEXT_WEAK));
            ui.add_space(10.0);
            if ui.add(egui::Button::new((ICON_GRID_VIEW.rich_text(), "Unisci più immagini in un mosaico…")).frame(false)).on_hover_text("Oppure trascina qui più immagini insieme").clicked() {
                self.new_mosaic_dialog(ctx);
            }
            if !self.status.is_empty() {
                ui.add_space(12.0);
                ui.label(RichText::new(&self.status).small().color(TEXT_WEAK));
            }
        });
    }
}

impl eframe::App for PlanetaryStudio {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = &ui.ctx().clone();
        self.handle_events(ctx);
        self.handle_shortcuts(ctx);
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
                // Floating over the image: how it is shown (top) and the zoom (bottom right).
                let canvas = ui.max_rect();
                // A bar of its own size, with its `pivot` corner at `at`, above the image.
                // Not over the mosaic window.
                let covered = self.mosaic_editor.open || self.deconv_compare.is_some();
                let float = |ui: &mut egui::Ui, id: &str, at: egui::Pos2, pivot: egui::Align2, add: &mut dyn FnMut(&mut egui::Ui)| {
                    if covered {
                        return;
                    }
                    egui::Area::new(egui::Id::unique(id)).order(egui::Order::Foreground).fixed_pos(at).pivot(pivot).show(ui.ctx(), |ui| {
                        egui::Frame::new().fill(egui::Color32::from_black_alpha(190)).corner_radius(10).inner_margin(egui::Margin::symmetric(8, 5)).show(ui, |ui| {
                            ui.horizontal(|ui| add(ui));
                        });
                    });
                };
                let mut preview = self.preview;
                float(ui, "view-bar", canvas.center_top() + Vec2::new(0.0, 8.0), egui::Align2::CENTER_TOP, &mut |ui| {
                    segmented(ui, &mut preview, &[(PreviewMode::Full, Some(ICON_IMAGE), "Intera"), (PreviewMode::Roi, Some(ICON_CROP_FREE), "ROI")]);
                    if roi_mode {
                        segmented(ui, &mut self.tool, &[(Tool::Pan, Some(ICON_PAN_TOOL), ""), (Tool::Roi, Some(ICON_HIGHLIGHT_ALT), "")]);
                    }
                    ui.add(egui::Separator::default().vertical().spacing(10.0));
                    // While comparing, the versions take the place of before/after.
                    let options = if self.snapshot.is_some() {
                        vec![(Compare::OnlyA, None, "A"), (Compare::Ab, Some(ICON_VERTICAL_SPLIT), "A | B"), (Compare::After, None, "B")]
                    } else {
                        vec![(Compare::After, None, "Dopo"), (Compare::Before, None, "Prima"), (Compare::Split, Some(ICON_VERTICAL_SPLIT), "Divisa")]
                    };
                    segmented(ui, &mut self.compare, &options);
                });
                if self.snapshot.is_some() {
                    float(ui, "compare-bar", canvas.center_bottom() - Vec2::new(0.0, 12.0), egui::Align2::CENTER_BOTTOM, &mut |ui| {
                        ui.label(ICON_COMPARE.rich_text().color(ACCENT));
                        ui.label(RichText::new("Confronto").strong().color(TEXT));
                        ui.add(egui::Separator::default().vertical().spacing(10.0));
                        if ui.button("Tieni A").on_hover_text("Torna alle impostazioni della versione A e chiudi il confronto").clicked() {
                            end_comparison = Some(true);
                        }
                        if ui.button("Tieni B").on_hover_text("Tieni le modifiche attuali e dimentica A").clicked() {
                            end_comparison = Some(false);
                        }
                        if ui.add(egui::Button::new((ICON_RESTART_ALT.rich_text(), "A = ora")).frame(false)).on_hover_text("Fissa il risultato attuale come nuova versione A").clicked() {
                            retake = true;
                        }
                    });
                }
                if preview != self.preview {
                    new_preview = Some(preview);
                }
                let viewer = &mut doc.viewer;
                float(ui, "zoom-bar", canvas.right_bottom() - Vec2::new(12.0, 12.0), egui::Align2::RIGHT_BOTTOM, &mut |ui| {
                    if icon_button(ui, ICON_ZOOM_OUT, "Riduci").clicked() {
                        viewer.set_zoom(viewer.zoom() / 1.25);
                    }
                    let zoom_label = RichText::new(format!("{:.0}%", viewer.zoom() * 100.0)).monospace().color(TEXT);
                    if ui.add(egui::Button::new(zoom_label).frame(false)).on_hover_text(format!("Dimensione reale ({})", ctx.format_shortcut(&ACTUAL_SIZE))).clicked() {
                        viewer.set_zoom(1.0);
                    }
                    if icon_button(ui, ICON_ZOOM_IN, "Ingrandisci").clicked() {
                        viewer.set_zoom(viewer.zoom() * 1.25);
                    }
                    if tool_button(ui, ICON_FIT_SCREEN, "Adatta", false).on_hover_text("Adatta alla finestra (F)").clicked() {
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

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The app with `image` open, as if loaded from a file.
    fn opened(size: [f32; 2], image: Image) -> egui_kittest::Harness<'static, PlanetaryStudio> {
        // Real frame times, so double clicks are double clicks.
        let mut harness = egui_kittest::Harness::builder().with_size(size).with_step_dt(1.0 / 60.0).build_eframe(|cc| PlanetaryStudio::new(cc, None));
        let analysis = Box::new(analyze(&image));
        let display = worker::display_image(&image, 4096);
        harness.state().events_tx.send(Event::Loaded { path: "test.tif".into(), image, display, analysis, elapsed: Duration::ZERO, mosaic: None, keep_settings: false }).unwrap();
        harness.run_steps(3);
        harness
    }

    /// Every special character used in labels must exist in the bundled fonts (otherwise it
    /// shows up as an empty box).
    #[test]
    fn ui_glyphs_exist() {
        let ctx = egui::Context::default();
        theme::apply(&ctx);
        let mut output = ctx.run_ui(Default::default(), |_| {});
        let missing: String = "°·×≈βσλ–…″⌘⇧⌥⌃⚠".chars().filter(|&c| !ctx.fonts_mut(|f| f.has_glyph(&egui::FontId::proportional(13.0), c))).collect();
        // No renderer here to upload the font atlas to.
        output.textures_delta.clear();
        assert!(missing.is_empty(), "missing glyphs: {missing}");
    }

    /// Regression: while a slider is dragged and results stream in, nothing may move — not
    /// the image, not the panel, not the cards below the one being edited.
    #[test]
    fn layout_is_still_while_processing() {
        let (w, h) = (500, 400);
        let plane: Vec<f32> = (0..w * h).map(|i| if ((i % w) as f32 - 250.0).hypot((i / w) as f32 - 200.0) < 120.0 { 0.5 + 0.2 * ((i / w) as f32 * 0.1).sin() } else { 0.02 }).collect();
        let image = Image::new(w, h, vec![plane.clone(), plane.clone(), plane], 16);
        let mut harness = opened([1400.0, 900.0], image);
        harness.run_steps(2);
        harness.state_mut().pipeline.deconv.enabled = true;
        let mut samples = Vec::new();
        for k in 0..240 {
            // A drag: the value changes every frame, so processing is always under way.
            harness.state_mut().pipeline.deconv.params.iterations = 10 + (k % 30) as u32;
            harness.step();
            std::thread::sleep(Duration::from_millis(4));
            let app = harness.state();
            let canvas = app.document.as_ref().unwrap().viewer.canvas();
            let below = harness.ctx.data(|d| d.get_temp::<f32>(egui::Id::unique(("card-top", "Wavelet"))));
            samples.push((canvas, app.side_width, below, app.worker.is_busy()));
        }
        assert!(samples.iter().any(|s| s.3), "processing never ran");
        let first = samples[20];
        for (i, s) in samples.iter().enumerate().skip(20) {
            assert_eq!(s.0, first.0, "image area moved at frame {i}");
            assert_eq!(s.1, first.1, "panel width changed at frame {i}");
            assert_eq!(s.2, first.2, "card below the deconvolution moved at frame {i}");
        }
    }

    /// A module's reset button and "Rimuovi" (drop every automatic choice).
    #[test]
    fn reset_and_remove_automatic_settings() {
        let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
        let mut harness = opened([1200.0, 800.0], image);
        let ctx = harness.ctx.clone();
        let app = harness.state_mut();
        let start = app.pipeline.clone();

        // A module's reset brings its settings back but leaves its switch alone.
        app.pipeline.deconv.enabled = true;
        app.pipeline.deconv.params.iterations = 77;
        app.apply_actions(&ctx, vec![PanelAction::ResetModule(Stage::Deconv)]);
        assert!(app.pipeline.deconv.enabled);
        assert_eq!(app.pipeline.deconv.params, start.deconv.params);

        // "Rimuovi" drops the analysis' choices, layers survive.
        app.apply_auto_all();
        app.add_layer(LayerTemplate::Ellipse);
        app.apply_actions(&ctx, vec![PanelAction::ClearAuto]);
        assert_eq!(app.pipeline.layers.len(), 1);
        assert_eq!(Pipeline { layers: vec![], ..app.pipeline.clone() }, neutral_pipeline());
    }

    /// Adding the rings' mask selects it; "Valori consigliati" restores its settings in place,
    /// keeping its name, switch, position and painted strokes.
    #[test]
    fn rings_mask_restores_in_place() {
        let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
        let mut harness = opened([1200.0, 800.0], image);
        let ctx = harness.ctx.clone();
        let app = harness.state_mut();
        app.add_layer(LayerTemplate::Ellipse);
        app.apply_actions(&ctx, vec![PanelAction::CreateRole(LayerRole::Rings)]);
        let id = app.pipeline.layers[1].id;
        assert_eq!(app.selected_layer, Some(id), "the new mask is selected");
        app.pipeline.layers.swap(0, 1);
        let fresh = app.pipeline.layers[0].clone();
        let rings = &mut app.pipeline.layers[0];
        (rings.name, rings.enabled) = ("Mie".into(), false);
        rings.wavelet.enabled = false;
        rings.mask.strokes.push(BrushStroke { points: vec![[1.0, 1.0]], radius: 3.0, hardness: 0.5, erase: false });
        app.apply_actions(&ctx, vec![PanelAction::RestoreRole(id)]);
        let rings = &app.pipeline.layers[0];
        assert_eq!((rings.id, rings.name.as_str(), rings.enabled, rings.mask.strokes.len()), (id, "Mie", false, 1));
        assert_eq!(rings.wavelet, fresh.wavelet);
        assert_eq!(app.pipeline.layers.len(), 2);
    }

    #[test]
    fn layer_adjustments_one_at_a_time() {
        // Tall enough for the whole layer editor to be on screen, where the click lands.
        let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
        let mut harness = opened([1400.0, 2400.0], image);
        {
            let app = harness.state_mut();
            app.prefs.current_area = "Maschere".into();
            app.add_layer(LayerTemplate::Ellipse);
            app.selected_layer = app.pipeline.layers.first().map(|l| l.id);
        }
        harness.run_steps(2);
        assert!(harness.state().show_mask, "an open layer shows what it selects");
        use egui_kittest::kittest::Queryable;
        harness.get_by_label("Contrasto").click();
        harness.run_steps(2);
        assert!(harness.state().prefs.collapsed_sections.contains("layer-open:contrast"), "the clicked adjustment opens");
        harness.get_by_label("Rumore").click();
        harness.run_steps(2);
        let open = &harness.state().prefs.collapsed_sections;
        assert!(open.contains("layer-open:noise") && !open.contains("layer-open:contrast"), "the open one closes");
    }

    /// With the quick preview on, a drag that changes the settings renders a reduced image;
    /// releasing renders full size again, and only that one counts as the real result.
    #[test]
    fn quick_preview_while_dragging() {
        let (w, h) = (800, 600);
        let plane: Vec<f32> = (0..w * h).map(|i| if ((i % w) as f32 - 400.0).hypot((i / w) as f32 - 300.0) < 200.0 { 0.6 } else { 0.02 }).collect();
        let mut harness = opened([1400.0, 900.0], Image::new(w, h, vec![plane], 16));
        harness.state_mut().prefs.fast_preview = true;
        // Pointer held down away from the image (as on a slider), settings changing.
        let press = |pressed| egui::Event::PointerButton { pos: egui::pos2(700.0, 20.0), button: egui::PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
        harness.event(egui::Event::PointerMoved(egui::pos2(700.0, 20.0)));
        harness.event(press(true));
        harness.run_steps(1);
        harness.state_mut().pipeline.contrast.params.amount = 0.5;
        harness.run_steps(1);
        assert_eq!(harness.state().last_request.as_ref().unwrap().proxy, 2);
        harness.event(press(false));
        harness.run_steps(1);
        assert_eq!(harness.state().last_request.as_ref().unwrap().proxy, 1);
        for _ in 0..300 {
            harness.step();
            if harness.state().last_output.as_ref().is_some_and(|o| o.pipeline.contrast.params.amount == 0.5) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let out = harness.state().last_output.as_ref().expect("full result");
        assert!(!out.proxy && out.image.width() == w);
    }

    /// Cards start closed; in an area, opening one closes the one that was open. Brush strokes
    /// go to the deringing area when that is the mask being edited.
    #[test]
    fn one_card_open_per_area() {
        let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
        let mut harness = opened([1400.0, 2000.0], image);
        {
            let app = harness.state_mut();
            assert!(app.prefs.expanded.is_empty(), "cards start closed");
            app.prefs.one_group_open = true;
            app.prefs.current_area = "Nitidezza".into();
        }
        harness.run_steps(2);
        use egui_kittest::kittest::Queryable;
        harness.get_by_label_contains("Deconvoluzione").click();
        harness.run_steps(2);
        assert!(harness.state().prefs.expanded.contains(&Stage::Deconv));
        harness.get_by_label_contains("Wavelet").click();
        harness.run_steps(2);
        let expanded = &harness.state().prefs.expanded;
        assert!(expanded.contains(&Stage::Wavelet) && !expanded.contains(&Stage::Deconv), "{expanded:?}");

        let app = harness.state_mut();
        app.selected_layer = Some(RING_MASK);
        app.add_stroke(vec![[10.0, 10.0]], false);
        assert_eq!(app.pipeline.dering.params.strokes.len(), 1);
    }

    /// The brush and the red overlay last only while their controls are on screen, and never
    /// outlive the layer they paint on.
    #[test]
    fn mask_tools_do_not_linger() {
        let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
        let mut harness = opened([1400.0, 2000.0], image);
        {
            let app = harness.state_mut();
            app.prefs.current_area = "Maschere".into();
            app.add_layer(LayerTemplate::Brush);
        }
        harness.run_steps(3);
        assert_eq!(harness.state().layer_tool, LayerTool::Brush, "painting right after adding a brush layer");

        // Leaving the masks' area puts the brush away.
        harness.state_mut().prefs.current_area = "Nitidezza".into();
        harness.run_steps(2);
        let app = harness.state();
        assert!(app.layer_tool == LayerTool::None && !app.show_mask);

        // The layer disappearing (as an undo of its creation does) forgets it as the selected mask.
        harness.state_mut().prefs.current_area = "Maschere".into();
        harness.state_mut().layer_tool = LayerTool::Brush;
        harness.run_steps(2);
        harness.state_mut().pipeline.layers.clear();
        harness.run_steps(2);
        let app = harness.state();
        assert!(app.selected_layer.is_none() && app.layer_tool == LayerTool::None);
    }

    /// A drag with the brush on the image becomes one stroke on the selected mask.
    #[test]
    fn brush_drag_paints_a_stroke() {
        let image = Image::new(400, 300, vec![vec![0.3; 400 * 300]], 16);
        let mut harness = opened([1400.0, 900.0], image);
        {
            let app = harness.state_mut();
            app.prefs.current_area = "Maschere".into();
            app.add_layer(LayerTemplate::Brush);
        }
        harness.run_steps(3);
        let canvas = harness.state().document.as_ref().unwrap().viewer.canvas();
        let (a, b) = (canvas.center() - Vec2::new(60.0, 0.0), canvas.center() + Vec2::new(60.0, 0.0));
        let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
        harness.event(egui::Event::PointerMoved(a));
        harness.run_steps(1);
        harness.event(button(a, true));
        harness.run_steps(1);
        for k in 1..=12 {
            harness.event(egui::Event::PointerMoved(a + (b - a) * (k as f32 / 12.0)));
            harness.run_steps(1);
        }
        harness.event(button(b, false));
        harness.run_steps(2);
        let strokes = &harness.state().pipeline.layers[0].mask.strokes;
        assert_eq!(strokes.len(), 1, "one stroke");
        assert!(strokes[0].points.len() > 2, "{:?}", strokes[0].points);
    }

    /// The A/B snapshot belongs to the image it was taken on.
    #[test]
    fn snapshot_follows_the_image() {
        let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
        let mut harness = opened([1200.0, 800.0], image);
        for _ in 0..100 {
            harness.step();
            if harness.state().last_output.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let ctx = harness.ctx.clone();
        harness.state_mut().take_snapshot(&ctx);
        assert!(harness.state().snapshot.is_some() && harness.state().compare == Compare::Ab);
        harness.state_mut().rotate(&ctx, Geometry::rotate_clockwise);
        harness.run_steps(2);
        assert!(harness.state().snapshot.is_none() && harness.state().compare != Compare::Ab);
    }

    /// "Tieni A" brings A's settings back; "Tieni B" keeps the current ones; both end the
    /// comparison.
    #[test]
    fn comparison_keeps_a_or_b() {
        let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
        let mut harness = opened([1200.0, 800.0], image);
        for _ in 0..100 {
            harness.step();
            if harness.state().last_output.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let ctx = harness.ctx.clone();
        let a = harness.state().pipeline.clone();
        harness.state_mut().take_snapshot(&ctx);
        harness.state_mut().pipeline.tone.params.current.gamma = 1.7;
        let b = harness.state().pipeline.clone();
        harness.state_mut().end_comparison(&ctx, true);
        assert!(harness.state().snapshot.is_none() && harness.state().compare == Compare::After);
        assert_eq!(harness.state().pipeline, a, "A's settings are back");

        harness.state_mut().take_snapshot(&ctx);
        harness.state_mut().pipeline = b.clone();
        harness.state_mut().end_comparison(&ctx, false);
        assert!(harness.state().snapshot.is_none());
        assert_eq!(harness.state().pipeline, b, "B stays");
    }

    /// Keys 1‥6 pick an area; a double click on a card's title switches the module off.
    #[test]
    fn keys_and_double_click() {
        let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
        let mut harness = opened([1400.0, 900.0], image);
        harness.key_press(Key::Num4);
        harness.run_steps(2);
        assert_eq!(harness.state().prefs.current_area, "Curve e livelli");
        harness.key_press(Key::Num1);
        harness.run_steps(2);
        assert_eq!(harness.state().prefs.current_area, "Nitidezza");

        use egui_kittest::kittest::Queryable;
        assert!(harness.state().pipeline.wavelet.enabled);
        // Two quick clicks on the title.
        let at = harness.get_by_label_contains("Wavelet").rect().center();
        let button = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
        harness.event(egui::Event::PointerMoved(at));
        harness.step();
        for pressed in [true, false, true, false] {
            harness.event(button(pressed));
            harness.step();
        }
        harness.run_steps(2);
        assert!(!harness.state().pipeline.wavelet.enabled, "double click switches it off");
        assert!(!harness.state().prefs.expanded.contains(&Stage::Wavelet), "and leaves it folded");
    }

    /// Quick export writes next to the original and never overwrites.
    #[test]
    fn quick_export_next_to_the_original() {
        let dir = std::env::temp_dir().join(format!("ps-quick-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
        let mut harness = opened([1200.0, 800.0], image);
        let doc = harness.state_mut().document.as_mut().unwrap();
        (doc.path, doc.export_name) = (dir.join("giove.tif"), "giove".into());
        harness.state_mut().prefs.export_format = "png".into();
        let ctx = harness.ctx.clone();
        for _ in 0..2 {
            harness.state_mut().quick_export(&ctx);
            for _ in 0..200 {
                harness.step();
                if harness.state().io_busy.is_none() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        assert!(dir.join("giove_ps.png").is_file() && dir.join("giove_ps-2.png").is_file());
        std::fs::remove_dir_all(dir).ok();
    }

    /// Rotating from the crop bar re-orients the working image and keeps cropping.
    #[test]
    fn rotation_reorients_the_image() {
        let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
        let mut harness = opened([1200.0, 800.0], image);
        let ctx = harness.ctx.clone();
        harness.state_mut().start_crop();
        harness.state_mut().finish_crop(&ctx, CropAction::Rotate(1));
        let app = harness.state();
        let doc = app.document.as_ref().unwrap();
        assert_eq!((doc.source.width(), doc.source.height()), (32, 64));
        assert!(app.cropping, "still cropping after a rotation");
        // The left column (darkest) is now the top row.
        assert!(doc.source.channels()[0][..32].iter().all(|&v| v == 0.0));
    }

    /// Holding Space over the image shows the original; releasing it shows the result again.
    #[test]
    fn space_bar_peeks_at_the_original() {
        let image = Image::new(64, 64, vec![vec![0.3; 64 * 64]], 16);
        let mut harness = opened([1200.0, 800.0], image);
        harness.hover_at(egui::pos2(300.0, 400.0));
        harness.run_steps(2);
        let key = |pressed| egui::Event::Key { key: Key::Space, physical_key: None, pressed, repeat: false, modifiers: Modifiers::NONE };
        harness.event(key(true));
        harness.run_steps(2);
        assert_eq!(harness.state().shown_compare, Compare::Before);
        harness.event(key(false));
        harness.run_steps(2);
        assert_eq!(harness.state().shown_compare, Compare::After);
    }

    /// Regression: content asking for slightly more than the panel's width made the
    /// resizable side panel grow every frame until it covered the image.
    #[test]
    fn side_panel_width_is_stable() {
        let (w, h) = (300, 200);
        let plane: Vec<f32> = (0..w * h).map(|i| if ((i % w) as f32 - 150.0).hypot((i / w) as f32 - 100.0) < 60.0 { 0.6 } else { 0.02 }).collect();
        let image = Image::new(w, h, vec![plane.clone(), plane.clone(), plane], 16);
        let mut harness = opened([1400.0, 900.0], image);
        harness.run_steps(2);
        {
            let app = harness.state_mut();
            // Every card open, the widest the panel gets.
            let all: Vec<Stage> = app.prefs.modules.iter().map(|m| m.stage).collect();
            app.prefs.expanded.extend(all);
            app.add_layer(LayerTemplate::Brush);
        }
        let mut widths = Vec::new();
        for _ in 0..300 {
            harness.step();
            widths.push(harness.state().side_width);
        }
        let (min, max) = widths[10..].iter().fold((f32::MAX, f32::MIN), |(lo, hi), &w| (lo.min(w), hi.max(w)));
        assert!(max - min < 0.5, "panel width drifted from {min} to {max}");
        assert!(max < 560.0, "panel too wide: {max}");
    }

    /// Renders the UI off-screen to PNGs for visual review:
    /// `PS_SNAPSHOT_IMAGE=in.tif PS_SNAPSHOT_OUT=dir cargo test -p ps-app --release -- --ignored snapshot`
    #[test]
    #[ignore]
    fn snapshot() {
        let input = PathBuf::from(std::env::var("PS_SNAPSHOT_IMAGE").expect("PS_SNAPSHOT_IMAGE"));
        let out_dir = PathBuf::from(std::env::var("PS_SNAPSHOT_OUT").expect("PS_SNAPSHOT_OUT"));
        let mut harness = egui_kittest::Harness::builder().with_size([1500.0, 940.0]).wgpu().build_eframe(|cc| PlanetaryStudio::new(cc, None));
        harness.run_steps(3);

        let ctx = harness.ctx.clone();
        harness.state_mut().load(&ctx, input);
        let settle = |harness: &mut egui_kittest::Harness<PlanetaryStudio>| {
            for _ in 0..600 {
                harness.step();
                let app = harness.state();
                let idle = app.io_busy.is_none() && !app.worker.is_busy() && !app.worker.is_background_busy();
                if app.document.is_some() && app.last_output.is_some() && idle {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            harness.run_steps(4);
        };
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_initial.png")).unwrap();
        harness.state_mut().apply_auto_all();
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_auto.png")).unwrap();

        // "Correzione artefatti" with the dering card open: indented cards, reset buttons.
        {
            let app = harness.state_mut();
            app.prefs.current_area = "Correzione artefatti".into();
            app.prefs.expanded = [Stage::Dering].into();
            // A stroke along the right limb, the brush in hand and the ring shown.
            let doc = app.document.as_ref().unwrap();
            let (w, h) = (doc.source.width() as f32, doc.source.height() as f32);
            let (cx, cy, r) = doc.analysis.disk.as_ref().map_or((w / 2.0, h / 2.0, 0.3 * w.min(h)), |d| (d.cx, d.cy, d.radius + 8.0));
            let points = (0..=20).map(|k| -0.9 + 0.09 * k as f32).map(|a| [cx + r * a.cos(), cy + r * a.sin()]).collect();
            app.pipeline.dering.params.strokes = vec![BrushStroke { points, radius: 6.0, hardness: 0.5, erase: false }];
            (app.selected_layer, app.layer_tool, app.brush.radius, app.show_ringing) = (Some(RING_MASK), LayerTool::Brush, 6.0, true);
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_dering.png")).unwrap();

        // The deconvolution card switched on, then the comparison of shapes and algorithms.
        {
            let app = harness.state_mut();
            app.prefs.current_area = "Nitidezza".into();
            app.prefs.expanded = [Stage::Deconv].into();
            app.pipeline.deconv.enabled = true;
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_deconv.png")).unwrap();
        // A box over the whole image: the PSF from the brightest compact object (a moon, if any).
        {
            let app = harness.state_mut();
            let size = app.document.as_ref().map(|d| Vec2::new(d.source.width() as f32, d.source.height() as f32)).unwrap();
            app.measure_psf(Rect::from_min_size(Pos2::ZERO, size));
            eprintln!("misura: {:?}", app.psf_note);
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_deconv_measured.png")).unwrap();
        {
            let app = harness.state_mut();
            let doc = app.document.as_ref().unwrap();
            app.pipeline.deconv.params = ps_core::preset::auto_deconv(doc.kind, &doc.analysis, ps_core::preset::AutoStrength::Normal);
        }
        let ctx = harness.ctx.clone();
        harness.state_mut().start_deconv_compare(&ctx);
        for _ in 0..600 {
            harness.step();
            let done = harness.state().deconv_compare.as_ref().is_some_and(|c| c.tiles.iter().all(|t| t.texture.is_some()));
            if done {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_deconv_compare.png")).unwrap();
        harness.state_mut().deconv_compare = None;

        // The equipment tab: a C8 with a 2.5× Barlow and a 2.9 µm camera.
        {
            let app = harness.state_mut();
            app.instrument_open = true;
            if let Some(doc) = &mut app.document {
                doc.setup.telescope = ps_core::equipment::telescopes().into_iter().find(|t| t.name.starts_with("Celestron C8"));
                doc.setup.camera = ps_core::equipment::cameras().into_iter().find(|c| c.name == "ZWO ASI462MC");
                doc.setup.barlow = ps_core::equipment::barlows().into_iter().find(|b| b.name == "Barlow 2,5×");
                doc.setup.filter = ps_core::equipment::filter_named("RGB");
            }
            app.equipment_changed();
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_instrument.png")).unwrap();
        harness.state_mut().instrument_open = false;
        // "Nitidezza intelligente": main sliders, curve, and the map of where it holds back.
        {
            let app = harness.state_mut();
            app.prefs.current_area = "Nitidezza".into();
            app.prefs.expanded = [Stage::Smart].into();
            (app.pipeline.smart.enabled, app.pipeline.smart.params.strength) = (true, 0.6);
        }
        settle(&mut harness);
        if let Some(node) = egui_kittest::kittest::Queryable::query_by_label(&harness, "AVANZATE") {
            node.click();
        }
        harness.run_steps(2);
        harness.state_mut().smart_view = SmartView::HoldBack;
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_smart.png")).unwrap();
        harness.state_mut().smart_view = SmartView::Resolution;
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_smart_resolution.png")).unwrap();
        // The whole tab, down to the spectrum.
        {
            let doc = harness.state().document.as_ref().unwrap();
            let (mut setup, mut capture, analysis, wavelet) = (doc.setup.clone(), doc.capture.clone(), doc.analysis.clone(), harness.state().pipeline.wavelet.params.clone());
            let mut tab = egui_kittest::Harness::builder().with_size([420.0, 1000.0]).with_accessibility_check(false).build_ui(move |ui| {
                crate::theme::apply(ui.ctx());
                egui::CentralPanel::default().show(ui, |ui| {
                    let view = InstrumentView { setup: &mut setup, capture: &mut capture, analysis: &analysis, wavelet: &wavelet };
                    instrument_body(ui, view, &mut Vec::new());
                });
            });
            tab.run();
            tab.render().unwrap().save(out_dir.join("ui_instrument_full.png")).unwrap();
        }
        harness.state_mut().prefs.current_area = "Nitidezza".into();

        // The wavelet card with free scales, then switched to the classic ones.
        harness.state_mut().prefs.expanded = [Stage::Wavelet].into();
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_wavelet.png")).unwrap();
        // Scrolled down inside the card: its title stays pinned at the top of the panel.
        harness.event(egui::Event::PointerMoved(egui::pos2(1300.0, 600.0)));
        for _ in 0..6 {
            harness.event(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: Vec2::new(0.0, -120.0), modifiers: Modifiers::NONE, phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel });
            harness.run_steps(1);
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_wavelet_sticky.png")).unwrap();
        harness.event(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: Vec2::new(0.0, 2000.0), modifiers: Modifiers::NONE, phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel });
        harness.run_steps(2);
        egui_kittest::kittest::Queryable::get_by_label(&harness, "Classiche 1·2·4·8").click();
        settle(&mut harness);
        assert_eq!(harness.state().pipeline.wavelet.params.scale_mode, ScaleMode::Dyadic);
        assert_eq!(harness.state().prefs.wavelet_scales, ScaleMode::Dyadic);
        harness.render().unwrap().save(out_dir.join("ui_wavelet_classic.png")).unwrap();

        egui_kittest::kittest::Queryable::get_by_label(&harness, "Libere").click();
        settle(&mut harness);

        // The moons' layer: what it selects on top and on the image, then one adjustment.
        {
            let app = harness.state_mut();
            app.prefs.current_area = "Maschere".into();
            if !app.pipeline.layers.iter().any(|l| l.role == Some(LayerRole::Moons)) {
                add_role_layer(&mut app.pipeline, LayerRole::Moons);
            }
            // A drawn mask that changes nothing yet, switched off, to show its warning dimmed.
            app.add_layer(LayerTemplate::Ellipse);
            if let Some(l) = app.pipeline.layers.last_mut() {
                l.enabled = false;
            }
            app.selected_layer = app.pipeline.layers.iter().find(|l| l.role == Some(LayerRole::Moons)).map(|l| l.id);
            app.layer_tool = LayerTool::None;
            app.prefs.collapsed_sections.insert("layer-open:wavelet".into());
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_layer.png")).unwrap();

        // The ROI being edited, with the button that goes back to moving the view.
        {
            let app = harness.state_mut();
            app.prefs.expanded.clear();
            app.set_preview(PreviewMode::Roi);
            app.tool = Tool::Roi;
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_roi.png")).unwrap();
        harness.state_mut().set_preview(PreviewMode::Full);

        // Zoomed split view with the deconvolution card folded.
        {
            let app = harness.state_mut();
            app.compare = Compare::Split;
            app.prefs.expanded.remove(&Stage::Deconv);
            app.document.as_mut().unwrap().viewer.set_zoom(2.5);
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_split_zoom.png")).unwrap();

        // Only "Curve e livelli" open: a committed step plus a new curve, histogram behind it.
        let open_only = |app: &mut PlanetaryStudio, group: &str| app.prefs.current_area = group.into();
        {
            let app = harness.state_mut();
            app.compare = Compare::After;
            app.document.as_mut().unwrap().viewer.request_fit();
            open_only(app, "Curve e livelli");
            app.prefs.expanded.extend([Stage::Tone, Stage::Curves]);
            app.pipeline.tone.params.current.gamma = 1.3;
            app.pipeline.tone.params.commit();
            app.pipeline.curves.params.current.curve = vec![[0.0, 0.0], [0.4, 0.5], [1.0, 1.0]];
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_tone.png")).unwrap();
        harness.state_mut().prefs.current_area = "Esporta".into();
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_export.png")).unwrap();

        // A mosaic of the frames listed in PS_SNAPSHOT_MOSAIC (separated by '|'): the window,
        // then the result.
        if let Ok(frames) = std::env::var("PS_SNAPSHOT_MOSAIC") {
            harness.state_mut().mosaic_editor.start(&ctx, frames.split('|').map(PathBuf::from).collect());
            for _ in 0..1500 {
                harness.step();
                if !harness.state().mosaic_editor.is_busy() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            harness.run_steps(4);
            harness.render().unwrap().save(out_dir.join("ui_mosaic.png")).unwrap();
            egui_kittest::kittest::Queryable::get_by_label_contains(&harness, "Crea il mosaico").click();
            for _ in 0..1500 {
                harness.step();
                if harness.state().document.as_ref().is_some_and(|d| d.mosaic.is_some()) && harness.state().last_output.is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            settle(&mut harness);
            harness.render().unwrap().save(out_dir.join("ui_mosaic_done.png")).unwrap();
            let doc = harness.state().document.as_ref().unwrap();
            eprintln!("mosaic {}x{}", doc.source.width(), doc.source.height());
        }

        // Saturn: the dedicated rings card, with the rest folded away.
        if let Ok(saturn) = std::env::var("PS_SNAPSHOT_SATURN") {
            harness.state_mut().load(&ctx, PathBuf::from(saturn));
            for _ in 0..200 {
                harness.step();
                // The fresh image drops the masks made on the previous one (same file or not).
                let app = harness.state();
                if app.document.as_ref().is_some_and(|d| d.kind == ImageKind::Saturn) && app.pipeline.layers.is_empty() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            {
                let app = harness.state_mut();
                open_only(app, "Maschere");
                app.apply_actions(&ctx, vec![PanelAction::CreateRole(LayerRole::Rings)]);
                app.selected_layer = app.pipeline.layers.iter().find(|l| l.role == Some(LayerRole::Rings)).map(|l| l.id);
                app.show_mask = true;
            }
            settle(&mut harness);
            harness.render().unwrap().save(out_dir.join("ui_saturn_rings.png")).unwrap();

            // Deringing on the globe only, its area shown on the image.
            {
                let app = harness.state_mut();
                open_only(app, "Nitidezza");
                app.prefs.expanded = [Stage::Dering].into();
                app.pipeline.dering.params.strength = 0.8;
                app.selected_layer = Some(RING_MASK);
                app.show_mask = true;
            }
            settle(&mut harness);
            harness.render().unwrap().save(out_dir.join("ui_saturn_dering.png")).unwrap();
        }

        // Crop tool and settings window.
        harness.state_mut().start_crop();
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_crop.png")).unwrap();
        harness.state_mut().finish_crop(&ctx, CropAction::Cancel);
        harness.state_mut().settings.open = true;
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_settings.png")).unwrap();
    }
}
