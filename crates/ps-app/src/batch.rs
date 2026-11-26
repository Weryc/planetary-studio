//! A series of images processed and exported into a folder with the same settings: every file
//! is opened, analysed (subject from the name, the log or the pixels), given the settings and
//! saved, one after the other in the background, with progress and a way to stop.

use crate::instrument::CaptureState;
use eframe::egui;
use ps_core::analysis::analyze;
use ps_core::equipment::Setup;
use ps_core::pipeline::{Aux, Pipeline, Processor, next_generation};
use ps_core::preset::AutoStrength;
use ps_core::wavelet::ScaleMode;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};

/// What every image of the series gets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// The pre-processed settings, worked out for each image.
    Preprocessed,
    /// The settings of the image open now (its mask layers and strokes stay out: they belong to it).
    Current,
}

pub struct Job {
    pub files: Vec<PathBuf>,
    pub dir: PathBuf,
    pub extension: String,
    pub suffix: String,
    pub mode: Mode,
    pub strength: AutoStrength,
    pub scales: ScaleMode,
    pub setup: Setup,
    pub current: Pipeline,
}

enum Step {
    Started(String),
    Done(PathBuf),
    Failed(String),
    Finished,
}

/// A series being processed.
pub struct Run {
    pub total: usize,
    pub done: usize,
    pub failed: Vec<String>,
    pub current: String,
    pub last: Option<PathBuf>,
    pub finished: bool,
    pub cancelled: bool,
    dir: PathBuf,
    rx: Receiver<Step>,
    cancel: Arc<AtomicBool>,
}

/// The first free name `stem{suffix}.ext` (then `-2`, `-3`…) in `dir`.
pub fn free_name(dir: &Path, stem: &str, suffix: &str, extension: &str) -> PathBuf {
    (1..)
        .map(|k| dir.join(if k == 1 { format!("{stem}{suffix}.{extension}") } else { format!("{stem}{suffix}-{k}.{extension}") }))
        .find(|p| !p.exists())
        .expect("some name is free")
}

/// One image through the job's settings.
fn process(path: &Path, job: &Job, cancel: &AtomicBool) -> Result<Option<PathBuf>, String> {
    let image = ps_io::load(path).map_err(|e| e.to_string())?;
    let mut analysis = analyze(&image);
    let mut capture = CaptureState::read(path);
    capture.body = capture.body.or(analysis.planet);
    capture.measure(&image, &analysis);
    let setup = Setup::from_capture(&capture.info, &job.setup);
    analysis.optics = setup.optics();
    analysis.curve_net = capture.body.and_then(|b| ps_core::curve_net::suggestion(&image, &analysis, b).ok());
    let pipeline = match job.mode {
        Mode::Preprocessed => ps_core::preset::preprocessed(analysis.kind, &analysis, &image, capture.body, job.strength, job.scales),
        Mode::Current => {
            let mut p = Pipeline { layers: Vec::new(), ..job.current.clone() };
            p.dering.params.strokes.clear();
            p
        }
    };
    let source = Arc::new(image);
    let masks = ps_core::pipeline::masks_for(&pipeline, &source);
    let aux = Aux { masks: &masks, ..Default::default() };
    let Some(out) = Processor::default().run(source, next_generation(), &pipeline, &aux, &|| cancel.load(Ordering::Relaxed), &|_, _| {}) else {
        return Ok(None);
    };
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let target = free_name(&job.dir, stem, &job.suffix, &job.extension);
    ps_io::save(&target, &out.image).map_err(|e| e.to_string())?;
    Ok(Some(target))
}

impl Run {
    pub fn start(ctx: &egui::Context, job: Job) -> Self {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let run = Self { total: job.files.len(), done: 0, failed: Vec::new(), current: String::new(), last: None, finished: false, cancelled: false, dir: job.dir.clone(), rx, cancel: Arc::clone(&cancel) };
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            for path in &job.files {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                tx.send(Step::Started(name.clone())).ok();
                ctx.request_repaint();
                let step = match process(path, &job, &cancel) {
                    Ok(Some(target)) => Step::Done(target),
                    Ok(None) => break,
                    Err(why) => Step::Failed(format!("{name}: {why}")),
                };
                tx.send(step).ok();
                ctx.request_repaint();
            }
            tx.send(Step::Finished).ok();
            ctx.request_repaint();
        });
        run
    }

    /// Take what the thread reported since the last call.
    pub fn poll(&mut self) {
        while let Ok(step) = self.rx.try_recv() {
            match step {
                Step::Started(name) => self.current = name,
                Step::Done(path) => {
                    self.done += 1;
                    self.last = Some(path);
                }
                Step::Failed(why) => {
                    self.done += 1;
                    self.failed.push(why);
                }
                Step::Finished => {
                    self.finished = true;
                    self.cancelled = self.cancel.load(Ordering::Relaxed);
                }
            }
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}
