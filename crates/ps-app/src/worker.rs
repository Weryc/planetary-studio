//! Background processing thread. The UI never computes: it submits the latest pipeline and
//! shows whatever result is ready. Requests are coalesced (only the newest is processed) and
//! a running job is cancelled as soon as a newer one arrives.
//!
//! While previewing a ROI, idle time is used to render the full image in the background so
//! that switching to the full view, or exporting, is immediate.

use crate::widgets::Histogram;
use eframe::egui::{self, Color32, ColorImage};
use rayon::prelude::*;
use ps_core::Image;
use ps_core::mask::{self, MaskSpec, Stroke};
use ps_core::dering::{self, AreaKey};
use ps_core::pipeline::{Aux, DERING_MASK, Pipeline, Processor, RING_MASK, SMART_MASK, Stage, StageTiming, next_generation};
use ps_core::smart::{SmartMap, SmartView};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Extra context processed around a ROI so edge-sensitive steps (large wavelet scales,
/// deconvolution) behave as they would on the full image.
const ROI_MARGIN: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Roi {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

#[derive(Clone)]
pub struct Request {
    pub source: Arc<Image>,
    pub source_gen: u64,
    pub pipeline: Pipeline,
    pub roi: Option<Roi>,
    pub max_texture_side: usize,
    /// LRGB luminance at the source's size, with its generation.
    pub luminance: Option<Plane>,
    pub bxt_executable: Option<PathBuf>,
    /// Layer whose mask should be returned as an overlay for display.
    pub mask_overlay: Option<u64>,
    /// Keep only the expensive stages cached.
    pub memory_saving: bool,
    /// Quick approximate preview of the image shrunk this many times (1 = full resolution).
    /// Only for the whole image.
    pub proxy: usize,
    /// Return the map of rings left by sharpening, as an overlay.
    pub ringing_map: bool,
    /// Return the clipped pixels, as an overlay.
    pub clip_warning: bool,
    /// Return this map of the smart sharpening, as an overlay.
    pub smart_view: SmartView,
}

#[derive(Clone)]
pub struct Output {
    pub source_gen: u64,
    pub pipeline: Pipeline,
    pub roi: Option<Roi>,
    /// Processed pixels: the whole image, or just the ROI.
    pub image: Arc<Image>,
    pub display: Arc<ColorImage>,
    pub histogram: Histogram,
    /// What the tone module's current step works on (same region as `image`), and its
    /// histogram for the curve editor.
    pub tone_input: Arc<Image>,
    pub tone_histogram: Histogram,
    /// Histogram of what the curves' current step works on.
    pub curves_histogram: Histogram,
    /// What the deconvolution did.
    pub deconv: Option<ps_core::deconv::DeconvReport>,
    pub timings: Vec<StageTiming>,
    pub warnings: Vec<(Stage, String)>,
    pub elapsed: Duration,
    /// Full-image render made in the background while a ROI was being previewed.
    pub background: bool,
    /// (layer id, overlay) for the requested layer mask, at full-image display size.
    pub mask_overlay: Option<(u64, Arc<ColorImage>)>,
    /// A small picture of each resolved layer mask, for the list of masks.
    pub mask_thumbs: Vec<MaskThumb>,
    /// A reduced, approximate preview: `image` is smaller than the source.
    pub proxy: bool,
    /// Rings left by sharpening (same region as `image`).
    pub ringing_overlay: Option<Arc<ColorImage>>,
    /// Clipped pixels (same region as `image`) and how many.
    pub clip_overlay: Option<(Arc<ColorImage>, usize)>,
    /// The smart sharpening's map (same region as `image`) and, for the resolution, the
    /// periods (px) its colours span.
    pub smart_overlay: Option<(Arc<ColorImage>, Option<(f32, f32)>)>,
}

/// A layer mask in miniature: white where it applies.
#[derive(Clone)]
pub struct MaskThumb {
    pub layer: u64,
    /// Changes whenever the mask does.
    pub generation: u64,
    pub image: Arc<ColorImage>,
    /// Selects (almost) nothing.
    pub empty: bool,
}

/// Longest side of a mask's miniature, in pixels.
const THUMB_SIDE: usize = 56;

#[derive(Clone, Copy)]
pub struct Progress {
    pub stage: Stage,
    pub fraction: f32,
}

pub struct Worker {
    jobs: Sender<(u64, Request)>,
    results: Receiver<Output>,
    latest: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    progress: Arc<Mutex<Option<Progress>>>,
    background_busy: Arc<AtomicBool>,
}

impl Worker {
    pub fn new(ctx: egui::Context) -> Self {
        let (jobs, job_rx) = channel::<(u64, Request)>();
        let (result_tx, results) = channel();
        let latest = Arc::new(AtomicU64::new(0));
        let completed = Arc::new(AtomicU64::new(0));
        let progress = Arc::new(Mutex::new(None));
        let background_busy = Arc::new(AtomicBool::new(false));
        let state = WorkerThread {
            jobs: job_rx,
            results: result_tx,
            latest: Arc::clone(&latest),
            completed: Arc::clone(&completed),
            progress: Arc::clone(&progress),
            background_busy: Arc::clone(&background_busy),
            ctx,
            masks: HashMap::new(),
            overlays: HashMap::new(),
            thumbs: HashMap::new(),
            roi_planes: HashMap::new(),
            proxy_planes: HashMap::new(),
        };
        std::thread::Builder::new()
            .name("processing".into())
            .spawn(move || state.run())
            .expect("spawn processing thread");
        Self { jobs, results, latest, completed, progress, background_busy }
    }

    pub fn submit(&self, request: Request) {
        let id = self.latest.fetch_add(1, Ordering::SeqCst) + 1;
        self.jobs.send((id, request)).ok();
    }

    /// Results that arrived since the last call.
    pub fn poll(&self) -> Vec<Output> {
        self.results.try_iter().collect()
    }

    pub fn is_busy(&self) -> bool {
        self.latest.load(Ordering::SeqCst) != self.completed.load(Ordering::SeqCst)
    }

    pub fn progress(&self) -> Option<Progress> {
        *self.progress.lock().unwrap()
    }

    /// True while the full image is being rendered in the background.
    pub fn is_background_busy(&self) -> bool {
        self.background_busy.load(Ordering::SeqCst)
    }
}

/// A plane (mask or luminance) with the generation identifying its content.
type Plane = (Arc<Vec<f32>>, u64);

/// What a full-size mask is computed from (besides the source).
#[derive(PartialEq)]
enum MaskKey {
    Layer(MaskSpec),
    Dering(AreaKey),
    /// Where the user painted over the ring.
    Ring(Vec<Stroke>),
    /// The subject and its moons: depends on the source only.
    Subject,
}

/// A full-size mask and the inputs it was computed from.
struct MaskEntry {
    key: MaskKey,
    source_gen: u64,
    plane: Arc<Vec<f32>>,
    generation: u64,
}

struct WorkerThread {
    jobs: Receiver<(u64, Request)>,
    results: Sender<Output>,
    latest: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    progress: Arc<Mutex<Option<Progress>>>,
    background_busy: Arc<AtomicBool>,
    ctx: egui::Context,
    /// Full-size masks by layer id.
    masks: HashMap<u64, MaskEntry>,
    /// Display overlays by mask generation.
    overlays: HashMap<u64, Arc<ColorImage>>,
    /// Mask miniatures by mask generation: (image, empty).
    thumbs: HashMap<u64, (Arc<ColorImage>, bool)>,
    /// ROI crops of auxiliary planes: (full generation, ROI) → (crop, its generation).
    roi_planes: HashMap<(u64, Roi), Plane>,
    /// Reduced auxiliary planes for the quick preview: (full generation, factor) → plane.
    proxy_planes: HashMap<(u64, usize), Plane>,
}

/// What a request is processed on, besides the full source.
enum Input<'a> {
    Full,
    Roi(&'a RoiSource),
    Proxy(&'a ProxySource),
}

/// The source shrunk for the quick preview.
struct ProxySource {
    source_gen: u64,
    factor: usize,
    image: Arc<Image>,
    generation: u64,
}

/// Cropped ROI input of the last request.
struct RoiSource {
    source_gen: u64,
    roi: Roi,
    image: Arc<Image>,
    generation: u64,
    /// Region of the full image that `image` covers.
    region: Roi,
}

impl WorkerThread {
    fn run(mut self) {
        // Separate caches so ROI previews and the full render don't evict each other.
        let mut roi_processor = Processor::default();
        let mut full_processor = Processor::default();
        // A third cache for the quick preview, so dragging doesn't evict full-size results.
        let mut proxy_processor = Processor::default();
        let mut proxy_source: Option<ProxySource> = None;
        // The cropped ROI source keeps its generation while the ROI is unchanged, so the
        // pipeline cache keeps working in ROI mode.
        let mut roi_source: Option<RoiSource> = None;
        // Full-image version of the last ROI request, rendered once nothing else is queued.
        let mut pending_full: Option<Request> = None;

        loop {
            let next = if pending_full.is_some() {
                match self.jobs.try_recv() {
                    Ok(job) => Some(job),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => return,
                }
            } else {
                match self.jobs.recv() {
                    Ok(job) => Some(job),
                    Err(_) => return,
                }
            };

            let Some(mut job) = next else {
                let req = pending_full.take().expect("checked above");
                self.background_busy.store(true, Ordering::SeqCst);
                let id = self.latest.load(Ordering::SeqCst);
                let latest = Arc::clone(&self.latest);
                let cancel = move || latest.load(Ordering::SeqCst) != id;
                full_processor.memory_saving = req.memory_saving;
                let output = self.render(&mut full_processor, req, Input::Full, &cancel, &|_, _| {}, true);
                self.background_busy.store(false, Ordering::SeqCst);
                if let Some(output) = output {
                    if self.results.send(output).is_err() {
                        return;
                    }
                    self.ctx.request_repaint();
                }
                continue;
            };

            if let Some(newer) = self.jobs.try_iter().last() {
                job = newer;
            }
            let (id, req) = job;
            pending_full = req.roi.map(|_| Request { roi: None, proxy: 1, ..req.clone() });

            let (processor, input) = match req.roi {
                None if req.proxy > 1 => {
                    let f = req.proxy;
                    if !matches!(&proxy_source, Some(c) if c.source_gen == req.source_gen && c.factor == f) {
                        proxy_source = Some(ProxySource { source_gen: req.source_gen, factor: f, image: Arc::new(req.source.downscaled(f)), generation: next_generation() });
                    }
                    (&mut proxy_processor, Input::Proxy(proxy_source.as_ref().expect("just set")))
                }
                None => (&mut full_processor, Input::Full),
                Some(roi) => {
                    let cached = matches!(&roi_source, Some(c) if c.source_gen == req.source_gen && c.roi == roi);
                    if !cached {
                        let region = region_with_margin(&req.source, roi);
                        let image = Arc::new(req.source.crop(region.x, region.y, region.w, region.h));
                        roi_source = Some(RoiSource { source_gen: req.source_gen, roi, image, generation: next_generation(), region });
                    }
                    (&mut roi_processor, Input::Roi(roi_source.as_ref().expect("just set")))
                }
            };
            processor.memory_saving = req.memory_saving;
            let latest = Arc::clone(&self.latest);
            let cancel = move || latest.load(Ordering::SeqCst) != id;
            let progress = Arc::clone(&self.progress);
            let ctx = self.ctx.clone();
            let report = move |stage: Stage, fraction: f32| {
                *progress.lock().unwrap() = Some(Progress { stage, fraction });
                ctx.request_repaint();
            };
            let output = self.render(processor, req, input, &cancel, &report, false);
            *self.progress.lock().unwrap() = None;
            self.completed.store(id, Ordering::SeqCst);

            // `None` = cancelled: a newer request is already queued.
            if let Some(output) = output {
                if self.results.send(output).is_err() {
                    return;
                }
                self.ctx.request_repaint();
            }
        }
    }

    /// Up-to-date full-size masks for the enabled layers and the deringing area.
    fn resolve_masks(&mut self, req: &Request) -> Vec<(u64, Arc<Vec<f32>>, u64)> {
        let p = &req.pipeline;
        let mut wanted: Vec<(u64, MaskKey)> =
            p.layers.iter().filter(|l| l.enabled || req.mask_overlay == Some(l.id)).map(|l| (l.id, MaskKey::Layer(l.mask.clone()))).collect();
        if ps_core::pipeline::needs_dering_area(p) || req.mask_overlay == Some(DERING_MASK) {
            wanted.push((DERING_MASK, MaskKey::Dering(p.dering.params.area_key())));
        }
        // The painted ring: also for its map and while its strokes are shown.
        if ps_core::pipeline::needs_ring_area(p) || req.mask_overlay == Some(RING_MASK) || (req.ringing_map && !p.dering.params.strokes.is_empty()) {
            wanted.push((RING_MASK, MaskKey::Ring(p.dering.params.strokes.clone())));
        }
        // The smart sharpening's maps show the subject only, even when it sharpens the sky.
        if ps_core::pipeline::needs_smart_mask(p) || req.smart_view != SmartView::None {
            wanted.push((SMART_MASK, MaskKey::Subject));
        }
        self.masks.retain(|id, _| wanted.iter().any(|(w, _)| w == id));
        wanted
            .into_iter()
            .map(|(id, key)| {
                let fresh = matches!(self.masks.get(&id), Some(e) if e.key == key && e.source_gen == req.source_gen);
                if !fresh {
                    let plane = Arc::new(match &key {
                        MaskKey::Layer(spec) => mask::render(spec, &req.source),
                        MaskKey::Dering(_) => dering::weight(&req.source, &p.dering.params),
                        MaskKey::Ring(strokes) => dering::painted(strokes, req.source.width(), req.source.height()),
                        MaskKey::Subject => mask::subject_and_moons(&req.source),
                    });
                    self.masks.insert(id, MaskEntry { key, source_gen: req.source_gen, plane, generation: next_generation() });
                }
                let e = &self.masks[&id];
                (id, Arc::clone(&e.plane), e.generation)
            })
            .collect()
    }

    /// The crop of a full-size plane matching the ROI input, cached so its generation is stable.
    fn roi_plane(&mut self, plane: &Arc<Vec<f32>>, generation: u64, full_w: usize, region: Roi, roi: Roi) -> Plane {
        self.roi_planes.retain(|(_, r), _| *r == roi);
        self.roi_planes
            .entry((generation, roi))
            .or_insert_with(|| {
                let crop = (region.y..region.y + region.h).flat_map(|y| plane[y * full_w + region.x..y * full_w + region.x + region.w].iter().copied()).collect();
                (Arc::new(crop), next_generation())
            })
            .clone()
    }

    /// A full-size plane shrunk like the quick preview's source, cached so its generation is
    /// stable.
    fn proxy_plane(&mut self, plane: &Arc<Vec<f32>>, generation: u64, w: usize, h: usize, factor: usize) -> Plane {
        self.proxy_planes.retain(|(_, f), _| *f == factor);
        self.proxy_planes
            .entry((generation, factor))
            .or_insert_with(|| (Arc::new(ps_core::image::downscale_plane(plane, w, h, factor)), next_generation()))
            .clone()
    }

    /// Run the pipeline on the full source, its ROI crop or its reduced copy.
    fn render(
        &mut self,
        processor: &mut Processor,
        req: Request,
        input: Input,
        cancel: &dyn Fn() -> bool,
        report: &dyn Fn(Stage, f32),
        background: bool,
    ) -> Option<Output> {
        let start = Instant::now();
        let full_masks = self.resolve_masks(&req);
        let (full_w, full_h) = (req.source.width(), req.source.height());
        let mut pipeline = req.pipeline.clone();
        let roi_input = match input {
            Input::Roi(c) => Some(c),
            _ => None,
        };
        let (input, input_gen, masks, luminance) = match input {
            Input::Proxy(c) => {
                let f = c.factor;
                pipeline = pipeline.downscaled(f);
                let masks = full_masks.iter().map(|(id, plane, g)| {
                    let (small, sg) = self.proxy_plane(plane, *g, full_w, full_h, f);
                    (*id, small, sg)
                });
                let masks: Vec<_> = masks.collect();
                let luminance = req.luminance.as_ref().map(|(plane, g)| self.proxy_plane(plane, *g, full_w, full_h, f));
                (Arc::clone(&c.image), c.generation, masks, luminance)
            }
            Input::Roi(c) => {
                let masks = full_masks.iter().map(|(id, plane, g)| {
                    let (crop, cg) = self.roi_plane(plane, *g, full_w, c.region, c.roi);
                    (*id, crop, cg)
                });
                let masks: Vec<_> = masks.collect();
                let luminance = req.luminance.as_ref().map(|(plane, g)| self.roi_plane(plane, *g, full_w, c.region, c.roi));
                (Arc::clone(&c.image), c.generation, masks, luminance)
            }
            Input::Full => (Arc::clone(&req.source), req.source_gen, full_masks.clone(), req.luminance.clone()),
        };
        let aux = Aux { masks: &masks, luminance, bxt_executable: req.bxt_executable.as_deref(), ringing_map: req.ringing_map, smart_view: req.smart_view };
        // What the clipping warning compares with.
        let unprocessed = Arc::clone(&input);
        let out = processor.run(input, input_gen, &pipeline, &aux, cancel, report)?;
        let to_region = |img: Arc<Image>| match (req.roi, roi_input) {
            (Some(roi), Some(c)) => Arc::new(img.crop(roi.x - c.region.x, roi.y - c.region.y, roi.w, roi.h)),
            _ => img,
        };
        // Without applied tone steps the tone input is the output itself: one histogram serves both.
        let same = Arc::ptr_eq(&out.image, &out.tone_input);
        let (curves_like_tone, curves_like_image) = (Arc::ptr_eq(&out.curves_input, &out.tone_input), Arc::ptr_eq(&out.curves_input, &out.image));
        let ringing_overlay = out.ringing.as_ref().map(|m| {
            let (w, h) = (out.image.width(), out.image.height());
            let full = Image::new(w, h, vec![m.to_vec()], 16);
            let map = to_region(Arc::new(full));
            Arc::new(heat_overlay(&map.channels()[0], map.width(), map.height(), req.max_texture_side, [255, 60, 230], 0.004, 0.03))
        });
        let smart_overlay = out.smart_map.as_ref().map(|map| {
            let (w, h) = (out.image.width(), out.image.height());
            let crop = |plane: &[f32]| to_region(Arc::new(Image::new(w, h, vec![plane.to_vec()], 16)));
            match map.as_ref() {
                SmartMap::HoldBack { zones, edges, excluded } => {
                    let (zones, edges, excluded) = (crop(zones), crop(edges), excluded.as_ref().map(|e| crop(e)));
                    let excluded = excluded.as_ref().map(|e| e.channels()[0].as_slice());
                    let overlay = hold_back_overlay(&zones.channels()[0], &edges.channels()[0], excluded, zones.width(), zones.height(), req.max_texture_side);
                    (Arc::new(overlay), None)
                }
                SmartMap::Resolution(periods) => {
                    let periods = crop(periods);
                    let range = resolution_range(&periods.channels()[0]);
                    (Arc::new(resolution_overlay(&periods.channels()[0], periods.width(), periods.height(), req.max_texture_side, range)), Some(range))
                }
            }
        });
        let image = to_region(out.image);
        let clip_overlay = req.clip_warning.then(|| {
            let before = to_region(unprocessed);
            let (overlay, count) = clip_overlay(&image, &before, req.max_texture_side);
            (Arc::new(overlay), count)
        });
        let tone_input = if same { Arc::clone(&image) } else { to_region(out.tone_input) };
        let display = Arc::new(display_image(&image, req.max_texture_side));
        let histogram = Histogram::compute(&image);
        let tone_histogram = if same { histogram.clone() } else { Histogram::compute(&tone_input) };
        let curves_histogram = if curves_like_tone {
            tone_histogram.clone()
        } else if curves_like_image {
            histogram.clone()
        } else {
            Histogram::compute(&to_region(out.curves_input))
        };
        let mask_overlay = req.mask_overlay.and_then(|id| {
            let (_, plane, generation) = full_masks.iter().find(|(lid, ..)| *lid == id)?;
            let overlay = self
                .overlays
                .entry(*generation)
                .or_insert_with(|| Arc::new(mask_overlay(plane, req.source.width(), req.source.height(), req.max_texture_side)))
                .clone();
            Some((id, overlay))
        });
        let (w, h) = (req.source.width(), req.source.height());
        let mask_thumbs = full_masks
            .iter()
            .filter(|(id, ..)| req.pipeline.layers.iter().any(|l| l.id == *id))
            .map(|(id, plane, generation)| {
                let (image, empty) = self.thumbs.entry(*generation).or_insert_with(|| mask_thumb(plane, w, h)).clone();
                MaskThumb { layer: *id, generation: *generation, image, empty }
            })
            .collect();
        let live: Vec<u64> = full_masks.iter().map(|m| m.2).collect();
        self.overlays.retain(|g, _| live.contains(g));
        self.thumbs.retain(|g, _| live.contains(g));
        Some(Output {
            source_gen: req.source_gen,
            pipeline: req.pipeline,
            roi: req.roi,
            image,
            display,
            histogram,
            tone_input,
            tone_histogram,
            curves_histogram,
            deconv: out.deconv,
            timings: out.timings,
            warnings: out.warnings,
            elapsed: start.elapsed(),
            background,
            mask_overlay,
            mask_thumbs,
            proxy: req.proxy > 1 && req.roi.is_none(),
            ringing_overlay,
            clip_overlay,
            smart_overlay,
        })
    }
}

/// The ROI plus [`ROI_MARGIN`] around it, clamped to the image.
fn region_with_margin(img: &Image, roi: Roi) -> Roi {
    let x0 = roi.x.saturating_sub(ROI_MARGIN);
    let y0 = roi.y.saturating_sub(ROI_MARGIN);
    let x1 = (roi.x + roi.w + ROI_MARGIN).min(img.width());
    let y1 = (roi.y + roi.h + ROI_MARGIN).min(img.height());
    Roi { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
}

/// Integer box-downsampling factor that fits `w`×`h` in the GPU texture limit.
fn display_factor(w: usize, h: usize, max_side: usize) -> usize {
    w.max(h).div_ceil(max_side).max(1)
}

/// Convert to an egui texture image, box-downsampling if it exceeds the GPU texture limit.
pub fn display_image(image: &Image, max_side: usize) -> ColorImage {
    let (w, h) = (image.width(), image.height());
    let rgba = image.to_rgba8();
    let factor = display_factor(w, h, max_side);
    if factor == 1 {
        return ColorImage::from_rgba_unmultiplied([w, h], &rgba);
    }
    let (dw, dh) = (w / factor, h / factor);
    let mut out = vec![0u8; dw * dh * 4];
    out.par_chunks_mut(dw * 4).enumerate().for_each(|(y, row)| {
        for x in 0..dw {
            let mut acc = [0u32; 4];
            for sy in 0..factor {
                for sx in 0..factor {
                    let s = ((y * factor + sy) * w + x * factor + sx) * 4;
                    for c in 0..4 {
                        acc[c] += rgba[s + c] as u32;
                    }
                }
            }
            for c in 0..4 {
                row[x * 4 + c] = (acc[c] / (factor * factor) as u32) as u8;
            }
        }
    });
    ColorImage::from_rgba_unmultiplied([dw, dh], &out)
}

/// `colour` over the pixels where `plane` rises from `lo` to `hi` (transparent elsewhere).
fn heat_overlay(plane: &[f32], w: usize, h: usize, max_side: usize, colour: [u8; 3], lo: f32, hi: f32) -> ColorImage {
    let f = display_factor(w, h, max_side);
    let (dw, dh) = (w / f, h / f);
    let pixels = (0..dw * dh)
        .into_par_iter()
        .map(|i| {
            let (x0, y0) = ((i % dw) * f, (i / dw) * f);
            // The strongest value in the block, so thin rings stay visible when reduced.
            let v = (0..f).flat_map(|dy| (0..f).map(move |dx| plane[(y0 + dy) * w + x0 + dx])).fold(0.0f32, f32::max);
            let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            Color32::from_rgba_unmultiplied(colour[0], colour[1], colour[2], (t * 200.0) as u8)
        })
        .collect();
    ColorImage::new([dw, dh], pixels)
}

/// Where the smart sharpening holds back: "Segui le zone" (blue), the limb protection
/// (orange), the sky it leaves out (grey); the strongest at each pixel.
fn hold_back_overlay(zones: &[f32], edges: &[f32], excluded: Option<&[f32]>, w: usize, h: usize, max_side: usize) -> ColorImage {
    let f = display_factor(w, h, max_side);
    let (dw, dh) = (w / f, h / f);
    let pixels = (0..dw * dh)
        .into_par_iter()
        .map(|i| {
            let k = (i / dw) * f * w + (i % dw) * f;
            let sky = excluded.map_or(0.0, |e| e[k].clamp(0.0, 1.0));
            let (z, e) = (zones[k].clamp(0.0, 1.0) * (1.0 - sky), edges[k].clamp(0.0, 1.0) * (1.0 - sky));
            let (c, a) = if sky >= z.max(e) {
                (crate::panels::SKY_COLOUR, sky)
            } else if e >= z {
                (crate::panels::EDGE_COLOUR, e)
            } else {
                (crate::panels::ZONES_COLOUR, z)
            };
            Color32::from_rgba_unmultiplied(c[0], c[1], c[2], (a * 150.0) as u8)
        })
        .collect();
    ColorImage::new([dw, dh], pixels)
}

/// The periods (px) the resolution colours span: most of the subject's, at least a factor 1.6
/// apart so small differences don't look large.
fn resolution_range(periods: &[f32]) -> (f32, f32) {
    let mut found: Vec<f32> = periods.iter().step_by(periods.len().div_ceil(1 << 16).max(1)).copied().filter(|&p| p > 0.0).collect();
    if found.len() < 16 {
        return (3.0, 25.0);
    }
    found.sort_by(f32::total_cmp);
    let (lo, hi) = (found[found.len() * 5 / 100], found[found.len() * 95 / 100]);
    let spread = (hi / lo).max(1.6);
    let centre = (lo * hi).sqrt();
    (centre / spread.sqrt(), centre * spread.sqrt())
}

/// The finest detail present, coloured over `range`; nothing off the subject, dark grey where
/// it holds no detail.
fn resolution_overlay(periods: &[f32], w: usize, h: usize, max_side: usize, range: (f32, f32)) -> ColorImage {
    let f = display_factor(w, h, max_side);
    let (dw, dh) = (w / f, h / f);
    let pixels = (0..dw * dh)
        .into_par_iter()
        .map(|i| {
            let p = periods[(i / dw) * f * w + (i % dw) * f];
            if p < 0.0 {
                Color32::TRANSPARENT
            } else {
                let c = crate::panels::resolution_colour(p, range).unwrap_or(Color32::from_rgb(40, 40, 46));
                Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 200)
            }
        })
        .collect();
    ColorImage::new([dw, dh], pixels)
}

/// Pixels burnt to white (red) and detail crushed to black where the unprocessed image had
/// some (blue); and how many.
fn clip_overlay(image: &Image, before: &Image, max_side: usize) -> (ColorImage, usize) {
    let (w, h) = (image.width(), image.height());
    let lum = |img: &Image, i: usize| -> f32 {
        match img.channels() {
            [r, g, b] => ps_core::color::luma([r[i], g[i], b[i]]),
            [m] => m[i],
            _ => 0.0,
        }
    };
    let comparable = (before.width(), before.height()) == (w, h);
    // 0 = fine, 1 = white, 2 = black.
    let state: Vec<u8> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            if image.channels().iter().any(|c| c[i] >= 0.998) {
                1
            } else if comparable && lum(image, i) <= 0.002 && lum(before, i) >= 0.02 {
                2
            } else {
                0
            }
        })
        .collect();
    let count = state.par_iter().filter(|s| **s != 0).count();
    let f = display_factor(w, h, max_side);
    let (dw, dh) = (w / f, h / f);
    let pixels = (0..dw * dh)
        .into_par_iter()
        .map(|i| {
            let (x0, y0) = ((i % dw) * f, (i / dw) * f);
            let worst = (0..f).flat_map(|dy| (0..f).map(move |dx| (y0 + dy) * w + x0 + dx)).map(|j| state[j]).max().unwrap_or(0);
            match worst {
                1 => Color32::from_rgba_unmultiplied(255, 40, 40, 230),
                2 => Color32::from_rgba_unmultiplied(40, 120, 255, 230),
                _ => Color32::TRANSPARENT,
            }
        })
        .collect();
    (ColorImage::new([dw, dh], pixels), count)
}

/// A mask shrunk to [`THUMB_SIDE`], white where it applies, and whether it selects (almost)
/// nothing. Framed on what it selects (with some room around), so a small planet's mask is
/// still readable; each thumbnail pixel takes the strongest value under it, so thin rings and
/// small moons don't vanish.
fn mask_thumb(plane: &[f32], w: usize, h: usize) -> (Arc<ColorImage>, bool) {
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if plane[y * w + x] > 0.05 {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    if x1 <= x0 {
        let pixels = vec![Color32::from_gray(20); THUMB_SIDE * THUMB_SIDE];
        return (Arc::new(ColorImage::new([THUMB_SIDE, THUMB_SIDE], pixels)), true);
    }
    // A square frame around what is selected, 30 % larger, kept inside the image.
    let side = (((x1 - x0).max(y1 - y0) as f32 * 1.3) as usize).clamp(1, w.max(h));
    let centre = |a: usize, b: usize, len: usize| ((a + b) / 2).saturating_sub(side / 2).min(len.saturating_sub(side.min(len)));
    let (fx, fy) = (centre(x0, x1, w), centre(y0, y1, h));
    let (fw, fh) = (side.min(w - fx), side.min(h - fy));
    let f = fw.max(fh).div_ceil(THUMB_SIDE).max(1);
    let (dw, dh) = ((fw / f).max(1), (fh / f).max(1));
    let pixels = (0..dw * dh)
        .map(|i| {
            let (bx, by) = (fx + (i % dw) * f, fy + (i / dw) * f);
            let mut m = 0.0f32;
            for y in by..(by + f).min(h) {
                for x in bx..(bx + f).min(w) {
                    m = m.max(plane[y * w + x]);
                }
            }
            Color32::from_gray((m.clamp(0.0, 1.0) * 235.0 + 20.0) as u8)
        })
        .collect();
    (Arc::new(ColorImage::new([dw, dh], pixels)), false)
}

/// Semi-transparent tint showing where a mask applies.
fn mask_overlay(plane: &[f32], w: usize, h: usize, max_side: usize) -> ColorImage {
    let f = display_factor(w, h, max_side);
    let (dw, dh) = (w / f, h / f);
    let pixels = (0..dw * dh)
        .map(|i| {
            let (x, y) = ((i % dw) * f, (i / dw) * f);
            let m = plane[y * w + x].clamp(0.0, 1.0);
            Color32::from_rgba_unmultiplied(255, 70, 50, (m * 120.0) as u8)
        })
        .collect();
    ColorImage::new([dw, dh], pixels)
}
