//! The learned sharpening under the three hypotheses on the night's seeing (calm, average,
//! turbulent), side by side on a small crop of the subject, so the one that looks right can be
//! picked. Made off the UI thread, on the image as loaded (the comparison is between the
//! hypotheses: what earlier steps did changes them alike).

use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use ps_core::Image;
use ps_core::analysis::Analysis;
use ps_core::learned::Night;
use ps_core::smart::SmartParams;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// Side of the crop compared (px of the image).
const SIDE: usize = 128;

/// The thumbnails being made, for the key they were started with.
type Pending = (u64, Receiver<Vec<(Night, ColorImage)>>);

#[derive(Default)]
pub struct SmartCompare {
    key: u64,
    pending: Option<Pending>,
    thumbs: Vec<(Night, TextureHandle)>,
}

impl SmartCompare {
    /// The thumbnails for the current settings, once made (empty while they are being made or
    /// when the hypotheses don't differ enough to be worth comparing).
    pub fn thumbs(&self) -> &[(Night, TextureHandle)] {
        &self.thumbs
    }

    /// Whether thumbnails are being made.
    pub fn busy(&self) -> bool {
        self.pending.is_some()
    }

    /// Keep the thumbnails in step with the image and the settings: started again when what
    /// they depend on changes, picked up when ready.
    pub fn update(&mut self, ctx: &egui::Context, source: &Arc<Image>, source_gen: u64, p: &SmartParams, analysis: &Analysis) {
        if let Some((key, rx)) = &self.pending
            && let Ok(images) = rx.try_recv()
        {
            if *key == self.key {
                self.thumbs = images.into_iter().map(|(n, img)| (n, ctx.load_texture(format!("smart-night-{n:?}"), img, TextureOptions::LINEAR))).collect();
            }
            self.pending = None;
        }
        let wanted = p.curve == ps_core::smart::SmartCurve::Learned && ps_core::learned::nights_differ(&p.learned_nights);
        let key = if wanted { key_of(source_gen, p) } else { 0 };
        if key == self.key {
            return;
        }
        self.key = key;
        self.thumbs.clear();
        if !wanted {
            self.pending = None;
            return;
        }
        let (tx, rx) = channel();
        let (source, p, centre) = (Arc::clone(source), p.clone(), centre(source, analysis));
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(make(&source, &p, centre));
            ctx.request_repaint();
        });
        self.pending = Some((key, rx));
    }
}

/// What the thumbnails depend on: the image and every setting but the chosen night and the
/// strength (each hypothesis brings its own).
fn key_of(source_gen: u64, p: &SmartParams) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    source_gen.hash(&mut h);
    for v in [p.cutoff, p.peak, p.limb_protection] {
        v.to_bits().hash(&mut h);
    }
    (p.adaptive, p.include_sky, p.protect_highlights, p.color as u8).hash(&mut h);
    for (n, s, t) in &p.learned_nights {
        (*n as u8, s.to_bits()).hash(&mut h);
        t.iter().for_each(|q| (q.0.to_bits(), q.1.to_bits()).hash(&mut h));
    }
    h.finish().max(1)
}

/// Middle of the subject: the disk's centre, else the image's.
pub(crate) fn centre(img: &Image, a: &Analysis) -> (usize, usize) {
    a.disk.map_or((img.width() / 2, img.height() / 2), |d| (d.cx.max(0.0) as usize, d.cy.max(0.0) as usize))
}

pub(crate) fn make(source: &Image, p: &SmartParams, (cx, cy): (usize, usize)) -> Vec<(Night, ColorImage)> {
    let (w, h) = (source.width(), source.height());
    let side = SIDE.min(w).min(h);
    let x0 = cx.saturating_sub(side / 2).min(w - side);
    let y0 = cy.saturating_sub(side / 2).min(h - side);
    let crop = source.crop(x0, y0, side, side);
    let subject = (!p.include_sky).then(|| ps_core::mask::subject_and_moons(&crop));
    p.learned_nights
        .iter()
        .map(|(n, strength, _)| {
            let params = SmartParams { night: Some(*n), strength: *strength, ..p.clone() };
            let out = ps_core::smart::sharpen(&crop, None, &params, subject.as_deref());
            (*n, crate::worker::display_image(&out, side))
        })
        .collect()
}
