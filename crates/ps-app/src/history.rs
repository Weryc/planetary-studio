//! Undo/redo of pipeline edits. A slider drag becomes one entry: changes are committed only
//! once the pointer is released.

use ps_core::pipeline::{Pipeline, Stage};
use std::collections::VecDeque;

const LIMIT: usize = 200;

pub struct History {
    undo: VecDeque<Pipeline>,
    redo: Vec<Pipeline>,
    committed: Pipeline,
}

impl History {
    pub fn new(pipeline: &Pipeline) -> Self {
        Self { undo: VecDeque::new(), redo: Vec::new(), committed: pipeline.clone() }
    }

    /// Call every frame.
    pub fn track(&mut self, current: &Pipeline, interacting: bool) {
        if !interacting && *current != self.committed {
            self.undo.push_back(std::mem::replace(&mut self.committed, current.clone()));
            if self.undo.len() > LIMIT {
                self.undo.pop_front();
            }
            self.redo.clear();
        }
    }

    /// What the next undo takes back, in words ("Livelli tonali", "Più regolazioni").
    pub fn undo_name(&self) -> Option<String> {
        self.undo.back().map(|before| describe(before, &self.committed))
    }

    /// What the next redo puts back.
    pub fn redo_name(&self) -> Option<String> {
        self.redo.last().map(|after| describe(&self.committed, after))
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self, current: &mut Pipeline) {
        if let Some(prev) = self.undo.pop_back() {
            self.redo.push(std::mem::replace(&mut self.committed, prev.clone()));
            *current = prev;
        }
    }

    pub fn redo(&mut self, current: &mut Pipeline) {
        if let Some(next) = self.redo.pop() {
            self.undo.push_back(std::mem::replace(&mut self.committed, next.clone()));
            *current = next;
        }
    }
}

/// The module that differs between `a` and `b`, or "several adjustments".
fn describe(a: &Pipeline, b: &Pipeline) -> String {
    let changed: Vec<Stage> = [
        (Stage::Align, a.align != b.align),
        (Stage::Bxt, a.bxt != b.bxt),
        (Stage::Deconv, a.deconv != b.deconv),
        (Stage::Denoise, a.denoise != b.denoise),
        (Stage::Wavelet, a.wavelet != b.wavelet),
        (Stage::Smart, a.smart != b.smart),
        (Stage::Contrast, a.contrast != b.contrast),
        (Stage::Dering, a.dering != b.dering),
        (Stage::Tone, a.tone != b.tone),
        (Stage::Curves, a.curves != b.curves),
        (Stage::PhotoColor, a.photocolor != b.photocolor),
        (Stage::Color, a.color != b.color),
        (Stage::Layers, a.layers != b.layers),
        (Stage::Mineral, a.mineral != b.mineral),
        (Stage::SunColor, a.suncolor != b.suncolor),
    ]
    .into_iter()
    .filter_map(|(stage, differs)| differs.then_some(stage))
    .collect();
    match changed.as_slice() {
        [one] => one.name().to_string(),
        _ => ps_core::tr!("Più regolazioni", "Several adjustments"),
    }
}
