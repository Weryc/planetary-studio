//! Undo/redo of pipeline edits. A slider drag becomes one entry: changes are committed only
//! once the pointer is released.

use ps_core::pipeline::Pipeline;

const LIMIT: usize = 200;

pub struct History {
    undo: Vec<Pipeline>,
    redo: Vec<Pipeline>,
    committed: Pipeline,
}

impl History {
    pub fn new(pipeline: &Pipeline) -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), committed: pipeline.clone() }
    }

    /// Call every frame.
    pub fn track(&mut self, current: &Pipeline, interacting: bool) {
        if !interacting && *current != self.committed {
            self.undo.push(std::mem::replace(&mut self.committed, current.clone()));
            if self.undo.len() > LIMIT {
                self.undo.remove(0);
            }
            self.redo.clear();
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self, current: &mut Pipeline) {
        if let Some(prev) = self.undo.pop() {
            self.redo.push(std::mem::replace(&mut self.committed, prev.clone()));
            *current = prev;
        }
    }

    pub fn redo(&mut self, current: &mut Pipeline) {
        if let Some(next) = self.redo.pop() {
            self.undo.push(std::mem::replace(&mut self.committed, next.clone()));
            *current = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drag_is_one_undo_step() {
        let mut p = Pipeline::default();
        let mut h = History::new(&p);
        for gain in [0.1, 0.2, 0.3] {
            p.wavelet.params.layers[0].gain = gain;
            h.track(&p, true);
        }
        h.track(&p, false);
        h.undo(&mut p);
        assert_eq!(p, Pipeline::default());
        h.redo(&mut p);
        assert_eq!(p.wavelet.params.layers[0].gain, 0.3);
    }
}
