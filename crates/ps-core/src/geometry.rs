//! Orientation: quarter turns and mirroring (lossless) and fine rotation (bilinear), applied
//! before cropping.

use crate::Image;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    /// Clockwise quarter turns (0‥3).
    pub quarter_turns: u8,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    /// Extra clockwise rotation in degrees, to straighten the image.
    pub angle: f32,
}

impl Geometry {
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    pub fn rotate_clockwise(&mut self) {
        self.quarter_turns = (self.quarter_turns + 1) % 4;
    }

    pub fn rotate_counterclockwise(&mut self) {
        self.quarter_turns = (self.quarter_turns + 3) % 4;
    }
}

pub fn apply(img: &Image, g: &Geometry) -> Image {
    let mut out = img.clone();
    for _ in 0..g.quarter_turns % 4 {
        out = rotate90(&out);
    }
    if g.flip_horizontal || g.flip_vertical {
        out = flip(&out, g.flip_horizontal, g.flip_vertical);
    }
    if g.angle != 0.0 {
        out = rotate(&out, g.angle);
    }
    out
}

/// Quarter turn clockwise.
fn rotate90(img: &Image) -> Image {
    let (w, h) = (img.width(), img.height());
    let channels = img
        .channels()
        .par_iter()
        .map(|c| {
            // New size h×w: out(x, y) = in(y, h − 1 − x).
            let mut out = vec![0.0f32; w * h];
            for y in 0..w {
                for x in 0..h {
                    out[y * h + x] = c[(h - 1 - x) * w + y];
                }
            }
            out
        })
        .collect();
    Image::new(h, w, channels, img.source_bit_depth)
}

fn flip(img: &Image, horizontal: bool, vertical: bool) -> Image {
    let (w, h) = (img.width(), img.height());
    let channels = img
        .channels()
        .par_iter()
        .map(|c| {
            (0..h)
                .flat_map(|y| {
                    let sy = if vertical { h - 1 - y } else { y };
                    (0..w).map(move |x| (sy, if horizontal { w - 1 - x } else { x }))
                })
                .map(|(y, x)| c[y * w + x])
                .collect()
        })
        .collect();
    Image::new(w, h, channels, img.source_bit_depth)
}

/// Rotate about the centre (clockwise, degrees), keeping the size; uncovered corners are black.
fn rotate(img: &Image, degrees: f32) -> Image {
    let (w, h) = (img.width(), img.height());
    let (s, c) = degrees.to_radians().sin_cos();
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let channels = img
        .channels()
        .par_iter()
        .map(|plane| {
            let mut out = vec![0.0f32; w * h];
            out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                for (x, o) in row.iter_mut().enumerate() {
                    // Inverse mapping: rotate the output position back into the source.
                    let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                    let sx = c * dx + s * dy + cx - 0.5;
                    let sy = -s * dx + c * dy + cy - 0.5;
                    if sx < 0.0 || sy < 0.0 || sx > (w - 1) as f32 || sy > (h - 1) as f32 {
                        continue;
                    }
                    let (x0, y0) = (sx as usize, sy as usize);
                    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                    let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
                    let p = |xx: usize, yy: usize| plane[yy * w + xx];
                    *o = p(x0, y0) * (1.0 - fx) * (1.0 - fy) + p(x1, y0) * fx * (1.0 - fy) + p(x0, y1) * (1.0 - fx) * fy + p(x1, y1) * fx * fy;
                }
            });
            out
        })
        .collect();
    Image::new(w, h, channels, img.source_bit_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3×2 image numbered 0‥5 row by row.
    fn numbered() -> Image {
        Image::new(3, 2, vec![(0..6).map(|v| v as f32).collect()], 16)
    }

    #[test]
    fn quarter_turn_clockwise() {
        let g = Geometry { quarter_turns: 1, ..Default::default() };
        let out = apply(&numbered(), &g);
        // 0 1 2      3 0
        // 3 4 5  →   4 1
        //            5 2
        assert_eq!((out.width(), out.height()), (2, 3));
        assert_eq!(out.channels()[0], vec![3.0, 0.0, 4.0, 1.0, 5.0, 2.0]);
    }

    #[test]
    fn four_turns_and_double_flips_are_identity() {
        let img = numbered();
        let mut g = Geometry::default();
        for _ in 0..4 {
            g.rotate_clockwise();
        }
        assert_eq!(apply(&img, &g).channels(), img.channels());
        let flipped = apply(&apply(&img, &Geometry { flip_horizontal: true, ..Default::default() }), &Geometry { flip_horizontal: true, ..Default::default() });
        assert_eq!(flipped.channels(), img.channels());
    }

    #[test]
    fn small_angle_keeps_the_centre() {
        let (w, h) = (41, 41);
        let mut plane = vec![0.0f32; w * h];
        plane[20 * w + 20] = 1.0;
        let out = apply(&Image::new(w, h, vec![plane], 16), &Geometry { angle: 7.0, ..Default::default() });
        assert!(out.channels()[0][20 * w + 20] > 0.9);
    }
}
