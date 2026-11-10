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

type Mat = [[i32; 2]; 2];

fn mul(a: Mat, b: Mat) -> Mat {
    let mut r = [[0; 2]; 2];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    r
}

/// Clockwise quarter turn on image coordinates (x right, y down).
const TURN: Mat = [[0, -1], [1, 0]];

/// The linear part of quarter turns then flips (what [`apply`] does before the fine rotation).
fn lossless_matrix(quarter_turns: u8, flip_h: bool, flip_v: bool) -> Mat {
    let mut m = [[1, 0], [0, 1]];
    for _ in 0..quarter_turns % 4 {
        m = mul(TURN, m);
    }
    mul([[if flip_h { -1 } else { 1 }, 0], [0, if flip_v { -1 } else { 1 }]], m)
}

impl Geometry {
    /// This orientation followed by an optional left-right mirror of the result and a clockwise
    /// rotation of `degrees`: the whole quarter-turn part goes into the lossless fields, only the
    /// remainder (at most 45°) into the fine angle, so a large turn doesn't cut the corners.
    pub fn then(&self, mirror: bool, degrees: f32) -> Geometry {
        let quarters = (degrees / 90.0).round();
        let fine = degrees - 90.0 * quarters;
        let mut d: Mat = [[1, 0], [0, 1]];
        if mirror {
            d = [[-1, 0], [0, 1]];
        }
        for _ in 0..(quarters as i32).rem_euclid(4) {
            d = mul(TURN, d);
        }
        let target = mul(d, lossless_matrix(self.quarter_turns, self.flip_horizontal, self.flip_vertical));
        let (q, h, v) = (0..4u8)
            .flat_map(|q| [false, true].into_iter().flat_map(move |h| [false, true].into_iter().map(move |v| (q, h, v))))
            .find(|&(q, h, v)| lossless_matrix(q, h, v) == target)
            .expect("the dihedral group is closed");
        // A mirror reverses the sense of the rotation applied before it.
        let before = if mirror { -self.angle } else { self.angle };
        let angle = (fine + before + 180.0).rem_euclid(360.0) - 180.0;
        Geometry { quarter_turns: q, flip_horizontal: h, flip_vertical: v, angle }
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

    /// `then` equals doing the mirror and the rotation on the already-oriented image.
    #[test]
    fn then_matches_doing_it_afterwards() {
        let img = Image::new(5, 4, vec![(0..20).map(|v| v as f32).collect()], 16);
        for g in [Geometry::default(), Geometry { quarter_turns: 1, flip_horizontal: true, flip_vertical: false, angle: 0.0 }, Geometry { quarter_turns: 3, flip_horizontal: false, flip_vertical: true, angle: 0.0 }] {
            for mirror in [false, true] {
                for deg in [0.0f32, 90.0, 180.0, -90.0, 270.0] {
                    let direct = apply(&img, &g.then(mirror, deg));
                    let mut step = apply(&img, &g);
                    if mirror {
                        step = flip(&step, true, false);
                    }
                    for _ in 0..((deg / 90.0).round() as i32).rem_euclid(4) {
                        step = rotate90(&step);
                    }
                    assert_eq!(direct.channels()[0], step.channels()[0], "{g:?} mirror {mirror} {deg}");
                }
            }
        }
        // The fine part stays in the angle and survives a mirror with its sign reversed.
        let t = Geometry { angle: 10.0, ..Default::default() }.then(true, 30.0);
        assert!((t.angle - 20.0).abs() < 1e-4 && t.flip_horizontal, "{t:?}");
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
