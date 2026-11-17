//! Stack a real planetary recording (SER) as AutoStakkert does, three times at once: all the
//! kept frames, the even ones and the odd ones. Half the difference of the two half stacks is
//! the full stack's random noise (the planet is the same in both, the noise is not): it is what
//! the noise estimator is calibrated against on real data.
//!
//! `cargo run --release -p ps-train --bin ser_stack -- file.ser out_dir [--keep 0.5] [--label x]`
//! → out_dir/<label>_k<keep>/{piena,pari,dispari}.tif (float32, grey-world balanced on the
//! disk), anteprima.png, meta.json.
//!
//! Steps: every frame graded (gradient energy of the lightly smoothed luminance on the lit
//! part, from 2×2 bins) and its planet located (centre of light); the best `keep` share kept
//! and cut around the planet; a reference from the best 10 % aligned on the best; each kept
//! frame aligned on it globally then on a grid of alignment points blended into a smooth warp,
//! resampled (Catmull–Rom) and added to the full stack and to its parity's half stack.
#[path = "sim_stack/tiff.rs"]
mod tiff;

use ps_core::align::{measure_shift, shift};
use rayon::prelude::*;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

const AP: usize = 64;
const AP_MARGIN: usize = 16;
const AP_SPACING: usize = 40;
const STEP: usize = 8;

struct Ser {
    file: std::fs::File,
    w: usize,
    h: usize,
    colour: i32,
    bytes: usize,
    frames: usize,
    big_endian: bool,
}

fn le_i32(b: &[u8], at: usize) -> i32 {
    i32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

impl Ser {
    fn open(path: &Path) -> Ser {
        let file = std::fs::File::open(path).unwrap();
        let mut head = [0u8; 178];
        file.read_exact_at(&mut head, 0).unwrap();
        assert_eq!(&head[0..14], b"LUCAM-RECORDER", "non è un file SER");
        let (colour, w, h, depth, frames) = (le_i32(&head, 18), le_i32(&head, 26) as usize, le_i32(&head, 30) as usize, le_i32(&head, 34), le_i32(&head, 38) as usize);
        let bytes = if depth > 8 { 2 } else { 1 };
        let mut s = Ser { file, w, h, colour, bytes, frames, big_endian: false };
        // The header's endianness flag is unreliable (writers disagree on its meaning): the
        // order that gives the smoother picture wins.
        if bytes == 2 {
            let raw = s.read_raw(frames / 2);
            let rough = |big: bool| -> f64 {
                let v: Vec<f64> = raw.chunks(2).map(|c| if big { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) } as f64).collect();
                v.windows(2).step_by(7).map(|p| (p[0] - p[1]).abs()).sum::<f64>() / v.iter().sum::<f64>().max(1.0)
            };
            s.big_endian = rough(true) < rough(false);
        }
        s
    }

    fn planes_per_pixel(&self) -> usize {
        if self.colour >= 100 { 3 } else { 1 }
    }

    fn bayer(&self) -> Option<(usize, usize)> {
        // Where red sits in the 2×2 cell: RGGB, GRBG, GBRG, BGGR.
        match self.colour {
            8 => Some((0, 0)),
            9 => Some((1, 0)),
            10 => Some((0, 1)),
            11 => Some((1, 1)),
            _ => None,
        }
    }

    fn read_raw(&self, i: usize) -> Vec<u8> {
        let n = self.w * self.h * self.planes_per_pixel() * self.bytes;
        let mut buf = vec![0u8; n];
        self.file.read_exact_at(&mut buf, 178 + (i * n) as u64).unwrap();
        buf
    }

    /// The frame as values 0‥1, one plane per sample (the mosaic for Bayer data).
    fn samples(&self, i: usize) -> Vec<f32> {
        let raw = self.read_raw(i);
        if self.bytes == 1 {
            raw.iter().map(|&v| v as f32 / 255.0).collect()
        } else {
            raw.chunks(2).map(|c| if self.big_endian { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) } as f32 / 65535.0).collect()
        }
    }
}

/// Bilinear demosaic of a window of the mosaic; `(rx, ry)` where red sits (window origins are
/// even, so the pattern is the frame's).
fn debayer(m: &[f32], w: usize, h: usize, (rx, ry): (usize, usize)) -> Vec<Vec<f32>> {
    let at = |x: isize, y: isize| m[ps_core::conv::reflect(y, h) * w + ps_core::conv::reflect(x, w)];
    let mut planes = vec![vec![0.0f32; w * h]; 3];
    for y in 0..h {
        for x in 0..w {
            let (xi, yi) = (x as isize, y as isize);
            let i = y * w + x;
            let here = at(xi, yi);
            let cross = (at(xi - 1, yi) + at(xi + 1, yi) + at(xi, yi - 1) + at(xi, yi + 1)) / 4.0;
            let diag = (at(xi - 1, yi - 1) + at(xi + 1, yi - 1) + at(xi - 1, yi + 1) + at(xi + 1, yi + 1)) / 4.0;
            let horiz = (at(xi - 1, yi) + at(xi + 1, yi)) / 2.0;
            let vert = (at(xi, yi - 1) + at(xi, yi + 1)) / 2.0;
            let (r, g, b) = match ((y + ry) % 2, (x + rx) % 2) {
                (0, 0) => (here, cross, diag),
                (1, 1) => (diag, cross, here),
                (0, 1) => (horiz, here, vert),
                _ => (vert, here, horiz),
            };
            planes[0][i] = r;
            planes[1][i] = g;
            planes[2][i] = b;
        }
    }
    planes
}

/// Luminance in 2×2 bins (one Bayer cell: colour doesn't matter) of a whole frame.
fn binned(s: &Ser, v: &[f32]) -> (Vec<f32>, usize, usize) {
    let (w, h) = (s.w / 2, s.h / 2);
    let p = s.planes_per_pixel();
    let at = |x: usize, y: usize| -> f32 { (0..p).map(|c| v[(y * s.w + x) * p + c]).sum::<f32>() / p as f32 };
    ((0..w * h).map(|i| (at(2 * (i % w), 2 * (i / w)) + at(2 * (i % w) + 1, 2 * (i / w)) + at(2 * (i % w), 2 * (i / w) + 1) + at(2 * (i % w) + 1, 2 * (i / w) + 1)) / 4.0).collect(), w, h)
}

/// Sharpness and centre of light of a frame (from its 2×2 bins).
fn grade(s: &Ser, i: usize) -> (f32, f32, f32) {
    let (l, w, h) = binned(s, &s.samples(i));
    let smooth = ps_core::mask::gaussian_approx(&l, w, h, 1.2);
    let top = smooth.iter().copied().fold(0.0f32, f32::max).max(1e-9);
    let (mut q, mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let k = y * w + x;
            if smooth[k] > 0.15 * top {
                let (gx, gy) = (smooth[k + 1] - smooth[k - 1], smooth[k + w] - smooth[k - w]);
                q += (gx * gx + gy * gy) as f64;
                sx += smooth[k] as f64 * x as f64;
                sy += smooth[k] as f64 * y as f64;
                sw += smooth[k] as f64;
            }
        }
    }
    ((q / (top as f64).powi(2)) as f32, (2.0 * sx / sw.max(1e-9)) as f32, (2.0 * sy / sw.max(1e-9)) as f32)
}

/// The frame's colour planes in a `side`² window centred (to the even pixel) on `(cx, cy)`.
fn window(s: &Ser, i: usize, cx: f32, cy: f32, side: usize) -> Vec<Vec<f32>> {
    let v = s.samples(i);
    let x0 = (((cx - side as f32 / 2.0).round().max(0.0) as usize).min(s.w - side)) & !1;
    let y0 = (((cy - side as f32 / 2.0).round().max(0.0) as usize).min(s.h - side)) & !1;
    let p = s.planes_per_pixel();
    if let Some(bayer) = s.bayer() {
        let m: Vec<f32> = (0..side * side).map(|k| v[(y0 + k / side) * s.w + x0 + k % side]).collect();
        return debayer(&m, side, side, bayer);
    }
    (0..p)
        .map(|c| {
            // RGB SERs may store BGR (colour id 101).
            let c = if s.colour == 101 { 2 - c } else { c };
            (0..side * side).map(|k| v[((y0 + k / side) * s.w + x0 + k % side) * p + c]).collect()
        })
        .collect()
}

fn luminance(planes: &[Vec<f32>]) -> Vec<f32> {
    if planes.len() == 1 {
        return planes[0].clone();
    }
    (0..planes[0].len()).map(|i| 0.2126 * planes[0][i] + 0.7152 * planes[1][i] + 0.0722 * planes[2][i]).collect()
}

fn cubic(t: f32) -> [f32; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [0.5 * (-t3 + 2.0 * t2 - t), 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0), 0.5 * (-3.0 * t3 + 4.0 * t2 + t), 0.5 * (t3 - t2)]
}

fn sample(p: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (wx, wy) = (cubic(x - x0), cubic(y - y0));
    let mut acc = 0.0;
    for (j, wyj) in wy.iter().enumerate() {
        let sy = (y0 as isize + j as isize - 1).clamp(0, h as isize - 1) as usize;
        let row = &p[sy * w..(sy + 1) * w];
        let mut r = 0.0;
        for (i, wxi) in wx.iter().enumerate() {
            r += wxi * row[(x0 as isize + i as isize - 1).clamp(0, w as isize - 1) as usize];
        }
        acc += wyj * r;
    }
    acc
}

fn patch(p: &[f32], w: usize, cx: usize, cy: usize, side: usize) -> Vec<f32> {
    let (x0, y0) = (cx - side / 2, cy - side / 2);
    (0..side).flat_map(|y| p[(y0 + y) * w + x0..(y0 + y) * w + x0 + side].iter().copied()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (ser_path, out) = (PathBuf::from(&args[0]), PathBuf::from(&args[1]));
    let opt = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let keep: f32 = opt("--keep").map_or(0.5, |v| v.parse().unwrap());
    let label = opt("--label").unwrap_or_else(|| ser_path.file_stem().unwrap().to_string_lossy().to_string());
    let s = Ser::open(&ser_path);
    eprintln!("{}: {}×{}, {} fotogrammi, colore {}, {} bit{}", ser_path.display(), s.w, s.h, s.frames, s.colour, s.bytes * 8, if s.big_endian { " (big endian)" } else { "" });

    // 1. Grade every frame.
    let grades: Vec<(f32, f32, f32)> = (0..s.frames).into_par_iter().map(|i| grade(&s, i)).collect();
    let mut order: Vec<usize> = (0..s.frames).collect();
    order.sort_by(|&a, &b| grades[b].0.total_cmp(&grades[a].0));
    let kept: Vec<usize> = order[..((keep * s.frames as f32) as usize).max(10)].to_vec();

    // 2. The planet's size, from the best frame, sets the window.
    let probe = window(&s, kept[0], grades[kept[0]].1, grades[kept[0]].2, (s.w.min(s.h)) & !1);
    let pw = s.w.min(s.h) & !1;
    let a = ps_core::analysis::analyze(&ps_core::Image::new(pw, pw, probe, 16));
    let diameter = a.disk.map_or(0.5 * pw as f32, |d| 2.0 * d.radius);
    let side = (ps_core::fft::next_fast_size(((1.6 * diameter) as usize).max(256)).min(s.w.min(s.h))) & !1;
    eprintln!("disco {diameter:.0} px, finestra {side}², tenuti {} su {}", kept.len(), s.frames);
    let frame = |i: usize| window(&s, i, grades[i].1, grades[i].2, side);

    // 3. Reference: the best 10 % aligned on the best.
    let best = luminance(&frame(kept[0]));
    let n_ref = (kept.len() / 10).clamp(5, kept.len());
    let refs: Vec<Vec<f32>> = kept[..n_ref].par_iter().map(|&i| {
        let l = luminance(&frame(i));
        let d = measure_shift(&best, &l, side, side);
        shift(&l, side, side, d)
    }).collect();
    let reference: Vec<f32> = (0..side * side).map(|k| refs.iter().map(|r| r[k]).sum::<f32>() / n_ref as f32).collect();
    let top = reference.iter().copied().fold(0.0f32, f32::max);
    let outer = AP + 2 * AP_MARGIN;
    let mut points = Vec::new();
    for cy in (outer / 2..side - outer / 2).step_by(AP_SPACING) {
        for cx in (outer / 2..side - outer / 2).step_by(AP_SPACING) {
            let p = patch(&reference, side, cx, cy, AP);
            if p.iter().sum::<f32>() / p.len() as f32 > 0.12 * top {
                points.push((cx, cy, patch(&reference, side, cx, cy, outer)));
            }
        }
    }
    let channels = if s.bayer().is_some() || s.planes_per_pixel() == 3 { 3 } else { 1 };
    let (gw, gh) = (side.div_ceil(STEP) + 1, side.div_ceil(STEP) + 1);
    let sigma = 0.8 * AP_SPACING as f32;

    // 4. Align and add: [full, even, odd].
    let zero = || (vec![vec![vec![0.0f32; side * side]; channels]; 3], [0usize; 3]);
    let (sums, counts) = kept
        .par_iter()
        .map(|&i| {
            let planes = frame(i);
            let lum = luminance(&planes);
            let g = measure_shift(&reference, &lum, side, side);
            let lum_g = shift(&lum, side, side, g);
            let local: Vec<(f32, f32, [f32; 2])> = points
                .iter()
                .filter_map(|(cx, cy, rp)| {
                    let d = measure_shift(rp, &patch(&lum_g, side, *cx, *cy, outer), outer, outer);
                    (d[0].hypot(d[1]) < 6.0).then_some((*cx as f32, *cy as f32, d))
                })
                .collect();
            let field: Vec<[f32; 2]> = (0..gw * gh)
                .map(|k| {
                    let (x, y) = (((k % gw) * STEP) as f32, ((k / gw) * STEP) as f32);
                    let (mut sx, mut sy, mut sw) = (0.0f32, 0.0f32, 0.05f32);
                    for (px, py, d) in &local {
                        let r2 = (x - px).powi(2) + (y - py).powi(2);
                        if r2 < 9.0 * sigma * sigma {
                            let wgt = (-r2 / (2.0 * sigma * sigma)).exp();
                            sx += wgt * d[0];
                            sy += wgt * d[1];
                            sw += wgt;
                        }
                    }
                    [g[0] + sx / sw, g[1] + sy / sw]
                })
                .collect();
            let at = |x: usize, y: usize| -> [f32; 2] {
                let (fx, fy) = (x as f32 / STEP as f32, y as f32 / STEP as f32);
                let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
                let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
                let f = |xx: usize, yy: usize| field[yy.min(gh - 1) * gw + xx.min(gw - 1)];
                let (a, b, c, d) = (f(x0, y0), f(x0 + 1, y0), f(x0, y0 + 1), f(x0 + 1, y0 + 1));
                std::array::from_fn(|k| (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty)
            };
            let aligned: Vec<Vec<f32>> = planes
                .iter()
                .map(|p| (0..side * side).map(|k| { let s = at(k % side, k / side); sample(p, side, side, (k % side) as f32 - s[0], (k / side) as f32 - s[1]) }).collect())
                .collect();
            (aligned, i % 2)
        })
        .fold(zero, |mut acc, (aligned, parity)| {
            for slot in [0, 1 + parity] {
                for (s, p) in acc.0[slot].iter_mut().zip(&aligned) {
                    s.iter_mut().zip(p).for_each(|(a, v)| *a += v);
                }
                acc.1[slot] += 1;
            }
            acc
        })
        .reduce(zero, |mut a, b| {
            for slot in 0..3 {
                for (x, y) in a.0[slot].iter_mut().zip(&b.0[slot]) {
                    x.iter_mut().zip(y).for_each(|(s, v)| *s += v);
                }
                a.1[slot] += b.1[slot];
            }
            a
        });
    let mut stacks: Vec<Vec<Vec<f32>>> = sums.into_iter().zip(counts).map(|(p, n)| p.into_iter().map(|c| c.into_iter().map(|v| v / n.max(1) as f32).collect()).collect()).collect();
    // Grey-world balance on the disk (the full stack's), the same for the three.
    if channels == 3 {
        let l = luminance(&stacks[0]);
        let topl = l.iter().copied().fold(0.0f32, f32::max);
        let lit: Vec<usize> = (0..l.len()).filter(|&k| l[k] > 0.3 * topl).collect();
        let mean = |c: &Vec<f32>| lit.iter().map(|&k| c[k] as f64).sum::<f64>() / lit.len() as f64;
        let m: Vec<f64> = stacks[0].iter().map(mean).collect();
        let gains: Vec<f32> = m.iter().map(|v| (m[1] / v.max(1e-12)) as f32).collect();
        for st in stacks.iter_mut() {
            for (c, g) in st.iter_mut().zip(&gains) {
                c.iter_mut().for_each(|v| *v *= g);
            }
        }
    }
    let dir = out.join(format!("{label}_k{:02}", (keep * 100.0).round() as u32));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, st) in ["piena", "pari", "dispari"].iter().zip(&stacks) {
        tiff::write_f32(&dir.join(format!("{name}.tif")), side, side, st).unwrap();
    }
    let l = luminance(&stacks[0]);
    let topv = { let mut v = l.clone(); v.sort_by(f32::total_cmp); v[v.len() * 999 / 1000].max(1e-6) };
    let prev = ps_core::Image::new(side, side, stacks[0].iter().map(|c| c.iter().map(|v| (v / topv).clamp(0.0, 1.0).powf(1.0 / 2.2)).collect()).collect(), 16);
    ps_io::save(&dir.join("anteprima.png"), &prev).unwrap();
    std::fs::write(
        dir.join("meta.json"),
        format!(
            "{{\"ser\": \"{}\", \"fotogrammi\": {}, \"tenuti\": {}, \"pari\": {}, \"dispari\": {}, \"quota\": {keep}, \"punti\": {}, \"disco_px\": {diameter:.1}, \"lato\": {side}}}\n",
            ser_path.file_name().unwrap().to_string_lossy(),
            s.frames,
            counts[0],
            counts[1],
            counts[2],
            points.len()
        ),
    )
    .unwrap();
    eprintln!("→ {}: {} fotogrammi ({} pari, {} dispari), {} punti di allineamento", dir.display(), counts[0], counts[1], counts[2], points.len());
}
