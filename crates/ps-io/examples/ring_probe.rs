//! Dering study: the image through the auto pipeline (dering off), then the ring painted over by
//! a simulated brush stroke and lifted.
//! `cargo run --release -p ps-io --example ring_probe -- out_dir image.tif [Gentle|Normal|Strong]`
//! - PUSH=k: wavelet gains ×k (exaggerated sharpening, as the user does to see the ring).
//! - STROKE="x,y;x,y;…" (image px) and BRUSH=radius: the stroke; ARC="depth,from°,to°" instead
//!   paints along the limb of the disk found, `depth` px inside it.
//! - STRONG_SMART=k: smart sharpening at strength k.
//! - CROP="x,y,w,h": also writes 4× enlargements of that part.
//!
//! Writes sharp.png, fixed.png, map.png (lift in magenta, stroke outline in cyan).
use ps_core::Image;
use ps_core::color::{ColorMode, working_planes};
use ps_core::dering::{self, DeringParams};
use ps_core::mask::Stroke;
use ps_core::pipeline::{Aux, Processor, next_generation};
use ps_core::preset::{AutoStrength, auto_pipeline_with};
use ps_core::wavelet::ScaleMode;
use std::path::Path;
use std::sync::Arc;

fn main() {
    let mut args = std::env::args().skip(1);
    let (out, path) = (args.next().unwrap(), args.next().unwrap());
    let strength = match args.next().as_deref() {
        Some("Gentle") => AutoStrength::Gentle,
        Some("Normal") => AutoStrength::Normal,
        _ => AutoStrength::Strong,
    };
    let env = |k: &str| std::env::var(k).ok();
    let img = Arc::new(ps_io::load(path.as_ref()).unwrap());
    let a = ps_core::analysis::analyze(&img);
    let mut p = auto_pipeline_with(a.kind, &a, strength, ScaleMode::Free);
    p.dering.enabled = false;
    p.align.enabled = false;
    if let Some(k) = env("PUSH").and_then(|v| v.parse::<f32>().ok()) {
        p.wavelet.params.layers.iter_mut().for_each(|l| l.gain = l.gain * k + (k - 1.0) * 0.3);
    }
    // STRONG_SMART=k: smart sharpening at strength k (the app's flow that drew the ring).
    if let Some(k) = env("STRONG_SMART").and_then(|v| v.parse::<f32>().ok()) {
        p.smart = ps_core::pipeline::Module::on(ps_core::smart::SmartParams { strength: k, ..p.smart.params.clone() });
    }
    let (w, h) = (img.width(), img.height());
    println!("{:?} {w}×{h} disk {:?}", a.kind, a.disk);
    let masks = ps_core::pipeline::masks_for(&p, &img);
    let mut sharp = Processor::default().run(Arc::clone(&img), next_generation(), &p, &Aux { masks: &masks, ..Default::default() }, &|| false, &|_, _| {}).unwrap().image;

    // SYNTH="depth,amp,sigma": adds a dark ring, a copy of the limb `depth` px inside it shifted
    // 3 px right, `amp` deep (fraction of the image's top) on the left half, fading on the right.
    let clean = Arc::clone(&sharp);
    if let Some(v) = env("SYNTH") {
        let v: Vec<f32> = v.split(',').map(|x| x.trim().parse().unwrap()).collect();
        let g = dering::find_globe(&img, dering::DeringRegion::Subject).expect("a whole disk");
        let top = sharp.channels()[0].iter().copied().fold(0.0f32, f32::max);
        let shifted = ps_core::mask::Ellipse { cx: g.cx + 3.0, rx: g.rx - v[0], ry: g.ry - v[0], ..g };
        let channels = sharp
            .channels()
            .iter()
            .map(|c| {
                c.iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                        if g.radius(x, y) >= 1.0 {
                            return val;
                        }
                        let dist = (shifted.radius(x, y) - 1.0) * shifted.rx.min(shifted.ry);
                        let side = (0.5 - 0.5 * ((x - g.cx) / g.rx)).clamp(0.0, 1.0);
                        val - v[1] * top * side * (-(dist / v[2]).powi(2) / 2.0).exp()
                    })
                    .collect()
            })
            .collect();
        sharp = Arc::new(Image::new(w, h, channels, sharp.source_bit_depth));
    }
    let radius = env("BRUSH").and_then(|v| v.parse::<f32>().ok()).unwrap_or(6.0);
    let points: Vec<[f32; 2]> = if let Some(s) = env("STROKE") {
        s.split(';').map(|p| {
            let v: Vec<f32> = p.split(',').map(|x| x.trim().parse().unwrap()).collect();
            [v[0], v[1]]
        }).collect()
    } else {
        let arc: Vec<f32> = env("ARC").unwrap_or("8,-60,60".into()).split(',').map(|x| x.trim().parse().unwrap()).collect();
        let g = dering::find_globe(&img, dering::DeringRegion::Subject).expect("a whole disk");
        println!("globe {g:?}");
        let (sin, cos) = g.angle.sin_cos();
        let steps = ((arc[2] - arc[1]) / 3.0).ceil() as usize;
        (0..=steps)
            .map(|k| {
                let phi = (arc[1] + (arc[2] - arc[1]) * k as f32 / steps as f32).to_radians();
                // A rough hand: ±1.5 px wobble.
                let depth = arc[0] + 1.5 * (k as f32 * 1.3).sin();
                let (u, v) = ((g.rx - depth) * phi.cos(), (g.ry - depth) * phi.sin());
                [g.cx + u * cos - v * sin, g.cy + u * sin + v * cos]
            })
            .collect()
    };
    let d = DeringParams { strokes: vec![Stroke { points, radius, hardness: 0.5, erase: false }], ..p.dering.params.clone() };
    let area = dering::painted(&d.strokes, w, h);
    let t = std::time::Instant::now();
    let ring = dering::find_ring(&sharp, &area, d.width).expect("painted");
    println!("ring in {:.3}s: reach {} px, max lift {:.4}", t.elapsed().as_secs_f32(), ring.reach, ring.map.iter().copied().fold(0.0f32, f32::max));
    let fixed = dering::dering(&sharp, &d, &area, Some(&ring));

    let s = working_planes(&sharp, ColorMode::Luminance).swap_remove(0);
    let f = working_planes(&fixed, ColorMode::Luminance).swap_remove(0);
    let cl = working_planes(&clean, ColorMode::Luminance).swap_remove(0);
    if env("SYNTH").is_some() {
        let (mut before, mut after) = (0.0f32, 0.0f32);
        for i in 0..w * h {
            if (cl[i] - s[i]) > 0.3 * (cl.iter().copied().fold(0.0f32, f32::max) * 0.0) + 0.01 && area[i] > 0.5 {
                before += cl[i] - s[i];
                after += (cl[i] - f[i]).abs();
            }
        }
        println!("synthetic ring under the brush: {before:.2} → {after:.2} ({:.0} % left)", 100.0 * after / before.max(1e-6));
    }
    let top = s.iter().copied().fold(1e-6f32, f32::max);
    let gray = |plane: &[f32]| -> Vec<f32> { plane.iter().map(|v| (v / top).clamp(0.0, 1.0)).collect() };
    let mut rgb = vec![vec![0.0f32; w * h]; 3];
    for i in 0..w * h {
        let k = (ring.map[i] / (0.15 * top)).clamp(0.0, 1.0);
        let outline = (area[i] > 0.4 && area[i] < 0.6) as u8 as f32 * 0.6;
        for (ch, c) in [1.0f32, 0.24, 0.9].iter().enumerate() {
            let base = s[i] / top * (1.0 - k) + c * k;
            let cyan = [0.0, 0.9, 1.0][ch];
            rgb[ch][i] = (base * (1.0 - outline) + cyan * outline).clamp(0.0, 1.0);
        }
    }
    if let Some(row) = env("ROW").and_then(|v| v.parse::<usize>().ok()) {
        let from: usize = env("FROM").and_then(|v| v.parse().ok()).unwrap_or(0);
        for x in from..(from + 40).min(w) {
            let i = row * w + x;
            println!("x {x} clean {:.4} sharp {:.4} fixed {:.4} lift {:.4} area {:.2}", cl[i], s[i], f[i], ring.map[i], area[i]);
        }
    }
    let views = [("sharp", vec![gray(&s)]), ("fixed", vec![gray(&f)]), ("map", rgb), ("clean", vec![gray(&cl)])];
    for (name, planes) in &views {
        ps_io::save(Path::new(&format!("{out}/{name}.png")), &Image::new(w, h, planes.clone(), 16)).unwrap();
    }
    if let Some(c) = env("CROP") {
        let c: Vec<usize> = c.split(',').map(|x| x.trim().parse().unwrap()).collect();
        let (x0, y0, cw, ch) = (c[0], c[1], c[2].min(w - c[0]), c[3].min(h - c[1]));
        let z = 4;
        for (name, planes) in &views {
            let zoomed: Vec<Vec<f32>> = planes.iter().map(|pl| (0..ch * z * cw * z).map(|i| pl[(y0 + i / (cw * z) / z) * w + x0 + i % (cw * z) / z]).collect()).collect();
            ps_io::save(Path::new(&format!("{out}/{name}_zoom.png")), &Image::new(cw * z, ch * z, zoomed, 16)).unwrap();
        }
    }
}
