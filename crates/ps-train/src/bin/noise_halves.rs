//! The real noise of real stacks, from their two half stacks (even and odd frames, written by
//! `ser_stack`): half their difference has the full stack's random noise. Measured band by band
//! inside the subject, framed as `ps_core::curve_net` frames it, for the grain (luminance) and
//! the colour (R − Y plus B − Y), next to the noise estimator's inputs on the full stack.
//!
//! `cargo run --release -p ps-train --bin noise_halves -- out.json stack_dir…`
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::analysis::analyze;
use ps_core::noise_net::{features, measure, planes};
use std::path::PathBuf;

fn arr(v: &[f32]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(","))
}

fn load(p: &std::path::Path) -> Image {
    // The project's reader also takes the mono float TIFFs the simulator writes.
    ml::load_any(p).unwrap_or_else(|| panic!("{}: illeggibile", p.display()))
}

/// `plane` moved by `d` px exactly (a phase ramp in frequency; same direction as
/// `ps_core::align::shift`).
fn fourier_shift(plane: &[f32], w: usize, h: usize, d: [f32; 2]) -> Vec<f32> {
    let plan = ps_core::fft::FftConvolver::plan(w, h);
    let mut spec = plan.forward(plane);
    for (idx, c) in spec.iter_mut().enumerate() {
        let (kx, ky) = (idx / h, idx % h);
        let fy = if ky <= h / 2 { ky as f32 } else { ky as f32 - h as f32 } / h as f32;
        let phase = std::f32::consts::TAU * (kx as f32 / w as f32 * d[0] + fy * d[1]);
        let (sn, cs) = phase.sin_cos();
        let (re, im) = (c.re, c.im);
        c.re = re * cs - im * sn;
        c.im = re * sn + im * cs;
    }
    plan.inverse(spec)
}

/// The stack with its sky clipped to black, as some programs leave it (a black point just above
/// the sky's 99th percentile): the estimator also learns to do without the sky.
fn black_sky(img: &Image, disk: Option<ps_core::analysis::Disk>) -> Image {
    let lum = ml::lum(img);
    let (w, h) = (img.width(), img.height());
    let (cx, cy, r) = disk.map_or((w as f32 / 2.0, h as f32 / 2.0, 0.3 * w as f32), |d| (d.cx, d.cy, d.radius));
    let mut sky: Vec<f32> = (0..w * h).filter(|&i| ((i % w) as f32 - cx).hypot((i / w) as f32 - cy) > 1.3 * r).map(|i| lum[i]).collect();
    sky.sort_by(f32::total_cmp);
    let level = sky.get(sky.len() * 99 / 100).copied().unwrap_or(0.0);
    Image::new(w, h, img.channels().iter().map(|p| p.iter().map(|v| (v - level).max(0.0)).collect()).collect(), 16)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(&args[0]);
    let mut rows = Vec::new();
    for dir in args[1..].iter().map(PathBuf::from) {
        // Real stacks (ser_stack) call the full one piena.tif, simulated ones grezza.tif.
        let full_path = if dir.join("piena.tif").exists() { dir.join("piena.tif") } else { dir.join("grezza.tif") };
        let (full, even, odd) = (load(&full_path), load(&dir.join("pari.tif")), load(&dir.join("dispari.tif")));
        let mut a = analyze(&full);
        // NOISE_HALVES_SATURN=1: Saturn's globe as the disk (to compare with Jupiter's data).
        if std::env::var("NOISE_HALVES_SATURN").is_ok()
            && let Some(z) = ps_core::saturn::zones(&full)
        {
            a.disk = Some(ps_core::analysis::Disk { cx: z.globe.cx, cy: z.globe.cy, radius: z.globe.rx });
        }
        let Some((m, f)) = measure(&full, &a) else {
            eprintln!("{}: nessun disco misurabile", dir.display());
            continue;
        };
        let (w, h) = (full.width(), full.height());
        // The two halves can sit a few hundredths of a pixel apart (frames the stacker aligned
        // a little wrong fall unevenly into them): that offset is not noise, and left in it
        // would draw the planet's belts into the difference. Each half moves half of it, exactly.
        let lum = |img: &Image| ps_core::color::working_planes(img, ps_core::color::ColorMode::Luminance).swap_remove(0);
        let offset = ps_core::align::measure_shift(&lum(&even), &lum(&odd), w, h);
        let move_by = |img: &Image, k: f32| -> Vec<Vec<f32>> { img.channels().iter().map(|c| fourier_shift(c, w, h, [k * offset[0], k * offset[1]])).collect() };
        let (even_c, odd_c) = (move_by(&even, -0.5), move_by(&odd, 0.5));
        let half_diff = Image::new(w, h, even_c.iter().zip(&odd_c).map(|(e, o)| e.iter().zip(o).map(|(a, b)| 0.5 * (a - b)).collect()).collect(), 16);
        let d = half_diff.crop(f.x0, f.y0, f.side, f.side);
        let (dl, dc) = planes(&d);
        let lum = f.power(&dl, &f.inside);
        let chroma = dc.map(|[r, b]| {
            let (pr, pb) = (f.power(&r, &f.inside), f.power(&b, &f.inside));
            pr.iter().zip(&pb).map(|(x, y)| (10f32.powf(*x) + 10f32.powf(*y)).log10()).collect::<Vec<f32>>()
        });
        let meta = std::fs::read_to_string(dir.join("meta.json")).unwrap_or_else(|_| "{}".into());
        rows.push(format!(
            "  {{\"stack\": \"{}\", \"scarto_px\": [{:.4},{:.4}], \"meta\": {}, \"x\": {}, \"x_nero\": {}, \"rumore_grana\": {}, \"rumore_colore\": {}, \"cielo\": {}, \"disco\": {:.1}}}",
            dir.file_name().unwrap().to_string_lossy(),
            offset[0],
            offset[1],
            meta.trim(),
            arr(&features(&m)),
            {
                let b = black_sky(&full, a.disk);
                let mut ab = analyze(&b);
                ab.disk = a.disk;
                measure(&b, &ab).map_or("null".to_string(), |(mb, _)| arr(&features(&mb)))
            },
            arr(&lum),
            chroma.as_ref().map_or("null".into(), |c| arr(c)),
            m.lum.sky.is_some(),
            f.diameter
        ));
        eprintln!("{}: misurato", dir.display());
    }
    std::fs::write(&out, format!("[\n{}\n]\n", rows.join(",\n"))).unwrap();
}
