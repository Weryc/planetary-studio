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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(&args[0]);
    let mut rows = Vec::new();
    for dir in args[1..].iter().map(PathBuf::from) {
        // Real stacks (ser_stack) call the full one piena.tif, simulated ones grezza.tif.
        let full_path = if dir.join("piena.tif").exists() { dir.join("piena.tif") } else { dir.join("grezza.tif") };
        let (full, even, odd) = (load(&full_path), load(&dir.join("pari.tif")), load(&dir.join("dispari.tif")));
        let a = analyze(&full);
        let Some((m, f)) = measure(&full, &a) else {
            eprintln!("{}: nessun disco misurabile", dir.display());
            continue;
        };
        let (w, h) = (full.width(), full.height());
        let half_diff = Image::new(w, h, even.channels().iter().zip(odd.channels()).map(|(e, o)| e.iter().zip(o).map(|(a, b)| 0.5 * (a - b)).collect()).collect(), 16);
        let d = half_diff.crop(f.x0, f.y0, f.side, f.side);
        let (dl, dc) = planes(&d);
        let lum = f.power(&dl, &f.inside);
        let chroma = dc.map(|[r, b]| {
            let (pr, pb) = (f.power(&r, &f.inside), f.power(&b, &f.inside));
            pr.iter().zip(&pb).map(|(x, y)| (10f32.powf(*x) + 10f32.powf(*y)).log10()).collect::<Vec<f32>>()
        });
        let meta = std::fs::read_to_string(dir.join("meta.json")).unwrap_or_else(|_| "{}".into());
        rows.push(format!(
            "  {{\"stack\": \"{}\", \"meta\": {}, \"x\": {}, \"rumore_grana\": {}, \"rumore_colore\": {}, \"cielo\": {}, \"disco\": {:.1}}}",
            dir.file_name().unwrap().to_string_lossy(),
            meta.trim(),
            arr(&features(&m)),
            arr(&lum),
            chroma.as_ref().map_or("null".into(), |c| arr(c)),
            m.lum.sky.is_some(),
            f.diameter
        ));
        eprintln!("{}: misurato", dir.display());
    }
    std::fs::write(&out, format!("[\n{}\n]\n", rows.join(",\n"))).unwrap();
}
