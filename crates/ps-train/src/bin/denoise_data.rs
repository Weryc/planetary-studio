//! Learned noise reduction, step 1: the clean stacks.
//!
//! `cargo run --release -p ps-train --bin denoise_data -- out_dir reference…`
//!
//! Each sharp reference gives the same truths as the learned sharpening (`ml_common/truths.rs`:
//! linear, at amateur sizes, Moon and Sun as whole disk and detailed tiles); each truth is seen
//! on the three simulated nights (lucky imaging through turbulence, best frames stacked), and
//! the luminance of that noiseless stack is what the network must give back. Noise is added
//! later, during training, so every pass sees new noise.
//!
//! Writes `out_dir/<name>.f32` (u32 width, u32 height, then the plane as little-endian f32)
//! and `out_dir/index.tsv` (file, group, width, height).
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ml::truths::truths;
use ml::*;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("out dir");
    std::fs::create_dir_all(&out).unwrap();
    let nights = Nights::new();
    let mut index = String::from("file\tgroup\twidth\theight\n");
    let mut alternate = 0usize;
    for path in args {
        let path = Path::new(&path);
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let Some(reference) = load_any(path) else {
            eprintln!("{stem}: illeggibile");
            continue;
        };
        let group = stem.trim_end_matches("_pub").to_string();
        let started = std::time::Instant::now();
        let mut made = 0;
        for (tag, truth, k) in truths(&stem, &reference) {
            alternate += 1;
            // Every other truth seen as a filtered mono camera sees it (one PSF, red-ish).
            let as_mono = alternate.is_multiple_of(2) || truth.is_mono();
            let truth = if as_mono { ml::truths::mono(&truth) } else { truth };
            for i in 0..seeing::NIGHTS.len() {
                let blurred = seeing::blur_with(&truth, &nights.kernels(i, as_mono, k));
                let y = lum(&blurred);
                let file = format!("{}_{}_n{i}.f32", stem.replace(' ', "_"), tag);
                let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{out}/{file}")).unwrap());
                f.write_all(&(blurred.width() as u32).to_le_bytes()).unwrap();
                f.write_all(&(blurred.height() as u32).to_le_bytes()).unwrap();
                for v in &y {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
                writeln!(index, "{file}\t{group}\t{}\t{}", blurred.width(), blurred.height()).unwrap();
                made += 1;
            }
        }
        eprintln!("{stem:<24} {made} stack puliti in {:.0} s", started.elapsed().as_secs_f32());
        std::fs::write(format!("{out}/index.tsv"), &index).unwrap();
    }
}
