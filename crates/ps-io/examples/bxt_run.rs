//! Runs BlurXTerminator on an image through the app's bridge:
//! `cargo run --release -p ps-io --example bxt_run -- in.tif out.tif [psf diameter px]`
use ps_core::bxt::{BxtParams, find_executable, run};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let (input, output) = (args.next().expect("input"), args.next().expect("output"));
    let exe = find_executable().expect("rc-astro not found");
    let img = ps_io::load(input.as_ref()).unwrap();
    let start = Instant::now();
    let mut params = BxtParams::default();
    if let Some(d) = args.next().and_then(|d| d.parse().ok()) {
        params.psf_diameter = d;
    }
    println!("PSF diameter {:.2} px", params.psf_diameter);
    let out = run(&img, &params, &exe, &|| false).unwrap().unwrap();
    println!("{} in {:.1} s", exe.display(), start.elapsed().as_secs_f64());
    ps_io::save(output.as_ref(), &out).unwrap();
}
