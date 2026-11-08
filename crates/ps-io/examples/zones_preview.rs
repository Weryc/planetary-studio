//! Where "Nitidezza intelligente" holds back, over the image (blue: zones, orange: edge
//! protection), as the app shows it with its automatic settings.
//! `cargo run --release -p ps-io --example zones_preview -- out.png image.tif`
use ps_core::Image;
use ps_core::color::{ColorMode, working_planes};

fn main() {
    let mut args = std::env::args().skip(1);
    let (out, path) = (args.next().expect("out.png"), args.next().expect("image"));
    let img = ps_io::load(path.as_ref()).unwrap();
    let a = ps_core::analysis::analyze(&img);
    let p = ps_core::preset::auto_smart(a.kind, &a, ps_core::preset::AutoStrength::Normal);
    println!("{:?}: protezione {:.2}, curva {:?}", a.kind, p.limb_protection, p.curve);
    let (w, h) = (img.width(), img.height());
    let plane = working_planes(&img, ColorMode::Luminance).swap_remove(0);
    let (zones, edges) = ps_core::smart::hold_back_map(&plane, w, h, &p);
    let top = plane.iter().copied().fold(1e-6f32, f32::max);
    let mut rgb = vec![vec![0.0f32; w * h]; 3];
    for i in 0..w * h {
        let g = plane[i] / top;
        let (z, e) = (zones[i].clamp(0.0, 1.0), edges[i].clamp(0.0, 1.0));
        let (c, a) = if e >= z { ([1.0, 0.59, 0.16], e) } else { ([0.27, 0.59, 1.0], z) };
        let a = a * 0.75;
        for k in 0..3 {
            rgb[k][i] = g * (1.0 - a) + c[k] * a;
        }
    }
    ps_io::save(out.as_ref(), &Image::new(w, h, rgb, 16)).unwrap();
}
