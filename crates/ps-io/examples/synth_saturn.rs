//! Writes a synthetic, slightly blurred Saturn for testing:
//! `cargo run -p ps-io --example synth_saturn -- out.tif`
#![allow(clippy::needless_range_loop)]
use ps_core::Image;
use ps_core::fft::FftConvolver;
use ps_core::psf::Psf;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "synthetic_saturn.tif".into());
    let (w, h, r) = (1000usize, 620usize, 150.0f32);
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let (s, c) = (-8f32).to_radians().sin_cos();
    let tints = [1.0f32, 0.88, 0.62];
    let mut channels = vec![vec![0.0f32; w * h]; 3];
    for i in 0..w * h {
        let (dx, dy) = ((i % w) as f32 + 0.5 - cx, (i / w) as f32 + 0.5 - cy);
        let (u, v) = (dx * c + dy * s, -dx * s + dy * c);
        let globe = (u / r).powi(2) + (v / (0.9 * r)).powi(2);
        let e = ((u / (2.27 * r)).powi(2) + (v / (0.42 * r)).powi(2)).sqrt();
        let ring = if (0.52..0.98).contains(&e) && !(0.82..0.85).contains(&e) { 0.55 + 0.1 * (e * 60.0).sin() } else { 0.0 };
        // The ring passes in front of the globe on the near side (v > 0).
        let value = if globe < 1.0 && !(ring > 0.0 && v > 0.0) { 0.75 * (1.0 - globe).powf(0.25) * (0.85 + 0.15 * (v * 0.08).sin()) } else { ring };
        for (channel, t) in channels.iter_mut().zip(tints) {
            channel[i] = value * t;
        }
    }
    let conv = FftConvolver::new(w, h, &Psf::Gaussian { sigma: 1.8 }.kernel());
    for ch in &mut channels {
        *ch = conv.convolve(ch).into_iter().map(|v| v.clamp(0.0, 1.0)).collect();
    }
    ps_io::save(path.as_ref(), &Image::new(w, h, channels, 16)).unwrap();
    println!("wrote {path}");
}
