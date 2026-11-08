//! Writes a synthetic, slightly blurred and noisy "planet" for testing:
//! `cargo run -p ps-io --example synth -- out.tif [size]`

use ps_core::Image;
use ps_core::fft::FftConvolver;
use ps_core::psf::Psf;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "synthetic_planet.tif".into());
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(900);
    let (cx, cy, r) = (n as f32 / 2.0, n as f32 / 2.0, n as f32 * 0.36);
    let mut state = 12345u32;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32 - 0.5
    };
    let tints = [1.0f32, 0.86, 0.68];
    let mut channels: Vec<Vec<f32>> = vec![vec![0.0; n * n]; 3];
    for y in 0..n {
        for x in 0..n {
            let (dx, dy) = ((x as f32 - cx) / r, (y as f32 - cy) / r);
            let d2 = dx * dx + dy * dy;
            if d2 >= 1.0 {
                continue;
            }
            let limb = (1.0 - d2).sqrt().powf(0.35);
            let bands = 0.75 + 0.25 * (dy * 22.0).sin() * (1.0 + 0.3 * (dx * 9.0 + dy * 40.0).sin());
            let spot = (-((dx - 0.25).powi(2) / 0.004 + (dy - 0.3).powi(2) / 0.0015)).exp();
            let fine = 0.04 * ((dx * 120.0).sin() * (dy * 95.0).cos());
            for (c, tint) in tints.iter().enumerate() {
                channels[c][y * n + x] = (0.78 * limb * (bands + fine) * tint - 0.15 * spot * (c as f32)).max(0.0);
            }
        }
    }
    let conv = FftConvolver::new(n, n, &Psf::Gaussian { sigma: 2.2 }.kernel());
    for c in &mut channels {
        *c = conv.convolve(c).into_iter().map(|v| (v + 0.012 * noise()).clamp(0.0, 1.0)).collect();
    }
    ps_io::save(path.as_ref(), &Image::new(n, n, channels, 16)).unwrap();
    println!("wrote {path}");
}
