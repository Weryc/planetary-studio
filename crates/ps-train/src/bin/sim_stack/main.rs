//! A lucky-imaging capture simulated frame by frame, then stacked as AutoStakkert would.
//!
//! `cargo run --release -p ps-train --bin sim_stack -- --ref riferimento.jpg [opzioni]`
//!
//! A sharp reference (Hubble, Juno…) is brought to the angular size and the pixel scale of a
//! telescope + camera. Each frame looks through its own two-layer atmosphere: the tilt changes
//! smoothly across the disk (warp) and so does the blur (PSFs on a 3×3 grid, blended). The
//! camera adds shot noise, read noise, row noise, hot pixels and the ADC; colour cameras see
//! it through an RGGB Bayer mosaic. The best frames are then graded, aligned (globally and on
//! alignment points), resampled and averaged by a real stacker: the noise comes out with the
//! blotches of real stacks because it is made the same way.
//!
//! Writes, under `--out` (default `Immagini_test/simulate`):
//!   <riferimento>/<telescopio>_<camera>/<run>/verita_campionata.tif   the truth at the sensor's pixels
//!   <riferimento>/<telescopio>_<camera>/<run>/limite_diffrazione.tif  … through the bare aperture
//!   (`--truth-only 1` writes just these two, for runs made before they lived there)
//!   <riferimento>/<telescopio>_<camera>/<seeing>_g<gain>_<bit>bit_s<seed>/
//!       grezza.tif (float32, linear: ≈ fotometria × truth ⊛ PSF), anteprime, meta.json
//!
//! Options (defaults in brackets): --scope [edgehd11] --camera [asi678mc] --diam arcsec [47]
//! --frame px [576] --frames [800] --keep share [0.25] --d-r0 [6] --gain basso|medio|alto
//! [medio] --bits 8|12 [8] --seed [1] --ground share [0.6] --high-km [8] --outer-m [25] --dispersion arcsec
//! [0.06] --label name --compare real_raw.tif
#[path = "../../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

mod camera;
mod optics;
mod rng;
mod stack;
mod tiff;

use camera::Sensor;
use optics::{ARCSEC_PER_RAD, Atmosphere, Optics, Turbulence};
use ps_core::Image;
use ps_core::fft::{FftConvolver, next_fast_size};
use rayon::prelude::*;
use rng::Rng;
use rustfft::num_complex::Complex32;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args(HashMap<String, String>);

impl Args {
    fn parse() -> Self {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let mut map = HashMap::new();
        let mut i = 0;
        while i < raw.len() {
            let key = raw[i].trim_start_matches("--").to_string();
            let value = raw.get(i + 1).cloned().unwrap_or_default();
            map.insert(key, value);
            i += 2;
        }
        Args(map)
    }
    fn str(&self, k: &str, default: &str) -> String {
        self.0.get(k).cloned().unwrap_or_else(|| default.to_string())
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> T {
        self.0.get(k).and_then(|v| v.parse().ok()).unwrap_or(default)
    }
}

/// `img` centred on `centre` in a black `side`² square.
fn embed(img: &Image, centre: (f32, f32), side: usize) -> Vec<Vec<f32>> {
    let (w, h) = (img.width() as isize, img.height() as isize);
    let (x0, y0) = ((centre.0 - side as f32 / 2.0).round() as isize, (centre.1 - side as f32 / 2.0).round() as isize);
    img.channels()
        .iter()
        .map(|p| {
            (0..side * side)
                .map(|i| {
                    let (x, y) = (x0 + (i % side) as isize, y0 + (i / side) as isize);
                    if (0..w).contains(&x) && (0..h).contains(&y) { p[(y * w + x) as usize] } else { 0.0 }
                })
                .collect()
        })
        .collect()
}

/// Bounding box of the lit subject (smoothed luminance above 6 % of its top): (width, centre).
/// For Saturn the ring span, which the disk analysis doesn't give.
fn lit_extent(img: &Image) -> (f32, (f32, f32)) {
    let (w, h) = (img.width(), img.height());
    let l = ps_core::mask::gaussian_approx(&ml::lum(img), w, h, 2.0);
    let mut sorted: Vec<f32> = l.iter().step_by(7).copied().collect();
    sorted.sort_by(f32::total_cmp);
    let top = sorted[sorted.len() * 999 / 1000];
    let (mut x0, mut x1, mut y0, mut y1) = (w, 0, h, 0);
    for (i, &v) in l.iter().enumerate() {
        if v > 0.06 * top {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
    }
    ((x1.saturating_sub(x0) + 1) as f32, ((x0 + x1) as f32 / 2.0, (y0 + y1) as f32 / 2.0))
}

/// The `frame`² centre of a `cw`² canvas, binned `f`×`f` to sensor pixels.
fn to_sensor(p: &[f32], cw: usize, frame: usize, f: usize) -> Vec<f32> {
    let o = (cw - frame) / 2;
    let s = frame / f;
    (0..s * s)
        .map(|i| {
            let (x, y) = ((i % s) * f + o, (i / s) * f + o);
            let mut acc = 0.0;
            for dy in 0..f {
                for dx in 0..f {
                    acc += p[(y + dy) * cw + x + dx];
                }
            }
            acc / (f * f) as f32
        })
        .collect()
}

fn convolve(plan: &FftConvolver, spectrum: &[Complex32], kernel: &[f32]) -> Vec<f32> {
    let mut k = plan.forward(kernel);
    k.iter_mut().zip(spectrum).for_each(|(a, b)| *a *= b);
    plan.inverse(k)
}

/// Linear weights of the 3 grid columns (or rows) at every canvas coordinate.
fn hat_weights(n: usize, points: &[f32]) -> Vec<Vec<f32>> {
    let step = points[1] - points[0];
    let mut w = vec![vec![0.0f32; n]; points.len()];
    for x in 0..n {
        let t = ((x as f32 - points[0]) / step).clamp(0.0, (points.len() - 1) as f32);
        let i = (t.floor() as usize).min(points.len() - 2);
        let f = t - i as f32;
        w[i][x] = 1.0 - f;
        w[i + 1][x] = f;
    }
    w
}

/// Display version: 99.9th percentile of the brightest channel at 1, gamma 2.2.
fn preview(planes: &[Vec<f32>], w: usize, h: usize) -> Image {
    let mut v: Vec<f32> = (0..w * h).step_by(3).map(|i| planes.iter().map(|p| p[i]).fold(0.0f32, f32::max)).collect();
    v.sort_by(f32::total_cmp);
    let top = v[v.len() * 999 / 1000].max(1e-6);
    let mut out: Vec<Vec<f32>> = planes.iter().map(|p| p.iter().map(|x| (x / top).clamp(0.0, 1.0).powf(1.0 / 2.2)).collect()).collect();
    if out.len() == 1 {
        out = vec![out[0].clone(), out[0].clone(), out[0].clone()];
    }
    Image::new(w, h, out, 16)
}

fn montage(images: &[Image]) -> Image {
    let gap = 8;
    let h = images.iter().map(|i| i.height()).max().unwrap();
    let w: usize = images.iter().map(|i| i.width()).sum::<usize>() + gap * (images.len() - 1);
    let mut planes = vec![vec![0.15f32; w * h]; 3];
    let mut x0 = 0;
    for img in images {
        for (c, plane) in planes.iter_mut().enumerate() {
            let src = &img.channels()[c];
            for y in 0..img.height() {
                plane[y * w + x0..y * w + x0 + img.width()].copy_from_slice(&src[y * img.width()..(y + 1) * img.width()]);
            }
        }
        x0 += img.width() + gap;
    }
    Image::new(w, h, planes, 16)
}

fn describe(name: &str, img: &Image) -> ps_core::analysis::Analysis {
    let a = ps_core::analysis::analyze(img);
    let shift = a.rgb_shift.map_or("-".into(), |s| format!("R({:+.2},{:+.2}) B({:+.2},{:+.2})", s.red[0], s.red[1], s.blue[0], s.blue[1]));
    eprintln!(
        "  {name:<10} disco r={:>6.1}  rumore={:.5}  SNR={:>6.0}  σ PSF bordo={:>5.2}  σ RGB={:<22}  sposta RGB {}",
        a.disk.map_or(f32::NAN, |d| d.radius),
        a.noise,
        a.snr,
        a.psf_sigma.unwrap_or(f32::NAN),
        a.channel_sigma.map_or("-".into(), |s| format!("{:.2}/{:.2}/{:.2}", s[0], s[1], s[2])),
        shift
    );
    a
}

/// Luminance power spectrum inside the disk, over its value at f = 0.03 (cycles/px).
fn spectrum(img: &Image, a: &ps_core::analysis::Analysis) -> Option<Vec<(f32, f32)>> {
    let sq = ml::inner_square(img.width(), img.height(), ml::outline(img, a), 64)?;
    let s = ml::smooth(&ml::spectrum_of(&ml::lum(img), img.width(), sq));
    let base = ml::at(&s, 0.03);
    Some(s.into_iter().map(|(f, p)| (f, p / base)).collect())
}

fn main() {
    let args = Args::parse();
    let started = Instant::now();
    let reference_path = PathBuf::from(args.str("ref", ""));
    let scope = optics::telescope(&args.str("scope", "edgehd11")).expect("telescopio sconosciuto");
    let cam = camera::camera(&args.str("camera", "asi678mc")).expect("camera sconosciuta");
    let gain = camera::gain(&args.str("gain", "medio")).expect("gain: basso, medio o alto");
    let bits: u32 = args.num("bits", 8);
    let seed: u64 = args.num("seed", 1);
    let stem0 = reference_path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    // What the reference shows: planets are scaled by their apparent size (Saturn by its ring
    // span), the Moon by the reference's own pixel scale and seen as a surface filling the frame.
    let body = args.str("body", ml::subject_of(&ml::group_of(&stem0)));
    let surface = body == "Moon";
    let diam: f32 = args.num("diam", match body.as_str() { "Saturn" => 42.0, "Mars" => 18.0, _ => 47.0 });
    // LROC WAC mosaic: 100 m/px at 384 400 km.
    let ref_arcsec: f32 = args.num("ref-arcsec", 0.1 / 384_400.0 * ARCSEC_PER_RAD);
    let frames: usize = args.num("frames", 800);
    let keep_share: f32 = args.num("keep", 0.25);
    let atm = Atmosphere { d_r0: args.num("d-r0", 6.0), ground_share: args.num("ground", 0.6), high_km: args.num("high-km", 8.0), outer_m: args.num("outer-m", 25.0) };
    let dispersion: f32 = args.num("dispersion", 0.06);
    let label = args.str("label", &format!("dr0_{:.1}", atm.d_r0));
    let out_root = PathBuf::from(args.str("out", "Immagini_test/simulate"));

    // Scales: the sensor pixel, and the finer pixel the optics are simulated on when the
    // sensor undersamples the PSF (each wavelength's PSF must be Nyquist-sampled).
    let sensor_arcsec = ARCSEC_PER_RAD * cam.pixel_um * 1e-6 / (scope.focal_mm * 1e-3);
    let lambda_d = 0.55e-6 / (scope.aperture_mm * 1e-3) * ARCSEC_PER_RAD;
    let lambdas = if cam.bayer { vec![620.0 / 550.0, 1.0, 460.0 / 550.0] } else { vec![685.0 / 550.0] };
    let min_l = lambdas.iter().copied().fold(f32::MAX, f32::min);
    let f = ((sensor_arcsec / lambda_d) / (0.45 * min_l)).ceil().max(1.0) as usize;
    let fine_arcsec = sensor_arcsec / f as f32;
    let frame: usize = args.num("frame", if surface { 384 } else { next_fast_size(((1.35 * diam / sensor_arcsec) as usize).max(192)) });
    let frame_fine = frame * f;
    let optics = Optics::new(scope, fine_arcsec, lambdas.clone());
    let channels = lambdas.len();
    eprintln!(
        "{} + {}: {:.4}\"/px, λ/D = {:.3}\" = {:.2} px, sovracampionamento {f}×, D/r0 = {:.1} (seeing ≈ {:.2}\")",
        scope.label,
        cam.label,
        sensor_arcsec,
        lambda_d,
        lambda_d / sensor_arcsec,
        atm.d_r0,
        0.98 * lambda_d * atm.d_r0
    );

    // The truth at the fine scale.
    let reference = ml::load_any(&reference_path).expect("riferimento illeggibile");
    let reference = if reference.width().max(reference.height()) > 3500 { ml::truths::scaled(&reference, 3500.0 / reference.width().max(reference.height()) as f32) } else { reference };
    let mut linear = ml::truths::linear(&reference);
    if !cam.bayer {
        linear = ml::truths::mono(&linear);
    } else if linear.is_mono() {
        let p = linear.channels()[0].clone();
        linear = Image::new(linear.width(), linear.height(), vec![p.clone(), p.clone(), p], 16);
    }
    let margin = (frame_fine / 5).max(64);
    let cw = next_fast_size(frame_fine + 2 * margin);
    let o = (cw - frame_fine) / 2;
    let canvas: Vec<Vec<f32>> = if surface {
        // The surface fills the whole canvas (no black border to darken the frame's edges),
        // centred on its most detailed zone.
        let small = ml::truths::scaled(&linear, ref_arcsec / fine_arcsec);
        assert!(small.width() >= cw && small.height() >= cw, "riferimento troppo piccolo per questo campo ({}×{} contro {cw})", small.width(), small.height());
        let centre = ml::truths::busiest(&small, cw, 1).first().copied().unwrap_or((small.width() as f32 / 2.0, small.height() as f32 / 2.0));
        let half = cw as f32 / 2.0;
        let centre = (centre.0.clamp(half, small.width() as f32 - half), centre.1.clamp(half, small.height() as f32 - half));
        eprintln!("verità: superficie a {ref_arcsec:.4}\"/px → riquadro {frame_fine}² su tela {cw}²");
        embed(&small, centre, cw)
    } else {
        let a = ps_core::analysis::analyze(&linear);
        let (size_px, centre) = if body == "Saturn" { lit_extent(&linear) } else { (a.disk.map_or(a.subject_size, |d| 2.0 * d.radius), ml::centre(&linear, &a)) };
        let factor = diam / fine_arcsec / size_px;
        let small = ml::truths::scaled(&linear, factor);
        let truth = embed(&small, (centre.0 * factor, centre.1 * factor), frame_fine);
        eprintln!("verità: {body} {size_px:.0} px → {:.0} px (fine), riquadro {frame_fine}²", size_px * factor);
        // Black margins wide enough for the seeing halo.
        truth
            .iter()
            .map(|p| {
                let mut c = vec![0.0f32; cw * cw];
                for y in 0..frame_fine {
                    c[(y + o) * cw + o..(y + o) * cw + o + frame_fine].copy_from_slice(&p[y * frame_fine..(y + 1) * frame_fine]);
                }
                c
            })
            .collect()
    };
    let plan = FftConvolver::plan(cw, cw);
    let spectra: Vec<Vec<Complex32>> = canvas.iter().map(|p| plan.forward(p)).collect();

    let stem = stem0.clone();
    let base = out_root.join(&stem).join(format!("{}_{}", scope.name, cam.name));
    let run_dir = base.join(format!("{label}_g{}_{bits}bit_s{seed}", gain.label));
    std::fs::create_dir_all(&run_dir).unwrap();

    // The two truths.
    let sampled: Vec<Vec<f32>> = canvas.iter().map(|p| to_sensor(p, cw, frame_fine, f)).collect();
    let airy = optics::airy(&optics);
    let diffraction: Vec<Vec<f32>> = (0..channels).map(|c| to_sensor(&convolve(&plan, &spectra[c], &optics::wrap_onto(&airy[c], cw, cw)), cw, frame_fine, f)).collect();
    // In the run's own folder: the truth depends on the disk's size, which changes between runs
    // of the same telescope and camera (a shared copy was overwritten by the next run).
    tiff::write_f32(&run_dir.join("verita_campionata.tif"), frame, frame, &sampled).unwrap();
    tiff::write_f32(&run_dir.join("limite_diffrazione.tif"), frame, frame, &diffraction).unwrap();
    if args.num("truth-only", 0) == 1 {
        eprintln!("verità scritte in {}", run_dir.display());
        return;
    }

    // The capture.
    let mut rng0 = Rng::new(seed);
    let sensor = Sensor::new(frame, frame, cam.bayer, bits, gain, &mut rng0);
    let drift_angle = rng0.uniform() as f32 * std::f32::consts::TAU;
    // Smaller on the Moon: a surface has no planet to grab first, the correlation does it all.
    let drift_len = if surface { 4.0 + 6.0 * rng0.uniform() as f32 } else { 10.0 + 15.0 * rng0.uniform() as f32 } * f as f32;
    let drift = (drift_len * drift_angle.cos(), drift_len * drift_angle.sin());
    // Atmospheric dispersion: red up a little, blue down more (fine px, before the stacker).
    let disp: Vec<f32> = if cam.bayer { vec![-0.4 * dispersion, 0.0, dispersion] } else { vec![0.0] }.into_iter().map(|d| d / fine_arcsec).collect();
    let field_arcsec = cw as f32 * fine_arcsec;
    let centre_c = cw as f32 / 2.0;
    let grid3: Vec<f32> = (0..3).map(|k| centre_c + (k as f32 - 1.0) * frame_fine as f32 / 3.0).collect();
    let grid7: Vec<f32> = (0..7).map(|k| centre_c + (k as f32 - 3.0) * frame_fine as f32 * 1.15 / 6.0).collect();
    let weights = hat_weights(cw, &grid3);
    let theta = |x: f32, y: f32| ((x - centre_c) * fine_arcsec, (y - centre_c) * fine_arcsec);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let tilt_rms = std::sync::Mutex::new((0.0f64, 0.0f64, 0usize));
    let captured: Vec<(f32, Vec<u16>)> = (0..frames)
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(i as u64 + 1));
            let turb = Turbulence::new(&atm, &scope, field_arcsec, &mut rng);
            // Tilt across the field (fine px).
            let tilts: Vec<[f32; 2]> = (0..49)
                .map(|k| {
                    let (_, a, b) = turb.plane(theta(grid7[k % 7], grid7[k / 7]), scope.obstruction);
                    [a / std::f32::consts::TAU / optics.pix_d, b / std::f32::consts::TAU / optics.pix_d]
                })
                .collect();
            {
                let mut t = tilt_rms.lock().unwrap();
                let centre = tilts[24];
                // Global tilt, and how much the tilt differs across the disk.
                let spread = tilts.iter().map(|d| (d[0] - centre[0]).powi(2) + (d[1] - centre[1]).powi(2)).sum::<f32>() / 49.0;
                *t = (t.0 + (centre[0].powi(2) + centre[1].powi(2)) as f64, t.1 + spread as f64, t.2 + 1);
            }
            // Blur: each channel through the 3×3 grid of PSFs (tilt removed), blended.
            let mut blended = vec![vec![0.0f32; cw * cw]; channels];
            for ky in 0..3 {
                for kx in 0..3 {
                    let th = theta(grid3[kx], grid3[ky]);
                    let (c0, a, b) = turb.plane(th, scope.obstruction);
                    let phase = |u: f32, v: f32| turb.phase(u, v, th) - c0 - a * u - b * v;
                    for (c, acc) in blended.iter_mut().enumerate() {
                        let psf = optics.psf(c, &phase);
                        let conv = convolve(&plan, &spectra[c], &optics::wrap_onto(&psf, cw, cw));
                        for y in 0..cw {
                            let wy = weights[ky][y];
                            if wy == 0.0 {
                                continue;
                            }
                            for x in 0..cw {
                                acc[y * cw + x] += wy * weights[kx][x] * conv[y * cw + x];
                            }
                        }
                    }
                }
            }
            // Move: tilt field, drift of the mount, dispersion.
            let t = i as f32 / frames as f32 - 0.5;
            let jitter = (0.5 * f as f32 * rng.gauss(), 0.5 * f as f32 * rng.gauss());
            let (gx, gy) = (drift.0 * t + jitter.0, drift.1 * t + jitter.1);
            let step7 = grid7[1] - grid7[0];
            let tilt_at = |x: f32, y: f32| -> [f32; 2] {
                let tx = ((x - grid7[0]) / step7).clamp(0.0, 6.0);
                let ty = ((y - grid7[0]) / step7).clamp(0.0, 6.0);
                let (ix, iy) = ((tx.floor() as usize).min(5), (ty.floor() as usize).min(5));
                let (fx, fy) = (tx - ix as f32, ty - iy as f32);
                let g = |a: usize, b: usize| tilts[b * 7 + a];
                std::array::from_fn(|k| (g(ix, iy)[k] * (1.0 - fx) + g(ix + 1, iy)[k] * fx) * (1.0 - fy) + (g(ix, iy + 1)[k] * (1.0 - fx) + g(ix + 1, iy + 1)[k] * fx) * fy)
            };
            let sensor_planes: Vec<Vec<f32>> = blended
                .iter()
                .enumerate()
                .map(|(c, acc)| {
                    let mut moved = vec![0.0f32; cw * cw];
                    for y in o..o + frame_fine {
                        for x in o..o + frame_fine {
                            let d = tilt_at(x as f32, y as f32);
                            moved[y * cw + x] = stack::sample(acc, cw, cw, x as f32 - d[0] - gx, y as f32 - d[1] - gy - disp[c]);
                        }
                    }
                    to_sensor(&moved, cw, frame_fine, f)
                })
                .collect();
            let raw = sensor.expose(&sensor_planes, &mut rng);
            let q = stack::quality(&raw, &sensor);
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_multiple_of(50) {
                eprintln!("  fotogrammi {n}/{frames}  ({:.0} s)", started.elapsed().as_secs_f32());
            }
            (q, raw)
        })
        .collect();
    let (tilt_sum, spread_sum, tilt_n) = *tilt_rms.lock().unwrap();
    let tilt_global = (tilt_sum / tilt_n as f64).sqrt() as f32 / f as f32;
    let tilt_spread = (spread_sum / tilt_n as f64).sqrt() as f32 / f as f32;
    eprintln!("tilt globale RMS {tilt_global:.2} px, differenza sul disco RMS {tilt_spread:.2} px (sensore)");

    let mut order: Vec<usize> = (0..frames).collect();
    order.sort_by(|&a, &b| captured[b].0.total_cmp(&captured[a].0));
    let keep = ((frames as f32 * keep_share).round() as usize).max(1);
    let raws: Vec<Vec<u16>> = captured.into_iter().map(|c| c.1).collect();
    let stacked = stack::stack(&raws, &order, keep, &sensor, surface);
    eprintln!("stack: {} fotogrammi su {frames}, {} punti di allineamento, spostamento locale medio {:.2} px", stacked.kept, stacked.points, stacked.mean_local);

    // Outputs.
    tiff::write_f32(&run_dir.join("grezza.tif"), frame, frame, &stacked.planes).unwrap();
    tiff::write_f32(&run_dir.join("pari.tif"), frame, frame, &stacked.halves[0]).unwrap();
    tiff::write_f32(&run_dir.join("dispari.tif"), frame, frame, &stacked.halves[1]).unwrap();
    let example = sensor.develop(&raws[order[keep / 2]]);
    let (p_truth, p_frame, p_stack, p_diff) = (preview(&sampled, frame, frame), preview(&example, frame, frame), preview(&stacked.planes, frame, frame), preview(&diffraction, frame, frame));
    ps_io::save(&run_dir.join("grezza_anteprima.png"), &p_stack).unwrap();
    ps_io::save(&run_dir.join("fotogramma_esempio.png"), &p_frame).unwrap();
    ps_io::save(&run_dir.join("confronto.png"), &montage(&[p_truth, p_frame, p_stack, p_diff])).unwrap();
    let photometry = gain.peak_e / (0.85 * sensor.e_per_adu * sensor.max_adu());
    let meta = format!(
        r#"{{
  "riferimento": "{stem}",
  "soggetto": "{body}",
  "telescopio": {{ "nome": "{}", "descrizione": "{}", "apertura_mm": {}, "ostruzione": {}, "focale_mm": {} }},
  "camera": {{ "nome": "{}", "descrizione": "{}", "pixel_um": {}, "bayer": {}, "bit": {bits}, "gain": "{}", "elettroni_picco": {}, "rumore_lettura_e": {}, "e_per_adu": {:.4}, "offset_adu": {} }},
  "scala_arcsec_px": {sensor_arcsec:.5},
  "lambda_d_arcsec": {lambda_d:.4},
  "sovracampionamento": {f},
  "diametro_arcsec": {diam},
  "atmosfera": {{ "d_r0": {}, "quota_strato_alto_km": {}, "quota_suolo": {}, "scala_esterna_m": {}, "seeing_arcsec": {:.3} }},
  "tilt_globale_rms_px": {tilt_global:.3},
  "tilt_differenziale_rms_px": {tilt_spread:.3},
  "deriva_px": [{:.2}, {:.2}],
  "dispersione_arcsec": {dispersion},
  "fotogrammi": {frames},
  "tenuti": {},
  "punti_allineamento": {},
  "seed": {seed},
  "fotometria": {photometry:.5},
  "nota": "grezza ≈ fotometria × (verità ⊛ PSF dello stack): offset tolto e bilanciamento del bianco applicato; le verità sono nelle unità del riferimento linearizzato"
}}
"#,
        scope.name,
        scope.label,
        scope.aperture_mm,
        scope.obstruction,
        scope.focal_mm,
        cam.name,
        cam.label,
        cam.pixel_um,
        cam.bayer,
        gain.label,
        gain.peak_e,
        gain.read_e,
        sensor.e_per_adu,
        sensor.offset_adu,
        atm.d_r0,
        atm.high_km,
        atm.ground_share,
        atm.outer_m,
        0.98 * lambda_d * atm.d_r0,
        drift.0 / f as f32,
        drift.1 / f as f32,
        stacked.kept,
        stacked.points
    );
    std::fs::write(run_dir.join("meta.json"), meta).unwrap();

    // Calibration: the simulated raw next to a real one.
    eprintln!("analisi:");
    let sim_img = Image::new(frame, frame, stacked.planes.iter().map(|p| p.iter().map(|v| v.max(0.0)).collect()).collect(), 16);
    let sim_a = describe("simulata", &sim_img);
    let compare = args.str("compare", "");
    if !compare.is_empty() {
        let real = ml::load_any(Path::new(&compare)).expect("grezza vera illeggibile");
        let real_a = describe("vera", &real);
        if let (Some(s), Some(r)) = (spectrum(&sim_img, &sim_a), spectrum(&real, &real_a)) {
            let mut tsv = String::from("f\tsimulata\tvera\n");
            eprintln!("  spettro della luminanza sul disco (rispetto a f = 0.03):");
            eprintln!("    f (cicli/px)   simulata      vera");
            for fq in [0.03f32, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45] {
                eprintln!("    {fq:<12.2} {:>10.2e} {:>10.2e}", ml::at(&s, fq), ml::at(&r, fq));
            }
            for (fq, v) in &s {
                tsv.push_str(&format!("{fq:.4}\t{v:.4e}\t{:.4e}\n", ml::at(&r, *fq)));
            }
            std::fs::write(run_dir.join("spettro_confronto.tsv"), tsv).unwrap();
        }
    }
    eprintln!("fatto in {:.0} s → {}", started.elapsed().as_secs_f32(), run_dir.display());
}
