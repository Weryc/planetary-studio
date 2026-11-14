//! Learned noise reduction, step 2: train the network of `ps_core::denoise_net`.
//!
//! `cargo run --release -p ps-train --bin denoise_train -- data_dir weights.bin [steps]`
//!
//! Examples are made on the fly from the clean stacks of `denoise_data`: a 256² crop (mostly
//! on the subject), noise with a random level (subject SNR 60–3000), grain (white, or smoothed
//! by 0.3–1.1 px as resampling in the stacker does) and share proportional to brightness
//! (photon noise); a random sky level; sometimes the same sharpening applied to the clean and
//! the noisy crop (what a deconvolution or wavelets do before the noise reduction runs). The
//! noise is measured on the noisy crop with the app's own estimator, so the network learns to
//! trust exactly the number the app will give it. It learns on the central 128².
//!
//! Whole references are kept apart for checking (every fifth group): every 1000 steps the
//! error on them is printed next to the noisy input's and to the two classic methods (starlet
//! thresholds for planets, bilateral for Moon and Sun), and the weights are saved.
//! Uses the GPU (Metal) when there is one.
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use ps_core::denoise_net::{LAYERS, Net, SIGMA_SCALE, detail, normalise};
use ps_core::enhance::{NoiseMethod, NoiseParams};
use ps_core::Image;
use rayon::prelude::*;
use std::path::Path;

const CROP: usize = 256;
/// Side of the learning patch and examples per step (`PS_PATCH`, `PS_BATCH` to change them).
static PATCH_SIZE: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();

fn patch_side() -> usize {
    sizes().0
}

fn batch_len() -> usize {
    sizes().1
}

fn sizes() -> (usize, usize) {
    *PATCH_SIZE.get_or_init(|| {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        (env("PS_PATCH", 96), env("PS_BATCH", 16))
    })
}

struct Clean {
    group: String,
    w: usize,
    h: usize,
    y: Vec<f32>,
    /// Normalisation scale (as the app computes it).
    scale: f32,
    /// Median of the lit part: the subject's brightness, for the SNR.
    lit: f32,
}

fn load(dir: &str) -> Vec<Clean> {
    let index = std::fs::read_to_string(format!("{dir}/index.tsv")).expect("index.tsv");
    index
        .lines()
        .skip(1)
        .par_bridge()
        .filter_map(|line| {
            let c: Vec<&str> = line.split('\t').collect();
            let b = std::fs::read(Path::new(dir).join(c[0])).ok()?;
            let w = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
            let h = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
            if w < CROP || h < CROP {
                return None;
            }
            let y: Vec<f32> = b[8..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let scale = normalise(&y);
            let mut lit: Vec<f32> = y.iter().step_by(3).copied().filter(|&v| v > 0.15 * scale).collect();
            if lit.is_empty() {
                return None;
            }
            let k = lit.len() / 2;
            let lit = *lit.select_nth_unstable_by(k, f32::total_cmp).1;
            Some(Clean { group: c[1].to_string(), w, h, y, scale, lit })
        })
        .collect()
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }
    fn gauss(&mut self) -> f32 {
        (-2.0 * self.next().max(1e-12).ln()).sqrt() * (std::f32::consts::TAU * self.next()).cos()
    }
}

/// Separable Gaussian blur with clamped edges.
fn blur(p: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.05 {
        return p.to_vec();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / s).collect();
    let mut tmp = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            tmp[y * w + x] = (-r..=r).map(|d| k[(d + r) as usize] * p[y * w + (x as isize + d).clamp(0, w as isize - 1) as usize]).sum();
        }
    }
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = (-r..=r).map(|d| k[(d + r) as usize] * tmp[(y as isize + d).clamp(0, h as isize - 1) as usize * w + x]).sum();
        }
    }
    out
}

/// One example: (noisy crop, clean crop, σ measured), all CROP², not normalised.
fn example(c: &Clean, rng: &mut Rng) -> (Vec<f32>, Vec<f32>, f32) {
    // Mostly on the subject; sometimes anywhere (limb, sky, shadows).
    let mut origin = (0, 0);
    for attempt in 0..8 {
        origin = ((rng.next() * (c.w - CROP + 1) as f32) as usize, (rng.next() * (c.h - CROP + 1) as f32) as usize);
        let mean: f32 = (0..CROP).step_by(8).flat_map(|y| (0..CROP).step_by(8).map(move |x| (x, y))).map(|(x, y)| c.y[(origin.1 + y) * c.w + origin.0 + x]).sum::<f32>() / 1024.0;
        if mean > 0.25 * c.lit || attempt == 7 || rng.next() < 0.1 {
            break;
        }
    }
    let clean: Vec<f32> = (0..CROP * CROP).map(|i| c.y[(origin.1 + i / CROP) * c.w + origin.0 + i % CROP]).collect();

    let sigma = c.lit / (60.0f32.ln() + rng.next() * (3000.0f32 / 60.0).ln()).exp();
    let photon = rng.range(0.0, 0.8);
    let grain = if rng.next() < 0.3 { 0.0 } else { rng.range(0.3, 1.1) };
    let field: Vec<f32> = (0..CROP * CROP).map(|_| rng.gauss()).collect();
    let field = blur(&field, CROP, CROP, grain);
    let sd = (field.iter().map(|v| v * v).sum::<f32>() / field.len() as f32).sqrt().max(1e-12);
    let sky = rng.range(0.0, 0.03) * c.scale;
    let mut noisy: Vec<f32> = clean.iter().zip(&field).map(|(&s, &g)| s + sky + sigma * ((1.0 - photon) + photon * s.max(0.0) / c.lit).sqrt() * g / sd).collect();
    let mut clean: Vec<f32> = clean.iter().map(|s| s + sky).collect();

    if rng.next() < 0.4 {
        // The same sharpening on both: x + a·(x − G_r x).
        let (a, r) = (rng.range(0.3, 3.0), rng.range(0.6, 2.0));
        for p in [&mut noisy, &mut clean] {
            let b = blur(p, CROP, CROP, r);
            p.iter_mut().zip(&b).for_each(|(v, b)| *v += a * (*v - b));
        }
    }
    let measured = ps_core::wavelet::estimate_noise(&noisy, CROP, CROP);
    (noisy, clean, measured)
}

/// Central PATCH² of a CROP² plane, at offset `o`.
fn patch(p: &[f32], o: (usize, usize)) -> impl Iterator<Item = f32> + '_ {
    (0..patch_side() * patch_side()).map(move |i| p[(o.1 + i / patch_side()) * CROP + o.0 + i % patch_side()])
}

/// The three input planes of a patch (luminance, detail in σ, σ plane), normalised; the detail
/// is taken on the whole crop so the patch edges see real neighbours, as tiles do in the app.
fn inputs(noisy: &[f32], sigma: f32, scale: f32, o: (usize, usize)) -> Vec<f32> {
    let y: Vec<f32> = noisy.iter().map(|v| v / scale).collect();
    let d = detail(&y, CROP, CROP, sigma);
    let mut input: Vec<f32> = patch(&y, o).collect();
    input.extend(patch(&d, o));
    input.extend(std::iter::repeat_n(SIGMA_SCALE * sigma, patch_side() * patch_side()));
    input
}

/// A batch: input (B,3,P,P) and target (B,1,P,P), normalised.
fn batch(data: &[Clean], train: &[usize], seed: u64) -> (Vec<f32>, Vec<f32>) {
    let parts: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_len())
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng::new(seed * 1000 + i as u64);
            let c = &data[train[(rng.next() * train.len() as f32) as usize % train.len()]];
            let (noisy, clean, sigma) = example(c, &mut rng);
            let o = (32 + (rng.next() * 64.0) as usize, 32 + (rng.next() * 64.0) as usize);
            let input = inputs(&noisy, sigma / c.scale, c.scale, o);
            (input, patch(&clean, o).map(|v| v / c.scale).collect())
        })
        .collect();
    let mut x = Vec::with_capacity(batch_len() * 3 * patch_side() * patch_side());
    let mut t = Vec::with_capacity(batch_len() * patch_side() * patch_side());
    for (i, c) in parts {
        x.extend(i);
        t.extend(c);
    }
    (x, t)
}

struct Model {
    convs: Vec<Conv2d>,
}

impl Model {
    fn new(vb: &VarBuilder) -> candle_core::Result<Model> {
        let convs = LAYERS
            .iter()
            .enumerate()
            .map(|(i, &(cin, cout, stride))| candle_nn::conv2d(cin, cout, 3, Conv2dConfig { padding: 1, stride, ..Default::default() }, vb.pp(format!("c{i}"))))
            .collect::<candle_core::Result<_>>()?;
        Ok(Model { convs })
    }

    /// The same graph as `Net::noise`.
    fn noise(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let c = &self.convs;
        let r = |i: usize, t: &Tensor| c[i].forward(t)?.relu();
        let up = |t: &Tensor| {
            let (_, _, h, w) = t.dims4()?;
            t.upsample_nearest2d(h * 2, w * 2)
        };
        let e0 = r(1, &r(0, x)?)?;
        let e1 = r(3, &r(2, &e0)?)?;
        let m = r(6, &r(5, &r(4, &e1)?)?)?;
        let u1 = (r(7, &up(&m)?)? + &e1)?;
        let c1 = r(8, &u1)?;
        let u0 = (r(9, &up(&c1)?)? + &e0)?;
        let c0 = r(11, &r(10, &u0)?)?;
        c[12].forward(&c0)
    }

    /// Noisy input minus the noise, which the network gives in units of σ (as `Net::noise`).
    fn clean(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let sigma = (x.narrow(1, 2, 1)? / SIGMA_SCALE as f64)?;
        x.narrow(1, 0, 1)? - (self.noise(x)? * sigma)?
    }

    fn bytes(&self) -> candle_core::Result<Vec<u8>> {
        let layers = self
            .convs
            .iter()
            .map(|c| Ok((c.weight().flatten_all()?.to_vec1::<f32>()?, c.bias().expect("bias").to_vec1::<f32>()?)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Net::to_bytes(&layers))
    }
}

/// Fixed check set from the held-out references: (input with σ plane, clean, noisy crop
/// as an image for the classic methods, scale, offset).
struct Check {
    input: Vec<f32>,
    clean: Vec<f32>,
    noisy: Vec<f32>,
    scale: f32,
    offset: (usize, usize),
    planet: bool,
}

fn checks(data: &[Clean], held: &[usize]) -> Vec<Check> {
    (0..240)
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng::new(0xC0FFEE + i as u64);
            let c = &data[held[i % held.len()]];
            let (noisy, clean, sigma) = example(c, &mut rng);
            let o = (64, 64);
            let input = inputs(&noisy, sigma / c.scale, c.scale, o);
            let planet = !(c.group.starts_with("moon") || c.group.starts_with("sdo") || c.group.starts_with("sun"));
            Check { input, clean: patch(&clean, o).map(|v| v / c.scale).collect(), noisy, scale: c.scale, offset: o, planet }
        })
        .collect()
}

fn rms(a: impl Iterator<Item = f32>, b: &[f32]) -> f64 {
    let (s, n) = a.zip(b).fold((0.0f64, 0usize), |(s, n), (x, y)| (s + ((x - y) as f64).powi(2), n + 1));
    (s / n as f64).sqrt()
}

/// Mean RMS error on the check set: (noisy, network, classic method for the subject).
fn evaluate(model: &Model, set: &[Check], dev: &Device, classic: &mut Option<(f64, f64)>) -> candle_core::Result<(f64, f64, f64, f64)> {
    let mut net = 0.0;
    for chunk in set.chunks(batch_len()) {
        let x: Vec<f32> = chunk.iter().flat_map(|c| c.input.iter().copied()).collect();
        let x = Tensor::from_vec(x, (chunk.len(), 3, patch_side(), patch_side()), dev)?;
        let n = model.clean(&x)?.flatten_all()?.to_vec1::<f32>()?;
        for (k, c) in chunk.iter().enumerate() {
            let out = (0..patch_side() * patch_side()).map(|i| n[k * patch_side() * patch_side() + i]);
            net += rms(out, &c.clean);
        }
    }
    let noisy: f64 = set.iter().map(|c| rms(c.input[..patch_side() * patch_side()].iter().copied(), &c.clean)).sum();
    let (planet, lunar) = *classic.get_or_insert_with(|| {
        let run = |method| -> f64 {
            set.par_iter()
                .map(|c| {
                    let img = Image::new(CROP, CROP, vec![c.noisy.clone()], 16);
                    let out = ps_core::enhance::denoise(&img, &NoiseParams { method, strength: 1.0, detail: 0.3, chroma: 0, sky: 0.0 });
                    rms(patch(&out.channels()[0], c.offset).map(|v| v / c.scale), &c.clean)
                })
                .sum()
        };
        (run(NoiseMethod::Planetary), run(NoiseMethod::Lunar))
    });
    let n = set.len() as f64;
    Ok((noisy / n, net / n, planet / n, lunar / n))
}

fn main() -> candle_core::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("data dir");
    let out = args.next().expect("weights.bin");
    let steps: usize = args.next().map_or(30_000, |s| s.parse().expect("steps"));
    let dev = if std::env::var_os("PS_CPU").is_some() { Device::Cpu } else { Device::new_metal(0).unwrap_or(Device::Cpu) };
    eprintln!("dispositivo: {dev:?}");

    let started = std::time::Instant::now();
    let data = load(&dir);
    let mut groups: Vec<&str> = data.iter().map(|c| c.group.as_str()).collect();
    groups.sort();
    groups.dedup();
    let held_groups: Vec<&str> = groups.iter().copied().skip(2).step_by(5).collect();
    let (held, train): (Vec<usize>, Vec<usize>) = (0..data.len()).partition(|&i| held_groups.contains(&data[i].group.as_str()));
    eprintln!("{} stack ({} per l'addestramento, {} tenuti da parte: {}) in {:.0} s", data.len(), train.len(), held.len(), held_groups.join(", "), started.elapsed().as_secs_f32());
    let set = checks(&data, &held);
    let t = std::time::Instant::now();
    let _ = batch(&data, &train, 0);
    eprintln!("un lotto di esempi in {:.3} s", t.elapsed().as_secs_f32());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(&vb)?;
    // The last layer starts at zero: at first the network changes nothing.
    for (name, var) in varmap.data().lock().unwrap().iter() {
        if name.starts_with(&format!("c{}.", LAYERS.len() - 1)) {
            var.set(&var.zeros_like()?)?;
        }
    }
    let base_lr = 1e-3;
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), ParamsAdamW { lr: base_lr, weight_decay: 1e-5, ..Default::default() })?;

    // Examples are made on other threads while the GPU trains.
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<f32>, Vec<f32>)>(6);
    {
        let train = train.clone();
        let data = std::sync::Arc::new(data);
        std::thread::spawn(move || {
            for step in 0.. {
                if tx.send(batch(&data, &train, step as u64 + 1)).is_err() {
                    break;
                }
            }
        });
    }

    let mut classic = None;
    let mut running = 0.0f64;
    let t0 = std::time::Instant::now();
    for step in 1..=steps {
        // Short warm-up, then cosine decay.
        let p = step as f64 / steps as f64;
        let lr = base_lr * (step as f64 / 300.0).min(1.0) * (0.02 + 0.98 * 0.5 * (1.0 + (std::f64::consts::PI * p).cos()));
        opt.set_learning_rate(lr);
        let (x, t) = rx.recv().expect("examples");
        let x = Tensor::from_vec(x, (batch_len(), 3, patch_side(), patch_side()), &dev)?;
        let t = Tensor::from_vec(t, (batch_len(), 1, patch_side(), patch_side()), &dev)?;
        let pred = model.clean(&x)?;
        // In units of each example's σ: clean and noisy stacks count the same.
        let sigma = (x.narrow(1, 2, 1)? / SIGMA_SCALE as f64)?;
        let loss = ((pred - t)? / sigma)?.abs()?.mean_all()?;
        opt.backward_step(&loss)?;
        running += loss.to_scalar::<f32>()? as f64;
        if step <= 10 {
            eprintln!("passo {step}: {:.2} s dall'inizio", t0.elapsed().as_secs_f32());
        }
        if step % 200 == 0 {
            eprintln!("passo {step:>6}  L1 {:.6}  lr {lr:.2e}  {:.0} passi/min", running / 200.0, step as f64 / t0.elapsed().as_secs_f64() * 60.0);
            running = 0.0;
        }
        if step % 1000 == 0 || step == steps {
            let (n, m, pl, lu) = evaluate(&model, &set, &dev, &mut classic)?;
            eprintln!("  verifica (RMS ×1000): rumoroso {:.3}  rete {:.3}  pianeti {:.3}  bilaterale {:.3}", n * 1e3, m * 1e3, pl * 1e3, lu * 1e3);
            std::fs::write(&out, model.bytes()?).expect("weights");
        }
    }
    // Per subject, at the end.
    for (label, planet) in [("pianeti", true), ("Luna e Sole", false)] {
        let part: Vec<Check> = set.iter().filter(|c| c.planet == planet).map(|c| Check { input: c.input.clone(), clean: c.clean.clone(), noisy: c.noisy.clone(), ..*c }).collect();
        if part.is_empty() {
            continue;
        }
        let (n, m, pl, lu) = evaluate(&model, &part, &dev, &mut None)?;
        eprintln!("{label} ({} esempi): rumoroso {:.3}  rete {:.3}  starlet {:.3}  bilaterale {:.3}", part.len(), n * 1e3, m * 1e3, pl * 1e3, lu * 1e3);
    }
    Ok(())
}
