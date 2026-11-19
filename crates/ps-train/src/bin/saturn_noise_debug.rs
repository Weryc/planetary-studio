//! What the Saturn noise reduction does to one image: each zone's measured band power next to
//! the estimated noise and the gain it gets, and the image reduced at a few strengths.
//!
//! `cargo run --release -p ps-train --bin saturn_noise_debug -- image.tif out_dir [strength…]`
#[path = "../../../ps-io/examples/ml_common/mod.rs"]
mod ml;

use ps_core::Image;
use ps_core::curve_net::{BANDS, band_centre};
use ps_core::saturn;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let img = ps_io::load(Path::new(&args[0])).unwrap();
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out).unwrap();
    let strengths: Vec<f32> = args[2..].iter().filter_map(|s| s.parse().ok()).collect();
    let a = ps_core::analysis::analyze(&img);
    println!("{}×{} {:?} colore {} disco {:?}", img.width(), img.height(), a.kind, !img.is_mono(), a.disk.map(|d| (d.cx, d.cy, d.radius)));
    let z = saturn::zones(&img).expect("zone");
    println!("globo {:?} ampiezza anelli {:.0} px", z.globe, z.span);
    let (m, f) = saturn::measure(&img, &a).expect("misura");
    println!("riquadro {} px a ({}, {}), pixel globo {} anelli {} cielo {}, anelli/globo luce {:.2}, cielo {}", f.side, f.x0, f.y0, f.planet.len(), f.rings.len(), f.sky.len(), f.ring_level, m.sky_note.unwrap_or("ok"));
    let x = saturn::noise_features(&m, &f);
    // The models are crate-private: go through the public denoise and print the predictions
    // via the exported check path.
    let (pl, pc, rl, rc) = saturn::noise_estimates(&x).expect("reti");
    let (y, chroma) = ps_core::noise_net::planes(&f.crop);
    let cpow = |px: &[usize]| chroma.as_ref().map(|[cr, cb]| f.power(cr, px).iter().zip(f.power(cb, px)).map(|(a, b)| (10f32.powf(*a) + 10f32.powf(b)).log10()).collect::<Vec<f32>>());
    let ring_px: &[usize] = if m.rings.is_some() { &f.rings } else { &f.planet };
    for (zone, px, nl, nc) in [("GLOBO", &f.planet[..], &pl, &pc), ("ANELLI", ring_px, &rl, &rc)] {
        let p = f.power(&y, px);
        let c = cpow(px);
        println!("\n{zone}: banda  periodo_px  potenza  rumore_stimato  rapporto(lin)  guadagno@1  | colore potenza rumore rapporto");
        for b in 2..BANDS {
            let fc = band_centre(b);
            let ratio = 10f32.powf(nl[b - 2] - p[b]);
            let cr = c.as_ref().map(|c| 10f32.powf(nc[b - 2] - c[b]));
            println!(
                "  {b:2} {:6.1} {:8.2} {:8.2} {:8.3} {:6.2} | {} {:7.2} {}",
                1.0 / fc,
                p[b],
                nl[b - 2],
                ratio,
                if fc < ps_core::noise_net::SMALLEST_NOISE_FREQUENCY { 1.0 } else { (1.0 - ratio).clamp(0.0, 1.0) },
                c.as_ref().map_or("-".into(), |c| format!("{:7.2}", c[b])),
                nc[b - 2],
                cr.map_or("-".into(), |r| format!("{r:7.3}"))
            );
        }
    }
    let lum = ml::lum(&img);
    let mut v: Vec<f32> = lum.iter().step_by(3).copied().collect();
    v.sort_by(f32::total_cmp);
    let top = v[v.len() * 999 / 1000].max(1e-6);
    let view = |im: &Image| {
        let c = im.clone();
        let ch: Vec<Vec<f32>> = c.channels().iter().map(|p| p.iter().map(|x| (x / top).clamp(0.0, 1.0).powf(1.0 / 2.2)).collect()).collect();
        Image::new(c.width(), c.height(), if ch.len() == 1 { vec![ch[0].clone(), ch[0].clone(), ch[0].clone()] } else { ch }, 16)
    };
    ps_io::save(&out.join("grezza.jpg"), &view(&img)).unwrap();
    // As in the app: the noise reduction, then the smart sharpening with Saturn's network curve.
    let mut an = a.clone();
    an.curve_net = ps_core::curve_net::suggestion(&img, &an, ps_core::capture::Body::Saturn).ok();
    let sp = ps_core::preset::auto_smart(an.kind, &an, ps_core::preset::AutoStrength::Normal);
    println!("\nnitidezza: {:?} forza {:.2} protezione bordo {:.2} colore {:?}", sp.curve, sp.strength, sp.limb_protection, sp.color);
    let subject = ps_core::mask::subject_and_moons(&img);
    // The app's full automatic pipeline, with the noise reduction at each strength.
    let auto = ps_core::preset::auto_pipeline(an.kind, &an);
    println!("pipeline automatica: deconv {} rumore {} {:?} forza {:.2} dettaglio {:.2} cielo {:.2} croma {} wavelet {} nitidezza {} {:?} forza {:.2} dering {}",
        auto.deconv.enabled, auto.denoise.enabled, auto.denoise.params.method, auto.denoise.params.strength, auto.denoise.params.detail, auto.denoise.params.sky, auto.denoise.params.chroma,
        auto.wavelet.enabled, auto.smart.enabled, auto.smart.params.curve, auto.smart.params.strength, auto.dering.enabled);
    let src = std::sync::Arc::new(img.clone());
    for s in std::iter::once(0.0).chain(strengths.iter().copied()) {
        let mut p = auto.clone();
        p.denoise.params.strength = s;
        let masks = ps_core::pipeline::masks_for(&p, &src);
        let aux = ps_core::pipeline::Aux { masks: &masks, ..Default::default() };
        let o = ps_core::pipeline::Processor::default().run(std::sync::Arc::clone(&src), ps_core::pipeline::next_generation(), &p, &aux, &|| false, &|_, _| {}).unwrap();
        ps_io::save(&out.join(format!("app_{s}.jpg")), &view(&o.image)).unwrap();
    }
    // A zoomed preview processes a crop (region + 64 px): the left ansa, as the app would.
    let (cx, cy, cw, ch) = (0usize, 40usize, 360usize.min(img.width()), 256usize.min(img.height() - 40));
    let crop = img.crop(cx, cy, cw, ch);
    let ca = ps_core::analysis::analyze(&crop);
    println!("ritaglio: {:?}, globo {:?}", ca.kind, saturn::zones(&crop).map(|z| (z.globe.cx + cx as f32, z.globe.cy + cy as f32, z.globe.rx)));
    for s in [1.0f32, 5.0] {
        let full = ps_core::noise_net::guided_denoise(&img, s).unwrap().crop(cx, cy, cw, ch);
        let part = ps_core::noise_net::guided_denoise(&crop, s).unwrap_or_else(|| crop.clone());
        ps_io::save(&out.join(format!("roi_intera_{s}.jpg")), &view(&full)).unwrap();
        ps_io::save(&out.join(format!("roi_ritaglio_{s}.jpg")), &view(&part)).unwrap();
    }
    // The app's pipeline on zoomed regions (ROI + 64 px margin), as the worker runs it, next
    // to the same region of the whole image: one cut through the globe, one on the right ansa.
    let (w, h) = (img.width(), img.height());
    let solo = std::env::var("SOLO_RUMORE").is_ok();
    let only_noise = |mut p: ps_core::pipeline::Pipeline| {
        if std::env::var("SENZA_RUMORE").is_ok() {
            p.denoise.enabled = false;
        }
        if std::env::var("SENZA_CIELO").is_ok() {
            p.denoise.params.sky = 0.0;
        }
        if std::env::var("NON_ADATTIVA").is_ok() {
            p.smart.params.adaptive = false;
        }
        if solo {
            let on = std::env::var("SOLO_RUMORE").unwrap_or_default();
            let keep = |n: &str| on.split(',').any(|s| s == n);
            (p.deconv.enabled, p.wavelet.enabled, p.smart.enabled, p.dering.enabled, p.contrast.enabled) =
                (p.deconv.enabled && keep("deconv"), p.wavelet.enabled && keep("wavelet"), p.smart.enabled && keep("smart"), p.dering.enabled && keep("dering"), p.contrast.enabled && keep("contrast"));
        }
        p
    };
    let (full5, smart_ref) = {
        let mut p = only_noise(auto.clone());
        p.denoise.params.strength = 5.0;
        let masks = ps_core::pipeline::masks_for(&p, &src);
        let aux = ps_core::pipeline::Aux { masks: &masks, ..Default::default() };
        let o = ps_core::pipeline::Processor::default().run(std::sync::Arc::clone(&src), ps_core::pipeline::next_generation(), &p, &aux, &|| false, &|_, _| {}).unwrap();
        println!("misure della nitidezza dall'intera: {}", o.smart_reference.is_some());
        (o.image, o.smart_reference)
    };
    ps_io::save(&out.join("intera_5.tif"), &full5).unwrap();
    for (name, x0, y0, x1, y1) in [("globo", 0usize, h / 8, w * 58 / 100, h * 7 / 8), ("ansa", w * 72 / 100, h / 6, w, h * 5 / 6)] {
        let roi = std::sync::Arc::new(img.crop(x0, y0, x1 - x0, y1 - y0));
        println!("ROI {name} {}×{}: tipo {:?}", roi.width(), roi.height(), ps_core::analysis::analyze(&roi).kind);
        let mut p = only_noise(auto.clone());
        p.denoise.params.strength = 5.0;
        // As the worker does: the whole image's masks, cropped to the region.
        let masks: Vec<_> = ps_core::pipeline::masks_for(&p, &src).into_iter().map(|(id, plane, g)| {
            let crop: Vec<f32> = (y0..y1).flat_map(|y| plane[y * w + x0..y * w + x1].to_vec()).collect();
            (id, std::sync::Arc::new(crop), g)
        }).collect();
        // As the worker does: the whole image's noise measures, placed at the region.
        let plan = ps_core::noise_net::plan(&img).map(|p| (std::sync::Arc::new(p), (x0, y0)));
        let reference = if std::env::var("SENZA_RIF").is_ok() { None } else { smart_ref.clone().map(|r| (r, 1)) };
        let aux = ps_core::pipeline::Aux { masks: &masks, noise_plan: plan, smart_reference: reference, ..Default::default() };
        let o = ps_core::pipeline::Processor::default().run(std::sync::Arc::clone(&roi), ps_core::pipeline::next_generation(), &p, &aux, &|| false, &|_, _| {}).unwrap();
        let whole = full5.crop(x0, y0, x1 - x0, y1 - y0);
        let (a, b) = (ml::lum(&whole), ml::lum(&o.image));
        let (rw, rh) = (x1 - x0, y1 - y0);
        let inner: Vec<f32> = (0..rw * rh).filter(|&i| (64..rw.saturating_sub(64)).contains(&(i % rw)) && (64..rh.saturating_sub(64)).contains(&(i / rw))).map(|i| (a[i] - b[i]).abs()).collect();
        let top = a.iter().copied().fold(0.0f32, f32::max).max(1e-6);
        for band in [64usize, 80, 96, 112] {
            let worst = (0..rw * rh).filter(|&i| (band..rw.saturating_sub(band)).contains(&(i % rw)) && (band..rh.saturating_sub(band)).contains(&(i / rw))).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max);
            println!("  a {band} px dai bordi: massima {:.4}", worst / top);
        }
        println!("ROI {name}: differenza dall'immagine intera (lontano dai bordi) media {:.4} massima {:.4} (della luce massima)", inner.iter().sum::<f32>() / inner.len().max(1) as f32 / top, inner.iter().copied().fold(0.0f32, f32::max) / top);
        ps_io::save(&out.join(format!("roi_{name}_ritaglio.jpg")), &view(&o.image)).unwrap();
        ps_io::save(&out.join(format!("roi_{name}_ritaglio.tif")), &o.image).unwrap();
        ps_io::save(&out.join(format!("roi_{name}_intera.jpg")), &view(&full5.crop(x0, y0, x1 - x0, y1 - y0))).unwrap();
    }
    for s in std::iter::once(0.0).chain(strengths) {
        let d = if s > 0.0 { ps_core::noise_net::guided_denoise(&img, s).unwrap() } else { img.clone() };
        ps_io::save(&out.join(format!("ridotta_{s}.jpg")), &view(&d)).unwrap();
        let strong = ps_core::smart::SmartParams { strength: std::env::var("FORZA").ok().and_then(|v| v.parse().ok()).unwrap_or(sp.strength), ..sp.clone() };
        let n = ps_core::smart::sharpen(&d, None, &strong, Some(&subject));
        ps_io::save(&out.join(format!("ridotta_{s}_nitida.jpg")), &view(&n)).unwrap();
    }
}
