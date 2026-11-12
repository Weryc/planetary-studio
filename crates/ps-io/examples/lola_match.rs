//! Prototype: fit a lunar picture onto the LOLA relief drawn for its time, and write a side-by-side:
//! `cargo run --release -p ps-io --example lola_match -- ldem_16.img out.png image.tif [yyyy-mm-dd-hhmm]`
use ps_core::capture::Moment;
use ps_core::lunarmap::{Dem, Reference, solve};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dem_path, out, path) = (&args[1], &args[2], std::path::Path::new(&args[3]));
    let info = ps_core::capture::from_name(path.file_stem().unwrap().to_str().unwrap());
    let when = args
        .get(4)
        .and_then(|t| ps_core::capture::from_name(t).time)
        .or(info.time)
        .unwrap_or(Moment { year: 2026, month: 1, day: 1, hour: 0, minute: 0, second: 0.0 });
    let t = std::time::Instant::now();
    let dem = Dem::load_pds(dem_path.as_ref()).unwrap();
    println!("DEM read in {:?}", t.elapsed());
    let t = std::time::Instant::now();
    let reference = Reference::new(&dem, &when, 2048);
    println!("reference drawn in {:?}", t.elapsed());
    // Self test: LOLA_SELF=u,v,rho,angle,mirror builds the picture from the reference itself.
    let (w, h, lum) = if let Ok(spec) = std::env::var("LOLA_SELF") {
        let q: Vec<f64> = spec.split(',').map(|v| v.parse().unwrap()).collect();
        let (u, v, rho, angle, mirrored) = (q[0], q[1], q[2], q[3], q[4] > 0.5);
        let or = ps_core::identify::Orientation { angle: angle as f32, mirrored, scale: 1.0 };
        let (w, h) = (785usize, 450usize);
        let (plain, n) = &reference.plain;
        let mut lum = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (bx, by) = or.body_offset(x as f64 - w as f64 / 2.0, y as f64 - h as f64 / 2.0);
                let (bx, by) = (u + bx / rho, v + by / rho);
                if bx * bx + by * by < 0.99 {
                    let (i, j) = (((bx + 1.0) / 2.0 * *n as f64) as usize, ((1.0 - by) / 2.0 * *n as f64) as usize);
                    lum[y * w + x] = plain[j.min(n - 1) * n + i.min(n - 1)];
                }
            }
        }
        (w, h, lum)
    } else {
        let img = ps_io::load(path).unwrap();
        (img.width(), img.height(), ps_core::color::working_planes(&img, ps_core::color::ColorMode::Luminance).swap_remove(0))
    };
    let t = std::time::Instant::now();
    // LOLA_START: take the disk and orientation Identifica finds (a picture with the limb in it) and
    // only polish them; otherwise search from scratch.
    let sol = if std::env::var("LOLA_START").is_ok() {
        let id = ps_core::identify::identify(&ps_io::load(path).unwrap(), ps_core::capture::Body::Moon, &when, None);
        let (Some((cx, cy, r)), Some(orientation)) = (id.disk, id.orientation) else {
            println!("no disk found");
            return;
        };
        let start = ps_core::identify::Placement { cx, cy, r, orientation };
        println!("start: centre ({cx:.0},{cy:.0}) r {r:.0} angle {:.1} mirrored {}", orientation.angle, orientation.mirrored);
        let (sol, before) = ps_core::lunarmap::refine_scan(&lum, w, h, &reference, start, std::env::var("LOLA_START").is_ok_and(|v| v == "scan"));
        println!("correlation before {before:.3}");
        Some(sol)
    } else {
        solve(&lum, w, h, &reference, true)
    };
    println!("solved in {:?}", t.elapsed());
    let Some(sol) = sol else {
        println!("no solution");
        return;
    };
    let p = sol.placement;
    println!("score {:.3} (runner-up {:.3}), disk centre ({:.0}, {:.0}) r {:.0} px, angle {:.1}°, mirrored {}", sol.score, sol.runner_up, p.cx, p.cy, p.r, p.orientation.angle, p.orientation.mirrored);
    // Side by side, at most 900 px wide each: the picture, and the reference drawn under the solution.
    let f = w.max(h).div_ceil(900).max(1);
    let (sw, sh) = ((w / f).max(1), (h / f).max(1));
    let small = if f > 1 { ps_core::image::downscale_plane(&lum, w, h, f) } else { lum.clone() };
    let top = small.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let (plain, n) = &reference.plain;
    let mut right = vec![0.0f32; sw * sh];
    for y in 0..sh {
        for x in 0..sw {
            let (dx, dy) = ((x as f64 + 0.5) * f as f64 - p.cx as f64, (y as f64 + 0.5) * f as f64 - p.cy as f64);
            let (bx, by) = p.orientation.body_offset(dx, dy);
            let (bx, by) = (bx / p.r as f64, by / p.r as f64);
            if bx * bx + by * by < 0.99 {
                let (i, j) = (((bx + 1.0) / 2.0 * *n as f64) as usize, ((1.0 - by) / 2.0 * *n as f64) as usize);
                right[y * sw + x] = plain[j.min(n - 1) * n + i.min(n - 1)];
            }
        }
    }
    let mut rgb: Vec<Vec<f32>> = vec![vec![0.0; sw * 2 * sh]; 3];
    for y in 0..sh {
        for x in 0..sw {
            let a = (small[y * sw + x] / top).clamp(0.0, 1.0).sqrt();
            let b = (right[y * sw + x] * 1.6).clamp(0.0, 1.0).sqrt();
            for c in rgb.iter_mut() {
                c[y * sw * 2 + x] = a;
                c[y * sw * 2 + sw + x] = b;
            }
        }
    }
    ps_io::save(out.as_ref(), &ps_core::Image::new(sw * 2, sh, rgb, 16)).unwrap();
}
