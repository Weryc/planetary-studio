//! Natural-colour Jupiter profile from three calibrated filter maps (Hubble OPAL FITS):
//! `jupiter_profile_rgb NAME red.fits green.fits blue.fits scale_r scale_g scale_b [flip]`
//! The FITS values times the scale are I/F (reflectance relative to a white Lambertian surface), so a
//! Sun-lit white surface is white in all three: no white balance to guess. The bins are the same as
//! `jupiter_profile` (planetographic rows, converted to planetocentric).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let name = &a[1];
    let scale: Vec<f32> = a[5..8].iter().map(|v| v.parse().unwrap()).collect();
    let flip = a.get(8).is_some_and(|v| v == "flip");
    let maps: Vec<ps_core::Image> = a[2..5].iter().map(|p| ps_io::load(std::path::Path::new(p)).unwrap()).collect();
    let (w, h) = (maps[0].width(), maps[0].height());
    let plane = |k: usize| -> Vec<f32> { maps[k].channels()[0].iter().map(|v| v * scale[k]).collect() };
    let (r, g, b) = (plane(0), plane(1), plane(2));
    // Map rows → planetographic latitude, 10 px/deg, ±79.8° covered.
    let lat_of = |row: usize| -> f64 {
        let row = if flip { h - 1 - row } else { row };
        90.0 - (row as f64 + 0.5) / h as f64 * 180.0
    };
    // Exposure: the median green I/F of the covered rows → linear 0.35 (a photo-like level).
    let mut gv: Vec<f32> = (0..h).filter(|&y| lat_of(y).abs() < 70.0).flat_map(|y| g[y * w..(y + 1) * w].iter().copied()).collect();
    gv.sort_by(|p, q| p.total_cmp(q));
    let k = 0.35 / gv[gv.len() / 2].max(1e-6);
    let oetf = |v: f32| {
        let v = (v * k).clamp(0.0, 1.0);
        if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
    };
    let f = 0.06487f64;
    const N: usize = 48;
    let mut sum = [[0.0f64; 3]; N];
    let mut cnt = [0.0f64; N];
    for row in 0..h {
        let lat_g = lat_of(row);
        if lat_g.abs() > 79.0 {
            continue;
        }
        let lat = ((1.0 - f).powi(2) * lat_g.to_radians().tan()).atan().to_degrees();
        let bin = ((lat + 60.0) / 2.5).floor();
        if !(0.0..N as f64).contains(&bin) {
            continue;
        }
        let med = |p: &[f32]| {
            let mut v: Vec<f32> = p[row * w..(row + 1) * w].iter().map(|x| oetf(*x)).collect();
            v.sort_by(|x, y| x.total_cmp(y));
            v[w / 2] as f64
        };
        let m = [med(&r), med(&g), med(&b)];
        for c in 0..3 {
            sum[bin as usize][c] += m[c];
        }
        cnt[bin as usize] += 1.0;
    }
    eprintln!("exposure k = {k:.3}");
    println!("/// {name}: median RGB per 2.5° bin of planetocentric latitude, −60° … +60°.");
    println!("pub const {name}: [[f32; 3]; {N}] = [");
    for bn in 0..N {
        println!("    [{:.4}, {:.4}, {:.4}],", sum[bn][0] / cnt[bn], sum[bn][1] / cnt[bn], sum[bn][2] / cnt[bn]);
    }
    println!("];");

    // The Great Red Spot's colour: the reddest compact patch (r − b over luminance) at 14°–32° S.
    let enc: Vec<[f32; 3]> = (0..w * h).map(|i| [oetf(r[i]), oetf(g[i]), oetf(b[i])]).collect();
    let red = |i: usize| {
        let [x, y, z] = enc[i];
        (x - z) / (0.2126 * x + 0.7152 * y + 0.0722 * z).max(1e-3)
    };
    let rows: Vec<usize> = (0..h).filter(|&y| (-32.0..=-14.0).contains(&lat_of(y))).collect();
    let (y0, y1) = (rows[0], rows[rows.len() - 1]);
    // Box-blur redness over 12×12 pixels (1.2°) to find the patch, not a noisy pixel.
    let mut best = (f32::MIN, 0usize, 0usize);
    let mut all: Vec<f32> = Vec::new();
    for y in (y0..y1).step_by(3) {
        for x in (0..w - 12).step_by(3) {
            let mut sum = 0.0;
            for dy in 0..12 {
                for dx in 0..12 {
                    sum += red((y + dy) * w + x + dx);
                }
            }
            let m = sum / 144.0;
            all.push(m);
            if m > best.0 {
                best = (m, x + 6, y + 6);
            }
        }
    }
    all.sort_by(|p, q| p.total_cmp(q));
    let median = all[all.len() / 2];
    let (px, py) = (best.1, best.2);
    let mut patch: Vec<[f32; 3]> = Vec::new();
    for y in py.saturating_sub(35)..(py + 35).min(h) {
        for x in px.saturating_sub(120)..(px + 120).min(w) {
            let i = y * w + x;
            if red(i) > median + 0.6 * (best.0 - median) {
                patch.push(enc[i]);
            }
        }
    }
    let mut grs = [0.0f32; 3];
    for c in 0..3 {
        let mut v: Vec<f32> = patch.iter().map(|p| p[c]).collect();
        v.sort_by(|p, q| p.total_cmp(q));
        grs[c] = v[v.len() / 2];
    }
    eprintln!("GRS at column {px}, row {py} (lat {:.1}), redness {:.3} vs band median {median:.3}, {} px", lat_of(py), best.0, patch.len());
    println!("pub const {name}_GRS: [f32; 3] = [{:.4}, {:.4}, {:.4}];", grs[0], grs[1], grs[2]);
    // A small preview of the composite, orientation check: rows top = north if the map is right way up.
    let (pw, ph) = (900usize, 450usize);
    let mut out = vec![vec![0.0f32; pw * ph]; 3];
    for y in 0..ph {
        for x in 0..pw {
            let (sx, sy) = (x * w / pw, y * h / ph);
            let sy = if flip { h - 1 - sy } else { sy };
            for (c, p) in [&r, &g, &b].into_iter().enumerate() {
                out[c][y * pw + x] = oetf(p[sy * w + sx]);
            }
        }
    }
    ps_io::save(std::path::Path::new(&std::env::var("PREVIEW").unwrap_or("/dev/null.png".into())), &ps_core::Image::new(pw, ph, out, 16)).ok();
}
