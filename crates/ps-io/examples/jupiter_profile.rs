//! Colour of Jupiter by latitude, from a NASA global map, as a Rust table for `photocolor_data.rs`:
//! `cargo run --release -p ps-io --example jupiter_profile -- map.tif planetographic|planetocentric NAME`
//! Rows are median RGB over longitude (robust to the Great Red Spot and the white ovals), averaged in
//! 2.5° bins of planetocentric latitude from −60° to +60°.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let img = ps_io::load(std::path::Path::new(&args[1])).unwrap();
    let graphic = args[2] == "planetographic";
    let name = &args[3];
    let (w, h) = (img.width(), img.height());
    let ch = img.channels();
    let f = 0.06487f64; // Jupiter's flattening
    const N: usize = 48;
    let mut sum = [[0.0f64; 3]; N];
    let mut count = [0.0f64; N];
    for row in 0..h {
        let lat_map = 90.0 - (row as f64 + 0.5) / h as f64 * 180.0;
        let lat = if graphic { (((1.0 - f).powi(2)) * lat_map.to_radians().tan()).atan().to_degrees() } else { lat_map };
        let bin = ((lat + 60.0) / 2.5).floor();
        if !(0.0..N as f64).contains(&bin) {
            continue;
        }
        // Median over the row of each channel; rows that are black (outside the map's coverage) skipped.
        let mut med = [0.0f64; 3];
        let mut valid = 0usize;
        for c in 0..3 {
            let mut v: Vec<f32> = ch[c][row * w..(row + 1) * w].to_vec();
            v.sort_by(|a, b| a.total_cmp(b));
            med[c] = v[w / 2] as f64;
            valid += (v[w / 2] > 0.02) as usize;
        }
        if valid < 3 {
            continue;
        }
        for c in 0..3 {
            sum[bin as usize][c] += med[c];
        }
        count[bin as usize] += 1.0;
    }
    println!("/// {name}: median RGB per 2.5° bin of planetocentric latitude, −60° … +60°.");
    println!("pub const {name}: [[f32; 3]; {N}] = [");
    for b in 0..N {
        if count[b] == 0.0 {
            println!("    [f32::NAN; 3],");
        } else {
            println!("    [{:.4}, {:.4}, {:.4}],", sum[b][0] / count[b], sum[b][1] / count[b], sum[b][2] / count[b]);
        }
    }
    println!("];");

    // The Great Red Spot's colour: the reddest compact patch at 14°–32° S (map latitude).
    let lat_of = |row: usize| 90.0 - (row as f64 + 0.5) / h as f64 * 180.0;
    let enc = |i: usize| [ch[0][i], ch[1][i], ch[2][i]];
    let red = |i: usize| {
        let [x, y, z] = enc(i);
        (x - z) / (0.2126 * x + 0.7152 * y + 0.0722 * z).max(1e-3)
    };
    let rows: Vec<usize> = (0..h).filter(|&y| (-32.0..=-14.0).contains(&lat_of(y))).collect();
    let (y0, y1) = (rows[0], rows[rows.len() - 1]);
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
                patch.push(enc(i));
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
}
