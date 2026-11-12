//! Photometric colour for Jupiter: the colour of its belts and zones brought towards a NASA map's.
//!
//! The star-field version of this asks a catalogue what colour each star is. A planet has no such
//! catalogue, but Jupiter's bands are the same at every longitude, so the *colour at each latitude*
//! is a fingerprint: from a NASA global map the median colour of every 2.5° band was kept (a few
//! hundred numbers, `photocolor_data.rs`), and so was the colour of the Great Red Spot. On the picture
//! the same is measured on the central strip of the disk (away from the limb, whose darkening is
//! colour-dependent) and the colour matrix that carries the picture's colours onto the map's is found:
//!
//! - **in linear light** (the picture's values are decoded with a gamma of 2.2, the maps' too), so the
//!   result doesn't depend on how stretched the picture is;
//! - **in two steps**: the neutral bands (zones, the high latitudes, whose reference colour is nearly
//!   grey) set the white balance, a diagonal gain; then a 3×3 matrix, pulled towards "no change", sets
//!   the saturation and hue from the belts and, when the picture shows it, the Red Spot;
//! - **with the planet's tilt** towards us (Meeus ch. 43, given by the caller from the capture time)
//!   taken into the latitude of every pixel: when the north pole leans towards us the equator
//!   lies below the disk's centre, by up to 3.5° of latitude.
//!
//! Brightness is untouched: the matrix is applied and the luminance put back, so only the colours
//! change. Which way is north is not known from the picture, so both are tried and the better one
//! kept (the Red Spot, in the southern hemisphere, helps to tell).

use crate::Image;
use crate::color::luma;
use crate::identify::{PlanetShape, planet_shape};
use crate::photocolor_data::{CASSINI_2000, CASSINI_2000_GRS, HUBBLE_2019, HUBBLE_2019_GRS, HUBBLE_2021, HUBBLE_2021_GRS};
use serde::{Deserialize, Serialize};

/// Bins of 2.5° of planetocentric latitude from −60° to +60°.
const BINS: usize = 48;
/// Jupiter's flattening.
const FLATTENING: f64 = 0.06487;
/// Fewest pixels in a band for it to count.
const MIN_PIXELS: usize = 40;
/// How strongly the matrix is held to the identity (relative to the data's weight).
const RIDGE: f64 = 0.35;
/// The transfer curve assumed for pictures and maps (sRGB is close to a 2.2 power).
const GAMMA: f32 = 2.2;

fn linear(v: f32) -> f32 {
    v.max(0.0).powf(GAMMA)
}

fn encode(v: f32) -> f32 {
    v.max(0.0).powf(1.0 / GAMMA)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reference {
    /// Cassini narrow-angle camera map, December 2000 (colours close to the eye's, lively).
    #[default]
    Cassini2000,
    /// Hubble OPAL 2019 in red, green and blue filters (631, 547, 467 nm) calibrated to reflectance:
    /// physical colours, pale. (`HubbleNatural2019` was its name in the first version.)
    #[serde(alias = "HubbleNatural2019")]
    Hubble2019,
    /// The same from Hubble OPAL 2021 (631, 502, 467 nm).
    Hubble2021,
}

impl Reference {
    pub const ALL: [Reference; 3] = [Reference::Cassini2000, Reference::Hubble2019, Reference::Hubble2021];

    pub fn label(self) -> &'static str {
        match self {
            Reference::Cassini2000 => "Cassini 2000",
            Reference::Hubble2019 => "Hubble 2019",
            Reference::Hubble2021 => "Hubble 2021",
        }
    }

    /// Band colours and the Red Spot's colour, as stored (encoded).
    fn data(self) -> (&'static [[f32; 3]; BINS], [f32; 3]) {
        match self {
            Reference::Cassini2000 => (&CASSINI_2000, CASSINI_2000_GRS),
            Reference::Hubble2019 => (&HUBBLE_2019, HUBBLE_2019_GRS),
            Reference::Hubble2021 => (&HUBBLE_2021, HUBBLE_2021_GRS),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotoColorParams {
    pub reference: Reference,
    /// 0 = untouched, 1 = fully carried onto the reference.
    pub strength: f32,
    /// Latitude of the Earth seen from Jupiter, degrees (north pole towards us is positive): filled in
    /// by the application from the capture time, 0 when the time is unknown.
    pub tilt: f32,
}

impl Default for PhotoColorParams {
    fn default() -> Self {
        Self { reference: Reference::default(), strength: 1.0, tilt: 0.0 }
    }
}

impl PhotoColorParams {
    pub fn neutral() -> Self {
        Self { strength: 0.0, ..Self::default() }
    }

    pub fn is_neutral(&self) -> bool {
        self.strength <= 0.0
    }
}

/// What the fit found.
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    /// Maps the picture's linear colours (sky removed) onto the reference's.
    pub matrix: [[f32; 3]; 3],
    /// The white-balance gains of the first step (already inside `matrix`).
    pub gains: [f32; 3],
    /// The sky level removed first, per channel (in the picture's own values).
    pub sky: [f32; 3],
    /// Bands that took part.
    pub bands: usize,
    /// Mean colour distance (luminance-normalised, linear) between the picture's bands and the map's,
    /// before and after.
    pub before: f32,
    pub after: f32,
    /// The disk's `+b` side (the planet's minor-axis direction) is north.
    pub north_at_plus_b: bool,
    /// The Red Spot was found and used.
    pub spot: bool,
}

/// A latitude band's median RGB (encoded, sky removed) and pixel count, if it has enough pixels.
type Band = Option<([f32; 3], usize)>;

/// What the picture's planet looks like, found once.
struct Geometry {
    shape: PlanetShape,
    w: usize,
    h: usize,
}

/// Planetocentric latitude (degrees) of the point seen at (`a`, `bb`) in the disk, in units of the
/// semi-axes, with the planet tilted by `tilt` (radians; north pole towards us positive) and `bb`'s
/// positive side north or not.
fn latitude(a: f64, bb: f64, tilt: f64, north_at_plus_b: bool) -> Option<f64> {
    let y = if north_at_plus_b { bb } else { -bb };
    let z2 = 1.0 - a * a - y * y;
    if z2 <= 0.0 {
        return None;
    }
    // y = sinφ cosB − z sinB on a sphere of unit radius, z = cosφ cosλ.
    let sin_beta = ((y + z2.sqrt() * tilt.sin()) / tilt.cos()).clamp(-1.0, 1.0);
    let beta = sin_beta.asin();
    Some(((1.0 - FLATTENING) * beta.tan()).atan().to_degrees())
}

impl Geometry {
    fn of(img: &Image) -> Option<Geometry> {
        let (w, h) = (img.width(), img.height());
        let [r, g, b] = img.channels() else { return None };
        let lum: Vec<f32> = (0..w * h).map(|i| luma([r[i], g[i], b[i]])).collect();
        let shape = planet_shape(&lum, w, h)?;
        // A planet that is too small has no bands to speak of; one that isn't oblate isn't Jupiter.
        (shape.major >= 25.0 && shape.minor / shape.major >= 0.85).then_some(Geometry { shape, w, h })
    }

    /// Visits the pixels of the disk: (index, a, bb) in semi-axis units.
    fn disk(&self, mut visit: impl FnMut(usize, f64, f64)) {
        let (s, c) = self.shape.angle.sin_cos();
        let (major, minor) = (self.shape.major as f64, self.shape.minor as f64);
        let reach = (major * 1.1) as isize;
        for y in (self.shape.cy as isize - reach).max(0)..(self.shape.cy as isize + reach).min(self.h as isize) {
            for x in (self.shape.cx as isize - reach).max(0)..(self.shape.cx as isize + reach).min(self.w as isize) {
                let (dx, dy) = (x as f64 - self.shape.cx as f64, y as f64 - self.shape.cy as f64);
                visit(y as usize * self.w + x as usize, (dx * c + dy * s) / major, (-dx * s + dy * c) / minor);
            }
        }
    }
}

/// Median RGB of each latitude band on the planet's central strip.
fn measure(img: &Image, geo: &Geometry, sky: [f32; 3], tilt: f64, north_at_plus_b: bool) -> Vec<Band> {
    let [r, g, b] = img.channels() else { return Vec::new() };
    let mut values: Vec<[Vec<f32>; 3]> = (0..BINS).map(|_| [Vec::new(), Vec::new(), Vec::new()]).collect();
    geo.disk(|i, a, bb| {
        // The central strip, well inside the limb.
        if a.abs() > 0.5 || a * a + bb * bb > 0.85 {
            return;
        }
        let Some(lat) = latitude(a, bb, tilt, north_at_plus_b) else { return };
        let bin = ((lat + 60.0) / 2.5).floor();
        if !(0.0..BINS as f64).contains(&bin) {
            return;
        }
        let v = &mut values[bin as usize];
        v[0].push(r[i] - sky[0]);
        v[1].push(g[i] - sky[1]);
        v[2].push(b[i] - sky[2]);
    });
    values
        .into_iter()
        .map(|mut v| {
            let n = v[0].len();
            (n >= MIN_PIXELS).then(|| {
                let mut m = [0.0f32; 3];
                for (k, plane) in v.iter_mut().enumerate() {
                    plane.sort_by(|p, q| p.total_cmp(q));
                    m[k] = plane[n / 2];
                }
                (m, n)
            })
        })
        .collect()
}

/// Redness of a colour: (R − B) over luminance.
fn redness(c: [f32; 3]) -> f32 {
    (c[0] - c[2]) / luma(c).max(1e-3)
}

/// The Great Red Spot: the reddest compact patch between 16° and 30° south, if there is a clear
/// one. Its median colour (encoded, sky removed).
fn find_spot(img: &Image, geo: &Geometry, sky: [f32; 3], tilt: f64, north_at_plus_b: bool) -> Option<[f32; 3]> {
    let [r, g, b] = img.channels() else { return None };
    let (w, h) = (geo.w, geo.h);
    // The region's pixels with their redness, in a bounding box of the planet.
    let reach = (geo.shape.major * 1.1) as isize;
    let (x0, y0) = ((geo.shape.cx as isize - reach).max(0) as usize, (geo.shape.cy as isize - reach).max(0) as usize);
    let (x1, y1) = (((geo.shape.cx as isize + reach) as usize).min(w), ((geo.shape.cy as isize + reach) as usize).min(h));
    let (bw, bh) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    if bw < 8 || bh < 8 {
        return None;
    }
    let mut value = vec![0.0f32; bw * bh];
    let mut weight = vec![0.0f32; bw * bh];
    geo.disk(|i, a, bb| {
        if a.abs() > 0.75 || a * a + bb * bb > 0.85 {
            return;
        }
        let Some(lat) = latitude(a, bb, tilt, north_at_plus_b) else { return };
        if !(-30.0..=-16.0).contains(&lat) {
            return;
        }
        let (x, y) = (i % w, i / w);
        let k = (y - y0) * bw + (x - x0);
        value[k] = redness([r[i] - sky[0], g[i] - sky[1], b[i] - sky[2]]);
        weight[k] = 1.0;
    });
    let total: f32 = weight.iter().sum();
    // The region must be a sizeable piece of the disk.
    let disk_area = std::f32::consts::PI * geo.shape.major * geo.shape.minor;
    if total < 0.02 * disk_area {
        return None;
    }
    // Box blur (masked) so a noisy pixel doesn't win.
    let radius = ((0.012 * geo.shape.major) as isize).clamp(1, 5);
    let blur = |src: &[f32]| -> Vec<f32> {
        let mut tmp = vec![0.0f32; bw * bh];
        for y in 0..bh {
            for x in 0..bw {
                let mut s = 0.0;
                for d in -radius..=radius {
                    let xx = x as isize + d;
                    if (0..bw as isize).contains(&xx) {
                        s += src[y * bw + xx as usize];
                    }
                }
                tmp[y * bw + x] = s;
            }
        }
        let mut out = vec![0.0f32; bw * bh];
        for y in 0..bh {
            for x in 0..bw {
                let mut s = 0.0;
                for d in -radius..=radius {
                    let yy = y as isize + d;
                    if (0..bh as isize).contains(&yy) {
                        s += tmp[yy as usize * bw + x];
                    }
                }
                out[y * bw + x] = s;
            }
        }
        out
    };
    let (num, den) = (blur(&value.iter().zip(&weight).map(|(v, w)| v * w).collect::<Vec<_>>()), blur(&weight));
    let smooth: Vec<f32> = num.iter().zip(&den).zip(&weight).map(|((n, d), w)| if *w > 0.0 && *d > 0.0 { n / d } else { f32::NEG_INFINITY }).collect();
    let mut inside: Vec<f32> = smooth.iter().copied().filter(|v| v.is_finite()).collect();
    inside.sort_by(|p, q| p.total_cmp(q));
    let median = inside[inside.len() / 2];
    let (peak_k, peak) = smooth.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (k, v)| if *v > best.1 { (k, *v) } else { best });
    if peak - median < 0.10 {
        return None;
    }
    // The connected patch around the peak above 60 % of the way from the median to the peak.
    let level = median + 0.6 * (peak - median);
    let mut seen = vec![false; bw * bh];
    let mut stack = vec![peak_k];
    seen[peak_k] = true;
    let mut patch: Vec<usize> = Vec::new();
    while let Some(k) = stack.pop() {
        patch.push(k);
        let (x, y) = (k % bw, k / bw);
        for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= bw as isize || ny >= bh as isize {
                continue;
            }
            let nk = ny as usize * bw + nx as usize;
            if !seen[nk] && smooth[nk] > level {
                seen[nk] = true;
                stack.push(nk);
            }
        }
    }
    // A spot is small, a few thousandths to a few hundredths of the disk.
    let area = patch.len() as f32;
    if area < 0.0015 * disk_area || area > 0.04 * disk_area {
        return None;
    }
    // And roundish: a belt's reddest stretch is long and thin, a spot is not.
    let (mut lo, mut hi) = ((usize::MAX, usize::MAX), (0usize, 0usize));
    for k in &patch {
        let (x, y) = (k % bw, k / bw);
        lo = (lo.0.min(x), lo.1.min(y));
        hi = (hi.0.max(x), hi.1.max(y));
    }
    let (pw, ph) = ((hi.0 - lo.0 + 1) as f32, (hi.1 - lo.1 + 1) as f32);
    if pw > 0.5 * geo.shape.major || ph > 0.4 * geo.shape.minor || pw / ph > 3.5 || ph / pw > 3.5 {
        return None;
    }
    let mut colour = [0.0f32; 3];
    for (c, plane) in [r, g, b].into_iter().enumerate() {
        let mut v: Vec<f32> = patch.iter().map(|k| plane[(k / bw + y0) * w + (k % bw + x0)] - sky[c]).collect();
        v.sort_by(|p, q| p.total_cmp(q));
        colour[c] = v[v.len() / 2];
    }
    Some(colour)
}

/// Each channel's 5th percentile over the picture: the sky.
fn sky_level(img: &Image) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for (k, plane) in img.channels().iter().enumerate().take(3) {
        let mut v: Vec<f32> = plane.iter().copied().step_by((plane.len() / 100_000).max(1)).collect();
        v.sort_by(|a, b| a.total_cmp(b));
        out[k] = v[v.len() / 20].max(0.0);
    }
    out
}

/// A colour (encoded) as linear light normalised to unit luminance.
fn normalised(c: [f32; 3]) -> Option<[f64; 3]> {
    if !c.iter().all(|v| v.is_finite()) {
        return None;
    }
    let l = [linear(c[0]), linear(c[1]), linear(c[2])];
    let y = luma(l) as f64;
    (y > 1e-6).then(|| [l[0] as f64 / y, l[1] as f64 / y, l[2] as f64 / y])
}

/// Solve the 3×3 system `a x = b`.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..3 {
            let f = a[row][col] / a[col][col];
            let pivot_row = a[col];
            for (v, p) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *v -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 3];
    for row in (0..3).rev() {
        let s: f64 = (row + 1..3).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    Some(x)
}

/// One colour pair to match: the picture's (unit-luminance linear), the reference's, and its weights.
struct Row {
    u: [f64; 3],
    v: [f64; 3],
    /// Weight for the white-balance step (neutral bands) and for the matrix step.
    neutral: f64,
    general: f64,
    spot: bool,
}

/// The fit for one reading of the picture.
struct Candidate {
    matrix: [[f64; 3]; 3],
    gains: [f64; 3],
    bands: usize,
    before: f32,
    after: f32,
    spot: bool,
}

fn map(m: &[[f64; 3]; 3], u: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|j| m[j][0] * u[0] + m[j][1] * u[1] + m[j][2] * u[2])
}

fn unit_luma(c: [f64; 3]) -> [f64; 3] {
    let y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    if y > 1e-9 { [c[0] / y, c[1] / y, c[2] / y] } else { c }
}

fn fit_candidate(photo: &[Band], spot: Option<[f32; 3]>, reference: Reference) -> Option<Candidate> {
    let (table, spot_ref) = reference.data();
    let refs: Vec<Option<[f64; 3]>> = table.iter().map(|c| normalised(*c)).collect();
    // How far each reference band is from grey, and the median of that: neutral bands are the ones
    // below it, belts the ones above.
    let chroma = |v: &[f64; 3]| ((v[0] - 1.0).powi(2) + (v[1] - 1.0).powi(2) + (v[2] - 1.0).powi(2)).sqrt();
    let mut all: Vec<f64> = refs.iter().flatten().map(chroma).collect();
    all.sort_by(|a, b| a.total_cmp(b));
    let c_ref = all[all.len() / 2].max(1e-3);
    let mut rows: Vec<Row> = Vec::new();
    for (b, p) in photo.iter().enumerate() {
        let Some((colour, n)) = p else { continue };
        let (Some(u), Some(v)) = (normalised(*colour), refs[b]) else { continue };
        let (root, c) = ((*n as f64).sqrt().min(40.0), chroma(&v));
        rows.push(Row { u, v, neutral: root * (1.0 - c / (1.2 * c_ref)).max(0.0), general: root * (0.4 + (c / c_ref).clamp(0.0, 2.0)), spot: false });
    }
    if rows.len() < 12 {
        return None;
    }
    let bands = rows.len();
    let mean_general = rows.iter().map(|r| r.general).sum::<f64>() / bands as f64;
    if let (Some(u), Some(v)) = (spot.and_then(normalised), normalised(spot_ref)) {
        rows.push(Row { u, v, neutral: 0.0, general: 3.0 * mean_general, spot: true });
    }
    // Step 1: white balance from the neutral bands.
    let wsum: f64 = rows.iter().map(|r| r.neutral).sum();
    let mut gains = [1.0f64; 3];
    if wsum > 1e-6 {
        for (c, gain) in gains.iter_mut().enumerate() {
            let num: f64 = rows.iter().map(|r| r.neutral * r.u[c] * r.v[c]).sum();
            let den: f64 = rows.iter().map(|r| r.neutral * r.u[c] * r.u[c]).sum();
            if den > 1e-9 {
                *gain = (num / den).clamp(0.6, 1.6);
            }
        }
        let y = 0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2];
        gains.iter_mut().for_each(|g| *g /= y);
    }
    // Step 2: the matrix on the balanced colours, held towards the identity.
    let balanced: Vec<[f64; 3]> = rows.iter().map(|r| [r.u[0] * gains[0], r.u[1] * gains[1], r.u[2] * gains[2]]).collect();
    let scale: f64 = rows.iter().zip(&balanced).map(|(r, u)| r.general * u.iter().map(|x| x * x).sum::<f64>()).sum::<f64>() / 3.0;
    let lambda = RIDGE * scale;
    let mut ata = [[0.0f64; 3]; 3];
    for (r, u) in rows.iter().zip(&balanced) {
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += r.general * u[i] * u[j];
            }
        }
    }
    for (i, row) in ata.iter_mut().enumerate() {
        row[i] += lambda;
    }
    let mut m2 = [[0.0f64; 3]; 3];
    for (j, out) in m2.iter_mut().enumerate() {
        let mut rhs = [0.0f64; 3];
        for (r, u) in rows.iter().zip(&balanced) {
            for i in 0..3 {
                rhs[i] += r.general * u[i] * r.v[j];
            }
        }
        rhs[j] += lambda;
        *out = solve3(ata, rhs)?;
    }
    // Everything in one matrix: the gains first, then the second step.
    let mut matrix = [[0.0f64; 3]; 3];
    for j in 0..3 {
        for i in 0..3 {
            matrix[j][i] = m2[j][i] * gains[i];
        }
    }
    let total: f64 = rows.iter().map(|r| r.general).sum();
    let dist = |m: Option<&[[f64; 3]; 3]>| -> f32 {
        let mut s = 0.0;
        for r in &rows {
            let mapped = m.map_or(r.u, |m| unit_luma(map(m, r.u)));
            let d: f64 = (0..3).map(|k| (mapped[k] - r.v[k]).powi(2)).sum::<f64>().sqrt();
            s += r.general * d;
        }
        (s / total) as f32
    };
    Some(Candidate { matrix, gains, bands, before: dist(None), after: dist(Some(&matrix)), spot: rows.iter().any(|r| r.spot) })
}

/// Measure the picture's bands and find the matrix. `None` if there is no oblate planet with
/// enough bands in it.
pub fn analyse(img: &Image, params: &PhotoColorParams) -> Option<Fit> {
    if img.is_mono() {
        return None;
    }
    let geo = Geometry::of(img)?;
    let sky = sky_level(img);
    let tilt = (params.tilt as f64).clamp(-8.0, 8.0).to_radians();
    let mut best: Option<(bool, Candidate)> = None;
    for north_at_plus_b in [true, false] {
        let photo = measure(img, &geo, sky, tilt, north_at_plus_b);
        let spot = find_spot(img, &geo, sky, tilt, north_at_plus_b);
        if let Some(c) = fit_candidate(&photo, spot, params.reference)
            && best.as_ref().is_none_or(|(_, b)| c.after < b.after)
        {
            best = Some((north_at_plus_b, c));
        }
    }
    let (north_at_plus_b, c) = best?;
    Some(Fit { matrix: c.matrix.map(|r| r.map(|v| v as f32)), gains: c.gains.map(|v| v as f32), sky, bands: c.bands, before: c.before, after: c.after, north_at_plus_b, spot: c.spot })
}

/// Carry the colours onto the reference by `params.strength`; the picture is returned as it came
/// if no fit is possible.
pub fn apply(img: &Image, params: &PhotoColorParams) -> Image {
    let Some(fit) = analyse(img, params) else { return img.clone() };
    apply_fit(img, &fit, params.strength)
}

pub fn apply_fit(img: &Image, fit: &Fit, strength: f32) -> Image {
    let [r, g, b] = img.channels() else { return img.clone() };
    let (w, h) = (img.width(), img.height());
    let k = strength.clamp(0.0, 1.0);
    let m = fit.matrix;
    let sky = fit.sky;
    let sky_grey = (sky[0] + sky[1] + sky[2]) / 3.0;
    let (mut or, mut og, mut ob) = (vec![0.0f32; w * h], vec![0.0f32; w * h], vec![0.0f32; w * h]);
    for i in 0..w * h {
        let c = [r[i], g[i], b[i]];
        // Linear light, sky removed.
        let d = [linear(c[0] - sky[0]), linear(c[1] - sky[1]), linear(c[2] - sky[2])];
        let mapped = [0, 1, 2].map(|j| m[j][0] * d[0] + m[j][1] * d[1] + m[j][2] * d[2]);
        // Back to the picture's own luminance, so only the colours move.
        let (y_in, y_out) = (luma(d), luma(mapped));
        let f = if y_out > 1e-9 && y_in > 0.0 { y_in / y_out } else { 1.0 };
        let new = [0, 1, 2].map(|j| encode(mapped[j] * f) + sky_grey);
        or[i] = c[0] + k * (new[0] - c[0]);
        og[i] = c[1] + k * (new[1] - c[1]);
        ob[i] = c[2] + k * (new[2] - c[2]);
    }
    Image::new(w, h, vec![or, og, ob], img.source_bit_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An oblate Jupiter whose band colours are `table` (encoded), cast by `cast` (per-channel gains in
    /// linear light), seen with `tilt` degrees and north up or down, with an optional red spot.
    fn synthetic(table: &[[f32; 3]; BINS], spot: Option<[f32; 3]>, cast: [f32; 3], north_down: bool, tilt: f64) -> Image {
        let (w, h) = (260usize, 260usize);
        let (cx, cy, major, minor) = (130.0f64, 130.0f64, 100.0f64, 93.5f64);
        let mut ch = vec![vec![0.02f32; w * h]; 3];
        for y in 0..h {
            for x in 0..w {
                let (a, b) = ((x as f64 - cx) / major, (y as f64 - cy) / minor);
                if a * a + b * b >= 1.0 {
                    continue;
                }
                // Image y down: north up means north at −b.
                let north_at_plus_b = north_down;
                let Some(lat) = latitude(a, b, tilt.to_radians(), north_at_plus_b) else { continue };
                let bin = (((lat + 60.0) / 2.5).floor().max(0.0) as usize).min(BINS - 1);
                let mut t = table[bin];
                // A spot at 23° S, a few degrees wide, left of the centre.
                if let Some(s) = spot
                    && (lat + 23.0).abs() < 4.0
                    && ((a + 0.25) * 100.0).abs() < 9.0
                {
                    t = s;
                }
                let limb = (1.0 - a * a - b * b).max(0.0).sqrt().powf(0.3) as f32;
                for k in 0..3 {
                    ch[k][y * w + x] = encode(0.02 + linear(t[k]) * cast[k] * 0.8 * limb).min(1.0);
                }
            }
        }
        Image::new(w, h, ch, 16)
    }

    fn params(reference: Reference, tilt: f32) -> PhotoColorParams {
        PhotoColorParams { reference, strength: 1.0, tilt }
    }

    #[test]
    fn a_colour_cast_is_taken_out() {
        for north_down in [false, true] {
            let img = synthetic(&HUBBLE_2019, None, [1.0, 0.7, 0.35], north_down, 0.0);
            let fit = analyse(&img, &params(Reference::Hubble2019, 0.0)).expect("a planet with bands");
            assert!(fit.bands >= 30, "{}", fit.bands);
            assert!(fit.after < fit.before * 0.4, "before {} after {}", fit.before, fit.after);
            assert!(fit.gains[2] > fit.gains[0] * 1.4, "blue must come up against red: {:?}", fit.gains);
            let out = apply_fit(&img, &fit, 1.0);
            let mid = 130 * 260 + 110;
            let (rb_in, rb_out) = (img.channels()[2][mid] / img.channels()[0][mid], out.channels()[2][mid] / out.channels()[0][mid]);
            assert!(rb_out > rb_in * 1.2, "blue/red {rb_in} -> {rb_out}");
        }
    }

    /// With the planet tilted, using the tilt in the latitudes fits better than ignoring it.
    #[test]
    fn the_tilt_improves_the_fit() {
        let img = synthetic(&CASSINI_2000, None, [1.0, 0.85, 0.7], false, 3.4);
        let with = analyse(&img, &params(Reference::Cassini2000, 3.4)).unwrap();
        let without = analyse(&img, &params(Reference::Cassini2000, 0.0)).unwrap();
        assert!(with.after <= without.after, "with tilt {} without {}", with.after, without.after);
        let wrong = analyse(&img, &params(Reference::Cassini2000, -3.4)).unwrap();
        assert!(with.after < wrong.after, "with {} against the wrong sign {}", with.after, wrong.after);
    }

    /// The Red Spot is found, used, and tells north from south.
    #[test]
    fn the_red_spot_is_found_and_decides_the_hemisphere() {
        let spot = [0.78, 0.5, 0.35];
        for north_down in [false, true] {
            let img = synthetic(&CASSINI_2000, Some(spot), [1.0, 0.9, 0.8], north_down, 0.0);
            let fit = analyse(&img, &params(Reference::Cassini2000, 0.0)).unwrap();
            assert!(fit.spot, "the spot should be found (north down: {north_down})");
            assert_eq!(fit.north_at_plus_b, north_down, "{fit:?}");
        }
        // No spot, no anchor.
        let img = synthetic(&CASSINI_2000, None, [1.0, 0.9, 0.8], false, 0.0);
        assert!(!analyse(&img, &params(Reference::Cassini2000, 0.0)).unwrap().spot);
    }

    #[test]
    fn strength_zero_and_non_planets_change_nothing() {
        let img = synthetic(&HUBBLE_2019, None, [1.0, 0.8, 0.6], false, 0.0);
        let same = apply(&img, &PhotoColorParams::neutral());
        assert_eq!(same.channels()[0], img.channels()[0]);
        // A flat grey frame has no planet.
        let flat = Image::new(100, 100, vec![vec![0.3; 10_000]; 3], 16);
        assert!(analyse(&flat, &PhotoColorParams::default()).is_none());
        assert_eq!(apply(&flat, &PhotoColorParams::default()).channels()[1], flat.channels()[1]);
        // Mono frames are skipped.
        assert!(analyse(&Image::new(100, 100, vec![vec![0.3; 10_000]], 16), &PhotoColorParams::default()).is_none());
    }

    #[test]
    fn luminance_is_kept() {
        let img = synthetic(&HUBBLE_2019, None, [1.0, 0.7, 0.5], false, 0.0);
        let out = apply(&img, &PhotoColorParams::default());
        let i = 130 * 260 + 120;
        let lum = |im: &Image| luma([linear(im.channels()[0][i]), linear(im.channels()[1][i]), linear(im.channels()[2][i])]);
        assert!((lum(&img) - lum(&out)).abs() < 0.02 * lum(&img).max(0.05), "{} {}", lum(&img), lum(&out));
    }
}
