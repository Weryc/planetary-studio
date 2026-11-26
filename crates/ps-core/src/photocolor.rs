//! Photometric colour for Jupiter, Saturn and Mars: the picture's colours brought towards those of a
//! NASA/ESA reference.
//!
//! - **Jupiter** and **Saturn**: the colour of each latitude band (below).
//! - **Saturn** in particular: the pixels where the rings cross the globe are left out, and the B
//!   ring's colour plays the part of Jupiter's Red Spot. The rings' tilt comes from the capture time
//!   or, without it, from the rings' outline.
//! - **Mars** has no bands: its colours follow the ground (bright deserts, dark albedo features,
//!   the polar caps), so the fingerprint is the colour at each level of brightness over the disk.
//!
//! The reference fingerprints are measured with the same code on Hubble and Cassini pictures
//! (`ps-io/examples/photocolor_refs.rs`).
//!
//! Jupiter: the colour of its belts and zones brought towards a NASA map's.
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
use crate::i18n::tl;
use crate::color::luma;
use crate::identify::{PlanetShape, planet_shape};
use crate::capture::Body;
use crate::photocolor_data::{CASSINI_2000, CASSINI_2000_GRS, HUBBLE_2019, HUBBLE_2019_GRS, HUBBLE_2021, HUBBLE_2021_GRS};
use crate::photocolor_data::{MARS_HUBBLE_2003, MARS_HUBBLE_2016, MARS_HUBBLE_2024, SATURN_CASSINI_2004, SATURN_CASSINI_2004_RING, SATURN_HUBBLE_2019, SATURN_HUBBLE_2019_RING, SATURN_HUBBLE_2022, SATURN_HUBBLE_2022_RING};
use serde::{Deserialize, Serialize};

/// Bins of 2.5° of planetocentric latitude from −60° to +60°.
const BINS: usize = 48;
/// Jupiter's flattening.
const FLATTENING: f64 = 0.06487;
/// Saturn's flattening.
const SATURN_FLATTENING: f64 = 0.09796;
/// Levels of brightness Mars's disk is split into.
pub const MARS_BINS: usize = 16;
/// The rings over the globe, in equatorial radii (C ring's inner edge to A ring's outer, with a margin).
const RINGS: (f64, f64) = (1.18, 2.35);
/// The B ring, whose colour is measured beside the globe.
const B_RING: (f64, f64) = (1.55, 1.92);
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
    Cassini2000,
    /// Hubble OPAL 2019 in red, green and blue filters (631, 547, 467 nm) calibrated to reflectance:
    /// physical colours, pale. (`HubbleNatural2019` was its name in the first version.)
    #[serde(alias = "HubbleNatural2019")]
    Hubble2019,
    /// The same from Hubble OPAL 2021 (631, 502, 467 nm).
    Hubble2021,
    /// Saturn, Hubble 2019 (heic1917a, colour composite).
    SaturnHubble2019,
    /// Saturn, Hubble OPAL 2022 (heic2312a).
    SaturnHubble2022,
    /// Saturn, Cassini 2004 natural-colour mosaic (PIA06193).
    SaturnCassini2004,
    /// Jupiter from the reflectance spectra of its belts and zones (see [`giant_spectra`]).
    #[default]
    JupiterSpectra,
    /// Saturn from the reflectance spectra of its bands (see [`giant_spectra`]).
    SaturnSpectra,
    /// Mars from its reflectance spectra in sunlight: the true colour of its dark areas and bright
    /// deserts (see [`mars_spectra`]).
    MarsSpectra,
    /// Mars, Hubble 2016 (heic1609a).
    MarsHubble2016,
    /// Mars, Hubble 2003 (opo0322a).
    MarsHubble2003,
    /// Mars, Hubble 2024 (heic2505b).
    MarsHubble2024,
}

impl Reference {
    pub const ALL: [Reference; 3] = [Reference::Cassini2000, Reference::Hubble2019, Reference::Hubble2021];

    pub fn label(self) -> &'static str {
        match self {
            Reference::Cassini2000 => "Cassini 2000",
            Reference::Hubble2019 | Reference::SaturnHubble2019 => "Hubble 2019",
            Reference::Hubble2021 => "Hubble 2021",
            Reference::SaturnHubble2022 => "Hubble 2022",
            Reference::SaturnCassini2004 => "Cassini 2004",
            Reference::MarsSpectra | Reference::JupiterSpectra | Reference::SaturnSpectra => tl("Spettri"),
            Reference::MarsHubble2016 => "Hubble 2016",
            Reference::MarsHubble2003 => "Hubble 2003",
            Reference::MarsHubble2024 => "Hubble 2024",
        }
    }

    /// The subject this reference is for.
    pub fn body(self) -> Body {
        match self {
            Reference::Cassini2000 | Reference::Hubble2019 | Reference::Hubble2021 | Reference::JupiterSpectra => Body::Jupiter,
            Reference::SaturnHubble2019 | Reference::SaturnHubble2022 | Reference::SaturnCassini2004 | Reference::SaturnSpectra => Body::Saturn,
            Reference::MarsSpectra | Reference::MarsHubble2016 | Reference::MarsHubble2003 | Reference::MarsHubble2024 => Body::Mars,
        }
    }

    /// The references for a subject (empty if there are none), the default first.
    pub fn for_body(body: Body) -> &'static [Reference] {
        match body {
            Body::Jupiter => &[Reference::JupiterSpectra, Reference::Hubble2021, Reference::Hubble2019, Reference::Cassini2000],
            Body::Saturn => &[Reference::SaturnSpectra, Reference::SaturnHubble2019, Reference::SaturnHubble2022, Reference::SaturnCassini2004],
            Body::Mars => &[Reference::MarsSpectra, Reference::MarsHubble2016, Reference::MarsHubble2003, Reference::MarsHubble2024],
            _ => &[],
        }
    }

    /// The fingerprint and the extra colour (Jupiter's Red Spot, Saturn's B ring), as stored (encoded).
    fn data(self) -> (&'static [[f32; 3]], Option<[f32; 3]>) {
        match self {
            Reference::Cassini2000 => (&CASSINI_2000, Some(CASSINI_2000_GRS)),
            Reference::Hubble2019 => (&HUBBLE_2019, Some(HUBBLE_2019_GRS)),
            Reference::Hubble2021 => (&HUBBLE_2021, Some(HUBBLE_2021_GRS)),
            Reference::SaturnHubble2019 => (&SATURN_HUBBLE_2019, Some(SATURN_HUBBLE_2019_RING)),
            Reference::SaturnHubble2022 => (&SATURN_HUBBLE_2022, Some(SATURN_HUBBLE_2022_RING)),
            Reference::SaturnCassini2004 => (&SATURN_CASSINI_2004, Some(SATURN_CASSINI_2004_RING)),
            Reference::JupiterSpectra => giant_spectra(&JUPITER_SPECTRA, &HUBBLE_2021, JUPITER_SPOT),
            Reference::SaturnSpectra => giant_spectra(&SATURN_SPECTRA, &SATURN_HUBBLE_2019, SATURN_RING_SPECTRUM),
            Reference::MarsSpectra => (mars_spectra(), None),
            Reference::MarsHubble2016 => (&MARS_HUBBLE_2016, None),
            Reference::MarsHubble2003 => (&MARS_HUBBLE_2003, None),
            Reference::MarsHubble2024 => (&MARS_HUBBLE_2024, None),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotoColorParams {
    pub reference: Reference,
    /// 0 = untouched, 1 = fully carried onto the reference.
    pub strength: f32,
    /// Latitude of the Earth seen from the planet, degrees (north pole towards us is positive): filled
    /// in by the application from the capture time, 0 when the time is unknown (Saturn's is then
    /// measured on its rings).
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
    latitude_on(a, bb, tilt, north_at_plus_b, FLATTENING)
}

/// [`latitude`] on a planet of the given flattening.
fn latitude_on(a: f64, bb: f64, tilt: f64, north_at_plus_b: bool, flattening: f64) -> Option<f64> {
    let y = if north_at_plus_b { bb } else { -bb };
    let z2 = 1.0 - a * a - y * y;
    if z2 <= 0.0 {
        return None;
    }
    // y = sinφ cosB − z sinB on a sphere of unit radius, z = cosφ cosλ.
    let sin_beta = ((y + z2.sqrt() * tilt.sin()) / tilt.cos()).clamp(-1.0, 1.0);
    let beta = sin_beta.asin();
    Some(((1.0 - flattening) * beta.tan()).atan().to_degrees())
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
    if photo.len() != table.len() {
        return None;
    }
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
    if let (Some(u), Some(v)) = (spot.and_then(normalised), spot_ref.and_then(normalised)) {
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
    match params.reference.body() {
        Body::Saturn => return analyse_saturn(img, params),
        Body::Mars => return analyse_mars(img, params),
        _ => {}
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

impl Geometry {
    /// Saturn's globe, from its rings (no oblateness test: the globe found is Saturn's).
    fn saturn(img: &Image) -> Option<Geometry> {
        let g = crate::mask::saturn_globe(img)?;
        let shape = PlanetShape { cx: g.cx, cy: g.cy, angle: g.angle as f64, major: g.rx.max(g.ry), minor: g.rx.min(g.ry) };
        (shape.major >= 25.0).then_some(Geometry { shape, w: img.width(), h: img.height() })
    }

    /// A round disk (Mars, possibly in phase).
    fn round(img: &Image) -> Option<Geometry> {
        let (w, h) = (img.width(), img.height());
        let [r, g, b] = img.channels() else { return None };
        let lum: Vec<f32> = (0..w * h).map(|i| luma([r[i], g[i], b[i]])).collect();
        let shape = planet_shape(&lum, w, h)?;
        (shape.major >= 20.0 && shape.minor / shape.major >= 0.75).then_some(Geometry { shape, w, h })
    }
}

/// The median colour of each list of pixels with enough of them.
fn medians(img: &Image, sky: [f32; 3], groups: Vec<Vec<usize>>, min: usize) -> Vec<Band> {
    let [r, g, b] = img.channels() else { return Vec::new() };
    groups
        .into_iter()
        .map(|px| {
            (px.len() >= min).then(|| {
                let mut m = [0.0f32; 3];
                for (k, plane) in [r, g, b].into_iter().enumerate() {
                    let mut v: Vec<f32> = px.iter().map(|&i| plane[i] - sky[k]).collect();
                    v.sort_by(|p, q| p.total_cmp(q));
                    m[k] = v[v.len() / 2];
                }
                (m, px.len())
            })
        })
        .collect()
}

/// sin of the rings' tilt seen from the picture: the ring system's outline is an ellipse whose
/// minor over major axis is |sin B|. From the lit pixels outside the globe.
fn ring_opening(img: &Image, geo: &Geometry) -> Option<f64> {
    let [r, g, b] = img.channels() else { return None };
    let lum: Vec<f32> = (0..geo.w * geo.h).map(|i| luma([r[i], g[i], b[i]])).collect();
    let mut sample: Vec<f32> = lum.iter().copied().step_by(7).collect();
    sample.sort_by(|p, q| p.total_cmp(q));
    let top = sample[sample.len() * 995 / 1000];
    let (mut us, mut vs) = (Vec::new(), Vec::new());
    let minor_over_major = (geo.shape.minor / geo.shape.major) as f64;
    let (s, c) = geo.shape.angle.sin_cos();
    let reach = (geo.shape.major * 2.6) as isize;
    for y in (geo.shape.cy as isize - reach).max(0)..(geo.shape.cy as isize + reach).min(geo.h as isize) {
        for x in (geo.shape.cx as isize - reach).max(0)..(geo.shape.cx as isize + reach).min(geo.w as isize) {
            let i = y as usize * geo.w + x as usize;
            if lum[i] < 0.15 * top {
                continue;
            }
            let (dx, dy) = (x as f64 - geo.shape.cx as f64, y as f64 - geo.shape.cy as f64);
            let (u, v) = ((dx * c + dy * s) / geo.shape.major as f64, (-dx * s + dy * c) / geo.shape.major as f64);
            if u * u + (v / minor_over_major).powi(2) < 1.15 {
                continue;
            }
            us.push(u.abs());
            vs.push(v.abs());
        }
    }
    if us.len() < 100 {
        return None;
    }
    let pick = |v: &mut Vec<f64>| {
        v.sort_by(|p, q| p.total_cmp(q));
        v[v.len() * 98 / 100]
    };
    let (ue, ve) = (pick(&mut us), pick(&mut vs));
    (ue > 1.5).then(|| (ve / ue).clamp(0.0, 0.5))
}

/// Reflectance of the giant planets in sunlight, linear, relative to green, averaged over a colour
/// camera's red, green and blue passbands. The disc mean is fixed by the measured full-disc
/// geometric albedos (ground-based spectrophotometry, Karkoschka 1998, as tabulated by Mallama,
/// Krobusek & Pavlov 2017, Icarus, and for Saturn's globe Mallama & Pavlov 2017):
///   Jupiter  B (436 nm) 0.443, V (549) 0.538, Rc (641) 0.513, Sloan g' (477) 0.509, r' (623) 0.575
///            → red/green about 1.07 (Sloan r', 550–700 nm; the Cousins Rc band, 0.95, is
///              pulled down by the methane bands at 619 and 727 nm), blue/green about 0.84 (B at 436 nm, g' at 477, a camera's blue
///              band averages 420–500 nm);
///   Saturn   B 0.339, V 0.517, Rc 0.553, g' 0.413, r' 0.590 (uncertainty 2–4 %)
///            → red/green about 1.14, blue/green about 0.68 (the globe is yellower than Jupiter).
/// (A camera's red filter peaks near 620–640 nm, so Sloan r' is the closer match than Rc.)
/// What the belts and zones add around that mean is the contrast seen in Cassini and Hubble
/// colour data: zones grey-cream, belts darker and absorbing blue more, with a reddish chromophore
/// (so the mean over a typical disc stays at the measured value). Each entry: (belt, zone).
const JUPITER_SPECTRA: ([f32; 3], [f32; 3]) = ([1.18, 1.0, 0.73], [0.96, 1.0, 0.95]);
const SATURN_SPECTRA: ([f32; 3], [f32; 3]) = ([1.19, 1.0, 0.62], [1.09, 1.0, 0.74]);
/// The Great Red Spot, whose chromophore absorbs blue and green more than the belts'.
const JUPITER_SPOT: [f32; 3] = [1.6, 1.0, 0.50];
/// The B ring: water ice with a little reddish contamination, much less yellow than the globe.
const SATURN_RING_SPECTRUM: [f32; 3] = [1.10, 1.0, 0.82];

/// A giant planet's reference from spectra: the latitude profile of brightness tells belts
/// (dark) from zones (bright) — taken from a map's fingerprint, `profile` — and each latitude gets
/// the colour that class has in sunlight, in between for the shades, instead of the colour a
/// camera and an image processing gave it.
fn giant_spectra(spectra: &([f32; 3], [f32; 3]), profile: &[[f32; 3]], extra: [f32; 3]) -> (&'static [[f32; 3]], Option<[f32; 3]>) {
    static JUPITER: std::sync::OnceLock<Vec<[f32; 3]>> = std::sync::OnceLock::new();
    static SATURN: std::sync::OnceLock<Vec<[f32; 3]>> = std::sync::OnceLock::new();
    let cell = if spectra.0 == JUPITER_SPECTRA.0 { &JUPITER } else { &SATURN };
    let table = cell.get_or_init(|| {
        let lum: Vec<f32> = profile.iter().map(|c| luma(c.map(linear))).collect();
        let mut sorted: Vec<f32> = lum.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let (lo, hi) = (sorted[sorted.len() / 20], sorted[sorted.len() * 19 / 20]);
        lum.iter()
            .map(|&l| {
                if !l.is_finite() {
                    return [f32::NAN; 3];
                }
                let t = ((l - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
                [0, 1, 2].map(|c| encode(l * (spectra.0[c] + t * (spectra.1[c] - spectra.0[c]))))
            })
            .collect()
    });
    (table, Some(extra.map(encode)))
}

/// Mars's reflectance in sunlight, linear, relative to green, averaged over a colour camera's red,
/// green and blue passbands (about 580–700, 490–580 and 420–500 nm), from telescopic and orbital
/// spectra (Bell et al., Mustard & Bell): the bright, dusty deserts climb steeply through the red
/// (I/F about 0.08 at 450 nm, 0.16 at 550, 0.33 at 650, 0.36 at 700), the dark basaltic areas
/// much less; the blue band's average sits below 450 nm's value (the reflectance keeps falling
/// towards 420 nm). Checked against true-colour views (Hubble, Rosetta, the rovers'
/// calibrated images): butterscotch deserts, brown dark areas. A white surface lit by the Sun is
/// white, so these are Mars's true colours.
const MARS_BRIGHT: [f32; 3] = [2.6, 1.0, 0.32];
const MARS_DARK: [f32; 3] = [1.9, 1.0, 0.46];

/// The reference for [`Reference::MarsSpectra`]: each brightness level of the disk (as
/// [`mars_levels`] splits it) between the dark areas' colour and the deserts', encoded. The
/// darkest levels are mostly dark ground, the brightest mostly desert; seeing mixes them in
/// between. The very brightest (a cap, a cloud) are left out: they are not always there.
fn mars_spectra() -> &'static [[f32; 3]] {
    static TABLE: std::sync::OnceLock<Vec<[f32; 3]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        (0..MARS_BINS)
            .map(|k| {
                if k + 1 == MARS_BINS {
                    return [f32::NAN; 3];
                }
                let t = (k as f32 + 0.5) / (MARS_BINS - 1) as f32;
                let t = t * t * (3.0 - 2.0 * t);
                // Brighter levels are brighter too (only the colour counts, but keep it plausible).
                let level = 0.12 + 0.2 * t;
                [0, 1, 2].map(|c| encode(level * (MARS_DARK[c] + t * (MARS_BRIGHT[c] - MARS_DARK[c]))))
            })
            .collect()
    })
}

/// Saturn's fingerprint read one way: each latitude band's median colour on the globe's central
/// strip (where the rings don't cross it), and the B ring's colour beside the globe.
fn saturn_bands(img: &Image, geo: &Geometry, sky: [f32; 3], tilt: f64, north_at_plus_b: bool) -> (Vec<Band>, Option<[f32; 3]>) {
    let ratio = (geo.shape.minor / geo.shape.major) as f64;
    let sin_b = tilt.sin().abs().max(0.01);
    // Where a pixel would lie in the ring plane (equatorial radii), for the rings in front.
    let ring_radius = |a: f64, bb: f64| (a * a + (bb * ratio / sin_b).powi(2)).sqrt();
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); BINS];
    let mut ring: Vec<usize> = Vec::new();
    let [pr, pg, pb] = img.channels() else { return (Vec::new(), None) };
    let lum = |i: usize| luma([pr[i] - sky[0], pg[i] - sky[1], pb[i] - sky[2]]);
    let top = {
        let mut v: Vec<f32> = (0..geo.w * geo.h).step_by(7).map(lum).collect();
        v.sort_by(|p, q| p.total_cmp(q));
        v[v.len() * 995 / 1000]
    };
    let (s, c) = geo.shape.angle.sin_cos();
    let (major, minor) = (geo.shape.major as f64, geo.shape.minor as f64);
    let reach = (major * 2.4) as isize;
    for y in (geo.shape.cy as isize - reach).max(0)..(geo.shape.cy as isize + reach).min(geo.h as isize) {
        for x in (geo.shape.cx as isize - reach).max(0)..(geo.shape.cx as isize + reach).min(geo.w as isize) {
            let i = y as usize * geo.w + x as usize;
            let (dx, dy) = (x as f64 - geo.shape.cx as f64, y as f64 - geo.shape.cy as f64);
            let (a, bb) = ((dx * c + dy * s) / major, (-dx * s + dy * c) / minor);
            let k = ring_radius(a, bb);
            let over_rings = (RINGS.0..RINGS.1).contains(&k) || (sin_b < 0.03 && (bb * minor).abs() < 3.0);
            if a * a + bb * bb > 1.15 {
                if (B_RING.0..B_RING.1).contains(&k) && a.abs() > 1.2 && lum(i) > 0.2 * top {
                    ring.push(i);
                }
                continue;
            }
            if a.abs() > 0.5 || a * a + bb * bb > 0.85 || over_rings {
                continue;
            }
            let Some(lat) = latitude_on(a, bb, tilt, north_at_plus_b, SATURN_FLATTENING) else { continue };
            let bin = ((lat + 60.0) / 2.5).floor();
            if (0.0..BINS as f64).contains(&bin) {
                bins[bin as usize].push(i);
            }
        }
    }
    let ring = medians(img, sky, vec![ring], MIN_PIXELS).pop().flatten().map(|(m, _)| m);
    (medians(img, sky, bins, MIN_PIXELS), ring)
}

/// Mars's fingerprint: the ground well inside the limb (and its lit part, if in phase; caps,
/// clouds and haze left out) split into [`MARS_BINS`] levels of brightness, the median colour of
/// each.
fn mars_levels(img: &Image, geo: &Geometry, sky: [f32; 3]) -> Vec<Band> {
    let [r, g, b] = img.channels() else { return Vec::new() };
    let mut px: Vec<(f32, usize)> = Vec::new();
    let mut red: Vec<(f32, usize)> = Vec::new();
    geo.disk(|i, a, bb| {
        if a * a + bb * bb < 0.7 {
            let c = [r[i] - sky[0], g[i] - sky[1], b[i] - sky[2]];
            red.push((redness(c), i));
        }
    });
    // The ground only: the least red fifth is the polar caps, clouds and the limb's bluish haze,
    // which the reference's ground colours don't describe.
    if red.len() < MARS_BINS * MIN_PIXELS {
        return vec![None; MARS_BINS];
    }
    red.sort_by(|p, q| p.0.total_cmp(&q.0));
    for &(_, i) in &red[red.len() / 5..] {
        px.push((luma([r[i] - sky[0], g[i] - sky[1], b[i] - sky[2]]), i));
    }
    if px.len() < MARS_BINS * MIN_PIXELS {
        return vec![None; MARS_BINS];
    }
    px.sort_by(|p, q| p.0.total_cmp(&q.0));
    let top = px[px.len() * 99 / 100].0;
    px.retain(|p| p.0 > 0.2 * top);
    let n = px.len();
    let groups = (0..MARS_BINS).map(|k| px[k * n / MARS_BINS..(k + 1) * n / MARS_BINS].iter().map(|p| p.1).collect()).collect();
    medians(img, sky, groups, MIN_PIXELS)
}

/// The tilts to try on Saturn: the given one (either way up), else the rings' opening both ways.
fn saturn_tilts(img: &Image, geo: &Geometry, params: &PhotoColorParams) -> Vec<f64> {
    if params.tilt != 0.0 {
        return vec![(params.tilt as f64).clamp(-30.0, 30.0).to_radians()];
    }
    let b = ring_opening(img, geo).map_or(0.0, |s| s.asin());
    if b < 0.01 { vec![0.0] } else { vec![b, -b] }
}

fn analyse_saturn(img: &Image, params: &PhotoColorParams) -> Option<Fit> {
    let geo = Geometry::saturn(img)?;
    let sky = sky_level(img);
    let mut best: Option<(bool, Candidate)> = None;
    for tilt in saturn_tilts(img, &geo, params) {
        for north_at_plus_b in [true, false] {
            let (photo, ring) = saturn_bands(img, &geo, sky, tilt, north_at_plus_b);
            if let Some(c) = fit_candidate(&photo, ring, params.reference)
                && best.as_ref().is_none_or(|(_, b)| c.after < b.after)
            {
                best = Some((north_at_plus_b, c));
            }
        }
    }
    let (north_at_plus_b, c) = best?;
    Some(Fit { matrix: c.matrix.map(|r| r.map(|v| v as f32)), gains: c.gains.map(|v| v as f32), sky, bands: c.bands, before: c.before, after: c.after, north_at_plus_b, spot: c.spot })
}

fn analyse_mars(img: &Image, params: &PhotoColorParams) -> Option<Fit> {
    let geo = Geometry::round(img)?;
    let sky = sky_level(img);
    let c = fit_candidate(&mars_levels(img, &geo, sky), None, params.reference)?;
    Some(Fit { matrix: c.matrix.map(|r| r.map(|v| v as f32)), gains: c.gains.map(|v| v as f32), sky, bands: c.bands, before: c.before, after: c.after, north_at_plus_b: true, spot: false })
}

/// Colours of a fingerprint (`None` where too few pixels) and its extra colour.
pub type Fingerprint = (Vec<Option<[f32; 3]>>, Option<[f32; 3]>);

/// A reference's fingerprint measured on a picture (for `photocolor_refs`): `tilt` in degrees,
/// `north_up` when the picture has north at the top. The colours and the extra colour (Red Spot,
/// B ring), encoded, sky removed.
pub fn fingerprint(img: &Image, body: Body, tilt: f32, north_up: bool) -> Option<Fingerprint> {
    let sky = sky_level(img);
    let tilt = (tilt as f64).to_radians();
    let geo = match body {
        Body::Saturn => Geometry::saturn(img)?,
        Body::Mars => Geometry::round(img)?,
        _ => Geometry::of(img)?,
    };
    // The +b axis points up when its y component is negative.
    let north_at_plus_b = (geo.shape.angle.cos() < 0.0) == north_up;
    let strip = |v: Vec<Band>| v.into_iter().map(|b| b.map(|(c, _)| c)).collect::<Vec<_>>();
    match body {
        Body::Saturn => {
            // A spacecraft's view (Cassini) is not the Earth's: the rings' outline says how open they are.
            let tilt = match ring_opening(img, &geo).map(|s| s.asin()) {
                Some(b) if (b - tilt.abs()).abs() > 5f64.to_radians() => b.copysign(if tilt == 0.0 { 1.0 } else { tilt }),
                _ => tilt,
            };
            let (bands, ring) = saturn_bands(img, &geo, sky, tilt, north_at_plus_b);
            Some((strip(bands), ring))
        }
        Body::Mars => Some((strip(mars_levels(img, &geo, sky)), None)),
        _ => Some((strip(measure(img, &geo, sky, tilt, north_at_plus_b)), find_spot(img, &geo, sky, tilt, north_at_plus_b))),
    }
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
