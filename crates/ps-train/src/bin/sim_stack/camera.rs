//! The camera: photons to electrons (shot noise), read noise, fixed-pattern noise and hot
//! pixels, the ADC (8 or 12 bit), and for colour cameras the RGGB Bayer mosaic and its
//! bilinear debayering, as a stacker does.
use crate::rng::Rng;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub name: &'static str,
    pub label: &'static str,
    pub pixel_um: f32,
    pub bayer: bool,
}

pub const CAMERAS: [Camera; 3] = [
    Camera { name: "asi678mc", label: "IMX678 a colori, 2.0 µm", pixel_um: 2.0, bayer: true },
    Camera { name: "imx462c", label: "IMX462 a colori, 2.9 µm", pixel_um: 2.9, bayer: true },
    Camera { name: "imx462m", label: "IMX462 mono (filtro IR 685), 2.9 µm", pixel_um: 2.9, bayer: false },
];

pub fn camera(name: &str) -> Option<Camera> {
    CAMERAS.iter().copied().find(|c| c.name == name)
}

/// Gain levels: high gain means short exposures, so fewer photons, with less read noise.
#[derive(Clone, Copy, Debug)]
pub struct Gain {
    pub label: &'static str,
    /// Electrons at the brightest part of the subject, per frame (green or mono).
    pub peak_e: f32,
    pub read_e: f32,
}

pub const GAINS: [Gain; 3] = [
    Gain { label: "basso", peak_e: 4000.0, read_e: 3.0 },
    Gain { label: "medio", peak_e: 1000.0, read_e: 1.5 },
    Gain { label: "alto", peak_e: 250.0, read_e: 0.9 },
];

pub fn gain(label: &str) -> Option<Gain> {
    GAINS.iter().copied().find(|g| g.label == label)
}

/// Relative quantum efficiency of R, G, B behind the Bayer filters (IR-cut in front).
pub const QE: [f32; 3] = [0.78, 1.0, 0.62];

/// One sensor during one capture: its fixed pattern stays the same in every frame.
pub struct Sensor {
    pub w: usize,
    pub h: usize,
    pub bayer: bool,
    pub bits: u32,
    pub gain: Gain,
    /// Electrons per ADU: the brightest part of the subject at 75 % of the range.
    pub e_per_adu: f32,
    pub offset_adu: f32,
    row_offsets: Vec<f32>,
    dark: Vec<f32>,
}

impl Sensor {
    pub fn new(w: usize, h: usize, bayer: bool, bits: u32, gain: Gain, rng: &mut Rng) -> Self {
        let max = ((1u32 << bits) - 1) as f32;
        let e_per_adu = gain.peak_e / (0.75 * max);
        let offset_adu = if bits <= 8 { 4.0 } else { 40.0 };
        let row_offsets = (0..h).map(|_| 0.4 * gain.read_e * rng.gauss()).collect();
        // A few warm pixels (1 per 10 000) with their dark current per frame: short planetary
        // exposures keep them faint.
        let mut dark = vec![0.0f32; w * h];
        for _ in 0..(w * h / 10000) {
            let i = (rng.uniform() * (w * h) as f64) as usize % (w * h);
            dark[i] = 5.0 + 60.0 * rng.uniform().powi(2) as f32;
        }
        Sensor { w, h, bayer, bits, gain, e_per_adu, offset_adu, row_offsets, dark }
    }

    pub fn max_adu(&self) -> f32 {
        ((1u32 << self.bits) - 1) as f32
    }

    /// Channel seen by pixel (x, y): RGGB.
    pub fn cfa(x: usize, y: usize) -> usize {
        match (y % 2, x % 2) {
            (0, 0) => 0,
            (1, 1) => 2,
            _ => 1,
        }
    }

    /// One frame: `planes` (linear, the truth's units, brightest part ≈ 0.85) to raw ADU.
    pub fn expose(&self, planes: &[Vec<f32>], rng: &mut Rng) -> Vec<u16> {
        let scale = self.gain.peak_e / 0.85;
        let max = self.max_adu();
        let mut out = vec![0u16; self.w * self.h];
        for y in 0..self.h {
            let row = self.row_offsets[y] + 0.15 * self.gain.read_e * rng.gauss();
            for x in 0..self.w {
                let i = y * self.w + x;
                let c = if self.bayer { Self::cfa(x, y) } else { 0 };
                let qe = if self.bayer { QE[c] } else { 1.0 };
                let mean = planes[c][i].max(0.0) * scale * qe + self.dark[i];
                let e = rng.poisson(mean) + self.gain.read_e * rng.gauss() + row;
                out[i] = (e / self.e_per_adu + self.offset_adu).round().clamp(0.0, max) as u16;
            }
        }
        out
    }

    /// Raw frame to linear planes (offset removed, white-balanced, 1 = full scale).
    pub fn develop(&self, raw: &[u16]) -> Vec<Vec<f32>> {
        let max = self.max_adu();
        let v: Vec<f32> = raw.iter().map(|&a| (a as f32 - self.offset_adu) / max).collect();
        if !self.bayer {
            return vec![v];
        }
        let planes = debayer(&v, self.w, self.h);
        planes.into_iter().enumerate().map(|(c, p)| p.into_iter().map(|x| x / QE[c]).collect()).collect()
    }
}

/// Bilinear demosaicing of an RGGB mosaic (mirrored edges keep the pattern's parity).
pub fn debayer(m: &[f32], w: usize, h: usize) -> Vec<Vec<f32>> {
    let at = |x: isize, y: isize| m[ps_core::conv::reflect(y, h) * w + ps_core::conv::reflect(x, w)];
    let mut planes = vec![vec![0.0f32; w * h]; 3];
    for y in 0..h {
        for x in 0..w {
            let (xi, yi) = (x as isize, y as isize);
            let i = y * w + x;
            let here = at(xi, yi);
            let cross = (at(xi - 1, yi) + at(xi + 1, yi) + at(xi, yi - 1) + at(xi, yi + 1)) / 4.0;
            let diag = (at(xi - 1, yi - 1) + at(xi + 1, yi - 1) + at(xi - 1, yi + 1) + at(xi + 1, yi + 1)) / 4.0;
            let horiz = (at(xi - 1, yi) + at(xi + 1, yi)) / 2.0;
            let vert = (at(xi, yi - 1) + at(xi, yi + 1)) / 2.0;
            let (r, g, b) = match (y % 2, x % 2) {
                (0, 0) => (here, cross, diag),
                (1, 1) => (diag, cross, here),
                (0, 1) => (horiz, here, vert),
                _ => (vert, here, horiz),
            };
            planes[0][i] = r;
            planes[1][i] = g;
            planes[2][i] = b;
        }
    }
    planes
}
