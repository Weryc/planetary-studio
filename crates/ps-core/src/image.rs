use rayon::prelude::*;

/// Planar floating-point image. Samples are linear and normalised to [0, 1].
///
/// Planar storage (one `Vec<f32>` per channel) keeps per-channel algorithms
/// cache-friendly and maps 1:1 onto single-channel GPU textures.
#[derive(Clone, Debug)]
pub struct Image {
    width: usize,
    height: usize,
    channels: Vec<Vec<f32>>,
    /// Bit depth of the file the image came from (informational; processing is always f32).
    pub source_bit_depth: u8,
}

impl Image {
    pub fn new(width: usize, height: usize, channels: Vec<Vec<f32>>, source_bit_depth: u8) -> Self {
        assert!(channels.len() == 1 || channels.len() == 3, "only mono or RGB images are supported");
        assert!(channels.iter().all(|c| c.len() == width * height), "channel length does not match dimensions");
        Self { width, height, channels, source_bit_depth }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn is_mono(&self) -> bool {
        self.channels.len() == 1
    }

    pub fn channels(&self) -> &[Vec<f32>] {
        &self.channels
    }

    pub fn channels_mut(&mut self) -> &mut [Vec<f32>] {
        &mut self.channels
    }

    /// Copy of the `w`×`h` region at (`x`, `y`); the region must lie inside the image.
    pub fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Image {
        assert!(x + w <= self.width && y + h <= self.height, "crop outside image");
        let channels = self.channels.iter().map(|c| (y..y + h).flat_map(|row| &c[row * self.width + x..row * self.width + x + w]).copied().collect()).collect();
        Image::new(w, h, channels, self.source_bit_depth)
    }

    /// The image extended by `pad` px to the right and below, so that seen as periodic (as the
    /// frequency-domain steps see it) it is continuous at every edge: past each edge first its
    /// mirror image, then the mirror image of the opposite edge, which wraps back onto it. A
    /// zoomed region processed this way doesn't wrap the bright subject cut at one edge onto
    /// the opposite one (stripes across the region); the original stays at the top left.
    pub fn mirror_extended(&self, pad: usize) -> Image {
        let channels = self.channels.iter().map(|c| mirror_extend_plane(c, self.width, self.height, pad)).collect();
        let (w, h) = extended_size(self.width, self.height, pad);
        Image::new(w, h, channels, self.source_bit_depth)
    }

    /// Box-average `factor`×`factor` blocks (any remainder at the right/bottom is dropped).
    pub fn downscaled(&self, factor: usize) -> Image {
        let f = factor.max(1);
        let (w, h) = ((self.width / f).max(1), (self.height / f).max(1));
        let channels = self.channels.iter().map(|c| downscale_plane(c, self.width, self.height, f)).collect();
        Image::new(w, h, channels, self.source_bit_depth)
    }

    /// Interleaved 8-bit RGBA, for display and 8-bit export.
    pub fn to_rgba8(&self) -> Vec<u8> {
        let n = self.width * self.height;
        let mut out = vec![255u8; n * 4];
        let quant = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let (r, g, b) = self.rgb_planes();
        out.par_chunks_mut(4).enumerate().for_each(|(i, px)| {
            px[0] = quant(r[i]);
            px[1] = quant(g[i]);
            px[2] = quant(b[i]);
        });
        out
    }

    /// Interleaved 16-bit samples (1 or 3 per pixel), for lossless export.
    pub fn to_interleaved_u16(&self) -> Vec<u16> {
        let n = self.width * self.height;
        let c = self.channels.len();
        let mut out = vec![0u16; n * c];
        out.par_chunks_mut(c).enumerate().for_each(|(i, px)| {
            for (o, plane) in px.iter_mut().zip(&self.channels) {
                *o = (plane[i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            }
        });
        out
    }

    /// The three colour planes (the one grey plane thrice for a mono image).
    pub fn rgb_planes(&self) -> (&[f32], &[f32], &[f32]) {
        match self.channels.as_slice() {
            [m] => (m, m, m),
            [r, g, b] => (r, g, b),
            _ => unreachable!(),
        }
    }
}

/// Box-average a `w`×`h` plane by `f` (at least 1×1 out).
/// The size of [`Image::mirror_extended`]: `pad` is at most each side.
pub fn extended_size(w: usize, h: usize, pad: usize) -> (usize, usize) {
    (w + 2 * pad.min(w), h + 2 * pad.min(h))
}

/// Index into a line of `n` extended by `pad` (see [`Image::mirror_extended`]).
fn extended_index(i: usize, n: usize, pad: usize) -> usize {
    if i < n {
        i
    } else if i < n + pad {
        2 * n - 1 - i
    } else {
        n + 2 * pad - 1 - i
    }
}

/// One plane of [`Image::mirror_extended`].
pub fn mirror_extend_plane(plane: &[f32], w: usize, h: usize, pad: usize) -> Vec<f32> {
    let (px, py) = (pad.min(w), pad.min(h));
    let (ew, eh) = extended_size(w, h, pad);
    let mut out = vec![0.0f32; ew * eh];
    out.par_chunks_mut(ew).enumerate().for_each(|(y, row)| {
        let sy = extended_index(y, h, py);
        for (x, o) in row.iter_mut().enumerate() {
            *o = plane[sy * w + extended_index(x, w, px)];
        }
    });
    out
}

pub fn downscale_plane(plane: &[f32], w: usize, h: usize, f: usize) -> Vec<f32> {
    let (dw, dh) = ((w / f).max(1), (h / f).max(1));
    let (fx, fy) = (f.min(w), f.min(h));
    let norm = 1.0 / (fx * fy) as f32;
    let mut out = vec![0.0f32; dw * dh];
    out.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for sy in 0..fy {
                let start = (y * fy + sy) * w + x * fx;
                acc += plane[start..start + fx].iter().sum::<f32>();
            }
            *o = acc * norm;
        }
    });
    out
}

/// Bilinear sample of a `w`×`h` plane at (x, y), pixel centres at +0.5, edges clamped.
pub fn bilinear_clamped(plane: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (x, y) = ((x - 0.5).clamp(0.0, (w - 1) as f32), (y - 0.5).clamp(0.0, (h - 1) as f32));
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let at = |x: usize, y: usize| plane[y * w + x];
    let top = at(x0, y0) + tx * (at(x1, y0) - at(x0, y0));
    let bottom = at(x0, y1) + tx * (at(x1, y1) - at(x0, y1));
    top + ty * (bottom - top)
}
