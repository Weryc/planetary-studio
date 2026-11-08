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
        for (ci, plane) in self.channels.iter().enumerate() {
            for (i, &v) in plane.iter().enumerate() {
                out[i * c + ci] = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            }
        }
        out
    }

    fn rgb_planes(&self) -> (&[f32], &[f32], &[f32]) {
        match self.channels.as_slice() {
            [m] => (m, m, m),
            [r, g, b] => (r, g, b),
            _ => unreachable!(),
        }
    }
}

/// Box-average a `w`×`h` plane by `f` (at least 1×1 out).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downscale_averages_blocks() {
        let img = Image::new(4, 2, vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], 16);
        assert_eq!(img.downscaled(2).channels()[0], vec![2.5, 4.5]);
    }

    #[test]
    fn mono_expands_to_grey_rgba() {
        let img = Image::new(2, 1, vec![vec![0.0, 1.0]], 16);
        assert_eq!(img.to_rgba8(), vec![0, 0, 0, 255, 255, 255, 255, 255]);
    }

    #[test]
    fn u16_roundtrip_is_exact_at_extremes() {
        let img = Image::new(1, 1, vec![vec![1.0], vec![0.0], vec![0.5]], 16);
        assert_eq!(img.to_interleaved_u16(), vec![65535, 0, 32768]);
    }
}
