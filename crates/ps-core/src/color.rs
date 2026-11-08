//! How a single-plane algorithm is applied to a colour image.

use crate::Image;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorMode {
    /// Process luminance only and add the change back to R, G and B: one third of the work,
    /// no colour fringing from per-channel differences.
    #[default]
    Luminance,
    /// Process R, G and B independently.
    Rgb,
}

/// Rec. 709 luminance weights.
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

pub fn luma([r, g, b]: [f32; 3]) -> f32 {
    LUMA[0] * r + LUMA[1] * g + LUMA[2] * b
}

/// Bring a colour inside 0‥1 by moving it towards the grey of the same luminance, instead of
/// clipping channels one by one: clipping a single channel changes the hue (a pixel with R and G
/// below zero shows as a pure blue dot). Luminance is kept unless it is itself out of range.
pub fn fit_gamut(c: [f32; 3]) -> [f32; 3] {
    let (lo, hi) = (c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]));
    if lo >= 0.0 && hi <= 1.0 {
        return c;
    }
    let y = luma(c).clamp(0.0, 1.0);
    // Largest t in 0‥1 with y + t (c − y) inside 0‥1 for every channel.
    let mut t = 1.0f32;
    for v in c {
        if v < 0.0 {
            t = t.min(y / (y - v));
        } else if v > 1.0 {
            t = t.min((1.0 - y) / (v - y));
        }
    }
    c.map(|v| (y + t * (v - y)).clamp(0.0, 1.0))
}

/// [`fit_gamut`] over a whole RGB image (mono images are clamped).
pub fn fit_gamut_image(img: &Image) -> Image {
    let channels = match img.channels() {
        [r, g, b] => {
            let mut out = vec![vec![0.0f32; r.len()]; 3];
            let [or, og, ob] = out.as_mut_slice() else { unreachable!() };
            or.par_iter_mut().zip(og.par_iter_mut()).zip(ob.par_iter_mut()).enumerate().for_each(|(i, ((ro, go), bo))| {
                [*ro, *go, *bo] = fit_gamut([r[i], g[i], b[i]]);
            });
            out
        }
        channels => channels.iter().map(|c| c.par_iter().map(|v| v.clamp(0.0, 1.0)).collect()).collect(),
    };
    Image::new(img.width(), img.height(), channels, img.source_bit_depth)
}

/// Whether any sample lies outside 0‥1.
pub fn out_of_gamut(img: &Image) -> bool {
    img.channels().par_iter().any(|c| c.par_iter().any(|v| !(0.0..=1.0).contains(v)))
}

/// The planes an algorithm should run on for this image and mode.
pub fn working_planes(img: &Image, mode: ColorMode) -> Vec<Vec<f32>> {
    match (img.channels(), mode) {
        ([r, g, b], ColorMode::Luminance) => {
            let mut l = vec![0.0f32; r.len()];
            l.par_iter_mut().enumerate().for_each(|(i, v)| *v = luma([r[i], g[i], b[i]]));
            vec![l]
        }
        (channels, _) => channels.to_vec(),
    }
}

/// Inverse of [`working_planes`]: build the output image from the processed planes.
/// `original_planes` are the unprocessed working planes (needed for the luminance delta).
pub fn merge_planes(img: &Image, mode: ColorMode, original_planes: &[Vec<f32>], processed: Vec<Vec<f32>>) -> Image {
    let (w, h, depth) = (img.width(), img.height(), img.source_bit_depth);
    match (img.channels(), mode) {
        ([_, _, _], ColorMode::Luminance) => {
            let (before, after) = (&original_planes[0], &processed[0]);
            // Brightening adds the change to every channel (colours don't oversaturate in the
            // highlights); darkening scales the channels, which keeps the hue and never drives a
            // channel below zero before the others (the source of coloured dots in the sky).
            let channels = img
                .channels()
                .par_iter()
                .map(|c| {
                    c.iter()
                        .zip(before)
                        .zip(after)
                        .map(|((&v, &l0), &l1)| match () {
                            _ if l1 >= l0 => v + (l1 - l0),
                            _ if l0 > 1e-6 => v * (l1.max(0.0) / l0),
                            _ => l1.max(0.0),
                        })
                        .collect()
                })
                .collect();
            Image::new(w, h, channels, depth)
        }
        _ => Image::new(w, h, processed, depth),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luminance_delta_is_added_to_each_channel() {
        let img = Image::new(1, 1, vec![vec![0.2], vec![0.4], vec![0.6]], 16);
        let planes = working_planes(&img, ColorMode::Luminance);
        assert_eq!(planes.len(), 1);
        let brighter = vec![vec![planes[0][0] + 0.1]];
        let out = merge_planes(&img, ColorMode::Luminance, &planes, brighter);
        for (c, want) in out.channels().iter().zip([0.3, 0.5, 0.7]) {
            assert!((c[0] - want).abs() < 1e-6);
        }
    }

    #[test]
    fn darkening_keeps_hue_and_stays_positive() {
        // A dim bluish sky pixel, darkened by more than its red and green.
        let img = Image::new(1, 1, vec![vec![0.002], vec![0.003], vec![0.01]], 16);
        let planes = working_planes(&img, ColorMode::Luminance);
        let out = merge_planes(&img, ColorMode::Luminance, &planes, vec![vec![planes[0][0] - 0.0025]]);
        let px: Vec<f32> = out.channels().iter().map(|c| c[0]).collect();
        assert!(px.iter().all(|v| *v >= 0.0), "{px:?}");
        assert!((px[2] / px[0] - 5.0).abs() < 1e-3, "same hue: {px:?}");
        assert!((luma([px[0], px[1], px[2]]) - (planes[0][0] - 0.0025)).abs() < 1e-6);
    }

    #[test]
    fn gamut_fit_desaturates_instead_of_clipping() {
        let c = fit_gamut([-0.02, 0.01, 0.08]);
        assert!(c.iter().all(|v| (0.0..=1.0).contains(v)), "{c:?}");
        assert!((luma(c) - luma([-0.02, 0.01, 0.08])).abs() < 1e-6, "luminance kept");
        assert_eq!(fit_gamut([0.2, 0.3, 0.4]), [0.2, 0.3, 0.4]);
        let bright = fit_gamut([1.3, 0.9, 0.6]);
        assert!(bright[0] <= 1.0 && bright[0] > bright[1] && bright[1] > bright[2], "{bright:?}");
    }

    #[test]
    fn mono_ignores_mode() {
        let img = Image::new(2, 1, vec![vec![0.1, 0.9]], 8);
        assert_eq!(working_planes(&img, ColorMode::Luminance), vec![vec![0.1, 0.9]]);
    }
}
