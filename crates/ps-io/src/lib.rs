//! Image import/export. TIFF, PNG and JPEG go through the `image` crate; FITS is read in-house.

use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, Rgb};
use ps_core::Image;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode error: {0}")]
    Decode(#[from] image::ImageError),
    #[error("malformed file: {0}")]
    Malformed(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// Extensions accepted by the open dialog.
pub const READ_EXTENSIONS: &[&str] = &["tif", "tiff", "png", "jpg", "jpeg", "fit", "fits", "fts"];

pub fn load(path: &Path) -> Result<Image, IoError> {
    let bytes = std::fs::read(path)?;
    if is_fits(path, &bytes) {
        return ps_core::fits::read(&bytes, ps_core::fits::Scaling::Normalise).map_err(IoError::Malformed);
    }
    let dynamic = image::load_from_memory(&bytes)?;
    Ok(from_dynamic(dynamic))
}

fn is_fits(path: &Path, bytes: &[u8]) -> bool {
    let ext = extension(path);
    matches!(ext.as_str(), "fit" | "fits" | "fts") || bytes.starts_with(b"SIMPLE  =")
}

fn from_dynamic(img: DynamicImage) -> Image {
    let color = img.color();
    let bit_depth = (color.bits_per_pixel() / color.channel_count() as u16) as u8;
    let (w, h) = (img.width() as usize, img.height() as usize);
    if color.has_color() {
        let rgb = img.into_rgb32f();
        let mut planes: Vec<Vec<f32>> = (0..3).map(|_| Vec::with_capacity(w * h)).collect();
        for px in rgb.pixels() {
            for (plane, &v) in planes.iter_mut().zip(px.0.iter()) {
                plane.push(v);
            }
        }
        Image::new(w, h, planes, bit_depth)
    } else {
        let luma = img.to_luma32f();
        Image::new(w, h, vec![luma.into_raw()], bit_depth)
    }
}

/// Save according to the file extension. PNG and TIFF are written at 16 bit, JPEG at 8 bit.
pub fn save(path: &Path, img: &Image) -> Result<(), IoError> {
    let (w, h) = (img.width() as u32, img.height() as u32);
    let format = ImageFormat::from_path(path)?;
    let dynamic = match format {
        ImageFormat::Png | ImageFormat::Tiff => {
            let data = img.to_interleaved_u16();
            if img.is_mono() {
                DynamicImage::ImageLuma16(ImageBuffer::<Luma<u16>, _>::from_raw(w, h, data).unwrap())
            } else {
                DynamicImage::ImageRgb16(ImageBuffer::<Rgb<u16>, _>::from_raw(w, h, data).unwrap())
            }
        }
        ImageFormat::Jpeg => {
            let rgba = img.to_rgba8();
            DynamicImage::ImageRgba8(ImageBuffer::from_raw(w, h, rgba).unwrap()).into_rgb8().into()
        }
        other => return Err(IoError::Unsupported(format!("export to {other:?}"))),
    };
    dynamic.save_with_format(path, format)?;
    Ok(())
}

fn extension(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png16_roundtrip_preserves_values() {
        let dir = std::env::temp_dir().join(format!("ps-io-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.png");
        let src = Image::new(2, 1, vec![vec![0.25, 0.75]], 16);
        save(&path, &src).unwrap();
        let back = load(&path).unwrap();
        assert!(back.is_mono());
        for (a, b) in src.channels()[0].iter().zip(&back.channels()[0]) {
            assert!((a - b).abs() < 1.0 / 65535.0);
        }
        std::fs::remove_dir_all(dir).ok();
    }
}
