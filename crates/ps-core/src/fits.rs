//! Minimal FITS primary-HDU reader and 32-bit float writer: 2-D mono or 3-plane RGB cubes.
//! Written in-house to avoid a C dependency (cfitsio) on Windows. Lives in the core because
//! the external-tool bridge (BlurXTerminator) exchanges float FITS files.

use crate::Image;

/// How sample values are mapped to the working range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// Map into 0‥1 (integer formats by their range, out-of-range floats by min/max).
    Normalise,
    /// Keep float samples as stored (used to read back our own processing results).
    KeepFloat,
}

const BLOCK: usize = 2880;
const CARD: usize = 80;

struct Header {
    bitpix: i32,
    axes: Vec<usize>,
    bzero: f64,
    bscale: f64,
    data_offset: usize,
}

pub fn read(bytes: &[u8], scaling: Scaling) -> Result<Image, String> {
    let h = parse_header(bytes)?;
    let (width, height, planes) = match h.axes.as_slice() {
        [w, h] => (*w, *h, 1),
        [w, h, 1] => (*w, *h, 1),
        [w, h, 3] => (*w, *h, 3),
        other => return Err(format!("unsupported FITS axes {other:?}")),
    };
    let bytes_per_sample = (h.bitpix.unsigned_abs() / 8) as usize;
    let n = width * height * planes;
    let data = bytes.get(h.data_offset..h.data_offset + n * bytes_per_sample).ok_or_else(|| "FITS data shorter than header declares".to_string())?;

    let raw: Vec<f64> = data.chunks_exact(bytes_per_sample).map(|b| decode_be(h.bitpix, b)).collect::<Result<_, _>>()?;
    let physical: Vec<f32> = raw.iter().map(|&v| (h.bzero + h.bscale * v) as f32).collect();
    let physical = match scaling {
        Scaling::KeepFloat if h.bitpix < 0 => physical.into_iter().map(|v| if v.is_finite() { v } else { 0.0 }).collect(),
        _ => normalise(physical, h.bitpix, h.bzero, h.bscale),
    };

    let plane_len = width * height;
    let channels = physical.chunks_exact(plane_len).map(|plane| flip_vertical(plane, width, height)).collect();
    let depth = if h.bitpix < 0 { 32 } else { h.bitpix as u8 };
    Ok(Image::new(width, height, channels, depth))
}

fn parse_header(bytes: &[u8]) -> Result<Header, String> {
    let mut bitpix = None;
    let mut naxis = 0usize;
    let mut axes = [0usize; 3];
    let mut bzero = 0.0;
    let mut bscale = 1.0;

    for (i, card) in bytes.as_chunks::<CARD>().0.iter().enumerate() {
        let text = std::str::from_utf8(card).map_err(|_| "non-ASCII FITS header".to_string())?;
        if i == 0 && !text.starts_with("SIMPLE") {
            return Err("missing SIMPLE card".into());
        }
        let key = text[..8].trim_end();
        if key == "END" {
            let header_len = (i + 1) * CARD;
            let data_offset = header_len.div_ceil(BLOCK) * BLOCK;
            let bitpix = bitpix.ok_or("missing BITPIX")?;
            return Ok(Header { bitpix, axes: axes[..naxis.min(3)].to_vec(), bzero, bscale, data_offset });
        }
        let Some(value) = card_value(text) else {
            continue;
        };
        match key {
            "BITPIX" => bitpix = value.parse().ok(),
            "NAXIS" => naxis = value.parse().unwrap_or(0),
            "NAXIS1" => axes[0] = value.parse().unwrap_or(0),
            "NAXIS2" => axes[1] = value.parse().unwrap_or(0),
            "NAXIS3" => axes[2] = value.parse().unwrap_or(0),
            "BZERO" => bzero = value.parse().unwrap_or(0.0),
            "BSCALE" => bscale = value.parse().unwrap_or(1.0),
            _ => {}
        }
    }
    Err("FITS header has no END card".into())
}

/// Value field of a `KEY     = value / comment` card.
fn card_value(card: &str) -> Option<&str> {
    if card.get(8..10) != Some("= ") {
        return None;
    }
    let v = &card[10..];
    Some(v.split('/').next().unwrap_or(v).trim())
}

fn decode_be(bitpix: i32, b: &[u8]) -> Result<f64, String> {
    Ok(match bitpix {
        8 => b[0] as f64,
        16 => i16::from_be_bytes([b[0], b[1]]) as f64,
        32 => i32::from_be_bytes(b.try_into().unwrap()) as f64,
        64 => i64::from_be_bytes(b.try_into().unwrap()) as f64,
        -32 => f32::from_be_bytes(b.try_into().unwrap()) as f64,
        -64 => f64::from_be_bytes(b.try_into().unwrap()),
        other => return Err(format!("unsupported BITPIX {other}")),
    })
}

/// Map physical values into [0, 1]. Unsigned-integer conventions map onto their full range
/// so faint images stay faint; anything else is stretched min..max.
fn normalise(mut v: Vec<f32>, bitpix: i32, bzero: f64, bscale: f64) -> Vec<f32> {
    let unsigned_max = match (bitpix, bzero, bscale) {
        (8, 0.0, 1.0) => Some(255.0),
        (16, 32768.0, 1.0) => Some(65535.0),
        (32, 2147483648.0, 1.0) => Some(4294967295.0),
        _ => None,
    };
    let (lo, hi) = match unsigned_max {
        Some(max) => (0.0, max),
        None => {
            let (lo, hi) = v.iter().filter(|x| x.is_finite()).fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            // Float data already in [0, 1] (Siril, AutoStakkert) is kept as-is.
            if bitpix < 0 && lo >= 0.0 && hi <= 1.0 { (0.0, 1.0) } else { (lo, hi) }
        }
    };
    let span = (hi - lo).max(f32::EPSILON);
    for x in &mut v {
        *x = if x.is_finite() { ((*x - lo) / span).clamp(0.0, 1.0) } else { 0.0 };
    }
    v
}

/// FITS stores the first row at the bottom of the image.
fn flip_vertical(plane: &[f32], width: usize, height: usize) -> Vec<f32> {
    plane.chunks_exact(width).rev().take(height).flatten().copied().collect()
}
