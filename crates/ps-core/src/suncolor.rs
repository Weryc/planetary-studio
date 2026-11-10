//! False colour for solar images: a grey (or colour) frame becomes yellow, gold, warm white…
//!
//! Each palette is a "gradient map" built from one exponent per channel: out_c = L^γ_c, with L
//! the pixel's brightness. Black stays black and pure white stays white; in between, a channel
//! with γ < 1 is lifted and one with γ > 1 is held back. Red lifted and blue held back gives
//! yellows and oranges whose shadows are deeper in colour than the highlights — as in the
//! photographs taken through solar filters — and the tone of every pixel follows only from
//! its brightness, so the result is exactly reproducible.

use crate::i18n::tl;
use crate::Image;
use crate::color::luma;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Palette {
    /// White light, barely warm (like a Baader film image).
    NeutralWhite,
    WarmWhite,
    PaleYellow,
    SunYellow,
    Gold,
    /// Deep orange, the classic H-alpha look.
    HalphaOrange,
    HalphaRed,
    /// Calcium K: violet-blue.
    CalciumK,
}

impl Palette {
    pub const ALL: [Palette; 8] = [Palette::NeutralWhite, Palette::WarmWhite, Palette::PaleYellow, Palette::SunYellow, Palette::Gold, Palette::HalphaOrange, Palette::HalphaRed, Palette::CalciumK];

    pub fn label(self) -> &'static str {
        match self {
            Palette::NeutralWhite => tl("Bianco neutro"),
            Palette::WarmWhite => tl("Bianco caldo"),
            Palette::PaleYellow => tl("Giallo pallido"),
            Palette::SunYellow => tl("Giallo sole"),
            Palette::Gold => tl("Oro"),
            Palette::HalphaOrange => tl("Arancio H-alfa"),
            Palette::HalphaRed => tl("Rosso H-alfa"),
            Palette::CalciumK => tl("Viola Calcio K"),
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Palette::NeutralWhite => tl("Luce bianca appena calda, come con il filtro in pellicola: per macchie e granulazione."),
            Palette::WarmWhite => tl("Bianco avorio: le zone scure prendono un tono crema."),
            Palette::PaleYellow => tl("Giallo tenue, i chiari restano quasi bianchi."),
            Palette::SunYellow => tl("Il giallo con cui di solito si immagina il Sole."),
            Palette::Gold => tl("Giallo-arancio caldo e saturo, ombre ambrate."),
            Palette::HalphaOrange => tl("L'aspetto classico delle riprese in H-alfa: arancio intenso."),
            Palette::HalphaRed => tl("H-alfa rosso cupo, utile per protuberanze e filamenti."),
            Palette::CalciumK => tl("Viola-blu delle riprese in Calcio K."),
        }
    }

    /// Exponent per channel (R, G, B).
    pub fn gammas(self) -> [f32; 3] {
        match self {
            Palette::NeutralWhite => [0.96, 1.0, 1.1],
            Palette::WarmWhite => [0.9, 1.0, 1.3],
            Palette::PaleYellow => [0.8, 0.92, 1.5],
            Palette::SunYellow => [0.6, 0.88, 2.0],
            Palette::Gold => [0.5, 0.95, 2.6],
            Palette::HalphaOrange => [0.45, 1.3, 2.8],
            Palette::HalphaRed => [0.4, 1.8, 3.2],
            Palette::CalciumK => [1.25, 1.6, 0.7],
        }
    }

    /// The colour a pixel of brightness `l` takes.
    pub fn map(self, l: f32) -> [f32; 3] {
        let l = l.clamp(0.0, 1.0);
        self.gammas().map(|g| l.powf(g))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SunColorParams {
    pub palette: Palette,
    /// 0 = the image as it is, 1 = the full palette.
    pub strength: f32,
}

impl Default for SunColorParams {
    fn default() -> Self {
        Self { palette: Palette::SunYellow, strength: 1.0 }
    }
}

impl SunColorParams {
    pub fn neutral() -> Self {
        Self { strength: 0.0, ..Default::default() }
    }

    pub fn is_neutral(&self) -> bool {
        self.strength <= 0.0
    }
}

/// The image coloured with the palette (always RGB).
pub fn apply(img: &Image, p: &SunColorParams) -> Image {
    let n = img.width() * img.height();
    let s = p.strength.clamp(0.0, 1.0);
    let ch = img.channels();
    // A lookup table: 4096 levels are finer than 16-bit exports can tell apart after the curve.
    const LEVELS: usize = 4096;
    let lut: Vec<[f32; 3]> = (0..=LEVELS).map(|k| p.palette.map(k as f32 / LEVELS as f32)).collect();
    let mut out: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0; n]).collect();
    let (r, rest) = out.split_at_mut(1);
    let (g, b) = rest.split_at_mut(1);
    r[0].par_iter_mut().zip(g[0].par_iter_mut()).zip(b[0].par_iter_mut()).enumerate().for_each(|(i, ((r, g), b))| {
        let orig = if ch.len() == 3 { [ch[0][i], ch[1][i], ch[2][i]] } else { [ch[0][i]; 3] };
        let l = luma(orig).clamp(0.0, 1.0);
        let f = l * LEVELS as f32;
        let k = (f as usize).min(LEVELS - 1);
        let t = f - k as f32;
        let c: [f32; 3] = std::array::from_fn(|c| lut[k][c] + (lut[k + 1][c] - lut[k][c]) * t);
        *r = orig[0] + (c[0] - orig[0]) * s;
        *g = orig[1] + (c[1] - orig[1]) * s;
        *b = orig[2] + (c[2] - orig[2]) * s;
    });
    Image::new(img.width(), img.height(), out, img.source_bit_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ends_stay_black_and_white() {
        for p in Palette::ALL {
            assert_eq!(p.map(0.0), [0.0; 3]);
            assert!(p.map(1.0).iter().all(|v| (v - 1.0).abs() < 1e-6));
        }
    }

    #[test]
    fn yellows_are_yellow() {
        for p in [Palette::PaleYellow, Palette::SunYellow, Palette::Gold, Palette::HalphaOrange] {
            let [r, g, b] = p.map(0.5);
            assert!(r > g && g > b, "{p:?}: {r} {g} {b}");
        }
        let [r, g, b] = Palette::CalciumK.map(0.5);
        assert!(b > r && b > g);
    }

    #[test]
    fn mono_becomes_colour_and_zero_strength_keeps_it() {
        let img = Image::new(3, 1, vec![vec![0.0, 0.5, 1.0]], 16);
        let out = apply(&img, &SunColorParams::default());
        assert!(!out.is_mono());
        let [r, g, b] = Palette::SunYellow.map(0.5);
        assert!((out.channels()[0][1] - r).abs() < 1e-3 && (out.channels()[1][1] - g).abs() < 1e-3 && (out.channels()[2][1] - b).abs() < 1e-3);
        let same = apply(&img, &SunColorParams::neutral());
        assert!(same.channels().iter().all(|c| c == &img.channels()[0]));
    }
}
