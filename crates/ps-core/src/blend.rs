//! How a mask's adjusted version of the image is laid over what is below it: the classic
//! blend modes of layered editors. `out = base + mask · (blended − base)`, where `blended` is
//! the mode's result for the base (backdrop) and the adjusted image (source).

use crate::i18n::tl;
use crate::Image;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    #[default]
    Normal,
    Darken,
    Multiply,
    ColorBurn,
    Lighten,
    Screen,
    ColorDodge,
    Overlay,
    SoftLight,
    HardLight,
    Difference,
    /// Hue and saturation of the adjusted image, brightness of the one below.
    Color,
    /// Brightness of the adjusted image, colours of the one below.
    Luminosity,
}

impl BlendMode {
    pub const ALL: [BlendMode; 13] = [
        BlendMode::Normal,
        BlendMode::Darken,
        BlendMode::Multiply,
        BlendMode::ColorBurn,
        BlendMode::Lighten,
        BlendMode::Screen,
        BlendMode::ColorDodge,
        BlendMode::Overlay,
        BlendMode::SoftLight,
        BlendMode::HardLight,
        BlendMode::Difference,
        BlendMode::Color,
        BlendMode::Luminosity,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Normal => tl("Normale"),
            BlendMode::Darken => tl("Scurisci"),
            BlendMode::Multiply => tl("Moltiplica"),
            BlendMode::ColorBurn => tl("Colore brucia"),
            BlendMode::Lighten => tl("Schiarisci"),
            BlendMode::Screen => tl("Scherma"),
            BlendMode::ColorDodge => tl("Scherma colore"),
            BlendMode::Overlay => tl("Sovrapponi"),
            BlendMode::SoftLight => tl("Luce soffusa"),
            BlendMode::HardLight => tl("Luce intensa"),
            BlendMode::Difference => tl("Differenza"),
            BlendMode::Color => tl("Colore"),
            BlendMode::Luminosity => tl("Luminosità"),
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            BlendMode::Normal => tl("La zona della maschera prende l'immagine regolata."),
            BlendMode::Darken => tl("Tiene il più scuro fra l'immagine sotto e quella regolata."),
            BlendMode::Multiply => tl("Scurisce: i chiari restano, i toni scuri si approfondiscono."),
            BlendMode::ColorBurn => tl("Scurisce aumentando il contrasto dei toni scuri."),
            BlendMode::Lighten => tl("Tiene il più chiaro fra l'immagine sotto e quella regolata."),
            BlendMode::Screen => tl("Schiarisce: i toni scuri restano, i chiari si illuminano."),
            BlendMode::ColorDodge => tl("Schiarisce aumentando il contrasto dei toni chiari."),
            BlendMode::Overlay => tl("Aumenta il contrasto: schiarisce i chiari e scurisce gli scuri dell'immagine sotto."),
            BlendMode::SoftLight => tl("Come Sovrapponi, ma più delicato."),
            BlendMode::HardLight => tl("Come Sovrapponi, ma guidato dall'immagine regolata: più energico."),
            BlendMode::Difference => tl("La differenza fra le due immagini: utile per vedere cosa cambia la regolazione."),
            BlendMode::Color => tl("Prende i colori dall'immagine regolata e la luminosità da quella sotto."),
            BlendMode::Luminosity => tl("Prende la luminosità dall'immagine regolata e i colori da quella sotto."),
        }
    }

    /// The result for one channel's value (0‥1) of the backdrop `b` and the source `s`. Not for
    /// [`BlendMode::Color`] and [`BlendMode::Luminosity`], which need the three channels.
    fn channel(self, b: f32, s: f32) -> f32 {
        match self {
            BlendMode::Normal => s,
            BlendMode::Darken => b.min(s),
            BlendMode::Multiply => b * s,
            BlendMode::ColorBurn => {
                if b >= 1.0 {
                    1.0
                } else if s <= 0.0 {
                    0.0
                } else {
                    1.0 - ((1.0 - b) / s).min(1.0)
                }
            }
            BlendMode::Lighten => b.max(s),
            BlendMode::Screen => 1.0 - (1.0 - b) * (1.0 - s),
            BlendMode::ColorDodge => {
                if b <= 0.0 {
                    0.0
                } else if s >= 1.0 {
                    1.0
                } else {
                    (b / (1.0 - s)).min(1.0)
                }
            }
            BlendMode::Overlay => BlendMode::HardLight.channel(s, b),
            BlendMode::HardLight => {
                if s <= 0.5 {
                    b * 2.0 * s
                } else {
                    BlendMode::Screen.channel(b, 2.0 * s - 1.0)
                }
            }
            BlendMode::SoftLight => {
                if s <= 0.5 {
                    b - (1.0 - 2.0 * s) * b * (1.0 - b)
                } else {
                    let d = if b <= 0.25 { ((16.0 * b - 12.0) * b + 4.0) * b } else { b.sqrt() };
                    b + (2.0 * s - 1.0) * (d - b)
                }
            }
            BlendMode::Difference => (b - s).abs(),
            BlendMode::Color | BlendMode::Luminosity => s,
        }
    }
}

fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn clip_colour(c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let (n, x) = (c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]));
    let mut c = c;
    if n < 0.0 && (l - n).abs() > 1e-9 {
        c = c.map(|v| l + (v - l) * l / (l - n));
    }
    if x > 1.0 && (x - l).abs() > 1e-9 {
        c = c.map(|v| l + (v - l) * (1.0 - l) / (x - l));
    }
    c
}

fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_colour(c.map(|v| v + d))
}

/// Lays `layer` over `base` through `mask`, in `mode`.
pub fn composite(base: &Image, layer: &Image, mask: &[f32], mode: BlendMode) -> Image {
    let mut channels = base.channels().to_vec();
    composite_into(&mut channels, layer.channels(), mask, mode);
    Image::new(base.width(), base.height(), channels, base.source_bit_depth)
}

/// The same, over `out` in place.
pub fn composite_into(out: &mut [Vec<f32>], layer: &[Vec<f32>], mask: &[f32], mode: BlendMode) {
    let clamp = |v: f32| v.clamp(0.0, 1.0);
    match mode {
        BlendMode::Normal => out.par_iter_mut().zip(layer).for_each(|(o, src)| {
            for ((o, &s), &m) in o.iter_mut().zip(src).zip(mask) {
                *o += m * (s - *o);
            }
        }),
        BlendMode::Color | BlendMode::Luminosity if out.len() == 3 && layer.len() == 3 => {
            let [r, g, b] = out else { return };
            r.par_iter_mut().zip(g.par_iter_mut()).zip(b.par_iter_mut()).enumerate().for_each(|(i, ((r, g), b))| {
                let m = mask[i];
                if m <= 0.0 {
                    return;
                }
                let backdrop = [*r, *g, *b];
                let source = [layer[0][i], layer[1][i], layer[2][i]];
                let mixed = if mode == BlendMode::Color { set_lum(source, lum(backdrop)) } else { set_lum(backdrop, lum(source)) };
                *r += m * (mixed[0] - *r);
                *g += m * (mixed[1] - *g);
                *b += m * (mixed[2] - *b);
            });
        }
        // One channel: colour has nothing to carry, brightness is the whole picture.
        BlendMode::Color => {}
        BlendMode::Luminosity => composite_into(out, layer, mask, BlendMode::Normal),
        _ => out.par_iter_mut().zip(layer).for_each(|(o, src)| {
            for ((o, &s), &m) in o.iter_mut().zip(src).zip(mask) {
                if m > 0.0 {
                    let mixed = mode.channel(clamp(*o), clamp(s));
                    *o += m * (mixed - *o);
                }
            }
        }),
    }
}
