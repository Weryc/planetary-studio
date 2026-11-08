//! Telescopes, cameras, Barlow lenses and filters, and what they mean for an image: the scale
//! in arcsec per pixel, the finest detail the optics can pass (the diffraction cut-off), how
//! far the image is oversampled, the diffraction pattern.
//!
//! The catalogue holds the instruments most used for planetary imaging; anything else is a
//! custom entry with its own numbers.

use crate::capture::CaptureInfo;
use crate::psf::Psf;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Telescope {
    pub name: String,
    pub aperture_mm: f32,
    pub focal_mm: f32,
    /// Central obstruction, fraction of the aperture's diameter (0 for refractors).
    pub obstruction: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub name: String,
    pub pixel_um: f32,
    /// Bayer colour sensor.
    pub colour: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Barlow {
    pub name: String,
    pub factor: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    pub name: String,
    /// Effective wavelength of the light reaching the sensor, nm.
    pub wavelength_nm: f32,
}

fn scope(name: &str, aperture_mm: f32, focal_mm: f32, obstruction: f32) -> Telescope {
    Telescope { name: name.into(), aperture_mm, focal_mm, obstruction }
}

fn camera(name: &str, pixel_um: f32, colour: bool) -> Camera {
    Camera { name: name.into(), pixel_um, colour }
}

pub fn telescopes() -> Vec<Telescope> {
    vec![
        scope("Celestron C5", 127.0, 1250.0, 0.39),
        scope("Celestron C6", 150.0, 1500.0, 0.38),
        scope("Celestron C8 / NexStar 8SE", 203.0, 2032.0, 0.34),
        scope("Celestron C9.25", 235.0, 2350.0, 0.36),
        scope("Celestron C11", 280.0, 2800.0, 0.34),
        scope("Celestron C14", 356.0, 3910.0, 0.32),
        scope("Celestron EdgeHD 8", 203.0, 2032.0, 0.34),
        scope("Celestron EdgeHD 9.25", 235.0, 2350.0, 0.36),
        scope("Celestron EdgeHD 11", 280.0, 2800.0, 0.34),
        scope("Celestron EdgeHD 14", 356.0, 3910.0, 0.32),
        scope("Meade LX90 / LX200 8\"", 203.0, 2000.0, 0.34),
        scope("Meade LX200 10\"", 254.0, 2500.0, 0.37),
        scope("Meade LX200 12\"", 305.0, 3048.0, 0.34),
        scope("Meade LX200 14\"", 356.0, 3556.0, 0.32),
        scope("Sky-Watcher Skymax 102", 102.0, 1300.0, 0.29),
        scope("Sky-Watcher Skymax 127", 127.0, 1500.0, 0.31),
        scope("Sky-Watcher Skymax 150", 150.0, 1800.0, 0.28),
        scope("Sky-Watcher Skymax 180", 180.0, 2700.0, 0.27),
        scope("Intes-Micro MN/MK 152", 152.0, 1800.0, 0.27),
        scope("Newton 150/750", 150.0, 750.0, 0.25),
        scope("Newton 200/1000", 200.0, 1000.0, 0.24),
        scope("Newton / Dobson 200/1200", 200.0, 1200.0, 0.20),
        scope("Newton / Dobson 250/1200", 254.0, 1200.0, 0.22),
        scope("Newton / Dobson 300/1500", 305.0, 1500.0, 0.20),
        scope("Newton / Dobson 350/1600", 355.0, 1650.0, 0.20),
        scope("Newton / Dobson 400/1800", 406.0, 1800.0, 0.20),
        scope("Ritchey-Chrétien 8\"", 203.0, 1624.0, 0.47),
        scope("Ritchey-Chrétien 10\"", 254.0, 2000.0, 0.45),
        scope("Rifrattore 80/600 ED", 80.0, 600.0, 0.0),
        scope("Rifrattore 102/714 apo", 102.0, 714.0, 0.0),
        scope("Rifrattore 100/900 ED", 100.0, 900.0, 0.0),
        scope("Rifrattore 120/900 ED", 120.0, 900.0, 0.0),
        scope("Rifrattore 130/910 apo", 130.0, 910.0, 0.0),
        scope("Rifrattore 150/1200 ED", 150.0, 1200.0, 0.0),
        scope("Takahashi FC-100DL", 100.0, 900.0, 0.0),
        scope("Takahashi TOA-130", 130.0, 1000.0, 0.0),
    ]
}

pub fn cameras() -> Vec<Camera> {
    vec![
        camera("ZWO ASI120MC", 3.75, true),
        camera("ZWO ASI120MM", 3.75, false),
        camera("ZWO ASI174MM", 5.86, false),
        camera("ZWO ASI178MC", 2.4, true),
        camera("ZWO ASI178MM", 2.4, false),
        camera("ZWO ASI183MC", 2.4, true),
        camera("ZWO ASI183MM", 2.4, false),
        camera("ZWO ASI224MC", 3.75, true),
        camera("ZWO ASI290MC", 2.9, true),
        camera("ZWO ASI290MM", 2.9, false),
        camera("ZWO ASI385MC", 3.75, true),
        camera("ZWO ASI462MC", 2.9, true),
        camera("ZWO ASI462MM", 2.9, false),
        camera("ZWO ASI482MC", 5.8, true),
        camera("ZWO ASI485MC", 2.9, true),
        camera("ZWO ASI533MC", 3.76, true),
        camera("ZWO ASI585MC", 2.9, true),
        camera("ZWO ASI585MM", 2.9, false),
        camera("ZWO ASI662MC", 2.9, true),
        camera("ZWO ASI664MC", 2.9, true),
        camera("ZWO ASI676MC", 2.0, true),
        camera("ZWO ASI678MC", 2.0, true),
        camera("ZWO ASI678MM", 2.0, false),
        camera("ZWO ASI715MC", 1.45, true),
        camera("SVBONY SV105", 3.0, true),
        camera("SVBONY SV205", 3.75, true),
        camera("SVBONY SV305", 2.9, true),
        camera("SVBONY SV305 Pro", 2.9, true),
        camera("SVBONY SV405CC", 3.76, true),
        camera("SVBONY SV605CC", 3.76, true),
        camera("SVBONY SV705C", 2.9, true),
        camera("SVBONY SV905C", 2.9, true),
        camera("Player One Ceres-C", 3.75, true),
        camera("Player One Neptune-C II", 2.9, true),
        camera("Player One Uranus-C", 2.9, true),
        camera("Player One Mars-C II", 2.9, true),
        camera("Player One Saturn-C", 3.76, true),
        camera("Player One Apollo-M Mini", 2.9, false),
        camera("QHY5III174M", 5.86, false),
        camera("QHY5III290M", 2.9, false),
        camera("QHY5III462C", 2.9, true),
        camera("QHY5III485C", 2.9, true),
        camera("QHY5III585C", 2.9, true),
        camera("QHY5III678C", 2.0, true),
        camera("QHY5III715C", 1.45, true),
        camera("ToupTek G3M462C", 2.9, true),
        camera("ToupTek G3M678C", 2.0, true),
        camera("The Imaging Source DMK 21AU618", 5.6, false),
        camera("The Imaging Source DFK 21AU618", 5.6, true),
        camera("The Imaging Source DMK 41AU02", 4.65, false),
        camera("Philips SPC900NC", 5.6, true),
    ]
}

pub fn barlows() -> Vec<Barlow> {
    let b = |name: &str, factor: f32| Barlow { name: name.into(), factor };
    vec![
        b("Nessuna", 1.0),
        b("Riduttore 0,63×", 0.63),
        b("Barlow 1,5×", 1.5),
        b("Barlow 2×", 2.0),
        b("Barlow 2,5×", 2.5),
        b("Barlow 3×", 3.0),
        b("Barlow 4×", 4.0),
        b("Barlow 5×", 5.0),
        b("Tele Vue Powermate 2,5×", 2.5),
        b("Tele Vue Powermate 4×", 4.0),
        b("Tele Vue Powermate 5×", 5.0),
        b("Baader VIP 2×", 2.0),
        b("APM ED Barlow 2,7×", 2.7),
        b("Baader FFC (Fluorite Flat Field Converter) 3×–8×", 4.0),
    ]
}

pub fn filters() -> Vec<Filter> {
    let f = |name: &str, wavelength_nm: f32| Filter { name: name.into(), wavelength_nm };
    // Long-pass IR filters: the effective wavelength on a typical planetary sensor, whose
    // sensitivity falls off towards 1 µm.
    vec![
        f("Nessuno / IR-cut (luminanza)", 550.0),
        f("L", 550.0),
        f("R", 630.0),
        f("G", 530.0),
        f("B", 460.0),
        f("IR 610 nm", 690.0),
        f("IR 642 nm", 720.0),
        f("IR 685 nm", 760.0),
        f("IR 742 nm", 800.0),
        f("IR 807 nm", 860.0),
        f("Metano CH4 889 nm", 889.0),
        f("UV 350–400 nm", 375.0),
        f("Baader Contrast Booster", 560.0),
        f("Neodimio", 560.0),
        f("H-alpha (solare)", 656.3),
        f("Continuum 540 nm", 540.0),
        f("Calcio K 393 nm", 393.4),
    ]
}

/// The equipment used for an image. Every part is optional; whatever is known is used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Setup {
    pub telescope: Option<Telescope>,
    pub camera: Option<Camera>,
    pub barlow: Option<Barlow>,
    pub filter: Option<Filter>,
    /// Camera binning (1 = none).
    #[serde(default = "one")]
    pub binning: u32,
    /// AutoStakkert drizzle or resampling (1 = none, 1.5 = 150 %).
    #[serde(default = "one_f")]
    pub drizzle: f32,
    /// Effective focal length when known directly (from a log), overriding telescope × Barlow.
    #[serde(default)]
    pub focal_mm: Option<f32>,
}

impl Default for Setup {
    fn default() -> Self {
        Self { telescope: None, camera: None, barlow: None, filter: None, binning: 1, drizzle: 1.0, focal_mm: None }
    }
}

fn one() -> u32 {
    1
}

fn one_f() -> f32 {
    1.0
}

/// What the equipment means for the image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Optics {
    pub arcsec_per_px: f32,
    pub focal_ratio: f32,
    pub wavelength_nm: f32,
    /// Finest frequency the aperture passes, cycles/px.
    pub cutoff: f32,
    /// How many times finer than needed the sampling is: 1 = Nyquist (the cut-off at 0.5
    /// cycles/px), 2 = twice as many pixels as the optics can fill.
    pub oversampling: f32,
    /// FWHM of the diffraction pattern, px.
    pub airy_fwhm_px: f32,
    pub psf: Psf,
}

impl Setup {
    pub fn is_empty(&self) -> bool {
        self.telescope.is_none() && self.camera.is_none() && self.barlow.is_none() && self.filter.is_none() && self.focal_mm.is_none()
    }

    /// Effective focal length, mm.
    pub fn focal(&self) -> Option<f32> {
        self.focal_mm.or_else(|| self.telescope.as_ref().map(|t| t.focal_mm * self.barlow.as_ref().map_or(1.0, |b| b.factor)))
    }

    /// Size of one pixel of the final image on the sensor, µm.
    pub fn pixel(&self) -> Option<f32> {
        self.camera.as_ref().map(|c| c.pixel_um * self.binning.max(1) as f32 / self.drizzle.max(0.1))
    }

    /// Wavelength used for the diffraction: the filter's, else green light.
    pub fn wavelength(&self) -> f32 {
        self.filter.as_ref().map_or(550.0, |f| f.wavelength_nm)
    }

    /// Everything that follows from the equipment; needs the telescope (aperture), the focal
    /// length and the pixel size.
    pub fn optics(&self) -> Option<Optics> {
        let aperture = self.telescope.as_ref()?.aperture_mm;
        let (focal, pixel) = (self.focal()?, self.pixel()?);
        let wavelength_nm = self.wavelength();
        let arcsec_per_px = 206.264_8 * pixel / focal;
        let focal_ratio = focal / aperture;
        // D/λ cycles per radian, times radians per pixel = p / (λ·N).
        let cutoff = pixel / (wavelength_nm * 1e-3 * focal_ratio);
        let obstruction = self.telescope.as_ref().map_or(0.0, |t| t.obstruction);
        let psf = Psf::Airy { aperture_mm: aperture, wavelength_nm, arcsec_per_px, obstruction };
        Some(Optics { arcsec_per_px, focal_ratio, wavelength_nm, cutoff, oversampling: 0.5 / cutoff, airy_fwhm_px: psf.fwhm_px(), psf })
    }

    /// The setup a capture log describes: catalogue entries matched by name (custom ones
    /// when the name is unknown but numbers are given), the rest left as it was in `base`.
    pub fn from_capture(info: &CaptureInfo, base: &Setup) -> Setup {
        let mut s = base.clone();
        if let Some(name) = &info.camera {
            s.camera = find(&cameras(), |c| &c.name, name).or_else(|| s.camera.clone().filter(|c| same(&c.name, name)));
        }
        if let Some(name) = &info.telescope {
            s.telescope = find(&telescopes(), |t| &t.name, name).or(s.telescope);
        }
        if let Some(name) = &info.filter {
            s.filter = filter_named(name).or(s.filter);
        }
        if let Some(b) = info.binning {
            s.binning = b.max(1);
        }
        if let Some(d) = info.drizzle {
            s.drizzle = d.max(0.1);
        }
        if info.focal_mm.is_some() {
            s.focal_mm = info.focal_mm;
        }
        s
    }
}

/// Names compared without case, spaces or punctuation.
fn squash(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase()
}

fn same(a: &str, b: &str) -> bool {
    squash(a) == squash(b)
}

/// The catalogue entry a free-form name refers to: exact (squashed) name, else the entry whose
/// model (the name without its brand) appears in it, the longest model first so "ASI462MC"
/// doesn't match "ASI462MM" and "C9.25" isn't taken for "C9".
fn find<T: Clone>(list: &[T], name_of: impl Fn(&T) -> &String, name: &str) -> Option<T> {
    let wanted = squash(name);
    if let Some(t) = list.iter().find(|t| squash(name_of(t)) == wanted) {
        return Some(t.clone());
    }
    let mut candidates: Vec<(usize, &T)> = list
        .iter()
        .filter_map(|t| {
            let full = name_of(t);
            let model = full.split_whitespace().last().map(squash).unwrap_or_default();
            (model.len() >= 3 && wanted.contains(&model)).then_some((model.len(), t))
        })
        .collect();
    candidates.sort_by_key(|c| std::cmp::Reverse(c.0));
    candidates.first().map(|(_, t)| (*t).clone())
}

/// The catalogue filter a file name or log means (`IR685`, `R`, `RGB`, `IR-cut`…).
pub fn filter_named(name: &str) -> Option<Filter> {
    let all = filters();
    let n = squash(name);
    let pick = |prefix: &str| all.iter().find(|f| f.name.starts_with(prefix)).cloned();
    match n.as_str() {
        "rgb" | "l" | "lum" | "ircut" | "uvir" | "uvircut" | "none" | "nessuno" | "clear" => pick("Nessuno"),
        "r" | "red" => pick("R"),
        "g" | "green" => pick("G"),
        "b" | "blue" => pick("B"),
        "ch4" | "methane" | "metano" => pick("Metano"),
        "uv" => pick("UV"),
        "ha" | "halpha" => pick("H-alpha"),
        "ir" | "nir" => pick("IR 685"),
        _ if n.starts_with("ir") => {
            let nm: f32 = n[2..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()?;
            all.iter().filter(|f| f.name.starts_with("IR ")).min_by(|a, b| (cut_on(a) - nm).abs().total_cmp(&(cut_on(b) - nm).abs())).cloned()
        }
        _ => all.iter().find(|f| squash(&f.name) == n).cloned(),
    }
}

/// The cut-on wavelength in an IR filter's name ("IR 685 nm" → 685).
fn cut_on(f: &Filter) -> f32 {
    f.name.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture;

    fn c8_462_2x() -> Setup {
        Setup {
            telescope: telescopes().into_iter().find(|t| t.name.starts_with("Celestron C8")),
            camera: cameras().into_iter().find(|c| c.name == "ZWO ASI462MC"),
            barlow: barlows().into_iter().find(|b| b.name == "Barlow 2×"),
            filter: filter_named("RGB"),
            ..Default::default()
        }
    }

    /// C8 at f/20 with 2.9 µm pixels in green light: 0.147″/px, cut-off p/(λN) = 0.264
    /// cycles/px, oversampled about 1.9×.
    #[test]
    fn optics_of_a_classic_setup() {
        let o = c8_462_2x().optics().unwrap();
        assert!((o.arcsec_per_px - 0.147).abs() < 0.002, "{o:?}");
        assert!((o.focal_ratio - 20.0).abs() < 0.1);
        assert!((o.cutoff - 0.2636).abs() < 0.003);
        assert!((o.oversampling - 1.9).abs() < 0.05);
    }

    #[test]
    fn names_from_logs_find_the_catalogue() {
        assert_eq!(find(&cameras(), |c| &c.name, "ZWO ASI462MM").unwrap().name, "ZWO ASI462MM");
        assert_eq!(find(&cameras(), |c| &c.name, "ASI462MC (USB3)").unwrap().name, "ZWO ASI462MC");
        assert_eq!(find(&cameras(), |c| &c.name, "SVBONY SV705C").unwrap().pixel_um, 2.9);
        assert_eq!(find(&telescopes(), |t| &t.name, "C9.25").unwrap().aperture_mm, 235.0);
        assert_eq!(filter_named("IR685").unwrap().name, "IR 685 nm");
        assert_eq!(filter_named("IR650").unwrap().name, "IR 642 nm");
        assert_eq!(filter_named("RGB").unwrap().wavelength_nm, 550.0);
        let info = capture::from_log("Camera=ZWO ASI224MC\nFilter=R\nFocalLength=5600mm\nBinning=no\n");
        let s = Setup::from_capture(&info, &c8_462_2x());
        assert_eq!(s.camera.as_ref().unwrap().name, "ZWO ASI224MC");
        assert_eq!(s.focal(), Some(5600.0));
        assert_eq!(s.filter.unwrap().name, "R");
    }
}
