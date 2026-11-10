//! Processing core: image model, wavelet and deconvolution algorithms, and the pipeline that
//! chains them. Has no GUI dependencies so it can back a CLI or batch tool as well.

pub mod adjust;
pub mod align;
pub mod analysis;
pub mod bxt;
pub mod capture;
pub mod color;
pub mod conv;
pub mod deconv;
pub mod dering;
pub mod enhance;
pub mod ephemeris;
pub mod equipment;
pub mod fft;
pub mod fits;
pub mod geometry;
pub mod i18n;
mod i18n_en;
pub mod identify;
pub mod gpu;
pub mod image;
pub mod mask;
pub mod moons;
pub mod mosaic;
pub mod pipeline;
pub mod preset;
pub mod smart;
pub mod psf;
pub mod resample;
pub mod selene;
pub mod spectrum;
pub mod suncolor;
pub mod wavelet;

pub use image::Image;
