//! Bridge to RC Astro's BlurXTerminator through its stand-alone command-line tool
//! (`rc-astro bxt …`, see rc-astro.com/stand-alone-rc-astro-tools). The image goes out and
//! comes back as 32-bit float TIFF, so nothing is lost in the round trip. (FITS was tried first,
//! but rc-astro reads and writes its row order inconsistently, returning the image flipped.)

use crate::Image;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BxtParams {
    /// `--sharpen-nonstellar`, 0‥1. Planets, Moon and Sun are entirely "non-stellar".
    pub sharpen_nonstellar: f32,
    /// `--sharpen-stars`, 0‥0.7.
    pub sharpen_stars: f32,
    /// Let BlurXTerminator measure the PSF (`--auto-nonstellar-psf`). It does so from stars,
    /// which planetary and lunar images don't have, so it is off by default and the diameter
    /// comes from the blur measured on the limb.
    pub auto_psf: bool,
    /// `--nonstellar-diameter` in pixels, used when `auto_psf` is off.
    pub psf_diameter: f32,
    /// `--correct-only`: fix aberrations without sharpening.
    pub correct_only: bool,
    /// `--lunar-planetary`: model tuned for the Moon, the Sun and planets (CLI ≥ 2.6).
    #[serde(default = "yes")]
    pub lunar_planetary: bool,
    /// Run the tool this many times in a row (each pass sharpens the previous result).
    #[serde(default = "one")]
    pub passes: u32,
}

fn one() -> u32 {
    1
}

fn yes() -> bool {
    true
}

impl Default for BxtParams {
    fn default() -> Self {
        Self { sharpen_nonstellar: 1.0, sharpen_stars: 0.0, auto_psf: false, psf_diameter: 3.0, correct_only: false, lunar_planetary: true, passes: 1 }
    }
}

/// Run all the requested passes, each on the previous result; `progress` gets 0‥1.
pub fn run_passes(img: &Image, p: &BxtParams, executable: &Path, cancel: &dyn Fn() -> bool, progress: &dyn Fn(f32)) -> Result<Option<Image>, String> {
    let passes = p.passes.clamp(1, 5);
    let mut current = img.clone();
    for k in 0..passes {
        progress(k as f32 / passes as f32);
        match run(&current, p, executable, cancel)? {
            Some(out) => current = out,
            None => return Ok(None),
        }
    }
    progress(1.0);
    Ok(Some(current))
}

/// Run BlurXTerminator on `img`. `Ok(None)` means cancelled (the process is killed).
pub fn run(img: &Image, p: &BxtParams, executable: &Path, cancel: &dyn Fn() -> bool) -> Result<Option<Image>, String> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!("planetary-studio-bxt-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).map_err(|e| format!("cartella temporanea: {e}"))?;
    let result = run_in(&dir, img, p, executable, cancel);
    std::fs::remove_dir_all(&dir).ok();
    result
}

fn run_in(dir: &Path, img: &Image, p: &BxtParams, executable: &Path, cancel: &dyn Fn() -> bool) -> Result<Option<Image>, String> {
    let (input, output, log) = (dir.join("in.tif"), dir.join("out.tif"), dir.join("log.txt"));
    write_tiff_f32(&input, img).map_err(|e| format!("scrittura file temporaneo: {e}"))?;
    let log_file = std::fs::File::create(&log).map_err(|e| e.to_string())?;

    let mut cmd = Command::new(executable);
    cmd.arg("bxt").arg(&input).arg("--output").arg(&output).args(["--overwrite", "--depth", "32F"]);
    if p.correct_only {
        cmd.arg("--correct-only");
    } else {
        cmd.arg("--sharpen-nonstellar").arg(format!("{:.3}", p.sharpen_nonstellar.clamp(0.0, 1.0)));
        cmd.arg("--sharpen-stars").arg(format!("{:.3}", p.sharpen_stars.clamp(0.0, 0.7)));
    }
    let flags = flags(executable);
    if !p.auto_psf {
        let diameter = p.psf_diameter.clamp(0.0, 8.0);
        let value = if flags.takes_radius { diameter / 2.0 } else { diameter };
        cmd.arg(flags.no_auto_psf).arg(flags.psf_size).arg(format!("{value:.2}"));
    }
    if p.lunar_planetary && flags.lunar_planetary {
        cmd.arg("--lunar-planetary");
    }
    cmd.stdin(Stdio::null()).stdout(log_file.try_clone().map_err(|e| e.to_string())?).stderr(log_file);

    let mut child = cmd.spawn().map_err(|e| format!("impossibile avviare {}: {e}", executable.display()))?;
    let status = loop {
        if cancel() {
            child.kill().ok();
            child.wait().ok();
            return Ok(None);
        }
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status,
            None => std::thread::sleep(Duration::from_millis(30)),
        }
    };
    if !status.success() {
        let text = std::fs::read_to_string(&log).unwrap_or_default();
        let last = text.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("nessun messaggio");
        return Err(format!("rc-astro è terminato con errore ({status}): {last}"));
    }
    let out = read_tiff_f32(&output).map_err(|e| format!("lettura del risultato: {e}"))?;
    if out.width() != img.width() || out.height() != img.height() || out.channels().len() != img.channels().len() {
        return Err("il risultato ha dimensioni diverse dall'originale".into());
    }
    Ok(Some(Image::new(out.width(), out.height(), out.channels().to_vec(), img.source_bit_depth)))
}

fn write_tiff_f32(path: &Path, img: &Image) -> Result<(), String> {
    use tiff::encoder::{TiffEncoder, colortype};
    let file = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
    let mut encoder = TiffEncoder::new(file).map_err(|e| e.to_string())?;
    let (w, h) = (img.width() as u32, img.height() as u32);
    match img.channels() {
        [mono] => encoder.write_image::<colortype::Gray32Float>(w, h, mono),
        [r, g, b] => {
            let data: Vec<f32> = r.iter().zip(g).zip(b).flat_map(|((&r, &g), &b)| [r, g, b]).collect();
            encoder.write_image::<colortype::RGB32Float>(w, h, &data)
        }
        _ => unreachable!("images are mono or RGB"),
    }
    .map_err(|e| e.to_string())
}

fn read_tiff_f32(path: &Path) -> Result<Image, String> {
    use tiff::ColorType;
    use tiff::decoder::{Decoder, DecodingResult, Limits};
    let file = std::io::BufReader::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
    let mut decoder = Decoder::new(file).map_err(|e| e.to_string())?.with_limits(Limits::unlimited());
    let (w, h) = decoder.dimensions().map_err(|e| e.to_string())?;
    let stride = match decoder.colortype().map_err(|e| e.to_string())? {
        ColorType::Gray(_) => 1,
        ColorType::GrayA(_) => 2,
        ColorType::RGB(_) => 3,
        ColorType::RGBA(_) => 4,
        other => return Err(format!("formato colore non supportato: {other:?}")),
    };
    let data: Vec<f32> = match decoder.read_image().map_err(|e| e.to_string())? {
        DecodingResult::F32(v) => v,
        DecodingResult::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U16(v) => v.into_iter().map(|x| x as f32 / 65535.0).collect(),
        DecodingResult::U8(v) => v.into_iter().map(|x| x as f32 / 255.0).collect(),
        _ => return Err("profondità di bit non supportata".into()),
    };
    let planes = if stride >= 3 { 3 } else { 1 };
    let channels = (0..planes).map(|c| data.iter().skip(c).step_by(stride).copied().collect()).collect();
    Ok(Image::new(w as usize, h as usize, channels, 32))
}

/// Option names differ between CLI releases (the PSF option was a "radius" in 1.x and is a
/// "diameter" in 2.x; lunar/planetary mode appeared in 2.6). Read them from the tool's help.
#[derive(Clone, Copy)]
struct Flags {
    no_auto_psf: &'static str,
    psf_size: &'static str,
    /// 1.x takes the PSF size as a radius, 2.x as a diameter.
    takes_radius: bool,
    lunar_planetary: bool,
}

fn flags(executable: &Path) -> Flags {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Flags>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(f) = cache.lock().unwrap().get(executable) {
        return *f;
    }
    let help = Command::new(executable)
        .args(["bxt", "--help"])
        .stdin(Stdio::null())
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr))
        .unwrap_or_default();
    let f = if help.contains("--nonstellar-diameter") {
        Flags { no_auto_psf: "--no-auto-nonstellar-psf", psf_size: "--nonstellar-diameter", takes_radius: false, lunar_planetary: help.contains("--lunar-planetary") }
    } else {
        Flags { no_auto_psf: "--no-auto-nonstellar-radius", psf_size: "--nonstellar-radius", takes_radius: true, lunar_planetary: false }
    };
    cache.lock().unwrap().insert(executable.to_path_buf(), f);
    f
}

/// Look for `rc-astro` on the PATH and in the usual install locations.
pub fn find_executable() -> Option<PathBuf> {
    let name = if cfg!(windows) { "rc-astro.exe" } else { "rc-astro" };
    let mut candidates: Vec<PathBuf> = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).map(|d| d.join(name)).collect()).unwrap_or_default();
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from);
    if cfg!(target_os = "macos") {
        // GUI apps on macOS don't inherit the shell PATH.
        candidates.extend(["/usr/local/bin", "/opt/homebrew/bin", "/Applications/RC-Astro", "/Applications/rc-astro"].map(|d| Path::new(d).join(name)));
        if let Some(home) = &home {
            candidates.extend([home.join(".local/bin").join(name), home.join("Applications/RC-Astro").join(name)]);
        }
    }
    if cfg!(windows) {
        for var in ["ProgramFiles", "LOCALAPPDATA"] {
            if let Some(base) = std::env::var_os(var) {
                let base = PathBuf::from(base);
                candidates.extend([base.join("RC-Astro").join(name), base.join("Programs").join("RC-Astro").join(name)]);
            }
        }
    }
    candidates.into_iter().find(|p| p.is_file())
}

/// Ask the tool for its licence status, to check the configuration. Times out after 15 s.
pub fn check(executable: &Path) -> Result<String, String> {
    let mut child = Command::new(executable).args(["bxt", "--license"]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| format!("impossibile avviare: {e}"))?;
    let start = Instant::now();
    while child.try_wait().map_err(|e| e.to_string())?.is_none() {
        if start.elapsed() > Duration::from_secs(15) {
            child.kill().ok();
            return Err("nessuna risposta entro 15 s".into());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    let text = text.trim().to_string();
    if out.status.success() { Ok(if text.is_empty() { "OK".into() } else { text }) } else { Err(text) }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A fake `rc-astro` that copies input to output, to test the plumbing without the real tool.
    fn fake_tool(dir: &Path, script: &str) -> PathBuf {
        let path = dir.join("rc-astro");
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn scratch() -> PathBuf {
        let d = std::env::temp_dir().join(format!("ps-bxt-test-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn round_trip_through_external_tool() {
        let dir = scratch();
        // args: bxt <in> --output <out> ...
        let tool = fake_tool(&dir, "#!/bin/sh\ncp \"$2\" \"$4\"\n");
        let img = Image::new(4, 3, vec![(0..12).map(|v| v as f32 / 11.0).collect()], 16);
        let out = run(&img, &BxtParams::default(), &tool, &|| false).unwrap().unwrap();
        assert_eq!(out.channels(), img.channels());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn reports_tool_errors() {
        let dir = scratch();
        let tool = fake_tool(&dir, "#!/bin/sh\necho 'License not activated' >&2\nexit 3\n");
        let img = Image::new(2, 2, vec![vec![0.5; 4]], 16);
        let err = run(&img, &BxtParams::default(), &tool, &|| false).unwrap_err();
        assert!(err.contains("License not activated"), "{err}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn float_tiff_roundtrip_keeps_values_and_orientation() {
        let dir = scratch();
        let path = dir.join("rt.tif");
        let img = Image::new(3, 2, (0..3).map(|c| (0..6).map(|i| (i * 3 + c) as f32 / 17.0 - 0.1).collect()).collect(), 32);
        write_tiff_f32(&path, &img).unwrap();
        assert_eq!(read_tiff_f32(&path).unwrap().channels(), img.channels());
        std::fs::remove_dir_all(dir).ok();
    }
}
