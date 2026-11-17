//! What can be learnt about a capture without looking at the pixels: the subject, the time and
//! the filter from the file name, the equipment from the capture software's log.
//!
//! File names: FireCapture's `2026-08-29-2009_9-U-RGB-Moon…` (date, time in UT with tenths of
//! a minute, observer, filter, profile), names starting `yyyy-mm-dd-hhmm` in general, and the
//! older `IR0015 18-07-18 23-42-44` (filter, counter, yy-mm-dd hh-mm-ss). AutoStakkert adds
//! `_AS_P15_lapl6_ap92` (and a drizzle tag when used).
//!
//! Logs are `key=value` (FireCapture's settings file, SharpCap's `.CameraSettings.txt`, whose
//! first line is the camera in brackets) or `key: value` lines; unknown keys are ignored and
//! the known ones are looked up by several spellings, so small differences between versions
//! don't matter.

use crate::i18n::tl;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Body {
    Moon,
    Sun,
    Mercury,
    Venus,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
}

impl Body {
    pub const ALL: [Body; 9] = [Body::Moon, Body::Sun, Body::Mercury, Body::Venus, Body::Mars, Body::Jupiter, Body::Saturn, Body::Uranus, Body::Neptune];

    pub fn label(self) -> &'static str {
        match self {
            Body::Moon => tl("Luna"),
            Body::Sun => tl("Sole"),
            Body::Mercury => tl("Mercurio"),
            Body::Venus => tl("Venere"),
            Body::Mars => tl("Marte"),
            Body::Jupiter => tl("Giove"),
            Body::Saturn => tl("Saturno"),
            Body::Uranus => tl("Urano"),
            Body::Neptune => tl("Nettuno"),
        }
    }

    /// The body named by one word of a file name or log, in English or Italian.
    pub fn from_word(word: &str) -> Option<Body> {
        let w = word.to_ascii_lowercase();
        let names: [(&[&str], Body); 9] = [
            (&["moon", "luna", "lunar"], Body::Moon),
            (&["sun", "sole", "solar"], Body::Sun),
            (&["mer", "mercury", "mercurio"], Body::Mercury),
            (&["ven", "venus", "venere"], Body::Venus),
            (&["mars", "marte"], Body::Mars),
            (&["jup", "jupiter", "giove"], Body::Jupiter),
            (&["sat", "saturn", "saturno"], Body::Saturn),
            (&["ura", "uranus", "urano"], Body::Uranus),
            (&["nep", "neptune", "nettuno"], Body::Neptune),
        ];
        names.iter().find(|(n, _)| n.contains(&w.as_str())).map(|(_, b)| *b)
    }
}

/// A moment in UT.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Moment {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: f32,
}

impl Moment {
    /// Julian date.
    pub fn julian(&self) -> f64 {
        let (mut y, mut m) = (self.year as i64, self.month as i64);
        if m <= 2 {
            y -= 1;
            m += 12;
        }
        let a = y.div_euclid(100);
        let b = 2 - a + a.div_euclid(4);
        let day = self.day as f64 + (self.hour as f64 + self.minute as f64 / 60.0 + self.second as f64 / 3600.0) / 24.0;
        (365.25 * (y + 4716) as f64).floor() + (30.6001 * (m + 1) as f64).floor() + day + b as f64 - 1524.5
    }

    /// The moment of a Julian date (Meeus, chapter 7).
    pub fn from_julian(jd: f64) -> Moment {
        let jd = jd + 0.5;
        let z = jd.floor();
        let f = jd - z;
        let a = if z < 2_299_161.0 {
            z
        } else {
            let alpha = ((z - 1_867_216.25) / 36_524.25).floor();
            z + 1.0 + alpha - (alpha / 4.0).floor()
        };
        let b = a + 1524.0;
        let c = ((b - 122.1) / 365.25).floor();
        let d = (365.25 * c).floor();
        let e = ((b - d) / 30.6001).floor();
        let day = b - d - (30.6001 * e).floor();
        let month = if e < 14.0 { e - 1.0 } else { e - 13.0 };
        let year = if month > 2.0 { c - 4716.0 } else { c - 4715.0 };
        let secs = (f * 86_400.0).round();
        Moment { year: year as i32, month: month as u32, day: day as u32, hour: (secs / 3600.0).floor() as u32 % 24, minute: ((secs / 60.0).floor() as u32) % 60, second: (secs % 60.0) as f32 }
    }

    fn valid(&self) -> bool {
        (1..=12).contains(&self.month) && (1..=31).contains(&self.day) && self.hour < 24 && self.minute < 60 && self.second < 61.0
    }
}

/// What a capture's file name and log tell.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CaptureInfo {
    pub body: Option<Body>,
    pub time: Option<Moment>,
    /// The filter as written (`RGB`, `IR685`, `G`…).
    pub filter: Option<String>,
    pub camera: Option<String>,
    pub telescope: Option<String>,
    /// Effective focal length (with Barlow), mm.
    pub focal_mm: Option<f32>,
    /// Image scale, arcsec per pixel, as the capture software computed it.
    pub arcsec_per_px: Option<f32>,
    pub binning: Option<u32>,
    /// AutoStakkert's drizzle (1.5 = 150 %).
    pub drizzle: Option<f32>,
    /// The subject's apparent diameter as the capture software computed it, arcsec.
    pub diameter_arcsec: Option<f32>,
}

impl CaptureInfo {
    /// `other`'s values where this has none.
    pub fn or(mut self, other: CaptureInfo) -> CaptureInfo {
        macro_rules! fill {
            ($($f:ident),*) => {$( if self.$f.is_none() { self.$f = other.$f; } )*};
        }
        fill!(body, time, filter, camera, telescope, focal_mm, arcsec_per_px, binning, drizzle, diameter_arcsec);
        self
    }
}

/// Filter names as capture software writes them in file names.
const FILTERS: [&str; 18] = ["RGB", "L", "R", "G", "B", "IR", "UV", "CH4", "IR610", "IR642", "IR685", "IR742", "IR807", "IRCUT", "IR-CUT", "UVIR", "NIR", "HA"];

fn is_filter(word: &str) -> bool {
    FILTERS.iter().any(|f| f.eq_ignore_ascii_case(word)) || (word.len() > 2 && word[..2].eq_ignore_ascii_case("IR") && word[2..].bytes().all(|c| c.is_ascii_digit()))
}

/// What the file name (stem, without extension) says.
pub fn from_name(stem: &str) -> CaptureInfo {
    let mut info = CaptureInfo::default();
    let words: Vec<&str> = stem.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).collect();
    // Letters only: `giove0008`.
    info.body = body_in(stem);
    info.time = iso_time(stem).or_else(|| short_time(stem)).or_else(|| loose_time(stem));
    // FireCapture: date-time_tenth-observer-FILTER-object.
    let dashed: Vec<&str> = stem.split('-').collect();
    if iso_time(stem).is_some() && dashed.len() >= 6 && is_filter(dashed[dashed.len() - 2].split('_').next().unwrap_or("")) {
        info.filter = Some(dashed[dashed.len() - 2].to_string());
    }
    // Older names: the filter in front of a counter (`IR0015`, `2R0009`, `G0013`).
    if info.filter.is_none() {
        let first = words.first().copied().unwrap_or("").trim_start_matches(|c: char| c.is_ascii_digit());
        let letters: String = first.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        let digits = &first[letters.len()..];
        if !letters.is_empty() && digits.len() >= 3 && digits.bytes().all(|c| c.is_ascii_digit()) && is_filter(&letters) {
            info.filter = Some(letters.to_ascii_uppercase());
        }
    }
    // AutoStakkert drizzle: `Driz150`, `drizzle15`, `_d15` / `_D1.5`.
    for w in &words {
        let lower = w.to_ascii_lowercase();
        let digits = lower.strip_prefix("drizzle").or_else(|| lower.strip_prefix("driz"));
        if let Some(v) = digits.and_then(|d| d.parse::<f32>().ok()) {
            info.drizzle = Some(if v >= 10.0 { if v >= 100.0 { v / 100.0 } else { v / 10.0 } } else { v });
        }
    }
    info
}

/// The subject a name speaks of: a whole word (`Jup`, `giove`), else a long name inside a word
/// (`Jupiter2026`, `MyJupiterRun`).
pub fn body_in(text: &str) -> Option<Body> {
    let words: Vec<&str> = text.split(|c: char| !c.is_ascii_alphabetic()).filter(|w| !w.is_empty()).collect();
    if let Some(b) = words.iter().find_map(|w| Body::from_word(w)) {
        return Some(b);
    }
    const LONG: [(&str, Body); 14] = [
        ("jupiter", Body::Jupiter),
        ("giove", Body::Jupiter),
        ("saturn", Body::Saturn),
        ("mercury", Body::Mercury),
        ("mercurio", Body::Mercury),
        ("venus", Body::Venus),
        ("venere", Body::Venus),
        ("marte", Body::Mars),
        ("uranus", Body::Uranus),
        ("urano", Body::Uranus),
        ("neptune", Body::Neptune),
        ("nettuno", Body::Neptune),
        ("moon", Body::Moon),
        ("luna", Body::Moon),
    ];
    let lower = text.to_ascii_lowercase();
    LONG.iter().find(|(n, _)| lower.contains(n)).map(|(_, b)| *b)
}

/// `yyyy-mm-dd-hhmm[_t]` or `yyyy-mm-dd_hh-mm-ss` at the start of the name.
fn iso_time(s: &str) -> Option<Moment> {
    let b = s.as_bytes();
    let num = |r: std::ops::Range<usize>| -> Option<u32> { s.get(r).filter(|t| !t.is_empty() && t.bytes().all(|c| c.is_ascii_digit())).and_then(|t| t.parse().ok()) };
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (year, month, day) = (num(0..4)? as i32, num(5..7)?, num(8..10)?);
    let mut m = Moment { year, month, day, hour: 0, minute: 0, second: 0.0 };
    if matches!(b.get(10), Some(b'-' | b'_')) {
        if let (Some(h), Some(mi)) = (num(11..13), num(13..15)) {
            // FireCapture: hhmm then `_t`, tenths of a minute.
            (m.hour, m.minute) = (h, mi);
            if b.get(15) == Some(&b'_')
                && let Some(t) = num(16..17)
            {
                m.second = t as f32 * 6.0;
            }
        } else if let (Some(h), Some(mi), Some(sec)) = (num(11..13), num(14..16), num(17..19)) {
            (m.hour, m.minute, m.second) = (h, mi, sec as f32);
        }
    }
    m.valid().then_some(m)
}

/// `… yy-mm-dd hh-mm-ss` anywhere in the name.
fn short_time(s: &str) -> Option<Moment> {
    let words: Vec<&str> = s.split([' ', '_']).collect();
    for pair in words.windows(2) {
        let date: Vec<&str> = pair[0].split('-').collect();
        let time: Vec<&str> = pair[1].split('-').collect();
        let all = |v: &[&str]| v.len() == 3 && v.iter().all(|t| t.len() == 2 && t.bytes().all(|c| c.is_ascii_digit()));
        if all(&date) && all(&time) {
            let n = |t: &str| t.parse::<u32>().unwrap_or(99);
            let m = Moment { year: 2000 + n(date[0]) as i32, month: n(date[1]), day: n(date[2]), hour: n(time[0]), minute: n(time[1]), second: n(time[2]) as f32 };
            if m.valid() {
                return Some(m);
            }
        }
    }
    None
}


/// A date written some other way, anywhere in the name: `20260829_201530`, `2026.08.29 20.15`,
/// `29-08-2026_2015`, `2026-08-29T20:15:30`. The time is optional (midnight when missing).
fn loose_time(s: &str) -> Option<Moment> {
    let b = s.as_bytes();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }
    let text = |k: usize| &s[runs[k].0..runs[k].1];
    let num = |k: usize| text(k).parse::<u32>().ok();
    let year_ok = |y: u32| (1990..=2100).contains(&y);
    // The one character between run `k` and the next.
    let sep = |k: usize| -> Option<u8> { (k + 1 < runs.len() && runs[k + 1].0 - runs[k].1 == 1).then(|| b[runs[k].1]) };
    let date_sep = |c: u8| matches!(c, b'-' | b'_' | b'.' | b'/' | b' ');
    // hhmm, hhmmss or hh mm [ss] from run `k` on.
    let time_at = |k: usize| -> Option<(u32, u32, f32)> {
        if k >= runs.len() {
            return None;
        }
        let t = match text(k).len() {
            6 => Some((num(k)? / 10_000, num(k)? / 100 % 100, (num(k)? % 100) as f32)),
            4 => Some((num(k)? / 100, num(k)? % 100, 0.0)),
            2 if k + 1 < runs.len() && text(k + 1).len() == 2 && sep(k) == sep(k).filter(|c| matches!(c, b'-' | b'_' | b'.' | b':')) && sep(k).is_some() => {
                let sec = if k + 2 < runs.len() && text(k + 2).len() == 2 && sep(k + 1) == sep(k) { num(k + 2)? as f32 } else { 0.0 };
                Some((num(k)?, num(k + 1)?, sec))
            }
            _ => None,
        }?;
        (t.0 < 24 && t.1 < 60 && t.2 < 61.0).then_some(t)
    };
    for k in 0..runs.len() {
        let len = text(k).len();
        // (year, month, day, index of the run after the date)
        let date: Option<(u32, u32, u32, usize)> = if len == 8 {
            let v = num(k)?;
            let (a, b2, c) = (v / 10_000, v / 100 % 100, v % 100);
            if year_ok(a) { Some((a, b2, c, k + 1)) } else if year_ok(c) { Some((c, b2, a, k + 1)) } else { None }
        } else if len == 4 && k + 2 < runs.len() && text(k + 1).len() == 2 && text(k + 2).len() == 2 && sep(k).is_some_and(date_sep) && sep(k) == sep(k + 1) {
            Some((num(k)?, num(k + 1)?, num(k + 2)?, k + 3)).filter(|d| year_ok(d.0))
        } else if len == 2 && k + 2 < runs.len() && text(k + 1).len() == 2 && text(k + 2).len() == 4 && sep(k).is_some_and(date_sep) && sep(k) == sep(k + 1) {
            Some((num(k + 2)?, num(k + 1)?, num(k)?, k + 3)).filter(|d| year_ok(d.0))
        } else {
            None
        };
        let Some((year, month, day, next)) = date else { continue };
        let mut m = Moment { year: year as i32, month, day, hour: 0, minute: 0, second: 0.0 };
        if !m.valid() {
            continue;
        }
        // The time follows after one separator (`_`, `-`, space, `T`).
        if next > 0 && next < runs.len() && runs[next].0 - runs[next - 1].1 == 1 && let Some((h, mi, sec)) = time_at(next) {
            (m.hour, m.minute, m.second) = (h, mi, sec);
        }
        return Some(m);
    }
    None
}

/// What a capture file's own header says: a SER file's header (camera, telescope, the UT of the
/// first frame) or a FITS header (`DATE-OBS`, `OBJECT`, `INSTRUME`, `TELESCOP`, `FILTER`).
pub fn from_header(path: &std::path::Path) -> Option<CaptureInfo> {
    use std::io::Read;
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    let limit = match ext.as_str() {
        "ser" => 178,
        "fit" | "fits" | "fts" => 28_800,
        _ => return None,
    };
    let mut bytes = Vec::new();
    std::fs::File::open(path).ok()?.take(limit).read_to_end(&mut bytes).ok()?;
    if ext == "ser" { from_ser_header(&bytes) } else { from_fits_header(&bytes) }
}

fn from_ser_header(h: &[u8]) -> Option<CaptureInfo> {
    if h.len() < 178 || !h.starts_with(b"LUCAM-RECORDER") {
        return None;
    }
    let text = |r: std::ops::Range<usize>| {
        let t = String::from_utf8_lossy(&h[r]).trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string();
        Some(t).filter(|t| !t.is_empty())
    };
    let mut info = CaptureInfo { camera: text(82..122), telescope: text(122..162), ..Default::default() };
    // .NET ticks (100 ns since 0001-01-01) in UT; a header written without a clock holds zero.
    let ticks = u64::from_le_bytes(h[170..178].try_into().ok()?);
    let unix = (ticks / 10_000_000) as i64 - 62_135_596_800;
    if unix > 631_152_000 {
        let jd = 2_440_587.5 + unix as f64 / 86_400.0;
        let mut m = Moment::from_julian(jd);
        m.second = (unix.rem_euclid(60)) as f32;
        info.time = Some(m).filter(Moment::valid);
    }
    Some(info)
}

fn from_fits_header(bytes: &[u8]) -> Option<CaptureInfo> {
    if !bytes.starts_with(b"SIMPLE") {
        return None;
    }
    let mut info = CaptureInfo::default();
    for card in bytes.chunks(80) {
        let Ok(text) = std::str::from_utf8(card) else { break };
        if text.starts_with("END") && text[3..].trim().is_empty() {
            break;
        }
        if text.get(8..10) != Some("= ") {
            continue;
        }
        let key = text[..8].trim();
        let value = text[10..].split('/').next().unwrap_or("").trim().trim_matches('\'').trim().to_string();
        if value.is_empty() {
            continue;
        }
        match key {
            "DATE-OBS" if info.time.is_none() => info.time = iso_datetime(&value),
            "DATE-AVG" => info.time = iso_datetime(&value).or(info.time),
            "OBJECT" => info.body = body_in(&value),
            "INSTRUME" | "CAMERA" => info.camera = Some(value),
            "TELESCOP" => info.telescope = Some(value),
            "FILTER" => info.filter = Some(value),
            _ => {}
        }
    }
    Some(info)
}

/// `2026-08-29T20:15:30.5` (a FITS or ISO 8601 stamp, UT).
fn iso_datetime(v: &str) -> Option<Moment> {
    let (date, time) = v.split_once(['T', ' ']).unwrap_or((v, ""));
    let d: Vec<u32> = date.split('-').filter_map(|p| p.parse().ok()).collect();
    if d.len() != 3 {
        return None;
    }
    let t: Vec<&str> = time.trim_end_matches('Z').split(':').collect();
    let part = |k: usize| t.get(k).and_then(|p| p.parse::<f32>().ok()).unwrap_or(0.0);
    let m = Moment { year: d[0] as i32, month: d[1], day: d[2], hour: part(0) as u32, minute: part(1) as u32, second: part(2) };
    m.valid().then_some(m)
}

/// Capture files beside `image` whose own header holds the capture's data: the image itself when
/// it is a FITS, and a SER video whose name the image's name starts with.
pub fn headers_beside(image: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut found = vec![image.to_path_buf()];
    let (Some(dir), Some(stem)) = (image.parent(), image.file_stem().and_then(|s| s.to_str())) else { return found };
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut sers: Vec<(usize, std::path::PathBuf)> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("ser")))
            .filter_map(|p| {
                let base = p.file_stem()?.to_str()?.to_string();
                (base.len() >= 8 && stem.starts_with(&base)).then_some((base.len(), p))
            })
            .collect();
        sers.sort_by_key(|f| std::cmp::Reverse(f.0));
        found.extend(sers.into_iter().map(|(_, p)| p));
    }
    found
}

/// What a capture log says.
pub fn from_log(text: &str) -> CaptureInfo {
    let mut info = CaptureInfo::default();
    let mut pairs: Vec<(String, String)> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        // SharpCap: the first line is the camera, in brackets.
        if i == 0 && line.starts_with('[') && line.ends_with(']') {
            info.camera = Some(line[1..line.len() - 1].trim().to_string());
            continue;
        }
        let Some((k, v)) = line.split_once('=').or_else(|| line.split_once(':')) else { continue };
        let (k, v) = (k.trim().to_ascii_lowercase(), v.trim().to_string());
        if !k.is_empty() && !v.is_empty() {
            pairs.push((k, v));
        }
    }
    let get = |keys: &[&str]| pairs.iter().find(|(k, _)| keys.contains(&k.as_str())).map(|(_, v)| v.clone());
    let number = |v: &str| -> Option<f32> {
        let t: String = v.trim().chars().skip_while(|c| !c.is_ascii_digit() && *c != '.').take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',').collect();
        t.replace(',', ".").parse().ok()
    };
    info.camera = info.camera.or_else(|| get(&["camera", "camera model", "cam", "device"]));
    info.telescope = get(&["telescope", "scope", "telescopio", "optics", "ota"]);
    info.filter = get(&["filter", "filtro", "filter name"]);
    info.focal_mm = get(&["focallength", "focal length", "focal_length", "focale", "focal"]).and_then(|v| number(&v)).filter(|&f| f > 10.0);
    info.arcsec_per_px = get(&["resolution", "image scale", "pixel scale", "scale"]).and_then(|v| number(&v)).filter(|&s| s > 0.0 && s < 100.0);
    info.diameter_arcsec = get(&["diameter", "diametro"]).and_then(|v| number(&v)).filter(|&d| d > 0.0);
    info.binning = get(&["binning", "bin"]).and_then(|v| if v.eq_ignore_ascii_case("no") || v.eq_ignore_ascii_case("off") { Some(1.0) } else { number(&v) }).map(|b| b.max(1.0) as u32);
    info.body = get(&["profile", "object", "target", "oggetto"]).and_then(|v| body_in(&v));
    info.drizzle = pairs.iter().find(|(k, _)| k.contains("drizzle")).and_then(|(_, v)| {
        if v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("no") {
            return Some(1.0);
        }
        number(v).map(|d| if d >= 10.0 { d / 100.0 } else { d })
    });
    info.time = log_time(&get, &pairs);
    info
}

/// FireCapture: `Date=290826` with `Date_format=ddMMyy` and `Mid(UT)=200955.123`; otherwise an
/// ISO date and a `hh:mm:ss` time under common keys.
fn log_time(get: &dyn Fn(&[&str]) -> Option<String>, pairs: &[(String, String)]) -> Option<Moment> {
    let date = get(&["date", "data", "date (ut)", "date(ut)"])?;
    let format = get(&["date_format", "date format"]).unwrap_or_default().to_ascii_lowercase();
    let digits: String = date.chars().filter(|c| c.is_ascii_digit()).collect();
    let (year, month, day) = if date.contains('-') && digits.len() == 8 && date.find('-') == Some(4) {
        (digits[0..4].parse().ok()?, digits[4..6].parse().ok()?, digits[6..8].parse().ok()?)
    } else if digits.len() == 6 {
        let (a, b, c): (u32, u32, u32) = (digits[0..2].parse().ok()?, digits[2..4].parse().ok()?, digits[4..6].parse().ok()?);
        match format.as_str() {
            "yymmdd" => (2000 + a as i32, b, c),
            "mmddyy" => (2000 + c as i32, a, b),
            _ => (2000 + c as i32, b, a),
        }
    } else if digits.len() == 8 {
        // ddMMyyyy
        (digits[4..8].parse().ok()?, digits[2..4].parse().ok()?, digits[0..2].parse().ok()?)
    } else {
        return None;
    };
    let time = ["mid(ut)", "start(ut)", "mid", "start", "time", "time (ut)", "ora"].iter().find_map(|k| pairs.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone()));
    let mut m = Moment { year, month, day, hour: 0, minute: 0, second: 0.0 };
    if let Some(t) = time {
        let clean: String = t.chars().filter(|c| c.is_ascii_digit() || *c == '.' || *c == ':').collect();
        let parts: Vec<&str> = clean.split(':').collect();
        let (h, mi, s) = if parts.len() >= 2 {
            (parts[0].parse().ok()?, parts[1].parse().ok()?, parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0))
        } else {
            let whole = clean.split('.').next().unwrap_or("");
            if whole.len() < 4 {
                return None;
            }
            let frac: f32 = clean.get(whole.len()..).and_then(|f| format!("0{f}").parse().ok()).unwrap_or(0.0);
            (whole[0..2].parse().ok()?, whole[2..4].parse().ok()?, whole.get(4..6).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0) + frac)
        };
        (m.hour, m.minute, m.second) = (h, mi, s);
    }
    m.valid().then_some(m)
}

/// Logs that may belong to the image `image` (a path): next to it, a `.txt` (or AutoStakkert
/// `.as3`/`.as4`, SharpCap `.CameraSettings.txt`) whose name the image's name starts with —
/// the stack keeps the capture's name and adds its own suffix.
pub fn logs_beside(image: &std::path::Path) -> Vec<std::path::PathBuf> {
    let (Some(dir), Some(stem)) = (image.parent(), image.file_stem().and_then(|s| s.to_str())) else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut found: Vec<(usize, std::path::PathBuf)> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| ["txt", "as3", "as4", "log"].contains(&e.to_ascii_lowercase().as_str())))
        .filter_map(|p| {
            let name = p.file_name()?.to_str()?.to_string();
            // Strip every extension: `capture.ser.txt`, `capture.CameraSettings.txt`.
            let base = name.split('.').next()?.to_string();
            (base.len() >= 8 && stem.starts_with(&base)).then_some((base.len(), p))
        })
        .collect();
    // The closest match first.
    found.sort_by_key(|f| std::cmp::Reverse(f.0));
    found.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn firecapture_names() {
        let i = from_name("2026-08-29-2009_9-U-RGB-Moon_AS_P20_lapl4_ap116");
        assert_eq!(i.body, Some(Body::Moon));
        assert_eq!(i.filter.as_deref(), Some("RGB"));
        let t = i.time.unwrap();
        assert_eq!((t.year, t.month, t.day, t.hour, t.minute), (2026, 8, 29, 20, 9));
        assert!((t.second - 54.0).abs() < 1e-3);
        let i = from_name("2026-09-19-2050_3-U-RGB-Sat_Exposure=31.3ms_Gain=406_AS_P15_lapl6_ap92");
        assert_eq!(i.body, Some(Body::Saturn));
        assert_eq!(from_name("2026-07-27-1939_8-U-G-Moon_AS_P25_lapl3_ap908").filter.as_deref(), Some("G"));
    }

    #[test]
    fn older_names() {
        let i = from_name("IR0015 18-07-18 23-42-44");
        assert_eq!(i.filter.as_deref(), Some("IR"));
        let t = i.time.unwrap();
        assert_eq!((t.year, t.month, t.day, t.hour, t.minute, t.second as u32), (2018, 7, 18, 23, 42, 44));
        assert_eq!(from_name("2R0009 19-07-05 22-27-42_P23").filter.as_deref(), Some("R"));
        assert_eq!(from_name("giove0008 16-03-20 19-12-37_AS_p40_g4_ap16").body, Some(Body::Jupiter));
        assert_eq!(from_name("prova_c80013 18-04-19 20-41-00_AS_P15").filter, None);
    }

    #[test]
    fn firecapture_log() {
        let log = "FireCapture v2.7.14  Settings\n------------------------------------\nCamera=ZWO ASI462MC\nFilter=IR685\nProfile=Jupiter\nDiameter=45.12\"\nFocalLength=4200mm\nResolution=0.14\"\nDate=290826\nStart(UT)=200955.123\nMid(UT)=201025.123\nDate_format=ddMMyy\nBinning=no\nGain=400 (66%)\n";
        let i = from_log(log);
        assert_eq!(i.camera.as_deref(), Some("ZWO ASI462MC"));
        assert_eq!(i.filter.as_deref(), Some("IR685"));
        assert_eq!(i.body, Some(Body::Jupiter));
        assert_eq!(i.focal_mm, Some(4200.0));
        assert_eq!(i.arcsec_per_px, Some(0.14));
        assert_eq!(i.diameter_arcsec, Some(45.12));
        assert_eq!(i.binning, Some(1));
        let t = i.time.unwrap();
        assert_eq!((t.year, t.month, t.day, t.hour, t.minute), (2026, 8, 29, 20, 10));
        assert!((t.second - 25.123).abs() < 1e-3);
    }

    #[test]
    fn sharpcap_and_autostakkert() {
        let i = from_log("[ZWO ASI224MC]\nBinning=2\nGain=300\n");
        assert_eq!((i.camera.as_deref(), i.binning), (Some("ZWO ASI224MC"), Some(2)));
        assert_eq!(from_log("Drizzle: 1.5x\nFrames: 500").drizzle, Some(1.5));
        assert_eq!(from_log("Drizzle=150%").drizzle, Some(1.5));
        assert_eq!(from_name("capture_AS_P20_Driz150").drizzle, Some(1.5));
    }


    #[test]
    fn dates_in_other_formats() {
        let t = |s: &str| from_name(s).time.map(|t| (t.year, t.month, t.day, t.hour, t.minute, t.second as u32));
        assert_eq!(t("Jupiter_20260829_201530"), Some((2026, 8, 29, 20, 15, 30)));
        assert_eq!(t("giove 2026.08.29 20.15.30 stack"), Some((2026, 8, 29, 20, 15, 30)));
        assert_eq!(t("Saturno_29-08-2026_2015"), Some((2026, 8, 29, 20, 15, 0)));
        assert_eq!(t("capture-2026-08-29T20:15:30"), Some((2026, 8, 29, 20, 15, 30)));
        assert_eq!(t("Moon_20260829"), Some((2026, 8, 29, 0, 0, 0)));
        assert_eq!(t("Jupiter_AS_P20_lapl4_ap116"), None);
        assert_eq!(from_name("MyJupiterRun").body, Some(Body::Jupiter));
        assert_eq!(from_name("Giove_20260829_201530").body, Some(Body::Jupiter));
    }

    #[test]
    fn headers_of_ser_and_fits() {
        let mut ser = vec![0u8; 178];
        ser[..14].copy_from_slice(b"LUCAM-RECORDER");
        ser[82..92].copy_from_slice(b"ASI462MC  ");
        // 2026-08-29 20:15:30 UT in .NET ticks.
        let unix: i64 = 1_788_034_530;
        let ticks = ((unix + 62_135_596_800) as u64) * 10_000_000;
        ser[170..178].copy_from_slice(&ticks.to_le_bytes());
        let i = from_ser_header(&ser).unwrap();
        assert_eq!(i.camera.as_deref(), Some("ASI462MC"));
        let t = i.time.unwrap();
        assert_eq!((t.year, t.month, t.day, t.hour, t.minute, t.second as u32), (2026, 8, 29, 20, 15, 30));
        let mut fits = String::new();
        for card in ["SIMPLE  =                    T", "DATE-OBS= '2026-08-29T20:15:30.250' / start", "OBJECT  = 'Jupiter '", "INSTRUME= 'ZWO ASI224MC'", "END"] {
            fits.push_str(&format!("{card:<80}"));
        }
        let i = from_fits_header(fits.as_bytes()).unwrap();
        assert_eq!(i.body, Some(Body::Jupiter));
        assert_eq!(i.camera.as_deref(), Some("ZWO ASI224MC"));
        assert_eq!(i.time.map(|t| (t.day, t.hour, t.minute)), Some((29, 20, 15)));
    }

    #[test]
    fn julian_date() {
        // J2000.0
        let m = Moment { year: 2000, month: 1, day: 1, hour: 12, minute: 0, second: 0.0 };
        assert!((m.julian() - 2_451_545.0).abs() < 1e-6);
    }
}
