//! Names of exported files: a base name the user can edit, plus keywords saying what the image
//! is and what was done to it.

use ps_core::i18n::tl;
use ps_core::analysis::ImageKind;
use ps_core::pipeline::Stage;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Words that can be added after the base name, in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NameKeyword {
    /// The subject: from the capture software's file name (Moon, Sat, Jup…), else the type.
    Object,
    /// Capture date: from the file name (`2026-09-19-2050…`), else the file's date.
    Date,
    /// Capture time (hhmm), the same way.
    Time,
    /// The modules that change the image (`deconv-wav-dering`).
    Modules,
    /// The user's suffix (`_ps`).
    Suffix,
}

impl NameKeyword {
    pub const ALL: [NameKeyword; 5] = [NameKeyword::Object, NameKeyword::Date, NameKeyword::Time, NameKeyword::Modules, NameKeyword::Suffix];

    pub fn label(self) -> &'static str {
        match self {
            NameKeyword::Object => tl("Oggetto"),
            NameKeyword::Date => tl("Data"),
            NameKeyword::Time => tl("Ora"),
            NameKeyword::Modules => tl("Moduli usati"),
            NameKeyword::Suffix => tl("Suffisso"),
        }
    }
}

/// Short tag of a module in file names.
pub fn module_tag(stage: Stage) -> &'static str {
    match stage {
        Stage::Align => "rgb",
        Stage::Bxt => "bxt",
        Stage::Deconv => "deconv",
        Stage::Denoise => "nr",
        Stage::Wavelet => "wav",
        Stage::Smart => tl("nitidezza"),
        Stage::Contrast => "usm",
        Stage::Dering => "dering",
        Stage::Tone => tl("livelli"),
        Stage::Curves => tl("curve"),
        Stage::Color => tl("colore"),
        Stage::Layers => tl("maschere"),
        Stage::Mineral => "mineral",
        Stage::Rings => tl("anelli"),
        Stage::Moons => tl("lune"),
        Stage::SunColor => tl("colore-sole"),
    }
}

/// What a name is built from.
pub struct NameSource<'a> {
    /// The base name (by default the original's).
    pub base: &'a str,
    /// The original file (for the capture date and the subject).
    pub original: &'a Path,
    pub kind: ImageKind,
    /// Tags of the modules that change the image, in processing order.
    pub modules: Vec<&'static str>,
    pub suffix: &'a str,
}

/// The file name without extension.
pub fn stem(source: &NameSource, keywords: &[NameKeyword]) -> String {
    let file = source.original.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let capture = capture_time(file).or_else(|| modified_time(source.original));
    let mut name = sanitize(source.base.trim());
    if name.is_empty() {
        name = tl("immagine").into();
    }
    for k in NameKeyword::ALL.into_iter().filter(|k| keywords.contains(k)) {
        let word = match k {
            NameKeyword::Object => Some(object(file, source.kind).to_string()),
            NameKeyword::Date => capture.map(|(y, mo, d, ..)| format!("{y:04}-{mo:02}-{d:02}")),
            NameKeyword::Time => capture.map(|(.., h, mi)| format!("{h:02}{mi:02}")),
            NameKeyword::Modules => (!source.modules.is_empty()).then(|| source.modules.join("-")),
            NameKeyword::Suffix => {
                // The suffix carries its own separator, if any.
                name.push_str(&sanitize(source.suffix));
                None
            }
        };
        // Not repeated when the base already says it (`2026-09-19-2050_3-…-Sat…`).
        if let Some(word) = word.filter(|w| !name.contains(w.as_str())) {
            name.push('_');
            name.push_str(&word);
        }
    }
    name
}

/// The subject named in the capture software's file name, else the image type.
fn object(file: &str, kind: ImageKind) -> &'static str {
    let lower = file.to_ascii_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_ascii_alphabetic()).filter(|w| !w.is_empty()).collect();
    let named = [
        (&["moon", "luna"][..], tl("Luna")),
        (&["sat", "saturn", "saturno"], tl("Saturno")),
        (&["jup", "jupiter", "giove"], tl("Giove")),
        (&["mars", "marte"], tl("Marte")),
        (&["sun", "sole"], tl("Sole")),
        (&["ven", "venus", "venere"], tl("Venere")),
        (&["mer", "mercury", "mercurio"], tl("Mercurio")),
        (&["ura", "uranus", "urano"], tl("Urano")),
        (&["nep", "neptune", "nettuno"], tl("Nettuno")),
    ];
    named.iter().find(|(names, _)| words.iter().any(|w| names.contains(w))).map_or(
        match kind {
            ImageKind::Saturn => tl("Saturno"),
            ImageKind::Planet => tl("Pianeta"),
            ImageKind::FullDisk => tl("Disco"),
            ImageKind::Surface => tl("Superficie"),
        },
        |(_, name)| name,
    )
}

/// (year, month, day, hour, minute) from a name starting `yyyy-mm-dd-hhmm` (SharpCap,
/// FireCapture, AutoStakkert).
fn capture_time(file: &str) -> Option<(u32, u32, u32, u32, u32)> {
    let b = file.as_bytes();
    let num = |r: std::ops::Range<usize>| -> Option<u32> { file.get(r).filter(|s| s.bytes().all(|c| c.is_ascii_digit())).and_then(|s| s.parse().ok()) };
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let (h, mi) = match (b.get(10), num(11..13), num(13..15)) {
        (Some(b'-' | b'_'), Some(h), Some(mi)) if h < 24 && mi < 60 => (h, mi),
        _ => (0, 0),
    };
    Some((y, mo, d, h, mi))
}

/// The file's modification time (UTC).
fn modified_time(path: &Path) -> Option<(u32, u32, u32, u32, u32)> {
    let secs = std::fs::metadata(path).ok()?.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs() as i64;
    let (days, rest) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (H. Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    Some((y as u32, m as u32, d as u32, (rest / 3600) as u32, (rest % 3600 / 60) as u32))
}

/// Without characters that file systems refuse.
fn sanitize(s: &str) -> String {
    s.chars().map(|c| if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control() { '-' } else { c }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source<'a>(base: &'a str, path: &'a Path) -> NameSource<'a> {
        NameSource { base, original: path, kind: ImageKind::Planet, modules: vec!["deconv", "wav"], suffix: "_ps" }
    }

    #[test]
    fn keywords_follow_the_base_name() {
        let path = Path::new("/x/2026-09-19-2050_3-U-RGB-Sat_AS_P15.tif");
        let s = source("Saturno buono", path);
        assert_eq!(stem(&s, &[]), "Saturno buono");
        assert_eq!(stem(&s, &[NameKeyword::Suffix]), "Saturno buono_ps");
        assert_eq!(stem(&s, &[NameKeyword::Modules, NameKeyword::Date, NameKeyword::Time]), "Saturno buono_2026-09-19_2050_deconv-wav");
        // "Saturno" is already in the base name.
        assert_eq!(stem(&s, &[NameKeyword::Object]), "Saturno buono");
        assert_eq!(stem(&source("a", Path::new("/x/Jup_001.tif")), &[NameKeyword::Object]), "a_Giove");
    }

    #[test]
    fn the_original_name_is_not_repeated() {
        let path = Path::new("/x/2026-09-19-2050_3-U-RGB-Sat.tif");
        let s = source("2026-09-19-2050_3-U-RGB-Sat", path);
        assert_eq!(stem(&s, &[NameKeyword::Date, NameKeyword::Suffix]), "2026-09-19-2050_3-U-RGB-Sat_ps");
    }

    #[test]
    fn forbidden_characters_are_replaced() {
        assert_eq!(stem(&source("a/b:c", Path::new("/x/y.tif")), &[]), "a-b-c");
    }

    #[test]
    fn civil_dates() {
        assert_eq!(capture_time("2026-08-29-2009_9-U-RGB-Moon"), Some((2026, 8, 29, 20, 9)));
        assert_eq!(capture_time("Moon_2026"), None);
    }
}
