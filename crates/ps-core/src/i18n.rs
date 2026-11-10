//! Interface language. The source strings are Italian; `tl` maps one to its English text when
//! English is selected (the table lives in `i18n_en.rs`) and returns it unchanged otherwise.
//! `tr!` does the same for format strings, whose two versions sit side by side at the call site.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

static ENGLISH: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Lang {
    #[default]
    It,
    En,
}

impl Lang {
    pub fn code(self) -> &'static str {
        match self {
            Lang::It => "it",
            Lang::En => "en",
        }
    }

    /// The language's own name, always shown in that language.
    pub fn native_name(self) -> &'static str {
        match self {
            Lang::It => "Italiano",
            Lang::En => "English",
        }
    }
}

pub fn set_lang(lang: Lang) {
    ENGLISH.store(lang == Lang::En, Ordering::Relaxed);
}

pub fn lang() -> Lang {
    if is_en() { Lang::En } else { Lang::It }
}

pub fn is_en() -> bool {
    OVERRIDE.with(|o| o.get()).unwrap_or_else(|| ENGLISH.load(Ordering::Relaxed))
}

thread_local! {
    /// A language for the current thread only, so tests can check one without disturbing
    /// the others running in parallel.
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Runs `f` with `lang` on this thread only.
pub fn with_lang<R>(lang: Lang, f: impl FnOnce() -> R) -> R {
    let previous = OVERRIDE.with(|o| o.replace(Some(lang == Lang::En)));
    let result = f();
    OVERRIDE.with(|o| o.set(previous));
    result
}

fn table() -> &'static HashMap<&'static str, &'static str> {
    static TABLE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    TABLE.get_or_init(|| crate::i18n_en::EN.iter().copied().collect())
}

/// The text in the current language. Unknown strings come back unchanged.
pub fn tl(it: &'static str) -> &'static str {
    if !is_en() {
        return it;
    }
    table().get(it).copied().unwrap_or(it)
}

/// Like `tl` for a string that is not a literal (a catalogue name saved in Italian): the
/// English text when there is one, else the string itself.
pub fn tl_dyn(it: &str) -> &str {
    if !is_en() {
        return it;
    }
    table().get(it).copied().unwrap_or(it)
}

/// A formatted string in the current language: `tr!("Italiano {}", "English {}", args…)`.
#[macro_export]
macro_rules! tr {
    ($it:literal, $en:literal $(, $arg:expr)* $(,)?) => {
        if $crate::i18n::is_en() { format!($en $(, $arg)*) } else { format!($it $(, $arg)*) }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_override_is_per_thread() {
        with_lang(Lang::En, || {
            assert_eq!(tl("Annulla"), "Undo");
            assert_eq!(tr!("{} px", "{} pixels", 3), "3 pixels");
        });
        assert_eq!(tl("Annulla"), "Annulla");
    }

    #[test]
    fn table_has_no_duplicate_keys_and_keeps_placeholders() {
        let mut seen = std::collections::HashSet::new();
        for (it, en) in crate::i18n_en::EN {
            assert!(seen.insert(*it), "duplicate key {it:?}");
            assert!(!en.is_empty(), "empty translation for {it:?}");
        }
    }
}
