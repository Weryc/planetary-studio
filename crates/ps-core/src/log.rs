//! The application's log: one text file the user can send when something goes wrong.
//!
//! Each line is `date time level [area] message`. What is written: the start (version, system),
//! each image opened and what the analysis found, the settings of each finished render and the
//! stages' warnings, what the learned networks decided (curves, noise estimates, Saturn's zones),
//! errors, and panics with where they happened.
//!
//! Where (see [`default_path`]): `~/Library/Logs/PlanetaryStudio/` on macOS,
//! `%LOCALAPPDATA%\PlanetaryStudio\` on Windows, `~/.local/state/planetary-studio/` on Linux:
//! the folder of the app's crash reports. The file is kept under [`MAX_BYTES`]: past it the previous one becomes `.1.log`.
//!
//! Until [`init`] is called (tests, `ps-train`) nothing is written anywhere.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Past this size the log is rotated (one previous file kept).
pub const MAX_BYTES: u64 = 5 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

impl Level {
    fn label(self) -> &'static str {
        match self {
            Level::Info => "INFO ",
            Level::Warn => "WARN ",
            Level::Error => "ERROR",
        }
    }
}

struct Sink {
    path: PathBuf,
    file: File,
}

static SINK: Mutex<Option<Sink>> = Mutex::new(None);

/// The log's usual place on this system.
pub fn default_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let dir = if cfg!(target_os = "macos") {
        home?.join("Library/Logs/PlanetaryStudio")
    } else if cfg!(target_os = "windows") {
        PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("PlanetaryStudio")
    } else {
        std::env::var_os("XDG_STATE_HOME").map(PathBuf::from).or_else(|| home.map(|h| h.join(".local/state")))?.join("planetary-studio")
    };
    Some(dir.join("planetary-studio.log"))
}

/// The file the log is being written to, once [`init`] succeeded.
pub fn current_path() -> Option<PathBuf> {
    SINK.lock().ok()?.as_ref().map(|s| s.path.clone())
}

/// Start writing to `path` (its folder is created), rotating it if too large, and log panics.
pub fn init(path: &Path) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    rotate_if_large(path);
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    if let Ok(mut s) = SINK.lock() {
        *s = Some(Sink { path: path.to_path_buf(), file });
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let place = info.location().map_or(String::new(), |l| format!(" ({}:{})", l.file(), l.line()));
        let what = info.payload().downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| info.payload().downcast_ref::<String>().cloned()).unwrap_or_default();
        let thread = std::thread::current().name().unwrap_or("?").to_string();
        write(Level::Error, "panic", &format!("thread {thread}: {what}{place}"));
        previous(info);
    }));
    Ok(())
}

fn rotate_if_large(path: &Path) {
    if std::fs::metadata(path).is_ok_and(|m| m.len() > MAX_BYTES) {
        let old = path.with_extension("1.log");
        let _ = std::fs::remove_file(&old);
        let _ = std::fs::rename(path, old);
    }
}

/// Write one line (nothing before [`init`]). Never fails: a log that can't be written is skipped.
pub fn write(level: Level, area: &str, message: &str) {
    let Ok(mut guard) = SINK.lock() else { return };
    let Some(sink) = guard.as_mut() else { return };
    let line = format!("{} {} [{area}] {}\n", timestamp(), level.label(), message.replace('\n', " | "));
    let _ = sink.file.write_all(line.as_bytes());
    // Rotate in the middle of a long session too.
    if sink.file.metadata().is_ok_and(|m| m.len() > MAX_BYTES) {
        let path = sink.path.clone();
        rotate_if_large(&path);
        if let Ok(f) = OpenOptions::new().create(true).append(true).open(&path) {
            sink.file = f;
        }
    }
}

pub fn info(area: &str, message: &str) {
    write(Level::Info, area, message);
}

pub fn warn(area: &str, message: &str) {
    write(Level::Warn, area, message);
}

pub fn error(area: &str, message: &str) {
    write(Level::Error, area, message);
}

/// Local-agnostic UTC time, `YYYY-MM-DD HH:MM:SS.mmm`.
fn timestamp() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs() as i64;
    let (days, rest) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}.{:03}Z", rest / 3600, rest / 60 % 60, rest % 60, now.subsec_millis())
}
