//! Crash report: a panic anywhere (the UI, the processing thread, a background job) is written
//! to a log file the user can attach to a bug report. Without it a release build, which has no
//! console, would just stop or freeze silently.

use std::path::PathBuf;

/// Folder of the log files, by the platform's convention.
pub fn log_dir() -> PathBuf {
    let home = || std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    if cfg!(target_os = "macos") {
        home().join("Library/Logs/PlanetaryStudio")
    } else if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir).join("PlanetaryStudio")
    } else {
        std::env::var_os("XDG_STATE_HOME").map(PathBuf::from).unwrap_or_else(|| home().join(".local/state")).join("planetary-studio")
    }
}

/// Install the panic hook (keeping the default one, which prints to the console).
pub fn install() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default(info);
        let thread = std::thread::current();
        let report = format!(
            "Planetary Studio {} — {} {}\nthread: {}\n{info}\n\n{}\n",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
            thread.name().unwrap_or("unnamed"),
            std::backtrace::Backtrace::force_capture()
        );
        let dir = log_dir();
        if std::fs::create_dir_all(&dir).is_ok() {
            use std::io::Write;
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("crash.log")) {
                let _ = writeln!(file, "{report}\n{}", "-".repeat(60));
            }
        }
    }));
}
