//! Finestre di sistema per aprire e salvare file, al riparo dalla chiusura dell'app.
//!
//! I dialoghi di `rfd` sono modali e sincroni: girano dentro il ciclo di eventi di winit, mentre
//! winit sta già gestendo un evento. Se in quel momento l'utente chiude l'app (Cmd+Q, menu,
//! Dock), macOS chiama `applicationWillTerminate:` sul delegato di winit, che prova a gestire un
//! secondo evento dentro il primo e va in panico dentro una callback Objective-C: l'app abortisce
//! (rapporto di crash con `panic_cannot_unwind` in `app_will_terminate`).
//!
//! Qui ogni dialogo gira con un delegato provvisorio che rifiuta la chiusura, chiude il dialogo
//! come annullato e la segna; al frame successivo l'app si chiude normalmente
//! ([`take_quit_request`]), salvando le preferenze come sempre.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

static QUIT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// `true` (una volta sola) se l'utente ha chiesto di chiudere l'app mentre un dialogo era aperto.
pub fn take_quit_request() -> bool {
    QUIT_REQUESTED.swap(false, Ordering::SeqCst)
}

/// Same builder as [`rfd::FileDialog`], with every dialog run under the quit guard.
#[derive(Default)]
pub struct FileDialog(rfd::FileDialog);

impl FileDialog {
    pub fn new() -> Self {
        Self(rfd::FileDialog::new())
    }
    pub fn set_title(self, title: impl Into<String>) -> Self {
        Self(self.0.set_title(title))
    }
    pub fn add_filter(self, name: impl Into<String>, extensions: &[impl ToString]) -> Self {
        Self(self.0.add_filter(name, extensions))
    }
    pub fn set_file_name(self, name: impl Into<String>) -> Self {
        Self(self.0.set_file_name(name))
    }
    pub fn set_directory<P: AsRef<Path>>(self, path: P) -> Self {
        Self(self.0.set_directory(path))
    }
    pub fn pick_file(self) -> Option<PathBuf> {
        guarded(|| self.0.pick_file())
    }
    pub fn pick_files(self) -> Option<Vec<PathBuf>> {
        guarded(|| self.0.pick_files())
    }
    pub fn pick_folder(self) -> Option<PathBuf> {
        guarded(|| self.0.pick_folder())
    }
    pub fn save_file(self) -> Option<PathBuf> {
        guarded(|| self.0.save_file())
    }
}

#[cfg(not(target_os = "macos"))]
fn guarded<T>(run: impl FnOnce() -> Option<T>) -> Option<T> {
    run()
}

#[cfg(target_os = "macos")]
fn guarded<T>(run: impl FnOnce() -> Option<T>) -> Option<T> {
    use objc2::MainThreadMarker;
    use objc2::runtime::ProtocolObject;
    use objc2_app_kit::NSApplication;

    let Some(mtm) = MainThreadMarker::new() else {
        return run();
    };
    let app = NSApplication::sharedApplication(mtm);
    // NSApp holds its delegate weakly: keep winit's alive while ours stands in.
    let original = app.delegate();
    let stand_in = mac::QuitGuard::new(mtm);
    app.setDelegate(Some(ProtocolObject::from_ref(&*stand_in)));
    let result = run();
    app.setDelegate(original.as_deref());
    if QUIT_REQUESTED.load(Ordering::SeqCst) { None } else { result }
}

#[cfg(target_os = "macos")]
mod mac {
    use super::QUIT_REQUESTED;
    use objc2::rc::Retained;
    use objc2::runtime::{NSObject, NSObjectProtocol};
    use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send};
    use objc2_app_kit::{NSApplication, NSApplicationDelegate, NSApplicationTerminateReply};
    use std::sync::atomic::Ordering;

    define_class!(
        // SAFETY: NSObject has no subclassing requirements; the class adds no ivars and no Drop.
        #[unsafe(super(NSObject))]
        #[thread_kind = MainThreadOnly]
        #[name = "PSDialogQuitGuard"]
        pub struct QuitGuard;

        unsafe impl NSObjectProtocol for QuitGuard {}

        unsafe impl NSApplicationDelegate for QuitGuard {
            /// Quitting while a dialog runs: close the dialog as cancelled and quit afterwards.
            #[unsafe(method(applicationShouldTerminate:))]
            fn should_terminate(&self, sender: &NSApplication) -> NSApplicationTerminateReply {
                QUIT_REQUESTED.store(true, Ordering::SeqCst);
                sender.abortModal();
                NSApplicationTerminateReply::TerminateCancel
            }
        }
    );

    impl QuitGuard {
        pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
            unsafe { msg_send![Self::alloc(mtm), init] }
        }
    }
}
