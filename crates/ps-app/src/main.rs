#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod history;
mod annotate;
mod identify_ui;
mod instrument;
mod mosaic_ui;
mod naming;
mod panels;
mod prefs;
mod resize_ui;
mod theme;
mod viewer;
mod widgets;
mod worker;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Planetary Studio")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([900.0, 600.0])
            .with_drag_and_drop(true)
            // Without this eframe swaps in its own default icon once the window opens.
            .with_icon(eframe::icon_data::from_png_bytes(include_bytes!("../../../packaging/icon_256.png")).expect("valid icon")),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "Planetary Studio",
        options,
        Box::new(|cc| {
            // A file passed on the command line (or via "Open with…") is opened at start-up.
            let initial_file = std::env::args_os().nth(1).map(Into::into);
            Ok(Box::new(app::PlanetaryStudio::new(cc, initial_file)))
        }),
    )
}
