//! Embeds the application icon into the Windows executable.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../packaging/icon.ico");
        res.set("ProductName", "Planetary Studio");
        res.set("FileDescription", "Planetary Studio");
        res.compile().expect("compile Windows resources");
    }
}
