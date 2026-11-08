//! Prints the automatic analysis of an image: `cargo run --release -p ps-io --example inspect -- file.tif`
fn main() {
    let path = std::env::args().nth(1).expect("path");
    let img = ps_io::load(path.as_ref()).unwrap();
    println!("{:#?}", ps_core::analysis::analyze(&img));
}
