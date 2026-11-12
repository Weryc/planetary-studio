//! What the learned model says about real images, as the app would see them: whether it
//! answers and the strength it suggests (1 = ×12 at the peak), which night hypotheses answer
//! and whether they are offered for comparison, and the belts and zones measured.
//!
//! `cargo run --release -p ps-io --example ml_check -- image…`
use ps_core::learned::smart_curve;

fn main() {
    println!("{:<52} {:>9} {:>10} {:>16}  bande scure / chiare (px)", "immagine", "tipo", "forza", "serate");
    for path in std::env::args().skip(1) {
        let Ok(img) = ps_io::load(path.as_ref()) else { continue };
        let a = ps_core::analysis::analyze(&img);
        let strength = smart_curve(&a).map_or("nessuna".to_string(), |(s, _)| format!("{:.0} %", s * 100.0));
        let name: String = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().chars().take(52).collect();
        let bands = a.bands.as_ref().map_or(String::new(), |b| format!("{:?} / {:?}", b.dark, b.bright));
        // The night hypotheses that answer (C, M, T) and whether they differ enough to compare.
        let nights = ps_core::learned::nights(&a);
        let tags: String = nights.iter().map(|n| format!("{:?}", n.0).chars().next().unwrap()).collect();
        let nights_text = format!("{tags}{}", if ps_core::learned::nights_differ(&nights) { " confronto" } else { "" });
        println!("{name:<52} {:>9} {strength:>10} {nights_text:>16}  {bands}", format!("{:?}", a.kind));
    }
}
