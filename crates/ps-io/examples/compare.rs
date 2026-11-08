//! Side-by-side crop and sharpness of two images:
//! `cargo run --release -p ps-io --example compare -- a.tif b.tif out.png`
use ps_core::Image;

fn sharpness(img: &Image) -> f64 {
    let (w, p) = (img.width(), &img.channels()[0]);
    p.windows(w + 1).map(|v| ((v[w] - v[0]) as f64).abs()).sum::<f64>() / p.len() as f64
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (x, y) = (ps_io::load(a[0].as_ref()).unwrap(), ps_io::load(a[1].as_ref()).unwrap());
    println!("sharpness {:.5} → {:.5}", sharpness(&x), sharpness(&y));
    let (w, h) = (x.width(), x.height());
    let (cw, ch) = (w / 2, h / 2);
    let (cx, cy) = (w / 4, h / 4);
    let channels = (0..x.channels().len())
        .map(|c| {
            (0..ch).flat_map(|r| {
                let row = (cy + r) * w + cx;
                x.channels()[c][row..row + cw].iter().chain(&y.channels()[c][row..row + cw]).copied().collect::<Vec<_>>()
            }).collect()
        })
        .collect();
    ps_io::save(a[2].as_ref(), &Image::new(2 * cw, ch, channels, 16)).unwrap();
}
