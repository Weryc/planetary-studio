//! Animation export: any sequence of RGB frames → an animated GIF (up to 50 fps, the most the
//! format allows) or an H.264 MP4 at any frame rate through an external `ffmpeg`.
//!
//! The exporter knows nothing about planets: it takes a [`FrameSource`] (the frames are
//! rendered on demand, so a minute of 60 fps video is never held in memory) and writes it out.
//! The animation of the loaded captures and the rotating 3D globe both feed it.
//!
//! GIF: one palette of 256 colours for the whole animation (median cut on the colours that
//! really occur, so planetary tones — black sky, ochre, cream, a little red — get most of the
//! entries), ordered dithering of adjustable strength (a fixed pattern does not flicker from
//! frame to frame the way an error-diffusion one does), looping, delays kept exact for any
//! frame rate by carrying the rounding of the centiseconds from frame to frame. The encoder is
//! in-house (LZW + the GIF89a blocks) so no dependency is needed.

use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// One 8-bit RGB frame, rows top to bottom, 3 bytes per pixel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbFrame {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl RgbFrame {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        assert_eq!(pixels.len(), width * height * 3, "RGB frame: wrong buffer size");
        Self { width, height, pixels }
    }

    /// The frame cropped to even sides (H.264 in 4:2:0 needs them).
    fn even(&self) -> std::borrow::Cow<'_, RgbFrame> {
        let (w, h) = (self.width & !1, self.height & !1);
        if (w, h) == (self.width, self.height) {
            return std::borrow::Cow::Borrowed(self);
        }
        let mut pixels = Vec::with_capacity(w * h * 3);
        for row in self.pixels.chunks(self.width * 3).take(h) {
            pixels.extend_from_slice(&row[..w * 3]);
        }
        std::borrow::Cow::Owned(RgbFrame { width: w, height: h, pixels })
    }
}

/// Where the frames of an animation come from. `frame(i)` may be called from several threads
/// and in any order; every frame has the same size.
pub trait FrameSource: Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn size(&self) -> (usize, usize);
    fn frame(&self, index: usize) -> RgbFrame;
}

impl FrameSource for Vec<RgbFrame> {
    fn len(&self) -> usize {
        <[RgbFrame]>::len(self)
    }
    fn size(&self) -> (usize, usize) {
        self.first().map_or((0, 0), |f| (f.width, f.height))
    }
    fn frame(&self, index: usize) -> RgbFrame {
        self[index].clone()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("no frames to write")]
    Empty,
    #[error("the frame size ({0}×{1}) is not supported")]
    Size(usize, usize),
    #[error("ffmpeg was not found: install it (for example `brew install ffmpeg`) to make MP4 videos")]
    NoFfmpeg,
    #[error("ffmpeg failed: {0}")]
    Ffmpeg(String),
    #[error("cancelled")]
    Cancelled,
}

/// Progress callback: (frames done, frames in all); return `false` to stop.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize) -> bool;

// ───────────────────────────── GIF ─────────────────────────────

/// The fastest a GIF can really play: its delays are whole centiseconds, and browsers treat
/// anything under 2 as 10.
pub const GIF_MAX_FPS: f32 = 50.0;

#[derive(Clone, Copy, Debug)]
pub struct GifOptions {
    /// Frames per second, 1‥[`GIF_MAX_FPS`].
    pub fps: f32,
    /// Times to play it again: 0 = forever.
    pub loops: u16,
    /// Strength of the dithering, 0 (none) ‥ 1 (one step of the 5-bit colour grid).
    pub dither: f32,
}

impl Default for GifOptions {
    fn default() -> Self {
        Self { fps: 25.0, loops: 0, dither: 0.5 }
    }
}

/// What was written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GifStats {
    pub frames: usize,
    pub bytes: usize,
    /// Playing time of one pass, seconds (the delays being whole centiseconds, it can be a hair
    /// off `frames / fps`).
    pub seconds: f32,
}

/// Delay of each frame in centiseconds, the rounding carried over so the total stays right.
pub fn gif_delays(frames: usize, fps: f32) -> Vec<u16> {
    let fps = fps.clamp(1.0, GIF_MAX_FPS) as f64;
    let at = |i: usize| (i as f64 * 100.0 / fps).round() as i64;
    (0..frames).map(|i| ((at(i + 1) - at(i)).max(2)) as u16).collect()
}

/// Write the sequence as an animated GIF.
pub fn write_gif(src: &dyn FrameSource, opts: &GifOptions, out: &mut dyn Write, progress: Progress) -> Result<GifStats, VideoError> {
    let n = src.len();
    let (w, h) = src.size();
    if n == 0 {
        return Err(VideoError::Empty);
    }
    if w == 0 || h == 0 || w > 65535 || h > 65535 {
        return Err(VideoError::Size(w, h));
    }
    // Palette from a spread of frames (all of them when there are few).
    let step = (n as f32 / 16.0).max(1.0);
    let mut hist = Histogram::new();
    let mut k = 0.0f32;
    while (k as usize) < n {
        hist.add(&src.frame(k as usize), 400_000 / n.min(16));
        k += step;
    }
    let palette = hist.palette();
    let lookup = palette.lookup();

    let delays = gif_delays(n, opts.fps);
    let mut written = CountingWriter { inner: BufWriter::new(out), bytes: 0 };
    let w16 = (w as u16).to_le_bytes();
    let h16 = (h as u16).to_le_bytes();
    written.write_all(b"GIF89a")?;
    written.write_all(&[w16[0], w16[1], h16[0], h16[1], 0xF7, 0, 0])?;
    for c in &palette.colours {
        written.write_all(c)?;
    }
    // Looping (Netscape extension).
    written.write_all(&[0x21, 0xFF, 0x0B])?;
    written.write_all(b"NETSCAPE2.0")?;
    written.write_all(&[3, 1, (opts.loops & 0xFF) as u8, (opts.loops >> 8) as u8, 0])?;

    let amplitude = opts.dither.clamp(0.0, 1.0) * 8.0;
    let mut indices = vec![0u8; w * h];
    let mut total = 0u32;
    for (i, &delay) in delays.iter().enumerate() {
        if !progress(i, n) {
            return Err(VideoError::Cancelled);
        }
        let frame = src.frame(i);
        if (frame.width, frame.height) != (w, h) {
            return Err(VideoError::Size(frame.width, frame.height));
        }
        quantise(&frame, &lookup, amplitude, &mut indices);
        // Graphic control: leave the frame in place, no transparency, the delay.
        written.write_all(&[0x21, 0xF9, 4, 0x04, (delay & 0xFF) as u8, (delay >> 8) as u8, 0, 0])?;
        written.write_all(&[0x2C, 0, 0, 0, 0, w16[0], w16[1], h16[0], h16[1], 0, 8])?;
        let mut data = Vec::with_capacity(indices.len() / 3);
        lzw_encode(&indices, &mut data);
        for block in data.chunks(255) {
            written.write_all(&[block.len() as u8])?;
            written.write_all(block)?;
        }
        written.write_all(&[0])?;
        total += delay as u32;
    }
    written.write_all(&[0x3B])?;
    written.flush()?;
    progress(n, n);
    Ok(GifStats { frames: n, bytes: written.bytes, seconds: total as f32 / 100.0 })
}

struct CountingWriter<W: Write> {
    inner: W,
    bytes: usize,
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes += n;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Colours on a 5-bit-per-channel grid: how many pixels fall in each cell and their true sum.
struct Histogram {
    count: Vec<u32>,
    sum: Vec<[u64; 3]>,
}

fn cell(r: u8, g: u8, b: u8) -> usize {
    ((r as usize >> 3) << 10) | ((g as usize >> 3) << 5) | (b as usize >> 3)
}

impl Histogram {
    fn new() -> Self {
        Self { count: vec![0; 32768], sum: vec![[0; 3]; 32768] }
    }

    /// Count about `budget` pixels of the frame.
    fn add(&mut self, f: &RgbFrame, budget: usize) {
        let n = f.width * f.height;
        let stride = (n / budget.max(1)).max(1);
        for p in f.pixels.chunks_exact(3).step_by(stride) {
            let c = cell(p[0], p[1], p[2]);
            self.count[c] += 1;
            for k in 0..3 {
                self.sum[c][k] += p[k] as u64;
            }
        }
    }

    /// 256 colours by median cut: the box with the most weight in its spread is split at the
    /// weighted median of its widest channel.
    fn palette(&self) -> Palette {
        struct Box {
            cells: Vec<u16>,
            population: u64,
            spread: u32,
            axis: usize,
        }
        let measure = |cells: Vec<u16>| {
            let (mut lo, mut hi) = ([31u32; 3], [0u32; 3]);
            let mut population = 0u64;
            for &c in &cells {
                let rgb = [(c as u32 >> 10) & 31, (c as u32 >> 5) & 31, c as u32 & 31];
                for k in 0..3 {
                    lo[k] = lo[k].min(rgb[k]);
                    hi[k] = hi[k].max(rgb[k]);
                }
                population += self.count[c as usize] as u64;
            }
            let ranges: [u32; 3] = std::array::from_fn(|k| hi[k].saturating_sub(lo[k]));
            // Green weighs more for the eye.
            let weighted = [ranges[0] * 3, ranges[1] * 4, ranges[2] * 2];
            let axis = (0..3).max_by_key(|&k| weighted[k]).unwrap_or(1);
            Box { cells, population, spread: weighted[axis], axis }
        };
        let used: Vec<u16> = (0..32768u16).filter(|&c| self.count[c as usize] > 0).collect();
        let mut boxes = vec![measure(used)];
        while boxes.len() < 256 {
            // The box worth splitting most: many pixels over a wide spread (and more than one cell).
            let Some((at, _)) = boxes.iter().enumerate().filter(|(_, b)| b.cells.len() > 1 && b.spread > 0).max_by_key(|(_, b)| b.population as u128 * (b.spread as u128 + 1)) else { break };
            let b = boxes.swap_remove(at);
            let axis = b.axis;
            let key = |c: &u16| match axis {
                0 => (*c as u32 >> 10) & 31,
                1 => (*c as u32 >> 5) & 31,
                _ => *c as u32 & 31,
            };
            let mut cells = b.cells;
            cells.sort_by_key(key);
            let half = b.population / 2;
            let (mut acc, mut cut) = (0u64, 1usize);
            for (i, &c) in cells.iter().enumerate() {
                acc += self.count[c as usize] as u64;
                if acc >= half {
                    cut = (i + 1).clamp(1, cells.len() - 1);
                    break;
                }
            }
            let right = cells.split_off(cut);
            boxes.push(measure(cells));
            boxes.push(measure(right));
        }
        let mut colours = vec![[0u8; 3]; 256];
        for (slot, b) in colours.iter_mut().zip(&boxes) {
            let (mut s, mut n) = ([0u64; 3], 0u64);
            for &c in &b.cells {
                n += self.count[c as usize] as u64;
                for k in 0..3 {
                    s[k] += self.sum[c as usize][k];
                }
            }
            if n > 0 {
                *slot = std::array::from_fn(|k| ((s[k] + n / 2) / n) as u8);
            }
        }
        Palette { colours }
    }
}

struct Palette {
    colours: Vec<[u8; 3]>,
}

impl Palette {
    /// For each cell of the 5-bit grid, the nearest palette entry (by a perceptual weighting).
    fn lookup(&self) -> Vec<u8> {
        let dist = |a: [i32; 3], b: [u8; 3]| {
            let (dr, dg, db) = (a[0] - b[0] as i32, a[1] - b[1] as i32, a[2] - b[2] as i32);
            3 * dr * dr + 4 * dg * dg + 2 * db * db
        };
        (0..32768usize)
            .map(|c| {
                let centre = [((c >> 10) & 31) as i32 * 8 + 4, ((c >> 5) & 31) as i32 * 8 + 4, (c & 31) as i32 * 8 + 4];
                (0..256).min_by_key(|&i| dist(centre, self.colours[i])).unwrap_or(0) as u8
            })
            .collect()
    }
}

/// 4×4 Bayer thresholds, 0‥15.
const BAYER: [[i32; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

fn quantise(f: &RgbFrame, lookup: &[u8], amplitude: f32, out: &mut [u8]) {
    for (i, (p, o)) in f.pixels.chunks_exact(3).zip(out.iter_mut()).enumerate() {
        let (x, y) = (i % f.width, i / f.width);
        // Zero-mean offset, ±amplitude/2 around the value.
        let d = ((BAYER[y & 3][x & 3] as f32 - 7.5) / 15.0 * amplitude).round() as i32;
        let v = |c: u8| (c as i32 + d).clamp(0, 255) as u8;
        *o = lookup[cell(v(p[0]), v(p[1]), v(p[2]))];
    }
}

/// GIF's variable-width LZW with 8-bit symbols.
fn lzw_encode(indices: &[u8], out: &mut Vec<u8>) {
    const CLEAR: u32 = 256;
    const END: u32 = 257;
    struct Bits<'a> {
        out: &'a mut Vec<u8>,
        acc: u32,
        n: u32,
    }
    impl Bits<'_> {
        fn put(&mut self, code: u32, size: u32) {
            self.acc |= code << self.n;
            self.n += size;
            while self.n >= 8 {
                self.out.push(self.acc as u8);
                self.acc >>= 8;
                self.n -= 8;
            }
        }
    }
    let mut bits = Bits { out, acc: 0, n: 0 };
    let mut table = vec![0u16; 4096 * 256];
    let mut touched: Vec<u32> = Vec::new();
    let (mut size, mut next) = (9u32, 258u32);
    bits.put(CLEAR, size);
    let Some((&first, rest)) = indices.split_first() else {
        bits.put(END, size);
        if bits.n > 0 {
            bits.out.push(bits.acc as u8);
        }
        return;
    };
    let mut prefix = first as u32;
    for &b in rest {
        let key = (prefix << 8) | b as u32;
        let hit = table[key as usize];
        if hit != 0 {
            prefix = hit as u32;
            continue;
        }
        bits.put(prefix, size);
        table[key as usize] = next as u16;
        touched.push(key);
        if next == (1 << size) && size < 12 {
            size += 1;
        }
        next += 1;
        if next == 4096 {
            bits.put(CLEAR, size);
            for k in touched.drain(..) {
                table[k as usize] = 0;
            }
            size = 9;
            next = 258;
        }
        prefix = b as u32;
    }
    bits.put(prefix, size);
    bits.put(END, size);
    if bits.n > 0 {
        bits.out.push(bits.acc as u8);
    }
}

// ───────────────────────────── MP4 ─────────────────────────────

/// Where `ffmpeg` is: `$PS_FFMPEG`, then the PATH, then the usual places (a GUI app on macOS
/// starts with a short PATH that leaves Homebrew out).
pub fn find_ffmpeg() -> Option<PathBuf> {
    let name = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("PS_FFMPEG") {
        candidates.push(PathBuf::from(p));
    }
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|d| d.join(name)));
    }
    for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", "/usr/bin", "/snap/bin", "C:\\ffmpeg\\bin", "C:\\Program Files\\ffmpeg\\bin"] {
        candidates.push(Path::new(dir).join(name));
    }
    candidates.into_iter().find(|p| p.is_file())
}

#[derive(Clone, Copy, Debug)]
pub struct Mp4Options {
    pub fps: f32,
    /// x264 quality: lower is better (14 is visually lossless).
    pub crf: u8,
}

impl Default for Mp4Options {
    fn default() -> Self {
        Self { fps: 60.0, crf: 14 }
    }
}

/// The ffmpeg command line for raw RGB frames on stdin (separate for the tests).
pub fn mp4_arguments(width: usize, height: usize, opts: &Mp4Options, output: &Path) -> Vec<std::ffi::OsString> {
    let mut a: Vec<std::ffi::OsString> = Vec::new();
    let mut add = |s: String| a.push(s.into());
    for s in ["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pixel_format", "rgb24"] {
        add(s.into());
    }
    add("-video_size".into());
    add(format!("{width}x{height}"));
    add("-framerate".into());
    add(format!("{}", opts.fps));
    for s in ["-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf"] {
        add(s.into());
    }
    add(opts.crf.to_string());
    for s in ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r"] {
        add(s.into());
    }
    add(format!("{}", opts.fps));
    a.push(output.as_os_str().to_owned());
    a
}

/// Write the sequence as an H.264 MP4 through `ffmpeg` (a path from [`find_ffmpeg`]). The
/// sides are cut to even numbers when they are not. Returns the number of frames written.
pub fn write_mp4(src: &dyn FrameSource, opts: &Mp4Options, ffmpeg: &Path, output: &Path, progress: Progress) -> Result<usize, VideoError> {
    let n = src.len();
    if n == 0 {
        return Err(VideoError::Empty);
    }
    let (w, h) = (src.size().0 & !1, src.size().1 & !1);
    if w < 2 || h < 2 {
        return Err(VideoError::Size(src.size().0, src.size().1));
    }
    let mut child = Command::new(ffmpeg).args(mp4_arguments(w, h, opts, output)).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound { VideoError::NoFfmpeg } else { VideoError::Io(e) }
    })?;
    let mut stderr = child.stderr.take();
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(s) = stderr.as_mut() {
            s.read_to_string(&mut text).ok();
        }
        text
    });
    let mut stdin = child.stdin.take().expect("piped stdin");
    let mut outcome = Ok(());
    for i in 0..n {
        if !progress(i, n) {
            outcome = Err(VideoError::Cancelled);
            break;
        }
        let frame = src.frame(i);
        let frame = frame.even();
        if (frame.width, frame.height) != (w, h) {
            outcome = Err(VideoError::Size(frame.width, frame.height));
            break;
        }
        if let Err(e) = stdin.write_all(&frame.pixels) {
            // ffmpeg quit early: its message says why.
            outcome = Err(VideoError::Io(e));
            break;
        }
    }
    drop(stdin);
    if matches!(outcome, Err(VideoError::Cancelled)) {
        child.kill().ok();
    }
    let status = child.wait()?;
    let log = reader.join().unwrap_or_default();
    match outcome {
        Err(VideoError::Cancelled) => {
            std::fs::remove_file(output).ok();
            Err(VideoError::Cancelled)
        }
        _ if !status.success() => Err(VideoError::Ffmpeg(log.trim().lines().last().unwrap_or("unknown error").to_string())),
        Err(e) => Err(e),
        Ok(()) => {
            progress(n, n);
            Ok(n)
        }
    }
}
