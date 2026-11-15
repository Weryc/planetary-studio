//! Uncompressed 32-bit float TIFF (mono or RGB, interleaved), the format stacks are kept in.
use std::io::Write;
use std::path::Path;

pub fn write_f32(path: &Path, w: usize, h: usize, planes: &[Vec<f32>]) -> std::io::Result<()> {
    let spp = planes.len();
    let data_len = w * h * spp * 4;
    let mut b: Vec<u8> = Vec::with_capacity(data_len + 256);
    b.extend_from_slice(b"II*\0");
    b.extend_from_slice(&0u32.to_le_bytes());
    for i in 0..w * h {
        for p in planes {
            b.extend_from_slice(&p[i].to_le_bytes());
        }
    }
    let shorts = |b: &mut Vec<u8>, v: u16| -> u32 {
        let at = b.len() as u32;
        for _ in 0..spp {
            b.extend_from_slice(&v.to_le_bytes());
        }
        at
    };
    let bps_at = shorts(&mut b, 32);
    let format_at = shorts(&mut b, 3);
    if b.len() % 2 == 1 {
        b.push(0);
    }
    let ifd = b.len() as u32;
    b[4..8].copy_from_slice(&ifd.to_le_bytes());
    const SHORT: u16 = 3;
    const LONG: u16 = 4;
    let (w32, h32) = (w as u32, h as u32);
    let many = |inline: u32, at: u32| if spp == 1 { inline } else { at };
    let entries: [(u16, u16, u32, u32); 11] = [
        (256, LONG, 1, w32),
        (257, LONG, 1, h32),
        (258, SHORT, spp as u32, many(32, bps_at)),
        (259, SHORT, 1, 1),
        (262, SHORT, 1, if spp == 3 { 2 } else { 1 }),
        (273, LONG, 1, 8),
        (277, SHORT, 1, spp as u32),
        (278, LONG, 1, h32),
        (279, LONG, 1, data_len as u32),
        (284, SHORT, 1, 1),
        (339, SHORT, spp as u32, many(3, format_at)),
    ];
    b.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, typ, count, value) in entries {
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&typ.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&value.to_le_bytes());
    }
    b.extend_from_slice(&0u32.to_le_bytes());
    std::fs::File::create(path)?.write_all(&b)
}
