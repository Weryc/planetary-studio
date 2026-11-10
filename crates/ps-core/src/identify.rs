//! "What am I looking at": names on the image from the ephemerides.
//!
//! The capture's time (file name or log) gives where every moon of Jupiter or Saturn is, or
//! which face of the Moon is lit; the image is then searched for how the sky is turned in it
//! (the camera's rotation, and a mirror flip from a diagonal or a Newtonian), so the names can
//! be placed on it:
//! - planets: the planet's outline gives the equator's direction (Saturn's rings, Jupiter's
//!   flattening); the small bright spots around it are matched against the predicted moons for
//!   the side and the mirror;
//! - the Moon, full disk: the limb's circle is fitted on the sharp edge (not the soft
//!   terminator), then every rotation and mirror is tried and the one whose predicted lighting
//!   and dark plains best match the image wins;
//! - close-ups of the Moon can't be placed: the craters along today's terminator are listed.
//!
//! Every choice can be overridden by hand ([`Orientation`]).

use crate::i18n::tl;
use crate::tr;
use crate::Image;
use crate::capture::{Body, Moment};
use crate::color::{ColorMode, working_planes};
use crate::ephemeris::{apparent_diameter, distance_km, geocentric_au, spherical};
use crate::moons::{self, MoonPosition};
use crate::selene::{self, FeatureKind, MoonView};

const RAD: f64 = std::f64::consts::PI / 180.0;

/// How the body's frame lies on the image: its +x axis (planets: west along the equator; the
/// Moon: selenographic east) points at `angle` (degrees, clockwise from the image's +x, as the
/// image's y runs down); `mirrored` when the image is flipped (north and east on the same side).
/// 0°, not mirrored = north up, east on the left: the sky as seen with the naked eye.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Orientation {
    pub angle: f32,
    pub mirrored: bool,
    /// Planets: how much the moons' distances from the planet are stretched (1 = as measured
    /// from the planet's outline); the time-and-moons fit adjusts it a little.
    #[serde(default = "one")]
    pub scale: f32,
}

fn one() -> f32 {
    1.0
}

impl Orientation {
    /// Image offset (y down) of a body-frame offset (y = north up).
    fn to_image(self, x: f64, y: f64) -> (f64, f64) {
        let (s, c) = (self.angle as f64 * RAD).sin_cos();
        let ux = if self.mirrored { -x } else { x };
        let uy = -y;
        (c * ux - s * uy, s * ux + c * uy)
    }

    /// Body-frame offset of an image offset.
    fn body_offset(self, dx: f64, dy: f64) -> (f64, f64) {
        let (s, c) = (self.angle as f64 * RAD).sin_cos();
        let (ux, uy) = (c * dx + s * dy, -s * dx + c * dy);
        (if self.mirrored { -ux } else { ux }, -uy)
    }

    /// What turns this image into the sky as seen by eye (north up, east left): a left-right
    /// mirror first when `mirrored`, then a clockwise turn by the returned degrees.
    pub fn upright(self) -> (bool, f32) {
        (self.mirrored, if self.mirrored { self.angle } else { -self.angle })
    }

    /// Where north points on the image, degrees clockwise from up.
    pub fn north_up_angle(self) -> f32 {
        let (dx, dy) = self.to_image(0.0, 1.0);
        (dx.atan2(-dy) / RAD) as f32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelKind {
    /// A moon of the planet found on the image where predicted.
    MoonFound,
    /// A moon where it should be, with no spot found there.
    MoonPredicted,
    Crater,
    Mare,
    /// Mountains, valleys, rilles.
    Feature,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub name: String,
    /// Image pixels.
    pub x: f32,
    pub y: f32,
    /// Size to draw (radius, px; 0 = a point).
    pub radius: f32,
    pub kind: LabelKind,
    /// On the terminator: the best-lit for relief.
    pub on_terminator: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Identification {
    pub labels: Vec<Label>,
    /// What was computed: (what, value).
    pub facts: Vec<(String, String)>,
    /// How it was found, and what is uncertain.
    pub notes: Vec<String>,
    /// The orientation used (found or given).
    pub orientation: Option<Orientation>,
    /// Whether it was found on the image rather than assumed.
    pub orientation_found: bool,
    /// Centre and radius (px) of the disk found: the planet's equatorial radius, the Moon's limb.
    pub disk: Option<(f32, f32, f32)>,
}

/// Identify what the image shows. `manual`: the user's orientation, used instead of searching.
pub fn identify(img: &Image, body: Body, when: &Moment, manual: Option<Orientation>) -> Identification {
    identify_with(img, body, when, manual, None)
}

/// Where the Moon's disk lies on a close-up that has no limb to measure, found from two craters
/// the user pointed at: the disk's centre and radius (px) and the turn of the Moon's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub cx: f32,
    pub cy: f32,
    pub r: f32,
    pub orientation: Orientation,
}

/// The disk and orientation that put two catalogue craters (`name`, and the image pixel where the
/// user found it) on their pixels. Two points fix the position, the size and the turn of the Moon's
/// frame; which way the image is mirrored they cannot tell, so `mirrored` is given.
pub fn place_on_craters(when: &Moment, a: (&str, (f64, f64)), b: (&str, (f64, f64)), mirrored: bool) -> Result<Placement, &'static str> {
    let v = selene::view(when);
    let find = |name: &str| selene::features().find(|f| f.kind == FeatureKind::Crater && f.name == name);
    let (Some(fa), Some(fb)) = (find(a.0), find(b.0)) else { return Err(tl("Scegli i due crateri dalla lista.")) };
    if fa.name == fb.name {
        return Err(tl("Servono due crateri diversi."));
    }
    let (xa, ya, front_a) = v.project(fa.lon, fa.lat);
    let (xb, yb, front_b) = v.project(fb.lon, fb.lat);
    if !front_a || !front_b {
        return Err(tl("Uno dei due crateri è sulla faccia nascosta della Luna."));
    }
    // Body frame (y north) to the "unmirrored, y down" frame the turn acts on.
    let u = |dx: f64, dy: f64| (if mirrored { -dx } else { dx }, -dy);
    let (ux, uy) = u(xb - xa, yb - ya);
    let (dx, dy) = (b.1.0 - a.1.0, b.1.1 - a.1.1);
    let (lu, ld) = (ux.hypot(uy), dx.hypot(dy));
    if lu < 0.02 {
        return Err(tl("I due crateri sono troppo vicini sulla Luna: scegline due più lontani."));
    }
    if ld < 6.0 {
        return Err(tl("I due punti cliccati sono troppo vicini."));
    }
    let r = ld / lu;
    let angle = ((dy.atan2(dx) - uy.atan2(ux)) / RAD).rem_euclid(360.0);
    let orientation = Orientation { angle: angle as f32, mirrored, scale: 1.0 };
    let (ox, oy) = orientation.to_image(xa, ya);
    Ok(Placement { cx: (a.1.0 - r * ox) as f32, cy: (a.1.1 - r * oy) as f32, r: r as f32, orientation })
}

/// [`identify`] with the Moon's disk given (from [`place_on_craters`]) instead of searched.
pub fn identify_with(img: &Image, body: Body, when: &Moment, manual: Option<Orientation>, placed: Option<Placement>) -> Identification {
    let lum = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let (w, h) = (img.width(), img.height());
    let mut out = Identification::default();
    common_facts(&mut out, body, when);
    match body {
        Body::Moon => moon(&mut out, &lum, w, h, when, manual, placed),
        Body::Sun => sun_facts(&mut out, when),
        Body::Jupiter | Body::Saturn => planet(&mut out, &lum, w, h, body, when, manual),
        _ => {
            if let Some(blob) = outline(&lum, w, h) {
                out.disk = Some((blob.cx, blob.cy, blob.major));
                scale_fact(&mut out, body, when, blob.major as f64);
            }
            out.notes.push(tr!("{} non ha lune abbastanza luminose da riconoscere.", "{} has no moons bright enough to recognise.", body.label()));
        }
    }
    out
}

/// The capture time (and the small turn and stretch) that put the predicted moons on the spots
/// found next to the planet — all of them together, not one.
#[derive(Clone, Debug, PartialEq)]
pub struct TimeFit {
    pub when: Moment,
    /// Fitted time minus the given one, minutes.
    pub shift_min: f64,
    /// Moons on a spot at the fitted time, out of those in the field.
    pub matched: usize,
    pub total: usize,
    /// The orientation given, corrected by a small turn and stretch of the moons' distances.
    pub orientation: Orientation,
}

/// Searches `±window_min` minutes around `when`, a few degrees of turn and a few percent of
/// stretch around `orientation`, for what puts every moon of `body` on a bright spot of the image
/// at once. Several moons pin the three numbers down together; with a single one only the time moves.
pub fn fit_time(img: &Image, body: Body, when: &Moment, orientation: Orientation, window_min: f64) -> Option<TimeFit> {
    if !matches!(body, Body::Jupiter | Body::Saturn) {
        return None;
    }
    let lum = working_planes(img, ColorMode::Luminance).swap_remove(0);
    let (w, h) = (img.width(), img.height());
    let o = outline(&lum, w, h)?;
    let tilt = if body == Body::Saturn { saturn_ring_tilt(when) } else { None };
    let rings_open = tilt.is_some_and(|b| b.abs() > 3.0) && o.major / o.minor > 1.3;
    let radius = if body == Body::Saturn && rings_open { o.major as f64 / RING_OUTER } else { o.major as f64 };
    let (cx, cy) = (o.cx as f64, o.cy as f64);
    let (sa, ca) = o.angle.sin_cos();
    let keep_out = |x: f64, y: f64| {
        let (dx, dy) = (x - cx, y - cy);
        let (u, v) = (dx * ca + dy * sa, -dx * sa + dy * ca);
        (u / (o.major as f64 + 3.0)).powi(2) + (v / (o.minor as f64 + 3.0)).powi(2) < 1.0
    };
    let found = spots(&lum, w, h, &o, keep_out);
    if found.is_empty() {
        return None;
    }
    let flat = moons::flattening(body);
    let tolerance = (0.35 * radius).max(5.0);
    let jd0 = when.julian();
    let base_scale = orientation.scale as f64;
    // A point of the search: minutes, degrees of turn, stretch.
    let eval = |dt: f64, da: f64, sc: f64, tolerance: f64| -> (f64, usize, usize) {
        let t = Moment::from_julian(jd0 + dt / 1440.0);
        let or = Orientation { angle: (orientation.angle as f64 + da) as f32, scale: 1.0, ..orientation };
        let (mut cost, mut matched, mut total) = (0.0, 0, 0);
        for m in moons::positions(body, &t).iter().filter(|m| !m.occulted(flat) && m.x.hypot(m.y) >= 1.15) {
            let k = radius * base_scale * sc;
            let (dx, dy) = or.to_image(m.x * k, m.y * k);
            let p = (cx + dx, cy + dy);
            if p.0 < 0.0 || p.1 < 0.0 || p.0 >= w as f64 || p.1 >= h as f64 {
                continue;
            }
            total += 1;
            let d = found.iter().map(|f| (f.0 as f64 - p.0).hypot(f.1 as f64 - p.1)).fold(f64::MAX, f64::min);
            // Reward a moon on a spot; one with none costs nothing.
            cost -= (1.0 - (d / tolerance).powi(2)).max(0.0);
            if d < 0.25 * tolerance {
                matched += 1;
            }
        }
        (cost, matched, total)
    };
    // Equal fits: the smaller change wins (the turn and the stretch cost a little).
    let penalty = |dt: f64, da: f64, sc: f64| 1e-6 * dt.abs() + 0.01 * da.abs() + 0.5 * (sc - 1.0).abs();
    let (_, _, moons_in_field) = eval(0.0, 0.0, 1.0, tolerance);
    // One moon can't pin the turn and the stretch down as well as the time.
    let free = moons_in_field >= 2;
    let turns: Vec<f64> = if free { (-4..=4).map(|k| k as f64 * 1.5).collect() } else { vec![0.0] };
    let stretches: Vec<f64> = if free { (-4..=4).map(|k| 1.0 + k as f64 * 0.02).collect() } else { vec![1.0] };
    let steps = window_min.ceil() as i64;
    let mut best = (0.0f64, 0.0f64, 1.0f64);
    let mut best_cost = f64::MAX;
    for k in -steps..=steps {
        for &da in &turns {
            for &sc in &stretches {
                let dt = k as f64;
                let (c, _, total) = eval(dt, da, sc, tolerance);
                if total == 0 || c >= 0.0 {
                    continue;
                }
                let c = c + penalty(dt, da, sc);
                if c < best_cost {
                    (best, best_cost) = ((dt, da, sc), c);
                }
            }
        }
    }
    if best_cost == f64::MAX {
        return None;
    }
    // Refine all three together, with a tighter tolerance so a moon without a spot pulls less.
    let tight = tolerance * 0.4;
    let mut at = best;
    let mut at_cost = eval(at.0, at.1, at.2, tight).0 + penalty(at.0, at.1, at.2);
    let mut step = (1.0f64, 1.0f64, 0.015f64);
    for _ in 0..40 {
        let mut improved = false;
        for axis in 0..3 {
            if !free && axis > 0 {
                continue;
            }
            for sign in [-1.0, 1.0] {
                let mut p = at;
                match axis {
                    0 => p.0 += sign * step.0,
                    1 => p.1 += sign * step.1,
                    _ => p.2 += sign * step.2,
                }
                let c = eval(p.0, p.1, p.2, tight).0 + penalty(p.0, p.1, p.2);
                if c < at_cost - 1e-12 {
                    (at, at_cost, improved) = (p, c, true);
                }
            }
        }
        if !improved {
            step = (step.0 / 2.0, step.1 / 2.0, step.2 / 2.0);
            if step.0 < 1.0 / 120.0 {
                break;
            }
        }
    }
    let (_, matched, total) = eval(at.0, at.1, at.2, tolerance);
    let fitted = Orientation { angle: (orientation.angle as f64 + at.1).rem_euclid(360.0) as f32, scale: (base_scale * at.2) as f32, ..orientation };
    Some(TimeFit { when: Moment::from_julian(jd0 + at.0 / 1440.0), shift_min: at.0, matched, total, orientation: fitted })
}

fn common_facts(out: &mut Identification, body: Body, when: &Moment) {
    out.facts.push((tl("Oggetto").into(), body.label().into()));
    out.facts.push((tl("Data e ora (UT)").into(), format!("{:04}-{:02}-{:02} {:02}:{:02}", when.year, when.month, when.day, when.hour, when.minute)));
    let d = apparent_diameter(body, when);
    let dist = distance_km(body, when);
    if matches!(body, Body::Moon) {
        out.facts.push((tl("Distanza").into(), format!("{:.0} km", dist)));
    } else {
        out.facts.push((tl("Distanza").into(), tr!("{:.3} UA", "{:.3} AU", dist / 149_597_870.7)));
    }
    out.facts.push((tl("Diametro apparente").into(), if d > 120.0 { format!("{:.1}′", d / 60.0) } else { format!("{d:.2}″") }));
}

/// The image scale from the radius in pixels.
fn scale_fact(out: &mut Identification, body: Body, when: &Moment, radius_px: f64) {
    let d = apparent_diameter(body, when);
    if radius_px > 2.0 {
        out.facts.push((tl("Scala dell'immagine").into(), format!("{:.3}″/px", d / (2.0 * radius_px))));
    }
}

// ---- planets -------------------------------------------------------------------------------

/// The planet's outline: centre, major-axis direction (radians, y down) and half-lengths.
struct Outline {
    cx: f32,
    cy: f32,
    angle: f64,
    major: f32,
    minor: f32,
    level: f32,
    background: f32,
    noise: f32,
    mask: Vec<bool>,
}

fn percentile(values: &[f32], q: f32) -> f32 {
    let mut v: Vec<f32> = values.iter().copied().step_by((values.len() / 200_000).max(1)).collect();
    if v.is_empty() {
        return 0.0;
    }
    let k = ((v.len() - 1) as f32 * q) as usize;
    *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}

fn outline(lum: &[f32], w: usize, h: usize) -> Option<Outline> {
    let background = percentile(lum, 0.2);
    // The 50th brightest pixel: the subject however small, not a hot pixel.
    let bright = {
        let mut v = lum.to_vec();
        let k = v.len().saturating_sub(50);
        *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
    };
    let noise = {
        let dev: Vec<f32> = lum.iter().step_by((lum.len() / 100_000).max(1)).filter(|v| **v < background + (bright - background) * 0.05).map(|v| (v - background).abs()).collect();
        percentile(&dev, 0.5) * 1.4826 + 1e-5
    };
    if bright - background < 10.0 * noise {
        return None;
    }
    let threshold = background + 0.2 * (bright - background);
    let above: Vec<bool> = lum.iter().map(|v| *v > threshold).collect();
    // Largest 4-connected region.
    let mut label = vec![0u32; w * h];
    let (mut best, mut best_size, mut next) = (0u32, 0usize, 0u32);
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !above[start] || label[start] != 0 {
            continue;
        }
        next += 1;
        let mut size = 0;
        stack.push(start);
        label[start] = next;
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if above[j] && label[j] == 0 {
                    label[j] = next;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        if size > best_size {
            (best, best_size) = (next, size);
        }
    }
    if best_size < 20 {
        return None;
    }
    let mask: Vec<bool> = label.iter().map(|l| *l == best).collect();
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0.0f64);
    let mut level = Vec::with_capacity(best_size);
    for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
        sx += (i % w) as f64;
        sy += (i / w) as f64;
        n += 1.0;
        level.push(lum[i]);
    }
    let (cx, cy) = (sx / n, sy / n);
    let (mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0);
    for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
        let (dx, dy) = ((i % w) as f64 - cx, (i / w) as f64 - cy);
        xx += dx * dx;
        yy += dy * dy;
        xy += dx * dy;
    }
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let (s, c) = angle.sin_cos();
    let (mut major, mut minor) = (Vec::with_capacity(best_size), Vec::with_capacity(best_size));
    for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
        let (dx, dy) = ((i % w) as f64 - cx, (i / w) as f64 - cy);
        major.push((dx * c + dy * s).abs() as f32);
        minor.push((-dx * s + dy * c).abs() as f32);
    }
    // The outline runs about half a pixel beyond the outermost pixel centres above threshold.
    let major = percentile(&major, 0.999) + 0.5;
    let minor = percentile(&minor, 0.999) + 0.5;
    Some(Outline { cx: cx as f32, cy: cy as f32, angle, major, minor, level: percentile(&level, 0.5), background, noise, mask })
}

/// Small bright spots away from the planet: (x, y, brightness).
fn spots(lum: &[f32], w: usize, h: usize, o: &Outline, keep_out: impl Fn(f64, f64) -> bool) -> Vec<(f32, f32, f32)> {
    // A 3×3 mean against noise.
    let mean = |x: usize, y: usize| {
        let mut s = 0.0;
        for yy in y - 1..=y + 1 {
            for xx in x - 1..=x + 1 {
                s += lum[yy * w + xx];
            }
        }
        s / 9.0
    };
    let threshold = o.background + (6.0 * o.noise / 3.0).max(0.03 * (o.level - o.background));
    let mut found: Vec<(f32, f32, f32)> = Vec::new();
    for y in 2..h.saturating_sub(2) {
        for x in 2..w.saturating_sub(2) {
            let v = lum[y * w + x];
            if v <= threshold || o.mask[y * w + x] || keep_out(x as f64, y as f64) {
                continue;
            }
            let m = mean(x, y);
            if m <= threshold {
                continue;
            }
            // A local maximum of the smoothed image.
            if !(y - 1..=y + 1).all(|yy| (x - 1..=x + 1).all(|xx| (xx == x && yy == y) || mean(xx, yy) <= m)) {
                continue;
            }
            found.push((x as f32, y as f32, m));
        }
    }
    found.sort_by(|a, b| b.2.total_cmp(&a.2));
    // One per moon: drop maxima close to a brighter one.
    let mut kept: Vec<(f32, f32, f32)> = Vec::new();
    for f in found {
        if kept.iter().all(|k| (k.0 - f.0).hypot(k.1 - f.1) > 6.0) {
            kept.push(f);
        }
        if kept.len() == 24 {
            break;
        }
    }
    kept
}

/// Outer edge of Saturn's A ring, in equatorial radii.
const RING_OUTER: f64 = 2.27;

fn saturn_ring_tilt(when: &Moment) -> Option<f64> {
    let jd = when.julian();
    let t = (jd - 2_451_545.0) / 36_525.0;
    let (lon, lat, _) = spherical(geocentric_au(Body::Saturn, jd)?);
    let i = (28.075216 - 0.012998 * t + 0.000004 * t * t) * RAD;
    let node = (169.508470 + 1.394681 * t + 0.000412 * t * t) * RAD;
    Some((i.sin() * lat.cos() * (lon - node).sin() - i.cos() * lat.sin()).asin() / RAD)
}

fn planet(out: &mut Identification, lum: &[f32], w: usize, h: usize, body: Body, when: &Moment, manual: Option<Orientation>) {
    let predicted = moons::positions(body, when);
    let flat = moons::flattening(body);
    if body == Body::Jupiter {
        let (cm1, cm2) = moons::jupiter_central_meridian(when);
        out.facts.push((tl("Meridiano centrale (Sistema I)").into(), format!("{cm1:.1}°")));
        out.facts.push((tl("Meridiano centrale (Sistema II)").into(), format!("{cm2:.1}°")));
    }
    let tilt = if body == Body::Saturn { saturn_ring_tilt(when) } else { None };
    if let Some(b) = tilt {
        out.facts.push((tl("Inclinazione degli anelli").into(), format!("{b:+.1}°")));
    }
    let Some(o) = outline(lum, w, h) else {
        out.notes.push(tl("Non trovo il pianeta nell'immagine.").into());
        return;
    };
    // Equatorial radius: Saturn's outline spans the rings unless they are nearly edge-on.
    let rings_open = tilt.is_some_and(|b| b.abs() > 3.0) && o.major / o.minor > 1.3;
    let radius = if body == Body::Saturn && rings_open { o.major as f64 / RING_OUTER } else { o.major as f64 };
    out.disk = Some((o.cx, o.cy, radius as f32));
    scale_fact(out, body, when, radius);
    let (cx, cy) = (o.cx as f64, o.cy as f64);
    let elongated = o.major / o.minor > if body == Body::Saturn { 1.08 } else { 1.035 };
    // Keep the spot search off the planet and its rings.
    let (sa, ca) = o.angle.sin_cos();
    let keep_out = |x: f64, y: f64| {
        let (dx, dy) = (x - cx, y - cy);
        let (u, v) = (dx * ca + dy * sa, -dx * sa + dy * ca);
        (u / (o.major as f64 + 3.0)).powi(2) + (v / (o.minor as f64 + 3.0)).powi(2) < 1.0
    };
    let found = spots(lum, w, h, &o, keep_out);
    let visible: Vec<&MoonPosition> = predicted.iter().filter(|m| !m.occulted(flat)).collect();

    let place = |or: Orientation, m: &MoonPosition| {
        let k = radius * or.scale as f64;
        let (dx, dy) = or.to_image(m.x * k, m.y * k);
        (cx + dx, cy + dy)
    };
    let inside = |(x, y): (f64, f64)| x >= 0.0 && y >= 0.0 && x < w as f64 && y < h as f64;
    let tolerance = (0.35 * radius).max(5.0);
    let score = |or: Orientation| -> (f64, usize) {
        let mut s = 0.0;
        let mut n = 0;
        for m in &visible {
            let p = place(or, m);
            if !inside(p) || m.x.hypot(m.y) < 1.15 {
                continue;
            }
            let d = found.iter().map(|f| (f.0 as f64 - p.0).hypot(f.1 as f64 - p.1)).fold(f64::MAX, f64::min);
            if d < tolerance {
                s += 1.0 - (d / tolerance).powi(2);
                n += 1;
            }
        }
        (s, n)
    };

    let (orientation, found_it) = if let Some(m) = manual {
        (m, true)
    } else {
        let axis = o.angle / RAD;
        let candidates: Vec<f64> = if elongated { (-20..=20).flat_map(|k| [axis + k as f64 * 0.5, axis + 180.0 + k as f64 * 0.5]).collect() } else { (0..720).map(|k| k as f64 * 0.5).collect() };
        let mut best = (Orientation { angle: axis as f32, mirrored: false, scale: 1.0 }, 0.0, 0usize);
        for a in candidates {
            for mirrored in [false, true] {
                let or = Orientation { angle: a.rem_euclid(360.0) as f32, mirrored, scale: 1.0 };
                let (s, n) = score(or);
                if s > best.1 + 1e-9 {
                    best = (or, s, n);
                }
            }
        }
        let found_it = best.2 >= 1;
        if found_it {
            out.notes.push(tr!("Orientamento trovato: {} {} su {} previst{} nel campo.", "Orientation found: {} {} out of {} predicted in the field{}.",
                best.2,
                if best.2 == 1 { tl("luna riconosciuta") } else { tl("lune riconosciute") },
                visible.iter().filter(|m| inside(place(best.0, m))).count(),
                if crate::i18n::is_en() { "" } else if visible.len() == 1 { "a" } else { "e" }
            ));
            if best.2 == 1 {
                out.notes.push(tl("Con una sola luna lo specchio dell'immagine resta incerto: controlla le posizioni.").into());
            }
        } else if elongated {
            out.notes.push(tl("Nessuna luna riconosciuta: l'asse viene dalla forma del pianeta, ma il lato (est/ovest) e lo specchio sono da verificare.").into());
        } else {
            out.notes.push(tl("Nessuna luna riconosciuta e il disco è quasi rotondo: orientamento ipotizzato (nord in alto). Correggilo a mano.").into());
            best.0 = Orientation { angle: 0.0, mirrored: false, scale: 1.0 };
        }
        (best.0, found_it)
    };
    out.orientation = Some(orientation);
    out.orientation_found = found_it && manual.is_none();

    for m in &predicted {
        let p = place(orientation, m);
        let dist = m.x.hypot(m.y);
        let side = if m.x > 0.0 { tl("ovest") } else { "est" };
        let state = if m.occulted(flat) {
            tl("dietro al pianeta").to_string()
        } else if m.in_transit(flat) {
            tl("davanti al disco (transito)").to_string()
        } else if !inside(p) {
            tr!("fuori campo, {dist:.1} raggi a {side}", "out of the field, {dist:.1} radii to the {side}")
        } else {
            tr!("{dist:.1} raggi a {side}", "{dist:.1} radii to the {side}")
        };
        out.facts.push((m.name.to_string(), state));
        if m.occulted(flat) || !inside(p) {
            continue;
        }
        let near = found.iter().map(|f| (f.0 as f64 - p.0).hypot(f.1 as f64 - p.1)).fold(f64::MAX, f64::min);
        let kind = if near < tolerance && dist >= 1.15 { LabelKind::MoonFound } else { LabelKind::MoonPredicted };
        out.labels.push(Label { name: m.name.to_string(), x: p.0 as f32, y: p.1 as f32, radius: 0.0, kind, on_terminator: false });
    }
}

// ---- the Moon -------------------------------------------------------------------------------

/// Largest side the Moon's search works at.
const MOON_WORK_SIDE: usize = 900;

fn downsample(lum: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize, usize) {
    let f = w.max(h).div_ceil(MOON_WORK_SIDE).max(1);
    if f == 1 {
        return (lum.to_vec(), w, h, 1);
    }
    (crate::image::downscale_plane(lum, w, h, f), (w / f).max(1), (h / f).max(1), f)
}

/// The limb's circle (centre, radius) on the small image, from the sharp edge where the
/// brightness falls outwards.
fn fit_limb(small: &[f32], w: usize, h: usize) -> Option<(f64, f64, f64, usize)> {
    if w < 16 || h < 16 {
        return None;
    }
    let at = |x: usize, y: usize| small[y * w + x];
    let mut edges: Vec<(f64, f64, f64, f64, f32)> = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            let g = gx.hypot(gy);
            if g > 0.0 {
                edges.push((x as f64, y as f64, (gx / g) as f64, (gy / g) as f64, g));
            }
        }
    }
    let mags: Vec<f32> = edges.iter().map(|e| e.4).collect();
    let cut = percentile(&mags, 0.97);
    edges.retain(|e| e.4 >= cut && e.4 > 0.0);
    if edges.len() < 30 {
        return None;
    }
    let (min_r, max_r) = (0.15 * w.min(h) as f64, 3.0 * w.max(h) as f64);
    // Deterministic pseudo-random sampling (same image, same answer).
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rand = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    let circle = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| -> Option<(f64, f64, f64)> {
        let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
        if d.abs() < 1e-9 {
            return None;
        }
        let (a2, b2, c2) = (a.0 * a.0 + a.1 * a.1, b.0 * b.0 + b.1 * b.1, c.0 * c.0 + c.1 * c.1);
        let ux = (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d;
        let uy = (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d;
        Some((ux, uy, (a.0 - ux).hypot(a.1 - uy)))
    };
    let inlier = |e: &(f64, f64, f64, f64, f32), (cx, cy, r): (f64, f64, f64)| {
        let (dx, dy) = (e.0 - cx, e.1 - cy);
        let d = dx.hypot(dy);
        // On the circle, and darker outside (the gradient points inwards).
        (d - r).abs() < 1.5 && (e.2 * dx + e.3 * dy) / d.max(1e-9) < -0.7
    };
    let mut best: Option<((f64, f64, f64), usize)> = None;
    for _ in 0..1500 {
        let (a, b, c) = (edges[rand(edges.len())], edges[rand(edges.len())], edges[rand(edges.len())]);
        let Some(circ) = circle((a.0, a.1), (b.0, b.1), (c.0, c.1)) else { continue };
        if !(min_r..max_r).contains(&circ.2) || ![a, b, c].iter().all(|e| inlier(e, circ)) {
            continue;
        }
        let n = edges.iter().filter(|e| inlier(e, circ)).count();
        if best.is_none_or(|(_, m)| n > m) {
            best = Some((circ, n));
        }
    }
    let (mut circ, _) = best?;
    // Least squares on the inliers (Kåsa fit), twice.
    for _ in 0..2 {
        let pts: Vec<(f64, f64)> = edges.iter().filter(|e| inlier(e, circ)).map(|e| (e.0, e.1)).collect();
        if pts.len() < 10 {
            break;
        }
        let n = pts.len() as f64;
        let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n);
        let (mut suu, mut svv, mut suv, mut suuu, mut svvv, mut suvv, mut svuu) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (x, y) in &pts {
            let (u, v) = (x - mx, y - my);
            suu += u * u;
            svv += v * v;
            suv += u * v;
            suuu += u * u * u;
            svvv += v * v * v;
            suvv += u * v * v;
            svuu += v * u * u;
        }
        let det = suu * svv - suv * suv;
        if det.abs() < 1e-9 {
            break;
        }
        let (r1, r2) = (0.5 * (suuu + suvv), 0.5 * (svvv + svuu));
        let uc = (r1 * svv - r2 * suv) / det;
        let vc = (r2 * suu - r1 * suv) / det;
        let r = (uc * uc + vc * vc + (suu + svv) / n).sqrt();
        circ = (uc + mx, vc + my, r);
    }
    let inliers: Vec<&(f64, f64, f64, f64, f32)> = edges.iter().filter(|e| inlier(e, circ)).collect();
    // The limb must cover a good arc, not a short straight-ish edge.
    let mut sectors = [false; 36];
    for e in &inliers {
        let a = (e.1 - circ.1).atan2(e.0 - circ.0);
        sectors[(((a / std::f64::consts::TAU + 0.5) * 36.0) as usize).min(35)] = true;
    }
    let arc = sectors.iter().filter(|s| **s).count();
    if inliers.len() < 40 || arc < 8 {
        return None;
    }
    Some((circ.0, circ.1, circ.2, arc))
}

fn moon(out: &mut Identification, lum: &[f32], w: usize, h: usize, when: &Moment, manual: Option<Orientation>, placed: Option<Placement>) {
    let v = selene::view(when);
    let waxing = v.waxing();
    out.facts.push((tl("Fase").into(), tr!("{} · illuminata al {:.0} %", "{} · {:.0} % illuminated", v.phase_name(waxing), v.illuminated * 100.0)));
    out.facts.push((tl("Colongitudine").into(), format!("{:.1}°", v.colongitude)));
    let (llon, llat) = (v.earth_lon / RAD, v.earth_lat / RAD);
    out.facts.push((
        tl("Librazione").into(),
        format!("{:.1}° {} · {:.1}° {}", llat.abs(), if llat >= 0.0 { "N" } else { "S" }, llon.abs(), if llon >= 0.0 { "E" } else { "O" }),
    ));
    // The disk: given by two craters, or the limb searched on a reduced copy.
    let (cx, cy, r, found) = if let Some(p) = placed {
        (p.cx as f64, p.cy as f64, p.r as f64, None)
    } else {
        let (small, sw, sh, f) = downsample(lum, w, h);
        let Some((scx, scy, sr, arc)) = fit_limb(&small, sw, sh) else {
            out.notes.push(tl("Il bordo della Luna non è nell'immagine: è un dettaglio della superficie. Indica due crateri qui sotto per collocarlo.").into());
            terminator_list(out, &v);
            return;
        };
        let (cx, cy, r) = ((scx + 0.5) * f as f64 - 0.5, (scy + 0.5) * f as f64 - 0.5, sr * f as f64);
        (cx, cy, r, Some((small, sw, sh, scx, scy, sr, arc)))
    };
    out.disk = Some((cx as f32, cy as f32, r as f32));
    scale_fact(out, Body::Moon, when, r);

    let orientation = if let Some(p) = placed {
        out.orientation_found = true;
        out.notes.push(tl("Collocata con i due crateri indicati: scala e rotazione vengono da lì. Più sono lontani tra loro, più è precisa.").into());
        p.orientation
    } else if let Some((small, sw, sh, scx, scy, sr, arc)) = found {
        if arc < 14 {
            out.notes.push(tl("Bordo trovato su un arco breve: il disco potrebbe essere stimato male.").into());
        }
        match manual {
            Some(m) => m,
            None => {
                let (or, margin) = moon_orientation(&small, sw, sh, (scx, scy, sr), &v);
                // How much of the disk the frame holds: a quarter of it constrains the turn poorly.
                let shown = disk_fraction(cx, cy, r, w, h);
                out.orientation_found = margin > 0.05 && shown > 0.5;
                out.notes.push(if shown <= 0.5 {
                    tr!("Nel campo c'è solo il {:.0} % del disco: l'orientamento è una stima. Ruota o specchia finché le etichette cadono sui crateri giusti, oppure indica due crateri.", "Only {:.0} % of the disk is in the field: the orientation is an estimate. Rotate or mirror until the labels land on the right craters, or point out two craters.", shown * 100.0)
                } else if margin > 0.05 {
                    tl("Orientamento trovato confrontando luce e mari previsti con l'immagine.").into()
                } else {
                    tl("Orientamento incerto (luce e mari si accordano quasi ugualmente con l'immagine specchiata): controlla le etichette.").into()
                });
                or
            }
        }
    } else {
        unreachable!("a disk comes from the placement or from the limb")
    };
    out.orientation = Some(orientation);

    // Labels: what lies on the visible, lit side (or right on the terminator), big enough to see.
    let px_per_km = r / selene::RADIUS_KM;
    let mut craters = 0;
    for feat in selene::features() {
        let (x, y, front) = v.project(feat.lon, feat.lat);
        if !front {
            continue;
        }
        let alt = v.sun_altitude(feat.lon, feat.lat);
        let (dx, dy) = orientation.to_image(x * r, y * r);
        let (px, py) = (cx + dx, cy + dy);
        if px < 0.0 || py < 0.0 || px >= w as f64 || py >= h as f64 {
            continue;
        }
        // Foreshortening towards the limb.
        let z = (1.0 - x * x - y * y).max(0.0).sqrt();
        let size = feat.diameter * px_per_km;
        let (kind, show) = match feat.kind {
            FeatureKind::Mare => (LabelKind::Mare, alt > -10.0 && size > 40.0),
            FeatureKind::Mountains | FeatureKind::Other => (LabelKind::Feature, alt > -2.0 && size > 30.0),
            FeatureKind::Crater => (LabelKind::Crater, alt > -1.0 && size * z.max(0.3) > 10.0),
        };
        if !show {
            continue;
        }
        if kind == LabelKind::Crater {
            craters += 1;
            if craters > 70 {
                continue;
            }
        }
        let radius = if kind == LabelKind::Crater { (size / 2.0) as f32 } else { 0.0 };
        out.labels.push(Label { name: feat.name.to_string(), x: px as f32, y: py as f32, radius, kind, on_terminator: alt.abs() < 6.0 });
    }
}

/// Fraction of the disk (centre, radius) inside the w×h frame.
fn disk_fraction(cx: f64, cy: f64, r: f64, w: usize, h: usize) -> f64 {
    let (mut inside, mut all) = (0usize, 0usize);
    for j in 0..60 {
        for i in 0..60 {
            let (u, v) = ((i as f64 + 0.5) / 30.0 - 1.0, (j as f64 + 0.5) / 30.0 - 1.0);
            if u * u + v * v < 1.0 {
                all += 1;
                let (x, y) = (cx + u * r, cy + v * r);
                if x >= 0.0 && y >= 0.0 && x < w as f64 && y < h as f64 {
                    inside += 1;
                }
            }
        }
    }
    inside as f64 / all.max(1) as f64
}

/// Every rotation and both mirrors: the one whose predicted brightness (lit or not, plain or
/// highland) correlates best with the image. Returns it and its lead over the best mirrored one.
fn moon_orientation(small: &[f32], w: usize, h: usize, (cx, cy, r): (f64, f64, f64), v: &MoonView) -> (Orientation, f64) {
    // The model on a grid of the Moon's frame.
    const N: usize = 160;
    let areas = selene::dark_areas();
    let model: Vec<Option<f32>> = (0..N * N)
        .map(|k| {
            let (x, y) = ((k % N) as f64 / (N - 1) as f64 * 2.0 - 1.0, 1.0 - (k / N) as f64 / (N - 1) as f64 * 2.0);
            let zz = 1.0 - x * x - y * y;
            if zz <= 0.0 {
                return None;
            }
            let z = zz.sqrt();
            // Back from the disk to selenographic coordinates (inverse of `project`).
            let (b0, l0) = (v.earth_lat, v.earth_lon);
            let lat = (y * b0.cos() + z * b0.sin()).clamp(-1.0, 1.0).asin();
            let lon = l0 + x.atan2(z * b0.cos() - y * b0.sin());
            let (lat_d, lon_d) = (lat / RAD, lon / RAD);
            let lit = v.sun_altitude(lon_d, lat_d);
            let albedo = if selene::is_dark(lat_d, lon_d, &areas) { 0.55 } else { 1.0 };
            Some(if lit <= 0.0 { 0.0 } else { albedo * (lit * RAD).sin().max(0.0).powf(0.15) as f32 })
        })
        .collect();
    // Image samples inside the disk.
    let step = (r / 45.0).max(1.0);
    let mut samples: Vec<(f64, f64, f32)> = Vec::new();
    let mut y = cy - r;
    while y <= cy + r {
        let mut x = cx - r;
        while x <= cx + r {
            let (dx, dy) = (x - cx, y - cy);
            if dx * dx + dy * dy < (0.97 * r).powi(2) && x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h {
                samples.push((dx / r, dy / r, small[y as usize * w + x as usize]));
            }
            x += step;
        }
        y += step;
    }
    let corr = |or: Orientation| -> f64 {
        let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (dx, dy, val) in &samples {
            let (x, y) = or.body_offset(*dx, *dy);
            let (i, j) = (((x + 1.0) / 2.0 * (N - 1) as f64).round(), ((1.0 - y) / 2.0 * (N - 1) as f64).round());
            if !(0.0..N as f64).contains(&i) || !(0.0..N as f64).contains(&j) {
                continue;
            }
            let Some(m) = model[j as usize * N + i as usize] else { continue };
            let (a, b) = (m as f64, *val as f64);
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
            n += 1.0;
        }
        if n < 10.0 {
            return -1.0;
        }
        let cov = sab / n - sa * sb / n / n;
        let va = saa / n - (sa / n).powi(2);
        let vb = sbb / n - (sb / n).powi(2);
        if va <= 1e-12 || vb <= 1e-12 { -1.0 } else { cov / (va * vb).sqrt() }
    };
    let mut best = [(Orientation { angle: 0.0, mirrored: false, scale: 1.0 }, -2.0f64), (Orientation { angle: 0.0, mirrored: true, scale: 1.0 }, -2.0f64)];
    for (k, mirrored) in [false, true].into_iter().enumerate() {
        for a in 0..360 {
            let or = Orientation { angle: a as f32, mirrored, scale: 1.0 };
            let c = corr(or);
            if c > best[k].1 {
                best[k] = (or, c);
            }
        }
        // Refine to a tenth of a degree.
        let centre = best[k].0.angle;
        for d in -10..=10 {
            let or = Orientation { angle: (centre + d as f32 * 0.1).rem_euclid(360.0), mirrored, scale: 1.0 };
            let c = corr(or);
            if c > best[k].1 {
                best[k] = (or, c);
            }
        }
    }
    let (win, lose) = if best[0].1 >= best[1].1 { (best[0], best[1]) } else { (best[1], best[0]) };
    (win.0, win.1 - lose.1)
}

/// For close-ups: the features near the terminator today (where the relief shows best).
fn terminator_list(out: &mut Identification, v: &MoonView) {
    let mut near: Vec<(&str, f64)> = selene::features()
        .filter(|f| f.kind == FeatureKind::Crater && v.project(f.lon, f.lat).2)
        .map(|f| (f.name, v.sun_altitude(f.lon, f.lat)))
        .filter(|(_, alt)| (-1.0..8.0).contains(alt))
        .collect();
    near.sort_by(|a, b| a.1.total_cmp(&b.1));
    if near.is_empty() {
        return;
    }
    let names: Vec<&str> = near.iter().take(18).map(|n| n.0).collect();
    out.facts.push((tl("Crateri sul terminatore").into(), names.join(", ")));
}

// ---- the Sun -------------------------------------------------------------------------------

fn sun_facts(out: &mut Identification, when: &Moment) {
    let jd = when.julian();
    let Some(sun) = geocentric_au(Body::Sun, jd) else { return };
    let (lambda, _, _) = spherical(sun);
    let theta = ((jd - 2_398_220.0) * 360.0 / 25.38).rem_euclid(360.0) * RAD;
    let i = 7.25 * RAD;
    let k = (73.6667 + 1.3958333 * (jd - 2_396_758.0) / 36_525.0) * RAD;
    let eps = 23.4393 * RAD;
    let x = (-lambda.cos() * eps.tan()).atan();
    let y = (-(lambda - k).cos() * i.tan()).atan();
    let p = (x + y) / RAD;
    let b0 = ((lambda - k).sin() * i.sin()).asin() / RAD;
    let eta = (-(lambda - k).sin() * i.cos()).atan2(-(lambda - k).cos());
    let l0 = (eta - theta).rem_euclid(std::f64::consts::TAU) / RAD;
    let rotation = 1690.0 + ((jd - 2_444_235.34) / 27.2753).floor();
    out.facts.push((tl("Angolo di posizione dell'asse (P)").into(), format!("{p:+.1}°")));
    out.facts.push((tl("Latitudine del centro (B₀)").into(), format!("{b0:+.1}°")));
    out.facts.push((tl("Longitudine del centro (L₀)").into(), format!("{l0:.1}°")));
    out.facts.push((tl("Rotazione di Carrington").into(), format!("{rotation:.0}")));
    out.notes.push(tl("Per il Sole calcolo l'orientamento del disco; le macchie cambiano ogni giorno e non hanno nomi da riportare.").into());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two craters pointed at on a close-up give the disk's centre, size and turn back, mirrored or not.
    #[test]
    fn two_craters_place_the_disk() {
        let when = Moment { year: 2026, month: 8, day: 29, hour: 20, minute: 20, second: 0.0 };
        let v = selene::view(&when);
        for (angle, mirrored) in [(37.0f32, false), (250.0, true), (0.0, false)] {
            let truth = Placement { cx: 1500.0, cy: 700.0, r: 4200.0, orientation: Orientation { angle, mirrored, scale: 1.0 } };
            let pixel = |name: &str| {
                let f = selene::features().find(|f| f.name == name).unwrap();
                let (x, y, front) = v.project(f.lon, f.lat);
                assert!(front, "{name}");
                let (dx, dy) = truth.orientation.to_image(x * truth.r as f64, y * truth.r as f64);
                (truth.cx as f64 + dx, truth.cy as f64 + dy)
            };
            let got = place_on_craters(&when, ("Tycho", pixel("Tycho")), ("Clavius", pixel("Clavius")), mirrored).unwrap();
            assert!((got.r - truth.r).abs() < 0.5 && (got.cx - truth.cx).abs() < 0.5 && (got.cy - truth.cy).abs() < 0.5, "{got:?}");
            assert!(((got.orientation.angle - angle + 180.0).rem_euclid(360.0) - 180.0).abs() < 0.01, "{got:?}");
        }
        assert!(place_on_craters(&when, ("Tycho", (10.0, 10.0)), ("Tycho", (500.0, 20.0)), false).is_err());
        assert!(place_on_craters(&when, ("Tycho", (10.0, 10.0)), ("Clavius", (12.0, 10.0)), false).is_err());
    }

    #[test]
    fn orientation_round_trip() {
        for or in [Orientation { angle: 0.0, mirrored: false, scale: 1.0 }, Orientation { angle: 37.0, mirrored: true, scale: 1.0 }, Orientation { angle: 250.0, mirrored: false, scale: 1.0 }] {
            let (dx, dy) = or.to_image(0.3, -0.7);
            let (x, y) = or.body_offset(dx, dy);
            assert!((x - 0.3).abs() < 1e-9 && (y + 0.7).abs() < 1e-9);
        }
        // North up, east left: a body-frame north offset goes up the image (y decreases).
        let (dx, dy) = Orientation { angle: 0.0, mirrored: false, scale: 1.0 }.to_image(0.0, 1.0);
        assert!(dx.abs() < 1e-9 && dy < 0.0);
        assert!(Orientation { angle: 90.0, mirrored: false, scale: 1.0 }.north_up_angle().abs() - 90.0 < 1e-3);
    }

    /// A synthetic Jupiter with its moons where Meeus puts them, turned and mirrored: the
    /// search must find the turn and the mirror.
    #[test]
    fn finds_jupiters_moons_on_a_synthetic_image() {
        let when = Moment { year: 2024, month: 12, day: 7, hour: 21, minute: 0, second: 0.0 };
        let truth = Orientation { angle: 23.0, mirrored: true, scale: 1.0 };
        let (w, h, r) = (1500usize, 700usize, 25.0);
        let (cx, cy) = (750.0, 350.0);
        let mut plane = vec![0.02f32; w * h];
        let (s, c) = (truth.angle as f64 * RAD).sin_cos();
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                // Equator along the rotated x axis, poles flattened.
                let (u, v) = (c * dx + s * dy, -s * dx + c * dy);
                if (u / r).powi(2) + (v / (r * 0.935)).powi(2) < 1.0 {
                    plane[y * w + x] = 0.7;
                }
            }
        }
        let predicted = moons::positions(Body::Jupiter, &when);
        for m in predicted.iter().filter(|m| !m.occulted(0.935) && m.x.hypot(m.y) > 1.3) {
            let (dx, dy) = truth.to_image(m.x * r, m.y * r);
            let (mx, my) = (cx + dx, cy + dy);
            for y in 0..h {
                for x in 0..w {
                    let d2 = (x as f64 - mx).powi(2) + (y as f64 - my).powi(2);
                    plane[y * w + x] += (0.5 * (-d2 / 4.0).exp()) as f32;
                }
            }
        }
        let img = Image::new(w, h, vec![plane], 16);
        let id = identify(&img, Body::Jupiter, &when, None);
        let or = id.orientation.unwrap_or_else(|| panic!("{:?}", id.notes));
        assert!(id.orientation_found, "{:?}", id.notes);
        assert_eq!(or.mirrored, truth.mirrored, "{or:?}");
        let diff = (or.angle - truth.angle).rem_euclid(360.0);
        assert!(diff.min(360.0 - diff) < 2.0, "{or:?}");
        assert!(id.labels.iter().filter(|l| l.kind == LabelKind::MoonFound).count() >= 2, "{:?}", id.labels);
        // A wrong capture time (37 minutes late) is put right by the moons' positions.
        let late = Moment::from_julian(when.julian() + 37.0 / 1440.0);
        let fit = fit_time(&img, Body::Jupiter, &late, or, 180.0).expect("fit");
        assert!((fit.shift_min + 37.0).abs() < 10.0, "{fit:?}");
        assert!(fit.matched >= 2, "{fit:?}");
        // Moons stretched by 4 % and the starting turn 3° off: time, turn and stretch together.
        let stretched = {
            let mut p = vec![0.02f32; w * h];
            let (s, c) = (truth.angle as f64 * RAD).sin_cos();
            for y in 0..h {
                for x in 0..w {
                    let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                    let (u, v) = (c * dx + s * dy, -s * dx + c * dy);
                    if (u / r).powi(2) + (v / (r * 0.935)).powi(2) < 1.0 {
                        p[y * w + x] = 0.7;
                    }
                }
            }
            for m in predicted.iter().filter(|m| !m.occulted(0.935) && m.x.hypot(m.y) > 1.3) {
                let (dx, dy) = truth.to_image(m.x * r * 1.04, m.y * r * 1.04);
                let (mx, my) = (cx + dx, cy + dy);
                for y in 0..h {
                    for x in 0..w {
                        let d2 = (x as f64 - mx).powi(2) + (y as f64 - my).powi(2);
                        p[y * w + x] += (0.5 * (-d2 / 4.0).exp()) as f32;
                    }
                }
            }
            Image::new(w, h, vec![p], 16)
        };
        let start = Orientation { angle: truth.angle + 3.0, ..truth };
        let fit = fit_time(&stretched, Body::Jupiter, &when, start, 30.0).expect("fit");
        assert!(fit.shift_min.abs() < 5.0, "{fit:?}");
        let diff = (fit.orientation.angle - truth.angle).rem_euclid(360.0);
        assert!(diff.min(360.0 - diff) < 1.5, "{fit:?}");
        assert!((fit.orientation.scale - 1.04).abs() < 0.025, "{fit:?}");
        // The fourth moon sits near the planet, where the picture has no spot for it.
        assert!(fit.matched >= 3, "{fit:?}");
    }

    /// A synthetic lit Moon (lighting and plains from the model itself), turned: the limb fit
    /// and the orientation search find it.
    #[test]
    fn finds_the_moons_orientation_on_a_synthetic_disk() {
        let when = Moment { year: 2025, month: 3, day: 9, hour: 20, minute: 0, second: 0.0 };
        let v = selene::view(&when);
        let truth = Orientation { angle: 140.0, mirrored: false, scale: 1.0 };
        let (w, h, r) = (600usize, 560usize, 240.0);
        let (cx, cy) = (300.0, 290.0);
        let areas = selene::dark_areas();
        let mut plane = vec![0.01f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = ((x as f64 - cx) / r, (y as f64 - cy) / r);
                if dx * dx + dy * dy >= 1.0 {
                    continue;
                }
                let (bx, by) = truth.body_offset(dx, dy);
                let z = (1.0 - bx * bx - by * by).max(0.0).sqrt();
                let (b0, l0) = (v.earth_lat, v.earth_lon);
                let lat = (by * b0.cos() + z * b0.sin()).asin() / RAD;
                let lon = (l0 + bx.atan2(z * b0.cos() - by * b0.sin())) / RAD;
                let alt = v.sun_altitude(lon, lat);
                if alt > 0.0 {
                    plane[y * w + x] = if selene::is_dark(lat, lon, &areas) { 0.35 } else { 0.7 };
                }
            }
        }
        let img = Image::new(w, h, vec![plane], 16);
        let id = identify(&img, Body::Moon, &when, None);
        let (dcx, dcy, dr) = id.disk.expect("the limb");
        assert!((dcx - 300.0).abs() < 3.0 && (dcy - 290.0).abs() < 3.0 && (dr - 240.0).abs() < 3.0, "{:?}", id.disk);
        let or = id.orientation.unwrap();
        assert!(!or.mirrored, "{or:?} {:?}", id.notes);
        let diff = (or.angle - truth.angle).rem_euclid(360.0);
        assert!(diff.min(360.0 - diff) < 3.0, "{or:?}");
        assert!(id.labels.iter().any(|l| l.kind == LabelKind::Crater));
    }
}
