//! The Moon as seen from the Earth at a moment: which face is turned towards us (libration),
//! where the Sun shines on it (phase, terminator), and the names of its main features.
//!
//! The Moon's position uses the largest terms of Meeus' series ("Astronomical Algorithms",
//! ch. 47; a few tenths of a degree), the Sun's the elements in [`crate::ephemeris`];
//! libration and the selenographic position of the Sun follow ch. 53 (optical libration only,
//! the physical part is under 0.1°).
//!
//! Features: the IAU names of the seas with approximate selenographic coordinates (about 1°,
//! ~30 km — enough to label a sea, not a crater).
//!
//! The Moon's own frame used here: x towards selenographic east (Mare Crisium's side), y
//! towards lunar north, in Moon radii, the disk centre at 0.

use crate::i18n::tl;
use crate::capture::{Body, Moment};
use crate::ephemeris::{geocentric_au, spherical};

const RAD: f64 = std::f64::consts::PI / 180.0;
/// Inclination of the lunar equator on the ecliptic.
const INCLINATION: f64 = 1.54242 * RAD;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoonView {
    /// Selenographic longitude and latitude of the point facing the Earth (libration), radians.
    pub earth_lon: f64,
    pub earth_lat: f64,
    /// Selenographic longitude and latitude of the point with the Sun overhead, radians.
    pub sun_lon: f64,
    pub sun_lat: f64,
    /// Illuminated fraction of the disk (0 = new, 1 = full).
    pub illuminated: f64,
    /// Selenographic colongitude of the Sun, degrees: the morning terminator lies at longitude
    /// −colongitude (0° at first quarter, 90° at full Moon, 180° at last quarter).
    pub colongitude: f64,
    /// Sun–Moon–Earth angle, degrees (0 = full).
    pub phase_angle: f64,
}

/// Geocentric ecliptic longitude, latitude (radians) and distance (km) of the Moon, and the mean
/// longitude of its ascending node and argument of latitude (radians), `t` in Julian centuries from J2000.
fn moon_position(t: f64) -> (f64, f64, f64, f64, f64) {
    let lp = (218.3164477 + 481_267.88123421 * t) * RAD;
    let d = (297.8501921 + 445_267.1114034 * t) * RAD;
    let m = (357.5291092 + 35_999.0502909 * t) * RAD;
    let mp = (134.9633964 + 477_198.8675055 * t) * RAD;
    let f = (93.2720950 + 483_202.0175233 * t) * RAD;
    let omega = (125.0445479 - 1934.1362891 * t) * RAD;
    // Longitude terms (10⁻⁶ degrees).
    let lon_terms: [(f64, [f64; 4]); 14] = [
        (6_288_774.0, [0.0, 0.0, 1.0, 0.0]),
        (1_274_027.0, [2.0, 0.0, -1.0, 0.0]),
        (658_314.0, [2.0, 0.0, 0.0, 0.0]),
        (213_618.0, [0.0, 0.0, 2.0, 0.0]),
        (-185_116.0, [0.0, 1.0, 0.0, 0.0]),
        (-114_332.0, [0.0, 0.0, 0.0, 2.0]),
        (58_793.0, [2.0, 0.0, -2.0, 0.0]),
        (57_066.0, [2.0, -1.0, -1.0, 0.0]),
        (53_322.0, [2.0, 0.0, 1.0, 0.0]),
        (45_758.0, [2.0, -1.0, 0.0, 0.0]),
        (-40_923.0, [0.0, 1.0, -1.0, 0.0]),
        (-34_720.0, [1.0, 0.0, 0.0, 0.0]),
        (-30_383.0, [0.0, 1.0, 1.0, 0.0]),
        (15_327.0, [2.0, 0.0, 0.0, -2.0]),
    ];
    let lat_terms: [(f64, [f64; 4]); 8] = [
        (5_128_122.0, [0.0, 0.0, 0.0, 1.0]),
        (280_602.0, [0.0, 0.0, 1.0, 1.0]),
        (277_693.0, [0.0, 0.0, 1.0, -1.0]),
        (173_237.0, [2.0, 0.0, 0.0, -1.0]),
        (55_413.0, [2.0, 0.0, -1.0, 1.0]),
        (46_271.0, [2.0, 0.0, -1.0, -1.0]),
        (32_573.0, [2.0, 0.0, 0.0, 1.0]),
        (17_198.0, [0.0, 0.0, 2.0, 1.0]),
    ];
    let arg = |c: [f64; 4]| c[0] * d + c[1] * m + c[2] * mp + c[3] * f;
    let lon = lp + lon_terms.iter().map(|(a, c)| a * arg(*c).sin()).sum::<f64>() * 1e-6 * RAD;
    let lat = lat_terms.iter().map(|(a, c)| a * arg(*c).sin()).sum::<f64>() * 1e-6 * RAD;
    let dist = moon_distance(t);
    (lon, lat, dist, omega, f)
}

fn moon_distance(t: f64) -> f64 {
    let d = (297.8501921 + 445_267.1114034 * t) * RAD;
    let m = (357.5291092 + 35_999.0502909 * t) * RAD;
    let mp = (134.9633964 + 477_198.8675055 * t) * RAD;
    385_000.56 - 20_905.355 * mp.cos() - 3_699.111 * (2.0 * d - mp).cos() - 2_955.968 * (2.0 * d).cos() - 569.925 * (2.0 * mp).cos() + 48.888 * m.cos()
}

/// Selenographic longitude and latitude (radians) of the point under a direction of ecliptic
/// longitude `lon`, latitude `lat` (as seen from the Moon's centre).
fn selenographic(lon: f64, lat: f64, omega: f64, f: f64) -> (f64, f64) {
    let w = lon - omega;
    let a = (w.sin() * lat.cos() * INCLINATION.cos() - lat.sin() * INCLINATION.sin()).atan2(w.cos() * lat.cos());
    let l = (a - f + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
    let b = (-w.sin() * lat.cos() * INCLINATION.sin() - lat.sin() * INCLINATION.cos()).asin();
    (l, b)
}

pub fn view(when: &Moment) -> MoonView {
    let jd = when.julian();
    let t = (jd - 2_451_545.0) / 36_525.0;
    let (lon, lat, dist, omega, f) = moon_position(t);
    let (earth_lon, earth_lat) = selenographic(lon, lat, omega, f);
    let sun = geocentric_au(Body::Sun, jd).expect("the Sun");
    let (sun_lon, _, sun_r) = spherical(sun);
    let sun_km = sun_r * 149_597_870.7;
    // The Sun seen from the Moon (Meeus 53): nearly opposite to the Earth's direction from it.
    let lh = sun_lon + std::f64::consts::PI + dist / sun_km * lat.cos() * (sun_lon - lon).sin();
    let bh = dist / sun_km * lat;
    let (s_lon, s_lat) = selenographic(lh, bh, omega, f);
    // Phase: elongation of the Moon from the Sun.
    let elong = (lat.cos() * (lon - sun_lon).cos()).acos();
    let phase_angle = (sun_km * elong.sin()).atan2(dist - sun_km * elong.cos());
    let illuminated = (1.0 + phase_angle.cos()) / 2.0;
    let colongitude = (90.0 - s_lon / RAD).rem_euclid(360.0);
    MoonView { earth_lon, earth_lat, sun_lon: s_lon, sun_lat: s_lat, illuminated, colongitude, phase_angle: phase_angle / RAD }
}

impl MoonView {
    /// Where a feature at (`lon`, `lat`, degrees, east and north positive) appears on the disk:
    /// (x, y) in the Moon's frame and whether it is on the visible side.
    pub fn project(&self, lon: f64, lat: f64) -> (f64, f64, bool) {
        let (lon, lat) = (lon * RAD, lat * RAD);
        let dl = lon - self.earth_lon;
        let x = lat.cos() * dl.sin();
        let y = self.earth_lat.cos() * lat.sin() - self.earth_lat.sin() * lat.cos() * dl.cos();
        let z = self.earth_lat.sin() * lat.sin() + self.earth_lat.cos() * lat.cos() * dl.cos();
        (x, y, z > 0.0)
    }

    /// Height of the Sun above the horizon at (`lon`, `lat`), degrees (negative = night).
    pub fn sun_altitude(&self, lon: f64, lat: f64) -> f64 {
        let (lon, lat) = (lon * RAD, lat * RAD);
        let c = lat.sin() * self.sun_lat.sin() + lat.cos() * self.sun_lat.cos() * (lon - self.sun_lon).cos();
        c.clamp(-1.0, 1.0).asin() / RAD
    }

    /// Phase name, in the interface language.
    pub fn phase_name(&self, waxing: bool) -> &'static str {
        let k = self.illuminated;
        match (k, waxing) {
            (k, _) if k < 0.03 => tl("Luna nuova"),
            (k, _) if k > 0.97 => tl("Luna piena"),
            (k, true) if k < 0.47 => tl("Falce crescente"),
            (k, true) if k < 0.53 => tl("Primo quarto"),
            (_, true) => tl("Gibbosa crescente"),
            (k, false) if k < 0.47 => tl("Falce calante"),
            (k, false) if k < 0.53 => tl("Ultimo quarto"),
            _ => tl("Gibbosa calante"),
        }
    }

    /// Waxing when the Sun shines on the Moon's eastern half seen from us (morning terminator).
    pub fn waxing(&self) -> bool {
        let (x, _, _) = self.project(self.sun_lon / RAD, self.sun_lat / RAD);
        // Selenographic east lit = Sun to the west of the Moon in the sky = evening sky = waxing.
        x > 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeatureKind {
    Mare,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feature {
    pub name: &'static str,
    /// Degrees, north positive.
    pub lat: f64,
    /// Degrees, east positive.
    pub lon: f64,
    /// Diameter, km.
    pub diameter: f64,
    pub kind: FeatureKind,
}

/// Moon radius, km.
pub const RADIUS_KM: f64 = 1737.4;

macro_rules! features {
    ($kind:ident: $(($name:expr, $lat:expr, $lon:expr, $d:expr)),* $(,)?) => { [$(Feature { name: $name, lat: $lat, lon: $lon, diameter: $d, kind: FeatureKind::$kind }),*] };
}

const MARIA: &[Feature] = &features!(Mare:
    ("Mare Imbrium", 32.8, -15.6, 1146.0), ("Mare Serenitatis", 28.0, 17.5, 707.0), ("Mare Tranquillitatis", 8.5, 31.4, 873.0),
    ("Mare Crisium", 17.0, 59.1, 556.0), ("Mare Fecunditatis", -7.8, 51.3, 840.0), ("Mare Nectaris", -15.2, 35.5, 333.0),
    ("Mare Nubium", -21.3, -16.6, 715.0), ("Mare Humorum", -24.4, -38.6, 389.0), ("Oceanus Procellarum", 18.4, -57.4, 1500.0),
    ("Mare Frigoris", 56.0, 1.4, 600.0), ("Mare Vaporum", 13.3, 3.6, 245.0), ("Mare Cognitum", -10.0, -23.1, 376.0),
    ("Mare Insularum", 7.5, -30.9, 513.0), ("Sinus Medii", 2.4, 1.7, 335.0), ("Sinus Iridum", 44.1, -31.5, 236.0),
    ("Mare Smythii", -1.3, 87.1, 373.0), ("Mare Marginis", 13.3, 86.1, 420.0), ("Mare Australe", -38.9, 93.0, 603.0),
    ("Mare Humboldtianum", 56.8, 81.5, 273.0), ("Mare Orientale", -19.4, -92.8, 327.0), ("Lacus Somniorum", 38.0, 29.2, 384.0),
    ("Palus Putredinis", 26.5, 0.4, 161.0),
);

/// Every named feature (the seas: the big plains are all the approximate catalogue can name).
pub fn features() -> impl Iterator<Item = &'static Feature> {
    MARIA.iter()
}

/// The dark plains, for matching the image's brightness pattern. The great elongated ones
/// are added as a few overlapping disks.
pub fn dark_areas() -> Vec<(f64, f64, f64)> {
    let mut v: Vec<(f64, f64, f64)> = MARIA.iter().filter(|m| m.name != "Oceanus Procellarum" && m.name != "Mare Frigoris").map(|m| (m.lat, m.lon, m.diameter / 2.0)).collect();
    v.extend([(10.0, -55.0, 450.0), (25.0, -50.0, 400.0), (-3.0, -40.0, 300.0), (35.0, -60.0, 250.0)]);
    v.extend([(56.0, -20.0, 150.0), (56.0, 0.0, 150.0), (57.0, 20.0, 150.0), (56.0, -35.0, 120.0)]);
    v.extend([(-5.2, -68.6, 70.0), (51.6, -9.4, 45.0)]);
    v
}

/// Whether (`lat`, `lon`) lies in a dark plain.
pub fn is_dark(lat: f64, lon: f64, areas: &[(f64, f64, f64)]) -> bool {
    areas.iter().any(|(alat, alon, r_km)| {
        let c = (lat * RAD).sin() * (alat * RAD).sin() + (lat * RAD).cos() * (alat * RAD).cos() * ((lon - alon) * RAD).cos();
        c.clamp(-1.0, 1.0).acos() * RADIUS_KM < *r_km
    })
}
