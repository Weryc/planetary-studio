//! The Moon as seen from the Earth at a moment: which face is turned towards us (libration),
//! where the Sun shines on it (phase, terminator), and the names of its main features.
//!
//! The Moon's position uses the largest terms of Meeus' series ("Astronomical Algorithms",
//! ch. 47; a few tenths of a degree), the Sun's the elements in [`crate::ephemeris`];
//! libration and the selenographic position of the Sun follow ch. 53 (optical libration only,
//! the physical part is under 0.1°).
//!
//! Features: IAU names with approximate selenographic coordinates (about 1°, ~30 km — enough
//! to label them, not to measure).
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
    Crater,
    Mare,
    Mountains,
    Other,
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

const CRATERS: &[Feature] = &features!(Crater:
    ("Copernico", 9.6, -20.1, 96.0), ("Tycho", -43.3, -11.4, 85.0), ("Keplero", 8.1, -38.0, 31.0), ("Aristarco", 23.7, -47.5, 40.0),
    ("Platone", 51.6, -9.4, 101.0), ("Clavius", -58.6, -14.7, 231.0), ("Tolomeo", -9.2, -1.8, 154.0), ("Alfonso", -13.4, -2.9, 111.0),
    ("Arzachel", -18.3, -1.9, 97.0), ("Albategnius", -11.2, 4.0, 129.0), ("Ipparco", -5.1, 5.2, 138.0), ("Teofilo", -11.4, 26.3, 99.0),
    ("Cirillo", -13.2, 24.0, 98.0), ("Caterina", -18.1, 23.4, 99.0), ("Posidonio", 31.9, 30.0, 95.0), ("Langrenus", -8.9, 61.0, 132.0),
    ("Petavius", -25.3, 60.4, 177.0), ("Vendelinus", -16.3, 61.6, 147.0), ("Eratostene", 14.5, -11.3, 58.0), ("Archimede", 29.7, -4.0, 81.0),
    ("Aristillo", 33.9, 1.2, 55.0), ("Autolico", 30.7, 1.5, 39.0), ("Aristotele", 50.2, 17.4, 88.0), ("Eudosso", 44.3, 16.3, 67.0),
    ("Atlante", 46.7, 44.4, 87.0), ("Ercole", 46.7, 39.1, 69.0), ("Endimione", 53.6, 56.5, 123.0), ("Cleomede", 27.7, 55.5, 125.0),
    ("Proclo", 16.1, 46.8, 27.0), ("Maurolico", -41.8, 14.0, 114.0), ("Stöfler", -41.1, 6.0, 126.0), ("Walther", -33.1, 0.7, 132.0),
    ("Deslandres", -32.5, -5.2, 234.0), ("Purbach", -25.5, -2.3, 115.0), ("Regiomontano", -28.4, -0.6, 108.0), ("Pitato", -29.9, -13.5, 106.0),
    ("Bullialdus", -20.7, -22.2, 61.0), ("Gassendi", -17.6, -40.0, 110.0), ("Schickard", -44.4, -54.6, 206.0), ("Wargentin", -49.6, -60.2, 84.0),
    ("Longomontano", -49.6, -21.7, 145.0), ("Maginus", -50.0, -6.2, 194.0), ("Moretus", -70.6, -5.5, 111.0), ("Grimaldi", -5.2, -68.6, 173.0),
    ("Riccioli", -3.0, -74.6, 139.0), ("Hevelius", 2.2, -67.6, 115.0), ("Reinhold", 3.3, -22.8, 42.0), ("Lansberg", -0.3, -26.6, 38.0),
    ("Encke", 4.6, -36.6, 28.0), ("Timocari", 26.7, -13.1, 33.0), ("Lambert", 25.8, -21.0, 30.0), ("Eulero", 23.3, -29.2, 27.0),
    ("Erodoto", 23.2, -49.7, 34.0), ("Pitagora", 63.5, -63.0, 145.0), ("J. Herschel", 62.0, -42.0, 165.0), ("Cassini", 40.2, 4.6, 57.0),
    ("Manilio", 14.5, 9.1, 38.0), ("Menelao", 16.3, 16.0, 27.0), ("Plinio", 15.4, 23.7, 43.0), ("Giulio Cesare", 9.0, 15.4, 90.0),
    ("Godin", 1.8, 10.2, 34.0), ("Agrippa", 4.1, 10.5, 44.0), ("Triesnecker", 4.2, 3.6, 26.0), ("Gutenberg", -8.6, 41.2, 74.0),
    ("Goclenio", -10.0, 45.0, 72.0), ("Macrobio", 21.3, 46.0, 63.0), ("Taruntius", 5.5, 46.5, 56.0), ("Furnerius", -36.0, 60.6, 135.0),
    ("Piccolomini", -29.7, 32.2, 88.0), ("Fracastoro", -21.5, 33.2, 112.0), ("Gauss", 35.7, 79.0, 177.0), ("Humboldt", -27.2, 80.9, 189.0),
    ("Bailly", -66.5, -69.1, 301.0), ("Schiller", -51.8, -39.0, 180.0), ("Hainzel", -41.3, -33.5, 70.0), ("Mersenius", -21.5, -49.2, 84.0),
    ("Letronne", -10.8, -42.5, 116.0), ("Flamsteed", -4.5, -44.3, 20.0), ("Fra Mauro", -6.1, -17.0, 101.0), ("Guericke", -11.5, -14.1, 63.0),
    ("Herschel", -5.7, -2.1, 40.0), ("Werner", -28.0, 3.3, 70.0), ("Aliacensis", -30.6, 5.2, 80.0), ("Rheita", -37.1, 47.2, 70.0),
    ("Janssen", -45.4, 40.3, 199.0), ("Metius", -40.3, 43.3, 87.0), ("Fabricius", -42.9, 42.0, 78.0), ("Stevinus", -32.5, 54.2, 74.0),
    ("Snellius", -29.3, 55.7, 82.0), ("Bürg", 45.0, 28.2, 40.0), ("Le Monnier", 26.6, 30.6, 61.0), ("Römer", 25.4, 36.4, 40.0),
    ("Mercator", -29.3, -26.1, 47.0), ("Campano", -28.0, -27.8, 48.0), ("Capuano", -34.1, -26.7, 60.0), ("Wilhelm", -43.1, -20.8, 107.0),
    ("Heinsius", -39.5, -17.7, 64.0), ("Hesiodo", -29.4, -16.3, 42.0), ("Abulfeda", -13.8, 13.9, 62.0), ("Delambre", -1.9, 17.5, 52.0),
    ("Messala", 39.2, 60.5, 125.0), ("Schiaparelli", 23.4, -58.8, 24.0), ("Mairan", 41.6, -43.4, 40.0),
);

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

const OTHERS: &[Feature] = &[
    Feature { name: "Montes Apenninus", lat: 18.9, lon: -3.7, diameter: 600.0, kind: FeatureKind::Mountains },
    Feature { name: "Montes Alpes", lat: 46.4, lon: -0.8, diameter: 280.0, kind: FeatureKind::Mountains },
    Feature { name: "Montes Caucasus", lat: 38.4, lon: 10.0, diameter: 445.0, kind: FeatureKind::Mountains },
    Feature { name: "Montes Carpatus", lat: 14.5, lon: -24.4, diameter: 361.0, kind: FeatureKind::Mountains },
    Feature { name: "Vallis Alpes", lat: 49.2, lon: 3.6, diameter: 155.0, kind: FeatureKind::Other },
    Feature { name: "Rupes Recta", lat: -22.1, lon: -7.8, diameter: 110.0, kind: FeatureKind::Other },
    Feature { name: "Vallis Schröteri", lat: 26.2, lon: -50.8, diameter: 168.0, kind: FeatureKind::Other },
    Feature { name: "Rima Hyginus", lat: 7.8, lon: 7.8, diameter: 220.0, kind: FeatureKind::Other },
];

/// Every named feature.
pub fn features() -> impl Iterator<Item = &'static Feature> {
    MARIA.iter().chain(OTHERS.iter()).chain(CRATERS.iter())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn at(year: i32, month: u32, day: u32, hour: u32) -> Moment {
        Moment { year, month, day, hour, minute: 0, second: 0.0 }
    }

    #[test]
    fn librations_stay_in_range_and_phases_are_right() {
        // Full Moon 2024-01-25 17:54 UT, new Moon 2024-02-09 22:59 UT, first quarter 2024-02-16 15:01 UT.
        let full = view(&at(2024, 1, 25, 18));
        assert!(full.illuminated > 0.99, "{}", full.illuminated);
        let new = view(&at(2024, 2, 9, 23));
        assert!(new.illuminated < 0.01, "{}", new.illuminated);
        let quarter = view(&at(2024, 2, 16, 15));
        assert!((quarter.illuminated - 0.5).abs() < 0.03, "{}", quarter.illuminated);
        assert!(quarter.waxing());
        // At first quarter the morning terminator runs near longitude 0: colongitude ≈ 0.
        let c = quarter.colongitude;
        assert!(!(15.0..345.0).contains(&c), "{c}");
        for k in 0..60 {
            let v = view(&at(2025, 1 + (k / 5) as u32, 1 + (k % 5) as u32 * 5, 0));
            assert!(v.earth_lon.abs() < 8.5 * RAD && v.earth_lat.abs() < 7.5 * RAD, "{v:?}");
        }
    }

    /// Meeus example 53.a: 1992 April 12, 0h TD: l = −1.206°, b = +4.194°.
    #[test]
    fn libration_matches_meeus() {
        let v = view(&at(1992, 4, 12, 0));
        assert!((v.earth_lon / RAD + 1.206).abs() < 0.3, "{}", v.earth_lon / RAD);
        assert!((v.earth_lat / RAD - 4.194).abs() < 0.3, "{}", v.earth_lat / RAD);
    }

    #[test]
    fn catalogue_is_sane() {
        assert!(features().all(|f| f.lat.abs() <= 90.0 && f.lon.abs() <= 180.0 && f.diameter > 0.0));
        let areas = dark_areas();
        assert!(is_dark(17.0, 59.0, &areas) && is_dark(32.0, -15.0, &areas));
        assert!(!is_dark(-43.3, -11.4, &areas), "Tycho is in the highlands");
    }
}
