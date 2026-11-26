//! Apparent diameters of the Sun, Moon and planets at a given moment: with the subject's size
//! in pixels, they give the image scale (arcsec per pixel) without any data from the user.
//!
//! Planets: JPL's Keplerian elements for 1800–2050 (Standish, "Approximate Positions of the
//! Planets"), geocentric distance from the two heliocentric positions; good to a fraction of
//! a percent in diameter. Moon: the largest terms of the ELP distance series (Meeus,
//! "Astronomical Algorithms", ch. 47), within ~0.1 %; the observer's position on the Earth
//! (up to ~1.7 % for the Moon) is left out.

use crate::capture::{Body, Moment};

/// Elements at J2000 and rates per century: a (au), e, I, L, ϖ, Ω (degrees).
type Elements = ([f64; 6], [f64; 6]);

const EARTH: Elements = ([1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0], [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0]);

fn elements(body: Body) -> Option<Elements> {
    Some(match body {
        Body::Mercury => ([0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593], [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081]),
        Body::Venus => ([0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255], [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418]),
        Body::Mars => ([1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891], [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343]),
        Body::Jupiter => ([5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909], [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106]),
        Body::Saturn => ([9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448], [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794]),
        Body::Uranus => ([19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503], [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589]),
        Body::Neptune => ([30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574], [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664]),
        Body::Sun | Body::Moon => return None,
    })
}

/// Equatorial diameter, km (Saturn: the globe, without the rings).
pub fn diameter_km(body: Body) -> f64 {
    match body {
        Body::Sun => 1_392_700.0,
        Body::Moon => 3_474.8,
        Body::Mercury => 4_879.4,
        Body::Venus => 12_103.6,
        Body::Mars => 6_792.4,
        Body::Jupiter => 142_984.0,
        Body::Saturn => 120_536.0,
        Body::Uranus => 51_118.0,
        Body::Neptune => 49_528.0,
    }
}

const AU_KM: f64 = 149_597_870.7;

/// Heliocentric ecliptic position (au) from Keplerian elements, `t` in centuries from J2000.
fn heliocentric(e: Elements, t: f64) -> [f64; 3] {
    let v: Vec<f64> = (0..6).map(|i| e.0[i] + e.1[i] * t).collect();
    let (a, ecc, inc, l, peri, node) = (v[0], v[1], v[2].to_radians(), v[3], v[4], v[5]);
    let omega = (peri - node).to_radians();
    let m = (l - peri).rem_euclid(360.0).to_radians();
    // Kepler's equation by Newton's method.
    let mut big_e = m + ecc * m.sin();
    for _ in 0..10 {
        big_e -= (big_e - ecc * big_e.sin() - m) / (1.0 - ecc * big_e.cos());
    }
    let (x, y) = (a * (big_e.cos() - ecc), a * (1.0 - ecc * ecc).sqrt() * big_e.sin());
    let (co, so, cn, sn, ci, si) = (omega.cos(), omega.sin(), node.to_radians().cos(), node.to_radians().sin(), inc.cos(), inc.sin());
    [(co * cn - so * sn * ci) * x + (-so * cn - co * sn * ci) * y, (co * sn + so * cn * ci) * x + (-so * sn + co * cn * ci) * y, (so * si) * x + (co * si) * y]
}

/// Distance from the Earth's centre, km.
pub fn distance_km(body: Body, when: &Moment) -> f64 {
    let jd = when.julian();
    let t = (jd - 2_451_545.0) / 36_525.0;
    match body {
        Body::Moon => moon_distance_km(t),
        Body::Sun => {
            let p = heliocentric(EARTH, t);
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() * AU_KM
        }
        _ => {
            let (p, earth) = (heliocentric(elements(body).expect("a planet"), t), heliocentric(EARTH, t));
            let d = [p[0] - earth[0], p[1] - earth[1], p[2] - earth[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() * AU_KM
        }
    }
}

/// Geocentric ecliptic position (J2000 ecliptic and equinox, au) of the Sun or a planet at
/// Julian date `jd`; for a planet, as seen (corrected for the light's travel time).
pub fn geocentric_au(body: Body, jd: f64) -> Option<[f64; 3]> {
    let t = |jd: f64| (jd - 2_451_545.0) / 36_525.0;
    let earth = heliocentric(EARTH, t(jd));
    if body == Body::Sun {
        return Some(earth.map(|v| -v));
    }
    let el = elements(body)?;
    let mut delay = 0.0;
    let mut d = [0.0; 3];
    for _ in 0..3 {
        let p = heliocentric(el, t(jd - delay));
        d = [p[0] - earth[0], p[1] - earth[1], p[2] - earth[2]];
        delay = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() / 173.144_6;
    }
    Some(d)
}

/// (longitude, latitude, distance): ecliptic spherical coordinates of `v`, radians and `v`'s units.
pub fn spherical(v: [f64; 3]) -> (f64, f64, f64) {
    let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (v[1].atan2(v[0]).rem_euclid(std::f64::consts::TAU), (v[2] / r).asin(), r)
}

fn moon_distance_km(t: f64) -> f64 {
    let d = (297.8501921 + 445_267.1114034 * t).to_radians();
    let m = (357.5291092 + 35_999.0502909 * t).to_radians();
    let mp = (134.9633964 + 477_198.8675055 * t).to_radians();
    let f = (93.2720950 + 483_202.0175233 * t).to_radians();
    let terms: [(f64, f64); 13] = [
        (-20_905.355, mp),
        (-3_699.111, 2.0 * d - mp),
        (-2_955.968, 2.0 * d),
        (-569.925, 2.0 * mp),
        (48.888, m),
        (-3.149, 2.0 * f),
        (246.158, 2.0 * d - 2.0 * mp),
        (-152.138, 2.0 * d - m - mp),
        (-170.733, 2.0 * d + mp),
        (-204.586, 2.0 * d - m),
        (-129.620, m - mp),
        (108.743, d),
        (104.755, m + mp),
    ];
    385_000.56 + terms.iter().map(|(c, arg)| c * arg.cos()).sum::<f64>()
}

/// Apparent equatorial diameter, arcsec.
pub fn apparent_diameter(body: Body, when: &Moment) -> f64 {
    let d = distance_km(body, when);
    2.0 * (diameter_km(body) / 2.0 / d).atan() * 180.0 / std::f64::consts::PI * 3600.0
}

/// The planet's north pole (IAU, J2000 equator): right ascension and declination, degrees.
fn pole(body: Body) -> Option<(f64, f64)> {
    match body {
        Body::Mars => Some((317.681, 52.887)),
        Body::Jupiter => Some((268.057, 64.495)),
        Body::Saturn => Some((40.589, 83.537)),
        _ => None,
    }
}

/// Unit vector of the planet's north pole on the J2000 ecliptic.
fn pole_ecliptic(body: Body) -> Option<[f64; 3]> {
    let (ra, dec) = pole(body)?;
    let (ra, dec, eps) = (ra.to_radians(), dec.to_radians(), 23.439_291_1f64.to_radians());
    let (x, y, z) = (dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
    Some([x, y * eps.cos() + z * eps.sin(), -y * eps.sin() + z * eps.cos()])
}

/// Latitudine planetocentrica del centro del disco (la Terra vista dal pianeta), gradi:
/// positiva quando il polo nord è inclinato verso di noi. Marte, Giove, Saturno (per Saturno è
/// anche l'inclinazione degli anelli).
pub fn sub_earth_latitude(body: Body, when: &Moment) -> Option<f64> {
    let p = pole_ecliptic(body)?;
    let d = geocentric_au(body, when.julian())?;
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    Some((-(p[0] * d[0] + p[1] * d[1] + p[2] * d[2]) / n).asin().to_degrees())
}

/// Angolo di fase (Sole–pianeta–Terra), gradi: quanto del disco è in ombra (Marte fino a ~47°).
pub fn phase_angle(body: Body, when: &Moment) -> Option<f64> {
    let jd = when.julian();
    let d = geocentric_au(body, jd)?;
    let sun = geocentric_au(Body::Sun, jd)?;
    // Planet → Earth is −d, planet → Sun is sun − d.
    let (a, b) = ([-d[0], -d[1], -d[2]], [sun[0] - d[0], sun[1] - d[1], sun[2] - d[2]]);
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    Some((dot / (na * nb)).clamp(-1.0, 1.0).acos().to_degrees())
}
