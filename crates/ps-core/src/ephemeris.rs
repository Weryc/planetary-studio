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

#[cfg(test)]
mod tests {
    use super::*;

    fn at(year: i32, month: u32, day: u32) -> Moment {
        Moment { year, month, day, hour: 0, minute: 0, second: 0.0 }
    }

    /// Published values: Jupiter at opposition 2022-09-26 49.9″, Mars at opposition 2018-07-27
    /// 24.3″, Saturn at opposition 2026 between 19 and 20.5″ (globe), Sun ~31.6′ in early January, Moon 29.4′–33.5′.
    #[test]
    fn known_diameters() {
        let close = |got: f64, want: f64, tol: f64| assert!((got - want).abs() < tol * want, "{got} vs {want}");
        close(apparent_diameter(Body::Jupiter, &at(2022, 9, 26)), 49.9, 0.01);
        close(apparent_diameter(Body::Mars, &at(2018, 7, 27)), 24.3, 0.015);
        assert!((19.0..20.5).contains(&apparent_diameter(Body::Saturn, &at(2026, 10, 4))));
        close(apparent_diameter(Body::Sun, &at(2026, 1, 3)), 32.53 * 60.0, 0.003);
        let moon = apparent_diameter(Body::Moon, &at(2026, 8, 29));
        assert!((29.3 * 60.0..33.6 * 60.0).contains(&moon), "{moon}");
    }
}
