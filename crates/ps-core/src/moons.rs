//! Where the moons of Jupiter and Saturn are, as seen from the Earth at a given moment.
//!
//! Jupiter: the four Galilean moons with the low-accuracy method of Meeus, "Astronomical
//! Algorithms" ch. 44 (a few hundredths of a Jupiter radius), plus the longitudes of the
//! central meridian in Systems I and II (ch. 43). Saturn: the eight major moons, ch. 46, a port
//! of Sonia Keys' `meeus/saturnmoons` (MIT licence), with Saturn's position from
//! [`crate::ephemeris`] instead of VSOP87 — a difference far below a pixel.
//!
//! Coordinates are in the planet's equatorial radii, measured from its centre on the sky: X
//! along the planet's equator, positive towards the sky's west; Y along its axis, positive
//! towards the planet's north pole.

use crate::i18n::tl;
use crate::capture::{Body, Moment};
use crate::ephemeris::{geocentric_au, spherical};

#[derive(Clone, Debug, PartialEq)]
pub struct MoonPosition {
    pub name: &'static str,
    pub x: f64,
    pub y: f64,
    /// Farther than the planet (passes behind it rather than in front).
    pub behind: bool,
}

impl MoonPosition {
    /// Hidden by the planet's globe (`flattening`: polar/equatorial radius).
    pub fn occulted(&self, flattening: f64) -> bool {
        self.behind && self.x * self.x + (self.y / flattening).powi(2) < 1.0
    }

    /// In front of the globe (its shadow or itself crosses the disk).
    pub fn in_transit(&self, flattening: f64) -> bool {
        !self.behind && self.x * self.x + (self.y / flattening).powi(2) < 1.0
    }
}

/// Polar over equatorial radius.
pub fn flattening(body: Body) -> f64 {
    match body {
        Body::Jupiter => 0.935,
        Body::Saturn => 0.902,
        _ => 1.0,
    }
}

/// The moons of `body` worth looking for (empty for bodies without bright moons).
pub fn positions(body: Body, when: &Moment) -> Vec<MoonPosition> {
    let jd = when.julian();
    match body {
        Body::Jupiter => galilean(jd).0.to_vec(),
        Body::Saturn => saturnian(jd),
        _ => Vec::new(),
    }
}

/// Longitudes of Jupiter's central meridian, Systems I and II, degrees.
pub fn jupiter_central_meridian(when: &Moment) -> (f64, f64) {
    let (_, cm) = galilean(when.julian());
    cm
}

const RAD: f64 = std::f64::consts::PI / 180.0;

fn galilean(jd: f64) -> ([MoonPosition; 4], (f64, f64)) {
    let d = jd - 2_451_545.0;
    let v = (172.74 + 0.00111588 * d) * RAD;
    let m = (357.529 + 0.9856003 * d) * RAD;
    let n = (20.02 + 0.0830853 * d + 0.329 * v.sin()) * RAD;
    let j = (66.115 + 0.9025179 * d - 0.329 * v.sin()) * RAD;
    let a = (1.915 * m.sin() + 0.02 * (2.0 * m).sin()) * RAD;
    let b = (5.555 * n.sin() + 0.168 * (2.0 * n).sin()) * RAD;
    let k = j + a - b;
    let r_earth = 1.00014 - 0.01671 * m.cos() - 0.00014 * (2.0 * m).cos();
    let r = 5.20872 - 0.25208 * n.cos() - 0.00611 * (2.0 * n).cos();
    let delta = (r * r + r_earth * r_earth - 2.0 * r * r_earth * k.cos()).sqrt();
    let psi = (r_earth / delta * k.sin()).asin();
    let lambda = (34.35 + 0.083091 * d + 0.329 * v.sin()) * RAD + b;
    let ds = 3.12 * RAD * (lambda + 42.8 * RAD).sin();
    let de = ds - 2.22 * RAD * psi.sin() * (lambda + 22.0 * RAD).cos() - 1.3 * RAD * (r - delta) / delta * (lambda - 100.5 * RAD).sin();
    let dd = d - delta / 173.0;
    let cm1 = (210.98 + 877.8169088 * dd + psi / RAD - b / RAD).rem_euclid(360.0);
    let cm2 = (187.23 + 870.1869088 * dd + psi / RAD - b / RAD).rem_euclid(360.0);
    let u1 = (163.8069 + 203.4058646 * dd) * RAD + psi - b;
    let u2 = (358.414 + 101.2916335 * dd) * RAD + psi - b;
    let u3 = (5.7176 + 50.234518 * dd) * RAD + psi - b;
    let u4 = (224.8092 + 21.48798 * dd) * RAD + psi - b;
    let g = (331.18 + 50.310482 * dd) * RAD;
    let h = (87.45 + 21.569231 * dd) * RAD;
    let (s12, c12) = (2.0 * (u1 - u2)).sin_cos();
    let (s23, c23) = (2.0 * (u2 - u3)).sin_cos();
    let moon = |name, u: f64, r: f64| {
        // u is counted from the inferior conjunction: between 90° and 270° the moon is beyond Jupiter.
        MoonPosition { name, x: r * u.sin(), y: -r * u.cos() * de.sin(), behind: u.cos() < 0.0 }
    };
    (
        [
            moon("Io", u1 + 0.473 * RAD * s12, 5.9057 - 0.0244 * c12),
            moon("Europa", u2 + 1.065 * RAD * s23, 9.3966 - 0.0882 * c23),
            moon(tl("Ganimede"), u3 + 0.165 * RAD * g.sin(), 14.9883 - 0.0216 * g.cos()),
            moon("Callisto", u4 + 0.843 * RAD * h.sin(), 26.3627 - 0.1939 * h.cos()),
        ],
        (cm1, cm2),
    )
}

// ---- Saturn (Meeus ch. 46) ---------------------------------------------------------------

struct Qs {
    t1: f64,
    t2: f64,
    t4: f64,
    t6: f64,
    t7: f64,
    t8: f64,
    t9: f64,
    t10: f64,
    t11: f64,
    w3: f64,
    w4: f64,
    w5: f64,
    w6: f64,
    w7: f64,
    w8: f64,
    w0: f64,
    w1: f64,
    w2: f64,
    s1: f64,
    c1: f64,
    e1: f64,
}

/// Longitude, radius vector, inclination and node of one moon.
struct R4 {
    lambda: f64,
    r: f64,
    gamma: f64,
    omega: f64,
}

impl Qs {
    fn new(jde: f64) -> Qs {
        let t1 = jde - 2_411_093.0;
        let t2 = t1 / 365.25;
        let t3 = (jde - 2_433_282.423) / 365.25 + 1950.0;
        let t4 = jde - 2_411_368.0;
        let t5 = t4 / 365.25;
        let t6 = jde - 2_415_020.0;
        let t7 = t6 / 36_525.0;
        let t10 = jde - 2_409_786.0;
        let (s1, c1) = (28.0817 * RAD).sin_cos();
        Qs {
            t1,
            t2,
            t4,
            t6,
            t7,
            t8: t6 / 365.25,
            t9: (jde - 2_442_000.5) / 365.25,
            t10,
            t11: t10 / 36_525.0,
            w0: 5.095 * RAD * (t3 - 1866.39),
            w1: (74.4 + 32.39 * t2) * RAD,
            w2: (134.3 + 92.62 * t2) * RAD,
            w3: (42.0 - 0.5118 * t5) * RAD,
            w4: (276.59 + 0.5118 * t5) * RAD,
            w5: (267.2635 + 1222.1136 * t7) * RAD,
            w6: (175.4762 + 1221.5515 * t7) * RAD,
            w7: (2.4891 + 0.002435 * t7) * RAD,
            w8: (113.35 - 0.2597 * t7) * RAD,
            s1,
            c1,
            e1: 0.05589 - 0.000346 * t7,
        }
    }

    fn mimas(&self) -> R4 {
        let w0 = self.w0;
        let l = (127.64 + 381.994497 * self.t1 - 43.57 * w0.sin() - 0.72 * (3.0 * w0).sin() - 0.02144 * (5.0 * w0).sin()) * RAD;
        let p = (106.1 + 365.549 * self.t2) * RAD;
        let m = l - p;
        let c = (2.18287 * m.sin() + 0.025988 * (2.0 * m).sin() + 0.00043 * (3.0 * m).sin()) * RAD;
        R4 { lambda: l + c, r: 3.06879 / (1.0 + 0.01905 * (m + c).cos()), gamma: 1.563 * RAD, omega: (54.5 - 365.072 * self.t2) * RAD }
    }

    fn enceladus(&self) -> R4 {
        let l = (200.317 + 262.7319002 * self.t1 + 0.25667 * self.w1.sin() + 0.20883 * self.w2.sin()) * RAD;
        let p = (309.107 + 123.44121 * self.t2) * RAD;
        let m = l - p;
        let c = (0.55577 * m.sin() + 0.00168 * (2.0 * m).sin()) * RAD;
        R4 { lambda: l + c, r: 3.94118 / (1.0 + 0.00485 * (m + c).cos()), gamma: 0.0262 * RAD, omega: (348.0 - 151.95 * self.t2) * RAD }
    }

    fn tethys(&self) -> R4 {
        let w0 = self.w0;
        let lambda = (285.306 + 190.69791226 * self.t1 + 2.063 * w0.sin() + 0.03409 * (3.0 * w0).sin() + 0.001015 * (5.0 * w0).sin()) * RAD;
        R4 { lambda, r: 4.880998, gamma: 1.0976 * RAD, omega: (111.33 - 72.2441 * self.t2) * RAD }
    }

    fn dione(&self) -> R4 {
        let l = (254.712 + 131.53493193 * self.t1 - 0.0215 * self.w1.sin() - 0.01733 * self.w2.sin()) * RAD;
        let p = (174.8 + 30.82 * self.t2) * RAD;
        let m = l - p;
        let c = (0.24717 * m.sin() + 0.00033 * (2.0 * m).sin()) * RAD;
        R4 { lambda: l + c, r: 6.24871 / (1.0 + 0.002157 * (m + c).cos()), gamma: 0.0139 * RAD, omega: (232.0 - 30.27 * self.t2) * RAD }
    }

    fn rhea(&self) -> R4 {
        let pp = (342.7 + 10.057 * self.t2) * RAD;
        let a1 = 0.000265 * pp.sin() + 0.001 * self.w4.sin();
        let a2 = 0.000265 * pp.cos() + 0.001 * self.w4.cos();
        let e = a1.hypot(a2);
        let p = a1.atan2(a2);
        let n = (345.0 - 10.057 * self.t2) * RAD;
        let lambda = (359.244 + 79.6900472 * self.t1 + 0.086754 * n.sin()) * RAD;
        let i = (28.0362 + 0.346898 * n.cos() + 0.0193 * self.w3.cos()) * RAD;
        let omega = (168.8034 + 0.736936 * n.sin() + 0.041 * self.w3.sin()) * RAD;
        self.subr(lambda, p, e, 8.725924, omega, i)
    }

    fn subr(&self, lambda: f64, p: f64, e: f64, a: f64, omega: f64, i: f64) -> R4 {
        let m = lambda - p;
        let (e2, e3) = (e * e, e * e * e);
        let (e4, e5) = (e2 * e2, e3 * e2);
        let c = (2.0 * e - 0.25 * e3 + 0.0520833333 * e5) * m.sin()
            + (1.25 * e2 - 0.458333333 * e4) * (2.0 * m).sin()
            + (1.083333333 * e3 - 0.671875 * e5) * (3.0 * m).sin()
            + 1.072917 * e4 * (4.0 * m).sin()
            + 1.142708 * e5 * (5.0 * m).sin();
        let r = a * (1.0 - e2) / (1.0 + e * (m + c).cos());
        let g = omega - 168.8112 * RAD;
        let (si, ci) = i.sin_cos();
        let (sg, cg) = g.sin_cos();
        let a1 = si * sg;
        let a2 = self.c1 * si * cg - self.s1 * ci;
        let gamma = a1.hypot(a2).asin();
        let u = a1.atan2(a2);
        let h = self.c1 * si - self.s1 * ci * cg;
        let psi = (self.s1 * sg).atan2(h);
        R4 { lambda: lambda + c + u - g - psi, r, gamma, omega: 168.8112 * RAD + u }
    }

    fn titan(&self) -> R4 {
        let l = (261.1582 + 22.57697855 * self.t4 + 0.074025 * self.w3.sin()) * RAD;
        let ip = (27.45141 + 0.295999 * self.w3.cos()) * RAD;
        let op = (168.66925 + 0.628808 * self.w3.sin()) * RAD;
        let (sip, cip) = ip.sin_cos();
        let (so, co) = (op - self.w8).sin_cos();
        let a1 = self.w7.sin() * so;
        let a2 = self.w7.cos() * sip - self.w7.sin() * cip * co;
        let g0 = 102.8623 * RAD;
        let psi = a1.atan2(a2);
        let s = a1.hypot(a2);
        let mut g = self.w4 - op - psi;
        let mut varpi = 0.0;
        for _ in 0..3 {
            varpi = self.w4 + 0.37515 * RAD * ((2.0 * g).sin() - (2.0 * g0).sin());
            g = varpi - op - psi;
        }
        let ep = 0.029092 + 0.00019048 * ((2.0 * g).cos() - (2.0 * g0).cos());
        let qq = 2.0 * (self.w5 - varpi);
        let b1 = sip * so;
        let b2 = self.w7.cos() * sip * co - self.w7.sin() * cip;
        let theta = b1.atan2(b2) + self.w8;
        let e = ep + 0.002778797 * ep * qq.cos();
        let p = varpi + 0.159215 * RAD * qq.sin();
        let u = 2.0 * self.w5 - 2.0 * theta + psi;
        let h = 0.9375 * ep * ep * qq.sin() + 0.1875 * s * s * (2.0 * (self.w5 - theta)).sin();
        let lambda = l - 0.254744 * RAD * (self.e1 * self.w6.sin() + 0.75 * self.e1 * self.e1 * (2.0 * self.w6).sin() + h);
        let i = ip + 0.031843 * RAD * s * u.cos();
        let omega = op + 0.031843 * RAD * s * u.sin() / sip;
        self.subr(lambda, p, e, 20.216193, omega, i)
    }

    fn hyperion(&self) -> R4 {
        let eta = (92.39 + 0.5621071 * self.t6) * RAD;
        let zeta = (148.19 - 19.18 * self.t8) * RAD;
        let theta = (184.8 - 35.41 * self.t9) * RAD;
        let theta_p = theta - 7.5 * RAD;
        let a_s = (176.0 + 12.22 * self.t8) * RAD;
        let b_s = (8.0 + 24.44 * self.t8) * RAD;
        let c_s = b_s + 5.0 * RAD;
        let varpi = (69.898 - 18.67088 * self.t8) * RAD;
        let phi = 2.0 * (varpi - self.w5);
        let chi = (94.9 - 2.292 * self.t8) * RAD;
        let a = 24.50601 - 0.08686 * eta.cos() - 0.00166 * (zeta + eta).cos() + 0.00175 * (zeta - eta).cos();
        let e = 0.103458 - 0.004099 * eta.cos() - 0.000167 * (zeta + eta).cos() + 0.000235 * (zeta - eta).cos() + 0.02303 * zeta.cos() - 0.00212 * (2.0 * zeta).cos()
            + 0.000151 * (3.0 * zeta).cos()
            + 0.00013 * phi.cos();
        let p = varpi
            + (0.15648 * chi.sin() - 0.4457 * eta.sin() - 0.2657 * (zeta + eta).sin() - 0.3573 * (zeta - eta).sin() - 12.872 * zeta.sin() + 1.668 * (2.0 * zeta).sin()
                - 0.2419 * (3.0 * zeta).sin()
                - 0.07 * phi.sin())
                * RAD;
        let lambda = (177.047 + 16.91993829 * self.t6) * RAD
            + (0.15648 * chi.sin() + 9.142 * eta.sin() + 0.007 * (2.0 * eta).sin() - 0.014 * (3.0 * eta).sin() + 0.2275 * (zeta + eta).sin() + 0.2112 * (zeta - eta).sin()
                - 0.26 * zeta.sin()
                - 0.0098 * (2.0 * zeta).sin()
                - 0.013 * a_s.sin()
                + 0.017 * b_s.sin()
                - 0.0303 * phi.sin())
                * RAD;
        let i = (27.3347 + 0.6434886 * chi.cos() + 0.315 * self.w3.cos() + 0.018 * theta.cos() - 0.018 * c_s.cos()) * RAD;
        let omega = (168.6812 + 1.40136 * chi.cos() + 0.68599 * self.w3.sin() - 0.0392 * c_s.sin() + 0.0366 * theta_p.sin()) * RAD;
        self.subr(lambda, p, e, a, omega, i)
    }

    fn iapetus(&self) -> R4 {
        let horner = |x: f64, c: [f64; 4]| ((c[3] * x + c[2]) * x + c[1]) * x + c[0];
        let l = (261.1582 + 22.57697855 * self.t4) * RAD;
        let varpi_p = (91.796 + 0.562 * self.t7) * RAD;
        let psi = (4.367 - 0.195 * self.t7) * RAD;
        let theta = (146.819 - 3.198 * self.t7) * RAD;
        let phi = (60.47 + 1.521 * self.t7) * RAD;
        let big_phi = (205.055 - 2.091 * self.t7) * RAD;
        let ep = 0.028298 + 0.001156 * self.t11;
        let varpi0 = (352.91 + 11.71 * self.t11) * RAD;
        let mu = (76.3852 + 4.53795125 * self.t10) * RAD;
        let ip = horner(self.t11, [18.4602, -0.9518, -0.072, 0.0054]) * RAD;
        let op = horner(self.t11, [143.198, -3.919, 0.116, 0.008]) * RAD;
        let ll = mu - varpi0;
        let g = varpi0 - op - psi;
        let g1 = varpi0 - op - phi;
        let ls = self.w5 - varpi_p;
        let gs = varpi_p - theta;
        let lt = l - self.w4;
        let gt = self.w4 - big_phi;
        let u1 = 2.0 * (ll + g - ls - gs);
        let u2 = ll + g1 - lt - gt;
        let u3 = ll + 2.0 * (g - ls - gs);
        let u4 = lt + gt - g1;
        let u5 = 2.0 * (ls + gs);
        let a = 58.935028 + 0.004638 * u1.cos() + 0.058222 * u2.cos();
        let e = ep - 0.0014097 * (g1 - gt).cos() + 0.0003733 * (u5 - 2.0 * g).cos() + 0.000118 * u3.cos() + 0.0002408 * ll.cos() + 0.0002849 * (ll + u2).cos() + 0.000619 * u4.cos();
        let w = (0.08077 * (g1 - gt).sin() + 0.02139 * (u5 - 2.0 * g).sin() - 0.00676 * u3.sin() + 0.0138 * ll.sin() + 0.01632 * (ll + u2).sin() + 0.03547 * u4.sin()) * RAD;
        let p = varpi0 + w / ep;
        let lambda =
            mu + (-0.04299 * u2.sin() - 0.00789 * u1.sin() - 0.06312 * ls.sin() - 0.00295 * (2.0 * ls).sin() - 0.02231 * u5.sin() + 0.0065 * (u5 + psi).sin()) * RAD;
        let s5 = ll + g1 + lt + gt + phi;
        let i = ip + (0.04204 * (u5 + psi).cos() + 0.00235 * s5.cos() + 0.0036 * (u2 + phi).cos()) * RAD;
        let wp = (0.04204 * (u5 + psi).sin() + 0.00235 * s5.sin() + 0.00358 * (u2 + phi).sin()) * RAD;
        let omega = op + wp / ip.sin();
        self.subr(lambda, p, e, a, omega, i)
    }
}

const SATURN_MOONS: [&str; 8] = ["Mimas", "Encelado", "Teti", "Dione", "Rea", "Titano", "Iperione", "Giapeto"];
const K: [f64; 9] = [0.0, 20947.0, 23715.0, 26382.0, 29876.0, 35313.0, 53800.0, 59222.0, 91820.0];

fn saturnian(jd: f64) -> Vec<MoonPosition> {
    let Some(geo) = geocentric_au(Body::Saturn, jd) else { return Vec::new() };
    let (lon, lat, delta) = spherical(geo);
    let jde = jd - delta / 173.1446;
    // Meeus' constants refer to the B1950 ecliptic; the position is on J2000's: undo the
    // general precession between the two.
    let years = (2_451_545.0 - 2_433_282.423) / 365.25;
    let lambda0 = lon - 5029.0966 / 3600.0 * RAD * years / 100.0;
    let beta0 = lat;
    let q = Qs::new(jde);
    let s4 = [q.mimas(), q.enceladus(), q.tethys(), q.dione(), q.rhea(), q.titan(), q.hyperion(), q.iapetus()];
    let (s2, c2) = (168.8112 * RAD).sin_cos();
    let mut big_x = [0.0f64; 9];
    let mut big_y = [0.0f64; 9];
    let mut big_z = [0.0f64; 9];
    for (j, m) in s4.iter().enumerate() {
        let u = m.lambda - m.omega;
        let w = m.omega - 168.8112 * RAD;
        let (su, cu) = u.sin_cos();
        let (sw, cw) = w.sin_cos();
        let (sg, cg) = m.gamma.sin_cos();
        big_x[j + 1] = m.r * (cu * cw - su * cg * sw);
        big_y[j + 1] = m.r * (su * cw * cg + cu * sw);
        big_z[j + 1] = m.r * su * sg;
    }
    big_z[0] = 1.0;
    let (sl, cl) = lambda0.sin_cos();
    let (sb, cb) = beta0.sin_cos();
    let mut a_ = [0.0f64; 9];
    let mut b_ = [0.0f64; 9];
    let mut c_ = [0.0f64; 9];
    for j in 0..9 {
        let a = big_x[j];
        let b = q.c1 * big_y[j] - q.s1 * big_z[j];
        let c = q.s1 * big_y[j] + q.c1 * big_z[j];
        let (a, b) = (c2 * a - s2 * b, s2 * a + c2 * b);
        let (a, b) = (a * sl - b * cl, a * cl + b * sl);
        a_[j] = a;
        b_[j] = b * cb + c * sb;
        c_[j] = c * cb - b * sb;
    }
    let d = a_[0].atan2(c_[0]);
    let (sd, cd) = d.sin_cos();
    (1..9)
        .map(|j| {
            let mut x = a_[j] * cd - c_[j] * sd;
            let y = a_[j] * sd + c_[j] * cd;
            let z = b_[j];
            let ratio = (x / s4[j - 1].r).clamp(-1.0, 1.0);
            x += z.abs() / K[j] * (1.0 - ratio * ratio).sqrt();
            let w = delta / (delta + z / 2475.0);
            MoonPosition { name: tl(SATURN_MOONS[j - 1]), x: x * w, y: y * w, behind: z > 0.0 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Moment from a Julian date (UT; the examples' TD differs by about a minute).
    fn moment(jd: f64) -> Moment {
        // 2448972.5 = 1992-12-16 0h, 2451439.5 = 1999-09-18 0h.
        let (base, (y, mo, d)) = if jd < 2_450_000.0 { (2_448_972.5, (1992, 12, 16)) } else { (2_451_439.5, (1999, 9, 18)) };
        let secs = (jd - base) * 86_400.0;
        Moment { year: y, month: mo, day: d, hour: (secs / 3600.0) as u32, minute: ((secs % 3600.0) / 60.0) as u32, second: (secs % 60.0) as f32 }
    }

    #[test]
    fn galilean_moons_match_meeus_example() {
        let p = positions(Body::Jupiter, &moment(2_448_972.500_68));
        let want = [(-3.44, 0.21), (7.44, 0.25), (1.24, 0.65), (7.08, 1.10)];
        for (m, (x, y)) in p.iter().zip(want) {
            assert!((m.x - x).abs() < 0.02 && (m.y - y).abs() < 0.02, "{}: {:.2} {:.2}", m.name, m.x, m.y);
        }
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn saturn_moons_match_meeus_example() {
        let p = positions(Body::Saturn, &moment(2_451_439.500_74));
        let want = [(3.102, -0.204), (3.823, 0.318_f64), (4.027, -1.061), (-5.365, -1.148), (-0.972, -3.136), (14.568, 4.738), (-18.001, -5.328), (-48.760, 4.137)];
        for (m, (x, y)) in p.iter().zip(want) {
            // Saturn's position from Keplerian elements instead of VSOP87.
            let tol = 0.01 * f64::hypot(x, y).max(3.0);
            assert!((m.x - x).abs() < tol && (m.y - y).abs() < tol, "{}: {:.3} {:.3} vs {x} {y}", m.name, m.x, m.y);
        }
    }
}
