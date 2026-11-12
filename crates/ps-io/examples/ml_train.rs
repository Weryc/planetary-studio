//! The learned sharpening: fit the model and check it honestly.
//!
//! `cargo run --release -p ps-io --example ml_train -- dataset.tsv report.json [learned_data.rs [other.tsv]]`
//!
//! Every method is judged leaving out a whole group at a time (one reference, at every size),
//! trained on the rest. Errors are RMS of ln(gain) on the
//! grid up to 1.5× the cut-off (beyond it the app's own grain cleaning acts), against:
//! - the truth (best gain back to the known sharp reference);
//! - the teacher (BlurXTerminator, on one variant per truth).
//!
//! Gains are taken between ×0.3 (past the cut-off the grain is cleaned) and ×30. With a third
//! argument, the chosen model trained on everything is written there, with one model per kind
//! of night (trained on that night's examples only, judged the same way); with a fourth, its
//! answers for that other dataset go to `report.json.extra.tsv`.
#[path = "ml_common/mod.rs"]
mod ml;

use ml::*;
use ps_core::learned::{FEATURES, GRID, STEPS};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// The nights of the seeing simulation, as the variants start.
const NIGHT_LABELS: [&str; 3] = ["calmo", "medio", "turbolento"];

/// Grid points judged (r ≤ 1.5).
const JUDGED: usize = 11;
const MAX_LN: f32 = 3.4;

/// Gains from ×0.3 (grain cleaned past the cut-off) to ×30, as in the app.
fn clamp(t: [f32; STEPS]) -> [f32; STEPS] {
    t.map(|v| v.clamp(ps_core::learned::MIN_GAIN.ln(), MAX_LN))
}

#[derive(Clone)]
struct Sample {
    row: usize,
    x: Vec<f32>,
    y: [f32; STEPS],
    weight: f32,
}

/// What a method learns from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Teacher {
    Truth,
    Bxt,
    Both,
}

fn samples(rows: &[Row], idx: &[usize], teacher: Teacher) -> Vec<Sample> {
    let mut out = Vec::new();
    for &i in idx {
        let r = &rows[i];
        if matches!(teacher, Teacher::Truth | Teacher::Both)
            && let Some(w) = r.wiener
        {
            out.push(Sample { row: i, x: r.features.clone(), y: clamp(w), weight: 1.0 });
        }
        if matches!(teacher, Teacher::Bxt | Teacher::Both)
            && let Some(b) = r.bxt
        {
            out.push(Sample { row: i, x: r.features.clone(), y: clamp(b), weight: if teacher == Teacher::Both { 0.5 } else { 1.0 } });
        }
    }
    out
}

trait Model {
    fn predict(&self, row: &Row) -> [f32; STEPS];
}

/// The app's fixed curve at a given strength (None: the automatic one by kind).
struct Fixed(Option<f32>);

fn fixed_curve(kind: &str, strength: f32) -> [f32; STEPS] {
    let p = ps_core::smart::SmartParams { strength, ..Default::default() };
    let _ = kind;
    clamp(GRID.map(|r| p.target(r * 0.2, 0.2).max(1e-3).ln()))
}

impl Model for Fixed {
    fn predict(&self, row: &Row) -> [f32; STEPS] {
        let auto = if row.kind == "Surface" || row.kind == "FullDisk" { 0.4 } else { 0.35 };
        fixed_curve(&row.kind, self.0.unwrap_or(auto))
    }
}

/// One curve for everything: the mean of the examples (a re-trained static curve).
struct Mean(BTreeMap<String, [f32; STEPS]>, bool);

impl Mean {
    fn fit(s: &[Sample], rows: &[Row], by_kind: bool) -> Self {
        let mut acc: BTreeMap<String, ([f64; STEPS], f64)> = BTreeMap::new();
        for q in s {
            for key in [String::new(), if by_kind { rows[q.row].kind.clone() } else { String::new() }] {
                let e = acc.entry(key).or_insert(([0.0; STEPS], 0.0));
                (0..STEPS).for_each(|k| e.0[k] += (q.weight * q.y[k]) as f64);
                e.1 += q.weight as f64;
            }
        }
        Mean(acc.into_iter().map(|(k, (s, n))| (k, s.map(|v| (v / n.max(1e-9)) as f32))).collect(), by_kind)
    }
}

impl Model for Mean {
    fn predict(&self, row: &Row) -> [f32; STEPS] {
        let key = if self.1 { row.kind.clone() } else { String::new() };
        *self.0.get(&key).or(self.0.get("")).unwrap()
    }
}

/// Feature standardisation learned on the training set (constant features are dropped).
#[derive(Clone)]
struct Scaler {
    mean: Vec<f32>,
    scale: Vec<f32>,
}

impl Scaler {
    fn fit(s: &[Sample]) -> Self {
        let n = s.len().max(1) as f64;
        let mean: Vec<f32> = (0..FEATURES).map(|j| (s.iter().map(|q| q.x[j] as f64).sum::<f64>() / n) as f32).collect();
        let scale = (0..FEATURES)
            .map(|j| {
                let sd = (s.iter().map(|q| ((q.x[j] - mean[j]) as f64).powi(2)).sum::<f64>() / n).sqrt() as f32;
                if sd > 1e-4 { 1.0 / sd } else { 0.0 }
            })
            .collect();
        Scaler { mean, scale }
    }

    fn z(&self, x: &[f32]) -> Vec<f32> {
        x.iter().zip(&self.mean).zip(&self.scale).map(|((v, m), s)| ((v - m) * s).clamp(-6.0, 6.0)).collect()
    }
}

/// Ridge regression, one linear map from the standardised features to the log-gains.
struct Ridge {
    scaler: Scaler,
    weights: Vec<[f32; FEATURES]>,
    bias: [f32; STEPS],
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, p);
        b.swap(c, p);
        let d = a[c][c];
        if d.abs() < 1e-12 {
            continue;
        }
        let pivot = a[c].clone();
        for r in 0..n {
            if r != c {
                let f = a[r][c] / d;
                if f != 0.0 {
                    a[r][c..].iter_mut().zip(&pivot[c..]).for_each(|(x, p)| *x -= f * p);
                    b[r] -= f * b[c];
                }
            }
        }
    }
    (0..n).map(|i| if a[i][i].abs() < 1e-12 { 0.0 } else { b[i] / a[i][i] }).collect()
}

impl Ridge {
    fn fit(s: &[Sample], lambda: f64) -> Self {
        let scaler = Scaler::fit(s);
        let p = FEATURES + 1;
        let zs: Vec<Vec<f32>> = s.iter().map(|q| scaler.z(&q.x)).collect();
        let mut a = vec![vec![0.0f64; p]; p];
        for (q, z) in s.iter().zip(&zs) {
            let w = q.weight as f64;
            for i in 0..p {
                let zi = if i < FEATURES { z[i] as f64 } else { 1.0 };
                for j in 0..p {
                    let zj = if j < FEATURES { z[j] as f64 } else { 1.0 };
                    a[i][j] += w * zi * zj;
                }
            }
        }
        let total: f64 = s.iter().map(|q| q.weight as f64).sum();
        for (i, row) in a.iter_mut().enumerate().take(FEATURES) {
            row[i] += lambda * total / 100.0;
        }
        let mut weights = vec![[0.0f32; FEATURES]; STEPS];
        let mut bias = [0.0f32; STEPS];
        for k in 0..STEPS {
            let b: Vec<f64> = (0..p).map(|i| s.iter().zip(&zs).map(|(q, z)| q.weight as f64 * (if i < FEATURES { z[i] as f64 } else { 1.0 }) * q.y[k] as f64).sum()).collect();
            let sol = solve(a.clone(), b);
            for j in 0..FEATURES {
                weights[k][j] = sol[j] as f32;
            }
            bias[k] = sol[FEATURES] as f32;
        }
        Ridge { scaler, weights, bias }
    }
}

impl Model for Ridge {
    fn predict(&self, row: &Row) -> [f32; STEPS] {
        let z = self.scaler.z(&row.features);
        clamp(std::array::from_fn(|k| self.bias[k] + self.weights[k].iter().zip(&z).map(|(w, v)| w * v).sum::<f32>()))
    }
}

/// The k most similar examples, averaged (closer ones count more).
struct Neighbours {
    scaler: Scaler,
    train: Vec<(Vec<f32>, [f32; STEPS], f32)>,
    k: usize,
}

impl Neighbours {
    fn fit(s: &[Sample], k: usize) -> Self {
        let scaler = Scaler::fit(s);
        let train = s.iter().map(|q| (scaler.z(&q.x), q.y, q.weight)).collect();
        Neighbours { scaler, train, k }
    }
}

impl Model for Neighbours {
    fn predict(&self, row: &Row) -> [f32; STEPS] {
        let z = self.scaler.z(&row.features);
        let mut d: Vec<(f32, usize)> = self.train.iter().enumerate().map(|(i, t)| (t.0.iter().zip(&z).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(), i)).collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut acc = [0.0f32; STEPS];
        let mut wsum = 0.0;
        for &(dist, i) in d.iter().take(self.k) {
            let w = self.train[i].2 / (1.0 + dist);
            (0..STEPS).for_each(|k| acc[k] += w * self.train[i].1[k]);
            wsum += w;
        }
        acc.map(|v| v / wsum.max(1e-9))
    }
}

fn rms(a: &[f32; STEPS], b: &[f32; STEPS]) -> f32 {
    ((0..JUDGED).map(|k| (a[k] - b[k]).powi(2)).sum::<f32>() / JUDGED as f32).sqrt()
}

/// Errors of one method, against the truth and against the teacher.
#[derive(Default, Clone)]
struct Score {
    truth: Vec<(usize, f32)>,
    teacher: Vec<(usize, f32)>,
}

fn mean(v: &[(usize, f32)]) -> f32 {
    v.iter().map(|q| q.1).sum::<f32>() / v.len().max(1) as f32
}

type Fit = Box<dyn Fn(&[Sample], &[Row]) -> Box<dyn Model>>;

fn method(name: &str, teacher: Teacher, fit: Fit) -> (String, Teacher, Fit) {
    (name.to_string(), teacher, fit)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rows = read_rows(args.next().expect("dataset.tsv").as_ref());
    let report = args.next().expect("report.json");
    let out_model = args.next();
    let groups: Vec<String> = {
        let mut g: Vec<String> = rows.iter().map(|r| r.group.clone()).collect();
        g.sort();
        g.dedup();
        g
    };
    eprintln!("{} esempi, {} gruppi, {} con verità, {} con BXT", rows.len(), groups.len(), rows.iter().filter(|r| r.wiener.is_some()).count(), rows.iter().filter(|r| r.bxt.is_some()).count());

    let mut methods: Vec<(String, Teacher, Fit)> = vec![
        method("Curva fissa di oggi (automatica)", Teacher::Truth, Box::new(|_, _| Box::new(Fixed(None)))),
        method("Curva fissa di oggi al 100 %", Teacher::Truth, Box::new(|_, _| Box::new(Fixed(Some(1.0))))),
        method("Curva media, dalla verità", Teacher::Truth, Box::new(|s, r| Box::new(Mean::fit(s, r, false)))),
        method("Curva media per tipo, dalla verità", Teacher::Truth, Box::new(|s, r| Box::new(Mean::fit(s, r, true)))),
        method("Curva media, da BXT", Teacher::Bxt, Box::new(|s, r| Box::new(Mean::fit(s, r, false)))),
    ];
    for (teacher, label) in [(Teacher::Truth, "verità"), (Teacher::Bxt, "BXT"), (Teacher::Both, "verità + BXT")] {
        for lambda in [1.0, 10.0, 100.0] {
            methods.push(method(&format!("Lineare λ={lambda}, da {label}"), teacher, Box::new(move |s, _| Box::new(Ridge::fit(s, lambda)))));
        }
        for k in [5, 15] {
            methods.push(method(&format!("Vicini k={k}, da {label}"), teacher, Box::new(move |s, _| Box::new(Neighbours::fit(s, k)))));
        }
    }

    let mut scores: Vec<Score> = vec![Score::default(); methods.len()];
    for g in &groups {
        let train: Vec<usize> = (0..rows.len()).filter(|&i| &rows[i].group != g).collect();
        let test: Vec<usize> = (0..rows.len()).filter(|&i| &rows[i].group == g).collect();
        for (m, (_, teacher, fit)) in methods.iter().enumerate() {
            let model = fit(&samples(&rows, &train, *teacher), &rows);
            for &i in &test {
                let p = model.predict(&rows[i]);
                if let Some(w) = rows[i].wiener {
                    scores[m].truth.push((i, rms(&p, &clamp(w))));
                }
                if let Some(b) = rows[i].bxt {
                    scores[m].teacher.push((i, rms(&p, &clamp(b))));
                }
            }
        }
    }

    // How far the teacher is from the truth, where both are known.
    let teacher_vs_truth: Vec<(usize, f32)> = rows.iter().enumerate().filter_map(|(i, r)| Some((i, rms(&clamp(r.bxt?), &clamp(r.wiener?))))).collect();

    println!("{:<44} {:>14} {:>14}", "metodo", "errore/verità", "errore/BXT");
    for ((name, ..), s) in methods.iter().zip(&scores) {
        println!("{name:<44} {:>14.3} {:>14.3}", mean(&s.truth), mean(&s.teacher));
    }
    println!("{:<44} {:>14.3}  ({} casi)", "BXT stesso, contro la verità", mean(&teacher_vs_truth), teacher_vs_truth.len());

    // Per kind, for the page.
    let kinds = ["Planet", "Saturn", "FullDisk", "Surface"];
    let by_kind = |v: &[(usize, f32)], kind: &str| mean(&v.iter().copied().filter(|&(i, _)| rows[i].kind == kind).collect::<Vec<_>>());

    // The chosen model: the best linear one against the truth (a few hundred numbers in the
    // app; the neighbours would carry every example along).
    let best = (0..methods.len()).filter(|&m| methods[m].0.starts_with("Lineare")).min_by(|&a, &b| mean(&scores[a].truth).total_cmp(&mean(&scores[b].truth))).unwrap();
    println!("scelto: {}", methods[best].0);

    // One model per night: trained on that night's examples only, judged on them leaving out a
    // group, against the main model on the same examples.
    for night in NIGHT_LABELS {
        let of_night: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].variant.starts_with(night)).collect();
        let (mut own, mut main) = (Vec::new(), Vec::new());
        for g in &groups {
            let train: Vec<usize> = of_night.iter().copied().filter(|&i| &rows[i].group != g).collect();
            let all_train: Vec<usize> = (0..rows.len()).filter(|&i| &rows[i].group != g).collect();
            let m_own = (methods[best].2)(&samples(&rows, &train, methods[best].1), &rows);
            let m_main = (methods[best].2)(&samples(&rows, &all_train, methods[best].1), &rows);
            for &i in of_night.iter().filter(|&&i| &rows[i].group == g) {
                if let Some(w) = rows[i].wiener {
                    own.push((i, rms(&m_own.predict(&rows[i]), &clamp(w))));
                    main.push((i, rms(&m_main.predict(&rows[i]), &clamp(w))));
                }
            }
        }
        println!("serata {night:<11} modello della serata {:.3}, modello unico {:.3} ({} esempi)", mean(&own), mean(&main), own.len());
    }

    let mut json = String::from("{\n\"methods\": [\n");
    for (m, ((name, teacher, _), s)) in methods.iter().zip(&scores).enumerate() {
        let kinds_text: Vec<String> = kinds.iter().map(|k| format!("\"{k}\": [{:.4}, {:.4}]", by_kind(&s.truth, k), by_kind(&s.teacher, k))).collect();
        writeln!(json, "  {{\"name\": \"{name}\", \"teacher\": \"{teacher:?}\", \"truth\": {:.4}, \"bxt\": {:.4}, \"kinds\": {{{}}}}}{}", mean(&s.truth), mean(&s.teacher), kinds_text.join(", "), if m + 1 < methods.len() { "," } else { "" }).unwrap();
    }
    writeln!(json, "],\n\"teacher_vs_truth\": {:.4},\n\"chosen\": \"{}\",", mean(&teacher_vs_truth), methods[best].0).unwrap();

    // Curves for a few examples: truth, teacher, today's fixed curve, the chosen model (held out).
    let all: Vec<usize> = (0..rows.len()).collect();
    json.push_str("\"examples\": [\n");
    let mut first = true;
    for r in &rows {
        if r.variant != "medio_snr450" {
            continue;
        }
        let train: Vec<usize> = all.iter().copied().filter(|&j| rows[j].group != r.group).collect();
        let model = (methods[best].2)(&samples(&rows, &train, methods[best].1), &rows);
        let p = model.predict(r);
        let fixed = Fixed(None).predict(r);
        let curve = |c: Option<[f32; STEPS]>| c.map_or("null".to_string(), |c| format!("[{}]", clamp(c).map(|v| format!("{:.3}", v.exp())).join(",")));
        if !first {
            json.push_str(",\n");
        }
        first = false;
        write!(
            json,
            "  {{\"name\": \"{}\", \"variant\": \"{}\", \"kind\": \"{}\", \"body\": \"{}\", \"cutoff\": {:.4}, \"truth\": {}, \"bxt\": {}, \"fixed\": {}, \"model\": {}}}",
            r.name.replace('"', ""),
            r.variant,
            r.kind,
            r.body,
            r.cutoff,
            curve(r.wiener),
            curve(r.bxt),
            curve(Some(fixed)),
            curve(Some(p))
        )
        .unwrap();
    }
    json.push_str("\n],\n");
    writeln!(json, "\"grid\": [{}]\n}}", GRID.map(|g| g.to_string()).join(",")).unwrap();
    std::fs::write(&report, json).unwrap();

    // The chosen method's held-out answers, for the image-level check (`ml_eval_ref`).
    let mut held = String::new();
    for g in &groups {
        let train: Vec<usize> = (0..rows.len()).filter(|&i| &rows[i].group != g).collect();
        let model = (methods[best].2)(&samples(&rows, &train, methods[best].1), &rows);
        for r in rows.iter().filter(|r| &r.group == g) {
            writeln!(held, "{}\t{}\t{}", r.name, r.variant, model.predict(r).map(|v| format!("{v:.4}")).join("\t")).unwrap();
        }
    }
    std::fs::write(format!("{report}.heldout.tsv"), held).unwrap();

    // The chosen method trained on everything, answering for another dataset (the real stacks).
    if let Some(extra) = args.next() {
        let other = read_rows(extra.as_ref());
        let model = (methods[best].2)(&samples(&rows, &(0..rows.len()).collect::<Vec<_>>(), methods[best].1), &rows);
        let mut text = String::new();
        for r in &other {
            writeln!(text, "{}\t{}\t{}", r.name, r.variant, model.predict(r).map(|v| format!("{v:.4}")).join("\t")).unwrap();
        }
        std::fs::write(format!("{report}.extra.tsv"), text).unwrap();
    }

    if let Some(path) = out_model {
        write_model(&path, &rows, best, &methods);
    }
}

/// The chosen model trained on everything, and one per night, as Rust tables for
/// `ps_core::learned`.
fn write_model(path: &str, rows: &[Row], best: usize, methods: &[(String, Teacher, Fit)]) {
    let name = &methods[best].0;
    let lambda: f64 = name.split("λ=").nth(1).and_then(|t| t.split(',').next()).and_then(|t| t.parse().ok()).unwrap_or(10.0);
    let list = |v: &[f32]| v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(", ");
    // One Weights literal from the examples `idx`.
    let weights = |idx: &[usize]| -> (String, usize) {
        let s = samples(rows, idx, methods[best].1);
        let ridge = Ridge::fit(&s, lambda);
        let zs: Vec<Vec<f32>> = s.iter().map(|q| ridge.scaler.z(&q.x)).collect();
        let lo: Vec<f32> = (0..FEATURES).map(|j| zs.iter().map(|z| z[j]).fold(f32::MAX, f32::min)).collect();
        let hi: Vec<f32> = (0..FEATURES).map(|j| zs.iter().map(|z| z[j]).fold(f32::MIN, f32::max)).collect();
        let mut t = String::from("Weights {\n");
        writeln!(t, "    mean: [{}],", list(&ridge.scaler.mean)).unwrap();
        writeln!(t, "    scale: [{}],", list(&ridge.scaler.scale)).unwrap();
        writeln!(t, "    bias: [{}],", list(&ridge.bias)).unwrap();
        t.push_str("    weights: [\n");
        for w in &ridge.weights {
            writeln!(t, "        [{}],", list(w)).unwrap();
        }
        t.push_str("    ],\n");
        writeln!(t, "    z_min: [{}],", list(&lo)).unwrap();
        writeln!(t, "    z_max: [{}],", list(&hi)).unwrap();
        t.push('}');
        (t, s.len())
    };
    let all: Vec<usize> = (0..rows.len()).collect();
    let (main, n) = weights(&all);
    let mut t = String::new();
    writeln!(t, "//! Weights of the learned sharpening models, written by `ps-io/examples/ml_train.rs` (do not edit).").unwrap();
    writeln!(t, "//! {name}; {n} examples from {} images; then one model per night (calm, average, turbulent).", { let mut v: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect(); v.dedup(); v.len() }).unwrap();
    writeln!(t, "#![allow(clippy::excessive_precision)]\nuse crate::learned::Weights;\n").unwrap();
    writeln!(t, "pub(crate) const ALL: Weights = {main};\n").unwrap();
    t.push_str("pub(crate) const NIGHTS: [Weights; 3] = [\n");
    for night in NIGHT_LABELS {
        let idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].variant.starts_with(night)).collect();
        writeln!(t, "    // {night}: {} examples", idx.len()).unwrap();
        writeln!(t, "    {},", weights(&idx).0.replace('\n', "\n    ")).unwrap();
    }
    t.push_str("];\n");
    std::fs::write(path, t).unwrap();
    eprintln!("modelli scritti in {path}");
}
