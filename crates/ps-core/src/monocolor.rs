//! Da immagini monocromatiche a RGB o LRGB: le riprese fatte con una camera mono e i filtri
//! R, G, B (ed eventualmente L) diventano un'unica immagine a colori.
//!
//! Due modi:
//! - **classico** ([`compose`]): i canali si allineano al canale di riferimento (L se c'è,
//!   altrimenti G) con la correlazione di fase e si fondono;
//! - **derotato** ([`compose_derotated`], Giove e Saturno): ogni canale è ripreso a un'ora
//!   diversa, e in pochi minuti il pianeta gira. Ogni canale viene portato allo stesso istante
//!   (la media delle ore) con [`crate::derotation`], poi i canali si fondono.
//!
//! LRGB: il colore viene da RGB e il dettaglio dal canale L. Ogni pixel RGB è scalato perché la
//! sua luminanza diventi quella di L (la tinta resta, la luminosità segue L); in precedenza L è
//! portato allo stesso livello medio di RGB, perché i filtri non hanno la stessa sensibilità.
//!
//! La funzione è pura: niente I/O. I controlli sugli ingressi ([`validate`]) restituiscono un
//! [`Error`] che dice cosa non va e in quale file.

use crate::align::{measure_shift, shift};
use crate::capture::Moment;
use crate::capture::Body;
use crate::derotation::{self, Frame, Globe, JupiterSystem, Params};
use crate::image::{Image, bilinear_clamped};
use crate::tr;
use rayon::prelude::*;

/// A che filtro appartiene un'immagine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    L,
    R,
    G,
    B,
}

impl Role {
    pub const ALL: [Role; 4] = [Role::L, Role::R, Role::G, Role::B];

    pub fn label(self) -> &'static str {
        match self {
            Role::L => "L",
            Role::R => "R",
            Role::G => "G",
            Role::B => "B",
        }
    }

    /// Il canale che il filtro scritto nel log o nel nome (`R`, `Red`, `Luminance`…) indica.
    pub fn from_word(word: &str) -> Option<Role> {
        match word.to_ascii_lowercase().as_str() {
            "r" | "red" | "rosso" => Some(Role::R),
            "g" | "green" | "verde" => Some(Role::G),
            "b" | "blue" | "blu" => Some(Role::B),
            "l" | "lum" | "luma" | "luminance" | "luminanza" => Some(Role::L),
            _ => None,
        }
    }
}

/// Il canale di un'immagine: prima il filtro del log o del file, poi le parole del nome
/// (`giove_R_0012`). Se il nome ne contiene più d'uno diversi non si indovina.
pub fn guess_role(filter: Option<&str>, stem: &str) -> Option<Role> {
    if let Some(r) = filter.and_then(|f| Role::from_word(f.trim())) {
        return Some(r);
    }
    let mut found: Vec<Role> = stem.split(|c: char| !c.is_ascii_alphabetic()).filter_map(Role::from_word).collect();
    found.dedup();
    if found.len() == 1 { Some(found[0]) } else { None }
}

/// Un'immagine in ingresso.
pub struct Input<'a> {
    /// Il nome del file, per i messaggi.
    pub name: String,
    pub image: &'a Image,
    pub role: Role,
    /// L'ora della ripresa (serve per la versione derotata).
    pub time: Option<Moment>,
    /// Il disco del pianeta, se già noto (altrimenti si cerca nell'immagine).
    pub globe: Option<Globe>,
}

/// Cosa non va negli ingressi.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Mancano immagini: ne servono almeno R, G e B.
    Missing(Role),
    /// Due immagini con lo stesso filtro.
    Duplicate { role: Role, first: String, second: String },
    /// L'immagine è a colori.
    NotMono(String),
    /// Dimensioni diverse (e il ricampionamento non è stato chiesto).
    Size { name: String, got: (usize, usize), expected: (usize, usize), reference: String },
    /// La versione derotata vale solo per Giove e Saturno.
    BodyNotDerotatable(Option<Body>),
    /// Manca l'ora della ripresa.
    NoTime(String),
    /// Il disco del pianeta non si trova.
    NoDisk(String),
    /// L'immagine è vuota o non si è potuta unire.
    Empty,
}

impl Error {
    /// Il messaggio, nella lingua corrente.
    pub fn message(&self) -> String {
        match self {
            Error::Missing(r) => tr!("Manca il canale {}: servono almeno R, G e B (L è facoltativo).", "The {} channel is missing: at least R, G and B are needed (L is optional).", r.label()),
            Error::Duplicate { role, first, second } => tr!("Due immagini hanno il canale {}: {first} e {second}. Ogni canale va una sola volta.", "Two images have the {} channel: {first} and {second}. Each channel can be used only once.", role.label()),
            Error::NotMono(name) => tr!("{name} è un'immagine a colori: per creare un RGB servono immagini monocromatiche.", "{name} is a colour image: monochrome images are needed to create an RGB."),
            Error::Size { name, got, expected, reference } => tr!(
                "{name} è {}×{}, ma {reference} è {}×{}: le immagini devono avere le stesse dimensioni (oppure ricampionale).",
                "{name} is {}×{}, but {reference} is {}×{}: the images must be the same size (or resample them).",
                got.0,
                got.1,
                expected.0,
                expected.1
            ),
            Error::BodyNotDerotatable(Some(b)) => tr!("La derotazione vale per Giove, Saturno e Marte, non per {}.", "Derotation applies to Jupiter, Saturn and Mars, not {}.", b.label()),
            Error::BodyNotDerotatable(None) => crate::i18n::tl("La derotazione vale solo per Giove e Saturno: scegli il pianeta.").to_string(),
            Error::NoTime(name) => tr!("Manca l'ora della ripresa di {name}: la derotazione la richiede (dal nome del file o dal log; si può scrivere a mano).", "The capture time of {name} is missing: derotation needs it (from the file name or the log; it can be typed in)."),
            Error::NoDisk(name) => tr!("Il disco del pianeta non si trova in {name}.", "The planet's disk was not found in {name}."),
            Error::Empty => crate::i18n::tl("Nessuna immagine da unire").to_string(),
        }
    }
}

/// Il risultato.
pub struct Output {
    pub image: Image,
    /// L'istante a cui sono portati i canali (solo per la versione derotata).
    pub moment: Option<Moment>,
    /// Minuti fra la prima e l'ultima ripresa (solo per la versione derotata).
    pub span_minutes: f64,
    pub lrgb: bool,
    /// Il verso di rotazione usato (solo per la versione derotata).
    pub mirrored: Option<bool>,
}

/// Come unire.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Ricampiona alla dimensione del canale di riferimento le immagini di dimensione diversa
    /// (altrimenti è un errore).
    pub resize: bool,
    /// Allinea i canali fra loro (modo classico).
    pub align: bool,
    pub system: JupiterSystem,
    /// Immagine specchiata (come in [`Params`]); `None` = si sceglie il verso che fa coincidere
    /// meglio i dischi (un verso sbagliato sposta i dettagli due volte invece di fermarli).
    pub mirrored: Option<bool>,
}

impl Default for Options {
    fn default() -> Self {
        Options { resize: false, align: true, system: JupiterSystem::II, mirrored: None }
    }
}

fn input_of<'a, 'b>(inputs: &'a [Input<'b>], role: Role) -> Option<&'a Input<'b>> {
    inputs.iter().find(|i| i.role == role)
}

/// I controlli: tutte monocromatiche, ogni canale una volta sola, R, G e B presenti, stessa
/// dimensione (a meno che `resize`). Restituisce la dimensione del risultato: quella del
/// canale di riferimento (L se c'è, altrimenti G).
pub fn validate(inputs: &[Input], resize: bool) -> Result<(usize, usize), Error> {
    for i in inputs {
        if !i.image.is_mono() {
            return Err(Error::NotMono(i.name.clone()));
        }
    }
    for (k, a) in inputs.iter().enumerate() {
        if let Some(b) = inputs[k + 1..].iter().find(|b| b.role == a.role) {
            return Err(Error::Duplicate { role: a.role, first: a.name.clone(), second: b.name.clone() });
        }
    }
    for role in [Role::R, Role::G, Role::B] {
        if input_of(inputs, role).is_none() {
            return Err(Error::Missing(role));
        }
    }
    let reference = input_of(inputs, Role::L).or_else(|| input_of(inputs, Role::G)).expect("G present");
    let size = (reference.image.width(), reference.image.height());
    if !resize
        && let Some(i) = inputs.iter().find(|i| (i.image.width(), i.image.height()) != size)
    {
        return Err(Error::Size { name: i.name.clone(), got: (i.image.width(), i.image.height()), expected: size, reference: reference.name.clone() });
    }
    Ok(size)
}

/// Il piano portato a `size` (bilineare), se serve.
fn sized(img: &Image, size: (usize, usize)) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let plane = &img.channels()[0];
    if (w, h) == size {
        return plane.clone();
    }
    let (sx, sy) = (w as f32 / size.0 as f32, h as f32 / size.1 as f32);
    let mut out = vec![0.0f32; size.0 * size.1];
    out.par_chunks_mut(size.0).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = bilinear_clamped(plane, w, h, (x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5);
        }
    });
    out
}

/// Dai piani R, G, B (e L) al risultato: LRGB se c'è L.
fn merge(r: Vec<f32>, g: Vec<f32>, b: Vec<f32>, l: Option<Vec<f32>>, size: (usize, usize), depth: u8) -> Image {
    let Some(l) = l else {
        return Image::new(size.0, size.1, vec![r, g, b], depth);
    };
    let n = size.0 * size.1;
    let lum = |i: usize| (r[i] + g[i] + b[i]) / 3.0;
    let mean_rgb = (0..n).map(|i| lum(i) as f64).sum::<f64>() / n.max(1) as f64;
    let mean_l = l.iter().map(|&v| v as f64).sum::<f64>() / n.max(1) as f64;
    let k = if mean_l > 1e-9 { (mean_rgb / mean_l) as f32 } else { 1.0 };
    // Sotto questo livello il colore non è affidabile: il pixel segue L in grigio.
    let floor = 0.01 * lum_peak(&r, &g, &b).max(1e-6);
    let mut planes = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    let [pr, pg, pb] = &mut planes;
    pr.par_iter_mut().zip(pg.par_iter_mut()).zip(pb.par_iter_mut()).enumerate().for_each(|(i, ((or, og), ob))| {
        let target = (l[i] * k).max(0.0);
        let y = lum(i);
        let ratio = (target / y.max(floor)).min(8.0);
        let w = (y / (2.0 * floor)).clamp(0.0, 1.0);
        let w = w * w * (3.0 - 2.0 * w);
        *or = (w * r[i] * ratio + (1.0 - w) * target).clamp(0.0, 1.0);
        *og = (w * g[i] * ratio + (1.0 - w) * target).clamp(0.0, 1.0);
        *ob = (w * b[i] * ratio + (1.0 - w) * target).clamp(0.0, 1.0);
    });
    Image::new(size.0, size.1, planes.into(), depth)
}

fn lum_peak(r: &[f32], g: &[f32], b: &[f32]) -> f32 {
    (0..r.len()).map(|i| (r[i] + g[i] + b[i]) / 3.0).fold(0.0, f32::max)
}

fn depth_of(inputs: &[Input]) -> u8 {
    inputs.iter().map(|i| i.image.source_bit_depth).max().unwrap_or(16).max(16)
}

/// Il modo classico: i canali portati a una stessa dimensione, allineati al riferimento e
/// fusi. `inputs` contiene R, G, B e, se si vuole l'LRGB, L.
pub fn compose(inputs: &[Input], options: &Options) -> Result<Output, Error> {
    let size = validate(inputs, options.resize)?;
    let reference = if input_of(inputs, Role::L).is_some() { Role::L } else { Role::G };
    let planes = |role: Role| input_of(inputs, role).map(|i| sized(i.image, size));
    let aligned = |role: Role| -> Option<Vec<f32>> {
        let plane = planes(role)?;
        if !options.align || role == reference {
            return Some(plane);
        }
        let target = planes(reference)?;
        let s = measure_shift(&target, &plane, size.0, size.1);
        Some(shift(&plane, size.0, size.1, s))
    };
    let (r, g, b, l) = (aligned(Role::R).ok_or(Error::Empty)?, aligned(Role::G).ok_or(Error::Empty)?, aligned(Role::B).ok_or(Error::Empty)?, aligned(Role::L));
    let lrgb = l.is_some();
    Ok(Output { image: merge(r, g, b, l, size, depth_of(inputs)), moment: None, span_minutes: 0.0, lrgb, mirrored: None })
}

/// I controlli della versione derotata: quelli di [`validate`], il pianeta (Giove o Saturno)
/// e l'ora di ogni ripresa. Restituisce il pianeta e la dimensione del risultato.
pub fn check_derotated(inputs: &[Input], body: Option<Body>, options: &Options) -> Result<(Body, (usize, usize)), Error> {
    let body = body.filter(|b| derotation::rotation_period(*b, options.system).is_some()).ok_or(Error::BodyNotDerotatable(body))?;
    let size = validate(inputs, options.resize)?;
    if let Some(i) = inputs.iter().find(|i| !i.time.is_some_and(|t| t.valid())) {
        return Err(Error::NoTime(i.name.clone()));
    }
    Ok((body, size))
}

/// Il modo derotato: ogni canale è portato all'istante medio delle riprese con la rotazione
/// del pianeta, e poi i canali si fondono (senza altri allineamenti: i centri dei dischi e il
/// loro affinamento li hanno già allineati). Solo Giove e Saturno, con l'ora di ogni ripresa.
pub fn compose_derotated(inputs: &[Input], body: Option<Body>, options: &Options) -> Result<Output, Error> {
    let (body, size) = check_derotated(inputs, body, options)?;
    let times: Vec<(&Input, Moment)> = inputs.iter().filter_map(|i| i.time.filter(Moment::valid).map(|t| (i, t))).collect();
    let moment = derotation::mean_moment(&times.iter().map(|(_, t)| (*t, 1.0f32)).collect::<Vec<_>>()).ok_or(Error::Empty)?;
    let span_minutes = {
        let (lo, hi) = times.iter().fold((f64::MAX, f64::MIN), |(lo, hi), (_, t)| (lo.min(t.julian()), hi.max(t.julian())));
        (hi - lo) * 24.0 * 60.0
    };
    // I canali alla stessa dimensione; il disco di ciascuno (dato o cercato).
    let images: Vec<Image> = inputs.iter().map(|i| Image::new(size.0, size.1, vec![sized(i.image, size)], i.image.source_bit_depth)).collect();
    let mut globes = Vec::with_capacity(inputs.len());
    for (input, image) in inputs.iter().zip(&images) {
        let globe = match input.globe {
            Some(g) => g,
            None => {
                let analysis = crate::analysis::analyze(image);
                derotation::find_globe(image, &analysis).ok_or_else(|| Error::NoDisk(input.name.clone()))?
            }
        };
        globes.push(globe);
    }
    // Raggio e orientamento comuni (la mediana), centri propri.
    let median = |mut v: Vec<f32>| {
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    let radius = median(globes.iter().map(|g| g.radius).collect());
    let angle = median(globes.iter().map(|g| g.angle).collect());
    for g in &mut globes {
        g.radius = radius;
        g.angle = angle;
    }
    let reference_index = inputs.iter().position(|i| i.role == Role::L).or_else(|| inputs.iter().position(|i| i.role == Role::G)).expect("G present");
    let base = Params::for_body(body, options.system, Some(&moment)).ok_or(Error::BodyNotDerotatable(Some(body)))?;
    let reference = globes[reference_index];
    let frames_for = || -> Vec<Frame> {
        images
            .iter()
            .zip(&globes)
            .zip(&times)
            .map(|((image, &globe), (_, t))| Frame { image, globe, hours: derotation::hours_between(t, &moment), weight: 1.0, gain: [1.0; 3] })
            .collect()
    };
    // The centres are refined so the derotated markings coincide; with the wrong turning
    // direction that refinement absorbs the error as a shift of the discs, so the discs of the
    // channels end up apart. The right direction is the one that leaves them together.
    let attempt = |mirrored: bool| -> Result<(bool, Vec<Vec<f32>>, f32), Error> {
        let params = Params { mirrored, ..base };
        let mut frames = frames_for();
        derotation::refine_centres(&mut frames, reference_index, &reference, &params, size);
        let mut planes = Vec::with_capacity(frames.len());
        for f in &frames {
            let out = derotation::stack(std::slice::from_ref(f), &reference, &params, size, 1.0, None).ok_or(Error::Empty)?;
            planes.push(out.channels()[0].clone());
        }
        let spread = disc_spread(&planes, size, &reference);
        Ok((mirrored, planes, spread))
    };
    let (mirrored, mut planes) = match options.mirrored {
        Some(m) => attempt(m).map(|(m, p, _)| (m, p))?,
        None => {
            let (a, b) = rayon::join(|| attempt(false), || attempt(true));
            let (a, b) = (a?, b?);
            let best = if b.2 < a.2 { b } else { a };
            (best.0, best.1)
        }
    };
    // After derotation the markings coincide; what is left is the centring of each disc and the
    // atmosphere's colour dispersion (the red and blue discs sit a pixel or more apart in
    // height). The measured residual brings every channel onto the reference.
    if options.align {
        let reference_plane = planes[reference_index].clone();
        for (k, plane) in planes.iter_mut().enumerate() {
            if k != reference_index {
                let s = measure_shift(&reference_plane, plane, size.0, size.1);
                *plane = shift(plane, size.0, size.1, s);
            }
        }
    }
    let mut take = |role: Role| inputs.iter().position(|i| i.role == role).map(|k| std::mem::take(&mut planes[k]));
    let (r, g, b, l) = (take(Role::R).ok_or(Error::Empty)?, take(Role::G).ok_or(Error::Empty)?, take(Role::B).ok_or(Error::Empty)?, take(Role::L));
    let lrgb = l.is_some();
    Ok(Output { image: merge(r, g, b, l, size, depth_of(inputs)), moment: Some(moment), span_minutes, lrgb, mirrored: Some(mirrored) })
}

/// How far apart the channels' discs are: the mean distance of each disc's centre (the centroid
/// of what is brighter than half the peak, near the planet) from the mean of the centres, px.
fn disc_spread(planes: &[Vec<f32>], size: (usize, usize), globe: &Globe) -> f32 {
    let (w, h) = size;
    let reach = 1.3 * globe.radius;
    let centres: Vec<[f32; 2]> = planes
        .iter()
        .map(|p| {
            let mut peak = 0.0f32;
            for y in 0..h {
                for x in 0..w {
                    if (x as f32 - globe.cx).hypot(y as f32 - globe.cy) <= reach {
                        peak = peak.max(p[y * w + x]);
                    }
                }
            }
            let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0.0f64);
            for y in 0..h {
                for x in 0..w {
                    if p[y * w + x] > 0.5 * peak && (x as f32 - globe.cx).hypot(y as f32 - globe.cy) <= reach {
                        sx += x as f64;
                        sy += y as f64;
                        n += 1.0;
                    }
                }
            }
            if n > 0.0 { [(sx / n) as f32, (sy / n) as f32] } else { [globe.cx, globe.cy] }
        })
        .collect();
    let k = centres.len().max(1) as f32;
    let mean = [centres.iter().map(|c| c[0]).sum::<f32>() / k, centres.iter().map(|c| c[1]).sum::<f32>() / k];
    centres.iter().map(|c| (c[0] - mean[0]).hypot(c[1] - mean[1])).sum::<f32>() / k
}
