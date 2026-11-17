//! "Identifica": what the ephemerides say is on the image — moons of Jupiter and Saturn, the
//! Moon's craters and seas, the Sun's orientation — in a tab of the side panel, with the names
//! over the image.
//!
//! The search runs off the interface thread ([`ps_core::identify::identify`]); the tab lets the
//! user correct the subject, the time and the orientation, and every change searches again. The
//! local time zone (daylight saving included, for the capture's own date) comes from the system;
//! for Jupiter and Saturn the time, turn and moons' distances are fitted to the moons found in
//! the picture as soon as it is open (and the image straightened, if asked). The «Annotazione»
//! card says what is shown over the image and, with its switch, prints it onto the picture
//! ([`crate::annotate`]).

use ps_core::i18n::tl;
use ps_core::tr;
use crate::annotate::{Corner, LabelOptions, Plan, Tint};
use crate::datetime::{civil_from_days, date_picker, days_from_civil, time_field};
use crate::panels::PanelAction;
use crate::theme::{self, ACCENT, TEXT, TEXT_WEAK};
use crate::widgets::{ParamSlider, caps, icon_button, note, segmented, toggle};
use eframe::egui::{self, Color32, Pos2, RichText, Stroke, Ui};
use egui_material_icons::icons::*;
use ps_core::Image;
use ps_core::capture::{Body, Moment};
use ps_core::identify::{Identification, LabelKind, Orientation, Placement, identify_with, place_on_craters};
use ps_core::selene;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// What a search was made for.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Key {
    source_gen: u64,
    body: Body,
    when: Moment,
    manual: Option<Orientation>,
    /// The Moon's disk from two craters the user pointed at.
    placed: Option<Placement>,
}

/// A crater chosen from the list and where the user found it on the image.
#[derive(Clone, Copy, Default)]
struct PlacePick {
    name: Option<&'static str>,
    /// Image pixel.
    pixel: Option<(f64, f64)>,
}

/// A running fit of time, turn and distances to the moons found: whether it started by itself,
/// the orientation it started from, and the answer to come.
struct Fitting {
    automatic: bool,
    upright: bool,
    start: Orientation,
    rx: Receiver<Option<ps_core::identify::TimeFit>>,
}

pub struct IdentifyState {
    /// The tab's two cards, folded or not: finding the object and its orientation, and what is
    /// shown and written on the image.
    pub identify_collapsed: bool,
    pub annotate_collapsed: bool,
    /// The names and caption are written on the image (a layer among the masks); otherwise the
    /// names are only drawn over it on screen.
    pub annotate: bool,
    /// The top-bar button: identify the subject and show the names on the image.
    pub active: bool,
    /// The search runs and the names show: the button is on or the tab is the one showing.
    pub open: bool,
    /// Date shown and typed in universal time (otherwise in local time, `offset_min` from UT).
    pub utc: bool,
    /// The zone follows the system and the date; off = the hours typed by hand.
    pub zone_auto: bool,
    /// Local time minus UT, minutes (the system's for the date, or typed).
    pub offset_min: i32,
    /// The zone asked of the system, per day: (day number, offset, daylight saving on).
    zone_cache: Option<(i64, i32, bool)>,
    /// The system being asked for a day's zone (it runs a command: off the interface thread).
    zone_pending: Option<(i64, Receiver<(i32, bool)>)>,
    /// The image the automatic fit already ran for.
    fitted_path: Option<std::path::PathBuf>,
    /// The orientation came from the fit on the moons.
    fit_applied: bool,
    /// After the fit on the moons, turn the image north up.
    fit_upright: bool,
    /// The fit asked to straighten: taken by the app, which turns the image.
    upright_pending: Option<(bool, f32)>,
    /// The time slider: minutes from `scrub_base`, the time the slider started from, and the
    /// time it last set (a different time, from the calendar or a fit, starts it over).
    scrub_min: f32,
    scrub_base: Option<Moment>,
    scrub_set: Option<Moment>,
    /// The user's choices, over what the file says.
    body: Option<Body>,
    time: Option<Moment>,
    manual: Option<Orientation>,
    /// Close-ups without a limb: two craters named and clicked, the disk they give, why it failed,
    /// which of the two is waiting for a click, and the mirror (two points can't tell it).
    picks: [PlacePick; 2],
    placed: Option<Placement>,
    place_error: Option<&'static str>,
    awaiting: Option<usize>,
    place_mirror: bool,
    result: Option<(Key, Arc<Identification>)>,
    pending: Option<(Key, Receiver<Identification>)>,
    /// A search for the time that puts the moons on their spots, and what it found.
    fitting: Option<Fitting>,
    fit_note: Option<String>,
    /// For which image the user's choices were made (forgotten for another image).
    image_path: Option<std::path::PathBuf>,
}

impl Default for IdentifyState {
    fn default() -> Self {
        Self { identify_collapsed: false, annotate_collapsed: false, annotate: false, active: false, open: false, utc: true, zone_auto: true, offset_min: 0, zone_cache: None, zone_pending: None, fitted_path: None, fit_applied: false, fit_upright: true, upright_pending: None, scrub_min: 0.0, scrub_base: None, scrub_set: None, body: None, time: None, manual: None, picks: [PlacePick::default(); 2], placed: None, place_error: None, awaiting: None, place_mirror: false, result: None, pending: None, fitting: None, fit_note: None, image_path: None }
    }
}

/// Local time minus UT in minutes at the UT moment `ut`, as the system's time zone says for that
/// date (so daylight saving is the one of the capture's day, not today's). `None` if the system
/// won't tell.
fn system_offset_at(ut: Moment) -> Option<i32> {
    let epoch = days_from_civil(ut.year, ut.month, ut.day) * 86_400 + i64::from(ut.hour) * 3600 + i64::from(ut.minute) * 60;
    let mut cmd;
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd = std::process::Command::new("powershell");
        cmd.args(["-NoProfile", "-Command", &format!("[int][TimeZoneInfo]::Local.GetUtcOffset([DateTimeOffset]::FromUnixTimeSeconds({epoch})).TotalMinutes")]);
        cmd.creation_flags(0x0800_0000);
    }
    #[cfg(not(target_os = "windows"))]
    {
        cmd = std::process::Command::new("date");
        if cfg!(target_os = "macos") {
            cmd.args(["-r", &epoch.to_string(), "+%z"]);
        } else {
            cmd.args(["-d", &format!("@{epoch}"), "+%z"]);
        }
    }
    let out = String::from_utf8(cmd.output().ok()?.stdout).ok()?;
    let text = out.trim();
    if cfg!(target_os = "windows") {
        return text.parse().ok();
    }
    if text.len() != 5 {
        return None;
    }
    let sign = if text.starts_with('-') { -1 } else { 1 };
    Some(sign * (text[1..3].parse::<i32>().ok()? * 60 + text[3..5].parse::<i32>().ok()?))
}

/// The system's zone at the UT moment `ut`: (offset in minutes, daylight saving on).
fn system_zone(ut: Moment) -> (i32, bool) {
    let offset = system_offset_at(ut).unwrap_or(0);
    // Standard time is the smaller of the January and July offsets of that year.
    let at = |month| system_offset_at(Moment { year: ut.year, month, day: 1, hour: 12, minute: 0, second: 0.0 }).unwrap_or(offset);
    (offset, offset > at(1).min(at(7)))
}

/// "UTC+2".
fn zone_name(offset_min: i32) -> String {
    let (sign, m) = (if offset_min < 0 { '-' } else { '+' }, offset_min.abs());
    if m % 60 == 0 { format!("UTC{sign}{}", m / 60) } else { format!("UTC{sign}{}:{:02}", m / 60, m % 60) }
}

/// `m` moved by `minutes` (calendar-aware).
fn shifted(m: Moment, minutes: i32) -> Moment {
    let total = days_from_civil(m.year, m.month, m.day) * 1440 + i64::from(m.hour) * 60 + i64::from(m.minute) + i64::from(minutes);
    let (days, rest) = (total.div_euclid(1440), total.rem_euclid(1440));
    let (year, month, day) = civil_from_days(days);
    Moment { year, month, day, hour: (rest / 60) as u32, minute: (rest % 60) as u32, second: m.second }
}

/// What the window needs from the open image.
pub struct IdentifyInput<'a> {
    pub path: &'a std::path::Path,
    pub source: &'a Arc<Image>,
    pub source_gen: u64,
    /// From the file name or the log, or what the pixels suggest.
    pub body: Option<Body>,
    pub time: Option<Moment>,
    /// "dal log", "dal nome del file".
    pub time_from: &'static str,
    /// The filter as the capture wrote it (`IR685`, `RGB`…).
    pub filter: Option<String>,
    /// Telescope and camera from the capture log.
    pub equipment: Option<String>,
}

impl IdentifyState {
    fn key(&self, input: &IdentifyInput) -> Option<Key> {
        Some(Key { source_gen: input.source_gen, body: self.body.or(input.body)?, when: self.time.or(input.time)?, manual: self.manual, placed: self.placed })
    }

    /// The system's zone for the capture's day: (offset in minutes, daylight saving on). Asked
    /// in the background; until the answer comes, the zone known so far.
    fn zone_for(&mut self, ctx: &egui::Context, ut: Moment) -> (i32, bool) {
        let day = days_from_civil(ut.year, ut.month, ut.day);
        if let Some((d, rx)) = &self.zone_pending
            && *d == day
            && let Ok((offset, dst)) = rx.try_recv()
        {
            self.zone_cache = Some((day, offset, dst));
            self.zone_pending = None;
        }
        match self.zone_cache {
            Some((d, offset, dst)) if d == day => (offset, dst),
            cached => {
                if self.zone_pending.as_ref().is_none_or(|(d, _)| *d != day) {
                    let (tx, rx) = channel();
                    let ctx = ctx.clone();
                    std::thread::spawn(move || {
                        tx.send(system_zone(ut)).ok();
                        ctx.request_repaint();
                    });
                    self.zone_pending = Some((day, rx));
                }
                (self.offset_min, cached.is_some_and(|(_, _, dst)| dst))
            }
        }
    }

    /// Forget a hand-set orientation and the fit (the image was turned or the subject changed).
    pub fn orientation_reset(&mut self) {
        (self.manual, self.fit_note, self.fit_applied) = (None, None, false);
        self.place_reset();
    }

    /// Forget the two craters and the disk they gave.
    fn place_reset(&mut self) {
        (self.picks, self.placed, self.place_error, self.awaiting) = ([PlacePick::default(); 2], None, None, None);
    }

    /// The image was cropped: the craters clicked on it moved by (`dx`, `dy`) pixels with it.
    pub fn image_shifted(&mut self, dx: f32, dy: f32) {
        for pick in &mut self.picks {
            pick.pixel = pick.pixel.map(|(x, y)| (x + f64::from(dx), y + f64::from(dy)));
        }
    }

    /// The top-bar button switched on: names on the image; the time and orientation are fitted
    /// to the moons as soon as the first answer is in (once per picture).
    pub fn switch_on(&mut self) {
        self.active = true;
    }

    /// Test helper: pick the subject as the user would in the tab.
    #[cfg(test)]
    pub fn set_body_for_test(&mut self, body: Body) {
        self.body = Some(body);
    }

    /// The time set by hand in the tab (UT), if any.
    pub fn chosen_time(&self) -> Option<Moment> {
        self.time
    }

    /// The subject picked by hand in the tab, if any.
    pub fn chosen_body(&self) -> Option<Body> {
        self.body
    }

    /// The viewer should take the next click as a crater's place.
    pub fn awaiting_click(&self) -> bool {
        self.open && self.awaiting.is_some()
    }

    /// The pixel the user clicked for the crater being placed.
    pub fn click(&mut self, pixel: (usize, usize)) {
        if let Some(k) = self.awaiting.take() {
            self.picks[k].pixel = Some((pixel.0 as f64, pixel.1 as f64));
        }
    }

    /// The disk from the two craters, at the time in use.
    fn solve_placement(&mut self, when: Moment) {
        (self.placed, self.place_error) = match self.picks {
            [PlacePick { name: Some(na), pixel: Some(pa) }, PlacePick { name: Some(nb), pixel: Some(pb) }] => match place_on_craters(&when, (na, pa), (nb, pb), self.place_mirror) {
                Ok(p) => (Some(p), None),
                Err(e) => (None, Some(e)),
            },
            _ => (None, None),
        };
    }

    /// Start a search if what it depends on changed; collect a finished one.
    pub fn update(&mut self, ctx: &egui::Context, input: &IdentifyInput) {
        if self.image_path.as_deref() != Some(input.path) {
            (self.body, self.time, self.manual, self.result, self.fit_note, self.zone_cache, self.fit_applied) = (None, None, None, None, None, None, false);
            self.place_reset();
            self.image_path = Some(input.path.to_path_buf());
        }
        if !self.open {
            return;
        }
        if let Some((key, rx)) = &self.pending
            && let Ok(found) = rx.try_recv()
        {
            self.result = Some((*key, Arc::new(found)));
            self.pending = None;
        }
        if let Some(fitting) = &self.fitting
            && let Ok(fit) = fitting.rx.try_recv()
        {
            let (automatic, start, upright) = (fitting.automatic, fitting.start, fitting.upright);
            self.fitting = None;
            self.fit_note = Some(match fit {
                // By itself it only acts when the moons agree: all of them, or at least two.
                Some(f) if automatic && f.matched < 2.min(f.total) => tr!("Solo {} lune su {} cadono su un punto luminoso: regola a mano o ricalcola.", "Only {} of {} moons land on a bright spot: adjust by hand or fit again.", f.matched, f.total),
                Some(f) => {
                    self.time = Some(f.when);
                    self.manual = Some(f.orientation);
                    self.fit_applied = true;
                    if upright {
                        let (mirror, degrees) = f.orientation.upright();
                        if mirror || degrees.rem_euclid(360.0).min(360.0 - degrees.rem_euclid(360.0)) >= 0.5 {
                            self.upright_pending = Some((mirror, degrees));
                        }
                    }
                    let turn = ((f.orientation.angle - start.angle + 180.0).rem_euclid(360.0)) - 180.0;
                    tr!("{} lune su {} sui punti luminosi · ora {:+.1} min · rotazione {:+.1}°", "{} of {} moons on bright spots · time {:+.1} min · rotation {:+.1}°", f.matched, f.total, f.shift_min, turn)
                }
                None => tl("Non trovo punti luminosi da far combaciare con le lune.").to_string(),
            });
        }
        if let Some(when) = self.time.or(input.time) {
            self.solve_placement(when);
        }
        let Some(key) = self.key(input) else { return };
        let current = self.result.as_ref().map(|(k, _)| *k) == Some(key) || self.pending.as_ref().map(|(k, _)| *k) == Some(key);
        if !current {
            let (tx, rx) = channel();
            let (image, ctx) = (Arc::clone(input.source), ctx.clone());
            std::thread::spawn(move || {
                tx.send(identify_with(&image, key.body, &key.when, key.manual, key.placed)).ok();
                ctx.request_repaint();
            });
            self.pending = Some((key, rx));
            return;
        }
        // The first answer for a new picture of a planet: fit time, turn and distances to the
        // moons at once, without being asked.
        if self.fitting.is_none()
            && self.pending.is_none()
            && self.fitted_path.as_deref() != Some(input.path)
            && let Some((k, id)) = &self.result
            && *k == key
            && matches!(key.body, Body::Jupiter | Body::Saturn)
        {
            self.fitted_path = Some(input.path.to_path_buf());
            if let Some(or) = id.orientation
                && id.labels.iter().any(|l| matches!(l.kind, LabelKind::MoonFound | LabelKind::MoonPredicted))
            {
                self.start_fit(ctx, input.source, key.body, key.when, or, true);
            }
        }
    }

    /// Back to the automatic way: the file's time and the orientation found, then the fit on the
    /// moons again.
    fn refit_automatically(&mut self) {
        self.time = None;
        self.orientation_reset();
        self.fitted_path = None;
    }

    fn start_fit(&mut self, ctx: &egui::Context, image: &Arc<Image>, body: Body, when: Moment, or: Orientation, automatic: bool) {
        let (tx, rx) = channel();
        let (image, ctx) = (Arc::clone(image), ctx.clone());
        let upright = self.fit_upright;
        std::thread::spawn(move || {
            tx.send(ps_core::identify::fit_time_with(&image, body, &when, or, 180.0, true)).ok();
            ctx.request_repaint();
        });
        self.fitting = Some(Fitting { automatic, upright, start: or, rx });
    }

    /// A straightening the fit asked for, once.
    pub fn take_upright(&mut self) -> Option<(bool, f32)> {
        self.upright_pending.take()
    }

    pub fn busy(&self) -> bool {
        self.pending.is_some() || self.fitting.is_some()
    }

    /// The time as written on the printed caption: UT, or local with the zone.
    fn time_text(&self, ut: Moment) -> String {
        let (shift, zone) = if self.utc { (0, "UT".to_string()) } else { (self.offset_min, format!("({})", zone_name(self.offset_min))) };
        let t = shifted(ut, shift);
        format!("{:04}-{:02}-{:02} {:02}:{:02} {zone}", t.year, t.month, t.day, t.hour, t.minute)
    }

    /// What to print onto the exported image, from the current result.
    pub fn plan(&self, input: &IdentifyInput, opt: &LabelOptions) -> Option<Plan> {
        let (key, id) = self.result.as_ref()?;
        let mut caption = Vec::new();
        let heading = Some(opt.title.trim().to_string()).filter(|t| !t.is_empty());
        if !opt.author.trim().is_empty() {
            caption.push(tr!("Foto: {}", "Photo: {}", opt.author.trim()));
        }
        if opt.object {
            caption.push(key.body.label().to_string());
        }
        if opt.filter
            && let Some(f) = &input.filter
        {
            caption.push(tr!("Filtro {f}", "Filter {f}"));
        }
        if opt.time {
            caption.push(self.time_text(key.when));
        }
        if !opt.location.trim().is_empty() {
            caption.push(opt.location.trim().to_string());
        }
        if opt.equipment
            && let Some(e) = &input.equipment
        {
            caption.push(e.clone());
        }
        // The facts are written in the language that was on when they were computed.
        let scale_keys = ["Scala dell'immagine", ps_core::i18n::with_lang(ps_core::i18n::Lang::En, || tl("Scala dell'immagine"))];
        if opt.scale
            && let Some((_, v)) = id.facts.iter().find(|(k, _)| scale_keys.contains(&k.as_str()))
        {
            caption.push(tr!("Scala {v}", "Scale {v}"));
        }
        if opt.angles
            && let Some(or) = id.orientation
        {
            caption.push(tr!("Nord a {:.0}° dall'alto · rotazione {:.0}°{}", "North {:.0}° from the top · rotation {:.0}°{}", or.north_up_angle().rem_euclid(360.0), or.angle.rem_euclid(360.0), if or.mirrored { tl(" · specchiata") } else { "" }));
        }
        if !opt.note.trim().is_empty() {
            caption.push(opt.note.trim().to_string());
        }
        Some(Plan { heading, labels: id.labels.clone(), from: (input.source.width(), input.source.height()), caption, north: id.orientation.map(Orientation::north_up_angle) })
    }

    /// The names over the image, as the «Annotazione» card says (colour, which moons, circles).
    /// Once they are printed onto the picture (the card's switch) they are not drawn twice.
    /// `to_screen` maps image pixels to the screen; `zoom` is screen pixels per image pixel.
    pub fn paint(&self, painter: &egui::Painter, to_screen: impl Fn(Pos2) -> Pos2, zoom: f32, opt: &LabelOptions) {
        if !self.open {
            return;
        }
        let Some((_, id)) = &self.result else { return };
        for (k, pick) in self.picks.iter().enumerate() {
            if let Some((x, y)) = pick.pixel {
                let at = to_screen(Pos2::new(x as f32 + 0.5, y as f32 + 0.5));
                let colour = Color32::from_rgb(255, 120, 200);
                painter.circle_stroke(at, 8.0, Stroke::new(2.0, colour));
                painter.text(at + egui::vec2(11.0, -9.0), egui::Align2::LEFT_BOTTOM, format!("{} {}", k + 1, pick.name.unwrap_or("")), egui::FontId::proportional(13.0), colour);
            }
        }
        if self.annotate {
            return;
        }
        let rgb = |c: [f32; 3]| Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8);
        let (base, predicted) = (rgb(opt.colour), rgb(opt.predicted_colour));
        let font = egui::FontId::proportional(13.0);
        let shadow = Color32::from_black_alpha(200);
        let text = |at: Pos2, s: &str, colour: Color32, anchor: egui::Align2| {
            for d in [egui::vec2(1.0, 1.0), egui::vec2(-1.0, 1.0), egui::vec2(1.0, -1.0), egui::vec2(-1.0, -1.0)] {
                painter.text(at + d, anchor, s, font.clone(), shadow);
            }
            painter.text(at, anchor, s, font.clone(), colour);
        };
        for l in &id.labels {
            let at = to_screen(Pos2::new(l.x + 0.5, l.y + 0.5));
            match l.kind {
                LabelKind::MoonFound | LabelKind::MoonPredicted => {
                    let found = l.kind == LabelKind::MoonFound;
                    if (found && !opt.moons) || (!found && !opt.predicted) {
                        continue;
                    }
                    // A moon only predicted has its own colour, and its ring is dashed.
                    let colour = if found { base } else { predicted };
                    let r = if opt.circles { (6.0 * zoom).clamp(7.0, 18.0) } else { 0.0 };
                    if opt.circles && found {
                        painter.circle_stroke(at, r, Stroke::new(1.5, colour));
                    } else if opt.circles {
                        for k in 0..12 {
                            let a0 = k as f32 / 12.0 * std::f32::consts::TAU;
                            let a1 = a0 + std::f32::consts::TAU / 24.0;
                            painter.line_segment([at + r * egui::vec2(a0.cos(), a0.sin()), at + r * egui::vec2(a1.cos(), a1.sin())], Stroke::new(1.5, colour));
                        }
                    }
                    text(at + egui::vec2(r + 4.0, 0.0), &l.name, colour, egui::Align2::LEFT_CENTER);
                }
                LabelKind::Crater => {
                    if !opt.craters {
                        continue;
                    }
                    let r = l.radius * zoom;
                    if r > 6.0 {
                        painter.circle_stroke(at, r, Stroke::new(1.0, base.gamma_multiply(0.6)));
                    }
                    if r > 3.0 || zoom >= 0.5 {
                        text(at, &l.name, base, egui::Align2::CENTER_CENTER);
                    }
                }
                LabelKind::Mare | LabelKind::Feature => {
                    if (l.kind == LabelKind::Mare && !opt.seas) || (l.kind == LabelKind::Feature && !opt.craters) {
                        continue;
                    }
                    text(at, &l.name, base, egui::Align2::CENTER_CENTER);
                }
            }
        }
    }

    /// Everything shown over the image, in one place: the names (live on screen), and, once the
    /// card's switch is on, the caption printed onto the picture with them.
    fn annotate_part(&mut self, ui: &mut Ui, opt: &mut LabelOptions) {
        let Some((key, _)) = &self.result else {
            note(ui, tl("Scegli l'oggetto e la data nella scheda Identificazione."));
            return;
        };
        let body = key.body;
        // One list of switches, in two columns.
        let mut items: Vec<(&mut bool, &str)> = Vec::new();
        let LabelOptions { moons, predicted, circles, craters, seas, object, filter, time, equipment, scale, angles, compass, .. } = opt;
        if matches!(body, Body::Jupiter | Body::Saturn) {
            items.extend([(moons, tl("Lune trovate")), (predicted, tl("Lune previste")), (circles, tl("Cerchi"))]);
        }
        if body == Body::Moon {
            items.extend([(craters, tl("Crateri")), (seas, tl("Mari"))]);
        }
        items.extend([(object, tl("Oggetto")), (filter, tl("Filtro")), (time, tl("Data e ora")), (equipment, tl("Strumento")), (scale, tl("Scala (″/pixel)")), (angles, tl("Nord e rotazione")), (compass, tl("Bussola"))]);
        ui.horizontal(|ui| {
            ui.label(caps(tl("COSA MOSTRARE")));
            crate::widgets::help_icon(ui, tl("I nomi si vedono sempre sull'immagine. Con l'interruttore della scheda acceso, nomi e didascalia vengono anche scritti sull'immagine (e nell'esportazione). Le lune «previste» sono quelle calcolate dove il punto luminoso non è stato trovato."));
        });
        egui::Grid::new("annotate-items").num_columns(2).spacing([14.0, 6.0]).show(ui, |ui| {
            for (k, (on, text)) in items.into_iter().enumerate() {
                ui.horizontal(|ui| {
                    toggle(ui, on);
                    ui.label(text);
                });
                if k % 2 == 1 {
                    ui.end_row();
                }
            }
        });

        ui.add_space(theme::space::S);
        let field = |ui: &mut Ui, label: &str, value: &mut String, hint_text: &str, width: f32| {
            ui.vertical(|ui| {
                ui.label(RichText::new(label).size(theme::size::SMALL).color(TEXT_WEAK));
                ui.add(egui::TextEdit::singleline(value).desired_width(width).hint_text(hint_text));
            });
        };
        let full = ui.available_width();
        field(ui, tl("Titolo (in grande)"), &mut opt.title, tl("Giove, 5 luglio 2019"), full);
        let half = (full - 10.0) / 2.0;
        ui.horizontal(|ui| {
            field(ui, tl("Autore"), &mut opt.author, tl("Nome e cognome"), half);
            field(ui, tl("Luogo"), &mut opt.location, tl("Città, paese"), half);
        });
        field(ui, tl("Nota"), &mut opt.note, tl("Elaborazione, condizioni…"), full);

        ui.add_space(theme::space::S);
        let planet = matches!(body, Body::Jupiter | Body::Saturn);
        colour_row(ui, if planet { tl("Colore lune trovate") } else { tl("Colore nomi") }, &mut opt.colour);
        if planet {
            colour_row(ui, tl("Colore lune previste"), &mut opt.predicted_colour);
        }
        colour_row(ui, tl("Colore del testo"), &mut opt.caption_colour);
        ui.horizontal(|ui| {
            ui.label(RichText::new(tl("Posizione")).size(theme::size::SMALL).color(TEXT_WEAK));
            egui::ComboBox::from_id_salt("caption-corner").selected_text(opt.corner.label()).show_ui(ui, |ui| {
                for c in Corner::ALL {
                    ui.selectable_value(&mut opt.corner, c, c.label());
                }
            });
        });
        ParamSlider::new(tl("Dimensione del testo"), &mut opt.size, 1.2..=5.0, 2.4).decimals(1).suffix(" %").show(ui);
    }

    /// Close-ups of the Moon without a limb: name two craters from the list and click them.
    fn place_section(&mut self, ui: &mut Ui, when: Moment, open: bool) {
        // The id is the title, as when this was a plain collapsing header: the open state is kept.
        let title = tl("POSIZIONA CON DUE CRATERI");
        crate::panels::advanced(ui, title, title, open, |ui| {
            note(ui, tl("Per i primi piani senza il bordo della Luna: scegli un cratere dalla lista e cliccalo nell'immagine, poi un secondo, il più lontano possibile dal primo. Da quei due punti ricavo scala, rotazione e posizione, e scrivo i nomi di tutti gli altri crateri."));
            let view = selene::view(&when);
            let mut list: Vec<&'static str> = selene::features().filter(|f| f.kind == selene::FeatureKind::Crater && view.project(f.lon, f.lat).2 && view.sun_altitude(f.lon, f.lat) > -1.0).map(|f| f.name).collect();
            list.sort_unstable();
            ui.add_space(theme::space::XS);
            for k in 0..2 {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}", k + 1)).strong().color(Color32::from_rgb(255, 120, 200)));
                    let pick = &mut self.picks[k];
                    egui::ComboBox::from_id_salt(("place-crater", k)).width(130.0).selected_text(pick.name.unwrap_or(tl("Cratere…"))).show_ui(ui, |ui| {
                        for n in &list {
                            ui.selectable_value(&mut pick.name, Some(*n), *n);
                        }
                    });
                    let waiting = self.awaiting == Some(k);
                    let text = if waiting {
                        tl("Clicca nell'immagine…")
                    } else if pick.pixel.is_some() {
                        tl("Rifai il clic")
                    } else {
                        tl("Clicca nell'immagine")
                    };
                    if ui.add_enabled(pick.name.is_some(), egui::Button::new(text).selected(waiting)).on_hover_text(tl("Poi clicca il centro del cratere sull'immagine")).clicked() {
                        self.awaiting = if waiting { None } else { Some(k) };
                    }
                });
            }
            ui.horizontal(|ui| {
                toggle(ui, &mut self.place_mirror);
                ui.label(tl("Specchiata")).on_hover_text(tl("Due punti non dicono se l'immagine è ribaltata: se i nomi cadono sui crateri sbagliati, prova a cambiare"));
                if ui.add_enabled_ui(self.picks.iter().any(|p| p.name.is_some() || p.pixel.is_some()), |ui| icon_button(ui, ICON_RESTART_ALT, tl("Azzera"))).inner.clicked() {
                    self.place_reset();
                }
            });
            if let Some(e) = self.place_error {
                ui.add(egui::Label::new(RichText::new(e).size(theme::size::SMALL).color(theme::WARNING)).wrap());
            } else if self.placed.is_some() {
                note(ui, tl("Collocata. Se un nome cade accanto al cratere giusto ma non sopra, il catalogo è approssimato (circa 30 km)."));
            }
        });
    }

    /// The tab's content.
    pub fn tab(&mut self, ui: &mut Ui, input: &IdentifyInput, opt: &mut LabelOptions, actions: &mut Vec<PanelAction>) {
        // Two cards like the other tabs': what the picture shows, and what is shown over it.
        let mut folded = self.identify_collapsed;
        crate::panels::section_card(ui, ICON_TRAVEL_EXPLORE, tl("Identificazione"), &mut folded, |ui| self.identify_part(ui, input, actions, opt));
        self.identify_collapsed = folded;
        let (mut folded, mut on) = (self.annotate_collapsed, self.annotate);
        crate::panels::switch_card(ui, ICON_EDIT_NOTE, tl("Annotazione"), &mut on, &mut folded, |ui| self.annotate_part(ui, opt));
        (self.annotate_collapsed, self.annotate) = (folded, on);
    }

    /// Object, date, orientation: what puts the moons and craters on their places.
    fn identify_part(&mut self, ui: &mut Ui, input: &IdentifyInput, actions: &mut Vec<PanelAction>, opt: &LabelOptions) {
        let ctx = ui.ctx().clone();
        let section = |ui: &mut Ui, title: &str| {
            ui.add_space(theme::space::M);
            ui.label(caps(title));
            ui.add_space(theme::space::XS);
        };

        // The object, and on the right what the marks on the image mean.
        let mut body = self.body.or(input.body);
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("identify-body").width(110.0).selected_text(body.map_or(tl("Oggetto…"), Body::label)).show_ui(ui, |ui| {
                for b in Body::ALL {
                    ui.selectable_value(&mut body, Some(b), b.label());
                }
            });
            help(ui, tl("Il soggetto ripreso: riconosciuto dal nome del file, dal log di ripresa o da com'è fatta l'immagine. Cambialo se non è giusto."));
            if matches!(body, Some(Body::Jupiter | Body::Saturn)) {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let rgb = |c: [f32; 3]| Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8);
                    ui.label(RichText::new(format!("◌ {}", tl("prevista"))).size(theme::size::SMALL).color(rgb(opt.predicted_colour)));
                    ui.label(RichText::new(format!("◯ {}", tl("trovata"))).size(theme::size::SMALL).color(rgb(opt.colour)));
                });
            }
        });
        if body != self.body.or(input.body) {
            self.body = body;
            self.orientation_reset();
        }

        // Date and time on one row: zone, calendar, clock.
        ui.add_space(theme::space::S);
        let ut_before = self.time.or(input.time);
        // The system's zone for this very date (daylight saving included), unless set by hand.
        let mut dst = false;
        if self.zone_auto
            && let Some(ut) = ut_before
        {
            (self.offset_min, dst) = self.zone_for(&ctx, ut);
        }
        let (utc_before, offset_before) = (self.utc, self.offset_min);
        let shift = if self.utc { 0 } else { self.offset_min };
        let ut = self.time.or(input.time).unwrap_or(Moment { year: 2026, month: 1, day: 1, hour: 0, minute: 0, second: 0.0 });
        let mut when = shifted(ut, shift);
        let before = when;
        let known = self.time.is_some() || input.time.is_some();
        let source = if self.fit_applied {
            tl("regolata sulle lune")
        } else if self.time.is_some() {
            tl("inserita a mano")
        } else if input.time.is_some() {
            input.time_from
        } else {
            tl("manca")
        };
        ui.horizontal_wrapped(|ui| {
            ui.scope(|ui| segmented(ui, &mut self.utc, &[(true, None, "UTC"), (false, None, tl("Locale"))]))
                .response
                .on_hover_text(if self.utc { tl("Tempo universale: quello che servono le effemeridi") } else { tl("La data nel fuso del computer: viene convertita in UTC da sola") });
            date_picker(ui, "identify-date", &mut when.year, &mut when.month, &mut when.day);
            time_field(ui, "identify-date", &mut when.hour, &mut when.minute);
            if !known {
                ui.label(RichText::new(tl("data mancante")).size(theme::size::SMALL).color(theme::WARNING));
            } else if self.time.is_some() && input.time.is_some() && icon_button(ui, ICON_RESTART_ALT, tl("Torna alla data e ora del file")).clicked() {
                self.time = None;
                self.orientation_reset();
                self.fitted_path = None;
            }
            let ut_now = shifted(when, -shift);
            help(
                ui,
                &tr!(
                    "Data e ora della ripresa ({}).\nUsate per il calcolo: {:04}-{:02}-{:02} {:02}:{:02} UTC.\nUTC è il tempo universale: in Italia è l'ora solare meno 1 ora, o l'ora legale meno 2.",
                    "Date and time of the capture ({}).\nUsed for the calculation: {:04}-{:02}-{:02} {:02}:{:02} UTC.\nUTC is universal time: in Italy it is standard time minus 1 hour, or summer time minus 2.",
                    source,
                    ut_now.year,
                    ut_now.month,
                    ut_now.day,
                    ut_now.hour,
                    ut_now.minute
                ),
            );
            if !self.utc {
                if self.zone_auto {
                    let name = format!("{} · {}", zone_name(self.offset_min), if dst { tl("legale") } else { tl("solare") });
                    if ui.add(egui::Button::new(RichText::new(name).size(theme::size::SMALL).color(TEXT_WEAK)).frame(false)).on_hover_text(tl("Fuso scelto dal computer per questa data. Clic per impostarlo a mano (se il computer è in un altro paese rispetto alla ripresa)")).clicked() {
                        self.zone_auto = false;
                    }
                } else {
                    let mut hours = self.offset_min as f32 / 60.0;
                    if ui.add(egui::DragValue::new(&mut hours).range(-12.0..=14.0).speed(0.25).max_decimals(2).prefix("UTC+").suffix(" h")).on_hover_text(tl("Fuso orario, ora legale compresa (Italia: +1 inverno, +2 estate)")).changed() {
                        self.offset_min = (hours * 60.0).round() as i32;
                    }
                    if icon_button(ui, ICON_AUTO_FIX_HIGH, tl("Torna al fuso del computer, con ora legale o solare secondo la data")).clicked() {
                        self.zone_auto = true;
                    }
                }
            }
        });
        let shift = if self.utc { 0 } else { self.offset_min };
        // Changing the zone keeps the numbers on screen and reads them in the new one.
        let zone_changed = self.utc != utc_before || self.offset_min != offset_before;
        if when != before || (zone_changed && ut_before.is_some()) {
            self.time = Some(shifted(when, -shift));
        }
        // A slider to move the time a little each way and watch the moons go, to see whether
        // the evening and the moons are the right ones.
        if known {
            let now = self.time.or(input.time);
            if self.scrub_set != now {
                (self.scrub_base, self.scrub_min) = (now, 0.0);
            }
            if let Some(base) = self.scrub_base {
                let before = self.scrub_min;
                ParamSlider::new(tl("Sposta l'ora"), &mut self.scrub_min, -120.0..=120.0, 0.0)
                    .decimals(0)
                    .suffix(" min")
                    .help(tl("Muove l'ora fino a due ore indietro o avanti: guarda le lune spostarsi e controlla che la serata e le lune siano quelle giuste. Doppio clic: torna all'ora di partenza"))
                    .show(ui);
                if self.scrub_min != before {
                    let t = shifted(base, self.scrub_min.round() as i32);
                    (self.time, self.scrub_set) = (Some(t), Some(t));
                }
            }
        }

        let Some((key, id)) = &self.result else {
            ui.add_space(theme::space::M);
            if self.pending.is_some() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(tl("Calcolo…"));
                });
            } else if body.is_none() {
                note(ui, tl("Scegli l'oggetto ripreso."));
            } else if !known {
                note(ui, tl("Imposta data e ora della ripresa per calcolare le effemeridi."));
            }
            return;
        };
        let (key, id) = (*key, Arc::clone(id));
        let can_fit = matches!(key.body, Body::Jupiter | Body::Saturn) && id.labels.iter().any(|l| matches!(l.kind, LabelKind::MoonFound | LabelKind::MoonPredicted));

        if let Some(or) = id.orientation {
            section(ui, tl("ORIENTAMENTO"));
            let locked = self.placed.is_some();
            if locked {
                ui.label(RichText::new(tl("Dai due crateri indicati")).color(TEXT));
            } else {
                // Automatic (found in the picture, fitted to the moons) or by hand.
                let mut by_hand = self.manual.is_some() && !self.fit_applied;
                let was = by_hand;
                ui.horizontal(|ui| {
                    let auto_label = if can_fit { tl("Sulle lune") } else { tl("Automatico") };
                    segmented(ui, &mut by_hand, &[(false, Some(ICON_AUTO_FIX_HIGH), auto_label), (true, Some(ICON_TUNE), tl("A mano"))]);
                    if self.fitting.is_some() {
                        ui.spinner();
                    } else if can_fit
                        && !by_hand
                        && icon_button(ui, ICON_REFRESH, tl("Cerca di nuovo ora, rotazione e distanza delle lune, in modo che tutte le lune calcolate cadano sui punti luminosi dell'immagine (ora entro ±3 ore)")).clicked()
                    {
                        self.refit_automatically();
                    }
                    help(ui, tl("Sulle lune: cerco insieme l'ora, una piccola rotazione e la distanza delle lune, così che tutte le lune calcolate cadano sui punti luminosi. A mano: regoli tu l'orientamento."));
                });
                if by_hand != was {
                    if by_hand {
                        self.manual = Some(self.manual.unwrap_or(or));
                        self.fit_applied = false;
                    } else if can_fit {
                        self.refit_automatically();
                    } else {
                        (self.manual, self.fit_applied) = (None, false);
                    }
                }
                if by_hand {
                    let mut manual = self.manual.unwrap_or(or);
                    let mut angle = ((manual.angle + 180.0).rem_euclid(360.0)) - 180.0;
                    let mut changed = ParamSlider::new(tl("Rotazione"), &mut angle, -180.0..=180.0, 0.0).decimals(1).suffix("°").show(ui);
                    ui.horizontal(|ui| {
                        if ui.button((ICON_ROTATE_RIGHT.rich_text(), tl("Ruota di 180°"))).clicked() {
                            angle += 180.0;
                            changed = true;
                        }
                        changed |= toggle(ui, &mut manual.mirrored).changed();
                        ui.label(tl("Specchiata")).on_hover_text(tl("Immagine ribaltata (diagonale, Newton): est e ovest scambiati"));
                    });
                    if changed {
                        manual.angle = angle.rem_euclid(360.0);
                        self.manual = Some(manual);
                        self.fit_applied = false;
                    }
                } else if let Some(text) = &self.fit_note {
                    note(ui, text);
                } else if !can_fit {
                    note(ui, if id.orientation_found { tl("Trovato nell'immagine") } else { tl("Stimato") });
                }
                if can_fit && !by_hand {
                    ui.horizontal(|ui| {
                        toggle(ui, &mut self.fit_upright);
                        ui.label(tl("Raddrizza dopo la regolazione")).on_hover_text(tl("Dopo la regolazione sulle lune ruota l'immagine con il nord in alto"));
                    });
                }
            }
            // North, and straightening the picture.
            let current = if locked { or } else { self.manual.unwrap_or(or) };
            let (mirror, degrees) = current.upright();
            let straight = !mirror && degrees.rem_euclid(360.0).min(360.0 - degrees.rem_euclid(360.0)) < 0.5;
            ui.add_space(theme::space::XS);
            ui.horizontal(|ui| {
                let text = if straight { tl("Nord in alto").to_string() } else { tr!("Nord a {:.0}° dall'alto{}", "North {:.0}° from the top{}", current.north_up_angle().rem_euclid(360.0), if current.mirrored { tl(" · specchiata") } else { "" }) };
                ui.label(RichText::new(text).size(theme::size::SMALL).color(TEXT_WEAK));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(!straight, egui::Button::new((ICON_NORTH.rich_text(), tl("Raddrizza"))))
                        .on_hover_text(tl("Ruota l'immagine (e la specchia, se serve) così che il nord stia in alto e le lune dal lato giusto, come il cielo a occhio nudo. Il ritaglio si azzera."))
                        .clicked()
                    {
                        actions.push(PanelAction::NorthUp { mirror, degrees });
                    }
                });
            });
        }

        if key.body == Body::Moon {
            self.place_section(ui, key.when, id.disk.is_none() || self.placed.is_some());
        }

        let facts = id.facts.clone();
        let notes = id.notes.clone();
        crate::panels::advanced(ui, "identify-advanced", tl("AVANZATE: EFFEMERIDI E METODO"), false, |ui| {
            egui::Grid::new("identify-facts").num_columns(2).spacing([12.0, 4.0]).striped(true).show(ui, |ui| {
                for (k, v) in &facts {
                    ui.label(RichText::new(k).color(TEXT_WEAK));
                    ui.add(egui::Label::new(RichText::new(v).color(TEXT)).wrap());
                    ui.end_row();
                }
            });
            if !notes.is_empty() {
                ui.add_space(theme::space::S);
                ui.label(caps(tl("COME È STATO TROVATO")));
                for n in &notes {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(ICON_INFO.rich_text().color(ACCENT));
                        ui.label(RichText::new(n).size(theme::size::SMALL).color(TEXT_WEAK));
                    });
                }
            }
        });
        if self.pending.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(tl("Aggiorno…")).size(theme::size::SMALL).color(TEXT_WEAK));
            });
        }
    }
}

/// A small «?» whose tooltip holds the explanation that would otherwise take room in the tab.
fn help(ui: &mut Ui, text: &str) {
    ui.label(ICON_HELP.rich_text().size(theme::size::ICON_SMALL).color(TEXT_WEAK)).on_hover_text(text);
}

/// A row of colour swatches and a free colour.
fn colour_row(ui: &mut Ui, label: &str, colour: &mut [f32; 3]) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(label).size(theme::size::SMALL).color(TEXT_WEAK));
        for t in Tint::ALL {
            let c = t.rgb();
            let swatch = Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8);
            let chosen = (0..3).all(|k| (colour[k] - c[k]).abs() < 0.004);
            let (rect, response) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::click());
            ui.painter().rect_filled(rect, 4.0, swatch);
            if chosen {
                ui.painter().rect_stroke(rect.expand(2.0), 5.0, Stroke::new(2.0, ACCENT), egui::StrokeKind::Outside);
            }
            if response.on_hover_text(t.label()).clicked() {
                *colour = c;
            }
        }
        ui.color_edit_button_rgb(colour).on_hover_text(tl("Un altro colore"));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_system_zone_for_a_date_is_plausible() {
        let summer = system_offset_at(Moment { year: 2026, month: 7, day: 15, hour: 12, minute: 0, second: 0.0 }).expect("the system answers");
        let winter = system_offset_at(Moment { year: 2026, month: 1, day: 15, hour: 12, minute: 0, second: 0.0 }).expect("the system answers");
        assert!((-720..=840).contains(&summer) && (-720..=840).contains(&winter));
        // Zones are whole quarter hours, and daylight saving only ever moves the clock forward in summer (north).
        assert!(summer % 15 == 0 && winter % 15 == 0);
        assert_eq!(zone_name(120), "UTC+2");
        assert_eq!(zone_name(-210), "UTC-3:30");
    }
}
