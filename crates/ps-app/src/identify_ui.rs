//! "Identifica": what the ephemerides say is on the image — moons of Jupiter and Saturn, the
//! Moon's craters and seas, the Sun's orientation — in a tab of the side panel, with the names
//! over the image.
//!
//! The search runs off the interface thread ([`ps_core::identify::identify`]); the tab lets the
//! user correct the subject, the time and the orientation, and every change searches again. The
//! local time zone (daylight saving included, for the capture's own date) comes from the system;
//! for Jupiter and Saturn the time, turn and moons' distances are fitted to the moons found in
//! the picture as soon as it is open. The names can be printed onto the exported image
//! ([`crate::annotate`]).

use ps_core::i18n::tl;
use ps_core::tr;
use crate::annotate::{Corner, LabelOptions, Plan, Tint};
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

/// The two halves of the tab: finding the object and its orientation, and writing on the image.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Sub {
    #[default]
    Identify,
    Annotate,
}

pub struct IdentifyState {
    /// Which half of the tab is showing.
    pub sub: Sub,
    /// The names and caption are written on the image (a layer among the masks).
    pub annotate: bool,
    /// The top-bar button: identify the subject and show the names on the image.
    pub active: bool,
    /// The search runs and the names show: the button is on or the tab is the one showing.
    pub open: bool,
    pub show_labels: bool,
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
    /// The top-bar button was switched on: fit time and orientation to the moons as soon as the
    /// first answer is in, as if «Regola sulle lune» had been pressed.
    fit_requested: bool,
    /// «Regola sulle lune» also turns and stretches (otherwise only the time moves), and then
    /// straightens the image to north up.
    fit_orientation: bool,
    fit_upright: bool,
    /// The fit asked to straighten: taken by the app, which turns the image.
    upright_pending: Option<(bool, f32)>,
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
        Self { sub: Sub::Identify, annotate: false, active: false, open: false, show_labels: true, utc: true, zone_auto: true, offset_min: 0, zone_cache: None, zone_pending: None, fitted_path: None, fit_applied: false, fit_requested: false, fit_orientation: true, fit_upright: false, upright_pending: None, body: None, time: None, manual: None, picks: [PlacePick::default(); 2], placed: None, place_error: None, awaiting: None, place_mirror: false, result: None, pending: None, fitting: None, fit_note: None, image_path: None }
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

/// A reminder when nothing but the names would be printed.
fn hint_if_empty(ui: &mut Ui, opt: &LabelOptions) {
    if opt.title.is_empty() && opt.author.is_empty() && opt.location.is_empty() && opt.note.is_empty() {
        note(ui, tl("Titolo, autore, luogo e nota restano fuori finché sono vuoti."));
    }
}

/// "UTC+2".
fn zone_name(offset_min: i32) -> String {
    let (sign, m) = (if offset_min < 0 { '-' } else { '+' }, offset_min.abs());
    if m % 60 == 0 { format!("UTC{sign}{}", m / 60) } else { format!("UTC{sign}{}:{:02}", m / 60, m % 60) }
}

fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = i64::from(y) - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (i64::from(m) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    ((yoe + era * 400 + i64::from(m <= 2)) as i32, m, d)
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

    /// The top-bar button switched on: names on the image, and the time and orientation fitted
    /// to the moons as soon as the first answer is in.
    pub fn switch_on(&mut self) {
        self.active = true;
        self.fit_requested = true;
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
                Some(f) if automatic && f.matched < 2 => tr!("Troppo poche lune su un punto luminoso ({} su {}) per regolare da sola: premi «Regola sulle lune» se vuoi provare.", "Too few moons land on a bright spot ({} of {}) to adjust automatically: press “Fit to moons” if you want to try.", f.matched, f.total),
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
                    tr!("Regolato su tutte le lune insieme: {} su {} cadono su un punto luminoso (ora {:+.1} min, rotazione {:+.1}°, distanze {:+.1} %).", "Fitted to all the moons together: {} of {} land on a bright spot (time {:+.1} min, rotation {:+.1}°, distances {:+.1} %).",
                        f.matched,
                        f.total,
                        f.shift_min,
                        turn,
                        (f.orientation.scale / start.scale - 1.0) * 100.0
                    )
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
        // Switched on from the top bar: fit time and orientation to the moons, as the button would.
        if self.fit_requested
            && self.fitting.is_none()
            && self.pending.is_none()
            && let Some((k, id)) = &self.result
            && *k == key
        {
            self.fit_requested = false;
            if matches!(key.body, Body::Jupiter | Body::Saturn)
                && let Some(or) = id.orientation
                && id.labels.iter().any(|l| matches!(l.kind, LabelKind::MoonFound | LabelKind::MoonPredicted))
            {
                self.fitted_path = Some(input.path.to_path_buf());
                self.start_fit(ctx, input.source, key.body, key.when, or, false);
                return;
            }
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

    fn start_fit(&mut self, ctx: &egui::Context, image: &Arc<Image>, body: Body, when: Moment, or: Orientation, automatic: bool) {
        let (tx, rx) = channel();
        let (image, ctx) = (Arc::clone(image), ctx.clone());
        let free = self.fit_orientation || automatic;
        let upright = self.fit_upright && !automatic;
        std::thread::spawn(move || {
            tx.send(ps_core::identify::fit_time_with(&image, body, &when, or, 180.0, free)).ok();
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

    /// The names over the image. `to_screen` maps image pixels to the screen; `zoom` is screen
    /// pixels per image pixel.
    pub fn paint(&self, painter: &egui::Painter, to_screen: impl Fn(Pos2) -> Pos2, zoom: f32, opt: &LabelOptions) {
        if !self.open || !self.show_labels {
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
                    let colour = if found { Color32::from_rgb(120, 230, 140) } else { Color32::from_rgb(255, 190, 90) };
                    let r = (6.0 * zoom).clamp(7.0, 18.0);
                    if found {
                        painter.circle_stroke(at, r, Stroke::new(1.5, colour));
                    } else {
                        // Dashed: where it should be.
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
                    let colour = if l.on_terminator { Color32::from_rgb(255, 225, 120) } else { Color32::from_rgb(235, 235, 235) };
                    let r = l.radius * zoom;
                    if r > 6.0 {
                        painter.circle_stroke(at, r, Stroke::new(1.0, colour.gamma_multiply(0.6)));
                    }
                    if r > 3.0 || zoom >= 0.5 {
                        text(at, &l.name, colour, egui::Align2::CENTER_CENTER);
                    }
                }
                LabelKind::Mare | LabelKind::Feature => {
                    if (l.kind == LabelKind::Mare && !opt.seas) || (l.kind == LabelKind::Feature && !opt.craters) {
                        continue;
                    }
                    let colour = Color32::from_rgb(140, 200, 255);
                    text(at, &l.name, colour, egui::Align2::CENTER_CENTER);
                }
            }
        }
    }

    /// The names and the caption written on the image, live, as a layer among the masks.
    fn annotate_part(&mut self, ui: &mut Ui, opt: &mut LabelOptions) {
        let section = |ui: &mut Ui, title: &str| {
            ui.add_space(theme::space::L);
            ui.label(caps(title));
            ui.add_space(theme::space::XS);
        };
        ui.add_space(theme::space::M);
        ui.horizontal(|ui| {
            toggle(ui, &mut self.annotate);
            ui.label(RichText::new(tl("Annota l'immagine")).strong().color(TEXT));
        });
        note(ui, if self.annotate { tl("Il testo è sull'immagine e si aggiorna mentre cambi le opzioni. Lo trovi tra le maschere come «Annotazione»: l'esportazione normale lo include.") } else { tl("Accendi per scrivere nomi e didascalia sull'immagine: compaiono subito e diventano una maschera «Annotazione».") });
        let Some((key, _)) = &self.result else {
            note(ui, tl("Prima scegli l'oggetto e la data nella scheda Identificazione."));
            return;
        };
        let key = *key;
        section(ui, tl("NOMI SULL'IMMAGINE"));
        // Switches in two columns.
        let switches = |ui: &mut Ui, id: &str, items: Vec<(&mut bool, &str)>| {
            egui::Grid::new(id).num_columns(2).spacing([14.0, 6.0]).show(ui, |ui| {
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
        };
        if matches!(key.body, Body::Jupiter | Body::Saturn) {
            switches(ui, "names-moons", vec![(&mut opt.moons, tl("Lune trovate")), (&mut opt.predicted, tl("Anche solo previste")), (&mut opt.circles, tl("Cerchi"))]);
        }
        if key.body == Body::Moon {
            switches(ui, "names-moon", vec![(&mut opt.craters, tl("Crateri")), (&mut opt.seas, tl("Mari"))]);
            note(ui, tl("In giallo i crateri sul terminatore."));
        }
        let colour_row = |ui: &mut Ui, label: &str, colour: &mut [f32; 3]| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(label).size(theme::size::SMALL).color(TEXT_WEAK));
                for t in Tint::ALL {
                    let c = t.rgb();
                    let swatch = Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8);
                    let chosen = *colour == c;
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
        };
        colour_row(ui, tl("Colore"), &mut opt.colour);

        section(ui, tl("DIDASCALIA"));
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
        field(ui, tl("Nota"), &mut opt.note, tl("Strumento, elaborazione…"), full);
        hint_if_empty(ui, opt);
        ui.add_space(theme::space::XS);
        ui.label(RichText::new(tl("Cosa scrivere")).size(theme::size::SMALL).color(TEXT_WEAK));
        switches(
            ui,
            "caption-items",
            vec![
                (&mut opt.object, tl("Oggetto")),
                (&mut opt.filter, tl("Filtro")),
                (&mut opt.time, tl("Data e ora")),
                (&mut opt.equipment, tl("Telescopio e camera")),
                (&mut opt.scale, tl("Scala (″/pixel)")),
                (&mut opt.angles, tl("Nord e rotazione")),
                (&mut opt.compass, tl("Bussola")),
            ],
        );
        ui.add_space(theme::space::XS);
        ui.horizontal(|ui| {
            ui.label(RichText::new(tl("Posizione")).size(theme::size::SMALL).color(TEXT_WEAK));
            egui::ComboBox::from_id_salt("caption-corner").selected_text(opt.corner.label()).show_ui(ui, |ui| {
                for c in Corner::ALL {
                    ui.selectable_value(&mut opt.corner, c, c.label());
                }
            });
        });
        colour_row(ui, tl("Colore del testo"), &mut opt.caption_colour);
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
        ui.scope(|ui| segmented(ui, &mut self.sub, &[(Sub::Identify, Some(ICON_TRAVEL_EXPLORE), tl("Identificazione")), (Sub::Annotate, Some(ICON_EDIT_NOTE), tl("Annotazione"))]));
        match self.sub {
            Sub::Identify => self.identify_part(ui, input, actions),
            Sub::Annotate => self.annotate_part(ui, opt),
        }
    }

    /// Object, date, orientation: what puts the moons and craters on their places.
    fn identify_part(&mut self, ui: &mut Ui, input: &IdentifyInput, actions: &mut Vec<PanelAction>) {
        let ctx = ui.ctx().clone();
        let section = |ui: &mut Ui, title: &str| {
            ui.add_space(theme::space::L);
            ui.label(caps(title));
            ui.add_space(theme::space::XS);
        };
        section(ui, tl("OGGETTO"));
        let mut body = self.body.or(input.body);
        egui::ComboBox::from_id_salt("identify-body").selected_text(body.map_or(tl("Scegli…"), Body::label)).show_ui(ui, |ui| {
            for b in Body::ALL {
                ui.selectable_value(&mut body, Some(b), b.label());
            }
        });
        if body != self.body.or(input.body) {
            self.body = body;
            self.orientation_reset();
        }

        section(ui, tl("DATA E ORA"));
        let ut_before = self.time.or(input.time);
        // The system's zone for this very date (daylight saving included), unless set by hand.
        let mut dst = false;
        if self.zone_auto
            && let Some(ut) = ut_before
        {
            (self.offset_min, dst) = self.zone_for(&ctx, ut);
        }
        let (utc_before, offset_before) = (self.utc, self.offset_min);
        let tip = if self.utc { tl("Tempo universale: quello che servono le effemeridi") } else { tl("La data nel fuso del computer: viene convertita in UTC da sola") };
        let shift = if self.utc { 0 } else { self.offset_min };
        let ut = self.time.or(input.time).unwrap_or(Moment { year: 2026, month: 1, day: 1, hour: 0, minute: 0, second: 0.0 });
        let mut when = shifted(ut, shift);
        let before = when;
        // The zone selector and the date on one row.
        ui.horizontal_wrapped(|ui| {
            ui.scope(|ui| segmented(ui, &mut self.utc, &[(true, None, "UTC"), (false, None, tl("Ora locale"))])).response.on_hover_text(tip);
            ui.add_space(theme::space::S);
            ui.add(egui::DragValue::new(&mut when.year).range(1900..=2100));
            ui.label("-");
            ui.add(egui::DragValue::new(&mut when.month).range(1..=12));
            ui.label("-");
            ui.add(egui::DragValue::new(&mut when.day).range(1..=31));
            ui.add_space(theme::space::M);
            ui.add(egui::DragValue::new(&mut when.hour).range(0..=23));
            ui.label(":");
            ui.add(egui::DragValue::new(&mut when.minute).range(0..=59));
        });
        if !self.utc {
            ui.horizontal(|ui| {
                if self.zone_auto {
                    ui.label(RichText::new(format!("{} · {}", zone_name(self.offset_min), if dst { tl("ora legale") } else { tl("ora solare") })).color(TEXT));
                    ui.label(RichText::new(tl("scelto dal computer per questa data")).size(theme::size::SMALL).color(TEXT_WEAK));
                    if ui.button((ICON_EDIT.rich_text(), tl("Modifica"))).on_hover_text(tl("Imposta il fuso a mano (se il computer è in un altro paese rispetto alla ripresa)")).clicked() {
                        self.zone_auto = false;
                    }
                } else {
                    ui.label(RichText::new("UTC").size(theme::size::SMALL).color(TEXT_WEAK));
                    let mut hours = self.offset_min as f32 / 60.0;
                    if ui.add(egui::DragValue::new(&mut hours).range(-12.0..=14.0).speed(0.25).max_decimals(2).prefix("+").suffix(" h")).on_hover_text(tl("Fuso orario, ora legale compresa (Italia: +1 inverno, +2 estate)")).changed() {
                        self.offset_min = (hours * 60.0).round() as i32;
                    }
                    if ui.button((ICON_AUTO_FIX_HIGH.rich_text(), tl("Automatico"))).on_hover_text(tl("Torna al fuso del computer, con ora legale o solare secondo la data")).clicked() {
                        self.zone_auto = true;
                    }
                }
            });
        }
        let shift = if self.utc { 0 } else { self.offset_min };
        // Changing the zone keeps the numbers on screen and reads them in the new one.
        let zone_changed = self.utc != utc_before || self.offset_min != offset_before;
        if when != before || (zone_changed && ut_before.is_some()) || (self.time.is_none() && input.time.is_none() && ui.button(tl("Usa questa data")).clicked()) {
            self.time = Some(shifted(when, -shift));
        }
        let source = if self.fit_applied {
            tl("regolata sulle lune")
        } else if self.time.is_some() {
            tl("inserita a mano")
        } else if input.time.is_some() {
            input.time_from
        } else {
            tl("manca: inseriscila per calcolare le effemeridi")
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(source).size(theme::size::SMALL).color(if input.time.is_none() && self.time.is_none() { theme::WARNING } else { TEXT_WEAK }));
            if self.time.is_some() && input.time.is_some() && ui.button((ICON_RESTART_ALT.rich_text(), tl("Torna a quella del file"))).clicked() {
                self.time = None;
                self.orientation_reset();
            }
        });
        if self.utc {
            note(ui, tl("UTC è il tempo universale: in Italia è l'ora solare meno 1 ora, o l'ora legale meno 2."));
        } else {
            let t = shifted(ut, 0);
            note(ui, &tr!("Corrisponde a {:04}-{:02}-{:02} {:02}:{:02} UTC, usato per il calcolo.", "Corresponds to {:04}-{:02}-{:02} {:02}:{:02} UTC, used for the calculation.", t.year, t.month, t.day, t.hour, t.minute));
        }

        let Some((key, id)) = &self.result else {
            ui.add_space(theme::space::L);
            if self.pending.is_some() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(tl("Calcolo…"));
                });
            } else if body.is_none() {
                note(ui, tl("Scegli l'oggetto ripreso."));
            }
            return;
        };
        let (key, id) = (*key, Arc::clone(id));

        if matches!(key.body, Body::Jupiter | Body::Saturn) && id.labels.iter().any(|l| matches!(l.kind, LabelKind::MoonFound | LabelKind::MoonPredicted)) {
            ui.add_space(theme::space::S);
            ui.horizontal_wrapped(|ui| {
                let can = id.orientation.is_some() && self.fitting.is_none();
                if ui
                    .add_enabled(can, egui::Button::new((ICON_SCHEDULE.rich_text(), tl("Regola sulle lune"))))
                    .on_hover_text(tl("Cerca insieme l'ora (entro ±3 ore), una piccola rotazione e la distanza delle lune, in modo che TUTTE le lune calcolate cadano sui punti luminosi dell'immagine"))
                    .clicked()
                    && let Some(or) = id.orientation
                {
                    self.start_fit(&ctx, input.source, key.body, key.when, or, false);
                }
                toggle(ui, &mut self.fit_orientation);
                ui.label(tl("Orientamento")).on_hover_text(tl("Regola anche la rotazione e la distanza delle lune; spento, si muove solo l'ora"));
                toggle(ui, &mut self.fit_upright);
                ui.label(tl("Raddrizza")).on_hover_text(tl("Dopo la regolazione ruota l'immagine con il nord in alto"));
                if self.fitting.is_some() {
                    ui.spinner();
                }
            });
            if let Some(text) = &self.fit_note {
                note(ui, text);
            }
        }

        section(ui, tl("SULL'IMMAGINE"));
        ui.horizontal(|ui| {
            toggle(ui, &mut self.show_labels);
            ui.label(tl("Mostra i nomi sull'immagine"));
        });
        if matches!(key.body, Body::Jupiter | Body::Saturn) {
            ui.horizontal(|ui| {
                ui.label(RichText::new("◯").color(Color32::from_rgb(120, 230, 140)));
                ui.label(RichText::new(tl("trovata nell'immagine")).size(theme::size::SMALL).color(TEXT_WEAK));
                ui.label(RichText::new("◌").color(Color32::from_rgb(255, 190, 90)));
                ui.label(RichText::new(tl("posizione prevista")).size(theme::size::SMALL).color(TEXT_WEAK));
            });
        }

        if let Some(or) = id.orientation {
            section(ui, tl("ORIENTAMENTO"));
            let locked = self.placed.is_some();
            let mut automatic = self.manual.is_none();
            if locked {
                ui.label(tl("Dai due crateri indicati"));
            }
            if !locked {
                ui.horizontal(|ui| {
                    if toggle(ui, &mut automatic).changed() {
                        self.manual = if automatic { None } else { Some(or) };
                        self.fit_applied = false;
                    }
                    ui.label(if self.fit_applied {
                        tl("Regolato sulle lune")
                    } else if automatic {
                        if id.orientation_found { tl("Trovato nell'immagine") } else { tl("Stimato") }
                    } else {
                        tl("Impostato a mano")
                    });
                });
            }
            let mut manual = if locked { or } else { self.manual.unwrap_or(or) };
            let mut angle = ((manual.angle + 180.0).rem_euclid(360.0)) - 180.0;
            let mut changed = false;
            if !locked {
                changed = ParamSlider::new(tl("Rotazione"), &mut angle, -180.0..=180.0, 0.0).decimals(1).suffix("°").show(ui);
                ui.horizontal(|ui| {
                    if ui.button((ICON_ROTATE_RIGHT.rich_text(), tl("Ruota di 180°"))).clicked() {
                        angle += 180.0;
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    changed |= toggle(ui, &mut manual.mirrored).changed();
                    ui.label(tl("Specchiata")).on_hover_text(tl("Immagine ribaltata (diagonale, Newton): est e ovest scambiati"));
                });
            }
            if changed {
                manual.angle = angle.rem_euclid(360.0);
                self.manual = Some(manual);
                self.fit_applied = false;
            }
            let north = manual.north_up_angle();
            note(ui, &tr!("Il nord dell'oggetto punta a {:.0}° dall'alto, in senso orario{}.", "The object's north points {:.0}° from the top, clockwise{}.", north.rem_euclid(360.0), if manual.mirrored { tl("; immagine specchiata") } else { "" }));
            let (mirror, degrees) = manual.upright();
            let straight = !mirror && degrees.rem_euclid(360.0).min(360.0 - degrees.rem_euclid(360.0)) < 0.5;
            ui.add_space(theme::space::S);
            if ui
                .add_enabled(!straight, egui::Button::new((ICON_NORTH.rich_text(), tl("Raddrizza: nord in alto"))).min_size(egui::vec2(ui.available_width(), 28.0)))
                .on_hover_text(tl("Ruota l'immagine (e la specchia, se serve) così che il nord stia in alto e le lune dal lato giusto, come il cielo a occhio nudo. Il ritaglio si azzera."))
                .clicked()
            {
                actions.push(PanelAction::NorthUp { mirror, degrees });
            }
            if straight {
                note(ui, tl("L'immagine è già dritta: nord in alto."));
            }
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
