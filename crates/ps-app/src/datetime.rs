//! Date and time controls: a button that opens a month calendar, and a time field that can be
//! typed in or stepped with the arrow keys and the mouse wheel. Calendar arithmetic lives here
//! too (civil dates and day numbers).

use crate::theme::{self, ACCENT, TEXT, TEXT_WEAK};
use eframe::egui::{self, RichText, Ui};
use egui::containers::menu::{MenuButton, MenuConfig};
use egui_material_icons::{MaterialIcon, icons::*};
use ps_core::i18n::tl;

/// Day number (days since 1970-01-01) of a civil date.
pub fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = i64::from(y) - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (i64::from(m) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// The civil date of a day number.
pub fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    ((yoe + era * 400 + i64::from(m <= 2)) as i32, m, d)
}

pub fn days_in_month(y: i32, m: u32) -> u32 {
    let next = if m == 12 { days_from_civil(y + 1, 1, 1) } else { days_from_civil(y, m + 1, 1) };
    (next - days_from_civil(y, m, 1)) as u32
}

fn month_name(m: u32) -> &'static str {
    [tl("gennaio"), tl("febbraio"), tl("marzo"), tl("aprile"), tl("maggio"), tl("giugno"), tl("luglio"), tl("agosto"), tl("settembre"), tl("ottobre"), tl("novembre"), tl("dicembre")][(m as usize + 11) % 12]
}

/// A button showing the date (`2026-08-29`) that opens a month calendar. True when the date changed.
pub fn date_picker(ui: &mut Ui, id: &str, year: &mut i32, month: &mut u32, day: &mut u32) -> bool {
    let mut changed = false;
    let label = format!("{:04}-{:02}-{:02}", *year, *month, *day);
    let config = MenuConfig::new().close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
    let view_id = ui.make_persistent_id((id, "view"));
    let (response, inner) = MenuButton::new((ICON_CALENDAR_MONTH.rich_text(), label)).config(config).ui(ui, |ui| {
        // The month being looked at is kept while the popup is open (the date moves only on a click).
        let (mut vy, mut vm) = ui.data(|d| d.get_temp::<(i32, u32)>(view_id)).unwrap_or((*year, *month));
        ui.set_min_width(200.0);
        ui.horizontal(|ui| {
            let step = |ui: &mut Ui, icon: MaterialIcon, tip: &str| ui.add(egui::Button::new(icon.rich_text()).frame(false)).on_hover_text(tip).clicked();
            if step(ui, ICON_KEYBOARD_DOUBLE_ARROW_LEFT, tl("Anno precedente")) {
                vy -= 1;
            }
            if step(ui, ICON_CHEVRON_LEFT, tl("Mese precedente")) {
                (vy, vm) = if vm == 1 { (vy - 1, 12) } else { (vy, vm - 1) };
            }
            ui.add_sized([100.0, 20.0], egui::Label::new(RichText::new(format!("{} {}", month_name(vm), vy)).strong().color(TEXT)));
            if step(ui, ICON_CHEVRON_RIGHT, tl("Mese successivo")) {
                (vy, vm) = if vm == 12 { (vy + 1, 1) } else { (vy, vm + 1) };
            }
            if step(ui, ICON_KEYBOARD_DOUBLE_ARROW_RIGHT, tl("Anno successivo")) {
                vy += 1;
            }
        });
        vy = vy.clamp(1900, 2100);
        ui.data_mut(|d| d.insert_temp(view_id, (vy, vm)));
        ui.add_space(theme::space::XS);
        // Monday first.
        let first = (days_from_civil(vy, vm, 1) + 3).rem_euclid(7) as u32;
        egui::Grid::new((id, "grid")).spacing([2.0, 2.0]).show(ui, |ui| {
            for name in [tl("Lu"), tl("Ma"), tl("Me"), tl("Gi"), tl("Ve"), tl("Sa"), tl("Do")] {
                ui.label(RichText::new(name).size(theme::size::SMALL).color(TEXT_WEAK));
            }
            ui.end_row();
            for cell in 0..42u32 {
                let d = cell as i64 - first as i64 + 1;
                if d >= 1 && d <= i64::from(days_in_month(vy, vm)) {
                    let chosen = (vy, vm, d as u32) == (*year, *month, *day);
                    let text = RichText::new(format!("{d}")).color(if chosen { theme::BG } else { TEXT });
                    let button = egui::Button::new(text).min_size(egui::vec2(26.0, 22.0)).fill(if chosen { ACCENT } else { egui::Color32::TRANSPARENT });
                    if ui.add(button).clicked() {
                        (*year, *month, *day) = (vy, vm, d as u32);
                        changed = true;
                        ui.close();
                    }
                } else {
                    ui.label("");
                }
                if cell % 7 == 6 {
                    ui.end_row();
                    if cell >= 34 && (cell as i64 - first as i64 + 2) > i64::from(days_in_month(vy, vm)) {
                        break;
                    }
                }
            }
        });
    });
    if inner.is_none() {
        ui.data_mut(|d| d.remove_temp::<(i32, u32)>(view_id));
    }
    response.on_hover_text(tl("Scegli la data da un calendario"));
    changed
}

/// Hours and minutes in one field (`20:09`): type it, or step with ↑↓ (minutes; with Shift,
/// hours) and the mouse wheel. True when the time changed.
pub fn time_field(ui: &mut Ui, id: &str, hour: &mut u32, minute: &mut u32) -> bool {
    let id = ui.make_persistent_id((id, "time"));
    let shown = format!("{:02}:{:02}", *hour, *minute);
    let mut text = ui.data(|d| d.get_temp::<String>(id)).filter(|_| ui.memory(|m| m.has_focus(id))).unwrap_or_else(|| shown.clone());
    let response = ui.add(egui::TextEdit::singleline(&mut text).id(id).desired_width(54.0).align(egui::Align2::CENTER_CENTER));
    let mut changed = false;
    let mut step = 0i32;
    if response.has_focus() {
        ui.input_mut(|i| {
            let shift = i.modifiers.shift;
            for (key, sign) in [(egui::Key::ArrowUp, 1), (egui::Key::ArrowDown, -1)] {
                if i.consume_key(egui::Modifiers::NONE, key) {
                    step += sign;
                }
                if i.consume_key(egui::Modifiers::SHIFT, key) {
                    step += sign * 60;
                }
            }
            let _ = shift;
        });
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.5 {
            step += if scroll > 0.0 { 1 } else { -1 } * if ui.input(|i| i.modifiers.shift) { 60 } else { 1 };
        }
    }
    if step != 0 {
        let total = (*hour as i32 * 60 + *minute as i32 + step).rem_euclid(1440);
        (*hour, *minute) = ((total / 60) as u32, (total % 60) as u32);
        text = format!("{:02}:{:02}", *hour, *minute);
        changed = true;
    } else if response.lost_focus() {
        if let Some((h, m)) = parse_time(&text) {
            changed = (h, m) != (*hour, *minute);
            (*hour, *minute) = (h, m);
        }
        text = format!("{:02}:{:02}", *hour, *minute);
    }
    ui.data_mut(|d| d.insert_temp(id, text));
    response.on_hover_text(tl("Ora e minuti: scrivili, oppure ↑ ↓ (con Maiusc: le ore) o la rotella del mouse"));
    changed
}

/// `20:09`, `2009`, `20.09`, `8 5`, `20`.
fn parse_time(text: &str) -> Option<(u32, u32)> {
    let t = text.trim();
    let parts: Vec<&str> = t.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty()).collect();
    let (h, m) = match parts.as_slice() {
        [one] if one.len() >= 3 => {
            let v: u32 = one.parse().ok()?;
            (v / 100 % 100, v % 100)
        }
        [one] => (one.parse().ok()?, 0),
        [h, m, ..] => (h.parse().ok()?, m.parse().ok()?),
        [] => return None,
    };
    (h < 24 && m < 60).then_some((h, m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_are_read_in_several_spellings() {
        assert_eq!(parse_time("20:09"), Some((20, 9)));
        assert_eq!(parse_time("2009"), Some((20, 9)));
        assert_eq!(parse_time("8.5"), Some((8, 5)));
        assert_eq!(parse_time("21"), Some((21, 0)));
        assert_eq!(parse_time("25:00"), None);
        assert_eq!(parse_time("abc"), None);
    }

    #[test]
    fn calendar_arithmetic() {
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2026, 2), 28);
        assert_eq!(days_in_month(2026, 12), 31);
        assert_eq!(civil_from_days(days_from_civil(2026, 8, 29)), (2026, 8, 29));
        // 2026-08-29 is a Saturday: Monday-first index 5.
        assert_eq!((days_from_civil(2026, 8, 29) + 3).rem_euclid(7), 5);
    }
}
