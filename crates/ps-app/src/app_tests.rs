use super::*;

/// The app with `image` open, as if loaded from a file.
fn opened(size: [f32; 2], image: Image) -> egui_kittest::Harness<'static, PlanetaryStudio> {
    // Real frame times, so double clicks are double clicks.
    let mut harness = egui_kittest::Harness::builder().with_size(size).with_step_dt(1.0 / 60.0).build_eframe(|cc| PlanetaryStudio::new(cc, None));
    let analysis = Box::new(analyze(&image));
    let display = worker::display_image(&image, 4096);
    harness.state().events_tx.send(Event::Loaded { path: "test.tif".into(), image, display, analysis, elapsed: Duration::ZERO, mosaic: None, keep_settings: false }).unwrap();
    harness.run_steps(3);
    harness
}

/// Every special character used in labels must exist in the bundled fonts (otherwise it
/// shows up as an empty box).
#[test]
fn ui_glyphs_exist() {
    let ctx = egui::Context::default();
    theme::apply(&ctx);
    let mut output = ctx.run_ui(Default::default(), |_| {});
    let missing: String = "°·×≈βσλ–…″⌘⇧⌥⌃⚠".chars().filter(|&c| !ctx.fonts_mut(|f| f.has_glyph(&egui::FontId::proportional(13.0), c))).collect();
    // No renderer here to upload the font atlas to.
    output.textures_delta.clear();
    assert!(missing.is_empty(), "missing glyphs: {missing}");
}

/// Every area of the panel, with all its cards open, reads as English: no Italian word is
/// left in any label the screen reader (and so the user) can see.
#[test]
fn english_interface_has_no_italian_left() {
    use egui_kittest::kittest::NodeT;
    let italian = ["il", "di", "del", "della", "delle", "dei", "che", "con", "sul", "sulla", "nel", "nella", "più", "dal", "gli", "non", "come", "immagine", "impostazioni", "elaborazione"];
    let plane: Vec<f32> = (0..200 * 200).map(|i| if ((i % 200) as f32 - 100.0).hypot((i / 200) as f32 - 100.0) < 60.0 { 0.6 } else { 0.02 }).collect();
    let image = Image::new(200, 200, vec![plane.clone(), plane.clone(), plane], 16);
    let leaks = ps_core::i18n::with_lang(ps_core::i18n::Lang::En, || {
        let mut harness = opened([1400.0, 2600.0], image);
        let mut leaks = std::collections::BTreeSet::new();
        let mut seen = 0;
        for area in [prefs_group::Group::Favorites, prefs_group::Group::Sharpness, prefs_group::Group::Noise, prefs_group::Group::Tone, prefs_group::Group::Color, prefs_group::Group::Regions, prefs_group::Group::Identify, prefs_group::Group::Export] {
            {
                let app = harness.state_mut();
                app.prefs.current_area = area.name().to_string();
                app.prefs.one_group_open = false;
                app.prefs.expanded = Stage::ALL.into_iter().collect();
            }
            harness.run_steps(3);
            for node in harness.root().children_recursive() {
                for text in [node.accesskit_node().label(), node.accesskit_node().value()].into_iter().flatten() {
                    seen += 1;
                    let words: Vec<String> = text.split(|c: char| !c.is_alphabetic()).map(str::to_lowercase).collect();
                    if italian.iter().any(|w| words.iter().any(|x| x == w)) {
                        leaks.insert(format!("{}: {text}", area.name()));
                    }
                }
            }
        }
        assert!(seen > 300, "only {seen} labels found: the test sees nothing");
        leaks
    });
    assert!(leaks.is_empty(), "Italian text in the English interface:\n{}", leaks.into_iter().collect::<Vec<_>>().join("\n"));
}

/// Regression: while a slider is dragged and results stream in, nothing may move — not
/// the image, not the panel, not the cards below the one being edited.
#[test]
fn layout_is_still_while_processing() {
    let (w, h) = (500, 400);
    let plane: Vec<f32> = (0..w * h).map(|i| if ((i % w) as f32 - 250.0).hypot((i / w) as f32 - 200.0) < 120.0 { 0.5 + 0.2 * ((i / w) as f32 * 0.1).sin() } else { 0.02 }).collect();
    let image = Image::new(w, h, vec![plane.clone(), plane.clone(), plane], 16);
    let mut harness = opened([1400.0, 900.0], image);
    harness.run_steps(2);
    harness.state_mut().pipeline.deconv.enabled = true;
    let mut samples = Vec::new();
    for k in 0..240 {
        // A drag: the value changes every frame, so processing is always under way.
        harness.state_mut().pipeline.deconv.params.iterations = 10 + (k % 30) as u32;
        harness.step();
        std::thread::sleep(Duration::from_millis(4));
        let app = harness.state();
        let canvas = app.document.as_ref().unwrap().viewer.canvas();
        let below = harness.ctx.data(|d| d.get_temp::<f32>(egui::Id::unique(("card-top", "Wavelet"))));
        samples.push((canvas, app.side_width, below, app.worker.is_busy()));
    }
    assert!(samples.iter().any(|s| s.3), "processing never ran");
    let first = samples[20];
    for (i, s) in samples.iter().enumerate().skip(20) {
        assert_eq!(s.0, first.0, "image area moved at frame {i}");
        assert_eq!(s.1, first.1, "panel width changed at frame {i}");
        assert_eq!(s.2, first.2, "card below the deconvolution moved at frame {i}");
    }
}

/// A module's reset button and "Rimuovi" (drop every automatic choice).
#[test]
fn reset_and_remove_automatic_settings() {
    let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
    let mut harness = opened([1200.0, 800.0], image);
    let ctx = harness.ctx.clone();
    let app = harness.state_mut();
    let start = app.pipeline.clone();

    // A module's reset brings its settings back but leaves its switch alone.
    app.pipeline.deconv.enabled = true;
    app.pipeline.deconv.params.iterations = 77;
    app.apply_actions(&ctx, vec![PanelAction::ResetModule(Stage::Deconv)]);
    assert!(app.pipeline.deconv.enabled);
    assert_eq!(app.pipeline.deconv.params, start.deconv.params);

    // "Rimuovi" drops the analysis' choices, layers survive.
    app.apply_auto_all();
    app.add_layer(LayerTemplate::Ellipse);
    app.apply_actions(&ctx, vec![PanelAction::ClearAuto]);
    assert_eq!(app.pipeline.layers.len(), 1);
    assert_eq!(Pipeline { layers: vec![], ..app.pipeline.clone() }, neutral_pipeline());
}

/// Adding the rings' mask selects it; "Valori consigliati" restores its settings in place,
/// keeping its name, switch, position and painted strokes.
#[test]
fn rings_mask_restores_in_place() {
    let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
    let mut harness = opened([1200.0, 800.0], image);
    let ctx = harness.ctx.clone();
    let app = harness.state_mut();
    app.add_layer(LayerTemplate::Ellipse);
    app.apply_actions(&ctx, vec![PanelAction::CreateRole(LayerRole::Rings)]);
    let id = app.pipeline.layers[1].id;
    assert_eq!(app.selected_layer, Some(id), "the new mask is selected");
    app.pipeline.layers.swap(0, 1);
    let fresh = app.pipeline.layers[0].clone();
    let rings = &mut app.pipeline.layers[0];
    (rings.name, rings.enabled) = ("Mie".into(), false);
    rings.wavelet.enabled = false;
    rings.mask.strokes.push(BrushStroke { points: vec![[1.0, 1.0]], radius: 3.0, hardness: 0.5, erase: false });
    app.apply_actions(&ctx, vec![PanelAction::RestoreRole(id)]);
    let rings = &app.pipeline.layers[0];
    assert_eq!((rings.id, rings.name.as_str(), rings.enabled, rings.mask.strokes.len()), (id, "Mie", false, 1));
    assert_eq!(rings.wavelet, fresh.wavelet);
    assert_eq!(app.pipeline.layers.len(), 2);
}

#[test]
fn layer_adjustments_one_at_a_time() {
    // Tall enough for the whole layer editor to be on screen, where the click lands.
    let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
    let mut harness = opened([1400.0, 2400.0], image);
    {
        let app = harness.state_mut();
        app.prefs.current_area = "Maschere".into();
        app.add_layer(LayerTemplate::Ellipse);
        app.selected_layer = app.pipeline.layers.first().map(|l| l.id);
        let layer = &mut app.pipeline.layers[0];
        assert!(!layer.overrides_any(), "a new mask starts with no adjustment");
        // Two adjustments inserted by the user.
        (layer.contrast.enabled, layer.denoise.enabled) = (true, true);
    }
    harness.run_steps(2);
    assert!(harness.state().show_mask, "an open layer shows what it selects");
    use egui_kittest::kittest::Queryable;
    harness.get_by_label("Contrasto").click();
    harness.run_steps(2);
    assert!(harness.state().prefs.collapsed_sections.contains("layer-open:contrast"), "the clicked adjustment opens");
    harness.get_by_label("Rumore").click();
    harness.run_steps(2);
    let open = &harness.state().prefs.collapsed_sections;
    assert!(open.contains("layer-open:noise") && !open.contains("layer-open:contrast"), "the open one closes");
}

/// With the quick preview on, a drag that changes the settings renders a reduced image;
/// releasing renders full size again, and only that one counts as the real result.
#[test]
fn quick_preview_while_dragging() {
    let (w, h) = (800, 600);
    let plane: Vec<f32> = (0..w * h).map(|i| if ((i % w) as f32 - 400.0).hypot((i / w) as f32 - 300.0) < 200.0 { 0.6 } else { 0.02 }).collect();
    let mut harness = opened([1400.0, 900.0], Image::new(w, h, vec![plane], 16));
    harness.state_mut().prefs.fast_preview = true;
    // Pointer held down away from the image (as on a slider), settings changing.
    let press = |pressed| egui::Event::PointerButton { pos: egui::pos2(700.0, 20.0), button: egui::PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
    harness.event(egui::Event::PointerMoved(egui::pos2(700.0, 20.0)));
    harness.event(press(true));
    harness.run_steps(1);
    harness.state_mut().pipeline.contrast.params.amount = 0.5;
    harness.run_steps(1);
    assert_eq!(harness.state().last_request.as_ref().unwrap().proxy, 2);
    harness.event(press(false));
    harness.run_steps(1);
    assert_eq!(harness.state().last_request.as_ref().unwrap().proxy, 1);
    for _ in 0..300 {
        harness.step();
        if harness.state().last_output.as_ref().is_some_and(|o| o.pipeline.contrast.params.amount == 0.5) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let out = harness.state().last_output.as_ref().expect("full result");
    assert!(!out.proxy && out.image.width() == w);
}

/// Cards start closed; in an area, opening one closes the one that was open. Brush strokes
/// go to the deringing area when that is the mask being edited.
#[test]
fn one_card_open_per_area() {
    let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
    let mut harness = opened([1400.0, 2000.0], image);
    {
        let app = harness.state_mut();
        assert!(app.prefs.expanded.is_empty(), "cards start closed");
        app.prefs.one_group_open = true;
        app.prefs.current_area = "Nitidezza".into();
    }
    harness.run_steps(2);
    use egui_kittest::kittest::Queryable;
    harness.get_by_label_contains("Deconvoluzione").click();
    harness.run_steps(2);
    assert!(harness.state().prefs.expanded.contains(&Stage::Deconv));
    harness.get_by_label_contains("Wavelet").click();
    harness.run_steps(2);
    let expanded = &harness.state().prefs.expanded;
    assert!(expanded.contains(&Stage::Wavelet) && !expanded.contains(&Stage::Deconv), "{expanded:?}");

    let app = harness.state_mut();
    app.selected_layer = Some(RING_MASK);
    app.add_stroke(vec![[10.0, 10.0]], false);
    assert_eq!(app.pipeline.dering.params.strokes.len(), 1);
}

/// The brush and the red overlay last only while their controls are on screen, and never
/// outlive the layer they paint on.
#[test]
fn mask_tools_do_not_linger() {
    let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
    let mut harness = opened([1400.0, 2000.0], image);
    {
        let app = harness.state_mut();
        app.prefs.current_area = "Maschere".into();
        app.add_layer(LayerTemplate::Brush);
    }
    harness.run_steps(3);
    assert_eq!(harness.state().layer_tool, LayerTool::Brush, "painting right after adding a brush layer");

    // Leaving the masks' area puts the brush away.
    harness.state_mut().prefs.current_area = "Nitidezza".into();
    harness.run_steps(2);
    let app = harness.state();
    assert!(app.layer_tool == LayerTool::None && !app.show_mask);

    // The layer disappearing (as an undo of its creation does) forgets it as the selected mask.
    harness.state_mut().prefs.current_area = "Maschere".into();
    harness.state_mut().layer_tool = LayerTool::Brush;
    harness.run_steps(2);
    harness.state_mut().pipeline.layers.clear();
    harness.run_steps(2);
    let app = harness.state();
    assert!(app.selected_layer.is_none() && app.layer_tool == LayerTool::None);
}

/// A drag with the brush on the image becomes one stroke on the selected mask.
#[test]
fn brush_drag_paints_a_stroke() {
    let image = Image::new(400, 300, vec![vec![0.3; 400 * 300]], 16);
    let mut harness = opened([1400.0, 900.0], image);
    {
        let app = harness.state_mut();
        app.prefs.current_area = "Maschere".into();
        app.add_layer(LayerTemplate::Brush);
    }
    harness.run_steps(3);
    let canvas = harness.state().document.as_ref().unwrap().viewer.canvas();
    let (a, b) = (canvas.center() - Vec2::new(60.0, 0.0), canvas.center() + Vec2::new(60.0, 0.0));
    let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
    harness.event(egui::Event::PointerMoved(a));
    harness.run_steps(1);
    harness.event(button(a, true));
    harness.run_steps(1);
    for k in 1..=12 {
        harness.event(egui::Event::PointerMoved(a + (b - a) * (k as f32 / 12.0)));
        harness.run_steps(1);
    }
    harness.event(button(b, false));
    harness.run_steps(2);
    let strokes = &harness.state().pipeline.layers[0].mask.strokes;
    assert_eq!(strokes.len(), 1, "one stroke");
    assert!(strokes[0].points.len() > 2, "{:?}", strokes[0].points);
}

/// The A/B snapshot belongs to the image it was taken on.
#[test]
fn snapshot_follows_the_image() {
    let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
    let mut harness = opened([1200.0, 800.0], image);
    for _ in 0..100 {
        harness.step();
        if harness.state().last_output.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let ctx = harness.ctx.clone();
    harness.state_mut().take_snapshot(&ctx);
    assert!(harness.state().snapshot.is_some() && harness.state().compare == Compare::Ab);
    harness.state_mut().rotate(&ctx, Geometry::rotate_clockwise);
    harness.run_steps(2);
    assert!(harness.state().snapshot.is_none() && harness.state().compare != Compare::Ab);
}

/// "Tieni A" brings A's settings back; "Tieni B" keeps the current ones; both end the
/// comparison.
#[test]
fn comparison_keeps_a_or_b() {
    let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
    let mut harness = opened([1200.0, 800.0], image);
    for _ in 0..100 {
        harness.step();
        if harness.state().last_output.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let ctx = harness.ctx.clone();
    let a = harness.state().pipeline.clone();
    harness.state_mut().take_snapshot(&ctx);
    harness.state_mut().pipeline.tone.params.current.gamma = 1.7;
    let b = harness.state().pipeline.clone();
    harness.state_mut().end_comparison(&ctx, true);
    assert!(harness.state().snapshot.is_none() && harness.state().compare == Compare::After);
    assert_eq!(harness.state().pipeline, a, "A's settings are back");

    harness.state_mut().take_snapshot(&ctx);
    harness.state_mut().pipeline = b.clone();
    harness.state_mut().end_comparison(&ctx, false);
    assert!(harness.state().snapshot.is_none());
    assert_eq!(harness.state().pipeline, b, "B stays");
}

/// Keys 1‥6 pick an area; a double click on a card's title switches the module off.
#[test]
fn keys_and_double_click() {
    let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
    let mut harness = opened([1400.0, 900.0], image);
    harness.key_press(Key::Num3);
    harness.run_steps(2);
    assert_eq!(harness.state().prefs.current_area, "Curve e livelli");
    harness.key_press(Key::Num1);
    harness.run_steps(2);
    assert_eq!(harness.state().prefs.current_area, "Nitidezza");

    use egui_kittest::kittest::Queryable;
    assert!(harness.state().pipeline.wavelet.enabled);
    // Two quick clicks on the title.
    let at = harness.get_by_label_contains("Wavelet").rect().center();
    let button = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
    harness.event(egui::Event::PointerMoved(at));
    harness.step();
    for pressed in [true, false, true, false] {
        harness.event(button(pressed));
        harness.step();
    }
    harness.run_steps(2);
    assert!(!harness.state().pipeline.wavelet.enabled, "double click switches it off");
    assert!(!harness.state().prefs.expanded.contains(&Stage::Wavelet), "and leaves it folded");
}

/// Quick export writes next to the original and never overwrites.
#[test]
fn quick_export_next_to_the_original() {
    let dir = std::env::temp_dir().join(format!("ps-quick-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
    let mut harness = opened([1200.0, 800.0], image);
    let doc = harness.state_mut().document.as_mut().unwrap();
    (doc.path, doc.export_name) = (dir.join("giove.tif"), "giove".into());
    harness.state_mut().prefs.export_format = "png".into();
    let ctx = harness.ctx.clone();
    for _ in 0..2 {
        harness.state_mut().quick_export(&ctx);
        for _ in 0..200 {
            harness.step();
            if harness.state().io_busy.is_none() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    assert!(dir.join("giove_ps.png").is_file() && dir.join("giove_ps-2.png").is_file());
    std::fs::remove_dir_all(dir).ok();
}

/// Rotating from the crop bar re-orients the working image and keeps cropping.
#[test]
fn rotation_reorients_the_image() {
    let image = Image::new(64, 32, vec![(0..64 * 32).map(|i| (i % 64) as f32 / 64.0).collect()], 16);
    let mut harness = opened([1200.0, 800.0], image);
    let ctx = harness.ctx.clone();
    harness.state_mut().start_crop();
    harness.state_mut().finish_crop(&ctx, CropAction::Rotate(1));
    let app = harness.state();
    let doc = app.document.as_ref().unwrap();
    assert_eq!((doc.source.width(), doc.source.height()), (32, 64));
    assert!(app.cropping, "still cropping after a rotation");
    // The left column (darkest) is now the top row.
    assert!(doc.source.channels()[0][..32].iter().all(|&v| v == 0.0));
}

/// Holding Space over the image shows the original; releasing it shows the result again.
#[test]
fn space_bar_peeks_at_the_original() {
    let image = Image::new(64, 64, vec![vec![0.3; 64 * 64]], 16);
    let mut harness = opened([1200.0, 800.0], image);
    harness.hover_at(egui::pos2(300.0, 400.0));
    harness.run_steps(2);
    let key = |pressed| egui::Event::Key { key: Key::Space, physical_key: None, pressed, repeat: false, modifiers: Modifiers::NONE };
    harness.event(key(true));
    harness.run_steps(2);
    assert_eq!(harness.state().shown_compare, Compare::Before);
    harness.event(key(false));
    harness.run_steps(2);
    assert_eq!(harness.state().shown_compare, Compare::After);
}

/// Regression: content asking for slightly more than the panel's width made the
/// resizable side panel grow every frame until it covered the image.
#[test]
fn side_panel_width_is_stable() {
    let (w, h) = (300, 200);
    let plane: Vec<f32> = (0..w * h).map(|i| if ((i % w) as f32 - 150.0).hypot((i / w) as f32 - 100.0) < 60.0 { 0.6 } else { 0.02 }).collect();
    let image = Image::new(w, h, vec![plane.clone(), plane.clone(), plane], 16);
    let mut harness = opened([1400.0, 900.0], image);
    harness.run_steps(2);
    {
        let app = harness.state_mut();
        // Every card open, the widest the panel gets.
        let all: Vec<Stage> = app.prefs.modules.iter().map(|m| m.stage).collect();
        app.prefs.expanded.extend(all);
        app.add_layer(LayerTemplate::Brush);
    }
    let mut widths = Vec::new();
    for _ in 0..300 {
        harness.step();
        widths.push(harness.state().side_width);
    }
    let (min, max) = widths[10..].iter().fold((f32::MAX, f32::MIN), |(lo, hi), &w| (lo.min(w), hi.max(w)));
    assert!(max - min < 0.5, "panel width drifted from {min} to {max}");
    assert!(max < 560.0, "panel too wide: {max}");
}

/// Renders the UI off-screen to PNGs for visual review:
/// `PS_SNAPSHOT_IMAGE=in.tif PS_SNAPSHOT_OUT=dir cargo test -p ps-app --release -- --ignored snapshot`
/// The "Ingrandisci" tab, the "Identifica" window with its names, and the solar colour card:
/// `PS_SNAPSHOT_IMAGE=<absolute path> PS_SNAPSHOT_OUT=dir cargo test -p ps-app -- --ignored snapshot_new_tools`
#[test]
#[ignore]
fn snapshot_new_tools() {
    let input = PathBuf::from(std::env::var("PS_SNAPSHOT_IMAGE").expect("PS_SNAPSHOT_IMAGE"));
    let out_dir = PathBuf::from(std::env::var("PS_SNAPSHOT_OUT").expect("PS_SNAPSHOT_OUT"));
    let mut harness = egui_kittest::Harness::builder().with_size([1500.0, std::env::var("PS_SNAPSHOT_H").ok().and_then(|h| h.parse().ok()).unwrap_or(940.0)]).wgpu().build_eframe(|cc| PlanetaryStudio::new(cc, None));
    if std::env::var("PS_SNAPSHOT_EN").is_ok() {
        harness.state_mut().prefs.language = ps_core::i18n::Lang::En;
        ps_core::i18n::set_lang(ps_core::i18n::Lang::En);
    }
    harness.run_steps(3);
    let ctx = harness.ctx.clone();
    harness.state_mut().load(&ctx, input);
    let settle = |harness: &mut egui_kittest::Harness<PlanetaryStudio>| {
        for _ in 0..900 {
            harness.step();
            let app = harness.state();
            let idle = app.io_busy.is_none() && !app.worker.is_busy() && !app.worker.is_background_busy() && !app.identify.busy();
            if app.document.is_some() && app.last_output.is_some() && idle {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        harness.run_steps(4);
    };
    settle(&mut harness);
    {
        let app = harness.state_mut();
        app.prefs.current_area = "Esporta".into();
        app.prefs.resize.enabled = true;
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_resize.png")).unwrap();
    harness.state_mut().prefs.current_area = "Identifica e annota".into();
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_identify.png")).unwrap();
    {
        let app = harness.state_mut();
        app.prefs.current_area = "Colore".into();
        app.prefs.expanded = [Stage::SunColor].into();
        app.pipeline.suncolor = Module::on(ps_core::suncolor::SunColorParams::default());
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_suncolor.png")).unwrap();
}

/// The Colore tab with Jupiter's photometric colour card open and switched on, before and after:
/// `PS_SNAPSHOT_IMAGE=<absolute path of a colour Jupiter> PS_SNAPSHOT_OUT=dir cargo test -p ps-app -- --ignored snapshot_photocolor`
#[test]
#[ignore]
fn snapshot_photocolor() {
    let input = PathBuf::from(std::env::var("PS_SNAPSHOT_IMAGE").expect("PS_SNAPSHOT_IMAGE"));
    let out_dir = PathBuf::from(std::env::var("PS_SNAPSHOT_OUT").expect("PS_SNAPSHOT_OUT"));
    let mut harness = egui_kittest::Harness::builder().with_size([1500.0, 940.0]).wgpu().build_eframe(|cc| PlanetaryStudio::new(cc, None));
    harness.run_steps(3);
    let ctx = harness.ctx.clone();
    harness.state_mut().load(&ctx, input);
    let settle = |harness: &mut egui_kittest::Harness<PlanetaryStudio>| {
        for _ in 0..900 {
            harness.step();
            let app = harness.state();
            let idle = app.io_busy.is_none() && !app.worker.is_busy() && !app.worker.is_background_busy() && !app.identify.busy();
            if app.document.is_some() && app.last_output.is_some() && idle {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        harness.run_steps(4);
    };
    settle(&mut harness);
    {
        let app = harness.state_mut();
        app.prefs.current_area = "Colore".into();
        app.prefs.expanded = [Stage::PhotoColor].into();
        // The subject is forced here: a file name with no planet in it wouldn't say Jupiter.
        app.identify.set_body_for_test(ps_core::capture::Body::Jupiter);
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_photocolor_off.png")).unwrap();
    harness.state_mut().pipeline.photocolor = Module::on(ps_core::photocolor::PhotoColorParams::default());
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_photocolor_on.png")).unwrap();
}

/// The final sharpening card with the learned curve: the three hypotheses on the seeing and
/// why the curve is shaped so (`PS_SNAPSHOT_IMAGE`, `PS_SNAPSHOT_OUT` as for [`snapshot`]).
#[test]
#[ignore]
fn snapshot_smart_learned() {
    let input = PathBuf::from(std::env::var("PS_SNAPSHOT_IMAGE").expect("PS_SNAPSHOT_IMAGE"));
    let out_dir = PathBuf::from(std::env::var("PS_SNAPSHOT_OUT").expect("PS_SNAPSHOT_OUT"));
    if std::env::var("PS_SNAPSHOT_EN").is_ok() {
        ps_core::i18n::set_lang(ps_core::i18n::Lang::En);
    }
    let img = std::sync::Arc::new(ps_io::load(&input).unwrap());
    let analysis = ps_core::analysis::analyze(&img);
    let mut params = ps_core::preset::auto_smart(analysis.kind, &analysis, ps_core::preset::AutoStrength::Normal);
    let auto = params.clone();
    let made = crate::smart_compare::make(&img, &params, crate::smart_compare::centre(&img, &analysis));
    let mut thumbs = Vec::new();
    let mut tab = egui_kittest::Harness::builder().with_size([420.0, 1500.0]).with_accessibility_check(false).build_ui(move |ui| {
        crate::theme::apply(ui.ctx());
        if thumbs.is_empty() {
            thumbs = made.iter().map(|(n, c)| (*n, ui.ctx().load_texture(format!("{n:?}"), c.clone(), egui::TextureOptions::LINEAR))).collect();
        }
        egui::CentralPanel::default().show(ui, |ui| {
            let view = panels::LearnedView { analysis: &analysis, thumbs: &thumbs, busy: false };
            panels::smart_body(ui, &mut params, &auto, &mut SmartView::None, None, &[], view);
        });
    });
    tab.run();
    tab.render().unwrap().save(out_dir.join("ui_smart_learned.png")).unwrap();
}

/// The pictures of the GitHub page, from the user's own images: Saturn processed, Jupiter
/// and the Moon before/after, Saturn's rings mask, a lunar mosaic. `PS_SHOW_OUT` (dir),
/// `PS_SHOW_JUPITER`, `PS_SHOW_SATURN`, `PS_SHOW_MOON` (absolute paths), `PS_SHOW_MOSAIC`
/// (frames separated by '|'); `PS_SNAPSHOT_EN` for the English interface.
#[test]
#[ignore]
fn snapshot_showcase() {
    let out_dir = PathBuf::from(std::env::var("PS_SHOW_OUT").expect("PS_SHOW_OUT"));
    let mut harness = egui_kittest::Harness::builder().with_size([1500.0, 940.0]).wgpu().build_eframe(|cc| PlanetaryStudio::new(cc, None));
    if std::env::var("PS_SNAPSHOT_EN").is_ok() {
        harness.state_mut().prefs.language = ps_core::i18n::Lang::En;
        ps_core::i18n::set_lang(ps_core::i18n::Lang::En);
    }
    harness.run_steps(3);
    let ctx = harness.ctx.clone();
    let settle = |harness: &mut egui_kittest::Harness<PlanetaryStudio>| {
        for _ in 0..900 {
            harness.step();
            let app = harness.state();
            let idle = app.io_busy.is_none() && !app.worker.is_busy() && !app.worker.is_background_busy() && !app.smart_compare.busy();
            if app.document.is_some() && app.last_output.is_some() && idle {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        harness.run_steps(6);
    };
    // Opens an image, processed automatically, on the sharpening tab with the final
    // sharpening open.
    let open = |harness: &mut egui_kittest::Harness<PlanetaryStudio>, path: &str| {
        harness.state_mut().load(&ctx, PathBuf::from(path));
        for _ in 0..300 {
            harness.step();
            if harness.state().document.as_ref().is_some_and(|d| d.path == Path::new(path)) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        settle(harness);
        // The full automatic, learned final sharpening included.
        harness.state_mut().apply_auto_all();
        let app = harness.state_mut();
        app.prefs.current_area = "Nitidezza".into();
        app.prefs.expanded = [Stage::Smart].into();
        settle(harness);
    };
    let save = |harness: &mut egui_kittest::Harness<PlanetaryStudio>, name: &str| harness.render().unwrap().save(out_dir.join(name)).unwrap();

    if let Ok(saturn) = std::env::var("PS_SHOW_SATURN") {
        open(&mut harness, &saturn);
        {
            let app = harness.state_mut();
            app.compare = Compare::After;
            app.document.as_mut().unwrap().viewer.set_zoom(2.0);
        }
        settle(&mut harness);
        save(&mut harness, "main.png");
        // The rings with their own mask.
        {
            let app = harness.state_mut();
            app.prefs.current_area = "Maschere".into();
            app.apply_actions(&ctx, vec![PanelAction::CreateRole(LayerRole::Rings)]);
            app.selected_layer = app.pipeline.layers.iter().find(|l| l.role == Some(LayerRole::Rings)).map(|l| l.id);
            app.show_mask = true;
        }
        settle(&mut harness);
        save(&mut harness, "saturn-masks.png");
    }
    for (var, name, zoom) in [("PS_SHOW_JUPITER", "jupiter-before-after.png", 2.5), ("PS_SHOW_MOON", "moon-detail.png", 2.0)] {
        if let Ok(path) = std::env::var(var) {
            open(&mut harness, &path);
            {
                let app = harness.state_mut();
                app.compare = Compare::Split;
                app.document.as_mut().unwrap().viewer.set_zoom(zoom);
            }
            settle(&mut harness);
            save(&mut harness, name);
        }
    }
    if let Ok(frames) = std::env::var("PS_SHOW_MOSAIC") {
        harness.state_mut().compare = Compare::After;
        harness.state_mut().mosaic_editor.start(&ctx, frames.split('|').map(PathBuf::from).collect());
        for _ in 0..1500 {
            harness.step();
            if !harness.state().mosaic_editor.is_busy() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        harness.run_steps(6);
        save(&mut harness, "lunar-mosaic.png");
    }
}

#[test]
#[ignore]
fn snapshot() {
    let input = PathBuf::from(std::env::var("PS_SNAPSHOT_IMAGE").expect("PS_SNAPSHOT_IMAGE"));
    let out_dir = PathBuf::from(std::env::var("PS_SNAPSHOT_OUT").expect("PS_SNAPSHOT_OUT"));
    let mut harness = egui_kittest::Harness::builder().with_size([1500.0, 940.0]).wgpu().build_eframe(|cc| PlanetaryStudio::new(cc, None));
    if std::env::var("PS_SNAPSHOT_EN").is_ok() {
        harness.state_mut().prefs.language = ps_core::i18n::Lang::En;
        ps_core::i18n::set_lang(ps_core::i18n::Lang::En);
    }
    harness.run_steps(3);

    let ctx = harness.ctx.clone();
    harness.state_mut().load(&ctx, input);
    let settle = |harness: &mut egui_kittest::Harness<PlanetaryStudio>| {
        for _ in 0..600 {
            harness.step();
            let app = harness.state();
            let idle = app.io_busy.is_none() && !app.worker.is_busy() && !app.worker.is_background_busy();
            if app.document.is_some() && app.last_output.is_some() && idle {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        harness.run_steps(4);
    };
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_initial.png")).unwrap();
    harness.state_mut().apply_auto_all();
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_auto.png")).unwrap();

    // "Rumore e artefatti" with the dering card open: indented cards, reset buttons.
    {
        let app = harness.state_mut();
        app.prefs.current_area = "Rumore e artefatti".into();
        app.prefs.expanded = [Stage::Dering].into();
        // A stroke along the right limb, the brush in hand and the ring shown.
        let doc = app.document.as_ref().unwrap();
        let (w, h) = (doc.source.width() as f32, doc.source.height() as f32);
        let (cx, cy, r) = doc.analysis.disk.as_ref().map_or((w / 2.0, h / 2.0, 0.3 * w.min(h)), |d| (d.cx, d.cy, d.radius + 8.0));
        let points = (0..=20).map(|k| -0.9 + 0.09 * k as f32).map(|a| [cx + r * a.cos(), cy + r * a.sin()]).collect();
        app.pipeline.dering.params.strokes = vec![BrushStroke { points, radius: 6.0, hardness: 0.5, erase: false }];
        (app.selected_layer, app.layer_tool, app.brush.radius, app.show_ringing) = (Some(RING_MASK), LayerTool::Brush, 6.0, true);
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_dering.png")).unwrap();

    // The deconvolution card switched on, then the comparison of shapes and algorithms.
    {
        let app = harness.state_mut();
        app.prefs.current_area = "Nitidezza".into();
        app.prefs.expanded = [Stage::Deconv].into();
        app.pipeline.deconv.enabled = true;
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_deconv.png")).unwrap();
    // A box over the whole image: the PSF from the brightest compact object (a moon, if any).
    {
        let app = harness.state_mut();
        let size = app.document.as_ref().map(|d| Vec2::new(d.source.width() as f32, d.source.height() as f32)).unwrap();
        app.measure_psf(Rect::from_min_size(Pos2::ZERO, size));
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_deconv_measured.png")).unwrap();
    {
        let app = harness.state_mut();
        let doc = app.document.as_ref().unwrap();
        app.pipeline.deconv.params = ps_core::preset::auto_deconv(doc.kind, &doc.analysis, ps_core::preset::AutoStrength::Normal);
    }
    let ctx = harness.ctx.clone();
    harness.state_mut().start_deconv_compare(&ctx);
    for _ in 0..600 {
        harness.step();
        let done = harness.state().deconv_compare.as_ref().is_some_and(|c| c.tiles.iter().all(|t| t.texture.is_some()));
        if done {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_deconv_compare.png")).unwrap();
    harness.state_mut().deconv_compare = None;

    // The equipment tab: a C8 with a 2.5× Barlow and a 2.9 µm camera.
    {
        let app = harness.state_mut();
        app.instrument_open = true;
        if let Some(doc) = &mut app.document {
            doc.setup.telescope = ps_core::equipment::telescopes().into_iter().find(|t| t.name.starts_with("Celestron C8"));
            doc.setup.camera = ps_core::equipment::cameras().into_iter().find(|c| c.name == "ZWO ASI462MC");
            doc.setup.barlow = ps_core::equipment::barlows().into_iter().find(|b| b.name == "Barlow 2,5×");
            doc.setup.filter = ps_core::equipment::filter_named("RGB");
        }
        app.equipment_changed();
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_instrument.png")).unwrap();
    harness.state_mut().instrument_open = false;
    // "Nitidezza intelligente": main sliders, curve, and the map of where it holds back.
    {
        let app = harness.state_mut();
        app.prefs.current_area = "Nitidezza".into();
        app.prefs.expanded = [Stage::Smart].into();
        (app.pipeline.smart.enabled, app.pipeline.smart.params.strength) = (true, 0.6);
    }
    settle(&mut harness);
    if let Some(node) = egui_kittest::kittest::Queryable::query_by_label(&harness, &tl("AVANZATE").to_uppercase()) {
        node.click();
    }
    harness.run_steps(2);
    harness.state_mut().smart_view = SmartView::HoldBack;
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_smart.png")).unwrap();
    harness.state_mut().smart_view = SmartView::Resolution;
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_smart_resolution.png")).unwrap();
    // The whole tab, down to the spectrum.
    {
        let doc = harness.state().document.as_ref().unwrap();
        let (mut setup, mut capture, analysis, wavelet) = (doc.setup.clone(), doc.capture.clone(), doc.analysis.clone(), harness.state().pipeline.wavelet.params.clone());
        let mut tab = egui_kittest::Harness::builder().with_size([420.0, 1000.0]).with_accessibility_check(false).build_ui(move |ui| {
            crate::theme::apply(ui.ctx());
            egui::CentralPanel::default().show(ui, |ui| {
                let view = InstrumentView { setup: &mut setup, capture: &mut capture, analysis: &analysis, wavelet: &wavelet };
                instrument_body(ui, view, &mut Vec::new());
            });
        });
        tab.run();
        tab.render().unwrap().save(out_dir.join("ui_instrument_full.png")).unwrap();
    }
    harness.state_mut().prefs.current_area = "Nitidezza".into();

    // The wavelet card with free scales, then switched to the classic ones.
    harness.state_mut().prefs.expanded = [Stage::Wavelet].into();
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_wavelet.png")).unwrap();
    // Scrolled down inside the card: its title stays pinned at the top of the panel.
    harness.event(egui::Event::PointerMoved(egui::pos2(1300.0, 600.0)));
    for _ in 0..6 {
        harness.event(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: Vec2::new(0.0, -120.0), modifiers: Modifiers::NONE, phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel });
        harness.run_steps(1);
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_wavelet_sticky.png")).unwrap();
    harness.event(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: Vec2::new(0.0, 2000.0), modifiers: Modifiers::NONE, phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel });
    harness.run_steps(2);
    egui_kittest::kittest::Queryable::get_by_label(&harness, tl("Classiche 1·2·4·8")).click();
    settle(&mut harness);
    assert_eq!(harness.state().pipeline.wavelet.params.scale_mode, ScaleMode::Dyadic);
    assert_eq!(harness.state().prefs.wavelet_scales, ScaleMode::Dyadic);
    harness.render().unwrap().save(out_dir.join("ui_wavelet_classic.png")).unwrap();

    egui_kittest::kittest::Queryable::get_by_label(&harness, tl("Libere")).click();
    settle(&mut harness);

    // The moons' layer: what it selects on top and on the image, then one adjustment.
    {
        let app = harness.state_mut();
        app.prefs.current_area = "Maschere".into();
        if !app.pipeline.layers.iter().any(|l| l.role == Some(LayerRole::Moons)) {
            add_role_layer(&mut app.pipeline, LayerRole::Moons);
        }
        // A drawn mask that changes nothing yet, switched off, to show its warning dimmed.
        app.add_layer(LayerTemplate::Ellipse);
        if let Some(l) = app.pipeline.layers.last_mut() {
            l.enabled = false;
        }
        app.selected_layer = app.pipeline.layers.iter().find(|l| l.role == Some(LayerRole::Moons)).map(|l| l.id);
        app.layer_tool = LayerTool::None;
        app.prefs.collapsed_sections.insert("layer-open:wavelet".into());
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_layer.png")).unwrap();

    // The ROI being edited, with the button that goes back to moving the view.
    {
        let app = harness.state_mut();
        app.prefs.expanded.clear();
        app.set_preview(PreviewMode::Roi);
        app.tool = Tool::Roi;
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_roi.png")).unwrap();
    harness.state_mut().set_preview(PreviewMode::Full);

    // Zoomed split view with the deconvolution card folded.
    {
        let app = harness.state_mut();
        app.compare = Compare::Split;
        app.prefs.expanded.remove(&Stage::Deconv);
        app.document.as_mut().unwrap().viewer.set_zoom(2.5);
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_split_zoom.png")).unwrap();

    // Only "Curve e livelli" open: a committed step plus a new curve, histogram behind it.
    let open_only = |app: &mut PlanetaryStudio, group: &str| app.prefs.current_area = group.into();
    {
        let app = harness.state_mut();
        app.compare = Compare::After;
        app.document.as_mut().unwrap().viewer.request_fit();
        open_only(app, "Curve e livelli");
        app.prefs.expanded.extend([Stage::Tone, Stage::Curves]);
        app.pipeline.tone.params.current.gamma = 1.3;
        app.pipeline.tone.params.commit();
        app.pipeline.curves.params.current.curve = vec![[0.0, 0.0], [0.4, 0.5], [1.0, 1.0]];
    }
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_tone.png")).unwrap();
    harness.state_mut().prefs.current_area = "Esporta".into();
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_export.png")).unwrap();

    // A mosaic of the frames listed in PS_SNAPSHOT_MOSAIC (separated by '|'): the window,
    // then the result.
    if let Ok(frames) = std::env::var("PS_SNAPSHOT_MOSAIC") {
        harness.state_mut().mosaic_editor.start(&ctx, frames.split('|').map(PathBuf::from).collect());
        for _ in 0..1500 {
            harness.step();
            if !harness.state().mosaic_editor.is_busy() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        harness.run_steps(4);
        harness.render().unwrap().save(out_dir.join("ui_mosaic.png")).unwrap();
        egui_kittest::kittest::Queryable::get_by_label_contains(&harness, tl("Crea il mosaico")).click();
        for _ in 0..1500 {
            harness.step();
            if harness.state().document.as_ref().is_some_and(|d| d.mosaic.is_some()) && harness.state().last_output.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_mosaic_done.png")).unwrap();
    }

    // Saturn: the dedicated rings card, with the rest folded away.
    if let Ok(saturn) = std::env::var("PS_SNAPSHOT_SATURN") {
        harness.state_mut().load(&ctx, PathBuf::from(saturn));
        for _ in 0..200 {
            harness.step();
            // The fresh image drops the masks made on the previous one (same file or not).
            let app = harness.state();
            if app.document.as_ref().is_some_and(|d| d.kind == ImageKind::Saturn) && app.pipeline.layers.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        {
            let app = harness.state_mut();
            open_only(app, "Maschere");
            app.apply_actions(&ctx, vec![PanelAction::CreateRole(LayerRole::Rings)]);
            app.selected_layer = app.pipeline.layers.iter().find(|l| l.role == Some(LayerRole::Rings)).map(|l| l.id);
            app.show_mask = true;
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_saturn_rings.png")).unwrap();

        // Deringing on the globe only, its area shown on the image.
        {
            let app = harness.state_mut();
            open_only(app, "Nitidezza");
            app.prefs.expanded = [Stage::Dering].into();
            app.pipeline.dering.params.strength = 0.8;
            app.selected_layer = Some(RING_MASK);
            app.show_mask = true;
        }
        settle(&mut harness);
        harness.render().unwrap().save(out_dir.join("ui_saturn_dering.png")).unwrap();
    }

    // Crop tool and settings window.
    harness.state_mut().start_crop();
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_crop.png")).unwrap();
    harness.state_mut().finish_crop(&ctx, CropAction::Cancel);
    harness.state_mut().settings.open = true;
    settle(&mut harness);
    harness.render().unwrap().save(out_dir.join("ui_settings.png")).unwrap();
}
