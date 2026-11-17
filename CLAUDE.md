# Planetary Studio — mappa per leggere il progetto

App Rust (egui) per elaborare immagini planetarie già impilate: wavelet, deconvoluzione, dering,
maschere a livelli, nitidezza e riduzione rumore appresi. README.md / README.it.md = vista utente.

## Workspace (`Cargo.toml`)
- `crates/ps-core` — algoritmi puri, niente I/O: `pipeline.rs` (ordine delle fasi), `wavelet.rs`,
  `deconv.rs`, `dering.rs`, `mask.rs`/`blend.rs`, `preset.rs` + `smart.rs` (impostazioni automatiche),
  `identify.rs`/`ephemeris.rs`/`moons.rs` (riconoscimento), `photocolor*.rs`, `gpu/`.
  Reti apprese: `curve_net*.rs` (nitidezza), `noise_net*.rs` e `denoise_net.rs` (rumore),
  `learned*.rs`; pesi in `denoise_weights.bin`. Stringhe: `i18n.rs` (it) + `i18n_en.rs`.
- `crates/ps-io` — caricamento/salvataggio (`src/lib.rs`) ed esempi di servizio in `examples/`
  (`inspect`, `identify`, `survey`, `synth*`, `compare`, `photocolor`, `noise_scales`, `zones_preview`).
  `examples/ml_common/` è usato anche da `ps-train` (via `#[path]`): non spostarlo.
- `crates/ps-app` — GUI: `app.rs` (stato), `panels.rs` (pannelli), `worker.rs` (thread), `viewer.rs`,
  `identify_ui.rs`, `mosaic_ui.rs`, `prefs.rs`. Test in `app_tests.rs`.
- `crates/ps-train` — strumenti di addestramento/valutazione (non in `default-members`): binari
  `denoise_*`, `noise_*`, `curve_*`, `sim_*`/`sim_stack`, `ser_stack`; script Python in `scripts/`.
- `packaging/` — `make_dmg.sh`, `make_windows.sh`, `make_linux.sh`, icone.

## Documentazione: cosa leggere
- `docs/ARCHITETTURA.md` (≈17 KB): obiettivi, stack, struttura, algoritmi di base, UI, roadmap.
- `docs/storia/ARCHITETTURA_STORIA.md` (≈90 KB): cronologia delle decisioni. **Non leggerla intera**:
  usa grep per parola chiave.
- `docs/STACKING_REPORT.md`: ricerca sullo stacking in stile AutoStakkert (lavoro futuro).

## Cartelle da NON leggere (ignorate da git)
- `target/` — solo build e DMG; `target/dmg/` ha solo l'ultimo DMG.
- `Immagini_test/` — immagini e dataset (`dati_ml/` = dati di addestramento spostati da `target/`,
  non cancellare: non si rigenerano facilmente). `.venv/` — Python.
- `_archivio/` — vecchi DMG, output di esperimenti, esempi di ricerca chiusi (`esempi_ps-io/`).
  Il loro risultato è già in `docs/storia/`; il codice è anche nella storia git.

## Convenzioni
- Documentazione e commenti in italiano; commenti di codice spesso in inglese.
- Verifica rapida: `cargo check --workspace --all-targets`.
