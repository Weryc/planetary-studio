# Planetary Studio — elaborazione e nitidezza planetaria per Mac, Windows e Linux

**Elaborazione veloce e non distruttiva di immagini planetarie e lunari già stackate: nitidezza con wavelet, deconvoluzione, riduzione del rumore e mosaici lunari — un'alternativa gratuita a RegiStax e AstroSurface per macOS, Windows e Linux.**

🇬🇧 [Read in English](README.md)

![Planetary Studio mentre elabora Saturno su macOS: nitidezza planetaria con wavelet e deconvoluzione](docs/screenshots/main.png)

Planetary Studio parte da dove finisce il tuo programma di stacking (AutoStakkert!, RegiStax, …): rende nitido, ripulisce e rifinisce il TIFF stackato di Giove, Saturno, Marte o della Luna — l'astrofotografia planetaria dopo il lucky imaging. Riconosce cosa hai aperto, propone impostazioni sensate e spiega perché, lasciando sempre a te ogni cursore.

> Lucky imaging e stacking sono fuori dallo scopo: apri un'immagine già stackata.

## Download

Scarica l'ultima versione dalla **[pagina delle Release](../../releases/latest)**:

| Sistema | File | Note |
|---|---|---|
| macOS (Intel e Apple Silicon) | `PlanetaryStudio-<versione>-universal.dmg` | Apri il DMG e trascina *Planetary Studio* in *Applicazioni*. |
| Windows 10/11 (64 bit) | `PlanetaryStudio-<versione>-windows-x64.exe` | Portatile: niente installazione, basta avviarlo. |
| Linux (x86_64) | `PlanetaryStudio-<versione>-linux-x86_64.AppImage` | Rendilo eseguibile (`chmod +x`) e avvialo. C'è anche un `.tar.gz` con il solo programma. Serve un driver grafico con Vulkan (va bene Mesa). |

**Primo avvio.** L'app non è firmata con un certificato Apple/Microsoft a pagamento, quindi la prima volta il sistema mostra un avviso:

- **macOS** — clic destro sull'app → *Apri* → *Apri*. Sulle versioni recenti: *Impostazioni di Sistema → Privacy e sicurezza → Apri comunque*.
- **Windows** — nella finestra di SmartScreen fai clic su *Ulteriori informazioni* → *Esegui comunque*.

## Funzioni

- **Riconoscimento automatico** del soggetto (pianeta, Saturno, disco intero, superficie lunare), con impostazioni strutturali e la spiegazione di ogni scelta automatica.
- **Deconvoluzione** — Richardson–Lucy (accelerata, con regolarizzazione TV e smorzamento) e Wiener; PSF misurata dall'immagine, disegnando un riquadro o cliccando una stella; anti-ringing integrato.
- **Wavelet** — starlet isotropa (*à trous*) con scale libere posizionate in base allo spettro dell'immagine, oppure scale classiche 1-2-4-8; guadagni adattivi al rumore e protezione dei bordi.
- **Nitidezza finale** — lavora sul risultato di tutte le altre regolazioni di nitidezza e rinforza proprio le scale dove c'è dettaglio reale, fermandosi al limite misurato sullo spettro dell'immagine.
- **Riduzione del rumore** per tipo di soggetto, **dering** a pennello: dipingi sopra l'anello scuro (dentro il bordo, nel cielo attorno, accanto a un cratere) e viene misurato e schiarito fino a ciò che ha accanto, senza toccare bordi e bande, **curve e livelli** (anche per canale), **colore**, allineamento RGB, Mineral Moon.
- **Maschere** — elabora a parte gli anelli di Saturno, il globo, le lune o una zona dipinta. Un elenco chiaro mostra di ogni maschera la miniatura, cosa cambia e l'ordine (si riordina trascinando).
- **Confronto A/B**, vista prima/dopo divisa, anteprima su ROI.
- **Strumentazione e diagnosi** — catalogo telescopi/camere, lettura dei log di FireCapture/SharpCap/AutoStakkert!, scala dell'immagine dalle effemeridi, grafico dello spettro.
- **Mosaici** di riprese lunari fatte con la stessa focale.
- **Formati**: TIFF 8/16/32 bit, FITS, PNG, JPEG.
- **Accelerazione GPU** (Metal, DirectX 12, Vulkan tramite wgpu) con ripiego automatico sulla CPU.
- Collegamento facoltativo a **BlurXTerminator**, se possiedi una licenza della sua versione a riga di comando.

## Galleria

| | |
|---|---|
| ![Giove prima e dopo nitidezza con wavelet e deconvoluzione, vista divisa](docs/screenshots/jupiter-before-after.png) | ![Saturno con gli anelli elaborati con una maschera dedicata](docs/screenshots/saturn-masks.png) |
| **Giove** — prima/dopo, vista divisa | **Saturno** — anelli resi nitidi con una maschera propria |
| ![Dettaglio di crateri lunari prima e dopo la nitidezza](docs/screenshots/moon-detail.png) | ![Mosaico lunare composto da due riprese](docs/screenshots/lunar-mosaic.png) |
| **Luna** — dettaglio dei crateri, prima/dopo | **Mosaico lunare** — riprese allineate e fuse |

![Vista prima/dopo divisa](docs/screenshots/compare.png)

## Compilare dai sorgenti

Serve [Rust](https://rustup.rs) (stable).

```bash
cargo run --release -p ps-app
```

Pacchetti:

```bash
packaging/make_dmg.sh --universal   # DMG per macOS, Intel + Apple Silicon
packaging/make_windows.sh           # .exe per Windows, cross-compilato da macOS (serve mingw-w64)
packaging/make_linux.sh             # AppImage + tar.gz per Linux (da Linux; per le release lo fa GitHub Actions)
```

Il progetto tecnico (algoritmi, pipeline, roadmap, bibliografia) è in [docs/ARCHITETTURA.md](docs/ARCHITETTURA.md).

## Licenza

Copyright © 2026 Weryc. Il codice di Planetary Studio è **consultabile** e distribuito con la **[PolyForm Noncommercial License 1.0.0](LICENSE.md)**.

In breve (fa fede il testo della licenza, in inglese):

- ✅ Puoi usarlo **gratis per qualsiasi scopo non commerciale** — hobby, studio, ricerca, scuole, associazioni, enti non profit.
- ✅ Puoi leggere il codice e modificarlo per scopi non commerciali, e condividerlo senza scopo di lucro mantenendo la licenza e la riga `Required Notice`.
- ❌ **Non puoi venderlo** né usarlo per scopi commerciali.
- © Il diritto d'autore e la proprietà intellettuale restano dell'autore.

Per un uso commerciale, o per qualsiasi uso non coperto dalla licenza, [apri una issue](../../issues) per chiedere il permesso.

Le librerie di terze parti mantengono le proprie licenze: vedi [THIRD_PARTY.md](THIRD_PARTY.md).
