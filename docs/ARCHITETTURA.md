# Planetary Studio — Architettura

> Nome provvisorio. Applicazione desktop (Windows + macOS) per l'**elaborazione** di immagini
> planetarie e lunari già stackate: wavelet, deconvoluzione e regolazioni.
> Luck imaging e stacking sono fuori scope.

## 1. Obiettivi

1. **Fluidità**: ogni slider aggiorna l'anteprima a ≥ 30 fps su un'immagine di 2000×2000.
2. **Qualità**: algoritmi presi dalla letteratura, con i parametri esposti in modo comprensibile.
3. **Non distruttivo**: l'originale non si tocca mai e tutto è una pipeline salvabile.
4. **Estendibile**: aggiungere un nuovo passo di elaborazione = implementare un trait.

## 2. Stack tecnologico (decisioni prese)

| Area            | Scelta                                    | Motivo |
|-----------------|-------------------------------------------|--------|
| Linguaggio      | Rust (edition 2024)                       | Prestazioni native, sicurezza memoria, un solo binario per Win/Mac |
| GUI             | `egui` + `eframe` (backend `wgpu`)        | UI immediate-mode reattiva; l'anteprima è una texture GPU senza copie |
| Calcolo GPU     | `wgpu` compute shader in WGSL             | Metal su macOS, DX12/Vulkan su Windows, stesso codice |
| Calcolo CPU     | `rayon` + loop vettorizzabili             | Export a piena precisione, fallback, riferimento per i test |
| FFT (CPU)       | `rustfft`                                 | Convoluzioni con PSF grandi (RL, Wiener) |
| I/O             | `image` (TIFF/PNG/JPEG), lettore FITS interno | Il FITS è semplice e così evitiamo dipendenze C (cfitsio) |
| Serializzazione | `serde` + `ron`/`json`                    | Preset e progetti |

### Ibrido GPU + CPU

```
               ┌──────────── ps-core ────────────┐
               │ Image (f32 planare, lineare)    │
               │ Pipeline = Vec<Step>            │
               │ trait Op { cpu(), gpu() }       │
               └──────┬───────────────┬──────────┘
                      │               │
          anteprima   ▼               ▼  export / test
              ps-gpu (wgpu WGSL)   ps-core::cpu (rayon)
                      │
                      ▼
              texture egui (zero-copy)
```

- L'**anteprima** gira su GPU, sulla vista corrente (ROI o versione ridotta al livello di zoom).
- L'**export** gira su CPU in f32 a piena risoluzione, oppure su GPU se l'utente lo sceglie.
- Ogni operatore ha un'implementazione CPU **di riferimento**: i test confrontano GPU e CPU
  con una tolleranza (es. errore massimo < 1e-4).

## 3. Struttura del workspace

```
planetary_studio/
├── Cargo.toml              workspace
├── docs/ARCHITETTURA.md
└── crates/
    ├── ps-core/            modello immagine, pipeline, algoritmi CPU (nessuna dipendenza GUI)
    │   └── src/{image.rs, pipeline.rs, wavelet.rs, deconv/, psf.rs}
    ├── ps-io/              lettura/scrittura TIFF, FITS, PNG, JPEG
    ├── ps-gpu/             (fase 2) kernel WGSL + scheduler wgpu
    └── ps-app/             eframe/egui: finestre, pannelli, anteprima
```

`ps-core` non dipende dalla GUI, così domani si può avere anche una CLI di batch
(`ps-cli process *.tif --preset jupiter.ron`).

## 4. Modello dati

- **Immagine**: `Image { width, height, channels: Vec<Vec<f32>> }`, planare, valori in [0,1]
  (le immagini a 8/16 bit sono normalizzate, i float FITS sono riscalati sul loro range).
  Il formato planare rende semplici sia la SIMD sia l'upload su GPU (una texture `R32Float`
  per canale).
- **Pipeline**: lista ordinata di `Step { enabled, op: OpParams }`, serializzabile.
  `OpParams` è un enum: `Wavelet(..)`, `Deconvolution(..)`, `Levels(..)`, …
- **Cache**: si conserva l'output di ogni passo. Se modifichi il passo *k*, si ricalcola
  solo da *k* in poi.

## 5. Algoritmi

### 5.1 Wavelet — Starlet / à trous isotropa

Riferimenti: Starck, Murtagh & Bertero, *Starlet Transform in Astronomical Data Processing*
(Handbook of Mathematical Methods in Imaging, 2011/2015); Starck, Fadili & Murtagh,
*The Undecimated Wavelet Decomposition and its Reconstruction*, IEEE TIP 16(2) 2007.

Algoritmo (per canale):

```
c0 = immagine
per j = 0..J-1:
    c_{j+1} = c_j * h_j        // h_j = kernel con 2^j - 1 zeri inseriti ("à trous")
    w_{j+1} = c_j - c_{j+1}    // dettaglio alla scala j
ricostruzione: out = c_J + Σ_j g_j · T_j(w_j)
```

- **Kernel** selezionabili (filtri separabili 1D):
  - B3-spline `[1, 4, 6, 4, 1] / 16` (default, quello di Starck)
  - Lineare `[1, 2, 1] / 4` (più "duro", come RegiStax)
  - Gaussiano con σ configurabile
- **Passo della scala**: diadico (2^j) o lineare (1 + j·k), per adattarsi al campionamento.
- **Livelli**: da 1 a 8.
- **Per livello**: guadagno `g_j` (0 = neutro, >0 aumenta la nitidezza), soglia di denoise
  `k_j · σ_j`.
- **Stima del rumore**: σ = MAD(w_1) / 0.6745 sul primo livello; per il livello *j*
  σ_j = σ · e_j, con i fattori di propagazione del B3
  (0.8908, 0.2007, 0.0856, 0.0413, 0.0205, 0.0103, …). Per gli altri kernel gli e_j si calcolano
  numericamente all'avvio, trasformando rumore bianco.
- **Soglia**: soft (`sign(w)·max(|w|−t, 0)`) o hard.
- **Bordi**: riflessione speculare (mirror), per evitare aloni sul bordo.
- **Colore**: modalità *luminanza collegata* (si elabora L e si riapplica il rapporto a RGB,
  come in RegiStax++) oppure *RGB separati* con guadagni per canale.

### 5.2 Deconvoluzione

**PSF parametriche** (`psf.rs`), normalizzate a somma 1:
- Gaussiana (σ)
- Moffat (α, β): modella il seeing meglio della gaussiana grazie alle ali più estese
- Disco di Airy da apertura D, lunghezza d'onda λ e scala in arcsec/pixel
  (`I(r) = [2 J1(x)/x]²`, x = π D r / λ), con ostruzione centrale opzionale

**Algoritmi**:

1. **Richardson-Lucy** (Richardson 1972, Lucy 1974):
   `u_{k+1} = u_k · ( PSF^T * ( d / (PSF * u_k) ) )`
   - Convoluzione diretta per PSF piccole (≤ 15 px), FFT per quelle grandi.
2. **Accelerazione Biggs-Andrews** (Biggs & Andrews, *Applied Optics* 36(8) 1997):
   estrapolazione vettoriale `y_k = x_k + α_k (x_k − x_{k−1})`, con
   `α_k = Σ(g_{k−1}·g_{k−2}) / Σ(g_{k−2}²)`, limitata a [0,1). Riportano fino a 40× in meno
   di iterazioni: fondamentale per un'anteprima interattiva. È anche ciò che usa `deconvlucy` di MATLAB.
3. **RL regolarizzato con Total Variation** (Dey et al., *Microscopy Res. Tech.* 2006):
   divide l'aggiornamento per `1 − λ · div(∇u/|∇u|)`; il valore tipico è λ ≈ 0.002.
   Riduce il ringing sui bordi dei dischi planetari.
4. **Damped RL** (White 1994, metodo usato da STSDAS/IRAF): attenua l'aggiornamento dove il residuo
   è dentro N·σ del rumore, così si evita di amplificare la granulosità.
5. **Wiener**: una sola coppia di FFT, `U = conj(H) · D / (|H|² + K)`. Anteprima istantanea.
6. **Van Cittert**: `u_{k+1} = u_k + β (d − PSF * u_k)`. Semplice e storico, utile per confronto.

**Pratiche anti-artefatto**: padding speculare prima della FFT; protezione del disco con una
maschera opzionale (la deconvoluzione si applica solo dove la luminanza supera una soglia, con
bordo sfumato); clamp ≥ 0.

### 5.3 Ordine consigliato di default

`Deconvoluzione → Wavelet → Livelli/Gamma → Colore`
(la deconvoluzione lavora meglio sul dato ancora "lineare", prima della nitidezza wavelet).


## Storia delle decisioni

Le sezioni cronologiche 5bis–5tervicies (auto-impostazioni, maschere, dering, GPU, wavelet, deconvoluzione, nitidezza appresa, riduzione rumore appresa, ecc.) sono in `docs/storia/ARCHITETTURA_STORIA.md`. Cercare per parola chiave con grep, senza leggerle tutte. Indice:

- ## 5bis. Riconoscimento automatico e impostazioni predefinite
- ## 5ter. Moduli aggiuntivi
- ## 5quater. Finitura, livelli e pennello
- ## 5quinquies. Impostazioni automatiche a due livelli
- ## 5sexies. Pannello più ordinato, tono a passi, geometria
- ## 5septies. Dering, ripristino, rimozione dell'automatico
- ## 5octies. Prestazioni (CPU, memoria, anteprima rapida, GPU)
- ## 5novies. Prove su immagini reali e correzioni (cartella `Immagini_test`)
- ## 5decies. Verso la 1.0: bug di stato e qualità della nitidezza
- ## 5undecies. Interfaccia: aree a schede, regole uniformi
- ## 5duodecies. Interfaccia rivista e dering concentrico
- ## 5terdecies. Nomi d'esportazione, confronto A/B, livelli per canale, dering rifatto, mosaico
- ## 5quaterdecies. Spettro dell'immagine e wavelet a scale libere
- ## 5quindecies. Strumentazione, log di ripresa, effemeridi, PSF dallo spettro
- ## 5sexdecies. Deconvoluzione rivista
- ## 5septdecies. Wavelet corrette e "Nitidezza intelligente"
- ## 5octodecies. Anello scuro dentro il disco (collimazione, diffrazione)
- ## 5novodecies. Dering sui pianeti: solo l'anello sotto il bordo
- ## 5vicies. Dering a pennello
- ## 5unvicies. Scala, Identifica, Colorazione solare (2026-11-09)
- ## 5duovicies. Lingua dell'interfaccia: italiano e inglese (2026-11-10)
- ## 5tervicies. Riduzione del rumore appresa (14 novembre 2026)

## 6. Interfaccia

```
┌───────────────────────────────────────────────────────────────┐
│ File  Modifica  Vista  Pipeline  Aiuto          [Prima|Dopo]  │
├────────────┬──────────────────────────────────┬───────────────┤
│ PIPELINE   │                                  │ PARAMETRI     │
│ ☑ Deconv   │                                  │ (del passo    │
│ ☑ Wavelet  │      ANTEPRIMA (pan / zoom)       │  selezionato) │
│ ☐ Livelli  │                                  │               │
│ [+ passo]  │                                  │ slider, curve │
│            │                                  │               │
│ PRESET ▾   │                                  │ ISTOGRAMMA    │
├────────────┴──────────────────────────────────┴───────────────┤
│ 2048×2048 · 16 bit mono · zoom 100% · GPU Metal · 8.2 ms      │
└───────────────────────────────────────────────────────────────┘
```

- Tema scuro di default (si lavora di notte), accento sobrio.
- Confronto prima/dopo con tasto premuto o split verticale trascinabile.
- Doppio clic su uno slider per azzerarlo; Ctrl+Z/Ctrl+Y sulla pipeline.
- Barra di stato con il tempo di elaborazione dell'ultimo frame (misuriamo sempre).

### Strategia di fluidità

Implementata nella fase 1–2:

- **Cache per passo** (`Processor`): ogni passo ricorda input e parametri; se non cambiano,
  il risultato viene riusato. Modificare le wavelet non ricalcola la deconvoluzione.
- **Decomposizione wavelet in cache**: guadagni e soglie cambiano solo la ricomposizione
  (≈ 26 ms su 12 MP, ≈ 1 ms su 0,6 MP).
- **ROI**: si elabora solo la regione più un margine di 64 px, così i bordi si comportano
  come sull'immagine intera.

Tempi misurati su CPU (Intel 6 core, `cargo run --release -p ps-core --example bench`):

| | 800×800 | 4000×3000 |
|---|---|---|
| Decomposizione wavelet (6 livelli) | 17 ms | 230 ms |
| Ricomposizione (slider guadagno) | 1 ms | 26 ms |
| RL 20 iterazioni, accelerato | 165 ms | 4,7 s |

Con la Luna a piena risoluzione la deconvoluzione su CPU non è interattiva: si usa la ROI
finché non arriva la fase 3 (GPU).

1. Il thread UI non calcola mai: invia i parametri a un worker e mostra l'ultimo risultato pronto.
2. Coalescenza: se arrivano 10 eventi slider mentre il worker lavora, si elabora solo l'ultimo.
3. Anteprima progressiva: prima a risoluzione ridotta (1/2 o 1/4), poi a piena risoluzione
   quando lo slider si ferma.
4. Per la deconvoluzione in anteprima si usa un numero di iterazioni ridotto oppure Wiener,
   e il calcolo completo parte al rilascio.

## 7. Roadmap

| Fase | Contenuto |
|------|-----------|
| **0 — Scheletro** ✅ | Workspace, finestra egui, apertura TIFF/PNG/JPEG/FITS, visualizzazione con zoom e pan, export PNG/TIFF 16 bit |
| **1 — Wavelet CPU** ✅ | Starlet completa (B3/lineare/gaussiano, scala diadica/lineare, 1–8 livelli, soglia MAD soft/hard); worker in background con coalescenza e annullamento; prima/dopo/divisa; anteprima intera o ROI |
| **2 — Deconvoluzione CPU** ✅ | PSF Gaussiana/Moffat/Airy; RL + Biggs-Andrews, TV, damping; Wiener; Van Cittert |
| **2bis — Semplificazione** ✅ | Riconoscimento del tipo, impostazioni automatiche, pannello unico con avanzate nascoste, elaborazione completa in background durante la ROI |
| **2ter — Strumenti** ✅ | Ritaglio, allineamento RGB, BlurXTerminator, colore, Mineral Moon, livelli con maschere, impostazioni, schede comprimibili |
| **2quater — Finitura** ✅ | Riduzione rumore per tipo, contrasto, curve e livelli, pennello sulle maschere, lune, finitura per livello, pannello per aree |
| **3 — GPU** | Port WGSL di convoluzione separabile, starlet e RL; test di confronto CPU/GPU |
| **4 — Pipeline** | Passi riordinabili, preset, progetto `.psproj`, undo/redo |
| **5 — Extra** | Livelli/curve, bilanciamento colore, allineamento RGB, derotazione, batch CLI, XISF |

## 8. Bibliografia

- J.-L. Starck, F. Murtagh, M. Bertero — *Starlet Transform in Astronomical Data Processing:
  Application to Source Detection and Image Deconvolution*, Handbook of Mathematical Methods
  in Imaging, Springer.
- J.-L. Starck, J. Fadili, F. Murtagh — *The Undecimated Wavelet Decomposition and its
  Reconstruction*, IEEE Trans. Image Processing 16(2):297–309, 2007. doi:10.1109/TIP.2006.887733
- F. Murtagh, J.-L. Starck, A. Bijaoui — *Multiresolution in astronomical image processing:
  a general framework*, Int. J. Imaging Syst. Technol., 1995.
- W. H. Richardson — *Bayesian-Based Iterative Method of Image Restoration*, JOSA 62(1), 1972.
- L. B. Lucy — *An iterative technique for the rectification of observed distributions*, AJ 79, 1974.
- D. S. C. Biggs, M. Andrews — *Acceleration of iterative image restoration algorithms*,
  Applied Optics 36(8):1766–1775, 1997.
- N. Dey et al. — *Richardson-Lucy algorithm with total variation regularization for 3D
  confocal microscope deconvolution*, Microscopy Research and Technique 69(4), 2006.
- R. L. White — *Image restoration using the damped Richardson-Lucy method*, SPIE 2198, 1994.
- A. F. J. Moffat — *A Theoretical Investigation of Focal Stellar Images in the Photographic
  Emulsion*, A&A 3, 1969.
- PixInsight — *Deconvolution and Noise Reduction Example with M81 and M82* (deringing globale
  e locale), https://pixinsight.com/examples/M81M82/
- L. Yuan, J. Sun, L. Quan, H.-Y. Shum — *Progressive Inter-scale and Intra-scale Non-blind Image
  Deconvolution*, ACM TOG (SIGGRAPH) 2008 (Richardson–Lucy bilaterale contro il ringing).
- J.-L. Roche et al. — *Lower Bound Constraint Algorithm* per ridurre il ringing con sfondo
  elevato, arXiv astro-ph/0306419.
- British Astronomical Association — *Mars edge rind* (2020), causa e trattamento del bordo.
- SharpCap — de-ring a soglia nel live stacking planetario (forum, 2023).

## 9. Pacchetti

- macOS: `packaging/make_dmg.sh` (`--universal` per Intel + Apple Silicon) → `target/dmg/*.dmg`
- Windows: `packaging/make_windows.sh` (cross-compilazione con mingw-w64) → `target/windows/*.zip`

## Area Maschere (ridisegno dei livelli)

Nell'interfaccia i livelli si chiamano **maschere** ("livelli" restava ambiguo con i livelli
wavelet e i livelli tonali); nel codice restano `Layer`. Prima ogni livello apriva il suo editor
dentro la riga, con cinque riquadri annidati, e Anelli e Lune erano schede a parte: non si
capiva cosa fosse acceso e cosa facesse.

- **Elenco** (`panels::masks_area`): una riga per maschera con maniglia di trascinamento,
  interruttore, miniatura della zona, nome (etichetta "auto" per Anelli/Lune), riga di stato e
  occhio. La riga di stato dice cosa cambia ("Wavelet · Contrasto"), "Come l'immagine intera",
  oppure un avviso: "Zona vuota" o "Non cambia nulla". Spenta = riga attenuata.
- **Ordine**: le maschere si fondono nell'ordine dell'elenco, quindi dove si sovrappongono vince
  la più in basso; si riordinano trascinando la maniglia (drag and drop di egui).
- **Miniature**: il worker riduce a 56 px ogni maschera già calcolata (`worker::mask_thumb`,
  in cache per generazione) e le manda nell'output; l'app le tiene come texture per id e le
  conserva anche quando la maschera si spegne.
- **Editor**: scheda separata sotto l'elenco, con ZONA (strumenti della maschera) e
  REGOLAZIONI: sei righe sempre visibili, ciascuna "proprie"/"come l'immagine" con
  interruttore, una sola aperta alla volta. Niente riquadri annidati: solo intestazioni e un
  rientro con una linea.
- **Preferenze**: le vecchie voci dei moduli Anelli/Lune si scartano al caricamento, così un
  "nascosto" rimasto non spegne più la loro maschera (c'è un test).
