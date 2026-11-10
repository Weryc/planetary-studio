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

## 5bis. Riconoscimento automatico e impostazioni predefinite

All'apertura (`ps-core/src/analysis.rs`, in parallelo alla preparazione della texture):

1. **Segmentazione**: luminanza ridotta a ≤ 512 px, soglia di Otsu, componente connessa più grande
   (le lune di Giove e le stelle vengono ignorate).
2. **Tipo**:
   - soggetto che tocca i bordi per > 25 % o copre > 80 % del fotogramma → *Superficie*;
   - bounding box allungato (> 1,6:1) e simmetrico rispetto al centro, diametro < 900 px →
     *Saturno* (una falce di Luna è asimmetrica e viene esclusa);
   - diametro ≥ 900 px → *Disco intero* (Luna o Sole), altrimenti *Pianeta*.
3. **Rumore**: σ dalla MAD del primo livello starlet; SNR = mediana del soggetto / σ.
4. **Sfocatura**: per dischi rotondi si misura la FWHM della derivata del profilo del bordo
   su 96 raggi (ogni raggio allineato al proprio bordo, così anche dischi schiacciati come
   Giove funzionano); scartati i raggi a basso contrasto (terminatore). σ = FWHM / 2,355.

`ps-core/src/preset.rs` traduce tipo + SNR + σ in parametri: preset per tipo, riduzione rumore e
damping scalati con 60/SNR, PSF = 0,85 × σ misurata (sottostimare la PSF costa solo un po' di
nitidezza, sovrastimarla crea aloni).

L'utente può cambiare tipo con un clic; "Ripristina impostazioni automatiche" riporta ai
valori calcolati.

## 5ter. Moduli aggiuntivi

Pipeline (`ps-core/src/pipeline.rs`), ogni fase in cache su (input, parametri):

```
sorgente → allineamento RGB → BlurXTerminator → deconvoluzione ─┬→ wavelet → colore ──────────────┐
                                                                └→ livello k: wavelet_k → colore_k ┤ fusione con maschera
                                                                                                   └→ Mineral Moon → uscita
```

- **Allineamento RGB** (`align.rs`): R e B vengono registrati sul canale G con una correlazione di
  fase parzialmente sbiancata (|C|^0,5; quella pura dà un picco debole e instabile sulle immagini
  sfocate, perché lì le alte frequenze sono solo rumore). Seguono un fit gaussiano a 3 punti e
  un raffinamento iterativo, poi lo spostamento si applica con Catmull–Rom. Precisione ≈ 0,1 px;
  il modulo si attiva da solo sopra 0,25 px.
- **BlurXTerminator** (`bxt.rs`): chiama `rc-astro bxt` con file TIFF a 32 bit float (il FITS
  tornava capovolto). I nomi delle opzioni vengono letti dall'`--help` della versione installata
  (1.x: `--nonstellar-radius`; 2.6+: `--nonstellar-diameter`, `--lunar-planetary`). La PSF
  automatica di BXT si basa sulle stelle, quindi per i pianeti il diametro viene dalla FWHM
  misurata sul bordo. Il processo si può annullare e gli errori diventano avvisi nella scheda.
- **Colore** (`adjust.rs`): saturazione e vividezza in Y + (C − Y), bilanciamento grey-world sul
  soggetto.
- **Mineral Moon**: saturazione forte, pulizia della crominanza (residuo starlet su Cb e Cr),
  rotazione della tinta, curva tonale monotona (Fritsch–Carlson), LRGB con una luminanza mono
  registrata automaticamente.
- **Livelli** (`mask.rs`): maschere automatiche (soggetto pieno con riempimento dei buchi,
  sfondo; globo e anelli di Saturno dalla geometria, con raggio dell'anello A = 2,27 × raggio
  del globo e schiacciamento 0,90) oppure ellissi e rettangoli, con sfumatura (3 box blur ≈
  gaussiana) e inversione. La deconvoluzione resta unica; ogni livello ha wavelet e colore propri.
- **Ritaglio**: non distruttivo (l'originale resta in memoria) e sposta di conseguenza le forme
  delle maschere.
- **Impostazioni** (⌘, / Ctrl+,): moduli preferiti, visibili o nascosti (i nascosti non vengono
  applicati), percorso di `rc-astro` con verifica della licenza, scelta automatica delle impostazioni.

## 5quater. Finitura, livelli e pennello

**Catena di finitura**, uguale per l'immagine intera e per ogni livello:
riduzione rumore → wavelet → contrasto → curve e livelli → colore (`enhance.rs`).
In un livello, ogni modulo attivato sostituisce quello globale; gli altri si ereditano. La
deconvoluzione resta unica, perché la sfocatura è la stessa ovunque.

- **Riduzione rumore**, con un metodo per ciascun tipo di soggetto:
  - *Pianeti*: soglia morbida sulle 3 scale starlet più fini (k = 2,2 / 1,4 / 0,7 σ). La grana
    fine sparisce, mentre bande e festoni, che stanno sulle scale più grandi, restano intatti.
  - *Luna e Sole*: filtro bilaterale (σs = 1,5 px, σr ∝ rumore misurato), che pulisce le zone
    piatte senza ammorbidire bordi di crateri, ombre e granulazione; "Dettaglio" restituisce una
    parte della trama fine.
  - In entrambi: pulizia del rumore di colore (solo Cb e Cr) e del cielo nero.
  - I preset scelgono il metodo in base al tipo e scalano l'intensità con l'SNR.
- **Contrasto**: maschera di contrasto (raggio e soglia anti-rumore) e contrasto locale a raggio
  ampio, sulla luminanza.
- **Curve e livelli**: nero, bianco, gamma e curva monotona; "Automatico" mette il nero sul cielo
  e il bianco sullo 0,05 % più luminoso.
- **Maschere**: le pennellate sono vettoriali (punti, raggio, durezza, aggiungi o togli) e si
  applicano sopra qualsiasi forma, compreso "Pennello", una maschera vuota da dipingere. Si
  possono salvare con le impostazioni, annullare e restano corrette dopo un ritaglio.
- **Lune**: corpi compatti più luminosi di 8σ rispetto al cielo e separati dal pianeta (così si
  trovano anche lune deboli come Titano). "Pianeta / soggetto" ora contiene solo il corpo
  principale, senza le lune.
- **Pannello per aree**: Preferiti, Preparazione, Nitidezza, Rumore, Tono e colore, Livelli;
  ogni area e ogni scheda si può comprimere.
- **Bug del pannello che si allargava**: in un pannello ridimensionabile, un contenuto anche di
  1 pt più largo lo fa crescere a ogni frame. Ora gli slider lasciano margine, il contenuto è
  vincolato e il pannello ha una larghezza massima. Un test lo verifica in 300 frame e, senza la
  correzione, il pannello arriva a 1371 px.
- **Simboli ⌘ ⇧ ⌥ ″**: mancano nei font di egui. Ora vengono presi da un font di simboli di
  sistema (Apple Symbols / Segoe UI Symbol), e un test verifica che tutti i caratteri speciali
  dell'interfaccia esistano.

## 5quinquies. Impostazioni automatiche a due livelli

- **Strutturali**, sempre scelte dall'analisi all'apertura (`preset::initial_pipeline`):
  - forma e dimensione della PSF (dalla sfocatura misurata sul bordo), algoritmo, TV,
    accelerazione;
  - wavelet: numero di livelli tale che la scala più grossa sia circa 1/10 del soggetto
    (4–8), passo iniziale 2 se l'immagine è sovracampionata (σ > 2,6), kernel lineare se è
    sottocampionata (σ < 1), progressione diadica;
  - metodo di riduzione rumore per tipo, diametro PSF per BlurXTerminator, spostamento RGB
    misurato.
- **Semplici** (nitidezza delle wavelet, iterazioni, intensità dell'antirumore…): all'apertura
  restano neutre o spente, quindi l'immagine non viene toccata. Si impostano a mano, oppure con
  "Impostazioni automatiche" (tutto) o con il pulsante Auto nell'intestazione di ogni modulo.
  L'opzione "Applica subito…" nelle Impostazioni ripristina il comportamento precedente.
- **Cambio di tipo** a mano: si ricalcola solo la struttura e le regolazioni fatte restano
  (`preset::restructure`).
- **Moduli adattati all'immagine**: per le immagini mono niente Allineamento RGB, Colore e
  Mineral Moon (le Curve sì); Mineral Moon solo per Luna e Sole; nei livelli le Lune solo per i
  pianeti, Globo e Anelli solo per Saturno. Si può disattivare nelle Impostazioni.
- "Tono e colore" (che ora contiene l'allineamento RGB) è la prima area del pannello.
- **Slider della nitidezza wavelet**: ora è lineare (−1…8). Con la scala logaritmica tre quarti
  della corsa stavano sotto un guadagno di 1, dove l'effetto non si vede.
- **Pannello che "ballava" durante l'elaborazione**: il pulsante di ripristino compariva e
  spariva ogni volta che un valore passava sopra o lontano da quello automatico, e la barra di
  avanzamento si inseriva dentro la scheda: entrambi spostavano il contenuto sotto lo slider.
  Ora il pulsante è sempre presente e la barra ha uno spazio fisso; il test
  `layout_is_still_while_processing` lo verifica frame per frame.
- **Icona**: senza un'icona impostata nella finestra, eframe metteva la propria nel Dock dopo
  l'avvio; ora è incorporata nell'eseguibile.

## 5sexies. Pannello più ordinato, tono a passi, geometria

- **Aree**: Nitidezza, Rumore, Tono, Colore, Livelli (in ordine di elaborazione). I preferiti
  sono vuoti di default. Le intestazioni sono barre ben distinte, con il numero di moduli
  attivi; con "una sola area aperta" (attivo di default) aprirne una chiude le altre.
- **All'apertura tutto è spento**: restano impostati solo i parametri strutturali.
  *Bug*: il campo `version` delle preferenze usava il default del contenitore, cioè la
  versione corrente, quindi le preferenze vecchie non venivano migrate e l'applicazione
  automatica restava accesa. Ora il default è a livello di campo (0); c'è un test.
- **"Perché queste impostazioni"** sotto "Avanzate": testi generati dall'analisi
  (`preset::explain`). Per esempio il numero di livelli rispetto alla dimensione del soggetto,
  il passo iniziale rispetto alla sfocatura, la PSF come 85 % della σ misurata, il metodo
  antirumore scelto per il tipo di soggetto.
- **Tono**: le regolazioni si possono applicare a passi ("Applica e continua"); ogni passo resta
  in una pila, si può riprendere ("Indietro") o togliere. L'istogramma dietro la curva è quello
  dell'immagine su cui lavora la regolazione corrente. Contagocce per il punto del nero e del
  bianco (media 3×3 della luminanza nel punto cliccato).
- **Anelli di Saturno e Lune**: maschere preconfigurate (`LayerRole`), proposte in cima al menu
  "Aggiungi maschera" (prima erano schede a parte). Anelli: wavelet sulle scale da 2–4 px e un
  po' di contrasto. Lune: un po' di nitidezza e toni medi alzati (le lune sono molto più deboli
  del pianeta). "Valori consigliati" le rimette a posto senza cambiare nome, posizione,
  interruttore e pennellate.
- **BlurXTerminator**: stato visibile ("passaggio k di n"), da 1 a 3 passaggi consecutivi.
- **Spazio**: con il puntatore sull'immagine mostra l'originale anche se un campo ha il focus,
  e non "preme" più il pulsante o l'interruttore che ha il focus; un'etichetta "PRIMA" lo
  conferma. C'è un test.
- **Geometria** (`geometry.rs`): rotazioni di 90° e specchiature senza perdita, raddrizzamento
  fine (bilineare, ±45°), dalla barra del ritaglio o dal menu Modifica. Ordine:
  originale → orientamento → ritaglio.

## 5septies. Dering, ripristino, rimozione dell'automatico

**Dering** (`ps-core/src/dering.rs`, area Nitidezza, solo per Pianeta/Saturno/Disco intero).
Deconvoluzione e wavelet amplificano le oscillazioni di Gibbs sul bordo netto del disco: anello
scuro appena dentro il lembo, bordo chiaro ("rind"), alone nel cielo. Diffrazione (primo anello
di Airy) e collimazione imperfetta (anello asimmetrico) mettono già un anello debole nei dati,
che la nitidezza rinforza; il documento BAA sul "Mars edge rind" nota che è più visibile con
elaborazione forte e buon seeing, e più largo alle lunghezze d'onda lunghe.

Metodo: *clamp anti-ringing*, la stessa idea del deringing globale di PixInsight e dei filtri
anti-ringing del ricampionamento. Il ringing crea valori fuori dall'intervallo che l'immagine non
elaborata ha attorno al pixel; il risultato viene quindi tenuto tra minimo e massimo locali del
riferimento (raggio ≈ σ della sfocatura), più una tolleranza proporzionale al range locale.
Agisce solo in una fascia lungo il bordo (maschera del soggetto sfumata, ±5σ) e, a scelta, nel
cielo; l'interno del disco non viene toccato. Opzione "su tutta l'immagine" per la collimazione.
Riferimento = immagine dopo l'allineamento RGB, prima di BXT/deconvoluzione. Posizione nella
catena: dopo il contrasto e prima di curve/livelli, anche nei livelli (con le impostazioni globali).
Strutturali automatiche: raggio, fascia, tolleranza (più alta per Saturno: Cassini è dettaglio vero).
Test: con R–L e PSF sovrastimata del 40 % il bordo chiaro scende da 0,76 a 0,09 (plateau 0,7).

**Ripristino.** Ogni cursore ha un pulsante ↺ (visibile quando il valore è diverso dal
predefinito, spazio sempre riservato). Ogni scheda ha ↺ in testata: riporta i parametri del
modulo al punto di partenza dell'immagine (strutturali automatiche, quantità neutre) senza
cambiare l'interruttore.

**Rimuovi automatiche.** Accanto a "Impostazioni automatiche": torna a `neutral_pipeline()`
(valori generici, tutto spento, i livelli restano). Nelle Impostazioni si può disattivare anche
la scelta automatica delle impostazioni avanzate all'apertura (`auto_structure`).

**Pannello.** Nell'area Maschere c'è un elenco piatto e, sotto, l'editor della sola maschera
selezionata (vedi "Area Maschere"). Le aree hanno titolo maiuscolo e
barra alta; le schede sono rientrate sotto l'area con una linea guida; "Avanzate" è un riquadro
annidato nella scheda.

## 5octies. Prestazioni (CPU, memoria, anteprima rapida, GPU)

Misure su 12 MP RGB, i5-8500 6 core, Radeon Pro 570X (`cargo run --release -p ps-core --example bench`).

- **R–L gaussiano separabile (CPU).** Una PSF gaussiana è separabile: due passate 1-D al posto
  delle FFT (`deconv::Blur::Separable`, taps = somme per colonna del kernel 2-D, identiche). Buffer
  allocati una volta sola per tutte le iterazioni. 20 iterazioni: 4,8 s → 1,6 s. Il risultato
  coincide con quello FFT entro il rumore numerico di R–L stesso (test `separable_gaussian_matches_fft`:
  la differenza è minore di quella che la FFT ha con sé stessa spostando l'ingresso di 1e-7).
  Moffat, Airy e Wiener restano su FFT.
- **Risparmia memoria** (Impostazioni, spento di default): `Processor::memory_saving` tiene in
  cache solo deconvoluzione, BXT, rumore, una scomposizione wavelet e i livelli.
- **Anteprima rapida** (Impostazioni, spenta di default): mentre si trascina un controllo del
  pannello l'immagine intera è elaborata ridotta 2× (4× sopra 3000 px) con
  `Pipeline::downscaled` (PSF, raggi, spostamenti RGB scalati; le scale wavelet scendono di un
  livello per dimezzamento). Al rilascio si torna a piena risoluzione; i risultati ridotti non
  sono mai usati per esportare, per il contagocce o per i valori dei pixel.
- **GPU** (`ps-core/src/gpu`, wgpu 30: Metal su Mac Intel e Apple Silicon, DX12/Vulkan su
  Windows, Vulkan su Linux). Prima fase: R–L con PSF gaussiana (accelerazione, TV, damping) in
  `deconv.wgsl`, un kernel per passo, un'iterazione per invio (annullamento e progresso restano
  reattivi). Fallback automatico alla CPU senza GPU, con GPU software, se l'immagine non entra
  in un buffer o se qualcosa fallisce. Test `gpu_matches_cpu` (saltato senza GPU). 20 iterazioni
  12 MP: 0,74 s (≈21 ms/iterazione + ≈0,3 s fissi di trasferimento). Interruttore nelle
  Impostazioni; `PS_NO_GPU=1` la disattiva. Il piano osservato viene inviato una volta sola
  (le stime iniziali si creano sulla GPU): 20 iterazioni 0,63 s, costo fisso ≈0,22 s.
- **GPU, seconda fase** (`gpu/ops.rs`, `ops.wgsl`): i filtri su cui si basano gli altri moduli —
  scale starlet (wavelet, rumore Pianeti, pulizia colore, stima del rumore), sfocatura
  gaussiana a tre box (contrasto, maschere, cielo, dering), filtro bilaterale (rumore Luna e
  Sole), minimo/massimo locale (dering). Ognuno ha il fallback CPU e un test di equivalenza
  (`gpu_filters_match_cpu`; min/max identico al bit).
  Misura sulla Radeon Pro 570X: inviare un piano da 12 MP costa ≈43 ms e riaverlo ≈50 ms, tanto
  quanto sfocarlo sulla CPU. Perciò le operazioni leggere (`gpu::Work::Light`) usano la GPU solo
  se condivide la memoria con la CPU (Apple Silicon, grafica integrata); quelle pesanti
  (`Work::Heavy`: bilaterale, R–L) sempre. Sotto 512×512 pixel resta tutto sulla CPU.
  Pipeline 12 MP: CPU 4,5–5,4 s → GPU 2,2 s. Curve, livelli e colore restano sulla CPU (30–50 ms,
  meno di un trasferimento). Un ulteriore passo sarebbe tenere le immagini sulla GPU tra un
  passo e l'altro, ma richiede di riscrivere la cache della pipeline.

## 5novies. Prove su immagini reali e correzioni (cartella `Immagini_test`)

Tre immagini dell'utente: Luna 3888×2192 (primo piano con lembo), due Saturno (1176×1112 e
992×704). Elaborazione automatica completa su GPU: 1,5 s la Luna, 0,23 s e 0,14 s i Saturni
(CPU: 3,7 / 0,35 / 0,2 s).

- **Riconoscimento di Saturno.** Con gli anelli inclinati di ~45° il riquadro del soggetto è
  quasi quadrato e Saturno veniva preso per un pianeta. L'allungamento ora viene dagli assi
  principali (autovalori della covarianza dei pixel del soggetto), indipendente dalla
  rotazione; la dimensione è l'estensione lungo l'asse maggiore. Test con anelli a 30–90°.
- **Rumore.** Gli stacker portano il cielo quasi a zero: sull'85–90 % dei pixel non c'è rumore e
  la stima MAD (σ ≈ 0,00001) descriveva il cielo. Ora si misura dove c'è segnale (pixel scelti
  sul primo livello liscio, non sui valori rumorosi, per non introdurre bias): σ ≈ 0,0015.
- **Sfocatura su Saturno.** Misurata sul bordo del globo (ellisse da `saturn_globe`), saltando i
  raggi entro 40° dall'asse degli anelli: σ ≈ 6,5 px sulle immagini di prova (prima: valore
  tipico 1,3). Con la PSF misurata la nitidezza automatica mostra la divisione di Cassini.
- **Allineamento RGB.** Su immagini morbide il picco di correlazione è largo e la stima vaga di
  0,2–0,4 px. Per soggetti interi nel campo si usano i baricentri dei canali sulla stessa
  regione (dal verde, allargata), con il livello del cielo di ogni canale sottratto; la
  correlazione di fase resta per le superfici.
- **Colore automatico.** Il grey world rendeva Saturno grigio-azzurro. Luna e Sole: grey world
  pieno; pianeti e Saturno: metà correzione in scala logaritmica (`preset::auto_color`).
- **Dering su Saturno.** L'area è una maschera visibile e ritoccabile col pennello
  (`dering::weight`, id `pipeline::DERING_MASK`, calcolata sull'immagine intera come le maschere
  dei livelli): fascia lungo il bordo della regione scelta (soggetto, globo, anelli) più il
  cielo. Predefinito su Saturno: solo il globo.
- **Pannello.** Moduli accesi ma neutri all'apertura (`initial_pipeline`), evidenziati solo
  quando cambiano l'immagine; schede chiuse all'inizio e una sola aperta per area; Livelli
  tonali e Curve separati (`Pipeline::curves`, piegati con i livelli in un'unica tabella);
  moltiplicatore della nitidezza wavelet (`WaveletParams::amplify`, ×1–×5).

## 5decies. Verso la 1.0: bug di stato e qualità della nitidezza

**Bug corretti**
- Puntini colorati nel cielo (2.501 → 10 sul Saturno di prova): la nitidezza in luminanza ora
  scala i canali quando scurisce (`color::merge_planes`), la saturazione si spegne nelle ombre
  profonde (`adjust::shadow_safe`) e alla fine i colori fuori 0‥1 vengono desaturati invece che
  tagliati canale per canale (`color::fit_gamut`, ultimo passo della pipeline).
- Strumenti che restavano attivi: il pennello e la maschera mostrata si ritirano quando i loro
  controlli non sono più a schermo (`settle_tools`, flag `MASK_TOOLS_SHOWN`); selezione di una
  maschera che non esiste più (undo) azzerata; contagocce, ritaglio e pennello si escludono;
  all'apertura di un'immagine tutto torna a riposo; una pennellata interrotta non resta in
  sospeso nel visore.

**Wavelet**: nitidezza adattiva al rumore (guadagno pieno solo oltre ~2σ del livello,
r²/(1+r²)) e protezione dei bordi (mappa dei bordi a grande scala, `Decomposition::edges`).

**Deconvoluzione**: arresto automatico (principio di discrepanza sul soggetto), anti-ringing
nelle iterazioni (limiti locali dell'osservato nell'area del dering), Wiener con K automatico,
PSF per canale in R·G·B, PSF misurata da una stella o luna (`psf::measure`), forma della PSF
stimata dal bordo (Moffat contro gaussiana, `analysis::fit_moffat`). La Moffat è applicata
come somma di 3–5 gaussiane (`Psf::gaussian_mixture`), separabili su CPU e GPU. Una versione
delle convoluzioni GPU con memoria condivisa è stata provata e scartata: sulla Radeon con
Metal era molto più lenta.

**Dering**: mappa degli aloni (`dering::ringing_map`), limitata al bordo e al cielo.

**Interfaccia**: intensità automatica Delicato/Normale/Forte (`preset::AutoStrength`), avviso
pixel bruciati, confronto A/B con istantanea.

## 5undecies. Interfaccia: aree a schede, regole uniformi

Scelte dell'utente (2026-11-06):
- **Pannello a schede**: in alto il tipo d'immagine (menu a tendina, "riconosciuto"/"scelto a
  mano", dettagli dell'analisi al passaggio del mouse) e il pulsante Automatico con freccia
  (intensità Delicato/Normale/Forte, "Rimuovi impostazioni automatiche"); poi una fila di
  schede per le aree (Nitidezza · Rumore · Tono · Colore · Livelli, con il numero di moduli
  attivi); sotto scorrono solo le schede dell'area scelta, una aperta alla volta. Tasti 1–5.
- **Intestazione delle schede**: pallino di stato (pieno arancione = agisce, vuoto = nessun
  effetto, barrato = escluso; clic per escludere) e menu ⋯ (Automatico, Ripristina). Doppio
  clic sul titolo = escludi/riattiva. Le sezioni dei livelli tengono l'interruttore, che lì
  vuol dire "impostazioni proprie".
- **Barra in alto con etichette**: Apri, Esporta (rapido: accanto all'originale, nel formato
  dell'ultima volta, mai sovrascrivendo, ⌘⇧E), Intera/ROI, Ritaglia, Dopo/Prima/Divisa/A-B,
  Istantanea, Bruciati, zoom (100 % = ⌘1), Impostazioni. "Esporta come…" nel menu File (⌘E).
- **Regole uniformi**: scala di spaziature (`theme::space` XS/S/M/L) e di testo
  (`theme::size`); aiuti di una frase, il resto al passaggio del mouse; nomi tecnici.
- Pannello più largo (420 pt di default). Istogramma generale invariato.

## 5duodecies. Interfaccia rivista e dering concentrico

Interfaccia (2026-11-06, seconda tornata):
- Intestazione delle schede: interruttore acceso/spento, poi icone Automatico e Ripristina
  sempre visibili (niente più pallino né menu ⋯).
- "Tono" diventa **Curve e livelli**. Istogramma grande che segue in tempo reale le
  regolazioni (con il profilo di prima in trasparenza), B/N o a colori; curve RGB, R, G, B
  (`ToneStep::channel_curves`).
- Riga d'inizio: interruttore "Impostazioni avanzate automatiche accese/spente" (colore solo
  se accese) e pulsante "Regola in automatico · intensità" con freccia.
- Scheda **Esporta** nel pannello (formato, suffisso, esporta/esporta come, preset).
- Barra in alto solo azioni (Apri | Ritaglia, Istantanea, Bruciati | Impostazioni); sopra
  l'immagine barra di visualizzazione (Intera/ROI, Dopo/Prima/Divisa/A-B); zoom in basso a
  destra.

Dering (`dering.rs`), riprogettato per il ringing *dentro* il disco, che sulle immagini reali
è concentrico al bordo, spesso su un solo lato, a volte una banda scura più interna:
1. **Disco**: Saturno → ellisse dalla geometria degli anelli (`saturn_globe`), poi riscalata
   dove la luminosità scende a metà lungo i raggi che arrivano al cielo (`fit_limb`); altri
   pianeti → ellisse dai momenti dei pixel sopra metà della luminosità mediana del soggetto,
   accettata solo se combacia con il soggetto (un primo piano che esce dal campo no). Calcolata
   sull'intera immagine e passata in `Aux::globe` (spostata per la ROI, ridotta per
   l'anteprima rapida).
2. **Profilo per spicchi**: D = luminanza(nitida) − luminanza(prima della nitidezza);
   36 spicchi, per ogni profondità sotto il bordo (1 px, fino a `depth` × raggio, 35 %) la
   mediana di D su un arco di 70°: bande e macchie attraversano l'arco, gli anelli lo seguono.
3. **Solo le bande scure**: il profilo corretto è la chiusura morfologica di D (buchi più
   stretti di ~16 % del raggio riempiti), mai sotto zero. Lo schiarimento del bordo (che è
   nitidezza) resta; si toglie solo ciò che scurisce lungo gli archi. Smussato fra spicchi
   vicini, sfumato verso la profondità massima.
4. Dentro il disco la correzione entra in 3 px dal bordo (niente cuciture dove gli anelli
   attraversano il lembo); il clamp sui minimi/massimi locali resta per l'orlo luminoso, con
   peso che va a zero alla "fascia sul bordo". Fuori (alone, cielo) e per la regione Anelli,
   solo il clamp come prima. L'area (`weight`) copre ora anche l'interno fino alla
   profondità, e le pennellate la modificano come prima.
- Nuovo parametro "Profondità anelli interni" (avanzate). Test: banda scura su metà disco
  tolta (> 70 %) lasciando le bande del pianeta (> 80 %); ellisse ritrovata.
- Limite noto: vicino ai poli gli archi sono paralleli alle bande del pianeta, quindi una
  banda scura molto stretta lì si attenua un po'.

## 5terdecies. Nomi d'esportazione, confronto A/B, livelli per canale, dering rifatto, mosaico

Scelte dell'utente (2026-11-06):
- **Nome dell'esportazione** (`ps-app/src/naming.rs`): campo modificabile (di default il nome
  dell'originale, ↺ per tornarci) più parole chiave a scelta, in quest'ordine: oggetto (dal
  nome del file di ripresa: Moon/Sat/Jup… → Luna/Saturno/Giove, altrimenti il tipo), data e ora
  di ripresa (`aaaa-mm-gg-hhmm` all'inizio del nome, come SharpCap/FireCapture/AS!3,
  altrimenti la data del file), moduli usati (`deconv-wav-dering…`), suffisso. Una parola
  già presente nel nome non si ripete. Anteprima del nome sotto. Vale per Esporta rapido e
  come proposta in "Esporta come…".
- **Confronta (A/B)** al posto di "Istantanea": fissa il risultato attuale come A, si continua
  a modificare B. Barra di visualizzazione: A | A·B | B. Barra in basso: "Tieni A" (tornano
  le impostazioni di A), "Tieni B" (A si dimentica), "A = ora". Il pulsante in alto, mentre si
  confronta, chiude tenendo B.
- **Curve e livelli**: un solo istogramma. Nella scheda Livelli sopra i cursori, nella scheda
  Curve dietro la curva (più grande, con neri/bianchi pieni sotto); in tutti e due segue in
  tempo reale la regolazione. Selettore RGB/R/G/B comune: anche i livelli hanno nero, bianco e
  toni medi per canale (`ToneStep::channel_levels`, applicati dopo la parte comune e prima
  della curva del canale).

**Dering rifatto** dopo le prove sui tre Saturni (Normale automatico):
- Il difetto che restava era la banda scura accanto agli anelli che passano davanti al globo
  (riga centrale: originale 136→134, nitida 128→109, ~9 px) e l'orlo chiaro lungo il bordo con
  la zona più scura dietro. Il dering non agiva affatto (risultato identico pixel per pixel):
  l'area era una fascia di ~5 px lungo il bordo del globo.
- Ora l'area è **dove l'originale ha un bordo netto vicino** (`weight`): ampiezza locale
  (max−min entro la "larghezza" degli aloni) rispetto alla luminosità del soggetto, da
  `edges` (10 %) a circa il doppio. Bande e dettagli fini hanno poca ampiezza e non sono
  toccati; il bordo del pianeta, gli anelli e l'anello davanti al globo sì. Su Saturno la
  zona si limita al globo (con il suo bordo e gli anelli che lo attraversano) o agli anelli.
- Il confronto con l'originale usa una finestra piccola (un quarto della larghezza, `window()`):
  così il valore è tenuto nell'intervallo del proprio lato del bordo. Larghezza dalla
  sfocatura: 2,5σ + 2 px.
- Anelli concentrici dentro il disco: oltre a riempire i solchi del profilo radiale,
  "Orlo chiaro" (50 %) abbassa lo schiarimento lungo il bordo verso il livello più interno,
  spicchio per spicchio: è quello che fa sembrare scura la fascia dietro.
- Interfaccia: Intensità, Mostra gli aloni; "Aloni sui bordi netti" (Larghezza, Contrasto
  lasciato, Bordi trattati, Globo/Anelli/Tutto, pennello); "Anelli dentro il disco" (Orlo
  chiaro, Profondità). Tolti "Fascia sul bordo", "Cielo", "Su tutta l'immagine".

**Mosaico** (`ps-core/src/mosaic.rs`, `ps-app/src/mosaic_ui.rs`), stessa focale:
- Coppie: correlazione di fase di copie ridotte (lato ~640 px), immagini rastremate ai bordi
  e senza la media pesata (altrimenti correlano i contorni dei fotogrammi), messe in una tela
  grande quanto la somma così che anche una piccola sovrapposizione dia un picco univoco; ogni
  picco ha quattro letture (±p modulo la tela); si provano rotazioni di 0, ±1, ±2° e sia la
  correlazione di fase pura (picchi stretti) sia quella parziale (sovrapposizioni strette).
  Ogni candidato è verificato con la correlazione normalizzata della zona comune.
- Affinamento a piena risoluzione: 16 finestre nella zona comune, ognuna correlata dopo aver
  ricampionato il secondo fotogramma sul primo; rotazione + traslazione ai minimi quadrati
  (scartate le finestre in disaccordo), quattro volte. Precisione nei test < 0,3 px, 0,1 mrad.
- Verifica finale **sul solo dettaglio fine** (immagine − sfocata): le grandi forme (bordo,
  mari) correlano discretamente anche nelle posizioni sbagliate. Sulle Lune di prova: coppie
  vere 0,76–0,97, sbagliate −0,02–0,35; soglia 0,5. Un fotogramma a ingrandimento diverso
  risulta "non si sovrappone".
- Albero di copertura massima dal primo fotogramma (le coppie restano in cache: attivare o
  togliere un fotogramma non riallinea). Guadagni per canale dalle zone comuni. Fusione con
  pesi che sfumano verso i bordi (bordi irregolari di AS!3 esclusi) e, se scelto, che
  premiano il fotogramma più nitido (peso ∝ nitidezza⁴).
- Finestra Mosaico: elenco con stato (riferimento / allineata / non si sovrappone / esclusa),
  anteprima con i riquadri numerati (trascina per ritoccare), sfumatura, uniforma luminosità,
  preferisci la più nitida, Aggiungi, Riallinea, Salva mosaico (`.psmosaic`, JSON con file,
  posizioni e guadagni), Crea/Aggiorna. Si apre trascinando più immagini, da File › Nuovo
  mosaico, o aprendo un `.psmosaic`. Il mosaico creato è il documento; File › Modifica
  mosaico lo riapre e l'aggiornamento conserva le regolazioni (non distruttivo).
- Tre Lune 3888×2200 → 6114×2402: allineamento 1,7 s, unione 1,8 s.

Sviluppo: su questo Mac si eseguono solo i test dei moduli toccati (debug, con filtro), mai
il giro completo, neanche prima dei pacchetti. Codice formattato con `max_width=200`,
`use_small_heuristics=Max` (non usare `cargo fmt` senza queste opzioni).

## 5quaterdecies. Spettro dell'immagine e wavelet a scale libere

**Spettro** (`ps-core/src/spectrum.rs`, in `Analysis::spectral`): potenza media per anello di
frequenza di un riquadro fino a 1536 px (finestra di Hann, 48 classi logaritmiche da 0,02 a
0,5 cicli/px), adattata con `P(f) = A·f^−α·e^(−4π²σ²f²) + N` (griglia su σ e α, A e N ai
minimi quadrati sugli errori relativi). Ne viene la **frequenza di taglio**, dove il segnale
scende al pavimento del rumore: il dettaglio più fine che l'immagine contiene davvero. Il
rapporto segnale/rumore per frequenza non cambia con i filtri lineari, quindi vale anche dopo
la deconvoluzione. Sulle immagini di prova: Saturni 0,19 cicli/px (periodo 5,3 px), Lune
0,10–0,27. La σ adattata **non** si usa come PSF: sulle prove con verità nota sovrastima del
25–45 % (il soggetto non segue una legge di potenza pura); anche i guadagni wavelet "ottimi"
calcolati dal modello peggioravano l'immagine e sono stati scartati.

**Scale libere** (`ScaleMode::Free`, predefinite): livello j = c_j − c_{j+1} con
c_k = immagine ⊛ G(σ_k), σ_0 = primo/2 e σ che raddoppia (passate à trous gaussiane, taps
spaziati per tenere il kernel corto; sotto 0,75 tap il kernel a tre tap con la stessa
varianza). Il primo livello (`first_scale`, sullo stesso piano dei 1·2·4 px classici) si
sceglie come `0,3 / taglio`: il livello ha il picco a 0,3/scala cicli/px, cioè sul taglio. Ciò
che è più fine di primo/2 è la **grana**: mai resa più nitida, solo ripulita
(`grain_denoise`). Il rumore del primo livello si misura anche direttamente (dopo la
deconvoluzione il rumore non è bianco e la stima scalata dalla grana verrebbe bassa).

**Intensità automatiche**: le tabelle per tipo restano quelle delle scale classiche; per le
libere si calcolano i guadagni che danno **la stessa nitidezza del dettaglio vero**
(`wavelet::equivalent_gains`: minimi quadrati sulla risposta in frequenza sotto il taglio,
tenendo conto di quanto la nitidezza adattiva toglie a ogni livello secondo il suo rapporto
segnale/rumore, più una leggera regolarità tra livelli vicini). Le soglie di rumore scalano
di un posto (quella del livello classico più fine va alla grana). `preset::convert_scales`
fa la stessa conversione quando si cambia scala nel pannello, nei due sensi.

**Prove** (`cargo run --release -p ps-io --example wavelet_bench -- Immagini_test/*.tif`):
immagini reali rimpicciolite 3× come verità, poi sfocate (σ 1,4 e 2,2) con rumore (0,0015 e
0,004), più una scena di crateri senza sfocatura. Errore RMS rispetto all'ingresso:
- guadagni ottimi per ciascuna scala (le scale a confronto, non le tabelle): libere meglio
  con poco rumore (Saturno 0,26–0,28 contro 0,31–0,34; Luna 0,41–0,47 contro 0,52–0,53),
  pari con molto rumore;
- wavelet automatiche da sole: su Saturno libere meglio in tutti gli 8 casi, sia alla loro
  quantità (0,82–0,88 contro 0,88–0,99) sia alla quantità migliore di ciascuna (0,80–0,83
  contro 0,81–0,98); su Luna e crateri alla pari (±3 %);
- pipeline automatica completa: Luna e crateri meglio in tutti i 16 casi (es. 2,38 contro
  2,60; 0,79 contro 0,86; 1,37 contro 1,72), Saturno peggio in tutti gli 8 (es. 1,57 contro
  1,24; 0,75 contro 0,73): dopo la deconvoluzione le libere aggiungono più nitidezza e la
  verità, a sua volta un po' sfocata, penalizza ogni nitidezza oltre sé stessa. A vista sui
  Saturni reali le due versioni sono quasi uguali, senza artefatti in più.

Interfaccia: nella scheda Wavelet "Scale: Libere / Classiche 1·2·4·8" (la scelta è
ricordata per le immagini successive), riga **Grana** con il suo cursore di rumore, ogni
livello con la sua dimensione in px; nelle avanzate "Primo livello" (1–4 px) per le libere,
kernel/progressione/passo per le classiche. La riduzione rumore planetaria usa sempre le
scale classiche (era cambiata per sbaglio con il nuovo predefinito).

## 5quindecies. Strumentazione, log di ripresa, effemeridi, PSF dallo spettro

**Catalogo** (`ps-core/src/equipment.rs`): 36 telescopi (Celestron C5–C14 ed EdgeHD, Meade
LX200, Sky-Watcher Skymax, Newton/Dobson, Ritchey-Chrétien, rifrattori ED/apo, Takahashi)
con apertura, focale e ostruzione; 51 camere (ZWO ASI, SVBONY, Player One, QHY, ToupTek,
The Imaging Source, Philips) con pixel e colore/mono; Barlow 1,5–5×, Powermate, riduttore
0,63×; filtri con la lunghezza d'onda efficace (IR passa-alto: quella media sul sensore).
Ogni voce ha anche "Personalizzato" con i propri numeri. `Setup::optics()` dà scala
(206,265·p/F), rapporto focale, frequenza di taglio della diffrazione p/(λN) cicli/px,
sovracampionamento 0,5/taglio, PSF di Airy con ostruzione.

**Log e nomi dei file** (`ps-core/src/capture.rs`): FireCapture (`Camera=`, `Filter=`,
`Profile=`, `FocalLength=`, `Resolution=`, `Diameter=`, `Date=` con `Date_format`,
`Mid(UT)=`, `Binning=`), SharpCap `.CameraSettings.txt` (camera tra parentesi quadre),
AutoStakkert (drizzle nel log o nel nome, `Driz150`); chiavi `chiave=valore` o `chiave:
valore` con più grafie. Nomi: FireCapture `aaaa-mm-gg-hhmm_t-OSS-FILTRO-Oggetto` e i vecchi
`IR0015 aa-mm-gg hh-mm-ss`. All'apertura si legge da solo il log nella stessa cartella il cui
nome è l'inizio di quello dell'immagine; altrimenti "Carica log…" o trascinarlo sulla finestra.
**Nota**: i formati sono ricostruiti dalla documentazione nota, senza log veri di prova.

**Effemeridi** (`ps-core/src/ephemeris.rs`): elementi kepleriani JPL 1800–2050 per i pianeti,
serie ELP ridotta (Meeus cap. 47) per la Luna; diametro apparente entro ~1 % (test: Giove
2022 49,9″, Marte 2018 24,3″, Sole al perielio). `analysis::image_scale` divide per il
diametro in pixel (globo di Saturno, disco degli altri): sui tre Saturni di settembre 2026
0,115″/px (≈ 5200 mm con pixel 2,9 µm), su quello del 1° settembre 0,296″/px (≈ 2020 mm).

**Effetti sull'elaborazione** (`Analysis::optics`): la frequenza del dettaglio usata per le
scale wavelet libere non supera mai il limite di diffrazione; senza spettro se ne usa l'80 %.

**Spettro con rumore colorato**: il pavimento ora è `N·e^(−4π²τ²f²)` (τ 0–0,8 px): lo stacking
con ricampionamento leviga il rumore e il pavimento scende verso le alte frequenze. Ora lo
spettro si misura anche sui pianeti piccoli 640×480 (prima scartati).

**PSF della deconvoluzione dallo spettro** (prove sulle immagini reali, `cargo run --release
-p ps-io --example real_tests -- cartella immagini…`): la sfocatura misurata sul bordo dei
pianeti (σ 4–8 px) è molto più larga di quella del dettaglio (lo spettro dà σ 1,5–2,5): la
allargano l'oscuramento al bordo e le ali del seeing. Con quella PSF la deconvoluzione dava su
Giove una tessitura "a macchie", su Saturno contorni da disegno, su Marte un anello scuro.
Una gaussiana piccola dallo spettro, invece, non fa quasi nulla (l'arresto automatico si ferma
dopo 1–4 iterazioni perché il modello spiega già i dati). Ora la PSF è un **Moffat** con nucleo
FWHM = 2,355 · 0,34 / taglio (la σ il cui MTF scende al 10 % al taglio) e β del bordo se
misurato, altrimenti 2 (limitato a 1,5–3). Risultati più naturali su Giove, Saturno, Marte,
Luna; il Saturno R0012 (anelli di taglio, prima quasi invariato) mostra ombra e anello. Lo
stesso nucleo regola larghezza del dering e PSF di BlurXTerminator.

**Scheda Strumento** (`ps-app/src/instrument.rs`): oggetto e data (dal nome o dal log,
modificabile), telescopio/camera/Barlow/filtro/binning/drizzle (ricordati per le immagini
successive), Carica log…, diagnosi (scala dallo strumento e misurata, accordo tra le due,
limite di diffrazione, campionamento con giudizio, disco di Airy, risoluzione raggiunta =
taglio dello spettro / taglio ottico, sfocatura sul bordo, segnale/rumore) e grafico dello
spettro con segnale, rumore, taglio, limite di diffrazione e posizione dei livelli wavelet.

Osservazioni ancora aperte dalle prove: due Saturni con anelli quasi di taglio (R0012,
G0013) vengono riconosciuti come "Pianeta"; sulle macchie solari la riduzione rumore "lunare"
rende la penombra a chiazze; il Marte più piccolo (2B0008, 59 px) non ha spettro misurabile.

## 5sexdecies. Deconvoluzione rivista

- **Perché "Sfocatura" non faceva niente**: all'apertura la deconvoluzione partiva con 0
  iterazioni (regola "moduli accesi ma neutri"), quindi la PSF non contava finché non si
  alzava "Al massimo". Ora parte **spenta ma già regolata** (PSF e iterazioni automatiche):
  accendendola lavora subito.
- **Scheda**: un solo cursore, "Sfocatura da togliere" (FWHM della PSF, stessa misura per
  tutte le forme; ↺ torna al valore automatico), la PSF disegnata accanto con una frase sulla
  sua forma, i pulsanti Confronta, Misura su stella o luna e "Valori automatici" (quando
  qualcosa è cambiato). Lo stato dice algoritmo, forma e dopo quante iterazioni si è fermata.
- **Avanzate**: algoritmo con spiegazione (Richardson–Lucy consigliato per pianeti, Luna e
  Sole; Wiener istantaneo, per prove o immagini molto pulite), forma della PSF con
  spiegazione, β, Airy "Dallo strumento", "Fermati da sola" e il massimo delle iterazioni
  (con il valore automatico come predefinito), anti-rumore, anti-ringing, Biggs–Andrews, TV,
  colore. I valori predefiniti dei cursori sono quelli automatici di questa immagine
  (`preset::auto_deconv`).
- **Van Cittert tolto** (le impostazioni salvate che lo usavano si aprono con R–L).
- **Confronta**: finestra 2×3 sulla zona visibile (al massimo 384 px di lato), calcolata in
  background: Gaussiana / Moffat / Airy con R–L e la stessa sfocatura; originale, R–L e
  Wiener con la forma attuale. La scelta in uso è evidenziata, "Usa questa" la applica.
- **Misura su stella o luna**: si disegna un riquadro (un clic = riquadro di 48 px).
  `psf::measure_in` prova i massimi locali dal più luminoso, un tentativo per oggetto,
  scartando zone piatte, bordi e oggetti più larghi di 16 px; la nota dice dove l'ha trovato,
  quanto è largo e quanto è rotondo (o perché non ha trovato niente). "Torna alla PSF
  automatica" annulla. Su 2R0009 trova la luna di Giove (9,8 px: disco della luna più
  sfocatura), non il pianeta.

## 5septdecies. Wavelet corrette e "Nitidezza intelligente"

**Diagnosi** (`cargo run --release -p ps-io --example wavelet_diag -- cartella img.tif [bxt.tif]`):
guadagno per dimensione del dettaglio (spettro dopo / spettro prima), misurato dentro il disco
(l'anello scuro che il rinforzo crea sul bordo, tagliato a zero contro il cielo, si leggerebbe
come dettaglio fine). Su 2R0009 BlurXTerminator concentra la forza appena sopra il limite del
dettaglio (×10–12 a 5–6 px) e ripulisce oltre (×0,8); le wavelet automatiche davano ×1,5 al
massimo, sulle scale grandi; lo stile RegiStax alzava tutto, rumore compreso.

**Bug corretti**: (1) il rumore del primo livello libero era misurato sul livello stesso, che
contiene il dettaglio vero: stima 20× troppo alta, la nitidezza adattiva gli lasciava il 14 %
della forza (ora 79 %); (2) la protezione dei bordi agiva su tutti i bordi forti, comprese le
bande di Giove: ora solo dove il bordo confina con il cielo (minimo locale scuro).

**Nitidezza intelligente** (`ps-core/src/smart.rs`, `Stage::Smart`, scheda a sé nell'area
Nitidezza dopo le Wavelet). Rifatta il 2026-11-07 dopo lo studio `ps-io/examples/cutoff_study.rs`
(16 immagini di test, BlurXTerminator come riferimento):

- *Il limite del dettaglio misurato oggi va bene*: `preset::detail_cutoff` cade entro il 10 % dal
  picco di BXT su 7 immagini su 9. Le alternative provate sono peggiori: spettro solo dentro il
  disco (su Saturno il quadrato utile è ~90 px, instabile), plateau / analisi di decorrelazione
  (Descloux 2019) / FRC a scacchiera (il rumore degli stack è fortemente correlato: sul cielo la
  potenza a 0,1 c/px è 2–9× quella a 0,45), MTF dal bordo (≈0,047 c/px ovunque: il bordo è
  morbido di suo).
- *Il difetto era la risposta reale*: i livelli a mezza ottava si sovrappongono e, dando a ognuno
  il valore della curva al proprio picco, al 100 % le strutture 2–4× più grandi del limite
  venivano rinforzate ×3–9 (BXT ×1,5–3,6), il picco cadeva a 0,85× il limite e la pulizia partiva
  da 1,2× (BXT rinforza ancora ×2–5 a 1,5×). Da qui "Dove spingere" al 130 % e l'aspetto impastato.
- *Ora*: curva obiettivo = forma mediana di BXT normalizzata sul limite (`SHAPE`, in log-guadagno:
  ×1,7 a 0,25×, picco al limite, ×3 circa a 1,5×, 1 a 2×), picco = 1 + 11 × nitidezza (100 % =
  ×12, come BXT; il cursore arriva al 160 %); pulizia della grana solo oltre 1,9× il limite,
  0,6 × nitidezza. Guadagni dei livelli adattati in modo moltiplicativo (60 passi, sempre ≥ 0)
  finché la somma dei livelli raggiunge la curva ai picchi; il primo livello a 1,8× il picco. Ciò
  che resta di errore è corretto esattamente in frequenza su tutto lo spettro (FFT con bordi
  specchiati; ×0,5–×2 fino a 1,9× il limite, solo riduzione oltre). Tolto il secondo cursore
  ("Pulizia"). Misurato dopo: al 100 % la risposta segue quella di BXT a ogni scala (scarti
  ≤ 30 %, un po' più generosa oltre 1,5× il limite).
- Rinforzo adattivo ("Segui le zone": per livello e pixel energia locale contro rumore atteso,
  S/(S+N)) e protezione del bordo dentro il disco quasi non cambiano la risposta.
- Limite aperto: a parità di risposta in frequenza BXT è più pulito sui pianeti (rinforza il
  segnale e non il rumore vicino al limite, dove S/N ≈ 1); sulla Luna al 100 % siamo già più
  nitidi di BXT.

Ritocchi del 2026-11-08: cursore fino al 180 %; "Dove spingere" (60–140 %) tra i cursori
principali; **curve** selezionabili (`SmartCurve`): Standard (quella misurata su BXT,
predefinita) e quattro varianti *progettate, non misurate*: Seeing scarso (picco a 0,7× il
limite, più spinta sulle strutture grandi, pulizia da 1,5×), Seeing ottimo (picco a 1,15×, fino
a 2,3×), Luna e superfici (spinta larga su tutte le dimensioni), Globi lisci (più dolce sul
dettaglio finissimo). Protezione del bordo predefinita abbassata (Pianeta 0,25, Saturno 0,3,
Disco intero 0,2; prima 0,5–0,6): dentro il disco non cambia nulla. In Avanzate, **Mostra dove
frena** (`smart::hold_back_map`, overlay `Overlay::SmartZones`, calcolato sull'immagine
originale): in blu dove "Segui le zone" riduce la nitidezza, in arancione dove la riduce la
protezione del bordo, intensità = quota di spinta tolta. Osservazione: subito fuori dal bordo
c'è un anello dove la nitidezza agisce piena (il bordo stesso conta come dettaglio), proprio dove
si formano gli aloni.
Valori per tipo (`preset::smart_structure`, `auto_smart`). Anteprima: `ps-io/examples/zones_preview.rs`.

**Mappa della risoluzione** (`smart::resolution_map`): per ogni punto il periodo del livello
più fine con più dettaglio che rumore; nella scheda come miniatura a colori con legenda.

Risultati dentro il disco di 2R0009 (8 / 6 / 5 / 4 / 2,2 px): intelligente 60 % ×6,7 / 6,6 /
5,6 / 3,4 / 0,67; deconv + intelligente 35 % ×2,8 / 6,2 / 7,1 / 6,3 / 1,4; BlurXTerminator
×7,6 / 12,5 / 10,7 / 4,4 / 0,81.

## 5octodecies. Anello scuro dentro il disco (collimazione, diffrazione)

Il vecchio "anelli concentrici" (mediana su archi di 70° a profondità fissa, solo ciò che la
nitidezza aggiungeva sotto il riferimento) non vedeva l'anello che si nota su Marte, Giove e
Saturno: con una collimazione imperfetta la sfocatura del bordo cambia da lato a lato, e
l'anello che la nitidezza ne ricava è una **copia del bordo spostata** (vicina al bordo da un
lato, più dentro dall'altro) e spesso c'è solo su metà disco. Sullo stack 2R0009 l'anello non
c'è (profilo radiale liscio): lo disegna la nitidezza.

`dering::limb_rings` / `ring_change`:
- La fascia sotto il bordo (fino a `depth` del raggio, letta un po' più a fondo) è srotolata
  in angolo (720 passi) × profondità; si usa il cambiamento della nitidezza `sharp − reference`.
- Bande e macchie c'erano già: il loro dettaglio (passa-banda sottile − largo, σ ≈ 0,6 e
  2,2 × la sfocatura) moltiplicato per il guadagno della nitidezza (minimi quadrati sul
  passa-banda) viene tolto. Resta ciò che la nitidezza ha disegnato senza un modello
  nell'originale.
- L'anello è la curva profondità(θ) = a + b cos θ + c sin θ (bordo scalato e spostato) che
  raccoglie più "solco" tutto intorno (ricerca a griglia grossa ogni 2°, poi fine); poi segue
  i solchi vicini (± sfocatura, smussato su 10°).
- Lungo la curva si adatta, per ogni angolo, retta + gaussiana scura (larghezza scelta fra
  cinque per massima varianza spiegata); la profondità non supera mai il riempimento della
  valle. Intensità = livello tenuto su due terzi di una finestra di ±30° (una banda dritta che
  costeggia il bordo curvo per pochi gradi non passa), poi smussata su ~8°.
- Si accetta un anello solo se copre almeno un quarto del bordo; al massimo due.
- Correzione: la gaussiana ricostruita, sottratta con l'Intensità del modulo, sfumata nei
  primi 3 px sotto il bordo. "Mostra gli aloni" la aggiunge in magenta.
- La pipeline calcola gli anelli una volta (cache `rings`) per il dering e per la mappa.
- Tolto "Orlo chiaro": l'orlo luminoso lo tiene già la parte sui bordi netti.
- Verificato: test sintetici (anello spostato su metà disco con bande inclinate: tolto per
  oltre il 60 %, bande ≥ 90 %; senza anello nessuna correzione) e la Giove dell'utente
  allineata a 2R0009 (anello a 5–13 px su metà disco, trovato e schiarito).

## 5novodecies. Dering sui pianeti: solo l'anello sotto il bordo

Prova dell'utente nell'app (Giove 2R0009, wavelet + nitidezza intelligente): "Mostra gli
aloni" colorava le bande, perché la parte "aloni sui bordi netti" tratta come bordo ogni salto
del 10 % entro la larghezza, e i bordi delle bande lo superano; la ricerca dell'anello
(5octodecies) non scattava, perché togliendo il dettaglio dell'originale toglieva anche la
curva del bordo e quindi l'anello. Con Larghezza 3 e Contrasto lasciato 0 l'utente vedeva
l'anello giusto e ha chiesto di fermarsi a quello.

Su un pianeta intero (regione "Tutto" con un disco trovato) ora:
- l'area è una fascia **sotto il bordo** profonda `radius + 2` px (più il cielo accanto); più
  dentro nulla, bande comprese;
- **cielo**: solo ciò che scende sotto il livello del cielo vero (mediana oltre 1,4 raggi meno
  3 MAD) viene alzato: tenere il cielo nell'intervallo dell'originale rimetteva il bagliore
  sfocato attorno al pianeta;
- **bordo vero e proprio** (profondità < 2σ della sfocatura): non toccato, né alzato né
  abbassato (abbassare l'orlo chiaro lo riammorbidiva, alzare la discesa rimetteva il bagliore);
- **fascia interna**: si schiarisce solo ciò che è sotto **lo stesso pixel** dell'originale più
  la variazione media che la nitidezza ha dato al disco subito più dentro (`inner_shift`);
  confrontarlo col minimo dei vicini lasciava passare l'anello, perché sulla pendenza del bordo
  il vicino più scuro sta più fuori.
- La mappa "Mostra gli aloni" (`ringing_map`, ora con il disco) segue le stesse regole.
Saturno (regione Globo/Anelli) resta com'era. Sulla Giove dell'utente allineata a 2R0009 la
mappa segna solo l'anello lungo il bordo; Marte 2018 e Giove 2016 quasi nulla.

## 5vicies. Dering a pennello

Le ricerche automatiche dell'anello (5octodecies, 5novodecies) non lo trovavano nemmeno quando
era ben visibile, e la parte "aloni sui bordi netti" scambiava le bande per anelli. Ora è
l'utente a dire dov'è: lo **dipinge con il pennello**, largo più o meno quanto l'anello; sotto
la pennellata `dering::find_ring` lo trova, lo misura e lo schiarisce. Tolti la ricerca
automatica (`limb_rings`, `ringing_map`, Profondità, Contrasto lasciato, Globo/Anelli) e il
disco passato alla pipeline (`Aux::globe`). `dering::weight` resta, automatico, solo come area
dell'anti-ringing della deconvoluzione (`DERING_MASK`); le pennellate sono un'area a parte
(`RING_MASK`, `dering::painted`).

Sotto la pennellata, per ogni pixel, lungo la direzione **attraverso l'anello**:
- **direzione**: quella del gradiente dell'immagine (tensore di struttura largo quanto la
  pennellata: l'anello segue il bordo netto che lo crea), pesata per quanto il bordo è netto
  rispetto al tipico; dove l'immagine è piatta conta la direzione della pennellata. Le bande che
  attraversano la pennellata non sono valli in questa direzione.
- **copia smussata solo lungo l'anello** (σ 1,5 px): il rumore non viene preso per valle e
  l'anello conserva la sua profondità.
- **due stime del fondo** sotto la valle: la *chiusura* (max poi min su ±r) riempie fino al
  lato più basso, sempre sicura (un gradino come il lembo non si alza); una *parabola* sui
  dintorni a r…2r px dai due lati segue pendenza e curvatura (oscuramento al bordo, la coda
  sfumata del lembo, che così non viene scambiata per valle) ed è accettata solo se i due lati si
  continuano (stessa pendenza, o entrambi piatti) e la valle sta ben sotto la parabola rispetto
  a quanto i dintorni se ne scostano. Con un gradino nascosto sotto la valle (anello nel cielo
  accanto al lembo) i lati non si continuano e resta la chiusura: il pianeta non si "ingrassa".
  Ogni pixel prende la parabola del punto vicino meglio centrato sulla valle, valutata alla
  propria posizione, mai sopra il più chiaro dei dintorni.
- **larghezza** r: la più piccola che dà il 95 % del riempimento massimo (95° percentile sotto
  la pennellata), al massimo metà della larghezza dipinta (o "Larghezza anello" in Avanzate).
- **dov'è**: il riempimento deve esserci lungo gran parte dell'anello (mediana lungo la
  pennellata su ±max(2r, 8) px: un'estremità di banda che attraversa resta) e uscire dal rumore
  (MAD di ciò che lo smussamento toglie). La correzione si estende fino a r px oltre la
  pennellata attraverso l'anello: la pennellata è un'indicazione, non un ritaglio.
- Ogni canale è riempito per conto suo; il risultato è smussato leggermente lungo l'anello.

Verifiche (`ring_probe`, ora con pennellata simulata `ARC`/`STROKE`, anello sintetico `SYNTH`):
anello decentrato aggiunto alla Giove 2R0009 tolto per circa tre quarti, bande intatte; anello
nero nel cielo riempito fino al livello del cielo senza toccare il lembo; lembo senza anello
(nitidezza Normale) intatto; su Giove con nitidezza esagerata e senza anello solo piccole
macchie dove le bande toccano la pennellata; cratere lunare: l'ombra larga resta.
Limiti: una valle vera parallela alla pennellata e più stretta dell'anello (una banda sottile
lungo il bordo, il solco attorno al picco centrale di un cratere) viene schiarita anch'essa.

## 5unvicies. Scala, Identifica, Colorazione solare (2026-11-09)

**Scala (scheda "Scala", `ps-core/src/resample.rs`, `ps-app/src/resize_ui.rs`).** Ingrandimento o riduzione
dell'immagine esportata, senza IA: kernel separabili classici (vicino più prossimo, bilineare, B-spline,
Mitchell-Netravali B=C=⅓, Catmull-Rom, Lanczos 2/3/4), centri dei pixel allineati, bordo che ripete
l'ultimo pixel; in riduzione il kernel si allarga di 1/scala (niente aliasing). "Anti-alone": ogni pixel
d'uscita è tenuto fra minimo e massimo dei 2×2 pixel d'ingresso intorno (miscelato con l'intensità), toglie
l'anello di Lanczos sul lembo contro il cielo nero. Metodo automatico per tipo (`method_for`): pianeti e
Saturno Lanczos 3 + anti-alone 1; superfici Lanczos 3 + 0,5; dischi interi (Luna, Sole) Mitchell.
La dimensione è un solo fattore (proporzioni bloccate) mostrato in %, px o cm a N dpi; massimo ×16 e
24000 px per lato. Si applica solo all'esportazione (pennellate e maschere restano alla misura originale);
anteprima a due riquadri (pixel originali / metodo) su un dettaglio al centro del soggetto. Il nome
esportato prende "ingrandita"/"ridotta". Il DPI non viene ancora scritto nei metadati TIFF/PNG.

**Identifica (pulsante in alto, `ps-core/src/identify.rs`, `moons.rs`, `selene.rs`, `ps-app/src/identify_ui.rs`).**
Dalla data UT (nome del file o log; correggibile) e dall'oggetto:
- Giove: lune galileiane col metodo a bassa precisione di Meeus cap. 44, meridiano centrale I e II (cap. 43).
  Saturno: 8 lune, Meeus cap. 46 (port di `soniakeys/meeus`, MIT), posizione di Saturno dagli elementi
  kepleriani di `ephemeris.rs` riportati all'eclittica B1950; inclinazione degli anelli (cap. 45).
  Test: coincidono con gli esempi 44.a e 46.a (attenzione: JD 2451439,5 = 18 settembre 1999).
- Nell'immagine: regione luminosa più grande (soglia 20 % fra fondo e 50° pixel più luminoso), assi dai
  momenti, raggio equatoriale = semiasse maggiore (Saturno con anelli aperti: /2,27). Puntini fuori dal
  pianeta = massimi locali della media 3×3 sopra 6σ. Orientamento = rotazione (±10° attorno all'asse
  trovato, o 360° se il disco è tondo) e specchio che fanno cadere più lune previste su puntini.
  Senza lune: asse dalla forma, lato/specchio da verificare (nota nella finestra).
- Luna (`selene.rs`): posizione con i termini maggiori di Meeus 47, librazione ottica e punto subsolare
  (cap. 53; test sull'esempio 53.a entro 0,3°), fase, colongitudine. Catalogo di ~100 crateri, 22 mari e
  qualche monte/valle con coordinate approssimate (~1°). Il lembo si trova con RANSAC su bordi forti in cui
  la luce cala verso l'esterno (esclude il terminatore), poi Kåsa; orientamento = rotazione e specchio
  con la massima correlazione fra immagine e un modello (illuminato sì/no × mari scuri 0,55). Se nel campo
  c'è meno di metà disco è dichiarato "stimato". Nessun lembo: elenco dei crateri sul terminatore.
- Sole: P, B₀, L₀, rotazione di Carrington (Meeus 29).
- Convenzione `Orientation`: angolo dell'asse +x dell'oggetto (ovest per i pianeti, est selenografico per
  la Luna) sull'immagine, in senso orario; 0°/non specchiata = nord in alto, est a sinistra.
- Interfaccia: finestra con oggetto, data/ora UT, mostra nomi/crateri, orientamento automatico o a mano
  (rotazione, 180°, specchiata), effemeridi e "come è stato trovato". Calcolo su un thread; nomi disegnati
  sopra l'immagine (lune verdi trovate / arancio tratteggiate previste; crateri gialli sul terminatore).
- Diagnosi: `ps-io --example identify -- out.png img [Corpo] [aaaa-mm-gg-hhmm]` disegna le etichette.
- Limiti: le immagini di prova lunari sono tutte primi piani con un pezzo di lembo; l'orientamento lì è
  plausibile ma non verificato sul cielo reale. Gli orari dei nomi vecchi (`yy-mm-dd hh-mm-ss`) potrebbero
  essere ora locale.

**Colorazione solare (modulo `Stage::SunColor`, scheda Colore, `ps-core/src/suncolor.rs`).** Mappa di
gradiente con un esponente per canale: out_c = L^γ_c sulla luminanza (nero e bianco restano tali). 8
tavolozze (bianco neutro/caldo, giallo pallido, giallo sole, oro, arancio e rosso H-alfa, viola Calcio K),
intensità = miscela con l'immagine. Ultimo passo della pipeline (dopo Mineral Moon), immagine sempre RGB.
Visibile per dischi interi e superfici, acceso ma neutro (intensità 0); Auto = Giallo sole pieno.

### Identifica rivista (2026-11-09, sera)

- **Scheda, non finestra.** "Identifica" è una scheda del pannello laterale (`Group::Identify`, tra le aree e
  Esporta); il pulsante in alto la seleziona. Calcolo e nomi sull'immagine sono attivi solo mentre la scheda è
  quella visibile. All'apertura di un file con oggetto e ora noti (nome, log o pixel) la scheda si apre da sola.
- **Fuso orario automatico.** "Ora locale" chiede al sistema l'offset *della data della ripresa* (`date -r`
  su macOS, `date -d @` su Linux, `TimeZoneInfo.Local` via PowerShell su Windows), quindi ora legale o solare
  giusta anche per riprese di mesi fa; ora legale = offset > min(offset di gennaio, di luglio). "Modifica"
  passa ai valori a mano, "Automatico" torna indietro. Si usa il fuso del computer, non quello del luogo.
- **Regolazione sulle lune (`identify::fit_time`).** Cerca insieme tempo (±180 min, passo 1 min), rotazione
  (±6°) e stretching delle distanze (±8 %), con il costo sommato su *tutte* le lune in campo (ogni luna
  vale `1-(d/tol)²` se cade vicino a un punto luminoso), poi rifinisce a coordinate con tolleranza stretta
  (0,4×) e penalità piccola su variazioni di tempo/rotazione/scala. Con una sola luna in campo si muove solo
  il tempo. `Orientation.scale` porta lo stretching (1 = raggio misurato). Parte da sola alla prima
  identificazione di un'immagine di Giove o Saturno e applica il risultato se almeno 2 lune cadono su un punto.
- **Raddrizza: nord in alto.** `Orientation::upright()` dà (specchio, gradi); `Geometry::then(mirror, gradi)`
  compone lo specchio e la rotazione sull'orientamento esistente (gruppo diedrale 2×2 per quarti di giro e
  specchi, solo il resto ≤45° va nell'angolo fine, con segno invertito se c'è uno specchio prima). Il ritaglio
  si azzera come per ogni cambio di orientamento.
- **Nomi stampati (`ps-app/src/annotate.rs`).** Livello disegnato alla risoluzione dell'immagine esportata,
  dopo l'eventuale ingrandimento: testo dai contorni di Ubuntu Light (stesso font dell'interfaccia) con
  `skrifa` e un rasterizzatore a copertura esatta (accumulo di aree con segno), alone scuro, cerchi, bussola
  N/S, didascalia (oggetto, filtro, data/ora UT o locale, scala, angolo del nord e rotazione). "Esporta con
  i nomi" salva `<nome>_nomi.<formato>` accanto all'originale; l'esportazione normale resta pulita. Opzioni
  salvate nelle preferenze (`Preferences.identify`).
- **Posizione con due crateri (2026-11-10, `identify::place_on_craters`).** Per i primi piani lunari senza bordo:
  nella scheda Identifica (sezione «Posiziona con due crateri», solo Luna) si sceglie un cratere dalla lista (quelli
  visibili e illuminati all'ora data), lo si clicca sull'immagine, poi un secondo. Due punti fissano similitudine
  (scala, rotazione, centro del disco) nel piano ortografico del disco, quindi la proiezione vicino al bordo resta
  giusta; lo specchio non si può ricavare da due punti, lo dà il pulsante «Specchiata». Il risultato è un
  `Placement {cx, cy, r, orientation}` che `identify_with` usa al posto della ricerca del bordo e
  dell'orientamento; con la collocazione attiva l'orientamento manuale è bloccato e «Raddrizza» resta. Il clic usa
  `Tool::Pick` del visore mentre `IdentifyState::awaiting_click()`. Test: `two_craters_place_the_disk`.
  Precisione limitata dal catalogo (~30 km): conviene scegliere due crateri lontani.
- **Rilevamento automatico dei crateri (provato e scartato, 2026-11-10).** Hough circolare sul gradiente e filtro ad
  anello (armoniche cos/sin, FFT, piramide a 4 livelli, ~200 ms) sulle riprese lunari reali: punteggi dominati da
  bordo della Luna e cornice, crateri grandi e morbidi mancati, ellissi vicino al bordo. Codice rimosso.
- **Crateri senza effemeridi (non fatto).** Nei primi piani lunari le effemeridi non dicono *dove* siamo sul
  disco. Metodi possibili: (1) riconoscere i crateri (Hough circolare o rilevamento di cerchi con ombra sul
  terminatore) e risolvere la posizione come con una "plate solving": confrontare la costellazione di
  crateri trovati (rapporti di diametri e distanze) con quella del catalogo proiettata con librazione e
  illuminazione note (geometric hashing); (2) registrare l'immagine su una mappa di riferimento (LRO/Kaguya)
  resa con la stessa illuminazione; (3) una piccola rete neurale che stima la zona. Il (1) è leggero e senza
  dati pesanti, ma serve un catalogo molto più fitto (migliaia di crateri) di quello attuale (~100).

## 5duovicies. Lingua dell'interfaccia: italiano e inglese (2026-11-10)

L'interfaccia esiste in italiano e in inglese; si cambia da Impostazioni › LINGUA · LANGUAGE e vale subito, senza riavviare. La scelta è in `Preferences::language`; alla prima apertura segue la lingua del sistema (macOS: `defaults read -g AppleLanguages`; altrove `LANG`), con l'italiano come ripiego. Nei test (`cfg!(test)`) il default resta l'italiano.

**Come funziona** (`ps-core/src/i18n.rs`): i testi nel codice restano in italiano, la lingua originale.
- `tl("testo italiano")` restituisce l'inglese cercandolo in `i18n_en.rs` (tabella IT → EN, ~820 voci); in italiano, o se manca la voce, restituisce il testo com'è. Costa una ricerca in una HashMap, fatta a ogni fotogramma.
- `tr!("italiano {}", "english {}", args…)` è il `format!` bilingue: le due versioni stanno una accanto all'altra nel punto d'uso, quindi il compilatore controlla segnaposto e argomenti di entrambe. Per i plurali si passa il suffisso come argomento (in inglese `""`).
- `tl_dyn(&str)` per nomi non letterali salvati in italiano (il catalogo strumenti: «Nessuno / IR-cut (luminanza)», «Barlow 2,5×»…).
- `with_lang(lang, || …)` imposta la lingua solo per il thread corrente: i test la usano senza disturbare quelli in parallelo.

**Chiavi che NON si traducono**: `Group::name()` è la chiave salvata in `current_area` (resta italiana); per l'etichetta della scheda c'è `Group::label()`. Gli stessi nomi dei livelli creati restano nella lingua in cui sono nati. Le costanti (`const`) non possono chiamare `tl`: contengono il testo italiano e lo si traduce dove lo si legge (`PLANETS`, `LAYER_SECTIONS`, `SATURN_MOONS`).

**Che cosa è stato reso inglese con attenzione**: «bruciati» → *clipped*, «seeing» e «lembo» → *limb*, «anelli» → *rings*, «mosaico» → *mosaic*, «fuso/ora legale» → *time zone/daylight saving*, «effemeridi» → *ephemerides*, «ritaglio» → *crop*, «Dering» e i nomi dei moduli restano termini tecnici. I nomi dei file esportati seguono la lingua (`Moon`, `Saturn`, `sharpen`, `levels`…). I nomi latini della Luna (Mare Tranquillitatis…) non cambiano.

**Controlli**: `english_interface_has_no_italian_left` apre ogni scheda con tutte le carte aperte, in inglese, e cerca parole italiane in ogni etichetta; `tab_names_are_translated_but_the_saved_key_is_not`, `i18n::tests::*`. Il test di snapshot accetta `PS_SNAPSHOT_EN=1` per produrre le immagini in inglese. Per aggiungere un testo: scrivilo in italiano dentro `tl(...)` e aggiungi la riga in `i18n_en.rs` (il test del dizionario segnala duplicati).

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
