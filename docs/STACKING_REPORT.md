# Stacking in stile AutoStakkert dentro Planetary Studio — report e lavoro da fare

Data: 2026-11-16. Scritto per essere incollato in un'altra chat: contiene contesto, fonti e compiti.

## Verdetto

Uno stacker "di base" nell'app è fattibile in 1–2 settimane perché il prototipo esiste già
(`crates/ps-train/src/bin/ser_stack.rs`, 356 righe, più `sim_stack/stack.rs`, 216 righe).
Per avvicinarsi davvero ad AutoStakkert 3 manca soprattutto **la selezione dei frame per ogni
punto di allineamento (AP)**, poi AP di più misure, un riferimento rifinito in modo iterativo e il
drizzle. Questo è il salto di qualità, non la somma dei frame.

Attenzione a una decisione già scritta: `docs/ARCHITETTURA.md` (riga 5) dice
"Luck imaging e stacking sono fuori scope". Va aggiornata se si procede.

## Cosa è verificato e cosa no

AutoStakkert è closed source. Quello che segue su AutoStakkert viene dal suo manuale (AS!2) e da
tutorial; la parte algoritmica dettagliata viene da **Siril** (documentazione ufficiale, molto
precisa) e da **Planetary System Stacker** (open source, Python), che fanno la stessa cosa e
dichiarano risultati simili. Non ho trovato un paper che descriva AutoStakkert: non esiste, per
quanto ho potuto cercare. I paper veri riguardano i pezzi (lucky imaging, drizzle, registrazione
sub-pixel).

## Come funziona AutoStakkert (pipeline)

1. **Analisi.** Ogni frame riceve un punteggio di qualità. Due stimatori: *Edge* (pianeti piccoli
   e luminosi, tutti gli AP usano gli stessi frame) e *Gradient/Laplace* (pianeti grandi: ogni AP
   usa il *proprio* sottoinsieme di frame). Parametro **Noise Robust**: valore alto per riprese
   rumorose (sfoca prima di misurare). **Force Global Quality** obbliga tutti gli AP a usare gli
   stessi frame, utile se si vedono cuciture.
2. **Allineamento globale grezzo.** Modo *Planet (COG)* segue il centro di luce del disco; modo
   *Surface* correla un dettaglio fisso (Luna, Sole). Se questo passo fallisce, tutto il resto fallisce.
3. **Riferimento.** Media dei migliori frame, numero scelto in automatico dalla qualità. Più frame =
   immagine più vera ma sfocata, quindi più difficile da correlare.
4. **Griglia di AP.** Uno o molti AP, di dimensioni diverse nella stessa immagine. Si scartano quelli
   sul fondo scuro (luminosità minima), sul bordo del disco e su zone senza struttura. Si
   consiglia che si sovrappongano.
5. **Selezione per AP.** Ogni AP ordina i frame in base alla qualità *nel proprio riquadro* e tiene
   i migliori N (percentuale o numero). Si possono provare più percentuali insieme.
6. **Spostamento locale.** Per ogni AP e frame si misura lo spostamento rispetto al riferimento.
7. **Somma e fusione.** I riquadri allineati si sommano per AP e si fondono tra loro con finestre
   morbide, per evitare cuciture.
8. **Opzioni.** *Normalize Stack* (uniforma la luminosità tra frame), *HQ Refine*, *Drizzle*
   1.5×/2×/3× (per immagini sottocampionate), *Crop/Expand*, *Dynamic Background*, sharpening
   opzionale con copia non nitida.

## Cosa dicono le fonti sui dettagli

**Siril — stacking multi-punto** (doc: siril.readthedocs.io, "Multi-Point Registration and Stacking")
- Qualità: **Laplaciano**, normalizzato per la luminosità media sopra soglia (così la trasparenza
  non sembra nitidezza).
- Riferimento: media del migliore 5% dei frame.
- Box di correlazione: mezza larghezza 24 px, passo della griglia 2,25 × mezza larghezza (54 px),
  raggio di ricerca 14 px per AP e 34 px globale.
- Rifiuto degli AP: luminosità minima, contrasto minimo, struttura minima (gradiente normalizzato
  0,04).
- **Raffinamento in due passate**: dopo la prima misura, per ogni AP si sommano i frame migliori
  (5% dei frame, tra 8 e 32) in un riferimento locale più nitido, e tutti gli spostamenti si
  rimisurano su quello. Se la seconda misura fallisce si tiene la prima.
- **Regolarizzazione degli spostamenti**: adattamento robusto sugli AP vicini (raggio 2,5 passi).
- **Selezione morbida**: i frame vicini al taglio sfumano con una finestra a coseno rialzato
  invece di un taglio netto.
- **Fusione**: finestra di Hann su ogni riquadro, livelli di luminosità riallineati nelle zone di
  sovrapposizione, fondo composto a parte per le zone senza AP.
- Un motore alternativo ("warp") interpola gli spostamenti in un unico campo di distorsione e
  ricampiona ogni frame una volta sola: nessun bordo da fondere. È quello che fa già il prototipo.
- Output a 1×, 1,5×, 2×, 3×; accumulo in float32; demosaico LMMSE a fine stacking.

**Planetary System Stacker** (github.com/Rolf-Hempel/PlanetarySystemStacker): classifica tutti i
frame, cerca sul migliore il riquadro con più struttura, allinea globalmente, fa la media dei
migliori, crea una maglia di AP (scartando i punti scuri o senza contrasto), per ogni AP
classifica i frame per contrasto locale e tiene i migliori N (diversi da punto a punto), calcola
gli spostamenti locali, somma i riquadri e li fonde su uno sfondo. Python + OpenCV + numpy.
Il dettaglio dei metodi di correlazione sta nella sua User Guide (PDF nel repo) o nel codice:
**non l'ho letto, da leggere prima di implementare**.

**Paper di riferimento**
- Fried 1978, *Probability of getting a lucky short-exposure image through turbulence*, JOSA 68,
  1651. Base teorica della selezione: la probabilità di un frame buono cala come
  exp(−0,1557·(D/r₀)²); con D/r₀ = 5 circa 1 frame su 9.
- Law, Mackay, Baldwin 2006, *Lucky imaging: high angular resolution imaging in the visible from
  the ground*, A&A 446, 739 (arXiv astro-ph/0507299). Selezione + somma shift-and-add: il guadagno
  di risoluzione con una perdita di segnale modesta.
- Fruchter & Hook 2002, *Drizzle: a method for the linear reconstruction of undersampled images*,
  PASP 114, 144 (arXiv astro-ph/9808087). Ogni pixel di ingresso diventa una "goccia" ridotta
  (parametro *pixfrac*) proiettata sulla griglia fine e pesata per area di sovrapposizione.
- Guizar-Sicairos, Thurman, Fienup 2008, *Efficient subpixel image registration algorithms*,
  Optics Letters 33, 156. Correlazione sub-pixel con DFT a matrice invece di FFT con zero-padding:
  molto più veloce, utile per misurare centinaia di AP per frame.
- Per i criteri di nitidezza (Laplaciano, Tenengrad, ecc.) la letteratura è Pertuz et al. 2013,
  *Analysis of focus measure operators for shape-from-focus*. Non sono riuscito ad aprirla, la
  cito dalla memoria: da verificare.

## Cosa c'è già nel repo

Il prototipo `ser_stack.rs` fa: lettura SER (Bayer, RGB/BGR, 8/16 bit, big endian), qualità di
ogni frame (energia del gradiente su luminanza sfocata, in bin 2×2), centro di luce, finestra
attorno al pianeta, riferimento dal migliore 10% allineato sul migliore, allineamento globale
(`ps_core::align::measure_shift`: correlazione di fase con whitening parziale, raffinamento
iterativo e fit gaussiano a tre punti, già buona), griglia di AP da 64 px con passo 40 px,
campo di spostamento morbido (pesi gaussiani) interpolato su una griglia da 8 px, ricampionamento
Catmull–Rom e somma. Scrive tre stack (completo, pari, dispari) per misurare il rumore casuale.
Parallelizzato con rayon.

`sim_stack/stack.rs` ne ripete la logica per il simulatore (dataset sintetico Giove già
generato: 175 run).

## Cosa manca rispetto ad AutoStakkert

| Manca | Effetto | Difficoltà |
|---|---|---|
| Selezione dei frame **per AP** | Il guadagno di nitidezza maggiore sui pianeti grandi | media |
| Riferimento rifinito per AP (seconda passata) | Misure più stabili, meno AP che sbagliano | media |
| Normalizzazione della qualità per luminosità | Evita che nuvole e trasparenza sembrino nitidezza | bassa |
| AP di dimensioni diverse + rifiuto per struttura/contrasto | Meno errori su bordo e zone piatte | bassa |
| Rifiuto degli spostamenti anomali + regolarizzazione robusta | Oggi si scarta solo se > 6 px | media |
| Selezione morbida al taglio | Meno rumore da frame al limite | bassa |
| Modo Surface (Luna/Sole) con dettaglio fisso | Il prototipo segue solo il centro di luce | media |
| Drizzle 1,5×/2×/3× (anche su Bayer) | Più risoluzione su immagini sottocampionate | alta |
| Correzione della dispersione atmosferica (RGB allineati a parte) | Rimuove frange rosse/blu alle alte latitudini | media |
| Normalize Stack, dark/flat | Riprese con cielo non nero | bassa |
| Lettura in streaming, AVI, altre sorgenti | SER da decine di GB, formati diversi | media-alta |
| Interfaccia, progresso, annullamento | Oggi è solo da riga di comando | media |

## Piano di lavoro (ordine consigliato)

Ogni fase deve chiudersi con una misura, non con una sensazione.

### Fase 0 — Preparazione
- Aggiornare `docs/ARCHITETTURA.md` (stacking non più fuori scope).
- Creare un modulo `ps-core::stack` (o un crate `ps-stack`) e spostarci la logica comune di
  `ser_stack.rs` e `sim_stack/stack.rs`, che oggi è duplicata. Il lettore SER va in `ps-io`.
- **Banco di prova**: i SER reali sono in `Immagini_test/ser` (Borodog 2023-11-10, Borodog
  2026-03-05, RedLion 2022-09-10, più le cartelle `stack`). Come verità si hanno gli stack
  AutoStakkert degli stessi autori dove disponibili e le simulazioni di `sim_stack` (verità nota).
  Metrica: errore rispetto alla verità sul simulato; sul reale confronto visivo e nitidezza a pari rumore.

### Fase 1 — Stacker base nell'app (1–2 settimane)
- Spostare lettura SER, grading, riferimento e warp in libreria, con API del tipo
  `analyse(path) -> Analysis` (qualità, centri) e `stack(analysis, options, progress) -> Image`.
- Esecuzione in `worker.rs` con progresso e annullamento; lettura dei frame a richiesta, non tutti in memoria.
- Interfaccia: aprire un SER, grafico di qualità ordinato (come AutoStakkert), percentuale o
  numero di frame, anteprima della griglia di AP, pulsante "Stack".
- Il risultato entra nella pipeline esistente come immagine grezza (denoise, nitidezza, ecc.).
- Fatto quando: lo stack di un SER di prova coincide con l'output attuale di `ser_stack` entro il rumore.

### Fase 2 — Qualità vera (2–3 settimane)  ← il lavoro che conta
1. **Qualità normalizzata** per luminosità media sopra soglia; opzione di robustezza al rumore (sfocatura prima del Laplaciano).
2. **Qualità per AP**: per ogni AP classificare i frame sul proprio riquadro e tenere i migliori N. Passare da "somma di frame warpati" a "per AP: media dei riquadri selezionati, poi fusione". Tenere come opzione il campo di warp unico per confronto.
3. **Fusione**: finestre di Hann, equalizzazione dei livelli nelle sovrapposizioni, fondo per le zone senza AP.
4. **Selezione morbida** al taglio (coseno rialzato), con larghezza adattata alla variazione tra AP vicini.
5. **Raffinamento a due passate**: riferimento locale per AP dai migliori 8–32 frame, rimisura degli spostamenti, ripiego sulla prima misura se fallisce.
6. **Rifiuto degli spostamenti anomali** e regolarizzazione robusta sugli AP vicini.
7. **AP multi-dimensione** e rifiuto per luminosità, contrasto e struttura minimi.
- Fatto quando: sul simulato l'errore verso la verità cala rispetto alla Fase 1 a pari numero di frame; sui SER reali non compaiono cuciture; il caso "Force Global Quality" funziona come ripiego.

### Fase 3 — Luna, Sole e colore (2–3 settimane)
- Modo *Surface*: ancoraggio su un dettaglio fisso, AP su tutta l'immagine, nessun centro di luce.
- Allineamento separato dei canali RGB (dispersione atmosferica), riusando `align::apply`, che già sposta R e B.
- *Normalize Stack*, dark e flat opzionali.

### Fase 4 — Drizzle e prestazioni (3–6 settimane)
- Drizzle con *pixfrac* e scala 1,5×/2×/3×, in float32, con mappa dei pesi; anche su Bayer (goccia per ogni pixel del mosaico, senza demosaico).
- Misura degli spostamenti con DFT a matrice (Guizar-Sicairos) al posto delle FFT intere per AP.
- Prestazioni: lettura in streaming, GPU (esiste `ps-core/src/gpu`), cache delle finestre.
- Fatto quando: drizzle 2× su un SER sottocampionato resta sopra il 1× + ingrandimento in nitidezza misurata e senza reticoli.

### Fase 5 — Integrazione con il resto dell'app
- Il prototipo già produce stack pari/dispari: serve a calibrare la stima del rumore appresa
  (`noise_net`). Salvare nei metadati dello stack il numero di frame e la percentuale tenuta,
  così denoise e nitidezza possono tenerne conto (oggi la rete non segue il numero di frame).
- Salvare le opzioni dello stack nel preset/progetto per renderlo riproducibile.

## Rischi e punti aperti
- Il guadagno della selezione per AP dipende dal seeing: su seeing buono è piccolo, su seeing
  cattivo grande. Va misurato prima di dichiarare riuscita la Fase 2.
- Le APs piccole su immagini molto rumorose sbagliano il match: servono rifiuto e raffinamento.
- Il drizzle su dati non sottocampionati peggiora o non cambia nulla (lo dice anche il manuale di
  AutoStakkert): va offerto, non attivato di default.
- Licenza: il repo è PolyForm-Noncommercial. Il codice di Siril (GPL) e di PSS (GPL) **non va
  copiato**: si implementa dalle descrizioni e dai paper. Da verificare prima di leggere i
  loro sorgenti.
- Non ho trovato né letto il codice sorgente di nessuno dei tre: tutto quanto sopra è documentazione e descrizioni.

## Fonti
- AutoStakkert!2, manuale: https://www.astrokraai.nl/software/manual/as2_planet.html
- Siril, Multi-Point Registration and Stacking: https://siril.readthedocs.io/en/latest/preprocessing/multipoint.html
- Planetary System Stacker: https://github.com/Rolf-Hempel/PlanetarySystemStacker
- Fried 1978, JOSA 68, 1651 (lucky exposure)
- Law, Mackay, Baldwin 2006: https://arxiv.org/abs/astro-ph/0507299
- Fruchter & Hook 2002 (drizzle): https://arxiv.org/abs/astro-ph/9808087 (precursore 1997: astro-ph/9708242)
- Guizar-Sicairos et al. 2008: https://labsites.rochester.edu/fienup/wp-content/uploads/2019/07/MGS_OL08_EffRegistration.pdf
- Tutorial con valori tipici AutoStakkert: https://www.photoartfromscience.com/single-post/2017/11/05/sharper-moon-shots-with-autostakkert
