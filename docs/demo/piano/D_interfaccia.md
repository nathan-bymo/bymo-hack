# Allegato D — Design system, schermate e componenti

Parte di [PIANO_DEMO.md](../PIANO_DEMO.md) (sezioni 10, 11, 12).

**Stato degli screenshot:** `docs/demo/reference/` **non esiste** alla data del piano (27/11/2026, 21:30 CET): i
4 screenshot non sono stati copiati. Quindi **nessun colore, font o misura qui sotto è misurato**. La descrizione
delle schermate viene dal prompt del team (sezione 3), non da un'analisi delle immagini. Il task M2.0 del
[piano di lavoro](E_piano_lavoro.md) misura i token appena gli screenshot sono disponibili; fino ad allora si usa
una **palette di ripiego dichiarata** e `docs/demo/VERIFICA.md` lo segnala.

---

## D.1 Design token

### D.1.1 Procedura di misura (da eseguire quando gli screenshot ci sono)

1. Copiare i file come `docs/demo/reference/A_erp_prodotto.png`, `B_master_data.png`, `C_processed_image.png`,
   `D_storefront.png` (cartella ignorata da git: vedi [§18](../PIANO_DEMO.md#18-gitignore-e-igiene-dei-dati)).
2. Leggere dimensioni e densità: `python3 -I -c "import sys;from PIL import Image;im=Image.open(sys.argv[1]);print(im.size, im.info.get('dpi'))" docs/demo/reference/A_erp_prodotto.png`.
   Se la larghezza supera 2.500 px, lo screenshot è probabilmente a 2× (retina): dividere per 2 tutte le misure in px.
3. Per ogni token, individuare in un visualizzatore un rettangolo uniforme `(x0, y0, x1, y1)` e prendere il colore
   più frequente (moda, non media: evita i bordi antialiasing):
   ```sh
   python3 -I -c "import sys;from PIL import Image;im=Image.open(sys.argv[1]).convert('RGB').crop(tuple(map(int,sys.argv[2:6])));n,c=max(im.getcolors(im.width*im.height));print('#%02x%02x%02x'%c, n)" docs/demo/reference/C_processed_image.png X0 Y0 X1 Y1
   ```
   Nessuno script salvato nel repo.
4. Font: misurare l'altezza delle maiuscole (cap height) di un'etichetta e di un valore in px; dimensione font ≈ cap
   height / 0,7 per un sans-serif di sistema. Spaziature: distanza tra righe consecutive della griglia
   label/valore. Raggi: contare i pixel della curva dell'angolo di pulsanti e campi.
5. Scrivere i valori in `src/demo/frontend/src/styles/tokens.css` con un commento per riga:
   `/* misurato: C_processed_image.png (x0,y0,x1,y1) */`, e aggiornare la tabella D.1.2 in `docs/demo/VERIFICA.md`.

### D.1.2 Tabella dei token

`Valore` resta **[DA MISURARE]** finché il punto 3 non è fatto. `Ripiego` = colore della palette predefinita di
Tailwind usato nel frattempo: **non** è un colore Galaxus e non va presentato come tale.

| token CSS | uso | screenshot e zona da campionare | valore | ripiego (Tailwind) |
|---|---|---|---|---|
| `--c-titlebar-bg` | barra titolo verde scuro "Products — …" | A, interno della barra titolo | [DA MISURARE] | `emerald-900` |
| `--c-titlebar-fg` | testo della barra titolo | A, lettere del titolo | [DA MISURARE] | `white` |
| `--c-accent-line` | linea rosso-arancio sotto l'header | A, la linea (altezza: contare i px) | [DA MISURARE] | `orange-600` |
| `--c-imagetype-bg` | pannello "Image type" azzurro | C, interno del pannello | [DA MISURARE] | `sky-50` |
| `--c-imagetype-border` | bordo del pannello | C, bordo | [DA MISURARE] | `sky-200` |
| `--c-switch-now` | pulsante viola "New design — Switch now →" | A, sfondo del pulsante | [DA MISURARE] | `violet-600` |
| `--c-link` | link sottolineati | A/B/C, testo di un link | [DA MISURARE] | `blue-700` |
| `--c-app-bg` | sfondo pagina (bianco/lavanda chiaro) | A, area vuota del pannello principale | [DA MISURARE] | `slate-50` |
| `--c-panel-bg` | sfondo sezioni/card | A, interno di una card a destra | [DA MISURARE] | `white` |
| `--c-border` | bordi sottili, separatori | A/B, linea tra righe | [DA MISURARE] | `slate-200` |
| `--c-text` | valori | A, testo di un valore | [DA MISURARE] | `slate-900` |
| `--c-text-muted` | etichette grigie | A, testo di un'etichetta | [DA MISURARE] | `slate-500` |
| `--c-toolbar-btn-bg` | pulsanti grigi della toolbar | A, `Enrich Product ▾` | [DA MISURARE] | `slate-100` |
| `--c-status-red` / `--c-status-blue` | pallini di stato | B, interno di un pallino | [DA MISURARE] | `red-600` / `blue-600` |
| `--c-store-pill` | pulsante nero "Mehr anzeigen" | D, interno del pulsante | [DA MISURARE] | `black` |
| `--c-store-price` | prezzo in grassetto | D | [DA MISURARE] | `slate-900` |
| `--c-conf-high` / `--c-conf-mid` / `--c-conf-low` | chip di confidenza (**nostri**, non Galaxus) | — | scelti per contrasto AA | `emerald-700` / `amber-700` / `red-700` su sfondi `-50` |
| `--c-invalid` | badge "seen during training — invalid" | — | nostro | `red-700` su `red-50` |
| `--font-sans` | tutto | — | font di sistema: `system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif` (nessun font scaricato) | idem |
| `--fs-label` / `--fs-value` / `--fs-title` / `--fs-section` | etichette, valori, barra titolo, titoli di sezione | A/C | [DA MISURARE] | 12 / 13 / 16 / 13 px (grassetto, sottolineato per le sezioni) |
| `--space-1…6` | griglia | A | [DA MISURARE] la spaziatura tra righe | 4 / 8 / 12 / 16 / 24 / 32 px |
| `--radius-field` | campo di ricerca "largo e arrotondato" | A | [DA MISURARE] | 9999px (pillola) |
| `--radius-btn` / `--radius-card` | pulsanti / card | A | [DA MISURARE] | 3px / 4px |
| `--shadow` | ombre | A ("quasi nessuna") | probabilmente `none` | `none` |

Uso in Tailwind (funziona con v3 `tailwind.config.ts` e v4 `@theme`): i token sono variabili CSS in `tokens.css`;
la config Tailwind li espone come colori, per esempio `colors: { titlebar: "var(--c-titlebar-bg)", … }` (v3) o
`@theme { --color-titlebar: var(--c-titlebar-bg); }` (v4). Si sceglie la versione in M0.4 e si fissa nel lockfile.

## D.2 Regole di stile e identità

- **Nessun logo, icona, font o immagine Galaxus** scaricato dal web o ritagliato dagli screenshot. Le scritte "ERP" e
  "GALAXUS" sono testo con un'icona semplice disegnata in SVG inline (sole/stella generici).
- **Badge permanente** nell'header di ogni schermata: **"Demo — Team Bymo · Hack Apertus"**, non chiudibile, anche in
  modalità presentazione e negli screenshot E2E.
- Il guscio ERP è **statico e finto**: voci di menu, widget e pagina prodotto hanno testi segnaposto (prodotto
  "Prodotto demo", ID `DEMO-0001`); nessun nome di dipendente, numero d'ordine, ID prodotto, prezzo o fornitore
  presi dagli screenshot. Le voci non implementate sono visibilmente inattive (cursore `not-allowed`, tooltip
  "non attivo nella demo"). Nessun testo suggerisce un collegamento all'ERP reale.
- I valori delle classi si mostrano con l'etichetta ERP e, al passaggio del mouse, la stringa interna esatta.

### D.2.1 Lingua dell'interfaccia: inglese **[DECISIONE PROPOSTA]**

L'ERP degli screenshot è in inglese ("Image type", "Cancel", "Enrich Product") e il pubblico è Galaxus: **tutti i
testi dell'app sono in inglese**. Il piano e la documentazione restano in italiano (regola del team); nelle
wireframe di D.4 alcune etichette sono descritte in italiano, ma il testo definitivo è quello della tabella qui
sotto (file `src/demo/frontend/src/strings.ts`, un solo punto di verità).

| chiave | testo UI (inglese, esatto) |
|---|---|
| `badge.demo` | Demo — Team Bymo · Hack Apertus |
| `badge.held_out` | held-out |
| `badge.phase1_internal` | phase-1 internal set (coverage) |
| `badge.seen_in_training` | seen during training — invalid *(tooltip: "This model was trained on these images: not a valid evaluation.")* |
| `badge.reported_only` | reported only |
| `badge.recomputed` | recomputed |
| `badge.kaggle_dev` | Kaggle dev (public) |
| `badge.na` | n/a |
| `badge.recorded` | Recorded {date} on {gpu} |
| `badge.live` | Live |
| `badge.parity_unverified` | parity not verified |
| `chart.reference_line` | Organisers' reference (Qwen 3.5 9B FT, dev) |
| `chart.reference_note` | Different set (Kaggle dev): not directly comparable with coverage. |
| `banner.coverage` | "coverage" is an internal phase-1 test set enriched with rare classes. It is meant to compare methods, not to estimate the Kaggle leaderboard. |
| `status.ai_suggested` / `status.reviewed` / `status.corrected` | AI suggested / Reviewed / Corrected |
| `confidence.head_softmax` | Head softmax probability |
| `confidence.letter_logprob` | Answer-letter probability (after the model's own description) |
| `confidence.sequence_prob` | Probability of the generated value |
| `confidence.not_comparable` | Confidence values from different methods are not comparable. |
| `output.invalid` | Invalid model output |
| `dimension.not_applicable` | Not scored for this kind of image (official rule applied to the prediction) |
| `context.missing` | No product context: this condition was not measured for the trained models. |
| `upload.never_dev_final` | Never upload Kaggle dev or final images. |
| `menu.inactive` | Not active in this demo |
| `model.unavailable.key` | Unavailable: LLM_API_KEY is not configured |
| `model.unavailable.gpu` | Unavailable: no GPU worker connected |
| `model.unavailable.weights` | Unavailable: weights not found locally ({repo}/{subfolder}) |

I messaggi `error.message` dell'API ([allegato C](C_api.md#c1-schema-di-errore-unificato)) sono anch'essi in inglese.

## D.3 Leggibilità e accessibilità

- Due densità: **Normale** (laptop, `font-size` radice 100%) e **Presentazione** (TV 1080p/4K: 125%, chip e
  dropdown più grandi, tabella con meno colonne visibili, pannelli di sviluppo nascosti). Interruttore in header e
  `?presentation=1` nell'URL.
- Contrasto testo/sfondo ≥ 4,5:1 (WCAG AA) per tutti i token, verificato in M6.4 con uno strumento di contrasto; i
  colori misurati che non lo rispettano si usano solo per sfondi, non per testo piccolo.
- La confidenza non è mai solo colore: chip = numero (2 decimali) + icona (▲ bassa) + colore.
- Tastiera: `J`/`K` immagine successiva/precedente nel dettaglio, `1`–`6` focus sulle 6 tendine, `S` salva, `Esc`
  annulla; focus visibile; tendine con `aria-label` = nome ERP della dimensione.
- Layout minimo 1280 px di larghezza (la demo è per laptop e TV); sotto, la colonna destra dei widget sparisce.

---

## D.4 Schermate

Valori nelle wireframe: `0.xx`, `x,x` e testi tra `<…>` sono **segnaposto**. I testi UI definitivi sono in inglese
(tabella [D.2.1](#d21-lingua-dellinterfaccia-inglese-decisione-proposta)); le etichette descritte in italiano qui sotto vanno tradotte.

### Schermata 1 — Guscio ERP e pagina prodotto demo

- **Scopo:** contesto Galaxus; punto di partenza della storia (pagina prodotto → `Enrich Product ▾` → classificazione).
- **Route:** layout `ErpShell` su tutte le route; pagina `/product/DEMO-0001`.
- **Riproduce:** screenshot A (guscio, barra titolo, toolbar, sezioni, colonna destra).

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ☀ ERP   (  🔍 Search for products, orders, customers, and more                )   [Demo — Team Bymo · Hack Apertus] │
├═════════════════════════════════════════════════════════════════════════════════════════════════════┤ ← --c-accent-line
│ ▸ Order management        │ ██ Products — Prodotto demo (Id DEMO-0001)  ☆ ███████████ │ ▾ Working Time    │
│ ▾ Products                │  Search · Advanced search                                 │   (segnaposto)    │
│ ▾ Product data            │  [Enrich Product ▾][Shops ▾][Deactivate ▾][Methods ▾][Log] │ ▾ Notifications   │
│   · Image classification  │   └─ Image classification (AI)  ← unica voce attiva        │   (segnaposto)    │
│     (AI)    ← NUOVA       │  ▼ Product Overview                                        │ ▸ User selection  │
│ ▸ Human Resources         │    Label ......... Value       Label ......... Value [img] │ ▸ Barcode         │
│ ★ Favourites              │  ▶ Provider order backlog   (inattiva)                     │ ▸ Sales Assistant │
│                           │  ▶ Chosen Sales Offer       (inattiva)                     │ ▸ Packing info    │
│ ┌───────────────────────┐ │  ▶ Retail information       (inattiva)                     │                   │
│ │ New design            │ │                                                            │                   │
│ │ Switch now →  (inatt.)│ │                                                            │                   │
│ └───────────────────────┘ │                                                            │                   │
└───────────────────────────┴────────────────────────────────────────────────────────────┴───────────────────┘
```

- **Componenti:** `ErpShell`, `ErpHeader` (`Wordmark`, `GlobalSearch`, `DemoBadge`, `DensityToggle`),
  `AccentLine`, `Sidebar` (`SidebarGroup`, `SidebarItem`, `SwitchNowButton`), `TitleBar`, `SubLinks`,
  `ActionToolbar` (`ToolbarDropdown`, `ToolbarButton`), `CollapsibleSection`, `LabelValueGrid`, `WidgetCard`.
- **Dati:** statici; il riquadro "Product Overview" mostra solo `Images: <n> (sessione demo)` da `GET /api/images`.
- **Interazioni:** `Enrich Product ▾ → Image classification (AI)` e la voce di sidebar portano alla coda della
  sessione corrente (o ne creano una con `POST /api/sessions`). `GlobalSearch` filtra per nome file le immagini
  della sessione (unica funzione attiva).
- **Stati:** backend irraggiungibile → banner rosso sotto l'header "Backend non raggiungibile (`/api/health`)".

### Schermata 2 — Coda di classificazione (batch)

- **Scopo:** caricare immagini, lanciare la classificazione, triage per confidenza più bassa, vedere la velocità.
- **Route:** `/image-classification?session=<session_id>`.
- **Riproduce:** stile di A (barra titolo, toolbar) e tabelle dense di B.

```
│ ██ Image classification (AI) — Sessione "<nome>"  ☆ █████████████████████████████████████████████████ │
│  Search · Advanced search                                                                              │
│  [Model: <nome modello> ● ready ▾] [▶ Run classification] [Compare models] [Export CSV] [Log]          │
│  ┌ Trascina qui immagini JPEG / PNG / WebP (max 20 MB, max 50 per volta) — oppure [Scegli file] ─────┐ │
│  │ Provenienza: (•) foto nostre  ( ) train/coverage  ( ) altro          ⚠ mai immagini dev o final   │ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│  ▼ Coda (12)      ▓▓▓▓▓▓▓▓░░░░  8/12 · x,x img/s misurati · <GPU o "gateway remoto"> · obiettivo ≥ 2 img/s │
│  ┌──┬──────────┬──────────────┬────────────┬───────────────┬────────────┬──────────┬──────────┬────────┬──────────────┐│
│  │☐ │ Immagine │ Subject      │ Background │ Perspective   │ Individuals│ Overlay  │ Language │Min conf│ Stato        ││
│  ├──┼──────────┼──────────────┼────────────┼───────────────┼────────────┼──────────┼──────────┼──────▲─┼──────────────┤│
│  │☐ │ 🖼 a.jpg │ <Classe> 0.xx│ <Cl.> 0.xx │ <Classe> 0.xx▲│ - 0.xx     │ - 0.xx   │ - 0.xx   │ 0.xx   │ AI suggested ││
│  │☐ │ 🖼 b.png │ …            │ …          │ … (grigio: non applicabile)                │        │ Corrected    ││
│  │☐ │ 🖼 c.jpg │ ⏳ in coda   │            │               │            │          │          │        │ —            ││
│  └──┴──────────┴──────────────┴────────────┴───────────────┴────────────┴──────────┴──────────┴────────┴──────────────┘│
│  Contesto prodotto: assente per 12/12 immagini — condizione non misurata per i modelli allenati [Aggiungi contesto] │
```

- **Componenti:** `TitleBar`, `ActionToolbar`, `ModelSelect`, `RunButton`, `UploadDropzone`, `ProgressBar`,
  `ThroughputBadge`, `ResultsTable` (`ResultRow`, `DimensionCell`, `ConfidenceChip`, `StatusPill`),
  `ModeBadge` (live/registrato), `ContextWarning`.
- **Dati:** `GET /api/images?session_id`, `GET /api/predictions?session_id&model_id`, `GET /api/models`,
  `POST /api/images`, `POST /api/predict`, `GET /api/jobs/{id}/events`, `GET /api/export.csv`.
- **Interazioni:** drop → upload con anteprima e lista dei rifiuti (codice errore leggibile); `Run classification`
  sulle righe selezionate o su tutte quelle senza predizione per il modello scelto; cambio di modello = nuova
  colonna di risultati per le **stesse** immagini senza ricaricarle (cache per modello); ordinamento per "Min conf"
  crescente (triage); clic su una riga → dettaglio; `Compare models` → schermata 4 con le righe selezionate.
- **Stati:** vuoto ("Nessuna immagine: trascina qui le foto"), caricamento modello (barra con messaggi da SSE),
  modello `unavailable` (pulsante Run disabilitato + motivo, es. "manca LLM_API_KEY" o "nessun worker GPU"),
  errore per immagine (cella rossa con codice), modalità registrato (badge "Recorded <date> on <GPU>" in
  testata e su ogni riga), immagine senza registrazione (`RECORDED_MISSING`: "non registrata per questo modello").

### Schermata 3 — Dettaglio immagine (cuore della demo)

- **Scopo:** il flusso reale di Galaxus: il modello pre-compila "Image type", una persona rivede, corregge, salva.
- **Route:** `/image-classification/:imageId?model=<model_id>`.
- **Riproduce:** screenshot C (anteprima grande in alto a sinistra, tabella metadati a destra, pannello azzurro
  "Image type" con icona robot e 6 tendine in due colonne, `✕ Cancel` a sinistra, salvataggio a destra).

```
│ ██ Processed image — <nome file> (img-demo-…)  ☆ ████████████████████████████████████████████████████ │
│  ← Coda · ‹ Precedente (K) · Successiva (J) ›                                                         │
│  ┌──────────────────────────────────────┐   ┌ Metadati ────────────────────────────────────────────┐ │
│  │                                      │   │ Manually uploaded           Sì (caricata nella demo) ⓘ│ │
│  │                                      │   │ Used by provider products   n/a ⓘ                     │ │
│  │          [ immagine grande ]         │   │ CDN URL                     n/a (file locale)          │ │
│  │                                      │   │ Width (px)                  <w>                        │ │
│  │                                      │   │ Height (px)                 <h>                        │ │
│  └──────────────────────────────────────┘   │ Quality Score               n/a                        │ │
│                                             │ Formato · dimensione        <JPEG> · <KB>               │ │
│                                             │ SHA-256                     <8 car.>…  [copia]          │ │
│                                             └────────────────────────────────────────────────────────┘ │
│  ┌ 🤖 Image type ──── Modello [<nome> ▾] · live · <ms> ms ─────────────────────────────────────────────┐ │
│  │ Subject      [<Etichetta>          ▾] 0.xx       Individuals  [ -                  ▾] 0.xx          │ │
│  │ Background   [<Etichetta>          ▾] 0.xx       Overlay      [ -                  ▾] 0.xx          │ │
│  │ Perspective  [<Etichetta>          ▾] 0.xx ▲     Language     [ -  (non applicabile)▾] 0.xx          │ │
│  │ ✎ Modificato da persona: Perspective                                                               │ │
│  │ ▸ Contesto prodotto (facoltativo)    ▸ Testo del modello (descrizione / JSON grezzo)               │ │
│  │ ⓘ Confidenza: softmax della testa — non confrontabile con quella dei modelli generativi             │ │
│  │ [✕ Cancel]                                                                         [ Save ]       │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
```

- **Componenti:** `TitleBar`, `ImagePreview` (zoom al clic), `MetadataTable`, `ImageTypePanel`
  (`RobotIcon`, `DimensionDropdown`, `ConfidenceChip`, `TopKPopover`, `EditedMarker`, `NotApplicableHint`),
  `ContextForm`, `RawOutputDisclosure`, `ConfidenceMethodNote`, `ModeBadge`.
- **Dati:** `GET /api/images/{id}` (metadati reali: larghezza, altezza, formato, byte, sha256; `Quality Score` = n/a
  perché non esiste una fonte), `GET /api/predictions?image_id&model_id`, `POST /api/predict` (se manca),
  `PUT /api/images/{id}/review`, `PUT /api/images/{id}/context`, `GET /api/taxonomy`.
- **Interazioni:** ogni tendina elenca **tutte** le classi della dimensione (ordine della tassonomia), con le prime
  3 del modello in cima e la loro probabilità; clic sul chip → `TopKPopover`; modificare un valore accende
  `EditedMarker` e il bordo della tendina; `Save` → `PUT …/review` → stato `Reviewed` (nessuna modifica) o
  `Corrected`; `Cancel` → ripristina la predizione; cambio modello nel pannello → ricarica la predizione di quel
  modello (o la calcola), senza ricaricare l'immagine; salvare il contesto → avviso "il contesto cambia l'input:
  rilancia la classificazione".
- **Stati:** nessuna predizione (pannello con pulsante "Classifica con <modello>"), in corso (scheletro delle
  tendine), output non valido (pannello con bordo rosso: "Output non valido del modello" + testo grezzo; tendine
  vuote da compilare a mano), modello non disponibile (motivo + link alla schermata performance), registrato
  (badge).

### Schermata 4 — Confronto modelli

- **Scopo:** stesse immagini, più modelli: dove sono d'accordo e dove no, e quanto costano in tempo.
- **Route:** `/image-classification/compare?session=<id>&models=<id1>,<id2>,…&images=<id…>`.

```
│ ██ Compare models — Sessione "<nome>" ██████████████████████████████████████████████████████████████████ │
│  Modelli: [☑ 8B zero-shot] [☑ 70B zero-shot] [☑ P11 LoRA+teste] [☐ Full FT] [☐ P10]   [▶ Calcola mancanti] │
│  Latenza misurata qui: 8B zs x,x s/img (gateway) · 70B zs x,x s/img (gateway) · P11 xx ms/img (<GPU>) · …  │
│  Filtro: (•) tutte ( ) solo disaccordi      Dimensioni: [tutte ▾]                                          │
│  ┌──────────┬───────────── Subject ─────────────┬──────── Background ────────┬ … ┬ Disaccordi ┐            │
│  │ Immagine │ 8B zs │ 70B zs │ P11  │ Full FT   │ 8B zs │ 70B zs │ P11 │ FT   │   │            │            │
│  ├──────────┼───────┼────────┼──────┼───────────┼───────┼────────┼─────┼──────┼───┼────────────┤            │
│  │ 🖼 a.jpg │ <c>   │ <c>    │ <c>  │ <c>       │ ░<c>░ │ <c>    │ <c> │ <c>  │   │ 1/6        │            │
│  └──────────┴───────┴────────┴──────┴───────────┴───────┴────────┴─────┴──────┴───┴────────────┘            │
│  ⓘ Le confidenze di modelli diversi non sono confrontabili tra loro (metodi diversi).                      │
```

- **Componenti:** `ComparisonGrid` (`ComparisonCell`, `DisagreementHighlight`), `ModelToggleList`,
  `LatencyStrip`, `ModeBadge`.
- **Dati:** `GET /api/predictions?session_id` per ogni modello; `POST /api/predict` per le celle mancanti;
  `GET /api/throughput?model_id`.
- **Interazioni:** cella in disaccordo (≥ 2 valori diversi tra i modelli selezionati) evidenziata; clic → dettaglio
  con quel modello; esclusione automatica, con motivo, dei modelli `unavailable`.
- **Stati:** meno di 2 modelli con risultati → messaggio "seleziona almeno due modelli con predizioni".

### Schermata 5 — Dashboard delle prestazioni

- **Scopo:** metriche oneste: da dove viene ogni numero, su quale set, con quali dati di training.
- **Route:** `/image-classification/performance?set=coverage`.

```
│ ██ Performance — modelli Apertus █████████████████████████████████████████████████████████████████████ │
│ ⚠ "coverage" è un set interno di fase 1 arricchito con classi rare: utile per confrontare metodi,        │
│   NON stima la classifica Kaggle (src/training/common/splits.py:12-13).   Set: [coverage ▾]  ☐ includi non validi │
│ ┌ Model card ───────────┐ ┌ Model card ───────────┐ ┌ Model card ───────────┐ ┌ Model card ─────────────┐ │
│ │ P11 LoRA + teste      │ │ 70B zero-shot v2      │ │ 8B zero-shot v2       │ │ Full FT (fase 2)        │ │
│ │ macro F1 0.xxx        │ │ macro F1 0.xxx        │ │ macro F1 0.xxx        │ │ coverage: ⛔ seen during│ │
│ │ [phase-1 internal]    │ │ [phase-1 internal]    │ │ [phase-1 internal]    │ │ training — invalid      │ │
│ │ [reported only]       │ │ [reported only]       │ │ [reported only]       │ │ Kaggle dev: n/a         │ │
│ │ train: 9.344 img      │ │ train: nessuno        │ │ train: nessuno        │ │ train: 11.876 img       │ │
│ │ x,x img/s · <GPU>     │ │ x,x img/s · gateway   │ │ …                     │ │ …                       │ │
│ └───────────────────────┘ └───────────────────────┘ └───────────────────────┘ └─────────────────────────┘ │
│ ▼ Macro F1 per dimensione         ─ ─ ─ 0.917 Organisers' reference (Qwen 3.5 9B FT, dev) — other set │
│   [barre raggruppate: 6 dimensioni × modelli]                                                             │
│ ▼ F1 per classe   [subject|background|perspective|individuals|overlay|language]   ★ = supporto < 30       │
│ ▼ Matrice di confusione  [perspective ▾] [modello ▾]  ☐ normalizza per riga  (solo dove esiste)          │
│ ▼ Confidenza: corrette vs sbagliate + calibrazione   → n/a finché non ci sono predizioni per immagine    │
│ ▼ Velocità e costo   [GPU: A100 80GB ▾] [$/h: 2,50]   barre img/s, linea obiettivo 2 img/s, $ / 1.000 img │
```

- **Componenti:** `CoverageWarningBanner`, `SetSelect`, `ModelCard`, `MetricBadge`, `DimensionBarChart`
  (`ReferenceLine`), `PerClassTable`, `ConfusionMatrix`, `ConfidenceHistogram`, `ReliabilityDiagram`,
  `CostPanel`, `SourceFootnote` (fonte `file:riga` sotto ogni grafico).
- **Dati:** `GET /api/metrics/compare?models=…&set=…`, `GET /api/metrics/{model_id}`, `GET /api/throughput`,
  `GET /api/settings`/`PUT`.
- **Interazioni:** cambio set; toggle "includi non validi" (aggiunge le barre rosse tratteggiate dei risultati
  contaminati, mai di default); tab per dimensione; selezione dimensione e modello della matrice; modifica di GPU e
  $/h (ricalcola solo il costo).
- **Stati:** set senza dati per un modello → card con "n/a su questo set"; matrice non disponibile → "servono
  predizioni per immagine: `src/demo/scripts/evaluate.py` (GPU, solo con ok)".
- Separazione dai dati per immagine: questa schermata non mostra mai la confidenza di un'immagine caricata, e le
  schermate 2–4 non mostrano mai metriche di set (solo un link "Come va questo modello sui set di prova →").

### Schermata 6 (stretch) — Anteprima storefront

- **Scopo:** cosa cambierebbe nel negozio: quale immagine diventa principale, quali si nascondono.
- **Route:** `/storefront-preview?session=<id>&model=<id>`.
- **Riproduce:** screenshot D (wordmark con stella, ricerca "Wonach suchst du?", albero categorie, card con brand
  grigio, prezzo in grassetto, titolo, riga specifiche, valutazione, pulsante nero a pillola "Mehr anzeigen").

```
│ ☆ GALAXUS   (  Wonach suchst du?                         )                [Demo — Team Bymo · Hack Apertus] │
│ Kategorien            │  Galleria proposta — REGOLA ILLUSTRATIVA, non la regola Galaxus     [Modifica regola] │
│  Prodotto demo        │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │
│  **Bilder**           │  │ #1 MAIN  │ │ #2       │ │ #3       │ │ #4       │ │ nascosta │                  │
│                       │  │ 🖼       │ │ 🖼       │ │ 🖼       │ │ 🖼       │ │ 🖼 (grig)│                  │
│                       │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘                  │
│                       │  <Brand>  ·  Prezzo n/a  ·  Prodotto demo  ·  <riga specifiche>  ·  ☆☆☆☆☆ n/a       │
│                       │  ( Mehr anzeigen )                                                                 │
│                       │  Perché #1: regola "foto principale" (subject full_product, background white, …)    │
```

- **Regola** in `src/demo/config/storefront_rule.yaml`, ispirata a `challenge_description.md:5-6` ("decides which
  image comes first in the gallery, which images are hidden…"):

```yaml
label: "Regola illustrativa (non è la regola Galaxus)"
idea_source: "challenge_description.md:5-6"
hide_if: {subject: [placeholder]}
rank:   # vince il primo gruppo che corrisponde; a parità, confidenza minima più alta
  - {name: "foto principale", when: {subject: [full_product], background: [white],
      perspective: [front_head_on, front_angle_left, front_angle_right], text_classification: [none], individuals: [none]}}
  - {name: "prodotto intero", when: {subject: [full_product]}}
  - {name: "lifestyle", when: {background: [scene_lifestyle]}}
  - {name: "dettaglio e confezione", when: {subject: [detail_closeup, multiview, package_and_product, packaging, delivery_scope, product_grouping, multipack_trade_unit, accessories]}}
  - {name: "grafica e testo", when: {subject: [infographic, energy_label, logo]}}
```

- **Componenti:** `StorefrontShell`, `StoreWordmark`, `StoreSearch`, `CategoryTree`, `GalleryStrip`,
  `ProductCard`, `PillButton`, `RuleExplainer`, `RuleEditor`.
- **Dati:** `GET /api/images?session_id`, `GET /api/predictions` (valori rivisti se ci sono), regola da
  `GET /api/settings` (campo `storefront_rule`).

---

## D.5 Catalogo dei componenti

| componente | responsabilità | props principali | stati | schermate |
|---|---|---|---|---|
| `ErpShell` | layout: header, linea d'accento, sidebar, area principale, colonna destra opzionale | `rightColumn?: ReactNode`, `density` | — | 1–5 |
| `ErpHeader` | wordmark "ERP", ricerca, `DemoBadge`, densità | `onSearch(q)` | backend offline | 1–5 |
| `Wordmark` | testo + icona SVG generica | `variant: "erp" \| "store"` | — | 1–6 |
| `DemoBadge` | "Demo — Team Bymo · Hack Apertus", sempre visibile | — | — | tutte |
| `AccentLine` | linea colorata sotto l'header | — | — | 1–5 |
| `Sidebar` / `SidebarGroup` / `SidebarItem` | gruppi espandibili, voce attiva, voci inattive | `groups[]`, `activePath` | aperto/chiuso, inattivo | 1–5 |
| `SwitchNowButton` | pulsante viola decorativo | — | inattivo | 1–5 |
| `TitleBar` | barra verde con titolo, ID e ☆ | `title`, `id?`, `starred?` | — | 1–5 |
| `SubLinks` | "Search · Advanced search" e navigazione | `links[]` | — | 1–3 |
| `ActionToolbar` / `ToolbarButton` / `ToolbarDropdown` | pulsanti grigi con menu | `items[]` | disabilitato con tooltip | 1, 2, 4 |
| `CollapsibleSection` | sezione con ▼ e titolo sottolineato | `title`, `defaultOpen`, `inactive?` | aperta/chiusa/inattiva | 1, 3, 5 |
| `LabelValueGrid` | griglia densa etichetta/valore a 2 colonne | `rows: {label, value, info?}[]` | valore `n/a` | 1, 3 |
| `WidgetCard` | card della colonna destra | `title` | chiusa | 1 |
| `UploadDropzone` | drag & drop (react-dropzone), scelta provenienza, rifiuti | `sessionId`, `onUploaded` | idle, trascinamento, upload, errori per file | 2 |
| `ModelSelect` | elenco modelli con pallino di stato e modalità | `value`, `onChange`, `models` | ready/not_loaded/loading/unavailable/registrato | 2, 3 |
| `RunButton` | lancia `POST /api/predict` | `modelId`, `imageIds` | disabilitato con motivo, in corso | 2, 4 |
| `ProgressBar` | avanzamento dal flusso SSE | `jobId` | in corso, completato, errore | 2, 4 |
| `ThroughputBadge` | img/s misurati e condizioni, obiettivo 2 img/s | `batch` | n/a | 2, 4 |
| `ResultsTable` / `ResultRow` / `DimensionCell` | coda con 6 dimensioni, ordinamento per min conf | `images`, `predictions`, `sort` | vuota, in coda, errore, non applicabile | 2 |
| `ConfidenceChip` | numero + icona + colore; metodo nel tooltip | `value: number \| null`, `method` | alta/media/bassa/n/a | 2, 3 |
| `StatusPill` | `AI suggested` / `Reviewed` / `Corrected` | `status` | — | 2, 3 |
| `ModeBadge` | "Live" o "Recorded … on …" | `mode`, `recorded?` | — | 2–4 |
| `ContextWarning` / `ContextForm` | avviso contesto assente; form dei 5 campi | `imageId`, `context` | salvato, modificato | 2, 3 |
| `ImagePreview` | immagine grande con zoom | `src`, `alt` | caricamento | 3 |
| `MetadataTable` | metadati reali dell'immagine, `n/a` dove non c'è fonte | `image` | — | 3 |
| `ImageTypePanel` | pannello azzurro con robot, 6 tendine, Cancel/Save | `imageId`, `modelId`, `prediction`, `onSave` | nessuna predizione, in corso, non valido, modificato, salvato | 3 |
| `DimensionDropdown` | tendina di una dimensione con top-3 in cima | `dimension`, `value`, `topK`, `applicable` | modificata, non applicabile | 3 |
| `TopKPopover` | le 3 classi più probabili con barre | `topK`, `method` | n/a | 3 |
| `EditedMarker` | "✎ modificato da persona" | `changed: string[]` | — | 3 |
| `RawOutputDisclosure` | descrizione zero-shot o JSON grezzo di P10 | `text` | assente | 3 |
| `ConfidenceMethodNote` | spiega il metodo della confidenza e la non confrontabilità | `method` | — | 3, 4 |
| `ComparisonGrid` / `ComparisonCell` | griglia immagine × dimensione × modello con disaccordi | `images`, `models`, `predictions`, `filter` | cella mancante, disaccordo | 4 |
| `LatencyStrip` | latenza misurata per modello | `throughput[]` | n/a | 4 |
| `CoverageWarningBanner` | avviso fisso su coverage ≠ classifica | `set` | — | 5 |
| `MetricBadge` | badge di validità e provenienza | `kind` | 6 tipi (§9.3 del piano) | 5, schede modello |
| `ModelCard` | sintesi per modello | `model`, `metrics`, `throughput` | n/a, non valido | 5 |
| `DimensionBarChart` | Recharts, barre per dimensione e modello + linea 0.917 | `data`, `reference` | dati parziali | 5 |
| `PerClassTable` | P/R/F1/supporto, classi rare evidenziate | `perClass`, `rareBelow=30` | n/a | 5 |
| `ConfusionMatrix` | heatmap (visx) con normalizzazione per riga | `labels`, `matrix`, `normalize` | non disponibile | 5 |
| `ConfidenceHistogram` / `ReliabilityDiagram` | distribuzione corrette/sbagliate, calibrazione | `pairs` | n/a senza predizioni per immagine | 5 |
| `CostPanel` | GPU, $/h, costo per 1.000 immagini, obiettivo 2 img/s | `throughput`, `settings` | n/a per API | 5 |
| `SourceFootnote` | "Fonte: file:riga · set · training" | `source`, `set`, `trainedOn` | — | 5 |
| `EmptyState` / `ErrorState` / `ModelUnavailableState` | stati vuoti ed errori coerenti (codice, messaggio, azione) | `code`, `message`, `action?` | — | tutte |
| `Toast` | conferme e errori brevi | `kind`, `text` | — | tutte |
| `DensityToggle`, `GlobalSearch`, `RobotIcon`, `NotApplicableHint`, `DisagreementHighlight`, `ModelToggleList`, `SetSelect`, `ReferenceLine` | piccoli componenti usati dentro quelli sopra: densità laptop/TV, ricerca (filtra per nome file), icona robot SVG, suggerimento "non applicabile", evidenziazione dei disaccordi, scelta dei modelli da confrontare, scelta del set, linea di riferimento 0.917 | — | — | 2–5 |
| `StorefrontShell`, `StoreWordmark`, `StoreSearch`, `CategoryTree`, `GalleryStrip`, `ProductCard`, `PillButton`, `RuleExplainer`, `RuleEditor` | schermata 6 | — | — | 6 |
