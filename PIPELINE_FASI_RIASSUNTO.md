# DEPTH-DOM Pipeline — Riepilogo Fasi

**Data:** 2026-04-04
**Pipeline:** 12 fasi per la generazione di segnali di trading ML-ready
**Input:** File `.depth` binari Sierra Chart (NQ futures)
**Output:** Modelli ML + bridge live per esecuzione in Sierra Chart

---

## Struttura Directory

```
DEPTH DOM VPS/
│
├── CLAUDE.md                     ← istruzioni progetto per Claude Code
├── PIPELINE_FASI_RIASSUNTO.md   ← questo file
│
├── P1/   depth_parser.py, main.py, README.md
├── P2/   vps_book_reconstructor.py, vps_book_reconstructor_fullnumba.py, README.md
├── P2b/  vps_phase2b_data_fusion.py, README.md
├── P3/   vps_feature_engineering_vectorized.py, README.md
├── P4/   vps_feature_engineering_agg.py, README.md
├── P5/   vps_cusum_sampler.py, README.md
├── P6/   vps_excursion_analysis_vectorized.py, README.md
├── P7/   vps_phase7_labeling.py, README.md
├── P7b/  vps_phase7b_macro_filter.py, README.md
├── P8/   vps_phase8_entry_model.py, README.md
├── P11/  vps_phase11_rl_execution.py, README.md
├── P12/  vps_phase12_sierra_bridge.py, README.md
│
├── SHARED/
│   ├── _pipeline_constants.py     ← costanti condivise (CANDIDATES, helpers)
│   ├── utils.py                  ← utilities condivise
│   └── README.md
│
├── ORCHESTRATOR/
│   ├── vps_multiday_runner.py    ← batch orchestrator
│   ├── incremental_p7p8_runner.py ← runner P7/P8 incrementale
│   ├── aggregate_results.py      ← aggregazione risultati
│   ├── plot_dashboard.py        ← dashboard plotting
│   ├── audit_pipeline.py         ← audit pipeline
│   ├── status_live.py            ← status live
│   ├── vps_watchdog.sh          ← watchdog
│   ├── cron_orchestrator.sh      ← cron 30-min orchestrator
│   └── README.md
│
├── INPUT/                        ← .depth source files (25 giorni)
│
├── VPS: /opt/depth-dom/  ← 24 script Python + 3 shell, flat
│         • depth_parser.py, main.py, aggregate_results.py, plot_dashboard.py
│         • vps_book_reconstructor.py (P2)
│         • vps_feature_engineering_vectorized.py (P3)
│         • vps_feature_engineering_agg.py (P4)
│         • vps_cusum_sampler.py (P5)
│         • vps_excursion_analysis_vectorized.py (P6)
│         • vps_phase7_labeling.py (P7)
│         • vps_phase7b_macro_filter.py (P7b)
│         • vps_phase8_entry_model.py (P8)
│         • vps_phase11_rl_execution.py (P11)
│         • vps_phase12_sierra_bridge.py (P12)
│         • vps_phase2b_data_fusion.py (P2b)
│         • vps_multiday_runner.py, vps_multiday_aggregator.py
│         • incremental_p7p8_runner.py, audit_pipeline.py, status_live.py
│         • _pipeline_constants.py, utils.py
│         • cron_orchestrator.sh, vps_watchdog.sh, archive_completed_day_v2.sh
```

---

## Panoramica del Pipeline

```
.depth (binario)
  │
  ▼ P1 ──► events.csv                    (~50-100 MB/giorno)
  │         │
  ▼ P2 ──► snapshots.csv                 (~427 MB/giorno)
  │         │
  ▼ P2b ─► snapshots_fused.csv           (LOB + Time & Sales)
  │         │
  ▼ P3 ──► features_dom.csv              (~310 MB/giorno)
  │         │
  ▼ P4 ──► features_dom_agg.csv
  │         │
  ▼ P5 ──► sampled_events.csv             (~1% degli eventi)
  │         │
  ▼ P6 ──► excursion_stats.csv
  │         │
  ▼ P7 ──► phase7_labels_*/              (3 candidati triple-barrier)
  │         │
  ▼ P7b ─► *_gex_pos.csv / *_gex_neg.csv (filtro macro)
  │         │
  ▼ P8 ──► model.pkl + report
  │
  ▼ P11 ─► Agente RL Actor-Critic         (live service)
  │
  ▼ P12 ─► Bridge TCP ──► Sierra Chart    (live service)
```

**Hardware VPS:** Contabo Cloud VPS 30 SSD — 8 core, 24GB RAM, 400GB SSD
**VPS:** `root@185.185.82.205` — codice in `/opt/depth-dom/`

---

## Fase 1 — Parsing Binario

**Script:** `vps_depth_parser.py` (optimized) / `PHASE_1_3/vps_depth_parser.py`

**Input:** `/opt/depth-dom/input/NQ*/YYYY-MM-DD.depth` (formato Sierra Chart proprietario)
**Output:** `events.csv`

### Formato File .depth (reverse-engineered)

```
Header (64 byte):
  00-03: Magic "SCDD"
  04-07: HeaderSize = 64
  08-11: RecordSize = 24
  12-15: Version

Record (24 byte):
  00-07: int64  DateTime (microsecondi da 1899-12-30)
  08:    uint8  Command  (0-7)
  09:    uint8  Flags
  10-11: uint16 NumOrders
  12-15: float32 Price
  16-19: uint32 Quantity
  20-23: reserved
```

### 8 Comandi DOM

| Codice | Nome | Effetto |
|--------|------|---------|
| 0 | NO_COMMAND | NOP — mantiene lo stato |
| 1 | CLEAR_BOOK | Reset completo del book (aste) |
| 2 | ADD_BID_LEVEL | Aggiunge livello al bid side |
| 3 | ADD_ASK_LEVEL | Aggiunge livello all'ask side |
| 4 | MODIFY_BID_LEVEL | Modifica quantità bid |
| 5 | MODIFY_ASK_LEVEL | Modifica quantità ask |
| 6 | DELETE_BID_LEVEL | Rimuove livello bid |
| 7 | DELETE_ASK_LEVEL | Rimuove livello ask |

**Ottimizzazioni:**
- Dict pre-allocato riutilizzato nel loop
- `struct.unpack` pre-compilato
- Streaming writer — nessun buffer completo in RAM
- Rimosso secondo pass `count_by_command()` (solo summary opzionale)

**Checkpoint:** `_checkpoints/p1_parse.done`

---

## Fase 2 — Ricostruzione Order Book

**Script:** `OPTIMIZATIONS/vps_book_reconstructor.py`

**Input:** `events.csv`
**Output:** `snapshots.csv`

### Algoritmo

Mantiene 10 livelli bid + 10 ask in strutture dati ottimizzate:
- **Bid:** max-heap (prezzo decrescente)
- **Ask:** min-heap (prezzo crescente)

Per ogni evento di `events.csv`:
1. Applica comando al book state
2. Ogni N eventi, emette snapshot completo

### Output Snapshot (56 colonne)

```
ts, best_bid, best_ask, spread, mid_price,
bid_px_1..10, bid_qty_1..10,
ask_px_1..10, ask_qty_1..10
```

**Bug fixati:**
- ✅ `sorted()` O(n log n) → `heapq.nlargest(10)` O(n log k) per top-10 livelli
- ✅ Crossed book rejection: skip snapshot se `best_bid >= best_ask`

**Checkpoint:** `_checkpoints/p2_reconstruct.done`

`★ Insight: heapq.nlargest(k, n)` è O(n log k) vs O(n log n) di sorted() — con n=100+ livelli e k=10, il risparmio è ~90%`

---

## Fase 2b — Data Fusion (LOB + Time & Sales)

**Script:** `OPTIMIZATIONS/vps_phase2b_data_fusion.py`

**Input:** `snapshots.csv` + trades.csv (esterno, es. Sierra Chart T&S)
**Output:** `snapshots_fused.csv` (o sovrascrive snapshots.csv)

### Perché Serve

Quando `bid_qty_1` scende, non sappiamo se è:
- **ΔC** (Cancellation) — ordine cancellato
- **ΔM** (Market Order) — ordine eseguito

**Formula accademica esatta:**
```
ΔC (Cancellation) = ΔV_total − M (trade volume)
```

### Algoritmo

`pandas.merge_asof` con direzione **backward**:
- Per ogni snapshot, associa l'ultimo trade avvenuto nel passato
- Evita look-ahead bias (no future information leakage)

**Nota:** Fase ufficiale del flusso P1→P2→P2b→P3. Integrata in `main.py` e `vps_multiday_runner.py`. Se `trades.csv` non disponibile per un giorno, la fase viene skippata automaticamente con warning e il flusso continua. Sentinel: `_checkpoints/p2b_fusion.done`

---

## Fase 3 — Feature Engineering Vettorializzata

**Script:** `OPTIMIZATIONS/vps_feature_engineering_vectorized.py`

**Input:** `snapshots.csv`
**Output:** `features_dom.csv` (~310 MB/giorno)

### Feature Calcolate (~50 colonne)

**LOB Spaziale (Ξ):**
- Densità fisica (tick distance) ai vari livelli

**Book Imbalance:**
```python
imbalance_1  = (bid_qty_1 − ask_qty_1) / (bid_qty_1 + ask_qty_1)
imbalance_5  = sum(bid_qty_1..5 − ask_qty_1..5) / sum(...)
imbalance_10 = sum(bid_qty_1..10 − ask_qty_1..10) / sum(...)
```

**Stack/Pull (Δquantity per livello):**
```python
stack_bid_1 = max(0, bid_qty_1(t) − bid_qty_1(t−1))   # quantity aggiunta
pull_bid_1  = max(0, bid_qty_1(t−1) − bid_qty_1(t))    # quantity rimossa
```

**Microprice:**
```python
microprice = (best_bid × ask_qty_1 + best_ask × bid_qty_1) / (bid_qty_1 + ask_qty_1)
```
Prezzo "fair" ponderato per volume — più informativo del mid quando c'è sbilanciamento.

**Altro:** spread_ticks, mid_price_diff, bid/ask_depth_5, depth_ratio

**Tecnica:** Fully vectorized pandas/numpy — nessun loop Python row-by-row.

**Bug fixato:** `depth_ratio = bid_depth_5 / ask_depth_5` poteva produrre `inf` → capped a 100.0.

**Checkpoint:** `_checkpoints/p3_features.done`

`★ Insight: vectorized pandas/numpy processa milioni di righe/secondo vs migliaia in Python row-by-row — 10-50× speedup`

---

## Fase 4 — Aggregazione Rolling Window

**Script:** `OPTIMIZATIONS/vps_feature_engineering_agg.py`

**Input:** `features_dom.csv`
**Output:** `features_dom_agg.csv`

### Feature Rolling

| Finestra | Feature | Descrizione |
|----------|---------|-------------|
| 1s | imbalance_mean/std | Media/devstd squilibrio |
| 5s | imbalance_mean/std | Media/devstd squilibrio |
| 30s | imbalance_mean/std | Media/devstd squilibrio |
| 1s | stack/pull aggregati | Somma stack/pull nella finestra |
| 1s | ps_net_weighted_mean | Microprice medio ponderato |
| 1s | exhaustion_count | Eventi con qty ≤ soglia |

### Bug Critico Risolto — O(n) → O(1)

```python
# PRIMA (O(n) per ogni chiamata — 3× per riga):
def get(self):
    return sum(1 for i in range(n) if self.bid_qty[i] <= THRESHOLD)

# DOPO (O(1) — running counter):
def update(self):
    if new_qty <= THRESHOLD:
        self.exhaustion_count += 1

def get(self):
    return self.exhaustion_count  # O(1)
```

**Risultato: 70-80% speedup** su questa fase.

**Checkpoint:** `_checkpoints/p4_agg.done`

`★ Insight: exhaustion_count = eventi dove il book si svuota (qty ≤ soglia). Predice liquidazione/pressione — running counter O(1) invece di O(n) è la differenza tra 390M iterazioni e 390M letture`

---

## Fase 5 — CUSUM Sampling

**Script:** `OPTIMIZATIONS/vps_cusum_sampler.py`

**Input:** `features_dom.csv` + `features_dom_agg.csv`
**Output:** `sampled_events.csv` (~1% degli eventi originali)

### Algoritmo CUSUM

```
Pass 1: Calcola h = 25° percentile(|Δmid_price|) non nulli
Pass 2: Accumula Δmid_price in CUSUM S
         Quando |S| > h → emetti evento → resetta S
```

**Risultato:** ~10K-50K eventi campionati da ~5-20M per sessione.

**Ottimizzazioni:**
- `np.percentile()` invece di `sort()` → O(n) vs O(n log n)
- `list[list[str]]` invece di `list[dict]` → elimina overhead dizionari

**Checkpoint:** `_checkpoints/p5_sample.done`

`★ Insight: CUSUM è un filtro passa-alto statistico — tiene solo "regime shifts" persistenti, filtra il rumore browniano. Campionare all'1% riduce drasticamente il carico P6-P8`

---

## Fase 6 — Analisi Escursioni

**Script:** `OPTIMIZATIONS/vps_excursion_analysis_vectorized.py`

**Input:** `snapshots.csv` + `sampled_events.csv`
**Output:** `excursion_stats.csv`

### Per Ogni Evento Campionato — 3 Orizzonti (30s/60s/120s)

| Metrica | Significato |
|---------|-------------|
| max_up_Xs_ticks | Quanto sale il prezzo nei Xs successivi |
| max_down_Xs_ticks | Quanto scende il prezzo nei Xs successivi |
| mae_Xs_ticks | Maximum Adverse Excursion (drawdown) |
| window_complete_Xs | Se Xs è finito con prezzo ritornato |
| horizon_end_ts_Xs | Timestamp di fine finestra |

### Algoritmo Single-Pass

```python
# Running max/min in un solo forward pass — O(n)
running_max = np.maximum.accumulate(prices)   # running max in O(n)
running_min = np.minimum.accumulate(prices)   # running min in O(n)

# Per ogni orizzonte: estrai metrica dai running accumulators — O(1)
mfe_30s = running_max[idx_30s] - ref_price
mfe_60s = running_max[idx_60s] - ref_price
mfe_120s = running_max[idx_120s] - ref_price
```

**Memory management:** Chunked processing (2M snapshot rows, 500K event rows per chunk) — peak ~1.5GB su 24GB VPS.

**Checkpoint:** `_checkpoints/p6_excursion.done`

`★ Insight: np.maximum.accumulate applica un running maximum in O(n) vettorizzato a livello C — il trick è che il running max alla posizione i è semplicemente il max di tutti i prezzi da start a i`

---

## Fase 7 — Triple-Barrier Labeling

**Script:** `OPTIMIZATIONS/vps_phase7_labeling.py`

**Input:** `excursion_stats.csv` + `snapshots.csv`
**Output:** `phase7_labels_*/` (directory per candidato)

### 3 Candidati Triple-Barrier (Tick Clock)

| Candidato | Vertical Barrier | Profit Target | Stop Loss |
|-----------|-----------------|---------------|-----------|
| C1        | 30 ticks        | 9.5 ticks    | 9.8 ticks |
| C2        | 60 ticks        | 20.0 ticks   | 20.0 ticks|
| C3        | 120 ticks       | 13.0 ticks   | 14.5 ticks|

### Labeling Rules

| Label | Significato | Condizione |
|-------|-------------|------------|
| +1 | Long | PT toccato **prima** di SL e prima della scadenza |
| -1 | Short | SL toccato **prima** di PT e prima della scadenza |
| 0 | Flat | Il tempo scade prima che PT o SL vengano toccati |

### First-Touch vs Max-Excursion

**P6 dice:** "max_up_30s = 12 ticks" — ma NON quando arriva.

**P7 scandisce** la sequenza cronologica reale:
```
Prezzo parte da 100
  tick 5:  raggiunge PT (100 + 9.5×0.25) → +1 label
  tick 8:  attraversa SL (99.75) → ignorato (gia label +1)
```

Solo il **primo** tocco determina la label.

### Engine

- **Numba JIT** parallelizzato su 8 core
- **Tick Clock** invece di tempo fisico (book update count)
- Scan parallelo su 500 eventi per batch

**Checkpoint:** `_checkpoints/p7_c1.done`, `p7_c2.done`, `p7_c3.done`

`★ Insight: Triple Barrier Method (López de Prado) trasforma labeling finanziario da problema aperto a classificazione supervised. First-touch è il cuore: solo il primo tocco conta, non il massimo toccato`

---

## Fase 7b — Filtro Macro (GEX / Beta-Surprise)

**Script:** `OPTIMIZATIONS/vps_phase7b_macro_filter.py`

**Input:** `sampled_events.csv` + macro data (GEX, VIX, Beta)
**Output:**
- `sampled_events_gex_pos.csv` → Mean Reversion regime
- `sampled_events_gex_neg.csv` → Momentum regime

### Cosa Filtra

**GEX Zero Crossing:** Transizioni vicino a GEX=0 — "zone infette" dove le dinamiche LOB si rompono.

**Beta Shock:** Variazioni estreme del beta (market sensitivity) — shocks esogeni.

### Perché Due Modelli

Regimi diversi → distribuzione ritorni diversa → pattern diversi → due modelli specializzati battono uno generico.

**Thresholds:**
- `GEX_EPSILON = 50M` — entro ±50M = territorio Zero
- `BETA_SHOCK_THRESH = 3.5` — z-score moltiplicativo

**Nota:** Usa mock data se CSV esterni non disponibili.

---

## Fase 8 — ML Entry Model

**Script:** `OPTIMIZATIONS/vps_phase8_entry_model.py`

**Input:** `sampled_events.csv` + `phase7_labels_*/`
**Output:**
- `model.pkl` — modello serializzato
- `phase8_trainval_results.csv` — metriche train/val/test
- `phase8_feature_importance.csv`
- `phase8_best_candidate.md`

### Features (~50)

Tutte da P3 + P4: imbalance, stack/pull, microprice, exhaustion, spread, rolling stats.

### Train/Val/Test Split (temporale)

```
Train: 00:00 — 06:00
Val:   06:00 — 08:00
Test:  08:00 — 09:30
```

**Metriche target:** `P_T` — Probabilità di Transazione Corretta (non accuracy standard).

### Modelli Testati (best available)

1. LightGBM
2. XGBoost
3. Random Forest
4. Logistic Regression (fallback)

**Checkpoint:** `_checkpoints/p8_ml.done`

`★ Insight: P_T cattura la vera utilità del modello per trading — accuracy 90% con stop sempre colpito = perdita. 52% direzione giusta con buona expectancy = profitto`

---

## Fase 11 — RL Trade Execution

**Script:** `OPTIMIZATIONS/vps_phase11_rl_execution.py`

**Tipo:** Live service (non batch)

### Actor-Critic con Logistic-Normal Policy

**Spazio Azione (K=6 azioni discrete):**

| Azione | Descrizione | Caratteristica |
|--------|-------------|----------------|
| a₀ | Market Order | Fill istantaneo, paga spread/slippage |
| a₁ | Limit Order +1 tick | Aggiunge liquidità |
| a₂ | Limit Order +2 tick | Più probabile fill, meno guadagno |
| a₃ | Limit Order +3 tick | — |
| a₄ | Limit Order +4 tick | — |
| a₅ | Limit Order +5 tick | — |
| a₆ | Hold | Nessun ordine, osserva |

### Logistic-Normal Distribution

Mappa `R^K → Simplex(K+1)` garantendo Σa = 1.0:

```python
z ~ N(μ, σ)                           # sampling con reparametrization trick
aₖ = exp(zₖ) / (1 + Σexp(zₖ))         # softmax projection
log P(a) = log N(z|μ, σ) − Σlog(aₖ)  # change of variables
```

**Queue Simulation:** Mock FIFO engine — simula posizione in coda, penalità cancellazione.

**Dependencies:** `torch`, `numpy`

---

## Fase 12 — Sierra Chart Live Bridge

**Script:** `OPTIMIZATIONS/vps_phase12_sierra_bridge.py`

**Tipo:** Live service (non batch)

### Server TCP Low-Latency

```
Python RL Agent → TCP:8089 → Sierra Chart ACSIL C++
```

### Configurazione Socket

```python
socket.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)   # DISABILITA Nagle
socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)   # Rebind rapido post-crash
```

**TCP_NODELAY** è critico: disabilita il buffering di Nagle per latenza sub-ms.

### Protocollo JSON

```python
{
    'action': 'allocate',
    'fractions': [0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.4],  # somma = 1.0
    'flatten': False
}
```

### Integrazione ACSIL

Chiama direttamente `sc.BuyEntry`, `sc.SellEntry`, `sc.Flatten` — **nessuno spreadsheet**, bypass completo per sub-millisecondo.

`★ Insight: TCP_NODELAY è fondamentale in HFT — Nagle ottimizza throughput TCP buffering multiple small packets, ma in trading la latenza è tutto. SO_REUSEADDR permette restart immediato post-crash`

---

## Orchestration

### PHASE_ORDER (ufficiale)

```
P1 → P2 → P2b → P3 → P4 → P5 → P6 → P7 (×3 candidati) → P8
```

### vps_multiday_runner.py

Coordina P1-P8 su tutti i giorni (4-6 worker paralleli).

```bash
# Full run
python3 vps_multiday_runner.py --workers 4 --force

# P1-P6 solo
python3 vps_multiday_runner.py --workers 4 --skip-p7-p8 --force

# Resume (skip già completati)
python3 vps_multiday_runner.py --workers 4 --resume
```

### incremental_p7p8_runner.py

Esegue P7+P8 per giorni con P1-P6 completo ma P7/P8 mancante. Checkpoint sentinel per idempotenza.

```bash
python3 incremental_p7p8_runner.py --output-dir /opt/depth-dom/output --workers 4
```

### cron_orchestrator.sh

Runs ogni 30 min via cron:
1. incremental_p7p8_runner
2. aggregate_results
3. plot_dashboard

---

## Checkpoint System

Ogni fase produce un sentinel file in `_checkpoints/`:

```
output/{date}/_checkpoints/
  p1_parse.done
  p2_reconstruct.done
  p3_features.done
  p4_agg.done
  p5_sample.done
  p6_excursion.done
  p7_c1.done / p7_c2.done / p7_c3.done
  p8_ml.done
```

**Formato sentinel:**
```
status=done
time=2026-04-04T18:30:00
error=   # opzionale, se failed
```

**Manifest:** `/opt/depth-dom/output/_multiday_manifest.csv` — tracking centralizzato per resume.

---

## Directory Structure su VPS

```
/opt/depth-dom/
├── vps_depth_parser.py              # P1
├── book_reconstructor.py            # P2
├── vps_feature_engineering_vectorized.py  # P3
├── vps_feature_engineering_agg.py   # P4
├── vps_cusum_sampler.py             # P5
├── vps_excursion_analysis_vectorized.py  # P6
├── vps_phase7_labeling.py           # P7
├── vps_phase7b_macro_filter.py      # P7b
├── vps_phase8_entry_model.py        # P8
├── vps_phase11_rl_execution.py      # P11
├── vps_phase12_sierra_bridge.py      # P12
├── vps_multiday_runner.py           # Orchestrator batch
├── incremental_p7p8_runner.py       # Orchestrator incremental
├── _pipeline_constants.py           # Costanti condivise
├── main.py                          # CLI single-day
├── input/NQ*/YYYY-MM-DD.depth       # Source data
└── output/
    ├── _multiday_manifest.csv
    ├── _p7p8_incremental_manifest.csv
    └── {YYYY-MM-DD}/
        ├── events.csv               # P1
        ├── snapshots.csv            # P2
        ├── features_dom.csv         # P3
        ├── features_dom_agg.csv     # P4
        ├── sampled_events.csv       # P5
        ├── excursion_stats.csv      # P6
        ├── phase7_labels_*/         # P7 (3 directory candidati)
        ├── _checkpoints/            # Sentinel files
        └── model.pkl                # P8
```

---

## Bug Noti (Stato)

| Bug | File | Stato | Prompt |
|-----|------|-------|--------|
| P3 depth_ratio = inf | vps_feature_engineering_vectorized.py | ✅ FIXED | — |
| P2 crossed book non rejected | vps_book_reconstructor.py | ✅ FIXED | — |
| P4 exhaustion_count O(n) | vps_feature_engineering_agg.py | ✅ FIXED | — |
| P7 binary search start_idx | vps_p7_global_runner.py | ✅ FIXED | — |
| P8 warm-start fake | vps_phase8_entry_model.py | ✅ FIXED (rinominato batch_retrain) | — |
| P4 ts format ms vs ISO | vps_feature_engineering_agg.py | ⚠️ Known (contract, non blocking) | — |
| P5 memory bulk load | vps_cusum_sampler.py | ⚠️ Known (alignment guard added) | — |
| P7 vertical barrier time-based | vps_phase7_labeling.py | ✅ FIXED (tick-based) | 1–8 |
| P2b offline/standalone | vps_phase2b_data_fusion.py | ✅ FIXED (integrato) | 1–8 |
| P8 old key names + local helpers | vps_phase8_entry_model.py | ✅ FIXED | 1–8 |

---

## Riferimenti

- Pipeline completo: `DEPTH-DOM-PIPELINE-DOCUMENTATION.md`
- Architettura HFT: `HFT_ARCHITECTURE_DOC.md`
- VPS access: `root@185.185.82.205` (primary), `root@96.30.209.74` (secondary)
- Log pipeline: `/opt/depth-dom/pipeline_full3.log`
- Log cron: `/opt/depth-dom/logs/cron_orchestrator.log`
