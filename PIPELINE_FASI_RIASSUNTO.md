# DEPTH-DOM Pipeline — Riepilogo Fasi

**Data:** 2026-04-13
**Pipeline:** 12 fasi per la generazione di segnali di trading ML-ready
**Input:** File `.depth` binari Sierra Chart (NQ futures)
**Output:** Modelli ML + bridge live per esecuzione in Sierra Chart
**Esecuzione:** LOCAL — nessun VPS richiesto

---

## Struttura Directory

```
NQdom/
│
├── CLAUDE.md                     ← istruzioni progetto per Claude Code
├── PIPELINE_FASI_RIASSUNTO.md   ← questo file
├── run_p1_to_p7_multiday.py    ← orchestrator P1→P7 (con P2b)
│
├── P1/   depth_parser.py, main.py, README.md
├── P2/   vps_book_reconstructor.py, README.md
├── P2b/  vps_phase2b_data_fusion.py, compute_ts_features.py,
│         split_sierra_trades_by_day.py, README.md
├── P3/   vps_feature_engineering_vectorized.py, README.md
├── P4/   vps_feature_engineering_agg.py, README.md
├── P5/   vps_cusum_sampler.py, README.md
├── P6/   vps_excursion_analysis_vectorized.py, README.md
├── P7/   vps_phase7_labeling.py, README.md
├── P8/   vps_phase8_entry_model.py, README.md
├── P11/  vps_phase11_rl_execution.py, README.md
├── P12/  vps_phase12_sierra_bridge.py, README.md
│
├── SHARED/
│   ├── _pipeline_constants.py     ← costanti condivise
│   └── README.md
│
├── ORCHESTRATOR/
│   ├── run_p1_to_p7_multiday.py   ← batch orchestrator P1-P7 (LOCAL)
│   ├── incremental_p7p8_runner.py ← runner P7/P8 incrementale (LOCAL)
│   ├── aggregate_results.py        ← aggregazione risultati (LOCAL)
│   ├── plot_dashboard.py           ← dashboard plotting
│   ├── audit_pipeline.py           ← audit pipeline
│   ├── audit_p7_results.py         ← P5-P7 data quality audit
│   └── status_live.py              ← status live
│
├── INPUT/                        ← .depth source files
├── INPUT_TS/                     ← Sierra Chart Time & Sales
└── output/                        ← output per giorno
```

---

## Panoramica del Pipeline

```
.depth (binario Sierra Chart)
  │
  ▼ P1 ──► events.csv                    (~45 MB/giorno)
  │         │
  ▼ P2 ──► snapshots.csv                 (~427 MB/giorno)
  │         │
  ▼ P2b ─► snapshots_fused.csv         (LOB + Time & Sales fusion)
  │         │
  ▼ P3 ──► features_dom.csv              (~310 MB/giorno)
  │         │
  ▼ P4 ──► features_dom_agg.csv          (~50 MB/giorno)
  │         │
  ▼ P5 ──► sampled_events.csv            (~31 MB/giorno, ~1% eventi)
  │         │
  ▼ P6 ──► excursion_stats.csv          (~15 MB/giorno)
  │         │
  ▼ P7 ──► phase7_labels_*/              (3 candidati triple-barrier)
  │         │
  ▼ P8 ──► model.pkl + report            (ML entry model)
  │
  ▼ P11 ─► Agente RL Actor-Critic        (live service)
  │
  ▼ P12 ─► Bridge TCP ──► Sierra Chart   (live service)
```

---

## Architettura DeepLOB (Apr 2026)

Riferimento: NotebookLM "Deep Limit Order Book Forecasting and Market Microstructure Analysis" (288 sources, aggiornato Apr 10, 2026)

### Orari di Trading

| Parametro | Valore | Note |
|-----------|--------|------|
| RTH (ET) | 09:30–16:00 | Regular Trading Hours |
| **Finestra filtrata (ET)** | **09:40–15:50** | Esclude asta apertura/chiusura (±10 min) |
| UTC (EDT=UTC-4) | 13:40–19:50 | Corretto per fuso marzo-aprile |

### Cadenza di Esecuzione

| Parametro | Valore |
|-----------|--------|
| Apertura posizioni | Ogni 5 minuti |
| Prima finestra entry | 09:45 ET |
| Ultima finestra entry | 15:30 ET |
| Inizio uscita forzata | 15:55 ET |
| Chiusura completa | 16:00 ET |

### Neutral Band

- **±2 basis points** attorno a 0 sulla finestra di averaging a 5 minuti
- Definisce la soglia tra "flat" e direzione (up/down)

### Costi di Transazione

| Contract | Costo round-turn |
|----------|-----------------|
| NQ | ~8 tick / ~$40 |
| ES | ~2 tick / ~$25 |

### Soglia PT per Deep Neural Nets

- **PT OOS ≥ 53%** richiesto prima di escalation a CNN+LSTM (UCL DeepLOB paper)
- Baseline: XGBoost/LightGBM deve superare questa soglia su test set

---

## Fase 1 — Parsing Binario

**Script:** `P1/depth_parser.py`

**Input:** `NQdom/INPUT/NQ*/YYYY-MM-DD.depth` (formato Sierra Chart proprietario)
**Output:** `NQdom/output/{date}/events.csv`

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

**Script:** `P2/vps_book_reconstructor.py`

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

**Script:** `P2b/vps_phase2b_data_fusion.py`

**Input:** `snapshots.csv` + `trades.csv` (esterno, es. Sierra Chart T&S)
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

**Ricerca automatica trades:** lo script cerca `trades.txt` o `trades.csv` nella stessa directory del `.depth` file. Se non presente, P2b skippa con warning.

**BUG FIX (Apr 12, 2026):** Sierra Chart CSV export contiene leading spaces nei nomi colonna (`" Time"`, `" Volume"`, ecc.). Fix: `pd.read_csv(skipinitialspace=True)` + `df.columns = [c.lower().strip() for c in df.columns]`. Senza questo fix, il formato non viene riconosciuto.

**Checkpoint:** `_checkpoints/p2b_fusion.done`

---

## Fase 3 — Feature Engineering Vettorializzata

**Script:** `P3/vps_feature_engineering_vectorized.py`

**Input:** `snapshots.csv` o `snapshots_fused.csv` (se P2b completato)
**Output:** `features_dom.csv` (~310 MB/giorno)

### Feature LOB (~50 colonne)

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

### Nuove Feature TS (Time & Sales) — P2b

Dopo la fusione con T&S, le seguenti feature vengono calcolate:

| Feature | Descrizione |
|---------|-------------|
| `delta_L`, `delta_L_1s`, `delta_L_5s` | Δ new passive limit liquidity |
| `delta_C`, `delta_C_1s`, `delta_C_5s` | Δ cancellations/spoofing (ΔV_total − M) |
| `delta_M`, `delta_M_1s`, `delta_M_5s` | Δ aggressive market orders from T&S |
| `stacked_imbalance_bid_3/5` | Institutional aggression bid across 3+ levels (≥300%) |
| `stacked_imbalance_ask_3/5` | Institutional aggression ask across 3+ levels (≥300%) |
| `volume_sequence_bid/ask` | Strictly increasing T&S volume across price levels |
| `bid_fade_3`, `ask_fade_3` | Volume diminishes across top/bottom 3 levels at extreme |
| `exhaustion_bid`, `exhaustion_ask` | Zero volume at bar high/low (red/green candle) |
| `unfinished_business_bid/ask` | Non-zero volume at new high/low (auction returns) |
| `closing_delta_extreme_bid/ask` | Closing delta ≥95% of max/min delta (momentum) |
| `tick_zscore` | NYSE TICK index Z-score (±2.5 reversal zones) |

**Tecnica:** Fully vectorized pandas/numpy — nessun loop Python row-by-row.

**Bug fixato:** `depth_ratio = bid_depth_5 / ask_depth_5` poteva produrre `inf` → capped a 100.0.

**Checkpoint:** `_checkpoints/p3_features.done`

`★ Insight: vectorized pandas/numpy processa milioni di righe/secondo vs migliaia in Python row-by-row — 10-50× speedup`

---

## Fase 4 — Aggregazione Rolling Window

**Script:** `P4/vps_feature_engineering_agg.py`

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

`★ Insight: exhaustion_count = eventi dove il book si svuota (qty ≤ soglia). Predice liquidazione/espressione — running counter O(1) invece di O(n) è la differenza tra 390M iterazioni e 390M letture`

---

## Fase 5 — CUSUM Sampling

**Script:** `P5/vps_cusum_sampler.py`

**Input:** `features_dom.csv` + `features_dom_agg.csv`
**Output:** `sampled_events.csv` (~1% degli eventi originali, ~31 MB/giorno)

### Algoritmo CUSUM (Adaptive, per-day)

```
Pass 1: Calcola h = 5° percentile(|Δmid_price|) non nulli (floor=0.5 tick)
Pass 1b: valida emissione su sample 200k righe — se rate > 10%, raddoppia h iterativamente
Pass 2: Accumula Δmid_price in CUSUM S
         Quando |S| > h → emetti evento → resetta S
```

**Risultato:** ~10K-50K eventi campionati da ~5-20M per sessione.
**Importante:** h viene calcolato per OGNI giorno dalla distribuzione dei prezzi di quel giorno — soglie diverse per day.

**Ottimizzazioni:**
- `np.percentile()` invece di `sort()` → O(n) vs O(n log n)
- `list[list[str]]` invece di `list[dict]` → elimina overhead dizionari

**Checkpoint:** `_checkpoints/p5_sample.done`

`★ Insight: CUSUM è un filtro passa-alto statistico — tiene solo "regime shifts" persistenti, filtra il rumore browniano. Campionare all'1% riduce drasticamente il carico P6-P8`

---

## Fase 6 — Analisi Escursioni

**Script:** `P6/vps_excursion_analysis_vectorized.py`

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

**Memory management:** Chunked processing (2M snapshot rows, 500K event rows per chunk) — peak ~1.5GB.

**Checkpoint:** `_checkpoints/p6_excursion.done`

`★ Insight: np.maximum.accumulate applica un running maximum in O(n) vettorizzato a livello C — il running max alla posizione i è semplicemente il max di tutti i prezzi da start a i`

---

## Fase 7 — Triple-Barrier Labeling

**Script:** `P7/vps_phase7_labeling.py`

**Input:** `excursion_stats.csv` + `snapshots.csv`
**Output:** `phase7_labels_*/` (directory per candidato)

### 3 Candidati Triple-Barrier (Tick Clock) — Aggiornati Apr 2026

| Candidato | Vertical Barrier | Profit Target | Stop Loss | Note |
|-----------|-----------------|---------------|-----------|------|
| C1 | **2000 ticks** | **10.0 ticks** | **10.0 ticks** | Scalping corto — 88.5% barrier hit |
| C2 | **4000 ticks** | **20.0 ticks** | **20.0 ticks** | Scalping medio — 64.7% barrier hit |
| C3 | **8000 ticks** | **40.0 ticks** | **40.0 ticks** | Intraday swing — 36.5% barrier hit |

**Risultati su dati Mar 13, 2026:**

| Candidato | Barrier hit | Balance ratio | Win rate |
|-----------|-------------|---------------|----------|
| 2000t/10/10 | 88.5% | 0.974 | 49.3% |
| 4000t/20/20 | 64.7% | 0.944 | 48.6% |
| 8000t/40/40 | 36.5% | 0.865 | 46.4% |

**Unità vertical barrier: TICK CLOCK** (book update count, NOT wall-clock seconds)

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
  tick 8:  attraversa SL (99.75) → ignorato (già label +1)
```

Solo il **primo tocco** determina la label.

### Engine

- **Numba JIT** parallelizzato su 8 core
- **Tick Clock** invece di tempo fisico (book update count)
- Scan parallelo su 500 eventi per batch
- Memory-safe micro-chunking (500K rows per batch)
- Float64→float32 downcasting

**Checkpoint:** `_checkpoints/p7_c1.done`, `p7_c2.done`, `p7_c3.done`

`★ Insight: Triple Barrier Method (López de Prado) trasforma labeling finanziario da problema aperto a classificazione supervised. First-touch è il cuore: solo il primo tocco conta, non il massimo toccato`

---
**Nota:** P7b (Filtro Macro GEX/Beta) è stato deprecato e rimosso.

---

## Fase 8 — ML Entry Model

**Script:** `P8/vps_phase8_entry_model.py`

**Input:** `sampled_events.csv` + `phase7_labels_*/`
**Output:**
- `model.pkl` — modello serializzato
- `phase8_trainval_results.csv` — metriche train/val/test
- `phase8_feature_importance.csv`
- `phase8_best_candidate.md`
- `phase8_oof_predictions.csv`

### Features (~70)

Tutte da P3 + P4 + **P2b TS features**:
- LOB: imbalance, stack/pull, microprice, spread, depth
- Rolling: rolling stats (1s, 5s, 30s windows)
- **TS (Time & Sales):** delta_L/C/M, stacked imbalances, volume sequencing, bid/ask fade, exhaustion, unfinished business, closing delta extremes, tick_zscore

### Train/Val/Test Split

**Multi-day Walk-Forward** (≥2 giorni):
- Train: primo 70% dei giorni
- Val: successivo 15%
- Test: ultimo 15%

**Fallback Intra-Day** (1 giorno solo):
```
Train: 13:40:00 — 18:00:00 UTC  (09:40–14:00 ET)
Val:   18:00:00 — 18:55:00 UTC  (14:00–14:55 ET)
Test:  18:55:00 — 19:50:00 UTC  (14:55–15:50 ET)
```

### Metrica Target

**P_T — Probabilità di Transazione Corretta** (non accuracy standard):
```
P_T = (PT + TT - CT) / CT
```
- PT = Potential Transactions (class occurrence in labels)
- TT = Predicted Transactions (class occurrence in predictions)
- CT = Correct Transactions (intersection)
- Obiettivo: P_T < 1.0 out-of-sample (maggiore è CT a parità di PT+TT, migliore)

**PT_THRESHOLD_DEEPLOB = 0.53** — soglia OOS per escalation a CNN+LSTM (UCL DeepLOB)

### Modelli Testati (best available)

1. LightGBM (preferito)
2. XGBoost
3. Random Forest
4. Logistic Regression (fallback)
5. HistGradientBoosting

**Checkpoint:** `_checkpoints/p8_ml.done`

`★ Insight: P_T cattura la vera utilità del modello per trading — accuracy 90% con stop sempre colpito = perdita. 52% direzione giusta con buona expectancy = profitto. Soglia 53% OOS validata da UCL DeepLOB come prerequisite per deep nets.`

---

## Fase 11 — RL Trade Execution

**Script:** `P11/vps_phase11_rl_execution.py`

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

---

## Fase 12 — Sierra Chart Live Bridge

**Script:** `P12/vps_phase12_sierra_bridge.py`

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

### run_p1_to_p7_multiday.py (LOCAL)

Orchestrator dedicato per P1→P7 (con P2b incluso). Cerca automaticamente `trades.txt/csv` accanto al `.depth` file per ogni giorno. Se trades assente per un giorno, P2b skippa con warning e il pipeline continua.

```bash
# Dry run
python3 NQdom/run_p1_to_p7_multiday.py --dry-run

# Resume (skip giorni con P7 già completo)
python3 NQdom/run_p1_to_p7_multiday.py --resume --workers 1

# Forza re-run completo
python3 NQdom/run_p1_to_p7_multiday.py --force --max-days 3
```

### incremental_p7p8_runner.py

Esegue P7+P8 per giorni con P1-P6 completo ma P7/P8 mancante.

```bash
python3 NQdom/incremental_p7p8_runner.py --output-dir NQdom/output --workers 4
```

---

## Checkpoint System

Ogni fase produce un sentinel file in `_checkpoints/`:

```
NQdom/output/{date}/_checkpoints/
  p1_parse.done
  p2_reconstruct.done
  p2b_fusion.done        ← NUOVO (Apr 2026)
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
time=2026-04-10T18:30:00
error=   # opzionale, se failed
```

**Manifest:** `NQdom/output/_p1p7_manifest.csv`

---

## Directory Structure Locale

```
NQdom/
├── INPUT/                        ← .depth source files
├── INPUT_TS/                     ← Sierra Chart Time & Sales (.txt)
│   └── by_day/                  ← split trades per day
└── output/
    ├── _p1p7_manifest.csv
    └── {YYYY-MM-DD}/
        ├── events.csv                  # P1
        ├── snapshots.csv               # P2
        ├── snapshots_fused.csv         # P2b (se T&S disponibile)
        ├── features_dom.csv            # P3
        ├── features_dom_agg.csv        # P4
        ├── sampled_events.csv          # P5
        ├── excursion_stats.csv         # P6
        ├── phase7_labels_*/           # P7 (3 directory candidati)
        │   ├── phase7_labels_2000ticks_10p0_10p0/
        │   ├── phase7_labels_4000ticks_20p0_20p0/
        │   └── phase7_labels_8000ticks_40p0_40p0/
        ├── _checkpoints/               # Sentinel files
        └── model.pkl                   # P8
```

---

## SHARED/_pipeline_constants.py — Costanti Aggiornate Apr 2026

```python
# 3 Candidati Triple-Barrier
CANDIDATES = [
    {"vb_ticks": 2000, "pt_ticks": 10.0, "sl_ticks": 10.0, "desc": "2000t/10/10"},
    {"vb_ticks": 4000, "pt_ticks": 20.0, "sl_ticks": 20.0, "desc": "4000t/20/20"},
    {"vb_ticks": 8000, "pt_ticks": 40.0, "sl_ticks": 40.0, "desc": "8000t/40/40"},
]

# Trading Hours (EDT = UTC-4, marzo-novembre)
TRADING_START_ET = "09:40"
TRADING_END_ET = "15:50"
TRADING_START_UTC = "13:40"
TRADING_END_UTC = "19:50"

# Execution Cadence
EXEC_INTERVAL_MIN = 5
EXEC_START_ET = "09:45"
EXEC_END_ET = "15:30"
EXIT_START_ET = "15:55"
EXIT_END_ET = "16:00"

# DeepLOB Architecture
NEUTRAL_BAND_BPS = 0.0002   # ±2 bps
PT_THRESHOLD_DEEPLOB = 0.53  # 53% OOS per deep nets
COST_BP_NQ = 8               # ~$40 round-turn NQ
COST_BP_ES = 2               # ~$10 round-turn ES
```

**ALL_FEATURE_PATTERNS:** ~70 feature columns da P3 + P4 + **P2b TS features** (delta_L/C/M, stacked_imbalance, volume_sequence, bid_fade, exhaustion, unfinished_business, closing_delta_extreme, tick_zscore)

---

## Bug Noti (Stato)

| Bug | File | Stato |
|-----|------|-------|
| P3 depth_ratio = inf | vps_feature_engineering_vectorized.py | ✅ FIXED |
| P2 crossed book non rejected | vps_book_reconstructor.py | ✅ FIXED |
| P4 exhaustion_count O(n) | vps_feature_engineering_agg.py | ✅ FIXED |
| P7 binary search start_idx | vps_p7_global_runner.py | ✅ FIXED |
| P8 warm-start fake | vps_phase8_entry_model.py | ✅ FIXED (batch_retrain) |
| P7 vertical barrier time-based | vps_phase7_labeling.py | ✅ FIXED (tick-based) |
| P2b offline/standalone | vps_phase2b_data_fusion.py | ✅ FIXED (integrato) |
| P8 old key names + local helpers | vps_phase8_entry_model.py | ✅ FIXED |
| P7 barrier params too tight (99%+ vertical expiry) | vps_phase7_labeling.py | ✅ FIXED (→ 2000/4000/8000 ticks) |
| P7/P8 Unicode arrows in output | vps_phase7_labeling.py | ✅ FIXED (ASCII per Windows) |
| P4 streaming CSV O(n²) | vps_feature_engineering_agg.py | ✅ FIXED |
| P5 CUSUM over-emission (96% rate) | vps_cusum_sampler.py | ✅ FIXED (adaptive calibration, 10% cap) |
| P2b Sierra CSV leading spaces in column names | vps_phase2b_data_fusion.py | ✅ FIXED (skipinitialspace + strip) |

---

## Riferimenti

- NotebookLM: "Deep Limit Order Book Forecasting and Market Microstructure Analysis" (ID: 077fed0a, 288 sources)
- Pipeline completo: `DEPTH-DOM-PIPELINE-DOCUMENTATION.md`
