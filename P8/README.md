# P8 — ML Entry Model

**Script:** `vps_phase8_entry_model.py`

## Cosa fa
Allena un classificatore multiclass (short/flat/long) per predire direzione di trade.
Optimizza per **P_T** (Probability of Correct Transaction), non accuracy/F1 standard.

## Modello Target
vb=2000tick / pt=10 / sl=10 (C1 P7) — 88.5% barrier hit, balance=0.974
Questo candidato ha il miglior balance e il dataset più grande per training.

## UTC Trading Window
**Sessione analizzata:** 13:40:00 — 19:50:00 UTC (09:40–15:50 ET, EDT=UTC-4)
**Nuova architettura (Apr 2026):** esclude i primi/ultimi 10 minuti di asta per rimuovere rumore (DeepLOB paper).
**Split temporali:**
```
Train: 13:40:00.000 — 18:00:00.000  (09:40–14:00 ET, ~70%)
Val:   18:00:00.000 — 18:55:00.000  (14:00–14:55 ET, ~15%)
Test:  18:55:00.000 — 19:50:00.000  (14:55–15:50 ET, ~15%)
```
**Nota:** Se ≥2 giorni disponibili usa Walk-Forward Multi-Day (70% / 15% / 15%).
**Fallback Intra-Day** (se <2 giorni): split orario equivalenti.

## Features
~70 da P3 + P4 + P2b: imbalance, stack/pull, microprice, exhaustion, spread, rolling stats.
**Nuove TS (Time & Sales) features (P2b):** ΔL/ΔC/ΔM (liquidity/cancel/market), stacked imbalances (≥300%),
volume sequencing, bid/ask fade, closing delta extremes, unfinished business, TICK z-score.
Colonne leakage esplicite: label, max_up/down, barrier_hit, ref_price, future excursions.

## Train/Val/Test Split
**Multi-day Walk-Forward** (se ≥2 giorni disponibili):
- Train: primo 70% dei giorni
- Val: successivo 15%
- Test: ultimo 15%

**Fallback Intra-Day** (se <2 giorni): split orario equivalenti.

## Metrica Target
`P_T` — Probabilità di Transazione Corretta:
- PT = Potential Transactions (class occurrence in labels)
- TT = Predicted Transactions (class occurrence in predictions)
- CT = Correct Transactions (intersection)
- pT = (PT + TT - CT) / CT

**Obiettivo:** P_T < 1.0 out-of-sample (maggiore è CT a parità di PT+TT, migliore è il modello)

## Modelli Testati
1. LightGBM (preferito se disponibile)
2. XGBoost
3. Random Forest
4. Logistic Regression (fallback)
5. HistGradientBoosting

## Output
- `phase8_best_model.pkl` — modello + scaler + feature_cols + label_map
- `phase8_trainval_results.csv` — tutti i risultati per candidato/modello
- `phase8_feature_importance.csv` — top 50 features
- `phase8_best_candidate.md` — report dettagliato
- `phase8_oof_predictions.csv` — predizioni out-of-sample

## Input / Output
- **Input:** `sampled_events.csv` + `phase7_labels_2000ticks_10p0_10p0/`
- **Output dir:** `output/{date}/`
- **Checkpoint:** `_checkpoints/p8_ml.done`

## Note
`--warm-start` fa batch retrain (non true incremental learning).
SHARED/_pipeline_constants.py contiene i CANDIDATES che devono essere allineati con P7.
