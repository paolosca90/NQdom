# P8 — ML Entry Model

**Script:** `vps_phase8_entry_model.py`

## Cosa fa
Allena un classificatore per predire direzione (+1, -1, 0).

## Features
~50 da P3 + P4: imbalance, stack/pull, microprice, exhaustion, spread, rolling stats.

## Train/Val/Test Split (temporale)
```
Train: 00:00 — 06:00
Val:   06:00 — 08:00
Test:  08:00 — 09:30
```

## Modelli Testati
1. LightGBM (preferito se disponibile)
2. XGBoost
3. Random Forest
4. Logistic Regression (fallback)

## Metrica Target
`P_T` — Probabilità di Transazione Corretta (non accuracy standard).

## Output
- `model.pkl` — modello serializzato
- `phase8_trainval_results.csv`
- `phase8_feature_importance.csv`
- `phase8_best_candidate.md`
- `phase8_oof_predictions.csv`

## Input / Output
- **Input:** `sampled_events.csv` + `phase7_labels_*/`
- **Checkpoint:** `_checkpoints/p8_ml.done`

## Nota
`--warm-start` fa batch retrain (non true incremental learning).
