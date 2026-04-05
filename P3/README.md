# P3 — Feature Engineering Vettorializzata

**Script:** `vps_feature_engineering_vectorized.py`

## Cosa fa
Computa ~50 indicatori microstrutturali per ogni snapshot del book.

## Feature Principali

### Book Imbalance
```python
imbalance_1  = (bid_qty_1 − ask_qty_1) / (bid_qty_1 + ask_qty_1)
imbalance_5  = sum(bid_qty_1..5 − ask_qty_1..5) / sum(...)
```

### Stack/Pull (Δquantity per livello)
```python
stack_bid_1 = max(0, bid_qty_1(t) − bid_qty_1(t−1))  # aggiunta
pull_bid_1  = max(0, bid_qty_1(t−1) − bid_qty_1(t))   # rimozione
```

### Microprice
```python
microprice = (best_bid × ask_qty_1 + best_ask × bid_qty_1) / (bid_qty_1 + ask_qty_1)
```

### Bug Fixato
`depth_ratio = bid_depth_5 / ask_depth_5` → capped a 100.0 (evita `inf`)

## Input / Output
- **Input:** `snapshots.csv`
- **Output:** `features_dom.csv` (~310 MB/giorno)
- **Checkpoint:** `_checkpoints/p3_features.done`

## Tecnica
Fully vectorized pandas/numpy — nessun loop Python row-by-row. 10-50× speedup.
