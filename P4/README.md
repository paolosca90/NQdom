# P4 — Aggregazione Rolling Window

**Script:** `vps_feature_engineering_agg.py`

## Cosa fa
Aggiunge statistiche rolling window (1s, 5s, 30s) alle feature P3.

## Feature Rolling

| Finestra | Feature |
|----------|---------|
| 1s/5s/30s | imbalance_mean/std |
| 1s | stack/pull aggregati |
| 1s | queue_exhaustion_count |

## Bug Critico Risolto — O(n) → O(1)

```python
# PRIMA: O(n) per ogni chiamata (iterava tutto il deque)
def get(self):
    return sum(1 for i in range(n) if qty[i] <= THRESHOLD)

# DOPO: O(1) running counter
exhaustion_count += 1  # in update()
return exhaustion_count  # in get()
```

**Risultato: 70-80% speedup** — un giorno di trading = 390M+ eventi.

## Input / Output
- **Input:** `features_dom.csv`
- **Output:** `features_dom_agg.csv`
- **Checkpoint:** `_checkpoints/p4_agg.done`
