# P6 — Analisi Escursioni

**Script:** `vps_excursion_analysis_vectorized.py`

## Cosa fa
Per ogni evento campionato, calcola dove va il prezzo nei 30s/60s/120s successivi.

## Output per Evento (3 orizzonti)

| Metrica | Significato |
|---------|-------------|
| max_up_Xs_ticks | Quanto sale il prezzo nei Xs |
| max_down_Xs_ticks | Quanto scende nei Xs |
| mae_Xs_ticks | Maximum Adverse Excursion |
| window_complete_Xs | Se Xs finito con prezzo ritornato |

## Algoritmo Single-Pass

```python
# Running max/min in un solo forward pass — O(n)
running_max = np.maximum.accumulate(prices)
running_min = np.minimum.accumulate(prices)
```

## Memory Management
Chunked processing (2M snapshot rows, 500K event rows per chunk) — peak ~1.5GB su 24GB VPS.

## Input / Output
- **Input:** `snapshots.csv` + `sampled_events.csv`
- **Output:** `excursion_stats.csv`
- **Checkpoint:** `_checkpoints/p6_excursion.done`
