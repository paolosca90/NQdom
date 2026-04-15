# SHARED — Costanti e Utilities Condivise

## _pipeline_constants.py

Costanti condivise da tutti i componenti del pipeline.

**3 Candidati Triple-Barrier (aggiornati Apr 2026):**
vb_ticks = vertical barrier in TICK-CLOCK updates (book update count, NOT seconds).

```python
CANDIDATES = [
    {"vb_ticks": 2000, "pt_ticks": 10.0, "sl_ticks": 10.0, "desc": "2000t/10/10"},
    {"vb_ticks": 4000, "pt_ticks": 20.0, "sl_ticks": 20.0, "desc": "4000t/20/20"},
    {"vb_ticks": 8000, "pt_ticks": 40.0, "sl_ticks": 40.0, "desc": "8000t/40/40"},
]
```

**Risultati attesi su dati Mar 13, 2026:**
| Candidato | Barrier hit | Balance | Win rate |
|-----------|-------------|---------|----------|
| 2000t/10/10 | 88.5% | 0.974 | 49.3% |
| 4000t/20/20 | 64.7% | 0.944 | 48.6% |
| 8000t/40/40 | 36.5% | 0.865 | 46.4% |

**Orario UTC di trading (DeepLOB, Apr 2026):** 13:40 — 19:50 UTC (09:40–15:50 ET, EDT=UTC-4)
**Nuovo:** esclude i primi/ultimi 10 minuti di asta per rimuovere rumore (DeepLOB paper).

**SPLIT temporali per P7/P8 (Multi-Day Walk-Forward ≥2 giorni):**
```
Train: 13:40:00.000000 — 18:00:00.000000  (09:40–14:00 ET, ~70%)
Val:   18:00:00.000000 — 18:55:00.000000  (14:00–14:55 ET, ~15%)
Test:  18:55:00.000000 — 19:50:00.000000  (14:55–15:50 ET, ~15%)
```
**Fallback Intra-Day** (se <2 giorni): split orario equivalenti.

**Features (~70):** LOB (imbalance, stack/pull, microprice) + P2b TS (delta_L/C/M, stacked_imbalance, volume_sequence, bid/ask_fade, exhaustion, unfinished_business, closing_delta_extreme, tick_zscore).

**Helpers:**
```python
label_filename(vb_ticks, pt_ticks, sl_ticks)   # → "phase7_labels_2000ticks_10p0_10p0"
label_filenames()                              # → lista di tutti i 3 nomi
parse_ts_to_ms(ts_str)                         # ISO ts → ms-from-midnight
```

## Note

Le versioni locali sono la **fonte di verità**.
CANDIDATES deve essere allineato con `_candidates_3.csv` in output/.
