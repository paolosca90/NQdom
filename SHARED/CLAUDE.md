# SHARED — Costanti e Utilities Condivise

<claude-mem-context>
# Recent Activity

### Apr 10, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3631 | 3:41 PM | 🔵 | Pipeline Constants Reveal Triple Barrier Label Candidates | ~275 |
| #3630 | 3:40 PM | 🔵 | P7 Grid File Location Discovered - Uses _candidates_3.csv | ~234 |

### Apr 11, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3837 | 3:43 PM | 🔵 | NQ trading pipeline project structure | ~302 |
</claude-mem-context>

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

**Risultati su dati Mar 13, 2026:**
| Candidato | Barrier hit | Balance | Win rate |
|-----------|-------------|---------|----------|
| 2000t/10/10 | 88.5% | 0.974 | 49.3% |
| 4000t/20/20 | 64.7% | 0.944 | 48.6% |
| 8000t/40/40 | 36.5% | 0.865 | 46.4% |

**SPLITS (usati in P7/P8) — NUOVI (Apr 2026, DeepLOB):**
```
train: "13:40:00.000000", "18:00:00.000000"   (09:40–14:00 ET, ~70%)
val:   "18:00:00.000000", "18:55:00.000000"   (14:00–14:55 ET, ~15%)
test:  "18:55:00.000000", "19:50:00.000000"   (14:55–15:50 ET, ~15%)
```

**Orario UTC di trading (DeepLOB, Apr 2026):** 13:40 — 19:50 UTC (09:40–15:50 ET, EDT=UTC-4)
**Nuovo:** esclude i primi/ultimi 10 minuti di asta per rimuovere rumore (DeepLOB paper).

**Costanti architettura DeepLOB:**
- `NEUTRAL_BAND_BPS = 0.0002` (±2 bps — flat quando |mid_change_5m| < this)
- `PT_THRESHOLD_DEEPLOB = 0.53` (53% OOS — soglia per deep nets CNN+LSTM)
- `EXEC_INTERVAL_MIN = 5` (posizioni ogni 5 minuti, 09:45–15:30 ET)
- `COST_BP_NQ = 8` (~$40 round-turn NQ)
- `COST_BP_ES = 2` (~$10 round-turn ES)

**Helpers:**
```python
label_filename(vb_ticks, pt_ticks, sl_ticks)   # → "phase7_labels_2000ticks_10p0_10p0"
label_filenames()                              # → lista di tutti i 3 nomi
parse_ts_to_ms(ts_str)                         # ISO ts → ms-from-midnight
```

## Note

Le versioni locali sono la **fonte di verità**.
CANDIDATES deve essere allineato con `_candidates_3.csv` in output/.
