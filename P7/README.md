# P7 — Triple-Barrier Labeling

**Script:** `vps_phase7_labeling.py`

## Cosa fa
Assegna label direzionali (+1, -1, 0) basati sul Triple Barrier Method.

## 3 Candidati Triple-Barrier
vb_ticks = vertical barrier in TICK-CLOCK updates (NOT seconds).

| Candidato | Vertical Barrier | Profit Target | Stop Loss |
|-----------|-----------------|--------------|-----------|
| C1 | 30 ticks | 9.5 tick | 9.8 tick |
| C2 | 60 ticks | 20.0 tick | 20.0 tick |
| C3 | 120 ticks | 13.0 tick | 14.5 tick |

## Labeling Rules

| Label | Significato | Condizione |
|-------|-------------|------------|
| +1 | Long | PT toccato **prima** di SL e scadenza |
| -1 | Short | SL toccato **prima** di PT e scadenza |
| 0 | Flat | Tempo scade prima di PT/SL |

## First-Touch vs Max-Excursion
P6 dice "max_up = 12 ticks" ma NON quando. P7 scandisce la sequenza
cronologica e dice quale barriera viene toccata **per prima**.

## Engine
Numba JIT parallelizzato su 8 core. Scan parallelo su 500 eventi/batch.

## Input / Output
- **Input:** `excursion_stats.csv` + `snapshots.csv`
- **Output:** `phase7_labels_*/` (directory per candidato)
- **Checkpoint:** `_checkpoints/p7_c1.done`, `p7_c2.done`, `p7_c3.done`

## Note
Viene invocato da `incremental_p7p8_runner.py` per ogni candidato su ogni giorno.
