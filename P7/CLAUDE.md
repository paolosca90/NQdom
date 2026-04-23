<claude-mem-context>
# Recent Activity

### Apr 10, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3649 | 3:53 PM | 🔴 | ASCII fix for Windows console compatibility | ~242 |
| #3644 | 3:48 PM | 🟣 | Phase 7 Labeling Re-Launched After Unicode Fix | ~135 |
| #3643 | " | 🔴 | P7 Script All Unicode Arrows Now Replaced | ~168 |
| #3642 | " | 🔴 | P7 Metrics Print Unicode Fixed | ~95 |
| #3641 | " | 🔴 | P7 Runtime Print Unicode Fix Applied | ~107 |
| #3640 | 3:47 PM | 🔴 | More Unicode Arrows Found in P7 Runtime Print Statements | ~173 |
| #3639 | " | 🔄 | P7 Script Docstring Usage Example Fixed | ~115 |
| #3638 | " | 🔴 | P7 Script Unicode Arrows Replaced with ASCII | ~189 |
| #3630 | 3:40 PM | 🔵 | P7 Grid File Location Discovered - Uses _candidates_3.csv | ~234 |

### Apr 14, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #4621 | 5:05 PM | 🟣 | New P3→P7 orchestrator with direct module imports for real-time output | ~386 |
| #4616 | 4:52 PM | 🟣 | Standalone P5-P7 orchestrator created with checkpoint resume | ~266 |

### Apr 20, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #5994 | 10:11 PM | 🔵 | Original p8_5 evaluator blocked — p8_ml has not run for any date | ~424 |

### Apr 21, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #6003 | 9:38 AM | 🟣 | P7 labeling now runs 21 candidates in 17 seconds after fix | ~236 |
| #6002 | " | 🔴 | P7 labeling script now handles snapshots_fused.csv column name mismatch | ~219 |
| #5997 | 9:35 AM | 🔵 | P7 labeling test failed - snapshots.csv path not found | ~212 |
</claude-mem-context>

# P7 — Triple-Barrier Labeling

**Script:** `vps_phase7_labeling.py`

## Cosa fa
Assegna label direzionali (+1, -1, 0) basati sul Triple Barrier Method con **First-Touch** logic.

## 3 Candidati Triple-Barrier (aggiornati Apr 2026)
vb_ticks = vertical barrier in TICK-CLOCK updates (book update count, NOT seconds).

| Candidato | Vertical Barrier | Profit Target | Stop Loss | Risultato |
|-----------|-----------------|--------------|-----------|-----------|
| C1 | 2000 ticks | 10.0 tick | 10.0 tick | 88.5% barrier hit, bal=0.974 |
| C2 | 4000 ticks | 20.0 tick | 20.0 tick | 64.7% barrier hit, bal=0.944 |
| C3 | 8000 ticks | 40.0 tick | 40.0 tick | 36.5% barrier hit, bal=0.865 |

**NOTA:** I vecchi parametri (30/60/120 VB con PT/SL 9.5/20/13) sono stati scartati perché
il 99.5-100% degli eventi scadeva verticalmente. Su NQ (ATR ~264 punti/giorno) servono VB più lunghi.

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
Numba JIT parallelizzato su 8 core.

## Input / Output
- **Input:** `excursion_stats.csv` + `snapshots.csv` + `sampled_events.csv`
- **Output:** `phase7_labels_*/` (directory per candidato)
- **Grid file:** `output/_candidates_3.csv`
- **Leaderboard:** `output/{date}/phase7_labeling_leaderboard.csv`

## Note
Per rigenerare con nuovi parametri: aggiornare `_candidates_3.csv` + CANDIDATES in SHARED,
poi rilanciare con `--force`.
