# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

DEPTH-DOM is a quantitative trading signal generation pipeline that processes Sierra Chart `.depth` binary order book files through 12 phases to produce ML-trained entry models for futures trading (NQ/ES).

- **VPS:** `root@185.185.82.205` — all heavy processing runs here at `/opt/depth-dom/`
- **Local:** This repo on your Mac at `/Users/paolo/Desktop/DEPTH DOM VPS/` — source of truth per ogni fase
- **Source data:** 131 `.depth` files in `/opt/depth-dom/input/NQ*/` on the VPS

---

## Directory Structure

```
NQdom/
├── CLAUDE.md                     ← questo file
├── PIPELINE_FASI_RIASSUNTO.md   ← riepilogo completo 12 fasi
│
├── P1/   depth_parser.py, main.py, README.md
├── P2/   vps_book_reconstructor.py, README.md
├── P2b/  vps_phase2b_data_fusion.py, compute_ts_features.py, README.md
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
│   ├── _pipeline_constants.py    ← costanti condivise (singola fonte di verità)
│   └── README.md
│
├── ORCHESTRATOR/
│   ├── vps_multiday_runner.py    ← batch orchestrator (VPS)
│   ├── incremental_p7p8_runner.py ← P7/P8 incremental runner
│   ├── run_p1_to_p7_multiday.py ← local P1-P7 multiday runner
│   ├── audit_pipeline.py
│   ├── status_live.py
│   ├── aggregate_results.py
│   ├── plot_dashboard.py
│   ├── vps_watchdog.sh
│   └── README.md
│
├── INPUT/                        ← .depth source files
├── INPUT_TS/                     ← Sierra Chart Time & Sales (.txt)
├── output/                        ← output per giorno
└── cron_orchestrator.sh
```

---

## Architecture

```
.depth (binario Sierra Chart)
  │
  ▼ P1 ──► events.csv
  │         │
  ▼ P2 ──► snapshots.csv
  │         │
  ▼ P2b ─► snapshots_fused.csv  (LOB + Time & Sales)
  │         │
  ▼ P3 ──► features_dom.csv
  │         │
  ▼ P4 ──► features_dom_agg.csv
  │         │
  ▼ P5 ──► sampled_events.csv
  │         │
  ▼ P6 ──► excursion_stats.csv
  │         │
  ▼ P7 ──► phase7_labels_*/      (3 candidati triple-barrier)
  │         │
  ▼ P8 ──► model.pkl + report
  │
  ▼ P11 ─► Agente RL Actor-Critic  (live service)
  │
  ▼ P12 ─► Bridge TCP ──► Sierra Chart  (live service)
```

**Time Filter:** P1 applica filtro UTC 13:40–19:50 (09:40–15:50 ET, EDT=UTC-4).
Tutti i file successivi ereditano la finestra.

---

## 3 Candidati Triple-Barrier (P7)

Definiti in `SHARED/_pipeline_constants.py` → `CANDIDATES`:

| Candidato | Vertical Barrier | Profit Target | Stop Loss | Note |
|-----------|-----------------|---------------|-----------|------|
| C1        | 2000 ticks      | 10.0 ticks    | 10.0 ticks| Scalping corto |
| C2        | 4000 ticks      | 20.0 ticks    | 20.0 ticks| Scalping medio |
| C3        | 8000 ticks      | 40.0 ticks    | 40.0 ticks| Intraday swing |

Vertical barrier unit: **TICK CLOCK** (book update count, NOT wall-clock seconds)
- 2000 ticks ≈ 88.5% di eventi tocca PT o SL
- Balance ratio ottimale: 0.974 (C1), 0.944 (C2), 0.865 (C3)

---

## Running the Pipeline

### Full Bulk Run (all available days) — on VPS

```bash
# P1-P8 on all days (4 workers, resume mode)
python3 /opt/depth-dom/vps_multiday_runner.py --workers 4 --cleanup-policy none --resume

# P1-P6 only (skip P7/P8)
python3 /opt/depth-dom/vps_multiday_runner.py --workers 4 --skip-p7-p8 --force

# Force re-run from start
python3 /opt/depth-dom/vps_multiday_runner.py --workers 4 --force
```

### Incremental P7/P8 Runner — on VPS (preferred after initial bulk)

```bash
python3 /opt/depth-dom/incremental_p7p8_runner.py --output-dir /opt/depth-dom/output --workers 4
```

Chiamato ogni 30 min dal cron job.

### Local Multiday P1-P7 (Mac/Windows)

```bash
# Target a specific day — use P1/main.py directly (NOT multiday runner --days, that doesn't exist)
python3 P1/main.py --days 2026-03-13 --force

# Multi-day sequential (RAM-safe, --workers 1)
python3 run_p1_to_p7_multiday.py --resume --workers 1

# Multiday runner limits to first N days from earliest date — use P1/main.py for single-day targeting
```

### Single Day Test — on VPS

```bash
python3 /opt/depth-dom/P1/main.py --days 2026-01-08 --force
```

### Cron Orchestrator

```
*/30 * * * * /bin/bash /opt/depth-dom/cron_orchestrator.sh
```

Steps: (1) incremental_p7p8_runner, (2) aggregate_results, (3) plot_dashboard.

---

## VPS — Data Flow Per Day

```
/opt/depth-dom/output/{YYYY-MM-DD}/
├── events.csv             (P1)
├── snapshots.csv          (P2) — ~427MB/day, deletable after P8
├── snapshots_fused.csv    (P2b) — fused LOB + Time & Sales
├── features_dom.csv       (P3) — ~310MB/day
├── features_dom_agg.csv   (P4)
├── sampled_events.csv     (P5) — kept permanently
├── excursion_stats.csv    (P6) — kept permanently
├── phase7_labels_*/       (P7) — directories, one per candidate
├── _checkpoints/
│   ├── p1_parse.done, p2_reconstruct.done, p2b_fusion.done, ...
│   └── p7_c1.done, p7_c2.done, p7_c3.done, p8_ml.done
└── model.pkl              (P8)
```

**Sentinel content:** `status=<state>`, `time=<iso>`, `error=<msg>` (non vuoti).

---

## Known Issues (Do Not Fix Without Coordination)

1. **P3 `depth_ratio = inf`** — **FIXED** in P3: capped at `MAX_DEPTH_RATIO = 100.0`.

2. **P2 crossed book not rejected** — **FIXED** in P2: `continue` skips snapshot when `best_bid >= best_ask`.

3. **P7b (Macro Filter) DEPRECATO** — Rimosso dal pipeline. P8 usa direttamente `sampled_events.csv`.

4. **P2b Sierra Chart CSV leading spaces** — **FIXED** in P2b: `pd.read_csv(skipinitialspace=True)` + column `.strip()` needed because Sierra exports columns with leading spaces (e.g. `" Time"`, `" Volume"`), causing format detection to fail.

5. **P5 CUSUM h threshold floor** — h must be at least `2 * tick_size = 0.5` to avoid over-emission. Adaptive calibration doubles h until emission rate ≤ 10%.

---

## VPS Access

```
Host: 185.185.82.205 (primary), 96.30.209.74 (secondary)
User: root
Auth: PreferredAuthentications=keyboard-interactive,password
```

Use `expect` or `sshpass` for non-interactive SSH from the Mac:
```bash
expect -c '
spawn ssh -o StrictHostKeyChecking=no root@185.185.82.205 {command}
expect "password:" { send "782789Pao!\r"; expect eof }
'
```

---

## Important File Locations

| What | Where |
|------|-------|
| Pipeline logs | `/opt/depth-dom/pipeline_full3.log` |
| P7/P8 run log | `/opt/depth-dom/logs/p7p8_run.log` |
| P7/P8 recovery log | `/opt/depth-dom/logs/p7p8_recovery_run3.log` |
| Cron log | `/opt/depth-dom/logs/cron_orchestrator.log` |
| Manifest | `/opt/depth-dom/output/_multiday_manifest.csv` |
| P7/P8 manifest | `/opt/depth-dom/output/_p7p8_incremental_manifest.csv` |
