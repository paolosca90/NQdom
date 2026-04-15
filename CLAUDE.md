# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

DEPTH-DOM is a quantitative trading signal generation pipeline that processes Sierra Chart `.depth` binary order book files through 12 phases to produce ML-trained entry models for futures trading (NQ/ES).

- **Local:** This repo at `C:\Users\Paolo\Desktop\NQ\NQdom\` — source of truth per ogni fase
- **Source data:** 131 `.depth` files in `NQdom/INPUT/NQ*/`
- **All pipeline execution runs LOCAL** — no VPS required

---

## Directory Structure

```
NQdom/
├── CLAUDE.md                     ← questo file
├── PIPELINE_FASI_RIASSUNTO.md   ← riepilogo completo 12 fasi
│
├── P1/   depth_parser.py, main.py
├── P2/   vps_book_reconstructor.py
├── P2b/  vps_phase2b_data_fusion.py, compute_ts_features.py
├── P3/   vps_feature_engineering_vectorized.py
├── P4/   vps_feature_engineering_agg.py
├── P5/   vps_cusum_sampler.py
├── P6/   vps_excursion_analysis_vectorized.py
├── P7/   vps_phase7_labeling.py
├── P8/   vps_phase8_entry_model.py
├── P11/  vps_phase11_rl_execution.py
├── P12/  vps_phase12_sierra_bridge.py
│
├── SHARED/
│   ├── _pipeline_constants.py    ← costanti condivise (singola fonte di verità)
│   └── README.md
│
├── ORCHESTRATOR/
│   ├── run_p1_to_p7_multiday.py ← P1-P7 multiday runner (LOCAL)
│   ├── incremental_p7p8_runner.py ← P7/P8 incremental runner
│   ├── audit_pipeline.py         ← pipeline audit
│   ├── audit_p7_results.py        ← P5-P7 data quality + label audit
│   ├── status_live.py
│   ├── aggregate_results.py
│   ├── plot_dashboard.py
│   └── README.md
│
├── INPUT/                        ← .depth source files
├── INPUT_TS/                     ← Sierra Chart Time & Sales (.txt)
└── output/                        ← output per giorno
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

## Running the Pipeline (LOCAL)

### P1-P7 Multiday (LOCAL)
```bash
# Multi-day sequential (RAM-safe)
python3 NQdom/run_p1_to_p7_multiday.py --resume --workers 1

# Force re-run from start
python3 NQdom/run_p1_to_p7_multiday.py --force --workers 1

# Target a specific day
python3 NQdom/P1/main.py --days 2026-03-13 --force
```

### Single Phase
```bash
python3 NQdom/P1/main.py --days 2026-04-10 --force
```

### P8 ML Model Training
```bash
python3 NQdom/P8/vps_phase8_entry_model.py \
    --features NQdom/output/2026-03-13/sampled_events.csv \
    --output   NQdom/output/2026-03-13/
```

### P5-P7 Data Audit
```bash
python3 NQdom/ORCHESTRATOR/audit_p7_results.py --output-dir NQdom/output
# Output: NQdom/output/_audit/
```

### Aggregate Cross-Day Results
```bash
python3 NQdom/ORCHESTRATOR/aggregate_results.py \
    --output-dir NQdom/output \
    --agg-dir    NQdom/output/aggregate
```

---

## Data Flow Per Day

```
NQdom/output/{YYYY-MM-DD}/
├── events.csv             (P1)
├── snapshots.csv          (P2) — ~427MB/day, deletable after P8
├── snapshots_fused.csv    (P2b) — fused LOB + Time & Sales
├── features_dom.csv       (P3) — ~310MB/day
├── features_dom_agg.csv   (P4)
├── sampled_events.csv     (P5) — kept permanently
├── excursion_stats.csv   (P6) — kept permanently
├── phase7_labels_*/      (P7) — directories, one per candidate
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

6. **P6 checkpoint staleness cascade bug (FIXED Apr 13, 2026)** — P6 stored checkpoint before P5 was fixed, causing 4 corrupt days (03-16, 03-17, 03-18, 03-27). **FIXED**: fingerprint-based staleness guard implemented in `P6/vps_excursion_analysis_vectorized.py` + `run_p1_to_p7_multiday.py`. All 20 days re-validated (1.00x ratio, fingerprint saved). Do NOT use `--force` on all days; it re-runs unnecessary phases.

7. **P7 ZeroDivisionError on fast days (FIXED Apr 13, 2026)** — `label_candidate()` crashed when `elapsed = 0.0` (few snapshots, fast processing). **FIXED**: guard added at line 214: `rate = n_valid / elapsed if elapsed > 0 else n_valid`.

8. **P2 checkpoint staleness (2026-03-17)** — `snapshots.csv` had only 8,459 rows while `events.csv` had 11.5M rows. P2 checkpoint was from partial run. Required P2→P3→P4→P5→P6→P7 full re-run for that day. **FIXED**: `events.csv` now grows correctly, all downstream phases re-run.

---

## Audit Scripts

### P5-P7 Data Quality + Label Audit (B1+B2)
```bash
python3 NQdom/ORCHESTRATOR/audit_p7_results.py --output-dir NQdom/output
```
Produces:
- `label_distribution_daily.csv` — per-day PT%/SL%/V%/balance_ratio
- `label_distribution_aggregate.csv` — aggregate across 20 days
- `label_outlier_days.csv` — days with balance_ratio < 0.80
- `label_summary.txt`
- `quality_row_counts.csv` — row counts per phase per day
- `quality_nan_rates.csv` — NaN% per file
- `quality_duplicates.csv` — duplicate timestamps
- `quality_anomalies.csv` — outlier days
- `quality_summary.txt`

### Pipeline Status Audit
```bash
python3 NQdom/ORCHESTRATOR/audit_pipeline.py
```
