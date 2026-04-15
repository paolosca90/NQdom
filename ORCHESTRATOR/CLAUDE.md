# ORCHESTRATOR — Pipeline Coordination (LOCAL)

All orchestration runs **LOCAL** on this machine. No VPS required.

## Scripts

| Script | Tipo | Uso |
|--------|------|-----|
| `run_p1_to_p7_multiday.py` | Batch | P1-P7 multiday LOCAL (preferred) |
| `incremental_p7p8_runner.py` | Incremental | P7/P8 per nuovi giorni |
| `audit_p7_results.py` | Audit | P5-P7 quality + label distribution |
| `audit_pipeline.py` | Audit | Pipeline checkpoint status per tutti i giorni |
| `aggregate_results.py` | Aggregate | Cross-day P8 results aggregation |
| `plot_dashboard.py` | Viz | Dashboard plots |
| `status_live.py` | Status | Live status dashboard |
| `vps_multiday_runner.py` | Batch | Deprecated — usa `run_p1_to_p7_multiday.py` |

## P7b Deprecated

P7b (Macro Filter con GEX/Beta-Surprise) è stato rimosso dal pipeline.
P8 ora consuma direttamente `sampled_events.csv`.

## Audit Outputs

Audit results go to `NQdom/output/_audit/`:
- `label_distribution_daily.csv` — PT/SL/V% per day per candidate
- `label_distribution_aggregate.csv` — aggregate across all days
- `label_outlier_days.csv` — days with balance_ratio < 0.80
- `label_summary.txt`
- `quality_row_counts.csv` — row counts per phase
- `quality_nan_rates.csv` — NaN% per file
- `quality_anomalies.csv` — detected anomalies
- `quality_summary.txt`

## Key Commands

```bash
# Audit P5-P7
python3 NQdom/ORCHESTRATOR/audit_p7_results.py --output-dir NQdom/output

# P1-P7 multiday LOCAL
python3 NQdom/run_p1_to_p7_multiday.py --resume --workers 1

# Aggregate cross-day results
python3 NQdom/ORCHESTRATOR/aggregate_results.py \
    --output-dir NQdom/output \
    --agg-dir    NQdom/output/aggregate
```


<claude-mem-context>
# Recent Activity

### Apr 11, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3999 | 8:19 PM | 🔵 | VPS multi-day DOM pipeline orchestrator exists | ~308 |
| #3837 | 3:43 PM | 🔵 | NQ trading pipeline project structure | ~302 |
</claude-mem-context>