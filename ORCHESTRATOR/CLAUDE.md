# ORCHESTRATOR — Pipeline Coordination

<claude-mem-context>
# Recent Activity

### Apr 11, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3999 | 8:19 PM | 🔵 | VPS multi-day DOM pipeline orchestrator exists | ~308 |
| #3837 | 3:43 PM | 🔵 | NQ trading pipeline project structure | ~302 |
</claude-mem-context>

## Scripts

- `vps_multiday_runner.py` — main batch orchestrator (P1-P8, 4 workers)
- `incremental_p7p8_runner.py` — incremental P7/P8 (every 30 min)
- `run_p1_to_p7_multiday.py` — local P1-P7 multiday runner
- `audit_pipeline.py` — pipeline audit
- `status_live.py` — live status dashboard
- `aggregate_results.py` — aggregate per-day results
- `plot_dashboard.py` — dashboard plots
- `vps_watchdog.sh` — process watchdog

## P7b Deprecated

P7b (Macro Filter con GEX/Beta-Surprise) è stato rimosso dal pipeline.
Rimosso da: `vps_multiday_runner.py` (PHASE_NAMES, run_phase7b, main loop),
directory `P7b/`, e tutta la documentazione.
P8 ora consuma direttamente `sampled_events.csv`.

## Cron

```
*/30 * * * * /bin/bash /opt/depth-dom/cron_orchestrator.sh
```
