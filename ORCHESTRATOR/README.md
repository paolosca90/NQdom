# ORCHESTRATOR — Coordinamento Pipeline

## vps_multiday_runner.py

Coordina P1-P8 su tutti i giorni (4-6 worker paralleli).

```bash
python3 vps_multiday_runner.py --workers 4 --force       # full run
python3 vps_multiday_runner.py --workers 4 --skip-p7-p8  # P1-P6 only
python3 vps_multiday_runner.py --workers 4 --resume       # resume
```

## incremental_p7p8_runner.py

Esegue P7+P8 per giorni con P1-P6 completo ma P7/P8 mancante.
Checkpoint sentinel per idempotenza. Genera automaticamente `snapshots.csv`
da `events.csv` se mancante (gap del vecchio pipeline Dec 2025).

```bash
python3 incremental_p7p8_runner.py --output-dir /opt/depth-dom/output --workers 4
```

Chiamato ogni 30 min da `cron_orchestrator.sh`.

## aggregate_results.py

Aggrega i risultati per-day in summary CSV.

## plot_dashboard.py

Genera i plot del dashboard.

## cron_orchestrator.sh

Runs ogni 30 min (root level):

```
*/30 * * * * /bin/bash /opt/depth-dom/cron_orchestrator.sh
```

Steps: (1) incremental_p7p8_runner, (2) aggregate_results, (3) plot_dashboard.

## audit_pipeline.py

Audit del pipeline — verifica stato e consistenza dei dati.

## status_live.py

Status live del pipeline — leggibile da terminale.

## vps_watchdog.sh

Watchdog per monitorare processi in crash.
