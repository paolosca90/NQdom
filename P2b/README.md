# P2b — Data Fusion (LOB + Time & Sales)

**Script:** `vps_phase2b_data_fusion.py`

## Cosa fa
Integra i trades (Time & Sales) sugli snapshot del book per distinguere
cancellazioni da market orders.

## Perché Serve
Quando `bid_qty_1` scende, non sappiamo se è:
- **ΔC** (Cancellation) — ordine cancellato
- **ΔM** (Market Order) — ordine eseguito

**Formula:** `ΔC = ΔV_total − M (trade volume)`

## Algoritmo
`pandas.merge_asof` con direzione **backward** — associa ad ogni snapshot
l'ultimo trade avvenuto nel passato (evita look-ahead bias).

## Input / Output
- **Input:** `snapshots.csv` + `trades.csv` (esterno)
- **Output:** `snapshots_fused.csv`
- **Nota:** Standalone — non nel batch runner principale.
