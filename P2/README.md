# P2 — Ricostruzione Order Book

**Script:** `vps_book_reconstructor.py`
**Variante:** `vps_book_reconstructor_fullnumba.py` (full Numba, più lento ma completo)

## Cosa fa
Ricostruisce lo stato completo del Limit Order Book applicando ogni evento di `events.csv`.
Mantiene 10 livelli bid + 10 ask con heapq ottimizzati.

## Algoritmo
1. Per ogni evento: aggiorna/aggiunge/cancella il livello corrispondente
2. Heap bid = max-heap, heap ask = min-heap
3. Ogni N eventi emette snapshot completo

## Ottimizzazioni
- `heapq.nlargest(10, bids)` → O(n log k) invece di `sorted()` O(n log n)
- Crossed book rejection: skip snapshot se `best_bid >= best_ask`

## Input / Output
- **Input:** `events.csv`
- **Output:** `snapshots.csv` (~427 MB/giorno)
- **Checkpoint:** `_checkpoints/p2_reconstruct.done`

## Output CSV (56 colonne)
```
ts, best_bid, best_ask, spread, mid_price,
bid_px_1..10, bid_qty_1..10,
ask_px_1..10, ask_qty_1..10
```

## Utilizzo
Di norma non viene chiamato direttamente — il runner orchestrator lo invoca automaticamente.
