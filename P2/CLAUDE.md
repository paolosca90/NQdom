# P2 — Order Book Reconstruction

<claude-mem-context>
# Recent Activity

### Apr 11, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #4035 | 9:11 PM | 🔵 | Numba-accelerated order book reconstruction | ~308 |

### Apr 13, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #4355 | 3:12 PM | 🔵 | P2 Module Structure: Order Book Reconstruction Functions | ~236 |
</claude-mem-context>

## Script

`vps_book_reconstructor.py`

## Key Logic

- Maintains 10 bid + 10 ask levels using heapq
- Events from events.csv applied in order
- Snapshot emitted every N events (not every event)
- Crossed book rejection: skip snapshot if best_bid >= best_ask (bug fixed)

## Output

snapshots.csv with columns: ts, best_bid, best_ask, spread, mid_price,
bid_px_1..10, bid_qty_1..10, ask_px_1..10, ask_qty_1..10
