# P3 — Feature Engineering

<claude-mem-context>
# Recent Activity

### Apr 11, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #4034 | 9:11 PM | 🔵 | Vectorized DOM feature engineering with pandas | ~345 |

### Apr 14, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #4671 | 9:01 PM | 🔴 | Missing temp file caused Phase 3 failure on 2026-04-07 | ~286 |
| #4670 | 9:00 PM | 🔴 | 2026-04-07 processing failed with exit code 1 | ~226 |
| #4668 | 8:50 PM | 🟣 | Batch processing pipeline resuming for remaining dates | ~251 |
| #4656 | 8:03 PM | 🔴 | Pandas CSV parsing error in feature engineering pipeline | ~280 |
| #4621 | 5:05 PM | 🟣 | New P3→P7 orchestrator with direct module imports for real-time output | ~386 |

### Apr 15, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #4730 | 7:58 AM | 🔵 | P3 vectorized feature engineering now running for 2026-03-16 fix | ~228 |
</claude-mem-context>

## Script

`vps_feature_engineering_vectorized.py`

## Key Features

- Book imbalance (level 1 and aggregated 5 levels)
- Stack/pull (quantity added/removed per level)
- Microprice: (best_bid × ask_qty + best_ask × bid_qty) / (bid_qty + ask_qty)
- Depth ratio (capped at 100.0 to avoid inf)
- ~50 features per snapshot

## Bug Fixed

depth_ratio capped at MAX_DEPTH_RATIO = 100.0 (evita inf)
