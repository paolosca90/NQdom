# P3 — Feature Engineering

<claude-mem-context>
# Recent Activity

### Apr 11, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #4034 | 9:11 PM | 🔵 | Vectorized DOM feature engineering with pandas | ~345 |
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
