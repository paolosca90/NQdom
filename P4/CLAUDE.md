# P4 — Rolling Window Aggregation

<claude-mem-context>
# Recent Activity

### Apr 10, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3606 | 3:19 PM | 🟣 | P4 temporal aggregation completed on full 8.4M-row dataset | ~337 |
| #3604 | 3:18 PM | 🔴 | P4 streaming CSV fix eliminates OOM on 8.4M-row dataset | ~396 |
</claude-mem-context>

## Script

`vps_feature_engineering_agg.py`

## Key Features

Rolling window aggregates: 1s, 5s, 30s
- imbalance_mean / imbalance_std per window
- stack/pull aggregated per 1s
- queue_exhaustion_count

## Bug Fixed

O(n) → O(1) per exhaustion counter: running increment instead of deque iteration.
70-80% speedup on 390M+ event days.
