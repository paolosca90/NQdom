"""
_pipeline_constants.py — Single source of truth for shared pipeline constants.
Imported by: local_multiday_runner.py, vps_multiday_runner.py,
             vps_multiday_aggregator.py, phase8_entry_model.py
"""

# ── Barrier Label Candidates ──────────────────────────────────────────────────
# vb_ticks = vertical barrier in TICK-CLOCK updates (NOT seconds)
# pt_ticks = profit target in ticks
# sl_ticks = stop loss in ticks
CANDIDATES = [
    {"vb_ticks": 30,  "pt_ticks": 9.5,  "sl_ticks": 9.8,  "desc": "30t/9.5/9.8"},
    {"vb_ticks": 60,  "pt_ticks": 20.0, "sl_ticks": 20.0, "desc": "60t/20/20"},
    {"vb_ticks": 120, "pt_ticks": 13.0, "sl_ticks": 14.5, "desc": "120t/13/14.5"},
]

# ── Derived helpers ───────────────────────────────────────────────────────────

def label_filename(vb_ticks: int, pt_ticks: float, sl_ticks: float) -> str:
    """Build Phase 7 label filename for a candidate (no .csv extension).

    Naming convention: phase7_labels_{vb_ticks}ticks_{pt_ticks}s_{sl_ticks}
    Example: phase7_labels_30ticks_9p5_9p8
    """
    pt_s = str(pt_ticks).replace(".", "p")
    sl_s = str(sl_ticks).replace(".", "p")
    return f"phase7_labels_{vb_ticks}ticks_{pt_s}_{sl_s}"


def label_filenames() -> list[str]:
    """All 3 candidate label filenames."""
    return [label_filename(c["vb_ticks"], c["pt_ticks"], c["sl_ticks"]) for c in CANDIDATES]


def parse_ts_to_ms(ts_str: str) -> int:
    """Fast parse of ISO-like timestamp string to milliseconds from midnight."""
    try:
        digits = "".join(c for c in ts_str if c.isdigit())
        hour = int(digits[8:10])
        minute = int(digits[10:12])
        sec = int(digits[12:14])
        ms = int(digits[14:17])
        return ((hour * 3600 + minute * 60 + sec) * 1000) + ms
    except Exception:
        return 0


# All feature column names expected from Phase 3 + Phase 4
ALL_FEATURE_PATTERNS = [
    "imbalance_1", "imbalance_5", "imbalance_10",
    "bid_depth_5", "ask_depth_5", "depth_ratio",
    "bid_qty_1", "ask_qty_1",
    "stack_bid_1", "pull_bid_1",
    "stack_bid_2", "pull_bid_2",
    "stack_bid_3", "pull_bid_3",
    "stack_bid_4", "pull_bid_4",
    "stack_bid_5", "pull_bid_5",
    "stack_ask_1", "pull_ask_1",
    "stack_ask_2", "pull_ask_2",
    "stack_ask_3", "pull_ask_3",
    "stack_ask_4", "pull_ask_4",
    "stack_ask_5", "pull_ask_5",
    "ps_weighted_bid", "ps_weighted_ask", "ps_net_weighted", "ps_delta_L1",
    "spread_ticks", "microprice", "mid_price_diff",
    "imbalance_mean_1s", "imbalance_std_1s",
    "imbalance_mean_5s", "imbalance_std_5s",
    "imbalance_mean_30s", "imbalance_std_30s",
    "ps_net_weighted_mean_1s", "ps_net_weighted_mean_5s", "ps_net_weighted_mean_30s",
    "pull_bid_1_sum_1s", "stack_bid_1_sum_1s",
    "pull_ask_1_sum_1s", "stack_ask_1_sum_1s",
    "ps_delta_L1_mean_1s", "ps_delta_L1_mean_5s", "ps_delta_L1_mean_30s",
    "update_rate_1s", "queue_exhaustion_1s",
]
