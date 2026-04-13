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
    {"vb_ticks": 2000, "pt_ticks": 10.0, "sl_ticks": 10.0, "desc": "2000t/10/10"},
    {"vb_ticks": 4000, "pt_ticks": 20.0, "sl_ticks": 20.0, "desc": "4000t/20/20"},
    {"vb_ticks": 8000, "pt_ticks": 40.0, "sl_ticks": 40.0, "desc": "8000t/40/40"},
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


# ── NEW ARCHITECTURE: Trading Hours (Apr 2026 — DeepLOB NotebookLM) ───────────
# Excludes first/last 10 minutes of RTH to remove auction noise
# EDT (UTC-4, March–November): RTH 09:30–16:00 ET = 13:30–20:00 UTC
# Filtered window: 09:40–15:50 ET = 13:40–19:50 UTC
TRADING_START_ET = "09:40"
TRADING_END_ET = "15:50"
TRADING_START_UTC = "13:40"
TRADING_END_UTC = "19:50"
# Execution cadence
EXEC_INTERVAL_MIN = 5        # open positions every 5 minutes
EXEC_START_ET = "09:45"     # first entry window
EXEC_END_ET = "15:30"       # last entry window
EXIT_START_ET = "15:55"     # begin forced exit
EXIT_END_ET = "16:00"       # all positions closed by
# Neutral band
NEUTRAL_BAND_BPS = 0.0002   # ±2 basis points (DeepLOB stable band)
# PT threshold for deep neural net deployment
PT_THRESHOLD_DEEPLOB = 0.53  # 53% OOS — must exceed before CNN+LSTM
# Cost assumptions
COST_BP_NQ = 8              # ~8 ticks / $40 round-turn for NQ
COST_BP_ES = 2              # ~2 ticks / $25 round-turn for ES


# All feature column names expected from Phase 3 + Phase 4 + Phase 2b (TS features)
# Updated Apr 2026 — aligned with DeepLOB NotebookLM architecture
ALL_FEATURE_PATTERNS = [
    # ── LOB imbalance ──────────────────────────────────────────────────────────
    "imbalance_1", "imbalance_5", "imbalance_10",
    "bid_depth_5", "ask_depth_5", "depth_ratio",
    "bid_qty_1", "ask_qty_1",
    # ── Stack / Pull (price-level queue dynamics) ──────────────────────────────
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
    # ── Priority score / microprice ────────────────────────────────────────────
    "ps_weighted_bid", "ps_weighted_ask", "ps_net_weighted", "ps_delta_L1",
    "spread_ticks", "microprice", "mid_price_diff",
    # ── Rolling stats (temporal aggregation P4) ────────────────────────────────
    "imbalance_mean_1s", "imbalance_std_1s",
    "imbalance_mean_5s", "imbalance_std_5s",
    "imbalance_mean_30s", "imbalance_std_30s",
    "ps_net_weighted_mean_1s", "ps_net_weighted_mean_5s", "ps_net_weighted_mean_30s",
    "pull_bid_1_sum_1s", "stack_bid_1_sum_1s",
    "pull_ask_1_sum_1s", "stack_ask_1_sum_1s",
    "ps_delta_L1_mean_1s", "ps_delta_L1_mean_5s", "ps_delta_L1_mean_30s",
    "update_rate_1s", "queue_exhaustion_1s",
    # ── NEW: Time & Sales / Order Flow features (P2b) ────────────────────────
    # Delta decomposition: ΔL (limit new), ΔC (cancel/spoof), ΔM (market order)
    "delta_L", "delta_C", "delta_M",
    "delta_L_1s", "delta_C_1s", "delta_M_1s",
    "delta_L_5s", "delta_C_5s", "delta_M_5s",
    # Stacked imbalance: institutional aggression across 3+ consecutive levels (≥300%)
    "stacked_imbalance_bid_3", "stacked_imbalance_ask_3",
    "stacked_imbalance_bid_5", "stacked_imbalance_ask_5",
    # Volume sequencing: strictly increasing T&S volume across price levels
    "volume_sequence_bid", "volume_sequence_ask",
    # Bid/Ask fade: volume diminishes across top/bottom 3 levels at extreme
    "bid_fade_3", "ask_fade_3",
    # Exhaustion: zero volume at bar high/low (red/green candle)
    "exhaustion_bid", "exhaustion_ask",
    # Unfinished business: non-zero volume at new high/low (auction must return)
    "unfinished_business_bid", "unfinished_business_ask",
    # Closing delta extremes: closing delta ≥95% of max/min delta (momentum)
    "closing_delta_extreme_bid", "closing_delta_extreme_ask",
    # TICK Z-score: NYSE TICK index Z-score for market exhaustion filter (±2.5)
    "tick_zscore",
]
