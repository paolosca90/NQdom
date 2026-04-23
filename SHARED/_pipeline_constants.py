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
    {"vb_ticks": 500,  "pt_ticks": 8.0,  "sl_ticks": 8.0,  "desc": "500t/8/8"},
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
    # ── Base price ─────────────────────────────────────────────────────────────
    "spread", "mid_price", "microprice", "imbalance_1",
    # ── Cumulative Delta (T&S-based) ───────────────────────────────────────────
    "delta_50", "delta_100", "delta_200", "delta_500",
    # ── Imbalance Trend ────────────────────────────────────────────────────────
    "imb_trend_10", "imb_trend_50",
    "imb_ma_ratio_10", "imb_ma_ratio_50",
    # ── Microprice Momentum ────────────────────────────────────────────────────
    "microprice_momentum_10", "microprice_momentum_50",
    "microprice_dev_from_ma",
    # ── Directional OFI ───────────────────────────────────────────────────────
    "ps_delta_L1", "ofi_50", "ofi_100", "ofi_500",
    # ── Stack Sweep ────────────────────────────────────────────────────────────
    "stack_sweep_bid_flag", "stack_sweep_ask_flag", "stack_sweep_any_flag",
    "bid_sweep_count", "ask_sweep_count",
    # ── Session ────────────────────────────────────────────────────────────────
    "vwap_dev_ticks", "vpin_100", "cum_delta_chunk",
    # ── P4 Rolling aggregates (temporal) ──────────────────────────────────────
    "imbalance_mean_1s", "imbalance_std_1s",
    "imbalance_mean_5s", "imbalance_std_5s",
    "imbalance_mean_30s", "imbalance_std_30s",
    "delta_50_mean_1s", "delta_50_std_1s",
    "delta_100_mean_1s", "delta_100_std_1s",
    "delta_200_mean_1s", "delta_200_std_1s",
    "imb_trend_10_mean_1s", "imb_trend_50_mean_1s",
    "imb_ma_ratio_10_mean_1s", "imb_ma_ratio_50_mean_1s",
    "microprice_momentum_10_mean_1s", "microprice_momentum_50_mean_1s",
    "microprice_dev_from_ma_mean_1s",
    "ofi_50_mean_1s", "ofi_100_mean_1s", "ofi_500_mean_1s",
    "ps_delta_L1_mean_1s", "ps_delta_L1_mean_5s", "ps_delta_L1_mean_30s",
    "stack_sweep_bid_flag_sum_1s", "stack_sweep_ask_flag_sum_1s",
    "stack_sweep_any_flag_sum_1s",
    "bid_sweep_count_mean_1s", "ask_sweep_count_mean_1s",
    "vwap_dev_ticks_mean_1s", "vpin_100_mean_1s",
    # ── P5 session delta features ──────────────────────────────────────────────
    "cum_delta_session", "delta_rolling_sum_5", "delta_rolling_sum_30",
    # ── OFI multi-livello (Kolm, Turiel & Westray 2023) ─────────────────────
    "ofi_L1", "ofi_L2", "ofi_L3", "ofi_L4", "ofi_L5",
    "ofi_L1_mean_1s", "ofi_L2_mean_1s", "ofi_L3_mean_1s",
    "ofi_L4_mean_1s", "ofi_L5_mean_1s",
    # ── Queue Imbalance per livello ──────────────────────────────────────────
    "qi_L1", "qi_L2", "qi_L3", "qi_L4", "qi_L5",
    "qi_L1_mean_1s", "qi_L2_mean_1s", "qi_L3_mean_1s",
    "qi_L4_mean_1s", "qi_L5_mean_1s",
    "qi_L1_mean_5s", "qi_L2_mean_5s",
    # ── Volatilità microprice ────────────────────────────────────────────────
    "microprice_vol_20", "microprice_vol_100",
    "microprice_vol_20_mean_1s", "microprice_vol_100_mean_1s",
    # ── Add/Cancel activity ratio ────────────────────────────────────────────
    "add_cancel_ratio_1s",
]
