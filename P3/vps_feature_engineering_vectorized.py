"""
Phase 3: Feature Engineering - FULLY VECTORIZED with pandas
==========================================================

All feature computation vectorized using pandas/numpy instead of row-by-row Python.
Expected speedup: 10-50x over pure Python compute_features().
"""

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Constants
TICK_SIZE = 0.25
CHUNK_SIZE = 250_000  # 250K per chunk — ~900MB per chunk, 106 chunks total, fits in 8GB RAM


def compute_features_vectorized(df: pd.DataFrame, prev_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Compute ALL DOM features for a chunk using fully vectorized pandas operations.

    Replaces the row-by-row compute_features() function with bulk DataFrame operations.
    This processes millions of rows per second instead of thousands.

    Args:
        df: DataFrame with columns [bid_qty_1..10, ask_qty_1..10, best_bid, best_ask, ...]
        prev_df: Previous chunk's last row (for delta calculations), or None

    Returns:
        DataFrame with all computed features
    """
    # Copy to avoid modifying original
    result = df.copy()

    # Best bid/ask (scalar columns) - use numpy for pandas 3.0 compatibility
    def to_float_series(s):
        vals = s.values.astype(np.float64)
        return np.where(np.isnan(vals), 0.0, vals)

    best_bid = to_float_series(df['best_bid'])
    best_ask = to_float_series(df['best_ask'])
    spread = to_float_series(df['spread'])
    mid_price = to_float_series(df['mid_price'])

    # === STATIC FEATURES ===

    # spread_ticks: dollar spread / tick_size
    result['spread_ticks'] = spread / TICK_SIZE

    # microprice: volume-weighted mid-price
    bid_qty_1 = to_float_series(df['bid_qty_1'])
    ask_qty_1 = to_float_series(df['ask_qty_1'])
    total_q = bid_qty_1 + ask_qty_1
    result['microprice'] = np.where(
        total_q > 0,
        (best_bid * ask_qty_1 + best_ask * bid_qty_1) / total_q,
        (best_bid + best_ask) / 2.0
    )

    # mid_price_diff: mid_price(t) - mid_price(t-1)
    if prev_df is not None and len(prev_df) > 0:
        prev_mid_vals = prev_df['mid_price'].values.astype(np.float64)
        prev_mid = np.where(np.isnan(prev_mid_vals), 0.0, prev_mid_vals)[-1]
        prev_mid_series = pd.Series(prev_mid, index=result.index)
        result['mid_price_diff'] = mid_price - prev_mid_series
    else:
        result['mid_price_diff'] = 0.0

    # imbalance_1: (bid_qty_1 - ask_qty_1) / (bid_qty_1 + ask_qty_1)
    b1_plus_a1 = bid_qty_1 + ask_qty_1
    result['imbalance_1'] = np.where(
        b1_plus_a1 > 0,
        (bid_qty_1 - ask_qty_1) / b1_plus_a1,
        0.0
    )

    # === DEPTH FEATURES (vectorized sum over levels) ===
    bid_cols = [f'bid_qty_{i}' for i in range(1, 11)]
    ask_cols = [f'ask_qty_{i}' for i in range(1, 11)]

    bid_qty_vals = df[bid_cols].values.astype(np.float64)
    bid_qty_df = np.where(np.isnan(bid_qty_vals), 0.0, bid_qty_vals)
    ask_qty_vals = df[ask_cols].values.astype(np.float64)
    ask_qty_df = np.where(np.isnan(ask_qty_vals), 0.0, ask_qty_vals)

    # Depth sums (levels 1-5 and 1-10) - use numpy slicing
    bid_depth_5 = bid_qty_df[:, :5].sum(axis=1)
    ask_depth_5 = ask_qty_df[:, :5].sum(axis=1)
    bid_depth_10 = bid_qty_df.sum(axis=1)
    ask_depth_10 = ask_qty_df.sum(axis=1)

    # imbalance_5
    b5_plus_a5 = bid_depth_5 + ask_depth_5
    result['imbalance_5'] = np.where(
        b5_plus_a5 > 0,
        (bid_depth_5 - ask_depth_5) / b5_plus_a5,
        0.0
    )

    # imbalance_10
    b10_plus_a10 = bid_depth_10 + ask_depth_10
    result['imbalance_10'] = np.where(
        b10_plus_a10 > 0,
        (bid_depth_10 - ask_depth_10) / b10_plus_a10,
        0.0
    )

    # bid_depth_5, ask_depth_5
    result['bid_depth_5'] = bid_depth_5
    result['ask_depth_5'] = ask_depth_5

    # depth_ratio (capped at 100.0)
    MAX_DEPTH_RATIO = 100.0
    result['depth_ratio'] = np.where(
        ask_depth_5 > 0,
        np.minimum(bid_depth_5 / ask_depth_5, MAX_DEPTH_RATIO),
        np.where(bid_depth_5 > 0, MAX_DEPTH_RATIO, 1.0)
    )

    # === DIAGONAL STACKED IMBALANCES ===
    # Detect when 3+ consecutive LOB levels all show ask_qty > bid_qty (or vice versa)
    # by a ratio >= 300%. This is the institutional "sweep" footprint.
    STACKED_RATIO = 3.0   # 300% threshold
    MIN_LEVELS = 3        # minimum consecutive levels

    # Per-level dominance flags: shape [n_rows, 10]
    ask_dominates = (ask_qty_df > bid_qty_df * STACKED_RATIO)  # ask > 3x bid
    bid_dominates = (bid_qty_df > ask_qty_df * STACKED_RATIO)  # bid > 3x ask

    # Count consecutive dominated levels from level 1 outward
    # Vectorized: cumulative product resets at first False
    # ask-side stacking count (how many levels from L1 are consecutively ask-dominated)
    ask_stack_count = np.zeros(len(df), dtype=np.int8)
    bid_stack_count = np.zeros(len(df), dtype=np.int8)

    # Use cumulative product trick: running streak from level 0
    ask_streak = ask_dominates[:, 0].astype(np.int8)
    bid_streak = bid_dominates[:, 0].astype(np.int8)
    ask_stack_count += ask_streak
    bid_stack_count += bid_streak

    for lvl in range(1, 10):
        ask_streak = ask_streak * ask_dominates[:, lvl].astype(np.int8)
        bid_streak = bid_streak * bid_dominates[:, lvl].astype(np.int8)
        ask_stack_count += ask_streak
        bid_stack_count += bid_streak

    # Final features
    result['stacked_imb_ask'] = ask_stack_count           # int8: 0-10
    result['stacked_imb_bid'] = bid_stack_count           # int8: 0-10
    result['stacked_imb_score'] = (                       # int8: -10 to +10
        ask_stack_count.astype(np.int16) - bid_stack_count.astype(np.int16)
    ).clip(-127, 127).astype(np.int8)
    result['stacked_imb_flag_ask'] = (ask_stack_count >= MIN_LEVELS).astype(np.int8)
    result['stacked_imb_flag_bid'] = (bid_stack_count >= MIN_LEVELS).astype(np.int8)

    # === EXHAUSTION / THIN PRINTS ===
    # DOM-native equivalent of bar exhaustion: flag when L1 has near-zero quantity.
    THIN_THRESHOLD = 2.0  # contracts — tune based on NQ typical L1 size

    # Ask side thin: very little ask volume at L1 (potential exhaustion of sellers)
    result['exhaustion_ask_thin'] = (ask_qty_1 <= THIN_THRESHOLD).astype(np.int8)
    # Bid side thin: very little bid volume at L1 (potential exhaustion of buyers)
    result['exhaustion_bid_thin'] = (bid_qty_1 <= THIN_THRESHOLD).astype(np.int8)

    # Extreme case: one side is zero (unfinished auction)
    result['exhaustion_ask_zero'] = (ask_qty_1 == 0.0).astype(np.int8)
    result['exhaustion_bid_zero'] = (bid_qty_1 == 0.0).astype(np.int8)

    # Exhaustion imbalance: how extreme is the L1 ratio?
    # +1.0 = complete ask dominance (bid exhausted); -1.0 = complete bid dominance (ask exhausted)
    total_L1 = bid_qty_1 + ask_qty_1
    result['exhaustion_ratio'] = np.where(
        total_L1 > 0,
        np.clip((ask_qty_1 - bid_qty_1) / total_L1, -1.0, 1.0),
        0.0
    ).astype(np.float32)

    # === SPATIAL LOB DENSITY (improved — gap detection) ===
    # Keep existing basic Xi for backward compatibility
    bid_px_1 = to_float_series(df['bid_px_1']) if 'bid_px_1' in df.columns else to_float_series(df['best_bid'])
    bid_px_10_val = (df['best_bid'].astype(float) - 9 * TICK_SIZE) if 'bid_px_10' not in df.columns else df['bid_px_10']
    bid_px_10 = to_float_series(bid_px_10_val)
    ask_px_1 = to_float_series(df['ask_px_1']) if 'ask_px_1' in df.columns else to_float_series(df['best_ask'])
    ask_px_10_val = (df['best_ask'].astype(float) + 9 * TICK_SIZE) if 'ask_px_10' not in df.columns else df['ask_px_10']
    ask_px_10 = to_float_series(ask_px_10_val)

    result['actual_depth_bid_Xi'] = ((bid_px_1 - bid_px_10) / TICK_SIZE).astype(np.float32)
    result['actual_depth_ask_Xi'] = ((ask_px_10 - ask_px_1) / TICK_SIZE).astype(np.float32)

    # NEW: Gap detection between consecutive populated levels
    # A "gap" is when price jump between levels > 1 tick (vacuum)
    bid_px_cols = [f'bid_px_{i}' for i in range(1, 11)]
    ask_px_cols = [f'ask_px_{i}' for i in range(1, 11)]

    available_bid_px = [c for c in bid_px_cols if c in df.columns]
    available_ask_px = [c for c in ask_px_cols if c in df.columns]

    if len(available_bid_px) >= 2:
        bid_px_arr = df[available_bid_px].values.astype(np.float64)
        bid_px_arr = np.where(np.isnan(bid_px_arr), np.nan, bid_px_arr)

        # Bid prices decrease as level increases, so gaps are px[i] - px[i+1]
        bid_gaps = bid_px_arr[:, :-1] - bid_px_arr[:, 1:]  # shape [n_rows, n_levels-1]
        bid_gaps_ticks = bid_gaps / TICK_SIZE

        result['lob_max_gap_bid'] = np.nanmax(bid_gaps_ticks, axis=1).astype(np.float32)
        result['lob_vacuum_count_bid'] = (bid_gaps_ticks > 2.0).sum(axis=1).astype(np.int8)
    else:
        result['lob_max_gap_bid'] = np.float32(0.0)
        result['lob_vacuum_count_bid'] = np.int8(0)

    if len(available_ask_px) >= 2:
        ask_px_arr = df[available_ask_px].values.astype(np.float64)
        ask_px_arr = np.where(np.isnan(ask_px_arr), np.nan, ask_px_arr)

        # Ask prices increase as level increases, so gaps are px[i+1] - px[i]
        ask_gaps = ask_px_arr[:, 1:] - ask_px_arr[:, :-1]
        ask_gaps_ticks = ask_gaps / TICK_SIZE

        result['lob_max_gap_ask'] = np.nanmax(ask_gaps_ticks, axis=1).astype(np.float32)
        result['lob_vacuum_count_ask'] = (ask_gaps_ticks > 2.0).sum(axis=1).astype(np.int8)
    else:
        result['lob_max_gap_ask'] = np.float32(0.0)
        result['lob_vacuum_count_ask'] = np.int8(0)

    # Combined vacuum signal
    result['lob_vacuum_score'] = (
        result['lob_vacuum_count_bid'].astype(np.int16) +
        result['lob_vacuum_count_ask'].astype(np.int16)
    ).clip(0, 127).astype(np.int8)


    # === STACK/PULL FEATURES ===
    if prev_df is not None and len(prev_df) > 0:
        # Get previous chunk's last row quantities (numpy arrays)
        prev_bid_vals = prev_df[bid_cols].values.astype(np.float64)[-1]
        prev_bid = np.where(np.isnan(prev_bid_vals), 0.0, prev_bid_vals)
        prev_ask_vals = prev_df[ask_cols].values.astype(np.float64)[-1]
        prev_ask = np.where(np.isnan(prev_ask_vals), 0.0, prev_ask_vals)

        # Broadcast prev row to all rows in current chunk
        prev_bid_df = np.tile(prev_bid, (len(df), 1))
        prev_ask_df = np.tile(prev_ask, (len(df), 1))
    else:
        prev_bid_df = np.zeros((len(df), 10), dtype=np.float64)
        prev_ask_df = np.zeros((len(df), 10), dtype=np.float64)

    # Stack/pull (vectorized numpy operations)
    stack_bid = np.maximum(0.0, bid_qty_df - prev_bid_df)
    pull_bid = np.maximum(0.0, prev_bid_df - bid_qty_df)
    stack_ask = np.maximum(0.0, ask_qty_df - prev_ask_df)
    pull_ask = np.maximum(0.0, prev_ask_df - ask_qty_df)

    # Stack/pull level 1
    result['stack_bid_1'] = stack_bid[:, 0]
    result['pull_bid_1'] = pull_bid[:, 0]
    result['stack_ask_1'] = stack_ask[:, 0]
    result['pull_ask_1'] = pull_ask[:, 0]

    # NOTE: tradedvol_bid e tradedvol_ask sono presenti solo se P2b ha
    # processato snapshots.csv (flusso ufficiale). Se assenti, il branch
    # fallback usa euristica heuristic flow decomposition (meno precisa).
    # Con il flusso ufficiale P2b questi campi sono SEMPRE presenti.

    # Exact Microstructural Flow Decomposition (ΔL vs ΔM vs ΔC)
    # If Data Fusion (vps_phase2b) appended the true MBO Trade Volume, we separate exact Cancellations.
    has_trades = 'traded_vol_bid' in df.columns and 'traded_vol_ask' in df.columns
    
    if has_trades:
        tr_bid = to_float_series(df['traded_vol_bid'])
        tr_ask = to_float_series(df['traded_vol_ask'])
        
        # Market sweeps match the missing LOB volumes.
        # Cancellation = Total Volume Removed (pull) - Trade Volume 
        # (Capped at 0 to avoid noise artifacts from async execution streams)
        cxl_bid_L1 = np.maximum(0.0, pull_bid[:, 0] - tr_bid)
        cxl_ask_L1 = np.maximum(0.0, pull_ask[:, 0] - tr_ask)
        
        cxl_bid_5 = np.maximum(0.0, pull_bid[:, :5].sum(axis=1) - tr_bid)
        cxl_ask_5 = np.maximum(0.0, pull_ask[:, :5].sum(axis=1) - tr_ask)
        
        result['flow_limit_add_bid_L1'] = stack_bid[:, 0]
        result['flow_cancellation_bid_L1'] = cxl_bid_L1
        result['flow_market_sell_L1'] = tr_bid
        
        result['flow_limit_add_ask_L1'] = stack_ask[:, 0]
        result['flow_cancellation_ask_L1'] = cxl_ask_L1
        result['flow_market_buy_L1'] = tr_ask
        
        result['flow_limit_add_bid_5'] = stack_bid[:, :5].sum(axis=1)
        result['flow_cancellation_bid_5'] = cxl_bid_5
        
        result['flow_limit_add_ask_5'] = stack_ask[:, :5].sum(axis=1)
        result['flow_cancellation_ask_5'] = cxl_ask_5
    else:
        # Fallback to Heuristic Flow Decomposition without T&S
        result['flow_limit_add_bid_L1'] = stack_bid[:, 0]
        result['flow_cancellation_bid_L1'] = pull_bid[:, 0]
        result['flow_market_sell_L1'] = 0.0
        
        result['flow_limit_add_ask_L1'] = stack_ask[:, 0]
        result['flow_cancellation_ask_L1'] = pull_ask[:, 0]
        result['flow_market_buy_L1'] = 0.0
        
        result['flow_limit_add_bid_5'] = stack_bid[:, :5].sum(axis=1)
        result['flow_cancellation_bid_5']   = pull_bid[:, :5].sum(axis=1)
        result['flow_limit_add_ask_5'] = stack_ask[:, :5].sum(axis=1)
        result['flow_cancellation_ask_5']   = pull_ask[:, :5].sum(axis=1)

    # === CUMULATIVE DELTA PROXY (components for later divergence detection) ===
    # These are computed per-chunk and output to CSV.
    # Full divergence detection happens in P3b on the sampled events sequence.
    # NOTE: Full Cumulative Delta Divergence (price_low + delta_higher_low) is implemented
    # in phase3b_temporal_features.py (P3b) which processes the CUSUM-sampled sequence
    # where rolling windows work correctly across the full trading day.
    if has_trades:
        # Aggressive buy = market_buy_L1 (taker hitting ask)
        # Aggressive sell = market_sell_L1 (taker hitting bid)
        agg_buy = to_float_series(df['traded_vol_ask'])   # ask side traded = aggressive buy
        agg_sell = to_float_series(df['traded_vol_bid'])  # bid side traded = aggressive sell
        result['delta_raw'] = (agg_buy - agg_sell).astype(np.float32)
    else:
        # Fallback: compute ps_delta_L1 directly from stack/pull numpy arrays
        result['delta_raw'] = (
            (stack_bid[:, 0] - pull_bid[:, 0]) - (stack_ask[:, 0] - pull_ask[:, 0])
        ).astype(np.float32)

    # Cumulative delta within chunk (resets each chunk — P3b will stitch across chunks)
    result['cum_delta_chunk'] = result['delta_raw'].cumsum().astype(np.float32)

    # Delta sign (useful for sequence pattern detection)
    result['delta_sign'] = np.sign(result['delta_raw']).astype(np.int8)

    # Stack/pull levels 2-5
    for i in range(1, 5):
        result[f'stack_bid_{i+1}'] = stack_bid[:, i]
        result[f'pull_bid_{i+1}'] = pull_bid[:, i]
        result[f'stack_ask_{i+1}'] = stack_ask[:, i]
        result[f'pull_ask_{i+1}'] = pull_ask[:, i]

    # ps_weighted_bid: Σ (stack_bid_N - pull_bid_N) × (1/N) for N=1..5
    weights = np.array([1.0 / (n + 1) for n in range(5)])
    result['ps_weighted_bid'] = (
        (stack_bid[:, :5] - pull_bid[:, :5]) * weights
    ).sum(axis=1)

    # ps_weighted_ask: Σ (stack_ask_N - pull_ask_N) × (1/N) for N=1..5
    result['ps_weighted_ask'] = (
        (stack_ask[:, :5] - pull_ask[:, :5]) * weights
    ).sum(axis=1)

    # ps_net_weighted
    result['ps_net_weighted'] = result['ps_weighted_bid'] - result['ps_weighted_ask']

    # ps_delta_L1
    result['ps_delta_L1'] = (
        (stack_bid[:, 0] - pull_bid[:, 0]) - (stack_ask[:, 0] - pull_ask[:, 0])
    )

    # Round outputs
    result['spread_ticks'] = result['spread_ticks'].round(6)
    result['microprice'] = result['microprice'].round(6)
    result['mid_price_diff'] = result['mid_price_diff'].round(6)
    result['imbalance_1'] = result['imbalance_1'].round(6)
    result['imbalance_5'] = result['imbalance_5'].round(6)
    result['imbalance_10'] = result['imbalance_10'].round(6)
    result['bid_depth_5'] = result['bid_depth_5'].round(2)
    result['ask_depth_5'] = result['ask_depth_5'].round(2)
    result['depth_ratio'] = result['depth_ratio'].round(6)
    result['bid_qty_1'] = bid_qty_1.round(2)
    result['ask_qty_1'] = ask_qty_1.round(2)
    result['stack_bid_1'] = result['stack_bid_1'].round(2)
    result['pull_bid_1'] = result['pull_bid_1'].round(2)
    result['ps_weighted_bid'] = result['ps_weighted_bid'].round(4)
    result['ps_weighted_ask'] = result['ps_weighted_ask'].round(4)
    result['ps_net_weighted'] = result['ps_net_weighted'].round(4)
    result['ps_delta_L1'] = result['ps_delta_L1'].round(4)

    for i in range(1, 6):
        result[f'stack_bid_{i}'] = result[f'stack_bid_{i}'].round(2)
        result[f'pull_bid_{i}'] = result[f'pull_bid_{i}'].round(2)
        result[f'stack_ask_{i}'] = result[f'stack_ask_{i}'].round(2)
        result[f'pull_ask_{i}'] = result[f'pull_ask_{i}'].round(2)

    result['actual_depth_bid_Xi'] = result['actual_depth_bid_Xi'].round(2)
    result['actual_depth_ask_Xi'] = result['actual_depth_ask_Xi'].round(2)
    result['flow_limit_add_bid_L1'] = result['flow_limit_add_bid_L1'].round(2)
    result['flow_cancellation_bid_L1'] = result['flow_cancellation_bid_L1'].round(2)
    result['flow_market_sell_L1'] = result['flow_market_sell_L1'].round(2)
    result['flow_limit_add_ask_L1'] = result['flow_limit_add_ask_L1'].round(2)
    result['flow_cancellation_ask_L1'] = result['flow_cancellation_ask_L1'].round(2)
    result['flow_market_buy_L1'] = result['flow_market_buy_L1'].round(2)
    result['flow_limit_add_bid_5'] = result['flow_limit_add_bid_5'].round(2)
    result['flow_cancellation_bid_5'] = result['flow_cancellation_bid_5'].round(2)
    result['flow_limit_add_ask_5'] = result['flow_limit_add_ask_5'].round(2)
    result['flow_cancellation_ask_5'] = result['flow_cancellation_ask_5'].round(2)

    # Select output columns (same as original)
    output_cols = [
        'ts', 'spread_ticks', 'microprice', 'mid_price_diff',
        'imbalance_1', 'imbalance_5', 'imbalance_10',
        'bid_depth_5', 'ask_depth_5', 'depth_ratio',
        'bid_qty_1', 'ask_qty_1',
        'stack_bid_1', 'pull_bid_1', 'stack_bid_2', 'pull_bid_2',
        'stack_bid_3', 'pull_bid_3', 'stack_bid_4', 'pull_bid_4',
        'stack_bid_5', 'pull_bid_5',
        'stack_ask_1', 'pull_ask_1', 'stack_ask_2', 'pull_ask_2',
        'stack_ask_3', 'pull_ask_3', 'stack_ask_4', 'pull_ask_4',
        'stack_ask_5', 'pull_ask_5',
        'ps_weighted_bid', 'ps_weighted_ask', 'ps_net_weighted', 'ps_delta_L1',
        # New Microstructural Features
        'actual_depth_bid_Xi', 'actual_depth_ask_Xi',
        'flow_limit_add_bid_L1', 'flow_cancellation_bid_L1', 'flow_market_sell_L1',
        'flow_limit_add_ask_L1', 'flow_cancellation_ask_L1', 'flow_market_buy_L1',
        'flow_limit_add_bid_5', 'flow_cancellation_bid_5',
        'flow_limit_add_ask_5', 'flow_cancellation_ask_5',
        # NEW order flow features
        'delta_raw', 'cum_delta_chunk', 'delta_sign',
        # Diagonal Stacked Imbalances
        'stacked_imb_ask', 'stacked_imb_bid', 'stacked_imb_score',
        'stacked_imb_flag_ask', 'stacked_imb_flag_bid',
        # Exhaustion / Thin Prints
        'exhaustion_ask_thin', 'exhaustion_bid_thin',
        'exhaustion_ask_zero', 'exhaustion_bid_zero', 'exhaustion_ratio',
        # LOB Spatial Density (improved)
        'lob_max_gap_bid', 'lob_max_gap_ask',
        'lob_vacuum_count_bid', 'lob_vacuum_count_ask', 'lob_vacuum_score',
    ]

    # === MEMORY DOWNCAST (float64 → float32, flags → int8) ===
    # Saves ~40% RAM on output columns
    float32_cols = [
        'spread_ticks', 'microprice', 'mid_price_diff',
        'imbalance_1', 'imbalance_5', 'imbalance_10',
        'bid_depth_5', 'ask_depth_5', 'depth_ratio',
        'bid_qty_1', 'ask_qty_1',
        'ps_weighted_bid', 'ps_weighted_ask', 'ps_net_weighted', 'ps_delta_L1',
        'actual_depth_bid_Xi', 'actual_depth_ask_Xi',
        'lob_max_gap_bid', 'lob_max_gap_ask',
        'exhaustion_ratio', 'delta_raw', 'cum_delta_chunk',
        'flow_limit_add_bid_L1', 'flow_cancellation_bid_L1', 'flow_market_sell_L1',
        'flow_limit_add_ask_L1', 'flow_cancellation_ask_L1', 'flow_market_buy_L1',
        'flow_limit_add_bid_5', 'flow_cancellation_bid_5',
        'flow_limit_add_ask_5', 'flow_cancellation_ask_5',
    ]
    int8_cols = [
        'stacked_imb_ask', 'stacked_imb_bid', 'stacked_imb_score',
        'stacked_imb_flag_ask', 'stacked_imb_flag_bid',
        'exhaustion_ask_thin', 'exhaustion_bid_thin',
        'exhaustion_ask_zero', 'exhaustion_bid_zero',
        'delta_sign', 'lob_vacuum_count_bid', 'lob_vacuum_count_ask',
        'lob_vacuum_score',
    ]
    for c in float32_cols:
        if c in result.columns:
            result[c] = result[c].astype(np.float32)
    for c in int8_cols:
        if c in result.columns:
            result[c] = result[c].astype(np.int8)

    return result[output_cols]


def compute_features_chunked(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """
    Read snapshots.csv in chunks, compute features using VECTORIZED pandas operations.
    Writes to temp CSV files, then concatenates at end (no OOM, ~3x faster than original).
    """
    import time
    import subprocess

    stats = {
        "rows_processed": 0,
        "first_ts": None,
        "last_ts": None,
        "rows_written": 0,
    }

    prev_df = None
    t_start = time.time()

    output_cols = [
        'ts', 'spread_ticks', 'microprice', 'mid_price_diff',
        'imbalance_1', 'imbalance_5', 'imbalance_10',
        'bid_depth_5', 'ask_depth_5', 'depth_ratio',
        'bid_qty_1', 'ask_qty_1',
        'stack_bid_1', 'pull_bid_1', 'stack_bid_2', 'pull_bid_2',
        'stack_bid_3', 'pull_bid_3', 'stack_bid_4', 'pull_bid_4',
        'stack_bid_5', 'pull_bid_5',
        'stack_ask_1', 'pull_ask_1', 'stack_ask_2', 'pull_ask_2',
        'stack_ask_3', 'pull_ask_3', 'stack_ask_4', 'pull_ask_4',
        'stack_ask_5', 'pull_ask_5',
        'ps_weighted_bid', 'ps_weighted_ask', 'ps_net_weighted', 'ps_delta_L1',
        # New Microstructural Features
        'actual_depth_bid_Xi', 'actual_depth_ask_Xi',
        'flow_limit_add_bid_L1', 'flow_cancellation_bid_L1', 'flow_market_sell_L1',
        'flow_limit_add_ask_L1', 'flow_cancellation_ask_L1', 'flow_market_buy_L1',
        'flow_limit_add_bid_5', 'flow_cancellation_bid_5',
        'flow_limit_add_ask_5', 'flow_cancellation_ask_5',
        # NEW order flow features
        'delta_raw', 'cum_delta_chunk', 'delta_sign',
        # Diagonal Stacked Imbalances
        'stacked_imb_ask', 'stacked_imb_bid', 'stacked_imb_score',
        'stacked_imb_flag_ask', 'stacked_imb_flag_bid',
        # Exhaustion / Thin Prints
        'exhaustion_ask_thin', 'exhaustion_bid_thin',
        'exhaustion_ask_zero', 'exhaustion_bid_zero', 'exhaustion_ratio',
        # LOB Spatial Density (improved)
        'lob_max_gap_bid', 'lob_max_gap_ask',
        'lob_vacuum_count_bid', 'lob_vacuum_count_ask', 'lob_vacuum_score',
    ]

    temp_dir = Path(output_path).parent
    temp_files = []

    chunk_num = 0
    with pd.read_csv(input_path, chunksize=CHUNK_SIZE, dtype=str) as reader:
        for chunk_df in reader:
            t0 = time.time()
            chunk_num += 1

            # Compute features
            features_df = compute_features_vectorized(chunk_df, prev_df)

            # Write chunk to temp CSV file
            temp_file = temp_dir / f".p3_chunk_{chunk_num}.tmp"
            features_df[output_cols].to_csv(temp_file, header=False, index=False)
            temp_files.append(temp_file)

            stats['rows_processed'] += len(chunk_df)
            stats['rows_written'] += len(features_df)

            if stats['first_ts'] is None:
                stats['first_ts'] = chunk_df['ts'].iloc[0]

            prev_df = chunk_df.tail(1).reset_index(drop=True)

            t1 = time.time()
            print(f"  Chunk {chunk_num}: {len(chunk_df):,} rows in {t1-t0:.2f}s ({len(chunk_df)/(t1-t0):,.0f} rows/sec)")

    # Concatenate all temp files into final output using Python's shutil (RAM-safe, zero overhead)
    print(f"\n  Concatenating {len(temp_files)} temp files...")
    t_cat = time.time()

    import shutil
    with open(output_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(output_cols)
        
        for tf in temp_files:
            with open(tf, 'r', encoding='utf-8') as fin:
                shutil.copyfileobj(fin, fout)
            tf.unlink()  # Delete temp file after appending

    t_cat_end = time.time()
    print(f"  Concatenation done in {t_cat_end-t_cat:.1f}s")

    t_end = time.time()
    stats['last_ts'] = prev_df['ts'].iloc[-1] if prev_df is not None else None

    print(f"\n  [P3] Total: {stats['rows_processed']:,} rows in {t_end-t_start:.1f}s")
    print(f"  [P3] Throughput: {stats['rows_processed'] / (t_end - t_start):,.0f} rows/sec")

    return stats


def print_report(stats: dict[str, Any]) -> None:
    """Print the Phase 3 terminal report."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING - PHASE 3 (VECTORIZED)")
    print("=" * 60)
    print(f"\n[Rows]")
    print(f"  Processed : {stats['rows_processed']:,}")
    print(f"  Written   : {stats['rows_written']:,}")
    print(f"\n[Time Range]")
    print(f"  First : {stats['first_ts'] or 'N/A'}")
    print(f"  Last  : {stats['last_ts'] or 'N/A'}")
    print("\n" + "=" * 60)
    print("Phase 3 completed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Synthetic test - 1000 rows
    import numpy as np
    import pandas as pd

    n = 1000
    np.random.seed(42)
    base_price = 19000.0

    test_df = pd.DataFrame({
        'ts': range(n),
        'best_bid': base_price + np.random.randn(n) * 0.5,
        'best_ask': base_price + 0.25 + np.random.randn(n) * 0.5,
        'spread': [0.25] * n,
        'mid_price': base_price + np.random.randn(n) * 0.5,
        **{f'bid_qty_{i}': np.random.randint(1, 50, n).astype(float) for i in range(1, 11)},
        **{f'ask_qty_{i}': np.random.randint(1, 50, n).astype(float) for i in range(1, 11)},
        **{f'bid_px_{i}': base_price - (i-1)*0.25 + np.random.randn(n)*0.01 for i in range(1, 11)},
        **{f'ask_px_{i}': base_price + 0.25 + (i-1)*0.25 + np.random.randn(n)*0.01 for i in range(1, 11)},
    })

    # Force a few rows with stacked imbalances for testing
    test_df.loc[100:105, [f'ask_qty_{i}' for i in range(1, 6)]] = 150.0
    test_df.loc[100:105, [f'bid_qty_{i}' for i in range(1, 6)]] = 5.0

    result = compute_features_vectorized(test_df, None)

    print("=== P3 NEW FEATURES VALIDATION ===")
    print(f"Output columns: {len(result.columns)}")
    print(f"Output shape: {result.shape}")
    print()

    # Check stacked imbalances
    print("STACKED IMBALANCES (rows 98-108):")
    print(result[['stacked_imb_ask','stacked_imb_bid','stacked_imb_score','stacked_imb_flag_ask']].iloc[98:108])

    # Check exhaustion
    print("\nEXHAUSTION (first 5 rows):")
    print(result[['exhaustion_ask_thin','exhaustion_bid_thin','exhaustion_ratio']].head())

    # Check vacuum
    print("\nLOB VACUUM (first 5 rows):")
    print(result[['lob_max_gap_bid','lob_max_gap_ask','lob_vacuum_score']].head())

    # Check delta
    print("\nDELTA (first 5 rows):")
    print(result[['delta_raw','cum_delta_chunk','delta_sign']].head())

    # Check dtypes
    print("\nDTYPES:")
    for col in ['stacked_imb_score', 'exhaustion_ratio', 'lob_max_gap_bid', 'delta_raw']:
        if col in result.columns:
            print(f"  {col}: {result[col].dtype}")

    # Validate stacked detection worked on forced rows
    assert result['stacked_imb_flag_ask'].iloc[100] == 1, "FAIL: stacked_imb_flag_ask not triggered"
    assert result['stacked_imb_flag_bid'].iloc[100] == 0, "FAIL: bid flag should be 0"
    print("\n[OK] All assertions passed - P3 new features validated")
    print("Run: python vps_feature_engineering_vectorized.py")
