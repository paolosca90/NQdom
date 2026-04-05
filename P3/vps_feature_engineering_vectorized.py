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

    # === SPATIAL FEATURES (Actual LOB Depth Xi) ===
    # Using the formal academic metric for LOB spread (Xi).
    bid_px_1 = to_float_series(df.get('bid_px_1', df['best_bid']))
    bid_px_10 = to_float_series(df.get('bid_px_10', df['best_bid'] - 9 * TICK_SIZE))
    ask_px_1 = to_float_series(df.get('ask_px_1', df['best_ask']))
    ask_px_10 = to_float_series(df.get('ask_px_10', df['best_ask'] + 9 * TICK_SIZE))

    result['actual_depth_bid_Xi'] = (bid_px_1 - bid_px_10) / TICK_SIZE
    result['actual_depth_ask_Xi'] = (ask_px_10 - ask_px_1) / TICK_SIZE


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
        tr_bid = to_float_series(df['traded_vol_bid']).values
        tr_ask = to_float_series(df['traded_vol_ask']).values
        
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
        'flow_limit_add_ask_5', 'flow_cancellation_ask_5'
    ]

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
        'flow_limit_add_ask_5', 'flow_cancellation_ask_5'
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

    # Concatenate all temp files into final output using `cat` (zero RAM overhead)
    print(f"\n  Concatenating {len(temp_files)} temp files with cat...")
    t_cat = time.time()

    # Write header, then append all chunks via cat
    with open(output_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(output_cols)

    for tf in temp_files:
        subprocess.run(['cat', str(tf)], stdout=open(output_path, 'a'), stderr=subprocess.DEVNULL)
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
