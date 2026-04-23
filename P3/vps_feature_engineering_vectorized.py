"""
Phase 3: Feature Engineering — MOMENTUM/DIRECTIONAL FOCUS
==========================================================
REBUILD AFTER P8 DIAGNOSTIC (Apr 19, 2026):

Root cause: P3/P4 instantaneous state features (|r|=0.014 max correlation)
are noise that drowns out any predictive signal in LGBM/XGB splits.

Strategy:
  - DROP  : static LOB snapshot noise (raw L1/L2 depths, instantaneous exhaustion)
  - KEEP  : rolling momentum features that capture ORDER FLOW DIRECTION
  - ADD   : cumulative delta, imbalance trends, microprice momentum,
             stack-sweep detection

All feature computation vectorized using pandas/numpy (float32 optimized).
"""

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Constants
TICK_SIZE = 0.25
CHUNK_SIZE = 100_000  # 100K per chunk


def to_float32_arr(s):
    """Convert series to float32 numpy array, replacing NaN with 0."""
    vals = s.values.astype(np.float64)
    return np.where(np.isnan(vals), 0.0, vals).astype(np.float32)


def compute_features_vectorized(
    df: pd.DataFrame, prev_df: pd.DataFrame | None, state: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Compute MOMENTUM/DIRECTIONAL features for a chunk using vectorized pandas.

    Rolling windows (up to 500 rows) are maintained across chunks via prev_df.
    All output columns are float32.

    NEW FEATURE GROUPS (Apr 2026):
      1. CUMULATIVE DELTA  — rolling sum of (agg_buy - agg_sell) over N events
      2. IMBALANCE TREND   — rate of change of L1 imbalance over rolling windows
      3. MICROPRICE MOMENTUM — microprice_t − microprice_{t-N}
      4. STACK SWEEP FLAG  — boolean: 3+ consecutive levels with >50% pull on 1 side
      5. DIRECTIONAL OFI   — Order Flow Imbalance at multiple window sizes
    """

    if state is None:
        state = {'cum_vol': 0.0, 'cum_pv': 0.0,
                 'prev_bid_qty_row': np.zeros(10, np.float32),
                 'prev_ask_qty_row': np.zeros(10, np.float32)}

    # ── Prepend prev_df for correct rolling windows ─────────────────────────────
    len_prev = len(prev_df) if prev_df is not None else 0
    if len_prev > 0:
        working_df = pd.concat([prev_df, df], ignore_index=True)
    else:
        working_df = df.copy()

    n_rows = len(working_df)
    result = pd.DataFrame(index=working_df.index)

    # ── Base price fields ────────────────────────────────────────────────────────
    best_bid    = to_float32_arr(working_df['best_bid'])
    best_ask    = to_float32_arr(working_df['best_ask'])
    mid_price   = to_float32_arr(working_df['mid_price'])
    spread      = to_float32_arr(working_df['spread'])

    result['ts']        = working_df['ts']
    result['spread']    = spread
    result['mid_price'] = mid_price

    # ── Microprice ──────────────────────────────────────────────────────────────
    bid_qty_1 = to_float32_arr(working_df['bid_qty_1'])
    ask_qty_1 = to_float32_arr(working_df['ask_qty_1'])
    total_q   = bid_qty_1 + ask_qty_1
    microprice = np.where(
        total_q > 0,
        (best_bid * ask_qty_1 + best_ask * bid_qty_1) / total_q,
        (best_bid + best_ask) / 2.0
    ).astype(np.float32)
    result['microprice'] = microprice

    # ── L1 Imbalance (keep — used as base for trend) ──────────────────────────
    b1_plus_a1 = bid_qty_1 + ask_qty_1
    imbalance_1 = np.where(b1_plus_a1 > 0, (bid_qty_1 - ask_qty_1) / b1_plus_a1, 0.0).astype(np.float32)
    result['imbalance_1'] = imbalance_1

    # ── T&S-based trade delta (EXCLUSIVE source for trade classification) ─────────
    # traded_vol_ask = cumulative buy volume (from T&S via P2b merge_asof)
    # traded_vol_bid = cumulative sell volume (from T&S via P2b merge_asof)
    # Per-snapshot trade_delta = diff(traded_vol_ask) - diff(traded_vol_bid)
    # This is the SAME classification as Sierra Chart T&S SC_TS_ASK/BID
    # and matches sc.GetTimeAndSales() in ACSIL Phase 9.
    # MBO quantity changes are NOT used for trade_delta — they conflate
    # market orders (ΔM) with cancellations (ΔC), destroying directional signal.
    tsv_buy  = to_float32_arr(working_df['traded_vol_ask'])
    tsv_sell = to_float32_arr(working_df['traded_vol_bid'])

    # Per-snapshot signed delta (diff of cumulative volumes)
    delta_vol_buy  = np.zeros(n_rows, dtype=np.float32)
    delta_vol_sell = np.zeros(n_rows, dtype=np.float32)
    delta_vol_buy[1:]  = tsv_buy[1:]  - tsv_buy[:-1]
    delta_vol_sell[1:] = tsv_sell[1:] - tsv_sell[:-1]
    # Clip negative diffs (book rebuilt between snapshots → zero contribution)
    delta_vol_buy  = np.maximum(0.0, delta_vol_buy)
    delta_vol_sell = np.maximum(0.0, delta_vol_sell)
    trade_delta = delta_vol_buy - delta_vol_sell
    trade_size = delta_vol_buy + delta_vol_sell
    td_series = pd.Series(trade_delta)

    # agg_buy / agg_sell for VPIN (T&S-based volume)
    agg_buy  = delta_vol_buy.astype(np.float32)
    agg_sell = delta_vol_sell.astype(np.float32)

    # ════════════════════════════════════════════════════════════════════════════
    # NEW FEATURE GROUP 2: IMBALANCE TREND (rate of change of L1 imbalance)
    # ════════════════════════════════════════════════════════════════════════════
    imb_series = pd.Series(imbalance_1)

    # imbalance now vs N events ago
    result['imb_trend_10'] = (imbalance_1 - imb_series.shift(10).fillna(0.0).values).astype(np.float32)
    result['imb_trend_50'] = (imbalance_1 - imb_series.shift(50).fillna(0.0).values).astype(np.float32)

    # imbalance rolling mean vs current imbalance (is pressure building or fading?)
    result['imb_ma_ratio_10'] = (imbalance_1 / (imb_series.rolling(10, min_periods=1).mean().values + 1e-9)).astype(np.float32)
    result['imb_ma_ratio_50'] = (imbalance_1 / (imb_series.rolling(50, min_periods=1).mean().values + 1e-9)).astype(np.float32)

    # ════════════════════════════════════════════════════════════════════════════
    # NEW FEATURE GROUP 3: MICROPRICE MOMENTUM
    # ════════════════════════════════════════════════════════════════════════════
    mp_series = pd.Series(microprice)

    result['microprice_momentum_10'] = (microprice - mp_series.shift(10).fillna(0.0).values).astype(np.float32)
    result['microprice_momentum_50'] = (microprice - mp_series.shift(50).fillna(0.0).values).astype(np.float32)

    # microprice vs its rolling mean — deviation from recent average
    result['microprice_dev_from_ma'] = (microprice - mp_series.rolling(20, min_periods=1).mean().values).astype(np.float32)

    # ════════════════════════════════════════════════════════════════════════════
    # NEW FEATURE GROUP 4: DIRECTIONAL OFI (Order Flow Imbalance)
    # ════════════════════════════════════════════════════════════════════════════
    # OFI = (stack_bid − pull_bid) − (stack_ask − pull_ask)  [priority score delta]
    bid_cols = [f'bid_qty_{i}' for i in range(1, 11)]
    ask_cols = [f'ask_qty_{i}' for i in range(1, 11)]

    bid_qty_vals = working_df[bid_cols].values.astype(np.float32)
    ask_qty_vals = working_df[ask_cols].values.astype(np.float32)
    bid_qty_vals = np.where(np.isnan(bid_qty_vals), 0.0, bid_qty_vals)
    ask_qty_vals = np.where(np.isnan(ask_qty_vals), 0.0, ask_qty_vals)

    # Previous snapshot (1-row shift within chunk)
    prev_bid = np.roll(bid_qty_vals, 1, axis=0)
    prev_ask = np.roll(ask_qty_vals, 1, axis=0)
    if len_prev > 0:
        # Cross-chunk: first row's prev = last row of previous chunk (stored in state)
        prev_bid[0] = state['prev_bid_qty_row']
        prev_ask[0] = state['prev_ask_qty_row']
    else:
        prev_bid[0] = bid_qty_vals[0]
        prev_ask[0] = ask_qty_vals[0]

    stack_bid = np.maximum(0.0, bid_qty_vals - prev_bid)
    pull_bid  = np.maximum(0.0, prev_bid - bid_qty_vals)
    stack_ask = np.maximum(0.0, ask_qty_vals - prev_ask)
    pull_ask  = np.maximum(0.0, prev_ask - ask_qty_vals)

    # PS delta L1 = net priority score change at best bid vs best ask
    # This is a MBO DOM spatial feature (not trade classification)
    ps_delta_L1 = (stack_bid[:, 0] - pull_bid[:, 0]) - (stack_ask[:, 0] - pull_ask[:, 0])
    result['ps_delta_L1'] = ps_delta_L1.astype(np.float32)

    # ── T&S-based agg already set above (agg_buy, agg_sell from traded_vol) ─────

    # ════════════════════════════════════════════════════════════════════════════
    result['delta_50']  = td_series.rolling(50,  min_periods=1).sum().values.astype(np.float32)
    result['delta_100'] = td_series.rolling(100, min_periods=1).sum().values.astype(np.float32)
    result['delta_200'] = td_series.rolling(200, min_periods=1).sum().values.astype(np.float32)
    result['delta_500'] = td_series.rolling(500, min_periods=1).sum().values.astype(np.float32)

    # Rolling OFI at multiple windows
    ofi_series = pd.Series(ps_delta_L1)
    result['ofi_50']  = ofi_series.rolling(50,  min_periods=1).sum().values.astype(np.float32)
    result['ofi_100'] = ofi_series.rolling(100, min_periods=1).sum().values.astype(np.float32)
    result['ofi_500'] = ofi_series.rolling(500, min_periods=1).sum().values.astype(np.float32)

    # ════════════════════════════════════════════════════════════════════════════
    # NEW FEATURE GROUP 5: STACK SWEEP DETECTION
    # Boolean flag: 3+ consecutive levels where volume dropped >50% (pull/cancel)
    # ════════════════════════════════════════════════════════════════════════════
    # Fractional drop per level: pull / (prev + 1e-9)
    frac_pull_bid = (pull_bid / (prev_bid + 1e-9)).astype(np.float32)
    frac_pull_ask = (pull_ask / (prev_ask + 1e-9)).astype(np.float32)

    # Binary: did this level experience a >50% reduction?
    bid_swept = (frac_pull_bid > 0.5).astype(np.int8)
    ask_swept = (frac_pull_ask > 0.5).astype(np.int8)

    # Count consecutive swept levels starting from L1 (best price)
    # For bid side: check if levels 0,1,2 are all swept
    sweep_bid_3 = ((bid_swept[:, 0] & bid_swept[:, 1] & bid_swept[:, 2])).astype(np.int8)
    sweep_ask_3 = ((ask_swept[:, 0] & ask_swept[:, 1] & ask_swept[:, 2])).astype(np.int8)

    result['stack_sweep_bid_flag'] = sweep_bid_3
    result['stack_sweep_ask_flag'] = sweep_ask_3
    result['stack_sweep_any_flag'] = np.maximum(sweep_bid_3, sweep_ask_3).astype(np.int8)

    # Total swept count across all levels (proxy for sweep intensity)
    result['bid_sweep_count']  = bid_swept.sum(axis=1).astype(np.int8)
    result['ask_sweep_count']  = ask_swept.sum(axis=1).astype(np.int8)

    # ════════════════════════════════════════════════════════════════════════════
    # DEPRECATED / COMMENTED FEATURES (kept as reference for audit)
    # These are state snapshots that diagnostic confirmed as noise (|r| < 0.014):
    #   bid_qty_1, ask_qty_1, bid_depth_5, ask_depth_5, depth_ratio,
    #   imbalance_5, imbalance_10, stacked_imb_*, exhaustion_*, stack/pull per level
    # They are NOT written to output to force tree models onto directional features.
    # ════════════════════════════════════════════════════════════════════════════

    # ── VWAP session deviation (keep — structural macro feature) ───────────────
    # MBO-native: tot_v from DOM quantity drops (no T&S dependency)
    tot_v      = agg_buy + agg_sell
    cum_v_arr  = pd.Series(tot_v).cumsum().values
    cum_pv_arr = pd.Series(mid_price * tot_v).cumsum().values
    vwap       = (state['cum_pv'] + cum_pv_arr) / (state['cum_vol'] + cum_v_arr + 1e-9)
    result['vwap_dev_ticks'] = ((mid_price - vwap) / TICK_SIZE).astype(np.float32)

    # Update state with chunk's contribution (for next chunk's VWAP + MBO continuity)
    net_chunk_v  = cum_v_arr[-1]  - (cum_v_arr[len_prev - 1]  if len_prev > 0 else 0.0)
    net_chunk_pv = cum_pv_arr[-1] - (cum_pv_arr[len_prev - 1] if len_prev > 0 else 0.0)
    # Carry forward last row's per-level bid/ask quantities for MBO trade_delta in next chunk
    last_bid_row = bid_qty_vals[-1].astype(np.float32)
    last_ask_row = ask_qty_vals[-1].astype(np.float32)
    out_state = {
        'cum_vol': state['cum_vol'] + net_chunk_v,
        'cum_pv':  state['cum_pv']  + net_chunk_pv,
        'prev_bid_qty_row': last_bid_row,
        'prev_ask_qty_row': last_ask_row,
    }

    # ── VPIN proxy (keep — known volume-synchronization feature) ───────────────
    imbalance_abs = np.abs(agg_buy - agg_sell)
    v_tot         = agg_buy + agg_sell
    roll_imb100   = pd.Series(imbalance_abs).rolling(100, min_periods=1).sum().values
    roll_tot100   = pd.Series(v_tot).rolling(100, min_periods=1).sum().values
    result['vpin_100'] = (roll_imb100 / (roll_tot100 + 1e-9)).astype(np.float32)

    # ── Session cumulative delta (reset each chunk — for audit) ─────────────────
    result['cum_delta_chunk'] = trade_delta.cumsum().astype(np.float32)

    # ════════════════════════════════════════════════════════════════════════════
    # OUTPUT COLUMNS — directional / momentum features ONLY
    # ════════════════════════════════════════════════════════════════════════════
    output_cols = [
        # Core IDs
        'ts',
        # Price
        'spread', 'mid_price', 'microprice',
        # L1 base
        'imbalance_1',
        # ── NEW: Cumulative Delta ──────────────────────────────────────────────
        'delta_50', 'delta_100', 'delta_200', 'delta_500',
        # ── NEW: Imbalance Trend ──────────────────────────────────────────────
        'imb_trend_10', 'imb_trend_50',
        'imb_ma_ratio_10', 'imb_ma_ratio_50',
        # ── NEW: Microprice Momentum ──────────────────────────────────────────
        'microprice_momentum_10', 'microprice_momentum_50',
        'microprice_dev_from_ma',
        # ── NEW: Directional OFI ───────────────────────────────────────────────
        'ps_delta_L1',
        'ofi_50', 'ofi_100', 'ofi_500',
        # ── NEW: Stack Sweep ───────────────────────────────────────────────────
        'stack_sweep_bid_flag', 'stack_sweep_ask_flag', 'stack_sweep_any_flag',
        'bid_sweep_count', 'ask_sweep_count',
        # Session
        'vwap_dev_ticks', 'vpin_100', 'cum_delta_chunk',
    ]

    # ── Subset to fresh rows only (drop prev_df tail) ─────────────────────────
    final_result = result.iloc[len_prev:].copy()

    # ── Enforce float32 / int8 dtypes ─────────────────────────────────────────
    int8_cols = {
        'stack_sweep_bid_flag', 'stack_sweep_ask_flag', 'stack_sweep_any_flag',
        'bid_sweep_count', 'ask_sweep_count',
    }
    for c in output_cols:
        if c == 'ts':
            continue
        if c in int8_cols:
            final_result[c] = final_result[c].round(0).fillna(0).astype(np.int8)
        else:
            final_result[c] = final_result[c].round(6).astype(np.float32)

    return final_result[output_cols], out_state


def compute_features_chunked(input_path: Path, output_path: Path) -> dict[str, Any]:
    import time
    import shutil

    stats = {
        "rows_processed": 0, "first_ts": None,
        "last_ts": None, "rows_written": 0
    }
    prev_df = None
    state   = None  # compute_features_vectorized handles full init on first call
    t_start = time.time()

    temp_dir   = Path(output_path).parent
    # Clean up any orphaned temp files from a previous interrupted run
    for tf in temp_dir.glob(".p3_chunk_*.tmp"):
        tf.unlink(missing_ok=True)
    temp_files = []
    chunk_num  = 0
    header_written = False
    header_cols    = None

    with pd.read_csv(input_path, chunksize=CHUNK_SIZE, dtype=str) as reader:
        for chunk_df in reader:
            t0 = time.time()
            chunk_num += 1

            for c in chunk_df.columns:
                if c != 'ts':
                    chunk_df[c] = pd.to_numeric(chunk_df[c], errors='coerce')

            features_df, state = compute_features_vectorized(chunk_df, prev_df, state)

            temp_file = temp_dir / f".p3_chunk_{chunk_num}.tmp"
            if not header_written:
                header_cols   = features_df.columns.tolist()
                header_written = True

            features_df.to_csv(temp_file, header=False, index=False)
            temp_files.append(temp_file)

            stats['rows_processed'] += len(chunk_df)
            stats['rows_written']   += len(features_df)

            if stats['first_ts'] is None:
                stats['first_ts'] = chunk_df['ts'].iloc[0]

            # Carry up to 500 rows for rolling windows (delta_500 needs 500)
            prev_df = chunk_df.tail(500).reset_index(drop=True)

            t1 = time.time()
            print(f"  Chunk {chunk_num}: {len(chunk_df):,} rows in {t1-t0:.2f}s "
                  f"(MOMENTUM features: delta/imb_trend/micro_momentum/sweep)")

    print(f"\n  Concatenating {len(temp_files)} temp files ...")
    t_cat = time.time()

    # Write to a hidden temp file first, then rename atomically
    temp_out = temp_dir / f".p3_features_dom.tmp"
    try:
        with open(temp_out, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            writer.writerow(header_cols)
            for tf in temp_files:
                with open(tf, 'r', encoding='utf-8') as fin:
                    shutil.copyfileobj(fin, fout)
                tf.unlink()
        # Atomic rename
        temp_out.replace(output_path)
    except Exception:
        # Clean up on failure: remove partial output and all temp files
        temp_out.unlink(missing_ok=True)
        for tf in temp_files:
            tf.unlink(missing_ok=True)
        raise

    t_cat_end = time.time()
    stats['last_ts'] = prev_df['ts'].iloc[-1] if prev_df is not None else None

    print(f"\n  [P3] Total: {stats['rows_processed']:,} rows in {t_cat_end-t_start:.1f}s")
    return stats


def print_report(stats: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING — PHASE 3 (MOMENTUM / DIRECTIONAL)")
    print("=" * 60)
    print(f"  Processed : {stats['rows_processed']:,}")
    print(f"  First     : {stats['first_ts'] or 'N/A'}")
    print(f"  Last      : {stats['last_ts'] or 'N/A'}")
    print("  Features  : delta_50/100/200/500, imb_trend_10/50,")
    print("              microprice_momentum_10/50, stack_sweep_*, ofi_50/100/500")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test suite
    n = 1200
    np.random.seed(42)
    base_px = 19000.0

    test_df = pd.DataFrame({
        'ts': [f"2026-03-13 {i//10:02d}:{(i%10)*6:02d}:00.000000 UTC" for i in range(n)],
        'best_bid': base_px + np.random.randn(n) * 0.5,
        'best_ask': base_px + 0.25 + np.random.randn(n) * 0.5,
        'spread':   [0.25] * n,
        'mid_price': base_px + np.random.randn(n) * 0.5,
        **{f'bid_qty_{i}': np.random.randint(1, 50, n).astype(float) for i in range(1, 11)},
        **{f'ask_qty_{i}': np.random.randint(1, 50, n).astype(float) for i in range(1, 11)},
        'traded_vol_bid': np.random.randint(0, 10, n).astype(float),
        'traded_vol_ask': np.random.randint(0, 10, n).astype(float),
    })

    chunk1 = test_df.iloc[:600].copy().reset_index(drop=True)
    chunk2 = test_df.iloc[600:].copy().reset_index(drop=True)

    res1, state1 = compute_features_vectorized(chunk1, None)
    res2, state2 = compute_features_vectorized(chunk2, prev_df=chunk1.tail(500), state=state1)

    print("\n[OK] Testing Momentum/Directional P3 Features:")
    print("  Columns:", len(res2.columns))
    print("  Column list:", list(res2.columns))
    print()
    print("  delta_100  (mean):",  res2['delta_100'].mean())
    print("  imb_trend_10 (mean):", res2['imb_trend_10'].mean())
    print("  microprice_momentum_10 (mean):", res2['microprice_momentum_10'].mean())
    print("  stack_sweep_bid_flag (sum):", res2['stack_sweep_bid_flag'].sum())
    print("  ofi_100 (mean):", res2['ofi_100'].mean())
    print("  vpin_100 (mean):", res2['vpin_100'].mean())
    print()
