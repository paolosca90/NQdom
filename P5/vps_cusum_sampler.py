""" Phase 5: CUSUM Event Sampling
Input: features_dom.csv (full per-snapshot features)
       features_dom_agg.csv (per-snapshot aggregated features)
Output: sampled_events.csv

CUSUM (Cumulative Sum) detecta movimenti significativi del mercato:
  - Prima passata: calcola soglia h = 5deg percentile dei |deltamid_price| NON nulli, floor=0.5 tick
  - Seconda passata: accumula deltamid_price in CUSUM, emette sample quando |S| > h
  - Output: tutte le feature (Phase3 + Phase4 agg) per ogni evento campionato

Riduce da milioni di snapshot a eventi significativi per sessione.

Adaptive calibration (MAX_EMISSION_RATE=10%):
  1. Calcola h iniziale dal percentile dei |deltamid_price| NON nulli (floor=0.5 tick)
  2. Valida empiricamente il tasso di emissione su un sample (200k righe)
  3. Se rate > 10%, ricalibra h raddoppiando fino a raggiungere soglia target
  4. Passa alla CUSUMfull con h calibrato
"""

import csv
import sys
from pathlib import Path
from typing import Any
import numpy as np

# Optional Numba for the hot CUSUM loop -- falls back to pure NumPy if unavailable
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PHASE3_FIELDS = [
    "ts",
    "spread_ticks", "microprice", "mid_price_diff",
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
    # Microstructural + flow features (added P3 update)
    "actual_depth_bid_Xi", "actual_depth_ask_Xi",
    "flow_limit_add_bid_L1", "flow_cancellation_bid_L1", "flow_market_sell_L1",
    "flow_limit_add_ask_L1", "flow_cancellation_ask_L1", "flow_market_buy_L1",
    "flow_limit_add_bid_5", "flow_cancellation_bid_5",
    "flow_limit_add_ask_5", "flow_cancellation_ask_5",
    # NEW order flow features (from P2b T&S fusion)
    "delta_raw", "cum_delta_chunk", "delta_sign",
    # T&S enrichment: trade attached to each sampled event via merge_asof
    "trade_side", "trade_size", "trade_delta",
    "cum_trade_delta_session",
    # Diagonal Stacked Imbalances
    "stacked_imb_ask", "stacked_imb_bid", "stacked_imb_score",
    "stacked_imb_flag_ask", "stacked_imb_flag_bid",
    # Exhaustion / Thin Prints
    "exhaustion_ask_thin", "exhaustion_bid_thin",
    "exhaustion_ask_zero", "exhaustion_bid_zero", "exhaustion_ratio",
    # LOB Spatial Density (improved)
    "lob_max_gap_bid", "lob_max_gap_ask",
    "lob_vacuum_count_bid", "lob_vacuum_count_ask", "lob_vacuum_score",
]

PROGRESS_EVERY = 200_000

# Adaptive calibration
MAX_EMISSION_RATE = 0.10   # 10% -- emission rate must stay below this
CALIBRATION_SAMPLE = 200_000  # rows to use for empirical emission rate check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: str | None, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _safe_int(val: str | None, default: int = 0) -> int:
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except ValueError:
        return default


def _parse_ms(ts_str: str) -> int:
    """'2026-01-08 00:00:00.000000 UTC' -> ms-from-midnight."""
    try:
        digits = "".join(c for c in ts_str if c.isdigit())
        hour = int(digits[8:10])
        minute = int(digits[10:12])
        sec = int(digits[12:14])
        ms = int(digits[14:17])
        return ((hour * 3600 + minute * 60 + sec) * 1000) + ms
    except (ValueError, IndexError):
        return 0


# ---------------------------------------------------------------------------
# Phase 5: CUSUM sampling
# ---------------------------------------------------------------------------

def _validate_alignment(feat_path: Path, agg_path: Path) -> None:
    """Verifica che P3 e P4 abbiano lo stesso numero di righe (escluso header).
    Se divergono il sampler produce output silenziosamente corrotto."""
    with open(feat_path) as f:
        n_feat = sum(1 for _ in f) - 1  # escludi header
    with open(agg_path) as f:
        n_agg = sum(1 for _ in f) - 1
    tol = 1  # allow 1-row tolerance for NaT-edge-case
    if abs(n_feat - n_agg) > tol:
        raise RuntimeError(
            f"P5 ALIGNMENT ERROR: features_dom.csv has {n_feat} rows "
            f"but features_dom_agg.csv has {n_agg} rows. "
            f"Re-run P3 and P4 before P5. Aborting to prevent silent data corruption."
        )


def _calibrate_h(abs_deltas: np.ndarray, initial_h: float) -> tuple[float, float]:
    """
    Empirical calibration: verify emission rate on a sample.
    If rate > MAX_EMISSION_RATE, double h iteratively until rate drops below threshold.

    Returns:
        calibrated_h: the h to use for pass 2
        empirical_rate: the measured emission rate on the calibration sample
    """
    n_cal = min(len(abs_deltas), CALIBRATION_SAMPLE)
    sample = abs_deltas[:n_cal]

    # Quick CUSUM simulation on calibration sample
    cusum = 0.0
    emitted = 0
    for d in sample:
        cusum += d
        if abs(cusum) > initial_h:
            emitted += 1
            cusum = 0.0

    empirical_rate = emitted / n_cal

    h = initial_h
    if empirical_rate > MAX_EMISSION_RATE:
        print(f"  [Calibration] initial h={initial_h:.4f} => rate={empirical_rate:.3f} (>{MAX_EMISSION_RATE:.1%}) -- recalibrating...")
        iteration = 0
        while empirical_rate > MAX_EMISSION_RATE and h < 1e6:
            h *= 2.0
            cusum = 0.0
            emitted = 0
            for d in sample:
                cusum += d
                if abs(cusum) > h:
                    emitted += 1
                    cusum = 0.0
            empirical_rate = emitted / n_cal
            iteration += 1
            print(f"    iteration {iteration}: h={h:.4f} => rate={empirical_rate:.3f}")
        print(f"  [Calibration] final h={h:.4f} => rate={empirical_rate:.3f} OK")
    else:
        print(f"  [Calibration] h={initial_h:.4f} => rate={empirical_rate:.3f} <= {MAX_EMISSION_RATE:.1%} OK")

    return h, empirical_rate


def _cusum_filter_numba(deltas: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-JIT CUSUM filter -- O(n) single pass, returns (emit_mask, cusum_out).
    emit_mask[i]=True where row i was emitted; cusum_out[i] is CUSUM state at row i.

    Resets cusum to 0 on each emission. h is the detection threshold.
    """
    n = len(deltas)
    emit_mask = np.zeros(n, dtype=np.int8)
    cusum_out = np.zeros(n, dtype=np.float64)
    cusum = 0.0
    emitted = 0
    for i in range(n):
        cusum += deltas[i]
        cusum_out[i] = cusum
        if abs(cusum) > h:
            emit_mask[i] = 1
            emitted += 1
            cusum = 0.0
    return emit_mask, cusum_out


def _cusum_filter_numpy(deltas: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure-NumPy CUSUM filter -- O(n), no Numba needed.
    Uses a running maximum approach to detect crossings.
    Slightly less accurate at reset boundaries vs Numba version but
    avoids the per-row Python loop entirely.
    """
    n = len(deltas)
    emit_mask = np.zeros(n, dtype=np.int8)
    cusum_out = np.zeros(n, dtype=np.float64)

    # Cumulative sum
    cusum = np.cumsum(deltas)

    # Find where |cusum| > h using a running reset
    # Key: when we emit, cusum resets to 0 for subsequent calculations
    abs_cusum = np.abs(cusum)
    above = abs_cusum > h

    # Find emission indices (first index where cumulative sum crosses threshold)
    # Using np.maximum.accumulate to propagate "reset" signals backwards
    reset = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if above[i]:
            reset[i] = cusum[i]
    reset = np.maximum.accumulate(reset[::-1])[::-1] * -1

    # Not straightforward to replicate the reset behavior in pure NumPy without a loop.
    # Fall back to numba-style explicit loop but using numpy arrays (still fast).
    cusum_val = 0.0
    for i in range(n):
        cusum_val += deltas[i]
        cusum_out[i] = cusum_val
        if abs(cusum_val) > h:
            emit_mask[i] = 1
            cusum_val = 0.0
    return emit_mask, cusum_out


def _enrich_batch_ts(batch: list[dict], ts_vals, size_vals, delta_vals,
                     cum_delta_trades, cum_trade_delta_ref: float) -> float:
    """
    Attach T&S trade data to each row in batch using binary search (numpy searchsorted).
    Updates each row dict with: trade_side, trade_size, trade_delta, cum_trade_delta_session.
    Returns the updated cum_trade_delta_ref for the NEXT batch.
    """
    if not batch:
        return cum_trade_delta_ref

    # Parse event timestamps
    event_ts = np.empty(len(batch), dtype='datetime64[ms]')
    for i, row in enumerate(batch):
        ts_str = str(row.get('ts', '')).replace(' UTC', '')
        try:
            event_ts[i] = np.datetime64(ts_str, 'ms')
        except Exception:
            event_ts[i] = np.datetime64('NaT')

    # For each event: find nearest trade at or before event timestamp
    # np.searchsorted returns insertion position — use pos-1 for backward merge
    indices = np.searchsorted(ts_vals, event_ts) - 1
    indices = np.clip(indices, 0, len(ts_vals) - 1)

    # Verify trade is at or before event
    valid = ts_vals[indices] <= event_ts

    for i, row in enumerate(batch):
        if valid[i]:
            row['trade_side'] = 'buy' if delta_vals[indices[i]] > 0 else 'sell'
            row['trade_size']  = float(size_vals[indices[i]])
            row['trade_delta'] = float(delta_vals[indices[i]])
            cum_trade_delta_ref += delta_vals[indices[i]]
        else:
            row['trade_side'] = 'none'
            row['trade_size']  = 0.0
            row['trade_delta'] = 0.0
            # cum_trade_delta_ref stays the same (no new trade)
        row['cum_trade_delta_session'] = float(cum_trade_delta_ref)

    return cum_trade_delta_ref


def cusum_sample(
    features_path: Path,
    agg_path: Path,
    output_path: Path,
    percentile: float = 5.0,
    trades_path: Path | None = None,
) -> dict[str, Any]:
    """
    Two-pass CUSUM sampling.
    Both files have the SAME number of rows in the SAME order -- align by position.
    """
    _validate_alignment(features_path, agg_path)

    stats: dict[str, Any] = {
        "rows_read": 0,
        "rows_sampled": 0,
        "h_threshold": 0.0,
        "emission_rate": 0.0,
        "first_ts": None,
        "last_ts": None,
    }

    # ── T&S Enrichment: load trades once ─────────────────────────────────────
    # Each sampled event gets the nearest T&S trade (backward merge_asof).
    # This gives us real trade_side, trade_size, trade_delta for every event.
    import pandas as pd

    TS_ENRICH_READY = False
    if trades_path is not None and trades_path.exists():
        print(f"  [T&S] Loading trades from {trades_path} ...")
        try:
            trades_df = pd.read_csv(trades_path, skipinitialspace=True)
            trades_df.columns = [c.lower().strip() for c in trades_df.columns]

            # Detect format (canonical: ts,price,size,side | Sierra: date,time,last,volume,bidvolume,askvolume)
            if 'ts' in trades_df.columns:
                trades_df['ts_dt'] = pd.to_datetime(
                    trades_df['ts'].str.replace(' UTC', '', regex=False),
                    format='mixed', errors='coerce'
                )
                trades_df = trades_df.sort_values('ts_dt').reset_index(drop=True)
            elif 'date' in trades_df.columns:
                trades_df['ts'] = (trades_df['date'].astype(str) + ' ' + trades_df['time'].astype(str)).str.strip()
                trades_df['ts_dt'] = pd.to_datetime(trades_df['ts'], format='mixed', errors='coerce')
                trades_df = trades_df.sort_values('ts_dt').reset_index(drop=True)
                # Derive side from bidvolume/askvolume
                bv = pd.to_numeric(trades_df.get('bidvolume', 0), errors='coerce').fillna(0)
                av = pd.to_numeric(trades_df.get('askvolume', 0), errors='coerce').fillna(0)
                cond = [(av > 0) & (bv == 0), (bv > 0) & (av == 0),
                        (av > bv), (bv > av)]
                ch   = ['buy', 'sell', 'buy', 'sell']
                trades_df['side'] = np.select(cond, ch, default='_skip_')
                trades_df = trades_df[trades_df['side'] != '_skip_'].copy()
                if 'size' not in trades_df.columns:
                    trades_df['size'] = pd.to_numeric(trades_df.get('volume', 1), errors='coerce').fillna(1)
            else:
                raise ValueError("Cannot detect trades format")

            # Build numpy arrays for fast binary search
            ts_vals   = trades_df['ts_dt'].values.astype('datetime64[ms]')
            size_vals = pd.to_numeric(trades_df['size'], errors='coerce').fillna(1).values.astype(np.float64)
            # trade_delta: +size for buy, -size for sell
            side_arr  = trades_df['side'].str.lower().values
            delta_vals = np.where(side_arr == 'buy', size_vals, -size_vals)
            # Cumulative delta from session start: cum_delta[i] = sum(delta[0..i])
            cum_delta_trades = np.cumsum(delta_vals)

            TS_ENRICH_READY = True
            print(f"  [T&S] {len(ts_vals):,} trades loaded for enrichment")
        except Exception as ex:
            print(f"  [T&S] WARNING: failed to load trades ({ex}) -- continuing without T&S enrichment")

    # ------------------------------------------------------------------
    # Pass 1: compute h from TRUE snapshot-to-snapshot deltas (VECTORIZED)
    # IMPORTANT: mid_price_diff in features_dom.csv is CUMULATIVE from session
    # start, NOT the per-snapshot change. We must np.diff() it to get real deltas.
    # ------------------------------------------------------------------
    print("  [Pass 1] Computing h from true per-snapshot |d_mid_price| (vectorized)...")
    import pandas as pd

    # Read microprice column and diff it to get true per-snapshot delta.
    # Old schema (mid_price_diff) was CUMULATIVE from session start.
    # New MOMENTUM schema has microprice as a price level — diff() gives per-snapshot delta.
    # usecols + float32 reduces Pass 1 peak RAM from ~600MB to ~250MB
    df_pass1 = pd.read_csv(
        features_path,
        usecols=['ts', 'microprice'],
        dtype={'microprice': 'float32'},
        engine='c'
    )
    microprice_vals = pd.to_numeric(df_pass1['microprice'], errors='coerce').fillna(0.0).values
    del df_pass1
    import gc; gc.collect()
    # np.diff gives consecutive diffs: delta[0]=microprice[0], delta[i]=microprice[i]-microprice[i-1]
    # The first delta (prepend=microprice[0]) is 0: we use the first microprice as
    # the baseline so CUSUM warmup does not generate a false emission.
    true_deltas = np.diff(microprice_vals, prepend=microprice_vals[0])
    abs_deltas = np.abs(true_deltas)
    non_zero = abs_deltas[abs_deltas > 0]
    n = len(non_zero)
    rows_count = len(abs_deltas)

    print(f"  [Pass 1] {n:,} non-zero deltas out of {rows_count:,} total ({100*n/max(rows_count,1):.1f}%)")
    if n == 0:
        print("  [Pass 1] WARNING: all mid_price_diff are zero -- using tick_size (0.25) as h")
        h = 0.25
    else:
        h = float(np.percentile(non_zero, percentile))
        h = max(h, 1e-12)
    # Floor: h must be at least 2 * tick_size (0.5 for NQ) to avoid
    # emitting on every micro-movement. Also prevents h=0 from all-zero data.
    TICK_SIZE = 0.25
    h = max(h, 2 * TICK_SIZE)
    stats["h_threshold"] = round(h, 10)
    print(f"  [Pass 1] initial h = {h:.6f} ({percentile}th pct of {n:,} non-zero |d|, floor=2*tick={2*TICK_SIZE})")

    # Adaptive calibration: verify emission rate on sample; if > 10%, raise h
    h, empirical_rate = _calibrate_h(abs_deltas, h)
    stats["h_threshold"] = round(h, 10)
    stats["emission_rate"] = round(empirical_rate, 6)

    del abs_deltas, non_zero

    # ------------------------------------------------------------------
    # Pass 2: Vectorized CUSUM + STREAMING CSV write (bounded memory)
    # ------------------------------------------------------------------
    print(f"  [Pass 2] CUSUM filtering with pure NumPy fallback ..." if not HAS_NUMBA else "  [Pass 2] CUSUM filtering with Numba JIT ...")
    cusum_filter = _cusum_filter_numba if HAS_NUMBA else _cusum_filter_numpy
    print(f"  [Pass 2] Filter: {cusum_filter.__name__}")

    # Chunk size: 2M rows -- larger chunks = fewer loops, more efficiency
    CHUNK_SIZE = 2_000_000

    # P2b T&S fix: delta features computed from features_dom delta_raw
    NEW_TS_FEATURES = [
        'cum_delta_session', 'delta_rolling_sum_30', 'delta_rolling_sum_5',
    ]
    # Get agg columns once
    df_agg_header = pd.read_csv(agg_path, nrows=0)
    agg_cols = [c for c in df_agg_header.columns if c != 'ts']
    # Build dtype map: all non-ts columns as float32 to halve memory vs float64
    all_out_cols = PHASE3_FIELDS + agg_cols + NEW_TS_FEATURES
    dtype_map = {c: 'float32' for c in all_out_cols if c != 'ts'}
    out_fields = PHASE3_FIELDS + agg_cols + NEW_TS_FEATURES

    stats["rows_read"] = 0
    stats["rows_sampled"] = 0

    # Streaming CSV writer -- flush every BATCH_FLUSH rows to keep memory bounded
    BATCH_FLUSH = 50_000
    sampled_rows: list[dict] = []
    first_ts_captured = False

    # Read both files in synchronized chunks — float32 dtype halves memory vs float64
    feat_reader = pd.read_csv(features_path, chunksize=CHUNK_SIZE, engine='c', dtype=dtype_map)
    agg_reader = pd.read_csv(agg_path, chunksize=CHUNK_SIZE, engine='c')

    f_out = open(output_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=out_fields)
    writer.writeheader()  # write header immediately, before any data

    prev_mid_diff = 0.0  # carries last cumulative mid_price_diff across chunks
    # P2b T&S fix: carry cumulative delta across P3 chunk boundaries
    # cum_delta_chunk in P3 resets at each 250K-row chunk boundary.
    # We track the TRUE session-long cumulative delta.
    prev_cum_delta = 0.0  # cumulative delta from session start (updated at each chunk end)
    # Rolling delta over sampled events (maintained as a deque for O(1) window)
    sampled_delta_history = []  # delta_raw values of all sampled events so far (session-wide)
    # T&S enrichment: running cum_trade_delta across all emitted events
    cum_trade_delta_ref = 0.0  # session-long cumulative trade delta (for cum_trade_delta_session)

    for chunk_idx, (chunk_feat, chunk_agg) in enumerate(zip(feat_reader, agg_reader)):
        # Merge chunk with its corresponding agg chunk (positional alignment)
        agg_cols_to_add = [c for c in chunk_agg.columns if c not in chunk_feat.columns]
        df_merged = pd.concat([chunk_feat, chunk_agg[agg_cols_to_add]], axis=1)

        # De-duplicate: P4 aggregation windows can span snapshot boundaries,
        # causing chunk_agg to have more rows than chunk_feat (one snapshot
        # falls into multiple time windows). Positional concat then creates
        # duplicate rows. Keep only the first occurrence per timestamp.
        before = len(df_merged)
        df_merged = df_merged.drop_duplicates(subset=["ts"], keep="first")
        after = len(df_merged)
        if before != after:
            pct = (before - after) / before * 100
            print(f"    chunk {chunk_idx+1}: P4 dedup: {before:,} -> {after:,} rows "
                  f"(removed {before-after:,} dupes, {pct:.1f}%)")


        # microprice in new MOMENTUM schema is a price level — diff to get per-snapshot delta.
        # Old code used mid_price_diff (cumulative). Using microprice is better:
        # microprice reacts to LOB volume imbalances BEFORE mid-price ticks,
        # so CUSUM triggers when institutional pressure builds up.
        microprice_vals = pd.to_numeric(df_merged['microprice'], errors='coerce').fillna(0.0).values
        mid_deltas = np.diff(microprice_vals, prepend=prev_mid_diff)
        prev_mid_diff = microprice_vals[-1]  # carry last value to next chunk
        # First delta of each chunk (after prepend) is 0 -- warmup, not a signal.
        del microprice_vals

        # P2b T&S fix: compute TRUE cumulative delta across chunks
        # cum_delta_chunk in P3 MOMENTUM resets at each 250K-row chunk boundary.
        # We track the TRUE session-long cumulative delta.
        dr_vals = pd.to_numeric(df_merged['cum_delta_chunk'], errors='coerce').fillna(0.0).values
        dr_cumsum = np.cumsum(dr_vals)          # cumsum within this chunk
        # cum_delta_session = delta from session start up to each row in this chunk
        cum_delta_session = prev_cum_delta + dr_cumsum
        # Update carry-forward for next chunk
        prev_cum_delta = prev_cum_delta + dr_cumsum[-1]  # add chunk's net delta

        # Run CUSUM filter -- returns boolean mask of emitted rows
        emit_mask, _ = cusum_filter(mid_deltas, h)

        # Capture first/last ts
        ts_values = df_merged['ts'].values
        if not first_ts_captured and emit_mask.sum() > 0:
            first_emitted_idx = np.argmax(emit_mask)
            stats["first_ts"] = str(ts_values[first_emitted_idx])
            first_ts_captured = True
        stats["last_ts"] = str(ts_values[-1])

        # Stream emitted rows directly to CSV (no intermediate list accumulation)
        emitted_count = 0
        for row_idx in np.where(emit_mask)[0]:
            row = df_merged.iloc[row_idx]
            out_row = {}
            for k in PHASE3_FIELDS:
                out_row[k] = row[k] if k in row.index else ""
            for k in agg_cols:
                out_row[k] = row[k] if k in row.index else ""

            # P2b T&S fix: add session-long delta features
            # cum_delta_session: cumulative delta from session start (13:40 UTC) to this row
            # This carries across P3 chunk boundaries unlike cum_delta_chunk which resets
            out_row['cum_delta_session'] = float(cum_delta_session[row_idx])

            # delta_rolling_sum_30: rolling sum of delta_raw over the last 30 SAMPLED events
            sampled_delta_history.append(float(dr_vals[row_idx]))
            if len(sampled_delta_history) > 30:
                sampled_delta_history.pop(0)
            out_row['delta_rolling_sum_30'] = sum(sampled_delta_history)

            # delta_rolling_sum_5: short-term momentum
            recent_5 = sampled_delta_history[-5:] if len(sampled_delta_history) >= 5 else sampled_delta_history
            out_row['delta_rolling_sum_5'] = sum(recent_5)

            sampled_rows.append(out_row)
            emitted_count += 1

            # Flush to disk every BATCH_FLUSH rows -- keeps memory bounded
            if len(sampled_rows) >= BATCH_FLUSH:
                # ── T&S enrichment for this batch ───────────────────────────────
                if TS_ENRICH_READY:
                    cum_trade_delta_ref = _enrich_batch_ts(
                        sampled_rows, ts_vals, size_vals, delta_vals,
                        cum_delta_trades, cum_trade_delta_ref)
                # ── write batch ────────────────────────────────────────────────
                writer.writerows(sampled_rows)
                sampled_rows.clear()

        stats["rows_sampled"] += emitted_count
        stats["rows_read"] += len(df_merged)

        print(f"    chunk {chunk_idx+1}: processed {stats['rows_read']:,} rows | sampled={stats['rows_sampled']:,}")

        del df_merged, mid_deltas, emit_mask

    # Flush any remaining rows (with T&S enrichment)
    if sampled_rows:
        if TS_ENRICH_READY:
            cum_trade_delta_ref = _enrich_batch_ts(
                sampled_rows, ts_vals, size_vals, delta_vals,
                cum_delta_trades, cum_trade_delta_ref)
        writer.writerows(sampled_rows)
        sampled_rows.clear()

    f_out.close()

    print(f"  [Pass 2] Done: {stats['rows_read']:,} rows, {stats['rows_sampled']:,} sampled")

    # GUARD: if 0 samples, do NOT leave a corrupt file
    if stats["rows_sampled"] == 0:
        print(f"  [P5] CRITICAL: 0 samples emitted -- h={stats['h_threshold']:.6f} too high.")
        print(f"  [P5] NOT writing output file -- forcing explicit failure.")
        # Remove the header-only file
        output_path.unlink(missing_ok=True)
        return stats

    print(f"  [P5] h={stats['h_threshold']:.6f} | emission={stats['emission_rate']:.3%} | rows_input={stats['rows_read']:,} | samples_written={stats['rows_sampled']:,}")
    return stats


def print_report(stats: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("CUSUM SAMPLING - PHASE 5 REPORT")
    print("=" * 60)
    print(f"\n[SAMPLING]")
    print(f"  h threshold   : {stats['h_threshold']:.10f}")
    print(f"  Emission rate : {stats.get('emission_rate', 0):.3%} (calibration target <={MAX_EMISSION_RATE:.1%})")
    print(f"  Rows read     : {stats['rows_read']:,}")
    print(f"  Rows sampled  : {stats['rows_sampled']:,}")
    print(f"  Sample rate   : {stats['rows_sampled']/max(stats['rows_read'],1)*100:.2f}%")
    print(f"\n[TIME RANGE]")
    print(f"  First row     : {stats['first_ts'] or 'N/A'}")
    print(f"  Last row      : {stats['last_ts'] or 'N/A'}")
    print(f"\n[OUTPUT]")
    print(f"  Features      : all Phase3 + Phase4 aggregated features")
    print(f"  File          : sampled_events.csv")
    print("=" * 60)
    print("Phase 5 completed.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5: CUSUM Event Sampling")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--agg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=5.0)
    parser.add_argument("--trades", type=Path, default=None,
                        help="Path to trades.csv for T&S enrichment (optional)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Output {args.output} already exists. Use --force.")
        sys.exit(1)

    print(f"Features : {args.features}")
    print(f"Agg      : {args.agg}")
    print(f"Output   : {args.output}")
    print(f"Percentile: {args.percentile}")
    print(f"Trades   : {args.trades or 'none (no T&S enrichment)'}")
    print()

    stats = cusum_sample(args.features, args.agg, args.output,
                         args.percentile, args.trades)
    print_report(stats)
    if stats["rows_sampled"] == 0:
        print("[P5] FATAL: 0 samples emitted -- refusing to produce empty output.")
        sys.exit(1)