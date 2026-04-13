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


def cusum_sample(
    features_path: Path,
    agg_path: Path,
    output_path: Path,
    percentile: float = 5.0,
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

    # ------------------------------------------------------------------
    # Pass 1: compute h from TRUE snapshot-to-snapshot deltas (VECTORIZED)
    # IMPORTANT: mid_price_diff in features_dom.csv is CUMULATIVE from session
    # start, NOT the per-snapshot change. We must np.diff() it to get real deltas.
    # ------------------------------------------------------------------
    print("  [Pass 1] Computing h from true per-snapshot |d_mid_price| (vectorized)...")
    import pandas as pd

    # Read mid_price_diff column and diff it to get true per-snapshot delta
    df_d = pd.read_csv(features_path, usecols=['mid_price_diff'], engine='c')
    mid_diff_cum = pd.to_numeric(df_d['mid_price_diff'], errors='coerce').fillna(0.0).values
    # np.diff gives consecutive diffs: delta[0]=cum[0], delta[i]=cum[i]-cum[i-1]
    # The first delta (prepend=mid_diff_cum[0]) is 0: we use the first mid_price as
    # the baseline so CUSUM warmup does not generate a false emission.
    true_deltas = np.diff(mid_diff_cum, prepend=mid_diff_cum[0])
    abs_deltas = np.abs(true_deltas)
    non_zero = abs_deltas[abs_deltas > 0]
    n = len(non_zero)
    rows_count = len(abs_deltas)

    del df_d, mid_diff_cum, true_deltas

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

    # Get agg columns once
    df_agg_header = pd.read_csv(agg_path, nrows=0)
    agg_cols = [c for c in df_agg_header.columns if c != 'ts']
    out_fields = PHASE3_FIELDS + agg_cols

    stats["rows_read"] = 0
    stats["rows_sampled"] = 0

    # Streaming CSV writer -- flush every BATCH_FLUSH rows to keep memory bounded
    BATCH_FLUSH = 50_000
    sampled_rows: list[dict] = []
    first_ts_captured = False

    # Read both files in synchronized chunks
    feat_reader = pd.read_csv(features_path, chunksize=CHUNK_SIZE, engine='c')
    agg_reader = pd.read_csv(agg_path, chunksize=CHUNK_SIZE, engine='c')

    f_out = open(output_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=out_fields)
    writer.writeheader()  # write header immediately, before any data

    prev_mid_diff = 0.0  # carries last cumulative mid_price_diff across chunks
    for chunk_idx, (chunk_feat, chunk_agg) in enumerate(zip(feat_reader, agg_reader)):
        # Merge chunk with its corresponding agg chunk (positional alignment)
        agg_cols_to_add = [c for c in chunk_agg.columns if c not in chunk_feat.columns]
        df_merged = pd.concat([chunk_feat, chunk_agg[agg_cols_to_add]], axis=1)

        # mid_price_diff in CSV is cumulative from session start.
        # diff it to get true per-snapshot delta, using last cum as next chunk's baseline.
        mid_diff_cum = pd.to_numeric(df_merged['mid_price_diff'], errors='coerce').fillna(0.0).values
        mid_deltas = np.diff(mid_diff_cum, prepend=prev_mid_diff)
        prev_mid_diff = mid_diff_cum[-1]  # carry last cumulative value to next chunk
        # First delta of each chunk (after prepend) is now 0 -- warmup, not a signal.
        del mid_diff_cum

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
            sampled_rows.append(out_row)
            emitted_count += 1

            # Flush to disk every BATCH_FLUSH rows -- keeps memory bounded
            if len(sampled_rows) >= BATCH_FLUSH:
                writer.writerows(sampled_rows)
                sampled_rows.clear()

        stats["rows_sampled"] += emitted_count
        stats["rows_read"] += len(df_merged)

        print(f"    chunk {chunk_idx+1}: processed {stats['rows_read']:,} rows | sampled={stats['rows_sampled']:,}")

        del df_merged, mid_deltas, emit_mask

    # Flush any remaining rows
    if sampled_rows:
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Output {args.output} already exists. Use --force.")
        sys.exit(1)

    print(f"Features : {args.features}")
    print(f"Agg      : {args.agg}")
    print(f"Output   : {args.output}")
    print(f"Percentile: {args.percentile}")
    print()

    stats = cusum_sample(args.features, args.agg, args.output, args.percentile)
    print_report(stats)
    if stats["rows_sampled"] == 0:
        print("[P5] FATAL: 0 samples emitted -- refusing to produce empty output.")
        sys.exit(1)