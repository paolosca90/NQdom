""" Phase 5: CUSUM Event Sampling
Input: features_dom.csv (full per-snapshot features)
       features_dom_agg.csv (per-snapshot aggregated features)
Output: sampled_events.csv

CUSUM (Cumulative Sum) detecta movimenti significativi del mercato:
  - Prima passata: calcola soglia h = 5° percentile dei |Δmid_price| NON nulli, floor=0.5 tick
  - Seconda passata: accumula Δmid_price in CUSUM, emette sample quando |S| > h
  - Output: tutte le feature (Phase3 + Phase4 agg) per ogni evento campionato

Riduce da milioni di snapshot a eventi significativi per sessione.

Adaptive calibration (MAX_EMISSION_RATE=10%):
  1. Calcola h iniziale dal percentile dei |Δmid_price| NON nulli (floor=0.5 tick)
  2. Valida empiricamente il tasso di emissione su un sample (200k righe)
  3. Se rate > 10%, ricalibra h raddoppiando fino a raggiungere soglia target
  4. Passa alla CUSUMfull con h calibrato
"""

import csv
import sys
from pathlib import Path
from typing import Any
import numpy as np

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
MAX_EMISSION_RATE = 0.10   # 10% — emission rate must stay below this
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
    if n_feat != n_agg:
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
        print(f"  [Calibration] initial h={initial_h:.4f} → rate={empirical_rate:.3f} (>{MAX_EMISSION_RATE:.1%}) — recalibrating...")
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
            print(f"    iteration {iteration}: h={h:.4f} → rate={empirical_rate:.3f}")
        print(f"  [Calibration] final h={h:.4f} → rate={empirical_rate:.3f} ✓")
    else:
        print(f"  [Calibration] h={initial_h:.4f} → rate={empirical_rate:.3f} ≤ {MAX_EMISSION_RATE:.1%} ✓")

    return h, empirical_rate


def cusum_sample(
    features_path: Path,
    agg_path: Path,
    output_path: Path,
    percentile: float = 5.0,
) -> dict[str, Any]:
    """
    Two-pass CUSUM sampling.
    Both files have the SAME number of rows in the SAME order — align by position.
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
    # Pass 1: compute h from NON-ZERO absolute mid_price_diff (VECTORIZED)
    # ------------------------------------------------------------------
    print("  [Pass 1] Computing h from non-zero |d_mid_price| (vectorized)...")
    import pandas as pd

    # Read only mid_price_diff column — vectorized parse, no row-by-row Python
    df_d = pd.read_csv(features_path, usecols=['mid_price_diff'], engine='c')

    # Vectorized: to_numeric (handles empty strings), abs, filter non-zero
    mid_diff = pd.to_numeric(df_d['mid_price_diff'], errors='coerce').fillna(0.0)
    abs_deltas = np.abs(mid_diff.values)
    non_zero = abs_deltas[abs_deltas > 0]
    n = len(non_zero)
    rows_count = len(abs_deltas)

    del df_d, mid_diff

    print(f"  [Pass 1] {n:,} non-zero deltas out of {rows_count:,} total ({100*n/max(rows_count,1):.1f}%)")
    if n == 0:
        print("  [Pass 1] WARNING: all mid_price_diff are zero — using tick_size (0.25) as h")
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
    # Pass 2: CUSUM filtering with positional alignment (CHUNKED for safety)
    # ------------------------------------------------------------------
    print("  [Pass 2] CUSUM filtering with positional alignment (chunked)...")

    import pandas as pd

    # Chunk size: 1M rows — fits in RAM, good I/O balance
    CHUNK_SIZE = 1_000_000

    # Get agg columns once
    df_agg_header = pd.read_csv(agg_path, nrows=0)
    agg_cols = [c for c in df_agg_header.columns if c != 'ts']
    agg_all_cols = ['ts'] + agg_cols

    cusum: float = 0.0
    stats["rows_read"] = 0
    out_fields = PHASE3_FIELDS + agg_cols

    # Buffer rows in memory; only write to disk if non-empty.
    # This prevents the "header-only" file problem that cascades silently.
    sampled_rows: list[dict] = []

    # Read both files in synchronized chunks
    feat_reader = pd.read_csv(features_path, chunksize=CHUNK_SIZE, engine='c')
    agg_reader = pd.read_csv(agg_path, chunksize=CHUNK_SIZE, engine='c')

    for chunk_idx, (chunk_feat, chunk_agg) in enumerate(zip(feat_reader, agg_reader)):
        # Merge chunk with its corresponding agg chunk (positional alignment)
        agg_cols_to_add = [c for c in chunk_agg.columns if c not in chunk_feat.columns]
        df_merged = pd.concat([chunk_feat, chunk_agg[agg_cols_to_add]], axis=1)

        # Iterate with itertuples (fast)
        for row in df_merged.itertuples(index=False):
            ts_str = row.ts
            mid_delta = _safe_float(getattr(row, "mid_price_diff", "0.0"), 0.0)
            cusum += mid_delta

            if stats["rows_read"] % PROGRESS_EVERY == 0:
                print(f"    Pass 2: {stats['rows_read']:,} rows | "
                      f"CUSUM={cusum:.6f} | sampled={stats['rows_sampled']}")

            if abs(cusum) > h:
                out_row = {k: getattr(row, k, "") for k in PHASE3_FIELDS}
                for k in agg_cols:
                    out_row[k] = getattr(row, k, "")
                sampled_rows.append(out_row)
                stats["rows_sampled"] += 1
                cusum = 0.0  # reset on detection

            if stats["first_ts"] is None:
                stats["first_ts"] = ts_str
            stats["last_ts"] = ts_str
            stats["rows_read"] += 1

        del df_merged

    print(f"  [Pass 2] Done: {stats['rows_read']:,} rows, {stats['rows_sampled']:,} sampled")

    # GUARD: if 0 samples, do NOT write output file — this breaks the silent
    # cascade where an empty file would cause P6→P7 to fail without clear signal.
    if stats["rows_sampled"] == 0:
        print(f"  [P5] CRITICAL: 0 samples emitted — h={stats['h_threshold']:.6f} too high.")
        print(f"  [P5] NOT writing output file — forcing explicit failure.")
        return stats

    # Write only if we have actual data rows
    print(f"  [P5] Writing {len(sampled_rows):,} samples to {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(f"  [P5] h={stats['h_threshold']:.6f} | emission={stats['emission_rate']:.3%} | rows_input={stats['rows_read']:,} | samples_written={len(sampled_rows):,}")
    return stats


def print_report(stats: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("CUSUM SAMPLING - PHASE 5 REPORT")
    print("=" * 60)
    print(f"\n[SAMPLING]")
    print(f"  h threshold   : {stats['h_threshold']:.10f}")
    print(f"  Emission rate : {stats.get('emission_rate', 0):.3%} (calibration target ≤{MAX_EMISSION_RATE:.1%})")
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
        print("[P5] FATAL: 0 samples emitted — refusing to produce empty output.")
        sys.exit(1)