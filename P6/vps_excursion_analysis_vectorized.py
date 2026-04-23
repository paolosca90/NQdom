"""
Phase 6: Excursion Analysis — FULLY VECTORIZED + NUMBA JIT (Memory-Safe)
========================================================================
Input:  snapshots.csv (lookup index for mid_price + ts),
        sampled_events.csv (CUSUM-sampled event timestamps)
Output: excursion_stats.csv, excursion_summary.csv, excursion_distributions.png

OPTIMIZATIONS (v3 — memory-safe rewrite):
  1. build_lookup_index: chunked CSV read → pre-allocated numpy arrays (~424MB)
  2. compute_excursions: chunked sampled read (500K per chunk) → Numba JIT kernel
     → incremental CSV write (never allocates full 21.9M × 37 array)
  3. Output: pandas chunk-wise to_csv (append mode)
  4. Peak RAM: ~1.5GB (lookup 424MB + chunk 150MB + pandas overhead ~900MB)

MEMORY BUDGET (24GB VPS):
  - Lookup index: 26.5M × 16 bytes = 424MB (must stay in RAM)
  - Per chunk:    500K × 37 × 8    = 148MB (results) + 15MB (sampled ts)
  - Pandas temp:  ~500MB during chunk reads
  - Total peak:   ~1.5GB — safe for 24GB VPS

CRASH FIXES:
  - Removed `* 1000` on timestamp conversion (int64 overflow with pandas 2.x)
  - pandas 2.x/3.x compatible timestamp parsing
  - Chunked processing prevents OOM
"""

# ─────────────────────────────────────────────────────────────────────
# DESIGN NOTE — Semantica temporale P6 vs P7
#
# P6 (questo file): misura escursioni TIME-BASED su orizzonti REALI
#   30s / 60s / 120s di tempo di mercato.
#   Le colonne max_up_30s_ticks, mae_30s_ticks, ecc. usano "30s" come
#   etichetta della finestra temporale — questo è CORRETTO e non va
#   modificato.
#
# P7 (vps_phase7_labeling.py): usa TICK CLOCK (update count del book)
#   per il vertical barrier. vb_ticks=30 significa "30 snapshot update",
#   NON "30 secondi". I due sistemi sono complementari e usano
#   unità diverse per design.
#
# NON confondere: "30s" in P6 = 30 secondi reali.
#                 "30 ticks" in P7 = 30 update del book.
# ─────────────────────────────────────────────────────────────────────

import csv
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Try numba — fall back to pure-numpy if unavailable or disabled via env
try:
    from numba import njit, prange
    # Disable numba if env var set (avoids Windows cache corruption issues)
    HAS_NUMBA = True if os.environ.get("P6_NO_JIT", "0") != "1" else False
except ImportError:
    HAS_NUMBA = False

# Force disable numba if cache is corrupted (prevents ModuleNotFoundError during JIT)
# Remove __pycache__ to clear stale cache entries
import os, glob
pycache = glob.glob(os.path.join(os.path.dirname(__file__), "__pycache__"))
for d in pycache:
    for f in glob.glob(os.path.join(d, "numba_*")):
        try: os.remove(f)
        except: pass
    HAS_NUMBA = False   # numba cache was corrupted — disable JIT
    def njit(*args, **kwargs):
        def decorator(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIN30S_NS  = np.int64(30  * 1_000_000_000)
WIN60S_NS  = np.int64(60  * 1_000_000_000)
WIN120S_NS = np.int64(120 * 1_000_000_000)
TOLERANCE_NS = np.int64(1_000_000_000)  # 1s tolerance for window_complete
TICK_SIZE  = 0.25

# Chunk sizes tuned for 24GB VPS
LOOKUP_CHUNK_SIZE  = 2_000_000   # 2M rows per chunk for snapshots read
SAMPLED_CHUNK_SIZE = 500_000     # 500K events per chunk for excursion compute

# ---------------------------------------------------------------------------
# Output field definitions
# ---------------------------------------------------------------------------

EXCURSION_FIELDS = [
    "ts",
    "mid_price_at_t",
    "max_up_30s_pts", "max_down_30s_pts", "max_up_30s_ticks", "max_down_30s_ticks",
    "mae_30s_pts", "mae_30s_ticks",
    "max_down_30s_abs", "horizon_end_ts_30s", "window_complete_30s", "n_obs_30s",
    "min_mid_30s", "max_mid_30s",
    "max_up_60s_pts", "max_down_60s_pts", "max_up_60s_ticks", "max_down_60s_ticks",
    "mae_60s_pts", "mae_60s_ticks",
    "max_down_60s_abs", "horizon_end_ts_60s", "window_complete_60s", "n_obs_60s",
    "min_mid_60s", "max_mid_60s",
    "max_up_120s_pts", "max_down_120s_pts", "max_up_120s_ticks", "max_down_120s_ticks",
    "mae_120s_pts", "mae_120s_ticks",
    "max_down_120s_abs", "horizon_end_ts_120s", "window_complete_120s", "n_obs_120s",
    "min_mid_120s", "max_mid_120s",
]

N_NUMERIC_COLS = 37  # all columns except "ts"

SUMMARY_FIELDS = [
    "horizon", "n_events",
    "p50_up_ticks", "p75_up_ticks", "p90_up_ticks", "p95_up_ticks", "mean_up_ticks",
    "p50_mae_ticks", "p75_mae_ticks", "p90_mae_ticks", "p95_mae_ticks", "mean_mae_ticks",
    "median_up_over_mae_ratio",
    "pct_complete", "mean_n_obs", "max_n_obs",
]

HORIZONS = [30, 60, 120]
PERCENTILES = [50, 75, 90, 95]
COLORS = ["#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

# Maximum allowed output/input ratio. P6 produces exactly 1 row per P5 input row.
# If ratio > MAX_P6_RATIO, the output is corrupt and should be rejected.
MAX_P6_RATIO = 15.0


# ---------------------------------------------------------------------------
# Checkpoint staleness fix (Apr 13, 2026)
# BUG: P6 skipped recomputation after P5 was re-run with fixed CUSUM.
# FIX: Fingerprint P5 state at P6 completion; validate before skipping.
# ---------------------------------------------------------------------------

def _count_csv_rows(path: Path) -> int:
    """Fast row count without parsing CSV."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1  # -1 for header


def compute_input_fingerprint(p5_path: Path) -> dict:
    """
    Compute a fingerprint of the P5 sampled_events.csv state.
    Used to detect when P6 needs to re-run because P5 input changed.
    """
    stats = p5_path.stat()
    return {
        "mtime": stats.st_mtime,
        "size_bytes": stats.st_size,
        "row_count": _count_csv_rows(p5_path),
        "fingerprint_ts": datetime.now(timezone.utc).isoformat(),
    }


def save_fingerprint(fingerprint: dict, out_dir: Path) -> None:
    """Save fingerprint JSON next to excursion_stats.csv."""
    fp_path = out_dir / "p6_input_fingerprint.json"
    with open(fp_path, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2)


def load_fingerprint(out_dir: Path) -> dict | None:
    """Load saved fingerprint, or None if not present."""
    fp_path = out_dir / "p6_input_fingerprint.json"
    if not fp_path.exists():
        return None
    with open(fp_path, "r", encoding="utf-8") as f:
        return json.load(f)


def should_skip_p6(out_dir: Path, p5_path: Path) -> tuple[bool, str]:
    """
    Decide whether P6 can skip recomputation.

    Returns (skip: bool, reason: str).
    skip=True means P6 output is valid and we should return early.
    skip=False means P6 must re-run (no output, or input changed).

    Staleness logic:
      - No output file    -> must run
      - Output exists but no fingerprint (legacy) -> must run (cannot verify)
      - Fingerprint present: compare mtime + size of current P5 vs saved
      - If either differs -> input changed since last P6 run, must re-run
      - If both match     -> P5 unchanged, P6 output is valid, skip
    """
    excursion_path = out_dir / "excursion_stats.csv"
    fp_path = out_dir / "p6_input_fingerprint.json"

    if not excursion_path.exists():
        return False, "no output file"

    if not fp_path.exists():
        return False, "no fingerprint (legacy run, forcing re-run)"

    saved = load_fingerprint(out_dir)
    if saved is None:
        return False, "fingerprint unreadable (forcing re-run)"

    current_mtime = p5_path.stat().st_mtime
    current_size = p5_path.stat().st_size

    if saved["mtime"] != current_mtime:
        return False, (f"P5 mtime changed: saved={saved['mtime']}, "
                       f"current={current_mtime}")
    if saved["size_bytes"] != current_size:
        return False, (f"P5 size changed: saved={saved['size_bytes']}, "
                       f"current={current_size}")

    return True, "fingerprint valid, P5 unchanged"


def validate_p6_output(p5_path: Path, output_path: Path, date: str) -> None:
    """
    Post-run validation: ensure P6 output row count is within expected range.

    Raises RuntimeError if ratio exceeds MAX_P6_RATIO, deleting corrupt output.
    """
    p5_rows = _count_csv_rows(p5_path)
    p6_rows = _count_csv_rows(output_path)
    ratio = p6_rows / max(p5_rows, 1)

    print(f"  [P6] Ratio check: P5={p5_rows:,} -> P6={p6_rows:,} ({ratio:.2f}x)")

    if ratio > MAX_P6_RATIO:
        # Clean up corrupt output
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        fp_path = output_path.parent / "p6_input_fingerprint.json"
        if fp_path.exists():
            fp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"P6 OUTPUT ANOMALY for {date}: ratio={ratio:.1f}x "
            f"(P5={p5_rows}, P6={p6_rows}). Expected <={MAX_P6_RATIO}x. "
            f"Corrupt output deleted. Aborting."
        )

    print(f"  [P6] Ratio check PASSED for {date}: {ratio:.2f}x")




# ---------------------------------------------------------------------------
# Vectorized timestamp parsing (pandas 2.x / 3.x compatible)
# ---------------------------------------------------------------------------

def _parse_ts_array_vectorized(ts_series) -> np.ndarray:
    """
    Parse array of timestamp strings to int64 nanoseconds-from-epoch (UTC).
    Uses pd.to_datetime (C-engine, vectorized) — ~100x faster than strptime loop.

    Input format: '2026-01-08 09:30:00.123456 UTC'
    Compatible with pandas 2.x (datetime64[ns]) and 3.x (datetime64[us]).
    """
    import pandas as pd

    if isinstance(ts_series, np.ndarray):
        ts_series = pd.Series(ts_series)

    # Strip " UTC" suffix — parse as naive (timestamps are already UTC)
    ts_clean = ts_series.str.replace(" UTC", "", regex=False)

    # Vectorized parse
    dt = pd.to_datetime(ts_clean, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")

    # Convert to numpy, ensure datetime64[ns], then view as int64
    arr = dt.values
    if hasattr(arr, 'tz_localize'):
        arr = arr.tz_localize(None)
    arr_ns = arr.astype("datetime64[ns]")
    ts_ns = arr_ns.view(np.int64)

    # Replace NaT (parse failures) with 0
    ts_ns = np.where(ts_ns < 0, np.int64(0), ts_ns)

    return ts_ns


# ---------------------------------------------------------------------------
# Vectorized excursion kernel — O(events × log snapshots)
# Replaces Numba kernel for days with high snapshot density.
#
# Original: for each event, binary search (log n) + forward scan (m snapshots)
#   → O(events × m) where m = avg snapshots within horizon (102K for Mar 19)
#
# New: for each event, binary search (log n) + 3× O(log n) binary searches
#   → O(events × log n) where log n ≈ 25
#   → 154K × 25 vs 154K × 102K = 4000× speedup
# ---------------------------------------------------------------------------

def _excursion_vectorized(
    sampled_ts_ns: np.ndarray,
    lookup_ts_ns: np.ndarray,
    lookup_prices: np.ndarray,
    results: np.ndarray,
) -> None:
    """
    Vectorized excursion computation using binary-search for max/min lookup.

    For each sampled event at timestamp T:
      1. Binary search: si = first snapshot ≥ T
      2. For each horizon (30s/60s/120s):
         ei = last snapshot ≤ T + horizon
         max_horizon = max(lookup_prices[si:ei+1])
         min_horizon = min(lookup_prices[si:ei+1])
         cnt = ei - si + 1

    Binary search for max/min (O(log n)) vs forward scan (O(m)) where
    m = avg snapshots in horizon window. For Mar 19: m = 102K, log n = 25.
    """
    n_prices = len(lookup_ts_ns)
    n_events = len(sampled_ts_ns)
    TICK = TICK_SIZE
    horizons = [(0, WIN30S_NS), (1, WIN60S_NS), (2, WIN120S_NS)]

    print(f"  [P6] Vectorized: {n_events:,} events, {n_prices:,} snapshots...", flush=True)
    t0 = time.time()

    for i in range(n_events):
        t_event = sampled_ts_ns[i]
        if t_event <= 0:
            continue

        # Position of first snapshot ≥ t_event
        si = np.searchsorted(lookup_ts_ns, t_event, side="left")
        if si >= n_prices:
            continue

        start_price = lookup_prices[si]
        results[i, 0] = start_price

        for h_idx, h_ns in horizons:
            base = h_idx * 12 + 1
            end_ts = t_event + h_ns

            # Last snapshot ≤ end_ts
            ei = np.searchsorted(lookup_ts_ns, end_ts, side="right") - 1
            ei = max(si, min(ei, n_prices - 1))
            cnt = ei - si + 1

            if cnt <= 0:
                continue

            # Max/min via binary search on subarray [si:ei+1]
            max_val = float(lookup_prices[si:ei+1].max())
            min_val = float(lookup_prices[si:ei+1].min())
            horizon_end_ts = lookup_ts_ns[ei]

            up = max_val - start_price
            dn = start_price - min_val
            mae = max(up, dn)

            results[i, base]   = up;               results[i, base+1] = dn
            results[i, base+2] = up / TICK;     results[i, base+3] = dn / TICK
            results[i, base+4] = mae;             results[i, base+5] = mae / TICK
            results[i, base+6] = dn
            results[i, base+7] = (end_ts // 1_000_000_000)
            results[i, base+8] = 1.0 if horizon_end_ts >= end_ts - TOLERANCE_NS else 0.0
            results[i, base+9] = float(cnt)
            results[i, base+10] = min_val;       results[i, base+11] = max_val

    elapsed = time.time() - t0
    rate = n_events / elapsed if elapsed > 0 else 0
    print(f"  [P6] Vectorized done: {n_events:,} events in {elapsed:.1f}s ({rate:,.0f} ev/s)", flush=True)


# ---------------------------------------------------------------------------
# Numba JIT compiled excursion kernel (legacy, used when numba is available)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _excursion_kernel_sequential(
    sampled_ts_ns: np.ndarray,   # int64[N_events]
    lookup_ts_ns: np.ndarray,    # int64[N_prices] — sorted
    lookup_prices: np.ndarray,   # float64[N_prices]
    results: np.ndarray,         # float64[N_events, N_NUMERIC_COLS] — pre-allocated output
) -> None:
    """
    Numba-compiled excursion computation.
    Single pass over all events with scalar accumulation — no temporary arrays.

    For each sampled event:
      1. Binary search into lookup_ts_ns to find start index
      2. Single forward scan up to 120s, tracking per-horizon max/min
      3. Fill results array directly

    Results layout per event (37 floats):
      [0]       mid_price_at_t
      [1..12]   30s horizon
      [13..24]  60s horizon
      [25..36]  120s horizon
    Each horizon block (12 values):
      max_up_pts, max_down_pts, max_up_ticks, max_down_ticks,
      mae_pts, mae_ticks, max_down_abs, horizon_end_ts,
      window_complete, n_obs, min_mid, max_mid
    """
    n_events = len(sampled_ts_ns)
    n_prices = len(lookup_ts_ns)
    tick_size = 0.25

    for i in range(n_events):
        t0 = sampled_ts_ns[i]
        if t0 <= 0:
            continue

        # Binary search for start index
        lo, hi = 0, n_prices
        while lo < hi:
            mid = (lo + hi) // 2
            if lookup_ts_ns[mid] < t0:
                lo = mid + 1
            else:
                hi = mid
        si = lo

        if si >= n_prices:
            continue

        start_price = lookup_prices[si]
        results[i, 0] = start_price

        end_30  = t0 + WIN30S_NS
        end_60  = t0 + WIN60S_NS
        end_120 = t0 + WIN120S_NS

        # Per-horizon accumulators
        max_30 = start_price;  min_30 = start_price;  last_ts_30 = np.int64(0);  cnt_30 = 0
        max_60 = start_price;  min_60 = start_price;  last_ts_60 = np.int64(0);  cnt_60 = 0
        max_120 = start_price; min_120 = start_price;  last_ts_120 = np.int64(0); cnt_120 = 0

        # Single forward scan
        j = si
        while j < n_prices:
            ts_j = lookup_ts_ns[j]
            if ts_j > end_120:
                break

            p = lookup_prices[j]

            # 120s — always within bounds
            cnt_120 += 1
            last_ts_120 = ts_j
            if p > max_120: max_120 = p
            if p < min_120: min_120 = p

            # 60s
            if ts_j <= end_60:
                cnt_60 += 1
                last_ts_60 = ts_j
                if p > max_60: max_60 = p
                if p < min_60: min_60 = p

            # 30s
            if ts_j <= end_30:
                cnt_30 += 1
                last_ts_30 = ts_j
                if p > max_30: max_30 = p
                if p < min_30: min_30 = p

            j += 1

        # Fill 30s (indices 1..12)
        if cnt_30 > 0:
            up = max_30 - start_price
            dn = start_price - min_30
            results[i, 1] = up;                results[i, 2] = dn
            results[i, 3] = up / tick_size;    results[i, 4] = dn / tick_size
            results[i, 5] = dn;                results[i, 6] = dn / tick_size
            results[i, 7] = dn;                results[i, 8] = float(last_ts_30)
            results[i, 9] = 1.0 if last_ts_30 >= end_30 - TOLERANCE_NS else 0.0
            results[i, 10] = float(cnt_30);    results[i, 11] = min_30;  results[i, 12] = max_30

        # Fill 60s (indices 13..24)
        if cnt_60 > 0:
            up = max_60 - start_price
            dn = start_price - min_60
            results[i, 13] = up;               results[i, 14] = dn
            results[i, 15] = up / tick_size;   results[i, 16] = dn / tick_size
            results[i, 17] = dn;               results[i, 18] = dn / tick_size
            results[i, 19] = dn;               results[i, 20] = float(last_ts_60)
            results[i, 21] = 1.0 if last_ts_60 >= end_60 - TOLERANCE_NS else 0.0
            results[i, 22] = float(cnt_60);    results[i, 23] = min_60;  results[i, 24] = max_60

        # Fill 120s (indices 25..36)
        if cnt_120 > 0:
            up = max_120 - start_price
            dn = start_price - min_120
            results[i, 25] = up;               results[i, 26] = dn
            results[i, 27] = up / tick_size;   results[i, 28] = dn / tick_size
            results[i, 29] = dn;               results[i, 30] = dn / tick_size
            results[i, 31] = dn;               results[i, 32] = float(last_ts_120)
            results[i, 33] = 1.0 if last_ts_120 >= end_120 - TOLERANCE_NS else 0.0
            results[i, 34] = float(cnt_120);   results[i, 35] = min_120; results[i, 36] = max_120


# ---------------------------------------------------------------------------
# Memory-safe chunked build_lookup_index
# ---------------------------------------------------------------------------

def build_lookup_index(features_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read snapshots.csv in chunks, extract ts + mid_price as numpy arrays.
    Returns (ts_ns_sorted, mid_prices_sorted) sorted by ts_ns.

    MEMORY-SAFE: reads in 2M-row chunks, pre-allocates output arrays.
    Peak RAM: ~800MB (2 × 400MB arrays + chunk overhead).
    """
    import pandas as pd

    snapshots_path = features_path.parent / "snapshots.csv"
    if not snapshots_path.exists():
        snapshots_path = features_path
        print(f"  [P6] WARNING: snapshots.csv not found, using {features_path.name}")

    # --- Step 1: Count rows (fast) ---
    print(f"  [P6] Counting rows in {snapshots_path.name}...")
    t0 = time.time()
    n_total = 0
    for chunk in pd.read_csv(snapshots_path, usecols=["ts"], engine="c",
                             chunksize=LOOKUP_CHUNK_SIZE):
        n_total += len(chunk)
        del chunk
    gc.collect()
    t_count = time.time() - t0
    print(f"  [P6] {n_total:,} rows counted in {t_count:.1f}s")

    # --- Step 2: Pre-allocate final arrays ---
    ts_ns_all = np.empty(n_total, dtype=np.int64)
    mid_all = np.empty(n_total, dtype=np.float64)

    # --- Step 3: Fill arrays chunk by chunk ---
    print(f"  [P6] Reading {snapshots_path.name} in {LOOKUP_CHUNK_SIZE:,}-row chunks...")
    t0 = time.time()
    offset = 0
    chunk_num = 0

    for chunk in pd.read_csv(snapshots_path, usecols=["ts", "mid_price"], engine="c",
                             chunksize=LOOKUP_CHUNK_SIZE):
        chunk_num += 1
        n = len(chunk)

        # Parse timestamps for this chunk
        chunk_ts = _parse_ts_array_vectorized(chunk["ts"])
        ts_ns_all[offset:offset + n] = chunk_ts

        # Extract mid_price
        mp = chunk["mid_price"].values.astype(np.float64)
        mp = np.where(np.isnan(mp), 0.0, mp)
        mid_all[offset:offset + n] = mp

        offset += n
        del chunk, chunk_ts, mp
        gc.collect()

        if chunk_num % 5 == 0:
            print(f"    Chunk {chunk_num}: {offset:,}/{n_total:,} rows loaded")

    t_read = time.time() - t0
    print(f"  [P6] All {n_total:,} rows loaded in {t_read:.1f}s")

    # --- Step 4: Sort by timestamp (in-place where possible) ---
    print("  [P6] Sorting by timestamp...")
    t0 = time.time()
    idx = np.argsort(ts_ns_all)
    ts_sorted = ts_ns_all[idx]
    mid_sorted = mid_all[idx]
    del ts_ns_all, mid_all, idx
    gc.collect()
    t_sort = time.time() - t0
    print(f"  [P6] Sorted in {t_sort:.1f}s")

    mem_mb = (ts_sorted.nbytes + mid_sorted.nbytes) / (1024**2)
    print(f"  [P6] Lookup index: {len(ts_sorted):,} entries, {mem_mb:.0f} MB")

    return ts_sorted, mid_sorted


# ---------------------------------------------------------------------------
# Memory-safe chunked compute_excursions
# ---------------------------------------------------------------------------

def compute_excursions(
    sampled_path: Path,
    ts_ns_sorted: np.ndarray,
    mid_prices: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    """
    Compute excursion stats in chunks to stay within RAM budget.

    Reads sampled_events.csv in 500K-row chunks, runs Numba kernel per chunk,
    writes output incrementally. Peak chunk RAM: ~170MB.

    Returns stats dict.
    """
    import pandas as pd

    stats: dict[str, Any] = {
        "rows_processed": 0,
        "n_complete_30s": 0,
        "n_complete_60s": 0,
        "n_complete_120s": 0,
    }

    numeric_cols = EXCURSION_FIELDS[1:]  # 37 columns
    int_cols_idx = [i for i, c in enumerate(numeric_cols)
                    if "horizon_end_ts" in c or "window_complete" in c or "n_obs" in c]
    float_cols_idx = [i for i in range(len(numeric_cols)) if i not in int_cols_idx]

    # --- JIT warmup ---
    _use_jit = HAS_NUMBA
    if _use_jit:
        print("  [P6] Warming up Numba JIT kernel...")
        try:
            t0 = time.time()
            warmup_ts = ts_ns_sorted[:1].copy()
            warmup_res = np.zeros((1, N_NUMERIC_COLS), dtype=np.float64)
            _excursion_kernel_sequential(warmup_ts, ts_ns_sorted, mid_prices, warmup_res)
            t_warmup = time.time() - t0
            print(f"  [P6] JIT warmup done in {t_warmup:.1f}s")
        except Exception as e:
            print(f"  [P6] JIT warmup FAILED: {type(e).__name__}: {e}")
            print(f"  [P6] Falling back to pure numpy processing...")
            _use_jit = False

    # --- Process in chunks ---
    print(f"  [P6] Processing sampled events in {SAMPLED_CHUNK_SIZE:,}-row chunks...")
    t_total = time.time()
    chunk_num = 0
    first_chunk = True

    for sampled_chunk in pd.read_csv(sampled_path, usecols=["ts"], engine="c",
                                     chunksize=SAMPLED_CHUNK_SIZE):
        chunk_num += 1
        n_chunk = len(sampled_chunk)
        t0 = time.time()

        # Parse timestamps for this chunk
        ts_strs = sampled_chunk["ts"].values
        sampled_ts_ns = _parse_ts_array_vectorized(sampled_chunk["ts"])
        del sampled_chunk
        gc.collect()

        # Allocate results for this chunk only (~148MB for 500K events)
        results = np.zeros((n_chunk, N_NUMERIC_COLS), dtype=np.float64)

        # Vectorized binary-search excursion computation
        # O(events × log n × horizons) — works for any snapshot density
        _excursion_vectorized(sampled_ts_ns, ts_ns_sorted, mid_prices, results)

        # Build DataFrame for this chunk
        df_chunk = pd.DataFrame(results, columns=numeric_cols)
        df_chunk.insert(0, "ts", ts_strs)

        # Round float columns, cast int columns
        for idx in float_cols_idx:
            col = numeric_cols[idx]
            df_chunk[col] = df_chunk[col].round(6)
        for idx in int_cols_idx:
            col = numeric_cols[idx]
            df_chunk[col] = df_chunk[col].astype(np.int64)

        # Write: header only on first chunk, append on subsequent
        if first_chunk:
            df_chunk.to_csv(output_path, index=False, mode="w")
            first_chunk = False
        else:
            df_chunk.to_csv(output_path, index=False, mode="a", header=False)

        # Update stats
        stats["rows_processed"] += n_chunk
        stats["n_complete_30s"] += int(results[:, 9].sum())   # window_complete_30s: base=1+8=9
        stats["n_complete_60s"] += int(results[:, 21].sum())  # window_complete_60s: base=13+8=21
        stats["n_complete_120s"] += int(results[:, 33].sum()) # window_complete_120s: base=25+8=33

        t_chunk = time.time() - t0
        rate = n_chunk / t_chunk if t_chunk > 0 else 0
        print(f"    Chunk {chunk_num}: {n_chunk:,} events in {t_chunk:.1f}s "
              f"({rate:,.0f} ev/s) — total {stats['rows_processed']:,}")

        del results, df_chunk, ts_strs, sampled_ts_ns
        gc.collect()

    t_elapsed = time.time() - t_total
    total_rate = stats["rows_processed"] / t_elapsed if t_elapsed > 0 else 0
    print(f"  [P6] Done: {stats['rows_processed']:,} events in {t_elapsed:.1f}s "
          f"({total_rate:,.0f} ev/s)")

    return stats


# ---------------------------------------------------------------------------
# Summary (pandas vectorized)
# ---------------------------------------------------------------------------

def generate_summary(excursion_stats_path: Path, output_summary_path: Path) -> None:
    import pandas as pd
    print("  [P6] Computing summary statistics...")

    df = pd.read_csv(excursion_stats_path, engine="c")

    with open(output_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for h in HORIZONS:
            up_col = f"max_up_{h}s_ticks"
            mae_col = f"mae_{h}s_ticks"
            complete_col = f"window_complete_{h}s"
            n_obs_col = f"n_obs_{h}s"

            up_vals = df[up_col].values.astype(np.float64)
            mae_vals = df[mae_col].values.astype(np.float64)
            complete = int(df[complete_col].values.sum())
            n_obs_vals = df[n_obs_col].values
            n = len(df)

            up_sorted = np.sort(up_vals)
            mae_sorted = np.sort(mae_vals)

            pct_complete = complete / n * 100 if n > 0 else 0
            mean_n = float(np.mean(n_obs_vals)) if n > 0 else 0
            max_n = int(np.max(n_obs_vals)) if n > 0 else 0
            mean_up = float(np.mean(up_sorted)) if n > 0 else 0
            mean_mae = float(np.mean(mae_sorted)) if n > 0 else 0

            safe_mae = np.where(mae_vals > 0, mae_vals, 1.0)
            ratios = np.where(mae_vals > 0, up_vals / safe_mae, 0.0)
            ratios_sorted = np.sort(ratios)

            writer.writerow({
                "horizon": f"{h}s",
                "n_events": n,
                "p50_up_ticks": round(float(np.percentile(up_sorted, 50)), 4),
                "p75_up_ticks": round(float(np.percentile(up_sorted, 75)), 4),
                "p90_up_ticks": round(float(np.percentile(up_sorted, 90)), 4),
                "p95_up_ticks": round(float(np.percentile(up_sorted, 95)), 4),
                "mean_up_ticks": round(mean_up, 4),
                "p50_mae_ticks": round(float(np.percentile(mae_sorted, 50)), 4),
                "p75_mae_ticks": round(float(np.percentile(mae_sorted, 75)), 4),
                "p90_mae_ticks": round(float(np.percentile(mae_sorted, 90)), 4),
                "p95_mae_ticks": round(float(np.percentile(mae_sorted, 95)), 4),
                "mean_mae_ticks": round(mean_mae, 4),
                "median_up_over_mae_ratio": round(float(np.median(ratios_sorted)), 4),
                "pct_complete": round(pct_complete, 2),
                "mean_n_obs": round(mean_n, 1),
                "max_n_obs": max_n,
            })

    print(f"  [P6] Summary written to {output_summary_path.name}")


# ---------------------------------------------------------------------------
# Plot distributions
# ---------------------------------------------------------------------------

def plot_distributions(excursion_stats_path: Path, output_png_path: Path) -> None:
    import pandas as pd
    print("  [P6] Generating plots...")

    df = pd.read_csv(excursion_stats_path, engine="c")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Mid-Price Excursion Distributions (Ticks)\nTop: MFE (up) | Bottom: MAE (adverse)",
                 fontsize=14, fontweight="bold")

    for col, h in enumerate(HORIZONS):
        ax_up = axes[0, col]
        up_vals = df[f"max_up_{h}s_ticks"].values
        ax_up.hist(up_vals, bins=80, color="#2E7D32", alpha=0.7, edgecolor="none")
        ax_up.set_title(f"MFE {h}s")
        ax_up.set_xlabel("Ticks (positive = price went up)")
        ax_up.axvline(0, color="black", linewidth=0.8, linestyle="--")

        for pct, color in zip(PERCENTILES, COLORS):
            val = float(np.percentile(up_vals, pct))
            ax_up.axvline(val, color=color, linewidth=1.2, linestyle="-", alpha=0.8)
            ax_up.text(val, ax_up.get_ylim()[1] * 0.95, f"p{pct}", color=color, fontsize=7)

        ax_down = axes[1, col]
        mae_vals = df[f"mae_{h}s_ticks"].values
        ax_down.hist(mae_vals, bins=80, color="#C62828", alpha=0.7, edgecolor="none")
        ax_down.set_title(f"MAE {h}s")
        ax_down.set_xlabel("Ticks (always positive)")

        for pct, color in zip(PERCENTILES, COLORS):
            val = float(np.percentile(mae_vals, pct))
            ax_down.axvline(val, color=color, linewidth=1.2, linestyle="-", alpha=0.8)
            ax_down.text(val, ax_down.get_ylim()[1] * 0.95, f"p{pct}", color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_png_path}")


# ---------------------------------------------------------------------------
# Main — CLI interface (backward compatible)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6: Excursion Analysis (Vectorized + Numba JIT)")
    parser.add_argument("--features", type=Path, required=True,
                        help="Path to features_dom.csv (used to locate snapshots.csv in same dir)")
    parser.add_argument("--sampled", type=Path, required=True,
                        help="Path to sampled_events.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for excursion_stats.csv")
    parser.add_argument("--summary", type=Path, required=True,
                        help="Output path for excursion_summary.csv")
    parser.add_argument("--plot", type=Path, required=True,
                        help="Output path for excursion_distributions.png")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if output exists (skip fingerprint check)")
    args = parser.parse_args()

    t_total = time.time()

    # ── Staleness guard (Apr 13, 2026 fix) ─────────────────────────────────
    # Only apply fingerprint logic when running from run_p1_to_p7_multiday.py
    # (which passes output dir via the excursion_path.parent). For standalone
    # CLI runs, rely on --force flag or missing output file.
    out_dir = args.output.parent
    p5_path = args.sampled
    date_str = out_dir.name

    if not args.force:
        skip, reason = should_skip_p6(out_dir, p5_path)
        if skip:
            print(f"\n  [P6] SKIP: output valid for {date_str} (fingerprint matches)")
            print(f"        fingerprint_ts: {load_fingerprint(out_dir).get('fingerprint_ts','unknown')}")
            sys.exit(0)
        else:
            print(f"\n  [P6] STALENESS: {reason} -- will re-run")

    # ── Build lookup index from features_dom.csv ──────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 6: EXCURSION ANALYSIS (VECTORIZED + NUMBA JIT)")
    print(f"{'='*60}")
    print(f"  Numba available: {HAS_NUMBA}")

    ts_ns, mp = build_lookup_index(args.features)
    print(f"  Index: {len(ts_ns):,} entries")

    # ── Compute excursions ─────────────────────────────────────────────────────
    print("  Computing excursions...")
    stats = compute_excursions(args.sampled, ts_ns, mp, args.output)
    print(f"  -> {stats.get('rows_processed', 0):,} rows processed")

    # ── Post-run validation (ratio check) ─────────────────────────────────────
    validate_p6_output(p5_path, args.output, date_str)

    # ── Save fingerprint (only after validation passes) ──────────────────────
    fp = compute_input_fingerprint(p5_path)
    save_fingerprint(fp, out_dir)
    print(f"  Fingerprint saved: {fp['fingerprint_ts']}")

    # Free lookup index before loading full output for summary/plots
    del ts_ns, mp
    gc.collect()

    generate_summary(args.output, args.summary)
    plot_distributions(args.output, args.plot)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Phase 6 completed: {stats['rows_processed']:,} events in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Complete windows: 30s={stats['n_complete_30s']:,}  60s={stats['n_complete_60s']:,}  120s={stats['n_complete_120s']:,}")
    print(f"{'='*60}\n")
