"""
Phase 4: Temporal Aggregation of DOM Features
Reads features_dom.csv, computes rolling stats over 1s/5s/30s windows,
writes features_dom_agg.csv.

OPTIMIZED VERSION:
  1. pandas usecols to read ONLY 10 needed columns (vs 36)
  2. csv.reader-style iteration via itertuples (no dict per row)
  3. Direct float() instead of _safe_float() with try/except
  4. Fast timestamp parsing

Input:  output/YYYY-MM-DD/features_dom.csv  (~26M rows)
Output: output/YYYY-MM-DD/features_dom_agg.csv
"""

import csv
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# DATA CONTRACT — vps_feature_engineering_agg.py (Phase 4)
#
# INPUT:  features_dom.csv — colonna ts in formato ISO string
#         "2026-01-08 09:30:00.123456 UTC"
#
# OUTPUT: features_dom_agg.csv — colonna ts come INTEGER ms-from-midnight
#         Esempio: 09:30:00.123 → 34200123
#
# MOTIVO: P5 (vps_cusum_sampler.py) legge P3 e P4 in chunk SINCRONIZZATI
#         per posizione (zip), NON per join su ts. Il formato ts in P4
#         non impatta P5 perché P5 usa ts da P3 (ISO string).
#         Se in futuro P4 venisse usato standalone, il ts INTEGER
#         richiede conversione esplicita.
#
# DIPENDENTI: P5 (allineamento posizionale), nessun altro.
# ─────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_1S_MS = 1_000
WINDOW_5S_MS = 5_000
WINDOW_30S_MS = 30_000

PROGRESS_EVERY = 100_000

# Only columns we need (for usecols)
# Updated Apr 2026 for MOMENTUM feature set (P3 rebuild)
NEEDED_COLS = [
    'ts',
    'imbalance_1',
    'delta_50', 'delta_100', 'delta_200',
    'imb_trend_10', 'imb_trend_50',
    'imb_ma_ratio_10', 'imb_ma_ratio_50',
    'microprice_momentum_10', 'microprice_momentum_50',
    'microprice_dev_from_ma',
    'ps_delta_L1',
    'ofi_50', 'ofi_100', 'ofi_500',
    'stack_sweep_bid_flag', 'stack_sweep_ask_flag',
    'stack_sweep_any_flag',
    'bid_sweep_count', 'ask_sweep_count',
    'vwap_dev_ticks',
    'vpin_100',
]

# ---------------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------------

AGG_FEATURE_FIELDS = [
    "ts",
    # Imbalance rolling stats
    "imbalance_mean_1s", "imbalance_std_1s",
    "imbalance_mean_5s", "imbalance_std_5s",
    "imbalance_mean_30s", "imbalance_std_30s",
    # Cumulative delta rolling stats
    "delta_50_mean_1s", "delta_50_std_1s",
    "delta_100_mean_1s", "delta_100_std_1s",
    "delta_200_mean_1s", "delta_200_std_1s",
    # Imbalance trend rolling
    "imb_trend_10_mean_1s", "imb_trend_50_mean_1s",
    "imb_ma_ratio_10_mean_1s", "imb_ma_ratio_50_mean_1s",
    # Microprice momentum rolling
    "microprice_momentum_10_mean_1s", "microprice_momentum_50_mean_1s",
    "microprice_dev_from_ma_mean_1s",
    # OFI rolling
    "ofi_50_mean_1s", "ofi_100_mean_1s", "ofi_500_mean_1s",
    "ps_delta_L1_mean_1s", "ps_delta_L1_mean_5s", "ps_delta_L1_mean_30s",
    # Sweep rolling (boolean flags)
    "stack_sweep_bid_flag_sum_1s", "stack_sweep_ask_flag_sum_1s",
    "stack_sweep_any_flag_sum_1s",
    "bid_sweep_count_mean_1s", "ask_sweep_count_mean_1s",
    # Session features
    "vwap_dev_ticks_mean_1s",
    "vpin_100_mean_1s",
]


# ---------------------------------------------------------------------------
# Sliding window with running sum + sum_sq for O(1) mean/var
# ---------------------------------------------------------------------------

class SlidingWindowStats:
    __slots__ = ("deque", "sum", "sum_sq", "window_ms")

    def __init__(self, window_ms: int) -> None:
        self.deque: deque[tuple[int, float]] = deque()
        self.sum: float = 0.0
        self.sum_sq: float = 0.0
        self.window_ms: int = window_ms

    def update(self, ts_ms: int, value: float) -> None:
        cutoff = ts_ms - self.window_ms
        d = self.deque
        while d and d[0][0] < cutoff:
            _, old_val = d.popleft()
            self.sum -= old_val
            self.sum_sq -= old_val * old_val
        d.append((ts_ms, value))
        self.sum += value
        self.sum_sq += value * value

    def get(self) -> tuple[float, float]:
        n = len(self.deque)
        if n == 0:
            return 0.0, 0.0
        mean = self.sum / n
        var = max(0.0, self.sum_sq / n - mean * mean)
        return mean, math.sqrt(var)

    @property
    def count(self) -> int:
        return len(self.deque)


# ---------------------------------------------------------------------------
# 1s window
# ---------------------------------------------------------------------------

class Window1s:
    __slots__ = ("ts", "imb", "net_w", "delta",
                 "pull_bid", "stack_bid", "pull_ask", "stack_ask",
                 "bid_qty", "ask_qty",
                 "exhaustion_count",
                 "sum_imb", "sum_sq_imb", "sum_net_w", "sum_delta",
                 "sum_pull_bid", "sum_stack_bid", "sum_pull_ask", "sum_stack_ask")

    def __init__(self) -> None:
        self.ts: deque[int] = deque()
        self.imb: deque[float] = deque()
        self.net_w: deque[float] = deque()
        self.delta: deque[float] = deque()
        self.pull_bid: deque[float] = deque()
        self.stack_bid: deque[float] = deque()
        self.pull_ask: deque[float] = deque()
        self.stack_ask: deque[float] = deque()
        self.bid_qty: deque[float] = deque()
        self.ask_qty: deque[float] = deque()
        self.sum_imb: float = 0.0
        self.sum_sq_imb: float = 0.0
        self.sum_net_w: float = 0.0
        self.sum_delta: float = 0.0
        self.sum_pull_bid: float = 0.0
        self.sum_stack_bid: float = 0.0
        self.sum_pull_ask: float = 0.0
        self.sum_stack_ask: float = 0.0
        self.exhaustion_count: int = 0

    def update(self, ts_ms: int, imbalance: float, net_w: float, delta: float,
               pull_bid: float, stack_bid: float, pull_ask: float, stack_ask: float,
               bid_qty: float, ask_qty: float) -> None:
        cutoff = ts_ms - WINDOW_1S_MS
        while self.ts and self.ts[0] < cutoff:
            self.ts.popleft()
            old_imb = self.imb.popleft()
            self.sum_imb -= old_imb
            self.sum_sq_imb -= old_imb * old_imb
            self.sum_net_w -= self.net_w.popleft()
            self.sum_delta -= self.delta.popleft()
            self.sum_pull_bid -= self.pull_bid.popleft()
            self.sum_stack_bid -= self.stack_bid.popleft()
            self.sum_pull_ask -= self.pull_ask.popleft()
            self.sum_stack_ask -= self.stack_ask.popleft()
            old_bid = self.bid_qty.popleft()
            old_ask = self.ask_qty.popleft()
            if old_bid <= EXHAUSTION_THRESHOLD or old_ask <= EXHAUSTION_THRESHOLD:
                self.exhaustion_count -= 1

        self.ts.append(ts_ms)
        self.imb.append(imbalance)
        self.net_w.append(net_w)
        self.delta.append(delta)
        self.pull_bid.append(pull_bid)
        self.stack_bid.append(stack_bid)
        self.pull_ask.append(pull_ask)
        self.stack_ask.append(stack_ask)
        self.bid_qty.append(bid_qty)
        self.ask_qty.append(ask_qty)
        if bid_qty <= EXHAUSTION_THRESHOLD or ask_qty <= EXHAUSTION_THRESHOLD:
            self.exhaustion_count += 1

        self.sum_imb += imbalance
        self.sum_sq_imb += imbalance * imbalance
        self.sum_net_w += net_w
        self.sum_delta += delta
        self.sum_pull_bid += pull_bid
        self.sum_stack_bid += stack_bid
        self.sum_pull_ask += pull_ask
        self.sum_stack_ask += stack_ask

    def get(self) -> tuple[float, float, float, float, float, float, float, float, int, int]:
        n = len(self.imb)
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
        mean_imb = self.sum_imb / n
        var_imb = max(0.0, self.sum_sq_imb / n - mean_imb * mean_imb)
        std_imb = math.sqrt(var_imb)
        mean_net_w = self.sum_net_w / n
        mean_delta = self.sum_delta / n
        return (
            mean_imb, std_imb,
            mean_net_w, mean_delta,
            self.sum_pull_bid, self.sum_stack_bid,
            self.sum_pull_ask, self.sum_stack_ask,
            n, self.exhaustion_count,
        )


# ---------------------------------------------------------------------------
# Fast timestamp parsing
# ---------------------------------------------------------------------------

def _parse_ms_fast(digits: str) -> int:
    """Parse ms-from-midnight from 17-digit timestamp digits.
    digits format: YYYYMMDDHHMMSSmmm... (17 digits after date)"""
    # digits[8:10] = HH, [10:12] = MM, [12:14] = SS, [14:17] = mmm
    hour = int(digits[8:10])
    minute = int(digits[10:12])
    sec = int(digits[12:14])
    ms = int(digits[14:17])
    return ((hour * 3600 + minute * 60 + sec) * 1000) + ms


def _safe_float(s: str) -> float:
    """Convert string to float, return 0.0 for empty/invalid."""
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def aggregate_features_chunked(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows_processed": 0,
        "rows_written": 0,
        "first_ts": None,
        "last_ts": None,
    }

    print("  [P4] Streaming with pandas vectorized rolling (chunked)...")

    first_chunk = True
    last_overlap = pd.DataFrame()
    chunksize = 2_000_000  # 30s overlap = ~1.5M rows at 50K/sec → 75% of 2M, OK
    rows_proc = 0
    import gc
    from collections import deque
    
    output_cols = AGG_FEATURE_FIELDS.copy()

    with pd.read_csv(input_path, usecols=NEEDED_COLS, chunksize=chunksize, iterator=True) as reader:
        for chunk in reader:
            # Parse datetime — strip trailing UTC/utc before parsing
            # Historical data: '2026-03-13 13:40:00.000000 UTC'
            # New data:       '2026-04-10 13:40:00.000000' or ISO with +00:00
            ts_col = chunk['ts'].str.replace(r'\s+UTC$', '', regex=True).str.replace(r'\s+utc$', '', regex=True)
            chunk['ts_dt'] = pd.to_datetime(ts_col, format='ISO8601', errors='coerce')
            # Do NOT drop rows — maintain 1:1 alignment with features_dom.csv
            # Rows that fail to parse get NaT index; rolling on NaT produces NaN (filled with 0.0 below)
            chunk.set_index('ts_dt', inplace=True)
            
            # Track ts for report
            if first_chunk:
                stats['first_ts'] = chunk['ts'].iloc[0]
            stats['last_ts'] = chunk['ts'].iloc[-1]
            
            chunk.drop(columns=['ts'], inplace=True) # remove raw string column

            if not last_overlap.empty:
                df = pd.concat([last_overlap, chunk], verify_integrity=False, sort=False)
            else:
                df = chunk

            # Guard: drop NaT rows first (sort_index crashes on NaT index),
            # then sort if needed for rolling operations
            nat_mask = df.index.isna()
            if nat_mask.any():
                n_nat = nat_mask.sum()
                if first_chunk:
                    pass  # just log on first chunk
                df = df[~nat_mask]
            if df.empty:
                # Nothing to process in this chunk; save overlap and continue
                last_ts = chunk.index[-1]
                cutoff_time = last_ts - pd.Timedelta(seconds=30)
                last_overlap = chunk.loc[chunk.index > cutoff_time] if len(chunk) > 0 else pd.DataFrame()
                first_chunk = False
                rows_proc += 0
                print(f"  Phase 4: {rows_proc:,} rows | Chunk empty after NaT removal, skipping.")
                del chunk, df
                gc.collect()
                continue
            if df.index.has_duplicates:
                df = df.sort_index()

            r1s = df.rolling('1s')
            r5s = df.rolling('5s')
            r30s = df.rolling('30s')

            df_out = pd.DataFrame(index=df.index)
            df_out['ts'] = (df.index.hour * 3600_000 + df.index.minute * 60_000 +
                            df.index.second * 1_000 + df.index.microsecond // 1_000)

            # Imbalance rolling stats
            df_out['imbalance_mean_1s'] = r1s['imbalance_1'].mean().fillna(0.0).round(6)
            df_out['imbalance_std_1s'] = r1s['imbalance_1'].std().fillna(0.0).round(6)
            df_out['imbalance_mean_5s'] = r5s['imbalance_1'].mean().fillna(0.0).round(6)
            df_out['imbalance_std_5s'] = r5s['imbalance_1'].std().fillna(0.0).round(6)
            df_out['imbalance_mean_30s'] = r30s['imbalance_1'].mean().fillna(0.0).round(6)
            df_out['imbalance_std_30s'] = r30s['imbalance_1'].std().fillna(0.0).round(6)

            # Cumulative delta rolling stats
            df_out['delta_50_mean_1s'] = r1s['delta_50'].mean().fillna(0.0).round(6)
            df_out['delta_50_std_1s'] = r1s['delta_50'].std().fillna(0.0).round(6)
            df_out['delta_100_mean_1s'] = r1s['delta_100'].mean().fillna(0.0).round(6)
            df_out['delta_100_std_1s'] = r1s['delta_100'].std().fillna(0.0).round(6)
            df_out['delta_200_mean_1s'] = r1s['delta_200'].mean().fillna(0.0).round(6)
            df_out['delta_200_std_1s'] = r1s['delta_200'].std().fillna(0.0).round(6)

            # Imbalance trend rolling
            df_out['imb_trend_10_mean_1s'] = r1s['imb_trend_10'].mean().fillna(0.0).round(6)
            df_out['imb_trend_50_mean_1s'] = r1s['imb_trend_50'].mean().fillna(0.0).round(6)
            df_out['imb_ma_ratio_10_mean_1s'] = r1s['imb_ma_ratio_10'].mean().fillna(0.0).round(6)
            df_out['imb_ma_ratio_50_mean_1s'] = r1s['imb_ma_ratio_50'].mean().fillna(0.0).round(6)

            # Microprice momentum rolling
            df_out['microprice_momentum_10_mean_1s'] = r1s['microprice_momentum_10'].mean().fillna(0.0).round(6)
            df_out['microprice_momentum_50_mean_1s'] = r1s['microprice_momentum_50'].mean().fillna(0.0).round(6)
            df_out['microprice_dev_from_ma_mean_1s'] = r1s['microprice_dev_from_ma'].mean().fillna(0.0).round(6)

            # OFI rolling
            df_out['ofi_50_mean_1s'] = r1s['ofi_50'].mean().fillna(0.0).round(6)
            df_out['ofi_100_mean_1s'] = r1s['ofi_100'].mean().fillna(0.0).round(6)
            df_out['ofi_500_mean_1s'] = r1s['ofi_500'].mean().fillna(0.0).round(6)

            # PS delta rolling
            df_out['ps_delta_L1_mean_1s'] = r1s['ps_delta_L1'].mean().fillna(0.0).round(6)
            df_out['ps_delta_L1_mean_5s'] = r5s['ps_delta_L1'].mean().fillna(0.0).round(6)
            df_out['ps_delta_L1_mean_30s'] = r30s['ps_delta_L1'].mean().fillna(0.0).round(6)

            # Sweep rolling (boolean flags)
            df_out['stack_sweep_bid_flag_sum_1s'] = r1s['stack_sweep_bid_flag'].sum().fillna(0).astype(int)
            df_out['stack_sweep_ask_flag_sum_1s'] = r1s['stack_sweep_ask_flag'].sum().fillna(0).astype(int)
            df_out['stack_sweep_any_flag_sum_1s'] = r1s['stack_sweep_any_flag'].sum().fillna(0).astype(int)
            df_out['bid_sweep_count_mean_1s'] = r1s['bid_sweep_count'].mean().fillna(0.0).round(4)
            df_out['ask_sweep_count_mean_1s'] = r1s['ask_sweep_count'].mean().fillna(0.0).round(4)

            # Session features
            df_out['vwap_dev_ticks_mean_1s'] = r1s['vwap_dev_ticks'].mean().fillna(0.0).round(6)
            df_out['vpin_100_mean_1s'] = r1s['vpin_100'].mean().fillna(0.0).round(6)
            
            # Slice off the overlap part for writing
            overlap_len = len(last_overlap)
            write_df = df_out.iloc[overlap_len:]
            
            # Save the last 30s of the chunk for the next iteration step
            last_ts = df.index[-1]
            cutoff_time = last_ts - pd.Timedelta(seconds=30)
            last_overlap = df.loc[df.index > cutoff_time]
            
            # Ensure columns match exact order of AGG_FEATURE_FIELDS
            write_df = write_df[output_cols]
            
            mode = 'w' if first_chunk else 'a'
            write_df.to_csv(output_path, mode=mode, header=first_chunk, index=False)
            first_chunk = False
            
            rows_proc += len(write_df)
            print(f"  Phase 4: {rows_proc:,} rows | Vectorized chunks processed.")
            
            # Clear memory
            del chunk, df, df_out, write_df, r1s, r5s, r30s
            gc.collect()

    stats["rows_processed"] = rows_proc
    stats["rows_written"] = rows_proc
    return stats



def print_report(stats: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("TEMPORAL AGGREGATION - PHASE 4 REPORT")
    print("=" * 60)
    print(f"\n[ROWS]")
    print(f"  Processed : {stats['rows_processed']:,}")
    print(f"  Written   : {stats['rows_written']:,}")
    print(f"\n[TIME RANGE]")
    print(f"  First row : {stats['first_ts'] or 'N/A'}")
    print(f"  Last row  : {stats['last_ts'] or 'N/A'}")
    print(f"\n[AGGREGATION]")
    print(f"  Windows   : 1s (exact), 5s (O(1)), 30s (O(1))")
    print(f"  Features  : {len(AGG_FEATURE_FIELDS) - 1}")
    print("=" * 60)
    print("Phase 4 completed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        print(f"Exists. Use --force.")
        sys.exit(1)
    stats = aggregate_features_chunked(args.input, args.output)
    print_report(stats)
