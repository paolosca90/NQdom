"""
Phase 7 — True First-Touch Triple Barrier Labeling
===================================================
For each CUSUM-sampled event, scans the real price path from snapshots.csv
and determines which barrier fires FIRST: PT / SL / or vertical expiry.

KEY SEMANTIC INVARIANT:
  Vertical barrier: TICK CLOCK only (book update count, NOT wall-clock seconds).
  vb_ticks=30 means 30 snapshots after the event, regardless of elapsed time.
  A "second" in the trading DOM is NOT a stable unit — tick count is.

Key methodological difference from Phase 6:
  Phase 6 computed MAX excursion (max_up_ticks, max_down_ticks) — tells you
  the furthest price reached in each direction, but NOT which barrier was hit
  first in time order.

  Phase 7 uses TRUE FIRST-TOUCH: scans the actual chronological price sequence
  and returns the first barrier to be touched. This matters when price crosses
  both PT and SL multiple times — only the FIRST crossing determines the label.

Labeling rules:
  +1  → PT (profit target) is the first barrier touched
  -1  → SL (stop loss)     is the first barrier touched
   0  → vertical barrier (tick clock) expires before either PT or SL is touched

Data sources:
  - sampled_events.csv     → event timestamps (ts)
  - excursion_stats.csv    → mid_price_at_t for each event (same row order)
  - snapshots.csv          → chronological mid-price path for first-touch scan

Usage:
  python3 phase7_labeling.py \\
      --snapshots  /opt/depth-dom/output/2026-01-08/snapshots.csv \\
      --sampled    /opt/depth-dom/output/2026-01-08/sampled_events.csv \\
      --refprice   /opt/depth-dom/output/2026-01-08/excursion_stats.csv \\
      --grid       /opt/depth-dom/output/_candidates_3.csv \\
      --output     /opt/depth-dom/output/2026-01-08/ \\
      --candidates 10
"""

import argparse, csv, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

# ── import the single authorized source for label filename construction ─────────
sys.path.insert(0, str(Path(__file__).parent.parent / "SHARED"))
from _pipeline_constants import label_filename

TICK_SIZE = 0.25
PROGRESS_EVERY = 10_000


# ── snapshot index (numpy) ────────────────────────────────────────────────────

def build_snapshot_index(snapshots_path: Path):
    """
    Fast snapshot loading via pandas. Loads directly from CSV using C engine,
    then parses datetimes instantly using fast string to int64 views.
    Memory footprint ~250MB vs 4GB Python dict overhead.
    """
    df = pd.read_csv(snapshots_path, usecols=["ts", "mid_price"], engine="c",
                     dtype={"mid_price": "float32"})
    # Fast convert to nanoseconds
    df["ts"] = df["ts"].str.replace(" UTC", "")
    df.dropna(subset=["ts"], inplace=True)
    ts_ns = pd.to_datetime(df["ts"], format="mixed", utc=True).values.view("int64")
    mid_p = df["mid_price"].fillna(0.0).values.astype(np.float64)
    
    order = np.argsort(ts_ns)
    return ts_ns[order], mid_p[order]


# ── load event reference prices ──────────────────────────────────────────────

def load_event_ref_prices(refprice_path: Path):
    """
    Fast load of ts + mid_price_at_t from excursion_stats.csv.
    Returns a pandas Series mapping ts_str -> mid_price_at_t.
    """
    df = pd.read_csv(refprice_path, usecols=["ts", "mid_price_at_t"], engine="c")
    df["ts"] = df["ts"].str.replace(" UTC", "")
    return df.set_index("ts")["mid_price_at_t"]


# ── numpy-accelerated first-touch scan (TICK CLOCK / PARALLEL) ───────────────

@njit(parallel=True, cache=True)
def _numba_first_touch_scan_tick(snap_ts: np.ndarray, snap_mid: np.ndarray,
                                 start_idx: np.ndarray, end_idx: np.ndarray,
                                 pt_prices: np.ndarray, sl_prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(start_idx)
    labels = np.zeros(n, dtype=np.int8)
    hit_prices = np.zeros(n, dtype=np.float64)
    n_snaps = len(snap_ts)

    for i in prange(n):
        idx_s = start_idx[i]
        idx_e = end_idx[i]
        if idx_e > n_snaps:
            idx_e = n_snaps
            
        n_win = idx_e - idx_s
        
        if n_win <= 1:
            labels[i] = 0
            continue
            
        pt_hit_idx = -1
        sl_hit_idx = -1
        pt_p = pt_prices[i]
        sl_p = sl_prices[i]
        
        # Scan window for first touch sequentially inside the parallel chunk
        for j in range(1, n_win):
            val = snap_mid[idx_s + j]
            if pt_hit_idx == -1 and val >= pt_p:
                pt_hit_idx = j
            if sl_hit_idx == -1 and val <= sl_p:
                sl_hit_idx = j
            if pt_hit_idx != -1 and sl_hit_idx != -1:
                break
                
        pt_reached = (pt_hit_idx != -1)
        sl_reached = (sl_hit_idx != -1)
            
        if pt_reached and not sl_reached:
            labels[i] = 1
            hit_prices[i] = snap_mid[idx_s + pt_hit_idx]
        elif sl_reached and not pt_reached:
            labels[i] = -1
            hit_prices[i] = snap_mid[idx_s + sl_hit_idx]
        elif pt_reached and sl_reached:
            if pt_hit_idx <= sl_hit_idx:
                labels[i] = 1
                hit_prices[i] = snap_mid[idx_s + pt_hit_idx]
            else:
                labels[i] = -1
                hit_prices[i] = snap_mid[idx_s + sl_hit_idx]
        else:
            labels[i] = 0
            
    return labels, hit_prices


def _scan_first_touch_numpy_tickclock(
    snap_ts: np.ndarray, snap_mid: np.ndarray,
    t0_arr: np.ndarray,   # shape (n_events,) — event start times in ns
    p0_arr: np.ndarray,   # shape (n_events,) — entry mid-prices
    pt_arr: np.ndarray,   # shape (n_events,) — PT in ticks
    sl_arr: np.ndarray,   # shape (n_events,) — SL in ticks
    vb_ticks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated parallel first-touch window scan using TICK CLOCK.
    Returns (labels, hit_prices).
    """
    n = len(t0_arr)
    pt_prices = (p0_arr + pt_arr * TICK_SIZE).astype(np.float64)
    sl_prices = (p0_arr - sl_arr * TICK_SIZE).astype(np.float64)
    
    # Still map realtime to initial offset
    start_idx = np.searchsorted(snap_ts, t0_arr, side="left")
    
    # KEY OPTIMIZATION: Tick Clock! 
    # Vertical Barrier is precisely start_idx + N updates.
    # Completely avoids searchsorted on end-times, standardizing variance!
    end_idx = start_idx + vb_ticks
    
    return _numba_first_touch_scan_tick(
        snap_ts, snap_mid, start_idx, end_idx, pt_prices, sl_prices
    )



# ── per-candidate labeling ────────────────────────────────────────────────────

def label_candidate(
    snap_ts: np.ndarray, snap_mid: np.ndarray,
    df_sampled: pd.DataFrame,
    vb_ticks: int, pt_ticks: float, sl_ticks: float,
    out_path: Path,
) -> dict:
    """
    Label all sampled events for one barrier candidate using true first-touch.
    Memory efficient via passing Pandas DataFrame directly.
    """
    n = len(df_sampled)

    t0_arr = df_sampled["ts_ns"].values
    p0_arr = df_sampled["p0"].values
    pt_arr = np.full(n, pt_ticks, dtype=np.float64)
    sl_arr = np.full(n, sl_ticks, dtype=np.float64)

    # Filter out events with no ref price
    valid_idx = np.where(p0_arr > 0)[0]
    n_valid = len(valid_idx)
    if n_valid < n:
        print(f"    Warning: {n - n_valid}/{n} events have no ref price (outside snapshot range)")

    print(f"    [{vb_ticks} ticks pt={pt_ticks} sl={sl_ticks}] "
          f"Scanning {n_valid:,} events …", end=" ", flush=True)
    t0 = time.time()
    labels, hit_prices = _scan_first_touch_numpy_tickclock(
        snap_ts, snap_mid,
        t0_arr[valid_idx],
        p0_arr[valid_idx],
        pt_arr[valid_idx],
        sl_arr[valid_idx],
        vb_ticks
    )
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s  ({n_valid/elapsed:.0f} events/sec)")

    # Vectorized fast CSV output
    valid_df = df_sampled.iloc[valid_idx].copy()
    valid_df["vb_ticks"] = vb_ticks
    valid_df["pt_ticks"] = pt_ticks
    valid_df["sl_ticks"] = sl_ticks
    valid_df["pt_price"] = (valid_df["p0"] + pt_ticks * TICK_SIZE).round(4)
    valid_df["sl_price"] = (valid_df["p0"] - sl_ticks * TICK_SIZE).round(4)
    valid_df["label"] = labels
    valid_df["barrier_hit"] = valid_df["label"].map({-1: "sl", 1: "pt", 0: "vertical"})
    valid_df["hit_price"] = np.where(hit_prices > 0, np.round(hit_prices, 4), 0.0)
    
    valid_df.rename(columns={"p0": "mid_price_at_t"}).to_csv(
        out_path, index=False, columns=[
            "ts", "mid_price_at_t", "vb_ticks", "pt_ticks", "sl_ticks", 
            "pt_price", "sl_price", "label", "barrier_hit", "hit_price"
        ]
    )

    # Metrics (only valid events)
    n_pt = int(np.sum(labels == 1))
    n_sl = int(np.sum(labels == -1))
    n_v  = int(np.sum(labels == 0))
    denom = n_pt + n_sl

    return {
        "vertical_barrier_ticks": vb_ticks,
        "pt_ticks":    pt_ticks,
        "sl_ticks":    sl_ticks,
        "pt_sl_ratio": round(pt_ticks / sl_ticks, 3) if sl_ticks > 0 else 0,
        "n_events":    n_valid,
        "n_pt":        n_pt,
        "n_sl":        n_sl,
        "n_vertical":  n_v,
        "pct_pt":       round(n_pt / n_valid * 100, 2) if n_valid else 0.0,
        "pct_sl":       round(n_sl / n_valid * 100, 2) if n_valid else 0.0,
        "pct_vertical": round(n_v  / n_valid * 100, 2) if n_valid else 0.0,
        "balance_ratio": round(min(n_pt, n_sl) / max(n_pt, n_sl), 3) if max(n_pt, n_sl) > 0 else 0.0,
        "win_rate":     round(n_pt / denom * 100, 2) if denom > 0 else 0.0,
        "payoff_theoretical": round((n_pt * pt_ticks) / (n_sl * sl_ticks), 3) if n_sl > 0 and sl_ticks > 0 else float("inf"),
        "output_file":  out_path.name,
    }


# ── temporal split (no random shuffle) ──────────────────────────────────────

SPLITS = {
    "train": ("00:00:00.000000", "06:00:00.000000"),
    "val":   ("06:00:00.000",   "08:00:00.000"),
    "test":  ("08:00:00.000",   "09:30:00.000"),
}

def ts_in_split(ts_str: str, lo: str, hi: str) -> bool:
    time_part = ts_str[11:29]
    return lo <= time_part < hi


# ── leaderboard ───────────────────────────────────────────────────────────────

LEADERBOARD_FIELDS = [
    "vertical_barrier_ticks","pt_ticks","sl_ticks","pt_sl_ratio",
    "n_events","n_pt","n_sl","n_vertical",
    "pct_pt","pct_sl","pct_vertical",
    "balance_ratio","win_rate","payoff_theoretical","output_file"
]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 7: First-Touch Triple Barrier Labeling")
    ap.add_argument("--snapshots",  type=Path, required=True,
                    help="Path to snapshots.csv")
    ap.add_argument("--sampled",    type=Path, required=True,
                    help="Path to sampled_events.csv")
    ap.add_argument("--refprice",   type=Path, required=True,
                    help="Path to excursion_stats.csv (for mid_price_at_t)")
    ap.add_argument("--grid",       type=Path, required=True,
                    help="Path to labeling_grid_candidates.csv")
    ap.add_argument("--output",     type=Path, required=True,
                    help="Output directory")
    ap.add_argument("--candidates", type=int, default=10,
                    help="Number of candidates (default: 10 seeds; use 180 for full grid)")
    ap.add_argument("--force",      action="store_true")
    ap.add_argument("--split",      choices=["train","val","test","all"],
                    default="all")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PHASE 7 — True First-Touch Triple Barrier Labeling")
    print(f"{'='*60}")
    print(f"  Snapshots  : {args.snapshots}")
    print(f"  Sampled    : {args.sampled}")
    print(f"  Refprice   : {args.refprice}")
    print(f"  Grid       : {args.grid}")
    print(f"  Output     : {args.output}")
    print(f"  Candidates : {args.candidates}")
    print(f"  Split      : {args.split}")

    # ── 1. Snapshot index ──────────────────────────────────────────────
    print(f"\n[1] Loading snapshot index …")
    snap_ts, snap_mid = build_snapshot_index(args.snapshots)
    t_range = (snap_ts[-1] - snap_ts[0]) / 1e9
    print(f"    {len(snap_ts):,} snapshots  ({t_range:.0f}s = {t_range/60:.1f}min span)")

    # ── 2. Reference prices (mid_price_at_t per event) ──────────────────
    print(f"\n[2] Loading reference prices from {args.refprice.name} …")
    ref_prices = load_event_ref_prices(args.refprice)
    print(f"    {len(ref_prices):,} events with ref price")

    # ── 3. Sampled events (with optional temporal split) ─────────────────
    print(f"\n[3] Loading sampled events ({args.split}) …")
    df_sampled = pd.read_csv(args.sampled, engine="c", dtype={"ts": str})
    
    if args.split != "all":
        lo_t, hi_t = SPLITS.get(args.split, (None, None))
        time_part = df_sampled["ts"].str[11:29]
        df_sampled = df_sampled[(time_part >= lo_t) & (time_part < hi_t)].copy()
        
    # Bulk datetime parsing and index joining
    df_sampled["ts"] = df_sampled["ts"].str.replace(" UTC", "")
    df_sampled["ts_ns"] = pd.to_datetime(df_sampled["ts"], format="mixed", utc=True).values.view("int64")
    df_sampled["p0"] = df_sampled["ts"].map(ref_prices.to_dict()).fillna(0.0)
    print(f"    {len(df_sampled):,} events loaded")

    # ── 4. Load grid ──────────────────────────────────────────────────
    print(f"\n[4] Loading barrier grid …")
    with open(args.grid, newline="", encoding="utf-8") as f:
        grid = list(csv.DictReader(f))
    print(f"    {len(grid)} total combinations in grid")
    candidates = grid[:args.candidates]
    print(f"    Processing {len(candidates)} candidate(s)")

    # ── 5. Label each candidate ─────────────────────────────────────────
    print(f"\n[5] First-touch labeling …")
    leaderboard = []

    # PERF: pre-trigger numba JIT compilation with dummy arrays to avoid
    # paying compilation cost during first real candidate
    print("[JIT] Warming up numba cache (Tick Clock mode) ...", end=" ", flush=True)
    _t_jit = time.time()
    _dummy_ts   = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)
    _dummy_mid  = np.array([100.0, 100.25, 99.75], dtype=np.float64)
    _dummy_si   = np.array([0], dtype=np.int64)
    _dummy_ei   = np.array([3], dtype=np.int64)
    _dummy_pt   = np.array([101.0], dtype=np.float64)
    _dummy_sl   = np.array([99.0],  dtype=np.float64)
    _numba_first_touch_scan_tick(
        _dummy_ts, _dummy_mid, _dummy_si, _dummy_ei, _dummy_pt, _dummy_sl
    )
    print(f"done ({time.time()-_t_jit:.1f}s)")

    for i, row in enumerate(candidates, 1):
        if "vb_ticks" in row:
            vb = int(float(row["vb_ticks"]))
        else:
            raise ValueError(
                f"Grid CSV must contain column 'vb_ticks'. "
                f"Found: {list(row.keys())}. "
                f"Re-generate grid with incremental_p7p8_runner.py."
            )
            
        pt   = float(row["pt_ticks"])
        sl   = float(row["sl_ticks"])
        label_name = label_filename(vb, pt, sl)
        out_path = args.output / label_name

        if out_path.exists() and not args.force:
            print(f"  [{i}/{len(candidates)}] {vb}ticks pt={pt} sl={sl} → SKIP (exists)")
            with open(out_path, newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
            n = len(existing)
            n_pt = sum(1 for r in existing if r["barrier_hit"] == "pt")
            n_sl = sum(1 for r in existing if r["barrier_hit"] == "sl")
            n_v  = n - n_pt - n_sl
            denom = n_pt + n_sl
            leaderboard.append({
                "vertical_barrier_ticks": vb, "pt_ticks": pt, "sl_ticks": sl,
                "pt_sl_ratio": round(pt/sl, 3) if sl>0 else 0,
                "n_events": n, "n_pt": n_pt, "n_sl": n_sl, "n_vertical": n_v,
                "pct_pt": round(n_pt/n*100,2) if n else 0,
                "pct_sl": round(n_sl/n*100,2) if n else 0,
                "pct_vertical": round(n_v/n*100,2) if n else 0,
                "balance_ratio": round(min(n_pt,n_sl)/max(n_pt,n_sl),3) if max(n_pt,n_sl)>0 else 0,
                "win_rate": round(n_pt/denom*100,2) if denom else 0,
                "payoff_theoretical": round((n_pt*pt)/(n_sl*sl),3) if n_sl and sl>0 else float("inf"),
                "output_file": out_path.name,
            })
            continue

        print(f"\n  [{i}/{len(candidates)}] vb={vb}ticks  pt={pt}  sl={sl}")
        metrics = label_candidate(
            snap_ts, snap_mid, df_sampled, vb, pt, sl, out_path
        )
        leaderboard.append(metrics)
        print(f"    → PT={metrics['pct_pt']:.1f}%  "
              f"SL={metrics['pct_sl']:.1f}%  "
              f"V={metrics['pct_vertical']:.1f}%  "
              f"bal={metrics['balance_ratio']:.3f}  "
              f"win={metrics['win_rate']:.1f}%  "
              f"payoff={metrics['payoff_theoretical']}")

    # ── 6. Save leaderboard ─────────────────────────────────────────────
    lboard_path = args.output / "phase7_labeling_leaderboard.csv"
    sorted_lb  = sorted(leaderboard,
                        key=lambda r: (-r["balance_ratio"], -r["win_rate"]))
    with open(lboard_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted_lb)
    print(f"\n[6] Leaderboard → {lboard_path}")

    # ── Print top candidates ──────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'vb':>5} {'PT':>5} {'SL':>5} {'ratio':>6}  {'bal':>5} {'win%':>6}  "
          f"{'payoff':>7}  {'PT%':>5} {'SL%':>5} {'V%':>5}")
    print(f"{'─'*70}")
    for r in sorted_lb[:10]:
        pf = r["payoff_theoretical"]
        pf_str = f"{pf:.3f}" if pf != float("inf") else "  inf"
        print(f"{r['vertical_barrier_ticks']:>5} "
              f"{r['pt_ticks']:>5.1f} "
              f"{r['sl_ticks']:>5.1f} "
              f"{r['pt_sl_ratio']:>6.3f}  "
              f"{r['balance_ratio']:>5.3f} "
              f"{r['win_rate']:>6.1f}%  "
              f"{pf_str:>7}  "
              f"{r['pct_pt']:>5.1f}% "
              f"{r['pct_sl']:>5.1f}% "
              f"{r['pct_vertical']:>5.1f}%")

    print(f"\n{'='*60}")
    print("PHASE 7 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────
# BREAKING CHANGES — rispetto alla versione precedente
#
# 1. Grid CSV: colonna rinominata vertical_barrier_sec → vb_ticks
# 2. Output CSV: colonna vb_sec → vb_ticks
# 3. Output dir: pattern phase7_labels_{N}s_* → phase7_labels_{N}ticks_*
# 4. label_candidate() param: vb_ticks/pt_ticks/sl_ticks (nessun alias)
# 5. Backward compat: RIMOSSA. Tutti i dati vanno rigenerati.
#
# DIPENDENTI GIÀ AGGIORNATI (prompt precedenti):
# - incremental_p7p8_runner.py  ✓
# - vps_phase8_entry_model.py   ✓
# - _pipeline_constants.py     ✓
# - aggregate_results.py        ✓
# ─────────────────────────────────────────────────────────────────────
