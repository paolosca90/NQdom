#!/usr/bin/env python3
"""
audit_p7_results.py -- P5-P7 Data Quality & Label Distribution Audit
====================================================================
Legge tutti i 20 giorni di output P5-P7 e produce:

  B1 -- Label Distribution:
    label_distribution_daily.csv      Per-day: candidate, PT%, SL%, V%, balance_ratio, win_rate
    label_distribution_aggregate.csv   Aggregato 20 giorni per candidato
    label_outlier_days.csv            Giorni con balance_ratio < 0.80
    label_summary.txt                  Tabella riassuntiva

  B2 -- Quality Audit:
    quality_nan_rates.csv              NaN% per file: sampled_events, excursion_stats, features_dom
    quality_row_counts.csv             Row counts per fase per giorno
    quality_duplicates.csv             Timestamp duplicati in sampled_events
    quality_anomalies.csv              Giorni outlier
    quality_summary.txt               Report riassuntivo

USO LOCALE:
    python3 NQdom/ORCHESTRATOR/audit_p7_results.py --output-dir NQdom/output

OUTPUT: NQdom/output/_audit/
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = REPO_ROOT / "output"
AUDIT_DIR = "_audit"

# ── Constants ─────────────────────────────────────────────────────────────────
TICK_SIZE = 0.25
BALANCE_THRESHOLD = 0.80  # Flag days below this as outlier
NAN_RATE_WARN = 5.0       # Warn if NaN rate > 5%
ROW_COUNT_STD_THRESH = 3.0 # Flag days where row_count deviates >3 std devs


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_day_dirs(output_dir: Path) -> list[Path]:
    """Find all YYYY-MM-DD day directories."""
    dirs = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name):
            dirs.append(d)
    return dirs


def write_csv(rows: list[dict], path: Path, fieldnames: list[str] = None):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(s: float) -> str:
    return f"{s:.4f}"


def pct_str(n: int, total: int) -> str:
    if total == 0:
        return "0.00"
    return f"{n/total*100:.2f}"


# ════════════════════════════════════════════════════════════════════════════════
# B1 — LABEL DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════════

def find_label_dirs(day_dir: Path) -> list[tuple[str, Path]]:
    """Find all phase7_labels_* directories in a day dir.

    Also checks for phase7_labeling_leaderboard.csv as the leaderboard
    contains per-candidate label metrics computed by P7. The actual
    label files may be nested one level deeper or absent on some days.
    """
    candidates = []
    seen_cnames = set()

    # 1. Direct label dirs
    for d in sorted(day_dir.iterdir()):
        if d.is_dir() and d.name.startswith("phase7_labels_"):
            parts = d.name.split("_")
            if len(parts) >= 4:
                vb = parts[2]
                pt = parts[3]
                sl = parts[4]
                cname = f"{vb}/{pt}/{sl}"
            else:
                cname = d.name
            if cname not in seen_cnames:
                candidates.append((cname, d))
                seen_cnames.add(cname)

    # 2. Leaderboard fallback: if no label dirs found but leaderboard exists
    if not candidates:
        lboard_path = day_dir / "phase7_labeling_leaderboard.csv"
        if lboard_path.exists():
            try:
                lb = pd.read_csv(lboard_path)
                if "vertical_barrier_ticks" in lb.columns:
                    for _, row in lb.iterrows():
                        vb = str(int(row["vertical_barrier_ticks"]))
                        pt = str(row["pt_ticks"])
                        sl = str(row["sl_ticks"])
                        cname = f"{vb}/{pt}/{sl}"
                        if cname not in seen_cnames:
                            candidates.append((cname, lboard_path))
                            seen_cnames.add(cname)
            except Exception:
                pass

    return candidates


def load_label_csv(label_dir: Path) -> pd.DataFrame | None:
    """Load label data from a directory or leaderboard file.

    - Directory: find first .csv inside the label directory
    - Leaderboard file: parse as leaderboard (multiple candidates in one file)
      by expanding rows into per-candidate DataFrames (used for aggregation only)
    """
    # If label_dir is actually the leaderboard CSV path
    if label_dir.suffix == ".csv":
        try:
            return pd.read_csv(label_dir, low_memory=False)
        except Exception:
            return None

    csv_files = sorted(label_dir.glob("*.csv"))
    if not csv_files:
        return None
    try:
        return pd.read_csv(csv_files[0], low_memory=False)
    except Exception:
        return None


def compute_label_metrics(df: pd.DataFrame, cname: str, date: str) -> dict:
    """Compute PT/SL/V percentages and balance ratio for one day+candidate.

    Handles two input formats:
    - Individual label CSV: df has a 'barrier_hit' column (+1/-1/0 or 'pt'/'sl'/'vertical')
    - Leaderboard CSV: df has aggregated columns (pct_pt, pct_sl, n_events, etc.)
      In this case cname is the candidate key and df is a single-row subset.
    """
    # Detect leaderboard format: has aggregated pct columns
    if "pct_pt" in df.columns and "pct_sl" in df.columns:
        row = df.iloc[0]
        return {
            "date": date,
            "candidate": cname,
            "n_events": int(row.get("n_events", 0)),
            "n_pt": int(row.get("n_pt", 0)),
            "n_sl": int(row.get("n_sl", 0)),
            "n_vertical": int(row.get("n_vertical", 0)),
            "pct_pt": round(float(row.get("pct_pt", 0)), 2),
            "pct_sl": round(float(row.get("pct_sl", 0)), 2),
            "pct_vertical": round(float(row.get("pct_vertical", 0)), 2),
            "barrier_hit_pct": round(float(row.get("pct_pt", 0)) + float(row.get("pct_sl", 0)), 2),
            "balance_ratio": round(float(row.get("balance_ratio", 0)), 4),
            "win_rate": round(float(row.get("win_rate", 0)), 2),
        }

    # Standard per-row label CSV
    total = len(df)
    if total == 0:
        return {}

    n_pt = int((df["barrier_hit"] == "pt").sum())
    n_sl = int((df["barrier_hit"] == "sl").sum())
    n_v  = int((df["barrier_hit"] == "vertical").sum())

    denom = n_pt + n_sl
    balance = min(n_pt, n_sl) / max(n_pt, n_sl) if max(n_pt, n_sl) > 0 else 0.0
    win_rate = n_pt / denom if denom > 0 else 0.0

    return {
        "date": date,
        "candidate": cname,
        "n_events": total,
        "n_pt": n_pt,
        "n_sl": n_sl,
        "n_vertical": n_v,
        "pct_pt": round(n_pt / total * 100, 2),
        "pct_sl": round(n_sl / total * 100, 2),
        "pct_vertical": round(n_v / total * 100, 2),
        "barrier_hit_pct": round((n_pt + n_sl) / total * 100, 2),
        "balance_ratio": round(balance, 4),
        "win_rate": round(win_rate * 100, 2),
    }


def audit_labels(days: list[Path]) -> tuple[list[dict], list[dict]]:
    """Audit label distribution across all days. Returns (daily, aggregate)."""
    daily_rows = []
    aggregate_map = defaultdict(lambda: {"n_events": 0, "n_pt": 0, "n_sl": 0, "n_vertical": 0})

    for day_dir in days:
        date = day_dir.name
        label_dirs = find_label_dirs(day_dir)

        if not label_dirs:
            daily_rows.append({
                "date": date, "candidate": "NONE", "n_events": 0,
                "n_pt": 0, "n_sl": 0, "n_vertical": 0,
                "pct_pt": 0, "pct_sl": 0, "pct_vertical": 0,
                "barrier_hit_pct": 0, "balance_ratio": 0, "win_rate": 0,
            })
            continue

        for cname, ldir in label_dirs:
            df = load_label_csv(ldir)
            if df is None:
                continue

            # Handle leaderboard CSV (no 'barrier_hit' column, uses pct_* columns)
            if "barrier_hit" not in df.columns and "pct_pt" in df.columns:
                # Leaderboard format: each row is a different candidate
                for _, row in df.iterrows():
                    metrics = compute_label_metrics(
                        pd.DataFrame([row]), f"{int(row['vertical_barrier_ticks'])}/{row['pt_ticks']}/{row['sl_ticks']}", date
                    )
                    if metrics:
                        daily_rows.append(metrics)
                        key = metrics["candidate"]
                        aggregate_map[key]["n_events"]  += metrics["n_events"]
                        aggregate_map[key]["n_pt"]       += metrics["n_pt"]
                        aggregate_map[key]["n_sl"]       += metrics["n_sl"]
                        aggregate_map[key]["n_vertical"] += metrics["n_vertical"]
                continue

            if "barrier_hit" not in df.columns:
                continue

            metrics = compute_label_metrics(df, cname, date)
            if metrics:
                daily_rows.append(metrics)
                key = cname
                aggregate_map[key]["n_events"]  += metrics["n_events"]
                aggregate_map[key]["n_pt"]       += metrics["n_pt"]
                aggregate_map[key]["n_sl"]       += metrics["n_sl"]
                aggregate_map[key]["n_vertical"] += metrics["n_vertical"]

    # Compute aggregate
    aggregate_rows = []
    for cname, agg in aggregate_map.items():
        n = agg["n_events"]
        pt = agg["n_pt"]
        sl = agg["n_sl"]
        v  = agg["n_vertical"]
        denom = pt + sl
        balance = min(pt, sl) / max(pt, sl) if max(pt, sl) > 0 else 0.0
        win_rate = pt / denom if denom > 0 else 0.0
        aggregate_rows.append({
            "candidate": cname,
            "n_days": len([r for r in daily_rows if r["candidate"] == cname and r["n_events"] > 0]),
            "n_events_total": n,
            "n_pt_total": pt,
            "n_sl_total": sl,
            "n_vertical_total": v,
            "pct_pt": round(pt / n * 100, 2) if n > 0 else 0,
            "pct_sl": round(sl / n * 100, 2) if n > 0 else 0,
            "pct_vertical": round(v / n * 100, 2) if n > 0 else 0,
            "barrier_hit_pct": round((pt + sl) / n * 100, 2) if n > 0 else 0,
            "balance_ratio": round(balance, 4),
            "win_rate": round(win_rate * 100, 2),
        })

    # Outlier days (balance < threshold)
    outlier_rows = [r for r in daily_rows if r["balance_ratio"] < BALANCE_THRESHOLD and r["n_events"] > 0]

    return daily_rows, aggregate_rows, outlier_rows


# ════════════════════════════════════════════════════════════════════════════════
# B2 — QUALITY AUDIT
# ════════════════════════════════════════════════════════════════════════════════

def count_rows(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    try:
        return sum(1 for _ in open(file_path, encoding="utf-8", errors="ignore")) - 1  # -1 header
    except Exception:
        return 0


def nan_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 100.0
    total = df.shape[0] * df.shape[1]
    if total == 0:
        return 0.0
    return round(df.isna().sum().sum() / total * 100, 4)


def nan_rate_per_col(df: pd.DataFrame) -> dict:
    """NaN rate per column."""
    result = {}
    for col in df.columns:
        n_nan = df[col].isna().sum()
        result[col] = round(n_nan / len(df) * 100, 4) if len(df) > 0 else 0.0
    return result


def count_duplicates(ts_series: pd.Series) -> int:
    return int(ts_series.duplicated().sum())


def audit_quality(days: list[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    """Audit data quality across all days. Returns (row_counts, nan_rates, anomalies)."""
    row_count_rows = []
    nan_rate_rows = []
    anomaly_rows = []

    # Files per phase
    PHASE_FILES = {
        "p1_events":        "events.csv",
        "p2_snapshots":     "snapshots.csv",
        "p2b_fused":        "snapshots_fused.csv",
        "p3_features":      "features_dom.csv",
        "p4_agg":           "features_dom_agg.csv",
        "p5_sampled":       "sampled_events.csv",
        "p6_excursion":     "excursion_stats.csv",
    }

    all_row_counts = defaultdict(list)

    # First pass: collect row counts for all days
    for day_dir in days:
        for key, fname in PHASE_FILES.items():
            fpath = day_dir / fname
            count = count_rows(fpath)
            all_row_counts[key].append(count)

    # Compute statistics
    row_stats = {}
    for key, counts in all_row_counts.items():
        arr = np.array(counts)
        arr = arr[arr > 0]  # exclude zeros
        if len(arr) > 0:
            row_stats[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        else:
            row_stats[key] = {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0}

    # Second pass: per-day analysis
    for day_dir in days:
        date = day_dir.name
        rc_row = {"date": date}
        nan_row = {"date": date}
        anomaly_row = {"date": date, "flags": ""}
        flags = []

        for key, fname in PHASE_FILES.items():
            fpath = day_dir / fname
            count = count_rows(fpath)
            rc_row[key] = count

            if count > 0:
                # NaN rate from sampled_events and excursion_stats (largest files)
                if key in ("p5_sampled", "p6_excursion", "p3_features"):
                    try:
                        df = pd.read_csv(fpath, low_memory=False, nrows=10000)
                        rate = nan_rate(df)
                        nan_row[f"{key}_nan_pct"] = rate
                        if rate > NAN_RATE_WARN:
                            flags.append(f"{key} NaN={rate:.2f}%")
                    except Exception:
                        nan_row[f"{key}_nan_pct"] = -1
                        flags.append(f"{key} read_error")
                else:
                    nan_row[f"{key}_nan_pct"] = -1

                # Anomaly: row count deviates > N std devs
                if key in row_stats and row_stats[key]["std"] > 0:
                    z = abs(count - row_stats[key]["mean"]) / row_stats[key]["std"]
                    if z > ROW_COUNT_STD_THRESH:
                        flags.append(f"{key} count={count} (z={z:.1f})")

                # Duplicates in sampled_events
                if key == "p5_sampled":
                    try:
                        df_ts = pd.read_csv(fpath, usecols=["ts"], low_memory=False, nrows=100000)
                        n_dup = count_duplicates(df_ts["ts"])
                        dup_pct = round(n_dup / len(df_ts) * 100, 4) if len(df_ts) > 0 else 0
                        nan_row["p5_dup_ts_pct"] = dup_pct
                        if dup_pct > 1.0:
                            flags.append(f"p5 duplicates={n_dup} ({dup_pct:.2f}%)")
                    except Exception:
                        nan_row["p5_dup_ts_pct"] = -1
                else:
                    nan_row["p5_dup_ts_pct"] = -1
            else:
                nan_row[f"{key}_nan_pct"] = -1
                nan_row["p5_dup_ts_pct"] = -1
                if key in ("p5_sampled", "p6_excursion"):
                    flags.append(f"{key} MISSING")

        row_count_rows.append(rc_row)
        nan_rate_rows.append(nan_row)

        if flags:
            anomaly_row["flags"] = " | ".join(flags)
            anomaly_rows.append(anomaly_row)

    return row_count_rows, nan_rate_rows, anomaly_rows


# ════════════════════════════════════════════════════════════════════════════════
# REPORTS
# ════════════════════════════════════════════════════════════════════════════════

def write_label_report(daily: list[dict], aggregate: list[dict], outliers: list[dict], out_dir: Path):
    lines = []
    lines.append("=" * 90)
    lines.append("LABEL DISTRIBUTION AUDIT -- 20 Days")
    lines.append(f"Generated: {pd.Timestamp.now()}")
    lines.append("=" * 90)

    # Per-day table
    lines.append("\n## Per-Day Label Distribution\n")
    lines.append(f"{'DATE':<12} {'CANDIDATE':<25} {'N_EVENTS':>10} "
                  f"{'PT%':>6} {'SL%':>6} {'V%':>6} "
                  f"{'BALANCE':>8} {'WIN%':>7}")
    lines.append("-" * 90)
    for r in sorted(daily, key=lambda x: (x["date"], x["candidate"])):
        if r["n_events"] > 0:
            lines.append(
                f"{r['date']:<12} {r['candidate']:<25} {r['n_events']:>10,} "
                f"{r['pct_pt']:>6.1f} {r['pct_sl']:>6.1f} {r['pct_vertical']:>6.1f} "
                f"{r['balance_ratio']:>8.4f} {r['win_rate']:>7.2f}"
            )

    # Aggregate table
    lines.append("\n## Aggregate Across All Days\n")
    lines.append(f"{'CANDIDATE':<25} {'N_DAYS':>7} {'N_EVENTS':>10} "
                  f"{'PT%':>6} {'SL%':>6} {'V%':>6} "
                  f"{'BALANCE':>8} {'WIN%':>7}")
    lines.append("-" * 90)
    for r in sorted(aggregate, key=lambda x: -x["balance_ratio"]):
        lines.append(
            f"{r['candidate']:<25} {r['n_days']:>7} {r['n_events_total']:>10,} "
            f"{r['pct_pt']:>6.1f} {r['pct_sl']:>6.1f} {r['pct_vertical']:>6.1f} "
            f"{r['balance_ratio']:>8.4f} {r['win_rate']:>7.2f}"
        )

    # Outlier days
    if outliers:
        lines.append("\n## Outlier Days (balance_ratio < 0.80)\n")
        for r in outliers:
            lines.append(f"  {r['date']} | {r['candidate']} | balance={r['balance_ratio']:.4f} | "
                          f"PT={r['pct_pt']:.1f}% SL={r['pct_sl']:.1f}% V={r['pct_vertical']:.1f}%")
    else:
        lines.append("\n## Outlier Days: NONE -- all days above balance_ratio threshold.")

    # Summary stats
    valid_rows = [r for r in daily if r["n_events"] > 0]
    if valid_rows:
        bal_ratios = [r["balance_ratio"] for r in valid_rows]
        lines.append(f"\n## Summary Statistics\n")
        lines.append(f"  Days audited:     {len(set(r['date'] for r in valid_rows))}")
        lines.append(f"  Balance ratio:    mean={np.mean(bal_ratios):.4f}  "
                     f"min={np.min(bal_ratios):.4f}  max={np.max(bal_ratios):.4f}  "
                     f"std={np.std(bal_ratios):.4f}")
        below = sum(1 for b in bal_ratios if b < BALANCE_THRESHOLD)
        lines.append(f"  Days below {BALANCE_THRESHOLD}:  {below}/{len(bal_ratios)}")

    lines.append("\n" + "=" * 90)

    path = out_dir / "label_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Label summary -> {path}")


def write_quality_report(row_counts: list[dict], nan_rates: list[dict],
                          anomalies: list[dict], out_dir: Path, days: list[Path]):
    lines = []
    lines.append("=" * 90)
    lines.append("DATA QUALITY AUDIT -- 20 Days")
    lines.append(f"Generated: {pd.Timestamp.now()}")
    lines.append("=" * 90)

    # Row counts
    lines.append("\n## Row Counts Per Day\n")
    if row_counts:
        keys = [k for k in row_counts[0].keys() if k != "date"]
        header = f"{'DATE':<12}" + "".join(f"{k:>12}" for k in keys)
        lines.append(header)
        lines.append("-" * (12 + 12 * len(keys)))
        for r in row_counts:
            row_str = f"{r['date']:<12}" + "".join(f"{(r.get(k, 0) or 0):>12,}" for k in keys)
            lines.append(row_str)

        # Summary stats
        lines.append("\n## Row Count Statistics\n")
        for k in keys:
            vals = [r.get(k, 0) or 0 for r in row_counts]
            vals_nz = [v for v in vals if v > 0]
            if vals_nz:
                lines.append(f"  {k:25s}: mean={np.mean(vals_nz):>10,.0f}  "
                              f"median={np.median(vals_nz):>10,.0f}  "
                              f"std={np.std(vals_nz):>8,.0f}  "
                              f"min={np.min(vals_nz):>10,}  max={np.max(vals_nz):>10,}")
            else:
                lines.append(f"  {k:25s}: no data")

    # NaN rates
    lines.append("\n## NaN Rate (% of cells) -- top files\n")
    nan_keys = [k for k in (nan_rates[0].keys() if nan_rates else []) if k != "date"]
    if nan_keys:
        header = f"{'DATE':<12}" + "".join(f"{k:>10}" for k in nan_keys[:8])
        lines.append(header)
        lines.append("-" * (12 + 10 * min(8, len(nan_keys))))
        for r in nan_rates:
            row_str = f"{r['date']:<12}" + "".join(
                f"{(r.get(k, -1) or -1):>10.2f}" for k in nan_keys[:8]
            )
            lines.append(row_str)

    # Anomalies
    lines.append("\n## Anomaly Flags\n")
    if anomalies:
        for r in anomalies:
            lines.append(f"  {r['date']:<12}: {r['flags']}")
    else:
        lines.append("  No anomalies detected.")

    # Missing files
    lines.append("\n## Missing Files Per Day\n")
    for day_dir in days:
        date = day_dir.name
        missing = []
        for fname in ["events.csv", "snapshots.csv", "features_dom.csv",
                      "sampled_events.csv", "excursion_stats.csv"]:
            if not (day_dir / fname).exists():
                missing.append(fname)
        if missing:
            lines.append(f"  {date}: {', '.join(missing)}")

    lines.append("\n" + "=" * 90)

    path = out_dir / "quality_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Quality summary -> {path}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="P5-P7 Data Quality & Label Distribution Audit (LOCAL only)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Path to NQdom/output directory (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    audit_dir = output_dir / AUDIT_DIR
    ensure_dir(audit_dir)

    print(f"\n{'='*70}")
    print("P5-P7 AUDIT -- Data Quality & Label Distribution")
    print(f"{'='*70}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Audit dir   : {audit_dir}")

    days = find_day_dirs(output_dir)
    print(f"  Days found  : {len(days)}")
    for d in days:
        print(f"    {d.name}")
    print()

    if not days:
        print("  ERROR: No day directories found.")
        sys.exit(1)

    # ── B1: Label Distribution ──────────────────────────────────────────────────
    print("[B1] Label Distribution Audit ..")
    daily_labels, agg_labels, outlier_days = audit_labels(days)

    label_daily_path = audit_dir / "label_distribution_daily.csv"
    label_agg_path = audit_dir / "label_distribution_aggregate.csv"
    label_outlier_path = audit_dir / "label_outlier_days.csv"

    write_csv(daily_labels, label_daily_path)
    write_csv(agg_labels, label_agg_path)
    write_csv(outlier_days, label_outlier_path)

    print(f"  Daily    -> {label_daily_path}  ({len(daily_labels)} rows)")
    print(f"  Aggregate -> {label_agg_path}  ({len(agg_labels)} rows)")
    print(f"  Outliers  -> {label_outlier_path}  ({len(outlier_days)} rows)")

    write_label_report(daily_labels, agg_labels, outlier_days, audit_dir)

    # ── B2: Quality Audit ──────────────────────────────────────────────────────
    print("\n[B2] Quality Audit ..")
    row_counts, nan_rates, anomalies = audit_quality(days)

    row_count_path = audit_dir / "quality_row_counts.csv"
    nan_rate_path = audit_dir / "quality_nan_rates.csv"
    anomaly_path = audit_dir / "quality_anomalies.csv"

    write_csv(row_counts, row_count_path)
    write_csv(nan_rates, nan_rate_path)
    write_csv(anomalies, anomaly_path)

    print(f"  Row counts -> {row_count_path}  ({len(row_counts)} rows)")
    print(f"  NaN rates  -> {nan_rate_path}  ({len(nan_rates)} rows)")
    print(f"  Anomalies  -> {anomaly_path}  ({len(anomalies)} rows)")

    write_quality_report(row_counts, nan_rates, anomalies, audit_dir, days)

    # ── Done ───────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"AUDIT COMPLETE -- All outputs in:")
    print(f"  {audit_dir}/")
    print(f"  label_distribution_daily.csv  |  label_distribution_aggregate.csv")
    print(f"  label_outlier_days.csv       |  label_summary.txt")
    print(f"  quality_row_counts.csv       |  quality_nan_rates.csv")
    print(f"  quality_anomalies.csv         |  quality_summary.txt")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
