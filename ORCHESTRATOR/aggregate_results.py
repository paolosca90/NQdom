#!/usr/bin/env python3
"""
aggregate_results.py — Multi-Day P8 Results & Excursion Aggregator
===================================================================
Scans all days where P8 is complete, reads phase8_trainval_results.csv,
and produces aggregate CSVs + a human-readable summary.

OUTPUTS (to --agg-dir /opt/depth-dom/output/aggregate/):
  multi_day_p8_results.csv     — concatenated per-day results
  excursion_aggregate.csv      — excursion statistics aggregated
  feature_importance_ranked.csv — feature importance by mean rank
  daily_metrics.csv            — per-day accuracy / F1 / precision / recall
  summary_report.txt           — human-readable summary

USAGE
    python3 aggregate_results.py \\
        --output-dir /opt/depth-dom/output \\
        --agg-dir    /opt/depth-dom/output/aggregate \\
        --symbol ES  \\
        --min-days 5
"""

import argparse
import csv
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
VPS_BASE = "/opt/depth-dom"
EXCURSION_TARGET = {
    "ES": {"30t": 10.0,  "60t": 10.0,  "120t": 10.0},   # ES ±10 points
    "NQ": {"30t": 20.0,  "60t": 20.0,  "120t": 20.0},   # NQ ±20 points
}
P8_SENTINEL = "p8_ml.done"
MANIFEST = "_p7p8_incremental_manifest.csv"


# ── Helpers ────────────────────────────────────────────────────────────────────

def sentinel_done(out_dir: Path, phase: str) -> bool:
    return (out_dir / "_checkpoints" / f"{phase}.done").exists()


def p8_complete(output_dir: Path, date: str) -> bool:
    day_dir = output_dir / date
    return sentinel_done(day_dir, "p8_ml") and (day_dir / "phase8_trainval_results.csv").exists()


def disk_free_gb(output_dir: Path) -> float:
    try:
        stat = os.statvfs(output_dir)
        return stat.f_bavail * stat.f_frsize / (1024**3)
    except Exception:
        return 999.0


import os  # deferring this to avoid top-level import issues on some systems


def discover_p8_days(output_dir: Path) -> list[str]:
    """Find all days where P8 is complete."""
    days = []
    if not output_dir.exists():
        return days
    for day_dir in sorted(output_dir.iterdir()):
        if not day_dir.is_dir() or not day_dir.name.startswith("20"):
            continue
        if p8_complete(output_dir, day_dir.name):
            days.append(day_dir.name)
    return days


def read_phase8_results(output_dir: Path, date: str) -> pd.DataFrame | None:
    """Read phase8_trainval_results.csv for one day."""
    path = output_dir / date / "phase8_trainval_results.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None


def read_feature_importance(output_dir: Path, date: str) -> pd.DataFrame | None:
    """Read phase8_feature_importance.csv for one day."""
    path = output_dir / date / "phase8_feature_importance.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_excursion_stats(output_dir: Path, date: str) -> pd.DataFrame | None:
    """Read excursion_stats.csv for one day."""
    path = output_dir / date / "excursion_stats.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None


# ── Aggregate P8 Results ───────────────────────────────────────────────────────

def aggregate_p8_results(output_dir: Path, days: list[str]) -> pd.DataFrame:
    """Concatenate all phase8_trainval_results.csv into one DataFrame."""
    rows = []
    for date in days:
        df = read_phase8_results(output_dir, date)
        if df is None:
            continue
        df = df.copy()
        df.insert(0, "date", date)
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_daily_metrics(p8_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-day aggregate metrics from the concatenated p8 results.
    Identifies the best candidate per day (highest accuracy) and extracts metrics.
    """
    if p8_df.empty:
        return pd.DataFrame()

    records = []
    for date, grp in p8_df.groupby("date", sort=True):
        # Find best candidate (highest accuracy)
        if "accuracy" not in grp.columns:
            continue
        best = grp.loc[grp["accuracy"].idxmax()]
        record = {
            "date": date,
            "best_candidate": best.get("candidate", best.get("candidate_id", "?")),
            "accuracy": best.get("accuracy", math.nan),
            "balanced_accuracy": best.get("balanced_accuracy", math.nan),
            "f1_macro": best.get("f1_macro", best.get("f1", math.nan)),
            "precision_macro": best.get("precision_macro", best.get("precision", math.nan)),
            "recall_macro": best.get("recall_macro", best.get("recall", math.nan)),
        }
        # Extract per-class metrics if available
        for cls in ["short", "flat", "long"]:
            for metric in ["precision", "recall", "f1"]:
                col = f"{metric}_{cls}"
                if col in grp.columns:
                    record[col] = best.get(col, math.nan)
        records.append(record)

    return pd.DataFrame(records)


def aggregate_feature_importance(output_dir: Path, days: list[str]) -> pd.DataFrame:
    """
    Read all phase8_feature_importance.csv files and compute mean rank per feature.
    Returns DataFrame with feature, mean_rank, n_days, std_rank.
    """
    all_fi = []
    for date in days:
        fi_df = read_feature_importance(output_dir, date)
        if fi_df is None:
            continue
        # Identify feature and importance columns
        feat_col = None
        imp_col = None
        for col in fi_df.columns:
            lcol = col.lower()
            if "feature" in lcol and feat_col is None:
                feat_col = col
            if "importance" in lcol or "gain" in lcol:
                imp_col = col
        if feat_col is None or imp_col is None:
            # Fallback: first col = feature, second = importance
            feat_col = fi_df.columns[0]
            imp_col = fi_df.columns[1] if len(fi_df.columns) > 1 else fi_df.columns[0]

        fi_df = fi_df.copy()
        fi_df["date"] = date
        fi_df["rank"] = fi_df[imp_col].rank(ascending=False)
        all_fi.append(fi_df[["date", feat_col, imp_col, "rank"]].rename(columns={feat_col: "feature"}))

    if not all_fi:
        return pd.DataFrame()

    combined = pd.concat(all_fi, ignore_index=True)
    agg = combined.groupby("feature")["rank"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["feature", "mean_rank", "std_rank", "n_days"]
    agg = agg.sort_values("mean_rank").reset_index(drop=True)
    return agg


# ── Aggregate Excursion Stats ──────────────────────────────────────────────────

def compute_excursion_aggregate(output_dir: Path, days: list[str],
                                 symbol: str = "ES") -> pd.DataFrame:
    """
    Aggregate excursion_stats.csv across all days to produce per-day
    excursion hit rates and median time-to-target.

    Hit rate: % of events where |max_down_X_ticks| >= threshold (for ES: 40 ticks = 10pts)
    For ES: ±10 pts = ±40 ticks (0.25 tick size)
    For NQ: ±20 pts = ±80 ticks (0.25 tick size)
    """
    tick_target = 40 if symbol == "ES" else 80  # ticks corresponding to ±10pt (ES) or ±20pt (NQ)

    records = []
    for date in days:
        df = read_excursion_stats(output_dir, date)
        if df is None:
            continue

        record = {"date": date}

        # P6 source columns use "30s" suffix (fixed feature column names);
        # output columns use new "30t" suffix.
        for p6_vb, out_vb in [("30s", "30t"), ("60s", "60t"), ("120s", "120t")]:
            # Excursion hit rate: how many events reach the target in each direction
            up_col = f"max_up_{p6_vb}_ticks"
            down_col = f"max_down_{p6_vb}_ticks"
            mae_col = f"mae_{p6_vb}_ticks"
            n_obs_col = f"n_obs_{p6_vb}"

            if up_col not in df.columns:
                continue

            up_vals = pd.to_numeric(df[up_col], errors="coerce")
            down_vals = pd.to_numeric(df[down_col], errors="coerce")
            mae_vals = pd.to_numeric(df[mae_col], errors="coerce")
            n_obs = pd.to_numeric(df[n_obs_col], errors="coerce") if n_obs_col in df.columns else None

            n_total = len(df)
            n_up_hit = (up_vals >= tick_target).sum()
            n_down_hit = (down_vals >= tick_target).sum()
            n_mae_valid = mae_vals.notna().sum()

            record[f"n_events_{out_vb}"] = n_total
            record[f"n_up_hit_{out_vb}"] = int(n_up_hit)
            record[f"n_down_hit_{out_vb}"] = int(n_down_hit)
            record[f"pct_up_hit_{out_vb}"] = round(n_up_hit / n_total * 100, 2) if n_total > 0 else 0
            record[f"pct_down_hit_{out_vb}"] = round(n_down_hit / n_total * 100, 2) if n_total > 0 else 0
            record[f"median_mae_{out_vb}_ticks"] = round(mae_vals.median(), 2) if n_mae_valid > 0 else math.nan
            horizon_col = f"horizon_end_ts_{p6_vb}"
            if horizon_col in df.columns:
                try:
                    start_ts = pd.to_numeric(df["ts"].str.extract(r"(\d+)$")[0], errors="coerce")
                    end_ts = pd.to_numeric(df[horizon_col].astype(str).str[-18:], errors="coerce")
                    win_col = f"window_complete_{p6_vb}"
                    win_complete = pd.to_numeric(df[win_col], errors="coerce") == 1
                    elapsed_s = (end_ts - start_ts) / 1e6  # nanoseconds to ms
                    valid_elapsed = elapsed_s[win_complete & elapsed_s.notna()]
                    if len(valid_elapsed) > 0:
                        record[f"median_time_to_target_{out_vb}_s"] = round(valid_elapsed.median() / 1000, 1)
                    else:
                        record[f"median_time_to_target_{out_vb}_s"] = math.nan
                except Exception:
                    record[f"median_time_to_target_{out_vb}_s"] = math.nan
            else:
                record[f"median_time_to_target_{out_vb}_s"] = math.nan

        records.append(record)

    return pd.DataFrame(records)


def aggregate_excursion_overall(esc_df: pd.DataFrame) -> dict:
    """Compute overall excursion statistics across all days."""
    if esc_df.empty:
        return {}
    result = {}
    for out_vb in ["30t", "60t", "120t"]:
        n_col = f"n_events_{out_vb}"
        up_col = f"pct_up_hit_{out_vb}"
        down_col = f"pct_down_hit_{out_vb}"
        mae_col = f"median_mae_{out_vb}_ticks"
        ttt_col = f"median_time_to_target_{out_vb}_s"
        if n_col not in esc_df.columns:
            continue
        total_events = esc_df[n_col].sum()
        total_up = esc_df[f"n_up_hit_{out_vb}"].sum()
        total_down = esc_df[f"n_down_hit_{out_vb}"].sum()
        result[f"overall_pct_up_hit_{out_vb}"] = round(total_up / total_events * 100, 2) if total_events > 0 else 0
        result[f"overall_pct_down_hit_{out_vb}"] = round(total_down / total_events * 100, 2) if total_events > 0 else 0
        all_mae = esc_df[mae_col].dropna()
        result[f"overall_median_mae_{out_vb}_ticks"] = round(all_mae.median(), 2) if len(all_mae) > 0 else math.nan
        all_ttt = esc_df[ttt_col].dropna()
        result[f"overall_median_ttt_{out_vb}_s"] = round(all_ttt.median(), 1) if len(all_ttt) > 0 else math.nan
    return result


# ── Class Distribution ─────────────────────────────────────────────────────────

def compute_class_distribution(output_dir: Path, days: list[str]) -> pd.DataFrame:
    """
    Read phase8_oof_predictions.csv per day and compute label distribution.
    Falls back to reading from phase8_trainval_results.csv if oof not available.
    """
    records = []
    for date in days:
        oof_path = output_dir / date / "phase8_oof_predictions.csv"
        results_path = output_dir / date / "phase8_trainval_results.csv"
        dist = {}
        dist["date"] = date

        if oof_path.exists():
            try:
                oof = pd.read_csv(oof_path)
                for cls in ["short", "flat", "long"]:
                    if cls in oof.columns:
                        dist[f"n_{cls}"] = int((oof[cls] == 1).sum())
                        dist[f"pct_{cls}"] = round((oof[cls] == 1).sum() / len(oof) * 100, 2)
            except Exception:
                pass
        elif results_path.exists():
            try:
                res = pd.read_csv(results_path)
                for cls in ["short", "flat", "long"]:
                    n_col = f"n_{cls}"
                    if n_col in res.columns:
                        dist[n_col] = int(res[n_col].sum())
                        dist[f"pct_{cls}"] = round(res[n_col].sum() / res[n_col].sum() * 100, 2)
            except Exception:
                pass

        if len(dist) > 1:
            records.append(dist)

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Write outputs ─────────────────────────────────────────────────────────────

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path):
    if df.empty:
        path.write_text("", encoding="utf-8")
        return
    df.to_csv(path, index=False, encoding="utf-8")


def write_summary(p8_df: pd.DataFrame,
                  daily_metrics: pd.DataFrame,
                  fi_agg: pd.DataFrame,
                  exc_agg: pd.DataFrame,
                  exc_overall: dict,
                  class_dist: pd.DataFrame,
                  agg_dir: Path,
                  symbol: str):
    lines = [
        "=" * 70,
        "MULTI-DAY PIPELINE AGGREGATE SUMMARY",
        f"Generated: {dt.datetime.now().isoformat()}",
        f"Symbol: {symbol}",
        "=" * 70,
        "",
    ]

    if not daily_metrics.empty:
        lines.append(f"Days with P8 complete: {len(daily_metrics)}")
        lines.append("")
        lines.append("-- Daily Metrics (best candidate per day) --")
        cols = ["date", "accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"]
        available = [c for c in cols if c in daily_metrics.columns]
        lines.append(daily_metrics[available].to_string(index=False))
        lines.append("")

        # Overall averages
        num_cols = ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"]
        avg_cols = [c for c in num_cols if c in daily_metrics.columns]
        if avg_cols:
            avgs = daily_metrics[avg_cols].mean()
            lines.append("-- Overall Average Metrics --")
            for c, v in avgs.items():
                lines.append(f"  {c:30s}: {v:.4f}")
            lines.append("")

    if not fi_agg.empty:
        lines.append("-- Top 20 Features (by mean rank) --")
        top20 = fi_agg.head(20)[["feature", "mean_rank", "n_days"]]
        lines.append(top20.to_string(index=False))
        lines.append("")

    if not exc_agg.empty:
        lines.append("-- Excursion Hit Rates --")
        for out_vb in ["30t", "60t", "120t"]:
            up_col = f"pct_up_hit_{out_vb}"
            down_col = f"pct_down_hit_{out_vb}"
            mae_col = f"median_mae_{out_vb}_ticks"
            if up_col in exc_agg.columns:
                avg_up = exc_agg[up_col].mean()
                avg_down = exc_agg[down_col].mean()
                all_mae = exc_agg[mae_col].dropna()
                median_mae = all_mae.median() if len(all_mae) > 0 else float("nan")
                lines.append(f"  {out_vb:5s}: up_hit={avg_up:.1f}%  down_hit={avg_down:.1f}%  "
                              f"median_mae={median_mae:.1f} ticks")
        lines.append("")

        if exc_overall:
            lines.append("-- Overall Excursion Averages --")
            for k, v in exc_overall.items():
                lines.append(f"  {k:40s}: {v}")
            lines.append("")

    if not class_dist.empty:
        lines.append("-- Label Class Distribution --")
        cols = ["date"] + [c for c in class_dist.columns if c != "date"]
        lines.append(class_dist.to_string(index=False))
        lines.append("")

    lines.append("=" * 70)
    path = agg_dir / "summary_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate P8 + excursion results")
    parser.add_argument("--output-dir", default=VPS_BASE + "/output",
                        help="Path to pipeline output directory")
    parser.add_argument("--agg-dir", default=VPS_BASE + "/output/aggregate",
                        help="Directory for aggregate outputs")
    parser.add_argument("--symbol", default="ES",
                        choices=["ES", "NQ"],
                        help="Symbol for excursion target thresholds")
    parser.add_argument("--min-days", type=int, default=1,
                        help="Minimum number of P8 days required to produce output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    agg_dir = Path(args.agg_dir)
    ensure_dir(agg_dir)

    print(f"=== aggregate_results.py ===  {dt.datetime.now().isoformat()}")
    print(f"  output-dir : {output_dir}")
    print(f"  agg-dir    : {agg_dir}")
    print(f"  symbol     : {args.symbol}")

    # Discover P8-complete days
    days = discover_p8_days(output_dir)
    print(f"\n  P8-complete days: {len(days)}")
    for d in days:
        print(f"    {d}")

    if len(days) < args.min_days:
        print(f"\n  [SKIP] Only {len(days)} days < min-days={args.min_days}")
        return

    # Disk safety
    free_gb = disk_free_gb(output_dir)
    print(f"  free-disk : {free_gb:.1f} GB")

    # ── Aggregate P8 results ───────────────────────────────────────────────────
    print("\n[1/5] Aggregating P8 trainval results ...")
    p8_df = aggregate_p8_results(output_dir, days)
    print(f"  Rows: {len(p8_df)}")
    if not p8_df.empty:
        write_csv(p8_df, agg_dir / "multi_day_p8_results.csv")

    # ── Daily metrics ───────────────────────────────────────────────────────────
    print("[2/5] Computing daily metrics ...")
    daily_metrics = compute_daily_metrics(p8_df)
    print(f"  Days with metrics: {len(daily_metrics)}")
    if not daily_metrics.empty:
        write_csv(daily_metrics, agg_dir / "daily_metrics.csv")

    # ── Feature importance ──────────────────────────────────────────────────────
    print("[3/5] Aggregating feature importance ...")
    fi_agg = aggregate_feature_importance(output_dir, days)
    print(f"  Features ranked: {len(fi_agg)}")
    if not fi_agg.empty:
        write_csv(fi_agg, agg_dir / "feature_importance_ranked.csv")

    # ── Excursion stats ────────────────────────────────────────────────────────
    print("[4/5] Aggregating excursion statistics ...")
    exc_agg = compute_excursion_aggregate(output_dir, days, symbol=args.symbol)
    print(f"  Days with excursion data: {len(exc_agg)}")
    if not exc_agg.empty:
        write_csv(exc_agg, agg_dir / "excursion_aggregate.csv")

    exc_overall = aggregate_excursion_overall(exc_agg)

    # ── Class distribution ──────────────────────────────────────────────────────
    print("[5/5] Computing class distribution ...")
    class_dist = compute_class_distribution(output_dir, days)
    if not class_dist.empty:
        write_csv(class_dist, agg_dir / "class_distribution.csv")

    # ── Summary report ──────────────────────────────────────────────────────────
    print("\n[+] Writing summary report ...")
    write_summary(p8_df, daily_metrics, fi_agg, exc_agg,
                  exc_overall, class_dist, agg_dir, args.symbol)

    print(f"\n=== Done. Outputs in {agg_dir} ===")


if __name__ == "__main__":
    main()
