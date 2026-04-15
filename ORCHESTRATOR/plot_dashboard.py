#!/usr/bin/env python3
"""
plot_dashboard.py — Multi-Day Pipeline Dashboard
================================================
Reads aggregate CSVs and produces a 3x2 panel PNG dashboard.

PANELS
    1  Daily Accuracy trend (line, x=date, y=accuracy + balanced_accuracy + f1)
    2  Excursion Hit Rate over time (stacked bar: up% vs down%)
    3  Feature Importance top-20 (horizontal bar, mean rank)
    4  Label class distribution per day (stacked bar)
    5  Median time-to-target trend (line, seconds to ±10/20pt per horizon)
    6  Confusion matrix heatmap (aggregated across all days)

OUTPUT
    output/aggregate/dashboard.png  — 3x2 figure, 300 DPI
    output/aggregate/panel_1.png  ... panel_6.png  — individual panels

USAGE
    python3 plot_dashboard.py \\
        --agg-dir NQdom/output/aggregate \\
        --out-dir NQdom/output/aggregate \\
        --style dark \\
        --dpi 300
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ── Style defaults ─────────────────────────────────────────────────────────────

STYLE_DARK = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "monospace",
}

STYLE_LIGHT = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f6f8fa",
    "axes.edgecolor":    "#d0d7de",
    "axes.labelcolor":   "#1f2328",
    "xtick.color":       "#656d76",
    "ytick.color":       "#656d76",
    "text.color":        "#1f2328",
    "grid.color":        "#d0d7de",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "legend.facecolor":  "#f6f8fa",
    "legend.edgecolor":  "#d0d7de",
    "font.family":       "monospace",
}

PALETTE = {
    "accuracy":          "#58a6ff",   # blue
    "balanced_accuracy": "#a371f7",   # purple
    "f1":                "#3fb950",   # green
    "precision":         "#f0883e",   # orange
    "recall":            "#db6d28",   # dark orange
    "up_hit":            "#3fb950",   # green
    "down_hit":          "#f85149",   # red
    "short":             "#f85149",   # red
    "flat":              "#8b949e",   # grey
    "long":              "#3fb950",   # green
    "conf_matrix":       "Blues",
}


# ── CSV readers ────────────────────────────────────────────────────────────────

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ── Panel 1: Daily Accuracy trend ────────────────────────────────────────────

def plot_daily_accuracy(ax, daily_metrics: pd.DataFrame, style: str):
    if daily_metrics.empty or "date" not in daily_metrics.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#8b949e")
        return

    df = daily_metrics.sort_values("date")
    x = range(len(df))
    labels = df["date"].tolist()

    for col, color_key in [
        ("accuracy", "accuracy"),
        ("balanced_accuracy", "balanced_accuracy"),
        ("f1_macro", "f1"),
    ]:
        if col in df.columns:
            color = PALETTE.get(color_key, "#58a6ff")
            values = pd.to_numeric(df[col], errors="coerce").tolist()
            ax.plot(x, values, marker="o", markersize=3, linewidth=1.5,
                    color=color, label=col.replace("_", " ").title())

    ax.set_title("Daily Accuracy & F1 Trend", fontsize=10, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    ax.legend(fontsize=7, loc="lower right")

    if len(labels) <= 10:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    else:
        # Subsample
        step = max(1, len(labels) // 10)
        ax.set_xticks(list(x[::step]))
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)],
                           rotation=45, ha="right", fontsize=7)


# ── Panel 2: Excursion Hit Rate ───────────────────────────────────────────────

def plot_excursion_hit_rate(ax, exc_agg: pd.DataFrame, style: str):
    if exc_agg.empty or "date" not in exc_agg.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#8b949e")
        return

    df = exc_agg.sort_values("date")

    vbs = ["30t", "60t", "120t"]
    n_vbs = len(vbs)
    x = np.arange(len(df))
    width = 0.25

    for i, vb in enumerate(vbs):
        up_col = f"pct_up_hit_{vb}"
        down_col = f"pct_down_hit_{vb}"
        if up_col not in df.columns or down_col not in df.columns:
            continue
        up_vals = pd.to_numeric(df[up_col], errors="coerce").fillna(0).tolist()
        down_vals = [-v for v in pd.to_numeric(df[down_col], errors="coerce").fillna(0).tolist()]

        offset = (i - n_vbs / 2) * width
        bars_up = ax.bar([xi + offset for xi in x], up_vals, width,
                         color=PALETTE["up_hit"], alpha=0.8, label=f"Up {vb}")
        bars_down = ax.bar([xi + offset for xi in x], down_vals, width,
                           color=PALETTE["down_hit"], alpha=0.8, label=f"Down {vb}")

    ax.set_title("Excursion Hit Rate (Up vs Down %)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Hit Rate (%)")
    ax.axhline(0, color="#30363d", linewidth=0.8)
    ax.grid(True, axis="y")

    labels = df["date"].tolist()
    if len(labels) <= 10:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    else:
        step = max(1, len(labels) // 10)
        ax.set_xticks(list(x[::step]))
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)],
                           rotation=45, ha="right", fontsize=7)

    # Custom legend avoiding duplication
    handles, labs = ax.get_legend_handles_labels()
    unique = dict(zip(labs, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=6, loc="lower right",
              ncol=2)


# ── Panel 3: Feature Importance ───────────────────────────────────────────────

def plot_feature_importance(ax, fi_agg: pd.DataFrame, style: str):
    if fi_agg.empty or "feature" not in fi_agg.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#8b949e")
        return

    df = fi_agg.head(20).sort_values("mean_rank")
    features = df["feature"].tolist()
    ranks = pd.to_numeric(df["mean_rank"], errors="coerce").tolist()
    n_days = pd.to_numeric(df["n_days"], errors="coerce").fillna(0).tolist()

    colors = [PALETTE["accuracy"]] * len(df)
    ax.barh(range(len(df)), ranks, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(features, fontsize=7)
    ax.set_xlabel("Mean Rank (lower = more important)")
    ax.set_title("Feature Importance (Top 20, Mean Rank)", fontsize=10, fontweight="bold")
    ax.grid(True, axis="x")
    ax.invert_yaxis()

    # Annotate with n_days
    for i, (rank, nd) in enumerate(zip(ranks, n_days)):
        ax.text(rank + 0.5, i, f"n={int(nd)}", va="center", fontsize=6,
                color="#8b949e")


# ── Panel 4: Label Class Distribution ─────────────────────────────────────────

def plot_class_distribution(ax, class_dist: pd.DataFrame, style: str):
    if class_dist.empty or "date" not in class_dist.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#8b949e")
        return

    df = class_dist.sort_values("date")
    x = range(len(df))
    labels = df["date"].tolist()

    short_vals = pd.to_numeric(df.get("pct_short", pd.Series([0]*len(df))),
                               errors="coerce").fillna(0).tolist()
    flat_vals  = pd.to_numeric(df.get("pct_flat",  pd.Series([0]*len(df))),
                               errors="coerce").fillna(0).tolist()
    long_vals  = pd.to_numeric(df.get("pct_long",  pd.Series([0]*len(df))),
                               errors="coerce").fillna(0).tolist()

    ax.bar(x, short_vals, color=PALETTE["short"], label="Short", alpha=0.85, width=0.6)
    ax.bar(x, flat_vals,  color=PALETTE["flat"],  label="Flat",  alpha=0.85, width=0.6,
           bottom=short_vals)
    ax.bar(x, long_vals,  color=PALETTE["long"],  label="Long",  alpha=0.85, width=0.6,
           bottom=[s+f for s, f in zip(short_vals, flat_vals)])

    ax.set_title("Label Class Distribution (% per Day)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="y")
    ax.legend(fontsize=7, loc="upper right")

    if len(labels) <= 10:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    else:
        step = max(1, len(labels) // 10)
        ax.set_xticks(list(x[::step]))
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)],
                           rotation=45, ha="right", fontsize=7)


# ── Panel 5: Time-to-target Trend ────────────────────────────────────────────

def plot_time_to_target(ax, exc_agg: pd.DataFrame, style: str):
    if exc_agg.empty or "date" not in exc_agg.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#8b949e")
        return

    df = exc_agg.sort_values("date")
    x = range(len(df))
    labels = df["date"].tolist()

    vbs = ["30t", "60t", "120t"]
    colors = ["#58a6ff", "#a371f7", "#3fb950"]

    for i, vb in enumerate(vbs):
        col = f"median_time_to_target_{vb}_s"
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        ax.plot(x, vals, marker="o", markersize=3, linewidth=1.5,
                color=colors[i], label=f"Median TTT {vb}")

    ax.set_title("Median Time-to-Target Trend (s)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Seconds")
    ax.grid(True)
    ax.legend(fontsize=7, loc="upper right")

    if len(labels) <= 10:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    else:
        step = max(1, len(labels) // 10)
        ax.set_xticks(list(x[::step]))
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)],
                           rotation=45, ha="right", fontsize=7)


# ── Panel 6: Confusion Matrix Heatmap ──────────────────────────────────────────

def plot_confusion_matrix(ax, p8_df: pd.DataFrame, style: str):
    if p8_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#8b949e")
        return

    # Try to aggregate confusion matrices from all days
    # Look for a column that stores the confusion matrix as JSON or list
    cm_rows = []
    for _, row in p8_df.iterrows():
        for col in p8_df.columns:
            if "confusion" in col.lower() or "cm" in col.lower():
                val = row[col]
                try:
                    # Try JSON
                    cm = json.loads(val) if isinstance(val, str) else val
                    if isinstance(cm, list) and len(cm) > 0:
                        cm_rows.extend(cm)
                except Exception:
                    pass

    if not cm_rows:
        ax.text(0.5, 0.5, "No CM data", ha="center", va="center",
                transform=ax.transAxes, color="#8b949e")
        return

    try:
        cm_array = np.array(cm_rows, dtype=float)
        if cm_array.ndim != 2 or cm_array.shape[0] != cm_array.shape[1]:
            ax.text(0.5, 0.5, "Invalid CM shape", ha="center", va="center",
                    transform=ax.transAxes, color="#8b949e")
            return

        n = cm_array.shape[0]
        labels = ["Short", "Flat", "Long"][:n]

        # Normalize by row (true labels)
        row_sums = cm_array.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_norm = cm_array / row_sums

        im = ax.imshow(cm_norm, cmap=PALETTE["conf_matrix"], vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Confusion Matrix (Row-Normalized)", fontsize=10, fontweight="bold")

        # Annotate cells
        for i in range(n):
            for j in range(n):
                color = "white" if cm_norm[i, j] > 0.5 else "#c9d1d9"
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        color=color, fontsize=9)

        import json  # ensure available for annotation parsing
        return im
    except Exception as e:
        ax.text(0.5, 0.5, f"CM error: {e}", ha="center", va="center",
                transform=ax.transAxes, color="#8b949e")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is not installed")
        sys.exit(1)

    # Auto-detect local paths
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    default_agg = str(repo_root / "output" / "aggregate")

    parser = argparse.ArgumentParser(description="Plot aggregate dashboard (LOCAL)")
    parser.add_argument("--agg-dir", default=default_agg,
                        help="Directory containing aggregate CSV files")
    parser.add_argument("--out-dir", default=default_agg,
                        help="Output directory for PNG files")
    parser.add_argument("--style", default="dark", choices=["dark", "light"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    agg_dir = Path(args.agg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    style = STYLE_DARK if args.style == "dark" else STYLE_LIGHT

    print(f"=== plot_dashboard.py ===  {dt.datetime.now().isoformat()}")
    print(f"  agg-dir : {agg_dir}")
    print(f"  out-dir : {out_dir}")
    print(f"  style   : {args.style}")
    print(f"  dpi     : {args.dpi}")

    # Load data
    daily_metrics = read_csv(agg_dir / "daily_metrics.csv")
    exc_agg = read_csv(agg_dir / "excursion_aggregate.csv")
    fi_agg = read_csv(agg_dir / "feature_importance_ranked.csv")
    class_dist = read_csv(agg_dir / "class_distribution.csv")
    p8_df = read_csv(agg_dir / "multi_day_p8_results.csv")

    has_data = not daily_metrics.empty or not exc_agg.empty or not fi_agg.empty

    # Apply style
    plt.rcParams.update(style)

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Depth-DOM Pipeline Dashboard", fontsize=14, fontweight="bold",
                 y=0.98, color="#c9d1d9")

    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]

    import datetime as dt

    plot_daily_accuracy(ax1, daily_metrics, args.style)
    plot_excursion_hit_rate(ax2, exc_agg, args.style)
    plot_feature_importance(ax3, fi_agg, args.style)
    plot_class_distribution(ax4, class_dist, args.style)
    plot_time_to_target(ax5, exc_agg, args.style)
    im = plot_confusion_matrix(ax6, p8_df, args.style)

    if im is not None:
        fig.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save main dashboard
    dash_path = out_dir / "dashboard.png"
    fig.savefig(dash_path, dpi=args.dpi, bbox_inches="tight",
                facecolor=style["figure.facecolor"])
    print(f"\n  Dashboard → {dash_path}")

    # Save individual panels
    panel_names = [
        "panel_1_daily_accuracy.png",
        "panel_2_excursion_hit_rate.png",
        "panel_3_feature_importance.png",
        "panel_4_class_distribution.png",
        "panel_5_time_to_target.png",
        "panel_6_confusion_matrix.png",
    ]

    for i, (ax, name) in enumerate(zip(axes.flat, panel_names)):
        fig2, ax2p = plt.subplots(figsize=(7, 5))
        fig2.patch.set_facecolor(style["figure.facecolor"])
        ax2p.set_facecolor(style["axes.facecolor"])

        # Re-plot into single panel
        if i == 0:
            plot_daily_accuracy(ax2p, daily_metrics, args.style)
        elif i == 1:
            plot_excursion_hit_rate(ax2p, exc_agg, args.style)
        elif i == 2:
            plot_feature_importance(ax2p, fi_agg, args.style)
        elif i == 3:
            plot_class_distribution(ax2p, class_dist, args.style)
        elif i == 4:
            plot_time_to_target(ax2p, exc_agg, args.style)
        elif i == 5:
            plot_confusion_matrix(ax2p, p8_df, args.style)

        plt.tight_layout()
        panel_path = out_dir / name
        fig2.savefig(panel_path, dpi=args.dpi, bbox_inches="tight",
                     facecolor=style["figure.facecolor"])
        plt.close(fig2)
        print(f"  Panel {i+1} → {panel_path}")

    plt.close(fig)
    print(f"\n=== Done ===")


if __name__ == "__main__":
    import datetime as dt
    main()
