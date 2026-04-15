"""
Phase 8 — ML Entry Model Baseline
==================================
Build a baseline multiclass entry model (short/flat/long) using DOM features
and first-touch triple-barrier labels from Phase 7.

  CANDIDATES (from _pipeline_constants.py):
    C1: vb_ticks=2000, pt_ticks=10.0, sl_ticks=10.0  (scalping corto)
    C2: vb_ticks=4000, pt_ticks=20.0, sl_ticks=20.0  (scalping medio)
    C3: vb_ticks=8000, pt_ticks=40.0, sl_ticks=40.0  (intraday swing)
  Vertical barrier unit: TICK CLOCK (book update count, NOT seconds)

  NEW ARCHITECTURE (from NotebookLM Deep LOB research):
    Trading hours: 09:40–15:50 ET (excludes first/last 10 min of auction noise)
    Execution cadence: positions open every 5 min between 09:45–15:30 ET
    Neutral band: ±2 bps around 0 (5-min averaging window)
    Cost: 1 bp round-turn (~$25 NQ, ~$10 ES)
    PT threshold: 53-54% OOS before deploying deep nets (UCL DeepLOB)
    Features: P2b TS features (ΔL, ΔM, ΔC), stacked imbalances, volume sequencing,
              bid/ask fade, closing delta extremes, TICK Z-score internals

OUTPUTS
  phase8_dataset_summary.csv
  phase8_trainval_results.csv
  phase8_feature_importance.csv
  phase8_best_candidate.md
  phase8_best_model.pkl
  phase8_oof_predictions.csv

USAGE (LOCAL):
  python3 phase8_entry_model.py \
      --features  NQdom/output/2026-03-13/sampled_events.csv \
      --output    NQdom/output/2026-03-13/
"""

import argparse
import csv
import datetime
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore")

# ── available ML backends ────────────────────────────────────────────────────
HAS_LGBM = False
HAS_XGB  = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    pass

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    pass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, log_loss,
)
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "SHARED"))
from _pipeline_constants import CANDIDATES, ALL_FEATURE_PATTERNS, label_filename

# ── constants ─────────────────────────────────────────────────────────────────
# NEW ARCHITECTURE: Trading hours 09:40–15:50 ET (EDT = UTC-4, excludes first/last 10 min of auctions)
# UTC: 13:40–19:50 (RTH 09:30–16:00 ET minus first/last 10 min)
# In EDT (UTC-4, active March–November): 09:30 ET = 13:30 UTC, 16:00 ET = 20:00 UTC
# Execution: positions open every 5 min between 09:45–15:30 ET, exit 15:55–16:00 ET
# Split: 70/15/15 of the 13:40–19:50 UTC window (370 min → 259/56/55 min)
SPLITS = {
    "train": ("13:40:00.000000", "18:00:00.000000"),  # 09:40–14:00 ET (4h20m = 260m ≈ 70%)
    "val":   ("18:00:00.000000", "18:55:00.000000"),  # 14:00–14:55 ET (55min ≈ 15%)
    "test":  ("18:55:00.000000", "19:50:00.000000"),  # 14:55–15:50 ET (55min ≈ 15%)
}

# NEW ARCHITECTURE: Neutral stability band (DeepLOB ±2 bps on 5-min averaging window)
NEUTRAL_BAND_BPS = 0.0002   # ±2 basis points — flat when |mid_change_5m| < this

# NEW ARCHITECTURE: PT threshold for escalating to deep neural nets (UCL DeepLOB paper)
PT_THRESHOLD_DEEPLOB = 0.53  # 53% OOS — must exceed before CNN+LSTM deployment

# NEW ARCHITECTURE: Execution cadence (DeepLOB benchmark)
EXEC_INTERVAL_MIN = 5         # open positions every 5 minutes
EXEC_START_ET = "09:45"       # first entry window
EXEC_END_ET = "15:30"         # last entry window
EXIT_START_ET = "15:55"       # begin forced exit
EXIT_END_ET = "16:00"         # all positions closed by

# Label mapping: -1→0 (short), 0→1 (flat), +1→2 (long)
LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
LABEL_NAMES = {0: "short", 1: "flat", 2: "long"}

# Leakage columns to EXCLUDE from features
LEAKAGE_COLS = {
    # Labels / targets
    "label", "barrier_hit", "barrier_hit_idx",
    # Future excursions (only available after the event)
    "max_up_30s_ticks",   "max_down_30s_ticks",
    "max_up_60s_ticks",   "max_down_60s_ticks",
    "max_up_120s_ticks",  "max_down_120s_ticks",
    "mfe_ticks", "mae_ticks",
    # Excursion direction flags
    "excursion_up_30s", "excursion_down_30s",
    "excursion_up_60s", "excursion_down_60s",
    "excursion_up_120s","excursion_down_120s",
    # Barrier reference price (derived from future path)
    "ref_price",
}

PROGRESS_EVERY = 10_000


# ── timestamp helpers ──────────────────────────────────────────────────────────

def in_split(ts_str: str, lo: str, hi: str) -> bool:
    """Return True if ts_str time-of-day falls in [lo, hi)."""
    t = ts_str[11:29]
    return lo <= t < hi


def split_mask_intraday(ts_list: list[str], split_name: str) -> np.ndarray:
    lo, hi = SPLITS[split_name]
    mask = np.array([in_split(ts, lo, hi) for ts in ts_list], dtype=bool)
    return mask


def split_mask_multiday(ts_list: list[str], split_name: str, unique_dates: list[str], train_pct: float, val_pct: float) -> np.ndarray:
    """
    Splits data by full days (Walk-Forward) to prevent data leakage across sessions.
    unique_dates must be a sorted list of unique YYYY-MM-DD strings. Minimum 3 days expected.
    """
    n_days = len(unique_dates)
    n_test = max(1, int(n_days * (1.0 - train_pct - val_pct)))
    n_val  = max(1, int(n_days * val_pct))
    n_train = n_days - n_val - n_test
    
    if n_train < 1:
        # Fallback to 1-1-x if ratios leave training empty
        n_train, n_val, n_test = 1, 1, max(1, n_days - 2)

    train_dates = set(unique_dates[:n_train])
    val_dates   = set(unique_dates[n_train:n_train + n_val])
    test_dates  = set(unique_dates[n_train + n_val:])
    
    if split_name == "train":
        target = train_dates
    elif split_name == "val":
        target = val_dates
    elif split_name == "test":
        target = test_dates
    else:
        target = set()
        
    return np.array([ts[:10] in target for ts in ts_list], dtype=bool)


# ── data loading ─────────────────────────────────────────────────────────────

def load_features(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load sampled_events.csv via pandas C engine (20-50x faster than csv.DictReader)."""
    import pandas as pd
    df = pd.read_csv(path, engine="c", low_memory=False, dtype=str)
    if "ts" in df.columns:
        df["ts"] = df["ts"].str.replace(" UTC", "", regex=False)
    return df, list(df.columns)


def load_labels(path: Path) -> dict[str, dict]:
    """Load phase7_labels_*.csv (or a dir containing one). Returns {ts: row_dict}."""
    labels = {}
    # If path is a directory, find the CSV inside
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            print(f"P8 ERROR: No label CSV found in {path}. "
                  f"Expected pattern: phase7_labels_{{N}}ticks_*. "
                  f"Did you run P7 with the new tick-clock naming?")
            return labels  # return empty — caller checks n_match
        path = csv_files[0]  # take first CSV found
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row["ts"].replace(" UTC", "")
            labels[ts] = row
    return labels


# ── feature engineering ───────────────────────────────────────────────────────

def safe_float(val: str | None, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Vectorized feature extraction from pre-loaded DataFrame. 10-30x faster than row iteration."""
    cols_present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} feature cols missing from DataFrame: {missing[:5]}")
    out = df[cols_present].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
    if len(missing) > 0:
        # pad with zeros for missing columns, maintain column order
        result = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
        for i, col in enumerate(feature_cols):
            if col in cols_present:
                j = cols_present.index(col)
                result[:, i] = out[:, j]
        return result
    return out


def validate_features(all_cols: list[str], df_sample) -> tuple[list[str], list[str]]:
    """
    Return (feature_cols, excluded_cols).
    - Explicit leakage columns
    - Timestamp / ID columns
    - Columns not in file
    - Columns with ≤1 unique value

    Supports both pd.DataFrame (fast vectorized path) and list[dict] (legacy).
    """
    excluded = set(LEAKAGE_COLS)
    for col in ["ts", "index", "idx", "event_id"]:
        excluded.add(col)

    # Candidate: any col that matches our known patterns or is numeric
    candidate_cols = [c for c in all_cols
                      if c not in excluded
                      and c in ALL_FEATURE_PATTERNS]

    # Also include any numeric col in ALL_FEATURE_PATTERNS that's in the file
    # (catch any extra agg features not explicitly listed)
    for col in all_cols:
        if col in excluded:
            continue
        if col in ALL_FEATURE_PATTERNS and col not in candidate_cols:
            candidate_cols.append(col)

    # Remove constant cols — DataFrame path uses vectorized nunique()
    constant = set()
    if hasattr(df_sample, 'iloc'):  # pandas DataFrame path
        for col in candidate_cols:
            if col not in df_sample.columns:
                continue
            nunique = df_sample[col].apply(pd.to_numeric, errors='coerce').nunique()
            if nunique <= 1:
                constant.add(col)
    else:  # legacy list[dict] path
        for col in candidate_cols:
            vals = set()
            for row in df_sample:
                v = safe_float(row.get(col), np.nan)
                if not np.isnan(v):
                    vals.add(v)
            if len(vals) <= 1:
                constant.add(col)

    feature_cols = [c for c in candidate_cols if c not in constant]
    excluded_list = list(excluded | constant)
    return feature_cols, excluded_list


# ── model definitions ─────────────────────────────────────────────────────────

def get_models():
    models = [
        ("LogisticRegression", LogisticRegression(
            multi_class="multinomial", max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42
        )),
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=50,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
        ("HistGradientBoosting", HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            early_stopping=True, validation_fraction=0.15,
            random_state=42
        )),
    ]
    if HAS_LGBM:
        models.append(("LightGBM", lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=50,
            class_weight="balanced", random_state=42,
            verbose=-1, n_jobs=-1
        )))
    if HAS_XGB:
        models.append(("XGBoost", xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            min_child_weight=50, eval_metric="mlogloss",
            use_label_encoder=False, random_state=42,
            n_jobs=-1, verbosity=0
        )))
    return models


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_prob: np.ndarray | None = None,
             classes: list[int] | None = None) -> dict:
    """Compute metrics dict for multiclass classification."""
    if classes is None:
        classes = sorted(set(y_true))

    metrics = {
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1":          round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1":        round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }

    # ── Custom pT Score (0=short, 2=long) ──
    for c, cname in [(0, "short"), (2, "long")]:
        PT = np.sum(y_true == c)
        TT = np.sum(y_pred == c)
        CT = np.sum((y_true == c) & (y_pred == c))
        
        # Penalizza pesantemente se il modello non esegue mai o non interseca niente (CT = 0)
        pT = (PT + TT - CT) / CT if CT > 0 else 9999.0
        
        metrics[f"PT_{cname}"] = int(PT)
        metrics[f"TT_{cname}"] = int(TT)
        metrics[f"CT_{cname}"] = int(CT)
        metrics[f"pT_{cname}"] = round(float(pT), 4)

    metrics["pT_avg"] = round((metrics["pT_short"] + metrics["pT_long"]) / 2.0, 4)

    if y_prob is not None and len(classes) <= 3:
        try:
            metrics["log_loss"] = round(log_loss(y_true, y_prob, labels=classes), 4)
        except Exception:
            metrics["log_loss"] = None

    report = classification_report(y_true, y_pred, labels=classes,
                                   output_dict=True, zero_division=0)
    for cls in classes:
        cls_key = str(cls)
        if cls_key in report:
            metrics[f"prec_{cls}"] = round(report[cls_key]["precision"], 4)
            metrics[f"rec_{cls}"]   = round(report[cls_key]["recall"], 4)
            metrics[f"f1_{cls}"]    = round(report[cls_key]["f1-score"], 4)
            metrics[f"n_{cls}"]     = int(report[cls_key]["support"])
            metrics[f"pct_{cls}"]    = round(report[cls_key]["support"] / len(y_true) * 100, 2)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def print_metrics(m: dict, title: str = "") -> None:
    if title:
        print(f"\n  {title}")
    print(f"    pT Avg Score:      {m.get('pT_avg', 'N/A')}  (lower is better, 1.0 is perfect)")
    print(f"    SHORT pT: {m.get('pT_short', 'N/A')} [PT:{m.get('PT_short', 0)} TT:{m.get('TT_short', 0)} CT:{m.get('CT_short', 0)}]")
    print(f"    LONG  pT: {m.get('pT_long', 'N/A')} [PT:{m.get('PT_long', 0)} TT:{m.get('TT_long', 0)} CT:{m.get('CT_long', 0)}]")
    print(f"    Accuracy:          {m.get('accuracy', 'N/A')}")
    print(f"    Balanced Accuracy: {m.get('balanced_accuracy', 'N/A')}")
    print(f"    Macro F1:          {m.get('macro_f1', 'N/A')}")
    ll = m.get('log_loss')
    if ll is not None:
        print(f"    Log Loss:          {ll}")
    print(f"    Class dist:        "
          f"short={m.get('n_0',0):5d}({m.get('pct_0',0):5.1f}%) "
          f"flat={m.get('n_1',0):5d}({m.get('pct_1',0):5.1f}%) "
          f"long={m.get('n_2',0):5d}({m.get('pct_2',0):5.1f}%)")
    if m.get('confusion_matrix') is not None:
        cm = np.array(m["confusion_matrix"])
        print(f"    Confusion matrix:\n{cm}")


# ── feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, name: str, feature_cols: list[str]) -> list[dict]:
    """Extract feature importance ranking from model."""
    imp = None
    if name in ("RandomForest", "LightGBM", "XGBoost", "HistGradientBoosting"):
        imp = model.feature_importances_
    elif name == "LogisticRegression":
        imp = np.mean(np.abs(model.coef_), axis=0)

    if imp is None:
        return []
    pairs = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    return [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs]


# ── incremental training ──────────────────────────────────────────────────────

def train_with_batch_retrain(X_new, y_new, model_path=None):
    """
    Retrain model on new batch only. NOTE: This is NOT true incremental/warm-start
    learning — previous training data is NOT retained. After each weekly update,
    only the most recent batch is used for training.

    For true incremental learning, use:
      - SGDClassifier with partial_fit() for online learning
      - Or full retrain on all historical data (recommended for tree models)

    Args:
        X_new: New feature matrix
        y_new: New labels
        model_path: Path to existing model.pkl (for reference only — not used for training)

    Returns:
        (None, None) — caller should use normal train path for full retrain
    """
    print(f"  WARNING: Batch retrain mode — previous training data is NOT retained.")
    print(f"  For tree models (RF/HGB/LGBM/XGB), use FULL retrain on all historical data.")
    print(f"  This function is a placeholder for future true incremental learning.")
    return None, None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 8: ML Entry Model Baseline")
    ap.add_argument("--features",  type=Path, required=True,
                    help="Path to sampled_events.csv")
    ap.add_argument("--output",    type=Path, required=True,
                    help="Output directory")
    ap.add_argument("--force",     action="store_true",
                    help="Overwrite existing outputs")
    ap.add_argument("--warm-start", type=str, default=None,
                    help="Path to existing model .pkl for batch retrain (WARNING: previous data not retained)")
    ap.add_argument("--train-pct", type=float, default=0.70,
                    help="Percentage of days to use for training in Walk-Forward split")
    ap.add_argument("--val-pct", type=float, default=0.15,
                    help="Percentage of days to use for validation in Walk-Forward split")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("PHASE 8 — ML Entry Model Baseline")
    print(f"{'='*70}")
    print(f"  Features : {args.features}")
    print(f"  Output   : {args.output}")
    print(f"  LightGBM : {HAS_LGBM}  |  XGBoost : {HAS_XGB}")

    # ── 1. Load features ────────────────────────────────────────────────────
    print(f"\n[1] Loading features …")
    t0 = time.time()
    features_rows, all_cols = load_features(args.features)
    print(f"    {len(features_rows):,} rows, {len(all_cols)} columns loaded "
          f"in {time.time()-t0:.1f}s")

    # ── 2. Validate features ───────────────────────────────────────────────
    print(f"\n[2] Validating features …")
    # PERF: use larger sample for constant detection — avoids false exclusions
    sample_size = min(50_000, len(features_rows))
    sample = features_rows.iloc[:sample_size]
    feature_cols, excluded_cols = validate_features(all_cols, sample)
    print(f"    {len(feature_cols)} feature columns selected")
    if excluded_cols:
        print(f"    Excluded ({len(excluded_cols)}): {excluded_cols[:15]}")

    # ── 3. Define candidates and load labels ──────────────────────────────
    print(f"\n[3] Loading Phase 7 labels …")
    label_candidates = {}
    search_dirs = [args.output]

    for cand in CANDIDATES:
        fname = label_filename(cand["vb_ticks"], cand["pt_ticks"], cand["sl_ticks"])
        found = False
        for d in search_dirs:
            fpath = d / fname
            if fpath.exists():
                labels = load_labels(fpath)
                # PERF: vectorized label match count using pandas isin
                n_match = features_rows["ts"].isin(labels.keys()).sum()
                label_candidates[cand["desc"]] = labels
                print(f"    {fname}: {n_match:,}/{len(features_rows):,} events matched")
                found = True
                break
        if not found:
            print(f"    WARNING: {fname} NOT FOUND")

    if not label_candidates:
        print("\n    ERROR: No label files found. Check paths.")
        # List available label files
        for d in search_dirs:
            if d.exists():
                print(f"    Files in {d}:")
                for f in sorted(d.glob("phase7_labels*.csv")):
                    print(f"      {f.name}")
        sys.exit(1)

    # ── 4. Process each candidate ──────────────────────────────────────────
    all_results   = []
    best_overall  = None
    best_overall_key = ""

    for cand in CANDIDATES:
        cname = cand["desc"]
        print(f"\n{'='*70}")
        print(f"CANDIDATE: vb={cand['vb_ticks']}ticks  pt={cand['pt_ticks']}  sl={cand['sl_ticks']}")
        print(f"{'='*70}")

        if cname not in label_candidates:
            print(f"    SKIP — no label file")
            continue

        labels = label_candidates[cname]

        # ── 4a. Align ─────────────────────────────────────────────────────
        print(f"\n[4a] Aligning features + labels …")
        # PERF: vectorized align using pandas merge — eliminates O(N) Python loop
        labels_df = pd.DataFrame.from_dict(labels, orient="index").reset_index()
        # index col = ts key; existing 'ts' col = ts from dict values (dup); rename to avoid
        labels_df.rename(columns={"index": "_ts_key", "label": "label_raw"}, inplace=True)
        labels_df.drop(columns=["ts"], inplace=True)  # drop dup ts from dict values
        labels_df.rename(columns={"_ts_key": "ts"}, inplace=True)
        labels_df["label_mapped"] = labels_df["label_raw"].map(LABEL_MAP)
        labels_df = labels_df.dropna(subset=["label_mapped"])

        merged = features_rows.merge(labels_df[["ts", "label_mapped"]], on="ts", how="inner")
        aligned_ts = merged["ts"].tolist()
        aligned_y = merged["label_mapped"].astype(np.int8).values
        aligned_X = merged  # DataFrame — used by build_feature_matrix directly

        n = len(merged)
        missing = len(features_rows) - n
        print(f"    Aligned: {n:,}  (no label: {missing:,})")
        if n == 0:
            continue

        y = aligned_y

        # ── 4b. Build feature matrix ──────────────────────────────────────
        print(f"\n[4b] Building X ({len(feature_cols)} features) …")
        X = build_feature_matrix(aligned_X, feature_cols)  # aligned_X is already a DataFrame
        del aligned_X; gc.collect()
        print(f"    X: {X.shape}  y: short={np.sum(y==0)} flat={np.sum(y==1)} long={np.sum(y==2)}")

        # ── 4c. Temporal split ────────────────────────────────────────────
        print(f"\n[4c] Temporal split …")
        unique_dates = sorted(list(set(ts[:10] for ts in aligned_ts)))
        n_days = len(unique_dates)
        
        if n_days < 2:
            print(f"    [AUTO-DETECT] {n_days} giorno(i) rilevato(i) (< 2). Fallback su split Intra-Day orario (SPLITS).")
            train_mask = split_mask_intraday(aligned_ts, "train")
            val_mask   = split_mask_intraday(aligned_ts, "val")
            test_mask  = split_mask_intraday(aligned_ts, "test")
        else:
            print(f"    [AUTO-DETECT] {n_days} giorni rilevati. Uso Walk-Forward Multi-Day split (no intra-day per evitare bias orario).")
            train_pct, val_pct = args.train_pct, args.val_pct
            train_mask = split_mask_multiday(aligned_ts, "train", unique_dates, train_pct, val_pct)
            val_mask   = split_mask_multiday(aligned_ts, "val", unique_dates, train_pct, val_pct)
            test_mask  = split_mask_multiday(aligned_ts, "test", unique_dates, train_pct, val_pct)
            train_pct, val_pct = args.train_pct, args.val_pct
            train_mask = split_mask_multiday(aligned_ts, "train", unique_dates, train_pct, val_pct)
            val_mask   = split_mask_multiday(aligned_ts, "val", unique_dates, train_pct, val_pct)
            test_mask  = split_mask_multiday(aligned_ts, "test", unique_dates, train_pct, val_pct)
        
        n_train, n_val, n_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
        print(f"    train={n_train:,}  val={n_val:,}  test={n_test:,}")

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        # ── 4d. Train all models ──────────────────────────────────────────
        print(f"\n[4d] Training models …")
        models = get_models()

        # Handle warm-start: load existing model for batch retrain
        if args.warm_start:
            try:
                warm_model, warm_scaler = train_with_batch_retrain(
                    X_train, y_train, model_path=args.warm_start
                )
                if warm_model is not None:
                    # Insert warmed model as first candidate
                    models = [(f"WarmStart({Path(args.warm_start).stem})", warm_model)] + models
                    print(f"  Warm-start model loaded and will be evaluated")
            except Exception as e:
                print(f"  WARNING: Could not load warm-start model: {e}")

        best_score  = float('inf')
        best_model  = None
        best_name   = ""
        best_scaler = None
        cand_results = []

        for mname, model in models:
            print(f"\n  --- {mname} ---")
            t_m = time.time()

            X_tr = X_train_s if mname == "LogisticRegression" else X_train
            X_vl = X_val_s   if mname == "LogisticRegression" else X_val

            try:
                model.fit(X_tr, y_train)
            except Exception as e:
                print(f"    Train failed: {e}")
                continue

            y_pred_val = model.predict(X_vl)
            y_prob_val = getattr(model, "predict_proba", lambda x: None)(X_vl)
            metrics_val = evaluate(y_val, y_pred_val, y_prob_val, classes=[0,1,2])
            elapsed = time.time() - t_m
            print_metrics(metrics_val, f"{mname} val ({elapsed:.1f}s)")

            # Test evaluation
            X_ts = X_test_s if mname == "LogisticRegression" else X_test
            y_pred_test = model.predict(X_ts)
            y_prob_test = getattr(model, "predict_proba", lambda x: None)(X_ts)
            metrics_test = evaluate(y_test, y_pred_test, y_prob_test, classes=[0,1,2])
            print_metrics(metrics_test, f"{mname} test")

            row = {
                "candidate": cname,
                "model": mname,
                "vb": cand["vb_ticks"], "pt": cand["pt_ticks"], "sl": cand["sl_ticks"],
                "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test),
                **{k: v for k, v in metrics_val.items() if k != "confusion_matrix"},
                **{f"test_{k}": v for k, v in metrics_test.items() if k != "confusion_matrix"},
            }
            cand_results.append(row)
            all_results.append(row)

            # Track best by Minimizing pT_avg on val
            sc = metrics_val.get("pT_avg", float('inf'))
            if sc < best_score:
                best_score = sc
                best_model = model
                best_name  = mname
                best_scaler = scaler if mname == "LogisticRegression" else None
                best_metrics_val   = metrics_val
                best_metrics_test  = metrics_test
                best_X_test = X_ts
                best_y_test = y_test
                best_ts_test = np.array(aligned_ts)[test_mask]

            del model; gc.collect()

        # ── 4e. Feature importance ────────────────────────────────────────
        if best_model is not None:
            print(f"\n[4e] Feature importance ({best_name}) …")
            fi = get_feature_importance(best_model, best_name, feature_cols)
            fi_cname = cname.replace("/","_").replace(" ","_")
            fi_path = args.output / f"phase8_fi_{fi_cname}.csv"
            with open(fi_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["rank","feature","importance"])
                w.writeheader()
                for rank, item in enumerate(fi[:50], 1):
                    w.writerow({"rank": rank, **item})
            print(f"    Top 10: {[f['feature'] for f in fi[:10]]}")
            print(f"    → {fi_path}")

            # Check for overall best (minimizing pT)
            if best_score < (best_overall[0] if best_overall else float('inf')):
                best_overall = (best_score, cname, best_name, cand, best_metrics_val,
                                best_metrics_test, fi, feature_cols,
                                best_scaler, best_X_test, best_y_test, best_ts_test, best_model)

        del X, X_train, X_val, X_test, X_train_s, X_val_s, X_test_s
        gc.collect()

    # ── 5. Write all output files ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("[5] Writing output files …")

    # 5a. Dataset summary
    summary_rows = []
    for cand in CANDIDATES:
        cname = cand["desc"]
        if cname not in label_candidates:
            continue
        labels = label_candidates[cname]
        # PERF: vectorized summary using pandas itertuples
        matched_mask = features_rows["ts"].isin(labels.keys())
        n = matched_mask.sum()
        if n == 0:
            continue
        ys = []
        tss = []
        for row in features_rows.itertuples():
            ts = row.ts
            if ts in labels:
                raw_lbl = int(labels[ts]["label"])
                mapped = LABEL_MAP.get(raw_lbl)
                if mapped is not None:
                    ys.append(mapped)
                    tss.append(ts)
        n = len(ys)
        n0, n1, n2 = sum(1 for y in ys if y==0), sum(1 for y in ys if y==1), sum(1 for y in ys if y==2)
        nonzero = n0 + n2
        br = round(min(n0,n2)/max(n0,n2), 4) if max(n0,n2) else 0
        tss = sorted(tss)
        summary_rows.append({
            "candidate":         cname,
            "vb_ticks":         cand["vb_ticks"],
            "pt_ticks":         cand["pt_ticks"],
            "sl_ticks":         cand["sl_ticks"],
            "n_events":         n,
            "n_features":       len(feature_cols),
            "n_short":          n0,
            "n_flat":           n1,
            "n_long":           n2,
            "pct_short":        round(n0/n*100, 2) if n else 0,
            "pct_flat":         round(n1/n*100, 2) if n else 0,
            "pct_long":         round(n2/n*100, 2) if n else 0,
            "balance_ratio":    br,
            "pct_vertical_exp": round(n1/n*100, 2) if n else 0,
            "time_range":       f"{tss[0][11:19] if tss else 'N/A'} → "
                                f"{tss[-1][11:19] if tss else 'N/A'}",
        })

    sum_path = args.output / "phase8_dataset_summary.csv"
    if summary_rows:
        with open(sum_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader(); w.writerows(summary_rows)
    print(f"    Dataset summary → {sum_path}")

    # 5b. Train/val results
    if all_results:
        res_path = args.output / "phase8_trainval_results.csv"
        with open(res_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys(), extrasaction="ignore")
            w.writeheader(); w.writerows(all_results)
        print(f"    Train/val results → {res_path}")

    # 5c. Feature importance (overall best)
    if best_overall:
        _, _, _, _, best_mv, best_mt, best_fi, _, _, _, _, _, _ = best_overall
        fi_path = args.output / "phase8_feature_importance.csv"
        with open(fi_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["rank","feature","importance"])
            w.writeheader()
            for rank, item in enumerate(best_fi[:50], 1):
                w.writerow({"rank": rank, **item})
        print(f"    Feature importance → {fi_path}")

    # 5d. Best model serialization
    model_path = args.output / "phase8_best_model.pkl"
    if best_overall:
        _, best_cname, best_mname, best_cand, best_mv, best_mt, best_fi, \
            feat_cols, best_scaler, X_test, y_test, ts_test, absolute_best_model = best_overall
        with open(model_path, "wb") as f:
            pickle.dump({
            "candidate":     best_cand,
            "model_name":   best_mname,
            "feature_cols": feat_cols,
            "scaler":       best_scaler,
            "label_map":    LABEL_MAP,
            "label_names":  LABEL_NAMES,
        }, f)
        print(f"    Best model metadata → {model_path}")

    # 5e. OOF predictions
    oof_path = args.output / "phase8_oof_predictions.csv"
    if best_overall:
        _, best_cname, best_mname, best_cand, best_mv, best_mt, best_fi, \
            feat_cols, best_scaler, X_test, y_test, ts_test, absolute_best_model = best_overall
        try:
            # Re-predict with the best model to get probabilities
            # (X_test was already computed — reuse it)
            y_prob_test = getattr(absolute_best_model, "predict_proba", lambda x: None)(X_test) \
                          if absolute_best_model is not None else None
            pred_class_arr = np.argmax(y_prob_test, axis=1) if y_prob_test is not None else None
            with open(oof_path, "w", newline="", encoding="utf-8") as f:
                fields = ["ts","y_true","pred_class","prob_short","prob_flat","prob_long"]
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for i in range(len(ts_test)):
                    p = y_prob_test[i] if y_prob_test is not None else (None, None, None)
                    pc = int(pred_class_arr[i]) if pred_class_arr is not None else ""
                    row_o = {
                        "ts":         ts_test[i],
                        "y_true":     int(y_test[i]) if i < len(y_test) else "",
                        "pred_class": pc,
                        "prob_short": round(p[0], 6) if p[0] is not None else "",
                        "prob_flat":  round(p[1], 6) if p[1] is not None else "",
                        "prob_long":  round(p[2], 6) if p[2] is not None else "",
                    }
                    w.writerow(row_o)
            print(f"    OOF predictions → {oof_path}")
        except Exception as e:
            print(f"    OOF predictions note: {e}")

    # ── 6. Best candidate report ──────────────────────────────────────────
    report_path = args.output / "phase8_best_candidate.md"
    if best_overall:
        bs, best_cname, best_mname, bc, mv, mt, fi, feat_cols, scaler, X_t, y_t, ts_t, absolute_best_model = best_overall

        # Category breakdown
        cats = {"imbalance": [], "stack_pull": [], "priority_score": [],
                "depth": [], "other": []}
        for item in fi[:50]:
            f = item["feature"].lower()
            if "imb" in f:
                cats["imbalance"].append(item)
            elif "stack_" in f or "pull_" in f:
                cats["stack_pull"].append(item)
            elif "ps_" in f or "weight" in f or "delta" in f or "microprice" in f or "spread" in f:
                cats["priority_score"].append(item)
            elif "depth" in f or "qty" in f or "ratio" in f:
                cats["depth"].append(item)
            else:
                cats["other"].append(item)

        report = f"""# Phase 8 — ML Entry Model Baseline Report

## Best Candidate

| Parameter | Value |
|-----------|-------|
| Vertical Barrier | {bc['vb_ticks']} ticks |
| Profit Target | {bc['pt_ticks']} ticks |
| Stop Loss | {bc['sl_ticks']} ticks |
| Candidate ID | {best_cname} |

## Best Model

**Model:** {best_mname}

## Selection Rationale

Selected by **Minimizing Average $p_T$** (primary) on the validation set. 
This custom metric optimally aligns trading executions (Predicted Transactions = TT) with ideal market opportunities (Potential Transactions = PT) by maximizing their chronological Intersection (Correct Transactions = CT).

## Validation Metrics (Microstructural $p_T$)

| Metric | Score | Detail |
|--------|-------|--------|
| **pT Avg** | `{mv.get('pT_avg', 'N/A')}` | |
| **pT Short** | `{mv.get('pT_short', 'N/A')}` | PT: {mv.get('PT_short')} / TT: {mv.get('TT_short')} / CT: {mv.get('CT_short')} |
| **pT Long** | `{mv.get('pT_long', 'N/A')}` | PT: {mv.get('PT_long')} / TT: {mv.get('TT_long')} / CT: {mv.get('CT_long')} |
| Accuracy | {mv.get('accuracy', 'N/A')} | |

### Per-Class (Validation)

| Class | Precision | Recall | F1 | Count | Pct |
|-------|-----------|--------|----|----|-----|
| Short (0) | {mv.get('prec_0','N/A')} | {mv.get('rec_0','N/A')} | {mv.get('f1_0','N/A')} | {mv.get('n_0','N/A')} | {mv.get('pct_0','N/A')}% |
| Flat (1) | {mv.get('prec_1','N/A')} | {mv.get('rec_1','N/A')} | {mv.get('f1_1','N/A')} | {mv.get('n_1','N/A')} | {mv.get('pct_1','N/A')}% |
| Long (2) | {mv.get('prec_2','N/A')} | {mv.get('rec_2','N/A')} | {mv.get('f1_2','N/A')} | {mv.get('n_2','N/A')} | {mv.get('pct_2','N/A')}% |

## Test Metrics
| Metric | Score | Detail |
|--------|-------|--------|
| **pT Avg** | `{mt.get('pT_avg', 'N/A')}` | |
| **pT Short** | `{mt.get('pT_short', 'N/A')}` | PT: {mt.get('PT_short')} / TT: {mt.get('TT_short')} / CT: {mt.get('CT_short')} |
| **pT Long** | `{mt.get('pT_long', 'N/A')}` | PT: {mt.get('PT_long')} / TT: {mt.get('TT_long')} / CT: {mt.get('CT_long')} |

## Feature Importance — Top 30

| Rank | Feature | Importance |
|------|---------|------------|
"""
        for rank, item in enumerate(fi[:30], 1):
            report += f"| {rank} | {item['feature']} | {item['importance']:.6f} |\n"

        report += f"""
## Feature Category Dominance (Top 50)

| Category | Count | Top Features |
|----------|-------|-------------|
| Imbalance | {len(cats['imbalance'])} | {', '.join(x['feature'] for x in cats['imbalance'][:5]) or '—'} |
| Stack/Pull | {len(cats['stack_pull'])} | {', '.join(x['feature'] for x in cats['stack_pull'][:5]) or '—'} |
| Priority/Microprice | {len(cats['priority_score'])} | {', '.join(x['feature'] for x in cats['priority_score'][:5]) or '—'} |
| Depth/Qty | {len(cats['depth'])} | {', '.join(x['feature'] for x in cats['depth'][:5]) or '—'} |
| Other | {len(cats['other'])} | {', '.join(x['feature'] for x in cats['other'][:5]) or '—'} |

## Overfitting Risk Assessment (pT framework)

1. **Trade execution failure:** Check if CT approaches zero, indicating the model places trades incorrectly.
2. **Class distribution shift:** Re-verify features if flat class defaults bypass model logic.
3. **Execution Edge:** High TT with Low CT strongly signals premature close logic or weak entry. Focus on scaling penalty for $p_T$.


## Excluded Leakage Columns

{', '.join(excluded_cols[:25])}
{'...' if len(excluded_cols) > 25 else ''}

## Next Steps

- Phase 9: Dynamic trade management (exit/EV adjustment) — NOT yet implemented
- Walk-forward validation across multiple days before production
- Time & Sales feature integration (future phase)
- Probability calibration if threshold-based entry is needed
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"    Report → {report_path}")

    print(f"\n{'='*70}")
    print("PHASE 8 COMPLETE")
    print(f"{'='*70}")
    if best_overall:
        print(f"\nBest: {best_overall[1]}  |  {best_overall[2]}  |  "
              f"bal_acc_val={best_overall[0]:.4f}")
    print(f"\nAll files → {args.output}")


if __name__ == "__main__":
    main()
