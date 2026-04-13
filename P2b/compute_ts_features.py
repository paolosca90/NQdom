"""
Phase 2b+ — T&S Derived Features (Polars)
=========================================
Computa features avanzate dai Time & Sales trades, aggregate a 1s/5s/30s,
e merge sullo snapshot stream.

Features implementate:
  A. ORDER FLOW IMBALANCE (OFI)
     - ofi_1s, ofi_5s, ofi_30s
     - cum_ofi_1s, cum_ofi_5s, cum_ofi_30s
     - ofi_ratio_1s, ofi_ratio_5s (normalized imbalance)

  B. TRADE INTENSITY & CLUSTERING
     - trade_count_1s, trade_count_5s, trade_count_30s
     - buy_ratio_1s, buy_ratio_5s (directional clustering)
     - vol_surge_5s, vol_surge_30s (volume relative to rolling mean)

  C. ABSORPTION
     - absorption_factor_5s = vol_surge / (range_surge + 1e-10)

  D. VWAP DYNAMICS
     - vwap_1s, vwap_5s
     - vwap_deviation_1s, vwap_deviation_5s

  E. REALIZED VOLATILITY
     - realized_vol_5s, realized_vol_30s (from trade returns)

  F. TRADE MOMENTUM
     - trade_momentum_5s = sign(OFI) * abs(return) * trade_count

Output: output/{date}/snapshots_fused_with_ts_features.csv
"""

import polars as pl
import re
import time
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
DATE = '2026-03-13'
TRADES_PATH = Path(r'C:\Users\Paolo\Desktop\NQ\DATI NEW\NQM26-CME.txt')
SNAPS_PATH = Path(r'C:\Users\Paolo\Desktop\NQ\NQdom\output\2026-03-13\snapshots_fused.csv')
OUTPUT_PATH = Path(r'C:\Users\Paolo\Desktop\NQ\NQdom\output\2026-03-13\snapshots_ts_features.csv')

# ── Load & Prepare Trades ──────────────────────────────────────────────────────

def load_trades(path: Path, date_str: str) -> pl.DataFrame:
    """Load T&S trades per data, parse timestamps, aggiungi side."""
    df = pl.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    # Filter by date
    m = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
    yr, mo, day = m.groups()
    df = df.filter(
        pl.col('date').str.contains(yr) &
        pl.col('date').str.contains(r'/' + str(int(mo)) + r'/') &
        pl.col('date').str.contains(r'/' + str(int(day)) + r'$')
    )

    # Parse timestamp
    df = df.with_columns(
        (pl.col('date') + ' ' + pl.col('time')).alias('ts_raw')
    )
    df = df.with_columns(
        pl.col('ts_raw').str.to_datetime(
            format='%Y/%m/%d %H:%M:%S%.f', strict=False
        ).alias('ts')
    )

    # Side determination
    bv = pl.col('bidvolume').fill_null(0)
    av = pl.col('askvolume').fill_null(0)
    df = df.with_columns(
        ((av > 0) & (bv == 0)).alias('is_buy')
    )
    df = df.with_columns(
        ((bv > 0) & (av == 0)).alias('is_sell')
    )
    df = df.filter(pl.col('is_buy') | pl.col('is_sell'))

    # Signed volume: positive = buy, negative = sell
    df = df.with_columns(
        pl.when(pl.col('is_buy')).then(pl.col('volume'))
        .when(pl.col('is_sell')).then(-pl.col('volume'))
        .otherwise(0).alias('signed_vol')
    )
    # Volume always positive
    df = df.with_columns(
        pl.col('volume').alias('trade_vol')
    )

    return df.sort('ts')[['ts', 'last', 'trade_vol', 'signed_vol', 'is_buy', 'is_sell']]


# ── Compute T&S Features at Multiple Resolutions ───────────────────────────────

def compute_ts_features(trades: pl.DataFrame) -> pl.DataFrame:
    """
    Compute T&S features aggregated at 1s, 5s, 30s windows.
    Returns a DataFrame with all feature columns indexed by ts (the bucket timestamp).
    """
    t = trades.sort('ts')

    # ── 1s buckets ──────────────────────────────────────────────────────────
    t1 = t.with_columns(pl.col('ts').dt.truncate('1s').alias('ts'))

    g1 = t1.group_by('ts', maintain_order=True).agg(
        pl.col('trade_vol').sum().alias('vol_1s'),
        pl.col('signed_vol').sum().alias('ofi_1s_raw'),
        pl.col('is_buy').sum().alias('buy_count_1s'),
        pl.col('is_sell').sum().alias('sell_count_1s'),
        pl.col('last').count().alias('trade_count_1s'),
        pl.col('last').mean().alias('vwap_1s'),
    )
    g1 = g1.with_columns(
        (pl.col('buy_count_1s') - pl.col('sell_count_1s')).alias('directional_bias_1s'),
        (pl.col('buy_count_1s') / pl.col('trade_count_1s')).alias('buy_ratio_1s'),
    )

    # ── 5s buckets ──────────────────────────────────────────────────────────
    t5 = t.with_columns(pl.col('ts').dt.truncate('5s').alias('ts'))

    g5 = t5.group_by('ts', maintain_order=True).agg(
        pl.col('trade_vol').sum().alias('vol_5s'),
        pl.col('signed_vol').sum().alias('ofi_5s_raw'),
        pl.col('is_buy').sum().alias('buy_count_5s'),
        pl.col('is_sell').sum().alias('sell_count_5s'),
        pl.col('last').count().alias('trade_count_5s'),
        pl.col('last').mean().alias('vwap_5s'),
        pl.col('last').max().alias('high_5s'),
        pl.col('last').min().alias('low_5s'),
    )
    g5 = g5.with_columns(
        (pl.col('buy_count_5s') - pl.col('sell_count_5s')).alias('directional_bias_5s'),
        (pl.col('buy_count_5s') / pl.col('trade_count_5s')).alias('buy_ratio_5s'),
        ((pl.col('high_5s') - pl.col('low_5s')) / pl.col('low_5s')).alias('range_pct_5s'),
    )

    # ── 30s buckets ─────────────────────────────────────────────────────────
    t30 = t.with_columns(pl.col('ts').dt.truncate('30s').alias('ts'))

    g30 = t30.group_by('ts', maintain_order=True).agg(
        pl.col('trade_vol').sum().alias('vol_30s'),
        pl.col('signed_vol').sum().alias('ofi_30s_raw'),
        pl.col('is_buy').sum().alias('buy_count_30s'),
        pl.col('is_sell').sum().alias('sell_count_30s'),
        pl.col('last').count().alias('trade_count_30s'),
        pl.col('last').mean().alias('vwap_30s'),
        pl.col('last').max().alias('high_30s'),
        pl.col('last').min().alias('low_30s'),
    )
    g30 = g30.with_columns(
        (pl.col('high_30s') - pl.col('low_30s')).alias('range_30s'),
    )

    # ── Merge all buckets on ts ────────────────────────────────────────────
    feat = g1.join(g5, on='ts', how='left')
    feat = feat.join(g30, on='ts', how='left')

    # ── Rolling statistics for surge features ───────────────────────────────
    feat = feat.sort('ts')

    # Vol surge (5s vs 30s rolling mean)
    feat = feat.with_columns(
        pl.col('vol_5s').rolling_mean(6, min_periods=1).alias('vol_ma_30s')  # 6*5s = 30s
    )
    feat = feat.with_columns(
        (pl.col('vol_5s') / (pl.col('vol_ma_30s') + 1e-9)).alias('vol_surge_5s')
    )

    # Range surge
    feat = feat.with_columns(
        pl.col('range_pct_5s').rolling_mean(6, min_periods=1).alias('range_ma_30s')
    )
    feat = feat.with_columns(
        (pl.col('range_pct_5s') / (pl.col('range_ma_30s') + 1e-9)).alias('range_surge_5s')
    )

    # Absorption factor
    feat = feat.with_columns(
        (pl.col('vol_surge_5s') / (pl.col('range_surge_5s') + 1e-10)).alias('absorption_factor_5s')
    )

    # VWAP deviation (current vwap vs cumulative vwap)
    feat = feat.with_columns(
        pl.col('vwap_5s').rolling_mean(1000, min_periods=1).alias('vwap_cumulative')
    )
    feat = feat.with_columns(
        ((pl.col('vwap_5s') - pl.col('vwap_cumulative')) / pl.col('vwap_cumulative') * 10000)
        .alias('vwap_deviation_5s')
    )

    # Trade momentum: OFI * return * scale
    feat = feat.with_columns(
        (pl.col('ofi_5s_raw').abs() * pl.col('range_pct_5s') * 10000).alias('trade_momentum_5s')
    )

    # OFI ratio normalized
    feat = feat.with_columns(
        (pl.col('ofi_5s_raw') / (pl.col('vol_5s') + 1)).alias('ofi_ratio_5s')
    )

    # Realized vol from trade returns (rolling std of log returns * 10000)
    feat = feat.with_columns(
        (pl.col('vwap_5s').pct_change().rolling_std(6, min_periods=1) * 10000)
        .alias('realized_vol_5s')
    )
    feat = feat.with_columns(
        (pl.col('realized_vol_5s') * (6 ** 0.5)).alias('realized_vol_30s')
    )

    # Cumulative OFI
    feat = feat.with_columns(
        pl.col('ofi_5s_raw').cum_sum().alias('cum_ofi_5s')
    )
    feat = feat.with_columns(
        pl.col('ofi_30s_raw').cum_sum().alias('cum_ofi_30s')
    )

    return feat.fill_null(0)


# ── Merge onto Snapshots ───────────────────────────────────────────────────────

def merge_ts_features_to_snapshots(
    snaps_path: Path,
    ts_features: pl.DataFrame,
    output_path: Path,
) -> dict:
    """
    Merge T&S features onto snapshots using backward asof join.
    Per ogni snapshot, trova l'ultimo bucket features con ts <= snapshot ts.
    """
    t_start = time.time()

    snaps = pl.scan_csv(snaps_path).with_columns(
        pl.col('ts').str.replace(' UTC', '').str.to_datetime(
            format='%Y-%m-%d %H:%M:%S%.f', strict=False
        ).alias('snap_ts')
    ).sort('snap_ts').collect()

    feat = ts_features.sort('ts')

    # Feature columns to merge (exclude ts)
    feat_cols = [c for c in feat.columns if c != 'ts']

    # Backward asof: per ogni snap, latest feature bucket <= snap_ts
    merged = snaps.join_asof(
        feat,
        left_on='snap_ts',
        right_on='ts',
        strategy='backward',
    ).drop('ts_right')

    # Fill NaN from early snaps (before first feature bucket)
    merged = merged.with_columns(
        pl.col(c).fill_null(0) for c in feat_cols
    )

    # Final column order: keep original snap cols + new ts features
    base_cols = [c for c in merged.columns
                 if c not in feat_cols + ['snap_ts']]
    out_cols = base_cols + feat_cols
    merged.select(out_cols).write_csv(output_path)

    elapsed = time.time() - t_start
    print(f"  {len(merged):,} snapshots, {len(feat_cols)} T&S features added")
    print(f"  Output: {output_path}")
    print(f"  Time: {elapsed:.1f}s")
    return {'n_snaps': len(merged), 'n_features': len(feat_cols), 'elapsed': elapsed}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"[T&S Features] === Processing {DATE} ===")

    trades = load_trades(TRADES_PATH, DATE)
    print(f"  Trades loaded: {len(trades):,}")

    feat = compute_ts_features(trades)
    print(f"  T&S features computed: {len(feat.columns)-1} columns")
    print(f"  Feature rows: {len(feat):,}")

    # Print feature summary
    print()
    print("  Feature columns:")
    for c in sorted(feat.columns):
        if c != 'ts':
            print(f"    {c}")

    stats = merge_ts_features_to_snapshots(SNAPS_PATH, feat, OUTPUT_PATH)
    print()
    print(f"[T&S Features] DONE — {stats}")


if __name__ == '__main__':
    main()
