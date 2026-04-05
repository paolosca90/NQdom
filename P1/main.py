# VPS-ALIGNED
"""
Sierra Chart .depth Parser - Batch Entry Point
Phase 1: Parse binary .depth file -> events.csv (streaming, per-day)
Phase 2: Reconstruct book from events.csv -> snapshots.csv (streaming, per-day)
Phase 2b: Fuse snapshots.csv + trades.csv -> snapshots_fused.csv (tradedvolbid/tradedvolask)
Phase 3: Compute DOM features -> features_dom.csv (streaming, per-day)
Phase 4: Temporal aggregation (1s/5s/30s) -> features_dom_agg.csv (streaming)
Phase 5: CUSUM event sampling -> sampled_events.csv
Phase 6: Excursion analysis -> excursion_stats.csv + summary + plots

Processes all .depth files in input/ directory, one per day.
Skips phases whose output already exists (idempotent / resumable).
"""

import sys
import csv as csvlib
from pathlib import Path

# ── project root so we can import P2.vps_*, P3.vps_*, etc. ──────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from P1.depth_parser import (
    DepthHeader,
    DepthRecord,
    read_header,
    validate_header,
    records_to_csv_stream,
)
from P2.vps_book_reconstructor import reconstruct
from P3.vps_feature_engineering_vectorized import compute_features_chunked
from P4.vps_feature_engineering_agg import aggregate_features_chunked
from P5.vps_cusum_sampler import cusum_sample
from P6.vps_excursion_analysis_vectorized import (
    build_lookup_index,
    compute_excursions,
    generate_summary,
    plot_distributions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_depth_files(input_dir: Path) -> list[tuple[Path, str]]:
    """
    Find all .depth files in input/ and sub-directories.
    Returns list of (filepath, date_string) sorted by date.
    """
    depth_files = sorted(input_dir.glob("*/*.depth")) or sorted(input_dir.glob("*.depth"))
    if not depth_files:
        raise FileNotFoundError(
            f"No .depth files found in {input_dir}. "
            "Place .depth files in input/ or input/YYYY-MM-DD/ subdirectories."
        )

    results = []
    for f in depth_files:
        # Extract date from filename e.g. NQH26-CME.2026-01-08.depth
        date_str = f.stem.split(".")[-1]  # "2026-01-08"
        results.append((f, date_str))
    return sorted(results, key=lambda x: x[1])


def ensure_output_dir(base_dir: Path, date_str: str) -> Path:
    """Create and return output/YYYY-MM-DD/ directory."""
    out = base_dir / "output" / date_str
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Phase 1 - Streaming parser (no memory accumulation)
# ---------------------------------------------------------------------------

def run_phase1(depth_path: Path, events_path: Path, force: bool = False) -> tuple[bool, DepthHeader, int, list[str]]:
    """
    Parse .depth file and stream to events.csv.
    Returns (skipped, header, record_count, warnings).
    If events.csv already exists and force=False, skips parsing.
    """
    if not force and events_path.exists():
        print(f"  [SKIP] {events_path.name} already exists — use --force to reprocess")
        header = _read_header_only(depth_path)
        warnings = []
        # Estimate record count from file size
        record_count = (depth_path.stat().st_size - 64) // 24
        return True, header, record_count, warnings

    print(f"  Parsing {depth_path.name} ...")
    with open(depth_path, "rb") as fh:
        header = read_header(fh)
        warnings = validate_header(header)

        with open(events_path, "w", newline="", encoding="utf-8") as csv_f:
            writer = csvlib.DictWriter(csv_f, fieldnames=[
                "record_index", "datetime_raw", "datetime_utc",
                "command_code", "command_name", "flags", "end_of_batch",
                "num_orders", "price", "quantity",
            ])
            writer.writeheader()
            record_count, parse_warnings = records_to_csv_stream(fh, writer)
            warnings.extend(parse_warnings)

    print(f"  -> {record_count:,} records -> {events_path.name}")
    return False, header, record_count, warnings


def _read_header_only(path: Path) -> DepthHeader:
    """Read only the 64-byte header without parsing records."""
    with open(path, "rb") as fh:
        return read_header(fh)


# ---------------------------------------------------------------------------
# Phase 2 - Streaming reconstructor
# ---------------------------------------------------------------------------

def run_phase2(events_path: Path, snapshots_path: Path, force: bool = False) -> tuple[bool, dict]:
    """
    Reconstruct book and stream snapshots.csv.
    Returns (skipped, stats).
    """
    if not force and snapshots_path.exists():
        print(f"  [SKIP] snapshots.csv already exists")
        return True, {}

    stats = reconstruct(events_path, snapshots_path)
    return False, stats


# ---------------------------------------------------------------------------
# Phase 2b - Trade + LOB Fusion
# ---------------------------------------------------------------------------

def run_phase2b(snapshots_path: Path, force: bool = False) -> tuple[bool, dict]:
    """
    Fuse snapshots.csv with trades.csv (Time & Sales) to add tradedvolbid/tradedvolask.
    Trades are sourced from the pre-split per-day canonical file at:
      /opt/depth-dom/OUTPUT_TS/by_day/{YYYY-MM-DD}/trades.csv

    Architecture note:
      TS preprocessing is PERSISTENT and ASYNC from depth availability.
      A contract file is split ONCE via split_sierra_trades_by_day.py (STEP 0B).
      When a depth day arrives (even weeks later), this function finds the
      matching trades already present and consumes it automatically.
      Runtime P2b is contract-agnostic — matching is only by date.

    Returns (skipped, stats).
    If trades.csv does not exist: returns (skipped, {"status": "skipped", ...}).
    """
    from P2b.vps_phase2b_data_fusion import load_trades, fuse_chunk
    import pandas as pd

    TS_BY_DAY_BASE = Path("/opt/depth-dom/OUTPUT_TS/by_day")

    sentinel_path = snapshots_path.parent / "_checkpoints" / "p2b_fusion.done"
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive date from the snapshots parent dir name (e.g. "2026-01-08")
    date_str = snapshots_path.parent.name
    trades_path = TS_BY_DAY_BASE / date_str / "trades.csv"

    if not trades_path.exists():
        print(f"  [P2b] WARNING: {trades_path} not found — SKIP (TS may not be pre-split yet)")
        return True, {"status": "skipped", "reason": f"trades.csv not found in OUTPUT_TS/by_day/{date_str}"}

    if not force and sentinel_path.exists():
        print(f"  [SKIP] p2b_fusion.done exists — use --force to reprocess")
        return True, {}

    if not force and (snapshots_path.parent / "snapshots_fused.csv").exists():
        print(f"  [SKIP] snapshots_fused.csv already exists")
        return True, {}

    print(f"  [P2b] Loading trades: {trades_path.name} ...")
    trades_df = load_trades(trades_path)
    if trades_df.empty:
        return True, {"status": "skipped", "reason": "no trades loaded"}

    temp_path = snapshots_path.parent / f".fusing_{snapshots_path.name}.tmp"
    total_snaps = 0
    total_matched = 0
    is_first = True

    CHUNK_SIZE = 250_000
    with pd.read_csv(snapshots_path, chunksize=CHUNK_SIZE, dtype=str) as reader:
        for chunk in reader:
            fused = fuse_chunk(chunk, trades_df)
            # Prune trades already behind current chunk timestamp
            if not fused.empty:
                last_ts = pd.to_datetime(
                    fused['ts'].iloc[-1].replace(" UTC", ""), format="mixed", utc=False
                )
                trades_df = trades_df[trades_df['ts_dt'] > last_ts]
            mode = "w" if is_first else "a"
            fused.to_csv(temp_path, index=False, mode=mode, header=is_first)
            matched = int((fused['traded_vol_bid'] > 0).sum() + (fused['traded_vol_ask'] > 0).sum())
            total_matched += matched
            total_snaps += len(fused)
            is_first = False

    # Atomic replacement
    temp_path.replace(snapshots_path)

    sentinel_path.write_text(
        f"status=done\ntime=\n", encoding="utf-8"
    )
    return False, {"status": "done", "snapshots_processed": total_snaps, "trade_events_fused": total_matched}


# ---------------------------------------------------------------------------
# Phase 3 - Feature Engineering
# ---------------------------------------------------------------------------

def run_phase3(snapshots_path: Path, features_path: Path, force: bool = False) -> tuple[bool, dict]:
    """
    Compute DOM features and stream to features_dom.csv.
    Returns (skipped, stats).
    """
    if not force and features_path.exists():
        print(f"  [SKIP] features_dom.csv already exists")
        return True, {}

    stats = compute_features_chunked(snapshots_path, features_path)
    return False, stats


# ---------------------------------------------------------------------------
# Phase 4 - Temporal Aggregation
# ---------------------------------------------------------------------------

def run_phase4(features_path: Path, features_agg_path: Path, force: bool = False) -> tuple[bool, dict]:
    """
    Compute rolling 1s/5s/30s aggregates and stream to features_dom_agg.csv.
    Returns (skipped, stats).
    """
    if not force and features_agg_path.exists():
        print(f"  [SKIP] features_dom_agg.csv already exists")
        return True, {}

    stats = aggregate_features_chunked(features_path, features_agg_path)
    return False, stats


# ---------------------------------------------------------------------------
# Phase 5 - CUSUM Sampling
# ---------------------------------------------------------------------------

def run_phase5(features_path: Path, agg_path: Path,
               sampled_path: Path, force: bool = False) -> tuple[bool, dict]:
    """
    CUSUM sampling of features_dom.csv + features_dom_agg.csv.
    Returns (skipped, stats).
    """
    if not force and sampled_path.exists():
        print(f"  [SKIP] sampled_events.csv already exists")
        return True, {}

    stats = cusum_sample(features_path, agg_path, sampled_path)
    return False, stats


# ---------------------------------------------------------------------------
# Phase 6 - Excursion Analysis
# ---------------------------------------------------------------------------

def run_phase6(
    features_path: Path,
    sampled_path: Path,
    excursion_path: Path,
    summary_path: Path,
    plot_path: Path,
    force: bool = False,
) -> tuple[bool, dict]:
    """
    Compute excursion statistics from sampled events.
    Returns (skipped, stats).
    """
    if not force and excursion_path.exists():
        print(f"  [SKIP] excursion_stats.csv already exists")
        return True, {}

    print("  Building lookup index from features_dom.csv...")
    ts_ns, mp = build_lookup_index(features_path)
    print(f"  -> {len(ts_ns):,} entries indexed")

    print("  Computing excursions...")
    stats = compute_excursions(sampled_path, ts_ns, mp, excursion_path)
    print(f"  -> {stats['rows_processed']:,} rows processed")

    print("  Generating summary...")
    generate_summary(excursion_path, summary_path)

    print("  Plotting distributions...")
    plot_distributions(excursion_path, plot_path)

    return False, stats


# ---------------------------------------------------------------------------
# Per-day report
# ---------------------------------------------------------------------------

def print_day_report(date_str: str, skipped_p1: bool, skipped_p2: bool,
                     skipped_p2b: bool,
                     skipped_p3: bool, skipped_p4: bool, skipped_p5: bool,
                     skipped_p6: bool,
                     header: DepthHeader | None,
                     record_count: int,
                     p2_stats: dict,
                     p2b_stats: dict,
                     p3_stats: dict,
                     p4_stats: dict,
                     p5_stats: dict,
                     p6_stats: dict) -> None:
    """Print a compact single-line summary per day."""
    if header:
        size_mb = header.file_size / 1024 / 1024
    else:
        size_mb = 0.0

    p1_status = "SKIP" if skipped_p1 else "DONE"
    p2_status = "SKIP" if skipped_p2 else "DONE"
    p2b_status = "SKIP" if skipped_p2b else "DONE"
    p3_status = "SKIP" if skipped_p3 else "DONE"
    p4_status = "SKIP" if skipped_p4 else "DONE"
    p5_status = "SKIP" if skipped_p5 else "DONE"
    p6_status = "SKIP" if skipped_p6 else "DONE"

    if not skipped_p2 and p2_stats:
        snaps = p2_stats.get("snapshots_generated", 0)
        avg_bid = p2_stats.get("avg_bid_levels", 0)
        avg_ask = p2_stats.get("avg_ask_levels", 0)
        if not skipped_p3 and p3_stats:
            feat_rows = p3_stats.get("rows_written", 0)
            agg_rows = p4_stats.get("rows_written", 0) if p4_stats else 0
            sampled_rows = p5_stats.get("rows_sampled", 0) if p5_stats else 0
            excursion_rows = p6_stats.get("rows_processed", 0) if p6_stats else 0
            print(
                f"  {date_str} | P1:{p1_status} P2:{p2_status} P2b:{p2b_status} P3:{p3_status} P4:{p4_status} P5:{p5_status} P6:{p6_status} | "
                f"{record_count:>10,} rec | {snaps:>8,} snaps | {feat_rows:>8,} feat | {agg_rows:>8,} agg | {sampled_rows:>7,} sampled | {excursion_rows:>7,} excursions | "
                f"avg_bid={avg_bid:.0f} avg_ask={avg_ask:.0f} | {size_mb:.1f}MB"
            )
        else:
            print(
                f"  {date_str} | P1:{p1_status} P2:{p2_status} P2b:{p2b_status} P3:{p3_status} P4:{p4_status} P5:{p5_status} P6:{p6_status} | "
                f"{record_count:>10,} rec | {snaps:>8,} snaps | "
                f"avg_bid={avg_bid:.0f} avg_ask={avg_ask:.0f} | {size_mb:.1f}MB"
            )
    else:
        print(
            f"  {date_str} | P1:{p1_status} P2:{p2_status} P2b:{p2b_status} P3:{p3_status} P4:{p4_status} P5:{p5_status} P6:{p6_status} | "
            f"{record_count:>10,} rec | {size_mb:.1f}MB"
        )


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Sierra Chart .depth batch processor")
    parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    parser.add_argument("--input-dir", type=Path, default=None, help="Override input directory")
    parser.add_argument("--days", type=str, default=None,
                        help="Comma-separated dates to process, e.g. 2026-01-08,2026-01-09 (default: all)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    input_dir = args.input_dir or (base_dir / "input")
    force = args.force

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("SIERRA CHART .depth PIPELINE (Phase 1 - 6)")
    print("=" * 70)
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {base_dir / 'output'}")
    print(f"Force reprocess: {force}")
    print()

    try:
        files = find_depth_files(input_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.days:
        target_days = set(args.days.split(","))
        files = [(f, d) for f, d in files if d in target_days]
        if not files:
            print(f"ERROR: None of the specified dates found: {target_days}", file=sys.stderr)
            return 1

    print(f"Found {len(files)} day(s) to process\n")

    # Track totals
    total_records = 0
    total_snapshots = 0
    total_features = 0
    total_agg = 0
    total_sampled = 0
    total_excursions = 0
    skipped_days = 0
    processed_days = 0

    for depth_path, date_str in files:
        print(f"[{date_str}]")
        out_dir = ensure_output_dir(base_dir, date_str)
        events_path = out_dir / "events.csv"
        snapshots_path = out_dir / "snapshots.csv"
        features_path = out_dir / "features_dom.csv"
        features_agg_path = out_dir / "features_dom_agg.csv"
        sampled_path = out_dir / "sampled_events.csv"
        excursion_path = out_dir / "excursion_stats.csv"
        summary_path = out_dir / "excursion_summary.csv"
        plot_path = out_dir / "excursion_distributions.png"

        # Phase 1
        skipped_p1, header, record_count, p1_warnings = run_phase1(
            depth_path, events_path, force
        )

        # Phase 2
        skipped_p2, p2_stats = run_phase2(events_path, snapshots_path, force)

        # Phase 2b — trades sourced from OUTPUT_TS/by_day/{date}/trades.csv
        skipped_p2b, p2b_stats = run_phase2b(snapshots_path, force)

        # Phase 3
        skipped_p3, p3_stats = run_phase3(snapshots_path, features_path, force)

        # Phase 4
        skipped_p4, p4_stats = run_phase4(features_path, features_agg_path, force)

        # Phase 5
        skipped_p5, p5_stats = run_phase5(
            features_path, features_agg_path, sampled_path, force
        )

        # Phase 6
        skipped_p6, p6_stats = run_phase6(
            snapshots_path, sampled_path, excursion_path, summary_path, plot_path, force
        )

        print_day_report(
            date_str, skipped_p1, skipped_p2, skipped_p2b, skipped_p3, skipped_p4,
            skipped_p5, skipped_p6, header,
            record_count if not skipped_p1 else 0,
            p2_stats, p2b_stats, p3_stats, p4_stats, p5_stats, p6_stats
        )

        if p1_warnings and not skipped_p1:
            for w in p1_warnings[:3]:
                print(f"  ! {w}", file=sys.stderr)

        if not skipped_p1:
            total_records += record_count
        if not skipped_p2 and p2_stats:
            total_snapshots += p2_stats.get("snapshots_generated", 0)
        if not skipped_p3 and p3_stats:
            total_features += p3_stats.get("rows_written", 0)
        if not skipped_p4 and p4_stats:
            total_agg += p4_stats.get("rows_written", 0)
        if not skipped_p5 and p5_stats:
            total_sampled += p5_stats.get("rows_sampled", 0)
        if not skipped_p6 and p6_stats:
            total_excursions += p6_stats.get("rows_processed", 0)

        if (skipped_p1 and skipped_p2 and skipped_p2b and
                skipped_p3 and skipped_p4 and skipped_p5 and skipped_p6):
            skipped_days += 1
        else:
            processed_days += 1

    # Final summary
    print()
    print("=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"  Days processed : {processed_days}")
    print(f"  Days skipped   : {skipped_days}")
    print(f"  Total records  : {total_records:,}")
    print(f"  Total snapshots: {total_snapshots:,}")
    print(f"  Total features : {total_features:,}")
    print(f"  Total agg rows : {total_agg:,}")
    print(f"  Total sampled  : {total_sampled:,}")
    print(f"  Total excursions: {total_excursions:,}")
    print(f"  Output dir     : {base_dir / 'output'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
