#!/usr/bin/env python3
"""
split_sierra_trades_by_day.py — STEP 0B Preprocessing Layer
===========================================================
Sierra Chart Time & Sales contract-file → per-day canonical trades.

Architecture:
  /opt/depth-dom/INPUT_TS/             ← raw Sierra exports (NQH26-CME.txt, NQM26-CME.txt, ...)
  /opt/depth-dom/OUTPUT_TS/by_day/     ← pre-split daily trades (persistent, date-keyed)
      YYYY-MM-DD/
          trades.csv
          _meta.json

Design principle:
  TS preprocessing is PERSISTENT and ASYNC from depth availability.
  A contract file is split ONCE into daily files. When a depth day arrives
  (even weeks later), run_phase2b() finds the matching trades already present
  at OUTPUT_TS/by_day/{DAY}/trades.csv and consumes it automatically.
  Runtime P2b is contract-agnostic — matching is only by date.

Default split mode: ALL contract days → OUTPUT_TS/by_day/
  (use --only-days-present-in-output to restrict to already-processed depth days)

Usage:
  # Default: split all days in the contract file
  python3 split_sierra_trades_by_day.py --input /opt/depth-dom/INPUT_TS/NQH26-CME.txt

  # Audit mode: only materialize days that already exist in OUTPUT/
  python3 split_sierra_trades_by_day.py \
      --input /opt/depth-dom/INPUT_TS/NQM26-CME.txt \
      --only-days-present-in-output \
      --output-ref-dir /opt/depth-dom/OUTPUT/

  # Force overwrite of specific conflicting days
  python3 split_sierra_trades_by_day.py --input ... --force-rebuild-day
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TS_OUTPUT_BASE = Path("/opt/depth-dom/OUTPUT_TS/by_day")
TS_CONTRACTS_DIR = Path("/opt/depth-dom/INPUT_TS")
OUTPUT_BASE = Path("/opt/depth-dom/output")

CHUNK_SIZE = 100_000  # rows per read chunk — RAM-safe for large files

# Canonical output columns
CANONICAL_COLS = ["ts", "price", "size", "side"]

# ---------------------------------------------------------------------------
# Sierra format detection & parsing
# ---------------------------------------------------------------------------

SIERRA_REQUIRED = {"date", "time", "last", "volume", "bidvolume", "askvolume"}
CANONICAL_REQUIRED = {"ts", "price", "size", "side"}


def detect_format(first_row: dict) -> str:
    """Detect whether the file is Sierra export or canonical format."""
    # Strip whitespace from column names (Sierra exports often have leading spaces)
    cols = {k.strip().lower() for k in first_row.keys()}
    if SIERRA_REQUIRED.issubset(cols):
        return "sierra"
    if CANONICAL_REQUIRED.issubset(canonical_from_any(first_row, cols)):
        return "canonical"
    raise ValueError(
        f"Unrecognized trade file format.\n"
        f"  Sierra required cols : {sorted(SIERRA_REQUIRED)}\n"
        f"  Canonical required  : {sorted(CANONICAL_REQUIRED)}\n"
        f"  File cols           : {sorted(cols)}"
    )


def canonical_from_any(row: dict, cols: set) -> set:
    """Return canonical col names found in row (lowercased, stripped)."""
    return {c.strip().lower() for c in row.keys()}


# ---------------------------------------------------------------------------
# Sierra -> Canonical row transformer
# ---------------------------------------------------------------------------

def _get(row: dict, key: str) -> str:
    """Case-insensitive, whitespace-tolerant key lookup."""
    key_lower = key.lower()
    for k, v in row.items():
        if k.strip().lower() == key_lower:
            return v
    return ""


def sierra_row_to_canonical(row: dict) -> dict | None:
    """
    Transform one Sierra row to canonical dict.
    Returns None if the row should be skipped (ambiguous / zero volume).
    """
    # Build timestamp — normalize date to zero-padded YYYY-MM-DD for path consistency
    raw_date = _get(row, "date").strip()  # e.g. "2025/11/28" or "2025/12/1"
    parts = raw_date.replace("/", "-").split("-")  # ["2025", "11", "28"] or ["2025", "12", "1"]
    if len(parts) == 3:
        try:
            date_iso = f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        except ValueError:
            return None
    else:
        date_iso = raw_date.replace("/", "-")  # fallback
    time_str = _get(row, "time").strip()
    if not date_iso or not time_str:
        return None

    ts = f"{date_iso} {time_str}"

    # Price and size
    try:
        price = float(_get(row, "last"))
        size = int(float(_get(row, "volume")))
    except ValueError:
        return None

    if size <= 0:
        return None

    # Derive side from BidVolume / AskVolume
    try:
        bv = float(_get(row, "bidvolume") or 0)
        av = float(_get(row, "askvolume") or 0)
    except ValueError:
        bv, av = 0.0, 0.0

    # Rules:
    # - AskVol > 0 AND BidVol == 0  → buy  (lifted Ask)
    # - BidVol > 0 AND AskVol == 0  → sell (hit Bid)
    # - Both > 0: AskVol > BidVol   → buy
    # - Both > 0: BidVol > AskVol    → sell
    # - Both == 0 or equal           → skip
    if av > 0 and bv == 0:
        side = "buy"
    elif bv > 0 and av == 0:
        side = "sell"
    elif av > bv:
        side = "buy"
    elif bv > av:
        side = "sell"
    else:
        return None  # ambiguous or zero — skip

    return {"ts": ts, "price": price, "size": size, "side": side}


# ---------------------------------------------------------------------------
# Streaming row reader (detects format on first row)
# ---------------------------------------------------------------------------

class StreamingTradeReader:
    """
    Stream a Sierra or canonical trade file row-by-row.
    Detects format from the first data row (skips header row automatically).
    Yields canonical dicts: {ts, price, size, side}
    """

    def __init__(self, path: Path):
        self.path = path
        self.format = None
        self._reader = None
        self._iter = None
        self._header = None

    def _open(self):
        """Open file and detect format from first data row."""
        encoding = self._try_encoding()
        self._file = open(self.path, "r", newline="", encoding=encoding)
        self._reader = csv.DictReader(self._file)
        self._header = self._reader.fieldnames
        first_row = next(self._reader, None)
        if first_row is None:
            raise ValueError(f"Trade file is empty: {self.path}")
        self.format = detect_format(first_row)
        self._iter = self._reader
        return first_row  # yield the first row we already consumed

    def _try_encoding(self) -> str:
        """Try utf-8 first, fall back to latin-1 for files with special chars."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                f.read(1024)
            return "utf-8"
        except UnicodeDecodeError:
            return "latin-1"

    def __iter__(self) -> "StreamingTradeReader":
        first = self._open()
        # Strip whitespace from column names in first row
        first = {k.strip(): v for k, v in first.items()}
        yield first  # emit first row (already read for format detection)
        for row in self._iter:
            # Normalize keys: strip leading/trailing whitespace from column names
            yield {k.strip(): v for k, v in row.items()}

    def close(self):
        if hasattr(self, "_file") and self._file:
            self._file.close()


# ---------------------------------------------------------------------------
# Day splitter core
# ---------------------------------------------------------------------------

def parse_date_from_sierra_row(row: dict) -> str | None:
    """Extract YYYY-MM-DD from Sierra date field (handles '2026/2/27' and '2026-02-27')."""
    d = row.get("date", "").strip()
    if not d:
        return None
    # Replace slashes with dashes, then try to parse
    d = d.replace("/", "-")
    # Handle mixed formats: "2026-2-27" -> "2026-02-27"
    parts = d.split("-")
    if len(parts) == 3:
        try:
            year, month, day = parts
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        except ValueError:
            return None
    return None


def parse_date_from_canonical_row(row: dict) -> str | None:
    """Extract YYYY-MM-DD from canonical ts field."""
    ts = row.get("ts", "").strip()
    if not ts:
        return None
    # ts format: "2026-02-27 00:01:25.226" or similar
    # Just take the date part
    return ts.split(" ")[0] if " " in ts else None


# ---------------------------------------------------------------------------
# Per-day writer
# ---------------------------------------------------------------------------

class PerDayTradeWriter:
    """
    Writes trades to OUTPUT_TS/by_day/{YYYY-MM-DD}/trades.csv
    with _meta.json written after CSV is flushed.

    Tracks: rows_written, skipped_ambiguous, skipped_zero, per-day stats.
    """

    def __init__(
        self,
        output_base: Path,
        contract_code: str,
        source_file: Path,
        split_mode: str,
        allowed_days: set[str] | None = None,
        force_rebuild: bool = False,
    ):
        self.output_base = output_base
        self.contract_code = contract_code
        self.source_file = source_file
        self.split_mode = split_mode
        self.allowed_days = allowed_days  # None = all days allowed
        self.force_rebuild = force_rebuild

        # Stats
        self.total_rows_seen = 0
        self.total_ambiguous = 0
        self.total_zero = 0
        self.total_written = 0

        # Per-day counters
        self.days_found: set[str] = set()
        self.days_written: set[str] = set()
        self.days_skipped_existing: set[str] = set()
        self.days_out_of_scope: set[str] = set()

        # Day stats (date -> trade count)
        self.day_trade_counts: dict[str, int] = {}

        # Current day buffer (list of canonical rows)
        self._current_day: str | None = None
        self._current_rows: list[dict] = []
        self._current_file: Path | None = None
        self._current_writer: csv.DictWriter | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_row(self, canonical_row: dict):
        """Accept a canonical-format trade row. Flushes previous day when date changes."""
        day = canonical_row.get("ts", "")[:10]  # "YYYY-MM-DD" (properly zero-padded)
        if not day:
            return

        self.total_rows_seen += 1

        # Check scope filter
        if self.allowed_days is not None and day not in self.allowed_days:
            self.days_out_of_scope.add(day)
            return

        # Day changed → flush previous day
        if self._current_day is not None and day != self._current_day:
            self._flush_day()

        self._current_day = day
        self.days_found.add(day)
        self._current_rows.append(canonical_row)

    def flush(self):
        """Flush the last open day. Call at EOF."""
        self._flush_day()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _day_dir(self, day: str) -> Path:
        return self.output_base / day

    def _day_trades_path(self, day: str) -> Path:
        return self._day_dir(day) / "trades.csv"

    def _day_meta_path(self, day: str) -> Path:
        return self._day_dir(day) / "_meta.json"

    def _check_conflict(self, day: str) -> tuple[bool, str | None]:
        """
        Check if day already exists.
        Returns (blocked, existing_contract).
        blocked=True means we cannot write without --force.
        """
        trades_path = self._day_trades_path(day)
        if not trades_path.exists():
            return False, None

        # Day exists — check if we should block
        meta_path = self._day_meta_path(day)
        existing_contract = "unknown"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                existing_contract = meta.get("contract_code", "unknown")
            except Exception:
                pass

        if self.force_rebuild:
            return False, existing_contract  # allowed to overwrite

        return True, existing_contract

    def _flush_day(self):
        """Write buffered rows for _current_day to disk."""
        if self._current_day is None or not self._current_rows:
            return
        day = self._current_day
        rows = self._current_rows

        # Conflict check
        blocked, existing_contract = self._check_conflict(day)
        if blocked:
            print(
                f"  CONFLICT: day {day} already exists in OUTPUT_TS/by_day from contract {existing_contract}.\n"
                f"           This may be a normal rollover overlap. Verify which contract should own the day,\n"
                f"           then re-run with --force-rebuild-day only if overwrite is intentional."
            )
            self.days_skipped_existing.add(day)
            self._current_day = None
            self._current_rows = []
            return

        # Write CSV
        day_dir = self._day_dir(day)
        day_dir.mkdir(parents=True, exist_ok=True)
        trades_path = self._day_trades_path(day)

        mode = "w"
        if trades_path.exists() and self.force_rebuild:
            mode = "w"
        elif not trades_path.exists():
            mode = "w"
        else:
            # Already checked conflict above; if we get here with force, overwrite
            mode = "w"

        with open(trades_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CANONICAL_COLS, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
            writer.writerows(rows)

        trade_count = len(rows)
        self.total_written += trade_count
        self.day_trade_counts[day] = self.day_trade_counts.get(day, 0) + trade_count
        self.days_written.add(day)

        # Write _meta.json
        meta = {
            "source_contract_file": str(self.source_file),
            "contract_code": self.contract_code,
            "trade_date": day,
            "rows_written": trade_count,
            "skipped_ambiguous_rows": 0,  # counted at row level
            "skipped_zero_rows": 0,
            "source_format": "sierra_export",
            "split_mode": self.split_mode,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "output_path": str(trades_path),
        }
        self._day_meta_path(day).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Reset buffer
        self._current_rows = []

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "total_rows_seen": self.total_rows_seen,
            "total_ambiguous": self.total_ambiguous,
            "total_zero": self.total_zero,
            "total_written": self.total_written,
            "days_found": sorted(self.days_found),
            "days_written": sorted(self.days_written),
            "days_skipped_existing": sorted(self.days_skipped_existing),
            "days_out_of_scope": sorted(self.days_out_of_scope),
            "day_trade_counts": {k: v for k, v in sorted(self.day_trade_counts.items())},
        }


# ---------------------------------------------------------------------------
# Main split logic
# ---------------------------------------------------------------------------

def split_contract(
    input_path: Path,
    output_base: Path,
    contract_code: str,
    split_mode: str,
    allowed_days: set[str] | None = None,
    force_rebuild: bool = False,
    verbose: bool = True,
    max_rows: int | None = None,
):
    """
    Stream-split a Sierra contract file into per-day trades.csv files.

    Parameters
    ----------
    input_path     : path to the raw Sierra contract export
    output_base    : base directory for OUTPUT_TS/by_day/
    contract_code  : e.g. "NQH26"
    split_mode     : "all_contract_days" or "only_days_present_in_output"
    allowed_days   : set of YYYY-MM-DD strings (for filtering mode)
    force_rebuild  : overwrite existing days
    verbose        : print progress
    max_rows       : if set, stop after processing this many rows (test mode)
    """
    t0 = time.time()
    writer = PerDayTradeWriter(
        output_base=output_base,
        contract_code=contract_code,
        source_file=input_path,
        split_mode=split_mode,
        allowed_days=allowed_days,
        force_rebuild=force_rebuild,
    )

    if verbose:
        print(f"[SPLIT] Contract: {contract_code} | Source: {input_path}")
        sz_mb = input_path.stat().st_size / (1024 ** 2)
        print(f"[SPLIT] File size: {sz_mb:,.0f} MB")

    reader = StreamingTradeReader(input_path)
    rows_in_current_day = 0
    rows_read_total = 0

    for row in reader:
        rows_read_total += 1
        if max_rows is not None and rows_read_total > max_rows:
            if verbose:
                print(f"[SPLIT] --max-rows limit reached ({max_rows:,}) — stopping")
            break
        if reader.format == "sierra":
            canonical = sierra_row_to_canonical(row)
            if canonical is None:
                writer.total_ambiguous += 1
                continue
        else:
            # Canonical: just validate required fields
            try:
                canonical = {
                    "ts": row["ts"],
                    "price": float(row["price"]),
                    "size": int(float(row["size"])),
                    "side": row["side"].lower().strip(),
                }
            except (ValueError, KeyError):
                writer.total_zero += 1
                continue

        writer.write_row(canonical)
        rows_in_current_day += 1

    reader.close()
    writer.flush()  # flush last day

    # Finalize: update _meta.json with aggregate stats
    summary = writer.summary()

    for day in summary["days_written"]:
        meta_path = writer._day_meta_path(day)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["skipped_ambiguous_rows"] = summary["total_ambiguous"]
            meta["skipped_zero_rows"] = summary["total_zero"]
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Print summary
    if verbose:
        _print_summary(summary, split_mode, time.time() - t0)

    return summary


def _print_summary(s: dict, split_mode: str, elapsed_s: float):
    days_found = s["days_found"]
    days_written = s["days_written"]
    days_skipped = s["days_skipped_existing"]
    days_out = s["days_out_of_scope"]

    print(f"[SPLIT] Contract days found in file: {len(days_found)}")
    print(f"[SPLIT] Split mode: {split_mode}")

    if days_out:
        print(f"[SPLIT] Days out of scope (not in OUTPUT/): {len(days_out)}")
        if len(days_out) <= 10:
            print(f"        → {', '.join(sorted(days_out))}")
        else:
            shown = sorted(days_out)[:5]
            print(f"        → {', '.join(shown)} ... ({len(days_out) - 5} more)")

    if days_skipped:
        print(f"[SPLIT] Days skipped (already exist): {len(days_skipped)}")
        if len(days_skipped) <= 10:
            print(f"        → {', '.join(sorted(days_skipped))}")
        else:
            shown = sorted(days_skipped)[:5]
            print(f"        → {', '.join(shown)} ... ({len(days_skipped) - 5} more)")

    print(f"[SPLIT] Days materialized: {len(days_written)}/{len(days_found)}")

    if s["day_trade_counts"]:
        top = sorted(s["day_trade_counts"].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"[SPLIT] Top 3 days by trade volume:")
        for rank, (day, count) in enumerate(top, 1):
            print(f"        {rank}. {day}  →  {count:,} trades")

    print(f"[SPLIT] Total rows seen  : {s['total_rows_seen']:,}")
    print(f"[SPLIT] Total rows written: {s['total_written']:,}")
    print(f"[SPLIT] Skipped (ambiguous): {s['total_ambiguous']:,}")
    print(f"[SPLIT] Skipped (zero/ invalid): {s['total_zero']:,}")
    print(f"[SPLIT] Elapsed: {elapsed_s:.1f}s")

    if not days_written:
        print(f"[SPLIT] WARNING: No days written. Check --only-days-present-in-output filter or file format.")


def discover_output_days(output_ref_dir: Path) -> set[str]:
    """
    Return set of YYYY-MM-DD strings for all day directories already present
    in output_ref_dir (e.g. /opt/depth-dom/OUTPUT/).
    """
    if not output_ref_dir.exists():
        return set()
    days = set()
    for entry in output_ref_dir.iterdir():
        if entry.is_dir() and entry.name[:4].isdigit():
            days.add(entry.name)
    return days


def derive_contract_code(input_path: Path, override: str | None) -> str:
    """Derive contract code from filename if not explicitly provided."""
    if override:
        return override.upper()
    # nqh26.txt -> NQH26, nqm26 -> NQM26
    name = input_path.stem.upper()
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="split_sierra_trades_by_day.py",
        description="Split a Sierra Chart Time & Sales contract export into per-day trades files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default: split all days in the contract\n"
            "  python3 split_sierra_trades_by_day.py --input /opt/depth-dom/INPUT_TS/NQH26-CME.txt\n"
            "\n"
            "  # Only materialize days already present in OUTPUT/\n"
            "  python3 split_sierra_trades_by_day.py --input ... \\\n"
            "      --only-days-present-in-output --output-ref-dir /opt/depth-dom/output/\n"
            "\n"
            "  # Force overwrite of existing conflicting days\n"
            "  python3 split_sierra_trades_by_day.py --input ... --force-rebuild-day\n"
        ),
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the Sierra contract Time & Sales export file",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=TS_OUTPUT_BASE,
        metavar="PATH",
        help=f"Base output directory for by_day/ subdirectory (default: {TS_OUTPUT_BASE})",
    )
    ap.add_argument(
        "--contract-code",
        type=str,
        default=None,
        metavar="CODE",
        help="Contract code (e.g. NQH26). Auto-derived from filename if omitted.",
    )
    ap.add_argument(
        "--only-days-present-in-output",
        action="store_true",
        dest="only_days_present",
        help=(
            "Restrict split to only days that already exist in --output-ref-dir. "
            "By default ALL contract days are materialized."
        ),
    )
    ap.add_argument(
        "--output-ref-dir",
        type=Path,
        default=OUTPUT_BASE,
        metavar="PATH",
        help=(
            "Reference directory to check for already-processed depth days. "
            f"Used only with --only-days-present-in-output. (default: {OUTPUT_BASE})"
        ),
    )
    ap.add_argument(
        "--force-rebuild-day",
        action="store_true",
        help="Overwrite days that already exist in OUTPUT_TS/by_day/ (default: block on conflict)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing N rows. For dry-run / test mode only. Default: None (process all).",
    )
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    input_path: Path = args.input
    output_base: Path = args.output_dir
    contract_code = derive_contract_code(input_path, args.contract_code)
    force_rebuild: bool = args.force_rebuild_day

    # Validate input exists
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Validate: --only-days-present-in-output requires --output-ref-dir to exist
    if args.only_days_present and not args.output_ref_dir.exists():
        print(
            f"ERROR: --only-days-present-in-output requires --output-ref-dir to exist.\n"
            f"       {args.output_ref_dir} does not exist.\n"
            f"       Either create the directory first or drop --only-days-present-in-output.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine split mode
    if args.only_days_present:
        split_mode = "only_days_present_in_output"
        allowed_days = discover_output_days(args.output_ref_dir)
        if not allowed_days:
            print(
                f"WARNING: --only-days-present-in-output specified but no days found in\n"
                f"         {args.output_ref_dir}. No days will be materialized.",
                file=sys.stderr,
            )
        else:
            print(f"[SPLIT] Filtering to {len(allowed_days)} days present in {args.output_ref_dir}")
    else:
        split_mode = "all_contract_days"
        allowed_days = None

    # Ensure output base exists
    output_base.mkdir(parents=True, exist_ok=True)

    # Print config
    print(f"[SPLIT] Output base : {output_base}")
    print(f"[SPLIT] Contract code: {contract_code}")
    print(f"[SPLIT] Split mode  : {split_mode}")
    if allowed_days is not None:
        print(f"[SPLIT] Allowed days : {len(allowed_days)} (from {args.output_ref_dir})")
    if args.max_rows is not None:
        print(f"[SPLIT] *** TEST MODE *** max-rows: {args.max_rows:,}")
    print()

    try:
        summary = split_contract(
            input_path=input_path,
            output_base=output_base,
            contract_code=contract_code,
            split_mode=split_mode,
            allowed_days=allowed_days,
            force_rebuild=force_rebuild,
            max_rows=args.max_rows,
        )

        if not summary["days_written"]:
            print(
                "\nWARNING: No days were written. Possible causes:\n"
                "  - Contract file has no data rows\n"
                "  - --only-days-present-in-output filtered out all days\n"
                "  - File format not recognized\n"
                "  - All days already exist (use --force-rebuild-day to overwrite)",
                file=sys.stderr,
            )
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: Split failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
