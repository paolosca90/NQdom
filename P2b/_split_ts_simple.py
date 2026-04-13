"""
Simple T&S splitter - reads NQM26-CME.txt and writes trades.csv
to each date directory alongside the .depth file.
Run once before run_p1_to_p7_multiday.py.
"""
import csv
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path("C:/Users/Paolo/Desktop/NQ/NQdom")
TS_FILE = ROOT / "INPUT_TS" / "NQM26-CME.txt"
INPUT_DIR = ROOT / "INPUT"

# Format: Date,Time,Open,High,Low,Last,Volume,NumberOfTrades,BidVolume,AskVolume
# Date: YYYY/M/D  Time: H:MM:SS.mmm (space-padded hour)


def parse_ts_line(line: str) -> tuple[str, dict] | None:
    """Parse a Sierra Chart T&S CSV line. Returns (date_str, trade_dict) or None."""
    line = line.strip()
    if not line:
        return None
    try:
        # Format: Date,Time,Open,High,Low,Last,Volume,...
        # Example: 2026/3/2, 13:40:14.059,24895.0,...
        parts = line.split(",")
        if len(parts) < 7:
            return None
        # Parse date: YYYY/M/D -> YYYY-MM-DD
        date_raw = parts[0].strip()  # "2026/3/2"
        date_parts = date_raw.split("/")
        date_str = f"{int(date_parts[0]):04d}-{int(date_parts[1]):02d}-{int(date_parts[2]):02d}"
        # Parse time: " 13:40:14.059" -> "13:40:14.059"
        time_str = parts[1].strip()
        ts_str = f"{date_str} {time_str}"
        price = float(parts[5].strip())  # Last column
        qty = int(parts[6].strip())     # Volume column
        bid_qty = int(parts[8].strip()) if len(parts) > 8 else 0
        ask_qty = int(parts[9].strip()) if len(parts) > 9 else 0
        # Derive side (aggressor): buy=hit ask, sell=hit bid
        if ask_qty > 0 and bid_qty == 0:
            side = "buy"
        elif bid_qty > 0 and ask_qty == 0:
            side = "sell"
        elif ask_qty > bid_qty:
            side = "buy"
        elif bid_qty > ask_qty:
            side = "sell"
        else:
            side = "_skip_"
        return date_str, {
            "ts": ts_str,
            "price": price,
            "size": qty,
            "side": side,
        }
    except (ValueError, IndexError):
        return None


def get_output_path(date_str: str) -> Path:
    """Return path where trades.csv should be written for this date."""
    return INPUT_DIR / date_str / "trades.csv"


def main():
    print(f"Reading T&S from: {TS_FILE}")
    print(f"Output dir:       {INPUT_DIR}")

    if not TS_FILE.exists():
        print(f"ERROR: {TS_FILE} not found")
        return 1

    # Collect trades per day
    by_day: dict[str, list] = {}
    total_lines = 0
    skipped = 0

    with open(TS_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_lines += 1
            result = parse_ts_line(line)
            if result is None:
                skipped += 1
                continue
            date_str, trade = result
            if date_str not in by_day:
                by_day[date_str] = []
            by_day[date_str].append(trade)

    print(f"Parsed {total_lines:,} lines: {len(by_day)} days, {skipped:,} skipped")
    if by_day:
        sample = list(by_day.keys())[:3]
        for d in sample:
            print(f"  {d}: {len(by_day[d]):,} trades")

    # Check which days have depth files (at root or in date subdirs)
    depth_files = {}
    for p in INPUT_DIR.glob("NQM26-CME.*.depth"):
        date_str = p.stem.split(".")[-1]
        depth_files[date_str] = p
    for p in INPUT_DIR.glob("*/NQM26-CME.*.depth"):
        date_str = p.stem.split(".")[-1]
        depth_files[date_str] = p
    print(f"Depth files found: {len(depth_files)}")

    days_with_both = sorted(by_day.keys() & depth_files.keys())
    print(f"Days with both T&S and depth: {len(days_with_both)}")

    # Ensure date subdirs exist (from _stage_input_days.py)
    for date_str in days_with_both:
        (INPUT_DIR / date_str).mkdir(exist_ok=True)

    # Write trades.csv for each day
    written = 0
    for date_str in days_with_both:
        out_path = get_output_path(date_str)

        fieldnames = ["ts", "price", "size", "side"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(by_day[date_str])

        n = len(by_day[date_str])
        print(f"  {date_str}: {n:,} trades -> {out_path.name}")
        written += 1

    print(f"\nDone: wrote {written} trades.csv files to INPUT/<date>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
