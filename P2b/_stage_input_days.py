"""
Stage INPUT directory into date-subdirectory structure:
INPUT/2026-03-13/NQM26-CME.2026-03-13.depth  (symlink)
INPUT/2026-03-13/trades.csv                   (from T&S split)

This allows run_p1_to_p7_multiday.py's discover_days() to find
trades.csv as a sibling to each .depth file.
"""
import os
import csv
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path("C:/Users/Paolo/Desktop/NQ/NQdom")
INPUT_DIR = ROOT / "INPUT"
TS_FILE = ROOT / "INPUT_TS" / "NQM26-CME.txt"
BY_DAY_TRADES = {p.parent.name: p for p in INPUT_DIR.glob("*/trades.csv")}

def parse_ts_date(dt_str: str) -> str | None:
    """Extract YYYY-MM-DD from T&S timestamp."""
    try:
        dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M:%S.%f")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

def main():
    # Group T&S trades by date
    by_day: dict[str, list] = {}
    total = 0
    with open(TS_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            parts = line.split(",")
            if len(parts) < 5:
                continue
            date_str = parse_ts_date(parts[0])
            if not date_str:
                continue
            try:
                price = float(parts[3].strip())
                qty = int(parts[4].strip())
            except ValueError:
                continue
            by_day.setdefault(date_str, []).append({
                "ts": parts[0].strip(),
                "price": price,
                "qty": qty
            })

    print(f"Parsed {total:,} T&S lines, {len(by_day)} days")

    # Depth files at root
    depth_files = sorted(INPUT_DIR.glob("NQM26-CME.*.depth"))
    print(f"Depth files: {len(depth_files)}")

    created = 0
    skipped_no_trades = []
    for depth_path in depth_files:
        date_str = depth_path.stem.split(".")[-1]  # "2026-03-13"
        day_dir = INPUT_DIR / date_str

        # Create date subdir
        day_dir.mkdir(exist_ok=True)

        # Symlink depth file into date dir (if not already there as a file)
        depth_in_day = day_dir / depth_path.name
        if not depth_in_day.exists():
            try:
                os.symlink(depth_path.resolve(), depth_in_day)
                print(f"  {date_str}: symlinked {depth_path.name}")
            except OSError:
                import shutil
                shutil.copy2(depth_path, depth_in_day)
                print(f"  {date_str}: copied {depth_path.name}")
        else:
            print(f"  {date_str}: depth already staged")

        # Write trades.csv if T&S data exists for this day
        if date_str in by_day:
            trades_path = day_dir / "trades.csv"
            if not trades_path.exists():
                with open(trades_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["ts", "price", "qty"])
                    writer.writeheader()
                    writer.writerows(by_day[date_str])
                print(f"  {date_str}: wrote {len(by_day[date_str]):,} trades")
            else:
                print(f"  {date_str}: trades.csv already exists")
            created += 1
        else:
            skipped_no_trades.append(date_str)

    print(f"\nStaged {created} days with trades.csv")
    if skipped_no_trades:
        print(f"Days without T&S data: {skipped_no_trades}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
