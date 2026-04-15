#!/usr/bin/env python3
"""
incremental_p7p8_runner.py — Incremental P7+P8 Pipeline Runner
==============================================================
Scans output/ for days where P1–P6 are done but P7/P8 are not yet complete.
Runs missing P7 candidates (phase7_labeling.py) then P8 (phase8_entry_model.py).
Idempotent via per-phase sentinel files. Supports --workers for parallelism.

USAGE
    python3 NQdom/incremental_p7p8_runner.py --output-dir NQdom/output --workers 4
    python3 NQdom/incremental_p7p8_runner.py --output-dir NQdom/output --dry-run
    python3 NQdom/incremental_p7p8_runner.py --output-dir NQdom/output --force
"""

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

# ── Auto-detect local paths ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
VPS_BASE = str(REPO_ROOT)  # used as cwd for subprocess

PHASE_NAMES = [
    "p1_parse", "p2_reconstruct", "p3_features", "p4_agg",
    "p5_sample", "p6_excursion",
    "p7_c1", "p7_c2", "p7_c3",
    "p8_ml",
]

sys.path.insert(0, str(Path(__file__).parent / "SHARED"))
from _pipeline_constants import CANDIDATES, label_filename

P6_SENTINEL = "p6_excursion.done"
P8_SENTINEL = "p8_ml.done"
MANIFEST_NAME = "_p7p8_incremental_manifest.csv"


# ── enums ─────────────────────────────────────────────────────────────────────

class PhaseStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    SKIPPED  = "skipped"


# ── helpers ───────────────────────────────────────────────────────────────────

def sentinel_path(out_dir: Path, phase: str) -> Path:
    return out_dir / "_checkpoints" / f"{phase}.done"


def sentinel_done(out_dir: Path, phase: str) -> bool:
    return sentinel_path(out_dir, phase).exists()


def write_sentinel(out_dir: Path, phase: str, status: str = "done",
                   error: str = ""):
    p = sentinel_path(out_dir, phase)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = f"status={status}\ntime={dt.datetime.now().isoformat()}\n"
    if error:
        content += f"error={error}\n"
    p.write_text(content, encoding="utf-8")


def run_cmd(cmd: str, timeout: int = 3600, cwd: str = VPS_BASE) -> tuple[str, str, int, float]:
    """Run shell cmd, return (stdout, stderr, exit_code, elapsed_s)."""
    start = time.time()
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd,
                          capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr, r.returncode, time.time() - start
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if e.stdout else ""
        err = e.stderr.decode() if e.stderr else "TIMEOUT"
        return out, err, -1, time.time() - start


def p7_done_for_candidate(out_dir: Path, cidx: int) -> bool:
    """
    True if label file exists with content (source of truth).
    The p7_cN.done sentinel may contain 'failed' from a partial earlier run
    where the process timed out after producing the file — we trust the
    file content over the sentinel status.
    """
    c = CANDIDATES[cidx - 1]
    label_name = label_filename(c["vb_ticks"], c["pt_ticks"], c["sl_ticks"])
    label_path = out_dir / label_name
    return label_path.is_dir() and any(label_path.iterdir())


def all_p7_done(out_dir: Path) -> bool:
    return all(p7_done_for_candidate(out_dir, i) for i in range(1, 4))


def p8_done(out_dir: Path) -> bool:
    return sentinel_done(out_dir, "p8_ml")


def disk_free_gb(output_dir: Path) -> float:
    try:
        stat = os.statvfs(output_dir)
        return stat.f_bavail * stat.f_frsize / (1024**3)
    except Exception:
        return 999.0


# ── P7 runner ─────────────────────────────────────────────────────────────────

def generate_snapshots_if_missing(out_dir: Path) -> tuple[bool, str]:
    """Generate snapshots.csv from events.csv using VPS book_reconstructor module.

    The Dec pipeline (vps_vps_multiday_runner.py) only produced events.csv.
    Phase 7 needs snapshots.csv (DOM ladder with mid_price timestamps).
    If snapshots.csv doesn't exist but events.csv does, run the Numba-accelerated
    book_reconstructor to build it first.
    """
    snapshots_path = out_dir / "snapshots.csv"
    events_path = out_dir / "events.csv"

    if snapshots_path.exists():
        return True, ""  # already present, nothing to do

    if not events_path.exists():
        return False, "both snapshots.csv and events.csv missing — cannot reconstruct"

    print(f"  [P2] snapshots.csv missing — generating from events.csv ...")
    # Use URL-quoted Python to avoid shell-escaping issues with nested quotes
    py_code = (
        f"import sys; sys.path.insert(0, '{VPS_BASE}'); "
        f"from book_reconstructor import reconstruct; "
        f"stats = reconstruct(r'{events_path}', r'{snapshots_path}'); "
        f"print(f'  [P2] Generated {{stats[\"snapshots_generated\"]:,}} snapshots')"
    )
    import urllib.parse
    encoded = urllib.parse.quote(py_code)
    cmd = f"python3 -c 'import urllib.parse; exec(urllib.parse.unquote(\"{encoded}\"))' 2>&1"
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=7200, cwd=VPS_BASE)
    if code != 0 or not snapshots_path.exists():
        msg = stderr[:300] if stderr else "book_reconstructor failed"
        return False, msg
    sz = snapshots_path.stat().st_size / (1024**3)
    print(f"  [P2] snapshots.csv ({sz:.2f} GB) generated in {elapsed:.0f}s")
    return True, ""


def run_phase7_candidate(date: str, out_dir: Path,
                         cidx: int,
                         force: bool = False) -> tuple[bool, str]:
    """Run phase7_labeling.py for one candidate (1-based index)."""
    c = CANDIDATES[cidx - 1]
    phase = f"p7_c{cidx}"

    sampled_path = out_dir / "sampled_events.csv"
    snapshots_path = out_dir / "snapshots.csv"
    excursion_path = out_dir / "excursion_stats.csv"
    grid_path = out_dir / "_candidates_3.csv"

    if not sampled_path.exists():
        return False, "sampled_events.csv missing"

    # Build snapshots.csv from events.csv if missing (Dec pipeline gap)
    if not snapshots_path.exists():
        ok, err = generate_snapshots_if_missing(out_dir)
        if not ok:
            return False, f"snapshots generation failed: {err}"

    if not snapshots_path.exists():
        return False, "snapshots.csv missing (generation failed)"
    if not excursion_path.exists():
        return False, "excursion_stats.csv missing"

    # Write grid file (idempotent)
    with open(grid_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vb_ticks", "pt_ticks", "sl_ticks"])
        w.writeheader()
        for cand in CANDIDATES:
            w.writerow({"vb_ticks": cand["vb_ticks"],
                        "pt_ticks": cand["pt_ticks"], "sl_ticks": cand["sl_ticks"]})

    label_name = label_filename(c["vb_ticks"], c["pt_ticks"], c["sl_ticks"])
    label_path = out_dir / label_name

    # Idempotency: trust the label file as source of truth over sentinel status
    if label_path.is_file() and label_path.stat().st_size > 0 and not force:
        print(f"  [P7-{cidx}/3] {c['desc']} — SKIP (file exists, {label_path.stat().st_size/1024/1024:.1f}MB)")
        return True, ""

    print(f"  [P7-{cidx}/3] {c['desc']} — running ...")
    cmd = (
        f"python3 phase7_labeling.py "
        f"--snapshots {snapshots_path} "
        f"--sampled {sampled_path} "
        f"--refprice {excursion_path} "
        f"--grid {grid_path} "
        f"--output {out_dir} "
        f"--candidates {cidx} "
        f"{'--force' if force else ''} "
        f"2>&1"
    )
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=3600, cwd=VPS_BASE)

    if code == 0 and label_path.exists():
        write_sentinel(out_dir, phase)
        sz = label_path.stat().st_size / (1024**2)
        print(f"  [P7-{cidx}/3] {c['desc']} ({sz:.1f}MB) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = stderr[:200] if stderr else "label dir not produced"
        write_sentinel(out_dir, phase, "failed", msg)
        print(f"  [P7-{cidx}/3] {c['desc']} — FAILED: {msg}")
        return False, msg


# ── P8 runner ─────────────────────────────────────────────────────────────────

def run_phase8(date: str, out_dir: Path,
               force: bool = False) -> tuple[bool, str]:
    """Run phase8_entry_model.py for all 3 candidates in one shot."""
    sampled_path = out_dir / "sampled_events.csv"
    if not sampled_path.exists():
        return False, "sampled_events.csv missing"

    p8_marker = out_dir / "phase8_trainval_results.csv"

    # Idempotency
    if sentinel_done(out_dir, "p8_ml") and not force and p8_marker.exists():
        print(f"  [P8] Already done — SKIP")
        return True, ""

    if p8_marker.exists() and not force:
        print(f"  [P8] phase8_trainval_results.csv exists — SKIP")
        write_sentinel(out_dir, "p8_ml")
        return True, ""

    print(f"  [P8] ML entry model training ...")
    cmd = (
        f"python3 phase8_entry_model.py "
        f"--features {sampled_path} "
        f"--output {out_dir} "
        f"{'--force' if force else ''} "
        f"2>&1"
    )
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=7200, cwd=VPS_BASE)

    if code == 0 and p8_marker.exists():
        write_sentinel(out_dir, "p8_ml")
        sz = p8_marker.stat().st_size / (1024**2)
        print(f"  [P8] phase8_trainval_results.csv ({sz:.1f}MB) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = stderr[:300] if stderr else "phase8 outputs not produced"
        write_sentinel(out_dir, "p8_ml", "failed", msg)
        print(f"  [P8] — FAILED: {msg}")
        return False, msg


# ── Per-day processing ─────────────────────────────────────────────────────────

def process_day(date: str, out_dir: Path,
                force: bool = False,
                manifest_rows: list[dict] = None) -> dict:
    """Process one day through missing P7 candidates + P8. Returns status dict."""
    t0 = time.time()
    result = {"date": date, "p7_run": [], "p7_status": [], "p8_status": "skipped",
              "error": "", "elapsed_s": 0.0}

    print(f"\n[{date}] Checking P7/P8 status ...")

    # ── P7: run missing candidates ───────────────────────────────────────────
    for cidx in range(1, 4):
        c = CANDIDATES[cidx - 1]
        phase = f"p7_c{cidx}"
        already_done = p7_done_for_candidate(out_dir, cidx)
        if already_done:
            print(f"  [P7-{cidx}/3] {c['desc']} — already done, skipping")
            result["p7_status"].append("done")
        else:
            ok, err = run_phase7_candidate(date, out_dir, cidx, force=force)
            result["p7_run"].append(cidx)
            result["p7_status"].append("done" if ok else "failed")
            if not ok:
                result["error"] = f"P7-{cidx} failed: {err}"
                result["elapsed_s"] = time.time() - t0
                return result

    # ── P8: only run if all 3 P7 candidates are done ─────────────────────────
    if all_p7_done(out_dir):
        if p8_done(out_dir) and not force:
            print(f"  [P8] Already done — SKIP")
            result["p8_status"] = "skipped"
        else:
            ok, err = run_phase8(date, out_dir, force=force)
            result["p8_status"] = "done" if ok else "failed"
            if not ok:
                result["error"] = f"P8 failed: {err}"
                result["elapsed_s"] = time.time() - t0
                return result
    else:
        print(f"  [P8] Skipped — not all P7 candidates complete "
              f"({sum(1 for s in result['p7_status'] if s == 'done')}/3 done)")

    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> dict[str, dict]:
    rows = {}
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["date"]] = row
    return rows


def save_manifest(manifest_path: Path, rows: list[dict]):
    if not rows:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_eligible_days(output_dir: Path) -> list[str]:
    """Find days with P6 done but P8 not done (P7 may be partial)."""
    eligible = []
    if not output_dir.exists():
        return eligible
    for day_dir in sorted(output_dir.iterdir()):
        if not day_dir.is_dir() or not day_dir.name.startswith("20"):
            continue
        # Require P6 complete
        if not sentinel_done(day_dir, "p6_excursion"):
            continue
        # Skip if P8 already done
        if p8_done(day_dir):
            continue
        eligible.append(day_dir.name)
    return eligible


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Incremental P7+P8 runner")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output"),
                        help="Path to output directory")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (days)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if sentinels exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print eligible days without running")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / MANIFEST_NAME

    print(f"=== incremental_p7p8_runner.py ===  {dt.datetime.now().isoformat()}")
    print(f"  output-dir : {output_dir}")
    print(f"  workers    : {args.workers}")
    print(f"  force      : {args.force}")
    print(f"  dry-run    : {args.dry_run}")

    # Disk safety check
    free_gb = disk_free_gb(output_dir)
    print(f"  free-disk  : {free_gb:.1f} GB")
    if free_gb < 20.0:
        print(f"  [WARN] Low disk space ({free_gb:.1f} GB < 20 GB) — continuing anyway")

    # Discover eligible days
    eligible = discover_eligible_days(output_dir)
    print(f"\n  Eligible days (P6 done, P8 not done): {len(eligible)}")
    for d in eligible:
        cp_done = [i for i in range(1, 4) if p7_done_for_candidate(output_dir / d, i)]
        print(f"    {d}  — P7 candidates done: {cp_done}/3")

    if not eligible:
        print("  Nothing to do.")
        return

    if args.dry_run:
        print("\n  [dry-run] Would process the above days.")
        return

    # Load existing manifest
    manifest = load_manifest(manifest_path)

    # Process
    if args.workers == 1:
        results = []
        for date in eligible:
            r = process_day(date, output_dir / date, force=args.force)
            results.append(r)
            # Update manifest
            manifest[date] = {
                "date": date,
                "last_run": dt.datetime.now().isoformat(),
                "p7_run": str(r["p7_run"]),
                "p7_status": ",".join(r["p7_status"]),
                "p8_status": r["p8_status"],
                "error": r["error"],
                "elapsed_s": r["elapsed_s"],
            }
    else:
        # Parallel over days
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_day, date, output_dir / date,
                               args.force): date
                for date in eligible
            }
            results = []
            for future in as_completed(futures):
                date = futures[future]
                try:
                    r = future.result()
                    results.append(r)
                except Exception as e:
                    results.append({"date": date, "error": str(e),
                                   "p7_status": [], "p8_status": "crashed"})
        # Update manifest
        for r in results:
            manifest[r["date"]] = {
                "date": r["date"],
                "last_run": dt.datetime.now().isoformat(),
                "p7_run": str(r.get("p7_run", [])),
                "p7_status": ",".join(r.get("p7_status", [])),
                "p8_status": r.get("p8_status", "unknown"),
                "error": r.get("error", ""),
                "elapsed_s": r.get("elapsed_s", 0.0),
            }

    # Save manifest
    save_manifest(manifest_path, list(manifest.values()))

    # Summary
    p8_completed = [r for r in results if r.get("p8_status") == "done"]
    p8_new = len(p8_completed)
    p7_run_total = sum(len(r.get("p7_run", [])) for r in results)
    print(f"\n=== Summary ===")
    print(f"  Days processed  : {len(results)}")
    print(f"  P7 runs done    : {p7_run_total}")
    print(f"  P8 runs done    : {p8_new}")
    print(f"  Errors          : {sum(1 for r in results if r.get('error'))}")
    print(f"  Manifest → {manifest_path}")

    if p8_new > 0:
        print(f"\n  [NOTE] {p8_new} new P8 completion(s) — run aggregate_results.py")


if __name__ == "__main__":
    main()
