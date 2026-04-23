#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_p1_to_p7_multiday.py — P1→P7 (inclusive P2b) Pipeline
==========================================================
Esegue l'intero pipeline P1–P7 (con P2b merge T&S) su tutti i giorni
disponibili in input-dir.

Uso:
    python run_p1_to_p7_multiday.py --input-dir INPUT --output-dir output --resume

Lo script è sequenziale (RAM-safe) e supporta resume/force.
P2b richiede trades.csv accanto al .depth file.
Se trades assente per un giorno, P2b skippa (warning) e il pipeline continua.

Fasi eseguite:
    P1 → events.csv
    P2 → snapshots.csv
    P2b → snapshots_fused.csv  (merge T&S con LOB)
    P3 → features_dom.csv
    P4 → features_dom_agg.csv
    P5 → sampled_events.csv
    P6 → excursion_stats.csv
    P7 → phase7_labels_{vb}ticks_*/  (3 candidati)
"""

import argparse
import csv
import datetime as dt
import gc
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path

# ── project root for imports ────────────────────────────────────────────────
_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(_root))

# ── phase imports ────────────────────────────────────────────────────────────
from P1.main import run_phase1 as p1_run_phase1
from P1.depth_parser import DepthHeader
from P2.vps_book_reconstructor import reconstruct as p2_reconstruct
from P3.vps_feature_engineering_vectorized import compute_features_chunked as p3_compute
from P4.vps_feature_engineering_agg import aggregate_features_chunked as p4_aggregate
from P5.vps_cusum_sampler import cusum_sample as p5_sample
from P6.vps_excursion_analysis_vectorized import (
    build_lookup_index as p6_build_index,
    compute_excursions as p6_compute,
    generate_summary as p6_summary,
    plot_distributions as p6_plot,
)
from P2b.vps_phase2b_data_fusion import load_trades as p2b_load_trades, fuse_chunk as p2b_fuse_chunk

# Fix Unicode output on Windows console (cp1252 can't encode arrows/emoji)
import io
if hasattr(sys.stdout, "encoding") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── constants ─────────────────────────────────────────────────────────────────
INPUT_DIR_DEFAULT  = _root / "INPUT"
OUTPUT_DIR_DEFAULT = _root / "output"

PHASE_NAMES = [
    "p1_parse", "p2_reconstruct", "p2b_fusion",
    "p3_features", "p4_agg", "p5_sample", "p6_excursion",
    "p7_c1", "p7_c2", "p7_c3",
]

DISK_FREE_WARNING_GB  = 20.0
DISK_FREE_CRITICAL_GB = 10.0

EST_DISK_PER_DAY_BYTES = {
    "events_csv":     45 * (1024**2),
    "snapshots_csv": 427 * (1024**2),
    "features_dom":  310 * (1024**2),
    "features_agg":   50 * (1024**2),
    "sampled_events": 31 * (1024**2),
    "excursion_stats": 15 * (1024**2),
    "labels":          3 * (1024**2),
}

CANDIDATES = [
    {"vb_ticks": 2000, "pt_ticks": 10.0, "sl_ticks": 10.0, "desc": "2000t/10/10"},
    {"vb_ticks": 4000, "pt_ticks": 20.0, "sl_ticks": 20.0, "desc": "4000t/20/20"},
    {"vb_ticks": 8000, "pt_ticks": 40.0, "sl_ticks": 40.0, "desc": "8000t/40/40"},
]


# ── enums ─────────────────────────────────────────────────────────────────────

class PhaseStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    SKIPPED  = "skipped"


# ── dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class DayManifest:
    date: str
    source_file: str = ""
    trades_file: str = ""
    source_size_bytes: int = 0
    status: str = "pending"
    error_message: str = ""
    runtime_seconds: float = 0.0
    disk_bytes_written: int = 0
    phases: dict = field(default_factory=lambda: {p: PhaseStatus.PENDING.value for p in PHASE_NAMES})

    def to_row(self) -> dict:
        row = {
            "date": self.date,
            "source_file": self.source_file,
            "trades_file": self.trades_file,
            "source_size_bytes": self.source_size_bytes,
            "status": self.status,
            "error_message": self.error_message,
            "runtime_seconds": round(self.runtime_seconds, 1),
            "disk_bytes_written": self.disk_bytes_written,
        }
        row.update({f"phase_{p}": self.phases.get(p, PhaseStatus.PENDING.value) for p in PHASE_NAMES})
        return row

    @staticmethod
    def from_row(row: dict) -> "DayManifest":
        dm = DayManifest(date=row["date"])
        dm.source_file = row.get("source_file", "")
        dm.trades_file = row.get("trades_file", "")
        dm.source_size_bytes = int(row.get("source_size_bytes", 0))
        dm.status = row.get("status", "pending")
        dm.error_message = row.get("error_message", "")
        dm.runtime_seconds = float(row.get("runtime_seconds", 0))
        dm.disk_bytes_written = int(row.get("disk_bytes_written", 0))
        for p in PHASE_NAMES:
            dm.phases[p] = row.get(f"phase_{p}", PhaseStatus.PENDING.value)
        return dm


# ── checkpoint manager ─────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.days: dict[str, DayManifest] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if not self.manifest_path.exists():
            return
        with open(self.manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dm = DayManifest.from_row(row)
                self.days[dm.date] = dm

    def save(self):
        with self._lock:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            rows = [dm.to_row() for dm in self.days.values()]
            if not rows:
                return
            with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    def get(self, date: str) -> DayManifest:
        if date not in self.days:
            self.days[date] = DayManifest(date=date)
        return self.days[date]

    def is_done(self, date: str, phase: str) -> bool:
        return self.get(date).phases.get(phase) == PhaseStatus.DONE.value

    def set_phase(self, date: str, phase: str, status: PhaseStatus, error: str = ""):
        dm = self.get(date)
        dm.phases[phase] = status.value
        if status == PhaseStatus.FAILED:
            dm.status = "failed"
            dm.error_message = error
        elif status == PhaseStatus.DONE and dm.status in ("pending", "running"):
            dm.status = "complete"
        self.save()

    def finalize(self, date: str, status: str, error: str = "",
                 runtime: float = 0.0, disk_bytes: int = 0):
        dm = self.get(date)
        dm.status = status
        dm.error_message = error
        dm.runtime_seconds = runtime
        dm.disk_bytes_written = disk_bytes
        self.save()


# ── storage monitor ─────────────────────────────────────────────────────────────

class StorageMonitor:
    def get_free_bytes(self) -> int:
        try:
            usage = shutil.disk_usage("/")
            return usage.free
        except Exception:
            return 0

    def get_free_gb(self) -> float:
        return self.get_free_bytes() / (1024**3)

    def check(self, action: str = "") -> bool:
        free_gb = self.get_free_gb()
        print(f"    [Storage] Free space: {free_gb:.1f} GB")
        if free_gb < DISK_FREE_CRITICAL_GB:
            print(f"    [Storage] CRITICAL: Only {free_gb:.1f}GB — {action} SKIPPED")
            return False
        if free_gb < DISK_FREE_WARNING_GB:
            print(f"    [Storage] WARNING: Only {free_gb:.1f}GB free")
        return True

    def get_dir_size(self, d: Path) -> int:
        total = 0
        for f in d.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def delete_file(self, p: Path, reason: str = "") -> bool:
        if not p.exists():
            return True
        try:
            sz = p.stat().st_size
            p.unlink()
            print(f"    [Cleanup] Deleted {p.name} ({sz/(1024**2):.0f}MB) — {reason}")
            return True
        except OSError as e:
            print(f"    [Cleanup] WARNING: Could not delete {p}: {e}")
            return False


# ── helpers ───────────────────────────────────────────────────────────────────
def sentinel_path(out_dir: Path, phase: str) -> Path:
    return out_dir / "_checkpoints" / f"{phase}.done"


def sentinel_done(out_dir: Path, phase: str) -> bool:
    return sentinel_path(out_dir, phase).exists()


def write_sentinel(out_dir: Path, phase: str, status: str = "done", error: str = ""):
    p = sentinel_path(out_dir, phase)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = f"status={status}\ntime={dt.datetime.now().isoformat()}\n"
    if error:
        content += f"error={error}\n"
    p.write_text(content, encoding="utf-8")


def discover_days(input_dir: Path) -> list[tuple[str, Path, Path | None, int]]:
    """
    Find all trading days from .depth files.
    Returns list of (date_str, depth_path, trades_path_or_none, size).
    Looks for trades.csv alongside the .depth file.
    Skips broken symlinks (common when .depth files are symlinked from a VPS path).
    """
    results = []
    for f in sorted(input_dir.rglob("*.depth")):
        # Skip broken symlinks (target may not exist on this machine)
        try:
            size = f.stat().st_size
        except OSError:
            continue  # broken symlink or inaccessible file

        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        date_str = m.group(1) if m else f.parent.name

        # Look for trades file in same directory
        trades_path = None
        for candidate in ["trades.csv", "trades.txt"]:
            tp = f.parent / candidate
            if tp.exists():
                trades_path = tp
                break

        results.append((date_str, f, trades_path, size))
    if not results:
        raise FileNotFoundError(f"No .depth files found in {input_dir}")
    return sorted(results, key=lambda x: x[0])


def _cand_label_name(c: dict) -> str:
    pt_s = str(c["pt_ticks"]).replace(".", "p")
    sl_s = str(c["sl_ticks"]).replace(".", "p")
    return f"phase7_labels_{c['vb_ticks']}ticks_{pt_s}_{sl_s}"


# ── phase runners ─────────────────────────────────────────────────────────────

def run_phase1(date: str, depth_path: Path, out_dir: Path,
                cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    events_path = out_dir / "events.csv"
    if sentinel_done(out_dir, "p1_parse") and events_path.exists() and not force:
        print("    [P1] Already done — SKIP")
        cp.set_phase(date, "p1_parse", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p1_parse", PhaseStatus.RUNNING)
    print(f"    [P1] Parsing {depth_path.name} ...")
    t0 = time.time()
    _, header, record_count, warnings = p1_run_phase1(depth_path, events_path, force)
    elapsed = time.time() - t0

    if events_path.exists():
        write_sentinel(out_dir, "p1_parse")
        cp.set_phase(date, "p1_parse", PhaseStatus.DONE)
        sz = events_path.stat().st_size / (1024**2)
        print(f"    [P1] events.csv ({sz:.0f}MB, {record_count:,} rec) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = "events.csv not produced"
        write_sentinel(out_dir, "p1_parse", "failed", msg)
        cp.set_phase(date, "p1_parse", PhaseStatus.FAILED, msg)
        return False, msg


def run_phase2(date: str, out_dir: Path,
               cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    events_path = out_dir / "events.csv"
    snapshots_path = out_dir / "snapshots.csv"
    if not events_path.exists():
        return False, "events.csv missing"

    if sentinel_done(out_dir, "p2_reconstruct") and snapshots_path.exists() and not force:
        print("    [P2] Already done — SKIP")
        cp.set_phase(date, "p2_reconstruct", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p2_reconstruct", PhaseStatus.RUNNING)
    print(f"    [P2] Reconstructing order book ...")
    t0 = time.time()
    p2_stats = p2_reconstruct(events_path, snapshots_path)
    elapsed = time.time() - t0

    if snapshots_path.exists():
        write_sentinel(out_dir, "p2_reconstruct")
        cp.set_phase(date, "p2_reconstruct", PhaseStatus.DONE)
        sz = snapshots_path.stat().st_size / (1024**2)
        snaps = p2_stats.get("snapshots_generated", 0)
        print(f"    [P2] snapshots.csv ({sz:.0f}MB, {snaps:,} snaps) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = "snapshots.csv not produced"
        write_sentinel(out_dir, "p2_reconstruct", "failed", msg)
        cp.set_phase(date, "p2_reconstruct", PhaseStatus.FAILED, msg)
        return False, msg


def run_phase2b(date: str, trades_path: Path | None, out_dir: Path,
                cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    snapshots_path = out_dir / "snapshots.csv"
    fused_path = out_dir / "snapshots_fused.csv"
    if not snapshots_path.exists():
        return False, "snapshots.csv missing"

    if trades_path is None:
        print("    [P2b] trades.csv NOT FOUND — SKIP (P2b richiede T&S)")
        write_sentinel(out_dir, "p2b_fusion", "skipped", "no trades file")
        cp.set_phase(date, "p2b_fusion", PhaseStatus.SKIPPED, "no trades file")
        return True, ""

    if sentinel_done(out_dir, "p2b_fusion") and fused_path.exists() and not force:
        print("    [P2b] Already done — SKIP")
        cp.set_phase(date, "p2b_fusion", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p2b_fusion", PhaseStatus.RUNNING)
    print(f"    [P2b] Fusing LOB + Time & Sales (trades: {trades_path.name}) ...")
    t0 = time.time()

    import pandas as pd
    try:
        trades_df = p2b_load_trades(trades_path)
        if trades_df.empty:
            raise ValueError("trades df empty")
    except Exception as e:
        msg = f"failed to load trades: {e}"
        write_sentinel(out_dir, "p2b_fusion", "failed", msg)
        cp.set_phase(date, "p2b_fusion", PhaseStatus.FAILED, msg)
        return False, msg

    temp_path = out_dir / f".fusing_{snapshots_path.name}.tmp"
    total_snaps = 0
    total_matched = 0
    is_first = True
    CHUNK_SIZE = 250_000

    try:
        with pd.read_csv(snapshots_path, chunksize=CHUNK_SIZE, dtype=str) as reader:
            for chunk in reader:
                fused = p2b_fuse_chunk(chunk, trades_df)
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

        temp_path.replace(fused_path)
    except Exception as e:
        msg = f"fusion failed: {e}"
        if temp_path.exists():
            temp_path.unlink()
        write_sentinel(out_dir, "p2b_fusion", "failed", msg)
        cp.set_phase(date, "p2b_fusion", PhaseStatus.FAILED, msg)
        return False, msg

    elapsed = time.time() - t0
    write_sentinel(out_dir, "p2b_fusion")
    cp.set_phase(date, "p2b_fusion", PhaseStatus.DONE)
    sz = fused_path.stat().st_size / (1024**2)
    print(f"    [P2b] snapshots_fused.csv ({sz:.0f}MB, {total_matched:,} matched) in {elapsed:.0f}s — DONE")
    return True, ""


def run_phase3(date: str, out_dir: Path,
               cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    # P3 legge da snapshots_fused.csv (P2b) se disponibile, altrimenti snapshots.csv
    snapshots_path = out_dir / "snapshots_fused.csv"
    if not snapshots_path.exists():
        snapshots_path = out_dir / "snapshots.csv"
    features_path = out_dir / "features_dom.csv"

    if not snapshots_path.exists():
        return False, f"{snapshots_path.name} missing"

    if sentinel_done(out_dir, "p3_features") and features_path.exists() and not force:
        print("    [P3] Already done — SKIP")
        cp.set_phase(date, "p3_features", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p3_features", PhaseStatus.RUNNING)
    print(f"    [P3] Computing DOM features ({snapshots_path.name}) ...")
    t0 = time.time()
    p3_stats = p3_compute(snapshots_path, features_path)
    elapsed = time.time() - t0

    if features_path.exists():
        write_sentinel(out_dir, "p3_features")
        cp.set_phase(date, "p3_features", PhaseStatus.DONE)
        sz = features_path.stat().st_size / (1024**2)
        rows = p3_stats.get("rows_written", 0) if p3_stats else 0
        print(f"    [P3] features_dom.csv ({sz:.0f}MB, {rows:,} rows) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = "features_dom.csv not produced"
        write_sentinel(out_dir, "p3_features", "failed", msg)
        cp.set_phase(date, "p3_features", PhaseStatus.FAILED, msg)
        return False, msg


def run_phase4(date: str, out_dir: Path,
               cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    features_path = out_dir / "features_dom.csv"
    agg_path = out_dir / "features_dom_agg.csv"
    if not features_path.exists():
        return False, "features_dom.csv missing"

    if sentinel_done(out_dir, "p4_agg") and agg_path.exists() and not force:
        print("    [P4] Already done — SKIP")
        cp.set_phase(date, "p4_agg", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p4_agg", PhaseStatus.RUNNING)
    print(f"    [P4] Computing rolling aggregates ...")
    t0 = time.time()
    p4_stats = p4_aggregate(features_path, agg_path)
    elapsed = time.time() - t0

    if agg_path.exists():
        write_sentinel(out_dir, "p4_agg")
        cp.set_phase(date, "p4_agg", PhaseStatus.DONE)
        sz = agg_path.stat().st_size / (1024**2)
        rows = p4_stats.get("rows_written", 0) if p4_stats else 0
        print(f"    [P4] features_dom_agg.csv ({sz:.0f}MB, {rows:,} rows) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = "features_dom_agg.csv not produced"
        write_sentinel(out_dir, "p4_agg", "failed", msg)
        cp.set_phase(date, "p4_agg", PhaseStatus.FAILED, msg)
        return False, msg


def run_phase5(date: str, out_dir: Path,
               cp: CheckpointManager, force: bool = False,
               trades_path: Path | None = None) -> tuple[bool, str]:
    features_path = out_dir / "features_dom.csv"
    agg_path = out_dir / "features_dom_agg.csv"
    sampled_path = out_dir / "sampled_events.csv"
    if not features_path.exists() or not agg_path.exists():
        return False, "features or agg missing"

    if sentinel_done(out_dir, "p5_sample") and sampled_path.exists() and not force:
        print("    [P5] Already done -- SKIP")
        cp.set_phase(date, "p5_sample", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p5_sample", PhaseStatus.RUNNING)
    print(f"    [P5] CUSUM sampling ...")
    t0 = time.time()
    p5_stats = p5_sample(features_path, agg_path, sampled_path,
                         percentile=5.0, trades_path=trades_path)
    elapsed = time.time() - t0

    # Strong fail-fast: zero samples means no output and no valid signal for P6→P7.
    # Must fail immediately — downstream phases cannot proceed with empty input.
    if p5_stats.get("rows_sampled", 0) == 0:
        msg = f"[P5] 0 samples emitted (h={p5_stats.get('h_threshold', '?')}). Cannot proceed."
        write_sentinel(out_dir, "p5_sample", "failed", msg)
        cp.set_phase(date, "p5_sample", PhaseStatus.FAILED, msg)
        return False, msg

    if sampled_path.exists():
        write_sentinel(out_dir, "p5_sample")
        cp.set_phase(date, "p5_sample", PhaseStatus.DONE)
        sz = sampled_path.stat().st_size / (1024**2)
        sampled = p5_stats.get("rows_sampled", 0) if p5_stats else 0
        print(f"    [P5] sampled_events.csv ({sz:.0f}MB, {sampled:,} sampled) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = "sampled_events.csv not produced"
        write_sentinel(out_dir, "p5_sample", "failed", msg)
        cp.set_phase(date, "p5_sample", PhaseStatus.FAILED, msg)
        return False, msg


def run_phase6(date: str, out_dir: Path,
               cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    # P6 uses fingerprint-based staleness guard (Apr 13, 2026 fix).
    # Import here so the module-level functions are available.
    from P6.vps_excursion_analysis_vectorized import (
        should_skip_p6 as p6_should_skip,
        compute_input_fingerprint as p6_fingerprint,
        save_fingerprint as p6_save_fp,
        validate_p6_output as p6_validate,
    )

    sampled_path = out_dir / "sampled_events.csv"
    if not sampled_path.exists():
        return False, "sampled_events.csv missing"

    # Staleness check using new P6 fingerprint logic
    if not force:
        skip, reason = p6_should_skip(out_dir, sampled_path)
        if skip:
            print("    [P6] Already done + input unchanged -- SKIP (fingerprint valid)")
            cp.set_phase(date, "p6_excursion", PhaseStatus.DONE)
            return True, ""
        else:
            print(f"    [P6] Input changed or no fingerprint: {reason} -- re-running")

    features_path = out_dir / "features_dom.csv"
    excursion_path = out_dir / "excursion_stats.csv"
    summary_path = out_dir / "excursion_summary.csv"
    plot_path = out_dir / "excursion_distributions.png"

    if not features_path.exists():
        return False, "features_dom.csv missing (required for lookup index)"

    cp.set_phase(date, "p6_excursion", PhaseStatus.RUNNING)
    print(f"    [P6] Excursion analysis (Numba JIT) ...")
    t0 = time.time()

    print("  Building lookup index from features_dom.csv...")
    ts_ns, mp = p6_build_index(features_path)
    print(f"  -> {len(ts_ns):,} entries indexed")

    print("  Computing excursions...")
    p6_stats = p6_compute(sampled_path, ts_ns, mp, excursion_path)
    print(f"  -> {p6_stats.get('rows_processed', 0):,} rows processed")

    print("  Validating output ratio...")
    p6_validate(sampled_path, excursion_path, date)

    print("  Saving fingerprint...")
    fp = p6_fingerprint(sampled_path)
    p6_save_fp(fp, out_dir)
    print(f"  -> fingerprint: {fp['fingerprint_ts']}")

    print("  Generating summary...")
    p6_summary(excursion_path, summary_path)

    print("  Plotting distributions...")
    p6_plot(excursion_path, plot_path)

    elapsed = time.time() - t0
    if excursion_path.exists():
        write_sentinel(out_dir, "p6_excursion")
        cp.set_phase(date, "p6_excursion", PhaseStatus.DONE)
        sz = excursion_path.stat().st_size / (1024**2)
        print(f"    [P6] excursion_stats.csv ({sz:.0f}MB) in {elapsed:.0f}s -- DONE")
        return True, ""
    else:
        msg = "excursion_stats.csv not produced"
        write_sentinel(out_dir, "p6_excursion", "failed", msg)
        cp.set_phase(date, "p6_excursion", PhaseStatus.FAILED, msg)
        return False, msg


def run_phase7(date: str, out_dir: Path,
               cp: CheckpointManager, storage: StorageMonitor,
               force: bool = False) -> tuple[bool, str]:
    sampled_path = out_dir / "sampled_events.csv"
    snapshots_path = out_dir / "snapshots.csv"
    excursion_path = out_dir / "excursion_stats.csv"

    if not sampled_path.exists():
        return False, "sampled_events.csv missing"
    if not snapshots_path.exists():
        return False, "snapshots.csv missing"
    if not excursion_path.exists():
        return False, "excursion_stats.csv missing"

    # Build 3-candidate grid file
    grid_path = out_dir / "_candidates_3.csv"
    with open(grid_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vb_ticks", "pt_ticks", "sl_ticks"])
        w.writeheader()
        for c in CANDIDATES:
            w.writerow({"vb_ticks": c["vb_ticks"],
                       "pt_ticks": c["pt_ticks"], "sl_ticks": c["sl_ticks"]})

    labels_ok = 0
    for i, c in enumerate(CANDIDATES, 1):
        label_name = _cand_label_name(c)
        label_path = out_dir / label_name
        p7_phase = f"p7_c{i}"

        if sentinel_done(out_dir, p7_phase) and label_path.exists() and not force:
            print(f"    [P7-{i}/3] {c['desc']} — SKIP (exists)")
            labels_ok += 1
            continue

        cp.set_phase(date, p7_phase, PhaseStatus.RUNNING)
        print(f"    [P7-{i}/3] {c['desc']} ...")
        t0 = time.time()
        cmd = [
            sys.executable,
            str(_root / "P7" / "vps_phase7_labeling.py"),
            "--snapshots", str(snapshots_path),
            "--sampled", str(sampled_path),
            "--refprice", str(excursion_path),
            "--grid", str(grid_path),
            "--output", str(out_dir),
            "--candidates", str(i),
        ]
        if force:
            cmd.append("--force")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - t0

        if r.returncode == 0 and label_path.exists():
            write_sentinel(out_dir, p7_phase)
            cp.set_phase(date, p7_phase, PhaseStatus.DONE)
            sz = label_path.stat().st_size / (1024**2)
            print(f"    [P7-{i}/3] {c['desc']} ({sz:.1f}MB) in {elapsed:.0f}s — DONE")
            labels_ok += 1
        else:
            msg = r.stderr[:200] if r.stderr else "label file not produced"
            write_sentinel(out_dir, p7_phase, "failed", msg)
            cp.set_phase(date, p7_phase, PhaseStatus.FAILED, msg)
            print(f"    [P7-{i}/3] {c['desc']} — FAILED: {msg}")

    all_ok = labels_ok == 3
    if all_ok:
        write_sentinel(out_dir, "p7_labels")
    return all_ok, "" if all_ok else f"Only {labels_ok}/3 candidates labeled"


# ── per-day pipeline ───────────────────────────────────────────────────────────

def process_day(
    date: str,
    depth_path: Path,
    trades_path: Path | None,
    out_dir: Path,
    cp: CheckpointManager,
    storage: StorageMonitor,
    force: bool = False,
) -> tuple[str, str, float, int]:
    """Process one trading day through P1→P7. Returns (status, error, runtime, disk)."""
    t0 = time.time()
    dm = cp.get(date)
    dm.source_file = str(depth_path)
    dm.trades_file = str(trades_path) if trades_path else ""
    dm.source_size_bytes = depth_path.stat().st_size
    dm.status = "running"
    cp.save()

    disk_at_start = storage.get_dir_size(out_dir)

    def disk_now() -> int:
        return storage.get_dir_size(out_dir)

    try:
        ok, err = run_phase1(date, depth_path, out_dir, cp, force)
        if not ok:
            cp.finalize(date, "failed", err, time.time()-t0, 0)
            return "failed", err, time.time()-t0, 0

        ok, err = run_phase2(date, out_dir, cp, force)
        if not ok:
            cp.finalize(date, "failed", err, time.time()-t0, 0)
            return "failed", err, time.time()-t0, 0

        ok, err = run_phase2b(date, trades_path, out_dir, cp, force)
        if not ok and "missing" not in err.lower():
            cp.finalize(date, "failed", err, time.time()-t0, 0)
            return "failed", err, time.time()-t0, 0

        ok, err = run_phase3(date, out_dir, cp, force)
        if not ok:
            cp.finalize(date, "failed", err, time.time()-t0, 0)
            return "failed", err, time.time()-t0, 0
        gc.collect()

        ok, err = run_phase4(date, out_dir, cp, force)
        if not ok:
            cp.finalize(date, "failed", err, time.time()-t0, 0)
            return "failed", err, time.time()-t0, 0
        gc.collect()

        ok, err = run_phase5(date, out_dir, cp, force, trades_path)
        if not ok:
            cp.finalize(date, "failed", err, time.time()-t0, 0)
            return "failed", err, time.time()-t0, 0
        gc.collect()

        ok, err = run_phase6(date, out_dir, cp, force)
        if not ok:
            cp.finalize(date, "failed", err, time.time()-t0, 0)
            return "failed", err, time.time()-t0, 0
        gc.collect()   # CRITICAL: ensures P6's 3.1GB sort arrays are freed before P7 subprocess allocates ~2.2GB

        ok, err = run_phase7(date, out_dir, cp, storage, force)
        if not ok:
            cp.finalize(date, "failed", err, time.time()-t0, 0)
            return "failed", err, time.time()-t0, 0

        # Free snapshots.csv after P7 completes (both P6 and P7 need it during execution)
        snaps = out_dir / "snapshots.csv"
        if snaps.exists():
            freed = snaps.stat().st_size / 1024**2
            snaps.unlink()
            print(f"    [GC] Deleted snapshots.csv ({freed:.0f}MB freed)")
        gc.collect()

        runtime = time.time() - t0
        disk_final = disk_now()
        disk_written = disk_final - disk_at_start
        cp.finalize(date, "complete", "", runtime, disk_written)
        return "complete", "", runtime, disk_written

    except Exception as e:
        cp.finalize(date, "exception", str(e)[:300], time.time()-t0, 0)
        return "exception", str(e)[:300], time.time()-t0, 0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="P1→P7 (inclusive P2b) Multi-Day Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_p1_to_p7_multiday.py --dry-run
  python3 run_p1_to_p7_multiday.py --resume --workers 1
  python3 run_p1_to_p7_multiday.py --force --max-days 3
        """
    )
    ap.add_argument("--input-dir",  type=Path, default=INPUT_DIR_DEFAULT)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    ap.add_argument("--workers",   type=int, default=1,
                    help="Parallel workers (default: 1 = sequential, RAM-safe)")
    ap.add_argument("--resume",     action="store_true",
                    help="Skip days with P7 labels already done")
    ap.add_argument("--force",      action="store_true",
                    help="Force re-run all phases (ignore checkpoints)")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Show days to process without running")
    ap.add_argument("--max-days",  type=int, default=0,
                    help="Limit number of days (default: all)")

    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = StorageMonitor()

    print(f"\n{'='*70}")
    print("P1→P7 MULTI-DAY PIPELINE (P2b INCLUSO)")
    print(f"{'='*70}")
    print(f"  Input dir  : {input_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Workers    : {args.workers} {'(sequential, RAM-safe)' if args.workers == 1 else f'({args.workers} parallel)'}")
    print(f"  Resume     : {args.resume}")
    print(f"  Force      : {args.force}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"{'='*70}\n")

    # ── Discover days ─────────────────────────────────────────────────────────
    print(f"[Discovery] Scanning {input_dir} for .depth files ...")
    try:
        days_info = discover_days(input_dir)
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    if not days_info:
        print(f"ERROR: No .depth files found in {input_dir}")
        sys.exit(1)

    print(f"  Found {len(days_info)} day(s):")
    for date_str, depth_path, trades_path, size in days_info:
        trades_flag = f" [T&S: {trades_path.name}]" if trades_path else " [NO T&S — P2b skip]"
        print(f"    {date_str}: {depth_path.name} ({size/(1024**2):.0f}MB){trades_flag}")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    manifest_path = output_dir / "_p1p7_manifest.csv"
    cp = CheckpointManager(manifest_path)
    for date_str, depth_path, trades_path, size in days_info:
        dm = cp.get(date_str)
        if not dm.source_file:
            dm.source_file = str(depth_path)
            dm.trades_file = str(trades_path) if trades_path else ""
            dm.source_size_bytes = size
    cp.save()

    # ── Filter days ───────────────────────────────────────────────────────────
    if args.resume:
        # Skip days with all 3 P7 labels done
        days = [
            d for d, dp, tp, sz in days_info
            if not all(
                sentinel_done(output_dir / d, f"p7_c{i}")
                for i in range(1, 4)
            )
        ]
        skipped = len(days_info) - len(days)
        print(f"\n[Resume] {skipped} giorni già completi (P7 done), {len(days)} da processare")
    else:
        days = [d for d, _, _, _ in days_info]

    if args.max_days > 0:
        days = days[:args.max_days]
        print(f"\n[max-days] Limitato a {args.max_days} giorni")

    if args.dry_run:
        print(f"\n[DRY RUN] Giorni che verrebbero processati ({len(days)}):")
        for d in days:
            print(f"  {d}")
        return

    if not days:
        print("\nNessun giorno da processare.")
        return

    # ── Storage check ─────────────────────────────────────────────────────────
    free_gb = storage.get_free_gb()
    print(f"\n[Storage] Free space: {free_gb:.1f} GB")

    # ── Run ──────────────────────────────────────────────────────────────────
    print(f"\n[Processing] {len(days)} giorno(i) ...")
    results = []

    for i, date_str in enumerate(days, 1):
        # find day info
        day_info = next((di for di in days_info if di[0] == date_str), None)
        if not day_info:
            continue
        _, depth_path, trades_path, size = day_info
        day_out = output_dir / date_str
        day_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(days)}] {'='*50}")
        print(f"  Date   : {date_str}")
        print(f"  Source : {depth_path.name} ({size/(1024**2):.0f}MB)")
        if trades_path:
            print(f"  T&S    : {trades_path.name}")

        if not storage.check(f"starting {date_str}"):
            results.append((date_str, "skipped_no_space", "Low disk", 0, 0))
            continue

        status, err, runtime, disk_bytes = process_day(
            date=date_str,
            depth_path=depth_path,
            trades_path=trades_path,
            out_dir=day_out,
            cp=cp,
            storage=storage,
            force=args.force,
        )
        results.append((date_str, status, err, runtime, disk_bytes))

        icon = {"complete": "✅", "partial": "⚠️", "failed": "❌",
                "exception": "💥", "skipped_no_space": "⛔"}.get(status, "?")
        print(f"\n  {icon} {date_str}: {status} | {runtime/60:.1f}min | "
              f"{disk_bytes/(1024**3):.3f}GB")
        if err:
            print(f"     Error: {err[:200]}")

        done_count = sum(1 for d, _, _, _, _ in results if d in
                         [di[0] for di in days_info] and
                         cp.get(d).status == "complete")
        print(f"[Progress] {done_count}/{len(days)} completi | "
              f"Free: {storage.get_free_gb():.1f}GB")

        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    success = sum(1 for _, s, _, _, _ in results if s == "complete")
    partial = sum(1 for _, s, _, _, _ in results if s == "partial")
    failed  = sum(1 for _, s, _, _, _ in results if s in ("failed", "exception"))
    skipped = sum(1 for _, s, _, _, _ in results if s == "skipped_no_space")

    print(f"\n{'='*70}")
    print("P1→P7 MULTI-DAY RUN COMPLETE")
    print(f"{'='*70}")
    print(f"  Complete: {success}")
    print(f"  Partial: {partial}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped} (no space)")
    print(f"  Free space (end): {storage.get_free_gb():.1f} GB")
    print(f"  Manifest → {manifest_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
