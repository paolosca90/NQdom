#!/usr/bin/env python3
"""
vps_multiday_runner.py — VPS-Only Multi-Day DOM Pipeline Orchestrator
=====================================================================
Runs ENTIRELY on the VPS. No local computation for processing.
All discovery, processing, checkpointing, logging, aggregation, and storage
management happen on the VPS filesystem.

DESIGN PRINCIPLES
    - Sequential day-by-day (RAM-safe, no parallelism by default)
    - Phase-level checkpointing via sentinel files + manifest CSV
    - Storage-aware: after each day, optionally delete intermediates
    - Failure-tolerant: one day failure does NOT kill the run
    - Idempotent: re-runnable without recomputing completed steps

PHASE EXECUTION MATRIX (each phase is CLI-callable on VPS):
    P1  → main.py --days {date}                    (depth → events)
    P2  → python3 -c "from book_reconstructor import ...; ..." (events → snapshots)
    P3  → python3 feature_engineering.py --input ... --output ...   (snapshots → features)
    P4  → python3 feature_engineering_agg.py --input ... --output ... (features → agg)
    P5  → python3 cusum_sampler.py --features ... --agg ... --output ...  (→ sampled)
    P6  → python3 excursion_analysis.py --features ... --sampled ...   (→ excursions)
    P7  → python3 phase7_labeling.py --snapshots ... --sampled ... --refprice ... --grid ... --candidates N
    P8  → python3 phase8_entry_model.py --features ... --output ...

CHECKPOINT STRATEGY
    Per-day sentinel files: output/{date}/_checkpoints/p{N}_{phase}.done
    Global manifest:       output/_multiday_manifest.csv

STORAGE LIFECYCLE
    snapshots_csv:  READY_TO_DELETE after P7+P8 done (~427MB/day savings)
    features_dom:  REGENERABLE      after P5 done (~310MB/day)
    features_agg:  REGENERABLE      after P5 done (~50MB/day)
    events_csv:    REGENERABLE      always (~45MB/day)
    Non-negotiable: sampled_events, excursion_stats, labels, phase8_outputs

USAGE
    python3 vps_multiday_runner.py \\
        --input-dir  /opt/depth-dom/input \\
        --output-dir /opt/depth-dom/output \\
        --cleanup-policy archive-ready \\
        --parallel 1 \\
        --resume
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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from _pipeline_constants import CANDIDATES
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# FileLock disabled: ThreadPoolExecutor + threading.Lock avoids pickling issues
# that arise with ProcessPoolExecutor + filelock.FileLock (not picklable).
HAS_FILELOCK = False

# ── Constants ─────────────────────────────────────────────────────────────────

VPS_BASE           = "/opt/depth-dom"
INPUT_DIR_DEFAULT = "/opt/depth-dom/input"
OUTPUT_DIR_DEFAULT= "/opt/depth-dom/output"

# Disk safety thresholds
DISK_FREE_WARNING_GB   = 20.0   # warn if < 20 GB free
DISK_FREE_CRITICAL_GB  = 10.0   # STOP if < 10 GB free

# Estimated per-day disk consumption
EST_DISK_PER_DAY_BYTES = {
    "events_csv":     45 * (1024**2),
    "snapshots_csv": 427 * (1024**2),
    "features_dom":  310 * (1024**2),
    "features_agg":   50 * (1024**2),
    "sampled_events": 31 * (1024**2),
    "excursion_stats":15 * (1024**2),
    "labels":          3 * (1024**2),
    "phase8_outputs": 10 * (1024**2),
}

PHASE_NAMES = [
    "p1_parse", "p2_reconstruct", "p2b_fusion", "p3_features", "p4_agg",
    "p5_sample", "p6_excursion",
    "p7_c1", "p7_c2", "p7_c3",
    "p7b_macro",
    "p8_ml",
]


# ── Enums ─────────────────────────────────────────────────────────────────────

class PhaseStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    SKIPPED  = "skipped"


class CleanupPolicy(str, Enum):
    NONE          = "none"
    ARCHIVE_READY = "archive-ready"  # delete snapshots.csv after P7+P8 done
    AGGRESSIVE    = "aggressive"     # also delete features_dom + features_agg


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class DayManifest:
    date: str
    source_file: str = ""
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
        dm.source_size_bytes = int(row.get("source_size_bytes", 0))
        dm.status = row.get("status", "pending")
        dm.error_message = row.get("error_message", "")
        dm.runtime_seconds = float(row.get("runtime_seconds", 0))
        dm.disk_bytes_written = int(row.get("disk_bytes_written", 0))
        for p in PHASE_NAMES:
            dm.phases[p] = row.get(f"phase_{p}", PhaseStatus.PENDING.value)
        return dm


# ── Checkpoint Manager ──────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.days: dict[str, DayManifest] = {}
        # PERF: FileLock provides cross-process protection vs threading.Lock (single-process only)
        if HAS_FILELOCK:
            self._filelock = FileLock(str(manifest_path.with_suffix(".lock")), timeout=30)
        else:
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
        # PERF: use FileLock for cross-process safety, fallback to threading.Lock
        manifest_lock = self._filelock if HAS_FILELOCK else self._lock
        with manifest_lock:
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


# ── Storage Monitor ─────────────────────────────────────────────────────────────

class StorageMonitor:
    def __init__(self, path: Path):
        self.path = path

    def get_free_bytes(self) -> int:
        try:
            stat = os.statvfs(self.path)
            return stat.f_bavail * stat.f_frsize
        except (AttributeError, OSError):
            try:
                import shutil
                usage = shutil.disk_usage(self.path)
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

    def estimate_remaining(self, n_days: int) -> dict:
        total = sum(EST_DISK_PER_DAY_BYTES.values()) * n_days
        free = self.get_free_bytes()
        return {
            "n_days": n_days,
            "needed_bytes": total,
            "needed_gb": total / (1024**3),
            "free_gb": free / (1024**3),
            "projected_gb": (free - total) / (1024**3),
            "safe": free > total + DISK_FREE_WARNING_GB * (1024**3),
        }

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

    def get_dir_size(self, d: Path) -> int:
        total = 0
        for f in d.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total


# ── Command Runner ─────────────────────────────────────────────────────────────

def run_cmd(cmd: str, timeout: int = 3600, cwd: str = VPS_BASE) -> tuple[str, str, int, float]:
    """Run shell cmd on VPS, return (stdout, stderr, exit_code, elapsed_s)."""
    start = time.time()
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd,
                          capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr, r.returncode, time.time() - start
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if e.stdout else ""
        err = e.stderr.decode() if e.stderr else "TIMEOUT"
        return out, err, -1, time.time() - start


# ── File Discovery ─────────────────────────────────────────────────────────────

def discover_depth_files(input_dir: Path) -> list[tuple[Path, str, int]]:
    """Find all .depth files recursively, extract dates from filename."""
    results = []
    for f in sorted(input_dir.rglob("*.depth")):
        fname = f.name
        m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
        date_str = m.group(1) if m else f.parent.name
        results.append((f, date_str, f.stat().st_size))
    return sorted(results, key=lambda x: x[1])


# ── Phase Check Helpers ────────────────────────────────────────────────────────

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


# ── Phase 1: Parse depth file ──────────────────────────────────────────────────

def run_phase1(date: str, depth_path: Path, out_dir: Path,
              cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    """Parse .depth → events.csv using main.py discovery."""
    if sentinel_done(out_dir, "p1_parse") and not force:
        print("    [P1] Already done — SKIP")
        return True, ""

    events_path = out_dir / "events.csv"
    if events_path.exists() and not force:
        print("    [P1] events.csv exists — SKIP")
        write_sentinel(out_dir, "p1_parse", "done")
        cp.set_phase(date, "p1_parse", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p1_parse", PhaseStatus.RUNNING)
    print(f"    [P1] Parsing {depth_path.name} ...")
    cmd = f"python3 main.py --days {date}{' --force' if force else ''} 2>&1"
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=7200, cwd=VPS_BASE)

    if code == 0 and events_path.exists():
        write_sentinel(out_dir, "p1_parse")
        cp.set_phase(date, "p1_parse", PhaseStatus.DONE)
        sz = events_path.stat().st_size / (1024**2)
        print(f"    [P1] events.csv ({sz:.0f}MB) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = stderr[:300] if stderr else "events.csv not produced"
        write_sentinel(out_dir, "p1_parse", "failed", msg)
        cp.set_phase(date, "p1_parse", PhaseStatus.FAILED, msg)
        return False, msg


# ── Phase 2: Book Reconstruct ─────────────────────────────────────────────────

def run_phase2(date: str, out_dir: Path,
              cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    """Reconstruct order book: events.csv → snapshots.csv."""
    if sentinel_done(out_dir, "p2_reconstruct") and not force:
        print("    [P2] Already done — SKIP")
        return True, ""

    events_path = out_dir / "events.csv"
    snapshots_path = out_dir / "snapshots.csv"
    if not events_path.exists():
        return False, "events.csv missing"

    if snapshots_path.exists() and not force:
        print("    [P2] snapshots.csv exists — SKIP")
        write_sentinel(out_dir, "p2_reconstruct")
        cp.set_phase(date, "p2_reconstruct", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p2_reconstruct", PhaseStatus.RUNNING)
    print(f"    [P2] Reconstructing order book ...")
    # Use triple-quoted heredoc to avoid all shell/Python quote conflicts
    cmd = (
        f"python3 - <<'PYEOF'\n"
        f"from pathlib import Path\n"
        f"from book_reconstructor import reconstruct\n"
        f"reconstruct(Path('{events_path}'), Path('{snapshots_path}'))\n"
        f"PYEOF\n"
    )
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=10800, cwd=VPS_BASE)

    if code == 0 and snapshots_path.exists():
        write_sentinel(out_dir, "p2_reconstruct")
        cp.set_phase(date, "p2_reconstruct", PhaseStatus.DONE)
        sz = snapshots_path.stat().st_size / (1024**2)
        print(f"    [P2] snapshots.csv ({sz:.0f}MB) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = stderr[:300] if stderr else "snapshots.csv not produced"
        write_sentinel(out_dir, "p2_reconstruct", "failed", msg)
        cp.set_phase(date, "p2_reconstruct", PhaseStatus.FAILED, msg)
        return False, msg


# ── Phase 3: Feature Engineering ───────────────────────────────────────────────

def run_phase3(date: str, out_dir: Path,
              cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    """Compute DOM features: snapshots.csv → features_dom.csv."""
    if sentinel_done(out_dir, "p3_features") and not force:
        print("    [P3] Already done — SKIP")
        return True, ""

    snapshots_path = out_dir / "snapshots.csv"
    features_path = out_dir / "features_dom.csv"
    if not snapshots_path.exists():
        return False, "snapshots.csv missing"

    if features_path.exists() and not force:
        print("    [P3] features_dom.csv exists — SKIP")
        write_sentinel(out_dir, "p3_features")
        cp.set_phase(date, "p3_features", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p3_features", PhaseStatus.RUNNING)
    print(f"    [P3] Computing DOM features ...")
    cmd = (
        f"python3 feature_engineering.py "
        f"--input {snapshots_path} "
        f"--output {features_path} "
        f"{'--force' if force else ''} "
        f"2>&1"
    )
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=7200, cwd=VPS_BASE)

    if code == 0 and features_path.exists():
        write_sentinel(out_dir, "p3_features")
        cp.set_phase(date, "p3_features", PhaseStatus.DONE)
        sz = features_path.stat().st_size / (1024**2)
        print(f"    [P3] features_dom.csv ({sz:.0f}MB) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = stderr[:300] if stderr else "features_dom.csv not produced"
        write_sentinel(out_dir, "p3_features", "failed", msg)
        cp.set_phase(date, "p3_features", PhaseStatus.FAILED, msg)
        return False, msg


# ── Phase 4: Aggregation ─────────────────────────────────────────────────────

def run_phase4(date: str, out_dir: Path,
              cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    """Temporal aggregation: features_dom.csv → features_dom_agg.csv."""
    if sentinel_done(out_dir, "p4_agg") and not force:
        print("    [P4] Already done — SKIP")
        return True, ""

    features_path = out_dir / "features_dom.csv"
    agg_path = out_dir / "features_dom_agg.csv"
    if not features_path.exists():
        return False, "features_dom.csv missing"

    if agg_path.exists() and not force:
        print("    [P4] features_dom_agg.csv exists — SKIP")
        write_sentinel(out_dir, "p4_agg")
        cp.set_phase(date, "p4_agg", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p4_agg", PhaseStatus.RUNNING)
    print(f"    [P4] Computing rolling aggregates ...")
    cmd = (
        f"python3 feature_engineering_agg.py "
        f"--input {features_path} "
        f"--output {agg_path} "
        f"{'--force' if force else ''} "
        f"2>&1"
    )
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=5400, cwd=VPS_BASE)

    if code == 0 and agg_path.exists():
        write_sentinel(out_dir, "p4_agg")
        cp.set_phase(date, "p4_agg", PhaseStatus.DONE)
        sz = agg_path.stat().st_size / (1024**2)
        print(f"    [P4] features_dom_agg.csv ({sz:.0f}MB) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = stderr[:300] if stderr else "features_dom_agg.csv not produced"
        write_sentinel(out_dir, "p4_agg", "failed", msg)
        cp.set_phase(date, "p4_agg", PhaseStatus.FAILED, msg)
        return False, msg


# ── Phase 5: CUSUM Sampling ──────────────────────────────────────────────────

def run_phase5(date: str, out_dir: Path,
              cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    """CUSUM sampling: features_dom.csv + features_dom_agg.csv → sampled_events.csv."""
    if sentinel_done(out_dir, "p5_sample") and not force:
        print("    [P5] Already done — SKIP")
        return True, ""

    features_path = out_dir / "features_dom.csv"
    agg_path = out_dir / "features_dom_agg.csv"
    sampled_path = out_dir / "sampled_events.csv"
    if not features_path.exists() or not agg_path.exists():
        return False, "features or agg missing"

    if sampled_path.exists() and not force:
        print("    [P5] sampled_events.csv exists — SKIP")
        write_sentinel(out_dir, "p5_sample")
        cp.set_phase(date, "p5_sample", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p5_sample", PhaseStatus.RUNNING)
    print(f"    [P5] CUSUM sampling ...")
    cmd = (
        f"python3 cusum_sampler.py "
        f"--features {features_path} "
        f"--agg {agg_path} "
        f"--output {sampled_path} "
        f"{'--force' if force else ''} "
        f"2>&1"
    )
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=3600, cwd=VPS_BASE)

    if code == 0 and sampled_path.exists():
        write_sentinel(out_dir, "p5_sample")
        cp.set_phase(date, "p5_sample", PhaseStatus.DONE)
        sz = sampled_path.stat().st_size / (1024**2)
        print(f"    [P5] sampled_events.csv ({sz:.0f}MB) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = stderr[:300] if stderr else "sampled_events.csv not produced"
        write_sentinel(out_dir, "p5_sample", "failed", msg)
        cp.set_phase(date, "p5_sample", PhaseStatus.FAILED, msg)
        return False, msg


# ── Phase 6: Excursion Analysis ───────────────────────────────────────────────

def run_phase6(date: str, out_dir: Path,
             cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    """Excursion analysis: snapshots + sampled → excursion_stats.csv.

    NOTE: The optimized P6 (Numba JIT) reads mid_price from snapshots.csv
    (not features_dom.csv). --features points to features_dom.csv but the
    script uses its parent directory to locate snapshots.csv automatically.
    """
    if sentinel_done(out_dir, "p6_excursion") and not force:
        print("    [P6] Already done — SKIP")
        return True, ""

    features_path = out_dir / "features_dom.csv"
    snapshots_path = out_dir / "snapshots.csv"
    sampled_path = out_dir / "sampled_events.csv"
    excursion_path = out_dir / "excursion_stats.csv"
    summary_path = out_dir / "excursion_summary.csv"
    plot_path = out_dir / "excursion_distributions.png"

    if not sampled_path.exists():
        return False, "sampled_events.csv missing"
    if not snapshots_path.exists():
        return False, "snapshots.csv missing (required by P6 for mid_price lookup)"

    if excursion_path.exists() and not force:
        print("    [P6] excursion_stats.csv exists — SKIP")
        write_sentinel(out_dir, "p6_excursion")
        cp.set_phase(date, "p6_excursion", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p6_excursion", PhaseStatus.RUNNING)
    print(f"    [P6] Excursion analysis (Numba JIT) ...")
    cmd = (
        f"python3 excursion_analysis.py "
        f"--features {features_path} "
        f"--sampled {sampled_path} "
        f"--output {excursion_path} "
        f"--summary {summary_path} "
        f"--plot {plot_path} "
        f"2>&1"
    )
    # Timeout: Numba JIT version finishes in ~5-8 min per day (was 73h+ before)
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=1200, cwd=VPS_BASE)

    if code == 0 and excursion_path.exists():
        write_sentinel(out_dir, "p6_excursion")
        cp.set_phase(date, "p6_excursion", PhaseStatus.DONE)
        sz = excursion_path.stat().st_size / (1024**2)
        print(f"    [P6] excursion_stats.csv ({sz:.0f}MB) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = stderr[:300] if stderr else "excursion_stats.csv not produced"
        write_sentinel(out_dir, "p6_excursion", "failed", msg)
        cp.set_phase(date, "p6_excursion", PhaseStatus.FAILED, msg)
        return False, msg


# ── Phase 7: Labeling (3 candidates) ─────────────────────────────────────────

def _cand_label_name(c: dict) -> str:
    """Build Phase 7 label directory name using canonical vb_ticks/pt_ticks/sl_ticks keys."""
    pt_s = str(c["pt_ticks"]).replace(".", "p")
    sl_s = str(c["sl_ticks"]).replace(".", "p")
    return f"phase7_labels_{c['vb_ticks']}ticks_{pt_s}_{sl_s}"


def run_phase7(date: str, out_dir: Path,
              cp: CheckpointManager,
              storage: StorageMonitor,
              force: bool = False) -> tuple[bool, str]:
    """First-touch labeling for 3 candidates."""
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

        # Skip if already done (sentinel) AND label file exists AND not forcing
        if sentinel_done(out_dir, p7_phase) and label_path.exists() and not force:
            print(f"    [P7-{i}/3] {c['desc']} — SKIP (exists)")
            labels_ok += 1
            continue

        cp.set_phase(date, p7_phase, PhaseStatus.RUNNING)
        print(f"    [P7-{i}/3] {c['desc']} ...")
        cmd = (
            f"python3 phase7_labeling.py "
            f"--snapshots {snapshots_path} "
            f"--sampled {sampled_path} "
            f"--refprice {excursion_path} "
            f"--grid {grid_path} "
            f"--output {out_dir} "
            f"--candidates {i} "
            f"{'--force' if force else ''} "
            f"2>&1"
        )
        stdout, stderr, code, elapsed = run_cmd(cmd, timeout=3600, cwd=VPS_BASE)

        if code == 0 and label_path.exists():
            write_sentinel(out_dir, p7_phase)
            cp.set_phase(date, p7_phase, PhaseStatus.DONE)
            sz = label_path.stat().st_size / (1024**2)
            print(f"    [P7-{i}/3] {c['desc']} ({sz:.1f}MB) in {elapsed:.0f}s — DONE")
            labels_ok += 1
        else:
            msg = stderr[:200] if stderr else "label file not produced"
            write_sentinel(out_dir, p7_phase, "failed", msg)
            cp.set_phase(date, p7_phase, PhaseStatus.FAILED, msg)
            print(f"    [P7-{i}/3] {c['desc']} — FAILED: {msg}")

    all_ok = labels_ok == 3
    if all_ok:
        write_sentinel(out_dir, "p7_labels")
    return all_ok, "" if all_ok else f"Only {labels_ok}/3 candidates labeled"


# ── Phase 7b: Macro Filter ───────────────────────────────────────────────

def run_phase7b(date: str, out_dir: Path,
                cp: CheckpointManager, force: bool = False) -> tuple[bool, str]:
    """Macro filter: removes infected Beta-Surprise zones and bifurcates GEX."""
    if sentinel_done(out_dir, "p7b_macro") and not force:
        print("    [P7b] Already done — SKIP")
        return True, ""

    sampled_path = out_dir / "sampled_events.csv"
    if not sampled_path.exists():
        return False, "sampled_events.csv missing"

    cp.set_phase(date, "p7b_macro", PhaseStatus.RUNNING)
    print(f"    [P7b] Applicazione filtri macroeconomici (GEX/Beta) ...")
    
    macro_script = Path(VPS_BASE) / "OPTIMIZATIONS" / "vps_phase7b_macro_filter.py"
    cmd = (
        f"python3 {macro_script} "
        f"--input {sampled_path} "
        f"--output-dir {out_dir} "
        f"2>&1"
    )
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=3600, cwd=VPS_BASE)

    if code == 0:
        write_sentinel(out_dir, "p7b_macro")
        cp.set_phase(date, "p7b_macro", PhaseStatus.DONE)
        print(f"    [P7b] Filtraggio macro terminato in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = stderr[:300] if stderr else "Errore Phase 7b script"
        write_sentinel(out_dir, "p7b_macro", "failed", msg)
        cp.set_phase(date, "p7b_macro", PhaseStatus.FAILED, msg)
        return False, msg


# ── Phase 8: Entry Model ──────────────────────────────────────────────────────

def run_phase8(date: str, out_dir: Path,
              cp: CheckpointManager,
              force: bool = False) -> tuple[bool, str]:
    """ML entry model training logic on target regime DataFrame."""
    if (out_dir / ".macro_dropped").exists() and not force:
        print("    [P8] Giornata scartata dai filtri Macro (P7b) — SKIP")
        write_sentinel(out_dir, "p8_ml", "skipped", "Macro dropped")
        cp.set_phase(date, "p8_ml", PhaseStatus.SKIPPED, "Macro Dropped")
        return True, ""

    sampled_pos = out_dir / "sampled_events_gex_pos.csv"
    sampled_neg = out_dir / "sampled_events_gex_neg.csv"

    if sampled_pos.exists():
        sampled_path = sampled_pos
    elif sampled_neg.exists():
        sampled_path = sampled_neg
    else:
        sampled_path = out_dir / "sampled_events.csv"

    if not sampled_path.exists():
        return False, f"{sampled_path.name} missing"

    p8_marker = out_dir / "phase8_trainval_results.csv"
    if sentinel_done(out_dir, "p8_ml") and not force and p8_marker.exists():
        print("    [P8] Already done — SKIP")
        return True, ""

    if p8_marker.exists() and not force:
        print("    [P8] phase8_trainval_results.csv exists — SKIP")
        write_sentinel(out_dir, "p8_ml")
        cp.set_phase(date, "p8_ml", PhaseStatus.DONE)
        return True, ""

    cp.set_phase(date, "p8_ml", PhaseStatus.RUNNING)
    print(f"    [P8] ML entry model training ...")
    cmd = (
        f"python3 phase8_entry_model.py "
        f"--features {sampled_path} "
        f"--output {out_dir} "
        f"{'--force' if force else ''} "
        f"2>&1"
    )
    stdout, stderr, code, elapsed = run_cmd(cmd, timeout=3600, cwd=VPS_BASE)

    if code == 0 and p8_marker.exists():
        write_sentinel(out_dir, "p8_ml")
        cp.set_phase(date, "p8_ml", PhaseStatus.DONE)
        sz = p8_marker.stat().st_size / (1024**2)
        print(f"    [P8] phase8_trainval_results.csv ({sz:.1f}MB) in {elapsed:.0f}s — DONE")
        return True, ""
    else:
        msg = stderr[:300] if stderr else "phase8 outputs not produced"
        write_sentinel(out_dir, "p8_ml", "failed", msg)
        cp.set_phase(date, "p8_ml", PhaseStatus.FAILED, msg)
        return False, msg


# ── Per-day Cleanup ────────────────────────────────────────────────────────────

def cleanup_day(date: str, out_dir: Path,
               policy: CleanupPolicy,
               storage: StorageMonitor,
               dry_run: bool = False) -> dict:
    """
    Apply cleanup policy after day completes.
    Returns dict of actions taken.
    """
    actions = {}
    if policy == CleanupPolicy.NONE:
        return actions

    to_delete: list[tuple[str, Path]] = []

    snapshots = out_dir / "snapshots.csv"
    features_dom = out_dir / "features_dom.csv"
    features_agg = out_dir / "features_dom_agg.csv"

    if snapshots.exists():
        if policy == CleanupPolicy.ARCHIVE_READY:
            to_delete.append(("snapshots_csv", snapshots))
        elif policy == CleanupPolicy.AGGRESSIVE:
            to_delete.append(("snapshots_csv", snapshots))

    if policy == CleanupPolicy.AGGRESSIVE:
        if features_dom.exists():
            to_delete.append(("features_dom", features_dom))
        if features_agg.exists():
            to_delete.append(("features_agg", features_agg))

    for role, path in to_delete:
        if dry_run:
            sz = path.stat().st_size / (1024**2)
            actions[role] = f"WOULD_DELETE {sz:.0f}MB"
        else:
            ok = storage.delete_file(path, f"cleanup_policy={policy.value}")
            actions[role] = "deleted" if ok else "failed"

    return actions


# ── Archive Manifest ──────────────────────────────────────────────────────────

def write_archive_manifest(date: str, out_dir: Path,
                           storage: StorageMonitor) -> Path:
    """Write per-day archive manifest CSV."""
    manifest_path = out_dir / f"archive_manifest_{date}.csv"

    file_defs = [
        ("raw_depth",        f"../../input/{date}/*.depth",    "KEEP_ON_VPS",        1, False),
        ("events_csv",       "events.csv",                    "REGENERABLE",         3, False),
        ("snapshots_csv",    "snapshots.csv",                 "READY_TO_DELETE",     5, True),
        ("features_dom",     "features_dom.csv",              "REGENERABLE",         4, True),
        ("features_agg",     "features_dom_agg.csv",          "REGENERABLE",         4, True),
        ("sampled_events",   "sampled_events.csv",             "KEEP_ON_VPS",         1, False),
        ("excursion_stats",  "excursion_stats.csv",            "KEEP_ON_VPS",         1, False),
        ("labels_c1",        _cand_label_name(CANDIDATES[0]),  "KEEP_ON_VPS",         1, False),
        ("labels_c2",        _cand_label_name(CANDIDATES[1]),  "KEEP_ON_VPS",         1, False),
        ("labels_c3",        _cand_label_name(CANDIDATES[2]),  "KEEP_ON_VPS",         1, False),
        ("phase8_summary",  "phase8_dataset_summary.csv",     "KEEP_ON_VPS",         1, False),
        ("phase8_results",  "phase8_trainval_results.csv",   "KEEP_ON_VPS",         1, False),
        ("phase8_fi",       "phase8_feature_importance.csv",  "KEEP_ON_VPS",         2, False),
        ("phase8_model_md", "phase8_best_candidate.md",      "KEEP_ON_VPS",         1, False),
        ("phase8_model_pkl","phase8_best_model.pkl",         "KEEP_ON_VPS",         1, False),
        ("run_manifest",    f"archive_manifest_{date}.csv",   "KEEP_ON_VPS",         1, False),
    ]

    rows = []
    for role, rel_path, retention, priority, delete_after in file_defs:
        if ".." in rel_path:
            continue  # skip external paths
        actual = out_dir / rel_path
        size = actual.stat().st_size if actual.exists() else 0
        rows.append({
            "date": date,
            "file_role": role,
            "file_path": str(actual),
            "file_size_bytes": size,
            "file_size_mb": round(size / (1024**2), 2),
            "retention_class": retention if actual.exists() else "MISSING",
            "archive_priority": priority,
            "delete_after_archive": delete_after,
            "notes": "",
        })

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        w.writeheader()
        w.writerows(rows)

    print(f"    [Archive] Manifest → {manifest_path.name} ({len(rows)} files)")
    return manifest_path


# ── Storage Report ─────────────────────────────────────────────────────────────

def write_storage_report(output_dir: Path,
                         days_complete: list[str],
                         days_failed: list[str],
                         storage: StorageMonitor) -> Path:
    report_path = output_dir / "archive_strategy_report.md"
    free_gb = storage.get_free_gb()

    # Total output size
    total_gb = 0
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("20"):
            total_gb += storage.get_dir_size(d) / (1024**3)

    # Estimate what's saved by cleanup
    snapshot_total = 0
    for date in days_complete:
        p = output_dir / date / "snapshots.csv"
        if p.exists():
            snapshot_total += 0
        else:
            snapshot_total += EST_DISK_PER_DAY_BYTES["snapshots_csv"]

    content = f"""# Archive Strategy Report
Generated: {dt.datetime.now().isoformat()}

## Current Disk Status

| Metric | Value |
|--------|-------|
| VPS Free Space | {free_gb:.1f} GB |
| Total Output Size | {total_gb:.2f} GB |
| Days Completed | {len(days_complete)} |
| Days Failed | {len(days_failed)} |

## Retention Classes

### KEEP_ON_VPS (non-negotiable)
- `sampled_events.csv` — ML feature input (~31MB/day)
- `excursion_stats.csv` — Phase 6 output, required for Phase 7 (~15MB/day)
- `phase7_labels_*` (3 files) — barrier labels for 3 candidates (~3MB/day)
- `phase8_*.csv/md/pkl` — ML model outputs (~10MB/day)

### REGENERABLE
- `events.csv` — Phase 1 output (~45MB/day). Regenerated from `.depth` file.
- `features_dom.csv` — Phase 3 output (~310MB/day). Regenerated from snapshots.
- `features_dom_agg.csv` — Phase 4 output (~50MB/day). Regenerated from features.

### READY_TO_DELETE (after archive-ready cleanup)
- `snapshots.csv` — ~427MB/day. Largest file. Safe to delete after P7+P8 complete.
  Can be regenerated from events.csv if needed (but takes significant CPU).

## Estimated Space Savings

| Policy | Per Day After Cleanup | 30 Days | 60 Days | 90 Days |
|--------|---------------------|---------|---------|---------|
| None (full retention) | ~891MB | ~26.7GB | ~53.4GB | ~80.2GB |
| archive-ready (snapshots deleted) | ~104MB | ~3.1GB | ~6.2GB | ~9.4GB |
| aggressive (all regenerable deleted) | ~54MB | ~1.6GB | ~3.2GB | ~4.9GB |

## Cleanup Commands

After confirming archive is on local storage, on VPS:
```bash
# Delete snapshots for a specific day (after P7+P8 done)
rm /opt/depth-dom/output/{{date}}/snapshots.csv

# Delete features for a specific day (only if needed)
rm /opt/depth-dom/output/{{date}}/features_dom.csv
rm /opt/depth-dom/output/{{date}}/features_dom_agg.csv
```

## Offload Checklist (per completed day)

Transfer from VPS to local for permanent storage:
- [ ] `output/{{date}}/sampled_events.csv`
- [ ] `output/{{date}}/excursion_stats.csv`
- [ ] `output/{{date}}/phase7_labels_*` (all 3)
- [ ] `output/{{date}}/phase8_*.csv` (all)
- [ ] `output/{{date}}/phase8_*.md`
- [ ] `output/{{date}}/phase8_*.pkl`
- [ ] `output/{{date}}/archive_manifest_{{date}}.csv`

Then on VPS: `rm -r /opt/depth-dom/output/{{date}}/snapshots.csv`
"""
    report_path.write_text(content, encoding="utf-8")
    print(f"    [Storage Report] → {report_path}")
    return report_path


# ── Resource Logger ─────────────────────────────────────────────────────────────

class ResourceLogger:
    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.entries.append(row)

    def log(self, date: str, phase: str, runtime_s: float,
            input_bytes: int, output_bytes: int):
        self.entries.append({
            "timestamp": dt.datetime.now().isoformat(),
            "date": date, "phase": phase,
            "runtime_seconds": round(runtime_s, 1),
            "input_size_bytes": input_bytes,
            "output_size_bytes": output_bytes,
        })

    def save(self):
        if not self.entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.entries[0].keys()))
            w.writeheader()
            w.writerows(self.entries)


# ── Per-day pipeline ────────────────────────────────────────────────────────────

def process_day(
    date: str,
    depth_path: Path,
    out_dir: Path,
    cp: CheckpointManager,
    storage: StorageMonitor,
    resource_log: ResourceLogger,
    policy: CleanupPolicy,
    force: bool = False,
    dry_run: bool = False,
    skip_p1_p6: bool = False,
    skip_p7_p8: bool = False,
) -> tuple[str, str, float, int]:
    """Process one trading day through all phases. Returns (status, error, runtime, disk)."""
    t0 = time.time()
    dm = cp.get(date)
    dm.source_file = str(depth_path)
    dm.source_size_bytes = depth_path.stat().st_size
    dm.status = "running"
    cp.save()

    disk_at_start = storage.get_dir_size(out_dir)

    def disk_now() -> int:
        return storage.get_dir_size(out_dir)

    try:
        # ── Phase 1 ────────────────────────────────────────────────────────
        if not skip_p1_p6:
            ok, err = run_phase1(date, depth_path, out_dir, cp, force)
            if not ok:
                cp.finalize(date, "failed", err, time.time()-t0, 0)
                return "failed", err, time.time()-t0, 0

        # ── Phase 2 ────────────────────────────────────────────────────────
        if not skip_p1_p6:
            ok, err = run_phase2(date, out_dir, cp, force)
            if not ok:
                cp.finalize(date, "failed", err, time.time()-t0, 0)
                return "failed", err, time.time()-t0, 0

        # ── Phase 3 ────────────────────────────────────────────────────────
        if not skip_p1_p6:
            ok, err = run_phase3(date, out_dir, cp, force)
            if not ok:
                cp.finalize(date, "failed", err, time.time()-t0, 0)
                return "failed", err, time.time()-t0, 0

        # ── Phase 4 ────────────────────────────────────────────────────────
        if not skip_p1_p6:
            ok, err = run_phase4(date, out_dir, cp, force)
            if not ok:
                cp.finalize(date, "failed", err, time.time()-t0, 0)
                return "failed", err, time.time()-t0, 0

        # ── Phase 5 ────────────────────────────────────────────────────────
        if not skip_p1_p6:
            ok, err = run_phase5(date, out_dir, cp, force)
            if not ok:
                cp.finalize(date, "failed", err, time.time()-t0, 0)
                return "failed", err, time.time()-t0, 0

        # ── Phase 6 ────────────────────────────────────────────────────────
        if not skip_p1_p6:
            ok, err = run_phase6(date, out_dir, cp, force)
            if not ok:
                cp.finalize(date, "failed", err, time.time()-t0, 0)
                return "failed", err, time.time()-t0, 0

        # ── Phase 7 ────────────────────────────────────────────────────────
        if not skip_p7_p8:
            ok, err = run_phase7(date, out_dir, cp, storage, force)
            if not ok:
                cp.finalize(date, "failed", err, time.time()-t0, 0)
                return "failed", err, time.time()-t0, 0

        # ── Phase 7b ───────────────────────────────────────────────────────
        if not skip_p7_p8:
            ok, err = run_phase7b(date, out_dir, cp, force)
            if not ok:
                cp.finalize(date, "failed", err, time.time()-t0, 0)
                return "failed", err, time.time()-t0, 0

        # ── Phase 8 ────────────────────────────────────────────────────────
        if not skip_p7_p8:
            ok, err = run_phase8(date, out_dir, cp, force)
            if not ok:
                cp.finalize(date, "failed", err, time.time()-t0, 0)
                return "failed", err, time.time()-t0, 0

        # ── Archive manifest ─────────────────────────────────────────────
        write_archive_manifest(date, out_dir, storage)

        # ── Cleanup ────────────────────────────────────────────────────────
        cleanup_day(date, out_dir, policy, storage)

        # ── Resource log ─────────────────────────────────────────────────
        runtime = time.time() - t0
        disk_final = disk_now()
        resource_log.log(date, "ALL", runtime, dm.source_size_bytes, disk_final)
        resource_log.save()

        # ── Finalize ─────────────────────────────────────────────────────
        disk_written = disk_final - disk_at_start
        all_phases_done = all(
            dm.phases.get(p) in (PhaseStatus.DONE.value, PhaseStatus.SKIPPED.value)
            for p in PHASE_NAMES
        )
        status = "complete" if all_phases_done else "partial"
        cp.finalize(date, status, "", runtime, disk_written)
        return status, "", runtime, disk_written

    except Exception as e:
        cp.finalize(date, "exception", str(e)[:300], time.time()-t0, 0)
        return "exception", str(e)[:300], time.time()-t0, 0


# ── Main ──────────────────────────────────────────────────────────────────────


def run_p7_if_ready(manifest_path: Path) -> bool:
    """Return True if all days have phase_p6_excursion=DONE."""
    import csv
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase_p6_excursion", "").lower() not in ("done", "complete"):
                return False
    return True


def _process_day_wrapper(args_tuple):
    """Wrapper for process_day to enable parallel execution."""
    date_str, depth_path, out_dir, cp, storage, resource_log, policy, force, skip_p1_p6, skip_p7_p8 = args_tuple
    return process_day(
        date=date_str,
        depth_path=depth_path,
        out_dir=out_dir,
        cp=cp,
        storage=storage,
        resource_log=resource_log,
        policy=policy,
        force=force,
        skip_p1_p6=skip_p1_p6,
        skip_p7_p8=skip_p7_p8,
    )


def process_days_parallel(days: list, date_to_file: dict, output_dir: Path,
                          cp, storage, resource_log, policy,
                          force: bool, skip_p1_p6: bool, skip_p7_p8: bool, max_workers: int = 6) -> list:
    """Process days in parallel using ThreadPoolExecutor. Returns results list."""
    # Build argument tuples for each day
    args_list = []
    for date_str in days:
        path, _, size = date_to_file[date_str]
        day_out = output_dir / date_str
        day_out.mkdir(parents=True, exist_ok=True)
        args_list.append((date_str, path, day_out, cp, storage, resource_log, policy, force, skip_p1_p6, skip_p7_p8))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {}
        for args in args_list:
            future = executor.submit(_process_day_wrapper, args)
            future_to_date[future] = args[0]  # args[0] is date_str
        
        for future in as_completed(future_to_date):
            date_str = future_to_date[future]
            try:
                result = future.result()
                results.append((date_str,) + result)
                icon = {"complete": "✅", "partial": "⚠️", "failed": "❌",
                        "exception": "💥", "skipped_no_space": "⛔"}.get(result[0], "?")
                print(f"  {icon} {date_str}: {result[0]} | {result[2]/60:.1f}min | {result[3]/(1024**3):.3f}GB")
                if result[1]:
                    print(f"     Error: {result[1][:200]}")
            except Exception as e:
                print(f"Day {date_str} FAILED: {e}")
                results.append((date_str, "exception", str(e), 0.0, 0))
    
    return results

def main():
    ap = argparse.ArgumentParser(
        description="VPS Multi-Day DOM Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 vps_multiday_runner.py --dry-run
  python3 vps_multiday_runner.py --cleanup-policy archive-ready --parallel 1 --resume
  python3 vps_multiday_runner.py --cleanup-policy aggressive --resume --force
  python3 vps_multiday_runner.py --archive-manifest-only
        """
    )
    ap.add_argument("--input-dir",  type=Path, default=INPUT_DIR_DEFAULT)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    ap.add_argument("--cleanup-policy",
                   default="archive-ready",
                   choices=["none", "archive-ready", "aggressive"],
                   help="Cleanup policy (default: archive-ready)")
    ap.add_argument("--parallel",   type=int, default=1)
    ap.add_argument("--workers", type=int, default=6,
                    help="Max parallel workers (default: 6, max: available cores)")
    ap.add_argument("--resume",     action="store_true")
    ap.add_argument("--force",      action="store_true")
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--max-days",  type=int, default=0)
    ap.add_argument("--archive-manifest-only", action="store_true")
    ap.add_argument("--skip-p1-p6", action="store_true",
                   help="Assume P1-P6 outputs already exist (skip to P7)")
    ap.add_argument("--skip-p7-p8", action="store_true",
                   help="Skip P7/P8 phases (run P1-P6 only)")

    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = CleanupPolicy(args.cleanup_policy)

    print(f"\n{'='*70}")
    print("VPS MULTI-DAY DOM PIPELINE ORCHESTRATOR")
    print(f"{'='*70}")
    print(f"  Input dir  : {input_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Cleanup    : {policy.value}")
    print(f"  Parallel   : {args.parallel} (sequential = RAM-safe)")
    print(f"  Resume     : {args.resume}")
    print(f"  Force      : {args.force}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"  Skip P1-P6 : {args.skip_p1_p6}")
    print(f"  Skip P7-P8 : {args.skip_p7_p8}")
    print(f"{'='*70}\n")

    # ── Storage ──────────────────────────────────────────────────────────────
    storage = StorageMonitor(output_dir)
    free_gb = storage.get_free_gb()
    print(f"[Storage] Free space: {free_gb:.1f} GB")

    # ── Discover .depth files ─────────────────────────────────────────────────
    print(f"\n[Discovery] Scanning {input_dir} for .depth files ...")
    try:
        depth_files = discover_depth_files(input_dir)
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    if not depth_files:
        if args.dry_run:
            print(f"WARNING: No .depth files found in {input_dir} (dry-run)")
        else:
            print(f"ERROR: No .depth files found in {input_dir}")
            sys.exit(1)

    print(f"  Found {len(depth_files)} .depth file(s):")
    for path, date_str, size in depth_files[:10]:
        print(f"    {date_str}: {path.name} ({size/(1024**2):.0f}MB)")
    if len(depth_files) > 10:
        print(f"    ... and {len(depth_files)-10} more")

    # Deduplicate by date (keep largest file per date)
    date_to_file = {}
    for path, date_str, size in depth_files:
        if date_str not in date_to_file or size > date_to_file[date_str][2]:
            date_to_file[date_str] = (path, date_str, size)

    unique_dates = sorted(date_to_file.keys())
    print(f"\n  {len(unique_dates)} unique trading day(s)")

    # ── Disk estimation ─────────────────────────────────────────────────────
    est = storage.estimate_remaining(len(unique_dates))
    print(f"\n[Storage Estimate]")
    print(f"  Days remaining  : {est['n_days']}")
    print(f"  Est. needed     : {est['needed_gb']:.1f} GB")
    print(f"  Current free    : {est['free_gb']:.1f} GB")
    print(f"  Projected free  : {est['projected_gb']:.1f} GB")
    if not est["safe"]:
        print(f"  ⚠️  Projected space EXCEEDS safe limit!")
        print(f"  Recommendation: --cleanup-policy archive-ready to save ~427MB/day")

    # ── Checkpoint / Manifest ────────────────────────────────────────────────
    manifest_path = output_dir / "_multiday_manifest.csv"
    cp = CheckpointManager(manifest_path)
    for date_str in unique_dates:
        dm = cp.get(date_str)
        if not dm.source_file:
            path, _, size = date_to_file[date_str]
            dm.source_file = str(path)
            dm.source_size_bytes = size
    cp.save()

    # ── Resource logger ──────────────────────────────────────────────────────
    resource_log = ResourceLogger(output_dir / "resource_usage_log.csv")

    # ── Pre-run inventory ────────────────────────────────────────────────────
    inventory_path = output_dir / "multiday_input_inventory.csv"
    with open(inventory_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file_path","file_name","inferred_date","file_size_bytes","status","notes"])
        w.writeheader()
        for path, date_str, size in depth_files:
            day_out = output_dir / date_str
            p8_done = (day_out / "phase8_trainval_results.csv").exists()
            status = "complete" if p8_done else ("found" if path.exists() else "missing")
            w.writerow({
                "file_path": str(path), "file_name": path.name,
                "inferred_date": date_str, "file_size_bytes": size,
                "status": status, "notes": "",
            })
    print(f"\n[Inventory] → {inventory_path}")

    # ── Filter days ──────────────────────────────────────────────────────────
    if args.resume:
        days = [
            d for d in unique_dates
            if not (output_dir / d / "phase8_trainval_results.csv").exists()
        ]
        print(f"\n[Resume] {len(unique_dates)-len(days)} days done, {len(days)} to process")
    else:
        days = list(unique_dates)

    if args.max_days > 0:
        days = days[:args.max_days]
        print(f"\n[max-days] Limited to {args.max_days} days")

    if args.archive_manifest_only:
        for date_str in days:
            day_out = output_dir / date_str
            if day_out.exists():
                write_archive_manifest(date_str, day_out, storage)
        write_storage_report(output_dir, days_complete=[], days_failed=[], storage=storage)
        print("Done.")
        return

    if not days:
        print("\nNo days to process.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(days)} days:")
        for d in days:
            print(f"  {d}")
        return

    # ── Main Loop ─────────────────────────────────────────────────────────────
    print(f"\n[Processing] Starting {len(days)} day(s) ...")
    results: list[tuple[str, str, str, float, int]] = []

    if args.workers == 1 or len(days) == 1:
        # Sequential processing (original behavior)
        for i, date_str in enumerate(days, 1):
            path, _, size = date_to_file[date_str]
            day_out = output_dir / date_str
            day_out.mkdir(parents=True, exist_ok=True)

            print(f"\n[{i}/{len(days)}] {'='*50}")
            print(f"  Date  : {date_str}")
            print(f"  Source: {path.name} ({size/(1024**2):.0f}MB)")

            if not storage.check(f"starting {date_str}"):
                results.append((date_str, "skipped_no_space", "Low disk", 0, 0))
                continue

            status, err, runtime, disk_bytes = process_day(
                date=date_str,
                depth_path=path,
                out_dir=day_out,
                cp=cp,
                storage=storage,
                resource_log=resource_log,
                policy=policy,
                force=args.force,
                skip_p1_p6=args.skip_p1_p6,
                skip_p7_p8=args.skip_p7_p8,
            )
            results.append((date_str, status, err, runtime, disk_bytes))

            icon = {"complete": "✅", "partial": "⚠️", "failed": "❌",
                    "exception": "💥", "skipped_no_space": "⛔"}.get(status, "?")
            print(f"\n  {icon} {date_str}: {status} | {runtime/60:.1f}min | {disk_bytes/(1024**3):.3f}GB")
            if err:
                print(f"     Error: {err[:200]}")

            # PERF: progress tracking
            done_count = sum(1 for d in unique_dates if cp.get(d).status == "complete")
            print(f"[Progress] {done_count}/{len(unique_dates)} days complete | "
                  f"Free: {storage.get_free_gb():.1f}GB")

            gc.collect()
    else:
        # Parallel processing
        print(f"\n[Parallel] Using {min(args.workers, len(days))} workers for {len(days)} days ...")
        results = process_days_parallel(
            days=days,
            date_to_file=date_to_file,
            output_dir=output_dir,
            cp=cp,
            storage=storage,
            resource_log=resource_log,
            policy=policy,
            force=args.force,
            skip_p1_p6=args.skip_p1_p6,
            skip_p7_p8=args.skip_p7_p8,
            max_workers=min(args.workers, len(days)),
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    success = sum(1 for _, s, _, _, _ in results if s == "complete")
    partial = sum(1 for _, s, _, _, _ in results if s == "partial")
    failed  = sum(1 for _, s, _, _, _ in results if s in ("failed", "exception"))
    skipped = sum(1 for _, s, _, _, _ in results if s == "skipped_no_space")

    print(f"\n{'='*70}")
    print("MULTI-DAY RUN COMPLETE")
    print(f"{'='*70}")
    print(f"  Complete: {success}")
    print(f"  Partial: {partial}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped} (no space)")
    print(f"  Free space (end): {storage.get_free_gb():.1f} GB")

    # Write failure log
    failed_entries = [(d, s, e) for d, s, e, _, _ in results if s in ("failed","exception")]
    if failed_entries:
        fail_path = output_dir / "phase8_multiday_failure_log.csv"
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["date","status","error_message"])
            w.writeheader()
            w.writerows([{"date": d, "status": s, "error_message": e} for d, s, e in failed_entries])
        print(f"  Failure log → {fail_path}")

    # Write manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        rows = [cp.get(d).to_row() for d in unique_dates]
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"  Manifest → {manifest_path}")

    # Storage report
    write_storage_report(
        output_dir,
        [d for d, s, _, _, _ in results if s == "complete"],
        [d for d, s, _, _, _ in results if s in ("failed","exception")],
        storage,
    )
    resource_log.save()

    print(f"\n  All outputs → {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
