#!/usr/bin/env python3
"""
status_live.py — Live Operational Status for DEPTH-DOM Pipeline
================================================================
Reads checkpoint files, process table, and log tail to produce a complete
operational view of every pipeline day.

Output: /opt/depth-dom/output/_live_status.json
Human-readable table printed to stdout.

Usage:
    python3 /opt/depth-dom/status_live.py
    python3 /opt/depth-dom/status_live.py --output-dir /opt/depth-dom/output
    python3 /opt/depth-dom/status_live.py --json-only   # suppress table output
    python3 /opt/depth-dom/status_live.py --plan         # grouped recovery queues
"""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# ── Constants ───────────────────────────────────────────────────────────────────

VPS_BASE = "/opt/depth-dom"
OUTPUT_DIR_DEFAULT = "/opt/depth-dom/output"
LIVE_STATUS_FILE = "_live_status.json"

PHASE_ORDER = [
    "p1_parse", "p2_reconstruct", "p2b_fusion", "p3_features", "p4_agg",
    "p5_sample", "p6_excursion",
    "p7_c1", "p7_c2", "p7_c3",
    "p8_ml",
]

# Phase output file mapping
PHASE_OUTPUT = {
    "p1_parse": "events.csv",
    "p2_reconstruct": "snapshots.csv",
    "p3_features": "features_dom.csv",
    "p4_agg": "features_dom_agg.csv",
    "p5_sample": "sampled_events.csv",
    "p6_excursion": "excursion_stats.csv",
    "p7_c1": None,   # P7 outputs are directories, handled separately
    "p7_c2": None,
    "p7_c3": None,
    "p8_ml": "phase8_trainval_results.csv",
}

SLOW_THRESHOLD_SEC = 600
HUNG_THRESHOLD_SEC = 3600
HEARTBEAT_FILE = "_heartbeat"

CANDIDATES = [
    {"idx": 1, "vb_ticks": 30,  "pt_ticks": 9.5,  "sl_ticks": 9.8},
    {"idx": 2, "vb_ticks": 60,  "pt_ticks": 20.0, "sl_ticks": 20.0},
    {"idx": 3, "vb_ticks": 120, "pt_ticks": 13.0, "sl_ticks": 14.5},
]


# ── Enums ─────────────────────────────────────────────────────────────────────

class DayState(str, Enum):
    DONE           = "done"           # all phases complete
    READY_P7       = "ready_p7"      # P6 done, P7 not started/needed, intermediates usable
    READY_P8       = "ready_p8"      # P7 done (all 3 candidates), P8 not done
    NEEDS_P7_RERUN = "needs_p7"     # P7 attempted but no valid label dirs exist
    NEEDS_P6       = "needs_p6"      # P6 failed or missing, P5 output exists
    NEEDS_P5       = "needs_p5"      # P5 failed, P4 output exists
    NEEDS_P4       = "needs_p4"      # P4 failed, P3 output exists
    NEEDS_P3       = "needs_p3"      # P3 failed, P2 output exists
    NEEDS_P2       = "needs_p2"      # P2 failed, P1 output exists
    NEEDS_P1       = "needs_p1"      # P1 failed — check inputs
    NO_INPUT       = "no_input"      # no .depth files found for this date
    RUNNING         = "running"       # active process detected
    SLOW            = "slow"          # running > slow threshold
    HUNG            = "hung"          # running > hung threshold
    STALLED         = "stalled"      # no process, no heartbeat, partial completion
    UNKNOWN         = "unknown"       # unclear state


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class PhaseDetail:
    name: str
    status: str          # pending / running / done / failed
    error: str | None
    done_file_age_sec: float | None
    output_exists: bool
    output_size_bytes: int | None


@dataclass
class DayStatus:
    date: str
    state: str
    state_reason: str

    # Phase details
    phases: list[dict]

    # Key boundary flags
    has_p1_output: bool       # events.csv exists
    has_p5_output: bool       # sampled_events.csv exists
    has_p6_output: bool       # excursion_stats.csv exists
    has_p7_label_dirs: bool   # any P7 label dir exists with CSV

    p7_label_dir_count: int   # how many P7 label dirs exist
    p7_sentinel_done_count: int  # how many p7_cN.done sentinels say done

    # Recovery info
    highest_reached_phase: str | None
    highest_valid_completed: str | None
    current_failing_boundary: str | None
    recommended_action: str
    action_reason: str

    # Process info
    pid: int | None
    elapsed_sec: float | None
    heartbeat_age_sec: float | None
    log_tail: str

    # For --plan grouping
    p1_failure_reason: str | None = None
    depth_file_count: int = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cand_label_name(c: dict) -> str:
    """Build Phase 7 label directory name using canonical vb_ticks/pt_ticks/sl_ticks keys."""
    pt_s = str(c["pt_ticks"]).replace(".", "p")
    sl_s = str(c["sl_ticks"]).replace(".", "p")
    return f"phase7_labels_{c['vb_ticks']}ticks_{pt_s}_{sl_s}"


def parse_done_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        result = {}
        for line in content.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                result[k.strip()] = v.strip()
        return result
    except OSError:
        return {}


def file_age_sec(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return dt.datetime.now().timestamp() - path.stat().st_mtime
    except OSError:
        return None


def find_heartbeat(out_dir: Path):
    hb = out_dir / HEARTBEAT_FILE
    hb2 = out_dir / "_checkpoints" / HEARTBEAT_FILE
    for candidate in [hb, hb2]:
        if candidate.exists():
            try:
                return candidate, file_age_sec(candidate)
            except OSError:
                pass
    return None, None


def get_all_pipeline_processes() -> dict[int, dict]:
    try:
        r = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10)
        processes = {}
        keywords = ["main.py", "phase7_labeling", "phase8_entry", "excursion_analysis",
                    "cusum_sampler", "feature_engineering", "book_reconstructor",
                    "vps_multiday_runner", "incremental_p7p8"]
        for line in r.stdout.splitlines():
            lower = line.lower()
            if any(k.lower() in lower for k in keywords) and "grep" not in lower:
                parts = line.split()
                if len(parts) >= 2:
                    pid = int(parts[1])
                    elapsed_str = parts[9] if len(parts) > 9 else "0"
                    try:
                        elapsed_min = float(elapsed_str)
                    except ValueError:
                        elapsed_min = 0.0
                    cmd = " ".join(parts[10:]) if len(parts) > 10 else ""
                    processes[pid] = {"cmd": cmd, "elapsed_min": elapsed_min}
        return processes
    except Exception:
        return {}


def get_log_tail(date: str, n_lines: int = 5) -> str:
    log_path = Path(VPS_BASE) / "pipeline_full3.log"
    if not log_path.exists():
        return ""
    try:
        r = subprocess.run(["grep", date, str(log_path)],
                          capture_output=True, text=True, timeout=10)
        lines = r.stdout.strip().splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def get_p7_label_dirs(out_dir: Path) -> tuple[int, int]:
    """Return (label_dir_count, sentinel_done_count) for P7.
    label_dir_count = number of label dirs with at least 1 CSV
    sentinel_done_count = number of p7_cN.done sentinels with status=done
    """
    label_count = 0
    sentinel_done = 0
    for c in CANDIDATES:
        key = f"p7_c{c['idx']}"
        ldir = out_dir / _cand_label_name(c)
        try:
            if ldir.is_dir() and any(f.suffix == ".csv" for f in ldir.iterdir()):
                label_count += 1
        except OSError:
            pass
        # Check sentinel
        done_file = out_dir / "_checkpoints" / f"{key}.done"
        info = parse_done_file(done_file)
        if info.get("status") == "done":
            sentinel_done += 1
    return label_count, sentinel_done


def check_p7_label_dirs_for_day(out_dir: Path) -> bool:
    """True if at least one P7 label dir has CSV files."""
    for c in CANDIDATES:
        ldir = out_dir / _cand_label_name(c)
        try:
            if ldir.is_dir() and any(f.suffix == ".csv" for f in ldir.iterdir()):
                return True
        except OSError:
            pass
    return False


def has_output(out_dir: Path, phase_key: str) -> tuple[bool, int | None]:
    """Check if phase output file exists. Returns (exists, size_bytes)."""
    fname = PHASE_OUTPUT.get(phase_key)
    if fname is None:
        return False, None
    p = out_dir / fname
    try:
        if p.exists() and p.is_file():
            return True, p.stat().st_size
    except OSError:
        pass
    return False, None


def get_depth_files_for_date(date: str) -> list[Path]:
    """Find .depth files matching a date."""
    input_dir = Path(VPS_BASE) / "input"
    year, month, day_num = date.split("-")
    pattern = f"*{year}{month}{day_num}*.depth"
    try:
        return sorted(input_dir.rglob(pattern))
    except Exception:
        return []


def inspect_day(date: str, out_dir: Path, now_ts: float) -> DayStatus:
    """Inspect a single day directory and return its DayStatus."""
    checkpoint_dir = out_dir / "_checkpoints"

    # ── Per-phase details ────────────────────────────────────────────────────
    phase_details = []
    for phase in PHASE_ORDER:
        done_file = checkpoint_dir / f"{phase}.done"
        info = parse_done_file(done_file)
        status = info.get("status", "pending")
        error = info.get("error", None) if status == "failed" else None
        age = file_age_sec(done_file)

        # Check output existence
        phase_key = phase  # same name works for output mapping
        out_exists, out_size = has_output(out_dir, phase_key)

        phase_details.append(PhaseDetail(
            name=phase,
            status=status,
            error=error,
            done_file_age_sec=age,
            output_exists=out_exists,
            output_size_bytes=out_size,
        ))

    # Quick lookups
    phases = {p.name: p for p in phase_details}
    done_phases = [p.name for p in phase_details if p.status == "done"]
    failed_phases = [p.name for p in phase_details if p.status == "failed"]
    pending_phases = [p.name for p in phase_details if p.status == "pending"]

    # ── Output file checks ──────────────────────────────────────────────────
    has_p1, p1_size = has_output(out_dir, "p1_parse")
    has_p5, p5_size = has_output(out_dir, "p5_sample")
    has_p6, p6_size = has_output(out_dir, "p6_excursion")
    has_p7_labels = check_p7_label_dirs_for_day(out_dir)

    p7_label_count, p7_sentinel_done = get_p7_label_dirs(out_dir)

    # ── .depth file check ───────────────────────────────────────────────────
    depth_files = get_depth_files_for_date(date)
    depth_file_count = len(depth_files)

    # ── Process check ───────────────────────────────────────────────────────
    processes = get_all_pipeline_processes()
    date_processes = [p for pid, p in processes.items()
                       if date in p.get("cmd", "")]

    hb_path, hb_age = find_heartbeat(out_dir)
    pid = date_processes[0]["pid"] if date_processes else None
    elapsed = date_processes[0]["elapsed_min"] * 60 if date_processes else None

    # ── Highest phases ─────────────────────────────────────────────────────
    all_phases_with_status = [p.name for p in phase_details if p.status != "pending"]
    highest_reached = all_phases_with_status[-1] if all_phases_with_status else None

    # highest_valid_completed: last done phase whose output file actually exists
    highest_valid = None
    for p in reversed(phase_details):
        if p.status == "done":
            # For P7, check label dir exists (not just sentinel)
            if p.name.startswith("p7"):
                ldir_name = _cand_label_name(CANDIDATES[int(p.name[-1]) - 1])
                ldir = out_dir / ldir_name
                try:
                    if ldir.is_dir() and any(f.suffix == ".csv" for f in ldir.iterdir()):
                        highest_valid = p.name
                        break
                except OSError:
                    pass
            else:
                # For non-P7: check output file exists
                out_exists, _ = has_output(out_dir, p.name)
                if out_exists:
                    highest_valid = p.name
                    break

    # ── P1 failure reason ─────────────────────────────────────────────────
    p1_failure_reason = None
    if phases["p1_parse"].status == "failed":
        err = phases["p1_parse"].error or ""
        if "events.csv not produced" in err or "events.csv missing" in err:
            if has_p1:
                p1_failure_reason = "events_exists_despite_failed_checkpoint"
            elif depth_file_count == 0:
                p1_failure_reason = "no_depth_files_found"
            else:
                p1_failure_reason = f"depth_files={depth_file_count}_but_parse_failed"
        else:
            p1_failure_reason = err or "unknown"

    # ── Classify state ─────────────────────────────────────────────────────
    state, reason = classify_state(
        phases, done_phases, failed_phases, pending_phases,
        has_p1, has_p5, has_p6, has_p7_labels,
        p7_label_count, p7_sentinel_done,
        date_processes, elapsed, hb_age,
        highest_valid, p1_failure_reason, depth_file_count,
    )

    # ── Recovery hint ──────────────────────────────────────────────────────
    action, action_reason = compute_recovery_action(
        state, phases, done_phases, failed_phases,
        has_p1, has_p5, has_p6, has_p7_labels,
        p7_label_count, p7_sentinel_done,
        highest_valid, p1_failure_reason, depth_file_count,
        phase_details,
    )

    log_tail = get_log_tail(date)

    return DayStatus(
        date=date,
        state=state,
        state_reason=reason,
        phases=[asdict(p) for p in phase_details],
        has_p1_output=has_p1,
        has_p5_output=has_p5,
        has_p6_output=has_p6,
        has_p7_label_dirs=has_p7_labels,
        p7_label_dir_count=p7_label_count,
        p7_sentinel_done_count=p7_sentinel_done,
        highest_reached_phase=highest_reached,
        highest_valid_completed=highest_valid,
        current_failing_boundary=failed_phases[-1] if failed_phases else None,
        recommended_action=action,
        action_reason=action_reason,
        pid=pid,
        elapsed_sec=elapsed,
        heartbeat_age_sec=hb_age,
        log_tail=log_tail[-500:] if log_tail else "",
        p1_failure_reason=p1_failure_reason,
        depth_file_count=depth_file_count,
    )


def classify_state(
    phases, done_phases, failed_phases, pending_phases,
    has_p1, has_p5, has_p6, has_p7_labels,
    p7_label_count, p7_sentinel_done,
    processes, elapsed, hb_age,
    highest_valid, p1_failure_reason, depth_file_count,
) -> tuple[str, str]:
    """Classify the overall state of a day."""

    # Active process
    if processes:
        if elapsed and elapsed > HUNG_THRESHOLD_SEC:
            return DayState.HUNG.value, f"process running >1h"
        if elapsed and elapsed > SLOW_THRESHOLD_SEC:
            return DayState.SLOW.value, f"process running {elapsed/60:.0f}min"
        return DayState.RUNNING.value, "process running"

    # Check NO_INPUT first
    if p1_failure_reason == "no_depth_files_found":
        if not has_p1 and not any([has_p5, has_p6]):
            return DayState.NO_INPUT.value, "no .depth files and no outputs — pipeline can't run"

    # All phases done
    if len(done_phases) == len(phases):
        return DayState.DONE.value, "all phases complete"

    # P8 complete
    if "p8_ml" in done_phases:
        return DayState.DONE.value, "P1-P8 complete"

    # P7 label dirs exist AND sentinel says all done — P8 is ready
    if p7_label_count == 3 and p7_sentinel_done == 3:
        return DayState.READY_P8.value, "all 3 P7 candidates done with label dirs"

    # P6 done with usable output, P7 labels exist but not all done — NEEDS_P7_RERUN
    # Key: if p7_label_count == 0 (no valid label dirs), P7 needs to be run even if sentinels exist
    if has_p6:
        if p7_label_count > 0:
            # Some labels exist — partial success
            return DayState.NEEDS_P7_RERUN.value, f"P6 done, P7 partial (labels={p7_label_count}/3)"
        elif p7_sentinel_done > 0:
            # Sentinels say done but no label dirs — need to rerun
            return DayState.NEEDS_P7_RERUN.value, f"P6 done, P7 attempted but no label dirs ({p7_sentinel_done} sentinels)"
        else:
            # P6 done, P7 not started
            return DayState.READY_P7.value, "P6 done, P7 not started"

    # P5 done with usable output, P6 missing
    if has_p5 and not has_p6:
        if "p6_excursion" in failed_phases:
            return DayState.NEEDS_P6.value, "P6 failed despite P5 output"
        return DayState.NEEDS_P6.value, "P6 not done despite P5 output"

    # P4 output exists, P5 missing
    if phases.get("p4_agg", {}).get("status") == "done" if isinstance(phases.get("p4_agg"), dict) else False:
        pass  # handled below
    p4_out_exists, _ = False, None  # will be recomputed
    for phase_name in ["p4_agg", "p3_features", "p2_reconstruct", "p1_parse"]:
        p_out, _ = (None, None)  # placeholder
        ph = phases.get(phase_name, PhaseDetail(name=phase_name, status="pending", error=None,
                                                 done_file_age_sec=None, output_exists=False, output_size_bytes=None))
        if isinstance(ph, dict):
            ph = PhaseDetail(**ph)
        if ph.status == "failed":
            out_exists = ph.output_exists
            if out_exists:
                # Has output despite failure — need to rerun from next phase
                next_phase_map = {
                    "p1_parse": DayState.NEEDS_P1.value,
                    "p2_reconstruct": DayState.NEEDS_P2.value,
                    "p3_features": DayState.NEEDS_P3.value,
                    "p4_agg": DayState.NEEDS_P4.value,
                }
                return next_phase_map.get(phase_name, DayState.UNKNOWN.value), \
                       f"{phase_name} failed but output exists"
            else:
                # No output — go further back
                continue

    # P3 output exists, P4 missing
    p3_done = phases.get("p3_features", PhaseDetail(name="p3_features", status="pending",
                                                      error=None, done_file_age_sec=None,
                                                      output_exists=False, output_size_bytes=None))
    if isinstance(p3_done, dict):
        p3_done = PhaseDetail(**p3_done)
    if p3_done.status == "done" and p3_done.output_exists:
        if "p4_agg" in failed_phases or not phases.get("p4_agg", PhaseDetail(name="p4_agg", status="pending", error=None, done_file_age_sec=None, output_exists=False, output_size_bytes=None)).output_exists:
            return DayState.NEEDS_P4.value, "P4 missing despite P3 output"

    # P2 output exists, P3 missing
    p2_done = phases.get("p2_reconstruct", PhaseDetail(name="p2_reconstruct", status="pending",
                                                        error=None, done_file_age_sec=None,
                                                        output_exists=False, output_size_bytes=None))
    if isinstance(p2_done, dict):
        p2_done = PhaseDetail(**p2_done)
    if p2_done.status == "done" and p2_done.output_exists:
        if "p3_features" in failed_phases or not phases.get("p3_features", PhaseDetail(name="p3_features", status="pending", error=None, done_file_age_sec=None, output_exists=False, output_size_bytes=None)).output_exists:
            return DayState.NEEDS_P3.value, "P3 missing despite P2 output"

    # P1 output exists, P2 missing
    if has_p1 and "p2_reconstruct" not in done_phases:
        if "p2_reconstruct" in failed_phases:
            return DayState.NEEDS_P2.value, "P2 failed despite P1 output"
        return DayState.NEEDS_P2.value, "P2 not done despite P1 output"

    # P1 failed
    if "p1_parse" in failed_phases:
        if p1_failure_reason == "no_depth_files_found":
            return DayState.NO_INPUT.value, "no .depth input files for this date"
        return DayState.NEEDS_P1.value, f"P1 failed: {p1_failure_reason}"

    # Nothing done
    if not done_phases and not failed_phases:
        return DayState.UNKNOWN.value, "no checkpoint activity"

    # Stalled: some done, some pending, no process
    if done_phases:
        return DayState.STALLED.value, "partial completion, no active process"

    return DayState.UNKNOWN.value, "unclear state"


def compute_recovery_action(
    state, phases, done_phases, failed_phases,
    has_p1, has_p5, has_p6, has_p7_labels,
    p7_label_count, p7_sentinel_done,
    highest_valid, p1_failure_reason, depth_file_count,
    phase_details,
) -> tuple[str, str]:
    """Compute the recommended next action."""
    phases_by_name = {p.name: p for p in phase_details}

    # State-driven actions
    if state in (DayState.DONE.value,):
        return "none", "all phases complete"

    if state == DayState.READY_P8.value:
        return "run_p8_only", "P7 all 3 candidates complete, P8 not done"

    if state == DayState.READY_P7.value:
        return "run_p7_candidates", "P6 complete, P7 not started — run all 3 candidates"

    if state == DayState.NEEDS_P7_RERUN.value:
        if has_p6 and p7_sentinel_done > 0 and p7_label_count == 0:
            return "run_p7_candidates", f"P6 done, P7 attempted ({p7_sentinel_done} sentinels done) but no label dirs — rerun P7"
        if has_p6 and p7_label_count > 0:
            return "run_p7_candidates", f"P6 done, P7 partial (labels={p7_label_count}/3) — complete remaining candidates"
        return "inspect_p7_failure", "P6 done but P7 state unclear"

    if state == DayState.NEEDS_P6.value:
        if has_p5:
            return "rerun_p6_only", "P5 output exists, P6 failed or missing — rerun P6"
        return "inspect_p5_output", "P6 needed but P5 output missing"

    if state == DayState.NEEDS_P5.value:
        p4_out = phases_by_name.get("p4_agg")
        if p4_out and p4_out.output_exists:
            return "rerun_p5_p6", "P4 output exists, P5 failed — rerun P5 then P6"
        return "inspect_p4_output", "P5 needed but P4 output missing"

    if state == DayState.NEEDS_P4.value:
        p3_out = phases_by_name.get("p3_features")
        if p3_out and p3_out.output_exists:
            return "rerun_p4_p6", "P3 output exists, P4 failed — rerun P4 then P5-P6"
        return "inspect_p3_output", "P4 needed but P3 output missing"

    if state == DayState.NEEDS_P3.value:
        p2_out = phases_by_name.get("p2_reconstruct")
        if p2_out and p2_out.output_exists:
            return "rerun_p3_p6", "P2 output exists, P3 failed — rerun P3 then P4-P6"
        return "inspect_p2_output", "P3 needed but P2 output missing"

    if state == DayState.NEEDS_P2.value:
        if has_p1:
            return "rerun_p2_p6", "P1 output exists, P2 failed — rerun P2 then P3-P6"
        return "inspect_p1_input", "P2 needed but P1 output missing"

    if state == DayState.NEEDS_P1.value:
        if p1_failure_reason == "no_depth_files_found":
            return "no_action_input_missing", "no .depth files found — manual investigation required"
        if p1_failure_reason and "events_exists_despite_failed_checkpoint" in str(p1_failure_reason):
            return "inspect_p1_checkpoint_bug", "events.csv exists despite p1_parse=failed — checkpoint bug"
        return "inspect_p1_input", f"P1 failed: {p1_failure_reason}"

    if state == DayState.NO_INPUT.value:
        return "manual_review_input", "no input files — manual investigation required"

    if state in (DayState.RUNNING.value, DayState.SLOW.value, DayState.HUNG.value):
        return "monitor", f"{state} — monitor only"

    if state == DayState.STALLED.value:
        if highest_valid:
            return f"rerun_from_{highest_valid}", f"stalled, last valid={highest_valid}"

    return "manual_review", f"unclear state: {state}"


# ── Recovery Queue Builder ───────────────────────────────────────────────────────

RECOVERY_GROUPS = [
    ("no_action_input_missing",  "NO INPUT — Manual Required"),
    ("manual_review_input",      "Manual Review — Input Missing"),
    ("manual_review",            "Manual Review — Unknown State"),
    ("inspect_p1_checkpoint_bug","Inspect — P1 Checkpoint Bug"),
    ("inspect_p1_input",        "Inspect — P1 Input/Parse"),
    ("inspect_p7_failure",       "Inspect — P7 Failure"),
    ("run_p8_only",             "Run P8 Only"),
    ("run_p7_candidates",       "Run P7 Candidates (all 3)"),
    ("rerun_p6_only",           "Rerun P6 Only"),
    ("rerun_p5_p6",             "Rerun P5-P6"),
    ("rerun_p4_p6",             "Rerun P4-P6"),
    ("rerun_p3_p6",             "Rerun P3-P6"),
    ("rerun_p2_p6",             "Rerun P2-P6"),
    ("rerun_from_p6",           "Rerun from P6"),
    ("rerun_from_p5",           "Rerun from P5"),
    ("rerun_from_p4",           "Rerun from P4"),
    ("rerun_from_p3",           "Rerun from P3"),
    ("rerun_from_p2",           "Rerun from P2"),
    ("rerun_from_p1",           "Rerun from P1"),
    ("rerun_full",              "Full Pipeline Rerun"),
    ("none",                    "Complete — No Action"),
    ("monitor",                 "Monitor (Running)"),
]


def build_recovery_queues(days: list[DayStatus]) -> dict:
    """Group days by recommended action."""
    queues = defaultdict(list)
    for day in days:
        action = day.recommended_action
        queues[action].append({
            "date": day.date,
            "state": day.state,
            "state_reason": day.state_reason,
            "action_reason": day.action_reason,
            "has_p1": day.has_p1_output,
            "has_p5": day.has_p5_output,
            "has_p6": day.has_p6_output,
            "has_p7_labels": day.has_p7_label_dirs,
            "p7_label_count": day.p7_label_dir_count,
            "highest_valid": day.highest_valid_completed,
            "p1_failure_reason": day.p1_failure_reason,
            "depth_file_count": day.depth_file_count,
        })

    # Sort each queue by date
    for action in queues:
        queues[action].sort(key=lambda x: x["date"])

    # Build ordered result
    ordered = []
    for action_key, group_label in RECOVERY_GROUPS:
        if action_key in queues:
            ordered.append({
                "action": action_key,
                "label": group_label,
                "count": len(queues[action_key]),
                "days": queues[action_key],
            })
            del queues[action_key]
    # Append any remaining actions not in the predefined list
    for action_key, day_list in sorted(queues.items()):
        ordered.append({
            "action": action_key,
            "label": f"Other: {action_key}",
            "count": len(day_list),
            "days": day_list,
        })

    return ordered


# ── Formatting helpers ─────────────────────────────────────────────────────────

def fmt_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "-"
    if size_bytes >= 1024**3:
        return f"{size_bytes/1024**3:.1f}G"
    if size_bytes >= 1024**2:
        return f"{size_bytes/1024**2:.0f}M"
    return f"{size_bytes/1024:.0f}K"


STATE_COLOR = {
    "done": "\033[92m",
    "ready_p7": "\033[96m",
    "ready_p8": "\033[96m",
    "needs_p7": "\033[93m",
    "needs_p6": "\033[93m",
    "needs_p5": "\033[93m",
    "needs_p4": "\033[93m",
    "needs_p3": "\033[93m",
    "needs_p2": "\033[93m",
    "needs_p1": "\033[91m",
    "no_input": "\033[91m",
    "running": "\033[94m",
    "slow": "\033[93m",
    "hung": "\033[91m",
    "stalled": "\033[93m",
    "unknown": "\033[90m",
}
RESET = "\033[0m"


def print_status_table(days: list[DayStatus]):
    """Print human-readable status table."""
    print()
    hdr = f"{'DATE':<12} {'STATE':<16} {'REACHED':<12} {'VALID':<12} {'P1':>6} {'P5':>8} {'P6':>8} {'P7L':>4} {'ACTION':<35}"
    print(hdr)
    print("-" * 120)

    for d in sorted(days, key=lambda x: x.date):
        color = STATE_COLOR.get(d.state, "")
        state_str = f"{color}{d.state:<16}{RESET}"
        reach = d.highest_reached_phase or "-"
        valid = d.highest_valid_completed or "-"
        p1 = fmt_size(None) if not d.has_p1_output else "Y"
        p5 = fmt_size(None) if not d.has_p5_output else "Y"
        p6 = fmt_size(None) if not d.has_p6_output else "Y"
        p7l = str(d.p7_label_dir_count)
        reason = d.state_reason[:30] if d.state_reason else ""
        action = d.recommended_action[:33]

        print(f"{d.date:<12} {state_str} {reach:<12} {valid:<12} {p1:>6} {p5:>8} {p6:>8} {p7l:>4} {action} {reason}")

    print()

    # Summary
    counts = defaultdict(int)
    for d in days:
        counts[d.state] += 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Total: {len(days)} days | {summary}")


def print_plan(days: list[DayStatus]):
    """Print grouped recovery queues."""
    queues = build_recovery_queues(days)

    print()
    print("=" * 100)
    print("RECOVERY PLAN — Grouped by Action")
    print("=" * 100)

    total_need_action = 0
    for group in queues:
        action = group["action"]
        label = group["label"]
        group_days = group["days"]
        count = group["count"]

        if action in ("none", "monitor"):
            continue  # skip

        total_need_action += count

        print(f"\n[{action}] {label} ({count} day{'s' if count != 1 else ''})")
        print("-" * 80)

        for day in group_days:
            notes = []
            if day["has_p1"]:
                notes.append("p1=Y")
            if day["has_p5"]:
                notes.append("p5=Y")
            if day["has_p6"]:
                notes.append("p6=Y")
            if day["p7_label_count"] > 0:
                notes.append(f"p7lbl={day['p7_label_count']}")
            if day["p1_failure_reason"]:
                notes.append(f"p1_fail={day['p1_failure_reason'][:40]}")
            if day["depth_file_count"] > 0:
                notes.append(f"depth={day['depth_file_count']}")

            note_str = f"  # {', '.join(notes)}" if notes else ""
            reason_str = f" — {day['action_reason']}" if day['action_reason'] else ""
            print(f"  {day['date']:<12} [{day['state']}]{reason_str}{note_str}")

    print()
    print("=" * 100)
    print(f"SUMMARY: {total_need_action} days need action")

    done_count = sum(1 for d in days if d.state == DayState.DONE.value)
    running_count = sum(1 for d in days if d.state in (DayState.RUNNING.value, DayState.SLOW.value))
    no_input = sum(1 for d in days if d.state == DayState.NO_INPUT.value)
    manual_review = sum(1 for d in days if d.state in (DayState.UNKNOWN.value, DayState.STALLED.value))
    p7_runnable = sum(1 for d in days if d.recommended_action == "run_p7_candidates")
    p8_runnable = sum(1 for d in days if d.recommended_action == "run_p8_only")

    print(f"  DONE: {done_count}")
    print(f"  RUNNING: {running_count}")
    print(f"  NO INPUT (manual): {no_input}")
    print(f"  MANUAL REVIEW (unclear): {manual_review}")
    print(f"  READY FOR P7 (incremental runner): {p7_runnable}")
    print(f"  READY FOR P8 (incremental runner): {p8_runnable}")


# ── Duplicate detection ───────────────────────────────────────────────────────

def detect_duplicate_runners() -> list[dict]:
    processes = get_all_pipeline_processes()
    runner_cmds = defaultdict(list)
    for pid, info in processes.items():
        cmd = info["cmd"]
        if "vps_multiday_runner" in cmd:
            kind = "vps_multiday_runner"
        elif "incremental_p7p8" in cmd:
            kind = "incremental_p7p8"
        elif "main.py" in cmd:
            kind = "main.py"
        else:
            kind = "phase_script"
        runner_cmds[kind].append({"pid": pid, **info})

    conflicts = []
    for kind, procs in runner_cmds.items():
        if len(procs) > 1 and kind in ("vps_multiday_runner", "incremental_p7p8", "main.py"):
            for p in procs:
                conflicts.append({
                    "kind": kind,
                    "pid": p["pid"],
                    "elapsed_min": p["elapsed_min"],
                    "cmd": p["cmd"][:100],
                    "issue": f"DUPLICATE_{kind.upper()}",
                })
    return conflicts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DEPTH-DOM Live Pipeline Status")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--json-only", action="store_true", help="Suppress table output")
    parser.add_argument("--plan", action="store_true", help="Show grouped recovery queues")
    parser.add_argument("--no-process-check", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    now_ts = dt.datetime.now().timestamp()

    if not output_dir.exists():
        print(f"ERROR: Output directory {output_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    day_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)
    ])

    if not day_dirs:
        print("WARNING: No day directories found")
        sys.exit(0)

    print(f"Inspecting {len(day_dirs)} day directories...")

    days = []
    for day_dir in day_dirs:
        date = day_dir.name
        try:
            status = inspect_day(date, day_dir, now_ts)
            days.append(status)
        except Exception as e:
            print(f"WARNING: Failed to inspect {date}: {e}", file=sys.stderr)

    duplicate_report = [] if args.no_process_check else detect_duplicate_runners()

    result = {
        "generated_at": dt.datetime.now().isoformat(),
        "vps_base": VPS_BASE,
        "output_dir": str(output_dir),
        "total_days": len(days),
        "duplicate_runners": duplicate_report,
        "recovery_queues": build_recovery_queues(days) if args.plan else [],
        "days": [asdict(d) for d in days],
    }

    live_path = output_dir / LIVE_STATUS_FILE
    live_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Written: {live_path}")

    if args.plan:
        print_plan(days)
        return

    if not args.json_only:
        print_status_table(days)

    # Duplicate warnings
    if duplicate_report:
        print("DUPLICATE / CONFLICTING RUNNERS:")
        for dup in duplicate_report:
            print(f"  [{dup['issue']}] PID={dup['pid']} elapsed={dup['elapsed_min']:.0f}min kind={dup['kind']}")
        print()

    # Exit code
    need_action = [d for d in days if d.recommended_action not in ("none", "monitor")]
    running = [d for d in days if d.state in (DayState.RUNNING.value,)]
    if running:
        sys.exit(0)
    if need_action:
        sys.exit(0)  # normal — days need action
    sys.exit(0)


if __name__ == "__main__":
    main()
