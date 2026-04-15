#!/usr/bin/env python3
"""
audit_pipeline.py — Deep audit of all days' real state (LOCAL).
For each day, reports:
- checkpoint states for all phases
- output file existence/sizes
- P7 label directory existence
- .depth file availability
- computed flags: highest_reached, highest_valid, p7_status, p8_status, ready flags

Usage:
    python3 NQdom/ORCHESTRATOR/audit_pipeline.py --output-dir NQdom/output
"""

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Auto-detect local paths ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Pipeline audit (LOCAL)")
parser.add_argument("--output-dir", type=Path, default=None,
                    help="Path to NQdom/output (default: auto-detect)")
parser.add_argument("--input-dir", type=Path, default=None,
                    help="Path to NQdom/INPUT (default: auto-detect)")
_args = parser.parse_args()

OUT_DIR = _args.output_dir or (REPO_ROOT / "output")
INPUT_DIR = _args.input_dir or (REPO_ROOT / "INPUT")

# ── SHARED imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(REPO_ROOT))
from SHARED._pipeline_constants import CANDIDATES

PHASE_ORDER = [
    "p1_parse", "p2_reconstruct", "p2b_fusion", "p3_features", "p4_agg",
    "p5_sample", "p6_excursion",
    "p7_c1", "p7_c2", "p7_c3",
    "p8_ml",
]

KEY_FILES = {
    "p1": "events.csv",
    "p2": "snapshots.csv",
    "p3": "features_dom.csv",
    "p4": "features_dom_agg.csv",
    "p5": "sampled_events.csv",
    "p6": "excursion_stats.csv",
    "p8": "model.pkl",
}

# CANDIDATES are imported from _pipeline_constants — DO NOT duplicate here.
# Add idx locally for audit reporting purposes only.
from _pipeline_constants import CANDIDATES

# Local idx map (readonly view — values sourced from _pipeline_constants)
_CANDIDATES_WITH_IDX = [{"idx": i + 1, **c} for i, c in enumerate(CANDIDATES)]


def label_dir_name(c: dict) -> str:
    """Build Phase 7 label directory name from a candidate dict."""
    pt_s = str(c["pt_ticks"]).replace(".", "p")
    sl_s = str(c["sl_ticks"]).replace(".", "p")
    return f"phase7_labels_{c['vb_ticks']}ticks_{pt_s}_{sl_s}"


def parse_done(path: Path) -> dict:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    result = {}
    for line in content.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


@dataclass
class DayAudit:
    date: str
    phases: dict = field(default_factory=dict)
    phase_errors: dict = field(default_factory=dict)
    phase_times: dict = field(default_factory=dict)
    output_files: dict = field(default_factory=dict)
    p7_label_dirs: dict = field(default_factory=dict)
    depth_files: list = field(default_factory=list)
    # Computed flags
    highest_reached_phase: Optional[str] = None
    highest_valid_completed: Optional[str] = None
    p7_status: str = "not_started"
    p8_status: str = "not_started"
    ready_for_p7: bool = False
    ready_for_p8: bool = False
    has_usable_intermediates: bool = False
    current_failing_boundary: Optional[str] = None
    next_valid_action: str = "unknown"
    p1_failure_reason: str = ""


def audit_day(date: str) -> DayAudit:
    out_dir = OUT_DIR / date
    ckpt_dir = out_dir / "_checkpoints"

    audit = DayAudit(date=date, phases={}, phase_errors={}, phase_times={},
                     output_files={}, p7_label_dirs={})

    # Checkpoints
    for phase in PHASE_ORDER:
        done_file = ckpt_dir / f"{phase}.done"
        info = parse_done(done_file)
        status = info.get("status", "pending")
        error = info.get("error", "") if status == "failed" else ""
        mtime = done_file.stat().st_mtime if done_file.exists() else None
        audit.phases[phase] = status
        audit.phase_errors[phase] = error
        audit.phase_times[phase] = mtime

    # Output files
    for phase_key, fname in KEY_FILES.items():
        p = out_dir / fname
        if p.exists() and p.is_file():
            try:
                audit.output_files[phase_key] = p.stat().st_size
            except OSError:
                audit.output_files[phase_key] = None
        else:
            audit.output_files[phase_key] = None

    # P7 label dirs
    for c in _CANDIDATES_WITH_IDX:
        ldir = out_dir / label_dir_name(c)
        key = f"p7_c{c['idx']}"
        try:
            has_csv = any(f.suffix == ".csv" for f in ldir.iterdir()) if ldir.is_dir() else False
        except OSError:
            has_csv = False
        audit.p7_label_dirs[key] = ldir.is_dir() and has_csv

    # .depth files for this date
    import re
    date_parts = date.split("-")
    year, month, day_num = date_parts
    depth_pattern = f"*{year}{month}{day_num}*.depth"
    audit.depth_files = sorted(INPUT_DIR.rglob(depth_pattern))

    return audit


def compute_flags(audit: DayAudit) -> DayAudit:
    phases = audit.phases
    done_phases = [p for p in PHASE_ORDER if phases.get(p) == "done"]
    failed_phases = [p for p in PHASE_ORDER if phases.get(p) == "failed"]
    pending_phases = [p for p in PHASE_ORDER if phases.get(p) == "pending"]

    # highest_reached_phase = last phase with ANY status (done or failed)
    all_started = [p for p in PHASE_ORDER if p in phases and phases[p] != "pending"]
    audit.highest_reached_phase = all_started[-1] if all_started else None

    # P7 classification
    p7_done = {k: v for k, v in audit.p7_label_dirs.items()}
    p7_sentinels = {f"p7_c{i}": phases.get(f"p7_c{i}") == "done" for i in range(1, 4)}
    p7_failed_sentinels = {f"p7_c{i}": phases.get(f"p7_c{i}") == "failed" for i in range(1, 4)}

    p7_sentinel_done_count = sum(1 for v in p7_sentinels.values() if v)
    p7_label_dir_count = sum(1 for v in p7_done.values() if v)
    p7_any_failed = any(p7_failed_sentinels.values())
    p7_all_done_sentinel = all(p7_sentinels.values())
    p7_all_done_label = all(p7_done.values())

    if p7_all_done_sentinel and p7_all_done_label:
        audit.p7_status = "complete"
    elif p7_sentinel_done_count > 0 and p7_label_dir_count > 0 and not p7_any_failed:
        audit.p7_status = "complete"
    elif p7_sentinel_done_count > 0 or p7_label_dir_count > 0:
        audit.p7_status = "partial"  # some succeeded, some failed
    elif p7_any_failed:
        audit.p7_status = "retryable"  # none succeeded but some failed — retry might work
    else:
        audit.p7_status = "not_started"

    # P8 classification
    p8_done = audit.output_files.get("p8") is not None
    p8_sentinel_done = phases.get("p8_ml") == "done"
    p8_sentinel_failed = phases.get("p8_ml") == "failed"

    if p8_done and p8_sentinel_done:
        audit.p8_status = "complete"
    elif p8_sentinel_failed:
        if audit.p7_status == "complete":
            audit.p8_status = "failed_after_valid_p7"
        else:
            audit.p8_status = "failed_no_valid_p7"
    else:
        audit.p8_status = "not_started"

    # ready_for_p7 = P6 done and no P7 at all (not started)
    p6_done = phases.get("p6_excursion") == "done"
    p6_usable = audit.output_files.get("p6") is not None  # excursion_stats.csv exists
    p5_usable = audit.output_files.get("p5") is not None  # sampled_events.csv exists
    p4_usable = audit.output_files.get("p4") is not None  # features_dom_agg.csv exists
    p3_usable = audit.output_files.get("p3") is not None  # features_dom.csv exists

    audit.has_usable_intermediates = p6_usable or p5_usable or p4_usable or p3_usable

    audit.ready_for_p7 = p6_done and p6_usable and audit.p7_status == "not_started"
    audit.ready_for_p8 = audit.p7_status == "complete" and audit.p8_status == "not_started"

    # Determine next_valid_action
    if audit.p8_status in ("failed_after_valid_p7",):
        audit.next_valid_action = "rerun_p8_only"
    elif audit.p7_status == "retryable":
        audit.next_valid_action = "run_p7_candidates"
    elif audit.p7_status == "partial":
        audit.next_valid_action = "run_p7_candidates"
    elif audit.ready_for_p7:
        audit.next_valid_action = "run_p7_p8_incremental"
    elif p6_done and p6_usable and audit.p7_status == "not_started" and not audit.has_usable_intermediates:
        # P6 done but no excursion_stats? can't proceed
        audit.next_valid_action = "inspect_p6_output"
    elif phases.get("p6_excursion") == "failed":
        # P6 failed — check what we have
        audit.next_valid_action = "rerun_p6"
    elif phases.get("p5_sample") == "failed":
        if p4_usable:
            audit.next_valid_action = "rerun_p5_p6"
        else:
            audit.next_valid_action = "inspect_p4_output"
    elif phases.get("p4_agg") == "failed":
        if p3_usable:
            audit.next_valid_action = "rerun_p4_p6"
        else:
            audit.next_valid_action = "inspect_p3_output"
    elif phases.get("p3_features") == "failed":
        audit.next_valid_action = "inspect_p2_output"
    elif phases.get("p2_reconstruct") == "failed":
        audit.next_valid_action = "inspect_p1_input"
    elif phases.get("p1_parse") == "failed":
        audit.next_valid_action = "inspect_p1_input"
    elif phases.get("p1_parse") == "done" and not any(phases.values()):
        audit.next_valid_action = "inspect_early_phase"
    elif not any(phases.values()):
        audit.next_valid_action = "full_rerun"
    else:
        audit.next_valid_action = f"unknown_phase_state"

    # Check P1 failure reason
    if phases.get("p1_parse") == "failed":
        err = audit.phase_errors.get("p1_parse", "")
        if "events.csv not produced" in err or "events.csv missing" in err:
            # Check if events.csv actually exists (pipeline may have created it later)
            if audit.output_files.get("p1"):
                audit.p1_failure_reason = "parse_failed_despite_events_exists"
            else:
                # No .depth files found
                if not audit.depth_files:
                    audit.p1_failure_reason = "no_depth_files_found"
                else:
                    audit.p1_failure_reason = f"depth_files_exist_but_parse_failed_count={len(audit.depth_files)}"
        else:
            audit.p1_failure_reason = err or "unknown"

    # Current failing boundary
    if audit.p8_status == "failed_after_valid_p7":
        audit.current_failing_boundary = "p8_ml"
    elif audit.p7_status == "retryable":
        failed_p7 = [p for p in PHASE_ORDER if p.startswith("p7") and phases.get(p) == "failed"]
        audit.current_failing_boundary = failed_p7[-1] if failed_p7 else None
    elif phases.get("p6_excursion") == "failed":
        audit.current_failing_boundary = "p6_excursion"
    elif phases.get("p5_sample") == "failed":
        audit.current_failing_boundary = "p5_sample"
    elif phases.get("p4_agg") == "failed":
        audit.current_failing_boundary = "p4_agg"
    elif phases.get("p3_features") == "failed":
        audit.current_failing_boundary = "p3_features"
    elif phases.get("p2_reconstruct") == "failed":
        audit.current_failing_boundary = "p2_reconstruct"
    elif phases.get("p1_parse") == "failed":
        audit.current_failing_boundary = "p1_parse"

    return audit


def audit_all():
    day_dirs = sorted([
        d for d in OUT_DIR.iterdir()
        if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)
    ])

    results = []
    for day_dir in day_dirs:
        audit = audit_day(day_dir.name)
        audit = compute_flags(audit)
        results.append(audit)

    return results


def print_audit(results):
    print(f"\n{'DATE':<12} {'PHASE_STATUS':<80} {'REACHED':<12} {'VALID':<12} {'P7':<10} {'P8':<10} {'ACTION':<30}")
    print("-" * 160)
    for a in results:
        phase_str = " ".join(f"{p[-3:]}:{a.phases[p][:1]}" for p in PHASE_ORDER if a.phases.get(p) != "pending")
        reach = a.highest_reached_phase or "-"
        valid = a.highest_valid_completed or "-"
        print(f"{a.date:<12} {phase_str:<80} {reach:<12} {valid:<12} {a.p7_status:<10} {a.p8_status:<10} {a.next_valid_action:<30}")

    print("\n")
    print("=" * 120)
    print("P7 DETAIL:")
    print(f"{'DATE':<12} {'p7_c1_sent':<12} {'p7_c1_lbl':<12} {'p7_c2_sent':<12} {'p7_c2_lbl':<12} {'p7_c3_sent':<12} {'p7_c3_lbl':<12} {'P7_STATUS':<12}")
    print("-" * 100)
    for a in results:
        c1s = "done" if a.phases.get("p7_c1") == "done" else ("fail" if a.phases.get("p7_c1") == "failed" else "-")
        c2s = "done" if a.phases.get("p7_c2") == "done" else ("fail" if a.phases.get("p7_c2") == "failed" else "-")
        c3s = "done" if a.phases.get("p7_c3") == "done" else ("fail" if a.phases.get("p7_c3") == "failed" else "-")
        c1l = "Y" if a.p7_label_dirs.get("p7_c1") else "N"
        c2l = "Y" if a.p7_label_dirs.get("p7_c2") else "N"
        c3l = "Y" if a.p7_label_dirs.get("p7_c3") else "N"
        print(f"{a.date:<12} {c1s:<12} {c1l:<12} {c2s:<12} {c2l:<12} {c3s:<12} {c3l:<12} {a.p7_status:<12}")

    print("\n")
    print("=" * 120)
    print("OUTPUT FILES:")
    print(f"{'DATE':<12} {'p1_evts':<12} {'p2_snaps':<12} {'p3_fdom':<12} {'p4_agg':<12} {'p5_samp':<12} {'p6_excur':<12} {'p8_ml':<10}")
    print("-" * 100)
    for a in results:
        def f(key):
            sz = a.output_files.get(key)
            if sz is None: return "MISSING"
            if sz >= 1024**3: return f"{sz/1024**3:.1f}G"
            if sz >= 1024**2: return f"{sz/1024**2:.0f}M"
            return f"{sz/1024:.0f}K"
        print(f"{a.date:<12} {f('p1'):<12} {f('p2'):<12} {f('p3'):<12} {f('p4'):<12} {f('p5'):<12} {f('p6'):<12} {f('p8'):<10}")

    print("\n")
    print("=" * 120)
    print("P1 FAILURE INSPECTION:")
    print(f"{'DATE':<12} {'p1_status':<12} {'p1_error':<40} {'depth_files':<6} {'p1_failure_reason':<40}")
    print("-" * 130)
    for a in results:
        if a.phases.get("p1_parse") == "failed":
            p1s = a.phases.get("p1_parse", "-")
            p1e = a.phase_errors.get("p1_parse", "-")[:38]
            df_count = len(a.depth_files)
            reason = a.p1_failure_reason[:38]
            print(f"{a.date:<12} {p1s:<12} {p1e:<40} {df_count:<6} {reason:<40}")


if __name__ == "__main__":
    results = audit_all()
    print_audit(results)
