#!/usr/bin/env python3
"""
vps_sequential_phases.py
========================
Esegue TUTTE le fasi in ordine sequenziale per tutti i 19 giorni:
  P1 -> P2 -> P2b -> P3 -> P4 -> P5 -> P6 -> P7
Per ogni fase, processa tutti i giorni prima di passare alla fase successiva.

Uso (locale):
  python vps_sequential_phases.py              # resume (skip se .done esiste)
  python vps_sequential_phases.py --force    # forza ricalcolo

Lo script viene copiato sul VPS e lanciato la.
"""

import argparse
import subprocess
import sys
from pathlib import Path

VPS      = "root@185.185.82.205"
VPS_BASE = "/opt/depth-dom"
# SSH key locale su Windows (passata a SSH locale, non al VPS)
_SSH_KEY_STR = "/c/Users/Paolo/.ssh/id_rsa"
SSH_OPTS = f"-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i {_SSH_KEY_STR}"
INPUT_DIR = "/opt/depth-dom/INPUT"          # maiuscolo, contiene i 19 giorni

DAYS = [
    "2026-03-13", "2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19",
    "2026-03-20", "2026-03-23", "2026-03-24", "2026-03-25", "2026-03-26",
    "2026-03-27", "2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02",
    "2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09",
]

# -- helpers ------------------------------------------------------------------

def ssh_cmd(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        f"ssh {SSH_OPTS} {VPS} \"{cmd}\"",
        shell=True, capture_output=True, text=True
    )

def sftp_upload(local_path: Path, remote_path: str):
    subprocess.run(
        f"scp {SSH_OPTS} {local_path} {VPS}:{remote_path}",
        shell=True, capture_output=True
    )

def log(msg: str):
    print(msg)
    sys.stdout.flush()

def run_phase_on_day(ssh_cmd_fn, phase_name: str, day: str, cmd: str) -> bool:
    """Esegue cmd per un giorno. Return True se skip, False se eseguito."""
    r = ssh_cmd_fn(cmd)
    if r.returncode == 0:
        log(f"  ? {day} {phase_name}")
        return False
    else:
        log(f"  ? {day} {phase_name}: {r.stderr[:200]}")
        return False

def check_done(day: str, phase_tag: str) -> bool:
    """Resume: true SE il checkpoint esiste E status=done (non failed)."""
    ck = f"{VPS_BASE}/output/{day}/_checkpoints/{phase_tag}.done"
    r = ssh_cmd(f"[ -f {ck} ] && grep -q 'status=done' {ck} && echo YES || echo NO")
    return r.stdout.strip() == "YES"

def touch_done(day: str, phase_tag: str):
    """Crea il sentinel .done."""
    ck_dir = f"{VPS_BASE}/output/{day}/_checkpoints"
    ck = f"{ck_dir}/{phase_tag}.done"
    ssh_cmd(f"mkdir -p {ck_dir} && echo 'status=done' > {ck}")

# -- fasi ---------------------------------------------------------------------

def phase_p1(force: bool):
    log("\n" + "="*60)
    log("FASE P1 -> events.csv (19 giorni)")
    log("="*60)
    for day in DAYS:
        if not force and check_done(day, "p1_parse"):
            log(f"  ?  {day} P1 -- skip ( gia fatto)")
            continue
        log(f"  ?  {day} P1...")
        r = ssh_cmd(
            f"cd {VPS_BASE}/P1 && python3 main.py "
            f"--input-dir {INPUT_DIR} --days {day} "
            + ("--force" if force else "")
        )
        if r.returncode == 0:
            touch_done(day, "p1_parse")
            log(f"  ? {day} P1 completato")
        else:
            log(f"  ? {day} P1 FALLITO:\n{r.stderr[-300:]}")


def phase_p2(force: bool):
    log("\n" + "="*60)
    log("FASE P2 -- events.csv -> snapshots.csv (19 giorni)")
    log("="*60)
    for day in DAYS:
        events = f"{VPS_BASE}/output/{day}/events.csv"
        snaps  = f"{VPS_BASE}/output/{day}/snapshots.csv"
        if not force and check_done(day, "p2_reconstruct"):
            log(f"  ?  {day} P2 -- skip ( gia fatto)")
            continue
        # skip se events.csv non esiste
        r = ssh_cmd(f"[ -f {events} ] && echo OK || echo MISSING")
        if r.stdout.strip() != "OK":
            log(f"  ?  {day} P2 -- events.csv manca, skip")
            continue
        log(f"  ?  {day} P2...")
        r = ssh_cmd(
            f"cd {VPS_BASE} && python3 P2/vps_book_reconstructor.py "
            f"--input {events} --output {snaps}"
            + (" --force" if force else "")
        )
        if r.returncode == 0:
            touch_done(day, "p2_reconstruct")
            log(f"  ? {day} P2 completato")
        else:
            log(f"  ? {day} P2 FALLITO:\n{r.stderr[-300:]}")


def phase_p2b(force: bool):
    log("\n" + "="*60)
    log("FASE P2b -- snapshots.csv + trades.csv -> snapshots_fused.csv (19 giorni)")
    log("="*60)
    for day in DAYS:
        snaps_in  = f"{VPS_BASE}/output/{day}/snapshots.csv"
        trades    = f"{VPS_BASE}/output_TS/by_day/{day}_trades.csv"
        fused    = f"{VPS_BASE}/output/{day}/snapshots_fused.csv"
        if not force and check_done(day, "p2b_fusion"):
            log(f"  ?  {day} P2b -- skip ( gia fatto)")
            continue
        r = ssh_cmd(f"[ -f {snaps_in} ] && echo OK || echo MISSING")
        if r.stdout.strip() != "OK":
            log(f"  ?  {day} P2b -- snapshots.csv manca, skip")
            continue
        r = ssh_cmd(f"[ -f {trades} ] && echo OK || echo MISSING")
        if r.stdout.strip() != "OK":
            log(f"  ?  {day} P2b -- trades.csv manca, skip (P2b richiede Time&Sales)")
            continue
        log(f"  ?  {day} P2b...")
        r = ssh_cmd(
            f"cd {VPS_BASE}/P2b && python3 vps_phase2b_data_fusion.py "
            f"--snapshots {snaps_in} --trades {trades} --output {fused}"
            + (" --force" if force else "")
        )
        if r.returncode == 0:
            touch_done(day, "p2b_fusion")
            log(f"  ? {day} P2b completato")
        else:
            log(f"  ? {day} P2b FALLITO:\n{r.stderr[-300:]}")


def phase_p3(force: bool):
    log("\n" + "="*60)
    log("FASE P3 -- snapshots.csv -> features_dom.csv (19 giorni)")
    log("="*60)
    for day in DAYS:
        snaps  = f"{VPS_BASE}/output/{day}/snapshots.csv"
        feats  = f"{VPS_BASE}/output/{day}/features_dom.csv"
        if not force and check_done(day, "p3_features"):
            log(f"  ?  {day} P3 -- skip ( gia fatto)")
            continue
        r = ssh_cmd(f"[ -f {snaps} ] && echo OK || echo MISSING")
        if r.stdout.strip() != "OK":
            log(f"  ?  {day} P3 -- snapshots.csv manca, skip")
            continue
        log(f"  ?  {day} P3...")
        r = ssh_cmd(
            f"cd {VPS_BASE}/P3 && python3 vps_feature_engineering_vectorized.py "
            f"--input {snaps} --output {feats}"
            + (" --force" if force else "")
        )
        if r.returncode == 0:
            touch_done(day, "p3_features")
            log(f"  ? {day} P3 completato")
        else:
            log(f"  ? {day} P3 FALLITO:\n{r.stderr[-300:]}")


def phase_p4(force: bool):
    log("\n" + "="*60)
    log("FASE P4 -- features_dom.csv -> features_dom_agg.csv (19 giorni)")
    log("="*60)
    for day in DAYS:
        feats  = f"{VPS_BASE}/output/{day}/features_dom.csv"
        agg    = f"{VPS_BASE}/output/{day}/features_dom_agg.csv"
        if not force and check_done(day, "p4_agg"):
            log(f"  ?  {day} P4 -- skip ( gia fatto)")
            continue
        r = ssh_cmd(f"[ -f {feats} ] && echo OK || echo MISSING")
        if r.stdout.strip() != "OK":
            log(f"  ?  {day} P4 -- features_dom.csv manca, skip")
            continue
        log(f"  ?  {day} P4...")
        r = ssh_cmd(
            f"cd {VPS_BASE}/P4 && python3 vps_feature_engineering_agg.py "
            f"--input {feats} --output {agg}"
            + (" --force" if force else "")
        )
        if r.returncode == 0:
            touch_done(day, "p4_agg")
            log(f"  ? {day} P4 completato")
        else:
            log(f"  ? {day} P4 FALLITO:\n{r.stderr[-300:]}")


def phase_p5(force: bool):
    log("\n" + "="*60)
    log("FASE P5 -- CUSUM sampling (19 giorni)")
    log("="*60)
    for day in DAYS:
        feats   = f"{VPS_BASE}/output/{day}/features_dom.csv"
        agg     = f"{VPS_BASE}/output/{day}/features_dom_agg.csv"
        sampled = f"{VPS_BASE}/output/{day}/sampled_events.csv"
        if not force and check_done(day, "p5_cusum"):
            log(f"  ?  {day} P5 -- skip ( gia fatto)")
            continue
        r = ssh_cmd(f"[ -f {feats} ] && echo OK || echo MISSING")
        if r.stdout.strip() != "OK":
            log(f"  ?  {day} P5 -- features_dom.csv manca, skip")
            continue
        log(f"  ?  {day} P5...")
        r = ssh_cmd(
            f"cd {VPS_BASE}/P5 && python3 vps_cusum_sampler.py "
            f"--features {feats} --agg {agg} --output {sampled}"
            + (" --force" if force else "")
        )
        if r.returncode == 0:
            touch_done(day, "p5_cusum")
            log(f"  ? {day} P5 completato")
        else:
            log(f"  ? {day} P5 FALLITO:\n{r.stderr[-300:]}")


def phase_p6(force: bool):
    log("\n" + "="*60)
    log("FASE P6 -- Excursion analysis (19 giorni)")
    log("="*60)
    for day in DAYS:
        feats   = f"{VPS_BASE}/output/{day}/features_dom.csv"
        sampled = f"{VPS_BASE}/output/{day}/sampled_events.csv"
        excursion = f"{VPS_BASE}/output/{day}/excursion_stats.csv"
        summary   = f"{VPS_BASE}/output/{day}/excursion_summary.csv"
        plot      = f"{VPS_BASE}/output/{day}/excursion_distributions.png"
        if not force and check_done(day, "p6_excursion"):
            log(f"  ?  {day} P6 -- skip ( gia fatto)")
            continue
        r = ssh_cmd(f"[ -f {sampled} ] && echo OK || echo MISSING")
        if r.stdout.strip() != "OK":
            log(f"  ?  {day} P6 -- sampled_events.csv manca, skip")
            continue
        log(f"  ?  {day} P6...")
        r = ssh_cmd(
            f"cd {VPS_BASE}/P6 && python3 vps_excursion_analysis_vectorized.py "
            f"--features {feats} --sampled {sampled} "
            f"--output {excursion} --summary {summary} --plot {plot}"
            + (" --force" if force else "")
        )
        if r.returncode == 0:
            touch_done(day, "p6_excursion")
            log(f"  ? {day} P6 completato")
        else:
            log(f"  ? {day} P6 FALLITO:\n{r.stderr[-300:]}")


def phase_p7(force: bool):
    log("\n" + "="*60)
    log("FASE P7 -- Triple-barrier labeling (19 giorni)")
    log("="*60)
    # verifica che _candidates_3.csv esista
    grid = f"{VPS_BASE}/output/_candidates_3.csv"
    r = ssh_cmd(f"[ -f {grid} ] && echo OK || echo MISSING")
    if r.stdout.strip() != "OK":
        log(f"  ? P7 -- {grid} manca! Crealo con SHARED/_pipeline_constants.py")
        return

    for day in DAYS:
        snaps   = f"{VPS_BASE}/output/{day}/snapshots.csv"
        sampled = f"{VPS_BASE}/output/{day}/sampled_events.csv"
        refprice= f"{VPS_BASE}/output/{day}/excursion_stats.csv"
        out_dir = f"{VPS_BASE}/output/{day}"
        if not force and check_done(day, "p7_c1"):
            log(f"  ?  {day} P7 -- skip ( gia fatto)")
            continue
        r = ssh_cmd(f"[ -f {sampled} ] && echo OK || echo MISSING")
        if r.stdout.strip() != "OK":
            log(f"  ?  {day} P7 -- sampled_events.csv manca, skip")
            continue
        log(f"  ?  {day} P7...")
        r = ssh_cmd(
            f"cd {VPS_BASE}/P7 && python3 vps_phase7_labeling.py "
            f"--snapshots {snaps} --sampled {sampled} "
            f"--refprice {refprice} --grid {grid} --output {out_dir} "
            f"--candidates 10"
            + (" --force" if force else "")
        )
        if r.returncode == 0:
            touch_done(day, "p7_c1")
            touch_done(day, "p7_c2")
            touch_done(day, "p7_c3")
            log(f"  ? {day} P7 completato")
        else:
            log(f"  ? {day} P7 FALLITO:\n{r.stderr[-300:]}")


# -- main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pipeline sequenziale P1-P7 (19 giorni)")
    ap.add_argument("--force", action="store_true", help="Forza ricalcolo anche se .done esiste")
    ap.add_argument("--start-phase", type=str, default="p1",
                    choices=["p1","p2","p2b","p3","p4","p5","p6","p7"],
                    help="Fase da cui iniziare (default: p1)")
    args = ap.parse_args()

    log("="*60)
    log("NQ SEQUENTIAL PIPELINE -> P1->P2->P2b->P3->P4->P5->P6->P7")
    log("="*60)
    log(f"  VPS      : {VPS}")
    log(f"  Base     : {VPS_BASE}")
    log(f"  Input    : {INPUT_DIR}")
    log(f"  Giorni   : {len(DAYS)}")
    log(f"  Modalita : {'FORCE' if args.force else 'RESUME'}")
    log(f"  Parte da : {args.start_phase}")
    log("="*60)

    # verifica connessione VPS
    r = ssh_cmd("echo OK && free -h | grep Mem")
    if r.returncode != 0:
        log(f"[ERR] VPS non raggiungibile: {r.stderr}")
        sys.exit(1)
    log(f"[OK] VPS connesso -- {r.stdout.strip().splitlines()[1]}")

    # sequenza fasi
    phases = [
        ("p1",  phase_p1),
        ("p2",  phase_p2),
        ("p2b", phase_p2b),
        ("p3",  phase_p3),
        ("p4",  phase_p4),
        ("p5",  phase_p5),
        ("p6",  phase_p6),
        ("p7",  phase_p7),
    ]

    started = False
    for name, fn in phases:
        if not started:
            if name == args.start_phase:
                started = True
            else:
                log(f"\n[SEL] Salto {name} (parte da {args.start_phase})")
                continue
        fn(args.force)

    log("\n" + "="*60)
    log("? PIPELINE SEQUENZIALE COMPLETATO")
    log("="*60)

if __name__ == "__main__":
    main()
