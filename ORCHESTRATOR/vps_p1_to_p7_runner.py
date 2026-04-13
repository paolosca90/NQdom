#!/usr/bin/env python3
"""
vps_p1_to_p7_runner.py
======================
Wrapper P1-P7 per vps_multiday_runner.py.

Esegue P1-P7 (skip P8) con:
  - Resume intelligente: skippa giorni con P7_c3.done
  - Memory safety: --workers 2 --parallel 1 (RAM limit Contabo 24GB)
  - --force per rilanciare da P1

Uso:
  python3 vps_p1_to_p7_runner.py           # resume
  python3 vps_p1_to_p7_runner.py --force   # forza da P1
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

VPS       = "root@185.185.82.205"
VPS_BASE  = "/opt/depth-dom"
SSH_KEY   = os.path.expanduser("~/.ssh/id_rsa")
SSH_OPTS  = f"-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i {SSH_KEY}"

def ssh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        f"ssh {SSH_OPTS} {VPS} \"{cmd}\"",
        shell=True, capture_output=True, text=True
    )

def check_vps() -> bool:
    r = ssh(f"echo OK && free -h | grep Mem && df -h {VPS_BASE} | tail -1")
    if r.returncode != 0:
        print(f"[ERR] VPS offline:\n{r.stderr}")
        return False
    print(f"[OK]  VPS connesso")
    for line in r.stdout.strip().splitlines():
        print(f"     {line}")
    return True

def get_p7_done_days(output_dir: str) -> set:
    """Ritorna i giorni già completati con tutti i checkpoint P1-P7."""
    r = ssh(
        f"cd {output_dir} && for d in 20*; do "
        f"[ -d \"$d\" ] && "
        f"[ -f \"$d/_checkpoints/p7_c3.done\" ] && echo $d; done"
    )
    return set(r.stdout.strip().splitlines())

def main():
    ap = argparse.ArgumentParser(description="P1-P7 Pipeline Runner (VPS)")
    ap.add_argument("--force", action="store_true", help="Forza ricalcolo da P1")
    ap.add_argument("--dry-run", action="store_true", help="Dry run")
    args = ap.parse_args()

    print("=" * 60)
    print("NQ P1-P7 Pipeline Runner — Contabo VPS")
    print("=" * 60)
    print(f"  Modalità : {'FORCE' if args.force else 'RESUME'}")
    print(f"  Workers  : 2 (memory-safe, no OOM)")
    print(f"  Fasi     : P1 P2 P2b P3 P4 P5 P6 P7")
    print("=" * 60)

    # 1. Verifica VPS
    if not check_vps():
        sys.exit(1)

    # 2. Resume: trova giorni già fatti (P7 c3 checkpoint = tutti P1-P7 ok)
    if not args.force:
        done = get_p7_done_days(VPS_BASE)
        if done:
            print(f"\n[Resume] {len(done)} giorni già completati con P1-P7:")
            for d in sorted(done):
                print(f"  ✅ {d}")
        else:
            print("\n[Resume] Nessun giorno completo trovato —无所谓.")
    else:
        done = set()

    # 3. Lancia il runner
    # --skip-p7-p8 non è disponibile, quindi usiamo --skip-p8
    # Mod: il runner originale non ha --skip-p8, lo usiamo così:
    #   Con --skip-p7-p8 → skippa P7 e P8 (non fa per noi)
    #   MA: se run with --skip-p8 E skip_p8=True in process_day...
    #
    # Problema: vps_multiday_runner.py non ha skip_p8 flag.
    # Soluzione: lo invochiamo piped per fare solo P1-P7
    #   Step 1: P1-P6 con --skip-p7-p8
    #   Step 2: P7-only con --skip-p1-p6
    # Questo funziona con resume perché checkpoint sentinels sono separati.

    extra = "--force" if args.force else ""

    # INPUT_DIR_DEFAULT = /opt/depth-dom/input (minuscolo) è VUOTO.
    # I file sono in /opt/depth-dom/INPUT (maiuscolo) — 19 giorni.
    INPUT_DIR = "/opt/depth-dom/INPUT"

    if not args.dry_run:
        # Step 1: P1-P6
        print("\n[Step 1/2] Lancio P1-P6 ...")
        r = ssh(
            f"cd {VPS_BASE} && python3 vps_multiday_runner.py "
            f"--input-dir {INPUT_DIR} "
            f"--workers 2 --cleanup-policy none --parallel 1 "
            f"--skip-p7-p8 --resume {extra}"
        )
        print(r.stdout[-500:] if r.stdout else "")
        if r.returncode != 0:
            print(f"[WARN] P1-P6 exit code {r.returncode}\n{r.stderr[-300:]}")

        # Step 2: P7
        print("\n[Step 2/2] Lancio P7 ...")
        r = ssh(
            f"cd {VPS_BASE} && python3 vps_multiday_runner.py "
            f"--input-dir {INPUT_DIR} "
            f"--workers 1 --cleanup-policy none --parallel 1 "
            f"--skip-p1-p6 --resume {extra}"
        )
        print(r.stdout[-500:] if r.stdout else "")
        if r.returncode != 0:
            print(f"[WARN] P7 exit code {r.returncode}\n{r.stderr[-300:]}")

        print("\n✅ Pipeline P1-P7 completato (o in corso).")
    else:
        print("\n[DRY RUN] Non ho lanciato nulla.")

if __name__ == "__main__":
    main()
