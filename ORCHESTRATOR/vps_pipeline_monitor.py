#!/usr/bin/env python3
"""
vps_pipeline_monitor.py
======================
Legge il manifest delle pipeline SU QUESTA MACCHINA (VPS) e invia
un report su Telegram ogni 5 minuti.

Cron (sul VPS):
  */5 * * * * /bin/bash /opt/depth-dom/pipeline_monitor.sh >> /opt/depth-dom/logs/pipeline_monitor.log 2>&1

Il wrapper pipeline_monitor.sh imposta TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID
prima di chiamare questo script.
"""

import csv
import datetime as dt
import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path

# ── Config (impostati dal wrapper) ───────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Paths (VPS locale) ───────────────────────────────────────
VPS_BASE  = "/opt/depth-dom"
VPS_OUT   = f"{VPS_BASE}/output"
MANIFEST  = f"{VPS_OUT}/_p1p7_manifest.csv"
LOCKFILE  = f"{VPS_BASE}/.pipeline_monitor.lock"

PHASE_COLS = [
    "phase_p1_parse", "phase_p2_reconstruct", "phase_p2b_fusion",
    "phase_p3_features", "phase_p4_agg", "phase_p5_sample",
    "phase_p6_excursion", "phase_p7_c1", "phase_p7_c2", "phase_p7_c3",
]

PHASE_LABELS = {
    "phase_p1_parse":       "P1",
    "phase_p2_reconstruct":  "P2",
    "phase_p2b_fusion":     "P2b",
    "phase_p3_features":    "P3",
    "phase_p4_agg":         "P4",
    "phase_p5_sample":      "P5",
    "phase_p6_excursion":   "P6",
    "phase_p7_c1":          "P7-C1",
    "phase_p7_c2":          "P7-C2",
    "phase_p7_c3":          "P7-C3",
}

STATUS_ICON = {
    "done":       "✅",
    "complete":   "✅",
    "running":    "🔄",
    "pending":    "⏳",
    "failed":     "❌",
    "exception":  "💥",
}

# ── Lock ────────────────────────────────────────────────────

def is_locked() -> bool:
    try:
        with open(LOCKFILE) as f:
            pid = int(f.read().strip())
        # Check se il pid esiste ancora
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False

def acquire_lock():
    with open(LOCKFILE, "w") as f:
        f.write(str(os.getpid()))

def release_lock():
    try:
        os.unlink(LOCKFILE)
    except FileNotFoundError:
        pass

# ── Manifest reader ──────────────────────────────────────────

def parse_manifest() -> list[dict]:
    if not os.path.exists(MANIFEST):
        return []
    try:
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"[ERR] parse_manifest: {e}")
        return []

# ── Memory ──────────────────────────────────────────────────

def get_vps_memory() -> str:
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        meminfo = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                meminfo[parts[0]] = int(parts[1]) * 1024  # KB → bytes
        total = meminfo.get("MemTotal:", 0)
        avail = meminfo.get("MemAvailable:", 0)
        used  = total - avail
        def gb(b): return b / (1024**3)
        return f"{gb(used):.1f}GB / {gb(total):.1f}GB"
    except Exception:
        return "?"

# ── Formatter ───────────────────────────────────────────────

def fmt_status(s: str) -> str:
    return STATUS_ICON.get(s.lower().strip(), s)

def fmt_day(row: dict) -> str:
    date   = row.get("date", "?")
    status = row.get("status", "?")
    err    = row.get("error_message", "")

    phase_parts = []
    for col in PHASE_COLS:
        v   = row.get(col, "pending").lower()
        lbl = PHASE_LABELS.get(col, col)
        ico = STATUS_ICON.get(v, "⭕")
        phase_parts.append(f"{ico}{lbl}")

    phase_str = " | ".join(phase_parts)
    ico = STATUS_ICON.get(status.lower(), "❓")
    line = f"{ico} *{date}*\n   {phase_str}"
    if err and status.lower() in ("failed", "exception"):
        line += f"\n   ⚠️ {err[:80]}"
    return line

def build_message(rows: list[dict], mem: str) -> str:
    now   = dt.datetime.now().strftime("%H:%M:%S")
    total = len(rows)
    done  = sum(1 for r in rows if r.get("status","").lower() in ("done","complete"))
    running = sum(1 for r in rows if r.get("status","").lower() == "running")
    failed  = sum(1 for r in rows if r.get("status","").lower() in ("failed","exception"))

    lines = [
        f"📊 *NQ Pipeline Monitor* `{now}`",
        f"🖥️  RAM: `{mem}`",
        "",
        f"✅ {done}/{total} giorni completi",
    ]
    if running:
        lines.append(f"🔄 {running} in corso")
    if failed:
        lines.append(f"❌ {failed} falliti")
    lines.append("")
    lines.append("── Dettaglio ──")
    for row in rows:
        lines.append(fmt_day(row))
    return "\n".join(lines)

# ── Telegram sender ──────────────────────────────────────────

def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID non impostati.")
        print(f"[TEXT]\n{text[:300]}")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "Markdown",
    }).encode()
    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return b'"ok":true' in resp.read()
    except Exception as e:
        print(f"[ERR] Telegram: {e}")
        return False

# ── Main ────────────────────────────────────────────────────

def main():
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Pipeline monitor start")

    if is_locked():
        print("[SKIP] Job precedente ancora in esecuzione.")
        sys.exit(0)

    acquire_lock()
    try:
        rows = parse_manifest()
        if not rows:
            print("[WARN] Manifest vuoto o non trovato.")
            sys.exit(0)

        mem  = get_vps_memory()
        msg  = build_message(rows, mem)
        print(f"Report:\n{msg}\n")

        ok = send_telegram(msg)
        print(f"Telegram: {'OK' if ok else 'FALLITO'}")
    finally:
        release_lock()

if __name__ == "__main__":
    main()
