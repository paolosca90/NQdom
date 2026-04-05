#!/usr/bin/env bash
# ==============================================================================
# vps_watchdog.sh — Lightweight Pipeline Watchdog
# ==============================================================================
# Runs from cron (every 5 min) or manually.
# Refreshes live status and checks for operational anomalies.
#
# Crontab entry (adjust path as needed):
#   */5 * * * * /bin/bash /opt/depth-dom/vps_watchdog.sh >> /opt/depth-dom/logs/watchdog.log 2>&1
#
# Exit codes:
#   0  = normal (quiet if nothing to report)
#   1  = operational issues detected
#   2  = duplicate runners detected (requires attention)
#   3  = system error
# ==============================================================================

set -uo pipefail

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_DIR="/opt/depth-dom/logs"
OUT_DIR="/opt/depth-dom/output"
STATUS_SCRIPT="/opt/depth-dom/status_live.py"
LIVE_STATUS="${OUT_DIR}/_live_status.json"
WATCHDOG_LOG="${LOG_DIR}/watchdog.log"
PID_FILE="/tmp/depth_dom_watchdog.pid"
ALERT_FILE="/tmp/depth_dom_watchdog.alerts"

# Thresholds
DUPLICATE_THRESHOLD=2          # Alert if >N runners of same type
STALLED_THRESHOLD_MIN=30      # Alert if day stalled >30 min without heartbeat
HUNG_THRESHOLD_MIN=60         # Alert if day running >60 min without progress

mkdir -p "${LOG_DIR}"

log() {
    echo "[${TIMESTAMP}] $*"
}

# ── Lockfile (prevent concurrent runs) ────────────────────────────────────────
acquire_lock() {
    if [[ -f "${PID_FILE}" ]]; then
        OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || echo "")
        if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
            log "[SKIP] Previous watchdog (PID ${OLD_PID}) still running"
            exit 0
        fi
        log "[WARN] Stale watchdog lock (PID ${OLD_PID}) — removing"
        rm -f "${PID_FILE}"
    fi
    echo $$ > "${PID_FILE}"
    trap 'rm -f "${PID_FILE}"' EXIT INT TERM
}

# ── Refresh live status ────────────────────────────────────────────────────────
# status_live.py exit codes:
#   0 = all clear (no failures)
#   1 = operational issues found (normal when pipeline has failures)
#   2+ = script error
refresh_status() {
    log "[INFO] Refreshing live status..."
    set +e
    python3 "${STATUS_SCRIPT}" --json-only >> "${WATCHDOG_LOG}" 2>&1
    exit_code=$?
    set -e
    if [[ ${exit_code} -eq 0 ]]; then
        log "[INFO] Live status updated: ${LIVE_STATUS}"
        return 0
    elif [[ ${exit_code} -eq 1 ]]; then
        log "[INFO] Live status generated — pipeline has failures (normal operational state)"
        return 0
    else
        log "[ERROR] status_live.py failed with exit code ${exit_code}"
        return 1
    fi
}

# ── Parse live status and emit alerts ────────────────────────────────────────
check_alerts() {
    if [[ ! -f "${LIVE_STATUS}" ]]; then
        log "[WARN] No live status file yet: ${LIVE_STATUS}"
        return 0
    fi

    local alert_count=0
    > "${ALERT_FILE}"  # clear alerts

    # ── 1. Check for duplicate runners ─────────────────────────────────────
    local duplicates=$(python3 -c "
import json
with open('${LIVE_STATUS}') as f:
    d = json.load(f)
dupes = d.get('duplicate_runners', [])
print(len(dupes))
" 2>/dev/null || echo "0")

    if [[ "${duplicates}" -gt 0 ]]; then
        log "[ALERT] ${duplicates} duplicate/conflicting runner(s) detected!"
        python3 -c "
import json
with open('${LIVE_STATUS}') as f:
    d = json.load(f)
for dup in d.get('duplicate_runners', []):
    print(f'  [{dup[\"issue\"]}] PID={dup[\"pid\"]} kind={dup[\"kind\"]} elapsed={dup[\"elapsed_min\"]:.0f}min')
" >> "${ALERT_FILE}" 2>/dev/null
        ((alert_count++))
    fi

    # ── 2. Check for hung / stalled days ─────────────────────────────────────
    local hung_count=$(python3 -c "
import json, datetime as dt
with open('${LIVE_STATUS}') as f:
    d = json.load(f)
hung = [day for day in d['days'] if day['state'] in ('hung', 'stalled')]
print(len(hung))
" 2>/dev/null || echo "0")

    if [[ "${hung_count}" -gt 0 ]]; then
        log "[ALERT] ${hung_count} hung/stalled day(s) detected!"
        python3 -c "
import json
with open('${LIVE_STATUS}') as f:
    d = json.load(f)
for day in d['days']:
    if day['state'] in ('hung', 'stalled'):
        print(f\"  [{day['date']}] {day['state']}: {day['reason']} | action: {day['recommended_action']}\")
" >> "${ALERT_FILE}" 2>/dev/null
        ((alert_count++))
    fi

    # ── 3. Check for retry loops ────────────────────────────────────────────
    local retry_count=$(python3 -c "
import json
with open('${LIVE_STATUS}') as f:
    d = json.load(f)
retry = [day for day in d['days'] if day['state'] == 'retry_loop']
print(len(retry))
" 2>/dev/null || echo "0")

    if [[ "${retry_count}" -gt 0 ]]; then
        log "[ALERT] ${retry_count} retry-loop day(s) detected!"
        python3 -c "
import json
with open('${LIVE_STATUS}') as f:
    d = json.load(f)
for day in d['days']:
    if day['state'] == 'retry_loop':
        print(f\"  [{day['date']}] retry loop at {day['error_phase']}: {day['error_message']}\")
" >> "${ALERT_FILE}" 2>/dev/null
        ((alert_count++))
    fi

    # ── 4. Summary to watchdog log ───────────────────────────────────────────
    local total=$(python3 -c "
import json
with open('${LIVE_STATUS}') as f:
    d = json.load(f)
counts = {}
for day in d['days']:
    st = day['state']
    counts[st] = counts.get(st, 0) + 1
print(' '.join(f'{k}={v}' for k,v in sorted(counts.items())))
" 2>/dev/null || echo "unknown")

    local ready=$(python3 -c "
import json
with open('${LIVE_STATUS}') as f:
    d = json.load(f)
ready = [day['date'] for day in d['days'] if day['state'] == 'ready_p7p8']
print(','.join(ready) if ready else 'none')
" 2>/dev/null || echo "unknown")

    log "[INFO] Pipeline summary: ${total}"
    log "[INFO] Ready for P7/P8: ${ready}"

    if [[ -s "${ALERT_FILE}" ]]; then
        log "[ALERT] See ${ALERT_FILE} for details"
    fi

    return ${alert_count}
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    acquire_lock
    log "[START] Watchdog run"

    if ! refresh_status; then
        log "[ERROR] Failed to refresh status"
        exit 3
    fi

    if ! check_alerts; then
        exit_code=$?
        if [[ ${exit_code} -ge 2 ]]; then
            log "[ALERT] Critical: duplicate runners detected — manual intervention required"
            exit 2
        fi
        log "[WARN] Operational issues detected — see above"
        exit 1
    fi

    log "[DONE] Watchdog run complete — all clear"
    exit 0
}

main "$@"
