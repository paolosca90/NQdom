#!/usr/bin/env bash
# ==============================================================================
# cron_orchestrator.sh — Cron-based Pipeline Orchestrator
# ==============================================================================
# Runs every 30 minutes via cron:
#   */30 * * * * /bin/bash /opt/depth-dom/cron_orchestrator.sh
#
# Steps:
#   1. incremental_p7p8_runner.py --workers 4
#   2. aggregate_results.py  (only if new P8 days completed)
#   3. plot_dashboard.py      (only if aggregate CSVs updated)
#
# Log: /opt/depth-dom/logs/cron_orchestrator.log (rotated if > 50 MB, keeps 3)
# Summary: /opt/depth-dom/logs/cron_last_run.txt
# Lockfile: /opt/depth-dom/cron_orchestrator.lock  (prevents concurrent runs)
# ==============================================================================

set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR="/opt/depth-dom"
OUTPUT_DIR="${BASE_DIR}/output"
LOG_DIR="${BASE_DIR}/logs"
LOCK_FILE="${BASE_DIR}/cron_orchestrator.lock"
AGG_DIR="${OUTPUT_DIR}/aggregate"
MANIFEST="${OUTPUT_DIR}/_p7p8_incremental_manifest.csv"
LOG_FILE="${LOG_DIR}/cron_orchestrator.log"
LAST_RUN="${LOG_DIR}/cron_last_run.txt"
MAX_LOG_SIZE=$((50 * 1024 * 1024))  # 50 MB

SYMBOL="NQ"
WORKERS=4
PYTHON="python3"

INCR_RUNNER="${BASE_DIR}/incremental_p7p8_runner.py"
AGG_RUNNER="${BASE_DIR}/aggregate_results.py"
PLOTTER="${BASE_DIR}/plot_dashboard.py"
# ───────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date '+%Y-%m-%d %H:%M')

# ── Initial check ─────────────────────────────────────────────────────────────
if [[ ! -d "$BASE_DIR" ]]; then
    echo "[ERROR] BASE_DIR $BASE_DIR not found. Aborting." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${AGG_DIR}"

# ── Logging ───────────────────────────────────────────────────────────────────

log() {
    echo "[${TIMESTAMP}] $*"
}

log_tee() {
    echo "[${TIMESTAMP}] $*" | tee -a "${LOG_FILE}"
}

# ── Log rotation ─────────────────────────────────────────────────────────────

rotate_log() {
    if [[ -f "${LOG_FILE}" ]]; then
        SIZE=$(stat -c%s "${LOG_FILE}" 2>/dev/null || echo 0)
        if (( SIZE > MAX_LOG_SIZE )); then
            # Rotate: keep last 3
            for i in 2 1; do
                if [[ -f "${LOG_FILE}.${i}" ]]; then
                    mv "${LOG_FILE}.${i}" "${LOG_FILE}.$((i+1))"
                fi
            done
            mv "${LOG_FILE}" "${LOG_FILE}.1"
            touch "${LOG_FILE}"
            log_tee "[ROTATE] Log rotated - exceeded 50MB, keeping last 3"
        fi
    fi
}

# ── Lockfile (prevent concurrent runs) ────────────────────────────────────────

acquire_lock() {
    if [[ -f "${LOCK_FILE}" ]]; then
        OLD_PID=$(cat "${LOCK_FILE}" 2>/dev/null || echo "")
        if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
            log_tee "[SKIP] Previous instance (PID ${OLD_PID}) still running"
            exit 0
        fi
        log_tee "[WARN] Stale lockfile (PID ${OLD_PID} not running) — removing"
        rm -f "${LOCK_FILE}"
    fi
    echo $$ > "${LOCK_FILE}"
    trap 'rm -f "${LOCK_FILE}"; exit' INT TERM EXIT
}

# ── Read previous state ────────────────────────────────────────────────────────

prev_p8_count() {
    if [[ -f "${LAST_RUN}" ]]; then
        grep -oP 'total_p8=\K\d+' "${LAST_RUN}" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

prev_agg_mtime() {
    if [[ -f "${AGG_DIR}/daily_metrics.csv" ]]; then
        stat -c %Y "${AGG_DIR}/daily_metrics.csv" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# ── Main ───────────────────────────────────────────────────────────────────────

main() {
    rotate_log
    acquire_lock

    log_tee "=== Cron orchestrator started ==="

    # ── Step 1: Run incremental P7/P8 runner ───────────────────────────────────
    log_tee "[STEP1] Running incremental_p7p8_runner.py ..."

    PREV_MF_MTIME=$(stat -c %Y "${MANIFEST}" 2>/dev/null || echo "0")

    ${PYTHON} "${INCR_RUNNER}" \
        --output-dir "${OUTPUT_DIR}" \
        --workers ${WORKERS} \
        >> "${LOG_FILE}" 2>&1

    NEW_MF_MTIME=$(stat -c %Y "${MANIFEST}" 2>/dev/null || echo "0")

    # Count new P8 completions
    NEW_P8=0
    TOTAL_P8=0
    BEST_F1="N/A"

    if [[ -f "${MANIFEST}" ]]; then
        NEW_P8=$(awk -F',' 'NR>1 && $6=="done" && $4!="" {count++} END {print count+0}' "${MANIFEST}" 2>/dev/null || echo "0")
        TOTAL_P8=$(awk -F',' 'NR>1 && $6=="done" {count++} END {print count+0}' "${MANIFEST}" 2>/dev/null || echo "0")
    fi

    # Count days with p8_ml.done
    TOTAL_P8=$(find "${OUTPUT_DIR}" -name "p8_ml.done" 2>/dev/null | wc -l | tr -d ' ')
    P8_DONE=$(( $(find "${OUTPUT_DIR}" -name "p8_ml.done" -newer "${MANIFEST}" 2>/dev/null | wc -l | tr -d ' ') ))

    log_tee "[STEP1] Done. P8 completed this run: ${P8_DONE}, total: ${TOTAL_P8}"

    # ── Step 2: Aggregate results ────────────────────────────────────────────────
    AGG_TRIGGERED="no"
    if (( P8_DONE > 0 )) || (( TOTAL_P8 > 0 )); then
        # Check disk space before aggregation
        FREE_GB=$(df -BG "${OUTPUT_DIR}" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G' || echo "999")
        if (( FREE_GB < 20 )); then
            log_tee "[STEP2] SKIP — disk space low (${FREE_GB}GB < 20GB)"
        else
            log_tee "[STEP2] Running aggregate_results.py ..."
            PREV_AGG_MTIME=$(prev_agg_mtime)

            ${PYTHON} "${AGG_RUNNER}" \
                --output-dir "${OUTPUT_DIR}" \
                --agg-dir "${AGG_DIR}" \
                --symbol ${SYMBOL} \
                --min-days 1 \
                >> "${LOG_FILE}" 2>&1

            NEW_AGG_MTIME=$(stat -c %Y "${AGG_DIR}/daily_metrics.csv" 2>/dev/null || echo "0")

            if [[ "${NEW_AGG_MTIME}" != "${PREV_AGG_MTIME}" ]] && [[ "${NEW_AGG_MTIME}" != "0" ]]; then
                AGG_TRIGGERED="yes"
                log_tee "[STEP2] Aggregate CSVs updated"

                # Extract best F1 from summary if available
                if [[ -f "${AGG_DIR}/summary_report.txt" ]]; then
                    BEST_F1=$(grep "f1_macro" "${AGG_DIR}/summary_report.txt" 2>/dev/null | head -1 | awk '{print $NF}' || echo "N/A")
                fi
            else
                log_tee "[STEP2] No aggregate CSV changes — skipping"
            fi
        fi
    else
        log_tee "[STEP2] No new P8 completions — skipping aggregation"
    fi

    # ── Step 3: Plot dashboard ──────────────────────────────────────────────────
    if [[ "${AGG_TRIGGERED}" == "yes" ]]; then
        log_tee "[STEP3] Running plot_dashboard.py ..."
        ${PYTHON} "${PLOTTER}" \
            --agg-dir "${AGG_DIR}" \
            --output-dir "${AGG_DIR}" \
            --style dark \
            --dpi 300 \
            >> "${LOG_FILE}" 2>&1
        log_tee "[STEP3] Dashboard updated"
    else
        log_tee "[STEP3] SKIP — no aggregate update"
    fi

    # ── Write one-line summary ─────────────────────────────────────────────────
    SUMMARY="${TIMESTAMP} | new_days=${P8_DONE} | total_p8=${TOTAL_P8} | agg_triggered=${AGG_TRIGGERED} | F1=${BEST_F1}"
    echo "${SUMMARY}" > "${LAST_RUN}"
    log_tee "=== Cron orchestrator finished ==="
}

main "$@"
