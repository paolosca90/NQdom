#!/usr/bin/env bash
# ============================================================
# vps_p1_to_p7_runner.sh
# ============================================================
# SSH into Contabo VPS e lancia P1-P7 su tutti i 19 giorni
# con resume (saltagià completati).
#
# Usage:
#   bash vps_p1_to_p7_runner.sh           # resume mode (default)
#   bash vps_p1_to_p7_runner.sh --force   # forza ricalcolo da P1
#
# VPS: root@185.185.82.205
#============================================================

set -e

FORCE_FLAG=""
if [ "$1" == "--force" ]; then
  FORCE_FLAG="--force"
  echo "[Runner] Modalità FORCE — ricalcolo da P1 per tutti i giorni"
else
  echo "[Runner] Modalità RESUME — salta giorni già completati"
fi

VPS="root@185.185.82.205"
VPS_BASE="/opt/depth-dom"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
SSH_CMD="ssh $SSH_OPTS $VPS"

echo "=========================================="
echo "NQ P1-P7 Pipeline Runner — Contabo VPS"
echo "=========================================="
echo "Giorni : tutti i 19 giorni (2026-03-13 → 2026-04-09)"
echo "Fasi   : P1 P2 P2b P3 P4 P5 P6 P7"
echo "Modalità: $([ -z "$FORCE_FLAG" ] && echo "RESUME" || echo "FORCE")"
echo "=========================================="

# Verifica connessione VPS
echo "[SSH] Verifica connessione a $VPS ..."
$SSH_CMD "echo 'OK' && free -h | grep Mem" || {
  echo "[ERRORE] Impossibile connettersi al VPS"
  exit 1
}

# Verifica spazio disco
echo ""
echo "[Storage] Spazio libero su VPS:"
$SSH_CMD "df -h /opt/depth-dom | tail -1"

# Lancia il pipeline orchestrator
# --skip-p7-p8 NON è disponibile come flag, quindi usiamo il runner
# esistente che fa P1-P8; P8 semplicemente non viene eseguito se i suoi
# input non sono pronti, oppure aggiungiamo un --skip-p8 se serve.
# Il runner originale supporta: --skip-p1-p6 e --skip-p7-p8.
#==> Per P1-P7 usiamo il runner base che包含 P1-P7 nel suo loop.
#    P8 è alla fine; se skip-p7-p8 è impossibile, P8 fallirà in modo
#    controllato perché mancano i suoi input specifici (non blocking).
#
# Soluzione: invochiamo il runner con --skip-p7-p8 E poi un secondo
# step per P7. Ma il manifest tracker non supporta skip parziali.
# Approccio più semplice: lasciamo che P8 venga tentato — se i label
# P7 non ci sono ancora, P8 fallirà presto e il job continuerà.
# Il resume farà sì che P8 venga ritentato dopo P7.
#
#oppure: run --skip-p7-p8 (così P1-P6 sono coperti), poi manualmente P7
# MA: il cron da 5 min deve poter riprendere.
#
# decisione: run P1-P6 con skip-p7-p8, poi un secondo runner per P7-only

echo ""
echo "[Pipeline] Step 1/2 — Lancio P1-P6 (resume mode)..."
$SSH_CMD "cd $VPS_BASE && python3 vps_multiday_runner.py \
  --workers 2 \
  --cleanup-policy none \
  --parallel 1 \
  --skip-p7-p8 \
  --resume \
  $FORCE_FLAG" || true

echo ""
echo "[Pipeline] Step 2/2 — Lancio P7 (resume mode)..."
$SSH_CMD "cd $VPS_BASE && python3 vps_multiday_runner.py \
  --workers 1 \
  --cleanup-policy none \
  --parallel 1 \
  --skip-p1-p6 \
  --resume \
  $FORCE_FLAG" || true

echo ""
echo "[Runner] Pipeline P1-P7 completato (o in corso)."
echo "[Runner] Usa 'bash vps_p1_to_p7_runner.sh --force' per rilanciare da P1."
