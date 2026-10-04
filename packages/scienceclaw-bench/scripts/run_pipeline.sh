#!/bin/bash
# Usage: scripts/run_pipeline.sh <config.yaml> [extra evolve args...]
# Runs evolve -> evaluate (all snapshots, id+ood+rep) -> report, logging to logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="$1"; shift || true
PY=.venv/bin/python
NAME=$(basename "$CFG" .yaml)
STAMP=$(date +%Y%m%d-%H%M%S)
LOG=logs/${NAME}-${STAMP}.log
echo "[pipeline] config=$CFG log=$LOG" | tee -a "$LOG"
$PY -u -m scienceclaw.cli evolve --config "$CFG" "$@" 2>&1 | tee -a "$LOG"
RUN=$(ls -td runs/${NAME}* | head -1)
echo "[pipeline] run dir: $RUN" | tee -a "$LOG"
$PY -u -m scienceclaw.cli evaluate --run "$RUN" --snapshots all --splits id,ood --workers 8 2>&1 | tee -a "$LOG"
$PY -u -m scienceclaw.cli report --run "$RUN" 2>&1 | tee -a "$LOG"
echo "[pipeline] done $RUN" | tee -a "$LOG"
