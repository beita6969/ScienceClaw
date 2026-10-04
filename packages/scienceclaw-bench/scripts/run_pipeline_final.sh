#!/bin/bash
# Usage: scripts/run_pipeline_final.sh <config.yaml> [extra evolve args...]
# evolve -> evaluate only A_0 and the last snapshot on id+ood (no D_rep) -> report. Cheaper than run_pipeline.sh
# (which evaluates every snapshot); resume an interrupted run with scripts/resume_pipeline_final.sh <run_dir>.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="$1"; shift || true
PY=${PY:-.venv/bin/python}
NAME=$(basename "$CFG" .yaml)
STAMP=$(date +%Y%m%d-%H%M%S)
LOG=logs/${NAME}-${STAMP}.log
mkdir -p logs
echo "[pipeline] config=$CFG log=$LOG" | tee -a "$LOG"
$PY -u -m scienceclaw.cli evolve --config "$CFG" "$@" 2>&1 | tee -a "$LOG"
RUN=$(ls -td runs/${NAME}-* | head -1)
scripts/resume_pipeline_final.sh "$RUN" --skip-evolve 2>&1 | tee -a "$LOG"
