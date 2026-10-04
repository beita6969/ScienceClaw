#!/bin/bash
# Usage: scripts/resume_pipeline.sh <run_dir>   — continue evolve in place, then evaluate + report.
# Launch detached so it survives launcher restarts:
#   nohup scripts/resume_pipeline.sh runs/X > logs/X.resume.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="$1"; PY=.venv/bin/python
echo "[pipeline] resume $RUN at $(date)"
$PY -u -m scienceclaw.cli evolve --resume "$RUN"
$PY -u -m scienceclaw.cli evaluate --run "$RUN" --snapshots all --splits id,ood --workers 8
$PY -u -m scienceclaw.cli report --run "$RUN"
echo "[pipeline] done $RUN at $(date)"
