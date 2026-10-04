#!/bin/bash
# Usage: scripts/resume_pipeline_final.sh <run_dir> [--skip-evolve]
# Continue evolve in place (unless --skip-evolve), then evaluate A_0 vs the last snapshot on id+ood and build the report.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="$1"; PY=${PY:-.venv/bin/python}
if [ "${2:-}" != "--skip-evolve" ]; then
  echo "[pipeline] resume evolve $RUN at $(date)"
  $PY -u -m scienceclaw.cli evolve --resume "$RUN"
fi
LAST=$(ls "$RUN/programs" | sed -nE 's/^A_?([0-9]+)$/\1/p' | sort -n | tail -1)
echo "[pipeline] evaluating A_0,A_${LAST} at $(date)"
$PY -u -m scienceclaw.cli evaluate --run "$RUN" --snapshots "A_0,A_${LAST}" --splits id,ood --no-rep --workers 8
$PY -u -m scienceclaw.cli report --run "$RUN" --pi-reference A_0
echo "[pipeline] done $RUN at $(date)"
