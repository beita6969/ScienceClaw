#!/bin/bash
# usage: scripts/run_toolon_batch.sh <tag> <disciplines,comma> <split,comma> <n> <workers>   (config configs/toolon_local.yaml)
# Runs scripts/probe.py for each split in turn; outputs runs/toolon_local_<tag>_<split>/ and logs/toolon_local_<tag>_<split>.log
cd "$(dirname "$0")/.." || exit 1
TAG=$1; DISC=$2; SPLITS=$3; N=${4:-2}; W=${5:-3}
mkdir -p logs
export SCIENCECLAW_REMOTE_SPOOL="$PWD/cache/remote_spool" PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for sp in ${SPLITS//,/ }; do
  nice -n 5 .venv/bin/python scripts/probe.py --config configs/toolon_local.yaml --disciplines "$DISC" --split "$sp" --n "$N" \
    --workers "$W" --out "runs/toolon_local_${TAG}_${sp}" > "logs/toolon_local_${TAG}_${sp}.log" 2>&1
done
echo "BATCH_DONE $TAG" >> "logs/toolon_local_${TAG}.done"
