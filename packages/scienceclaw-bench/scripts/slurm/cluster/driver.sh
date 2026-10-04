#!/bin/bash
# Relay driver (login node, light): wait for a job to run + its Qwen endpoints to answer, then start the node-local broker and the batches.
# usage: driver.sh <jobid> "<disciplines>" [workers=8] [tag=H1] [tool_slots=2,3,2,3] [splits=id,ood] [n=4]
set -euo pipefail
J=${1:-}; DISC=${2:-}; W=${3:-8}; TAG=${4:-H1}; SLOTS=${5:-2,3,2,3}; SPLITS=${6:-id,ood}; N=${7:-4}
F=${SCIENCECLAW_WORK_ROOT:?set SCIENCECLAW_WORK_ROOT}; L=${SCIENCECLAW_STORE_ROOT:-$F}; R=$L/sc-runs; mkdir -p $R/logs
[ -n "$J" ] && [ -n "$DISC" ] || { echo "usage: $0 <jobid> <disciplines> [workers] [tag] [tool_slots] [splits] [n]" >&2; exit 2; }
[[ "$W" =~ ^[1-9][0-9]*$ ]] || { echo "workers must be a positive integer: $W" >&2; exit 2; }
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "n must be a positive integer: $N" >&2; exit 2; }
[[ "$DISC" =~ ^[A-Za-z0-9_,]+$ ]] || { echo "invalid disciplines: $DISC" >&2; exit 2; }
[[ "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid tag: $TAG" >&2; exit 2; }
[[ "$SLOTS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "invalid tool slots: $SLOTS" >&2; exit 2; }
[[ "$SPLITS" =~ ^(src|val|id|ood)(,(src|val|id|ood))*$ ]] || { echo "invalid splits: $SPLITS" >&2; exit 2; }

# Keep polling configurable, but reject malformed values before arithmetic or
# sleep can turn a typo into a busy loop or an unbounded wait.
POLL_S=${SCIENCECLAW_POLL_S:-10}
JOB_WAIT_S=${SCIENCECLAW_JOB_WAIT_S:-0}       # 0 keeps the historical unlimited pending wait
ENDPOINT_WAIT_S=${SCIENCECLAW_ENDPOINT_WAIT_S:-1800}
[[ "$POLL_S" =~ ^[1-9][0-9]*$ ]] || { echo "SCIENCECLAW_POLL_S must be a positive integer: $POLL_S" >&2; exit 2; }
[[ "$JOB_WAIT_S" =~ ^[0-9]+$ ]] || { echo "SCIENCECLAW_JOB_WAIT_S must be a non-negative integer: $JOB_WAIT_S" >&2; exit 2; }
[[ "$ENDPOINT_WAIT_S" =~ ^[1-9][0-9]*$ ]] || { echo "SCIENCECLAW_ENDPOINT_WAIT_S must be a positive integer: $ENDPOINT_WAIT_S" >&2; exit 2; }

# Direct driver use must have the same no-GPU preflight as chain.sh.
MANIFEST_CHECK=$F/sc-tools/check_formal_manifest.py
if [ -r "$MANIFEST_CHECK" ]; then
  SCIENCECLAW_REPO=$F/scienceclaw "$F/envs/sc-run/bin/python" "$MANIFEST_CHECK" "$DISC" "$TAG" "$SPLITS" "$N"
elif [[ "$TAG" == SOTA* ]]; then
  echo "missing formal manifest checker: $MANIFEST_CHECK" >&2
  exit 2
fi
"$F/envs/sc-run/bin/python" "$F/sc-tools/check_avail.py" "$DISC" --strict --splits "$SPLITS" --n "$N"

# A long-pending lprod job remains supported; terminal/vanished jobs abort
# instead of leaving a login-node process polling forever.
job_state() { squeue -h -j "$J" -o %T 2>/dev/null | head -n 1 || true; }
job_deadline=0
[ "$JOB_WAIT_S" -gt 0 ] && job_deadline=$(( $(date +%s) + JOB_WAIT_S ))
while :; do
  state=$(job_state)
  case "$state" in
    RUNNING) break ;;
    PENDING|CONFIGURING|REQUEUED|RESIZING)
      if [ "$job_deadline" -gt 0 ] && [ "$(date +%s)" -ge "$job_deadline" ]; then
        echo "timed out waiting for job $J to become RUNNING (state=$state)" >&2
        exit 1
      fi
      sleep "$POLL_S" ;;
    "")
      echo "job $J is no longer present in squeue; refusing to launch batches" >&2
      exit 1 ;;
    *)
      echo "job $J reached terminal/non-runnable state: $state" >&2
      exit 1 ;;
  esac
done
echo "$(date +%T) job $J running on $(squeue -h -j $J -o %N)"
deadline=$(( $(date +%s) + ENDPOINT_WAIT_S ))
while :; do
  ready=$(ls "$L/sc-serve/endpoints" 2>/dev/null | grep "^$J-g" | grep -vc starting || true)
  [ "$ready" -ge 2 ] && break
  [ "$(date +%s)" -lt "$deadline" ] || {
    echo "timed out waiting for two ready Qwen endpoints for job $J" >&2
    exit 1
  }
  [ "$(job_state)" = RUNNING ] || { echo "job $J stopped while endpoints were loading" >&2; exit 1; }
  sleep "$POLL_S"
done
echo "$(date +%T) endpoints ready"
[ -n "${NOBROKER:-}" ] || nohup srun --jobid=$J --overlap -N1 -n1 -c4 --gres=gpu:a100:4 --export=ALL $F/sc-tools/broker_step.sh $SLOTS >> $R/logs/broker_$(squeue -h -j $J -o %N).local.log 2>&1 < /dev/null &
[ -n "${NOBROKER:-}" ] || sleep 20
for sp in ${SPLITS//,/ }; do
  nohup srun --jobid=$J --overlap -N1 -n1 -c14 --gres=gpu:a100:4 --export=ALL $F/scienceclaw/scripts/slurm/run_batch.sh $TAG $DISC $sp $N $W 2,3 > $R/logs/srun_${TAG}_${sp}_$J.out 2>&1 < /dev/null &
  sleep 5
done
echo "$(date +%T) batches launched"
