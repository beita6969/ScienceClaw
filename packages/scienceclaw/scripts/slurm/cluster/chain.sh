#!/bin/bash
# Login-node chain (light): when no work2-sc-dbg job of mine is left, submit the next 30-min debug job and start the given drivers on it.
# usage: chain.sh <driver spec file>   (one driver per line: "<NOBROKER 0|1>|<disciplines>|<workers>|<tag>|<splits>|<n>")
set -euo pipefail
F=${SCIENCECLAW_WORK_ROOT:?set SCIENCECLAW_WORK_ROOT}; L=${SCIENCECLAW_STORE_ROOT:-$F}; R=$L/sc-runs
SPEC=${1:-}
[ -n "$SPEC" ] && [ -r "$SPEC" ] || { echo "usage: $0 <readable spec file>" >&2; exit 2; }

POLL_S=${SCIENCECLAW_CHAIN_POLL_S:-20}
EXISTING_WAIT_S=${SCIENCECLAW_EXISTING_WAIT_S:-0}  # 0 preserves waiting for an active chain
[[ "$POLL_S" =~ ^[1-9][0-9]*$ ]] || { echo "SCIENCECLAW_CHAIN_POLL_S must be a positive integer: $POLL_S" >&2; exit 2; }
[[ "$EXISTING_WAIT_S" =~ ^[0-9]+$ ]] || { echo "SCIENCECLAW_EXISTING_WAIT_S must be a non-negative integer: $EXISTING_WAIT_S" >&2; exit 2; }

# Validate every row before submitting a GPU job.  This catches missing data
# and empty split pools while the login node is still free.
PY=$F/envs/sc-run/bin/python
[ -x "$PY" ] || { echo "missing run-environment interpreter: $PY" >&2; exit 2; }
MANIFEST_CHECK=$F/sc-tools/check_formal_manifest.py
rows=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    ""|[[:space:]]*\#*) continue ;;
  esac
  IFS='|' read -r nb disc w tag splits cnt extra <<< "$line"
  [ -z "${extra:-}" ] || { echo "invalid spec row (too many fields): $line" >&2; exit 2; }
  case "$nb" in 0|1) ;; *) echo "NOBROKER must be 0 or 1: $line" >&2; exit 2 ;; esac
  [[ "$disc" =~ ^[A-Za-z0-9_,]+$ ]] || { echo "invalid disciplines: $disc" >&2; exit 2; }
  [[ "$w" =~ ^[1-9][0-9]*$ ]] || { echo "workers must be a positive integer: $w" >&2; exit 2; }
  [[ "$tag" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid tag: $tag" >&2; exit 2; }
  [[ "$splits" =~ ^(src|val|id|ood)(,(src|val|id|ood))*$ ]] || { echo "invalid splits: $splits" >&2; exit 2; }
  [[ "$cnt" =~ ^[1-9][0-9]*$ ]] || { echo "n must be a positive integer: $cnt" >&2; exit 2; }
  if [ -r "$MANIFEST_CHECK" ]; then
    SCIENCECLAW_REPO=$F/scienceclaw "$PY" "$MANIFEST_CHECK" "$disc" "$tag" "$splits" "$cnt" || {
      echo "formal manifest check failed; no GPU job was submitted for: $line" >&2
      exit 2
    }
  elif [[ "$tag" == SOTA* ]]; then
    echo "missing formal manifest checker: $MANIFEST_CHECK" >&2
    exit 2
  fi
  "$PY" "$F/sc-tools/check_avail.py" "$disc" --strict --splits "$splits" --n "$cnt" || {
    echo "preflight failed; no GPU job was submitted for: $line" >&2
    exit 2
  }
  rows=$((rows + 1))
done < "$SPEC"
[ "$rows" -gt 0 ] || { echo "spec has no runnable rows: $SPEC" >&2; exit 2; }

# Consume the complete squeue output instead of grep -q: with pipefail,
# grep's early close can make squeue exit with SIGPIPE and break this guard.
debug_job_present() {
  squeue -u "$USER" -h -o "%j" | grep -F "work2-sc-dbg" >/dev/null
}
existing_deadline=0
[ "$EXISTING_WAIT_S" -gt 0 ] && existing_deadline=$(( $(date +%s) + EXISTING_WAIT_S ))
while debug_job_present; do
  if [ "$existing_deadline" -gt 0 ] && [ "$(date +%s)" -ge "$existing_deadline" ]; then
    echo "timed out waiting for an existing work2-sc-dbg job to leave the queue" >&2
    exit 1
  fi
  sleep "$POLL_S"
done
cd $F/sc-serve
for i in 1 2 3 4 5 6; do
  J=$(sbatch --parsable --job-name=work2-sc-dbg -A ${SCIENCECLAW_SLURM_ACCOUNT:?set SCIENCECLAW_SLURM_ACCOUNT} -p ${SCIENCECLAW_SLURM_PARTITION:?set SCIENCECLAW_SLURM_PARTITION} --qos=${SCIENCECLAW_SLURM_QOS:-normal} -N1 --gres=gpu:4 -c 32 --mem=400G --time=00:30:00 hold_and_serve.sbatch 2>/dev/null) && break
  sleep 30
done
[ -n "$J" ] || { echo "sbatch failed"; exit 1; }
echo "$(date +%T) submitted $J"
n=0
while IFS='|' read -r nb disc w tag splits cnt; do
  case "$nb" in
    ""|[[:space:]]*\#*) continue ;;
  esac
  n=$((n+1))
  if [ "$nb" = 1 ]; then ( NOBROKER=1 nohup $F/sc-tools/driver.sh $J "$disc" $w $tag 2,3,2,3 $splits $cnt > $R/logs/driver_${J}_$n.log 2>&1 < /dev/null & )
  else ( nohup $F/sc-tools/driver.sh $J "$disc" $w $tag 2,3,2,3 $splits $cnt > $R/logs/driver_${J}_$n.log 2>&1 < /dev/null & ); fi
done < $SPEC
echo "$(date +%T) drivers started for $J"
