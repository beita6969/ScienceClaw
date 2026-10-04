#!/bin/bash
#SBATCH --job-name=p3-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_bprod
#SBATCH --account=aifac_f02_774
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/p3-stage-20261003/logs/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/p3-stage-20261003/logs/%x-%j.out
set -euo pipefail
F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000
L=/leonardo_scratch/large/userexternal/rqian000
mkdir -p "$L/p3-stage-20261003/logs"
MODE=${P3_MODE:?set P3_MODE=beats or buildings}
PY=${P3_PY:-$F/envs/sc-harness/bin/python}
case "$MODE" in
  beats) CKPT="$L/p3-stage-20261003/beats/BEATs_iter3.pt"; exec "$PY" "$F/scienceclaw/scripts/leonardo/p3_asset_smoke.py" beats --checkpoint "$CKPT";;
  buildings) CKPT="$L/p3-stage-20261003/buildingsbench/Transformer_Gaussian_L.pt"; exec "$PY" "$F/scienceclaw/scripts/leonardo/p3_asset_smoke.py" buildings --checkpoint "$CKPT";;
esac
