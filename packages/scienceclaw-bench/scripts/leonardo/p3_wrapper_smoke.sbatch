#!/bin/bash
# Real wrapper smoke for staged P3 assets; no ScienceClaw task data or targets.
# Submit with normal 1-GPU QoS after the asset paths are staged.
#SBATCH --job-name=p3-wrapper-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=aifac_f02_774
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/p3-stage-20261003/logs/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/p3-stage-20261003/logs/%x-%j.out
set -euo pipefail
F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000
L=/leonardo_scratch/large/userexternal/rqian000
export PYTHONPATH=$F/scienceclaw
export SCIENCECLAW_BEATS_MODEL=$L/p3-stage-20261003/beats/BEATs_iter3.pt
export SCIENCECLAW_BEATS_SOURCE=$L/p3-stage-20261003/upstream/unilm/beats
export SCIENCECLAW_BUILDINGS_MODEL=$L/p3-stage-20261003/buildingsbench/Transformer_Gaussian_L.pt
export SCIENCECLAW_BUILDINGS_SOURCE=$L/p3-stage-20261003/upstream/BuildingsBench
$F/envs/sc-harness/bin/python - <<'PY'
import json
import numpy as np
from scilib import beats, buildingsbench

emb = beats.encode(np.zeros((1, 16000), dtype=np.float32), device="cuda")
features = {name: np.zeros((1, buildingsbench.TOTAL_LENGTH, 1), dtype=np.float32)
            for name in buildingsbench.FEATURES}
pred = buildingsbench.forecast(features, device="cuda")
print(json.dumps({"beats": list(emb.shape), "buildingsbench": list(pred.shape),
                  "finite": bool(np.isfinite(emb).all() and np.isfinite(pred).all())}, sort_keys=True))
PY
