#!/bin/bash
R="/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$R"
cd "$R/reports/below_peers_rootcause/scripts/f48"
P=/private/tmp/claude-501/sc-scratch/f48
for s in id:20 ood:30; do
  sp=${s%%:*}; n=${s##*:}
  nice -n 5 "$R/.venv/bin/python" pool_eval.py $P/plm_scores_v2.npz $sp $n $P/pool/pool_v2_$sp.json > $P/pool/pool_v2_$sp.log 2>&1
done
