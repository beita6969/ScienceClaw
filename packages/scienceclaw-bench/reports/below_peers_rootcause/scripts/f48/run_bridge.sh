#!/bin/bash
R="/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$R" SCIENCECLAW_REMOTE_SPOOL="$R/cache/remote_spool"
cd "$R/reports/below_peers_rootcause/scripts/f48"
P=/private/tmp/claude-501/sc-scratch/f48
"$R/.venv/bin/python" pool_bridge.py $P/pool/pool_id.json 4 $P/pool/bridge_id.json > $P/pool/bridge_id.log 2>&1
