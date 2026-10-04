# Running the ScienceClaw stack on a Slurm cluster

Everything (probe, sandbox, self-hosted vLLM, GPU-tool broker) can sit on one GPU node. The scripts under
`scripts/slurm/` are written against two environment variables:

* `SCIENCECLAW_WORK_ROOT` — fast file system: the checkout as `scienceclaw/`, the virtualenvs under `envs/`, and the helper
  copies under `sc-tools/` (copy the files of this directory there before use).
* `SCIENCECLAW_STORE_ROOT` — large file system: model weights (`models/`), caches, vLLM endpoint files (`sc-serve/`), tool
  state (`sc-tools/`), datasets and run output (`sc-runs/`). Defaults to `SCIENCECLAW_WORK_ROOT`.

Account, partition and QoS are never baked into a script: pass them to `sbatch`, or set `SCIENCECLAW_SLURM_ACCOUNT`,
`SCIENCECLAW_SLURM_PARTITION` and `SCIENCECLAW_SLURM_QOS` for `chain.sh`. `SCIENCECLAW_SSH_HOST` is the ssh alias of the
login node for `tunnel.sh` and `start_broker.sh`.

* `setup_tool_env.sh` / `setup_run_env.sh`: venvs `sc-harness` (torch + tools; used by the broker/worker) and `sc-run`
  (torch-free; probe + sandbox, so tools go through the spool). Unpinned requirement lists `req_*_u.txt` (python3.11).
* `stage_models.sh [asset ...]`: stages pretrained tool weights into `$SCIENCECLAW_MODELS` from the registry
  (`scienceclaw/tools/weights.json`; `python -m scienceclaw.cli weights status|plan`).
* `broker_step.sh`: long-lived srun step with the node-local GPU-tool broker (`broker.py --local`, several slots per GPU).
* `driver.sh <jobid> <disciplines> [workers] [tag] [slots] [splits] [n]`: waits for a job and its vLLM endpoints, then
  starts the broker and `scripts/slurm/run_batch.sh` batches. The holding job is `hold_and_serve.sbatch`. A vanished or
  terminal job aborts immediately; the endpoint wait defaults to 1800 s (`SCIENCECLAW_ENDPOINT_WAIT_S`), polling is set by
  `SCIENCECLAW_POLL_S`, and an optional finite pending-job limit is `SCIENCECLAW_JOB_WAIT_S` (zero waits indefinitely).
* `chain.sh <spec>` validates every row and its requested split/episode count before submitting. An existing job can be
  bounded with `SCIENCECLAW_EXISTING_WAIT_S` (zero waits indefinitely); `SCIENCECLAW_CHAIN_POLL_S` sets the polling.
  Empty and comment rows are ignored in both validation and launch.
* `summarize.py <tag>...`: pass counts plus separate `tool`, direct-library, and retrieved-operator counts. The latter are
  read from trajectory nodes because `result.json.uses` does not include built-in task tools.
* `validate_formal_results.py <tag> [run-root] [--split id|ood|src|val]`: read-only post-run evidence gate for a formal tag.
  It checks manifest episode counts, every `result.json` (`z=1`, `completed`, `reproducible`, `within_budget`, `hard_ok`),
  and every trajectory for successfully executed `required_tool_refs` whose output lineage reaches the final `submit.y`; a
  successful call that is disconnected from the submitted prediction fails closed. The optional split filter lets
  concurrent id/ood launches validate independently.
* `check_avail.py <FoRxx,...>`: adapter availability and episode plan; run on the login node before launching a batch.
  `--strict --splits id,ood --n 4` fails before `sbatch` if any requested pool is short. Direct batches also validate
  split, count, worker and endpoint-port syntax; `SCIENCECLAW_ENDPOINT_PORTS` overrides the default `21000,21001` health
  probes.
* `check_formal_manifest.py <discipline> <tag> <split[,split...]> <n>`: formal `SOTA*` tags must match
  `configs/formal_toolon_manifest.json` exactly (discipline, split, episode count, blocked/unavailable status, and required
  `ToolSpec`). `chain.sh`, `driver.sh`, and `run_batch.sh` invoke it before any broker or GPU batch; engineering tags such
  as `H6` remain unconstrained.
