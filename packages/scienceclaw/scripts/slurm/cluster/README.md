# Running the ScienceClaw stack on a Slurm cluster

The LLM server, the sandboxed engine process and the GPU-tool broker can all sit on one GPU node. The scripts under
`scripts/slurm/` are written against two environment variables:

* `SCIENCECLAW_WORK_ROOT` — fast file system: the checkout as `scienceclaw/`, the virtualenvs under `envs/`, and the helper
  copies under `sc-tools/` (copy the files of this directory there before use).
* `SCIENCECLAW_STORE_ROOT` — large file system: model weights (`models/`), caches, vLLM endpoint files (`sc-serve/`), tool
  state (`sc-tools/`) and datasets. Defaults to `SCIENCECLAW_WORK_ROOT`.

Account, partition and QoS are never baked into a script: pass them to `sbatch`. `SCIENCECLAW_SSH_HOST` is the ssh alias of
the login node for `tunnel.sh` and `start_broker.sh`.

* `setup_tool_env.sh` / `setup_run_env.sh`: venvs `sc-harness` (torch + tools; used by the broker/worker) and `sc-run`
  (torch-free; engine process and sandbox, so heavy tools go through the spool). Unpinned requirement lists
  `req_*_u.txt` (python3.11).
* `stage_models.sh [asset ...]`: stages pretrained tool weights into `$SCIENCECLAW_MODELS` from the registry
  (`scienceclaw/tools/weights.json`; `python -m scienceclaw.cli weights status|plan`, or `python -m scienceclaw.cli setup`).
* `../serve_vllm.sbatch`, `../hold_and_serve.sbatch`, `../launch_in_allocation.sh`: a self-hosted OpenAI-compatible LLM
  server on one GPU node (several servers per node; the remaining GPUs stay free for the GPU tools).
* `../tunnel.sh`: forwards the running LLM servers to local ports and prints the matching `llm.endpoints` list.
* `broker_step.sh`, `../start_broker.sh`: the node-local GPU-tool broker (`scripts/remote/broker.py`, several slots per GPU).
* `../stage_clip.sh`: stages the OpenAI CLIP ViT-B/32 component for the remote tool worker.
