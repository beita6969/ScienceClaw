# ScienceClaw integration map

This package is the paper benchmark and self-evolution engine embedded in the
original ScienceClaw gateway. The integration keeps the two execution layers
connected through a small, typed boundary:

| Layer | Location | Responsibility |
| --- | --- | --- |
| Gateway and long-lived agent | repository root (`src/`, `extensions/`, `skills/`) | OpenClaw routing, provider/session handling, memory, and the original skill/plugin system |
| Benchmark engine | `packages/scienceclaw/scienceclaw/` | typed graphs, adapters, replay, scoring, receipts, and evolution candidates |
| Scientific operators | `packages/scienceclaw/scilib/` | frozen dataset tools and optional pretrained wrappers |
| Canvas skill | `skills/scienceclaw-canvas/` | how the gateway agent orchestrates a task on the canvas |
| Dataset skills | `skills/scienceclaw-benchmark-for30/` … `skills/scienceclaw-benchmark-for52/` | task contracts, tool routing, and evidence rules for FoR30–FoR52 |
| Evolution skill | `skills/scienceclaw-evolution/` | candidate replay, visible-dev gates, provenance, and promotion policy |
| Canvas, tools, program | `packages/scienceclaw/scienceclaw/{canvas,tools,program}/`, `evolution/live.py` | stepwise typed-workflow sessions, the scilib/weights registry, the versioned Skill/Operator store, gated evolution from finished sessions |
| Native plugin | `extensions/scienceclaw/` | `scienceclaw_canvas`, `scienceclaw_tools`, `scienceclaw_program`, `scienceclaw_evolve`, `scienceclaw_eval` over `python -m scienceclaw.rpc` |

The engine process does not receive provider credentials, and live task inputs
are read only from the configured input roots. Formal ID/OOD evaluation and
remote launchers remain explicit server-side operations; the plugin exposes
only interactive canvas sessions, inspection and reporting. Self-evolution remains in the Python engine and is
usable through the existing gateway skills: candidates require replay,
visible-dev validation, hard-constraint checks, reproducibility, provenance,
and the configured budget gate before promotion.

The package copy includes the benchmark source, operators, configs, launch
scripts, tests, and documentation. Dataset deliveries, model weights, caches,
run outputs, formal reports, logs, virtual environments, and credentials are
deployment-local and are excluded from the public code checkout. Before adding a new operator or skill, keep
its source and routing metadata in the package or root skill directory, then
record its provenance in the episode receipt.

## Deployment roots and optional stacks

Install the lightweight engine with `pip install -e packages/scienceclaw`.
On a host that will run domain tools, add one or more optional stacks, for
example `pip install -e 'packages/scienceclaw[vision,audio,nlp]'` or
`[all]` on a prepared GPU image. The extras are dependency groups only; model
weights remain deployment-local and are never pulled into this repository.

Point the runtime at the externally managed data and weight roots:

```bash
export SCIENCECLAW_DATA_ROOT=/srv/scienceclaw/datasets
export SCIENCECLAW_MODELS=/srv/scienceclaw/models
```

`SCIENCECLAW_DATA_ROOT` should contain the Hugging Face-delivered dataset
snapshot in the directory layout expected by the selected FoR adapter. If it
is omitted, the package falls back to `~/.cache/scienceclaw/datasets`; the
checked-in YAML files do not contain a machine-specific path. `SCIENCECLAW_MODELS`
is consumed by the pretrained wrappers and the Slurm launchers.

## Quick orientation

1. Load `skills/scienceclaw-canvas/SKILL.md` to orchestrate a task on the canvas
   (`scienceclaw_canvas`, `scienceclaw_tools`).
2. For benchmark work load `skills/scienceclaw-benchmark/SKILL.md` and the matching
   `skills/scienceclaw-benchmark-for30` through `for52` skill.
3. Use `scienceclaw_program` to inspect or roll back the active Skill/Operator version,
   `scienceclaw_evolve` to learn from a verified session through the validation gate, and
   `scienceclaw_eval` for catalog, task availability or report inspection.
4. Run formal evaluation with the existing server-side launcher and preserve
   its config, hashes, and per-episode receipts.

## Models, tools and safety switches

* **Language model.** `llm.backend` is `openai` (an OpenAI-compatible endpoint), `command` (a local program that reads the
  prompt on stdin and prints the completion; `llm.command` or `SCIENCECLAW_LLM_COMMAND`, with `{system}` and `{model}`
  placeholders) or `package.module:factory`. Model names and credentials come from the environment.
* **Tool library.** `python -m scienceclaw.cli tools status` shows which of the 47 scilib modules run here and why not;
  `python -m scienceclaw.cli weights status|plan|verify` stages the pretrained checkpoints and the upstream sources some
  wrappers need. The typed operators in `scienceclaw/program/specs/` wrap the main entry points; `program.check.check_operator`
  runs one through the real canvas on concrete inputs.
* **Code-node sandbox.** Workers run with an allow-listed environment, a runtime audit guard (protected data and state
  locations, sockets, programs other than the interpreter, native libraries, the engine's sources) and, where `unshare`
  works, in new user, network and pid namespaces (`SCIENCECLAW_SANDBOX_ISOLATION=auto|off|require`). Run untrusted
  workloads in a container as well.
* **Program changes learned from live sessions** wait as `ready` until the user promotes them:
  `python -m scienceclaw.cli live candidates|show|promote|rollback|history` (`autoPromote` in the plugin configuration
  makes the gate promote by itself).
