# ScienceClaw integration map

This package is the paper benchmark and self-evolution engine embedded in the
original ScienceClaw gateway. The integration keeps the two execution layers
connected through a small, typed boundary:

| Layer | Location | Responsibility |
| --- | --- | --- |
| Gateway and long-lived agent | repository root (`src/`, `extensions/`, `skills/`) | OpenClaw routing, provider/session handling, memory, and the original skill/plugin system |
| Benchmark engine | `packages/scienceclaw/scienceclaw/` | typed graphs, adapters, replay, scoring, receipts, and evolution candidates |
| Scientific operators | `packages/scienceclaw/scilib/` | frozen dataset tools and optional pretrained wrappers |
| Dataset skills | `skills/scienceclaw-benchmark-for30/` … `skills/scienceclaw-benchmark-for52/` | task contracts, tool routing, and evidence rules for FoR30–FoR52 |
| Evolution skill | `skills/scienceclaw-evolution/` | candidate replay, visible-dev gates, provenance, and promotion policy |
| Native bridge | `extensions/scienceclaw/` | bounded JSON calls for catalog, task availability, and reports |

The gateway bridge does not receive provider credentials or arbitrary shell
commands. Formal ID/OOD evaluation and remote launchers remain explicit
server-side operations; the bridge exposes only inspectable development and
reporting operations. Self-evolution remains in the Python engine and is
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

1. Load `skills/scienceclaw-benchmark/SKILL.md` for routing policy.
2. Load the matching `skills/scienceclaw-benchmark-for30` through `for52` skill.
3. Use the optional `scienceclaw_eval` tool for catalog, task availability, or
   report inspection.
4. Run formal evaluation with the existing server-side launcher and preserve
   its config, hashes, and per-episode receipts.
