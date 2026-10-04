# ScienceClaw integration map

This package is the paper benchmark and self-evolution engine embedded in the
original ScienceClaw gateway. The integration keeps the two execution layers
connected through a small, typed boundary:

| Layer | Location | Responsibility |
| --- | --- | --- |
| Gateway and long-lived agent | repository root (`src/`, `extensions/`, `skills/`) | OpenClaw routing, provider/session handling, memory, and the original skill/plugin system |
| Benchmark engine | `packages/scienceclaw-bench/scienceclaw/` | typed graphs, adapters, replay, scoring, receipts, and evolution candidates |
| Scientific operators | `packages/scienceclaw-bench/scilib/` | frozen dataset tools and optional pretrained wrappers |
| Dataset skills | `skills/scienceclaw-benchmark-for30/` … `skills/scienceclaw-benchmark-for52/` | task contracts, tool routing, and evidence rules for FoR30–FoR52 |
| Evolution skill | `skills/scienceclaw-evolution/` | candidate replay, visible-dev gates, provenance, and promotion policy |
| Native bridge | `extensions/scienceclaw-bench/` | bounded JSON calls for catalog, task inventory, offline smoke, and reports |

The gateway bridge does not receive provider credentials or arbitrary shell
commands. Formal ID/OOD evaluation and remote launchers remain explicit
server-side operations; the bridge exposes only inspectable development and
reporting operations. Self-evolution remains in the Python engine and is
usable through the existing gateway skills: candidates require replay,
visible-dev validation, hard-constraint checks, reproducibility, provenance,
and the configured budget gate before promotion.

The package copy includes the benchmark source, operators, configs, launch
scripts, tests, reports, and documentation. Datasets, model weights, caches,
run outputs, logs, virtual environments, and credentials are deployment-local
and are excluded by `.gitignore`. Before adding a new operator or skill, keep
its source and routing metadata in the package or root skill directory, then
record its provenance in the episode receipt.

## Quick orientation

1. Load `skills/scienceclaw-benchmark/SKILL.md` for routing policy.
2. Load the matching `skills/scienceclaw-benchmark-for30` through `for52` skill.
3. Use the optional `scienceclaw_bench` tool for catalog, task inventory, TOY
   smoke, or report inspection.
4. Run formal evaluation with the existing server-side launcher and preserve
   its config, hashes, and per-episode receipts.
