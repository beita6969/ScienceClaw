---
name: scienceclaw-benchmark
description: "Operate the integrated ScienceClaw benchmark and its self-evolving agent workflow. Use when running, comparing, reproducing, or improving any FoR30–FoR52 benchmark task, selecting frozen SOTA tools, or diagnosing score gaps."
metadata: { "openclaw": { "emoji": "🧪" } }
---

# ScienceClaw benchmark orchestration

Use this skill when a request concerns the paper's 23-discipline benchmark, a
benchmark score, a frozen SOTA component, a tool-on run, or the self-evolution
loop that produces and validates skills/operators.

## Routing

1. Call the optional `scienceclaw_bench` tool with `operation=catalog` to see the
   current adapters, tool references, configs, and scripts.
2. Load the matching dataset skill `scienceclaw-benchmark-for30` through
   `scienceclaw-benchmark-for52` for the task contract and preferred evidence.
3. Read only the matching reference in `references/` when a detailed protocol is
   needed. Do not load all 23 references for a single task.
4. Keep the benchmark engine and the gateway separate: the gateway discovers
   tools, while the Python package owns typed graphs, adapters, replay, scoring,
   and receipts.

## Self-evolution contract

The agent may propose skills and operators, but every candidate must be replayable
and attributable to a concrete episode. Promote a candidate only after source
replay, visible-dev validation, hard-constraint checks, reproducibility, and the
configured budget gate pass. A skill is a routing recipe; it must not hide labels
or silently change a dataset split.

## Evidence boundaries

- `train`, `dev`, and explicitly documented visible pools may guide tool choice.
- `id`/`ood` targets are evaluator-only. The policy must not see them during
  construction, skill writing, or tool selection.
- Never reuse an evaluated formal item or relax an acceptance rule to make a
  candidate pass.
- Store configs, program snapshots, tool provenance, hashes, and per-episode
  receipts with every reported result.
- Treat external weights as frozen assets and disclose training-domain overlap;
  do not train a new checkpoint inside this workflow.

Dataset-specific routing lives in the 23 `scienceclaw-benchmark-for*` skills.
