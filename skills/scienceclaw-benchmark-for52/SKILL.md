---
name: scienceclaw-benchmark-for52
description: "Run and improve the FoR52 Psych-201 discrete benchmark route with the integrated ScienceClaw agent. Use when working on Psychology scores, tools, visible-dev selection, or formal evidence."
metadata: { "openclaw": { "emoji": "📊" } }
---

# FoR52 — Psych-201 discrete

Use this skill for the `FoR52` adapter. The task metric is **micro accuracy**;
maximize is better. The adapter module is
`scienceclaw.bench.tasks.for52_psych201`.

## Tool surface

The adapter currently declares these tool references:

- `load_dev_inputs`
- `load_eval_inputs`
- `load_train`
- `psych_adaptive_predict`
- `psych_domain_adaptive_predict`
- `psych_fixed_predict`
- `score_dev`

Treat the list as a capability inventory, not permission to call every tool.
Select one frozen route plus a clearly named baseline, then record the exact
provenance and configuration used.

## Workflow

1. Read `docs/tasks/FoR52.md` and call `scienceclaw_bench` with
   `operation=catalog` before changing a route.
2. Build the candidate from visible `load_train` data and use `score_dev` or the
   documented visible split for selection. Keep the output shape, unit, and hard
   constraints from the adapter unchanged.
3. Prefer an existing frozen checkpoint or remote wrapper. Do not train new
   weights, infer hidden targets, or use an evaluation item to choose a skill.
4. For self-evolution, let the solver produce a replayable graph, attribute the
   passing change to a skill/operator bundle, and validate it against the
   incumbent before promotion.
5. For formal work, use only an approved mutually exclusive launcher and record
   the manifest tag. If capacity or a compliant asset is missing, record the
   blocker instead of retrying an evaluated item.

See `skills/scienceclaw-benchmark/references/protocol.md` for the shared
promotion and evidence contract.
