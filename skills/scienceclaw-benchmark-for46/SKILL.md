---
name: scienceclaw-benchmark-for46
description: "Run and improve the FoR46 HumanEval → MBPP benchmark route with the integrated ScienceClaw agent. Use when working on Information and computing sciences scores, tools, visible-dev selection, or formal evidence."
metadata: { "openclaw": { "emoji": "📊" } }
---

# FoR46 — HumanEval → MBPP

Use this skill for the `FoR46` adapter. The task metric is **execution pass@1**;
maximize is better. The adapter module is
`scienceclaw.bench.tasks.for46_code`.

## Tool surface

The adapter currently declares these tool references:

- `load_eval_inputs`
- `run_visible_tests`

Treat the list as a capability inventory, not permission to call every tool.
Select one frozen route plus a clearly named baseline, then record the exact
provenance and configuration used.

## Workflow

1. Read `docs/tasks/FoR46.md` and call `scienceclaw_eval` with
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
