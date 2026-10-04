---
name: scienceclaw-benchmark-for39
description: "Run and improve the FoR39 Eedi NeurIPS 2020 Task 4 benchmark route with the integrated ScienceClaw agent. Use when working on Education scores, tools, visible-dev selection, or formal evidence."
metadata: { "openclaw": { "emoji": "📊" } }
---

# FoR39 — Eedi NeurIPS 2020 Task 4

Use this skill for the `FoR39` adapter. The task metric is **organizer 10-mask accuracy**;
maximize is better. The adapter module is
`scienceclaw.bench.tasks.for39_eedi`.

## Tool surface

The adapter currently declares these tool references:

- `load_dev_inputs`
- `load_eval_inputs`
- `load_train`
- `query_answers`
- `query_dev_answers`
- `score_dev`

Treat the list as a capability inventory, not permission to call every tool.
Select one frozen route plus a clearly named baseline, then record the exact
provenance and configuration used.

## Workflow

1. Read `docs/tasks/FoR39.md` and call `scienceclaw_bench` with
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
