---
name: scienceclaw-benchmark-for49
description: "Run and improve the FoR49 SMT-COMP 2025 QF_NIA benchmark route with the integrated ScienceClaw agent. Use when working on Mathematical sciences scores, tools, visible-dev selection, or formal evidence."
metadata: { "openclaw": { "emoji": "📊" } }
---

# FoR49 — SMT-COMP 2025 QF_NIA

Use this skill for the `FoR49` adapter. The task metric is **oracle-agreement accuracy**;
maximize is better. The adapter module is
`scienceclaw.bench.tasks.for49_smt`.

## Tool surface

The adapter currently declares these tool references:

- `load_dev_inputs`
- `load_eval_inputs`
- `load_train`
- `score_dev`
- `z3_check`

Treat the list as a capability inventory, not permission to call every tool.
Select one frozen route plus a clearly named baseline, then record the exact
provenance and configuration used.

## Workflow

1. Read `docs/tasks/FoR49.md` and call `scienceclaw_bench` with
   `operation=catalog` before changing a route.
2. Build the candidate from visible `load_train` data and use `score_dev` or the
   documented visible split for selection. Keep the output shape, unit, and hard
   constraints from the adapter unchanged.
   For the visible-dev solver route, start with
   `z3_check(query_timeout_s=30, workers=8, memory_mb=4096)` and let
   `score_dev` choose it against the 10-second baseline. This is the current
   fast/high-recall default: on three local `val` dev episodes (seeds 900–902)
   it produced accuracies 0.9375, 1.0000, and 0.8750 with no wrong definite
   answers. The 120-second setting is an escalation for a difficult formal
   episode when the episode wall budget allows it; it is not a scorer change.
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
