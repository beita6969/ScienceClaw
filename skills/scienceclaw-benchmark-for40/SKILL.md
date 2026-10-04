---
name: scienceclaw-benchmark-for40
description: "Run and improve the FoR40 DCASE 2024 Task 2 benchmark route with the integrated ScienceClaw agent. Use when working on Engineering scores, tools, visible-dev selection, or formal evidence."
metadata: { "openclaw": { "emoji": "📊" } }
---

# FoR40 — DCASE 2024 Task 2

Use this skill for the `FoR40` adapter. The task metric is **official DCASE score**;
maximize is better. The adapter module is
`scienceclaw.bench.tasks.for40_dcase`.

## Tool surface

The adapter currently declares these tool references:

- `load_eval_inputs`
- `load_train`
- `log_mel_spectrogram`
- `audio_embedding` (optional frozen AST/CLAP pooled features; requires the configured local or remote encoder)

Treat the list as a capability inventory, not permission to call every tool.
Select one frozen route plus a clearly named baseline, then record the exact
provenance and configuration used.

Keep `log_mel_spectrogram` as the default route. When the audio encoder is
available, call `audio_embedding` on both visible training and evaluation
waveforms, then use `scilib.anomsound.embedding_scores` and
`scilib.anomsound.rank_average` to produce one score per evaluation clip. The
AST/CLAP route is a candidate to compare against the log-mel route; it is not a
 reason to change the adapter's default or to access hidden labels.
The current visible comparison is positive but modest: pooled embeddings with
the rank-averaged `nn2_pool` component reached 0.5592 on the ID pool versus
0.5473 for the log-mel route, so keep it as an optional candidate rather than
claiming a universal replacement.

## Workflow

1. Read `docs/tasks/FoR40.md` and call `scienceclaw_eval` with
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
