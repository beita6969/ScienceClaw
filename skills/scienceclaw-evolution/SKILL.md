---
name: scienceclaw-evolution
description: Operate ScienceClaw's program-level self-evolution when a source episode yields a skill or operator candidate that must be replayed, checked on visible development data, and either promoted or rejected with receipts. Use scienceclaw-benchmark for dataset and tool routing.
metadata:
  openclaw:
    emoji: "🧬"
---

# ScienceClaw self-evolution

Use the Python engine under `packages/scienceclaw/scienceclaw/evolution/` as
the authoritative evolution implementation. The gateway's long-lived agent and
the benchmark engine remain separate: the gateway supplies sessions and skills;
the engine supplies typed programs, replay, receipts, and promotion decisions.

## Choose the run mode

- To inspect a finished run, call the optional `scienceclaw_eval` tool with
  `operation=report` and that run's `runId`.
- For a real benchmark, load `scienceclaw-benchmark` and the matching
  `scienceclaw-benchmark-for*` skill first. The server-side stream launcher owns
  the dataset and split configuration.
- For a skill/operator proposal, use the source stream and the engine's
  `Evolver`; do not hand-edit a promoted `AgentProgram` or treat a successful
  live solve as promotion evidence.

`scienceclaw_program` shows the active program version, its promotion receipts and
rolls back to an earlier version; `scienceclaw_eval` exposes only `catalog`,
`list_tasks`, and `report`. Neither exposes arbitrary shell commands or formal
ID/OOD evaluation. Use the
checked-in launcher for an explicitly approved server-side batch and retain its
manifest and receipts.

## Evolve from your own sessions

A finished, replay-verified canvas session is a source episode. With the optional
`scienceclaw_evolve` tool the same gate that governs benchmark evolution applies to live work:

1. Register validation tasks with `operation=val_add` — tasks the user confirms as
   representative, given as a task declaration or as the `sessionId` of a passed session.
   Keep them disjoint from the task a candidate is learned from; the engine refuses to
   validate a candidate on its own source.
2. `operation=propose` (or `run`, which also gates) builds the linked Skill/Operator bundle
   from the session's repair. `operation=gate` re-solves the source task with the candidate
   (`R_src = Pass ∧ Use`), then solves the validation tasks with the incumbent and the
   candidate under the frozen model, and promotes only on `H_val`, budget and strict
   `Q_val` improvement. These are background jobs: poll `operation=status`.
3. Read `operation=show` for the decision and reasons. A candidate without validation tasks or
   without an available model stays pending; it is never promoted by default. Promotions
   are versioned and reversible with `scienceclaw_program(operation=rollback)`.

## Candidate loop

1. Freeze the incumbent program and record its source revision, tool registry,
   config, dataset split, and environment fingerprint.
2. Build a candidate from a completed source episode or an explicitly reviewed
   skill/operator patch. Keep the candidate's provenance and changed files in
   its bundle.
3. Replay the candidate from its receipt. A replay failure is a rejected
   candidate, even if a live run appeared to improve a score.
4. Compare the candidate with the incumbent on the visible development split.
   Check the primary metric direction, hard constraints, reproducibility, token
   and wall-clock budgets, and the configured improvement margin.
5. Promote only after every configured gate passes. Keep the incumbent and the
   rejection reasons so the next iteration remains auditable.

The concrete order is:

`source solve → replay-verified e⁻/e⁺ → split control/executable edits →
skill patch and boundary-replayed operator candidates → apply bundle with
versioned ω → source replay (R_src) → visible D_val validation → H_val,
budget, and Q_val gate → commit snapshot or reject`.

`R_src` is `Pass(e_src) ∧ Use(ω)` on the same passing replay evidence. A
candidate that passes a live solve but is absent from the passing evidence, or
whose operator fails boundary replay, is rejected before visible validation.

Do not use hidden ID/OOD items to select a candidate, tune a threshold, or
generate a skill. Formal evaluation is a separate server-side operation.

## Skill and tool changes

Treat a new skill as routing and evidence guidance, not as an undocumented
scorer change. Treat a new operator as a frozen, versioned tool with explicit
inputs, outputs, dependency notes, and a deterministic replay check.
When a candidate depends on a pretrained model or external service, record the
model identifier, license/authorization state, content hash, and fallback
behavior; do not silently download weights or forward gateway credentials.

When a patch model writes a skill, require explicit applicability, a numbered
procedure, pitfalls, linked tool/operator refs, and source provenance. Scrub
episode IDs, item IDs, absolute paths, and hidden labels or scores before the
skill is installed. A generated skill may change retrieval, but may not change
an adapter's split, evaluator, or acceptance rule.

## Gate and stop rules

Use the configured `qval`, `qval_eps`, `hval_mode`, `min_improved_episodes`,
`max_regressed_episodes`, logical-token budget, and wall-time budget. The
default gate is strict: `H_val` must hold, the candidate must be within both
absolute and relative validation budgets, and `Q_val` must strictly improve
the incumbent. The noise guard requires the configured number of individually
improved visible episodes; a configuration with a single validation episode
should set `min_improved_episodes=1`.

Reject and record a candidate when source or boundary replay fails, the
candidate is not actually used, a visible hard constraint/schema/integrity/
reproducibility/solver-error gate fails, a budget is exceeded, or visible
comparison is not a strict improvement. An infrastructure outage is an error
receipt with bounded retry, not evidence of model regression. Resume only with
the same run directory and matching configuration; a changed fingerprint or
configuration requires a new run.

## Evidence to retain

Keep the existing run receipts and promotion reason so a result can be
reproduced from the same run directory. Use the shared benchmark protocol in
`skills/scienceclaw-benchmark/references/protocol.md` for episode-level
evidence. The operational priority is to run the next approved benchmark and
measure its effect; do not create extra audit documents for a routine run.
