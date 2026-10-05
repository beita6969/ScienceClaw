---
name: scienceclaw-evolution
description: Operate ScienceClaw's program-level self-evolution - turn a finished, replay-verified canvas session into a Skill/Operator candidate, gate it on user-registered validation tasks, and leave promotion (and rollback) to the user, with receipts.
metadata:
  openclaw:
    emoji: "🧬"
---

# ScienceClaw self-evolution

The Python engine under `packages/scienceclaw/scienceclaw/evolution/` is the authoritative
implementation. The gateway agent supplies the work (canvas sessions) and starts the gate through
`scienceclaw_evolve`; the engine supplies typed programs, replay, receipts and the admit/reject
decision. The language model is never trained: only the program A = (Skills, typed Operators)
changes, as a versioned, reversible store.

## Surface

- `scienceclaw_canvas`: solves a task and yields the replay-verified session that evolution learns from
  (see `scienceclaw-canvas`).
- `scienceclaw_evolve` (`val_add`, `val_list`, `val_remove`, `propose`, `gate`, `run`, `status`,
  `candidates`, `show`): builds candidates and runs the gate. `propose`, `gate` and `run` are
  background jobs; poll `operation=status` (`waitSeconds` up to 300).
- `scienceclaw_program` (`summary`, `skills`, `operators`, `show`, `history`, `rollback`): shows the active
  program version, its promotion receipts and rolls back to an earlier version.
- User side, outside the agent's tools: `scienceclaw live candidates|show|promote|rollback|history`
  (equivalently `python -m scienceclaw.cli live ...`).

None of these exposes a shell. Do not hand-edit a promoted `AgentProgram`, and do not treat a
successful live solve as promotion evidence: a change enters the program only through the gate.

## Evolve from your own sessions

A finished, replay-verified canvas session is a source episode (D_src); the validation set D_val is
a set of tasks the user registered as representative.

1. Register at least two validation tasks with `operation=val_add`, only tasks the user confirms as
   representative, given as a task declaration (with constraints) or as the `sessionId` of a passed
   session. Keep them disjoint from the task a candidate is learned from: the engine refuses to
   validate a candidate on its own source, and a validation task whose input files changed blocks the
   gate until it is registered again.
2. `operation=propose` (or `run`, which also gates) builds the linked Skill/Operator bundle from the
   session's repair (a replay-verified failure followed by the edits that made it pass); without an
   earlier replay-verified failure no Skill candidate is formed. `operation=gate` re-solves the source task with the
   candidate (`R_src = Pass ∧ Use`), then solves the validation tasks with the incumbent and the
   candidate under the frozen model, and admits the candidate only if every hard constraint holds on
   every validation task, the cost stays within budget and the success rate strictly improves (with at
   least `min_improved_episodes` validation tasks individually improved). A candidate derived from an
   older program is refused when a component it changes has moved on.
3. Read `operation=show` for the decision and reasons. An admitted candidate is `ready`: the user
   promotes it with `scienceclaw live promote <id>` (or enabled `autoPromote` in the plugin
   configuration). Nothing is promoted without validation tasks or an available model. Promotions
   are versioned, leave receipts and are reversible with `scienceclaw_program(operation=rollback)`
   or `scienceclaw live rollback <version>`.

`variant` selects what is learned: `full` (linked Skill + Operator bundle, the default),
`workflow_only`, `skill_only`, `operator_only` or `unlinked` (Skill and Operator gated independently).

The concrete order is:

`finished session → evolution instance (e⁻, e⁺, δ) → split control/executable edits → skill patch
and boundary-replayed operator candidates → bundle B = (ΔS, ΔO) applied with versioned ω → source
replay (R_src) → validation on D_val (H_val, budget, Q_val) → ready or rejected → promotion by the user`.

`R_src` is `Pass(e_src) ∧ Use(ω)` on the same passing replay evidence. A candidate that passes a live
solve but is absent from the passing evidence, or whose operator fails boundary replay, is rejected
before validation.

## Gate and stop rules

The live gate is strict: `H_val` must hold on every validation task (every hard constraint, schema,
integrity and reproducibility check, no solver error), the candidate must be within both the absolute
and the relative validation budgets, and `Q_val` (MacroSR over the validation tasks) must strictly
improve on the incumbent. The noise guard (`min_improved_episodes`, default 2; `max_regressed_episodes`,
default 1) means a validation set smaller than the threshold can never admit a candidate.

Reject and record a candidate when source or boundary replay fails, the candidate is not actually used,
a hard constraint, schema, integrity or reproducibility gate fails, a budget is exceeded, or the
comparison is not a strict improvement. A model outage is not evidence against a candidate: it stays
`pending` (blocked) with the reason recorded, and can be gated again. A gate also stays `pending`
when fewer validation tasks than required are registered, when the source task is itself a validation
task, or when the active program changed under the candidate.

## Skill and operator changes

Treat a new Skill as routing and evidence guidance, not as a change to how a task is accepted. Treat a
new Operator as a frozen, versioned, typed component with explicit inputs, outputs, a contract, and a
deterministic boundary-replay check.
When a candidate depends on a pretrained model or external service, record the model identifier,
license/authorization state, content hash and fallback behavior; do not silently download weights or
forward gateway credentials.

When a patch model writes a skill, require explicit applicability, a numbered procedure, pitfalls,
linked tool/operator refs and source provenance. Instance-specific tokens (task and session ids, absolute
paths) are scrubbed before a skill is stored. A generated skill may change retrieval, but may not change
a task's constraints or acceptance rule.

## Evidence to retain

Candidate records, bundles, gate reports and promotion receipts live under the engine state directory
(`home`); `scienceclaw_program(operation=history)` lists the promotions. Keep them so a promoted version
can be traced to its source session, its validation tasks and the gate decision. Do not create extra
audit documents for a routine run.
