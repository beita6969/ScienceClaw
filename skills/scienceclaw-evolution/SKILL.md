---
name: scienceclaw-evolution
description: Run ScienceClaw's program-level self-evolution loop when a benchmark candidate, tool wrapper, or skill patch must be proposed, replayed, validated on visible development data, and promoted with provenance and budget evidence. Use this for evolution planning and review; use scienceclaw-benchmark for dataset routing.
metadata:
  openclaw:
    emoji: "🧬"
---

# ScienceClaw self-evolution

Use the Python engine under `packages/scienceclaw-bench/scienceclaw/evolution/` as
the authoritative evolution implementation. The gateway's long-lived agent and
the benchmark engine remain separate: the gateway supplies sessions and skills;
the engine supplies typed programs, replay, receipts, and promotion decisions.

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

Do not use hidden ID/OOD items to select a candidate, tune a threshold, or
generate a skill. Formal evaluation is a separate server-side operation.

## Skill and tool changes

Treat a new skill as routing and evidence guidance, not as an undocumented
scorer change. Treat a new operator as a frozen, versioned tool with explicit
inputs, outputs, dependency notes, and a deterministic smoke or replay check.
When a candidate depends on a pretrained model or external service, record the
model identifier, license/authorization state, content hash, and fallback
behavior; do not silently download weights or forward gateway credentials.

## Evidence to retain

Every accepted or rejected candidate should leave a receipt containing the
incumbent/candidate fingerprints, split and item identifiers, visible score
comparison, hard-constraint result, replay result, logical token cost, wall
time, and rejection or promotion reason. Use the shared benchmark protocol in
`skills/scienceclaw-benchmark/references/protocol.md` for the episode-level
evidence vocabulary.
