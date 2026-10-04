---
name: scienceclaw-benchmark
description: "Route and reproduce an integrated ScienceClaw FoR30–FoR52 benchmark run, including frozen tool selection, visible-dev scoring, self-evolution, and evidence-backed score diagnosis. Use for benchmark work; use the general gateway skills for unrelated tasks."
metadata: { "openclaw": { "emoji": "🧪" } }
---

# ScienceClaw benchmark orchestration

Use this skill when a request concerns the paper's 23-discipline benchmark, a
benchmark score, a frozen SOTA component, a tool-on run, or the self-evolution
loop that produces and validates skills/operators.

## Routing

1. Call `scienceclaw_bench(operation=catalog)` once at the start of a task and
   record the adapter, tool refs, config names, and launcher names it returns.
   Call `operation=list_tasks` when availability or data-root status matters.
2. Load exactly one matching dataset skill, `scienceclaw-benchmark-for30`
   through `scienceclaw-benchmark-for52`, and read its task document before
   selecting a tool. Its metric direction and output contract take precedence
   over generic SOTA intuition.
3. Use `load_train` and the adapter's visible `score_dev` (or its documented
   visible substitute) to compare one frozen route with a named baseline. Keep
   tool/checkpoint/preprocessing provenance in the run receipt.
4. Use `scienceclaw_bench(operation=report)` with the `runId` of an existing
   server-side run to inspect its report.
5. Keep the benchmark engine and gateway separate: the gateway routes and
   records; the Python package owns typed graphs, adapters, replay, scoring,
   self-evolution, and receipts. Read only the relevant reference below.

## Self-evolution contract

The agent may propose skills and operators, but every candidate must be
attributable to a concrete source episode and replayable from its bundle. The
promotion order is `source replay → visible D_val → H_val → logical-token and
wall budgets → strict Q_val improvement → snapshot`. A skill is a routing
recipe; it must not hide labels, alter an evaluator, or silently change a
dataset split. Read `scienceclaw-evolution` for the candidate and receipt
contract.

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

For score diagnosis, separate a route failure from an accounting failure:
check split coverage, metric direction, pooled-vs-episode score kind, tool
provenance, and the report's expected episode counts before changing a model.
If a visible comparison does not beat the incumbent under the same config and
budget, record the rejection and move to the next route. Do not tune against
an evaluated formal item.

Dataset-specific routing lives in the 23 `scienceclaw-benchmark-for*` skills.
