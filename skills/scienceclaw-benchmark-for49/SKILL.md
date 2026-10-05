---
name: scienceclaw-benchmark-for49
description: "Use when a task gives SMT-LIB v2 scripts (quantifier-free nonlinear integer or integer-real arithmetic, QF_NIA / QF_NIRA, SMT-COMP style single-query instances) and asks for the satisfiability verdict of each one (sat, unsat or unknown) as a list in input order, judged by agreement with the status published with the benchmark. Corresponds to FoR49 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# SMT satisfiability (QF_NIA)

## Task
- Inputs: a list of SMT-LIB scripts (comments and `set-info` metadata removed), optionally some labelled example scripts with their status.
- Deliverable: a list of strings, same length and order, each `sat`, `unsat` or `unknown`.

## Quality
Accuracy: the fraction of scripts whose label equals the reference status. `unknown` never agrees, and a wrong definite answer (sat vs unsat) is the worst outcome, so report correct / wrong-definite / unknown counts separately. A baseline is the majority status of the labelled examples for every script.

## Tools
- `scilib.logic.solve_all(queries, threads=2, timeout_s=10, rlimit=None, memory_mb=2048, params=None)` and `check(query, ...)`: Z3 Python API, one context per query; each result has `status`, `reason` (timeout, resource limit, out of memory, or `error: ...` for a rejected script or parameter) and `seconds`. Typed operator `smt_satisfiability_batch`.
- `scilib.logic.decide(queries, train_status=None, fallback=None, budget_s=240, threads=2, first_rlimit=16_000_000, first_timeout_s=8, factor=4, max_rounds=4)`: escalating rounds (rlimit and wall limit grow by `factor`) over the still-undecided queries within one wall budget, returning `labels`, `status`, `stage`, `reason`, `n_decided`. Operator `smt_escalating_decide`. The node's own time limit must exceed `budget_s`.
- Needs the `z3-solver` package (check with `scienceclaw_tools(operation=show, target=logic)`); no pretrained weights. The memory cap is process-wide, shared by parallel threads.

## Route
1. Run `decide` with the budget the task allows (threads up to the CPU cores), or `solve_all` with a short limit and then rerun the undecided scripts with larger limits.
2. A Z3 `sat` or `unsat` is final; never override it with a heuristic. A `reason` starting with `error:` means a bad parameter or unsupported syntax, not a hard instance: fix and rerun instead of guessing.
3. For scripts still undecided, report `unknown`, or, if scored by agreement, use `fallback="majority"` with `train_status` and state clearly that these labels are guesses, not solver answers.

## Rules
- Do not read `:status` annotations if any survive in a script; that is the label.
- Only asserted formulas are solved (`check-sat` and `set-info` are ignored by the parser); keep `set-logic` as given.
- Keep the output length, order and label vocabulary exactly as required.
