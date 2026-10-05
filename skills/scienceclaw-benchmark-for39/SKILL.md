---
name: scienceclaw-benchmark-for39
description: "Use when a task asks to predict whether students answer held-out diagnostic multiple-choice questions correctly, from a large student-by-question correctness matrix and a small per-student budget of adaptively chosen revealed answers (about 10), delivered as a 0/1 prediction per target cell and judged by accuracy; Eedi / NeurIPS 2020 Education Challenge Task 4 adaptive testing with item response theory (FoR39 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Adaptive testing and student answer prediction

## Task
- Input: a training matrix `answers (n_train, Q)` (1 correct, 0 incorrect, -1 not answered; optional question subjects and student metadata); for the students to predict, `can_query (n, Q)` bool (cells that may be revealed) and `targets (n, Q)` bool (cells to predict). A querying tool declared by the task reveals a student's answers to chosen question ids, up to a per-student budget (10 in the companion benchmark).
- Deliverable: int array `(n, Q)` with 0 or 1 on every target cell and -1 elsewhere.
- Quality: accuracy over all target answers (the organiser averages over 10 query/target masks), higher is better. Reference: predict each question's most common training correctness.

## Routes
- Two-parameter IRT with `scilib.adaptive`: `fit_item_curves(answers)` (marginal-likelihood EM, seconds for thousands of students), `ability_posterior`, `predict_proba(model, revealed)`, `select_queries(model, can_query, revealed, k, budget, method="batch"|"bald")`, `predict(model, revealed, targets)`, `merge_revealed`. `model` may be the training matrix itself (the fit is cached per process). Typed operators: `irt_item_response_fit`, `irt_adaptive_question_selection`, `irt_correctness_prediction`.
- Pattern: split the budget into a few rounds. Round 1 `select_queries(answers, can_query, [], k1)`; query; round 2 `select_queries(answers, can_query, [rev1], k2)`; query; finally `predict(answers, [rev1, rev2], targets)`. Selection is deterministic and the list for k is a prefix of the list for a larger k.
- Before spending real queries, simulate the procedure on held-out training students (hide a random 20 percent of their answered cells as targets, query from the rest) to compare random, BALD and batch selection with the majority reference.

## Pitfalls
- Treat the budget as counting distinct revealed cells over the whole solve, including exploratory or later-discarded calls. Fix the number and size of rounds before querying: changing k or the round plan changes the cells later rounds ask for and can exceed the budget.
- Only answered, non-target questions are queryable; never try to read target answers or labels of the students being predicted, and fit item curves on the training matrix only.
- Each query call returns the `revealed` array for its own cells only (-1 not revealed, 0 incorrect, 1 correct). Keep every array and pass the list; do not pass `revealed_values` (chosen option 1-4), which is a different encoding.
- Per-student evidence is only about 10 answers, so accuracy gains over the majority reference are modest and noisy on a few dozen students; report the gain together with its sample size.
