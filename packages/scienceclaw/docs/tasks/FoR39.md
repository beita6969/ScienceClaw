# FoR39 Education — Eedi NeurIPS 2020 Education Challenge, Task 4 (organizer 10-mask accuracy, higher is better)

Adapter: `scienceclaw/bench/tasks/for39_eedi.py` (`Adapter = EediTask4Adapter`). Status: **available**.

## Dataset
| field | value |
|---|---|
| name / version | Eedi NeurIPS 2020 Education Challenge, official `data.zip` / `starter_kit.zip`, Task 3/4 members only |
| URL / license | https://www.eedischool.com/projects/neurips-education-challenge (competition data terms) |
| local path | `<DATA_ROOT>/for39-eedi-task4/source_members/data/` (members CRC-verified by the data team; the `.part` archive fragments are not used) |
| files | `train_data/train_task_3_4.csv` (4,918 students, 1,382,727 answers, 948 questions), `test_data/test_public_task_4_more_splits.csv` (615 students, 74,891 answers), `test_data/test_private_task_4_more_splits.csv` (615 students, 51,299 answers), `metadata/{question_metadata_task_3_4,subject_metadata,student_metadata_task_3_4}.csv` |
| parsing cache | dense int8 matrices in `cache/tasks/FoR39/*.npz` (source data untouched) |

Official protocol (starter kit `evaluation.py`, retained locally): for each of 10 masks (`IsTarget_0..9`) a fresh model queries 10 answers per test student one at a time, then predicts correctness of the target answers; accuracy over all targets per mask, averaged over the 10 masks.

## Items, pools, splits
* **Item** = one test student with **one** official mask (`mask = rank mod 10` of a salted hash rank → masks balanced within ±1). Id `eedi-u<UserId>`. Rationale: in a single workflow, reveals made under one mask could expose target answers of another mask of the same student; one mask per student removes that leakage path (the organizer loop avoids it by building a fresh model per mask).
* **IID pool** = official public test students (615) → src 338 / val 92 / id 185 (55/15/30 %, `partition_seed = 20260928`).
* **OOD pool** = official private-leaderboard test students (615). `lineage.ood_kind = "proxy_within_dataset"`: student-disjoint sample of the same population (fewer answers per student: 83 vs 122 on average); no demographic/time shift is claimed.
* Train (4,918), public and private students are pairwise disjoint (verified in tests).
* **Episode** = 16 students (block `e` of a seeded permutation). Capacity: src 21, val 5, id 11, ood 38.

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | 4,854 training students (all but 64 dev students): `answers` (1/0, −1 unanswered), `answer_values` (option 1–4), `question_subjects`, `subjects` table, `student_meta` (Gender, YearOfBirth, PremiumPupil) |
| `load_dev_inputs` / `query_dev_answers` / `score_dev` | the same protocol on 64 held-out training students with seeded 80/20 query/target masks (visible dev signal; `_dev_evaluate` is None). Dev queries are capped at 10 per call but not trace-audited |
| `load_eval_inputs` | `can_query`, `targets` (16×948 bool), `student_meta` — no correctness |
| `query_answers(selections)` | `selections` (16×k, k ≤ 10, question ids or −1) → `revealed` (1/0 at queried cells, −1 elsewhere), `revealed_values` (option 1–4), `receipt` (HMAC-signed JSON of the revealed cells) |

**Query rule** (strict reconstruction choice, same as the data team's harness): only answered, non-target questions of the item's mask are queryable; target answers are never revealed. The starter kit's own `can_query` does not exclude targets — documented deviation. **Budget**: ≤ 10 distinct revealed answers per student across the **whole solve** (review finding LEAK-7), not only the final workflow. `query_answers` may be chained (adaptive rounds); the hard constraint `query_budget` builds a per-solve ledger: (a) the `receipt` outputs of the tool nodes in the replay trace, and (b) every `receipt.pkl` the executor persisted under the solve's interactive-session directory (`<run_dir>/exec`) and the replay directory (`Trace.run_dir` = `<run_dir>/replay/kNNN`), recursively, so answers revealed by nodes the policy later removed or replaced, or by nodes inside operator bodies, are charged. Receipts are plain-string pickles (read with a restricted unpickler that resolves no globals; foreign `receipt` ports such as arrays are ignored), HMAC-verified (key = sha256 of the episode id and the hidden answers; unforgeable without the labels) and unioned per (student, question) cell, so identical replays count once. Dev receipts and receipts of other episodes do not count. It fails closed if the trace or its `run_dir` is missing or not in the solver layout, the session directory is absent, a ledger file is truncated or a signature is invalid. The evaluation of a replay sees the ledger as of that step, which is exactly the information the graph could have been built from (later steps cannot influence an earlier graph). Residual: a call that never persisted an output (timed out or failed to store) reveals nothing to the policy and is not counted.

## Metric, reference, acceptance (D_V)
* Metric: for each mask present in the episode, accuracy over all target answers of the students with that mask; score = mean over those masks (organizer formula with one mask per student). Pooled: correct/total summed per mask over episodes, then averaged over masks (invalid episodes counted with the reference predictions).
* Reference = organizer starter baseline: predict each question's most common training correctness (`argmax(bincount)`, ties → 0); its query policy (most-answered question) does not affect its predictions.
* Acceptance: `score >= reference + 0.02`.
* Hard constraints: `output_shape` (int array 16×948, values in {−1,0,1}, 0/1 on every target cell), `query_budget`.
* Calibration (adapter check, not a research result): reference accuracy ≈ 0.60–0.72 per episode (mean ≈ 0.65; historical frozen agent 0.62); a simple 1-parameter IRT ability fit on 10 popularity-chosen reveals reached +0.02 on average and was accepted on ~50 % of episodes.

## Domain library (`scilib/adaptive.py`)
The objective ends with `scilib.describe("adaptive")` (numpy/scipy only, deterministic, no data of its own). Building blocks: `fit_item_curves(answers)` (one-dimensional two-parameter IRT by marginal ML / EM on an 81-point ability grid with sparse matrices, about 5-10 s for 4,854 x 948), `ability_posterior` / `predict_proba` (posterior of a student given the revealed cells), `select_queries(model, can_query, revealed, k, budget, method)` (BALD or greedy BatchBALD over `can_query & not revealed`, capped by the per-student budget, ties to the smaller question id, list for k is a prefix of the list for a larger k), `predict(model, revealed, targets)` (0/1 on targets, -1 elsewhere), `merge_revealed`. `model` may be the fitted dict or the training matrix itself (fit cached per process; every sandboxed code node is a fresh process, so a node that passes the matrix pays one fit, about 5-10 s). `revealed` may be a list with the `revealed` output of every earlier `query_answers` call, because each call returns only its own cells.
* Determinism matters here: the budget ledger unions the receipts of the whole solve (exec + replay directories), so a replay that asks for different cells than the executed graph adds cells and can exceed 10. The library never uses random numbers or the clock (identical picks with 1 and 2 BLAS threads, verified).
* The module docstring documents the `-1 / 0 / 1` encoding of `revealed`, that `revealed_values` (option 1-4) is a different array, and the whole-session ledger; it states no reference, margin or expected gain (F5).
* Library-level check (hand-written graph: code node -> `query_answers` -> ... -> predict code node, run through the real Solver, sandbox, replay and evaluator; 3 `src` episodes, plan 4+3+3 reveals): primary 0.710 / 0.654 / 0.743 against references 0.615 / 0.634 / 0.666 (gain +0.095 / +0.020 / +0.077, all accepted). Offline simulation on all 75 rebuilt episodes (one seed): mean gain per episode +0.037 (random 10 reveals), +0.045 (BALD 10 reveals), +0.050 (batch selection, 4+3+3 rounds); about 56 of 75 accepted. End to end with the 27B policy (`configs/dev5_local.yaml`, same 3 `src` episodes, `runs/probe_for39_scilib`): 3/3 accepted, primary 0.696 / 0.654 / 0.742 (stock A_0 before: 0.697 / 0.646 / 0.716, 2/3 accepted), 9 steps, 45-62k policy tokens; the policy wrote two rounds of 5 reveals with `select_queries(answers, can_query, [rev1], k=5)` and `predict(answers, [rev1, rev2], targets)`. A guess-parameter (3PL) floor and an ability prior from the answered-set mask gave no robust gain and were not included.

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 300, max_llm_items 64`. Loading the training matrix from cache takes < 0.5 s.

## Deviations from the historical protocol
* Exact historical sample ids / masks were **not recovered**; rebuilt splits. id = 4×16 = 64 students, ood = 64 students.
* One official mask per student instead of all ten per student (leakage-safe; same expectation, higher variance).
* Strict target-hidden query rule; querying is a batch/chained tool call, not the organizer's 10 single-question rounds (chaining allows adaptivity).
* The data team's `reconstructed_v1` CSVs (128 source / 64 validation students with new 80/20 masks, 128 public-test evaluation students) are not used; the adapter reads the official members directly and uses the official public/private test masks for every evaluation item.
