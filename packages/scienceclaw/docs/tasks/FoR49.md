# FoR49 Mathematical sciences — SMT-COMP 2025 QF_NonLinearIntArith single query (oracle-agreement accuracy, higher is better)

Adapter: `scienceclaw/bench/tasks/for49_smt.py` (`Adapter = SmtAdapter`). Status: **available** (requires the
frozen difficulty screen `scienceclaw/bench/tasks/for49_screen_v1.json.gz` and a z3 binary).
Tests: `tests/test_task_FoR49.py`.

## Dataset
| field | value |
|---|---|
| name / version | SMT-LIB release 2025 (non-incremental benchmarks), logics `QF_NIA` and `QF_NIRA`, Zenodo DOI 10.5281/zenodo.15493090 (2025-05-22) — the library from which SMT-COMP 2025 draws the QF_NonLinearIntArith single-query division |
| URL / license | https://zenodo.org/records/15493090, https://smt-comp.github.io/2025/results/ — CC-BY-4.0 unless a benchmark states otherwise |
| local path | `<DATA_ROOT>/for49-smtcomp2025-nonlinearintarith/source_library/non-incremental/{QF_NIA,QF_NIRA}` (complete archives, publisher MD5 + local sha256 verified by the data team), candidate frame `reconstructed_v1/candidate-frame.jsonl` |
| eligible population | 20,184 of 25,455 files (one matching `set-logic`, one `check-sat`, definite `:status`, byte duplicates removed; 5,271 unknown-status files excluded); ≤ 1 MB: **19,930** |
| label | the published SMT-LIB `(set-info :status …)` (the label SMT-COMP scores against; not an independent certificate) |
| solver | Z3 5.1.0 (binary of the `z3-solver` wheel in the venv; `SCIENCECLAW_Z3` overrides) |

## Items, difficulty screen, pools, splits
* **Item** = one eligible benchmark (≤ 1 MB). The agent receives a *sanitized* query: `;` comments and every
  `set-info` (status, source, license, category) plus `get-*`/`echo` commands are removed (asserted: no `:status`
  survives). Tools never expose file names. Item id = SMT-LIB path.
* **Difficulty screen** (SMT-COMP removes easy benchmarks; here a solver screen replaces the unavailable
  competition selection): every eligible item was run once through z3 5.1.0 with a **1-second CPU-time limit**
  (RLIMIT_CPU — independent of machine load) and `rlimit=16,000,000`, memory 2 GB. *hard* = undecided, *easy* =
  decided and equal to the official status, *conflict* = decided against the official status (excluded).
  Frozen in `for49_screen_v1.json.gz` (hard/conflict ids + digest of the eligible set; regenerate with
  `python -m scienceclaw.bench.tasks.for49_smt screen 8` then `... freeze`). Results: 9,201 hard / 10,729 easy / 0 conflict among the 19,930 eligible items.
* **Lineage unit**: benchmark family + program/problem stem (e.g. `20170427-VeryMax/CInteger/Stroeder_15__GCD2.c`,
  VeryMax property points and numbered variants collapsed; MathProblems grouped by puzzle template). One
  representative per (group, status, tier), chosen by a seed-independent hash.
* **IID** = families `20170427-VeryMax`, `20220315-MathProblems`, `calypto` (groups cut once into train / dev / src /
  val / id = 25/5/35/10/25 %). **OOD** = all other families (AProVE, LassoRanker, leipzig, sqrtmodinv-hoenicke,
  mcm, UltimateAutomizer, UltimateLassoRanker, elster, …): `lineage.ood_kind = "proxy_within_dataset"`
  (generator/family shift inside SMT-LIB; there is no second SMT benchmark set).
* **Episode** = 16 items: 12 hard (6 sat + 6 unsat) + 4 easy; total `#sat = 6 + (h mod 5)` ∈ [6, 10]
  (h = sha256 of split seed and episode index). Strata are drawn block-wise from seeded permutations
  (prefix-stable; item- and lineage-group-disjoint across splits). Units (hard sat/unsat, easy sat/unsat): train 171/79/559/235, dev 31/13/109/48, src 201/99/808/321, val 62/22/221/92, id 150/74/554/239, ood 517/169/1351/558 (6,683 units). Capacity (16 items, limited by hard-unsat units): src 16, val 3, id 12, ood 28 episodes.

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | 48 labelled sanitized queries (18 hard sat + 18 hard unsat + 12 easy sat) of the train groups, `status` |
| `load_dev_inputs` / `score_dev(dev_pred)` | 16 dev queries (episode composition) / `dev_accuracy`, `dev_reference_accuracy`, `dev_n_wrong_definite` (visible dev signal; `_dev_evaluate` is None) |
| `load_eval_inputs` | the 16 sanitized evaluation queries, in output order |
| `z3_check(queries)` | per query `status` ∈ {sat, unsat, unknown}, `time_s`, `reason` (timeout / memout / error), `z3_version`. config: `query_timeout_s` 1–120 (default 10, z3 `-T`; the executor's own node-level `timeout_s` still caps the whole call), `memory_mb` 256–4096, `workers` 1–8, `rlimit` (deterministic resource limit), `params` (z3 `name=value` tokens). Runs in fresh subprocesses in a temp dir; z3 error output ⇒ `unknown`; results memoized per (query, configuration) within a process |

**Domain library `scilib.logic`** (shown to the policy through `scilib.describe("logic")` at the end of the objective):
`check` / `solve_all` (Z3 Python API, one context per query, `threads` queries at a time, wall limit + deterministic
`rlimit` + process-wide memory cap; rejected script/parameter -> `unknown`, reason `error: ...`), `majority_status`, and
`decide(queries, train_status=None, fallback=None, budget_s=240, ...)`: escalating rlimit/wall rounds (16e6 x 4^r, 8 s x 4^r)
over the still-undecided queries with the remaining budget divided over the waiting queries; a returned sat/unsat is
Z3's own answer, undecided queries get the `fallback` label (`sat`/`unsat`/`unknown`/`majority` of `train_status`).
Motivation (frozen run `full_local-20260929-132008`, A_0): id e01 (0.75 vs reference 0.5625) and ood e00 (0.8125 vs 0.625)
missed the accuracy clause by one item; the agent spent 14 steps on z3 parameter errors that the tool reported as
`memout` / `no answer (exit 110)` (fixed: the rejected parameter is now reported in `reason`) and left the undecided
queries as `unknown`, which never agrees with the official status.

## Metric, reference, acceptance (D_V)
* **Oracle-agreement accuracy** = fraction of items with `y[i] ==` official status; `unknown` never agrees. Also
  reported: `n_wrong_definite` (sat↔unsat errors), `n_unknown`. Pooled metric = accuracy over all items
  (invalid episodes contribute the reference predictions).
* **Reference**: majority status of the episode's training labels (always `sat`: 30 of 48) → accuracy = #sat/16.
* **Acceptance:** `accuracy >= reference + 0.2`.
* `details`: `reference`, `norm_score` (= accuracy/reference, clipped [0, 10]), `pooled_payload` (`item_ids`,
  `y_true`, `y_pred`, `y_ref`).
* **Hard constraints:** `output_format` (list of 16 strings), `allowed_labels` ({sat, unsat, unknown}).
* Calibration (adapter check, not a research result): plain `z3_check` with a 10 s timeout on four episodes
  (id seed 1, ood seed 2) gave accuracy 0.875 / 0.6875 / 0.8125 / 0.9375 (1–5 `unknown`, 0 wrong definite
  answers) against thresholds 0.70 / 0.825 / 0.70 / 0.575 → 3 of 4 accepted; answering `sat` for the unknowns
  accepted all 4. Under heavy machine load (10 s wall ≈ a few CPU-s) only ~6 of 16 hard items were decided. The
  task is therefore moderately hard for a z3-using agent; a second-stage screen with a larger CPU budget would
  make the hard tier harder (open option).

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 2400, max_node_s 900, max_llm_items 32`.

## Deviations from the historical protocol
* Exact SMT-COMP 2025 selected/scrambled instances and the historical 64/64 sample ids were **not recovered**;
  these are rebuilt splits from the SMT-LIB 2025 source library. id = 4×16 = 64 and ood = 4×16 = 64 items.
* The easy-benchmark removal is a z3 5.1.0 1-CPU-second screen (not the competition's previous-year solver
  results); 75 % of each episode is screen-hard, balanced in status.
* OOD = held-out benchmark families (within-library proxy).
