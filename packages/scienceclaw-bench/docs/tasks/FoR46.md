# FoR46 Information and computing sciences — HumanEval → MBPP (execution pass@1, higher is better)

Adapter: `scienceclaw/bench/tasks/for46_code.py` (`Adapter = CodeAdapter`). Status: **available**.
Tests: `tests/test_task_FoR46.py` (21 tests, ~11 s; skipped when the data are missing).

## Dataset
| field | value |
|---|---|
| IID source | OpenAI HumanEval, `HumanEval.jsonl.gz` (164 problems), github.com/openai/human-eval @ `6d43fb980f9fee3c892a914eda09951f772ad10d`, sha256 `b796127e…79ef`, MIT license |
| OOD source | Google MBPP *sanitized* (`sanitized-mbpp.json`, 427 hand-verified problems), github.com/google-research/google-research/tree/master/mbpp @ `f46ca8374b4cddef97ca4208ad986049d74d296a`, sha256 `ca95deaa…c8e9`, CC BY 4.0 (dataset card) |
| local path | `<DATA_ROOT>/for46-humaneval-mbpp/` (`receipt.json`, `reconstructed_v2/sampling-manifest.json`, role `*.jsonl`) |
| roles | data-team normative sampling **`reconstructed_v2`** (seed 20260928, sha256 ranking of native ids): `source_train` 36, `validation` 64, `heldout_id` 64 (HumanEval), `heldout_ood` 64 and `heldout_extra` 64 (MBPP sanitized, official test range 11–510) |

The problems (prompt, tests, canonical solution) come from the two raw dataset files; the role files only choose
ids. The adapter reads `partitions[role].selected_ids` from `sampling-manifest.json` (MBPP ids are bare native
numbers, mapped to `MBPP/<n>`) and checks `count`, pairwise disjointness, dataset membership (HumanEval roles →
`HumanEval/*`, OOD roles → `MBPP/*` with native id in 11–510) and that every id exists. A malformed manifest raises
(reported by `available()`); it never silently switches to another role assignment. The manifest's sha256 is
recorded in every episode's `lineage.receipt` (`roles_version`, `roles_manifest_sha256`, `roles_status`).

## Items, pools, splits
* **Item** = one Python programming problem with hidden unit tests. Item id = `HumanEval/<n>` or `MBPP/<task_id>`.
* **IID = HumanEval**, **OOD = MBPP** (a genuinely different dataset: `lineage.ood_kind = "cross_dataset"`).
* Pools (fixed, seed independent) and the maximum number of item-disjoint episodes of 16:

| split | pool (v2 role) | size | max disjoint episodes (16 items) |
|---|---|---|---|
| `src` | `source_train` (HumanEval) | 36 | 2 per cycle (recycled, see below) |
| `val` | `validation` (HumanEval) | 64 | 4 |
| `id` | `heldout_id` (HumanEval) | 64 | 4 |
| `ood` | `heldout_ood` (MBPP) then reserve `heldout_extra` (MBPP) | 64 + 64 | 4 + 4 = 8 |

  Default config (rounds 7, n_val 2, n_id 4, n_ood 4, items_per_episode 16): src 7 episodes (recycled), val 2, id 4
  (= the whole pool), ood 4 (= the whole `heldout_ood` role). Requests beyond capacity raise `PoolExhausted`
  (val/id above 4, ood above 8).
* **Episode formation.** Episode `e` of a split is block `e` of a seeded permutation of the pool (seed = SplitPlan's
  per-split seed; `draw_blocks`), so `build_episodes` is deterministic and prefix-stable and val/id/ood episodes
  never share an item. Items are never repeated inside an episode. The four splits use disjoint pools, so
  cross-split lineage is disjoint by construction (SplitPlan's `LineageOverlapError` check passes).
* **Source recycling (36 problems, rounds = 7).** 36 problems give only 2 disjoint episodes of 16 (32 items, 4 left
  over). For 7 rounds the `src` split therefore starts a fresh seeded permutation after every 2 episodes: episodes
  0–1, 2–3, 4–5 and 6 belong to cycles 0, 1, 2, 3 (`lineage.src_cycle = [0,0,1,1,2,2,3]`). The two episodes of one
  cycle are item-disjoint; items recur across *source* episodes only (SplitPlan reports them as same-split
  duplicates; `lineage.src_items_recycled = true`) and never in val/id/ood. Which 4 problems of a cycle are left
  out depends on the seed. Changing `items_per_episode` changes the number of disjoint source episodes per cycle
  (36 // ipe); ipe > 36 raises `PoolExhausted`.
* **OOD layers.** `ood` episodes 0–3 come from `heldout_ood`; if more are requested, episodes 4–7 come from the
  reserve `heldout_extra` with their own seeded stream, so extending `n_ood` never changes the first four episodes
  (prefix-stable, tested). `lineage.ood_layer` is `"role"` or `"reserve"`. The reserve is disjoint from `heldout_ood`
  and from every HumanEval pool.
* `lineage` also carries `roles` (`reconstructed_v2`), `partition`, `item_ids`, `src_cycle` (src only),
  `ood_layer` (ood only) and `receipt`.
* **`replication` role.** In v2 the manifest's `replication` partition is D_rep = the already seen source prefix
  (`data_file` = `retention.jsonl`, byte-identical to `source_train.jsonl`), not new items, and is not a pool here:
  the framework's `rep` episodes are frozen copies of `src` episodes. The legacy file `replication.jsonl` (and the
  `adapter/replication-*` extracts) are the MBPP `heldout_extra` reserve, not D_rep; the adapter never reads them.

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_eval_inputs` | `problems`: 16 dicts with `prompt` (HumanEval stub with signature + docstring, or the MBPP task text), `entry_point`, `visible_tests` (newline-separated `assert` lines), `kind`; `entry_points` |
| `run_visible_tests(codes)` | runs each candidate against its **public** tests only, one fresh subprocess per problem (assembled exactly like the hidden evaluation); `results` (passed, n_tests, n_passed, error, failing_tests), `passed`, `visible_pass_rate`. config `test_timeout_s` 0.5–30 (default 5) |

Visible tests: HumanEval = the docstring examples (`>>>` doctests, and `f(x) ➞ y` / `==` / `=>` / `->` /
`returns` lines whose arguments and result are Python literals), rendered as `assert f(args) == expected`; examples
that the dataset's own canonical solution fails are dropped: 4 examples in 3 problems (docstring errata of
HumanEval/47 and /148, two recurrence-definition lines of HumanEval/130 read as examples); cached in
`cache/tasks/FoR46/visible_filter_v1.json`. 133 of 164 HumanEval problems (all MBPP problems) have ≥ 1 visible
test. MBPP = the first `test_list` assert (EvalPlus MBPP+ prompt convention) plus `test_imports`; the other asserts
are hidden. Hidden tests, `check()` functions and canonical solutions never reach the policy (tools return only
`Problem.public`); they are used only inside `evaluate`. There is no labelled training set (the solver is the
LLM). **Dev signal**: `Episode._dev_evaluate(y)` returns the visible-test pass rate of the submission (public tests
only; hidden tests are never run for the policy).

## Metric, reference, acceptance (D_V)
* **pass@1** with one program per problem, as in `human_eval.evaluation` (Chen et al. 2021): a problem is solved
  iff its program runs all hidden tests without exception. HumanEval program = `prompt + y[i] + "\n\n" + test +
  check(entry_point)` (official concatenation; a complete function definition in `y[i]` overrides the
  docstring-only stub, a bare body continues it); MBPP program = `y[i] + test_imports + all test_list asserts`.
* Official scorer (data team, `receipt.json` / v2 README): human-eval `unsafe_execute` with a 3 s native timeout,
  `estimate_pass_at_k(n=1, k=1)`, run under macOS Seatbelt; the 328 canonical references pass 328/328. Ours gives
  the same per-problem verdict for programs that are not borderline in time; our timeout is 6 s per statement
  (a loaded laptop CPU), and unlike the official run the pass rate is computed by this adapter, not read from the
  data team's output. Verified here: the canonical solutions of all five pools (292 problems: 36 + 64 + 64 + 64 +
  64) pass, and an empty solution scores 0 on all 292; both are asserted (on the evaluated episodes) in the tests.
* Execution: fresh `python -I` subprocess per problem, cwd = temp dir, minimal environment (no credentials,
  proxies to a dead port), 6 s SIGALRM limit per test statement, RLIMIT_CPU / RLIMIT_FSIZE, destructive
  `os`/`shutil`/`subprocess` functions disabled as in the official `reliability_guard`, and, new with v2 work,
  the socket `connect / bind / getaddrinfo / …` functions disabled inside the process (no network). Whole-process-group
  kill on overrun. The job (program + tests) is sent on stdin — no test file exists in the candidate's working
  directory — and the verdict is written to a private directory and must echo a per-run random nonce;
  `sys._getframe`, `inspect` and `gc` are disabled. On **macOS** the subprocess also runs under `sandbox-exec`
  (`(deny network*)`, file writes only inside its two private temp directories) when a one-time start-up probe shows
  the OS boundary works; nested/unsupported hosts fall back silently to the in-process guards, per job, and the
  result reports `sandbox` = `seatbelt` or `none`. `SCIENCECLAW_FOR46_SEATBELT=0` switches the OS layer off. This is
  best-effort isolation of LLM-written code, not a hard security boundary (as in the official harness, adversarial
  in-process introspection cannot be fully excluded).
* Pooled metric: pass@1 over all items of the episodes; an invalid output counts as 16 failures.
* **Reference baseline** (deterministic, visible data only): the *visible-example memorizer* — returns the
  documented output for argument tuples that appear in the visible examples, `None` otherwise (hidden-test
  outcome cached in `cache/tasks/FoR46/memorizer_pass_v1.json`, keyed by problem id; it solves 1 of the 292 problems
  of all roles, pass@1 ≈ 0.0034, so `norm_score` ≈ 10·pass@1).
* **Acceptance:** `pass@1 >= reference + 0.5`.
* `details`: `reference`, `norm_score` = pass@1 / max(reference, 0.1) clipped to [0, 10] (floor documented: with a
  ~0 reference, norm = 10·pass@1), `pooled_payload` = {`item_ids`, `passed`, `ref_passed`},
  `per_item_error_kind`.
* **Hard constraints:** `output_format` (list of exactly 16 strings) and `implemented` (for every problem the
  assembled program parses and its last top-level definition of the entry point has a body beyond a docstring).

## Domain library `scilib.codegen`
The episode objective ends with `scilib.describe("codegen")`, the module docstring of `scilib/codegen.py`. The library
packages the mechanical parts of a code-generation workflow so that a workflow graph stays short: `build_prompts`
(one generation prompt per problem, styles `direct` / `plan` / `examples`, built only from the fields returned by
`load_eval_inputs`), `sanitize` (model reply texts -> program sources), `repair_prompts` (prompts for the problems whose
visible tests fail, from the `run_visible_tests` output) and `select_best` (per problem, the candidate passing the most
visible tests). Code nodes cannot call the model, so the model call is an `llm` node with template `{item}` between two
code nodes; the docstring states the item and output shapes of that node, its item and token limits, and what the agent
sees of each output port.
* `sanitize` follows the assembly rule of the adapter (stub: prompt + y[i]; description: y[i] alone), the 6 s hidden limit
  and the `implemented` constraint. It extracts fenced blocks (also unterminated ones), keeps the entry-point definition
  and the top-level symbols it reaches, drops demo code (`print`, `input`, `__main__`, asserts), unused imports and
  `__future__` imports, renames a function with another name to the entry point (an alias assignment would not satisfy
  `implemented`), completes body-only replies of stub problems, and falls back to a syntactically valid placeholder.
* The library reads no hidden test and no data file; it uses only what `load_eval_inputs` and `run_visible_tests` return.
* Tests: `tests/test_scilib_codegen.py` (synthetic formats; canonical solutions in fenced+demo, body-only and truncated-fence
  form run through the real evaluator on one src and one ood episode), `tests/test_task_FoR46.py::test_objective_documents_the_domain_library`.

## What changed from `reconstructed_v1` to `reconstructed_v2`
| aspect | v1 | v2 |
|---|---|---|
| HumanEval roles | source 36 / validation 64 / heldout_id 64 | byte-identical (same ids, same files) |
| OOD role | `cross_dataset_mbpp`: 64 MBPP sanitized ids from the whole file | `heldout_ood`: 64 MBPP sanitized ids from the **official test range 11–510** only; sha256 ranking, seed 20260928. Only 12 of 64 ids coincide with v1 |
| OOD reserve | none (the adapter extended with its own sha256 ranking) | `heldout_extra`: 64 further test-range ids, disjoint from `heldout_ood` |
| `replication` | not present | D_rep = the seen source prefix (alias of `retention.jsonl`), not new items |
| manifest schema | `partitions[role].selected_ids/count/data_file/sha256` | same schema, plus `episode-protocol.json`; flag `fully_configured_agent_episode: false` |
| adapter pools | `val` = first 32 of `validation`, `src` = `source_train` + other 32 validation (68), ood extended by own ranking | `val` = full `validation` (64 → 4 episodes), `src` = `source_train` only (36), ood reserve = `heldout_extra` |

Consequences for the adapter: the old `val` split (2 episodes) now has capacity 4; the source pool shrinks from 68 to
36 problems (2 disjoint episodes per cycle instead of 4), so the 7 source episodes recycle four times; the MBPP
OOD episodes now use only official-test problems, and their extension is a data-team reserve instead of an
adapter-made ranking. Consequence for results: OOD items changed for 52 of 64 problems, so numbers measured with v1
are not comparable item-by-item with v2 numbers; the memorizer cache is keyed by problem id and remains valid.

## Fallbacks
`Adapter(roles=None)` (default) uses `reconstructed_v2`, then `reconstructed_v1`, then a seeded sha256 ranking of
the native ids with the v2 sizes (`lineage.roles = "hash_rank"`; MBPP restricted to ids 11–510). `roles="v2"|"v1"|"hash"`
forces one of them (`ValueError` when the requested manifest is missing). Under v1 the pools are src 36 /
val 64 / id 64 / ood 64 (MBPP `cross_dataset_mbpp`) with an adapter-made ood reserve. The data-root can be set with
`SCIENCECLAW_DATA_ROOT`.

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 180, max_llm_items 64 (= 4 × 16)`. Running
the hidden tests of 16 problems takes < 1 s on the laptop (8 parallel subprocesses) unless a candidate loops.

## Deviations from the historical protocol and caveats
* Exact historical sample ids / split rules were **not recovered**; these are rebuilt splits (data-team
  `reconstructed_v2` sampling). Historical protocol: 64 IID + 64 OOD items; here id = 4×16 = 64 HumanEval and
  ood = 4×16 = 64 MBPP items.
* Source episodes recycle HumanEval problems (the 36 source problems cannot fill 7×16 disjoint items); this is the
  only place where items repeat across episodes.
* `configs/episode-designs/FoR46.json` does not exist in this repository; the episode design is the default
  SplitPlan (rounds 7, ipe 16) documented above.
* MBPP prompts show one test (EvalPlus convention) instead of all three (original MBPP prompt), so that the
  hidden tests are not fully visible; the hidden check still runs all asserts.
* Hidden-test timeout 6 s per statement (official human-eval default 3 s) for a loaded laptop CPU.
* The sandbox is best-effort (see above); the Seatbelt layer exists on macOS only.
