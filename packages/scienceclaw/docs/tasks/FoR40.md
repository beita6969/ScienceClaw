# FoR40 Engineering — DCASE 2024 Task 2 first-shot anomalous sound detection (official DCASE score, higher is better)

Adapter: `scienceclaw/bench/tasks/for40_dcase.py` (`Adapter = DCASE2024Task2Adapter`), design `FoR40.v3`.
Status: **available** with the data team's `reconstructed_v3` delivery. The default plan (src 7 / val 2 / id 4 episodes of
16 clips) fits; **the OOD split is empty by design** (see below).

## Dataset
| field | value |
|---|---|
| name / version | DCASE 2024 Task 2, development set, Zenodo 10902294 (seven machine types, one section each); data team delivery `reconstructed_v3` (design `configs/episode-designs/FoR40.v3.json`) |
| URL | https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring |
| license | CC-BY-NC-SA-4.0 (Zenodo record) |
| local path | `<DATA_ROOT>/for40-dcase2024-task2/reconstructed_v3/` — `roles/{source,val,id}.json`, `source-normal-support.json`, the WAV files referenced by their `path` fields, `runtime/dcase_native_v3/` (data team's native evaluator and baseline receipts) |
| format | 16 kHz mono 16-bit PCM WAV, 10 s (12 s for ToyCar / ToyTrain); decoded with numpy / scipy only |
| machines | ToyCar, ToyTrain, bearing, fan, gearbox, slider, valve |

Frozen sha256 of the role/support files (pinned in the adapter, `FROZEN_SHA256`; a mismatch makes `available()` False):

| file | sha256 (prefix) | records |
|---|---|---|
| `roles/source.json` | `cd337ac8016f` | 7 cohorts × 30 probe clips = 210 |
| `roles/val.json` | `889c44989d75` | 7 cohorts × 30 = 210 |
| `roles/id.json` | `9f9b25e106ef` | 7 cohorts × 20 = 140 |
| `source-normal-support.json` | `1a21f2390889` | 7 machines × 24 = 168 |

Natural role counts: source 210, validation 210, ID 140, source_fit_support 168, OOD 0.

## Items, pools, splits
* **Item** = one probe clip; output = anomaly score. Item ids are the delivery's opaque one-time handles
  (`dcase2024t2/v3/<machine>/<public_id>`), never the label-bearing official file names.
* **Native cohort** = one machine type inside one role, containing both domains and both classes. src and val cohorts hold
  30 clips (10 source-normal, 10 source-anomaly, 5 target-normal, 5 target-anomaly); id cohorts hold 20 (5 per stratum).
* **Splits** (nothing is discovered or re-partitioned): `src` = `roles/source.json`, `val` = `roles/val.json`,
  `id` = `roles/id.json`. The three files are clip- and PCM-disjoint (re-checked at load), so src/val/id episodes are
  item-disjoint (lineage) for any seed; `partition_seed` only salts the episode draws.
* **Episode** = 16 clips of ONE machine type, 4 per (domain × condition) stratum drawn without replacement from that
  machine's cohort, plus the machine's fit support (below). Machines are visited round-robin in a seeded order
  (prefix-stable in `n`). Test-clip domains are not given (as in the challenge).
* **Capacity at 16 clips/episode**: each cohort supports exactly ONE episode per split (a 16-clip episode uses 4 of the 5
  target clips), i.e. src 7, val 7, id 7. With `items_per_episode = 8` two episodes per cohort (14 per split); 24 items is
  impossible. The default plan needs 7 / 2 / 4.
* **OOD is empty.** The delivery contains no unseen-machine or cross-dataset pool and both official domains occur
  inside src/val, so no OOD is invented. `build_episodes("ood", ...)` returns `[]`; `SplitPlan` records the shortfall as
  a warning ("FoR40/ood: adapter returned 0 of 4 requested episodes"), the pooled OOD score is None, and
  `available()` requires no OOD capacity. `lineage["ood_kind"]` is None.

## Fit support (`source_fit_support`)
The 168 pinned official *source-domain normal training* clips (24 per machine) are the visible training data of an
episode (`load_train`): every episode of machine M, in every split, sees the same 24 clips of M. They are file- and
PCM-disjoint from all 560 probe clips, are **not** evaluation items (absent from `lineage["item_ids"]`, listed in
`lineage["train_item_ids"]`, so they never trigger a lineage overlap) and contain **no target-domain clip and no
anomaly**: the target-domain shift has to be inferred, as in the challenge. Train clips carry the official attribute
strings.

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_train` | the machine's 24 source-normal training clips: `waveforms` (n×T float32, zero-padded), `lengths`, `domains` (all "source"), `attributes`, `sample_rate` |
| `load_eval_inputs` | the 16 probe clips' `waveforms`, `lengths`, `sample_rate` (no names, labels or domains) |
| `log_mel_spectrogram` | log-mel power spectrogram in dB (Hann, centred; n_fft 1024, hop 512, n_mels 128) |

No visible dev signal (first-shot setting: no labelled anomalies for a machine type): `_dev_evaluate` is None and there
is no `score_dev` tool.

## Domain library (`scilib/anomsound.py`)
The objective ends with `scilib.describe("anomsound")` (numpy / scipy / scikit-learn only, deterministic, loads nothing of its own, about a second per episode). Building blocks: `log_mel_list(waveforms, lengths, sample_rate)` (the same log-mel as `log_mel_spectrogram`, cut to each clip's true length; `trim_mels` does the same for the tool's padded output), `clip_descriptors(mels, kind)` ('ms' per-mel mean+std, 'm', 'mx', 'q95'), `component_scores(train_mels, eval_mels)` (six raw anomaly scores: `nn_train`, `nn2_pool`, `nn_pool`, `lof`, `band_max`, `maha`; the first is the 1-NN statistic of the stock reference, `nn2_pool` / `nn_pool` / `lof` place the unlabeled eval clips in the point cloud together with the 24 training clips, `band_max` searches the nearest training clip per mel band on the temporal maximum, `maha` is a Ledoit-Wolf shrunk Mahalanobis distance), `rank_average(scores, members)` (mean rank in (0, 1], ties averaged), `fit_predict` / `score_clips` (one call from the mel lists or from the `load_train` / `load_eval_inputs` outputs, members default to `DEFAULT_MEMBERS = (nn2_pool, lof, nn_pool, band_max, maha)`), `probe_report` (Spearman matrix between components, diagnostic) and `official_score` (the task metric for arrays with known labels). The docstring is factual (what each function takes and computes); it states no reference value, margin or recommended recipe and no property of the evaluation clips (`tests/test_scilib_anomsound.py` and `tests/test_task_FoR40.py::test_objective_documents_the_domain_library` assert this on the visible text).
* Design evidence (everything computed from visible training clips plus the hidden labels only for scoring in offline experiments): frame-level scoring methods were tried and dropped (no gain over clip-level statistics with 24 training clips); a rank average of complementary clip-level scores was the only variant with a stable gain. Offline on 168 fresh rebuilt episodes (seeds 100-107, all splits): mean gain over the reference +0.023 (SD 0.07 per episode), accepted in 54 % (src +0.027 / 55 %, val +0.009 / 45 %, id +0.034 / 62 %). Individual components are much more variable than the ensemble on 16-clip episodes (per-episode AUC noise is about 0.07).
* Library-level check (hand-written one-node graph `score_clips(load_train, load_eval_inputs)` through the real sandbox and evaluator; 7 `src` episodes, seed 20260928): primary against reference 0.426/0.423, 0.405/0.422, 0.756/0.642, 0.553/0.635, 0.774/0.703, 0.741/0.721, 0.486/0.465; accepted 4/7. Same node on val and id (7 episodes each): accepted 3/7 and 5/7.
* End to end with the 27B policy (`configs/dev5_local.yaml`, the 3 `src` episodes of `runs/probe_pass10_n3`, output `runs/probe_for40_scilib`): stock A_0 before 0.458 / 0.209 / 0.754 against references 0.596 / 0.303 / 0.719 (1/3 accepted, 8-10 steps, 36-53k policy tokens); after 0.630 / 0.294 / 0.735 (1/3 accepted, 5 steps, about 24k tokens; the policy ran `log_mel_list` -> `component_scores` -> `rank_average` in one code node, once with all six components, twice with `DEFAULT_MEMBERS`). The acceptance flips on a coin toss: episode 02 is 0.004 below reference + 0.02, episode 01 has one anti-correlated component (`maha` alone 0.08 there). Four further `src` episodes (`--skip 3 --n 4`, `runs/probe_for40_scilib_b`): 4/4 accepted (primary 0.325 / 0.799 / 0.584 / 0.304), no stock-A_0 comparison for those.
* Caveat: acceptance stays noise-limited (16 clips per episode, few target-domain clips); the library raises the mean primary but cannot make individual episodes deterministic passes.

## Pretrained audio embeddings (`scilib/audioenc.py`, `anomsound.embedding_scores`)
`scilib.describe_extra("audioenc")` is appended to the objective only when `audioenc.available()` (local torch + weights, or the remote GPU bridge). `audioenc.embed(waveforms, lengths, sample_rate, model, kind)` returns one float32 vector per clip: model `ast_audioset` (`MIT/ast-finetuned-audioset-10-10-0.4593`; kind `pooled` 768 / `logits` 527, fp16) or `clap_htsat` (`laion/clap-htsat-unfused` audio tower; kind `pooled` 768 / `proj` 512, fp32). Clips are resampled to the model rate (16 / 48 kHz) and cut into windows of at most 10.24 s / 10 s (one window, or the first and last window); the vector is the mean over windows. `anomsound.embedding_scores(train_emb, eval_emb)` returns `nn_train`, `nn2_pool`, `nn_pool`, `lof` on L2-normalised rows (same pool convention as `component_scores`). Both docstrings are factual (inputs, outputs, training corpora, cost); no value, recipe or property of the evaluation clips (`tests/test_scilib_audioenc.py`, `tests/test_adapter_visible_text.py`). Weights on the GPU host: `models/audioenc/{ast_audioset,clap_htsat_unfused}`; worker whitelist entry `("audioenc", "embed")`.
* Cost (shipped code, GPU 2 of the shared H800 host): AST 120-140 clips/s, CLAP 28 clips/s; one bridge call with 24 support + 16 probe clips takes 28-32 s per model. The WAV rows are float32 (1, n_samples), one bridge chunk each; `reports/below_peers_rootcause/scripts/f40/prestage_blobs.py` creates the chunks on the GPU host from the staged WAVs, so no waveform has to be uploaded (hashes of all 728 clips checked against the adapter's own arrays: 0 mismatches).
* Offline result with the pre-declared pipeline (rank average of `nn2_pool` on AST `pooled` and CLAP `pooled`, no log-mel member), official pooled score, current log-mel ensemble in brackets: src 0.646 (0.450), val 0.517 (0.533), id 0.559 (0.547), harness id episodes 0.592 (0.547). Mean over the three cohorts 0.574 vs 0.510, diff +0.064, 95% CI (-0.012, +0.235), better on 2 of 3 cohorts. Details and caveats: `reports/below_peers_rootcause/f40.md`.
* **Option in the main library (added 2026-10-01).** Same reason as the time-series tools (0 of 6 FoR40 episodes imported `scilib.audioenc`): `anomsound.score_clips(train, evalset, members=None, embed=None, embed_members=('nn2_pool',))`; with `embed=['ast_audioset', 'clap_htsat']` the score is the rank average of `<model>:nn2_pool` over the models (the pre-declared pipeline; no log-mel member unless `members` is given). `embed=None` (default) is the previous call and makes no GPU call. Tests: `tests/test_pretrained_options.py`.

## Metric, reference, acceptance (D_V)
* Metric (replicates `dcase2024_task2_evaluator.py`): AUC(source) over source normals + all anomalies; AUC(target) over
  target normals + all anomalies; pAUC = `roc_auc_score(y, s, max_fpr=0.1)` over all clips; official score = harmonic mean
  of the three (values floored at float eps). Pooled: items grouped by machine over all episodes, three values per machine,
  harmonic mean over machines. On the complete native cohorts the pooled metric equals the data team's
  `runtime/dcase_native_v3` baseline primaries to 1e-10 (src 0.4321, val 0.5121, id 0.4848; asserted in the tests).
* Reference: 1-nearest-neighbour distance of standardized per-mel mean/std log-mel statistics to the machine's 24 training
  clips (deterministic).
* Acceptance: `score >= reference + 0.02`.
* Hard constraints: `c_vector` (1-D, length = items), `c_finite`.
* Levels on the default plan (7 / 2 / 4 episodes of 16 clips). Pooled reference primary: src 0.380 / val 0.529 /
  id 0.503 (per-episode reference ranges 0.20-0.74 src, 0.52-0.54 val, 0.40-0.66 id). The tests' trivial hand-written
  solution (calls `load_train` / `load_eval_inputs`, scores by the mean squared standardized distance of log-RMS, spectral
  centroid and zero-crossing rate to the training clips) reaches pooled 0.341 / 0.501 / 0.420, i.e. it is accepted on
  4/7, 1/2, 2/4 episodes: the reference is neither trivially beaten nor out of reach. Per-episode scores on 16 clips are
  coarse (few target clips); the pooled values over machines are the meaningful quantity.

## Integrity checks (fail closed)
`available()` loads the design (frozen sha256 of the four files, declared stratum counts, source-normal support, uniqueness
of handles / PCM / file hashes / official ids over all 728 clips, every cohort machine has support), checks that every
referenced WAV exists with the recorded size, and checks capacity against the default plan. Every WAV read re-verifies
sha256, frame count and sample rate. The `path` fields of the records are absolute paths on the data team's machine and are
re-based below the configured data root at the `for40-dcase2024-task2/` marker.
`Adapter.verify_files()` re-hashes all 728 files.

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 300, max_llm_items 2 x items`.

## Deviations and caveats
* Episodes hold 16 clips of one machine instead of the full 200-clip machine section, so per-episode AUCs are coarse;
  cohorts are 30 (20) clips per machine, so one episode per cohort and split at 16 clips.
* The visible training data is 24 source-domain normal clips per machine (the official set has 990 source + 10 target
  training clips); there is no target-domain training clip.
* Only a single section per machine and no formal OOD split: generalisation to unseen machine types is not measured.
* Historical (paper) sample ids were not recovered; the split is the data team's frozen design, not the paper's.
* Item ids are opaque handles; the official file names (which encode domain and label) are in the receipts, not in the
  visible data.
