# FoR47 Language, communication and culture — CoNLL-2018 UD dependency parsing (LAS, higher is better)

Adapter: `scienceclaw/bench/tasks/for47_ud.py` (`Adapter`; shared helpers in `_adapter_utils_for43_45_47_48_50.py`).
Status: **available** (UD_French-GSD train/dev/test and UD_French-PUD test present).

## Dataset
| field | value |
|---|---|
| name / version | Universal Dependencies **2.2** (the CoNLL 2018 shared-task release), `ud-treebanks-v2.2.tgz` (publisher MD5 `699c40b259f0d6b4935598411664fcf3`) |
| URL | https://hdl.handle.net/11234/1-2837 (LINDAT) |
| license | per treebank (UD 2.2 licence page `license-UD-2.2.html`); fr_gsd CC BY-SA 4.0, fr_pud CC BY-SA 3.0 |
| local path | `<DATA_ROOT>/for47-ud-conll2018/subset/ud-treebanks-v2.2/<treebank>/*.conllu` (complete members extracted from the official archive; the full archive is also in `reconstructed_v1/upstream/`) |
| official scorer | `conll18_ud_eval.py` v1.2 (MPL 2.0; copy in `reconstructed_v1/evaluator/upstream/`) |

## Items, pools, splits
* **Item** = one sentence with **gold tokenization**: syntactic words (FORM) in order + multiword-token ranges
  (e.g. `du` = `de le`); label = HEAD and DEPREL per word. Evaluation sentences have 3–40 words. Item id
  `ud/<treebank>/<official split>/<sent_id>`.
* **IID** = `UD_French-GSD`: src / val = official **dev** file partitioned once by sentence (75 / 25 %, `partition_seed
  = 20260928`; 995 / 331 sentences), id = official **test** file (372 sentences).
* **OOD** = `UD_French-PUD` official test file (924 sentences; parallel news/Wikipedia sentences annotated by another
  team). `ood_kind = "cross_treebank"` (a different dataset of the same language). As in CoNLL 2018, PUD has no training
  data and is parsed with the source treebank's training data. Configurable: `Adapter(ood_treebank=...)` (e.g.
  `UD_Russian-Taiga` for a cross-language OOD; `iid_treebank` likewise).
* Episodes: 16 sentences drawn prefix-stably from a seeded permutation; items never repeat inside a split.

## Visible data and tools (D_E)
Per episode a seeded sample of the official **train** file of the IID treebank (train, dev and test files are
disjoint in UD; sentences whose word sequence equals an evaluation sentence are removed defensively):

| tool | returns |
|---|---|
| `load_train` | 1000 annotated sentences (id, form, lemma, upos, xpos, feats, head, deprel per word; multiword tokens) + the same as CoNLL-U text |
| `load_dev_inputs` | 64 further training sentences (3–40 words, gold tokenization only) |
| `score_dev(dev_parses)` | official LAS / UAS on the dev slice + reference LAS (visible dev signal; `_dev_evaluate` is None) |
| `load_eval_inputs` | the 16 evaluation sentences (gold tokenization, no trees) in output order |
| `read_conllu(conllu)` | CoNLL-U reader |
| `check_trees(parses, n_words?)` | official tree-validity rules (heads in range, one root, no cycle) + UD relation inventory |

## Required output
`y`: list of 16 dicts `{"head": [int]*n_words, "deprel": [str]*n_words}` (1-based heads, 0 = root), in word order.
(The evaluator also accepts a list of `[head, deprel]` pairs per sentence.)

## Metric, reference, acceptance (D_V)
* **LAS** of the official CoNLL 2018 evaluator with gold tokenization: fraction of words whose head and DEPREL
  (universal part before `:`; subtypes ignored) match; punctuation included; micro over all words of the episode (the
  official script scores a whole file). `tests/test_task_for47.py::test_matches_official_conll18_evaluator` writes gold and
  system CoNLL-U files and checks LAS and UAS against the unchanged official `evaluate()` (agreement to 1e-12).
  Pooled metric: words-weighted LAS over the given episodes. UAS reported as auxiliary.
* **Reference** = right-branching chain (word i → i+1, last word root) with each word's most frequent non-root relation
  (by lower-cased form) in the episode's training sample. Default plan pooled reference LAS: src 0.260, val 0.261,
  id 0.250, ood 0.265.
* **Acceptance**: `LAS >= reference + 0.15` (`margin` configurable).
* `details`: `reference`, `norm_score` (LAS/reference clipped to [0,10]), `pooled_payload` (`n_words`, `las_correct`,
  `uas_correct`, `ref_las_correct`).
* **Hard constraints**: `output_length`, `parse_format` (one int head + one relation per word), `valid_tree` (official
  evaluator rules — it rejects files with multiple roots or cycles), `ud_relations` (37 UD v2 relations).

## Domain library (`scilib/udparse.py`)
The episode objective ends with `scilib.describe("udparse")`, the module docstring: the sentence/parse dict formats of the tool
outputs, `UDParser` (UPOS tagger = linear SVM on word/affix/shape/neighbour features; edge-factored arc scorer over hashed
head/dependent/context/between-material features, averaged passive-aggressive training; relation labeller = linear SVM on hashed
arc features; arcs and labeller trained on 5-fold out-of-fold tags), the exact single-root decoders `eisner` (projective) and
`chu_liu_edmonds`, `fit_predict(train, targets)` (one call: fit, then parse every list of target sentences), `cross_validate`,
`las_uas` (the adapter's metric, parity tested) and `check_parse`. It needs only FORM of the target sentences (no upos / lemma), never
returns an invalid tree (a guard falls back to a valid tree), and reads no file. The docstring states no reference, margin or
recommendation (F5). Tests: `tests/test_scilib_udparse.py` (decoders against brute force for n <= 5, metric parity with the
adapter, sandbox import, synthetic treebank) and `tests/test_task_for47.py::test_objective_documents_the_domain_library`.
* Optional pretrained parser, `fit_predict(train, targets, pretrained=True)`: parses the target sentences with `scilib.udparse_pretrained`
  (Stanza French-GSD models, CamemBERT-large encoder; `train` unused). Listed in the `udparse` docstring; the training data of those
  models (French-GSD train/dev seen, other treebanks not) is stated in the `udparse_pretrained` docstring. Test:
  `tests/test_pretrained_options.py` (routing and the unavailable-case error, with a stand-in parser).
* **Measured** (hand-written node `fit_predict(load_train, [load_dev_inputs, load_eval_inputs])` in the sandbox runner + episode
  evaluator; ~30 s per node incl. 1000 training sentences, load ~50): src episodes 0/1/2 LAS 0.752 / 0.790 / 0.791 vs reference
  0.255 / 0.300 / 0.267 (acceptance line reference + 0.15), norm score 2.95 / 2.63 / 2.96; dev LAS 0.764 / 0.815 / 0.765 for the
  same fit. On 400 held-out `src` sentences (private harness, 1000 training sentences): UPOS 0.945, LAS 0.774 / UAS 0.820
  (Eisner) and 0.766 / 0.813 (Chu-Liu/Edmonds); with gold tags 0.816. `use_morph=True` (+0.0) and `n_models=2` (+0.002) are not worth
  their cost; 12 epochs are no better than 8.
  The same node on `ood` (UD_French-PUD) episodes 0/1/2: LAS 0.721 / 0.726 / 0.700 vs reference 0.252 / 0.256 / 0.238, all accepted.
* **End to end** (`scripts/probe.py --config configs/dev5_local.yaml --disciplines FoR47 --split src --n 2`, local 27B policy): 2/2
  accepted, primary 0.717 / 0.798, norm 2.90 / 3.29, 8 / 9 steps, 47k / 53k tokens (before, without the library, `runs/probe_local27b_a0`: failed, LAS
  0.003, norm 0.012, 24 steps, 208k tokens). The agent calls `fit_predict` in its own code nodes and reads `score_dev` before finishing.
* **Known gaps**: first-order arc scores only (no second-order / neural features), relation labeller is arc-local (no joint
  decoding), multiword-token ranges are accepted but unused.

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 2400, max_node_s 600, max_llm_items 8 × max(items, dev) = 512`.

## Known deviations
* Exact historical sample ids / treebank choice were **not** recovered: rebuilt splits.
* Gold tokenization is given (the CoNLL 2018 task started from raw text); MLAS/BLEX are not computed (no gold
  morphology is requested).
* Evaluation sentences are limited to 3–40 words.
* The data team's `reconstructed_v1` 8-language sample is not used; this adapter uses one IID treebank and a
  cross-treebank OOD as required by the protocol.
