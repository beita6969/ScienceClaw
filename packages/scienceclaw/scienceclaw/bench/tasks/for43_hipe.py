"""FoR43 History, heritage and archaeology — HIPE-OCRepair-2026 OCR post-correction, metric weighted cMER-micro (min).

Data: the pinned public release ``v0.9.5`` (commit ``bd463f7e``) of HIPE-OCRepair-Bench
(https://github.com/hipe-eval/HIPE-OCRepair-2026-data), unpacked by the data team under
``<DATA_ROOT>/for43-hipe-ocrepair-2026/upstream/data/v0.9``. Every record is one *transcription unit* (a
newspaper chunk/paragraph/article or a book page) with the raw OCR hypothesis and the manually verified
ground truth (semi-diplomatic transcription).

* **Item** = one transcription unit: input = the OCR text + document metadata (corpus, language, date,
  publication title, document type); hidden label = the ground-truth transcription. Item id =
  ``hipe/<dataset>/<lang>/<file split>/<document_id>``. The OCR ``quality_report`` (CER/WER computed against
  the ground truth) is never exposed.
* **Test sets / strata.** A *test set* is ``<dataset>/<language>`` as in the official ranking
  (e.g. ``icdar2017/en``, ``dta19-l1/de``).
* **Pools.** IID pool = the official competition newspaper test sets whose gold is public in v0.9.5
  (``icdar2017`` en/fr and ``impresso-snippets`` de/en/fr, file split ``test``). OOD pool = constituent
  corpora from *different collections and document types* that never appear in the IID pool:
  ``dta19`` (19th-century German *books* from the Deutsches Textarchiv, OCR artificially degraded at noise
  levels 0/1/2; official ``dev`` split — its test gold is masked) and ``overproof-combined`` (Trove / Chronicling
  America newspaper articles, official ``test`` + ``dev``). For DTA, the three noise levels of a page share the
  same ground truth; exactly one level per page is used (chosen by a hash of the page id), so no page occurs
  twice. ``lineage["ood_kind"] = "cross_corpus_within_benchmark"``: a different primary dataset (collection,
  OCR engine, document type) inside the HIPE-OCRepair-Bench umbrella — no second OCR post-correction
  benchmark is available locally.
* **Splits.** The IID pool is partitioned once (``partition_seed``) at the level of *natural documents*
  (ICDAR periodical document, Impresso page, DTA book, Overproof article; same grouping as the data team's
  ``reconstructed_v1``) into src / val / id (55 / 15 / 30 %), stratified by test set; splits are therefore item-
  and document-disjoint. Episodes draw ``items_per_episode`` units balanced over the test sets of their pool.
* **Visible data (D_E).** Labelled (OCR, ground truth) pairs from the *official train/dev* partitions of the
  same corpora as the episode's pool (IID: icdar2017 train/dev, impresso-snippets train/dev; OOD: dta19 train at
  levels 0/1/2, overproof train). Every natural document that contains an evaluation item of any split is
  removed from the visible pool (the official ICDAR test chunks come from periodicals whose other chunks are in
  the official train set). Per episode: ``n_train`` pairs via ``load_train`` and a disjoint ``n_dev`` slice whose
  ground truth is held by ``score_dev`` (the episode's ``_dev_evaluate`` is None).
* **Metric (D_V).** Weighted cMER-micro exactly as the official HIPE-OCRepair-2026 ranking: both texts are
  normalised with the official scorer's ``norm()`` (lower-case, ß->ss / ligature mappings, ``¬\\n`` and ``—\\n``
  removed, non-word characters -> space, whitespace collapsed); per transcription unit the character alignment
  counts (H, S, D, I) come from ``jiwer.process_characters(reference, hypothesis)`` (the scorer's dependency);
  per test set ``cMER_micro = ΣS+ΣD+ΣI / ΣH+ΣS+ΣD+ΣI``; the score is the weighted mean of the per-test-set
  values with the official weights (``dta19-l0/l1/l2`` 1/3 each, every other test set 1), renormalised over the
  test sets present in the episode. Reference: ``HIPE-OCRepair-scorer`` commit d1e76e44, ``ocrepair_eval.py``;
  participation guidelines §5.5.
* **Reference baseline** (deterministic): copy the OCR input (the official "no edits" baseline).
  **Acceptance:** ``weighted cMER-micro <= (1 - rel_margin) * reference`` (default rel_margin = 0.05, i.e. at
  least a 5 % relative error reduction over the uncorrected OCR).
* **Hard constraints:** list of ``n_items`` strings; every length plausible (normalised length within
  [0.5, 2.0] x the normalised OCR length — catches truncated or run-away outputs).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib

from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for43_45_47_48_50 import (
    PARTITION_SEED, Lazy, PoolExhausted, as_list, balanced_sample, c_list_length, check_split, draw_stratified,
    episode_id, episode_rng, ids_digest, norm_score, partition_groups, read_jsonl, receipt_info, resolve_data_root,
    stable_int,
)

CODE = "FoR43"
FAMILY = "Humanities & law"
DATASET_DIR = "for43-hipe-ocrepair-2026"
DATA_SUBDIR = Path("upstream") / "data" / "v0.9"
IID_TEST_SETS = ("icdar2017/en", "icdar2017/fr", "impresso-snippets/de", "impresso-snippets/en", "impresso-snippets/fr")
OOD_TEST_SETS = ("dta19-l0/de", "dta19-l1/de", "dta19-l2/de", "overproof-combined/en")
IID_FRACTIONS = {"src": 0.55, "val": 0.15, "id": 0.30}
LENGTH_RATIO = (0.5, 2.0)
_FNAME = re.compile(r"^hipe-ocrepair-bench_(?P<bench>v[0-9.]+)_(?P<ds>[^_]+)_(?P<dsver>v[0-9.]+)_(?P<split>[a-z\-]+)_"
                    r"(?P<lang>[a-z]{2})\.jsonl$")


def official_weight(test_set: str) -> float:
    """Official ranking weight of a test set: DTA levels 1/3 each, every other test set 1 (guidelines §5.5)."""
    return 1.0 / 3.0 if test_set.startswith("dta19-l") else 1.0


# ----------------------------------------------------------------------------------------------- official scoring
def hipe_norm(string: str) -> str:
    """Official HIPE-OCRepair normalisation (verbatim logic of ``hipe_ocrepair_scorer.ocrepair_eval.norm``, MIT)."""
    string = string.lower()
    string = string.replace("ß", "ss")
    string = string.replace("ꝛ", "r")
    string = string.replace("œ", "oe")
    string = string.replace("æ", "ae")
    string = string.replace("aͤ", "ä")
    string = string.replace("oͤ", "ö")
    string = string.replace("uͤ", "ü")
    string = string.replace("—\n", "")
    string = string.replace("¬\n", "")
    string = re.sub(r"[^\w]", " ", string, flags=re.UNICODE)
    string = re.sub(r"_", " ", string)
    string = re.sub(r"\s+", " ", string)
    return string.strip()


def char_counts(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    """(hits, substitutions, deletions, insertions) of ``jiwer.process_characters`` on *normalised* texts."""
    from jiwer import process_characters
    ref, hyp = hipe_norm(reference), hipe_norm(hypothesis)
    if not ref:
        raise ValueError("empty normalised reference")
    o = process_characters(ref, hyp)
    return int(o.hits), int(o.substitutions), int(o.deletions), int(o.insertions)


def mer_from_counts(c: Any) -> float:
    h, s, d, i = (float(x) for x in c)
    tot = h + s + d + i
    return 0.0 if tot == 0 else (s + d + i) / tot


def weighted_cmer_micro(counts: list[Any], test_sets: list[str]) -> tuple[float, dict[str, float]]:
    """Weighted mean over test sets of cMER-micro (official weights renormalised over present test sets)."""
    per: dict[str, np.ndarray] = {}
    for c, ts in zip(counts, test_sets):
        per.setdefault(ts, np.zeros(4, dtype=float))
        per[ts] += np.asarray(c, dtype=float)
    scores = {ts: mer_from_counts(v) for ts, v in per.items()}
    w = {ts: official_weight(ts) for ts in scores}
    tot = sum(w.values())
    return float(sum(w[ts] * scores[ts] for ts in scores) / tot), scores


# ----------------------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class _Unit:
    uid: str
    dataset: str          # e.g. "icdar2017", "dta19-l1"
    language: str
    test_set: str         # "<dataset>/<language>"
    file_split: str       # train | dev | test | dev-unmatched | train-unmatched
    doc_id: str
    group: str            # natural document group
    page: str             # DTA page key without level (others: doc id)
    ocr: str
    gt: str
    date: str
    title: str
    doc_type: str


def natural_group(dataset: str, language: str, doc_id: str) -> str:
    """Natural document of a unit (data team's reconstructed_v1 grouping rules)."""
    base = dataset.split("-l")[0] if dataset.startswith("dta19") else dataset
    if base == "icdar2017":
        g = doc_id.rsplit("_", 1)[0]
    elif base == "dta19":
        g = re.sub(r"_l\d+_p\d+$", "", doc_id)
    elif base == "impresso-snippets":
        g = re.sub(r"_par\d+$", "", doc_id)
    else:                                     # impresso-nzz page, overproof article
        g = doc_id
    return f"{base}/{language}/{g}"


def _load_units(root: Path, max_chars: int) -> dict[str, _Unit]:
    ddir = root / DATASET_DIR / DATA_SUBDIR
    units: dict[str, _Unit] = {}
    for path in sorted(ddir.glob("*/*/*.jsonl")):
        m = _FNAME.match(path.name)
        if m is None or m.group("split") == "masked-test":
            continue
        ds, lang, fsplit = m.group("ds"), m.group("lang"), m.group("split")
        for rec in read_jsonl(path):
            gt_block = rec.get("ground_truth") or {}
            if gt_block.get("exclude_from_icdar_evaluation") is True:
                continue
            gt = gt_block.get("transcription_unit") or ""
            ocr = (rec.get("ocr_hypothesis") or {}).get("transcription_unit") or ""
            if not hipe_norm(gt) or not hipe_norm(ocr) or len(ocr) > max_chars or len(gt) > max_chars:
                continue
            meta = rec.get("document_metadata") or {}
            did = str(meta.get("document_id"))
            uid = f"hipe/{ds}/{lang}/{fsplit}/{did}"
            page = re.sub(r"_l\d+_", "_", did) if ds.startswith("dta19") else did
            units[uid] = _Unit(uid, ds, lang, f"{ds}/{lang}", fsplit, did, natural_group(ds, lang, did), page, ocr, gt,
                               str(meta.get("date", "")), str(meta.get("publication_title", "")),
                               str(meta.get("document_type", "")))
    if not units:
        raise FileNotFoundError(f"no HIPE-OCRepair records under {ddir}")
    return units


@dataclass
class _Pools:
    eval_split: dict[str, dict[str, list[str]]]    # split -> test set -> uids
    train: dict[str, dict[str, list[str]]]         # "iid"/"ood" -> test set -> uids (visible pool)
    excluded_train_groups: int


def _build_pools(units: dict[str, _Unit], partition_seed: int) -> _Pools:
    # ---- IID evaluation pool: official test files of the ranking newspaper test sets
    iid = [u for u in units.values() if u.test_set in IID_TEST_SETS and u.file_split == "test"]
    part = partition_groups({u.uid: u.group for u in iid}, IID_FRACTIONS, f"{CODE}|{partition_seed}|iid",
                            {u.uid: u.test_set for u in iid})
    ev: dict[str, dict[str, list[str]]] = {}
    for s, ids in part.items():
        ev[s] = {}
        for uid in ids:
            ev[s].setdefault(units[uid].test_set, []).append(uid)
    # ---- OOD evaluation pool: DTA books (dev, one noise level per page) + Overproof (test + dev)
    ood: dict[str, list[str]] = {}
    dta_pages: dict[str, list[_Unit]] = {}
    for u in units.values():
        if u.dataset.startswith("dta19-l") and u.file_split == "dev":
            dta_pages.setdefault(u.page, []).append(u)
        elif u.test_set == "overproof-combined/en" and u.file_split in ("test", "dev"):
            ood.setdefault(u.test_set, []).append(u.uid)
    for page, variants in sorted(dta_pages.items()):
        variants = sorted(variants, key=lambda v: v.dataset)
        pick = variants[stable_int("dta-level", partition_seed, page) % len(variants)]
        ood.setdefault(pick.test_set, []).append(pick.uid)
    ev["ood"] = {ts: sorted(v) for ts, v in ood.items()}
    # ---- visible pools: official train/dev partitions, minus every natural document holding an eval item
    eval_groups = {units[uid].group for sp in ev.values() for lst in sp.values() for uid in lst}
    train: dict[str, dict[str, list[str]]] = {"iid": {}, "ood": {}}
    excluded = 0
    for u in units.values():
        if u.group in eval_groups:
            excluded += int(u.file_split in ("train", "dev"))
            continue
        if u.test_set in IID_TEST_SETS and u.file_split in ("train", "dev"):
            train["iid"].setdefault(u.test_set, []).append(u.uid)
        elif u.dataset.startswith("dta19-l") and u.file_split == "train":
            train["ood"].setdefault(u.test_set, []).append(u.uid)
        elif u.test_set == "overproof-combined/en" and u.file_split == "train":
            train["ood"].setdefault(u.test_set, []).append(u.uid)
    for pool in train.values():
        for ts in pool:
            pool[ts].sort()
    return _Pools(ev, train, excluded)


def _public(u: _Unit, index: int | None = None) -> dict:
    d = {"dataset": u.dataset, "language": u.language, "test_set": u.test_set, "document_type": u.doc_type,
         "date": u.date, "publication_title": u.title, "ocr_text": u.ocr}
    if index is not None:
        d = {"index": index, **d}
    return d


# ----------------------------------------------------------------------------------------------- adapter
class HipeOCRepairAdapter:
    """TaskAdapter for FoR43 (see module docstring)."""

    discipline = CODE
    name = "HIPE-OCRepair-2026"
    family = FAMILY
    metric = "weighted cMER-micro"
    direction = "min"
    task_type = "ocr_post_correction"

    def __init__(self, data_root: str | None = None, partition_seed: int = PARTITION_SEED, n_train: int = 96,
                 n_dev: int = 16, rel_margin: float = 0.05, max_chars: int = 12000, budget: Budget | None = None,
                 **_: Any) -> None:
        self.root = resolve_data_root(data_root)
        self.partition_seed = int(partition_seed)
        self.n_train, self.n_dev = int(n_train), int(n_dev)
        self.rel_margin = float(rel_margin)
        self.max_chars = int(max_chars)
        self.budget = budget
        self._units: Lazy[dict[str, _Unit]] = Lazy(lambda: _load_units(self.root, self.max_chars))
        self._pools: Lazy[_Pools] = Lazy(lambda: _build_pools(self._units.get(), self.partition_seed))
        self._ref_lock = threading.Lock()
        self._ref_counts: dict[str, tuple[int, int, int, int]] = {}

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        ddir = self.root / DATASET_DIR / DATA_SUBDIR
        if not ddir.is_dir():
            return False, f"missing HIPE-OCRepair data directory {ddir}"
        have = {m.group("ds") + "/" + m.group("lang") + "/" + m.group("split")
                for p in ddir.glob("*/*/*.jsonl") if (m := _FNAME.match(p.name))}
        need = [f"{ts}/test" for ts in IID_TEST_SETS] + [f"{ts}/dev" for ts in OOD_TEST_SETS] + \
               [f"{ts}/train" for ts in IID_TEST_SETS + OOD_TEST_SETS]
        missing = [n for n in need if n not in have]
        if missing:
            return False, f"missing HIPE-OCRepair files (dataset/lang/split): {missing}"
        try:
            import jiwer  # noqa: F401
        except ImportError:
            return False, "the 'jiwer' package (dependency of the official HIPE-OCRepair scorer) is not installed"
        return True, f"HIPE-OCRepair-Bench v0.9.5 at {ddir}"

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 1:
            raise ValueError("items_per_episode must be >= 1")
        units = self._units.get()
        pools = self._pools.get()
        order = OOD_TEST_SETS if split == "ood" else IID_TEST_SETS
        rng = episode_rng(CODE, split, seed, self.partition_seed)
        draws = draw_stratified(pools.eval_split[split], int(n), int(items_per_episode), rng, f"{CODE}/{split}",
                                strata_order=order)
        return [self._episode(units, pools, split, k, int(seed), items) for k, items in enumerate(draws)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Weighted cMER-micro over all items of all episodes (invalid outputs scored as the copy-OCR reference)."""
        counts: list[Any] = []
        sets: list[str] = []
        for p in per_episode:
            if not p:
                continue
            ts = list(p.get("test_sets") or [])
            c = p.get("counts_pred")
            if c is None:
                c = p.get("counts_ref")
            if c is None or len(c) != len(ts):
                raise ValueError("FoR43 pooled payload needs test_sets and counts_pred (or counts_ref) of equal length")
            counts.extend(c)
            sets.extend(ts)
        if not counts:
            return None
        return weighted_cmer_micro(counts, sets)[0]

    # ------------------------------------------------------------ helpers
    def _copy_counts(self, u: _Unit) -> tuple[int, int, int, int]:
        with self._ref_lock:
            c = self._ref_counts.get(u.uid)
        if c is None:
            c = char_counts(u.gt, u.ocr)
            with self._ref_lock:
                self._ref_counts[u.uid] = c
        return c

    def _visible_sample(self, pools: _Pools, pool: str, ep_key: str) -> tuple[list[str], list[str]]:
        """Seeded dev slice and training sample, each balanced over the pool's test sets (disjoint)."""
        rng = np.random.default_rng(stable_int(CODE, "visible", ep_key))
        dev = balanced_sample(pools.train[pool], self.n_dev, rng)
        train = balanced_sample(pools.train[pool], self.n_train, rng, exclude=set(dev))
        return sorted(train), sorted(dev)

    # ------------------------------------------------------------ episode
    def _episode(self, units: dict[str, _Unit], pools: _Pools, split: str, k: int, seed: int, items: list[str]
                 ) -> Episode:
        eid = episode_id(CODE, split, seed, k)
        pool = "ood" if split == "ood" else "iid"
        ev = [units[i] for i in items]
        n_items = len(ev)
        tr_ids, dev_ids = self._visible_sample(pools, pool, eid)
        tr = [units[i] for i in tr_ids]
        dv = [units[i] for i in dev_ids]
        n_tr, n_dev = len(tr), len(dv)
        test_sets = [u.test_set for u in ev]
        present = sorted(set(test_sets))
        ocr_norm_len = [len(hipe_norm(u.ocr)) for u in ev]

        train_rows = [{**_public(u), "gt_text": u.gt} for u in tr]
        dev_rows = [_public(u, j) for j, u in enumerate(dv)]
        eval_rows = [_public(u, j) for j, u in enumerate(ev)]
        dev_ref = Lazy(lambda: weighted_cmer_micro([self._copy_counts(u) for u in dv], [u.test_set for u in dv])[0])

        # ---- D_E tools (visible data only)
        def load_train(inputs: dict, config: dict) -> dict:
            return {"train": [dict(r) for r in train_rows]}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_items": [dict(r) for r in dev_rows]}

        def score_dev(inputs: dict, config: dict) -> dict:
            pred, why = as_list(inputs.get("dev_predictions"))
            if pred is None or len(pred) != n_dev:
                raise ValueError(f"dev_predictions must be a list of {n_dev} strings ({why or f'got {len(pred)}'})")
            if not all(isinstance(p, str) for p in pred):
                raise ValueError("every dev prediction must be a string")
            counts = [char_counts(u.gt, p) for u, p in zip(dv, pred)]
            score, per = weighted_cmer_micro(counts, [u.test_set for u in dv])
            return {"dev_weighted_cmer_micro": score, "dev_reference_weighted_cmer_micro": float(dev_ref.get()),
                    "dev_per_test_set": per}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"items": [dict(r) for r in eval_rows]}

        def text_mer(inputs: dict, config: dict) -> dict:
            a, wa = as_list(inputs.get("texts_a"))
            b, wb = as_list(inputs.get("texts_b"))
            if a is None or b is None or len(a) != len(b):
                raise ValueError(f"texts_a and texts_b must be lists of equal length ({wa or wb})")
            counts, mers = [], []
            for x, y in zip(a, b):
                if not isinstance(x, str) or not isinstance(y, str) or not hipe_norm(x):
                    raise ValueError("texts_a must be non-empty strings and texts_b strings")
                c = char_counts(x, y)
                counts.append(c)
                mers.append(mer_from_counts(c))
            tot = np.sum(np.asarray(counts, dtype=float), axis=0) if counts else np.zeros(4)
            return {"mer": np.asarray(mers, dtype=float), "micro_mer": mer_from_counts(tot)}

        text_item = "dict: index, dataset, language, test_set, document_type, date, publication_title, ocr_text"
        tools = [
            ToolSpec("load_train", f"{n_tr} labelled transcription units from the official train/dev partitions of "
                     "the same corpora (no document shared with any evaluation unit). Each entry: dataset, "
                     "language, test_set, document_type, date, publication_title, ocr_text (raw OCR) and gt_text "
                     "(manually verified transcription).",
                     {}, {"train": PortSchema("list", (n_tr,), dtype="dict", description="labelled OCR/GT pairs")},
                     load_train),
            ToolSpec("load_dev_inputs", f"{n_dev} further labelled-pool units (disjoint from load_train) whose "
                     "ground truth is withheld; score corrections of them with score_dev.",
                     {}, {"dev_items": PortSchema("list", (n_dev,), dtype="dict", description=text_item)},
                     load_dev_inputs),
            ToolSpec("score_dev", "Weighted cMER-micro (official HIPE-OCRepair normalisation and weights; lower is "
                     "better) of dev_predictions (one corrected text per dev item, same order) against the withheld "
                     "dev ground truth, plus the same metric for the uncorrected OCR and per-test-set cMER-micro.",
                     {"dev_predictions": PortSchema("list", (n_dev,), dtype="str", description="corrected dev texts")},
                     {"dev_weighted_cmer_micro": PortSchema("number", unit="1"),
                      "dev_reference_weighted_cmer_micro": PortSchema("number", unit="1"),
                      "dev_per_test_set": PortSchema("dict", description="test set -> cMER-micro")},
                     score_dev),
            ToolSpec("load_eval_inputs", f"The {n_items} evaluation transcription units (ground truth hidden), in the "
                     "order that y must follow.",
                     {}, {"items": PortSchema("list", (n_items,), dtype="dict", description=text_item)},
                     load_eval_inputs),
            ToolSpec("text_mer", "Character-level match error rate between paired texts after the official HIPE "
                     "normalisation: mer[i] = cMER(reference=texts_a[i], hypothesis=texts_b[i]); micro_mer pools "
                     "the alignment counts.",
                     {"texts_a": PortSchema("list", ("m",), dtype="str"), "texts_b": PortSchema("list", ("m",), dtype="str")},
                     {"mer": PortSchema("array", ("m",), unit="1", dtype="float"), "micro_mer": PortSchema("number", unit="1")},
                     text_mer),
        ]

        # ---- D_V constraints
        def c_strings(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            lst, why = as_list(yv)
            if lst is None:
                return False, why
            bad = [j for j, t in enumerate(lst) if not isinstance(t, str)]
            return not bad, ("all entries are strings" if not bad else f"non-string entries at positions {bad[:10]}")

        lo, hi = LENGTH_RATIO

        def c_length(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            lst, why = as_list(yv)
            if lst is None or len(lst) != n_items:
                return False, why or f"len(y) != {n_items}"
            bad = []
            for j, t in enumerate(lst):
                if not isinstance(t, str):
                    bad.append(j)
                    continue
                r = len(hipe_norm(t)) / max(ocr_norm_len[j], 1)
                if not (lo <= r <= hi):
                    bad.append(j)
            return not bad, ("all lengths plausible" if not bad else
                             f"normalised length outside [{lo}, {hi}] x OCR length at positions {bad[:10]}")

        constraints = [
            c_list_length(n_items, "one corrected transcription per evaluation unit"),
            ConstraintSpec("strings", "every entry of y is a string (the corrected transcription)", c_strings),
            ConstraintSpec("length_plausible", f"for every unit, the normalised length of the correction is within "
                           f"[{lo}, {hi}] x the normalised length of its OCR text", c_length),
        ]

        # ---- D_V evaluator (hidden labels live only in this closure)
        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            ref_counts = [self._copy_counts(u) for u in ev]
            ref, ref_per = weighted_cmer_micro(ref_counts, test_sets)
            payload = {"item_ids": list(items), "test_sets": list(test_sets), "counts_pred": None,
                       "counts_ref": [list(c) for c in ref_counts]}
            base = {"reference": ref, "reference_name": "copy the OCR input (official no-edit baseline)",
                    "rel_margin": self.rel_margin, "n_items": n_items, "test_sets_present": present,
                    "reference_per_test_set": ref_per}
            lst, why = as_list(yv)
            if lst is not None and len(lst) != n_items:
                lst, why = None, f"length {len(lst)} != {n_items}"
            if lst is not None and not all(isinstance(t, str) for t in lst):
                lst, why = None, "non-string entries"
            if lst is None:
                return EvalResult(metrics={"reference_weighted_cmer_micro": ref}, primary=None, direction="min",
                                  accepted=False, details={**base, "norm_score": 0.0, "pooled_payload": payload,
                                                           "invalid": why})
            counts = [char_counts(u.gt, t) for u, t in zip(ev, lst)]
            score, per = weighted_cmer_micro(counts, test_sets)
            mers = [mer_from_counts(c) for c in counts]
            ref_mers = [mer_from_counts(c) for c in ref_counts]
            pref = float(np.mean([np.sign(r - m) for m, r in zip(mers, ref_mers)]))
            tot = np.sum(np.asarray(counts, dtype=float), axis=0)
            payload["counts_pred"] = [list(c) for c in counts]
            metrics = {"weighted_cmer_micro": score, "reference_weighted_cmer_micro": ref,
                       "cmer_micro_unweighted": mer_from_counts(tot), "cmer_macro": float(np.mean(mers)),
                       "pref_score_cmer_macro": pref}
            return EvalResult(metrics=metrics, primary=score, direction="min",
                              accepted=bool(score <= (1.0 - self.rel_margin) * ref),
                              details={**base, "norm_score": norm_score(score, ref, "min"), "per_test_set": per,
                                       "pooled_payload": payload})

        objective = (
            "History, heritage and archaeology - OCR post-correction of historical documents (HIPE-OCRepair-2026). "
            "Each evaluation item is one transcription unit (a newspaper chunk, paragraph or article, or a book page; "
            "English, French or German, 17th-20th century) given as its raw OCR text with metadata (dataset, "
            "language, test_set, document_type, date, publication_title). No page images are available. The hidden "
            "target is the manually verified semi-diplomatic transcription: historical spelling and archaic forms "
            "are kept; line breaks and soft hyphens ('¬' + newline) need not be reproduced.\n"
            f"Visible data: load_train returns {n_tr} labelled units (ocr_text, gt_text) from the official "
            f"train/dev partitions of the same corpora; load_dev_inputs returns {n_dev} further units whose ground "
            "truth is withheld and score_dev scores corrections of them; load_eval_inputs returns the evaluation "
            "units; text_mer computes normalised character match error rates between paired texts.\n"
            f"Deliverable y: a list of {n_items} strings; y[i] is the corrected transcription of items[i] from "
            "load_eval_inputs (same order).\n"
            "Evaluation: both texts are normalised as in the official scorer (lower-cased, ß->ss and ligature "
            "mappings, punctuation and other non-word characters replaced by spaces, whitespace collapsed); per test "
            "set, cMER-micro = (S+D+I)/(H+S+D+I) summed over its units from a character alignment; the score is the "
            "weighted mean over the test sets present (dta19-l0/l1/l2 weight 1/3 each, all others 1). Lower is "
            "better.\n"
            + scilib.describe("ocrfix")
        )
        lineage = {
            "dataset": "HIPE-OCRepair-Bench", "version": "v0.9.5 (commit bd463f7e340280473d9e5bab5e231f4922ea1a3b)",
            "source_url": "https://github.com/hipe-eval/HIPE-OCRepair-2026-data",
            "license": "per corpus: icdar2017 CC BY-NC-SA 4.0; impresso-snippets CC BY-NC-SA 4.0; dta19 CC BY-NC 4.0 "
                       "(DTA pages CC BY-SA 4.0); overproof research use only",
            "receipt": receipt_info(self.root / DATASET_DIR / "receipt.json"),
            "scorer": "HIPE-OCRepair-scorer d1e76e447629ea9cf8dead32ae3c44b0d48d77b3 (norm + jiwer character alignment)",
            "pool": pool, "test_sets": present,
            "pool_source": ("dta19 official dev split (one noise level per page) + overproof-combined official "
                            "test and dev" if pool == "ood" else
                            "official labelled test files icdar2017 en/fr + impresso-snippets de/en/fr"),
            "ood_kind": "cross_corpus_within_benchmark" if pool == "ood" else None,
            "ood_shift": ("different collections / document types / OCR engines: DTA 19th-c. German books with "
                          "synthetic OCR degradation, Trove + Chronicling America newspapers") if pool == "ood" else None,
            "split": split, "split_seed": seed, "partition_seed": self.partition_seed, "index": k,
            "item_ids": list(items), "n_items": n_items,
            "groups": sorted({u.group for u in ev}),
            "train_source": "official train/dev partitions of the pool's corpora, documents holding eval items removed",
            "n_train": n_tr, "n_dev": n_dev, "train_ids_sha256": ids_digest(tr_ids),
            "dev_ids_sha256": ids_digest(dev_ids), "dev_item_ids": list(dev_ids),
            "rebuilt_split": True, "historical_ids_recovered": False,
        }
        return Episode(
            id=eid, discipline=CODE, family=FAMILY, split=split, task_type=self.task_type, objective=objective,
            required_output=PortSchema("list", (n_items,), dtype="str",
                                       description="corrected transcription per evaluation unit, load_eval_inputs order"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=3600.0, max_node_s=900.0,
                                         max_llm_items=8 * n_items),
            lineage=lineage,
            acceptance=(f"weighted cMER-micro <= {1.0 - self.rel_margin:g} x the weighted cMER-micro of the uncorrected "
                        "OCR (copy-input reference) on the same units"),
            tolerance={"rtol": 1e-6, "atol": 1e-8},
            tags=[CODE, "history", "heritage", "OCR", "post-correction", "historical-newspapers", "text", "cMER",
                  "HIPE-OCRepair", *sorted({u.language for u in ev}), pool],
            metric=self.metric, direction=self.direction, n_items=n_items,
            _evaluate=evaluate, _dev_evaluate=None,
        )


Adapter = HipeOCRepairAdapter

__all__ = ["Adapter", "HipeOCRepairAdapter", "hipe_norm", "char_counts", "weighted_cmer_micro", "official_weight",
           "PoolExhausted"]
