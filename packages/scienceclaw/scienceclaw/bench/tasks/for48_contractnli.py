"""FoR48 Law and legal studies — ContractNLI evidence identification, metric mAP (max).

Data: ContractNLI (Koreeda & Manning, Findings of EMNLP 2021), initial release 2021-10-05, the official
``contract-nli.zip`` (https://stanfordnlp.github.io/contract-nli/, CC BY 4.0) extracted by the data team under
``<DATA_ROOT>/for48-contractnli/upstream/contract-nli``: 607 non-disclosure agreements (official train 423 /
dev 61 / test 123), each annotated for the same 17 hypotheses with an NLI label (Entailment / Contradiction /
NotMentioned) and the evidence *spans* (sentences or list items; candidate spans are given per document).

* **Item** = one (contract, hypothesis) pair whose gold label is Entailment or Contradiction — exactly the pairs
  on which the official evidence-identification metrics are defined (the official scorer drops gold
  NotMentioned pairs). Input: the contract text, its candidate spans, the hypothesis; hidden labels: the
  evidence span set and the NLI label. Item id = ``contractnli/<document id>/<hypothesis key>``. Pairs whose
  evidence set is empty are excluded (AP undefined).
* **Pools (proxy OOD).** No second contract-NLI dataset is available locally; the shift is the *document source*
  inside ContractNLI: IID = contracts found by web search (``document_type = search-pdf``), OOD = contracts
  from SEC EDGAR filings (``sec-text`` / ``sec-html``), which differ in layout, drafting conventions and length
  (``lineage["ood_kind"] = "proxy_within_dataset"``). Evaluation documents come from the official *dev + test*
  partitions; IID documents are partitioned once (``partition_seed``) *by document* into src / val / id
  (50 / 15 / 35 %), so no contract is shared between splits. Episodes draw ``items_per_episode`` pairs.
* **Visible data (D_E).** A per-episode sample of ``n_train_docs`` fully annotated contracts of the official
  *train* partition from the same source type (all 17 hypotheses: label + evidence span indices), and
  ``n_dev_docs`` further train contracts whose annotations are held by ``score_dev`` (the episode's
  ``_dev_evaluate`` is None). ``load_hypotheses`` lists the 17 hypotheses.
* **Metric (D_V).** Evidence-identification mAP: for every item the average precision of its span scores against
  the gold evidence spans (``sklearn.metrics.average_precision_score``, as in the official
  ``contract_nli/evaluation.py::evaluate_spans``), averaged over items — the official
  ``micro_label_macro_doc.span.map`` aggregation (mean AP over document–hypothesis pairs). Auxiliary: the
  official precision at 80 % recall (same aggregation, ``p_at_r80``; optimistic), the same precision pooled over
  all spans of all pairs (``p_at_r80_pooled``, the aggregation the ContractNLI paper reports for P@R80), the
  ``macro_label_micro_doc`` mAP (AP pooled per hypothesis, averaged over hypotheses), the binary NLI accuracy
  (Entailment vs Contradiction, micro over pairs) and the accuracy of the per-hypothesis majority label
  (``majority_label_accuracy``, the trivial label baseline the accuracy has to be read against). The historical
  paper's choice among the four official aggregations was not recovered (see the task card).
* **Reference baseline** (deterministic): span score = fraction of the hypothesis' content-word stems present in
  the span (lexical overlap); label = the majority of Entailment/Contradiction for that hypothesis in the visible
  training contracts. **Acceptance:** ``mAP >= reference mAP + margin`` (default 0.10).
* **Hard constraints:** list of ``n_items`` dicts; every ``span_scores`` has one finite value in [0, 1] per
  candidate span of its contract; every ``label`` is ``Entailment`` or ``Contradiction``.
"""
from __future__ import annotations

import collections
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib
from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for43_45_47_48_50 import (
    PARTITION_SEED, Lazy, PoolExhausted, as_list, c_list_length, check_split, draw_stratified, episode_id,
    episode_rng, ids_digest, norm_score, partition_groups, read_json, receipt_info, resolve_data_root, stable_int,
)

CODE = "FoR48"
FAMILY = "Humanities & law"
DATASET_DIR = "for48-contractnli"
DATA_SUBDIR = Path("upstream") / "contract-nli"
IID_TYPES = ("search-pdf",)
OOD_TYPES = ("sec-text", "sec-html")
IID_FRACTIONS = {"src": 0.50, "val": 0.15, "id": 0.35}
LABELS = ("Entailment", "Contradiction")
_STOP = set("a an the of to in on for by with and or any all some shall may not be is are was were that this "
            "which from as at its it their such other than under upon".split())


# ----------------------------------------------------------------------------------------------- official metrics
def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """``sklearn.metrics.average_precision_score`` (official ContractNLI span mAP building block)."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


def precision_at_recall(y_true: np.ndarray, y_prob: np.ndarray, recall: float = 0.8) -> float:
    """Verbatim logic of the official ``contract_nli.evaluation.precision_at_recall`` (Apache-2.0)."""
    from sklearn.metrics import precision_score
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return float("nan")
    threshs = np.sort(np.unique(y_prob))[::-1]
    y_preds = y_prob[None, :] >= threshs[:, None]
    recalls = np.logical_and(y_true[None, :], y_preds).sum(axis=1) / np.sum(y_true)
    thresh = threshs[np.where(recalls >= recall)[0][0]]
    return float(precision_score(y_true, y_prob >= thresh, zero_division=0.0))


def _stems(text: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


def overlap_scores(hypothesis: str, span_texts: list[str]) -> np.ndarray:
    hs = _stems(hypothesis)
    if not hs:
        return np.zeros(len(span_texts))
    return np.asarray([len(hs & _stems(t)) / len(hs) for t in span_texts], dtype=float)


# ----------------------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class _Doc:
    doc_id: str
    official_split: str
    doc_type: str
    text: str
    spans: tuple[tuple[int, int], ...]
    annotations: dict          # hypothesis key -> {"choice", "spans"}

    def span_texts(self) -> list[str]:
        return [self.text[a:b] for a, b in self.spans]


@dataclass
class _Data:
    docs: dict[str, _Doc]
    hypotheses: dict[str, dict]                      # key -> {"hypothesis", "short_description"}
    eval_split: dict[str, dict[str, list[str]]]      # split -> stratum -> item ids "doc/hyp"
    train_docs: dict[str, list[str]]                 # "iid"/"ood" -> doc ids (official train)


def _item(doc_id: str, key: str) -> str:
    return f"contractnli/{doc_id}/{key}"


def _split_item(item: str) -> tuple[str, str]:
    _, doc, key = item.split("/")
    return doc, key


def _load(root: Path, partition_seed: int) -> _Data:
    ddir = root / DATASET_DIR / DATA_SUBDIR
    docs: dict[str, _Doc] = {}
    hyps: dict[str, dict] = {}
    for part in ("train", "dev", "test"):
        d = read_json(ddir / f"{part}.json")
        hyps.update(d["labels"])
        for doc in d["documents"]:
            did = str(doc["id"])
            docs[did] = _Doc(did, part, str(doc["document_type"]), doc["text"],
                             tuple((int(a), int(b)) for a, b in doc["spans"]),
                             dict(doc["annotation_sets"][0]["annotations"]))

    def pairs(ds: list[_Doc]) -> dict[str, str]:
        out = {}
        for doc in ds:
            for key, a in doc.annotations.items():
                if a["choice"] in LABELS and len(a["spans"]) > 0:
                    out[_item(doc.doc_id, key)] = doc.doc_id
        return out

    iid_docs = [d for d in docs.values() if d.official_split in ("dev", "test") and d.doc_type in IID_TYPES]
    ood_docs = [d for d in docs.values() if d.official_split in ("dev", "test") and d.doc_type in OOD_TYPES]
    part = partition_groups(pairs(iid_docs), IID_FRACTIONS, f"{CODE}|{partition_seed}|iid")
    ev = {s: {"all": sorted(v)} for s, v in part.items()}
    ev["ood"] = {"all": sorted(pairs(ood_docs))}
    train = {"iid": sorted(d.doc_id for d in docs.values() if d.official_split == "train" and d.doc_type in IID_TYPES),
             "ood": sorted(d.doc_id for d in docs.values() if d.official_split == "train" and d.doc_type in OOD_TYPES)}
    return _Data(docs, hyps, ev, train)


# ----------------------------------------------------------------------------------------------- adapter
class ContractNLIAdapter:
    """TaskAdapter for FoR48 (see module docstring)."""

    discipline = CODE
    name = "ContractNLI"
    family = FAMILY
    metric = "mAP"
    direction = "max"
    task_type = "legal_evidence_identification"

    def __init__(self, data_root: str | None = None, partition_seed: int = PARTITION_SEED, n_train_docs: int = 40,
                 n_dev_docs: int = 8, max_dev_items: int = 32, margin: float = 0.10, budget: Budget | None = None,
                 **_: Any) -> None:
        self.root = resolve_data_root(data_root)
        self.partition_seed = int(partition_seed)
        self.n_train_docs, self.n_dev_docs = int(n_train_docs), int(n_dev_docs)
        self.max_dev_items = int(max_dev_items)
        self.margin = float(margin)
        self.budget = budget
        self._data: Lazy[_Data] = Lazy(lambda: _load(self.root, self.partition_seed))

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        ddir = self.root / DATASET_DIR / DATA_SUBDIR
        missing = [str(ddir / f"{p}.json") for p in ("train", "dev", "test") if not (ddir / f"{p}.json").is_file()]
        if missing:
            return False, f"missing ContractNLI files: {missing}"
        try:
            import sklearn  # noqa: F401
        except ImportError:
            return False, "scikit-learn is not installed"
        return True, f"ContractNLI (2021-10-05 release) at {ddir}"

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 1:
            raise ValueError("items_per_episode must be >= 1")
        data = self._data.get()
        rng = episode_rng(CODE, split, seed, self.partition_seed)
        draws = draw_stratified(data.eval_split[split], int(n), int(items_per_episode), rng, f"{CODE}/{split}")
        return [self._episode(data, split, k, int(seed), items) for k, items in enumerate(draws)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """mAP (mean per-pair AP) over all items of all episodes (invalid outputs scored with the reference)."""
        aps: list[float] = []
        for p in per_episode:
            if not p:
                continue
            a = p.get("ap_pred")
            if a is None:
                a = p.get("ap_ref")
            if a is None:
                raise ValueError("FoR48 pooled payload needs ap_pred or ap_ref")
            aps.extend(float(v) for v in a)
        return float(np.mean(aps)) if aps else None

    # ------------------------------------------------------------ helpers
    def _visible(self, data: _Data, pool: str, ep_key: str) -> tuple[list[str], list[str]]:
        rng = np.random.default_rng(stable_int(CODE, "visible", ep_key))
        ids = data.train_docs[pool]
        m = min(self.n_train_docs + self.n_dev_docs, len(ids))
        pick = [ids[j] for j in rng.permutation(len(ids))[:m]]
        return sorted(pick[self.n_dev_docs:]), sorted(pick[:self.n_dev_docs])

    @staticmethod
    def _majority(data: _Data, train_ids: list[str]) -> dict[str, str]:
        cnt: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for did in train_ids:
            for key, a in data.docs[did].annotations.items():
                if a["choice"] in LABELS:
                    cnt[key][a["choice"]] += 1
        return {k: ("Contradiction" if c["Contradiction"] > c["Entailment"] else "Entailment") for k, c in cnt.items()}

    # ------------------------------------------------------------ episode
    def _episode(self, data: _Data, split: str, k: int, seed: int, items: list[str]) -> Episode:
        eid = episode_id(CODE, split, seed, k)
        pool = "ood" if split == "ood" else "iid"
        pairs = [_split_item(i) for i in items]
        n_items = len(items)
        tr_ids, dev_ids = self._visible(data, pool, eid)
        majority = self._majority(data, tr_ids)
        ev_docs = sorted({d for d, _ in pairs})
        n_spans = [len(data.docs[d].spans) for d, _ in pairs]
        gold_spans = [np.isin(np.arange(len(data.docs[d].spans)), data.docs[d].annotations[h]["spans"]).astype(int)
                      for d, h in pairs]
        gold_labels = [data.docs[d].annotations[h]["choice"] for d, h in pairs]

        def doc_public(did: str) -> dict:
            doc = data.docs[did]
            return {"doc_id": did, "text": doc.text, "spans": [list(s) for s in doc.spans],
                    "span_texts": doc.span_texts()}

        hyp_table = {kk: {"hypothesis": v["hypothesis"], "short_description": v["short_description"]}
                     for kk, v in sorted(data.hypotheses.items())}
        train_annotations = [{"doc_id": did, "hypothesis_key": kk, "label": a["choice"], "evidence_spans": list(a["spans"])}
                             for did in tr_ids for kk, a in sorted(data.docs[did].annotations.items())]
        dev_all = [(did, kk) for did in dev_ids for kk, a in sorted(data.docs[did].annotations.items())
                   if a["choice"] in LABELS and a["spans"]]
        dev_rng = np.random.default_rng(stable_int(CODE, "dev-items", eid))
        dev_pairs = sorted(dev_all[j] for j in dev_rng.permutation(len(dev_all))[:self.max_dev_items])
        dev_docs = sorted({d for d, _ in dev_pairs})
        eval_rows = [{"index": j, "doc_id": d, "hypothesis_key": h, "hypothesis": data.hypotheses[h]["hypothesis"],
                      "n_spans": len(data.docs[d].spans)} for j, (d, h) in enumerate(pairs)]
        dev_rows = [{"index": j, "doc_id": d, "hypothesis_key": h, "hypothesis": data.hypotheses[h]["hypothesis"],
                     "n_spans": len(data.docs[d].spans)} for j, (d, h) in enumerate(dev_pairs)]
        n_dev = len(dev_pairs)

        def ref_pred(did: str, key: str) -> tuple[np.ndarray, str]:
            doc = data.docs[did]
            return overlap_scores(data.hypotheses[key]["hypothesis"], doc.span_texts()), majority.get(key, "Entailment")

        ref_scores = [ref_pred(d, h) for d, h in pairs]
        ref_ap = [average_precision(g, s) for g, (s, _) in zip(gold_spans, ref_scores)]
        ref_map = float(np.mean(ref_ap))

        def _coerce(p: Any, ns: int) -> tuple[np.ndarray | None, str | None, str]:
            if not isinstance(p, dict):
                return None, None, "entry must be a dict with 'label' and 'span_scores'"
            s = p.get("span_scores")
            try:
                arr = np.asarray(s.tolist() if hasattr(s, "tolist") else s, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                return None, None, "span_scores must be numeric"
            if arr.shape[0] != ns:
                return None, None, f"span_scores has {arr.shape[0]} values, the contract has {ns} spans"
            lab = p.get("label")
            return arr, (lab if isinstance(lab, str) else None), ""

        def _score_items(preds: list, golds: list[np.ndarray], labs: list[str], keys: list[str]) -> dict:
            aps, prs, correct, correct_maj = [], [], 0, 0
            per_h: dict[str, list[tuple[np.ndarray, np.ndarray]]] = collections.defaultdict(list)
            for p, g, lab, key in zip(preds, golds, labs, keys):
                arr, plab, why = _coerce(p, len(g))
                if arr is None:
                    raise ValueError(why)
                aps.append(average_precision(g, arr))
                prs.append(precision_at_recall(g, arr, 0.8))
                correct += int(plab == lab)
                correct_maj += int(majority.get(key, "Entailment") == lab)
                per_h[key].append((g, arr))
            macro_label = float(np.mean([average_precision(np.concatenate([g for g, _ in v]),
                                                           np.concatenate([a for _, a in v])) for v in per_h.values()]))
            p_pooled = precision_at_recall(np.concatenate([g for v in per_h.values() for g, _ in v]),
                                           np.concatenate([a for v in per_h.values() for _, a in v]), 0.8)
            return {"map": float(np.mean(aps)), "p_at_r80": float(np.nanmean(prs)), "aps": aps,
                    "p_at_r80_pooled": p_pooled, "map_macro_label_micro_doc": macro_label,
                    "nli_binary_accuracy": correct / max(len(aps), 1),
                    "majority_label_accuracy": correct_maj / max(len(aps), 1)}

        dev_gold = [np.isin(np.arange(len(data.docs[d].spans)), data.docs[d].annotations[h]["spans"]).astype(int)
                    for d, h in dev_pairs]
        dev_labels = [data.docs[d].annotations[h]["choice"] for d, h in dev_pairs]
        dev_keys = [h for _, h in dev_pairs]
        dev_ref = Lazy(lambda: _score_items([{"label": ref_pred(d, h)[1], "span_scores": ref_pred(d, h)[0]}
                                             for d, h in dev_pairs], dev_gold, dev_labels, dev_keys))

        # ---- D_E tools
        def load_train(inputs: dict, config: dict) -> dict:
            return {"train_documents": [doc_public(d) for d in tr_ids],
                    "train_annotations": copy.deepcopy(train_annotations)}

        def load_hypotheses(inputs: dict, config: dict) -> dict:
            return {"hypotheses": {kk: dict(v) for kk, v in hyp_table.items()}}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_items": [dict(r) for r in dev_rows], "dev_documents": [doc_public(d) for d in dev_docs]}

        def score_dev(inputs: dict, config: dict) -> dict:
            preds, why = as_list(inputs.get("dev_predictions"))
            if preds is None or len(preds) != n_dev:
                raise ValueError(f"dev_predictions must be a list of {n_dev} dicts ({why})")
            r = _score_items(preds, dev_gold, dev_labels, dev_keys)
            return {"dev_map": r["map"], "dev_p_at_r80": r["p_at_r80"], "dev_nli_binary_accuracy": r["nli_binary_accuracy"],
                    "dev_reference_map": dev_ref.get()["map"]}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"items": [dict(r) for r in eval_rows], "documents": [doc_public(d) for d in ev_docs]}

        item_doc = "dict: index, doc_id, hypothesis_key, hypothesis, n_spans"
        doc_doc = "dict: doc_id, text, spans [[start, end] char offsets], span_texts"
        tools = [
            ToolSpec("load_train", f"{len(tr_ids)} annotated contracts from the official ContractNLI train partition "
                     "(same document source as the evaluation contracts): train_documents (doc_id, text, candidate "
                     "spans, span_texts) and train_annotations (one row per contract and hypothesis: label in "
                     "Entailment / Contradiction / NotMentioned and evidence_spans = indices into that contract's "
                     "spans).",
                     {}, {"train_documents": PortSchema("list", (len(tr_ids),), dtype="dict", description=doc_doc),
                          "train_annotations": PortSchema("list", (len(train_annotations),), dtype="dict")},
                     load_train),
            ToolSpec("load_hypotheses", "The 17 fixed ContractNLI hypotheses: key -> {hypothesis, short_description}.",
                     {}, {"hypotheses": PortSchema("dict", description="hypothesis key -> {hypothesis, short_description}")}, load_hypotheses),
            ToolSpec("load_dev_inputs", f"{n_dev} (contract, hypothesis) pairs from {len(dev_docs)} further train "
                     "contracts whose annotations are withheld (each pair has gold label Entailment or "
                     "Contradiction); score predictions for them with score_dev.",
                     {}, {"dev_items": PortSchema("list", (n_dev,), dtype="dict", description=item_doc),
                          "dev_documents": PortSchema("list", (len(dev_docs),), dtype="dict", description=doc_doc)},
                     load_dev_inputs),
            ToolSpec("score_dev", "Evidence mAP, precision at 80% recall and binary NLI accuracy of dev_predictions "
                     "(same format as y, one per dev item) against the withheld dev annotations, plus the mAP of the "
                     "adapter's reference predictor.",
                     {"dev_predictions": PortSchema("list", (n_dev,), dtype="dict")},
                     {"dev_map": PortSchema("number", unit="1"), "dev_p_at_r80": PortSchema("number", unit="1"),
                      "dev_nli_binary_accuracy": PortSchema("number", unit="1"),
                      "dev_reference_map": PortSchema("number", unit="1")},
                     score_dev),
            ToolSpec("load_eval_inputs", f"The {n_items} evaluation (contract, hypothesis) pairs (annotations hidden) "
                     f"in the order that y must follow, and the {len(ev_docs)} contracts they refer to.",
                     {}, {"items": PortSchema("list", (n_items,), dtype="dict", description=item_doc),
                          "documents": PortSchema("list", (len(ev_docs),), dtype="dict", description=doc_doc)},
                     load_eval_inputs),
        ]

        # ---- D_V constraints
        def _entries(yv: Any) -> tuple[list | None, str]:
            lst, why = as_list(yv)
            if lst is None:
                return None, why
            if len(lst) != n_items:
                return None, f"len(y)={len(lst)} != {n_items}"
            return lst, ""

        def c_scores(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            lst, why = _entries(yv)
            if lst is None:
                return False, why
            for j, (p, ns) in enumerate(zip(lst, n_spans)):
                arr, _, msg = _coerce(p, ns)
                if arr is None:
                    return False, f"item {j}: {msg}"
                if not np.all(np.isfinite(arr)) or arr.min(initial=0.0) < 0 or arr.max(initial=0.0) > 1:
                    return False, f"item {j}: span_scores must be finite values in [0, 1]"
            return True, "every span_scores vector is valid"

        def c_labels(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            lst, why = _entries(yv)
            if lst is None:
                return False, why
            bad = [j for j, p in enumerate(lst) if not isinstance(p, dict) or p.get("label") not in LABELS]
            return not bad, ("all labels valid" if not bad else f"label not in {list(LABELS)} at items {bad[:10]}")

        constraints = [
            c_list_length(n_items, "one prediction per (contract, hypothesis) pair"),
            ConstraintSpec("span_scores_valid", "every y[i]['span_scores'] has exactly one finite score in [0, 1] "
                           "per candidate span of its contract (in span order)", c_scores),
            ConstraintSpec("labels_valid", "every y[i]['label'] is 'Entailment' or 'Contradiction'", c_labels),
        ]

        # ---- D_V evaluator
        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            payload = {"item_ids": list(items), "ap_pred": None, "ap_ref": list(ref_ap)}
            base = {"reference": ref_map, "reference_name": "hypothesis/span lexical overlap + per-hypothesis majority "
                    "label", "margin": self.margin, "n_items": n_items, "n_documents": len(ev_docs)}
            lst, why = _entries(yv)
            r: dict | None = None
            if lst is not None:
                try:
                    r = _score_items(lst, gold_spans, gold_labels, [h for _, h in pairs])
                except ValueError as ex:
                    why = str(ex)
            if r is None:
                return EvalResult(metrics={"reference_map": ref_map}, primary=None, direction="max", accepted=False,
                                  details={**base, "norm_score": 0.0, "pooled_payload": payload, "invalid": why})
            payload["ap_pred"] = r["aps"]
            metrics = {"map": r["map"], "p_at_r80": r["p_at_r80"], "map_macro_label_micro_doc": r["map_macro_label_micro_doc"],
                       "nli_binary_accuracy": r["nli_binary_accuracy"], "p_at_r80_pooled": r["p_at_r80_pooled"],
                       "majority_label_accuracy": r["majority_label_accuracy"], "reference_map": ref_map}
            return EvalResult(metrics=metrics, primary=r["map"], direction="max",
                              accepted=bool(r["map"] >= ref_map + self.margin),
                              details={**base, "norm_score": norm_score(r["map"], ref_map, "max"),
                                       "pooled_payload": payload})

        source = "SEC EDGAR filings" if pool == "ood" else "contracts found by web search (PDF)"
        objective = (
            "Law and legal studies - document-level natural language inference and evidence identification on "
            "non-disclosure agreements (ContractNLI). Each evaluation item is a pair (contract, hypothesis); the "
            f"contracts of this episode are {source}. Every contract comes with its candidate evidence spans "
            "(sentences or list items, given as character offsets and texts). For every evaluation pair the "
            "contract either entails or contradicts the hypothesis (pairs where the hypothesis is not mentioned are "
            "not evaluated). Hidden targets: the set of spans that are evidence for the decision, and the label.\n"
            f"Visible data: load_train returns {len(tr_ids)} annotated contracts from the official train partition; "
            "load_hypotheses lists the 17 hypotheses; load_dev_inputs returns further pairs whose annotations are "
            "withheld and score_dev scores predictions for them; load_eval_inputs returns the evaluation pairs and "
            "their contracts.\n"
            f"Deliverable y: a list of {n_items} dicts; y[i] = {{'label': 'Entailment' | 'Contradiction', "
            "'span_scores': [...]} for items[i] (same order), where span_scores has one score in [0, 1] per "
            "candidate span of the item's contract (span order), higher meaning more likely evidence.\n"
            "Evaluation: mean over items of the average precision of span_scores against the gold evidence spans "
            "(mAP); precision at 80% recall and label accuracy are reported as auxiliary metrics.\n"
            + scilib.describe("contracts") + scilib.describe_extra("textenc")
        )
        lineage = {
            "dataset": "ContractNLI", "version": "initial release 2021-10-05",
            "source_url": "https://stanfordnlp.github.io/contract-nli/", "license": "CC BY 4.0",
            "receipt": receipt_info(self.root / DATASET_DIR / "receipt.json"),
            "pool": pool, "document_types": sorted({data.docs[d].doc_type for d in ev_docs}),
            "pool_source": "official dev + test partitions, gold Entailment/Contradiction pairs with evidence",
            "ood_kind": "proxy_within_dataset" if pool == "ood" else None,
            "ood_shift": "document source: SEC EDGAR filings (sec-text/sec-html) instead of web-search PDFs"
            if pool == "ood" else None,
            "split": split, "split_seed": seed, "partition_seed": self.partition_seed, "index": k,
            "item_ids": list(items), "n_items": n_items, "documents": ev_docs,
            "train_source": "official train partition, same document source", "n_train_docs": len(tr_ids),
            "n_dev_docs": len(dev_ids), "train_ids_sha256": ids_digest(tr_ids), "dev_doc_ids": list(dev_ids),
            "metric_aggregation": "micro_label_macro_doc.span.map (mean AP over document-hypothesis pairs)",
            "rebuilt_split": True, "historical_ids_recovered": False,
        }
        return Episode(
            id=eid, discipline=CODE, family=FAMILY, split=split, task_type=self.task_type, objective=objective,
            required_output=PortSchema("list", (n_items,), dtype="dict",
                                       description="{'label': str, 'span_scores': [float]} per evaluation pair"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=2400.0, max_node_s=600.0,
                                         max_llm_items=8 * max(n_items, n_dev)),
            lineage=lineage,
            acceptance=(f"mAP >= reference mAP + {self.margin:g}; reference = lexical-overlap span scores with the "
                        "per-hypothesis majority label of the visible training contracts"),
            tolerance={"rtol": 1e-6, "atol": 1e-8},
            tags=[CODE, "law", "contracts", "NDA", "natural-language-inference", "evidence-retrieval", "mAP",
                  "ContractNLI", "english", pool],
            metric=self.metric, direction=self.direction, n_items=n_items,
            _evaluate=evaluate, _dev_evaluate=None,
        )


Adapter = ContractNLIAdapter

__all__ = ["Adapter", "ContractNLIAdapter", "average_precision", "precision_at_recall", "overlap_scores",
           "PoolExhausted"]
