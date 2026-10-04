"""FoR50 Philosophy and religious studies — SemEval-2023 Task 4 ValueEval (human values behind arguments), official F1 (max).

Data: Touché23-ValueEval, Zenodo record 10564870 (version 2024-01-24, DOI 10.5281/zenodo.10564870, CC BY 4.0),
downloaded by the data team under ``<DATA_ROOT>/for50-valueeval-2023/raw``.

* **Item** = one argument (Conclusion, Stance ``in favor of`` / ``against``, Premise); hidden label = the binary
  vector of the 20 level-2 value categories the argument resorts to (column order of the official label files).
  Item id = ``valueeval/<Argument ID>``.
* **Pools.** IID = the main dataset (IBM-ArgQ / Group Discussion Ideas / Conference on the Future of Europe
  arguments; the official splits keep arguments with the same conclusion together): src / val from the official
  *validation* split (partitioned once by conclusion, 75 / 25 %), id from the official *test* split. OOD = the
  official supplementary test set **Nahj al-Balagha** (279 arguments from and based on Islamic religious texts,
  translated from Farsi) — a different source/dataset of the same task (``lineage["ood_kind"] =
  "cross_dataset"``). The New York Times supplementary test set is *not* used: its argument texts are not in the
  public deposit (labels only).
* **Visible data (D_E).** A per-episode sample of ``n_train`` labelled arguments of the official *training* split
  (``load_train``: arguments table + 0/1 label matrix), a disjoint ``n_dev`` training slice whose labels are held
  by ``score_dev`` (the episode's ``_dev_evaluate`` is None), and the official value taxonomy
  (``load_value_taxonomy``: the 20 categories with their level-1 values and example effects). OOD episodes also
  train on the main training split, as in the shared task.
* **Metric (D_V).** Official F1 of the ValueEval'23 evaluator (touche-code ``semeval23/human-value-detection/
  evaluator/evaluator.py``, version 2023-08-13): over the value categories with at least one gold-positive
  argument, precision_c = TP/(TP+FP) (0 without predicted positives) and recall_c = TP/(TP+FN) are averaged
  (macro precision P, macro recall R) and the score is the harmonic mean ``F1 = 2PR/(P+R)`` (0 when P = R = 0 —
  the data team's documented zero guard; the upstream script divides by zero there). Note this is not the mean
  of per-category F1 values, which is reported as auxiliary (``mean_category_f1``).
* **Reference baseline** (deterministic): predict all 20 categories for every argument (the shared task's
  "1-baseline"). **Acceptance:** ``F1 >= reference F1 + margin`` (default 0.05).
* **Hard constraints:** y has shape (n_items, 20); every entry is 0 or 1.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib

from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for43_45_47_48_50 import (
    PARTITION_SEED, Lazy, PoolExhausted, check_split, draw_stratified, episode_id, episode_rng, ids_digest,
    norm_score, partition_groups, read_json, receipt_info, resolve_data_root, stable_int,
)

CODE = "FoR50"
FAMILY = "Humanities & law"
DATASET_DIR = "for50-valueeval-2023"
VALUES = ("Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement",
          "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition",
          "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring",
          "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance",
          "Universalism: objectivity")
VAL_FRACTIONS = {"src": 0.75, "val": 0.25}
_FILES = {"training": ("arguments-training.tsv", "labels-training.tsv"),
          "validation": ("arguments-validation.tsv", "labels-validation.tsv"),
          "test": ("arguments-test.tsv", "labels-test.tsv"),
          "test-nahjalbalagha": ("arguments-test-nahjalbalagha.tsv", "labels-test-nahjalbalagha.tsv")}

# The specialist route is deliberately a fixed, auditable configuration.  It is a
# visible-data baseline/tool, not a second scorer or a tunable policy parameter.
# Keeping these values here (rather than inheriting library defaults implicitly)
# makes a full-split run reproducible across hosts and reports.
VALUEEVAL_TOOL_SEED = 0
VALUEEVAL_TOOL_C = 0.3
VALUEEVAL_TOOL_FOLDS = 5
VALUEEVAL_TOOL_K = 16
VALUEEVAL_TOOL_DECISION = "expected_f1"
VALUEEVAL_TOOL_N_JOBS = 2
VALUEEVAL_TOOL_CONFIG = {
    "seed": VALUEEVAL_TOOL_SEED,
    "C": VALUEEVAL_TOOL_C,
    "n_folds": VALUEEVAL_TOOL_FOLDS,
    "k": VALUEEVAL_TOOL_K,
    "decision": VALUEEVAL_TOOL_DECISION,
    "n_jobs": VALUEEVAL_TOOL_N_JOBS,
    "groups": "conclusion",
}


# ----------------------------------------------------------------------------------------------- official metric
def official_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """ValueEval'23 official evaluator: F1 = harmonic mean of macro precision and macro recall.

    Categories without gold positives are skipped (as upstream); a category with no predicted positives has
    precision 0. Returns F1, macro precision, macro recall, the mean of per-category F1 and #scored categories.
    """
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    precs, recs, f1s = [], [], []
    for c in range(t.shape[1]):
        relevant = int(t[:, c].sum())
        if relevant == 0:
            continue
        positives = int(p[:, c].sum())
        tp = int(((p[:, c] == 1) & (t[:, c] == 1)).sum())
        prec = tp / positives if positives else 0.0
        rec = tp / relevant
        precs.append(prec)
        recs.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    if not precs:
        return {"f1": float("nan"), "precision": float("nan"), "recall": float("nan"), "mean_category_f1": float("nan"),
                "n_categories": 0}
    P, R = float(np.mean(precs)), float(np.mean(recs))
    return {"f1": (2 * P * R / (P + R)) if P + R else 0.0, "precision": P, "recall": R,
            "mean_category_f1": float(np.mean(f1s)), "n_categories": len(precs)}


# ----------------------------------------------------------------------------------------------- data
@dataclass(frozen=True)
class _Arg:
    uid: str
    arg_id: str
    source: str             # official file split name
    conclusion: str
    stance: str
    premise: str
    labels: tuple[int, ...]


def _read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _load(root: Path) -> dict[str, _Arg]:
    raw = root / DATASET_DIR / "raw"
    args: dict[str, _Arg] = {}
    for src, (af, lf) in _FILES.items():
        labels = {r["Argument ID"]: tuple(int(r[v]) for v in VALUES) for r in _read_tsv(raw / lf)}
        for r in _read_tsv(raw / af):
            aid = r["Argument ID"]
            if aid not in labels:
                continue
            uid = f"valueeval/{aid}"
            if uid in args:
                raise ValueError(f"duplicate argument id {aid}")
            args[uid] = _Arg(uid, aid, src, r["Conclusion"], r["Stance"], r["Premise"], labels[aid])
    return args


@dataclass
class _Data:
    args: dict[str, _Arg]
    eval_split: dict[str, dict[str, list[str]]]
    train_ids: list[str]
    taxonomy: dict


def _build(root: Path, partition_seed: int) -> _Data:
    args = _load(root)
    val = {u: a.conclusion for u, a in args.items() if a.source == "validation"}
    part = partition_groups(val, VAL_FRACTIONS, f"{CODE}|{partition_seed}|validation")
    ev = {"src": {"all": sorted(part["src"])}, "val": {"all": sorted(part["val"])},
          "id": {"all": sorted(u for u, a in args.items() if a.source == "test")},
          "ood": {"all": sorted(u for u, a in args.items() if a.source == "test-nahjalbalagha")}}
    eval_texts = {(args[u].conclusion, args[u].premise) for sp in ev.values() for u in sp["all"]}
    train = sorted(u for u, a in args.items() if a.source == "training" and (a.conclusion, a.premise) not in eval_texts)
    taxonomy = read_json(root / DATASET_DIR / "raw" / "value-categories.json")
    return _Data(args, ev, train, taxonomy)


def _as_matrix(y: Any, n: int) -> tuple[np.ndarray | None, str]:
    if y is None:
        return None, "no output"
    if isinstance(y, (str, bytes, dict)):
        return None, f"y must be an (n, 20) 0/1 matrix, got {type(y).__name__}"
    try:
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as ex:
        return None, f"y is not numeric: {ex}"
    if arr.shape != (n, len(VALUES)):
        return None, f"y has shape {list(arr.shape)}, required [{n}, {len(VALUES)}]"
    return arr, ""


def _fixed_fit_predict(train_arguments: Any, train_labels: Any, eval_arguments: Any,
                       n_values: int = len(VALUES)) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the fixed visible-data ValueEval route used by the task tool and full-pool audit.

    The evaluator never calls this helper: it is a prediction route over the tables and labels supplied by
    ``load_train`` plus the label-free table supplied by ``load_eval_inputs``.  In particular, ``eval_arguments`` is
    checked as a text table and is never allowed to carry a target/label column.  The fixed configuration is written
    out in the returned provenance so that a full-split report cannot silently inherit changed library defaults.
    """
    import pandas as pd

    if not isinstance(train_arguments, pd.DataFrame) or not isinstance(eval_arguments, pd.DataFrame):
        raise TypeError("train_arguments and eval_arguments must be pandas DataFrames")
    required_train = {"conclusion", "stance", "premise"}
    train_columns = set(train_arguments.columns)
    missing_train = sorted(required_train - train_columns)
    if missing_train:
        raise ValueError(f"train_arguments missing required columns: {missing_train}")
    unexpected_train = sorted(train_columns - (required_train | {"argument_id"}))
    if unexpected_train:
        raise ValueError(f"train_arguments contains forbidden target columns: {unexpected_train}")
    required_eval = {"conclusion", "stance", "premise"}
    eval_columns = set(eval_arguments.columns)
    missing_eval = sorted(required_eval - eval_columns)
    if missing_eval:
        raise ValueError(f"eval_arguments missing required columns: {missing_eval}")
    # ``argument_id`` is allowed only on the visible training table.  An evaluation id, labels, or target aliases
    # would violate the task's opaque-row boundary, so reject every column outside the three text fields.
    forbidden_eval = sorted(eval_columns - required_eval)
    if forbidden_eval:
        raise ValueError(f"eval_arguments contains forbidden target/id columns: {forbidden_eval}")

    raw_labels = np.asarray(train_labels)
    if raw_labels.shape != (len(train_arguments), int(n_values)):
        raise ValueError(f"train_labels has shape {list(raw_labels.shape)}, required [{len(train_arguments)}, {n_values}]")
    if raw_labels.dtype.kind not in "biuf" or not np.all(np.isfinite(raw_labels)):
        raise ValueError("train_labels must be finite numeric values")
    if not np.all(np.isin(raw_labels, (0, 1))):
        raise ValueError("train_labels must contain only 0 and 1")
    labels = raw_labels.astype(int, copy=False)
    # Explicit groups make the no-conclusion-leakage rule independent of scilib's default implementation.
    groups = train_arguments["conclusion"].astype(str).to_numpy()
    from scilib import valueeval
    result = valueeval.fit_predict(
        train_arguments,
        labels,
        [eval_arguments],
        n_folds=VALUEEVAL_TOOL_FOLDS,
        C=VALUEEVAL_TOOL_C,
        k=VALUEEVAL_TOOL_K,
        seed=VALUEEVAL_TOOL_SEED,
        n_jobs=VALUEEVAL_TOOL_N_JOBS,
        groups=groups,
        decision=VALUEEVAL_TOOL_DECISION,
    )
    try:
        raw_pred = np.asarray(result[0])
    except (IndexError, TypeError, ValueError) as ex:
        raise ValueError(f"scilib.valueeval.fit_predict returned no usable target matrix: {ex}") from ex
    if raw_pred.shape != (len(eval_arguments), int(n_values)):
        raise ValueError(f"valueeval predictions have shape {list(raw_pred.shape)}, required [{len(eval_arguments)}, {n_values}]")
    if raw_pred.dtype.kind not in "biuf" or not np.all(np.isfinite(raw_pred)):
        raise ValueError("valueeval predictions must be finite numeric values")
    if not np.all(np.isin(raw_pred, (0, 1))):
        raise ValueError("valueeval predictions must contain only 0 and 1")
    pred = raw_pred.astype(int, copy=False)
    info = dict(VALUEEVAL_TOOL_CONFIG)
    info.update({
        "method": "scilib.valueeval.fit_predict",
        "n_train": int(len(train_arguments)),
        "n_eval": int(len(eval_arguments)),
        "n_values": int(n_values),
        "oof_f1": float(getattr(result, "oof_f1", float("nan"))),
        "label_source": "visible load_train.labels only",
        "target_source": "label-free eval_arguments only",
    })
    return pred, info


# ----------------------------------------------------------------------------------------------- adapter
class ValueEvalAdapter:
    """TaskAdapter for FoR50 (see module docstring)."""

    discipline = CODE
    name = "SemEval-2023-Task4-ValueEval"
    family = FAMILY
    metric = "official F1"
    direction = "max"
    task_type = "multi_label_value_classification"

    def __init__(self, data_root: str | None = None, partition_seed: int = PARTITION_SEED, n_train: int = 2000,
                 n_dev: int = 64, margin: float = 0.05, budget: Budget | None = None, **_: Any) -> None:
        self.root = resolve_data_root(data_root)
        self.partition_seed = int(partition_seed)
        self.n_train, self.n_dev = int(n_train), int(n_dev)
        self.margin = float(margin)
        self.budget = budget
        self._data: Lazy[_Data] = Lazy(lambda: _build(self.root, self.partition_seed))

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        raw = self.root / DATASET_DIR / "raw"
        need = [raw / f for pair in _FILES.values() for f in pair] + [raw / "value-categories.json"]
        missing = [str(p) for p in need if not p.is_file()]
        if missing:
            return False, f"missing ValueEval files: {missing}"
        return True, f"Touché23-ValueEval 2024-01-24 at {raw}"

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 1:
            raise ValueError("items_per_episode must be >= 1")
        data = self._data.get()
        rng = episode_rng(CODE, split, seed, self.partition_seed)
        draws = draw_stratified(data.eval_split[split], int(n), int(items_per_episode), rng, f"{CODE}/{split}")
        return [self._episode(data, split, k, int(seed), items) for k, items in enumerate(draws)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Official F1 over all items of all episodes (invalid outputs scored with the all-ones reference)."""
        t_rows: list[list[int]] = []
        p_rows: list[list[int]] = []
        for p in per_episode:
            if not p:
                continue
            t = p.get("y_true")
            y = p.get("y_pred")
            if y is None:
                y = p.get("y_ref")
            if t is None or y is None or len(t) != len(y):
                raise ValueError("FoR50 pooled payload needs y_true and y_pred (or y_ref) of equal length")
            t_rows.extend(t)
            p_rows.extend(y)
        if not t_rows:
            return None
        v = official_f1(np.asarray(t_rows), np.asarray(p_rows))["f1"]
        return None if not np.isfinite(v) else float(v)

    def pooled_diagnostics(self, per_episode: list[dict]) -> dict[str, Any]:
        """Return pooled and slice-reference diagnostics without changing the primary metric.

        The benchmark's 16-item episodes can omit categories with no positive labels.  This trusted-side helper
        exposes the number of pooled items/categories and compares the pooled score with the mean of per-episode
        scores.  It is intended for reports and full-pool audits; it is never shown to the policy and does not alter
        ``pooled_metric`` or acceptance.
        """
        scores: list[float] = []
        refs: list[float] = []
        t_rows: list[list[int]] = []
        p_rows: list[list[int]] = []
        n_payloads = 0
        for p in per_episode:
            if not p:
                continue
            n_payloads += 1
            t = p.get("y_true")
            y = p.get("y_pred") if p.get("y_pred") is not None else p.get("y_ref")
            if t is None or y is None or len(t) != len(y):
                raise ValueError("FoR50 diagnostics need y_true and y_pred (or y_ref) of equal length")
            tt, yy = np.asarray(t, dtype=int), np.asarray(y, dtype=int)
            if tt.ndim != 2 or tt.shape[1] != len(VALUES) or yy.shape != tt.shape:
                raise ValueError(f"FoR50 diagnostics need matrices with shape (n, {len(VALUES)})")
            val = official_f1(tt, yy)["f1"]
            ref = official_f1(tt, np.ones_like(tt))["f1"]
            if np.isfinite(val):
                scores.append(float(val))
            if np.isfinite(ref):
                refs.append(float(ref))
            t_rows.extend(tt.tolist()); p_rows.extend(yy.tolist())
        if not t_rows:
            return {"n_episodes": n_payloads, "n_valid_episode_f1": 0, "n_items": 0, "n_scored_categories": 0,
                    "pooled_f1": None, "mean_episode_f1": None, "mean_slice_reference_f1": None,
                    "slice_to_pool_delta": None}
        pooled = official_f1(np.asarray(t_rows), np.asarray(p_rows))
        return {"n_episodes": n_payloads, "n_valid_episode_f1": len(scores), "n_items": len(t_rows),
                "n_scored_categories": int(pooled["n_categories"]), "pooled_f1": float(pooled["f1"]),
                "mean_episode_f1": float(np.mean(scores)) if scores else None,
                "mean_slice_reference_f1": float(np.mean(refs)) if refs else None,
                "slice_to_pool_delta": (float(np.mean(scores) - pooled["f1"]) if scores else None),
                "diagnostic_only": True}

    def full_split_reference(self, split: str) -> dict[str, Any]:
        """Compute the all-values reference on a complete evaluation split for comparability reports."""
        check_split(split)
        data = self._data.get()
        ids = list(data.eval_split[split]["all"])
        y = np.asarray([data.args[u].labels for u in ids], dtype=int)
        out = official_f1(y, np.ones_like(y))
        return {"split": split, "n_items": len(ids), "n_categories": int(out["n_categories"]),
                "reference_f1": float(out["f1"]), "diagnostic_only": True}

    def full_split_diagnostics(self, split: str) -> dict[str, Any]:
        """Run the fixed visible-data route on every item of ``split`` for a trusted-side comparability audit.

        This is intentionally separate from :meth:`build_episodes` and from ``evaluate``.  It uses the complete
        official training table and labels to fit the fixed TF-IDF/OOF route, then scores the resulting predictions
        against the complete split inside the trusted adapter.  Only aggregate metrics and configuration metadata are
        returned; matrices, item ids and hidden targets never leave this method.  The agent-facing ToolSpec calls the
        same ``_fixed_fit_predict`` helper but receives only visible train/evaluation tables.
        """
        check_split(split)
        data = self._data.get()
        ids = list(data.eval_split[split]["all"])
        train_ids = list(data.train_ids)
        nv = len(VALUES)

        def table(rows: list[_Arg], with_id: bool) -> Any:
            import pandas as pd
            d = {"conclusion": [a.conclusion for a in rows], "stance": [a.stance for a in rows],
                 "premise": [a.premise for a in rows]}
            if with_id:
                d = {"argument_id": [a.arg_id for a in rows], **d}
            return pd.DataFrame(d)

        tr = [data.args[u] for u in train_ids]
        ev = [data.args[u] for u in ids]
        train_table = table(tr, True)
        train_labels = np.asarray([a.labels for a in tr], dtype=int)
        eval_table = table(ev, False)
        pred, info = _fixed_fit_predict(train_table, train_labels, eval_table, nv)
        truth = np.asarray([a.labels for a in ev], dtype=int)
        score = official_f1(truth, pred)
        ref = official_f1(truth, np.ones_like(truth))
        out = {
            "split": split,
            "partition_seed": int(self.partition_seed),
            "n_train": len(tr),
            "n_items": len(ev),
            "train_ids_sha256": ids_digest(train_ids),
            "split_ids_sha256": ids_digest(ids),
            "receipt": receipt_info(self.root / DATASET_DIR / "receipt.json"),
            "n_categories": int(score["n_categories"]),
            "f1": float(score["f1"]),
            "precision": float(score["precision"]),
            "recall": float(score["recall"]),
            "mean_category_f1": float(score["mean_category_f1"]),
            "reference_f1": float(ref["f1"]),
            "reference_delta": float(score["f1"] - ref["f1"]),
            "diagnostic_only": True,
            "tool": info,
        }
        return out

    # ------------------------------------------------------------ episode
    def _visible(self, data: _Data, ep_key: str) -> tuple[list[str], list[str]]:
        rng = np.random.default_rng(stable_int(CODE, "visible", ep_key))
        ids = data.train_ids
        # The official splits keep arguments of one conclusion together, so the evaluation conclusions never occur in
        # training. The dev slice is therefore held out by conclusion too (a random slice shares its conclusion with
        # ~95 % of the visible training arguments and inflated dev F1 by ~0.17 over the real evaluation).
        by_conc: dict[str, list[str]] = {}
        for u in ids:
            by_conc.setdefault(data.args[u].conclusion, []).append(u)
        conc = sorted(by_conc)
        per_conc = 8
        dev: list[str] = []
        dev_conc: set[str] = set()
        for j in rng.permutation(len(conc)):
            if len(dev) >= self.n_dev:
                break
            members = by_conc[conc[j]]
            dev_conc.add(conc[j])
            dev += [members[i] for i in rng.permutation(len(members))[:min(per_conc, self.n_dev - len(dev))]]
        pool = [u for u in ids if data.args[u].conclusion not in dev_conc]
        tr = [pool[j] for j in rng.permutation(len(pool))[:min(self.n_train, len(pool))]]
        return sorted(tr), sorted(dev)

    def _episode(self, data: _Data, split: str, k: int, seed: int, items: list[str]) -> Episode:
        eid = episode_id(CODE, split, seed, k)
        pool = "ood" if split == "ood" else "iid"
        ev = [data.args[u] for u in items]
        n_items = len(ev)
        tr_ids, dev_ids = self._visible(data, eid)
        tr = [data.args[u] for u in tr_ids]
        dv = [data.args[u] for u in dev_ids]
        n_tr, n_dev = len(tr), len(dv)
        y_true = np.asarray([a.labels for a in ev], dtype=int)
        y_ref = np.ones_like(y_true)
        ref = official_f1(y_true, y_ref)
        dev_true = np.asarray([a.labels for a in dv], dtype=int)
        dev_ref = official_f1(dev_true, np.ones_like(dev_true))["f1"]

        def table(rows: list[_Arg], with_id: bool) -> Any:
            import pandas as pd
            d = {"conclusion": [a.conclusion for a in rows], "stance": [a.stance for a in rows],
                 "premise": [a.premise for a in rows]}
            if with_id:
                d = {"argument_id": [a.arg_id for a in rows], **d}
            return pd.DataFrame(d)

        train_tab = table(tr, True)
        train_lab = np.asarray([a.labels for a in tr], dtype=np.int64)
        dev_tab = table(dv, False)
        eval_tab = table(ev, False)

        # ---- D_E tools
        def load_train(inputs: dict, config: dict) -> dict:
            return {"arguments": train_tab.copy(), "labels": train_lab.copy(), "value_names": list(VALUES)}

        def load_value_taxonomy(inputs: dict, config: dict) -> dict:
            return {"taxonomy": {k2: {k3: list(v3) for k3, v3 in v2.items()} for k2, v2 in data.taxonomy.items()},
                    "value_names": list(VALUES)}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_arguments": dev_tab.copy()}

        def score_dev(inputs: dict, config: dict) -> dict:
            arr, why = _as_matrix(inputs.get("dev_predictions"), n_dev)
            if arr is None:
                raise ValueError(f"dev_predictions: {why}")
            if not np.all(np.isin(arr, (0.0, 1.0))):
                raise ValueError("dev_predictions must contain only 0 and 1")
            r = official_f1(dev_true, arr.astype(int))
            return {"dev_f1": r["f1"], "dev_precision": r["precision"], "dev_recall": r["recall"],
                    "dev_reference_f1": dev_ref}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"arguments": eval_tab.copy(), "value_names": list(VALUES)}

        def fit_predict_tool(inputs: dict, config: dict) -> dict:
            """Fixed visible-data route; ``config`` is intentionally ignored so parameters cannot drift per episode."""
            pred, info = _fixed_fit_predict(inputs.get("train_arguments"), inputs.get("train_labels"),
                                            inputs.get("eval_arguments"), nv)
            return {"y": pred, "provenance": info}

        arg_doc = "columns conclusion, stance ('in favor of' | 'against'), premise"
        nv = len(VALUES)
        tools = [
            ToolSpec("load_train", f"{n_tr} labelled arguments of the official ValueEval training split: arguments "
                     f"table (argument_id, conclusion, stance, premise) and labels[i, c] = 1 if argument i resorts to value "
                     "category value_names[c].",
                     {}, {"arguments": PortSchema("table", (n_tr, 4), description="argument_id, conclusion, stance, premise"),
                          "labels": PortSchema("array", (n_tr, nv), unit="1", dtype="int", description="0/1 labels"),
                          "value_names": PortSchema("list", (nv,), dtype="str")},
                     load_train),
            ToolSpec("load_value_taxonomy", "The official value taxonomy: for each of the 20 value categories its "
                     "level-1 values and exemplary effects an argument may target.",
                     {}, {"taxonomy": PortSchema("dict"), "value_names": PortSchema("list", (nv,), dtype="str")},
                     load_value_taxonomy),
            ToolSpec("load_dev_inputs", f"{n_dev} further training-split arguments whose labels are withheld; score "
                     "predictions for them with score_dev.",
                     {}, {"dev_arguments": PortSchema("table", (n_dev, 3), description=arg_doc)}, load_dev_inputs),
            ToolSpec("score_dev", "Official ValueEval F1 (and macro precision/recall) of dev_predictions (a 0/1 matrix "
                     f"of shape ({n_dev}, {nv}), rows = dev arguments, columns = value_names) against the withheld dev "
                     "labels, plus the reference predictor's F1.",
                     {"dev_predictions": PortSchema("array", (n_dev, nv), unit="1", dtype="int")},
                     {"dev_f1": PortSchema("number", unit="1"), "dev_precision": PortSchema("number", unit="1"),
                      "dev_recall": PortSchema("number", unit="1"), "dev_reference_f1": PortSchema("number", unit="1")},
                     score_dev),
            ToolSpec("load_eval_inputs", f"The {n_items} evaluation arguments (labels hidden) in the order that the "
                     "rows of y must follow, and the column order value_names.",
                     {}, {"arguments": PortSchema("table", (n_items, 3), description=arg_doc),
                          "value_names": PortSchema("list", (nv,), dtype="str")},
                     load_eval_inputs),
            ToolSpec("fit_predict", "Fixed visible-data ValueEval predictor: fit scilib.valueeval on the labelled "
                     "training table and labels, then return a 0/1 matrix for the label-free evaluation table. It "
                     "never reads evaluation labels; the returned y is ready for submit.",
                     {"train_arguments": PortSchema("table", (n_tr, 4), description="visible argument_id + conclusion, stance, premise"),
                      "train_labels": PortSchema("array", (n_tr, nv), unit="1", dtype="int", description="visible 0/1 labels"),
                      "eval_arguments": PortSchema("table", (n_items, 3), description=arg_doc)},
                     {"y": PortSchema("array", (n_items, nv), unit="1", dtype="int",
                                      description="0/1 predictions, rows in eval_arguments order"),
                      "provenance": PortSchema("dict", description="fixed route parameters and visible-data provenance")},
                     fit_predict_tool,
                     config_doc=(f"seed={VALUEEVAL_TOOL_SEED}; C={VALUEEVAL_TOOL_C:g}; "
                                 f"n_folds={VALUEEVAL_TOOL_FOLDS}; k={VALUEEVAL_TOOL_K}; "
                                 f"decision={VALUEEVAL_TOOL_DECISION}; n_jobs={VALUEEVAL_TOOL_N_JOBS}; groups=conclusion")),
        ]

        # ---- D_V constraints
        def c_shape(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            arr, why = _as_matrix(yv, n_items)
            return arr is not None, (why or f"shape ({n_items}, {nv})")

        def c_binary(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            arr, why = _as_matrix(yv, n_items)
            if arr is None:
                return False, why
            bad = int((~np.isin(arr, (0.0, 1.0))).sum())
            return bad == 0, ("all entries are 0 or 1" if bad == 0 else f"{bad} entries are not 0/1")

        constraints = [
            ConstraintSpec("output_shape", f"y is a 0/1 matrix of shape ({n_items}, {nv}): one row per evaluation "
                           "argument (load_eval_inputs order), one column per value category (value_names order)",
                           c_shape),
            ConstraintSpec("binary", "every entry of y is 0 or 1", c_binary),
        ]

        # ---- D_V evaluator
        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            payload = {"item_ids": list(items), "y_true": y_true.tolist(), "y_pred": None, "y_ref": y_ref.tolist()}
            base = {"reference": ref["f1"], "reference_name": "predict all 20 value categories (1-baseline)",
                    "margin": self.margin, "n_items": n_items, "n_scored_categories": ref["n_categories"]}
            arr, why = _as_matrix(yv, n_items)
            if arr is not None and not np.all(np.isin(arr, (0.0, 1.0))):
                arr, why = None, "entries other than 0/1"
            if arr is None:
                return EvalResult(metrics={"reference_f1": ref["f1"]}, primary=None, direction="max", accepted=False,
                                  details={**base, "norm_score": 0.0, "pooled_payload": payload, "invalid": why})
            pred = arr.astype(int)
            r = official_f1(y_true, pred)
            payload["y_pred"] = pred.tolist()
            metrics = {"f1": r["f1"], "precision": r["precision"], "recall": r["recall"],
                       "mean_category_f1": r["mean_category_f1"], "reference_f1": ref["f1"]}
            return EvalResult(metrics=metrics, primary=r["f1"], direction="max",
                              accepted=bool(r["f1"] >= ref["f1"] + self.margin),
                              details={**base, "norm_score": norm_score(r["f1"], ref["f1"], "max"),
                                       "pooled_payload": payload})

        source = ("the Nahj al-Balagha supplementary test set (arguments from and based on Islamic religious texts, "
                  "translated from Farsi)" if pool == "ood" else "the main ValueEval dataset (debate portals, group "
                  "discussion ideas, the Conference on the Future of Europe)")
        objective = (
            "Philosophy and religious studies - identifying the human values behind arguments (SemEval-2023 Task 4 "
            f"ValueEval). Each evaluation item is an English argument from {source}: a conclusion, the stance of the "
            "premise towards it ('in favor of' or 'against') and the premise. The hidden target is the set of value "
            "categories (of 20 in Schwartz's refined value continuum, e.g. 'Self-direction: thought', 'Security: "
            "societal', 'Universalism: nature') that the argument draws on; an argument can draw on several.\n"
            f"Visible data: load_train returns {n_tr} labelled arguments of the official training split; "
            "load_value_taxonomy returns the official description of every category; load_dev_inputs returns "
            f"{n_dev} further training arguments whose labels are withheld and score_dev scores predictions for "
            "them; load_eval_inputs returns the evaluation arguments. As in the shared task, the conclusions of the "
            "evaluation arguments do not occur among the training arguments, and neither do those of the dev "
            "arguments.\n"
            "The explicit fit_predict tool accepts only the visible load_train argument table/labels and the "
            "label-free load_eval_inputs argument table, and returns y directly. It runs the fixed "
            f"scilib.valueeval.fit_predict route (seed={VALUEEVAL_TOOL_SEED}, C={VALUEEVAL_TOOL_C:g}, "
            f"n_folds={VALUEEVAL_TOOL_FOLDS}, k={VALUEEVAL_TOOL_K}, decision={VALUEEVAL_TOOL_DECISION}, "
            f"n_jobs={VALUEEVAL_TOOL_N_JOBS}, groups=conclusion); do not pass or infer hidden evaluation labels.\n"
            f"Deliverable y: a 0/1 matrix of shape ({n_items}, {nv}); y[i, c] = 1 if argument i (load_eval_inputs "
            "order) draws on value category value_names[c].\n"
            "Evaluation: official ValueEval F1 = harmonic mean of the macro-averaged precision and the macro-averaged "
            "recall over the categories that have at least one positive argument in the episode; a category without "
            f"a predicted positive has precision 0. The score is computed on these {n_items} arguments alone.\n"
            + scilib.describe("valueeval")
        )
        lineage = {
            "dataset": "Touché23-ValueEval (SemEval-2023 Task 4)", "version": "2024-01-24 (Zenodo 10564870)",
            "source_url": "https://doi.org/10.5281/zenodo.10564870", "license": "CC BY 4.0",
            "receipt": receipt_info(self.root / DATASET_DIR / "receipt.json"),
            "pool": pool,
            "pool_source": ("official supplementary test set test-nahjalbalagha" if pool == "ood" else
                            f"official {'test' if split == 'id' else 'validation'} split of the main dataset"),
            "ood_kind": "cross_dataset" if pool == "ood" else None,
            "ood_shift": "Nahj al-Balagha arguments (religious sources, Farsi translations) vs. debate-portal arguments"
            if pool == "ood" else None,
            "split": split, "split_seed": seed, "partition_seed": self.partition_seed, "index": k,
            "item_ids": list(items), "n_items": n_items,
            "train_source": "official training split", "n_train": n_tr, "n_dev": n_dev,
            "train_ids_sha256": ids_digest(tr_ids), "dev_item_ids": list(dev_ids),
            "rebuilt_split": True, "historical_ids_recovered": False,
        }
        return Episode(
            id=eid, discipline=CODE, family=FAMILY, split=split, task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n_items, nv), unit="1", dtype="int",
                                       description="0/1 value-category matrix, rows = evaluation arguments"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=1800.0, max_node_s=600.0,
                                         max_llm_items=4 * max(n_items, n_dev)),
            lineage=lineage,
            acceptance=f"official F1 >= reference F1 + {self.margin:g}; reference = predicting all 20 categories",
            tolerance={"rtol": 1e-6, "atol": 1e-8},
            tags=[CODE, "philosophy", "religion", "human-values", "argument-mining", "multi-label-classification",
                  "ValueEval", "SemEval-2023", "english", pool],
            metric=self.metric, direction=self.direction, n_items=n_items,
            _evaluate=evaluate, _dev_evaluate=None,
        )


Adapter = ValueEvalAdapter

__all__ = ["Adapter", "ValueEvalAdapter", "official_f1", "VALUES", "PoolExhausted",
           "VALUEEVAL_TOOL_CONFIG", "VALUEEVAL_TOOL_SEED", "VALUEEVAL_TOOL_C", "VALUEEVAL_TOOL_FOLDS",
           "VALUEEVAL_TOOL_K", "VALUEEVAL_TOOL_DECISION", "VALUEEVAL_TOOL_N_JOBS"]
