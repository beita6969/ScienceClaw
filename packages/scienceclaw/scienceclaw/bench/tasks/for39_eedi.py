"""FoR39 Education — Eedi NeurIPS 2020 Education Challenge Task 4 (personalised question selection),
metric: organizer 10-mask accuracy (max).

Data: official Task 3/4 members retrieved by the data team (CRC-verified) under
``<DATA_ROOT>/for39-eedi-task4/source_members/data``: ``train_task_3_4.csv`` (4,918 students, 1.38 M answers to 948
diagnostic maths questions), ``test_public_task_4_more_splits.csv`` and ``test_private_task_4_more_splits.csv``
(615 students each, with the organizers' ten ``IsTarget_0..9`` masks), and question / subject / student metadata.

Official protocol (starter kit ``evaluation.py``): for every mask ``m`` a model queries 10 answers per test student
(one at a time), then predicts the correctness of that student's ``IsTarget_m`` answers; accuracy is computed over
all target answers of the mask and the score is the mean of the ten mask accuracies.

This adapter:

* **Item** = one test student with **one** of the ten official masks (``mask = rank % 10`` of a salted hash rank,
  so masks are balanced). One mask per student prevents cross-mask leakage: under a different mask the same
  student's target answers are queryable, and a workflow holding several masks' reveals at once could read them.
  (In the organizer loop every mask is a fresh model, so no leakage occurs there.) Item id ``eedi-u<UserId>``.
* **Query rule** (strict, as in the data team's reconstruction): only answered, non-target questions of the item's
  mask may be queried; at most ``QUERY_BUDGET`` = 10 distinct answers per student in total. ``query_answers`` may be
  called several times (chained calls allow adaptive selection); every call returns an HMAC-signed receipt and the
  hard constraint ``query_budget`` unions the receipts of the **whole solve**: the tool nodes of
  the final trace *and* every ``receipt.pkl`` the executor persisted in the solve's session (``<run_dir>/exec``) and
  replay (``<run_dir>/replay/kNNN``) directories, so answers revealed by nodes the policy later removed or replaced, or
  by nodes inside operator bodies, still count. Fails closed when that ledger cannot be located. The starter kit's
  own ``can_query`` does not exclude target answers; the strict rule is a documented reconstruction choice.
* **Pools.** IID = official public test students (``src``/``val``/``id`` sub-pools 55/15/30 %, fixed by
  ``partition_seed``); OOD = official private-leaderboard test students. Both are student-disjoint samples of the
  same population (no demographic/time shift is claimed): ``lineage["ood_kind"] = "proxy_within_dataset"``.
* **Visible data.** ``load_train``: official training answer matrix (all training students except ``n_dev`` held-out
  dev students) + question subjects + student metadata. Dev: ``load_dev_inputs`` / ``query_dev_answers`` /
  ``score_dev`` reproduce the protocol on the held-out training students with seeded 80/20 query/target masks
  (visible dev signal; ``_dev_evaluate`` is None).
* **Metric.** For each mask present in the episode, accuracy over all target answers of the items with that mask;
  score = mean over those masks (organizer formula restricted to one mask per student).
* **Reference** = the organizer starter baseline: predict each question's most common training correctness
  (``argmax(bincount)``; the starter's query policy does not affect its predictions). **Acceptance:**
  ``score >= reference + margin`` (default 0.02).
* **Hard constraints:** y is an integer array of shape (items, 948) with values in {-1, 0, 1} and a 0/1 prediction
  on every target cell; ``query_budget`` (above).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scilib

from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for34_39_40_44_51 import (
    PARTITION_SEED, Lazy, as_float_array, cache_dir, check_split, draw_episodes, episode_id, episode_rng, hash_rank,
    norm_score, partition_pool, receipt_info, resolve_data_root,
)

CODE = "FoR39"
FAMILY = "Social & behavior"
DATASET_DIR = "for39-eedi-task4"
N_Q = 948
N_MASKS = 10
QUERY_BUDGET = 10
IID_FRACTIONS = {"src": 0.55, "val": 0.15, "id": 0.30}
RECEIPT_KIND = "eedi-task4-query-receipt"
# solver layout (agent/solver.py): <run_dir>/exec = interactive session, <run_dir>/replay/kNNN = reset replay (Trace.run_dir)
SESSION_DIR = "exec"
REPLAY_DIR = "replay"
RECEIPT_FILE = "receipt.pkl"
_CACHE_VERSION = "v1"
_FILES = {
    "train": Path("train_data") / "train_task_3_4.csv",
    "public": Path("test_data") / "test_public_task_4_more_splits.csv",
    "private": Path("test_data") / "test_private_task_4_more_splits.csv",
    "questions": Path("metadata") / "question_metadata_task_3_4.csv",
    "subjects": Path("metadata") / "subject_metadata.csv",
    "students": Path("metadata") / "student_metadata_task_3_4.csv",
}


@dataclass
class _Matrix:
    users: np.ndarray                 # (n,) int
    correct: np.ndarray               # (n, 948) int8: -1 unanswered, 0/1 IsCorrect
    value: np.ndarray                 # (n, 948) int8: -1 unanswered, 1..4 chosen option
    targets: np.ndarray | None        # (n, 10, 948) bool (test sets only)


@dataclass
class _EediData:
    train: _Matrix
    public: _Matrix
    private: _Matrix
    question_subjects: list[list[int]]
    subjects: pd.DataFrame
    students: pd.DataFrame            # indexed by UserId: Gender, YearOfBirth, PremiumPupil


def _data_dir(root: Path) -> Path:
    return root / DATASET_DIR / "source_members" / "data"


def _pivot(df: pd.DataFrame, with_targets: bool) -> _Matrix:
    users = np.sort(df["UserId"].unique())
    row = np.searchsorted(users, df["UserId"].to_numpy())
    col = df["QuestionId"].to_numpy()
    correct = np.full((users.size, N_Q), -1, dtype=np.int8)
    value = np.full((users.size, N_Q), -1, dtype=np.int8)
    correct[row, col] = df["IsCorrect"].to_numpy(dtype=np.int8)
    value[row, col] = df["AnswerValue"].to_numpy(dtype=np.int8)
    targets = None
    if with_targets:
        targets = np.zeros((users.size, N_MASKS, N_Q), dtype=bool)
        for m in range(N_MASKS):
            t = df[f"IsTarget_{m}"].to_numpy() == 1
            targets[row[t], m, col[t]] = True
    return _Matrix(users.astype(np.int64), correct, value, targets)


def _load_matrix(dd: Path, key: str) -> _Matrix:
    src = dd / _FILES[key]
    st = src.stat()
    cache = cache_dir(CODE) / f"{key}_{_CACHE_VERSION}_{st.st_size}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return _Matrix(z["users"], z["correct"], z["value"], z["targets"] if "targets" in z.files else None)
    m = _pivot(pd.read_csv(src), with_targets=(key != "train"))
    arrays = {"users": m.users, "correct": m.correct, "value": m.value}
    if m.targets is not None:
        arrays["targets"] = m.targets
    tmp = cache.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(cache)
    return m


def _load(root: Path) -> _EediData:
    dd = _data_dir(root)
    q = pd.read_csv(dd / _FILES["questions"])
    qs: list[list[int]] = [[] for _ in range(N_Q)]
    for qid, subj in zip(q["QuestionId"], q["SubjectId"]):
        qs[int(qid)] = [int(v) for v in json.loads(subj)]
    subjects = pd.read_csv(dd / _FILES["subjects"], encoding="utf-8-sig")
    st = pd.read_csv(dd / _FILES["students"])
    st["YearOfBirth"] = pd.to_datetime(st["DateOfBirth"], errors="coerce").dt.year
    students = st.set_index("UserId")[["Gender", "YearOfBirth", "PremiumPupil"]].astype(float)
    return _EediData(_load_matrix(dd, "train"), _load_matrix(dd, "public"), _load_matrix(dd, "private"), qs,
                     subjects, students)


def _student_meta(data: _EediData, users: np.ndarray) -> pd.DataFrame:
    return data.students.reindex(users.astype(int)).reset_index(drop=True)


def majority_answers(correct: np.ndarray) -> np.ndarray:
    """Starter-kit baseline: most common correctness per question (argmax of bincount; ties -> 0, no data -> 0)."""
    n1 = (correct == 1).sum(axis=0)
    n0 = (correct == 0).sum(axis=0)
    return (n1 > n0).astype(np.int8)


def as_prediction_matrix(y: Any, n: int) -> tuple[np.ndarray | None, str]:
    """y -> int8 array (n, 948) with values in {-1, 0, 1}, or (None, reason)."""
    arr, why = as_float_array(y)
    if arr is None:
        return None, why
    if arr.shape != (n, N_Q):
        return None, f"y must have shape ({n}, {N_Q}), got {list(arr.shape)}"
    if not np.all(np.isfinite(arr)):
        return None, "y contains non-finite values"
    if not np.all(np.isin(arr, (-1.0, 0.0, 1.0))):
        return None, "y values must be -1 (no prediction), 0 or 1"
    return arr.astype(np.int8), ""


# ----------------------------------------------------------------------------------------------- adapter
class EediTask4Adapter:
    """TaskAdapter for FoR39 (see module docstring)."""

    discipline = CODE
    name = "eedi-neurips2020-task4"
    family = FAMILY
    metric = "organizer 10-mask accuracy"
    direction = "max"
    task_type = "adaptive_assessment_student_modelling"

    def __init__(self, data_root: str | None = None, partition_seed: int = PARTITION_SEED, n_dev: int = 64,
                 margin: float = 0.02, budget: Budget | None = None, **_: Any) -> None:
        self.root = resolve_data_root(data_root)
        self.partition_seed = int(partition_seed)
        self.n_dev = int(n_dev)
        self.margin = float(margin)
        self.budget = budget
        self._data: Lazy[_EediData] = Lazy(lambda: _load(self.root))
        self._setup: Lazy[dict] = Lazy(self._make_setup)

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        dd = _data_dir(self.root)
        missing = [str(dd / p) for p in _FILES.values() if not (dd / p).exists()]
        if missing:
            return False, f"missing Eedi Task 4 files: {missing}"
        return True, f"Eedi NeurIPS 2020 Task 3/4 official members at {dd}"

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 1:
            raise ValueError("items_per_episode must be >= 1")
        setup = self._setup.get()
        rng = episode_rng(CODE, split, seed, self.partition_seed)
        draws = draw_episodes({0: setup["pools"][split]}, {0: items_per_episode}, int(n), rng, f"{CODE}/{split}")
        return [self._episode(split, k, int(seed), items) for k, items in enumerate(draws)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Mean over masks of the pooled per-mask accuracy (invalid outputs counted with the reference)."""
        correct = np.zeros(N_MASKS)
        total = np.zeros(N_MASKS)
        for p in per_episode:
            if not p:
                continue
            masks = list(p.get("masks") or [])
            tot = list(p.get("total") or [])
            cor = p.get("correct")
            if cor is None:
                cor = p.get("ref_correct")
            if cor is None or not (len(masks) == len(tot) == len(cor)):
                raise ValueError("FoR39 pooled payload needs masks, total and correct (or ref_correct) of equal length")
            for m, c, t in zip(masks, cor, tot):
                correct[int(m)] += float(c)
                total[int(m)] += float(t)
        have = total > 0
        if not have.any():
            return None
        return float(np.mean(correct[have] / total[have]))

    # ------------------------------------------------------------ setup
    def _make_setup(self) -> dict:
        data = self._data.get()
        salt = f"{CODE}|{self.partition_seed}"
        pub_ids = [f"eedi-u{u}" for u in data.public.users]
        priv_ids = [f"eedi-u{u}" for u in data.private.users]
        pools = partition_pool(pub_ids, IID_FRACTIONS, f"{salt}|iid")
        pools["ood"] = hash_rank(priv_ids, f"{salt}|ood")
        mask_of: dict[str, int] = {}
        for ids, tag in ((pub_ids, "public"), (priv_ids, "private")):
            for r, i in enumerate(hash_rank(ids, f"{salt}|mask|{tag}")):
                mask_of[i] = r % N_MASKS
        where: dict[str, tuple[str, int]] = {}
        for tag, mat in (("public", data.public), ("private", data.private)):
            for r, u in enumerate(mat.users):
                where[f"eedi-u{u}"] = (tag, r)
        # dev: n_dev training students held out of load_train, one seeded 80/20 query/target mask each
        train_ids = [f"eedi-u{u}" for u in data.train.users]
        dev_ids = hash_rank(train_ids, f"{salt}|dev")[:self.n_dev]
        dev_rows = np.array(sorted(int(np.searchsorted(data.train.users, int(i[6:]))) for i in dev_ids), dtype=np.int64)
        keep = np.ones(data.train.users.size, dtype=bool)
        keep[dev_rows] = False
        rng = np.random.default_rng(int(hashlib.sha256(f"{salt}|devmask".encode()).hexdigest()[:15], 16))
        dev_correct = data.train.correct[dev_rows]
        dev_targets = np.zeros_like(dev_correct, dtype=bool)
        for r in range(dev_rows.size):
            obs = np.flatnonzero(dev_correct[r] >= 0)
            k = max(1, int(round(0.2 * obs.size)))
            dev_targets[r, rng.permutation(obs)[:k]] = True
        train_rows = np.flatnonzero(keep)
        majority = majority_answers(data.train.correct[train_rows])
        return {"pools": pools, "mask_of": mask_of, "where": where, "train_rows": train_rows, "dev_rows": dev_rows,
                "dev_targets": dev_targets, "majority": majority}

    # ------------------------------------------------------------ episode
    def _episode(self, split: str, k: int, seed: int, items: list[str]) -> Episode:
        data, setup = self._data.get(), self._setup.get()
        eid = episode_id(CODE, split, seed, k)
        n = len(items)
        mats = {"public": data.public, "private": data.private}
        rows = [setup["where"][i] for i in items]
        masks = np.array([setup["mask_of"][i] for i in items], dtype=np.int64)
        correct = np.stack([mats[t].correct[r] for t, r in rows])                   # hidden
        value = np.stack([mats[t].value[r] for t, r in rows])                       # hidden
        users = np.array([mats[t].users[r] for t, r in rows], dtype=np.int64)
        targets = np.stack([mats[t].targets[r, m] for (t, r), m in zip(rows, masks)])
        can_query = (correct >= 0) & ~targets
        majority = setup["majority"]
        tr_rows, dev_rows, dev_targets = setup["train_rows"], setup["dev_rows"], setup["dev_targets"]
        dev_correct = data.train.correct[dev_rows]
        dev_value = data.train.value[dev_rows]
        dev_can_query = (dev_correct >= 0) & ~dev_targets
        n_dev, n_tr = dev_rows.size, tr_rows.size
        key = hashlib.sha256(b"FoR39-receipt|" + eid.encode() + b"|" + hashlib.sha256(
            correct.tobytes() + value.tobytes()).digest()).digest()

        def _counts(pred: np.ndarray) -> tuple[list[int], list[int]]:
            cor = [int(((pred[i] == correct[i]) & targets[i]).sum()) for i in range(n)]
            tot = [int(targets[i].sum()) for i in range(n)]
            return cor, tot

        ref_pred = np.repeat(majority[None, :], n, axis=0)
        ref_cor, tot = _counts(ref_pred)

        def _score(cor: list[int]) -> float:
            accs = []
            for m in sorted(set(masks.tolist())):
                sel = masks == m
                accs.append(float(np.sum(np.array(cor)[sel]) / max(1, np.sum(np.array(tot)[sel]))))
            return float(np.mean(accs))

        ref_score = _score(ref_cor)
        dev_ref_acc = float(((majority[None, :] == dev_correct) & dev_targets).sum() / dev_targets.sum())

        # ---- receipts
        def _sign(body: dict) -> str:
            msg = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            return hmac.new(key, msg, hashlib.sha256).hexdigest()

        def _reveal(sel: Any, cq: np.ndarray, cor: np.ndarray, val: np.ndarray, which: str) -> dict:
            arr, why = as_float_array(sel)
            if arr is None:
                raise ValueError(f"selections: {why}")
            if arr.ndim == 1 and cq.shape[0] == 1:
                arr = arr[None, :]
            if arr.ndim != 2 or arr.shape[0] != cq.shape[0]:
                raise ValueError(f"selections must have shape ({cq.shape[0]}, k) with k <= {QUERY_BUDGET}, "
                                 f"got {list(arr.shape)}")
            if arr.shape[1] > QUERY_BUDGET:
                raise ValueError(f"at most {QUERY_BUDGET} questions per student per call (got {arr.shape[1]})")
            if not np.all(np.isfinite(arr)) or not np.all(arr == np.round(arr)):
                raise ValueError("selections must be integer question ids (or -1 for no query)")
            sel_i = arr.astype(np.int64)
            revealed = np.full(cq.shape, -1, dtype=np.int8)
            revealed_v = np.full(cq.shape, -1, dtype=np.int8)
            cells: list[list[int]] = []
            for i in range(sel_i.shape[0]):
                qs = [int(q) for q in sel_i[i] if q != -1]
                if len(set(qs)) != len(qs):
                    raise ValueError(f"row {i}: duplicate question ids {qs}")
                for q in qs:
                    if not (0 <= q < N_Q) or not cq[i, q]:
                        raise ValueError(f"row {i}: question {q} is not queryable for this student (can_query is False)")
                    revealed[i, q] = cor[i, q]
                    revealed_v[i, q] = val[i, q]
                    cells.append([i, q])
            body = {"kind": RECEIPT_KIND, "episode": eid, "set": which, "cells": sorted(cells)}
            return {"revealed": revealed, "revealed_values": revealed_v,
                    "receipt": json.dumps({**body, "mac": _sign(body)}, sort_keys=True)}

        # ---- D_E tools
        def load_train(inputs: dict, config: dict) -> dict:
            return {"answers": data.train.correct[tr_rows].copy(), "answer_values": data.train.value[tr_rows].copy(),
                    "question_subjects": [list(s) for s in data.question_subjects],
                    "subjects": data.subjects.copy(), "student_meta": _student_meta(data, data.train.users[tr_rows])}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_can_query": dev_can_query.copy(), "dev_targets": dev_targets.copy(),
                    "dev_student_meta": _student_meta(data, data.train.users[dev_rows])}

        def query_dev_answers(inputs: dict, config: dict) -> dict:
            out = _reveal(inputs.get("selections"), dev_can_query, dev_correct, dev_value, "dev")
            return {"revealed": out["revealed"], "revealed_values": out["revealed_values"]}

        def score_dev(inputs: dict, config: dict) -> dict:
            pred, why = as_prediction_matrix(inputs.get("predictions"), n_dev)
            if pred is None:
                raise ValueError(f"predictions: {why}")
            if np.any(pred[dev_targets] == -1):
                raise ValueError("predictions must be 0/1 on every dev target cell")
            return {"dev_accuracy": float(((pred == dev_correct) & dev_targets).sum() / dev_targets.sum()),
                    "dev_reference_accuracy": dev_ref_acc}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"can_query": can_query.copy(), "targets": targets.copy(),
                    "student_meta": _student_meta(data, users)}

        def query_answers(inputs: dict, config: dict) -> dict:
            return _reveal(inputs.get("selections"), can_query, correct, value, "eval")

        mat_desc = "rows = students, columns = QuestionId 0..947"
        tools = [
            ToolSpec("load_train", f"Official Task 3/4 training data of {n_tr} students: answers[s, q] = 1 correct, "
                     "0 incorrect, -1 not answered; answer_values[s, q] = chosen option 1-4 (-1 not answered); "
                     "question_subjects[q] = subject ids of question q (hierarchy in subjects: SubjectId, Name, ParentId, "
                     "Level); student_meta = Gender, YearOfBirth, PremiumPupil (NaN = unknown) per student row.",
                     {}, {"answers": PortSchema("array", (n_tr, N_Q), unit="1", dtype="int", description=mat_desc),
                          "answer_values": PortSchema("array", (n_tr, N_Q), dtype="int", description=mat_desc),
                          "question_subjects": PortSchema("list", (N_Q,), description="list of subject ids per question"),
                          "subjects": PortSchema("table", ("n_subjects", 4), description="SubjectId, Name, ParentId, Level"),
                          "student_meta": PortSchema("table", (n_tr, 3), description="Gender, YearOfBirth, PremiumPupil")},
                     load_train),
            ToolSpec("load_dev_inputs", f"Dev replica of the protocol on {n_dev} held-out training students (not in "
                     "load_train), one mask each: dev_can_query[s, q] = queryable, dev_targets[s, q] = answer to predict.",
                     {}, {"dev_can_query": PortSchema("array", (n_dev, N_Q), dtype="bool", description=mat_desc),
                          "dev_targets": PortSchema("array", (n_dev, N_Q), dtype="bool", description=mat_desc),
                          "dev_student_meta": PortSchema("table", (n_dev, 3))},
                     load_dev_inputs),
            ToolSpec("query_dev_answers", f"Reveals dev answers: selections[s] = up to {QUERY_BUDGET} question ids "
                     "(-1 = none) with dev_can_query True; returns revealed (1/0 correctness, -1 elsewhere) and "
                     "revealed_values (option 1-4, -1 elsewhere).",
                     {"selections": PortSchema("array", (n_dev, "k"), dtype="int")},
                     {"revealed": PortSchema("array", (n_dev, N_Q), dtype="int"),
                      "revealed_values": PortSchema("array", (n_dev, N_Q), dtype="int")},
                     query_dev_answers),
            ToolSpec("score_dev", "Accuracy of 0/1 predictions (shape n_dev x 948; -1 allowed off-target) on the dev "
                     "target cells; also the accuracy of the reference predictor on the same cells.",
                     {"predictions": PortSchema("array", (n_dev, N_Q), dtype="int")},
                     {"dev_accuracy": PortSchema("number", unit="1"), "dev_reference_accuracy": PortSchema("number", unit="1")},
                     score_dev),
            ToolSpec("load_eval_inputs", f"The {n} evaluation students (answers hidden), in the row order y must follow: "
                     "can_query[s, q] = question q was answered by student s and may be queried; targets[s, q] = "
                     "answer whose correctness must be predicted (never queryable); student_meta per student.",
                     {}, {"can_query": PortSchema("array", (n, N_Q), dtype="bool", description=mat_desc),
                          "targets": PortSchema("array", (n, N_Q), dtype="bool", description=mat_desc),
                          "student_meta": PortSchema("table", (n, 3))},
                     load_eval_inputs),
            ToolSpec("query_answers", f"Reveals evaluation answers: selections[s] = up to {QUERY_BUDGET} question ids "
                     "(-1 = none) with can_query True. Returns revealed[s, q] (1/0 correctness, -1 not revealed), "
                     "revealed_values (option 1-4, -1 not revealed) and a signed receipt. Across ALL query_answers "
                     f"calls of the whole session at most {QUERY_BUDGET} distinct answers per student may be revealed, "
                     "including calls in nodes you later remove or replace (checked on the signed receipts of every "
                     "executed call, not only those in the final workflow).",
                     {"selections": PortSchema("array", (n, "k"), dtype="int")},
                     {"revealed": PortSchema("array", (n, N_Q), dtype="int"),
                      "revealed_values": PortSchema("array", (n, N_Q), dtype="int"),
                      "receipt": PortSchema("text", description="signed record of the revealed cells")},
                     query_answers),
        ]

        def c_output(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            pred, why = as_prediction_matrix(yv, n)
            if pred is None:
                return False, why
            miss = int((pred[targets] == -1).sum())
            return miss == 0, ("0/1 prediction on every target cell" if miss == 0
                               else f"{miss} target cells without a 0/1 prediction")

        def c_budget(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            return check_query_budget(trace, eid, key, n)

        constraints = [
            ConstraintSpec("output_shape", f"y is an integer array of shape ({n}, {N_Q}) (rows = load_eval_inputs "
                           "students) with values in {-1, 0, 1} and a 0/1 prediction on every target cell", c_output),
            ConstraintSpec("query_budget", f"at most {QUERY_BUDGET} distinct answers per evaluation student are revealed "
                           "by all query_answers calls of the whole session, also of nodes later removed or replaced "
                           "(verified from the signed receipts of every executed call)", c_budget),
        ]

        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            payload = {"item_ids": list(items), "masks": masks.tolist(), "total": list(tot), "correct": None,
                       "ref_correct": list(ref_cor)}
            base = {"reference": ref_score, "reference_name": "organizer starter baseline (per-question majority)",
                    "margin": self.margin, "n_items": n, "n_target_answers": int(np.sum(tot))}
            pred, why = as_prediction_matrix(yv, n)
            if pred is not None and np.any(pred[targets] == -1):
                pred, why = None, "missing predictions on target cells"
            if pred is None:
                return EvalResult(metrics={"reference_accuracy": ref_score}, primary=None, direction="max",
                                  accepted=False, details={**base, "norm_score": 0.0, "pooled_payload": payload,
                                                           "invalid": why})
            cor, _ = _counts(pred)
            score = _score(cor)
            payload["correct"] = cor
            return EvalResult(metrics={"mask_accuracy": score, "reference_accuracy": ref_score}, primary=score,
                              direction="max", accepted=bool(score >= ref_score + self.margin),
                              details={**base, "norm_score": norm_score(score, ref_score, "max"),
                                       "pooled_payload": payload})

        pool = "ood" if split == "ood" else "iid"
        objective = (
            "Education - personalised assessment (Eedi NeurIPS 2020 Education Challenge, Task 4). Students answered "
            f"multiple-choice diagnostic maths questions (QuestionId 0..{N_Q - 1}). For each of the {n} evaluation "
            "students, a set of their answered questions is held out as targets. Task: reveal the correctness of at "
            f"most {QUERY_BUDGET} of the student's other (queryable) answers with query_answers, then predict for every "
            "target question whether the student answered it correctly (1) or not (0).\n"
            f"Visible data: load_train returns the answer matrices of {n_tr} training students with question subjects "
            "and student metadata; load_dev_inputs / query_dev_answers / score_dev provide the same protocol on "
            f"{n_dev} held-out training students; load_eval_inputs returns can_query and targets masks of the "
            "evaluation students. query_answers only reveals queryable cells; target answers are never revealed.\n"
            f"Deliverable y: an integer array of shape ({n}, {N_Q}); y[s, q] = predicted correctness (0 or 1) of "
            "student s (row order of load_eval_inputs) on question q for every cell with targets[s, q] True; other "
            "cells are ignored (use -1). Evaluation metric: accuracy on the target answers (organizer 10-mask "
            "accuracy: accuracy per official mask, averaged over masks).\n"
            + scilib.describe("adaptive")
        )
        lineage = {
            "dataset": "Eedi NeurIPS 2020 Education Challenge Task 4",
            "version": "official data.zip / starter_kit.zip Task 3/4 members",
            "source_url": "https://www.eedischool.com/projects/neurips-education-challenge",
            "license": "see https://www.eedischool.com/projects/neurips-education-challenge (competition data terms)",
            "receipt": receipt_info(self.root / DATASET_DIR / "receipt.json"),
            "pool": pool, "pool_source": ("official private-leaderboard test file test_private_task_4_more_splits.csv"
                                          if pool == "ood" else
                                          "official public test file test_public_task_4_more_splits.csv"),
            "ood_kind": "proxy_within_dataset" if pool == "ood" else None,
            "ood_shift": ("official private-leaderboard partition: student-disjoint sample of the same population "
                          "(fewer answers per student on average); no demographic or time shift claimed")
            if pool == "ood" else None,
            "split": split, "split_seed": seed, "partition_seed": self.partition_seed, "index": k,
            "item_ids": list(items), "masks": masks.tolist(), "n_items": n, "n_target_answers": int(np.sum(tot)),
            "query_budget": QUERY_BUDGET, "query_rule": "strict: only answered, non-target questions of the item mask",
            "n_train_students": int(n_tr), "n_dev_students": int(n_dev),
            "rebuilt_split": True, "historical_ids_recovered": False,
        }
        return Episode(
            id=eid, discipline=CODE, family=FAMILY, split=split, task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n, N_Q), unit="1", dtype="int",
                                       description="0/1 correctness predictions on target cells, -1 elsewhere"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=1800.0, max_node_s=300.0,
                                         max_llm_items=4 * n),
            lineage=lineage,
            acceptance=(f"10-mask accuracy >= reference + {self.margin:g}; reference = organizer starter baseline "
                        "(most common training correctness per question), hard constraints incl. query budget hold"),
            tolerance={"rtol": 0.0, "atol": 0.0},
            tags=[CODE, "education", "student-modelling", "knowledge-tracing", "active-learning", "question-selection",
                  "binary-prediction", "accuracy", "eedi"],
            metric=self.metric, direction=self.direction, n_items=n,
            _evaluate=evaluate, _dev_evaluate=None,
        )


class _ForeignPickle(pickle.UnpicklingError):
    """The file references a class / function, so it is not a receipt (those are plain strings)."""


class _PlainUnpickler(pickle.Unpickler):
    """Unpickler for ledger files: a receipt is a plain ``str``, so no global (class / function) may be resolved.

    Besides skipping foreign ``receipt`` ports (arrays, tables) this keeps the scan from executing pickle payloads
    that policy code might have written next to the executor's own values.
    """

    def find_class(self, module: str, name: str) -> Any:
        raise _ForeignPickle(f"ledger pickles may not reference {module}.{name}")


def _load_plain(path: Path) -> Any:
    with open(path, "rb") as f:
        return _PlainUnpickler(f).load()


def solve_ledger_roots(trace: Trace) -> tuple[list[Path], str]:
    """Directories that hold every ``query_answers`` receipt this solve has ever produced.

    The solver lays a solve out as ``<run_dir>/exec`` (the interactive session: every executor run of every step, also
    of nodes the policy later removed or replaced, and of nodes inside operator bodies) and
    ``<run_dir>/replay/kNNN`` (the reset replay that ``trace`` describes, ``trace.run_dir``). Each tool node persists its
    outputs as ``<values>/<fingerprint>/<port>.pkl``, so the receipts are the ``receipt.pkl`` files below those two
    directories. Returns ``([], reason)`` when the layout is not the solver's (the budget then cannot be verified).
    """
    rd = str(getattr(trace, "run_dir", "") or "")
    if not rd:
        return [], "trace has no run_dir, so the solve's query ledger cannot be located"
    replay = Path(rd)
    if replay.parent.name != REPLAY_DIR:
        return [], f"trace.run_dir {rd!r} is not <solve>/{REPLAY_DIR}/<replay>; the solve's query ledger cannot be located"
    session = replay.parent.parent / SESSION_DIR
    if not session.is_dir():
        return [], f"session directory {str(session)!r} is missing; the solve's query ledger is incomplete"
    return [replay, session], ""


def check_query_budget(trace: Trace | None, eid: str, key: bytes, n: int) -> tuple[bool, str]:
    """Per-solve ledger check: at most QUERY_BUDGET distinct evaluation cells revealed per student.

    The ledger is the union of the signed eval receipts (a) of the tool nodes in ``trace`` and (b) of every
    ``receipt.pkl`` the executor persisted in the solve's session and replay directories (:func:`solve_ledger_roots`).
    (b) covers queries the policy made during the session and then deleted or replaced before submitting, and queries
    inside operator bodies, none of which appear in the final trace. Fails closed when the ledger cannot be located,
    a receipt is unreadable or a signature is invalid. Receipts that belong to another episode or to the dev set do
    not count.
    """
    if trace is None:
        return False, "query budget cannot be verified without an execution trace"
    roots, why = solve_ledger_roots(trace)
    if why:
        return False, why
    revealed: dict[int, set[int]] = {}
    macs: set[str] = set()
    seen: set[str] = set()

    def absorb(label: str, raw: Any) -> str | None:
        """Add one receipt to the ledger; returns an error text when the ledger cannot be trusted."""
        try:
            rc = json.loads(raw) if isinstance(raw, str) else None
        except json.JSONDecodeError:
            rc = None
        if not isinstance(rc, dict) or rc.get("kind") != RECEIPT_KIND:
            return None                               # a 'receipt' port of some other tool / node
        body = {k: rc[k] for k in ("kind", "episode", "set", "cells") if k in rc}
        if rc.get("episode") != eid or rc.get("set") != "eval":
            return None                               # dev receipts do not count
        msg = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        mac = str(rc.get("mac", ""))
        if not hmac.compare_digest(hmac.new(key, msg, hashlib.sha256).hexdigest(), mac):
            return f"{label}: receipt signature invalid"
        try:
            cells = [(int(i), int(q)) for i, q in rc["cells"]]
        except (KeyError, TypeError, ValueError):
            return f"{label}: malformed receipt cells"
        macs.add(mac)
        for i, q in cells:
            revealed.setdefault(i, set()).add(q)
        return None

    for nid, rec in trace.records.items():            # (a) the workflow's own tool nodes
        if rec.kind != "tool" or rec.status != "ok":
            continue
        if "receipt" not in (rec.outputs_summary or {}) and "receipt" not in (rec.output_refs or {}):
            continue
        ref = (rec.output_refs or {}).get("receipt")
        if not ref:
            return False, f"node {nid}: receipt output was not recorded; query budget cannot be verified"
        try:
            seen.add(str(Path(ref).resolve()))
            raw = _load_plain(Path(ref))
        except _ForeignPickle:
            continue                                  # a 'receipt' port of some other tool (not a string)
        except Exception as ex:   # unreadable pickle -> cannot verify: fail closed
            return False, f"node {nid}: cannot read receipt ({type(ex).__name__}: {ex})"
        err = absorb(f"node {nid}", raw)
        if err:
            return False, err
    for root in roots:                                # (b) everything the executor ever persisted for this solve
        for f in sorted(root.rglob(RECEIPT_FILE)):
            fid = str(f.resolve())
            if fid in seen:
                continue
            seen.add(fid)
            try:
                raw = _load_plain(f)
            except _ForeignPickle:
                continue
            except Exception as ex:
                return False, f"ledger file {f.name} under {root.name}: cannot read ({type(ex).__name__}: {ex})"
            err = absorb(f"ledger file under {root.name}", raw)
            if err:
                return False, err
    over = {i: len(s) for i, s in revealed.items() if len(s) > QUERY_BUDGET}
    if over:
        return False, f"students over the {QUERY_BUDGET}-answer budget in the solve ledger (row: revealed): {over}"
    used = max((len(s) for s in revealed.values()), default=0)
    return True, f"{len(macs)} query receipt(s) in the solve ledger; max {used} distinct answers revealed per student"


Adapter = EediTask4Adapter

__all__ = ["Adapter", "EediTask4Adapter", "check_query_budget", "solve_ledger_roots", "majority_answers", "QUERY_BUDGET",
           "N_Q"]
