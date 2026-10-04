"""Synthetic regression TaskAdapter used as a deterministic fixture for the pipeline tests.

Task (discipline code ``"SYN"``, family ``"Engineering & computing"``): noisy 2-D regression

    y = sin(x1) + 0.5 * x2**2 + eps,   eps ~ N(0, NOISE_SD**2)

* IID pool (splits ``src``/``val``/``id``): x1, x2 ~ U[-2, 2].
* OOD pool (split ``ood``): a *shifted x-range* (x1 ~ U[1, 4], x2 ~ U[0.5, 3]), the same discipline with a
  different dataset, mirroring the same-discipline cross-dataset OOD split of the benchmark.

D_E tools (run in the parent, return visible data only):

* ``load_train()   -> X_train (n_train x 2), y_train (n_train,)``
* ``load_eval_inputs() -> X_eval (n_items x 2)``

Rows of ``X_eval`` = the episode's ``items_per_episode`` hidden evaluation items **plus** ``n_dev``
*dev rows*. Dev rows are a holdout slice carved by the adapter from the episode's visible training pool
(same generator/distribution as the training data, drawn after the training rows); their labels are
withheld from ``load_train`` and used only by the visible dev evaluator. Dev rows are interleaved with
the hidden rows at deterministic positions (a seeded permutation) that are never disclosed to the policy.
Consequently ``n_items`` (the required output length) = ``items_per_episode + n_dev``.

D_V:

* hard constraints (visible): ``length`` (len(y) == n_items) and ``finite`` (all entries finite);
* metric: RMSE over the hidden rows (direction ``min``);
* reference baseline: predict the training-set mean for every row;
* acceptance: RMSE <= 0.9 * reference RMSE;
* dev evaluator (visible): RMSE over the dev rows (plus the same statistic for the reference baseline).

``EvalResult.details`` carries ``reference``, ``norm_score`` (= reference/RMSE clipped to [0, 10]) and
``pooled_payload`` = {"y_true", "y_pred", "y_ref"} over the hidden rows. :meth:`SyntheticAdapter.pooled_metric`
computes the pooled RMSE over all episodes; an invalid submission (``y_pred`` is None) is scored with the
reference-baseline predictions ``y_ref`` (i.e. it earns no credit over the baseline).
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from scienceclaw.core.schema import PortSchema
from scienceclaw.core.trace import Trace
from scienceclaw.bench.task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec

SYN_CODE = "SYN"
SYN_FAMILY = "Engineering & computing"
SYN_NAME = "synthetic-sin-regression"
SYN_VERSION = "synthetic-v1"

NOISE_SD = 0.1
IID_RANGE: tuple[tuple[float, float], tuple[float, float]] = ((-2.0, 2.0), (-2.0, 2.0))
OOD_RANGE: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 4.0), (0.5, 3.0))
ACCEPT_RATIO = 0.9
NORM_CLIP = 10.0

_POOL_OF_SPLIT = {"src": "iid", "val": "iid", "id": "iid", "ood": "ood"}


def synthetic_truth(X: np.ndarray) -> np.ndarray:
    """Noise-free response f(x) = sin(x1) + 0.5*x2^2 (used by tests as the 'perfect' predictor)."""
    X = np.asarray(X, dtype=float)
    return np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def _derive_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:15], 16)


def _as_float_vector(y: Any) -> tuple[np.ndarray | None, str]:
    """Convert a submitted output to a 1-D float vector; returns (vector | None, reason)."""
    try:
        arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as ex:
        return None, f"output is not numeric: {type(ex).__name__}: {ex}"
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        return None, f"output must be 1-D, got shape {list(arr.shape)}"
    return arr, ""


class SyntheticAdapter:
    """Synthetic regression adapter implementing the ``bench.task.TaskAdapter`` protocol."""

    discipline = SYN_CODE
    name = SYN_NAME
    family = SYN_FAMILY
    metric = "RMSE"
    direction = "min"
    task_type = "regression"

    def __init__(self, n_train: int = 48, n_dev: int = 8, noise_sd: float = NOISE_SD,
                 budget: Budget | None = None) -> None:
        if n_train < 2 or n_dev < 1:
            raise ValueError("SyntheticAdapter needs n_train >= 2 and n_dev >= 1")
        self.n_train = int(n_train)
        self.n_dev = int(n_dev)
        self.noise_sd = float(noise_sd)
        self.budget = budget

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        return True, "synthetic (generated on the fly, no data files)"

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        if split not in _POOL_OF_SPLIT:
            raise ValueError(f"SyntheticAdapter builds splits {tuple(_POOL_OF_SPLIT)}; 'rep' episodes are copies of "
                             f"src episodes made by SplitPlan (got {split!r})")
        if items_per_episode < 1:
            raise ValueError("items_per_episode must be >= 1")
        return [self._make_episode(split, k, int(seed), int(items_per_episode)) for k in range(int(n))]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Pooled RMSE over the hidden rows of all episodes (invalid outputs scored as the reference)."""
        y_true: list[float] = []
        y_pred: list[float] = []
        for p in per_episode:
            if not p:
                continue
            t = list(p.get("y_true") or [])
            pred = p.get("y_pred")
            if pred is None:
                pred = p.get("y_ref")
            if pred is None or len(pred) != len(t):
                raise ValueError("SYN pooled payload needs y_true and y_pred (or y_ref) of equal length")
            y_true.extend(float(v) for v in t)
            y_pred.extend(float(v) for v in pred)
        if not y_true:
            return None
        return _rmse(np.asarray(y_true), np.asarray(y_pred))

    # ------------------------------------------------------------ episodes
    def _make_episode(self, split: str, k: int, seed: int, n_eval: int) -> Episode:
        pool = _POOL_OF_SPLIT[split]
        ep_seed = _derive_seed(SYN_VERSION, seed, split, k)
        rng = np.random.default_rng(ep_seed)
        (lo1, hi1), (lo2, hi2) = IID_RANGE if pool == "iid" else OOD_RANGE
        n_tot = self.n_train + self.n_dev + n_eval
        X = np.column_stack([rng.uniform(lo1, hi1, n_tot), rng.uniform(lo2, hi2, n_tot)])
        y = synthetic_truth(X) + rng.normal(0.0, self.noise_sd, n_tot)

        tr = slice(0, self.n_train)
        dv = slice(self.n_train, self.n_train + self.n_dev)
        ev = slice(self.n_train + self.n_dev, n_tot)
        X_train, y_train = X[tr].copy(), y[tr].copy()
        X_dev, y_dev = X[dv].copy(), y[dv].copy()
        X_hid, y_hid = X[ev].copy(), y[ev].copy()

        n_items = n_eval + self.n_dev
        perm = rng.permutation(n_items)            # position p of X_eval holds stacked row perm[p]
        stacked = np.vstack([X_hid, X_dev])
        X_eval = stacked[perm]
        hidden_pos = np.array([int(np.where(perm == i)[0][0]) for i in range(n_eval)], dtype=int)
        dev_pos = np.array([int(np.where(perm == n_eval + i)[0][0]) for i in range(self.n_dev)], dtype=int)

        train_mean = float(y_train.mean())
        tag = f"{ep_seed & 0xFFFFFF:06x}"
        ep_id = f"SYN-{split}-{k:02d}-{tag}"
        item_ids = [f"SYN:{pool}:{split}:{k}:{tag}:e{i}" for i in range(n_eval)]
        dev_ids = [f"SYN:{pool}:{split}:{k}:{tag}:d{i}" for i in range(self.n_dev)]

        def load_train(inputs: dict, config: dict) -> dict:
            return {"X_train": X_train.copy(), "y_train": y_train.copy()}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"X_eval": X_eval.copy()}

        tools = [
            ToolSpec("load_train", "Labelled training data: feature matrix X_train (columns x1, x2) and response y_train.",
                     {}, {"X_train": PortSchema("array", (self.n_train, 2), dtype="float", description="features x1, x2"),
                          "y_train": PortSchema("array", (self.n_train,), dtype="float", description="observed response")},
                     load_train),
            ToolSpec("load_eval_inputs", "Unlabelled inputs X_eval (columns x1, x2) whose responses must be predicted.",
                     {}, {"X_eval": PortSchema("array", (n_items, 2), dtype="float", description="features x1, x2")},
                     load_eval_inputs),
        ]

        def c_length(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            arr, why = _as_float_vector(yv)
            if arr is None:
                return False, why
            return (arr.shape[0] == n_items), f"len(y)={arr.shape[0]}, required {n_items}"

        def c_finite(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            arr, why = _as_float_vector(yv)
            if arr is None:
                return False, why
            n_bad = int((~np.isfinite(arr)).sum())
            return n_bad == 0, ("all entries finite" if n_bad == 0 else f"{n_bad} non-finite entries")

        constraints = [
            ConstraintSpec("length", f"y is a 1-D array with exactly one prediction per row of X_eval ({n_items})", c_length),
            ConstraintSpec("finite", "every prediction is a finite real number", c_finite),
        ]

        ref_pred_hidden = np.full(n_eval, train_mean)
        ref_rmse = _rmse(y_hid, ref_pred_hidden)

        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            payload = {"y_true": y_hid.tolist(), "y_pred": None, "y_ref": ref_pred_hidden.tolist()}
            arr, why = _as_float_vector(yv)
            if arr is not None and arr.shape[0] != n_items:
                why = f"length {arr.shape[0]} != {n_items}"
            elif arr is not None and not np.all(np.isfinite(arr)):
                why = "non-finite predictions"
            if why:
                return EvalResult(metrics={"reference_rmse": ref_rmse}, primary=None, direction="min",
                                  accepted=False,
                                  details={"reference": ref_rmse, "norm_score": 0.0, "pooled_payload": payload,
                                           "invalid": why})
            pred = arr[hidden_pos]
            r = _rmse(y_hid, pred)
            payload["y_pred"] = pred.tolist()
            norm = NORM_CLIP if r == 0 else float(min(max(ref_rmse / r, 0.0), NORM_CLIP))
            return EvalResult(metrics={"rmse": r, "reference_rmse": ref_rmse}, primary=r, direction="min",
                              accepted=bool(r <= ACCEPT_RATIO * ref_rmse),
                              details={"reference": ref_rmse, "norm_score": norm, "pooled_payload": payload})

        dev_ref = _rmse(y_dev, np.full(self.n_dev, train_mean))

        def dev_evaluate(yv: Any) -> dict:
            arr, why = _as_float_vector(yv)
            if arr is None:
                return {"error": why}
            if arr.shape[0] != n_items:
                return {"error": f"length {arr.shape[0]} != {n_items}"}
            if not np.all(np.isfinite(arr[dev_pos])):
                return {"error": "non-finite predictions on dev rows"}
            return {"dev_rmse": _rmse(y_dev, arr[dev_pos]), "dev_reference_rmse": dev_ref, "n_dev": self.n_dev,
                    "note": "RMSE (lower is better) on rows reserved from the training pool and embedded in X_eval; "
                            "dev_reference_rmse = RMSE of predicting the training mean"}

        objective = (
            "Predict the scalar response for every row of X_eval. Each row has two real-valued features (x1, x2); "
            "the response is an unknown function of the features observed with noise. Labelled training data "
            f"(X_train, y_train; {self.n_train} rows) and the unlabelled inputs (X_eval; {n_items} rows) are provided "
            "by the tools. Deliverable: a 1-D float array y with one prediction per row of X_eval, in the same order."
        )
        return Episode(
            id=ep_id, discipline=SYN_CODE, family=SYN_FAMILY, split=split, task_type=self.task_type,
            objective=objective, required_output=PortSchema("array", ("n_items",), dtype="float",
                                                            description="one prediction per row of X_eval"),
            tools=tools, constraints=constraints, budget=self.budget or Budget(),
            lineage={"dataset": "synthetic: y = sin(x1) + 0.5*x2^2 + N(0, sd^2)", "version": SYN_VERSION,
                     "pool": pool, "x_range": [list(r) for r in (IID_RANGE if pool == "iid" else OOD_RANGE)],
                     "noise_sd": self.noise_sd, "seed": ep_seed, "split_seed": seed, "index": k,
                     "item_ids": item_ids, "dev_item_ids": dev_ids, "n_train": self.n_train,
                     "n_eval_items": n_eval, "n_dev": self.n_dev},
            acceptance=f"RMSE on the hidden evaluation rows <= {ACCEPT_RATIO} x the RMSE of the reference predictor "
                       "(the training-set mean)",
            tolerance={"rtol": 1e-6, "atol": 1e-8}, tags=["regression", "tabular", "synthetic"],
            metric=self.metric, direction=self.direction, n_items=n_items,
            _evaluate=evaluate, _dev_evaluate=dev_evaluate,
        )
