"""FoR34 Chemical sciences — OGB ``ogbg-molhiv`` (HIV replication inhibition), metric ROC-AUC (max).

Data: the official OGB release ``hiv.zip`` (OGB registry v1, 41,127 molecules, MoleculeNet HIV / DTP AIDS
Antiviral Screen labels, official scaffold split train 32,901 / valid 4,113 / test 4,113) as extracted by the
data team under ``<DATA_ROOT>/for34-ogbg-molhiv/extracted/hiv``.

* **Item** = one molecule (SMILES) with its binary label (1 = confirmed/moderately active, 0 = inactive).
  Item id = ``ogbg-molhiv/<graph index>`` (row of ``mapping/mol.csv.gz``).
* **Pools.** IID pool = official scaffold ``valid`` partition, cut once (``partition_seed``, stratified by label)
  into disjoint src / val / id sub-pools (55 / 15 / 30 %). OOD pool = official scaffold ``test`` partition.
  No second HIV-type MoleculeNet set is available locally, so the OOD shift is the official *scaffold* shift
  inside the same dataset (the OGB scaffold split puts the rarest Bemis-Murcko scaffolds into ``test``;
  ``valid`` and ``test`` scaffolds are disjoint): ``lineage["ood_kind"] = "proxy_within_dataset"``.
* **Episodes.** ``items_per_episode`` molecules (default 16) stratified so that every episode contains both
  classes: ``max(1, round(pos_frac * items))`` actives (default 25 %) and the rest inactives. This enriches
  actives relative to the natural ~2-3 % rate (ROC-AUC is prevalence invariant).
* **Visible data (D_E).** A per-episode deterministic sample of the official scaffold ``train`` partition
  (scaffold-disjoint from every evaluation molecule): ``n_train`` labelled molecules via ``load_train`` and a
  disjoint dev slice of ``n_dev`` molecules whose labels are held by ``score_dev`` (visible dev signal; the
  episode's ``_dev_evaluate`` is None). ``featurize_molecules`` is an RDKit featurizer (Morgan / MACCS /
  descriptor vectors).
* **Metric (D_V).** ROC-AUC = ``sklearn.metrics.roc_auc_score(y_true, y_score)`` — exactly the computation of
  the OGB ``Evaluator('ogbg-molhiv')`` for its single task.
* **Reference baseline** (deterministic): L2 logistic regression (``class_weight="balanced"``) on 11 trivial
  graph-count features (atoms, bonds, ring count, element fractions, aromatic / ring-atom fractions from the OGB
  raw node features), fit on the episode's ``load_train`` rows. **Acceptance:** ``AUC >= AUC_ref + margin``
  (default margin 0.05).
* **Hard constraints:** 1-D vector of length ``items``; finite; every value in [0, 1].
"""
from __future__ import annotations

import hashlib
import threading
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
    PARTITION_SEED, Lazy, PoolExhausted, as_float_vector, c_finite, c_range, c_vector, cache_dir, check_split,
    draw_episodes, episode_id, episode_rng, norm_score, partition_pool, receipt_info, resolve_data_root,
)

CODE = "FoR34"
FAMILY = "Physical & Earth"
DATASET_DIR = "for34-ogbg-molhiv"
IID_FRACTIONS = {"src": 0.55, "val": 0.15, "id": 0.30}
TRIVIAL_FEATURES = ("n_atoms", "n_bonds", "n_rings", "frac_C", "frac_N", "frac_O", "frac_S", "frac_halogen",
                    "frac_other", "frac_aromatic", "frac_in_ring")
_CACHE_VERSION = "v1"

RDKIT_DESCRIPTORS = ("MolWt", "HeavyAtomCount", "NumHAcceptors", "NumHDonors", "NumRotatableBonds", "RingCount",
                     "NumAromaticRings", "NumAliphaticRings", "TPSA", "MolLogP", "MolMR", "FractionCSP3",
                     "NumHeteroatoms", "NHOHCount", "NOCount", "NumValenceElectrons", "BertzCT", "LabuteASA",
                     "BalabanJ", "HallKierAlpha", "Kappa1", "Kappa2", "Chi0v", "Chi1v")


@dataclass
class _MolhivData:
    smiles: np.ndarray            # (N,) object
    labels: np.ndarray            # (N,) int8
    split: dict[str, np.ndarray]  # official scaffold split -> graph indices
    trivial: np.ndarray           # (N, len(TRIVIAL_FEATURES)) float


def _hiv_dir(root: Path) -> Path:
    return root / DATASET_DIR / "extracted" / "hiv"


def _trivial_features(hiv: Path) -> np.ndarray:
    """11 trivial per-graph count features from the OGB raw graph files (cached as .npz)."""
    cache = cache_dir(CODE) / f"trivial_features_{_CACHE_VERSION}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return z["X"]
    n_nodes = pd.read_csv(hiv / "raw" / "num-node-list.csv.gz", header=None)[0].to_numpy(dtype=np.int64)
    n_edges = pd.read_csv(hiv / "raw" / "num-edge-list.csv.gz", header=None)[0].to_numpy(dtype=np.int64)
    node = pd.read_csv(hiv / "raw" / "node-feat.csv.gz", header=None).to_numpy(dtype=np.int64)
    z_num = node[:, 0] + 1                     # OGB atom feature 0 = index into atomic numbers 1..118
    graph_of = np.repeat(np.arange(n_nodes.shape[0]), n_nodes)

    def frac(mask: np.ndarray) -> np.ndarray:
        return np.bincount(graph_of, weights=mask.astype(float), minlength=n_nodes.shape[0]) / np.maximum(n_nodes, 1)

    hal = np.isin(z_num, [9, 17, 35, 53])
    known = np.isin(z_num, [6, 7, 8, 16, 9, 17, 35, 53])
    X = np.column_stack([
        n_nodes, n_edges, n_edges - n_nodes + 1,
        frac(z_num == 6), frac(z_num == 7), frac(z_num == 8), frac(z_num == 16), frac(hal), frac(~known),
        frac(node[:, 7] == 1), frac(node[:, 8] == 1),
    ]).astype(float)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, X=X)
    tmp.replace(cache)
    return X


def _load(root: Path) -> _MolhivData:
    hiv = _hiv_dir(root)
    mol = pd.read_csv(hiv / "mapping" / "mol.csv.gz")
    split = {s: pd.read_csv(hiv / "split" / "scaffold" / f"{s}.csv.gz", header=None)[0].to_numpy(dtype=np.int64)
             for s in ("train", "valid", "test")}
    return _MolhivData(smiles=mol["smiles"].to_numpy(dtype=object), labels=mol["HIV_active"].to_numpy(dtype=np.int8),
                       split=split, trivial=_trivial_features(hiv))


def _item_id(i: int) -> str:
    return f"ogbg-molhiv/{int(i)}"


def _idx(item: str) -> int:
    return int(item.rsplit("/", 1)[1])


# ----------------------------------------------------------------------------------------------- RDKit featurizer
def _parse(smiles: str):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:                                  # e.g. hypervalent metal complexes: keep the graph unsanitized
        m = Chem.MolFromSmiles(smiles, sanitize=False)
        if m is not None:
            m.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(m)
    return m


def featurize_smiles(smiles: list[str], kind: str = "morgan", radius: int = 2, n_bits: int = 2048
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """RDKit features for a list of SMILES -> (X (n, d) float32, valid (n,) bool, feature names).

    kind: ``morgan`` (bit vector, ECFP-like, ``radius``/``n_bits``), ``morgan_counts`` (count vector),
    ``maccs`` (166 MACCS keys) or ``descriptors`` (RDKIT_DESCRIPTORS). Unparsable molecules give NaN rows.
    """
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    radius, n_bits = int(radius), int(n_bits)
    if kind in ("morgan", "morgan_counts"):
        if not (0 <= radius <= 6 and 16 <= n_bits <= 16384):
            raise ValueError("morgan needs 0 <= radius <= 6 and 16 <= n_bits <= 16384")
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        names = [f"morgan{'c' if kind == 'morgan_counts' else ''}_{j}" for j in range(n_bits)]
    elif kind == "maccs":
        names = [f"maccs_{j}" for j in range(167)]
    elif kind == "descriptors":
        names = list(RDKIT_DESCRIPTORS)
    else:
        raise ValueError(f"unknown featurizer kind {kind!r}; use morgan | morgan_counts | maccs | descriptors")
    X = np.full((len(smiles), len(names)), np.nan, dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        m = _parse(str(s))
        if m is None:
            continue
        try:
            if kind == "morgan":
                X[i] = gen.GetFingerprintAsNumPy(m)
            elif kind == "morgan_counts":
                X[i] = gen.GetCountFingerprintAsNumPy(m)
            elif kind == "maccs":
                X[i] = np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32)
            else:
                X[i] = [float(getattr(Descriptors, n)(m)) for n in names]
            valid[i] = True
        except Exception:   # RDKit raises assorted C++ errors on unsanitized molecules; the row stays NaN
            X[i] = np.nan
    return X, valid, names


# ----------------------------------------------------------------------------------------------- adapter
class MolhivAdapter:
    """TaskAdapter for FoR34 (see module docstring)."""

    discipline = CODE
    name = "ogbg-molhiv"
    family = FAMILY
    metric = "ROC-AUC"
    direction = "max"
    task_type = "molecular_property_classification"

    def __init__(self, data_root: str | None = None, partition_seed: int = PARTITION_SEED, n_train: int = 4000,
                 n_dev: int = 1000, pos_frac: float = 0.25, margin: float = 0.05, budget: Budget | None = None,
                 **_: Any) -> None:
        self.root = resolve_data_root(data_root)
        self.partition_seed = int(partition_seed)
        self.n_train, self.n_dev = int(n_train), int(n_dev)
        self.pos_frac = float(pos_frac)
        self.margin = float(margin)
        self.budget = budget
        self._data: Lazy[_MolhivData] = Lazy(lambda: _load(self.root))
        self._pools_lock = threading.Lock()
        self._pools: dict[str, dict[int, list[str]]] | None = None

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        hiv = _hiv_dir(self.root)
        need = [hiv / "mapping" / "mol.csv.gz", *[hiv / "split" / "scaffold" / f"{s}.csv.gz" for s in
                                                    ("train", "valid", "test")],
                *[hiv / "raw" / f for f in ("num-node-list.csv.gz", "num-edge-list.csv.gz", "node-feat.csv.gz")]]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            return False, f"missing OGB ogbg-molhiv files: {missing}"
        try:
            import rdkit  # noqa: F401
        except ImportError:
            return False, "RDKit is not installed (needed by the featurize_molecules tool)"
        return True, f"OGB ogbg-molhiv v1 at {hiv}"

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 2:
            raise ValueError("ROC-AUC episodes need >= 2 items (both classes)")
        data = self._data.get()
        pools = self._get_pools(data)[split]
        n_pos = max(1, int(round(self.pos_frac * items_per_episode)))
        n_pos = min(n_pos, items_per_episode - 1)
        rng = episode_rng(CODE, split, seed, self.partition_seed)
        draws = draw_episodes(pools, {1: n_pos, 0: items_per_episode - n_pos}, int(n), rng, f"{CODE}/{split}")
        return [self._episode(data, split, k, int(seed), items) for k, items in enumerate(draws)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """ROC-AUC over the concatenated items of all episodes (invalid outputs scored with the reference)."""
        from sklearn.metrics import roc_auc_score
        y_true: list[int] = []
        y_score: list[float] = []
        for p in per_episode:
            if not p:
                continue
            t = list(p.get("y_true") or [])
            s = p.get("y_score")
            if s is None:
                s = p.get("y_ref")
            if s is None or len(s) != len(t):
                raise ValueError("FoR34 pooled payload needs y_true and y_score (or y_ref) of equal length")
            y_true.extend(int(v) for v in t)
            y_score.extend(float(v) for v in s)
        if not y_true or len(set(y_true)) < 2:
            return None
        return float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))

    # ------------------------------------------------------------ pools
    def _get_pools(self, data: _MolhivData) -> dict[str, dict[int, list[str]]]:
        with self._pools_lock:
            if self._pools is None:
                valid_ids = [_item_id(i) for i in data.split["valid"]]
                strata = {_item_id(i): int(data.labels[i]) for i in data.split["valid"]}
                part = partition_pool(valid_ids, IID_FRACTIONS, f"{CODE}|{self.partition_seed}|iid", strata)
                pools: dict[str, dict[int, list[str]]] = {}
                for s, ids in part.items():
                    pools[s] = {c: [i for i in ids if strata[i] == c] for c in (0, 1)}
                test = data.split["test"]
                pools["ood"] = {c: [_item_id(i) for i in test if int(data.labels[i]) == c] for c in (0, 1)}
                self._pools = pools
            return self._pools

    def _train_sample(self, data: _MolhivData, ep_key: str) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic label-stratified sample of the official scaffold train partition -> (train_idx, dev_idx)."""
        rng = np.random.default_rng(int(hashlib.sha256(f"{CODE}|train|{ep_key}".encode()).hexdigest()[:15], 16))
        tr = data.split["train"]
        pos, neg = tr[data.labels[tr] == 1], tr[data.labels[tr] == 0]
        rate = pos.size / tr.size
        n_tot = min(self.n_train + self.n_dev, tr.size)
        k_pos = int(round(rate * n_tot))
        sel_pos = rng.permutation(pos)[:k_pos]
        sel_neg = rng.permutation(neg)[:n_tot - k_pos]
        dev_pos = int(round(rate * self.n_dev))
        dev = np.concatenate([sel_pos[:dev_pos], sel_neg[:self.n_dev - dev_pos]])
        train = np.concatenate([sel_pos[dev_pos:], sel_neg[self.n_dev - dev_pos:]])
        return rng.permutation(train), rng.permutation(dev)

    # ------------------------------------------------------------ episode
    def _episode(self, data: _MolhivData, split: str, k: int, seed: int, items: list[str]) -> Episode:
        from sklearn.metrics import roc_auc_score

        eid = episode_id(CODE, split, seed, k)
        idx = np.array([_idx(i) for i in items], dtype=np.int64)
        y_true = data.labels[idx].astype(int)
        n_items = len(items)
        tr_idx, dev_idx = self._train_sample(data, eid)
        tr_smiles = [str(s) for s in data.smiles[tr_idx]]
        tr_labels = data.labels[tr_idx].astype(np.int64)
        dev_smiles = [str(s) for s in data.smiles[dev_idx]]
        dev_labels = data.labels[dev_idx].astype(int)
        ev_smiles = [str(s) for s in data.smiles[idx]]
        n_tr, n_dev = len(tr_smiles), len(dev_smiles)

        ref = Lazy(lambda: self._reference(data, tr_idx, idx, dev_idx))

        # ---- D_E tools (visible data only)
        def load_train(inputs: dict, config: dict) -> dict:
            return {"smiles": list(tr_smiles), "labels": tr_labels.copy()}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_smiles": list(dev_smiles)}

        def score_dev(inputs: dict, config: dict) -> dict:
            s, why = as_float_vector(inputs.get("dev_scores"), n_dev)
            if s is None:
                raise ValueError(f"dev_scores: {why}")
            if not np.all(np.isfinite(s)):
                raise ValueError("dev_scores contains non-finite values")
            return {"dev_roc_auc": float(roc_auc_score(dev_labels, s)),
                    "dev_reference_roc_auc": float(ref.get()["dev_auc"]), "dev_n_active": int(dev_labels.sum())}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"smiles": list(ev_smiles)}

        def featurize(inputs: dict, config: dict) -> dict:
            smi = inputs.get("smiles")
            if smi is None or isinstance(smi, (str, bytes)):
                raise ValueError("smiles must be a list of SMILES strings")
            X, valid, names = featurize_smiles([str(s) for s in list(smi)], kind=str(config.get("kind", "morgan")),
                                               radius=int(config.get("radius", 2)),
                                               n_bits=int(config.get("n_bits", 2048)))
            return {"X": X, "valid": valid, "feature_names": names}

        tools = [
            ToolSpec("load_train", f"{n_tr} labelled training molecules sampled from the official OGB scaffold-split "
                     "training partition (scaffolds disjoint from all evaluation molecules). labels[i] = 1 if "
                     "smiles[i] was confirmed or moderately active in the DTP AIDS Antiviral Screen, else 0.",
                     {}, {"smiles": PortSchema("list", (n_tr,), dtype="str", description="SMILES strings"),
                          "labels": PortSchema("array", (n_tr,), unit="1", dtype="int", description="0/1 activity")},
                     load_train),
            ToolSpec("load_dev_inputs", f"{n_dev} further training-partition molecules (disjoint from load_train) "
                     "whose labels are withheld; score them with score_dev.",
                     {}, {"dev_smiles": PortSchema("list", (n_dev,), dtype="str", description="SMILES strings")},
                     load_dev_inputs),
            ToolSpec("score_dev", "ROC-AUC of dev_scores (one score per dev_smiles molecule, same order; higher = "
                     "more likely active) against the withheld dev labels; also returns the ROC-AUC of the adapter's "
                     "reference model on the same molecules.",
                     {"dev_scores": PortSchema("array", (n_dev,), dtype="float", description="score per dev molecule")},
                     {"dev_roc_auc": PortSchema("number", unit="1"), "dev_reference_roc_auc": PortSchema("number", unit="1"),
                      "dev_n_active": PortSchema("number", unit="1")},
                     score_dev),
            ToolSpec("load_eval_inputs", f"SMILES of the {n_items} evaluation molecules (labels hidden), in the order "
                     "that y must follow.",
                     {}, {"smiles": PortSchema("list", (n_items,), dtype="str", description="SMILES strings")},
                     load_eval_inputs),
            ToolSpec("featurize_molecules", "RDKit featurizer: X[i] = feature vector of smiles[i]; valid[i] = False "
                     "(row NaN) if RDKit cannot build the molecule.",
                     {"smiles": PortSchema("list", ("n",), dtype="str", description="SMILES strings")},
                     {"X": PortSchema("array", ("n", "d"), dtype="float"), "valid": PortSchema("array", ("n",), dtype="bool"),
                      "feature_names": PortSchema("list", ("d",), dtype="str")},
                     featurize,
                     config_doc="{kind: morgan|morgan_counts|maccs|descriptors (default morgan), radius: int (default 2), "
                                "n_bits: int (default 2048)}"),
        ]

        constraints: list[ConstraintSpec] = [
            c_vector(n_items, "one activity probability per evaluation molecule"),
            c_finite(n_items),
            c_range(n_items, 0.0, 1.0, "probability_range", "every value of y is a probability in [0, 1]"),
        ]

        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            r = ref.get()
            payload = {"item_ids": list(items), "y_true": y_true.tolist(), "y_score": None, "y_ref": r["eval_scores"]}
            arr, why = as_float_vector(yv, n_items)
            if arr is not None and not np.all(np.isfinite(arr)):
                arr, why = None, "non-finite values"
            base = {"reference": r["eval_auc"], "reference_name": "logistic regression on 11 trivial graph counts",
                    "margin": self.margin, "n_items": n_items, "n_active": int(y_true.sum())}
            if arr is None:
                return EvalResult(metrics={"reference_roc_auc": r["eval_auc"]}, primary=None, direction="max",
                                  accepted=False, details={**base, "norm_score": 0.0, "pooled_payload": payload,
                                                           "invalid": why})
            auc = float(roc_auc_score(y_true, arr))
            payload["y_score"] = arr.tolist()
            return EvalResult(metrics={"roc_auc": auc, "reference_roc_auc": r["eval_auc"]}, primary=auc, direction="max",
                              accepted=bool(auc >= r["eval_auc"] + self.margin),
                              details={**base, "norm_score": norm_score(auc, r["eval_auc"], "max"),
                                       "pooled_payload": payload})

        pool = "ood" if split == "ood" else "iid"
        objective = (
            "Chemical sciences - molecular property prediction on OGB ogbg-molhiv (MoleculeNet HIV). Each evaluation "
            "item is a small molecule given as a SMILES string. Its hidden label is 1 if the compound was confirmed or "
            "moderately active (inhibition of HIV replication) in the NCI DTP AIDS Antiviral Screen and 0 if inactive. "
            f"Task: for each of the {n_items} evaluation molecules, give the probability that it is active.\n"
            f"Visible data: load_train returns {n_tr} labelled molecules from the official OGB scaffold-split training "
            "partition (their Bemis-Murcko scaffolds never occur among the evaluation molecules; a few percent are "
            f"active). load_dev_inputs returns {n_dev} further training-partition molecules without labels; score_dev "
            "returns the ROC-AUC of scores for them; about half of these dev molecules have a Bemis-Murcko scaffold "
            "that also occurs among the load_train molecules. load_eval_inputs returns the evaluation SMILES (labels "
            "hidden). featurize_molecules computes RDKit fingerprints or descriptors. Every evaluation episode "
            "contains both active and inactive molecules.\n"
            f"Deliverable y: a 1-D float array of length {n_items}; y[i] is the predicted probability (in [0, 1]) that "
            "the i-th molecule returned by load_eval_inputs is active (same order). Evaluation metric: ROC-AUC of y "
            "against the hidden labels.\n"
            + scilib.describe("molecules")
        )
        lineage = {
            "dataset": "OGB ogbg-molhiv", "version": "OGB registry v1 (hiv.zip, 2020-05-04)",
            "source_url": "https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip",
            "license": "MIT (OGB); labels from MoleculeNet HIV / NCI DTP AIDS Antiviral Screen",
            "receipt": receipt_info(self.root / DATASET_DIR / "receipt.json"),
            "pool": pool, "pool_source": "official scaffold split 'test'" if pool == "ood"
            else "official scaffold split 'valid'",
            "ood_kind": "proxy_within_dataset" if pool == "ood" else None,
            "ood_shift": ("scaffold shift: OGB scaffold split test partition (rarest scaffolds, disjoint from the "
                          "valid-partition scaffolds of src/val/id and from the training partition)") if pool == "ood" else None,
            "split": split, "split_seed": seed, "partition_seed": self.partition_seed, "index": k,
            "item_ids": list(items), "n_items": n_items, "n_active": int(y_true.sum()),
            "train_source": "official scaffold split 'train'", "n_train": n_tr, "n_dev": n_dev,
            "train_ids_sha256": hashlib.sha256(",".join(map(str, tr_idx.tolist())).encode()).hexdigest(),
            "dev_ids_sha256": hashlib.sha256(",".join(map(str, dev_idx.tolist())).encode()).hexdigest(),
            "rebuilt_split": True, "historical_ids_recovered": False,
        }
        return Episode(
            id=eid, discipline=CODE, family=FAMILY, split=split, task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n_items,), unit="1", dtype="float",
                                       description="activity probability per evaluation molecule, load_eval_inputs order"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=1800.0, max_node_s=300.0,
                                         max_llm_items=4 * n_items),
            lineage=lineage,
            acceptance=(f"ROC-AUC >= reference ROC-AUC + {self.margin:g}; reference = L2 logistic regression "
                        "(class_weight=balanced) on 11 trivial graph-count features fit on the load_train molecules"),
            tolerance={"rtol": 1e-6, "atol": 1e-8},
            tags=[CODE, "chemistry", "molecules", "SMILES", "binary-classification", "ROC-AUC", "HIV", "bioactivity",
                  "ogbg-molhiv", "rdkit"],
            metric=self.metric, direction=self.direction, n_items=n_items,
            _evaluate=evaluate, _dev_evaluate=None,
        )

    def _reference(self, data: _MolhivData, tr_idx: np.ndarray, ev_idx: np.ndarray, dev_idx: np.ndarray) -> dict:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        def feats(ix: np.ndarray) -> np.ndarray:
            X = data.trivial[ix].copy()
            X[:, :3] = np.log1p(np.maximum(X[:, :3], 0.0))
            return X

        sc = StandardScaler().fit(feats(tr_idx))
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(feats(tr_idx)), data.labels[tr_idx])
        ev = clf.predict_proba(sc.transform(feats(ev_idx)))[:, 1]
        dv = clf.predict_proba(sc.transform(feats(dev_idx)))[:, 1]
        return {"eval_scores": ev.tolist(), "eval_auc": float(roc_auc_score(data.labels[ev_idx], ev)),
                "dev_auc": float(roc_auc_score(data.labels[dev_idx], dv))}


Adapter = MolhivAdapter

__all__ = ["Adapter", "MolhivAdapter", "featurize_smiles", "PoolExhausted"]
