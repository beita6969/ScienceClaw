"""scilib.molecules: featurisation, scaffold groups, tree ensemble, grouped CV, interface text, sandbox import."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

import scilib  # noqa: E402
from scilib import molecules as mol  # noqa: E402
from scienceclaw.runtime.integrity import scan_code  # noqa: E402

SUBST = ["C", "O", "N", "Cl", "F", "C(F)(F)F", "C#N", "[N+](=O)[O-]", "C(=O)O", "S(=O)(=O)N", "OC", "CC", "NC(C)=O"]
RINGS = ["c1ccc({a})c({b})c1", "C1CCC({a})C({b})C1", "c1cnc({a})c({b})c1", "C1CC({a})C({b})C1", "c1cc({a})c({b})s1",
         "c1nc({a})c({b})cn1"]
NITRO = "[N+](=O)[O-]"


def synthetic(n: int, seed: int, ring_ids=range(6)):
    """Substituted rings; label 1 iff the molecule carries a nitro group."""
    rng = np.random.default_rng(seed)
    smiles, labels, rings = [], [], []
    for _ in range(n):
        r = int(rng.choice(list(ring_ids)))
        a, b = rng.choice(SUBST, 2)
        smiles.append(RINGS[r].format(a=a, b=b))
        labels.append(int(NITRO in (a, b)))
        rings.append(r)
    return smiles, np.array(labels), np.array(rings)


def test_featurize_shapes_names_and_bad_input():
    smi = ["CCO", "c1ccccc1", "not-a-smiles", "", "CC(=O)Oc1ccccc1C(=O)O"]
    X, valid, names = mol.featurize(smi, n_jobs=1)
    assert X.shape == (5, 4480) == (5, len(names)) and X.dtype == np.float32
    assert valid.tolist() == [True, True, False, False, True]
    assert np.isfinite(X).all() and not X[2].any() and not X[3].any() and X[0].any()
    assert names == mol.feature_names("concat") and len(set(names)) == len(names)
    assert np.abs(X).max() <= 1e6
    # a unit-level check of the count semantics: benzene has 6 atoms -> 6 radius-0 environments summed over bits
    X0, _, _ = mol.featurize(["c1ccccc1"], "morgan_counts", radius=0, n_bits=64, n_jobs=1)
    assert X0.shape == (1, 64) and X0.sum() == 6


def test_kinds_are_concatenated_in_the_order_given():
    smi = ["CCO", "CCN", "c1ccccc1O"]
    parts = [mol.featurize(smi, k, n_bits=128, n_jobs=1)[0] for k in ("maccs", "descriptors", "morgan_counts")]
    both, _, names = mol.featurize(smi, ["maccs", "descriptors", "morgan_counts"], n_bits=128, n_jobs=1)
    assert np.array_equal(both, np.hstack(parts)) and len(names) == both.shape[1]
    assert mol.featurize(smi, "maccs", n_jobs=1)[0].shape == (3, 167)
    assert mol.featurize(smi, "atompair_counts", n_bits=256, n_jobs=1)[0].shape == (3, 256)
    with pytest.raises(ValueError):
        mol.featurize(smi, "bogus")
    with pytest.raises(ValueError):
        mol.featurize(smi, "morgan_counts", n_bits=4)


def test_unsanitisable_smiles_are_featurised_unsanitised():
    from rdkit import Chem
    weird = "C[N](C)(C)(C)(C)C"                         # pentavalent nitrogen: fails sanitisation, parses without it
    assert Chem.MolFromSmiles(weird) is None
    X, valid, _ = mol.featurize([weird, "CCO"], "maccs", n_jobs=1)
    assert valid.tolist() == [True, True] and X[0].any()
    assert np.isfinite(mol.featurize([weird], n_jobs=1)[0]).all()


def test_parallel_matches_serial_and_is_deterministic():
    smi, _, _ = synthetic(230, 0)                          # above the in-process threshold -> joblib path
    a, va, _ = mol.featurize(smi, "maccs", n_jobs=2)
    b, vb, _ = mol.featurize(smi, "maccs", n_jobs=1)
    assert np.array_equal(a, b) and np.array_equal(va, vb)


def test_scaffold_groups():
    smi = ["c1ccccc1C", "c1ccccc1CC", "Oc1ccccc1", "C1CCCCC1C", "CCO", "CCCC", "junk", "junk2"]
    g = mol.scaffold_groups(smi)
    assert g[0] == g[1] == g[2]                             # benzene scaffold
    assert g[3] != g[0]                                     # cyclohexane scaffold
    assert g[4] == g[5]                                     # no ring: empty scaffold
    assert len({g[6], g[7], g[0], g[3], g[4]}) == 5         # every unparsable SMILES has its own id
    assert g.dtype.kind == "i" and g[0] == 0


def test_tree_ensemble_learns_and_is_deterministic():
    smi, y, _ = synthetic(200, 1)
    te, ye, _ = synthetic(100, 2)
    X, _, _ = mol.featurize(smi, "concat", n_bits=128, n_jobs=1)
    Xq, _, _ = mol.featurize(te, "concat", n_bits=128, n_jobs=1)
    m1 = mol.TreeEnsemble(n_trees=30, seed=3).fit(X, y)
    s1, s2 = m1.predict(Xq), mol.TreeEnsemble(n_trees=30, seed=3, n_jobs=1).fit(X, y).predict(Xq)
    assert np.array_equal(s1, s2)                           # independent of n_jobs
    assert s1.shape == (100,) and np.isfinite(s1).all() and 0.0 <= s1.min() <= s1.max() <= 1.0
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(ye, s1) > 0.9
    with pytest.raises(ValueError):
        mol.TreeEnsemble(n_trees=5).fit(X, np.zeros(len(X), int))
    single = m1.predict(Xq[:1])                             # a score does not depend on the other molecules scored
    assert single[0] == pytest.approx(s1[0])


def test_fit_predict_smiles_and_matrices_agree():
    smi, y, _ = synthetic(160, 4)
    q1, yq, _ = synthetic(40, 5)
    q2, _, _ = synthetic(10, 6)
    kw = dict(kinds=["maccs", "morgan_counts"], n_bits=128, n_trees=25, n_jobs=1)
    s1, s2 = mol.fit_predict(smi, y, q1, q2, **kw)
    assert s1.shape == (40,) and s2.shape == (10,)
    one = mol.fit_predict(smi, y, q1, **kw)
    assert isinstance(one, np.ndarray) and np.array_equal(one, s1)          # a single set gives an array, not a list
    Xtr, _, _ = mol.featurize(smi, kw["kinds"], n_bits=128, n_jobs=1)
    Xq, _, _ = mol.featurize(q1, kw["kinds"], n_bits=128, n_jobs=1)
    assert np.array_equal(mol.fit_predict(Xtr, y, Xq, n_trees=25, n_jobs=1), s1)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(yq, s1) > 0.9
    with pytest.raises(ValueError):
        mol.fit_predict(smi, y)
    with pytest.raises(ValueError):
        mol.fit_predict(smi, y[:-1], q1, **kw)
    with pytest.raises(ValueError):
        mol.fit_predict(Xtr, y, Xq[:, :-1], n_trees=5)


def test_unparsable_molecules_still_get_finite_scores():
    smi, y, _ = synthetic(120, 7)
    smi[3] = "not-a-smiles"                                 # unparsable training row: dropped
    q = ["CCO", "not-a-smiles", "c1ccc([N+](=O)[O-])c(C)c1", "CC"]
    s = mol.fit_predict(smi, y, q, kinds="maccs", n_trees=20, n_jobs=1)
    assert np.isfinite(s).all() and ((0 <= s) & (s <= 1)).all()
    ok = np.array([True, False, True, True])
    assert s[1] == pytest.approx(np.median(s[ok]))          # neutral score for the unparsable query
    allbad = mol.fit_predict(smi, y, ["x", "y"], kinds="maccs", n_trees=10, n_jobs=1)
    assert allbad.tolist() == [0.5, 0.5]


def test_grouped_cv_respects_groups_and_detects_signal():
    smi, y, _ = synthetic(240, 8)
    r = mol.grouped_cv_auc(smi, y, "scaffold", n_splits=4, kinds="maccs", n_trees=20, n_jobs=1)
    g, fold = r["groups"], r["fold"]
    assert set(fold.tolist()) == {0, 1, 2, 3}
    for gid in np.unique(g):
        assert len(set(fold[g == gid].tolist())) == 1       # a scaffold never straddles two folds
    assert r["auc"] > 0.9 and len(r["fold_aucs"]) == 4 and np.isfinite(r["oof"]).all()
    plain = mol.grouped_cv_auc(smi, y, None, n_splits=3, kinds="maccs", n_trees=10, n_jobs=1)
    assert plain["groups"] is None and plain["auc"] > 0.9
    own = mol.grouped_cv_auc(smi, y, np.arange(len(y)) % 7, n_splits=3, kinds="maccs", n_trees=10, n_jobs=1)
    assert own["auc"] > 0.9
    X, _, _ = mol.featurize(smi, "maccs", n_jobs=1)
    with pytest.raises(ValueError):
        mol.grouped_cv_auc(X, y, "scaffold", kinds="maccs")
    with pytest.raises(ValueError):
        mol.grouped_cv_auc(smi, y, np.arange(3))


def test_grouped_cv_excludes_unparsable_rows():
    smi, y, _ = synthetic(100, 9)
    smi[0] = "not-a-smiles"
    r = mol.grouped_cv_auc(smi, y, "scaffold", n_splits=3, kinds="maccs", n_trees=10, n_jobs=1)
    assert np.isnan(r["oof"][0]) and r["fold"][0] == -1 and np.isfinite(r["oof"][1:]).all()


def test_describe_lists_every_public_name_and_kinds():
    text = scilib.describe("molecules")
    for name in ("featurize", "scaffold_groups", "fit_predict", "grouped_cv_auc", "TreeEnsemble", *mol.KINDS, "concat"):
        assert name in text


def test_code_node_may_import_scilib():
    code = "from scilib.molecules import fit_predict\n\ndef run(inputs, config):\n    return {}\n"
    assert scan_code(code) == []
