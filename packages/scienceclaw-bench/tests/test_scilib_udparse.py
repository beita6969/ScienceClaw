"""scilib.udparse: exact tree decoders, metric parity with the FoR47 adapter, interface text, sandbox import, synthetic end-to-end."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import scilib
from scilib import udparse as up
from scienceclaw.bench.tasks import for47_ud as m
from scienceclaw.runtime.integrity import scan_code


def _is_projective(heads):
    for i, hi in enumerate(heads, 1):
        for j, hj in enumerate(heads, 1):
            a, b = sorted((i, hi))
            c, d = sorted((j, hj))
            if a < c < b < d:
                return False
    return True


def _brute(S, projective):
    n = S.shape[1]
    best, best_val = None, -np.inf
    for hs in itertools.product(range(n + 1), repeat=n):
        if any(hs[i] == i + 1 for i in range(n)) or up.check_parse({"head": list(hs), "deprel": ["dep"] * n}):
            continue
        if projective and not _is_projective(hs):
            continue
        v = sum(S[hs[i], i] for i in range(n))
        if v > best_val:
            best, best_val = hs, v
    return best_val


def test_decoders_return_the_best_single_root_tree():
    rng = np.random.default_rng(0)
    for _ in range(60):
        n = int(rng.integers(1, 6))
        S = rng.normal(size=(n + 1, n))
        for fn, proj in ((up.eisner, True), (up.chu_liu_edmonds, False)):
            h = fn(S)
            assert up.check_parse({"head": h.tolist(), "deprel": ["dep"] * n}) == []
            assert sum(S[h[i], i] for i in range(n)) == pytest.approx(_brute(S, proj))
            if proj:
                assert _is_projective(h.tolist())


def test_decoders_on_long_sentences_and_degenerate_scores():
    rng = np.random.default_rng(1)
    for n in (1, 2, 30, 80):
        for S in (rng.normal(size=(n + 1, n)), np.zeros((n + 1, n)), np.full((n + 1, n), -1e9)):
            for fn in (up.eisner, up.chu_liu_edmonds):
                h = fn(S)
                assert len(h) == n and up.check_parse({"head": h.tolist(), "deprel": ["dep"] * n}) == []


def test_check_parse_and_las_uas_match_the_adapter():
    rng = np.random.default_rng(2)
    for _ in range(40):
        n = int(rng.integers(2, 9))
        gold_h = up._tree_or_chain(rng.integers(0, n + 1, n))
        rels = ["nsubj", "obl:mod", "det", "punct", "root"]
        gold = m.Sentence("u", "s", "", tuple("abcdefghi"[:n]), ("_",) * n, ("X",) * n, ("_",) * n, ("_",) * n,
                          tuple(gold_h), tuple(rng.choice(rels, n)), ())
        h = rng.integers(0, n + 1, n).tolist()
        r = [str(x) for x in rng.choice(rels + ["obl", "nsubj:pass"], n)]
        assert (not up.check_parse({"head": h, "deprel": r})) == (not [i for i in m.tree_issues(h, r) if not i.startswith("relations")])
        sent = {"words": [{"form": f, "head": gh, "deprel": gr} for f, gh, gr in zip(gold.forms, gold.heads, gold.deprels)]}
        las, uas = m.attachment_counts(gold, h, r)
        got = up.las_uas([sent], [{"head": h, "deprel": r}])
        assert got["las"] == pytest.approx(las / n) and got["uas"] == pytest.approx(uas / n) and got["n_words"] == n
    assert up.las_uas([sent], [[[hh, rr] for hh, rr in zip(h, r)]])["n_words"] == n      # pair-list parses accepted


def test_describe_lists_every_public_name():
    text = scilib.describe("udparse")
    for name in up.__all__:
        assert name in text


def test_code_node_may_import_scilib():
    assert scan_code("from scilib import udparse\n\ndef run(inputs, config):\n    return {}\n") == []


# --------------------------------------------------------------------------------------------- synthetic treebank
_DET, _ADJ = ["le", "la", "un", "une", "ce"], ["petit", "grand", "rouge", "vieux", "beau", "lent"]
_NOUN = ["chat", "chien", "maison", "arbre", "livre", "table", "enfant", "jardin", "route", "pomme"]
_VERB = ["voit", "prend", "aime", "cherche", "porte", "garde"]
_ADP = ["dans", "sur", "sous", "vers"]


def _phrase(rng, noun_rel, verb):
    """[det] noun [adj] as word dicts whose ``head`` refers to other word dicts (None = root)."""
    noun = {"form": str(rng.choice(_NOUN)), "upos": "NOUN", "rel": noun_rel, "head": verb}
    out = [{"form": str(rng.choice(_DET)), "upos": "DET", "rel": "det", "head": noun}, noun]
    if rng.random() < 0.5:
        out.append({"form": str(rng.choice(_ADJ)), "upos": "ADJ", "rel": "amod", "head": noun})
    return out


def _sentence(rng):
    verb = {"form": str(rng.choice(_VERB)), "upos": "VERB", "rel": "root", "head": None}
    items = _phrase(rng, "nsubj", verb) + [verb] + _phrase(rng, "obj", verb)
    if rng.random() < 0.5:
        obl = _phrase(rng, "obl", verb)
        items += [{"form": str(rng.choice(_ADP)), "upos": "ADP", "rel": "case", "head": obl[1]}] + obl
    items.append({"form": ".", "upos": "PUNCT", "rel": "punct", "head": verb})
    pos = {id(w): i + 1 for i, w in enumerate(items)}
    return {"words": [{"id": i + 1, "form": w["form"], "upos": w["upos"], "feats": "_",
                       "head": 0 if w["head"] is None else pos[id(w["head"])], "deprel": w["rel"]}
                      for i, w in enumerate(items)]}


def _treebank(n, seed):
    rng = np.random.default_rng(seed)
    return [_sentence(rng) for _ in range(n)]


def test_synthetic_generator_is_a_valid_treebank():
    for s in _treebank(30, 0):
        assert up.check_parse({"head": [w["head"] for w in s["words"]], "deprel": [w["deprel"] for w in s["words"]]}) == []


def test_fit_predict_learns_a_planted_grammar_and_is_deterministic():
    train, test = _treebank(150, 0), _treebank(40, 1)
    bare = [{"words": [{"id": w["id"], "form": w["form"]} for w in s["words"]], "multiword_tokens": []} for s in test]
    kw = dict(epochs=4, jackknife_folds=3, n_models=1)
    (p1,) = up.fit_predict(train, [bare], **kw)
    (p2,) = up.fit_predict(train, [bare], **kw)
    assert p1 == p2 and len(p1) == len(test)
    for s, p in zip(test, p1):
        assert up.check_parse(p, len(s["words"])) == []
        assert all(r in up.UD_RELATIONS for r in p["deprel"])
    sc = up.las_uas(test, p1)
    assert sc["las"] > 0.9 and sc["uas"] >= sc["las"]


def test_parser_object_api_and_edge_cases():
    train = _treebank(80, 3)
    model = up.UDParser(decode="cle", epochs=3, jackknife_folds=0).fit(train)
    forms = [["le", "xyzzy", "voit", "un", "inconnu", "."], ["ok"], ["inconnu"] * 3]
    parses = model.parse(forms)                                   # plain form lists, unseen words, one-word sentence
    assert [len(p["head"]) for p in parses] == [6, 1, 3]
    assert parses[1] == {"head": [0], "deprel": ["root"]}
    assert all(up.check_parse(p) == [] for p in parses)
    assert model.parse([]) == [] and model.parse([{"words": []}]) == [{"head": [], "deprel": []}]
    S = model.arc_scores(forms)
    assert S[0].shape == (7, 6) and model.tag(forms)[0][0] == "DET"
    with pytest.raises(ValueError):
        up.UDParser(decode="greedy")


def test_tagger_and_jackknife_and_cross_validate():
    train = _treebank(120, 5)
    tg = up.UPOSTagger().fit(train)
    test = _treebank(30, 6)
    pred = tg.predict(test)
    acc = np.mean([a == w["upos"] for ps, s in zip(pred, test) for a, w in zip(ps, s["words"])])
    assert acc > 0.95
    oof = tg.jackknife(n_folds=4)
    assert [len(x) for x in oof] == [len(s["words"]) for s in train]
    cv = up.cross_validate(train[:60], n_folds=2, epochs=3, jackknife_folds=2)
    assert set(cv) >= {"las", "uas", "upos_accuracy", "n_words", "per_fold"} and len(cv["per_fold"]) == 2
    assert cv["las"] > 0.7
