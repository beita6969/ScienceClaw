"""scilib.captions: chrF++ parity with sacrebleu and the adapter, the greedy consensus string, folds, interface text."""
from __future__ import annotations

import random
import tempfile
from functools import partial

import numpy as np
import pytest

import scilib
from scilib import captions as C
from scienceclaw.bench.tasks import for45_americasnlp as m
from scienceclaw.runtime.integrity import scan_code

ALPHABET = list("abcdeñáx'") + ["-", ".", ",", "(", ")", "¿", "!", " "]


def _rand_text(rng: random.Random, max_words: int) -> str:
    words = ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 6))) for _ in range(rng.randint(0, max_words))]
    return " ".join(words)


def _language(seed: int, n: int, lo: int = 8, hi: int = 20, vocab: int = 300) -> list[str]:
    """Captions of a synthetic language: heavy-tailed (Zipf 1.1) words from a fixed invented vocabulary."""
    rng = np.random.default_rng(seed)
    letters = np.array(list("aeiokmnptuwy"))
    words = ["".join(rng.choice(letters, size=int(rng.integers(2, 7)))) for _ in range(vocab)]
    p = 1.0 / np.arange(1, vocab + 1) ** 1.1
    p /= p.sum()
    return [" ".join(rng.choice(words, size=int(rng.integers(lo, hi)), p=p)) for _ in range(n)]


def test_chrf_matches_sacrebleu_and_the_adapter():
    from sacrebleu.metrics import CHRF
    ref_metric = CHRF(word_order=2)
    rng = random.Random(1)
    for _ in range(400):
        hyp, ref = _rand_text(rng, 9), _rand_text(rng, 9)
        if not ref.strip():
            continue
        assert C.chrf(hyp, ref) == pytest.approx(ref_metric.sentence_score(hyp, [ref]).score, abs=1e-9)
    for hyp, ref in [("the cat sat", "the cat sat"), ("a cat sat on", "the cat sat"), ("", "abc"), ("abc", "a"),
                     ("¿qué? sí.", "qué sí"), ("kai-ta (wa)", "kai ta wa"), ("  ", "x y")]:
        assert C.chrf(hyp, ref) == pytest.approx(m.chrf_pp(hyp, ref), abs=1e-9)
    assert C.chrf("the cat sat", "the cat sat") == pytest.approx(100.0)
    assert C.chrf("", "abc") == 0.0


def test_mean_chrf_and_weights():
    refs = ["kai ta wa", "ta wa kuk", "wa"]
    assert C.mean_chrf("ta wa", refs) == pytest.approx(np.mean([C.chrf("ta wa", r) for r in refs]))
    assert C.mean_chrf("ta wa", refs, [1, 1, 1]) == pytest.approx(C.mean_chrf("ta wa", refs))
    assert C.mean_chrf("ta wa", refs, [0, 0, 5]) == pytest.approx(C.chrf("ta wa", "wa"))
    with pytest.raises(ValueError):
        C.mean_chrf("x", [])
    with pytest.raises(ValueError):
        C.mean_chrf("x", refs, [1, 1])


def test_medoid_matches_adapter_reference():
    caps = _language(3, 15)
    langs = ["x"] * len(caps)
    assert C.medoid(caps) == m.medoid_captions(caps, langs)["x"]
    assert C.medoid(caps + caps[:4]) == C.medoid(caps)                 # duplicates count once
    assert C.medoid(["only"]) == "only" and C.medoid(["  b ", "b", " "]) == "b"
    assert C.medoid(["aa bb", "aa bb cc", "zz"]) in ("aa bb", "aa bb cc")
    with pytest.raises(ValueError):
        C.medoid(["", "  "])
    with pytest.raises(TypeError):
        C.medoid(["a", 3])


def _naive_greedy(caps: list[str], max_words: int, vocab_size: int) -> str:
    """The definition in the module docstring, evaluated with the public ``mean_chrf`` only (no incremental code)."""
    from collections import Counter
    df = Counter(w for c in caps for w in set(c.split()))
    vocab = [w for w, _ in sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:vocab_size]]
    words: list[str] = []
    best = 0.0
    while len(words) < max_words:
        scores = [C.mean_chrf(" ".join(words + [v]), caps) for v in vocab]
        j = int(np.argmax(scores))
        if not scores[j] > best + 1e-9:
            break
        best = scores[j]
        words.append(vocab[j])
    return " ".join(words)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_consensus_equals_naive_greedy_and_reports_its_score(seed):
    caps = _language(seed, 9, vocab=25)
    det = C.consensus_details(caps, max_words=8, vocab_size=20, min_captions=1, jackknife=False)
    if not det["fallback"]:
        assert det["text"] == _naive_greedy(caps, 8, 20)
    assert det["visible_score"] == pytest.approx(C.mean_chrf(det["text"], caps))
    assert det["text"] == C.consensus_caption(caps, max_words=8, vocab_size=20, min_captions=1, jackknife=False)
    assert (len(det["words"]) <= 8 or det["fallback"]) and det["n_captions"] == 9
    if not det["fallback"]:
        assert det["step_scores"] == sorted(det["step_scores"]) and det["step_scores"][-1] == pytest.approx(det["visible_score"])


def test_consensus_edge_cases_and_determinism():
    caps = _language(4, 12)
    a = C.consensus_caption(caps)
    assert a == C.consensus_caption(caps) == C.consensus_caption(list(reversed(caps))) == C.consensus_caption(caps + [" "])
    assert a.strip() and len(a.split()) <= 25 and len(a) <= 1000
    capped = C.consensus_details(caps, max_words=3)
    assert len(capped["words"]) <= 3 or capped["fallback"]
    assert len(C.consensus_details(caps, max_words=3, min_captions=1)["words"]) <= 3 or capped["fallback"]
    assert len(C.consensus_details(_language(11, 12), max_words=4)["step_scores"]) <= 4
    short = C.consensus_caption(caps, max_chars=30)
    assert 0 < len(short) <= 30
    assert C.consensus_caption(["only one caption"]) == "only one caption"
    two = C.consensus_caption(["ta wa kuk", "kai ta wa"])
    assert two.strip() and C.mean_chrf(two, ["ta wa kuk", "kai ta wa"]) > 0
    assert C.consensus_caption(caps, min_captions=99) == C.medoid(caps)
    assert C.consensus_details(caps, min_captions=99)["fallback"] is True
    assert C.consensus_caption(caps, vocab_size=0) == C.medoid(caps)                 # nothing to append -> medoid
    with pytest.raises(ValueError):
        C.consensus_caption([])
    with pytest.raises(ValueError):
        C.consensus_caption(["", " "])
    with pytest.raises(TypeError):
        C.consensus_caption(["a", None])
    # long captions never produce an over-long string
    long_caps = [" ".join(["w%d" % j for j in range(i, i + 260)]) for i in range(0, 30, 10)]
    assert len(C.consensus_caption(long_caps, max_chars=1000)) <= 1000
    assert len(C.medoid(long_caps)) > 1000 and len(C.consensus_details(long_caps)["text"]) <= 1000


def test_consensus_weights():
    caps = _language(5, 10)
    assert C.consensus_caption(caps, weights=[2.0] * 10) == C.consensus_caption(caps)
    heavy = C.consensus_details(caps, weights=[0.0] * 9 + [1.0], jackknife=False)
    assert heavy["visible_score"] == pytest.approx(C.chrf(heavy["text"], caps[9]))
    assert heavy["visible_score"] > C.chrf(C.consensus_caption(caps, jackknife=False), caps[9])
    with pytest.raises(ValueError):
        C.consensus_caption(caps, weights=[1.0] * 9)
    with pytest.raises(ValueError):
        C.consensus_caption(caps, weights=[-1.0] + [1.0] * 9)


def test_consensus_generalises_on_a_synthetic_language_and_leave_one_out_sees_it():
    caps = _language(0, 24)
    train, held = caps[:12], caps[12:]
    assert C.mean_chrf(C.consensus_caption(train), held) > C.mean_chrf(C.medoid(train), held) + 1.0
    res = C.loo_compare(caps[:14], {"medoid": "medoid", "consensus": "consensus"}, max_folds=5, seed=0)
    assert res["difference"]["consensus"]["mean"] > 1.0
    assert res["difference"]["medoid"] == {"mean": 0.0, "sem": 0.0, "share_positive": 0.0}
    assert res["n_folds"] == 5


def test_by_language_and_fit_predict_never_mix_languages():
    a = [" ".join(w) for w in _lang_words("aeiou", 7, 11)]
    b = [" ".join(w) for w in _lang_words("kmnptw", 8, 12)]
    rows = ([{"id": str(i), "iso_lang": "aaa", "caption": c, "has_image": False} for i, c in enumerate(a)] +
            [{"id": str(i), "iso_lang": "bbb", "caption": c, "has_image": True} for i, c in enumerate(b)] +
            [{"id": "e", "iso_lang": "bbb", "caption": "  ", "has_image": False}])
    groups = C.by_language(rows)
    assert list(groups) == ["aaa", "bbb"] and groups["aaa"] == a and groups["bbb"] == b
    items = [{"index": 0, "iso_lang": "bbb"}, {"index": 1, "iso_lang": "aaa"}, {"index": 2, "iso_lang": "bbb"},
             {"index": 3, "iso_lang": "ccc"}]
    for method in ("consensus", "medoid"):
        y = C.fit_predict(rows, items, method=method)
        assert len(y) == 4 and all(isinstance(s, str) and s.strip() for s in y)
        assert y[0] == y[2] and set(y[0].replace(" ", "")) <= set("kmnptw")
        assert set(y[1].replace(" ", "")) <= set("aeiou")
        assert y[3] == "a"                                                   # language without training captions
    assert C.fit_predict(rows, items, method="medoid")[0] in b
    assert C.fit_predict(rows, items, fallback="zz")[3] == "zz"
    assert C.fit_predict(rows, []) == []
    with pytest.raises(ValueError):
        C.fit_predict(rows, items, method="nope")


def _lang_words(letters: str, lo: int, hi: int) -> list[list[str]]:
    rng = np.random.default_rng(len(letters))
    pool = ["".join(rng.choice(list(letters), size=int(rng.integers(2, 5)))) for _ in range(12)]
    return [[pool[int(j)] for j in rng.integers(0, 12, size=int(rng.integers(lo, hi)))] for _ in range(10)]


def test_loo_score_folds_seeds_and_errors():
    caps = _language(7, 14)
    loo = C.loo_score(caps, "medoid")
    assert loo["n_folds"] == 14 and loo["n_test"] == 1.0 and set(loo["per_language"]) == {""}
    assert loo["mean"] == pytest.approx(np.mean([C.chrf(C.medoid(caps[:i] + caps[i + 1:]), caps[i]) for i in range(14)]))
    sub = C.loo_score(caps, "medoid", max_folds=5, seed=3)
    assert sub["n_folds"] == 5 and sub == C.loo_score(caps, "medoid", max_folds=5, seed=3)
    rep = C.loo_score(caps, partial(C.consensus_caption, max_words=6), n_train=8, n_repeats=4, seed=1)
    assert rep["n_folds"] == 4 and rep["n_test"] == 6.0 and 0 < rep["mean"] < 100
    assert rep == C.loo_score(caps, partial(C.consensus_caption, max_words=6), n_train=8, n_repeats=4, seed=1)
    assert rep["mean"] != C.loo_score(caps, partial(C.consensus_caption, max_words=6), n_train=8, n_repeats=4, seed=2)["mean"]
    assert C.loo_score(caps, "medoid", n_train=999, n_repeats=2)["n_test"] == 1.0          # clipped: one test caption left
    multi = C.loo_score({"aaa": caps, "bbb": _language(8, 9)}, "medoid")
    assert multi["n_folds"] == 23 and multi["mean"] == pytest.approx(np.mean([v["mean"] for v in multi["per_language"].values()]))
    assert C.loo_score({"aaa": caps, "one": ["single"]}, "medoid")["n_folds"] == 14         # a 1-caption language has no fold
    with pytest.raises(ValueError):
        C.loo_score(["single"], "medoid")
    with pytest.raises(ValueError):
        C.loo_score(caps, "no_such_strategy")
    with pytest.raises(TypeError):
        C.loo_score(caps, lambda train: 3)
    with pytest.raises(TypeError):
        C.loo_score(caps, 5)


def test_loo_compare_uses_identical_folds():
    caps = _language(9, 12)
    seen = {"a": [], "b": []}

    def spy(name):
        def fn(train):
            seen[name].append(tuple(train))
            return C.medoid(train)
        return fn
    res = C.loo_compare(caps, {"a": spy("a"), "b": spy("b")}, n_train=6, n_repeats=3, seed=4)
    assert seen["a"] == seen["b"] and len(seen["a"]) == 3
    assert res["difference"]["b"]["mean"] == pytest.approx(0.0) and res["scores"]["a"]["mean"] == pytest.approx(res["scores"]["b"]["mean"])
    assert res["scores"]["a"]["mean"] == pytest.approx(C.loo_score(caps, "medoid", n_train=6, n_repeats=3, seed=4)["mean"])
    assert set(res["per_language"][""]) == {"a", "b"}


def test_singleton_rule_gives_no_credit_for_a_word_against_its_only_caption():
    caps = ["aa bb xxxxx", "aa bb yyyyy", "aa bb zzzzz", "aa bb wwwww"]
    on = C.consensus_details(caps, jackknife=True)
    off = C.consensus_details(caps, jackknife=False)
    assert on["fallback"] is False and set(on["words"]) <= {"aa", "bb"}
    assert set(off["words"]) & {"xxxxx", "yyyyy", "zzzzz", "wwwww"}          # memorised words pay off without the rule
    # a word shared by two captions is credited normally
    shared = ["aa bb xxxxx", "aa bb xxxxx", "aa bb yyyyy", "aa bb zzzzz"]
    assert "xxxxx" in C.consensus_details(shared, jackknife=True)["words"]
    # the rule changes only which words are credited, never the number of captions or the word budget
    for det in (on, off):
        assert det["n_captions"] == 4 and len(det["words"]) <= 25


def test_loo_lengths_matches_loo_compare_of_consensus_at_each_length():
    caps = _language(12, 12)
    lens = (3, 6, 12)
    res = C.loo_lengths(caps, lens, vocab_size=40, max_folds=5, seed=2)
    ref = C.loo_compare(caps, {str(n): partial(C.consensus_caption, max_words=n, vocab_size=40) for n in lens},
                        max_folds=5, seed=2)
    assert res["lengths"] == [3, 6, 12] and res["n_folds"] == 5
    for n in lens:
        assert res["scores"][str(n)]["mean"] == pytest.approx(ref["scores"][str(n)]["mean"], abs=1e-9)
    assert res["difference"]["3"]["mean"] == 0.0
    with pytest.raises(ValueError):
        C.loo_lengths(caps, [0, -1])


def test_loo_folds_are_capped_by_default():
    caps = _language(13, 34)
    calls = []

    def fn(train):
        calls.append(len(train))
        return "ta wa"
    assert C.loo_score(caps, fn)["n_folds"] == 30 and len(calls) == 30
    assert C.loo_score(caps, fn, max_folds=None)["n_folds"] == len(set(caps))


def _sentences_language(seed: int, n: int) -> list[str]:
    rng = np.random.default_rng(seed)
    pool = ["".join(rng.choice(list("aeiokmnptuwy"), size=int(rng.integers(3, 8)))) for _ in range(40)]
    p = 1.0 / np.arange(1, 41) ** 1.0
    p /= p.sum()
    caps = []
    for _ in range(n):
        clauses = [" ".join(rng.choice(pool, size=int(rng.integers(2, 5)), p=p)) + rng.choice([".", ",", ";"])
                   for _ in range(int(rng.integers(2, 5)))]
        caps.append(" ".join(clauses))
    return caps


def test_mbr_caption_is_a_sequence_of_clauses_of_the_given_captions():
    caps = _sentences_language(21, 14)
    text = C.mbr_caption(caps)
    assert text == C.mbr_caption(list(reversed(caps))) == C.mbr_caption(caps + [" "])
    units = set(C._clause_units(caps))
    if text != C.medoid(caps):
        assert len(text.split()) <= 40 and len(text) <= 1000
        rest = text
        for clause in sorted(units, key=len, reverse=True):          # every part of the string is a clause of some caption
            rest = rest.replace(clause, " ")
        assert not rest.strip()
    assert len(C.mbr_caption(caps, max_words=5).split()) <= 5 or C.mbr_caption(caps, max_words=5) == C.medoid(caps)
    assert C.mbr_caption(["only one caption"]) == "only one caption"
    assert 0 < len(C.mbr_caption(caps, max_chars=40)) <= 40
    heavy = C.mbr_caption(caps, weights=[0.0] * 13 + [1.0])
    assert heavy.strip()
    with pytest.raises(ValueError):
        C.mbr_caption(caps, weights=[1.0] * 13)
    with pytest.raises(ValueError):
        C.mbr_caption([" "])


def test_mbr_is_available_in_fit_predict_and_the_validation_helpers():
    caps = _sentences_language(22, 12)
    rows = [{"iso_lang": "x", "caption": c} for c in caps]
    items = [{"iso_lang": "x"}, {"iso_lang": "y"}]
    y = C.fit_predict(rows, items, method="mbr")
    assert y[0] == C.mbr_caption(caps) and y[1] == "a"
    assert C.fit_predict(rows, items, method="mbr", max_words=8)[0] == C.mbr_caption(caps, max_words=8)
    assert C.fit_predict(rows, items)[0] == C.consensus_caption(caps, max_words=25, vocab_size=300)
    res = C.loo_compare(caps, {"medoid": "medoid", "mbr": "mbr"}, max_folds=4, seed=1)
    assert res["n_folds"] == 4 and set(res["scores"]) == {"medoid", "mbr"}


def test_describe_lists_every_public_name():
    text = scilib.describe("captions")
    for name in C.__all__:
        assert name in text
    assert text.startswith("Library `scilib.captions`")


def test_code_node_may_import_scilib_and_runs_in_the_sandbox():
    code = ("from scilib import captions as C\n\n"
            "def run(inputs, config):\n"
            "    rows = inputs['train']\n"
            "    return {'y': C.fit_predict(rows, inputs['items']), 'loo': C.loo_score(C.by_language(rows)['x'], 'medoid')['mean']}\n")
    assert scan_code(code) == []
    from scienceclaw.core.graph import Node
    from scienceclaw.core.schema import PortSchema
    from scienceclaw.runtime.sandbox import run_code_node
    rows = [{"iso_lang": "x", "caption": c} for c in _language(10, 10)]
    vals = {"train": rows, "items": [{"iso_lang": "x"}, {"iso_lang": "x"}]}
    node = Node(id="solve", kind="code", code=code, inputs={k: PortSchema("any") for k in vals},
                outputs={"y": PortSchema("any"), "loo": PortSchema("any")})
    with tempfile.TemporaryDirectory() as td:
        out, meta = run_code_node(node, vals, td, 120)
    assert out is not None, meta
    assert out["y"] == C.fit_predict(rows, vals["items"]) and out["loo"] == pytest.approx(C.loo_score([r["caption"] for r in rows], "medoid")["mean"])
