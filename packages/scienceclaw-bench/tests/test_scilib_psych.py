"""scilib.psych: metrics, parsing, features, models and the entry point on synthetic two-armed-bandit sessions."""
from __future__ import annotations

import math

import numpy as np
import pytest

import scilib
from scilib import psych
from scienceclaw.runtime.integrity import scan_code
from scienceclaw.bench.tasks import for52_psych201 as adapter


# ----------------------------------------------------------------------------------------------- synthetic data
def _bandit_session(seed: int, n: int = 60, alpha: float = 0.4, beta: float = 3.0, study: str = "wilson2014humans"):
    """One participant playing a two-armed bandit (keys Q/Y) with a delta-rule softmax policy."""
    rng = np.random.default_rng(seed)
    p_win = {"Q": 0.75, "Y": 0.25}
    q = {"Q": 0.0, "Y": 0.0}
    lines = ["There are two slot machines, labeled Q and Y."]
    responses = []
    for _ in range(n):
        pq = 1 / (1 + math.exp(-beta * (q["Q"] - q["Y"])))
        k = "Q" if rng.random() < pq else "Y"
        r = float(rng.random() < p_win[k])
        q[k] += alpha * (r - q[k])
        line = f"You press <<{k}>> and get {int(r)} points."
        responses.append({"pos": sum(len(x) + 1 for x in lines) + len("You press "), "response": k,
                          "options": ["Q", "Y"]})
        lines.append(line)
    text = "\n".join(lines)
    return {"study": study, "text": text, "responses": responses}


def _sessions(count=24, **kw):
    return [_bandit_session(i, **kw) for i in range(count)]


def _cut(session, i):
    r = session["responses"][i]
    return {"study": session["study"], "history": session["text"][: r["pos"]], "options": list(r["options"]),
            "options_text": "Q, Y", "target": r["response"]}


# ----------------------------------------------------------------------------------------------- metrics
def test_mode_prediction_matches_adapter():
    cases = [("You press <<Q>> <<Y>>", ["Q", "Y"]), ("You press <<Q>> <<Q>> <<Y>>", ["Q", "Y"]), ("nothing", ["B", "A"]),
             ("<<Y>> <<Z>> <<Z>>", ["Y", "W"]), ("<<A>> <<B>> <<B>> <<A>>", ["A", "B", "C"])]
    for hist, opts in cases:
        assert psych.mode_prediction(hist, opts) == adapter.reference_prediction(hist, opts)
    rng = np.random.default_rng(0)
    for _ in range(50):
        hist = " ".join(f"<<{rng.choice(list('ABCD'))}>>" for _ in range(rng.integers(0, 12)))
        opts = list(rng.permutation(list("ABCD"))[: rng.integers(2, 5)])
        assert psych.mode_prediction(hist, opts) == adapter.reference_prediction(hist, opts)


def test_accuracy_and_summarize():
    assert psych.accuracy(["A", "B"], ["A", "A"]) == 0.5
    assert psych.accuracy(["A", "B"], ["A", "B"], [["A"], ["A"]]) == 0.5          # illegal prediction is wrong
    assert math.isnan(psych.accuracy([], []))
    with pytest.raises(ValueError):
        psych.accuracy(["A"], [])
    s = psych.summarize(["Q", "Y"], ["Q", "Q"], ["<<Q>> <<Q>>", "<<Y>> <<Q>> <<Y>>"], [["Q", "Y"], ["Q", "Y"]])
    assert s["n_items"] == 2 and s["accuracy"] == 0.5 and s["mode_accuracy"] == 1.0
    assert "margin" not in s


# ----------------------------------------------------------------------------------------------- parsing
def test_parse_trials_outcome_formats():
    tr = psych.parse_trials("There are two machines.\nYou press <<Q>> and get 36 points.\nYou press <<Y>> and get -4 points.\n"
                            "You press <<Q>> and get 0.5 points.\nYou press ", options=["Q", "Y"])
    assert tr.choice_keys == ["Q", "Y", "Q"] and list(tr.reward) == [36.0, -4.0, 0.5]
    assert tr.target_options == ["Q", "Y"]
    # counterfactual amounts of the unchosen option (anllo2024weird)
    tr = psych.parse_trials("You encounter stimuli G, X. You press <<G>>. You receive a reward of 0. "
                            "You would have received 10, had you pressed X.\nYou encounter stimuli G, X. You press ",
                            options=["G", "X"])
    assert tr.reward[0] == 0.0 and tr.forgone[0] == {"X": 10.0} and tr.options[0] == ["G", "X"]
    # "Obtained Outcome = ." (nasioulas2024feedback): the amount after the marker's sentence
    tr = psych.parse_trials("Choose: (50pts, 70%; 0pts) vs (33pts, 100%). You press <<R>>. Obtained Outcome = 33.\n"
                            "Choose: (50pts, 70%; 0pts) vs (33pts, 100%). You press ", options=["L", "R"])
    assert tr.reward[0] == 33.0
    tr = psych.parse_trials("You chose <<A>>. Obtained Outcome = 4, Forgone Outcome = 7\nYou chose ", options=["A", "B"])
    assert tr.reward[0] == 4.0 and tr.forgone[0] == {"B": 7.0}
    # gains and losses in one line: the net value (steingroever2015data)
    tr = psych.parse_trials("You press <<Z>>. You win 100.0$ and lose 150.0$.\nYou press ", options=["Z", "Y"])
    assert tr.reward[0] == -50.0
    # categorical feedback -> 1 / 0 (dezfouli2019, badham2017deficits, binz2022heuristics)
    tr = psych.parse_trials("Block 1: You pressed <<L>> and received a food reward.\nYou pressed <<L>> and received no food "
                            "reward.\nYou pressed ", options=["L", "R"])
    assert list(tr.reward) == [1.0, 0.0]
    tr = psych.parse_trials("You see a big white square. You press <<O>>. The correct category is O.\n"
                            "You see a small black square. You press <<O>>. The correct category is P.\nYou press ",
                            options=["O", "P"])
    assert list(tr.reward) == [1.0, 0.0]
    tr = psych.parse_trials("Alien Q scores 1.7 higher on attribute 1. You press <<Q>>. Alien Q wins\nYou press ",
                            options=["Q", "C"])
    assert tr.reward[0] == 1.0
    # outcome-less text: nan rewards, choices still parsed
    tr = psych.parse_trials("Do you take the gamble? <<A>>\nDo you take the gamble? <<B>>\nDo you take the gamble? ",
                            options=["A", "B"])
    assert tr.choice_keys == ["A", "B"] and np.isnan(tr.reward).all()


def test_parse_trials_dynamic_options_and_given_options():
    txt = ("You can choose between option K and option W. You press <<W>>.\n"
           "You can choose between option A and option K. You press <<K>>.\nYou can choose between option A and option W. You press ")
    tr = psych.parse_trials(txt, options=["A", "W"])
    assert tr.options == [["K", "W"], ["A", "K"]]
    tr = psych.parse_trials("x <<A>> y <<B>> z ", options=["A", "B", "C"], trial_options=[["A", "B"], ["A", "B", "C"]])
    assert tr.options == [["A", "B"], ["A", "B", "C"]]
    tr = psych.parse_trials("x <<A>> y <<B>> z ", options=["A", "B"], trial_options={2: ["A", "B", "C"], 9: ["B"]})
    assert tr.options[0] == ["A", "B", "C"] and tr.options[1] == ["A", "B"]
    choice, avail, rew, forg = tr.arrays()
    assert avail.shape == (3, len(tr.keys)) and forg.shape == (2, len(tr.keys))


# ----------------------------------------------------------------------------------------------- features
def test_trial_features_use_only_earlier_trials():
    s = _bandit_session(3, n=30)
    full = psych.parse_trials(s["text"], options=["Q", "Y"])
    names, F, ctx, avail, choice = psych.trial_features(full)
    assert F.shape[0] == len(full) + 1 and len(names) == F.shape[2] and ctx.shape[0] == len(full) + 1
    for cut in (5, 12, 20):
        head = psych.parse_trials(s["text"][: s["responses"][cut]["pos"]], options=["Q", "Y"])
        _, F2, ctx2, _, _ = psych.trial_features(head)
        assert F2.shape[0] == cut + 1
        assert np.allclose(F2, F[: cut + 1]) and np.allclose(ctx2, ctx[: cut + 1])      # prefix property: no look-ahead
    assert F[0].sum() == 0.0                                                            # no history before trial 0
    last = names.index("last")
    for t in range(1, len(full)):
        assert F[t, choice[t - 1], last] == 1.0 and F[t, :, last].sum() == 1.0


def test_make_pseudo_items_are_cut_inside_sessions():
    sessions = _sessions(5, n=20)
    items = psych.make_pseudo_items(sessions, per_session=3, seed=1)
    assert len(items) == 15 and psych.make_pseudo_items(sessions, 3, 1) == items                # deterministic
    for it in items:
        s = sessions[it["group"]]
        assert s["text"].startswith(it["history"]) and it["history"].count("<<") >= 2
        assert it["target"] in it["options"] and it["target"] == s["text"][len(it["history"]) + 2] and it["history"].endswith("You press ")


# ----------------------------------------------------------------------------------------------- models / entry point
def test_fit_predict_shapes_validity_and_determinism():
    sess = _sessions(10, n=40)
    items = [_cut(s, 25) for s in sess[:4]]
    for method in ("mode", "last", "kernel", "qlearn", "wsls"):
        a = psych.fit_predict(items, method=method)
        assert a == psych.fit_predict(items, method=method)
        assert len(a) == 4 and all(x in ("Q", "Y") for x in a)
    keys, proba = psych.fit_predict(items, method="qlearn", return_proba=True)
    assert [max(p, key=p.get) for p in proba] == keys
    assert all(abs(sum(p.values()) - 1) < 1e-9 for p in proba)
    with pytest.raises(ValueError):
        psych.fit_predict(items, method="nope")
    assert psych.fit_predict([]) == []


def test_fit_predict_reference_equals_participant_mode():
    sess = _sessions(6, n=30)
    items = [_cut(s, 20) for s in sess]
    ref = psych.fit_predict(items, method="mode")
    assert ref == [psych.mode_prediction(it["history"], it["options"]) for it in items]


def test_personal_models_recover_a_q_learner():
    """On delta-rule softmax players the fitted choice models predict at least as well as the participant mode."""
    sess = _sessions(16, n=80, beta=6.0)
    items = [_cut(s, i) for s in sess for i in (50, 70)]
    y = [it["target"] for it in items]
    scores = {m: psych.accuracy(y, psych.fit_predict(items, method=m)) for m in ("mode", "last", "qlearn", "kernel")}
    assert scores["qlearn"] >= scores["mode"] - 0.05, scores
    assert max(scores["qlearn"], scores["kernel"]) >= scores["last"] - 0.05, scores
    assert scores["qlearn"] > 0.6, scores


def test_gbdt_and_auto_run_with_training_sessions():
    train = _sessions(14, n=60)
    sess = [_bandit_session(100 + i, n=50) for i in range(6)]
    items = [_cut(s, 30) for s in sess]
    extra = [_cut(_bandit_session(200 + i, n=50), 30) for i in range(3)]
    for method in ("gbdt", "auto"):
        a = psych.fit_predict(items, train, method=method, extra_items=extra, seed=1)
        assert a == psych.fit_predict(items, train, method=method, extra_items=extra, seed=1)
        assert len(a) == 6 and all(x in ("Q", "Y") for x in a)
    # without training rows "gbdt" falls back to the participant mode instead of failing
    assert psych.fit_predict(items[:2], method="gbdt") == psych.fit_predict(items[:2], method="mode")


def test_handles_short_histories_and_unseen_options():
    items = [{"study": "x", "history": "Intro text.\n", "options": ["A", "B"]},
             {"study": "x", "history": "Intro <<A>> first.\n", "options": ["A", "B"]},
             {"study": "x", "history": "Intro <<A>> <<A>> <<C>>\n", "options": ["B", "A"]}]
    for method in psych.MODELS:
        out = psych.fit_predict(items, method=method)
        assert all(o in it["options"] for o, it in zip(out, items))


def test_cross_validate_folds_are_grouped_by_session():
    sess = _sessions(12, n=30)
    res = psych.cross_validate(sess, method="last", n_splits=3, per_session=2, seed=0, extra_methods=("mode",))
    assert set(res) == {"last", "mode"}
    for r in res.values():
        assert r["n_items"] == 24 and 0.0 <= r["accuracy"] <= 1.0 and "by_study" in r
    assert res["mode"]["accuracy"] == pytest.approx(res["mode"]["mode_accuracy"])
    one = psych.cross_validate(sess, method="mode", n_splits=3, per_session=2, seed=0)
    assert one["accuracy"] == pytest.approx(res["mode"]["accuracy"])


# ----------------------------------------------------------------------------------------------- language-model answers
def _long_item(n=800, seed=5):
    s = _bandit_session(seed, n=n)
    it = _cut(s, n - 1)
    return it


def test_build_prompts_styles_and_length():
    it = _long_item()
    assert len(it["history"]) > 20000
    short = {"study": "x", "history": "Task text.\nYou press <<Q>> and get 1 points.\nYou press ", "options": ["Q", "Y"]}
    for style in psych.PROMPT_STYLES:
        pr = psych.build_prompts([it, short], style=style)
        assert len(pr) == 2 and all(isinstance(p, str) for p in pr)
        assert len(pr[0]) < 16000 + 900 < 32000                                   # far below the llm-node prompt limit
        assert "Legal keys: Q, Y" in pr[0] and pr[0].endswith("Answer with the key only.")
        assert pr[0].startswith("Below is the transcript") and it["history"][-200:] in pr[0]     # newest trials are kept
        assert it["history"][:100] in pr[0] and "\n[...]\n" in pr[0]                 # instructions kept, middle cut
        assert "Task text.\nYou press <<Q>> and get 1 points.\nYou press " in pr[1] and "[...]" not in pr[1]
        assert psych.build_prompts([it, short], style=style) == pr                    # deterministic
    assert len(psych.build_prompts([it], "long")[0]) > len(psych.build_prompts([it], "transcript")[0])
    assert "probabilities" not in psych.build_prompts([short], "transcript")[0]
    assert psych.build_prompts([]) == []
    with pytest.raises(ValueError):
        psych.build_prompts([short], style="nope")


def test_build_prompts_model_style_states_the_fitted_probabilities():
    train = _sessions(8, n=40)
    items = [_cut(_bandit_session(50 + i, n=50), 40) for i in range(3)]
    pr = psych.build_prompts(items, "model", train_sessions=train)
    _, proba = psych.fit_predict(items, train, "personal", return_proba=True)
    for p, pb in zip(pr, proba):
        assert f"Q {pb['Q']:.2f}, Y {pb['Y']:.2f}" in p
    assert pr == psych.build_prompts(items, "model", train_sessions=train)


def test_parse_answers_cases():
    items = [{"options": ["J", "U"]}] * 8 + [{"options": ["A", "B", "C"]}]
    texts = ["J", " u.\n", "<<U>>", "The participant will press <<J>>.", "Answer: U", "J or U", "", None, "**B**"]
    assert psych.parse_answers(texts, items) == ["J", "U", "U", "J", "U", None, None, None, "B"]
    assert psych.parse_answers([], []) == []
    assert psych.parse_answers(["Z"], [{"options": ["J", "U"]}]) == [None]           # illegal key
    with pytest.raises(ValueError):
        psych.parse_answers(["J"], items)


def test_fit_predict_llm_answers_blend():
    sess = _sessions(10, n=40)
    items = [_cut(s, 25) for s in sess[:5]]
    base, pb = psych.fit_predict(items, method="kernel", return_proba=True)
    other = [("Y" if k == "Q" else "Q") for k in base]
    # weight 0 and missing answers leave the model untouched
    assert psych.fit_predict(items, method="kernel", llm_answers=[other], llm_weight=0.0) == base
    assert psych.fit_predict(items, method="kernel", llm_answers=[[None] * 5]) == base
    assert psych.fit_predict(items, method="kernel", llm_answers=[]) == base
    # weight 1: the answers decide; a legal answer list with one dissenter per item is a vote share
    assert psych.fit_predict(items, method="kernel", llm_answers=[other], llm_weight=1.0) == other
    keys, pr = psych.fit_predict(items, method="kernel", llm_answers=[other, base, other], llm_weight=0.5, return_proba=True)
    for k, o, p0, p1 in zip(keys, other, pb, pr):
        assert abs(sum(p1.values()) - 1) < 1e-9
        assert p1[o] == pytest.approx(0.5 * p0[o] + 0.5 * 2 / 3)
    # an item without a valid answer keeps its model probabilities
    a2 = list(other)
    a2[0] = None
    _, p2 = psych.fit_predict(items, method="kernel", llm_answers=[a2], llm_weight=0.5, return_proba=True)
    assert p2[0] == pytest.approx(pb[0])
    # the default weight is a real blend: a unanimous answer against a near-tie flips the item
    allk = psych.fit_predict(items, method="all", llm_answers=[other])
    assert set(allk) >= {"llm", "auto_llm", "auto", "personal"} and allk["llm"] == other
    with pytest.raises(ValueError):
        psych.fit_predict(items, method="kernel", llm_answers=[other[:2]])
    with pytest.raises(ValueError):
        psych.fit_predict(items, method="kernel", llm_answers=[other], llm_weight=1.5)


def test_documented_example_wiring_runs():
    """The two example code-node bodies of the module docstring pass the integrity scan and run on synthetic items."""
    import re
    import textwrap
    blocks = re.findall(r"^( +)def run\(inputs, config\):\n((?:\1 +.*\n|\n)+)", psych.__doc__, re.M)
    assert len(blocks) == 2
    fns = []
    for indent, body in blocks:
        code = "def run(inputs, config):\n" + textwrap.indent(textwrap.dedent(body), "    ")
        assert scan_code(code) == [], code
        ns: dict = {}
        exec(code, ns)
        fns.append(ns["run"])
    train = _sessions(8, n=40)
    ev = [_cut(_bandit_session(70 + i, n=50), 30) for i in range(3)]
    dev = [_cut(_bandit_session(80 + i, n=50), 30) for i in range(4)]
    prompts = fns[0]({"eval_items": ev, "dev_items": dev}, {})["prompts"]
    assert len(prompts) == 7 and all(isinstance(p, str) for p in prompts)
    replies = ["Q", "Y", "Q", "Y", None, "Q", "<<Y>>"]
    out = fns[1]({"eval_items": ev, "dev_items": dev, "train_sessions": train, "replies": replies}, {})
    assert len(out["y"]) == 3 and len(out["dev_pred"]) == 4
    assert all(k in ("Q", "Y") for k in out["y"] + out["dev_pred"])


# ----------------------------------------------------------------------------------------------- documentation / integrity
def test_describe_lists_public_names_and_scan_code_allows_import():
    doc = scilib.describe("psych")
    assert doc.startswith("Library `scilib.psych`")
    for name in psych.__all__:
        if name not in ("Trials", "MODELS"):
            assert name in doc, name
    code = "from scilib.psych import fit_predict\n\ndef run(inputs, config):\n    return {'y': []}\n"
    assert scan_code(code) == []
