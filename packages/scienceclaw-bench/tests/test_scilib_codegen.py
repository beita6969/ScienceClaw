"""scilib.codegen: prompt builders, reply sanitizer (assembly rule of FoR46), repair prompts, candidate selection."""
from __future__ import annotations

import re

import pytest

import scilib
from scilib import codegen as g

STUB = {"prompt": 'from typing import List\n\n\ndef add_all(xs: List[int]) -> int:\n    """Sum of the list.\n    >>> add_all([1, 2])\n    3 (100% sure)\n    """\n',
        "entry_point": "add_all", "visible_tests": "assert add_all([1, 2]) == 3", "kind": "python_function_stub"}
DESC = {"prompt": "Write a function to find the largest element of a list, e.g. {1: 2}.", "entry_point": "biggest",
        "visible_tests": "assert biggest([1, 5, 2]) == 5", "kind": "task_description"}
NOTEST = {"prompt": "Write a function that returns 1.", "entry_point": "one", "visible_tests": "", "kind": "task_description"}
PROBS = [STUB, DESC, NOTEST]

BODY = "    total = 0\n    for x in xs:\n        total += x\n    return total\n"
FULL_STUB = "def add_all(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"
FULL_DESC = "def biggest(a):\n    return max(a)\n"


def run(problem, y, *args):
    """Execute the assembled program in-process (synthetic code only) and call the entry point."""
    ok, why = g.is_implemented(problem, y)
    assert ok, why
    ns: dict = {}
    exec(compile(g.assembled_program(problem, y), "<assembled>", "exec"), ns)
    return ns[problem["entry_point"]](*args)


# ------------------------------------------------------------------------------------------------ docs
def test_describe_is_factual_and_lists_the_functions():
    text = scilib.describe("codegen")
    for name in ("build_prompts", "sanitize", "repair_prompts", "select_best", "is_implemented", "assembled_program"):
        assert name in text
    assert "6 s" in text and "at most 800 characters" in text and "2000" in text
    low = text.lower()
    for bad in ("reference", "accept", "best recipe", "you should", "baseline", "success criterion", "hidden test file"):
        assert bad not in low, bad


def test_all_exports_exist():
    assert all(hasattr(g, n) for n in g.__all__)


# ------------------------------------------------------------------------------------------------ prompts
def test_build_prompts_content_and_styles():
    pr = {s: g.build_prompts(PROBS, s) for s in g.STYLES}
    for s, ps in pr.items():
        assert len(ps) == 3 and all(isinstance(p, str) for p in ps)
        assert "`add_all`" in ps[0] and "assert add_all([1, 2]) == 3" in ps[0] and "100% sure" in ps[0]   # '%' and braces pass through
        assert "{1: 2}" in ps[1] and "assert biggest([1, 5, 2]) == 5" in ps[1]
        assert "No public tests" in ps[2] and "assert" not in ps[2]
        assert all("6 s" in p for p in ps)
    assert len({tuple(v) for v in pr.values()}) == 3                       # the styles differ
    assert "7 s" in g.build_prompts([DESC], time_limit_s=7)[0]
    assert "time limit" not in g.build_prompts([DESC], time_limit_s=None)[0]


def test_build_prompts_input_errors():
    with pytest.raises(ValueError, match="unknown style"):
        g.build_prompts(PROBS, "fancy")
    with pytest.raises(ValueError, match="entry_point"):
        g.build_prompts([{"prompt": "x"}])
    with pytest.raises(TypeError, match="list of problem dicts"):
        g.build_prompts("abc")


# ------------------------------------------------------------------------------------------------ sanitize
@pytest.mark.parametrize("name,text", [
    ("fenced+demo", "Here you go:\n```python\n" + FULL_STUB + "\nif __name__ == '__main__':\n    print(add_all([1]))\nprint('x')\nassert True\n```\nDone."),
    ("unfenced", FULL_STUB),
    ("truncated_fence", "```python\n" + FULL_STUB),
    ("prose_then_def", "Sure, simple:\n\n" + FULL_STUB + "\nHope this helps."),
    ("body_only", BODY),
    ("body_dedented", "total = 0\nfor x in xs:\n    total += x\nreturn total"),
    ("body_fenced", "```python\n" + BODY + "```"),
    ("wrong_name", "```python\n" + FULL_STUB.replace("add_all", "solve") + "```"),
    ("two_blocks", "```python\nassert True\n```\nand\n```python\n" + FULL_STUB + "```\n```python\nprint(add_all([1]))\n```"),
    ("other_tag_first", "```text\n42\n```\n```python\n" + FULL_STUB + "```"),
    ("input_call", FULL_STUB + "\nn = input()\n"),
    ("future_import", "from __future__ import annotations\n" + FULL_STUB),
    ("unused_import", "import numpy as np\nimport os\n" + FULL_STUB),
])
def test_sanitize_stub_formats_give_working_programs(name, text):
    (y,) = g.sanitize([text], [STUB])
    assert run(STUB, y, [1, 2, 3]) == 6, name
    assert "numpy" not in y and "__future__" not in y and "input" not in y
    assert g.is_implemented(STUB, y) == (True, "")


@pytest.mark.parametrize("name,text", [
    ("fenced+demo", "```python\n" + FULL_DESC + "print(biggest([1]))\n```"),
    ("unfenced", FULL_DESC),
    ("truncated_fence", "```python\n" + FULL_DESC),
    ("wrong_name", "```python\ndef find_max(a):\n    return max(a)\n```"),
    ("recursive_wrong_name", "```python\ndef f(a):\n    return a[0] if len(a) == 1 else max(a[0], f(a[1:]))\n```"),
    ("prose", "Approach: use max.\n\n" + FULL_DESC + "\nThat's it."),
])
def test_sanitize_description_formats(name, text):
    (y,) = g.sanitize([text], [DESC])
    assert run(DESC, y, [3, 9, 4]) == 9, name


def test_sanitize_keeps_what_the_entry_point_needs_and_drops_the_rest():
    text = ("```python\nimport sys\nimport functools\nfrom math import gcd\nMOD = 10 ** 9 + 7\n"
            "def build():\n    return [1, 2]\nTABLE = build()\n"
            "@functools.lru_cache(maxsize=None)\ndef helper(n):\n    return n % MOD\n"
            "class Box:\n    def __init__(self, v):\n        self.v = v\n"
            "def unused():\n    return gcd(2, 4)\n"
            "def biggest(a):\n    return max(Box(helper(x)).v for x in a) + TABLE[0] - 1\n"
            "x = input()\nprint(biggest([1]))\nif __name__ == '__main__':\n    biggest([2])\n```")
    (y,) = g.sanitize([text], [DESC])
    assert run(DESC, y, [3, 9, 4]) == 9
    assert "unused" not in y and "gcd" not in y and "input" not in y and "print" not in y and "__main__" not in y
    assert "functools" in y and "MOD" in y and "def build" in y
    assert y.startswith(g.PRELUDE)


def test_sanitize_placeholders_never_break_the_hard_constraint():
    for text in (None, "", "I cannot help with that.", "```python\n```", "```python\ndef biggest(a):\n    'doc'\n```",
                 "def biggest(a:\n", "```python\ndef biggest(a):\n    for x in a:\n"):
        (y,) = g.sanitize([text], [DESC])
        assert g.is_implemented(DESC, y)[0] and y == "def biggest(*args, **kwargs):\n    pass\n"
    for text in (None, "prose only", "def add_all(xs):\n    '''doc'''\n"):
        (y,) = g.sanitize([text], [STUB])
        assert y == "    pass\n" and g.is_implemented(STUB, y)[0]


def test_sanitize_truncated_output_keeps_the_longest_valid_prefix():
    text = "```python\ndef biggest(a):\n    best = a[0]\n    for x in a:\n        if x > best:\n            best = x\n    return best\n\ndef other(b):\n    return sor"
    (y,) = g.sanitize([text], [DESC])
    assert run(DESC, y, [1, 7, 3]) == 7


def test_sanitize_assembly_rule_and_implemented():
    assert g.assembled_program(STUB, "Y") == STUB["prompt"] + "Y"
    assert g.assembled_program({**STUB, "prompt": "def f():\n    '''d'''"}, "Y") == "def f():\n    '''d'''\nY"
    assert g.assembled_program(DESC, "Y") == "Y"
    assert g.is_implemented(STUB, "    pass\n") == (True, "")
    assert g.is_implemented(STUB, "")[0] is False
    assert "syntax" in g.is_implemented(STUB, "    return (\n")[1]
    assert "no top-level function" in g.is_implemented(DESC, "x = 1")[1]
    assert "no body" in g.is_implemented(DESC, 'def biggest(a):\n    """d"""\n')[1]
    assert g.is_implemented(DESC, 'def biggest(a):\n    """d"""\n    return 1\n')[0]
    # last definition wins, as in the evaluation
    assert g.is_implemented(DESC, 'def biggest(a):\n    return 1\ndef biggest(a):\n    """d"""\n')[0] is False


def test_sanitize_prelude_supplies_common_forgotten_imports():
    text = "def biggest(a):\n    return max(Counter(a).items(), key=lambda kv: (kv[0]))[0] if math.isfinite(1.0) else 0\n"
    (y,) = g.sanitize([text], [DESC])
    assert run(DESC, y, [1, 1, 2]) == 2


def test_sanitize_index_and_base():
    base = ["B0", "B1", "B2"]
    out = g.sanitize(["```python\n" + FULL_DESC + "```"], PROBS, index=[1], base=base)
    assert out[0] == "B0" and out[2] == "B2" and run(DESC, out[1], [4, 8]) == 8
    out = g.sanitize([None], PROBS, index=[2])                             # no base: placeholders elsewhere
    assert out[0] == "    pass\n" and out[2].startswith("def one(")
    assert g.sanitize([], PROBS, index=[], base=base) == base
    with pytest.raises(ValueError, match="3 problems"):
        g.sanitize(["a"], PROBS)
    with pytest.raises(ValueError, match="outside"):
        g.sanitize(["a"], PROBS, index=[5])
    with pytest.raises(TypeError, match="llm node"):
        g.sanitize("abc", PROBS)


def test_sanitize_is_deterministic_and_fast():
    text = "```python\n" + FULL_DESC + "```\n" * 1
    assert g.sanitize([text] * 3, [DESC] * 3) == g.sanitize([text] * 3, [DESC] * 3)
    big = "x = 1\n" * 50_000
    (y,) = g.sanitize([big], [DESC])
    assert y.startswith("def biggest")


# ------------------------------------------------------------------------------------------------ repair / select
def res(passed, n_tests=1, n_passed=None, error=None, failing=()):
    return {"passed": passed, "n_tests": n_tests, "n_passed": int(passed) * n_tests if n_passed is None else n_passed,
            "error": error, "failing_tests": list(failing), "failing_idx": []}


def test_repair_prompts_target_failing_problems_only():
    codes = g.sanitize(["```python\n" + FULL_STUB + "```", "```python\ndef biggest(a):\n    return min(a)\n```", None], PROBS)
    results = {"results": [res(True), res(False, 1, 0, "AssertionError", ["assert biggest([1, 5, 2]) == 5"]),
                           res(False, 0, 0, "no visible tests")], "passed": [True, False, False], "visible_pass_rate": 1 / 3}
    rep = g.repair_prompts(PROBS, codes, results)
    assert rep["index"] == [1, 2] and len(rep["prompts"]) == 2
    p = rep["prompts"][0]
    assert "return min(a)" in p and "assert biggest([1, 5, 2]) == 5" in p and "AssertionError" in p and "0 of 1" in p
    assert "from typing import *" not in p                                # the prelude is not shown
    assert rep["prompts"][1] == g.build_prompts([NOTEST], "plan")[0]      # a placeholder is regenerated, not repaired
    assert g.repair_prompts(PROBS, codes, results["results"]) == rep        # the list of per-problem dicts works too
    assert g.repair_prompts(PROBS, codes, {"results": [res(True)] * 3}) == {"prompts": [], "index": []}
    with pytest.raises(ValueError, match="3 entries|expected 3"):
        g.repair_prompts(PROBS, codes, {"results": [res(True)]})
    with pytest.raises(ValueError, match="codes"):
        g.repair_prompts(PROBS, ["x"], results)
    with pytest.raises(TypeError, match="run_visible_tests"):
        g.repair_prompts(PROBS, codes, "nope")


def test_repair_prompt_clips_long_errors_and_shows_stub_bodies():
    long_err = "Traceback...\n" + "x" * 5000 + "\nValueError: the end"
    body_only = g.sanitize([BODY], [STUB])[0]
    rep = g.repair_prompts([STUB], [BODY], [res(False, 1, 0, long_err, ["assert add_all([1, 2]) == 3"])])
    assert "ValueError: the end" in rep["prompts"][0] and len(rep["prompts"][0]) < 3500
    assert "def add_all(xs: List[int])" in rep["prompts"][0]              # the raw body is shown inside its stub
    assert body_only.startswith(g.PRELUDE)


def test_select_best():
    a, b, c = ["a0", "a1", "a2"], ["b0", "b1", "b2"], ["c0", "c1", "c2"]
    ra = {"results": [res(True, 2, 2), res(False, 3, 1), res(False, 2, 0)]}
    rb = {"results": [res(True, 2, 2), res(False, 3, 2), res(True, 2, 2)]}
    rc = {"results": [res(False, 2, 1), res(True, 3, 3), res(False, 2, 1)]}
    assert g.select_best([a, b, c], [ra, rb, rc]) == ["a0", "c1", "b2"]     # ties -> earlier set
    out, choice = g.select_best([a, b], [ra, rb], return_choice=True)
    assert out == ["a0", "b1", "b2"] and choice == [0, 1, 1]
    zero = {"results": [res(True, 0, 0), res(False, 0, 0), res(False, 0, 0)]}    # no visible tests: compiled beats not compiled
    assert g.select_best([b, a], [zero, {"results": [res(False, 0, 0)] * 3}]) == ["b0", "b1", "b2"]
    assert g.select_best([a], [ra]) == a
    for bad in (lambda: g.select_best([], []), lambda: g.select_best([a, ["x"]], [ra, rb]), lambda: g.select_best([a, b], [ra])):
        with pytest.raises(ValueError):
            bad()


# ------------------------------------------------------------------------------------------------ real episodes
def _episode_pair():
    from scienceclaw.bench.tasks import for46_code as m
    adapter = m.Adapter()
    ok, why = adapter.available()
    if not ok:
        pytest.skip(f"FoR46 data unavailable: {why}")
    return m, adapter


@pytest.mark.parametrize("split", ["id", "ood"])
def test_canonical_solutions_in_messy_formats_pass_the_real_evaluator(split):
    m, adapter = _episode_pair()
    (ep,) = adapter.build_episodes(split, 1, seed=7)
    data = adapter._data.get()
    tools = {t.name: t.fn for t in ep.tools}
    probs = tools["load_eval_inputs"]({}, {})["problems"]
    P = [data.problems[i] for i in ep.lineage["item_ids"]]

    def full(p):
        return (p.prompt + p.canonical) if p.dataset == "HumanEval" else p.canonical

    formats = {
        "fenced+demo": lambda p: f"Solution:\n```python\n{full(p)}\nif __name__ == '__main__':\n    print({p.entry_point}(1))\nassert True\n```",
        "body_only": lambda p: p.canonical,
        "truncated_fence": lambda p: f"```python\n{full(p)}",
    }
    for name, wrap in formats.items():
        y = g.sanitize([wrap(p) for p in P], probs)
        r = ep.evaluate(y, None)
        assert all(r.h.values()), name                                    # output_format and implemented hold
        assert r.metrics["pass@1"] >= 0.9, (split, name, r.metrics["pass@1"])
    # the placeholder is a valid but failing answer
    y = g.sanitize([None] * len(P), probs)
    r = ep.evaluate(y, None)
    assert all(r.h.values()) and r.metrics["pass@1"] == 0.0
    vis = tools["run_visible_tests"]({"codes": g.sanitize([full(p) for p in P], probs)}, {})
    assert vis["visible_pass_rate"] == 1.0
    fb = g.repair_prompts(probs, y, tools["run_visible_tests"]({"codes": y}, {}))
    assert len(fb["prompts"]) == len(fb["index"]) > 0
    assert not re.search(r"canonical|hidden_test", " ".join(fb["prompts"]))
