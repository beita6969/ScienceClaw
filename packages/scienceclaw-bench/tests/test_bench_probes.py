"""Generic hidden-probe API (LEAK-1): ``Episode.run_probes`` re-runs a graph on adapter-derived episodes."""
from __future__ import annotations

import numpy as np

from scienceclaw.bench.task import Budget, ConstraintSpec, Episode, EvalResult, Probe, ToolSpec
from scienceclaw.core.schema import PortSchema
from scienceclaw.core.trace import Trace


def _episode(with_probe: bool = True) -> Episode:
    def load(inputs, config):
        return {"x": [1, 2, 3, 4]}

    tool = ToolSpec("load", "toy", {}, {"x": PortSchema("list", (4,), None, "int", "")}, load)
    probe_ok = ConstraintSpec("prefix_ok", "hidden", lambda y, tr: (
        ((tr.probes.get("p", {}).get("ok", False), tr.probes.get("p", {}).get("msg", ""))) if tr and tr.probes
        else (True, "not probed")), visible=False)

    def evaluate(y, trace):
        return EvalResult(primary=1.0, accepted=True, details={"pooled_payload": {"k": 1}})

    def probes(y):
        def load_short(inputs, config):
            return {"x": [1, 2]}
        short = ToolSpec("load", "toy", {}, dict(tool.outputs), load_short)
        return [Probe("p", {"tools": [short], "constraints": []},
                      lambda y2: (list(y2) == list(y)[:2], f"got {list(y2)}"))]

    return Episode(id="toy-0", discipline="TOY", family="toy", split="id", task_type="t", objective="o",
                   required_output=PortSchema("list", (4,), None, "int", ""), tools=[tool], constraints=[probe_ok],
                   budget=Budget(), metric="m", direction="max", n_items=4, _evaluate=evaluate,
                   _probes=probes if with_probe else None)


def test_run_probes_records_verdicts_and_feeds_hidden_constraints():
    ep = _episode()
    y = [1, 2, 3, 4]
    seen = []

    def causal(derived, name):                          # a graph that only looks at rows so far: prefix of the full run
        seen.append((derived.id, name, derived.tool("load").fn({}, {})["x"], derived._evaluate, derived._probes))
        return derived.tool("load").fn({}, {})["x"], Trace()

    tr = Trace()
    out = ep.run_probes(y, tr, causal)
    assert out == {"p": {"ok": True, "msg": "got [1, 2]"}}
    assert tr.probes == out
    assert seen == [("toy-0#p", "p", [1, 2], None, None)]           # derived episode: cut inputs, no evaluator, no probes
    res = ep.evaluate(y, tr)
    assert res.h["prefix_ok"] and res.z == 1

    def look_ahead(derived, name):                      # a graph whose output depends on rows it should not see
        return [9, 9], Trace()

    tr2 = Trace()
    ep.run_probes(y, tr2, look_ahead)
    assert not tr2.probes["p"]["ok"]
    res2 = ep.evaluate(y, tr2)
    assert not res2.h["prefix_ok"] and res2.z == 0 and res2.accepted


def test_run_probes_failures_fail_the_probe():
    ep = _episode()
    y = [1, 2, 3, 4]

    def boom(derived, name):
        raise RuntimeError("node crashed")

    out = ep.run_probes(y, Trace(), boom)
    assert not out["p"]["ok"] and "RuntimeError" in out["p"]["msg"]
    out = ep.run_probes(y, Trace(), lambda d, n: (None, Trace()))
    assert not out["p"]["ok"] and "no output" in out["p"]["msg"]


def test_no_probes_means_not_probed_and_unaffected():
    ep = _episode(with_probe=False)
    tr = Trace()
    assert ep.run_probes([1, 2, 3, 4], tr, lambda d, n: (_ for _ in ()).throw(AssertionError("must not run"))) == {}
    assert tr.probes == {}
    assert ep.probe_specs(None) == [] and _episode().probe_specs(None) == []
    res = ep.evaluate([1, 2, 3, 4], tr)
    assert res.h["prefix_ok"] and res.z == 1                        # never probed: the hidden check passes
    assert ep.run_probes([1, 2, 3, 4], None, lambda d, n: ([1, 2], Trace())) == {}


def test_trace_probes_roundtrip_and_backward_compat():
    tr = Trace(wall_s=1.5)
    assert "probes" not in tr.to_dict()                             # unchanged serialisation when no probe ran
    tr.probes = {"p": {"ok": False, "msg": "x"}}
    back = Trace.from_dict(tr.to_dict())
    assert back.probes == tr.probes and back.wall_s == 1.5
    legacy = tr.to_dict()
    legacy.pop("probes")
    assert Trace.from_dict(legacy).probes == {}
    assert np.isclose(Trace.from_dict(legacy).wall_s, 1.5)
