"""Runtime hardening (review findings RT-2/4/5/6/7/8/11): transient taint, llm template / cost guards, number
parsing, node deadlines, the tool lock, summary guards and contract type / unit checks."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_runtime_executor import (  # noqa: E402
    ARR, Counter, StubLLM, code, graph, make_episode, make_scale_op, submit, tool,
)

import scienceclaw.runtime.executor as executor_mod  # noqa: E402
from scienceclaw.bench.task import ToolSpec  # noqa: E402
from scienceclaw.core.graph import Edge, Node, WorkflowGraph  # noqa: E402
from scienceclaw.core.operators import Contract, OperatorSpec, check_contract_entries  # noqa: E402
from scienceclaw.core.program import AgentProgram  # noqa: E402
from scienceclaw.core.schema import PortSchema, summarize_value, value_has_type  # noqa: E402
from scienceclaw.runtime.executor import (  # noqa: E402
    Executor, format_summary, parse_llm_text, render_template, template_fields,
)
from scienceclaw.runtime.values import load_value  # noqa: E402

ITEMS3 = "def run(inputs, config):\n    return {'items': ['a x', 'b y', 'c z']}\n"


def llm_graph(prompt: str, config: dict | None = None, items_src: str = ITEMS3, extra: list[Node] | None = None,
              edges: list[tuple] | None = None) -> WorkflowGraph:
    mk = code("mk", items_src, {}, {"items": {"type": "list"}})
    ln = Node("l", "llm", prompt=prompt, config=dict(config or {}), inputs={"items": PortSchema("list"), **{
        n.id: PortSchema("any") for n in (extra or [])}}, outputs={"outputs": PortSchema("list")})
    return graph([mk, ln, *(extra or [])], [("mk", "items", "l", "items"), *(edges or [])])


@pytest.fixture()
def ep():
    return make_episode(Counter(), Counter())


# ----------------------------------------------------------------------------- RT-2
class FlakyLLM:
    """Item 1 fails in the first call round, every item succeeds afterwards."""

    def __init__(self) -> None:
        self.rounds = 0

    def chat_many(self, role, batch, **kw):
        self.rounds += 1
        out = []
        for i, m in enumerate(batch):
            if self.rounds == 1 and i == 1:
                out.append(SimpleNamespace(text="", error="HTTP 500", usage={"calls": 1}))
            else:
                out.append(SimpleNamespace(text="4", error=None, usage={"calls": 1}))
        return out


def test_descendants_of_a_transient_node_are_not_cached(tmp_path) -> None:
    ep = make_episode(Counter(), Counter(), required=PortSchema("list"))
    count_ok = ("def run(inputs, config):\n    return {'m': [sum(v is not None for v in inputs['x'])]}\n")
    ln = Node("l", "llm", prompt="Value of {item}", config={"parse": "number"}, inputs={"items": PortSchema("list")},
              outputs={"outputs": PortSchema("list")})
    g = graph([code("mk", ITEMS3, {}, {"items": {"type": "list"}}), ln,
               code("c", count_ok, {"x": {"type": "list"}}, {"m": {"type": "list"}}), submit(ep)],
              [("mk", "items", "l", "items"), ("l", "outputs", "c", "x"), ("c", "m", "s", "y")])
    llm = FlakyLLM()
    ex = Executor(ep, AgentProgram(), llm, tmp_path)
    cp, records, y = ex.execute(g, ex.new_checkpoint())
    assert records["l"].status == "ok" and records["l"].outputs_summary["outputs"]["llm_errors"] == 1
    assert y == [2]
    fps = g.fingerprints()
    assert fps["mk"] in cp.records
    assert all(fps[n] not in cp.records for n in ("l", "c", "s"))          # the whole tainted cone is left out
    cp, records, y = ex.execute(g, cp)                                     # the LLM recovered: consumers see the new values
    assert llm.rounds == 2 and y == [3]
    assert records["mk"].cached and not records["c"].cached and not records["s"].cached
    assert all(fps[n] in cp.records for n in ("l", "c", "s"))              # now complete, hence cached
    _, records, _ = ex.execute(g, cp)
    assert all(r.cached for r in records.values()) and llm.rounds == 2


# ----------------------------------------------------------------------------- RT-4
def test_template_fields_and_accessors() -> None:
    assert template_fields("Rate {text} in {unit}; raw={item} {text}") == (["text", "unit", "item"], [])
    assert template_fields('Return {"label": ...} for {{literal}} {v:.2f}') == (["v"], [])
    names, acc = template_fields("Q: {item[0]} {doc.text} {x}")
    assert names == ["item", "doc", "x"] and acc == ["{item[0]}", "{doc.text}"]


def test_llm_prompt_that_ignores_the_item_is_rejected_before_any_call(ep, tmp_path) -> None:
    llm = StubLLM(lambda t: "x")
    ex = Executor(ep, AgentProgram(), llm, tmp_path)
    _, records, _ = ex.execute(llm_graph("Summarize the dataset."), ex.new_checkpoint())
    assert records["l"].status == "error" and "does not use the item" in records["l"].error
    assert "all 3 requests would be identical" in records["l"].error and not llm.calls
    # plain-string items and a field placeholder: the unresolved name is reported and the item hint is given
    _, records, _ = ex.execute(llm_graph("Sentence: {sentence}"), None)
    assert records["l"].status == "error" and "unresolved placeholders: ['sentence']" in records["l"].error
    assert "use {item}" in records["l"].error and not llm.calls
    # dict items: the available keys are listed
    dict_items = "def run(inputs, config):\n    return {'items': [{'doc_id': 1, 'text': 'a'}, {'doc_id': 2, 'text': 'b'}]}\n"
    _, records, _ = ex.execute(llm_graph("Summarize.", items_src=dict_items), None)
    assert "['doc_id', 'text']" in records["l"].error and not llm.calls
    # a single item may legitimately use only the shared ports
    single = "def run(inputs, config):\n    return {'items': ['only']}\n"
    _, records, _ = ex.execute(llm_graph("Say hello.", items_src=single), None)
    assert records["l"].status == "ok" and len(llm.calls) == 1


def test_llm_prompt_with_index_access_is_rejected(ep, tmp_path) -> None:
    llm = StubLLM(lambda t: "x")
    ex = Executor(ep, AgentProgram(), llm, tmp_path)
    _, records, _ = ex.execute(llm_graph("First char: {item[0]}"), ex.new_checkpoint())
    assert records["l"].status == "error" and "attribute / index access" in records["l"].error and not llm.calls


def test_unresolved_placeholders_are_reported_in_the_summary(ep, tmp_path) -> None:
    llm = StubLLM(lambda t: "ok")
    ex = Executor(ep, AgentProgram(), llm, tmp_path)
    _, records, _ = ex.execute(llm_graph("Classify {item} using {rubric}"), ex.new_checkpoint())
    rec = records["l"]
    assert rec.status == "ok"
    assert rec.outputs_summary["outputs"]["unresolved_placeholders"] == ["rubric"]
    assert "unresolved_placeholders=[\"rubric\"]" in format_summary(rec.outputs_summary["outputs"])
    assert llm.calls[0]["batch"][0][1]["content"] == "Classify a x using {rubric}"


def test_parse_failure_examples_are_reported(ep, tmp_path) -> None:
    llm = StubLLM(lambda t: "I cannot tell from this text")
    ex = Executor(ep, AgentProgram(), llm, tmp_path)
    _, records, _ = ex.execute(llm_graph("Score {item}", {"parse": "number"}), ex.new_checkpoint())
    s = records["l"].outputs_summary["outputs"]
    assert s["parse_failures"] == 3 and s["parse_failure_examples"] == ["I cannot tell from this text"] * 3
    assert "parse_failure_examples" in format_summary(s)


# ----------------------------------------------------------------------------- RT-5
@pytest.mark.parametrize("text,expected", [
    ("3", 3.0), ("-3.5", -3.5), ("−3.5", -3.5), ("1,234.5", 1234.5), ("1,234,567", 1234567.0),
    ("The answer is 4", 4.0), ("1. The answer is 4", 4.0), ("Answer: 3.0.", 3.0), ("7.5, i.e. 7.5", 7.5),
    ("the value is -3.5e-2 units", -0.035), ("1e3", 1000.0), (".5", 0.5),
    ("80%", None), ("about 80 percent or 80%", None), ("between 3 and 4", None), ("3 (out of 5)", None),
    ("nan", None), ("inf", None), ("no digits", None), ("", None),
])
def test_number_parsing(text: str, expected) -> None:
    assert parse_llm_text(text, "number") == (pytest.approx(expected) if expected is not None else None)


# ----------------------------------------------------------------------------- RT-6
def test_timed_out_tool_is_not_cached_and_keeps_the_tool_lock(tmp_path) -> None:
    ep = make_episode(Counter(), Counter(), max_node_s=0.3)
    release, started, calls = threading.Event(), threading.Event(), Counter()

    def blocking(inputs, config):
        calls.n += 1
        started.set()
        release.wait(10)
        return {"v": 1.0}

    ep.tools.append(ToolSpec("block", "blocks until released", {}, {"v": PortSchema("number")}, blocking))
    ex = Executor(ep, AgentProgram(), None, tmp_path)
    g = graph([tool("w", ep, "block")], [])
    cp, records, _ = ex.execute(g, ex.new_checkpoint())
    assert records["w"].status == "error" and "timeout" in records["w"].error and calls.n == 1
    assert g.fingerprints()["w"] not in cp.records                          # load-dependent: not cached
    # the abandoned call still runs: a new tool call is refused at once instead of running concurrently
    t0 = time.monotonic()
    cp, records, _ = ex.execute(g, cp)
    assert time.monotonic() - t0 < 0.25 and calls.n == 1
    assert "still running" in records["w"].error and g.fingerprints()["w"] not in cp.records
    release.set()
    for th in threading.enumerate():
        if th.name == "tool-block":
            th.join(5)
    cp, records, _ = ex.execute(g, cp)                                      # lock free again: the tool runs and succeeds
    assert records["w"].status == "ok" and calls.n == 2


def test_repeated_timeout_of_a_code_node_is_cached(tmp_path) -> None:
    ep = make_episode(Counter(), Counter(), max_node_s=0.5)
    ex = Executor(ep, AgentProgram(), None, tmp_path)
    sleepy = "import time\ndef run(inputs, config):\n    time.sleep(30)\n    return {'m': 1}\n"
    g = graph([code("z", sleepy, {}, {"m": ARR})], [])
    cp, records, _ = ex.execute(g, ex.new_checkpoint())
    assert records["z"].status == "error" and "timeout" in records["z"].error
    fp = g.fingerprints()["z"]
    assert fp not in cp.records                                             # first timeout: retried next time
    cp, records, _ = ex.execute(g, cp)
    assert not records["z"].cached and records["z"].status == "error"
    assert fp in cp.records                                                 # second timeout: cached like any error
    _, records, _ = ex.execute(g, cp)
    assert records["z"].cached


class SlowLLM:
    def __init__(self, delays: list[float]) -> None:
        self.delays, self.calls = list(delays), 0

    def chat_many(self, role, batch, **kw):
        d = self.delays[min(self.calls, len(self.delays) - 1)]
        self.calls += 1
        time.sleep(d)
        return [SimpleNamespace(text="1", error=None, usage={"calls": 1}) for _ in batch]


def test_llm_node_deadline_keeps_finished_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(executor_mod, "LLM_CHUNK_ITEMS", 2)
    ep = make_episode(Counter(), Counter(), max_node_s=1.0)
    six = "def run(inputs, config):\n    return {'items': list('abcdef')}\n"
    llm = SlowLLM([0.4, 5.0])
    ex = Executor(ep, AgentProgram(), llm, tmp_path)
    t0 = time.monotonic()
    cp, records, _ = ex.execute(llm_graph("Score {item}", {"parse": "number"}, six), ex.new_checkpoint())
    assert time.monotonic() - t0 < 3.0                                      # the hanging chunk was abandoned
    rec = records["l"]
    assert rec.status == "ok"
    assert load_value(rec.output_refs["outputs"]) == [1.0, 1.0, None, None, None, None]
    assert rec.outputs_summary["outputs"]["llm_timed_out_items"] == 4
    assert "llm_timed_out_items=4" in format_summary(rec.outputs_summary["outputs"])
    assert g_fp(cp, "l") is False                                           # partial result: transient, not cached


def g_fp(cp, nid_prefix: str) -> bool:
    """True iff some cached record belongs to node ``nid_prefix``."""
    return any(r.node_id == nid_prefix for r in cp.records.values())


def test_llm_node_where_nothing_finishes_is_a_timeout_error(tmp_path) -> None:
    ep = make_episode(Counter(), Counter(), max_node_s=0.3)
    llm = SlowLLM([3.0])
    ex = Executor(ep, AgentProgram(), llm, tmp_path)
    t0 = time.monotonic()
    cp, records, _ = ex.execute(llm_graph("Score {item}", {"timeout_s": 0.2}), ex.new_checkpoint())
    assert time.monotonic() - t0 < 2.0
    assert records["l"].status == "error" and records["l"].error.startswith("timeout: no llm item finished within 0.2")
    assert g_fp(cp, "l") is False
    cp, records, _ = ex.execute(llm_graph("Score {item}", {"timeout_s": 0.2}), cp)      # second timeout: cached
    assert g_fp(cp, "l") is True and llm.calls == 2


def test_operator_node_has_an_overall_deadline(tmp_path) -> None:
    ep = make_episode(Counter(), Counter(), max_node_s=0.5)

    def nap(inputs, config):
        time.sleep(0.4)
        return {"x": inputs["x"]}

    port = {"x": PortSchema("any")}
    ep.tools.append(ToolSpec("nap", "sleeps 0.4 s", port, port, nap))
    body_nodes = {n: Node(n, "tool", ref="nap", inputs=dict(port), outputs=dict(port)) for n in "abcd"}
    body = WorkflowGraph(body_nodes, [Edge.make("a", "x", "b", "x"), Edge.make("b", "x", "c", "x"),
                                       Edge.make("c", "x", "d", "x")])
    op = OperatorSpec("napper", 1, "napper", "four naps", body, {"x": PortSchema("array")}, {"y": PortSchema("array")},
                      {"x": [("a", "x")]}, {"y": ("d", "x")}, Contract())
    ex = Executor(ep, AgentProgram(operators={"napper": op}), None, tmp_path)
    src = "import numpy as np\ndef run(inputs, config):\n    return {'m': np.ones(3)}\n"
    g = WorkflowGraph()
    g.nodes["f"] = code("f", src, {}, {"m": ARR})
    g.nodes["o"] = Node("o", "operator", ref="op:napper", inputs={"x": PortSchema("array")}, outputs={"y": PortSchema("array")})
    g.edges = [Edge.make("f", "m", "o", "x")]
    t0 = time.monotonic()
    cp, records, _ = ex.execute(g, ex.new_checkpoint())
    assert time.monotonic() - t0 < 5.0
    assert records["o"].status == "error" and "failed inside its body" in records["o"].error
    assert "timeout" in records["o"].error                                   # 4 x 0.4 s > 2 x max_node_s = 1.0 s
    assert g.fingerprints()["o"] not in cp.records                           # first timeout of the operator: not cached


# ----------------------------------------------------------------------------- RT-8
def test_summary_failures_do_not_discard_a_valid_node(tmp_path) -> None:
    ep = make_episode(Counter(), Counter(), required=PortSchema("list"))
    huge = "def run(inputs, config):\n    return {'m': [10 ** 400, 1]}\n"     # float conversion overflows in summarize_value
    with pytest.raises(OverflowError):
        summarize_value([10 ** 400, 1])
    ex = Executor(ep, AgentProgram(), None, tmp_path)
    g = graph([code("c", huge, {}, {"m": {"type": "list"}}), submit(ep)], [("c", "m", "s", "y")])
    cp, records, y = ex.execute(g, ex.new_checkpoint())
    assert records["c"].status == "ok" and y == [10 ** 400, 1]
    assert "summary_error" in records["c"].outputs_summary["m"]
    assert "summary_error=OverflowError" in format_summary(records["c"].outputs_summary["m"])
    fb = ex._feedback(0, SimpleNamespace(type="finish", describe=lambda: "finish"), g, records, y, None, [], True, 0.0)
    assert fb.y_summary["summary_error"].startswith("OverflowError")
    assert "summary_error" in fb.render()


# ----------------------------------------------------------------------------- RT-11
def test_oversized_llm_prompt_is_rejected(ep, tmp_path) -> None:
    big = code("big", "def run(inputs, config):\n    return {'ctx': 'x' * 40000}\n", {}, {"ctx": {"type": "text"}})
    g = llm_graph("Label {item} given {ctx}", extra=[big], edges=[("big", "ctx", "l", "big")])
    # the shared port is called after its node id in this helper; rename the placeholder accordingly
    g.nodes["l"].prompt = "Label {item} given {big}"
    g.edges = [Edge.make("mk", "items", "l", "items"), Edge.make("big", "ctx", "l", "big")]
    assert g.validate() == []
    llm = StubLLM(lambda t: "x")
    ex = Executor(ep, AgentProgram(), llm, tmp_path)
    _, records, _ = ex.execute(g, ex.new_checkpoint())
    assert records["l"].status == "error" and "above the limit of 32000" in records["l"].error and not llm.calls
    ex.max_llm_prompt_chars = 100_000                                        # a task may raise the limit
    _, records, _ = ex.execute(g, ex.new_checkpoint())
    assert records["l"].status == "ok" and len(llm.calls) == 1


def test_llm_max_tokens_is_clamped_to_the_executor_role(ep, tmp_path) -> None:
    llm = StubLLM(lambda t: "x")
    llm.cfg = SimpleNamespace(executor=SimpleNamespace(max_tokens=100))
    ex = Executor(ep, AgentProgram(), llm, tmp_path)
    cp, records, _ = ex.execute(llm_graph("Echo {item}", {"max_tokens": 100_000}), ex.new_checkpoint())
    assert llm.calls[0]["kw"]["max_tokens"] == 100
    assert records["l"].outputs_summary["outputs"]["max_tokens_clamped_to"] == 100
    llm2 = StubLLM(lambda t: "x")                                            # no role config: the default executor limit
    _, records, _ = Executor(ep, AgentProgram(), llm2, tmp_path / "b").execute(
        llm_graph("Echo {item}", {"max_tokens": 50}), None)
    assert llm2.calls[0]["kw"]["max_tokens"] == 50 and "max_tokens_clamped_to" not in records["l"].outputs_summary["outputs"]
    llm3 = StubLLM(lambda t: "x")
    _, records, _ = Executor(ep, AgentProgram(), llm3, tmp_path / "c").execute(
        llm_graph("Echo {item}", {"max_tokens": 9999}), None)
    assert llm3.calls[0]["kw"]["max_tokens"] == 2000
    llm4 = StubLLM(lambda t: "x")
    _, records, _ = Executor(ep, AgentProgram(), llm4, tmp_path / "d").execute(
        llm_graph("Echo {item}", {"max_tokens": "lots"}), None)
    assert records["l"].status == "error" and "max_tokens must be a positive integer" in records["l"].error and not llm4.calls


# ----------------------------------------------------------------------------- RT-7
def test_contract_type_check_looks_at_the_value() -> None:
    entries = [{"port": "x", "check": "type", "value": "number"}]
    schemas = {"x": PortSchema("number")}
    assert check_contract_entries(entries, {"x": 3.0}, schemas) == []
    assert check_contract_entries(entries, {"x": np.float64(2)}, schemas) == []
    assert check_contract_entries(entries, {"x": "3"}, schemas) == ["x: value of type str is not a number"]
    assert check_contract_entries([{"port": "x", "check": "type", "value": "table"}], {"x": [1]}, schemas)
    assert check_contract_entries([{"port": "x", "check": "type", "value": "any"}], {"x": "3"}, schemas) == []
    assert value_has_type(np.arange(3), "array") and not value_has_type(True, "number")


def test_contract_unit_check_uses_the_delivered_unit() -> None:
    entries = [{"port": "x", "check": "unit", "value": "K"}]
    schemas = {"x": PortSchema("array")}                       # the operator declares no unit for x
    assert check_contract_entries(entries, {"x": 1}, schemas) == ["x: unit None violates K"]      # legacy: declared unit only
    assert check_contract_entries(entries, {"x": 1}, schemas, units={"x": "K"}) == []
    assert check_contract_entries(entries, {"x": 1}, schemas, units={"x": None}) == []            # unspecified upstream
    assert check_contract_entries(entries, {"x": 1}, schemas, units={"x": "degC"}) == ["x: unit degC violates K"]


def test_operator_node_checks_type_and_unit_of_what_is_delivered(ep, tmp_path) -> None:
    op = make_scale_op("typed", contract=Contract(pre=[{"port": "x", "check": "type", "value": "number"},
                                                       {"port": "x", "check": "unit", "value": "K"}]))
    text_src = "def run(inputs, config):\n    return {'m': '3'}\n"

    def run(unit: str | None, out_type: str = "any") -> list[str]:
        g = WorkflowGraph()
        g.nodes["f"] = code("f", text_src, {}, {"m": {"type": out_type, **({"unit": unit} if unit else {})}})
        g.nodes["o"] = Node("o", "operator", ref="op:typed", inputs={"x": PortSchema("any")},
                            outputs={"y": PortSchema("array")})
        g.edges = [Edge.make("f", "m", "o", "x")]
        ex = Executor(ep, AgentProgram(operators={"typed": op}), None, tmp_path / f"{unit}")
        _, records, _ = ex.execute(g, None)
        return records["o"].contract_violations

    viol = run("K")
    assert "pre: x: value of type str is not a number" in viol and not any("unit" in v for v in viol)
    assert "pre: x: unit degC violates K" in run("degC")
    assert not any("unit" in v for v in run(None))
