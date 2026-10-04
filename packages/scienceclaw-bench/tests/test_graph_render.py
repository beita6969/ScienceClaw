"""render_compact / ToolSpec.signature (review F3, F4): full code with a total cap, an unambiguous cut marker and
port descriptions; summarize_value item_keys."""
from __future__ import annotations

from scienceclaw.bench.task import ToolSpec
from scienceclaw.core.graph import Edge, Node, WorkflowGraph, cut_code
from scienceclaw.core.schema import PortSchema, summarize_value


def body(n_lines: int, tag: str = "x") -> str:
    return "def run(inputs, config):\n" + "".join(f"    {tag}{i} = {i} * 2\n" for i in range(n_lines)) + "    return {'m': 1}\n"


def code_graph(*bodies: str) -> WorkflowGraph:
    g = WorkflowGraph()
    for i, b in enumerate(bodies):
        g.nodes[f"c{i}"] = Node(f"c{i}", "code", code=b, outputs={"m": PortSchema("number")})
    return g


def test_code_is_rendered_in_full_up_to_the_total_cap() -> None:
    src = body(400)                                        # ~ 8 KB, far above the old 1200-char cut
    assert 6000 < len(src) < 12000
    text = code_graph(src).render_compact()
    assert "x399 = 399 * 2" in text and "cut for display" not in text and "truncated" not in text


def test_total_cap_cuts_the_longest_bodies_first_at_line_boundaries() -> None:
    short, long1, long2 = body(10, "s"), body(900, "a"), body(900, "b")
    text = code_graph(short, long1, long2).render_compact()
    assert "s9 = 9 * 2" in text                           # the short body survives whole
    assert text.count("[cut for display:") == 2
    assert 10_000 < len(text) < 14_500                     # about 12k in total
    for line in text.splitlines():                         # every kept line is a whole source line
        if line.strip().startswith(("a", "b")) and "=" in line:
            assert line.strip().endswith("* 2"), line
    assert "the stored code is complete" in text
    # the marker counts what is not shown
    marker = next(ln for ln in text.splitlines() if "[cut for display:" in ln)
    assert "more lines (" in marker and "chars) of this node's code are not shown here" in marker


def test_per_node_cap_keeps_working_for_callers_that_pass_it() -> None:
    src = body(400)
    text = code_graph(src).render_compact(max_code_chars=900)
    assert "[cut for display:" in text and "x399" not in text and "x10 = " in text
    assert "x399" in code_graph(src).render_compact(max_code_chars=None, max_total_code_chars=None)


def test_cut_code_edges() -> None:
    assert cut_code("a\nb\n", 100) == "a\nb\n" and cut_code("a\nb", None) == "a\nb"
    one_line = "x" * 50
    out = cut_code(one_line, 20)                           # no newline to cut at: cut in the line
    assert out.startswith("x" * 19 + "\n# [cut for display: 1 more lines (31 chars)")
    out = cut_code("l1\nl2\nl3\nl4", 7)
    assert out.splitlines()[:2] == ["l1", "l2"] and "2 more lines (5 chars)" in out


def test_port_descriptions_and_config_and_prompt_markers() -> None:
    g = WorkflowGraph()
    g.nodes["t"] = Node("t", "tool", ref="load", outputs={
        "data": PortSchema("array", ("n", 3), unit="K", description="rows are\nsamples,  columns are bands")})
    g.nodes["l"] = Node("l", "llm", prompt="Q " * 1500, config={"k": "v" * 500}, inputs={"items": PortSchema("list")},
                        outputs={"outputs": PortSchema("list")})
    g.edges = []
    text = g.render_compact()
    assert "    out data: rows are samples, columns are bands" in text
    assert "in items:" not in text                         # ports without a description add no line
    assert "[config cut for display:" in text and "[prompt cut for display:" in text
    assert "the stored prompt is complete" in text


def test_tool_signature_lists_port_descriptions() -> None:
    fn = lambda i, c: {}
    plain = ToolSpec("t", "desc", {}, {"y": PortSchema("array")}, fn)
    assert plain.signature() == "tool:t() -> (y: <array>)\n    desc"
    spec = ToolSpec("t", "desc", {"x": PortSchema("table", description="input table\nwith header")},
                    {"y": PortSchema("array", description="per-row score"), "z": PortSchema("number")}, fn, config_doc="k: int")
    sig = spec.signature()
    assert sig.startswith("tool:t(x: <table>) -> (y: <array>, z: <number>)\n    desc  config: k: int")
    assert "\n    in x: input table with header" in sig and "\n    out y: per-row score" in sig and "out z" not in sig


def test_summarize_value_lists_item_keys_for_records() -> None:
    recs = [{"id": 1, "text": "a"}, {"id": 2, "text": "b", "label": 0}]
    s = summarize_value(recs)
    assert s["len"] == 2 and s["item_keys"] == ["id", "text", "label"]
    assert "item_keys" not in summarize_value([1, 2, 3]) and "item_keys" not in summarize_value(["a", "b"])
    many = [{f"k{i}": i for i in range(40)}]
    assert len(summarize_value(many)["item_keys"]) == 20
