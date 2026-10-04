"""Review-fix tests for the policy prompts (F4, F5, F7, F8, F9, F10, F11) and the compact history (M1)."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_agent_support import ACCEPTANCE_TEXT, make_episode  # noqa: E402

from scienceclaw.agent import prompts  # noqa: E402
from scienceclaw.agent.prompts import (  # noqa: E402
    ORCHESTRATIONS, available_packages, build_step_message, build_system_prompt, find_strategy_phrases,
    fixed_workflow_tools, summarize_feedback,
)
from scienceclaw.agent.solver import FIXED_CODE_NODE, build_fixed_workflow, scrub_volatile  # noqa: E402
from scienceclaw.bench.task import ToolSpec  # noqa: E402
from scienceclaw.core.graph import WorkflowGraph  # noqa: E402
from scienceclaw.core.schema import PortSchema, summarize_value  # noqa: E402
from scienceclaw.core.trace import NodeRecord  # noqa: E402
from scienceclaw.runtime.executor import Feedback  # noqa: E402


def _section(prompt: str, title: str) -> str:
    m = re.search(rf"^# {re.escape(title)}\n(.*?)(?=^# |\Z)", prompt, re.S | re.M)
    assert m, f"section {title!r} missing"
    return m.group(1)


def _episode_with_input_tool():
    ep = make_episode()
    ep.tools.append(ToolSpec(name="needs_input", description="requires an input", inputs={"z": PortSchema(type="list")},
                             outputs={"w": PortSchema(type="list")}, fn=lambda inputs, config: {"w": inputs["z"]}))
    return ep


# ------------------------------------------------------------------------------------------- F5
@pytest.mark.parametrize("orchestration", ORCHESTRATIONS)
def test_acceptance_section_is_generic_and_uniform(orchestration: str) -> None:
    a = build_system_prompt(make_episode(), [], [], orchestration, fixed_code_node="code")
    other = make_episode("ep2", objective="Classify the sample as one of the labels; report probabilities.")
    other.discipline = "FoR34"
    b = build_system_prompt(other, [], [], orchestration, fixed_code_node="code")
    sec = _section(a, "Acceptance")
    assert sec == _section(b, "Acceptance"), "the acceptance text must not depend on the task"
    low = sec.lower()
    assert "reference predictor" in low and "margin" in low and "hidden evaluation items" in low
    assert "constraints" in low and "replayed" in low and "budget" in low
    for banned in ("seasonal naive", "climatology", "success criterion", "recipe", ACCEPTANCE_TEXT):
        assert banned.lower() not in low
    assert find_strategy_phrases(sec) == []


def test_acceptance_never_names_metric_or_reference_values() -> None:
    p = build_system_prompt(make_episode(), [], [], "canvas")
    sec = _section(p, "Acceptance")
    assert "secret_metric_zq" not in sec and "0.9876543" not in sec and "0.5" not in sec
    assert "not shown to you" in sec


# ------------------------------------------------------------------------------------------ F11
def test_finish_text_matches_the_solver_choice_of_deliverable() -> None:
    p = build_system_prompt(make_episode(), [], [], "canvas")
    flat = " ".join(p.split())
    assert "is not always the current canvas" in flat
    assert "latest step at which the value at the submit node's input \"y\" was computed" in flat
    assert "passed the run's hidden verification" in flat
    assert "best development score on visible data (ties: the later step)" in flat
    assert "An episode without any such step counts as failed" in flat
    # the text names neither the hidden score nor the acceptance rule and is the same for every orchestration
    for orch in ("single_turn", "single_operator", "fixed_workflow"):
        q = " ".join(build_system_prompt(make_episode(), [], [], orch, fixed_code_node="code").split())
        assert prompts._FINISH in q


# ---------------------------------------------------------------------------------------- F7/F8
@pytest.mark.parametrize("orchestration", ["single_operator", "fixed_workflow"])
def test_restricted_orchestrations_do_not_list_unusable_components(orchestration: str) -> None:
    p = build_system_prompt(make_episode(), [], [], orchestration, fixed_code_node="core")
    for absent in ("- operator:", "- llm:", "Operators from your library", "op:<id>", '"kind": "llm"', '"prompt"',
                   "llm node items per run", "operator contract violations"):
        assert absent not in p, absent
    assert "- code:" in p and "- tool:" in p and "- submit:" in p


@pytest.mark.parametrize("orchestration", ["canvas", "single_turn"])
def test_full_orchestrations_list_operators_and_llm_nodes(orchestration: str) -> None:
    p = build_system_prompt(make_episode(), [], [], orchestration)
    for present in ("- operator:", "- llm:", "Operators from your library", "op:<id>", "llm node items per run",
                    "operator contract violations"):
        assert present in p, present


def test_operator_config_is_not_advertised() -> None:
    p = build_system_prompt(make_episode(), [], [], "canvas")
    op_line = next(ln for ln in p.splitlines() if ln.startswith("- operator:"))
    assert "take no config" in op_line
    assert '"config"' not in op_line
    assert "operator config" not in p.lower() and "η" not in p


def test_fixed_workflow_prompt_restricts_actions_and_wired_tools() -> None:
    ep = _episode_with_input_tool()
    p = build_system_prompt(ep, [], [], "fixed_workflow", fixed_code_node="core")
    tools = _section(p, "Tools")
    assert "load_x" in tools and "needs_input" not in tools
    assert [t.name for t in fixed_workflow_tools(ep)] == ["load_x"]
    g, code_id = build_fixed_workflow(ep)
    assert code_id == FIXED_CODE_NODE
    assert {n.ref for n in g.nodes.values() if n.kind == "tool"} == {"load_x"}
    assert 'modify_node on node "core"' in p
    assert '"type": "add_node"' not in p and '"type": "add_edge"' not in p and '"wire"' not in p
    assert '"code" | "code_edit" | "config"' in p
    canvas = build_system_prompt(ep, [], [], "canvas")
    assert "needs_input" in _section(canvas, "Tools")


def test_single_operator_prompt_keeps_its_component_limit() -> None:
    p = build_system_prompt(make_episode(), [], [], "single_operator")
    assert "at most one code node" in p
    assert '"code" | "code_edit" | "config" | "inputs" | "outputs" | "wire"' in p


# ------------------------------------------------------------------------------------------- F4
@pytest.mark.parametrize("orchestration", ORCHESTRATIONS)
def test_code_edit_is_documented_as_an_interface_rule(orchestration: str) -> None:
    p = build_system_prompt(make_episode(), [], [], orchestration, fixed_code_node="code")
    assert '"code_edit" is {"find": "<text>", "replace": "<text>"}' in p
    flat = " ".join(p.split())
    assert "exact occurrence" in flat and "zero or several times" in flat
    assert "cannot be combined with \"code\"" in flat
    assert find_strategy_phrases(p) == []


# ------------------------------------------------------------------------------------------ F10
def test_prompt_states_python_values_of_every_port_type() -> None:
    p = build_system_prompt(make_episode(), [], [], "canvas")
    line = next(ln for ln in p.splitlines() if ln.startswith("Python values:"))
    for needle in ("number carries an int or float", "text a str", "array a numpy.ndarray", "table a pandas.DataFrame",
                   "list a list", "dict a dict", "series a pandas.Series", "any carries any picklable object"):
        assert needle in line, needle
    assert "checked against the deliverable schema" in line


def _packages_sentence(prompt: str) -> str:
    m = re.search(r"; (the Python .*? available)\.", prompt)
    assert m, "package sentence missing"
    return m.group(1)


def test_package_list_is_probed_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    hidden = {"scipy", "statsmodels", "torch", "lightgbm", "xgboost"}

    def fake(name: str, *a, **kw):
        return None if name.split(".")[0] in hidden else real(name, *a, **kw)

    available_packages.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", fake)
    try:
        pk = available_packages()
        assert not hidden & set(pk) and "numpy" in pk
        sentence = _packages_sentence(build_system_prompt(make_episode(), [], [], "canvas"))
        assert sentence.startswith("the Python packages numpy") and sentence.endswith("and the standard library are available")
        assert "scipy" not in sentence and "statsmodels" not in sentence
        assert all(name in sentence for name in pk)
    finally:
        monkeypatch.undo()
        available_packages.cache_clear()
    assert set(available_packages()) >= {"numpy"}          # the real environment again


def test_package_list_without_any_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    available_packages.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **kw: None)
    try:
        assert available_packages() == ()
        sentence = _packages_sentence(build_system_prompt(make_episode(), [], [], "canvas"))
        assert sentence == "the Python standard library is available"
    finally:
        monkeypatch.undo()
        available_packages.cache_clear()


def test_package_list_matches_installed_packages() -> None:
    available_packages.cache_clear()
    sentence = _packages_sentence(build_system_prompt(make_episode(), [], [], "canvas"))
    for shown in available_packages():
        assert shown in sentence
    for module, shown in prompts._PACKAGE_CANDIDATES:
        if importlib.util.find_spec(module) is None:
            assert shown not in sentence


# ----------------------------------------------------------------------- dev score visibility
@pytest.mark.parametrize("orchestration", ORCHESTRATIONS)
def test_prompt_mentions_the_dev_score_only_when_it_is_shown(orchestration: str) -> None:
    shown = build_system_prompt(make_episode(), [], [], orchestration, fixed_code_node="code", show_dev_score=True)
    hidden = build_system_prompt(make_episode(), [], [], orchestration, fixed_code_node="code", show_dev_score=False)
    assert "development score on visible data when the task provides one" in _section(shown, "Feedback")
    assert "development score on visible data when the task provides one" not in _section(hidden, "Feedback")
    assert "the development score, the failed constraints" in _section(shown, "Feedback")
    assert "the development score, the failed constraints" not in _section(hidden, "Feedback")
    # the sentence stays grammatical: one final ", and " and it ends with a period
    for p in (shown, hidden):
        fb = _section(p, "Feedback").splitlines()[0]
        assert fb.count(", and ") == 1 and "After each action you receive: " in fb
        assert not re.search(r",\s*[,.]", fb) and not fb.endswith(",")
    # the best-dev clause of the finish text is the only place that speaks of visible scores in both variants
    assert "best development score on visible data" in hidden


@pytest.mark.parametrize("show_dev", [True, False])
@pytest.mark.parametrize("orchestration", ORCHESTRATIONS)
def test_every_prompt_variant_is_free_of_strategy_phrases(orchestration: str, show_dev: bool) -> None:
    ep = _episode_with_input_tool()
    p = build_system_prompt(ep, [], [], orchestration, fixed_code_node="core", show_dev_score=show_dev)
    assert find_strategy_phrases(p) == []


# --------------------------------------------------------------------- F9/M1: compact history
class _Rec:
    def __init__(self, status: str, cached: bool = False, error: str | None = None) -> None:
        self.status, self.cached, self.error = status, cached, error


def _fb(**kw):
    base = {"action_ok": True, "records": {}, "visible_constraints": {}}
    base.update(kw)
    return base


def test_compact_history_line_carries_y_summary_and_dev_score() -> None:
    y = summarize_value([1.0, 2.0, 3.5, 4.0])
    line = summarize_feedback(_fb(submit_ready=True, y_summary=y, dev={"score": 0.75, "direction": "max"},
                                  records={"a": _Rec("ok", cached=True), "b": _Rec("ok")}))
    assert line.startswith("applied; y: list shape=4 dtype=float64")
    assert "range=[1, 4]" in line and "mean=2.625" in line
    assert "head=" not in line                                    # the value list is not repeated in the history
    assert 'dev: {"direction": "max", "score": 0.75}' in line
    assert line.index("y: ") < line.index("dev: ") < line.index("nodes: ")
    assert "nodes: b=ok, +1 cached ok" in line


def test_compact_history_without_y_or_dev() -> None:
    line = summarize_feedback(_fb(records={"a": _Rec("pending")}, submit_ready=False))
    assert "no y" in line and "dev:" not in line and "nodes: a=pending" in line


def test_compact_history_lists_changed_and_failed_nodes_only() -> None:
    recs = {f"c{i}": _Rec("ok", cached=True) for i in range(6)}
    recs["bad"] = _Rec("error", error="Traceback\n  File x\nValueError: boom")
    recs["new"] = _Rec("ok")
    line = summarize_feedback(_fb(records=recs))
    assert "bad=error (ValueError: boom)" in line and "new=ok" in line and "+6 cached ok" in line
    assert "c0" not in line


def test_compact_history_caps_the_node_list() -> None:
    recs = {f"n{i:02d}": _Rec("error", error=f"E{i}") for i in range(12)}
    line = summarize_feedback(_fb(records=recs), max_chars=2000)
    assert "+4 more" in line and line.count("=error") == 8


def test_compact_history_keeps_rejections_and_constraints() -> None:
    assert summarize_feedback({"action_ok": False, "action_error": "bad"}) == "rejected: bad"
    line = summarize_feedback(_fb(visible_constraints={"finite": [False, "nan"], "ok": [True, ""]}))
    assert "constraints failed: finite" in line and "ok" not in line.split("constraints failed: ")[1]


def test_compact_history_scrubs_before_truncating() -> None:
    run_a, run_b = "/runs/aaaa1111/x", "/runs/bbbb2222/y"
    err = f"Traceback\nFileNotFoundError: {run_a}/work/f-1a2b3c4d/data.csv " + "z" * 300
    lines = []
    for run, e in ((run_a, err), (run_b, err.replace(run_a, run_b))):
        scrub = lambda t, r=run: scrub_volatile(t, r)           # noqa: E731
        lines.append(summarize_feedback(_fb(records={"f": _Rec("error", error=e)}), scrub=scrub))
    assert lines[0] == lines[1]
    assert "aaaa1111" not in lines[0] and "1a2b3c4d" not in lines[0]
    # the shortening inside summarize_feedback (last error line at 120 chars, whole line at max_chars) would land
    # inside the uid or the run path for some of these paddings: no unscrubbed fragment may survive
    for pad in range(40, 140):
        text = "q" * pad + f" {run_a}/work/f-1a2b3c4d/out.pkl"
        for max_chars in (400, 90):
            out = summarize_feedback(_fb(records={"f": _Rec("error", error=text)}), max_chars=max_chars,
                                     scrub=lambda t: scrub_volatile(t, run_a))
            assert "1a2b3c4" not in out and "aaaa" not in out and "/runs" not in out, (pad, max_chars, out)


def test_step_message_is_independent_of_run_dir_and_uid() -> None:
    def history(run: str, uid: str) -> list[dict]:
        res = f"error in {run}/work/f-{uid}/x.py line 3 (0.31s)"
        return [{"step": 0, "action": f"modify_node f [code] in {run}", "thought": f"see {run}", "result": res}]

    msgs = []
    for run, uid in (("/runs/aaaa1111/r", "1a2b3c4d"), ("/other/bbbb2222/r", "9f8e7d6c")):
        # the feedback text arrives already scrubbed (solver._render_feedback); the history is scrubbed here
        msgs.append(build_step_message(1, WorkflowGraph(), "feedback text", history(run, uid),
                                       {"max_steps": 4, "steps_left": 3},
                                       scrub=lambda t, r=run: scrub_volatile(t, r)))
    assert msgs[0] == msgs[1]
    assert "<run>" in msgs[0] and "work/f/" in msgs[0] and "1a2b3c4d" not in msgs[0]


def test_step_message_scrubs_before_truncating_old_entries() -> None:
    run = "/runs/aaaa1111/some/very/long/run/directory/name"
    res = "x" * 140 + f" {run}/work/f-1a2b3c4d/y"
    hist = [{"step": i, "action": "a", "thought": "", "result": res} for i in range(3)]
    msg = build_step_message(3, WorkflowGraph(), None, hist, {"max_steps": 9, "steps_left": 6}, window=1,
                             scrub=lambda t: scrub_volatile(t, run))
    assert "aaaa1111" not in msg and "/runs/" not in msg and "1a2b3c4d" not in msg


# ---------------------------------------------------------------- feedback render / uid regex
def test_work_uid_regex() -> None:
    cases = {
        "work/f-1a2b3c4d/x.py": "work/f/x.py",
        "in work/f-1a2b3c4d: boom": "in work/f: boom",
        "(work/my_node-1a2b3c4d)": "(work/my_node)",
        "work/f-1a2b3c4d": "work/f",
        "work/f-1a2b3": "work/f",                                 # a uid cut short at the very end of the text
        "work/f-1a": "work/f",
        "work/a-b-1a2b3c4d/": "work/a-b/",
        "work/f-1a2b3c4d5/x": "work/f-1a2b3c4d5/x",               # longer hex run: not a uid
        "network/f-1a2b3c4d/x": "network/f/x",
        "work/f-abc rest": "work/f-abc rest",                      # short hex in the middle of the text stays
        "no path here-1a2b3c4d": "no path here-1a2b3c4d",
    }
    for src, want in cases.items():
        assert scrub_volatile(src) == want, src


def _error_feedback(run: str, error: str, stdout: str = "") -> Feedback:
    rec = NodeRecord(node_id="f", fingerprint="fp", status="error", wall_s=0.31, error=error, stdout_tail=stdout, kind="code")
    return Feedback(step=0, action_ok=True, records={"f": rec}, action_desc="modify_node f")


def test_feedback_render_scrubs_before_clipping() -> None:
    run = "/runs/aaaa1111/long/run/dir"
    # the traceback tail is clipped at 1500 chars by _clip_tail: the path sits right at the clip boundary
    for pad in range(1440, 1520, 3):
        error = "E" * pad + f" {run}/work/f-1a2b3c4d/data.csv not found"
        fb = _error_feedback(run, error, stdout=f"loading {run}/work/f-1a2b3c4d/data.csv")
        text = fb.render(scrub=lambda t: scrub_volatile(t, run))
        assert "aaaa1111" not in text and "1a2b3c4d" not in text and "/runs/" not in text, pad
        assert "(0.31s)" not in text and "ERROR" in text


def test_feedback_render_scrubs_before_the_final_cut() -> None:
    run = "/runs/aaaa1111/long/run/dir"
    recs = {f"n{i}": NodeRecord(node_id=f"n{i}", fingerprint="f", status="error", wall_s=0.0,
                                error=f"boom {run}/work/n{i}-1a2b3c4d/f", kind="code") for i in range(60)}
    fb = Feedback(step=0, action_ok=True, records=recs)
    for limit in range(600, 1400, 37):
        text = fb.render(max_chars=limit, scrub=lambda t: scrub_volatile(t, run))
        assert len(text) <= limit
        assert "aaaa1111" not in text and "1a2b3c4d" not in text, limit


def test_feedback_render_without_scrub_is_unchanged() -> None:
    run = "/runs/aaaa1111/x"
    fb = _error_feedback(run, f"boom {run}/work/f-1a2b3c4d/y")
    text = fb.render()
    assert f"{run}/work/f-1a2b3c4d/y" in text
    assert fb.render(scrub=None) == text


def test_scrubbed_render_is_identical_across_run_dirs() -> None:
    texts = []
    for run, uid in (("/runs/aaaa1111/x", "1a2b3c4d"), ("/tmp/other/b2/x", "0f0f0f0f")):
        fb = _error_feedback(run, f"boom {run}/work/f-{uid}/y", stdout=f"cwd {run}/work/f-{uid}")
        texts.append(fb.render(scrub=lambda t, r=run: scrub_volatile(t, r)))
    assert texts[0] == texts[1]


def test_summarize_value_lists_record_keys_for_rendering() -> None:
    s = summarize_value([{"id": 1, "text": "a"}, {"id": 2, "label": "b"}])
    assert s["item_keys"] == ["id", "text", "label"]
    line = summarize_feedback(_fb(submit_ready=True, y_summary=s))
    assert 'item_keys=["id", "text", "label"]' in line
