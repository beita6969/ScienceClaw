"""Prompt tests: interface-only content (no strategy phrases), required elements, determinism."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_agent_support import ACCEPTANCE_TEXT, SECRET_METRIC, make_episode  # noqa: E402

from scienceclaw.agent.prompts import (  # noqa: E402
    ORCHESTRATIONS, build_step_message, build_system_prompt, find_strategy_phrases, summarize_action,
    summarize_feedback,
)
from scienceclaw.core.graph import Node, WorkflowGraph  # noqa: E402
from scienceclaw.core.operators import Contract, OperatorSpec  # noqa: E402
from scienceclaw.core.schema import PortSchema  # noqa: E402
from scienceclaw.core.skills import Skill  # noqa: E402

# Advice that an initial prompt must never contain (DESIGN decision 9) — all must be caught by the scanner.
ADVICE_SAMPLES = [
    "Start by loading the data.", "Begin with a quick look at the inputs.", "First, check for NaNs.",
    "Use cross-validation to pick the model.", "Add a submit node early.", "You should validate the output.",
    "We recommend a gradient boosting model.", "Make sure to handle missing values.", "Try to keep the graph small.",
    "Consider normalizing the features.", "Tip: inspect the stdout.", "A good strategy is to iterate.",
    "Compare against a baseline.", "Do a sanity check of shapes.", "Explore the data before modelling.",
    "Proceed step by step.", "Improve the score by tuning hyperparameters.", "Avoid overfitting.",
    "Prefer simple models.", "Don't forget the units.", "Remember to convert units.", "Debug failing nodes.",
    "It is important to standardize inputs.", "Plan your workflow carefully.", "A fallback model is useful.",
]


def _skill() -> Skill:
    return Skill(id="s1", version=2, title="Robust forecasting recovery",
                 body="Start by loading the data, then check for NaNs and use cross-validation.", tags=["FoR49"])


def _operator() -> OperatorSpec:
    body = WorkflowGraph({"c": Node("c", "code", code="def run(i, c):\n    return {'y': i['x']}\n",
                                    inputs={"x": PortSchema("array")}, outputs={"y": PortSchema("array")})})
    return OperatorSpec(id="op1", version=1, name="identity", description="Passes an array through.", body=body,
                        inputs={"x": PortSchema("array", shape=("n",), unit="K")},
                        outputs={"y": PortSchema("array", shape=("n",), unit="K")},
                        input_map={"x": [("c", "x")]}, output_map={"y": ("c", "y")},
                        contract=Contract(pre=[{"port": "x", "check": "finite"}], applicability={"disciplines": ["FoR49"]}))


def test_scanner_catches_advice() -> None:
    for s in ADVICE_SAMPLES:
        assert find_strategy_phrases(s), f"scanner missed advice: {s!r}"


@pytest.mark.parametrize("orchestration", ORCHESTRATIONS)
def test_initial_system_prompt_has_no_strategy(orchestration: str) -> None:
    prompt = build_system_prompt(make_episode(), [], [], orchestration, fixed_code_node="code")
    hits = find_strategy_phrases(prompt)
    assert hits == [], f"strategy phrasing in the {orchestration} system prompt: {hits}"


def test_step_message_template_has_no_strategy() -> None:
    msg = build_step_message(0, WorkflowGraph(), None, [], {"max_steps": 12, "steps_left": 12})
    assert find_strategy_phrases(msg) == []


def test_system_prompt_contents() -> None:
    ep = make_episode()
    p = build_system_prompt(ep, [], [], "canvas")
    # canvas, node kinds, port syntax, actions, uses
    for needle in ("Workflow Canvas", "directed acyclic graph", "- tool:", "- operator:", "- code:", "- llm:",
                   "- submit:", "def run(inputs: dict, config: dict) -> dict", "numpy", "no network access",
                   '"parse"', '"type": "add_node"', '"type": "finish"', '"uses"', "PortSchema JSON",
                   '"conversion"', "x_dst = a * x_src + b", "Skills from your library", "Operators from your library"):
        assert needle in p, needle
    # episode: objective, required output, visible constraints, tools, budget
    assert ep.objective in p
    assert "one value per x" in p and '{"type": "list"}' in p
    assert "- finite: all values finite" in p
    assert ep.tools[0].signature() in p
    assert "at most 12" in p and "at most 200000" in p
    # never: hidden constraints, acceptance rule, metric name
    for hidden in ("INVISIBLE_CONSTRAINT_TEXT", "hidden_rule_q", ACCEPTANCE_TEXT, SECRET_METRIC):
        assert hidden not in p
    assert "Orchestration: canvas" in p and '"type": "batch"' not in p


def test_system_prompt_renders_retrieved_components_in_full() -> None:
    sk, op = _skill(), _operator()
    p = build_system_prompt(make_episode(), [sk], [op], "canvas")
    assert sk.render() in p and op.render() in p
    # strategy text is allowed when it comes from a learned Skill
    assert "check for NaNs" in p


def test_orchestration_variants() -> None:
    ep = make_episode()
    st = build_system_prompt(ep, [], [], "single_turn", max_steps=1)
    assert '{"type": "batch", "actions": [<ACTION>, <ACTION>, ...]}' in st and "at most 1\n" in st
    so = build_system_prompt(ep, [], [], "single_operator")
    assert "at most one code node" in so
    fw = build_system_prompt(ep, [], [], "fixed_workflow", fixed_code_node="core")
    assert 'modify_node on node "core"' in fw
    with pytest.raises(ValueError):
        build_system_prompt(ep, [], [], "freestyle")


def test_prompts_are_deterministic() -> None:
    a = build_system_prompt(make_episode(), [_skill()], [_operator()], "canvas")
    b = build_system_prompt(make_episode(), [_skill()], [_operator()], "canvas")
    assert a == b


def test_step_message_history_window_and_budget() -> None:
    g = WorkflowGraph({"n1": Node("n1", "code", code="def run(i, c):\n    return {}\n",
                                  outputs={"y": PortSchema("number")})})
    hist = [{"step": i, "action": f"act{i}", "thought": f"why{i}", "result": f"res{i}"} for i in range(5)]
    msg = build_step_message(5, g, "FEEDBACK-TEXT", hist, {"max_steps": 8, "steps_left": 3}, window=2)
    assert "## Step 6 of 8 (3 left including this one)" in msg
    assert "- step 1: act0 -> res0" in msg                     # old entries: one line, no thought
    assert "why0" not in msg and "why4" in msg and "why3" in msg  # window entries carry the thought
    assert "FEEDBACK-TEXT" in msg and "[n1] kind=code" in msg
    assert msg.rstrip().endswith("Reply with exactly one JSON object in the action format.")
    # no wall-clock values are rendered (identical contexts -> identical prompts)
    assert "wall" not in msg


def test_summaries() -> None:
    assert summarize_action({"type": "add_node", "payload": {"node": {"id": "a", "kind": "tool", "ref": "t"}}}) \
        == "add_node a kind=tool ref=t"
    assert summarize_action({"type": "modify_node", "payload": {"id": "a", "patch": {"code": "x", "config": {}}}}) \
        == "modify_node a [code,config]"
    assert summarize_action(None) == "(no parsable action)"
    assert summarize_feedback({"action_ok": False, "action_error": "bad"}) == "rejected: bad"
    assert "constraints failed: finite" in summarize_feedback(
        {"action_ok": True, "records": {}, "visible_constraints": {"finite": [False, "nan"]}})


def test_step_message_shows_policy_tokens_left():
    from scienceclaw.agent.prompts import build_step_message
    from scienceclaw.core.graph import WorkflowGraph
    msg = build_step_message(2, WorkflowGraph(), None, [], {"max_steps": 12, "steps_left": 10, "policy_tokens_left": 143000})
    assert "Step 3 of 12 (10 left including this one); policy_tokens_left: 143000" in msg


def test_step_message_shows_cost_of_last_reply():
    msg = build_step_message(3, WorkflowGraph(), None, [], {"max_steps": 24, "steps_left": 21, "policy_tokens_left": 90000,
                                                            "last_reply_tokens": 9000})
    assert "Step 4 of 24 (21 left including this one); last_reply_tokens: 9000, policy_tokens_left: 90000" in msg
