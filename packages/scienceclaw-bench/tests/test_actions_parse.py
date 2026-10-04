"""parse_action: policy JSON -> Action, for every action type and every error path."""
from __future__ import annotations

import json

import pytest

from scienceclaw.core.actions import ACTION_TYPES, Action, extract_json_object, parse_action

CODE = "def run(inputs, config):\n    return {'m': inputs['x'] * 2}\n"


def wrap(action: dict, uses=None, thought: str = "because") -> str:
    return json.dumps({"thought": thought, "action": action, "uses": uses or []})


def ok(text: str) -> Action:
    a, err = parse_action(text)
    assert err is None, err
    assert a is not None
    return a


def bad(text: str) -> str:
    a, err = parse_action(text)
    assert a is None
    assert isinstance(err, str) and err
    return err


# ------------------------------------------------------------------------------ every type
def test_parse_add_node_tool_and_uses_filtering() -> None:
    a = ok(wrap({"type": "add_node", "node": {"id": "load", "kind": "tool", "ref": "load_data", "config": {"k": 1}}},
                uses=["skill:s1", "op:o1", "tool:load_data", "junk", "skill:s1"]))
    assert a.type == "add_node"
    assert a.payload["node"]["ref"] == "load_data"
    assert a.payload["node"]["config"] == {"k": 1}
    assert a.uses == ["skill:s1", "op:o1"]       # only program components, deduplicated, in order
    assert a.thought == "because"
    assert "add_node" in a.raw


def test_parse_add_node_code_llm_operator_submit() -> None:
    a = ok(wrap({"type": "add_node", "node": {"id": "c1", "kind": "code", "code": CODE,
                                              "inputs": {"x": {"type": "array"}}, "outputs": {"m": {"type": "array"}}}}))
    assert a.payload["node"]["code"] == CODE
    a = ok(wrap({"type": "add_node", "node": {"id": "l1", "kind": "llm", "prompt": "Classify {text}",
                                              "config": {"parse": "choice", "choices": ["a", "b"]}}}))
    assert a.payload["node"]["prompt"] == "Classify {text}"
    a = ok(wrap({"type": "add_node", "node": {"id": "o1", "kind": "operator", "ref": "op:smooth"}}))
    assert a.payload["node"]["ref"] == "op:smooth"
    a = ok(wrap({"type": "add_node", "node": {"id": "out", "kind": "submit", "inputs": {"junk": "ignored"}}}))
    assert "inputs" not in a.payload["node"]            # submit ports come from the task, not the policy
    assert a.payload["node"]["config"] == {}


def test_parse_remove_modify_edges_finish_batch() -> None:
    assert ok(wrap({"type": "remove_node", "id": "n1"})).payload == {"id": "n1"}
    a = ok(wrap({"type": "modify_node", "id": "n1", "patch": {"config": {"alpha": None, "beta": 2}}}))
    assert a.payload["patch"]["config"] == {"alpha": None, "beta": 2}
    e = {"src": "a", "src_port": "out", "dst": "b", "dst_port": "x",
         "conversion": {"from": "K", "to": "degC", "factor": 1, "offset": -273.15}}
    a = ok(wrap({"type": "add_edge", "edge": e}))
    assert a.payload["edge"]["conversion"]["offset"] == -273.15
    a = ok(wrap({"type": "remove_edge", "edge": {"src": "a", "dst": "b"}}))
    assert a.payload["edge"] == {"src": "a", "dst": "b"}
    assert ok(wrap({"type": "finish"})).type == "finish"
    a = ok(wrap({"type": "batch", "actions": [
        {"type": "add_node", "node": {"id": "t", "kind": "tool", "ref": "load"}},
        {"action": {"type": "add_edge", "edge": {"src": "t", "dst": "s"}}}]}))
    assert [s["type"] for s in a.payload["actions"]] == ["add_node", "add_edge"]
    assert set(ACTION_TYPES) == {"add_node", "remove_node", "modify_node", "add_edge", "remove_edge", "finish", "batch"}


def test_flat_payload_leniency() -> None:
    a = ok(wrap({"type": "add_node", "id": "t1", "kind": "tool", "ref": "load"}))
    assert a.payload["node"]["id"] == "t1"
    a = ok(wrap({"type": "add_edge", "src": "a", "dst": "b", "dst_port": "x"}))
    assert a.payload["edge"] == {"src": "a", "dst": "b", "dst_port": "x"}
    a = ok(json.dumps({"type": "remove_node", "id": "x"}))       # bare action object
    assert a.type == "remove_node" and a.uses == []


# ------------------------------------------------------------------------------ tolerant JSON
def test_code_fences_prose_newlines_trailing_commas() -> None:
    body = wrap({"type": "remove_node", "id": "n1"})
    assert ok(f"Sure, here it is:\n```json\n{body}\n```\nDone.").payload == {"id": "n1"}
    assert ok(f"I will remove it. {body} That's all.").payload == {"id": "n1"}
    # raw (unescaped) newlines inside the code string, as models often emit
    raw = ('{"thought": "x", "action": {"type": "add_node", "node": {"id": "c", "kind": "code", '
           '"code": "def run(inputs, config):\n    return {\\"y\\": 1}", "outputs": {"y": {"type": "number"}}}}, "uses": []}')
    a = ok(raw)
    assert "\n    return" in a.payload["node"]["code"]
    a = ok('{"thought": "t", "action": {"type": "remove_node", "id": "n2",}, "uses": [],}')
    assert a.payload == {"id": "n2"}


def test_extract_json_object_prefers_action_container() -> None:
    text = 'noise {"a": 1} more {"action": {"type": "finish"}} tail {'
    assert extract_json_object(text) == {"action": {"type": "finish"}}
    assert extract_json_object("no json here") is None


# ------------------------------------------------------------------------------ error paths
@pytest.mark.parametrize("text,needle", [
    ("", "empty response"),
    ("I think we should add a node.", "no JSON object"),
    ('{"thought": "x", "uses": []}', "no \"action\""),
    ('{"action": "add_node"}', "must be an object"),
    (wrap({"type": "delete_everything"}), "action.type must be one of"),
    (wrap({"type": "add_node"}), "add_node requires"),
    (wrap({"type": "add_node", "node": {"id": "bad id!", "kind": "tool", "ref": "x"}}), "node.id must match"),
    (wrap({"type": "add_node", "node": {"id": "x" * 41, "kind": "tool", "ref": "x"}}), "node.id must match"),
    (wrap({"type": "add_node", "node": {"id": "n", "kind": "python"}}), "node.kind must be one of"),
    (wrap({"type": "add_node", "node": {"id": "n", "kind": "tool"}}), "requires \"ref\""),
    (wrap({"type": "add_node", "node": {"id": "n", "kind": "operator", "ref": " "}}), "requires \"ref\""),
    (wrap({"type": "add_node", "node": {"id": "n", "kind": "code", "outputs": {"y": {"type": "number"}}}}), "requires \"code\""),
    (wrap({"type": "add_node", "node": {"id": "n", "kind": "code", "code": CODE}}), "non-empty \"outputs\""),
    (wrap({"type": "add_node", "node": {"id": "n", "kind": "llm"}}), "non-empty \"prompt\""),
    (wrap({"type": "add_node", "node": {"id": "n", "kind": "code", "code": CODE,
                                        "outputs": {"y": {"type": "matrix"}}}}), "unknown port type"),
    (wrap({"type": "add_node", "node": {"id": "n", "kind": "code", "code": CODE,
                                        "outputs": {"bad port": {"type": "array"}}}}), "invalid port name"),
    (wrap({"type": "add_node", "node": {"id": "n", "kind": "code", "code": CODE, "config": [1],
                                        "outputs": {"y": {"type": "array"}}}}), "config must be an object"),
    (wrap({"type": "remove_node"}), "remove_node requires"),
    (wrap({"type": "modify_node", "id": "n"}), "non-empty \"patch\""),
    (wrap({"type": "modify_node", "id": "n", "patch": {}}), "non-empty \"patch\""),
    (wrap({"type": "modify_node", "id": "n", "patch": {"kind": "code"}}), "unsupported patch key"),
    (wrap({"type": "modify_node", "id": "n", "patch": {"code": ""}}), "patch.code must be a non-empty string"),
    (wrap({"type": "modify_node", "id": "n", "patch": {"config": "x"}}), "patch.config must be an object"),
    (wrap({"type": "modify_node", "id": "n", "patch": {"inputs": {"x": {"type": "nope"}}}}), "unknown port type"),
    (wrap({"type": "add_edge", "edge": {"src": "a"}}), "edge.dst must be a node id"),
    (wrap({"type": "add_edge"}), "requires \"edge\""),
    (wrap({"type": "add_edge", "edge": {"src": "a", "dst": "b", "src_port": 3}}), "edge.src_port must be a port name"),
    (wrap({"type": "add_edge", "edge": {"src": "a", "dst": "b", "conversion": {"from": "K", "to": "C"}}}), "missing ['factor']"),
    (wrap({"type": "add_edge", "edge": {"src": "a", "dst": "b",
                                        "conversion": {"from": "K", "to": "C", "factor": "1"}}}), "factor must be a number"),
    (wrap({"type": "batch", "actions": []}), "non-empty \"actions\""),
    (wrap({"type": "batch", "actions": [{"type": "batch", "actions": [{"type": "finish"}]}]}), "batch action #0"),
    (wrap({"type": "batch", "actions": [{"type": "finish"}, {"type": "remove_node"}]}), "batch action #1 (remove_node)"),
])
def test_parse_errors(text: str, needle: str) -> None:
    err = bad(text)
    assert needle in err, err


def test_batch_can_be_disallowed() -> None:
    a, err = parse_action(wrap({"type": "batch", "actions": [{"type": "finish"}]}), allow_batch=False)
    assert a is None and "not allowed" in err


def test_action_dict_roundtrip_and_policy_format() -> None:
    a = ok(wrap({"type": "modify_node", "id": "n", "patch": {"prompt": "p"}}, uses=["skill:k"]))
    b = Action.from_dict(a.to_dict())
    assert (b.type, b.payload, b.uses, b.thought, b.raw) == (a.type, a.payload, a.uses, a.thought, a.raw)
    c = Action.from_dict({"thought": "t", "action": {"type": "remove_node", "id": "q"}, "uses": ["op:o"]})
    assert (c.type, c.payload, c.uses) == ("remove_node", {"id": "q"}, ["op:o"])
    d = Action.from_dict({"type": "finish"})
    assert d.type == "finish" and d.payload == {}
    assert a.action_json() == {"type": "modify_node", "id": "n", "patch": {"prompt": "p"}}
    assert "modify_node n" in a.describe()
