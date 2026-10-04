"""Tests for scienceclaw.llm.extract_json (tolerant JSON-object extraction)."""
from __future__ import annotations

import time

import pytest

from scienceclaw.llm import extract_json


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('  {"a": [1, 2, {"b": null}]}\n', {"a": [1, 2, {"b": None}]}),
    ('```json\n{"type": "finish"}\n```', {"type": "finish"}),
    ('```\n{"type": "finish"}\n```', {"type": "finish"}),
    ('Sure! Here is the action:\n```JSON\n{"x": 2}\n```\nLet me know.', {"x": 2}),
    ('I will add a node. {"action": {"type": "remove_node", "id": "n3"}} Done.',
     {"action": {"type": "remove_node", "id": "n3"}}),
    ('{"a": 1, "b": [1, 2,], "c": {"d": 3,},}', {"a": 1, "b": [1, 2], "c": {"d": 3}}),
    ('{"ok": True, "missing": None, "bad": False}', {"ok": True, "missing": None, "bad": False}),
    ("{'type': 'finish', 'uses': ['skill:s1'], 'flag': True}", {"type": "finish", "uses": ["skill:s1"], "flag": True}),
    ('{"s": "True, story", "t": "a,}",}', {"s": "True, story", "t": "a,}"}),
    ('Use {x} placeholders; then {"prompt": "{item} -> label"}', {"prompt": "{item} -> label"}),
    ("Here's my plan {not json}: {\"a\": 1}", {"a": 1}),
    ('first {"a": 1} second {"b": 2}', {"a": 1}),
    ('{"note": "escaped \\" quote and } brace"} trailing', {"note": 'escaped " quote and } brace'}),
])
def test_extract_json_cases(text: str, expected: dict) -> None:
    assert extract_json(text) == expected


def test_raw_newlines_inside_strings_are_accepted() -> None:
    text = '{"action": {"type": "add_node", "node": {"code": "def run(inputs, config):\n    return {\'y\': 1}\n"}}}'
    obj = extract_json(text)
    assert obj is not None
    assert obj["action"]["node"]["code"].startswith("def run(inputs, config):\n")


def test_fence_with_prose_inside() -> None:
    text = "```json\nHere you go: {\"k\": \"v\"}\n```"
    assert extract_json(text) == {"k": "v"}


@pytest.mark.parametrize("text", ["", None, "no json here", "[1, 2, 3]", '{"a": 1', "{x}", "```json\n```", "42"])
def test_extract_json_returns_none(text) -> None:
    assert extract_json(text) is None


def test_many_braces_is_fast() -> None:
    text = "code: " + "{a} " * 2000 + ' {"ok": 1}'
    t0 = time.monotonic()
    extract_json(text)
    assert time.monotonic() - t0 < 2.0
