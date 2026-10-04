import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "leonardo" / "migration" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("scienceclaw_leonardo_summarize", SCRIPT)
assert SPEC and SPEC.loader
summarize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize)


def _row(node, status):
    return {
        "action": {"payload": {"node": node}},
        "feedback": {"records": {node["id"]: {"status": status}}},
    }


def test_trace_usage_ignores_failed_nodes(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    rows = [
        _row({"id": "ok_tool", "kind": "tool", "ref": "predict_pretrained"}, "ok"),
        _row({"id": "failed_tool", "kind": "tool", "ref": "separate_htdemucs"}, "error"),
        _row({"id": "ok_code", "kind": "code", "code": "import scilib.matphonon_mlip"}, "ok"),
        _row({"id": "failed_code", "kind": "code", "code": "import scilib.phenoseg_m2f"}, "error"),
        # A proposal without an executor record is not evidence of a call.
        {"action": {"payload": {"node": {"id": "unexecuted", "kind": "tool", "ref": "fit_sevennet_mlip"}}}},
    ]
    trajectory.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    tools, direct = summarize.trace_usage_details(trajectory)

    assert tools == {"predict_pretrained"}
    assert direct == {"matphonon_mlip"}
    assert summarize.trace_usage(trajectory) == (True, True)
