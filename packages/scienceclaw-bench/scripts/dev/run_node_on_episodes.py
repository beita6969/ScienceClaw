"""Development check of a hand-written code node: run ``--code file.py`` (def run(inputs, config)) in the real
sandbox on the visible data of the first N episodes of a split and print dev/eval scores.

The node receives ``inputs[tool_name][port]`` for every no-input tool; it must return {"dev_pred": ..., "y": ...}
(``dev_pred`` is passed to score_dev's first input port, whatever its name).
Usage: python scripts/dev/run_node_on_episodes.py --disc FoR50 --code node.py --split val --n 3
"""
from __future__ import annotations

import argparse
import tempfile
import time
from types import SimpleNamespace

from scienceclaw.bench.splits import load_adapters
from scienceclaw.core.graph import Node
from scienceclaw.core.schema import PortSchema
from scienceclaw.runtime.sandbox import run_code_node


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disc", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args()
    ad = load_adapters(SimpleNamespace(disciplines=[a.disc], data_root=None))[a.disc]
    src = open(a.code).read()
    for ep in ad.build_episodes(a.split, a.n, a.seed, a.items):
        vals: dict = {}
        for t in ep.tools:
            if not t.inputs:
                vals[t.name] = t.fn({}, {})
        node = Node(id="solve", kind="code", code=src, inputs={k: PortSchema("any") for k in vals},
                    outputs={"y": PortSchema("any"), "dev_pred": PortSchema("any")})
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            out, meta = run_code_node(node, vals, td, a.timeout)
        if out is None:
            print(ep.id, "NODE FAILED", meta["status"], (meta.get("error") or "")[-600:], meta.get("stdout_tail", "")[-300:])
            continue
        dev = ep.tool("score_dev")
        port = next(iter(dev.inputs)) if dev is not None and dev.inputs else "dev_predictions"
        d = dev.fn({port: out["dev_pred"]}, {}) if dev is not None and out.get("dev_pred") is not None else {}
        d = {k: v for k, v in d.items() if isinstance(v, (int, float))}
        r = ep._evaluate(out["y"], None)
        print(ep.id, f"{time.time() - t0:.0f}s dev={ {k: round(v, 3) for k, v in d.items()} } "
              f"primary={r.primary and round(r.primary, 4)} ref={r.details.get('reference') and round(r.details['reference'], 4)} "
              f"accepted={r.accepted} norm={r.details.get('norm_score') and round(r.details['norm_score'], 3)}")


if __name__ == "__main__":
    main()
