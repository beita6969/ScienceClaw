"""Smoke/throughput test of a self-hosted vLLM endpoint (no credentials involved).

Usage: python scripts/leonardo/bench_endpoint.py http://127.0.0.1:18001/v1 [--n 32] [--conc 16] [--model sc-llm]
Checks: JSON mode, thinking disabled (no <think>/reasoning text), completion speed under concurrency.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

PROMPT = ("You are a graph-program editor. Return ONLY a JSON object {\"action\": {\"type\": \"add_node\", "
          "\"payload\": {\"node\": {\"id\": <str>, \"kind\": \"tool\", \"ref\": <str>}}}, \"why\": <str>} that adds a node "
          "named n%d calling the tool `load_smiles` for a molecular property task. Context filler: " + "lorem ipsum " * 300)


def call(base: str, model: str, i: int, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT % i}], "max_tokens": max_tokens,
            "temperature": 0.0, "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    text = d["choices"][0]["message"].get("content") or ""
    try:
        json.loads(text)
        ok = True
    except Exception:
        ok = False
    return {"wall": time.time() - t0, "json_ok": ok, "think": "<think>" in text, "usage": d.get("usage", {}), "text": text[:160]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--model", default="sc-llm")
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as pool:
        rs = list(pool.map(lambda i: call(a.base, a.model, i, a.max_tokens), range(a.n)))
    wall = time.time() - t0
    ptok = sum(r["usage"].get("prompt_tokens", 0) for r in rs)
    ctok = sum(r["usage"].get("completion_tokens", 0) for r in rs)
    print(f"n={a.n} conc={a.conc} wall={wall:.1f}s prompt_tok={ptok} completion_tok={ctok} "
          f"gen_tok/s={ctok / wall:.1f} json_ok={sum(r['json_ok'] for r in rs)}/{a.n} think={sum(r['think'] for r in rs)}")
    print("median latency %.1fs, max %.1fs" % (sorted(r["wall"] for r in rs)[len(rs) // 2], max(r["wall"] for r in rs)))
    print("sample:", rs[0]["text"])


if __name__ == "__main__":
    main()
