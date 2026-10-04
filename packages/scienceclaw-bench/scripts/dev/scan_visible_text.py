"""List every sentence of the policy-visible episode text that talks about the reference, baseline, organiser or acceptance.

Usage: python scripts/dev/scan_visible_text.py [--config configs/full_local.yaml] [--split val] [--disciplines FoR30,...]
Reads one episode per discipline and prints, per discipline, the sentences of the objective, tool / port descriptions
and visible constraints that match TERMS (DESIGN section 7 F5: these texts must not state the reference method or the
acceptance rule). Output is for review; the strict FoR33/35/37/38/41 test is tests/test_adapter_visible_text.py.
"""
from __future__ import annotations

import argparse
import json
import re

from scienceclaw.bench.splits import SplitPlan, load_adapters
from scienceclaw.config import load_config

TERMS = re.compile(r"reference|baseline|starter|organi[sz]er|margin|beat|acceptance|accepted|success criterion|"
                   r"official no-edit|no[- ]edit|majority|naive|persistence|climatolog", re.I)


def texts(ep) -> list[tuple[str, str]]:
    out = [("objective", ep.objective)]
    out += [(f"tool:{t.name}", t.config_doc or "") for t in ep.tools]
    out.append(("public_view", json.dumps(ep.public_view(), default=str, ensure_ascii=False)))
    out.append(("required_output", json.dumps(ep.required_output.to_dict(), default=str, ensure_ascii=False)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full_local.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--disciplines", default="")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.disciplines:
        cfg.bench.disciplines = a.disciplines.split(",")
    plan = SplitPlan.build(cfg.bench, load_adapters(cfg.bench))
    for code in plan.order:
        eps = plan.episodes[a.split].get(code, [])
        if not eps:
            print(f"## {code}: no {a.split} episode")
            continue
        seen = set()
        print(f"## {code}")
        for where, txt in texts(eps[0]):
            for sent in re.split(r"(?<=[.;])\s+|\\n|\n", txt):
                if TERMS.search(sent) and sent not in seen:
                    seen.add(sent)
                    print(f"  [{where}] {sent.strip()[:300]}")


if __name__ == "__main__":
    main()
