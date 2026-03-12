#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Dict, List


def parse_list(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("scores must be a comma-separated list")
    return [float(p) for p in parts]


def parse_names(raw: str, count: int) -> List[str]:
    if not raw:
        return [f"p{i+1}" for i in range(count)]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != count:
        raise ValueError("names count must match scores count")
    return parts


def summarize(scores: List[float], names: List[str]) -> Dict[str, object]:
    ranking = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    notes = []
    if ranking and ranking[0][1] < 0.1:
        notes.append("All sensitivities are low; consider alternative outputs.")
    return {"ranking": ranking, "notes": notes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize sensitivity scores and rank parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores", required=True, help="Comma-separated sensitivity scores")
    parser.add_argument("--names", default=None, help="Comma-separated parameter names")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        scores = parse_list(args.scores)
        names = parse_names(args.names, len(scores))
        result = summarize(scores, names)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "scores": scores,
            "names": names,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Sensitivity summary")
    for name, score in result["ranking"]:
        print(f"  {name}: {score:.6g}")


if __name__ == "__main__":
    main()
