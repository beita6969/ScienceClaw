#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Dict, List


def parse_list(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("value list must be a comma-separated list")
    return [float(p) for p in parts]


def build_surrogate(x: List[float], y: List[float], model: str) -> Dict[str, object]:
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if model not in {"rbf", "poly"}:
        raise ValueError("model must be rbf or poly")
    if len(x) < 2:
        raise ValueError("need at least 2 samples")

    mean_y = sum(y) / len(y)
    mse = sum((yi - mean_y) ** 2 for yi in y) / len(y)
    return {
        "model_type": model,
        "metrics": {"mse": mse},
        "notes": ["Surrogate is a placeholder; replace with real model."],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simple surrogate model summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--x", required=True, help="Comma-separated input values")
    parser.add_argument("--y", required=True, help="Comma-separated output values")
    parser.add_argument("--model", choices=["rbf", "poly"], default="rbf", help="Surrogate type")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        x = parse_list(args.x)
        y = parse_list(args.y)
        result = build_surrogate(x, y, args.model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {"x": x, "y": y, "model": args.model},
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Surrogate summary")
    print(f"  model: {result['model_type']}")
    print(f"  mse: {result['metrics']['mse']:.6g}")


if __name__ == "__main__":
    main()
