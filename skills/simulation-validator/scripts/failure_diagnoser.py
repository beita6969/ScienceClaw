#!/usr/bin/env python3
import argparse
import json
import re
import sys
from typing import Dict, List


PATTERNS = [
    (re.compile(r"nan|inf|overflow", re.IGNORECASE), "Numerical blow-up", "Reduce dt, tighten tolerances, or increase damping."),
    (re.compile(r"diverg|residual", re.IGNORECASE), "Convergence failure", "Check solver/preconditioner settings and matrix conditioning."),
    (re.compile(r"out of memory|allocation failed", re.IGNORECASE), "Memory exhaustion", "Reduce resolution or enable out-of-core options."),
    (re.compile(r"disk full|permission denied", re.IGNORECASE), "I/O error", "Check disk space and permissions."),
]


def diagnose(log_text: str) -> Dict[str, object]:
    causes: List[str] = []
    fixes: List[str] = []
    for pattern, cause, fix in PATTERNS:
        if pattern.search(log_text):
            causes.append(cause)
            fixes.append(fix)
    if not causes:
        causes.append("Unknown")
        fixes.append("Inspect logs manually and review recent parameter changes.")
    return {"probable_causes": causes, "recommended_fixes": fixes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose failed simulations from logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log", required=True, help="Path to log file")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        with open(args.log, "r", encoding="utf-8") as handle:
            text = handle.read()
        result = diagnose(text)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {"inputs": {"log": args.log}, "results": result}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Failure diagnosis")
    for cause, fix in zip(result["probable_causes"], result["recommended_fixes"]):
        print(f"  cause: {cause}")
        print(f"  fix: {fix}")


if __name__ == "__main__":
    main()
