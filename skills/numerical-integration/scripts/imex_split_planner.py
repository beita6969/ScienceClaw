#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Dict, List


def parse_terms(raw: str) -> List[str]:
    if raw is None:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts


def plan_imex(
    stiff_terms: List[str],
    nonstiff_terms: List[str],
    coupling: str,
    accuracy: str,
    stiffness_ratio: float,
    conservative: bool,
) -> Dict[str, object]:
    if coupling not in {"weak", "moderate", "strong"}:
        raise ValueError("coupling must be weak, moderate, or strong")
    if accuracy not in {"low", "medium", "high"}:
        raise ValueError("accuracy must be low, medium, or high")
    if stiffness_ratio <= 0:
        raise ValueError("stiffness_ratio must be positive")
    if not stiff_terms and not nonstiff_terms:
        raise ValueError("Provide at least one stiff or non-stiff term")

    implicit_terms = stiff_terms
    explicit_terms = nonstiff_terms

    notes: List[str] = []
    recommended: List[str] = []
    if stiff_terms and nonstiff_terms:
        recommended.append("IMEX-ARK")
        recommended.append("SBDF (semi-implicit BDF)")
    elif stiff_terms:
        recommended.append("BDF")
        recommended.append("Radau IIA")
    else:
        recommended.append("RK45")
        recommended.append("DOP853")

    if conservative:
        notes.append("Preserve conserved quantities; avoid overly aggressive splitting.")

    if coupling == "strong" or stiffness_ratio >= 1e4:
        splitting = "imex-coupled"
        notes.append("Strong coupling: avoid loose operator splitting.")
    elif accuracy == "high":
        splitting = "strang"
        notes.append("Strang splitting for higher accuracy.")
    else:
        splitting = "lie"
        notes.append("Lie splitting for efficiency.")

    return {
        "implicit_terms": implicit_terms,
        "explicit_terms": explicit_terms,
        "recommended_integrator": recommended,
        "splitting_strategy": splitting,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan IMEX split for stiff/non-stiff coupling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stiff-terms", default=None, help="Comma-separated stiff terms")
    parser.add_argument("--nonstiff-terms", default=None, help="Comma-separated non-stiff terms")
    parser.add_argument(
        "--coupling",
        choices=["weak", "moderate", "strong"],
        default="moderate",
        help="Coupling strength between operators",
    )
    parser.add_argument(
        "--accuracy",
        choices=["low", "medium", "high"],
        default="medium",
        help="Desired accuracy level",
    )
    parser.add_argument(
        "--stiffness-ratio",
        type=float,
        default=1e3,
        help="Estimated stiffness ratio",
    )
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="Preserve conserved quantities",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = plan_imex(
            stiff_terms=parse_terms(args.stiff_terms),
            nonstiff_terms=parse_terms(args.nonstiff_terms),
            coupling=args.coupling,
            accuracy=args.accuracy,
            stiffness_ratio=args.stiffness_ratio,
            conservative=args.conservative,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "stiff_terms": parse_terms(args.stiff_terms),
            "nonstiff_terms": parse_terms(args.nonstiff_terms),
            "coupling": args.coupling,
            "accuracy": args.accuracy,
            "stiffness_ratio": args.stiffness_ratio,
            "conservative": args.conservative,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("IMEX split plan")
    print(f"  implicit_terms: {', '.join(result['implicit_terms']) or 'none'}")
    print(f"  explicit_terms: {', '.join(result['explicit_terms']) or 'none'}")
    print(f"  splitting_strategy: {result['splitting_strategy']}")
    print(f"  recommended_integrator: {', '.join(result['recommended_integrator'])}")
    for note in result["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
