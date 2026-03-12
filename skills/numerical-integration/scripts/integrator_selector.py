#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Dict, List


def select_integrator(
    stiff: bool,
    oscillatory: bool,
    event_detection: bool,
    jacobian_available: bool,
    implicit_allowed: bool,
    accuracy: str,
    dimension: int,
    low_memory: bool,
) -> Dict[str, List[str] | str]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if accuracy not in {"low", "medium", "high"}:
        raise ValueError("accuracy must be low, medium, or high")

    recommended: List[str] = []
    alternatives: List[str] = []
    notes: List[str] = []

    if stiff:
        if implicit_allowed or jacobian_available:
            recommended.extend(["BDF", "Radau IIA"])
            if jacobian_available:
                recommended.append("Rosenbrock")
            else:
                alternatives.append("Rosenbrock (needs Jacobian)")
        else:
            recommended.extend(["IMEX", "RK-Chebyshev"])
            notes.append("Stiff problem without implicit solves; expect smaller dt.")
    else:
        if oscillatory:
            recommended.extend(["Symplectic Verlet", "Stormer-Verlet"])
            alternatives.append("RK45 (if invariants are not critical)")
        else:
            recommended.append("RK45")
            if accuracy == "high":
                alternatives.append("DOP853")
            elif accuracy == "low":
                alternatives.append("RK23")

    if event_detection:
        notes.append("Prefer methods with dense output for event detection.")
        if "RK45" in recommended:
            alternatives.append("RK45 (dense output)")
        else:
            alternatives.append("DOP853 (dense output)")

    if low_memory or dimension > 1_000_000:
        notes.append("Large state: consider low-storage RK or linearly implicit methods.")
        alternatives.append("Low-storage RK")

    if stiff and not jacobian_available:
        notes.append("Provide Jacobian or Jacobian-vector products for efficiency.")

    if accuracy == "high":
        notes.append("Tight tolerances required; expect smaller dt and higher cost.")

    return {
        "recommended": recommended,
        "alternatives": alternatives,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a suitable time integrator based on problem characteristics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stiff", action="store_true", help="Treat the problem as stiff")
    parser.add_argument(
        "--oscillatory",
        action="store_true",
        help="System exhibits oscillatory dynamics",
    )
    parser.add_argument(
        "--event-detection",
        action="store_true",
        help="Events or root finding required",
    )
    parser.add_argument(
        "--jacobian-available",
        action="store_true",
        help="Jacobian or Jv product is available",
    )
    parser.add_argument(
        "--implicit-allowed",
        action="store_true",
        help="Implicit solves are feasible",
    )
    parser.add_argument(
        "--accuracy",
        choices=["low", "medium", "high"],
        default="medium",
        help="Desired accuracy level",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1,
        help="State dimension (for memory considerations)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Prefer low-memory methods",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = select_integrator(
            stiff=args.stiff,
            oscillatory=args.oscillatory,
            event_detection=args.event_detection,
            jacobian_available=args.jacobian_available,
            implicit_allowed=args.implicit_allowed,
            accuracy=args.accuracy,
            dimension=args.dimension,
            low_memory=args.low_memory,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "stiff": args.stiff,
            "oscillatory": args.oscillatory,
            "event_detection": args.event_detection,
            "jacobian_available": args.jacobian_available,
            "implicit_allowed": args.implicit_allowed,
            "accuracy": args.accuracy,
            "dimension": args.dimension,
            "low_memory": args.low_memory,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Integrator selection")
    print(f"  recommended: {', '.join(result['recommended'])}")
    if result["alternatives"]:
        print(f"  alternatives: {', '.join(result['alternatives'])}")
    for note in result["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
