#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Dict, List


def select_solver(
    symmetric: bool,
    positive_definite: bool,
    sparse: bool,
    size: int,
    nearly_symmetric: bool,
    ill_conditioned: bool,
    complex_valued: bool,
    memory_limited: bool,
) -> Dict[str, List[str] | str]:
    if size <= 0:
        raise ValueError("size must be positive")

    recommended: List[str] = []
    alternatives: List[str] = []
    notes: List[str] = []

    large = size >= 200_000
    if symmetric:
        if positive_definite:
            if sparse or large or memory_limited:
                recommended.append("CG")
                alternatives.append("MINRES")
                notes.append("Use IC/AMG preconditioning for SPD systems.")
            else:
                recommended.append("Cholesky")
                alternatives.append("CG")
        else:
            recommended.append("MINRES")
            alternatives.append("SYMMLQ")
            notes.append("Indefinite symmetric system; avoid CG.")
    else:
        if nearly_symmetric:
            recommended.append("BiCGSTAB")
            alternatives.append("GMRES")
        else:
            recommended.append("GMRES (restarted)")
            alternatives.append("BiCGSTAB")
            notes.append("Use ILU/AMG preconditioning for nonsymmetric systems.")

    if complex_valued:
        notes.append("Ensure solver supports complex arithmetic.")
    if ill_conditioned:
        notes.append("Consider scaling/equilibration and stronger preconditioning.")

    if memory_limited and "GMRES (restarted)" in recommended:
        notes.append("Restarted GMRES reduces memory at the cost of robustness.")

    return {
        "recommended": recommended,
        "alternatives": alternatives,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a linear solver based on matrix properties.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symmetric", action="store_true", help="Matrix is symmetric")
    parser.add_argument(
        "--positive-definite",
        action="store_true",
        help="Matrix is positive definite",
    )
    parser.add_argument("--sparse", action="store_true", help="Matrix is sparse")
    parser.add_argument("--size", type=int, required=True, help="Matrix size (n)")
    parser.add_argument(
        "--nearly-symmetric",
        action="store_true",
        help="Matrix is nearly symmetric",
    )
    parser.add_argument(
        "--ill-conditioned",
        action="store_true",
        help="Matrix is ill-conditioned",
    )
    parser.add_argument(
        "--complex-valued",
        action="store_true",
        help="Matrix has complex entries",
    )
    parser.add_argument(
        "--memory-limited",
        action="store_true",
        help="Memory is constrained",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = select_solver(
            symmetric=args.symmetric,
            positive_definite=args.positive_definite,
            sparse=args.sparse,
            size=args.size,
            nearly_symmetric=args.nearly_symmetric,
            ill_conditioned=args.ill_conditioned,
            complex_valued=args.complex_valued,
            memory_limited=args.memory_limited,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "symmetric": args.symmetric,
            "positive_definite": args.positive_definite,
            "sparse": args.sparse,
            "size": args.size,
            "nearly_symmetric": args.nearly_symmetric,
            "ill_conditioned": args.ill_conditioned,
            "complex_valued": args.complex_valued,
            "memory_limited": args.memory_limited,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Solver selection")
    print(f"  recommended: {', '.join(result['recommended'])}")
    if result["alternatives"]:
        print(f"  alternatives: {', '.join(result['alternatives'])}")
    for note in result["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
