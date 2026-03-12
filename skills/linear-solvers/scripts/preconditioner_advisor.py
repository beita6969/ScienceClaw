#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Dict, List


def advise_preconditioner(
    matrix_type: str,
    sparse: bool,
    ill_conditioned: bool,
    saddle_point: bool,
    symmetric: bool,
) -> Dict[str, List[str] | str]:
    if matrix_type not in {"spd", "symmetric-indefinite", "nonsymmetric"}:
        raise ValueError("matrix_type must be spd, symmetric-indefinite, or nonsymmetric")

    suggested: List[str] = []
    notes: List[str] = []

    if saddle_point:
        suggested.append("Block preconditioner (Schur complement)")
        notes.append("Use physics-informed block structure when available.")
        return {"suggested": suggested, "notes": notes}

    if matrix_type == "spd":
        suggested.extend(["Incomplete Cholesky (IC)", "AMG"])
    elif matrix_type == "symmetric-indefinite":
        suggested.extend(["Incomplete LDL^T", "AMG"])
    else:
        suggested.extend(["ILU(0)/ILUT", "AMG"])

    if not sparse:
        notes.append("Dense systems: direct solver or dense preconditioner may be better.")
    if ill_conditioned:
        notes.append("Increase fill-in or use multigrid for robustness.")
    if symmetric and matrix_type == "nonsymmetric":
        notes.append("Check symmetry assumption; matrix_type may be incorrect.")

    return {"suggested": suggested, "notes": notes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest preconditioners for a linear system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matrix-type",
        required=True,
        choices=["spd", "symmetric-indefinite", "nonsymmetric"],
        help="Matrix type",
    )
    parser.add_argument("--sparse", action="store_true", help="Matrix is sparse")
    parser.add_argument(
        "--ill-conditioned",
        action="store_true",
        help="Matrix is ill-conditioned",
    )
    parser.add_argument(
        "--saddle-point",
        action="store_true",
        help="Saddle-point system",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Matrix is symmetric (sanity check)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = advise_preconditioner(
            matrix_type=args.matrix_type,
            sparse=args.sparse,
            ill_conditioned=args.ill_conditioned,
            saddle_point=args.saddle_point,
            symmetric=args.symmetric,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "matrix_type": args.matrix_type,
            "sparse": args.sparse,
            "ill_conditioned": args.ill_conditioned,
            "saddle_point": args.saddle_point,
            "symmetric": args.symmetric,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Preconditioner advice")
    print(f"  suggested: {', '.join(result['suggested'])}")
    for note in result["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
