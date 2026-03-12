#!/usr/bin/env python3
import argparse
import json
import math
import sys
from typing import Dict


def estimate_error(
    dt: float,
    scheme: str,
    commutator_norm: float,
    target_error: float,
) -> Dict[str, object]:
    if dt <= 0:
        raise ValueError("dt must be positive")
    if commutator_norm < 0:
        raise ValueError("commutator_norm must be non-negative")
    if scheme not in {"lie", "strang"}:
        raise ValueError("scheme must be lie or strang")
    if target_error < 0:
        raise ValueError("target_error must be non-negative")

    order = 1 if scheme == "lie" else 2
    error_est = commutator_norm * (dt ** (order + 1))

    substeps = 1
    dt_effective = dt
    if target_error > 0 and error_est > target_error:
        ratio = error_est / target_error
        substeps = int(math.ceil(ratio ** (1.0 / (order + 1))))
        substeps = max(substeps, 1)
        dt_effective = dt / substeps
        error_est = commutator_norm * (dt_effective ** (order + 1))

    return {
        "scheme": scheme,
        "order": order,
        "error_estimate": error_est,
        "dt_effective": dt_effective,
        "substeps": substeps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate operator splitting error and suggest substeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dt", type=float, required=True, help="Base time step")
    parser.add_argument(
        "--scheme",
        choices=["lie", "strang"],
        default="strang",
        help="Splitting scheme",
    )
    parser.add_argument(
        "--commutator-norm",
        type=float,
        required=True,
        help="Estimated commutator norm",
    )
    parser.add_argument(
        "--target-error",
        type=float,
        default=0.0,
        help="Target splitting error (optional)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = estimate_error(
            dt=args.dt,
            scheme=args.scheme,
            commutator_norm=args.commutator_norm,
            target_error=args.target_error,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "dt": args.dt,
            "scheme": args.scheme,
            "commutator_norm": args.commutator_norm,
            "target_error": args.target_error,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Splitting error estimate")
    print(f"  scheme: {result['scheme']}")
    print(f"  order: {result['order']}")
    print(f"  error_estimate: {result['error_estimate']:.6g}")
    print(f"  dt_effective: {result['dt_effective']:.6g}")
    print(f"  substeps: {result['substeps']}")


if __name__ == "__main__":
    main()
