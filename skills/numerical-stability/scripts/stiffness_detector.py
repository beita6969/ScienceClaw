#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np


def parse_eigs(raw: str) -> np.ndarray:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("eigs must be a comma-separated list")
    return np.array([complex(p) for p in parts], dtype=complex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect stiffness from eigenvalues or a Jacobian matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eigs", help="Comma-separated eigenvalues")
    group.add_argument("--jacobian", help="Path to Jacobian matrix (.npy or text)")
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Delimiter for text Jacobians (default: any whitespace)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e3,
        help="Stiffness ratio threshold",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def load_matrix(path: str, delimiter: Optional[str]) -> np.ndarray:
    _, ext = os.path.splitext(path)
    if ext == ".npy":
        return np.load(path)
    return np.loadtxt(path, delimiter=delimiter)


def compute_stiffness(eigs: np.ndarray, threshold: float) -> Dict[str, object]:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if eigs.size == 0:
        raise ValueError("eigs must be non-empty")
    if not np.all(np.isfinite(eigs)):
        raise ValueError("eigs contain non-finite values")

    abs_eigs = np.abs(eigs)
    nonzero = abs_eigs[abs_eigs > 0]
    ratio = float(np.max(nonzero) / np.min(nonzero)) if nonzero.size else float("inf")
    stiff = ratio >= threshold
    recommendation = "implicit (BDF/Radau)" if stiff else "explicit (RK/Adams)"

    return {
        "stiffness_ratio": ratio,
        "stiff": stiff,
        "recommendation": recommendation,
        "nonzero_count": int(nonzero.size),
        "total_count": int(eigs.size),
    }


def main() -> None:
    args = parse_args()
    try:
        if args.eigs is not None:
            eigs = parse_eigs(args.eigs)
            source = "eigs"
        else:
            if not os.path.exists(args.jacobian):
                print(f"Jacobian not found: {args.jacobian}", file=sys.stderr)
                sys.exit(2)
            jacobian = load_matrix(args.jacobian, args.delimiter)
            if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
                print("Jacobian must be square.", file=sys.stderr)
                sys.exit(2)
            eigs = np.linalg.eigvals(jacobian)
            source = "jacobian"
        results = compute_stiffness(eigs, args.threshold)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "source": source,
            "threshold": args.threshold,
        },
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Stiffness detection")
    print(f"  stiffness ratio: {results['stiffness_ratio']:.6g}")
    print(f"  stiff: {results['stiff']}")
    print(f"  recommendation: {results['recommendation']}")


if __name__ == "__main__":
    main()
