#!/usr/bin/env python3
"""Analyze Jacobian matrix quality for nonlinear solvers."""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np


def diagnose_jacobian(
    matrix: np.ndarray,
    finite_diff_matrix: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Analyze Jacobian matrix quality.

    Args:
        matrix: The Jacobian matrix to analyze
        finite_diff_matrix: Optional finite-difference approximation for comparison
        tolerance: Tolerance for rank deficiency detection

    Returns:
        Dictionary with Jacobian diagnostics
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    if matrix.size == 0:
        raise ValueError("matrix must not be empty")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    m, n = matrix.shape
    notes: List[str] = []

    # Compute SVD for condition number and rank analysis
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "shape": [m, n],
            "condition_number": float("inf"),
            "rank_deficient": True,
            "estimated_rank": 0,
            "singular_value_min": 0.0,
            "singular_value_max": 0.0,
            "jacobian_quality": "singular",
            "finite_diff_error": None,
            "notes": ["SVD computation failed; matrix may be ill-formed."],
        }

    sv_max = float(singular_values[0])
    sv_min = float(singular_values[-1])

    # Condition number
    if sv_min > 1e-30:
        condition_number = sv_max / sv_min
    else:
        condition_number = float("inf")

    # Estimate numerical rank
    rank_tol = max(m, n) * sv_max * np.finfo(float).eps
    estimated_rank = int(np.sum(singular_values > rank_tol))
    rank_deficient = estimated_rank < min(m, n)

    # Classify Jacobian quality
    if condition_number == float("inf") or sv_min < 1e-14:
        jacobian_quality = "near-singular"
        notes.append("Near-singular Jacobian; regularization may be needed.")
    elif condition_number > 1e10:
        jacobian_quality = "ill-conditioned"
        notes.append("Highly ill-conditioned; use iterative refinement or scaling.")
    elif condition_number > 1e6:
        jacobian_quality = "moderately-conditioned"
        notes.append("Moderate conditioning; standard methods should work.")
    else:
        jacobian_quality = "good"
        notes.append("Well-conditioned Jacobian.")

    if rank_deficient:
        notes.append(f"Rank deficient: estimated rank {estimated_rank} < min({m}, {n}).")

    # Compare with finite difference approximation if provided
    finite_diff_error = None
    if finite_diff_matrix is not None:
        if finite_diff_matrix.shape != matrix.shape:
            notes.append("Finite-diff matrix shape mismatch; skipping comparison.")
        else:
            diff = matrix - finite_diff_matrix
            relative_error = np.linalg.norm(diff) / (np.linalg.norm(matrix) + 1e-30)
            finite_diff_error = float(relative_error)

            if relative_error > 0.1:
                notes.append(f"Large discrepancy with finite-diff ({relative_error:.2e}); check analytic Jacobian.")
            elif relative_error > 0.01:
                notes.append(f"Moderate discrepancy with finite-diff ({relative_error:.2e}).")
            else:
                notes.append("Jacobian matches finite-difference approximation well.")

    return {
        "shape": [m, n],
        "condition_number": condition_number,
        "rank_deficient": rank_deficient,
        "estimated_rank": estimated_rank,
        "singular_value_min": sv_min,
        "singular_value_max": sv_max,
        "jacobian_quality": jacobian_quality,
        "finite_diff_error": finite_diff_error,
        "notes": notes,
    }


def load_matrix(path: str) -> np.ndarray:
    """Load matrix from text file."""
    try:
        return np.loadtxt(path)
    except Exception as e:
        raise ValueError(f"Failed to load matrix from {path}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Jacobian matrix quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matrix",
        type=str,
        required=True,
        help="Path to Jacobian matrix file (text format)",
    )
    parser.add_argument(
        "--finite-diff-matrix",
        type=str,
        default=None,
        help="Path to finite-difference Jacobian for comparison",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for rank deficiency detection",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        matrix = load_matrix(args.matrix)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    finite_diff_matrix = None
    if args.finite_diff_matrix:
        try:
            finite_diff_matrix = load_matrix(args.finite_diff_matrix)
            if finite_diff_matrix.ndim == 1:
                finite_diff_matrix = finite_diff_matrix.reshape(1, -1)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)

    try:
        result = diagnose_jacobian(
            matrix=matrix,
            finite_diff_matrix=finite_diff_matrix,
            tolerance=args.tolerance,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload: Dict[str, Any] = {
        "inputs": {
            "matrix": args.matrix,
            "finite_diff_matrix": args.finite_diff_matrix,
            "tolerance": args.tolerance,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Jacobian diagnostics")
    print(f"  shape: {result['shape']}")
    print(f"  condition_number: {result['condition_number']:.2e}")
    print(f"  rank_deficient: {result['rank_deficient']}")
    print(f"  estimated_rank: {result['estimated_rank']}")
    print(f"  singular_value_min: {result['singular_value_min']:.2e}")
    print(f"  singular_value_max: {result['singular_value_max']:.2e}")
    print(f"  jacobian_quality: {result['jacobian_quality']}")
    if result["finite_diff_error"] is not None:
        print(f"  finite_diff_error: {result['finite_diff_error']:.2e}")
    for note in result["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
