#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np


def load_matrix(path: str, delimiter: Optional[str]) -> np.ndarray:
    _, ext = os.path.splitext(path)
    if ext == ".npy":
        return np.load(path)
    return np.loadtxt(path, delimiter=delimiter)


def compute_scaling(
    matrix: np.ndarray,
    symmetry_tol: float,
    symmetric: bool,
) -> Dict[str, object]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains non-finite values")

    m, n = matrix.shape
    if symmetric and m != n:
        raise ValueError("symmetric scaling requires a square matrix")

    abs_matrix = np.abs(matrix)
    row_max = np.max(abs_matrix, axis=1)
    col_max = np.max(abs_matrix, axis=0)

    zero_rows = [int(i) for i, v in enumerate(row_max) if v == 0]
    zero_cols = [int(i) for i, v in enumerate(col_max) if v == 0]

    row_scale = [1.0 / v if v > 0 else 1.0 for v in row_max]
    col_scale = [1.0 / v if v > 0 else 1.0 for v in col_max]

    symmetric_scale = None
    is_symmetric = bool(np.allclose(matrix, matrix.T, atol=symmetry_tol, rtol=0.0))
    if symmetric:
        symmetric_scale = [1.0 / np.sqrt(v) if v > 0 else 1.0 for v in row_max]

    notes: List[str] = []
    if zero_rows:
        notes.append("Zero rows detected; scaling set to 1 for those rows.")
    if zero_cols:
        notes.append("Zero cols detected; scaling set to 1 for those cols.")
    if symmetric and not is_symmetric:
        notes.append("Matrix is not symmetric within tolerance; check inputs.")

    return {
        "shape": [m, n],
        "row_scale": row_scale,
        "col_scale": col_scale,
        "row_scale_min": float(min(row_scale)) if row_scale else 0.0,
        "row_scale_max": float(max(row_scale)) if row_scale else 0.0,
        "col_scale_min": float(min(col_scale)) if col_scale else 0.0,
        "col_scale_max": float(max(col_scale)) if col_scale else 0.0,
        "zero_rows": zero_rows,
        "zero_cols": zero_cols,
        "symmetric_scale": symmetric_scale,
        "symmetric": is_symmetric,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest row/column scaling for matrix equilibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--matrix", required=True, help="Path to matrix file (.npy or text)")
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Delimiter for text matrices (default: any whitespace)",
    )
    parser.add_argument(
        "--symmetry-tol",
        type=float,
        default=1e-8,
        help="Tolerance for symmetry check",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Request symmetric scaling (uses row max norms)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.matrix):
        print(f"Matrix not found: {args.matrix}", file=sys.stderr)
        sys.exit(2)

    try:
        matrix = load_matrix(args.matrix, args.delimiter)
        results = compute_scaling(matrix, args.symmetry_tol, args.symmetric)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "matrix": args.matrix,
            "symmetry_tol": args.symmetry_tol,
            "symmetric": args.symmetric,
        },
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Scaling equilibration")
    print(f"  shape: {results['shape']}")
    print(f"  row_scale_min: {results['row_scale_min']:.6g}")
    print(f"  row_scale_max: {results['row_scale_max']:.6g}")
    print(f"  col_scale_min: {results['col_scale_min']:.6g}")
    print(f"  col_scale_max: {results['col_scale_max']:.6g}")
    if results["symmetric_scale"] is not None:
        print("  symmetric_scale: provided")
    for note in results["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
