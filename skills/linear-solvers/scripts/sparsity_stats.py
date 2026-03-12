#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Optional

import numpy as np


def load_matrix(path: str, delimiter: Optional[str]) -> np.ndarray:
    _, ext = os.path.splitext(path)
    if ext == ".npy":
        return np.load(path)
    return np.loadtxt(path, delimiter=delimiter)


def compute_stats(matrix: np.ndarray, symmetry_tol: float) -> dict:
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains non-finite values")

    m, n = matrix.shape
    nnz = int(np.count_nonzero(matrix))
    density = float(nnz) / float(m * n) if m * n > 0 else 0.0

    # bandwidth: max |i-j| where A_ij != 0
    rows, cols = np.nonzero(matrix)
    if rows.size:
        bandwidth = int(np.max(np.abs(rows - cols)))
    else:
        bandwidth = 0

    symmetric = bool(np.allclose(matrix, matrix.T, atol=symmetry_tol, rtol=0.0))
    return {
        "shape": [m, n],
        "nnz": nnz,
        "density": density,
        "bandwidth": bandwidth,
        "symmetry": symmetric,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute sparsity statistics for a matrix.",
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
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.matrix):
        print(f"Matrix not found: {args.matrix}", file=sys.stderr)
        sys.exit(2)

    try:
        matrix = load_matrix(args.matrix, args.delimiter)
        results = compute_stats(matrix, args.symmetry_tol)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "matrix": args.matrix,
            "symmetry_tol": args.symmetry_tol,
        },
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Sparsity stats")
    print(f"  shape: {results['shape']}")
    print(f"  nnz: {results['nnz']}")
    print(f"  density: {results['density']:.6g}")
    print(f"  bandwidth: {results['bandwidth']}")
    print(f"  symmetric: {results['symmetry']}")


if __name__ == "__main__":
    main()
