#!/usr/bin/env python3
import argparse
import json
import math
import sys
from typing import Dict, List, Optional, Tuple


def parse_list(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("value list must be a comma-separated list")
    return [float(p) for p in parts]


def compute_norms(vec: List[float]) -> Dict[str, float]:
    if not vec:
        raise ValueError("vector must be non-empty")
    if any(not math.isfinite(v) for v in vec):
        raise ValueError("vector contains non-finite values")
    l1 = sum(abs(v) for v in vec)
    l2 = math.sqrt(sum(v * v for v in vec))
    linf = max(abs(v) for v in vec)
    return {"l1": l1, "l2": l2, "linf": linf}


def select_norm_value(norms: Dict[str, float], norm: str) -> float:
    if norm == "l1":
        return norms["l1"]
    if norm == "l2":
        return norms["l2"]
    if norm == "inf":
        return norms["linf"]
    raise ValueError("norm must be l1, l2, or inf")


def compute_residual_metrics(
    residual: List[float],
    rhs: Optional[List[float]],
    initial: Optional[List[float]],
    abs_tol: float,
    rel_tol: float,
    norm: str,
    require_both: bool,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]], Optional[Dict[str, float]], Dict[str, object]]:
    if abs_tol < 0 or rel_tol < 0:
        raise ValueError("abs_tol and rel_tol must be non-negative")

    residual_norms = compute_norms(residual)

    reference = None
    note = None
    if rhs is not None:
        reference = rhs
        if initial is not None:
            note = "Using rhs as reference; initial ignored."
    elif initial is not None:
        reference = initial

    reference_norms = compute_norms(reference) if reference is not None else None
    relative_norms = None
    if reference_norms is not None:
        relative_norms = {
            key: residual_norms[key] / reference_norms[key]
            if reference_norms[key] != 0
            else float("inf")
            for key in residual_norms
        }

    norm_value = select_norm_value(residual_norms, norm)
    ref_value = select_norm_value(reference_norms, norm) if reference_norms else None
    rel_value = norm_value / ref_value if ref_value not in (None, 0) else None

    converged_abs = norm_value <= abs_tol
    converged_rel = None if rel_value is None else rel_value <= rel_tol
    if converged_rel is None:
        converged = converged_abs
    else:
        converged = converged_abs and converged_rel if require_both else converged_abs or converged_rel

    meta = {
        "norm_used": norm,
        "norm_value": norm_value,
        "reference_value": ref_value,
        "relative_value": rel_value,
        "converged_abs": converged_abs,
        "converged_rel": converged_rel,
        "converged": converged,
        "note": note,
    }
    return residual_norms, reference_norms, relative_norms, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute residual norms and evaluate stopping criteria.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--residual", required=True, help="Comma-separated residual values")
    parser.add_argument("--rhs", default=None, help="Comma-separated RHS values")
    parser.add_argument("--initial", default=None, help="Comma-separated initial residual values")
    parser.add_argument("--abs-tol", type=float, default=1e-8, help="Absolute tolerance")
    parser.add_argument("--rel-tol", type=float, default=1e-6, help="Relative tolerance")
    parser.add_argument(
        "--norm",
        choices=["l1", "l2", "inf"],
        default="l2",
        help="Norm for convergence check",
    )
    parser.add_argument(
        "--require-both",
        action="store_true",
        help="Require both abs and rel criteria for convergence",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        residual = parse_list(args.residual)
        rhs = parse_list(args.rhs) if args.rhs is not None else None
        initial = parse_list(args.initial) if args.initial is not None else None
        residual_norms, reference_norms, relative_norms, meta = compute_residual_metrics(
            residual=residual,
            rhs=rhs,
            initial=initial,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            norm=args.norm,
            require_both=args.require_both,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "residual": residual,
            "rhs": rhs,
            "initial": initial,
            "abs_tol": args.abs_tol,
            "rel_tol": args.rel_tol,
            "norm": args.norm,
            "require_both": args.require_both,
        },
        "results": {
            "residual_norms": residual_norms,
            "reference_norms": reference_norms,
            "relative_norms": relative_norms,
            "meta": meta,
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Residual norms")
    print(f"  norm: {meta['norm_used']}")
    print(f"  norm_value: {meta['norm_value']:.6g}")
    if meta["relative_value"] is not None:
        print(f"  relative_value: {meta['relative_value']:.6g}")
    print(f"  converged: {meta['converged']}")
    if meta["note"]:
        print(f"  note: {meta['note']}")


if __name__ == "__main__":
    main()
