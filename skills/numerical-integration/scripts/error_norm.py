#!/usr/bin/env python3
import argparse
import json
import math
import sys
from typing import List, Optional, Tuple


def parse_list(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("value list must be a comma-separated list")
    return [float(p) for p in parts]


def compute_error_norm(
    error: List[float],
    solution: Optional[List[float]],
    scale: Optional[List[float]],
    rtol: float,
    atol: float,
    norm: str,
    min_scale: float,
) -> Tuple[float, float, float, float]:
    if not error:
        raise ValueError("error list must be non-empty")
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if min_scale < 0:
        raise ValueError("min_scale must be non-negative")
    if norm not in {"rms", "inf"}:
        raise ValueError("norm must be 'rms' or 'inf'")

    if scale is None:
        if solution is None:
            raise ValueError("solution or scale must be provided")
        if len(solution) != len(error):
            raise ValueError("solution length must match error length")
        scale = [max(min_scale, atol + rtol * abs(y)) for y in solution]
    else:
        if len(scale) != len(error):
            raise ValueError("scale length must match error length")
        if any(s <= 0 for s in scale):
            raise ValueError("scale values must be positive")

    scaled = [e / s for e, s in zip(error, scale)]
    abs_scaled = [abs(v) for v in scaled]
    if norm == "inf":
        error_norm = max(abs_scaled)
    else:
        error_norm = math.sqrt(sum(v * v for v in scaled) / len(scaled))

    return (
        error_norm,
        max(abs_scaled),
        min(scale),
        max(scale),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute scaled error norm for adaptive time stepping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--error", required=True, help="Comma-separated error values")
    parser.add_argument(
        "--solution",
        default=None,
        help="Comma-separated solution values (for scaling)",
    )
    parser.add_argument(
        "--scale",
        default=None,
        help="Comma-separated scale values (overrides solution-based scaling)",
    )
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    parser.add_argument(
        "--norm",
        choices=["rms", "inf"],
        default="rms",
        help="Error norm type",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.0,
        help="Lower bound for scale values",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        error = parse_list(args.error)
        solution = parse_list(args.solution) if args.solution is not None else None
        scale = parse_list(args.scale) if args.scale is not None else None
        error_norm, max_component, scale_min, scale_max = compute_error_norm(
            error=error,
            solution=solution,
            scale=scale,
            rtol=args.rtol,
            atol=args.atol,
            norm=args.norm,
            min_scale=args.min_scale,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "error": error,
            "solution": solution,
            "scale": scale,
            "rtol": args.rtol,
            "atol": args.atol,
            "norm": args.norm,
            "min_scale": args.min_scale,
        },
        "results": {
            "error_norm": error_norm,
            "max_component": max_component,
            "scale_min": scale_min,
            "scale_max": scale_max,
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Error norm")
    print(f"  norm: {args.norm}")
    print(f"  error_norm: {error_norm:.6g}")
    print(f"  max_component: {max_component:.6g}")
    print(f"  scale_min: {scale_min:.6g}")
    print(f"  scale_max: {scale_max:.6g}")


if __name__ == "__main__":
    main()
