#!/usr/bin/env python3
import argparse
import json
import math
import sys
from typing import Dict, Optional


def compute_grid(
    length: float,
    resolution: int,
    dims: int,
    dx: Optional[float],
) -> Dict[str, object]:
    if length <= 0:
        raise ValueError("length must be positive")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if dims <= 0:
        raise ValueError("dims must be positive")

    if dx is None:
        dx = length / resolution
    if dx <= 0:
        raise ValueError("dx must be positive")

    counts = [int(math.ceil(length / dx)) for _ in range(dims)]
    notes = []
    if dx * counts[0] < length:
        notes.append("Grid does not fully cover length; consider smaller dx.")

    return {
        "dx": dx,
        "counts": counts,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate grid spacing and cell counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--length", type=float, required=True, help="Domain length")
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="Target number of cells along length",
    )
    parser.add_argument("--dims", type=int, default=2, help="Dimensions (1,2,3)")
    parser.add_argument("--dx", type=float, default=None, help="Override dx")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = compute_grid(
            length=args.length,
            resolution=args.resolution,
            dims=args.dims,
            dx=args.dx,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "length": args.length,
            "resolution": args.resolution,
            "dims": args.dims,
            "dx": args.dx,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Grid sizing")
    print(f"  dx: {result['dx']:.6g}")
    print(f"  counts: {result['counts']}")
    for note in result["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
