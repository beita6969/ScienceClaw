#!/usr/bin/env python3
import argparse
import json
import math
import sys
from typing import Dict, List, Optional


def compute_ramp(
    dt_start: float,
    dt_target: float,
    steps: int,
    kind: str,
) -> List[float]:
    if steps <= 0:
        return []
    if kind == "linear":
        return [
            dt_start + (dt_target - dt_start) * (i + 1) / steps for i in range(steps)
        ]
    if kind == "geometric":
        if dt_start <= 0 or dt_target <= 0:
            raise ValueError("dt_start and dt_target must be positive for geometric ramp")
        ratio = (dt_target / dt_start) ** (1.0 / steps)
        return [dt_start * (ratio ** (i + 1)) for i in range(steps)]
    raise ValueError("ramp_kind must be linear or geometric")


def plan_timestep(
    dt_target: float,
    dt_limit: float,
    safety: float,
    dt_min: Optional[float],
    dt_max: Optional[float],
    ramp_steps: int,
    ramp_kind: str,
    preview_steps: int,
) -> Dict[str, object]:
    if dt_target <= 0 or dt_limit <= 0:
        raise ValueError("dt_target and dt_limit must be positive")
    if safety <= 0:
        raise ValueError("safety must be positive")
    if dt_min is not None and dt_min <= 0:
        raise ValueError("dt_min must be positive")
    if dt_max is not None and dt_max <= 0:
        raise ValueError("dt_max must be positive")
    if dt_min is not None and dt_max is not None and dt_min > dt_max:
        raise ValueError("dt_min must be <= dt_max")

    dt_recommended = min(dt_target, dt_limit) * safety
    if dt_min is not None:
        dt_recommended = max(dt_recommended, dt_min)
    if dt_max is not None:
        dt_recommended = min(dt_recommended, dt_max)

    notes: List[str] = []
    if dt_recommended < dt_target * safety:
        notes.append("Recommended dt reduced by stability limit.")
    if dt_min is not None and dt_recommended == dt_min:
        notes.append("Recommended dt hits minimum limit.")
    if dt_max is not None and dt_recommended == dt_max:
        notes.append("Recommended dt hits maximum limit.")

    ramp_schedule = []
    if ramp_steps > 0:
        dt_start = dt_recommended / max(ramp_steps, 1)
        ramp_schedule = compute_ramp(dt_start, dt_recommended, ramp_steps, ramp_kind)
        if preview_steps > 0:
            ramp_schedule = ramp_schedule[:preview_steps]

    return {
        "dt_limit": dt_limit,
        "dt_recommended": dt_recommended,
        "ramp_schedule": ramp_schedule,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan time-step based on limits and ramping strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dt-target", type=float, required=True, help="Target dt")
    parser.add_argument("--dt-limit", type=float, required=True, help="Stability dt limit")
    parser.add_argument("--safety", type=float, default=1.0, help="Safety factor")
    parser.add_argument("--dt-min", type=float, default=None, help="Minimum dt")
    parser.add_argument("--dt-max", type=float, default=None, help="Maximum dt")
    parser.add_argument("--ramp-steps", type=int, default=0, help="Number of ramp steps")
    parser.add_argument(
        "--ramp-kind",
        choices=["linear", "geometric"],
        default="linear",
        help="Ramp strategy",
    )
    parser.add_argument(
        "--preview-steps",
        type=int,
        default=10,
        help="Number of ramp steps to preview in output",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = plan_timestep(
            dt_target=args.dt_target,
            dt_limit=args.dt_limit,
            safety=args.safety,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
            ramp_steps=args.ramp_steps,
            ramp_kind=args.ramp_kind,
            preview_steps=args.preview_steps,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "dt_target": args.dt_target,
            "dt_limit": args.dt_limit,
            "safety": args.safety,
            "dt_min": args.dt_min,
            "dt_max": args.dt_max,
            "ramp_steps": args.ramp_steps,
            "ramp_kind": args.ramp_kind,
            "preview_steps": args.preview_steps,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Timestep plan")
    print(f"  dt_limit: {result['dt_limit']:.6g}")
    print(f"  dt_recommended: {result['dt_recommended']:.6g}")
    if result["ramp_schedule"]:
        print(f"  ramp_schedule: {result['ramp_schedule']}")
    for note in result["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
