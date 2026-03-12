#!/usr/bin/env python3
import argparse
import json
import math
import sys
from typing import Dict, List


def schedule_outputs(t_start: float, t_end: float, interval: float, max_outputs: int) -> Dict[str, object]:
    if t_end <= t_start:
        raise ValueError("t_end must be greater than t_start")
    if interval <= 0:
        raise ValueError("interval must be positive")
    if max_outputs <= 0:
        raise ValueError("max_outputs must be positive")

    times: List[float] = []
    t = t_start
    while t <= t_end + 1e-12 and len(times) < max_outputs:
        times.append(t)
        t += interval

    return {
        "interval": interval,
        "count": len(times),
        "output_times": times,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate output schedule for a run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--t-start", type=float, required=True, help="Start time")
    parser.add_argument("--t-end", type=float, required=True, help="End time")
    parser.add_argument("--interval", type=float, required=True, help="Output interval")
    parser.add_argument(
        "--max-outputs",
        type=int,
        default=10000,
        help="Maximum outputs to emit",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = schedule_outputs(args.t_start, args.t_end, args.interval, args.max_outputs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "t_start": args.t_start,
            "t_end": args.t_end,
            "interval": args.interval,
            "max_outputs": args.max_outputs,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Output schedule")
    print(f"  interval: {result['interval']:.6g}")
    print(f"  count: {result['count']}")
    if result["output_times"]:
        print(f"  first: {result['output_times'][0]:.6g}")
        print(f"  last: {result['output_times'][-1]:.6g}")


if __name__ == "__main__":
    main()
