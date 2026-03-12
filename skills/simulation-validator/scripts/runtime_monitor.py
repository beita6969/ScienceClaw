#!/usr/bin/env python3
import argparse
import json
import re
import sys
from typing import Dict, List, Optional, Tuple


def parse_log(
    text: str,
    residual_pattern: str,
    dt_pattern: str,
) -> Tuple[List[float], List[float]]:
    residuals: List[float] = []
    dts: List[float] = []
    res_re = re.compile(residual_pattern, re.IGNORECASE)
    dt_re = re.compile(dt_pattern, re.IGNORECASE)

    for line in text.splitlines():
        res_match = res_re.search(line)
        if res_match:
            try:
                residuals.append(float(res_match.group(1)))
            except ValueError:
                continue
        dt_match = dt_re.search(line)
        if dt_match:
            try:
                dts.append(float(dt_match.group(1)))
            except ValueError:
                continue
    return residuals, dts


def compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "last": None}
    return {"min": min(values), "max": max(values), "last": values[-1]}


def monitor(
    residuals: List[float],
    dts: List[float],
    residual_growth: float,
    dt_drop: float,
) -> Dict[str, object]:
    alerts: List[str] = []
    if residuals:
        for i in range(1, len(residuals)):
            if residuals[i - 1] > 0 and residuals[i] / residuals[i - 1] > residual_growth:
                alerts.append("Residual increased > threshold.")
                break
    if dts:
        if min(dts) > 0 and max(dts) / min(dts) > dt_drop:
            alerts.append("Time step reduced > threshold.")

    return {
        "alerts": alerts,
        "residual_stats": compute_stats(residuals),
        "dt_stats": compute_stats(dts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor runtime logs for convergence and dt issues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log", required=True, help="Path to log file")
    parser.add_argument(
        "--residual-pattern",
        default=r"residual[^0-9eE+\-]*([0-9][0-9eE+\.-]*)",
        help="Regex to capture residual value",
    )
    parser.add_argument(
        "--dt-pattern",
        default=r"dt[^0-9eE+\-]*([0-9][0-9eE+\.-]*)",
        help="Regex to capture dt value",
    )
    parser.add_argument(
        "--residual-growth",
        type=float,
        default=10.0,
        help="Residual growth threshold",
    )
    parser.add_argument(
        "--dt-drop",
        type=float,
        default=100.0,
        help="dt reduction threshold (max/min)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        with open(args.log, "r", encoding="utf-8") as handle:
            text = handle.read()
        residuals, dts = parse_log(text, args.residual_pattern, args.dt_pattern)
        result = monitor(residuals, dts, args.residual_growth, args.dt_drop)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "log": args.log,
            "residual_pattern": args.residual_pattern,
            "dt_pattern": args.dt_pattern,
            "residual_growth": args.residual_growth,
            "dt_drop": args.dt_drop,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Runtime monitor")
    for alert in result["alerts"]:
        print(f"  alert: {alert}")
    print(f"  residual_stats: {result['residual_stats']}")
    print(f"  dt_stats: {result['dt_stats']}")


if __name__ == "__main__":
    main()
