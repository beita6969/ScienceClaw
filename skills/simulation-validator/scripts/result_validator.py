#!/usr/bin/env python3
import argparse
import json
import math
import sys
from typing import Dict, List, Optional


def load_metrics(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_metrics(
    metrics: Dict[str, object],
    bound_min: Optional[float],
    bound_max: Optional[float],
    mass_tol: float,
) -> Dict[str, object]:
    checks: Dict[str, bool] = {}
    failed: List[str] = []

    mass_initial = metrics.get("mass_initial")
    mass_final = metrics.get("mass_final")
    if mass_initial is not None and mass_final is not None:
        try:
            drift = abs(float(mass_final) - float(mass_initial)) / max(abs(float(mass_initial)), 1e-12)
            checks["mass_conserved"] = drift <= mass_tol
            if not checks["mass_conserved"]:
                failed.append("mass_conserved")
        except (TypeError, ValueError):
            checks["mass_conserved"] = False
            failed.append("mass_conserved")

    energy_history = metrics.get("energy_history")
    if isinstance(energy_history, list) and energy_history:
        try:
            energies = [float(v) for v in energy_history]
            checks["energy_decreases"] = energies[-1] <= energies[0]
            if not checks["energy_decreases"]:
                failed.append("energy_decreases")
        except (TypeError, ValueError):
            checks["energy_decreases"] = False
            failed.append("energy_decreases")

    field_min = metrics.get("field_min")
    field_max = metrics.get("field_max")
    if bound_min is not None or bound_max is not None:
        ok = True
        if field_min is not None and bound_min is not None:
            ok = ok and (float(field_min) >= bound_min)
        if field_max is not None and bound_max is not None:
            ok = ok and (float(field_max) <= bound_max)
        checks["bounds_satisfied"] = ok
        if not ok:
            failed.append("bounds_satisfied")

    has_nan = metrics.get("has_nan")
    if has_nan is not None:
        checks["no_nan"] = not bool(has_nan)
        if not checks["no_nan"]:
            failed.append("no_nan")

    if not checks:
        checks["no_checks"] = True

    passed = sum(1 for v in checks.values() if v)
    confidence = passed / max(len(checks), 1)

    return {
        "checks": checks,
        "failed_checks": failed,
        "confidence_score": confidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate simulation results from metrics JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON")
    parser.add_argument("--bound-min", type=float, default=None, help="Minimum bound")
    parser.add_argument("--bound-max", type=float, default=None, help="Maximum bound")
    parser.add_argument("--mass-tol", type=float, default=1e-3, help="Mass tolerance")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        metrics = load_metrics(args.metrics)
        result = validate_metrics(
            metrics=metrics,
            bound_min=args.bound_min,
            bound_max=args.bound_max,
            mass_tol=args.mass_tol,
        )
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "metrics": args.metrics,
            "bound_min": args.bound_min,
            "bound_max": args.bound_max,
            "mass_tol": args.mass_tol,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Result validation")
    print(f"  confidence_score: {result['confidence_score']:.6g}")
    for name, status in result["checks"].items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
