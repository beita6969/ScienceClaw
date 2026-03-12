#!/usr/bin/env python3
"""Analyze residual history to classify convergence type."""
import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional


def analyze_convergence(
    residuals: List[float],
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    """Analyze residual history to classify convergence behavior.

    Args:
        residuals: List of residual norms from solver iterations
        tolerance: Convergence tolerance

    Returns:
        Dictionary with convergence analysis results
    """
    if not residuals:
        raise ValueError("residuals must not be empty")

    if any(r < 0 for r in residuals):
        raise ValueError("residuals must be non-negative")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    n = len(residuals)
    final_residual = residuals[-1]
    converged = final_residual <= tolerance

    # Handle single residual case
    if n == 1:
        return {
            "converged": converged,
            "iterations": n,
            "final_residual": final_residual,
            "convergence_type": "unknown",
            "estimated_rate": None,
            "diagnosis": "Insufficient data for convergence analysis.",
            "recommended_action": "Continue iterations to gather more data.",
        }

    # Check for divergence
    if n >= 2 and residuals[-1] > residuals[0] * 1.1:
        return {
            "converged": False,
            "iterations": n,
            "final_residual": final_residual,
            "convergence_type": "diverged",
            "estimated_rate": None,
            "diagnosis": "Residuals are increasing; solver is diverging.",
            "recommended_action": "Reduce step size, add damping, or check Jacobian accuracy.",
        }

    # Check for stagnation
    if n >= 3:
        recent = residuals[-3:]
        if len(recent) == 3:
            rel_change = abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-30)
            if rel_change < 0.01 and not converged:
                return {
                    "converged": False,
                    "iterations": n,
                    "final_residual": final_residual,
                    "convergence_type": "stagnated",
                    "estimated_rate": None,
                    "diagnosis": "Residual has stagnated without reaching tolerance.",
                    "recommended_action": "Improve preconditioner, check for near-singularity, or use different solver.",
                }

    # Estimate convergence rate from log-residuals
    rates = []
    for i in range(1, n):
        if residuals[i - 1] > 1e-30 and residuals[i] > 1e-30:
            ratio = residuals[i] / residuals[i - 1]
            if ratio > 0 and ratio < 1:
                rates.append(ratio)

    if not rates:
        convergence_type = "unknown"
        estimated_rate = None
    else:
        avg_rate = sum(rates) / len(rates)
        estimated_rate = avg_rate

        # Try to detect quadratic convergence by looking at log-log behavior
        # For quadratic: log(r_{k+1}) ≈ 2 * log(r_k)
        if n >= 4 and all(r > 1e-30 for r in residuals):
            log_residuals = [math.log10(r) for r in residuals if r > 0]
            if len(log_residuals) >= 4:
                # Check if log(r_{k+1}) / log(r_k) ≈ 2 for quadratic
                log_ratios = []
                for i in range(1, len(log_residuals)):
                    if abs(log_residuals[i - 1]) > 0.1:
                        log_ratios.append(log_residuals[i] / log_residuals[i - 1])

                if log_ratios and all(1.5 < r < 2.5 for r in log_ratios[-3:]):
                    convergence_type = "quadratic"
                elif avg_rate < 0.3:
                    convergence_type = "superlinear"
                elif avg_rate < 0.9:
                    convergence_type = "linear"
                else:
                    convergence_type = "sublinear"
            else:
                if avg_rate < 0.3:
                    convergence_type = "superlinear"
                elif avg_rate < 0.9:
                    convergence_type = "linear"
                else:
                    convergence_type = "sublinear"
        else:
            if avg_rate < 0.3:
                convergence_type = "superlinear"
            elif avg_rate < 0.9:
                convergence_type = "linear"
            else:
                convergence_type = "sublinear"

    # Generate diagnosis and recommendation
    diagnosis_map = {
        "quadratic": "Quadratic convergence indicates optimal Newton behavior.",
        "superlinear": "Superlinear convergence; quasi-Newton methods working well.",
        "linear": "Linear convergence; rate is acceptable but could be improved.",
        "sublinear": "Sublinear convergence; solver is making slow progress.",
        "unknown": "Could not determine convergence type.",
    }

    action_map = {
        "quadratic": "Continue with current solver; convergence is optimal.",
        "superlinear": "Current setup is effective; monitor for stagnation.",
        "linear": "Consider stronger preconditioner or switch to Newton if Jacobian available.",
        "sublinear": "Switch to Newton method, improve globalization, or check problem formulation.",
        "unknown": "Gather more iterations for analysis.",
    }

    diagnosis = diagnosis_map.get(convergence_type, "Unknown convergence behavior.")
    recommended_action = action_map.get(convergence_type, "Review solver configuration.")

    if converged:
        recommended_action = "Solver converged successfully."

    return {
        "converged": converged,
        "iterations": n,
        "final_residual": final_residual,
        "convergence_type": convergence_type,
        "estimated_rate": estimated_rate,
        "diagnosis": diagnosis,
        "recommended_action": recommended_action,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze convergence from residual history.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--residuals",
        type=str,
        required=True,
        help="Comma-separated residual values",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="Convergence tolerance",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        residuals = [float(x.strip()) for x in args.residuals.split(",")]
    except ValueError:
        print("Error: residuals must be comma-separated numbers", file=sys.stderr)
        sys.exit(2)

    try:
        result = analyze_convergence(
            residuals=residuals,
            tolerance=args.tolerance,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload: Dict[str, Any] = {
        "inputs": {
            "residuals": residuals,
            "tolerance": args.tolerance,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Convergence analysis")
    print(f"  converged: {result['converged']}")
    print(f"  iterations: {result['iterations']}")
    print(f"  final_residual: {result['final_residual']:.2e}")
    print(f"  convergence_type: {result['convergence_type']}")
    if result["estimated_rate"] is not None:
        print(f"  estimated_rate: {result['estimated_rate']:.4f}")
    print(f"  diagnosis: {result['diagnosis']}")
    print(f"  recommended_action: {result['recommended_action']}")


if __name__ == "__main__":
    main()
