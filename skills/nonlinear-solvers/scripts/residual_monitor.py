#!/usr/bin/env python3
"""Monitor residual patterns and detect failure modes."""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def monitor_residuals(
    residuals: List[float],
    function_evals: Optional[List[int]] = None,
    step_sizes: Optional[List[float]] = None,
    target_tolerance: float = 1e-10,
) -> Dict[str, Any]:
    """Monitor residual patterns and detect failure modes.

    Args:
        residuals: List of residual norms
        function_evals: Optional list of cumulative function evaluations
        step_sizes: Optional list of step sizes taken
        target_tolerance: Target convergence tolerance

    Returns:
        Dictionary with monitoring results and recommendations
    """
    if not residuals:
        raise ValueError("residuals must not be empty")

    if any(r < 0 for r in residuals):
        raise ValueError("residuals must be non-negative")

    if target_tolerance <= 0:
        raise ValueError("target_tolerance must be positive")

    n = len(residuals)
    patterns_detected: List[str] = []
    alerts: List[str] = []
    recommendations: List[str] = []

    # Basic stats
    initial_residual = residuals[0]
    final_residual = residuals[-1]

    if initial_residual > 1e-30:
        residual_reduction = final_residual / initial_residual
    else:
        residual_reduction = 1.0

    # Check for convergence
    converged = final_residual <= target_tolerance

    # Function evaluation efficiency
    total_function_evals = None
    efficiency = None
    if function_evals and len(function_evals) == n:
        total_function_evals = function_evals[-1]
        if total_function_evals > 0:
            # Orders of magnitude reduced per function eval
            if residual_reduction > 0 and residual_reduction < 1:
                import math
                log_reduction = -math.log10(residual_reduction)
                efficiency = log_reduction / total_function_evals
            else:
                efficiency = 0.0

    # Step size analysis
    step_size_trend = None
    if step_sizes and len(step_sizes) >= 2:
        avg_early = sum(step_sizes[:len(step_sizes)//2 + 1]) / (len(step_sizes)//2 + 1)
        avg_late = sum(step_sizes[len(step_sizes)//2:]) / (len(step_sizes) - len(step_sizes)//2)

        if avg_early > 1e-30:
            ratio = avg_late / avg_early
            if ratio < 0.5:
                step_size_trend = "decreasing"
                patterns_detected.append("step_size_decreasing")
            elif ratio > 2.0:
                step_size_trend = "increasing"
                patterns_detected.append("step_size_increasing")
            else:
                step_size_trend = "stable"

    # Pattern detection
    if n >= 3:
        # Check for oscillation
        oscillations = 0
        for i in range(1, n - 1):
            if (residuals[i] > residuals[i-1] and residuals[i] > residuals[i+1]) or \
               (residuals[i] < residuals[i-1] and residuals[i] < residuals[i+1]):
                oscillations += 1

        if oscillations >= n // 3:
            patterns_detected.append("oscillating")
            alerts.append("Residual oscillation detected; may indicate step size issues.")
            recommendations.append("Consider trust region or reduced initial step size.")

        # Check for plateau
        recent = residuals[-min(5, n):]
        if len(recent) >= 3:
            rel_change = max(recent) / (min(recent) + 1e-30)
            if rel_change < 1.1 and not converged:
                patterns_detected.append("plateau")
                alerts.append("Residual plateau detected; solver may be stagnating.")
                recommendations.append("Try stronger preconditioner or different solver.")

        # Check for divergence
        if n >= 2 and residuals[-1] > residuals[0] * 1.5:
            patterns_detected.append("diverging")
            alerts.append("Residuals are increasing; solver is diverging!")
            recommendations.append("Reduce step size, check Jacobian, or add regularization.")

        # Check for very slow progress
        if n >= 10:
            halfway = n // 2
            early_reduction = residuals[halfway] / (residuals[0] + 1e-30)
            late_reduction = residuals[-1] / (residuals[halfway] + 1e-30)

            if early_reduction > 0.9 and late_reduction > 0.9:
                patterns_detected.append("slow_convergence")
                alerts.append("Very slow convergence detected.")
                recommendations.append("Consider Newton method with good Jacobian.")

    # Check for initial spike (common with bad initial guess)
    if n >= 2 and residuals[1] > residuals[0] * 2:
        patterns_detected.append("initial_spike")
        recommendations.append("Consider better initial guess or damped first step.")

    # Final recommendations
    if not patterns_detected:
        patterns_detected.append("normal")

    if converged:
        recommendations.insert(0, "Solver converged successfully.")
    elif not recommendations:
        recommendations.append("Continue monitoring; no immediate issues detected.")

    result: Dict[str, Any] = {
        "residual_reduction": residual_reduction,
        "iterations": n,
        "converged": converged,
        "patterns_detected": patterns_detected,
        "alerts": alerts,
        "recommendations": recommendations,
    }

    if total_function_evals is not None:
        result["function_evals"] = total_function_evals
    if efficiency is not None:
        result["efficiency"] = efficiency
    if step_size_trend is not None:
        result["step_size_trend"] = step_size_trend

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor residual patterns and detect failure modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--residuals",
        type=str,
        required=True,
        help="Comma-separated residual values",
    )
    parser.add_argument(
        "--function-evals",
        type=str,
        default=None,
        help="Comma-separated cumulative function evaluation counts",
    )
    parser.add_argument(
        "--step-sizes",
        type=str,
        default=None,
        help="Comma-separated step sizes",
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=1e-10,
        help="Target convergence tolerance",
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

    function_evals = None
    if args.function_evals:
        try:
            function_evals = [int(x.strip()) for x in args.function_evals.split(",")]
        except ValueError:
            print("Error: function-evals must be comma-separated integers", file=sys.stderr)
            sys.exit(2)

    step_sizes = None
    if args.step_sizes:
        try:
            step_sizes = [float(x.strip()) for x in args.step_sizes.split(",")]
        except ValueError:
            print("Error: step-sizes must be comma-separated numbers", file=sys.stderr)
            sys.exit(2)

    try:
        result = monitor_residuals(
            residuals=residuals,
            function_evals=function_evals,
            step_sizes=step_sizes,
            target_tolerance=args.target_tolerance,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload: Dict[str, Any] = {
        "inputs": {
            "residuals": residuals,
            "function_evals": function_evals,
            "step_sizes": step_sizes,
            "target_tolerance": args.target_tolerance,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Residual monitoring")
    print(f"  residual_reduction: {result['residual_reduction']:.2e}")
    print(f"  iterations: {result['iterations']}")
    print(f"  converged: {result['converged']}")
    print(f"  patterns: {', '.join(result['patterns_detected'])}")
    for alert in result["alerts"]:
        print(f"  ALERT: {alert}")
    for rec in result["recommendations"]:
        print(f"  recommendation: {rec}")


if __name__ == "__main__":
    main()
