#!/usr/bin/env python3
"""Recommend line search vs trust region globalization strategy."""
import argparse
import json
import sys
from typing import Any, Dict, List


def advise_globalization(
    problem_type: str,
    jacobian_quality: str,
    previous_failures: int,
    oscillating_residual: bool,
    step_rejection_rate: float,
) -> Dict[str, Any]:
    """Recommend globalization strategy for nonlinear solvers.

    Args:
        problem_type: Type of problem (root-finding, optimization, least-squares)
        jacobian_quality: Quality of Jacobian (good, ill-conditioned, near-singular)
        previous_failures: Number of previous solver failures
        oscillating_residual: Whether residual history shows oscillation
        step_rejection_rate: Fraction of rejected steps (0.0 to 1.0)

    Returns:
        Dictionary with globalization strategy recommendations
    """
    valid_problem_types = {"root-finding", "optimization", "least-squares"}
    if problem_type not in valid_problem_types:
        raise ValueError(f"problem_type must be one of {valid_problem_types}")

    valid_jacobian_quality = {"good", "ill-conditioned", "near-singular"}
    if jacobian_quality not in valid_jacobian_quality:
        raise ValueError(f"jacobian_quality must be one of {valid_jacobian_quality}")

    if previous_failures < 0:
        raise ValueError("previous_failures must be non-negative")

    if not 0.0 <= step_rejection_rate <= 1.0:
        raise ValueError("step_rejection_rate must be between 0.0 and 1.0")

    notes: List[str] = []

    # Decision logic for strategy selection
    use_trust_region = False

    # Trust region preferred for ill-conditioned or near-singular Jacobians
    if jacobian_quality in {"ill-conditioned", "near-singular"}:
        use_trust_region = True
        notes.append("Trust region provides better stability for poor Jacobian quality.")

    # Trust region preferred for high failure rate
    if previous_failures >= 2:
        use_trust_region = True
        notes.append("Multiple failures suggest trust region for more robust steps.")

    # Trust region preferred for oscillating residuals
    if oscillating_residual:
        use_trust_region = True
        notes.append("Oscillating residuals indicate step size issues; trust region helps.")

    # Trust region preferred for high step rejection
    if step_rejection_rate > 0.3:
        use_trust_region = True
        notes.append("High step rejection rate favors trust region approach.")

    # Problem-type specific adjustments
    if problem_type == "least-squares":
        use_trust_region = True
        notes.append("Trust region is standard for least-squares (Levenberg-Marquardt).")

    if use_trust_region:
        strategy = "trust-region"

        # Select trust region type
        if jacobian_quality == "near-singular":
            trust_region_type = "Levenberg-Marquardt"
            notes.append("LM regularization handles near-singular Jacobian.")
        elif problem_type == "optimization":
            trust_region_type = "Steihaug-CG"
            notes.append("Steihaug-CG efficient for large-scale optimization.")
        else:
            trust_region_type = "dogleg"
            notes.append("Dogleg combines Cauchy and Newton steps effectively.")

        # Initial damping parameter
        if jacobian_quality == "near-singular":
            initial_damping = 1.0
        elif jacobian_quality == "ill-conditioned":
            initial_damping = 0.1
        else:
            initial_damping = 0.01

        parameters = {
            "initial_radius": 1.0,
            "max_radius": 100.0,
            "eta1": 0.25,
            "eta2": 0.75,
            "gamma1": 0.25,
            "gamma2": 2.0,
        }

        return {
            "strategy": strategy,
            "line_search_type": None,
            "trust_region_type": trust_region_type,
            "initial_damping": initial_damping,
            "parameters": parameters,
            "notes": notes,
        }

    else:
        strategy = "line-search"

        # Select line search type
        if problem_type == "optimization":
            line_search_type = "Wolfe"
            notes.append("Wolfe conditions ensure sufficient decrease and curvature.")
        elif previous_failures > 0:
            line_search_type = "backtracking"
            notes.append("Backtracking is simple and robust after failures.")
        else:
            line_search_type = "Armijo"
            notes.append("Armijo sufficient decrease condition for root-finding.")

        # Parameters for line search
        if jacobian_quality == "ill-conditioned":
            initial_damping = 0.5
            notes.append("Starting with reduced step for ill-conditioned problem.")
        else:
            initial_damping = 1.0

        parameters = {
            "c1": 1e-4,  # Armijo/Wolfe sufficient decrease
            "c2": 0.9,  # Wolfe curvature condition
            "alpha_init": 1.0,
            "rho": 0.5,  # Backtracking factor
            "max_backtracks": 20,
        }

        return {
            "strategy": strategy,
            "line_search_type": line_search_type,
            "trust_region_type": None,
            "initial_damping": initial_damping,
            "parameters": parameters,
            "notes": notes,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend globalization strategy for nonlinear solvers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problem-type",
        type=str,
        required=True,
        choices=["root-finding", "optimization", "least-squares"],
        help="Type of problem",
    )
    parser.add_argument(
        "--jacobian-quality",
        type=str,
        default="good",
        choices=["good", "ill-conditioned", "near-singular"],
        help="Quality of Jacobian",
    )
    parser.add_argument(
        "--previous-failures",
        type=int,
        default=0,
        help="Number of previous solver failures",
    )
    parser.add_argument(
        "--oscillating",
        action="store_true",
        help="Residual history shows oscillation",
    )
    parser.add_argument(
        "--step-rejection-rate",
        type=float,
        default=0.0,
        help="Fraction of rejected steps (0.0 to 1.0)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        result = advise_globalization(
            problem_type=args.problem_type,
            jacobian_quality=args.jacobian_quality,
            previous_failures=args.previous_failures,
            oscillating_residual=args.oscillating,
            step_rejection_rate=args.step_rejection_rate,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload: Dict[str, Any] = {
        "inputs": {
            "problem_type": args.problem_type,
            "jacobian_quality": args.jacobian_quality,
            "previous_failures": args.previous_failures,
            "oscillating_residual": args.oscillating,
            "step_rejection_rate": args.step_rejection_rate,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Globalization strategy")
    print(f"  strategy: {result['strategy']}")
    if result["line_search_type"]:
        print(f"  line_search_type: {result['line_search_type']}")
    if result["trust_region_type"]:
        print(f"  trust_region_type: {result['trust_region_type']}")
    print(f"  initial_damping: {result['initial_damping']}")
    for note in result["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
