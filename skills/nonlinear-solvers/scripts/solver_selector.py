#!/usr/bin/env python3
"""Select nonlinear solver based on problem characteristics."""
import argparse
import json
import sys
from typing import Any, Dict, List


def select_solver(
    jacobian_available: bool,
    jacobian_expensive: bool,
    problem_size: int,
    spd_hessian: bool,
    smooth_objective: bool,
    constraint_type: str,
    memory_limited: bool,
    high_accuracy: bool,
) -> Dict[str, List[str]]:
    """Select nonlinear solver based on problem characteristics.

    Args:
        jacobian_available: Whether analytic Jacobian is available
        jacobian_expensive: Whether Jacobian computation is expensive
        problem_size: Number of unknowns
        spd_hessian: Whether Hessian is symmetric positive definite
        smooth_objective: Whether objective/residual is smooth
        constraint_type: Type of constraints (none, bound, equality, inequality)
        memory_limited: Whether memory is constrained
        high_accuracy: Whether high accuracy is required

    Returns:
        Dictionary with recommended solvers, alternatives, and notes
    """
    if problem_size <= 0:
        raise ValueError("problem_size must be positive")

    valid_constraints = {"none", "bound", "equality", "inequality"}
    if constraint_type not in valid_constraints:
        raise ValueError(f"constraint_type must be one of {valid_constraints}")

    recommended: List[str] = []
    alternatives: List[str] = []
    notes: List[str] = []

    large_problem = problem_size >= 10_000

    # Handle constrained optimization
    if constraint_type == "inequality":
        recommended.append("SQP (Sequential Quadratic Programming)")
        alternatives.append("Interior Point")
        notes.append("Inequality constraints require specialized solvers.")
        if not jacobian_available:
            notes.append("Consider finite-difference Jacobian or quasi-Newton Hessian.")
        return {"recommended": recommended, "alternatives": alternatives, "notes": notes}

    if constraint_type == "equality":
        recommended.append("SQP")
        alternatives.append("Augmented Lagrangian")
        notes.append("Equality constraints: use Lagrange multipliers or penalty methods.")
        return {"recommended": recommended, "alternatives": alternatives, "notes": notes}

    if constraint_type == "bound":
        if spd_hessian:
            recommended.append("L-BFGS-B")
            alternatives.append("Trust-Region Reflective")
        else:
            recommended.append("Trust-Region Reflective")
            alternatives.append("L-BFGS-B")
        notes.append("Bound constraints handled via projected methods.")
        return {"recommended": recommended, "alternatives": alternatives, "notes": notes}

    # Unconstrained case
    if jacobian_available and not jacobian_expensive:
        if high_accuracy:
            recommended.append("Newton (full)")
            alternatives.append("Modified Newton")
            notes.append("Full Newton provides quadratic convergence near solution.")
        elif large_problem:
            recommended.append("Newton-Krylov (GMRES)")
            alternatives.append("Newton-Krylov (BiCGSTAB)")
            notes.append("Newton-Krylov avoids forming full Jacobian for large problems.")
        else:
            recommended.append("Newton (full)")
            alternatives.append("Modified Newton")
    elif jacobian_available and jacobian_expensive:
        if memory_limited:
            recommended.append("L-BFGS")
            alternatives.append("Broyden")
            notes.append("L-BFGS uses limited memory quasi-Newton updates.")
        else:
            recommended.append("Modified Newton")
            alternatives.append("Broyden")
            notes.append("Modified Newton reuses Jacobian for multiple iterations.")
    else:
        # No Jacobian available
        if smooth_objective:
            if memory_limited or large_problem:
                recommended.append("L-BFGS")
                alternatives.append("Broyden (good)")
                notes.append("Quasi-Newton methods build approximate Jacobian.")
            else:
                recommended.append("BFGS")
                alternatives.append("SR1")
                notes.append("BFGS provides superlinear convergence for smooth problems.")
        else:
            recommended.append("Anderson Acceleration")
            alternatives.append("Picard (fixed-point)")
            notes.append("Non-smooth problems may benefit from fixed-point methods.")

    # Additional recommendations based on problem characteristics
    if spd_hessian and not jacobian_available:
        notes.append("SPD Hessian: BFGS maintains positive definiteness.")

    if large_problem and "Newton (full)" in recommended:
        notes.append("For very large problems, consider Newton-Krylov instead.")

    if not smooth_objective:
        notes.append("Non-smooth: consider subgradient or bundle methods.")

    return {"recommended": recommended, "alternatives": alternatives, "notes": notes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a nonlinear solver based on problem characteristics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--jacobian-available",
        action="store_true",
        help="Analytic Jacobian is available",
    )
    parser.add_argument(
        "--jacobian-expensive",
        action="store_true",
        help="Jacobian computation is expensive",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Problem size (number of unknowns)",
    )
    parser.add_argument(
        "--spd-hessian",
        action="store_true",
        help="Hessian is symmetric positive definite",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Objective/residual is smooth",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default="none",
        choices=["none", "bound", "equality", "inequality"],
        help="Type of constraints",
    )
    parser.add_argument(
        "--memory-limited",
        action="store_true",
        help="Memory is constrained",
    )
    parser.add_argument(
        "--high-accuracy",
        action="store_true",
        help="High accuracy is required",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = select_solver(
            jacobian_available=args.jacobian_available,
            jacobian_expensive=args.jacobian_expensive,
            problem_size=args.size,
            spd_hessian=args.spd_hessian,
            smooth_objective=args.smooth,
            constraint_type=args.constraints,
            memory_limited=args.memory_limited,
            high_accuracy=args.high_accuracy,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload: Dict[str, Any] = {
        "inputs": {
            "jacobian_available": args.jacobian_available,
            "jacobian_expensive": args.jacobian_expensive,
            "problem_size": args.size,
            "spd_hessian": args.spd_hessian,
            "smooth_objective": args.smooth,
            "constraint_type": args.constraints,
            "memory_limited": args.memory_limited,
            "high_accuracy": args.high_accuracy,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Solver selection")
    print(f"  recommended: {', '.join(result['recommended'])}")
    if result["alternatives"]:
        print(f"  alternatives: {', '.join(result['alternatives'])}")
    for note in result["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
