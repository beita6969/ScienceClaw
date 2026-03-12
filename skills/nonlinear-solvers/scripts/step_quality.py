#!/usr/bin/env python3
"""Evaluate Newton/quasi-Newton step quality for trust region decisions."""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def evaluate_step(
    predicted_reduction: float,
    actual_reduction: float,
    step_norm: float,
    gradient_norm: float,
    trust_radius: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate step quality for trust region management.

    Args:
        predicted_reduction: Model-predicted decrease in objective
        actual_reduction: Actual decrease in objective
        step_norm: Norm of the step taken
        gradient_norm: Norm of gradient at current point
        trust_radius: Current trust radius (if using trust region)

    Returns:
        Dictionary with step quality assessment and recommendations
    """
    if predicted_reduction < 0:
        raise ValueError("predicted_reduction must be non-negative (model should predict decrease)")

    if step_norm < 0:
        raise ValueError("step_norm must be non-negative")

    if gradient_norm < 0:
        raise ValueError("gradient_norm must be non-negative")

    if trust_radius is not None and trust_radius <= 0:
        raise ValueError("trust_radius must be positive")

    notes: List[str] = []

    # Compute the ratio (actual/predicted reduction)
    if predicted_reduction > 1e-30:
        ratio = actual_reduction / predicted_reduction
    else:
        # Near zero predicted reduction - step is near stationary point
        if abs(actual_reduction) < 1e-30:
            ratio = 1.0
            notes.append("Near stationary point; both reductions negligible.")
        else:
            ratio = float("inf") if actual_reduction > 0 else float("-inf")
            notes.append("Predicted reduction near zero but actual reduction nonzero.")

    # Classify step quality based on ratio
    # Standard thresholds: eta1 = 0.25, eta2 = 0.75
    if ratio < 0:
        step_quality = "very_poor"
        accept_step = False
        notes.append("Negative ratio: objective increased instead of decreased.")
    elif ratio < 0.1:
        step_quality = "poor"
        accept_step = False
        notes.append("Very poor agreement between model and actual behavior.")
    elif ratio < 0.25:
        step_quality = "marginal"
        accept_step = True
        notes.append("Marginal step quality; accepting but reducing trust radius.")
    elif ratio < 0.75:
        step_quality = "good"
        accept_step = True
        notes.append("Good agreement between model and actual behavior.")
    else:
        step_quality = "excellent"
        accept_step = True
        notes.append("Excellent step quality; model is accurate.")

    # Trust radius adjustment recommendation
    trust_radius_action = None
    suggested_trust_radius = None

    if trust_radius is not None:
        if ratio < 0.1:
            trust_radius_action = "shrink_aggressive"
            suggested_trust_radius = trust_radius * 0.25
            notes.append("Aggressively shrinking trust radius due to poor step.")
        elif ratio < 0.25:
            trust_radius_action = "shrink"
            suggested_trust_radius = trust_radius * 0.5
            notes.append("Shrinking trust radius.")
        elif ratio > 0.75 and step_norm >= 0.9 * trust_radius:
            trust_radius_action = "expand"
            suggested_trust_radius = min(trust_radius * 2.0, 100.0)
            notes.append("Expanding trust radius as step hit boundary.")
        else:
            trust_radius_action = "maintain"
            suggested_trust_radius = trust_radius
            notes.append("Maintaining current trust radius.")
    else:
        # Line search mode
        if ratio < 0.25:
            notes.append("Consider reducing initial step size for line search.")
        elif ratio > 0.9:
            notes.append("Full Newton step successful; line search may be unnecessary.")

    # Check for potential issues
    if step_norm < 1e-14:
        notes.append("Warning: step norm is nearly zero; may be at local minimum or saddle.")

    if gradient_norm < 1e-10 and not accept_step:
        notes.append("Near-zero gradient with rejected step; check for saddle point.")

    # Cauchy decrease check: should get at least some fraction of Cauchy decrease
    cauchy_decrease_expected = gradient_norm * min(step_norm, gradient_norm)
    if accept_step and actual_reduction < 0.1 * cauchy_decrease_expected and gradient_norm > 1e-10:
        notes.append("Step achieved less than expected Cauchy decrease.")

    return {
        "ratio": ratio if not isinstance(ratio, float) or ratio != float("inf") else None,
        "step_quality": step_quality,
        "accept_step": accept_step,
        "trust_radius_action": trust_radius_action,
        "suggested_trust_radius": suggested_trust_radius,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate step quality for trust region decisions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predicted-reduction",
        type=float,
        required=True,
        help="Model-predicted decrease in objective",
    )
    parser.add_argument(
        "--actual-reduction",
        type=float,
        required=True,
        help="Actual decrease in objective",
    )
    parser.add_argument(
        "--step-norm",
        type=float,
        required=True,
        help="Norm of the step taken",
    )
    parser.add_argument(
        "--gradient-norm",
        type=float,
        required=True,
        help="Norm of gradient at current point",
    )
    parser.add_argument(
        "--trust-radius",
        type=float,
        default=None,
        help="Current trust radius (optional)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        result = evaluate_step(
            predicted_reduction=args.predicted_reduction,
            actual_reduction=args.actual_reduction,
            step_norm=args.step_norm,
            gradient_norm=args.gradient_norm,
            trust_radius=args.trust_radius,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload: Dict[str, Any] = {
        "inputs": {
            "predicted_reduction": args.predicted_reduction,
            "actual_reduction": args.actual_reduction,
            "step_norm": args.step_norm,
            "gradient_norm": args.gradient_norm,
            "trust_radius": args.trust_radius,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Step quality evaluation")
    if result["ratio"] is not None:
        print(f"  ratio: {result['ratio']:.4f}")
    print(f"  step_quality: {result['step_quality']}")
    print(f"  accept_step: {result['accept_step']}")
    if result["trust_radius_action"]:
        print(f"  trust_radius_action: {result['trust_radius_action']}")
    if result["suggested_trust_radius"] is not None:
        print(f"  suggested_trust_radius: {result['suggested_trust_radius']:.4f}")
    for note in result["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
