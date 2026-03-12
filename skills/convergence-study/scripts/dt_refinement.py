#!/usr/bin/env python3
"""Temporal convergence study via dt-refinement analysis."""
import argparse
import json
import math
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze temporal convergence by computing observed order from timestep refinement data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps", type=str, required=True,
        help="Comma-separated timestep sizes (e.g. 0.04,0.02,0.01)",
    )
    parser.add_argument(
        "--values", type=str, required=True,
        help="Comma-separated solution values at each timestep level",
    )
    parser.add_argument(
        "--expected-order", type=float, default=None,
        help="Expected convergence order for assessment",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def compute_dt_refinement(timesteps, values, expected_order=None):
    """Compute temporal convergence order from timestep refinement data.

    Parameters
    ----------
    timesteps : list of float
        Timestep sizes (must be positive).
    values : list of float
        Solution values corresponding to each timestep.
    expected_order : float or None
        Expected convergence order for assessment.

    Returns
    -------
    dict
        Results with observed_orders, mean_order, richardson_extrapolated_value,
        in_asymptotic_range, convergence_assessment, and notes.
    """
    if len(timesteps) != len(values):
        raise ValueError("timesteps and values must have the same length")
    if len(timesteps) < 2:
        raise ValueError("At least 2 refinement levels required")
    for t in timesteps:
        if not math.isfinite(t) or t <= 0:
            raise ValueError("All timesteps must be positive finite numbers")
    for v in values:
        if not math.isfinite(v):
            raise ValueError("All values must be finite numbers")

    # Sort by timestep descending (coarsest first)
    paired = sorted(zip(timesteps, values), key=lambda x: -x[0])
    timesteps_sorted = [p[0] for p in paired]
    values_sorted = [p[1] for p in paired]

    notes = []
    observed_orders = []

    # Compute observed orders from consecutive triplets using log-ratio
    if len(timesteps_sorted) >= 3:
        for i in range(len(timesteps_sorted) - 2):
            dt_coarse = timesteps_sorted[i]
            dt_mid = timesteps_sorted[i + 1]
            dt_fine = timesteps_sorted[i + 2]
            f_coarse = values_sorted[i]
            f_mid = values_sorted[i + 1]
            f_fine = values_sorted[i + 2]

            e_coarse = abs(f_coarse - f_mid)
            e_fine = abs(f_mid - f_fine)

            if e_fine == 0 or e_coarse == 0:
                notes.append(
                    "Zero error difference at levels %d-%d; "
                    "cannot compute order for this pair" % (i, i + 2)
                )
                continue

            r_coarse = dt_coarse / dt_mid

            if r_coarse <= 0:
                continue

            p = math.log(e_coarse / e_fine) / math.log(r_coarse)
            observed_orders.append(p)

            if p < 0:
                notes.append(
                    "Negative observed order (%.2f) at levels %d-%d: "
                    "possible divergence" % (p, i, i + 2)
                )

    # Mean order
    mean_order = None
    if observed_orders:
        mean_order = sum(observed_orders) / len(observed_orders)

    # Check for pre-asymptotic behavior
    in_asymptotic_range = True
    if len(observed_orders) >= 2:
        for i in range(len(observed_orders) - 1):
            p1 = observed_orders[i]
            p2 = observed_orders[i + 1]
            avg = (abs(p1) + abs(p2)) / 2.0
            if avg > 0 and abs(p1 - p2) / avg > 0.5:
                in_asymptotic_range = False
                notes.append("Pre-asymptotic behavior detected: observed order varies >50%% between consecutive pairs")
                break
    elif len(observed_orders) == 1:
        in_asymptotic_range = True
    else:
        in_asymptotic_range = None

    # Richardson extrapolation from finest two levels
    richardson_extrapolated_value = None
    dt_fine = timesteps_sorted[-1]
    dt_next = timesteps_sorted[-2]
    f_fine = values_sorted[-1]
    f_next = values_sorted[-2]
    r = dt_next / dt_fine

    if mean_order is not None and mean_order > 0:
        richardson_extrapolated_value = f_fine + (f_fine - f_next) / (r ** mean_order - 1)
    elif expected_order is not None and expected_order > 0:
        richardson_extrapolated_value = f_fine + (f_fine - f_next) / (r ** expected_order - 1)
        notes.append("Richardson extrapolation used expected order (no observed order available)")

    # Convergence assessment
    convergence_assessment = "unknown"
    if expected_order is not None and mean_order is not None:
        if abs(mean_order - expected_order) / expected_order <= 0.1:
            convergence_assessment = "PASS: observed order (%.2f) within 10%% of expected (%.2f)" % (
                mean_order, expected_order,
            )
        else:
            convergence_assessment = "FAIL: observed order (%.2f) differs from expected (%.2f) by >10%%" % (
                mean_order, expected_order,
            )
    elif mean_order is not None:
        convergence_assessment = "Observed order: %.2f (no expected order given for comparison)" % mean_order
    elif len(timesteps_sorted) < 3:
        convergence_assessment = "Insufficient levels to compute observed order (need >= 3)"

    return {
        "inputs": {
            "timesteps": timesteps_sorted,
            "values": values_sorted,
            "expected_order": expected_order,
        },
        "results": {
            "observed_orders": observed_orders,
            "mean_order": mean_order,
            "richardson_extrapolated_value": richardson_extrapolated_value,
            "in_asymptotic_range": in_asymptotic_range,
            "convergence_assessment": convergence_assessment,
            "notes": notes,
        },
    }


def main():
    args = parse_args()
    try:
        timesteps = [float(x) for x in args.timesteps.split(",")]
        values = [float(x) for x in args.values.split(",")]
    except ValueError:
        print("Error: timesteps and values must be comma-separated numbers", file=sys.stderr)
        sys.exit(2)

    try:
        result = compute_dt_refinement(timesteps, values, args.expected_order)
    except ValueError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        r = result["results"]
        print("Temporal Convergence Study (dt-refinement)")
        print("  levels: %d" % len(timesteps))
        if r["observed_orders"]:
            print("  observed orders: %s" % ", ".join("%.4f" % o for o in r["observed_orders"]))
        if r["mean_order"] is not None:
            print("  mean order: %.4f" % r["mean_order"])
        if r["richardson_extrapolated_value"] is not None:
            print("  Richardson extrapolated value: %.6g" % r["richardson_extrapolated_value"])
        if r["in_asymptotic_range"] is not None:
            print("  in asymptotic range: %s" % r["in_asymptotic_range"])
        print("  assessment: %s" % r["convergence_assessment"])
        for note in r["notes"]:
            print("  note: %s" % note)


if __name__ == "__main__":
    main()
