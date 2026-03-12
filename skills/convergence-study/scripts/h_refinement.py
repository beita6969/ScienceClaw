#!/usr/bin/env python3
"""Spatial convergence study via h-refinement analysis."""
import argparse
import json
import math
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze spatial convergence by computing observed order from grid refinement data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spacings", type=str, required=True,
        help="Comma-separated grid spacings (e.g. 0.4,0.2,0.1)",
    )
    parser.add_argument(
        "--values", type=str, required=True,
        help="Comma-separated solution values at each spacing level",
    )
    parser.add_argument(
        "--expected-order", type=float, default=None,
        help="Expected convergence order for assessment",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def compute_h_refinement(spacings, values, expected_order=None):
    """Compute spatial convergence order from grid refinement data.

    Parameters
    ----------
    spacings : list of float
        Grid spacings (must be positive).
    values : list of float
        Solution values corresponding to each spacing.
    expected_order : float or None
        Expected convergence order for assessment.

    Returns
    -------
    dict
        Results with observed_orders, mean_order, richardson_extrapolated_value,
        in_asymptotic_range, convergence_assessment, and notes.
    """
    if len(spacings) != len(values):
        raise ValueError("spacings and values must have the same length")
    if len(spacings) < 2:
        raise ValueError("At least 2 refinement levels required")
    for s in spacings:
        if not math.isfinite(s) or s <= 0:
            raise ValueError("All spacings must be positive finite numbers")
    for v in values:
        if not math.isfinite(v):
            raise ValueError("All values must be finite numbers")

    # Sort by spacing descending (coarsest first)
    paired = sorted(zip(spacings, values), key=lambda x: -x[0])
    spacings_sorted = [p[0] for p in paired]
    values_sorted = [p[1] for p in paired]

    notes = []
    observed_orders = []

    # Compute observed orders from consecutive triplets using log-ratio
    if len(spacings_sorted) >= 3:
        for i in range(len(spacings_sorted) - 2):
            h_coarse = spacings_sorted[i]
            h_mid = spacings_sorted[i + 1]
            h_fine = spacings_sorted[i + 2]
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

            r_coarse = h_coarse / h_mid
            r_fine = h_mid / h_fine

            if r_coarse <= 0 or r_fine <= 0:
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
    h_fine = spacings_sorted[-1]
    h_next = spacings_sorted[-2]
    f_fine = values_sorted[-1]
    f_next = values_sorted[-2]
    r = h_next / h_fine

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
    elif len(spacings_sorted) < 3:
        convergence_assessment = "Insufficient levels to compute observed order (need >= 3)"

    return {
        "inputs": {
            "spacings": spacings_sorted,
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
        spacings = [float(x) for x in args.spacings.split(",")]
        values = [float(x) for x in args.values.split(",")]
    except ValueError:
        print("Error: spacings and values must be comma-separated numbers", file=sys.stderr)
        sys.exit(2)

    try:
        result = compute_h_refinement(spacings, values, args.expected_order)
    except ValueError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        r = result["results"]
        print("Spatial Convergence Study (h-refinement)")
        print("  levels: %d" % len(spacings))
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
