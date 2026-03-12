#!/usr/bin/env python3
"""Richardson extrapolation for solution verification."""
import argparse
import json
import math
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform Richardson extrapolation to estimate grid/timestep-independent solution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spacings", type=str, required=True,
        help="Comma-separated spacings (e.g. 0.02,0.01)",
    )
    parser.add_argument(
        "--values", type=str, required=True,
        help="Comma-separated solution values at each spacing level",
    )
    parser.add_argument(
        "--order", type=float, required=True,
        help="Assumed convergence order",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def compute_richardson_extrapolation(spacings, values, order):
    """Perform Richardson extrapolation.

    Parameters
    ----------
    spacings : list of float
        Grid spacings or timestep sizes (positive).
    values : list of float
        Solution values at each level.
    order : float
        Assumed convergence order (positive).

    Returns
    -------
    dict
        Results with extrapolated_value, error_estimate, relative_error_estimate,
        and optionally observed_order and order_consistent.
    """
    if len(spacings) != len(values):
        raise ValueError("spacings and values must have the same length")
    if len(spacings) < 2:
        raise ValueError("At least 2 refinement levels required")
    if order <= 0:
        raise ValueError("order must be positive")
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

    # Richardson extrapolation from finest two levels
    h_fine = spacings_sorted[-1]
    h_coarse = spacings_sorted[-2]
    f_fine = values_sorted[-1]
    f_coarse = values_sorted[-2]
    r = h_coarse / h_fine

    extrapolated_value = f_fine + (f_fine - f_coarse) / (r ** order - 1)
    error_estimate = abs(f_fine - f_coarse) / (r ** order - 1)
    relative_error_estimate = error_estimate / abs(extrapolated_value) if extrapolated_value != 0 else float("inf")

    result = {
        "extrapolated_value": extrapolated_value,
        "error_estimate": error_estimate,
        "relative_error_estimate": relative_error_estimate,
    }

    # With 3+ levels, compute observed order and check consistency
    if len(spacings_sorted) >= 3:
        h1 = spacings_sorted[-1]  # finest
        h2 = spacings_sorted[-2]
        h3 = spacings_sorted[-3]  # coarsest of three
        f1 = values_sorted[-1]
        f2 = values_sorted[-2]
        f3 = values_sorted[-3]

        e_fine = abs(f2 - f1)
        e_coarse = abs(f3 - f2)

        if e_fine > 0 and e_coarse > 0:
            r_ratio = h2 / h1
            observed_order = math.log(e_coarse / e_fine) / math.log(r_ratio)
            result["observed_order"] = observed_order
            result["order_consistent"] = abs(observed_order - order) / order <= 0.1
        else:
            result["observed_order"] = None
            result["order_consistent"] = None

    return {
        "inputs": {
            "spacings": spacings_sorted,
            "values": values_sorted,
            "order": order,
        },
        "results": result,
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
        result = compute_richardson_extrapolation(spacings, values, args.order)
    except ValueError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        r = result["results"]
        print("Richardson Extrapolation")
        print("  assumed order: %.2f" % args.order)
        print("  extrapolated value: %.6g" % r["extrapolated_value"])
        print("  error estimate: %.6g" % r["error_estimate"])
        print("  relative error estimate: %.6g" % r["relative_error_estimate"])
        if "observed_order" in r and r["observed_order"] is not None:
            print("  observed order: %.4f" % r["observed_order"])
            print("  order consistent: %s" % r["order_consistent"])


if __name__ == "__main__":
    main()
