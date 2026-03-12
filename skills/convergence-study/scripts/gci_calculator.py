#!/usr/bin/env python3
"""Grid Convergence Index (GCI) calculator following Roache's method."""
import argparse
import json
import math
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Grid Convergence Index (GCI) for solution verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spacings", type=str, required=True,
        help="Comma-separated grid spacings for exactly 3 levels (e.g. 0.04,0.02,0.01)",
    )
    parser.add_argument(
        "--values", type=str, required=True,
        help="Comma-separated solution values at each spacing level",
    )
    parser.add_argument(
        "--safety-factor", type=float, default=1.25,
        help="Safety factor for GCI (default 1.25 for 3+ grids)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def compute_gci(spacings, values, safety_factor=1.25):
    """Compute Grid Convergence Index.

    Parameters
    ----------
    spacings : list of float
        Exactly 3 grid spacings (positive).
    values : list of float
        Solution values at each spacing level.
    safety_factor : float
        Safety factor (default 1.25).

    Returns
    -------
    dict
        Results with observed_order, gci_fine, gci_coarse, asymptotic_ratio,
        in_asymptotic_range, refinement_ratio_21, refinement_ratio_32,
        and extrapolated_value.
    """
    if len(spacings) != 3 or len(values) != 3:
        raise ValueError("Exactly 3 refinement levels required for GCI calculation")
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")
    for s in spacings:
        if not math.isfinite(s) or s <= 0:
            raise ValueError("All spacings must be positive finite numbers")
    for v in values:
        if not math.isfinite(v):
            raise ValueError("All values must be finite numbers")

    # Sort by spacing ascending: h1 (fine) < h2 (medium) < h3 (coarse)
    paired = sorted(zip(spacings, values), key=lambda x: x[0])
    h1, f1 = paired[0]  # fine
    h2, f2 = paired[1]  # medium
    h3, f3 = paired[2]  # coarse

    r21 = h2 / h1
    r32 = h3 / h2

    e21 = f2 - f1
    e32 = f3 - f2

    # Check for oscillatory convergence
    if e21 != 0 and e32 != 0 and (e32 * e21) < 0:
        raise ValueError("Oscillatory convergence detected")

    # Check for identical solutions
    if e21 == 0 and e32 == 0:
        return {
            "inputs": {
                "spacings": [h1, h2, h3],
                "values": [f1, f2, f3],
                "safety_factor": safety_factor,
            },
            "results": {
                "observed_order": None,
                "gci_fine": 0.0,
                "gci_coarse": 0.0,
                "asymptotic_ratio": None,
                "in_asymptotic_range": None,
                "refinement_ratio_21": r21,
                "refinement_ratio_32": r32,
                "extrapolated_value": f1,
            },
        }

    # Observed order
    if e21 == 0:
        raise ValueError("Fine and medium solutions are identical but coarse differs; cannot compute order")
    if e32 == 0:
        raise ValueError("Medium and coarse solutions are identical but fine differs; cannot compute order")

    p = abs(math.log(abs(e32 / e21))) / math.log(r21)

    # GCI fine and coarse
    gci_fine = safety_factor * abs(e21 / f1) / (r21 ** p - 1) if f1 != 0 else float("inf")
    gci_coarse = safety_factor * abs(e32 / f2) / (r32 ** p - 1) if f2 != 0 else float("inf")

    # Asymptotic ratio
    if gci_fine != 0 and math.isfinite(gci_fine) and math.isfinite(gci_coarse):
        asymptotic_ratio = gci_coarse / (r21 ** p * gci_fine)
    else:
        asymptotic_ratio = None

    in_asymptotic_range = None
    if asymptotic_ratio is not None:
        in_asymptotic_range = abs(asymptotic_ratio - 1.0) < 0.1

    # Richardson extrapolated value
    extrapolated_value = f1 + (f1 - f2) / (r21 ** p - 1)

    return {
        "inputs": {
            "spacings": [h1, h2, h3],
            "values": [f1, f2, f3],
            "safety_factor": safety_factor,
        },
        "results": {
            "observed_order": p,
            "gci_fine": gci_fine,
            "gci_coarse": gci_coarse,
            "asymptotic_ratio": asymptotic_ratio,
            "in_asymptotic_range": in_asymptotic_range,
            "refinement_ratio_21": r21,
            "refinement_ratio_32": r32,
            "extrapolated_value": extrapolated_value,
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
        result = compute_gci(spacings, values, args.safety_factor)
    except ValueError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        r = result["results"]
        print("Grid Convergence Index (GCI)")
        print("  refinement ratio r21: %.4f" % r["refinement_ratio_21"])
        print("  refinement ratio r32: %.4f" % r["refinement_ratio_32"])
        if r["observed_order"] is not None:
            print("  observed order: %.4f" % r["observed_order"])
        print("  GCI fine: %.6g" % r["gci_fine"])
        print("  GCI coarse: %.6g" % r["gci_coarse"])
        if r["asymptotic_ratio"] is not None:
            print("  asymptotic ratio: %.4f" % r["asymptotic_ratio"])
        if r["in_asymptotic_range"] is not None:
            print("  in asymptotic range: %s" % r["in_asymptotic_range"])
        print("  extrapolated value: %.6g" % r["extrapolated_value"])


if __name__ == "__main__":
    main()
