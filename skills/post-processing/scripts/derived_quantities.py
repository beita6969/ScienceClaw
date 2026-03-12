#!/usr/bin/env python3
"""
Derived Quantities Calculator - Compute physical quantities from simulation data.

Calculates derived quantities like interface area, volume fractions,
gradients, fluxes, and integrals from raw simulation field data.

Usage:
    python derived_quantities.py --input field.json --quantity volume_fraction --field phi --threshold 0.5 --json
    python derived_quantities.py --input field.json --quantity gradient_magnitude --field phi --json
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file and return contents."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_field_data(data: Dict[str, Any], field_name: str) -> Optional[List]:
    """Extract field data as nested list."""
    if field_name in data:
        return data[field_name]
    if "fields" in data and field_name in data["fields"]:
        field_data = data["fields"][field_name]
        if isinstance(field_data, dict) and "values" in field_data:
            return field_data["values"]
        return field_data
    if "_data" in data and field_name in data["_data"]:
        return data["_data"][field_name]
    return None


def flatten_field(field: Any) -> List[float]:
    """Flatten nested list to 1D array."""
    if not isinstance(field, list):
        if isinstance(field, (int, float)):
            return [float(field)]
        return []

    result = []
    for item in field:
        result.extend(flatten_field(item))
    return result


def get_field_shape(field: List) -> List[int]:
    """Get shape of nested list."""
    shape = []
    current = field
    while isinstance(current, list):
        shape.append(len(current))
        if len(current) > 0:
            current = current[0]
        else:
            break
    return shape


def get_grid_spacing(data: Dict[str, Any], shape: List[int]) -> Dict[str, float]:
    """Extract or compute grid spacing."""
    spacing = {}

    # Try to get from data
    for key in ["dx", "dy", "dz"]:
        if key in data:
            spacing[key] = data[key]

    # Try to compute from domain size and shape
    if "Lx" in data and len(shape) >= 1:
        if shape[-1] > 1:
            spacing["dx"] = data["Lx"] / (shape[-1] - 1)
    if "Ly" in data and len(shape) >= 2:
        if shape[-2] > 1:
            spacing["dy"] = data["Ly"] / (shape[-2] - 1)
    if "Lz" in data and len(shape) >= 3:
        if shape[-3] > 1:
            spacing["dz"] = data["Lz"] / (shape[-3] - 1)

    # Default to 1.0 if not found
    spacing.setdefault("dx", 1.0)
    spacing.setdefault("dy", 1.0)
    spacing.setdefault("dz", 1.0)

    return spacing


def compute_volume_fraction(
    field: List,
    threshold: float = 0.5,
    above: bool = True
) -> Dict[str, Any]:
    """Compute volume fraction above/below threshold."""
    values = flatten_field(field)

    if not values:
        return {"volume_fraction": None, "count_above": 0, "total": 0}

    # Check for NaN/Inf values that would silently corrupt results
    for v in values:
        if not math.isfinite(v):
            raise ValueError(
                f"Field contains non-finite value ({v}); cannot compute volume fraction"
            )

    if above:
        count = sum(1 for v in values if v >= threshold)
    else:
        count = sum(1 for v in values if v < threshold)

    total = len(values)
    fraction = count / total

    return {
        "threshold": threshold,
        "condition": "above" if above else "below",
        "volume_fraction": fraction,
        "count": count,
        "total": total
    }


def compute_interface_area_2d(
    field: List[List[float]],
    threshold: float,
    dx: float,
    dy: float
) -> Dict[str, Any]:
    """Estimate interface area in 2D using marching squares concept."""
    ny = len(field)
    if ny == 0:
        return {"interface_length": 0, "cells_crossed": 0}

    nx = len(field[0])

    interface_segments = 0

    for j in range(ny - 1):
        for i in range(nx - 1):
            # Get values at cell corners
            v00 = field[j][i]
            v10 = field[j][i + 1]
            v01 = field[j + 1][i]
            v11 = field[j + 1][i + 1]

            # Count how many corners are above threshold
            above = sum(1 for v in [v00, v10, v01, v11] if v >= threshold)

            # If some above and some below, interface crosses cell
            if 0 < above < 4:
                interface_segments += 1

    # Approximate interface length
    avg_cell_size = math.sqrt(dx * dy)
    interface_length = interface_segments * avg_cell_size

    return {
        "threshold": threshold,
        "interface_length": interface_length,
        "cells_crossed": interface_segments,
        "method": "marching_squares_approximation"
    }


def compute_interface_area(
    field: List,
    threshold: float,
    spacing: Dict[str, float]
) -> Dict[str, Any]:
    """Compute interface area for 2D/3D fields."""
    shape = get_field_shape(field)

    if len(shape) == 2:
        return compute_interface_area_2d(
            field, threshold, spacing["dx"], spacing["dy"]
        )

    # For 1D, count crossings
    if len(shape) == 1:
        crossings = 0
        for i in range(len(field) - 1):
            if (field[i] - threshold) * (field[i + 1] - threshold) < 0:
                crossings += 1
        return {
            "threshold": threshold,
            "crossings": crossings,
            "method": "1d_crossing_count"
        }

    return {"error": "3D interface area not implemented", "shape": shape}


def compute_gradient_2d(
    field: List[List[float]],
    dx: float,
    dy: float
) -> Tuple[List[List[float]], List[List[float]]]:
    """Compute gradient of 2D field using central differences."""
    ny = len(field)
    if ny == 0:
        return [], []
    nx = len(field[0])
    # Validate uniform row lengths (ragged arrays would crash)
    for j in range(ny):
        if len(field[j]) != nx:
            raise ValueError(
                f"Ragged array: row 0 has {nx} columns but row {j} has {len(field[j])}"
            )

    grad_x = [[0.0] * nx for _ in range(ny)]
    grad_y = [[0.0] * nx for _ in range(ny)]

    for j in range(ny):
        for i in range(nx):
            # X gradient
            if i == 0:
                grad_x[j][i] = (field[j][1] - field[j][0]) / dx
            elif i == nx - 1:
                grad_x[j][i] = (field[j][nx - 1] - field[j][nx - 2]) / dx
            else:
                grad_x[j][i] = (field[j][i + 1] - field[j][i - 1]) / (2 * dx)

            # Y gradient
            if j == 0:
                grad_y[j][i] = (field[1][i] - field[0][i]) / dy
            elif j == ny - 1:
                grad_y[j][i] = (field[ny - 1][i] - field[ny - 2][i]) / dy
            else:
                grad_y[j][i] = (field[j + 1][i] - field[j - 1][i]) / (2 * dy)

    return grad_x, grad_y


def compute_gradient_magnitude(
    field: List,
    spacing: Dict[str, float]
) -> Dict[str, Any]:
    """Compute gradient magnitude of field."""
    # Check for NaN/Inf in input field
    flat = flatten_field(field)
    for v in flat:
        if not math.isfinite(v):
            raise ValueError(
                f"Field contains non-finite value ({v}); cannot compute gradient"
            )

    shape = get_field_shape(field)

    if len(shape) == 1:
        # 1D gradient — need at least 2 elements
        if shape[0] < 2:
            return {
                "gradient_magnitude": [0.0] * shape[0],
                "max": 0.0,
                "mean": 0.0,
                "method": "central_difference_1d",
                "notes": ["Single-element field; gradient is zero"]
            }
        dx = spacing["dx"]
        grad = []
        for i in range(len(field)):
            if i == 0:
                g = (field[1] - field[0]) / dx
            elif i == len(field) - 1:
                g = (field[-1] - field[-2]) / dx
            else:
                g = (field[i + 1] - field[i - 1]) / (2 * dx)
            grad.append(abs(g))

        return {
            "gradient_magnitude": grad,
            "max": max(grad),
            "mean": sum(grad) / len(grad),
            "method": "central_difference_1d"
        }

    if len(shape) == 2:
        grad_x, grad_y = compute_gradient_2d(
            field, spacing["dx"], spacing["dy"]
        )

        # Compute magnitude
        ny, nx = shape
        mag = [[0.0] * nx for _ in range(ny)]
        all_mag = []

        for j in range(ny):
            for i in range(nx):
                m = math.sqrt(grad_x[j][i] ** 2 + grad_y[j][i] ** 2)
                mag[j][i] = m
                all_mag.append(m)

        return {
            "gradient_magnitude": mag,
            "max": max(all_mag),
            "mean": sum(all_mag) / len(all_mag),
            "method": "central_difference_2d"
        }

    return {"error": "Gradient not implemented for this dimension", "shape": shape}


def compute_integral(
    field: List,
    spacing: Dict[str, float]
) -> Dict[str, Any]:
    """Compute volume integral of field."""
    values = flatten_field(field)
    shape = get_field_shape(field)

    if not values:
        return {"integral": 0, "method": "none"}

    # Compute cell volume
    if len(shape) == 1:
        cell_volume = spacing["dx"]
    elif len(shape) == 2:
        cell_volume = spacing["dx"] * spacing["dy"]
    elif len(shape) == 3:
        cell_volume = spacing["dx"] * spacing["dy"] * spacing["dz"]
    else:
        cell_volume = 1.0

    integral = sum(values) * cell_volume

    return {
        "integral": integral,
        "mean_value": sum(values) / len(values),
        "cell_volume": cell_volume,
        "total_cells": len(values),
        "method": f"midpoint_quadrature_{len(shape)}d"
    }


def compute_total_variation(field: List, spacing: Dict[str, float]) -> Dict[str, Any]:
    """Compute total variation (sum of absolute gradients)."""
    shape = get_field_shape(field)

    if len(shape) == 1:
        dx = spacing["dx"]
        tv = sum(abs(field[i + 1] - field[i]) for i in range(len(field) - 1))
        return {
            "total_variation": tv,
            "normalized_tv": tv / ((len(field) - 1) * dx) if len(field) > 1 else 0,
            "method": "1d_difference"
        }

    if len(shape) == 2:
        ny, nx = shape
        tv = 0.0

        for j in range(ny):
            for i in range(nx - 1):
                tv += abs(field[j][i + 1] - field[j][i])

        for j in range(ny - 1):
            for i in range(nx):
                tv += abs(field[j + 1][i] - field[j][i])

        domain_area = (nx * spacing["dx"]) * (ny * spacing["dy"])
        return {
            "total_variation": tv,
            "normalized_tv": tv / domain_area if domain_area > 0 else 0,
            "method": "2d_difference"
        }

    return {"error": "TV not implemented for this dimension", "shape": shape}


def compute_mass(field: List, spacing: Dict[str, float]) -> Dict[str, Any]:
    """Compute total mass (integral of field)."""
    result = compute_integral(field, spacing)
    return {
        "total_mass": result["integral"],
        "mean_density": result["mean_value"],
        "volume": result["cell_volume"] * result["total_cells"],
        "method": result["method"]
    }


def compute_centroid(field: List, spacing: Dict[str, float]) -> Dict[str, Any]:
    """Compute centroid of field (weighted by values)."""
    shape = get_field_shape(field)

    if len(shape) == 1:
        dx = spacing["dx"]
        total = sum(field)
        if total == 0:
            return {"centroid_x": None, "total": 0}

        cx = sum(i * dx * field[i] for i in range(len(field))) / total
        return {"centroid_x": cx, "total": total}

    if len(shape) == 2:
        ny, nx = shape
        dx, dy = spacing["dx"], spacing["dy"]

        total = sum(flatten_field(field))
        if total == 0:
            return {"centroid_x": None, "centroid_y": None, "total": 0}

        cx = sum(
            i * dx * field[j][i]
            for j in range(ny) for i in range(nx)
        ) / total

        cy = sum(
            j * dy * field[j][i]
            for j in range(ny) for i in range(nx)
        ) / total

        return {"centroid_x": cx, "centroid_y": cy, "total": total}

    return {"error": "Centroid not implemented for this dimension"}


QUANTITY_FUNCTIONS = {
    "volume_fraction": compute_volume_fraction,
    "interface_area": compute_interface_area,
    "gradient_magnitude": compute_gradient_magnitude,
    "integral": compute_integral,
    "total_variation": compute_total_variation,
    "mass": compute_mass,
    "centroid": compute_centroid
}


def main():
    parser = argparse.ArgumentParser(
        description="Compute derived quantities from simulation data"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input field file (JSON)"
    )
    parser.add_argument(
        "--quantity", "-q",
        required=True,
        choices=list(QUANTITY_FUNCTIONS.keys()),
        help="Quantity to compute"
    )
    parser.add_argument(
        "--field", "-f",
        required=True,
        help="Field name to use"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Threshold for volume_fraction/interface_area (default: 0.5)"
    )
    parser.add_argument(
        "--dx",
        type=float,
        help="Grid spacing in x (overrides file data)"
    )
    parser.add_argument(
        "--dy",
        type=float,
        help="Grid spacing in y (overrides file data)"
    )
    parser.add_argument(
        "--dz",
        type=float,
        help="Grid spacing in z (overrides file data)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    try:
        # Load data
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        data = load_json_file(args.input)

        # Get field data
        field = get_field_data(data, args.field)
        if field is None:
            print(f"Error: Field '{args.field}' not found", file=sys.stderr)
            sys.exit(1)

        shape = get_field_shape(field)

        # Get grid spacing
        spacing = get_grid_spacing(data, shape)
        if args.dx:
            spacing["dx"] = args.dx
        if args.dy:
            spacing["dy"] = args.dy
        if args.dz:
            spacing["dz"] = args.dz

        # Compute quantity
        if args.quantity == "volume_fraction":
            result = compute_volume_fraction(field, args.threshold)
        elif args.quantity == "interface_area":
            result = compute_interface_area(field, args.threshold, spacing)
        elif args.quantity == "gradient_magnitude":
            result = compute_gradient_magnitude(field, spacing)
            # Don't include full field in output by default
            if "gradient_magnitude" in result and isinstance(result["gradient_magnitude"], list):
                del result["gradient_magnitude"]
        else:
            func = QUANTITY_FUNCTIONS[args.quantity]
            result = func(field, spacing)

        # Add metadata
        result["source_file"] = args.input
        result["field"] = args.field
        result["quantity"] = args.quantity
        result["shape"] = shape
        result["spacing"] = spacing

        # Output
        if args.json:
            envelope = {
                "inputs": {
                    "input_file": args.input,
                    "quantity": args.quantity,
                    "field": args.field,
                    "threshold": args.threshold,
                    "dx": spacing["dx"],
                    "dy": spacing["dy"],
                },
                "results": result,
            }
            print(json.dumps(envelope, indent=2))
        else:
            print(f"Derived Quantity: {args.quantity}")
            print(f"Source: {args.input}")
            print(f"Field: {args.field}")
            print(f"Shape: {shape}")
            print(f"Grid spacing: dx={spacing['dx']}, dy={spacing['dy']}")
            print()

            # Print main result based on quantity
            if args.quantity == "volume_fraction":
                print(f"Threshold: {result['threshold']}")
                print(f"Condition: {result['condition']}")
                print(f"Volume fraction: {result['volume_fraction']:.6f}")
                print(f"Count: {result['count']} / {result['total']}")

            elif args.quantity == "interface_area":
                if "interface_length" in result:
                    print(f"Interface length: {result['interface_length']:.6f}")
                    print(f"Cells crossed: {result['cells_crossed']}")
                elif "crossings" in result:
                    print(f"Crossings: {result['crossings']}")

            elif args.quantity == "gradient_magnitude":
                print(f"Max gradient: {result['max']:.6g}")
                print(f"Mean gradient: {result['mean']:.6g}")

            elif args.quantity == "integral":
                print(f"Integral: {result['integral']:.6g}")
                print(f"Mean value: {result['mean_value']:.6g}")

            elif args.quantity == "total_variation":
                print(f"Total variation: {result['total_variation']:.6g}")
                print(f"Normalized TV: {result['normalized_tv']:.6g}")

            elif args.quantity == "mass":
                print(f"Total mass: {result['total_mass']:.6g}")
                print(f"Mean density: {result['mean_density']:.6g}")
                print(f"Volume: {result['volume']:.6g}")

            elif args.quantity == "centroid":
                if result.get("centroid_x") is not None:
                    print(f"Centroid X: {result['centroid_x']:.6g}")
                if result.get("centroid_y") is not None:
                    print(f"Centroid Y: {result['centroid_y']:.6g}")

            print(f"\nMethod: {result.get('method', 'N/A')}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
