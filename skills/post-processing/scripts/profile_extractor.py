#!/usr/bin/env python3
"""
Profile Extractor - Extract 1D line profiles from 2D/3D field data.

Extracts profiles along specified axes or arbitrary lines through
simulation field data.

Usage:
    python profile_extractor.py --input field.json --field phi --axis x --slice-position 0.5 --json
    python profile_extractor.py --input field.json --field phi --start "0,0.5" --end "1,0.5" --json
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


def get_field_shape(field_data: List) -> List[int]:
    """Get shape of field data."""
    shape = []
    current = field_data
    while isinstance(current, list):
        shape.append(len(current))
        if len(current) > 0:
            current = current[0]
        else:
            break
    return shape


def get_grid_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract grid information from data."""
    info = {}

    # Try to get domain size
    for key in ["domain", "grid", "mesh"]:
        if key in data:
            domain = data[key]
            if isinstance(domain, dict):
                info.update(domain)

    # Try individual keys
    for key in ["Lx", "Ly", "Lz", "nx", "ny", "nz", "dx", "dy", "dz"]:
        if key in data:
            info[key] = data[key]

    # Try bounds
    if "bounds" in data:
        bounds = data["bounds"]
        if isinstance(bounds, dict):
            info["bounds"] = bounds

    return info


def interpolate_1d(values: List[float], position: float) -> float:
    """Linear interpolation in 1D array."""
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]

    # Clamp position to [0, 1]
    if position <= 0:
        return values[0]
    if position >= 1:
        return values[-1]

    # Map position [0,1] to index
    idx_float = position * (n - 1)
    idx_low = int(idx_float)
    idx_high = min(idx_low + 1, n - 1)

    frac = idx_float - idx_low
    return values[idx_low] * (1 - frac) + values[idx_high] * frac


def get_value_2d(field: List[List[float]], i: int, j: int) -> float:
    """Get value from 2D field with bounds checking."""
    ny = len(field)
    if ny == 0:
        return 0.0
    nx = len(field[0])

    i = max(0, min(i, nx - 1))
    j = max(0, min(j, ny - 1))

    return field[j][i]


def interpolate_2d(
    field: List[List[float]],
    x_frac: float,
    y_frac: float
) -> float:
    """Bilinear interpolation in 2D field."""
    ny = len(field)
    if ny == 0:
        return 0.0
    nx = len(field[0])

    # Map fractions to indices
    x_idx = x_frac * (nx - 1)
    y_idx = y_frac * (ny - 1)

    i0 = int(x_idx)
    j0 = int(y_idx)
    i1 = min(i0 + 1, nx - 1)
    j1 = min(j0 + 1, ny - 1)

    fx = x_idx - i0
    fy = y_idx - j0

    # Bilinear interpolation
    v00 = get_value_2d(field, i0, j0)
    v10 = get_value_2d(field, i1, j0)
    v01 = get_value_2d(field, i0, j1)
    v11 = get_value_2d(field, i1, j1)

    return (
        v00 * (1 - fx) * (1 - fy) +
        v10 * fx * (1 - fy) +
        v01 * (1 - fx) * fy +
        v11 * fx * fy
    )


def extract_axis_profile(
    field: List,
    axis: str,
    slice_position: float,
    grid_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract profile along axis at slice position."""
    shape = get_field_shape(field)
    ndim = len(shape)

    if ndim == 1:
        # 1D data - just return as is
        return {
            "axis": "x",
            "coordinates": list(range(len(field))),
            "values": list(field),
            "points": len(field)
        }

    if ndim == 2:
        ny, nx = shape

        if axis.lower() == "x":
            # Profile along x at y = slice_position
            j = int(slice_position * (ny - 1))
            j = max(0, min(j, ny - 1))

            values = [field[j][i] for i in range(nx)]
            coords = [i / (nx - 1) if nx > 1 else 0.5 for i in range(nx)]

            # Adjust coordinates if domain info available
            if "Lx" in grid_info:
                coords = [c * grid_info["Lx"] for c in coords]

            return {
                "axis": "x",
                "slice_axis": "y",
                "slice_position": slice_position,
                "slice_index": j,
                "coordinates": coords,
                "values": values,
                "points": nx
            }

        elif axis.lower() == "y":
            # Profile along y at x = slice_position
            i = int(slice_position * (nx - 1))
            i = max(0, min(i, nx - 1))

            values = [field[j][i] for j in range(ny)]
            coords = [j / (ny - 1) if ny > 1 else 0.5 for j in range(ny)]

            if "Ly" in grid_info:
                coords = [c * grid_info["Ly"] for c in coords]

            return {
                "axis": "y",
                "slice_axis": "x",
                "slice_position": slice_position,
                "slice_index": i,
                "coordinates": coords,
                "values": values,
                "points": ny
            }

    raise ValueError(f"Cannot extract {axis} profile from {ndim}D data")


def extract_line_profile(
    field: List,
    start: Tuple[float, ...],
    end: Tuple[float, ...],
    num_points: int,
    grid_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract profile along arbitrary line."""
    shape = get_field_shape(field)
    ndim = len(shape)

    if ndim == 1:
        # 1D - return whole array
        return {
            "start": (0.0,),
            "end": (1.0,),
            "coordinates": list(range(len(field))),
            "values": list(field),
            "points": len(field)
        }

    if ndim == 2:
        ny, nx = shape

        # Generate points along line
        values = []
        distances = []

        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0.5

            # Interpolate position
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])

            # Clamp to [0, 1]
            x_frac = max(0, min(1, x))
            y_frac = max(0, min(1, y))

            # Interpolate value
            value = interpolate_2d(field, x_frac, y_frac)
            values.append(value)

            # Distance along line
            dx = x - start[0]
            dy = y - start[1]
            dist = math.sqrt(dx * dx + dy * dy)
            distances.append(dist)

        # Normalize distances
        if distances[-1] > 0:
            line_length = math.sqrt(
                (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
            )
            # Scale by domain if available
            if "Lx" in grid_info and "Ly" in grid_info:
                line_length *= math.sqrt(
                    grid_info["Lx"] ** 2 + grid_info["Ly"] ** 2
                )

        return {
            "start": start,
            "end": end,
            "coordinates": distances,
            "values": values,
            "points": num_points,
            "line_length": distances[-1] if distances else 0
        }

    raise ValueError(f"Line profiles not supported for {ndim}D data")


def parse_point(s: str) -> Tuple[float, ...]:
    """Parse point string like '0.5,0.5' or '0,0.5,0'."""
    parts = s.strip().split(",")
    return tuple(float(p.strip()) for p in parts)


def compute_profile_statistics(values: List[float]) -> Dict[str, Any]:
    """Compute statistics for profile values."""
    if not values:
        return {}

    n = len(values)
    mean = sum(values) / n

    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": std,
        "range": max(values) - min(values)
    }


def detect_interface(
    values: List[float],
    coordinates: List[float],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Detect interface location in profile."""
    crossings = []

    for i in range(1, len(values)):
        v0, v1 = values[i - 1], values[i]
        if (v0 - threshold) * (v1 - threshold) < 0:
            # Linear interpolation to find crossing
            frac = (threshold - v0) / (v1 - v0)
            x_cross = coordinates[i - 1] + frac * (coordinates[i] - coordinates[i - 1])
            crossings.append({
                "position": x_cross,
                "index": i - 1 + frac,
                "direction": "rising" if v1 > v0 else "falling"
            })

    return {
        "threshold": threshold,
        "crossings": crossings,
        "count": len(crossings)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract line profiles from field data"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input field file (JSON)"
    )
    parser.add_argument(
        "--field", "-f",
        required=True,
        help="Field name to extract"
    )
    parser.add_argument(
        "--axis", "-a",
        choices=["x", "y", "z"],
        help="Axis to extract profile along"
    )
    parser.add_argument(
        "--slice-position",
        type=float,
        default=0.5,
        help="Position along perpendicular axis (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--start",
        help="Start point for line profile (e.g., '0,0.5')"
    )
    parser.add_argument(
        "--end",
        help="End point for line profile (e.g., '1,0.5')"
    )
    parser.add_argument(
        "--points", "-n",
        type=int,
        default=100,
        help="Number of points for line profile (default: 100)"
    )
    parser.add_argument(
        "--detect-interface",
        action="store_true",
        help="Detect interface crossings"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for interface detection (default: 0.5)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.axis is None and (args.start is None or args.end is None):
        print("Error: Either --axis or both --start and --end required",
              file=sys.stderr)
        sys.exit(1)

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

        # Get grid info
        grid_info = get_grid_info(data)

        # Extract profile
        if args.axis:
            result = extract_axis_profile(
                field, args.axis, args.slice_position, grid_info
            )
        else:
            start = parse_point(args.start)
            end = parse_point(args.end)
            result = extract_line_profile(
                field, start, end, args.points, grid_info
            )

        # Add metadata
        result["source_file"] = args.input
        result["field"] = args.field
        result["statistics"] = compute_profile_statistics(result["values"])

        # Detect interface if requested
        if args.detect_interface:
            result["interface"] = detect_interface(
                result["values"],
                result["coordinates"],
                args.threshold
            )

        # Output
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Profile Extraction: {args.field}")
            print(f"Source: {args.input}")

            if "axis" in result:
                print(f"Axis: {result['axis']}")
                if "slice_position" in result:
                    print(f"Slice at {result['slice_axis']}={result['slice_position']:.3f}")

            if "start" in result and "end" in result:
                print(f"Line: {result['start']} -> {result['end']}")
                print(f"Length: {result.get('line_length', 'N/A')}")

            print(f"Points: {result['points']}")

            stats = result["statistics"]
            print(f"\nStatistics:")
            print(f"  Min: {stats['min']:.6g}")
            print(f"  Max: {stats['max']:.6g}")
            print(f"  Mean: {stats['mean']:.6g}")
            print(f"  Range: {stats['range']:.6g}")

            if "interface" in result:
                iface = result["interface"]
                print(f"\nInterface Detection (threshold={iface['threshold']}):")
                print(f"  Crossings found: {iface['count']}")
                for c in iface["crossings"]:
                    print(f"    {c['direction']} at x={c['position']:.4f}")

            # Print first/last few values
            print(f"\nProfile data (first 5):")
            for i in range(min(5, result["points"])):
                x = result["coordinates"][i]
                y = result["values"][i]
                print(f"  {x:.4f}: {y:.6g}")

            if result["points"] > 5:
                print("  ...")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
