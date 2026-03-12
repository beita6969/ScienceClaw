#!/usr/bin/env python3
"""
Statistical Analyzer - Compute statistics on simulation field data.

Computes global and regional statistics, distributions, and correlations
for field data from simulations.

Usage:
    python statistical_analyzer.py --input field.json --field phi --json
    python statistical_analyzer.py --input field.json --field phi --histogram --bins 50 --json
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
    """Flatten nested list to 1D array of floats."""
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


def compute_basic_statistics(values: List[float]) -> Dict[str, Any]:
    """Compute basic descriptive statistics."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "variance": None
        }

    n = len(values)
    mean = sum(values) / n
    min_val = min(values)
    max_val = max(values)

    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        variance = 0.0
        std = 0.0

    return {
        "count": n,
        "min": min_val,
        "max": max_val,
        "range": max_val - min_val,
        "mean": mean,
        "std": std,
        "variance": variance
    }


def compute_percentiles(
    values: List[float],
    percentiles: List[float] = None
) -> Dict[str, float]:
    """Compute specified percentiles."""
    if percentiles is None:
        percentiles = [0, 25, 50, 75, 100]

    if not values:
        return {f"p{int(p)}": None for p in percentiles}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    result = {}
    for p in percentiles:
        idx = (p / 100) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower

        value = sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac
        result[f"p{int(p)}"] = value

    return result


def compute_median(values: List[float]) -> Optional[float]:
    """Compute median value."""
    if not values:
        return None

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    if n % 2 == 0:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    else:
        return sorted_vals[n // 2]


def compute_skewness(values: List[float], mean: float, std: float) -> Optional[float]:
    """Compute skewness (third standardized moment)."""
    if len(values) < 3 or std == 0:
        return None

    n = len(values)
    m3 = sum((x - mean) ** 3 for x in values) / n
    return m3 / (std ** 3)


def compute_kurtosis(values: List[float], mean: float, std: float) -> Optional[float]:
    """Compute excess kurtosis (fourth standardized moment - 3)."""
    if len(values) < 4 or std == 0:
        return None

    n = len(values)
    m4 = sum((x - mean) ** 4 for x in values) / n
    return (m4 / (std ** 4)) - 3


def compute_histogram(
    values: List[float],
    num_bins: int = 50,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None
) -> Dict[str, Any]:
    """Compute histogram of values."""
    if not values:
        return {"bins": [], "counts": [], "densities": []}

    if range_min is None:
        range_min = min(values)
    if range_max is None:
        range_max = max(values)

    if range_max == range_min:
        range_max = range_min + 1  # Avoid division by zero

    bin_width = (range_max - range_min) / num_bins
    bin_edges = [range_min + i * bin_width for i in range(num_bins + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]

    counts = [0] * num_bins
    for v in values:
        if v < range_min:
            counts[0] += 1
        elif v >= range_max:
            counts[-1] += 1
        else:
            idx = int((v - range_min) / bin_width)
            idx = min(idx, num_bins - 1)
            counts[idx] += 1

    total = sum(counts)
    densities = [c / (total * bin_width) if total > 0 else 0 for c in counts]

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_width": bin_width,
        "counts": counts,
        "densities": densities,
        "total": total
    }


def detect_distribution_type(
    values: List[float],
    histogram: Dict[str, Any]
) -> Dict[str, Any]:
    """Attempt to classify distribution type."""
    if not values:
        return {"type": "unknown", "confidence": 0}

    counts = histogram["counts"]
    if not counts:
        return {"type": "unknown", "confidence": 0}

    n_bins = len(counts)
    total = sum(counts)

    # Find peaks (including edge bins)
    peaks = []

    # Check left edge
    if n_bins >= 2 and counts[0] > counts[1]:
        peaks.append(0)

    # Check interior peaks
    for i in range(1, n_bins - 1):
        if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
            peaks.append(i)

    # Check right edge
    if n_bins >= 2 and counts[-1] > counts[-2]:
        peaks.append(n_bins - 1)

    # Check for bimodal with gap in middle (e.g., values at 0 and 1 only)
    # This is common in phase-field simulations
    non_zero_bins = [i for i, c in enumerate(counts) if c > 0]
    if len(non_zero_bins) >= 2:
        # Check if there's a gap between non-zero bins
        first_nz = non_zero_bins[0]
        last_nz = non_zero_bins[-1]
        zero_in_middle = any(counts[i] == 0 for i in range(first_nz + 1, last_nz))
        if zero_in_middle and counts[first_nz] > 0 and counts[last_nz] > 0:
            # Bimodal with gap - typical phase-field distribution
            return {"type": "bimodal", "confidence": 0.9, "peaks": 2}

    # Analyze distribution shape
    if len(peaks) == 0:
        # Monotonic - check if uniform or exponential
        if counts[0] > 0 and counts[-1] > 0:
            non_zero_counts = [c for c in counts if c > 0]
            if non_zero_counts:
                ratio = max(non_zero_counts) / min(non_zero_counts)
                if ratio < 2:
                    return {"type": "uniform", "confidence": 0.8, "peaks": 0}
        return {"type": "monotonic", "confidence": 0.5, "peaks": 0}

    elif len(peaks) == 1:
        # Single peak - could be normal, skewed, etc.
        peak_idx = peaks[0]
        center = n_bins // 2

        if abs(peak_idx - center) < n_bins * 0.1:
            return {"type": "unimodal_symmetric", "confidence": 0.7, "peaks": 1}
        else:
            return {"type": "unimodal_skewed", "confidence": 0.6, "peaks": 1}

    elif len(peaks) == 2:
        # Bimodal - common for two-phase systems
        return {"type": "bimodal", "confidence": 0.8, "peaks": 2}

    else:
        return {"type": "multimodal", "confidence": 0.5, "peaks": len(peaks)}


def parse_region_condition(condition: str) -> callable:
    """Parse simple region condition string like 'x>0.3 and x<0.7'."""
    # For simplicity, return a function that always returns True
    # In a full implementation, this would parse and evaluate conditions

    def always_true(*args):
        return True

    return always_true


def compute_regional_statistics(
    field: List,
    region_mask: Optional[List] = None
) -> Dict[str, Any]:
    """Compute statistics for a region of the field."""
    flat = flatten_field(field)

    if region_mask:
        flat_mask = flatten_field(region_mask)
        values = [v for v, m in zip(flat, flat_mask) if m]
    else:
        values = flat

    return compute_basic_statistics(values)


def analyze_spatial_variation(field: List) -> Dict[str, Any]:
    """Analyze spatial variation of field."""
    shape = get_field_shape(field)

    if len(shape) < 2:
        return {"type": "1D", "spatial_analysis": "not_applicable"}

    # For 2D fields
    if len(shape) == 2:
        ny, nx = shape

        # Row-wise statistics
        row_means = []
        for j in range(ny):
            row_sum = sum(field[j])
            row_means.append(row_sum / nx)

        # Column-wise statistics
        col_means = []
        for i in range(nx):
            col_sum = sum(field[j][i] for j in range(ny))
            col_means.append(col_sum / ny)

        # Variation along each axis
        row_var = compute_basic_statistics(row_means)
        col_var = compute_basic_statistics(col_means)

        return {
            "dimensions": 2,
            "shape": shape,
            "x_variation": {
                "mean_range": col_var["range"],
                "std": col_var["std"]
            },
            "y_variation": {
                "mean_range": row_var["range"],
                "std": row_var["std"]
            },
            "is_x_uniform": col_var["std"] < 0.01 * abs(col_var["mean"]) if col_var["mean"] else True,
            "is_y_uniform": row_var["std"] < 0.01 * abs(row_var["mean"]) if row_var["mean"] else True
        }

    return {"dimensions": len(shape), "spatial_analysis": "not_implemented"}


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics on simulation field data"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input field file (JSON)"
    )
    parser.add_argument(
        "--field", "-f",
        required=True,
        help="Field name to analyze"
    )
    parser.add_argument(
        "--region",
        help="Region condition (e.g., 'x>0.3 and x<0.7')"
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Compute histogram"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)"
    )
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="Analyze spatial variation"
    )
    parser.add_argument(
        "--percentiles",
        help="Comma-separated percentiles to compute (e.g., '5,25,50,75,95')"
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
        values = flatten_field(field)

        if not values:
            print(f"Error: No numeric values in field '{args.field}'", file=sys.stderr)
            sys.exit(1)

        # Compute statistics
        result = {
            "source_file": args.input,
            "field": args.field,
            "shape": shape,
            "basic_statistics": compute_basic_statistics(values)
        }

        # Add median and moments
        stats = result["basic_statistics"]
        result["median"] = compute_median(values)
        result["skewness"] = compute_skewness(values, stats["mean"], stats["std"])
        result["kurtosis"] = compute_kurtosis(values, stats["mean"], stats["std"])

        # Percentiles
        if args.percentiles:
            pcts = [float(p) for p in args.percentiles.split(",")]
        else:
            pcts = [0, 5, 25, 50, 75, 95, 100]
        result["percentiles"] = compute_percentiles(values, pcts)

        # Histogram
        if args.histogram:
            hist = compute_histogram(values, args.bins)
            result["histogram"] = hist
            result["distribution"] = detect_distribution_type(values, hist)

        # Spatial analysis
        if args.spatial:
            result["spatial_variation"] = analyze_spatial_variation(field)

        # Regional statistics (if requested)
        if args.region:
            # Note: Full region parsing not implemented
            result["region"] = {
                "condition": args.region,
                "note": "Region filtering requires coordinate data"
            }

        # Output
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Statistical Analysis: {args.field}")
            print(f"Source: {args.input}")
            print(f"Shape: {shape}")
            print()

            stats = result["basic_statistics"]
            print("Basic Statistics:")
            print(f"  Count: {stats['count']}")
            print(f"  Min: {stats['min']:.6g}")
            print(f"  Max: {stats['max']:.6g}")
            print(f"  Range: {stats['range']:.6g}")
            print(f"  Mean: {stats['mean']:.6g}")
            print(f"  Std: {stats['std']:.6g}")
            print(f"  Median: {result['median']:.6g}")

            if result["skewness"] is not None:
                print(f"  Skewness: {result['skewness']:.4f}")
            if result["kurtosis"] is not None:
                print(f"  Kurtosis: {result['kurtosis']:.4f}")

            print("\nPercentiles:")
            for k, v in result["percentiles"].items():
                if v is not None:
                    print(f"  {k}: {v:.6g}")

            if "histogram" in result:
                print("\nHistogram computed with", args.bins, "bins")
                dist = result.get("distribution", {})
                print(f"Distribution type: {dist.get('type', 'unknown')}")
                print(f"  Peaks detected: {dist.get('peaks', 0)}")

            if "spatial_variation" in result:
                sv = result["spatial_variation"]
                print("\nSpatial Variation:")
                if "x_variation" in sv:
                    print(f"  X-direction std: {sv['x_variation']['std']:.6g}")
                if "y_variation" in sv:
                    print(f"  Y-direction std: {sv['y_variation']['std']:.6g}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
