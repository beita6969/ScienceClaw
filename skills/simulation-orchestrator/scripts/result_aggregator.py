#!/usr/bin/env python3
"""Aggregate results from completed simulation campaign jobs.

This script collects results from multiple simulation runs and provides
summary statistics, identifies best/worst runs, and exports combined data.

Usage:
    python result_aggregator.py --campaign-dir ./sweep --metric objective_value

Output (JSON):
    {
        "summary": {"count": 15, "completed": 12, "failed": 3},
        "statistics": {"min": 0.1, "max": 0.9, "mean": 0.45, "std": 0.2},
        "best_run": {"job_id": "job_0005", "value": 0.1, "params": {...}},
        "failed_runs": ["job_0003", "job_0008", "job_0012"]
    }
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def load_campaign(config_dir: str) -> Dict[str, Any]:
    """Load campaign state from config directory."""
    campaign_path = os.path.join(config_dir, "campaign.json")
    if not os.path.exists(campaign_path):
        raise ValueError(f"Campaign not found: {campaign_path}")
    with open(campaign_path, "r") as f:
        return json.load(f)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a configuration file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def find_result_file(job: Dict[str, Any], config_dir: str) -> Optional[str]:
    """Find the result file for a job.

    Args:
        job: Job dictionary
        config_dir: Campaign directory

    Returns:
        Path to result file if found, None otherwise
    """
    job_id = job["job_id"]
    config_file = job["config_file"]

    # Try common result file patterns
    patterns = [
        f"result_{job_id}.json",
        f"{job_id}_result.json",
        config_file.replace("config_", "result_"),
        f"output_{job_id}.json",
        job.get("output_file", ""),
    ]

    for pattern in patterns:
        if pattern:
            result_path = os.path.join(config_dir, pattern)
            if os.path.exists(result_path):
                return result_path

    return None


def load_result(result_path: str) -> Dict[str, Any]:
    """Load a result file."""
    with open(result_path, "r") as f:
        return json.load(f)


def extract_metric(result: Dict[str, Any], metric: str) -> Optional[float]:
    """Extract a metric value from a result dictionary.

    Supports nested keys like "results.objective" via dot notation.

    Args:
        result: Result dictionary
        metric: Metric name (supports dot notation for nested keys)

    Returns:
        Metric value if found, None otherwise
    """
    keys = metric.split(".")
    value = result

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None

    if isinstance(value, (int, float)):
        return float(value)
    return None


def compute_statistics(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute basic statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with min, max, mean, std, median
    """
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None, "median": None}

    n = len(values)
    mean_val = sum(values) / n
    sorted_vals = sorted(values)

    if n % 2 == 0:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    else:
        median_val = sorted_vals[n // 2]

    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0

    return {
        "min": min(values),
        "max": max(values),
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
    }


def aggregate_results(
    config_dir: str,
    metric: str,
    minimize: bool = True,
) -> Dict[str, Any]:
    """Aggregate results from all completed jobs.

    Args:
        config_dir: Campaign directory
        metric: Metric name to extract and analyze
        minimize: If True, best = lowest value; if False, best = highest

    Returns:
        Dictionary with summary, statistics, best_run, failed_runs
    """
    campaign = load_campaign(config_dir)

    results_data = []
    failed_runs = []
    completed_count = 0

    for job in campaign["jobs"]:
        job_id = job["job_id"]

        if job["status"] == "failed":
            failed_runs.append(job_id)
            continue

        result_path = find_result_file(job, config_dir)
        if result_path is None:
            continue

        try:
            result = load_result(result_path)
            value = extract_metric(result, metric)

            if value is not None:
                config = load_config(job["config_path"])
                results_data.append({
                    "job_id": job_id,
                    "value": value,
                    "config_file": job["config_file"],
                    "params": config,
                })
                completed_count += 1
        except (json.JSONDecodeError, IOError):
            failed_runs.append(job_id)

    # Compute statistics
    values = [r["value"] for r in results_data]
    statistics = compute_statistics(values)

    # Find best run
    best_run = None
    if results_data:
        if minimize:
            best_run = min(results_data, key=lambda x: x["value"])
        else:
            best_run = max(results_data, key=lambda x: x["value"])

    # Find worst run
    worst_run = None
    if results_data:
        if minimize:
            worst_run = max(results_data, key=lambda x: x["value"])
        else:
            worst_run = min(results_data, key=lambda x: x["value"])

    return {
        "summary": {
            "total_jobs": len(campaign["jobs"]),
            "completed": completed_count,
            "failed": len(failed_runs),
            "metric": metric,
            "minimize": minimize,
        },
        "statistics": statistics,
        "best_run": best_run,
        "worst_run": worst_run,
        "failed_runs": failed_runs,
        "all_results": results_data,
    }


def export_table(results: Dict[str, Any], output_path: str) -> None:
    """Export results to a CSV-like table.

    Args:
        results: Aggregation results
        output_path: Path to write table
    """
    all_results = results.get("all_results", [])
    if not all_results:
        return

    # Get all parameter keys
    param_keys = set()
    for r in all_results:
        param_keys.update(r.get("params", {}).keys())
    param_keys = sorted(param_keys)

    # Write CSV
    with open(output_path, "w") as f:
        # Header
        header = ["job_id", "value"] + param_keys
        f.write(",".join(header) + "\n")

        # Rows
        for r in all_results:
            row = [r["job_id"], str(r["value"])]
            for key in param_keys:
                row.append(str(r.get("params", {}).get(key, "")))
            f.write(",".join(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate results from simulation campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--campaign-dir",
        required=True,
        help="Directory containing campaign files",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help="Metric to extract (supports dot notation, e.g., 'results.energy')",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Maximize metric instead of minimize",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        results = aggregate_results(
            config_dir=args.campaign_dir,
            metric=args.metric,
            minimize=not args.maximize,
        )

        if args.export_csv:
            export_table(results, args.export_csv)

        if args.json:
            # Simplified output (exclude all_results for brevity)
            output = {
                "summary": results["summary"],
                "statistics": {
                    k: round(v, 8) if v is not None else None
                    for k, v in results["statistics"].items()
                },
                "best_run": results["best_run"],
                "failed_runs": results["failed_runs"],
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Metric: {args.metric}")
            print(f"Completed: {results['summary']['completed']} / {results['summary']['total_jobs']}")
            print(f"Failed: {results['summary']['failed']}")
            print()
            stats = results["statistics"]
            if stats["mean"] is not None:
                print(f"Statistics:")
                print(f"  Min: {stats['min']:.6g}")
                print(f"  Max: {stats['max']:.6g}")
                print(f"  Mean: {stats['mean']:.6g}")
                print(f"  Std: {stats['std']:.6g}")
                print(f"  Median: {stats['median']:.6g}")
            if results["best_run"]:
                print()
                print(f"Best run: {results['best_run']['job_id']} = {results['best_run']['value']:.6g}")

    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
