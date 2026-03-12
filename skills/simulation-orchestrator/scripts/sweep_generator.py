#!/usr/bin/env python3
"""Generate parameter sweep configurations for multi-simulation campaigns.

This script creates multiple configuration files by varying parameters across
specified ranges. Supports grid (full factorial), linspace (uniform), and
LHS (Latin Hypercube Sampling) methods.

Usage:
    python sweep_generator.py --base-config sim.json --params "dt:1e-4:1e-2:5" --method linspace --output-dir ./sweep

Output (JSON):
    {
        "configs": ["config_0000.json", "config_0001.json", ...],
        "parameter_space": {"dt": [0.0001, 0.00325, 0.0055, 0.00775, 0.01]},
        "sweep_method": "linspace",
        "total_runs": 5
    }
"""

import argparse
import copy
import itertools
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple


def parse_param_spec(spec: str) -> Tuple[str, float, float, int]:
    """Parse parameter specification string.

    Formats:
        name:min:max:count  -> linspace/grid with count points
        name:min:max        -> for LHS, just bounds (count from --samples)

    Returns:
        (name, min_val, max_val, count) where count=-1 if not specified
    """
    parts = spec.strip().split(":")
    if len(parts) == 4:
        name, min_val, max_val, count = parts
        fmin, fmax = float(min_val), float(max_val)
        if fmin >= fmax:
            raise ValueError(
                f"Invalid range for '{name}': min ({fmin}) must be less than max ({fmax})"
            )
        return name, fmin, fmax, int(count)
    elif len(parts) == 3:
        name, min_val, max_val = parts
        fmin, fmax = float(min_val), float(max_val)
        if fmin >= fmax:
            raise ValueError(
                f"Invalid range for '{name}': min ({fmin}) must be less than max ({fmax})"
            )
        return name, fmin, fmax, -1
    else:
        raise ValueError(
            f"Invalid param spec: {spec}. Use 'name:min:max:count' or 'name:min:max'"
        )


def parse_params(params_str: str) -> List[Tuple[str, float, float, int]]:
    """Parse comma-separated parameter specifications."""
    specs = [s.strip() for s in params_str.split(",") if s.strip()]
    return [parse_param_spec(s) for s in specs]


def linspace(start: float, stop: float, count: int) -> List[float]:
    """Generate linearly spaced values."""
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def generate_grid(params: List[Tuple[str, float, float, int]]) -> List[Dict[str, float]]:
    """Generate full factorial grid of parameter combinations."""
    param_names = [p[0] for p in params]
    param_values = [linspace(p[1], p[2], p[3]) for p in params]

    configs = []
    for combo in itertools.product(*param_values):
        configs.append(dict(zip(param_names, combo)))

    return configs


def generate_linspace(
    params: List[Tuple[str, float, float, int]]
) -> Tuple[List[Dict[str, float]], Dict[str, List[float]]]:
    """Generate grid sweep using linspace for each parameter."""
    configs = generate_grid(params)
    param_space = {p[0]: linspace(p[1], p[2], p[3]) for p in params}
    return configs, param_space


def generate_lhs(
    params: List[Tuple[str, float, float, int]], samples: int, seed: int = 42
) -> Tuple[List[Dict[str, float]], Dict[str, List[float]]]:
    """Generate Latin Hypercube Sampling configurations."""
    random.seed(seed)
    n_params = len(params)

    # Create intervals for each dimension
    configs = []
    param_space: Dict[str, List[float]] = {p[0]: [] for p in params}

    # Generate LHS: each parameter range divided into n intervals,
    # one sample per interval, randomly ordered
    for i in range(n_params):
        name, min_val, max_val, _ = params[i]
        interval_size = (max_val - min_val) / samples
        # Create one point per interval
        points = []
        for j in range(samples):
            low = min_val + j * interval_size
            high = low + interval_size
            points.append(random.uniform(low, high))
        random.shuffle(points)
        param_space[name] = points

    # Combine into configs
    for i in range(samples):
        config = {}
        for name in param_space:
            config[name] = param_space[name][i]
        configs.append(config)

    return configs, param_space


def load_base_config(path: str) -> Dict[str, Any]:
    """Load base configuration file."""
    with open(path, "r") as f:
        return json.load(f)


def merge_config(base: Dict[str, Any], overrides: Dict[str, float]) -> Dict[str, Any]:
    """Merge override parameters into base config."""
    result = copy.deepcopy(base)
    result.update(overrides)
    return result


def write_configs(
    base_config: Dict[str, Any],
    param_configs: List[Dict[str, float]],
    output_dir: str,
) -> List[str]:
    """Write configuration files to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    written = []

    for i, params in enumerate(param_configs):
        merged = merge_config(base_config, params)
        filename = f"config_{i:04d}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(merged, f, indent=2)
        written.append(filename)

    return written


def generate_sweep(
    base_config_path: str,
    params_str: str,
    method: str,
    output_dir: str,
    samples: int = 10,
    seed: int = 42,
    force: bool = False,
) -> Dict[str, Any]:
    """Generate parameter sweep configurations.

    Args:
        base_config_path: Path to base configuration JSON
        params_str: Comma-separated parameter specifications
        method: Sweep method (grid, linspace, lhs)
        output_dir: Directory to write configurations
        samples: Number of samples for LHS method
        seed: Random seed for reproducibility
        force: Overwrite existing output directory

    Returns:
        Dictionary with configs, parameter_space, sweep_method, total_runs
    """
    # Validate inputs
    if not os.path.exists(base_config_path):
        raise ValueError(f"Base config not found: {base_config_path}")

    if os.path.exists(output_dir) and not force:
        raise ValueError(
            f"Output directory exists: {output_dir}. Use --force to overwrite."
        )

    # Parse parameters
    params = parse_params(params_str)
    if not params:
        raise ValueError("No parameters specified")

    # Load base config
    base_config = load_base_config(base_config_path)

    # Generate parameter combinations
    if method == "grid":
        for p in params:
            if p[3] <= 0:
                raise ValueError(f"Grid method requires count for parameter {p[0]}")
        configs, param_space = generate_linspace(params)
    elif method == "linspace":
        for p in params:
            if p[3] <= 0:
                raise ValueError(f"Linspace method requires count for parameter {p[0]}")
        configs, param_space = generate_linspace(params)
    elif method == "lhs":
        configs, param_space = generate_lhs(params, samples, seed)
    else:
        raise ValueError(f"Unknown method: {method}. Use grid, linspace, or lhs.")

    # Write configuration files
    written = write_configs(base_config, configs, output_dir)

    # Write manifest
    manifest = {
        "configs": written,
        "parameter_space": param_space,
        "sweep_method": method,
        "total_runs": len(configs),
        "base_config": os.path.basename(base_config_path),
        "parameters": [{"name": p[0], "min": p[1], "max": p[2]} for p in params],
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate parameter sweep configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Path to base configuration JSON file",
    )
    parser.add_argument(
        "--params",
        required=True,
        help="Parameter specs: 'name:min:max:count,...' or 'name:min:max,...' for LHS",
    )
    parser.add_argument(
        "--method",
        choices=["grid", "linspace", "lhs"],
        default="linspace",
        help="Sweep method",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write configuration files",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples for LHS method",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
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
        result = generate_sweep(
            base_config_path=args.base_config,
            params_str=args.params,
            method=args.method,
            output_dir=args.output_dir,
            samples=args.samples,
            seed=args.seed,
            force=args.force,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    if args.json:
        # Output subset for JSON (exclude full param_space for brevity)
        output = {
            "configs": result["configs"],
            "parameter_space": {
                k: [round(v, 8) for v in vals]
                for k, vals in result["parameter_space"].items()
            },
            "sweep_method": result["sweep_method"],
            "total_runs": result["total_runs"],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Generated {result['total_runs']} configurations")
        print(f"Method: {result['sweep_method']}")
        print(f"Output directory: {args.output_dir}")
        print(f"Configs: {', '.join(result['configs'][:5])}...")


if __name__ == "__main__":
    main()
