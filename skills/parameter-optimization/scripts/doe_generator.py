#!/usr/bin/env python3
import argparse
import json
import random
import sys
import warnings
from typing import Dict, List


def lhs_samples(dim: int, budget: int, seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    samples = []
    for d in range(dim):
        points = [(i + rng.random()) / budget for i in range(budget)]
        rng.shuffle(points)
        if d == 0:
            samples = [[p] for p in points]
        else:
            for i, p in enumerate(points):
                samples[i].append(p)
    return samples


def quasi_random_samples(dim: int, budget: int, seed: int) -> List[List[float]]:
    """Generate quasi-random samples using additive recurrence.

    Note: This is a simplified quasi-random sequence, not a true Sobol sequence.
    For production use, consider scipy.stats.qmc.Sobol for actual Sobol sequences.
    """
    rng = random.Random(seed)
    # Use golden ratio based quasi-random for better uniformity than pure random
    phi = (1 + 5 ** 0.5) / 2  # golden ratio
    alpha = [((i + 1) * phi) % 1 for i in range(dim)]
    samples = []
    start = rng.random()
    for n in range(budget):
        point = [((start + (n + 1) * alpha[d]) % 1) for d in range(dim)]
        samples.append(point)
    return samples


def factorial_samples(dim: int, budget: int) -> List[List[float]]:
    levels = int(round(budget ** (1.0 / dim)))
    levels = max(levels, 2)
    grid = [i / (levels - 1) for i in range(levels)]
    samples = [[]]
    for _ in range(dim):
        samples = [s + [g] for s in samples for g in grid]
    return samples[:budget]


def generate_doe(dim: int, budget: int, method: str, seed: int) -> Dict[str, object]:
    if dim <= 0:
        raise ValueError("params must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")
    valid_methods = {"lhs", "sobol", "quasi-random", "factorial"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of: {', '.join(sorted(valid_methods))}")

    if method == "lhs":
        samples = lhs_samples(dim, budget, seed)
    elif method in {"sobol", "quasi-random"}:
        if method == "sobol":
            warnings.warn(
                "Method 'sobol' is deprecated; use 'quasi-random' instead. "
                "This is NOT a true Sobol sequence but a quasi-random additive recurrence.",
                DeprecationWarning,
                stacklevel=2,
            )
        samples = quasi_random_samples(dim, budget, seed)
    else:
        samples = factorial_samples(dim, budget)

    return {
        "method": method,
        "samples": samples,
        "coverage": {"count": len(samples), "dimension": dim},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate design of experiments samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--params", type=int, required=True, help="Number of parameters")
    parser.add_argument("--budget", type=int, required=True, help="Sample budget")
    parser.add_argument(
        "--method",
        choices=["lhs", "sobol", "quasi-random", "factorial"],
        default="lhs",
        help="DOE method (sobol uses quasi-random sequence)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_doe(args.params, args.budget, args.method, args.seed)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "params": args.params,
            "budget": args.budget,
            "method": args.method,
            "seed": args.seed,
        },
        "results": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("DOE samples")
    print(f"  method: {result['method']}")
    print(f"  count: {result['coverage']['count']}")


if __name__ == "__main__":
    main()
