#!/usr/bin/env python3
import argparse
import json
import math
import sys
from typing import Dict, Optional

import numpy as np


def parse_coeffs(raw: str) -> np.ndarray:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("coeffs must be a comma-separated list")
    return np.array([float(p) for p in parts], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute amplification factor for a linear update stencil.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coeffs",
        required=True,
        help="Comma-separated stencil coefficients from negative to positive index",
    )
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing")
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Index of coefficient corresponding to j=0 (default: center index)",
    )
    parser.add_argument("--kmin", type=float, default=None, help="Minimum wavenumber")
    parser.add_argument("--kmax", type=float, default=None, help="Maximum wavenumber")
    parser.add_argument("--nk", type=int, default=256, help="Number of k samples")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def compute_amplification(
    coeffs: np.ndarray,
    dx: float,
    nk: int,
    offset: Optional[int],
    kmin: Optional[float],
    kmax: Optional[float],
) -> Dict[str, object]:
    if dx <= 0:
        raise ValueError("dx must be positive")
    if nk <= 1:
        raise ValueError("nk must be > 1")

    n = len(coeffs)
    if n == 0:
        raise ValueError("coeffs must be non-empty")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("coeffs must be finite (no NaN or Inf values)")

    if offset is None:
        offset = n // 2
    if offset < 0 or offset >= n:
        raise ValueError("offset must be within coefficient indices")

    if kmin is None:
        kmin = -math.pi / dx
    if kmax is None:
        kmax = math.pi / dx
    if kmin >= kmax:
        raise ValueError("kmin must be < kmax")

    j = np.arange(-offset, n - offset)
    ks = np.linspace(kmin, kmax, nk)
    phase = np.exp(1j * np.outer(ks, j) * dx)
    amplification = phase @ coeffs
    amp_mag = np.abs(amplification)

    max_idx = int(np.argmax(amp_mag))
    max_amp = float(amp_mag[max_idx])
    k_at_max = float(ks[max_idx])
    stable = max_amp <= 1.0 + 1e-12
    warning = None
    if n % 2 == 0 and offset == n // 2:
        warning = "Even-length stencil: confirm offset aligns with j=0."

    return {
        "inputs": {
            "coeffs": coeffs.tolist(),
            "dx": dx,
            "kmin": kmin,
            "kmax": kmax,
            "nk": nk,
            "offset": offset,
        },
        "results": {
            "max_amplification": max_amp,
            "k_at_max": k_at_max,
            "stable": stable,
            "warning": warning,
        },
    }


def main() -> None:
    args = parse_args()
    try:
        coeffs = parse_coeffs(args.coeffs)
        payload = compute_amplification(
            coeffs=coeffs,
            dx=args.dx,
            nk=args.nk,
            offset=args.offset,
            kmin=args.kmin,
            kmax=args.kmax,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Von Neumann analysis")
    print(f"  coeffs: {payload['inputs']['coeffs']}")
    print(
        "  k range: [{:.6g}, {:.6g}] with {} samples".format(
            payload["inputs"]["kmin"],
            payload["inputs"]["kmax"],
            payload["inputs"]["nk"],
        )
    )
    print(
        "  max amplification: {:.6g} at k={:.6g}".format(
            payload["results"]["max_amplification"],
            payload["results"]["k_at_max"],
        )
    )
    print(f"  stable: {payload['results']['stable']}")
    if payload["results"]["warning"]:
        print(f"  warning: {payload['results']['warning']}")


if __name__ == "__main__":
    main()
