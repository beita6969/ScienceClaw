#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


def get_free_disk_space_gb(path: str) -> Optional[float]:
    """Get free disk space in GB. Cross-platform (Windows, Linux, macOS)."""
    try:
        if sys.platform == "win32":
            # Windows: use ctypes to call GetDiskFreeSpaceExW
            import ctypes

            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(os.path.abspath(path)),
                None,
                None,
                ctypes.pointer(free_bytes),
            )
            return free_bytes.value / (1024**3)
        else:
            # Unix-like: use os.statvfs
            stat = os.statvfs(path)
            return (stat.f_bavail * stat.f_frsize) / (1024**3)
    except (OSError, AttributeError):
        return None


def load_config(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise ValueError(f"Config not found: {path}")
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    # Minimal YAML-like fallback: key: value per line
    config: Dict[str, object] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                if "." in value or "e" in value.lower():
                    parsed = float(value)
                else:
                    parsed = int(value)
            except ValueError:
                parsed = value
            config[key] = parsed
    return config


def parse_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def parse_ranges(raw: Optional[str]) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    if not raw:
        return ranges
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError("range entries must be name:min:max")
        name, min_val, max_val = part.split(":", 2)
        ranges[name.strip()] = (float(min_val), float(max_val))
    return ranges


def preflight_check(
    config: Dict[str, object],
    required: List[str],
    ranges: Dict[str, Tuple[float, float]],
    output_dir: Optional[str],
    min_free_gb: float,
) -> Dict[str, object]:
    blockers: List[str] = []
    warnings: List[str] = []

    params = config.get("parameters", {})
    if not isinstance(params, dict):
        params = {}

    for key in required:
        if key not in config and key not in params:
            blockers.append(f"Missing required parameter: {key}")

    for key, (min_val, max_val) in ranges.items():
        value = config.get(key, params.get(key))
        if value is None:
            warnings.append(f"Range check skipped; missing {key}.")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            blockers.append(f"Non-numeric value for {key}.")
            continue
        if numeric < min_val or numeric > max_val:
            blockers.append(f"{key} out of range [{min_val}, {max_val}].")

    output_dir = output_dir or config.get("output_dir")
    if output_dir:
        if not os.path.exists(output_dir):
            warnings.append("Output directory does not exist; will be created.")
        else:
            if not os.access(output_dir, os.W_OK):
                blockers.append("Output directory not writable.")
    else:
        warnings.append("No output directory specified.")

    if min_free_gb > 0:
        free_gb = get_free_disk_space_gb(".")
        if free_gb is not None and free_gb < min_free_gb:
            blockers.append(f"Insufficient disk space: {free_gb:.2f} GB free.")
        elif free_gb is None:
            warnings.append("Could not determine free disk space.")

    if "material_source" not in config and "materials_source" not in config:
        warnings.append("Material property source not specified.")

    status = "PASS"
    if blockers:
        status = "BLOCK"
    elif warnings:
        status = "WARN"

    return {
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-flight simulation validation checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to simulation config (JSON)")
    parser.add_argument(
        "--required",
        default=None,
        help="Comma-separated required parameters",
    )
    parser.add_argument(
        "--ranges",
        default=None,
        help="Range checks name:min:max (comma-separated)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for checks",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.1,
        help="Minimum free disk space (GB)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        report = preflight_check(
            config=config,
            required=parse_list(args.required),
            ranges=parse_ranges(args.ranges),
            output_dir=args.output_dir,
            min_free_gb=args.min_free_gb,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    payload = {
        "inputs": {
            "config": args.config,
            "required": parse_list(args.required),
            "ranges": args.ranges,
            "output_dir": args.output_dir,
            "min_free_gb": args.min_free_gb,
        },
        "report": report,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Preflight report")
    print(f"  status: {report['status']}")
    for item in report["blockers"]:
        print(f"  blocker: {item}")
    for item in report["warnings"]:
        print(f"  warning: {item}")


if __name__ == "__main__":
    main()
