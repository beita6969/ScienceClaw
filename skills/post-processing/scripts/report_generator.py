#!/usr/bin/env python3
"""
Report Generator - Generate summary reports from simulation results.

Aggregates information from simulation output files and generates
structured reports including statistics, convergence info, and validation.

Usage:
    python report_generator.py --input results/ --output report.json --json
    python report_generator.py --input results/ --sections "summary,statistics" --json
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file and return contents."""
    with open(filepath, "r") as f:
        return json.load(f)


def find_data_files(directory: str) -> Dict[str, List[str]]:
    """Find and categorize data files in directory."""
    files = {
        "field_files": [],
        "history_files": [],
        "config_files": [],
        "log_files": [],
        "other_files": []
    }

    if not os.path.isdir(directory):
        return files

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue

        lower = filename.lower()

        if "field" in lower or "snapshot" in lower or "output" in lower:
            files["field_files"].append(filename)
        elif "history" in lower or "log" in lower and lower.endswith(".json"):
            files["history_files"].append(filename)
        elif "config" in lower or "params" in lower or "settings" in lower:
            files["config_files"].append(filename)
        elif lower.endswith(".log") or lower.endswith(".txt"):
            files["log_files"].append(filename)
        elif lower.endswith(".json") or lower.endswith(".csv"):
            files["other_files"].append(filename)

    # Sort files
    for key in files:
        files[key] = sorted(files[key])

    return files


def extract_simulation_info(directory: str, files: Dict[str, List[str]]) -> Dict[str, Any]:
    """Extract simulation metadata from config files."""
    info = {
        "config_found": False,
        "parameters": {},
        "grid": {},
        "time": {}
    }

    for config_file in files.get("config_files", []):
        try:
            config = load_json_file(os.path.join(directory, config_file))
            info["config_found"] = True
            info["config_file"] = config_file

            # Extract common parameters
            for key in ["dt", "dx", "dy", "dz", "nx", "ny", "nz"]:
                if key in config:
                    if key.startswith("d"):
                        info["grid"][key] = config[key]
                    elif key.startswith("n"):
                        info["grid"][key] = config[key]

            for key in ["dt", "t_max", "t_end", "n_steps", "max_iterations"]:
                if key in config:
                    info["time"][key] = config[key]

            # Store all parameters
            info["parameters"] = config
            break  # Use first config file found

        except Exception:
            continue

    return info


def analyze_field_files(directory: str, files: List[str]) -> Dict[str, Any]:
    """Analyze field output files."""
    analysis = {
        "count": len(files),
        "first_file": files[0] if files else None,
        "last_file": files[-1] if files else None,
        "timesteps": []
    }

    if not files:
        return analysis

    # Try to extract timesteps from filenames
    for filename in files:
        # Look for patterns like _0001.json, _100.json, etc.
        base = os.path.splitext(filename)[0]
        parts = base.split("_")
        for part in reversed(parts):
            if part.isdigit():
                analysis["timesteps"].append(int(part))
                break

    if analysis["timesteps"]:
        analysis["timesteps"].sort()
        analysis["first_timestep"] = analysis["timesteps"][0]
        analysis["last_timestep"] = analysis["timesteps"][-1]

    # Analyze last field file for statistics
    if files:
        try:
            last_file = os.path.join(directory, files[-1])
            data = load_json_file(last_file)

            analysis["final_state"] = {}

            # Find numeric fields
            for key, value in data.items():
                if isinstance(value, list):
                    flat = flatten_list(value)
                    if flat and all(isinstance(x, (int, float)) for x in flat):
                        analysis["final_state"][key] = {
                            "min": min(flat),
                            "max": max(flat),
                            "mean": sum(flat) / len(flat),
                            "count": len(flat)
                        }
        except Exception:
            pass

    return analysis


def analyze_history_files(directory: str, files: List[str]) -> Dict[str, Any]:
    """Analyze history/convergence files."""
    analysis = {
        "count": len(files),
        "quantities": [],
        "convergence": {}
    }

    if not files:
        return analysis

    # Analyze first history file
    try:
        history_file = os.path.join(directory, files[0])
        data = load_json_file(history_file)

        # Handle nested history
        if "history" in data:
            data = data["history"]

        # Find time series quantities
        for key, value in data.items():
            if isinstance(value, list) and value:
                analysis["quantities"].append(key)

                # Analyze convergence for common quantities
                if any(q in key.lower() for q in ["residual", "error", "energy", "mass"]):
                    values = [v for v in value if isinstance(v, (int, float))]
                    if values:
                        analysis["convergence"][key] = {
                            "initial": values[0],
                            "final": values[-1],
                            "min": min(values),
                            "max": max(values),
                            "iterations": len(values)
                        }

                        # Check if converged
                        if len(values) > 10:
                            recent = values[-10:]
                            mean = sum(recent) / len(recent)
                            var = sum((v - mean) ** 2 for v in recent) / len(recent)
                            rel_var = math.sqrt(var) / abs(mean) if mean != 0 else 0
                            analysis["convergence"][key]["converged"] = rel_var < 0.01

    except Exception:
        pass

    return analysis


def flatten_list(data: Any) -> List[float]:
    """Flatten nested list."""
    if not isinstance(data, list):
        if isinstance(data, (int, float)):
            return [float(data)]
        return []

    result = []
    for item in data:
        result.extend(flatten_list(item))
    return result


def generate_summary_section(
    directory: str,
    files: Dict[str, List[str]],
    sim_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate summary section of report."""
    summary = {
        "directory": directory,
        "generated_at": datetime.now().isoformat(),
        "files_found": {
            "field_files": len(files.get("field_files", [])),
            "history_files": len(files.get("history_files", [])),
            "config_files": len(files.get("config_files", [])),
            "total": sum(len(v) for v in files.values())
        },
        "simulation_configured": sim_info.get("config_found", False)
    }

    if sim_info.get("grid"):
        summary["grid"] = sim_info["grid"]

    if sim_info.get("time"):
        summary["time_settings"] = sim_info["time"]

    return summary


def generate_statistics_section(
    field_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate statistics section of report."""
    stats = {
        "field_files_analyzed": field_analysis.get("count", 0),
        "timesteps_found": len(field_analysis.get("timesteps", []))
    }

    if field_analysis.get("final_state"):
        stats["final_state_fields"] = list(field_analysis["final_state"].keys())
        stats["final_state_statistics"] = field_analysis["final_state"]

    return stats


def generate_convergence_section(
    history_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate convergence section of report."""
    convergence = {
        "quantities_tracked": history_analysis.get("quantities", []),
        "analysis": history_analysis.get("convergence", {})
    }

    # Overall assessment
    all_converged = all(
        v.get("converged", False)
        for v in convergence["analysis"].values()
    )

    if convergence["analysis"]:
        convergence["overall_assessment"] = "converged" if all_converged else "not_converged"
    else:
        convergence["overall_assessment"] = "no_data"

    return convergence


def generate_validation_section(
    field_analysis: Dict[str, Any],
    history_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate validation section of report."""
    validation = {
        "checks": [],
        "warnings": [],
        "passed": True
    }

    # Check for NaN/Inf in final state
    if field_analysis.get("final_state"):
        for field, stats in field_analysis["final_state"].items():
            if stats.get("min") is not None:
                if math.isnan(stats["min"]) or math.isnan(stats["max"]):
                    validation["warnings"].append(f"NaN values in field: {field}")
                    validation["passed"] = False
                elif math.isinf(stats["min"]) or math.isinf(stats["max"]):
                    validation["warnings"].append(f"Inf values in field: {field}")
                    validation["passed"] = False
                else:
                    validation["checks"].append({
                        "field": field,
                        "check": "finite_values",
                        "passed": True
                    })

    # Check convergence
    for key, conv in history_analysis.get("convergence", {}).items():
        if "residual" in key.lower() or "error" in key.lower():
            if conv.get("final", 0) > conv.get("initial", 0):
                validation["warnings"].append(f"Residual increased: {key}")
            if not conv.get("converged", True):
                validation["warnings"].append(f"Not converged: {key}")

    return validation


def generate_report(
    directory: str,
    sections: List[str]
) -> Dict[str, Any]:
    """Generate complete report."""
    # Find files
    files = find_data_files(directory)

    # Extract simulation info
    sim_info = extract_simulation_info(directory, files)

    # Analyze data
    field_analysis = analyze_field_files(directory, files.get("field_files", []))
    history_analysis = analyze_history_files(directory, files.get("history_files", []))

    # Build report
    report = {
        "report_version": "1.0.0",
        "generator": "post-processing/report_generator.py"
    }

    if "summary" in sections or "all" in sections:
        report["summary"] = generate_summary_section(directory, files, sim_info)

    if "statistics" in sections or "all" in sections:
        report["statistics"] = generate_statistics_section(field_analysis)

    if "convergence" in sections or "all" in sections:
        report["convergence"] = generate_convergence_section(history_analysis)

    if "validation" in sections or "all" in sections:
        report["validation"] = generate_validation_section(field_analysis, history_analysis)

    if "files" in sections or "all" in sections:
        report["files"] = files

    if "parameters" in sections or "all" in sections:
        if sim_info.get("parameters"):
            report["parameters"] = sim_info["parameters"]

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary reports from simulation results"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing simulation results"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--sections", "-s",
        default="all",
        help="Report sections to include (comma-separated: "
             "summary,statistics,convergence,validation,files,parameters,all)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (implied if --output is .json)"
    )

    args = parser.parse_args()

    try:
        # Validate input
        if not os.path.exists(args.input):
            print(f"Error: Directory not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        if not os.path.isdir(args.input):
            print(f"Error: Not a directory: {args.input}", file=sys.stderr)
            sys.exit(1)

        # Parse sections
        sections = [s.strip().lower() for s in args.sections.split(",")]

        # Generate report
        report = generate_report(args.input, sections)

        # Determine output format
        output_json = args.json
        if args.output and args.output.endswith(".json"):
            output_json = True

        # Output
        if args.output:
            with open(args.output, "w") as f:
                if output_json:
                    json.dump(report, f, indent=2)
                else:
                    f.write(format_report_text(report))
            print(f"Report written to: {args.output}")
        else:
            if output_json:
                print(json.dumps(report, indent=2))
            else:
                print(format_report_text(report))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def format_report_text(report: Dict[str, Any]) -> str:
    """Format report as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("SIMULATION ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    if "summary" in report:
        s = report["summary"]
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Directory: {s.get('directory', 'N/A')}")
        lines.append(f"Generated: {s.get('generated_at', 'N/A')}")
        lines.append("")

        files = s.get("files_found", {})
        lines.append(f"Files found: {files.get('total', 0)}")
        lines.append(f"  - Field files: {files.get('field_files', 0)}")
        lines.append(f"  - History files: {files.get('history_files', 0)}")
        lines.append(f"  - Config files: {files.get('config_files', 0)}")
        lines.append("")

        if s.get("grid"):
            lines.append("Grid:")
            for k, v in s["grid"].items():
                lines.append(f"  {k}: {v}")
            lines.append("")

    if "statistics" in report:
        s = report["statistics"]
        lines.append("STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Field files analyzed: {s.get('field_files_analyzed', 0)}")
        lines.append(f"Timesteps found: {s.get('timesteps_found', 0)}")
        lines.append("")

        if s.get("final_state_statistics"):
            lines.append("Final state statistics:")
            for field, stats in s["final_state_statistics"].items():
                lines.append(f"  {field}:")
                lines.append(f"    Range: [{stats['min']:.6g}, {stats['max']:.6g}]")
                lines.append(f"    Mean: {stats['mean']:.6g}")
            lines.append("")

    if "convergence" in report:
        c = report["convergence"]
        lines.append("CONVERGENCE")
        lines.append("-" * 40)
        lines.append(f"Quantities tracked: {len(c.get('quantities_tracked', []))}")
        lines.append(f"Overall: {c.get('overall_assessment', 'N/A')}")
        lines.append("")

        for q, analysis in c.get("analysis", {}).items():
            status = "CONVERGED" if analysis.get("converged") else "NOT CONVERGED"
            lines.append(f"  {q}: {status}")
            lines.append(f"    Initial: {analysis.get('initial', 'N/A')}")
            lines.append(f"    Final: {analysis.get('final', 'N/A')}")
        lines.append("")

    if "validation" in report:
        v = report["validation"]
        lines.append("VALIDATION")
        lines.append("-" * 40)
        status = "PASSED" if v.get("passed") else "FAILED"
        lines.append(f"Status: {status}")

        if v.get("warnings"):
            lines.append("Warnings:")
            for w in v["warnings"]:
                lines.append(f"  - {w}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
