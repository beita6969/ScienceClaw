#!/usr/bin/env python3
"""Manage simulation campaign lifecycle: init, status, list jobs.

This script initializes campaign tracking structures and provides
status summaries for multi-simulation campaigns.

Usage:
    python campaign_manager.py --action init --config-dir ./sweep --command "python sim.py --config {config}"
    python campaign_manager.py --action status --config-dir ./sweep

Output (JSON):
    {
        "campaign_id": "sweep_001",
        "status": "in_progress",
        "jobs": {"pending": 10, "running": 2, "completed": 5, "failed": 1},
        "progress": 0.33
    }
"""

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional


def generate_campaign_id() -> str:
    """Generate a unique campaign identifier."""
    return f"campaign_{uuid.uuid4().hex[:8]}"


def load_manifest(config_dir: str) -> Dict[str, Any]:
    """Load campaign manifest from config directory."""
    manifest_path = os.path.join(config_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise ValueError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        return json.load(f)


def load_campaign(config_dir: str) -> Dict[str, Any]:
    """Load campaign state from config directory."""
    campaign_path = os.path.join(config_dir, "campaign.json")
    if not os.path.exists(campaign_path):
        raise ValueError(f"Campaign not initialized. Run with --action init first.")
    with open(campaign_path, "r") as f:
        return json.load(f)


def save_campaign(config_dir: str, campaign: Dict[str, Any]) -> None:
    """Save campaign state to config directory."""
    campaign_path = os.path.join(config_dir, "campaign.json")
    with open(campaign_path, "w") as f:
        json.dump(campaign, f, indent=2)


def init_campaign(
    config_dir: str, command_template: str, output_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """Initialize a new campaign from sweep configurations.

    Args:
        config_dir: Directory containing sweep configs and manifest
        command_template: Command to run, with {config} placeholder
        output_pattern: Pattern for output files (optional)

    Returns:
        Campaign state dictionary
    """
    manifest = load_manifest(config_dir)

    campaign_id = generate_campaign_id()

    jobs = []
    for i, config_file in enumerate(manifest["configs"]):
        config_path = os.path.join(config_dir, config_file)
        job = {
            "job_id": f"job_{i:04d}",
            "config_file": config_file,
            "config_path": config_path,
            "command": command_template.replace("{config}", config_path),
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "exit_code": None,
            "output_file": None,
        }
        jobs.append(job)

    campaign = {
        "campaign_id": campaign_id,
        "config_dir": os.path.abspath(config_dir),
        "command_template": command_template,
        "output_pattern": output_pattern,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jobs": jobs,
        "manifest": manifest,
    }

    save_campaign(config_dir, campaign)
    return campaign


def get_campaign_status(config_dir: str) -> Dict[str, Any]:
    """Get current campaign status summary.

    Returns:
        Dictionary with campaign_id, status, jobs counts, progress
    """
    campaign = load_campaign(config_dir)

    # Count job statuses
    status_counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    for job in campaign["jobs"]:
        status = job["status"]
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts[status] = 1

    total = len(campaign["jobs"])
    completed = status_counts.get("completed", 0)
    failed = status_counts.get("failed", 0)
    progress = (completed + failed) / total if total > 0 else 0.0

    # Determine overall status
    if status_counts["pending"] == total:
        overall_status = "not_started"
    elif completed + failed == total:
        overall_status = "completed" if failed == 0 else "completed_with_failures"
    elif status_counts["running"] > 0:
        overall_status = "in_progress"
    else:
        overall_status = "in_progress"

    return {
        "campaign_id": campaign["campaign_id"],
        "status": overall_status,
        "jobs": status_counts,
        "progress": round(progress, 4),
        "total_jobs": total,
        "created_at": campaign.get("created_at", "unknown"),
    }


def list_jobs(config_dir: str, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """List jobs in campaign, optionally filtered by status.

    Args:
        config_dir: Campaign directory
        status_filter: Filter by status (pending, running, completed, failed)

    Returns:
        List of job dictionaries
    """
    campaign = load_campaign(config_dir)

    jobs = campaign["jobs"]
    if status_filter:
        jobs = [j for j in jobs if j["status"] == status_filter]

    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage simulation campaign lifecycle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["init", "status", "list"],
        help="Action to perform",
    )
    parser.add_argument(
        "--config-dir",
        required=True,
        help="Directory containing sweep configurations",
    )
    parser.add_argument(
        "--command",
        default="python sim.py --config {config}",
        help="Command template with {config} placeholder (for init)",
    )
    parser.add_argument(
        "--output-pattern",
        default=None,
        help="Pattern for output files, e.g., 'result_{job_id}.json'",
    )
    parser.add_argument(
        "--status-filter",
        choices=["pending", "running", "completed", "failed"],
        default=None,
        help="Filter jobs by status (for list action)",
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
        if args.action == "init":
            result = init_campaign(
                config_dir=args.config_dir,
                command_template=args.command,
                output_pattern=args.output_pattern,
            )
            if args.json:
                output = {
                    "campaign_id": result["campaign_id"],
                    "total_jobs": len(result["jobs"]),
                    "config_dir": result["config_dir"],
                    "command_template": result["command_template"],
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"Campaign initialized: {result['campaign_id']}")
                print(f"Total jobs: {len(result['jobs'])}")
                print(f"Config dir: {result['config_dir']}")

        elif args.action == "status":
            result = get_campaign_status(args.config_dir)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"Campaign: {result['campaign_id']}")
                print(f"Status: {result['status']}")
                print(f"Progress: {result['progress']*100:.1f}%")
                print(f"Jobs: pending={result['jobs']['pending']}, "
                      f"running={result['jobs']['running']}, "
                      f"completed={result['jobs']['completed']}, "
                      f"failed={result['jobs']['failed']}")

        elif args.action == "list":
            jobs = list_jobs(args.config_dir, args.status_filter)
            if args.json:
                print(json.dumps({"jobs": jobs}, indent=2))
            else:
                for job in jobs:
                    print(f"{job['job_id']}: {job['status']} - {job['config_file']}")

    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
