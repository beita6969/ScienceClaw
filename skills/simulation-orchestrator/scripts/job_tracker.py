#!/usr/bin/env python3
"""Track individual job status within a simulation campaign.

This script updates job status based on output files or markers,
and provides detailed job information.

Usage:
    python job_tracker.py --campaign-dir ./sweep --update
    python job_tracker.py --campaign-dir ./sweep --job-id job_0001

Output (JSON):
    {
        "job_id": "job_0001",
        "status": "completed",
        "start_time": "2024-12-24 10:00:00",
        "end_time": "2024-12-24 10:05:32",
        "exit_code": 0
    }
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional


def load_campaign(config_dir: str) -> Dict[str, Any]:
    """Load campaign state from config directory."""
    campaign_path = os.path.join(config_dir, "campaign.json")
    if not os.path.exists(campaign_path):
        raise ValueError(f"Campaign not found: {campaign_path}")
    with open(campaign_path, "r") as f:
        return json.load(f)


def save_campaign(config_dir: str, campaign: Dict[str, Any]) -> None:
    """Save campaign state to config directory."""
    campaign_path = os.path.join(config_dir, "campaign.json")
    with open(campaign_path, "w") as f:
        json.dump(campaign, f, indent=2)


def detect_job_status(
    job: Dict[str, Any], config_dir: str, result_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """Detect job status based on output files.

    Status detection logic:
    1. Check for result file -> completed
    2. Check for error file -> failed
    3. Check for running marker -> running
    4. Otherwise -> pending

    Args:
        job: Job dictionary
        config_dir: Campaign directory
        result_pattern: Pattern for result files

    Returns:
        Updated job dictionary
    """
    job_id = job["job_id"]

    # Look for common result file patterns
    result_patterns = [
        f"result_{job_id}.json",
        f"{job_id}_result.json",
        job["config_file"].replace("config_", "result_").replace(".json", ".json"),
    ]

    if result_pattern:
        result_patterns.insert(0, result_pattern.format(job_id=job_id))

    # Check for result files
    for pattern in result_patterns:
        result_path = os.path.join(config_dir, pattern)
        if os.path.exists(result_path):
            job["status"] = "completed"
            job["output_file"] = pattern
            if job["end_time"] is None:
                job["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if job["exit_code"] is None:
                job["exit_code"] = 0
            return job

    # Check for error markers
    error_patterns = [
        f"error_{job_id}.log",
        f"{job_id}.error",
        f"{job_id}_failed",
    ]

    for pattern in error_patterns:
        error_path = os.path.join(config_dir, pattern)
        if os.path.exists(error_path):
            job["status"] = "failed"
            if job["end_time"] is None:
                job["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if job["exit_code"] is None:
                job["exit_code"] = 1
            return job

    # Check for running marker
    running_patterns = [
        f"{job_id}.running",
        f"{job_id}.lock",
    ]

    for pattern in running_patterns:
        running_path = os.path.join(config_dir, pattern)
        if os.path.exists(running_path):
            job["status"] = "running"
            if job["start_time"] is None:
                job["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return job

    # No status change detected
    return job


def update_all_jobs(
    config_dir: str, result_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """Update status of all jobs in campaign.

    Args:
        config_dir: Campaign directory
        result_pattern: Pattern for result files

    Returns:
        Summary of status changes
    """
    campaign = load_campaign(config_dir)

    changes = {"updated": 0, "unchanged": 0}
    status_counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}

    for i, job in enumerate(campaign["jobs"]):
        old_status = job["status"]
        updated_job = detect_job_status(job, config_dir, result_pattern)
        campaign["jobs"][i] = updated_job

        new_status = updated_job["status"]
        if old_status != new_status:
            changes["updated"] += 1
        else:
            changes["unchanged"] += 1

        if new_status in status_counts:
            status_counts[new_status] += 1

    save_campaign(config_dir, campaign)

    return {
        "campaign_id": campaign["campaign_id"],
        "changes": changes,
        "status_counts": status_counts,
        "total_jobs": len(campaign["jobs"]),
    }


def get_job_info(config_dir: str, job_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific job.

    Args:
        config_dir: Campaign directory
        job_id: Job identifier

    Returns:
        Job dictionary with all details
    """
    campaign = load_campaign(config_dir)

    for job in campaign["jobs"]:
        if job["job_id"] == job_id:
            return job

    raise ValueError(f"Job not found: {job_id}")


def mark_job_status(
    config_dir: str,
    job_id: str,
    status: str,
    exit_code: Optional[int] = None,
) -> Dict[str, Any]:
    """Manually mark a job's status.

    Args:
        config_dir: Campaign directory
        job_id: Job identifier
        status: New status (pending, running, completed, failed)
        exit_code: Optional exit code

    Returns:
        Updated job dictionary
    """
    if status not in ["pending", "running", "completed", "failed"]:
        raise ValueError(f"Invalid status: {status}")

    campaign = load_campaign(config_dir)

    for i, job in enumerate(campaign["jobs"]):
        if job["job_id"] == job_id:
            job["status"] = status
            if status == "running" and job["start_time"] is None:
                job["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if status in ["completed", "failed"] and job["end_time"] is None:
                job["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if exit_code is not None:
                job["exit_code"] = exit_code
            campaign["jobs"][i] = job
            save_campaign(config_dir, campaign)
            return job

    raise ValueError(f"Job not found: {job_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track job status within a simulation campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--campaign-dir",
        required=True,
        help="Directory containing campaign files",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update status of all jobs based on output files",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Get info for specific job",
    )
    parser.add_argument(
        "--mark-status",
        choices=["pending", "running", "completed", "failed"],
        default=None,
        help="Manually mark job status (requires --job-id)",
    )
    parser.add_argument(
        "--exit-code",
        type=int,
        default=None,
        help="Exit code for manual status update",
    )
    parser.add_argument(
        "--result-pattern",
        default=None,
        help="Pattern for result files, e.g., 'output_{job_id}.json'",
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
        if args.update:
            result = update_all_jobs(args.campaign_dir, args.result_pattern)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"Campaign: {result['campaign_id']}")
                print(f"Updated: {result['changes']['updated']} jobs")
                print(f"Status: pending={result['status_counts']['pending']}, "
                      f"running={result['status_counts']['running']}, "
                      f"completed={result['status_counts']['completed']}, "
                      f"failed={result['status_counts']['failed']}")

        elif args.mark_status and args.job_id:
            result = mark_job_status(
                args.campaign_dir, args.job_id, args.mark_status, args.exit_code
            )
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"Job {args.job_id} marked as {args.mark_status}")

        elif args.job_id:
            result = get_job_info(args.campaign_dir, args.job_id)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"Job ID: {result['job_id']}")
                print(f"Status: {result['status']}")
                print(f"Config: {result['config_file']}")
                print(f"Command: {result['command']}")
                if result["start_time"]:
                    print(f"Start: {result['start_time']}")
                if result["end_time"]:
                    print(f"End: {result['end_time']}")
                if result["exit_code"] is not None:
                    print(f"Exit code: {result['exit_code']}")

        else:
            print("Specify --update, --job-id, or --mark-status", file=sys.stderr)
            sys.exit(1)

    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
