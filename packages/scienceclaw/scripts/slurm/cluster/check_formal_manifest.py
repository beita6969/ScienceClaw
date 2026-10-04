#!/usr/bin/env python3
"""Validate a requested cluster batch against the formal tool-ON manifest.

The manifest is an execution gate, rather than a second scorer.  Engineering
tags (for example ``H6``) remain usable, but tags beginning with ``SOTA`` are
required to name a manifest batch.  This prevents a typo in a formal command
from silently producing a run with the wrong split, episode count, or task
tool.  The check is intentionally label-free and only reads JSON/source files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    configured = os.environ.get("SCIENCECLAW_REPO")
    if configured:
        return Path(configured)
    # scripts/slurm/cluster/check_formal_manifest.py in the checkout.
    return Path(__file__).resolve().parents[3]


def _manifest_path(repo: Path) -> Path:
    configured = os.environ.get("SCIENCECLAW_FORMAL_MANIFEST")
    return Path(configured) if configured else repo / "configs" / "formal_toolon_manifest.json"


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load and structurally validate the immutable acceptance manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read formal manifest {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != 1 or not isinstance(raw.get("batches"), list):
        raise ValueError(f"formal manifest has unsupported schema: {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in raw["batches"]:
        if not isinstance(row, dict):
            raise ValueError("formal manifest batch is not an object")
        tag = row.get("tag")
        discipline = row.get("discipline")
        mode = row.get("mode")
        splits = row.get("splits")
        refs = row.get("required_tool_refs")
        if not isinstance(tag, str) or not tag or tag in out:
            raise ValueError(f"formal manifest has invalid or duplicate tag: {tag!r}")
        if not isinstance(discipline, str) or not discipline:
            raise ValueError(f"formal manifest {tag}: invalid discipline")
        if mode not in {"formal_tool_on", "blocked"} or not isinstance(splits, dict):
            raise ValueError(f"formal manifest {tag}: invalid mode/splits")
        if not isinstance(refs, list) or not all(isinstance(x, str) and x for x in refs):
            raise ValueError(f"formal manifest {tag}: invalid required_tool_refs")
        for split, spec in splits.items():
            if split not in {"src", "val", "id", "ood"} or not isinstance(spec, dict):
                raise ValueError(f"formal manifest {tag}: invalid split {split!r}")
            episodes = spec.get("episodes")
            if not isinstance(episodes, int) or episodes < 0:
                raise ValueError(f"formal manifest {tag}/{split}: invalid episodes")
            skip = spec.get("skip", 0)
            if not isinstance(skip, int) or skip < 0:
                raise ValueError(f"formal manifest {tag}/{split}: invalid skip")
            capacity = spec.get("capacity")
            if capacity is not None and (not isinstance(capacity, int) or capacity < 0):
                raise ValueError(f"formal manifest {tag}/{split}: invalid capacity")
        out[tag] = row
    return out


def _task_source(repo: Path, discipline: str) -> Path | None:
    # Keep this independent of importing adapters: the login node can validate
    # the tool contract before loading any dataset or consuming a GPU slot.
    match = re.fullmatch(r"FoR(\d+)", discipline)
    if not match:
        return None
    candidates = sorted((repo / "scienceclaw" / "bench" / "tasks").glob(f"for{int(match.group(1)):02d}_*.py"))
    return candidates[0] if candidates else None


def validate_request(repo: Path, manifest_path: Path, discipline_arg: str, tag: str,
                     split_arg: str, episodes: int, skip: int = 0) -> list[str]:
    """Return errors for a batch request; an empty list means it is allowed."""
    if episodes < 1:
        return [f"episode count must be positive: {episodes}"]
    if skip < 0:
        return [f"episode skip must be non-negative: {skip}"]
    disciplines = [x.strip() for x in discipline_arg.split(",") if x.strip()]
    splits = [x.strip() for x in split_arg.split(",") if x.strip()]
    if not disciplines or not splits:
        return ["discipline and split lists must not be empty"]
    # Non-formal engineering/debug tags intentionally remain unconstrained.
    # SOTA-prefixed tags are reserved for the manifest and fail closed.
    if not manifest_path.exists():
        if not tag.startswith("SOTA"):
            return []
        return [f"formal tag {tag!r} cannot be checked because manifest is missing: {manifest_path}"]
    manifest = load_manifest(manifest_path)
    if tag not in manifest:
        return [f"formal tag {tag!r} is absent from {manifest_path}"] if tag.startswith("SOTA") else []
    row = manifest[tag]
    expected_disc = row["discipline"]
    errors: list[str] = []
    if disciplines != [expected_disc]:
        errors.append(f"{tag}: manifest discipline is {expected_disc}, requested {discipline_arg}")
    if row["mode"] != "formal_tool_on":
        reason = row.get("blocked_reason", "manifest marks this batch blocked")
        errors.append(f"{tag}: {reason}")
    declared_splits = row["splits"]
    for split in splits:
        spec = declared_splits.get(split)
        if spec is None:
            errors.append(f"{tag}: split {split} is not declared in the formal manifest")
            continue
        expected = spec.get("episodes", 0)
        if expected <= 0 or spec.get("status") == "unavailable":
            errors.append(f"{tag}/{split}: manifest marks this split unavailable")
        elif episodes != expected:
            errors.append(f"{tag}/{split}: manifest requires n={expected}, requested n={episodes}")
        expected_skip = int(spec.get("skip", 0))
        if skip != expected_skip:
            errors.append(f"{tag}/{split}: manifest requires skip={expected_skip}, requested skip={skip}")
        capacity = spec.get("capacity")
        if isinstance(capacity, int) and skip + episodes > capacity:
            errors.append(
                f"{tag}/{split}: request uses episodes [{skip}, {skip + episodes}) "
                f"beyond declared pool capacity {capacity}"
            )
    source = _task_source(repo, expected_disc)
    if source is None:
        errors.append(f"{tag}: cannot locate adapter source for {expected_disc}")
    else:
        text = source.read_text(encoding="utf-8")
        for ref in row["required_tool_refs"]:
            # ToolSpec names are quoted literals; this avoids accepting a
            # prose mention that is not actually registered as a task tool.
            pattern = rf"ToolSpec\(\s*[\"']{re.escape(ref)}[\"']"
            if not re.search(pattern, text):
                errors.append(f"{tag}: required ToolSpec {ref!r} is absent from {source}")
    return errors


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate a cluster batch against formal_toolon_manifest.json")
    ap.add_argument("discipline", help="discipline code, or a comma-separated list")
    ap.add_argument("tag")
    ap.add_argument("split", help="split, or a comma-separated list")
    ap.add_argument("n", type=int, help="episodes per requested split")
    ap.add_argument("--skip", type=int, default=0, help="number of deterministic prefix episodes already consumed")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = _args()
    repo = args.repo or _repo_root()
    manifest = args.manifest or _manifest_path(repo)
    try:
        errors = validate_request(repo, manifest, args.discipline, args.tag, args.split, args.n, args.skip)
    except ValueError as exc:
        print(f"formal manifest check failed: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        return 2
    if args.tag.startswith("SOTA"):
        print(f"formal manifest OK: {args.tag} {args.discipline} {args.split} n={args.n}")
    else:
        print(f"engineering tag allowed without formal manifest: {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
