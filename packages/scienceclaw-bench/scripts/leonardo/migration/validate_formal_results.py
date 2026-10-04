#!/usr/bin/env python3
"""Audit completed formal tool-ON results without changing scorer semantics.

The launch-time manifest protects the requested batch specification.  This
post-run check protects the evidence that is actually reported: every
expected episode must have a successful result receipt and trajectory, satisfy
all hard result gates, and show the manifest's task tools as successfully
executed ToolSpec nodes whose output lineage reaches the final ``submit.y``
value.  A successful call that is ignored by a later CPU approximation does
not satisfy formal tool-ON evidence.  It is deliberately read-only and does
not call an adapter, scorer, broker, or model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:  # Script execution (the sibling module is on sys.path).
    from check_formal_manifest import load_manifest
except ImportError:  # pragma: no cover - package/importlib users
    try:
        from .check_formal_manifest import load_manifest
    except ImportError:
        # Tests and embedders may load this file directly by path, where no
        # package context or sibling module search path is installed.
        import importlib.util

        _gate_path = Path(__file__).with_name("check_formal_manifest.py")
        _gate_spec = importlib.util.spec_from_file_location("scienceclaw_formal_manifest_gate", _gate_path)
        if _gate_spec is None or _gate_spec.loader is None:
            raise
        _gate_module = importlib.util.module_from_spec(_gate_spec)
        _gate_spec.loader.exec_module(_gate_module)
        load_manifest = _gate_module.load_manifest


_REQUIRED_RESULT_FLAGS = ("completed", "reproducible", "within_budget", "hard_ok")
_SPLITS = ("src", "val", "id", "ood")


def _default_root() -> Path:
    configured = os.environ.get("SCIENCECLAW_RUN_ROOT")
    if configured:
        return Path(configured)
    return Path("/leonardo_scratch/large/userexternal/rqian000/sc-runs")


def _manifest_for(repo: Path | None, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    configured = os.environ.get("SCIENCECLAW_FORMAL_MANIFEST")
    if configured:
        return Path(configured)
    if repo is not None:
        return repo / "configs" / "formal_toolon_manifest.json"
    configured_repo = os.environ.get("SCIENCECLAW_REPO")
    if configured_repo:
        return Path(configured_repo) / "configs" / "formal_toolon_manifest.json"

    # The checkout has two supported layouts: the source checkout keeps this
    # file under scripts/leonardo/migration, while Leonardo copies it to the
    # sibling $F/sc-tools directory.  Prefer an existing manifest from either
    # layout instead of deriving /leonardo_scratch/fast/configs by accident.
    here = Path(__file__).resolve()
    candidates = [here.parents[1] / "scienceclaw", here.parents[3]]
    for base in candidates:
        candidate = base / "configs" / "formal_toolon_manifest.json"
        if candidate.is_file():
            return candidate
    return candidates[0] / "configs" / "formal_toolon_manifest.json"


def _episode_dirs(split_root: Path) -> list[Path]:
    """Return direct episode directories, keeping the audit scope deterministic."""
    if not split_root.is_dir():
        return []
    return sorted((entry for entry in split_root.iterdir() if entry.is_dir()), key=lambda p: p.name)


def _trajectory_evidence(path: Path, errors: list[str]) -> tuple[set[str], set[str]]:
    """Read successful task tools and the output lineage of the final submit.

    A successful tool call by itself is insufficient evidence for a formal
    tool-ON run: the tool's output must be one of the values that reaches the
    final ``submit.y`` input.  Each trajectory line is a graph snapshot, so
    records and node metadata are merged across lines.  This accepts cached
    replays and repeated snapshots while rejecting a tool that was called and
    then ignored by an unrelated CPU approximation.
    """
    refs: set[str] = set()
    successful_refs: dict[str, list[dict[str, Any]]] = {}
    node_meta: dict[str, dict[str, Any]] = {}
    submit_inputs: list[dict[str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path}: cannot read trajectory: {exc}")
        return refs, set()
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{lineno}: invalid trajectory JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path}:{lineno}: trajectory record is not an object")
            continue
        action = row.get("action")
        if not isinstance(action, dict):
            continue
        payload = action.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        node = payload.get("node")
        if not isinstance(node, dict):
            node = action.get("node")
        node_id = node.get("id") if isinstance(node, dict) else None
        if isinstance(node_id, str) and node_id:
            node_meta[node_id] = node
        feedback = row.get("feedback")
        records = feedback.get("records") if isinstance(feedback, dict) else None
        if not isinstance(records, dict):
            continue
        for record_id, record in records.items():
            if not isinstance(record_id, str) or not isinstance(record, dict):
                continue
            if record.get("status") != "ok":
                continue
            successful_refs.setdefault(record_id, []).append(record)
            kind = record.get("kind") or node_meta.get(record_id, {}).get("kind")
            if kind == "submit":
                inputs = record.get("input_refs")
                if isinstance(inputs, dict):
                    submit_inputs.append({k: v for k, v in inputs.items() if isinstance(k, str) and isinstance(v, str)})
            meta = node_meta.get(record_id, {})
            ref = meta.get("ref")
            if (kind == "tool" and isinstance(ref, str) and ref):
                refs.add(ref)

    # Build a reverse value-flow index.  A path may have more than one
    # producer across snapshots; retaining all producers is conservative and
    # handles cached/replayed records without trusting a line ordering quirk.
    producers: dict[str, set[str]] = {}
    for node_id, records in successful_refs.items():
        for record in records:
            outputs = record.get("output_refs")
            if not isinstance(outputs, dict):
                continue
            for value_path in outputs.values():
                if isinstance(value_path, str) and value_path:
                    producers.setdefault(value_path, set()).add(node_id)

    # The final successful submit snapshot is the last one in the file.  The
    # y input is the value that the scorer receives, so only its ancestors are
    # relevant to formal tool usage.
    y_input = None
    for inputs in reversed(submit_inputs):
        candidate = inputs.get("y")
        if isinstance(candidate, str) and candidate:
            y_input = candidate
            break
    if y_input is None:
        return refs, set()

    ancestor_paths: set[str] = set()
    seen_nodes: set[str] = set()
    stack = [y_input]
    while stack:
        value_path = stack.pop()
        if not isinstance(value_path, str) or not value_path or value_path in ancestor_paths:
            continue
        ancestor_paths.add(value_path)
        for node_id in producers.get(value_path, ()):
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            for record in successful_refs.get(node_id, ()):
                inputs = record.get("input_refs")
                if isinstance(inputs, dict):
                    stack.extend(v for v in inputs.values() if isinstance(v, str))
    return refs, ancestor_paths


def _successful_task_tools(path: Path, errors: list[str]) -> set[str]:
    """Read successful ToolSpec records from a trajectory receipt.

    Kept as a small compatibility helper for report scripts.  Formal
    validation additionally checks output lineage via :func:`_trajectory_evidence`.
    """
    refs, _ = _trajectory_evidence(path, errors)
    return refs


def _validate_episode(episode: Path, required_refs: list[str], errors: list[str]) -> None:
    result_path = episode / "result.json"
    trajectory_path = episode / "trajectory.jsonl"
    if not result_path.is_file():
        errors.append(f"{episode}: missing result.json")
    if not trajectory_path.is_file():
        errors.append(f"{episode}: missing trajectory.jsonl")

    result: dict[str, Any] | None = None
    if result_path.is_file():
        try:
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                errors.append(f"{result_path}: result is not an object")
            else:
                result = raw
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{result_path}: invalid result JSON: {exc}")

    if result is not None:
        if type(result.get("z")) is not int or result.get("z") != 1:
            errors.append(f"{result_path}: z must be integer 1 (got {result.get('z')!r})")
        for field in _REQUIRED_RESULT_FLAGS:
            if result.get(field) is not True:
                errors.append(f"{result_path}: {field} must be true (got {result.get(field)!r})")

    refs: set[str] = set()
    ancestor_paths: set[str] = set()
    if trajectory_path.is_file():
        refs, ancestor_paths = _trajectory_evidence(trajectory_path, errors)
    for ref in required_refs:
        if ref not in refs:
            errors.append(f"{episode}: required task-tool {ref!r} has no successful feedback.records entry")
            continue
        # Check the actual producer paths for this required ref.  A tool call
        # can be successful yet disconnected from the submitted prediction.
        connected = False
        try:
            lines = trajectory_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                row = json.loads(line)
                action = row.get("action") if isinstance(row, dict) else None
                payload = action.get("payload") if isinstance(action, dict) else None
                node = payload.get("node") if isinstance(payload, dict) else None
                if not isinstance(node, dict):
                    node = action.get("node") if isinstance(action, dict) else None
                if not isinstance(node, dict) or node.get("kind") != "tool" or node.get("ref") != ref:
                    continue
                node_id = node.get("id")
                feedback = row.get("feedback") if isinstance(row, dict) else None
                records = feedback.get("records") if isinstance(feedback, dict) else None
                record = records.get(node_id) if isinstance(records, dict) and isinstance(node_id, str) else None
                if not isinstance(record, dict) or record.get("status") != "ok":
                    continue
                outputs = record.get("output_refs")
                if isinstance(outputs, dict) and any(isinstance(p, str) and p in ancestor_paths for p in outputs.values()):
                    connected = True
                    break
        except (OSError, json.JSONDecodeError):
            # Parsing errors were already reported by _trajectory_evidence.
            connected = False
        if not connected:
            # A later replay/modify snapshot may carry the successful record
            # for a producer node while its action payload names only the
            # submit (or another node).  ``_trajectory_evidence`` already
            # merges those records by node id; use the same merged metadata
            # here instead of requiring the producer and its feedback to be
            # on one JSONL line.
            merged_nodes: dict[str, dict[str, Any]] = {}
            merged_records: dict[str, list[dict[str, Any]]] = {}
            for line in lines:
                row = json.loads(line)
                action = row.get("action") if isinstance(row, dict) else None
                payload = action.get("payload") if isinstance(action, dict) else None
                node = payload.get("node") if isinstance(payload, dict) else None
                if isinstance(node, dict) and isinstance(node.get("id"), str):
                    merged_nodes[node["id"]] = node
                feedback = row.get("feedback") if isinstance(row, dict) else None
                records = feedback.get("records") if isinstance(feedback, dict) else None
                if isinstance(records, dict):
                    for record_id, record in records.items():
                        if isinstance(record_id, str) and isinstance(record, dict) and record.get("status") == "ok":
                            merged_records.setdefault(record_id, []).append(record)
            for record_id, records in merged_records.items():
                meta = merged_nodes.get(record_id, {})
                if meta.get("kind") != "tool" or meta.get("ref") != ref:
                    continue
                for record in records:
                    outputs = record.get("output_refs")
                    if isinstance(outputs, dict) and any(
                        isinstance(p, str) and p in ancestor_paths for p in outputs.values()
                    ):
                        connected = True
                        break
                if connected:
                    break
        if not connected:
            errors.append(
                f"{episode}: required task-tool {ref!r} was called successfully, "
                "but its output is not connected to final submit.y"
            )


def validate_results(root: Path, tag: str, manifest_path: Path, *, repo: Path | None = None,
                     split_filter: str | None = None) -> list[str]:
    """Return post-run evidence errors for one manifest tag.

    Empty output means every declared split has exactly its manifest episode
    count and every episode satisfies the result and successful-tool checks.
    The function performs no writes.
    """
    errors: list[str] = []
    try:
        manifest = load_manifest(manifest_path)
    except ValueError as exc:
        return [str(exc)]
    row = manifest.get(tag)
    if row is None:
        return [f"formal tag {tag!r} is absent from {manifest_path}"]
    if row.get("mode") != "formal_tool_on":
        return [f"{tag}: manifest mode is {row.get('mode')!r}; formal post-run evidence is blocked"]
    if not root.is_dir():
        return [f"run root does not exist or is not a directory: {root}"]

    if split_filter is not None and split_filter not in _SPLITS:
        return [f"{tag}: unknown split filter {split_filter!r}"]
    declared = row.get("splits", {})
    refs = row.get("required_tool_refs", [])
    for split in _SPLITS:
        if split_filter is not None and split != split_filter:
            continue
        spec = declared.get(split)
        if spec is None:
            continue
        split_root = root / f"toolon_{tag}_{split}"
        episodes = _episode_dirs(split_root)
        expected = spec.get("episodes", 0)
        status = spec.get("status")
        if status == "unavailable" or expected == 0:
            if episodes:
                errors.append(
                    f"{tag}/{split}: manifest expects zero episodes ({status or 'unavailable'}), "
                    f"found {len(episodes)} under {split_root}"
                )
            continue
        if len(episodes) != expected:
            errors.append(f"{tag}/{split}: manifest requires {expected} episode(s), found {len(episodes)} under {split_root}")
        for episode in episodes:
            _validate_episode(episode, refs, errors)
    return errors


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only post-run validator for formal tool-ON batches")
    parser.add_argument("tag", help="manifest tag, for example SOTA30")
    parser.add_argument("root", nargs="?", type=Path, default=None, help="run root (defaults to SCIENCECLAW_RUN_ROOT)")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=None, help="checkout containing configs/ (only used for manifest lookup)")
    parser.add_argument("--split", choices=_SPLITS, default=None,
                        help="validate only this split (useful while id/ood batches run concurrently)")
    return parser.parse_args()


def main() -> int:
    args = _args()
    root = args.root or _default_root()
    manifest = _manifest_for(args.repo, args.manifest)
    errors = validate_results(root, args.tag, manifest, repo=args.repo, split_filter=args.split)
    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        return 2
    print(f"formal post-run evidence OK: {args.tag} root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
