import collections
import glob
import json
import os
import re
import sys

R = "/leonardo_scratch/large/userexternal/rqian000/sc-runs"
tags = sys.argv[1:] or ["H1"]

# ``result.json.uses`` records retrieved skills/operators, but not the task's built-in ToolSpec nodes.
# Audit the trajectory as well so a call such as FoR30's ``predict_pretrained`` is not mistaken for a
# direct smoke. Keep direct code imports separate: they exercise a library path, but are not a task-tool call.
BASE_REFS = {
    "load_train", "load_dev_inputs", "load_eval_inputs", "load_history", "load_covariates", "load_context",
    "load_data", "score_dev", "score_pretrained_dev", "check_trees", "read_conllu", "submit",
}
DIRECT_MARKERS = (
    "udparse_pretrained", "pre.parse_gold_tokens", "phenoseg_m2f", "predict_pretrained",
    "separate_htdemucs", "audiosep_pretrained", "fit_sevennet_mlip", "matphonon_mlip", "chronos_2", "chronos",
)


def trace_usage_details(path):
    """Return successfully executed non-baseline tool refs and direct-code markers.

    A trajectory can contain a proposed node whose execution failed.  Counting the
    action alone therefore overstates capability use; the corresponding executor
    record is authoritative for whether that node actually ran successfully.
    """
    tool_refs, direct_refs = set(), set()
    try:
        with open(path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, OSError):
                    continue
                action = row.get("action") or {}
                payload = action.get("payload") or {}
                node = payload.get("node") or action.get("node") or {}
                node_id = node.get("id")
                feedback = row.get("feedback") or {}
                records = feedback.get("records") or {}
                record = records.get(node_id) if isinstance(records, dict) and node_id else None
                # ``records`` is produced by the executor for every node in the
                # resulting graph.  Missing records are intentionally treated as
                # not executed, rather than as successful proposals.
                if not isinstance(record, dict) or record.get("status") != "ok":
                    continue
                if node.get("kind") == "tool":
                    ref = node.get("ref")
                    baseline = ref in BASE_REFS or (isinstance(ref, str) and ref.startswith(("load_", "score_", "check_", "read_")))
                    if ref and not baseline:
                        tool_refs.add(ref)
                if node.get("kind") == "code":
                    code = node.get("code") or ""
                    direct_refs.update(marker for marker in DIRECT_MARKERS if re.search(re.escape(marker), code))
    except OSError:
        pass
    return tool_refs, direct_refs


def trace_usage(path):
    """Compatibility wrapper returning the historical boolean pair."""
    tool_refs, direct_refs = trace_usage_details(path)
    return bool(tool_refs), bool(direct_refs)


def summarize(root=R, selected_tags=None):
    """Print and return pass/usage counts for completed episodes under ``root``."""
    selected_tags = list(selected_tags or ["H1"])
    # Each split stores pass, finished, running, built-in tool calls, direct library calls, and retrieved operators.
    # ``used`` below remains an output alias for the historical retrieved-operator count.
    res = collections.defaultdict(lambda: {"id": [0, 0, 0, 0, 0, 0], "ood": [0, 0, 0, 0, 0, 0]})
    for tag in selected_tags:
        for sp in ("id", "ood"):
            for d in sorted(glob.glob(f"{root}/toolon_{tag}_{sp}/*/")):
                code = os.path.basename(d.rstrip("/")).split("-")[0]
                if os.path.exists(d + "result.json"):
                    with open(d + "result.json") as fh:
                        r = json.load(fh)
                    res[code][sp][1] += 1
                    res[code][sp][0] += int(r["z"] == 1)
                    tool, direct = trace_usage(d + "trajectory.jsonl")
                    res[code][sp][3] += int(tool)
                    res[code][sp][4] += int(direct)
                    res[code][sp][5] += int(bool(r.get("uses")) or bool(r.get("retrieved", {}).get("operators")))
                else:
                    res[code][sp][2] += 1

    tp = {"id": [0, 0], "ood": [0, 0]}
    for c in sorted(res):
        a = res[c]
        print(
            f"{c}: id {a['id'][0]}/{a['id'][1]} tool {a['id'][3]}/{a['id'][1]} direct {a['id'][4]}/{a['id'][1]} "
            f"operator {a['id'][5]}/{a['id'][1]} used {a['id'][5]}/{a['id'][1]} (+{a['id'][2]} running)  "
            f"ood {a['ood'][0]}/{a['ood'][1]} tool {a['ood'][3]}/{a['ood'][1]} direct {a['ood'][4]}/{a['ood'][1]} "
            f"operator {a['ood'][5]}/{a['ood'][1]} used {a['ood'][5]}/{a['ood'][1]} (+{a['ood'][2]} running)"
        )
        for sp in ("id", "ood"):
            tp[sp][0] += a[sp][0]
            tp[sp][1] += a[sp][1]
    print("TOTAL id %d/%d  ood %d/%d" % (tp["id"][0], tp["id"][1], tp["ood"][0], tp["ood"][1]))
    return res


if __name__ == "__main__":
    summarize(selected_tags=sys.argv[1:] or ["H1"])
