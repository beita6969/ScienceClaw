"""Evolution trigger and repair attribution (paper Eq. 9).

An evolution instance ends at a replay-verified success e+ = e_{t,k+} with Pass(e+) = 1. Its preceding
endpoint e- is the latest replay-verified failure before it (or none), and the repair delta is the
shortest reproduced repair: the atomic edits that turn the replayed failing graph into the replayed
passing graph.

Indexing convention (important). ``SolveResult.steps`` are 0-based and ``Evidence.step = s`` means "the
graph replayed after the action of step ``s`` was applied". In the paper's indexing, the graph that
action a_{t,k} is applied to is G_{t,k}; the replayed failing graph is therefore G_{t,s-+1} and the
replayed passing graph is G_{t,s++1}, and Eq. 9's window  k- <= k < k+  (paper indices) is the set of
step records  s- < s <= s+  (code indices). ``EvolutionInstance.k_minus`` / ``k_plus`` store the *step
indices* s-, s+ of the two evidence endpoints; ``delta_steps`` lists the steps whose actions form delta.
Without e- the window starts at step 0 (every action up to and including s+), and no Skill candidate is
formed downstream (Eq. 9, second case).

Actions that the executor rejected (``feedback["action_ok"] is False``: the graph did not change) and
unparseable turns are not part of delta; rejected edits are kept in ``EvolutionInstance.rejected`` so the
Skill patch can mention them as pitfalls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.trace import Evidence

__all__ = ["EvolutionInstance", "extract_instances", "to_action"]


@dataclass
class EvolutionInstance:
    """One evolution instance i (Eq. 9) extracted from a source-mode :class:`SolveResult`."""
    episode_id: str
    k_minus: int | None                  # step index s- of e- (None if no replay-verified failure precedes e+)
    k_plus: int                          # step index s+ of e+
    e_minus: Evidence | None
    e_plus: Evidence
    delta: list[Any]                     # core.actions.Action objects of the window (effective edits only)
    delta_steps: list[int]               # step index of each entry of ``delta``
    uses_in_delta: set[str]              # union of StepRecord.uses (nu) over the window
    # ---- additive fields (not part of DESIGN 8.5; all optional)
    program_version: str = ""
    feedback_minus: dict | None = None   # visible feedback of step s- (errors, visible constraints, dev score)
    feedback_plus: dict | None = None    # visible feedback of step s+
    rejected: list[dict] = field(default_factory=list)   # [{"step", "action", "error"}] rejected edits in the window
    run_dir: str = ""

    @property
    def window(self) -> tuple[int, int]:
        """Inclusive range of step indices considered for delta."""
        lo = self.k_minus + 1 if self.k_minus is not None else 0
        return lo, self.k_plus

    def summary(self) -> dict:
        """JSON-safe description for logs and receipts (no hidden labels)."""
        return {
            "episode_id": self.episode_id, "k_minus": self.k_minus, "k_plus": self.k_plus,
            "has_e_minus": self.e_minus is not None, "delta_steps": list(self.delta_steps),
            "delta_types": [getattr(a, "type", "?") for a in self.delta],
            "uses_in_delta": sorted(self.uses_in_delta), "n_rejected": len(self.rejected),
            "program_version": self.program_version,
            "e_minus_fp": getattr(self.e_minus, "graph_fp", None) if self.e_minus is not None else None,
            "e_plus_fp": getattr(self.e_plus, "graph_fp", None),
        }


def to_action(obj: Any) -> Any | None:
    """Coerce a StepRecord action (``Action`` object or its dict form) to a ``core.actions.Action``."""
    if obj is None:
        return None
    if hasattr(obj, "type") and hasattr(obj, "payload"):
        return obj
    if isinstance(obj, dict):
        from ..core.actions import Action  # lazy: core.actions is written by another module owner

        return Action.from_dict(obj)
    raise TypeError(f"cannot interpret step action of type {type(obj).__name__}")


def _feedback_dict(fb: Any) -> dict:
    if fb is None:
        return {}
    if isinstance(fb, dict):
        return fb
    if hasattr(fb, "to_dict"):
        return fb.to_dict()
    return dict(getattr(fb, "__dict__", {}))


def _steps_by_index(result: Any) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for i, rec in enumerate(getattr(result, "steps", None) or []):
        k = getattr(rec, "step", None)
        out[int(k) if k is not None else i] = rec
    return out


def extract_instances(result: Any, max_instances: int = 1) -> list[EvolutionInstance]:
    """Eq. 9: evolution instances of a source-mode solve.

    Each instance ends at a passing evidence e+ that starts a run of passing evidence (the first passing
    evidence is always the first candidate, so the default ``max_instances=1`` yields exactly the first
    replay-verified success). e- is the latest failing evidence before e+ (``None`` if there is none).
    """
    if max_instances <= 0:
        return []
    evidence = sorted(enumerate(getattr(result, "evidence", None) or []), key=lambda t: (t[1].step, t[0]))
    evs: list[Evidence] = [e for _, e in evidence]
    steps = _steps_by_index(result)
    instances: list[EvolutionInstance] = []
    for j, e_plus in enumerate(evs):
        if not e_plus.passed:
            continue
        if j > 0 and evs[j - 1].passed:
            continue  # inside a run of passing evidence: no new repair ends here
        e_minus = next((evs[i] for i in range(j - 1, -1, -1) if not evs[i].passed), None)
        k_plus = int(e_plus.step)
        k_minus = int(e_minus.step) if e_minus is not None else None
        lo = k_minus + 1 if k_minus is not None else 0
        delta: list[Any] = []
        delta_steps: list[int] = []
        uses: set[str] = set()
        rejected: list[dict] = []
        for k in sorted(s for s in steps if lo <= s <= k_plus):
            rec = steps[k]
            uses |= {str(u) for u in (getattr(rec, "uses", None) or [])}
            raw = getattr(rec, "action", None)
            if raw is None or getattr(rec, "parse_error", None):
                continue
            fb = _feedback_dict(getattr(rec, "feedback", None))
            if fb.get("action_ok") is False:
                rejected.append({"step": k, "action": raw if isinstance(raw, dict) else _action_dict(raw),
                                 "error": fb.get("action_error") or "; ".join(fb.get("validation_errors") or [])})
                continue
            delta.append(to_action(raw))
            delta_steps.append(k)
        fb_minus = _feedback_dict(getattr(steps.get(k_minus), "feedback", None)) if k_minus is not None else None
        fb_plus = _feedback_dict(getattr(steps.get(k_plus), "feedback", None))
        instances.append(EvolutionInstance(
            episode_id=str(getattr(result, "episode_id", "")), k_minus=k_minus, k_plus=k_plus,
            e_minus=e_minus, e_plus=e_plus, delta=delta, delta_steps=delta_steps, uses_in_delta=uses,
            program_version=str(getattr(result, "program_version", "")),
            feedback_minus=fb_minus if k_minus is not None and k_minus in steps else None,
            feedback_plus=fb_plus if k_plus in steps else None,
            rejected=rejected, run_dir=str(getattr(result, "run_dir", "") or ""),
        ))
        if len(instances) >= max_instances:
            break
    return instances


def _action_dict(action: Any) -> dict:
    if hasattr(action, "to_dict"):
        return action.to_dict()
    return {"type": getattr(action, "type", "?"), "payload": getattr(action, "payload", {})}
