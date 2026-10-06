"""Program self-evolution (paper Section 5.2, Eq. 9-13 and Eq. 2-3).

Public API (DESIGN section 8.5):

* :func:`extract_instances` / :class:`EvolutionInstance`   - Eq. 9 repair attribution
* :func:`split_edits`                                      - Eq. 10 control edits / executable components
* :func:`make_skill_candidates`                            - Eq. 11 Patch_Theta0 Skill candidates
* :func:`make_operator_candidate`, :func:`boundary_replay` - Eq. 12 Operator candidates + BReplay
* :func:`build_bundle`                                     - the linked bundle B_i = (dS_i, dO_i)
* :func:`source_replay_check`                              - Eq. 13 R_src = Pass and Use
* :class:`ValReport`, :class:`ValidationGate`              - Eq. 2 feasibility + Eq. 3 strict improvement
* :class:`Evolver`                                         - the stream loop with snapshots A_0..A_R
* :class:`LiveEvolution`                                   - the same loop driven by live canvas sessions
"""
from .attribution import EvolutionInstance, extract_instances
from .bundle import Bundle, build_bundle
from .evolver import Evolver
from .live import LiveEvolution
from .operator_abstraction import boundary_replay, make_operator_candidate
from .skill_patch import make_skill_candidates
from .split import split_edits
from .validation import ValidationGate, ValReport, source_replay_check

__all__ = ["EvolutionInstance", "extract_instances", "split_edits", "make_skill_candidates",
           "make_operator_candidate", "boundary_replay", "Bundle", "build_bundle",
           "source_replay_check", "ValReport", "ValidationGate", "Evolver", "LiveEvolution"]
