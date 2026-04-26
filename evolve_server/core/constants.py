"""
Shared constants and enums for the evolve server.
"""

from __future__ import annotations

import re
from enum import IntEnum

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,}$")


class FailureType(IntEnum):
    """Five-way failure taxonomy for a bad turn."""

    SKILL_CONTENT_STALE = 1
    SKILL_MISSELECT = 2
    SKILL_GAP = 3
    TOOL_ERROR = 4
    MODEL_BASELINE = 5


FAILURE_LABELS: dict[int, str] = {
    FailureType.SKILL_CONTENT_STALE: "Skill content stale",
    FailureType.SKILL_MISSELECT: "Skill misselected",
    FailureType.SKILL_GAP: "Skill gap",
    FailureType.TOOL_ERROR: "Tool usage error",
    FailureType.MODEL_BASELINE: "Model baseline capability",
}


NO_SKILL_KEY = "__no_skill__"


class DecisionAction:
    """Allowed evolution-decision action identifiers.

    The first four are the original "what to do given current skill +
    sessions" outcomes.  The trailing three were added so the LLM can
    interact with already-pending candidates instead of stacking
    duplicates: when ``evolve_skill_from_sessions`` is called with the
    ``pending_candidates`` list non-empty, the LLM can decide that
    one of those pending entries already covers the new evidence
    (``SKIP_REDUNDANT``), needs to be replaced with a tighter
    proposal (``UPDATE_PENDING``), or is plain wrong and should be
    marked rejected outright (``REJECT_PENDING``).
    """

    CREATE = "create_skill"
    IMPROVE = "improve_skill"
    OPTIMIZE_DESC = "optimize_description"
    SKIP = "skip"
    # Pending-pool-aware outcomes — only valid when pending candidates
    # for this skill exist at decision time.
    SKIP_REDUNDANT = "skip_redundant"
    UPDATE_PENDING = "update_pending_candidate"
    REJECT_PENDING = "reject_pending_candidate"
