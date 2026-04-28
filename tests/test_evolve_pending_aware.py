# -*- coding: utf-8 -*-
"""Unit tests for the pending-pool-aware evolve actions.

Covers the new actions ``DecisionAction.SKIP_REDUNDANT`` /
``UPDATE_PENDING_CANDIDATE`` / ``REJECT_PENDING_CANDIDATE`` introduced
so the LLM can interact with already-queued validation candidates
instead of stacking duplicates.

Three layers of behaviour:

1. ``_build_pending_candidates_block`` renders the right Markdown
   shape (caps, ordering, content excerpt for improve_skill only).
2. ``_parse_evolve_result`` accepts the new action shapes, demotes
   malformed responses to ``skip``, and preserves ``target_pending_job_id``.
3. End-to-end: a sample LLM response with each new action parses to
   the expected dict shape.

The integration with ``ValidationStore.save_decision`` /
``ValidationStore.save_job`` is exercised separately in
``test_dashboard.py`` / live runs — those side effects don't fit a
pure-parser unit test.
"""
from __future__ import annotations

import json

import pytest

from evolve_server.core.constants import DecisionAction
from evolve_server.pipeline.execution import (
    _build_pending_candidates_block,
    _parse_evolve_result,
    _preserve_existing_body_for_optimize_desc,
)


# --- _build_pending_candidates_block -------------------------------------- #


def _job(job_id: str, action: str = "optimize_description", **extra) -> dict:
    """Build a validation_jobs/<id>.json-shaped dict for tests."""
    base = {
        "job_id": job_id,
        "candidate_skill_name": "demo",
        "proposed_action": action,
        "candidate_skill": {
            "name": "demo",
            "description": extra.pop("description", "stub description"),
        },
        "rationale": extra.pop("rationale", ""),
    }
    base.update(extra)
    return base


class TestBuildPendingCandidatesBlock:
    def test_empty_returns_empty_string(self):
        # Caller short-circuits the section header on empty.
        assert _build_pending_candidates_block([]) == ""

    def test_renders_section_header(self):
        block = _build_pending_candidates_block([_job("20260101000000-x-aaa")])
        assert "Pending candidates already queued for this skill" in block
        assert "20260101000000-x-aaa" in block

    def test_orders_newest_first(self):
        # Timestamp prefix means descending string sort = newest-first.
        items = [
            _job("20260101000000-demo-aaa", description="oldest"),
            _job("20260201000000-demo-bbb", description="middle"),
            _job("20260301000000-demo-ccc", description="newest"),
        ]
        block = _build_pending_candidates_block(items)
        idx_oldest = block.index("oldest")
        idx_middle = block.index("middle")
        idx_newest = block.index("newest")
        assert idx_newest < idx_middle < idx_oldest

    def test_caps_at_max_entries(self):
        items = [
            _job(f"20260101{i:06d}-demo-z", description=f"d{i}")
            for i in range(8)
        ]
        block = _build_pending_candidates_block(items, max_entries=2)
        # Only the two newest descriptions appear.
        assert "d7" in block
        assert "d6" in block
        assert "d0" not in block
        assert "d3" not in block

    def test_includes_body_excerpt_only_for_improve_skill(self):
        # improve_skill candidates carry full body — the LLM needs to
        # see it to decide whether to update / reject.
        improve = _job(
            "20260101000000-demo-imp",
            action="improve_skill",
            candidate_skill={
                "name": "demo",
                "description": "d",
                "content": "## Section A\n\nbody text here\n",
            },
        )
        block = _build_pending_candidates_block([improve])
        assert "body text here" in block

        # optimize_description doesn't carry meaningful body changes
        # — skip the excerpt to keep the prompt budget lean.
        optimize = _job(
            "20260101000000-demo-opt",
            action="optimize_description",
            candidate_skill={
                "name": "demo",
                "description": "d",
                "content": "the long-form body content that should NOT be in the prompt",
            },
        )
        block2 = _build_pending_candidates_block([optimize])
        assert (
            "the long-form body content that should NOT be in the prompt"
            not in block2
        )

    def test_skips_non_dict_entries(self):
        # Defensive — a hand-edited validation_jobs/ might have a junk
        # entry; the renderer mustn't crash.
        block = _build_pending_candidates_block(
            [_job("20260101000000-real-aaa"), "not a dict", None],
        )
        assert "20260101000000-real-aaa" in block

    def test_excerpt_truncation(self):
        big_body = "x" * 5000
        improve = _job(
            "20260101000000-demo-big",
            action="improve_skill",
            candidate_skill={
                "name": "demo",
                "description": "d",
                "content": big_body,
            },
        )
        block = _build_pending_candidates_block([improve])
        # The body section is clipped — no full 5000-char run survives.
        assert "[truncated]" in block
        assert block.count("x" * 1500) == 0


# --- _parse_evolve_result for pending-aware actions ----------------------- #


def _wrap_json(d: dict) -> str:
    """Mimic the LLM's typical fence-wrapped output to ensure the
    parser strips Markdown fences correctly."""
    return "```json\n" + json.dumps(d) + "\n```"


class TestParseSkipRedundant:
    def test_well_formed_returns_target(self):
        raw = _wrap_json({
            "action": "skip_redundant",
            "rationale": "pending 20260101000000-demo-x already covers this",
            "target_pending_job_id": "20260101000000-demo-x",
        })
        out = _parse_evolve_result(raw, "demo")
        assert out is not None
        assert out["action"] == DecisionAction.SKIP_REDUNDANT
        assert out["target_pending_job_id"] == "20260101000000-demo-x"
        # No skill payload required for this action.
        assert "skill" not in out

    def test_missing_target_demotes_to_plain_skip(self):
        raw = _wrap_json({
            "action": "skip_redundant",
            "rationale": "pending covers it (i forgot the id)",
        })
        out = _parse_evolve_result(raw, "demo")
        assert out is not None
        assert out["action"] == DecisionAction.SKIP


class TestParseUpdatePending:
    def test_well_formed_carries_target_and_skill(self):
        raw = _wrap_json({
            "action": "update_pending_candidate",
            "rationale": "tighten description",
            "target_pending_job_id": "20260101000000-demo-x",
            "skill": {
                "name": "demo",
                "description": "tighter description",
            },
        })
        out = _parse_evolve_result(raw, "demo")
        assert out is not None
        assert out["action"] == DecisionAction.UPDATE_PENDING
        assert out["target_pending_job_id"] == "20260101000000-demo-x"
        assert out["skill"]["description"] == "tighter description"

    def test_fills_missing_skill_name(self):
        raw = _wrap_json({
            "action": "update_pending_candidate",
            "rationale": "...",
            "target_pending_job_id": "20260101000000-demo-x",
            "skill": {"description": "no name in payload"},
        })
        out = _parse_evolve_result(raw, "demo")
        assert out["skill"]["name"] == "demo"

    def test_missing_skill_demotes_to_skip(self):
        raw = _wrap_json({
            "action": "update_pending_candidate",
            "rationale": "...",
            "target_pending_job_id": "20260101000000-demo-x",
        })
        out = _parse_evolve_result(raw, "demo")
        assert out["action"] == DecisionAction.SKIP

    def test_missing_target_demotes_to_skip(self):
        raw = _wrap_json({
            "action": "update_pending_candidate",
            "rationale": "...",
            "skill": {"name": "demo", "description": "x"},
        })
        out = _parse_evolve_result(raw, "demo")
        assert out["action"] == DecisionAction.SKIP


class TestParseRejectPending:
    def test_well_formed_returns_target(self):
        raw = _wrap_json({
            "action": "reject_pending_candidate",
            "rationale": "contradicted by new evidence",
            "target_pending_job_id": "20260101000000-demo-x",
        })
        out = _parse_evolve_result(raw, "demo")
        assert out["action"] == DecisionAction.REJECT_PENDING
        assert out["target_pending_job_id"] == "20260101000000-demo-x"

    def test_missing_target_demotes_to_skip(self):
        raw = _wrap_json({
            "action": "reject_pending_candidate",
            "rationale": "wrong",
        })
        out = _parse_evolve_result(raw, "demo")
        assert out["action"] == DecisionAction.SKIP


class TestOriginalActionsUnchanged:
    """Regression — making sure adding new branches didn't break the
    four pre-existing actions."""

    @pytest.mark.parametrize("action", [
        DecisionAction.IMPROVE,
        DecisionAction.OPTIMIZE_DESC,
    ])
    def test_existing_skill_actions_round_trip(self, action):
        raw = _wrap_json({
            "action": action,
            "rationale": "...",
            "skill": {
                "name": "demo",
                "description": "d",
                "content": "body",
            },
        })
        out = _parse_evolve_result(raw, "demo")
        assert out["action"] == action
        assert out["skill"]["name"] == "demo"

    def test_create_skill_round_trip(self):
        raw = _wrap_json({
            "action": "create_skill",
            "rationale": "...",
            "skill": {
                "name": "new-thing",
                "description": "d",
                "content": "body",
            },
        })
        out = _parse_evolve_result(raw, "demo")
        assert out["action"] == DecisionAction.CREATE
        assert out["skill"]["name"] == "new-thing"

    def test_skip_round_trip(self):
        raw = _wrap_json({"action": "skip", "rationale": "no signal"})
        out = _parse_evolve_result(raw, "demo")
        assert out["action"] == DecisionAction.SKIP
        assert "skill" not in out


class TestPreserveExistingBodyForOptimizeDesc:
    """Regression — ``optimize_description`` LLM contract returns only
    ``name`` + ``description``. Without the splice, the candidate stored
    in the validation job has empty body and finalize destroys the live
    SKILL.md.  See ``_EVOLVE_FROM_SESSIONS_SYSTEM`` lines 159-168 for the
    contract that motivates this defensive splice.
    """

    def _current(self):
        return {
            "name": "demo",
            "description": "old desc",
            "content": "# Heading\n\nfull body that must survive\n",
            "category": "general",
        }

    def test_splices_body_when_optimize_desc_has_empty_content(self):
        parsed = {
            "action": DecisionAction.OPTIMIZE_DESC,
            "skill": {"name": "demo", "description": "tightened desc"},
        }
        out = _preserve_existing_body_for_optimize_desc(parsed, self._current())
        assert out is parsed
        assert out["skill"]["content"] == "# Heading\n\nfull body that must survive\n"
        assert out["skill"]["category"] == "general"
        assert out["skill"]["description"] == "tightened desc"

    def test_does_not_overwrite_when_llm_returned_body(self):
        parsed = {
            "action": DecisionAction.OPTIMIZE_DESC,
            "skill": {
                "name": "demo",
                "description": "tightened",
                "content": "llm-provided body",
            },
        }
        out = _preserve_existing_body_for_optimize_desc(parsed, self._current())
        assert out["skill"]["content"] == "llm-provided body"

    def test_no_op_for_other_actions(self):
        parsed = {
            "action": DecisionAction.IMPROVE,
            "skill": {"name": "demo", "description": "x"},
        }
        out = _preserve_existing_body_for_optimize_desc(parsed, self._current())
        assert "content" not in out["skill"]

    def test_no_op_when_current_skill_missing(self):
        parsed = {
            "action": DecisionAction.OPTIMIZE_DESC,
            "skill": {"name": "demo", "description": "x"},
        }
        out = _preserve_existing_body_for_optimize_desc(parsed, None)
        assert "content" not in out["skill"]

    def test_no_op_when_current_skill_has_no_body(self):
        parsed = {
            "action": DecisionAction.OPTIMIZE_DESC,
            "skill": {"name": "demo", "description": "x"},
        }
        out = _preserve_existing_body_for_optimize_desc(
            parsed,
            {"name": "demo", "description": "old"},
        )
        assert "content" not in out["skill"]

    def test_does_not_overwrite_existing_category(self):
        parsed = {
            "action": DecisionAction.OPTIMIZE_DESC,
            "skill": {
                "name": "demo",
                "description": "tightened",
                "category": "workflow",
            },
        }
        out = _preserve_existing_body_for_optimize_desc(parsed, self._current())
        assert out["skill"]["category"] == "workflow"

    def test_handles_none_parsed(self):
        # _parse_evolve_result returns None on malformed JSON; helper
        # must propagate None unchanged.
        assert _preserve_existing_body_for_optimize_desc(None, self._current()) is None

    def test_handles_skill_field_missing(self):
        # Skip actions have no "skill" key — guard against KeyError.
        parsed = {"action": DecisionAction.SKIP, "rationale": "..."}
        out = _preserve_existing_body_for_optimize_desc(parsed, self._current())
        assert out == {"action": DecisionAction.SKIP, "rationale": "..."}
