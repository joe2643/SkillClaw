# -*- coding: utf-8 -*-
"""Tests for ``DashboardService._post_publish_sync_to_local``.

The publish chain is: evolve_server writes new candidate to the
SkillClaw shared bucket → ``sync_skills`` pulls from bucket to the
local ``skills_dir`` (which CoPaw points at as its ``skill_pool``)
→ next agent invocation re-loads the updated skill.

Without ``_post_publish_sync_to_local`` chained into ``trigger_evolve``,
step 2 only happened when an operator clicked "Sync" manually — so a
running CoPaw agent kept rendering the stale prompt-injected skill
content for as long as the dashboard's auto-finalize path didn't
re-sync.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from tests.test_dashboard import DashboardFixture
from skillclaw.dashboard_server import DashboardService


class PostPublishSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = DashboardFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def test_calls_sync_skills_when_sharing_enabled(self) -> None:
        """Sharing is on (DashboardFixture default) → must use the
        bidirectional ``sync_skills`` so any newly-published skills in
        the shared bucket land in ``skills_dir`` for CoPaw to pick up."""
        service = DashboardService(self.fixture.config)
        with patch.object(
            service, "sync_skills",
            return_value={
                "operation": "sync",
                "target": {"backend": "local"},
                "result": {"pull": {}, "push": {}},
                "sync": {"skills": 5, "sessions": 0, "validation_jobs": 0},
            },
        ) as mock_sync_skills:
            out = service._post_publish_sync_to_local({"published": 1})

        mock_sync_skills.assert_called_once()
        self.assertIn("sync", out)
        self.assertEqual(out["sync"]["skills"], 5)
        # The bidirectional pull/push detail is exposed so callers
        # (and the dashboard frontend) can show what changed.
        self.assertIn("skills_synced", out)
        self.assertEqual(out["skills_synced"], {"pull": {}, "push": {}})

    def test_falls_back_to_plain_sync_when_sharing_disabled(self) -> None:
        """If sharing is disabled the embedded fallback in
        ``trigger_evolve`` would have already raised — but the helper
        is defensive: a future caller from another path won't crash
        on a config that lacks sharing."""
        service = DashboardService(self.fixture.config)
        service.config.sharing_enabled = False
        with patch.object(
            service, "sync",
            return_value={"summary": {"skills": 0}},
        ) as mock_plain_sync, patch.object(
            service, "sync_skills",
        ) as mock_sync_skills:
            out = service._post_publish_sync_to_local({})

        mock_sync_skills.assert_not_called()
        mock_plain_sync.assert_called_once()
        # ``skills_synced`` is omitted on the disabled-sharing path —
        # the dashboard frontend can detect "no skill push happened".
        self.assertNotIn("skills_synced", out)
        self.assertEqual(out["sync"]["skills"], 0)

    def test_swallows_sync_skills_failure_and_falls_back(self) -> None:
        """A flaky shared bucket / network must NEVER roll back the
        evolve publish — the new skill is already in the bucket; this
        helper just loses the local-pool refresh.  Log + fall through
        to plain ``sync`` so the dashboard projection still updates."""
        service = DashboardService(self.fixture.config)
        with patch.object(
            service, "sync_skills",
            side_effect=RuntimeError("simulated S3 timeout"),
        ) as mock_sync_skills, patch.object(
            service, "sync",
            return_value={"summary": {"skills": 7}},
        ) as mock_plain_sync:
            out = service._post_publish_sync_to_local({})

        mock_sync_skills.assert_called_once()
        mock_plain_sync.assert_called_once()
        self.assertEqual(out["sync"]["skills"], 7)
        self.assertNotIn("skills_synced", out)


if __name__ == "__main__":
    unittest.main()
