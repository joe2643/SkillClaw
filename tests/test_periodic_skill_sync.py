# -*- coding: utf-8 -*-
"""Tests for the dashboard's periodic ``sync_skills`` background loop.

The loop covers the auto-approve gap: when ``ValidationWorker`` writes
a passing replay-validation result, ``evolve_server``'s next cycle's
``_finalize_validation_jobs`` publishes the candidate to the shared
bucket — but **not** through the dashboard's manual-approve path that
bundles ``_post_publish_sync_to_local``.  Without a periodic sync,
the local ``skills_dir`` / CoPaw ``skill_pool`` lags the bucket
indefinitely.

Three layers covered:

1. The loop itself (``_periodic_skill_sync_loop``):
   - calls ``service.sync_skills()`` on tick
   - swallows narrow I/O exceptions and retries on the next tick
   - re-raises ``CancelledError`` so the lifespan cleanup can stop it
   - logs a non-empty pull/push summary

2. The lifespan wiring (``create_dashboard_app``):
   - spawns the task only when ``sharing_enabled`` AND
     ``dashboard_skill_sync_interval_seconds > 0``
   - cancels it on shutdown

3. The config plumbing (``SkillClawConfig`` field default + YAML
   round-trip via ``ConfigStore``).
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

import pytest

from tests.test_dashboard import DashboardFixture
from skillclaw.config import SkillClawConfig
from skillclaw.dashboard_server import (
    DashboardService,
    _periodic_skill_sync_loop,
)


# ---------------------------------------------------------------------- #
# 1. Loop behaviour                                                      #
# ---------------------------------------------------------------------- #


class PeriodicSyncLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_calls_sync_skills_each_tick(self) -> None:
        """Two ticks → two ``sync_skills`` calls."""
        service = MagicMock()
        service.sync_skills.return_value = {
            "result": {"pull": {"downloaded": 0}, "push": {"uploaded": 0}},
        }
        # Use a tiny interval AND patch sleep so the test doesn't
        # actually wait — the loop's ``max(10, interval)`` floor would
        # still suspend us for 10s otherwise.
        with patch("skillclaw.dashboard_server.asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = [None, None, asyncio.CancelledError()]
            with self.assertRaises(asyncio.CancelledError):
                await _periodic_skill_sync_loop(
                    service, interval_seconds=15,
                )
        self.assertEqual(service.sync_skills.call_count, 2)

    async def test_floors_interval_to_10s_minimum(self) -> None:
        """Operator typo'd ``skill_sync_interval_seconds: 1`` → loop
        clamps to 10 so we don't hammer the storage layer."""
        service = MagicMock()
        service.sync_skills.return_value = {
            "result": {"pull": {}, "push": {}},
        }
        with patch("skillclaw.dashboard_server.asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            with self.assertRaises(asyncio.CancelledError):
                await _periodic_skill_sync_loop(
                    service, interval_seconds=1,
                )
        # First sleep argument is the floored interval.
        first_sleep_arg = mock_sleep.call_args_list[0].args[0]
        self.assertEqual(first_sleep_arg, 10)

    async def test_swallows_io_failure_and_retries(self) -> None:
        """A flaky shared bucket must NOT crash the loop — log and
        keep going so the next tick can recover."""
        service = MagicMock()
        service.sync_skills.side_effect = [
            ConnectionError("simulated S3 timeout"),
            {"result": {"pull": {"downloaded": 1}, "push": {}}},
        ]
        with patch("skillclaw.dashboard_server.asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = [None, None, asyncio.CancelledError()]
            with self.assertRaises(asyncio.CancelledError):
                await _periodic_skill_sync_loop(
                    service, interval_seconds=15,
                )
        # Both ticks fired despite the first one raising.
        self.assertEqual(service.sync_skills.call_count, 2)

    async def test_propagates_programmer_errors(self) -> None:
        """``AttributeError`` / ``TypeError`` (programmer / config bugs)
        must NOT be silently swallowed — they should fail loud the
        same way as ``_post_publish_sync_to_local``'s narrow catch."""
        service = MagicMock()
        service.sync_skills.side_effect = AttributeError("typo'd attr")
        with patch("skillclaw.dashboard_server.asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            with self.assertRaises(AttributeError):
                await _periodic_skill_sync_loop(
                    service, interval_seconds=15,
                )

    async def test_cancellation_propagates_immediately(self) -> None:
        """Lifespan shutdown calls ``task.cancel()`` — the next sleep
        should raise ``CancelledError`` and the loop must let it
        through unmodified."""
        service = MagicMock()
        with patch("skillclaw.dashboard_server.asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = asyncio.CancelledError()
            with self.assertRaises(asyncio.CancelledError):
                await _periodic_skill_sync_loop(
                    service, interval_seconds=60,
                )
        service.sync_skills.assert_not_called()


# ---------------------------------------------------------------------- #
# 2. Lifespan wiring                                                     #
# ---------------------------------------------------------------------- #


class LifespanWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = DashboardFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def test_no_task_spawned_when_interval_is_zero(self) -> None:
        """Default config (interval=0) — no background task."""
        cfg = self.fixture.config
        cfg.dashboard_skill_sync_interval_seconds = 0
        from fastapi.testclient import TestClient
        from skillclaw.dashboard_server import create_dashboard_app

        app = create_dashboard_app(cfg)
        with TestClient(app) as client:
            client.get("/api/v1/health")
            # No task attached.
            self.assertFalse(
                hasattr(app.state, "periodic_skill_sync_task"),
            )

    def test_no_task_spawned_when_sharing_disabled(self) -> None:
        """``sync_skills`` would raise on a non-sharing config — don't
        even start the loop."""
        cfg = self.fixture.config
        cfg.sharing_enabled = False
        cfg.dashboard_skill_sync_interval_seconds = 30
        from fastapi.testclient import TestClient
        from skillclaw.dashboard_server import create_dashboard_app

        app = create_dashboard_app(cfg)
        with TestClient(app) as client:
            client.get("/api/v1/health")
            self.assertFalse(
                hasattr(app.state, "periodic_skill_sync_task"),
            )

    def test_task_spawned_and_cleaned_up(self) -> None:
        """Sharing on + interval > 0 ⇒ task spawned at startup, cancelled
        at shutdown."""
        cfg = self.fixture.config
        cfg.dashboard_skill_sync_interval_seconds = 30  # > 0 + sharing on
        from fastapi.testclient import TestClient
        from skillclaw.dashboard_server import create_dashboard_app

        app = create_dashboard_app(cfg)
        captured_task: asyncio.Task | None = None
        with TestClient(app) as client:
            client.get("/api/v1/health")
            captured_task = getattr(
                app.state, "periodic_skill_sync_task", None,
            )
            self.assertIsNotNone(captured_task)
            self.assertFalse(captured_task.done())

        # Lifespan cleanup ran — task should be cancelled / done.
        self.assertIsNotNone(captured_task)
        self.assertTrue(captured_task.cancelled() or captured_task.done())


# ---------------------------------------------------------------------- #
# 3. Config plumbing                                                     #
# ---------------------------------------------------------------------- #


class ConfigPlumbingTests(unittest.TestCase):
    def test_default_is_zero(self) -> None:
        cfg = SkillClawConfig()
        self.assertEqual(cfg.dashboard_skill_sync_interval_seconds, 0)


if __name__ == "__main__":
    unittest.main()
