# -*- coding: utf-8 -*-
"""Tests for PRMScorer's process-wide concurrency throttle.

Background
----------
Previously, 3 parallel-voting sessions could fire 9 concurrent LLM calls,
saturating DashScope's concurrency quota. The scorer is now equipped with
an ``asyncio.Semaphore(concurrency)`` so ``concurrency=1`` strictly
serializes all scoring LLM calls across every session.
"""

# pylint: disable=protected-access
import asyncio
from unittest.mock import MagicMock

import pytest

from skillclaw.prm_scorer import PRMScorer


def _fake_completion(text: str = "Score: 1"):
    choice = MagicMock()
    choice.message.content = text
    completion = MagicMock()
    completion.choices = [choice]
    return completion


class _RecordingClient:
    """Minimal chat-completions client that records start/end timestamps."""

    def __init__(self, delay: float = 0.05):
        self._delay = delay
        self.started = 0
        self.finished = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()
        self.chat = MagicMock()
        self.chat.completions.create = self._create

    def _create(self, **_kwargs):
        # synchronous call because PRMScorer wraps it in asyncio.to_thread
        import time
        time.sleep(self._delay)
        return _fake_completion()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _make_scorer(concurrency: int, client) -> PRMScorer:
    return PRMScorer(
        prm_url="http://unused",
        prm_model="test",
        api_key="x",
        prm_m=1,
        concurrency=concurrency,
        llm_client=client,
    )


@pytest.mark.asyncio
async def test_concurrency_1_serializes_calls_across_sessions(monkeypatch):
    """With concurrency=1, many parallel evaluate() calls run strictly serially."""
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    class TrackingClient:
        def __init__(self):
            self.chat = MagicMock()
            self.chat.completions.create = self._create

        def _create(self, **_kwargs):
            nonlocal in_flight, peak
            # acquire by hand to record exact simultaneity
            with _SyncLock():
                in_flight += 1
                peak = max(peak, in_flight)
            import time
            time.sleep(0.02)
            with _SyncLock():
                in_flight -= 1
            return _fake_completion()

    scorer = _make_scorer(concurrency=1, client=TrackingClient())

    # Fire 5 concurrent evaluations from 5 different "sessions"
    tasks = [
        asyncio.create_task(
            scorer.evaluate("response", "instruction", session_id=f"s{i}", turn_num=1)
        )
        for i in range(5)
    ]
    await asyncio.gather(*tasks)

    assert peak == 1, (
        f"peak concurrent LLM calls was {peak}; concurrency=1 should serialize to 1"
    )


@pytest.mark.asyncio
async def test_concurrency_3_allows_up_to_3_in_flight(monkeypatch):
    """concurrency=3 lets up to 3 LLM calls run in parallel but no more."""
    in_flight = 0
    peak = 0

    class TrackingClient:
        def __init__(self):
            self.chat = MagicMock()
            self.chat.completions.create = self._create

        def _create(self, **_kwargs):
            nonlocal in_flight, peak
            with _SyncLock():
                in_flight += 1
                peak = max(peak, in_flight)
            import time
            time.sleep(0.05)
            with _SyncLock():
                in_flight -= 1
            return _fake_completion()

    scorer = _make_scorer(concurrency=3, client=TrackingClient())

    tasks = [
        asyncio.create_task(
            scorer.evaluate("r", "i", session_id=f"s{i}", turn_num=1)
        )
        for i in range(10)
    ]
    await asyncio.gather(*tasks)

    assert 1 < peak <= 3, (
        f"peak concurrent LLM calls was {peak}; concurrency=3 should cap at 3"
    )


@pytest.mark.asyncio
async def test_semaphore_released_on_exception():
    """If a scoring call raises, the semaphore must still be released."""
    calls = {"n": 0}

    class FlakyClient:
        def __init__(self):
            self.chat = MagicMock()
            self.chat.completions.create = self._create

        def _create(self, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated upstream failure")
            return _fake_completion()

    scorer = _make_scorer(concurrency=1, client=FlakyClient())

    # First call fails; second must still be able to acquire the semaphore.
    r1 = await scorer.evaluate("r", "i", session_id="s1", turn_num=1)
    r2 = await scorer.evaluate("r", "i", session_id="s2", turn_num=1)

    assert r1["score"] == 0.0  # failure → 0 via majority vote
    assert r2["score"] == 1.0
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_default_concurrency_is_one():
    """Default concurrency must be 1 so enabling PRM never exceeds 1 LLM call at a time."""
    scorer = PRMScorer(
        prm_url="http://unused",
        prm_model="test",
        prm_m=1,
        llm_client=_SilentClient(),
    )
    # Introspect the semaphore value
    assert scorer._llm_semaphore._value == 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _SyncLock:
    """Thread-safe guard used inside sync mock bodies (asyncio.to_thread runs in a thread)."""
    _lock = None

    def __enter__(self):
        if _SyncLock._lock is None:
            import threading
            _SyncLock._lock = threading.Lock()
        _SyncLock._lock.acquire()
        return self

    def __exit__(self, *_exc):
        _SyncLock._lock.release()


class _SilentClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions.create = MagicMock(return_value=_fake_completion())
