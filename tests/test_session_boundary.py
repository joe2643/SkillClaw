"""Unit tests for the SkillClaw session-boundary detector.

Covers ``_resolve_external_session`` — the helper that watches for
``/compact``, ``/clear``, ``--resume`` and idle-timeout boundaries even
when the caller already supplies an explicit session_id (Claude Code's
``x-claude-code-session-id``, Codex's ``session_id``, OpenClaw's
``X-Session-Id``, CoPaw's ``/ingest`` body field).  Also re-asserts
the original TUI heuristic in ``_resolve_tui_session`` still works
after the refactor + lock.

No HTTP / disk I/O except the dedicated /ingest end-to-end tests
(which use ASGITransport): we instantiate ``SkillClawAPIServer``
directly, stub the tokenizer loader, and intercept ``_close_session``
so the test doesn't touch the record-upload path.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from skillclaw.api_server import SkillClawAPIServer
from skillclaw.config import SkillClawConfig

# Every test in this module is an async coroutine — apply the
# pytest-asyncio marker module-wide so the existing test config
# (no asyncio_mode=auto in pyproject) doesn't reject them.
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------- #
# Fixtures                                                         #
# ---------------------------------------------------------------- #


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, tmp_path) -> SkillClawAPIServer:
    """Bare-bones SkillClawAPIServer with the tokenizer skipped.

    ``_close_session`` is replaced with an async spy so tests can assert
    it fires on boundary without exercising the upload pipeline.
    """
    monkeypatch.setattr(SkillClawAPIServer, "_load_tokenizer", lambda self: None)
    srv = SkillClawAPIServer(
        SkillClawConfig(
            record_enabled=False,
            record_dir=str(tmp_path / "records"),
        ),
    )

    closed: list[tuple[str, str]] = []

    async def _spy_close(session_id: str, reason: str = "explicit") -> None:
        closed.append((session_id, reason))

    monkeypatch.setattr(srv, "_close_session", _spy_close)
    srv._closed_sessions_log = closed  # type: ignore[attr-defined]
    return srv


# ---------------------------------------------------------------- #
# _sanitize_raw_session_id                                         #
# ---------------------------------------------------------------- #


class TestSanitize:
    # These two are sync tests inside an async-marked module; the
    # async wrapper accepts sync coroutines but emits a warning.
    # Make them async no-op-wrap to silence the warning.
    async def test_strips_seg_suffix(self) -> None:
        assert SkillClawAPIServer._sanitize_raw_session_id("abc:seg-2") == "abc"
        assert SkillClawAPIServer._sanitize_raw_session_id("abc:seg-99") == "abc"

    async def test_passes_clean_id_through(self) -> None:
        assert SkillClawAPIServer._sanitize_raw_session_id("abc") == "abc"
        assert SkillClawAPIServer._sanitize_raw_session_id("console:joe") == "console:joe"
        # ``:seg-`` only matches as a whole-suffix pattern with digits.
        assert SkillClawAPIServer._sanitize_raw_session_id("abc:seg-foo") == "abc:seg-foo"
        assert SkillClawAPIServer._sanitize_raw_session_id("seg-2") == "seg-2"

    async def test_sanitization_applies_inside_resolver(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # Caller-supplied "abc:seg-2" must NOT collide with the
        # segment-2 namespace owned by raw_sid "abc".
        sid_a = await server._resolve_external_session("abc", msg_count=10)
        # Now a malicious/lucky caller sends "abc:seg-2" — it gets
        # stripped to "abc" before keying lookup.
        sid_b = await server._resolve_external_session("abc:seg-2", msg_count=10)
        assert sid_a == sid_b == "abc"
        # Single bucket, not two.
        assert list(server._external_session_meta.keys()) == ["abc"]


# ---------------------------------------------------------------- #
# _resolve_external_session                                        #
# ---------------------------------------------------------------- #


class TestExternalSessionFirstRequest:
    async def test_first_request_returns_raw_session_id_unchanged(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # First time we see a session_id, no boundary call should fire
        # and the externally-supplied ID should round-trip verbatim.
        sid = await server._resolve_external_session(
            "claude-abc-123",
            msg_count=10,
        )
        assert sid == "claude-abc-123"
        assert server._closed_sessions_log == []

    async def test_first_request_records_metadata(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        await server._resolve_external_session(
            "claude-abc-123",
            msg_count=10,
        )
        meta = server._external_session_meta["claude-abc-123"]
        assert meta["segment"] == 1
        assert meta["current_sid"] == "claude-abc-123"
        assert meta["last_msg_count"] == 10
        assert meta["last_request_time"] <= time.time()


class TestExternalSessionMonotonicGrowth:
    async def test_growing_msg_count_keeps_same_segment(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # A normal conversation grows by 1-2 messages per turn — that
        # must NOT be treated as a boundary.
        first = await server._resolve_external_session("claude-abc-123", msg_count=10)
        second = await server._resolve_external_session("claude-abc-123", msg_count=12)
        third = await server._resolve_external_session("claude-abc-123", msg_count=14)
        assert first == second == third == "claude-abc-123"
        assert server._closed_sessions_log == []

    async def test_same_msg_count_keeps_same_segment(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        first = await server._resolve_external_session("claude-abc-123", msg_count=10)
        second = await server._resolve_external_session("claude-abc-123", msg_count=10)
        assert first == second
        assert server._closed_sessions_log == []


class TestExternalSessionCompactBoundary:
    async def test_msg_count_drop_triggers_new_segment(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # Simulate Claude Code's /compact: 50 → 5 messages while
        # preserving the session_id on the wire.  Must see new segment.
        first = await server._resolve_external_session("claude-abc-123", msg_count=50)
        second = await server._resolve_external_session("claude-abc-123", msg_count=5)
        assert first == "claude-abc-123"
        assert second == "claude-abc-123:seg-2"
        assert server._closed_sessions_log == [
            ("claude-abc-123", "external_boundary"),
        ]

    async def test_msg_count_drop_to_zero_triggers_new_segment(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        await server._resolve_external_session("claude-abc-123", msg_count=30)
        second = await server._resolve_external_session("claude-abc-123", msg_count=1)
        assert second == "claude-abc-123:seg-2"


class TestExternalSessionInactivity:
    async def test_inactivity_timeout_triggers_new_segment(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # Shorten the inactivity window so the test doesn't sleep 5 min.
        server._tui_inactivity_timeout = 1

        first = await server._resolve_external_session("claude-abc-123", msg_count=10)
        assert first == "claude-abc-123"

        # Force the stored timestamp backwards instead of sleeping.
        meta = server._external_session_meta["claude-abc-123"]
        meta["last_request_time"] -= 5

        second = await server._resolve_external_session("claude-abc-123", msg_count=11)
        # Even though msg_count grew (no compact), the idle gap should
        # have rotated the segment.
        assert second == "claude-abc-123:seg-2"
        assert server._closed_sessions_log == [
            ("claude-abc-123", "external_boundary"),
        ]


class TestExternalSessionMultipleBoundaries:
    async def test_repeated_compacts_increment_segment_number(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # A long chat with three /compact operations should yield
        # seg-1 (implicit), seg-2, seg-3, seg-4.
        sids = []
        sids.append(await server._resolve_external_session("claude-abc-123", msg_count=40))
        sids.append(await server._resolve_external_session("claude-abc-123", msg_count=5))   # #1
        await server._resolve_external_session("claude-abc-123", msg_count=20)
        sids.append(await server._resolve_external_session("claude-abc-123", msg_count=3))   # #2
        await server._resolve_external_session("claude-abc-123", msg_count=15)
        sids.append(await server._resolve_external_session("claude-abc-123", msg_count=2))   # #3
        assert sids == [
            "claude-abc-123",
            "claude-abc-123:seg-2",
            "claude-abc-123:seg-3",
            "claude-abc-123:seg-4",
        ]
        # One close per boundary — three total, following segment chain.
        assert server._closed_sessions_log == [
            ("claude-abc-123", "external_boundary"),
            ("claude-abc-123:seg-2", "external_boundary"),
            ("claude-abc-123:seg-3", "external_boundary"),
        ]


class TestExternalSessionIsolation:
    async def test_different_raw_session_ids_tracked_independently(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # Two concurrent Claude Code instances each get their own
        # segment counter — a compact in one must not bump the other.
        await server._resolve_external_session("claude-alpha", msg_count=30)
        await server._resolve_external_session("claude-beta", msg_count=30)
        a2 = await server._resolve_external_session("claude-alpha", msg_count=3)
        b2 = await server._resolve_external_session("claude-beta", msg_count=32)
        assert a2 == "claude-alpha:seg-2"
        assert b2 == "claude-beta"
        # Only the alpha session closed.
        assert server._closed_sessions_log == [
            ("claude-alpha", "external_boundary"),
        ]

    async def test_model_switch_keeps_shared_boundary_state(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # Bucket key is raw_session_id ONLY (not (raw_sid, model)).
        # A mid-session model switch (e.g. user toggles Claude Code
        # from sonnet to opus) MUST keep the same boundary state so a
        # compact on the new model still triggers a segment bump.
        await server._resolve_external_session("claude-abc-123", msg_count=50)
        # Model switch + compact in one move.
        sid = await server._resolve_external_session("claude-abc-123", msg_count=5)
        assert sid == "claude-abc-123:seg-2"
        assert server._closed_sessions_log == [
            ("claude-abc-123", "external_boundary"),
        ]


class TestExternalSessionConcurrency:
    async def test_concurrent_compact_only_bumps_segment_once(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # Single-flight gate.  Two coroutines that both observe the
        # boundary condition simultaneously must NOT both bump the
        # segment counter — without the lock, the second caller would
        # see meta still pointing at seg-1 across the _close_session
        # await, increment to seg-2, while the first caller also
        # increments to seg-2 → both write the same current_sid AND
        # _close_session fires twice on the same old sid.
        await server._resolve_external_session("claude-abc-123", msg_count=50)
        results = await asyncio.gather(
            server._resolve_external_session("claude-abc-123", msg_count=5),
            server._resolve_external_session("claude-abc-123", msg_count=5),
        )
        # Both callers see the SAME post-boundary sid (seg-2).
        assert results == ["claude-abc-123:seg-2", "claude-abc-123:seg-2"]
        # And _close_session fired exactly ONCE on the pre-boundary sid.
        assert server._closed_sessions_log == [
            ("claude-abc-123", "external_boundary"),
        ]
        # Final state: segment counter at exactly 2 (NOT 3 or higher).
        assert server._external_session_meta["claude-abc-123"]["segment"] == 2


class TestExternalSessionEviction:
    async def test_close_session_evicts_matching_external_meta(
        self,
        server: SkillClawAPIServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # _external_session_meta must be evicted when the matching
        # current_sid is closed — otherwise the dict grows monotonically
        # AND a subsequent request would re-route into the just-closed sid.
        await server._resolve_external_session("claude-abc-123", msg_count=10)
        assert "claude-abc-123" in server._external_session_meta

        # Restore the real _close_session so the eviction code path runs.
        # The fixture replaced it with a spy that doesn't evict.
        monkeypatch.undo()
        monkeypatch.setattr(SkillClawAPIServer, "_load_tokenizer", lambda self: None)

        await server._close_session("claude-abc-123", reason="test")
        # External meta for raw_sid "claude-abc-123" should now be gone.
        assert "claude-abc-123" not in server._external_session_meta

    async def test_close_session_only_evicts_matching_current_sid(
        self,
        server: SkillClawAPIServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Eviction matches on current_sid, not raw_session_id.  If raw
        # "abc" has been bumped to seg-2 (current_sid="abc:seg-2"),
        # closing the original sid "abc" must NOT evict the meta —
        # current_sid is "abc:seg-2", not "abc".
        await server._resolve_external_session("abc", msg_count=20)
        await server._resolve_external_session("abc", msg_count=2)
        assert server._external_session_meta["abc"]["current_sid"] == "abc:seg-2"

        monkeypatch.undo()
        monkeypatch.setattr(SkillClawAPIServer, "_load_tokenizer", lambda self: None)

        # Closing the ORIGINAL sid (no longer current) — should leave meta intact.
        await server._close_session("abc", reason="late_cleanup")
        assert "abc" in server._external_session_meta

        # Closing the CURRENT sid (seg-2) — should evict.
        await server._close_session("abc:seg-2", reason="test")
        assert "abc" not in server._external_session_meta


# ---------------------------------------------------------------- #
# _resolve_tui_session regression (plus its own concurrency)       #
# ---------------------------------------------------------------- #


class TestTuiSessionRegression:
    """The existing TUI heuristic still works AND now shares the
    single-flight gate so the same race that affected the external
    path is closed here too."""

    async def test_first_tui_request_creates_session(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        sid = await server._resolve_tui_session("grok-4", msg_count=5)
        assert sid.startswith("tui-grok-4-")
        assert server._closed_sessions_log == []

    async def test_tui_msg_count_drop_rotates_session(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        first = await server._resolve_tui_session("grok-4", msg_count=30)
        second = await server._resolve_tui_session("grok-4", msg_count=3)
        assert first != second
        assert second.startswith("tui-grok-4-")
        assert server._closed_sessions_log == [(first, "tui_boundary")]

    async def test_concurrent_tui_compact_only_closes_once(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # Same single-flight property as the external-session test.
        first = await server._resolve_tui_session("grok-4", msg_count=30)
        results = await asyncio.gather(
            server._resolve_tui_session("grok-4", msg_count=3),
            server._resolve_tui_session("grok-4", msg_count=3),
        )
        # Both callers see the SAME post-boundary sid.
        assert results[0] == results[1]
        assert results[0] != first  # different from pre-boundary sid
        # And _close_session fired exactly ONCE on the pre-boundary sid.
        assert server._closed_sessions_log == [(first, "tui_boundary")]


# ---------------------------------------------------------------- #
# /v1/sessions/ingest boundary detection                           #
# ---------------------------------------------------------------- #


class TestIngestEndpointBoundary:
    """The /ingest endpoint is the CoPaw-side ingestion path — it
    receives session records directly instead of going through the LLM
    proxy.  Boundary detection on this path uses the same helper, with
    the body's ``session_id`` field stamped post-segmentation before
    the record gets written to disk."""

    async def test_growing_msg_count_keeps_raw_session_id(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import httpx

        monkeypatch.setattr(SkillClawAPIServer, "_load_tokenizer", lambda self: None)
        srv = SkillClawAPIServer(
            SkillClawConfig(
                record_enabled=True,
                record_dir=str(tmp_path / "records"),
            ),
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=srv.app),
            base_url="http://test",
        )

        async def _post(turn: int, msg_count: int) -> None:
            body = {
                "session_id": "console:joe",
                "turn": turn,
                "timestamp": "2026-05-17 00:00:00",
                "messages": [
                    {"role": "user", "content": f"msg-{i}"}
                    for i in range(msg_count)
                ],
            }
            resp = await client.post("/v1/sessions/ingest", json=body)
            assert resp.status_code == 200, resp.text

        try:
            await _post(turn=1, msg_count=2)
            await _post(turn=2, msg_count=4)
            await _post(turn=3, msg_count=6)
        finally:
            await client.aclose()

        log_path = tmp_path / "records" / "conversations.jsonl"
        import json
        lines = [json.loads(l) for l in log_path.read_text().splitlines()]
        assert [l["session_id"] for l in lines] == [
            "console:joe",
            "console:joe",
            "console:joe",
        ]

    async def test_ingest_msg_count_drop_segments_session(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import httpx

        monkeypatch.setattr(SkillClawAPIServer, "_load_tokenizer", lambda self: None)
        srv = SkillClawAPIServer(
            SkillClawConfig(
                record_enabled=True,
                record_dir=str(tmp_path / "records"),
            ),
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=srv.app),
            base_url="http://test",
        )

        async def _post(turn: int, msg_count: int) -> None:
            body = {
                "session_id": "console:joe",
                "turn": turn,
                "timestamp": "2026-05-17 00:00:00",
                "messages": [
                    {"role": "user", "content": f"msg-{i}"}
                    for i in range(msg_count)
                ],
            }
            resp = await client.post("/v1/sessions/ingest", json=body)
            assert resp.status_code == 200, resp.text

        try:
            await _post(turn=10, msg_count=30)
            await _post(turn=11, msg_count=1)  # drop → segment
        finally:
            await client.aclose()

        log_path = tmp_path / "records" / "conversations.jsonl"
        import json
        lines = [json.loads(l) for l in log_path.read_text().splitlines()]
        assert lines[0]["session_id"] == "console:joe"
        assert lines[1]["session_id"] == "console:joe:seg-2"
        # Both records keep their original turn numbers — segmentation
        # affects session_id only, not the within-session turn counter.
        assert lines[0]["turn"] == 10
        assert lines[1]["turn"] == 11

    async def test_ingest_shares_boundary_state_with_llm_proxy(
        self,
        server: SkillClawAPIServer,
    ) -> None:
        # Bucket key is raw_session_id only — meaning the /ingest path
        # and the LLM-proxy path SHARE boundary state for the same
        # raw_sid.  This is intentional: a user who is both proxying
        # LLM calls AND ingesting under "console:joe" gets one coherent
        # session lineage, not two parallel ones.  Confirmed by:
        # touching once via ingest path, then a follow-up via the
        # proxy path uses the same meta.
        first = await server._resolve_external_session("console:joe", msg_count=10)
        second = await server._resolve_external_session("console:joe", msg_count=12)
        assert first == second == "console:joe"
        # Single bucket, not two.
        assert list(server._external_session_meta.keys()) == ["console:joe"]
