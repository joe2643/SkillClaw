# -*- coding: utf-8 -*-
"""Integration tests for the finalize-time body splice on
``optimize_description`` candidates.

The pure helper is unit-tested in ``test_evolve_pending_aware.py``;
this file covers the wiring inside ``EvolveServer._finalize_validation_jobs``
that fetches the live SKILL.md from the bucket, splices the body, and
either publishes or rejects when the splice can't be performed.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evolve_server.core.config import EvolveServerConfig
from evolve_server.core.constants import DecisionAction
from evolve_server.engines.workflow import EvolveServer


GROUP = "test_group"


def _write(root: Path, key: str, content: bytes) -> None:
    path = root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_server(tmp_root: Path) -> EvolveServer:
    config = EvolveServerConfig(
        storage_backend="local",
        local_root=str(tmp_root),
        group_id=GROUP,
        publish_mode="validated",
        validation_required_results=1,
        validation_required_approvals=1,
        validation_min_mean_score=0.5,
        validation_max_rejections=1,
        history_path=str(tmp_root / "history.jsonl"),
    )
    return EvolveServer(config, mock=True, mock_root=str(tmp_root))


def _seed_job(
    tmp_root: Path,
    *,
    job_id: str,
    skill_name: str,
    candidate_content: str,
    extra_frontmatter: dict | None = None,
) -> None:
    candidate = {
        "name": skill_name,
        "description": "tightened description",
        "content": candidate_content,
    }
    if extra_frontmatter is not None:
        candidate["extra_frontmatter"] = extra_frontmatter
    job = {
        "job_id": job_id,
        "candidate_skill_name": skill_name,
        "candidate_skill": candidate,
        "proposed_action": DecisionAction.OPTIMIZE_DESC,
    }
    _write(
        tmp_root,
        f"{GROUP}/validation_jobs/{job_id}.json",
        json.dumps(job).encode("utf-8"),
    )
    result = {
        "validator_mode": "test",
        "decision": "accept",
        "accepted": True,
        "score": 1.0,
    }
    _write(
        tmp_root,
        f"{GROUP}/validation_results/{job_id}/r.json",
        json.dumps(result).encode("utf-8"),
    )


def _seed_live_skill(
    tmp_root: Path,
    *,
    skill_name: str,
    body: str,
    description: str = "old desc",
    extra_frontmatter: str = "",
) -> None:
    extra = ("\n" + extra_frontmatter) if extra_frontmatter else ""
    md = (
        "---\n"
        f"name: {skill_name}\n"
        f"description: {description}\n"
        "category: general"
        f"{extra}\n"
        "---\n\n"
        f"{body}\n"
    )
    _write(
        tmp_root,
        f"{GROUP}/skills/{skill_name}/SKILL.md",
        md.encode("utf-8"),
    )


def _read_published(tmp_root: Path, skill_name: str) -> str:
    return (
        tmp_root / GROUP / "skills" / skill_name / "SKILL.md"
    ).read_text(encoding="utf-8")


def _read_decision(tmp_root: Path, job_id: str) -> dict:
    return json.loads(
        (tmp_root / GROUP / "validation_decisions" / f"{job_id}.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.mark.asyncio
async def test_empty_body_candidate_splices_live_skill_body_at_finalize():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        live_body = "# Real Heading\n\nbody paragraph that must survive\n"
        _seed_live_skill(root, skill_name="demo", body=live_body)
        _seed_job(
            root,
            job_id="20260101000000-demo-001",
            skill_name="demo",
            candidate_content="",
        )

        server = _make_server(root)
        records, summary = await server._finalize_validation_jobs()

        assert summary["published"] == 1
        assert summary["rejected"] == 0
        assert any(
            r.get("action") == "published_after_validation"
            and r.get("skill_name") == "demo"
            for r in records
        )
        published = _read_published(root, "demo")
        assert "tightened description" in published
        assert "body paragraph that must survive" in published
        decision = _read_decision(root, "20260101000000-demo-001")
        assert decision["status"] == "published"


@pytest.mark.asyncio
async def test_empty_body_with_no_live_skill_rejects_instead_of_clobbering():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # No live skill seeded — fetch will return None.
        _seed_job(
            root,
            job_id="20260101000000-orphan-002",
            skill_name="orphan_skill",
            candidate_content="",
        )

        server = _make_server(root)
        records, summary = await server._finalize_validation_jobs()

        assert summary["published"] == 0
        assert summary["rejected"] == 1
        # No SKILL.md should have been written for the orphan.
        assert not (root / GROUP / "skills" / "orphan_skill" / "SKILL.md").exists()
        decision = _read_decision(root, "20260101000000-orphan-002")
        assert decision["status"] == "rejected"
        assert "empty body" in decision["reason"]


@pytest.mark.asyncio
async def test_nonempty_candidate_body_skips_splice_and_publishes_as_is():
    # Regression: the splice path must not run when the candidate already
    # has content — otherwise an `improve_skill`-shaped optimize candidate
    # (rare but allowed) would lose its LLM-authored body.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _seed_live_skill(root, skill_name="demo", body="OLD body\n")
        _seed_job(
            root,
            job_id="20260101000000-demo-003",
            skill_name="demo",
            candidate_content="NEW body from candidate\n",
        )

        server = _make_server(root)
        records, summary = await server._finalize_validation_jobs()

        assert summary["published"] == 1
        published = _read_published(root, "demo")
        assert "NEW body from candidate" in published
        assert "OLD body" not in published


@pytest.mark.asyncio
async def test_extra_frontmatter_preserved_during_finalize_splice():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _seed_live_skill(
            root,
            skill_name="demo",
            body="body\n",
            extra_frontmatter="metadata:\n  builtin_skill_version: '1.0'",
        )
        _seed_job(
            root,
            job_id="20260101000000-demo-004",
            skill_name="demo",
            candidate_content="",
        )

        server = _make_server(root)
        await server._finalize_validation_jobs()

        published = _read_published(root, "demo")
        # Whatever the YAML emitter does, the metadata field should
        # survive the splice — empty body splice was the only path that
        # could drop it.
        assert "builtin_skill_version" in published
