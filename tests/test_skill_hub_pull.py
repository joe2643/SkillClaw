# -*- coding: utf-8 -*-
"""Tests for SkillHub pull behaviour: mirror vs. incremental + unified storage.

Background
----------
Session-end auto-pull used to run with ``mirror=True`` and would delete any
local skill not present in the remote manifest. This wiped freshly-created
CoPaw skills that hadn't been pushed yet. We switched the auto-pull to
``mirror=False`` (incremental) so locally-authored skills are never auto-deleted.

These tests exercise ``SkillHub.pull_skills`` directly against a local
object store backend to regression-guard both modes.
"""

# pylint: disable=protected-access
import hashlib
import os

import pytest

from skillclaw.skill_hub import SkillHub


SKILL_A_BODY = b"---\nname: alpha\ndescription: A\ncategory: general\n---\nalpha body\n"
SKILL_B_BODY = b"---\nname: beta\ndescription: B\ncategory: general\n---\nbeta body\n"
SKILL_B_UPDATED = b"---\nname: beta\ndescription: B2\ncategory: general\n---\nbeta updated\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_skill(skills_dir: str, name: str, body: bytes) -> None:
    os.makedirs(os.path.join(skills_dir, name), exist_ok=True)
    with open(os.path.join(skills_dir, name, "SKILL.md"), "wb") as f:
        f.write(body)


def _build_hub(store_root: str, group_id: str = "g") -> SkillHub:
    return SkillHub(
        backend="local",
        endpoint="",
        bucket="",
        access_key_id="",
        secret_access_key="",
        local_root=store_root,
        group_id=group_id,
    )


def _seed_remote(hub: SkillHub, skills: dict[str, bytes]) -> None:
    """Upload skill bodies via the bucket and write a matching manifest."""
    manifest = {}
    for name, body in skills.items():
        hub._bucket.put_object(hub._skill_key(name), body)
        manifest[name] = {"name": name, "sha256": _sha256(body)}
    hub._save_remote_manifest(manifest)


@pytest.fixture
def layout(tmp_path):
    """Independent remote store + skills dir."""
    store = tmp_path / "store"
    skills = tmp_path / "skills"
    store.mkdir()
    skills.mkdir()
    return {
        "store": str(store),
        "skills": str(skills),
        "hub": _build_hub(str(store)),
    }


@pytest.fixture
def unified_layout(tmp_path):
    """local_root/group_id/skills == skills_dir (the production config)."""
    store = tmp_path / "workspaces"
    group_id = "default"
    skills = store / group_id / "skills"
    store.mkdir()
    skills.mkdir(parents=True)
    return {
        "store": str(store),
        "group_id": group_id,
        "skills": str(skills),
        "hub": _build_hub(str(store), group_id=group_id),
    }


# ---------------------------------------------------------------------------
# mirror=False (incremental, auto-pull path)
# ---------------------------------------------------------------------------


def test_incremental_pull_never_deletes_local_extras(layout):
    """Regression: locally-authored CoPaw skills must survive session-end pulls."""
    _seed_remote(layout["hub"], {"alpha": SKILL_A_BODY})
    _write_skill(layout["skills"], "local_only", SKILL_A_BODY)

    result = layout["hub"].pull_skills(layout["skills"], mirror=False)

    assert result["deleted"] == 0
    assert os.path.exists(
        os.path.join(layout["skills"], "local_only", "SKILL.md")
    ), "incremental pull must not delete locally-authored skills"
    assert os.path.exists(os.path.join(layout["skills"], "alpha", "SKILL.md"))


def test_incremental_pull_downloads_missing_remote(layout):
    _seed_remote(layout["hub"], {"alpha": SKILL_A_BODY, "beta": SKILL_B_BODY})

    result = layout["hub"].pull_skills(layout["skills"], mirror=False)

    assert result["downloaded"] == 2
    assert result["skipped"] == 0
    assert result["deleted"] == 0


def test_incremental_pull_skips_unchanged_skills(layout):
    _seed_remote(layout["hub"], {"alpha": SKILL_A_BODY})
    _write_skill(layout["skills"], "alpha", SKILL_A_BODY)

    result = layout["hub"].pull_skills(layout["skills"], mirror=False)

    assert result["downloaded"] == 0
    assert result["skipped"] == 1


def test_incremental_pull_updates_when_remote_sha_differs(layout):
    _write_skill(layout["skills"], "beta", SKILL_B_BODY)
    _seed_remote(layout["hub"], {"beta": SKILL_B_UPDATED})

    result = layout["hub"].pull_skills(layout["skills"], mirror=False)

    assert result["downloaded"] == 1
    with open(os.path.join(layout["skills"], "beta", "SKILL.md"), "rb") as f:
        assert f.read() == SKILL_B_UPDATED


# ---------------------------------------------------------------------------
# mirror=True (manual CLI path) — delete-semantics preserved
# ---------------------------------------------------------------------------


def test_mirror_pull_still_deletes_local_extras(layout):
    """Manual ``skillclaw skills pull`` keeps its explicit mirror-delete semantics."""
    _seed_remote(layout["hub"], {"alpha": SKILL_A_BODY})
    _write_skill(layout["skills"], "local_only", SKILL_B_BODY)

    result = layout["hub"].pull_skills(layout["skills"], mirror=True)

    assert result["deleted"] == 1
    assert not os.path.exists(os.path.join(layout["skills"], "local_only"))
    assert os.path.exists(os.path.join(layout["skills"], "alpha", "SKILL.md"))


def test_mirror_pull_empty_manifest_is_noop(layout):
    """Empty manifest is treated as no-op to avoid accidental wipe."""
    _write_skill(layout["skills"], "local_only", SKILL_A_BODY)

    result = layout["hub"].pull_skills(layout["skills"], mirror=True)

    assert result["downloaded"] == 0
    assert result["deleted"] == 0
    assert os.path.exists(os.path.join(layout["skills"], "local_only"))


# ---------------------------------------------------------------------------
# Unified storage: local_root/group_id/skills == skills_dir
# ---------------------------------------------------------------------------


def test_unified_storage_incremental_pull_is_idempotent(unified_layout):
    """When the remote pool IS the local skills dir, pull should be a no-op."""
    hub = unified_layout["hub"]
    skills = unified_layout["skills"]

    # Write a skill directly to the skills dir; register in manifest so
    # the unified pool has a matching entry.
    _write_skill(skills, "alpha", SKILL_A_BODY)
    hub._save_remote_manifest({
        "alpha": {"name": "alpha", "sha256": _sha256(SKILL_A_BODY)},
    })

    result = hub.pull_skills(skills, mirror=False)

    assert result["downloaded"] == 0
    assert result["skipped"] == 1
    assert result["deleted"] == 0
    # File untouched
    with open(os.path.join(skills, "alpha", "SKILL.md"), "rb") as f:
        assert f.read() == SKILL_A_BODY


def test_unified_storage_new_copaw_skill_survives_auto_pull(unified_layout):
    """End-to-end: CoPaw drops a new skill, session-end pull must not touch it."""
    hub = unified_layout["hub"]
    skills = unified_layout["skills"]

    # Existing registered skill
    _write_skill(skills, "alpha", SKILL_A_BODY)
    hub._save_remote_manifest({
        "alpha": {"name": "alpha", "sha256": _sha256(SKILL_A_BODY)},
    })

    # CoPaw creates a new skill directly in the dir (no push, no manifest entry)
    _write_skill(skills, "new_copaw_skill", SKILL_B_BODY)

    result = hub.pull_skills(skills, mirror=False)

    assert result["deleted"] == 0
    assert os.path.exists(
        os.path.join(skills, "new_copaw_skill", "SKILL.md")
    ), "new CoPaw skill must survive the auto-pull"


def test_unified_storage_mirror_pull_still_wipes_unregistered(unified_layout):
    """Explicit mirror pull still prunes — users opt in to this via CLI."""
    hub = unified_layout["hub"]
    skills = unified_layout["skills"]

    _write_skill(skills, "alpha", SKILL_A_BODY)
    hub._save_remote_manifest({
        "alpha": {"name": "alpha", "sha256": _sha256(SKILL_A_BODY)},
    })
    _write_skill(skills, "unregistered", SKILL_B_BODY)

    result = hub.pull_skills(skills, mirror=True)

    assert result["deleted"] == 1
    assert not os.path.exists(os.path.join(skills, "unregistered"))
