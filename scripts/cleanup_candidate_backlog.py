# -*- coding: utf-8 -*-
"""Reject superseded pending candidates so the queue keeps at most one
pending candidate per (skill_name, proposed_action) pair.

**Why this exists.**  The evolve_server's per-cycle decision loop has
no dedup — every cycle that re-evaluates a skill seen in recent
sessions can queue another candidate, even if a previous candidate
for the same skill is still pending.  Concretely: a single skill
``channel_message`` accumulated 11 ``optimize_description`` candidates
across two days, all proposing slight variants of the same
description tightening.  The validation worker at
``max_jobs_per_day=5`` can't keep up; the dashboard candidate pool
balloons.

**What this script does.**

For each ``(candidate_skill_name, proposed_action)`` pair, find the
**latest** pending candidate (job_id is timestamp-prefixed so a string
sort gives chronological order) and reject the older ones by writing
a ``validation_decisions/<job_id>.json`` marking them
``status=rejected`` with reason ``"superseded by newer pending
candidate"``.

**What this script does NOT do.**

* Delete ``validation_jobs/<job_id>.json`` or
  ``candidate_skills/<job_id>/`` — once a decision file exists, the
  validation worker stops picking the job up; keeping the artefacts
  preserves the audit trail and the dashboard's
  "rejected — superseded" view continues to render correctly.
* Touch already-decided jobs (those with a ``validation_decisions``
  entry).
* Pick across different ``proposed_action`` values — an
  ``optimize_description`` candidate for ``pdf`` doesn't supersede an
  ``improve_skill`` candidate for ``pdf``; they're independent.

Run from anywhere; uses ``WORKING_DIR/local-share/<group_id>/`` so it
respects whatever path the live ``MediaServer`` / shared store uses.

Idempotent: a second run is a no-op once the queue has at most one
pending per (skill, action) pair.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".skillclaw" / "local-share" / "qwenpaw"


def cleanup(root: Path, dry_run: bool) -> int:
    jobs_dir = root / "validation_jobs"
    decisions_dir = root / "validation_decisions"

    if not jobs_dir.is_dir():
        print(
            f"validation_jobs/ not found under {root}; nothing to clean up",
            file=sys.stderr,
        )
        return 1

    decisions_dir.mkdir(parents=True, exist_ok=True)

    # Gather pending jobs grouped by (skill_name, action).
    pending: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    total_jobs = 0
    decided = 0
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"WARN: skipping unreadable job {path.name}: {exc}",
                file=sys.stderr,
            )
            continue
        total_jobs += 1
        job_id = str(job.get("job_id", "") or "")
        if not job_id:
            continue
        if (decisions_dir / f"{job_id}.json").exists():
            decided += 1
            continue
        skill_name = str(job.get("candidate_skill_name", "") or "")
        action = str(job.get("proposed_action", "") or "")
        pending[(skill_name, action)].append((job_id, job))

    n_keys = len(pending)
    n_pending = sum(len(v) for v in pending.values())
    n_to_reject = sum(max(0, len(v) - 1) for v in pending.values())

    print(
        f"scanned {total_jobs} jobs: {decided} decided, "
        f"{n_pending} pending across {n_keys} (skill, action) groups; "
        f"{n_to_reject} to mark superseded",
    )

    if n_to_reject == 0:
        print("queue already clean — nothing to do")
        return 0

    rejected_per_skill: dict[str, int] = defaultdict(int)
    now_iso = datetime.now(timezone.utc).isoformat()
    for (skill, action), items in sorted(pending.items()):
        if len(items) <= 1:
            continue
        # Sort ascending by job_id; the timestamp prefix means the last
        # element is the newest.  Keep the newest, reject everything
        # before it.
        items.sort(key=lambda pair: pair[0])
        keep_id, _ = items[-1]
        for job_id, _job in items[:-1]:
            decision = {
                "job_id": job_id,
                "status": "rejected",
                "reason": (
                    f"superseded by newer pending candidate {keep_id} "
                    "(cleanup_candidate_backlog.py)"
                ),
                "result_count": 0,
                "accepted_count": 0,
                "rejected_count": 1,
                "mean_score": None,
                "decided_at": now_iso,
                "supersedes_keep": keep_id,
            }
            target = decisions_dir / f"{job_id}.json"
            if dry_run:
                print(f"  [dry-run] would reject {job_id} (skill={skill})")
            else:
                target.write_text(
                    json.dumps(decision, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            rejected_per_skill[skill] += 1

    print()
    print("Per-skill rejection counts:")
    for skill, count in sorted(
        rejected_per_skill.items(), key=lambda kv: -kv[1],
    ):
        print(f"  {skill:35s} -{count}")

    print()
    if dry_run:
        print(
            f"DRY RUN: would write {sum(rejected_per_skill.values())} "
            "rejection decisions; re-run without --dry-run to apply",
        )
    else:
        print(
            f"DONE: wrote {sum(rejected_per_skill.values())} "
            "rejection decisions",
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=(
            "Path to the SkillClaw shared root (defaults to "
            "~/.skillclaw/local-share/qwenpaw)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without writing decisions.",
    )
    args = parser.parse_args()
    return cleanup(Path(args.root).expanduser(), args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
