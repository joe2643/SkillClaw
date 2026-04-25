"""
Session Summarization: for each session, build a lossless structured
trajectory (programmatic) and a trajectory-aware analytical summary (LLM).

Attaches to each session dict:
- ``_trajectory``: structured text preserving the exact step-by-step path
- ``_summary``: LLM-generated analysis focusing on causal chains and insights
- ``_skills_referenced``, ``_avg_prm``, ``_has_tool_errors``: metadata
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

from ..core.llm_client import AsyncLLMClient
from ..core.utils import compact_tool_calls, compact_tool_observations

logger = logging.getLogger(__name__)

_SUMMARIZER_DEBUG_DIR = ""

# ------------------------------------------------------------------ #
#  Programmatic trajectory builder (zero information loss)             #
# ------------------------------------------------------------------ #

_PROMPT_MAX = 400
_RESPONSE_MAX = 400
_TOOL_ARG_MAX = 400
_TOOL_RESULT_MAX = 400
_TOOL_ERR_MAX = 300
_MAX_TOOLS_PER_STEP = 8


def _clip(text: Any, limit: int) -> str:
    s = str(text or "").strip().replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "…"


def _format_tool_calls(turn: dict) -> list[str]:
    """Render tool calls with their results/errors into compact lines."""
    raw_calls = turn.get("tool_calls") or []
    raw_results = turn.get("tool_results") or []
    raw_observations = turn.get("tool_observations") or []
    raw_errors = turn.get("tool_errors") or []

    result_by_id: dict[str, dict] = {}
    for r in raw_results:
        if isinstance(r, dict) and r.get("tool_call_id"):
            result_by_id[r["tool_call_id"]] = r
    for o in raw_observations:
        if isinstance(o, dict) and o.get("tool_call_id"):
            result_by_id.setdefault(o["tool_call_id"], o)

    error_by_tool: dict[str, list[str]] = {}
    for e in raw_errors:
        if isinstance(e, dict):
            tname = str(e.get("tool_name") or "")
            content = _clip(e.get("content", ""), _TOOL_ERR_MAX)
            error_by_tool.setdefault(tname, []).append(content)

    lines: list[str] = []
    for tc in raw_calls[:_MAX_TOOLS_PER_STEP]:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = str(func.get("name") or "unknown")
        args = _clip(func.get("arguments", ""), _TOOL_ARG_MAX)
        call_id = str(tc.get("id") or "")

        outcome = ""
        r = result_by_id.get(call_id)
        if r:
            if r.get("has_error"):
                err_type = r.get("error_type", "")
                err_content = _clip(r.get("content", ""), _TOOL_RESULT_MAX)
                outcome = f" → ✗ [{err_type}] {err_content}" if err_type else f" → ✗ {err_content}"
            else:
                content = _clip(r.get("content", ""), _TOOL_RESULT_MAX)
                cmd = _clip(r.get("command", ""), 80)
                if cmd:
                    outcome = f" → ✓ cmd={cmd}"
                    if content:
                        outcome += f" | {content}"
                elif content:
                    outcome = f" → ✓ {content}"
                else:
                    outcome = " → ✓"

        if not outcome and name in error_by_tool:
            errs = error_by_tool[name]
            outcome = f" → ✗ {errs[0]}"

        lines.append(f"    {name}({args}){outcome}")

    leftover_errors = []
    called_names = {
        (tc.get("function") or {}).get("name", "") for tc in raw_calls[:_MAX_TOOLS_PER_STEP] if isinstance(tc, dict)
    }
    for tname, errs in error_by_tool.items():
        if tname not in called_names:
            for e in errs:
                leftover_errors.append(f"    ⚠ {tname}: {e}")
    lines.extend(leftover_errors[:3])

    if len(raw_calls) > _MAX_TOOLS_PER_STEP:
        lines.append(f"    ... +{len(raw_calls) - _MAX_TOOLS_PER_STEP} more tool calls")

    return lines


def build_session_trajectory(session: dict) -> str:
    """Build a structured trajectory preserving the step-by-step path.

    If the session contains aggregated rollouts (turns carry ``_rollout_idx``),
    the trajectory is organised per-rollout with a header showing ORM score
    and success flag.  The user prompt is shown once at the top and omitted
    from subsequent steps to avoid redundancy.

    Each step shows: skills used, tool calls with outcomes, agent response
    snippet, and PRM / ORM score where available.
    """
    turns = session.get("turns", [])
    if not turns:
        return "(empty session)"

    # ---- detect rollout structure ------------------------------------ #
    has_rollouts = any(t.get("_rollout_idx") is not None for t in turns)

    # Deduplicate user prompt: show once, then omit repeats
    first_prompt = _clip((turns[0].get("prompt_text") or ""), _PROMPT_MAX)

    if has_rollouts:
        return _build_rollout_trajectory(turns, first_prompt)
    return _build_flat_trajectory(turns, first_prompt)


def _build_flat_trajectory(turns: list[dict], first_prompt: str) -> str:
    """Single-rollout (or non-aggregated) trajectory."""
    blocks: list[str] = []
    for i, t in enumerate(turns, 1):
        blocks.append(_format_step(t, i, first_prompt, show_prompt=(i == 1)))
    return "\n".join(blocks)


def _build_rollout_trajectory(turns: list[dict], first_prompt: str) -> str:
    """Multi-rollout aggregated trajectory with per-rollout headers."""
    # Group turns by _rollout_idx
    rollouts: dict[int, list[dict]] = {}
    for t in turns:
        idx = t.get("_rollout_idx", 0)
        rollouts.setdefault(idx, []).append(t)

    blocks: list[str] = []
    if first_prompt:
        blocks.append(f"Task: {first_prompt}")
        blocks.append("")

    for rollout_idx in sorted(rollouts.keys()):
        rollout_turns = rollouts[rollout_idx]
        # Extract ORM score from the rollout metadata on turns
        orm = rollout_turns[0].get("_rollout_score")
        success = rollout_turns[0].get("_rollout_success")
        orm_str = f"ORM={orm}" if orm is not None else "ORM=n/a"
        suc_str = f"success={success}" if success is not None else ""
        header_parts = [f"Rollout {rollout_idx}", orm_str]
        if suc_str:
            header_parts.append(suc_str)
        blocks.append(f"═══ {' | '.join(header_parts)} ═══")

        for step_i, t in enumerate(rollout_turns, 1):
            blocks.append(_format_step(t, step_i, first_prompt, show_prompt=False))
        blocks.append("")  # blank line between rollouts

    return "\n".join(blocks)


def _format_step(
    turn: dict,
    step_num: int,
    first_prompt: str,
    *,
    show_prompt: bool,
) -> str:
    """Format a single step line for the trajectory."""
    prompt = _clip(turn.get("prompt_text", ""), _PROMPT_MAX)
    response = _clip(turn.get("response_text", ""), _RESPONSE_MAX)

    skills = []
    for s in turn.get("read_skills") or []:
        name = s.get("skill_name", "").strip() if isinstance(s, dict) else str(s or "").strip()
        if name:
            skills.append(name)
    modified = []
    for s in turn.get("modified_skills") or []:
        name = s.get("skill_name", "").strip() if isinstance(s, dict) else str(s or "").strip()
        if name:
            modified.append(name)
    injected = [str(s or "").strip() for s in (turn.get("injected_skills") or []) if str(s or "").strip()]

    prm = turn.get("prm_score")
    prm_str = f"PRM={prm}" if prm is not None else ""

    header_parts = [f"[Step {step_num}]"]
    if prm_str:
        header_parts.append(prm_str)
    if skills:
        header_parts.append(f"read_skills={skills}")
    if modified:
        header_parts.append(f"modified_skills={modified}")
    if injected:
        header_parts.append(f"injected={injected}")
    header = " | ".join(header_parts)

    lines = [header]
    if show_prompt and prompt:
        lines.append(f"  User: {prompt}")

    tool_lines = _format_tool_calls(turn)
    if tool_lines:
        lines.append("  Tools:")
        lines.extend(tool_lines)

    if response:
        lines.append(f"  Agent: {response}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  LLM-based analytical summary (trajectory-aware)                     #
# ------------------------------------------------------------------ #

_SUMMARIZE_SESSION_SYSTEM = """\
You are a concise analyst for an AI coding assistant framework called SkillClaw.

Given a complete agent session, produce a trajectory-aware analytical summary \
(8-15 sentences) that captures:

1. **Goal**: The overall task the user wanted to accomplish.
2. **Key trajectory**: The step-by-step path the agent took — what it tried, \
in what order, and why (e.g., "read skill X → attempted approach Y → hit \
error Z → switched to W").
3. **Skill effectiveness**: For each skill that was read, injected, or \
modified, did it help or hurt? Was it relevant to the task? Was any guidance \
missing or wrong?
4. **Critical turning points**: Where things went right or wrong. What \
caused failures? What enabled successes?
5. **Tool usage patterns**: Which tools were used effectively, which caused \
errors, and any recurring patterns.
6. **Outcome**: Final result quality and what could have gone better.

Focus on preserving the SEQUENCE of events and CAUSAL RELATIONSHIPS. This \
summary will be used to decide whether skills need improvement, so be \
specific about what skill guidance helped, what was missing, and what was \
misleading.

Output ONLY the plain-text summary — no JSON, no markdown fences.
"""

_SUMMARY_PROMPT_MAX_CHARS = 2000
_SUMMARY_RESPONSE_MAX_CHARS = 2000


def _build_session_payload(session: dict) -> dict[str, Any]:
    """Build a compact representation of the session for the LLM.

    Deduplicates repeating user prompts — only the first occurrence is
    included; subsequent turns with the same prompt get ``prompt: "(same)"``.
    """
    turns = session.get("turns", [])
    first_prompt = (turns[0].get("prompt_text") or "")[:_SUMMARY_PROMPT_MAX_CHARS] if turns else ""
    interactions: list[dict[str, Any]] = []

    for idx, t in enumerate(turns):
        raw_prompt = (t.get("prompt_text") or "")[:_SUMMARY_PROMPT_MAX_CHARS]
        prompt = raw_prompt if idx == 0 else ("(same)" if raw_prompt == first_prompt else raw_prompt)

        interaction: dict[str, Any] = {
            "prompt": prompt,
            # Keep enough tail content for command-heavy sessions where the
            # decisive environment knowledge often appears near the end.
            "response": (t.get("response_text") or "")[:_SUMMARY_RESPONSE_MAX_CHARS],
            "prm_score": t.get("prm_score"),
        }
        # Carry rollout metadata so the summarizer can reason per-rollout
        ri = t.get("_rollout_idx")
        if ri is not None:
            interaction["rollout_idx"] = ri
            rs = t.get("_rollout_score")
            if rs is not None:
                interaction["rollout_score"] = rs
            rsu = t.get("_rollout_success")
            if rsu is not None:
                interaction["rollout_success"] = rsu

        read_skills = t.get("read_skills") or []
        if read_skills:
            interaction["read_skills"] = [
                s.get("skill_name", "") if isinstance(s, dict) else str(s or "") for s in read_skills
            ]
        modified_skills = t.get("modified_skills") or []
        if modified_skills:
            interaction["modified_skills"] = [
                s.get("skill_name", "") if isinstance(s, dict) else str(s or "") for s in modified_skills
            ]
        injected = t.get("injected_skills") or []
        if injected:
            interaction["injected_skills"] = injected
        tc = compact_tool_calls(t.get("tool_calls"), max_items=6)
        if tc:
            interaction["tool_calls"] = tc
        tr = compact_tool_observations(t.get("tool_results"), max_items=6)
        if tr:
            interaction["tool_results"] = tr
        to = compact_tool_observations(t.get("tool_observations"), max_items=4)
        if to:
            interaction["tool_observations"] = to
        te = t.get("tool_errors") or []
        if te:
            interaction["tool_errors"] = te
        interactions.append(interaction)

    payload: dict[str, Any] = {
        "session_id": session.get("session_id", ""),
        "total_interactions": len(turns),
        "interactions": interactions,
    }
    agg = session.get("aggregate")
    if agg:
        payload["aggregate"] = {
            "rollout_count": agg.get("rollout_count"),
            "scores": agg.get("scores"),
            "mean_score": agg.get("mean_score"),
            "success_count": agg.get("success_count"),
            "fail_count": agg.get("fail_count"),
            "stability": agg.get("stability"),
        }
    return payload


# ------------------------------------------------------------------ #
#  Metadata extraction                                                 #
# ------------------------------------------------------------------ #


def _extract_session_metadata(session: dict) -> None:
    """Extract skill references and compute aggregate metrics for a session.

    Attaches the following keys directly to the session dict:
    - ``_skills_referenced``: set of skill names the session referenced.
      Strong signal: explicit ``read_skills`` / ``modified_skills``.
      Weak signal (fallback only): ``injected_skills`` — used when the
      session has *zero* explicit read/modified entries.  This catches
      CoPaw-ingest sessions where the LLM happily answers from the
      prompt-injected skill content without ever calling
      ``read_file`` on the SKILL.md, which would otherwise leak every
      such session into the ``NO_SKILL_KEY`` bucket and starve
      ``evolve_skill_from_sessions``.
    - ``_skill_signal_strength``: ``"explicit"`` (read/modified) or
      ``"injected"`` (fallback) so downstream consumers can weight
      sessions accordingly.
    - ``_prm_scores``: list of all non-None PRM scores
    - ``_avg_prm``: mean PRM (or None if no scores)
    - ``_has_tool_errors``: True if any interaction had tool errors
    """
    explicit: set[str] = set()
    injected: set[str] = set()
    prm_scores: list[float] = []
    has_tool_errors = False

    for turn in session.get("turns", []):
        for item in turn.get("read_skills") or []:
            name = item.get("skill_name", "").strip() if isinstance(item, dict) else str(item or "").strip()
            if name:
                explicit.add(name)
        for item in turn.get("modified_skills") or []:
            name = item.get("skill_name", "").strip() if isinstance(item, dict) else str(item or "").strip()
            if name:
                explicit.add(name)
        for item in turn.get("injected_skills") or []:
            name = item.get("skill_name", "").strip() if isinstance(item, dict) else str(item or "").strip()
            if name:
                injected.add(name)
        prm = turn.get("prm_score")
        if prm is not None:
            prm_scores.append(prm)
        if turn.get("tool_errors"):
            has_tool_errors = True

    if explicit:
        session["_skills_referenced"] = explicit
        session["_skill_signal_strength"] = "explicit"
    else:
        session["_skills_referenced"] = injected
        session["_skill_signal_strength"] = "injected" if injected else "none"
    session["_prm_scores"] = prm_scores
    session["_avg_prm"] = round(sum(prm_scores) / len(prm_scores), 3) if prm_scores else None
    session["_has_tool_errors"] = has_tool_errors


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #


_PRM_JUDGE_SYSTEM = (
    "You are a quality reviewer for conversational responses.\n"
    "You will be shown a user instruction and the assistant response to that instruction.\n"
    "Based on instruction alignment and task completion quality, decide whether the response was "
    "helpful (+1), unhelpful (-1), or unclear (0).\n"
    "Do NOT compare against any follow-up turn.\n"
    "Only evaluate whether the response addresses the given instruction.\n"
    "Use +1 when the response clearly follows and substantially completes the instruction.\n"
    "Use -1 when the response is off-task, wrong, or fails to complete core requirements.\n"
    "Use 0 when completion is ambiguous or evidence is insufficient.\n"
    "Think briefly, then end your reply with exactly one of: Score: 1 / Score: -1 / Score: 0"
)
_PRM_SCORE_RE = re.compile(r"Score:\s*(-?[01])\b", re.IGNORECASE)


def _content_text(msg: Any) -> str:
    """Flatten a message ``content`` (str or list-of-blocks) to plain text."""
    if not isinstance(msg, dict):
        return ""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for blk in c:
            if isinstance(blk, dict):
                if blk.get("type") == "text":
                    parts.append(str(blk.get("text", "")))
                else:
                    parts.append(str(blk))
            else:
                parts.append(str(blk))
        return "".join(parts)
    return ""


def _last_role_msg(messages: list, role: str) -> Optional[dict]:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == role:
            return m
    return None


def _first_role_msg_after_index(messages: list, role: str, start: int) -> Optional[dict]:
    for m in (messages or [])[start:]:
        if isinstance(m, dict) and m.get("role") == role:
            return m
    return None


def _derive_instruction_response_pair(
    turn_record: dict,
    next_turn_record: Optional[dict],
) -> tuple[str, str]:
    """Recover (instruction, response) for a CoPaw-ingest turn.

    Each ingest record's ``messages`` is the snapshot fed INTO the LLM
    for that turn — i.e. system + history + the user's instruction.
    The LLM's response lands in the NEXT turn's ``messages`` as the
    first new ``assistant`` message after the instruction.
    """
    msgs = turn_record.get("messages") or []
    last_user = _last_role_msg(msgs, "user")
    instruction = _content_text(last_user) if last_user else ""

    response = ""
    if next_turn_record is not None:
        next_msgs = next_turn_record.get("messages") or []
        # Find the index in next_msgs where our instruction lives, then
        # look for the first assistant after it.  Falls back to the
        # last assistant in next_msgs if instruction can't be matched
        # (which can happen when memory compaction rewrites old turns).
        idx = -1
        if last_user is not None:
            for i, m in enumerate(next_msgs):
                if (
                    isinstance(m, dict)
                    and m.get("role") == "user"
                    and _content_text(m) == instruction
                ):
                    idx = i + 1
                    break
        candidate = (
            _first_role_msg_after_index(next_msgs, "assistant", idx)
            if idx >= 0
            else _last_role_msg(next_msgs, "assistant")
        )
        response = _content_text(candidate) if candidate else ""
    return instruction.strip(), response.strip()


def _attach_flat_turn_text(session: dict) -> None:
    """Backfill ``prompt_text`` / ``response_text`` on each turn from
    the OpenAI-shape ``messages`` block.

    The proxy era's ingest record carried these flat fields directly
    (see ``TOLERATED_EXTRAS`` in CoPaw's capture-schema contract test),
    but the in-process CoPaw capture hook only emits ``messages``.
    Several downstream sites then silently degrade:

    * ``evolve_server.engines.workflow._build_replay_cases`` — without
      ``prompt_text``/``response_text`` it produces an empty
      ``replay_cases`` list, which causes every queued validation job
      to fail with ``validation job missing replay_cases``.
    * The summarizer's own trajectory and judge prompts read flat
      fields too.

    Run this once per session before the metadata extractor and the
    lazy PRM pass — both then read consistent values.

    Idempotent: turns that already have both fields are skipped.
    """
    turns = session.get("turns") or []
    if not turns:
        return
    turns.sort(key=lambda t: int(t.get("turn_num") or 0))

    for i, turn in enumerate(turns):
        has_prompt = isinstance(turn.get("prompt_text"), str) and turn["prompt_text"]
        has_response = isinstance(turn.get("response_text"), str) and turn["response_text"]
        if has_prompt and has_response:
            continue
        next_turn = turns[i + 1] if i + 1 < len(turns) else None
        instruction, response = _derive_instruction_response_pair(
            turn, next_turn,
        )
        if not has_prompt and instruction:
            turn["prompt_text"] = instruction
        if not has_response and response:
            turn["response_text"] = response


async def _attach_lazy_prm_scores(
    llm: AsyncLLMClient, session: dict,
) -> None:
    """Best-effort PRM scoring for turns that arrived without one.

    Bridges the gap left by CoPaw's ``/v1/sessions/ingest`` path —
    the proxy used to compute ``prm_score`` per turn before flushing,
    but in-process CoPaw doesn't have a PRM scorer in the request
    path.  We fill the gap here, after sessions have been drained
    into the evolve queue.

    Idempotent: turns that already have a valid score are skipped.
    Best-effort: any failure (LLM error, malformed messages) leaves
    the score as ``None`` so downstream code falls back to its
    no-score branch.
    """
    turns = session.get("turns") or []
    if not turns:
        return
    # Sort by turn_num so neighbour lookups are stable.
    turns.sort(key=lambda t: int(t.get("turn_num") or 0))

    for i, turn in enumerate(turns):
        if turn.get("prm_score") is not None:
            continue
        next_turn = turns[i + 1] if i + 1 < len(turns) else None
        instruction, response = _derive_instruction_response_pair(
            turn, next_turn,
        )
        if not instruction or not response:
            continue
        msgs = [
            {"role": "system", "content": _PRM_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Instruction:\n{instruction}\n\n"
                    f"Response:\n{response}\n\n"
                    "Was the response helpful for this instruction? "
                    "End with Score: 1, Score: -1, or Score: 0."
                ),
            },
        ]
        try:
            raw = await llm.chat(msgs, max_tokens=256, temperature=0.6)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug(
                "[PRM-lazy] score skipped for turn %s of session %s: %s",
                turn.get("turn_num"), session.get("session_id"), e,
            )
            continue
        m = _PRM_SCORE_RE.findall(raw or "")
        if not m:
            continue
        try:
            v = int(m[-1])
        except ValueError:
            continue
        if v in (-1, 0, 1):
            turn["prm_score"] = float(v)


async def summarize_session(llm: AsyncLLMClient, session: dict) -> str:
    """Summarize an entire session via LLM (trajectory-aware)."""
    payload = _build_session_payload(session)
    messages = [
        {"role": "system", "content": _SUMMARIZE_SESSION_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        return await llm.chat(messages, max_tokens=100000, temperature=0.2)
    except Exception as e:
        logger.warning(
            "[Summarizer] LLM call failed for session %s: %s",
            session.get("session_id"),
            e,
        )
        return ""


def set_summarizer_debug_dir(path: str) -> None:
    """Set the debug dump directory used by summarization."""
    global _SUMMARIZER_DEBUG_DIR
    _SUMMARIZER_DEBUG_DIR = str(path or "").strip()


async def summarize_sessions_parallel(
    llm: AsyncLLMClient,
    sessions: list[dict],
) -> list[str]:
    """Preprocess and summarize all sessions in parallel.

    For each session:
    1. Lazily fill missing ``prm_score`` per turn (CoPaw-ingest path
       lacks the proxy's per-turn PRM, so we judge here using the same
       LLM the rest of the pipeline already shares).
    2. Extract metadata (``_skills_referenced``, ``_avg_prm``, etc.)
    3. Build programmatic ``_trajectory`` (lossless)
    4. Generate ``_summary`` via LLM (trajectory-aware analysis)

    Returns the list of summary strings (same order as *sessions*).
    """
    if not sessions:
        return []

    # Backfill flat ``prompt_text`` / ``response_text`` for turns
    # captured from the in-process CoPaw hook (which only emits
    # ``messages``).  Cheap and synchronous — must run before lazy
    # PRM (which currently re-derives the same pair) and before
    # ``_build_replay_cases`` reads them.
    for session in sessions:
        _attach_flat_turn_text(session)

    # Phase 2: backfill PRM scores for turns ingested without one.
    # Runs in parallel across sessions (each session's turns are
    # scored serially to avoid burst-saturating the upstream).
    await asyncio.gather(
        *[_attach_lazy_prm_scores(llm, s) for s in sessions],
        return_exceptions=True,
    )

    for session in sessions:
        _extract_session_metadata(session)
        session["_trajectory"] = build_session_trajectory(session)

    summaries = await asyncio.gather(
        *[summarize_session(llm, s) for s in sessions],
        return_exceptions=True,
    )

    result: list[str] = []
    for session, summary in zip(sessions, summaries):
        if isinstance(summary, BaseException):
            logger.warning(
                "[Summarizer] exception for session %s: %s",
                session.get("session_id"),
                summary,
            )
            summary = ""
        session["_summary"] = summary
        result.append(summary)

    # ---- debug dump -------------------------------------------------- #
    debug_dir = _SUMMARIZER_DEBUG_DIR
    if debug_dir:
        import pathlib

        ddir = pathlib.Path(debug_dir) / "summarizer"
        ddir.mkdir(parents=True, exist_ok=True)
        for session in sessions:
            sid = session.get("session_id", "unknown").replace("/", "_")
            (ddir / f"{sid}_trajectory.txt").write_text(
                session.get("_trajectory", ""),
                encoding="utf-8",
            )
            (ddir / f"{sid}_summary.txt").write_text(
                session.get("_summary", ""),
                encoding="utf-8",
            )
            meta = {
                "_skills_referenced": sorted(session.get("_skills_referenced") or []),
                "_avg_prm": session.get("_avg_prm"),
                "_has_tool_errors": session.get("_has_tool_errors"),
                "_prm_scores": session.get("_prm_scores"),
            }
            (ddir / f"{sid}_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        logger.info("[DebugDump] wrote summarizer artifacts to %s", ddir)
    # ------------------------------------------------------------------ #

    logger.info("[Summarizer] summarized %d sessions", len(result))
    return result
