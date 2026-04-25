"""Embedding-based near-duplicate detection across the skill catalog.

Closes the one quirk ``execute_merge`` doesn't address: it merges
**name collisions** only.  Two skills with different names but very
similar bodies stay in the catalog forever — over time evolve_server
keeps creating slight variants ("transcribe", "audio_to_text",
"voice_recognition") that the agent then has to disambiguate at
inject time, eating the prompt budget.

This module embeds each skill's description plus body, computes
pairwise cosine similarity, and surfaces pairs over a threshold so
the operator (or, eventually, the dashboard) can choose to delete
one of them.

Embedding service is configurable but defaults to a local
``bge-m3`` instance on ``http://localhost:9876/v1`` — same one
``copaw`` uses for memory search.  Anything OpenAI-compat works.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import httpx


DEFAULT_EMBED_URL = "http://localhost:9876/v1"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_THRESHOLD = 0.85


@dataclass(frozen=True)
class DuplicatePair:
    """A pair of skills whose embeddings are at or above the threshold."""

    skill_a: str
    skill_b: str
    similarity: float

    def __lt__(self, other: "DuplicatePair") -> bool:
        # Sort highest similarity first, then by name for stability.
        return (-self.similarity, self.skill_a, self.skill_b) < (
            -other.similarity, other.skill_a, other.skill_b,
        )


EmbedFn = Callable[[str], list[float]]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1].  Returns 0 for either vector empty
    or zero-magnitude — both shapes that would otherwise raise."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def gather_skill_corpus(skills_dir: str | Path) -> dict[str, str]:
    """Load every ``<skills_dir>/<name>/SKILL.md`` into a dict
    keyed by skill name.

    Skills with missing or unreadable SKILL.md are skipped — those
    are usually half-deleted artefacts or in-flight writes; surfacing
    them here would just create noise in the duplicate report.
    """
    root = Path(skills_dir)
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            text = skill_md.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            out[entry.name] = text
    return out


def make_http_embed_fn(
    *, base_url: str = DEFAULT_EMBED_URL,
    model: str = DEFAULT_EMBED_MODEL,
    api_key: str = "",
    timeout: float = 30.0,
) -> EmbedFn:
    """Return an ``EmbedFn`` that calls a remote OpenAI-compat
    ``/embeddings`` endpoint.  Bge-m3 on a local infinity server
    works without an API key; remote OpenAI / together.ai etc.
    require one."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/embeddings"

    def _embed(text: str) -> list[float]:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url,
                headers=headers,
                json={"input": text, "model": model},
            )
            response.raise_for_status()
            payload = response.json()
        data = payload.get("data") or []
        if not data:
            return []
        first = data[0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list):
            return [float(x) for x in first["embedding"]]
        return []

    return _embed


def find_near_duplicates(
    corpus: dict[str, str],
    *,
    embed_fn: EmbedFn,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[DuplicatePair]:
    """Embed each entry in *corpus* once and return all pairs whose
    cosine similarity is ``>= threshold``, sorted highest-first."""
    if threshold > 1.0 or threshold < -1.0:
        raise ValueError(f"threshold must be in [-1.0, 1.0], got {threshold}")

    names = sorted(corpus.keys())
    embeddings: dict[str, list[float]] = {}
    for name in names:
        text = corpus.get(name) or ""
        if not text:
            continue
        try:
            vec = embed_fn(text)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if vec:
            embeddings[name] = vec

    pairs: list[DuplicatePair] = []
    embedded_names = sorted(embeddings.keys())
    for i, name_a in enumerate(embedded_names):
        vec_a = embeddings[name_a]
        for name_b in embedded_names[i + 1:]:
            sim = cosine(vec_a, embeddings[name_b])
            if sim >= threshold:
                pairs.append(DuplicatePair(name_a, name_b, round(sim, 4)))

    pairs.sort()
    return pairs


def render_report(pairs: Iterable[DuplicatePair]) -> str:
    """Format duplicate pairs as a human-readable text block.  Empty
    string when the list is empty so the caller can short-circuit."""
    rows = list(pairs)
    if not rows:
        return ""
    lines = [f"{p.similarity:.4f}  {p.skill_a}  ⟷  {p.skill_b}" for p in rows]
    return "\n".join(lines)
