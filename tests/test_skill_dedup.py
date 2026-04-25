# -*- coding: utf-8 -*-
"""Unit tests for ``skillclaw.skill_dedup``.

The HTTP-backed embed fn is exercised separately at integration time
against a live bge-m3 instance.  Here we cover the algebra (cosine,
threshold gating, pair sorting) and the corpus-gathering walk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from skillclaw.skill_dedup import (
    DuplicatePair,
    cosine,
    find_near_duplicates,
    gather_skill_corpus,
    render_report,
)


class TestCosine:
    def test_identical_vectors_similarity_one(self):
        assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_similarity_zero(self):
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_similarity_negative_one(self):
        assert cosine([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)

    def test_empty_vector_returns_zero_not_raises(self):
        # A real failure mode — a missing or empty embedding from the
        # service must not poison the whole report.
        assert cosine([], [1.0, 2.0]) == 0.0
        assert cosine([1.0, 2.0], []) == 0.0

    def test_mismatched_dimensions_returns_zero(self):
        assert cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_zero_vector_returns_zero(self):
        # Pathological but possible if the embedding model yields a
        # null embedding for an empty-after-clean text.
        assert cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


class TestGatherSkillCorpus:
    def test_skips_dirs_without_skill_md(self, tmp_path: Path):
        (tmp_path / "alpha").mkdir()
        (tmp_path / "alpha" / "SKILL.md").write_text("alpha body")
        (tmp_path / "beta").mkdir()
        # beta has no SKILL.md — should be silently skipped
        (tmp_path / "gamma").mkdir()
        (tmp_path / "gamma" / "SKILL.md").write_text("gamma body")

        corpus = gather_skill_corpus(tmp_path)
        assert set(corpus) == {"alpha", "gamma"}
        assert corpus["alpha"] == "alpha body"

    def test_skips_empty_skill_md(self, tmp_path: Path):
        # An empty file represents an in-progress write — no signal,
        # would just embed to garbage and inflate the noise floor.
        (tmp_path / "stub").mkdir()
        (tmp_path / "stub" / "SKILL.md").write_text("   \n  \n")
        assert gather_skill_corpus(tmp_path) == {}

    def test_returns_empty_when_dir_missing(self, tmp_path: Path):
        assert gather_skill_corpus(tmp_path / "does-not-exist") == {}


class TestFindNearDuplicates:
    def test_returns_empty_when_no_pairs_meet_threshold(self):
        # Three orthogonal unit vectors — nothing should match.
        embeddings = {
            "alpha": [1.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0],
            "gamma": [0.0, 0.0, 1.0],
        }
        corpus = {n: f"body {n}" for n in embeddings}
        pairs = find_near_duplicates(
            corpus,
            embed_fn=lambda text: embeddings[text.split()[1]],
            threshold=0.5,
        )
        assert pairs == []

    def test_surfaces_pair_above_threshold(self):
        # alpha and beta near-identical, gamma orthogonal.
        embeddings = {
            "alpha": [1.0, 0.0],
            "beta": [0.99, 0.14],   # cos ≈ 0.99
            "gamma": [0.0, 1.0],
        }
        corpus = {n: f"body {n}" for n in embeddings}
        pairs = find_near_duplicates(
            corpus,
            embed_fn=lambda text: embeddings[text.split()[1]],
            threshold=0.85,
        )
        assert len(pairs) == 1
        pair = pairs[0]
        assert {pair.skill_a, pair.skill_b} == {"alpha", "beta"}
        assert pair.similarity >= 0.85

    def test_pairs_sorted_highest_similarity_first(self):
        embeddings = {
            "a": [1.0, 0.0, 0.0, 0.0],
            "b": [0.99, 0.05, 0.0, 0.0],   # very high
            "c": [0.92, 0.0, 0.39, 0.0],   # medium
            "d": [0.86, 0.0, 0.0, 0.51],   # just above threshold
        }
        corpus = {n: f"body {n}" for n in embeddings}
        pairs = find_near_duplicates(
            corpus,
            embed_fn=lambda text: embeddings[text.split()[1]],
            threshold=0.85,
        )
        sims = [p.similarity for p in pairs]
        assert sims == sorted(sims, reverse=True)

    def test_skipped_skill_when_embed_fn_raises(self):
        def flaky_embed(text: str) -> list[float]:
            if "broken" in text:
                raise RuntimeError("embedding service down")
            return [1.0, 0.0]

        corpus = {"good_a": "alpha", "good_b": "alpha", "bad": "broken"}
        pairs = find_near_duplicates(
            corpus, embed_fn=flaky_embed, threshold=0.5,
        )
        # 'bad' silently dropped; the two 'good' skills still pair.
        names_seen = {n for p in pairs for n in (p.skill_a, p.skill_b)}
        assert "bad" not in names_seen
        assert {"good_a", "good_b"} <= names_seen

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            find_near_duplicates({}, embed_fn=lambda t: [], threshold=1.5)

    def test_no_self_pair_when_corpus_has_one_entry(self):
        pairs = find_near_duplicates(
            {"only": "body"},
            embed_fn=lambda t: [1.0, 0.0],
            threshold=0.0,
        )
        assert pairs == []


class TestRenderReport:
    def test_empty_pairs_returns_empty_string(self):
        assert render_report([]) == ""

    def test_includes_similarity_and_both_names(self):
        pairs = [DuplicatePair("alpha", "beta", 0.9123)]
        out = render_report(pairs)
        assert "alpha" in out
        assert "beta" in out
        assert "0.9123" in out


class TestDuplicatePairOrdering:
    def test_higher_similarity_sorts_first(self):
        low = DuplicatePair("a", "b", 0.85)
        high = DuplicatePair("c", "d", 0.95)
        assert sorted([low, high]) == [high, low]

    def test_equal_similarity_falls_back_to_name(self):
        # Determinism matters — two equal-sim pairs must always sort
        # the same way so the CLI output is reproducible.
        p1 = DuplicatePair("alpha", "zulu", 0.9)
        p2 = DuplicatePair("alpha", "yankee", 0.9)
        assert sorted([p1, p2]) == [p2, p1]
