"""Hybrid retrieval: semantic search over ChromaDB, BM25 keyword search, and RRF fusion.

`ranking.py` is the only caller: it takes the top-N_SEMANTIC and top-N_BM25
candidates by each ranking and unions them into one pool, which RRF then
orders (Step 11). The semantic path applies a Chroma prefilter covering the
must-haves Chroma can express (skill presence, total years, degree level); a
candidate it excludes fails the exact same condition in match_filtering.py's
post-filter, so excluding it here costs no recall that the post-filter
wouldn't have removed anyway.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import chromadb
from rank_bm25 import BM25Okapi

import skills
from match_models import JDRequirements
from resume_models import CandidateProfile

_MULTI_WORD_ALIASES = sorted(
    (
        (alias, canonical.lower().replace(" ", "_"))
        for canonical, aliases in skills.SKILL_ALIASES.items()
        for alias in aliases
        if not alias.isalnum()
    ),
    key=lambda pair: -len(pair[0]),
)
_SINGLE_WORD_ALIASES = {
    alias: canonical.lower().replace(" ", "_")
    for canonical, aliases in skills.SKILL_ALIASES.items()
    for alias in aliases
    if alias.isalnum()
}
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and fold skill aliases to their canonical form, so "k8s" and "kubernetes"
    both tokenize to "kubernetes" and can match each other under BM25."""
    normalized = text.lower()
    for alias, canonical_token in _MULTI_WORD_ALIASES:
        normalized = re.sub(rf"(?<!\w){re.escape(alias)}(?!\w)", canonical_token, normalized)
    tokens = _TOKEN_RE.findall(normalized)
    return [_SINGLE_WORD_ALIASES.get(token, token) for token in tokens]


def build_prefilter(jd: JDRequirements) -> dict | None:
    """Chroma `where` filter covering the must-haves array metadata can express."""
    conditions: list[dict] = [{"skills": {"$contains": skill}} for skill in jd.skill_year_minimums]
    for group_skills, _ in jd.skill_year_alternatives:
        conditions.append({"$or": [{"skills": {"$contains": skill}} for skill in group_skills]})
    if jd.total_years_minimum is not None:
        conditions.append({"experience_years": {"$gte": jd.total_years_minimum}})
    if jd.degree_minimum:
        conditions.append({"education_level": {"$gte": jd.degree_minimum}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


@dataclass
class SemanticHit:
    candidate_id: str
    score: float
    section_scores: dict[str, float] = field(default_factory=dict)
    top_chunks: list[tuple[float, str, str]] = field(default_factory=list)


def _aggregate_hits(
    per_candidate: dict[str, list[tuple[float, str, str]]],
) -> dict[str, SemanticHit]:
    """A candidate's score is `0.7*max + 0.3*mean(top-3)` cosine similarity across
    their chunk hits. Shared by `semantic_search` (nearest-chunk query) and
    `semantic_search_for_ids` (exact per-candidate lookup), so both aggregate the
    same way (R21)."""
    hits: dict[str, SemanticHit] = {}
    for candidate_id, chunk_hits in per_candidate.items():
        chunk_hits.sort(key=lambda hit: hit[0], reverse=True)
        sims = [sim for sim, _, _ in chunk_hits]
        top3 = sims[:3]
        score = 0.7 * sims[0] + 0.3 * (sum(top3) / len(top3))
        section_scores: dict[str, float] = {}
        for sim, section, _ in chunk_hits:
            section_scores[section] = max(section_scores.get(section, 0.0), sim)
        hits[candidate_id] = SemanticHit(
            candidate_id=candidate_id,
            score=score,
            section_scores=section_scores,
            top_chunks=chunk_hits[:2],
        )
    return hits


def semantic_search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    jd: JDRequirements,
    pool_size: int,
    apply_prefilter: bool = True,
) -> dict[str, SemanticHit]:
    """Query Chroma for the `pool_size` nearest chunks and group them by candidate
    (similarity = 1 - cosine distance, since chunks are stored in cosine space).

    `apply_prefilter=False` (soft filtering) skips the Chroma `where` filter: a
    soft-filtered candidate that fails a must-have should still get a real semantic
    score plus the soft penalty, not a semantic score of 0 (R1).

    Candidates outside the nearest `pool_size` chunks aren't included at all; use
    `semantic_search_for_ids` to get an exact score for a specific candidate (R21).
    """
    where = build_prefilter(jd) if apply_prefilter else None
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": pool_size,
        "include": ["metadatas", "documents", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    result = collection.query(**kwargs)

    per_candidate: dict[str, list[tuple[float, str, str]]] = {}
    metadatas, documents, distances = (
        result["metadatas"][0],
        result["documents"][0],
        result["distances"][0],
    )
    for metadata, document, distance in zip(metadatas, documents, distances, strict=True):
        similarity = 1.0 - distance
        per_candidate.setdefault(metadata["candidate_id"], []).append(
            (similarity, metadata["section"], document)
        )
    return _aggregate_hits(per_candidate)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_search_for_ids(
    collection: chromadb.Collection, query_embedding: list[float], candidate_ids: list[str]
) -> dict[str, SemanticHit]:
    """Exact semantic score for every one of `candidate_ids`, regardless of how they
    ranked in the nearest-chunk query: fetches every stored chunk for those
    candidates by id and computes cosine similarity against `query_embedding`
    directly, instead of relying on them appearing in a `pool_size`-limited
    `semantic_search` result. Used to give pool candidates that only made it in
    through BM25 a real semantic score instead of defaulting to 0 (R21)."""
    if not candidate_ids:
        return {}
    result = collection.get(
        where={"candidate_id": {"$in": candidate_ids}},
        include=["metadatas", "documents", "embeddings"],
    )
    per_candidate: dict[str, list[tuple[float, str, str]]] = {}
    for metadata, document, embedding in zip(
        result["metadatas"], result["documents"], result["embeddings"], strict=True
    ):
        similarity = _cosine_similarity(query_embedding, list(embedding))
        per_candidate.setdefault(metadata["candidate_id"], []).append(
            (similarity, metadata["section"], document)
        )
    return _aggregate_hits(per_candidate)


def bm25_search(
    candidates: list[CandidateProfile], query_text: str, reference_text: str
) -> tuple[dict[str, float], float]:
    """BM25 score per candidate over their full resume text, plus a fixed reference
    score (R20): `reference_text` (the JD's own condensed requirement text) is added
    to the corpus as a pseudo-document and scored against `query_text` the same way,
    so the reference reflects that JD's own vocabulary and corpus-wide IDF rather
    than whichever candidates happen to be in the pool."""
    corpus_tokens = [_tokenize(candidate.full_text) for candidate in candidates] + [
        _tokenize(reference_text)
    ]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(_tokenize(query_text))
    candidate_scores = {
        candidate.candidate_id: float(score)
        for candidate, score in zip(candidates, scores[:-1], strict=True)
    }
    return candidate_scores, float(scores[-1])


def skill_coverage(target_skills: list[str], profile: CandidateProfile) -> float:
    """Share of `target_skills` present on the candidate's profile."""
    if not target_skills:
        return 1.0
    have = set(profile.skills)
    return len(set(target_skills) & have) / len(target_skills)


def rrf_fuse(rankings: list[list[str]], k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion: sum 1/(k + rank) across each ranking a candidate appears in."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, candidate_id in enumerate(ranking, start=1):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + 1.0 / (k + rank)
    return scores
