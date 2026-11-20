"""Hybrid retrieval: semantic search over ChromaDB, BM25 keyword search, and RRF fusion.

Both retrieval paths run over the full candidate pool independently and are
then merged with Reciprocal Rank Fusion, so a strong keyword match survives
even when its embedding scores lower. The semantic path applies a Chroma
prefilter covering the must-haves Chroma can express (skill presence, total
years, degree level); a candidate it excludes fails the exact same condition
in match_filtering.py's post-filter, so excluding it here costs no recall
that the post-filter wouldn't have removed anyway.
"""

from __future__ import annotations

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


def semantic_search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    jd: JDRequirements,
    pool_size: int,
    apply_prefilter: bool = True,
) -> dict[str, SemanticHit]:
    """Query Chroma for the `pool_size` nearest chunks and group them by candidate.

    A candidate's score is `0.7*max + 0.3*mean(top-3)` cosine similarity
    (similarity = 1 - cosine distance, since chunks are stored in cosine space).

    `apply_prefilter=False` (soft filtering) skips the Chroma `where` filter: a
    soft-filtered candidate that fails a must-have should still get a real semantic
    score plus the soft penalty, not a semantic score of 0 (R1).
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


def bm25_search(candidates: list[CandidateProfile], query_text: str) -> dict[str, float]:
    """BM25 score per candidate over their full resume text."""
    corpus_tokens = [_tokenize(candidate.full_text) for candidate in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(_tokenize(query_text))
    return {
        candidate.candidate_id: float(score)
        for candidate, score in zip(candidates, scores, strict=True)
    }


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
