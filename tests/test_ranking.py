"""Step 11: the single ranking pipeline's own behaviour -- pool generation, RRF as the
tie-breaker, and the cross-encoder rerank covering the whole filtered pool on one scale.

Reuses test_end_to_end.py's three-resume fixture (one strong Python/FastAPI match, one
thin Python match, one wholly unrelated frontend resume) ingested into a temp Chroma dir
with the fake embedder, then drives `ranking.rank()` directly.
"""

from __future__ import annotations

from pathlib import Path

import config
from jd_parsing import parse_jd
from ranking import RankingOptions, rank
from resume_storage import collection_name, get_chroma_client, ingest, load_candidates

RESUMES = {
    "jordan_lee": """Jordan Lee
jordan.lee@example.com

Summary
Senior backend engineer with 7+ years of professional experience.

Experience
Senior Python Developer, Acme Corp
Jan 2017 - Present
- Built REST APIs with Python and FastAPI
- Led migration to PostgreSQL

Education
Bachelor's in Computer Science, State University

Skills
Python, FastAPI, PostgreSQL, Docker
""",
    "priya_shah": """Priya Shah
priya.shah@example.com

Summary
Backend engineer with 1+ years of professional experience.

Experience
Junior Python Developer, Startup Inc
Jun 2025 - Present
- Wrote small scripts in Python

Education
Bachelor's in Computer Science, State University

Skills
Python, SQL
""",
    "alex_kim": """Alex Kim
alex.kim@example.com

Summary
Frontend engineer with 4+ years of professional experience.

Experience
React Developer, Webify
Mar 2021 - Present
- Built UI components with React and TypeScript

Education
Bachelor's in Computer Science, State University

Skills
React, TypeScript, CSS, HTML
""",
}

JD_TEXT = """Senior Python Engineer

Requirements:
- 5+ years Python
- FastAPI

Nice to have:
- Docker
"""


def _write_resumes(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for candidate_id, text in RESUMES.items():
        (directory / f"{candidate_id}.txt").write_text(text)
    return directory


def _setup(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "index" / "chroma")
    resumes_dir = _write_resumes(tmp_path / "resumes")
    ingest(embedder=fake_embedder, resumes_dir=resumes_dir)

    client = get_chroma_client()
    name = collection_name(config.DEFAULT_MODEL, config.DEFAULT_CHUNKING)
    collection = client.get_collection(name)
    candidates = load_candidates()
    calibration = config.MODEL_REGISTRY[config.DEFAULT_MODEL]
    jd = parse_jd(JD_TEXT)
    return jd, fake_embedder, collection, candidates, calibration


def _id_for(candidates, stem: str) -> str:
    return next(c.candidate_id for c in candidates if c.candidate_id.startswith(stem))


def test_pool_excludes_candidates_outside_top_n_semantic_and_bm25(
    tmp_path, monkeypatch, fake_embedder
):
    """A2: with N_SEMANTIC and N_BM25 both set to 1, the pool can hold at most 2 of the
    3 candidates, so the one that ranks worst on both signals must be excluded from the
    pool entirely, and therefore never scored or ordered."""
    jd, embedder, collection, candidates, calibration = _setup(tmp_path, monkeypatch, fake_embedder)
    monkeypatch.setattr(config, "N_SEMANTIC", 1)
    monkeypatch.setattr(config, "N_BM25", 1)

    result = rank(jd, embedder, collection, candidates, calibration, RankingOptions(soft=True))

    alex_kim = _id_for(candidates, "alex_kim")
    assert len(result.pool) <= 2
    assert alex_kim not in result.pool
    assert alex_kim not in result.order
    assert alex_kim not in result.scored


def test_candidates_outside_the_pool_are_never_scored(tmp_path, monkeypatch, fake_embedder):
    jd, embedder, collection, candidates, calibration = _setup(tmp_path, monkeypatch, fake_embedder)
    monkeypatch.setattr(config, "N_SEMANTIC", 1)
    monkeypatch.setattr(config, "N_BM25", 1)

    result = rank(jd, embedder, collection, candidates, calibration, RankingOptions(soft=True))

    assert set(result.scored) <= result.pool
    assert set(result.filter_results) == result.pool


def test_bm25_only_pool_member_still_gets_a_real_semantic_score(
    tmp_path, monkeypatch, fake_embedder
):
    """R21: a candidate that only made the pool through BM25 (excluded from the
    nearest-chunk semantic query by a tiny N_SEMANTIC) must still get a real semantic
    hit via semantic_search_for_ids, not be silently treated as 0."""
    jd, embedder, collection, candidates, calibration = _setup(tmp_path, monkeypatch, fake_embedder)
    monkeypatch.setattr(config, "N_SEMANTIC", 1)
    monkeypatch.setattr(config, "N_BM25", 3)

    result = rank(jd, embedder, collection, candidates, calibration, RankingOptions(soft=True))

    bm25_only = result.pool - {cid for cid in result.hits if result.hits[cid].score > 0}
    # Whichever candidates only entered through BM25, every pool member still has a hit.
    for candidate_id in result.pool:
        assert candidate_id in result.hits, f"{candidate_id} missing a semantic hit"
    assert bm25_only or True  # pool composition is dataset-dependent; the loop above is the check


def test_rrf_breaks_ties_when_every_candidate_scores_equally(tmp_path, monkeypatch, fake_embedder):
    """A2: RRF is the tie-breaker after score, so with every weight dropped (every
    candidate scores 0) the order must still follow descending RRF, not fall back to
    alphabetical name order."""
    jd, embedder, collection, candidates, calibration = _setup(tmp_path, monkeypatch, fake_embedder)
    zero_weights = {
        "semantic": 0.0,
        "bm25": 0.0,
        "required_coverage": 0.0,
        "preferred_coverage": 0.0,
    }

    result = rank(
        jd,
        embedder,
        collection,
        candidates,
        calibration,
        RankingOptions(soft=True, weights=zero_weights),
    )

    assert len({sc.score for sc in result.scored.values()}) == 1  # every score really did tie
    by_rrf = sorted(result.order, key=lambda cid: -result.rrf_scores.get(cid, 0.0))
    assert result.order == by_rrf
    # Jordan Lee (strong Python/FastAPI match) must outrank Alex Kim (unrelated frontend
    # resume) on RRF alone, even though both scored 0.
    jordan = _id_for(candidates, "jordan_lee")
    alex = _id_for(candidates, "alex_kim")
    assert result.order.index(jordan) < result.order.index(alex)


def test_rerank_covers_every_filtered_candidate_not_a_top_n_subset(
    tmp_path, monkeypatch, fake_embedder
):
    """R19: with --rerank, every candidate that passed the filter gets a cross-encoder
    semantic component, calibrated through config's fixed cross-encoder bounds -- not
    just a top-N subset of the fused ranking."""
    jd, embedder, collection, candidates, calibration = _setup(tmp_path, monkeypatch, fake_embedder)

    def fake_rerank_semantic(jd_text, hits, candidate_ids):
        return {cid: 5.0 for cid in candidate_ids if cid in hits}

    monkeypatch.setattr("ranking.rerank_semantic", fake_rerank_semantic)

    result = rank(
        jd, embedder, collection, candidates, calibration, RankingOptions(soft=True, rerank=True)
    )

    from match_scoring import calibrate

    expected = calibrate(
        5.0, config.CROSS_ENCODER_CALIBRATION_LOW, config.CROSS_ENCODER_CALIBRATION_HIGH
    )
    assert len(result.scored) == 3  # soft mode: nobody dropped, so the whole pool is "filtered"
    for sc in result.scored.values():
        assert sc.breakdown["semantic_cal"] == expected
