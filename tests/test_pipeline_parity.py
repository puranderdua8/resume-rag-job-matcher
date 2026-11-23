"""Step 11's own Checks (not the separate Step 11 verification step's full parity
matrix): `job_matcher` and `evaluate`/the notebooks must order candidates the same
way for the same JD and options (both call `ranking.rank()`), and a metrics run
must never touch the index `job_matcher.py` serves (A6).
"""

from __future__ import annotations

from pathlib import Path

import config
import evaluate
from jd_parsing import parse_jd
from job_matcher import match_jd
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


def test_job_matcher_and_ranking_agree_on_order_for_the_same_jd_and_options(
    tmp_path, monkeypatch, fake_embedder
):
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

    options = RankingOptions(mode="hybrid", soft=True, rerank=False)
    direct_order = rank(jd, fake_embedder, collection, candidates, calibration, options).order

    result = match_jd(jd, fake_embedder, collection, candidates, k=len(candidates), soft=True)
    candidates_by_name = {c.name: c.candidate_id for c in candidates}
    matched_order = [candidates_by_name[m["candidate_name"]] for m in result["top_matches"]]

    assert matched_order == direct_order


def test_evaluate_run_never_touches_the_serving_index(tmp_path, monkeypatch, fake_embedder):
    """A6: evaluate.py must ingest into config.EVAL_INDEX_DIR/EVAL_CHROMA_DIR, never
    the one job_matcher.py serves."""
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "index" / "chroma")
    monkeypatch.setattr(config, "EVAL_INDEX_DIR", tmp_path / "index_eval")
    monkeypatch.setattr(config, "EVAL_CHROMA_DIR", tmp_path / "index_eval" / "chroma")
    monkeypatch.setattr(config, "JOB_DESCRIPTIONS_DIR", tmp_path / "jds")
    monkeypatch.setattr(config, "GROUND_TRUTH_PATH", tmp_path / "ground_truth.json")
    monkeypatch.setattr(config, "RESUME_SPECS_PATH", tmp_path / "resume_specs.json")
    resumes_dir = _write_resumes(tmp_path / "resumes")
    ingest(embedder=fake_embedder, resumes_dir=resumes_dir)

    serving_chroma_dir = config.CHROMA_DIR
    mtime_before = serving_chroma_dir.stat().st_mtime

    (tmp_path / "jds").mkdir()
    (tmp_path / "jds" / "senior_python.txt").write_text(JD_TEXT)
    config.GROUND_TRUTH_PATH.write_text(
        '{"senior_python": {"jordan_lee.txt": 2, "priya_shah.txt": 1, "alex_kim.txt": 0}}'
    )
    config.RESUME_SPECS_PATH.write_text("[]")
    monkeypatch.setattr(evaluate, "get_embedder", lambda model: fake_embedder)

    evaluate.run_eval(warm_runs=0)

    assert serving_chroma_dir.exists()
    assert serving_chroma_dir.stat().st_mtime == mtime_before
    assert config.EVAL_CHROMA_DIR.exists()
    assert config.EVAL_CHROMA_DIR != serving_chroma_dir
