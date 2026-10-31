"""End to end: ingest 3 resumes into a temp Chroma dir with the fake embedder, then match
a JD against them, through the same `ingest`/`match_jd` functions the CLIs call.

Also covers "the filter runs before the top-K cut": with `k` far larger than the number
of candidates, the two candidates that fail the must-have must still land in
`filtered_out`, never in `top_matches` -- that has to be the filter's doing, not k's.
"""

from __future__ import annotations

from pathlib import Path

import config
from jd_parsing import parse_jd
from job_matcher import match_jd
from resume_storage import collection_name, get_chroma_client, ingest

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


def test_ingest_and_match_end_to_end(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "index" / "chroma")
    resumes_dir = _write_resumes(tmp_path / "resumes")

    stats = ingest(embedder=fake_embedder, resumes_dir=resumes_dir)
    assert stats.resumes_loaded == 3
    assert stats.resumes_skipped == []

    client = get_chroma_client()
    name = collection_name(config.DEFAULT_MODEL, config.DEFAULT_CHUNKING)
    collection = client.get_collection(name)

    jd = parse_jd(JD_TEXT)
    # k is far larger than the candidate pool, so "only Jordan Lee makes it through"
    # has to come from the must-have filter, not from the top-K cut.
    result = match_jd(jd, fake_embedder, collection, k=100)

    assert result["job_description"] == "Senior Python Engineer"

    top_names = {m["candidate_name"] for m in result["top_matches"]}
    filtered_names = {f["candidate_name"] for f in result["filtered_out"]}
    assert top_names == {"Jordan Lee"}
    assert filtered_names == {"Priya Shah", "Alex Kim"}

    required_keys = {
        "candidate_name",
        "resume_path",
        "match_score",
        "matched_skills",
        "relevant_excerpts",
        "reasoning",
    }
    for match in result["top_matches"]:
        assert required_keys <= match.keys()
        assert 0 <= match["match_score"] <= 100


def test_excerpts_survive_a_different_working_directory(tmp_path, monkeypatch, fake_embedder):
    """R8: match_evidence reads resumes via M1's `search_in_file`, which resolves a
    relative path against the process's current working directory. Excerpts must not
    depend on job_matcher.py being run from the repo root."""
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "index" / "chroma")
    resumes_dir = _write_resumes(tmp_path / "resumes")
    ingest(embedder=fake_embedder, resumes_dir=resumes_dir)

    client = get_chroma_client()
    name = collection_name(config.DEFAULT_MODEL, config.DEFAULT_CHUNKING)
    collection = client.get_collection(name)
    jd = parse_jd(JD_TEXT)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = match_jd(jd, fake_embedder, collection, k=100)
    jordan = next(m for m in result["top_matches"] if m["candidate_name"] == "Jordan Lee")
    assert jordan["relevant_excerpts"]
