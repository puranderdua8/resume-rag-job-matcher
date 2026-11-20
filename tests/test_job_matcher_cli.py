"""CLI-level checks for job_matcher.py: --k validation, slug fallback, and a clear
error when the index is missing (R23)."""

from __future__ import annotations

import pytest

import config
from job_matcher import main, slugify


def test_slugify_falls_back_to_a_hash_when_nothing_alphanumeric_survives():
    assert slugify("!!!") != ""
    assert slugify("!!!").startswith("jd_")
    # Deterministic, so re-running the CLI on the same JD overwrites the same file.
    assert slugify("!!!") == slugify("!!!")


def test_slugify_keeps_the_readable_slug_when_one_exists():
    assert slugify("Senior ML Engineer") == "senior_ml_engineer"


def test_k_must_be_at_least_1(capsys):
    with pytest.raises(SystemExit):
        main(["--text", "Some JD", "--k", "0"])
    assert "--k must be >= 1" in capsys.readouterr().err


def test_missing_index_gives_a_clear_message_naming_the_ingest_command(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "index" / "chroma")

    with pytest.raises(SystemExit):
        main(["--text", "Some JD"])

    err = capsys.readouterr().err
    assert "No index found" in err
    assert "python resume_rag.py ingest" in err
