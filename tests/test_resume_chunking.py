"""Experience-entry splitting and long-chunk splitting (bullet glyphs, hard-split, R15/R17)."""

from __future__ import annotations

import logging

import config
from resume_chunking import _split_long_chunk, build_chunks, split_experience_entries


def test_date_only_line_pulls_in_the_preceding_title_line():
    lines = ["Software Engineer", "Jan 2020 - Present", "- Did work"]
    entries = split_experience_entries(lines)
    assert entries == [["Software Engineer", "Jan 2020 - Present", "- Did work"]]


def test_title_and_date_sharing_a_line_does_not_pull_in_the_previous_bullet():
    """R15: when the date line also carries the title ("Eng, A | Jan 2020 - Present"),
    the previous job's last bullet must not be pulled into this entry."""
    lines = [
        "Engineer, Acme",
        "Jan 2018 - Dec 2019",
        "- Did old work",
        "Eng, Beta | Jan 2020 - Present",
        "- Did new work",
    ]
    entries = split_experience_entries(lines)
    assert entries == [
        ["Engineer, Acme", "Jan 2018 - Dec 2019", "- Did old work"],
        ["Eng, Beta | Jan 2020 - Present", "- Did new work"],
    ]


def test_splits_on_any_bullet_glyph_not_just_dash():
    text = "\n".join(["• " + ("word " * 60).strip() for _ in range(4)])
    pieces = _split_long_chunk(text, max_chars=200)
    assert len(pieces) > 1


def test_hard_splits_a_single_bullet_still_over_max_chars():
    text = "- " + "a" * 5000
    pieces = _split_long_chunk(text, max_chars=1000)
    assert all(len(piece) <= 1000 for piece in pieces)
    assert "".join(p.removeprefix("- ") for p in pieces) == text.removeprefix("- ")


def test_build_chunks_warns_when_embed_text_exceeds_the_model_budget(caplog):
    full_text = "x" * (config.EMBEDDING_MAX_SEQ_CHARS + 500)
    with caplog.at_level(logging.WARNING):
        build_chunks("cand", "Name", {}, full_text, strategy="whole")
    assert any("embed text is" in record.message for record in caplog.records)
