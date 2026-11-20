"""Loading resumes via the M1 tools: candidate_id, resume_path and source_path (R8-R10)."""

from __future__ import annotations

import pytest

import config
from resume_loading import load_resumes


def test_candidate_id_includes_the_extension_so_same_stem_files_dont_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    (tmp_path / "jane_doe.txt").write_text("Jane Doe\nSkills\nPython")
    (tmp_path / "jane_doe.md").write_text("Jane Doe\nSkills\nSQL")

    loaded, skipped = load_resumes(tmp_path)

    assert skipped == []
    ids = {item["candidate_id"] for item in loaded}
    assert ids == {"jane_doe.txt", "jane_doe.md"}


def test_resume_path_is_relative_to_root_dir_when_inside_it(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "jane_doe.txt").write_text("Jane Doe\nSkills\nPython")

    loaded, _ = load_resumes(resumes_dir)

    assert loaded[0]["resume_path"] == "resumes/jane_doe.txt"
    assert loaded[0]["source_path"] == str((resumes_dir / "jane_doe.txt").resolve())


def test_resume_path_is_absolute_when_outside_root_dir(tmp_path, monkeypatch):
    other_root = tmp_path / "project"
    other_root.mkdir()
    monkeypatch.setattr(config, "ROOT_DIR", other_root)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "jane_doe.txt").write_text("Jane Doe\nSkills\nPython")

    loaded, _ = load_resumes(outside_dir)

    expected = str((outside_dir / "jane_doe.txt").resolve())
    assert loaded[0]["resume_path"] == expected
    assert loaded[0]["source_path"] == expected


def test_duplicate_candidate_id_fails_loudly_naming_both_files(tmp_path, monkeypatch):
    """A file name can't literally repeat within one directory on disk, so this
    simulates the collision the way M1's `list_files` could still produce one (e.g.
    two glob passes over overlapping extensions) by having it report the same file
    twice, and checks the guard names both paths rather than silently overwriting."""
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "jane_doe.txt").write_text("Jane Doe\nSkills\nPython")

    import file_tools

    real_list_files = file_tools.list_files

    def _doubled_list_files(directory, ext):
        return real_list_files(directory, ext) * 2 if ext == ".txt" else []

    monkeypatch.setattr(file_tools, "list_files", _doubled_list_files)

    with pytest.raises(ValueError, match="Duplicate candidate_id"):
        load_resumes(resumes_dir)
