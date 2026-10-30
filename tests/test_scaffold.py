"""Smoke tests for the Step 1 scaffold: the M1 shim, config and skills taxonomy."""

from __future__ import annotations

import config
import m1
import skills


def test_m1_shim_lists_sample_resumes():
    result = m1.list_files("vendor/file-system-assistant/sample_resumes")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(item.get("success", True) for item in result)


def test_config_has_a_valid_default_model():
    assert config.DEFAULT_MODEL in config.MODEL_REGISTRY
    assert config.TOP_K > 0


def test_skills_taxonomy_is_nonempty():
    assert "Python" in skills.canonical_skills()
