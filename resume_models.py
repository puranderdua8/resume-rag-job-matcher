"""Structured data for Part A: one chunk, one candidate's profile, one ingest run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    candidate_id: str
    section: str
    text: str
    embed_text: str


@dataclass
class CandidateProfile:
    candidate_id: str
    name: str
    resume_path: str
    source_path: str = ""
    email: str | None = None
    phone: str | None = None
    skills: list[str] = field(default_factory=list)
    skill_sections: dict[str, list[str]] = field(default_factory=dict)
    skill_years: dict[str, float] = field(default_factory=dict)
    experience_years: float = 0.0
    education_level: int = 0
    education_degree: str | None = None
    education_field: str | None = None
    education_institution: str | None = None
    sections: dict[str, str] = field(default_factory=dict)
    full_text: str = ""
    chunks: list[Chunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateProfile:
        chunk_dicts = data.pop("chunks", [])
        profile = cls(**data)
        profile.chunks = [Chunk(**c) for c in chunk_dicts]
        return profile


@dataclass
class IngestStats:
    model: str
    chunking: str
    collection: str
    resumes_loaded: int = 0
    resumes_skipped: list[str] = field(default_factory=list)
    chunk_counts: dict[str, int] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
