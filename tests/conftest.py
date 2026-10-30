"""Shared pytest fixtures.

`fake_embedder` duck-types `resume_embedding.Embedder` (`embed_documents`/`embed_query`)
with a deterministic hash-based vector, so ingest/match tests never download a model or
touch the network, per AGENTS.md's "no network in tests" rule.
"""

from __future__ import annotations

import hashlib
import math

import pytest

DIMENSIONS = 32


def _hash_vector(text: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for word in text.lower().split():
        digest = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[digest % DIMENSIONS] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hash_vector(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
