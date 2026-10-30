"""Wraps a SentenceTransformer model with this project's query/document conventions."""

from __future__ import annotations

from typing import Any

import config


class Embedder:
    def __init__(self, model: Any, spec: config.ModelSpec):
        self.model = model
        self.spec = spec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            [text],
            prompt=self.spec.query_prompt,
            prompt_name=self.spec.query_prompt_name,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return embedding.tolist()


def get_embedder(model_name: str = config.DEFAULT_MODEL) -> Embedder:
    from sentence_transformers import SentenceTransformer

    spec = config.MODEL_REGISTRY[model_name]
    return Embedder(SentenceTransformer(spec.hf_id), spec)
