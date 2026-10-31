"""Tunable configuration: paths, model registry, retrieval and scoring knobs.

Centralised so Steps 3-5 and the notebook (Step 6) tune the same numbers
instead of hard-coding them in multiple places.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JOB_DESCRIPTIONS_DIR = DATA_DIR / "job_descriptions"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
RESUME_SPECS_PATH = DATA_DIR / "resume_specs.json"

INDEX_DIR = ROOT_DIR / "index"
CHROMA_DIR = INDEX_DIR / "chroma"

OUTPUTS_DIR = ROOT_DIR / "outputs"
MATCHES_DIR = OUTPUTS_DIR / "matches"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"


@dataclass(frozen=True)
class ModelSpec:
    """One embedding model's HuggingFace id, query prompt and score calibration.

    `calibration_low`/`calibration_high` map a model's raw cosine similarity
    onto [0, 1] for scoring; they're per-model because similarity ranges
    differ across models. Starting values here are placeholders, tuned
    against the dataset in the Step 6 notebook.
    """

    hf_id: str
    dimensions: int
    query_prompt: str | None = None
    query_prompt_name: str | None = None
    calibration_low: float = 0.3
    calibration_high: float = 0.9


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "bge-small": ModelSpec(
        hf_id="BAAI/bge-small-en-v1.5",
        dimensions=384,
        query_prompt="Represent this sentence for searching relevant passages: ",
        calibration_low=0.35,
        calibration_high=0.85,
    ),
    "minilm": ModelSpec(
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        calibration_low=0.2,
        calibration_high=0.7,
    ),
    "qwen3-0.6b": ModelSpec(
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        dimensions=1024,
        query_prompt_name="query",
        calibration_low=0.4,
        calibration_high=0.9,
    ),
}

DEFAULT_MODEL = "bge-small"
DEFAULT_CHUNKING = "section"

# Retrieval
TOP_K = 10
SEMANTIC_POOL_SIZE = 50
RRF_K = 60

# Scoring: score = 100 * sum(weight * component), components in [0, 1].
SCORE_WEIGHTS = {
    "semantic": 0.50,
    "bm25": 0.20,
    "required_coverage": 0.25,
    "preferred_coverage": 0.05,
}
SOFT_FILTER_PENALTY = 20

# Chunking
MAX_CHUNK_CHARS = 1000
FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 100
# Conservative char-count proxy for the ~512-token max sequence length shared by every
# model in MODEL_REGISTRY, so resume_chunking can warn without needing a tokenizer on hand.
EMBEDDING_MAX_SEQ_CHARS = 2000

# Dataset generation (Step 2): fixed seed so the synthetic dataset is reproducible.
DATASET_SEED = 20260214
