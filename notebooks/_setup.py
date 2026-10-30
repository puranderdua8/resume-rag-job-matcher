"""Shared setup for the Step 6 experiment notebooks: path fix, one default ingest,
the loaded ground truth/specs, and the scoring helpers calibration and failure
analysis need.

Each notebooks/NN_*.ipynb does `import _setup as s` and reads off `s.<name>`
instead of repeating this boilerplate in all seven notebooks.
"""

from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

# nbconvert runs the kernel with the notebook's own directory as cwd, not the repo
# root, so the project's top-level modules (config.py and friends) aren't importable
# without this: walk up from cwd to the directory that has them.
_project_root = Path.cwd()
while not (_project_root / "config.py").exists() and _project_root != _project_root.parent:
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

warnings.filterwarnings("ignore")

import config  # noqa: E402
from eval_data import jd_id_for_path, load_ground_truth, load_resume_specs  # noqa: E402, F401
from eval_extraction import extraction_accuracy  # noqa: E402, F401
from evaluate import run_eval  # noqa: E402, F401
from jd_parsing import parse_jd  # noqa: E402, F401
from match_filtering import filter_candidates  # noqa: E402
from match_scoring import score_candidate  # noqa: E402
from resume_embedding import get_embedder  # noqa: E402
from resume_storage import collection_name, get_chroma_client, ingest, load_candidates  # noqa: E402
from retrieval import bm25_search, semantic_search  # noqa: E402

embedder = get_embedder(config.DEFAULT_MODEL)
ingest_stats = ingest(
    model=config.DEFAULT_MODEL, chunking=config.DEFAULT_CHUNKING, embedder=embedder
)
client = get_chroma_client()
collection = client.get_collection(collection_name(config.DEFAULT_MODEL, config.DEFAULT_CHUNKING))
candidates = load_candidates(config.DEFAULT_MODEL, config.DEFAULT_CHUNKING)
candidates_by_id = {c.candidate_id: c for c in candidates}
calibration = config.MODEL_REGISTRY[config.DEFAULT_MODEL]

specs = load_resume_specs()
specs_by_id = {spec["candidate_id"]: spec for spec in specs}
ground_truth = load_ground_truth()
jd_paths = sorted(config.JOB_DESCRIPTIONS_DIR.glob("*.txt"))


def score_all(jd, weights=None, soft=True):
    """Score every candidate against `jd` with the default (bge-small/section) index,
    the same formula `job_matcher.match_jd` uses, minus evidence/output."""
    jd_embedding = embedder.embed_query(jd.raw_text)
    hits = semantic_search(collection, jd_embedding, jd, config.SEMANTIC_POOL_SIZE)
    bm25_scores = bm25_search(candidates, jd.raw_text)
    bm25_max = max(bm25_scores.values(), default=0.0)
    filter_results = filter_candidates(jd, candidates, soft=soft)
    scored = {}
    for profile in candidates:
        cid = profile.candidate_id
        scored[cid] = score_candidate(
            profile,
            jd,
            semantic_raw=hits[cid].score if cid in hits else 0.0,
            bm25_raw=bm25_scores.get(cid, 0.0),
            bm25_max=bm25_max,
            calibration=calibration,
            filter_penalty=filter_results[cid].penalty,
            weights=weights,
        )
    return scored


def ranking_for(jd, weights=None, soft=True):
    scored = score_all(jd, weights=weights, soft=soft)
    return sorted(scored, key=lambda cid: (-scored[cid].score, candidates_by_id[cid].name))


@contextmanager
def temp_rrf_k(k):
    original = config.RRF_K
    config.RRF_K = k
    try:
        yield
    finally:
        config.RRF_K = original
