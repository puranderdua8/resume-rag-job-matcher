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
from ranking import RankingOptions  # noqa: E402
from ranking import rank as rank_jd  # noqa: E402
from resume_embedding import get_embedder  # noqa: E402
from resume_storage import collection_name, get_chroma_client, ingest, load_candidates  # noqa: E402

# A6: a separate index directory, never the one job_matcher.py serves.
embedder = get_embedder(config.DEFAULT_MODEL)
ingest_stats = ingest(
    model=config.DEFAULT_MODEL,
    chunking=config.DEFAULT_CHUNKING,
    embedder=embedder,
    index_dir=config.EVAL_INDEX_DIR,
    chroma_dir=config.EVAL_CHROMA_DIR,
)
client = get_chroma_client(config.EVAL_CHROMA_DIR)
collection = client.get_collection(collection_name(config.DEFAULT_MODEL, config.DEFAULT_CHUNKING))
candidates = load_candidates(
    config.DEFAULT_MODEL, config.DEFAULT_CHUNKING, index_dir=config.EVAL_INDEX_DIR
)
candidates_by_id = {c.candidate_id: c for c in candidates}
calibration = config.MODEL_REGISTRY[config.DEFAULT_MODEL]

specs = load_resume_specs()
# R9 changed candidate_id to the file name *with* its extension; resume_specs.json's
# own candidate_id field predates that and is still the bare stem, so key by the
# file name derived from resume_path instead (same fix as evaluate.py's specs_by_id).
specs_by_id = {Path(spec["resume_path"]).name: spec for spec in specs}
ground_truth = load_ground_truth()
jd_paths = sorted(config.JOB_DESCRIPTIONS_DIR.glob("*.txt"))


def score_all(jd, weights=None, soft=True, mode="hybrid", rerank=False):
    """Score every candidate against `jd` through `ranking.rank()` (Step 11: the
    same pipeline `job_matcher.match_jd` and `evaluate.py` use), minus evidence
    and output formatting."""
    options = RankingOptions(mode=mode, soft=soft, rerank=rerank, weights=weights)
    result = rank_jd(jd, embedder, collection, candidates, calibration, options)
    return result.scored


def ranking_for(jd, weights=None, soft=True, mode="hybrid", rerank=False):
    scored = score_all(jd, weights=weights, soft=soft, mode=mode, rerank=rerank)
    return sorted(scored, key=lambda cid: (-scored[cid].score, candidates_by_id[cid].name))


@contextmanager
def temp_rrf_k(k):
    original = config.RRF_K
    config.RRF_K = k
    try:
        yield
    finally:
        config.RRF_K = original
