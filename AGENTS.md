# Agent Guidelines: Resume RAG + Job Matcher

These are the canonical instructions for any AI coding agent working in this repo (Claude Code, Codex, Cursor, Gemini CLI and others). Tool-specific files such as [CLAUDE.md](CLAUDE.md) extend this file. They must not repeat or contradict it.

## Project

This repo is a RAG system that indexes resumes and matches them against job descriptions. It is Milestone 2 of an AI-first software engineering assignment.

- **`resume_rag.py`** (Part A) loads resumes with the Milestone 1 tools, splits them into chunks by section, extracts metadata, embeds the chunks locally with sentence-transformers, and stores them in ChromaDB.
- **`job_matcher.py`** (Part B) combines semantic and BM25 retrieval with Reciprocal Rank Fusion (RRF). It filters candidates on must-have requirements, scores each one from 0 to 100 with reasoning, and writes JSON.

## Source of truth: PLAN.md

[PLAN.md](PLAN.md) holds the full design, every implementation step, and the definition of done.

1. **Before you start**, read PLAN.md. Find the first unchecked item in its **Progress** list. That is the current step.
2. **Work one step at a time, in order, on its own branch.** See [Git workflow](#git-workflow). Don't start the next step until the current one meets its own checks and its PR is merged.
3. **When a step is done**, do all of the following in the same commit:
   - Tick the step in the Progress list.
   - Update anything in PLAN.md the implementation changed.
   - Commit, push the branch, and open a PR.
4. **When you deviate from the plan**, record the change and the reason in PLAN.md. Never leave the plan describing code that doesn't exist.

## Hard rules

- **Load resumes only through Milestone 1.** Use the tools from [file-system-assistant](https://github.com/puranderdua8/file-system-assistant), vendored as a git submodule at `vendor/file-system-assistant` and imported via `file_tools.py`. Don't write a separate PDF/DOCX loader.
- **Don't shadow M1's module names.** M1's modules are top-level files: `fs_tools`, `parsers`, `sandbox`, `errors`, `utils`. Never create files with those names in this repo.
- **M1 tools never raise.** They return `{"success": False, "error": ...}` instead, and `list_files` returns a one-item list containing that dict. Check `success` and skip failed files with a logged warning.
- **Leave `FS_ROOT` unset.** It turns on M1's sandbox, which this project doesn't use.
- **Keep the output contract.** Every entry in `top_matches` must have `candidate_name`, `resume_path`, `match_score` (an int from 0 to 100), `matched_skills`, `relevant_excerpts` and `reasoning`. Extra fields are allowed, but never remove or rename these.
- **Run must-have filters before the top-K cut.**
- **Configure the vector DB the PLAN.md way.** That means ChromaDB 1.5 or newer, a local `PersistentClient`, precomputed embeddings, `embedding_function=None`, and `configuration={"hnsw": {"space": "cosine"}}`. Skills are stored as `string[]` metadata so `$contains` filters work.
- **No network in tests.** Use the fake embedder and a temp Chroma directory. Never download models or call an LLM from `pytest`.
- **The LLM is optional.** Every feature works without `GEMINI_API_KEY`. `--llm` falls back to template reasoning on any error.
- **Keep secrets and generated state out of git.** Never commit `.env`, `index/`, `.venv/` or model caches. The resumes in `data/` are fictional and synthetic, so they are committed. Never add real personal data.

## Commands

Run these after Step 1 is done.

Set up the environment:

```bash
git submodule update --init --recursive
uv venv -p 3.13 .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

Lint and test. All of these must pass before every commit:

```bash
ruff check . && ruff format --check . && pytest -q
```

Generate the dataset:

```bash
python scripts/generate_dataset.py
```

Build the index and match:

```bash
python resume_rag.py ingest && python resume_rag.py stats
python job_matcher.py --all
```

Write the metrics to `outputs/metrics.json`:

```bash
python evaluate.py
```

Run the Step 6 notebooks end to end (seven standalone notebooks, `01_...` through `07_...`; each re-ingests its own collection(s), so order doesn't matter):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

## Code conventions

- **Formatting:** Python 3.12 or newer. Ruff with line-length 100 and lint rules E/F/I/B/UP; the config is in `pyproject.toml`, matching Milestone 1.
- **Code size:** keep modules small and flat. Don't add abstractions the plan doesn't call for.
- **Data types:** use dataclasses for structured data such as `CandidateProfile`, `Chunk`, `JDRequirements` and `MatchResult`. Use type hints everywhere.
- **Configuration:** tunable numbers live in `config.py`. That includes K, score weights, calibration bounds, RRF k and the model registry. Never hard-code them inline.
- **Determinism:** the dataset generator uses a fixed seed, and ingest is idempotent (it recreates the collection).
- **Comments:** only where the reason isn't obvious from the code.

## Git workflow

- **One branch per plan step.** Branch from an up-to-date `main` before starting a step, e.g. `step-1-scaffold`. Never commit a step's work directly to `main`.
- Use one commit per plan step, at minimum. Messages are imperative and state what changed, e.g. "Add section-aware chunking and metadata extraction".
- **Push the branch and open a PR against `main`.** Wait for review and keep CI green on the PR; don't merge your own PR without review.
- Merge once review is complete, then delete the branch. Don't start the next step's branch until the merge lands.
- Don't rewrite published history.
- No AI attribution: leave `Co-Authored-By` trailers and "Generated with" lines out of commits and PR descriptions.
