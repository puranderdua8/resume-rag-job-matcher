# Demo video script

Timed for 3–4 minutes. Record with the repo already cloned and the venv already set up (`uv venv`, `uv pip install -r requirements-dev.txt`) — skip the install wait, mention it happened.

## 1. Repo tour (30s)

- Open the repo root. Point out `resume_rag.py` (Part A) and `job_matcher.py` (Part B) as the two CLI entry points, and that the pipeline logic for each lives split across several small, single-concern modules (`resume_sections.py`, `resume_metadata.py`, `jd_parsing.py`, `retrieval.py`, ...).
- Open [PLAN.md](../PLAN.md), scroll to the **Progress** checklist. Say this is the source of truth the whole build followed, one step at a time.
- Open [README.md](../README.md)'s pipeline diagram. Trace it once: resumes in on the left (Part A), a job description in on the right (Part B), both meeting at the Chroma index in the middle.

## 2. `ingest` and `stats` (45s)

```bash
python resume_rag.py ingest
python resume_rag.py stats
```

- While `ingest` runs, narrate: load resumes via the Milestone 1 file tools, detect sections, extract metadata (name/skills/years/education), chunk, embed locally with `bge-small`, store in ChromaDB.
- On `stats`, point out: 36 resumes loaded, 0 skipped; chunk counts per section; the per-profile table (years, education level, skill count, chunk count) — pick one row and say what each column means.

## 3. Walk through a JD match's JSON (60s)

```bash
python job_matcher.py --jd data/job_descriptions/jd_senior_ml_engineer.txt
```

- Open the printed/written JSON. Walk through one `top_matches` entry end to end:
  - `match_score` and `score_breakdown` — the four weighted components.
  - `matched_skills` — required skills listed first.
  - `relevant_excerpts` — the resume text the score is actually based on.
  - `reasoning` — the plain-English summary of the same evidence.
- Mention `--llm` exists as an optional flag that paraphrases this same evidence with Gemini, and that it's off by default so the demo is fully reproducible without an API key.

## 4. The must-have filter in action (45s)

- Scroll to `filtered_out` in the same JSON. Pick one candidate and read their `reasons` aloud (e.g. "Python: 3.0 yrs < 5 required").
- Say explicitly: this candidate was excluded by the must-have filter on extracted metadata, not by scoring badly — the filter runs before the top-K cut, so a required-skill shortfall can't be hidden by a high semantic score elsewhere.
- Optionally demo an override live:

  ```bash
  python job_matcher.py --jd data/job_descriptions/jd_backend_python_fastapi.txt \
      --must-have "3+ years Python" --soft
  ```

  Point out the `soft_penalty` in `score_breakdown` for a candidate who now appears in `top_matches` instead of `filtered_out`.

## 5. Notebook charts (45s)

Open 1–2 of the seven notebooks already executed (don't re-run live — too slow for a demo):

- [`notebooks/03_experiment_b_search_mode.ipynb`](../notebooks/03_experiment_b_search_mode.ipynb): semantic vs. keyword vs. hybrid, and the nDCG numbers.
- [`notebooks/07_calibration_and_analysis.ipynb`](../notebooks/07_calibration_and_analysis.ipynb): the score-calibration distribution (relevant vs. irrelevant candidates) and the PCA of resume embeddings colored by role.

## 6. Latency (15s)

```bash
python evaluate.py
```

- Point at the printed latency table (or the figures already in README.md's Metrics section): ~48ms total query latency at p50, dominated by embedding the JD and the BM25 pass; a full 36-resume ingest is ~7s.
- Close on the README's Metrics and Limitations sections as the honest summary of what these numbers do and don't claim (synthetic labels, small dataset).
