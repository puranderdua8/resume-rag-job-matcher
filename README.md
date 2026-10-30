# Resume RAG + Job Matcher

A retrieval-augmented system for matching resumes to job descriptions. It splits resumes into chunks by section, embeds them locally with sentence-transformers, and stores them in ChromaDB with extracted metadata (name, skills, years of experience, education). It then ranks candidates for a job description using hybrid search: semantic similarity combined with BM25 keyword matching through Reciprocal Rank Fusion (RRF).

Each match gets a 0–100 score with reasoning based on which resume sections matched, and must-have requirements such as "5+ years Python" are filtered first.

> **Status: in development.** The design and step-by-step implementation plan are in [PLAN.md](PLAN.md). Its progress checklist shows what's done so far.

Resumes are loaded with the file-system tools from Milestone 1, [file-system-assistant](https://github.com/puranderdua8/file-system-assistant).

## Planned components

| File | Purpose |
|---|---|
| `resume_rag.py` | Load, section, chunk, extract metadata, embed and store in ChromaDB |
| `job_matcher.py` | Parse the JD, run hybrid retrieval, filter on must-haves, score, explain, and output JSON |
| `evaluate.py` | Retrieval accuracy (P@K, Recall@K, MRR, nDCG) and latency |
| `notebooks/experiments.ipynb` | Experiments on chunking, search mode, embedding models, weights and reranking |
| `data/` | 36 synthetic resumes, 6 job descriptions and ground-truth labels |

## For AI agents

Coding agents should start with [AGENTS.md](AGENTS.md). [CLAUDE.md](CLAUDE.md) adds notes for Claude Code.

## License

[MIT](LICENSE)
