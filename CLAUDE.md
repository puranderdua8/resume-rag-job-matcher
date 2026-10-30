@AGENTS.md

# Claude Code notes

This file adds to [AGENTS.md](AGENTS.md), which is imported above. If the two disagree, AGENTS.md wins, so update it rather than overriding it here.

- **Start of a session:** read [PLAN.md](PLAN.md), identify the current step from its Progress checklist, and turn that step's bullets into your task list.
- **Large plan changes:** if a step needs a bigger design change than the plan allows for, propose it and get approval before you build it. Record the agreed change in PLAN.md.
- **Library APIs:** check current ChromaDB and sentence-transformers APIs against their docs or the installed package before you write code. Both have changed recently: Chroma 1.5 added array metadata, and the old `hnsw:space` metadata is now legacy.
- **Long commands:** model downloads, full ingest and notebook execution can take minutes. Run them in the background.
- **Before reporting a step done:** run the lint and test commands from AGENTS.md, plus that step's own checks from PLAN.md, and say what you could not verify.
