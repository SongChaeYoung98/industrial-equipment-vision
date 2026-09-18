# Repository instructions for AI agents

## Commit message format

Every commit message in this repository must use this structure:

```text
<type>: <short imperative summary>

- <brief change or behavior>
- <brief change or behavior>

Files updated:
- <repository-relative/path>
- <repository-relative/path>

YYYY-MM-DD HH:mm
```

Use a conventional type such as `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, or `perf`.
Keep the subject concise and describe the user-visible or engineering outcome. The body must use
short dash bullets. List every meaningful file changed under `Files updated:`. Use the local
Asia/Seoul date and 24-hour time for the final line.

Example:

```text
feat: add anchor-based equipment pseudo-labeling

- Extract masked DINOv2 embeddings from the full media set
- Assign labels from human-reviewed anchor folders

Files updated:
- tools/run_full_pipeline.py
- ml/data/processed/full_pipeline/final_records.json

2026-09-18 15:30
```

Do not invent file paths. Before committing, inspect `git status` and include the actual changed
files. Do not commit raw datasets, derived media, model weights, credentials, or runtime logs.

## Repository safety

- Do not modify or delete production-server data.
- Treat `ml/data/raw/`, `ml/data/derived/`, `ml/data/processed/`, `models/`, and virtual environments
  as local/runtime data unless the user explicitly requests versioning them.
- Do not push commits or create external changes unless the user explicitly asks.
