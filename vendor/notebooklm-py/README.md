# notebooklm-py Operator Pin

This directory records the pinned external NotebookLM automation dependency for
public release audio generation.

- Repository: `https://github.com/teng-lin/notebooklm-py`
- Commit: `1712d76cc2906d96a9386177e554529d677567a8`
- License noted upstream: MIT

This is an unofficial NotebookLM automation path. It uses Playwright browser
automation, requires a Google account and cookies/profile state for live runs,
is rate-limited, can break without notice, and is not a hosted public endpoint.
Phase 09 exposes it only through a dry-run/live-gated operator and self-host
wrapper.

The per-deck production lane (`conference_pipeline.podcast_notebooklm`, `scripts/generate_notebooklm_podcast.py`) installs the PyPI release pinned by the `notebooklm` extra in `pyproject.toml`; the engine version actually used is recorded per run in each `audio/<deck>/notebooklm_podcast/notebook_manifest.json`. The commit pin above remains the historical Phase 09 public-release reference.
