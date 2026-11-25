"""Staged NotebookLM source filenames must never trip the Drive backup's secret-file guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from conference_pipeline.podcast_notebooklm import backup_safe_slug, slugify
from scripts.sync_to_gdrive import is_secret_like_path


@pytest.mark.parametrize(
    "title",
    [
        # CVPR 2026 deck 10: this talk title made the whole backup refuse to run.
        "Secrets in the Embeddings: What Unlabeled Data Can Tell Us",
        "Password-Free Authentication for Robots",
        "Cookie Banners and Visual Consent",
        "Refresh Token Rotation in Agent Toolchains",
        "Leaking an API Key Through a Screenshot",
        "Credential Stuffing Detection",
    ],
)
def test_staged_name_passes_the_backup_guard(title: str) -> None:
    name = f"15__{backup_safe_slug('talk-transcript')}__{backup_safe_slug(title)}.md"
    assert not is_secret_like_path(Path(name))
    # ...while the plain slug really would have tripped it.
    assert is_secret_like_path(Path(f"15__talk-transcript__{slugify(title)}.md"))


def test_ordinary_titles_keep_their_slug() -> None:
    for title in ("Visual Token Pruning for Fast VLMs", "FLARE: Failure-Aware Recovery", "Key Frames Matter"):
        assert backup_safe_slug(title) == slugify(title)
