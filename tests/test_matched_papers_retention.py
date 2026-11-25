"""A matching re-run keeps the list it replaced when the matches changed."""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.paper_matching.matcher import (
    MATCHED_PAPERS_SUPERSEDED_INFIX,
    retain_superseded_matched_papers,
)


def _write(path: Path, titles: list[str], created_at: str) -> bytes:
    payload = {"created_at": created_at, "papers": [{"title": title, "rank": i + 1} for i, title in enumerate(titles)]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.read_bytes()


def _retained(directory: Path) -> list[Path]:
    return sorted(directory.glob(f"matched_papers{MATCHED_PAPERS_SUPERSEDED_INFIX}*.json"))


def test_changed_matches_keep_the_previous_list(tmp_path: Path) -> None:
    path = tmp_path / "matched_papers.json"
    previous = _write(path, ["AnyID", "EvoID"], "2026-09-20T00:00:00Z")
    _write(path, ["MultiCrafter"], "2026-09-22T00:00:00Z")

    retained = retain_superseded_matched_papers(path, previous)

    assert retained is not None and _retained(tmp_path) == [retained]
    assert retained.read_bytes() == previous


def test_identical_matches_retain_nothing(tmp_path: Path) -> None:
    path = tmp_path / "matched_papers.json"
    previous = _write(path, ["AnyID"], "2026-09-20T00:00:00Z")
    _write(path, ["AnyID"], "2026-09-22T00:00:00Z")

    assert retain_superseded_matched_papers(path, previous) is None
    assert _retained(tmp_path) == []


def test_a_first_run_retains_nothing(tmp_path: Path) -> None:
    path = tmp_path / "matched_papers.json"
    _write(path, ["AnyID"], "2026-09-22T00:00:00Z")

    assert retain_superseded_matched_papers(path, None) is None
