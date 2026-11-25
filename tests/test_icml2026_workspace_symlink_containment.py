"""Tests for resolve_workspace_member's tolerance of a deliberate workspace symlink.

Context: conferences/icml2026/storage/transcription-ocr and
storage/workshops/papers are symlinks into a Google Drive-backed on-demand
artifact store (per storage_manifest.json). A strict symlink-following
containment check rejected every artifact recorded under those trees during
corpus-inventory-index, producing thousands of spurious path_outside_workspace
blockers. resolve_workspace_member now checks containment lexically
(``os.path.abspath``, which normalizes ``.``/``..`` without following
symlinks) since every caller here validates a single already-known,
pipeline-recorded path value rather than walking a directory tree.

conference_pipeline.workflow._resolve_path_inside_workspace is a separate,
independent copy of the same pattern (Path.resolve() then relative_to) used
by the exhaustive-extraction walker (extractions/walker.py) and
iclr2025_live.py. It hit the identical bug live: the LLM full-read
extraction stage logged extraction.path_escape for every one of the 43
newly-recovered workshop-talk annotated transcripts under
storage/transcription-ocr/workshop-talks/, silently dropping them from the
deep-evidence layer the whole recovery effort was about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conference_pipeline import slideslive_recovery, transcription_ocr, workshop_talk_segmentation
from conference_pipeline.indexing import corpus_vector_store
from conference_pipeline.workflow import _resolve_path_inside_workspace
from scripts import run_icml2026_production as runner
from scripts import run_iclr2026_production as iclr_runner

WORKSPACE_MEMBER_RESOLVERS = [
    pytest.param(runner.resolve_workspace_member, id="run_icml2026_production"),
    pytest.param(iclr_runner.resolve_workspace_member, id="run_iclr2026_production"),
    pytest.param(transcription_ocr.resolve_workspace_member, id="transcription_ocr"),
    pytest.param(corpus_vector_store.resolve_workspace_member, id="corpus_vector_store"),
    pytest.param(workshop_talk_segmentation.resolve_workspace_member, id="workshop_talk_segmentation"),
]


def _unwrap(result: object) -> Path | None:
    # slideslive_recovery.resolve_workspace_member returns a bare Path | None;
    # the rest return a (Path | None, str | None) tuple.
    if isinstance(result, tuple):
        return result[0]
    return result


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
@pytest.mark.parametrize("resolve_member", WORKSPACE_MEMBER_RESOLVERS)
def test_resolve_workspace_member_tolerates_symlinked_workspace_ancestor(
    tmp_path: Path, resolve_member
) -> None:
    """The workspace root itself (not just a leaf artifact dir) can sit behind a symlink.

    Regression for mixing a symlink-following .resolve() on one side of the
    containment check with a lexical os.path.abspath() on the other: if
    workspace_root is computed via .resolve() but the candidate path is
    computed via os.path.abspath(), the two use different namespaces
    whenever the workspace path itself (not just an artifact under it) has
    a symlinked ancestor, and every containment check then spuriously
    fails. Both sides must use the same non-resolving normalization.
    """
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent)

    workspace = linked_parent / "conference"
    (real_parent / "conference").mkdir()
    (real_parent / "conference" / "output.txt").write_text("hello", encoding="utf-8")

    resolved = _unwrap(resolve_member(workspace, "output.txt"))
    assert resolved is not None
    assert resolved.read_text(encoding="utf-8") == "hello"


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_resolve_workspace_member_slideslive_recovery_tolerates_symlinked_workspace_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent)

    workspace = linked_parent / "conference"
    (real_parent / "conference").mkdir()
    (real_parent / "conference" / "output.txt").write_text("hello", encoding="utf-8")

    resolved = slideslive_recovery.resolve_workspace_member(workspace, "output.txt")
    assert resolved is not None
    assert resolved.read_text(encoding="utf-8") == "hello"


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_resolve_workspace_member_slideslive_recovery_allows_leaf_drive_backed_symlink(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conference"
    workspace.mkdir()
    external_root = tmp_path / "drive-backed-store"
    external_root.mkdir()
    (external_root / "output.txt").write_text("hello", encoding="utf-8")
    linked_dir = workspace / "storage" / "transcription-ocr"
    linked_dir.parent.mkdir(parents=True, exist_ok=True)
    linked_dir.symlink_to(external_root)

    resolved = slideslive_recovery.resolve_workspace_member(
        workspace, "storage/transcription-ocr/output.txt"
    )
    assert resolved is not None
    assert resolved.read_text(encoding="utf-8") == "hello"


def test_resolve_workspace_member_allows_workspace_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    workspace.mkdir()
    external_root = tmp_path / "drive-backed-store"
    external_root.mkdir()
    real_file = external_root / "output.txt"
    real_file.write_text("hello", encoding="utf-8")
    linked_dir = workspace / "storage" / "transcription-ocr"
    linked_dir.parent.mkdir(parents=True, exist_ok=True)
    linked_dir.symlink_to(external_root)

    resolved, error = runner.resolve_workspace_member(
        workspace, "storage/transcription-ocr/output.txt"
    )
    assert error is None
    assert resolved is not None
    assert resolved.exists()
    assert resolved.read_text(encoding="utf-8") == "hello"


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
@pytest.mark.parametrize("resolve_member", WORKSPACE_MEMBER_RESOLVERS)
def test_resolve_workspace_member_allows_leaf_drive_backed_symlink(
    tmp_path: Path, resolve_member
) -> None:
    """The original bug: storage/transcription-ocr is a Drive-backed symlink whose
    real target sits outside the workspace tree entirely. A regression here means
    every artifact under that symlink is spuriously rejected as path_outside_workspace,
    which is exactly what happened live across thousands of corpus-inventory-index
    records before this fix."""
    workspace = tmp_path / "conference"
    workspace.mkdir()
    external_root = tmp_path / "drive-backed-store"
    external_root.mkdir()
    (external_root / "output.txt").write_text("hello", encoding="utf-8")
    linked_dir = workspace / "storage" / "transcription-ocr"
    linked_dir.parent.mkdir(parents=True, exist_ok=True)
    linked_dir.symlink_to(external_root)

    resolved = _unwrap(resolve_member(workspace, "storage/transcription-ocr/output.txt"))
    assert resolved is not None
    assert resolved.read_text(encoding="utf-8") == "hello"


def test_resolve_workspace_member_still_blocks_dotdot_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    resolved, error = runner.resolve_workspace_member(workspace, "../outside.txt")
    assert resolved is None
    assert error == "path_outside_workspace"


def test_resolve_workspace_member_blocks_absolute_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    resolved, error = runner.resolve_workspace_member(workspace, str(outside))
    assert resolved is None
    assert error == "path_outside_workspace"


def test_resolve_path_inside_workspace_allows_workspace_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    workspace.mkdir()
    external_root = tmp_path / "drive-backed-store"
    external_root.mkdir()
    real_file = external_root / "workshop-talks" / "annotated.json"
    real_file.parent.mkdir(parents=True)
    real_file.write_text("{}\n", encoding="utf-8")
    linked_dir = workspace / "storage" / "transcription-ocr"
    linked_dir.parent.mkdir(parents=True, exist_ok=True)
    linked_dir.symlink_to(external_root)

    resolved = _resolve_path_inside_workspace(
        workspace,
        "storage/transcription-ocr/workshop-talks/annotated.json",
        label="extraction-input",
    )
    assert resolved.read_text(encoding="utf-8") == "{}\n"


def test_resolve_path_inside_workspace_still_blocks_dotdot_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    workspace.mkdir()
    (tmp_path / "outside.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must stay inside the conference workspace"):
        _resolve_path_inside_workspace(workspace, "../outside.json", label="extraction-input")
