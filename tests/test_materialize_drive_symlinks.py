"""Drive-symlink materializer: plan, copy+verify+swap, revert, fail-safe paths."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "materialize_drive_symlinks.py"


def _load():
    spec = importlib.util.spec_from_file_location("materialize_drive_symlinks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mdl = _load()


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    """A fake repo (``repo/``) with a workspace whose storage links into ``drive/``."""

    drive = tmp_path / "drive" / "ICML-raw"
    (drive / "storage" / "corpus" / "papers").mkdir(parents=True)
    (drive / "storage" / "corpus" / "papers" / "a.txt").write_text("alpha", encoding="utf-8")
    (drive / "storage" / "corpus" / "index.json").write_text("{}", encoding="utf-8")
    (drive / "audio").mkdir()
    (drive / "audio" / "t.m4a").write_bytes(b"\x00" * 64)
    repo = tmp_path / "repo"
    workspace = repo / "conferences" / "demo"
    (workspace / "storage").mkdir(parents=True)
    (workspace / "storage" / "keep.json").write_text("{}", encoding="utf-8")
    os.symlink(drive / "storage" / "corpus", workspace / "storage" / "corpus")
    os.symlink(drive / "audio", workspace / "audio")
    # A link inside the repo is not a Drive payload and must be ignored.
    os.symlink(workspace / "storage" / "keep.json", workspace / "storage" / "alias.json")
    return repo, workspace


def test_finds_only_links_that_leave_the_repo(tmp_path: Path) -> None:
    repo, workspace = _layout(tmp_path)
    found = mdl.find_linked_payloads(workspace, repo_root=repo)
    assert [p.relative_link for p in found] == ["audio", "storage/corpus"]
    corpus = mdl.measure(found[1])
    assert (corpus.file_count, corpus.total_bytes) == (2, len("alpha") + len("{}"))


def test_apply_copies_verifies_swaps_and_revert_restores(tmp_path: Path, monkeypatch) -> None:
    repo, workspace = _layout(tmp_path)
    monkeypatch.setattr(mdl, "FREE_SPACE_MARGIN_BYTES", 0)
    original_find = mdl.find_linked_payloads
    monkeypatch.setattr(mdl, "find_linked_payloads", lambda ws: original_find(ws, repo_root=repo))

    assert mdl.main(["--workspace", str(workspace)]) == 0  # dry run
    assert (workspace / "storage" / "corpus").is_symlink()

    assert mdl.main(["--workspace", str(workspace), "--apply", "--hash"], copier=mdl.rsync_or_copytree) == 0
    corpus = workspace / "storage" / "corpus"
    assert corpus.is_dir() and not corpus.is_symlink()
    assert (corpus / "papers" / "a.txt").read_text(encoding="utf-8") == "alpha"
    assert not (workspace / "audio").is_symlink()
    # Drive keeps its copy.
    assert (tmp_path / "drive" / "ICML-raw" / "audio" / "t.m4a").is_file()
    receipt = json.loads((workspace / "storage" / mdl.DEFAULT_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert {r["relative_link"]: r["status"] for r in receipt["results"]} == {
        "audio": "materialized",
        "storage/corpus": "materialized",
    }
    assert receipt["hash_verified"] is True

    assert mdl.main(["--workspace", str(workspace), "--revert", "--only", "audio"]) == 0
    assert (workspace / "audio").is_symlink()
    backups = [p.name for p in workspace.iterdir() if p.name.startswith("audio.local-")]
    assert len(backups) == 1  # the local copy is renamed, never deleted
    assert corpus.is_dir() and not corpus.is_symlink()


def test_a_copy_that_fails_verification_leaves_the_link(tmp_path: Path, monkeypatch) -> None:
    repo, workspace = _layout(tmp_path)
    monkeypatch.setattr(mdl, "FREE_SPACE_MARGIN_BYTES", 0)
    original_find = mdl.find_linked_payloads
    monkeypatch.setattr(mdl, "find_linked_payloads", lambda ws: original_find(ws, repo_root=repo))

    def lossy_copier(source: Path, destination: Path) -> None:
        mdl.rsync_or_copytree(source, destination)
        for path in destination.rglob("a.txt"):
            path.unlink()

    code = mdl.main(["--workspace", str(workspace), "--apply", "--only", "storage/corpus"], copier=lossy_copier)
    assert code == 1
    assert (workspace / "storage" / "corpus").is_symlink()
    assert (workspace / "storage" / ("corpus" + mdl.MATERIALIZING_SUFFIX)).is_dir()
    receipt = json.loads((workspace / "storage" / mdl.DEFAULT_RECEIPT_NAME).read_text(encoding="utf-8"))
    (entry,) = receipt["results"]
    assert entry["status"] == "verify_failed"
    assert "missing: papers/a.txt" in entry["mismatches"]


def test_apply_refuses_without_space_or_with_unreadable_targets(tmp_path: Path, monkeypatch) -> None:
    repo, workspace = _layout(tmp_path)
    original_find = mdl.find_linked_payloads
    monkeypatch.setattr(mdl, "find_linked_payloads", lambda ws: original_find(ws, repo_root=repo))
    monkeypatch.setattr(mdl, "FREE_SPACE_MARGIN_BYTES", 10**18)
    assert mdl.main(["--workspace", str(workspace), "--apply"]) == 1
    assert (workspace / "audio").is_symlink()

    monkeypatch.setattr(mdl, "FREE_SPACE_MARGIN_BYTES", 0)
    os.symlink(tmp_path / "drive" / "gone", workspace / "storage" / "dangling")
    assert mdl.main(["--workspace", str(workspace), "--apply"]) == 1
    assert (workspace / "audio").is_symlink()


def test_only_rejects_paths_that_are_not_drive_links(tmp_path: Path, monkeypatch) -> None:
    repo, workspace = _layout(tmp_path)
    original_find = mdl.find_linked_payloads
    monkeypatch.setattr(mdl, "find_linked_payloads", lambda ws: original_find(ws, repo_root=repo))
    with pytest.raises(SystemExit):
        mdl.parse_args(["--workspace", str(workspace), "--apply", "--revert"])
    assert mdl.main(["--workspace", str(workspace), "--only", "storage/keep.json"]) == 2


def test_an_unreadable_target_is_never_measured_as_empty(tmp_path: Path, monkeypatch) -> None:
    """Regression: os.walk swallows PermissionError, so a sandbox that cannot read
    the Drive mount saw every family as an empty tree."""

    repo, workspace = _layout(tmp_path)
    found = {p.relative_link: p for p in mdl.find_linked_payloads(workspace, repo_root=repo)}
    real_listdir = os.listdir

    def denied(path, *args, **kwargs):
        if "drive" in str(path):
            raise PermissionError(1, "Operation not permitted", str(path))
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "listdir", denied)
    measured = mdl.measure(found["storage/corpus"])
    assert measured.readable is False
    assert "PermissionError" in (measured.error or "")


def test_an_empty_target_is_not_swapped(tmp_path: Path) -> None:
    repo, workspace = _layout(tmp_path)
    empty = tmp_path / "drive" / "ICML-raw" / "empty"
    empty.mkdir()
    link = workspace / "storage" / "empty"
    os.symlink(empty, link)
    payload = mdl.LinkedPayload(link=link, target=empty, relative_link="storage/empty")
    result = mdl.materialize_one(payload, hash_files=False)
    assert result.status == "verify_failed"
    assert link.is_symlink()
