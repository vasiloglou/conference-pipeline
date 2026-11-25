"""Hermetic pytest suite for scripts/restore_drive_symlinks.py.

Every test in Groups A–G builds a synthetic workspace under ``tmp_path`` and
exercises a specific LINK-09 behavior verb without touching the real mounted
Google Drive or any filesystem state outside ``tmp_path``.

Exactly one gated test (``test_iclr_verify_smoke_passes_when_drive_mounted``)
is permitted to reference the real ICLR 2026 Drive leaf; it is guarded by a
``pytest.mark.skipif`` so CI and unmounted dev machines stay green.

Pattern: mirrors ``tests/test_sync_to_gdrive.py`` — importlib.util module
load, make_restore_workspace / write_restore_manifest tmp_path helpers,
monkeypatched env isolation, no Drive dependency in hermetic tests.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Real ICLR Drive leaf (gated smoke test only — never referenced in hermetic tests).
# Read at run time from conferences/iclr2026/DRIVE.md ("Mounted path, when
# available:"), so no machine path is committed in this file; absent in a
# checkout without the conference workspace (the exported code repository).
# ---------------------------------------------------------------------------
ICLR_DRIVE_DOC = Path(__file__).resolve().parents[1] / "conferences" / "iclr2026" / "DRIVE.md"


# ---------------------------------------------------------------------------
# Hermetic env isolation (T-03-03 mitigation — DoS via real-Drive discovery)
#
# Every hermetic test overrides the Drive leaf via a tmp_path fake DRIVE.md, and
# the restore module currently reads no environment variables. As defense in
# depth — and so no GDRIVE_* var can ever steer a test at the real mounted Drive
# should env-var-based resolution be added later — this autouse fixture deletes
# every GDRIVE_*-prefixed variable from the environment before each test runs.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_gdrive_env(monkeypatch):
    """Delete all GDRIVE_* env vars so no test can discover the real Drive mount."""
    for name in list(os.environ):
        if name.startswith("GDRIVE_"):
            monkeypatch.delenv(name, raising=False)
    # Explicit guards for the known ecosystem vars even when unset in this env.
    for name in ("GDRIVE_PUBLIC_PATH", "GDRIVE_BACKUP_PATH", "GDRIVE_BACKUP_LEAF"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Module loader (mirrors test_sync_to_gdrive.py lines 11-19 exactly)
# ---------------------------------------------------------------------------

def load_restore_module():
    """Load scripts/restore_drive_symlinks.py via importlib and return the module."""
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "restore_drive_symlinks.py"
    )
    spec = importlib.util.spec_from_file_location("restore_drive_symlinks", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _iclr_drive_leaf() -> Path | None:
    """The mounted ICLR 2026 Drive leaf named in DRIVE.md, or None."""
    if not ICLR_DRIVE_DOC.is_file():
        return None
    mounted, _leaf_name = load_restore_module().parse_drive_doc(ICLR_DRIVE_DOC)
    return mounted


ICLR_DRIVE_LEAF = _iclr_drive_leaf()


# ---------------------------------------------------------------------------
# Workspace + manifest helpers (mirrors test_sync_to_gdrive.py helpers)
# ---------------------------------------------------------------------------

def make_restore_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal synthetic conference workspace under tmp_path.

    Returns (conference_dir, drive_leaf) where drive_leaf is a real directory
    under tmp_path that serves as the fake mounted Drive leaf. The
    ``conferences/demo2026/DRIVE.md`` points at this fake leaf so hermetic
    tests never discover the real Drive mount from the workspace config.
    """
    conference_dir = tmp_path / "conferences" / "demo2026"
    conference_dir.mkdir(parents=True)

    # Fake Drive leaf — a real directory under tmp_path, not the real mount.
    drive_leaf = tmp_path / "fake-drive-root" / "demo-drive-raw"
    drive_leaf.mkdir(parents=True)

    # config.yaml required by resolve_conference_paths
    (conference_dir / "config.yaml").write_text(
        "conference:\n  name: Demo 2026\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )

    # DRIVE.md with the fake leaf path so the module never reads the real mount.
    drive_doc = conference_dir / "DRIVE.md"
    drive_doc.write_text(
        "# Demo 2026 Drive\n\n"
        "- Drive leaf: `demo-drive-raw`\n"
        "- Mounted path, when available:\n"
        f"  `{drive_leaf}`\n",
        encoding="utf-8",
    )

    return conference_dir, drive_leaf


def write_restore_manifest(path: Path, families: list[dict]) -> None:
    """Write a storage_manifest.json with the given artifact families.

    Every family MUST carry ``storage_class`` — required by
    ``_classify_local_target`` and family-selection logic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace": "conferences/demo2026",
                "inventory_exemptions": [],
                "artifact_families": families,
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Group A — Family selection (D-02a)
# ---------------------------------------------------------------------------

def test_family_selection_includes_public_and_backup(tmp_path: Path, monkeypatch) -> None:
    """public and backup storage_class families must appear as eligible."""
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    # Populate Drive leaf so the directory-level path families have existing targets.
    (drive_leaf / "decks").mkdir()
    (drive_leaf / "archives").mkdir()
    (drive_leaf / "git-owned").mkdir()
    (drive_leaf / "cache-data").mkdir()

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "decks",
                "path": "decks",
                "storage_class": "public",
                "sync_profiles": {"public": True, "backup": True},
            },
            {
                "id": "archives",
                "path": "archives",
                "storage_class": "backup",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "git_control",
                "path": "git-owned",
                "storage_class": "git",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "cache_data",
                "path": "cache-data",
                "storage_class": "cache",
                "sync_profiles": {"public": False, "backup": False},
            },
        ],
    )

    config = module.RestoreConfig(
        conference_dir=conference_dir,
        config_path=conference_dir / "config.yaml",
        drive_leaf=drive_leaf,
        drive_leaf_name="demo-drive-raw",
        want_json=False,
    )
    plan = module.build_restore_plan(config)

    eligible_ids = {fp.family_id for fp in plan.families if fp.eligible}
    skipped_ids = {fp.family_id for fp in plan.families if not fp.eligible}

    assert "decks" in eligible_ids, "public family must be eligible"
    assert "archives" in eligible_ids, "backup family must be eligible"
    assert "git_control" in skipped_ids, "git family must be skipped"
    assert "cache_data" in skipped_ids, "cache family must be skipped"


def test_family_selection_skips_git_and_cache(tmp_path: Path, monkeypatch) -> None:
    """Skipped git and cache families must carry a non-empty reason string."""
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    (drive_leaf / "git-owned").mkdir()
    (drive_leaf / "cache-data").mkdir()

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "git_control",
                "path": "git-owned",
                "storage_class": "git",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "cache_data",
                "path": "cache-data",
                "storage_class": "cache",
                "sync_profiles": {"public": False, "backup": False},
            },
        ],
    )

    config = module.RestoreConfig(
        conference_dir=conference_dir,
        config_path=conference_dir / "config.yaml",
        drive_leaf=drive_leaf,
        drive_leaf_name="demo-drive-raw",
        want_json=False,
    )
    plan = module.build_restore_plan(config)

    for fp in plan.families:
        assert not fp.eligible, f"{fp.family_id} should be skipped"
        for entry in fp.entries:
            assert entry.reason, (
                f"Skipped entry for {fp.family_id} must carry a non-empty reason"
            )
            assert entry.action == module.ACTION_SKIPPED


# ---------------------------------------------------------------------------
# Group B — Drive-path resolution + fail-closed (D-02b)
# ---------------------------------------------------------------------------

def test_fail_closed_missing_workspace(tmp_path: Path, monkeypatch) -> None:
    """main() must return non-zero when the conference slug directory is absent."""
    module = load_restore_module()

    # Use --conference-dir pointing at a path that does NOT exist
    nonexistent = tmp_path / "conferences" / "ghost-slug"

    exit_code = module.main([
        "--conference-dir", str(nonexistent),
        "--dry-run",
    ])
    assert exit_code != 0, "Expected non-zero exit for missing workspace"


def test_fail_closed_missing_manifest(tmp_path: Path, monkeypatch) -> None:
    """main() must return 1 when storage/storage_manifest.json is absent."""
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)
    # Deliberately do NOT create storage/storage_manifest.json

    exit_code = module.main([
        "--conference-dir", str(conference_dir),
        "--drive-path", str(drive_leaf),
        "--dry-run",
    ])
    assert exit_code == 1, "Expected exit code 1 for missing manifest"


def test_fail_closed_missing_drive_mount(tmp_path: Path, monkeypatch) -> None:
    """main() must return 1 when the Drive leaf path does not exist."""
    module = load_restore_module()
    conference_dir, _drive_leaf = make_restore_workspace(tmp_path)

    # Provide a manifest so the only failure point is the Drive leaf
    (drive_leaf_nonexistent := tmp_path / "drive" / "nonexistent").parent.mkdir(parents=True)

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "decks",
                "path": "decks",
                "storage_class": "public",
                "sync_profiles": {"public": True, "backup": True},
            },
        ],
    )

    # Override DRIVE.md to point at a path that does not exist
    (conference_dir / "DRIVE.md").write_text(
        "# Demo 2026 Drive\n\n"
        "- Drive leaf: `demo-drive-raw`\n"
        "- Mounted path, when available:\n"
        f"  `{drive_leaf_nonexistent}`\n",
        encoding="utf-8",
    )

    exit_code = module.main([
        "--conference-dir", str(conference_dir),
        "--dry-run",
    ])
    assert exit_code == 1, "Expected exit code 1 for missing Drive mount"


# ---------------------------------------------------------------------------
# Group C — Dry-run no-mutation (D-02c)
# ---------------------------------------------------------------------------

def test_dry_run_lists_actions_without_mutating(tmp_path: Path, monkeypatch) -> None:
    """Dry-run must exit 0 and create NO symlinks under conference_dir."""
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    # Populate Drive leaf with a real file so the path family has an existing target
    (drive_leaf / "decks").mkdir(parents=True)
    (drive_leaf / "decks" / "slide.png").write_bytes(b"\x89PNG\r\n")

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "decks",
                "path": "decks",
                "storage_class": "public",
                "sync_profiles": {"public": True, "backup": True},
            },
        ],
    )

    exit_code = module.main([
        "--conference-dir", str(conference_dir),
        "--drive-path", str(drive_leaf),
        "--dry-run",
    ])
    assert exit_code == 0, f"Dry-run must exit 0, got {exit_code}"

    # Walk the conference_dir tree and assert no symlinks were created
    for item in conference_dir.rglob("*"):
        assert not item.is_symlink(), (
            f"Dry-run must not create symlinks, but found one at: {item}"
        )


# ---------------------------------------------------------------------------
# Group D — Safe symlink creation (D-02d)
# ---------------------------------------------------------------------------

def test_apply_creates_symlink_for_absent_path(tmp_path: Path, monkeypatch) -> None:
    """apply must create a symlink when the local path is absent."""
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    # Create a real directory in the Drive leaf
    corpus_drive = drive_leaf / "storage" / "corpus"
    corpus_drive.mkdir(parents=True)
    (corpus_drive / "data.bin").write_bytes(b"binary_payload")

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "corpus",
                "path": "storage/corpus",
                "storage_class": "backup",
                "sync_profiles": {"public": False, "backup": True},
            },
        ],
    )

    # Ensure local path does NOT exist
    local_corpus = conference_dir / "storage" / "corpus"
    assert not local_corpus.exists()

    exit_code = module.main([
        "--conference-dir", str(conference_dir),
        "--drive-path", str(drive_leaf),
        "--apply",
    ])
    assert exit_code == 0, f"apply must exit 0, got {exit_code}"

    assert local_corpus.is_symlink(), "local path must be a symlink after apply"
    # The symlink must resolve to the Drive target
    resolved = local_corpus.resolve()
    assert resolved == corpus_drive.resolve(), (
        f"Symlink must resolve to Drive target. Got: {resolved}"
    )


def test_apply_creates_intermediate_dirs_for_glob_family(tmp_path: Path, monkeypatch) -> None:
    """For a path_glob family, intermediate dirs must be real dirs (not symlinks).

    The parent of a created leaf symlink must never itself be a symlink.
    """
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    # Drive leaf: decks/topic-a/slide_001.png
    (drive_leaf / "decks" / "topic-a").mkdir(parents=True)
    (drive_leaf / "decks" / "topic-a" / "slide_001.png").write_bytes(b"PNG_DATA")

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "presentation_payloads",
                "path_glob": "decks/**/*.png",
                "storage_class": "public",
                "sync_profiles": {"public": True, "backup": True},
            },
        ],
    )

    # Build and apply plan directly (avoids real Drive glob enumeration)
    config = module.RestoreConfig(
        conference_dir=conference_dir,
        config_path=conference_dir / "config.yaml",
        drive_leaf=drive_leaf,
        drive_leaf_name="demo-drive-raw",
        want_json=False,
    )
    plan = module.build_restore_plan(config)
    result = module.apply_restore_plan(plan)

    assert result.hard_fault is None, f"apply must not hard-fault: {result.hard_fault}"

    # The leaf symlink must exist
    leaf_link = conference_dir / "decks" / "topic-a" / "slide_001.png"
    assert leaf_link.is_symlink(), "Glob leaf must be a symlink"

    # Intermediate directory must be a real dir, not a symlink
    topic_a_dir = conference_dir / "decks" / "topic-a"
    assert topic_a_dir.is_dir(), "Intermediate dir must be a real directory"
    assert not topic_a_dir.is_symlink(), (
        "Intermediate dir must NOT be a symlink — git-owned parent protection"
    )

    # The decks/ parent must also be a real dir (never symlinked over)
    decks_dir = conference_dir / "decks"
    assert decks_dir.is_dir()
    assert not decks_dir.is_symlink(), (
        "conference_dir/decks must NOT be a symlink"
    )


# ---------------------------------------------------------------------------
# Group E — Idempotent rerun (D-02e)
# ---------------------------------------------------------------------------

def test_apply_idempotent_already_linked(tmp_path: Path, monkeypatch) -> None:
    """Re-running apply when the symlink already points to the right target
    must yield ACTION_ALREADY_LINKED and leave the symlink unchanged."""
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    corpus_drive = drive_leaf / "storage" / "corpus"
    corpus_drive.mkdir(parents=True)
    (corpus_drive / "data.bin").write_bytes(b"payload")

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "corpus",
                "path": "storage/corpus",
                "storage_class": "backup",
                "sync_profiles": {"public": False, "backup": True},
            },
        ],
    )

    # Pre-create the correct symlink
    local_corpus = conference_dir / "storage" / "corpus"
    local_corpus.parent.mkdir(parents=True, exist_ok=True)
    local_corpus.symlink_to(corpus_drive.resolve())

    config = module.RestoreConfig(
        conference_dir=conference_dir,
        config_path=conference_dir / "config.yaml",
        drive_leaf=drive_leaf,
        drive_leaf_name="demo-drive-raw",
        want_json=False,
    )
    plan = module.build_restore_plan(config)
    result = module.apply_restore_plan(plan)

    assert result.hard_fault is None
    # The entry must be in already_linked, not applied_links
    assert len(result.already_linked) == 1, (
        "Expected 1 already_linked entry"
    )
    assert result.already_linked[0].action == module.ACTION_ALREADY_LINKED
    # The symlink must remain pointing at the correct target
    assert local_corpus.is_symlink()
    assert local_corpus.resolve() == corpus_drive.resolve()


# ---------------------------------------------------------------------------
# Group F — Unsafe-overwrite refusal (D-02f)
# ---------------------------------------------------------------------------

def test_apply_refuses_to_overwrite_real_file(tmp_path: Path, monkeypatch) -> None:
    """A real file at the link location must become a blocker, not be overwritten.

    Per D-04, blockers are exit 0 (not a hard fault).
    """
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    corpus_drive = drive_leaf / "storage" / "corpus"
    corpus_drive.mkdir(parents=True)
    (corpus_drive / "data.bin").write_bytes(b"Drive payload")

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "corpus",
                "path": "storage/corpus",
                "storage_class": "backup",
                "sync_profiles": {"public": False, "backup": True},
            },
        ],
    )

    # Place a real file at the link location (not a directory, not a symlink)
    local_storage = conference_dir / "storage"
    local_storage.mkdir(parents=True, exist_ok=True)
    real_file = local_storage / "corpus"
    real_file.write_bytes(b"local real file")

    config = module.RestoreConfig(
        conference_dir=conference_dir,
        config_path=conference_dir / "config.yaml",
        drive_leaf=drive_leaf,
        drive_leaf_name="demo-drive-raw",
        want_json=False,
    )
    plan = module.build_restore_plan(config)
    result = module.apply_restore_plan(plan)

    assert result.hard_fault is None, "Blocker must not be a hard fault"
    # The real file must still exist (was NOT replaced)
    assert real_file.exists() and not real_file.is_symlink(), (
        "Real file must remain untouched"
    )
    # The blocker must be recorded
    assert len(result.blockers) >= 1, "Expected at least one blocker entry"
    assert any(
        b.action == module.ACTION_UNSAFE_EXISTING_LOCAL for b in result.blockers
    ), "Blocker action must be ACTION_UNSAFE_EXISTING_LOCAL"


def test_apply_refuses_to_overwrite_nonempty_dir(tmp_path: Path, monkeypatch) -> None:
    """A real non-empty directory at the link location must resolve to ACTION_UNSAFE_EXISTING_LOCAL."""
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    corpus_drive = drive_leaf / "storage" / "corpus"
    corpus_drive.mkdir(parents=True)
    (corpus_drive / "data.bin").write_bytes(b"Drive payload")

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "corpus",
                "path": "storage/corpus",
                "storage_class": "backup",
                "sync_profiles": {"public": False, "backup": True},
            },
        ],
    )

    # Place a real non-empty directory at the link location
    local_corpus = conference_dir / "storage" / "corpus"
    local_corpus.mkdir(parents=True, exist_ok=True)
    (local_corpus / "existing_file.txt").write_text("local content", encoding="utf-8")

    config = module.RestoreConfig(
        conference_dir=conference_dir,
        config_path=conference_dir / "config.yaml",
        drive_leaf=drive_leaf,
        drive_leaf_name="demo-drive-raw",
        want_json=False,
    )
    plan = module.build_restore_plan(config)

    # Check via _classify_local_target directly — the dir should be classified unsafe
    action, reason = module._classify_local_target(local_corpus, corpus_drive.resolve())
    assert action == module.ACTION_UNSAFE_EXISTING_LOCAL, (
        f"Non-empty dir must be classified as unsafe, got: {action}"
    )
    assert reason, "Reason must be non-empty"

    # Also verify apply agrees
    result = module.apply_restore_plan(plan)
    assert result.hard_fault is None, "Blocker must not be a hard fault (D-04)"
    assert len(result.blockers) >= 1, "Non-empty dir must be a blocker"
    assert any(
        b.action == module.ACTION_UNSAFE_EXISTING_LOCAL for b in result.blockers
    )


# ---------------------------------------------------------------------------
# Group G — Verifier (D-02g)
# ---------------------------------------------------------------------------

def test_verify_passes_when_all_symlinks_resolve(tmp_path: Path, monkeypatch) -> None:
    """verify_restore_result must return status='passed' when all links resolve."""
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    corpus_drive = drive_leaf / "storage" / "corpus"
    corpus_drive.mkdir(parents=True)
    (corpus_drive / "data.bin").write_bytes(b"payload")

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "corpus",
                "path": "storage/corpus",
                "storage_class": "backup",
                "sync_profiles": {"public": False, "backup": True},
            },
        ],
    )

    # Create a correct symlink
    local_corpus = conference_dir / "storage" / "corpus"
    local_corpus.parent.mkdir(parents=True, exist_ok=True)
    local_corpus.symlink_to(corpus_drive.resolve())

    config = module.RestoreConfig(
        conference_dir=conference_dir,
        config_path=conference_dir / "config.yaml",
        drive_leaf=drive_leaf,
        drive_leaf_name="demo-drive-raw",
        want_json=False,
    )
    plan = module.build_restore_plan(config)

    # Build a PlannedLink representing the created symlink
    created_link = module.PlannedLink(
        family_id="corpus",
        storage_class="backup",
        relative_path=Path("storage/corpus"),
        local_path=local_corpus,
        drive_path=corpus_drive.resolve(),
        is_directory=True,
        action=module.ACTION_ALREADY_LINKED,
        reason="pre-created for test",
    )

    result = module.verify_restore_result([created_link], plan)
    assert result.status == "passed", f"Expected 'passed', got: {result.status}"
    assert list(result.unresolved_targets) == [], (
        f"Expected no unresolved targets, got: {result.unresolved_targets}"
    )
    assert list(result.git_control_violations) == [], (
        f"Expected no git-control violations, got: {result.git_control_violations}"
    )


def test_verify_fails_on_unresolved_target(tmp_path: Path, monkeypatch) -> None:
    """verify_restore_result must return status='failed' for a dangling symlink."""
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "corpus",
                "path": "storage/corpus",
                "storage_class": "backup",
                "sync_profiles": {"public": False, "backup": True},
            },
        ],
    )

    # Create a DANGLING symlink (target does not exist)
    local_corpus = conference_dir / "storage" / "corpus"
    local_corpus.parent.mkdir(parents=True, exist_ok=True)
    nonexistent_drive = drive_leaf / "storage" / "corpus_GONE"
    local_corpus.symlink_to(nonexistent_drive)  # dangling
    assert not nonexistent_drive.exists(), "Drive target must not exist for this test"

    config = module.RestoreConfig(
        conference_dir=conference_dir,
        config_path=conference_dir / "config.yaml",
        drive_leaf=drive_leaf,
        drive_leaf_name="demo-drive-raw",
        want_json=False,
    )
    plan = module.build_restore_plan(config)

    created_link = module.PlannedLink(
        family_id="corpus",
        storage_class="backup",
        relative_path=Path("storage/corpus"),
        local_path=local_corpus,
        drive_path=nonexistent_drive,
        is_directory=True,
        action=module.ACTION_LINK,
        reason="test dangling",
    )

    result = module.verify_restore_result([created_link], plan)
    assert result.status == "failed", (
        f"Expected 'failed' for dangling symlink, got: {result.status}"
    )
    assert len(result.unresolved_targets) >= 1, (
        "Expected at least one unresolved target"
    )


def test_verify_flags_git_control_violation(tmp_path: Path, monkeypatch) -> None:
    """verify_restore_result must flag a symlink whose path is covered by a git-class family."""
    module = load_restore_module()
    conference_dir, drive_leaf = make_restore_workspace(tmp_path)

    # A "git" storage_class family that owns "storage" path
    write_restore_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "storage_control",
                "path": "storage",
                "storage_class": "git",
                "sync_profiles": {"public": False, "backup": False},
            },
        ],
    )

    # Create a Drive target directory
    git_drive_target = drive_leaf / "storage"
    git_drive_target.mkdir(parents=True)

    # Create a symlink at the git-class path
    local_storage_link = conference_dir / "storage_link"
    local_storage_link.symlink_to(git_drive_target.resolve())

    config = module.RestoreConfig(
        conference_dir=conference_dir,
        config_path=conference_dir / "config.yaml",
        drive_leaf=drive_leaf,
        drive_leaf_name="demo-drive-raw",
        want_json=False,
    )
    plan = module.build_restore_plan(config)

    # Build a link entry that has the git-class path as its relative_path
    git_link_entry = module.PlannedLink(
        family_id="storage_control",
        storage_class="git",
        relative_path=Path("storage"),
        local_path=conference_dir / "storage",
        drive_path=git_drive_target.resolve(),
        is_directory=True,
        action=module.ACTION_LINK,
        reason="test git-control violation",
    )

    # Create the symlink at the git-controlled path for verify to inspect
    local_git_path = conference_dir / "storage"
    if not local_git_path.exists() and not local_git_path.is_symlink():
        local_git_path.symlink_to(git_drive_target.resolve())

    result = module.verify_restore_result([git_link_entry], plan)
    assert len(result.git_control_violations) >= 1, (
        f"Expected git-control violation for git-class path, got: {result.git_control_violations}"
    )
    assert result.status == "failed", (
        f"Expected 'failed' due to git-control violation, got: {result.status}"
    )


def _drive_leaf_listable(path: Path) -> bool:
    try:
        os.listdir(path)
    except PermissionError:
        return False
    except OSError:
        return True
    return True


# ---------------------------------------------------------------------------
# Gated ICLR 2026 smoke test (exactly one; skips when Drive is absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "conferences" / "iclr2026" / "storage" / "download-corpus" / "download_corpus.json").exists(),
    reason="ICLR 2026 on-disk storage lanes not present in this checkout; --verify resolves symlinks against them",
)
@pytest.mark.skipif(
    ICLR_DRIVE_LEAF is None,
    reason="conferences/iclr2026/DRIVE.md names no mounted Drive path in this checkout",
)
@pytest.mark.skipif(
    ICLR_DRIVE_LEAF is not None and not ICLR_DRIVE_LEAF.is_dir(),
    reason="ICLR 2026 Drive leaf not mounted; skipping live smoke test",
)
@pytest.mark.skipif(
    ICLR_DRIVE_LEAF is not None and ICLR_DRIVE_LEAF.is_dir() and not _drive_leaf_listable(ICLR_DRIVE_LEAF),
    reason="ICLR 2026 Drive leaf is mounted but unreadable in this environment (sandbox PermissionError); --verify would see an empty tree",
)
def test_iclr_verify_smoke_passes_when_drive_mounted(tmp_path: Path) -> None:
    """Run --verify against the live ICLR 2026 Drive leaf and assert it passes.

    This test is the ONLY test in this file that references the real mounted
    Drive path. It is skipped on CI and unmounted dev machines.

    The receipt is written to tmp_path (ephemeral) — the committed proof receipt
    is produced by Plan 03-03, not here.
    """
    module = load_restore_module()
    receipt_path = tmp_path / "smoke-receipt.json"

    exit_code = module.main([
        "iclr2026",
        "--verify",
        "--receipt-path",
        str(receipt_path),
    ])

    assert exit_code == 0, (
        f"--verify against live Drive must exit 0, got {exit_code}. "
        f"Check receipt: {receipt_path}"
    )

    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)

    assert receipt["verification"]["status"] == "passed", (
        f"Verification status must be 'passed', got: {receipt['verification']['status']}"
    )
    assert receipt["verification"]["git_control_violations"] == [], (
        "Expected 0 git-control violations in live verify"
    )
    assert receipt["verification"]["links_checked"] >= 4000, (
        f"Expected >= 4000 links checked (live run showed 4378), "
        f"got: {receipt['verification']['links_checked']}"
    )
