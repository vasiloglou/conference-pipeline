"""TDD contract coverage for the fixed CVPR r6 artifact authority.

These tests deliberately use a fixture contract.  Production owners may not
replace the CVPR workspace, run identity, or artifact paths through a CLI
argument; only local unit tests receive an injected fixture root.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from pathlib import Path

import pytest

import conference_pipeline.acquisition.cvpr_r6_artifact_contract as artifact_contract_module
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_ARTIFACT_PATHS,
    R6_RUN_ID,
    R6_STAGE_ID,
    R6ArtifactContract,
    R6ArtifactContractError,
    archive_r6_audio_checkpoint_relative_path,
    read_r6_audio_checkpoint,
    read_r6_audio_checkpoint_archive,
    r6_active_workspace_lease_descriptor_count,
    validate_r6_manifest_view,
    validate_r6_registry_view,
    write_r6_audio_checkpoint,
    write_r6_audio_checkpoint_archive,
    write_r6_cli_decision,
    write_r6_continuation_diagnostic,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _continuation_diagnostic_payload() -> dict[str, object]:
    """Return the entire closed diagnostic shape with harmless test digests."""

    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_continuation_diagnostic",
        "reason_code": "worker_no_progress",
        "recovery_inventory_sha256": "a" * 64,
        "audio_policy_digest": "b" * 64,
        "source_scope_sha256": "c" * 64,
        "record_count": 529,
        "checkpoint_sha256": "d" * 64,
        "lineage_state": "successor",
        "historical_checkpoint_lineage_sha256": "e" * 64,
        "historical_checkpoint_archive_sha256": "f" * 64,
    }


def test_continuation_diagnostic_writer_accepts_only_the_closed_bound_shape(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    contract = R6ArtifactContract.for_test_fixture(workspace)
    payload = _continuation_diagnostic_payload()

    write_r6_continuation_diagnostic(contract=contract, payload=payload)
    assert json.loads(contract.path("continuation_diagnostic").read_text(encoding="utf-8")) == payload

    malformed = _continuation_diagnostic_payload()
    malformed["checkpoint_sha256"] = "D" * 64
    with pytest.raises(R6ArtifactContractError, match="digest"):
        write_r6_continuation_diagnostic(contract=contract, payload=malformed)

    malformed = _continuation_diagnostic_payload()
    malformed["historical_checkpoint_archive_sha256"] = "f" * 63
    with pytest.raises(R6ArtifactContractError, match="successor lineage"):
        write_r6_continuation_diagnostic(contract=contract, payload=malformed)

    malformed = _continuation_diagnostic_payload()
    malformed["caller_payload"] = "must-not-publish"
    with pytest.raises(R6ArtifactContractError, match="shape"):
        write_r6_continuation_diagnostic(contract=contract, payload=malformed)

    absent = _continuation_diagnostic_payload()
    absent["lineage_state"] = "absent"
    absent["historical_checkpoint_lineage_sha256"] = None
    absent["historical_checkpoint_archive_sha256"] = None
    write_r6_continuation_diagnostic(contract=contract, payload=absent)


def test_cli_decision_writer_rejects_open_or_contradictory_outcomes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    contract = R6ArtifactContract.for_test_fixture(workspace)
    unavailable = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "phase": "pre_admission",
        "reason_code": "admission_invalid",
        "exit_code": 1,
        "binding_state": "unavailable",
        "recovery_inventory_sha256": None,
        "audio_policy_digest": None,
        "source_scope_sha256": None,
        "record_count": None,
        "checkpoint_sha256": None,
        "lineage_state": None,
        "historical_checkpoint_lineage_sha256": None,
        "historical_checkpoint_archive_sha256": None,
    }
    write_r6_cli_decision(contract=contract, payload=unavailable)
    assert json.loads(contract.path("cli_decision").read_text(encoding="utf-8")) == unavailable

    malformed = dict(unavailable)
    malformed["exception"] = "not-a-wire-field"
    with pytest.raises(R6ArtifactContractError, match="shape"):
        write_r6_cli_decision(contract=contract, payload=malformed)
    malformed = dict(unavailable)
    malformed["checkpoint_sha256"] = "a" * 64
    with pytest.raises(R6ArtifactContractError, match="unavailable"):
        write_r6_cli_decision(contract=contract, payload=malformed)


def _temporary_production_repository(tmp_path: Path) -> tuple[Path, Path]:
    """Create the lexical production layout without touching real storage."""

    repository = tmp_path / "repo"
    workspace = repository / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    return repository, workspace


def _write_fixed_fixture_downloader(repository: Path) -> Path:
    """Create a local executable only for descriptor-capability tests."""

    executable = repository / ".venv" / "bin" / "yt-dlp"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\necho fixture-version\n", encoding="utf-8")
    executable.chmod(0o700)
    return executable


def test_production_contract_accepts_a_normal_lexical_repository_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, workspace = _temporary_production_repository(tmp_path)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)

    contract = R6ArtifactContract.production()

    assert contract.workspace == workspace
    assert contract.path("raw_audio") == workspace / R6_ARTIFACT_PATHS["raw_audio"]


@pytest.mark.parametrize("redirect", ("workspace", "ancestor", "repository"))
def test_production_contract_rejects_redirected_anchor_before_external_sentinel_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, redirect: str
) -> None:
    """Production must reject a redirect before a contract grants it authority."""

    repository, workspace = _temporary_production_repository(tmp_path)
    external = tmp_path / "external"
    external_workspace = external / "conferences" / "cvpr2026"
    external_workspace.mkdir(parents=True)
    sentinel = external_workspace / "sentinel.bin"
    sentinel.write_bytes(b"must-not-be-read")

    repository_root = repository
    if redirect == "workspace":
        workspace.rmdir()
        workspace.symlink_to(external_workspace, target_is_directory=True)
    elif redirect == "ancestor":
        (repository / "conferences" / "cvpr2026").rmdir()
        (repository / "conferences").rmdir()
        (repository / "conferences").symlink_to(external / "conferences", target_is_directory=True)
    else:
        redirected_repository = tmp_path / "repo-link"
        redirected_repository.symlink_to(repository, target_is_directory=True)
        repository_root = redirected_repository

    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository_root)
    original_read_bytes = Path.read_bytes

    def no_external_read(path: Path) -> bytes:
        if path == sentinel:
            raise AssertionError("external sentinel must remain unread")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", no_external_read)
    with pytest.raises(R6ArtifactContractError, match="anchor|workspace|symlink"):
        R6ArtifactContract.production()
    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)
    assert sentinel.read_bytes() == b"must-not-be-read"


def test_production_contract_revalidates_the_retained_workspace_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, workspace = _temporary_production_repository(tmp_path)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    external = tmp_path / "external"
    external.mkdir()
    workspace.rmdir()
    workspace.symlink_to(external, target_is_directory=True)

    with pytest.raises(R6ArtifactContractError, match="anchor|workspace|symlink"):
        contract.ensure_workspace_trusted()


def test_fixture_contract_never_resolves_a_direct_workspace_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    external.mkdir()
    workspace.symlink_to(external, target_is_directory=True)

    with pytest.raises(R6ArtifactContractError, match="fixture workspace.*symlink"):
        R6ArtifactContract.for_test_fixture(workspace)


def test_r6_artifact_contract_is_the_one_fixed_path_and_identity_authority(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    contract = R6ArtifactContract.for_test_fixture(workspace)

    assert contract.workspace == workspace.resolve()
    assert contract.run_id == R6_RUN_ID
    assert contract.stage_id == R6_STAGE_ID
    assert contract.path("recovery_inventory") == (
        workspace / R6_ARTIFACT_PATHS["recovery_inventory"]
    )
    assert contract.path("final_receipt") == workspace / R6_ARTIFACT_PATHS["final_receipt"]
    assert contract.r6_root == workspace / R6_ARTIFACT_PATHS["r6_root"]
    assert contract.audio_root == workspace / R6_ARTIFACT_PATHS["raw_audio"]
    assert contract.material_payload_root == workspace / R6_ARTIFACT_PATHS["material_payloads"]
    assert contract.speech_root == workspace / R6_ARTIFACT_PATHS["speech_text"]
    assert contract.paper_root == workspace / R6_ARTIFACT_PATHS["paper_text"]
    assert contract.ocr_root == workspace / R6_ARTIFACT_PATHS["slide_ocr"]


@pytest.mark.parametrize(
    "candidate",
    (
        Path("storage/transcription-ocr/cvpr2026/r6/../other.json"),
        Path("storage/workflow-runs/other/stage-05-transcription-ocr-r6.json"),
        Path("/tmp/escape.json"),
    ),
)
def test_r6_artifact_contract_rejects_escaped_or_replaced_paths(
    tmp_path: Path, candidate: Path
) -> None:
    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    with pytest.raises(R6ArtifactContractError):
        contract.require_path("final_receipt", candidate)


def test_r6_artifact_contract_rejects_an_alternate_workspace_without_fixture_injection(
    tmp_path: Path,
) -> None:
    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    with pytest.raises(R6ArtifactContractError):
        contract.require_workspace(tmp_path / "conferences" / "other")
    with pytest.raises(R6ArtifactContractError, match="fixed CVPR workspace"):
        R6ArtifactContract(tmp_path / "conferences" / "other")


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_r6_contract_validates_derived_manifest_and_registry_views() -> None:
    manifest = json.loads(
        (REPO_ROOT / "conferences" / "cvpr2026" / "storage" / "storage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    registry = json.loads(
        (REPO_ROOT / "src" / "conference_pipeline" / "stage_owner_registry.json").read_text(
            encoding="utf-8"
        )
    )
    validate_r6_manifest_view(manifest)
    validate_r6_registry_view(registry)


def test_every_r6_path_owner_imports_the_shared_contract_without_old_receipt_spellings() -> None:
    owners = (
        "scripts/run_cvpr2026_r6_media_recovery.py",
        "src/conference_pipeline/acquisition/_cvpr_r6_media_owner_impl.py",
        "scripts/run_cvpr2026_r6_transcription_ocr.py",
        "scripts/acquire_cvpr2026_r6_materials.py",
        "scripts/estimate_cvpr2026_r6_costs.py",
        "src/conference_pipeline/acquisition/cvpr_r6_media_recovery.py",
        "src/conference_pipeline/acquisition/cvpr_r6_material_index.py",
        "src/conference_pipeline/acquisition/public_material_acquisition.py",
        "src/conference_pipeline/r6_transcription_ocr_execution.py",
        "src/conference_pipeline/stage_contracts.py",
    )
    old_spellings = (
        "audio-recovery-receipt.json",
        "material-acquisition-receipt.json",
        "r6-cost-estimate-receipt.json",
        "budget-ledger.json",
        "provider-receipts.json",
        "r6-processing-inventory.json",
    )
    for owner in owners:
        source = (REPO_ROOT / owner).read_text(encoding="utf-8")
        # The public runner is deliberately a capability-free CLI shell.  Its
        # private fixed-owner implementation is the contract-owning source.
        if owner != "scripts/run_cvpr2026_r6_media_recovery.py":
            assert "cvpr_r6_artifact_contract" in source, owner
        assert not any(spelling in source for spelling in old_spellings), owner


def test_contract_rejects_drifted_static_views_before_any_owner_can_run() -> None:
    manifest = {"artifact_families": []}
    with pytest.raises(R6ArtifactContractError, match="manifest view drifted"):
        validate_r6_manifest_view(manifest)
    with pytest.raises(R6ArtifactContractError, match="lacks r6 binding"):
        validate_r6_registry_view({"stage_bindings": {}})


def test_r6_audio_checkpoint_archive_is_one_fixed_contract_family(tmp_path: Path) -> None:
    """The exceptional historical checkpoint has one digest-addressed archive home."""

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        archive_r6_audio_checkpoint_relative_path,
        r6_audio_checkpoint_archive_root_relative_path,
    )

    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    digest = "a" * 64
    archive_root = Path("storage/transcription-ocr/cvpr2026/r6/audio-recovery-archive")

    assert r6_audio_checkpoint_archive_root_relative_path() == archive_root
    assert archive_r6_audio_checkpoint_relative_path(digest) == archive_root / f"{digest}.json"
    assert contract.path("audio_recovery_archive_root") == contract.workspace / archive_root
    with pytest.raises(R6ArtifactContractError):
        archive_r6_audio_checkpoint_relative_path("not-a-digest")


def test_r6_archive_writer_and_lineage_reader_use_the_one_fixed_digest_leaf(
    tmp_path: Path,
) -> None:
    """The only archive interface derives and rechecks the leaf from bytes."""

    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    raw = b'{"historical":"checkpoint"}'
    digest = hashlib.sha256(raw).hexdigest()

    written = write_r6_audio_checkpoint_archive(
        contract=contract,
        original_sha256=digest,
        raw=raw,
    )

    assert written == archive_r6_audio_checkpoint_relative_path(digest)
    assert read_r6_audio_checkpoint_archive(contract=contract, original_sha256=digest) == raw

    # A crashed caller may safely repeat the exact immutable publication.
    assert write_r6_audio_checkpoint_archive(
        contract=contract,
        original_sha256=digest,
        raw=raw,
    ) == written
    assert read_r6_audio_checkpoint_archive(contract=contract, original_sha256=digest) == raw


def test_r6_archive_writer_rejects_existing_different_bytes_or_hard_link(
    tmp_path: Path,
) -> None:
    """Idempotency accepts exact bytes only, never an alias or stale body."""

    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    raw = b'{"historical":"checkpoint"}'
    digest = hashlib.sha256(raw).hexdigest()
    archive_root = contract.audio_checkpoint_archive_root
    archive_root.mkdir(parents=True)
    leaf = archive_root / f"{digest}.json"

    leaf.write_bytes(b"different")
    with pytest.raises(R6ArtifactContractError, match="different bytes"):
        write_r6_audio_checkpoint_archive(
            contract=contract,
            original_sha256=digest,
            raw=raw,
        )

    leaf.unlink()
    leaf.write_bytes(raw)
    alias = archive_root / "alias.json"
    alias.hardlink_to(leaf)
    with pytest.raises(R6ArtifactContractError, match="link count"):
        write_r6_audio_checkpoint_archive(
            contract=contract,
            original_sha256=digest,
            raw=raw,
        )
    with pytest.raises(R6ArtifactContractError, match="link count"):
        read_r6_audio_checkpoint_archive(contract=contract, original_sha256=digest)


@pytest.mark.parametrize("poisoned", ("root", "intermediate", "leaf"))
def test_r6_archive_refuses_root_intermediate_or_leaf_symlink_before_external_io(
    tmp_path: Path, poisoned: str
) -> None:
    """No archive write may traverse a fixture-controlled link outside the contract."""

    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    raw = b'{"historical":"checkpoint"}'
    digest = hashlib.sha256(raw).hexdigest()
    archive_root = contract.audio_checkpoint_archive_root
    external = tmp_path / "external-sentinel"
    external.mkdir()
    sentinel = external / "sentinel.json"
    sentinel.write_bytes(b"must-not-change")

    if poisoned == "root":
        archive_root.parent.mkdir(parents=True)
        archive_root.symlink_to(external, target_is_directory=True)
    elif poisoned == "intermediate":
        contract.workspace.mkdir(parents=True)
        (contract.workspace / "storage").symlink_to(external, target_is_directory=True)
    else:
        archive_root.mkdir(parents=True)
        (archive_root / f"{digest}.json").symlink_to(sentinel)

    with pytest.raises(R6ArtifactContractError, match="archive"):
        write_r6_audio_checkpoint_archive(
            contract=contract,
            original_sha256=digest,
            raw=raw,
        )

    assert sentinel.read_bytes() == b"must-not-change"
    assert not (external / f"{digest}.json").exists()


@pytest.mark.parametrize("poisoned", ("root", "intermediate", "leaf"))
def test_r6_archive_lineage_reader_refuses_every_symlink_endpoint(
    tmp_path: Path, poisoned: str
) -> None:
    """A successor cannot consume archive bytes through a redirected lineage path."""

    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    raw = b'{"historical":"checkpoint"}'
    digest = hashlib.sha256(raw).hexdigest()
    archive_root = contract.audio_checkpoint_archive_root
    external = tmp_path / "external-sentinel"
    external.mkdir()
    sentinel = external / "sentinel.json"
    sentinel.write_bytes(b"outside-contract")

    if poisoned == "root":
        archive_root.parent.mkdir(parents=True)
        archive_root.symlink_to(external, target_is_directory=True)
    elif poisoned == "intermediate":
        contract.workspace.mkdir(parents=True)
        (contract.workspace / "storage").symlink_to(external, target_is_directory=True)
    else:
        archive_root.mkdir(parents=True)
        (archive_root / f"{digest}.json").symlink_to(sentinel)

    with pytest.raises(R6ArtifactContractError, match="archive"):
        read_r6_audio_checkpoint_archive(contract=contract, original_sha256=digest)

    assert sentinel.read_bytes() == b"outside-contract"


@pytest.mark.parametrize("poisoned", ("root", "parent", "leaf"))
def test_r6_raw_audio_reader_refuses_every_link_before_consuming_external_bytes(
    tmp_path: Path, poisoned: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw audio is a separate no-follow boundary, not an archive alias."""

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        read_r6_raw_audio_payload,
    )

    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    raw_root = contract.audio_root
    leaf = raw_root / "nested" / "fixture.mp3"
    external = tmp_path / "external-sentinel"
    external.mkdir()
    sentinel = external / "fixture.mp3"
    sentinel.write_bytes(b"must-not-be-read")

    if poisoned == "root":
        raw_root.parent.mkdir(parents=True)
        raw_root.symlink_to(external, target_is_directory=True)
    elif poisoned == "parent":
        raw_root.mkdir(parents=True)
        (raw_root / "nested").symlink_to(external, target_is_directory=True)
    else:
        leaf.parent.mkdir(parents=True)
        leaf.symlink_to(sentinel)

    original_read_bytes = Path.read_bytes

    def no_external_read(path: Path) -> bytes:
        if path.resolve() == sentinel.resolve():
            raise AssertionError("external raw-audio sentinel must not be consumed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", no_external_read)
    declared = leaf.relative_to(contract.workspace).as_posix()

    with pytest.raises(R6ArtifactContractError, match="raw audio"):
        read_r6_raw_audio_payload(contract=contract, local_media_path=declared)

    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)
    assert sentinel.read_bytes() == b"must-not-be-read"


def test_r6_raw_audio_boundary_prepares_and_reads_only_a_regular_contract_leaf(
    tmp_path: Path,
) -> None:
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        prepare_r6_raw_audio_target,
        read_r6_raw_audio_payload,
    )

    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    target = prepare_r6_raw_audio_target(
        contract=contract, item_id="r5:caption:0001", video_id="abcDEF12345"
    )
    target.write_bytes(b"safe-fixture-audio")
    payload, relative = read_r6_raw_audio_payload(
        contract=contract, local_media_path=str(target)
    )

    assert payload == b"safe-fixture-audio"
    assert relative == target.relative_to(contract.workspace)

    for unsafe in (
        "../outside.mp3",
        "/tmp/outside.mp3",
        "https://example.invalid/audio.mp3",
        "storage/transcription-ocr/cvpr2026/r6/raw-audio/*.mp3",
    ):
        with pytest.raises(R6ArtifactContractError, match="raw audio"):
            read_r6_raw_audio_payload(contract=contract, local_media_path=unsafe)


def _redirect_production_component_after_lease(
    *,
    tmp_path: Path,
    repository: Path,
    redirect: str,
) -> Path:
    """Replace exactly one fixed production anchor after a batch lease exists."""

    external_repository = tmp_path / "external-repository"
    external_workspace = external_repository / "conferences" / "cvpr2026"
    external_workspace.mkdir(parents=True)
    if redirect == "workspace":
        original = tmp_path / "original-workspace"
        (repository / "conferences" / "cvpr2026").rename(original)
        (repository / "conferences" / "cvpr2026").symlink_to(
            external_workspace, target_is_directory=True
        )
    elif redirect == "ancestor":
        original = tmp_path / "original-conferences"
        (repository / "conferences").rename(original)
        (repository / "conferences").symlink_to(
            external_repository / "conferences", target_is_directory=True
        )
    elif redirect == "repository":
        original = tmp_path / "original-repository"
        repository.rename(original)
        repository.symlink_to(external_repository, target_is_directory=True)
    else:  # pragma: no cover - parametrized fixture invariant.
        raise AssertionError(f"unexpected redirect target: {redirect}")
    return external_workspace


@pytest.mark.parametrize("redirect", ("workspace", "ancestor", "repository"))
def test_leased_raw_audio_access_revalidates_the_full_production_anchor_before_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, redirect: str
) -> None:
    """A private lease cannot turn a later lexical redirect into raw-byte authority."""

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        prepare_r6_raw_audio_target,
        read_r6_raw_audio_payload,
    )

    repository, workspace = _temporary_production_repository(tmp_path)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    lease = contract._acquire_workspace_lease()
    external_workspace = _redirect_production_component_after_lease(
        tmp_path=tmp_path, repository=repository, redirect=redirect
    )
    external_leaf = external_workspace / R6_ARTIFACT_PATHS["raw_audio"] / "item-video.mp3"
    external_leaf.parent.mkdir(parents=True)
    external_leaf.write_bytes(b"external-sentinel")
    read_calls: list[int] = []
    original_read = artifact_contract_module.os.read

    def poison_raw_reader(descriptor: int, count: int) -> bytes:
        read_calls.append(descriptor)
        raise AssertionError("external raw-audio bytes must remain unread")

    monkeypatch.setattr(artifact_contract_module.os, "read", poison_raw_reader)
    declared = (R6_ARTIFACT_PATHS["raw_audio"] / "item-video.mp3").as_posix()
    with pytest.raises(R6ArtifactContractError, match="anchor|workspace|symlink"):
        read_r6_raw_audio_payload(
            contract=contract, local_media_path=declared, _workspace_lease=lease
        )
    with pytest.raises(R6ArtifactContractError, match="anchor|workspace|symlink"):
        prepare_r6_raw_audio_target(
            contract=contract,
            item_id="new-item",
            video_id="abcDEF12345",
            _workspace_lease=lease,
        )
    monkeypatch.setattr(artifact_contract_module.os, "read", original_read)
    assert read_calls == []
    assert external_leaf.read_bytes() == b"external-sentinel"
    assert not (
        external_workspace
        / R6_ARTIFACT_PATHS["raw_audio"]
        / "new-item-abcDEF12345.mp3"
    ).exists()


@pytest.mark.parametrize("redirect", ("workspace", "ancestor", "repository"))
def test_leased_checkpoint_read_rejects_anchor_redirect_before_external_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, redirect: str
) -> None:
    """Checkpoint control bytes are never reopened through a lexical workspace path."""

    repository, _workspace = _temporary_production_repository(tmp_path)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    write_r6_audio_checkpoint(contract=contract, payload={"status": "running"})
    lease = contract._acquire_workspace_lease()
    external_workspace = _redirect_production_component_after_lease(
        tmp_path=tmp_path, repository=repository, redirect=redirect
    )
    external_checkpoint = external_workspace / R6_ARTIFACT_PATHS["audio_recovery_receipt"]
    external_checkpoint.parent.mkdir(parents=True)
    external_checkpoint.write_bytes(b'{"status":"external"}')
    reads: list[int] = []
    original_read = artifact_contract_module.os.read

    def poison_reader(descriptor: int, count: int) -> bytes:
        reads.append(descriptor)
        raise AssertionError("external checkpoint bytes must remain unread")

    monkeypatch.setattr(artifact_contract_module.os, "read", poison_reader)
    with pytest.raises(R6ArtifactContractError, match="anchor|workspace|symlink"):
        read_r6_audio_checkpoint(contract=contract, _workspace_lease=lease)
    monkeypatch.setattr(artifact_contract_module.os, "read", original_read)

    assert reads == []
    assert external_checkpoint.read_bytes() == b'{"status":"external"}'


def test_leased_raw_audio_access_revalidates_each_normal_production_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounded batch token keeps normal repeated accesses, never trust caching."""

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        prepare_r6_raw_audio_target,
        read_r6_raw_audio_payload,
    )

    repository, _workspace = _temporary_production_repository(tmp_path)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    lease = contract._acquire_workspace_lease()
    opened: list[str] = []
    original_open = artifact_contract_module._open_r6_retained_production_workspace

    def counted_open(anchor):  # type: ignore[no-untyped-def]
        opened.append("workspace")
        return original_open(anchor)

    monkeypatch.setattr(
        artifact_contract_module, "_open_r6_retained_production_workspace", counted_open
    )
    for index in range(2):
        target = prepare_r6_raw_audio_target(
            contract=contract,
            item_id=f"item-{index}",
            video_id=f"abcDEF1234{index}",
            _workspace_lease=lease,
        )
        target.write_bytes(f"audio-{index}".encode("utf-8"))
        payload, relative = read_r6_raw_audio_payload(
            contract=contract,
            local_media_path=target.relative_to(contract.workspace).as_posix(),
            _workspace_lease=lease,
        )
        assert payload == f"audio-{index}".encode("utf-8")
        assert relative == target.relative_to(contract.workspace)

    # Target preparation and publication reads both revalidate every row;
    # correctness takes priority over the former single-walk optimization.
    assert len(opened) == 4


@pytest.mark.parametrize("redirect", ("workspace", "ancestor", "repository"))
def test_scoped_lease_context_holds_original_chain_and_rejects_a_prechild_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, redirect: str
) -> None:
    """A replacement after context acquisition cannot reach a child seam."""

    repository, _workspace = _temporary_production_repository(tmp_path)
    _write_fixed_fixture_downloader(repository)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    lease = contract._acquire_workspace_lease()
    baseline = r6_active_workspace_lease_descriptor_count()

    with contract.workspace_lease_context(lease) as context:
        assert r6_active_workspace_lease_descriptor_count() == baseline + 3
        external_workspace = _redirect_production_component_after_lease(
            tmp_path=tmp_path, repository=repository, redirect=redirect
        )
        sentinel = external_workspace / "outside.txt"
        sentinel.write_bytes(b"must-not-run")
        with pytest.raises(R6ArtifactContractError, match="anchor|workspace|symlink"):
            context.assert_child_dispatch_allowed()
        assert sentinel.read_bytes() == b"must-not-run"

    assert r6_active_workspace_lease_descriptor_count() == baseline
    with pytest.raises(R6ArtifactContractError, match="closed"):
        context.assert_child_dispatch_allowed()


def test_scoped_lease_context_capability_is_descriptor_bound_and_cleanup_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixed child remains descriptor-bound and every handle is revoked."""

    repository, _workspace = _temporary_production_repository(tmp_path)
    _write_fixed_fixture_downloader(repository)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    lease = contract._acquire_workspace_lease()
    baseline = r6_active_workspace_lease_descriptor_count()

    context = contract.workspace_lease_context(lease)
    assert r6_active_workspace_lease_descriptor_count() == baseline + 3
    capability = context.open_fixed_yt_dlp_capability()
    with capability:
        assert capability.command_target.startswith("/dev/fd/")
        assert str(repository) not in capability.command_target
        assert r6_active_workspace_lease_descriptor_count() == baseline + 4
    assert r6_active_workspace_lease_descriptor_count() == baseline + 3
    context.close()
    context.close()
    assert r6_active_workspace_lease_descriptor_count() == baseline
    with pytest.raises(R6ArtifactContractError, match="closed"):
        capability.command_target
    with pytest.raises(R6ArtifactContractError, match="lease"):
        contract.workspace_lease_context(object())


def test_executable_capability_copied_context_close_releases_once_and_revokes_stale_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign ContextVar close cannot strand an inherited executable descriptor."""

    repository, _workspace = _temporary_production_repository(tmp_path)
    _write_fixed_fixture_downloader(repository)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    lease = contract._acquire_workspace_lease()
    baseline = r6_active_workspace_lease_descriptor_count()
    context = contract.workspace_lease_context(lease)
    capability = context.open_fixed_yt_dlp_capability()
    capability.__enter__()

    copied = contextvars.copy_context()
    with pytest.raises(R6ArtifactContractError, match="context|close"):
        copied.run(capability.close)

    # The copied context cannot reset the owner token, but cleanup must still
    # close/release exactly once.  The owner keeps only a stale, fail-closed
    # ContextVar reference until its context itself exits.
    assert r6_active_workspace_lease_descriptor_count() == baseline + 3
    with pytest.raises(R6ArtifactContractError, match="closed"):
        artifact_contract_module.active_r6_repository_executable_capability()
    capability.close()
    context.close()
    context.close()
    assert r6_active_workspace_lease_descriptor_count() == baseline


def test_executable_capability_rejects_double_activation_and_leaf_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The descriptor ticket is one-shot and pins content as well as inode."""

    repository, _workspace = _temporary_production_repository(tmp_path)
    executable = _write_fixed_fixture_downloader(repository)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    lease = contract._acquire_workspace_lease()
    context = contract.workspace_lease_context(lease)
    capability = context.open_fixed_yt_dlp_capability()
    try:
        capability.__enter__()
        with pytest.raises(R6ArtifactContractError, match="active"):
            capability.__enter__()
        executable.write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
        executable.chmod(0o700)
        with pytest.raises(R6ArtifactContractError, match="downloader|changed"):
            capability.assert_dispatch_allowed()
    finally:
        capability.close()
        context.close()


def test_executable_capability_close_releases_accounting_when_descriptor_close_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close failures cannot strand global accounting or make a ticket reusable."""

    repository, _workspace = _temporary_production_repository(tmp_path)
    _write_fixed_fixture_downloader(repository)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    lease = contract._acquire_workspace_lease()
    baseline = r6_active_workspace_lease_descriptor_count()
    context = contract.workspace_lease_context(lease)
    capability = context.open_fixed_yt_dlp_capability()
    descriptor = capability.descriptor
    real_close = artifact_contract_module.os.close

    def failing_capability_close(candidate: int) -> None:
        if candidate == descriptor:
            raise OSError("controlled close failure")
        real_close(candidate)

    monkeypatch.setattr(artifact_contract_module.os, "close", failing_capability_close)
    with pytest.raises(R6ArtifactContractError, match="close"):
        capability.close()
    assert r6_active_workspace_lease_descriptor_count() == baseline + 3
    with pytest.raises(R6ArtifactContractError, match="closed"):
        capability.assert_dispatch_allowed()
    # The mock prevented the kernel close; clean up the disposable descriptor
    # explicitly before the surrounding context closes its independent FDs.
    real_close(descriptor)
    context.close()
    assert r6_active_workspace_lease_descriptor_count() == baseline


def test_executable_capability_context_exit_releases_after_raised_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A body exception still tears down the one active ContextVar ticket."""

    repository, _workspace = _temporary_production_repository(tmp_path)
    _write_fixed_fixture_downloader(repository)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    lease = contract._acquire_workspace_lease()
    baseline = r6_active_workspace_lease_descriptor_count()
    with pytest.raises(RuntimeError, match="body"):
        with contract.workspace_lease_context(lease) as context:
            with context.open_fixed_yt_dlp_capability():
                raise RuntimeError("body")
    assert r6_active_workspace_lease_descriptor_count() == baseline


def test_scoped_lease_context_opens_the_production_chain_once_for_its_owner_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner context has one retained chain, not a cached lexical path."""

    repository, _workspace = _temporary_production_repository(tmp_path)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    lease = contract._acquire_workspace_lease()
    opens: list[str] = []
    original_open = artifact_contract_module._open_r6_retained_production_workspace_chain

    def counted_open(anchor):  # type: ignore[no-untyped-def]
        opens.append("chain")
        return original_open(anchor)

    monkeypatch.setattr(
        artifact_contract_module, "_open_r6_retained_production_workspace_chain", counted_open
    )
    with contract.workspace_lease_context(lease) as context:
        context.assert_child_dispatch_allowed()
        context.assert_child_dispatch_allowed()

    assert opens == ["chain"]


def test_active_owner_context_does_not_suppress_per_access_raw_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holding child authority never grants raw payload path authority."""

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        prepare_r6_raw_audio_target,
        read_r6_raw_audio_payload,
    )

    repository, _workspace = _temporary_production_repository(tmp_path)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    lease = contract._acquire_workspace_lease()
    calls: list[str] = []
    original_open = artifact_contract_module._open_r6_retained_production_workspace

    def counted_open(anchor):  # type: ignore[no-untyped-def]
        calls.append("raw-revalidation")
        return original_open(anchor)

    monkeypatch.setattr(
        artifact_contract_module, "_open_r6_retained_production_workspace", counted_open
    )
    with contract.workspace_lease_context(lease):
        target = prepare_r6_raw_audio_target(
            contract=contract,
            item_id="item",
            video_id="abcDEF12345",
            _workspace_lease=lease,
        )
        target.write_bytes(b"audio")
        payload, _relative = read_r6_raw_audio_payload(
            contract=contract,
            local_media_path=target.relative_to(contract.workspace).as_posix(),
            _workspace_lease=lease,
        )
    assert payload == b"audio"
    assert calls == ["raw-revalidation", "raw-revalidation"]
