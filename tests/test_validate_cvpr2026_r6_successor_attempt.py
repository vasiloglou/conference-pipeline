"""Offline security coverage for the one-shot r6 successor validator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import R6ArtifactContract


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_cvpr2026_r6_successor_attempt.py"


def _module():
    spec = importlib.util.spec_from_file_location("r6_successor_attempt", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_context(tmp_path: Path, *, verifier: Any = None):
    module = _module()
    repository = tmp_path / "repository"
    phase = repository.joinpath(*module.PHASE_COMPONENTS)
    phase.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    contract = R6ArtifactContract.for_test_fixture(workspace)
    context = module.ValidationContext(
        contract=contract,
        repository=repository,
        stage_verifier=verifier or module.verify_stage_contract,
    )
    return module, context, phase, workspace


def _facts(module, *, state: str = "absent") -> dict[str, Any]:
    return {
        "predecessor_sha256": module.DECISION_SHA256,
        "predecessor_byte_length": module.DECISION_SIZE,
        "recovery_inventory_sha256": "1" * 64,
        "audio_authority_sha256": "2" * 64,
        "ordered_identity_sha256": "3" * 64,
        "audio_authority_row_count": 2002,
        "baseline_effective_terminal_records": 529,
        "baseline_acquired_records": 0,
        "successor_namespace_state": state,
    }


def _evidence(module, *, state: str = "absent"):
    return module.BindingEvidence(
        facts=_facts(module, state=state),
        inventory={},
        authority=SimpleNamespace(row_count=2002),
        lineage=module.SuccessorLineage(
            state,
            module.DECISION_SHA256 if state == "complete" else None,
            "4" * 64 if state == "complete" else None,
        ),
    )


def _write_contract_json(context, key: str, value: Any) -> Path:
    path = context.contract.path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _write_control(module, context, evidence) -> bytes:
    raw = module._control_text(evidence.facts)
    module._write_control_exclusive(context, raw)
    return raw


def _publish_lineage(module, context) -> Path:
    root = context.contract.path("cli_decision_successor_root")
    archive = root / "archive"
    archive.mkdir(parents=True)
    archive_name = module.DECISION_SHA256 + ".json"
    (archive / archive_name).write_bytes(module._legacy_admission.LEGACY_V1_BYTES)
    admission = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_successor_admission",
        "r6_run_id": module.R6_RUN_ID,
        "predecessor_relation": "immutable_cli_decision_successor",
        "predecessor_sha256": module.DECISION_SHA256,
        "archive_name": archive_name,
        "admission_schema_version": 1,
    }
    admission_raw = module._closed_json(admission)
    (root / "admission.json").write_bytes(admission_raw)
    claim = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_successor_claim",
        "r6_run_id": module.R6_RUN_ID,
        "predecessor_relation": "immutable_cli_decision_successor",
        "predecessor_sha256": module.DECISION_SHA256,
        "admission_schema_version": 1,
        "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
    }
    (root / "claim.json").write_bytes(module._closed_json(claim))
    return root


def test_ordered_identity_is_canonical_and_rejects_duplicate_pairs() -> None:
    module = _module()
    rows = [{"r5_record_id": str(index), "video_id": f"v{index}"} for index in range(2002)]
    digest = module._ordered_identity(rows)
    assert len(digest) == 64
    rows[-1] = dict(rows[0])
    with pytest.raises(module.ValidationError, match="cardinality"):
        module._ordered_identity(rows)


def test_storage_manifest_reader_uses_the_fixed_workspace_relative_leaf(tmp_path: Path) -> None:
    module, context, _phase, workspace = _fixture_context(tmp_path)
    path = workspace / "storage" / "storage_manifest.json"
    path.parent.mkdir()
    path.write_text('{"schema_version":1}\n', encoding="utf-8")
    assert module._read_workspace_relative_json(
        context.contract,
        Path("storage/storage_manifest.json"),
        label="storage_manifest",
    ) == {"schema_version": 1}


def test_control_parser_requires_the_exact_closed_key_set() -> None:
    module = _module()
    with pytest.raises(module.ValidationError, match="key set"):
        module._parse_control(b"---\nphase: 05\n---\n")


@pytest.mark.parametrize(
    "mutation",
    (
        lambda raw: raw[:-1],
        lambda raw: raw + b"\n",
        lambda raw: raw.replace(b'"reason_code": "admission_invalid"', b'"reason_code": "owner_import_failed"'),
        lambda raw: raw.replace(b'  "audio_policy_digest"', b' "audio_policy_digest"'),
        lambda raw: raw.replace(b'  "audio_policy_digest": null,\n  "binding_state"', b'  "binding_state": "unavailable",\n  "audio_policy_digest"'),
        lambda raw: raw + b"{}",
    ),
    ids=("truncated", "newline", "field", "whitespace", "key-order", "extension"),
)
def test_shared_legacy_predicate_admits_only_the_pinned_bytes(mutation) -> None:
    module = _module()
    raw = module._legacy_admission.LEGACY_V1_BYTES
    assert len(raw) == 483
    assert hashlib.sha256(raw).hexdigest() == module.DECISION_SHA256
    assert module._legacy_admission.is_exact_legacy_v1(raw)
    assert not module._legacy_admission.is_exact_legacy_v1(mutation(raw))
    assert not module._legacy_admission.is_exact_legacy_v1(bytearray(raw))


def test_control_write_handles_partial_writes_and_reopens_exact_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, context, _phase, _workspace = _fixture_context(tmp_path)
    evidence = _evidence(module)
    original_write = module.os.write
    writes: list[int] = []

    def partial_write(descriptor: int, payload: Any) -> int:
        limited = payload[: max(1, len(payload) // 3)]
        written = original_write(descriptor, limited)
        writes.append(written)
        return written

    monkeypatch.setattr(module.os, "write", partial_write)
    raw = _write_control(module, context, evidence)
    assert len(writes) > 1
    assert module._read_control_raw(context) == raw
    assert module._read_control(context, evidence)["no_retry"] == "true"


def test_zero_length_control_write_fails_without_retry_or_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, context, phase, _workspace = _fixture_context(tmp_path)
    evidence = _evidence(module)
    monkeypatch.setattr(module.os, "write", lambda _descriptor, _payload: 0)
    with pytest.raises(module.ValidationError, match="short control write"):
        module._write_control_exclusive(context, module._control_text(evidence.facts))
    assert (phase / module.CONTROL_NAME).exists()
    with pytest.raises(module.ValidationError, match="existing control"):
        module._write_control_exclusive(context, module._control_text(evidence.facts))


@pytest.mark.parametrize("key", tuple(_module().CONTROL_KEYS))
def test_every_control_fact_and_constant_is_byte_sealed(tmp_path: Path, key: str) -> None:
    module, context, phase, _workspace = _fixture_context(tmp_path)
    evidence = _evidence(module)
    raw = module._control_text(evidence.facts)
    line = next(item for item in raw.splitlines(keepends=True) if item.startswith(key.encode() + b":"))
    replacement = key.encode() + b": tampered\n"
    (phase / module.CONTROL_NAME).write_bytes(raw.replace(line, replacement, 1))
    with pytest.raises(module.ValidationError, match="control bytes"):
        module._read_control(context, evidence)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda raw: raw[:-1],
        lambda raw: raw + b"\n",
        lambda raw: raw.replace(b"phase:", b"extra: x\nphase:", 1),
        lambda raw: raw.replace(b"phase:", b"phase_renamed:", 1),
        lambda raw: raw.replace(b"phase: 05", b"phase:  05", 1),
    ),
    ids=("truncated", "extension", "extra-key", "missing-key", "whitespace"),
)
def test_control_format_mutations_fail_exact_readback(tmp_path: Path, mutator) -> None:
    module, context, phase, _workspace = _fixture_context(tmp_path)
    evidence = _evidence(module)
    raw = module._control_text(evidence.facts)
    (phase / module.CONTROL_NAME).write_bytes(mutator(raw))
    with pytest.raises(module.ValidationError, match="control bytes"):
        module._read_control(context, evidence)


def test_planning_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    module = _module()
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (repository / ".planning").symlink_to(outside, target_is_directory=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = module.ValidationContext(
        contract=R6ArtifactContract.for_test_fixture(workspace),
        repository=repository,
    )
    with pytest.raises(module.ValidationError):
        module._control_present(context)


def test_planning_ancestor_rename_race_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, context, phase, _workspace = _fixture_context(tmp_path)
    evidence = _evidence(module)
    (phase / module.CONTROL_NAME).write_bytes(module._control_text(evidence.facts))
    original = module._read_regular_at
    moved = phase.with_name(phase.name + "-moved")

    def swap_after_read(parent_fd: int, name: str) -> bytes:
        raw = original(parent_fd, name)
        os.rename(phase, moved)
        phase.mkdir()
        return raw

    monkeypatch.setattr(module, "_read_regular_at", swap_after_read)
    with pytest.raises(module.ValidationError, match="ancestor"):
        module._read_control_raw(context)


def test_complete_successor_lineage_is_exact_and_descriptor_validated(tmp_path: Path) -> None:
    module, context, _phase, _workspace = _fixture_context(tmp_path)
    root = _publish_lineage(module, context)
    lineage = module._validate_successor_lineage(context.contract)
    assert lineage.state == "complete"
    assert lineage.predecessor_sha256 == module.DECISION_SHA256
    assert {path.name for path in root.iterdir()} == {"archive", "admission.json", "claim.json"}


@pytest.mark.parametrize(
    "variant",
    (
        "missing-claim",
        "extra-root",
        "extra-archive",
        "archive-symlink",
        "archive-bytes",
        "admission-bytes",
        "claim-bytes",
        "admission-directory",
    ),
)
def test_partial_redirected_or_mutated_successor_lineage_fails_closed(
    tmp_path: Path,
    variant: str,
) -> None:
    module, context, _phase, _workspace = _fixture_context(tmp_path)
    root = _publish_lineage(module, context)
    digest_leaf = root / "archive" / f"{module.DECISION_SHA256}.json"
    if variant == "missing-claim":
        (root / "claim.json").unlink()
    elif variant == "extra-root":
        (root / "unexpected").write_text("x", encoding="utf-8")
    elif variant == "extra-archive":
        (root / "archive" / "unexpected").write_text("x", encoding="utf-8")
    elif variant == "archive-symlink":
        archive = root / "archive"
        archive.rename(root / "real-archive")
        archive.symlink_to(root / "real-archive", target_is_directory=True)
    elif variant == "archive-bytes":
        digest_leaf.write_bytes(digest_leaf.read_bytes() + b"\n")
    elif variant == "admission-bytes":
        (root / "admission.json").write_text("{}\n", encoding="utf-8")
    elif variant == "claim-bytes":
        (root / "claim.json").write_text("{}\n", encoding="utf-8")
    else:
        (root / "admission.json").unlink()
        (root / "admission.json").mkdir()
    with pytest.raises(module.ValidationError):
        module._validate_successor_lineage(context.contract)


@pytest.mark.parametrize(
    ("row", "control", "lineage", "accepted"),
    (
        ("schema_a_preflight", False, "absent", True),
        ("schema_a_preflight", False, "empty", True),
        ("schema_a_preflight", True, "absent", False),
        ("post_claim", True, "absent", True),
        ("post_claim", True, "complete", False),
        ("schema_b_non_success", True, "absent", True),
        ("schema_b_non_success", True, "empty", True),
        ("schema_b_non_success", True, "complete", True),
        ("schema_b_non_success", False, "complete", False),
        ("schema_b_succeeded", True, "complete", True),
        ("schema_b_succeeded", True, "empty", False),
        ("schema_b_succeeded", False, "complete", False),
    ),
)
def test_four_state_control_lineage_matrix(row: str, control: bool, lineage: str, accepted: bool) -> None:
    module = _module()
    if accepted:
        module._enforce_state_matrix(row=row, control_present=control, lineage_state=lineage)
    else:
        with pytest.raises(module.ValidationError):
            module._enforce_state_matrix(row=row, control_present=control, lineage_state=lineage)


@pytest.mark.parametrize("status", ("missing", "malformed", "failed", "unsupported"))
def test_non_success_receipt_branches_never_reach_complete_or_strict_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
) -> None:
    module, context, _phase, _workspace = _fixture_context(tmp_path)
    evidence = _evidence(module)
    _write_control(module, context, evidence)
    receipt_path = context.contract.path("audio_recovery_receipt")
    receipt_path.parent.mkdir(parents=True)
    if status == "malformed":
        receipt_path.write_bytes(b"{")
    elif status == "failed":
        receipt_path.write_text('{"status":"failed"}\n', encoding="utf-8")
    elif status == "unsupported":
        receipt_path.write_text('{"status":7}\n', encoding="utf-8")
    monkeypatch.setattr(module, "binding", lambda _context=None: evidence)
    monkeypatch.setattr(module, "validate_r6_audio_recovery_receipt", lambda **_kwargs: pytest.fail("complete validator called"))
    monkeypatch.setattr(module, "validate_r6_audio_recovery_authority_rows", lambda *_args, **_kwargs: pytest.fail("full rows called"))
    assert module.command_terminal("fully_bound_launch_claim", "actual", context) == 0


def test_running_receipt_validates_only_the_ordered_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, context, _phase, _workspace = _fixture_context(tmp_path)
    evidence = _evidence(module)
    _write_control(module, context, evidence)
    _write_contract_json(context, "audio_recovery_receipt", {"status": "running"})
    row = {"item_id": "item-0", "video_id": "video-0", "source_url": "https://youtu.be/video-0"}
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "binding", lambda _context=None: evidence)
    monkeypatch.setattr(
        module,
        "validate_r6_audio_recovery_resume",
        lambda **_kwargs: SimpleNamespace(effective_records=(row,), acquired_records=()),
    )
    monkeypatch.setattr(
        module,
        "validate_r6_audio_recovery_authority_rows",
        lambda *_args, **kwargs: calls.append(kwargs) or (row,),
    )
    monkeypatch.setattr(module, "validate_r6_audio_recovery_receipt", lambda **_kwargs: pytest.fail("complete validator called"))
    assert module.command_terminal("fully_bound_launch_claim", "actual", context) == 0
    assert calls == [{
        "recovery_inventory_sha256": evidence.facts["recovery_inventory_sha256"],
        "allow_prefix": True,
        "start_index": 0,
    }]


def test_running_receipt_rejects_archive_current_identity_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, context, _phase, _workspace = _fixture_context(tmp_path)
    evidence = _evidence(module)
    row = {"item_id": "duplicate", "video_id": "video", "source_url": "https://youtu.be/video"}
    monkeypatch.setattr(module, "_historical_archive_records", lambda *_args: (row,))
    monkeypatch.setattr(
        module,
        "validate_r6_audio_recovery_resume",
        lambda **_kwargs: SimpleNamespace(effective_records=(dict(row),), acquired_records=()),
    )
    monkeypatch.setattr(module, "validate_r6_audio_recovery_authority_rows", lambda *_args, **_kwargs: (row,))
    with pytest.raises(module.ValidationError, match="overlap"):
        module._validate_running_receipt(context, evidence, {"status": "running"})


def test_succeeded_requires_complete_lineage_before_full_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, context, _phase, _workspace = _fixture_context(tmp_path)
    evidence = _evidence(module, state="empty")
    _write_control(module, context, evidence)
    _write_contract_json(context, "audio_recovery_receipt", {"status": "succeeded"})
    monkeypatch.setattr(module, "binding", lambda _context=None: evidence)
    monkeypatch.setattr(module, "validate_r6_audio_recovery_receipt", lambda **_kwargs: pytest.fail("complete validator called"))
    with pytest.raises(module.ValidationError, match="schema_b_succeeded"):
        module.command_terminal("fully_bound_launch_claim", "actual", context)


def test_only_complete_bound_succeeded_state_reaches_strict_stage_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    strict_calls: list[dict[str, Any]] = []

    def verifier(**kwargs: Any):
        strict_calls.append(kwargs)
        return SimpleNamespace(status="passed", blockers=[])

    module, context, _phase, _workspace = _fixture_context(tmp_path, verifier=verifier)
    evidence = _evidence(module, state="complete")
    _write_control(module, context, evidence)
    _write_contract_json(context, "audio_recovery_receipt", {"status": "succeeded"})
    _write_contract_json(context, "final_receipt", {"status": "succeeded"})
    rows = tuple(
        {"item_id": f"item-{index}", "video_id": f"video-{index}", "source_url": f"https://youtu.be/v{index}"}
        for index in range(2002)
    )
    manifest_checks: list[bool] = []
    monkeypatch.setattr(module, "binding", lambda _context=None: evidence)
    monkeypatch.setattr(
        module,
        "validate_r6_audio_recovery_receipt",
        lambda **_kwargs: SimpleNamespace(
            effective_records=rows,
            acquired_records=(),
            terminal_records=rows,
        ),
    )
    monkeypatch.setattr(module, "validate_r6_audio_recovery_authority_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(module, "_repeat_manifest_ownership_check", lambda _context: manifest_checks.append(True))
    assert module.command_terminal("fully_bound_launch_claim", "actual", context) == 0
    assert manifest_checks == [True]
    assert len(strict_calls) == 1
    assert strict_calls[0]["strict_execution"] is True
    assert strict_calls[0]["required_owner_ids"] == [module.REQUIRED_OWNER_ID]


@pytest.mark.parametrize(
    "argv",
    (
        ["post-launch", "--expect-schema", "fully_bound_launch_claim"],
        ["post-launch", "--expect-schema", "wrong", "--require-consumed-claim", "--non-writing"],
        ["terminal", "--expect-schema", "fully_bound_launch_claim"],
        ["terminal", "--expect-schema", "preflight_zero_invocation", "--receipt-branch", "actual"],
        ["terminal", "--expect-schema", "wrong"],
    ),
)
def test_unsupported_cli_combinations_fail_before_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    module = _module()
    monkeypatch.setattr(module, "command_terminal", lambda *_args, **_kwargs: pytest.fail("filesystem command reached"))
    with pytest.raises(SystemExit) as exc_info:
        module.main(argv)
    assert exc_info.value.code == 2


def test_exact_post_launch_cli_contract_selects_actual_schema_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(module, "command_terminal", lambda *args: calls.append(args) or 0)
    assert module.main([
        "post-launch",
        "--expect-schema",
        "fully_bound_launch_claim",
        "--require-consumed-claim",
        "--non-writing",
    ]) == 0
    assert calls == [("fully_bound_launch_claim", "actual", None)]
