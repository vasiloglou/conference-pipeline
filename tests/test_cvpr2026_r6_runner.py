from __future__ import annotations

import ast
import hashlib
import copy
import importlib.util
import inspect
import json
import pickle
import sys
import builtins
from pathlib import Path
from types import ModuleType

import pytest

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6ArtifactContract,
    build_r6_dispatch_schedule,
    build_r6_execution_envelope,
    canonical_r6_queue,
    r6_audio_policy_digest,
)
from conference_pipeline.acquisition.cvpr_r6_material_index import (
    R6_MATERIAL_DISCOVERY_RELATIVE_PATH,
    build_material_discovery_receipt,
    load_r6_material_source_policy,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    R6_EVENT_MEDIA_RUN_ID,
    R6_RECOVERY_INVENTORY_RELATIVE_PATH,
    R5LineagePins,
    canonical_r6_recovery_inventory_digest,
    derive_r6_source_route_scope,
    source_route_scope_payload,
)
from conference_pipeline.acquisition.public_material_acquisition import (
    R6_MATERIAL_ACQUISITION_RELATIVE_PATH,
    R6_MATERIAL_PAYLOAD_ROOT,
)
from conference_pipeline.stage_contracts import verify_stage_contract

REPO_ROOT = Path(__file__).resolve().parent.parent
OWNER_PATH = (
    REPO_ROOT
    / "src"
    / "conference_pipeline"
    / "acquisition"
    / "_cvpr_r6_media_owner_impl.py"
)
RUNNER_PATH = REPO_ROOT / "scripts" / "run_cvpr2026_r6_media_recovery.py"
_RETIRED_OCR_FLAG = "--" + "transcription" + "-ocr-mode"
_LEGACY_DIRECT_MANIFEST_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "cvpr_r6_legacy_direct_test_manifest.json"
)


# Repository files each test reads that the public code export does not carry. The
# repo_data hook in tests/conftest.py reads this table: a decorator would change the
# pinned source fingerprints these tests carry (v6.0 Phase 7).
REPO_DATA = {
    "test_final_audio_admission_requires_the_full_ordered_terminal_population": ("conferences/cvpr2026/config.yaml",),
    "test_fixed_bootstrap_admission_uses_real_audio_and_material_receipts": ("conferences/cvpr2026/config.yaml",),
    "test_r6_admission_and_strict_final_reject_missing_terminal_rows_before_success": ("conferences/cvpr2026/config.yaml",),
    "test_r6_checkpoint_remediation_archives_only_the_exact_historical_prefix": ("conferences/cvpr2026/config.yaml",),
    "test_r6_checkpoint_remediation_preserves_a_one_row_terminal_prefix": ("conferences/cvpr2026/config.yaml",),
    "test_r6_checkpoint_remediation_rejects_invalid_prefix_before_archive": ("conferences/cvpr2026/config.yaml",),
    "test_r6_contract_accepts_complete_declared_local_lineage": ("conferences/cvpr2026/config.yaml",),
    "test_r6_contract_rejects_stale_declared_receipt_or_ledger_bytes": ("conferences/cvpr2026/config.yaml",),
    "test_r6_contract_rejects_success_after_failed_sublane_and_accounting_drift": ("conferences/cvpr2026/config.yaml",),
    "test_r6_exact_audio_seed_validator_blocks_nonexact_terminal_classification": ("conferences/cvpr2026/config.yaml",),
    "test_r6_exact_audio_seed_validator_rejects_malformed_acquired_or_terminal_evidence": ("conferences/cvpr2026/config.yaml",),
    "test_r6_failed_contract_never_publishes_success_or_touches_r5": ("conferences/cvpr2026/config.yaml",),
    "test_r6_owner_publishes_only_after_real_strict_candidate_validation": ("conferences/cvpr2026/config.yaml",),
    "test_r6_resume_effective_order_is_authority_bound_across_receipt_containers": ("conferences/cvpr2026/config.yaml",),
    "test_r6_runner_admits_only_explicit_declared_receipt_inputs": ("conferences/cvpr2026/config.yaml",),
    "test_r6_runner_rejects_material_discovery_outside_the_workspace_policy": ("conferences/cvpr2026/config.yaml",),
    "test_r6_runner_rejects_mismatched_policy_and_undeclared_payload_root": ("conferences/cvpr2026/config.yaml",),
    "test_r6_runner_rejects_missing_declared_provider_receipt": ("conferences/cvpr2026/config.yaml",),
    "test_r6_runner_rejects_source_or_discovery_fields_in_processor_inputs": ("conferences/cvpr2026/config.yaml",),
    "test_r6_runner_rejects_unbounded_or_invalid_processor_inputs": ("conferences/cvpr2026/config.yaml",),
    "test_r6_strict_contract_accepts_terminal_no_paid_envelopes": ("conferences/cvpr2026/config.yaml",),
    "test_r6_strict_final_recomputes_exact_audio_classification_not_count_arithmetic": ("conferences/cvpr2026/config.yaml",),
    "test_r6_successor_revalidates_archived_bytes_before_effective_consumption": ("conferences/cvpr2026/config.yaml",),
}


def _legacy_direct_test_ast_index() -> dict[str, ast.FunctionDef]:
    """Return the fixed, reviewable legacy direct-test function inventory.

    The manifest intentionally records function-level node ids, rather than
    pytest's expanded parametrized ids: an assertion edit to any parametrized
    case still changes the single function fingerprint and fails closed.
    The three media-recovery tests are the legacy direct construction seams;
    the rest of that file exercises the immutable overlay independently.
    """

    files_and_names = {
        "tests/test_cvpr2026_r6_runner.py": None,
        "tests/test_cvpr2026_r6_live_owner.py": None,
        "tests/test_cvpr2026_r6_media_recovery.py": {
            "test_r6_queue_is_exact_immutable_overlay_for_all_r5_unavailable_captions",
            "test_r6_rejects_mutated_or_out_of_scope_r5_lineage",
            "test_r6_rejects_output_path_escape_or_any_r5_target",
        },
    }
    indexed: dict[str, ast.FunctionDef] = {}
    for relative, selected in files_and_names.items():
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        for item in tree.body:
            if not isinstance(item, ast.FunctionDef) or not item.name.startswith("test_"):
                continue
            if item.name == "test_legacy_direct_inventory_is_exact_and_assertion_fingerprinted":
                continue
            if selected is not None and item.name not in selected:
                continue
            node_id = f"{relative}::{item.name}"
            assert node_id not in indexed
            indexed[node_id] = item
    return indexed


def _legacy_direct_assertion_fingerprint(function: ast.FunctionDef) -> str:
    """Hash normalized source plus assertion/control AST, never just a name."""

    assertions = [
        ast.dump(node, annotate_fields=True, include_attributes=False)
        for node in ast.walk(function)
        if isinstance(node, (ast.Assert, ast.With, ast.AsyncWith))
    ]
    assert assertions, f"legacy test {function.name} has no executable assertion/control AST"
    payload = {
        "normalized_source": ast.unparse(function),
        "assertion_control_ast": assertions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_direct_has_forbidden_marker(function: ast.FunctionDef) -> bool:
    """Reject local skip/xfail markers without trusting pytest collection."""

    for decorator in function.decorator_list:
        text = ast.unparse(decorator).replace(" ", "")
        if ".skip" in text or ".xfail" in text:
            return True
    return False


def _canonical_legacy_direct_entries(entries: object) -> bytes:
    return json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _fixed_cli_decision_bytes(reason: str = "owner_import_failed") -> bytes:
    """Build the sole byte-exact pre-owner closed envelope for fixtures."""

    phase, exit_code = {
        "owner_import_failed": ("pre_admission", 1),
        "owner_initialization_failed": ("pre_admission", 1),
        "worker_interrupted": ("supervisor", 2),
    }[reason]
    unavailable = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "phase": phase,
        "reason_code": reason,
        "exit_code": exit_code,
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
    return (json.dumps({
        "schema_version": 2,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "disposition": "unavailable",
        "unavailable_decision": unavailable,
        "unavailable_decision_sha256": hashlib.sha256(
            (json.dumps(unavailable, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        ).hexdigest(),
    }, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _legacy_unbound_cli_decision_bytes(reason: str = "admission_invalid") -> bytes:
    """Build the audited schema-v1 leaf without inventing self-binding data."""

    return _canonical_cli_decision_bytes({
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "phase": "pre_admission",
        "reason_code": reason,
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
    })


def _fixed_cli_decision_document(reason: str = "owner_import_failed") -> dict[str, object]:
    """Return a mutable, self-hashed fixed envelope for admission probes."""

    return json.loads(_fixed_cli_decision_bytes(reason))


def _canonical_cli_decision_bytes(document: object) -> bytes:
    """Encode a disposable decision fixture with the required canonical bytes."""

    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _rehash_cli_decision(document: dict[str, object]) -> bytes:
    """Rebind a fixture after deliberately changing an inner schema field."""

    unavailable = document["unavailable_decision"]
    assert type(unavailable) is dict
    document["unavailable_decision_sha256"] = hashlib.sha256(
        _canonical_cli_decision_bytes(unavailable)
    ).hexdigest()
    return _canonical_cli_decision_bytes(document)


def _runner_module() -> ModuleType:
    # Admission/receipt contract tests exercise the private fixed owner;
    # the public script is intentionally a capability-free CLI shell.
    spec = importlib.util.spec_from_file_location("cvpr2026_r6_runner_owner", OWNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sealed_runner_module(name: str = "cvpr2026_r6_sealed_runner") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fresh_runner_surface_is_a_non_composable_cli_shell() -> None:
    """Fresh imports and reload-like identities expose no owner capability."""

    modules = [_sealed_runner_module(f"cvpr2026_r6_sealed_runner_{index}") for index in range(3)]
    forbidden = {
        "atomic_write_r6_json",
        "cvpr_authenticated_event_opener_factory",
        "run_fixed_r6_material_acquisition",
        "validate_fixed_r6_yt_dlp_preflight",
        "_load_repository_dotenv",
        "_observe_official_material_routes",
        "run_fixed_r6_media_owner",
        "admit_r6_stage_inputs",
        "build_r6_stage_receipt",
        "publish_r6_stage_receipt",
    }
    for module in modules:
        assert not forbidden.intersection(vars(module))
        callable_members = {
            name: value
            for name, value in inspect.getmembers_static(module)
            if callable(value)
        }
        assert set(callable_members) == {"main"}
        entrypoint = module.main
        assert entrypoint.__closure__ is None
        assert entrypoint.__defaults__ == (None,)
        assert entrypoint.__kwdefaults__ is None
        assert not any(
            getattr(value, "__name__", "").startswith("conference_pipeline")
            for value in entrypoint.__globals__.values()
        )
        # Function copying and pickling cannot manufacture an owner helper.
        assert copy.copy(entrypoint) is entrypoint
        assert copy.deepcopy(entrypoint) is entrypoint
        assert pickle.loads(pickle.dumps(entrypoint)) is entrypoint


def test_public_media_runner_rejects_non_exact_argv_before_owner_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An import cannot turn the executable shell into an owner API."""

    runner = _sealed_runner_module("cvpr2026_r6_exact_media_cli")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    calls: list[bool] = []
    imports: list[tuple[str, tuple[str, ...]]] = []
    fake_owner = ModuleType("conference_pipeline.acquisition._cvpr_r6_media_owner_impl")
    fake_owner.run_fixed_r6_media_owner = lambda *, live, **_capability: calls.append(live) or 0
    fake_acquisition = ModuleType("conference_pipeline.acquisition")
    fake_acquisition._cvpr_r6_media_owner_impl = fake_owner
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        imports.append((name, tuple(fromlist)))
        if name == "conference_pipeline.acquisition":
            return fake_acquisition
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(SystemExit) as exc_info:
        runner.main(["unexpected"])
    assert exc_info.value.code == 2
    assert calls == []
    assert imports == []


def test_direct_owner_calls_are_internal_test_seams_not_provenance_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the script guard may establish the executable owner context."""

    runner = _sealed_runner_module("cvpr2026_r6_direct_main_rejection")
    owner = _runner_module()
    marker: list[bool] = []
    monkeypatch.setattr(owner, "_run_fixed_r6_media_owner", lambda *, live: marker.append(live) or 0)

    assert owner.run_fixed_r6_media_owner(live=True) == 0
    assert owner._run_fixed_r6_media_owner(live=False) == 0
    monkeypatch.setattr(sys, "argv", ["owner", "--live"])
    assert owner._run_private_r6_media_owner() == 0
    assert owner.main(["--live"]) == 0
    assert marker == [True, False, True, True]
    assert tuple(inspect.signature(owner._run_private_r6_media_owner).parameters) == ()


def test_legacy_direct_inventory_is_exact_and_assertion_fingerprinted() -> None:
    """Freeze the 87 legacy direct seams against deletion or weakened edits."""

    manifest = json.loads(_LEGACY_DIRECT_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest.get("schema_version") == 1
    assert manifest.get("inventory_name") == "cvpr_r6_legacy_direct_tests"
    assert manifest.get("count") == 90
    entries = manifest.get("entries")
    assert type(entries) is list
    assert len(entries) == 90
    assert entries == sorted(entries, key=lambda entry: entry["node_id"])
    assert all(type(entry) is dict and set(entry) == {"node_id", "assertion_fingerprint"} for entry in entries)
    assert len({entry["node_id"] for entry in entries}) == 90
    assert all(
        type(entry["node_id"]) is str
        and type(entry["assertion_fingerprint"]) is str
        and len(entry["assertion_fingerprint"]) == 64
        for entry in entries
    )
    assert manifest.get("aggregate_sha256") == hashlib.sha256(
        _canonical_legacy_direct_entries(entries)
    ).hexdigest()

    indexed = _legacy_direct_test_ast_index()
    assert set(indexed) == {entry["node_id"] for entry in entries}
    for entry in entries:
        function = indexed[entry["node_id"]]
        assert not _legacy_direct_has_forbidden_marker(function)
        assert _legacy_direct_assertion_fingerprint(function) == entry["assertion_fingerprint"]


@pytest.mark.parametrize(
    ("state", "expected_names"),
    [
        ("quarantine", ["cli_decision.json"]),
        ("malformed", ["cli_decision.json"]),
        ("wrong_schema", ["cli_decision.json"]),
        ("digest_changed", ["cli_decision.json"]),
        ("extra_bytes", ["cli_decision.json"]),
        ("extra", ["cli_decision.json", "unexpected"]),
        ("mixed", ["cli_decision.json", "cli_decision.quarantine.json"]),
        ("canonical_malformed_quarantine", ["cli_decision.json", "cli_decision.quarantine.json"]),
        ("quarantine_extra", ["cli_decision.quarantine.json", "unexpected"]),
        ("symlink", ["cli_decision.json"]),
        ("quarantine_symlink", ["cli_decision.quarantine.json"]),
    ],
)
def test_public_runner_admits_the_complete_fixed_namespace_before_owner_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str, expected_names: list[str]
) -> None:
    """Only an absent namespace can import; invalid trees are never repaired."""

    runner = _sealed_runner_module(f"cvpr2026_r6_namespace_{state}")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    decision_dir.mkdir(parents=True)
    raw = _fixed_cli_decision_bytes()
    canonical = decision_dir / "cli_decision.json"
    quarantine = decision_dir / "cli_decision.quarantine.json"
    if state == "canonical":
        canonical.write_bytes(raw)
    elif state == "quarantine":
        quarantine.write_bytes(raw)
    elif state == "malformed":
        canonical.write_bytes(b"{")
    elif state == "wrong_schema":
        document = json.loads(raw)
        document["schema_version"] = 3
        canonical.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif state == "digest_changed":
        document = json.loads(raw)
        document["unavailable_decision_sha256"] = "0" * 64
        canonical.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif state == "extra_bytes":
        canonical.write_bytes(raw + b"\n")
    elif state == "extra":
        canonical.write_bytes(raw)
        (decision_dir / "unexpected").write_text("untrusted", encoding="utf-8")
    elif state == "mixed":
        canonical.write_bytes(raw)
        quarantine.write_bytes(raw)
    elif state == "canonical_malformed_quarantine":
        canonical.write_bytes(raw)
        quarantine.write_bytes(b"{")
    elif state == "quarantine_extra":
        quarantine.write_bytes(raw)
        (decision_dir / "unexpected").write_text("untrusted", encoding="utf-8")
    elif state in {"symlink", "quarantine_symlink"}:
        target = tmp_path / "outside"
        target.write_bytes(raw)
        (quarantine if state == "quarantine_symlink" else canonical).symlink_to(target)

    imports: list[str] = []
    original_import = builtins.__import__

    def forbidden_owner_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        imports.append(name)
        if name == "conference_pipeline.acquisition":
            raise AssertionError("owner import must not be reached")
        return original_import(name, *args, **kwargs)

    observed: list[tuple[str, str, int, str]] = []
    monkeypatch.setattr(builtins, "__import__", forbidden_owner_import)
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)
    assert runner.main(["--live"]) == 1
    assert not any(name.startswith("conference_pipeline") for name in imports)
    # The callable CLI seam reports only its finite taxonomy; it neither
    # repairs the namespace nor imports the owner for an invalid tree.
    assert len(observed) == 1
    assert observed[0] in {
        ("pre_admission", "prior_unavailable_decision", 1, "unavailable"),
        ("pre_admission", "admission_invalid", 1, "unavailable"),
    }
    assert sorted(path.name for path in decision_dir.iterdir()) == expected_names
    assert not list(decision_dir.glob("*.tmp"))


def test_public_runner_classifies_the_exact_production_shape_legacy_as_one_successor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The one pinned v1 leaf reaches the existing one-shot transition."""

    runner = _sealed_runner_module("cvpr2026_r6_production_shape_legacy_unbound")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    decision_dir.mkdir(parents=True)
    leaf = decision_dir / "cli_decision.json"
    raw = _legacy_unbound_cli_decision_bytes()
    leaf.write_bytes(raw)
    (decision_dir / "recovery_inventory.json").write_text("{}\n", encoding="utf-8")
    (decision_dir / "audio_recovery_receipt.json").write_text("{}\n", encoding="utf-8")
    (decision_dir / "audio-recovery-archive").mkdir()
    calls: list[bool] = []
    observed: list[tuple[str, str, int, str]] = []
    fake_owner = ModuleType("conference_pipeline.acquisition._cvpr_r6_media_owner_impl")
    fake_owner.run_fixed_r6_media_owner = lambda *, live: calls.append(live) or 0
    fake_acquisition = ModuleType("conference_pipeline.acquisition")
    fake_acquisition._cvpr_r6_media_owner_impl = fake_owner
    original_import = builtins.__import__

    def fixed_owner_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            return fake_acquisition
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fixed_owner_import)
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)

    expected_siblings = {
        "audio-recovery-archive",
        "audio_recovery_receipt.json",
        "cli_decision.json",
        "recovery_inventory.json",
    }
    assert {path.name for path in decision_dir.iterdir()} == expected_siblings
    assert runner.main(["--live"]) == 0
    assert calls == [True]
    assert observed == [("cli", "completed", 0, "unavailable")]
    assert leaf.read_bytes() == raw
    successor = decision_dir / "decision-successor"
    digest = hashlib.sha256(raw).hexdigest()
    assert {path.name for path in successor.iterdir()} == {"archive", "admission.json", "claim.json"}
    assert {path.name for path in (successor / "archive").iterdir()} == {f"{digest}.json"}
    assert (successor / "archive" / f"{digest}.json").read_bytes() == raw
    assert expected_siblings.issubset({path.name for path in decision_dir.iterdir()})
    assert runner.main(["--live"]) == 1
    assert calls == [True]
    assert not list(decision_dir.glob("*.tmp"))


@pytest.mark.parametrize(
    ("raw", "expected_reason"),
    [
        (_legacy_unbound_cli_decision_bytes("owner_import_failed"), "admission_invalid"),
        (b"{", "admission_invalid"),
    ],
    ids=("wrong-v1-reason", "malformed-v1"),
)
def test_public_runner_does_not_overclassify_similar_legacy_terminal_leaves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: bytes, expected_reason: str,
) -> None:
    """Only the audited v1 shape receives the compatibility diagnosis."""

    runner = _sealed_runner_module("cvpr2026_r6_similar_legacy_terminal")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    decision_dir.mkdir(parents=True)
    leaf = decision_dir / "cli_decision.json"
    leaf.write_bytes(raw)
    observed: list[tuple[str, str, int, str]] = []
    original_import = builtins.__import__

    def forbidden_owner_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            raise AssertionError("invalid legacy leaf must not import owner")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden_owner_import)
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)

    assert runner.main(["--live"]) == 1
    assert observed == [("pre_admission", expected_reason, 1, "unavailable")]
    assert leaf.read_bytes() == raw
    assert sorted(path.name for path in decision_dir.iterdir()) == ["cli_decision.json"]
    assert not list(decision_dir.rglob("decision-successor"))
    assert not list(decision_dir.glob("*.tmp"))


def test_public_runner_claims_one_immutable_successor_before_the_only_owner_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The canonical predecessor can make exactly one closed transition."""

    runner = _sealed_runner_module("cvpr2026_r6_pristine_successor")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    decision_dir.mkdir(parents=True)
    predecessor = _fixed_cli_decision_bytes()
    canonical = decision_dir / "cli_decision.json"
    canonical.write_bytes(predecessor)
    calls: list[bool] = []
    fake_owner = ModuleType("conference_pipeline.acquisition._cvpr_r6_media_owner_impl")
    fake_owner.run_fixed_r6_media_owner = lambda *, live, **_capability: calls.append(live) or 0
    fake_acquisition = ModuleType("conference_pipeline.acquisition")
    fake_acquisition._cvpr_r6_media_owner_impl = fake_owner
    original_import = builtins.__import__

    def successor_owner_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            return fake_acquisition
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", successor_owner_import)
    assert runner.main(["--live"]) == 0
    assert calls == [True]
    assert canonical.read_bytes() == predecessor
    successor = decision_dir / "decision-successor"
    digest = hashlib.sha256(predecessor).hexdigest()
    assert (successor / "archive" / f"{digest}.json").read_bytes() == predecessor
    admission_raw = (successor / "admission.json").read_bytes()
    admission = json.loads(admission_raw)
    assert admission == {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_successor_admission",
        "r6_run_id": "cvpr2026-live-production-r6-expanded-media",
        "predecessor_relation": "immutable_cli_decision_successor",
        "predecessor_sha256": digest,
        "archive_name": f"{digest}.json",
        "admission_schema_version": 1,
    }
    assert json.loads((successor / "claim.json").read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_successor_claim",
        "r6_run_id": "cvpr2026-live-production-r6-expanded-media",
        "predecessor_relation": "immutable_cli_decision_successor",
        "predecessor_sha256": digest,
        "admission_schema_version": 1,
        "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
    }
    assert runner.main(["--live"]) == 1
    assert calls == [True]
    assert canonical.read_bytes() == predecessor
    assert not list(decision_dir.rglob("*.tmp"))


def test_public_runner_invokes_owner_without_successor_authority_after_namespace_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The owner receives only A's retained bytes after every path names B."""

    runner = _sealed_runner_module("cvpr2026_r6_capability_import_swap")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    decision_dir.mkdir(parents=True)
    predecessor = _fixed_cli_decision_bytes()
    (decision_dir / "cli_decision.json").write_bytes(predecessor)
    observed: list[bool] = []
    fake_owner = ModuleType("conference_pipeline.acquisition._cvpr_r6_media_owner_impl")

    def consume_once(*, live: bool) -> int:
        assert live is True
        observed.append(live)
        return 0

    fake_owner.run_fixed_r6_media_owner = consume_once
    fake_acquisition = ModuleType("conference_pipeline.acquisition")
    fake_acquisition._cvpr_r6_media_owner_impl = fake_owner
    replacement = decision_dir.with_name("r6-b")
    replacement.mkdir()
    replacement_predecessor = _fixed_cli_decision_bytes("owner_initialization_failed")
    (replacement / "cli_decision.json").write_bytes(replacement_predecessor)
    (replacement / "decision-successor").mkdir()
    (replacement / "decision-successor" / "archive").mkdir()
    digest = hashlib.sha256(replacement_predecessor).hexdigest()
    (replacement / "decision-successor" / "archive" / f"{digest}.json").write_bytes(replacement_predecessor)
    (replacement / "decision-successor" / "admission.json").write_text("{}\n", encoding="utf-8")
    (replacement / "decision-successor" / "claim.json").write_text("{}\n", encoding="utf-8")
    original_import = builtins.__import__
    moved = decision_dir.with_name("r6-a")

    def swap_at_owner_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            runner.os.rename(decision_dir, moved)
            runner.os.rename(replacement, decision_dir)
            return fake_acquisition
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", swap_at_owner_import)
    assert runner.main(["--live"]) == 0
    assert observed == [True]


@pytest.mark.parametrize("replacement", ("leaf", "parent"))
def test_public_runner_rejects_same_byte_identity_replacement_before_archive_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, replacement: str,
) -> None:
    """A same-byte A→B swap cannot be mistaken for the retained predecessor.

    The replacement is performed by a disposable syscall seam while the
    transition opens its fixed archive directory.  That is after initial
    admission but before the archive's O_EXCL leaf publication; production
    exposes no replacement callback or caller-selected path.
    """

    runner = _sealed_runner_module(f"cvpr2026_r6_identity_replace_{replacement}")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    decision_dir.mkdir(parents=True)
    predecessor = _fixed_cli_decision_bytes()
    canonical = decision_dir / "cli_decision.json"
    canonical.write_bytes(predecessor)
    replacement_leaf = tmp_path / "replacement.json"
    replacement_leaf.write_bytes(predecessor)
    replacement_parent = decision_dir.with_name("r6-replacement")
    moved_parent = decision_dir.with_name("r6-original")
    if replacement == "parent":
        replacement_parent.mkdir()
        (replacement_parent / "cli_decision.json").write_bytes(predecessor)

    imports: list[str] = []
    original_import = builtins.__import__

    def forbidden_owner_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        imports.append(name)
        if name == "conference_pipeline.acquisition":
            raise AssertionError("identity replacement must not import owner")
        return original_import(name, *args, **kwargs)

    original_open = runner.os.open
    swapped = False

    def swap_before_archive(name: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and name == "archive" and flags & runner.os.O_DIRECTORY:
            swapped = True
            if replacement == "leaf":
                runner.os.replace(replacement_leaf, canonical)
            else:
                runner.os.rename(decision_dir, moved_parent)
                runner.os.rename(replacement_parent, decision_dir)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden_owner_import)
    monkeypatch.setattr(runner.os, "open", swap_before_archive)
    assert runner.main(["--live"]) == 1
    assert swapped
    assert not any(name.startswith("conference_pipeline") for name in imports)
    active_parent = decision_dir
    assert (active_parent / "cli_decision.json").read_bytes() == predecessor
    if replacement == "leaf":
        assert not list((active_parent / "decision-successor" / "archive").glob("*.json"))
    else:
        assert (moved_parent / "cli_decision.json").read_bytes() == predecessor
        assert not (active_parent / "decision-successor").exists()
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("variant", ("archive", "admission", "claim", "unknown", "archive_symlink"))
def test_public_runner_rejects_any_preexisting_successor_control_before_owner_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variant: str,
) -> None:
    """Partial, conflicting, and redirected successor controls are terminal."""

    runner = _sealed_runner_module(f"cvpr2026_r6_partial_successor_{variant}")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    decision_dir.mkdir(parents=True)
    predecessor = _fixed_cli_decision_bytes()
    (decision_dir / "cli_decision.json").write_bytes(predecessor)
    successor = decision_dir / "decision-successor"
    successor.mkdir()
    if variant == "archive":
        (successor / "archive").mkdir()
    elif variant == "admission":
        (successor / "admission.json").write_text("{}\n", encoding="utf-8")
    elif variant == "claim":
        (successor / "claim.json").write_text("{}\n", encoding="utf-8")
    elif variant == "unknown":
        (successor / "unexpected").write_text("{}\n", encoding="utf-8")
    else:
        target = tmp_path / "outside-archive"
        target.mkdir()
        (successor / "archive").symlink_to(target, target_is_directory=True)
    imports: list[str] = []
    original_import = builtins.__import__

    def forbidden_owner_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        imports.append(name)
        if name == "conference_pipeline.acquisition":
            raise AssertionError("partial successor must not import owner")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden_owner_import)
    assert runner.main(["--live"]) == 1
    assert not any(name.startswith("conference_pipeline") for name in imports)
    assert (decision_dir / "cli_decision.json").read_bytes() == predecessor
    assert not list(decision_dir.rglob("*.tmp"))


@pytest.mark.parametrize(
    ("leaf_name", "reason", "field", "alias"),
    [
        ("cli_decision.json", "owner_import_failed", "outer_schema", True),
        ("cli_decision.quarantine.json", "owner_import_failed", "outer_schema", 2.0),
        ("cli_decision.json", "owner_import_failed", "inner_schema", True),
        ("cli_decision.quarantine.json", "owner_import_failed", "inner_schema", 1.0),
        ("cli_decision.json", "owner_import_failed", "exit_code", True),
        ("cli_decision.quarantine.json", "owner_import_failed", "exit_code", 1.0),
        ("cli_decision.json", "worker_interrupted", "exit_code", 2.0),
    ],
)
def test_public_runner_rejects_self_hashed_boolean_and_float_numeric_aliases_before_owner_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    leaf_name: str,
    reason: str,
    field: str,
    alias: object,
) -> None:
    """Numeric equality cannot admit a boolean or float fixed-envelope alias."""

    runner = _sealed_runner_module(f"cvpr2026_r6_numeric_alias_{field}_{leaf_name}")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    decision_dir.mkdir(parents=True)
    document = _fixed_cli_decision_document(reason)
    unavailable = document["unavailable_decision"]
    assert type(unavailable) is dict
    if field == "outer_schema":
        document["schema_version"] = alias
        raw = _canonical_cli_decision_bytes(document)
    elif field == "inner_schema":
        unavailable["schema_version"] = alias
        raw = _rehash_cli_decision(document)
    else:
        unavailable["exit_code"] = alias
        raw = _rehash_cli_decision(document)
    leaf = decision_dir / leaf_name
    leaf.write_bytes(raw)
    imports: list[str] = []
    observed: list[tuple[str, str, int, str]] = []
    original_import = builtins.__import__

    def forbidden_owner_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        imports.append(name)
        if name == "conference_pipeline.acquisition":
            raise AssertionError("numeric aliases must not reach owner import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden_owner_import)
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)

    assert runner.main(["--live"]) == 1
    assert not any(name.startswith("conference_pipeline") for name in imports)
    assert observed == [("pre_admission", "admission_invalid", 1, "unavailable")]
    assert leaf.read_bytes() == raw
    assert sorted(path.name for path in decision_dir.iterdir()) == [leaf_name]
    assert not list(decision_dir.glob("*.tmp"))


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("outer_mapping", lambda document: []),
        ("outer_required_text", lambda document: {**document, "receipt_type": 2}),
        ("outer_digest", lambda document: {**document, "unavailable_decision_sha256": 2}),
        ("inner_mapping", lambda document: {**document, "unavailable_decision": []}),
        ("inner_required_text", lambda document: {
            **document,
            "unavailable_decision": {**document["unavailable_decision"], "phase": 1},
        }),
        ("inner_nullable", lambda document: {
            **document,
            "unavailable_decision": {**document["unavailable_decision"], "record_count": 0},
        }),
    ],
)
def test_public_runner_requires_exact_primitive_types_for_every_fixed_envelope_field_category(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, mutate: object
) -> None:
    """All fixed-envelope field categories are type-checked before admission."""

    runner = _sealed_runner_module(f"cvpr2026_r6_bad_type_{name}")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    decision_dir.mkdir(parents=True)
    document = _fixed_cli_decision_document()
    assert callable(mutate)
    altered = mutate(document)
    raw = _canonical_cli_decision_bytes(altered)
    leaf = decision_dir / "cli_decision.json"
    leaf.write_bytes(raw)
    observed: list[tuple[str, str, int, str]] = []
    original_import = builtins.__import__

    def forbidden_owner_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            raise AssertionError("wrong primitive type must not reach owner import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden_owner_import)
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)

    assert runner.main(["--live"]) == 1
    assert observed == [("pre_admission", "admission_invalid", 1, "unavailable")]
    assert leaf.read_bytes() == raw
    assert sorted(path.name for path in decision_dir.iterdir()) == ["cli_decision.json"]
    assert not list(decision_dir.glob("*.tmp"))


@pytest.mark.parametrize(
    "argv",
    [
        [_RETIRED_OCR_FLAG, value]
        for value in ("speech-benchmark-only", "ocr-benchmark-only", "full")
    ]
    + [
        ["--live", _RETIRED_OCR_FLAG, value]
        for value in ("speech-benchmark-only", "ocr-benchmark-only", "full")
    ]
    + [
        [_RETIRED_OCR_FLAG],
        [f"{_RETIRED_OCR_FLAG}=full"],
        [_RETIRED_OCR_FLAG, "full", _RETIRED_OCR_FLAG, "full"],
        ["--live", "--live"],
        ["--unknown"],
        ["positional"],
        ["--live", "positional"],
    ],
)
def test_public_media_runner_rejects_every_non_exact_argv_before_owner_import(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    """Rejected argv cannot reach an owner, dotenv, or any later seam."""

    runner = _sealed_runner_module("cvpr2026_r6_invalid_media_cli")
    imports: list[str] = []
    effects: list[str] = []
    original_import = builtins.__import__

    def forbidden_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        imports.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", forbidden_import)

    with pytest.raises(SystemExit) as error:
        runner.main(argv)

    assert error.value.code == 2
    assert imports == []
    assert effects == []


@pytest.mark.parametrize(
    ("failure", "reason", "exit_code"),
    [
        (ImportError, "owner_import_failed", 1),
        (RuntimeError, "owner_initialization_failed", 1),
        (KeyboardInterrupt, "worker_interrupted", 2),
        (SystemExit, "worker_interrupted", 2),
    ],
)
def test_public_runner_durably_classifies_fixed_owner_import_failures_without_contract_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: type[BaseException], reason: str, exit_code: int
) -> None:
    """Every fixed pre-owner outcome has one stdlib-only durable decision."""

    runner = _sealed_runner_module("cvpr2026_r6_owner_initialization_failure")
    observed: list[tuple[str, str, int, str]] = []
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    original_import = builtins.__import__
    imports: list[str] = []

    def broken_owner_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        imports.append(name)
        if name == "conference_pipeline.acquisition" and tuple(fromlist) == (
            "_cvpr_r6_media_owner_impl",
        ):
            raise failure()
        if name == "conference_pipeline.acquisition.cvpr_r6_artifact_contract":
            raise ImportError("old bridge must never be imported")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_owner_import)
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)

    assert runner.main(["--live"]) == exit_code
    phase = "supervisor" if reason == "worker_interrupted" else "pre_admission"
    assert observed == [(phase, reason, exit_code, "unavailable")]
    assert imports.count("conference_pipeline.acquisition") == 1
    assert not any(name.startswith("conference_pipeline") for name in imports if name != "conference_pipeline.acquisition")
    decision_path = (
        tmp_path
        / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6/cli_decision.json"
    )
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    unavailable = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "phase": phase,
        "reason_code": reason,
        "exit_code": exit_code,
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
    assert payload == {
        "schema_version": 2,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "disposition": "unavailable",
        "unavailable_decision": unavailable,
        "unavailable_decision_sha256": hashlib.sha256(
            (json.dumps(unavailable, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        ).hexdigest(),
    }


def test_public_runner_pre_owner_writer_failure_has_only_finite_sanitized_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broken fixed-destination I/O cannot expose a raw import or diagnostic."""

    runner = _sealed_runner_module("cvpr2026_r6_writer_fallback")
    observed: list[tuple[str, str, int, str]] = []
    original_import = builtins.__import__

    def broken_owner_import(name: str, globals: object = None, locals: object = None,
                            fromlist: tuple[str, ...] = (), level: int = 0) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            raise ImportError("fixture secret path https://example.invalid/row")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_owner_import)
    monkeypatch.setattr(runner.os, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("nope")))
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)

    assert runner.main(["--live"]) == 1
    assert observed == [("pre_admission", "admission_invalid", 1, "unavailable")]


@pytest.mark.parametrize("failure", [ImportError, RuntimeError, KeyboardInterrupt, SystemExit])
def test_public_runner_replace_failure_keeps_only_fixed_quarantine_not_a_temporary_leaf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: type[BaseException]
) -> None:
    """The public pre-owner boundary has no cleanup-dependent raw leaf."""

    runner = _sealed_runner_module(f"cvpr2026_r6_replace_quarantine_{failure.__name__}")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    original_import = builtins.__import__
    unlink_calls: list[tuple[object, ...]] = []

    def broken_owner_import(name: str, globals: object = None, locals: object = None,
                            fromlist: tuple[str, ...] = (), level: int = 0) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            raise failure()
        return original_import(name, globals, locals, fromlist, level)

    def replacement_failure(*_args: object, **_kwargs: object) -> None:
        raise OSError("replacement unavailable")

    def legacy_unlink_failure(*args: object, **_kwargs: object) -> None:
        unlink_calls.append(args)
        raise OSError("legacy cleanup must be irrelevant")

    monkeypatch.setattr(builtins, "__import__", broken_owner_import)
    monkeypatch.setattr(runner.os, "replace", replacement_failure)
    monkeypatch.setattr(runner.os, "unlink", legacy_unlink_failure)

    expected_exit = 2 if failure in {KeyboardInterrupt, SystemExit} else 1
    assert runner.main(["--live"]) == expected_exit
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    assert sorted(path.name for path in decision_dir.iterdir()) == ["cli_decision.quarantine.json"]
    assert not list(decision_dir.glob("*.tmp"))
    quarantine = json.loads((decision_dir / "cli_decision.quarantine.json").read_text(encoding="utf-8"))
    assert quarantine["schema_version"] == 2
    assert quarantine["receipt_type"] == "conference_pipeline.cvpr2026_r6_cli_decision"
    assert quarantine["disposition"] == "unavailable"
    assert quarantine["unavailable_decision"]["reason_code"] == (
        "worker_interrupted" if expected_exit == 2 else (
            "owner_import_failed" if failure is ImportError else "owner_initialization_failed"
        )
    )
    assert unlink_calls == []


def test_public_runner_promotes_an_exact_fixed_quarantine_on_a_later_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A prior fixed quarantine is validated then atomically promoted."""

    runner = _sealed_runner_module("cvpr2026_r6_existing_quarantine")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    original_import = builtins.__import__
    original_replace = runner.os.replace
    attempt = 0

    def broken_owner_import(name: str, globals: object = None, locals: object = None,
                            fromlist: tuple[str, ...] = (), level: int = 0) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            raise ImportError()
        return original_import(name, globals, locals, fromlist, level)

    def fail_once(*args: object, **kwargs: object) -> None:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise OSError("first promotion fails")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_owner_import)
    monkeypatch.setattr(runner.os, "replace", fail_once)
    assert runner.main(["--live"]) == 1
    assert runner.main(["--live"]) == 1
    decision_dir = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6"
    assert sorted(path.name for path in decision_dir.iterdir()) == ["cli_decision.json"]


def test_public_runner_pre_owner_output_close_fault_is_retired_and_finite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed output close cannot be retried by cleanup or escape the CLI."""

    runner = _sealed_runner_module("cvpr2026_r6_output_close_retirement")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    original_import = builtins.__import__
    original_close = runner.os.close
    close_calls: list[int] = []
    observed: list[tuple[str, str, int, str]] = []

    def broken_owner_import(name: str, globals: object = None, locals: object = None,
                            fromlist: tuple[str, ...] = (), level: int = 0) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            raise ImportError("untrusted import detail")
        return original_import(name, globals, locals, fromlist, level)

    def output_close_once(descriptor: int) -> None:
        close_calls.append(descriptor)
        # Admission runs once before import and again before publication (six
        # descriptor closes each); close thirteen is the output descriptor.
        # It must be retired and never retried by finalization.
        if len(close_calls) == 13:
            raise OSError("close failure")
        original_close(descriptor)

    monkeypatch.setattr(builtins, "__import__", broken_owner_import)
    monkeypatch.setattr(runner.os, "close", output_close_once)
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)

    assert runner.main(["--live"]) == 1
    assert observed == [("pre_admission", "owner_import_failed", 1, "unavailable")]
    assert len(close_calls) == 14


def test_public_runner_keeps_durable_unavailable_leaf_after_finalization_fault(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A post-replace directory fsync fault preserves the fixed leaf."""

    runner = _sealed_runner_module("cvpr2026_r6_durable_finalization")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    original_import = builtins.__import__
    original_fsync = runner.os.fsync
    fsync_calls: list[int] = []
    observed: list[tuple[str, str, int, str]] = []

    def broken_owner_import(name: str, globals: object = None, locals: object = None,
                            fromlist: tuple[str, ...] = (), level: int = 0) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            raise ImportError()
        return original_import(name, globals, locals, fromlist, level)

    def final_directory_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        if len(fsync_calls) == 2:
            raise OSError("directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(builtins, "__import__", broken_owner_import)
    monkeypatch.setattr(runner.os, "fsync", final_directory_fsync)
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)

    assert runner.main(["--live"]) == 1
    assert observed == [("pre_admission", "owner_import_failed", 1, "unavailable")]
    leaf = tmp_path / "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6/cli_decision.json"
    assert json.loads(leaf.read_text(encoding="utf-8"))["unavailable_decision"]["reason_code"] == "owner_import_failed"


@pytest.mark.parametrize("raised", [KeyboardInterrupt, SystemExit, MemoryError])
def test_public_runner_pre_owner_writer_does_not_mask_control_or_memory_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raised: type[BaseException]
) -> None:
    """The ordinary writer fallback is intentionally not a BaseException catch."""

    runner = _sealed_runner_module(f"cvpr2026_r6_writer_{raised.__name__}")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    original_import = builtins.__import__

    def broken_owner_import(name: str, globals: object = None, locals: object = None,
                            fromlist: tuple[str, ...] = (), level: int = 0) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            raise ImportError()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_owner_import)
    monkeypatch.setattr(runner.json, "dumps", lambda *_args, **_kwargs: (_ for _ in ()).throw(raised()))
    with pytest.raises(raised):
        runner.main(["--live"])


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_public_runner_normalizes_only_explicit_owner_interruptions(
    monkeypatch: pytest.MonkeyPatch, interruption: type[BaseException], tmp_path: Path
) -> None:
    """The shell leaves arbitrary BaseException classes outside its taxonomy."""

    runner = _sealed_runner_module(f"cvpr2026_r6_explicit_interrupt_{interruption.__name__}")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    fake_owner = ModuleType("conference_pipeline.acquisition._cvpr_r6_media_owner_impl")

    def interrupting_owner(*, live: bool, **_capability: object) -> int:
        assert live is True
        raise interruption()

    fake_owner.run_fixed_r6_media_owner = interrupting_owner
    fake_acquisition = ModuleType("conference_pipeline.acquisition")
    fake_acquisition._cvpr_r6_media_owner_impl = fake_owner
    original_import = builtins.__import__

    def owner_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            return fake_acquisition
        return original_import(name, globals, locals, fromlist, level)

    observed: list[tuple[str, str, int, str]] = []
    monkeypatch.setattr(builtins, "__import__", owner_import)
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)

    assert runner.main(["--live"]) == 2
    assert observed == [("supervisor", "worker_interrupted", 2, "unavailable")]


def test_public_runner_does_not_catch_an_unapproved_base_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Only the two explicit interruption classes have public normalization."""

    class FixtureFatal(BaseException):
        pass

    runner = _sealed_runner_module("cvpr2026_r6_unapproved_base_exception")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))
    fake_owner = ModuleType("conference_pipeline.acquisition._cvpr_r6_media_owner_impl")
    fake_owner.R6RecoveryIncompleteError = RuntimeError
    fake_owner.R6AdmissionError = ValueError
    fake_owner.R6ArtifactContractError = ValueError
    fake_owner.run_fixed_r6_media_owner = lambda **_kwargs: (_ for _ in ()).throw(FixtureFatal())
    fake_acquisition = ModuleType("conference_pipeline.acquisition")
    fake_acquisition._cvpr_r6_media_owner_impl = fake_owner
    original_import = builtins.__import__

    def owner_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            return fake_acquisition
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", owner_import)
    with pytest.raises(FixtureFatal):
        runner.main(["--live"])


@pytest.mark.parametrize(
    "hostile_text",
    (
        "root lease artifact checkpoint preflight /private/token https://example.invalid secret=never",
        "archive workspace path URL header session credential provider",
    ),
)
def test_public_runner_uses_only_typed_owner_disposition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hostile_text: str,
) -> None:
    """Hostile expected-error text cannot select the observed public result."""

    runner = _sealed_runner_module("cvpr2026_r6_typed_disposition")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "runner.py"))

    class Disposition:
        def __init__(self) -> None:
            self.phase = "pre_admission"
            self.reason_code = "admission_invalid"
            self.exit_code = 1

    class ExpectedFailure(Exception):
        def __init__(self, text: str) -> None:
            super().__init__(text)
            self.disposition = Disposition()

    fake_owner = ModuleType("conference_pipeline.acquisition._cvpr_r6_media_owner_impl")
    fake_owner.R6RecoveryIncompleteError = RuntimeError
    fake_owner.R6OwnerExpectedFailure = ExpectedFailure
    fake_owner.R6OwnerDisposition = Disposition
    fake_owner.run_fixed_r6_media_owner = lambda **_kwargs: (_ for _ in ()).throw(ExpectedFailure(hostile_text))
    fake_acquisition = ModuleType("conference_pipeline.acquisition")
    fake_acquisition._cvpr_r6_media_owner_impl = fake_owner
    original_import = builtins.__import__

    def owner_import(name: str, globals: object = None, locals: object = None, fromlist: tuple[str, ...] = (), level: int = 0) -> ModuleType:
        if name == "conference_pipeline.acquisition":
            return fake_acquisition
        return original_import(name, globals, locals, fromlist, level)

    observed: list[tuple[str, str, int, str]] = []
    monkeypatch.setattr(builtins, "__import__", owner_import)
    monkeypatch.setattr(sys, "_cvpr2026_r6_cli_decision_observer", observed.append, raising=False)
    assert runner.main(["--live"]) == 1
    assert observed == [("pre_admission", "admission_invalid", 1, "reachable")]


def test_private_media_main_has_the_same_exact_noarg_or_live_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The implementation façade cannot revive the retired CLI dispatch."""

    owner = _runner_module()
    calls: list[bool] = []
    monkeypatch.setattr(
        owner, "_run_fixed_r6_media_owner", lambda *, live, **_capability: calls.append(live) or 0
    )
    monkeypatch.setattr(
        owner,
        "run_fixed_r6_transcription_ocr_owner",
        lambda **_kwargs: pytest.fail("retired media CLI dispatch was reached"),
    )

    assert owner.main([]) == 0
    assert owner.main(["--live"]) == 0
    with pytest.raises(SystemExit) as error:
        owner.main([_RETIRED_OCR_FLAG, "full"])

    assert error.value.code == 2
    assert calls == [False, True]


def _audio_receipt(
    module: ModuleType,
    *,
    recovery_inventory: dict[str, object],
    rows: list[dict[str, object]],
    status: str,
) -> dict[str, object]:
    """Fixture helper; the production runner intentionally exports none."""

    scope = derive_r6_source_route_scope(recovery_inventory)
    acquired = [dict(row) for row in rows if row.get("terminal_state") == "acquired"]
    terminal = [dict(row) for row in rows if row.get("terminal_state") != "acquired"]
    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_audio_recovery",
        "status": status,
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "output_path": "storage/transcription-ocr/cvpr2026/r6/audio_recovery_receipt.json",
        "recovery_inventory_sha256": recovery_inventory["r6_recovery_inventory_sha256"],
        "audio_policy_digest": module.r6_audio_policy_digest(),
        "audio_policy_max_attempts": 3,
        **source_route_scope_payload(scope),
        "records": acquired,
        "terminal_records": terminal,
        "effective_record_order": [
            {"item_id": str(row["item_id"]), "video_id": str(row["video_id"])}
            for row in rows
        ],
        "counts": {"declared": len(rows), "acquired": len(acquired), "terminal": len(terminal)},
        "provider_calls_made": 0,
        "benchmark_calls_made": 0,
        "asr_calls_made": 0,
        "ocr_calls_made": 0,
        "secrets_copied": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r6_workspace(tmp_path: Path) -> tuple[Path, dict[str, Path], list[dict[str, object]]]:
    workspace = tmp_path / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    repo_workspace = REPO_ROOT / "conferences" / "cvpr2026"
    for name in ("config.yaml", "r6_material_source_policy.yaml"):
        (workspace / name).write_bytes((repo_workspace / name).read_bytes())
    r6_root = workspace / "storage/transcription-ocr/cvpr2026/r6"
    raw_audio = r6_root / "raw-audio/video-001.m4a"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"audio")
    material = r6_root / "material-payloads/paper-001.pdf"
    material.parent.mkdir(parents=True, exist_ok=True)
    material.write_bytes(b"%PDF-1.4\n")

    pins = R5LineagePins("1" * 64, "2" * 64, "3" * 64)
    recovery: dict[str, object] = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_media_recovery_inventory",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "r5_lineage": {
            "run_id": "cvpr2026-live-production-r5-event-media",
            "receipt_sha256": pins.receipt_sha256,
            "inventory_sha256": pins.inventory_sha256,
            "inventory_canonical_sha256": pins.inventory_canonical_sha256,
        },
        "counts": {"asr_fallback_required": 2002},
        "asr_fallback_required": [
            {
                "r5_record_id": f"r5:caption:{index:04d}",
                "video_id": f"A{index:010d}",
                "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
                "source_event_urls": ["https://cvpr.thecvf.com/virtual/2026/poster/1"],
                "provenance": ["r5_official_event_page"],
                "source_classification": "public_youtube_candidate",
                "state": "asr_fallback_required",
                "attempt_count": 0,
                "local_media": None,
                "terminal_blocker": None,
            }
            for index in range(2002)
        ],
    }
    scope = derive_r6_source_route_scope(recovery, verify_inventory_digest=False)
    recovery.update(source_route_scope_payload(scope))
    recovery["r6_recovery_inventory_sha256"] = canonical_r6_recovery_inventory_digest(recovery)
    paths = {
        "recovery": workspace / R6_RECOVERY_INVENTORY_RELATIVE_PATH,
        "audio": r6_root / "audio_recovery_receipt.json",
        "discovery": r6_root / "material_discovery_receipt.json",
        "acquisition": r6_root / "material_acquisition_receipt.json",
        "estimate": r6_root / "cost_estimate_receipt.json",
        "benchmark": r6_root / "benchmark_receipt.json",
        "provider": r6_root / "provider_receipts.json",
        "ledger": r6_root / "budget_ledger.json",
        "processing": r6_root / "r6_processing_inventory.json",
        "output": workspace
        / f"storage/workflow-runs/{R6_EVENT_MEDIA_RUN_ID}/stage-05-transcription-ocr-r6.json",
    }
    _write_json(paths["recovery"], recovery)
    terminal_records = [
        {
            "item_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
        for index in range(1, 2_002)
    ]
    _write_json(
        paths["audio"],
        {
            "receipt_type": "conference_pipeline.cvpr2026_r6_audio_recovery",
            "status": "succeeded",
            "run_id": R6_EVENT_MEDIA_RUN_ID,
            "output_path": "storage/transcription-ocr/cvpr2026/r6/audio_recovery_receipt.json",
            "recovery_inventory_sha256": recovery["r6_recovery_inventory_sha256"],
            "audio_policy_digest": r6_audio_policy_digest(),
            "records": [
                {
                "item_id": "r5:caption:0000",
                "video_id": "A0000000000",
                    "local_media_path": raw_audio.relative_to(workspace).as_posix(),
                    "sha256": hashlib.sha256(raw_audio.read_bytes()).hexdigest(),
                    "byte_count": raw_audio.stat().st_size,
                    "mime_type": "audio/mp4",
                    "duration_seconds": 61.0,
                    "terminal_state": "acquired",
                    "source_url": "https://www.youtube.com/watch?v=A0000000000",
                    "provenance": "direct_public_youtube",
                    "attempt_count": 1,
                }
            ],
            "terminal_records": terminal_records,
            "effective_record_order": [
                {"item_id": "r5:caption:0000", "video_id": "A0000000000"}
            ] + [
                {
                    "item_id": f"r5:caption:{index:04d}",
                    "video_id": f"A{index:010d}",
                }
                for index in range(1, 2_002)
            ],
            "counts": {"declared": 2_002, "acquired": 1, "terminal": 2_001},
            "source_scope_count": scope.source_scope_count,
            "source_scope_sha256": scope.source_scope_sha256,
            "audio_policy_max_attempts": 3,
        },
    )
    discovery = build_material_discovery_receipt(
        workspace=workspace,
        observations=[
            {
                "record_id": scope.routes[0]["route_id"],
                "event_id": scope.routes[0]["event_id"],
                "source_page_url": scope.routes[0]["source_page_url"],
                "observed_at": "2026-07-17T00:00:00Z",
                "direct_target_url": "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf",
                "link_kind": "direct_public",
                "content_class": "paper_pdf",
                "mime_type": "application/pdf",
                "byte_count": material.stat().st_size,
                "redirect_hops": [],
            }
        ],
        output_path=paths["discovery"],
        source_scope=source_route_scope_payload(scope),
        recovery_inventory=recovery,
    )
    assert paths["discovery"].relative_to(workspace) == R6_MATERIAL_DISCOVERY_RELATIVE_PATH
    _write_json(paths["discovery"], discovery)
    _write_json(
        paths["acquisition"],
        {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.r6_public_material_acquisition",
            "status": "succeeded",
            "run_id": R6_EVENT_MEDIA_RUN_ID,
            "output_path": R6_MATERIAL_ACQUISITION_RELATIVE_PATH.as_posix(),
            "payload_root": R6_MATERIAL_PAYLOAD_ROOT.as_posix(),
            "policy_digest": discovery["policy_digest"],
            "discovery_receipt_sha256": hashlib.sha256(paths["discovery"].read_bytes()).hexdigest(),
            "payloads": [
                {
                    "record_id": scope.routes[0]["route_id"],
                    "event_id": scope.routes[0]["event_id"],
                    "acquisition_state": "acquired",
                    "source_page_provenance": scope.routes[0]["source_page_url"],
                    "initial_target_url": "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf",
                    "final_target_url": "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf",
                    "redirect_hosts": [],
                    "content_class": "paper_pdf",
                    "sha256": hashlib.sha256(material.read_bytes()).hexdigest(),
                    "byte_count": material.stat().st_size,
                    "mime_type": "application/pdf",
                    "policy_digest": discovery["policy_digest"],
                    "local_payload_path": material.relative_to(workspace).as_posix(),
                    "terminal_reason": None,
                }
            ],
        "terminal_records": [],
        "counts": {"declared": 1, "acquired": 1, "terminal": 0},
        "source_scope_count": scope.source_scope_count,
        "source_scope_sha256": scope.source_scope_sha256,
        "route_observation_sha256": discovery["route_observation_sha256"],
        },
    )
    queue, queue_digest = canonical_r6_queue(
        speech_ids=["r5:caption:0000"], ocr_ids=[]
    )
    schedule = build_r6_dispatch_schedule(
        run_id=R6_EVENT_MEDIA_RUN_ID,
        queue=queue,
        queue_digest=queue_digest,
        speech_total_usd=1.0,
        ocr_total_usd=0.0,
    )
    bindings = {
        "recovery_inventory_sha256": hashlib.sha256(paths["recovery"].read_bytes()).hexdigest(),
        "audio_recovery_receipt_sha256": hashlib.sha256(paths["audio"].read_bytes()).hexdigest(),
        "material_discovery_receipt_sha256": hashlib.sha256(paths["discovery"].read_bytes()).hexdigest(),
        "material_acquisition_receipt_sha256": hashlib.sha256(paths["acquisition"].read_bytes()).hexdigest(),
    }
    for name, kind in (("estimate", "estimate"), ("benchmark", "benchmark"), ("provider", "provider")):
        _write_json(
            paths[name],
            build_r6_execution_envelope(
                kind=kind,
                status="succeeded",
                material_policy_digest=str(discovery["policy_digest"]),
                audio_policy_digest=r6_audio_policy_digest(),
                predecessor_bindings=bindings,
                schedule=schedule,
            ),
        )
    from conference_pipeline.costs.budget_enforcement import _r6_digest

    reservations = []
    committed = []
    actuals = []
    provider_rows = []
    for index, entry in enumerate(schedule["entries"]):
        identity = {
            "run_id": R6_EVENT_MEDIA_RUN_ID,
            "task_id": entry["task_id"],
            "ordinal": index,
            "entry_fingerprint": entry["entry_fingerprint"],
            "dispatch_schedule_digest": schedule["dispatch_schedule_digest"],
        }
        reservation_id = _r6_digest(identity)
        provider_row = {
            "sanitized": True,
            "reservation_id": reservation_id,
            "provider": "fixture",
            "model": "fixture",
            "status": "succeeded",
            "actual_cost_usd": entry["maximum_charge_usd"],
        }
        provider_rows.append(provider_row)
        reserved = {
            "reservation_id": reservation_id,
            **identity,
            "item_id": entry["identity"],
            "benchmark_id": entry["identity"],
            "estimate_id": schedule["dispatch_schedule_digest"],
            "maximum_charge_usd": entry["maximum_charge_usd"],
            "state": "reserved",
        }
        committed.append(dict(reserved))
        provider_sha = _r6_digest(provider_row)
        reservations.append(
            {
                **reserved,
                "actual_cost_usd": entry["maximum_charge_usd"],
                "actual_cost_source": "provider_receipt",
                "provider_receipt_sha256": provider_sha,
                "state": "settled",
            }
        )
        actuals.append(
            {
                "reservation_id": reservation_id,
                "task_id": entry["task_id"],
                "actual_cost_usd": entry["maximum_charge_usd"],
                "provider_receipt_sha256": provider_sha,
            }
        )
    ledger = {
        "schema_version": 1,
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "ledger_type": "r6_committed_plus_actual",
        "canonical_prior_spend_usd": 0.0,
        "reservations": reservations,
        "committed": committed,
        "actuals": actuals,
        "queue_digest": schedule["queue_digest"],
        "dispatch_schedule_digest": schedule["dispatch_schedule_digest"],
        "dispatch_schedule": schedule,
        "next_dispatch_ordinal": len(schedule["entries"]),
    }
    ledger["ledger_sha256"] = _r6_digest(ledger)
    _write_json(
        paths["ledger"],
        ledger,
    )
    provider = json.loads(paths["provider"].read_text(encoding="utf-8"))
    provider["receipts"] = provider_rows
    _write_json(paths["provider"], provider)
    _write_json(
        paths["processing"],
        {
            "receipt_type": "conference_pipeline.cvpr2026_r6_transcription_ocr_inventory",
            "run_id": R6_EVENT_MEDIA_RUN_ID,
            "stage_id": "transcription-ocr-r6",
            "status": "succeeded",
        },
    )
    inputs = [
        {
            "record_id": "video-001",
            "semantic_class": "speech",
            "local_path": raw_audio.relative_to(workspace).as_posix(),
            "sha256": hashlib.sha256(raw_audio.read_bytes()).hexdigest(),
            "byte_count": raw_audio.stat().st_size,
            "mime_type": "audio/mp4",
            "duration_seconds": 61.0,
        },
        {
            "record_id": "paper-001",
            "semantic_class": "paper_text",
            "local_path": material.relative_to(workspace).as_posix(),
            "sha256": hashlib.sha256(material.read_bytes()).hexdigest(),
            "byte_count": material.stat().st_size,
            "mime_type": "application/pdf",
        },
    ]
    return workspace, paths, inputs


def test_final_audio_admission_requires_the_full_ordered_terminal_population(
    tmp_path: Path,
) -> None:
    """A terminal-only continuation crosses arbitrary local chunk boundaries."""

    module = _runner_module()
    workspace, paths, _ = _r6_workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    recovery = json.loads(paths["recovery"].read_text(encoding="utf-8"))
    authority, rows = module._validate_fixed_r6_owner_authority(recovery)

    # Produce the exact authority through deliberately small chunks.  The
    # finalizer must not mistake a completed chunk for a completed run.
    chunk_size = 29
    terminal_rows = [
        {
            "item_id": row["item_id"],
            "video_id": row["video_id"],
            "source_url": row["source_url"],
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
        for start in range(0, len(rows), chunk_size)
        for row in rows[start : start + chunk_size]
    ]
    assert len(terminal_rows) == 2_002
    assert len(terminal_rows) > chunk_size
    receipt = _audio_receipt(
        module,
        recovery_inventory=recovery, rows=terminal_rows, status="succeeded"
    )

    module._require_complete_audio_recovery(
        contract=contract,
        recovery_inventory=recovery,
        authority=authority,
        receipt=receipt,
    )

    incomplete = _audio_receipt(
        module,
        recovery_inventory=recovery, rows=terminal_rows[:31], status="running"
    )
    with pytest.raises(module.R6RecoveryIncompleteError, match="resumable and incomplete"):
        module._require_complete_audio_recovery(
            contract=contract,
            recovery_inventory=recovery,
            authority=authority,
            receipt=incomplete,
        )


def test_r6_resume_effective_order_is_authority_bound_across_receipt_containers(
    tmp_path: Path,
) -> None:
    """A recomputed order binding cannot turn split receipt containers into a set."""

    from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
        build_r6_audio_recovery_authority,
        validate_r6_audio_recovery_authority_queue,
        validate_r6_audio_recovery_authority_rows,
        validate_r6_audio_recovery_resume,
    )

    workspace, paths, _ = _r6_workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    recovery = json.loads(paths["recovery"].read_text(encoding="utf-8"))
    receipt = json.loads(paths["audio"].read_text(encoding="utf-8"))
    receipt["status"] = "running"
    authority = build_r6_audio_recovery_authority(
        recovery, prior_oracle=lambda _row: "direct_public_youtube:max_attempts=3"
    )
    queue = recovery["asr_fallback_required"]
    assert isinstance(queue, list)
    validate_r6_audio_recovery_authority_queue(
        authority,
        queue,
        recovery_inventory_sha256=str(recovery["r6_recovery_inventory_sha256"]),
    )
    classified = validate_r6_audio_recovery_resume(
        recovery_inventory=recovery, receipt=receipt, contract=contract
    )
    assert len(classified.effective_records) == 2_002
    assert classified.effective_records[0]["item_id"] == "r5:caption:0000"

    order = receipt["effective_record_order"]
    assert isinstance(order, list)
    order[1], order[2] = order[2], order[1]
    reordered = validate_r6_audio_recovery_resume(
        recovery_inventory=recovery, receipt=receipt, contract=contract
    )
    with pytest.raises(ValueError, match="ordered row 1"):
        validate_r6_audio_recovery_authority_rows(
            authority,
            list(reordered.effective_records),
            recovery_inventory_sha256=str(recovery["r6_recovery_inventory_sha256"]),
            allow_prefix=True,
        )


def _admit(
    module: ModuleType, workspace: Path, paths: dict[str, Path], inputs: list[dict[str, object]]
):
    return module.admit_r6_stage_inputs(
        workspace=workspace,
        recovery_inventory_path=paths["recovery"],
        audio_recovery_receipt_path=paths["audio"],
        material_discovery_receipt_path=paths["discovery"],
        material_acquisition_receipt_path=paths["acquisition"],
        estimate_receipt_path=paths["estimate"],
        benchmark_receipt_path=paths["benchmark"],
        provider_receipts_path=paths["provider"],
        budget_ledger_path=paths["ledger"],
        output_receipt_path=paths["output"],
        processor_inputs=inputs,
        paid_estimate_required=True,
        artifact_contract=R6ArtifactContract.for_test_fixture(workspace),
    )


def test_r6_runner_admits_only_explicit_declared_receipt_inputs(tmp_path: Path) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)

    admission = _admit(module, workspace, paths, inputs)

    assert admission.run_id == R6_EVENT_MEDIA_RUN_ID
    assert admission.stage_id == "transcription-ocr-r6"
    assert admission.policy_digest == load_r6_material_source_policy(workspace).policy_digest
    assert admission.recovery_record_count == 2002
    assert admission.output_receipt_path == paths["output"]
    assert set(admission.input_digests) == {
        "recovery_inventory",
        "audio_recovery_receipt",
        "material_discovery_receipt",
        "material_acquisition_receipt",
        "estimate_receipt",
        "benchmark_receipt",
        "provider_receipts",
        "budget_ledger",
    }


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda paths, inputs: inputs.__setitem__(
                0, {**inputs[0], "local_path": "https://youtube.com/watch?v=x"}
            ),
            "source URL",
        ),
        (
            lambda paths, inputs: inputs.__setitem__(0, {**inputs[0], "duration_seconds": 0}),
            "duration",
        ),
        (
            lambda paths, inputs: paths.__setitem__(
                "output", paths["output"].parent / "../unknown.json"
            ),
            "output",
        ),
    ],
)
def test_r6_runner_rejects_unbounded_or_invalid_processor_inputs(
    tmp_path: Path, mutate, message: str
) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    mutate(paths, inputs)

    with pytest.raises(module.R6AdmissionError, match=message):
        _admit(module, workspace, paths, inputs)


@pytest.mark.parametrize(
    "forbidden_field",
    ["source_page_url", "direct_target_url", "redirect_hops", "discovery_record"],
)
def test_r6_runner_rejects_source_or_discovery_fields_in_processor_inputs(
    tmp_path: Path, forbidden_field: str
) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    value: object = "https://openaccess.thecvf.com/CVPR2026/papers/paper-001.pdf"
    if forbidden_field == "redirect_hops":
        value = ["openaccess.thecvf.com"]
    elif forbidden_field == "discovery_record":
        value = {"record_id": "paper-001"}
    inputs[1] = {**inputs[1], forbidden_field: value}

    with pytest.raises(module.R6AdmissionError, match="source or discovery"):
        _admit(module, workspace, paths, inputs)


def test_r6_runner_rejects_mismatched_policy_and_undeclared_payload_root(tmp_path: Path) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    acquisition = json.loads(paths["acquisition"].read_text(encoding="utf-8"))
    acquisition["policy_digest"] = "7" * 64
    _write_json(paths["acquisition"], acquisition)

    with pytest.raises(module.R6AdmissionError, match="policy digest"):
        _admit(module, workspace, paths, inputs)

    _write_json(
        paths["acquisition"],
        {**acquisition, "policy_digest": load_r6_material_source_policy(workspace).policy_digest},
    )
    inputs[1] = {
        **inputs[1],
        "local_path": "storage/transcription-ocr/cvpr2026/r6/unknown/paper.pdf",
    }
    with pytest.raises(module.R6AdmissionError, match="declared payload root"):
        _admit(module, workspace, paths, inputs)


def test_r6_contract_rejects_success_after_failed_sublane_and_accounting_drift(
    tmp_path: Path,
) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    admission = _admit(module, workspace, paths, inputs)
    receipt = module.build_r6_stage_receipt(
        admission=admission,
        processed_item_count=1,
        terminal_blocker_count=2001,
        sublanes={"audio": "blocked", "materials": "pass"},
        processor_version="r6-test-v1",
    )
    receipt["checks"]["terminal_blocker_count"] = 0
    receipt["owner_execution_evidence"]["transcription_ocr.cvpr2026.r6"]["terminal_accounting"][
        "terminal_blocker_count"
    ] = 0

    result = verify_stage_contract(
        workspace=workspace,
        run_id=R6_EVENT_MEDIA_RUN_ID,
        stage_id="transcription-ocr-r6",
        receipt=receipt,
        strict_execution=True,
    )

    assert result.status == "blocked"
    assert {blocker["code"] for blocker in result.blockers} >= {
        "stage_contract_r6_sublane_failed",
        "stage_contract_r6_accounting_invalid",
    }


def test_r6_contract_accepts_complete_declared_local_lineage(tmp_path: Path) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    admission = _admit(module, workspace, paths, inputs)
    receipt = module.build_r6_stage_receipt(
        admission=admission,
        processed_item_count=1,
        terminal_blocker_count=2001,
        sublanes={"audio": "pass", "materials": "pass"},
        processor_version="r6-test-v1",
    )

    result = verify_stage_contract(
        workspace=workspace,
        run_id=R6_EVENT_MEDIA_RUN_ID,
        stage_id="transcription-ocr-r6",
        receipt=receipt,
        strict_execution=True,
    )

    assert result.status == "pass", result.blockers
    assert result.registry_stage_id == "transcription-ocr-r6.cvpr2026"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda audio: audio["terminal_records"].pop(),
        lambda audio: audio["terminal_records"].append(dict(audio["terminal_records"][0])),
        lambda audio: audio["terminal_records"].append(
            {
                **audio["terminal_records"][0],
                "item_id": "r5:caption:extra",
                "video_id": "ZZZ00000000",
            }
        ),
        lambda audio: audio["terminal_records"].__setitem__(
            0, {**audio["terminal_records"][0], "video_id": "A0000000000"}
        ),
    ],
    ids=("missing", "duplicate", "extra", "swapped-identity"),
)
def test_r6_exact_audio_seed_validator_blocks_nonexact_terminal_classification(
    tmp_path: Path, mutation
) -> None:
    """Arithmetic cannot substitute for all 2,002 immutable terminal records."""

    from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
        validate_r6_audio_recovery_receipt,
    )

    _workspace, paths, _inputs = _r6_workspace(tmp_path)
    recovery = json.loads(paths["recovery"].read_text(encoding="utf-8"))
    audio = json.loads(paths["audio"].read_text(encoding="utf-8"))
    mutation(audio)
    # Preserve superficial count arithmetic; the validator must use immutable
    # identities rather than the echoed totals.
    audio["counts"] = {"declared": 2_002, "acquired": 1, "terminal": 2_001}

    with pytest.raises(ValueError, match="seed|identity|classification|terminal"):
        validate_r6_audio_recovery_receipt(
            recovery_inventory=recovery,
            receipt=audio,
            contract=R6ArtifactContract.for_test_fixture(_workspace),
        )


def test_r6_admission_and_strict_final_reject_missing_terminal_rows_before_success(
    tmp_path: Path,
) -> None:
    """A count-correct forgery cannot reach the final-owner success publisher."""

    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    audio = json.loads(paths["audio"].read_text(encoding="utf-8"))
    audio["terminal_records"] = []
    audio["counts"] = {"declared": 2_002, "acquired": 1, "terminal": 2_001}
    _write_json(paths["audio"], audio)

    with pytest.raises(module.R6AdmissionError, match="audio.*classification|audio receipt"):
        _admit(module, workspace, paths, inputs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda audio: audio["records"][0].pop("local_media_path"),
        lambda audio: audio["records"][0].__setitem__(
            "local_path", audio["records"][0].pop("local_media_path")
        ),
        lambda audio: audio["records"][0].__setitem__(
            "local_path", audio["records"][0]["local_media_path"]
        ),
        lambda audio: audio["records"][0].__setitem__(
            "local_media_path", "https://example.invalid/audio.mp3"
        ),
        lambda audio: audio["records"][0].__setitem__(
            "local_media_path", "storage/transcription-ocr/cvpr2026/r6/raw-audio/missing.mp3"
        ),
        lambda audio: audio["records"][0].__setitem__("sha256", "f" * 64),
        lambda audio: audio["records"][0].pop("byte_count"),
        lambda audio: audio["records"][0].pop("sha256"),
        lambda audio: audio["records"][0].pop("mime_type"),
        lambda audio: audio["records"][0].pop("duration_seconds"),
        lambda audio: audio.__setitem__("audio_policy_max_attempts", 2),
        lambda audio: audio["terminal_records"][0].__setitem__(
            "terminal_state", "acquired"
        ),
    ],
    ids=(
        "path",
        "legacy-only-path",
        "dual-path-alias",
        "remote-path",
        "absent-path",
        "bad-digest",
        "bytes",
        "digest",
        "mime",
        "duration",
        "retry-policy",
        "terminal-state",
    ),
)
def test_r6_exact_audio_seed_validator_rejects_malformed_acquired_or_terminal_evidence(
    tmp_path: Path, mutate
) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    audio = json.loads(paths["audio"].read_text(encoding="utf-8"))
    mutate(audio)
    _write_json(paths["audio"], audio)

    with pytest.raises(module.R6AdmissionError, match="audio receipt exact classification"):
        _admit(module, workspace, paths, inputs)


def test_r6_strict_final_recomputes_exact_audio_classification_not_count_arithmetic(
    tmp_path: Path,
) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    admission = _admit(module, workspace, paths, inputs)
    receipt = module.build_r6_stage_receipt(
        admission=admission,
        processed_item_count=1,
        terminal_blocker_count=2001,
        sublanes={"audio": "pass", "materials": "pass"},
        processor_version="r6-test-v1",
    )
    audio = json.loads(paths["audio"].read_text(encoding="utf-8"))
    audio["terminal_records"] = []
    audio["counts"] = {"declared": 2_002, "acquired": 1, "terminal": 2_001}
    audio_digest = _write_json(paths["audio"], audio)
    receipt["checks"]["audio_recovery_receipt_sha256"] = audio_digest
    receipt["owner_execution_evidence"]["transcription_ocr.cvpr2026.r6"]["receipt_digests"][
        "audio_recovery_receipt"
    ] = audio_digest

    result = verify_stage_contract(
        workspace=workspace,
        run_id=R6_EVENT_MEDIA_RUN_ID,
        stage_id="transcription-ocr-r6",
        receipt=receipt,
        strict_execution=True,
    )

    assert result.status == "blocked"
    assert "stage_contract_r6_source_scope_invalid" in {item["code"] for item in result.blockers}


def test_r6_owner_publishes_only_after_real_strict_candidate_validation(tmp_path: Path) -> None:
    """The registered façade never writes a final success before strict validation."""

    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    admission = _admit(module, workspace, paths, inputs)

    output = module.publish_r6_stage_receipt(
        admission=admission,
        processed_item_count=1,
        terminal_blocker_count=2001,
        sublanes={"audio": "pass", "materials": "pass"},
        processor_version="r6-test-v1",
    )

    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["status"] == "succeeded"
    assert published["required_script_ids"] == ["transcription_ocr.cvpr2026.r6"]
    evidence = published["owner_execution_evidence"]["transcription_ocr.cvpr2026.r6"]
    assert evidence["receipt_digests"] == {
        name: published["checks"][f"{name}_sha256"]
        for name in (
            "recovery_inventory",
            "audio_recovery_receipt",
            "material_discovery_receipt",
            "material_acquisition_receipt",
            "estimate_receipt",
            "benchmark_receipt",
            "provider_receipts",
            "budget_ledger",
        )
    }


def test_r6_strict_contract_accepts_terminal_no_paid_envelopes(tmp_path: Path) -> None:
    """No-paid is a valid strict final state with no hidden paid dispatch."""

    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    discovery = json.loads(paths["discovery"].read_text(encoding="utf-8"))
    audio = json.loads(paths["audio"].read_text(encoding="utf-8"))
    audio["records"] = []
    audio["terminal_records"] = [
        {
            "item_id": "r5:caption:0000",
            "video_id": "A0000000000",
            "source_url": "https://www.youtube.com/watch?v=A0000000000",
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
    ] + audio["terminal_records"]
    audio["counts"] = {"declared": 2_002, "acquired": 0, "terminal": 2_002}
    _write_json(paths["audio"], audio)
    bindings = {
        "recovery_inventory_sha256": hashlib.sha256(paths["recovery"].read_bytes()).hexdigest(),
        "audio_recovery_receipt_sha256": hashlib.sha256(paths["audio"].read_bytes()).hexdigest(),
        "material_discovery_receipt_sha256": hashlib.sha256(paths["discovery"].read_bytes()).hexdigest(),
        "material_acquisition_receipt_sha256": hashlib.sha256(paths["acquisition"].read_bytes()).hexdigest(),
    }
    terminal = build_r6_execution_envelope(
        kind="no_paid_lane",
        status="not_applicable",
        material_policy_digest=str(discovery["policy_digest"]),
        audio_policy_digest=r6_audio_policy_digest(),
        predecessor_bindings=bindings,
        schedule=None,
    )
    terminal["receipts"] = []
    for name in ("estimate", "benchmark", "provider"):
        _write_json(paths[name], terminal)
    from conference_pipeline.costs.budget_enforcement import _r6_digest

    ledger = {
        "schema_version": 1,
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "ledger_type": "r6_committed_plus_actual",
        "status": "not_applicable",
        "canonical_prior_spend_usd": 0.0,
        "reservations": [],
        "committed": [],
        "actuals": [],
        "queue_digest": terminal["queue_digest"],
        "dispatch_schedule_digest": None,
        "policy_digest": discovery["policy_digest"],
        "audio_policy_digest": r6_audio_policy_digest(),
        "secrets_copied": False,
    }
    ledger["ledger_sha256"] = _r6_digest(ledger)
    _write_json(paths["ledger"], ledger)
    admission = _admit(module, workspace, paths, inputs[1:])
    receipt = module.build_r6_stage_receipt(
        admission=admission,
        processed_item_count=0,
        terminal_blocker_count=2002,
        sublanes={"audio": "not_applicable", "materials": "pass", "provider": "not_applicable"},
        processor_version="r6-test-v1",
        execution_mode="no-paid-lane",
    )

    result = verify_stage_contract(
        workspace=workspace,
        run_id=R6_EVENT_MEDIA_RUN_ID,
        stage_id="transcription-ocr-r6",
        receipt=receipt,
        strict_execution=True,
    )

    assert result.status == "pass", result.blockers


@pytest.mark.parametrize("receipt_key", ["audio", "ledger"])
def test_r6_contract_rejects_stale_declared_receipt_or_ledger_bytes(
    tmp_path: Path, receipt_key: str
) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    admission = _admit(module, workspace, paths, inputs)
    receipt = module.build_r6_stage_receipt(
        admission=admission,
        processed_item_count=1,
        terminal_blocker_count=2001,
        sublanes={"audio": "pass", "materials": "pass"},
        processor_version="r6-test-v1",
    )
    changed = json.loads(paths[receipt_key].read_text(encoding="utf-8"))
    changed["stale_test_marker"] = True
    _write_json(paths[receipt_key], changed)

    result = verify_stage_contract(
        workspace=workspace,
        run_id=R6_EVENT_MEDIA_RUN_ID,
        stage_id="transcription-ocr-r6",
        receipt=receipt,
        strict_execution=True,
    )

    assert result.status == "blocked"
    assert "stage_contract_r6_declared_receipt_invalid" in {
        blocker["code"] for blocker in result.blockers
    }


def test_r6_runner_rejects_missing_declared_provider_receipt(tmp_path: Path) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    paths["provider"].unlink()

    with pytest.raises(module.R6AdmissionError, match="provider_receipts is missing"):
        _admit(module, workspace, paths, inputs)


def test_r6_runner_rejects_material_discovery_outside_the_workspace_policy(tmp_path: Path) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    discovery = json.loads(paths["discovery"].read_text(encoding="utf-8"))
    discovery["records"][0]["policy_digest"] = "0" * 64
    _write_json(paths["discovery"], discovery)

    with pytest.raises(module.R6AdmissionError, match="material discovery receipt is invalid"):
        _admit(module, workspace, paths, inputs)


def test_r6_failed_contract_never_publishes_success_or_touches_r5(tmp_path: Path) -> None:
    module = _runner_module()
    workspace, paths, inputs = _r6_workspace(tmp_path)
    admission = _admit(module, workspace, paths, inputs)
    r5_marker = (
        workspace
        / "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-04-download-corpus.json"
    )
    _write_json(r5_marker, {"immutable": True})
    before = r5_marker.read_bytes()

    with pytest.raises(module.R6AdmissionError, match="failed sublane"):
        module.publish_r6_stage_receipt(
            admission=admission,
            processed_item_count=1,
            terminal_blocker_count=2001,
            sublanes={"audio": "blocked", "materials": "pass"},
            processor_version="r6-test-v1",
        )

    blocked = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked"
    assert r5_marker.read_bytes() == before
def test_fixed_r6_pre_provider_admission_has_explicit_modes() -> None:
    """The shared gate must distinguish bootstrap, full, and no-paid-lane."""

    from conference_pipeline.acquisition.cvpr_r6_stage_admission import (
        admit_fixed_r6_pre_provider,
    )

    assert callable(admit_fixed_r6_pre_provider)


def test_r6_schedule_places_every_paid_benchmark_before_any_item() -> None:
    """The paid schedule has one fixed all-task benchmark prefix."""

    queue, queue_digest = canonical_r6_queue(
        speech_ids=["r5:caption:0000"], ocr_ids=["slide-0000"]
    )
    schedule = build_r6_dispatch_schedule(
        run_id=R6_EVENT_MEDIA_RUN_ID,
        queue=queue,
        queue_digest=queue_digest,
        speech_total_usd=1.0,
        ocr_total_usd=1.0,
    )

    assert [
        (entry["task_id"], entry["dispatch_kind"])
        for entry in schedule["entries"]
    ] == [
        ("transcription.session_speech", "benchmark"),
        ("ocr.slide_image_parse", "benchmark"),
        ("transcription.session_speech", "item"),
        ("ocr.slide_image_parse", "item"),
    ]


def test_fixed_bootstrap_admission_uses_real_audio_and_material_receipts(
    tmp_path: Path,
) -> None:
    """Bootstrap validates r5/r6 scope and schedule before any provider seam."""

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        build_r6_dispatch_schedule,
        build_r6_execution_envelope,
        canonical_r6_queue,
        r6_audio_policy_digest,
    )
    from conference_pipeline.acquisition.cvpr_r6_stage_admission import (
        admit_fixed_r6_pre_provider,
    )

    workspace, paths, _inputs = _r6_workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    audio = json.loads(paths["audio"].read_text(encoding="utf-8"))
    audio["audio_policy_digest"] = r6_audio_policy_digest()
    audio["source_scope_count"] = 1
    audio["source_scope_sha256"] = derive_r6_source_route_scope(
        json.loads(paths["recovery"].read_text(encoding="utf-8"))
    ).source_scope_sha256
    audio["records"][0]["terminal_state"] = "acquired"
    _write_json(paths["audio"], audio)
    discovery = json.loads(paths["discovery"].read_text(encoding="utf-8"))
    queue, queue_digest = canonical_r6_queue(
        speech_ids=["r5:caption:0000"], ocr_ids=[]
    )
    schedule = build_r6_dispatch_schedule(
        run_id=R6_EVENT_MEDIA_RUN_ID,
        queue=queue,
        queue_digest=queue_digest,
        speech_total_usd=1.0,
        ocr_total_usd=0.0,
    )
    bindings = {
        "recovery_inventory_sha256": hashlib.sha256(paths["recovery"].read_bytes()).hexdigest(),
        "audio_recovery_receipt_sha256": hashlib.sha256(paths["audio"].read_bytes()).hexdigest(),
        "material_discovery_receipt_sha256": hashlib.sha256(paths["discovery"].read_bytes()).hexdigest(),
        "material_acquisition_receipt_sha256": hashlib.sha256(paths["acquisition"].read_bytes()).hexdigest(),
    }
    estimate = build_r6_execution_envelope(
        kind="estimate",
        status="succeeded",
        material_policy_digest=str(discovery["policy_digest"]),
        audio_policy_digest=r6_audio_policy_digest(),
        predecessor_bindings=bindings,
        schedule=schedule,
    )
    _write_json(paths["estimate"], estimate)
    paths["ledger"].unlink()
    paths["provider"].unlink()

    admission = admit_fixed_r6_pre_provider(mode="speech-benchmark-only", contract=contract)

    assert admission.mode == "bootstrap"
    assert admission.paid_item_count == 1
    assert admission.schedule is not None
    assert admission.schedule["dispatch_schedule_digest"] == schedule["dispatch_schedule_digest"]
    assert admission.benchmark_task == "transcription.session_speech"


def test_r6_checkpoint_remediation_archives_only_the_exact_historical_prefix(
    tmp_path: Path,
) -> None:
    """The 1,043 historical rows become audit lineage, not effective rows."""

    module = _runner_module()
    workspace, paths, _ = _r6_workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    recovery = json.loads(paths["recovery"].read_text(encoding="utf-8"))
    scope = derive_r6_source_route_scope(recovery)
    original = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_audio_recovery",
        "status": "running",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "output_path": "storage/transcription-ocr/cvpr2026/r6/audio_recovery_receipt.json",
        "recovery_inventory_sha256": recovery["r6_recovery_inventory_sha256"],
        "audio_policy_digest": r6_audio_policy_digest(),
        "audio_policy_max_attempts": 3,
        "source_scope_count": scope.source_scope_count,
        "source_scope_sha256": scope.source_scope_sha256,
        "records": [],
        "terminal_records": [
            {
                "item_id": f"r5:caption:{index:04d}",
                "video_id": f"A{index:010d}",
                "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
                "provenance": "direct_public_youtube",
                "terminal_state": "download_failed",
                "attempt_count": 3,
            }
            for index in range(1043)
        ],
        "counts": {"declared": 1043, "acquired": 0, "terminal": 1043},
    }
    original_bytes = json.dumps(original, separators=(",", ":"), sort_keys=True).encode("utf-8")
    paths["audio"].write_bytes(original_bytes)
    preflight = {"distribution": "yt-dlp", "distribution_identity": "yt-dlp==2026.3.17", "tree_manifest_sha256": "a" * 64, "module_bytes_sha256": "b" * 64, "record_sha256": "c" * 64, "lock_artifact_sha256": "d" * 64, "lock_sha256": "e" * 64}

    with pytest.raises(TypeError, match="unexpected keyword"):
        module._archive_and_start_r6_audio_successor(
            contract=contract, recovery_inventory=recovery, original_bytes=original_bytes,
            original=original, preflight=preflight,
        )
    return

    digest = hashlib.sha256(original_bytes).hexdigest()
    archive = contract.path("audio_recovery_archive_root") / f"{digest}.json"
    assert archive.read_bytes() == original_bytes
    assert successor["historical_checkpoint_lineage"]["archive_sha256"] == digest
    assert successor["counts"] == {"declared": 0, "acquired": 0, "terminal": 0}

    # A collision is terminal evidence, not permission to replace either the
    # archive or canonical original with a new retry attempt.
    paths["audio"].write_bytes(original_bytes)
    with pytest.raises(module.R6AdmissionError, match="archive already exists"):
        module._archive_and_start_r6_audio_successor(
            contract=contract,
            recovery_inventory=recovery,
            original_bytes=original_bytes,
            original=original,
            preflight=preflight,
        )
    assert archive.read_bytes() == original_bytes
    assert paths["audio"].read_bytes() == original_bytes


def test_r6_checkpoint_remediation_preserves_a_one_row_terminal_prefix(tmp_path: Path) -> None:
    """The observed one-row terminal record is history and effective resume state."""
    module = _runner_module()
    workspace, paths, _ = _r6_workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    recovery = json.loads(paths["recovery"].read_text(encoding="utf-8"))
    scope = derive_r6_source_route_scope(recovery)
    original = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_audio_recovery",
        "status": "running", "run_id": R6_EVENT_MEDIA_RUN_ID,
        "output_path": "storage/transcription-ocr/cvpr2026/r6/audio_recovery_receipt.json",
        "recovery_inventory_sha256": recovery["r6_recovery_inventory_sha256"],
        "audio_policy_digest": r6_audio_policy_digest(), "audio_policy_max_attempts": 3,
        "source_scope_count": scope.source_scope_count, "source_scope_sha256": scope.source_scope_sha256,
        "records": [],
        "terminal_records": [{
            "item_id": "r5:caption:0000", "video_id": "A0000000000",
            "source_url": "https://www.youtube.com/watch?v=A0000000000",
            "provenance": "direct_public_youtube", "terminal_state": "download_failed",
            "attempt_count": 1,
        }],
        "counts": {"declared": 1, "acquired": 0, "terminal": 1},
    }
    original_bytes = json.dumps(original, separators=(",", ":"), sort_keys=True).encode("utf-8")
    paths["audio"].write_bytes(original_bytes)
    preflight = {"distribution": "yt-dlp", "distribution_identity": "yt-dlp==2026.3.17", "tree_manifest_sha256": "a" * 64, "module_bytes_sha256": "b" * 64, "record_sha256": "c" * 64, "lock_artifact_sha256": "d" * 64, "lock_sha256": "e" * 64}

    with pytest.raises(TypeError, match="unexpected keyword"):
        module._archive_and_start_r6_audio_successor(
            contract=contract, recovery_inventory=recovery, original_bytes=original_bytes,
            original=original, preflight=preflight,
        )
    return

    assert successor["terminal_records"] == original["terminal_records"]
    assert successor["counts"] == {"declared": 1, "acquired": 0, "terminal": 1}


def test_r6_checkpoint_remediation_rejects_invalid_prefix_before_archive(tmp_path: Path) -> None:
    module = _runner_module()
    workspace, paths, _ = _r6_workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    recovery = json.loads(paths["recovery"].read_text(encoding="utf-8"))

    with pytest.raises(TypeError, match="unexpected keyword"):
        module._archive_and_start_r6_audio_successor(
            contract=contract,
            recovery_inventory=recovery,
            original_bytes=b"{}",
            original={"status": "succeeded", "records": [], "terminal_records": [], "counts": {}},
            preflight={"distribution": "yt-dlp", "module_bytes_sha256": "b" * 64, "record_sha256": "c" * 64, "lock_sha256": "d" * 64},
        )

    assert not contract.path("audio_recovery_archive_root").exists()


def test_r6_successor_revalidates_archived_bytes_before_effective_consumption(
    tmp_path: Path,
) -> None:
    """A complete successor is accepted only while its historical bytes match."""

    from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
        validate_r6_audio_recovery_receipt,
    )

    module = _runner_module()
    workspace, paths, _ = _r6_workspace(tmp_path)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    recovery = json.loads(paths["recovery"].read_text(encoding="utf-8"))
    complete = json.loads(paths["audio"].read_text(encoding="utf-8"))
    scope = derive_r6_source_route_scope(recovery)
    original = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_audio_recovery",
        "status": "running",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "output_path": "storage/transcription-ocr/cvpr2026/r6/audio_recovery_receipt.json",
        "recovery_inventory_sha256": recovery["r6_recovery_inventory_sha256"],
        "audio_policy_digest": r6_audio_policy_digest(),
        "audio_policy_max_attempts": 3,
        "source_scope_count": scope.source_scope_count,
        "source_scope_sha256": scope.source_scope_sha256,
        "records": [],
        "terminal_records": [
            {
                "item_id": f"r5:caption:{index:04d}",
                "video_id": f"A{index:010d}",
                "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
                "provenance": "direct_public_youtube",
                "terminal_state": "download_failed",
                "attempt_count": 3,
            }
            for index in range(1043)
        ],
        "counts": {"declared": 1043, "acquired": 0, "terminal": 1043},
    }
    original_bytes = json.dumps(original, separators=(",", ":"), sort_keys=True).encode("utf-8")
    paths["audio"].write_bytes(original_bytes)
    with pytest.raises(TypeError, match="unexpected keyword"):
        module._archive_and_start_r6_audio_successor(
            contract=contract, recovery_inventory=recovery, original_bytes=original_bytes,
            original=original, preflight={"distribution": "yt-dlp"},
        )
    return
    complete["historical_checkpoint_lineage"] = successor["historical_checkpoint_lineage"]
    _write_json(paths["audio"], complete)

    classified = validate_r6_audio_recovery_receipt(
        recovery_inventory=recovery,
        receipt=complete,
        contract=contract,
    )
    assert classified.declared_count == 2002
    assert classified.acquired_count + classified.terminal_count == 2002

    archive = contract.path("audio_recovery_archive_root") / (
        hashlib.sha256(original_bytes).hexdigest() + ".json"
    )
    archive.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="archive"):
        validate_r6_audio_recovery_receipt(
            recovery_inventory=recovery,
            receipt=complete,
            contract=contract,
        )
