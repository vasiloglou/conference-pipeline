"""Offline security contracts for the fixed CVPR r6 media owner."""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from hashlib import sha256
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import conference_pipeline.acquisition.cvpr_r6_artifact_contract as artifact_contract_module
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_PUBLIC_AUDIO_POLICY,
    R6ArtifactContract,
    R6ArtifactContractError,
    R6RepositoryExecutableCapability,
    r6_active_workspace_lease_descriptor_count,
)
from conference_pipeline.acquisition.cvpr_r6_material_index import (
    MaterialPolicyError,
    build_material_discovery_receipt,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    R6_EVENT_MEDIA_RUN_ID,
    R5LineagePins,
    build_r6_recovery_inventory,
    canonical_r6_recovery_inventory_digest,
    derive_r6_source_route_scope,
    source_route_scope_payload,
)
from conference_pipeline.acquisition.cvpr_event_media import (
    canonical_event_media_inventory_digest,
)
from conference_pipeline.acquisition.youtube_audio_recovery import _assert_options

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = (
    REPO_ROOT
    / "src"
    / "conference_pipeline"
    / "acquisition"
    / "_cvpr_r6_media_owner_impl.py"
)


# Repository files each test reads that the public code export does not carry. The
# repo_data hook in tests/conftest.py reads this table: a decorator would change the
# pinned source fingerprints these tests carry (v6.0 Phase 7).
REPO_DATA = {
    "test_fixed_owner_rejects_authority_drift_before_every_r6_side_effect": ("conferences/cvpr2026/config.yaml",),
    "test_full_recomputed_classification_drift_stops_owner_before_all_effects": ("conferences/cvpr2026/config.yaml",),
    "test_r6_source_scope_rejects_dropped_route_observation_before_live_owner": ("conferences/cvpr2026/config.yaml",),
}


def _owner_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cvpr2026_r6_live_owner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fresh_runner_import_exposes_no_operational_alias_or_receipt_builder() -> None:
    """The CLI module itself must not be a composable continuation capability."""

    module = _owner_module()
    forbidden = {
        "read_r6_audio_checkpoint",
        "write_r6_audio_checkpoint",
        "write_r6_audio_checkpoint_archive",
        "preflight_fixed_r6_yt_dlp",
        "recover_public_youtube_audio",
        "_audio_receipt",
    }
    for name in forbidden:
        assert name not in vars(module)
        assert not hasattr(module, name)
        with pytest.raises(AttributeError):
            getattr(module, name)
    members = dict(inspect.getmembers(module, callable))
    assert not forbidden.intersection(members)
    for _name, value in members.items():
        if not inspect.isfunction(value) or getattr(value, "__module__", None) != module.__name__:
            continue
        assert not forbidden.intersection(value.__code__.co_freevars)
        assert not getattr(value, "__closure__", None)


def _write_fixture_json(path: Path, payload: dict[str, object]) -> str:
    """Write one disposable fixture control record and return its file digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def _temporary_pinned_r5_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> R6ArtifactContract:
    """Materialize the real immutable-r5 reader inputs below a test contract.

    The runner still calls ``_fixed_r5_recovery_inventory``.  Only its three
    immutable pins are redirected to the bytes owned by this fixture, which is
    the same narrow seam used by an isolated repository copy.
    """

    workspace = tmp_path / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    candidates = [
        {
            "record_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "video_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "source_event_urls": [
                f"https://cvpr.thecvf.com/virtual/2026/poster/{index % 3 + 1}"
            ],
            "provenance": ["r5_official_event_page"],
            "caption_status": "youtube_caption_unavailable",
            "caption": None,
            "audio_downloaded": False,
            "video_downloaded": False,
        }
        for index in range(2_002)
    ]
    inventory: dict[str, object] = {
        "run_id": "cvpr2026-live-production-r5-event-media",
        "caption_candidates": candidates,
        "counts": {"caption_candidate_count": 2_002},
    }
    inventory["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(inventory)
    inventory_path = workspace / "storage/download-corpus/event-media/event_media_inventory.json"
    inventory_file_digest = _write_fixture_json(inventory_path, inventory)
    receipt: dict[str, object] = {
        "run_id": "cvpr2026-live-production-r5-event-media",
        "stage_id": "download-corpus",
        "mode": "live",
        "status": "succeeded",
        "stage_contract": {"status": "pass", "stage_id": "download-corpus"},
        "event_media_inventory": {
            "path": "storage/download-corpus/event-media/event_media_inventory.json",
            "inventory_sha256": inventory["event_media_inventory_sha256"],
        },
        "checks": {
            "event_media_inventory_path": "storage/download-corpus/event-media/event_media_inventory.json",
            "event_media_inventory_sha256": inventory_file_digest,
            "event_media_inventory_canonical_sha256": inventory[
                "event_media_inventory_sha256"
            ],
        },
    }
    receipt_path = workspace / module._R5_RECEIPT_RELATIVE_PATH
    receipt_digest = _write_fixture_json(receipt_path, receipt)
    monkeypatch.setattr(module.R6ArtifactContract, "production", lambda: contract)
    monkeypatch.setattr(module, "_R5_RECEIPT_SHA256", receipt_digest)
    monkeypatch.setattr(module, "_R5_INVENTORY_SHA256", inventory_file_digest)
    monkeypatch.setattr(
        module,
        "_R5_INVENTORY_CANONICAL_SHA256",
        str(inventory["event_media_inventory_sha256"]),
    )
    return contract


def _fixture_preflight() -> dict[str, str]:
    """The validated in-process downloader provenance used by contract tests."""

    return {
        "distribution": "yt-dlp",
        "distribution_identity": "yt-dlp==2026.3.17",
        "tree_manifest_sha256": "a" * 64,
        "module_bytes_sha256": "b" * 64,
        "record_sha256": "c" * 64,
        "lock_artifact_sha256": "d" * 64,
        "lock_sha256": "e" * 64,
    }


def _terminal_downloader(
    module: ModuleType, calls: list[tuple[str, str]], *, stop_after: int
):
    """Return only terminal public-source facts, then model an interruption."""

    def _recover(**kwargs: object) -> dict[str, object]:
        if len(calls) >= stop_after:
            # The bounded downloader seam is the only substituted operation.
            # The already persisted running receipt is re-read and classified
            # by the next real owner invocation.
            raise module.R6RecoveryIncompleteError("fixture interruption")
        item_id = str(kwargs["item_id"])
        source_url = str(kwargs["source_url"])
        calls.append((item_id, source_url))
        return {
            "item_id": item_id,
            "video_id": source_url.rsplit("=", 1)[1],
            "source_url": source_url,
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }

    return _recover


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_fixed_owner_publishes_one_closed_interruption_then_reraises(
    monkeypatch: pytest.MonkeyPatch, interruption: type[BaseException]
) -> None:
    """The real owner catches only the two operational interruption classes."""

    module = _owner_module()
    published: list[tuple[str, str, int]] = []
    monkeypatch.setattr(
        module,
        "_run_fixed_r6_media_owner",
        lambda **_kwargs: (_ for _ in ()).throw(interruption()),
    )
    monkeypatch.setattr(
        module,
        "_publish_owner_cli_decision",
        lambda *, phase, reason_code, exit_code: published.append((phase, reason_code, exit_code)),
    )

    with pytest.raises(interruption):
        module.run_fixed_r6_media_owner(live=True)
    assert published == [("supervisor", "worker_interrupted", 2)]


def test_fixed_owner_publishes_unexpected_exception_then_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected owner faults remain observable after one finite decision."""

    module = _owner_module()
    published: list[tuple[str, str, int]] = []
    fault = RuntimeError("fixture-only unexpected owner fault")
    monkeypatch.setattr(
        module,
        "_run_fixed_r6_media_owner",
        lambda **_kwargs: (_ for _ in ()).throw(fault),
    )
    monkeypatch.setattr(
        module,
        "_publish_owner_cli_decision",
        lambda *, phase, reason_code, exit_code: published.append((phase, reason_code, exit_code)),
    )

    with pytest.raises(RuntimeError) as raised:
        module.run_fixed_r6_media_owner(live=True)
    assert raised.value is fault
    assert published == [("pre_admission", "unexpected_owner_failure", 1)]


def test_fixed_owner_leaves_memory_error_unpublished_and_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory exhaustion is never relabeled as an owner decision."""

    module = _owner_module()
    published: list[tuple[str, str, int]] = []
    fault = MemoryError("fixture-only memory exhaustion")
    monkeypatch.setattr(
        module,
        "_run_fixed_r6_media_owner",
        lambda **_kwargs: (_ for _ in ()).throw(fault),
    )
    monkeypatch.setattr(
        module,
        "_publish_owner_cli_decision",
        lambda *, phase, reason_code, exit_code: published.append((phase, reason_code, exit_code)),
    )

    with pytest.raises(MemoryError) as raised:
        module.run_fixed_r6_media_owner(live=True)
    assert raised.value is fault
    assert published == []


@pytest.mark.parametrize("fault_type", [GeneratorExit, type("FixtureFatal", (BaseException,), {})])
def test_fixed_owner_leaves_unapproved_base_exceptions_unpublished(
    monkeypatch: pytest.MonkeyPatch, fault_type: type[BaseException]
) -> None:
    """Only the two operational interruptions are caught above Exception."""

    module = _owner_module()
    published: list[tuple[str, str, int]] = []
    fault = fault_type()
    monkeypatch.setattr(
        module,
        "_run_fixed_r6_media_owner",
        lambda **_kwargs: (_ for _ in ()).throw(fault),
    )
    monkeypatch.setattr(
        module,
        "_publish_owner_cli_decision",
        lambda *, phase, reason_code, exit_code: published.append((phase, reason_code, exit_code)),
    )

    with pytest.raises(fault_type) as raised:
        module.run_fixed_r6_media_owner(live=True)
    assert raised.value is fault
    assert published == []


def _audio_receipt(
    module: ModuleType,
    *,
    recovery_inventory: dict[str, object],
    rows: list[dict[str, object]],
    status: str,
    historical_checkpoint_lineage: dict[str, object] | None = None,
) -> dict[str, object]:
    """Fixture-only producer; the runner intentionally exposes none."""

    scope = module.derive_r6_source_route_scope(recovery_inventory)
    acquired = [dict(row) for row in rows if row.get("terminal_state") == "acquired"]
    for row in acquired:
        if "local_path" in row or not isinstance(row.get("local_media_path"), str):
            raise module.R6AdmissionError("acquired audio row lacks the canonical local_media_path")
    terminal = [dict(row) for row in rows if row.get("terminal_state") != "acquired"]
    receipt: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_audio_recovery",
        "status": status,
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "output_path": "storage/transcription-ocr/cvpr2026/r6/audio_recovery_receipt.json",
        "recovery_inventory_sha256": recovery_inventory["r6_recovery_inventory_sha256"],
        "audio_policy_digest": module.r6_audio_policy_digest(),
        "audio_policy_max_attempts": R6_PUBLIC_AUDIO_POLICY["max_attempts"],
        **module.source_route_scope_payload(scope),
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
    if historical_checkpoint_lineage is not None:
        receipt["historical_checkpoint_lineage"] = dict(historical_checkpoint_lineage)
    return receipt


def test_inprocess_options_reject_any_auth_or_subprocess_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def forbidden(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        raise AssertionError("subprocess must not receive an auth-capable command")

    monkeypatch.setattr(subprocess, "run", forbidden)

    with pytest.raises(Exception, match="policy"):
        _assert_options({"noplaylist": True, "retries": 0, "netrc": True}, download=False)

    assert calls == []


def test_actual_audio_receipt_binds_the_fixed_public_audio_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The producer, rather than a hand-built fixture, carries policy proof."""

    module = _owner_module()
    monkeypatch.setattr(module, "derive_r6_source_route_scope", lambda _recovery: object())
    monkeypatch.setattr(
        module,
        "source_route_scope_payload",
        lambda _scope: {"source_scope_count": 1, "source_scope_sha256": "a" * 64},
    )
    receipt = _audio_receipt(
        module,
        recovery_inventory={"r6_recovery_inventory_sha256": "b" * 64},
        rows=[
            {
                "item_id": "talk-1",
                "terminal_state": "download_failed",
                "video_id": "abcDEF12345",
            }
        ],
        status="succeeded",
    )
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import r6_audio_policy_digest

    assert receipt["audio_policy_digest"] == r6_audio_policy_digest()
    assert receipt["audio_policy_max_attempts"] == R6_PUBLIC_AUDIO_POLICY["max_attempts"]


def test_audio_receipt_rejects_a_legacy_or_dual_acquired_path_before_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owner cannot persist an acquired row that needs field translation."""

    module = _owner_module()
    monkeypatch.setattr(module, "derive_r6_source_route_scope", lambda _recovery: object())
    monkeypatch.setattr(
        module,
        "source_route_scope_payload",
        lambda _scope: {"source_scope_count": 1, "source_scope_sha256": "a" * 64},
    )
    acquired = {
        "item_id": "talk-1",
        "video_id": "abcDEF12345",
        "terminal_state": "acquired",
        "local_media_path": "storage/transcription-ocr/cvpr2026/r6/raw-audio/talk-1.mp3",
        "local_path": "storage/transcription-ocr/cvpr2026/r6/raw-audio/talk-1.mp3",
    }

    with pytest.raises(module.R6AdmissionError, match="local_media_path"):
        _audio_receipt(
            module,
            recovery_inventory={"r6_recovery_inventory_sha256": "b" * 64},
            rows=[acquired],
            status="running",
        )


def test_real_public_audio_producer_reaches_running_resume_and_complete_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One fake public download keeps its canonical path through every owner boundary."""

    from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
        validate_r6_audio_recovery_receipt,
        validate_r6_audio_recovery_resume,
    )

    module = _owner_module()
    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    recovery = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_media_recovery_inventory",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "counts": {"asr_fallback_required": 2_002},
        "asr_fallback_required": [
            {
                "r5_record_id": f"r5:caption:{index:04d}",
                "video_id": f"A{index:010d}",
                "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
                "state": "asr_fallback_required",
                "source_classification": "public_youtube_candidate",
                "source_event_urls": ["https://cvpr.thecvf.com/virtual/2026/poster/1"],
                "provenance": ["r5_official_event_page"],
            }
            for index in range(2_002)
        ],
    }
    scope = derive_r6_source_route_scope(recovery, verify_inventory_digest=False)
    recovery.update(source_route_scope_payload(scope))
    recovery["r6_recovery_inventory_sha256"] = canonical_r6_recovery_inventory_digest(recovery)
    preflight = {"downloader": "yt-dlp", "module_identity": "repository_interpreter", "version_sha256": "b" * 64, "source_sha256": "c" * 64}
    running: dict[str, object] | None = None
    payload = b"fixture-audio"
    target = contract.audio_root / "r5:caption:0000-A0000000000.mp3"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    def producer(**kwargs: object) -> dict[str, object]:
        if kwargs["item_id"] == "r5:caption:0000":
            return {"item_id": kwargs["item_id"], "video_id": "A0000000000", "source_url": kwargs["source_url"], "provenance": "direct_public_youtube", "terminal_state": "acquired", "duration_seconds": 60.0, "local_media_path": str(target), "sha256": __import__("hashlib").sha256(payload).hexdigest(), "mime_type": "audio/mpeg", "byte_count": len(payload), "attempt_count": 1}
        return {
            "item_id": kwargs["item_id"],
            "video_id": str(kwargs["source_url"]).rsplit("=", 1)[1],
            "source_url": kwargs["source_url"],
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }

    def capture(*, payload: dict[str, object], **_kwargs: object) -> None:
        nonlocal running
        if payload["counts"] == {"declared": 1, "acquired": 1, "terminal": 0}:
            running = payload

    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", lambda: preflight)
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio", producer)
    monkeypatch.setattr("conference_pipeline.acquisition.cvpr_r6_artifact_contract.write_r6_audio_checkpoint", capture)
    # The exact-owner fixture below still enumerates all 2,002 ordered rows;
    # receipt-shape validation has dedicated exhaustive tests and would turn
    # this behavioural equivalence test into quadratic control-plane work.
    class _Classified:
        acquired_records = ()
        terminal_records = ()
        acquired_count = 1
        terminal_count = 2001
        declared_count = 2002
    monkeypatch.setattr(module, "validate_r6_audio_recovery_resume", lambda **_kwargs: _Classified())
    monkeypatch.setattr(module, "validate_r6_audio_recovery_receipt", lambda **_kwargs: _Classified())
    monkeypatch.setattr(
        module,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("audio checkpoints must not use lexical atomic_write_json")
        ),
    )

    # Receipt-shape validation is a one-row running-checkpoint contract.  Do
    # not turn it into 2,002 fake recoveries; the row-oracle test below owns
    # exhaustive semantic coverage with a bounded representative count.
    first = recovery["asr_fallback_required"][0]
    running = _audio_receipt(
        module,
        recovery_inventory=recovery,
        rows=[producer(item_id=first["r5_record_id"], source_url=first["source_url"])],
        status="running",
    )
    running_record = running["records"][0]
    assert "local_media_path" in running_record
    assert "local_path" not in running_record
    assert validate_r6_audio_recovery_resume(
        recovery_inventory=recovery, receipt=running, contract=contract
    ).acquired_count == 1


def test_exact_2002_row_oracle_manifest_is_complete_and_bounded() -> None:
    """All rows are proven once; only a partition representative needs recovery."""
    from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
        build_r6_audio_row_oracle_manifest,
    )
    recovery = {
        "asr_fallback_required": [
            {
                "r5_record_id": f"r5:caption:{index:04d}",
                "video_id": f"A{index:010d}",
                "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
                "state": "asr_fallback_required",
                "source_classification": "public_youtube_candidate",
            }
            for index in range(2_002)
        ],
        "r6_recovery_inventory_sha256": "a" * 64,
    }
    manifest = build_r6_audio_row_oracle_manifest(
        recovery, prior_oracle=lambda row: "canonical-public" if row["source_url"].startswith("https://www.youtube.com/") else "invalid"
    )
    assert manifest["row_count"] == 2_002
    assert len(manifest["rows"]) == 2_002
    assert len(manifest["partitions"]) == 1
    assert manifest["partitions"][0]["representative_index"] == 0
    assert manifest["partitions"][0]["row_count"] == 2_002
    assert len(manifest["manifest_sha256"]) == 64


def test_row_authority_rejects_reordered_or_identity_drift_before_partition() -> None:
    """A work queue cannot define the oracle that later admits it."""
    from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
        build_r6_audio_recovery_authority,
        partition_r6_audio_recovery_authority,
        validate_r6_audio_recovery_authority_queue,
    )

    queue = [
        {
            "r5_record_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "state": "asr_fallback_required",
            "source_classification": "public_youtube_candidate",
        }
        for index in range(2_002)
    ]
    inventory = {"r6_recovery_inventory_sha256": "a" * 64, "asr_fallback_required": queue}
    authority = build_r6_audio_recovery_authority(
        inventory, prior_oracle=lambda row: "public" if int(row["r5_record_id"].rsplit(":", 1)[1]) % 2 else "blocked"
    )
    assert len(validate_r6_audio_recovery_authority_queue(
        authority, queue, recovery_inventory_sha256="a" * 64
    )) == 2_002
    assert sum(partition["row_count"] for partition in partition_r6_audio_recovery_authority(
        authority, queue, recovery_inventory_sha256="a" * 64
    )) == 2_002
    reordered = list(reversed(queue))
    with pytest.raises(ValueError, match="ordered row"):
        validate_r6_audio_recovery_authority_queue(
            authority, reordered, recovery_inventory_sha256="a" * 64
        )
    altered = [dict(row) for row in queue]
    altered[17]["source_url"] = "https://www.youtube.com/watch?v=A0000000000"
    with pytest.raises(ValueError, match="ordered row"):
        validate_r6_audio_recovery_authority_queue(
            authority, altered, recovery_inventory_sha256="a" * 64
        )


def test_exact_2002_owner_stops_after_lease_redirect_before_downloader_or_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redirected batch lease cannot reach either owner side-effect seam."""

    module = _owner_module()
    repository = tmp_path / "repo"
    workspace = repository / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    external = tmp_path / "external"
    (external / "conferences" / "cvpr2026").mkdir(parents=True)
    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    contract = R6ArtifactContract.production()
    recovery = {
        "r6_recovery_inventory_sha256": "a" * 64,
        "asr_fallback_required": [
            {
                "r5_record_id": f"r5:caption:{index:04d}",
                "video_id": f"A{index:010d}",
                "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
                "state": "asr_fallback_required",
                "source_classification": "public_youtube_candidate",
            }
            for index in range(2_002)
        ],
    }
    preflight = {
        "downloader": "yt-dlp",
        "executable_identity": "repository_venv",
        "version_sha256": "b" * 64,
    }
    original_acquire = R6ArtifactContract._acquire_workspace_lease

    def redirect_after_lease(self: R6ArtifactContract) -> object:
        lease = original_acquire(self)
        (repository / "conferences").rename(tmp_path / "original-conferences")
        (repository / "conferences").symlink_to(external / "conferences", target_is_directory=True)
        return lease

    downloader_calls: list[object] = []
    checkpoint_writes: list[object] = []
    monkeypatch.setattr(R6ArtifactContract, "_acquire_workspace_lease", redirect_after_lease)
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", lambda: preflight)
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        lambda **_kwargs: downloader_calls.append("downloader") or {},
    )
    monkeypatch.setattr(
        module,
        "atomic_write_json",
        lambda *_args, **_kwargs: checkpoint_writes.append("checkpoint"),
    )

    # The worker is sealed: importing it cannot supply a contract or queue.
    # Descriptor-redirection coverage continues through the production owner.
    with pytest.raises(TypeError, match="unexpected keyword"):
        module._recover_public_audio(contract=contract, recovery_inventory=recovery)

    assert len(recovery["asr_fallback_required"]) == 2_002
    assert downloader_calls == []
    assert checkpoint_writes == []


def test_owner_stops_before_workers_when_inprocess_preflight_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _owner_module()
    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", lambda: (_ for _ in ()).throw(module.R6DownloaderPreflightError("missing")))
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("worker must remain unreachable")))
    # A direct caller cannot provide any authority, queue, preflight, or root.
    with pytest.raises(TypeError, match="unexpected keyword"):
        module._recover_public_audio(contract=contract, recovery_inventory={"r6_recovery_inventory_sha256": "a" * 64, "asr_fallback_required": []})


def test_imported_continuation_helpers_reject_forged_key_before_every_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the owner-local continuation key can enter a worker/checkpoint helper."""

    module = _owner_module()
    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    touched: list[str] = []
    monkeypatch.setattr(module, "_load_existing_checkpoint", lambda **_kwargs: touched.append("read"))
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", lambda: touched.append("preflight"))
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio", lambda **_kwargs: touched.append("worker"))
    monkeypatch.setattr(module, "_load_repository_dotenv", lambda *_args: touched.append("dotenv"))
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: touched.append("observe"))
    monkeypatch.setattr(module, "run_fixed_r6_material_acquisition", lambda **_kwargs: touched.append("material"))

    with pytest.raises(TypeError, match="unexpected keyword"):
        module._supervisor_checkpoint_state(
            contract=contract,
            workspace_lease=object(),
            recovery_inventory={},
            authority=object(),
            _owner_key=object(),
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        module._recover_public_audio_in_scoped_context(
            contract=contract,
            recovery_inventory={},
            workspace_lease=object(),
            authority=object(),
            validated_rows=(),
            max_new_rows=1,
            preflight=None,
            _owner_key=object(),
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        module._supervise_fixed_r6_audio(invocation=object())
    with pytest.raises(TypeError, match="unexpected keyword"):
        module._recover_public_audio(invocation=object())
    assert touched == []
    assert not hasattr(module, "_FIXED_R6_INVOCATIONS")
    assert not hasattr(module, "_FIXED_R6_PREFIX_CACHES")


def test_all_importable_continuation_legacy_routes_have_no_capability_inputs() -> None:
    """Forged identity objects cannot reopen an owner-only continuation."""

    module = _owner_module()
    forged = object.__new__(type("forged", (), {"__qualname__": "x._LexicalContinuationKey"}))
    legacy = (
        "_load_existing_checkpoint",
        "_supervisor_checkpoint_state",
        "_supervise_fixed_r6_audio",
        "_archived_terminal_prefix_rows",
        "_archive_and_start_r6_audio_successor",
        "_recover_public_audio",
        "_recover_public_audio_in_scoped_context",
    )
    callable_members = {
        name: value for name, value in inspect.getmembers(module, callable)
        if getattr(value, "__module__", None) == module.__name__
    }
    for name in legacy:
        route = callable_members[name]
        assert not inspect.signature(route).parameters
        with pytest.raises(TypeError):
            route(forged)


def test_successor_authority_has_no_importable_arbitrary_input_issuer() -> None:
    """No runner, contract, or owner surface accepts successor authority."""

    module = _owner_module()
    modules = (module, artifact_contract_module)
    issuer_fragments = ("issue", "issuer", "register", "mint", "witness")
    for candidate in modules:
        surfaces = {
            name: value for name, value in inspect.getmembers(candidate, callable)
            if getattr(value, "__module__", None) == candidate.__name__
        }
        assert not [
            name for name in surfaces
            if any(fragment in name.lower() for fragment in issuer_fragments)
        ]

    assert "_consume_validated_successor_capability" not in vars(module)
    assert "successor_capability" not in inspect.signature(
        module.run_fixed_r6_media_owner
    ).parameters


def test_successor_authority_has_no_reentrant_consumer_surface() -> None:
    """The owner exports no capability consumer for arbitrary callers to reenter."""

    module = _owner_module()
    assert not any(
        "successor" in name.lower() and "capability" in name.lower()
        for name, value in inspect.getmembers(module, callable)
        if getattr(value, "__module__", None) == module.__name__
    )


def test_every_owner_entry_rejects_forged_successor_authority_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No owner entrypoint has a caller-supplied successor-control slot."""

    module = _owner_module()
    marker: list[str] = []
    monkeypatch.setattr(
        module,
        "_run_fixed_r6_media_owner",
        lambda *, live: marker.append("owner") or 0,
    )
    forbidden = ("successor_capability", "successor_witness", "authority_record")
    entries = (
        module.main,
        module.run_fixed_r6_media_owner,
        module._run_fixed_r6_media_owner,
    )
    for entry in entries:
        parameters = inspect.signature(entry).parameters
        assert not set(parameters).intersection(forbidden)
        for name in forbidden:
            with pytest.raises(TypeError):
                entry(live=True, **{name: object()})
            assert marker == []


def test_owner_validates_all_2002_authority_rows_before_preflight_or_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queue order/identity drift cannot reach even a harmless preflight."""
    module = _owner_module()
    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    queue = [{
        "r5_record_id": f"r5:caption:{index:04d}",
        "video_id": f"A{index:010d}",
        "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
        "state": "asr_fallback_required", "source_classification": "public_youtube_candidate",
    } for index in range(2_002)]
    recovery = {"r6_recovery_inventory_sha256": "a" * 64, "asr_fallback_required": queue}
    # The authority builder accepts control-plane fixtures with no envelope;
    # mutate after construction to exercise the actual owner seam.
    from conference_pipeline.acquisition.cvpr_r6_media_recovery import build_r6_audio_recovery_authority
    frozen = build_r6_audio_recovery_authority(recovery, prior_oracle=lambda _row: "public")
    recovery["asr_fallback_required"] = list(reversed(queue))
    touched: list[str] = []
    monkeypatch.setattr(module, "build_r6_audio_recovery_authority", lambda *_args, **_kwargs: frozen)
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", lambda: touched.append("preflight"))
    monkeypatch.setattr("conference_pipeline.acquisition.cvpr_r6_artifact_contract.write_r6_audio_checkpoint", lambda **_kwargs: touched.append("checkpoint"))
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio", lambda **_kwargs: touched.append("worker"))
    with pytest.raises(TypeError, match="unexpected keyword"):
        module._recover_public_audio(contract=contract, recovery_inventory=recovery)
    assert touched == []


@pytest.mark.parametrize(
    ("variant", "indexes"),
    (
        ("later-terminal-inserted", (0, 1, 1_001)),
        ("later-terminal-reordered", (0, 2, 1)),
    ),
)
def test_resume_rejects_nonprefix_effective_rows_before_every_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    indexes: tuple[int, ...],
) -> None:
    """Individually valid later terminal rows cannot bypass resume admission."""

    module = _owner_module()
    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    queue = [
        {
            "r5_record_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "state": "asr_fallback_required",
            "source_classification": "public_youtube_candidate",
        }
        for index in range(2_002)
    ]
    recovery = {"r6_recovery_inventory_sha256": "a" * 64, "asr_fallback_required": queue}
    authority = module.build_r6_audio_recovery_authority(
        recovery, prior_oracle=lambda _row: "direct_public_youtube:max_attempts=3"
    )
    validated_rows = module.validate_r6_audio_recovery_authority_queue(
        authority, queue, recovery_inventory_sha256="a" * 64
    )
    effective_rows = [
        {
            "item_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
        for index in indexes
    ]
    touched: list[str] = []
    successor = {
        "status": "running",
        "historical_checkpoint_lineage": {"synthetic": variant},
    }
    monkeypatch.setattr(
        module,
        "_load_existing_checkpoint",
        lambda **_kwargs: (b"synthetic", successor),
    )
    # The shared receipt classifier historically accepted the rows as an
    # individually valid set.  This seam isolates the owner admission that
    # must now bind its effective sequence to the full immutable authority.
    monkeypatch.setattr(module, "_validate_audio_resume", lambda **_kwargs: effective_rows)
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", lambda: touched.append("preflight"))
    monkeypatch.setattr("conference_pipeline.acquisition.cvpr_r6_artifact_contract.write_r6_audio_checkpoint", lambda **_kwargs: touched.append("checkpoint"))
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio", lambda **_kwargs: touched.append("worker"))
    monkeypatch.setattr(module, "_load_repository_dotenv", lambda *_args: touched.append("dotenv"))

    with pytest.raises(TypeError, match="unexpected keyword"):
        module._recover_public_audio(
            contract=contract,
            recovery_inventory=recovery,
            authority=authority,
            validated_rows=validated_rows,
        )

    assert touched == []


def test_owner_malformed_admission_stops_before_dotenv_network_or_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _owner_module()
    _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    touched: list[str] = []

    def malformed(_contract: object) -> dict[str, object]:
        raise module.R6AdmissionError("immutable r5 lineage mismatch")

    monkeypatch.setattr(module, "_fixed_r5_recovery_inventory", malformed)
    monkeypatch.setattr(module, "_load_repository_dotenv", lambda _contract: touched.append("env"))
    monkeypatch.setattr(module, "cvpr_authenticated_event_opener_factory", lambda **_kwargs: touched.append("auth"))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: touched.append("child"))

    with pytest.raises(module.R6AdmissionError, match="authority admission"):
        module.run_fixed_r6_media_owner(live=True)

    assert touched == []


def test_cli_sanitizes_only_current_admission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The public owner boundary must not leak an invalid receipt's details."""

    module = _owner_module()
    _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    touched: list[str] = []

    def malformed(_contract: object) -> dict[str, object]:
        raise module.R6AdmissionError(
            "stale receipt /tmp/private-checkpoint.json token=do-not-print"
        )

    monkeypatch.setattr(module, "_fixed_r5_recovery_inventory", malformed)
    monkeypatch.setattr(module, "_load_repository_dotenv", lambda _contract: touched.append("env"))
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: touched.append("observe"))

    assert module.main(["--live"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert touched == []

    def unrelated(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("programming error")

    monkeypatch.setattr(module, "run_fixed_r6_media_owner", unrelated)
    with pytest.raises(RuntimeError, match="programming error"):
        module.main(["--live"])


@pytest.mark.parametrize(
    "mutation",
    ("reordered", "identity", "url", "duplicate", "omission", "classification"),
)
def test_fixed_owner_rejects_authority_drift_before_every_r6_side_effect(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public entrypoint, rather than its worker, is the first gate."""

    module = _owner_module()
    queue = [
        {
            "r5_record_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "state": "asr_fallback_required",
            "source_classification": "public_youtube_candidate",
        }
        for index in range(2_002)
    ]
    inventory = {
        "r6_recovery_inventory_sha256": "a" * 64,
        "asr_fallback_required": queue,
    }
    frozen = module.build_r6_audio_recovery_authority(
        inventory, prior_oracle=lambda _row: "direct_public_youtube:max_attempts=1"
    )
    mutated = deepcopy(inventory)
    rows = mutated["asr_fallback_required"]
    assert isinstance(rows, list)
    if mutation == "reordered":
        rows.reverse()
    elif mutation == "identity":
        rows[0]["r5_record_id"] = "r5:caption:changed"
    elif mutation == "url":
        rows[0]["source_url"] = "https://www.youtube.com/watch?v=Changed1234"
    elif mutation == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif mutation == "omission":
        rows.pop()
    else:
        rows[0]["source_classification"] = "authenticated_only"

    touched: list[str] = []
    monkeypatch.setattr(module, "_fixed_r5_recovery_inventory", lambda _contract: mutated)
    monkeypatch.setattr(module, "build_r6_audio_recovery_authority", lambda *_args, **_kwargs: frozen)
    monkeypatch.setattr(module, "atomic_write_r6_json", lambda **_kwargs: touched.append("inventory-write"))
    monkeypatch.setattr(module, "_recover_public_audio", lambda **_kwargs: touched.append("audio"))
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: touched.append("observe"))
    monkeypatch.setattr(module, "run_fixed_r6_material_acquisition", lambda **_kwargs: touched.append("acquire"))
    monkeypatch.setattr(module, "_load_repository_dotenv", lambda _contract: touched.append("dotenv"))
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", lambda: touched.append("preflight"))
    with pytest.raises(module.R6AdmissionError, match="authority"):
        # The dry owner boundary isolates immutable-authority admission from
        # the separately tested live checkpoint-binding boundary.
        module.run_fixed_r6_media_owner(live=False)

    assert touched == []


@pytest.mark.parametrize(
    "conflict",
    ("inventory_digest", "scope_digest", "scope_count", "source_routes"),
)
def test_pre_authority_binding_rejects_checkpoint_facts_from_another_inventory(
    conflict: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A syntactically valid checkpoint cannot select its own recovery scope."""

    module = _owner_module()
    contract = _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    recovery = module._fixed_r5_recovery_inventory(contract)
    checkpoint = _audio_receipt(
        module, recovery_inventory=recovery, rows=[], status="running"
    )
    if conflict == "inventory_digest":
        checkpoint["recovery_inventory_sha256"] = "f" * 64
    elif conflict == "scope_digest":
        checkpoint["source_scope_sha256"] = "e" * 64
    elif conflict == "scope_count":
        checkpoint["source_scope_count"] = int(checkpoint["source_scope_count"]) + 1
    else:
        routes = checkpoint["source_routes"]
        assert isinstance(routes, list) and routes
        hostile_route = dict(routes[0])
        hostile_route["route_id"] = "cvpr2026:r6:route:" + "f" * 64
        checkpoint["source_routes"] = [hostile_route, *routes[1:]]

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        read_r6_audio_checkpoint,
        write_r6_audio_checkpoint,
    )

    write_r6_audio_checkpoint(contract=contract, payload=checkpoint)
    before = read_r6_audio_checkpoint(contract=contract)
    assert before is not None
    touched: list[str] = []
    monkeypatch.setattr(module.R6ArtifactContract, "production", lambda: contract)
    monkeypatch.setattr(module, "_validate_fixed_r6_owner_authority", lambda _inventory: pytest.fail("authority reached"))
    monkeypatch.setattr(module, "_recover_public_audio", lambda **_kwargs: touched.append("audio"))
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: touched.append("observe"))
    monkeypatch.setattr(module, "run_fixed_r6_material_acquisition", lambda **_kwargs: touched.append("materials"))

    assert module.main(["--live"]) == 1
    assert read_r6_audio_checkpoint(contract=contract) == before
    assert touched == []
    decision = json.loads(contract.path("cli_decision").read_text(encoding="utf-8"))
    assert decision == {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "phase": "artifact",
        "reason_code": "artifact_unreadable_or_invalid",
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


def test_pre_authority_binding_is_reachable_only_for_trusted_inventory_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matching inventory-derived checkpoint retains the closed binding shape."""

    module = _owner_module()
    contract = _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    recovery = module._fixed_r5_recovery_inventory(contract)
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import write_r6_audio_checkpoint

    write_r6_audio_checkpoint(
        contract=contract,
        payload=_audio_receipt(module, recovery_inventory=recovery, rows=[], status="running"),
    )
    lease = contract._acquire_workspace_lease()
    with contract.workspace_lease_context(lease):
        binding = module._pre_authority_cli_binding(
            contract=contract, workspace_lease=lease, recovery_inventory=recovery
        )
    scope = module.derive_r6_source_route_scope(recovery)
    assert binding["binding_state"] == "reachable"
    assert binding["recovery_inventory_sha256"] == recovery["r6_recovery_inventory_sha256"]
    assert binding["source_scope_sha256"] == scope.source_scope_sha256


@pytest.mark.parametrize(
    ("field", "mutation"),
    (
        ("state", "missing"),
        ("state", None),
        ("state", "unknown"),
        ("state", "caption_available"),
        ("source_classification", "missing"),
        ("source_classification", None),
        ("source_classification", "unknown"),
        ("source_classification", "authenticated_only"),
    ),
    ids=(
        "state-missing", "state-null", "state-unknown", "state-mismatch",
        "source-missing", "source-null", "source-unknown", "source-mismatch",
    ),
)
def test_full_recomputed_classification_drift_stops_owner_before_all_effects(
    field: str, mutation: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A freshly hashed 2,002-row receipt cannot launder classification drift."""

    module = _owner_module()
    rows = [
        {
            "r5_record_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "state": "asr_fallback_required",
            "source_classification": "public_youtube_candidate",
        }
        for index in range(2_002)
    ]
    inventory: dict[str, object] = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_media_recovery_inventory",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "counts": {"asr_fallback_required": 2_002},
        "asr_fallback_required": rows,
    }
    if mutation == "missing":
        rows[1].pop(field)
    else:
        rows[1][field] = mutation
    inventory["r6_recovery_inventory_sha256"] = canonical_r6_recovery_inventory_digest(inventory)

    touched: list[str] = []
    monkeypatch.setattr(module, "_fixed_r5_recovery_inventory", lambda _contract: inventory)
    monkeypatch.setattr(module, "atomic_write_r6_json", lambda **_kwargs: touched.append("inventory"))
    monkeypatch.setattr(module, "_recover_public_audio", lambda **_kwargs: touched.append("audio"))
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: touched.append("observe"))
    monkeypatch.setattr(module, "run_fixed_r6_material_acquisition", lambda **_kwargs: touched.append("materials"))
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", lambda: touched.append("preflight"))
    monkeypatch.setattr(module, "_load_repository_dotenv", lambda _contract: touched.append("dotenv"))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: touched.append("child"))

    with pytest.raises(module.R6AdmissionError, match="authority"):
        # Authority classification is a dry pre-live gate.  Live mode adds
        # the independently tested descriptor binding before it.
        module.run_fixed_r6_media_owner(live=False)

    assert touched == []


def test_owner_rejects_a_running_or_prefix_audio_result_before_every_later_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A resumable checkpoint is never an owner-success capability."""

    module = _owner_module()
    queue = [
        {
            "r5_record_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "state": "asr_fallback_required",
            "source_classification": "public_youtube_candidate",
            "source_event_urls": [
                f"https://cvpr.thecvf.com/virtual/2026/poster/{index % 3}"
            ],
            "provenance": ["r5_official_event_page"],
        }
        for index in range(2_002)
    ]
    inventory: dict[str, object] = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_media_recovery_inventory",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "counts": {"asr_fallback_required": 2_002},
        "asr_fallback_required": queue,
    }
    inventory["r6_recovery_inventory_sha256"] = canonical_r6_recovery_inventory_digest(inventory)
    scope = derive_r6_source_route_scope(inventory, verify_inventory_digest=False)
    inventory.update(source_route_scope_payload(scope))
    inventory["r6_recovery_inventory_sha256"] = canonical_r6_recovery_inventory_digest(inventory)

    # Deliberately use a non-boundary prefix: success must not depend on an
    # incidental internal chunk size.
    prefix = [
        {
            "item_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
        for index in range(31)
    ]
    # The observed frozen lineage is 1,043 terminal records followed by a
    # 31-row successor prefix.  This fixture advances only the successor to
    # row 235, preserving every prior identity and never re-enqueuing either
    # historical population.  The worker seam is deliberately bounded here;
    # the owner boundary, not an incidental worker chunk size, must decide the
    # outcome.
    archived = [
        {
            "item_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
        for index in range(1_043)
    ]
    successor_prefix = prefix
    continued = [
        {
            "item_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
        for index in range(235)
    ]
    assert [row["item_id"] for row in archived] == [
        f"r5:caption:{index:04d}" for index in range(1_043)
    ]
    assert [row["item_id"] for row in successor_prefix] == [
        f"r5:caption:{index:04d}" for index in range(31)
    ]
    assert [row["item_id"] for row in continued[:31]] == [
        row["item_id"] for row in successor_prefix
    ]
    assert len({row["item_id"] for row in continued}) == 235
    # The archived checkpoint is immutable evidence, while its admitted
    # successor retains the exact prior prefix as control-plane state; the
    # bounded worker itself is never called in this test, so no historical or
    # prefix identity can be retried.
    assert len(archived) == 1_043
    # Use the production receipt builder, then let the real resume classifier
    # consume it at the owner boundary.  A bounded worker seam is sufficient
    # to stop at row 235; it must not replace receipt classification.
    running = _audio_receipt(
        module,
        recovery_inventory=inventory, rows=continued, status="running"
    )
    # The downstream guard is deliberately a pure admission boundary.  It no
    # longer replaces the lexical continuation operation with an importable
    # supervisor seam.
    contract = R6ArtifactContract.for_test_fixture(tmp_path)
    authority, _validated_rows = module._validate_fixed_r6_owner_authority(inventory)
    with pytest.raises(module.R6RecoveryIncompleteError, match="resumable and incomplete"):
        module._require_complete_audio_recovery(
            contract=contract,
            recovery_inventory=inventory,
            authority=authority,
            receipt=running,
        )


def test_real_archive_successor_continuation_reaches_235_without_retries_or_downstream_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the production archive/resume loop, not a fabricated receipt.

    The temporary source tree contains no real CVPR storage.  The only seam is
    the public downloader's terminal result: it interrupts after the exact
    prefix that makes the persisted successor a canonical 31-row, then
    235-row, running receipt.
    """

    module = _owner_module()
    contract = _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    recovery = module._fixed_r5_recovery_inventory(contract)
    historical_rows = [
        {
            "item_id": row["r5_record_id"],
            "video_id": row["video_id"],
            "source_url": row["source_url"],
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
        for row in recovery["asr_fallback_required"][:1_043]
    ]
    original = _audio_receipt(
        module,
        recovery_inventory=recovery, rows=historical_rows, status="running"
    )
    # The real owner reads this descriptor-backed current checkpoint, archives
    # its exact bytes, and writes its successor itself.
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        archive_r6_audio_checkpoint_relative_path,
        read_r6_audio_checkpoint,
        write_r6_audio_checkpoint,
        write_r6_audio_checkpoint_archive,
    )

    write_r6_audio_checkpoint(contract=contract, payload=original)
    original_bytes = read_r6_audio_checkpoint(contract=contract)
    assert original_bytes is not None
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", _fixture_preflight)
    downstream: list[str] = []
    monkeypatch.setattr(
        module, "_observe_official_material_routes", lambda **_kwargs: downstream.append("observe")
    )
    monkeypatch.setattr(
        module,
        "run_fixed_r6_material_acquisition",
        lambda **_kwargs: downstream.append("material"),
    )
    monkeypatch.setattr(module, "_load_repository_dotenv", lambda *_args: downstream.append("dotenv"))

    first_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        _terminal_downloader(module, first_calls, stop_after=31),
    )
    with pytest.raises(module.R6RecoveryIncompleteError, match="fixture interruption"):
        module.run_fixed_r6_media_owner(live=True)

    archive = contract.path("audio_recovery_archive_root") / (
        __import__("hashlib").sha256(original_bytes).hexdigest() + ".json"
    )
    assert archive.read_bytes() == original_bytes
    first_successor_bytes = read_r6_audio_checkpoint(contract=contract)
    assert first_successor_bytes is not None
    first_successor = json.loads(first_successor_bytes)
    assert first_successor["status"] == "running"
    assert first_successor["counts"] == {"declared": 31, "acquired": 0, "terminal": 31}
    assert [item_id for item_id, _url in first_calls] == [
        row["r5_record_id"] for row in recovery["asr_fallback_required"][1_043:1_074]
    ]

    second_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        _terminal_downloader(module, second_calls, stop_after=204),
    )
    with pytest.raises(module.R6RecoveryIncompleteError, match="fixture interruption"):
        module.run_fixed_r6_media_owner(live=True)

    partial_bytes = read_r6_audio_checkpoint(contract=contract)
    assert partial_bytes is not None
    partial = json.loads(partial_bytes)
    assert partial["status"] == "running"
    assert partial["counts"] == {"declared": 235, "acquired": 0, "terminal": 235}
    assert archive.read_bytes() == original_bytes
    assert first_successor_bytes != partial_bytes
    all_calls = first_calls + second_calls
    assert len(all_calls) == 235
    historical_identities = {
        (row["item_id"], row["video_id"]) for row in historical_rows
    }
    prefix_identities = {
        (row["item_id"], row["video_id"])
        for row in first_successor["terminal_records"]
    }
    assert not historical_identities.intersection(
        (item_id, url.rsplit("=", 1)[1]) for item_id, url in all_calls
    )
    assert not prefix_identities.intersection(
        (item_id, url.rsplit("=", 1)[1]) for item_id, url in second_calls
    )
    assert downstream == []

    # The public CLI must expose the same incomplete control-plane outcome
    # without printing the checkpoint or any fixture path.
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        _terminal_downloader(module, [], stop_after=0),
    )
    assert module.main(["--live"]) == 2
    diagnostic = json.loads(
        contract.path("continuation_diagnostic").read_text(encoding="utf-8")
    )
    # The public result is nonzero and its only durable explanation is a
    # receipt-bound, non-sensitive reason -- never the checkpoint or URL set.
    assert diagnostic == {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_continuation_diagnostic",
        "reason_code": "worker_no_progress",
        "recovery_inventory_sha256": recovery["r6_recovery_inventory_sha256"],
        "audio_policy_digest": module.r6_audio_policy_digest(),
        "source_scope_sha256": partial["source_scope_sha256"],
        "record_count": 1_278,
        "checkpoint_sha256": __import__("hashlib").sha256(partial_bytes).hexdigest(),
        "lineage_state": "successor",
        "historical_checkpoint_lineage_sha256": __import__("hashlib").sha256(
            json.dumps(
                partial["historical_checkpoint_lineage"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "historical_checkpoint_archive_sha256": partial[
            "historical_checkpoint_lineage"
        ]["archive_sha256"],
    }
    assert not {"records", "terminal_records", "output_path", "source_url"}.intersection(diagnostic)
    assert downstream == []


def test_one_real_live_invocation_supervises_durable_235_to_529_to_full_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixed CLI owns safe continuation; it does not require a new handoff.

    First create the descriptor-backed partial using the existing one-batch
    primitive, then invoke the real owner once.  Checkpoint publication is
    observed only as durable receipts: terminal identities are never sent to
    the bounded downloader a second time and material waits for exact full
    authority.
    """

    module = _owner_module()
    contract = _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    recovery = module._fixed_r5_recovery_inventory(contract)
    historical_rows = [
        {
            "item_id": row["r5_record_id"], "video_id": row["video_id"],
            "source_url": row["source_url"], "provenance": "direct_public_youtube",
            "terminal_state": "download_failed", "attempt_count": 3,
        }
        for row in recovery["asr_fallback_required"][:1_043]
    ]
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        archive_r6_audio_checkpoint_relative_path,
        read_r6_audio_checkpoint,
        write_r6_audio_checkpoint,
        write_r6_audio_checkpoint_archive,
    )

    write_r6_audio_checkpoint(
        contract=contract,
        payload=_audio_receipt(
            module,
            recovery_inventory=recovery, rows=historical_rows, status="running"
        ),
    )
    preflight_calls: list[str] = []

    def preflight_once() -> dict[str, str]:
        preflight_calls.append("preflight")
        return _fixture_preflight()

    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", preflight_once)
    seed_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        _terminal_downloader(module, seed_calls, stop_after=235),
    )
    # Build the real 235-row durable prefix through the contract fixture, not
    # by importing the sealed worker with caller-selected authority/batch data.
    # The supervisor owns a successor with immutable historical lineage; use
    # the existing real historical first invocation to make it before the
    # production CLI continuation.
    # A first real CLI run creates the durable 235-row successor; test setup
    # never calls an importable worker/factory with caller-selected controls.
    assert module.main(["--live"]) == 2
    partial_raw = read_r6_audio_checkpoint(contract=contract)
    assert partial_raw is not None
    assert json.loads(partial_raw)["counts"]["declared"] == 235
    # The assertion below is specifically for the single supervised CLI invocation.
    preflight_calls.clear()

    calls: list[tuple[str, str]] = []
    checkpoints: list[int] = []
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        write_r6_audio_checkpoint as original_write,
    )

    def capture_checkpoint(**kwargs: object) -> object:
        payload = kwargs["payload"]
        assert isinstance(payload, dict)
        counts = payload.get("counts")
        if isinstance(counts, dict):
            checkpoints.append(int(counts["declared"]))
        return original_write(**kwargs)

    monkeypatch.setattr("conference_pipeline.acquisition.cvpr_r6_artifact_contract.write_r6_audio_checkpoint", capture_checkpoint)
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        _terminal_downloader(module, calls, stop_after=9_999),
    )
    downstream: list[str] = []
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: downstream.append("observe"))
    monkeypatch.setattr(module, "run_fixed_r6_material_acquisition", lambda **_kwargs: downstream.append("material"))

    assert module.main(["--live"]) == 0
    final_raw = read_r6_audio_checkpoint(contract=contract)
    assert final_raw is not None
    final = json.loads(final_raw)
    assert final["status"] == "succeeded"
    assert final["counts"] == {"declared": 2_002, "acquired": 0, "terminal": 2_002}
    # Fixed 294-row passes make the first successor 529, proving that the
    # controlled invocation, rather than a new operator command, continued.
    assert 529 in checkpoints
    assert preflight_calls == ["preflight"]
    assert downstream == ["observe", "material"]
    assert [item_id for item_id, _url in calls] == [
        row["r5_record_id"] for row in recovery["asr_fallback_required"][1_278:]
    ]
    assert not {item_id for item_id, _url in seed_calls}.intersection(
        item_id for item_id, _url in calls
    )


def test_real_529_successor_no_progress_is_diagnostic_and_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact observed-sized successor cannot be a silent successful CLI run."""

    module = _owner_module()
    contract = _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    recovery = module._fixed_r5_recovery_inventory(contract)
    historical_rows = [
        {
            "item_id": row["r5_record_id"], "video_id": row["video_id"],
            "source_url": row["source_url"], "provenance": "direct_public_youtube",
            "terminal_state": "download_failed", "attempt_count": 3,
        }
        for row in recovery["asr_fallback_required"][:1_043]
    ]
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        read_r6_audio_checkpoint,
        write_r6_audio_checkpoint,
    )

    write_r6_audio_checkpoint(
        contract=contract,
        payload=_audio_receipt(
            module, recovery_inventory=recovery, rows=historical_rows, status="running"
        ),
    )
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp",
        _fixture_preflight,
    )
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        _terminal_downloader(module, [], stop_after=529),
    )
    assert module.main(["--live"]) == 2
    checkpoint = read_r6_audio_checkpoint(contract=contract)
    assert checkpoint is not None
    before = json.loads(checkpoint)
    assert before["status"] == "running"
    assert before["counts"] == {"declared": 529, "acquired": 0, "terminal": 529}

    blocked: list[str] = []
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        _terminal_downloader(module, blocked, stop_after=0),
    )
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: pytest.fail("material observation reached"))
    monkeypatch.setattr(module, "run_fixed_r6_material_acquisition", lambda **_kwargs: pytest.fail("material acquisition reached"))
    assert module.main(["--live"]) == 2
    after = read_r6_audio_checkpoint(contract=contract)
    assert after == checkpoint
    assert blocked == []
    diagnostic = json.loads(contract.path("continuation_diagnostic").read_text(encoding="utf-8"))
    assert diagnostic["reason_code"] == "worker_no_progress"
    assert diagnostic["record_count"] == 1_572
    assert diagnostic["recovery_inventory_sha256"] == recovery["r6_recovery_inventory_sha256"]
    assert diagnostic["checkpoint_sha256"] == __import__("hashlib").sha256(checkpoint).hexdigest()
    assert diagnostic["lineage_state"] == "successor"
    assert diagnostic["historical_checkpoint_archive_sha256"] == before[
        "historical_checkpoint_lineage"
    ]["archive_sha256"]
    decision = json.loads(contract.path("cli_decision").read_text(encoding="utf-8"))
    assert decision == {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "phase": "supervisor",
        "reason_code": "worker_no_progress",
        "exit_code": 2,
        "binding_state": "reachable",
        "recovery_inventory_sha256": recovery["r6_recovery_inventory_sha256"],
        "audio_policy_digest": module.r6_audio_policy_digest(),
        "source_scope_sha256": before["source_scope_sha256"],
        "record_count": 1_572,
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint).hexdigest(),
        "lineage_state": "successor",
        "historical_checkpoint_lineage_sha256": __import__("hashlib").sha256(
            json.dumps(
                before["historical_checkpoint_lineage"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "historical_checkpoint_archive_sha256": before["historical_checkpoint_lineage"]["archive_sha256"],
    }


def test_same_count_checkpoint_replacement_at_final_diagnostic_revalidation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A byte-distinct 529 successor cannot inherit a stale diagnostic claim."""

    module = _owner_module()
    contract = _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    recovery = module._fixed_r5_recovery_inventory(contract)
    historical_rows = [
        {
            "item_id": row["r5_record_id"], "video_id": row["video_id"],
            "source_url": row["source_url"], "provenance": "direct_public_youtube",
            "terminal_state": "download_failed", "attempt_count": 3,
        }
        for row in recovery["asr_fallback_required"][:1_043]
    ]
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        archive_r6_audio_checkpoint_relative_path,
        read_r6_audio_checkpoint,
        write_r6_audio_checkpoint,
        write_r6_audio_checkpoint_archive,
    )

    write_r6_audio_checkpoint(
        contract=contract,
        payload=_audio_receipt(
            module, recovery_inventory=recovery, rows=historical_rows, status="running"
        ),
    )
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp",
        _fixture_preflight,
    )
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        _terminal_downloader(module, [], stop_after=529),
    )
    assert module.main(["--live"]) == 2
    canonical = read_r6_audio_checkpoint(contract=contract)
    assert canonical is not None
    assert json.loads(canonical)["counts"]["declared"] == 529
    # This first diagnostic proves that a later failed admission cannot leave
    # stale evidence behind merely because its replacement has the same order.
    contract.path("continuation_diagnostic").unlink()

    original_reader = artifact_contract_module.read_r6_audio_checkpoint
    original_writer = artifact_contract_module.write_r6_audio_checkpoint
    original_archive = json.loads(
        (
            contract.path("audio_recovery_archive_root")
            / (json.loads(canonical)["historical_checkpoint_lineage"]["archive_sha256"] + ".json")
        ).read_text(encoding="utf-8")
    )
    original_archive["continuation_fixture_metadata"] = "alternate-valid-history"
    alternate_archive_raw = (
        json.dumps(original_archive, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    alternate_archive_digest = __import__("hashlib").sha256(alternate_archive_raw).hexdigest()
    write_r6_audio_checkpoint_archive(
        contract=contract,
        original_sha256=alternate_archive_digest,
        raw=alternate_archive_raw,
    )
    reads = 0
    swapped = False

    def replace_only_before_the_final_revalidation(*, contract: R6ArtifactContract, **kwargs: object) -> bytes | None:
        nonlocal reads, swapped
        raw = original_reader(contract=contract, **kwargs)
        reads += 1
        # The descriptor-bound pre-authority diagnostic context performs the
        # first read.  Call five is the diagnostic snapshot and call six is
        # its immediately-before-publication re-read.
        # Replace only after the former has been read, preserving every
        # effective row/order and immutable policy/authority fact.
        if reads == 5 and raw is not None:
            replacement = json.loads(raw)
            lineage = replacement["historical_checkpoint_lineage"]
            lineage["archive_sha256"] = alternate_archive_digest
            lineage["archive_path"] = archive_r6_audio_checkpoint_relative_path(
                alternate_archive_digest
            ).as_posix()
            lineage["archive_byte_count"] = len(alternate_archive_raw)
            original_writer(contract=contract, payload=replacement)
            swapped = True
        return raw

    monkeypatch.setattr(
        artifact_contract_module,
        "read_r6_audio_checkpoint",
        replace_only_before_the_final_revalidation,
    )
    terminal_attempts: list[str] = []
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        _terminal_downloader(module, terminal_attempts, stop_after=0),
    )
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: pytest.fail("observation reached"))
    monkeypatch.setattr(module, "run_fixed_r6_material_acquisition", lambda **_kwargs: pytest.fail("materials reached"))

    assert module.main(["--live"]) == 1
    assert swapped
    replacement = original_reader(contract=contract)
    assert replacement is not None and replacement != canonical
    assert json.loads(replacement)["counts"] == json.loads(canonical)["counts"]
    assert json.loads(replacement)["historical_checkpoint_lineage"]["archive_sha256"] != (
        json.loads(canonical)["historical_checkpoint_lineage"]["archive_sha256"]
    )
    assert not contract.path("continuation_diagnostic").exists()
    # The bounded seam is invoked before it can dispatch any record; no
    # terminal checkpoint identity is re-dispatched and no later seam runs.
    assert terminal_attempts == []

def test_real_full_terminal_checkpoint_reaches_material_once_and_cli_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain a real immutable-r5/full-receipt owner success path."""

    module = _owner_module()
    contract = _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    recovery = module._fixed_r5_recovery_inventory(contract)
    terminal_rows = [
        {
            "item_id": row["r5_record_id"], "video_id": row["video_id"],
            "source_url": row["source_url"], "provenance": "direct_public_youtube",
            "terminal_state": "download_failed", "attempt_count": 3,
        }
        for row in recovery["asr_fallback_required"]
    ]
    receipt = _audio_receipt(
        module,
        recovery_inventory=recovery, rows=terminal_rows, status="succeeded"
    )
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import write_r6_audio_checkpoint

    write_r6_audio_checkpoint(contract=contract, payload=receipt)
    touched: list[str] = []
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: touched.append("observe"))
    monkeypatch.setattr(
        module, "run_fixed_r6_material_acquisition", lambda **_kwargs: touched.append("material")
    )
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", lambda: touched.append("preflight"))
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio", lambda **_kwargs: touched.append("worker"))

    assert module.main(["--live"]) == 0
    assert touched == ["observe", "material"]


@pytest.mark.parametrize("variant", ("duplicate", "reordered"))
def test_real_archive_successor_full_reassembly_is_exact_before_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """A real 1,043 archive plus successor must reassemble all 2,002 in order.

    Unlike the direct-full-receipt coverage above, this drives the actual
    historical archive writer, successor offset, final reassembly, owner, and
    CLI.  The only replaced operation is the public downloader result.
    """

    module = _owner_module()
    contract = _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    recovery = module._fixed_r5_recovery_inventory(contract)
    historical_rows = [
        {
            "item_id": row["r5_record_id"],
            "video_id": row["video_id"],
            "source_url": row["source_url"],
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
        for row in recovery["asr_fallback_required"][:1_043]
    ]
    original = _audio_receipt(
        module,
        recovery_inventory=recovery, rows=historical_rows, status="running"
    )
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        read_r6_audio_checkpoint,
        write_r6_audio_checkpoint,
    )

    write_r6_audio_checkpoint(contract=contract, payload=original)
    original_bytes = read_r6_audio_checkpoint(contract=contract)
    assert original_bytes is not None
    calls: list[tuple[str, str]] = []
    touched: list[str] = []
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", _fixture_preflight)
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        _terminal_downloader(module, calls, stop_after=9_999),
    )
    monkeypatch.setattr(
        module, "_observe_official_material_routes", lambda **_kwargs: touched.append("observe")
    )
    monkeypatch.setattr(
        module,
        "run_fixed_r6_material_acquisition",
        lambda **_kwargs: touched.append("material"),
    )
    monkeypatch.setattr(module, "_load_repository_dotenv", lambda *_args: touched.append("dotenv"))

    assert module.main(["--live"]) == 0
    assert touched == ["observe", "material"]
    assert [item_id for item_id, _url in calls] == [
        row["r5_record_id"] for row in recovery["asr_fallback_required"][1_043:]
    ]
    archive = contract.path("audio_recovery_archive_root") / (
        __import__("hashlib").sha256(original_bytes).hexdigest() + ".json"
    )
    assert archive.read_bytes() == original_bytes
    final_bytes = read_r6_audio_checkpoint(contract=contract)
    assert final_bytes is not None
    final = json.loads(final_bytes)
    assert final["status"] == "succeeded"
    assert final["counts"] == {"declared": 2_002, "acquired": 0, "terminal": 2_002}
    assert [row["item_id"] for row in final["terminal_records"]] == [
        row["r5_record_id"] for row in recovery["asr_fallback_required"]
    ]

    # A forged final control record cannot use the valid archive as cover for
    # either duplicate identity or producer-order drift.  This re-enters the
    # real current-checkpoint -> owner -> CLI admission path, before material.
    forged = deepcopy(final)
    terminal = forged["terminal_records"]
    order = forged["effective_record_order"]
    assert isinstance(terminal, list) and isinstance(order, list)
    if variant == "duplicate":
        terminal[1] = deepcopy(terminal[0])
        order[1] = deepcopy(order[0])
    else:
        terminal[1], terminal[1_044] = terminal[1_044], terminal[1]
        order[1], order[1_044] = order[1_044], order[1]
    write_r6_audio_checkpoint(contract=contract, payload=forged)
    touched.clear()
    assert module.main(["--live"]) == 1
    assert touched == []
    assert archive.read_bytes() == original_bytes


@pytest.mark.parametrize("variant", ("malformed", "stale", "reordered"))
def test_real_current_checkpoint_admission_is_sanitized_at_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], variant: str
) -> None:
    """Current-envelope failures use the real r5 reader and owner boundary."""

    module = _owner_module()
    contract = _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    recovery = module._fixed_r5_recovery_inventory(contract)
    rows = [
        {
            "item_id": row["r5_record_id"], "video_id": row["video_id"],
            "source_url": row["source_url"], "provenance": "direct_public_youtube",
            "terminal_state": "download_failed", "attempt_count": 3,
        }
        for row in recovery["asr_fallback_required"][:1_043]
    ]
    current: dict[str, object] = _audio_receipt(
        module,
        recovery_inventory=recovery, rows=rows, status="running"
    )
    if variant == "malformed":
        current = {"status": "running", "private": "/tmp/not-output"}
    elif variant == "stale":
        current["recovery_inventory_sha256"] = "0" * 64
    else:
        terminal = current["terminal_records"]
        assert isinstance(terminal, list)
        terminal[0], terminal[1] = terminal[1], terminal[0]
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import write_r6_audio_checkpoint

    write_r6_audio_checkpoint(contract=contract, payload=current)
    touched: list[str] = []
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp", lambda: touched.append("preflight"))
    monkeypatch.setattr("conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio", lambda **_kwargs: touched.append("worker"))
    monkeypatch.setattr(module, "_load_repository_dotenv", lambda *_args: touched.append("dotenv"))
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: touched.append("observe"))
    monkeypatch.setattr(module, "run_fixed_r6_material_acquisition", lambda **_kwargs: touched.append("material"))

    with pytest.raises(module.R6AdmissionError):
        module.run_fixed_r6_media_owner(live=True)
    assert module.main(["--live"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert touched == []


def test_owner_all_terminal_exact_authority_reaches_material_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a fully classified synthetic 2,002-row receipt is owner-success."""

    module = _owner_module()
    workspace = tmp_path / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    contract = R6ArtifactContract.for_test_fixture(workspace)
    queue = [
        {
            "r5_record_id": f"r5:caption:{index:04d}",
            "video_id": f"A{index:010d}",
            "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
            "state": "asr_fallback_required",
            "source_classification": "public_youtube_candidate",
            "source_event_urls": [
                f"https://cvpr.thecvf.com/virtual/2026/poster/{index % 3}"
            ],
            "provenance": ["r5_official_event_page"],
        }
        for index in range(2_002)
    ]
    inventory: dict[str, object] = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_media_recovery_inventory",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "counts": {"asr_fallback_required": 2_002},
        "asr_fallback_required": queue,
    }
    scope = derive_r6_source_route_scope(inventory, verify_inventory_digest=False)
    inventory.update(source_route_scope_payload(scope))
    inventory["r6_recovery_inventory_sha256"] = canonical_r6_recovery_inventory_digest(inventory)
    terminal_rows = [
        {
            "item_id": row["r5_record_id"],
            "video_id": row["video_id"],
            "source_url": row["source_url"],
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
        for row in queue
    ]
    receipt = _audio_receipt(
        module,
        recovery_inventory=inventory, rows=terminal_rows, status="succeeded"
    )
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import write_r6_audio_checkpoint

    write_r6_audio_checkpoint(contract=contract, payload=receipt)
    touched: list[str] = []
    monkeypatch.setattr(module.R6ArtifactContract, "production", lambda: contract)
    monkeypatch.setattr(module, "_fixed_r5_recovery_inventory", lambda _contract: inventory)
    monkeypatch.setattr(module, "atomic_write_r6_json", lambda **_kwargs: touched.append("inventory"))
    monkeypatch.setattr(module, "_observe_official_material_routes", lambda **_kwargs: touched.append("observe"))
    monkeypatch.setattr(
        module, "run_fixed_r6_material_acquisition", lambda **_kwargs: touched.append("materials")
    )

    assert module.run_fixed_r6_media_owner(live=True) == 0
    assert module.main(["--live"]) == 0
    assert touched == ["inventory", "observe", "materials", "inventory", "observe", "materials"]


def test_fixed_audio_batch_uses_eight_workers_and_persists_queue_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parallel I/O must remain bounded while the receipt stays canonical."""

    import threading
    import time

    module = _owner_module()
    contract = _temporary_pinned_r5_contract(tmp_path, monkeypatch, module)
    recovery = module._fixed_r5_recovery_inventory(contract)
    historical_rows = [
        {
            "item_id": row["r5_record_id"],
            "video_id": row["video_id"],
            "source_url": row["source_url"],
            "provenance": "direct_public_youtube",
            "terminal_state": "download_failed",
            "attempt_count": 3,
        }
        for row in recovery["asr_fallback_required"][:1_043]
    ]
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import write_r6_audio_checkpoint

    write_r6_audio_checkpoint(
        contract=contract,
        payload=_audio_receipt(
            module, recovery_inventory=recovery, rows=historical_rows, status="running"
        ),
    )
    active = 0
    peak = 0
    lock = threading.Lock()

    def producer(**kwargs: object) -> dict[str, object]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.002)
            source_url = str(kwargs["source_url"])
            return {
                "item_id": kwargs["item_id"],
                "video_id": source_url.rsplit("=", 1)[1],
                "source_url": source_url,
                "provenance": "direct_public_youtube",
                "terminal_state": "download_failed",
                "attempt_count": 1,
            }
        finally:
            with lock:
                active -= 1

    touched: list[str] = []
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.preflight_fixed_r6_yt_dlp",
        _fixture_preflight,
    )
    monkeypatch.setattr(
        "conference_pipeline.acquisition.youtube_audio_recovery.recover_public_youtube_audio",
        producer,
    )
    monkeypatch.setattr(module, "_load_repository_dotenv", lambda *_args: touched.append("dotenv"))
    monkeypatch.setattr(
        module, "_observe_official_material_routes", lambda **_kwargs: touched.append("observe")
    )
    monkeypatch.setattr(
        module, "run_fixed_r6_material_acquisition", lambda **_kwargs: touched.append("material")
    )

    assert module.run_fixed_r6_media_owner(live=True) == 0
    assert peak == 8
    receipt = json.loads(contract.path("audio_recovery_receipt").read_text())
    assert receipt["status"] == "succeeded"
    records = receipt["terminal_records"]
    assert [row["item_id"] for row in records] == [
        row["r5_record_id"] for row in recovery["asr_fallback_required"]
    ]
    assert touched == ["observe", "material"]


def test_live_owner_rejects_a_redirected_production_workspace_before_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A production workspace link cannot reach the immutable-r5 reader."""

    module = _owner_module()
    repository = tmp_path / "repo"
    workspace = repository / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    workspace.rmdir()
    workspace.symlink_to(external, target_is_directory=True)
    touched: list[str] = []

    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    monkeypatch.setattr(
        module,
        "_fixed_r5_recovery_inventory",
        lambda _contract: touched.append("r5-reader") or {},
    )

    with pytest.raises(R6ArtifactContractError, match="anchor|workspace|symlink"):
        module.run_fixed_r6_media_owner(live=False)

    assert touched == []


def test_r6_source_scope_rejects_dropped_route_observation_before_live_owner(
    tmp_path: Path,
) -> None:
    candidates = []
    for index in range(2_002):
        video_id = f"A{index:010d}"
        candidates.append(
            {
                "record_id": f"r5:caption:{index:04d}",
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "source_event_urls": [
                    f"https://cvpr.thecvf.com/virtual/2026/poster/{1 + index % 2}"
                ],
                "provenance": ["r5_official_event_page"],
                "caption_status": "youtube_caption_unavailable",
                "caption": None,
                "audio_downloaded": False,
                "video_downloaded": False,
            }
        )
    inventory = {
        "run_id": "cvpr2026-live-production-r5-event-media",
        "caption_candidates": candidates,
        "counts": {"caption_candidate_count": 2_002},
    }
    from conference_pipeline.acquisition.cvpr_event_media import (
        canonical_event_media_inventory_digest,
    )

    inventory["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(inventory)
    receipt = {
        "run_id": "cvpr2026-live-production-r5-event-media",
        "stage_id": "download-corpus",
        "mode": "live",
        "status": "succeeded",
        "stage_contract": {"status": "pass", "stage_id": "download-corpus"},
        "event_media_inventory": {
            "path": "storage/download-corpus/event-media/event_media_inventory.json",
            "inventory_sha256": inventory["event_media_inventory_sha256"],
        },
        "checks": {
            "event_media_inventory_path": "storage/download-corpus/event-media/event_media_inventory.json",
            "event_media_inventory_sha256": "2" * 64,
            "event_media_inventory_canonical_sha256": inventory["event_media_inventory_sha256"],
        },
    }
    import hashlib
    import json

    inventory_sha = "2" * 64
    pins = R5LineagePins(
        receipt_sha256=hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        inventory_sha256=inventory_sha,
        inventory_canonical_sha256=str(inventory["event_media_inventory_sha256"]),
    )
    recovery = build_r6_recovery_inventory(
        r5_receipt=receipt,
        r5_inventory=inventory,
        pins=pins,
        workspace=tmp_path,
        output_path=tmp_path / "storage/transcription-ocr/cvpr2026/r6/recovery_inventory.json",
        r5_receipt_file_sha256=pins.receipt_sha256,
        r5_inventory_file_sha256=inventory_sha,
        contract=R6ArtifactContract.for_test_fixture(tmp_path),
    )
    scope = derive_r6_source_route_scope(recovery)
    assert scope.source_scope_count == 2
    assert source_route_scope_payload(scope)["source_scope_count"] == 2
    assert recovery["run_id"] == R6_EVENT_MEDIA_RUN_ID

    workspace = tmp_path / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    for name in ("config.yaml", "r6_material_source_policy.yaml"):
        (workspace / name).write_bytes((REPO_ROOT / "conferences" / "cvpr2026" / name).read_bytes())
    first_route = scope.routes[0]
    with pytest.raises(MaterialPolicyError, match="every source route"):
        build_material_discovery_receipt(
            workspace=workspace,
            observations=[
                {
                    "record_id": first_route["route_id"],
                    "event_id": first_route["event_id"],
                    "source_page_url": first_route["source_page_url"],
                    "observed_at": "2026-07-17T00:00:00Z",
                    "direct_target_url": None,
                    "link_kind": "absent",
                    "content_class": "paper_pdf",
                    "mime_type": "application/pdf",
                    "byte_count": 1,
                    "redirect_hops": [],
                }
            ],
            output_path=workspace / "storage/transcription-ocr/cvpr2026/r6/material_discovery_receipt.json",
            source_scope=source_route_scope_payload(scope),
            recovery_inventory=recovery,
        )
