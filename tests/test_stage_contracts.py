from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

import conference_pipeline.stage_contracts as stage_contracts
from conference_pipeline.stage_contracts import (
    resolve_stage_contract_binding,
    verify_pre_spend_contract,
    verify_tavily_priority_gap_pre_spend_contract,
    verify_stage_contract,
)
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)
from tests.lineage_fixtures import bind_fixture_lineage


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage6_workspace(tmp_path: Path, run_id: str = "run-1") -> tuple[Path, dict[str, object]]:
    workspace = tmp_path / "workspace"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-04-download-corpus.json",
        {"stage_id": "download-corpus", "status": "succeeded"},
    )
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-05-transcription-ocr.json",
        {"stage_id": "transcription-ocr", "status": "succeeded"},
    )
    _write_json(workspace / "storage" / "corpus" / "corpus_inventory.json", {"ok": True})
    _write_json(
        workspace / "storage" / "vector-store" / run_id / "index_digest.json",
        {"ok": True},
    )
    receipt = {
        "stage_id": "corpus-inventory-index",
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": [
            "indexing.token_bounded_chunking",
            "indexing.local_embedding_materialization_policy",
        ],
        "checks": {
            "vector_embeddings_written": True,
            "chunk_counts": {"papers": 2, "artifacts": 3},
            "embedded_chunk_counts": {"papers": 2, "artifacts": 3},
            "vector_materialization": {
                "stores": {
                    "papers": {
                        "embedding_sidecar_manifest_path": "storage/vector-store/run-1/stores/papers/embedding-sidecars/policy/manifest.json",
                        "embedding_sidecar_manifest_sha256": "paper-sidecar-sha",
                        "chroma_insert_batch_size": 4096,
                    },
                    "artifacts": {
                        "embedding_sidecar_manifest_path": "storage/vector-store/run-1/stores/artifacts/embedding-sidecars/policy/manifest.json",
                        "embedding_sidecar_manifest_sha256": "artifact-sidecar-sha",
                        "chroma_insert_batch_size": 4096,
                    },
                }
            },
        },
    }
    bind_fixture_lineage(receipt, workspace, run_id, "corpus-inventory-index")
    return workspace, receipt


def test_verify_stage_contract_passes_complete_stage6_evidence(tmp_path: Path) -> None:
    workspace, receipt = _stage6_workspace(tmp_path)

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-inventory-index",
        receipt=receipt,
    )

    assert result.status == "pass"
    assert result.blockers == []
    assert "indexing.token_bounded_chunking" in result.required_script_ids


def test_verify_stage_contract_blocks_missing_output(tmp_path: Path) -> None:
    """Stage-05 (transcription-ocr) has no required_receipts registry entry for
    corpus-inventory-index -- the CVPR runner's own inline admission logic
    (_validate_transcription_ocr_admission) covers that more rigorously via a
    cross-run search, mirroring the earlier Stage-4 precedent. This generic
    contract-level check now only covers the required_outputs gate."""

    workspace, receipt = _stage6_workspace(tmp_path)
    (workspace / "storage" / "vector-store" / "run-1" / "index_digest.json").unlink()

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-inventory-index",
        receipt=receipt,
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_required_output_missing" in codes


def test_verify_stage_contract_passes_rendered_podcast_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "stage-08-qa-gap-fill-synthesis.json",
        {
            "stage_id": "qa-gap-fill-synthesis",
            "status": "succeeded",
            "checks": {
                "executive_summary_status": "pass",
                "trend_report_status": "pass",
            },
        },
    )
    output_dir = workspace / "audio" / "overview" / "focused_podcast"
    for filename in (
        "script.json",
        "script.md",
        "audio_manifest.json",
        "audio_ready_package.json",
        "podcast.mp3",
    ):
        path = output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    receipt = {
        "stage_id": "podcast",
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": ["podcast.focused_script", "podcast.render_audio"],
        "checks": {
            "podcast_script_status": "pass",
            "podcast_audio_status": "pass",
            "podcast_audio_sha256": "audio-sha",
            "provider_calls_made": True,
        },
    }

    bind_fixture_lineage(receipt, workspace, run_id, "podcast")
    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="podcast",
        receipt=receipt,
    )

    assert result.status == "pass"
    assert result.blockers == []
    assert "podcast.render_audio" in result.required_script_ids


def test_verify_stage_contract_blocks_stale_digest_and_mock_mode(
    tmp_path: Path,
) -> None:
    workspace, receipt = _stage6_workspace(tmp_path)
    receipt["script_registry_sha256"] = "stale"
    receipt["mode"] = "mock"

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-inventory-index",
        receipt=receipt,
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_registry_digest_mismatch" in codes
    assert "stage_contract_non_live_evidence_rejected" in codes


def test_verify_stage_contract_blocks_missing_required_check(tmp_path: Path) -> None:
    workspace, receipt = _stage6_workspace(tmp_path)
    receipt["checks"]["vector_embeddings_written"] = False

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-inventory-index",
        receipt=receipt,
    )

    assert {
        blocker["code"] for blocker in result.blockers
    } >= {"stage_contract_required_check_mismatch"}


def test_verify_stage_contract_blocks_missing_stage6_embedding_sidecar_evidence(
    tmp_path: Path,
) -> None:
    workspace, receipt = _stage6_workspace(tmp_path)
    stores = receipt["checks"]["vector_materialization"]["stores"]
    stores["papers"]["embedding_sidecar_manifest_path"] = ""

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-inventory-index",
        receipt=receipt,
    )

    assert {
        blocker["code"] for blocker in result.blockers
    } >= {"stage_contract_required_check_not_truthy"}


def _lineage_registry(
    registry_path: Path,
    *,
    stage_id: str = "lineage-stage",
    required_receipts: list[dict[str, object]] | None = None,
    required_upstream_stages: list[str] | None = None,
    input_artifacts: list[dict[str, object]] | None = None,
    upstream_bindings: dict[str, dict[str, object]] | None = None,
) -> None:
    contract: dict[str, object] = {"required_receipts": required_receipts or []}
    if required_upstream_stages:
        contract["required_upstream_stages"] = required_upstream_stages
    if input_artifacts:
        contract["lineage"] = {"input_artifacts": input_artifacts}
    bindings: dict[str, object] = {
        stage_id: {
            "approved_script_ids": [],
            "optional_script_ids": [],
            "required_script_ids": [],
            "forbidden_script_ids": [],
            "runtime_contract": contract,
        }
    }
    for upstream_id, extra in (upstream_bindings or {}).items():
        bindings[upstream_id] = {
            "approved_script_ids": [],
            "optional_script_ids": [],
            "required_script_ids": [],
            "forbidden_script_ids": [],
            "runtime_contract": {"required_receipts": []},
            **extra,
        }
    _write_json(
        registry_path,
        {
            "schema_version": 2,
            "registry_name": "test",
            "planning_required_task_ids": [],
            "scripts": {},
            "stage_bindings": bindings,
            "task_bindings": {},
            "skill_contracts": {},
            "evidence_rules": {},
            "tasks": {},
        },
    )


def _stage_receipt_payload(*, stage_id: str, output_digest: str | None, lineage: dict | None = None) -> dict:
    payload: dict[str, object] = {
        "stage_id": stage_id,
        "stage_order": 6,
        "status": "succeeded",
        "mode": "live",
        "output_digest": output_digest,
        "started_at": "2026-01-01T00:00:00Z",
    }
    if lineage is not None:
        payload["lineage"] = lineage
    return payload


def _bound_receipt(
    workspace: Path,
    run_id: str,
    registry_path: Path,
    *,
    stage_id: str = "lineage-stage",
    run_dir_resolver: bool = False,
) -> dict[str, object]:
    from conference_pipeline.receipt_lineage import bind_lineage, receipt_path_resolver_from_registry

    resolver = None
    if run_dir_resolver:
        resolver = receipt_path_resolver_from_registry(
            workspace, workspace / "storage" / "workflow-runs" / run_id, registry_path=registry_path
        )
    lineage = bind_lineage(
        workspace, run_id, stage_id, registry_path=registry_path, receipt_path_for_stage=resolver
    )
    return {
        "stage_id": stage_id,
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": registry_sha256(registry_path),
        "required_script_ids": [],
        "checks": {},
        "output_digest": "deadbeef",
        "lineage": lineage,
    }


def test_verify_stage_contract_rejects_fresh_after_spec(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "upstream.json",
        {"stage_id": "upstream", "status": "succeeded"},
    )
    registry_path = tmp_path / "registry.json"
    _lineage_registry(
        registry_path,
        stage_id="fresh-stage",
        required_receipts=[
            {
                "path": "storage/workflow-runs/<run-id>/upstream.json",
                "fresh_after": "analysis/analysis_manifest.json",
            }
        ],
    )
    # The loader itself refuses the key ...
    with pytest.raises(Exception, match="fresh_after"):
        load_stage_script_registry(registry_path)
    # ... and the verifier refuses it too when handed such a spec directly.
    blockers: list[dict[str, object]] = []
    stage_contracts._check_required_receipt(
        workspace,
        run_id,
        "fresh-stage",
        {"path": "storage/workflow-runs/<run-id>/upstream.json", "fresh_after": "x"},
        blockers,
        [],
        artifact_id=None,
    )
    assert {blocker["code"] for blocker in blockers} == {"stage_contract_spec_key_unsupported"}


def test_verify_stage_contract_blocks_receipt_without_lineage(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry_path = tmp_path / "registry.json"
    _lineage_registry(registry_path)
    receipt = {
        "stage_id": "lineage-stage",
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": registry_sha256(registry_path),
        "required_script_ids": [],
        "checks": {},
    }
    strict = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="lineage-stage",
        receipt=receipt,
        registry_path=registry_path,
        require_lineage=True,
    )
    assert {blocker["code"] for blocker in strict.blockers} == {"stage_contract_lineage_missing"}
    lenient = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="lineage-stage",
        receipt=receipt,
        registry_path=registry_path,
        require_lineage=False,
    )
    assert lenient.status == "pass"


def test_verify_stage_contract_passes_when_lineage_matches_and_blocks_when_upstream_changes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    upstream_path = workspace / "storage" / "workflow-runs" / run_id / "stage-06-upstream.json"
    _write_json(upstream_path, _stage_receipt_payload(stage_id="upstream", output_digest="abc", lineage={}))
    registry_path = tmp_path / "registry.json"
    _lineage_registry(
        registry_path,
        required_receipts=[{"path": "storage/workflow-runs/<run-id>/stage-06-upstream.json"}],
    )
    receipt = _bound_receipt(workspace, run_id, registry_path)
    entry = receipt["lineage"]["upstream_receipts"][
        f"storage/workflow-runs/{run_id}/stage-06-upstream.json"
    ]
    assert entry["digest_kind"] == "output_digest" and entry["digest"] == "abc"

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="lineage-stage",
        receipt=receipt,
        registry_path=registry_path,
        require_lineage=True,
    )
    assert result.status == "pass", result.blockers
    assert any("lineage bound to" in item for item in result.accepted_evidence)

    # Upstream re-ran and produced different output: the recorded digest no longer matches.
    _write_json(upstream_path, _stage_receipt_payload(stage_id="upstream", output_digest="def", lineage={}))
    changed = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="lineage-stage",
        receipt=receipt,
        registry_path=registry_path,
        require_lineage=True,
    )
    codes = [blocker["code"] for blocker in changed.blockers]
    assert codes == ["stage_contract_upstream_changed"]
    assert changed.blockers[0]["upstream_stage_id"] == "upstream"
    assert changed.blockers[0]["recorded"] == "abc" and changed.blockers[0]["observed"] == "def"


def test_verify_stage_contract_blocks_unbound_required_receipt(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-06-upstream.json",
        _stage_receipt_payload(stage_id="upstream", output_digest="abc", lineage={}),
    )
    registry_path = tmp_path / "registry.json"
    _lineage_registry(registry_path)
    receipt = _bound_receipt(workspace, run_id, registry_path)  # bound with no required receipts
    # The registry now requires a receipt the lineage never recorded.
    _lineage_registry(
        registry_path,
        required_receipts=[{"path": "storage/workflow-runs/<run-id>/stage-06-upstream.json"}],
    )
    receipt["script_registry_sha256"] = registry_sha256(registry_path)
    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="lineage-stage",
        receipt=receipt,
        registry_path=registry_path,
        require_lineage=True,
    )
    assert {blocker["code"] for blocker in result.blockers} == {"stage_contract_lineage_upstream_unbound"}


def test_verify_stage_contract_blocks_upstream_stage_receipt_without_output_digest(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-06-upstream.json",
        _stage_receipt_payload(stage_id="upstream", output_digest=None),
    )
    registry_path = tmp_path / "registry.json"
    _lineage_registry(
        registry_path,
        required_receipts=[{"path": "storage/workflow-runs/<run-id>/stage-06-upstream.json"}],
    )
    receipt = _bound_receipt(workspace, run_id, registry_path)
    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="lineage-stage",
        receipt=receipt,
        registry_path=registry_path,
        require_lineage=True,
    )
    assert {blocker["code"] for blocker in result.blockers} == {
        "stage_contract_upstream_output_digest_missing"
    }


def test_verify_stage_contract_binds_non_stage_receipt_by_content_and_ignores_volatile_keys(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    sub_path = workspace / "storage" / "workflow-runs" / run_id / "07D-topic-reconciliation.json"
    _write_json(sub_path, {"status": "succeeded", "topic_count": 3, "generated_at": "2026-01-01T00:00:00Z"})
    registry_path = tmp_path / "registry.json"
    _lineage_registry(
        registry_path,
        required_receipts=[{"path": "storage/workflow-runs/<run-id>/07D-topic-reconciliation.json"}],
    )
    receipt = _bound_receipt(workspace, run_id, registry_path)
    key = f"storage/workflow-runs/{run_id}/07D-topic-reconciliation.json"
    assert receipt["lineage"]["upstream_receipts"][key]["digest_kind"] == "content_digest"

    _write_json(sub_path, {"status": "succeeded", "topic_count": 3, "generated_at": "2026-02-02T00:00:00Z"})
    same = verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="lineage-stage", receipt=receipt,
        registry_path=registry_path, require_lineage=True,
    )
    assert same.status == "pass", same.blockers

    _write_json(sub_path, {"status": "succeeded", "topic_count": 4, "generated_at": "2026-02-02T00:00:00Z"})
    changed = verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="lineage-stage", receipt=receipt,
        registry_path=registry_path, require_lineage=True,
    )
    assert [blocker["code"] for blocker in changed.blockers] == ["stage_contract_upstream_changed"]


def test_verify_stage_contract_blocks_changed_registry_input_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    manifest = workspace / "analysis" / "analysis_manifest.json"
    _write_json(manifest, {"topics": 1})
    registry_path = tmp_path / "registry.json"
    _lineage_registry(
        registry_path,
        input_artifacts=[{"name": "analysis_manifest", "path": "analysis/analysis_manifest.json"}],
    )
    receipt = _bound_receipt(workspace, run_id, registry_path)
    assert receipt["lineage"]["input_artifacts"]["analysis_manifest"]["path"] == "analysis/analysis_manifest.json"
    ok = verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="lineage-stage", receipt=receipt,
        registry_path=registry_path, require_lineage=True,
    )
    assert ok.status == "pass", ok.blockers

    _write_json(manifest, {"topics": 2})
    changed = verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="lineage-stage", receipt=receipt,
        registry_path=registry_path, require_lineage=True,
    )
    assert [blocker["code"] for blocker in changed.blockers] == ["stage_contract_input_artifact_changed"]

    manifest.unlink()
    missing = verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="lineage-stage", receipt=receipt,
        registry_path=registry_path, require_lineage=True,
    )
    assert [blocker["code"] for blocker in missing.blockers] == ["stage_contract_input_artifact_missing"]


def test_verify_stage_contract_required_upstream_stage_is_transitively_verified(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    registry_path = tmp_path / "registry.json"
    # root -> middle -> lineage-stage, each binding the previous by required_upstream_stages.
    _lineage_registry(
        registry_path,
        required_upstream_stages=["middle"],
        upstream_bindings={
            "root": {"receipt_filename": "stage-01-root.json"},
            "middle": {
                "receipt_filename": "stage-02-middle.json",
                "runtime_contract": {"required_upstream_stages": ["root"]},
            },
        },
    )
    _write_json(run_dir / "stage-01-root.json", _stage_receipt_payload(stage_id="root", output_digest="r1", lineage={"upstream_receipts": {}}))
    middle = _bound_receipt(workspace, run_id, registry_path, stage_id="middle", run_dir_resolver=True)
    middle.update({"stage_order": 2, "output_digest": "m1"})
    _write_json(run_dir / "stage-02-middle.json", middle)
    receipt = _bound_receipt(workspace, run_id, registry_path, run_dir_resolver=True)
    assert receipt["lineage"]["upstream_receipts"][f"storage/workflow-runs/{run_id}/stage-02-middle.json"]["stage_id"] == "middle"

    ok = verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="lineage-stage", receipt=receipt,
        registry_path=registry_path, require_lineage=True,
    )
    assert ok.status == "pass", ok.blockers

    # Root re-ran with new output; middle was NOT re-run, so middle's digest still
    # matches but middle's own lineage is stale -> lineage-stage is stale too.
    _write_json(run_dir / "stage-01-root.json", _stage_receipt_payload(stage_id="root", output_digest="r2", lineage={"upstream_receipts": {}}))
    stale = verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="lineage-stage", receipt=receipt,
        registry_path=registry_path, require_lineage=True,
    )
    codes = [blocker["code"] for blocker in stale.blockers]
    assert codes == ["stage_contract_upstream_stale"], stale.blockers
    assert stale.blockers[0]["chain"] == [f"storage/workflow-runs/{run_id}/stage-02-middle.json"]
    assert stale.blockers[0]["upstream_blockers"][0]["code"] == "stage_contract_upstream_changed"

    # Unbinding the required upstream (a lineage recorded before the binding existed) is a blocker.
    del receipt["lineage"]["upstream_receipts"][f"storage/workflow-runs/{run_id}/stage-02-middle.json"]
    unbound = verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="lineage-stage", receipt=receipt,
        registry_path=registry_path, require_lineage=True,
    )
    assert [blocker["code"] for blocker in unbound.blockers] == ["stage_contract_lineage_upstream_unbound"]


def test_verify_stage_contract_blocks_nested_required_receipt_check(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "llm.json",
        {
            "stage_id": "llm-full-read",
            "status": "succeeded",
            "scope": "transcripts",
            "checks": {
                "completion_state": "partial",
                "completed_unit_count": 25,
                "expected_unit_count": 100,
            },
        },
    )
    registry_path = tmp_path / "registry.json"
    _write_json(
        registry_path,
        {
            "schema_version": 2,
            "registry_name": "test",
            "planning_required_task_ids": [],
            "scripts": {},
            "stage_bindings": {
                "analysis": {
                    "approved_script_ids": [],
                    "optional_script_ids": [],
                    "required_script_ids": [],
                    "forbidden_script_ids": [],
                    "runtime_contract": {
                        "required_receipts": [
                            {
                                "path": "storage/workflow-runs/<run-id>/llm.json",
                                "checks": [
                                    {"path": "scope", "equals": "all"},
                                    {"path": "checks.completion_state", "equals": "complete"},
                                    {
                                        "path": "checks.completed_unit_count",
                                        "equals_path": "checks.expected_unit_count",
                                    },
                                ],
                            }
                        ]
                    },
                }
            },
            "task_bindings": {},
            "skill_contracts": {},
            "evidence_rules": {},
            "tasks": {},
        },
    )
    receipt = {
        "stage_id": "analysis",
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": registry_sha256(registry_path),
        "required_script_ids": [],
        "checks": {},
    }

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="analysis",
        receipt=receipt,
        registry_path=registry_path,
    )

    assert {
        blocker["code"] for blocker in result.blockers
    } >= {"stage_contract_required_check_mismatch"}


def test_verify_stage_contract_blocks_analysis_without_narrative_outputs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-06-corpus-inventory-index.json",
        {"stage_id": "corpus-inventory-index", "status": "succeeded"},
    )
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "14A-exhaustive-topic-extraction-receipt.json",
        {"stage_id": "exhaustive-topic-extraction", "status": "succeeded"},
    )
    _write_json(
        workspace / "storage" / "analysis" / "topic-strategy" / run_id / "topic_strategy_receipt.json",
        {"stage_id": "analysis-topic-strategy", "status": "succeeded"},
    )
    _write_json(workspace / "analysis" / "analysis_manifest.json", {"status": "succeeded"})
    _write_json(
        workspace / "storage" / "extractions" / run_id / "by-topic" / "topic.json",
        {"topic_id": "topic"},
    )
    _write_json(
        workspace / "storage" / "extractions" / run_id / "by-artifact" / "artifact.json",
        {"artifact_id": "artifact"},
    )
    registry = load_stage_script_registry()
    receipt = {
        "stage_id": "analysis",
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": list(required_script_ids_for_stage(registry, "analysis")),
        "checks": {
            "stage6_vector_embeddings_written": True,
            "analysis_manifest_status": "succeeded",
            "analysis_manifest_topic_count": 12,
            "narrative_status": "succeeded",
        },
    }

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="analysis",
        receipt=receipt,
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_required_output_missing" in codes
    assert "stage_contract_required_output_glob_missing" in codes


def test_verify_stage_contract_strict_execution_blocks_bare_owner_id_echo(
    tmp_path: Path,
) -> None:
    workspace, receipt = _stage6_workspace(tmp_path)
    # No owner_execution_evidence at all -- bare echo of required_script_ids only.

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-inventory-index",
        receipt=receipt,
        strict_execution=True,
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_owner_execution_evidence_missing" in codes


def test_verify_stage_contract_strict_execution_rejects_registry_id_copy_as_evidence(
    tmp_path: Path,
) -> None:
    workspace, receipt = _stage6_workspace(tmp_path)
    # Evidence tokens that merely echo the owner id back do not count.
    receipt["owner_execution_evidence"] = {
        "indexing.token_bounded_chunking": "indexing.token_bounded_chunking",
        "indexing.local_embedding_materialization_policy": (
            "indexing.local_embedding_materialization_policy"
        ),
    }

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-inventory-index",
        receipt=receipt,
        strict_execution=True,
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_owner_execution_evidence_missing" in codes


def test_verify_stage_contract_strict_execution_passes_with_distinct_execution_evidence(
    tmp_path: Path,
) -> None:
    workspace, receipt = _stage6_workspace(tmp_path)
    receipt["owner_execution_evidence"] = {
        "indexing.token_bounded_chunking": (
            "storage/vector-store/run-1/stores/papers/embedding-sidecars/policy/manifest.json"
        ),
        "indexing.local_embedding_materialization_policy": "provider_calls_made=true",
    }

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-inventory-index",
        receipt=receipt,
        strict_execution=True,
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_owner_execution_evidence_missing" not in codes


def test_verify_stage_contract_default_non_strict_ignores_execution_evidence(
    tmp_path: Path,
) -> None:
    workspace, receipt = _stage6_workspace(tmp_path)

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-inventory-index",
        receipt=receipt,
    )

    assert result.status == "pass"
    assert result.blockers == []


def test_source_discovery_owner_resolution_is_workspace_scoped() -> None:
    cvpr = resolve_stage_contract_binding(
        workspace="conferences/cvpr2026",
        stage_id="source-discovery",
    )
    iclr = resolve_stage_contract_binding(
        workspace="conferences/iclr2026",
        stage_id="source-discovery",
    )

    assert cvpr.registry_stage_id == "source-discovery.cvpr2026"
    assert cvpr.required_script_ids == ["acquisition.source_discovery.cvpr2026"]
    assert iclr.registry_stage_id == "source-discovery.iclr2026"
    assert iclr.required_script_ids == ["acquisition.source_discovery.iclr2026"]


@pytest.mark.parametrize(
    "workspace",
    ["conferences/unknown2026", "conferences/cvpr2026/../iclr2026", "../cvpr2026"],
)
def test_source_discovery_owner_resolution_fails_closed_for_unknown_or_traversal_workspace(
    workspace: str,
) -> None:
    with pytest.raises(ValueError, match="workspace stage binding"):
        resolve_stage_contract_binding(workspace=workspace, stage_id="source-discovery")


def test_source_discovery_contract_rejects_other_workspace_owner_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    registry = load_stage_script_registry()
    receipt = {
        "stage_id": "source-discovery",
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": ["acquisition.source_discovery.iclr2026"],
    }

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="source-discovery",
        receipt=receipt,
    )

    assert result.registry_stage_id == "source-discovery.cvpr2026"
    assert result.required_script_ids == ["acquisition.source_discovery.cvpr2026"]
    assert "stage_contract_required_owner_ids_missing" in {
        blocker["code"] for blocker in result.blockers
    }


def _cvpr_corpus_freeze_workspace(
    tmp_path: Path,
    run_id: str = "run-1",
) -> tuple[Path, dict[str, object]]:
    """Create only the contract evidence required for a CVPR corpus freeze."""

    workspace = tmp_path / "conferences" / "cvpr2026"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-02-source-discovery.json",
        {
            "stage_id": "source-discovery",
            "status": "succeeded",
            "mode": "live",
            "canonical_inventory": {"inventory_sha256": "a" * 64},
            "stage_contract": {"status": "pass"},
        },
    )
    _write_json(
        workspace / "storage" / "corpus-freeze" / "corpus_freeze.json",
        {"status": "succeeded"},
    )
    report_path = workspace / "storage" / "corpus-freeze" / "corpus_freeze_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# CVPR corpus freeze\n", encoding="utf-8")
    receipt = {
        "stage_id": "corpus-freeze",
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": ["acquisition.corpus_freeze.cvpr2026"],
        "owner_execution_evidence": {
            "acquisition.corpus_freeze.cvpr2026": {
                "owner_function": "run_corpus_freeze_stage",
                "manifest_path": "storage/corpus-freeze/corpus_freeze.json",
            }
        },
        "checks": {
            "source_lineage_integrity": True,
            "source_inventory_digest_pinned": True,
            "complete_canonical_accounting": True,
            "no_cap_hits": True,
            "included_items_planned_or_unreachable": True,
        },
    }
    bind_fixture_lineage(receipt, workspace, run_id, "corpus-freeze")
    return workspace, receipt


def test_corpus_freeze_owner_resolution_is_workspace_scoped() -> None:
    cvpr = resolve_stage_contract_binding(
        workspace="conferences/cvpr2026",
        stage_id="corpus-freeze",
    )
    iclr = resolve_stage_contract_binding(
        workspace="conferences/iclr2026",
        stage_id="corpus-freeze",
    )

    assert cvpr.registry_stage_id == "corpus-freeze.cvpr2026"
    assert cvpr.required_script_ids == ["acquisition.corpus_freeze.cvpr2026"]
    assert iclr.registry_stage_id == "corpus-freeze.iclr2026"
    assert iclr.required_script_ids == ["acquisition.corpus_freeze"]


def test_cvpr_corpus_freeze_strict_contract_requires_lineage_and_selection_evidence(
    tmp_path: Path,
) -> None:
    workspace, receipt = _cvpr_corpus_freeze_workspace(tmp_path)

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-freeze",
        receipt=receipt,
        strict_execution=True,
    )

    assert result.status == "pass"
    assert result.registry_stage_id == "corpus-freeze.cvpr2026"
    assert result.required_script_ids == ["acquisition.corpus_freeze.cvpr2026"]


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda workspace, receipt: receipt.__setitem__(
                "required_script_ids", ["acquisition.corpus_freeze"]
            ),
            "stage_contract_required_owner_ids_missing",
        ),
        (
            lambda workspace, receipt: receipt.__setitem__("mode", "mock"),
            "stage_contract_non_live_evidence_rejected",
        ),
        (
            lambda workspace, receipt: (
                workspace
                / "storage"
                / "workflow-runs"
                / "run-1"
                / "stage-02-source-discovery.json"
            ).unlink(),
            "stage_contract_required_receipt_missing",
        ),
        (
            lambda workspace, receipt: (
                workspace / "storage" / "corpus-freeze" / "corpus_freeze.json"
            ).unlink(),
            "stage_contract_required_output_missing",
        ),
        (
            lambda workspace, receipt: receipt.__setitem__(
                "script_registry_sha256", "stale"
            ),
            "stage_contract_registry_digest_mismatch",
        ),
        (
            lambda workspace, receipt: receipt.pop("owner_execution_evidence"),
            "stage_contract_owner_execution_evidence_missing",
        ),
        (
            lambda workspace, receipt: receipt["checks"].__setitem__(
                "complete_canonical_accounting", False
            ),
            "stage_contract_required_check_mismatch",
        ),
    ],
    ids=(
        "iclr-owner",
        "non-live",
        "missing-source-receipt",
        "missing-manifest-output",
        "stale-registry",
        "missing-owner-execution",
        "failed-accounting-check",
    ),
)
def test_cvpr_corpus_freeze_strict_contract_rejects_incomplete_evidence(
    tmp_path: Path,
    mutate: object,
    expected_code: str,
) -> None:
    workspace, receipt = _cvpr_corpus_freeze_workspace(tmp_path)
    mutate(workspace, receipt)  # type: ignore[operator]

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="corpus-freeze",
        receipt=receipt,
        strict_execution=True,
    )

    assert result.status == "blocked"
    assert expected_code in {blocker["code"] for blocker in result.blockers}


def _cvpr_download_corpus_workspace(
    tmp_path: Path,
    run_id: str = "run-1",
) -> tuple[Path, dict[str, object]]:
    """Create only the strict evidence required for CVPR Stage 4."""

    workspace = tmp_path / "conferences" / "cvpr2026"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-03-corpus-freeze.json",
        {
            "stage_id": "corpus-freeze",
            "status": "succeeded",
            "mode": "live",
            "stage_contract": {"status": "pass"},
            "checks": {
                "source_lineage_integrity": True,
                "source_inventory_digest_pinned": True,
                "complete_canonical_accounting": True,
                "no_cap_hits": True,
                "included_items_planned_or_unreachable": True,
            },
        },
    )
    _write_json(
        workspace / "storage" / "download-corpus" / "download_corpus.json",
        {"status": "succeeded"},
    )
    report_path = workspace / "storage" / "download-corpus" / "download_corpus_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# CVPR download corpus\n", encoding="utf-8")
    receipt = {
        "stage_id": "download-corpus",
        "status": "succeeded",
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": ["acquisition.download_corpus.cvpr2026"],
        "owner_execution_evidence": {
            "acquisition.download_corpus.cvpr2026": {
                "owner_function": "run_download_corpus_stage",
                "ledger_path": "storage/download-corpus/download_corpus.json",
            }
        },
        "checks": {
            "download_corpus_cvpr2026_execution_status": "pass",
            "phase_02_inventory_sha256_pinned": True,
            "phase_02_inventory_sha256": "c" * 64,
            "phase_03_selection_sha256_pinned": True,
            "phase_03_selection_sha256": "d" * 64,
            "program_item_count": 14583,
            "program_item_result_count": 14583,
            "complete_program_item_accounting": True,
            "no_cap_hits": True,
            "disk_capacity": {
                "positive_available_bytes": True,
                "scope_ledger_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
            },
            "storage_manifest_sha256": "e" * 64,
            "download_corpus_ledger_sha256": "f" * 64,
            "validated_local_artifacts": True,
            "artifact_payload_digests_verified": True,
            "complete_result_or_blocker_accounting": True,
            "policy_exclusions_visible": True,
            "focused_recovery_disposition": "not_applicable",
            "provider_and_recovery_evidence_sanitized": True,
            "secrets_copied": False,
        },
    }
    bind_fixture_lineage(receipt, workspace, run_id, "download-corpus")
    return workspace, receipt


def test_download_corpus_owner_resolution_is_workspace_scoped() -> None:
    cvpr = resolve_stage_contract_binding(
        workspace="conferences/cvpr2026",
        stage_id="download-corpus",
    )
    iclr = resolve_stage_contract_binding(
        workspace="conferences/iclr2026",
        stage_id="download-corpus",
    )

    assert cvpr.registry_stage_id == "download-corpus.cvpr2026"
    assert cvpr.required_script_ids == ["acquisition.download_corpus.cvpr2026"]
    assert iclr.registry_stage_id == "download-corpus.iclr2026"
    assert iclr.required_script_ids == ["acquisition.download_corpus"]


def test_cvpr_registration_media_profile_is_explicit_and_leaves_r3_default_unchanged() -> None:
    """r4 must not silently inherit the public r3 Stage-4 contract."""

    standard = resolve_stage_contract_binding(
        workspace="conferences/cvpr2026",
        stage_id="download-corpus",
    )
    registration_media = resolve_stage_contract_binding(
        workspace="conferences/cvpr2026",
        stage_id="download-corpus",
        profile="registration-media",
    )

    assert standard.registry_stage_id == registration_media.registry_stage_id == "download-corpus.cvpr2026"
    assert standard.required_script_ids == registration_media.required_script_ids == [
        "acquisition.download_corpus.cvpr2026"
    ]
    assert standard.runtime_contract["required_outputs"] == [
        {"path": "storage/download-corpus/download_corpus.json"},
        {"path": "storage/download-corpus/download_corpus_report.md"},
    ]
    assert registration_media.runtime_contract["required_outputs"] == [
        {"path": "storage/download-corpus/registration-media/registration_media.json"}
    ]
    assert registration_media.runtime_contract["required_run_id"] == CVPR_R4_RUN_ID
    assert registration_media.runtime_contract["semantic_validators"] == [
        "cvpr_registration_media_stage"
    ]


def test_cvpr_event_media_profile_requires_the_full_virtual_program_census() -> None:
    event_media = resolve_stage_contract_binding(
        workspace="conferences/cvpr2026",
        stage_id="download-corpus",
        profile="event-media-coverage",
    )

    assert event_media.runtime_contract["required_run_id"] == (
        "cvpr2026-live-production-r5-event-media"
    )
    assert event_media.runtime_contract["semantic_validators"] == [
        "cvpr_event_media_stage"
    ]
    assert event_media.runtime_contract["required_outputs"] == [
        {"path": "storage/download-corpus/event-media/event_media_inventory.json"}
    ]
    required = {
        item["path"]: item.get("equals")
        for item in event_media.runtime_contract["required_checks"]
        if "equals" in item
    }
    assert required["checks.event_route_count"] == 5258
    assert required["checks.event_category_counts.poster"] == 5023
    assert required["checks.r4_reused_poster_route_count"] == 2565


def test_cvpr_registration_media_profile_rejects_historical_r3_run_before_any_overlay_fallback(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    receipt = {
        "stage_id": "download-corpus",
        "run_id": "cvpr2026-live-production-r3-20260716",
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": ["acquisition.download_corpus.cvpr2026"],
        "owner_execution_evidence": {
            "acquisition.download_corpus.cvpr2026": {"owner_function": "fixture"}
        },
        "checks": {},
    }

    result = verify_stage_contract(
        workspace=workspace,
        run_id="cvpr2026-live-production-r3-20260716",
        stage_id="download-corpus",
        receipt=receipt,
        strict_execution=True,
        profile="registration-media",
    )

    assert result.status == "blocked"
    assert "stage_contract_registration_media_lineage_mismatch" in {
        blocker["code"] for blocker in result.blockers
    }


def test_cvpr_download_corpus_strict_contract_requires_complete_evidence(
    tmp_path: Path,
) -> None:
    workspace, receipt = _cvpr_download_corpus_workspace(tmp_path)

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="download-corpus",
        receipt=receipt,
        strict_execution=True,
    )

    assert result.status == "pass"
    assert result.registry_stage_id == "download-corpus.cvpr2026"
    assert result.required_script_ids == ["acquisition.download_corpus.cvpr2026"]


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda workspace, receipt: receipt.__setitem__(
                "required_script_ids", ["acquisition.download_corpus"]
            ),
            "stage_contract_required_owner_ids_missing",
        ),
        (
            lambda workspace, receipt: receipt.__setitem__("mode", "dry-run"),
            "stage_contract_non_live_evidence_rejected",
        ),
        (
            lambda workspace, receipt: (
                workspace
                / "storage"
                / "workflow-runs"
                / "run-1"
                / "stage-03-corpus-freeze.json"
            ).unlink(),
            "stage_contract_required_receipt_missing",
        ),
        (
            lambda workspace, receipt: (
                workspace / "storage" / "download-corpus" / "download_corpus.json"
            ).unlink(),
            "stage_contract_required_output_missing",
        ),
        (
            lambda workspace, receipt: (
                workspace / "storage" / "download-corpus" / "download_corpus_report.md"
            ).unlink(),
            "stage_contract_required_output_missing",
        ),
        (
            lambda workspace, receipt: receipt.__setitem__(
                "script_registry_sha256", "stale"
            ),
            "stage_contract_registry_digest_mismatch",
        ),
        (
            lambda workspace, receipt: receipt.pop("owner_execution_evidence"),
            "stage_contract_owner_execution_evidence_missing",
        ),
        (
            lambda workspace, receipt: receipt["checks"].__setitem__(
                "complete_program_item_accounting", False
            ),
            "stage_contract_required_check_mismatch",
        ),
        (
            lambda workspace, receipt: receipt["checks"]["disk_capacity"].__setitem__(
                "positive_available_bytes", False
            ),
            "stage_contract_required_check_mismatch",
        ),
        (
            lambda workspace, receipt: receipt["checks"]["disk_capacity"].pop(
                "scope_ledger_sha256"
            ),
            "stage_contract_required_check_not_truthy",
        ),
    ],
    ids=(
        "iclr-owner",
        "non-live",
        "missing-phase-3-receipt",
        "missing-ledger",
        "missing-report",
        "stale-registry",
        "missing-owner-execution",
        "failed-program-item-accounting",
        "nonpositive-disk-capacity",
        "unbound-disk-capacity-scope",
    ),
)
def test_cvpr_download_corpus_strict_contract_rejects_incomplete_evidence(
    tmp_path: Path,
    mutate: object,
    expected_code: str,
) -> None:
    workspace, receipt = _cvpr_download_corpus_workspace(tmp_path)
    mutate(workspace, receipt)  # type: ignore[operator]

    result = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="download-corpus",
        receipt=receipt,
        strict_execution=True,
    )

    assert result.status == "blocked"
    assert expected_code in {blocker["code"] for blocker in result.blockers}


CVPR_R4_RUN_ID = "cvpr2026-live-production-r4-registration-media"


@pytest.mark.parametrize("failure_status", ["registration_auth_required", "registered_access_denied"])
def test_registration_media_stage_contract_rejects_post_sentinel_auth_failures(
    failure_status: str,
) -> None:
    """A wall or denial after the sentinel cannot be strict r4 success."""

    rows = [
        {
            "transport_eligible": True,
            "auth_status": failure_status,
            "result_status": failure_status,
        },
        {
            "transport_eligible": False,
            "auth_status": "not_required",
            "result_status": "outside_authenticated_official_event_boundary",
        },
    ]

    assert not stage_contracts._registration_media_requests_complete_with_authenticated_access(rows)


def test_registration_media_stage_contract_requires_truthful_transport_execution_evidence() -> None:
    checks = {
        "registration_media_authenticated_request_count": 1,
        "registration_media_transport_created": True,
    }
    receipt = {
        "owner_execution_evidence": {
            "acquisition.download_corpus.cvpr2026": {
                "transport_created": True,
                "authenticated_request_count": 1,
            }
        }
    }

    assert stage_contracts._registration_media_execution_evidence_matches(
        receipt=receipt,
        checks=checks,
    )
    receipt["owner_execution_evidence"]["acquisition.download_corpus.cvpr2026"][
        "authenticated_request_count"
    ] = 0
    assert not stage_contracts._registration_media_execution_evidence_matches(
        receipt=receipt,
        checks=checks,
    )

    checks.update(
        {
            "registration_media_authenticated_request_count": 0,
            "registration_media_transport_created": False,
        }
    )
    receipt["owner_execution_evidence"]["acquisition.download_corpus.cvpr2026"].update(
        {
            "transport_created": False,
            "authenticated_request_count": 0,
        }
    )
    assert not stage_contracts._registration_media_execution_evidence_matches(
        receipt=receipt,
        checks=checks,
    )


CVPR_R5_RUN_ID = "cvpr2026-live-production-r5-event-media"
CVPR_R5_RECEIPT_SHA256 = "d6b0e6ac3dcea684ae699021d90c9ef737e239325b2fa496177efc25349038bc"
CVPR_R5_INVENTORY_SHA256 = "e6c7e00c3919e5652433a94ade336b121c83297dbd8780500576532d7e945a09"
CVPR_R5_CANONICAL_SHA256 = "09e2d6ec1d8150b5e78f7d263018aeb227d73ee3a4109d13d9be162bf37bc55e"
CVPR_R4_OVERLAY_SHA256 = "7f622be554f054425f6dff6ae0911b3a0b83acb6699fe5a15ed37e70cd0d0d12"


def _write_cvpr_workspace_config(workspace: Path) -> None:
    (workspace / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (workspace / "config.yaml").write_text(
        "conference:\n"
        "  name: CVPR 2026 fixture\n"
        "urls:\n"
        "  virtual: https://cvpr.thecvf.com/virtual/2026\n",
        encoding="utf-8",
    )


def _canonical_json_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_r4_strict_semantics_passes_workspace_boundary_to_overlay_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    _write_cvpr_workspace_config(workspace)
    _write_json(workspace / stage_contracts.OVERLAY_LEDGER_RELATIVE_PATH, {})
    captured: dict[str, object] = {}

    def _require_boundary(overlay: object, *, boundary: object) -> None:
        captured["boundary"] = boundary

    monkeypatch.setattr(stage_contracts, "validate_registration_media_overlay", _require_boundary)
    monkeypatch.setattr(
        stage_contracts,
        "load_pinned_r3_predecessors",
        lambda _workspace: stage_contracts.PINNED_R3_PREDECESSORS,
    )

    stage_contracts._check_cvpr_registration_media_stage_semantics(
        workspace=workspace,
        run_id=CVPR_R4_RUN_ID,
        stage_id="download-corpus",
        receipt={"run_id": CVPR_R4_RUN_ID, "stage_id": "download-corpus"},
        contract={"required_run_id": CVPR_R4_RUN_ID},
        blockers=[],
        accepted=[],
    )

    boundary = captured["boundary"]
    assert getattr(boundary, "origin") == "https://cvpr.thecvf.com"
    assert getattr(boundary, "root_segments") == ("virtual", "2026")


def test_r5_stage4_strict_semantics_passes_workspace_boundary_to_all_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    _write_cvpr_workspace_config(workspace)
    _write_json(workspace / stage_contracts.EVENT_MEDIA_INVENTORY_RELATIVE_PATH, {})
    _write_json(
        workspace / stage_contracts.OVERLAY_LEDGER_RELATIVE_PATH,
        {"official_event_requests": [], "official_event_media_records": []},
    )
    captured: dict[str, object] = {}

    def _require_inventory_boundary(inventory: object, *, boundary: object, **kwargs: object) -> None:
        captured["inventory"] = boundary

    def _require_overlay_boundary(overlay: object, *, boundary: object) -> None:
        captured["overlay"] = boundary

    def _require_reuse_boundary(
        inventory: object,
        *,
        boundary: object,
        r4_requests: object,
        r4_media_records: object,
    ) -> None:
        captured["reuse"] = boundary

    monkeypatch.setattr(stage_contracts, "validate_event_media_inventory", _require_inventory_boundary)
    monkeypatch.setattr(stage_contracts, "validate_registration_media_overlay", _require_overlay_boundary)
    monkeypatch.setattr(stage_contracts, "validate_r4_poster_reuse", _require_reuse_boundary)

    stage_contracts._check_cvpr_event_media_stage_semantics(
        workspace=workspace,
        run_id=CVPR_R5_RUN_ID,
        stage_id="download-corpus",
        receipt={"run_id": CVPR_R5_RUN_ID, "stage_id": "download-corpus"},
        contract={"required_run_id": CVPR_R5_RUN_ID},
        blockers=[],
        accepted=[],
    )

    assert set(captured) == {"inventory", "overlay", "reuse"}
    for boundary in captured.values():
        assert getattr(boundary, "origin") == "https://cvpr.thecvf.com"
        assert getattr(boundary, "root_segments") == ("virtual", "2026")


@pytest.mark.parametrize(
    ("config_text", "expected_code"),
    [
        (None, "stage_contract_registration_media_predecessor_invalid"),
        (
            "conference:\n  name: CVPR 2026 fixture\nurls:\n  virtual: http://cvpr.thecvf.com/virtual/2026\n",
            "stage_contract_registration_media_predecessor_invalid",
        ),
    ],
)
def test_r4_strict_semantics_translates_workspace_boundary_failures(
    tmp_path: Path, config_text: str | None, expected_code: str
) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    _write_cvpr_workspace_config(workspace)
    _write_json(workspace / stage_contracts.OVERLAY_LEDGER_RELATIVE_PATH, {})
    if config_text is None:
        (workspace / "config.yaml").unlink()
    else:
        (workspace / "config.yaml").write_text(config_text, encoding="utf-8")
    blockers: list[dict[str, object]] = []

    stage_contracts._check_cvpr_registration_media_stage_semantics(
        workspace=workspace,
        run_id=CVPR_R4_RUN_ID,
        stage_id="download-corpus",
        receipt={"run_id": CVPR_R4_RUN_ID, "stage_id": "download-corpus"},
        contract={"required_run_id": CVPR_R4_RUN_ID},
        blockers=blockers,
        accepted=[],
    )

    assert [blocker["code"] for blocker in blockers] == [expected_code]


@pytest.mark.parametrize(
    "config_text",
    [
        None,
        "conference:\n  name: CVPR 2026 fixture\nurls:\n  virtual: https://cvpr.thecvf.com/virtual/%2e%2e\n",
    ],
)
def test_r5_stage4_strict_semantics_translates_workspace_boundary_failures(
    tmp_path: Path, config_text: str | None
) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    _write_cvpr_workspace_config(workspace)
    _write_json(workspace / stage_contracts.EVENT_MEDIA_INVENTORY_RELATIVE_PATH, {})
    if config_text is None:
        (workspace / "config.yaml").unlink()
    else:
        (workspace / "config.yaml").write_text(config_text, encoding="utf-8")
    blockers: list[dict[str, object]] = []

    stage_contracts._check_cvpr_event_media_stage_semantics(
        workspace=workspace,
        run_id=CVPR_R5_RUN_ID,
        stage_id="download-corpus",
        receipt={"run_id": CVPR_R5_RUN_ID, "stage_id": "download-corpus"},
        contract={"required_run_id": CVPR_R5_RUN_ID},
        blockers=blockers,
        accepted=[],
    )

    assert [blocker["code"] for blocker in blockers] == ["stage_contract_event_media_inventory_invalid"]


def _r5_event_media_inventory() -> dict[str, object]:
    """Build synthetic, complete publisher evidence without production payloads."""

    categories = (
        ("poster", "poster", 5_023),
        ("invited-talk", "keynote", 3),
        ("tutorial", "tutorial", 19),
        ("workshop", "workshop", 153),
        ("session", "oral", 24),
        ("session", "session", 19),
        ("events", "main_event", 17),
    )
    events: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}
    global_index = 0
    poster_index = 0
    for route_family, category, count in categories:
        category_counts[category] = count
        for local_index in range(1, count + 1):
            global_index += 1
            title = "Oral Session" if category == "oral" else f"Synthetic {category} {local_index}"
            direct = global_index <= 2_350
            origin = (
                "r4_registration_media"
                if route_family == "poster" and poster_index < 2_565
                else "r5_authenticated_event_page"
            )
            if route_family == "poster":
                poster_index += 1
            events.append(
                {
                    "source_record_id": f"source-{global_index}",
                    "source_url": f"https://cvpr.thecvf.com/virtual/2026/{route_family}/{global_index}",
                    "route_family": route_family,
                    "event_category": category,
                    "title": title,
                    "page_status": "direct_youtube_recovered" if direct else "official_video_not_published",
                    "outcome_origin": origin,
                    "video_ids": [f"v{global_index:010d}"] if direct else [],
                    "audio_downloaded": False,
                    "video_downloaded": False,
                }
            )
    coverage = {
        category: {
            "event_route_count": count,
            "terminal_route_count": count,
            "direct_youtube_route_count": sum(
                row["event_category"] == category and row["page_status"] == "direct_youtube_recovered"
                for row in events
            ),
            "status": "complete",
        }
        for category, count in category_counts.items()
    }
    candidates: list[dict[str, object]] = []
    for index in range(1, 2_353):
        video_id = f"v{index:010d}"
        available = index <= 350
        candidates.append(
            {
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "source_event_urls": [],
                "provenance": ["r5_official_event_page"],
                "caption_status": "youtube_caption_available" if available else "youtube_caption_unavailable",
                "caption": (
                    {
                        "local_path": f"storage/download-corpus/event-media/captions/{video_id}.en.vtt",
                        "mime": "text/vtt",
                        "byte_count": 1,
                        "sha256": "a" * 64,
                        "transcript_source": "youtube_caption_manual",
                        "language": "en",
                    }
                    if available
                    else None
                ),
                "audio_downloaded": False,
                "video_downloaded": False,
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_event_media_inventory",
        "run_id": CVPR_R5_RUN_ID,
        "source_lineage": {
            "source_inventory_sha256": "a" * 64,
            "r4_registration_media_overlay_sha256": CVPR_R4_OVERLAY_SHA256,
        },
        "events": events,
        "caption_candidates": candidates,
        "coverage": coverage,
        "counts": {
            "event_route_count": 5_258,
            "event_categories": category_counts,
            "oral_session_route_count": 24,
            "terminal_event_route_count": 5_258,
            "direct_youtube_route_count": 2_350,
            "r4_reused_poster_route_count": 2_565,
            "caption_candidate_count": 2_352,
        },
        "boundaries": {
            "audio_downloaded": False,
            "video_downloaded": False,
            "asr_dispatched": False,
            "ocr_dispatched": False,
            "provider_calls_made": 0,
        },
    }
    from conference_pipeline.acquisition.cvpr_event_media import canonical_event_media_inventory_digest

    payload["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(payload)
    return payload


def _r5_stage5_inventory() -> dict[str, object]:
    transcript_records = [
        {
            "source_identity": f"v{index:010d}",
            "source_class": "caption_transcript",
            "outcome": "caption_used_no_asr",
            "video_id": f"v{index:010d}",
            "language": "en",
            "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "caption"}],
            "caption_sha256": "a" * 64,
            "output_sha256": "b" * 64,
        }
        for index in range(1, 351)
    ]
    blockers = [
        {
            "source_identity": f"v{index:010d}",
            "code": "youtube_caption_unavailable",
            "asr_eligible": False,
        }
        for index in range(351, 2_353)
    ]
    workshop_table = [
        {
            "source_identity": "workshop:valid-pdf",
            "predecessor_receipt_sha256": "c" * 64,
            "receipt_index_sha256": "d" * 64,
            "declared_path": "storage/workshops/papers/valid.pdf",
            "mime_type": "application/pdf",
            "byte_count": 11,
            "sha256": "e" * 64,
            "outcome_class": "workshop_paper_text_extraction",
            "output_digest": "f" * 64,
        },
        {
            "source_identity": "workshop:unproven",
            "predecessor_receipt_sha256": None,
            "receipt_index_sha256": "d" * 64,
            "declared_path": "storage/workshops/papers/unproven.pdf",
            "mime_type": "application/pdf",
            "byte_count": 0,
            "sha256": None,
            "outcome_class": "workshop_pdf_lineage_unproven",
            "output_digest": None,
        },
        {
            "source_identity": "workshop:zip-content",
            "predecessor_receipt_sha256": "c" * 64,
            "receipt_index_sha256": "d" * 64,
            "declared_path": "storage/workshops/papers/zip-content.pdf",
            "mime_type": "application/pdf",
            "byte_count": 12,
            "sha256": "0" * 64,
            "outcome_class": "workshop_pdf_zip_content",
            "output_digest": None,
        },
    ]
    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_inventory",
        "run_id": CVPR_R5_RUN_ID,
        "processor_identity": "cvpr2026.caption_materializer",
        "processor_version": "1",
        "semantic_class": "caption_first_no_asr",
        "compatibility": {
            "run_id": CVPR_R5_RUN_ID,
            "stage4_receipt_sha256": "1" * 64,
            "event_media_file_sha256": "<filled-by-fixture>",
            "event_media_canonical_sha256": "<filled-by-fixture>",
            "r4_overlay_sha256": CVPR_R4_OVERLAY_SHA256,
            "processor_identity": "cvpr2026.caption_materializer",
            "processor_version": "1",
            "semantic_class": "caption_first_no_asr",
        },
        "input_digests": {
            "stage4_receipt_sha256": "1" * 64,
            "event_media_file_sha256": "<filled-by-fixture>",
            "event_media_canonical_sha256": "<filled-by-fixture>",
            "r4_overlay_sha256": CVPR_R4_OVERLAY_SHA256,
        },
        "capacity_reservation": {
            "accepted_caption_input_bytes": 350,
            "accepted_caption_count": 350,
            "reservation_bytes": 700,
            "safety_margin_bytes": 35,
            "manifest_sha256": "9" * 64,
        },
        "caption_transcripts": transcript_records,
        "blockers": blockers,
        "paper_text_records": [],
        "workshop_pdf_provenance_table": workshop_table,
        "workshop_talk_boundary": {
            "status": "not_claimed",
            "reason": "no receipt-backed workshop talk media",
        },
        "asr_queue": [],
        "ocr_queue": [],
        "media_recovery_queue": [],
        "provider_calls": [],
        "counts": {
            "terminal_route_count": 5_258,
            "caption_candidate_count": 2_352,
            "caption_used_no_asr_count": 350,
            "youtube_caption_unavailable_count": 2_002,
            "asr_queue_count": 0,
            "ocr_queue_count": 0,
            "provider_call_count": 0,
        },
        "resume": {"reused_output_digest": None, "superseded_attempt_count": 0},
        "superseded_caption_attempts": [],
    }


def _cvpr_transcription_ocr_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object], Path]:
    """Create published-shape r5 evidence and a scoped temporary registry."""

    workspace = tmp_path / "conferences" / "cvpr2026"
    _write_cvpr_workspace_config(workspace)
    stage4_inventory_path = workspace / "storage/download-corpus/event-media/event_media_inventory.json"
    event_media_inventory = _r5_event_media_inventory()
    _write_json(stage4_inventory_path, event_media_inventory)
    event_media_file_sha256 = hashlib.sha256(stage4_inventory_path.read_bytes()).hexdigest()
    from conference_pipeline.acquisition.cvpr_event_media import canonical_event_media_inventory_digest

    event_media_canonical_sha256 = canonical_event_media_inventory_digest(event_media_inventory)
    stage4_path = workspace / f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-04-download-corpus.json"
    _write_json(
        stage4_path,
        {
            "stage_id": "download-corpus",
            "stage_order": 4,
            "status": "succeeded",
            "mode": "live",
            "run_id": CVPR_R5_RUN_ID,
            "stage_contract": {
                "status": "pass",
                "stage_id": "download-corpus",
            },
            "event_media_inventory": {
                "path": stage4_inventory_path.relative_to(workspace).as_posix(),
                "inventory_sha256": event_media_canonical_sha256,
            },
            "checks": {
                "event_media_inventory_path": stage4_inventory_path.relative_to(workspace).as_posix(),
                "event_media_inventory_sha256": event_media_file_sha256,
                "event_media_inventory_canonical_sha256": event_media_canonical_sha256,
                "r4_registration_media_overlay_sha256": CVPR_R4_OVERLAY_SHA256,
            },
        },
    )
    stage4_receipt_sha256 = hashlib.sha256(stage4_path.read_bytes()).hexdigest()
    stage5_inventory_path = workspace / "storage/transcription-ocr/cvpr2026/inventory.json"
    stage5_inventory = _r5_stage5_inventory()
    papers_dir = workspace / "storage/workshops/papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    valid_pdf = papers_dir / "valid.pdf"
    valid_pdf.write_bytes(b"%PDF-1.4\nA")
    zip_content = papers_dir / "zip-content.pdf"
    zip_content.write_bytes(b"PK\x03\x04not-a-pdf")
    workshop_table = stage5_inventory["workshop_pdf_provenance_table"]
    assert isinstance(workshop_table, list)
    valid_row, _unproven_row, zip_row = workshop_table
    assert isinstance(valid_row, dict) and isinstance(zip_row, dict)
    valid_row["byte_count"] = valid_pdf.stat().st_size
    valid_row["sha256"] = hashlib.sha256(valid_pdf.read_bytes()).hexdigest()
    zip_row["byte_count"] = zip_content.stat().st_size
    zip_row["sha256"] = hashlib.sha256(zip_content.read_bytes()).hexdigest()
    stage5_inventory["compatibility"]["event_media_file_sha256"] = event_media_file_sha256  # type: ignore[index]
    stage5_inventory["compatibility"]["event_media_canonical_sha256"] = event_media_canonical_sha256  # type: ignore[index]
    stage5_inventory["compatibility"]["stage4_receipt_sha256"] = stage4_receipt_sha256  # type: ignore[index]
    stage5_inventory["input_digests"]["event_media_file_sha256"] = event_media_file_sha256  # type: ignore[index]
    stage5_inventory["input_digests"]["event_media_canonical_sha256"] = event_media_canonical_sha256  # type: ignore[index]
    stage5_inventory["input_digests"]["stage4_receipt_sha256"] = stage4_receipt_sha256  # type: ignore[index]
    stage5_inventory["inventory_sha256"] = _canonical_json_digest(
        {key: value for key, value in stage5_inventory.items() if key != "inventory_sha256"}
    )
    _write_json(stage5_inventory_path, stage5_inventory)
    stage5_inventory_file_sha256 = hashlib.sha256(stage5_inventory_path.read_bytes()).hexdigest()
    workshop_table_sha256 = _canonical_json_digest(workshop_table)

    registry_payload = json.loads(
        (Path(__file__).resolve().parents[1] / "src/conference_pipeline/stage_owner_registry.json").read_text(
            encoding="utf-8"
        )
    )
    contract = registry_payload["stage_bindings"]["transcription-ocr"]["workspace_bindings"][
        "conferences/cvpr2026"
    ]["runtime_contract"]
    contract.update(
        {
            "required_run_id": CVPR_R5_RUN_ID,
            "semantic_validators": ["cvpr_transcription_ocr"],
            "required_outputs": [
                {"path": "storage/transcription-ocr/cvpr2026/inventory.json"}
            ],
            "required_receipts": [
                {
                    "path": f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-04-download-corpus.json",
                    # Pinned historical receipt: bound by content, never by output_digest.
                    "lineage": "content",
                    "checks": [
                        {"path": "stage_id", "equals": "download-corpus"},
                        {"path": "run_id", "equals": CVPR_R5_RUN_ID},
                        {"path": "stage_contract.status", "equals": "pass"},
                    ],
                }
            ],
            "required_checks": [
                {"path": "stage_id", "equals": "transcription-ocr"},
                {"path": "status", "equals": "succeeded"},
                {"path": "run_id", "equals": CVPR_R5_RUN_ID},
                {"path": "checks.transcription_ocr_cvpr2026_execution_status", "equals": "pass"},
                {"path": "checks.stage_04_run_id", "equals": CVPR_R5_RUN_ID},
                {"path": "checks.stage_04_receipt_sha256", "equals": stage4_receipt_sha256},
                {"path": "checks.event_media_inventory_sha256", "equals": event_media_file_sha256},
                {"path": "checks.event_media_inventory_canonical_sha256", "equals": event_media_canonical_sha256},
                {"path": "checks.r4_registration_media_overlay_sha256", "equals": CVPR_R4_OVERLAY_SHA256},
                {"path": "checks.transcription_ocr_inventory_sha256", "truthy": True},
                {"path": "checks.workshop_pdf_provenance_table_sha256", "truthy": True},
                {"path": "checks.secrets_copied", "equals": False},
            ],
        }
    )
    registry_path = tmp_path / "stage-owner-registry.json"
    _write_json(registry_path, registry_payload)
    monkeypatch.setattr(stage_contracts, "R5_EVENT_MEDIA_RECEIPT_SHA256", stage4_receipt_sha256, raising=False)
    monkeypatch.setattr(stage_contracts, "R5_EVENT_MEDIA_FILE_SHA256", event_media_file_sha256, raising=False)
    monkeypatch.setattr(stage_contracts, "R5_EVENT_MEDIA_CANONICAL_DIGEST", event_media_canonical_sha256, raising=False)
    monkeypatch.setattr(stage_contracts, "R4_OVERLAY_DIGEST", CVPR_R4_OVERLAY_SHA256, raising=False)
    receipt = {
        "stage_id": "transcription-ocr",
        "status": "succeeded",
        "mode": "live",
        "run_id": CVPR_R5_RUN_ID,
        "script_registry_sha256": registry_sha256(registry_path),
        "required_script_ids": ["transcription_ocr.cvpr2026"],
        "owner_execution_evidence": {
            "transcription_ocr.cvpr2026": {
                "owner_function": "run_cvpr_transcription_ocr_stage",
                "inventory_path": "storage/transcription-ocr/cvpr2026/inventory.json",
            }
        },
        "checks": {
            "transcription_ocr_cvpr2026_execution_status": "pass",
            "stage_04_run_id": CVPR_R5_RUN_ID,
            "stage_04_receipt_path": stage4_path.relative_to(workspace).as_posix(),
            "stage_04_receipt_sha256": stage4_receipt_sha256,
            "event_media_inventory_path": stage4_inventory_path.relative_to(workspace).as_posix(),
            "event_media_inventory_sha256": event_media_file_sha256,
            "event_media_inventory_canonical_sha256": event_media_canonical_sha256,
            "r4_registration_media_overlay_sha256": CVPR_R4_OVERLAY_SHA256,
            "stage_05_receipt_path": f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-05-transcription-ocr.json",
            "transcription_ocr_inventory_path": stage5_inventory_path.relative_to(workspace).as_posix(),
            "transcription_ocr_inventory_sha256": stage5_inventory_file_sha256,
            "transcription_ocr_inventory_canonical_sha256": stage5_inventory["inventory_sha256"],
            "workshop_pdf_provenance_table_sha256": workshop_table_sha256,
            "workshop_pdf_provenance_accepted_count": 1,
            "workshop_pdf_provenance_blocker_count": 2,
            "event_route_count": 5_258,
            "caption_candidate_count": 2_352,
            "caption_used_no_asr_count": 350,
            "youtube_caption_unavailable_count": 2_002,
            "asr_eligible_input_count": 0,
            "ocr_eligible_input_count": 0,
            "asr_queue_count": 0,
            "ocr_queue_count": 0,
            "provider_call_count": 0,
            "provider_calls_made": 0,
            "disk_capacity": {
                "positive_available_bytes": True,
                "manifest_sha256": "9" * 64,
                "output_reservation_bytes": 700,
            },
            "secrets_copied": False,
        },
    }
    retry_control_paths = {
        "archive": f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-05-transcription-ocr.pre-repair-blocked.json",
        "consumed_authorization": f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-05-transcription-ocr.retry-authorization.json",
        "attempt": f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-05-transcription-ocr.retry-attempt.json",
    }
    retry_control_digests: dict[str, dict[str, str]] = {}
    for key, relative_path in retry_control_paths.items():
        control_path = workspace / relative_path
        _write_json(control_path, {"control": key, "run_id": CVPR_R5_RUN_ID})
        retry_control_digests[key] = {
            "path": relative_path,
            "sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
        }
    receipt["checks"]["retry_control_digests"] = retry_control_digests  # type: ignore[index]
    return workspace, receipt, registry_path


def _rewrite_published_stage4_and_rebind(
    workspace: Path,
    receipt: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    mutate: object,
) -> None:
    """Change one producer field while keeping every downstream digest binding honest."""

    stage4_path = workspace / f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-04-download-corpus.json"
    stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
    mutate(stage4)  # type: ignore[operator]
    _write_json(stage4_path, stage4)
    stage4_sha256 = hashlib.sha256(stage4_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        stage_contracts, "R5_EVENT_MEDIA_RECEIPT_SHA256", stage4_sha256, raising=False
    )
    checks = receipt["checks"]
    assert isinstance(checks, dict)
    checks["stage_04_receipt_sha256"] = stage4_sha256
    inventory_path = workspace / "storage/transcription-ocr/cvpr2026/inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["compatibility"]["stage4_receipt_sha256"] = stage4_sha256
    inventory["input_digests"]["stage4_receipt_sha256"] = stage4_sha256
    inventory["inventory_sha256"] = _canonical_json_digest(
        {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    )
    _write_json(inventory_path, inventory)
    checks["transcription_ocr_inventory_sha256"] = hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest()
    checks["transcription_ocr_inventory_canonical_sha256"] = inventory["inventory_sha256"]


def _strict_cvpr_stage5_result(
    workspace: Path, receipt: dict[str, object], registry_path: Path
) -> object:
    bind_fixture_lineage(
        receipt, workspace, CVPR_R5_RUN_ID, "transcription-ocr", registry_path=registry_path, stamp=False
    )
    return verify_stage_contract(
        workspace=workspace,
        run_id=CVPR_R5_RUN_ID,
        stage_id="transcription-ocr",
        receipt=receipt,
        strict_execution=True,
        registry_path=registry_path,
    )


def test_cvpr_transcription_ocr_owner_resolution_is_workspace_scoped() -> None:
    cvpr = resolve_stage_contract_binding(
        workspace="conferences/cvpr2026",
        stage_id="transcription-ocr",
    )
    iclr = resolve_stage_contract_binding(
        workspace="conferences/iclr2026",
        stage_id="transcription-ocr",
    )

    assert cvpr.registry_stage_id == "transcription-ocr.cvpr2026"
    assert cvpr.required_script_ids == ["transcription_ocr.cvpr2026"]
    assert iclr.required_script_ids == ["transcription_ocr.local_media_processing"]


def test_cvpr_r6_transcription_ocr_contract_is_a_distinct_workspace_binding() -> None:
    r5 = resolve_stage_contract_binding(
        workspace="conferences/cvpr2026",
        stage_id="transcription-ocr",
    )
    r6 = resolve_stage_contract_binding(
        workspace="conferences/cvpr2026",
        stage_id="transcription-ocr-r6",
    )

    assert r5.registry_stage_id == "transcription-ocr.cvpr2026"
    assert r6.registry_stage_id == "transcription-ocr-r6.cvpr2026"
    assert r6.required_script_ids == ["transcription_ocr.cvpr2026.r6"]
    assert r6.runtime_contract["required_run_id"] == "cvpr2026-live-production-r6-expanded-media"
    assert r6.runtime_contract["semantic_validators"] == ["cvpr_transcription_ocr_r6"]


@pytest.mark.repo_data("conferences/cvpr2026/config.yaml")
def test_cvpr_r6_strict_contract_recomputes_a_refreshed_ledger_digest(
    tmp_path: Path,
) -> None:
    """Strict final validation rejects semantic ledger drift behind a new digest."""

    from conference_pipeline.costs.budget_enforcement import _r6_digest
    from tests.test_cvpr2026_r6_runner import _admit, _r6_workspace, _runner_module, _write_json

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
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    ledger["actuals"][0]["actual_cost_usd"] = 0.0
    ledger["ledger_sha256"] = _r6_digest(
        {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    )
    _write_json(paths["ledger"], ledger)
    ledger_file_digest = hashlib.sha256(paths["ledger"].read_bytes()).hexdigest()
    receipt["checks"]["budget_ledger_sha256"] = ledger_file_digest
    receipt["checks"]["budget_ledger_digest"] = ledger["ledger_sha256"]
    evidence = receipt["owner_execution_evidence"]["transcription_ocr.cvpr2026.r6"]
    evidence["receipt_digests"]["budget_ledger"] = ledger_file_digest

    result = verify_stage_contract(
        workspace=workspace,
        run_id="cvpr2026-live-production-r6-expanded-media",
        stage_id="transcription-ocr-r6",
        receipt=receipt,
        strict_execution=True,
    )

    assert result.status == "blocked"
    assert "stage_contract_r6_budget_ledger_invalid" in {
        blocker["code"] for blocker in result.blockers
    }


def test_cvpr_transcription_ocr_strict_contract_requires_exact_r5_caption_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, receipt, registry_path = _cvpr_transcription_ocr_workspace(tmp_path, monkeypatch)

    result = _strict_cvpr_stage5_result(workspace, receipt, registry_path)

    assert result.status == "pass", result.blockers
    assert result.registry_stage_id == "transcription-ocr.cvpr2026"
    assert result.required_script_ids == ["transcription_ocr.cvpr2026"]


def test_r5_stage5_strict_semantics_passes_workspace_boundary_to_inventory_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, receipt, registry_path = _cvpr_transcription_ocr_workspace(tmp_path, monkeypatch)
    captured: dict[str, object] = {}
    original = stage_contracts.validate_event_media_inventory

    def _require_boundary(inventory: object, *, boundary: object, **kwargs: object) -> None:
        captured["boundary"] = boundary
        original(inventory, boundary=boundary, **kwargs)

    monkeypatch.setattr(stage_contracts, "validate_event_media_inventory", _require_boundary)

    result = _strict_cvpr_stage5_result(workspace, receipt, registry_path)

    assert result.status == "pass", result.blockers
    boundary = captured["boundary"]
    assert getattr(boundary, "origin") == "https://cvpr.thecvf.com"
    assert getattr(boundary, "root_segments") == ("virtual", "2026")


@pytest.mark.parametrize(
    "config_text",
    [
        None,
        "conference:\n  name: CVPR 2026 fixture\nurls:\n  virtual: http://cvpr.thecvf.com/virtual/2026\n",
        "conference:\n  name: CVPR 2026 fixture\nurls:\n  virtual: https://foreign.example/virtual/2027\n",
    ],
)
def test_r5_stage5_strict_semantics_translates_workspace_boundary_failures_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_text: str | None
) -> None:
    workspace, receipt, registry_path = _cvpr_transcription_ocr_workspace(tmp_path, monkeypatch)
    stage4_path = workspace / f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-04-download-corpus.json"
    inventory_path = workspace / stage_contracts.EVENT_MEDIA_INVENTORY_RELATIVE_PATH
    stage4_before = hashlib.sha256(stage4_path.read_bytes()).hexdigest()
    inventory_before = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    config_path = workspace / "config.yaml"
    if config_text is None:
        config_path.unlink()
    else:
        config_path.write_text(config_text, encoding="utf-8")

    result = _strict_cvpr_stage5_result(workspace, receipt, registry_path)

    assert result.status == "blocked"
    assert "stage_contract_r5_event_media_inventory_invalid" in {
        blocker["code"] for blocker in result.blockers
    }
    assert hashlib.sha256(stage4_path.read_bytes()).hexdigest() == stage4_before
    assert hashlib.sha256(inventory_path.read_bytes()).hexdigest() == inventory_before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda stage4: stage4.pop("stage_contract"),
        lambda stage4: stage4["stage_contract"].__setitem__("status", "blocked"),
        lambda stage4: stage4["stage_contract"].__setitem__("stage_id", "other-stage"),
        lambda stage4: stage4.__setitem__("stage_id", "other-stage"),
        lambda stage4: stage4.__setitem__("run_id", "cvpr2026-r4"),
        lambda stage4: stage4.__setitem__("mode", "dry-run"),
        lambda stage4: stage4.__setitem__("status", "blocked"),
        lambda stage4: stage4["event_media_inventory"].__setitem__("path", "storage/other.json"),
        lambda stage4: stage4["event_media_inventory"].pop("inventory_sha256"),
        lambda stage4: stage4["checks"].__setitem__("event_media_inventory_path", "storage/other.json"),
        lambda stage4: stage4["checks"].__setitem__("event_media_inventory_sha256", "0" * 64),
        lambda stage4: stage4["checks"].__setitem__(
            "event_media_inventory_canonical_sha256", "0" * 64
        ),
        lambda stage4: stage4["checks"].__setitem__(
            "r4_registration_media_overlay_sha256", "0" * 64
        ),
    ],
    ids=(
        "missing-stage-contract",
        "stage-contract-status",
        "stage-contract-id",
        "stage4-id",
        "stage4-run",
        "stage4-mode",
        "stage4-status",
        "event-media-path",
        "event-media-canonical-binding",
        "published-inventory-path-check",
        "published-inventory-file-check",
        "published-inventory-canonical-check",
        "published-r4-lineage-check",
    ),
)
def test_cvpr_transcription_ocr_strict_contract_rejects_published_stage4_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: object,
) -> None:
    workspace, receipt, registry_path = _cvpr_transcription_ocr_workspace(tmp_path, monkeypatch)
    _rewrite_published_stage4_and_rebind(workspace, receipt, monkeypatch, mutate)

    result = _strict_cvpr_stage5_result(workspace, receipt, registry_path)

    assert result.status == "blocked"
    assert "stage_contract_r5_lineage_mismatch" in {
        blocker["code"] for blocker in result.blockers
    } or "stage_contract_required_receipt_check_mismatch" in {
        blocker["code"] for blocker in result.blockers
    }


def test_cvpr_transcription_ocr_strict_contract_rejects_unpinned_stage4_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, receipt, registry_path = _cvpr_transcription_ocr_workspace(tmp_path, monkeypatch)
    stage4_path = workspace / f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-04-download-corpus.json"
    stage4_path.write_bytes(stage4_path.read_bytes() + b"\n")

    result = _strict_cvpr_stage5_result(workspace, receipt, registry_path)

    assert result.status == "blocked"
    assert "stage_contract_r5_lineage_mismatch" in {
        blocker["code"] for blocker in result.blockers
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda workspace, receipt: receipt["checks"].pop("retry_control_digests"),
        lambda workspace, receipt: receipt["checks"].__setitem__(
            "retry_control_digests", {"archive": {"path": "x", "sha256": "0" * 64}}
        ),
        lambda workspace, receipt: receipt["checks"]["retry_control_digests"].__setitem__(
            "extra", {"path": "x", "sha256": "0" * 64}
        ),
        lambda workspace, receipt: receipt["checks"]["retry_control_digests"].__setitem__(
            "archive", {"path": "storage/elsewhere.json", "sha256": "0" * 64}
        ),
        lambda workspace, receipt: receipt["checks"]["retry_control_digests"]["archive"].__setitem__(
            "sha256", "UPPERCASE"
        ),
        lambda workspace, receipt: (
            workspace
            / f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-05-transcription-ocr.retry-attempt.json"
        ).write_text("tampered\n", encoding="utf-8"),
    ],
    ids=(
        "missing-map",
        "missing-named-controls",
        "extra-control",
        "substituted-path",
        "non-sha256-digest",
        "post-hash-control-tamper",
    ),
)
def test_cvpr_transcription_ocr_strict_contract_rejects_invalid_retry_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: object,
) -> None:
    workspace, receipt, registry_path = _cvpr_transcription_ocr_workspace(tmp_path, monkeypatch)
    mutate(workspace, receipt)  # type: ignore[operator,index]

    result = _strict_cvpr_stage5_result(workspace, receipt, registry_path)

    assert result.status == "blocked"
    assert "stage_contract_r5_retry_control_invalid" in {
        blocker["code"] for blocker in result.blockers
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda workspace, receipt: receipt["checks"]["retry_control_digests"].__setitem__(
            "consumed_authorization", "REDACTED"
        ),
        lambda workspace, receipt: receipt["checks"]["retry_control_digests"]["archive"].__setitem__(
            "path", "storage/foreign.json"
        ),
        lambda workspace, receipt: receipt["checks"]["retry_control_digests"]["attempt"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda workspace, receipt: receipt["checks"]["retry_control_digests"].__setitem__(
            "extra", {"path": "storage/extra.json", "sha256": "0" * 64}
        ),
    ],
    ids=("redacted-consumed-control", "foreign-path", "stale-hash", "extra-control"),
)
def test_cvpr_transcription_ocr_strict_contract_rejects_reloaded_invalid_retry_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: object
) -> None:
    """Strict validation must reject malformed controls after durable JSON reload."""

    workspace, receipt, registry_path = _cvpr_transcription_ocr_workspace(tmp_path, monkeypatch)
    mutate(workspace, receipt)  # type: ignore[operator,index]
    receipt_path = workspace / f"storage/workflow-runs/{CVPR_R5_RUN_ID}/stage-05-transcription-ocr.json"
    _write_json(receipt_path, receipt)
    reloaded = json.loads(receipt_path.read_text(encoding="utf-8"))

    result = _strict_cvpr_stage5_result(workspace, reloaded, registry_path)

    assert result.status == "blocked"
    assert "stage_contract_r5_retry_control_invalid" in {
        blocker["code"] for blocker in result.blockers
    }


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda workspace, receipt: receipt.__setitem__(
                "required_script_ids", ["transcription_ocr.local_media_processing"]
            ),
            "stage_contract_required_owner_ids_missing",
        ),
        (
            lambda workspace, receipt: receipt.__setitem__("mode", "dry-run"),
            "stage_contract_non_live_evidence_rejected",
        ),
        (
            lambda workspace, receipt: receipt.__setitem__("script_registry_sha256", "stale"),
            "stage_contract_registry_digest_mismatch",
        ),
        (
            lambda workspace, receipt: receipt.pop("owner_execution_evidence"),
            "stage_contract_owner_execution_evidence_missing",
        ),
            (lambda workspace, receipt: receipt.__setitem__("run_id", "cvpr2026-r4"), "stage_contract_required_check_mismatch"),
        (lambda workspace, receipt: receipt["checks"].__setitem__("stage_04_receipt_sha256", "0" * 64), "stage_contract_required_check_mismatch"),
        (lambda workspace, receipt: receipt["checks"].__setitem__("event_media_inventory_sha256", "0" * 64), "stage_contract_required_check_mismatch"),
        (lambda workspace, receipt: receipt["checks"].__setitem__("event_media_inventory_canonical_sha256", "0" * 64), "stage_contract_required_check_mismatch"),
        (lambda workspace, receipt: receipt["checks"].__setitem__("r4_registration_media_overlay_sha256", "0" * 64), "stage_contract_required_check_mismatch"),
            (lambda workspace, receipt: receipt["checks"].__setitem__("transcription_ocr_inventory_sha256", "0" * 64), "stage_contract_r5_processor_mismatch"),
            (lambda workspace, receipt: receipt["checks"].__setitem__("workshop_pdf_provenance_table_sha256", "0" * 64), "stage_contract_r5_workshop_provenance_invalid"),
        (lambda workspace, receipt: receipt["checks"].__setitem__("provider_calls_made", 1), "stage_contract_r5_queue_activity"),
        (lambda workspace, receipt: receipt["checks"].__setitem__("secrets_copied", True), "stage_contract_required_check_mismatch"),
    ],
    ids=(
        "generic-owner",
        "non-live",
        "stale-registry",
        "missing-owner-execution",
        "obsolete-r4-run",
        "wrong-r5-receipt-digest",
        "wrong-r5-inventory-file-digest",
        "wrong-r5-inventory-canonical-digest",
        "wrong-immutable-r4-lineage",
        "wrong-stage5-output-digest",
        "wrong-workshop-provenance-binding",
        "provider-activity-in-empty-lane",
        "secret-safety-failure",
    ),
)
def test_cvpr_transcription_ocr_strict_contract_rejects_false_success_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mutate: object,
    expected_code: str,
) -> None:
    workspace, receipt, registry_path = _cvpr_transcription_ocr_workspace(tmp_path, monkeypatch)
    mutate(workspace, receipt)  # type: ignore[operator]

    result = verify_stage_contract(
        workspace=workspace,
        run_id=CVPR_R5_RUN_ID,
        stage_id="transcription-ocr",
        receipt=receipt,
        strict_execution=True,
        registry_path=registry_path,
    )

    assert result.status == "blocked"
    assert expected_code in {blocker["code"] for blocker in result.blockers}


@pytest.mark.parametrize(
    ("path", "value", "expected_code"),
    [
        ("counts.caption_used_no_asr_count", 349, "stage_contract_r5_class_accounting_mismatch"),
        ("capacity_reservation.accepted_caption_count", 349, "stage_contract_r5_capacity_binding_mismatch"),
        ("processor_identity", "other.processor", "stage_contract_r5_processor_mismatch"),
        ("processor_version", "2", "stage_contract_r5_processor_mismatch"),
        ("provider_calls", [{"provider": "invented"}], "stage_contract_r5_queue_activity"),
        ("workshop_talk_boundary.status", "claimed", "stage_contract_workshop_coverage_contradiction"),
    ],
)
def test_cvpr_transcription_ocr_r5_semantics_reject_false_success_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    value: object,
    expected_code: str,
) -> None:
    workspace, receipt, registry_path = _cvpr_transcription_ocr_workspace(tmp_path, monkeypatch)
    inventory_path = workspace / "storage/transcription-ocr/cvpr2026/inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    target = inventory
    *parents, leaf = path.split(".")
    for parent in parents:
        target = target[parent]
    target[leaf] = value
    inventory["inventory_sha256"] = _canonical_json_digest(
        {key: item for key, item in inventory.items() if key != "inventory_sha256"}
    )
    _write_json(inventory_path, inventory)
    receipt["checks"]["transcription_ocr_inventory_sha256"] = hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest()
    receipt["checks"]["transcription_ocr_inventory_canonical_sha256"] = inventory["inventory_sha256"]

    result = verify_stage_contract(
        workspace=workspace,
        run_id=CVPR_R5_RUN_ID,
        stage_id="transcription-ocr",
        receipt=receipt,
        strict_execution=True,
        registry_path=registry_path,
    )

    assert result.status == "blocked"
    assert expected_code in {blocker["code"] for blocker in result.blockers}


def _frontier_authority_fixture(workspace: Path, run_id: str) -> dict[str, object]:
    allocations = [
        {
            "provider": "openrouter",
            "stage": "source-discovery",
            "ceiling_usd": 1.25,
        }
    ]
    registry_path = Path(__file__).resolve().parents[1] / "src" / "conference_pipeline" / "model_governance" / "registry.yaml"
    return {
        "authority_kind": "source_discovery_frontier_bound",
        "authority_digest": "a" * 64,
        "run_id": run_id,
        "workspace": str(workspace.resolve()),
        "base_full_estimate_digest": "b" * 64,
        "candidate_frontier_sha256": "c" * 64,
        "retained_join_quantity": 2,
        "model_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "provider": "openrouter",
        "stage": "source-discovery",
        "provider_stage_allocations": allocations,
        "spend_authority": {
            "mode": "automatic_under_usd_800",
            "provider_stage_allocations": allocations,
        },
        "fresh_volatile_preflight_required": True,
    }


def test_pre_spend_contract_requires_exact_frontier_authorization_without_dispatch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    authority = _frontier_authority_fixture(workspace, "run-1")

    result = verify_pre_spend_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="source-discovery",
        mode="live",
        frontier_authority=authority,
        base_full_estimate_digest="b" * 64,
        candidate_frontier_sha256="c" * 64,
        retained_join_quantity=2,
        provider="openrouter",
        provider_stage="source-discovery",
        volatile_preflight={
            "status": "succeeded",
            "frontier_authority_digest": "a" * 64,
        },
    )

    assert result.status == "precondition_passed"
    assert result.blockers == []
    assert result.provider_dispatch_permitted is False
    assert result.provider_calls_made == 0
    assert result.actual_cost_usd == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda authority, volatile: authority.pop("base_full_estimate_digest"),
        lambda authority, volatile: authority.__setitem__("model_registry_sha256", "stale"),
        lambda authority, volatile: authority.__setitem__("candidate_frontier_sha256", "changed"),
        lambda authority, volatile: authority.__setitem__("retained_join_quantity", 3),
        lambda authority, volatile: authority["provider_stage_allocations"].__setitem__(
            0, {"provider": "openrouter", "stage": "analysis", "ceiling_usd": 1.25}
        ),
        lambda authority, volatile: authority["provider_stage_allocations"][0].__setitem__(
            "ceiling_usd", math.inf
        ),
        lambda authority, volatile: volatile.__setitem__("frontier_authority_digest", "stale"),
    ],
)
def test_pre_spend_contract_rejects_stale_missing_or_unpriced_authority(
    tmp_path: Path,
    mutation: object,
) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    authority = copy.deepcopy(_frontier_authority_fixture(workspace, "run-1"))
    volatile = {"status": "succeeded", "frontier_authority_digest": "a" * 64}
    mutation(authority, volatile)  # type: ignore[operator]

    result = verify_pre_spend_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="source-discovery",
        mode="live",
        frontier_authority=authority,
        base_full_estimate_digest="b" * 64,
        candidate_frontier_sha256="c" * 64,
        retained_join_quantity=2,
        provider="openrouter",
        provider_stage="source-discovery",
        volatile_preflight=volatile,
    )

    assert result.status == "blocked"
    assert result.provider_dispatch_permitted is False
    assert result.provider_calls_made == 0
    assert result.actual_cost_usd == 0.0
    assert {blocker["code"] for blocker in result.blockers} == {"precondition_contract_unmet"}


def test_pre_spend_contract_cannot_serve_as_final_stage_success(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    authority = _frontier_authority_fixture(workspace, "run-1")
    pre_spend = verify_pre_spend_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="source-discovery",
        mode="live",
        frontier_authority=authority,
        base_full_estimate_digest="b" * 64,
        candidate_frontier_sha256="c" * 64,
        retained_join_quantity=2,
        provider="openrouter",
        provider_stage="source-discovery",
        volatile_preflight={
            "status": "succeeded",
            "frontier_authority_digest": "a" * 64,
        },
    )

    final = verify_stage_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="source-discovery",
        receipt=pre_spend.to_dict(),
    )

    assert final.status == "blocked"
    assert "stage_contract_required_output_missing" in {
        blocker["code"] for blocker in final.blockers
    }


def test_tavily_priority_gap_pre_spend_contract_binds_exact_non_model_frontier(
    tmp_path: Path,
) -> None:
    """Tavily is priced search, not an invented model-governance task."""

    workspace = tmp_path / "conferences" / "cvpr2026"
    registry_path = Path(__file__).resolve().parents[1] / "src" / "conference_pipeline" / "model_governance" / "registry.yaml"
    task_specs_path = Path(__file__).resolve().parents[1] / "src" / "conference_pipeline" / "costs" / "task_specs.yaml"
    pricing_path = Path(__file__).resolve().parents[1] / "src" / "conference_pipeline" / "costs" / "pricing_catalog.json"
    allocations = [{"provider": "tavily", "stage": "download-corpus", "ceiling_usd": 0.02}]
    authority = {
        "authority_kind": "cvpr2026_tavily_priority_gap_bound",
        "authority_digest": "a" * 64,
        "run_id": "run-1",
        "workspace": str(workspace.resolve()),
        "base_full_estimate_digest": "b" * 64,
        "program_spine_sha256": "c" * 64,
        "harvest_inventory_sha256": "d" * 64,
        "query_policy_sha256": "e" * 64,
        "priority_gap_frontier_sha256": "f" * 64,
        "priority_gap_quantity": 2,
        "priority_program_item_ids": ["cvpr:item:001", "cvpr:item:002"],
        "task_id": "acquisition.youtube_talk_harvest.tavily_priority_gap_search",
        "cost_task_id": "cvpr2026_tavily_priority_gap_search",
        "model_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "task_specs_sha256": hashlib.sha256(task_specs_path.read_bytes()).hexdigest(),
        "pricing_catalog_sha256": hashlib.sha256(pricing_path.read_bytes()).hexdigest(),
        "provider": "tavily",
        "stage": "download-corpus",
        "provider_stage_allocations": allocations,
        "spend_authority": {
            "mode": "automatic_under_usd_800",
            "provider_stage_allocations": allocations,
        },
        "fresh_volatile_preflight_required": True,
    }

    result = verify_tavily_priority_gap_pre_spend_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="download-corpus",
        mode="live",
        frontier_authority=authority,
        base_full_estimate_digest="b" * 64,
        program_spine_sha256="c" * 64,
        harvest_inventory_sha256="d" * 64,
        query_policy_sha256="e" * 64,
        priority_gap_frontier_sha256="f" * 64,
        priority_gap_quantity=2,
        provider="tavily",
        provider_stage="download-corpus",
        volatile_preflight={
            "status": "succeeded",
            "frontier_authority_digest": "a" * 64,
            "provider_probe_scope": {"provider": "tavily", "stage": "download-corpus"},
        },
    )

    assert result.status == "precondition_passed"
    assert result.provider_dispatch_permitted is False
    authority["priority_gap_quantity"] = 3
    blocked = verify_tavily_priority_gap_pre_spend_contract(
        workspace=workspace,
        run_id="run-1",
        stage_id="download-corpus",
        mode="live",
        frontier_authority=authority,
        base_full_estimate_digest="b" * 64,
        program_spine_sha256="c" * 64,
        harvest_inventory_sha256="d" * 64,
        query_policy_sha256="e" * 64,
        priority_gap_frontier_sha256="f" * 64,
        priority_gap_quantity=2,
        provider="tavily",
        provider_stage="download-corpus",
        volatile_preflight={"status": "succeeded", "frontier_authority_digest": "a" * 64},
    )
    assert blocked.status == "blocked"
    assert blocked.provider_calls_made == 0


def test_r5_expected_counts_are_derived_from_the_bound_event_media_inventory() -> None:
    """The r5 accounting contract derives its expected counts from the
    digest-verified event-media inventory instead of pinning one run's
    numbers as literals, so a workspace with different coverage is judged on
    internal consistency, not on equality with CVPR 2026's July harvest."""

    inventory = {
        "counts": {"terminal_event_route_count": 12},
        "caption_candidates": [
            {"video_id": "a", "caption_status": "youtube_caption_available"},
            {"video_id": "b", "caption_status": "youtube_caption_available"},
            {"video_id": "c", "caption_status": "youtube_caption_unavailable"},
            {"video_id": "d", "caption_status": "pending"},
            "not-a-mapping",
        ],
    }

    assert stage_contracts._r5_expected_counts_from_event_media(inventory) == {
        "terminal_route_count": 12,
        "caption_candidate_count": 5,
        "caption_used_no_asr_count": 2,
        "youtube_caption_unavailable_count": 1,
        "asr_queue_count": 0,
        "ocr_queue_count": 0,
        "provider_call_count": 0,
    }
    assert stage_contracts._r5_expected_counts_from_event_media({}) == {
        "terminal_route_count": 0,
        "caption_candidate_count": 0,
        "caption_used_no_asr_count": 0,
        "youtube_caption_unavailable_count": 0,
        "asr_queue_count": 0,
        "ocr_queue_count": 0,
        "provider_call_count": 0,
    }


def test_lineage_is_required_by_default() -> None:
    """v5.1: every production runner binds lineage, so the verifier fails closed
    on a receipt without one unless a caller explicitly opts out."""

    assert stage_contracts.REQUIRE_LINEAGE_DEFAULT is True
