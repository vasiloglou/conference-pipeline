import json
import hashlib
import os
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    "status",
    ["blocked", "deferred_blocked", "failed", "unavailable", "unsupported"],
)
def test_iclr_v151_summary_halts_on_all_v192_blocking_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    from conference_pipeline.iclr2025_live import run_summary_path, write_stage_receipt

    write_stage_receipt(
        tmp_path,
        run_id="v151-test",
        stage_id="v151-byte-reconciliation",
        status=status,
        blockers=[{"reason": f"synthetic {status}"}],
    )

    summary = yaml.safe_load(run_summary_path(tmp_path, "v151-test").read_text())
    assert summary["status"] == "blocked"
    assert summary["halt_status"] == "halted"
    assert summary["halt_stage_id"] == "v151-byte-reconciliation"
    assert summary["blocker_receipt_path"].endswith("v151-67-byte-reconciliation.json")


def test_iclr_v151_summary_prefers_v151_blocker_over_stale_historical_receipts(
    tmp_path: Path,
) -> None:
    from conference_pipeline.iclr2025_live import run_summary_path, write_stage_receipt

    write_stage_receipt(
        tmp_path,
        run_id="v151-test",
        stage_id="analysis",
        status="blocked",
        blockers=[{"reason": "historical stale analysis blocker"}],
    )
    write_stage_receipt(
        tmp_path,
        run_id="v151-test",
        stage_id="v151-byte-reconciliation",
        status="blocked",
        blockers=[{"reason": "current v1.51 source boundary blocker"}],
    )

    summary = yaml.safe_load(run_summary_path(tmp_path, "v151-test").read_text())
    assert summary["status"] == "blocked"
    assert summary["halt_stage_id"] == "v151-byte-reconciliation"
    assert summary["blocker_receipt_path"].endswith("v151-67-byte-reconciliation.json")


def test_iclr_v151_paper_matching_requires_live_mode(tmp_path: Path) -> None:
    from conference_pipeline.iclr2025_live import run_production_stage

    receipt = run_production_stage(
        tmp_path,
        stage="v151-paper-matching",
        run_id="v151-test",
        max_papers=1,
        resume=True,
        render_audio=False,
        approval_file=None,
        max_media_items=None,
        max_pdf_pages_per_item=None,
        media_kind="all",
        audio_workers=1,
        local_workers=1,
        use_llm_visual_ocr=False,
        mode="dry-run",
    )

    assert receipt["status"] == "blocked"
    assert receipt["stage_id"] == "v151-paper-matching"
    assert "--mode live" in receipt["blockers"][0]["reason"]


def test_iclr_v151_stage_catalog_includes_recovery_and_paper_matching() -> None:
    from conference_pipeline.iclr2025_live import (
        CONTINUATION_STAGE_FILES,
        V151_RECOVERY_RUN_ID,
        V151_STAGE_IDS,
    )

    assert V151_RECOVERY_RUN_ID == "v151-iclr2025-recovery"
    assert "v151-paper-matching" in CONTINUATION_STAGE_FILES
    assert "v151-presentation-sonnet-svg" in V151_STAGE_IDS


@pytest.mark.repo_data("conferences/iclr2025/config.yaml")
def test_iclr_v151_slide_ocr_and_svg_provider_contracts() -> None:
    config = yaml.safe_load(Path("conferences/iclr2025/config.yaml").read_text())
    workflow = config["workflow"]

    visual_ocr = workflow["transcription_ocr"]["visual_ocr"]
    assert visual_ocr["enabled"] is False
    assert visual_ocr["posture"] == "skipped_with_rationale"

    presentation = workflow["presentation"]
    assert presentation["programmatic_figure_provider"] == "anthropic"
    assert presentation["programmatic_figure_model"] == "claude-sonnet-4-6"

    task_specs = yaml.safe_load(
        Path("src/conference_pipeline/costs/task_specs.yaml").read_text()
    )
    figure_codegen = next(
        task for task in task_specs["tasks"] if task["task_id"] == "programmatic_figure_codegen"
    )
    assert figure_codegen["provider"] == "anthropic"
    # Shared forward-looking specs moved to claude-sonnet-5 in the July 2026
    # model-upgrade sweep; the pinned v1.51 conference config above stays on
    # claude-sonnet-4-6.
    assert set(figure_codegen["models"].values()) == {"claude-sonnet-5"}

    from conference_pipeline.model_governance import default_model, default_provider, load_registry

    assert default_provider("presentation.svg_figure_generation") == "anthropic"
    assert default_model("presentation.svg_figure_generation") == "claude-sonnet-5"
    svg_generation = load_registry().get("presentation.svg_figure_generation")
    assert all(
        selection.provider != "gemini" for selection in svg_generation.allowed_selections
    )


def _write_succeeded_v15_transcription_evidence(workspace_root: Path) -> None:
    from conference_pipeline.iclr2025_live import PRODUCTION_RUN_ID, stage_path

    source_receipt = stage_path(workspace_root, PRODUCTION_RUN_ID, "transcription-ocr")
    source_receipt.parent.mkdir(parents=True, exist_ok=True)
    source_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": PRODUCTION_RUN_ID,
                "stage_id": "transcription-ocr",
                "status": "succeeded",
                "artifacts": [],
                "blockers": [],
                "details": {"transcript_ocr_artifact_count": 2939},
            }
        ),
        encoding="utf-8",
    )
    receipt_bundle = (
        workspace_root / "storage" / "transcription-ocr" / "v15_transcription_ocr_receipts.json"
    )
    receipt_bundle.parent.mkdir(parents=True, exist_ok=True)
    text_path = (
        workspace_root
        / "storage"
        / "transcription-ocr"
        / "text-extraction"
        / "openreview"
        / "paper-1.txt"
    )
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_content = "reusable paper text"
    text_path.write_text(text_content, encoding="utf-8")
    text_sha = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
    receipt_bundle.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "artifact_count": 1,
                "artifacts": [
                    {
                        "artifact_id": "02_model_architectures:paper-1",
                        "source_path": "cache/acquisition-downloads/openreview/paper-1.pdf",
                        "status": "text_extracted",
                        "text_path": "storage/transcription-ocr/text-extraction/openreview/paper-1.txt",
                        "sha256": text_sha,
                    }
                ],
                "media_artifacts": [],
            }
        ),
        encoding="utf-8",
    )


def _write_v151_phase68_inputs(workspace_root: Path) -> None:
    from conference_pipeline.iclr2025_live import write_stage_receipt

    write_stage_receipt(
        workspace_root,
        run_id="v151-test",
        stage_id="v151-byte-reconciliation",
        status="succeeded",
        details={
            "outcome": "passed",
            "outcome_reason_code": "cap_free_source_boundary_confirmed",
            "source_cap_status": {
                "source_cap_free": True,
                "source_query_records_fetched": [3703],
            },
        },
    )
    acquisition_dir = workspace_root / "storage" / "acquisition"
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    (acquisition_dir / "v15_selected_corpus.json").write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "02_model_architectures",
                        "papers": [
                            {
                                "id": "paper-1",
                                "title": "A Paper",
                                "forum_url": "https://openreview.net/forum?id=paper-1",
                                "pdf_url": "https://openreview.net/pdf?id=paper-1",
                            }
                        ],
                    }
                ],
                "source_gap_records": [],
            }
        ),
        encoding="utf-8",
    )
    (acquisition_dir / "v15_download_receipts.json").write_text(
        json.dumps(
            {
                "local_artifacts": [
                    {
                        "paper_id": "paper-1",
                        "source_kind": "paper_pdf",
                        "local_path": "cache/acquisition-downloads/openreview/paper-1.pdf",
                        "sha256": "b" * 64,
                        "storage_lane": "cache/acquisition-downloads",
                        "retention_class": "local-cache",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (acquisition_dir / "v15_media_download_receipts.json").write_text(
        json.dumps({"local_media_artifacts": []}),
        encoding="utf-8",
    )


def test_iclr_v151_slide_ocr_skip_rebinds_v15_transcription_receipt(tmp_path: Path) -> None:
    from conference_pipeline.iclr2025_live import (
        run_v151_slide_ocr_skip_corpus_stage,
        stage_path,
    )

    _write_v151_phase68_inputs(tmp_path)
    _write_succeeded_v15_transcription_evidence(tmp_path)

    receipt = run_v151_slide_ocr_skip_corpus_stage(tmp_path, run_id="v151-test")

    rebound_path = stage_path(tmp_path, "v151-test", "transcription-ocr")
    rebound = json.loads(rebound_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "succeeded"
    assert receipt["details"]["transcription_ocr_rebind"]["status"] == "succeeded"
    assert any(
        artifact["kind"] == "v151-transcription-ocr-rebind-receipt"
        for artifact in receipt["artifacts"]
    )
    assert rebound["status"] == "succeeded"
    assert rebound["details"]["recovery_rebind"] is True
    assert rebound["details"]["old_v15_receipts_are_completion_evidence"] is False
    assert rebound["details"]["slide_ocr_posture"] == "skipped_with_rationale"
    assert receipt["details"]["corpus_rebind"]["status"] == "succeeded"
    assert receipt["details"]["corpus_rebind"]["source_cap_free"] is True

    inventory = json.loads(
        (tmp_path / "storage" / "corpus" / "v15_corpus_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    assert inventory["v151_rebind"]["old_v15_receipts_are_completion_evidence"] is False
    assert inventory["v151_rebind"]["source_cap_free"] is True
    assert inventory["slide_ocr_posture"] == "skipped_with_rationale"
    assert inventory["artifacts"][0]["processing_status"] == "text_extracted"
    assert inventory["artifacts"][0]["ocr_path"].endswith("paper-1.txt")


def test_iclr_v151_slide_ocr_skip_requires_cap_free_phase67_receipt(tmp_path: Path) -> None:
    from conference_pipeline.iclr2025_live import run_v151_slide_ocr_skip_corpus_stage

    _write_succeeded_v15_transcription_evidence(tmp_path)

    receipt = run_v151_slide_ocr_skip_corpus_stage(tmp_path, run_id="v151-test")

    assert receipt["status"] == "blocked"
    assert "Phase 68 requires a succeeded cap-free Phase 67" in receipt["blockers"][0]["reason"]


def test_iclr_v151_slide_ocr_skip_blocks_when_rebind_inputs_missing(tmp_path: Path) -> None:
    from conference_pipeline.iclr2025_live import run_v151_slide_ocr_skip_corpus_stage

    _write_succeeded_v15_transcription_evidence(tmp_path)
    from conference_pipeline.iclr2025_live import write_stage_receipt

    write_stage_receipt(
        tmp_path,
        run_id="v151-test",
        stage_id="v151-byte-reconciliation",
        status="succeeded",
        details={
            "outcome_reason_code": "cap_free_source_boundary_confirmed",
            "source_cap_status": {"source_cap_free": True},
        },
    )

    receipt = run_v151_slide_ocr_skip_corpus_stage(tmp_path, run_id="v151-test")

    assert receipt["status"] == "blocked"
    assert "corpus inventory rebind inputs" in receipt["blockers"][0]["reason"]


def test_iclr_v151_corpus_rebind_rejects_missing_or_stale_text_evidence(
    tmp_path: Path,
) -> None:
    from conference_pipeline.iclr2025_live import run_v151_slide_ocr_skip_corpus_stage

    _write_v151_phase68_inputs(tmp_path)
    _write_succeeded_v15_transcription_evidence(tmp_path)

    stale_inventory_path = tmp_path / "storage" / "corpus" / "v15_corpus_inventory.json"
    stale_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    stale_inventory_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "artifact_id": "02_model_architectures:paper-1",
                        "local_path": "cache/acquisition-downloads/openreview/paper-1.pdf",
                        "provenance_digest": "not-the-current-digest",
                        "ocr_path": "storage/transcription-ocr/text-extraction/openreview/missing.txt",
                        "text_extraction_sha256": "f" * 64,
                        "processing_status": "text_extracted",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    receipt = run_v151_slide_ocr_skip_corpus_stage(tmp_path, run_id="v151-test")
    inventory = json.loads(stale_inventory_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "succeeded"
    assert inventory["artifacts"][0]["processing_status"] == "text_extracted"
    assert inventory["artifacts"][0]["ocr_path"].endswith("paper-1.txt")


def test_iclr_v151_exhaustive_rebinds_transcription_before_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import conference_pipeline.iclr2025_live as live

    _write_succeeded_v15_transcription_evidence(tmp_path)

    def fake_exhaustive_topic_extraction(
        workspace_root: Path,
        *,
        run_id: str,
        force_rebuild: bool,
    ) -> dict:
        assert live.stage_succeeded(workspace_root, run_id, "transcription-ocr")
        return live.write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="exhaustive-topic-extraction",
            status="succeeded",
            details={"force_rebuild": force_rebuild},
        )

    monkeypatch.setattr(live, "run_exhaustive_topic_extraction", fake_exhaustive_topic_extraction)

    receipt = live.run_v151_exhaustive_topic_extraction_stage(
        tmp_path,
        run_id="v151-test",
        mode="live",
        force_rebuild=False,
    )

    assert receipt["status"] == "succeeded"
    assert receipt["details"]["provider_mode"] == "mlx"
    assert live.stage_succeeded(tmp_path, "v151-test", "transcription-ocr")


def test_local_researcher_uses_model_as_fast_and_smart_model() -> None:
    from conference_pipeline.research.researcher import build_researcher

    researcher = build_researcher(
        {
            "provider": "local",
            "model": "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
            "base_url": "http://127.0.0.1:8080/v1",
        }
    )

    assert researcher.fast_model == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    assert researcher.smart_model == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    assert researcher.model == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"


def test_iclr_v151_paper_matching_uses_local_qwen_research_and_builds_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import conference_pipeline.iclr2025_live as live
    import conference_pipeline.paper_matching as paper_matching

    calls: dict[str, object] = {}

    def fake_vector_store(workspace_root: Path, *, run_id: str, **_kwargs: object) -> dict:
        for store_kind in ("artifacts", "papers"):
            coverage_path = (
                workspace_root
                / "storage"
                / "vector-store"
                / run_id
                / "stores"
                / store_kind
                / "coverage.json"
            )
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            coverage_path.write_text(
                json.dumps({"schema_version": 1, "store_kind": store_kind, "chunk_count": 1}),
                encoding="utf-8",
            )
        return live.write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="enhanced-indexing-vector-store",
            status="succeeded",
        )

    def fake_paper_matching_stage(
        *,
        workspace_root: Path,
        run_id: str,
        topic_id: str,
        config: object,
    ) -> dict:
        calls["keypoint_provider"] = config.keypoint_provider
        calls["keypoint_model"] = config.keypoint_model
        calls["research_provider"] = config.research_provider
        calls["research_model"] = config.research_model
        calls["output_top_n"] = config.output_top_n
        calls["runtime_kind"] = os.environ.get("GPTR_LOCAL_LLM_RUNTIME_KIND")
        calls["smart_model"] = os.environ.get("GPTR_LOCAL_LLM_SMART_MODEL")
        calls.setdefault("topics", []).append(topic_id)
        assert live.stage_succeeded(workspace_root, run_id, "enhanced-indexing-vector-store")
        return {
            "status": "succeeded",
            "run_id": run_id,
            "topics": [topic_id],
            "artifacts": [],
        }

    monkeypatch.setattr(live, "run_enhanced_indexing_vector_store", fake_vector_store)
    monkeypatch.setattr(paper_matching, "run_paper_matching_stage", fake_paper_matching_stage)

    receipt = live.run_v151_paper_matching_stage(
        tmp_path,
        run_id="v151-test",
        mode="live",
    )

    assert receipt["status"] == "succeeded"
    assert calls["keypoint_provider"] == "local"
    assert calls["keypoint_model"] == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    assert calls["research_provider"] == "local"
    assert calls["research_model"] == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    assert calls["output_top_n"] == 20
    assert calls["runtime_kind"] == "mlx"
    assert calls["smart_model"] == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    assert calls["topics"] == list(live.V151_PRODUCTION_TOPIC_IDS)
    assert receipt["details"]["frontier_extractor_provider"] == "local"
    assert receipt["details"]["frontier_extractor_model"] == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    assert receipt["details"]["paper_matching_cap_verification"]["status"] == "pass"


def test_iclr_v151_paper_matching_can_scope_to_one_topic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import conference_pipeline.iclr2025_live as live
    import conference_pipeline.paper_matching as paper_matching

    calls: dict[str, object] = {}

    def fake_vector_store(workspace_root: Path, *, run_id: str, **_kwargs: object) -> dict:
        for store_kind in ("artifacts", "papers"):
            coverage_path = (
                workspace_root
                / "storage"
                / "vector-store"
                / run_id
                / "stores"
                / store_kind
                / "coverage.json"
            )
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            coverage_path.write_text(
                json.dumps({"schema_version": 1, "store_kind": store_kind, "chunk_count": 1}),
                encoding="utf-8",
            )
        return live.write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="enhanced-indexing-vector-store",
            status="succeeded",
        )

    def fake_paper_matching_stage(
        *,
        workspace_root: Path,
        run_id: str,
        topic_id: str,
        config: object,
    ) -> dict:
        calls.setdefault("topics", []).append(topic_id)
        return {
            "status": "succeeded",
            "run_id": run_id,
            "topics": [topic_id],
            "artifacts": [],
        }

    monkeypatch.setattr(live, "run_enhanced_indexing_vector_store", fake_vector_store)
    monkeypatch.setattr(paper_matching, "run_paper_matching_stage", fake_paper_matching_stage)

    receipt = live.run_v151_paper_matching_stage(
        tmp_path,
        run_id="v151-test",
        mode="live",
        topic_id="02_model_architectures",
    )

    assert receipt["status"] == "succeeded"
    assert calls["topics"] == ["02_model_architectures"]
    assert receipt["details"]["analysis_scope"] == "topic"
    assert receipt["details"]["scoped_topic_ids"] == ["02_model_architectures"]
    assert receipt["details"]["paper_matching_cap_verification"]["status"] == "pass"


def test_paper_matching_cap_verification_uses_workflow_config(tmp_path: Path) -> None:
    import conference_pipeline.iclr2025_live as live

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "workflow": {
                    "paper_matching": {"output_top_n": 30},
                    "deck": {"max_papers": 30},
                }
            }
        ),
        encoding="utf-8",
    )

    verification = live.paper_matching_cap_verification(tmp_path)

    assert verification["status"] == "pass"
    assert verification["observed_caps"] == {
        "workflow.paper_matching.output_top_n": 30,
        "v151.paper_matching.output_top_n": 30,
        "analysis.matched_paper_top_k": 30,
        "workflow.deck.max_papers": 30,
    }


def test_paper_matching_cap_verification_blocks_hidden_downstream_caps(
    tmp_path: Path,
) -> None:
    import conference_pipeline.iclr2025_live as live

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "workflow": {
                    "paper_matching": {"output_top_n": 20},
                    "deck": {"max_papers": 20},
                }
            }
        ),
        encoding="utf-8",
    )

    verification = live.paper_matching_cap_verification(
        tmp_path,
        paper_matching_output_top_n=8,
        analysis_matched_paper_top_k=5,
        deck_max_papers=10,
    )

    assert verification["status"] == "blocked"
    assert {
        blocker["cap_name"] for blocker in verification["blockers"]
    } == {
        "v151.paper_matching.output_top_n",
        "analysis.matched_paper_top_k",
        "workflow.deck.max_papers",
    }
    assert {
        blocker["code"] for blocker in verification["blockers"]
    } == {"paper_matching_cap_below_configured_output_top_n"}


def test_iclr_v151_paper_matching_blocks_when_deck_cap_undercuts_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import conference_pipeline.iclr2025_live as live
    import conference_pipeline.paper_matching as paper_matching

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "workflow": {
                    "paper_matching": {"output_top_n": 20},
                    "deck": {"max_papers": 8},
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_vector_store(workspace_root: Path, *, run_id: str, **_kwargs: object) -> dict:
        for store_kind in ("artifacts", "papers"):
            coverage_path = (
                workspace_root
                / "storage"
                / "vector-store"
                / run_id
                / "stores"
                / store_kind
                / "coverage.json"
            )
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            coverage_path.write_text(
                json.dumps({"schema_version": 1, "store_kind": store_kind, "chunk_count": 1}),
                encoding="utf-8",
            )
        return live.write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="enhanced-indexing-vector-store",
            status="succeeded",
        )

    def fail_paper_matching_stage(**_kwargs: object) -> dict:
        raise AssertionError("paper matching should not run when caps are blocked")

    monkeypatch.setattr(live, "run_enhanced_indexing_vector_store", fake_vector_store)
    monkeypatch.setattr(paper_matching, "run_paper_matching_stage", fail_paper_matching_stage)

    receipt = live.run_v151_paper_matching_stage(
        tmp_path,
        run_id="v151-test",
        mode="live",
    )

    assert receipt["status"] == "blocked"
    assert receipt["details"]["paper_matching_cap_verification"]["status"] == "blocked"
    assert receipt["blockers"][0]["code"] == "paper_matching_cap_below_configured_output_top_n"
    assert receipt["blockers"][0]["cap_name"] == "workflow.deck.max_papers"


PHASE67_DETAIL_KEYS = {
    "outcome",
    "outcome_reason_code",
    "source_cap_status",
    "stale_cache_status",
    "byte_reconciliation",
    "budget_guardrail",
    "old_v15_receipts_are_completion_evidence",
    "recovery_boundary",
}

PHASE67_SOURCE_CAP_KEYS = {
    "source_cap_free",
    "source_receipt_path",
    "source_receipt_sha256",
    "source_receipt_status",
    "rate_limits_max_papers",
    "rate_limits_page_size",
    "paper_import_scope",
    "full_accepted_paper_corpus_requested",
    "source_query_explicit_max_papers",
    "source_query_records_fetched",
    "stale_cap_evidence",
}

PHASE67_STALE_CACHE_KEYS = {
    "openreview_cache_path",
    "openreview_cache_sha256",
    "openreview_papers_count",
    "prior_180_paper_cache_status",
    "quarantine_status",
}

PHASE67_BYTE_KEYS = {
    "download_receipt_status",
    "media_receipt_status",
    "download_receipt_path",
    "download_receipt_sha256",
    "media_receipt_path",
    "media_receipt_sha256",
    "referenced_download_bytes_status",
    "referenced_media_bytes_status",
    "referenced_byte_digest_status",
    "referenced_download_bytes_count",
    "referenced_media_bytes_count",
    "missing_referenced_byte_paths",
    "digest_mismatches",
    "reusable_local_bytes_decision",
    "reusable_backup_bytes_decision",
    "backup_reconciliation_status",
}

PHASE67_BUDGET_KEYS = {
    "config_path",
    "budget_cap_usd",
    "budget_cap_role",
    "budget_cap_preserved",
    "corpus_cap_workaround_used",
    "actual_cost_scan_scope",
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_phase67_config(workspace_root: Path) -> None:
    (workspace_root / "config.yaml").write_text(
        "workflow:\n"
        "  provider_cost:\n"
        "    budget_cap_usd: 1000\n",
        encoding="utf-8",
    )


def _seed_phase67_workspace(
    workspace_root: Path,
    *,
    source_max_papers: int | None = None,
    source_page_size: int | None = 1000,
    paper_import_scope: str = "all_openreview_accepted",
    full_corpus_requested: bool = True,
    source_explicit_max_papers: int | None = None,
    source_records_fetched: int = 181,
    command: str | None = None,
    openreview_paper_count: int = 181,
    omit_download_receipt: bool = False,
    omit_media_receipt: bool = False,
    empty_download_receipt: bool = False,
    empty_media_receipt: bool = False,
    missing_download_byte: bool = False,
    missing_media_byte: bool = False,
    mismatched_download_digest: bool = False,
    mismatched_media_digest: bool = False,
    use_digest_field_for_media: bool = False,
    include_approval_metadata_path: bool = False,
) -> None:
    _write_phase67_config(workspace_root)
    acquisition = workspace_root / "storage" / "acquisition"

    source_receipt = {
        "schema_version": 1,
        "mode": "live",
        "command": command,
        "rate_limits": {
            "max_papers": source_max_papers,
            "page_size": source_page_size,
            "paper_import_scope": paper_import_scope,
        },
        "guardrails": {
            "full_accepted_paper_corpus_requested": full_corpus_requested,
            "secrets_copied": False,
        },
        "source_queries": [
            {
                "source_family": "openreview-venue",
                "explicit_max_papers": source_explicit_max_papers,
                "records_fetched": source_records_fetched,
            }
        ],
    }
    _write_json(acquisition / "v15_source_import_receipt.json", source_receipt)
    _write_json(
        acquisition / "v15_openreview_papers.json",
        {
            "schema_version": 1,
            "papers": [{"id": f"paper-{index}"} for index in range(openreview_paper_count)],
        },
    )

    download_content = b"download-bytes"
    media_content = b"media-bytes"
    download_rel = "cache/acquisition-downloads/openreview/paper.pdf"
    media_rel = "cache/acquisition-downloads/media/session/raw.m4a"
    download_path = workspace_root / download_rel
    media_path = workspace_root / media_rel
    if not missing_download_byte:
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download_path.write_bytes(download_content)
    if not missing_media_byte:
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(media_content)

    download_digest = _sha256_bytes(download_content)
    media_digest = _sha256_bytes(media_content)
    if mismatched_download_digest:
        download_digest = _sha256_bytes(b"wrong-download")
    if mismatched_media_digest:
        media_digest = _sha256_bytes(b"wrong-media")

    approval_metadata = {
        "receipt": {
            "path": "/outside-workspace/live_acquisition_approval.json",
            "exists": True,
        }
    }

    if not omit_download_receipt:
        download_receipt = {
            "schema_version": 1,
            "local_artifacts": (
                []
                if empty_download_receipt
                else [
                    {
                        "artifact_path": download_rel,
                        "sha256": download_digest,
                    }
                ]
            ),
        }
        if include_approval_metadata_path:
            download_receipt["approval"] = approval_metadata
        _write_json(
            acquisition / "v15_download_receipts.json",
            download_receipt,
        )
    if not omit_media_receipt:
        media_record = {"media_path": media_rel}
        if use_digest_field_for_media:
            media_record["digest"] = media_digest
            media_record["digest_algorithm"] = "sha256"
        else:
            media_record["sha256"] = media_digest
        media_receipt = {
            "schema_version": 1,
            "local_media_artifacts": [] if empty_media_receipt else [media_record],
        }
        if include_approval_metadata_path:
            media_receipt["approval"] = approval_metadata
        _write_json(
            acquisition / "v15_media_download_receipts.json",
            media_receipt,
        )


def _run_phase67(workspace_root: Path) -> dict:
    from conference_pipeline.iclr2025_live import run_v151_byte_reconciliation_stage

    return run_v151_byte_reconciliation_stage(workspace_root, run_id="v151-test")


def _assert_phase67_contract(receipt: dict) -> dict:
    assert receipt["schema_version"] == 1
    assert receipt["stage_id"] == "v151-byte-reconciliation"
    assert receipt["phase"] == 67
    assert receipt["status"] in {"succeeded", "blocked"}
    assert receipt["secrets_copied"] is False
    details = receipt["details"]
    assert PHASE67_DETAIL_KEYS <= set(details)
    assert PHASE67_SOURCE_CAP_KEYS <= set(details["source_cap_status"])
    assert PHASE67_STALE_CACHE_KEYS <= set(details["stale_cache_status"])
    assert PHASE67_BYTE_KEYS <= set(details["byte_reconciliation"])
    assert PHASE67_BUDGET_KEYS <= set(details["budget_guardrail"])
    assert details["old_v15_receipts_are_completion_evidence"] is False
    assert details["recovery_boundary"] == "do_not_redownload_by_default"
    if receipt["status"] == "blocked":
        assert details["outcome"] == "blocked"
        assert details["outcome_reason_code"] == "source_boundary_unresolved"
        assert receipt["blockers"]
        assert all(blocker["retryable"] is True for blocker in receipt["blockers"])
    return details


@pytest.mark.parametrize(
    ("source_kwargs", "expected_field"),
    [
        ({"source_max_papers": 180}, "rate_limits.max_papers"),
        ({"source_page_size": 180}, "rate_limits.page_size"),
    ],
)
def test_v151_byte_reconciliation_blocks_structured_source_caps(
    tmp_path: Path,
    source_kwargs: dict[str, int],
    expected_field: str,
) -> None:
    _seed_phase67_workspace(tmp_path, **source_kwargs)

    receipt = _run_phase67(tmp_path)

    details = _assert_phase67_contract(receipt)
    assert receipt["status"] == "blocked"
    stale_fields = {
        evidence["field"] for evidence in details["source_cap_status"]["stale_cap_evidence"]
    }
    assert expected_field in stale_fields


@pytest.mark.parametrize("command", ["runner --max-papers 180", "runner --max-papers=180"])
def test_v151_byte_reconciliation_blocks_cli_cap_spellings(
    tmp_path: Path,
    command: str,
) -> None:
    _seed_phase67_workspace(tmp_path, command=command)

    receipt = _run_phase67(tmp_path)

    details = _assert_phase67_contract(receipt)
    assert receipt["status"] == "blocked"
    stale_fields = {
        evidence["field"] for evidence in details["source_cap_status"]["stale_cap_evidence"]
    }
    assert "command.max_papers" in stale_fields


def test_v151_byte_reconciliation_blocks_prior_180_paper_openreview_cache(
    tmp_path: Path,
) -> None:
    _seed_phase67_workspace(tmp_path, openreview_paper_count=180)

    receipt = _run_phase67(tmp_path)

    details = _assert_phase67_contract(receipt)
    assert receipt["status"] == "blocked"
    assert details["stale_cache_status"]["openreview_papers_count"] == 180
    assert (
        details["stale_cache_status"]["prior_180_paper_cache_status"]
        == "blocked_stale_180"
    )
    assert details["stale_cache_status"]["quarantine_status"] == "blocked_not_reused"


def test_v151_byte_reconciliation_succeeds_with_cap_free_digest_verified_bytes(
    tmp_path: Path,
) -> None:
    _seed_phase67_workspace(tmp_path, use_digest_field_for_media=True)

    receipt = _run_phase67(tmp_path)

    details = _assert_phase67_contract(receipt)
    byte_details = details["byte_reconciliation"]
    assert receipt["status"] == "succeeded"
    assert details["outcome"] == "passed"
    assert details["outcome_reason_code"] == "cap_free_source_boundary_confirmed"
    assert details["source_cap_status"]["source_cap_free"] is True
    assert details["stale_cache_status"]["prior_180_paper_cache_status"] == "accepted_cap_free"
    assert byte_details["download_receipt_status"] == "accepted"
    assert byte_details["media_receipt_status"] == "accepted"
    assert byte_details["referenced_download_bytes_status"] == "accepted"
    assert byte_details["referenced_media_bytes_status"] == "accepted"
    assert byte_details["referenced_byte_digest_status"] == "accepted"
    assert byte_details["referenced_download_bytes_count"] == 1
    assert byte_details["referenced_media_bytes_count"] == 1
    assert byte_details["missing_referenced_byte_paths"] == []
    assert byte_details["digest_mismatches"] == []
    assert (
        byte_details["reusable_backup_bytes_decision"]
        == "accepted_existing_local_no_restore_needed"
    )
    assert details["budget_guardrail"]["budget_cap_usd"] == 1000
    assert details["budget_guardrail"]["budget_cap_role"] == "halt_and_approve_guardrail"
    assert details["budget_guardrail"]["budget_cap_preserved"] is True
    assert details["budget_guardrail"]["corpus_cap_workaround_used"] is False


def test_v151_byte_reconciliation_ignores_approval_metadata_path(
    tmp_path: Path,
) -> None:
    _seed_phase67_workspace(
        tmp_path,
        include_approval_metadata_path=True,
        use_digest_field_for_media=True,
    )

    receipt = _run_phase67(tmp_path)

    details = _assert_phase67_contract(receipt)
    byte_details = details["byte_reconciliation"]
    assert receipt["status"] == "succeeded"
    assert byte_details["download_receipt_status"] == "accepted"
    assert byte_details["media_receipt_status"] == "accepted"
    assert byte_details["referenced_download_bytes_count"] == 1
    assert byte_details["referenced_media_bytes_count"] == 1
    assert byte_details["missing_referenced_byte_paths"] == []


@pytest.mark.parametrize(
    ("seed_kwargs", "reason_code"),
    [
        ({"missing_download_byte": True}, "required_download_bytes_missing"),
        ({"missing_media_byte": True}, "required_media_bytes_missing"),
    ],
)
def test_v151_byte_reconciliation_blocks_missing_referenced_local_bytes(
    tmp_path: Path,
    seed_kwargs: dict[str, bool],
    reason_code: str,
) -> None:
    _seed_phase67_workspace(tmp_path, **seed_kwargs)

    receipt = _run_phase67(tmp_path)

    details = _assert_phase67_contract(receipt)
    assert receipt["status"] == "blocked"
    assert reason_code in {blocker["reason_code"] for blocker in receipt["blockers"]}
    assert details["byte_reconciliation"]["missing_referenced_byte_paths"]


@pytest.mark.parametrize(
    ("seed_kwargs", "reason_code"),
    [
        ({"mismatched_download_digest": True}, "download_byte_digest_mismatch"),
        ({"mismatched_media_digest": True}, "media_byte_digest_mismatch"),
    ],
)
def test_v151_byte_reconciliation_blocks_digest_mismatches(
    tmp_path: Path,
    seed_kwargs: dict[str, bool],
    reason_code: str,
) -> None:
    _seed_phase67_workspace(tmp_path, **seed_kwargs)

    receipt = _run_phase67(tmp_path)

    details = _assert_phase67_contract(receipt)
    assert receipt["status"] == "blocked"
    assert reason_code in {blocker["reason_code"] for blocker in receipt["blockers"]}
    assert details["byte_reconciliation"]["digest_mismatches"]


@pytest.mark.parametrize(
    ("seed_kwargs", "reason_code"),
    [
        ({"omit_download_receipt": True}, "required_download_receipts_missing"),
        ({"omit_media_receipt": True}, "required_media_receipts_missing"),
    ],
)
def test_v151_byte_reconciliation_blocks_missing_required_byte_receipts(
    tmp_path: Path,
    seed_kwargs: dict[str, bool],
    reason_code: str,
) -> None:
    _seed_phase67_workspace(tmp_path, **seed_kwargs)

    receipt = _run_phase67(tmp_path)

    _assert_phase67_contract(receipt)
    assert receipt["status"] == "blocked"
    assert reason_code in {blocker["reason_code"] for blocker in receipt["blockers"]}


@pytest.mark.parametrize(
    ("seed_kwargs", "reason_code", "receipt_status_key", "reference_status_key", "count_key"),
    [
        (
            {"empty_download_receipt": True},
            "required_download_bytes_missing",
            "download_receipt_status",
            "referenced_download_bytes_status",
            "referenced_download_bytes_count",
        ),
        (
            {"empty_media_receipt": True},
            "required_media_bytes_missing",
            "media_receipt_status",
            "referenced_media_bytes_status",
            "referenced_media_bytes_count",
        ),
    ],
)
def test_v151_byte_reconciliation_blocks_empty_required_byte_receipts(
    tmp_path: Path,
    seed_kwargs: dict[str, bool],
    reason_code: str,
    receipt_status_key: str,
    reference_status_key: str,
    count_key: str,
) -> None:
    _seed_phase67_workspace(tmp_path, **seed_kwargs)

    receipt = _run_phase67(tmp_path)

    details = _assert_phase67_contract(receipt)
    byte_details = details["byte_reconciliation"]
    assert receipt["status"] == "blocked"
    assert reason_code in {blocker["reason_code"] for blocker in receipt["blockers"]}
    assert byte_details[receipt_status_key] == "blocked_no_referenced_bytes"
    assert byte_details[reference_status_key] == "blocked_no_referenced_bytes"
    assert byte_details["referenced_byte_digest_status"] == "blocked_no_referenced_bytes"
    assert byte_details[count_key] == 0
    assert byte_details["missing_referenced_byte_paths"] == []


@pytest.mark.repo_data("conferences/iclr2025/config.yaml")
def test_v151_byte_reconciliation_preserves_real_budget_cap_guardrail() -> None:
    config = yaml.safe_load(Path("conferences/iclr2025/config.yaml").read_text())

    assert config["workflow"]["provider_cost"]["budget_cap_usd"] == 1000
