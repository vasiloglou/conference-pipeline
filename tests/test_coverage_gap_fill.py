"""Unit tests for Phase 16A coverage analysis + gap fill stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _seed_minimal_inputs(
    workspace_root: Path,
    *,
    run_id: str,
    extraction_count: int = 12,
    chunk_count: int = 8,
    research_count: int = 1,
    media_evidence_count: int = 8,
    paper_only_marker_count: int = 0,
    include_final_report: bool = True,
    include_qa_report: bool = True,
    research_classes: dict | None = None,
    extraction_id_prefix: str = "media",
) -> None:
    """Seed the minimal inputs for run_coverage_gap_fill_stage on a tmp workspace.

    Produces 10 production topics by default with all-evidence-present
    citation manifest entries + matching analysis manifest entries + a
    Phase 16 v15_coverage_report.json that records media-evidence ref counts.
    """
    topics = [
        "02_model_architectures",
        "03_training_optimization",
        "04_representation_learning",
        "05_agents_reasoning",
        "06_multimodal_systems",
        "07_evaluation_benchmarks",
        "08_efficiency_scaling",
        "09_safety_alignment",
        "10_scientific_applications",
        "11_datasets_tooling",
    ]

    # Citation manifest.
    citation_topics = []
    for tid in topics:
        if extraction_id_prefix == "media":
            extraction_ids = [
                f"media:oral-{30000 + i}:redacted.m4a"
                for i in range(extraction_count)
            ]
        elif extraction_id_prefix == "ocr":
            extraction_ids = [
                f"media:oral-{30000 + i}:redacted.png"
                for i in range(extraction_count)
            ]
        elif extraction_id_prefix == "paper":
            extraction_ids = [f"paper:{tid}:p{i}" for i in range(extraction_count)]
        else:
            extraction_ids = [f"unknown:{tid}:{i}" for i in range(extraction_count)]

        citation_topics.append(
            {
                "topic_id": tid,
                "extraction_record_ids": extraction_ids,
                "vector_store_chunk_ids": [
                    f"{tid}:chunk:{i}" for i in range(chunk_count)
                ],
                "research_ids": [f"research:{tid}" for _ in range(research_count)],
                "markdown_path": f"analysis/{tid}.md",
            }
        )
    citation_manifest_path = (
        workspace_root / "analysis" / "analysis_citation_manifest.json"
    )
    citation_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    citation_manifest_path.write_text(
        json.dumps(
            {
                "evidence_chain_version": "v1.5",
                "generated_at": "2026-04-27T00:00:00Z",
                "run_id": run_id,
                "schema_version": 1,
                "topics": citation_topics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Analysis manifest.
    analysis_manifest = {
        "schema_version": 1,
        "content_sha256": "deadbeef" * 8,
        "generated_at": "2026-04-27T00:00:00Z",
        "topics": (
            [
                {
                    "id": tid,
                    "markdown_path": f"analysis/{tid}.md",
                    "citation_counts": {
                        "extraction_record_ids": extraction_count,
                        "vector_store_chunk_ids": chunk_count,
                        "research_ids": research_count,
                    },
                }
                for tid in topics
            ]
            if include_final_report
            else []
        ),
    }
    (workspace_root / "analysis" / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2), encoding="utf-8"
    )

    # Phase 14A coverage.
    extraction_dir = workspace_root / "storage" / "extractions" / run_id
    extraction_dir.mkdir(parents=True, exist_ok=True)
    (extraction_dir / "v15_extraction_coverage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "expected_count": 2979,
                "coverage_ratio": 0.985,
                "thresholds": {"not_attempted_max": 0},
                "classes": {
                    "blocked": 40,
                    "covered": 2934,
                    "not_attempted": 0,
                    "skipped_unusable": 4,
                    "unclassifiable": 1,
                },
                "by_kind_counts": {},
                "generated_at": "2026-04-27T00:00:00Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Phase 14B coverage.
    vstore_dir = workspace_root / "storage" / "vector-store" / run_id
    vstore_dir.mkdir(parents=True, exist_ok=True)
    (vstore_dir / "v15_index_coverage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "expected_count": 2979,
                "coverage_ratio": 0.985,
                "thresholds": {"not_attempted_max": 0},
                "classes": {
                    "blocked": 40,
                    "indexed": 2934,
                    "not_attempted": 0,
                    "skipped_unusable": 5,
                },
                "by_topic_counts": {tid: chunk_count for tid in topics},
                "extraction_receipt_digest": "abc",
                "collection_name": "v15_iclr2025_production_extractions",
                "generated_at": "2026-04-27T00:00:00Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Phase 14C coverage.
    research_dir = workspace_root / "storage" / "research" / run_id
    research_dir.mkdir(parents=True, exist_ok=True)
    (research_dir / "v15_research_coverage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "expected_count": 10,
                "coverage_ratio": 1.0,
                "thresholds": {"not_attempted_max": 0},
                "classes": research_classes
                or {
                    "covered": 0,
                    "failed": 0,
                    "not_attempted": 0,
                    "skipped": 0,
                    "thin": 10,
                },
                "by_topic_counts": {
                    tid: {
                        "citations": 15,
                        "local_hits": 0,
                    }
                    for tid in topics
                },
                "bound_extraction_receipt_digest": "abc",
                "bound_vector_store_receipt_digest": "def",
                "generated_at": "2026-04-27T00:00:00Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Phase 16 QA coverage report.
    qa_dir = workspace_root / "analysis" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    if include_qa_report:
        reports = [
            {
                "exists": True,
                "media_evidence_reference_count": media_evidence_count,
                "paper_only_marker_count": paper_only_marker_count,
                "paper_text_reference_count": 0,
                "path": f"analysis/{tid}.md",
                "stale_markers": [],
            }
            for tid in topics
        ]
        (qa_dir / "v15_coverage_report.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generated_at": "2026-04-27T00:00:00Z",
                    "status": "pass",
                    "topic_count": 10,
                    "blockers": [],
                    "topics": [],
                    "analysis_media_evidence_gate": {
                        "blockers": [],
                        "report_count": 10,
                        "reports": reports,
                        "media_artifact_summary": {
                            "completed_media_artifact_count": 2939,
                            "media_artifact_count": 2939,
                            "speech_transcript_count": 752,
                            "visual_or_text_media_count": 2187,
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Phase 14A/14B/14C stage receipts (for digest binding).
    runs_dir = workspace_root / "storage" / "workflow-runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    for filename, status in (
        ("14A-exhaustive-topic-extraction-receipt.json", "succeeded"),
        ("14B-enhanced-indexing-and-vector-store-receipt.json", "succeeded"),
        ("14C-query-planning-gpt-researcher-receipt.json", "succeeded"),
    ):
        (runs_dir / filename).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage_id": filename.replace("-receipt.json", "").replace(
                        "14A-", ""
                    ),
                    "run_id": run_id,
                    "status": status,
                    "phase": 14,
                    "command": "fixture",
                    "artifacts": [],
                    "blockers": [],
                    "details": {},
                    "secrets_copied": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def test_well_covered_when_full_evidence_chain(tmp_path):
    from conference_pipeline.iclr2025_live import run_coverage_gap_fill_stage

    run_id = "test-run-1"
    _seed_minimal_inputs(tmp_path, run_id=run_id)
    report = run_coverage_gap_fill_stage(tmp_path, run_id=run_id)

    assert report["status"] == "pass"
    assert report["classes"]["well_covered"] == 10
    assert report["classes"]["under_covered"] == 0
    assert report["classes"]["missing_evidence"] == 0
    assert report["classes"]["gap"] == 0
    assert len(report["topics"]) == 10
    # Receipt written.
    receipt = json.loads(
        (
            tmp_path
            / "storage"
            / "workflow-runs"
            / run_id
            / "16A-coverage-gap-fill-receipt.json"
        ).read_text()
    )
    assert receipt["status"] == "succeeded"
    assert receipt["secrets_copied"] is False
    # Report digest binding.
    assert report["bound_extraction_receipt_digest"] is not None


def test_missing_evidence_when_extraction_count_zero(tmp_path):
    from conference_pipeline.iclr2025_live import run_coverage_gap_fill_stage

    run_id = "test-run-2"
    _seed_minimal_inputs(tmp_path, run_id=run_id, extraction_count=0)
    report = run_coverage_gap_fill_stage(tmp_path, run_id=run_id)

    assert report["status"] == "fail"
    assert report["classes"]["missing_evidence"] >= 1
    # All blockers carry an actionable fix recommendation.
    for entry in report["blockers"]:
        assert entry.get("topic_id")
        assert entry.get("gap_class")
        assert entry.get("fix_recommendation")


def test_gap_when_final_report_missing(tmp_path):
    from conference_pipeline.iclr2025_live import run_coverage_gap_fill_stage

    run_id = "test-run-3"
    _seed_minimal_inputs(tmp_path, run_id=run_id, include_final_report=False)
    report = run_coverage_gap_fill_stage(tmp_path, run_id=run_id)

    assert report["status"] == "fail"
    assert report["classes"]["gap"] >= 1


def test_gap_when_paper_only_marker_present(tmp_path):
    from conference_pipeline.iclr2025_live import run_coverage_gap_fill_stage

    run_id = "test-run-3a"
    _seed_minimal_inputs(
        tmp_path,
        run_id=run_id,
        paper_only_marker_count=1,
    )
    report = run_coverage_gap_fill_stage(tmp_path, run_id=run_id)
    assert report["status"] == "fail"
    assert report["classes"]["gap"] == 10


def test_under_covered_when_below_threshold(tmp_path):
    from conference_pipeline.iclr2025_live import run_coverage_gap_fill_stage

    run_id = "test-run-4"
    _seed_minimal_inputs(
        tmp_path,
        run_id=run_id,
        extraction_count=4,  # below extraction_min_per_topic (8)
        chunk_count=8,
        research_count=1,
        media_evidence_count=8,
    )
    report = run_coverage_gap_fill_stage(tmp_path, run_id=run_id)

    # With media + research present, missing-evidence is not triggered;
    # under_covered fires due to extraction count below threshold.
    assert report["status"] == "fail"
    assert report["classes"]["under_covered"] == 10


def test_source_family_classification(tmp_path):
    from conference_pipeline.iclr2025_live import _classify_extraction_record_id

    assert (
        _classify_extraction_record_id("media:oral-31735:redacted.m4a")
        == "transcript"
    )
    assert (
        _classify_extraction_record_id("media:oral-31735:redacted.png")
        == "ocr"
    )
    assert (
        _classify_extraction_record_id("media:oral-31735:redacted.pdf")
        == "ocr"
    )
    assert _classify_extraction_record_id("paper:abc") == "paper_pdf"
    assert _classify_extraction_record_id("transcript:xyz") == "transcript"
    assert _classify_extraction_record_id("ocr:xyz") == "ocr"
    assert _classify_extraction_record_id("text:foo.transcript.json") == "transcript"
    assert _classify_extraction_record_id("text:foo.ocr.json") == "ocr"
    assert _classify_extraction_record_id("text:foo.paper.json") == "paper_pdf"
    assert _classify_extraction_record_id("unknown:x") == "other"


def test_resume_idempotent_on_digest_match(tmp_path):
    from conference_pipeline.iclr2025_live import run_coverage_gap_fill_stage

    run_id = "test-run-5"
    _seed_minimal_inputs(tmp_path, run_id=run_id)
    report1 = run_coverage_gap_fill_stage(tmp_path, run_id=run_id)
    report2 = run_coverage_gap_fill_stage(tmp_path, run_id=run_id)

    # Same bound digests + same content; on the second call, should resume.
    assert report1["status"] == "pass"
    assert report2["status"] == "pass"
    assert (
        report1["bound_extraction_receipt_digest"]
        == report2["bound_extraction_receipt_digest"]
    )
    receipt = json.loads(
        (
            tmp_path
            / "storage"
            / "workflow-runs"
            / run_id
            / "16A-coverage-gap-fill-receipt.json"
        ).read_text()
    )
    assert receipt["details"]["resumed"] is True


def test_force_rebuild_regenerates_report(tmp_path):
    from conference_pipeline.iclr2025_live import run_coverage_gap_fill_stage

    run_id = "test-run-6"
    _seed_minimal_inputs(tmp_path, run_id=run_id)
    report1 = run_coverage_gap_fill_stage(tmp_path, run_id=run_id)
    report2 = run_coverage_gap_fill_stage(
        tmp_path, run_id=run_id, force_rebuild=True
    )

    assert report1["status"] == report2["status"] == "pass"
    receipt = json.loads(
        (
            tmp_path
            / "storage"
            / "workflow-runs"
            / run_id
            / "16A-coverage-gap-fill-receipt.json"
        ).read_text()
    )
    assert receipt["details"]["resumed"] is False


def test_blocked_when_inputs_missing(tmp_path):
    from conference_pipeline.iclr2025_live import run_coverage_gap_fill_stage

    receipt = run_coverage_gap_fill_stage(tmp_path, run_id="empty-run")
    # The runner returns the stage receipt payload (status=blocked) when
    # no inputs are available.
    assert receipt["status"] == "blocked"
    assert receipt["blockers"]
    codes = {b.get("code") for b in receipt["blockers"]}
    assert "coverage_gap_fill_input_missing" in codes


def test_python_stage_dispatch_branch_registered():
    """Anticipate the 14B/14C miss: ensure coverage-gap-fill is dispatchable."""
    import inspect

    from conference_pipeline import workflow

    src = inspect.getsource(workflow.run_stage)
    assert 'stage.id == "coverage-gap-fill"' in src
    assert "run_coverage_gap_fill_stage" in src


def test_stage_registered_in_workflow():
    from conference_pipeline import workflow

    assert "coverage-gap-fill" in {s.id for s in workflow.WORKFLOW_STAGES}
    assert workflow.is_coverage_stage_id("coverage-gap-fill")
    assert workflow.WORKFLOW_STAGE_BY_ID["coverage-gap-fill"].dispatch_kind == "python_stage"


def test_stage_files_row():
    from conference_pipeline.iclr2025_live import STAGE_FILES

    assert STAGE_FILES["coverage-gap-fill"] == (
        16,
        "16A-coverage-gap-fill-receipt.json",
    )
