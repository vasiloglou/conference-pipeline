"""Tests for incremental 07D/07H receipt writes (fix option 5).

A mid-run kill of topic reconciliation must leave 07D (and the 07H reference
workflow) receipts that reflect true partial progress instead of a stale
all-blocked snapshot from a prior attempt.
"""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.analysis import topic_reconciliation as reconciliation_module
from conference_pipeline.analysis import (
    topic_reconciliation_reference_workflow as reference_workflow_module,
)
from conference_pipeline.analysis.topic_reconciliation import (
    DEFAULT_QWEN36_MODEL,
    run_topic_reconciliation,
)
from conference_pipeline.analysis.topic_reconciliation_reference_workflow import (
    run_topic_reconciliation_reference_workflow,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True))


def _seed_topic(workspace: Path, topic_id: str) -> None:
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json",
        {
            "schema_version": 1,
            "run_id": "test-run",
            "topic_id": topic_id,
            "title": f"Title for {topic_id}",
            "status": "deep-narrative-v1",
            "research_id": f"test-run:{topic_id}:abc",
            "research_path": f"analysis/provenance/query_plans/{topic_id}.json",
            "finding_text": "Evidence spans papers, talks, and deep dives.",
            "chunks": [
                {
                    "chunk_id": f"{topic_id}-chunk-1",
                    "extraction_record_id": f"{topic_id}-record-1",
                    "source_artifact_path": "storage/source.txt",
                    "quote": "Some evidence quote.",
                },
            ],
            "llm_full_read_evidence": {
                "status": "succeeded",
                "unit_count": 1,
                "input_kind_counts": {"transcript_segment": 1},
                "selected_items": [{"unit_id": f"{topic_id}-u1", "summary": "Talk summary."}],
            },
            "matched_papers": [],
            "extraction_record_ids": [f"{topic_id}-record-1", f"{topic_id}-u1"],
        },
    )
    _write(
        workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__papers.md",
        "# Papers\n\nSome evidence.\n",
    )
    _write(
        workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__transcripts.md",
        "# Transcripts\n\nSome talk evidence.\n",
    )


def _workspace_with_topics(tmp_path: Path, topic_ids: list[str]) -> Path:
    workspace = tmp_path / "workspace"
    for topic_id in topic_ids:
        _seed_topic(workspace, topic_id)
    return workspace


def _receipt_path(workspace: Path, run_id: str) -> Path:
    return workspace / "storage" / "workflow-runs" / run_id / "07D-topic-reconciliation.json"


def test_incremental_receipt_reflects_partial_progress_after_simulated_kill(
    tmp_path: Path, monkeypatch
) -> None:
    topic_ids = ["01_topic", "02_topic", "03_topic"]
    workspace = _workspace_with_topics(tmp_path, topic_ids)
    run_id = "test-run"

    real_write_json_atomic = reconciliation_module.write_json_atomic
    receipt_path = _receipt_path(workspace, run_id)
    call_count = {"interim_receipt_writes": 0}

    def _flaky_write_json_atomic(path, payload):
        if path == receipt_path:
            call_count["interim_receipt_writes"] += 1
            if call_count["interim_receipt_writes"] == 3:
                raise RuntimeError("simulated-kill-before-third-interim-write")
        real_write_json_atomic(path, payload)

    monkeypatch.setattr(reconciliation_module, "write_json_atomic", _flaky_write_json_atomic)

    try:
        run_topic_reconciliation(
            workspace=workspace,
            run_id=run_id,
            mode="dry-run",
            topics=topic_ids,
        )
        raised = False
    except RuntimeError as exc:
        raised = "simulated-kill" in str(exc)

    assert raised is True
    assert receipt_path.exists()
    on_disk = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert on_disk["completed_topic_count"] == 2
    assert on_disk["status"] == "in_progress"
    assert len(on_disk["output_paths"]) == 2


def test_clean_full_run_final_receipt_shape_unchanged(tmp_path: Path) -> None:
    topic_ids = ["01_topic", "02_topic"]
    workspace = _workspace_with_topics(tmp_path, topic_ids)
    run_id = "test-run"

    receipt = run_topic_reconciliation(
        workspace=workspace,
        run_id=run_id,
        mode="dry-run",
        topics=topic_ids,
    )

    assert receipt["status"] == "succeeded"
    assert receipt["completed_topic_count"] == 2
    assert receipt["topic_count"] == 2
    assert len(receipt["output_paths"]) == 2
    assert receipt["blockers"] == []

    on_disk = json.loads(_receipt_path(workspace, run_id).read_text(encoding="utf-8"))
    assert on_disk == receipt


def test_interim_reference_workflow_receipt_reflects_reconciliation_progress(
    tmp_path: Path, monkeypatch
) -> None:
    topic_ids = ["01_topic", "02_topic"]
    workspace = _workspace_with_topics(tmp_path, topic_ids)
    run_id = "test-run"
    receipt_07h_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "07H-topic-reconciliation-reference-workflow.json"
    )

    real_run_topic_reference_validation = (
        reference_workflow_module.run_topic_reference_validation
    )

    def _raising_validation(*args, **kwargs):
        raise RuntimeError("simulated-kill-before-validation-chain")

    monkeypatch.setattr(
        reference_workflow_module, "run_topic_reference_validation", _raising_validation
    )

    try:
        run_topic_reconciliation_reference_workflow(
            workspace=workspace,
            run_id=run_id,
            mode="dry-run",
            topics=topic_ids,
            run_reconciliation=True,
        )
        raised = False
    except RuntimeError as exc:
        raised = "simulated-kill" in str(exc)

    assert raised is True
    assert receipt_07h_path.exists()
    on_disk = json.loads(receipt_07h_path.read_text(encoding="utf-8"))
    assert on_disk["checks"]["final_validation_status"] == "in_progress"
    assert on_disk["completed_topic_count"] == 2
    assert on_disk["status"] == "in_progress"
