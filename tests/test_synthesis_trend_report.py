"""Unit tests for Phase 19A executive + trend synthesis stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


_TOPICS: tuple[str, ...] = (
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
)



def _alias_model(alias: str) -> str:
    """The catalog's current model for an alias, so model upgrades move these tests with them."""
    from conference_pipeline.model_governance.registry import load_model_catalog

    return load_model_catalog()[alias]["model"]

def _seed_phase_19a_inputs(
    workspace_root: Path,
    *,
    run_id: str,
    topic_report_body: str | None = None,
    extraction_count: int = 12,
    chunk_count: int = 8,
    research_count: int = 1,
    write_phase_16a_receipt: bool = True,
    write_coverage_gap_report: bool = True,
    write_topic_reports: bool = True,
) -> dict[str, str]:
    """Seed the inputs needed by run_synthesis_trend_report_stage on tmp_path.

    Returns a dict with the bound digests so tests can assemble synthesis
    bodies that satisfy citation floors.
    """
    analysis_dir = workspace_root / "analysis"
    qa_dir = analysis_dir / "qa"
    runs_dir = workspace_root / "storage" / "workflow-runs" / run_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Citation manifest.
    citation_topics = []
    for tid in _TOPICS:
        citation_topics.append(
            {
                "topic_id": tid,
                "extraction_record_ids": [
                    f"media:oral-{30000 + i}:redacted.m4a"
                    for i in range(extraction_count)
                ],
                "vector_store_chunk_ids": [
                    f"{tid}:chunk:{i}" for i in range(chunk_count)
                ],
                "research_ids": [f"research:{tid}" for _ in range(research_count)],
                "markdown_path": f"analysis/{tid}.md",
            }
        )
    citation_manifest = {
        "evidence_chain_version": "v1.5",
        "generated_at": "2026-04-27T00:00:00Z",
        "run_id": run_id,
        "schema_version": 1,
        "topics": citation_topics,
    }
    (analysis_dir / "analysis_citation_manifest.json").write_text(
        json.dumps(citation_manifest, indent=2), encoding="utf-8"
    )

    # Analysis manifest.
    analysis_manifest_sha = "deadbeef" * 8
    analysis_manifest = {
        "schema_version": 1,
        "content_sha256": analysis_manifest_sha,
        "generated_at": "2026-04-27T00:00:00Z",
        "topics": [
            {
                "id": tid,
                "markdown_path": f"analysis/{tid}.md",
                "citation_counts": {
                    "extraction_record_ids": extraction_count,
                    "vector_store_chunk_ids": chunk_count,
                    "research_ids": research_count,
                },
            }
            for tid in _TOPICS
        ],
    }
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2), encoding="utf-8"
    )

    # Coverage gap report.
    coverage_gap_report = {
        "schema_version": 1,
        "generated_at": "2026-04-27T00:00:00Z",
        "run_id": run_id,
        "status": "pass",
        "thresholds": {
            "extraction_min_per_topic": 8,
            "chunk_min_per_topic": 5,
            "research_min_per_topic": 1,
            "min_well_covered_topics": 10,
        },
        "classes": {
            "well_covered": 10,
            "under_covered": 0,
            "missing_evidence": 0,
            "gap": 0,
        },
        "topics": [],
        "blockers": [],
        "bound_extraction_receipt_digest": "abc",
        "bound_vector_store_receipt_digest": "def",
        "bound_query_research_receipt_digest": "ghi",
        "bound_analysis_manifest_content_sha256": analysis_manifest_sha,
    }
    coverage_gap_path = qa_dir / "coverage_gap_report.json"
    if write_coverage_gap_report:
        coverage_gap_path.write_text(
            json.dumps(coverage_gap_report, indent=2), encoding="utf-8"
        )

    # Topic reports.
    if write_topic_reports:
        body = (
            topic_report_body
            if topic_report_body is not None
            else "# Topic\n\nFixture body.\n"
        )
        for tid in _TOPICS:
            (analysis_dir / f"{tid}.md").write_text(body, encoding="utf-8")

    # Phase 16A receipt.
    phase_16a_receipt_path = (
        runs_dir / "16A-coverage-gap-fill-receipt.json"
    )
    if write_phase_16a_receipt:
        phase_16a_receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage_id": "coverage-gap-fill",
                    "run_id": run_id,
                    "phase": 16,
                    "status": "succeeded",
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

    # Compute the coverage gap report sha256 the way the runner will.
    coverage_gap_sha = hashlib.sha256(
        coverage_gap_path.read_bytes()
        if coverage_gap_path.exists()
        else b""
    ).hexdigest()

    return {
        "analysis_manifest_content_sha256": analysis_manifest_sha,
        "coverage_gap_report_sha256": coverage_gap_sha,
    }


def _build_synthesis_body(
    *,
    digests: dict[str, str],
    kind: str,
    extraction_id_count: int = 6,
    research_id_count: int = 4,
    topic_count: int = 5,
    pad_chars: int = 4000,
) -> str:
    """Compose a synthesis body that satisfies the citation floors."""
    extraction_ids = [
        f"media:oral-{30000 + i}:redacted.m4a"
        for i in range(extraction_id_count)
    ]
    research_ids = [f"research:{_TOPICS[i]}" for i in range(research_id_count)]
    topic_paths = [f"analysis/{_TOPICS[i]}.md" for i in range(topic_count)]

    pad = "Filler narrative. " * (pad_chars // 18)
    body_lines = [
        f"# {kind.replace('_', ' ').title()}\n",
        "## Cited topic reports\n",
        *[f"- {p}\n" for p in topic_paths],
        "## Cited extraction record ids\n",
        *[f"- `{record}`\n" for record in extraction_ids],
        "## Cited research ids\n",
        *[f"- `{record}`\n" for record in research_ids],
        "## Body\n",
        pad,
        "\n",
        "<!-- synthesis-bound: "
        f"analysis_manifest_content_sha256={digests['analysis_manifest_content_sha256']}; "
        f"coverage_gap_report_sha256={digests['coverage_gap_report_sha256']}; "
        f"kind={kind} -->\n",
    ]
    return "\n".join(body_lines)


def test_run_synthesis_trend_report_stage_happy_path(tmp_path):
    from conference_pipeline.iclr2025_live import (
        FakeSynthesisClient,
        run_synthesis_trend_report_stage,
    )

    run_id = "test-syn-1"
    digests = _seed_phase_19a_inputs(tmp_path, run_id=run_id)
    client = FakeSynthesisClient(
        executive_summary_text=_build_synthesis_body(
            digests=digests, kind="executive_summary"
        ),
        trend_report_text=_build_synthesis_body(
            digests=digests, kind="trend_report", pad_chars=6000
        ),
        input_tokens=80000,
        output_tokens=10000,
    )
    receipt = run_synthesis_trend_report_stage(
        tmp_path, run_id=run_id, llm_client=client
    )
    assert receipt["status"] == "succeeded", receipt
    details = receipt["details"]
    assert details["bound_analysis_manifest_content_sha256"] == digests[
        "analysis_manifest_content_sha256"
    ]
    assert details["bound_coverage_gap_report_sha256"] == digests[
        "coverage_gap_report_sha256"
    ]
    assert details["provider"] == "anthropic"
    assert details["model"] == _alias_model("opus")
    assert details["cost_cap_exceeded"] is False
    assert details["estimated_cost_usd"] >= 0.0
    counts_es = details["executive_summary_citation_counts"]
    counts_tr = details["trend_report_citation_counts"]
    assert counts_es["topic_report_paths"] >= 3
    assert counts_es["extraction_record_ids"] >= 5
    assert counts_es["research_ids"] >= 3
    assert counts_tr["topic_report_paths"] >= 3
    assert counts_tr["extraction_record_ids"] >= 5
    assert counts_tr["research_ids"] >= 3
    assert (tmp_path / "analysis" / "executive_summary.md").exists()
    assert (tmp_path / "analysis" / "trend_report.md").exists()


def test_run_synthesis_trend_report_stage_blocked_when_inputs_missing(tmp_path):
    from conference_pipeline.iclr2025_live import (
        run_synthesis_trend_report_stage,
    )

    receipt = run_synthesis_trend_report_stage(tmp_path, run_id="missing-run")
    assert receipt["status"] == "blocked"
    codes = {b.get("code") for b in receipt["blockers"]}
    assert "synthesis_input_missing" in codes


def test_run_synthesis_trend_report_stage_blocks_on_citation_floor(tmp_path):
    from conference_pipeline.iclr2025_live import (
        FakeSynthesisClient,
        run_synthesis_trend_report_stage,
    )

    run_id = "test-syn-floor"
    digests = _seed_phase_19a_inputs(tmp_path, run_id=run_id)
    # Body that omits citations entirely; synthesis must fail closed.
    weak_body = (
        "# Executive Summary\n\n"
        "Synthesis without citations.\n" * 200
    )
    client = FakeSynthesisClient(
        executive_summary_text=weak_body,
        trend_report_text=weak_body,
    )
    receipt = run_synthesis_trend_report_stage(
        tmp_path, run_id=run_id, llm_client=client
    )
    assert receipt["status"] == "blocked", receipt
    codes = {b.get("code") for b in receipt["blockers"]}
    assert "synthesis_citation_floor_unmet" in codes


def test_run_synthesis_trend_report_stage_blocks_when_cost_cap_exceeded(
    tmp_path,
):
    from conference_pipeline.iclr2025_live import (
        FakeSynthesisClient,
        run_synthesis_trend_report_stage,
    )

    run_id = "test-syn-cost"
    digests = _seed_phase_19a_inputs(tmp_path, run_id=run_id)
    # Force a token usage that estimates > $5 (needs ~10M input tokens at
    # the catalog rate). Use 200M input tokens to be safe.
    client = FakeSynthesisClient(
        executive_summary_text=_build_synthesis_body(
            digests=digests, kind="executive_summary"
        ),
        trend_report_text=_build_synthesis_body(
            digests=digests, kind="trend_report", pad_chars=6000
        ),
        input_tokens=200_000_000,
        output_tokens=10_000_000,
    )
    receipt = run_synthesis_trend_report_stage(
        tmp_path, run_id=run_id, llm_client=client
    )
    assert receipt["status"] == "blocked", receipt
    codes = {b.get("code") for b in receipt["blockers"]}
    assert "synthesis_cost_cap_exceeded" in codes
    assert receipt["details"]["cost_cap_exceeded"] is True


def test_run_synthesis_trend_report_stage_resume_on_digest_match(tmp_path):
    from conference_pipeline.iclr2025_live import (
        FakeSynthesisClient,
        run_synthesis_trend_report_stage,
    )

    run_id = "test-syn-resume"
    digests = _seed_phase_19a_inputs(tmp_path, run_id=run_id)
    client = FakeSynthesisClient(
        executive_summary_text=_build_synthesis_body(
            digests=digests, kind="executive_summary"
        ),
        trend_report_text=_build_synthesis_body(
            digests=digests, kind="trend_report", pad_chars=6000
        ),
        input_tokens=80000,
        output_tokens=10000,
    )
    receipt1 = run_synthesis_trend_report_stage(
        tmp_path, run_id=run_id, llm_client=client
    )
    assert receipt1["status"] == "succeeded"
    initial_request_count = client.request_count

    # Second call should resume without re-invoking the client.
    receipt2 = run_synthesis_trend_report_stage(
        tmp_path, run_id=run_id, llm_client=client
    )
    assert receipt2["status"] == "succeeded"
    assert receipt2["details"].get("resumed") is True
    assert client.request_count == initial_request_count


def test_python_stage_dispatch_branch_registered():
    """Anticipate the 14B/14C/16A dispatch miss: synthesis-trend-report wired."""
    import inspect

    from conference_pipeline import workflow

    src = inspect.getsource(workflow.run_stage)
    assert 'stage.id == "synthesis-trend-report"' in src
    assert "run_synthesis_trend_report_stage" in src


def test_stage_registered_in_workflow():
    from conference_pipeline import workflow

    assert "synthesis-trend-report" in {s.id for s in workflow.WORKFLOW_STAGES}
    assert workflow.is_synthesis_stage_id("synthesis-trend-report")
    assert (
        workflow.WORKFLOW_STAGE_BY_ID["synthesis-trend-report"].dispatch_kind
        == "python_stage"
    )


def test_stage_files_row():
    from conference_pipeline.iclr2025_live import STAGE_FILES

    assert STAGE_FILES["synthesis-trend-report"] == (
        19,
        "19A-synthesis-trend-report-receipt.json",
    )
