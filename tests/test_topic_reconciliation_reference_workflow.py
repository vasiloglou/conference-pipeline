from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.analysis.topic_reconciliation_reference_workflow import (
    run_topic_reconciliation_reference_workflow,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True))


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    topic_id = "01_reasoning_post_training"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json",
        {
            "schema_version": 1,
            "topic_id": topic_id,
            "title": "Reasoning and post-training",
            "topic_reconciliation": {
                "status": "succeeded",
                "markdown_path": "analysis/topic_reconciliations/01_reasoning_post_training.md",
            },
            "chunks": [
                {
                    "chunk_id": "speech_transcript:abc123:segment:00001",
                    "talk_title": "Reasoning Cache",
                }
            ],
            "llm_full_read_evidence": {
                "selected_items": [
                    {
                        "unit_id": "paper_abstract:2jkAk3EP0v",
                        "input_kind": "paper_abstract",
                        "paper_title": "Latent Thinking Optimization",
                    }
                ]
            },
            "matched_papers": [],
            "extraction_record_ids": [
                "paper_abstract:2jkAk3EP0v",
                "speech_transcript:abc123:segment:00001",
            ],
        },
    )
    _write_json(
        workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__papers.json",
        {
            "topic_id": topic_id,
            "notable_evidence": [
                {
                    "claim": "Latent Thinking Optimization uses latent optimization.",
                    "source_labels": ["Latent Thinking Optimization"],
                    "unit_ids": ["paper_abstract:2jkAk3EP0v"],
                }
            ],
        },
    )
    _write(
        workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md",
        "# Topic Evidence Reconciliation\n\n"
        "- Presentation ID: 2jkAk3EP0v\n"
        "- Timestamp: abc123:segment:00001\n",
    )
    return workspace


def test_reference_workflow_packages_validation_and_deterministic_repair(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)

    receipt = run_topic_reconciliation_reference_workflow(
        workspace=workspace,
        run_id="test-run",
        mode="dry-run",
        topics=["01_reasoning_post_training"],
        run_reconciliation=False,
    )

    assert receipt["status"] == "succeeded"
    assert receipt["checks"]["final_invalid_reference_count"] == 0
    assert receipt["checks"]["final_bare_reference_warning_count"] == 0
    assert receipt["checks"]["deterministic_safety_fix_step_ran"] is True
    assert (
        workspace
        / "storage"
        / "workflow-runs"
        / "test-run"
        / "07H-topic-reconciliation-reference-workflow.json"
    ).exists()
    repaired = (
        workspace / "analysis" / "topic_reconciliations" / "01_reasoning_post_training.md"
    ).read_text(encoding="utf-8")
    assert "`paper_abstract:2jkAk3EP0v`" in repaired
    assert "speech_transcript:abc123:segment:00001" in repaired


def test_unledgered_strip_runs_after_repair_not_before_it(tmp_path: Path) -> None:
    """The ledger guard is a backstop, never a pre-emption.

    Placing it at 07D write time (an earlier attempt) deleted every malformed
    citation before 07F/07G could canonicalize it, so validation passed
    immediately, the repair steps never ran at all, and a live CVPR 2026 run
    lost 38 recoverable citations across 10 topics -- citation_count fell
    307 -> 268 while broad_repair_step_ran/candidate_resolution_step_ran both
    flipped to False. Repair must always get first refusal.
    """

    import conference_pipeline.analysis.topic_reconciliation_reference_workflow as wf

    workspace = _workspace(tmp_path)
    call_order: list[str] = []

    real_repair = wf.run_topic_reference_repair
    real_strip = wf.strip_unledgered_citations

    def tracking_repair(**kwargs: object) -> object:
        call_order.append("repair")
        return real_repair(**kwargs)

    def tracking_strip(text: str, **kwargs: object) -> object:
        call_order.append("strip")
        return real_strip(text, **kwargs)

    wf.run_topic_reference_repair = tracking_repair  # type: ignore[assignment]
    wf.strip_unledgered_citations = tracking_strip  # type: ignore[assignment]
    try:
        receipt = run_topic_reconciliation_reference_workflow(
            workspace=workspace,
            run_id="test-run",
            mode="dry-run",
            topics=["01_reasoning_post_training"],
            run_reconciliation=False,
        )
    finally:
        wf.run_topic_reference_repair = real_repair  # type: ignore[assignment]
        wf.strip_unledgered_citations = real_strip  # type: ignore[assignment]

    # This fixture validates clean, so the guard must stay out of it entirely:
    # a backstop that fires on already-passing output would delete good
    # citations for no reason.
    assert "strip" not in call_order
    assert receipt["checks"]["unledgered_citation_removal_count"] == 0
    assert receipt["unledgered_citations_removed"] == []
    # And the citations repair is responsible for are still present.
    repaired = (
        workspace / "analysis" / "topic_reconciliations" / "01_reasoning_post_training.md"
    ).read_text(encoding="utf-8")
    assert "`paper_abstract:2jkAk3EP0v`" in repaired


def test_shorthand_ids_are_canonicalized_after_the_unledgered_strip(tmp_path: Path) -> None:
    """Live CVPR 2026 topic 08: a list item rested on a fabricated id and a
    prose line cited a real paper as backticked shorthand without its
    ``paper_abstract:`` prefix. 07F's deterministic canonicalization was
    discarded because the topic still failed on the fabrication, the strip
    then removed the fabrication last, and the shorthand survived to block
    final validation as a bare id. After the strip, canonicalize again.
    """

    import conference_pipeline.analysis.topic_reconciliation_reference_workflow as wf

    workspace = _workspace(tmp_path)
    _write(
        workspace / "analysis" / "topic_reconciliations" / "01_reasoning_post_training.md",
        "# Topic Evidence Reconciliation\n\n"
        # A foreign-prefix id the deterministic pass cannot rewrite, so the
        # topic stays blocked through 07F and its canonicalization is discarded.
        "*   `proceedings_paper:cvf_findings:cvf_13227e5cd6bfedd70d0d` (Invented): unsupported.\n"
        "*   Papers (`2jkAk3EP0v`) and talks (`speech_transcript:abc123:segment:00001`) agree.\n",
    )

    def blocked_receipt(**kwargs: object) -> dict[str, object]:
        return {"status": "blocked", "blockers": [], "checks": {}}

    real_resolve = wf.resolve_topic_reference_candidates
    real_repair = wf.run_topic_reference_repair
    wf.resolve_topic_reference_candidates = blocked_receipt  # type: ignore[assignment]

    def repair(**kwargs: object) -> object:
        # Keep the real deterministic (dry-run) pass; stub the paid broad repair.
        if kwargs.get("mode") == "dry-run":
            return real_repair(**kwargs)
        return blocked_receipt()

    wf.run_topic_reference_repair = repair  # type: ignore[assignment]
    try:
        receipt = run_topic_reconciliation_reference_workflow(
            workspace=workspace,
            run_id="test-run",
            mode="live",
            topics=["01_reasoning_post_training"],
            run_reconciliation=False,
        )
    finally:
        wf.resolve_topic_reference_candidates = real_resolve  # type: ignore[assignment]
        wf.run_topic_reference_repair = real_repair  # type: ignore[assignment]

    assert receipt["status"] == "succeeded", receipt["blockers"]
    assert receipt["checks"]["final_bare_reference_warning_count"] == 0
    assert receipt["checks"]["final_unknown_reference_count"] == 0
    text = (
        workspace / "analysis" / "topic_reconciliations" / "01_reasoning_post_training.md"
    ).read_text(encoding="utf-8")
    assert "cvf_13227e5cd6bfedd70d0d" not in text
    assert "`paper_abstract:2jkAk3EP0v`" in text
    assert any(
        entry.get("action") == "deterministic_safety_fixes_after_strip"
        for entry in receipt["unledgered_citations_removed"]
    )
