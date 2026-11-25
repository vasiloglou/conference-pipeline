"""Phase 25 structural-check unit tests."""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline import workflow
from conference_pipeline.presentation.artifact_truth import (
    HUMAN_REVIEW_NOT_APPROVED,
    validate_artifact_truth,
)
from conference_pipeline.presentation import validation as presentation_validation
from conference_pipeline.presentation.manifest import _artifact_sha256, _compute_content_sha256
from conference_pipeline.presentation.validation import run_validation_stage
from tests.presentation_validation.fixtures import (
    add_video_artifacts,
    add_zoom_companion,
    add_human_review_receipt,
    break_mapping_count,
    break_pdf,
    break_svg,
    make_validation_workspace,
    remove_presenter_slide_id,
    remove_generated_svgs,
    rewrite_pdf_page_count,
)


def _topic_dir(workspace):
    return workspace.root / "decks" / "01_demo_topic"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _set_density_targets(workspace, *, hyperlinks: float = 0.0, svgs: float = 0.0) -> None:
    manifest_path = workspace.root / "decks" / "presentation_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["presentations"][0]["target_hyperlinks_per_slide"] = hyperlinks
    data["presentations"][0]["target_svgs_per_slide"] = svgs
    for presentation in data["presentations"]:
        for artifact in presentation.get("source_analysis_artifacts", []):
            artifact["sha256"] = _artifact_sha256(
                workspace.root / artifact["path"],
                kind=artifact["kind"],
            )
        for state in presentation.get("stage_states", {}).values():
            for artifact in state.get("artifacts", []):
                artifact["sha256"] = _artifact_sha256(
                    workspace.root / artifact["path"],
                    kind=artifact["kind"],
                )
    data["content_sha256"] = _compute_content_sha256(data)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _records(tmp_path, stage_id):
    workspace = make_validation_workspace(tmp_path)
    return workspace, run_validation_stage(
        workspace,
        workflow.get_workflow_stage(stage_id),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )


def test_slide_mapping_schema_passes_on_valid_mapping(tmp_path):
    _, records = _records(tmp_path, "slide_consolidation_validation")
    assert any(record.check_id == "slide_mapping_schema" and record.status == "pass" for record in records)


def test_render_export_receipt_mode_does_not_require_human_bulk_rollout_approval(
    tmp_path: Path,
) -> None:
    # Render/export artifact truth writes artifact_truth_render_export_report.json;
    # separate frame checks own slide-count mismatch and stale frame hashes.
    deck_dir = tmp_path / "decks" / "04_safety_alignment_privacy"
    _write_json(
        deck_dir / "render_export_receipt.json",
        {
            "schema_version": 1,
            "status": "pass",
            "frame_manifest": {
                "path": "slide_frame_manifest.json",
                "sha256": "a" * 64,
            },
            "contact_sheet_path": str(deck_dir / "render_contact_sheet.png"),
            "legacy_markdown_paths": [],
        },
    )
    (deck_dir / "render_contact_sheet.png").write_bytes(b"contact-sheet")

    result = validate_artifact_truth(deck_dir, checks={"receipts"})

    assert result.status == "pass"
    assert HUMAN_REVIEW_NOT_APPROVED not in {finding.code for finding in result.findings}


def test_hyperlink_density_meets_configured_target(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    _set_density_targets(workspace, hyperlinks=2.0)
    links = "\n".join(
        f"- [Source {index}](https://example.com/{index})" for index in range(1, 7)
    )
    _write_text(
        _topic_dir(workspace) / "4.5_draft_slides_with_hyperlinks.md",
        f"## Slide 1: Opening\n\n{links}\n\n## Slide 2: Method\n\nB.\n\n## Slide 3: Close\n\nC.\n",
    )

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("research_enrichment_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "hyperlink_density_meets_target" and record.status == "pass"
        for record in records
    )


def test_hyperlink_density_detects_underlinked_deck(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    _set_density_targets(workspace, hyperlinks=2.0)
    _write_text(
        _topic_dir(workspace) / "4.5_draft_slides_with_hyperlinks.md",
        (
            "## Slide 1: Opening\n\n"
            "- [Only one](https://example.com/one)\n\n"
            "## Slide 2: Method\n\nB.\n\n"
            "## Slide 3: Close\n\nC.\n"
        ),
    )

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("research_enrichment_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "hyperlink_density_meets_target" and record.status == "fail"
        for record in records
    )


def test_svg_density_and_hyperlink_preservation_pass(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    linked_body = "\n".join(
        f"- [Source {index}](https://example.com/{index})" for index in range(1, 4)
    )
    _write_text(
        _topic_dir(workspace) / "4.5_draft_slides_with_hyperlinks.md",
        f"## Slide 1: Opening\n\n{linked_body}\n\n## Slide 2: Method\n\nB.\n\n## Slide 3: Close\n\nC.\n",
    )
    _write_text(
        _topic_dir(workspace) / "5_draft_slides_with_images.md",
        (
            "## Slide 1: Opening\n\n"
            f"{linked_body}\n\n"
            "![Figure one](assets/generated/slide_01.svg)\n\n"
            "## Slide 2: Method\n\n"
            "![Figure two](assets/generated/slide_02.svg)\n\n"
            "## Slide 3: Close\n\n"
            "![Figure three](assets/generated/slide_03.svg)\n"
        ),
    )
    _set_density_targets(workspace, hyperlinks=1.0, svgs=1.0)

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("figures_structural_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "hyperlink_preservation_4_5_to_5" and record.status == "pass"
        for record in records
    )
    assert any(
        record.check_id == "svg_density_meets_target" and record.status == "pass"
        for record in records
    )


def test_svg_density_excludes_title_slide(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    _write_text(
        _topic_dir(workspace) / "4.5_draft_slides_with_hyperlinks.md",
        "## Slide 1: Method\n\nB.\n",
    )
    _write_text(
        _topic_dir(workspace) / "5_draft_slides_with_images.md",
        (
            "## Slide 0: Demo Topic\n\n"
            "<!--\nSLIDE_TYPE: title\n-->\n\n"
            "## Slide 1: Method\n\n"
            "![Figure one](assets/generated/slide_01.svg)\n"
        ),
    )
    _set_density_targets(workspace, svgs=1.0)

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("figures_structural_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "svg_density_meets_target"
        and record.status == "pass"
        and "slides=1" in record.message
        for record in records
    )


def test_svg_density_detects_underillustrated_deck(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    _set_density_targets(workspace, svgs=1.0)

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("figures_structural_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "svg_density_meets_target" and record.status == "fail"
        for record in records
    )


def test_slide_mapping_markdown_count_matches(tmp_path):
    _, records = _records(tmp_path, "slide_consolidation_validation")
    assert any(
        record.check_id == "slide_mapping_markdown_count_equals_mapping"
        and record.status == "pass"
        for record in records
    )


def test_slide_mapping_markdown_count_detects_drift(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    break_mapping_count(workspace, new_count=2)
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("slide_consolidation_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )
    assert any(
        record.check_id == "slide_mapping_markdown_count_equals_mapping"
        and record.status == "fail"
        for record in records
    )


def test_slide_mapping_revealjs_section_count_matches(tmp_path):
    _, records = _records(tmp_path, "revealjs_render_validation")
    assert any(
        record.check_id == "slide_mapping_revealjs_section_count_equals_mapping"
        and record.status == "pass"
        for record in records
    )


def test_zoom_companion_does_not_inflate_mapping_count(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    add_zoom_companion(workspace, slide_num=2)
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("revealjs_render_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )
    assert any(
        record.check_id == "slide_mapping_revealjs_section_count_equals_mapping"
        and record.status == "pass"
        for record in records
    )


def test_svg_well_formed_on_valid_and_malformed(tmp_path):
    _, valid_records = _records(tmp_path, "figures_structural_validation")
    assert any(record.check_id == "svg_well_formed" and record.status == "pass" for record in valid_records)

    workspace = make_validation_workspace(tmp_path / "broken")
    break_svg(workspace)
    broken_records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("figures_structural_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )
    assert any(record.check_id == "svg_well_formed" and record.status == "fail" for record in broken_records)


def test_presenter_script_slide_ids_resolve(tmp_path):
    _, records = _records(tmp_path, "presenter_script_validation")
    assert any(record.check_id == "presenter_script_slide_ids_resolve" and record.status == "pass" for record in records)


def test_presenter_script_slide_ids_detect_missing_reference(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    remove_presenter_slide_id(workspace, slide_num=2)
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("presenter_script_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )
    assert any(
        record.check_id == "presenter_script_slide_ids_resolve"
        and record.status == "fail"
        for record in records
    )


def test_pdf_parses_and_page_count_matches_target_slides(tmp_path):
    _, records = _records(tmp_path, "pdf_export_validation")
    assert any(record.check_id == "pdf_parses" and record.status == "pass" for record in records)
    assert any(
        record.check_id == "pdf_page_count_matches_target_slides"
        and record.status == "pass"
        for record in records
    )


def test_pdf_page_count_matches_reveal_section_count_with_zoom_companions(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    add_zoom_companion(workspace, slide_num=2)
    rewrite_pdf_page_count(workspace, page_count=5)
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("pdf_export_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "pdf_page_count_matches_target_slides"
        and record.status == "pass"
        and "expected slide/page counts (3, 5)" in record.message
        for record in records
    )


def test_pdf_page_count_uses_largest_page_tree_count(tmp_path):
    pdf_path = tmp_path / "nested-page-tree.pdf"
    pdf_path.write_bytes(
        b"%PDF-1.7\n"
        b"1 0 obj << /Type /Pages /Count 8 >> endobj\n"
        b"2 0 obj << /Type /Pages /Count 32 >> endobj\n"
    )

    assert presentation_validation._pdf_page_count(pdf_path) == 32


def test_pdf_page_count_mismatch_warns_with_evidence(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    rewrite_pdf_page_count(workspace, page_count=2)
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("pdf_export_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        strict_required_artifacts=True,
    )

    assert any(
        record.check_id == "pdf_page_count_matches_target_slides"
        and record.status == "warn"
        and record.evidence_paths == ("decks/01_demo_topic/presentation.pdf",)
        for record in records
    )


def test_pdf_parses_detects_unreadable_pdf(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    break_pdf(workspace)
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("pdf_export_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )
    assert any(record.check_id == "pdf_parses" and record.status == "fail" for record in records)


def test_video_checks_skip_missing_video_files(tmp_path):
    _, records = _records(tmp_path, "video_generation_validation")
    assert {
        (record.check_id, record.status)
        for record in records
    } >= {
        ("video_artifacts_nonempty", "skipped_missing"),
        ("video_duration_readable", "skipped_missing"),
        ("video_stream_durations_match", "skipped_missing"),
        ("video_duration_matches_build_receipts", "skipped_missing"),
        ("video_tail_not_black", "skipped_missing"),
    }


def test_video_duration_checks_pass_when_streams_and_receipts_align(tmp_path, monkeypatch):
    workspace = make_validation_workspace(tmp_path)
    add_video_artifacts(workspace, duration_seconds=120.0)

    def fake_probe(path: Path):
        if path.suffix == ".m4a":
            return presentation_validation._MediaDurations(
                format_duration=120.0,
                audio_duration=120.0,
            )
        return presentation_validation._MediaDurations(
            format_duration=120.0,
            video_duration=120.0,
            audio_duration=120.1,
            video_frame_rate="25/1",
            video_frame_count=3000,
        )

    monkeypatch.setattr(presentation_validation, "_probe_media_durations", fake_probe)
    monkeypatch.setattr(presentation_validation, "_detect_black_tail_seconds", lambda *_args: 0.0)

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("video_generation_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "video_stream_durations_match" and record.status == "pass"
        for record in records
    )
    assert any(
        record.check_id == "video_duration_matches_build_receipts"
        and record.status == "pass"
        for record in records
    )
    assert any(
        record.check_id == "video_tail_not_black" and record.status == "pass"
        for record in records
    )


def test_video_duration_checks_fail_when_audio_outlives_video_stream(tmp_path, monkeypatch):
    workspace = make_validation_workspace(tmp_path)
    add_video_artifacts(workspace, duration_seconds=160.0)

    def fake_probe(path: Path):
        if path.suffix == ".m4a":
            return presentation_validation._MediaDurations(
                format_duration=160.0,
                audio_duration=160.0,
            )
        if path.name == "presentation_video_notebooklm.mp4":
            return presentation_validation._MediaDurations(
                format_duration=160.0,
                video_duration=100.0,
                audio_duration=160.0,
                video_frame_rate="25/1",
                video_frame_count=2500,
            )
        return presentation_validation._MediaDurations(
            format_duration=160.0,
            video_duration=160.0,
            audio_duration=160.0,
            video_frame_rate="25/1",
            video_frame_count=4000,
        )

    monkeypatch.setattr(presentation_validation, "_probe_media_durations", fake_probe)
    monkeypatch.setattr(presentation_validation, "_detect_black_tail_seconds", lambda *_args: 0.0)

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("video_generation_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "video_stream_durations_match"
        and record.status == "fail"
        and "presentation_video_notebooklm.mp4" in record.message
        and "delta" in record.message
        for record in records
    )


def test_video_duration_budget_fails_above_40_minutes(tmp_path, monkeypatch):
    workspace = make_validation_workspace(tmp_path)
    add_video_artifacts(workspace, duration_seconds=2501.0)

    def fake_probe(path: Path):
        if path.suffix == ".m4a":
            return presentation_validation._MediaDurations(
                format_duration=2501.0,
                audio_duration=2501.0,
            )
        return presentation_validation._MediaDurations(
            format_duration=2501.0,
            video_duration=2501.0,
            audio_duration=2501.0,
            video_frame_rate="25/1",
            video_frame_count=62525,
        )

    monkeypatch.setattr(presentation_validation, "_probe_media_durations", fake_probe)
    monkeypatch.setattr(presentation_validation, "_detect_black_tail_seconds", lambda *_args: 0.0)

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("video_generation_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "video_duration_under_40_minutes"
        and record.status == "fail"
        and "exceeds" in record.message
        for record in records
    )


def test_video_tail_check_fails_when_black_tail_is_detected(tmp_path, monkeypatch):
    workspace = make_validation_workspace(tmp_path)
    add_video_artifacts(workspace, duration_seconds=120.0)

    def fake_probe(path: Path):
        if path.suffix == ".m4a":
            return presentation_validation._MediaDurations(
                format_duration=120.0,
                audio_duration=120.0,
            )
        return presentation_validation._MediaDurations(
            format_duration=120.0,
            video_duration=120.0,
            audio_duration=120.0,
            video_frame_rate="25/1",
            video_frame_count=3000,
        )

    monkeypatch.setattr(presentation_validation, "_probe_media_durations", fake_probe)
    monkeypatch.setattr(
        presentation_validation,
        "_detect_black_tail_seconds",
        lambda *_args: presentation_validation.VIDEO_BLACK_TAIL_FAIL_SECONDS + 0.5,
    )

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("video_generation_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "video_tail_not_black"
        and record.status == "fail"
        and "black tail" in record.message
        for record in records
    )


def test_video_duration_checks_fail_when_final_mp4_does_not_match_receipts(
    tmp_path,
    monkeypatch,
):
    workspace = make_validation_workspace(tmp_path)
    add_video_artifacts(workspace, duration_seconds=120.0)

    def fake_probe(path: Path):
        if path.suffix == ".m4a":
            return presentation_validation._MediaDurations(
                format_duration=120.0,
                audio_duration=120.0,
            )
        return presentation_validation._MediaDurations(
            format_duration=100.0,
            video_duration=100.0,
            audio_duration=100.0,
            video_frame_rate="25/1",
            video_frame_count=2500,
        )

    monkeypatch.setattr(presentation_validation, "_probe_media_durations", fake_probe)
    monkeypatch.setattr(presentation_validation, "_detect_black_tail_seconds", lambda *_args: 0.0)

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("video_generation_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "video_duration_matches_build_receipts"
        and record.status == "fail"
        and "delta" in record.message
        for record in records
    )


def test_strict_required_artifact_mode_fails_missing_video_files(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("video_generation_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        strict_required_artifacts=True,
    )

    assert {
        (record.check_id, record.status)
        for record in records
    } >= {
        ("video_artifacts_nonempty", "fail"),
        ("video_duration_readable", "fail"),
        ("video_stream_durations_match", "fail"),
        ("video_duration_matches_build_receipts", "fail"),
        ("video_tail_not_black", "fail"),
    }


def test_strict_required_artifact_mode_fails_missing_figure_svgs(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    remove_generated_svgs(workspace)
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("figures_structural_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        strict_required_artifacts=True,
    )

    assert any(
        record.check_id == "svg_well_formed"
        and record.status == "fail"
        and "No declared SVG artifacts" in record.message
        for record in records
    )


def test_strict_required_artifact_mode_fails_missing_human_review(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("human_review_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        strict_required_artifacts=True,
    )

    assert any(
        record.checkpoint_stage_id == "human_review" and record.status == "fail"
        for record in records
    )


def test_strict_required_artifact_mode_passes_present_human_review(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    add_human_review_receipt(workspace)
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("human_review_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        strict_required_artifacts=True,
    )

    assert any(
        record.checkpoint_stage_id == "human_review"
        and record.status == "pass"
        and record.evidence_paths == ("decks/01_demo_topic/human_review_receipt.json",)
        for record in records
    )


def test_strict_required_artifact_mode_fails_unapproved_human_review(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    add_human_review_receipt(workspace, status="needs_revision")
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("human_review_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        strict_required_artifacts=True,
    )

    assert any(
        record.checkpoint_stage_id == "human_review"
        and record.status == "fail"
        and "partial" in record.message
        for record in records
    )


def test_visual_qc_report_present_is_warn_only(tmp_path):
    _, records = _records(tmp_path, "visual_qc_validation")
    assert any(
        record.check_id == "visual_qc_report_present"
        and record.status == "warn"
        and record.severity == "content"
        for record in records
    )
    assert any(
        record.check_id == "vision_llm_visual_qc_passed"
        and record.status == "pass"
        and record.severity == "content"
        for record in records
    )


def test_visual_qc_validation_fails_blocked_vision_llm_report(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    report_path = (
        workspace.root
        / "decks"
        / "01_demo_topic"
        / "assets"
        / "generated"
        / "visual_llm_qc_report.json"
    )
    report_path.write_text(
        '{"schema_version":1,"status":"blocked","errors":[{"check_id":"vision_llm_visual_qc_available"}],"warnings":[],"slides":[]}\n',
        encoding="utf-8",
    )
    manifest_path = workspace.root / "decks" / "presentation_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["presentations"][0]["stage_states"]["visual_qc"]["artifacts"][1][
        "sha256"
    ] = _artifact_sha256(report_path, kind="file")
    data["content_sha256"] = _compute_content_sha256(data)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("visual_qc_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "vision_llm_visual_qc_passed"
        and record.status == "fail"
        for record in records
    )


def test_revealjs_html_parses_on_valid_fixture(tmp_path):
    _, records = _records(tmp_path, "revealjs_render_validation")
    assert any(record.check_id == "revealjs_html_parses" and record.status == "pass" for record in records)
    assert any(
        record.check_id == "revealjs_reference_labels_visible" and record.status == "pass"
        for record in records
    )
    assert any(
        record.check_id == "revealjs_title_slide_branded" and record.status == "pass"
        for record in records
    )
    assert any(
        record.check_id == "revealjs_content_brand_mark_present" and record.status == "pass"
        for record in records
    )


def test_revealjs_validation_fails_visible_ref_number_labels(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    html_path = _topic_dir(workspace) / "presentation.html"
    html_path.write_text(
        html_path.read_text(encoding="utf-8").replace(">Demo Paper<", ">Ref 1<"),
        encoding="utf-8",
    )
    _set_density_targets(workspace)

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("revealjs_render_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "revealjs_reference_labels_visible"
        and record.status == "fail"
        and "generic visible label" in record.message
        for record in records
    )


def test_revealjs_validation_fails_missing_brand_treatment(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    _write_text(
        _topic_dir(workspace) / "presentation.html",
        (
            "<!doctype html><html><body>"
            '<section data-markdown-slide="1" class="slide-content">One</section>'
            '<section data-markdown-slide="2" class="slide-content">Two</section>'
            '<section data-markdown-slide="3" class="slide-content">Three</section>'
            "</body></html>\n"
        ),
    )
    _set_density_targets(workspace)

    records = run_validation_stage(
        workspace,
        workflow.get_workflow_stage("revealjs_render_validation"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
    )

    assert any(
        record.check_id == "revealjs_title_slide_branded" and record.status == "fail"
        for record in records
    )
    assert any(
        record.check_id == "revealjs_content_brand_mark_present" and record.status == "fail"
        for record in records
    )
