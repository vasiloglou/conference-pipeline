from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.presentation.repair_loop import run_repair_loop
from conference_pipeline.presentation.brand_template import (
    BRAND_REFERENCE_NAME,
    brand_assets_sha256,
    validate_brand_template_lock,
)
from conference_pipeline.presentation.rendered_qc import inspect_rendered_slides
from conference_pipeline.presentation.structured_video import (
    export_render_spec_artifacts,
    render_spec_to_html,
)


def write_html_fixture(tmp_path: Path, body: str) -> tuple[Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "01_demo_topic"
    deck_dir.mkdir(parents=True)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ margin: 0; background: white; }}
    section {{ width: 1280px; height: 720px; position: relative; background: white; }}
    h1 {{ font: 42px Arial, sans-serif; color: #16213e; }}
  </style>
</head>
<body>{body}</body>
</html>
"""
    html_path = deck_dir / "presentation.html"
    html_path.write_text(html, encoding="utf-8")
    return workspace, deck_dir


def write_render_spec_fixture(deck_dir: Path) -> Path:
    asset = deck_dir / "assets" / "generated" / "slide-001.svg"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="240">'
        '<rect width="400" height="240" fill="#158158"/>'
        '<text x="24" y="128" font-size="32" fill="white">Real asset</text>'
        "</svg>\n",
        encoding="utf-8",
    )
    render_spec = deck_dir / "render_spec.json"
    render_spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presentation_id": "04_safety_alignment_privacy",
                "author": "Nikolaos Vasiloglou",
                "analysis_report_url": "https://drive.google.com/file/d/final-report",
                "input_hashes": {"presentation_deck.json": "a" * 64},
                "theme_contract": {
                    "brand_reference_name": "relationalai-neurips2025-reference",
                    "title_logo": {"required": True},
                    "decorative_art": {"required": True},
                    "content_brand_bug": {"required": True},
                    "readable_source_chips": {"required": True},
                },
                "slides": [
                    {
                        "slide_id": "sl_01_real_asset",
                        "layout_blocks": [{"kind": "heading", "text": "Real rendered asset"}],
                        "visual_asset_refs": [
                            "decks/04_safety_alignment_privacy/assets/generated/slide-001.svg"
                        ],
                        "source_chips": [
                            {
                                "title": "Agents in the Wild: Safety, Security, and Beyond",
                                "speakers": ["Don Song"],
                                "url": "https://openreview.net/forum?id=real",
                            }
                        ],
                        "speaker_notes": "Explain the rendered asset and source chip.",
                    },
                    {
                        "slide_id": "sl_02_real_asset",
                        "layout_blocks": [{"kind": "heading", "text": "Real content slide asset"}],
                        "visual_asset_refs": [
                            "decks/04_safety_alignment_privacy/assets/generated/slide-001.svg"
                        ],
                        "source_chips": [
                            {
                                "title": "Agents in the Wild: Safety, Security, and Beyond",
                                "speakers": ["Don Song"],
                                "url": "https://openreview.net/forum?id=real",
                            }
                        ],
                        "evidence_caption": "Readable evidence caption, not a narration dump.",
                        "visual_takeaway": "The figure gets the main area while sources stay compact.",
                        "speaker_notes": (
                            "This full narration sentence should stay in the spoken script and "
                            "must not be printed as body copy on the slide."
                        ),
                    },
                    {
                        "slide_id": "s-resources",
                        "slide_kind": "resources",
                        "layout_blocks": [{"kind": "heading", "text": "Further reading"}],
                        "resources": [
                            {
                                "title": "Presentation 01: LLM reasoning",
                                "url": "../01_reasoning_post_training/presentation.html",
                                "kind": "topic_presentation",
                            }
                        ],
                        "analysis_url": "https://drive.google.com/file/d/final-report",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return render_spec


@pytest.mark.repo_data("assets/brand/relationalai/logo_white.png")
def test_render_spec_html_uses_real_assets_not_visible_asset_path_text(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "04_safety_alignment_privacy"
    deck_dir.mkdir(parents=True)
    render_spec = write_render_spec_fixture(deck_dir)

    html_path = render_spec_to_html(
        render_spec,
        output_html=deck_dir / "presentation.html",
        slide_mapping=deck_dir / "presentation_slide_mapping.json",
        workspace=workspace,
    )

    html = html_path.read_text(encoding="utf-8")
    assert f'<meta name="presentation-template" content="{BRAND_REFERENCE_NAME}">' in html
    assert f'<meta name="presentation-template-sha256" content="{brand_assets_sha256()}">' in html
    assert '<meta name="brand-asset-logo"' in html
    assert '<meta name="brand-asset-decorative_art"' in html
    assert '<meta name="brand-asset-bug"' in html
    assert '<meta name="brand-asset-section_bg"' in html
    assert '<img class="title-logo"' in html
    assert '<img class="decorative-art"' in html
    assert '<img class="content-brand-bug"' in html
    assert '<svg class="title-logo"' not in html
    assert ">RAI</div>" not in html
    assert "width: 309px" in html
    assert "height: 48px" in html
    assert "right: 24px" in html
    assert "bottom: 18px" in html
    assert "grid-template-columns: minmax(190px, 220px) minmax(880px, 1fr)" in html
    assert '<img class="slide-visual proof-visual"' in html
    assert 'src="assets/generated/slide-001.svg"' in html
    assert ">decks/04_safety_alignment_privacy/assets/generated/slide-001.svg<" not in html
    assert "Readable evidence caption, not a narration dump." in html
    assert "This full narration sentence should stay" not in html
    assert 'href="../01_reasoning_post_training/presentation.html"' in html
    lock = validate_brand_template_lock(
        html_content=html,
        assets_sha256=brand_assets_sha256(),
        reference_name=BRAND_REFERENCE_NAME,
    )
    assert lock.status == "pass"
    assert "presentation_slide_mapping.json" in (deck_dir / "presentation_slide_mapping.json").name
    mapping = json.loads((deck_dir / "presentation_slide_mapping.json").read_text(encoding="utf-8"))
    assert mapping["slides"][0]["slide_id"] == "sl_01_real_asset"
    assert mapping["slides"][0]["slide_type"] == "title"
    assert mapping["slides"][0]["html_section_index"] == 0
    assert mapping["slides"][0]["frame_order"] == 1


@pytest.mark.repo_data("assets/brand/relationalai/logo_white.png")
def test_executive_lesson_render_hides_internal_source_locator(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "executive_summary_short"
    deck_dir.mkdir(parents=True)
    render_spec = write_render_spec_fixture(deck_dir)
    payload = json.loads(render_spec.read_text(encoding="utf-8"))
    payload["presentation_id"] = "executive_summary_short"
    payload["presentation_type"] = "executive-summary"
    payload["slides"][1].update(
        {
            "slide_id": "executive_summary_short-s003-system-progress",
            "executive_lesson_id": "exec_lesson_01",
            "source_anchor": "Agentic systems transcripts, Summary and Key Points",
            "source_locator": "analysis/exhaustive_summaries/02_agentic_systems__transcripts.md#summary",
            "evidence_caption": (
                "Tight transcript paraphrase: Agentic evidence reframes autonomy "
                "as environments, tools, memory, verifiers, monitors, incentives, "
                "and governance around the model."
            ),
        }
    )
    render_spec.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html_path = render_spec_to_html(
        render_spec,
        output_html=deck_dir / "presentation.html",
        slide_mapping=deck_dir / "presentation_slide_mapping.json",
        workspace=workspace,
    )

    html = html_path.read_text(encoding="utf-8")
    assert "Tight transcript paraphrase:" not in html
    assert "analysis/exhaustive_summaries" not in html
    assert "Agentic evidence reframes autonomy" in html
    assert "Agentic systems transcripts, Summary and Key Points" in html


def test_structured_renderer_blocks_tiny_svg_text_before_export(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "04_safety_alignment_privacy"
    deck_dir.mkdir(parents=True)
    render_spec = write_render_spec_fixture(deck_dir)
    (deck_dir / "assets" / "generated" / "slide-001.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="240">'
        '<text x="10" y="20" font-size="8">tiny</text></svg>\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="visual legibility gate failed"):
        render_spec_to_html(
            render_spec,
            output_html=deck_dir / "presentation.html",
            slide_mapping=deck_dir / "presentation_slide_mapping.json",
            workspace=workspace,
        )


def test_structured_renderer_blocks_scaffold_raw_ids_and_text_fragments(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "04_safety_alignment_privacy"
    deck_dir.mkdir(parents=True)
    render_spec = write_render_spec_fixture(deck_dir)
    (deck_dir / "assets" / "generated" / "slide-001.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="240">'
        '<text x="10" y="20" font-size="24">source evidence -> visual mechanism -> audience claim</text>'
        '<text x="10" y="52" font-size="24">speech_transcript:demo:segment:00001</text>'
        '<text x="10" y="84" font-size="24">Evaluation claim ends with the</text>'
        "</svg>\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="internal_scaffold_visible"):
        render_spec_to_html(
            render_spec,
            output_html=deck_dir / "presentation.html",
            slide_mapping=deck_dir / "presentation_slide_mapping.json",
            workspace=workspace,
        )


def test_structured_renderer_blocks_svg_source_person_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "04_safety_alignment_privacy"
    deck_dir.mkdir(parents=True)
    render_spec = write_render_spec_fixture(deck_dir)
    payload = json.loads(render_spec.read_text(encoding="utf-8"))
    payload["slides"][0]["source_chips"] = [
        {
            "label": "SRC-011",
            "title": "Agent Beat Security Arena",
            "speakers": ["Davina Jane"],
            "authors": [],
            "systems": ["Agent Beat Security Arena adversarial competition system"],
            "works": ["Agent Beat Security Arena: An Adversarial Prompt Injection Competition"],
        },
        {
            "label": "SRC-024",
            "title": "Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?",
            "authors": ["Alan Ritter"],
            "speakers": [],
            "systems": ["VLM-GEOPRIVACY contextual-integrity benchmark"],
            "works": ["VLM-GEOPRIVACY"],
        },
    ]
    render_spec.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (deck_dir / "assets" / "generated" / "slide-001.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="240">'
        '<text x="10" y="20" font-size="24">Agent Beat Security Arena</text>'
        '<text x="10" y="52" font-size="24">Alan Ritter</text>'
        '<text x="10" y="84" font-size="24">SRC-011</text>'
        "</svg>\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_person_mismatch:SRC-011:Alan Ritter"):
        render_spec_to_html(
            render_spec,
            output_html=deck_dir / "presentation.html",
            slide_mapping=deck_dir / "presentation_slide_mapping.json",
            workspace=workspace,
        )


@pytest.mark.repo_data("assets/brand/relationalai/logo_white.png")
def test_export_render_spec_artifacts_records_frame_manifest_contact_sheet_and_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "04_safety_alignment_privacy"
    deck_dir.mkdir(parents=True)
    render_spec = write_render_spec_fixture(deck_dir)

    def fake_export_pdf(*, input_html: Path, output_pdf: Path) -> None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        output_pdf.write_bytes(b"%PDF-1.7\n% Playwright Chromium fixture\n")

    def fake_export_frames(*, html_path: Path, slide_mapping: Path, frames_dir: Path):
        from PIL import Image

        frames_dir.mkdir(parents=True, exist_ok=True)
        frame = frames_dir / "slide_001.png"
        Image.new("RGB", (1280, 720), "#158158").save(frame)
        return [{"slide_id": "sl_01_real_asset", "frame_path": frame, "index": 1}]

    monkeypatch.setattr(
        "conference_pipeline.presentation.structured_video.export_pdf_with_playwright",
        fake_export_pdf,
    )
    monkeypatch.setattr(
        "conference_pipeline.presentation.structured_video.export_slide_frames_with_playwright",
        fake_export_frames,
    )

    receipt = export_render_spec_artifacts(
        render_spec=render_spec,
        output_html=deck_dir / "presentation.html",
        output_pdf=deck_dir / "presentation.pdf",
        slide_mapping=deck_dir / "presentation_slide_mapping.json",
        frame_manifest=deck_dir / "slide_frame_manifest.json",
        frames_dir=deck_dir / "rendered_frames",
        contact_sheet=deck_dir / "render_contact_sheet.png",
        receipt_output=deck_dir / "render_export_receipt.json",
        workspace=workspace,
        command=["export_revealjs_pdf.py", "--render-spec", "render_spec.json"],
    )

    assert receipt["status"] == "pass"
    assert receipt["legacy_markdown_paths"] == []
    assert receipt["slide_count"] == 1
    assert Path(receipt["contact_sheet_path"]).name == "render_contact_sheet.png"
    frame_manifest = json.loads((deck_dir / "slide_frame_manifest.json").read_text(encoding="utf-8"))
    assert frame_manifest["frames"][0]["sha256"]
    assert frame_manifest["frames"][0]["byte_count"] > 0
    assert frame_manifest["frames"][0]["dimensions"] == {"width": 1280, "height": 720}
    assert "frame_manifest" in receipt
    assert "Playwright" in receipt["export_commands"][0]["engine"]
    assert "Chromium" in receipt["export_commands"][0]["engine"]


def test_rendered_qc_fails_blank_svg_fixture(tmp_path: Path) -> None:
    workspace, deck_dir = write_html_fixture(
        tmp_path,
        """<section>
  <svg width="400" height="240" viewBox="0 0 400 240"></svg>
</section>""",
    )

    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "assets" / "generated" / "visual_qc_report.json",
        screenshots_dir=deck_dir / "assets" / "generated" / "qc" / "screenshots",
    )

    assert any(error["check_id"] == "svg_nonblank" for error in report["errors"])
    assert (workspace / str(report["slides"][0]["screenshot_path"])).is_file()


def test_rendered_qc_records_screenshot_paths_and_viewport(tmp_path: Path) -> None:
    _, deck_dir = write_html_fixture(
        tmp_path,
        """<section>
  <h1>Generated Figure</h1>
  <svg width="500" height="260" viewBox="0 0 500 260">
    <rect x="20" y="20" width="460" height="220" fill="#2f80ed"/>
    <text x="250" y="145" text-anchor="middle" fill="white" font-size="42">QC</text>
  </svg>
</section>""",
    )

    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "assets" / "generated" / "visual_qc_report.json",
        screenshots_dir=deck_dir / "assets" / "generated" / "qc" / "screenshots",
    )

    assert report["viewport"]["width"] == 1280
    assert report["viewport"]["height"] == 720
    assert not any(
        error["check_id"] == "no_text_figure_overlap" for error in report["errors"]
    )
    screenshot_paths = [slide["screenshot_path"] for slide in report["slides"]]
    assert screenshot_paths
    assert all("assets/generated/qc/screenshots/" in path for path in screenshot_paths)
    on_disk_report = json.loads(
        (deck_dir / "assets" / "generated" / "visual_qc_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert on_disk_report["viewport"] == report["viewport"]


def test_rendered_qc_counts_hidden_title_template_assets_across_deck(tmp_path: Path) -> None:
    _, deck_dir = write_html_fixture(
        tmp_path,
        """<section class="slide-title" data-slide-type="title">
  <div class="title-logo">RelationalAI</div>
  <div class="decorative-art">Template art</div>
  <h1>Safety Alignment Privacy</h1>
</section>
<section>
  <div class="content-brand-bug">RAI</div>
  <div class="reference-chips">SRC-001</div>
  <h1>Content slide</h1>
</section>""",
    )

    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "assets" / "generated" / "visual_qc_report.json",
        screenshots_dir=deck_dir / "assets" / "generated" / "qc" / "screenshots",
    )

    assert report["status"] == "passed"
    assert not any(
        error["slide_number"] == 2
        and error["check_id"] in {"title slide", "title logo", "decorative art"}
        for error in report["errors"]
    )


def test_rendered_qc_flags_clipped_main_slide_visual(tmp_path: Path) -> None:
    _, deck_dir = write_html_fixture(
        tmp_path,
        """<section>
  <img class="slide-visual" style="position:absolute; left:80px; top:640px; width:400px; height:120px;"
       src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='120'%3E%3Crect width='400' height='120' fill='%23158158'/%3E%3C/svg%3E">
</section>""",
    )

    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "assets" / "generated" / "visual_qc_report.json",
        screenshots_dir=deck_dir / "assets" / "generated" / "qc" / "screenshots",
    )

    assert any(error["check_id"] == "figure_clipped" for error in report["errors"])


def test_rendered_qc_clears_stale_slide_screenshots(tmp_path: Path) -> None:
    _, deck_dir = write_html_fixture(
        tmp_path,
        """<section><h1>Current slide</h1></section>""",
    )
    screenshots_dir = deck_dir / "assets" / "generated" / "qc" / "screenshots"
    screenshots_dir.mkdir(parents=True)
    stale = screenshots_dir / "slide_999.png"
    stale.write_bytes(b"stale")
    keep = screenshots_dir / "notes.txt"
    keep.write_text("operator note\n", encoding="utf-8")

    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "assets" / "generated" / "visual_qc_report.json",
        screenshots_dir=screenshots_dir,
    )

    assert len(report["slides"]) == 1
    assert not stale.exists()
    assert keep.is_file()
    assert (screenshots_dir / "slide_001.png").is_file()


def test_rendered_qc_inspects_unmapped_figure_zoom_sections(tmp_path: Path) -> None:
    _, deck_dir = write_html_fixture(
        tmp_path,
        """<section><h1>Mapped content slide</h1></section>
<section class="slide-figure-zoom">
  <svg width="400" height="240" viewBox="0 0 400 240"></svg>
</section>""",
    )
    (deck_dir / "presentation_slide_mapping.json").write_text('{"1": 0}\n', encoding="utf-8")

    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "assets" / "generated" / "visual_qc_report.json",
        screenshots_dir=deck_dir / "assets" / "generated" / "qc" / "screenshots",
    )

    assert len(report["slides"]) == 2
    assert any(error["check_id"] == "svg_nonblank" for error in report["errors"])
    assert (deck_dir / "assets" / "generated" / "qc" / "screenshots" / "slide_002.png").is_file()


def test_rendered_qc_flags_missing_render_artifact_truth_elements(tmp_path: Path) -> None:
    _, deck_dir = write_html_fixture(
        tmp_path,
        """<section>
  <h1>Safety Alignment Privacy</h1>
</section>
<section>
  <p class="visual-ref">assets/generated/slide-001.svg</p>
</section>""",
    )

    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "assets" / "generated" / "visual_qc_report.json",
        screenshots_dir=deck_dir / "assets" / "generated" / "qc" / "screenshots",
    )

    check_ids = {error["check_id"] for error in report["errors"]}
    assert {
        "title slide",
        "title logo",
        "decorative art",
        "content brand bug",
        "source chips",
        "file-path text",
    }.issubset(check_ids)


def test_repair_loop_records_attempts_and_stops_at_cap(tmp_path: Path) -> None:
    workspace, deck_dir = write_html_fixture(
        tmp_path,
        """<section>
  <svg width="400" height="240" viewBox="0 0 400 240"></svg>
</section>""",
    )
    generated_dir = deck_dir / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    (generated_dir / "slide_001_blank.svg").write_text(
        '<svg width="400" height="240" viewBox="0 0 400 240"></svg>\n',
        encoding="utf-8",
    )
    (deck_dir / "5_draft_slides_with_images.md").write_text("# Slides\n", encoding="utf-8")
    (deck_dir / "presentation_slide_mapping.json").write_text("{}\n", encoding="utf-8")

    receipt = run_repair_loop(
        workspace,
        "01_demo_topic",
        slides_md=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=generated_dir,
        html_output=deck_dir / "presentation.html",
        slide_mapping=deck_dir / "presentation_slide_mapping.json",
        validation_report=generated_dir / "validation_report.md",
        visual_qc_report=generated_dir / "visual_qc_report.md",
        screenshots_dir=generated_dir / "qc" / "screenshots",
        receipt_output=generated_dir / "repair_loop_receipt.json",
        max_attempts=2,
    )

    assert receipt["max_attempts"] == 2
    assert len(receipt["attempts"]) == 2
    assert receipt["status"] == "blocked"
    on_disk = json.loads((generated_dir / "repair_loop_receipt.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "blocked"


def test_repair_loop_regenerates_figures_from_llm_feedback(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "01_demo_topic"
    generated_dir = deck_dir / "assets" / "generated"
    generated_dir.mkdir(parents=True)
    slides = deck_dir / "5_draft_slides_with_images.md"
    slides.write_text(
        """# Demo

## Slide 1: Agentic Workflow Search

- Automated search explores agent/tool combinations.
- Workflow quality depends on measured constraints.

<!--
SLIDE_TYPE: content
FIGURE_NEED: Diagram showing an algorithm testing different combinations of agents and tools to find the optimal workflow.
SPEAKER_NOTES: Notes.
-->

![Generated figure for slide 1](assets/generated/slide_001_old.svg)
""",
        encoding="utf-8",
    )
    (generated_dir / "slide_001_old.svg").write_text(
        '<svg width="400" height="240" viewBox="0 0 400 240"><text font-size="8">tiny</text></svg>\n',
        encoding="utf-8",
    )
    (deck_dir / "source_pack_receipt.json").write_text("{}\n", encoding="utf-8")
    (deck_dir / "ranked_papers.json").write_text("[]\n", encoding="utf-8")
    (deck_dir / "asset_inventory.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "slide_number": 1,
                        "path": "decks/01_demo_topic/assets/generated/slide_001_old.svg",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (deck_dir / "presentation.html").write_text("<html></html>\n", encoding="utf-8")
    (deck_dir / "presentation_slide_mapping.json").write_text('{"1": 0}\n', encoding="utf-8")
    (generated_dir / "visual_llm_qc_report.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "errors": [
                    {
                        "slide_number": 1,
                        "check_id": "text_clipped",
                        "severity": "blocker",
                        "description": "workflow card copy is cut off",
                        "evidence": "right-side diagram",
                        "suggested_fix": "regenerate with larger labels and no clipped footer",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = run_repair_loop(
        workspace,
        "01_demo_topic",
        slides_md=slides,
        generated_dir=generated_dir,
        html_output=deck_dir / "presentation.html",
        slide_mapping=deck_dir / "presentation_slide_mapping.json",
        validation_report=generated_dir / "validation_report.md",
        visual_qc_report=generated_dir / "visual_qc_report.md",
        screenshots_dir=generated_dir / "qc" / "screenshots",
        receipt_output=generated_dir / "repair_loop_receipt.json",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        llm_qc_report=generated_dir / "visual_llm_qc_report.json",
        repair_feedback_output=generated_dir / "visual_repair_feedback.json",
        max_attempts=1,
        rerender_after_repair=False,
    )

    attempt = receipt["attempts"][0]
    assert receipt["status"] == "blocked"
    assert attempt["repair_action"] == "regenerated_programmatic_figures"
    assert attempt["repair_feedback_repair_count"] == 1
    assert attempt["llm_qc_rerun_required"] is True
    assert attempt["llm_qc_block_reason"] == "fresh governed vision-LLM QC required after repair"
    feedback = json.loads((generated_dir / "visual_repair_feedback.json").read_text(encoding="utf-8"))
    assert feedback["repairs"][0]["slide_number"] == 1
    assert "remove clipped footer/citation text and shorten labels" in feedback["repairs"][0]["directives"]
    inventory = json.loads((deck_dir / "asset_inventory.json").read_text(encoding="utf-8"))
    assert inventory["figures"][0]["repair_profile"]["source"] == "visual_llm_qc_feedback"
    repaired_svg = Path(workspace / inventory["figures"][0]["path"])
    assert repaired_svg.is_file()
    repaired_text = repaired_svg.read_text(encoding="utf-8")
    assert "Task goal" in repaired_text
    assert "Best" in repaired_text
    assert "workflow" in repaired_text
    assert "Sources: deck links and speaker notes" not in repaired_text


# ---------------------------------------------------------------------------
# Text geometry: overflow past the canvas, and text painted over text.
#
# Added 2026-09-15. Both executive decks shipped a slide whose list ran ~956px
# past the 720px canvas (cropped by section { overflow: hidden }) with the
# absolutely-positioned footer landing on top of the topic list. The inspector
# measured `clipped` only for <svg>/<img> and overlap only text-vs-figure, so
# it reported the slide clean. These are the two checks that would have caught
# it without spending a provider call.
# ---------------------------------------------------------------------------


def test_rendered_qc_flags_text_overflowing_the_slide_canvas(tmp_path: Path) -> None:
    items = "\n".join(f"<li>Overflowing list item number {i}</li>" for i in range(40))
    _, deck_dir = write_html_fixture(
        tmp_path,
        f"""<section style="overflow: hidden;">
  <h1>Too much content</h1>
  <ul style="font: 24px Arial, sans-serif;">{items}</ul>
</section>""",
    )

    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "assets" / "generated" / "visual_qc_report.json",
        screenshots_dir=deck_dir / "assets" / "generated" / "qc" / "screenshots",
    )

    overflow = [e for e in report["errors"] if e["check_id"] == "text_within_slide"]
    assert overflow, "text running past the slide canvas must be an error"
    assert "beyond the slide canvas" in overflow[0]["message"]


def test_rendered_qc_flags_text_painted_over_other_text(tmp_path: Path) -> None:
    _, deck_dir = write_html_fixture(
        tmp_path,
        """<section>
  <h1>Heading</h1>
  <p style="position: absolute; left: 60px; top: 300px; width: 600px;
            font: 28px Arial, sans-serif;">Underneath text that is covered up</p>
  <p style="position: absolute; left: 60px; top: 300px; width: 600px;
            font: 28px Arial, sans-serif;">Footer painted on top of it</p>
</section>""",
    )

    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "assets" / "generated" / "visual_qc_report.json",
        screenshots_dir=deck_dir / "assets" / "generated" / "qc" / "screenshots",
    )

    overlaps = [e for e in report["errors"] if e["check_id"] == "no_text_text_overlap"]
    assert overlaps, "text painted over other text must be an error"


def test_rendered_qc_does_not_flag_a_well_laid_out_two_column_slide(tmp_path: Path) -> None:
    """Guard against false positives: side-by-side columns must stay clean."""
    left = "\n".join(f"<li>Left item {i}</li>" for i in range(5))
    right = "\n".join(f"<li>Right item {i}</li>" for i in range(5))
    _, deck_dir = write_html_fixture(
        tmp_path,
        f"""<section>
  <h1>Two columns</h1>
  <div style="display: flex; gap: 40px; font: 20px Arial, sans-serif;">
    <ul style="width: 500px;">{left}</ul>
    <ul style="width: 500px;">{right}</ul>
  </div>
</section>""",
    )

    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "assets" / "generated" / "visual_qc_report.json",
        screenshots_dir=deck_dir / "assets" / "generated" / "qc" / "screenshots",
    )

    geometry = [
        e
        for e in report["errors"]
        if e["check_id"] in {"text_within_slide", "no_text_text_overlap"}
    ]
    assert geometry == [], f"unexpected text-geometry findings: {geometry}"


def test_rendered_qc_reports_slide_text_lint_before_any_vision_call(tmp_path: Path) -> None:
    """A leaked LaTeX macro, a raw Markdown link and a label cut off on '&' are string
    properties: the inspector reports them from the deck's own HTML, so they are found
    free of charge and before the paid vision pass."""

    bad = (
        '<section data-slide-id="sl_a"><h2>Speed and stability</h2>'
        '<div class="reference-chips" data-qc-role="source-chips"><span>Sources</span>'
        '<a class="source-chip" href="https://x.org/1">Pingzhi Li: textttFlashSchNet</a>'
        '<a class="source-chip" href="https://x.org/2">4th Structured Probabilistic Inference &amp;</a></div>'
        '<p>ComplexMCP, [OpenReview](https://openreview.net/forum?id=1)</p>'
        '<figure><img data-qc-role="figure" src="fig.svg"></figure></section>'
    )
    workspace, deck_dir = write_html_fixture(tmp_path, bad)
    (deck_dir / "fig.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect width="200" height="100" fill="#158158"/>'
        '<text x="8" y="40" font-size="24">Speed and stability</text></svg>'
    )
    report = inspect_rendered_slides(
        deck_dir / "presentation.html",
        output_report=deck_dir / "report.json",
        screenshots_dir=deck_dir / "screens",
    )
    lint_errors = {e["check_id"] for e in report["errors"] if e.get("source") == "slide_text_lint"}
    assert lint_errors == {"leaked_latex_macro", "dangling_label_end", "raw_markdown_link"}
    assert report["status"] == "failed"
    assert report["text_lint"]["errors"] == 3 and report["text_lint"]["warnings"] == 1
    assert [w["check_id"] for w in report["warnings"]] == ["duplicate_title_in_figure"]
    assert all(w["severity"] == "warning" for w in report["warnings"])
    assert report["slides"][0]["text_lint"], "per-slide findings are recorded next to the screenshot"


def test_rendered_qc_text_lint_is_silent_on_clean_text(tmp_path: Path) -> None:
    workspace, deck_dir = write_html_fixture(
        tmp_path,
        '<section><h2>Clean</h2><div data-qc-role="source-chips"><a class="source-chip" href="https://x.org">'
        "Michael McCabe: Walrus</a></div></section>",
    )
    report = inspect_rendered_slides(
        deck_dir / "presentation.html", output_report=deck_dir / "report.json", screenshots_dir=deck_dir / "screens",
    )
    assert not [e for e in report["errors"] if e.get("source") == "slide_text_lint"]
    assert report["text_lint"] == {"lint_version": "slide-text-lint/1", "errors": 0, "warnings": 0, "slides_with_findings": 0}
