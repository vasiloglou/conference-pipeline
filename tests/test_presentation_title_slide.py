from __future__ import annotations
import pytest

import json
import subprocess
import sys
from pathlib import Path

from conference_pipeline.presentation.brand_template import (
    BRAND_REFERENCE_NAME,
    brand_assets_sha256,
    validate_brand_template_lock,
)
from conference_pipeline.presentation.render_spec import validate_render_spec_payload


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_add_title_slide_cli_inserts_canonical_slide_0(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    slides_md = workspace / "decks" / "01_demo_topic" / "5_draft_slides_with_images.md"
    write_text(
        workspace / "config.yaml",
        "conference:\n  name: DemoConf 2026\n",
    )
    write_text(
        slides_md,
        "# Old Title\n\n## Slide 1: First Claim\n\n- Evidence\n\n---\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/add_title_slide_to_slides_md.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_demo_topic",
            "--slides-md",
            str(slides_md),
            "--display-title",
            "Demo Topic",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["has_slide_0_title"] is True
    updated = slides_md.read_text(encoding="utf-8")
    assert updated.startswith("# Demo Topic at DemoConf 2026")
    assert "## Slide 0: Demo Topic at DemoConf 2026" in updated
    assert "SLIDE_TYPE: title" in updated
    assert updated.index("## Slide 0") < updated.index("## Slide 1")


def test_add_title_slide_cli_strips_deck_only_suffix_from_cover(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    slides_md = workspace / "decks" / "01_demo_topic" / "5_draft_slides_with_images.md"
    write_text(
        workspace / "config.yaml",
        "conference:\n  name: DemoConf 2026\n",
    )
    write_text(
        slides_md,
        "# Demo Topic Slide Brief at DemoConf 2026\n\n## Slide 1: First Claim\n\n- Evidence\n\n---\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/add_title_slide_to_slides_md.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--slides-md",
            str(slides_md),
            "--display-title",
            "Demo Topic Slide Brief",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    updated = slides_md.read_text(encoding="utf-8")
    assert updated.startswith("# Demo Topic at DemoConf 2026")
    assert "## Slide 0: Demo Topic at DemoConf 2026" in updated
    assert "# Demo Topic\n" in updated
    assert "Slide Brief" not in updated.split("## Slide 1", 1)[0]


@pytest.mark.repo_data("assets/brand/relationalai/logo_white.png")
def test_revealjs_uses_neurips_relationalai_brand_reference(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    slides_md = workspace / "decks" / "01_demo_topic" / "5_draft_slides_with_images.md"
    output_html = workspace / "decks" / "01_demo_topic" / "presentation.html"
    slide_mapping = workspace / "decks" / "01_demo_topic" / "presentation_slide_mapping.json"
    write_text(
        slides_md,
        """# Demo Topic

## Slide 0: Demo Topic
<!--
SLIDE_TYPE: title
-->

# Demo Topic
## DemoConf 2026

---

## Slide 1: First Claim

- Evidence
""",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/md_to_revealjs.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_demo_topic",
            "--slides-md",
            str(slides_md.relative_to(workspace)),
            "--output-html",
            str(output_html.relative_to(workspace)),
            "--slide-mapping",
            str(slide_mapping.relative_to(workspace)),
            "--require-brand-assets",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    html = output_html.read_text(encoding="utf-8")
    assert "--navy: #1a1a2e" in html
    assert "--rai-blue: #058DC7" in html
    assert "relationalai-neurips2025-reference" in html
    assert '<body class="">' in html
    assert "RelationalAI logo" in html


@pytest.mark.repo_data("assets/brand/relationalai/logo_white.png")
def test_revealjs_marks_long_title_slides_with_fit_class(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    slides_md = workspace / "decks" / "01_demo_topic" / "5_draft_slides_with_images.md"
    output_html = workspace / "decks" / "01_demo_topic" / "presentation.html"
    slide_mapping = workspace / "decks" / "01_demo_topic" / "presentation_slide_mapping.json"
    write_text(
        slides_md,
        """# Demo Topic

## Slide 0: Extremely Long Cover
<!--
SLIDE_TYPE: title
-->

# Safety, alignment, privacy, security, fairness, adversarial robustness, and governance
## DemoConf 2026

---

## Slide 1: First Claim

- Evidence
""",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/md_to_revealjs.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_demo_topic",
            "--slides-md",
            str(slides_md.relative_to(workspace)),
            "--output-html",
            str(output_html.relative_to(workspace)),
            "--slide-mapping",
            str(slide_mapping.relative_to(workspace)),
            "--require-brand-assets",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    html = output_html.read_text(encoding="utf-8")
    assert 'class="slide-title title-xlong"' in html
    assert ".slide-title.title-xlong h1" in html


@pytest.mark.repo_data("assets/brand/relationalai/logo_white.png")
def test_render_spec_cli_uses_brand_locked_assets_and_figures(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "04_safety_alignment_privacy"
    asset_path = deck_dir / "assets" / "generated" / "phase16" / "s01.svg"
    output_html = deck_dir / "presentation.html"
    slide_mapping = deck_dir / "presentation_slide_mapping.json"
    render_spec = deck_dir / "render_spec.json"
    write_text(
        asset_path,
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360">'
        '<rect width="640" height="360" fill="#058DC7"/>'
        '<text x="32" y="190" font-size="34" fill="white">Proof object</text>'
        "</svg>\n",
    )
    write_text(
        render_spec,
        json.dumps(
            {
                "schema_version": 1,
                "presentation_id": "04_safety_alignment_privacy",
                "author": "Nikolaos Vasiloglou",
                "analysis_report_url": "https://drive.google.com/final-report",
                "input_hashes": {"presentation_deck.json": "a" * 64},
                "slides": [
                    {
                        "slide_id": "s00",
                        "layout_blocks": [
                            {
                                "kind": "heading",
                                "text": "Safety, alignment, and privacy at ICLR 2026",
                            }
                        ],
                        "visual_asset_refs": [
                            "decks/04_safety_alignment_privacy/assets/generated/phase16/s01.svg"
                        ],
                        "source_chips": [
                            {
                                "title": "Opening Remarks",
                                "speakers": ["Don Song"],
                                "url": "https://openreview.net/forum?id=demo",
                            }
                        ],
                        "speaker_notes": "Introduce the canary deck.",
                    },
                    {
                        "slide_id": "s01",
                        "layout_blocks": [{"kind": "heading", "text": "A source-bound claim"}],
                        "visual_asset_refs": [
                            "decks/04_safety_alignment_privacy/assets/generated/phase16/s01.svg"
                        ],
                        "source_chips": [
                            {
                                "title": "Opening Remarks",
                                "speakers": ["Don Song"],
                                "url": "https://openreview.net/forum?id=demo",
                            }
                        ],
                        "speaker_notes": "Explain the real proof object without repeating author credits.",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/md_to_revealjs.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "04_safety_alignment_privacy",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "04_safety_alignment_privacy",
            "--render-spec",
            str(render_spec.relative_to(workspace)),
            "--output-html",
            str(output_html.relative_to(workspace)),
            "--slide-mapping",
            str(slide_mapping.relative_to(workspace)),
            "--require-brand-assets",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    html = output_html.read_text(encoding="utf-8")
    assert f'<meta name="presentation-template" content="{BRAND_REFERENCE_NAME}">' in html
    assert f'<meta name="presentation-template-sha256" content="{brand_assets_sha256()}">' in html
    assert '<img class="title-logo"' in html
    assert '<img class="content-brand-bug"' in html
    assert '<img class="slide-visual proof-visual"' in html
    assert 'src="assets/generated/phase16/s01.svg"' in html
    assert "Prepared by Nikolaos Vasiloglou" not in html
    lock = validate_brand_template_lock(
        html_content=html,
        assets_sha256=brand_assets_sha256(),
        reference_name=BRAND_REFERENCE_NAME,
    )
    assert lock.status == "pass"


def test_render_spec_rejects_minimal_structured_css_theme_replacement() -> None:
    payload = {
        "schema_version": 1,
        "presentation_id": "01_demo_topic",
        "input_hashes": {
            "presentation_blueprint.json": "c" * 64,
            "presentation_deck.json": "d" * 64,
            "source_binding_report.json": "e" * 64,
            "visual_plan.json": "f" * 64,
            "asset_inventory.json": "a" * 64,
            "delivery_single_speaker.json": "1" * 64,
            "delivery_notebooklm.json": "2" * 64,
        },
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": "https://drive.google.com/demo-report",
        "theme_contract": {
            "renderer_path": "minimal_structured_css",
            "theme_reference": "minimal structured CSS replacement",
            "title_slide": {"required": True},
            "title_logo": {"required": False, "safe_area_css_px": {"width": 1, "height": 1}},
            "decorative_art": {"required": False},
            "color_palette": {"required": ["teal"]},
            "content_brand_bug": {"required": False, "safe_area_css_px": {"width": 1, "height": 1}},
            "section_styling": {"required": False},
            "readable_source_chips": {"required": False},
            "uncropped_logos": {"required": False},
            "coverage_thresholds": {
                "alpha_non_background_min": 0.01,
                "luminance_standard_deviation_min": 1.0,
            },
        },
        "slides": [
            {
                "slide_id": "slide-001",
                "state_id": "state-001",
                "layout_blocks": [{"kind": "heading", "text": "Contract first"}],
                "source_chips": [{"label": "SRC-001", "url": "https://openreview.net/forum?id=demo"}],
                "visual_asset_refs": ["decks/01_demo_topic/assets/generated/vis-001.svg"],
                "speaker_notes": "Prepared by Nikolaos Vasiloglou.",
                "timing_seconds": 18,
            }
        ],
        "delivery": {
            "single_speaker": {"contract_path": "decks/01_demo_topic/delivery_single_speaker.json"},
            "notebooklm": {"contract_path": "decks/01_demo_topic/delivery_notebooklm.json"},
        },
    }

    result = validate_render_spec_payload(payload)

    assert result.status == "fail"
    assert any("RelationalAI" in error.message or "minimal structured CSS" in error.message for error in result.errors)
