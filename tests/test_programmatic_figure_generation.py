from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

from conference_pipeline.presentation.programmatic_figures import (
    APPROVED_SVG_CODEGEN_MODEL,
    _figure_type,
    generate_programmatic_figures,
)
from conference_pipeline.presentation.renderer_registry import (
    FIGURE_RENDERER_MISMATCH,
    UNSUPPORTED_IN_REGISTRY_V1,
    validate_figure_design_brief,
    validate_figure_design_brief_v2,
)
from conference_pipeline.presentation.semantic_svg_review import review_programmatic_figures
from conference_pipeline.presentation.visual_plan import validate_visual_plan_payload
from scripts.presentations.generate_programmatic_figures import generate_structured_visual_plan
from scripts.presentations.review_programmatic_figures import review_programmatic_figure_inventory

REPO_ROOT = Path(__file__).resolve().parent.parent



def _opus_model() -> str:
    """The catalog's current Opus model (these tests pin the tier, not a generation)."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def write_deck_fixture(tmp_path: Path, slides: str) -> tuple[Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "01_demo_topic"
    deck_dir.mkdir(parents=True)
    (deck_dir / "3.5_draft_slides_consolidated.md").write_text(slides, encoding="utf-8")
    (deck_dir / "source_pack_receipt.json").write_text(
        '{"schema_version": 1, "topic_id": "01_demo_topic"}\n',
        encoding="utf-8",
    )
    (deck_dir / "ranked_papers.json").write_text('{"papers": []}\n', encoding="utf-8")
    return workspace, deck_dir


def write_ranked_papers(deck_dir: Path, papers: list[dict[str, object]]) -> None:
    (deck_dir / "ranked_papers.json").write_text(
        json.dumps({"papers": papers}, indent=2),
        encoding="utf-8",
    )


def write_structured_deck_fixture(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "04_safety_alignment_privacy"
    deck_dir.mkdir(parents=True)
    deck_payload = {
        "schema_version": 1,
        "presentation_id": "04_safety_alignment_privacy",
        "slides": [
            {
                "slide_id": "sl_01_agent_safety",
                "title": "Safety moved from refusal to deployed-system trust",
                "visual_placeholders": [
                    {
                        "visual_id": "sl_01_agent_safety_visual",
                        "role": (
                            "Mechanism map connecting model output, agent loop, tools, "
                            "memory, and oversight"
                        ),
                    }
                ],
                "source_label_slots": ["SRC-001", "SRC-002"],
                "narration_hooks": [
                    (
                        "Show Don Song and Bo Li evidence connecting agent tools, memory, "
                        "and oversight to deployed-system safety."
                    )
                ],
            }
        ],
    }
    source_binding_payload = {
        "schema_version": 1,
        "presentation_id": "04_safety_alignment_privacy",
        "slides": [
            {
                "slide_id": "sl_01_agent_safety",
                "visible_references": [
                    {
                        "label": "SRC-001",
                        "source_id": "speech_transcript:aiwild:segment:00001",
                        "title": "Agents in the Wild: Safety, Security, and Beyond",
                        "source_type": "talk",
                        "speakers": ["Don Song"],
                        "authors": [],
                        "contribution_note": (
                            "Frames agentic systems as safety-critical deployed systems "
                            "with tools, memory, and environment attack surfaces."
                        ),
                        "relevance_rationale": (
                            "Don Song grounds the slide's system-boundary claim in the "
                            "AIWILD workshop evidence."
                        ),
                    },
                    {
                        "label": "SRC-002",
                        "source_id": "speech_transcript:aiwild:segment:00054",
                        "title": "Principles and Practices for Secure and Trustworthy Agentic AI",
                        "source_type": "talk",
                        "speakers": ["Bo Li"],
                        "authors": [],
                        "contribution_note": (
                            "Links indirect prompt injection and attacker/defender agent "
                            "benchmarks to oversight requirements."
                        ),
                        "relevance_rationale": (
                            "Bo Li supplies the attack-and-defense relationship that the "
                            "visual must depict."
                        ),
                    },
                ],
            }
        ],
    }
    (deck_dir / "presentation_deck.json").write_text(
        json.dumps(deck_payload, indent=2),
        encoding="utf-8",
    )
    (deck_dir / "source_binding_report.json").write_text(
        json.dumps(source_binding_payload, indent=2),
        encoding="utf-8",
    )
    return workspace, deck_dir


def test_programmatic_figures_write_svg_inventory_and_markdown(tmp_path: Path) -> None:
    workspace, deck_dir = write_deck_fixture(
        tmp_path,
        """## Slide 1: Scaling Curve

- Precision scaling changes the curve.

<!--
FIGURE_NEED: line graph showing parameter scaling vs precision scaling
-->

---

## Slide 2: Evidence Flow

- Evidence is routed through deterministic source packs.

<!--
FIGURE_NEED: flowchart showing source pack to SVG generation
-->
""",
    )

    inventory = generate_programmatic_figures(
        workspace,
        "01_demo_topic",
        input_slides=deck_dir / "3.5_draft_slides_consolidated.md",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        output_slides=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=deck_dir / "assets" / "generated",
    )

    generated = sorted((deck_dir / "assets" / "generated").glob("slide_001_*.svg"))
    assert generated
    assert generated[0].stat().st_size > 200
    assert (deck_dir / "asset_inventory.json").is_file()
    assert inventory["figures"][0]["renderer"] in {"svgwrite", "matplotlib", "plotly"}
    assert inventory["figures"][0]["renderer_family"] == "direct_svg"
    assert inventory["figures"][0]["figure_design_brief"]["renderer_family"] == "direct_svg"
    assert inventory["figures"][0]["figure_design_brief"]["blueprint_id"] is None
    assert inventory["figures"][0]["figure_design_brief"]["claim_to_visualize"]
    assert "assets/generated/" in (deck_dir / "5_draft_slides_with_images.md").read_text(
        encoding="utf-8"
    )
    assert "paper_crop_completion" not in json.dumps(inventory)
    assert inventory["model_governance"]["model_id"] == APPROVED_SVG_CODEGEN_MODEL
    assert (deck_dir / "assets" / "generated" / "figure_specs.json").is_file()


def test_structured_visual_plan_requires_source_grounded_figure_brief() -> None:
    payload = {
        "presentation_id": "04_safety_alignment_privacy",
        "slides": [
            {
                "slide_id": "sl_01",
                "figures": [
                    {
                        "visual_id": "sl_01_visual",
                        "figure_design_brief": "Source-grounded map",
                        "depiction_type": "mechanism",
                        "renderer_family": "direct_svg",
                        "required_entities": ["Agent loop"],
                        "required_relationships": ["tools -> oversight"],
                        "source_labels": ["SRC-001"],
                        "source_ids": [],
                        "contribution_note": "Don Song grounds the deployed-system safety claim.",
                        "narration_hook": "Name the agent loop and oversight relationship.",
                        "forbidden_patterns": ["decorative", "fake chart ticks"],
                        "asset_paths": ["decks/demo/assets/generated/sl_01.svg"],
                    }
                ],
            }
        ],
    }

    result = validate_visual_plan_payload(payload)

    assert result.status == "fail"
    assert any("source_ids" in error.message for error in result.errors)


def test_structured_generation_records_source_ids_relationships_and_real_asset_hashes(
    tmp_path: Path,
) -> None:
    workspace, deck_dir = write_structured_deck_fixture(tmp_path)
    generated_dir = deck_dir / "assets" / "generated"

    payload = generate_structured_visual_plan(
        SimpleNamespace(
            workspace=workspace,
            presentation_id="04_safety_alignment_privacy",
            deck_contract=deck_dir / "presentation_deck.json",
            source_binding_report=deck_dir / "source_binding_report.json",
            visual_plan=deck_dir / "visual_plan.json",
            asset_inventory=deck_dir / "asset_inventory.json",
            figure_specs_output=generated_dir / "figure_specs.json",
            generated_dir=generated_dir,
            dry_run=False,
        )
    )

    visual_plan = json.loads((deck_dir / "visual_plan.json").read_text(encoding="utf-8"))
    figure = visual_plan["slides"][0]["figures"][0]
    assert figure["figure_design_brief"]
    assert figure["depiction_type"] == "mechanism"
    assert figure["visual_concept"]
    assert figure["composition_budget"]["dominant_flow_direction"] == "left_to_right"
    assert figure["required_entities"]
    assert figure["required_relationships"]
    assert figure["source_labels"] == ["SRC-001", "SRC-002"]
    assert figure["source_ids"] == [
        "speech_transcript:aiwild:segment:00001",
        "speech_transcript:aiwild:segment:00054",
    ]
    assert "Don Song" in figure["contribution_note"]
    assert "Bo Li" in figure["contribution_note"]
    # Entities are real named people/works, not topic-keyword scaffold labels.
    assert "don song" in figure["narration_hook"].lower()
    assert "bullet-box" in figure["forbidden_patterns"]

    specs_payload = json.loads((generated_dir / "figure_specs.json").read_text(encoding="utf-8"))
    spec = specs_payload["figure_specs"][0]
    assert spec["source_ids"] == figure["source_ids"]
    assert spec["required_entities"] == figure["required_entities"]
    assert spec["required_relationships"] == figure["required_relationships"]
    assert spec["visual_concept"] == figure["visual_concept"]
    assert spec["composition_budget"] == figure["composition_budget"]

    inventory = json.loads((deck_dir / "asset_inventory.json").read_text(encoding="utf-8"))
    asset = inventory["generated_assets"][0]
    assert payload["status"] == "succeeded"
    assert asset["production_evidence"] is True
    assert len(asset["sha256"]) == 64
    assert asset["semantic_review_ref"] == "assets/generated/semantic_review.json"
    assert asset["model_governance"]["status"] == "deterministic_local_no_provider_call"


def test_generate_programmatic_figures_requires_opus_for_phase16(tmp_path: Path) -> None:
    workspace, deck_dir = write_structured_deck_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/generate_programmatic_figures.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "04_safety_alignment_privacy",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "04_safety_alignment_privacy",
            "--deck-contract",
            str(deck_dir / "presentation_deck.json"),
            "--source-binding-report",
            str(deck_dir / "source_binding_report.json"),
            "--visual-plan",
            str(deck_dir / "visual_plan.json"),
            "--asset-inventory",
            str(deck_dir / "asset_inventory.json"),
            "--generated-dir",
            str(deck_dir / "assets" / "generated" / "phase16"),
            "--svg-stack-research",
            str(REPO_ROOT / "pyproject.toml"),
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-6",
            "--provider-cost-approved",
            "--allowed-renderer",
            "drawsvg",
            "--disallowed-default-renderer",
            "matplotlib",
            "--disallowed-default-renderer",
            "plotly",
            "--figure-code-dir",
            str(deck_dir / "assets" / "generated" / "phase16" / "code"),
            "--figure-specs-output",
            str(deck_dir / "figure_specs.json"),
            "--model-receipt-output",
            str(deck_dir / "figure_generation_model_receipt.json"),
            "--code-review-output",
            str(deck_dir / "figure_code_review_report.json"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert _opus_model() in result.stderr


def test_generate_programmatic_figures_blocks_matplotlib_plotly_default_route(
    tmp_path: Path,
) -> None:
    workspace, deck_dir = write_structured_deck_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/generate_programmatic_figures.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "04_safety_alignment_privacy",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "04_safety_alignment_privacy",
            "--deck-contract",
            str(deck_dir / "presentation_deck.json"),
            "--source-binding-report",
            str(deck_dir / "source_binding_report.json"),
            "--visual-plan",
            str(deck_dir / "visual_plan.json"),
            "--asset-inventory",
            str(deck_dir / "asset_inventory.json"),
            "--generated-dir",
            str(deck_dir / "assets" / "generated" / "phase16"),
            "--svg-stack-research",
            str(REPO_ROOT / "pyproject.toml"),
            "--provider",
            "anthropic",
            "--model",
            _opus_model(),
            "--provider-cost-approved",
            "--allowed-renderer",
            "plotly",
            "--disallowed-default-renderer",
            "matplotlib",
            "--disallowed-default-renderer",
            "plotly",
            "--figure-code-dir",
            str(deck_dir / "assets" / "generated" / "phase16" / "code"),
            "--figure-specs-output",
            str(deck_dir / "figure_specs.json"),
            "--model-receipt-output",
            str(deck_dir / "figure_generation_model_receipt.json"),
            "--code-review-output",
            str(deck_dir / "figure_code_review_report.json"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "matplotlib and plotly" in result.stderr


def test_figure_specs_input_reuses_live_receipt_and_cleans_stale_svgs(tmp_path: Path) -> None:
    workspace, deck_dir = write_deck_fixture(
        tmp_path,
        """## Slide 1: Source-Grounded Figure

- The figure should be rerenderable from a prior live spec receipt.

<!--
FIGURE_NEED: hub diagram showing source-grounded FigureSpec reuse
-->
""",
    )

    generate_programmatic_figures(
        workspace,
        "01_demo_topic",
        input_slides=deck_dir / "3.5_draft_slides_consolidated.md",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        output_slides=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=deck_dir / "assets" / "generated",
    )

    generated_dir = deck_dir / "assets" / "generated"
    specs_path = generated_dir / "figure_specs.json"
    fake_receipt = {
        "schema_version": 1,
        "task_id": "presentation.svg_figure_generation",
        "provider": "anthropic",
        "model_id": APPROVED_SVG_CODEGEN_MODEL,
        "status": "complete",
        "request_count": 1,
        "figure_count": 1,
    }
    specs_payload = json.loads(specs_path.read_text(encoding="utf-8"))
    specs_payload["sonnet_figure_spec_receipt"] = fake_receipt
    specs_path.write_text(json.dumps(specs_payload), encoding="utf-8")
    (generated_dir / "sonnet_figure_spec_receipt.json").write_text(
        json.dumps(fake_receipt),
        encoding="utf-8",
    )
    stale_svg = generated_dir / "slide_999_stale.svg"
    stale_svg.write_text("<svg><text>stale</text></svg>", encoding="utf-8")

    inventory = generate_programmatic_figures(
        workspace,
        "01_demo_topic",
        input_slides=deck_dir / "3.5_draft_slides_consolidated.md",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        output_slides=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=generated_dir,
        figure_specs_input=specs_path,
        figure_specs_output=specs_path,
    )

    assert not stale_svg.exists()
    assert len(list(generated_dir.glob("slide_*.svg"))) == 1
    assert inventory["model_governance"]["status"] == "live_sonnet_specs_reused_local_render"
    assert inventory["sonnet_figure_spec_receipt"] == fake_receipt
    rewritten_specs = json.loads(specs_path.read_text(encoding="utf-8"))
    assert rewritten_specs["sonnet_figure_spec_receipt"] == fake_receipt
    assert any(record["kind"] == "figure_specs_input" for record in inventory["inputs"])


def test_programmatic_figures_records_waiver_when_need_cannot_be_generated(
    tmp_path: Path,
) -> None:
    workspace, deck_dir = write_deck_fixture(
        tmp_path,
        """## Slide 1: Unsupported Visual

- Needs manual design.

<!--
FIGURE_NEED: unsupported photorealistic reconstruction of private lab footage
-->
""",
    )

    inventory = generate_programmatic_figures(
        workspace,
        "01_demo_topic",
        input_slides=deck_dir / "3.5_draft_slides_consolidated.md",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        output_slides=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=deck_dir / "assets" / "generated",
        max_waivers=1,
    )

    assert inventory["waivers"]
    assert inventory["waivers"][0]["slide_number"] == 1
    assert inventory["waivers"][0]["reason"]


def test_attack_surface_need_routes_to_risk_visual_not_surface(tmp_path: Path) -> None:
    workspace, deck_dir = write_deck_fixture(
        tmp_path,
        """## Slide 3: Agentic Systems: The New Threat Landscape

- Address the exponential increase in attack surfaces when models autonomously interact with external APIs.

<!--
FIGURE_NEED: Diagram showing the expanded attack surface of an agentic system
-->
""",
    )

    inventory = generate_programmatic_figures(
        workspace,
        "01_demo_topic",
        input_slides=deck_dir / "3.5_draft_slides_consolidated.md",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        output_slides=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=deck_dir / "assets" / "generated",
    )

    assert inventory["figures"][0]["figure_type"] == "risk_surface"
    assert inventory["figures"][0]["figure_spec"]["depiction_type"] == "risk_surface"
    svg_text = next((deck_dir / "assets" / "generated").glob("slide_003_*.svg")).read_text(
        encoding="utf-8"
    )
    assert "Search landscape: promising regions emerge" not in svg_text
    assert "Attack Surface" in svg_text


def test_programmatic_figures_use_ranked_papers_for_figure_spec(tmp_path: Path) -> None:
    workspace, deck_dir = write_deck_fixture(
        tmp_path,
        """## Slide 8: PySpatial: LLM-Driven Spatial Reasoning

- Translate visual and geometric problems into programmatic logic.

<!--
FIGURE_NEED: Workflow diagram showing visual input translated to Python code for spatial reasoning.
-->
""",
    )
    write_ranked_papers(
        deck_dir,
        [
            {
                "rank": 1,
                "title": "PySpatial: LLM-Driven Spatial Reasoning via Code Generation",
                "paper_id": "paper-1",
                "url": "https://openreview.net/pdf/paper-1.pdf",
                "evidence_snippet": "Uses PySpatial API to solve spatial reasoning by generating Python code.",
            }
        ],
    )

    inventory = generate_programmatic_figures(
        workspace,
        "01_demo_topic",
        input_slides=deck_dir / "3.5_draft_slides_consolidated.md",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        output_slides=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=deck_dir / "assets" / "generated",
    )

    spec = inventory["figures"][0]["figure_spec"]
    assert spec["model_id"] == APPROVED_SVG_CODEGEN_MODEL
    assert spec["depiction_type"] == "code_pipeline"
    assert "PySpatial" in json.dumps(spec)
    svg_text = next((deck_dir / "assets" / "generated").glob("slide_008_*.svg")).read_text(
        encoding="utf-8"
    )
    assert "Python program" in svg_text
    assert "PySpatial" in svg_text


def test_density_fallback_is_disabled_by_default_for_production(tmp_path: Path) -> None:
    workspace, deck_dir = write_deck_fixture(
        tmp_path,
        """## Slide 1: No Explicit Figure

- This slide has no explicit figure need.

---

## Slide 2: Still No Figure

- Another slide without a figure marker.
""",
    )

    inventory = generate_programmatic_figures(
        workspace,
        "01_demo_topic",
        input_slides=deck_dir / "3.5_draft_slides_consolidated.md",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        output_slides=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=deck_dir / "assets" / "generated",
        target_svgs_per_slide=1.0,
    )

    assert inventory["figures"] == []
    assert inventory["density_fallback"]["enabled"] is False


def test_semantic_review_fails_generic_placeholder_svg(tmp_path: Path) -> None:
    workspace, deck_dir = write_deck_fixture(
        tmp_path,
        """## Slide 1: Placeholder

- Placeholder.

<!--
FIGURE_NEED: 3D rendering of a protein-ligand binding pocket.
-->
""",
    )
    generated = deck_dir / "assets" / "generated"
    generated.mkdir(parents=True)
    svg_path = generated / "slide_001_placeholder.svg"
    svg_path.write_text(
        """<svg viewBox="0 0 1280 720"><text>Spatial evidence model</text></svg>""",
        encoding="utf-8",
    )
    inventory = {
        "figures": [
            {
                "slide_number": 1,
                "slide_title": "Placeholder",
                "need": "3D rendering of a protein-ligand binding pocket.",
                "path": "decks/01_demo_topic/assets/generated/slide_001_placeholder.svg",
                "figure_spec": {
                    "figure_need": "3D rendering of a protein-ligand binding pocket.",
                    "slide_title": "Placeholder",
                    "depiction_type": "molecular_scene",
                    "entities": ["Protein pocket", "Ligand"],
                    "quantities": [],
                },
            }
        ]
    }
    (deck_dir / "asset_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")

    review = review_programmatic_figures(
        workspace,
        generated_dir=generated,
        asset_inventory=deck_dir / "asset_inventory.json",
        report_output=generated / "semantic_review.json",
        markdown_output=generated / "semantic_review.md",
    )

    assert review["status"] == "failed"
    check_ids = {issue["check_id"] for issue in review["issues"]}
    assert "generic_template_phrase" in check_ids
    assert "required_entity_coverage" in check_ids


def test_semantic_review_fails_placeholder_chart_ticks(tmp_path: Path) -> None:
    workspace, deck_dir = write_deck_fixture(
        tmp_path,
        """## Slide 1: Scaling

- Scaling slide.

<!--
FIGURE_NEED: chart showing scaling to 12M tokens on GPU.
-->
""",
    )
    generated = deck_dir / "assets" / "generated"
    generated.mkdir(parents=True)
    svg_path = generated / "slide_001_scaling.svg"
    svg_path.write_text(
        """<svg viewBox="0 0 1280 720"><text>4K</text><text>16K</text><text>128K</text></svg>""",
        encoding="utf-8",
    )
    inventory = {
        "figures": [
            {
                "slide_number": 1,
                "slide_title": "Scaling",
                "need": "chart showing scaling to 12M tokens on GPU.",
                "path": "decks/01_demo_topic/assets/generated/slide_001_scaling.svg",
                "figure_spec": {
                    "figure_need": "chart showing scaling to 12M tokens on GPU.",
                    "slide_title": "Scaling",
                    "depiction_type": "scaling_chart",
                    "entities": ["GPU"],
                    "quantities": ["12M tokens"],
                },
            }
        ]
    }
    (deck_dir / "asset_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")

    review = review_programmatic_figures(
        workspace,
        generated_dir=generated,
        asset_inventory=deck_dir / "asset_inventory.json",
        report_output=generated / "semantic_review.json",
    )

    assert review["status"] == "failed"
    assert any(issue["check_id"] == "placeholder_chart_ticks" for issue in review["issues"])


def test_semantic_review_blocks_weak_structured_figures(tmp_path: Path) -> None:
    workspace, deck_dir = write_structured_deck_fixture(tmp_path)
    generated_dir = deck_dir / "assets" / "generated"
    generated_dir.mkdir(parents=True)
    svg_path = generated_dir / "weak.svg"
    svg_path.write_text(
        """<svg viewBox="0 0 1280 720">
<text>Decorative summary</text>
<text>Safety moved from refusal to deployed-system trust</text>
<text>Agents need oversight</text>
</svg>""",
        encoding="utf-8",
    )
    inventory = {
        "schema_version": 1,
        "presentation_id": "04_safety_alignment_privacy",
        "generated_assets": [
            {
                "path": "decks/04_safety_alignment_privacy/assets/generated/weak.svg",
                "slide_id": "sl_01_agent_safety",
                "visual_id": "sl_01_agent_safety_visual",
                "renderer_family": "direct_svg",
                "production_evidence": True,
                "sha256": "0" * 64,
                "semantic_review_ref": "assets/generated/semantic_review.json",
            }
        ],
        "figure_specs": [
            {
                "slide_id": "sl_01_agent_safety",
                "visual_id": "sl_01_agent_safety_visual",
                "figure_design_brief": "Decorative summary card",
                "depiction_type": "decorative",
                "renderer_family": "direct_svg",
                "required_entities": ["Agent loop", "Tool memory", "Oversight gate"],
                "required_relationships": ["tools -> oversight"],
                "source_labels": [],
                "source_ids": [],
                "contribution_note": "Don Song grounds the deployed-system safety claim.",
                "narration_hook": "Explain agent tools, memory, and oversight.",
                "forbidden_patterns": ["decorative", "bullet-box"],
            }
        ],
    }
    (deck_dir / "asset_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")

    review = review_programmatic_figure_inventory(
        workspace,
        generated_dir=generated_dir,
        asset_inventory=deck_dir / "asset_inventory.json",
        report_output=generated_dir / "semantic_review.json",
        markdown_output=generated_dir / "semantic_review.md",
    )

    assert review["status"] == "fail"
    check_ids = {finding["check_id"] for finding in review["findings"]}
    assert "decorative_visual" in check_ids
    assert "missing_source" in check_ids
    assert "bullet-box" in check_ids
    assert "visual/narration mismatch" in check_ids


def test_surface_and_spatial_needs_are_not_the_same_template() -> None:
    assert (
        _figure_type(
            "3D loss landscape visualization comparing sharp vs flat minima",
            slide_title="Generalization Geometry",
        )
        == "surface"
    )
    assert (
        _figure_type(
            "3D rendering of a protein-ligand binding pocket",
            slide_title="Molecular Geometry",
        )
        == "spatial"
    )


def test_spatial_renderer_uses_complete_callout_copy(tmp_path: Path) -> None:
    workspace, deck_dir = write_deck_fixture(
        tmp_path,
        """## Slide 32: Explicit Structure as Dominant Remedy

- Identify explicit spatial structure modeling as the primary solution to multimodal spatial reasoning failures.

<!--
FIGURE_NEED: Comparison diagram: Implicit 2D learning vs. Explicit 3D structure modeling.
-->
""",
    )

    generate_programmatic_figures(
        workspace,
        "01_demo_topic",
        input_slides=deck_dir / "3.5_draft_slides_consolidated.md",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        output_slides=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=deck_dir / "assets" / "generated",
    )

    svg_text = next((deck_dir / "assets" / "generated").glob("slide_032_*.svg")).read_text(
        encoding="utf-8"
    )
    assert "primary solution to</text>" not in svg_text
    assert "3D" in svg_text
    assert "Explicit" in svg_text


def test_figure_design_brief_carries_blueprint_metadata(tmp_path: Path) -> None:
    workspace, deck_dir = write_deck_fixture(
        tmp_path,
        """## Slide 1: Blueprint Visual

- The figure should visualize the approved slide claim.

<!--
BLUEPRINT_ID: bp-demo-001
BLUEPRINT_SLIDE_ID: slide-001
BLUEPRINT_CLUSTER_ID: cluster-001
APPROVED_SOURCE_LABELS: SRC-001, SRC-002
VISUAL_ROLE: Blueprint dependency map linking evidence to narration
NARRATION_BEAT: The audience sees how the source plan drives the visual.
FIGURE_NEED: dependency map linking source evidence to slide narration
-->
""",
    )

    inventory = generate_programmatic_figures(
        workspace,
        "01_demo_topic",
        input_slides=deck_dir / "3.5_draft_slides_consolidated.md",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        output_slides=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=deck_dir / "assets" / "generated",
        dry_run=True,
    )

    brief = inventory["figures"][0]["figure_design_brief"]
    assert brief["blueprint_id"] == "bp-demo-001"
    assert brief["slide_id"] == "slide-001"
    assert brief["cluster_id"] == "cluster-001"
    assert brief["source_labels"] == ["SRC-001", "SRC-002"]
    assert brief["narration_hook"] == "The audience sees how the source plan drives the visual."


def test_renderer_registry_accepts_matplotlib_and_plotly_only_for_quantitative_charts() -> None:
    base_brief = {
        "blueprint_id": "bp-demo-001",
        "slide_id": "slide-001",
        "cluster_id": "cluster-001",
        "source_labels": ["SRC-001"],
        "claim_to_visualize": "Scaling improves when context grows.",
        "depiction_type": "scaling_chart",
        "required_labels": ["tokens", "latency"],
        "forbidden_patterns": ["fake chart ticks"],
        "narration_hook": "Call out the measured scaling relationship.",
    }

    assert validate_figure_design_brief(
        {**base_brief, "renderer_family": "matplotlib", "quantitative_chart": True}
    ) == ()
    assert validate_figure_design_brief(
        {**base_brief, "renderer_family": "plotly", "quantitative_chart": True}
    ) == ()
    mismatch = validate_figure_design_brief(
        {**base_brief, "renderer_family": "plotly", "quantitative_chart": False}
    )
    assert any(error["code"] == FIGURE_RENDERER_MISMATCH for error in mismatch)


def test_renderer_registry_rejects_unsupported_domain_renderers() -> None:
    errors = validate_figure_design_brief(
        {
            "blueprint_id": "bp-demo-001",
            "slide_id": "slide-001",
            "cluster_id": "cluster-001",
            "source_labels": ["SRC-001"],
            "claim_to_visualize": "Molecular fit controls binding.",
            "depiction_type": "molecular_scene",
            "renderer_family": "molecule_protein",
            "required_labels": ["ligand", "pocket"],
            "forbidden_patterns": ["decorative molecule"],
            "narration_hook": "Explain the pocket-to-ligand geometry.",
        }
    )

    assert any(error["code"] == UNSUPPORTED_IN_REGISTRY_V1 for error in errors)
    assert "unsupported_in_registry_v1" in json.dumps(errors)


def test_phase16_required_entity_coverage_mirrors_semantic_reviewer(tmp_path: Path) -> None:
    """A rendered SVG missing a required entity label is caught inside the
    codegen attempt loop with the same presence rule the post-hoc semantic
    reviewer applies (exact phrase, all distinctive tokens, or >= 60% of them
    for 3+-token entities), so the redraw happens before the review stage."""

    from scripts.presentations.generate_programmatic_figures import _phase16_coverage_issues

    svg = tmp_path / "figure.svg"
    svg.write_text(
        "<svg><text>Closed checkpoint</text><text>baseline column</text>"
        "<text>Molmo2 &amp; OpenMMReasoner</text><text>Christopher Clark, Jieyu Zhang</text></svg>",
        encoding="utf-8",
    )
    row = {
        "asset_path": svg,
        "required_entities": [
            "Closed checkpoint (baseline column)",  # tokens split across text nodes -> present
            "Molmo2",
            "Rohun Tripathi",  # absent
            "Open curation recipe audit trail",  # 0/5 tokens -> absent
        ],
    }
    issues = _phase16_coverage_issues(row)
    assert [issue.split("'")[1] for issue in issues] == ["Rohun Tripathi", "Open curation recipe audit trail"]
    assert _phase16_coverage_issues({"asset_path": tmp_path / "nope.svg", "required_entities": ["x"]}) == []


def test_renderer_registry_v2_rejects_empty_source_grounding() -> None:
    brief = {
        "blueprint_id": "",
        "slide_id": "",
        "cluster_id": "",
        "source_labels": [],
        "claim_to_visualize": "",
        "depiction_type": "",
        "renderer_family": "vision_redraw_svg",
        "required_labels": [],
        "forbidden_patterns": [],
        "narration_hook": "",
        "grounding": {
            "openreview_id": "",
            "paper_title": "",
            "page": "",
            "bbox": [],
            "caption": "",
            "crop_path": "",
            "crop_sha256": "",
            "license": "",
            "grounding_mode": "",
        },
    }

    errors = validate_figure_design_brief_v2(brief)

    assert errors
    assert {
        error["path"] for error in errors
    } >= {
        "$.figure_design_brief.source_labels",
        "$.figure_design_brief.required_labels",
        "$.figure_design_brief.grounding.openreview_id",
        "$.figure_design_brief.grounding.crop_sha256",
        "$.figure_design_brief.grounding.grounding_mode",
    }


def test_renderer_registry_v1_preserves_legacy_placeholder_contract() -> None:
    brief = {
        "blueprint_id": "",
        "slide_id": "",
        "cluster_id": "",
        "source_labels": [],
        "claim_to_visualize": "",
        "depiction_type": "",
        "renderer_family": "direct_svg",
        "required_labels": [],
        "forbidden_patterns": [],
        "narration_hook": "",
    }

    assert validate_figure_design_brief(brief) == ()


def test_renderer_registry_v2_accepts_complete_grounded_redraw(tmp_path: Path) -> None:
    crop_path = tmp_path / "source-crop.png"
    crop_bytes = b"source-grounded-crop"
    crop_path.write_bytes(crop_bytes)
    brief = {
        "blueprint_id": "bp-demo-001",
        "slide_id": "slide-001",
        "cluster_id": "cluster-001",
        "source_labels": ["SRC-001"],
        "claim_to_visualize": "The method preserves the paper's mechanism.",
        "depiction_type": "mechanism_map",
        "renderer_family": "vision_redraw_svg",
        "required_labels": ["input", "output"],
        "forbidden_patterns": ["unlabeled arrows"],
        "narration_hook": "Explain the source-backed mechanism.",
        "grounding": {
            "openreview_id": "noteAAAA01",
            "paper_title": "Source Grounding Paper",
            "page": 2,
            "bbox": [0.0, 0.0, 10.0, 5.0],
            "caption": "Figure 2: grounded mechanism.",
            "crop_path": str(crop_path),
            "crop_sha256": hashlib.sha256(crop_bytes).hexdigest(),
            "license": "CC BY 4.0",
            "grounding_mode": "vision_redraw",
        },
    }

    assert validate_figure_design_brief_v2(brief) == ()
