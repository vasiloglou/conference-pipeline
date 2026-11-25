"""Synthetic presentation workspace helpers for Phase 25 validation tests."""

from __future__ import annotations

import json
import re
import hashlib
from io import BytesIO
from pathlib import Path

from conference_pipeline import workflow
from conference_pipeline.presentation.manifest import (
    LegacyMarkdownCompatibility,
    PRESENTATION_STAGE_IDS,
    PresentationArtifactRef,
    PresentationEntry,
    PresentationStageState,
    PresentationTopicScope,
    _compute_content_sha256,
    emit_presentation_manifest,
)
from conference_pipeline.presentation.blueprint import write_blueprint_receipt


TOPIC_ID = "01_demo_topic"
_APPROVED_HUMAN_REVIEW_STATUSES = {"accepted", "approved"}
_APPROVED_HUMAN_REVIEW_DECISIONS = {"approved_for_bulk_rollout"}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def minimal_pdf(page_count: int) -> bytes:
    """Return a tiny valid PDF with the requested number of blank pages."""

    if page_count < 1:
        raise ValueError("page_count must be positive")
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=640, height=360)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def _topic_dir(workspace: workflow.WorkflowWorkspace) -> Path:
    return workspace.root / "decks" / TOPIC_ID


def _artifact(
    workspace: workflow.WorkflowWorkspace,
    relative_path: str,
    *,
    kind: str = "file",
) -> PresentationArtifactRef:
    return PresentationArtifactRef.from_path(
        workspace.root / relative_path,
        workspace_root=workspace.root,
        kind=kind,
    )


def _stage_state(
    workspace: workflow.WorkflowWorkspace,
    status: str,
    *paths: str,
    kind: str = "file",
) -> PresentationStageState:
    return PresentationStageState(
        status=status,
        artifacts=tuple(_artifact(workspace, path, kind=kind) for path in paths),
        notes=(),
    )


def _human_review_stage_state(
    workspace: workflow.WorkflowWorkspace,
    receipt_path: Path,
) -> PresentationStageState:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        approved = False
    else:
        approved = isinstance(receipt, dict) and (
            str(receipt.get("status", "")).strip().lower()
            in _APPROVED_HUMAN_REVIEW_STATUSES
            or str(receipt.get("decision", "")).strip().lower()
            in _APPROVED_HUMAN_REVIEW_DECISIONS
        )
    return _stage_state(
        workspace,
        "complete" if approved else "partial",
        f"decks/{TOPIC_ID}/human_review_receipt.json",
    )


def _presentation_entry(workspace: workflow.WorkflowWorkspace) -> PresentationEntry:
    stage_states = {
        stage_id: PresentationStageState(status="missing", artifacts=(), notes=())
        for stage_id in PRESENTATION_STAGE_IDS
    }
    legacy_paths = [
        f"decks/{TOPIC_ID}/1_draft_slides.md",
        f"decks/{TOPIC_ID}/2_draft_slides_enriched_with_papers.md",
        f"decks/{TOPIC_ID}/4.5_draft_slides_with_hyperlinks.md",
        f"decks/{TOPIC_ID}/5_draft_slides_with_images.md",
        f"decks/{TOPIC_ID}/presentation_slide_mapping.json",
        f"decks/{TOPIC_ID}/assets/generated/validation_report.md",
        f"decks/{TOPIC_ID}/presentation.html",
        f"decks/{TOPIC_ID}/6_presenter_script.md",
        f"decks/{TOPIC_ID}/presentation.pdf",
    ]
    if (_topic_dir(workspace) / "assets" / "generated" / "slide_01.svg").is_file():
        legacy_paths.append(f"decks/{TOPIC_ID}/assets/generated/slide_01.svg")
    video_generation_artifacts: list[PresentationArtifactRef] = []
    for relative_path, kind in (
        (f"decks/{TOPIC_ID}/video_build", "directory"),
        (f"decks/{TOPIC_ID}/video_build_notebooklm", "directory"),
        (f"decks/{TOPIC_ID}/presentation_video.mp4", "file"),
        (f"decks/{TOPIC_ID}/presentation_video_notebooklm.mp4", "file"),
    ):
        if (workspace.root / relative_path).exists():
            video_generation_artifacts.append(_artifact(workspace, relative_path, kind=kind))
    stage_states.update(
        {
            "presentation_blueprint": _stage_state(
                workspace,
                "complete",
                f"decks/{TOPIC_ID}/presentation_blueprint.json",
                f"decks/{TOPIC_ID}/presentation_blueprint.md",
                f"decks/{TOPIC_ID}/presentation_blueprint_receipt.json",
                f"decks/{TOPIC_ID}/blueprint_review.json",
                f"decks/{TOPIC_ID}/blueprint_review.md",
            ),
            "visual_qc": _stage_state(
                workspace,
                "complete",
                f"decks/{TOPIC_ID}/assets/generated/visual_qc_report.md",
                f"decks/{TOPIC_ID}/assets/generated/visual_llm_qc_report.json",
            ),
            "video_generation": PresentationStageState(
                status="complete",
                artifacts=tuple(video_generation_artifacts),
                notes=(),
            ),
        }
    )
    receipt_path = workspace.root / "decks" / TOPIC_ID / "human_review_receipt.json"
    if receipt_path.is_file():
        stage_states["human_review"] = _human_review_stage_state(workspace, receipt_path)
    return PresentationEntry(
        id=TOPIC_ID,
        presentation_type="topic-deep-dive",
        title="Demo Topic",
        topic_scope=PresentationTopicScope(mode="single-topic", topics=(TOPIC_ID,)),
        target_duration_minutes=12,
        target_slide_count=3,
        source_analysis_artifacts=(
            _artifact(workspace, f"analysis/{TOPIC_ID}.md"),
        ),
        stage_states=stage_states,
        legacy_markdown=LegacyMarkdownCompatibility(
            support_level="legacy-mock-only",
            production_evidence=False,
            artifacts=tuple(
                _artifact(workspace, relative_path)
                for relative_path in legacy_paths
                if (workspace.root / relative_path).is_file()
            ),
            notes=(
                "Legacy validation fixture artifacts are non-production compatibility evidence only.",
            ),
        ),
    )


def _emit_manifest(workspace: workflow.WorkflowWorkspace) -> None:
    emit_presentation_manifest(
        workspace,
        [_presentation_entry(workspace)],
        generator="tests.presentation_validation@dev",
    )


def _refresh_manifest_content_sha(workspace: workflow.WorkflowWorkspace) -> None:
    manifest_path = workspace.root / "decks" / "presentation_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["content_sha256"] = _compute_content_sha256(data)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_validation_workspace(tmp_path: Path) -> workflow.WorkflowWorkspace:
    """Create a minimal manifest-registered presentation workspace."""

    workspace_path = tmp_path / "conferences" / "demo2026"
    _write_text(
        workspace_path / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
        ),
    )
    _write_text(
        workspace_path / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    _write_text(workspace_path / "analysis" / f"{TOPIC_ID}.md", "# Demo Topic\n\nStable body.\n")

    topic_dir = workspace_path / "decks" / TOPIC_ID
    blueprint = {
        "audience_promise": "Operators understand the validation contract.",
        "blueprint_id": "bp-validation-fixture",
        "conclusion_synthesis": "Blueprint validation precedes draft generation.",
        "deck_thesis": "Blueprint artifacts gate presentation draft generation.",
        "provenance": {
            "source_metadata": [
                {
                    "label": "SRC-001",
                    "path": f"analysis/{TOPIC_ID}.md",
                }
            ]
        },
        "revision": 1,
        "slides": [
            {
                "estimated_minutes": 3,
                "intent": "Introduce the blueprint validation fixture.",
                "narration_beat": "The validation suite sees a complete approved blueprint.",
                "slide_id": "slide-001",
                "slide_type": "content",
                "source_plan": [
                    {
                        "label": "SRC-001",
                        "rationale": "The analysis fixture anchors this slide.",
                    }
                ],
                "visual_role": "Validation flow showing blueprint approval before draft generation",
            }
        ],
        "subtopic_clusters": [
            {
                "cluster_id": "cluster-001",
                "slide_bindings": ["slide-001"],
                "sources": ["SRC-001"],
                "title": "Blueprint validation",
            }
        ],
        "target_runtime_minutes": 12,
        "target_slide_count": 3,
        "telescopic_section_map": [
            {
                "cluster_ids": ["cluster-001"],
                "section_id": "section-001",
                "title": "Validation",
            }
        ],
    }
    blueprint_path = topic_dir / "presentation_blueprint.json"
    _write_text(blueprint_path, json.dumps(blueprint, indent=2, sort_keys=True) + "\n")
    blueprint_sha = hashlib.sha256(blueprint_path.read_bytes()).hexdigest()
    _write_text(topic_dir / "presentation_blueprint.md", "# Presentation Blueprint\n")
    _write_text(
        topic_dir / "blueprint_review.json",
        json.dumps(
            {
                "approval": {"status": "approved_for_draft_generation"},
                "approved_at": "2026-06-04T00:00:00Z",
                "approved_by": "validation-fixture",
                "blueprint_sha256": blueprint_sha,
                "gate_types": {
                    "LLM-judge": "pass",
                    "deterministic": "pass",
                    "human": "pass",
                },
                "review_artifact": f"decks/{TOPIC_ID}/blueprint_review.md",
                "status": "pass",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_text(topic_dir / "blueprint_review.md", "# Blueprint Review\n")
    write_blueprint_receipt(
        workspace_path,
        blueprint_path,
        receipt_path=topic_dir / "presentation_blueprint_receipt.json",
        presentation_id=TOPIC_ID,
        review_path=topic_dir / "blueprint_review.json",
    )
    _write_text(
        topic_dir / "1_draft_slides.md",
        "# Demo Presentation\n\n## Slide 1: Opening\n\nA.\n\n## Slide 2: Method\n\nB.\n\n## Slide 3: Close\n\nC.\n",
    )
    _write_text(topic_dir / "2_draft_slides_enriched_with_papers.md", "# Enriched\n")
    _write_text(topic_dir / "4.5_draft_slides_with_hyperlinks.md", "## Slide 1: Opening\n\nA.\n\n## Slide 2: Method\n\nB.\n\n## Slide 3: Close\n\nC.\n")
    _write_text(topic_dir / "5_draft_slides_with_images.md", "## Slide 1: Opening\n\nA.\n\n## Slide 2: Method\n\nB.\n\n## Slide 3: Close\n\nC.\n")
    _write_text(topic_dir / "presentation_slide_mapping.json", json.dumps({"1": 1, "2": 2, "3": 3}, indent=2) + "\n")
    _write_text(
        topic_dir / "presentation.html",
        (
            "<!doctype html><html><head>"
            '<meta name="presentation-template" content="relationalai-neurips2025-reference">'
            "<style>"
            ".reveal .slides section.slide-title::before {"
            "content: ''; background-image: url('data:image/png;base64,logo');"
            "}"
            ".reveal .slides section.slide-content::after {"
            "content: ''; background-image: url('data:image/png;base64,bug');"
            "}"
            ".reveal .slides section.slide-content-with-figure::after {"
            "content: ''; background-image: url('data:image/png;base64,bug');"
            "}"
            ".reveal .slides section.slide-figure-zoom::after {"
            "content: ''; background-image: url('data:image/png;base64,bug');"
            "}"
            "</style></head><body>"
            '<section class="slide-title">Demo Topic</section>'
            '<section data-markdown-slide="1" class="slide-content">One '
            '<div class="reference-chips"><span>Sources</span>'
            '<a href="https://example.com/demo">Demo Paper</a></div></section>'
            '<section data-markdown-slide="2" class="slide-content">Two</section>'
            '<section data-markdown-slide="3" class="slide-content-with-figure">Three</section>'
            "</body></html>\n"
        ),
    )
    _write_text(
        topic_dir / "6_presenter_script.md",
        "### [SLIDE 1] Opening\n\nSpeak.\n\n### [SLIDE 2] Method\n\nSpeak.\n\n### [SLIDE 3] Close\n\nSpeak.\n",
    )
    _write_text(topic_dir / "assets" / "generated" / "slide_01.svg", "<svg><title>Slide 1</title></svg>\n")
    _write_text(topic_dir / "assets" / "generated" / "visual_qc_report.md", "# Visual QC\n\nWarn only.\n")
    _write_text(
        topic_dir / "assets" / "generated" / "visual_llm_qc_report.json",
        '{"schema_version":1,"status":"passed","errors":[],"warnings":[],"slides":[]}\n',
    )
    _write_text(topic_dir / "assets" / "generated" / "validation_report.md", "# SVG validation\n\nOK.\n")
    _write_text(topic_dir / "video_build_notebooklm" / "metadata.json", '{"duration_seconds": 60}\n')
    _write_bytes(topic_dir / "presentation.pdf", minimal_pdf(3))

    workspace = workflow.resolve_workspace(workspace_path)
    _emit_manifest(workspace)
    return workspace


def break_mapping_count(
    workspace: workflow.WorkflowWorkspace,
    new_count: int = 2,
) -> None:
    mapping_path = _topic_dir(workspace) / "presentation_slide_mapping.json"
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    data = {str(index): index for index in range(1, new_count + 1)}
    mapping_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _emit_manifest(workspace)


def break_svg(
    workspace: workflow.WorkflowWorkspace,
    svg_name: str = "slide_01.svg",
) -> None:
    (_topic_dir(workspace) / "assets" / "generated" / svg_name).write_text(
        "<svg><broken></svg>\n",
        encoding="utf-8",
    )
    _emit_manifest(workspace)


def remove_generated_svgs(workspace: workflow.WorkflowWorkspace) -> None:
    """Remove declared generated SVGs and refresh the manifest."""

    generated_dir = _topic_dir(workspace) / "assets" / "generated"
    for svg_path in sorted(generated_dir.glob("*.svg")):
        svg_path.unlink()
    _emit_manifest(workspace)


def break_pdf(workspace: workflow.WorkflowWorkspace) -> None:
    (_topic_dir(workspace) / "presentation.pdf").write_bytes(b"not a pdf\n")
    _emit_manifest(workspace)


def rewrite_pdf_page_count(
    workspace: workflow.WorkflowWorkspace,
    page_count: int,
) -> None:
    (_topic_dir(workspace) / "presentation.pdf").write_bytes(minimal_pdf(page_count))
    _emit_manifest(workspace)


def add_video_artifacts(
    workspace: workflow.WorkflowWorkspace,
    *,
    duration_seconds: float = 120.0,
) -> None:
    topic_dir = _topic_dir(workspace)
    for filename in ("presentation_video.mp4", "presentation_video_notebooklm.mp4"):
        (topic_dir / filename).write_bytes(b"synthetic mp4 placeholder\n")
    half_duration = duration_seconds / 2
    for build_dir_name in ("video_build", "video_build_notebooklm"):
        build_dir = topic_dir / build_dir_name
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "narration_track.m4a").write_bytes(b"synthetic audio placeholder\n")
        (build_dir / "slides_concat.txt").write_text(
            (
                "file 'slide_001.mp4'\n"
                f"duration {half_duration:.3f}\n"
                "file 'slide_002.mp4'\n"
                f"duration {half_duration:.3f}\n"
            ),
            encoding="utf-8",
        )
        (build_dir / "metadata.json").write_text(
            json.dumps({"duration_seconds": duration_seconds}, indent=2) + "\n",
            encoding="utf-8",
        )
    _emit_manifest(workspace)


def add_human_review_receipt(
    workspace: workflow.WorkflowWorkspace,
    status: str = "accepted",
) -> None:
    (_topic_dir(workspace) / "human_review_receipt.json").write_text(
        json.dumps({"status": status}, indent=2) + "\n",
        encoding="utf-8",
    )
    _emit_manifest(workspace)


def remove_presenter_slide_id(
    workspace: workflow.WorkflowWorkspace,
    slide_num: int,
) -> None:
    script_path = _topic_dir(workspace) / "6_presenter_script.md"
    text = script_path.read_text(encoding="utf-8")
    text = re.sub(rf"^### \[SLIDE {slide_num}\].*\n", "", text, flags=re.MULTILINE)
    script_path.write_text(text, encoding="utf-8")
    _emit_manifest(workspace)


def add_zoom_companion(
    workspace: workflow.WorkflowWorkspace,
    slide_num: int,
) -> None:
    html_path = _topic_dir(workspace) / "presentation.html"
    text = html_path.read_text(encoding="utf-8")
    text = text.replace("</body>", f'<section data-visibility="uncounted">Zoom {slide_num}</section></body>')
    html_path.write_text(text, encoding="utf-8")
    script_path = _topic_dir(workspace) / "6_presenter_script.md"
    script_path.write_text(
        script_path.read_text(encoding="utf-8") + f"\n### [ZOOM {slide_num}] Detail\n\nZoom notes.\n",
        encoding="utf-8",
    )
    _emit_manifest(workspace)
