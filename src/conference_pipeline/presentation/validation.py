"""Structural validation of manifest-registered presentation artifacts.

Emits per-check records, one stage receipt per Phase 22 checkpoint, and a
per-run presentation_validation_report.json under storage/workflow-runs/<run-id>/.
Reads only artifacts declared by presentation_manifest.json. Does NOT import
or subprocess-call scripts/presentations/* (per CONTEXT.md D-10, extending
the Phase 24-03 argv/metadata-only rule to the validation harness).
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence

from conference_pipeline import workflow
from conference_pipeline.presentation.errors import PresentationValidationError
from conference_pipeline.presentation.video_delivery import (  # noqa: F401 -- re-exported names
    VIDEO_BLACK_TAIL_FAIL_SECONDS,
    VIDEO_BLACK_TAIL_LOOKBACK_SECONDS,
    VIDEO_MAX_DURATION_SECONDS,
    VIDEO_RECEIPT_DURATION_TOLERANCE_SECONDS,
    VIDEO_STREAM_DURATION_TOLERANCE_SECONDS,
    _BUILD_DURATION_JSON_KEYS,
    _VIDEO_SUFFIXES,
    _MediaDurations,
    _build_dir_for_video,
    _detect_black_tail_seconds,
    _duration_json_values,
    _expected_duration_sources,
    _mp4_duration_seconds,
    _parse_float,
    _parse_int,
    _probe_media_durations,
    _sum_concat_durations,
    black_tail_verdict,
    build_receipt_agreement_verdict,
    duration_budget_verdict,
    stream_alignment_verdict,
)
from conference_pipeline.presentation.blueprint import (
    BLUEPRINT_SOURCE_BINDING_MISSING,
    BLUEPRINT_SOURCE_UNRESOLVED,
    validate_blueprint_file,
    validate_blueprint_review_file,
)
from conference_pipeline.presentation.manifest import (
    MANIFEST_RELATIVE_PATH,
    LoadedPresentationManifest,
    PresentationArtifactRef,
    PresentationEntry,
    _relative_to_workspace,
    _short_commit,
    load_and_validate_manifest,
)
from conference_pipeline.workflow_runs import (
    workflow_run_dir,
    write_json_atomic,
)

VALIDATION_REPORT_FILENAME = "presentation_validation_report.json"
VALIDATION_REPORT_SCHEMA_VERSION = 1
CHECK_STATUS_VALUES = frozenset({"pass", "fail", "warn", "skipped_missing"})
CHECK_SEVERITY_VALUES = frozenset({"structural", "content"})

VALIDATION_STAGE_IDS: tuple[str, ...] = (
    "presentation_blueprint_validation",
    "deck_contract_generation_validation",
    "source_binding_validation",
    "visual_plan_generation_validation",
    "render_spec_generation_validation",
    "render_export_validation",
    "visual_qc_validation",
    "delivery_generation_validation",
    "video_generation_validation",
    "video_sync_validation_receipt",
    "human_review_validation",
    # Legacy/debug validators remain available for migration fixtures and
    # structural regression tests. They do not map to production stage proof.
    "slide_consolidation_validation",
    "research_enrichment_validation",
    "figures_structural_validation",
    "revealjs_render_validation",
    "presenter_script_validation",
    "pdf_export_validation",
)

VALIDATION_STAGE_TO_CHECKPOINT: dict[str, str] = {
    "presentation_blueprint_validation": "presentation_blueprint",
    "deck_contract_generation_validation": "deck_contract_generation",
    "source_binding_validation": "source_binding",
    "visual_plan_generation_validation": "visual_plan_generation",
    "render_spec_generation_validation": "render_spec_generation",
    "render_export_validation": "render_export",
    "visual_qc_validation": "visual_qc",
    "delivery_generation_validation": "delivery_generation",
    "video_generation_validation": "video_generation",
    "video_sync_validation_receipt": "video_sync_validation",
    "human_review_validation": "human_review",
    "slide_consolidation_validation": "render_export",
    "research_enrichment_validation": "source_binding",
    "figures_structural_validation": "visual_plan_generation",
    "revealjs_render_validation": "render_export",
    "presenter_script_validation": "delivery_generation",
    "pdf_export_validation": "render_export",
}
VALIDATION_STAGE_ID_SET = frozenset(VALIDATION_STAGE_IDS)
LEGACY_MARKDOWN_VALIDATION_STAGE_IDS = frozenset(
    {
        "slide_consolidation_validation",
        "research_enrichment_validation",
        "figures_structural_validation",
        "revealjs_render_validation",
        "presenter_script_validation",
        "pdf_export_validation",
    }
)
STRICT_REQUIRED_CHECKPOINT_STAGE_IDS: tuple[str, ...] = (
    "presentation_blueprint",
    "deck_contract_generation",
    "source_binding",
    "visual_plan_generation",
    "render_spec_generation",
    "render_export",
    "video_generation",
    "video_sync_validation",
    "visual_qc",
    "human_review",
    "figure_generation",
    "pdf_export",
    "presenter_script",
    "revealjs_render",
    "research_enrichment",
    "slide_consolidation",
)
STRICT_REQUIRED_CHECKPOINT_STAGE_ID_SET = frozenset(
    STRICT_REQUIRED_CHECKPOINT_STAGE_IDS
)

CHECK_IDS: tuple[str, ...] = (
    "blueprint_schema_valid",
    "blueprint_approved",
    "source_plan_coverage",
    "slide_mapping_schema",
    "slide_mapping_markdown_count_equals_mapping",
    "slide_mapping_revealjs_section_count_equals_mapping",
    "presenter_script_slide_ids_resolve",
    "hyperlink_density_meets_target",
    "hyperlink_preservation_4_5_to_5",
    "svg_density_meets_target",
    "svg_well_formed",
    "pdf_parses",
    "pdf_page_count_matches_target_slides",
    "video_artifacts_nonempty",
    "video_duration_readable",
    "video_duration_under_40_minutes",
    "video_stream_durations_match",
    "video_duration_matches_build_receipts",
    "video_tail_not_black",
    "visual_qc_report_present",
    "vision_llm_visual_qc_passed",
    "revealjs_html_parses",
    "revealjs_reference_labels_visible",
    "revealjs_title_slide_branded",
    "revealjs_content_brand_mark_present",
)

# Video timing constants and media probes live in presentation.video_delivery
# (imported above) so the shared production chain can run them path-based.

_SLIDE_HEADER_RE = re.compile(r"^## Slide (\d+):", re.MULTILINE)
_PRESENTER_HEADER_RE = re.compile(r"^### \[(?:SLIDE|ZOOM) (\d+)\]", re.MULTILINE)
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\((https?://[^)\s]+)[^)]*\)")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_REFERENCE_CHIPS_BLOCK_RE = re.compile(
    r'<div\b[^>]*class=["\'][^"\']*\breference-chips\b[^"\']*["\'][^>]*>(?P<body>.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)
_REFERENCE_CHIP_LABEL_RE = re.compile(
    r"<(?:a|em)\b[^>]*>(?P<label>.*?)</(?:a|em)>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_GENERIC_REFERENCE_LABEL_RE = re.compile(
    r"^\s*(?:ref(?:erence)?\.?\s*\d*|source\s*\d*)\s*$",
    re.IGNORECASE,
)
_REFERENCE_LABEL_BAD_SUFFIXES = frozenset(
    {"a", "an", "and", "at", "for", "from", "how", "of", "the", "to", "via", "with"}
)


@dataclass(frozen=True)
class PresentationCheckRecord:
    """One machine-checkable structural or content check against a presentation."""

    check_id: str
    status: str
    severity: str
    message: str
    evidence_paths: tuple[str, ...]
    validation_stage_id: str
    checkpoint_stage_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "evidence_paths": list(self.evidence_paths),
            "validation_stage_id": self.validation_stage_id,
            "checkpoint_stage_id": self.checkpoint_stage_id,
        }


class _RevealSectionCounter(HTMLParser):
    """Counts total sections and distinct data-markdown-slide values."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.total_sections = 0
        self.distinct_markdown_slides: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "section":
            return
        self.total_sections += 1
        for name, value in attrs:
            if name == "data-markdown-slide" and value is not None:
                self.distinct_markdown_slides.add(value)


def _count_reveal_sections(html_path: Path) -> tuple[int, int]:
    parser = _RevealSectionCounter()
    parser.feed(html_path.read_text(encoding="utf-8"))
    parser.close()
    return parser.total_sections, len(parser.distinct_markdown_slides)


def _markdown_slide_numbers(md_path: Path) -> set[int]:
    text = md_path.read_text(encoding="utf-8")
    return {int(match.group(1)) for match in _SLIDE_HEADER_RE.finditer(text)}


def _markdown_slide_count(md_path: Path) -> int:
    return len(_markdown_slide_numbers(md_path))


def _markdown_svg_density_slide_count(md_path: Path) -> int:
    """Count slides eligible for SVG density checks.

    Title slides are inserted and visually owned by the title-slide stage, while
    figure generation is responsible for the body/section deck. Counting title
    slides here can make a deck fail density despite satisfying the figure
    generation contract.
    """

    text = md_path.read_text(encoding="utf-8")
    matches = list(_SLIDE_HEADER_RE.finditer(text))
    count = 0
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        if "SLIDE_TYPE: title" in block:
            continue
        count += 1
    return count


def _markdown_hyperlink_count(md_path: Path) -> int:
    text = md_path.read_text(encoding="utf-8")
    return len(_MARKDOWN_LINK_RE.findall(text))


def _markdown_svg_reference_count(md_path: Path) -> int:
    text = md_path.read_text(encoding="utf-8")
    total = 0
    for match in _MARKDOWN_IMAGE_RE.finditer(text):
        raw_target = match.group(1).strip()
        normalized = raw_target.split("#", 1)[0].split("?", 1)[0].lower()
        if normalized.endswith(".svg"):
            total += 1
    return total


def _required_density_count(slide_count: int, target_per_slide: float) -> int:
    return math.ceil(slide_count * target_per_slide)


def _presenter_slide_numbers(script_path: Path) -> set[int]:
    text = script_path.read_text(encoding="utf-8")
    return {int(match.group(1)) for match in _PRESENTER_HEADER_RE.finditer(text)}


def _load_slide_mapping(mapping_path: Path) -> dict[int, int]:
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PresentationValidationError(f"{mapping_path}: mapping JSON must be an object")
    slides = raw.get("slides")
    if isinstance(slides, list):
        result: dict[int, int] = {}
        for index, item in enumerate(slides):
            if not isinstance(item, dict):
                raise PresentationValidationError(
                    f"{mapping_path}: slides[{index}] must be an object"
                )
            slide_number = item.get("slide_number", index)
            if not isinstance(slide_number, int) or isinstance(slide_number, bool):
                raise PresentationValidationError(
                    f"{mapping_path}: slides[{index}].slide_number must be an integer"
                )
            result[index] = slide_number
        return result
    result: dict[int, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.isdigit():
            raise PresentationValidationError(
                f"{mapping_path}: mapping key must be a stringified integer: {key!r}"
            )
        if not isinstance(value, int) or isinstance(value, bool):
            raise PresentationValidationError(
                f"{mapping_path}: mapping value must be an integer: {value!r}"
            )
        result[int(key)] = value
    return result


def _pdf_page_count(pdf_path: Path) -> int | None:
    if pdf_path is None:
        return None
    try:
        data = pdf_path.read_bytes()
    except OSError:
        return None
    # Phase 25 deliberately uses a stdlib heuristic so validation has no
    # optional PDF parser dependency. It is sufficient for export-readiness
    # checks and is locked by the local PDF spike.
    matches = re.findall(rb"/Type\s*/Pages\b(?:(?!endobj).)*?/Count\s+(\d+)", data, re.S)
    if not matches:
        return None
    try:
        return max(int(match) for match in matches)
    except ValueError:
        return None


def _workspace_path(workspace: workflow.WorkflowWorkspace, artifact: PresentationArtifactRef) -> Path:
    return workspace.root / artifact.path


def _record(
    *,
    check_id: str,
    status: str,
    severity: str,
    message: str,
    evidence_paths: Sequence[str | Path],
    validation_stage_id: str,
    checkpoint_stage_id: str,
    workspace_root: Path | None = None,
) -> PresentationCheckRecord:
    normalized: list[str] = []
    for evidence_path in evidence_paths:
        if isinstance(evidence_path, Path):
            if workspace_root is None:
                raise PresentationValidationError("workspace_root required for Path evidence")
            normalized.append(_relative_to_workspace(evidence_path, workspace_root))
        else:
            if Path(evidence_path).is_absolute():
                raise PresentationValidationError("evidence paths must be workspace-relative")
            normalized.append(evidence_path)
    return PresentationCheckRecord(
        check_id=check_id,
        status=status,
        severity=severity,
        message=message,
        evidence_paths=tuple(normalized),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
    )


def _artifacts_for_stage(
    presentation: PresentationEntry,
    checkpoint_stage_id: str,
    *,
    suffixes: Sequence[str] = (),
) -> tuple[PresentationArtifactRef, ...]:
    state = presentation.stage_states.get(checkpoint_stage_id)
    if state is None:
        artifacts = (
            presentation.legacy_markdown.artifacts
            if presentation.legacy_markdown is not None
            else ()
        )
        if not suffixes:
            return artifacts
        return tuple(
            artifact
            for artifact in artifacts
            if any(artifact.path.endswith(suffix) for suffix in suffixes)
        )
    if not suffixes:
        return state.artifacts
    return tuple(
        artifact
        for artifact in state.artifacts
        if any(artifact.path.endswith(suffix) for suffix in suffixes)
    )


def _first_artifact_path(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    checkpoint_stage_id: str,
    suffix: str,
) -> Path | None:
    artifacts = _artifacts_for_stage(presentation, checkpoint_stage_id, suffixes=(suffix,))
    if not artifacts:
        for state in presentation.stage_states.values():
            for artifact in state.artifacts:
                if artifact.path.endswith(suffix):
                    return _workspace_path(workspace, artifact)
        candidate = workspace.root / "decks" / presentation.id / suffix
        return candidate if candidate.exists() else None
    return _workspace_path(workspace, artifacts[0])


def _maybe_skip_for_manifest_gap(
    presentation: PresentationEntry,
    validation_stage_id: str,
) -> PresentationCheckRecord | None:
    checkpoint_stage_id = VALIDATION_STAGE_TO_CHECKPOINT[validation_stage_id]
    if (
        validation_stage_id in LEGACY_MARKDOWN_VALIDATION_STAGE_IDS
        and presentation.legacy_markdown is not None
    ):
        return None
    state = presentation.stage_states.get(checkpoint_stage_id)
    if state is None:
        return None if presentation.legacy_markdown is not None else PresentationCheckRecord(
            check_id=f"{checkpoint_stage_id}_manifest_state",
            status="skipped_missing",
            severity="structural",
            message=(
                f"Manifest does not declare {checkpoint_stage_id}; "
                "Phase 25 does not regenerate artifacts."
            ),
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    if state.status == "complete":
        return None
    return PresentationCheckRecord(
        check_id=f"{checkpoint_stage_id}_manifest_state",
        status="skipped_missing",
        severity="structural",
        message=(
            f"Manifest records {checkpoint_stage_id} as {state.status!r}; "
            "Phase 25 does not regenerate artifacts."
        ),
        evidence_paths=(),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
    )


def _check_slide_mapping_schema(
    workspace: workflow.WorkflowWorkspace,
    mapping_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    if mapping_path is None:
        return _record(
            check_id="slide_mapping_schema",
            status="fail",
            severity="structural",
            message="presentation_slide_mapping.json is not declared in the manifest",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    try:
        mapping = _load_slide_mapping(mapping_path)
    except (OSError, json.JSONDecodeError, PresentationValidationError) as exc:
        return _record(
            check_id="slide_mapping_schema",
            status="fail",
            severity="structural",
            message=f"presentation_slide_mapping.json is invalid: {type(exc).__name__}",
            evidence_paths=(mapping_path,),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    return _record(
        check_id="slide_mapping_schema",
        status="pass",
        severity="structural",
        message=f"presentation_slide_mapping.json parses with {len(mapping)} entries",
        evidence_paths=(mapping_path,),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_slide_mapping_markdown_count_equals_mapping(
    workspace: workflow.WorkflowWorkspace,
    markdown_path: Path | None,
    mapping_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    if markdown_path is None or mapping_path is None:
        return _record(
            check_id="slide_mapping_markdown_count_equals_mapping",
            status="fail",
            severity="structural",
            message="markdown or mapping artifact is not declared",
            evidence_paths=tuple(path for path in (markdown_path, mapping_path) if path is not None),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    try:
        slides = _markdown_slide_numbers(markdown_path)
        mapping = _load_slide_mapping(mapping_path)
    except (OSError, json.JSONDecodeError, PresentationValidationError) as exc:
        return _record(
            check_id="slide_mapping_markdown_count_equals_mapping",
            status="fail",
            severity="structural",
            message=f"markdown/mapping count check failed to parse inputs: {type(exc).__name__}",
            evidence_paths=(markdown_path, mapping_path),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    status = "pass" if len(slides) == len(mapping) else "fail"
    return _record(
        check_id="slide_mapping_markdown_count_equals_mapping",
        status=status,
        severity="structural",
        message=f"markdown slide headers ({len(slides)}) and mapping entries ({len(mapping)}) compared",
        evidence_paths=(markdown_path, mapping_path),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_slide_mapping_revealjs_section_count_equals_mapping(
    workspace: workflow.WorkflowWorkspace,
    html_path: Path | None,
    mapping_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    if html_path is None or mapping_path is None:
        return _record(
            check_id="slide_mapping_revealjs_section_count_equals_mapping",
            status="fail",
            severity="structural",
            message="rendered HTML or mapping artifact is not declared",
            evidence_paths=tuple(path for path in (html_path, mapping_path) if path is not None),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    try:
        total_sections, distinct_sections = _count_reveal_sections(html_path)
        mapping = _load_slide_mapping(mapping_path)
    except (OSError, json.JSONDecodeError, PresentationValidationError) as exc:
        return _record(
            check_id="slide_mapping_revealjs_section_count_equals_mapping",
            status="fail",
            severity="structural",
            message=f"Reveal.js section count check failed to parse inputs: {type(exc).__name__}",
            evidence_paths=(html_path, mapping_path),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    status = "pass" if distinct_sections == len(mapping) and total_sections >= distinct_sections else "fail"
    return _record(
        check_id="slide_mapping_revealjs_section_count_equals_mapping",
        status=status,
        severity="structural",
        message=(
            f"Reveal.js sections total={total_sections}, distinct data-markdown-slide="
            f"{distinct_sections}, mapping entries={len(mapping)}"
        ),
        evidence_paths=(html_path, mapping_path),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_svg_well_formed(
    workspace: workflow.WorkflowWorkspace,
    svg_path: Path,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    try:
        ET.parse(svg_path)
    except ET.ParseError as exc:
        return _record(
            check_id="svg_well_formed",
            status="fail",
            severity="structural",
            message=f"SVG failed strict XML parse: {exc}",
            evidence_paths=(svg_path,),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    return _record(
        check_id="svg_well_formed",
        status="pass",
        severity="structural",
        message="SVG parses as well-formed XML",
        evidence_paths=(svg_path,),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_hyperlink_density_meets_target(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    markdown_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    target = presentation.target_hyperlinks_per_slide
    if target <= 0:
        return _record(
            check_id="hyperlink_density_meets_target",
            status="skipped_missing",
            severity="content",
            message="No hyperlink density target is configured for this presentation",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    if markdown_path is None:
        return _record(
            check_id="hyperlink_density_meets_target",
            status="fail",
            severity="content",
            message="4.5_draft_slides_with_hyperlinks.md is not declared",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    try:
        slide_count = _markdown_slide_count(markdown_path)
        hyperlink_count = _markdown_hyperlink_count(markdown_path)
    except OSError as exc:
        return _record(
            check_id="hyperlink_density_meets_target",
            status="fail",
            severity="content",
            message=f"Could not read hyperlink-enriched markdown: {type(exc).__name__}",
            evidence_paths=(markdown_path,),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    required = _required_density_count(slide_count, target)
    status = "pass" if slide_count > 0 and hyperlink_count >= required else "fail"
    return _record(
        check_id="hyperlink_density_meets_target",
        status=status,
        severity="content",
        message=(
            f"hyperlinks={hyperlink_count}, slides={slide_count}, "
            f"target_per_slide={target:.2f}, required={required}"
        ),
        evidence_paths=(markdown_path,),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_hyperlink_preservation_4_5_to_5(
    workspace: workflow.WorkflowWorkspace,
    source_markdown_path: Path | None,
    output_markdown_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    if source_markdown_path is None or output_markdown_path is None:
        return _record(
            check_id="hyperlink_preservation_4_5_to_5",
            status="fail",
            severity="content",
            message="Hyperlink source or figure-enriched markdown is not declared",
            evidence_paths=tuple(
                path for path in (source_markdown_path, output_markdown_path) if path is not None
            ),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    try:
        source_links = _markdown_hyperlink_count(source_markdown_path)
        output_links = _markdown_hyperlink_count(output_markdown_path)
    except OSError as exc:
        return _record(
            check_id="hyperlink_preservation_4_5_to_5",
            status="fail",
            severity="content",
            message=f"Could not read markdown for hyperlink-preservation check: {type(exc).__name__}",
            evidence_paths=(source_markdown_path, output_markdown_path),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    return _record(
        check_id="hyperlink_preservation_4_5_to_5",
        status="pass" if output_links >= source_links else "fail",
        severity="content",
        message=(
            "figure-enriched markdown preserves hyperlinks: "
            f"source_links={source_links}, output_links={output_links}"
        ),
        evidence_paths=(source_markdown_path, output_markdown_path),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_svg_density_meets_target(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    markdown_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    target = presentation.target_svgs_per_slide
    if target <= 0:
        return _record(
            check_id="svg_density_meets_target",
            status="skipped_missing",
            severity="content",
            message="No SVG density target is configured for this presentation",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    if markdown_path is None:
        return _record(
            check_id="svg_density_meets_target",
            status="fail",
            severity="content",
            message="5_draft_slides_with_images.md is not declared",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    try:
        slide_count = _markdown_svg_density_slide_count(markdown_path)
        svg_reference_count = _markdown_svg_reference_count(markdown_path)
    except OSError as exc:
        return _record(
            check_id="svg_density_meets_target",
            status="fail",
            severity="content",
            message=f"Could not read figure-enriched markdown: {type(exc).__name__}",
            evidence_paths=(markdown_path,),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    required = _required_density_count(slide_count, target)
    status = "pass" if slide_count > 0 and svg_reference_count >= required else "fail"
    return _record(
        check_id="svg_density_meets_target",
        status=status,
        severity="content",
        message=(
            f"svg_refs={svg_reference_count}, slides={slide_count}, "
            f"target_per_slide={target:.2f}, required={required}"
        ),
        evidence_paths=(markdown_path,),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_revealjs_html_parses(
    workspace: workflow.WorkflowWorkspace,
    html_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    if html_path is None:
        return _record(
            check_id="revealjs_html_parses",
            status="fail",
            severity="structural",
            message="presentation.html is not declared",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    try:
        _count_reveal_sections(html_path)
    except OSError as exc:
        return _record(
            check_id="revealjs_html_parses",
            status="fail",
            severity="structural",
            message=f"presentation.html could not be read: {type(exc).__name__}",
            evidence_paths=(html_path,),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    return _record(
        check_id="revealjs_html_parses",
        status="pass",
        severity="structural",
        message="presentation.html parsed with stdlib HTMLParser",
        evidence_paths=(html_path,),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _read_html_for_check(
    workspace: workflow.WorkflowWorkspace,
    html_path: Path | None,
    *,
    check_id: str,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> tuple[str | None, PresentationCheckRecord | None]:
    if html_path is None:
        return None, _record(
            check_id=check_id,
            status="fail",
            severity="structural",
            message="presentation.html is not declared",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    try:
        return html_path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, _record(
            check_id=check_id,
            status="fail",
            severity="structural",
            message=f"presentation.html could not be read: {type(exc).__name__}",
            evidence_paths=(html_path,),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )


def _visible_chip_labels(html_text: str) -> list[str]:
    labels: list[str] = []
    for block in _REFERENCE_CHIPS_BLOCK_RE.finditer(html_text):
        for label_match in _REFERENCE_CHIP_LABEL_RE.finditer(block.group("body")):
            label = _HTML_TAG_RE.sub("", label_match.group("label"))
            label = re.sub(r"\s+", " ", unescape(label)).strip()
            if label:
                labels.append(label)
    return labels


def _check_revealjs_reference_labels_visible(
    workspace: workflow.WorkflowWorkspace,
    html_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    html_text, failure = _read_html_for_check(
        workspace,
        html_path,
        check_id="revealjs_reference_labels_visible",
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
    )
    if failure is not None:
        return failure
    assert html_text is not None
    labels = _visible_chip_labels(html_text)
    failures: list[str] = []
    for label in labels:
        normalized = label.strip()
        if _GENERIC_REFERENCE_LABEL_RE.match(normalized):
            failures.append(f"generic visible label `{normalized}`")
            continue
        words = re.findall(r"[A-Za-z0-9]+", normalized)
        if not words:
            failures.append("empty/non-word visible label")
            continue
        if words[-1].lower() in _REFERENCE_LABEL_BAD_SUFFIXES:
            failures.append(f"clipped-looking visible label `{normalized}`")
    status = "fail" if failures else "pass"
    message = (
        f"reference chip labels checked: {len(labels)} visible labels"
        if labels
        else "no reference chip labels found"
    )
    if failures:
        message += "; failures: " + "; ".join(failures[:8])
    return _record(
        check_id="revealjs_reference_labels_visible",
        status=status,
        severity="content",
        message=message,
        evidence_paths=(html_path,) if html_path is not None else (),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_revealjs_title_slide_branded(
    workspace: workflow.WorkflowWorkspace,
    html_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    html_text, failure = _read_html_for_check(
        workspace,
        html_path,
        check_id="revealjs_title_slide_branded",
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
    )
    if failure is not None:
        return failure
    assert html_text is not None
    has_title_slide = re.search(
        r"<section\b[^>]*class=[\"'][^\"']*\bslide-title\b",
        html_text,
        re.IGNORECASE,
    ) is not None
    has_reference_template = "relationalai-neurips2025-reference" in html_text.lower()
    has_title_logo_rule = (
        "section.slide-title::before" in html_text
        and "background-image: url(" in html_text
        and "template-logo-missing" not in _body_class_value(html_text)
    )
    missing = []
    if not has_title_slide:
        missing.append("slide-title section")
    if not has_reference_template:
        missing.append("RelationalAI reference template marker")
    if not has_title_logo_rule:
        missing.append("title-slide RelationalAI logo rule")
    return _record(
        check_id="revealjs_title_slide_branded",
        status="pass" if not missing else "fail",
        severity="content",
        message=(
            "title slide carries RelationalAI reference branding"
            if not missing
            else "missing " + ", ".join(missing)
        ),
        evidence_paths=(html_path,) if html_path is not None else (),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _body_class_value(html_text: str) -> str:
    match = re.search(r"<body\b[^>]*class=[\"'](?P<class>[^\"']*)[\"']", html_text, re.IGNORECASE)
    return match.group("class") if match else ""


def _brand_after_rule_is_visible(html_text: str, selector: str) -> bool:
    pattern = re.compile(
        rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(html_text)
    if match is None:
        return False
    body = match.group("body").lower()
    return (
        "background-image: url(" in body
        and "content: ''" in body
        and "display: none" not in body
        and "content: none" not in body
    )


def _check_revealjs_content_brand_mark_present(
    workspace: workflow.WorkflowWorkspace,
    html_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    html_text, failure = _read_html_for_check(
        workspace,
        html_path,
        check_id="revealjs_content_brand_mark_present",
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
    )
    if failure is not None:
        return failure
    assert html_text is not None
    body_class = _body_class_value(html_text)
    has_content_slide = re.search(
        r"<section\b[^>]*class=[\"'][^\"']*\b(?:slide-content|slide-content-with-figure|slide-figure-zoom)\b",
        html_text,
        re.IGNORECASE,
    ) is not None
    selectors = (
        ".reveal .slides section.slide-content::after",
        ".reveal .slides section.slide-content-with-figure::after",
        ".reveal .slides section.slide-figure-zoom::after",
    )
    visible_rule_count = sum(
        1 for selector in selectors if _brand_after_rule_is_visible(html_text, selector)
    )
    missing = []
    if not has_content_slide:
        missing.append("content/figure slide section")
    if "template-bug-missing" in body_class:
        missing.append("RelationalAI brand bug asset")
    if visible_rule_count == 0:
        missing.append("visible content-slide brand rule")
    return _record(
        check_id="revealjs_content_brand_mark_present",
        status="pass" if not missing else "fail",
        severity="content",
        message=(
            f"content/figure slides carry RelationalAI brand mark rules ({visible_rule_count})"
            if not missing
            else "missing " + ", ".join(missing)
        ),
        evidence_paths=(html_path,) if html_path is not None else (),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_presenter_script_slide_ids_resolve(
    workspace: workflow.WorkflowWorkspace,
    script_path: Path | None,
    mapping_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    if script_path is None or mapping_path is None:
        return _record(
            check_id="presenter_script_slide_ids_resolve",
            status="fail",
            severity="structural",
            message="presenter script or mapping artifact is not declared",
            evidence_paths=tuple(path for path in (script_path, mapping_path) if path is not None),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    try:
        script_slides = _presenter_slide_numbers(script_path)
        mapping = _load_slide_mapping(mapping_path)
    except (OSError, json.JSONDecodeError, PresentationValidationError) as exc:
        return _record(
            check_id="presenter_script_slide_ids_resolve",
            status="fail",
            severity="structural",
            message=f"presenter-script check failed to parse inputs: {type(exc).__name__}",
            evidence_paths=(script_path, mapping_path),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    missing = sorted(set(mapping) - script_slides)
    status = "pass" if not missing else "fail"
    return _record(
        check_id="presenter_script_slide_ids_resolve",
        status=status,
        severity="structural",
        message=f"presenter script covers mapping slides; missing={missing}",
        evidence_paths=(script_path, mapping_path),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_pdf_parses(
    workspace: workflow.WorkflowWorkspace,
    pdf_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    page_count = _pdf_page_count(pdf_path) if pdf_path is not None else None
    status = "pass" if page_count is not None else "fail"
    return _record(
        check_id="pdf_parses",
        status=status,
        severity="structural",
        message=(
            f"PDF parser read {page_count} pages"
            if page_count is not None
            else "PDF parser could not read page count"
        ),
        evidence_paths=(pdf_path,) if pdf_path is not None else (),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_pdf_page_count_matches_target_slides(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    pdf_path: Path | None,
    html_path: Path | None,
    mapping_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    page_count = _pdf_page_count(pdf_path) if pdf_path is not None else None
    expected_counts = {presentation.target_slide_count}
    try:
        if html_path is not None and mapping_path is not None:
            total_sections, distinct_sections = _count_reveal_sections(html_path)
            mapping = _load_slide_mapping(mapping_path)
            if distinct_sections == len(mapping) and total_sections >= distinct_sections:
                expected_counts.add(total_sections)
    except (OSError, json.JSONDecodeError, PresentationValidationError):
        pass
    if page_count is None:
        status = "fail"
    elif page_count in expected_counts:
        status = "pass"
    else:
        status = "warn"
    expected_label = ", ".join(str(value) for value in sorted(expected_counts))
    return _record(
        check_id="pdf_page_count_matches_target_slides",
        status=status,
        severity="structural",
        message=(
            f"PDF pages ({page_count}) compared to expected slide/page counts "
            f"({expected_label})"
        ),
        evidence_paths=(pdf_path,) if pdf_path is not None else (),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_video_artifacts_nonempty(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    video_paths = _video_artifact_paths(workspace, presentation, checkpoint_stage_id)
    if not video_paths:
        return _record(
            check_id="video_artifacts_nonempty",
            status="skipped_missing",
            severity="structural",
            message="No declared video files to inspect",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    empty = [path for path in video_paths if path.stat().st_size <= 0]
    return _record(
        check_id="video_artifacts_nonempty",
        status="pass" if not empty else "fail",
        severity="structural",
        message=f"{len(video_paths) - len(empty)} of {len(video_paths)} video artifacts are non-empty",
        evidence_paths=tuple(video_paths),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _video_artifact_paths(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    checkpoint_stage_id: str,
) -> tuple[Path, ...]:
    paths = [
        _workspace_path(workspace, artifact)
        for artifact in _artifacts_for_stage(
            presentation,
            checkpoint_stage_id,
            suffixes=_VIDEO_SUFFIXES,
        )
    ]
    deck_dir = workspace.root / "decks" / presentation.id
    for filename in ("presentation_video.mp4", "presentation_video_notebooklm.mp4"):
        candidate = deck_dir / filename
        if candidate.exists() and candidate not in paths:
            paths.append(candidate)
    return tuple(paths)


def _check_video_duration_readable(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    video_paths = _video_artifact_paths(workspace, presentation, checkpoint_stage_id)
    if not video_paths:
        return _record(
            check_id="video_duration_readable",
            status="skipped_missing",
            severity="structural",
            message="No declared video files to inspect",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    durations: list[float | None] = []
    for path in video_paths:
        probe = _probe_media_durations(path)
        duration = None if probe is None else probe.format_duration
        if duration is None and path.suffix.lower() == ".mp4":
            duration = _mp4_duration_seconds(path)
        durations.append(duration)
    status = "pass" if all(duration and duration > 0 for duration in durations) else "warn"
    return _record(
        check_id="video_duration_readable",
        status=status,
        severity="structural",
        message=f"video duration metadata values: {durations}",
        evidence_paths=tuple(video_paths),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _video_delivery_record(
    verdict,
    *,
    check_id: str,
    severity: str,
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    """Run one path-based ``video_delivery`` verdict over the declared videos.

    The probe and tail-detector are passed from THIS module's namespace so a
    test that patches ``presentation.validation._probe_media_durations`` still
    reaches the verdict.
    """

    video_paths = _video_artifact_paths(workspace, presentation, checkpoint_stage_id)
    if not video_paths:
        return _record(
            check_id=check_id,
            status="skipped_missing",
            severity=severity,
            message="No declared video files to inspect",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    result = verdict(
        video_paths,
        probe=_probe_media_durations,
        detect_black_tail=_detect_black_tail_seconds,
    )
    return _record(
        check_id=result.check_id,
        status=result.status,
        severity=result.severity,
        message=result.message,
        evidence_paths=tuple(result.evidence_paths),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_video_duration_under_40_minutes(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    return _video_delivery_record(
        duration_budget_verdict,
        check_id="video_duration_under_40_minutes",
        severity="content",
        workspace=workspace,
        presentation=presentation,
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
    )


def _check_video_stream_durations_match(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    return _video_delivery_record(
        stream_alignment_verdict,
        check_id="video_stream_durations_match",
        severity="structural",
        workspace=workspace,
        presentation=presentation,
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
    )


def _check_video_duration_matches_build_receipts(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    return _video_delivery_record(
        build_receipt_agreement_verdict,
        check_id="video_duration_matches_build_receipts",
        severity="structural",
        workspace=workspace,
        presentation=presentation,
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
    )


def _check_video_tail_not_black(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    return _video_delivery_record(
        black_tail_verdict,
        check_id="video_tail_not_black",
        severity="structural",
        workspace=workspace,
        presentation=presentation,
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
    )


def _check_visual_qc_report_present(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    artifacts = _artifacts_for_stage(presentation, checkpoint_stage_id)
    paths = tuple(_workspace_path(workspace, artifact) for artifact in artifacts)
    return _record(
        check_id="visual_qc_report_present",
        status="warn",
        severity="content",
        message="Visual QC is operator-reviewed; Phase 25 does not assert subjective quality",
        evidence_paths=paths,
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_vision_llm_visual_qc_passed(
    workspace: workflow.WorkflowWorkspace,
    presentation: PresentationEntry,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    report_path = _first_artifact_path(
        workspace,
        presentation,
        checkpoint_stage_id,
        "visual_llm_qc_report.json",
    )
    if report_path is None or not report_path.is_file():
        return _record(
            check_id="vision_llm_visual_qc_passed",
            status="fail",
            severity="content",
            message="vision-LLM visual QC report is missing",
            evidence_paths=(() if report_path is None else (report_path,)),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _record(
            check_id="vision_llm_visual_qc_passed",
            status="fail",
            severity="content",
            message=f"vision-LLM visual QC report is not parseable JSON: {exc}",
            evidence_paths=(report_path,),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    status_value = str(payload.get("status") or "").strip().lower()
    errors = payload.get("errors") or []
    warnings = payload.get("warnings") or []
    if status_value == "passed" and not errors:
        status = "pass"
    elif status_value == "warnings" and not errors:
        status = "warn"
    else:
        status = "fail"
    return _record(
        check_id="vision_llm_visual_qc_passed",
        status=status,
        severity="content",
        message=(
            f"vision-LLM visual QC status={status_value or 'unknown'}; "
            f"errors={len(errors) if isinstance(errors, list) else 'unknown'}; "
            f"warnings={len(warnings) if isinstance(warnings, list) else 'unknown'}"
        ),
        evidence_paths=(report_path,),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_blueprint_schema_valid(
    workspace: workflow.WorkflowWorkspace,
    blueprint_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    if blueprint_path is None:
        return _record(
            check_id="blueprint_schema_valid",
            status="fail",
            severity="structural",
            message="presentation_blueprint.json is not declared",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    result = validate_blueprint_file(blueprint_path)
    return _record(
        check_id="blueprint_schema_valid",
        status="pass" if not result.errors else "fail",
        severity="structural",
        message="presentation_blueprint.json passes schema validation"
        if not result.errors
        else "; ".join(record.message for record in result.errors),
        evidence_paths=(blueprint_path,),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_blueprint_approved(
    workspace: workflow.WorkflowWorkspace,
    blueprint_path: Path | None,
    review_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    if blueprint_path is None or review_path is None:
        return _record(
            check_id="blueprint_approved",
            status="fail",
            severity="structural",
            message="presentation_blueprint.json or blueprint_review.json is not declared",
            evidence_paths=tuple(path for path in (blueprint_path, review_path) if path is not None),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
            workspace_root=workspace.root,
        )
    result = validate_blueprint_review_file(blueprint_path, review_path)
    return _record(
        check_id="blueprint_approved",
        status="pass" if not result.errors else "fail",
        severity="content",
        message="blueprint_review.json is approved_for_draft_generation and hash-bound"
        if not result.errors
        else "; ".join(record.message for record in result.errors),
        evidence_paths=(blueprint_path, review_path),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_source_plan_coverage(
    workspace: workflow.WorkflowWorkspace,
    blueprint_path: Path | None,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> PresentationCheckRecord:
    if blueprint_path is None:
        return _record(
            check_id="source_plan_coverage",
            status="fail",
            severity="content",
            message="presentation_blueprint.json is not declared",
            evidence_paths=(),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    result = validate_blueprint_file(blueprint_path)
    source_errors = [
        record
        for record in result.errors
        if record.code in {BLUEPRINT_SOURCE_BINDING_MISSING, BLUEPRINT_SOURCE_UNRESOLVED}
    ]
    return _record(
        check_id="source_plan_coverage",
        status="pass" if not source_errors else "fail",
        severity="content",
        message="source_plan_coverage binds every cluster source plan to approved metadata"
        if not source_errors
        else "; ".join(record.message for record in source_errors),
        evidence_paths=(blueprint_path,),
        validation_stage_id=validation_stage_id,
        checkpoint_stage_id=checkpoint_stage_id,
        workspace_root=workspace.root,
    )


def _check_unreferenced_deck_lanes(
    workspace: workflow.WorkflowWorkspace,
    loaded_manifest: LoadedPresentationManifest,
    validation_stage_id: str,
    checkpoint_stage_id: str,
) -> list[PresentationCheckRecord]:
    registered = {presentation.id for presentation in loaded_manifest.manifest.presentations}
    warnings: list[PresentationCheckRecord] = []
    decks_dir = workspace.root / "decks"
    if not decks_dir.is_dir():
        return warnings
    for child in sorted(path for path in decks_dir.iterdir() if path.is_dir()):
        if child.name in registered or child.name in {"master_presentation", "executive_summary"}:
            continue
        warnings.append(
            _record(
                check_id="unreferenced_deck_lane",
                status="warn",
                severity="content",
                message="Deck-lane artifact is not referenced by presentation_manifest.json",
                evidence_paths=(child,),
                validation_stage_id=validation_stage_id,
                checkpoint_stage_id=checkpoint_stage_id,
                workspace_root=workspace.root,
            )
        )
    return warnings


def _run_checks_for_stage(
    validation_stage_id: str,
    workspace: workflow.WorkflowWorkspace,
    loaded_manifest: LoadedPresentationManifest,
    presentation: PresentationEntry,
) -> list[PresentationCheckRecord]:
    checkpoint_stage_id = VALIDATION_STAGE_TO_CHECKPOINT[validation_stage_id]
    skip = _maybe_skip_for_manifest_gap(presentation, validation_stage_id)
    if skip is not None:
        return [skip]

    mapping_path = _first_artifact_path(
        workspace,
        presentation,
        "revealjs_render",
        "presentation_slide_mapping.json",
    )
    html_path = _first_artifact_path(workspace, presentation, "revealjs_render", "presentation.html")
    markdown_path = _first_artifact_path(
        workspace,
        presentation,
        "figure_generation",
        "5_draft_slides_with_images.md",
    )
    hyperlink_markdown_path = _first_artifact_path(
        workspace,
        presentation,
        "research_enrichment",
        "4.5_draft_slides_with_hyperlinks.md",
    )
    script_path = _first_artifact_path(
        workspace,
        presentation,
        "presenter_script",
        "6_presenter_script.md",
    )
    blueprint_path = _first_artifact_path(
        workspace,
        presentation,
        "presentation_blueprint",
        "presentation_blueprint.json",
    )
    blueprint_review_path = _first_artifact_path(
        workspace,
        presentation,
        "presentation_blueprint",
        "blueprint_review.json",
    )

    if validation_stage_id == "presentation_blueprint_validation":
        return [
            _check_blueprint_schema_valid(
                workspace, blueprint_path, validation_stage_id, checkpoint_stage_id
            ),
            _check_blueprint_approved(
                workspace,
                blueprint_path,
                blueprint_review_path,
                validation_stage_id,
                checkpoint_stage_id,
            ),
            _check_source_plan_coverage(
                workspace, blueprint_path, validation_stage_id, checkpoint_stage_id
            ),
        ]

    if validation_stage_id == "slide_consolidation_validation":
        return [
            _check_slide_mapping_schema(
                workspace, mapping_path, validation_stage_id, checkpoint_stage_id
            ),
            _check_slide_mapping_markdown_count_equals_mapping(
                workspace, markdown_path, mapping_path, validation_stage_id, checkpoint_stage_id
            ),
        ]
    if validation_stage_id == "research_enrichment_validation":
        artifacts = _artifacts_for_stage(presentation, checkpoint_stage_id)
        return [
            _record(
                check_id=f"{checkpoint_stage_id}_artifacts_declared",
                status="pass" if artifacts else "skipped_missing",
                severity="structural",
                message=f"{checkpoint_stage_id} has {len(artifacts)} declared artifacts",
                evidence_paths=tuple(artifact.path for artifact in artifacts),
                validation_stage_id=validation_stage_id,
                checkpoint_stage_id=checkpoint_stage_id,
            ),
            _check_hyperlink_density_meets_target(
                workspace,
                presentation,
                hyperlink_markdown_path,
                validation_stage_id,
                checkpoint_stage_id,
            ),
        ]
    if validation_stage_id == "figures_structural_validation":
        svg_paths = [
            _workspace_path(workspace, artifact)
            for artifact in _artifacts_for_stage(
                presentation,
                "figure_generation",
                suffixes=(".svg",),
            )
        ]
        density_records = [
            _check_hyperlink_preservation_4_5_to_5(
                workspace,
                hyperlink_markdown_path,
                markdown_path,
                validation_stage_id,
                checkpoint_stage_id,
            ),
            _check_svg_density_meets_target(
                workspace,
                presentation,
                markdown_path,
                validation_stage_id,
                checkpoint_stage_id,
            ),
        ]
        if not svg_paths:
            return density_records + [
                _record(
                    check_id="svg_well_formed",
                    status="skipped_missing",
                    severity="structural",
                    message="No declared SVG artifacts to inspect",
                    evidence_paths=(),
                    validation_stage_id=validation_stage_id,
                    checkpoint_stage_id=checkpoint_stage_id,
                )
            ]
        return density_records + [
            _check_svg_well_formed(workspace, path, validation_stage_id, checkpoint_stage_id)
            for path in svg_paths
        ]
    if validation_stage_id == "revealjs_render_validation":
        return [
            _check_revealjs_html_parses(
                workspace, html_path, validation_stage_id, checkpoint_stage_id
            ),
            _check_revealjs_reference_labels_visible(
                workspace, html_path, validation_stage_id, checkpoint_stage_id
            ),
            _check_revealjs_title_slide_branded(
                workspace, html_path, validation_stage_id, checkpoint_stage_id
            ),
            _check_revealjs_content_brand_mark_present(
                workspace, html_path, validation_stage_id, checkpoint_stage_id
            ),
            _check_slide_mapping_revealjs_section_count_equals_mapping(
                workspace, html_path, mapping_path, validation_stage_id, checkpoint_stage_id
            ),
        ]
    if validation_stage_id == "presenter_script_validation":
        return [
            _check_presenter_script_slide_ids_resolve(
                workspace, script_path, mapping_path, validation_stage_id, checkpoint_stage_id
            )
        ]
    if validation_stage_id == "pdf_export_validation":
        pdf_path = _first_artifact_path(workspace, presentation, "pdf_export", "presentation.pdf")
        return [
            _check_pdf_parses(workspace, pdf_path, validation_stage_id, checkpoint_stage_id),
            _check_pdf_page_count_matches_target_slides(
                workspace,
                presentation,
                pdf_path,
                html_path,
                mapping_path,
                validation_stage_id,
                checkpoint_stage_id,
            ),
        ]
    if validation_stage_id == "video_generation_validation":
        return [
            _check_video_artifacts_nonempty(
                workspace, presentation, validation_stage_id, checkpoint_stage_id
            ),
            _check_video_duration_readable(
                workspace, presentation, validation_stage_id, checkpoint_stage_id
            ),
            _check_video_duration_under_40_minutes(
                workspace, presentation, validation_stage_id, checkpoint_stage_id
            ),
            _check_video_stream_durations_match(
                workspace, presentation, validation_stage_id, checkpoint_stage_id
            ),
            _check_video_duration_matches_build_receipts(
                workspace, presentation, validation_stage_id, checkpoint_stage_id
            ),
            _check_video_tail_not_black(
                workspace, presentation, validation_stage_id, checkpoint_stage_id
            ),
        ]
    if validation_stage_id == "visual_qc_validation":
        return [
            _check_visual_qc_report_present(
                workspace, presentation, validation_stage_id, checkpoint_stage_id
            ),
            _check_vision_llm_visual_qc_passed(
                workspace, presentation, validation_stage_id, checkpoint_stage_id
            ),
        ]
    if validation_stage_id == "human_review_validation":
        artifacts = _artifacts_for_stage(presentation, checkpoint_stage_id)
        if artifacts:
            return [
                _record(
                    check_id="human_review_manifest_state",
                    status="pass",
                    severity="content",
                    message="Human review receipt is present in the presentation manifest",
                    evidence_paths=tuple(artifact.path for artifact in artifacts),
                    validation_stage_id=validation_stage_id,
                    checkpoint_stage_id=checkpoint_stage_id,
                )
            ]
        return _check_unreferenced_deck_lanes(
            workspace,
            loaded_manifest,
            validation_stage_id,
            checkpoint_stage_id,
        ) or [
            _record(
                check_id="human_review_manifest_state",
                status="skipped_missing",
                severity="content",
                message="Human review remains an operator checkpoint",
                evidence_paths=(),
                validation_stage_id=validation_stage_id,
                checkpoint_stage_id=checkpoint_stage_id,
            )
        ]
    artifacts = _artifacts_for_stage(presentation, checkpoint_stage_id)
    return [
        _record(
            check_id=f"{checkpoint_stage_id}_artifacts_declared",
            status="pass" if artifacts else "skipped_missing",
            severity="structural",
            message=f"{checkpoint_stage_id} has {len(artifacts)} declared artifacts",
            evidence_paths=tuple(artifact.path for artifact in artifacts),
            validation_stage_id=validation_stage_id,
            checkpoint_stage_id=checkpoint_stage_id,
        )
    ]


def enforce_strict_required_artifacts(
    records: Sequence[PresentationCheckRecord],
) -> list[PresentationCheckRecord]:
    """Convert missing required in-scope artifacts into failing records."""

    strict_records: list[PresentationCheckRecord] = []
    for record in records:
        should_fail = (
            record.checkpoint_stage_id in STRICT_REQUIRED_CHECKPOINT_STAGE_ID_SET
            and (
                record.status == "skipped_missing"
                or (record.status == "warn" and not record.evidence_paths)
            )
        )
        if not should_fail:
            strict_records.append(record)
            continue
        strict_records.append(
            PresentationCheckRecord(
                check_id=record.check_id,
                status="fail",
                severity=record.severity,
                message=(
                    "Strict required-artifact mode: "
                    f"{record.checkpoint_stage_id} is in scope and cannot be "
                    f"treated as {record.status!r}. {record.message}"
                ),
                evidence_paths=record.evidence_paths,
                validation_stage_id=record.validation_stage_id,
                checkpoint_stage_id=record.checkpoint_stage_id,
            )
        )
    return strict_records


def _select_presentation(
    loaded_manifest: LoadedPresentationManifest,
    *,
    presentation_type: str | None,
    topic: str | None,
) -> PresentationEntry:
    if presentation_type == "topic-deep-dive" and not topic:
        raise PresentationValidationError("topic-deep-dive validation requires --topic")
    candidates = [
        presentation
        for presentation in loaded_manifest.manifest.presentations
        if presentation_type is None or presentation.presentation_type == presentation_type
    ]
    if topic:
        candidates = [presentation for presentation in candidates if presentation.id == topic]
    if not candidates:
        raise PresentationValidationError("No matching presentation entry in manifest")
    if len(candidates) > 1:
        raise PresentationValidationError("Presentation validation selection is ambiguous")
    return candidates[0]


def run_validation_stage(
    workspace: workflow.WorkflowWorkspace,
    stage: workflow.WorkflowStage,
    *,
    presentation_type: str | None,
    topic: str | None,
    strict_required_artifacts: bool = False,
    loaded_manifest: LoadedPresentationManifest | None = None,
) -> list[PresentationCheckRecord]:
    if stage.id not in VALIDATION_STAGE_IDS:
        raise PresentationValidationError(f"unknown validation stage id: {stage.id!r}")
    if loaded_manifest is None:
        loaded_manifest = load_and_validate_manifest(workspace, require_path_existence=True)
    presentation = _select_presentation(
        loaded_manifest,
        presentation_type=presentation_type,
        topic=topic,
    )
    records = _run_checks_for_stage(stage.id, workspace, loaded_manifest, presentation)
    if strict_required_artifacts:
        return enforce_strict_required_artifacts(records)
    return records


def write_validation_report(
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
    records: Sequence[PresentationCheckRecord],
    *,
    manifest_content_sha256: str,
    selected_scope: dict[str, Any],
    strict_required_artifacts: bool = False,
) -> Path:
    run_dir = workflow_run_dir(workspace, run_id)
    report_path = run_dir / VALIDATION_REPORT_FILENAME
    totals = {
        status: sum(1 for record in records if record.status == status)
        for status in sorted(CHECK_STATUS_VALUES)
    }
    payload = {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "generator": "conference-pipeline/presentation-validate@1",
        "short_commit": _short_commit(),
        "run_id": run_id,
        "workspace": workspace.root.name,
        "manifest_path": MANIFEST_RELATIVE_PATH,
        "manifest_content_sha256": manifest_content_sha256,
        "selected_scope": selected_scope,
        "strict_required_artifacts": strict_required_artifacts,
        "totals": totals,
        "checks": [record.to_dict() for record in records],
    }
    write_json_atomic(report_path, payload)
    return report_path
