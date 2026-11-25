"""Semantic review for generated presentation SVG figures."""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conference_pipeline.presentation.programmatic_figures import (
    APPROVED_SVG_CODEGEN_MODEL,
    APPROVED_SVG_CODEGEN_PROVIDER,
    SVG_CODEGEN_TASK_ID,
)
from conference_pipeline.workflow_runs import write_json_atomic

GENERIC_TEMPLATE_PHRASES = (
    "Architecture work is moving",
    "Search landscape: promising regions emerge",
    "Spatial evidence model",
    "Scaling pressure curve",
    "Scaffold value",
    "Metric trend",
    "Evidence maps to mechanism",
    "map cited evidence to an architecture decision",
)

PLACEHOLDER_CHART_TICKS = ("4K", "16K", "64K", "128K", "1.0x", "1.7x", "2.4x")
VISUAL_EXAMPLE_TERMS = (
    "visual examples",
    "rendering",
    "sequence",
    "demonstrating",
    "frames",
    "3d",
    "geometry",
    "molecule",
    "binding pocket",
)


@dataclass(frozen=True)
class SemanticIssue:
    file: str
    slide_number: int | None
    check_id: str
    severity: str
    message: str


def review_programmatic_figures(
    workspace_root: Path,
    *,
    generated_dir: Path,
    asset_inventory: Path,
    report_output: Path,
    markdown_output: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    workspace = workspace_root.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace must be an existing directory: {workspace}")
    generated_root = _resolve_inside(workspace, generated_dir, label="generated_dir")
    inventory_path = _resolve_inside(workspace, asset_inventory, label="asset_inventory")
    report_path = _resolve_inside(workspace, report_output, label="report_output")
    markdown_path = (
        _resolve_inside(workspace, markdown_output, label="markdown_output")
        if markdown_output is not None
        else None
    )
    inventory = _load_inventory(inventory_path)
    figures = inventory.get("figures") if isinstance(inventory, dict) else None
    if not isinstance(figures, list):
        figures = []

    issues: list[SemanticIssue] = []
    reviewed: list[dict[str, object]] = []
    if not figures:
        issues.append(
            SemanticIssue(
                file=_relative(inventory_path, workspace),
                slide_number=None,
                check_id="no_figures_in_inventory",
                severity="error",
                message="asset inventory has no generated figures to semantically review",
            )
        )

    for figure in figures:
        if not isinstance(figure, dict):
            continue
        figure_result, figure_issues = _review_figure(figure, workspace, generated_root)
        reviewed.append(figure_result)
        issues.extend(figure_issues)

    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warn"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "failed" if errors else "passed",
        "model_governance": {
            "task_id": SVG_CODEGEN_TASK_ID,
            "provider": APPROVED_SVG_CODEGEN_PROVIDER,
            "model_id": APPROVED_SVG_CODEGEN_MODEL,
            "status": "required_for_live_codegen",
        },
        "generated_dir": _relative(generated_root, workspace),
        "asset_inventory": _relative(inventory_path, workspace),
        "figures_reviewed": len(reviewed),
        "issue_counts": {
            "error": len(errors),
            "warn": len(warnings),
        },
        "figures": reviewed,
        "issues": [asdict(issue) for issue in issues],
    }
    if not dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(report_path, payload)
        if markdown_path is not None:
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def _review_figure(
    figure: dict[str, object],
    workspace: Path,
    generated_root: Path,
) -> tuple[dict[str, object], list[SemanticIssue]]:
    del generated_root
    issues: list[SemanticIssue] = []
    slide_number = _int_or_none(figure.get("slide_number"))
    figure_path_value = str(figure.get("path") or "")
    file_label = figure_path_value or "<missing>"
    spec = figure.get("figure_spec") if isinstance(figure.get("figure_spec"), dict) else {}
    need = str(spec.get("figure_need") or figure.get("need") or "")
    title = str(spec.get("slide_title") or figure.get("slide_title") or "")
    depiction_type = str(spec.get("depiction_type") or figure.get("figure_type") or "")
    entities = [str(item) for item in spec.get("entities", [])] if isinstance(spec, dict) else []
    quantities = [str(item) for item in spec.get("quantities", [])] if isinstance(spec, dict) else []

    svg_text = ""
    text_nodes: list[str] = []
    if not figure_path_value:
        issues.append(
            SemanticIssue(
                file=file_label,
                slide_number=slide_number,
                check_id="missing_svg_path",
                severity="error",
                message="figure inventory entry has no SVG path",
            )
        )
    else:
        svg_path = _resolve_inside(workspace, Path(figure_path_value), label="figure.path")
        if not svg_path.exists():
            issues.append(
                SemanticIssue(
                    file=file_label,
                    slide_number=slide_number,
                    check_id="missing_svg_file",
                    severity="error",
                    message="figure inventory path does not exist",
                )
            )
        else:
            svg_text = svg_path.read_text(encoding="utf-8", errors="replace")
            text_nodes = _svg_text_nodes(svg_text)

    if bool(figure.get("density_fallback")):
        issues.append(
            SemanticIssue(
                file=file_label,
                slide_number=slide_number,
                check_id="density_fallback_used",
                severity="error",
                message="density fallback cannot satisfy production semantic figure generation",
            )
        )

    generic_hits = [
        phrase for phrase in GENERIC_TEMPLATE_PHRASES if phrase.lower() in svg_text.lower()
    ]
    if generic_hits:
        issues.append(
            SemanticIssue(
                file=file_label,
                slide_number=slide_number,
                check_id="generic_template_phrase",
                severity="error",
                message="generic figure template phrases found: " + ", ".join(generic_hits),
            )
        )

    entity_coverage = _entity_coverage(entities, svg_text)
    if entities and entity_coverage < 0.25:
        issues.append(
            SemanticIssue(
                file=file_label,
                slide_number=slide_number,
                check_id="required_entity_coverage",
                severity="error",
                message=(
                    f"only {entity_coverage:.0%} of required entities appear in SVG text; "
                    "figure may not be source-grounded"
                ),
            )
        )

    copied_ratio = _copied_source_ratio(text_nodes, title=title, need=need)
    if copied_ratio > 0.78 and len(" ".join(text_nodes).split()) >= 24:
        issues.append(
            SemanticIssue(
                file=file_label,
                slide_number=slide_number,
                check_id="copied_slide_language",
                severity="warn",
                message=(
                    f"SVG text overlaps source language at {copied_ratio:.0%}; "
                    "prefer depiction labels over copied narration"
                ),
            )
        )

    placeholder_hits = [tick for tick in PLACEHOLDER_CHART_TICKS if tick in svg_text]
    if depiction_type in {"benchmark_chart", "scaling_chart", "line_chart", "bar_chart"}:
        quantity_text = " ".join(quantities)
        invented_ticks = [tick for tick in placeholder_hits if tick not in quantity_text]
        if invented_ticks:
            issues.append(
                SemanticIssue(
                    file=file_label,
                    slide_number=slide_number,
                    check_id="placeholder_chart_ticks",
                    severity="error",
                    message="chart uses placeholder ticks not found in source quantities: "
                    + ", ".join(invented_ticks),
                )
            )

    need_lower = need.lower()
    if any(term in need_lower for term in VISUAL_EXAMPLE_TERMS):
        concrete_terms = (
            "3D",
            "Frame",
            "Python",
            "Protein",
            "Ligand",
            "Query",
            "Agent",
            "axis",
            "Tool",
            "Verifier",
            "Encoder",
            "Representation",
        )
        if not any(term.lower() in svg_text.lower() for term in concrete_terms):
            issues.append(
                SemanticIssue(
                    file=file_label,
                    slide_number=slide_number,
                    check_id="abstract_visual_example",
                    severity="error",
                    message="requested a concrete visual example but SVG contains only abstract labels",
                )
            )

    figure_result = {
        "file": file_label,
        "slide_number": slide_number,
        "depiction_type": depiction_type,
        "entity_coverage": entity_coverage,
        "copied_text_ratio": copied_ratio,
        "generic_phrase_hits": generic_hits,
        "placeholder_chart_ticks": placeholder_hits,
        "status": "failed"
        if any(issue.severity == "error" for issue in issues)
        else "passed",
    }
    return figure_result, issues


def _load_inventory(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_inside(workspace: Path, candidate: Path, *, label: str) -> Path:
    path = candidate.expanduser()
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def _relative(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace).as_posix()


def _svg_text_nodes(svg_text: str) -> list[str]:
    nodes = []
    for match in re.finditer(r"<text\b[^>]*>(.*?)</text>", svg_text, flags=re.DOTALL):
        value = re.sub(r"<[^>]+>", "", match.group(1))
        value = html.unescape(value)
        value = re.sub(r"\s+", " ", value).strip()
        if value:
            nodes.append(value)
    return nodes


def _entity_coverage(entities: list[str], svg_text: str) -> float:
    required = [entity for entity in entities if len(entity.strip()) >= 3]
    if not required:
        return 1.0
    lower_svg = svg_text.lower()
    svg_tokens = set(_tokens(svg_text))
    matched = 0
    for entity in required:
        if entity.lower() in lower_svg:
            matched += 1
            continue
        entity_tokens = set(_tokens(entity))
        if entity_tokens and entity_tokens.issubset(svg_tokens):
            matched += 1
    return matched / len(required)


def _copied_source_ratio(text_nodes: list[str], *, title: str, need: str) -> float:
    source_tokens = _tokens(f"{title} {need}")
    if not source_tokens:
        return 0.0
    svg_tokens = _tokens(" ".join(text_nodes))
    if not svg_tokens:
        return 0.0
    overlap = sum(1 for token in svg_tokens if token in source_tokens)
    return overlap / len(svg_tokens)


def _tokens(text: str) -> list[str]:
    ignored = {"slide", "source", "specific", "evidence", "show", "showing", "diagram"}
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower())
        if token not in ignored
    ]


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _markdown_report(payload: dict[str, object]) -> str:
    lines = [
        "# Programmatic Figure Semantic Review",
        "",
        f"- Status: {payload.get('status')}",
        f"- Figures reviewed: {payload.get('figures_reviewed')}",
    ]
    issue_counts = payload.get("issue_counts") if isinstance(payload.get("issue_counts"), dict) else {}
    lines.append(f"- Errors: {issue_counts.get('error', 0)}")
    lines.append(f"- Warnings: {issue_counts.get('warn', 0)}")
    lines.append("")
    issues = payload.get("issues") if isinstance(payload.get("issues"), list) else []
    if not issues:
        lines.append("No semantic figure issues found.")
    else:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            lines.append(
                "- "
                f"{issue.get('severity', 'unknown').upper()} "
                f"{issue.get('check_id')} "
                f"slide={issue.get('slide_number')} "
                f"file={issue.get('file')}: {issue.get('message')}"
            )
    lines.append("")
    return "\n".join(lines)
