#!/usr/bin/env python3
"""Review generated programmatic SVGs for semantic informativeness."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.workflow_runs import write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--presentation-type", required=True)
    parser.add_argument("--topic")
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--asset-inventory", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.presentation_type == "topic-deep-dive" and not args.topic:
        parser.error("--topic is required for topic-deep-dive semantic figure review")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = review_programmatic_figure_inventory(
        args.workspace,
        generated_dir=args.dir,
        asset_inventory=args.asset_inventory,
        report_output=args.report_output,
        markdown_output=args.markdown_output,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["status"] in {"fail", "failed"} else 0


def review_programmatic_figure_inventory(
    workspace_root: Path,
    *,
    generated_dir: Path,
    asset_inventory: Path,
    report_output: Path,
    markdown_output: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the Phase 15 semantic_review gate over source-grounded figure records."""

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
    inventory = _load_json(inventory_path)
    generated_assets = _generated_assets(inventory)
    figure_specs = _figure_specs(inventory)
    specs_by_key = {
        (str(spec.get("slide_id") or ""), str(spec.get("visual_id") or "")): spec
        for spec in figure_specs
    }

    findings: list[dict[str, Any]] = []
    reviewed_assets: list[dict[str, Any]] = []
    if not generated_assets:
        findings.append(
            _finding(
                "no_generated_assets",
                "error",
                "asset inventory has no generated_assets records",
                file=_relative(inventory_path, workspace),
            )
        )
    if not figure_specs:
        findings.append(
            _finding(
                "missing_figure_specs",
                "error",
                "asset inventory has no figure_specs records",
                file=_relative(inventory_path, workspace),
            )
        )

    for asset in generated_assets:
        key = (str(asset.get("slide_id") or ""), str(asset.get("visual_id") or ""))
        spec = specs_by_key.get(key, {})
        asset_findings = _review_asset(asset, spec, workspace=workspace, generated_root=generated_root)
        findings.extend(asset_findings)
        reviewed_assets.append(
            {
                "slide_id": key[0],
                "visual_id": key[1],
                "path": asset.get("path"),
                "status": "fail" if asset_findings else "pass",
                "required_entities": _string_list(spec.get("required_entities")),
                "required_relationships": _string_list(spec.get("required_relationships")),
                "source_labels": _string_list(spec.get("source_labels")),
                "source_ids": _string_list(spec.get("source_ids")),
            }
        )

    errors = [finding for finding in findings if finding["severity"] == "error"]
    payload: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "fail" if errors else "pass",
        "presentation_id": inventory.get("presentation_id"),
        "semantic_review": {
            "blocks_render_spec_generation": bool(errors),
            "render_spec_generation_allowed": not errors,
            "status_rule": "status is pass only when every generated asset passes",
        },
        "generated_dir": _relative(generated_root, workspace),
        "asset_inventory": _relative(inventory_path, workspace),
        "assets_reviewed": len(reviewed_assets),
        "findings": findings,
        "reviewed_assets": reviewed_assets,
    }
    if not dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(report_path, payload)
        if markdown_path is not None:
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def _review_asset(
    asset: dict[str, Any],
    spec: dict[str, Any],
    *,
    workspace: Path,
    generated_root: Path,
) -> list[dict[str, Any]]:
    del generated_root
    findings: list[dict[str, Any]] = []
    file_label = str(asset.get("path") or "<missing>")
    slide_id = str(asset.get("slide_id") or "")
    visual_id = str(asset.get("visual_id") or "")
    source_labels = _string_list(spec.get("source_labels")) or _string_list(asset.get("source_labels"))
    source_ids = _string_list(spec.get("source_ids")) or _string_list(asset.get("source_ids"))
    required_entities = _string_list(spec.get("required_entities"))
    required_relationships = _string_list(spec.get("required_relationships"))
    narration_hook = str(spec.get("narration_hook") or "").strip()
    depiction_type = str(spec.get("depiction_type") or "").strip().lower()
    brief_text = _brief_semantic_text(spec.get("figure_design_brief"))

    svg_text = ""
    text_nodes: list[str] = []
    if not file_label or file_label == "<missing>":
        findings.append(_finding("missing_asset_path", "error", "generated asset path is missing", slide_id=slide_id, visual_id=visual_id, file=file_label))
    else:
        svg_path = _resolve_inside(workspace, Path(file_label), label="generated asset path")
        if not svg_path.exists():
            findings.append(_finding("missing_asset_file", "error", "generated asset file is missing", slide_id=slide_id, visual_id=visual_id, file=file_label))
        else:
            svg_text = svg_path.read_text(encoding="utf-8", errors="replace")
            text_nodes = _svg_text_nodes(svg_text)
            expected_sha = str(asset.get("sha256") or "").strip()
            actual_sha = _sha256_file(svg_path)
            if expected_sha and expected_sha != actual_sha:
                findings.append(_finding("asset_hash_mismatch", "error", "generated asset sha256 does not match asset_inventory.json", slide_id=slide_id, visual_id=visual_id, file=file_label))

    if asset.get("production_evidence") is not True:
        findings.append(_finding("missing_production_evidence", "error", "production_evidence must be true for real generated assets", slide_id=slide_id, visual_id=visual_id, file=file_label))
    if not source_labels or not source_ids:
        findings.append(_finding("missing_source", "error", "missing source labels or source ids for source-grounded figure", slide_id=slide_id, visual_id=visual_id, file=file_label))
    if not required_entities:
        findings.append(_finding("missing_required_entities", "error", "required_entities must name depicted source entities", slide_id=slide_id, visual_id=visual_id, file=file_label))
    if not required_relationships:
        findings.append(_finding("missing_required_relationships", "error", "required_relationships must name depicted source relationships", slide_id=slide_id, visual_id=visual_id, file=file_label))

    if depiction_type in {"decorative", "ornament", "background"} or "decorative" in brief_text.lower():
        findings.append(_finding("decorative_visual", "error", "decorative visuals cannot satisfy semantic_review", slide_id=slide_id, visual_id=visual_id, file=file_label))

    if _looks_like_bullet_box(text_nodes=text_nodes, spec=spec):
        findings.append(_finding("bullet-box", "error", "bullet-box figures and copied text boxes are blocked", slide_id=slide_id, visual_id=visual_id, file=file_label))

    missing_entities = _missing_terms(required_entities, svg_text)
    missing_relationships = _missing_relationship_terms(required_relationships, svg_text)
    if missing_entities:
        findings.append(_finding("required_entity_coverage", "error", "required_entities absent from SVG text: " + ", ".join(missing_entities), slide_id=slide_id, visual_id=visual_id, file=file_label))
    if missing_relationships:
        findings.append(_finding("required_relationship_coverage", "error", "required_relationships absent from SVG text: " + ", ".join(missing_relationships), slide_id=slide_id, visual_id=visual_id, file=file_label))
    if _narration_mismatch(narration_hook, svg_text, required_entities, required_relationships):
        findings.append(_finding("visual/narration mismatch", "error", "figure text does not depict the narration hook entities and relationships", slide_id=slide_id, visual_id=visual_id, file=file_label))
    return findings


def _generated_assets(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    assets = inventory.get("generated_assets")
    if isinstance(assets, list):
        return [asset for asset in assets if isinstance(asset, dict)]
    legacy = inventory.get("figures")
    if isinstance(legacy, list):
        return [asset for asset in legacy if isinstance(asset, dict)]
    return []


def _figure_specs(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    specs = inventory.get("figure_specs")
    if isinstance(specs, list):
        return [spec for spec in specs if isinstance(spec, dict)]
    legacy = inventory.get("figures")
    if isinstance(legacy, list):
        rows: list[dict[str, Any]] = []
        for figure in legacy:
            if not isinstance(figure, dict):
                continue
            spec = figure.get("figure_spec")
            if isinstance(spec, dict):
                rows.append(
                    {
                        **spec,
                        "slide_id": str(figure.get("slide_id") or figure.get("slide_number") or ""),
                        "visual_id": str(figure.get("visual_id") or figure.get("path") or ""),
                    }
                )
        return rows
    return []


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_inside(workspace: Path, candidate: Path, *, label: str) -> Path:
    path = candidate.expanduser()
    if not path.is_absolute():
        path = path if path.exists() else workspace / path
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def _relative(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace).as_posix()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _svg_text_nodes(svg_text: str) -> list[str]:
    nodes: list[str] = []
    for match in re.finditer(r"<text\b[^>]*>(.*?)</text>", svg_text, flags=re.DOTALL):
        value = re.sub(r"<[^>]+>", "", match.group(1))
        value = html.unescape(value)
        value = re.sub(r"\s+", " ", value).strip()
        if value:
            nodes.append(value)
    return nodes


def _looks_like_bullet_box(*, text_nodes: list[str], spec: dict[str, Any]) -> bool:
    brief_text = _brief_semantic_text(spec.get("figure_design_brief")).lower()
    if "bullet box" in brief_text or "bullet-box" in brief_text or "summary card" in brief_text:
        return True
    if len(text_nodes) >= 3:
        short_lines = [node for node in text_nodes if len(node.split()) <= 10]
        if len(short_lines) == len(text_nodes) and not any("->" in node or "=>" in node for node in text_nodes):
            return True
    return False


def _brief_semantic_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        fields = [
            value.get("visual_role"),
            value.get("claim_to_visualize"),
            value.get("depiction_type"),
        ]
        return " ".join(str(field) for field in fields if str(field or "").strip())
    return ""


def _missing_terms(required: list[str], svg_text: str) -> list[str]:
    lower_svg = svg_text.lower()
    svg_tokens = set(_tokens(svg_text))
    missing: list[str] = []
    for term in required:
        if _term_present(term, lower_svg, svg_tokens):
            continue
        missing.append(term)
    return missing


def _missing_relationship_terms(relationships: list[str], svg_text: str) -> list[str]:
    lower_svg = svg_text.lower()
    svg_tokens = set(_tokens(svg_text))
    missing: list[str] = []
    for relationship in relationships:
        tokens = [token for token in _tokens(relationship) if token not in {"and", "the"}]
        if not tokens:
            continue
        matched = sum(1 for token in tokens if token in svg_tokens)
        if relationship.lower() in lower_svg or matched >= max(1, min(2, len(tokens))):
            continue
        missing.append(relationship)
    return missing


def _narration_mismatch(
    narration_hook: str,
    svg_text: str,
    required_entities: list[str],
    required_relationships: list[str],
) -> bool:
    if not narration_hook:
        return True
    lower_svg = svg_text.lower()
    svg_tokens = set(_tokens(svg_text))
    required_terms = [*required_entities, *required_relationships]
    if required_terms:
        missing_count = sum(
            1 for term in required_terms if not _term_present(term, lower_svg, svg_tokens)
        )
        return missing_count > len(required_terms) / 2
    hook_tokens = [token for token in _tokens(narration_hook) if len(token) > 4]
    if not hook_tokens:
        return True
    matched = sum(1 for token in hook_tokens if token in svg_tokens)
    return matched < max(1, len(hook_tokens) // 3)


def _term_present(term: str, lower_svg: str, svg_tokens: set[str]) -> bool:
    normalized = str(term).strip().lower()
    if not normalized:
        return True
    if normalized in lower_svg:
        return True
    tokens = set(_tokens(normalized))
    if bool(tokens) and tokens.issubset(svg_tokens):
        return True
    # Mirror the codegen acceptance rule: long multi-word entities may be
    # legitimately shortened during layout repair; a clear majority of the
    # distinctive tokens suffices.
    if len(tokens) >= 3:
        present = sum(1 for token in tokens if token in svg_tokens)
        return present / len(tokens) >= 0.6
    return False


def _tokens(text: str) -> list[str]:
    ignored = {"source", "slide", "figure", "visual", "show", "showing", "depict"}
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower())
        if token not in ignored
    ]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _finding(
    check_id: str,
    severity: str,
    message: str,
    *,
    file: str = "",
    slide_id: str = "",
    visual_id: str = "",
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "severity": severity,
        "message": message,
        "file": file,
        "slide_id": slide_id,
        "visual_id": visual_id,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Programmatic Figure Semantic Review",
        "",
        f"- semantic_review status: {payload.get('status')}",
        f"- Assets reviewed: {payload.get('assets_reviewed')}",
        "- Gate: render spec generation is blocked unless status is `pass`.",
        "- Required fields: required_entities, required_relationships, source_labels, source_ids.",
        "- Blocked classes: decorative visual, missing source, bullet-box, visual/narration mismatch.",
        "",
    ]
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    if not findings:
        lines.append("No semantic figure issues found.")
    else:
        lines.append("## Findings")
        lines.append("")
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            lines.append(
                "- "
                f"{finding.get('severity', 'unknown').upper()} "
                f"{finding.get('check_id')} "
                f"slide={finding.get('slide_id')} "
                f"visual={finding.get('visual_id')} "
                f"file={finding.get('file')}: {finding.get('message')}"
            )
    lines.append("")
    reviewed = payload.get("reviewed_assets") if isinstance(payload.get("reviewed_assets"), list) else []
    if reviewed:
        lines.append("## Reviewed Assets")
        lines.append("")
        for item in reviewed:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- "
                f"{item.get('status')} {item.get('slide_id')} {item.get('visual_id')} "
                f"required_entities={item.get('required_entities')} "
                f"required_relationships={item.get('required_relationships')}"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
