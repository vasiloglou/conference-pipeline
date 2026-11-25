#!/usr/bin/env python3
"""Prepare topic-deep-dive presentation_frame.json handoffs.

The live topic presentation builder intentionally fails closed when
``decks/<topic>/presentation_frame.json`` is missing.  This helper creates the
audience-facing frame after analysis synthesis exists and before blueprint
generation begins.  It is deterministic and provider-free: it reads the topic
manifest plus analysis markdown, writes validated frame contracts, and records
a receipt for the production supervisor.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conference_pipeline.analysis._digest import sha256_file  # noqa: E402
from conference_pipeline.conference_identity import conference_name  # noqa: E402
from conference_pipeline.presentation.presentation_frame import (  # noqa: E402
    validate_presentation_frame_payload,
)


DEFAULT_GENERAL_QUESTIONS = (
    "What changed in {topic} at {conference_label}, beyond the headline results?",
    "Which claims are supported by the strongest paper, workshop, and transcript evidence?",
    "Where do the methods still break under domain constraints, data limits, or deployment pressure?",
    "What should a technical audience watch next as this area matures?",
)

# Optional per-workspace audience-title overrides, read from
# ``decks/topic_title_overrides.json`` when present. No conference-specific
# titles are hardcoded here; the audience title otherwise comes from the topic's
# manifest ``title``.
TOPIC_TITLE_OVERRIDES_FILENAME = "topic_title_overrides.json"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _workspace(path: str) -> Path:
    workspace = Path(path)
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace
    return workspace


def _topic_id(topic: dict[str, Any]) -> str:
    for key in ("topic_id", "id", "presentation_id"):
        value = str(topic.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"topic row lacks topic_id/id: {topic!r}")


def _topic_title(topic: dict[str, Any], topic_id: str) -> str:
    for key in ("title", "topic_title", "display_title", "name"):
        value = str(topic.get(key) or "").strip()
        if value:
            return value
    return topic_id.replace("_", " ").strip().capitalize()


def load_topics(workspace: Path, selected: set[str] | None = None) -> list[dict[str, str]]:
    manifest_path = workspace / "analysis" / "analysis_manifest.json"
    strategy_path = workspace / "analysis" / "topic_strategy.json"
    rows: list[dict[str, Any]] = []
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        topics = manifest.get("topics")
        if isinstance(topics, list):
            rows = [row for row in topics if isinstance(row, dict)]
    if not rows and strategy_path.is_file():
        strategy = _load_json(strategy_path)
        topics = strategy.get("final_topics")
        if isinstance(topics, list):
            rows = [row for row in topics if isinstance(row, dict)]
    if not rows:
        raise FileNotFoundError(
            f"no topic rows found in {manifest_path} or {strategy_path}"
        )

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        topic_id = _topic_id(row)
        if selected and topic_id not in selected:
            continue
        if topic_id in seen:
            continue
        seen.add(topic_id)
        source_path = str(
            row.get("analysis_path")
            or row.get("deep_analysis_path")
            or row.get("path")
            or f"analysis/{topic_id}.md"
        )
        result.append(
            {
                "topic_id": topic_id,
                "title": _topic_title(row, topic_id),
                "source_analysis_path": source_path,
            }
        )
    if selected:
        missing = sorted(selected - {row["topic_id"] for row in result})
        if missing:
            raise ValueError(f"selected topic(s) not present in manifest: {', '.join(missing)}")
    return result


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _section(markdown: str, heading: str) -> str:
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s+|\Z)",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(markdown)
    return match.group(1).strip() if match else ""


def _first_paragraph(text: str) -> str:
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        cleaned = _normalize_text(re.sub(r"^[-*]\s+", "", paragraph.strip()))
        if cleaned:
            return cleaned
    return ""


def _cap_sentence(text: str, *, max_chars: int = 420) -> str:
    text = _normalize_text(text)
    if len(text) <= max_chars:
        return text
    candidates = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    length = 0
    for sentence in candidates:
        if not sentence:
            continue
        projected = length + (1 if out else 0) + len(sentence)
        if projected > max_chars:
            break
        out.append(sentence)
        length = projected
    if out:
        return " ".join(out)
    return text[: max_chars - 1].rstrip() + "…"


def derive_field_thesis(markdown: str, *, topic_title: str, conference_label: str) -> str:
    summary = _first_paragraph(_section(markdown, "Executive Summary"))
    if summary:
        return _cap_sentence(summary)
    return (
        f"The {conference_label} evidence for {topic_title} should be framed as a "
        "field-level story first: what changed, which evidence is strongest, "
        "where the systems still fail, and what decisions follow from the corpus."
    )


def derive_evidence_threads(markdown: str) -> list[str]:
    bullets: list[str] = []
    for section_name in ("Key Findings", "Slide Brief", "Recommendations"):
        section = _section(markdown, section_name)
        for line in section.splitlines():
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue
            cleaned = _normalize_text(stripped.lstrip("- ").strip())
            cleaned = re.sub(r"^\*\*(.*?)\*\*[:—-]?\s*", r"\1", cleaned)
            if cleaned:
                bullets.append(_cap_sentence(cleaned, max_chars=150))
            if len(bullets) >= 5:
                return bullets
    return bullets


def load_topic_title_overrides(workspace: Path) -> dict[str, str]:
    """Optional per-workspace audience-title overrides.

    Reads ``decks/topic_title_overrides.json`` (a flat ``{topic_id: title}``
    map) when present; returns an empty mapping otherwise. This replaces the
    former hardcoded module dict so audience titles default to the manifest.
    """
    path = workspace / "decks" / TOPIC_TITLE_OVERRIDES_FILENAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value).strip()
        for key, value in payload.items()
        if isinstance(value, str) and str(value).strip()
    }


def audience_title(
    topic_id: str, retrieval_title: str, overrides: dict[str, str] | None = None
) -> str:
    override = (overrides or {}).get(topic_id)
    if override:
        return override
    title = retrieval_title.strip().rstrip(".")
    if len(title) <= 72:
        return title
    return title[:71].rstrip(" ,;:-") + "…"


def build_frame(
    *,
    workspace: Path,
    topic: dict[str, str],
    conference_label: str,
    title_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, Any], Path]:
    topic_id = topic["topic_id"]
    source_rel = Path(topic["source_analysis_path"])
    source_path = workspace / source_rel
    if not source_path.is_file():
        raise FileNotFoundError(f"analysis report missing for {topic_id}: {source_rel}")
    markdown = source_path.read_text(encoding="utf-8", errors="replace")
    retrieval_title = topic["title"]
    field_thesis = derive_field_thesis(
        markdown,
        topic_title=retrieval_title,
        conference_label=conference_label,
    )
    questions = [
        template.format(topic=retrieval_title, conference_label=conference_label)
        for template in DEFAULT_GENERAL_QUESTIONS
    ]
    evidence_threads = derive_evidence_threads(markdown)
    frame: dict[str, Any] = {
        "schema_version": 1,
        "presentation_id": topic_id,
        "topic_id": topic_id,
        "retrieval_title": retrieval_title,
        "audience_title": audience_title(topic_id, retrieval_title, title_overrides),
        "field_thesis": field_thesis,
        "general_questions": questions,
        "source_analysis_path": source_rel.as_posix(),
        "analysis_evidence": {
            "source_lines": [
                f"{source_rel.as_posix()} Executive Summary",
                f"{source_rel.as_posix()} Key Findings",
                f"{source_rel.as_posix()} Slide Brief",
            ],
            "source_analysis_sha256": sha256_file(source_path),
            "broad_evidence_threads": evidence_threads,
            "specialist_terms_deferred_from_opening": [
                "method-local mechanisms",
                "paper-specific ablations",
                "benchmark-table details",
                "architecture diagrams",
            ],
        },
        "opening_anchor_policy": {
            "preferred_opening_evidence": [
                "analysis executive summary",
                "validated topic reconciliation and reference ledger",
                "exhaustive paper summary",
                "exhaustive transcript summary",
                "source-bound key findings and slide brief",
            ],
            "rules": [
                "Use the audience_title for the title slide and opening thesis.",
                "Open with field-level stakes and questions before paper mechanisms.",
                "Introduce specialist papers as evidence for the frame, not as the frame itself.",
                "Prefer reconciled, named, source-bound evidence over unresolved artifacts.",
            ],
        },
        "generated_at": _now(),
        "generator": "scripts/presentations/prepare_topic_presentation_frames.py",
    }
    return frame, source_path


def prepare_frames(
    *,
    workspace: Path,
    topics: list[dict[str, str]],
    conference_label: str,
    overwrite: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    title_overrides = load_topic_title_overrides(workspace)
    for topic in topics:
        topic_id = topic["topic_id"]
        deck_dir = workspace / "decks" / topic_id
        frame_path = deck_dir / "presentation_frame.json"
        if frame_path.is_file() and not overwrite:
            existing = _load_json(frame_path)
            validation = validate_presentation_frame_payload(existing).to_dict()
            rows.append(
                {
                    "topic_id": topic_id,
                    "status": "existing",
                    "frame_path": str(frame_path.relative_to(workspace)),
                    "presentation_frame_validation": validation,
                    "presentation_frame_sha256": sha256_file(frame_path),
                }
            )
            continue
        frame, source_path = build_frame(
            workspace=workspace,
            topic=topic,
            conference_label=conference_label,
            title_overrides=title_overrides,
        )
        validation = validate_presentation_frame_payload(frame)
        if validation.status != "pass":
            rows.append(
                {
                    "topic_id": topic_id,
                    "status": "failed_validation",
                    "frame_path": str(frame_path.relative_to(workspace)),
                    "presentation_frame_validation": validation.to_dict(),
                }
            )
            continue
        _write_json(frame_path, frame)
        rows.append(
            {
                "topic_id": topic_id,
                "status": "written",
                "frame_path": str(frame_path.relative_to(workspace)),
                "source_analysis_path": str(source_path.relative_to(workspace)),
                "source_analysis_sha256": sha256_file(source_path),
                "presentation_frame_sha256": sha256_file(frame_path),
                "presentation_frame_validation": validation.to_dict(),
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--conference-label",
        default=None,
        help="Conference display label; defaults to conference.name from the "
        "workspace config.yaml when omitted.",
    )
    parser.add_argument("--topic", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--receipt-path", default="")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = _workspace(args.workspace)
    conference_label = args.conference_label or conference_name(
        workspace, default="the conference"
    )
    selected = set(args.topic) if args.topic else None
    receipt_path = (
        workspace / args.receipt_path
        if args.receipt_path
        else workspace
        / "storage"
        / "workflow-runs"
        / (args.run_id or "presentation-frame-prep")
        / "presentation-frame-prep.json"
    )
    try:
        topics = load_topics(workspace, selected)
        rows = prepare_frames(
            workspace=workspace,
            topics=topics,
            conference_label=conference_label,
            overwrite=args.overwrite,
        )
        failed = [row for row in rows if row.get("status") == "failed_validation"]
        receipt = {
            "schema_version": 1,
            "status": "failed" if failed else "succeeded",
            "generated_at": _now(),
            "generator": "scripts/presentations/prepare_topic_presentation_frames.py",
            "workspace": str(workspace),
            "run_id": args.run_id or None,
            "conference_label": conference_label,
            "topic_count": len(rows),
            "written_count": sum(1 for row in rows if row.get("status") == "written"),
            "existing_count": sum(1 for row in rows if row.get("status") == "existing"),
            "failed_count": len(failed),
            "frames": rows,
        }
        _write_json(receipt_path, receipt)
        if args.summary:
            print(
                "presentation frames: "
                f"status={receipt['status']} written={receipt['written_count']} "
                f"existing={receipt['existing_count']} failed={receipt['failed_count']} "
                f"receipt={receipt_path.relative_to(workspace)}"
            )
        return 1 if failed else 0
    except Exception as exc:  # noqa: BLE001 - CLI receipt boundary
        receipt = {
            "schema_version": 1,
            "status": "failed",
            "generated_at": _now(),
            "generator": "scripts/presentations/prepare_topic_presentation_frames.py",
            "workspace": str(workspace),
            "run_id": args.run_id or None,
            "error": str(exc),
        }
        _write_json(receipt_path, receipt)
        print(f"prepare_topic_presentation_frames failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
