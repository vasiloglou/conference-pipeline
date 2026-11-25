#!/usr/bin/env python3
"""Compile an approved Presentation Blueprint into presentation_deck.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.presentation.deck_contract import validate_deck_contract_payload
from conference_pipeline.workflow_runs import write_json_atomic

APPROVED_BLUEPRINT_REVIEW_STATUSES = {
    "approved_for_deck_contract_generation",
    "approved_for_draft_generation",
}
REQUIRED_REVIEW_GATES = {"deterministic", "LLM-judge", "human"}
DEFAULT_MAX_SLIDES_WITHOUT_EXPANSION_JUSTIFICATION = 14
REQUIRED_APPROVED_EXPANSION_FIELDS = frozenset(
    {
        "approved",
        "target_slide_count",
        "max_runtime_minutes",
        "blueprint_sha256",
        "storyboard_sha256",
        "source_plan_sha256",
        "review_receipt_path",
        "rationale",
    }
)


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_sibling_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_json(path)


def _require_current_approved_blueprint_review(
    *,
    blueprint_path: Path,
    blueprint_receipt_path: Path,
    blueprint_review_path: Path,
    blueprint_sha256: str,
) -> dict[str, Any]:
    review = _load_json(blueprint_review_path)
    receipt = _load_json(blueprint_receipt_path)
    errors: list[str] = []
    if review.get("status") != "pass":
        errors.append("blueprint_review.json status must be pass")
    approval = review.get("approval")
    if not isinstance(approval, dict) or approval.get("status") not in APPROVED_BLUEPRINT_REVIEW_STATUSES:
        errors.append(
            "blueprint_review.json approval.status must be approved_for_deck_contract_generation "
            "or approved_for_draft_generation"
        )
    if review.get("blueprint_sha256") != blueprint_sha256:
        errors.append("blueprint_review.json blueprint_sha256 does not match presentation_blueprint.json")
    if receipt.get("blueprint_sha256") != blueprint_sha256:
        errors.append("presentation_blueprint_receipt.json blueprint_sha256 does not match presentation_blueprint.json")
    gate_types = review.get("gate_types")
    if not isinstance(gate_types, dict):
        errors.append("blueprint_review.json must record gate_types")
    else:
        for gate in sorted(REQUIRED_REVIEW_GATES):
            if gate_types.get(gate) != "pass":
                errors.append(f"blueprint_review.json gate_types.{gate} must be pass")
    if errors:
        details = "; ".join(errors)
        raise ValueError(f"current approved blueprint review required: {details}")
    return review


def _has_approved_expansion_justification(
    blueprint: dict[str, Any],
    *,
    slide_count: int,
) -> bool:
    expansion = blueprint.get("approved_expansion")
    if not isinstance(expansion, dict):
        return False
    if REQUIRED_APPROVED_EXPANSION_FIELDS - set(expansion):
        return False
    if expansion.get("approved") is not True:
        return False
    if expansion.get("target_slide_count") != slide_count:
        return False
    runtime = expansion.get("max_runtime_minutes")
    if not isinstance(runtime, int | float) or runtime <= 0 or runtime > 40:
        return False
    for field in ("blueprint_sha256", "storyboard_sha256", "source_plan_sha256"):
        value = expansion.get(field)
        if not isinstance(value, str) or len(value) != 64:
            return False
    return True


def _source_records_by_cluster(cluster_plan: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(cluster_plan, dict):
        return records
    clusters = cluster_plan.get("clusters")
    if not isinstance(clusters, list):
        return records
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        source_records: list[dict[str, Any]] = []
        for field in ("primary_sources", "supporting_sources"):
            values = cluster.get(field)
            if isinstance(values, list):
                source_records.extend(item for item in values if isinstance(item, dict))
        records[cluster_id] = source_records
    return records


def _ledger_by_source_id(ledger: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(ledger, dict):
        return {}
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_id = str(entry.get("source_id") or "").strip()
        if source_id:
            by_id[source_id] = entry
    return by_id


def _slot_label(slot: Any) -> str:
    if isinstance(slot, dict):
        return str(slot.get("label") or "").strip()
    return str(slot).strip()


def _slot_cluster_id(slot: Any) -> str:
    if isinstance(slot, dict):
        return str(slot.get("cluster_id") or "").strip()
    return ""


def _source_index_from_label(label: str) -> int | None:
    match = re.fullmatch(r"SRC-(\d+)", label.strip())
    if not match:
        return None
    return int(match.group(1)) - 1


def _mention_from_source(
    *,
    label: str,
    source: dict[str, Any],
    ledger_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    source_id = str((ledger_entry or source).get("source_id") or "").strip()
    title = str(
        (ledger_entry or source).get("resolved_title")
        or (ledger_entry or source).get("title")
        or ""
    ).strip()
    authors = _string_list((ledger_entry or source).get("authors"))
    speakers = _string_list((ledger_entry or source).get("speakers"))
    works = _string_list((ledger_entry or source).get("works"))
    contribution_note = str((ledger_entry or source).get("contribution_note") or "").strip()
    return {
        "label": label,
        "source_id": source_id,
        "title": title,
        "authors": authors,
        "speakers": speakers,
        "works": works or ([title] if title else []),
        "contribution_note": contribution_note,
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _mentions_from_blueprint(slide: dict[str, Any]) -> list[dict[str, Any]]:
    mentions = slide.get("speaker_author_work_mentions") or slide.get(
        "named_paper_talk_speaker_author_mentions"
    )
    if not isinstance(mentions, list):
        return []
    normalized: list[dict[str, Any]] = []
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        label = str(mention.get("label") or "").strip()
        source_id = str(mention.get("source_id") or "").strip()
        title = str(mention.get("title") or mention.get("resolved_title") or "").strip()
        works = _string_list(mention.get("works"))
        if not title and works:
            title = works[0]
        normalized.append(
            {
                "label": label,
                "source_id": source_id,
                "title": title,
                "authors": _string_list(mention.get("authors")),
                "speakers": _string_list(mention.get("speakers")),
                "works": works or ([title] if title else []),
                "contribution_note": str(mention.get("contribution_note") or "").strip(),
            }
        )
    return normalized


def _source_mentions_for_slide(
    slide: dict[str, Any],
    *,
    source_slots: Sequence[Any],
    cluster_sources: dict[str, list[dict[str, Any]]],
    ledger_entries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    # SRC-NNN labels are GLOBAL handles: one stable label per source across the
    # whole deck. Resolve each label via the blueprint's own mention (which holds
    # the correct global label->source mapping) and enrich it from the ledger. Do
    # NOT map a label's number to a per-cluster source index ("SRC-001" is not
    # "this cluster's slot 1") — that made the same label resolve to a different
    # source on every slide and bound two labels to one source.
    blueprint_mentions = {
        str(mention.get("label") or ""): mention for mention in _mentions_from_blueprint(slide)
    }
    for slot in source_slots:
        label = _slot_label(slot)
        if not label:
            continue
        mention = blueprint_mentions.get(label)
        if mention is None:
            continue
        source_id = str(mention.get("source_id") or "").strip()
        ledger_entry = ledger_entries.get(source_id)
        if ledger_entry:
            mention = _mention_from_source(label=label, source=mention, ledger_entry=ledger_entry)
        mentions.append(mention)
    if mentions:
        return mentions
    return _mentions_from_blueprint(slide)


def _validate_generated_source_mentions(
    *,
    slide_id: str,
    labels: Sequence[str],
    mentions: Sequence[dict[str, Any]],
) -> None:
    missing: list[str] = []
    mentions_by_label = {str(mention.get("label") or ""): mention for mention in mentions}
    for label in labels:
        mention = mentions_by_label.get(label)
        if not mention:
            missing.append(label)
            continue
        if not str(mention.get("source_id") or "").strip():
            missing.append(f"{label}:source_id")
        if not str(mention.get("title") or "").strip():
            missing.append(f"{label}:title")
        if not str(mention.get("contribution_note") or "").strip():
            missing.append(f"{label}:contribution_note")
    if missing:
        raise ValueError(
            f"slide {slide_id} is missing source-ledger mentions for: {', '.join(missing)}"
        )


def _slide_from_blueprint(
    slide: dict[str, Any],
    *,
    index: int,
    cluster_sources: dict[str, list[dict[str, Any]]],
    ledger_entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    slide_id = str(slide.get("slide_id") or f"slide-{index + 1:03d}")
    source_labels = slide.get("source_labels") or slide.get("source_plan") or []
    if isinstance(source_labels, list):
        slots = [
            str(item.get("label") if isinstance(item, dict) else item).strip()
            for item in source_labels
            if str(item.get("label") if isinstance(item, dict) else item).strip()
        ]
    else:
        slots = []
    visual_role = str(slide.get("visual_role") or slide.get("visual_role_description") or "story visual")
    mentions = _source_mentions_for_slide(
        slide,
        source_slots=source_labels if isinstance(source_labels, list) else [],
        cluster_sources=cluster_sources,
        ledger_entries=ledger_entries,
    )
    source_ids = [str(mention.get("source_id") or "").strip() for mention in mentions]
    source_ids = [source_id for source_id in source_ids if source_id]
    _validate_generated_source_mentions(slide_id=slide_id, labels=slots, mentions=mentions)
    return {
        "slide_id": slide_id,
        "section_id": str(slide.get("section_id") or slide.get("section") or "section-1"),
        "title": str(slide.get("title") or slide.get("intent") or f"Slide {index + 1}"),
        "claims": [str(claim) for claim in slide.get("claims", [])] or [str(slide.get("intent") or "Source-grounded claim")],
        "visible_text": [str(text) for text in slide.get("visible_text", [])] or [str(slide.get("intent") or "Source-grounded slide")],
        "source_label_slots": slots,
        "source_ids": source_ids,
        "named_paper_talk_speaker_author_mentions": list(mentions),
        "author_attribution": "Author: Nikolaos Vasiloglou",
        "analysis_report_link_required": True,
        "visual_role": visual_role,
        "visual_placeholders": [
            {
                "visual_id": f"{slide_id}-visual",
                "role": visual_role,
            }
        ],
        "narration_hooks": [str(slide.get("visual_narration_hook") or slide.get("narration_beat") or "Explain the visual evidence.")],
        "timing_beats": [
            {
                "beat_id": f"{slide_id}-beat",
                "seconds": float(slide.get("estimated_seconds") or slide.get("estimated_minutes", 0) * 60 or 30),
            }
        ],
    }


def build_deck_contract(
    *,
    blueprint: dict[str, Any],
    blueprint_sha256: str,
    cluster_plan: dict[str, Any] | None = None,
    speaker_author_work_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slides = blueprint.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("presentation_blueprint.json must contain slides")
    if (
        len(slides) > DEFAULT_MAX_SLIDES_WITHOUT_EXPANSION_JUSTIFICATION
        and not _has_approved_expansion_justification(blueprint, slide_count=len(slides))
    ):
        raise ValueError(
            "presentation_blueprint.json has more than 14 slides without explicit approved_expansion"
        )
    sections = blueprint.get("telescopic_section_map")
    if not isinstance(sections, list) or not sections:
        sections = [{"section_id": "section-1", "title": "Presentation"}]
    cluster_sources = _source_records_by_cluster(cluster_plan)
    ledger_entries = _ledger_by_source_id(speaker_author_work_ledger)
    payload = {
        "schema_version": 1,
        "presentation_id": str(blueprint.get("blueprint_id") or "presentation"),
        "blueprint_sha256": blueprint_sha256,
        "deck_thesis": str(blueprint.get("deck_thesis") or ""),
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": str(blueprint.get("analysis_report_url") or blueprint.get("final_report_url") or "https://drive.google.com/"),
        "cluster_coverage_justification": blueprint.get("cluster_coverage_justification"),
        "sections": sections,
        "slides": [
            _slide_from_blueprint(
                slide if isinstance(slide, dict) else {},
                index=index,
                cluster_sources=cluster_sources,
                ledger_entries=ledger_entries,
            )
            for index, slide in enumerate(slides)
        ],
    }
    if blueprint.get("approved_expansion") is not None:
        payload["approved_expansion"] = blueprint.get("approved_expansion")
    result = validate_deck_contract_payload(payload)
    if result.errors:
        raise ValueError(json.dumps(result.to_dict(), indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--presentation-id")
    parser.add_argument("--presentation-type")
    parser.add_argument("--topic")
    parser.add_argument("--presentation-blueprint", type=Path, required=True)
    parser.add_argument("--blueprint-receipt", type=Path, required=True)
    parser.add_argument("--blueprint-review", type=Path, required=True)
    parser.add_argument("--research-cluster-plan", type=Path)
    parser.add_argument("--speaker-author-work-ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    blueprint = _load_json(args.presentation_blueprint)
    blueprint_sha256 = sha256_file(args.presentation_blueprint)
    _require_current_approved_blueprint_review(
        blueprint_path=args.presentation_blueprint,
        blueprint_receipt_path=args.blueprint_receipt,
        blueprint_review_path=args.blueprint_review,
        blueprint_sha256=blueprint_sha256,
    )
    cluster_plan_path = args.research_cluster_plan or args.presentation_blueprint.parent / "research_cluster_plan.json"
    ledger_path = args.speaker_author_work_ledger or args.presentation_blueprint.parent / "speaker_author_work_ledger.json"
    cluster_plan = _load_sibling_json(cluster_plan_path)
    ledger = _load_sibling_json(ledger_path)
    payload = build_deck_contract(
        blueprint=blueprint,
        blueprint_sha256=blueprint_sha256,
        cluster_plan=cluster_plan,
        speaker_author_work_ledger=ledger,
    )
    input_hashes = {
        "presentation_blueprint.json": blueprint_sha256,
        "presentation_blueprint_receipt.json": sha256_file(args.blueprint_receipt),
        "blueprint_review.json": sha256_file(args.blueprint_review),
    }
    if cluster_plan_path.is_file():
        input_hashes["research_cluster_plan.json"] = sha256_file(cluster_plan_path)
    if ledger_path.is_file():
        input_hashes["speaker_author_work_ledger.json"] = sha256_file(ledger_path)
    receipt = {
        "schema_version": 1,
        "generated_at": _isoformat_z(),
        "generator": "scripts/presentations/generate_deck_contract.py",
        "status": "dry_run" if args.dry_run else "succeeded",
        "presentation_id": args.presentation_id,
        "presentation_blueprint": str(args.presentation_blueprint),
        "blueprint_receipt": str(args.blueprint_receipt),
        "blueprint_review": str(args.blueprint_review),
        "blueprint_sha256": blueprint_sha256,
        "input_hashes": input_hashes,
        "presentation_deck": str(args.output),
        "presentation_deck_sha256": None if args.dry_run else None,
    }
    if args.dry_run:
        print(json.dumps({"presentation_deck": payload, "receipt": receipt}, indent=2, sort_keys=True))
        return 0
    write_json_atomic(args.output, payload)
    receipt["presentation_deck_sha256"] = sha256_file(args.output)
    write_json_atomic(args.receipt_output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
