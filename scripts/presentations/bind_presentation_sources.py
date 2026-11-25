#!/usr/bin/env python3
"""Bind visible source references into source_binding_report.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.presentation.source_binding import (
    validate_source_binding_report_payload,
)
from conference_pipeline.workflow_runs import write_json_atomic

BLOCKED_VISIBLE_SOURCE_STATUSES = {
    "unresolved",
    "ambiguous",
    "title_mismatch",
    "homepage_only",
    "packet_too_large",
}


# Kept in step with generate_render_spec.BOOKEND_SLIDE_KINDS: only the two
# structural slides add_bookend_slides.py injects are skipped when binding
# sources. A deck that tags every slide with a slide_kind (the
# executive-overview family) still binds every editorial slide.
BOOKEND_SLIDE_KINDS = frozenset({"events", "resources"})


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _ledger_by_source_id(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("speaker_author_work_ledger.json must contain entries")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_id = str(entry.get("source_id") or "").strip()
        if source_id:
            by_id[source_id] = entry
    if not by_id:
        raise ValueError("speaker_author_work_ledger.json contains no ledger entries")
    return by_id


def _cluster_source_ids(cluster_plan: dict[str, Any]) -> set[str]:
    clusters = cluster_plan.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError("research_cluster_plan.json must contain clusters")
    source_ids: set[str] = set()
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        for field in ("primary_sources", "supporting_sources"):
            sources = cluster.get(field)
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                source_id = str(source.get("source_id") or "").strip()
                if source_id:
                    source_ids.add(source_id)
    if not source_ids:
        raise ValueError("research_cluster_plan.json contains no source ids")
    return source_ids


def _blocked_statuses(ledger: dict[str, Any]) -> set[str]:
    statuses = set(BLOCKED_VISIBLE_SOURCE_STATUSES)
    configured = ledger.get("blocked_visible_source_statuses")
    if isinstance(configured, list):
        statuses.update(str(status).strip() for status in configured if str(status).strip())
    return statuses


def _reference_from_ledger(
    *,
    label: str,
    source_id: str,
    ledger_entry: dict[str, Any],
    blocked_statuses: set[str],
) -> dict[str, Any]:
    status = str(ledger_entry.get("status") or "").strip()
    if status in blocked_statuses:
        raise ValueError(f"source {source_id} has blocked ledger status {status}")
    if ledger_entry.get("visible_source_chip_eligible") is not True:
        raise ValueError(f"source {source_id} is not eligible for a visible label")
    authors = _string_list(ledger_entry.get("authors"))
    speakers = _string_list(ledger_entry.get("speakers"))
    works = _string_list(ledger_entry.get("works"))
    title = str(ledger_entry.get("resolved_title") or ledger_entry.get("title") or "").strip()
    # A visible reference must name a real entity: a person OR a specific named
    # work. Title-only works (papers whose authors were not extracted) are bound
    # by their work title; the eligibility check above already blocks generic
    # placeholder titles, so an entry that is eligible with a resolved title is a
    # citable work even without resolved people.
    if not authors and not speakers and not works and not title:
        raise ValueError(f"source {source_id} is missing authors, speakers, or a named work")
    contribution_note = str(ledger_entry.get("contribution_note") or "").strip()
    source_family = str(ledger_entry.get("source_family") or "").strip()
    if not title:
        raise ValueError(f"source {source_id} is missing resolved title")
    if not contribution_note:
        raise ValueError(f"source {source_id} is missing contribution_note")
    if not source_family:
        raise ValueError(f"source {source_id} is missing source_family")
    confidence = ledger_entry.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError(f"source {source_id} is missing numeric confidence")
    evidence_paths = _string_list(ledger_entry.get("evidence_paths"))
    url = str(ledger_entry.get("url") or "").strip()
    reference: dict[str, Any] = {
        "label": label,
        "title": title,
        "source_id": source_id,
        "authors": authors,
        "speakers": speakers,
        "works": _string_list(ledger_entry.get("works")),
        "source_type": str(
            ledger_entry.get("source_type")
            or ("talk" if speakers else "paper")
        ),
        "contribution_note": contribution_note,
        "source_family": source_family,
        "confidence": float(confidence),
        "ledger_status": status or "resolved",
        "relevance_rationale": contribution_note,
    }
    if url:
        reference["url"] = url
    elif evidence_paths:
        reference["path"] = evidence_paths[0]
    else:
        raise ValueError(f"source {source_id} is missing URL/path evidence")
    if not authors and not speakers:
        # A recorded talk whose speaker could not be resolved is still a real,
        # attributable source: the talk itself. Attribute the visible reference
        # to the recording (title-only works otherwise fail the artifact-truth
        # author/speaker/talk/resource check) using only what the source id
        # proves -- the video id -- never a name guessed from captions.
        talk_attribution = _talk_attribution_from_source_id(source_id, source_family=source_family)
        if talk_attribution is not None:
            reference["talks"] = [talk_attribution["talk"]]
            reference["resources"] = [talk_attribution["resource"]]
            reference["source_type"] = "talk"
    return reference


_YOUTUBE_TRANSCRIPT_SOURCE_ID = re.compile(r"^speech_transcript:(?P<video_id>[A-Za-z0-9_-]{6,})(?::|$)")


def _talk_attribution_from_source_id(source_id: str, *, source_family: str) -> dict[str, str] | None:
    match = _YOUTUBE_TRANSCRIPT_SOURCE_ID.match(str(source_id or "").strip())
    if match is None:
        return None
    video_id = match.group("video_id")
    # Transcript-lane ids carry the lane as a prefix ("r7-caption-recovery_<id>",
    # "yt-organizer-channels-..._<id>"); a YouTube id is the trailing 11
    # characters. The whole token in a watch URL is a dead link.
    if len(video_id) > 11 and video_id[-12] == "_":
        video_id = video_id[-11:]
    return {
        "talk": f"Recorded talk (YouTube {video_id}, {source_family or 'transcript'} transcript)",
        "resource": f"https://www.youtube.com/watch?v={video_id}",
    }


def build_source_binding_report(
    *,
    deck_contract: dict[str, Any],
    approved_source_plan_sha256: str,
    speaker_author_work_ledger: dict[str, Any],
    research_cluster_plan: dict[str, Any],
    source_repair_findings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ledger_entries = _ledger_by_source_id(speaker_author_work_ledger)
    # A reviewed repair that unlocks visible use is the pipeline's sanctioned
    # answer to a blocked ("ambiguous") source: an operator recorded evidence
    # that it is what it says. The reviewed values replace the blocked ones,
    # exactly as blueprint generation applies them.
    repaired: dict[str, dict[str, Any]] = {}
    unrepaired_entries = dict(ledger_entries)
    for finding in (source_repair_findings or {}).get("findings") or []:
        if not isinstance(finding, dict) or finding.get("visible_use_unlocked") is not True:
            continue
        source_id = str(finding.get("source_id") or "").strip()
        value = finding.get("repaired_value")
        if source_id in ledger_entries and isinstance(value, dict):
            entry = dict(ledger_entries[source_id])
            entry.update({key: value[key] for key in value if value[key] not in (None, "", [])})
            entry["visible_source_chip_eligible"] = True
            ledger_entries[source_id] = entry
            repaired[source_id] = finding
    reviewed_repair_sources: list[dict[str, Any]] = []
    cluster_ids = _cluster_source_ids(research_cluster_plan)
    blocked = _blocked_statuses(speaker_author_work_ledger)
    slides = []
    dropped_unnamed_sources = []
    for slide in deck_contract.get("slides", []):
        if not isinstance(slide, dict):
            continue
        if str(slide.get("slide_kind") or "").strip().lower() in BOOKEND_SLIDE_KINDS:
            # Structural bookend slides (events/resources) are injected by
            # add_bookend_slides.py AFTER binding and carry no source-label
            # slots; a re-bind over a contract that already holds them must
            # skip them exactly as the bookend step itself does.
            continue
        refs = []
        labels = _string_list(slide.get("source_label_slots"))
        source_ids = _string_list(slide.get("source_ids"))
        if len(labels) != len(source_ids):
            raise ValueError(
                f"slide {slide.get('slide_id')} source label count does not match source_ids count"
            )
        slide_id = str(slide.get("slide_id") or "")
        for label, source_id in zip(labels, source_ids, strict=True):
            ledger_entry = ledger_entries.get(source_id)
            scope = _string_list(repaired.get(source_id, {}).get("slide_ids"))
            in_scope = source_id in repaired and (
                not scope or any(slide_id == item or slide_id.startswith(item) for item in scope)
            )
            if source_id in repaired and not in_scope:
                ledger_entry = unrepaired_entries.get(source_id)
            # The ledger is the eligibility authority. Editorial blueprints
            # legitimately select sources beyond the keyword-derived research
            # cluster plan, so absence there only blocks when the ledger does
            # not mark the source chip-eligible either. A source can reach
            # here with no ledger entry, or a real claim but no resolvable
            # title/author/speaker at all (a caption-only transcript segment
            # whose only "identity" was its own id's namespace prefix, fixed
            # in source_inventory.py to no longer fabricate one) -- refusing
            # the WHOLE deck's binding over one such citation blocked all
            # of a topic's other, real, well-attributed slides on it. Drop
            # just that one citation instead, recorded here rather than
            # silently vanished (the figure compiler's own rule for content
            # it cannot draw), and let the slide fall back to whichever of
            # its OTHER cited sources are real -- a slide left with zero
            # real citations still fails below ("visible_references must
            # contain at least one approved source"), which is the correct
            # place for THAT failure to surface, since it means the whole
            # slide's claim needs a real source, not just this one label.
            eligible = bool(ledger_entry) and (
                source_id in cluster_ids or ledger_entry.get("visible_source_chip_eligible")
            )
            if not eligible:
                dropped_unnamed_sources.append(
                    {
                        "slide_id": str(slide.get("slide_id") or ""),
                        "label": label,
                        "source_id": source_id,
                        "reason": (
                            "missing_from_ledger" if ledger_entry is None
                            else "not_chip_eligible_and_not_in_cluster_plan"
                        ),
                    }
                )
                continue
            if in_scope:
                reviewed_repair_sources.append(
                    {
                        "slide_id": str(slide.get("slide_id") or ""),
                        "source_id": source_id,
                        "reviewer_notes": str(repaired[source_id].get("reviewer_notes") or ""),
                    }
                )
            refs.append(
                _reference_from_ledger(
                    label=label,
                    source_id=source_id,
                    ledger_entry=ledger_entry,
                    blocked_statuses=blocked,
                )
            )
        slides.append(
            {
                "slide_id": str(slide.get("slide_id") or f"slide-{len(slides) + 1:03d}"),
                "visible_references": refs,
                "unresolved_sources": [],
            }
        )
    payload = {
        "schema_version": 1,
        "presentation_id": str(deck_contract.get("presentation_id") or "presentation"),
        "approved_source_plan_sha256": approved_source_plan_sha256,
        "dropped_unnamed_sources": dropped_unnamed_sources,
        "reviewed_repair_sources": reviewed_repair_sources,
        "slides": slides,
    }
    result = validate_source_binding_report_payload(payload)
    if result.errors:
        raise ValueError(json.dumps(result.to_dict(), indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--presentation-id")
    parser.add_argument("--presentation-type")
    parser.add_argument("--topic")
    parser.add_argument("--deck-contract", type=Path, required=True)
    parser.add_argument("--matched-papers", type=Path)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--source-pack", type=Path)
    parser.add_argument("--speaker-author-work-ledger", type=Path)
    parser.add_argument("--research-cluster-plan", type=Path)
    parser.add_argument("--source-repair-findings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    deck_contract = _load_json(args.deck_contract)
    ledger_path = args.speaker_author_work_ledger or args.deck_contract.parent / "speaker_author_work_ledger.json"
    cluster_plan_path = args.research_cluster_plan or args.deck_contract.parent / "research_cluster_plan.json"
    if not ledger_path.is_file():
        raise ValueError("source binding requires speaker_author_work_ledger.json for ledger visible label eligibility")
    if not cluster_plan_path.is_file():
        raise ValueError("source binding requires research_cluster_plan.json for source family binding")
    ledger = _load_json(ledger_path)
    cluster_plan = _load_json(cluster_plan_path)
    source_hash_basis = cluster_plan_path
    repair_path = args.source_repair_findings or args.deck_contract.parent / "source_repair_findings.json"
    repair_findings = _load_json(repair_path) if repair_path.is_file() else None
    payload = build_source_binding_report(
        deck_contract=deck_contract,
        approved_source_plan_sha256=sha256_file(source_hash_basis),
        speaker_author_work_ledger=ledger,
        research_cluster_plan=cluster_plan,
        source_repair_findings=repair_findings,
    )
    payload["input_hashes"] = {
        "presentation_deck.json": sha256_file(args.deck_contract),
        "speaker_author_work_ledger.json": sha256_file(ledger_path),
        "research_cluster_plan.json": sha256_file(cluster_plan_path),
    }
    if repair_findings is not None:
        payload["input_hashes"]["source_repair_findings.json"] = sha256_file(repair_path)
    if args.source_pack and args.source_pack.is_file():
        payload["input_hashes"]["source_pack"] = sha256_file(args.source_pack)
    if args.matched_papers and args.matched_papers.is_file():
        payload["input_hashes"]["matched_papers"] = sha256_file(args.matched_papers)
    if args.source_metadata and args.source_metadata.is_file():
        payload["input_hashes"]["source_metadata"] = sha256_file(args.source_metadata)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    write_json_atomic(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
