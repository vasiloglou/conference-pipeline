"""Build hash-bound source corpus inventories for presentation canaries."""

from __future__ import annotations

from conference_pipeline.presentation.contribution_notes import _LEDGER_SECTION_NOTE_RE, _note_matches_work

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conference_pipeline.conference_identity import conference_name
from conference_pipeline.presentation.display_text import clean_display_text, latex_to_text
from conference_pipeline.workflow_runs import write_json_atomic

SOURCE_CORPUS_INVENTORY_SCHEMA_VERSION = 1
SPEAKER_AUTHOR_WORK_LEDGER_SCHEMA_VERSION = 1
SOURCE_REPAIR_FINDINGS_SCHEMA_VERSION = 1
RESEARCH_CLUSTER_PLAN_SCHEMA_VERSION = 1
PRIMARY_SOURCE_CONFIDENCE_THRESHOLD = 0.75
SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD = 0.60
SOURCE_FAMILY_BALANCE_MAX_FRACTION = 0.70
# Lever (b), Phase 10.2: the literal marker a narrative-source-pack chunk's
# PDF-extracted `quote` must contain for its trailing text to be treated as
# usable abstract prose, and the minimum length (after `_clean_contribution_note`)
# that trailing text must have to be persisted as a derived contribution note.
# Measured against the live corpus (10.2-RESEARCH.md Q1): ~80% of current
# paper-family candidates clear this bar; the remaining ~20% have no literal
# marker in their extracted text and are left ineligible by design, not patched
# around with a bibliographic or constant substitute.
DERIVED_ABSTRACT_NOTE_MARKER = "Abstract "
DERIVED_ABSTRACT_NOTE_MIN_CHARS = 40
BLOCKED_VISIBLE_SOURCE_STATUSES = frozenset(
    {
        "unresolved",
        "ambiguous",
        "title_mismatch",
        "homepage_only",
        "packet_too_large",
        "off_topic_artifact",
    }
)

# Provenance marker written on every ledger row whose resolved title came from
# the deterministic local event-media join.
#
# This key and value are a CROSS-PLAN CONTRACT with the pseudo-people guard in
# `scripts/presentations/generate_editorial_blueprint.py`, which tests
# `entry.get("resolved_title_kind") != "local_join"` by EXACT EQUALITY at both
# of `_people_from_title`'s call sites. A missing key, an empty value, or any
# other value all read as UNMARKED and re-open the fabrication path that was
# measured to present 341 distinct noun phrases -- "Oral Session", "Computer
# Vision", "Artificial Intelligence" -- as named human researchers across the
# twelve CVPR2026 decks. The marker is written independently of whether the
# author join resolved: the ~19 deck videos with a real title and no joinable
# authors are precisely the rows the guard exists to protect.
LOCAL_JOIN_RESOLVED_TITLE_KEY = "resolved_title_kind"
LOCAL_JOIN_RESOLVED_TITLE_KIND = "local_join"
# A long omnibus recording (a whole workshop day) is segmented into many
# three-minute units that the LLM full-read classifies independently. A few of
# those segments legitimately touch another topic ("6G will be designed by
# agentic AI"), which is enough for the RECORDING to enter that topic's source
# pool with a high per-segment confidence -- and then to surface on the deck's
# opening "major events" slide as if the telco workshop were an agentic-AI
# event. The artifact-level topic share separates "this recording is about the
# topic" from "this recording mentions the topic": an omnibus artifact whose
# segments mostly classify elsewhere is not a visible source for this deck.
OMNIBUS_ARTIFACT_MIN_SEGMENTS = 30
OMNIBUS_ARTIFACT_MIN_TOPIC_SHARE = 0.25

# Research clusters are DERIVED per topic from the topic's own slide brief and
# analysis (see ``derive_research_cluster_map``). This generic, topic-neutral
# fallback is used only when no slide-brief Key Points and no narrative themes can
# be resolved, so the deck stays source-grounded for any conference/topic rather
# than carrying one topic's content as a hidden default.
_KEYWORD_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
        "is", "are", "be", "as", "at", "by", "from", "that", "this", "these", "those",
        "it", "its", "into", "than", "then", "not", "no", "can", "cannot", "must",
        "should", "now", "more", "most", "less", "via", "across", "their", "they",
        "becoming", "become", "becomes", "remain", "remains", "central", "concrete",
        "favored", "single", "rather", "uniform", "when", "what", "how", "why", "which",
        "whose", "still", "over", "while", "about", "between", "each", "both", "one",
    }
)


_OLD_NUMBERED_MARKDOWN_RE = re.compile(r"^\d+(?:\.\d+)?_draft_.*\.md$")
_SOURCE_REFERENCE_RE = re.compile(
    r"(?:paper_abstract:[A-Za-z0-9]+|speech_transcript:[A-Za-z0-9]+:segment:\d+)"
)


@dataclass(frozen=True)
class InventoryRecord:
    kind: str
    path: str
    sha256: str
    byte_count: int
    production_evidence: bool
    evidence_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "production_evidence": self.production_evidence,
            "evidence_status": self.evidence_status,
        }


@dataclass(frozen=True)
class SkippedInventoryRecord:
    kind: str
    path: str
    reason: str
    production_evidence: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "reason": self.reason,
            "production_evidence": self.production_evidence,
        }


@dataclass(frozen=True)
class InventoryBlocker:
    code: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


@dataclass(frozen=True)
class SpeakerAuthorWorkLedgerEntry:
    source_id: str
    resolved_title: str
    authors: list[str]
    speakers: list[str]
    works: list[str]
    source_family: str
    confidence: float
    status: str
    visible_source_chip_eligible: bool
    contribution_note: str
    rank: int
    evidence_paths: list[str]
    packet_token_estimate: int | None = None
    packet_raw_candidate_count: int | None = None
    packet_max_candidates_per_reference: int | None = None
    packet_reference_line_text: str = ""
    contribution_note_kind: str = ""
    # Additive, optional, LAST. Emitted only when non-empty (see `to_dict`), so
    # every pre-existing entry keeps its exact key set.
    resolved_title_kind: str = ""
    artifact_topic_share: float | None = None
    artifact_dominant_topic_id: str = ""
    artifact_segment_count: int | None = None
    contribution_note_source: str = "none"

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_id": self.source_id,
            "resolved_title": self.resolved_title,
            "authors": self.authors,
            "speakers": self.speakers,
            "works": self.works,
            "source_family": self.source_family,
            "confidence": self.confidence,
            "status": self.status,
            "visible_source_chip_eligible": self.visible_source_chip_eligible,
            "contribution_note": self.contribution_note,
            "contribution_note_source": self.contribution_note_source,
            "rank": self.rank,
            "evidence_paths": self.evidence_paths,
        }
        if self.packet_token_estimate is not None:
            payload["packet_token_estimate"] = self.packet_token_estimate
            payload["packet_raw_candidate_count"] = self.packet_raw_candidate_count
            payload["packet_max_candidates_per_reference"] = self.packet_max_candidates_per_reference
            payload["packet_reference_line_text"] = self.packet_reference_line_text
        if self.contribution_note_kind:
            payload["contribution_note_kind"] = self.contribution_note_kind
        if self.resolved_title_kind:
            payload["resolved_title_kind"] = self.resolved_title_kind
        if self.artifact_topic_share is not None:
            payload["artifact_topic_share"] = self.artifact_topic_share
            payload["artifact_dominant_topic_id"] = self.artifact_dominant_topic_id
            payload["artifact_segment_count"] = self.artifact_segment_count
        return payload


def build_source_corpus_inventory(
    workspace_root: Path,
    *,
    topic_id: str,
    deck_dir: Path,
    source_inputs: list[Path],
    current_artifacts: list[Path],
    output_json: Path,
    receipt_path: Path,
) -> dict[str, object]:
    """Write a Topic source corpus inventory and receipt.

    Source inputs must stay inside ``workspace_root``. Current canary artifacts
    must stay inside ``deck_dir`` because they are rejected deck-local evidence,
    not general source evidence. Missing required inputs are recorded as
    blocker findings so later plans cannot silently skip absent source files.
    """

    workspace = workspace_root.expanduser().resolve()
    if not workspace.exists():
        raise ValueError(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise ValueError(f"workspace must be a directory: {workspace}")

    safe_topic_id = _require_safe_identifier(topic_id, label="topic_id")
    resolved_deck_dir = _resolve_inside_workspace(
        workspace,
        deck_dir,
        label="deck directory",
    )
    if not resolved_deck_dir.exists():
        raise ValueError(f"deck directory does not exist: {_relative_path(resolved_deck_dir, workspace)}")
    if not resolved_deck_dir.is_dir():
        raise ValueError(f"deck directory must be a directory: {_relative_path(resolved_deck_dir, workspace)}")

    resolved_output_json = _resolve_inside_workspace(
        workspace,
        output_json,
        label="output_json",
    )
    resolved_receipt_path = _resolve_inside_workspace(
        workspace,
        receipt_path,
        label="receipt_path",
    )

    generated_at = _isoformat_z()
    source_records: list[InventoryRecord] = []
    current_canary_artifacts: list[InventoryRecord] = []
    skipped_records: list[SkippedInventoryRecord] = []
    blockers: list[InventoryBlocker] = []

    for source_input in source_inputs:
        resolved = _resolve_inside_workspace(
            workspace,
            source_input,
            label="source input",
        )
        relative = _relative_path(resolved, workspace)
        kind = _source_kind(relative)
        if not resolved.exists():
            skipped_records.append(
                SkippedInventoryRecord(
                    kind=kind,
                    path=relative,
                    reason="required input is missing",
                )
            )
            blockers.append(
                InventoryBlocker(
                    code="required_input_missing",
                    message="Required source input is missing.",
                    path=relative,
                )
            )
            continue
        if not resolved.is_file():
            skipped_records.append(
                SkippedInventoryRecord(
                    kind=kind,
                    path=relative,
                    reason="required input is not a file",
                )
            )
            blockers.append(
                InventoryBlocker(
                    code="required_input_not_file",
                    message="Required source input is not a file.",
                    path=relative,
                )
            )
            continue
        if _is_rejected_non_production_source(resolved):
            skipped_records.append(
                SkippedInventoryRecord(
                    kind=kind,
                    path=relative,
                    reason="rejected non-production source evidence",
                )
            )
            continue
        source_records.append(
            _inventory_record(
                resolved,
                workspace=workspace,
                kind=kind,
                production_evidence=True,
                evidence_status="production_evidence",
            )
        )

    for current_artifact in current_artifacts:
        resolved = _resolve_inside_deck(
            workspace=workspace,
            deck_dir=resolved_deck_dir,
            candidate=current_artifact,
            label="current artifact",
        )
        relative = _relative_path(resolved, workspace)
        kind = _current_artifact_kind(resolved.name)
        if not resolved.exists():
            skipped_records.append(
                SkippedInventoryRecord(
                    kind=kind,
                    path=relative,
                    reason="current canary artifact is missing",
                )
            )
            blockers.append(
                InventoryBlocker(
                    code="current_artifact_missing",
                    message="Required current canary artifact is missing.",
                    path=relative,
                )
            )
            continue
        if not resolved.is_file():
            skipped_records.append(
                SkippedInventoryRecord(
                    kind=kind,
                    path=relative,
                    reason="current canary artifact is not a file",
                )
            )
            blockers.append(
                InventoryBlocker(
                    code="current_artifact_not_file",
                    message="Required current canary artifact is not a file.",
                    path=relative,
                )
            )
            continue
        current_canary_artifacts.append(
            _inventory_record(
                resolved,
                workspace=workspace,
                kind=kind,
                production_evidence=False,
                evidence_status="rejected_current_canary",
            )
        )

    source_records = sorted(source_records, key=lambda item: item.path)
    current_canary_artifacts = sorted(
        current_canary_artifacts,
        key=lambda item: item.path,
    )
    skipped_records = sorted(skipped_records, key=lambda item: (item.path, item.reason))
    blockers = sorted(blockers, key=lambda item: (item.path, item.code))
    counts = {
        "input_count": len(source_inputs),
        "source_record_count": len(source_records),
        "current_canary_artifact_count": len(current_canary_artifacts),
        "skipped_record_count": len(skipped_records),
        "blocker_count": len(blockers),
        "missing_required_input_count": sum(
            1 for blocker in blockers if blocker.code == "required_input_missing"
        ),
        "missing_current_artifact_count": sum(
            1 for blocker in blockers if blocker.code == "current_artifact_missing"
        ),
    }
    inventory_payload = {
        "schema_version": SOURCE_CORPUS_INVENTORY_SCHEMA_VERSION,
        "topic_id": safe_topic_id,
        "generated_at": generated_at,
        "deck_dir": _relative_path(resolved_deck_dir, workspace),
        "inputs": [record.to_dict() for record in source_records],
        "current_canary_artifacts": [
            record.to_dict() for record in current_canary_artifacts
        ],
        "source_records": [record.to_dict() for record in source_records],
        "skipped_records": [record.to_dict() for record in skipped_records],
        "counts": counts,
        "blockers": [blocker.to_dict() for blocker in blockers],
    }
    write_json_atomic(resolved_output_json, inventory_payload)

    output_paths = [
        _relative_path(resolved_output_json, workspace),
        _relative_path(resolved_receipt_path, workspace),
    ]
    receipt_payload = {
        "schema_version": SOURCE_CORPUS_INVENTORY_SCHEMA_VERSION,
        "receipt_type": "conference_pipeline.presentation.source_corpus_inventory",
        "generated_at": generated_at,
        "generated_by": "scripts/presentations/build_presentation_source_corpus_inventory.py",
        "topic_id": safe_topic_id,
        "deck_dir": _relative_path(resolved_deck_dir, workspace),
        "status": "blocked" if blockers else "succeeded",
        "inputs": [record.to_dict() for record in source_records],
        "current_canary_artifacts": [
            record.to_dict() for record in current_canary_artifacts
        ],
        "outputs": output_paths,
        "counts": counts,
        "skipped_records": [record.to_dict() for record in skipped_records],
        "blockers": [blocker.to_dict() for blocker in blockers],
        "blocker_count": len(blockers),
    }
    write_json_atomic(resolved_receipt_path, receipt_payload)
    return receipt_payload


def build_speaker_author_work_ledger(
    workspace_root: Path,
    *,
    inventory_json: Path,
    output_json: Path,
    receipt_path: Path,
) -> dict[str, object]:
    """Write a source eligibility ledger from a source corpus inventory."""

    workspace = workspace_root.expanduser().resolve()
    if not workspace.exists():
        raise ValueError(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise ValueError(f"workspace must be a directory: {workspace}")

    resolved_inventory_json = _resolve_inside_workspace(
        workspace,
        inventory_json,
        label="inventory_json",
    )
    resolved_output_json = _resolve_inside_workspace(
        workspace,
        output_json,
        label="output_json",
    )
    resolved_receipt_path = _resolve_inside_workspace(
        workspace,
        receipt_path,
        label="receipt_path",
    )
    inventory_payload = _load_json_object(
        resolved_inventory_json,
        workspace=workspace,
        label="source corpus inventory",
    )
    topic_id = _require_safe_identifier(str(inventory_payload.get("topic_id", "")), label="topic_id")
    inventory_sha256 = _sha256_file(resolved_inventory_json)
    source_records = _inventory_source_records(inventory_payload)
    contribution_notes = _contribution_notes_from_inventory(
        workspace=workspace,
        source_records=source_records,
    )

    drafts: dict[str, dict[str, Any]] = {}
    input_hashes: list[dict[str, str]] = [
        {
            "kind": "source_corpus_inventory",
            "path": _relative_path(resolved_inventory_json, workspace),
            "sha256": inventory_sha256,
        }
    ]
    blockers: list[InventoryBlocker] = []

    for record in source_records:
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            continue
        source_path = _resolve_inside_workspace(workspace, path_value, label="inventory source path")
        if not source_path.is_file():
            blockers.append(
                InventoryBlocker(
                    code="ledger_source_missing",
                    message="Inventory-listed source file is missing.",
                    path=path_value,
                )
            )
            continue
        input_hashes.append(
            {
                "kind": str(record.get("kind") or "source_input"),
                "path": path_value,
                "sha256": _sha256_file(source_path),
            }
        )
        payload = _try_load_json_object(source_path)
        if payload is None:
            continue
        if "topic_reference_ledgers" in path_value and isinstance(payload.get("entries"), list):
            for entry in payload.get("entries", []):
                if isinstance(entry, dict):
                    _merge_ledger_draft(
                        drafts,
                        _draft_from_reference_ledger_entry(
                            entry,
                            source_path=path_value,
                            contribution_notes=contribution_notes,
                        ),
                    )
        if path_value.endswith("__packets.json") or isinstance(payload.get("packets"), list):
            for packet in payload.get("packets", []):
                if isinstance(packet, dict):
                    packet_status = _packet_blocked_status(packet)
                    packet_work_item = packet.get("work_item")
                    packet_line_text = (
                        str(packet_work_item.get("line_text") or "")
                        if isinstance(packet_work_item, dict)
                        else ""
                    )
                    for candidate in packet.get("candidates", []):
                        if isinstance(candidate, dict):
                            _merge_ledger_draft(
                                drafts,
                                _draft_from_candidate_packet_entry(
                                    candidate,
                                    source_path=path_value,
                                    packet_status=packet_status,
                                    contribution_notes=contribution_notes,
                                    packet_token_estimate=packet.get("token_estimate"),
                                    packet_raw_candidate_count=packet.get("raw_candidate_count"),
                                    packet_max_candidates_per_reference=(
                                        packet.get("config", {}).get("max_candidates_per_reference")
                                        if isinstance(packet.get("config"), dict)
                                        else None
                                    ),
                                    packet_reference_line_text=packet_line_text,
                                ),
                            )

    full_read_units_added, full_read_paths = _drafts_from_topic_full_read_units(
        workspace, topic_id, inventory_payload, drafts
    )
    for full_read_path in full_read_paths:
        input_hashes.append(
            {
                "kind": "topic_full_read_units",
                "path": _relative_path(full_read_path, workspace),
                "sha256": _sha256_file(full_read_path),
            }
        )
    enriched_paper_titles = _enrich_paper_titles_from_openreview(workspace, drafts)
    enriched_authors = _enrich_paper_authors_from_openreview(workspace, drafts)
    transcript_enrichment = _enrich_speech_transcript_metadata_from_local_join(
        workspace, drafts
    )
    enriched_transcript_titles = _enrich_speech_transcript_titles_from_item_pages(workspace, drafts)
    off_topic_artifacts = _apply_transcript_artifact_topic_share(workspace, drafts, topic_id=topic_id)
    unverified_speakers_dropped = _drop_unverified_transcript_speakers(workspace, drafts)
    analysis_notes = _contribution_notes_from_inventory(
        workspace=workspace, source_records=source_records,
        work_titles={sid: str(d.get("resolved_title") or "") for sid, d in drafts.items()},
    )
    for sid, draft in drafts.items():
        if sid in analysis_notes and (not draft.get("contribution_note") or draft.get("contribution_note_source") in {"analysis_line", "full_read_summary"}):
            draft["contribution_note"] = analysis_notes[sid]
            draft["contribution_note_source"] = "analysis_line"
    enriched_contribution_notes = _enrich_paper_contribution_notes(workspace, drafts, topic_id=topic_id)
    generated_at = _isoformat_z()
    entries = _finalize_ledger_entries(drafts)
    counts = {
        "entry_count": len(entries),
        "visible_source_chip_eligible_count": sum(
            1 for entry in entries if entry.visible_source_chip_eligible
        ),
        "blocked_status_count": sum(
            1 for entry in entries if entry.status in BLOCKED_VISIBLE_SOURCE_STATUSES
        ),
        "missing_people_count": sum(
            1 for entry in entries if entry.status == "missing_people"
        ),
        "missing_contribution_count": sum(
            1 for entry in entries if entry.status == "missing_contribution"
        ),
        "input_hash_count": len(input_hashes),
        "blocker_count": len(blockers),
        "openreview_author_enriched_count": enriched_authors,
        "contribution_note_enriched_count": enriched_contribution_notes,
        "derived_abstract_contribution_note_count": sum(entry.contribution_note_kind == "derived_abstract" for entry in entries),
        # Key name deliberately unchanged so the persisted receipt shape stays
        # stable across the rewrite, even though the pass no longer reads item
        # pages.
        "item_page_title_enriched_count": transcript_enrichment["enriched"],
        # Counted off the FINALIZED entries rather than off the pass's own
        # tally, so this number is disk truth: it is the count of rows whose
        # provenance marker actually survives serialization, which is the only
        # form in which the pseudo-people guard can read it.
        "local_join_title_marked_count": sum(
            1
            for entry in entries
            if entry.resolved_title_kind == LOCAL_JOIN_RESOLVED_TITLE_KIND
        ),
        "transcript_extraction_note_count": transcript_enrichment["notes"],
        "openreview_title_enriched_count": enriched_paper_titles,
        "item_page_title_fallback_enriched_count": enriched_transcript_titles,
        "off_topic_artifact_count": off_topic_artifacts,
        "topic_full_read_cited_unit_count": full_read_units_added,
        "unverified_transcript_speaker_dropped_count": unverified_speakers_dropped,
    }
    ledger_payload = {
        "schema_version": SPEAKER_AUTHOR_WORK_LEDGER_SCHEMA_VERSION,
        "ledger_type": "conference_pipeline.presentation.speaker_author_work_ledger",
        "topic_id": topic_id,
        "generated_at": generated_at,
        "inventory_path": _relative_path(resolved_inventory_json, workspace),
        "inventory_sha256": inventory_sha256,
        "blocked_visible_source_statuses": sorted(BLOCKED_VISIBLE_SOURCE_STATUSES),
        "supporting_confidence_threshold": SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD,
        "entries": [entry.to_dict() for entry in entries],
        "counts": counts,
        "blockers": [blocker.to_dict() for blocker in blockers],
    }
    write_json_atomic(resolved_output_json, ledger_payload)

    receipt_payload = {
        "schema_version": SPEAKER_AUTHOR_WORK_LEDGER_SCHEMA_VERSION,
        "receipt_type": "conference_pipeline.presentation.speaker_author_work_ledger",
        "generated_at": generated_at,
        "generated_by": "scripts/presentations/build_speaker_author_work_ledger.py",
        "topic_id": topic_id,
        "status": "blocked" if blockers else "succeeded",
        "inventory_path": _relative_path(resolved_inventory_json, workspace),
        "inventory_sha256": inventory_sha256,
        "input_hashes": sorted(input_hashes, key=lambda item: (item["path"], item["kind"])),
        "outputs": [
            _relative_path(resolved_output_json, workspace),
            _relative_path(resolved_receipt_path, workspace),
        ],
        "counts": counts,
        "blockers": [blocker.to_dict() for blocker in blockers],
        "blocker_count": len(blockers),
    }
    write_json_atomic(resolved_receipt_path, receipt_payload)
    return receipt_payload


def build_source_repair_findings(
    workspace_root: Path,
    *,
    inventory_json: Path,
    ledger_json: Path,
    output_json: Path,
    receipt_path: Path,
    reviewed_repairs_json: Path | None = None,
) -> dict[str, object]:
    """Write explicit source repair findings from a speaker/author/work ledger."""

    workspace = workspace_root.expanduser().resolve()
    if not workspace.exists():
        raise ValueError(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise ValueError(f"workspace must be a directory: {workspace}")

    resolved_inventory_json = _resolve_inside_workspace(
        workspace,
        inventory_json,
        label="inventory_json",
    )
    resolved_ledger_json = _resolve_inside_workspace(
        workspace,
        ledger_json,
        label="ledger_json",
    )
    resolved_output_json = _resolve_inside_workspace(
        workspace,
        output_json,
        label="output_json",
    )
    resolved_receipt_path = _resolve_inside_workspace(
        workspace,
        receipt_path,
        label="receipt_path",
    )
    inventory_payload = _load_json_object(
        resolved_inventory_json,
        workspace=workspace,
        label="source corpus inventory",
    )
    ledger_payload = _load_json_object(
        resolved_ledger_json,
        workspace=workspace,
        label="speaker author work ledger",
    )
    topic_id = _require_safe_identifier(str(ledger_payload.get("topic_id", "")), label="topic_id")
    inventory_sha256 = _sha256_file(resolved_inventory_json)
    ledger_sha256 = _sha256_file(resolved_ledger_json)
    input_hashes = [
        {
            "kind": "source_corpus_inventory",
            "path": _relative_path(resolved_inventory_json, workspace),
            "sha256": inventory_sha256,
        },
        {
            "kind": "speaker_author_work_ledger",
            "path": _relative_path(resolved_ledger_json, workspace),
            "sha256": ledger_sha256,
        },
    ]
    reviewed_repairs: dict[str, dict[str, Any]] = {}
    if reviewed_repairs_json is not None:
        resolved_reviewed_repairs_json = _resolve_inside_workspace(
            workspace,
            reviewed_repairs_json,
            label="reviewed_repairs_json",
        )
        repairs_payload = _load_json_object(
            resolved_reviewed_repairs_json,
            workspace=workspace,
            label="reviewed source repairs",
        )
        input_hashes.append(
            {
                "kind": "reviewed_source_repairs",
                "path": _relative_path(resolved_reviewed_repairs_json, workspace),
                "sha256": _sha256_file(resolved_reviewed_repairs_json),
            }
        )
        reviewed_repairs = _reviewed_repair_map(repairs_payload)

    entries = ledger_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("speaker author work ledger entries must be a list")

    findings = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_id = str(entry.get("source_id") or "")
        repair = reviewed_repairs.get(source_id)
        finding_code = _source_repair_finding_code(entry)
        if finding_code or repair is not None:
            findings.append(
                _source_repair_finding(
                    entry,
                    finding_code=finding_code or str(repair.get("finding_code") or "reviewed_repair"),
                    repair=repair,
                )
            )

    generated_at = _isoformat_z()
    counts = {
        "entry_count": len([entry for entry in entries if isinstance(entry, dict)]),
        "finding_count": len(findings),
        "reviewed_count": sum(1 for finding in findings if finding["review_status"] == "reviewed"),
        "visible_use_unlocked_count": sum(
            1 for finding in findings if finding["visible_use_unlocked"] is True
        ),
        "needs_review_count": sum(1 for finding in findings if finding["review_status"] != "reviewed"),
        "input_hash_count": len(input_hashes),
        "inventory_blocker_count": _blocker_count(inventory_payload),
        "ledger_blocker_count": _blocker_count(ledger_payload),
    }
    findings_payload = {
        "schema_version": SOURCE_REPAIR_FINDINGS_SCHEMA_VERSION,
        "findings_type": "conference_pipeline.presentation.source_repair_findings",
        "topic_id": topic_id,
        "generated_at": generated_at,
        "inventory_path": _relative_path(resolved_inventory_json, workspace),
        "inventory_sha256": inventory_sha256,
        "ledger_path": _relative_path(resolved_ledger_json, workspace),
        "ledger_sha256": ledger_sha256,
        "primary_confidence_threshold": PRIMARY_SOURCE_CONFIDENCE_THRESHOLD,
        "supporting_confidence_threshold": SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD,
        "blocked_visible_source_statuses": sorted(BLOCKED_VISIBLE_SOURCE_STATUSES),
        "findings": findings,
        "counts": counts,
    }
    write_json_atomic(resolved_output_json, findings_payload)

    receipt_payload = {
        "schema_version": SOURCE_REPAIR_FINDINGS_SCHEMA_VERSION,
        "receipt_type": "conference_pipeline.presentation.source_repair_findings",
        "generated_at": generated_at,
        "generated_by": "scripts/presentations/repair_presentation_sources.py",
        "topic_id": topic_id,
        "status": "succeeded",
        "inventory_path": _relative_path(resolved_inventory_json, workspace),
        "inventory_sha256": inventory_sha256,
        "ledger_path": _relative_path(resolved_ledger_json, workspace),
        "ledger_sha256": ledger_sha256,
        "input_hashes": sorted(input_hashes, key=lambda item: (item["path"], item["kind"])),
        "outputs": [
            _relative_path(resolved_output_json, workspace),
            _relative_path(resolved_receipt_path, workspace),
        ],
        "counts": counts,
        "blocker_count": 0,
    }
    write_json_atomic(resolved_receipt_path, receipt_payload)
    return receipt_payload


_KEY_POINT_BLOCK_RE = re.compile(r"##\s*Key Points\s*(.+?)(?:\n##\s|\Z)", re.S)
_KEY_POINT_BULLET_RE = re.compile(r"^- (.+?)(?=\n- |\Z)", re.S | re.M)
_BOLD_LEAD_RE = re.compile(r"^\*\*(.+?)\*\*")


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _read_json_file(path: Path) -> dict[str, Any] | None:
    text = _read_text_file(path)
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _slugify_cluster(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "cluster"


# Finite verbs that typically follow a claim's subject. We cut the title at the
# first of these to keep a grammatically COMPLETE subject noun phrase as the
# heading (e.g. "Hybrid sparse/dense rewards are a concrete response..." ->
# "Hybrid sparse/dense rewards") rather than a mid-phrase fragment.
_TITLE_VERB_MARKERS = frozenset(
    {
        "is", "are", "was", "were", "be", "been", "being",
        "becomes", "become", "becoming", "remains", "remain",
        "can", "cannot", "could", "must", "should", "shall",
        "will", "would", "may", "might", "needs", "need",
        "makes", "make", "moves", "move", "shifts", "shift",
        "favored", "favors", "favor", "wins", "win", "beats", "beat",
        "shows", "show", "decides", "decide", "turns", "turn",
        "improves", "improve", "enables", "enable", "reduces", "reduce",
        "increases", "increase", "scales", "scale", "requires", "require",
        "learns", "learn", "outperforms", "outperform", "matters", "matter",
        "helps", "help", "drives", "drive", "predicts", "predict",
        "explains", "explain", "emerges", "emerge", "depends", "depend",
        "works", "work", "fails", "fail", "holds", "hold", "has", "have",
        "gets", "get", "lets", "let", "gives", "give",
        "changes", "change", "allows", "allow", "supports", "support",
        "yields", "yield", "produces", "produce", "causes", "cause",
        "brings", "bring", "treats", "treat", "frames", "frame",
        "raises", "raise", "lowers", "lower", "breaks", "break",
        "closes", "close", "opens", "open", "splits", "split",
        "becomes", "become", "remains", "stays", "stay", "keeps", "keep",
    }
)
_TITLE_CONNECTOR_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for",
        "with", "at", "is", "are", "that", "this", "by", "as", "from",
        "into", "over", "than", "but", "nor", "via", "per",
    }
)


def _short_cluster_title(title: str, limit: int = 48) -> str:
    compact = re.sub(r"\s+", " ", title).strip().rstrip(".")
    if len(compact) <= limit:
        return compact
    # Slide/figure headings render in a fixed band; keep titles concise so they are
    # not hard-truncated mid-sentence. Prefer a leading clause (before the first
    # comma/colon/semicolon/dash) when that already reads as a complete title.
    for sep in (",", ";", ":", " — ", " - "):
        head = compact.split(sep, 1)[0].strip()
        if 12 <= len(head) <= limit:
            return head
    # Otherwise take the SUBJECT noun phrase: the words before the first finite verb.
    words = compact.split()
    cut = 0
    for index, word in enumerate(words):
        if index >= 2 and word.lower().strip(".,;:'\"()") in _TITLE_VERB_MARKERS:
            cut = index
            break
    if cut:
        subject = " ".join(words[:cut]).strip(" -:,")
        # A complete subject phrase that slightly overruns the band still reads far
        # better than a mid-phrase cut, so allow modest headroom here.
        if 10 <= len(subject) <= limit + 12:
            return subject
    # Last resort: whole leading words up to the band, ending on a content word
    # (never a dangling connector) so the heading still reads as a complete phrase.
    kept: list[str] = []
    for word in words:
        if len(" ".join([*kept, word])) > limit:
            break
        kept.append(word)
    while kept and kept[-1].lower().strip(".,;:'\"()") in _TITLE_CONNECTOR_WORDS:
        kept.pop()
    return " ".join(kept) or compact[:limit].rstrip()


def _derive_topic_title(
    narrative: dict[str, Any] | None,
    analysis_text: str | None,
    topic_id: str,
) -> str:
    if isinstance(narrative, dict):
        title = str(narrative.get("title") or "").strip()
        if title:
            return title
    if analysis_text:
        for line in analysis_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                heading = stripped[2:].strip()
                if heading:
                    return heading
    humanized = re.sub(r"^\d+[_-]", "", topic_id).replace("_", " ").strip()
    return humanized or topic_id


def _parse_slide_brief_key_points(text: str) -> list[dict[str, Any]]:
    """Parse the ``## Key Points`` bullets of a slide brief into topic clusters.

    Each bullet's leading ``**bold**`` phrase becomes the cluster title, the full
    bullet (with source ids stripped) becomes the story contribution, and any cited
    ``paper_abstract:``/``speech_transcript:`` ids become the cluster's primary
    source-id bindings. Returns ``[]`` when no Key Points section is present.
    """

    block = _KEY_POINT_BLOCK_RE.search(text or "")
    if not block:
        return []
    points: list[dict[str, Any]] = []
    for raw in _KEY_POINT_BULLET_RE.findall(block.group(1)):
        bullet = raw.strip()
        if not bullet:
            continue
        bold = _BOLD_LEAD_RE.match(bullet)
        if bold:
            title = bold.group(1).strip().rstrip(".")
        else:
            title = re.split(r"[.:]", bullet, 1)[0].strip()
        if not title:
            continue
        cited = _unique_strings(_SOURCE_REFERENCE_RE.findall(bullet))
        points.append(
            {
                "title": title,
                "story": _clean_contribution_note(bullet),
                "cited_ids": cited,
            }
        )
    return points


def _cluster_keywords(title: str, story: str, cited_titles: list[str]) -> tuple[str, ...]:
    text = " ".join([title, story, " ".join(cited_titles)]).lower()
    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text)
        if token not in _KEYWORD_STOPWORDS
    ]
    keywords: list[str] = []
    for token in tokens:
        if token not in keywords:
            keywords.append(token)
    title_tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", title.lower())
        if token not in _KEYWORD_STOPWORDS
    ]
    for first, second in zip(title_tokens, title_tokens[1:]):
        bigram = f"{first} {second}"
        if bigram not in keywords:
            keywords.append(bigram)
    return tuple(keywords[:16])


def _generic_cluster_map(
    topic_title: str, *, conference_label: str = ""
) -> tuple[dict[str, object], ...]:
    base_keywords = tuple(
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", topic_title.lower())
        if token not in _KEYWORD_STOPWORDS
    ) or ("research",)
    label = str(conference_label or "").strip()
    corpus_evidence = f"the {label} corpus evidence" if label else "the conference corpus evidence"
    return (
        {
            "cluster_id": "topic-overview",
            "title": _short_cluster_title(topic_title),
            "keywords": base_keywords,
            "story_contribution": (
                f"{topic_title}: {corpus_evidence} grounded in named "
                "speakers, authors, and works."
            ),
            "cited_source_ids": (),
        },
    )


def derive_research_cluster_map(
    workspace: Path,
    *,
    topic_id: str,
    source_pool: list[dict[str, Any]],
) -> tuple[tuple[dict[str, object], ...], str]:
    """Derive a per-topic research cluster map from the topic's slide brief and
    analysis (deterministic, no LLM). Falls back to a single topic-overview cluster
    when no Key Points / narrative themes are available."""

    title_by_id = {
        str(source.get("source_id") or ""): str(source.get("resolved_title") or "")
        for source in source_pool
    }
    narrative = _read_json_file(
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"
    )
    analysis_text = _read_text_file(workspace / "analysis" / f"{topic_id}.md")
    brief_text = _read_text_file(workspace / "analysis" / "slide_briefs" / f"{topic_id}.md")
    topic_title = _derive_topic_title(narrative, analysis_text, topic_id)

    clusters: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for point in _parse_slide_brief_key_points(brief_text or ""):
        cluster_id = _slugify_cluster(str(point["title"]))
        while cluster_id in seen_ids:
            cluster_id = f"{cluster_id}-{len(seen_ids)}"
        seen_ids.add(cluster_id)
        cited = tuple(str(c) for c in point["cited_ids"] if str(c) in title_by_id)
        cited_titles = [title_by_id[c] for c in cited]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "title": _short_cluster_title(str(point["title"])),
                "keywords": _cluster_keywords(
                    str(point["title"]), str(point["story"]), cited_titles
                ),
                "story_contribution": str(point["story"]),
                "cited_source_ids": cited,
            }
        )
    if not clusters:
        conference_label = conference_name(workspace, default="conference")
        return _generic_cluster_map(topic_title, conference_label=conference_label), topic_title
    return tuple(clusters), topic_title


def build_research_cluster_plan(
    workspace_root: Path,
    *,
    inventory_json: Path,
    ledger_json: Path,
    repair_findings_json: Path,
    output_json: Path,
    markdown_output: Path,
    receipt_path: Path,
) -> dict[str, object]:
    """Write a source-grounded research_cluster_plan JSON, Markdown projection, and receipt."""

    workspace = workspace_root.expanduser().resolve()
    if not workspace.exists():
        raise ValueError(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise ValueError(f"workspace must be a directory: {workspace}")

    resolved_inventory_json = _resolve_inside_workspace(
        workspace,
        inventory_json,
        label="inventory_json",
    )
    resolved_ledger_json = _resolve_inside_workspace(
        workspace,
        ledger_json,
        label="ledger_json",
    )
    resolved_repair_findings_json = _resolve_inside_workspace(
        workspace,
        repair_findings_json,
        label="repair_findings_json",
    )
    resolved_output_json = _resolve_inside_workspace(
        workspace,
        output_json,
        label="output_json",
    )
    resolved_markdown_output = _resolve_inside_workspace(
        workspace,
        markdown_output,
        label="markdown_output",
    )
    resolved_receipt_path = _resolve_inside_workspace(
        workspace,
        receipt_path,
        label="receipt_path",
    )

    inventory_payload = _load_json_object(
        resolved_inventory_json,
        workspace=workspace,
        label="source corpus inventory",
    )
    ledger_payload = _load_json_object(
        resolved_ledger_json,
        workspace=workspace,
        label="speaker author work ledger",
    )
    repair_payload = _load_json_object(
        resolved_repair_findings_json,
        workspace=workspace,
        label="source repair findings",
    )
    topic_id = _require_safe_identifier(
        str(ledger_payload.get("topic_id") or inventory_payload.get("topic_id") or ""),
        label="topic_id",
    )
    input_hashes = {
        "source_corpus_inventory.json": _sha256_file(resolved_inventory_json),
        "speaker_author_work_ledger.json": _sha256_file(resolved_ledger_json),
        "source_repair_findings.json": _sha256_file(resolved_repair_findings_json),
    }
    source_pool = _research_cluster_source_pool(
        ledger_payload=ledger_payload,
        repair_payload=repair_payload,
    )
    if not source_pool:
        raise ValueError("research_cluster_plan requires at least one eligible source candidate")

    cluster_map, topic_title = derive_research_cluster_map(
        workspace,
        topic_id=topic_id,
        source_pool=source_pool,
    )
    # Restrict the selectable pool to topic-relevant sources: every paper the slide
    # brief actually cites, plus non-paper sources (talks/transcripts). This stops
    # an off-topic, high-rank paper that merely keyword-matches a cluster title from
    # being bound (e.g. a theorem-proving paper landing on a reward-design slide).
    cited_union = frozenset(
        str(source_id)
        for definition in cluster_map
        for source_id in definition.get("cited_source_ids", ())
    )
    selectable_pool = [
        source
        for source in source_pool
        if str(source.get("source_id") or "") in cited_union
        or not str(source.get("source_id") or "").startswith("paper_abstract:")
    ]
    if not selectable_pool:
        selectable_pool = source_pool
    clusters = [
        _build_research_cluster(
            definition,
            source_pool=selectable_pool,
            cited_source_ids=frozenset(
                str(source_id) for source_id in definition.get("cited_source_ids", ())
            ),
        )
        for definition in cluster_map
    ]
    validation_payload = {
        "schema_version": RESEARCH_CLUSTER_PLAN_SCHEMA_VERSION,
        "plan_type": "conference_pipeline.presentation.research_cluster_plan",
        "topic_id": topic_id,
        "topic_title": topic_title,
        "generated_at": _isoformat_z(),
        "production_evidence": True,
        "default_cluster_map": [
            {
                "cluster_id": str(definition["cluster_id"]),
                "title": str(definition["title"]),
            }
            for definition in cluster_map
        ],
        "primary_confidence_threshold": PRIMARY_SOURCE_CONFIDENCE_THRESHOLD,
        "supporting_confidence_threshold": SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD,
        "source_family_balance_threshold": SOURCE_FAMILY_BALANCE_MAX_FRACTION,
        "blocked_visible_source_statuses": sorted(BLOCKED_VISIBLE_SOURCE_STATUSES),
        "input_hashes": input_hashes,
        "source_pool_counts": _research_source_pool_counts(source_pool),
        "clusters": clusters,
    }
    validation = validate_research_cluster_plan_payload(validation_payload)
    if validation["status"] != "pass":
        raise ValueError(json.dumps(validation, indent=2, sort_keys=True))
    validation_payload["validation"] = validation
    validation_payload["counts"] = {
        "cluster_count": len(clusters),
        "primary_source_count": sum(len(cluster["primary_sources"]) for cluster in clusters),
        "supporting_source_count": sum(len(cluster["supporting_sources"]) for cluster in clusters),
        "cluster_blocker_count": sum(len(cluster["blockers"]) for cluster in clusters),
    }

    write_json_atomic(resolved_output_json, validation_payload)
    _write_text_atomic(
        resolved_markdown_output,
        _research_cluster_plan_markdown(validation_payload),
    )

    receipt_payload = {
        "schema_version": RESEARCH_CLUSTER_PLAN_SCHEMA_VERSION,
        "receipt_type": "conference_pipeline.presentation.research_cluster_plan",
        "generated_at": validation_payload["generated_at"],
        "generated_by": "scripts/presentations/build_research_cluster_plan.py",
        "topic_id": topic_id,
        "status": "succeeded",
        "production_evidence": True,
        "input_hashes": input_hashes,
        "outputs": [
            _relative_path(resolved_output_json, workspace),
            _relative_path(resolved_markdown_output, workspace),
            _relative_path(resolved_receipt_path, workspace),
        ],
        "counts": validation_payload["counts"],
        "validation": validation,
        "blocker_count": validation_payload["counts"]["cluster_blocker_count"],
    }
    write_json_atomic(resolved_receipt_path, receipt_payload)
    return receipt_payload


def planned_research_cluster_plan_writes(
    workspace_root: Path,
    *,
    inventory_json: Path,
    ledger_json: Path,
    repair_findings_json: Path,
    output_json: Path,
    markdown_output: Path,
    receipt_path: Path,
) -> list[Path]:
    """Return validated research cluster output paths without writing files."""

    workspace = workspace_root.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace must be an existing directory: {workspace}")
    for label, candidate in (
        ("inventory_json", inventory_json),
        ("ledger_json", ledger_json),
        ("repair_findings_json", repair_findings_json),
    ):
        resolved = _resolve_inside_workspace(workspace, candidate, label=label)
        if not resolved.is_file():
            raise ValueError(f"{label} must be an existing file: {resolved}")
    return [
        _resolve_inside_workspace(workspace, output_json, label="output_json"),
        _resolve_inside_workspace(workspace, markdown_output, label="markdown_output"),
        _resolve_inside_workspace(workspace, receipt_path, label="receipt_path"),
    ]


def research_cluster_plan_receipt_summary(receipt: dict[str, object]) -> dict[str, object]:
    counts = receipt.get("counts")
    return {
        "schema_version": receipt.get("schema_version"),
        "topic_id": receipt.get("topic_id"),
        "generated_at": receipt.get("generated_at"),
        "status": receipt.get("status"),
        "input_hashes": receipt.get("input_hashes", {}),
        "counts": counts if isinstance(counts, dict) else {},
        "outputs": receipt.get("outputs", []),
        "blocker_count": receipt.get("blocker_count", 0),
    }


def validate_research_cluster_plan_payload(payload: Any) -> dict[str, object]:
    """Validate research cluster source eligibility, balance, and required fields."""

    errors: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        return {
            "status": "fail",
            "errors": [
                {
                    "code": "research_cluster_plan_root_not_object",
                    "message": "research_cluster_plan root must be an object.",
                    "path": "$",
                }
            ],
        }
    if payload.get("production_evidence") is not True:
        errors.append(
            _research_cluster_finding(
                "research_cluster_plan_not_production_evidence",
                "research_cluster_plan JSON must be marked as production evidence.",
                "$.production_evidence",
            )
        )
    clusters = payload.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        errors.append(
            _research_cluster_finding(
                "research_cluster_plan_clusters_missing",
                "research_cluster_plan clusters must be a non-empty list.",
                "$.clusters",
            )
        )
        return {"status": "fail", "errors": errors}

    required_cluster_fields = {
        "cluster_id",
        "title",
        "story_contribution",
        "primary_sources",
        "supporting_sources",
        "retained_rationale",
        "merged_rationale",
        "dropped_rationale",
        "source_family_balance",
        "blockers",
    }
    for cluster_index, cluster in enumerate(clusters):
        cluster_path = f"$.clusters[{cluster_index}]"
        if not isinstance(cluster, dict):
            errors.append(
                _research_cluster_finding(
                    "research_cluster_not_object",
                    "research cluster must be an object.",
                    cluster_path,
                )
            )
            continue
        missing_fields = sorted(required_cluster_fields - set(cluster))
        if missing_fields:
            errors.append(
                _research_cluster_finding(
                    "research_cluster_missing_required_fields",
                    f"research cluster missing required fields: {', '.join(missing_fields)}",
                    cluster_path,
                )
            )
        for field in (
            "cluster_id",
            "title",
            "story_contribution",
            "retained_rationale",
            "merged_rationale",
            "dropped_rationale",
        ):
            if not _non_empty_string(cluster.get(field)):
                errors.append(
                    _research_cluster_finding(
                        f"research_cluster_{field}_missing",
                        f"{field} must be a non-empty string.",
                        f"{cluster_path}.{field}",
                    )
                )
        primary_sources = cluster.get("primary_sources")
        if not isinstance(primary_sources, list):
            errors.append(
                _research_cluster_finding(
                    "primary_sources_not_list",
                    "primary_sources must be a list.",
                    f"{cluster_path}.primary_sources",
                )
            )
            primary_sources = []
        if not 2 <= len(primary_sources) <= 4:
            errors.append(
                _research_cluster_finding(
                    "primary_source_count_out_of_range",
                    "Each research cluster must have 2-4 primary sources.",
                    f"{cluster_path}.primary_sources",
                )
            )
        supporting_sources = cluster.get("supporting_sources")
        if not isinstance(supporting_sources, list):
            errors.append(
                _research_cluster_finding(
                    "supporting_sources_not_list",
                    "supporting_sources must be a list.",
                    f"{cluster_path}.supporting_sources",
                )
            )
            supporting_sources = []
        for source_index, source in enumerate(primary_sources):
            errors.extend(
                _validate_research_cluster_source(
                    source,
                    min_confidence=PRIMARY_SOURCE_CONFIDENCE_THRESHOLD,
                    path=f"{cluster_path}.primary_sources[{source_index}]",
                    role="primary",
                )
            )
        for source_index, source in enumerate(supporting_sources):
            errors.extend(
                _validate_research_cluster_source(
                    source,
                    min_confidence=SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD,
                    path=f"{cluster_path}.supporting_sources[{source_index}]",
                    role="supporting",
                )
            )
        balance = cluster.get("source_family_balance")
        if not isinstance(balance, dict):
            errors.append(
                _research_cluster_finding(
                    "source_family_balance_not_object",
                    "source_family_balance must be an object.",
                    f"{cluster_path}.source_family_balance",
                )
            )
        else:
            alternatives_exist = balance.get("eligible_alternative_families_exist") is True
            max_fraction = balance.get("max_family_fraction")
            if (
                alternatives_exist
                and isinstance(max_fraction, (int, float))
                and not isinstance(max_fraction, bool)
                and float(max_fraction) > SOURCE_FAMILY_BALANCE_MAX_FRACTION
            ) or balance.get("passes_70_percent_rule") is False:
                errors.append(
                    _research_cluster_finding(
                        "source_family_overconcentrated",
                        "Visible cluster exceeds the 70 percent source family balance rule.",
                        f"{cluster_path}.source_family_balance",
                    )
                )
    return {"status": "fail" if errors else "pass", "errors": errors}


def planned_source_repair_writes(
    workspace_root: Path,
    *,
    inventory_json: Path,
    ledger_json: Path,
    output_json: Path,
    receipt_path: Path,
    reviewed_repairs_json: Path | None = None,
) -> list[Path]:
    """Return validated source repair output paths without writing files."""

    workspace = workspace_root.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace must be an existing directory: {workspace}")
    for label, candidate in (
        ("inventory_json", inventory_json),
        ("ledger_json", ledger_json),
        ("reviewed_repairs_json", reviewed_repairs_json),
    ):
        if candidate is None:
            continue
        resolved = _resolve_inside_workspace(workspace, candidate, label=label)
        if not resolved.is_file():
            raise ValueError(f"{label} must be an existing file: {resolved}")
    return [
        _resolve_inside_workspace(workspace, output_json, label="output_json"),
        _resolve_inside_workspace(workspace, receipt_path, label="receipt_path"),
    ]


def source_repair_receipt_summary(receipt: dict[str, object]) -> dict[str, object]:
    counts = receipt.get("counts")
    return {
        "schema_version": receipt.get("schema_version"),
        "topic_id": receipt.get("topic_id"),
        "generated_at": receipt.get("generated_at"),
        "status": receipt.get("status"),
        "inventory_sha256": receipt.get("inventory_sha256"),
        "ledger_sha256": receipt.get("ledger_sha256"),
        "counts": counts if isinstance(counts, dict) else {},
        "outputs": receipt.get("outputs", []),
        "blocker_count": receipt.get("blocker_count", 0),
    }


def planned_speaker_author_work_ledger_writes(
    workspace_root: Path,
    *,
    inventory_json: Path,
    output_json: Path,
    receipt_path: Path,
) -> list[Path]:
    """Return validated ledger output paths without writing files."""

    workspace = workspace_root.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace must be an existing directory: {workspace}")
    resolved_inventory_json = _resolve_inside_workspace(
        workspace,
        inventory_json,
        label="inventory_json",
    )
    if not resolved_inventory_json.is_file():
        raise ValueError(f"inventory_json must be an existing file: {resolved_inventory_json}")
    return [
        _resolve_inside_workspace(workspace, output_json, label="output_json"),
        _resolve_inside_workspace(workspace, receipt_path, label="receipt_path"),
    ]


def speaker_author_work_ledger_receipt_summary(receipt: dict[str, object]) -> dict[str, object]:
    counts = receipt.get("counts")
    return {
        "schema_version": receipt.get("schema_version"),
        "topic_id": receipt.get("topic_id"),
        "generated_at": receipt.get("generated_at"),
        "status": receipt.get("status"),
        "inventory_sha256": receipt.get("inventory_sha256"),
        "counts": counts if isinstance(counts, dict) else {},
        "outputs": receipt.get("outputs", []),
        "blocker_count": receipt.get("blocker_count", 0),
    }


def visible_source_chip_eligible(
    entry: dict[str, Any],
    *,
    min_confidence: float = SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD,
) -> bool:
    """Return whether a ledger entry may be shown as a visible source chip."""

    status = str(entry.get("status") or "").strip()
    if status in BLOCKED_VISIBLE_SOURCE_STATUSES:
        return False
    confidence = entry.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return False
    if float(confidence) < min_confidence:
        return False
    if not _non_empty_string(entry.get("resolved_title")):
        return False
    if not _non_empty_string(entry.get("contribution_note")):
        return False
    authors = entry.get("authors")
    speakers = entry.get("speakers")
    has_people = (isinstance(authors, list) and bool(authors)) or (
        isinstance(speakers, list) and bool(speakers)
    )
    if has_people:
        return True
    # A source with a specific named work (a real paper/talk title, not a generic
    # placeholder) is citable as a work-source chip even when its authors were not
    # resolved (status ``missing_people``). Without this, a topic whose paper
    # authors were not extracted surfaces only its talks, leaving the deck to name
    # papers it cannot show as bound sources (attribution drift). A "work" is a
    # valid named entity in its own right.
    works = entry.get("works")
    if isinstance(works, list) and any(_non_empty_string(work) for work in works):
        return True
    title = str(entry.get("resolved_title") or "").strip()
    return bool(title) and not _is_generic_source_title(title)


def repair_visible_use_unlocked(repair: dict[str, Any]) -> bool:
    """Return whether reviewed repair evidence unlocks visible source use."""

    if str(repair.get("review_status") or "").strip() != "reviewed":
        return False
    visible_use = str(repair.get("visible_use") or "primary").strip()
    threshold = (
        SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD
        if visible_use == "supporting"
        else PRIMARY_SOURCE_CONFIDENCE_THRESHOLD
    )
    repaired_value = repair.get("repaired_value")
    if not isinstance(repaired_value, dict):
        return False
    status = str(repaired_value.get("status") or "").strip()
    title = str(repaired_value.get("resolved_title") or "").strip()
    if status in BLOCKED_VISIBLE_SOURCE_STATUSES:
        return False
    if _is_generic_source_title(title) or title == "Structured presentation source":
        return False
    entry = {
        "resolved_title": title,
        "authors": _string_list(repaired_value.get("authors")),
        "speakers": _string_list(repaired_value.get("speakers")),
        "contribution_note": str(repaired_value.get("contribution_note") or ""),
        "confidence": repair.get("confidence"),
        "status": status,
    }
    return visible_source_chip_eligible(entry, min_confidence=threshold)


def planned_source_corpus_inventory_writes(
    workspace_root: Path,
    *,
    deck_dir: Path,
    output_json: Path,
    receipt_path: Path,
) -> list[Path]:
    """Return validated output paths without writing files."""

    workspace = workspace_root.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace must be an existing directory: {workspace}")
    resolved_deck_dir = _resolve_inside_workspace(
        workspace,
        deck_dir,
        label="deck directory",
    )
    resolved_output_json = _resolve_inside_workspace(
        workspace,
        output_json,
        label="output_json",
    )
    resolved_receipt_path = _resolve_inside_workspace(
        workspace,
        receipt_path,
        label="receipt_path",
    )
    if not resolved_deck_dir.is_dir():
        raise ValueError(f"deck directory must be an existing directory: {resolved_deck_dir}")
    return [resolved_output_json, resolved_receipt_path]


def inventory_receipt_summary(receipt: dict[str, object]) -> dict[str, object]:
    counts = receipt.get("counts")
    return {
        "schema_version": receipt.get("schema_version"),
        "topic_id": receipt.get("topic_id"),
        "generated_at": receipt.get("generated_at"),
        "status": receipt.get("status"),
        "counts": counts if isinstance(counts, dict) else {},
        "outputs": receipt.get("outputs", []),
        "blocker_count": receipt.get("blocker_count", 0),
    }


def _research_cluster_source_pool(
    *,
    ledger_payload: dict[str, Any],
    repair_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = ledger_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("speaker author work ledger entries must be a list")
    source_by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source = _research_cluster_source_from_ledger_entry(entry)
        if source is not None:
            _merge_research_cluster_source(source_by_id, source)

    findings = repair_payload.get("findings")
    if not isinstance(findings, list):
        raise ValueError("source repair findings must contain a findings list")
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        source = _research_cluster_source_from_repair_finding(finding)
        if source is not None:
            _merge_research_cluster_source(source_by_id, source)

    return sorted(source_by_id.values(), key=_research_cluster_source_sort_key)


def _research_cluster_source_from_ledger_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    source_id = _first_string(entry.get("source_id"))
    if not source_id:
        return None
    return {
        "source_id": source_id,
        "resolved_title": str(entry.get("resolved_title") or ""),
        "authors": _string_list(entry.get("authors")),
        "speakers": _string_list(entry.get("speakers")),
        "works": _string_list(entry.get("works")),
        "source_family": str(entry.get("source_family") or "unknown"),
        "confidence": _coerce_confidence(entry.get("confidence"), default=0.0),
        "status": str(entry.get("status") or "unresolved"),
        "visible_source_chip_eligible": bool(entry.get("visible_source_chip_eligible")),
        "contribution_note": str(entry.get("contribution_note") or ""),
        "rank": int(entry.get("rank") or 999999),
        "evidence_paths": _string_list(entry.get("evidence_paths")),
        "eligibility_basis": "speaker_author_work_ledger",
    }


def _research_cluster_source_from_repair_finding(
    finding: dict[str, Any],
) -> dict[str, Any] | None:
    if finding.get("visible_use_unlocked") is not True:
        return None
    source_id = _first_string(finding.get("source_id"))
    repaired_value = finding.get("repaired_value")
    original_value = finding.get("original_value")
    if not source_id or not isinstance(repaired_value, dict):
        return None
    if not isinstance(original_value, dict):
        original_value = {}
    status = str(repaired_value.get("status") or "resolved")
    title = str(repaired_value.get("resolved_title") or original_value.get("resolved_title") or "")
    authors = _string_list(repaired_value.get("authors") or original_value.get("authors"))
    speakers = _string_list(repaired_value.get("speakers") or original_value.get("speakers"))
    contribution_note = str(
        repaired_value.get("contribution_note")
        or original_value.get("contribution_note")
        or finding.get("reviewer_notes")
        or ""
    )
    source = {
        "source_id": source_id,
        "resolved_title": title,
        "authors": authors,
        "speakers": speakers,
        "works": _unique_strings(
            _string_list(original_value.get("works")) + _string_list(title)
        ),
        "source_family": str(
            repaired_value.get("source_family")
            or original_value.get("source_family")
            or "unknown"
        ),
        "confidence": _coerce_confidence(finding.get("confidence"), default=0.0),
        "status": status,
        "visible_source_chip_eligible": True,
        "contribution_note": contribution_note,
        "rank": 0,
        "evidence_paths": _unique_strings(
            _string_list(finding.get("source_path")) + _string_list(source_id)
        ),
        "eligibility_basis": "reviewed_source_repair",
    }
    return source if visible_source_chip_eligible(source) else None


def _merge_research_cluster_source(
    source_by_id: dict[str, dict[str, Any]],
    source: dict[str, Any],
) -> None:
    existing = source_by_id.get(str(source["source_id"]))
    if existing is None or _research_cluster_source_quality(source) > _research_cluster_source_quality(existing):
        source_by_id[str(source["source_id"])] = source


def _research_cluster_source_quality(source: dict[str, Any]) -> tuple[float, int, float, int]:
    is_visible = 1 if visible_source_chip_eligible(source) else 0
    is_repair = 1 if source.get("eligibility_basis") == "reviewed_source_repair" else 0
    confidence = float(source.get("confidence") or 0.0)
    rank = int(source.get("rank") or 999999)
    return (float(is_visible), is_repair, confidence, -rank)


def _research_cluster_source_sort_key(source: dict[str, Any]) -> tuple[float, str, int, str]:
    return (
        -float(source.get("confidence") or 0.0),
        str(source.get("source_family") or ""),
        int(source.get("rank") or 999999),
        str(source.get("source_id") or ""),
    )


def _research_source_pool_counts(source_pool: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "source_count": len(source_pool),
        "eligible_primary_count": sum(
            1
            for source in source_pool
            if _research_cluster_source_eligible(
                source,
                min_confidence=PRIMARY_SOURCE_CONFIDENCE_THRESHOLD,
            )
        ),
        "eligible_supporting_count": sum(
            1
            for source in source_pool
            if _research_cluster_source_eligible(
                source,
                min_confidence=SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD,
            )
        ),
        "reviewed_repair_source_count": sum(
            1 for source in source_pool if source.get("eligibility_basis") == "reviewed_source_repair"
        ),
    }


def _build_research_cluster(
    definition: dict[str, object],
    *,
    source_pool: list[dict[str, Any]],
    cited_source_ids: frozenset[str] = frozenset(),
) -> dict[str, object]:
    keywords = [
        str(keyword).lower()
        for keyword in definition.get("keywords", ())
        if isinstance(keyword, str) and keyword.strip()
    ]
    title = str(definition["title"])
    cluster_pool = [
        source
        for source in source_pool
        if str(source.get("source_id") or "") in cited_source_ids
        or _research_source_matches_keywords(source, keywords)
    ]
    primary_sources = _select_research_cluster_sources(
        cluster_pool,
        fallback_pool=source_pool,
        min_confidence=PRIMARY_SOURCE_CONFIDENCE_THRESHOLD,
        min_count=2,
        max_count=4,
        excluded_source_ids=frozenset(),
        prefer_ids=cited_source_ids,
    )
    primary_ids = {str(source["source_id"]) for source in primary_sources}
    supporting_sources = _select_research_cluster_sources(
        cluster_pool,
        fallback_pool=source_pool,
        min_confidence=SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD,
        min_count=1,
        max_count=3,
        excluded_source_ids=primary_ids,
        prefer_ids=cited_source_ids,
    )
    visible_sources = primary_sources + supporting_sources
    balance_pool = cluster_pool if cluster_pool else source_pool
    balance = _research_source_family_balance(visible_sources, eligible_pool=balance_pool)
    dropped_count = max(
        0,
        len(
            [
                source
                for source in cluster_pool
                if str(source.get("source_id") or "") not in primary_ids
                and str(source.get("source_id") or "")
                not in {str(item["source_id"]) for item in supporting_sources}
            ]
        ),
    )
    dropped_rationale = (
        f"Dropped {dropped_count} lower-ranked or duplicate {title} candidates after "
        "confidence, blocked-status, people metadata, and contribution-note checks."
        if cluster_pool
        else (
            "No cluster-specific eligible alternatives were available in the ledger; "
            "the plan reused the strongest eligible reviewed source pool and keeps "
            "the cluster explicit for blueprint review."
        )
    )
    return {
        "cluster_id": str(definition["cluster_id"]),
        "title": title,
        "story_contribution": str(definition["story_contribution"]),
        "primary_sources": [_research_cluster_source_payload(source) for source in primary_sources],
        "supporting_sources": [
            _research_cluster_source_payload(source) for source in supporting_sources
        ],
        "retained_rationale": (
            f"Retained {len(primary_sources)} primary and {len(supporting_sources)} "
            f"supporting sources for {title} with primary confidence >= "
            f"{PRIMARY_SOURCE_CONFIDENCE_THRESHOLD:.2f} and supporting confidence >= "
            f"{SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD:.2f}."
        ),
        "merged_rationale": (
            "Merged duplicate source ids and preferred reviewed repairs, source-family "
            "diversity, higher confidence, and lower ledger rank."
        ),
        "dropped_rationale": dropped_rationale,
        "source_family_balance": balance,
        "blockers": [],
    }


def _research_source_matches_keywords(source: dict[str, Any], keywords: list[str]) -> bool:
    if not keywords:
        return False
    searchable = " ".join(
        [
            str(source.get("resolved_title") or ""),
            str(source.get("contribution_note") or ""),
            " ".join(_string_list(source.get("works"))),
        ]
    ).lower()
    return any(keyword in searchable for keyword in keywords)


def _select_research_cluster_sources(
    candidates: list[dict[str, Any]],
    *,
    fallback_pool: list[dict[str, Any]],
    min_confidence: float,
    min_count: int,
    max_count: int,
    excluded_source_ids: frozenset[str],
    prefer_ids: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    combined = _unique_research_sources(
        [
            source
            for source in candidates + fallback_pool
            if str(source.get("source_id") or "") not in excluded_source_ids
            and _research_cluster_source_eligible(source, min_confidence=min_confidence)
        ]
    )

    def _ranked(source: dict[str, Any]) -> tuple[Any, ...]:
        # Preferred (cluster-cited) sources sort ahead of generic keyword/global
        # matches so each cluster keeps its own distinct evidence.
        preferred = 0 if str(source.get("source_id") or "") in prefer_ids else 1
        return (preferred, _research_cluster_source_sort_key(source))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_families: set[str] = set()
    while len(selected) < min(max_count, len(combined)):
        remaining = [
            source
            for source in combined
            if str(source.get("source_id") or "") not in selected_ids
        ]
        if not remaining:
            break
        diverse = [
            source
            for source in remaining
            if str(source.get("source_family") or "") not in selected_families
        ]
        pool = diverse or remaining
        next_source = sorted(pool, key=_ranked)[0]
        selected.append(next_source)
        selected_ids.add(str(next_source["source_id"]))
        selected_families.add(str(next_source.get("source_family") or "unknown"))
        if len(selected) >= max_count:
            break
        remaining_after_select = [
            source
            for source in combined
            if str(source.get("source_id") or "") not in selected_ids
        ]
        can_add_new_family = any(
            str(source.get("source_family") or "unknown") not in selected_families
            for source in remaining_after_select
        )
        if len(selected) >= min_count and (
            len(selected_families) > 1 or not can_add_new_family
        ):
            break
    if len(selected) < min_count:
        for source in combined:
            source_id = str(source.get("source_id") or "")
            if source_id in selected_ids:
                continue
            selected.append(source)
            selected_ids.add(source_id)
            if len(selected) >= min_count:
                break
    return selected[:max_count]


def _unique_research_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_id = str(source.get("source_id") or "")
        if not source_id:
            continue
        _merge_research_cluster_source(by_id, source)
    return sorted(by_id.values(), key=_research_cluster_source_sort_key)


def _research_cluster_source_eligible(
    source: dict[str, Any],
    *,
    min_confidence: float,
) -> bool:
    return visible_source_chip_eligible(source, min_confidence=min_confidence)


def _research_cluster_source_payload(source: dict[str, Any]) -> dict[str, object]:
    return {
        "source_id": str(source.get("source_id") or ""),
        "resolved_title": str(source.get("resolved_title") or ""),
        "authors": _string_list(source.get("authors")),
        "speakers": _string_list(source.get("speakers")),
        "works": _string_list(source.get("works")),
        "source_family": str(source.get("source_family") or "unknown"),
        "confidence": _coerce_confidence(source.get("confidence"), default=0.0),
        "status": str(source.get("status") or ""),
        "contribution_note": str(source.get("contribution_note") or ""),
        "rank": int(source.get("rank") or 999999),
        "evidence_paths": _string_list(source.get("evidence_paths")),
        "eligibility_basis": str(source.get("eligibility_basis") or "speaker_author_work_ledger"),
    }


def _research_source_family_balance(
    visible_sources: list[dict[str, Any]],
    *,
    eligible_pool: list[dict[str, Any]],
) -> dict[str, object]:
    families: dict[str, int] = {}
    for source in visible_sources:
        family = str(source.get("source_family") or "unknown")
        families[family] = families.get(family, 0) + 1
    total = sum(families.values())
    max_family = ""
    max_fraction = 0.0
    if total:
        max_family, max_count = sorted(
            families.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        max_fraction = round(max_count / total, 4)
    eligible_families = {
        str(source.get("source_family") or "unknown")
        for source in eligible_pool
        if _research_cluster_source_eligible(
            source,
            min_confidence=SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD,
        )
    }
    alternatives_exist = len(eligible_families) > 1
    return {
        "families": families,
        "max_family": max_family,
        "max_family_fraction": max_fraction,
        "eligible_alternative_families_exist": alternatives_exist,
        "passes_70_percent_rule": (
            True
            if not alternatives_exist
            else max_fraction <= SOURCE_FAMILY_BALANCE_MAX_FRACTION
        ),
        "rule": "no visible cluster draws more than 70% of source chips from one family when eligible alternatives exist",
    }


def _research_cluster_plan_markdown(payload: dict[str, object]) -> str:
    lines = [
        f"# Research Cluster Plan — {payload.get('topic_id') or 'topic'}",
        "",
        "production_evidence: false",
        "",
        "This Markdown is a review projection only. The production evidence is `research_cluster_plan.json` plus `research_cluster_plan_receipt.json`.",
        "",
        f"Topic: `{payload.get('topic_id')}`",
        "",
    ]
    clusters = payload.get("clusters")
    if isinstance(clusters, list):
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            lines.extend(
                [
                    f"## {cluster.get('title')}",
                    "",
                    f"- cluster_id: `{cluster.get('cluster_id')}`",
                    f"- story_contribution: {cluster.get('story_contribution')}",
                    f"- retained_rationale: {cluster.get('retained_rationale')}",
                    f"- merged_rationale: {cluster.get('merged_rationale')}",
                    f"- dropped_rationale: {cluster.get('dropped_rationale')}",
                    f"- source_family_balance: {json.dumps(cluster.get('source_family_balance'), sort_keys=True)}",
                    "",
                    "### Primary Sources",
                ]
            )
            for source in cluster.get("primary_sources", []):
                if isinstance(source, dict):
                    lines.append(_research_source_markdown_row(source))
            lines.extend(["", "### Supporting Sources"])
            for source in cluster.get("supporting_sources", []):
                if isinstance(source, dict):
                    lines.append(_research_source_markdown_row(source))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _research_source_markdown_row(source: dict[str, Any]) -> str:
    people = _string_list(source.get("authors")) + _string_list(source.get("speakers"))
    return (
        f"- `{source.get('source_id')}` - {source.get('resolved_title')} "
        f"({source.get('source_family')}, confidence {source.get('confidence')}, "
        f"people: {', '.join(people)})"
    )


def _validate_research_cluster_source(
    source: Any,
    *,
    min_confidence: float,
    path: str,
    role: str,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not isinstance(source, dict):
        return [
            _research_cluster_finding(
                f"{role}_source_not_object",
                f"{role} source must be an object.",
                path,
            )
        ]
    for field in ("source_id", "resolved_title", "source_family", "contribution_note"):
        if not _non_empty_string(source.get(field)):
            errors.append(
                _research_cluster_finding(
                    f"{role}_source_{field}_missing",
                    f"{field} must be a non-empty string.",
                    f"{path}.{field}",
                )
            )
    status = str(source.get("status") or "")
    if status in BLOCKED_VISIBLE_SOURCE_STATUSES:
        errors.append(
            _research_cluster_finding(
                "visible_source_status_blocked",
                "Visible source status is blocked for research_cluster_plan use.",
                f"{path}.status",
            )
        )
    confidence = source.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or float(confidence) < min_confidence
    ):
        errors.append(
            _research_cluster_finding(
                f"{role}_source_confidence_below_threshold",
                f"{role} source confidence must be >= {min_confidence:.2f}.",
                f"{path}.confidence",
            )
        )
    authors = source.get("authors")
    speakers = source.get("speakers")
    has_people = (isinstance(authors, list) and bool(authors)) or (
        isinstance(speakers, list) and bool(speakers)
    )
    works = source.get("works")
    has_named_work = (isinstance(works, list) and any(_non_empty_string(work) for work in works)) or (
        _non_empty_string(source.get("resolved_title"))
        and not _is_generic_source_title(str(source.get("resolved_title") or ""))
    )
    # A visible source must name a real entity: a person (author/speaker) OR a
    # specific named work (paper/talk title). Title-only works are citable when
    # authors were not extracted (status missing_people); generic placeholders
    # remain blocked above by the eligibility/status checks.
    if not has_people and not has_named_work:
        errors.append(
            _research_cluster_finding(
                "visible_source_people_missing",
                "Visible source must include authors, speakers, or a specific named work.",
                path,
            )
        )
    return errors


def _research_cluster_finding(code: str, message: str, path: str) -> dict[str, str]:
    return {
        "code": code,
        "message": message,
        "path": path,
    }


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _require_safe_identifier(value: str, *, label: str) -> str:
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"{label} must be a non-empty path-safe identifier")
    return value


def _resolve_inside_workspace(root: Path, candidate: str | Path, *, label: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def _resolve_inside_deck(
    *,
    workspace: Path,
    deck_dir: Path,
    candidate: str | Path,
    label: str,
) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        if path.parts and path.parts[0] == "decks":
            path = workspace / path
        else:
            path = deck_dir / path
    resolved = path.resolve()
    try:
        resolved.relative_to(deck_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the deck directory: {resolved}") from exc
    return resolved


def _relative_path(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_json_object(path: Path, *, workspace: Path, label: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {_relative_path(path, workspace)}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be a JSON object: {_relative_path(path, workspace)}")
    return loaded


def _try_load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _inventory_source_records(inventory_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = inventory_payload.get("source_records")
    if not isinstance(records, list):
        records = inventory_payload.get("inputs")
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _contribution_notes_from_inventory(
    *,
    workspace: Path,
    source_records: list[dict[str, Any]],
    work_titles: dict[str, str] | None = None,
) -> dict[str, str]:
    notes: dict[str, str] = {}
    for record in source_records:
        path_value = record.get("path")
        kind = record.get("kind")
        if not isinstance(path_value, str) or kind not in {"analysis_report", "deep_dive"}:
            continue
        path = _resolve_inside_workspace(workspace, path_value, label="analysis contribution source")
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if _LEDGER_SECTION_NOTE_RE.match(line):
                continue
            clean = _clean_contribution_note(line)
            if not clean:
                continue
            ids = set(re.findall(r"(?:paper_abstract|paper_text):[A-Za-z0-9_-]+|speech_transcript:[A-Za-z0-9_-]+:segment:\d+", line))
            ids.update('paper_abstract:' + item for item in re.findall(r"openreview\.net/forum\?id=([A-Za-z0-9_-]+)", line))
            ids.update(re.findall(r"`([a-z_]+:[A-Za-z0-9_:-]+)`", line))
            for source_id in ids:
                title = (work_titles or {}).get(source_id, '')
                head = title.split(":", 1)[0].strip()
                names_own = bool(head and re.search(r"(?<!\w)" + re.escape(head) + r"(?!\w)", clean.split("https://", 1)[0], re.I))
                if len(ids) > 1 and not (title and names_own and (_note_matches_work(clean, title) or len(head) < 5)):
                    continue
                notes.setdefault(source_id, clean)
    return notes


def _draft_from_reference_ledger_entry(
    entry: dict[str, Any],
    *,
    source_path: str,
    contribution_notes: dict[str, str],
) -> dict[str, Any] | None:
    source_id = _first_string(entry.get("canonical_id"), entry.get("source_id"))
    if not source_id:
        return None
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    titles = _string_list(entry.get("titles"))
    labels = _string_list(entry.get("source_labels"))
    source_paths = _string_list(entry.get("source_paths"))
    # A speech-transcript segment with no real title resolved can carry its own
    # source id's namespace prefix ("event_media_caption") in `source_labels`
    # instead -- a caption-kind tag, not a title. `_is_generic_source_title`
    # (via `_first_non_generic_string` below) is deliberately blind to this: it
    # has twelve other call sites across ICLR/ICML and flags 0 of 581 live CVPR
    # transcript titles by design, so it cannot be extended here. Reuse the
    # CONTAINED `_is_transcript_placeholder_title` check built for exactly this
    # shape instead, before either field feeds `title` or the `works` list
    # below -- unfiltered, `works` carried the same raw token through verbatim
    # (real bug: three CVPR 2026 decks, ten sources titled "event_media_caption").
    titles = [value for value in titles if not _is_transcript_placeholder_title(value, source_id)]
    labels = [value for value in labels if not _is_transcript_placeholder_title(value, source_id)]
    title = _first_non_generic_string(titles) or _first_non_generic_string(labels) or ""
    authors = _people_list(metadata.get("authors")) or _people_list(metadata.get("author"))
    speakers = _people_list(metadata.get("speakers")) or _people_list(metadata.get("talk_speaker"))
    contribution = _first_string(
        metadata.get("claim"),
        metadata.get("contribution_note"),
        contribution_notes.get(source_id),
    )
    confidence = _coerce_confidence(metadata.get("confidence"), default=1.0)
    works = _unique_strings(labels or titles)
    return {
        "source_id": source_id,
        "resolved_title": title,
        "authors": authors,
        "speakers": speakers,
        "works": works,
        "source_family": _source_family(entry.get("source_family"), source_id=source_id),
        "confidence": confidence,
        "status": _initial_source_status(
            title=title,
            authors=authors,
            speakers=speakers,
            contribution_note=contribution,
            source_paths=source_paths,
        ),
        "contribution_note": contribution,
        "contribution_note_source": "analysis_line" if contribution and contribution == contribution_notes.get(source_id) else "packet_claim",
        "_packet_claim_note": contribution if contribution != contribution_notes.get(source_id) else "",
        "evidence_paths": _unique_strings([source_path] + source_paths),
    }


_CITED_SOURCE_KEYS = ("source_id", "source_ids")


def _deck_cited_source_ids(workspace: Path, inventory_payload: dict[str, Any]) -> set[str]:
    """Source ids the deck's own blueprint and deck contract cite, if they exist yet."""

    deck_dir_value = inventory_payload.get("deck_dir")
    if not isinstance(deck_dir_value, str) or not deck_dir_value:
        return set()
    try:
        deck_dir = _resolve_inside_workspace(workspace, deck_dir_value, label="deck_dir")
    except ValueError:
        return set()
    cited: set[str] = set()

    def walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, dict):
            for child_key, value in node.items():
                walk(value, child_key)
        elif isinstance(node, list):
            for value in node:
                walk(value, key)
        elif isinstance(node, str) and key in _CITED_SOURCE_KEYS and node.strip():
            cited.add(node.strip())

    for name in ("presentation_blueprint.json", "presentation_deck.json"):
        payload = _try_load_json_object(deck_dir / name)
        if payload is not None:
            walk(payload)
    return cited


def _drafts_from_topic_full_read_units(
    workspace: Path,
    topic_id: str,
    inventory_payload: dict[str, Any],
    drafts: dict[str, dict[str, Any]],
) -> tuple[int, list[Path]]:
    """Add ledger entries for works the deck cites from its own topic's full read.

    A deck is authored from one analysis revision and re-bound against a later
    one. When the later topic reference ledger no longer lists a paper the deck
    cites, the binder drops it as ``missing_from_ledger`` while the slides and
    narration still name it (CVPR 2026: Z-Order Transformer, Semantic Foam,
    Dexterous World Models, ... on six decks). The small-model full read keeps
    every unit it judged important to a topic, keyed by the same unit id the
    deck cites, with title, authors and a one-line summary. Only ids the deck
    actually cites and no other lane supplied are taken from it, and only with
    a title, named people and a summary -- never a bare id.
    """

    cited_all = _deck_cited_source_ids(workspace, inventory_payload)
    cited = cited_all - set(drafts)
    # Cited entries whose note is only a pointer stub get the same unit's
    # summary -- but only when the unit is that very work (same title): an id
    # can carry another work's claim (CVPR 2026 deck 07: a "FairLLaVA" note on
    # the id of "Robust Fair Disease Diagnosis in CT Images").
    stub_notes = {
        source_id
        for source_id in cited_all & set(drafts)
        if not str(drafts[source_id].get("contribution_note") or "").strip()
        or _CONTRIBUTION_NOTE_STUB_RE.match(str(drafts[source_id].get("contribution_note") or ""))
    }
    if not cited and not stub_notes:
        return 0, []
    # The deck's own topic first; a cited paper the small model filed under a
    # neighbouring topic is the same paper with the same title and authors.
    own = sorted(workspace.glob(f"storage/extractions/*/llm-full-read/by-topic/{topic_id}.json"))
    others = sorted(
        path
        for path in workspace.glob("storage/extractions/*/llm-full-read/by-topic/*.json")
        if path not in own
    )
    paths = own + others
    added = 0
    used: list[Path] = []
    for path in paths:
        payload = _try_load_json_object(path)
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            continue
        path_used = False
        for item in items:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("unit_id") or "").strip()
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            if source_id in stub_notes:
                summary = _first_string(item.get("summary"))
                unit_title = _first_string(metadata.get("title"), metadata.get("talk_title"))
                draft_title = str(drafts[source_id].get("resolved_title") or "").strip()
                if summary and unit_title and draft_title and (
                    _normalized_title_key(unit_title) == _normalized_title_key(draft_title)
                ):
                    drafts[source_id]["contribution_note"] = summary
                    drafts[source_id]["contribution_note_source"] = "full_read_summary"
                    stub_notes.discard(source_id)
                    path_used = True
                continue
            if source_id not in cited or source_id in drafts:
                continue
            title = _first_string(metadata.get("title"), metadata.get("talk_title"))
            authors = _people_list(metadata.get("authors"))
            speakers = _people_list(metadata.get("speakers"))
            summary = _first_string(item.get("summary"))
            if not title or not (authors or speakers) or not summary:
                continue
            _merge_ledger_draft(
                drafts,
                {
                    "source_id": source_id,
                    "resolved_title": title,
                    "authors": authors,
                    "speakers": speakers,
                    "works": [title],
                    # metadata.source_family_id is the corpus lane ("paper_text"),
                    # not the ledger's family vocabulary.
                    "source_family": _source_family(None, source_id=source_id),
                    "confidence": _coerce_confidence(item.get("confidence"), default=0.0),
                    "status": _initial_source_status(
                        title=title,
                        authors=authors,
                        speakers=speakers,
                        contribution_note=summary,
                        source_paths=[],
                    ),
                    "contribution_note": summary,
                    "contribution_note_source": "full_read_summary",
                    "evidence_paths": _unique_strings(
                        [_relative_path(path, workspace), str(item.get("source_artifact_path") or "")]
                    ),
                },
            )
            added += 1
            path_used = True
        if path_used:
            used.append(path)
    return added, used


def _draft_from_candidate_packet_entry(
    candidate: dict[str, Any],
    *,
    source_path: str,
    packet_status: str | None,
    contribution_notes: dict[str, str],
    packet_token_estimate: int | None = None,
    packet_raw_candidate_count: int | None = None,
    packet_max_candidates_per_reference: int | None = None,
    packet_reference_line_text: str = "",
) -> dict[str, Any] | None:
    source_id = _first_string(candidate.get("canonical_id"), candidate.get("source_id"))
    if not source_id:
        return None
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    titles = _string_list(candidate.get("titles_or_labels")) or _string_list(candidate.get("titles"))
    source_paths = _string_list(candidate.get("source_paths"))
    title = _first_non_generic_string(titles) or ""
    authors = _people_list(metadata.get("authors")) or _people_list(metadata.get("author"))
    speakers = _people_list(metadata.get("speakers")) or _people_list(metadata.get("talk_speaker"))
    contribution = _first_string(
        metadata.get("claim"),
        candidate.get("contribution_note"),
        contribution_notes.get(source_id),
        _snippet_contribution(candidate.get("snippets")),
    )
    confidence = _coerce_confidence(candidate.get("score"), default=0.0)
    status = packet_status or _initial_source_status(
        title=title,
        authors=authors,
        speakers=speakers,
        contribution_note=contribution,
        source_paths=source_paths,
    )
    return {
        "source_id": source_id,
        "resolved_title": title,
        "authors": authors,
        "speakers": speakers,
        "works": _unique_strings(titles),
        "source_family": _source_family(candidate.get("source_family"), source_id=source_id),
        "confidence": confidence,
        "status": status,
        "contribution_note": contribution,
        "contribution_note_source": "analysis_line" if contribution and contribution == contribution_notes.get(source_id) else "packet_claim",
        "_packet_claim_note": contribution if contribution != contribution_notes.get(source_id) else "",
        "evidence_paths": _unique_strings([source_path] + source_paths),
        "packet_token_estimate": packet_token_estimate,
        "packet_raw_candidate_count": packet_raw_candidate_count,
        "packet_max_candidates_per_reference": packet_max_candidates_per_reference,
        "packet_reference_line_text": packet_reference_line_text,
    }


_OPENREVIEW_ACCEPTED_PAPERS_REL = Path("storage/download-corpus/openreview_accepted_papers.json")


def _openreview_author_index(workspace: Path) -> dict[str, list[str]]:
    """OpenReview note id -> author names from the accepted-papers download."""

    path = workspace / _OPENREVIEW_ACCEPTED_PAPERS_REL
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    papers = payload if isinstance(payload, list) else payload.get("papers") or []
    index: dict[str, list[str]] = {}
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        note_id = str(paper.get("id") or paper.get("forum") or "").strip()
        authors = [str(a).strip() for a in paper.get("authors") or [] if str(a).strip()]
        if note_id and authors:
            index[note_id] = authors
    return index


# NOTE (Phase 10.3, decision D4): the media-asset -> item-page join that used
# to live here is gone. It grepped transcript evidence paths for a
# `media_asset/<item>/` shape that ZERO transcript evidence paths carry, then
# read an item-page probe directory that is absent in all four conference
# workspaces, so it resolved nothing anywhere. Its removal is enforced at
# SYMBOL level: `_MEDIA_ASSET_ITEM_RE`, `_item_page_title` and
# `_item_page_people` (with their HTML/JSON-LD scraping support) must not be
# reintroduced. Transcript titles now come from the local event-media
# inventory and transcript authors from the corpus's own title -> author
# index; see `_enrich_speech_transcript_metadata_from_local_join` below.


def _normalized_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _extraction_title_matches(metadata: dict[str, Any], expected_title: str, *, require_title: bool = False) -> bool:
    """Does a full-read extraction record describe ``expected_title``?

    A multi-talk session recording carries the SESSION name in ``title`` and
    the talk's own name in ``talk_title``; a segment is that talk, so either
    field may match. Preferring ``title`` alone rejected every segment of
    ICML 2026's 139-segment Pluralistic Alignment Workshop recording and left
    its talks with no contribution note although the right summary was on
    disk. A record naming neither field passes unless ``require_title``.
    """

    expected = _normalized_title_key(expected_title)
    names = [str(metadata.get(key) or "") for key in ("talk_title", "title")]
    names = [name for name in names if name]
    if not names:
        return not require_title
    return any(_normalized_title_key(name) == expected for name in names)


def _openreview_title_author_index(workspace: Path) -> dict[str, list[str]]:
    """Normalized paper title -> author names from the accepted-papers download."""

    path = workspace / _OPENREVIEW_ACCEPTED_PAPERS_REL
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    papers = payload if isinstance(payload, list) else payload.get("papers") or []
    index: dict[str, list[str]] = {}
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        title_key = _normalized_title_key(str(paper.get("title") or ""))
        authors = [str(a).strip() for a in paper.get("authors") or [] if str(a).strip()]
        if title_key and authors:
            index[title_key] = authors
    return index


# Real conference video ids are not hexadecimal -- they routinely contain
# hyphens and underscores and may start with a hyphen (e.g. `-4oS_yo7MY4`).
# The former `[0-9a-fA-F]+` class matched 0 of the 534 live CVPR transcript
# source ids, which made `_transcript_extraction_summary` dead for every
# transcript. The class is widened but deliberately NOT relaxed to a
# wildcard: group 1 is interpolated straight into a filesystem glob rooted at
# the extractions directory, and `[A-Za-z0-9_-]` excludes `/`, `\` and `.`,
# so no path separator or parent-directory sequence can survive the match.
# The pattern must stay anchored at both ends for the same reason.
# The item-page enrichment pass below is the SECOND chance at a transcript's
# title, kept alongside the deterministic local event-media join above. The two
# read different evidence: the join reads local event media (the only evidence
# present in the CVPR 2026 workspace), the pass below reads acquisition
# item-page probe HTML (present for ICLR/ICML/NeurIPS). Rows it resolves carry
# no `resolved_title_kind` marker, so the pseudo-people guard in
# `generate_editorial_blueprint.py` still refuses to derive people from their
# titles -- the safe direction.


_ITEM_PAGE_PROBES_REL = Path("cache/acquisition-downloads/item-page-probes")

_MEDIA_ASSET_ITEM_RE = re.compile(r"media_asset/([A-Za-z0-9_-]+)/")

_HTML_TITLE_RE = re.compile(r"<title>([^<]+)</title>", re.IGNORECASE)

_PROGRAM_TITLE_PREFIX_RE = re.compile(
    r"^[A-Z][A-Za-z]{1,7}\s+(?:Poster|Oral|Spotlight|Talk|Workshop|Tutorial|Invited\s+Talk)\s+",
)

def _item_page_title(workspace: Path, item_id: str) -> str:
    """Program-page title for one acquired item, from its item-page probe HTML."""

    probes_dir = workspace / _ITEM_PAGE_PROBES_REL
    if not probes_dir.is_dir():
        return ""
    for probe in sorted(probes_dir.glob(f"*-{item_id}-*.html")):
        try:
            html_text = probe.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = _HTML_TITLE_RE.search(html_text)
        if not match:
            continue
        # Program pages HTML-escape their titles ("It&#x27;s a TRAP!"); a chip or
        # a TTS line must never carry the entity.
        title = _PROGRAM_TITLE_PREFIX_RE.sub("", html.unescape(match.group(1)).strip()).strip()
        if title and not _is_generic_source_title(title):
            return title
    return ""

_LD_JSON_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)

def _collect_ld_json_people(node: Any, people: list[str]) -> None:
    """Depth-first collect schema.org ``Person`` names from parsed JSON-LD."""

    if isinstance(node, dict):
        if node.get("@type") == "Person":
            name = str(node.get("name") or "").strip()
            if name and name not in people:
                people.append(name)
        for value in node.values():
            _collect_ld_json_people(value, people)
    elif isinstance(node, list):
        for item in node:
            _collect_ld_json_people(item, people)

def _item_page_people(workspace: Path, item_id: str) -> list[str]:
    """Authoritative author/presenter names for one acquired item from the
    program page's schema.org JSON-LD (``Person`` entries). The virtual-site
    poster page names every author, so a transcript whose speaker was never
    resolved can still be attributed from the same page that named the talk."""

    probes_dir = workspace / _ITEM_PAGE_PROBES_REL
    if not probes_dir.is_dir():
        return []
    for probe in sorted(probes_dir.glob(f"*-{item_id}-*.html")):
        try:
            html_text = probe.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        people: list[str] = []
        for block in _LD_JSON_RE.findall(html_text):
            try:
                payload = json.loads(block)
            except (json.JSONDecodeError, ValueError):
                continue
            _collect_ld_json_people(payload, people)
        if people:
            return people
    return []

def _enrich_speech_transcript_titles_from_item_pages(
    workspace: Path, drafts: dict[str, dict[str, Any]]
) -> int:
    """Resolve untitled speech-transcript sources from acquisition item pages.

    Transcript ledger drafts are keyed by media-asset digests and often arrive
    with no resolved title, leaving whole talks ineligible for visible source
    chips even though the acquisition lane already downloaded the program page
    naming the work. Join each transcript's ``media_asset/<item>/`` evidence
    path to its item-page probe and take the program title; when that title
    matches an accepted paper, take the paper's authors too."""

    # A transcript needs enrichment when it has no real title (empty or a
    # generic placeholder such as ``transcript_segment``) OR no attributed
    # people. The program page already downloaded names both.
    needs_enrichment = [
        draft
        for draft in drafts.values()
        if str(draft.get("source_id") or "").startswith("speech_transcript:")
        and (
            not str(draft.get("resolved_title") or "").strip()
            or _is_generic_source_title(str(draft.get("resolved_title") or ""))
            or not (
                _string_list(draft.get("authors")) or _string_list(draft.get("speakers"))
            )
        )
    ]
    if not needs_enrichment:
        return 0
    title_author_index = _openreview_title_author_index(workspace)
    enriched = 0
    for draft in needs_enrichment:
        source_id = str(draft.get("source_id") or "")
        current_title = str(draft.get("resolved_title") or "").strip()
        labels = [current_title, *_string_list(draft.get("works"))]
        # The acquisition item id is recoverable from three places: a
        # ``media_asset/<item>/`` evidence path, a locator label the reference
        # ledger carried instead of a title ("poster-63306" IS the item id), and
        # the extraction record's own artifact path for this transcript.
        extraction_meta = _transcript_extraction_metadata(workspace, source_id)
        artifact_paths = list(extraction_meta.get("paths") or [])
        item_ids: list[str] = []
        for evidence_path in [*_string_list(draft.get("evidence_paths")), *artifact_paths]:
            item_ids.extend(_MEDIA_ASSET_ITEM_RE.findall(evidence_path))
        for label in labels:
            locator_item = _locator_item_id(label)
            if locator_item:
                item_ids.append(locator_item)
        item_ids = list(dict.fromkeys(item_id for item_id in item_ids if item_id))
        title = current_title if not _is_generic_source_title(current_title) else ""
        page_people: list[str] = []
        for item_id in item_ids:
            if not title:
                title = _item_page_title(workspace, item_id)
            if not page_people:
                page_people = _item_page_people(workspace, item_id)
            if title and page_people:
                break
        media_speakers: list[str] = []
        if not title or not page_people:
            # Workshop/talk recordings are named by the transcription metadata
            # and the talk-media inventory, not by a poster page:
            # "talk_media_001_<hash>-consolidated" resolves to the workshop or
            # talk title the acquisition lane recorded.
            media_title, media_speakers = _talk_media_title_and_speakers(
                workspace, labels=labels, extraction_meta=extraction_meta
            )
            if not title and media_title:
                title = media_title
        if not title and not page_people and not media_speakers:
            continue
        changed = False
        if media_speakers and not _string_list(draft.get("speakers")):
            draft["speakers"] = list(media_speakers)
            changed = True
        if title:
            if _is_generic_source_title(current_title) or not current_title:
                draft["resolved_title"] = title
            draft["works"] = _unique_strings(
                [w for w in _string_list(draft.get("works")) if not _is_generic_source_title(w)]
                + [title]
            )
            changed = True
        if not _string_list(draft.get("authors")):
            # The paper's own OpenReview author list, keyed by the resolved
            # title, wins; program-page JSON-LD is the fallback. The virtual
            # site can carry a display name the paper does not (ICML 2026
            # poster-64994: "Michael Lee" for the published Chia-Hsuan Lee).
            # "Poster: X" is the paper X; the kind is the chip's tag, not the title.
            bare_title = re.sub(r"^\s*[A-Za-z][A-Za-z /-]{1,24}:\s+", "", title)
            authors = (title_author_index.get(_normalized_title_key(title))
                       or title_author_index.get(_normalized_title_key(bare_title)) or page_people)
            if authors:
                draft["authors"] = list(authors)
                changed = True
        if not str(draft.get("contribution_note") or "").strip():
            summary = _transcript_extraction_summary(
                workspace, str(draft.get("source_id") or "")
            )
            if summary:
                draft["contribution_note"] = summary
                draft["contribution_note_source"] = "full_read_summary"
                changed = True
        if str(draft.get("status") or "") in {"unresolved", "missing_people", "homepage_only"}:
            draft["status"] = _initial_source_status(
                title=str(draft.get("resolved_title") or ""),
                authors=_string_list(draft.get("authors")),
                speakers=_string_list(draft.get("speakers")),
                contribution_note=str(draft.get("contribution_note") or ""),
                source_paths=[],
            )
        if changed:
            enriched += 1
    return enriched


_SPEECH_TRANSCRIPT_ID_RE = re.compile(
    r"^speech_transcript:([A-Za-z0-9_-]+)(?::segment:(\d+))?$"
)
_EXTRACTIONS_REL = Path("storage/extractions")


def _transcript_extraction_summary(workspace: Path, source_id: str, *, expected_title: str = "") -> str:
    """One-sentence contribution note for a talk segment, taken from the
    already-computed LLM full-read extraction ``summary``. A transcript whose
    contribution was never carried into the ledger is still describable from the
    same extraction that produced its evidence."""

    match = _SPEECH_TRANSCRIPT_ID_RE.match(source_id.strip())
    if not match:
        return ""
    unit_id, segment = match.group(1), match.group(2)
    extractions_dir = workspace / _EXTRACTIONS_REL
    if not extractions_dir.is_dir():
        return ""
    # Two filename shapes, searched in order. `event_media_caption_` is the
    # prefix every one of the 4,532 on-disk CVPR unit files actually carries;
    # `speech_transcript_` is the historical naming, retained so a workspace
    # written under the older convention keeps resolving. Searching only the
    # historical prefix was the second, independent half of the dead-note
    # defect: widening the id pattern alone still matches no file at all.
    #
    # `segment` stays the captured STRING. The id's "00005" interpolates to
    # `segment_00005`, which is how the file is named on disk; int-converting
    # it to `segment_5` would break a currently-correct detail.
    if segment is not None:
        patterns = [
            f"*/llm-full-read/by-unit/event_media_caption_{unit_id}_segment_{segment}.json",
            f"*/llm-full-read/by-unit/speech_transcript_{unit_id}_segment_{segment}.json",
        ]
    else:
        patterns = [
            f"*/llm-full-read/by-unit/event_media_caption_{unit_id}_segment_*.json",
            f"*/llm-full-read/by-unit/speech_transcript_{unit_id}_segment_*.json",
        ]
    for pattern in patterns:
        for path in sorted(extractions_dir.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            record = payload.get("record") or {}
            metadata = record.get("metadata") or payload.get("metadata") or {}
            if expected_title and not _extraction_title_matches(metadata, expected_title):
                continue
            summary = str(payload.get("summary") or "").strip()
            if summary:
                return summary
    if expected_title:
        for path in sorted(extractions_dir.glob("*/llm-full-read/by-topic/*.json")):
            payload = _try_load_json_object(path) or {}
            for item in payload.get("items", []):
                if not isinstance(item, dict) or item.get("unit_id") != source_id:
                    continue
                metadata = item.get("metadata") or {}
                if _extraction_title_matches(metadata, expected_title, require_title=True):
                    return str(item.get("summary") or "")
    return ""


# --- local join primitives for transcript title/author metadata -----------
#
# Every field these read already exists in a committed local JSON artifact
# with a documented schema. Nothing here parses, infers, or scrapes: anything
# that re-derives what the corpus already states is the mechanism by which
# fabricated metadata enters a deck.

_EVENT_MEDIA_INVENTORY_REL = Path(
    "storage/download-corpus/event-media/event_media_inventory.json"
)
_CORPUS_INVENTORY_REL = Path("storage/corpus/corpus_inventory.json")
_PAPER_SOURCE_FAMILIES = ("paper_text", "workshop_paper_text")


def _event_media_title_index(workspace: Path) -> dict[str, str]:
    """Map a video id to a real human title from the local event-media inventory.

    Measured on the CVPR2026 workspace: 2,352 distinct video ids, resolving
    350/350 corpus transcript records and 181/181 deck transcript video ids —
    100% of both populations. ``program_spine.json`` resolves a strict subset
    (164/181) and agrees on 100% of the overlap, which is why it is used as a
    test-time witness rather than a runtime fallback: as a fallback it would
    be dead code, as a witness it catches a mis-wired join key.

    Where two events claim the same video id the first registration wins, so
    the mapping is deterministic under file order rather than dependent on it.

    An absent, unreadable, or malformed inventory yields an empty mapping, so
    any consumer of this index is a strict no-op on a workspace that has no
    such file — which is the actual on-disk state of the iclr2026, icml2026
    and neurips2025 workspaces, and is this helper's containment argument for
    every sibling conference.
    """

    payload = _try_load_json_object(workspace / _EVENT_MEDIA_INVENTORY_REL)
    if payload is None:
        return {}
    events = payload.get("events")
    if not isinstance(events, list):
        return {}
    index: dict[str, str] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        title = str(event.get("title") or "").strip()
        if not title:
            continue
        video_ids = event.get("video_ids")
        if not isinstance(video_ids, list):
            continue
        for video_id in video_ids:
            key = str(video_id).strip()
            if key:
                index.setdefault(key, title)
    # Transcripts recovered through an evidence lane (organizer-channel
    # harvests, ASR fallbacks) were never events in the inventory above; the
    # corpus inventory records their own title. Only lane records are read, so
    # a workspace with no evidence lanes keeps its index unchanged.
    corpus = _try_load_json_object(workspace / _CORPUS_INVENTORY_REL)
    records = corpus.get("artifact_records") if isinstance(corpus, dict) else None
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict) or not record.get("evidence_lane_id"):
            continue
        if record.get("source_family_id") != "speech_transcript":
            continue
        key = str(record.get("video_id") or "").strip()
        title = str(record.get("title") or "").strip()
        if key and title:
            index.setdefault(key, title)
    return index


def _transcript_title_lookup(title_index: dict[str, str], unit_id: str) -> str:
    """Title for a transcript unit id, bare (``gHgP7UxMoig``) or lane-qualified
    (``r6-asr-fallback_gHgP7UxMoig``). The lane prefix names HOW the talk was
    transcribed; the video it names is the same talk the index knows."""

    if not unit_id:
        return ""
    if unit_id in title_index:
        return title_index[unit_id]
    match = re.match(r"^[A-Za-z0-9-]+_([A-Za-z0-9_-]{11})$", unit_id)
    return title_index.get(match.group(1), "") if match else ""


def _person_key(name: str) -> str:
    folded = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z ]", " ", folded).split())


def _drop_unverified_transcript_speakers(workspace: Path, drafts: dict[str, dict[str, Any]]) -> int:
    """Keep a talk's speaker only when the corpus knows that person as an author.

    Transcript speakers are names a model heard in the audio. On CVPR 2026, 20
    of 27 were not in the corpus's 26k-author metadata and most were garbled
    ("Kareem Aiden" for M. Kerem Aydin, "Ho Chen" for Haochen Tian, "Maria
    Brick"); Gate A reviewers verified them against the papers. A dropped name
    loses nothing false -- the chip names the talk by its title. A workspace
    without a corpus inventory is left untouched.
    """

    payload = _try_load_json_object(workspace / _CORPUS_INVENTORY_REL)
    records = payload.get("artifact_records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return 0
    known = {
        _person_key(author)
        for record in records
        if isinstance(record, dict)
        for author in (record.get("authors") or [])
        if isinstance(author, str)
    }
    dropped = 0
    for draft in drafts.values():
        if not str(draft.get("source_id") or "").startswith("speech_transcript:"):
            continue
        speakers = _string_list(draft.get("speakers"))
        kept = [name for name in speakers if _person_key(name) in known]
        if kept != speakers:
            dropped += len(speakers) - len(kept)
            draft["speakers"] = kept
    return dropped


def _corpus_title_author_index(workspace: Path) -> dict[str, list[str]]:
    """Map a NORMALIZED title key to an author list from the corpus's own papers.

    The normalization is load-bearing and reuses the module's existing
    ``_normalized_title_key`` rather than introducing a second normalizer:
    measured on the CVPR2026 workspace, 317 transcript titles join to authors
    under the normalized key versus only 306 under a plain exact-lowercase
    key. A divergent normalizer is exactly how that join silently degrades,
    and the loss looks like missing data rather than like a defect.

    A key resolving to more than one distinct author tuple is AMBIGUOUS and is
    dropped, never arbitrated — crediting a talk to the wrong researchers
    because two papers happen to normalize alike is attribution poisoning, and
    an honestly unattributed row is strictly better. Measured: 6 ambiguous
    keys out of 4,523, and 12 records carrying an empty author list, which
    contribute no key either.

    An absent, unreadable, or malformed inventory yields an empty mapping and
    therefore a strict no-op, exactly as for the event-media index above.
    """

    payload = _try_load_json_object(workspace / _CORPUS_INVENTORY_REL)
    if payload is None:
        return {}
    records = payload.get("artifact_records")
    if not isinstance(records, list):
        return {}
    candidates: dict[str, set[tuple[str, ...]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("source_family_id") not in _PAPER_SOURCE_FAMILIES:
            continue
        key = _normalized_title_key(str(record.get("paper_title") or ""))
        if not key:
            continue
        raw_authors = record.get("authors")
        authors = tuple(
            str(author).strip()
            for author in (raw_authors if isinstance(raw_authors, list) else [])
            if str(author).strip()
        )
        candidates.setdefault(key, set()).add(authors)
    return {
        key: list(next(iter(tuples)))
        for key, tuples in candidates.items()
        if len(tuples) == 1 and next(iter(tuples))
    }


_TRANSCRIPT_SESSION_KIND_LABELS = frozenset(
    {
        "paper presentation",
        "oral presentation",
        "poster presentation",
        "spotlight presentation",
        "recorded talk",
        "invited talk",
        "talk",
        "workshop",
        "keynote",
        "panel",
        "tutorial",
        "poster",
        "oral",
        "spotlight",
    }
)


def _is_transcript_placeholder_title(title: str, source_id: str) -> bool:
    """Is this transcript title a placeholder rather than a real work title?

    Three live shapes exist in the CVPR2026 ledgers: the bare namespace token
    (``event_media_caption``), the "uniquified" form that suffixes the token
    with the entry's OWN unit id (``event_media_caption:<video id>``), and a
    THIRD, differently-punctuated uniquified form seen on `source_labels`
    specifically: the qualifier and video id are `_`-joined in the entry's
    own `source_id` (``...caption_Jmt6YCzY8ks``) but `:`-joined in this
    label (``...caption:Jmt6YCzY8ks``) -- an exact-string match on `remainder
    == unit_id` missed this by one character and let it through as a real
    title on a live deck. `unit_id` (captured by `_SPEECH_TRANSCRIPT_ID_RE`)
    never contains a colon, so normalizing only `remainder`'s colons to
    underscores before comparing costs nothing on the shapes that already
    matched and catches this one too.

    Contained here on purpose. ``_is_generic_source_title`` has twelve
    in-module call sites that ICLR and ICML exercise through the shared
    presentation CLIs, and it flags 0 of the 581 live CVPR transcript titles
    anyway — so it cannot serve this purpose and must not be extended to.

    A snake_case namespace token suffixed with a bare id is a placeholder even
    when the id is a DIFFERENT entry's: a caption-recovery lane can inherit the
    original recording's placeholder as its title, and on 2026-09-25 CVPR 2026
    deck 03 showed ``event_media_caption:2tY78O-CuWY`` as a source chip for
    the recovery video ``Bj68xPcQyfE``. No human title has that shape. A real
    title is not a placeholder even when it legitimately contains a colon.
    """

    cleaned = str(title or "").strip()
    if not cleaned:
        return True
    # A session-kind label names the slot, not the work (eight CVPR 2026
    # transcripts were titled "paper presentation" by the local join).
    if cleaned.casefold() in _TRANSCRIPT_SESSION_KIND_LABELS:
        return True
    namespace, separator, remainder = cleaned.partition(":")
    namespace, remainder = namespace.strip(), remainder.strip()
    if not separator or not remainder:
        return bool(re.fullmatch(r"[a-z][a-z0-9_]*", namespace))
    if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", namespace) and re.fullmatch(
        r"[A-Za-z0-9_:\-]+", remainder
    ):
        return True
    match = _SPEECH_TRANSCRIPT_ID_RE.match(str(source_id or "").strip())
    unit_id = match.group(1) if match else ""
    return bool(unit_id) and remainder.replace(":", "_") == unit_id


def _enrich_speech_transcript_metadata_from_local_join(
    workspace: Path, drafts: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Give speech transcripts real titles, authors and notes from local joins.

    Three deterministic joins over machine-written local JSON, and nothing
    else:

    * **title** — the video id parsed out of the transcript's own source id,
      looked up in the local event-media inventory (2,352 live entries;
      resolves 181/181 CVPR deck transcript video ids).
    * **authors** — the resolved title, normalized, looked up in the corpus's
      own paper title → author index (4,517 live entries; ambiguous keys
      dropped rather than arbitrated).
    * **contribution note** — the already-computed LLM full-read extraction
      summary for the transcript's own unit.

    Nothing here parses a person out of a title, reads a cached HTML page,
    consults the program spine, or falls back to an event label. A transcript
    whose title has no author join stays honestly unattributed: measured, 19 of
    the 181 CVPR deck videos take that path, and that is the designed outcome
    rather than a gap to be filled with a substitute.

    Replaces ``_enrich_speech_transcript_titles_from_item_pages``, which
    grepped transcript evidence paths for a ``media_asset/<item>/`` shape that
    ZERO transcript evidence paths carry, then fell back to an item-page probe
    directory absent from all four conference workspaces. What that pass did
    succeed at doing was appending the entry's OWN placeholder title to its
    works list — which is how ``event_media_caption:<vid>`` became a countable
    "distinct work". Replacing the join sources without also replacing the
    placeholder test would have preserved that fabrication, which is why the
    needs-enrichment filter below uses the contained
    ``_is_transcript_placeholder_title`` rule rather than the shared
    ``_is_generic_source_title`` predicate: the shared predicate flags 0 of the
    581 live transcript titles and cannot serve this purpose.

    Returns per-pass counts: ``enriched`` (rows changed), ``marked`` (rows that
    received the provenance marker) and ``notes`` (contribution notes taken
    from an extraction summary).
    """

    counts = {"enriched": 0, "marked": 0, "notes": 0}
    needs_enrichment = [
        draft
        for draft in drafts.values()
        if str(draft.get("source_id") or "").startswith("speech_transcript:")
        and (
            _is_transcript_placeholder_title(
                str(draft.get("resolved_title") or ""),
                str(draft.get("source_id") or ""),
            )
            or not (
                _string_list(draft.get("authors")) or _string_list(draft.get("speakers"))
            )
            or not str(draft.get("contribution_note") or "").strip()
            or ":segment:" in str(draft.get("source_id") or "")
        )
    ]
    if not needs_enrichment:
        return counts

    # Loaded ONCE per pass, never per draft.
    title_index = _event_media_title_index(workspace)
    author_index = _corpus_title_author_index(workspace)
    if not title_index and not author_index and not (workspace / _EXTRACTIONS_REL).is_dir():
        # Containment property: a workspace carrying none of the three local
        # join artifacts is left byte-identical. That is the actual on-disk
        # state of the iclr2026, icml2026 and neurips2025 workspaces, and it is
        # what makes this shared-module rewrite inert for every sibling
        # conference reaching it through the shared presentation CLIs.
        return counts

    for draft in needs_enrichment:
        source_id = str(draft.get("source_id") or "")
        id_match = _SPEECH_TRANSCRIPT_ID_RE.match(source_id.strip())
        unit_id = id_match.group(1) if id_match else ""
        segment_meta = _transcript_extraction_metadata(workspace, source_id)
        segmented = bool(id_match and id_match.group(2) is not None)
        joined_title = str(segment_meta.get("talk_title") or "")
        has_segment_lane = any((workspace / _EXTRACTIONS_REL).glob(
            f"*/llm-full-read/by-unit/*_{unit_id}_segment_*.json"
        )) if segmented else False
        if not joined_title and not has_segment_lane:
            # Caption-only inventories for single-work videos have no talk
            # segmentation lane. Preserve that title join, but never substitute
            # another segment's (or its session's) title for a missing segment.
            joined_title = _transcript_title_lookup(title_index, unit_id)
        if draft.get("talk_id") and segment_meta.get("talk_id") != draft["talk_id"]:
            draft["status"] = "title_mismatch"
            continue
        prior_title = str(draft.get("resolved_title") or "")
        # The index can itself hold a placeholder (a recovery lane inherits the
        # original recording's "event_media_caption:<id>", a slot label such as
        # "paper presentation"); that is no join.
        if joined_title and _is_transcript_placeholder_title(joined_title, source_id):
            joined_title = ""
        changed = False

        if joined_title:
            if (segmented and segment_meta.get("talk_title")) or _is_transcript_placeholder_title(
                str(draft.get("resolved_title") or ""), source_id
            ):
                draft["resolved_title"] = joined_title
                if segmented and prior_title != joined_title:
                    draft["authors"] = []
                    draft["speakers"] = []
                    draft["contribution_note"] = ""
                changed = True
            # Rebuild `works` from the entry's OWN final title: every
            # placeholder-shaped member is dropped rather than carried, and the
            # real title is present exactly once. Note the entry's own title is
            # used, not the joined one, so a row that kept a pre-existing real
            # title never gains a work it does not claim.
            existing_works = _string_list(draft.get("works"))
            works = _unique_strings(
                [
                    work
                    for work in existing_works
                    if not _is_transcript_placeholder_title(work, source_id)
                    and not (segmented and work == prior_title and work != draft.get("resolved_title"))
                ]
                + [str(draft.get("resolved_title") or "")]
            )
            if works != existing_works:
                draft["works"] = works
                changed = True
            if draft.get(LOCAL_JOIN_RESOLVED_TITLE_KEY) != LOCAL_JOIN_RESOLVED_TITLE_KIND:
                draft[LOCAL_JOIN_RESOLVED_TITLE_KEY] = LOCAL_JOIN_RESOLVED_TITLE_KIND
                changed = True
            counts["marked"] += 1
        # No join: the title and the works list are left exactly as they were
        # and NO marker is written. An unmarked row is the honest signal that
        # nothing was joined, and it is what keeps the guard's negative control
        # meaningful.

        if not _string_list(draft.get("authors")):
            # The corpus's own title → author index is the ONLY author source.
            # A key it dropped as ambiguous, or never had, yields nothing: no
            # parsed name, no event label, no spine entry, no scraped page.
            authors = author_index.get(
                _normalized_title_key(str(draft.get("resolved_title") or ""))
            )
            if authors:
                draft["authors"] = list(authors)
                changed = True

        if segmented or not str(draft.get("contribution_note") or "").strip():
            summary = _transcript_extraction_summary(workspace, source_id)
            if summary:
                draft["contribution_note"] = summary
                draft["contribution_note_source"] = "full_read_summary"
                counts["notes"] += 1
                changed = True

        # Status promotion is preserved EXACTLY: only `unresolved` and
        # `missing_people` may be promoted, and promotion to `resolved` still
        # requires real people. The ambiguous entries stay blocked.
        if str(draft.get("status") or "") in {"unresolved", "missing_people"}:
            draft["status"] = (
                "resolved"
                if _string_list(draft.get("authors")) or _string_list(draft.get("speakers"))
                else "missing_people"
            )
        if changed:
            counts["enriched"] += 1
    return counts


_BY_ARTIFACT_TRANSCRIPT_GLOB = "*/by-artifact/speech_transcript_{hex_id}.json"
_BY_UNIT_TRANSCRIPT_GLOB = "*/llm-full-read/by-unit/speech_transcript_{hex_id}_segment_*.json"


def _transcript_extraction_metadata(workspace: Path, source_id: str) -> dict[str, Any]:
    """Provenance the extraction lane recorded for a transcript.

    The ledger draft usually only carries the reference-ledger JSON as evidence,
    but the extraction records name the media file the transcript came from
    (``cache/.../media_asset/poster-63306/...`` or a workshop recording), its
    sha256, its media URL, and -- for workshop recordings -- the workshop/talk
    title and speaker. Those are the join keys to the acquisition item page and
    the talk-media inventory."""

    match = _SPEECH_TRANSCRIPT_ID_RE.match(str(source_id or "").strip())
    if not match:
        return {}
    hex_id = match.group(1)
    segment = match.group(2)
    extractions_dir = workspace / _EXTRACTIONS_REL
    if not extractions_dir.is_dir():
        return {}
    result: dict[str, Any] = {"paths": []}
    unit_pattern = _BY_UNIT_TRANSCRIPT_GLOB.format(hex_id=hex_id).replace(
        "segment_*", f"segment_{segment}" if segment is not None else "segment_*"
    )
    patterns = [unit_pattern, unit_pattern.replace("speech_transcript_", "event_media_caption_")]
    artifact_pattern = _BY_ARTIFACT_TRANSCRIPT_GLOB.format(hex_id=hex_id)
    if segment is None:
        patterns.insert(0, artifact_pattern)
    else:
        patterns.append(artifact_pattern)
    for pattern in patterns:
        for path in sorted(extractions_dir.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            record = payload.get("record") if isinstance(payload.get("record"), dict) else payload
            if not isinstance(record, dict):
                continue
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            for candidate in (
                record.get("source_artifact_path"),
                metadata.get("source_artifact_path"),
                metadata.get("local_path"),
                metadata.get("source_path"),
            ):
                text = str(candidate or "").strip()
                if text and text not in result["paths"]:
                    result["paths"].append(text)
            for key in ("source_sha256", "media_url", "source_url", "talk_id", "talk_title", "workshop_title", "talk_speaker", "slideslive_id"):
                if segment is not None and pattern == artifact_pattern and key in {"talk_id", "talk_title", "workshop_title", "talk_speaker"}:
                    continue  # Whole-session metadata cannot identify this segment's talk.
                value = metadata.get(key)
                if value is not None and str(value).strip() and str(value).strip().lower() != "none":
                    result.setdefault(key, str(value).strip())
            # One file per lane is enough; the by-artifact record carries the
            # media path, the by-unit record carries the talk/workshop metadata.
            break
        if segment is None and (result.get("talk_title") or result.get("workshop_title") or result.get("source_sha256")):
            break
    return result


_TALK_MEDIA_INVENTORY_REL = Path("storage/workshops/talk_media_inventory.json")
_TALK_MEDIA_STEM_RE = re.compile(r"(talk_media_\d+_[0-9a-f]{6,})", re.I)
_MEDIA_PLACEHOLDER_TITLES = frozenset({"slideslive embedded media", "embedded media"})


_CONSOLIDATED_PREFIX_RE = re.compile(r"^\s*consolidated\s+recording\s*:\s*", re.I)


def _recording_title(talk_title: str, workshop_title: str) -> str:
    """Viewer-facing title for a workshop/talk recording.

    ``Consolidated recording: <workshop>`` (an all-day recording) is presented as
    the workshop itself (no suffix: bookend rows and chips wrap or truncate long
    names, and the workshop title is the reference the audience can look up); a
    specific talk keeps its own title."""

    talk = " ".join(str(talk_title or "").split())
    workshop = " ".join(str(workshop_title or "").split())
    if talk.lower() in _MEDIA_PLACEHOLDER_TITLES:
        talk = ""
    stripped = _CONSOLIDATED_PREFIX_RE.sub("", talk)
    if stripped != talk:
        talk = stripped or workshop
    return talk or workshop


def _talk_media_title_and_speakers(
    workspace: Path, *, labels: list[str], extraction_meta: dict[str, Any]
) -> tuple[str, list[str]]:
    """Title and speaker for a transcript named after a talk-media file.

    Prefers the transcription metadata (which names the workshop/talk and the
    speaker), then joins the acquisition talk-media inventory by media-file stem
    (``talk_media_001_9f4274a876``), sha256, or media URL."""

    title = _recording_title(
        str(extraction_meta.get("talk_title") or ""),
        str(extraction_meta.get("workshop_title") or ""),
    )
    speakers = _people_list(extraction_meta.get("talk_speaker"))
    if title and speakers:
        return title, speakers
    stems: set[str] = set()
    for text in [*labels, *(extraction_meta.get("paths") or [])]:
        for stem in _TALK_MEDIA_STEM_RE.findall(str(text or "")):
            stems.add(stem.lower())
    sha256 = str(extraction_meta.get("source_sha256") or "").strip().lower()
    media_url = str(extraction_meta.get("media_url") or "").strip().lower()
    if not stems and not sha256 and not media_url:
        return title, speakers
    path = workspace / _TALK_MEDIA_INVENTORY_REL
    if not path.is_file():
        return title, speakers
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, ValueError):
        return title, speakers
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return title, speakers
    for record in records:
        if not isinstance(record, dict):
            continue
        local_path = str(record.get("local_path") or "").lower()
        record_sha = str(record.get("sha256") or "").strip().lower()
        record_media = str(record.get("media_url") or "").strip().lower()
        matched = (
            any(stem in local_path for stem in stems)
            or (bool(sha256) and sha256 == record_sha)
            or (bool(media_url) and media_url == record_media)
        )
        if not matched:
            continue
        if not title:
            title = _recording_title(
                str(record.get("talk_title") or ""), str(record.get("workshop_title") or "")
            )
        if not speakers:
            speakers = _people_list(record.get("talk_speaker"))
        if title or speakers:
            return title, speakers
    return title, speakers


def _openreview_id_title_index(workspace: Path) -> dict[str, str]:
    """OpenReview note id -> paper title from the accepted-papers download."""

    path = workspace / _OPENREVIEW_ACCEPTED_PAPERS_REL
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    papers = payload if isinstance(payload, list) else payload.get("papers") or []
    index: dict[str, str] = {}
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        note_id = str(paper.get("id") or paper.get("forum") or "").strip()
        title = " ".join(str(paper.get("title") or "").split())
        if note_id and title:
            index[note_id] = title
    return index


def _enrich_paper_titles_from_openreview(
    workspace: Path, drafts: dict[str, dict[str, Any]]
) -> int:
    """Replace locator/generic paper titles ("papers") with the OpenReview title.

    Candidate packets sometimes carry the corpus directory name as the only
    label for a ``paper_abstract:<id>`` reference; the accepted-papers download
    already names the work by id."""

    needs_title = [
        draft
        for draft in drafts.values()
        if str(draft.get("source_id") or "").startswith("paper_abstract:")
        and (
            not str(draft.get("resolved_title") or "").strip()
            or _is_generic_source_title(str(draft.get("resolved_title") or ""))
        )
    ]
    if not needs_title:
        return 0
    title_index = _openreview_id_title_index(workspace)
    if not title_index:
        return 0
    enriched = 0
    for draft in needs_title:
        source_id = str(draft.get("source_id") or "")
        match = re.match(r"^paper_abstract:([A-Za-z0-9_-]+)$", source_id)
        if not match:
            continue
        title = title_index.get(match.group(1))
        if not title:
            continue
        draft["resolved_title"] = title
        draft["works"] = _unique_strings(
            [w for w in _string_list(draft.get("works")) if not _is_generic_source_title(w)]
            + [title]
        )
        if str(draft.get("status") or "") in {"unresolved", "missing_people", "homepage_only"}:
            draft["status"] = _initial_source_status(
                title=title,
                authors=_string_list(draft.get("authors")),
                speakers=_string_list(draft.get("speakers")),
                contribution_note=str(draft.get("contribution_note") or ""),
                source_paths=[],
            )
        enriched += 1
    return enriched


_BY_TOPIC_GLOB = "*/llm-full-read/by-topic/*.json"


def _transcript_topic_segment_counts(workspace: Path) -> dict[str, dict[str, int]]:
    """``speech_transcript:<hex>`` -> {topic_id: classified segment count}."""

    extractions_dir = workspace / _EXTRACTIONS_REL
    counts: dict[str, dict[str, int]] = {}
    if not extractions_dir.is_dir():
        return counts
    for path in sorted(extractions_dir.glob(_BY_TOPIC_GLOB)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        topic_id = str(payload.get("topic_id") or path.stem).strip()
        items = payload.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("input_kind") or "") != "transcript_segment":
                continue
            unit_id = str(item.get("unit_id") or item.get("source_artifact_id") or "")
            match = _SPEECH_TRANSCRIPT_ID_RE.match(unit_id.strip())
            if not match:
                continue
            artifact_id = f"speech_transcript:{match.group(1)}"
            per_topic = counts.setdefault(artifact_id, {})
            per_topic[topic_id] = per_topic.get(topic_id, 0) + 1
    return counts


def artifact_topic_share(
    per_topic_counts: dict[str, int], *, topic_id: str
) -> tuple[float, str, int]:
    """(share of segments classified to ``topic_id``, dominant topic, total segments)."""

    total = sum(max(0, int(count)) for count in per_topic_counts.values())
    if total <= 0:
        return 0.0, "", 0
    dominant = max(per_topic_counts.items(), key=lambda item: (item[1], item[0]))[0]
    share = per_topic_counts.get(topic_id, 0) / total
    return share, dominant, total


def is_off_topic_omnibus_artifact(
    per_topic_counts: dict[str, int],
    *,
    topic_id: str,
    min_segments: int = OMNIBUS_ARTIFACT_MIN_SEGMENTS,
    min_share: float = OMNIBUS_ARTIFACT_MIN_TOPIC_SHARE,
) -> bool:
    share, dominant, total = artifact_topic_share(per_topic_counts, topic_id=topic_id)
    if total < min_segments:
        return False
    if dominant == topic_id:
        return False
    return share < min_share


def _segment_is_its_own_talk(workspace: Path, draft: dict[str, Any]) -> bool:
    source_id = str(draft.get("source_id") or "")
    title = _normalized_title_key(str(draft.get("resolved_title") or ""))
    if ":segment:" not in source_id or not title:
        return False
    talk_title = str(_transcript_extraction_metadata(workspace, source_id).get("talk_title") or "")
    return _normalized_title_key(talk_title) == title


def _apply_transcript_artifact_topic_share(
    workspace: Path, drafts: dict[str, dict[str, Any]], *, topic_id: str
) -> int:
    """Block omnibus recordings that merely mention this topic (see the
    ``OMNIBUS_ARTIFACT_*`` constants) and annotate every transcript draft with
    its artifact-level topic share for review."""

    transcript_drafts = [
        draft
        for draft in drafts.values()
        if str(draft.get("source_id") or "").startswith("speech_transcript:")
    ]
    if not transcript_drafts or not topic_id:
        return 0
    counts = _transcript_topic_segment_counts(workspace)
    if not counts:
        return 0
    blocked = 0
    for draft in transcript_drafts:
        match = _SPEECH_TRANSCRIPT_ID_RE.match(str(draft.get("source_id") or "").strip())
        if not match:
            continue
        per_topic = counts.get(f"speech_transcript:{match.group(1)}")
        if not per_topic:
            continue
        share, dominant, total = artifact_topic_share(per_topic, topic_id=topic_id)
        draft["artifact_topic_share"] = round(share, 4)
        draft["artifact_dominant_topic_id"] = dominant
        draft["artifact_segment_count"] = total
        # A segment attributed to a named speaker is a specific talk inside the
        # recording (an invited talk at a multi-track workshop); the recording's
        # overall topic mix says nothing about that one talk. Only recording-level
        # entries (no attributed speaker) are judged by the artifact share.
        if _string_list(draft.get("speakers")):
            continue
        if is_off_topic_omnibus_artifact(per_topic, topic_id=topic_id):
            # So is a segment whose own talk title names the resolved work, even
            # with no speaker recorded (ICML 2026 deck 04: The Flexibility Trap,
            # talk_004 of the Oral 3A diffusion session).
            if _segment_is_its_own_talk(workspace, draft):
                continue
            draft["status"] = "off_topic_artifact"
            blocked += 1
    return blocked


_NARRATIVE_SOURCE_PACK_REL = Path("analysis/narrative_source_packs")
_MAX_CONTRIBUTION_NOTE_CHARS = 240


def _first_sentences(text: str, *, max_chars: int = _MAX_CONTRIBUTION_NOTE_CHARS) -> str:
    """A clean one-or-two sentence excerpt, never cut mid-word."""

    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    window = cleaned[: max_chars + 1]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut >= 60:
        return window[: cut + 1].strip()
    return window[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "."


def _paper_extraction_summary(workspace: Path, paper_id: str, *, expected_title: str = "") -> str:
    """LLM full-read summary for one paper, from the extraction lane."""

    if not paper_id:
        return ""
    extractions_dir = workspace / _EXTRACTIONS_REL
    if not extractions_dir.is_dir():
        return ""
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", paper_id)
    for pattern in (
        f"*/llm-full-read/by-unit/paper_abstract_{safe_id}.json",
        f"*/llm-full-read/by-unit/paper_abstract_{safe_id}_*.json",
        f"*/llm-full-read/by-unit/paper_text_{safe_id}.json",
        f"*/llm-full-read/by-unit/paper_text_{safe_id}_*.json",
    ):
        for path in sorted(extractions_dir.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            record = payload.get("record") or {}
            metadata = record.get("metadata") or payload.get("metadata") or {}
            if expected_title and not _extraction_title_matches(metadata, expected_title):
                continue
            summary = str(payload.get("summary") or "").strip()
            if summary:
                return summary
    if expected_title:
        for path in sorted(extractions_dir.glob("*/llm-full-read/by-topic/*.json")):
            payload = _try_load_json_object(path) or {}
            for item in payload.get("items", []):
                if not isinstance(item, dict) or item.get("unit_id") not in {f"paper_abstract:{paper_id}", f"paper_text:{paper_id}"}:
                    continue
                metadata = item.get("metadata") or {}
                if _extraction_title_matches(metadata, expected_title, require_title=True):
                    return str(item.get("summary") or "")
    return ""


def _narrative_pack_paper_notes(workspace: Path, topic_id: str) -> dict[str, str]:
    """``paper_id`` -> grounded evidence sentence from the topic's narrative
    source pack (``matched_papers`` evidence snippets and full-read summaries).

    These are the analysis lane's own recorded evidence for the paper; a paper
    the deck may cite but cannot describe is otherwise dropped from the visible
    catalog even though the workspace already says what it contributes."""

    notes: dict[str, str] = {}
    if not topic_id:
        return notes
    path = workspace / _NARRATIVE_SOURCE_PACK_REL / f"{topic_id}.json"
    if not path.is_file():
        return notes
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, ValueError):
        return notes
    if not isinstance(payload, dict):
        return notes
    for key in ("matched_papers", "llm_full_read_evidence", "paper_matching_evidence"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            paper_id = str(row.get("paper_id") or row.get("id") or "").strip()
            if not paper_id or paper_id in notes:
                continue
            note = _first_sentences(
                str(row.get("evidence_snippet") or row.get("summary") or row.get("claim") or "")
            )
            if note:
                notes[paper_id] = note
    return notes


def _openreview_abstract_index(workspace: Path) -> dict[str, str]:
    path = workspace / _OPENREVIEW_ACCEPTED_PAPERS_REL
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    papers = payload if isinstance(payload, list) else payload.get("papers") or []
    index: dict[str, str] = {}
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("id") or paper.get("forum") or "").strip()
        abstract = _first_sentences(str(paper.get("abstract") or ""))
        if paper_id and abstract:
            index[paper_id] = abstract
    return index


def _note_is_not_a_contribution(note: str, own_title: str) -> bool:
    """A note that describes nothing: a section header, a "Sources:" stub, a
    resource line of links and file paths, or the work's own title repeated
    (a packet claim that only echoes the title). ICML 2026 decks 03-05 carried
    ~20 of these per deck after the multi-work line filter; each has a
    full-read summary the fallback order can supply instead."""
    stripped = note.strip()
    if not stripped or _LEDGER_SECTION_NOTE_RE.match(stripped) or _CONTRIBUTION_NOTE_STUB_RE.match(stripped):
        return True
    if _RESOURCE_LINE_NOTE_RE.search(stripped):
        return True
    # An analysis paragraph that italicises two or more works ("*Spectron* ...
    # *Grouter* ...") is a survey of the section, not this work's contribution,
    # even when it names this work too (ICML 2026 deck 05 Grouter/Attn-QAT).
    # Opening markers are counted, so a note truncated inside the second title still counts.
    # Bold work names count too ("**T-measure** ... **A Theoretical Framework** ..."),
    # but not a bold lead sentence ("**A metric needs a justification.**").
    bold_works = [span for span in _BOLD_SPAN_RE.findall(stripped) if not span.rstrip().endswith((".", "?", "!", ":"))]
    if len(_ITALIC_WORK_OPEN_RE.findall(stripped)) + len(bold_works) >= 2:
        return True
    normalized = _normalized_title_key(stripped)
    own = _normalized_title_key(own_title)
    # "Poster: X" and "X" are the same work; the kind is the chip's tag, not the title.
    bare_own = _normalized_title_key(re.sub(r"^\s*[A-Za-z][A-Za-z /-]{1,24}:\s+", "", own_title))
    return bool(own) and normalized in {own, bare_own}


def _note_names_title(note: str, title: str) -> bool:
    """The note names the work: its short title (before any colon) appears as a word."""
    bare = re.sub(r"^\s*[A-Za-z][A-Za-z /-]{1,24}:\s+", "", title)
    return any(_note_names_short_title(note, candidate) for candidate in (title, bare))


def _note_names_short_title(note: str, title: str) -> bool:
    """The title's head (before any colon) appears in the note as a word.

    An all-caps acronym head matches case-sensitively: "REAL: Resolving
    Knowledge Conflicts ..." must not match "real-world" (ICML 2026 deck 04,
    where it disowned FG-TTL's every note and left its ledger note empty).
    """
    head = title.split(":", 1)[0].strip()
    if len(head) < 4:
        return False
    if " " not in head and head == head.upper():
        return bool(re.search(r"(?<!\w)" + re.escape(head) + r"(?!\w)", note))
    return bool(re.search(r"(?<!\w)" + re.escape(head.lower()) + r"(?!\w)", note.lower()))


def _note_belongs_to_entry(note: str, source_id: str, drafts: dict[str, dict[str, Any]]) -> bool:
    own_title = str(drafts[source_id].get("resolved_title") or "")
    if _note_is_not_a_contribution(note, own_title):
        return False
    # A named short title is stronger evidence than a shared topic word.
    names = lambda title: _note_names_short_title(note, title)  # noqa: E731
    return names(own_title) or not any(
        sid != source_id and names(str(other.get("resolved_title") or ""))
        for sid, other in drafts.items())


def _enrich_paper_contribution_notes(
    workspace: Path, drafts: dict[str, dict[str, Any]], *, topic_id: str
) -> int:
    """Repair missing/misattributed notes in evidence priority order."""
    pack = _narrative_pack_paper_notes(workspace, topic_id)
    abstracts = _openreview_abstract_index(workspace)
    derived = {sid: {**draft, "contribution_note": ""} for sid, draft in drafts.items()}
    _enrich_paper_contribution_notes_from_narrative_source_pack(workspace, topic_id, derived)
    enriched = 0
    # One note carried by works with different titles is a group claim ("SAEs can be
    # extended with polynomial decoding, or ensembled"), not either work's own
    # contribution (ICML 2026 deck 07: PolySAE/Ensembling SAEs, Certified Circuits/
    # variance analysis). Each such work falls back to its own evidence.
    titles_by_note: dict[str, set[str]] = {}
    for other in drafts.values():
        note = str(other.get("contribution_note") or "").strip()
        title = _normalized_title_key(str(other.get("resolved_title") or ""))
        if note and title:
            titles_by_note.setdefault(note, set()).add(title)
    shared_notes = {note for note, titles in titles_by_note.items() if len(titles) >= 2}
    for sid, draft in drafts.items():
        current = str(draft.get("contribution_note") or "")
        valid = (current.strip() not in shared_notes and _note_belongs_to_entry(current, sid, drafts)
                 # An analysis-pack line that never names its work is section framing
                 # ("A credible program should use both ..."), not that work's result.
                 and (draft.get("contribution_note_source") != "analysis_line"
                      or _note_names_title(current, str(draft.get("resolved_title") or ""))))
        paper_id = sid.split(":", 1)[-1]
        def own_summary() -> str:
            return (_paper_extraction_summary(workspace, paper_id, expected_title=str(draft.get("resolved_title") or ""))
                    if sid.startswith(("paper_abstract:", "paper_text:")) else
                    _transcript_extraction_summary(workspace, sid, expected_title=str(draft.get("resolved_title") or "")))
        if valid and draft.get("contribution_note_source") == "analysis_line":
            # Analysis-pack lines are section synthesis: they name neighbouring works in
            # plain text ("TICL learns ...; Fed-CAW weights ...; TRACE models ...", ICML
            # 2026 deck 10), which no markup rule can see. The work's own full-read
            # summary wins whenever it exists and belongs to the work.
            own = _first_sentences(own_summary())
            if own and own.strip() not in shared_notes and _note_belongs_to_entry(own, sid, drafts):
                draft["contribution_note"] = own
                draft["contribution_note_source"] = "full_read_summary"
                enriched += 1
            continue
        if valid:
            draft.setdefault("contribution_note_source", "full_read_summary")
            continue
        summary = own_summary()
        candidates = [
            ("full_read_summary", _first_sentences(summary)),
            ("packet_claim", str(draft.get("_packet_claim_note") or "")),
            ("narrative_pack", pack.get(paper_id, "") or str(derived[sid].get("contribution_note") or "")),
            ("openreview_abstract", abstracts.get(paper_id, "")),
        ]
        draft["contribution_note"] = ""
        draft["contribution_note_source"] = "none"
        for source, note in candidates:
            if note.strip() not in shared_notes and _note_belongs_to_entry(note, sid, drafts):
                draft["contribution_note"] = note
                draft["contribution_note_source"] = source
                if source == "narrative_pack" and derived[sid].get("contribution_note_kind"):
                    draft["contribution_note_kind"] = derived[sid]["contribution_note_kind"]
                enriched += 1
                break
        if draft.get("contribution_note") and draft.get("status") in {"missing_contribution", "missing_people", "unresolved", "homepage_only"}:
            draft["status"] = _initial_source_status(title=str(draft.get("resolved_title") or ""), authors=_string_list(draft.get("authors")), speakers=_string_list(draft.get("speakers")), contribution_note=draft["contribution_note"], source_paths=[])
    return enriched


def _enrich_paper_authors_from_openreview(
    workspace: Path, drafts: dict[str, dict[str, Any]]
) -> int:
    """Fill missing paper authors from the OpenReview accepted-papers metadata.

    The reference ledgers and candidate packets often carry paper titles without
    author lists, which leaves whole decks unable to NAME any scholarship — the
    Topic 1 build shipped zero named people while every bound paper had a full
    author list available in the download corpus."""

    needs_authors = any(
        not _string_list(draft.get("authors"))
        and str(draft.get("source_id") or "").startswith("paper_abstract:")
        for draft in drafts.values()
    )
    if not needs_authors:
        return 0
    author_index = _openreview_author_index(workspace)
    if not author_index:
        return 0
    enriched = 0
    for draft in drafts.values():
        if _string_list(draft.get("authors")):
            continue
        source_id = str(draft.get("source_id") or "")
        match = re.match(r"^paper_abstract:([A-Za-z0-9_-]+)$", source_id)
        if not match:
            continue
        authors = author_index.get(match.group(1))
        if not authors:
            continue
        draft["authors"] = list(authors)
        if str(draft.get("status") or "") == "missing_people":
            draft["status"] = "resolved"
        enriched += 1
    return enriched


def _chunk_abstract_window(chunk: dict[str, Any]) -> str:
    """The widest note-derivation text a pack chunk offers.

    ``quote`` is a *selector* input — ``source_pack_selection`` lowercases it
    for keyword ranking and for its admission filter — so it is frozen at its
    truncation width and must never be widened in place. ``abstract_window``
    is the separate, derivation-only wide field. Reading through this one
    accessor keeps every note reader on the same resolution rule; a workspace
    whose chunks predate the wide field falls back to ``quote`` and behaves
    exactly as it did before."""

    return str(chunk.get("abstract_window") or chunk.get("quote") or "")


def _narrative_pack_abstract_chunks_by_hash(pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a narrative source pack's paper chunks by bare hash.

    When two chunks share one hash, keep whichever carries the literal
    ``DERIVED_ABSTRACT_NOTE_MARKER`` in its derivation window in preference to
    one that does not — a plain last-match-wins loop would discard the
    abstract-bearing chunk whenever a later, non-abstract chunk for the same
    paper follows it. Both sides of the comparison resolve through
    ``_chunk_abstract_window`` so the incoming chunk and the incumbent cannot
    drift onto different fields."""

    by_hash: dict[str, dict[str, Any]] = {}
    chunks = pack.get("chunks")
    if not isinstance(chunks, list):
        return by_hash
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        match = re.match(
            r"^paper:([A-Za-z0-9_-]+)$", str(chunk.get("extraction_record_id") or "")
        )
        if not match:
            continue
        hash_ = match.group(1)
        existing = by_hash.get(hash_)
        is_better = existing is None or (
            DERIVED_ABSTRACT_NOTE_MARKER in _chunk_abstract_window(chunk)
            and DERIVED_ABSTRACT_NOTE_MARKER not in _chunk_abstract_window(existing)
        )
        if is_better:
            by_hash[hash_] = chunk
    return by_hash


def _abstract_derived_contribution_note(quote: str) -> str:
    """Extract usable abstract-derived note text from a chunk's quote.

    Returns an empty string when the marker is absent or when the cleaned
    text after it is shorter than ``DERIVED_ABSTRACT_NOTE_MIN_CHARS`` — the
    caller must leave the draft untouched in that case rather than fall back
    to a title, bibliographic line, raw quote, or any constant."""

    idx = quote.find(DERIVED_ABSTRACT_NOTE_MARKER)
    if idx < 0:
        return ""
    text = quote[idx + len(DERIVED_ABSTRACT_NOTE_MARKER):]
    cleaned = _clean_contribution_note(text)
    if len(cleaned) < DERIVED_ABSTRACT_NOTE_MIN_CHARS:
        return ""
    return cleaned


def _enrich_paper_contribution_notes_from_narrative_source_pack(
    workspace: Path, topic_id: str, drafts: dict[str, dict[str, Any]]
) -> int:
    """Give paper-family drafts a deterministic, source-grounded fallback note.

    ``build_reference_ledger`` never sets ``claim``/``contribution_note`` for
    paper-family entries, which leaves them ineligible for a visible source
    chip no matter how good the underlying selection is. This pass supplies a
    note extracted verbatim from the paper's own abstract text in the topic's
    narrative source pack (``analysis/narrative_source_packs/<topic_id>.json``),
    when that text is genuinely usable — a fallback, not a gate relaxation.
    Research measured this fallback's real yield at 48/60 (80%) of current
    candidates; the remaining ~20% have no usable abstract text and are left
    ineligible, as designed, not patched around."""

    pack_path = workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"
    if not pack_path.is_file():
        return 0
    pack = _try_load_json_object(pack_path)
    if pack is None:
        return 0
    chunks_by_hash = _narrative_pack_abstract_chunks_by_hash(pack)
    if not chunks_by_hash:
        return 0
    enriched = 0
    for draft in drafts.values():
        if str(draft.get("contribution_note") or "").strip():
            continue
        if not (_string_list(draft.get("authors")) or _string_list(draft.get("speakers"))):
            continue
        source_id = str(draft.get("source_id") or "")
        match = re.match(r"^paper_abstract:([A-Za-z0-9_-]+)$", source_id)
        if not match:
            continue
        chunk = chunks_by_hash.get(match.group(1))
        if chunk is None:
            continue
        note = _abstract_derived_contribution_note(_chunk_abstract_window(chunk))
        if not note:
            continue
        draft["contribution_note"] = note
        draft["contribution_note_kind"] = "derived_abstract"
        draft["contribution_note_source"] = "narrative_pack"
        if str(draft.get("status") or "") == "missing_contribution":
            draft["status"] = "resolved"
        enriched += 1
    return enriched


def _merge_ledger_draft(drafts: dict[str, dict[str, Any]], draft: dict[str, Any] | None) -> None:
    if not draft:
        return
    source_id = draft["source_id"]
    existing = drafts.get(source_id)
    if existing is None:
        drafts[source_id] = draft
        return
    if not existing.get("contribution_note") and draft.get("contribution_note"):
        existing["contribution_note_source"] = draft.get("contribution_note_source", "full_read_summary")
    if not existing.get("_packet_claim_note") and draft.get("_packet_claim_note"):
        existing["_packet_claim_note"] = draft["_packet_claim_note"]
    for key in ("resolved_title", "source_family", "contribution_note"):
        if not existing.get(key) and draft.get(key):
            existing[key] = draft[key]
    for key in ("authors", "speakers", "works", "evidence_paths"):
        existing[key] = _unique_strings(_string_list(existing.get(key)) + _string_list(draft.get(key)))
    existing["confidence"] = max(float(existing.get("confidence") or 0.0), float(draft.get("confidence") or 0.0))
    existing["status"] = _merge_status(str(existing.get("status") or ""), str(draft.get("status") or ""))


# Paper titles and abstracts arrive with their typesetting still in them: inline
# math ("$\oslash$ Source Models Leak What They Shouldn't $\nrightarrow$") and PDF
# line-break hyphens ("identify sam- ples from unknown classes"). Both reached
# CVPR 2026 slides verbatim through source chips. The ledger is the one place
# every chip title and note passes, so it is cleaned here, once -- by the shared
# rules in ``display_text``, where text macros such as ``\\texttt{X}`` are
# unwrapped rather than leaking their own name onto the slide.
def _latex_math_to_text(math: str) -> str:
    return latex_to_text(math)


def _clean_display_text(text: str) -> str:
    """Strip typesetting (inline math, text macros, PDF line-break hyphens) from a title or note."""

    return clean_display_text(text)


_HONORIFICS = frozenset({"dr", "dr.", "prof", "prof.", "professor", "mr", "mr.", "ms", "ms.", "mrs", "mrs.", "sir"})


def _is_full_person_name(name: str) -> bool:
    """At least two name tokens once honorifics are set aside ("Dr. Jen" is not)."""

    tokens = [token for token in str(name).split() if token.lower() not in _HONORIFICS]
    return len(tokens) >= 2


# "Source: *Title*, ." / "Sources: `paper:7546...`" -- a pointer left where a
# contribution note should be; shown verbatim on CVPR 2026 resource slides.
_CONTRIBUTION_NOTE_STUB_RE = re.compile(r"^\s*Sources?\s*:\s")
# A resource line: Markdown links, backticked file paths, or an "*Title* — ;" entry
# with nothing after the dash. Such lines list where a work lives, not what it did.
_BOLD_SPAN_RE = re.compile(r"\*\*([^*]+?)\*\*")
_ITALIC_WORK_OPEN_RE = re.compile(r"(?<![*\w])\*(?=[^*\s])")
_RESOURCE_LINE_NOTE_RE = re.compile(r"\]\(https?://|`[^`]*/[^`]*`|\*\s*[—–-]\s*;|^\s*Presentation artifacts:")


def _finalize_ledger_entries(drafts: dict[str, dict[str, Any]]) -> list[SpeakerAuthorWorkLedgerEntry]:
    ranked_drafts = sorted(
        drafts.values(),
        key=lambda item: (
            -float(item.get("confidence") or 0.0),
            str(item.get("source_family") or ""),
            str(item.get("source_id") or ""),
        ),
    )
    entries: list[SpeakerAuthorWorkLedgerEntry] = []
    for rank, draft in enumerate(ranked_drafts, start=1):
        source_id = str(draft["source_id"])
        title = _clean_display_text(str(draft.get("resolved_title") or ""))
        # A placeholder or slot label ("workshop", "event_media_caption:<id>")
        # is no title; shown as one it reached a CVPR 2026 slide as the chip.
        if title and _is_transcript_placeholder_title(title, source_id):
            title = ""
        # A one-word "speaker" is a caption first name or an extraction slip
        # ("Bob", "Parent"), never a verifiable person; authors come from
        # paper metadata and are kept as given.
        speakers = [name for name in _string_list(draft.get("speakers")) if _is_full_person_name(name)]
        authors = _string_list(draft.get("authors"))
        status = str(draft.get("status") or "unresolved")
        if status == "resolved" and not (authors or speakers):
            status = "missing_people"
        note = _clean_display_text(str(draft.get("contribution_note") or ""))
        if not _note_belongs_to_entry(note, source_id, drafts):
            note = ""
        entry_payload = {
            "resolved_title": title,
            "authors": authors,
            "speakers": speakers,
            "contribution_note": note,
            "confidence": float(draft.get("confidence") or 0.0),
            "status": status,
        }
        entries.append(
            SpeakerAuthorWorkLedgerEntry(
                source_id=str(draft["source_id"]),
                resolved_title=entry_payload["resolved_title"],
                authors=entry_payload["authors"],
                speakers=entry_payload["speakers"],
                works=_unique_strings(
                    [
                        _clean_display_text(work)
                        for work in _string_list(draft.get("works"))
                        # A locator ("talk_015", a segmentation id) names where
                        # a talk sits in a recording, never a work it shows.
                        if not _is_locator_label(work)
                        and not _is_transcript_placeholder_title(work, source_id)
                    ]
                ),
                source_family=str(draft.get("source_family") or "unknown"),
                confidence=entry_payload["confidence"],
                status=entry_payload["status"],
                visible_source_chip_eligible=visible_source_chip_eligible(entry_payload),
                contribution_note=entry_payload["contribution_note"],
                contribution_note_source=str(draft.get("contribution_note_source") or "none") if note else "none",
                rank=rank,
                evidence_paths=_string_list(draft.get("evidence_paths")),
                packet_token_estimate=draft.get("packet_token_estimate"),
                packet_raw_candidate_count=draft.get("packet_raw_candidate_count"),
                packet_max_candidates_per_reference=draft.get("packet_max_candidates_per_reference"),
                packet_reference_line_text=str(draft.get("packet_reference_line_text") or ""),
                contribution_note_kind=str(draft.get("contribution_note_kind") or ""),
                resolved_title_kind=str(
                    draft.get(LOCAL_JOIN_RESOLVED_TITLE_KEY) or ""
                ),
                artifact_topic_share=draft.get("artifact_topic_share"),
                artifact_dominant_topic_id=str(draft.get("artifact_dominant_topic_id") or ""),
                artifact_segment_count=draft.get("artifact_segment_count"),
            )
        )
    return entries


def _reviewed_repair_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    repairs = payload.get("repairs")
    if not isinstance(repairs, list):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for repair in repairs:
        if not isinstance(repair, dict):
            continue
        source_id = repair.get("source_id")
        if isinstance(source_id, str) and source_id.strip():
            mapped[source_id.strip()] = repair
    return mapped


def _source_repair_finding_code(entry: dict[str, Any]) -> str:
    title = str(entry.get("resolved_title") or "").strip()
    status = str(entry.get("status") or "").strip()
    if title == "Structured presentation source":
        return "generic_structured_presentation_source"
    if _is_generic_source_title(title):
        if title.lower() == "openreview":
            return "generic_openreview_fallback"
        return "homepage_only"
    if status in BLOCKED_VISIBLE_SOURCE_STATUSES:
        return status
    if status == "missing_people":
        return "missing_author_or_speaker"
    if status == "missing_contribution":
        return "missing_contribution_note"
    if not _string_list(entry.get("authors")) and not _string_list(entry.get("speakers")):
        return "missing_author_or_speaker"
    if not str(entry.get("contribution_note") or "").strip():
        return "missing_contribution_note"
    confidence = entry.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return "confidence_missing"
    if float(confidence) < SUPPORTING_SOURCE_CONFIDENCE_THRESHOLD:
        return "confidence_below_supporting_threshold"
    if entry.get("visible_source_chip_eligible") is False:
        return "visible_source_chip_ineligible"
    return ""


def _source_repair_finding(
    entry: dict[str, Any],
    *,
    finding_code: str,
    repair: dict[str, Any] | None,
) -> dict[str, object]:
    repaired_value = repair.get("repaired_value") if isinstance(repair, dict) else {}
    if not isinstance(repaired_value, dict):
        repaired_value = {}
    confidence = (
        repair.get("confidence")
        if isinstance(repair, dict) and "confidence" in repair
        else entry.get("confidence")
    )
    confidence_value = _coerce_confidence(confidence, default=0.0)
    review_status = (
        str(repair.get("review_status") or "needs_review")
        if isinstance(repair, dict)
        else "needs_review"
    )
    reviewer_notes = (
        str(repair.get("reviewer_notes") or "")
        if isinstance(repair, dict)
        else ""
    )
    if not reviewer_notes:
        reviewer_notes = _default_reviewer_notes(finding_code)
    repair_payload = {
        **(repair if isinstance(repair, dict) else {}),
        "confidence": confidence_value,
        "review_status": review_status,
        "repaired_value": repaired_value,
    }
    return {
        "finding_code": finding_code,
        "source_id": str(entry.get("source_id") or ""),
        "source_path": _source_path_for_finding(entry),
        "original_value": _source_repair_original_value(entry),
        "repaired_value": repaired_value,
        "confidence": confidence_value,
        "review_status": review_status,
        "visible_use_unlocked": repair_visible_use_unlocked(repair_payload),
        "reviewer_notes": reviewer_notes,
        # A review can vouch for a source on the slides it was checked against
        # and not others it is also cited on (CVPR 2026 deck 08: CATS supports
        # the confidence-routing slides, not the layer-selection one).
        "slide_ids": _string_list(repair.get("slide_ids")) if isinstance(repair, dict) else [],
    }


def _source_repair_original_value(entry: dict[str, Any]) -> dict[str, object]:
    value: dict[str, object] = {
        "resolved_title": str(entry.get("resolved_title") or ""),
        "authors": _string_list(entry.get("authors")),
        "speakers": _string_list(entry.get("speakers")),
        "works": _string_list(entry.get("works")),
        "source_family": str(entry.get("source_family") or ""),
        "confidence": _coerce_confidence(entry.get("confidence"), default=0.0),
        "status": str(entry.get("status") or ""),
        "visible_source_chip_eligible": bool(entry.get("visible_source_chip_eligible")),
        "contribution_note": str(entry.get("contribution_note") or ""),
    }
    if entry.get("packet_token_estimate") is not None:
        value["packet_token_estimate"] = entry.get("packet_token_estimate")
        value["packet_raw_candidate_count"] = entry.get("packet_raw_candidate_count")
        value["packet_max_candidates_per_reference"] = entry.get("packet_max_candidates_per_reference")
        value["packet_reference_line_text"] = str(entry.get("packet_reference_line_text") or "")
    return value


def _source_path_for_finding(entry: dict[str, Any]) -> str:
    evidence_paths = _string_list(entry.get("evidence_paths"))
    if evidence_paths:
        return evidence_paths[0]
    return str(entry.get("source_id") or "")


def _default_reviewer_notes(finding_code: str) -> str:
    notes = {
        "missing_author_or_speaker": "No reviewed author or speaker evidence is available; visible use remains blocked.",
        "missing_contribution_note": "No reviewed contribution note is available; visible use remains blocked.",
        "generic_structured_presentation_source": "Structured presentation source is a generic fallback and remains blocked until reviewed evidence resolves it.",
        "generic_openreview_fallback": "OpenReview fallback lacks resolved work and people metadata; visible use remains blocked.",
        "homepage_only": "Homepage-only source evidence remains blocked until reviewed title and people metadata resolve it.",
        "unresolved": "Unresolved source id remains blocked until reviewed repair evidence resolves it.",
        "ambiguous": "Ambiguous source candidate remains blocked until reviewed repair evidence resolves it.",
        "title_mismatch": "Title mismatch remains blocked until reviewed repair evidence resolves it.",
        "packet_too_large": "Oversized candidate packet remains blocked until reviewed repair evidence resolves it.",
    }
    return notes.get(finding_code, "Visible source use remains blocked until reviewed repair evidence resolves this finding.")


def _blocker_count(payload: dict[str, Any]) -> int:
    blockers = payload.get("blockers")
    return len(blockers) if isinstance(blockers, list) else 0


# A packet whose candidate list was cut by the count cap while its token
# estimate stayed well under the model's input budget was shortened for
# bookkeeping, not because evidence had to be dropped. Blocking every candidate
# in such a packet is what kept five fully corroborated ICML 2026 topic-01
# papers (real OpenReview titles, authors and extraction summaries) out of the
# visible catalog. Only a truncation that actually pressed against the budget
# still blocks.
PACKET_BUDGET_PRESSURE_RATIO = 0.9


def _packet_truncation_was_budget_bound(packet: dict[str, Any]) -> bool:
    config = packet.get("config") if isinstance(packet.get("config"), dict) else {}
    try:
        estimate = float(packet.get("token_estimate") or 0.0)
        budget = float(config.get("max_packet_input_tokens") or 0.0)
    except (TypeError, ValueError):
        return True
    if budget <= 0:
        return True
    return estimate >= budget * PACKET_BUDGET_PRESSURE_RATIO


def _packet_blocked_status(packet: dict[str, Any]) -> str | None:
    if packet.get("packet_truncated") is True and _packet_truncation_was_budget_bound(packet):
        return "packet_too_large"
    blockers = packet.get("blockers")
    if not isinstance(blockers, list):
        return None
    codes = {
        str(blocker.get("code") or "")
        for blocker in blockers
        if isinstance(blocker, dict)
    }
    if any("ambiguous" in code for code in codes):
        return "ambiguous"
    if any("title_mismatch" in code for code in codes):
        return "title_mismatch"
    if any("homepage" in code for code in codes):
        return "homepage_only"
    if any("packet" in code and "large" in code for code in codes):
        return "packet_too_large"
    if codes:
        return "unresolved"
    return None


def _initial_source_status(
    *,
    title: str,
    authors: list[str],
    speakers: list[str],
    contribution_note: str,
    source_paths: list[str],
) -> str:
    if not title:
        return "unresolved"
    if _is_generic_source_title(title):
        return "homepage_only"
    if title.lower().strip() == "structured presentation source":
        return "homepage_only"
    if _is_homepage_only(source_paths, title=title):
        return "homepage_only"
    if not authors and not speakers:
        return "missing_people"
    if not contribution_note:
        return "missing_contribution"
    return "resolved"


def _merge_status(existing: str, incoming: str) -> str:
    if existing in BLOCKED_VISIBLE_SOURCE_STATUSES and incoming in BLOCKED_VISIBLE_SOURCE_STATUSES:
        return existing
    if existing == "resolved":
        return existing
    if incoming == "resolved":
        return incoming
    if existing:
        return existing
    return incoming or "unresolved"


def _source_family(value: Any, *, source_id: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if source_id.startswith("paper_abstract:"):
        return "papers"
    if source_id.startswith("speech_transcript:"):
        return "talks"
    return "unknown"


def _coerce_confidence(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(max(0.0, min(1.0, float(value))), 4)
    return default


def _first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_non_generic_string(values: list[str]) -> str:
    for value in values:
        if value and not _is_generic_source_title(value):
            return value
    return ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique_strings(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _people_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return _unique_strings(part.strip() for part in re.split(r",|;|\band\b", value) if part.strip())
    return _string_list(value)


def _unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _snippet_contribution(snippets: Any) -> str:
    if not isinstance(snippets, list):
        return ""
    for snippet in snippets:
        if not isinstance(snippet, dict):
            continue
        text = _clean_contribution_note(str(snippet.get("text") or ""))
        if text:
            return text
    return ""


def _clean_contribution_note(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    compact = compact.strip("- ")
    compact = re.sub(r"`(?:paper_abstract:[A-Za-z0-9]+|speech_transcript:[A-Za-z0-9]+:segment:\d+|:[0-9]+)`", "", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) > 280:
        compact = compact[:277].rstrip() + "..."
    return compact


# Acquisition/locator labels that name WHERE an artifact lives, never WHAT it is.
# A poster always corresponds to a paper, so "poster-63306" is a locator for that
# paper, not a title; likewise "talk_media_001_<hash>-consolidated" is a media
# file name and "papers" is a corpus directory. Any of these surfacing as a
# ledger ``resolved_title`` leaks into source chips ("poster-63306") and into the
# narration ("this poster shows ..."), which is exactly the audience-visible
# defect reported on the ICML 2026 agentic-AI deck.
_LOCATOR_LABEL_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:poster|oral|spotlight|session|talk|paper|workshop|tutorial|event|item)[-_ ]?\d+$", re.I),
    re.compile(r"^talk_media_\d+_[0-9a-f]{6,}(?:[-_][a-z0-9]+)*$", re.I),
    re.compile(r"^[0-9a-f]{12,}$", re.I),
    re.compile(r"^(?:speech_transcript|paper_abstract|workshop_paper|media_asset|talk_media)[:_].*$", re.I),
    re.compile(r"^(?:papers|artifacts|posters|talks|transcripts|media|sources|records|by[-_ ]topic)$", re.I),
    re.compile(r"^[a-z0-9]+-raw(?:\.[a-z0-9]+)?$", re.I),
    # A programme slot: "Oral Session 2A: 3D Reconstruction [1:00-2:15]". It
    # names WHEN a block of talks happened, not what any one of them showed,
    # and it reached two CVPR 2026 blueprints as a cited work. The bare forms
    # above ("oral-12") never matched it because it reads like a title.
    # ...and the session must be IDENTIFIED ("Session 2A", "Session 6D"). A
    # talk genuinely called "Poster session highlights on 6G" is a work.
    re.compile(r"^(?:oral|poster|spotlight|demo|workshop|tutorial|keynote|panel)\s+session\s+\d+[a-z]?\b", re.I),
    # ...or anything tagged with a programme time range, whatever it is called.
    re.compile(r".*\[\s*\d{1,2}:\d{2}\s*[-\u2013]\s*\d{1,2}:\d{2}\s*\]\s*$"),
)


def _is_locator_label(value: str) -> bool:
    """Return whether ``value`` is an acquisition locator rather than a work title."""

    text = " ".join(str(value or "").split()).strip()
    if not text:
        return False
    return any(pattern.match(text) for pattern in _LOCATOR_LABEL_RES)


is_locator_label = _is_locator_label


def _locator_item_id(value: str) -> str:
    """``poster-63306`` / ``poster 63306`` -> ``poster-63306`` (acquisition item id)."""

    match = re.match(
        r"^(poster|oral|spotlight|session|talk|workshop|tutorial|event)[-_ ]?(\d+)$",
        " ".join(str(value or "").split()).strip(),
        re.I,
    )
    if not match:
        return ""
    return f"{match.group(1).lower()}-{match.group(2)}"


# Corpus/lane nouns that name WHAT KIND of thing a source is, never which work
# it is. The plural forms arrive as corpus directory names ("papers"); the
# singular forms arrive as candidate-packet labels ("paper"), which is how five
# distinct ICML 2026 topic-01 papers all reached the ledger titled "paper" and
# collapsed into a single catalog work.
_KIND_NOUN_TITLES = frozenset(
    {
        "paper",
        "papers",
        "poster",
        "posters",
        "talk",
        "talks",
        "abstract",
        "abstracts",
        "artifact",
        "artifacts",
        "work",
        "works",
        "item",
        "items",
        "record",
        "records",
        "reference",
        "references",
        "session",
        "sessions",
        "event",
        "events",
        "media",
        "source",
        "sources",
        "workshop paper",
        "conference paper",
        "paper abstract",
    }
)


def _is_generic_source_title(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if normalized in {
        "openreview",
        "openreview forum",
        "homepage",
        "structured presentation source",
        "presentation source",
        "paper url",
        "transcript segment",
        "transcript",
        "speech transcript",
    }:
        return True
    if normalized in _KIND_NOUN_TITLES:
        return True
    return _is_locator_label(value)


def _is_homepage_only(source_paths: list[str], *, title: str) -> bool:
    if not source_paths:
        return False
    if not _is_generic_source_title(title):
        return False
    return any(path.startswith(("http://", "https://")) for path in source_paths)


def _inventory_record(
    path: Path,
    *,
    workspace: Path,
    kind: str,
    production_evidence: bool,
    evidence_status: str,
) -> InventoryRecord:
    stat = path.stat()
    return InventoryRecord(
        kind=kind,
        path=_relative_path(path, workspace),
        sha256=_sha256_file(path),
        byte_count=stat.st_size,
        production_evidence=production_evidence,
        evidence_status=evidence_status,
    )


def _source_kind(relative_path: str) -> str:
    path = Path(relative_path)
    parts = path.parts
    name = path.name
    if len(parts) >= 2 and parts[0] == "analysis":
        if parts[1] == "deep_dive_topics":
            return "deep_dive"
        if parts[1] == "exhaustive_summaries":
            if "__papers" in name:
                return "exhaustive_papers"
            if "__transcripts" in name:
                return "exhaustive_transcripts"
            return "exhaustive_summary"
        if parts[1] == "narrative_source_packs":
            return "narrative_source_pack"
        if parts[1] == "provenance":
            return "provenance"
        if parts[1] == "topic_reference_ledgers":
            return "topic_reference_ledger"
        if parts[1] == "topic_reference_validation":
            return "topic_reference_validation"
        if parts[1] == "topic_reference_repairs":
            return "topic_reference_repair"
        if parts[1] == "topic_reference_candidate_resolution":
            return "topic_reference_candidate_resolution"
        return "analysis_report"
    if name == "source_attributions.json":
        return "source_attributions"
    if name in {"ranked_papers.json", "source_pack_ranked_papers.json"}:
        return "ranked_papers"
    if _is_old_numbered_markdown_name(name):
        return "legacy_numbered_markdown"
    return "source_input"


def _current_artifact_kind(name: str) -> str:
    if name == "presentation.html":
        return "current_canary_html"
    if name == "presentation.pdf":
        return "current_canary_pdf"
    if name == "presentation_video.mp4":
        return "current_canary_single_speaker_mp4"
    if name == "presentation_video_notebooklm.mp4":
        return "current_canary_notebooklm_mp4"
    if name == "source_binding_report.json":
        return "current_source_binding_report"
    if name == "human_review_receipt.json":
        return "current_human_review_receipt"
    return "current_canary_artifact"


def _is_rejected_non_production_source(path: Path) -> bool:
    name = path.name
    return _is_old_numbered_markdown_name(name) or "SCAFFOLD" in name.upper()


def _is_old_numbered_markdown_name(name: str) -> bool:
    return _OLD_NUMBERED_MARKDOWN_RE.match(name) is not None


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
