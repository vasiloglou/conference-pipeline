"""Receipt-bound, provider-free CVPR 2026 r5 Stage 5 materialization.

This adapter is deliberately narrow: it converts the exact succeeded r5
event-media handoff into caption transcripts and final captionless blockers.
It has no acquisition, media download, ASR, OCR, crawler, or provider path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.event_media import (
    EVENT_MEDIA_INVENTORY_RELATIVE_PATH,
    R5_EVENT_MEDIA_RUN_ID,
    canonical_event_media_inventory_digest,
    validate_event_media_inventory,
)
from conference_pipeline.acquisition.youtube_talks import (
    caption_sidecar_matches_video_id,
    parse_caption_file,
)
from conference_pipeline.conference_identity import (
    ConferenceIdentityError,
    load_conference_identity,
    parse_virtual_site_boundary,
)
from conference_pipeline.transcription_ocr import (
    atomic_write_json,
    extract_pdf_text_record,
    resolve_workspace_member,
    sha256_file,
    workspace_relpath,
)


R5_EVENT_MEDIA_RECEIPT_SHA256 = "d6b0e6ac3dcea684ae699021d90c9ef737e239325b2fa496177efc25349038bc"
R5_EVENT_MEDIA_FILE_SHA256 = "e6c7e00c3919e5652433a94ade336b121c83297dbd8780500576532d7e945a09"
R5_EVENT_MEDIA_CANONICAL_DIGEST = "09e2d6ec1d8150b5e78f7d263018aeb227d73ee3a4109d13d9be162bf37bc55e"
R4_OVERLAY_DIGEST = "7f622be554f054425f6dff6ae0911b3a0b83acb6699fe5a15ed37e70cd0d0d12"

PROCESSOR_IDENTITY = "cvpr2026.caption_materializer"
PROCESSOR_VERSION = "1"
STAGE5_DIR = Path("storage/transcription-ocr/cvpr2026")
STAGE5_INVENTORY = STAGE5_DIR / "inventory.json"
STAGE5_CAPTIONS_DIR = STAGE5_DIR / "caption-transcripts"
STAGE5_PAPER_TEXT_DIR = STAGE5_DIR / "paper-text"
_CAPTION_MIMES = frozenset({"text/vtt", "application/x-subrip"})


class CvprTranscriptionOcrAdmissionError(ValueError):
    """Raised when a declared Stage 5 input is not receipt-backed and safe."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _json_digest(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "inventory_sha256"}
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CvprTranscriptionOcrAdmissionError(f"{label} is not a readable JSON object") from exc
    if not isinstance(payload, dict):
        raise CvprTranscriptionOcrAdmissionError(f"{label} is not a JSON object")
    return payload


def _workspace_path(workspace: Path, path_value: object, *, label: str) -> Path:
    resolved, error = resolve_workspace_member(workspace, path_value)
    if resolved is None or error is not None:
        raise CvprTranscriptionOcrAdmissionError(f"{label} is outside the workspace or missing")
    return resolved


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if value != expected:
        raise CvprTranscriptionOcrAdmissionError(f"{label} does not match the r5 authority")


def validate_cvpr_r5_handoff_read_only(
    *, workspace: Path, receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, str]]:
    """Validate published r5 Stage-4 authority without receiving any Stage-5 path.

    The byte digest is deliberately checked before parsing the producer receipt
    or opening its declared inventory.  This is the shared, no-write admission
    boundary for the Stage-5 owner and its independent verifier.
    """

    resolved_receipt = _workspace_path(workspace, receipt_path, label="stage-04 receipt")
    receipt_digest = sha256_file(resolved_receipt)
    _require_exact(receipt_digest, R5_EVENT_MEDIA_RECEIPT_SHA256, label="stage-04 receipt byte digest")
    receipt = _load_object(resolved_receipt, label="stage-04 receipt")
    _require_exact(receipt.get("run_id"), R5_EVENT_MEDIA_RUN_ID, label="stage-04 run id")
    _require_exact(receipt.get("stage_id"), "download-corpus", label="stage-04 id")
    _require_exact(receipt.get("stage_order"), 4, label="stage-04 order")
    _require_exact(receipt.get("mode"), "live", label="stage-04 mode")
    _require_exact(receipt.get("status"), "succeeded", label="stage-04 status")
    stage_contract = receipt.get("stage_contract")
    if not isinstance(stage_contract, Mapping):
        raise CvprTranscriptionOcrAdmissionError("stage-04 receipt lacks its published stage contract")
    _require_exact(stage_contract.get("status"), "pass", label="stage-04 contract status")
    _require_exact(stage_contract.get("stage_id"), "download-corpus", label="stage-04 contract id")

    inventory_ref = receipt.get("event_media_inventory")
    if not isinstance(inventory_ref, Mapping):
        raise CvprTranscriptionOcrAdmissionError("stage-04 receipt lacks its event-media inventory binding")
    _require_exact(
        inventory_ref.get("path"), EVENT_MEDIA_INVENTORY_RELATIVE_PATH.as_posix(), label="event-media inventory path"
    )
    _require_exact(
        inventory_ref.get("inventory_sha256"),
        R5_EVENT_MEDIA_CANONICAL_DIGEST,
        label="event-media inventory published canonical digest",
    )
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping):
        raise CvprTranscriptionOcrAdmissionError("stage-04 receipt lacks published authority checks")
    _require_exact(
        checks.get("event_media_inventory_path"),
        EVENT_MEDIA_INVENTORY_RELATIVE_PATH.as_posix(),
        label="event-media inventory check path",
    )
    _require_exact(
        checks.get("event_media_inventory_sha256"), R5_EVENT_MEDIA_FILE_SHA256, label="event-media inventory check file digest"
    )
    _require_exact(
        checks.get("event_media_inventory_canonical_sha256"),
        R5_EVENT_MEDIA_CANONICAL_DIGEST,
        label="event-media inventory check canonical digest",
    )
    _require_exact(
        checks.get("r4_registration_media_overlay_sha256"), R4_OVERLAY_DIGEST, label="immutable r4 overlay digest"
    )

    inventory_path = _workspace_path(workspace, inventory_ref["path"], label="event-media inventory")
    if sha256_file(inventory_path) != R5_EVENT_MEDIA_FILE_SHA256:
        raise CvprTranscriptionOcrAdmissionError("event-media inventory file digest does not match its receipt")
    inventory = _load_object(inventory_path, label="event-media inventory")
    if canonical_event_media_inventory_digest(inventory) != R5_EVENT_MEDIA_CANONICAL_DIGEST:
        raise CvprTranscriptionOcrAdmissionError("event-media inventory canonical digest does not match its receipt")
    try:
        boundary = parse_virtual_site_boundary(load_conference_identity(workspace).virtual_url)
        validate_event_media_inventory(
            inventory,
            boundary=boundary,
            require_terminal=True,
            require_caption_terminal=True,
        )
    except (ConferenceIdentityError, ValueError) as exc:
        raise CvprTranscriptionOcrAdmissionError("event-media inventory fails the r5 terminal contract") from exc
    lineage = inventory.get("source_lineage")
    if not isinstance(lineage, Mapping) or lineage.get("r4_registration_media_overlay_sha256") != R4_OVERLAY_DIGEST:
        raise CvprTranscriptionOcrAdmissionError("event-media inventory does not bind the immutable r4 overlay")
    return (
        receipt,
        inventory,
        inventory_path,
        {
            "stage4_receipt_sha256": receipt_digest,
            "event_media_file_sha256": sha256_file(inventory_path),
            "event_media_canonical_sha256": canonical_event_media_inventory_digest(inventory),
            "r4_overlay_sha256": R4_OVERLAY_DIGEST,
        },
    )


def _admit_r5_handoff(
    *, workspace: Path, receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, str]]:
    """Keep the internal admission name while using the no-write boundary."""

    return validate_cvpr_r5_handoff_read_only(workspace=workspace, receipt_path=receipt_path)


def _caption_input(
    *, workspace: Path, candidate: Mapping[str, Any], seen_source_identities: set[str]
) -> dict[str, Any]:
    video_id = candidate.get("video_id")
    if not isinstance(video_id, str) or not video_id or video_id in seen_source_identities:
        raise CvprTranscriptionOcrAdmissionError("caption candidate has a duplicate or invalid source identity")
    seen_source_identities.add(video_id)
    if candidate.get("caption_status") != "youtube_caption_available":
        raise CvprTranscriptionOcrAdmissionError("caption input is not terminally available")
    caption = candidate.get("caption")
    if not isinstance(caption, Mapping):
        raise CvprTranscriptionOcrAdmissionError("available caption lacks sidecar provenance")
    path = _workspace_path(workspace, caption.get("local_path"), label="caption sidecar")
    if not caption_sidecar_matches_video_id(path, video_id):
        raise CvprTranscriptionOcrAdmissionError("caption sidecar is not bound to its video id")
    if caption.get("mime") not in _CAPTION_MIMES:
        raise CvprTranscriptionOcrAdmissionError("caption sidecar MIME is not admissible")
    if not isinstance(caption.get("byte_count"), int) or caption["byte_count"] <= 0:
        raise CvprTranscriptionOcrAdmissionError("caption sidecar byte count is invalid")
    if not _is_sha256(caption.get("sha256")):
        raise CvprTranscriptionOcrAdmissionError("caption sidecar digest is invalid")
    if not path.exists() or path.stat().st_size != caption["byte_count"] or sha256_file(path) != caption["sha256"]:
        raise CvprTranscriptionOcrAdmissionError("caption sidecar bytes do not match receipt provenance")
    if not isinstance(caption.get("language"), str) or not caption["language"].strip():
        raise CvprTranscriptionOcrAdmissionError("caption sidecar language is invalid")
    if caption.get("transcript_source") not in {"youtube_caption_manual", "youtube_caption_auto"}:
        raise CvprTranscriptionOcrAdmissionError("caption sidecar provenance is invalid")
    segments = parse_caption_file(path)
    if not segments:
        raise CvprTranscriptionOcrAdmissionError("caption sidecar contains no timed speech segments")
    return {
        "source_identity": video_id,
        "source_class": "caption_transcript",
        "video_id": video_id,
        "source_event_urls": list(candidate.get("source_event_urls", [])),
        "provenance": list(candidate.get("provenance", [])),
        "caption_path": workspace_relpath(workspace, path),
        "caption_sha256": caption["sha256"],
        "caption_byte_count": caption["byte_count"],
        "caption_mime": caption["mime"],
        "language": caption["language"],
        "caption_provenance": caption["transcript_source"],
        "segments": segments,
        "normalized_speech_text": " ".join(str(segment["text"]) for segment in segments).strip(),
    }


def _caption_blocker(candidate: Mapping[str, Any], *, seen_source_identities: set[str]) -> dict[str, Any]:
    video_id = candidate.get("video_id")
    if not isinstance(video_id, str) or not video_id or video_id in seen_source_identities:
        raise CvprTranscriptionOcrAdmissionError("caption candidate has a duplicate or invalid source identity")
    seen_source_identities.add(video_id)
    if candidate.get("caption_status") != "youtube_caption_unavailable" or candidate.get("caption") is not None:
        raise CvprTranscriptionOcrAdmissionError("unavailable-caption class was reclassified or has sidecar metadata")
    return {
        "source_identity": video_id,
        "source_class": "caption_blocker",
        "video_id": video_id,
        "source_event_urls": list(candidate.get("source_event_urls", [])),
        "provenance": list(candidate.get("provenance", [])),
        "code": "youtube_caption_unavailable",
        "asr_eligible": False,
        "reason": "r5 caption-only policy forbids audio recovery and ASR fallback",
    }


def _materialize_caption(
    *, workspace: Path, input_record: Mapping[str, Any], compatibility: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    output_path = workspace / STAGE5_CAPTIONS_DIR / f"{input_record['video_id']}-{str(input_record['caption_sha256'])[:12]}.json"
    output = {
        "processor_identity": PROCESSOR_IDENTITY,
        "processor_version": PROCESSOR_VERSION,
        "semantic_class": "caption_transcript",
        "input_compatibility": dict(compatibility),
        "outcome": "caption_used_no_asr",
        **dict(input_record),
    }
    if output_path.exists():
        try:
            prior = _load_object(output_path, label="prior caption transcript")
        except CvprTranscriptionOcrAdmissionError:
            prior = None
        if prior is not None and all(prior.get(key) == value for key, value in output.items()):
            digest = sha256_file(output_path)
            return ({**prior, "output_path": workspace_relpath(workspace, output_path), "output_sha256": digest}, None)
        superseded = {
            "source_identity": input_record["source_identity"],
            "prior_output_path": workspace_relpath(workspace, output_path),
            "reason": "caption_output_incompatible",
        }
    else:
        superseded = None
    atomic_write_json(output_path, output)
    return (
        {
            **output,
            "output_path": workspace_relpath(workspace, output_path),
            "output_sha256": sha256_file(output_path),
        },
        superseded,
    )


def _workshop_row(
    *, workspace: Path, row: Mapping[str, Any], index_digest: str
) -> dict[str, Any]:
    source_identity = row.get("source_identity")
    base = {
        "source_identity": source_identity,
        "predecessor_receipt_sha256": row.get("predecessor_receipt_sha256"),
        "receipt_index_sha256": row.get("receipt_index_sha256"),
        "declared_path": row.get("declared_path"),
        "mime_type": row.get("mime_type"),
        "byte_count": row.get("byte_count"),
        "sha256": row.get("sha256"),
    }
    if not isinstance(source_identity, str) or not source_identity:
        return {**base, "outcome_class": "workshop_pdf_source_identity_invalid", "output_digest": None}
    if not _is_sha256(row.get("predecessor_receipt_sha256")) or row.get("receipt_index_sha256") != index_digest:
        return {**base, "outcome_class": "workshop_pdf_lineage_unproven", "output_digest": None}
    if row.get("mime_type") != "application/pdf":
        return {**base, "outcome_class": "workshop_pdf_mime_invalid", "output_digest": None}
    try:
        source_path = _workspace_path(workspace, row.get("declared_path"), label="workshop PDF")
    except CvprTranscriptionOcrAdmissionError:
        return {**base, "outcome_class": "workshop_pdf_path_unproven", "output_digest": None}
    if not source_path.exists() or not isinstance(row.get("byte_count"), int) or not _is_sha256(row.get("sha256")):
        return {**base, "outcome_class": "workshop_pdf_lineage_unproven", "output_digest": None}
    if source_path.read_bytes()[:4] == b"PK\x03\x04":
        return {**base, "outcome_class": "workshop_pdf_zip_content", "output_digest": None}
    if source_path.stat().st_size != row["byte_count"] or sha256_file(source_path) != row["sha256"]:
        return {**base, "outcome_class": "workshop_pdf_digest_mismatch", "output_digest": None}
    extracted = extract_pdf_text_record(
        workspace=workspace,
        lane="cvpr2026-workshop-paper",
        source_path=source_path,
        output_root=workspace / STAGE5_PAPER_TEXT_DIR,
        resume=True,
    )
    return {
        **base,
        "outcome_class": "workshop_paper_text_extraction",
        "paper_text_record": extracted,
        "output_digest": extracted.get("output_sha256"),
    }


def _workshop_table(*, workspace: Path, receipt_index: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if receipt_index is None:
        return []
    index_digest = receipt_index.get("sha256")
    predecessor_receipt_digest = receipt_index.get("predecessor_receipt_sha256")
    rows = receipt_index.get("rows")
    if (
        not _is_sha256(index_digest)
        or not _is_sha256(predecessor_receipt_digest)
        or not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
    ):
        raise CvprTranscriptionOcrAdmissionError("workshop receipt index is not a bounded digest-bound table")
    table: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise CvprTranscriptionOcrAdmissionError("workshop receipt index has a non-object row")
        source_identity = row.get("source_identity")
        if not isinstance(source_identity, str) or source_identity in seen:
            raise CvprTranscriptionOcrAdmissionError("workshop receipt index has duplicate source identity")
        seen.add(source_identity)
        if row.get("predecessor_receipt_sha256") != predecessor_receipt_digest:
            table.append(
                {
                    "source_identity": source_identity,
                    "predecessor_receipt_sha256": row.get("predecessor_receipt_sha256"),
                    "receipt_index_sha256": row.get("receipt_index_sha256"),
                    "declared_path": row.get("declared_path"),
                    "mime_type": row.get("mime_type"),
                    "byte_count": row.get("byte_count"),
                    "sha256": row.get("sha256"),
                    "outcome_class": "workshop_pdf_lineage_unproven",
                    "output_digest": None,
                }
            )
            continue
        table.append(_workshop_row(workspace=workspace, row=row, index_digest=index_digest))
    return table


def _caption_outputs_complete(*, workspace: Path, prior: Mapping[str, Any], compatibility: Mapping[str, Any]) -> bool:
    transcripts = prior.get("caption_transcripts")
    if not isinstance(transcripts, list):
        return False
    for record in transcripts:
        if not isinstance(record, Mapping):
            return False
        try:
            output_path = _workspace_path(workspace, record.get("output_path"), label="prior caption output")
        except CvprTranscriptionOcrAdmissionError:
            return False
        if not output_path.exists() or not _is_sha256(record.get("output_sha256")) or sha256_file(output_path) != record["output_sha256"]:
            return False
        try:
            output = _load_object(output_path, label="prior caption output")
        except CvprTranscriptionOcrAdmissionError:
            return False
        if (
            output.get("processor_identity") != PROCESSOR_IDENTITY
            or output.get("processor_version") != PROCESSOR_VERSION
            or output.get("semantic_class") != "caption_transcript"
            or output.get("input_compatibility") != compatibility
        ):
            return False
    return True


def _resume_payload(
    *, workspace: Path, output_path: Path, compatibility: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, int]:
    if not output_path.exists():
        return None, 0
    try:
        prior = _load_object(output_path, label="prior Stage-5 inventory")
    except CvprTranscriptionOcrAdmissionError:
        prior = None
    if (
        prior is not None
        and prior.get("compatibility") == compatibility
        and prior.get("inventory_sha256") == _json_digest(prior)
        and _caption_outputs_complete(workspace=workspace, prior=prior, compatibility=compatibility)
    ):
        return prior, 0
    if prior is not None:
        prior_digest = str(prior.get("inventory_sha256") or sha256_file(output_path))[:16]
    else:
        prior_digest = sha256_file(output_path)[:16]
    superseded = output_path.with_name(f"inventory.superseded-{prior_digest}.json")
    output_path.replace(superseded)
    return None, 1


def build_cvpr_r5_transcription_ocr_inventory(
    *,
    workspace: Path,
    receipt_path: Path,
    workshop_pdf_receipt_index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize only admitted r5 captions and declared workshop-PDF rows.

    The upstream receipt and inventory are both checked before any output is
    opened.  Every candidate has one final class: a caption transcript or a
    final no-ASR blocker.  The function never discovers a file by scanning.
    """

    workspace = workspace.resolve()
    _receipt, inventory, _inventory_path, input_digests = _admit_r5_handoff(
        workspace=workspace, receipt_path=receipt_path
    )
    compatibility = {
        "run_id": R5_EVENT_MEDIA_RUN_ID,
        **input_digests,
        "processor_identity": PROCESSOR_IDENTITY,
        "processor_version": PROCESSOR_VERSION,
        "semantic_class": "caption_first_no_asr",
    }
    output_path = workspace / STAGE5_INVENTORY
    prior, superseded_attempt_count = _resume_payload(
        workspace=workspace, output_path=output_path, compatibility=compatibility
    )
    if prior is not None:
        reused = copy.deepcopy(prior)
        reused["resume"] = {
            "reused_output_digest": prior["inventory_sha256"],
            "superseded_attempt_count": 0,
        }
        return reused

    candidates = inventory.get("caption_candidates")
    if not isinstance(candidates, list):
        raise CvprTranscriptionOcrAdmissionError("admitted event-media inventory lacks caption candidates")
    seen_source_identities: set[str] = set()
    caption_inputs: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise CvprTranscriptionOcrAdmissionError("caption candidate is not an object")
        if candidate.get("caption_status") == "youtube_caption_available":
            caption_inputs.append(_caption_input(workspace=workspace, candidate=candidate, seen_source_identities=seen_source_identities))
        else:
            blockers.append(_caption_blocker(candidate, seen_source_identities=seen_source_identities))

    materialized = [
        _materialize_caption(workspace=workspace, input_record=record, compatibility=compatibility)
        for record in caption_inputs
    ]
    caption_transcripts = [record for record, _superseded in materialized]
    superseded_caption_attempts = [attempt for _record, attempt in materialized if attempt is not None]
    workshop_table = _workshop_table(workspace=workspace, receipt_index=workshop_pdf_receipt_index)
    counts = {
        "terminal_route_count": int(inventory["counts"]["terminal_event_route_count"]),
        "caption_candidate_count": len(candidates),
        "caption_used_no_asr_count": len(caption_transcripts),
        "youtube_caption_unavailable_count": len(blockers),
        "asr_queue_count": 0,
        "ocr_queue_count": 0,
        "provider_call_count": 0,
    }
    total_caption_bytes = sum(int(record["caption_byte_count"]) for record in caption_inputs)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_inventory",
        "run_id": R5_EVENT_MEDIA_RUN_ID,
        "processor_identity": PROCESSOR_IDENTITY,
        "processor_version": PROCESSOR_VERSION,
        "semantic_class": "caption_first_no_asr",
        "compatibility": compatibility,
        "input_digests": {
            **input_digests,
        },
        "capacity_reservation": {
            "accepted_caption_input_bytes": total_caption_bytes,
            "accepted_caption_count": len(caption_inputs),
            "reservation_bytes": max(1, total_caption_bytes * 2),
            "safety_margin_bytes": max(1, total_caption_bytes // 10),
        },
        "caption_transcripts": caption_transcripts,
        "blockers": blockers,
        "paper_text_records": [
            row["paper_text_record"] for row in workshop_table if isinstance(row.get("paper_text_record"), Mapping)
        ],
        "workshop_pdf_provenance_table": workshop_table,
        "workshop_talk_boundary": {"status": "not_claimed", "reason": "no receipt-backed workshop talk media"},
        "asr_queue": [],
        "ocr_queue": [],
        "media_recovery_queue": [],
        "provider_calls": [],
        "counts": counts,
        "resume": {"reused_output_digest": None, "superseded_attempt_count": superseded_attempt_count},
        "superseded_caption_attempts": superseded_caption_attempts,
    }
    payload["inventory_sha256"] = _json_digest(payload)
    atomic_write_json(output_path, payload)
    return payload
