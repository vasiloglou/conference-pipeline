"""Upstream-receipt walker for the exhaustive-topic-extraction stage.

The walker traverses every completed transcript, OCR JSON, local-text-extraction,
page-image OCR, and selected paper artifact recorded in either the historical
v1.5 receipts or the current corpus/transcription inventories and yields
:class:`ExtractionInput` rows ready for the classifier. All paths are resolved
through :func:`workflow._resolve_path_inside_workspace` so a malformed upstream
receipt cannot escape the workspace sandbox.

The walker also persists per-artifact JSON records and aggregates them into a
per-topic JSON index. Resume semantics: when the upstream digest matches the
record's stored ``source_artifact_digest``, classification is skipped.
``--rebuild`` forces full re-extraction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from conference_pipeline.extractions.classifier import classify_artifact
from conference_pipeline.extractions.records import (
    ExtractionRecord,
    aggregate_by_topic,
    safe_artifact_filename,
    write_artifact_record,
)
from conference_pipeline.iclr2025_live import (
    COMPLETED_MEDIA_PROCESSING_STATUSES,
    load_json,
    load_topic_families,
    require_file,
    sha256_file,
)
from conference_pipeline.workflow import _resolve_path_inside_workspace

LOG = logging.getLogger("conference_pipeline.extractions.walker")

_PAPER_KIND = "paper_pdf"
_PAPER_METADATA_KIND = "paper_metadata_fallback"
_TRANSCRIPT_KIND = "transcript"
_OCR_JSON_KIND = "ocr_json"
_PDF_TEXT_KIND = "pdf_text"
_PAGE_IMAGE_KIND = "page_image"

_INPUT_KIND_ORDER = (
    _TRANSCRIPT_KIND,
    _OCR_JSON_KIND,
    _PDF_TEXT_KIND,
    _PAGE_IMAGE_KIND,
    _PAPER_KIND,
    _PAPER_METADATA_KIND,
)


@dataclass(frozen=True)
class ExtractionInput:
    """Single artifact ready for classification.

    ``status`` is the upstream ``processing_status`` (or the receipt status for
    paper PDFs); items not in
    :data:`conference_pipeline.iclr2025_live.COMPLETED_MEDIA_PROCESSING_STATUSES`
    are still yielded so the classifier can record an explicit
    ``upstream_blocked`` reason.
    """

    artifact_id: str
    kind: str
    text_path: str | None
    digest: str
    status: str
    source_path: str
    topic_hint: str | None
    text_content: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkerResult:
    """Aggregate counters returned by :func:`walk_and_extract`."""

    expected_count: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    by_class_counts: dict[str, int] = field(default_factory=dict)
    resumed_count: int = 0
    extracted_count: int = 0
    rewritten_count: int = 0
    by_kind_counts: dict[str, int] = field(default_factory=dict)
    # Records whose artifact no longer exists in the corpus (renamed or removed
    # upstream), and topic files for topics no current record carries; both
    # are moved aside, never deleted.
    pruned_artifact_count: int = 0
    pruned_topic_count: int = 0
    pruned_dir: str | None = None


def _safe_read_text(path: Path, *, max_chars: int = 200_000) -> str | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        # Defensive cap to avoid loading huge files into memory.
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_chars)
    except OSError:
        return None


def _safe_read_ocr_text(path: Path, *, max_chars: int = 200_000) -> str | None:
    raw = _safe_read_text(path, max_chars=max_chars)
    if raw is None:
        return None
    # OCR JSON files have a top-level ``text`` field; other shapes degrade gracefully.
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(payload, dict):
        for key in ("text", "ocr_text", "transcript"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value[:max_chars]
        for key in ("segments", "talks", "transcript_segments"):
            blocks = payload.get(key)
            if isinstance(blocks, list):
                chunks: list[str] = []
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    chunks.extend(
                        str(block.get(field) or "").strip()
                        for field in ("talk_title", "talk_speaker", "description", "text")
                        if str(block.get(field) or "").strip()
                    )
                joined = "\n".join(chunks).strip()
                if joined:
                    return joined[:max_chars]
        # Concatenate any list-of-blocks shape.
        blocks = payload.get("blocks")
        if isinstance(blocks, list):
            chunks = [
                str(block.get("text") or block.get("content") or "")
                for block in blocks
                if isinstance(block, dict)
            ]
            joined = "\n".join(part for part in chunks if part).strip()
            if joined:
                return joined[:max_chars]
    return None


def _infer_kind(corpus_row: dict[str, Any], receipt_row: dict[str, Any]) -> str:
    source_family = str(corpus_row.get("source_family") or corpus_row.get("source_family_id") or "")
    processing_status = str(corpus_row.get("processing_status") or receipt_row.get("status") or "")
    indexed_text_kind = str(corpus_row.get("indexed_text_kind") or "")
    text_path = receipt_row.get("text_path") or corpus_row.get("text_path")

    if source_family in {"openreview-venue", "paper_text", "workshop_paper_text"}:
        return _PAPER_KIND
    if source_family == "speech_transcript":
        return _TRANSCRIPT_KIND
    if indexed_text_kind in {"transcript_json", "annotated_transcript_json"}:
        return _TRANSCRIPT_KIND
    if indexed_text_kind == "ocr_json":
        return _OCR_JSON_KIND
    if processing_status == "transcribed" and corpus_row.get("transcript_path"):
        return _TRANSCRIPT_KIND
    if processing_status == "ocr_done" and corpus_row.get("ocr_path"):
        return _OCR_JSON_KIND
    if processing_status == "pdf_pages_rendered":
        return _PDF_TEXT_KIND
    if processing_status == "image_artifact_indexed":
        return _PAGE_IMAGE_KIND
    if processing_status == "local_text_extracted":
        return _PDF_TEXT_KIND
    if text_path:
        return _PDF_TEXT_KIND
    return _PAGE_IMAGE_KIND


def _row_text_value(
    corpus_row: dict[str, Any],
    receipt_row: dict[str, Any],
    key: str,
) -> str | None:
    for row in (corpus_row, receipt_row):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _select_text_relpath(corpus_row: dict[str, Any], receipt_row: dict[str, Any]) -> str | None:
    indexed_text_kind = str(corpus_row.get("indexed_text_kind") or "").strip()
    source_family = str(corpus_row.get("source_family") or corpus_row.get("source_family_id") or "").strip()

    if indexed_text_kind == "annotated_transcript_json":
        for key in (
            "annotated_transcript_json_path",
            "indexed_text_path",
            "transcript_path",
            "text_path",
            "transcript_json_path",
        ):
            value = _row_text_value(corpus_row, receipt_row, key)
            if value:
                return value

    if source_family == "speech_transcript" or indexed_text_kind == "transcript_json":
        for key in (
            "transcript_path",
            "text_path",
            "annotated_transcript_json_path",
            "transcript_json_path",
            "indexed_text_path",
        ):
            value = _row_text_value(corpus_row, receipt_row, key)
            if value:
                return value

    for key in (
        "indexed_text_path",
        "annotated_transcript_json_path",
        "transcript_json_path",
        "transcript_path",
        "ocr_path",
        "text_path",
    ):
        value = _row_text_value(corpus_row, receipt_row, key)
        if value:
            return value
    text_path = receipt_row.get("text_path")
    if isinstance(text_path, str) and text_path.strip():
        return text_path
    return None


def _transcript_path_from_json_sidecar(path: Path) -> str | None:
    raw = _safe_read_text(path, max_chars=200_000)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("transcript_path", "text_path", "indexed_text_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and not value.strip().lower().endswith(".json"):
            return value
    return None


def _digest_for_row(corpus_row: dict[str, Any], resolved_path: Path | None) -> str:
    for key in (
        "text_sha256",
        "indexed_text_sha256",
        "provenance_digest",
        "text_extraction_sha256",
        "source_sha256",
    ):
        digest = corpus_row.get(key)
        if isinstance(digest, str) and digest:
            return digest
    if resolved_path is not None and resolved_path.exists() and resolved_path.is_file():
        try:
            return sha256_file(resolved_path)
        except OSError:
            return ""
    return ""


def _paper_id_from_artifact(
    artifact_id: str,
    *,
    source_path: str | None = None,
) -> str | None:
    """Best-effort OpenReview paper id extraction for de-duping paper inputs."""
    raw = str(artifact_id or "").strip()
    for prefix in ("openreview:", "paper:", "paper_metadata_fallback:"):
        if raw.startswith(prefix):
            value = raw.split(":", 1)[1].strip()
            return value or None
    if ":" in raw:
        suffix = raw.rsplit(":", 1)[-1].strip()
        if suffix:
            return suffix
    if source_path:
        import re

        match = re.search(r"/openreview/([^/]+)-paper_pdf\.pdf$", str(source_path))
        if match:
            return match.group(1)
    return None


def _metadata_digest_for_paper(paper: dict[str, Any]) -> str:
    normalized = json.dumps(paper, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    import hashlib

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _openreview_metadata_text(paper: dict[str, Any]) -> str:
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()
    primary_area = str(paper.get("primary_area") or "").strip()
    venue = str(paper.get("venue") or "").strip()
    authors = paper.get("authors") if isinstance(paper.get("authors"), list) else []
    keywords = paper.get("keywords") if isinstance(paper.get("keywords"), list) else []
    lines = [
        f"Paper Title: {title}",
        f"OpenReview ID: {paper.get('id') or paper.get('forum') or ''}",
        f"Venue: {venue}",
        f"Primary Area: {primary_area}",
        "Authors: " + ", ".join(str(author) for author in authors if str(author).strip()),
        "Keywords: " + ", ".join(str(keyword) for keyword in keywords if str(keyword).strip()),
        "",
        "Abstract:",
        abstract,
    ]
    return "\n".join(line for line in lines if line is not None).strip()


def _first_existing_workspace_file(workspace_root: Path, relpaths: tuple[str, ...]) -> Path:
    for relpath in relpaths:
        path = workspace_root / relpath
        if path.exists():
            return path
    return require_file(workspace_root, relpaths[0])


def _corpus_rows(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    rows = corpus.get("artifacts")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    rows = corpus.get("artifact_records")
    if isinstance(rows, list):
        return [
            row
            for row in rows
            if isinstance(row, dict) and bool(row.get("indexable_text"))
        ]
    return []


def _transcription_rows(transcription: dict[str, Any]) -> list[dict[str, Any]]:
    rows = transcription.get("artifacts")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    result: list[dict[str, Any]] = []
    lanes = transcription.get("lanes")
    if isinstance(lanes, dict):
        for lane in lanes.values():
            if not isinstance(lane, dict):
                continue
            lane_rows = lane.get("records")
            if isinstance(lane_rows, list):
                result.extend(row for row in lane_rows if isinstance(row, dict))
    return result


def _processing_status_for_input(
    corpus_row: dict[str, Any],
    receipt_row: dict[str, Any],
    *,
    kind: str,
) -> str:
    status = str(corpus_row.get("processing_status") or receipt_row.get("status") or "").strip()
    if status and status != "succeeded":
        return status
    if kind == _TRANSCRIPT_KIND:
        return "transcribed"
    if kind == _OCR_JSON_KIND:
        return "ocr_done"
    if kind == _PAGE_IMAGE_KIND:
        return "image_artifact_indexed"
    if kind == _PAPER_KIND:
        return "text_extracted"
    return "local_text_extracted" if corpus_row.get("indexable_text") else status


def _input_metadata(corpus_row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if isinstance(corpus_row.get("metadata"), dict):
        metadata.update(corpus_row["metadata"])
    for key in (
        "source_family_id",
        "retention_class",
        "storage_lane",
        "indexed_text_kind",
        "source_sha256",
        "text_sha256",
        "indexed_text_sha256",
        "indexed_text_path",
        "transcript_json_path",
        "annotated_transcript_json_path",
        "transcript_path",
        "text_path",
    ):
        value = corpus_row.get(key)
        if value not in (None, ""):
            metadata.setdefault(key, value)
    return metadata


def iterate_extraction_inputs(workspace_root: Path) -> Iterator[ExtractionInput]:
    """Yield one :class:`ExtractionInput` per upstream artifact.

    Joins three upstream sources:

    * ``v15_corpus_inventory.json`` or ``corpus_inventory.json`` — canonical
      processing-status truth for completed media artifacts and derived text.
    * ``v15_transcription_ocr_receipts.json`` or
      ``transcription_ocr_inventory.json`` — provides text/transcript paths for
      paper PDFs and speech artifacts when the corpus row does not already carry
      them. Optional: absent for workspaces whose corpus inventory already
      embeds transcript/OCR paths directly (an empty payload is used instead).
    * ``v15_download_receipts.json`` — covers selected paper PDFs that have
      local text but were not surfaced through the transcription receipts.

    Paths that escape ``workspace_root`` are skipped with a warning rather than
    raised — the row is still surfaced with ``text_content=None`` so the
    classifier can record it as ``empty_or_too_short``.
    """
    workspace_root = workspace_root.resolve()
    # The transcription-receipts sidecar only supplements the corpus
    # inventory for rows that do not already carry their own transcript/OCR
    # paths -- every row below already defaults its receipt_row to {} when no
    # matching receipt entry exists, so a workspace whose corpus inventory
    # already embeds every needed transcript_path/ocr_path directly (e.g. a
    # conference's own Stage-6 corpus-inventory-index output) may never
    # produce either legacy receipts filename at all. That must not hard-block
    # extraction: treat this source as optional, like v15_download_receipts.json
    # below, rather than require_file-ing it.
    transcription_receipts_path: Path | None = None
    for candidate_relpath in (
        "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
        "storage/transcription-ocr/transcription_ocr_inventory.json",
    ):
        candidate_path = workspace_root / candidate_relpath
        if candidate_path.exists():
            transcription_receipts_path = candidate_path
            break
    corpus_inventory_path = _first_existing_workspace_file(
        workspace_root,
        (
            "storage/corpus/v15_corpus_inventory.json",
            "storage/corpus/corpus_inventory.json",
        ),
    )
    transcription = (
        load_json(transcription_receipts_path)
        if transcription_receipts_path is not None
        else {}
    )
    corpus = load_json(corpus_inventory_path)

    receipt_index: dict[str, dict[str, Any]] = {}
    for row in _transcription_rows(transcription):
        artifact_id = str(row.get("artifact_id") or "").strip()
        if not artifact_id:
            continue
        # First-write-wins: keep the first row per artifact_id so duplicates
        # in the receipts file (currently 444 of them) do not double-count.
        receipt_index.setdefault(artifact_id, row)

    seen: set[str] = set()
    seen_paper_ids: set[str] = set()

    # Phase 1: iterate the corpus inventory — the canonical processing-status
    # source of truth.
    for corpus_row in _corpus_rows(corpus):
        artifact_id = str(corpus_row.get("artifact_id") or "").strip()
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)

        receipt_row = receipt_index.get(artifact_id, {})
        kind = _infer_kind(corpus_row, receipt_row)
        text_relpath = _select_text_relpath(corpus_row, receipt_row)

        resolved_path: Path | None = None
        if text_relpath:
            try:
                resolved_path = _resolve_path_inside_workspace(
                    workspace_root, text_relpath, label="extraction-input"
                )
            except ValueError:
                LOG.warning(
                    "extraction.path_escape artifact_id=%s relpath=%s",
                    artifact_id,
                    text_relpath,
                )
                resolved_path = None
                text_relpath = None

        text_content: str | None = None
        if resolved_path is not None:
            if kind == _OCR_JSON_KIND or str(text_relpath or "").lower().endswith(".json"):
                text_content = _safe_read_ocr_text(resolved_path)
                if text_content is None and kind == _TRANSCRIPT_KIND:
                    transcript_relpath = _transcript_path_from_json_sidecar(resolved_path)
                    if transcript_relpath:
                        try:
                            transcript_path = _resolve_path_inside_workspace(
                                workspace_root,
                                transcript_relpath,
                                label="extraction-input-transcript-sidecar",
                            )
                        except ValueError:
                            LOG.warning(
                                "extraction.sidecar_path_escape artifact_id=%s relpath=%s",
                                artifact_id,
                                transcript_relpath,
                            )
                        else:
                            transcript_text = _safe_read_text(transcript_path)
                            if transcript_text is not None:
                                text_relpath = transcript_relpath
                                resolved_path = transcript_path
                                text_content = transcript_text
            else:
                text_content = _safe_read_text(resolved_path)

        digest = _digest_for_row(corpus_row, resolved_path)
        processing_status = _processing_status_for_input(
            corpus_row,
            receipt_row,
            kind=kind,
        )
        topic_hint = (
            corpus_row.get("topic_id")
            if isinstance(corpus_row.get("topic_id"), str)
            else None
        )
        source_path_raw = (
            corpus_row.get("local_path")
            or corpus_row.get("source_path")
            or receipt_row.get("source_path")
            or text_relpath
            or ""
        )

        LOG.info(
            "extraction.input artifact_id=%s kind=%s digest=%s status=%s",
            artifact_id,
            kind,
            digest[:12] if digest else "",
            processing_status,
        )

        if kind == _PAPER_KIND:
            paper_id = _paper_id_from_artifact(artifact_id, source_path=str(source_path_raw))
            if paper_id:
                seen_paper_ids.add(paper_id)

        yield ExtractionInput(
            artifact_id=artifact_id,
            kind=kind,
            text_path=text_relpath,
            digest=digest,
            status=processing_status,
            source_path=str(source_path_raw),
            topic_hint=topic_hint,
            text_content=text_content,
            metadata=_input_metadata(corpus_row),
        )

    # Phase 2: paper rows that are present in transcription receipts but not in
    # the corpus inventory (e.g., paper PDFs in `text_extracted` state). Stale
    # media rows in `blocked_pending_media_transcription_ocr` status are
    # filtered out — they were superseded by re-keyed corpus inventory entries
    # post-transcription.
    for artifact_id, receipt_row in receipt_index.items():
        if artifact_id in seen:
            continue
        status = str(receipt_row.get("status") or "")
        if status != "text_extracted":
            continue
        seen.add(artifact_id)
        kind = _infer_kind({}, receipt_row)
        text_relpath = _select_text_relpath({}, receipt_row)

        resolved_path: Path | None = None
        if text_relpath:
            try:
                resolved_path = _resolve_path_inside_workspace(
                    workspace_root, text_relpath, label="extraction-input"
                )
            except ValueError:
                LOG.warning(
                    "extraction.path_escape artifact_id=%s relpath=%s",
                    artifact_id,
                    text_relpath,
                )
                resolved_path = None
                text_relpath = None

        text_content: str | None = None
        if resolved_path is not None:
            if kind == _OCR_JSON_KIND or str(text_relpath or "").lower().endswith(".json"):
                text_content = _safe_read_ocr_text(resolved_path)
                if text_content is None and kind == _TRANSCRIPT_KIND:
                    transcript_relpath = _transcript_path_from_json_sidecar(resolved_path)
                    if transcript_relpath:
                        try:
                            transcript_path = _resolve_path_inside_workspace(
                                workspace_root,
                                transcript_relpath,
                                label="extraction-input-transcript-sidecar",
                            )
                        except ValueError:
                            LOG.warning(
                                "extraction.sidecar_path_escape artifact_id=%s relpath=%s",
                                artifact_id,
                                transcript_relpath,
                            )
                        else:
                            transcript_text = _safe_read_text(transcript_path)
                            if transcript_text is not None:
                                text_relpath = transcript_relpath
                                resolved_path = transcript_path
                                text_content = transcript_text
            else:
                text_content = _safe_read_text(resolved_path)

        digest = _digest_for_row({}, resolved_path)
        processing_status = str(receipt_row.get("status") or "")
        source_path_raw = str(receipt_row.get("source_path") or text_relpath or "")
        if kind == _PAPER_KIND:
            paper_id = _paper_id_from_artifact(artifact_id, source_path=source_path_raw)
            if paper_id:
                seen_paper_ids.add(paper_id)

        yield ExtractionInput(
            artifact_id=artifact_id,
            kind=kind,
            text_path=text_relpath,
            digest=digest,
            status=processing_status,
            source_path=source_path_raw,
            topic_hint=None,
            text_content=text_content,
        )

    # Phase 3: selected paper PDFs from download receipts that did not appear
    # in either the corpus inventory or the transcription receipts. Any paper
    # already keyed as `openreview:<id>` (or any id ending in `:<id>`) in the
    # earlier phases is treated as already-seen.
    download_receipts_path = (
        workspace_root / "storage" / "acquisition" / "v15_download_receipts.json"
    )
    if download_receipts_path.exists():
        download = load_json(download_receipts_path)
        for art in download.get("local_artifacts", []) or []:
            if not isinstance(art, dict):
                continue
            paper_id = str(art.get("paper_id") or "").strip()
            if not paper_id:
                continue
            artifact_id = f"paper:{paper_id}"
            if artifact_id in seen:
                continue
            # Detect already-covered paper rows under different id schemes.
            if any(
                existing.endswith(f":{paper_id}") or existing == f"openreview:{paper_id}"
                for existing in seen
            ):
                continue
            seen.add(artifact_id)
            seen_paper_ids.add(paper_id)
            text_relpath = f"storage/transcription-ocr/text-extraction/openreview/{paper_id}.txt"
            try:
                resolved_path = _resolve_path_inside_workspace(
                    workspace_root, text_relpath, label="extraction-input"
                )
            except ValueError:
                LOG.warning(
                    "extraction.path_escape artifact_id=%s relpath=%s",
                    artifact_id,
                    text_relpath,
                )
                continue
            text_content = _safe_read_text(resolved_path) if resolved_path else None
            digest = (
                str(art.get("sha256") or "")
                or (sha256_file(resolved_path) if resolved_path.exists() else "")
            )
            yield ExtractionInput(
                artifact_id=artifact_id,
                kind=_PAPER_KIND,
                text_path=text_relpath,
                digest=digest,
                status="text_extracted",
                source_path=str(art.get("local_path") or ""),
                topic_hint=(
                    art["topic_ids"][0]
                    if isinstance(art.get("topic_ids"), list) and art["topic_ids"]
                    else None
                ),
                text_content=text_content,
            )

    # Phase 4: OpenReview metadata fallback for fetched papers whose PDFs were
    # not selected/downloaded by the legacy v1.5 4-papers-per-topic corpus. This
    # gives downstream paper-store GPT-Researcher broad recall without requiring
    # every accepted paper PDF to be downloaded and text-extracted first.
    openreview_path = (
        workspace_root / "storage" / "acquisition" / "v15_openreview_papers.json"
    )
    if openreview_path.exists():
        openreview = load_json(openreview_path)
        taxonomy_topic_ids = tuple(family.id for family in load_topic_families(workspace_root))
        for paper in openreview.get("papers", []) or []:
            if not isinstance(paper, dict):
                continue
            paper_id = str(paper.get("id") or paper.get("forum") or "").strip()
            if not paper_id or paper_id in seen_paper_ids:
                continue
            artifact_id = f"paper_metadata_fallback:{paper_id}"
            if artifact_id in seen:
                continue
            text_content = _openreview_metadata_text(paper)
            if not text_content:
                continue
            seen.add(artifact_id)
            seen_paper_ids.add(paper_id)
            yield ExtractionInput(
                artifact_id=artifact_id,
                kind=_PAPER_METADATA_KIND,
                text_path="storage/acquisition/v15_openreview_papers.json",
                digest=_metadata_digest_for_paper(paper),
                status="text_extracted",
                source_path="storage/acquisition/v15_openreview_papers.json",
                topic_hint=None,
                text_content=text_content,
                metadata={
                    "source_family": _PAPER_METADATA_KIND,
                    "paper_id": paper_id,
                    "paper_title": str(paper.get("title") or "").strip(),
                    "abstract": str(paper.get("abstract") or "").strip(),
                    "primary_area": str(paper.get("primary_area") or "").strip(),
                    "venue": str(paper.get("venue") or "").strip(),
                    "forum_url": str(paper.get("forum_url") or "").strip(),
                    "pdf_url": str(paper.get("pdf_url") or "").strip(),
                    "authors": paper.get("authors") if isinstance(paper.get("authors"), list) else [],
                    "keywords": paper.get("keywords") if isinstance(paper.get("keywords"), list) else [],
                    "topic_ids": list(taxonomy_topic_ids),
                    "metadata_fallback_reason": "openreview_record_not_selected_for_pdf_text_extraction",
                },
            )


def existing_digest_matches(record_path: Path, digest: str) -> bool:
    """Return True when the persisted record's digest equals ``digest``."""
    if not record_path.exists() or not digest:
        return False
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(payload.get("source_artifact_digest") or "") == digest


def _record_path(workspace_root: Path, run_id: str, artifact_id: str) -> Path:
    return (
        workspace_root
        / "storage"
        / "extractions"
        / run_id
        / "by-artifact"
        / f"{safe_artifact_filename(artifact_id)}.json"
    )


def walk_and_extract(
    workspace_root: Path,
    *,
    run_id: str,
    force_rebuild: bool = False,
) -> WalkerResult:
    """Run the walker → classifier → record-write loop and aggregate by topic."""
    workspace_root = workspace_root.resolve()
    extractions_root = workspace_root / "storage" / "extractions" / run_id
    by_artifact_dir = extractions_root / "by-artifact"
    by_topic_dir = extractions_root / "by-topic"
    by_artifact_dir.mkdir(parents=True, exist_ok=True)
    by_topic_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_topic_families(workspace_root)
    result = WalkerResult()
    by_kind: dict[str, int] = {}

    current_record_names: set[str] = set()
    for input_ in iterate_extraction_inputs(workspace_root):
        result.expected_count += 1
        by_kind[input_.kind] = by_kind.get(input_.kind, 0) + 1

        record_path = _record_path(workspace_root, run_id, input_.artifact_id)
        current_record_names.add(record_path.name)

        if not force_rebuild and existing_digest_matches(record_path, input_.digest):
            result.resumed_count += 1
            result.skipped_count += 1
            continue

        # Only treat as a digest change when we have a non-empty input digest
        # AND a persisted record whose digest is also non-empty and disagrees.
        # An empty input digest cannot prove upstream change, so it must not
        # inflate ``rewritten_due_to_digest_change`` on idempotent resumes.
        digest_changed = (
            record_path.exists()
            and bool(input_.digest)
            and not existing_digest_matches(record_path, input_.digest)
        )
        if digest_changed and force_rebuild is False:
            # Stale record — remove before re-classifying so coverage doesn't
            # double-count.
            try:
                record_path.unlink()
            except OSError:
                pass
            result.rewritten_count += 1

        record = classify_artifact(input_, taxonomy, run_id=run_id)
        write_artifact_record(record_path, record)
        result.processed_count += 1
        result.extracted_count += 1

        if record.skipped_reason == "upstream_blocked":
            result.by_class_counts["blocked"] = (
                result.by_class_counts.get("blocked", 0) + 1
            )
        elif record.skipped_reason in {
            "empty_or_too_short",
            "corrupt",
            "language_unsupported",
            "empty",
        }:
            result.by_class_counts["skipped_unusable"] = (
                result.by_class_counts.get("skipped_unusable", 0) + 1
            )
        elif record.topic_ids:
            result.by_class_counts["covered"] = (
                result.by_class_counts.get("covered", 0) + 1
            )
        else:
            result.by_class_counts["unclassifiable"] = (
                result.by_class_counts.get("unclassifiable", 0) + 1
            )

    _prune_obsolete_records(
        extractions_root, by_artifact_dir, by_topic_dir, current_record_names, result
    )
    topic_counts = aggregate_by_topic(by_artifact_dir, by_topic_dir)
    _prune_obsolete_topic_files(extractions_root, by_topic_dir, set(topic_counts), result)
    result.by_kind_counts = by_kind
    return result


def _pruned_root(extractions_root: Path, result: WalkerResult) -> Path:
    if result.pruned_dir is None:
        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result.pruned_dir = str(extractions_root / "pruned" / stamp)
    return Path(result.pruned_dir)


def _prune_obsolete_records(
    extractions_root: Path,
    by_artifact_dir: Path,
    by_topic_dir: Path,
    current_record_names: set[str],
    result: WalkerResult,
) -> None:
    """Move aside per-artifact records whose artifact is no longer an input.

    The corpus can rename or drop artifacts between runs (for example when
    recordings are split into talks or ids change). Their old records would
    otherwise stay on disk forever and be aggregated into the per-topic index
    -- with whatever topic taxonomy they were classified under.
    """

    if not by_artifact_dir.exists() or not current_record_names:
        return
    for path in sorted(by_artifact_dir.glob("*.json")):
        if path.name in current_record_names:
            continue
        target = _pruned_root(extractions_root, result) / "by-artifact" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        result.pruned_artifact_count += 1


def _prune_obsolete_topic_files(
    extractions_root: Path, by_topic_dir: Path, current_topics: set[str], result: WalkerResult
) -> None:
    """Move aside per-topic files for topics no current record carries."""

    if not by_topic_dir.exists():
        return
    current_names = {f"{safe_artifact_filename(topic)}.json" for topic in current_topics}
    for path in sorted(by_topic_dir.glob("*.json")):
        if path.name in current_names:
            continue
        target = _pruned_root(extractions_root, result) / "by-topic" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        result.pruned_topic_count += 1


__all__ = [
    "ExtractionInput",
    "WalkerResult",
    "existing_digest_matches",
    "iterate_extraction_inputs",
    "walk_and_extract",
]
