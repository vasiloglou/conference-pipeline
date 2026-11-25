"""Durable record of which talk-split (annotated) transcript belongs to which recording.

Workshop talk segmentation writes an annotated transcript per multi-talk
recording. The link from recording to annotated transcript used to live only
as fields patched into the transcription-ocr inventory, so any rerun of that
stage (for example to extract newly downloaded PDFs) rebuilt the inventory and
silently dropped every link. Corpus indexing then refused the recordings with
``workshop_talk_segmentation_required`` although the splits were on disk.

This module keeps the links in their own ledger,
``storage/transcription-ocr/workshop-talks/annotation_bindings.json``, keyed by
the recording's ``source_sha256``. Each entry pins the transcript it was split
from and the annotated file's digest, so a binding attaches only while that
chain still holds:

- the recording's current transcript digest equals the pinned one (a
  re-transcription invalidates the split), and
- the annotated file exists with the pinned digest.

A binding that fails the chain is reported as a blocker code, never ignored.
Nothing here is conference specific.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEDGER_RELATIVE_PATH = Path("storage/transcription-ocr/workshop-talks/annotation_bindings.json")
LEDGER_RECEIPT_TYPE = "conference_pipeline.workshop_talk_annotation_bindings"

#: Lane-record fields a binding restores (paths are workspace-relative).
BINDING_PATH_FIELDS = (
    "annotated_transcript_json_path",
    "workshop_talks_json_path",
    "workshop_talk_discovery_receipt_path",
    "workshop_talk_segmentation_receipt_path",
)

BLOCKER_TRANSCRIPT_CHANGED = "annotation_binding_transcript_changed"
BLOCKER_ANNOTATED_MISSING = "annotation_binding_annotated_missing"
BLOCKER_ANNOTATED_DIGEST_MISMATCH = "annotation_binding_annotated_digest_mismatch"


class AnnotationBindingError(ValueError):
    """The ledger exists but cannot be trusted (unreadable or malformed)."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ledger_path(workspace: Path) -> Path:
    return Path(workspace) / LEDGER_RELATIVE_PATH


def workspace_relative(workspace: Path, value: object) -> str | None:
    """A path value as workspace-relative text; ``None`` when empty or outside."""

    text = str(value or "").strip()
    if not text:
        return None
    root = Path(os.path.abspath(str(workspace)))
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return Path(os.path.abspath(text)).relative_to(root).as_posix()
        except ValueError:
            return None
    normalized = Path(os.path.normpath(text))
    if normalized.parts and normalized.parts[0] == "..":
        return None
    return normalized.as_posix()


def load_bindings(workspace: Path) -> dict[str, dict[str, Any]]:
    """``source_sha256 -> binding``; an absent ledger means no bindings."""

    path = ledger_path(workspace)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationBindingError(f"annotation binding ledger unreadable: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("receipt_type") != LEDGER_RECEIPT_TYPE:
        raise AnnotationBindingError("annotation binding ledger has the wrong receipt_type")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise AnnotationBindingError("annotation binding ledger has no bindings map")
    return {str(key): dict(value) for key, value in bindings.items() if isinstance(value, Mapping)}


def write_bindings(workspace: Path, bindings: Mapping[str, Mapping[str, Any]], *, writer: str) -> Path:
    path = ledger_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "receipt_type": LEDGER_RECEIPT_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "writer": writer,
        "binding_count": len(bindings),
        "bindings": {key: dict(bindings[key]) for key in sorted(bindings)},
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def make_binding(
    workspace: Path,
    *,
    source_sha256: str,
    transcript_sha256: str,
    annotated_transcript_json_path: object,
    fields: Mapping[str, Any] | None = None,
    bound_by: str,
) -> dict[str, Any]:
    """Build one binding, pinning the annotated file's current digest."""

    if not source_sha256 or not transcript_sha256:
        raise AnnotationBindingError("a binding needs source_sha256 and transcript_sha256")
    annotated_rel = workspace_relative(workspace, annotated_transcript_json_path)
    if annotated_rel is None:
        raise AnnotationBindingError("annotated transcript path is empty or outside the workspace")
    annotated = Path(workspace) / annotated_rel
    if not annotated.is_file():
        raise AnnotationBindingError(f"annotated transcript missing: {annotated_rel}")
    binding: dict[str, Any] = {
        "source_sha256": source_sha256,
        "transcript_sha256": transcript_sha256,
        "annotated_transcript_json_path": annotated_rel,
        "annotated_transcript_json_sha256": _sha256_file(annotated),
        "bound_by": bound_by,
    }
    for key, value in (fields or {}).items():
        if key in BINDING_PATH_FIELDS and key != "annotated_transcript_json_path":
            rel = workspace_relative(workspace, value)
            if rel is not None:
                binding[key] = rel
        elif key == "workshop_talk_count" and value not in (None, ""):
            binding[key] = int(value)
    return binding


def upsert_bindings(workspace: Path, new: Iterable[Mapping[str, Any]], *, writer: str) -> Path:
    bindings = load_bindings(workspace)
    for binding in new:
        bindings[str(binding["source_sha256"])] = dict(binding)
    return write_bindings(workspace, bindings, writer=writer)


def binding_for_record(
    workspace: Path,
    bindings: Mapping[str, Mapping[str, Any]],
    lane_record: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Fields to restore on a transcription lane record, or a blocker code.

    ``(None, None)`` means the ledger has nothing for this recording.
    """

    binding = bindings.get(str(lane_record.get("source_sha256") or ""))
    if binding is None:
        return None, None
    if str(lane_record.get("transcript_sha256") or "") != str(binding.get("transcript_sha256") or ""):
        return None, BLOCKER_TRANSCRIPT_CHANGED
    annotated = Path(workspace) / str(binding.get("annotated_transcript_json_path") or "")
    if not annotated.is_file():
        return None, BLOCKER_ANNOTATED_MISSING
    if _sha256_file(annotated) != binding.get("annotated_transcript_json_sha256"):
        return None, BLOCKER_ANNOTATED_DIGEST_MISMATCH
    fields = {key: binding[key] for key in BINDING_PATH_FIELDS if binding.get(key)}
    if binding.get("workshop_talk_count") is not None:
        fields["workshop_talk_count"] = binding["workshop_talk_count"]
    return fields, None


def bindings_from_corpus_inventory(
    workspace: Path,
    corpus_inventory: Mapping[str, Any],
    transcription_inventory: Mapping[str, Any],
    *,
    annotated_root: str = "storage/transcription-ocr/workshop-talks/",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover bindings from an earlier corpus index (backfill).

    Takes speech records the index built from an annotated transcript under
    ``annotated_root`` and pins them to the current transcription inventory's
    transcript digest. A record is recovered only when the index's pinned
    annotated digest still matches the file and the transcript text it was
    split from is unchanged. Returns ``(bindings, skipped)``.
    """

    lane = ((transcription_inventory.get("lanes") or {}).get("session_speech_transcription") or {})
    current = {
        str(record.get("source_sha256")): record
        for record in lane.get("records") or []
        if isinstance(record, Mapping) and record.get("source_sha256")
    }
    bindings: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in corpus_inventory.get("artifact_records") or []:
        if not isinstance(record, Mapping) or record.get("indexed_text_kind") != "annotated_transcript_json":
            continue
        annotated_rel = workspace_relative(workspace, record.get("indexed_text_path"))
        if annotated_rel is None or not annotated_rel.startswith(annotated_root):
            continue
        source_sha = str(record.get("source_sha256") or "")
        lane_record = current.get(source_sha)
        reason = None
        if lane_record is None:
            reason = "recording not in the current transcription inventory"
        elif str(lane_record.get("transcript_sha256") or "") != str(record.get("text_sha256") or ""):
            reason = "transcript changed since the index was built"
        else:
            annotated = Path(workspace) / annotated_rel
            if not annotated.is_file():
                reason = "annotated transcript missing"
            elif _sha256_file(annotated) != record.get("indexed_text_sha256"):
                reason = "annotated transcript changed since the index was built"
        if reason:
            skipped.append({"source_sha256": source_sha, "annotated_transcript_json_path": annotated_rel, "reason": reason})
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        bindings.append(
            make_binding(
                workspace,
                source_sha256=source_sha,
                transcript_sha256=str(lane_record.get("transcript_sha256")),
                annotated_transcript_json_path=annotated_rel,
                fields={key: metadata.get(key) for key in (*BINDING_PATH_FIELDS, "workshop_talk_count")},
                bound_by="backfill_from_corpus_inventory",
            )
        )
    return bindings, skipped
