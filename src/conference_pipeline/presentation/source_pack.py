"""Build extraction-derived source packs for presentation decks."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conference_pipeline.workflow_runs import write_json_atomic

SOURCE_PACK_SCHEMA_VERSION = 1
COUNT_KEYS = (
    "source_claims",
    "visuals",
    "equations",
    "methods",
    "findings",
    "limitations",
)

_TOKEN_RE = re.compile(r"[A-Za-z]+-\d+|[A-Za-z0-9]{6,}")
_COUNT_PATTERNS = {
    "visuals": re.compile(r"\b(figure|fig\.|plot|chart|diagram|image|visual)\b", re.I),
    "equations": re.compile(r"\b(equation|formula|theorem|lemma|loss|objective)\b|[$=\u2211]", re.I),
    "methods": re.compile(r"\b(method|approach|algorithm|architecture|framework|model)\b", re.I),
    "findings": re.compile(r"\b(result|show|outperform|improve|achieve|finding)\w*\b", re.I),
    "limitations": re.compile(r"\b(limitation|failure|future work|however|but|caveat)\b", re.I),
}


@dataclass(frozen=True)
class SourceInput:
    kind: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SkippedRecord:
    index: int
    reason: str
    record_keys: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "reason": self.reason,
            "record_keys": self.record_keys,
        }


@dataclass(frozen=True)
class SourcePackReceipt:
    topic_id: str
    extraction_run_id: str
    generated_at: str
    inputs: list[SourceInput]
    outputs: list[str]
    counts: dict[str, int]
    skipped_records: list[SkippedRecord]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SOURCE_PACK_SCHEMA_VERSION,
            "topic_id": self.topic_id,
            "extraction_run_id": self.extraction_run_id,
            "generated_at": self.generated_at,
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": self.outputs,
            "counts": dict(self.counts),
            "skipped_records": [record.to_dict() for record in self.skipped_records],
        }


def build_deck_source_pack(
    workspace_root: Path,
    presentation_id: str,
    extraction_run_id: str,
    *,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Build deck-local source-pack JSON files from extraction records.

    All paths are resolved inside ``workspace_root``. The returned dictionary is
    the same payload written to ``source_pack_receipt.json``.
    """

    workspace = workspace_root.expanduser().resolve()
    if not workspace.exists():
        raise ValueError(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise ValueError(f"workspace must be a directory: {workspace}")

    safe_presentation_id = _require_safe_identifier(
        presentation_id,
        label="presentation_id",
    )
    safe_extraction_run_id = _require_safe_identifier(
        extraction_run_id,
        label="extraction_run_id",
    )

    topic_path = _resolve_inside(
        workspace,
        Path("storage")
        / "extractions"
        / safe_extraction_run_id
        / "by-topic"
        / f"{safe_presentation_id}.json",
        label="topic extraction",
    )
    output_root = (
        _resolve_inside(workspace, output_dir, label="output_dir")
        if output_dir is not None
        else _resolve_inside(
            workspace,
            Path("decks") / safe_presentation_id,
            label="output_dir",
        )
    )

    input_index: dict[str, SourceInput] = {}
    topic_payload = _load_json_input(
        topic_path,
        workspace=workspace,
        kind="by_topic",
        input_index=input_index,
    )
    topic_items = _topic_items(topic_payload)
    valid_items, skipped_records = _split_valid_topic_items(topic_items)

    extraction_root = _resolve_inside(
        workspace,
        Path("storage") / "extractions" / safe_extraction_run_id,
        label="extraction root",
    )
    by_artifact_records = _load_matching_by_artifact_records(
        workspace=workspace,
        extraction_root=extraction_root,
        presentation_id=safe_presentation_id,
        topic_items=valid_items,
        input_index=input_index,
    )
    review_payload = _load_matching_review(
        workspace=workspace,
        extraction_root=extraction_root,
        presentation_id=safe_presentation_id,
        input_index=input_index,
    )
    ocr_payloads = _load_matching_ocr(
        workspace=workspace,
        topic_items=valid_items,
        input_index=input_index,
    )

    counts = _build_counts(
        topic_items=valid_items,
        by_artifact_records=by_artifact_records,
        review_payload=review_payload,
        ocr_payloads=ocr_payloads,
    )

    extracted_content = {
        "schema_version": SOURCE_PACK_SCHEMA_VERSION,
        "topic_id": safe_presentation_id,
        "extraction_run_id": safe_extraction_run_id,
        "by_topic_path": _relative_path(topic_path, workspace),
        "topic": _minimal_topic_metadata(topic_payload),
        "items": valid_items,
        "by_artifact_records": by_artifact_records,
        "review": review_payload,
        "ocr_texts": ocr_payloads,
        "counts": counts,
        "skipped_records": [record.to_dict() for record in skipped_records],
    }
    source_attributions = _build_source_attributions(
        workspace=workspace,
        topic_path=topic_path,
        topic_items=valid_items,
        by_artifact_records=by_artifact_records,
        ocr_payloads=ocr_payloads,
    )
    ranked_papers = _build_ranked_papers(
        presentation_id=safe_presentation_id,
        extraction_run_id=safe_extraction_run_id,
        topic_items=valid_items,
        by_artifact_records=by_artifact_records,
    )

    outputs = {
        "extracted_content.json": extracted_content,
        "source_attributions.json": source_attributions,
        "ranked_papers.json": ranked_papers,
        "source_pack_ranked_papers.json": ranked_papers,
    }
    output_paths: list[str] = []
    for filename, payload in outputs.items():
        path = _resolve_inside(output_root, filename, label=f"output {filename}")
        write_json_atomic(path, payload)
        output_paths.append(_relative_path(path, workspace))

    receipt = SourcePackReceipt(
        topic_id=safe_presentation_id,
        extraction_run_id=safe_extraction_run_id,
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        inputs=sorted(input_index.values(), key=lambda item: item.path),
        outputs=sorted(output_paths + [_relative_path(output_root / "source_pack_receipt.json", workspace)]),
        counts=counts,
        skipped_records=skipped_records,
    )
    receipt_path = _resolve_inside(
        output_root,
        "source_pack_receipt.json",
        label="source_pack_receipt",
    )
    receipt_payload = receipt.to_dict()
    write_json_atomic(receipt_path, receipt_payload)
    return receipt_payload


def _require_safe_identifier(value: str, *, label: str) -> str:
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"{label} must be a non-empty path-safe identifier")
    return value


def _resolve_inside(root: Path, candidate: str | Path, *, label: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = root / path
    # Lexical containment (os.path.abspath normalizes ./.. WITHOUT following
    # symlinks), so a deliberate workspace-owned symlink -- for example the
    # Drive-backed storage/transcription-ocr on-demand artifact leaf -- is
    # admitted rather than rejected just because its real target resolves
    # outside the repository tree. A textually-outside absolute path or a
    # ".." escape is still rejected (normpath collapses ".." before the
    # comparison, and an absolute sibling path stays outside the root), and
    # actual file access still transparently follows any symlink components.
    resolved = Path(os.path.abspath(str(path)))
    root_abs = Path(os.path.abspath(str(root)))
    try:
        resolved.relative_to(root_abs)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def _relative_path(path: Path, workspace: Path) -> str:
    # Same lexical containment as _resolve_inside: don't follow the workspace's
    # own Drive-backed symlinks when computing a display-relative path.
    resolved = Path(os.path.abspath(str(path)))
    workspace_abs = Path(os.path.abspath(str(workspace)))
    try:
        return resolved.relative_to(workspace_abs).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _record_input(
    path: Path,
    *,
    workspace: Path,
    kind: str,
    input_index: dict[str, SourceInput],
) -> None:
    relative = _relative_path(path, workspace)
    input_index[relative] = SourceInput(
        kind=kind,
        path=relative,
        sha256=_sha256_file(path),
    )


def _load_json_input(
    path: Path,
    *,
    workspace: Path,
    kind: str,
    input_index: dict[str, SourceInput],
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required input is missing: {_relative_path(path, workspace)}")
    _record_input(path, workspace=workspace, kind=kind, input_index=input_index)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON input: {_relative_path(path, workspace)}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON input must be an object: {_relative_path(path, workspace)}")
    return loaded


def _topic_items(payload: dict[str, Any]) -> list[Any]:
    items = payload.get("items")
    if isinstance(items, list):
        return items
    if "artifact_id" in payload:
        return [payload]
    return []


def _split_valid_topic_items(items: list[Any]) -> tuple[list[dict[str, Any]], list[SkippedRecord]]:
    valid: list[dict[str, Any]] = []
    skipped: list[SkippedRecord] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            skipped.append(
                SkippedRecord(
                    index=index,
                    reason="record is not a JSON object",
                    record_keys=[],
                )
            )
            continue
        if not _record_artifact_id(item):
            skipped.append(
                SkippedRecord(
                    index=index,
                    reason="missing artifact_id",
                    record_keys=sorted(str(key) for key in item.keys()),
                )
            )
            continue
        valid.append(item)
    return valid, skipped


def _record_artifact_id(record: dict[str, Any]) -> str | None:
    artifact_id = record.get("artifact_id")
    if isinstance(artifact_id, str) and artifact_id:
        return artifact_id
    source_artifact_id = record.get("source_artifact_id")
    if isinstance(source_artifact_id, str) and source_artifact_id:
        return source_artifact_id
    return None


def _record_identifiers(record: dict[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    for key in ("artifact_id", "source_artifact_id", "source_artifact_path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            identifiers.add(value)
    return identifiers


def _artifact_tokens(records: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for record in records:
        for identifier in _record_identifiers(record):
            identifier = identifier.strip()
            if ":" in identifier:
                parts = identifier.split(":")[1:]
            else:
                parts = [identifier]
            for part in parts:
                for token in _TOKEN_RE.findall(part):
                    if len(token) >= 6 and any(char.isdigit() for char in token):
                        tokens.add(token.lower())
    return tokens


def _load_matching_by_artifact_records(
    *,
    workspace: Path,
    extraction_root: Path,
    presentation_id: str,
    topic_items: list[dict[str, Any]],
    input_index: dict[str, SourceInput],
) -> list[dict[str, Any]]:
    by_artifact_dir = _resolve_inside(extraction_root, "by-artifact", label="by-artifact")
    if not by_artifact_dir.is_dir():
        return []

    selected: dict[Path, dict[str, Any]] = {}
    for name in _by_artifact_candidate_names(topic_items):
        path = _resolve_inside(by_artifact_dir, name, label="by-artifact file")
        if not path.is_file():
            continue
        payload = _try_load_json_file(path)
        if payload is None:
            continue
        _record_input(
            path,
            workspace=workspace,
            kind="by_artifact",
            input_index=input_index,
        )
        selected[path] = {
            "path": _relative_path(path, workspace),
            "record": payload,
        }
    if selected:
        return [selected[path] for path in sorted(selected)]

    topic_identifiers: set[str] = set()
    for item in topic_items:
        topic_identifiers.update(_record_identifiers(item))

    for path in sorted(by_artifact_dir.glob("*.json")):
        path = _resolve_inside(by_artifact_dir, path.name, label="by-artifact file")
        should_include = path.name.startswith(f"{presentation_id}_")
        payload: dict[str, Any] | None = None
        if not should_include:
            payload = _try_load_json_file(path)
            should_include = bool(payload and _record_identifiers(payload) & topic_identifiers)
        if should_include:
            if payload is None:
                payload = _try_load_json_file(path)
            if payload is not None:
                _record_input(
                    path,
                    workspace=workspace,
                    kind="by_artifact",
                    input_index=input_index,
                )
                selected[path] = {
                    "path": _relative_path(path, workspace),
                    "record": payload,
                }

    return [selected[path] for path in sorted(selected)]


def _by_artifact_candidate_names(records: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for record in records:
        for identifier in (_record_artifact_id(record), record.get("source_artifact_id")):
            if not isinstance(identifier, str) or not identifier:
                continue
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", identifier).strip("_")
            if safe:
                names.add(f"{safe}.json")
    return names


def _try_load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _load_matching_review(
    *,
    workspace: Path,
    extraction_root: Path,
    presentation_id: str,
    input_index: dict[str, SourceInput],
) -> dict[str, object] | None:
    review_path = _resolve_inside(
        extraction_root,
        Path("review") / f"{presentation_id}-dereferenced.md",
        label="dereferenced review",
    )
    if not review_path.is_file():
        return None
    _record_input(
        review_path,
        workspace=workspace,
        kind="dereferenced_review",
        input_index=input_index,
    )
    text = review_path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": _relative_path(review_path, workspace),
        "text_excerpt": _excerpt(text),
    }


def _load_matching_ocr(
    *,
    workspace: Path,
    topic_items: list[dict[str, Any]],
    input_index: dict[str, SourceInput],
) -> list[dict[str, str]]:
    ocr_root = _resolve_inside(
        workspace,
        Path("storage") / "transcription-ocr",
        label="transcription OCR",
    )
    if not ocr_root.is_dir():
        return []

    tokens = _artifact_tokens(topic_items)
    if not tokens:
        return []

    matches: list[dict[str, str]] = []
    for path in sorted(ocr_root.rglob("*.txt")):
        path = _resolve_inside(ocr_root, path.relative_to(ocr_root), label="OCR text")
        relative = _relative_path(path, workspace)
        relative_lower = relative.lower()
        matching_token = _matching_path_token(relative_lower, tokens)
        if matching_token is None:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        _record_input(path, workspace=workspace, kind="ocr_text", input_index=input_index)
        matches.append(
            {
                "path": relative,
                "matched_artifact_token": matching_token,
                "text_excerpt": _excerpt(text),
            }
        )
    return matches


def _matching_path_token(relative_lower: str, tokens: set[str]) -> str | None:
    path_tokens = {
        token.lower()
        for token in _TOKEN_RE.findall(relative_lower)
        if len(token) >= 6 and any(char.isdigit() for char in token)
    }
    matches = path_tokens & tokens
    if not matches:
        return None
    return sorted(matches, key=lambda token: (-len(token), token))[0]


def _excerpt(text: str, *, limit: int = 2000) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _minimal_topic_metadata(payload: dict[str, Any]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in ("topic_id", "artifact_count", "schema_version", "generated_at"):
        if key in payload:
            metadata[key] = payload[key]
    return metadata


def _build_counts(
    *,
    topic_items: list[dict[str, Any]],
    by_artifact_records: list[dict[str, Any]],
    review_payload: dict[str, object] | None,
    ocr_payloads: list[dict[str, str]],
) -> dict[str, int]:
    text_parts = [
        json.dumps(topic_items, sort_keys=True),
        json.dumps(by_artifact_records, sort_keys=True),
    ]
    if review_payload:
        text_parts.append(str(review_payload.get("text_excerpt", "")))
    text_parts.extend(payload.get("text_excerpt", "") for payload in ocr_payloads)
    combined_text = "\n".join(text_parts)
    counts = {key: 0 for key in COUNT_KEYS}
    counts["source_claims"] = len(topic_items)
    for key, pattern in _COUNT_PATTERNS.items():
        counts[key] = len(pattern.findall(combined_text))
    return counts


def _build_source_attributions(
    *,
    workspace: Path,
    topic_path: Path,
    topic_items: list[dict[str, Any]],
    by_artifact_records: list[dict[str, Any]],
    ocr_payloads: list[dict[str, str]],
) -> dict[str, object]:
    artifact_paths: dict[str, str] = {}
    for entry in by_artifact_records:
        record = entry.get("record")
        if isinstance(record, dict):
            artifact_id = _record_artifact_id(record)
            if artifact_id:
                artifact_paths[artifact_id] = str(entry.get("path", ""))

    ocr_by_token: dict[str, list[str]] = {}
    for payload in ocr_payloads:
        ocr_by_token.setdefault(payload["matched_artifact_token"], []).append(payload["path"])

    sources: list[dict[str, object]] = []
    for item in topic_items:
        artifact_id = _record_artifact_id(item)
        tokens = _artifact_tokens([item])
        ocr_paths = sorted(
            {
                path
                for token in tokens
                for path in ocr_by_token.get(token, [])
            }
        )
        sources.append(
            {
                "artifact_id": artifact_id,
                "confidence": item.get("confidence"),
                "source_artifact_id": item.get("source_artifact_id"),
                "source_artifact_path": item.get("source_artifact_path"),
                "source_artifact_digest": item.get("source_artifact_digest"),
                "by_topic_path": _relative_path(topic_path, workspace),
                "by_artifact_path": artifact_paths.get(str(artifact_id)),
                "ocr_paths": ocr_paths,
            }
        )

    return {
        "schema_version": SOURCE_PACK_SCHEMA_VERSION,
        "sources": sources,
    }


def _build_ranked_papers(
    *,
    presentation_id: str,
    extraction_run_id: str,
    topic_items: list[dict[str, Any]],
    by_artifact_records: list[dict[str, Any]],
) -> dict[str, object]:
    by_artifact_by_id: dict[str, dict[str, Any]] = {}
    for entry in by_artifact_records:
        record = entry.get("record")
        if isinstance(record, dict):
            artifact_id = _record_artifact_id(record)
            if artifact_id:
                by_artifact_by_id[artifact_id] = record

    ranked = sorted(
        topic_items,
        key=lambda item: _confidence_score(item),
        reverse=True,
    )
    papers: list[dict[str, object]] = []
    for rank, item in enumerate(ranked, start=1):
        artifact_id = _record_artifact_id(item)
        artifact_record = by_artifact_by_id.get(str(artifact_id), {})
        papers.append(
            {
                "rank": rank,
                "artifact_id": artifact_id,
                "score": _confidence_score(item),
                "source_artifact_id": item.get("source_artifact_id"),
                "source_artifact_path": item.get("source_artifact_path"),
                "source_artifact_digest": item.get("source_artifact_digest"),
                "extraction_method": item.get("extraction_method")
                or artifact_record.get("extraction_method"),
                "quality_metadata": artifact_record.get("quality_metadata", {}),
                "skipped_reason": artifact_record.get("skipped_reason"),
            }
        )
    return {
        "schema_version": SOURCE_PACK_SCHEMA_VERSION,
        "topic_id": presentation_id,
        "extraction_run_id": extraction_run_id,
        "papers": papers,
    }


def _confidence_score(item: dict[str, Any]) -> float:
    value = item.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0
