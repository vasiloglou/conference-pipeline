"""Assemble a conference release inventory: storage lanes, provenance, digests,
and review status (Phase 11 / PROD-13).

Conference-general. The caller supplies the workspace and run id; everything
else is read from that workspace's own manifests and receipts, so this module
carries no conference-specific literals.

What a release inventory is, and is not. It is a record of what a completed
production run produced, where each artifact lives, what produced it, its
digest, and whether it passed its own review. It is NOT a publication: nothing
here uploads or exposes anything, and publication stays blocked unless a
target AND an approval are both supplied by the operator.

Digest policy. Deliverables are hashed individually; bulk lanes (the download
corpus, caches) are accounted at lane level by file count and byte size. A
40 GB corpus is not re-hashed to describe a deck, and pretending otherwise
would make the stage unrunnable rather than more honest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RELEASE_INVENTORY_SCHEMA_VERSION = 1

# Evidence the roadmap classifies as historical-or-blocker only: it can never
# satisfy release completion.
#
# These are checked as receipt FIELDS, never as free text anywhere in the
# payload. A receipt legitimately records ``"dry_run": false``, and CVPR 2026
# has a paper called "Scaffold: Elastic ..." whose title appears verbatim in
# the source-discovery receipt -- a substring scan blocks the release on a
# paper title, which is worse than not checking at all.
REJECTED_MODE_VALUES = frozenset({"dry-run", "dry_run", "dryrun", "mock", "scaffold", "readiness"})
# Flags that disqualify the receipt when TRUE (a False value is a receipt
# affirming it did real work, which is the opposite of a problem).
REJECTED_TRUE_FLAGS: tuple[str, ...] = (
    "dry_run",
    "mock",
    "mocked",
    "scaffold",
    "scaffolded",
    "readiness_only",
    "metadata_only",
    "needs_live_proof",
)

# Lanes whose contents are accounted rather than hashed file by file.
BULK_STORAGE_CLASSES = frozenset({"cache"})


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(root: Path) -> tuple[str, int, int]:
    """Deterministic digest over a directory's (relative path, sha256) pairs.

    Returns (digest, file_count, byte_count). Order-independent of the
    filesystem's own iteration order, so the same tree always digests the same.
    """

    rows: list[tuple[str, str]] = []
    total = 0
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item.is_symlink():
            continue
        rows.append((item.relative_to(root).as_posix(), sha256_file(item)))
        total += item.stat().st_size
    outer = hashlib.sha256()
    for name, digest in sorted(rows):
        outer.update(f"{name}:{digest}\n".encode())
    return outer.hexdigest(), len(rows), total


def directory_size(root: Path) -> tuple[int, int]:
    """(file_count, byte_count) without hashing -- for bulk lanes."""

    count = 0
    total = 0
    for item in root.rglob("*"):
        if item.is_file() and not item.is_symlink():
            count += 1
            total += item.stat().st_size
    return count, total


def rejected_evidence_markers(payload: Any) -> list[str]:
    """Which rejected-evidence markers a receipt carries, by field.

    Walks the payload structurally: a ``mode`` naming a rejected value, or any
    of the disqualifying flags set to a truthy value. Free text is never
    matched -- see the constants above for why.
    """

    markers: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            mode = node.get("mode")
            if isinstance(mode, str) and mode.strip().lower() in REJECTED_MODE_VALUES:
                markers.append(f"mode={mode.strip().lower()}")
            for flag in REJECTED_TRUE_FLAGS:
                if node.get(flag) is True:
                    markers.append(f"{flag}=true")
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(payload)
    return sorted(set(markers))


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def storage_lanes(workspace: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Group the storage manifest's artifact families by storage class."""

    families = manifest.get("artifact_families")
    if not isinstance(families, list):
        return {"error": "storage_manifest.artifact_families missing or not a list"}

    lanes: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for family in families:
        if not isinstance(family, Mapping):
            continue
        storage_class = str(family.get("storage_class") or "unclassified")
        path_value = str(family.get("path") or "")
        target = workspace / path_value if path_value else None
        present = bool(target and target.exists())
        row: dict[str, Any] = {
            "id": str(family.get("id") or ""),
            "path": path_value,
            "storage_class": storage_class,
            "share_policy": str(family.get("share_policy") or ""),
            "producer": str(family.get("producer") or ""),
            "rebuildable": bool(family.get("rebuildable")),
            "present": present,
        }
        if present and target is not None:
            if target.is_dir():
                count, size = directory_size(target)
                row.update({"kind": "directory", "file_count": count, "size_bytes": size})
            else:
                row.update(
                    {"kind": "file", "file_count": 1, "size_bytes": target.stat().st_size}
                )
        rows.append(row)
        lane = lanes.setdefault(
            storage_class,
            {"family_count": 0, "present_count": 0, "file_count": 0, "size_bytes": 0},
        )
        lane["family_count"] += 1
        lane["present_count"] += int(present)
        lane["file_count"] += int(row.get("file_count") or 0)
        lane["size_bytes"] += int(row.get("size_bytes") or 0)
    return {"lanes": lanes, "families": rows}


def digest_deliverables(
    workspace: Path, deliverables: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    """Hash each named deliverable, recording absence rather than skipping it."""

    rows: list[dict[str, Any]] = []
    for item in deliverables:
        relative = str(item.get("path") or "")
        target = workspace / relative
        row: dict[str, Any] = {
            "id": str(item.get("id") or relative),
            "path": relative,
            "kind": str(item.get("kind") or ""),
            "produced_by": str(item.get("produced_by") or ""),
            "present": target.exists(),
        }
        if target.is_file():
            row.update(
                {
                    "sha256": sha256_file(target),
                    "size_bytes": target.stat().st_size,
                    "entry_kind": "file",
                }
            )
        elif target.is_dir():
            digest, count, size = directory_digest(target)
            row.update(
                {
                    "sha256": digest,
                    "file_count": count,
                    "size_bytes": size,
                    "entry_kind": "directory",
                }
            )
        rows.append(row)
    return rows


def deck_review_status(
    workspace: Path,
    decks_dir: str,
    deck_ids: Sequence[str],
    *,
    audio_dir: str = "audio",
) -> list[dict[str, Any]]:
    """Per-deck review evidence: blueprint approval, figure QC, video sync, NotebookLM podcast."""

    rows: list[dict[str, Any]] = []
    for deck_id in deck_ids:
        deck_dir = workspace / decks_dir / deck_id
        podcast_dir = workspace / audio_dir / deck_id / "notebooklm_podcast"
        review = _load_json(deck_dir / "blueprint_review.json")
        qc = _load_json(deck_dir / "assets" / "generated" / "visual_llm_qc_report.json")
        svg = _load_json(deck_dir / "figure_svg_validation_report.json")
        row: dict[str, Any] = {
            "deck_id": deck_id,
            "blueprint_review_status": (review or {}).get("status"),
            "llm_judge_artifact": (deck_dir / "blueprint_llm_judge.md").is_file(),
            "figure_svg_validation": (svg or {}).get("status"),
            "presentation_pdf": (deck_dir / "presentation.pdf").is_file(),
        }
        if isinstance(qc, Mapping):
            slides = qc.get("slides")
            if isinstance(slides, list):
                counts: dict[str, int] = {}
                for slide in slides:
                    if isinstance(slide, Mapping):
                        key = str(slide.get("status") or "unknown")
                        counts[key] = counts.get(key, 0) + 1
                row["visual_qc"] = counts
        for style in ("single_speaker", "notebooklm"):
            sync = _load_json(deck_dir / f"video_sync_validation_{style}.json")
            row[f"video_sync_{style}"] = (sync or {}).get("status")
            row[f"video_{style}"] = (
                deck_dir
                / ("presentation_video.mp4" if style == "single_speaker" else "presentation_video_notebooklm.mp4")
            ).is_file()
        podcast_manifest = _load_json(podcast_dir / "notebook_manifest.json")
        row["podcast_notebooklm"] = (podcast_dir / "podcast.m4a").is_file()
        row["podcast_notebooklm_manifest_status"] = (
            ((podcast_manifest or {}).get("generation") or {}).get("status")
            if isinstance(podcast_manifest, Mapping)
            else None
        )
        row["podcast_notebooklm_prompt_variant"] = (
            ((podcast_manifest or {}).get("prompt") or {}).get("variant_used")
            if isinstance(podcast_manifest, Mapping)
            else None
        )
        rows.append(row)
    return rows


def publication_status(
    *, target: str | None, approved_by: str | None
) -> dict[str, Any]:
    """Publication is blocked unless BOTH a target and an approval are given.

    Done gate 2. Assembling a release bundle is not permission to publish it,
    and an inventory that quietly counted as publication approval would make
    the gate meaningless.
    """

    target_value = (target or "").strip()
    approver = (approved_by or "").strip()
    if not target_value and not approver:
        reason = "no publication target and no operator approval supplied"
    elif not target_value:
        reason = "publication approval supplied without a target"
    elif not approver:
        reason = "publication target supplied without operator approval"
    else:
        reason = ""
    return {
        "blocked": bool(reason),
        "target": target_value or None,
        "approved_by": approver or None,
        "reason": reason or None,
    }


def build_release_inventory(
    *,
    workspace: Path,
    run_id: str,
    conference_name: str,
    stage_receipts: Mapping[str, Path],
    deliverables: Sequence[Mapping[str, str]],
    decks_dir: str,
    deck_ids: Sequence[str],
    publish_target: str | None = None,
    publish_approved_by: str | None = None,
) -> dict[str, Any]:
    """Assemble the full release inventory payload."""

    manifest_path = workspace / "storage" / "storage_manifest.json"
    manifest = _load_json(manifest_path) or {}
    storage = storage_lanes(workspace, manifest)
    if manifest_path.is_file():
        # The manifest is a workspace CONTRACT, read by every stage and
        # produced by none of them, so it is recorded here -- hashed, like any
        # other input -- rather than listed as a deliverable that would then
        # need a producing-stage receipt it can never have. Stage 12 blocked
        # on exactly that: a deliverable attributed to a stage that did not
        # exist.
        storage["manifest_path"] = "storage/storage_manifest.json"
        storage["manifest_sha256"] = sha256_file(manifest_path)
    provenance: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for stage_id, receipt_path in stage_receipts.items():
        payload = _load_json(receipt_path)
        entry: dict[str, Any] = {
            "stage_id": stage_id,
            "receipt_path": str(receipt_path),
            "present": receipt_path.is_file(),
            "status": (payload or {}).get("status") if isinstance(payload, Mapping) else None,
        }
        if receipt_path.is_file():
            entry["sha256"] = sha256_file(receipt_path)
        markers = rejected_evidence_markers(payload) if payload is not None else []
        if markers:
            # Recorded, never silently dropped: an operator reading the
            # inventory can see exactly which receipt carries which marker.
            entry["rejected_evidence_markers"] = markers
            rejected.append({"stage_id": stage_id, "markers": markers})
        provenance.append(entry)

    return {
        "schema_version": RELEASE_INVENTORY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "workspace": str(workspace),
        "conference_name": conference_name,
        "run_id": run_id,
        "storage": storage,
        "provenance": provenance,
        "deliverables": digest_deliverables(workspace, deliverables),
        "review_status": deck_review_status(workspace, decks_dir, deck_ids),
        "publication": publication_status(
            target=publish_target, approved_by=publish_approved_by
        ),
        "rejected_evidence": rejected,
    }


def release_inventory_blockers(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fail-closed checks over an assembled inventory (done gates 1 and 3)."""

    blockers: list[dict[str, Any]] = []
    storage = inventory.get("storage")
    if not isinstance(storage, Mapping) or not storage.get("lanes"):
        blockers.append(
            {
                "code": "release_storage_lanes_missing",
                "detail": "storage_manifest.json did not yield any artifact-family lanes",
            }
        )
    missing = [
        row.get("path")
        for row in inventory.get("deliverables", [])
        if isinstance(row, Mapping) and not row.get("present")
    ]
    if missing:
        blockers.append({"code": "release_deliverable_missing", "detail": missing})
    undigested = [
        row.get("path")
        for row in inventory.get("deliverables", [])
        if isinstance(row, Mapping) and row.get("present") and not row.get("sha256")
    ]
    if undigested:
        blockers.append({"code": "release_deliverable_not_digested", "detail": undigested})
    unfinished = [
        entry.get("stage_id")
        for entry in inventory.get("provenance", [])
        if isinstance(entry, Mapping)
        and (not entry.get("present") or entry.get("status") != "succeeded")
    ]
    if unfinished:
        blockers.append({"code": "release_upstream_stage_not_succeeded", "detail": unfinished})
    if inventory.get("rejected_evidence"):
        blockers.append(
            {
                "code": "release_rejected_evidence_cited",
                "detail": inventory["rejected_evidence"],
            }
        )
    return blockers
