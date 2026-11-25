"""Reusable duplicate/reference slide detection and OCR skip marking."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from conference_pipeline.slideslive_recovery import (
    atomic_write_json,
    sha256_file,
    utc_now_iso,
    workspace_relpath,
)

try:
    import imagehash
except ImportError:  # pragma: no cover - exercised when optional wheel is absent.
    class _SimpleImageHash:
        def __init__(self, bits: tuple[int, ...]):
            self.bits = bits

        def __sub__(self, other: _SimpleImageHash) -> int:
            return sum(left != right for left, right in zip(self.bits, other.bits, strict=True))

        def __str__(self) -> str:
            value = 0
            for bit in self.bits:
                value = (value << 1) | bit
            return f"{value:016x}"

    class _ImageHashFallback:
        ImageHash = _SimpleImageHash

        @staticmethod
        def phash(image: Image.Image) -> _SimpleImageHash:
            grayscale = image.convert("L").resize((8, 8))
            pixels = list(grayscale.tobytes())
            average = sum(pixels) / len(pixels)
            return _SimpleImageHash(tuple(1 if pixel >= average else 0 for pixel in pixels))

    imagehash = _ImageHashFallback()  # type: ignore[assignment]

SLIDE_IMAGE_PATTERNS = ("slide_*.png", "slide_*.jpg", "slide_*.jpeg", "slide_*.webp")
REFERENCE_DETECTION_OWNER_ID = "acquisition.slide_reference_detection"
SLIDE_CLEANUP_OWNER_ID = "acquisition.slide_cleanup"
REFERENCE_DETECTION_OWNER_REFS = (
    REFERENCE_DETECTION_OWNER_ID,
    "conference_pipeline.slide_cleanup",
    "src/conference_pipeline/slide_cleanup.py",
    "scripts/detect_reference_slides.py",
)
SLIDE_CLEANUP_OWNER_REFS = (
    SLIDE_CLEANUP_OWNER_ID,
    "conference_pipeline.slide_cleanup",
    "src/conference_pipeline/slide_cleanup.py",
    "scripts/cleanup_slides.py",
)


@dataclass(frozen=True)
class SlideDeck:
    """A deck-like directory containing slide images and optional sync data."""

    deck_id: str
    slides_dir: Path
    images_dir: Path


def compute_phash(image_path: Path) -> imagehash.ImageHash | None:
    try:
        with Image.open(image_path) as image:
            return imagehash.phash(image)
    except Exception:
        return None


def slide_images(images_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in SLIDE_IMAGE_PATTERNS:
        files.extend(images_dir.glob(pattern))
    return sorted(files)


def discover_slide_decks(
    *,
    data_dir: Path | None = None,
    slides_root: Path | None = None,
) -> list[SlideDeck]:
    """Find old NeurIPS-style and download-corpus slide decks."""

    candidates: list[Path] = []
    if slides_root is not None:
        if (slides_root / "images").is_dir():
            candidates.append(slides_root)
        candidates.extend(path for path in slides_root.rglob("*") if (path / "images").is_dir())
    elif data_dir is not None:
        candidates.extend((data_dir / "decks").glob("*/slides"))

    decks: list[SlideDeck] = []
    seen: set[Path] = set()
    for slides_dir in candidates:
        images_dir = slides_dir / "images"
        if not images_dir.is_dir() or not slide_images(images_dir):
            continue
        resolved = slides_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        parent_name = slides_dir.parent.name
        deck_id = parent_name if slides_dir.name in {"slides", "slideslive_slides"} else slides_dir.name
        decks.append(SlideDeck(deck_id=deck_id, slides_dir=slides_dir, images_dir=images_dir))
    return sorted(decks, key=lambda deck: deck.deck_id)


def detect_reference_slides(
    *,
    decks: list[SlideDeck],
    output_dir: Path,
    check_first: int = 5,
    min_frequency: float = 0.3,
    similarity_threshold: int = 8,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Detect generic slides that appear near the start of many decks."""

    hash_groups: dict[str, list[tuple[SlideDeck, Path, imagehash.ImageHash]]] = defaultdict(list)
    checked_decks = 0
    for deck in decks:
        candidates = slide_images(deck.images_dir)[:check_first]
        if not candidates:
            continue
        checked_decks += 1
        for image_path in candidates:
            phash = compute_phash(image_path)
            if phash is None:
                continue
            matched_key = None
            for key, group_items in hash_groups.items():
                if phash - group_items[0][2] <= similarity_threshold:
                    matched_key = key
                    break
            if matched_key is None:
                matched_key = str(phash)
            hash_groups[matched_key].append((deck, image_path, phash))

    min_count = max(3, int(checked_decks * min_frequency)) if checked_decks else 3
    reference_groups: list[tuple[str, list[tuple[SlideDeck, Path, imagehash.ImageHash]], float]] = []
    for hash_value, items in hash_groups.items():
        deck_ids = {deck.deck_id for deck, _, _ in items}
        if len(deck_ids) >= min_count:
            reference_groups.append((hash_value, items, len(deck_ids) / checked_decks))
    reference_groups.sort(key=lambda item: -item[2])

    references = []
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    for index, (hash_value, items, frequency) in enumerate(reference_groups, start=1):
        representative = items[0][1]
        output_name = f"reference_{index:02d}{representative.suffix.lower()}"
        output_path = output_dir / output_name
        if not dry_run:
            shutil.copyfile(representative, output_path)
        references.append(
            {
                "filename": output_name,
                "hash": hash_value,
                "frequency": frequency,
                "num_presentations": len({deck.deck_id for deck, _, _ in items}),
                "example_presentations": sorted({deck.deck_id for deck, _, _ in items})[:5],
            }
        )

    summary = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.slide_reference_detection",
        "stage_script_owner": {
            "script_id": REFERENCE_DETECTION_OWNER_ID,
            "owner_refs": list(REFERENCE_DETECTION_OWNER_REFS),
        },
        "status": "succeeded",
        "checked_deck_count": checked_decks,
        "unique_slide_pattern_count": len(hash_groups),
        "reference_slide_count": len(references),
        "check_first": check_first,
        "min_frequency": min_frequency,
        "similarity_threshold": similarity_threshold,
        "dry_run": dry_run,
        "output_dir": str(output_dir),
        "reference_slides": references,
        "ended_at": utc_now_iso(),
    }
    if not dry_run:
        atomic_write_json(output_dir / "metadata.json", summary)
    return summary


def load_reference_hashes(reference_dir: Path) -> list[imagehash.ImageHash]:
    hashes: list[imagehash.ImageHash] = []
    if not reference_dir.exists():
        return hashes
    for image_path in sorted(reference_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        phash = compute_phash(image_path)
        if phash is not None:
            hashes.append(phash)
    return hashes


def load_or_create_sync(deck: SlideDeck) -> dict[str, Any]:
    sync_path = deck.slides_dir / "sync.json"
    if sync_path.exists():
        return json.loads(sync_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "slides": [
            {"index": index, "image": image_path.name}
            for index, image_path in enumerate(slide_images(deck.images_dir), start=1)
        ],
    }


def cleanup_slides(
    *,
    decks: list[SlideDeck],
    reference_dir: Path | None = None,
    duplicate_threshold: int = 5,
    reference_threshold: int = 8,
    dry_run: bool = False,
    workspace: Path | None = None,
    download_corpus_path: Path | None = None,
) -> dict[str, Any]:
    """Mark duplicate/reference slides for OCR skipping without deleting files."""

    reference_hashes = load_reference_hashes(reference_dir) if reference_dir else []
    total_stats = {
        "presentations_processed": 0,
        "presentations_skipped": 0,
        "total_slides": 0,
        "duplicates_marked": 0,
        "reference_marked": 0,
        "already_marked": 0,
    }
    skipped_paths: dict[str, str] = {}
    deck_summaries: list[dict[str, Any]] = []
    for deck in decks:
        images = slide_images(deck.images_dir)
        if not images:
            total_stats["presentations_skipped"] += 1
            continue
        sync_data = load_or_create_sync(deck)
        slide_entries = sync_data.setdefault("slides", [])
        slide_map = {
            int(entry.get("index") or index): entry
            for index, entry in enumerate(slide_entries, start=1)
            if isinstance(entry, dict)
        }
        hashes: dict[int, imagehash.ImageHash] = {}
        image_by_index: dict[int, Path] = {}
        for index, image_path in enumerate(images, start=1):
            try:
                slide_index = int(image_path.stem.split("_")[1])
            except (IndexError, ValueError):
                slide_index = index
            phash = compute_phash(image_path)
            if phash is None:
                continue
            hashes[slide_index] = phash
            image_by_index[slide_index] = image_path
            slide_map.setdefault(slide_index, {"index": slide_index, "image": image_path.name})

        stats = {
            "deck_id": deck.deck_id,
            "total_slides": len(images),
            "duplicates_marked": 0,
            "reference_marked": 0,
            "already_marked": 0,
        }
        previous_hash: imagehash.ImageHash | None = None
        for slide_index in sorted(hashes):
            phash = hashes[slide_index]
            entry = slide_map[slide_index]
            skip_reason = None
            if entry.get("skip_ocr"):
                stats["already_marked"] += 1
                previous_hash = phash
                continue
            if previous_hash is not None and phash - previous_hash <= duplicate_threshold:
                skip_reason = "duplicate_of_previous"
                stats["duplicates_marked"] += 1
            elif reference_hashes and any(
                phash - reference_hash <= reference_threshold for reference_hash in reference_hashes
            ):
                skip_reason = "generic_conference_slide"
                stats["reference_marked"] += 1
            if skip_reason:
                entry["skip_ocr"] = True
                entry["skip_reason"] = skip_reason
                image_path = image_by_index[slide_index]
                if workspace is not None:
                    skipped_paths[workspace_relpath(workspace, image_path)] = skip_reason
                else:
                    skipped_paths[str(image_path)] = skip_reason
            previous_hash = phash

        sync_data["slides"] = [slide_map[index] for index in sorted(slide_map)]
        if not dry_run:
            atomic_write_json(deck.slides_dir / "sync.json", sync_data)
        total_stats["presentations_processed"] += 1
        total_stats["total_slides"] += stats["total_slides"]
        total_stats["duplicates_marked"] += stats["duplicates_marked"]
        total_stats["reference_marked"] += stats["reference_marked"]
        total_stats["already_marked"] += stats["already_marked"]
        deck_summaries.append(stats)

    corpus_updates = apply_download_corpus_skip_flags(
        workspace=workspace,
        download_corpus_path=download_corpus_path,
        skipped_paths=skipped_paths,
        dry_run=dry_run,
    )
    summary = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.slide_cleanup",
        "stage_script_owner": {
            "script_id": SLIDE_CLEANUP_OWNER_ID,
            "owner_refs": list(SLIDE_CLEANUP_OWNER_REFS),
        },
        "status": "succeeded",
        "reference_dir": str(reference_dir) if reference_dir else None,
        "duplicate_threshold": duplicate_threshold,
        "reference_threshold": reference_threshold,
        "dry_run": dry_run,
        "stats": total_stats,
        "download_corpus_updates": corpus_updates,
        "skipped_slide_count": len(skipped_paths),
        "deck_summaries": deck_summaries,
        "ended_at": utc_now_iso(),
    }
    return summary


def apply_download_corpus_skip_flags(
    *,
    workspace: Path | None,
    download_corpus_path: Path | None,
    skipped_paths: dict[str, str],
    dry_run: bool,
) -> dict[str, Any]:
    if workspace is None or download_corpus_path is None or not skipped_paths:
        return {"updated_artifact_count": 0, "path": str(download_corpus_path) if download_corpus_path else None}
    path = download_corpus_path if download_corpus_path.is_absolute() else workspace / download_corpus_path
    if not path.exists():
        return {"updated_artifact_count": 0, "path": workspace_relpath(workspace, path), "missing": True}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"updated_artifact_count": 0, "path": workspace_relpath(workspace, path), "invalid": True}

    updated = 0
    for family in ("direct_download_artifacts", "local_artifacts"):
        artifacts = payload.get(family)
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            local_path = str(artifact.get("local_path") or "")
            reason = skipped_paths.get(local_path)
            if not reason:
                continue
            artifact["skip_ocr"] = True
            artifact["skip_reason"] = reason
            artifact["ocr_exclusion_source"] = SLIDE_CLEANUP_OWNER_ID
            updated += 1
    if updated and not dry_run:
        payload["slide_cleanup"] = {
            "stage_script_owner": {
                "script_id": SLIDE_CLEANUP_OWNER_ID,
                "owner_refs": list(SLIDE_CLEANUP_OWNER_REFS),
            },
            "status": "succeeded",
            "skip_marked_artifact_count": updated,
            "ended_at": utc_now_iso(),
        }
        payload["updated_at"] = utc_now_iso()
        atomic_write_json(path, payload)
    return {
        "updated_artifact_count": updated,
        "path": workspace_relpath(workspace, path),
        "sha256_after": sha256_file(path) if updated and not dry_run else None,
    }
