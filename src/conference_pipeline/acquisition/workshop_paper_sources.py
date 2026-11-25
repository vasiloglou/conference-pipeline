"""Registry of supplementary workshop-paper download receipts (conference-neutral).

Why this exists
---------------
The base workshop corpus comes from one OpenReview download receipt. On CVPR
2026 that covered 35 of ~150 scheduled workshops; the rest of the papers came
from other hosts found later (CVF Open Access proceedings, workshop websites,
arXiv). Each of those acquisitions writes its own download receipt with the
same ``output_dir`` + ``per_workshop`` layout and per-paper
``workshop_paper_receipt`` files. A corpus-inventory stage should admit all of
them without growing a hardcoded path per source, so every supplementary
source registers its receipt here and the stage loads the registry.

Layout (workspace-relative)::

    storage/workshops/workshop_paper_sources.json

Registry::

    {"schema_version": 1,
     "receipt_type": "conference_pipeline.workshop_paper_source_registry",
     "sources": [{"source_id": "cvf_openaccess",
                  "receipt_path": "storage/workshops/cvf_openaccess_download_receipt.json",
                  "receipt_sha256": "..."}]}

The registry pins each receipt's sha256: a receipt rewritten after
registration blocks admission until it is re-registered (the downloader
scripts re-register on every run that acquired papers). Sources are listed in
precedence order; each downloader already skips papers acquired by the base
OpenReview download and by the sources registered before it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_RELATIVE_PATH = Path("storage/workshops/workshop_paper_sources.json")
REGISTRY_RECEIPT_TYPE = "conference_pipeline.workshop_paper_source_registry"
DOWNLOAD_RECEIPT_TYPE = "conference_pipeline.workshop_paper_download_receipt"
# "partial" is admissible: a workshop-site source normally ends partial because
# some listed papers have no public PDF (or a host throttled a few fetches).
# Those gaps are recorded per paper in the receipt, and every admitted paper is
# digest-checked individually, so the corpus admits what was actually acquired.
ADMISSIBLE_STATUSES = frozenset({"succeeded", "partial"})
SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _blocker(code: str, **details: Any) -> dict[str, Any]:
    return {"code": code, **details}


def registry_path(workspace: Path) -> Path:
    return workspace / REGISTRY_RELATIVE_PATH


def register_source(*, workspace: Path, source_id: str, receipt_path: Path | str) -> dict[str, Any]:
    """Add or re-pin ``source_id``'s download receipt; returns the registry payload.

    ``receipt_path`` may be absolute or workspace-relative; it is stored
    workspace-relative. Only a succeeded or partial workshop-paper download
    receipt can be registered.
    """
    workspace = workspace.resolve()
    candidate = Path(receipt_path)
    # Accept an absolute path, a workspace-relative path, or a path that
    # already includes the workspace prefix (what a caller holding
    # ``<workspace>/<relative>`` naturally passes).
    options = [candidate] if candidate.is_absolute() else [workspace / candidate, Path.cwd() / candidate]
    receipt_file = next((option.resolve() for option in options if option.is_file()), options[0].resolve())
    receipt = _read_json(receipt_file)
    if receipt is None:
        raise FileNotFoundError(f"download receipt missing or invalid: {receipt_file}")
    if receipt.get("receipt_type") != DOWNLOAD_RECEIPT_TYPE or receipt.get("status") not in ADMISSIBLE_STATUSES:
        raise ValueError(
            f"only a succeeded/partial {DOWNLOAD_RECEIPT_TYPE} can be registered "
            f"(got {receipt.get('receipt_type')!r} / {receipt.get('status')!r})"
        )
    try:
        relative = receipt_file.relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"receipt is outside the workspace: {receipt_file}") from error
    entry = {
        "source_id": source_id,
        "receipt_path": relative.as_posix(),
        "receipt_sha256": _sha256_file(receipt_file),
    }
    registry = _read_json(registry_path(workspace)) or {}
    sources = [s for s in registry.get("sources", []) or [] if isinstance(s, dict)]
    index = next((i for i, s in enumerate(sources) if s.get("source_id") == source_id), None)
    if index is None:
        sources.append(entry)
    else:
        sources[index] = entry
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": REGISTRY_RECEIPT_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
    }
    path = registry_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_registered_sources(workspace: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(sources, blockers)`` for the workspace registry.

    Each source is ``{source_id, receipt_path, receipt_sha256, receipt}``. A
    missing registry means no supplementary sources (not a blocker); any
    registered receipt that is missing, drifted, or not admissible is a blocker.
    """
    workspace = workspace.resolve()
    path = registry_path(workspace)
    if not path.exists():
        return [], []
    registry = _read_json(path)
    if registry is None or registry.get("receipt_type") != REGISTRY_RECEIPT_TYPE:
        return [], [_blocker("workshop_paper_source_registry_invalid", path=REGISTRY_RELATIVE_PATH.as_posix())]
    sources: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in registry.get("sources", []) or []:
        if not isinstance(entry, Mapping) or not entry.get("source_id") or not entry.get("receipt_path"):
            blockers.append(_blocker("workshop_paper_source_entry_invalid"))
            continue
        source_id = str(entry["source_id"])
        if source_id in seen:
            blockers.append(_blocker("workshop_paper_source_duplicate", source_id=source_id))
            continue
        seen.add(source_id)
        receipt_file = (workspace / str(entry["receipt_path"])).resolve()
        try:
            receipt_file.relative_to(workspace)
        except ValueError:
            blockers.append(_blocker("workshop_paper_source_outside_workspace", source_id=source_id))
            continue
        if not receipt_file.is_file():
            blockers.append(_blocker("workshop_paper_source_receipt_missing", source_id=source_id, path=str(entry["receipt_path"])))
            continue
        actual = _sha256_file(receipt_file)
        if actual != entry.get("receipt_sha256"):
            blockers.append(
                _blocker("workshop_paper_source_receipt_digest_mismatch", source_id=source_id, path=str(entry["receipt_path"]))
            )
            continue
        receipt = _read_json(receipt_file)
        if (
            receipt is None
            or receipt.get("receipt_type") != DOWNLOAD_RECEIPT_TYPE
            or receipt.get("status") not in ADMISSIBLE_STATUSES
        ):
            blockers.append(_blocker("workshop_paper_source_receipt_not_admissible", source_id=source_id))
            continue
        sources.append(
            {"source_id": source_id, "receipt_path": str(entry["receipt_path"]), "receipt_sha256": actual, "receipt": receipt}
        )
    return sources, blockers
