"""Draft a conference's public export manifest from its storage manifest.

``release init`` (v6.0-Public-Release Phase 3, spec section 6.3) writes a manifest in
which nothing is published: ``families`` is empty and every storage-manifest family is
listed in ``excluded_families`` with a reason. The draft fails validation on purpose
(``families`` needs at least one entry) until the operator promotes a family by moving
its row into ``families`` and filling in the publication fields.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

DRAFT_REASON = "default: not reviewed for publication"
# The schema's own rule: an optional field is added without a version bump (IF3).
DRAFT_SCHEMA_VERSION = "1.0.0"


class ManifestDraftError(ValueError):
    """The storage manifest is missing or unusable, so no draft can be written."""


def storage_manifest_path(conference_dir: Path) -> Path:
    return conference_dir / "storage" / "storage_manifest.json"


def public_export_manifest_path(conference_dir: Path) -> Path:
    return conference_dir / "storage" / "public_export_manifest.json"


def load_storage_families(conference_dir: Path) -> list[dict[str, Any]]:
    """The storage manifest's ``artifact_families``, in file order."""
    path = storage_manifest_path(conference_dir)
    if not path.is_file():
        raise ManifestDraftError(f"no storage manifest at {path.relative_to(conference_dir)}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestDraftError("storage manifest is not readable JSON") from exc
    families = payload.get("artifact_families") if isinstance(payload, dict) else None
    if not isinstance(families, list) or not families:
        raise ManifestDraftError("storage manifest has no artifact_families")
    for family in families:
        if not isinstance(family, dict) or not isinstance(family.get("id"), str):
            raise ManifestDraftError("storage manifest family without an id")
    return families


def resolve_source_sha(project_root: Path) -> str:
    """``git rev-parse HEAD`` at the project root, or "" when it cannot be resolved."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def draft_public_export_manifest(conference_dir: Path, *, source_sha: str) -> dict[str, Any]:
    """The draft manifest: no family published, every storage family excluded."""
    excluded = [
        {
            "storage_manifest_id": family["id"],
            "path": family.get("path") or family.get("path_glob") or "",
            "storage_class": family.get("storage_class", ""),
            "share_policy": family.get("share_policy", ""),
            "reason": DRAFT_REASON,
        }
        for family in load_storage_families(conference_dir)
    ]
    return {
        "schema_version": DRAFT_SCHEMA_VERSION,
        "conference": conference_dir.name,
        "release": {"source_sha": source_sha},
        "families": [],
        "excluded_families": excluded,
    }


def refresh_public_export_manifest(conference_dir: Path, manifest: dict[str, Any]) -> int:
    """Add an ``excluded_families`` row for each storage family the manifest does not mention.

    Promoted families and existing rows are never changed and nothing is promoted (v6.0
    Phase 6, spec section 6.7). Returns the number of rows added; edits ``manifest`` in place.
    """
    mentioned = {str(f.get("storage_manifest_id")) for f in manifest.get("families") or [] if isinstance(f, dict)}
    excluded = manifest.setdefault("excluded_families", [])
    mentioned |= {str(row.get("storage_manifest_id")) for row in excluded if isinstance(row, dict)}
    added = 0
    for family in load_storage_families(conference_dir):
        if family["id"] in mentioned:
            continue
        excluded.append({
            "storage_manifest_id": family["id"],
            "path": family.get("path") or family.get("path_glob") or "",
            "storage_class": family.get("storage_class", ""),
            "share_policy": family.get("share_policy", ""),
            "reason": DRAFT_REASON,
        })
        added += 1
    return added
