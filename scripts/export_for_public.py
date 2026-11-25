#!/usr/bin/env python3
"""Build deterministic public release bundles from a public export manifest."""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import mimetypes
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.publishing.bundle_redaction import RedactionError, apply_redaction_rules  # noqa: E402
from conference_pipeline.publishing.cdn import (  # noqa: E402
    is_completed_live_upload_receipt,
    object_key_for,
)
from conference_pipeline.publishing.export_gate import (  # noqa: E402
    GateClosed,
    GateReceipt,
    _is_secret_like_path,
    run_gate,
    validate_public_export_manifest,
)
from conference_pipeline.publishing.release_layout import stamped_urls, workspace_release_settings  # noqa: E402
from conference_pipeline.publishing.receipt_paths import (  # noqa: E402
    ReceiptPathError,
    confine_workflow_receipt_path,
    write_json_atomic_secure,
)

RECEIPT_TYPE = "conference_pipeline.public_export"
ERROR_EXISTING_BUNDLE = "public export bundle already exists"
ERROR_REVIEWED_RECEIPT = "reviewed dry-run receipt does not match current export plan"
ERROR_CAUTION_ACK = "caution family acknowledgement required"
ERROR_PLACEHOLDER_CDN = "placeholder CDN base URL is not allowed for write mode"
ERROR_IMMUTABLE_BUNDLE = "public export bundle is immutable after publication"
ERROR_REPRODUCIBILITY = "reproducibility digest mismatch"

LIVE_PUBLICATION_RECEIPT_TYPES = frozenset(
    {
        "conference_pipeline.cloudflare_pages_deploy",
        "conference_pipeline.hf_dataset_push",
        "conference_pipeline.provider_success",
        "conference_pipeline.public_cdn_upload",
        "conference_pipeline.public_repo_push",
        "conference_pipeline.zenodo_deposit",
    }
)


class PublicExportBlocked(RuntimeError):
    """Raised when public export fails closed before producing a release."""

    def __init__(self, reason: str, receipt: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.receipt = receipt or {}


@dataclass(frozen=True)
class ExportConfig:
    conference_dir: Path
    export_manifest: Path
    output_dir: Path
    bundle_id: str | None
    dry_run: bool
    receipt_path: Path | None = None
    reviewed_dry_run_receipt: Path | None = None
    ack_caution_families: tuple[str, ...] = ()
    source_sha: str | None = None
    orchestrator_stage: str | None = None


@dataclass(frozen=True)
class ExportFamily:
    family_id: str
    private_path: Path
    public_path: str
    storage_manifest_id: str
    redistribution_status: str
    license: str
    provenance: str
    include_globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    large_media_policy: dict[str, Any]
    format_contract: dict[str, Any]
    cdn_base_url: str
    checksum_required: bool
    public_export_notes: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class ExportItem:
    family_id: str
    source_path: Path
    source_relative_path: Path
    release_path: Path
    delivery: str
    size_bytes: int
    sha256: str
    mime_type: str
    cdn_url: str | None = None
    cdn_object_key: str | None = None
    # The redacted bundle copy, when a family's redaction_rules changed the file (Phase 6).
    content: bytes | None = None


@dataclass(frozen=True)
class ExcludedArtifact:
    family_id: str
    path: str
    reason: str


@dataclass
class PublicationTransaction:
    bundle_root: Path
    backup_root: Path
    previous_bundle_present: bool
    staging_digest: str
    published: bool = False
    committed: bool = False
    backup_cleanup_started: bool = False
    cleanup_pending: bool = False


@dataclass(frozen=True)
class ExportPlan:
    config: ExportConfig
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    conference: str
    source_sha: str
    bundle_id: str
    families: tuple[ExportFamily, ...]
    included_families: tuple[str, ...]
    excluded_artifacts: tuple[ExcludedArtifact, ...]
    items: tuple[ExportItem, ...]
    redacted_files: tuple[dict[str, Any], ...] = ()
    binary_markers: tuple[str, ...] = ()

    @property
    def bundle_root(self) -> Path:
        return self.config.output_dir / self.bundle_id

    @property
    def cdn_items(self) -> tuple[ExportItem, ...]:
        return tuple(item for item in self.items if item.delivery == "cdn")

    @property
    def inline_items(self) -> tuple[ExportItem, ...]:
        return tuple(item for item in self.items if item.delivery == "inline")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic_secure(path, payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _portable_bundle_files(root: Path) -> list[Path]:
    if root.is_symlink():
        raise PublicExportBlocked("portable bundle root must not be a symlink")
    if not root.is_dir():
        raise PublicExportBlocked("portable bundle root must be a directory")

    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise PublicExportBlocked(f"portable bundle contains symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PublicExportBlocked(f"portable bundle contains non-regular node: {relative}")
        files.append(path)
    return files


def _tree_rows_sha256(root: Path, files: list[Path]) -> str:
    rows = [
        "\0".join(
            (
                path.relative_to(root).as_posix(),
                sha256_file(path),
                str(path.stat().st_size),
            )
        )
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix())
    ]
    return sha256_text("\n".join(rows) + ("\n" if rows else ""))


def bundle_tree_sha256(root: Path) -> str:
    """Hash sorted bundle-relative file identity rows for a portable tree."""
    return _tree_rows_sha256(root, _portable_bundle_files(root))


def tree_sha256_for(root: Path, relative_paths: list[str]) -> str:
    """``bundle_tree_sha256`` restricted to the named bundle-relative files.

    Used to re-verify the export-owned part of a bundle after the derived stages
    (index, site, metadata) have added files beside it. A named file that is missing,
    a symlink or not a regular file raises ``PublicExportBlocked``.
    """
    root = root.resolve()
    files: list[Path] = []
    for relative in relative_paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise PublicExportBlocked(f"bundle file missing or not regular: {relative}")
        ensure_under(path, root, label="bundle_path")
        files.append(path)
    return _tree_rows_sha256(root, files)


def source_git_sha(repo_root: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return "unknown"


def safe_relative_path(raw: str, *, label: str) -> Path:
    value = raw.strip()
    if not value:
        raise PublicExportBlocked(f"{label} must not be empty")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PublicExportBlocked(f"{label} must stay inside the conference workspace")
    return candidate


def ensure_under(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PublicExportBlocked(f"{label} escapes {root_resolved}") from exc
    return resolved


def ensure_workspace_source_path(path: Path, root: Path, *, label: str) -> Path:
    """Keep manifest-approved source paths lexical so workspace symlinks remain valid."""
    candidate = path.absolute()
    root_absolute = root.absolute()
    try:
        candidate.relative_to(root_absolute)
    except ValueError as exc:
        raise PublicExportBlocked(f"{label} escapes {root_absolute}") from exc
    return candidate


def normalize_release_path(public_path: str, bundle_id: str) -> Path:
    replaced = public_path.replace("{bundle}", bundle_id).strip("/")
    rel = safe_relative_path(replaced or ".", label="family.public_path")
    parts = rel.parts
    prefix = ("exports", "public-release", bundle_id)
    if len(parts) >= 3 and parts[:3] == prefix:
        rel = Path(*parts[3:]) if len(parts) > 3 else Path(".")
    return Path(".") if rel.as_posix() == "." else rel


def is_secret_like_release_path(path: Path) -> bool:
    return any(_is_secret_like_path(Path(part)) for part in path.parts if part not in {"."})


def load_export_manifest(manifest_path: Path) -> dict[str, Any]:
    validate_public_export_manifest(manifest_path)
    return read_json(manifest_path)


def _tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return ()


def parse_families(manifest: dict[str, Any]) -> tuple[ExportFamily, ...]:
    families: list[ExportFamily] = []
    for family in manifest.get("families", []):
        if not isinstance(family, dict):
            raise PublicExportBlocked("manifest family must be an object")
        families.append(
            ExportFamily(
                family_id=str(family["id"]),
                private_path=safe_relative_path(str(family["private_path"]), label="family.private_path"),
                public_path=str(family["public_path"]),
                storage_manifest_id=str(family["storage_manifest_id"]),
                redistribution_status=str(family["redistribution_status"]),
                license=str(family["license"]),
                provenance=str(family["provenance"]),
                include_globs=_tuple(family.get("include_globs")),
                exclude_globs=_tuple(family.get("exclude_globs")),
                large_media_policy=dict(family.get("large_media_policy") or {}),
                format_contract=dict(family.get("format_contract") or {}),
                cdn_base_url=str(family.get("cdn_base_url") or ""),
                checksum_required=bool(family.get("checksum_required", False)),
                public_export_notes=family.get("public_export_notes"),
                raw=family,
            )
        )
    return tuple(sorted(families, key=lambda entry: entry.family_id))


def manifest_source_sha(config: ExportConfig, manifest: dict[str, Any]) -> str:
    if config.source_sha:
        return config.source_sha
    release = manifest.get("release") or {}
    if isinstance(release, dict) and str(release.get("source_sha") or "").strip():
        return str(release["source_sha"]).strip()
    return source_git_sha()


def resolve_bundle_id(config: ExportConfig, manifest: dict[str, Any], source_sha: str) -> str:
    if config.bundle_id:
        return config.bundle_id
    release = manifest.get("release") or {}
    if isinstance(release, dict) and str(release.get("release_tag") or "").strip():
        return str(release["release_tag"]).strip()
    short_sha = source_sha[:12] if source_sha and source_sha != "unknown" else "unknown"
    return f"{manifest.get('conference', 'conference')}-{short_sha}"


def matches_any(path: Path, patterns: Sequence[str]) -> bool:
    text = path.as_posix()
    return any(fnmatch.fnmatch(text, pattern) for pattern in patterns)


def _family_source_blocker(
    family: ExportFamily,
    path: Path,
    reason: str,
) -> PublicExportBlocked:
    return PublicExportBlocked(
        reason,
        {
            "family_id": family.family_id,
            "path": path.as_posix(),
            "reason": reason,
        },
    )


def _family_selection_exclusion(
    family: ExportFamily,
    relative_path: Path,
) -> str | None:
    if family.include_globs and not matches_any(relative_path, family.include_globs):
        return "not_matched_by_include_globs"
    if family.exclude_globs and matches_any(relative_path, family.exclude_globs):
        return "matched_exclude_globs"
    return None


def iter_family_files(conference_dir: Path, family: ExportFamily) -> tuple[list[Path], list[ExcludedArtifact]]:
    source_root = ensure_workspace_source_path(conference_dir / family.private_path, conference_dir, label=family.family_id)
    if not source_root.exists():
        reason = "selected_broken_symlink" if source_root.is_symlink() else "missing_required_source"
        raise _family_source_blocker(family, family.private_path, reason)

    if source_root.is_file():
        candidates = [source_root]
    else:
        candidates = list(source_root.rglob("*"))

    included: list[Path] = []
    excluded: list[ExcludedArtifact] = []
    for candidate in sorted(candidates, key=lambda p: p.relative_to(source_root if source_root.is_dir() else source_root.parent).as_posix()):
        base = source_root if source_root.is_dir() else source_root.parent
        rel = candidate.relative_to(base)
        if candidate.is_dir() and not candidate.is_symlink():
            continue
        exclusion_reason = _family_selection_exclusion(family, rel)
        if exclusion_reason is not None:
            excluded.append(ExcludedArtifact(family.family_id, rel.as_posix(), exclusion_reason))
            continue
        if candidate.is_symlink() and not candidate.exists():
            lexical_path = family.private_path / rel
            raise _family_source_blocker(
                family,
                lexical_path,
                "selected_broken_symlink",
            )
        if not candidate.is_file():
            continue
        included.append(candidate)
    return included, excluded


def ensure_unique_release_paths(items: Iterable[ExportItem]) -> None:
    seen: dict[str, ExportItem] = {}
    for item in items:
        release_path = item.release_path.as_posix()
        previous = seen.get(release_path)
        if previous is not None:
            raise PublicExportBlocked(
                "duplicate_release_path",
                {
                    "family_id": item.family_id,
                    "conflicting_family_id": previous.family_id,
                    "path": release_path,
                    "reason": "duplicate_release_path",
                },
            )
        seen[release_path] = item


def max_inline_bytes(family: ExportFamily) -> int | None:
    max_mb = family.format_contract.get("max_inline_size_mb")
    if max_mb is None:
        max_mb = family.large_media_policy.get("max_inline_size_mb")
    if max_mb is None:
        return None
    return int(float(max_mb) * 1024 * 1024)


def choose_delivery(family: ExportFamily, source_path: Path) -> str:
    if family.large_media_policy.get("delivery") == "cdn-only":
        return "cdn"
    delivery = family.format_contract.get("delivery") or family.large_media_policy.get("delivery")
    if delivery in {"inline-if-small", "inline_if_lt_5mb_else_cdn_url"}:
        limit = max_inline_bytes(family)
        if limit is not None and source_path.stat().st_size > limit:
            return "cdn"
    return "inline"


def cdn_url_for(
    family: ExportFamily,
    bundle_id: str,
    release_base: Path,
    source_relative_path: Path,
) -> str:
    if not family.cdn_base_url:
        raise PublicExportBlocked(f"cdn_base_url required for CDN family {family.family_id}")
    base = family.cdn_base_url.rstrip("/")
    media_release_path = release_base / source_relative_path
    suffix = media_release_path.as_posix().strip("/")
    return f"{base}/{bundle_id}/{suffix}"


def validate_cdn_target(
    family: ExportFamily,
    bundle_id: str,
    media_release_path: Path,
    cdn_url: str,
) -> str:
    """Validate one immutable CDN URL and return its uploader object key."""
    base_url = family.cdn_base_url.rstrip("/")
    parsed_base = urlparse(base_url)
    parsed_url = urlparse(cdn_url)
    for label, parsed in (("cdn_base_url", parsed_base), ("cdn_url", parsed_url)):
        if parsed.scheme != "https" or not parsed.hostname:
            raise PublicExportBlocked(f"invalid {label} for CDN family {family.family_id}")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise PublicExportBlocked(f"invalid {label} for CDN family {family.family_id}")

    media_path = safe_relative_path(
        media_release_path.as_posix(),
        label=f"{family.family_id}.media_release_path",
    )
    expected_url = f"{base_url}/{bundle_id}/{media_path.as_posix().strip('/')}"
    if cdn_url != expected_url:
        raise PublicExportBlocked(f"invalid CDN target for family {family.family_id}")

    url_parts = tuple(part for part in parsed_url.path.split("/") if part)
    if url_parts.count(bundle_id) != 1:
        raise PublicExportBlocked(f"invalid CDN bundle path for family {family.family_id}")

    expected_key = "/".join(
        part
        for part in (
            parsed_base.path.strip("/"),
            bundle_id,
            media_path.as_posix().strip("/"),
        )
        if part
    )
    object_key = object_key_for(cdn_url, "")
    if object_key != expected_key:
        raise PublicExportBlocked(f"CDN object key mismatch for family {family.family_id}")
    return object_key


def build_export_plan(config: ExportConfig) -> ExportPlan:
    conference_dir = config.conference_dir.resolve()
    manifest_path = config.export_manifest.resolve()
    ensure_under(manifest_path, conference_dir, label="export_manifest")

    manifest = load_export_manifest(manifest_path)
    source_sha = manifest_source_sha(config, manifest)
    bundle_id = resolve_bundle_id(config, manifest, source_sha)
    manifest_sha = sha256_file(manifest_path)
    families = parse_families(manifest)

    items: list[ExportItem] = []
    excluded: list[ExcludedArtifact] = []
    included_families: list[str] = []
    redacted: list[dict[str, Any]] = []
    binary_markers: list[str] = []

    for family in families:
        release_base = normalize_release_path(family.public_path, bundle_id)
        if is_secret_like_release_path(release_base):
            raise PublicExportBlocked(
                "secret_like_public_path",
                {
                    "family_id": family.family_id,
                    "public_path": family.public_path,
                    "reason": "secret_like_public_path",
                },
            )
        included_families.append(family.family_id)
        files, family_excluded = iter_family_files(conference_dir, family)
        excluded.extend(family_excluded)
        source_root = ensure_workspace_source_path(conference_dir / family.private_path, conference_dir, label=family.family_id)
        base = source_root if source_root.is_dir() else source_root.parent
        for source_path in files:
            rel = source_path.relative_to(base)
            delivery = choose_delivery(family, source_path)
            mime_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
            media_release_path = release_base / rel
            cdn_url = (
                cdn_url_for(family, bundle_id, release_base, rel)
                if delivery == "cdn"
                else None
            )
            cdn_object_key = (
                validate_cdn_target(family, bundle_id, media_release_path, cdn_url)
                if cdn_url is not None
                else None
            )
            release_path = media_release_path
            if delivery == "cdn":
                release_path = release_path.with_name(f"{release_path.name}.cdn.json")
            if is_secret_like_release_path(release_path):
                raise PublicExportBlocked(
                    "secret_like_release_path",
                    {
                        "family_id": family.family_id,
                        "path": release_path.as_posix(),
                        "reason": "secret_like_release_path",
                    },
                )
            content = None
            rules = family.raw.get("redaction_rules") or []
            if delivery == "inline" and any(isinstance(rule, dict) for rule in rules):
                try:
                    outcome = apply_redaction_rules(
                        source_path.read_bytes(), rel.as_posix(), rules, workspace=conference_dir,
                        source_path=source_path,
                    )
                except RedactionError as exc:
                    raise PublicExportBlocked("redaction_rule_invalid", {"family_id": family.family_id, "reason": str(exc)}) from exc
                content = outcome.content
                if content is not None:
                    redacted.append({"path": release_path.as_posix(), "rules": list(outcome.kinds), "replacements": outcome.replacements})
                if outcome.binary_marker:
                    binary_markers.append(release_path.as_posix())
            items.append(
                ExportItem(
                    family_id=family.family_id,
                    source_path=source_path,
                    source_relative_path=rel,
                    release_path=release_path,
                    delivery=delivery,
                    size_bytes=len(content) if content is not None else source_path.stat().st_size,
                    sha256=hashlib.sha256(content).hexdigest() if content is not None else sha256_file(source_path),
                    mime_type=mime_type,
                    cdn_url=cdn_url,
                    cdn_object_key=cdn_object_key,
                    content=content,
                )
            )
            ensure_unique_release_paths(items)

    return ExportPlan(
        config=config,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        conference=str(manifest.get("conference")),
        source_sha=source_sha,
        bundle_id=bundle_id,
        families=families,
        included_families=tuple(sorted(included_families)),
        excluded_artifacts=tuple(sorted(excluded, key=lambda item: (item.family_id, item.path, item.reason))),
        items=tuple(sorted(items, key=lambda item: item.release_path.as_posix())),
        redacted_files=tuple(sorted(redacted, key=lambda row: row["path"])),
        binary_markers=tuple(sorted(binary_markers)),
    )


def gate_candidate_paths(plan: ExportPlan) -> list[Path]:
    return sorted({item.source_path for item in plan.items}, key=lambda path: path.as_posix())


def caution_family_ids(plan: ExportPlan) -> tuple[str, ...]:
    return tuple(
        family.family_id
        for family in plan.families
        if family.redistribution_status == "caution"
    )


def enforce_caution_ack(plan: ExportPlan) -> None:
    missing = sorted(set(caution_family_ids(plan)) - set(plan.config.ack_caution_families))
    if missing:
        receipt = build_receipt_payload(
            plan,
            status="blocked",
            exit_code=1,
            error=ERROR_CAUTION_ACK,
            extra={"missing_caution_acknowledgements": missing},
        )
        raise PublicExportBlocked(ERROR_CAUTION_ACK, receipt)


def run_plan_level_gate(
    plan: ExportPlan,
    *,
    receipt_dir: Path | None = None,
    _scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> dict[str, GateReceipt]:
    try:
        return run_gate(plan.manifest_path, gate_candidate_paths(plan), receipt_dir=receipt_dir, _scanner=_scanner)
    except GateClosed as exc:
        raise PublicExportBlocked(exc.reason, exc.receipt) from exc


def release_payload(plan: ExportPlan) -> dict[str, Any]:
    release = dict(plan.manifest.get("release") or {})
    release.pop("released_at", None)
    return {
        "bundle_id": plan.bundle_id,
        "conference": plan.conference,
        "source_sha": plan.source_sha,
        "export_manifest_sha256": plan.manifest_sha256,
        "included_families": list(plan.included_families),
        "family_licenses": family_licenses(plan),
        "checksum_file": "CHECKSUMS.json",
        "release": release,
        # The public URLs, stamped at build from the workspace release: block (v6.0
        # Phase 7.1): a restamp after publication would change the tree digest.
        **stamped_urls(workspace_release_settings(plan.config.conference_dir)),
    }


def family_licenses(plan: ExportPlan) -> dict[str, str]:
    """Each included family's manifest license, read by every file that states one."""
    return {family.family_id: family.license for family in sorted(plan.families, key=lambda item: item.family_id)}


def write_cdn_sidecar(path: Path, item: ExportItem) -> None:
    if item.cdn_url is None or item.cdn_object_key is None:
        raise PublicExportBlocked(f"missing CDN URL for {item.release_path.as_posix()}")
    write_json_atomic(
        path,
        {
            "cdn_url": item.cdn_url,
            "mime_type": item.mime_type,
            "object_key": item.cdn_object_key,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "source_filename": item.source_path.name,
        },
    )


def materialize_items(plan: ExportPlan, staging_root: Path) -> list[Path]:
    emitted: list[Path] = []
    for item in plan.items:
        dest = ensure_under(staging_root / item.release_path, staging_root, label="release_path")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if item.delivery == "cdn":
            write_cdn_sidecar(dest, item)
        elif item.content is not None:
            dest.write_bytes(item.content)
        else:
            shutil.copy2(item.source_path, dest)
        emitted.append(dest)
    return emitted


def write_checksums(staging_root: Path) -> None:
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(staging_root.rglob("*"), key=lambda p: p.relative_to(staging_root).as_posix()):
        if not path.is_file() or path.name == "CHECKSUMS.json":
            continue
        rel = path.relative_to(staging_root).as_posix()
        if rel == "RELEASE.json":
            artifact_type = "metadata"
        elif rel.endswith(".cdn.json"):
            artifact_type = "cdn_sidecar"
        else:
            artifact_type = "inline"
        entries[rel] = {
            "artifact_type": artifact_type,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    write_json_atomic(staging_root / "CHECKSUMS.json", entries)


def _expected_artifact_type(relative_path: str) -> str:
    if relative_path == "RELEASE.json":
        return "metadata"
    if relative_path.endswith(".cdn.json"):
        return "cdn_sidecar"
    return "inline"


def _load_bundle_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        raise PublicExportBlocked(f"malformed {label}") from exc


def verify_materialized_bundle(
    plan: ExportPlan,
    root: Path,
    *,
    _scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> dict[str, Any]:
    """Verify produced bytes against the reviewed export plan before publication."""
    files = _portable_bundle_files(root)
    actual_paths = {path.relative_to(root).as_posix() for path in files}
    expected_paths = {item.release_path.as_posix() for item in plan.items}
    expected_paths.update({"RELEASE.json", "CHECKSUMS.json"})
    if actual_paths != expected_paths:
        raise PublicExportBlocked(
            "materialized artifact set mismatch",
            {
                "reason": "materialized_artifact_set_mismatch",
                "missing_count": len(expected_paths - actual_paths),
                "extra_count": len(actual_paths - expected_paths),
            },
        )
    for relative_path in actual_paths:
        if is_secret_like_release_path(Path(relative_path)):
            raise PublicExportBlocked(
                "secret_like_materialized_path",
                {"path": relative_path, "reason": "secret_like_materialized_path"},
            )

    checksums = _load_bundle_object(root / "CHECKSUMS.json", label="CHECKSUMS.json")
    expected_checksum_paths = expected_paths - {"CHECKSUMS.json"}
    if set(checksums) != expected_checksum_paths:
        raise PublicExportBlocked("CHECKSUMS.json artifact set mismatch")
    for relative_path in sorted(expected_checksum_paths):
        entry = checksums.get(relative_path)
        if not isinstance(entry, dict):
            raise PublicExportBlocked(f"invalid checksum entry: {relative_path}")
        artifact = root / relative_path
        expected_entry = {
            "artifact_type": _expected_artifact_type(relative_path),
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }
        if entry != expected_entry:
            raise PublicExportBlocked(f"checksum mismatch: {relative_path}")

    release = _load_bundle_object(root / "RELEASE.json", label="RELEASE.json")
    release_identity = {
        "bundle_id": plan.bundle_id,
        "conference": plan.conference,
        "source_sha": plan.source_sha,
        "export_manifest_sha256": plan.manifest_sha256,
        "included_families": list(plan.included_families),
        "family_licenses": family_licenses(plan),
        "checksum_file": "CHECKSUMS.json",
    }
    identity_mismatches = [
        key for key, value in release_identity.items() if release.get(key) != value
    ]
    nested_release = release.get("release")
    if not isinstance(nested_release, dict):
        identity_mismatches.append("release")
    else:
        nested_source_sha = nested_release.get("source_sha")
        if nested_source_sha not in {None, "", plan.source_sha}:
            identity_mismatches.append("release.source_sha")
        nested_tag = nested_release.get("release_tag")
        if nested_tag not in {None, "", plan.bundle_id}:
            identity_mismatches.append("release.release_tag")
    if identity_mismatches:
        raise PublicExportBlocked(
            f"RELEASE.json identity mismatch: {', '.join(sorted(identity_mismatches))}"
        )

    for item in plan.cdn_items:
        sidecar = _load_bundle_object(root / item.release_path, label=item.release_path.as_posix())
        expected_sidecar = {
            "cdn_url": item.cdn_url,
            "mime_type": item.mime_type,
            "object_key": item.cdn_object_key,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "source_filename": item.source_path.name,
        }
        if sidecar != expected_sidecar:
            raise PublicExportBlocked(f"CDN sidecar mismatch: {item.release_path.as_posix()}")

    for item in plan.inline_items:
        artifact = root / item.release_path
        if (
            artifact.stat().st_size != item.size_bytes
            or sha256_file(artifact) != item.sha256
        ):
            raise PublicExportBlocked(
                f"inline artifact source mismatch: {item.release_path.as_posix()}"
            )

    try:
        run_gate(plan.manifest_path, files, _scanner=_scanner)
    except GateClosed as exc:
        raise PublicExportBlocked(exc.reason, exc.receipt) from exc

    return {
        "status": "passed",
        "artifact_count": len(files),
        "exact_artifact_set": True,
        "checksums_verified": True,
        "portable_regular_files": True,
        "cdn_sidecars_verified": True,
        "manifest_source_identity_verified": True,
        "safety_gates_verified": True,
        "bundle_tree_sha256": bundle_tree_sha256(root),
    }


def _remove_publication_path(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _publication_evidence(
    transaction: PublicationTransaction,
    *,
    failure_point: str | None,
    rollback_performed: bool,
    rollback_authority_retained_through_receipt: bool,
) -> dict[str, Any]:
    return {
        "previous_bundle_present": transaction.previous_bundle_present,
        "published_version_evidence": [],
        "failure_point": failure_point,
        "rollback_performed": rollback_performed,
        "cleanup_pending": transaction.cleanup_pending,
        "rollback_authority_retained_through_receipt": (
            rollback_authority_retained_through_receipt
        ),
    }


def rollback_publication(
    transaction: PublicationTransaction,
    *,
    failure_point: str,
) -> dict[str, Any]:
    """Restore the pre-publication filesystem state when authority remains."""
    rollback_performed = False
    if not transaction.committed and not transaction.backup_cleanup_started:
        if transaction.previous_bundle_present:
            if transaction.backup_root.exists() or transaction.backup_root.is_symlink():
                try:
                    if (
                        transaction.bundle_root.exists()
                        or transaction.bundle_root.is_symlink()
                    ):
                        _remove_publication_path(transaction.bundle_root)
                    os.replace(transaction.backup_root, transaction.bundle_root)
                    rollback_performed = True
                    transaction.published = False
                except Exception:
                    rollback_performed = False
        elif transaction.bundle_root.exists() or transaction.bundle_root.is_symlink():
            try:
                _remove_publication_path(transaction.bundle_root)
                rollback_performed = True
                transaction.published = False
            except Exception:
                rollback_performed = False
    return _publication_evidence(
        transaction,
        failure_point=failure_point,
        rollback_performed=rollback_performed,
        rollback_authority_retained_through_receipt=False,
    )


def commit_publication(transaction: PublicationTransaction) -> dict[str, Any]:
    """Release rollback authority only after the success receipt is durable."""
    if transaction.committed:
        return _publication_evidence(
            transaction,
            failure_point=None,
            rollback_performed=False,
            rollback_authority_retained_through_receipt=True,
        )
    if not transaction.published:
        raise PublicExportBlocked("publication transaction is not active")

    # The verified final becomes authoritative before any backup byte is
    # destroyed. Once cleanup starts, the backup can never be rollback input.
    transaction.committed = True
    if transaction.backup_root.exists() or transaction.backup_root.is_symlink():
        transaction.backup_cleanup_started = True
        try:
            _remove_publication_path(transaction.backup_root)
        except Exception as exc:
            transaction.cleanup_pending = True
            raise PublicExportBlocked(
                "publication_backup_cleanup_pending",
                {
                    "reason": "publication_backup_cleanup_pending",
                    "cause": type(exc).__name__,
                    "failure_point": "publication_backup_cleanup",
                    "publication": _publication_evidence(
                        transaction,
                        failure_point="publication_backup_cleanup",
                        rollback_performed=False,
                        rollback_authority_retained_through_receipt=True,
                    ),
                },
            ) from exc
    return _publication_evidence(
        transaction,
        failure_point=None,
        rollback_performed=False,
        rollback_authority_retained_through_receipt=True,
    )


def publish_verified_staging(
    staging_root: Path,
    bundle_root: Path,
) -> PublicationTransaction:
    """Publish verified staging while retaining rollback authority."""
    if staging_root.parent != bundle_root.parent:
        raise PublicExportBlocked("staging and final bundle must be adjacent")
    if not staging_root.is_dir() or staging_root.is_symlink():
        raise PublicExportBlocked("verified staging directory is required")

    previous_bundle_present = bundle_root.exists() or bundle_root.is_symlink()
    backup_root = bundle_root.with_name(
        f".{bundle_root.name}.backup-{uuid.uuid4().hex}"
    )
    transaction = PublicationTransaction(
        bundle_root=bundle_root,
        backup_root=backup_root,
        previous_bundle_present=previous_bundle_present,
        staging_digest=bundle_tree_sha256(staging_root),
    )
    failure_point = "prior_final_to_backup" if previous_bundle_present else "staging_to_final"

    try:
        if previous_bundle_present:
            os.replace(bundle_root, backup_root)
            failure_point = "staging_to_final"
        os.replace(staging_root, bundle_root)
        transaction.published = True
        failure_point = "final_verification"
        if bundle_tree_sha256(bundle_root) != transaction.staging_digest:
            raise PublicExportBlocked("published bundle digest mismatch")
        return transaction
    except Exception as exc:
        publication = rollback_publication(
            transaction,
            failure_point=failure_point,
        )
        if staging_root.exists() or staging_root.is_symlink():
            _remove_publication_path(staging_root)
        raise PublicExportBlocked(
            "atomic_publication_failed",
            {
                "reason": "atomic_publication_failed",
                "cause": type(exc).__name__,
                "failure_point": failure_point,
                "publication": publication,
            },
        ) from exc


def actual_artifacts(staging_root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(staging_root.rglob("*"), key=lambda p: p.relative_to(staging_root).as_posix()):
        if path.is_file():
            artifacts.append(
                {
                    "path": path.relative_to(staging_root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return artifacts


def default_receipt_path(plan: ExportPlan) -> Path:
    suffix = "dry-run" if plan.config.dry_run else "write"
    return (
        plan.config.conference_dir
        / "storage"
        / "workflow-runs"
        / f"public-export-{plan.bundle_id}-{suffix}-receipt.json"
    )


def _receipt_project_root(conference_dir: Path) -> Path:
    conference = conference_dir.absolute()
    if conference.parent.name == "conferences":
        return conference.parent.parent
    try:
        conference.relative_to(PROJECT_ROOT)
    except ValueError:
        return conference.parent
    return PROJECT_ROOT


def _unsafe_receipt_path(exc: ReceiptPathError) -> PublicExportBlocked:
    return PublicExportBlocked(
        "unsafe_receipt_path",
        {
            "label": exc.label,
            "reason": "unsafe_receipt_path",
            "reason_code": exc.code,
        },
    )


def _confined_export_receipt_path(
    config: ExportConfig,
    candidate: Path,
    *,
    label: str = "receipt_path",
    must_exist: bool = False,
) -> Path:
    try:
        return confine_workflow_receipt_path(
            config.conference_dir,
            candidate,
            project_root=_receipt_project_root(config.conference_dir),
            label=label,
            must_exist=must_exist,
        )
    except ReceiptPathError as exc:
        raise _unsafe_receipt_path(exc) from exc


def validate_explicit_receipt_path(config: ExportConfig) -> Path | None:
    if config.receipt_path is None:
        return None
    return _confined_export_receipt_path(config, config.receipt_path)


def receipt_path_for(plan: ExportPlan) -> Path:
    return _confined_export_receipt_path(
        plan.config,
        plan.config.receipt_path or default_receipt_path(plan),
    )


def reviewed_receipt_path_for(plan: ExportPlan) -> Path:
    """Return the regular, conference-local receipt allowed to authorize writes."""
    candidate = plan.config.reviewed_dry_run_receipt
    if candidate is None:
        raise PublicExportBlocked(ERROR_REVIEWED_RECEIPT)
    return _confined_export_receipt_path(
        plan.config,
        candidate,
        label="reviewed_dry_run_receipt",
        must_exist=True,
    )


def planned_artifacts(plan: ExportPlan) -> list[dict[str, Any]]:
    return [
        {
            "family_id": item.family_id,
            "path": item.release_path.as_posix(),
            "delivery": item.delivery,
            "source_sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in plan.items
    ]


def cdn_stub_artifacts(plan: ExportPlan) -> list[dict[str, Any]]:
    return [
        {
            "family_id": item.family_id,
            "path": item.release_path.as_posix(),
            "cdn_url": item.cdn_url,
            "object_key": item.cdn_object_key,
            "source_sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in plan.cdn_items
    ]


def cdn_upload_source_contracts(plan: ExportPlan) -> list[dict[str, Any]]:
    """Bind future CDN uploads to reviewed bytes without exposing target paths."""
    conference_dir = plan.config.conference_dir.absolute()
    rows: list[dict[str, Any]] = []
    for item in plan.cdn_items:
        source_path = item.source_path.absolute()
        try:
            lexical_source = source_path.relative_to(conference_dir)
        except ValueError as exc:
            raise PublicExportBlocked(
                "cdn_source_identity_mismatch",
                {
                    "family_id": item.family_id,
                    "path": item.release_path.as_posix(),
                    "reason": "cdn_source_identity_mismatch",
                },
            ) from exc
        if lexical_source.is_absolute() or ".." in lexical_source.parts:
            raise PublicExportBlocked(
                "cdn_source_identity_mismatch",
                {
                    "family_id": item.family_id,
                    "path": item.release_path.as_posix(),
                    "reason": "cdn_source_identity_mismatch",
                },
            )
        rows.append(
            {
                "family_id": item.family_id,
                "path": item.release_path.as_posix(),
                "object_key": item.cdn_object_key,
                "source_path": lexical_source.as_posix(),
                "source_sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["family_id"]),
            str(row["path"]),
            str(row["object_key"]),
            str(row["source_path"]),
        ),
    )


def verify_cdn_source_identity(plan: ExportPlan) -> dict[str, Any]:
    """Re-open every CDN source and require the reviewed hash and size."""
    rows = cdn_upload_source_contracts(plan)
    rows_by_path = {str(row["path"]): row for row in rows}
    for item in plan.cdn_items:
        failure_receipt = {
            "family_id": item.family_id,
            "path": rows_by_path[item.release_path.as_posix()]["source_path"],
            "reason": "cdn_source_identity_mismatch",
        }
        try:
            lexical_stat = item.source_path.lstat()
            if not (
                stat.S_ISREG(lexical_stat.st_mode)
                or stat.S_ISLNK(lexical_stat.st_mode)
            ):
                raise OSError("CDN source is not a regular file or symlink")
            current_stat = item.source_path.stat()
            if not stat.S_ISREG(current_stat.st_mode):
                raise OSError("CDN source target is not a regular file")
            current_sha256 = sha256_file(item.source_path)
            current_size = item.source_path.stat().st_size
        except (FileNotFoundError, OSError, PermissionError) as exc:
            raise PublicExportBlocked(
                "cdn_source_identity_mismatch",
                failure_receipt,
            ) from exc
        if current_size != item.size_bytes or current_sha256 != item.sha256:
            raise PublicExportBlocked(
                "cdn_source_identity_mismatch",
                failure_receipt,
            )

    canonical_rows = json.dumps(
        rows,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "cdn_upload_sources": rows,
        "cdn_upload_sources_sha256": sha256_text(canonical_rows),
    }


def build_receipt_payload(
    plan: ExportPlan,
    *,
    status: str,
    exit_code: int,
    error: str | None = None,
    actual: list[dict[str, Any]] | None = None,
    gate_receipts: dict[str, GateReceipt] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "status": status,
        "conference": plan.conference,
        "bundle_id": plan.bundle_id,
        "source_sha": plan.source_sha,
        "export_manifest_path": str(plan.manifest_path),
        "export_manifest_sha256": plan.manifest_sha256,
        "output_dir": str(plan.config.output_dir),
        "bundle_path": str(plan.bundle_root),
        "dry_run": plan.config.dry_run,
        "exit_code": exit_code,
        "included_families": list(plan.included_families),
        "planned_artifacts": planned_artifacts(plan),
        "actual_artifacts": actual or [],
        "planned_artifact_count": len(plan.items),
        "actual_artifact_count": len(actual or []),
        "cdn_stubs": cdn_stub_artifacts(plan),
        "redacted_files": list(plan.redacted_files),
        "binary_markers": list(plan.binary_markers),
        "excluded_artifacts": [dataclasses.asdict(item) for item in plan.excluded_artifacts],
        "license_gating": {
            "caution_families": list(caution_family_ids(plan)),
            "acknowledged_caution_families": list(plan.config.ack_caution_families),
            "gate_receipts": {
                key: dataclasses.asdict(value)
                for key, value in (gate_receipts or {}).items()
            },
        },
        "generated_at": utc_now(),
        "error": error,
        "secret_values_recorded": False,
    }
    if plan.config.orchestrator_stage is not None:
        payload["orchestrator_stage"] = plan.config.orchestrator_stage
    if extra:
        payload.update(extra)
    return payload


def write_export_receipt(
    plan: ExportPlan,
    *,
    status: str,
    exit_code: int,
    error: str | None = None,
    actual: list[dict[str, Any]] | None = None,
    gate_receipts: dict[str, GateReceipt] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = receipt_path_for(plan)
    write_json_atomic(
        path,
        build_receipt_payload(
            plan,
            status=status,
            exit_code=exit_code,
            error=error,
            actual=actual,
            gate_receipts=gate_receipts,
            extra=extra,
        ),
    )
    return path


def _sanitized_failure_reason(exc: Exception) -> str:
    if not isinstance(exc, PublicExportBlocked):
        return type(exc).__name__
    reason = exc.reason.strip().splitlines()[0][:200]
    lowered = reason.lower()
    secret_markers = (
        "api_key=",
        "apikey=",
        "password=",
        "secret=",
        "token=",
        "/volumes/",
        "google drive",
    )
    if any(marker in lowered for marker in secret_markers):
        return "public_export_blocked"
    return reason or "public_export_blocked"


def _sanitized_blocker_receipt(exc: Exception) -> dict[str, Any]:
    reason = _sanitized_failure_reason(exc)
    if not isinstance(exc, PublicExportBlocked):
        return {"reason": reason}
    allowed_fields = {
        "cause",
        "extra_count",
        "failure_point",
        "family_id",
        "conflicting_family_id",
        "label",
        "missing_count",
        "path",
        "public_path",
        "reason",
        "reason_code",
    }
    receipt: dict[str, Any] = {}
    for key in allowed_fields:
        value = exc.receipt.get(key)
        if isinstance(value, (str, int, float, bool)) and value not in {"", None}:
            text = str(value)
            lowered = text.lower()
            if not Path(text).is_absolute() and "google drive" not in lowered and "/volumes/" not in lowered:
                receipt[key] = value
    receipt.setdefault("reason", reason)
    return receipt


def _publication_evidence_from_exception(
    exc: Exception,
    *,
    previous_bundle_present: bool,
) -> dict[str, Any]:
    default = {
        "previous_bundle_present": previous_bundle_present,
        "published_version_evidence": [],
        "failure_point": None,
        "rollback_performed": False,
        "cleanup_pending": False,
        "rollback_authority_retained_through_receipt": False,
    }
    receipt = getattr(exc, "receipt", None)
    if not isinstance(receipt, dict):
        return default
    publication = receipt.get("publication")
    if not isinstance(publication, dict):
        return default
    raw_evidence = publication.get("published_version_evidence")
    published_version_evidence = (
        [dict(item) for item in raw_evidence if isinstance(item, dict)]
        if isinstance(raw_evidence, list)
        else []
    )
    return {
        "previous_bundle_present": bool(
            publication.get("previous_bundle_present", previous_bundle_present)
        ),
        "published_version_evidence": published_version_evidence,
        "failure_point": publication.get("failure_point"),
        "rollback_performed": bool(publication.get("rollback_performed", False)),
        "cleanup_pending": bool(publication.get("cleanup_pending", False)),
        "rollback_authority_retained_through_receipt": bool(
            publication.get("rollback_authority_retained_through_receipt", False)
        ),
    }


def _reproducibility_evidence_from_exception(exc: Exception) -> dict[str, Any] | None:
    receipt = getattr(exc, "receipt", None)
    if not isinstance(receipt, dict):
        return None
    reproducibility = receipt.get("reproducibility")
    if not isinstance(reproducibility, dict):
        return None
    allowed = {
        "source_sha",
        "export_manifest_sha256",
        "prior_bundle_tree_sha256",
        "current_bundle_tree_sha256",
        "byte_identical",
    }
    return {
        key: reproducibility.get(key)
        for key in allowed
        if key in reproducibility
    }


def _failed_receipt_path(config: ExportConfig, plan: ExportPlan | None) -> Path:
    bundle_id = plan.bundle_id if plan is not None else (config.bundle_id or "unknown-bundle")
    suffix = "dry-run" if config.dry_run else "write"
    default_path = (
        config.conference_dir
        / "storage"
        / "workflow-runs"
        / f"public-export-{bundle_id}-{suffix}-receipt.json"
    )
    if config.receipt_path is not None:
        try:
            return _confined_export_receipt_path(config, config.receipt_path)
        except PublicExportBlocked:
            pass
    return _confined_export_receipt_path(config, default_path)


def write_failed_export_receipt(
    config: ExportConfig,
    exc: Exception,
    *,
    plan: ExportPlan | None = None,
    failure_stage: str,
) -> Path:
    reason = _sanitized_failure_reason(exc)
    status = "blocked" if isinstance(exc, PublicExportBlocked) else "failed"
    blocker_receipt = _sanitized_blocker_receipt(exc)
    if plan is not None:
        publication = _publication_evidence_from_exception(
            exc,
            previous_bundle_present=plan.bundle_root.exists(),
        )
        reproducibility = _reproducibility_evidence_from_exception(exc)
        failure_extra: dict[str, Any] = {
            "failure_stage": failure_stage,
            "blocker_receipt": blocker_receipt,
            "publication": publication,
            "secret_values_recorded": False,
        }
        if reproducibility is not None:
            failure_extra["reproducibility"] = reproducibility
        payload = build_receipt_payload(
            plan,
            status=status,
            exit_code=1,
            error=reason,
            extra=failure_extra,
        )
    else:
        payload = {
            "schema_version": 1,
            "receipt_type": RECEIPT_TYPE,
            "status": status,
            "conference": config.conference_dir.name,
            "bundle_id": config.bundle_id,
            "dry_run": config.dry_run,
            "exit_code": 1,
            "failure_stage": failure_stage,
            "blocker_receipt": blocker_receipt,
            "secret_values_recorded": False,
            "generated_at": utc_now(),
            "error": reason,
        }
        if config.orchestrator_stage is not None:
            payload["orchestrator_stage"] = config.orchestrator_stage
    path = _failed_receipt_path(config, plan)
    write_json_atomic(path, payload)
    return path


def render_preflight_report(plan: ExportPlan) -> str:
    return "\n".join(
        [
            "=" * 60,
            "Conference Public Export",
            "=" * 60,
            f"Conference: {plan.conference}",
            f"Bundle ID: {plan.bundle_id}",
            f"Manifest: {plan.manifest_path}",
            f"Output bundle: {plan.bundle_root}",
            f"Included families: {', '.join(plan.included_families) if plan.included_families else '(none)'}",
            f"Excluded artifacts: {len(plan.excluded_artifacts)}",
            f"Redacted files: {len(plan.redacted_files)}",
            f"CDN stubs: {len(plan.cdn_items)}",
            f"Dry run: {'enabled' if plan.config.dry_run else 'disabled'}",
            "=" * 60,
        ]
    )


def load_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise PublicExportBlocked(f"{ERROR_REVIEWED_RECEIPT}: unreadable receipt") from exc
    if payload.get("receipt_type") != RECEIPT_TYPE:
        raise PublicExportBlocked(f"{ERROR_REVIEWED_RECEIPT}: wrong receipt_type")
    return payload


def _sorted_artifact_contract(artifacts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(artifact) for artifact in artifacts),
        key=lambda artifact: (
            str(artifact.get("family_id", "")),
            str(artifact.get("path", "")),
            str(artifact.get("delivery", "")),
            str(artifact.get("cdn_url", "")),
        ),
    )


def validate_reviewed_dry_run_receipt(
    plan: ExportPlan,
    receipt_path: Path | None = None,
) -> Path:
    if receipt_path is None:
        confined_path = reviewed_receipt_path_for(plan)
    else:
        confined_path = _confined_export_receipt_path(
            plan.config,
            receipt_path,
            label="reviewed_dry_run_receipt",
            must_exist=True,
        )
    receipt = load_receipt(confined_path)
    expected = {
        "status": "succeeded",
        "dry_run": True,
        "exit_code": 0,
        "conference": plan.conference,
        "bundle_id": plan.bundle_id,
        "source_sha": plan.source_sha,
        "export_manifest_sha256": plan.manifest_sha256,
        "output_dir": str(plan.config.output_dir),
    }
    mismatches = [
        key for key, value in expected.items() if receipt.get(key) != value
    ]
    if mismatches:
        raise PublicExportBlocked(f"{ERROR_REVIEWED_RECEIPT}: {', '.join(sorted(mismatches))}")
    if _sorted_artifact_contract(receipt.get("planned_artifacts", [])) != _sorted_artifact_contract(
        planned_artifacts(plan)
    ):
        raise PublicExportBlocked(f"{ERROR_REVIEWED_RECEIPT}: planned_artifacts")
    if _sorted_artifact_contract(receipt.get("cdn_stubs", [])) != _sorted_artifact_contract(
        cdn_stub_artifacts(plan)
    ):
        raise PublicExportBlocked(f"{ERROR_REVIEWED_RECEIPT}: cdn_stubs")
    return confined_path


def validate_write_cdn_targets(plan: ExportPlan) -> None:
    if not plan.cdn_items:
        return
    cdn_family_ids = {item.family_id for item in plan.cdn_items}
    for family in plan.families:
        if family.family_id not in cdn_family_ids:
            continue
        base_url = family.cdn_base_url.lower()
        if "example.invalid" in base_url or "placeholder" in base_url:
            raise PublicExportBlocked(f"{ERROR_PLACEHOLDER_CDN}: {family.family_id}")
    family_by_id = {family.family_id: family for family in plan.families}
    for item in plan.cdn_items:
        family = family_by_id[item.family_id]
        media_release_path = item.release_path.with_name(
            item.release_path.name.removesuffix(".cdn.json")
        )
        object_key = validate_cdn_target(
            family,
            plan.bundle_id,
            media_release_path,
            item.cdn_url or "",
        )
        if item.cdn_object_key != object_key:
            raise PublicExportBlocked(f"CDN object key mismatch for family {item.family_id}")


def _released_at_value(payload: dict[str, Any]) -> str | None:
    top_level = payload.get("released_at")
    if isinstance(top_level, str) and top_level.strip():
        return top_level.strip()
    nested = payload.get("release")
    if isinstance(nested, dict):
        value = nested.get("released_at")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _receipt_bundle_values(payload: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("bundle_id", "release_tag"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            values.add(value.strip())
    for key in ("release", "bundle"):
        nested = payload.get(key)
        if not isinstance(nested, dict):
            continue
        for nested_key in ("bundle_id", "release_tag"):
            value = nested.get(nested_key)
            if isinstance(value, str) and value.strip():
                values.add(value.strip())
    return values


def _is_successful_live_publication_receipt(
    path: Path,
    payload: dict[str, Any],
    bundle_id: str,
) -> bool:
    if payload.get("status") != "succeeded" or payload.get("exit_code") != 0:
        return False
    if type(payload.get("dry_run")) is not bool or payload.get("dry_run") is not False:
        return False
    receipt_type = str(payload.get("receipt_type") or "")
    if payload.get("live") is not True or receipt_type not in LIVE_PUBLICATION_RECEIPT_TYPES:
        return False
    explicit_bundles = _receipt_bundle_values(payload)
    if bundle_id not in explicit_bundles:
        return False
    if receipt_type == "conference_pipeline.public_cdn_upload":
        return is_completed_live_upload_receipt(payload, bundle_id)
    required_fields = {
        "conference_pipeline.cloudflare_pages_deploy": ("deployment_id",),
        "conference_pipeline.hf_dataset_push": ("dataset_url", "commit_sha"),
        "conference_pipeline.provider_success": ("public_url", "commit_sha"),
        "conference_pipeline.public_repo_push": ("commit_sha",),
        "conference_pipeline.zenodo_deposit": ("doi", "record_url"),
    }.get(receipt_type, ())
    if not all(payload.get(field) not in (None, "") for field in required_fields):
        return False
    if receipt_type == "conference_pipeline.cloudflare_pages_deploy":
        return any(payload.get(field) not in (None, "") for field in ("deployment_url", "public_url"))
    if receipt_type == "conference_pipeline.public_repo_push":
        return any(payload.get(field) not in (None, "") for field in ("public_repo_url", "public_url", "url"))
    return True


def published_bundle_evidence(plan: ExportPlan) -> list[dict[str, Any]]:
    """Return secret-free evidence that permanently freezes this bundle ID."""
    evidence: list[dict[str, Any]] = []
    manifest_release = plan.manifest.get("release")
    if isinstance(manifest_release, dict):
        released_at = _released_at_value(manifest_release)
        if released_at:
            evidence.append(
                {
                    "source": "export_manifest",
                    "path": plan.manifest_path.relative_to(
                        plan.config.conference_dir.resolve()
                    ).as_posix(),
                    "released_at": released_at,
                }
            )

    final_release_path = plan.bundle_root / "RELEASE.json"
    if final_release_path.is_file() and not final_release_path.is_symlink():
        try:
            final_release = read_json(final_release_path)
        except (json.JSONDecodeError, OSError, ValueError):
            final_release = {}
        released_at = _released_at_value(final_release)
        if released_at:
            evidence.append(
                {
                    "source": "final_release",
                    "path": "RELEASE.json",
                    "released_at": released_at,
                }
            )

    receipt_root = plan.config.conference_dir / "storage" / "workflow-runs"
    if receipt_root.is_dir():
        for path in sorted(receipt_root.rglob("*.json"), key=lambda item: item.as_posix()):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                payload = read_json(path)
            except (json.JSONDecodeError, OSError, ValueError):
                continue
            if not _is_successful_live_publication_receipt(path, payload, plan.bundle_id):
                continue
            evidence.append(
                {
                    "source": "live_publication_receipt",
                    "path": path.relative_to(plan.config.conference_dir).as_posix(),
                    "receipt_type": payload.get("receipt_type"),
                }
            )
    return evidence


def _inspect_existing_bundle(
    plan: ExportPlan,
    root: Path,
    *,
    _scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> dict[str, Any]:
    files = _portable_bundle_files(root)
    paths = {path.relative_to(root).as_posix() for path in files}
    if "RELEASE.json" not in paths or "CHECKSUMS.json" not in paths:
        raise PublicExportBlocked("existing bundle metadata is incomplete")
    if any(is_secret_like_release_path(Path(path)) for path in paths):
        raise PublicExportBlocked("existing bundle contains secret-like path")

    checksums = _load_bundle_object(root / "CHECKSUMS.json", label="CHECKSUMS.json")
    expected_checksum_paths = paths - {"CHECKSUMS.json"}
    if set(checksums) != expected_checksum_paths:
        raise PublicExportBlocked("existing CHECKSUMS.json artifact set mismatch")
    for relative_path in sorted(expected_checksum_paths):
        entry = checksums.get(relative_path)
        artifact = root / relative_path
        expected_entry = {
            "artifact_type": _expected_artifact_type(relative_path),
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }
        if not isinstance(entry, dict) or entry != expected_entry:
            raise PublicExportBlocked(f"existing bundle checksum mismatch: {relative_path}")

    release = _load_bundle_object(root / "RELEASE.json", label="RELEASE.json")
    if release.get("bundle_id") != plan.bundle_id or release.get("conference") != plan.conference:
        raise PublicExportBlocked("existing bundle identity mismatch")
    source_sha = release.get("source_sha")
    manifest_sha = release.get("export_manifest_sha256")
    if not isinstance(source_sha, str) or not source_sha:
        raise PublicExportBlocked("existing bundle source identity missing")
    if not isinstance(manifest_sha, str) or not manifest_sha:
        raise PublicExportBlocked("existing bundle manifest identity missing")

    if source_sha == plan.source_sha and manifest_sha == plan.manifest_sha256:
        verification = verify_materialized_bundle(plan, root, _scanner=_scanner)
        tree_digest = verification["bundle_tree_sha256"]
    else:
        try:
            run_gate(plan.manifest_path, files, _scanner=_scanner)
        except GateClosed as exc:
            raise PublicExportBlocked(exc.reason, exc.receipt) from exc
        tree_digest = bundle_tree_sha256(root)
    return {
        "source_sha": source_sha,
        "export_manifest_sha256": manifest_sha,
        "bundle_tree_sha256": tree_digest,
    }


def ensure_write_allowed(
    plan: ExportPlan,
) -> tuple[list[dict[str, Any]], Path]:
    reviewed_receipt_path = validate_reviewed_dry_run_receipt(plan)
    validate_write_cdn_targets(plan)
    evidence = published_bundle_evidence(plan)
    if evidence:
        raise PublicExportBlocked(
            ERROR_IMMUTABLE_BUNDLE,
            {
                "reason": "published_bundle_immutable",
                "publication": {
                    "previous_bundle_present": plan.bundle_root.exists(),
                    "published_version_evidence": evidence,
                    "rollback_performed": False,
                },
            },
        )
    return evidence, reviewed_receipt_path


def materialize_isolated(
    plan: ExportPlan,
    dest: Path,
    *,
    _scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> dict[str, Any]:
    """Materialize the plan's bundle into ``dest`` for comparison, never publishing it.

    This is the reproduction path (v6.0 Phase 3, spec section 6.4). It runs the same
    gates, materialization, ``RELEASE.json``, ``CHECKSUMS.json`` and verification as
    ``execute_export_plan``, but it writes only under ``dest``: no receipt, no staging
    directory beside the published bundle, no publication. It therefore skips
    ``ensure_write_allowed``, whose immutability guard protects the published bundle
    root that this function never touches. ``dest`` must be absent or empty.
    """
    if dest.is_symlink() or (dest.exists() and (not dest.is_dir() or any(dest.iterdir()))):
        raise PublicExportBlocked("isolated bundle directory must be absent or empty")
    run_plan_level_gate(plan, _scanner=_scanner)
    enforce_caution_ack(plan)
    dest.mkdir(parents=True, exist_ok=True)
    materialize_items(plan, dest)
    write_json_atomic(dest / "RELEASE.json", release_payload(plan))
    write_checksums(dest)
    verification = verify_materialized_bundle(plan, dest, _scanner=_scanner)
    files = _portable_bundle_files(dest)
    return {
        "bundle_id": plan.bundle_id,
        "conference": plan.conference,
        "source_sha": plan.source_sha,
        "export_manifest_sha256": plan.manifest_sha256,
        "bundle_tree_sha256": bundle_tree_sha256(dest),
        "file_count": len(files),
        "artifacts": [
            {
                "path": path.relative_to(dest).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
        "bundle_verification_status": verification.get("status"),
    }


def execute_export_plan(
    plan: ExportPlan,
    *,
    _scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> Path:
    if plan.config.dry_run:
        raise ValueError("execute_export_plan requires dry_run=False")

    published_evidence, reviewed_receipt_path = ensure_write_allowed(plan)
    previous_bundle = None
    if plan.bundle_root.exists() or plan.bundle_root.is_symlink():
        previous_bundle = _inspect_existing_bundle(
            plan,
            plan.bundle_root,
            _scanner=_scanner,
        )
    gate_receipts = run_plan_level_gate(plan, _scanner=_scanner)
    enforce_caution_ack(plan)

    plan.config.output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{plan.bundle_id}.staging-", dir=plan.config.output_dir))
    transaction: PublicationTransaction | None = None
    failure_point = "materialize_staging"
    try:
        materialize_items(plan, staging)
        write_json_atomic(staging / "RELEASE.json", release_payload(plan))
        write_checksums(staging)
        failure_point = "staging_verification"
        bundle_verification = verify_materialized_bundle(
            plan,
            staging,
            _scanner=_scanner,
        )
        prior_tree_digest = (
            previous_bundle["bundle_tree_sha256"]
            if previous_bundle is not None
            else None
        )
        current_tree_digest = bundle_verification["bundle_tree_sha256"]
        byte_identical = (
            prior_tree_digest == current_tree_digest
            if prior_tree_digest is not None
            else None
        )
        reproducibility = {
            "source_sha": plan.source_sha,
            "export_manifest_sha256": plan.manifest_sha256,
            "prior_bundle_tree_sha256": prior_tree_digest,
            "current_bundle_tree_sha256": current_tree_digest,
            "byte_identical": byte_identical,
        }
        same_inputs = bool(
            previous_bundle is not None
            and previous_bundle["source_sha"] == plan.source_sha
            and previous_bundle["export_manifest_sha256"] == plan.manifest_sha256
        )
        if same_inputs and not byte_identical:
            raise PublicExportBlocked(
                ERROR_REPRODUCIBILITY,
                {
                    "reason": "reproducibility_digest_mismatch",
                    "reproducibility": reproducibility,
                    "publication": {
                        "previous_bundle_present": True,
                        "published_version_evidence": published_evidence,
                        "rollback_performed": False,
                    },
                },
            )
        failure_point = "cdn_source_identity"
        cdn_source_identity = verify_cdn_source_identity(plan)
        failure_point = "publish_verified_staging"
        transaction = publish_verified_staging(staging, plan.bundle_root)
        publication = _publication_evidence(
            transaction,
            failure_point=None,
            rollback_performed=False,
            rollback_authority_retained_through_receipt=True,
        )
        publication["published_version_evidence"] = published_evidence
        failure_point = "final_verification"
        final_verification = verify_materialized_bundle(
            plan,
            plan.bundle_root,
            _scanner=_scanner,
        )
        if final_verification != bundle_verification:
            raise PublicExportBlocked("final bundle verification drift")
        failure_point = "artifact_enumeration"
        actual = actual_artifacts(plan.bundle_root)
        failure_point = "success_receipt_persistence"
        write_export_receipt(
            plan,
            status="succeeded",
            exit_code=0,
            actual=actual,
            gate_receipts=gate_receipts,
            extra={
                "reviewed_dry_run_receipt_path": str(
                    reviewed_receipt_path
                ),
                "bundle_verification": final_verification,
                "publication": publication,
                "reproducibility": reproducibility,
                **cdn_source_identity,
            },
        )
        failure_point = "publication_commit"
        commit_publication(transaction)
        return plan.bundle_root
    except Exception as exc:
        if transaction is not None and not transaction.committed:
            publication = rollback_publication(
                transaction,
                failure_point=failure_point,
            )
            publication["published_version_evidence"] = published_evidence
        elif (
            isinstance(exc, PublicExportBlocked)
            and isinstance(exc.receipt.get("publication"), dict)
        ):
            publication = _publication_evidence_from_exception(
                exc,
                previous_bundle_present=previous_bundle is not None,
            )
            publication["published_version_evidence"] = published_evidence
        else:
            publication = {
                "previous_bundle_present": previous_bundle is not None,
                "published_version_evidence": published_evidence,
                "failure_point": failure_point,
                "rollback_performed": False,
                "cleanup_pending": False,
                "rollback_authority_retained_through_receipt": False,
            }
        if staging.exists() or staging.is_symlink():
            _remove_publication_path(staging)
        if isinstance(exc, PublicExportBlocked):
            exc.receipt["publication"] = publication
            exc.receipt.setdefault("failure_point", failure_point)
            raise
        raise PublicExportBlocked(
            "publication_transaction_failed",
            {
                "reason": "publication_transaction_failed",
                "cause": type(exc).__name__,
                "failure_point": failure_point,
                "publication": publication,
            },
        ) from exc


def dry_run_export(
    plan: ExportPlan,
    *,
    _scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> str:
    gate_receipts = run_plan_level_gate(plan, _scanner=_scanner)
    enforce_caution_ack(plan)
    report = render_preflight_report(plan)
    write_export_receipt(
        plan,
        status="succeeded",
        exit_code=0,
        gate_receipts=gate_receipts,
    )
    return report


def run_export(
    config: ExportConfig,
    *,
    _scanner: Callable[[list[Path]], list[dict]] | None = None,
) -> tuple[int, str]:
    plan: ExportPlan | None = None
    failure_stage = "validate_receipt_path"
    try:
        validate_explicit_receipt_path(config)
        failure_stage = "build_export_plan"
        plan = build_export_plan(config)
        if config.dry_run:
            failure_stage = "dry_run_export"
            report = dry_run_export(plan, _scanner=_scanner)
            return 0, report
        failure_stage = "execute_export_plan"
        execute_export_plan(plan, _scanner=_scanner)
        return 0, f"Public export written: {plan.bundle_root}"
    except Exception as exc:
        try:
            write_failed_export_receipt(
                config,
                exc,
                plan=plan,
                failure_stage=failure_stage,
            )
        except (PublicExportBlocked, ReceiptPathError):
            pass
        status = "blocked" if isinstance(exc, PublicExportBlocked) else "failed"
        return 1, f"public export {status}: {_sanitized_failure_reason(exc)}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a public release bundle")
    parser.add_argument("--conference-dir", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bundle-id")
    parser.add_argument("--source-sha")
    parser.add_argument("--orchestrator-stage")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument("--reviewed-dry-run-receipt", type=Path)
    parser.add_argument("--ack-caution-family", action="append", default=[])
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ExportConfig:
    conference_dir = args.conference_dir.resolve()
    export_manifest = (
        args.export_manifest.resolve()
        if args.export_manifest is not None
        else conference_dir / "storage" / "public_export_manifest.json"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else conference_dir / "exports" / "public-release"
    )
    return ExportConfig(
        conference_dir=conference_dir,
        export_manifest=export_manifest,
        output_dir=output_dir,
        bundle_id=args.bundle_id,
        dry_run=args.dry_run,
        receipt_path=args.receipt_path,
        reviewed_dry_run_receipt=args.reviewed_dry_run_receipt,
        ack_caution_families=tuple(args.ack_caution_family),
        source_sha=args.source_sha,
        orchestrator_stage=args.orchestrator_stage,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    exit_code, message = run_export(config_from_args(args))
    print(message)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
