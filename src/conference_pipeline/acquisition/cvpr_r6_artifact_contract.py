"""The single executable artifact contract for CVPR 2026 r6.

This module owns the fixed CVPR workspace, r6 run identity, and every r6
control/payload/output path.  Production command owners use ``production()``;
test suites may inject an isolated workspace through ``for_test_fixture()``.
No production owner may accept a substitute workspace, run id, receipt, or
output path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


class R6ArtifactContractError(ValueError):
    """Raised for a substituted or escaped r6 artifact boundary."""


R6_RUN_ID = "cvpr2026-live-production-r6-expanded-media"
R6_STAGE_ID = "transcription-ocr-r6"
R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT = 2_002
R6_WORKSPACE_RELATIVE_PATH = Path("conferences/cvpr2026")
R6_ALL_IN_CAP_USD = 800.0
R6_SPEECH_CAP_USD = 200.0
R6_OCR_CAP_USD = 30.0
R6_ENVELOPE_VERSION = 1

# This is deliberately data rather than an ad-hoc script constant.  The audio
# producer and every later consumer bind this finite policy digest, so a
# changed downloader/auth surface cannot be mistaken for the old public lane.
R6_PUBLIC_AUDIO_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "origin": "direct_public_youtube",
        "hosts": ["youtube.com", "youtu.be"],
        "downloader": "yt-dlp",
        "shell": False,
        "playlist": False,
        "credentials": False,
        "proxy": False,
        "max_attempts": 3,
        "timeout_seconds": 120,
    }
)

# The values are relative to the one fixed CVPR workspace.  Consumers import
# this mapping instead of maintaining independent r6 path literals.
R6_ARTIFACT_PATHS: Mapping[str, Path] = MappingProxyType(
    {
        "r6_root": Path("storage/transcription-ocr/cvpr2026/r6"),
        "recovery_inventory": Path("storage/transcription-ocr/cvpr2026/r6/recovery_inventory.json"),
        "audio_recovery_receipt": Path(
            "storage/transcription-ocr/cvpr2026/r6/audio_recovery_receipt.json"
        ),
        # A deliberately tiny control-plane outcome for a fixed-owner pass.
        # It binds to an already-validated checkpoint, but never contains the
        # checkpoint body, local paths, URLs, headers, or provider data.
        "continuation_diagnostic": Path(
            "storage/transcription-ocr/cvpr2026/r6/continuation_diagnostic.json"
        ),
        # This is deliberately separate from the historical continuation
        # receipt.  Keeping the old receipt readable makes an interrupted
        # upgrade auditable, while the CLI decision record accounts for every
        # finite non-success branch at the public boundary.
        "cli_decision": Path(
            "storage/transcription-ocr/cvpr2026/r6/cli_decision.json"
        ),
        # The public fixed CLI owns this tiny successor-control namespace
        # before it imports an owner.  It is intentionally control evidence,
        # not an acquisition payload: the predecessor decision remains at its
        # canonical path and the one O_EXCL archive/admission/claim lineage is
        # rooted here.
        "cli_decision_successor_root": Path(
            "storage/transcription-ocr/cvpr2026/r6/decision-successor"
        ),
        # This root is control evidence only.  It is conditionally populated
        # when the one historical downloader-preflight checkpoint must be
        # preserved before its canonical successor can start.
        "audio_recovery_archive_root": Path(
            "storage/transcription-ocr/cvpr2026/r6/audio-recovery-archive"
        ),
        "material_discovery_receipt": Path(
            "storage/transcription-ocr/cvpr2026/r6/material_discovery_receipt.json"
        ),
        "material_acquisition_receipt": Path(
            "storage/transcription-ocr/cvpr2026/r6/material_acquisition_receipt.json"
        ),
        "cost_estimate_receipt": Path(
            "storage/transcription-ocr/cvpr2026/r6/cost_estimate_receipt.json"
        ),
        "benchmark_receipt": Path("storage/transcription-ocr/cvpr2026/r6/benchmark_receipt.json"),
        "provider_receipts": Path("storage/transcription-ocr/cvpr2026/r6/provider_receipts.json"),
        "budget_ledger": Path("storage/transcription-ocr/cvpr2026/r6/budget_ledger.json"),
        "processing_inventory": Path(
            "storage/transcription-ocr/cvpr2026/r6/r6_processing_inventory.json"
        ),
        "material_payloads": Path("storage/transcription-ocr/cvpr2026/r6/material-payloads"),
        "raw_audio": Path("storage/transcription-ocr/cvpr2026/r6/raw-audio"),
        "speech_text": Path("storage/transcription-ocr/cvpr2026/r6/speech-text"),
        "paper_text": Path("storage/transcription-ocr/cvpr2026/r6/paper-text"),
        "slide_ocr": Path("storage/transcription-ocr/cvpr2026/r6/slide-ocr"),
        "auth_pages": Path("storage/transcription-ocr/cvpr2026/r6/auth-pages"),
        "logs": Path("storage/transcription-ocr/cvpr2026/r6/logs"),
        "session": Path("storage/transcription-ocr/cvpr2026/r6/session"),
        "final_receipt": Path(
            "storage/workflow-runs/cvpr2026-live-production-r6-expanded-media/"
            "stage-05-transcription-ocr-r6.json"
        ),
    }
)

R6_RECEIPT_KEYS = (
    "recovery_inventory",
    "audio_recovery_receipt",
    "material_discovery_receipt",
    "material_acquisition_receipt",
    "cost_estimate_receipt",
    "benchmark_receipt",
    "provider_receipts",
    "budget_ledger",
)
R6_MANIFEST_FAMILY_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "r6_recovery_inventory": "recovery_inventory",
        "r6_audio_recovery_receipt": "audio_recovery_receipt",
        "r6_audio_recovery_checkpoint_archive": "audio_recovery_archive_root",
        "r6_cli_decision_successor": "cli_decision_successor_root",
        "r6_material_discovery_receipt": "material_discovery_receipt",
        "r6_material_acquisition_receipt": "material_acquisition_receipt",
        "r6_material_payloads": "material_payloads",
        "r6_cost_estimate_receipt": "cost_estimate_receipt",
        "r6_benchmark_receipt": "benchmark_receipt",
        "r6_provider_receipts": "provider_receipts",
        "r6_budget_ledger": "budget_ledger",
        "r6_processing_inventory": "processing_inventory",
        "r6_speech_text": "speech_text",
        "r6_paper_text": "paper_text",
        "r6_slide_ocr": "slide_ocr",
        "r6_final_receipt": "final_receipt",
        "r6_raw_audio_cache": "raw_audio",
        "r6_auth_page_cache": "auth_pages",
        "r6_logs": "logs",
        "r6_session_cache": "session",
    }
)


def repository_root() -> Path:
    """Return the lexical repository root without accepting caller input.

    ``production()`` validates this lexical anchor before any caller can turn
    it into a realpath.  Do not use ``resolve`` here: resolving the anchor
    would give a repository-root symlink authority before the no-follow walk
    below can reject it.
    """

    return Path(__file__).parents[3]


@dataclass(frozen=True)
class _R6ProductionWorkspaceAnchor:
    """The lexical, descriptor-verified production root identities.

    We intentionally retain device/inode identity rather than a long-lived
    descriptor.  Every production consumer reopens the same lexical three
    component chain using ``O_NOFOLLOW`` and compares identities before it can
    derive a contract path.  That is equivalent to retaining an
    unredirectable descriptor identity without leaking a descriptor through a
    process-lifetime contract object.
    """

    repository_root: Path
    conferences_root: Path
    workspace: Path
    repository_identity: tuple[int, int]
    conferences_identity: tuple[int, int]
    workspace_identity: tuple[int, int]


def _r6_anchor_directory_flags() -> int:
    """Return the fail-closed flags needed for a production anchor walk."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if (
        not isinstance(no_follow, int)
        or not isinstance(directory, int)
        or no_follow <= 0
        or directory <= 0
        or os.open not in os.supports_dir_fd
    ):
        raise R6ArtifactContractError("r6 production anchor no-follow containment is unavailable")
    return os.O_RDONLY | directory | no_follow


def _r6_anchor_identity(metadata: os.stat_result) -> tuple[int, int]:
    return (int(metadata.st_dev), int(metadata.st_ino))


def _r6_regular_descriptor_digest(descriptor: int) -> str:
    """Hash one retained regular descriptor without granting it path authority.

    The descriptor is opened through an already-verified parent.  Rewinding it
    is intentional: the only executable consumer is a child process, and its
    dispatch target always starts at offset zero.  A digest makes an in-place
    content mutation fail closed even when the device/inode remains stable.
    """

    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return digest.hexdigest()
    except OSError as exc:
        raise R6ArtifactContractError("r6 fixed downloader cannot be hashed safely") from exc


def _assert_r6_anchor_directory(metadata: os.stat_result, *, label: str) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise R6ArtifactContractError(f"r6 production anchor {label} is a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise R6ArtifactContractError(f"r6 production anchor {label} is not a directory")


def _r6_anchor_lstat(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise R6ArtifactContractError(f"r6 production anchor {label} is unavailable") from exc
    _assert_r6_anchor_directory(metadata, label=label)
    return metadata


def _open_r6_anchor_directory(
    *,
    parent_fd: int | None,
    component: Path | str,
    expected: os.stat_result,
    label: str,
) -> int:
    """Open one already-lstat'd anchor node and pin its identity."""

    flags = _r6_anchor_directory_flags()
    try:
        if parent_fd is None:
            descriptor = os.open(component, flags)
        else:
            descriptor = os.open(component, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise R6ArtifactContractError(
            f"r6 production anchor {label} cannot be opened safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        _assert_r6_anchor_directory(opened, label=label)
        if _r6_anchor_identity(opened) != _r6_anchor_identity(expected):
            raise R6ArtifactContractError(f"r6 production anchor {label} changed during validation")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _capture_r6_production_workspace_anchor(repository: Path) -> _R6ProductionWorkspaceAnchor:
    """Validate repo → conferences → cvpr2026 before any path resolution.

    This deliberately accepts no alternate workspace and performs neither
    ``Path.resolve`` nor ``realpath``.  The lstat checks provide lexical
    rejection of links and the descriptor walk prevents a replacement race
    between inspection and use.
    """

    repository = Path(repository)
    if not repository.is_absolute():
        raise R6ArtifactContractError("r6 production repository anchor must be absolute")
    conferences = repository / "conferences"
    workspace = conferences / "cvpr2026"
    repository_metadata = _r6_anchor_lstat(repository, label="repository root")
    repository_fd = _open_r6_anchor_directory(
        parent_fd=None,
        component=repository,
        expected=repository_metadata,
        label="repository root",
    )
    try:
        try:
            conferences_metadata = os.lstat("conferences", dir_fd=repository_fd)
        except (OSError, TypeError) as exc:
            raise R6ArtifactContractError(
                "r6 production anchor conferences is unavailable"
            ) from exc
        _assert_r6_anchor_directory(conferences_metadata, label="conferences")
        conferences_fd = _open_r6_anchor_directory(
            parent_fd=repository_fd,
            component="conferences",
            expected=conferences_metadata,
            label="conferences",
        )
        try:
            try:
                workspace_metadata = os.lstat("cvpr2026", dir_fd=conferences_fd)
            except (OSError, TypeError) as exc:
                raise R6ArtifactContractError(
                    "r6 production anchor cvpr2026 workspace is unavailable"
                ) from exc
            _assert_r6_anchor_directory(workspace_metadata, label="cvpr2026 workspace")
            workspace_fd = _open_r6_anchor_directory(
                parent_fd=conferences_fd,
                component="cvpr2026",
                expected=workspace_metadata,
                label="cvpr2026 workspace",
            )
            try:
                return _R6ProductionWorkspaceAnchor(
                    repository_root=repository,
                    conferences_root=conferences,
                    workspace=workspace,
                    repository_identity=_r6_anchor_identity(repository_metadata),
                    conferences_identity=_r6_anchor_identity(conferences_metadata),
                    workspace_identity=_r6_anchor_identity(workspace_metadata),
                )
            finally:
                os.close(workspace_fd)
        finally:
            os.close(conferences_fd)
    finally:
        os.close(repository_fd)


def _require_r6_anchor_identity(
    metadata: os.stat_result, *, expected: tuple[int, int], label: str
) -> None:
    """Require an opened or inspected anchor node to retain its captured identity."""

    _assert_r6_anchor_directory(metadata, label=label)
    if _r6_anchor_identity(metadata) != expected:
        raise R6ArtifactContractError(
            f"r6 production anchor {label} changed after validation"
        )


def _open_r6_retained_production_workspace(
    anchor: _R6ProductionWorkspaceAnchor,
) -> int:
    """Return a fresh descriptor for the original fixed production workspace.

    A batch lease deliberately carries no path authority.  Each raw-audio
    operation must reopen the original lexical repository chain with
    ``O_NOFOLLOW`` and compare every inode/device pair to the identities
    captured at contract construction.  The returned descriptor, rather than
    ``anchor.workspace``, is the authority used for the subsequent component
    traversal.
    """

    repository_fd, conferences_fd, workspace_fd = _open_r6_retained_production_workspace_chain(
        anchor
    )
    try:
        return workspace_fd
    finally:
        os.close(conferences_fd)
        os.close(repository_fd)


def _open_r6_retained_production_workspace_chain(
    anchor: _R6ProductionWorkspaceAnchor,
) -> tuple[int, int, int]:
    """Open and retain the fixed production repository-to-workspace chain.

    The scoped r6 owner context needs all three descriptors to remain live
    while it validates and dispatches a fixed child.  Per-raw-operation
    callers continue to use the one-workspace-descriptor helper above.
    """

    repository_metadata = _r6_anchor_lstat(anchor.repository_root, label="repository root")
    _require_r6_anchor_identity(
        repository_metadata,
        expected=anchor.repository_identity,
        label="repository root",
    )
    repository_fd = _open_r6_anchor_directory(
        parent_fd=None,
        component=anchor.repository_root,
        expected=repository_metadata,
        label="repository root",
    )
    try:
        _require_r6_anchor_identity(
            os.fstat(repository_fd),
            expected=anchor.repository_identity,
            label="repository root",
        )
        try:
            conferences_metadata = os.lstat("conferences", dir_fd=repository_fd)
        except (OSError, TypeError) as exc:
            raise R6ArtifactContractError(
                "r6 production anchor conferences is unavailable"
            ) from exc
        _require_r6_anchor_identity(
            conferences_metadata,
            expected=anchor.conferences_identity,
            label="conferences",
        )
        conferences_fd = _open_r6_anchor_directory(
            parent_fd=repository_fd,
            component="conferences",
            expected=conferences_metadata,
            label="conferences",
        )
        try:
            _require_r6_anchor_identity(
                os.fstat(conferences_fd),
                expected=anchor.conferences_identity,
                label="conferences",
            )
            try:
                workspace_metadata = os.lstat("cvpr2026", dir_fd=conferences_fd)
            except (OSError, TypeError) as exc:
                raise R6ArtifactContractError(
                    "r6 production anchor cvpr2026 workspace is unavailable"
                ) from exc
            _require_r6_anchor_identity(
                workspace_metadata,
                expected=anchor.workspace_identity,
                label="cvpr2026 workspace",
            )
            workspace_fd = _open_r6_anchor_directory(
                parent_fd=conferences_fd,
                component="cvpr2026",
                expected=workspace_metadata,
                label="cvpr2026 workspace",
            )
            try:
                _require_r6_anchor_identity(
                    os.fstat(workspace_fd),
                    expected=anchor.workspace_identity,
                    label="cvpr2026 workspace",
                )
            except Exception:
                os.close(workspace_fd)
                raise
            return repository_fd, conferences_fd, workspace_fd
        except Exception:
            os.close(conferences_fd)
            raise
    except Exception:
        os.close(repository_fd)
        raise


def canonical_r6_json(payload: object) -> str:
    """Serialize finite, sanitized control-plane data deterministically."""

    try:
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise R6ArtifactContractError("r6 control data must be finite JSON") from exc
    return rendered


def r6_digest(payload: object) -> str:
    """Return the one canonical digest used by r6 control-plane records."""

    return hashlib.sha256(canonical_r6_json(payload).encode("utf-8")).hexdigest()


def r6_audio_policy_digest() -> str:
    """Return the immutable digest for the public-only audio recovery policy."""

    return r6_digest(dict(R6_PUBLIC_AUDIO_POLICY))


def _require_r6_digest(value: object, *, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise R6ArtifactContractError(f"{label} must be a SHA-256 digest")
    return value


def r6_audio_checkpoint_archive_root_relative_path() -> Path:
    """Return the sole fixed archive root for the narrow r6 remediation."""

    return R6_ARTIFACT_PATHS["audio_recovery_archive_root"]


def archive_r6_audio_checkpoint_relative_path(original_sha256: object) -> Path:
    """Build one digest-addressed archive leaf without caller path input."""

    digest = _require_r6_digest(original_sha256, label="original checkpoint digest")
    return r6_audio_checkpoint_archive_root_relative_path() / f"{digest}.json"


def _archive_leaf_parts(original_sha256: object) -> tuple[Path, Path, str]:
    """Validate the one archive root/leaf shape before filesystem access."""

    archive_root = r6_audio_checkpoint_archive_root_relative_path()
    archive_leaf = archive_r6_audio_checkpoint_relative_path(original_sha256)
    if (
        archive_root.is_absolute()
        or archive_leaf.is_absolute()
        or archive_leaf.parent != archive_root
        or not archive_root.parts
        or any(part in {"", ".", ".."} for part in archive_root.parts)
        or archive_leaf.name != f"{_require_r6_digest(original_sha256, label='original checkpoint digest')}.json"
    ):
        raise R6ArtifactContractError("r6 archive fixed leaf is invalid")
    return archive_root, archive_leaf, archive_leaf.name


def write_r6_audio_checkpoint_archive(
    *,
    contract: R6ArtifactContract,
    original_sha256: object,
    raw: bytes,
    _workspace_lease: object | None = None,
) -> Path:
    """Atomically create and byte-verify the sole r6 checkpoint archive leaf.

    This API accepts neither a root nor a target path.  It creates only the
    digest-derived leaf through no-follow directory descriptors and returns
    its fixed relative identity for sanitized lineage metadata.
    """

    if not isinstance(raw, bytes) or not raw:
        raise R6ArtifactContractError("r6 archive bytes are invalid")
    digest = _require_r6_digest(original_sha256, label="original checkpoint digest")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise R6ArtifactContractError("r6 archive bytes do not match the fixed digest")
    archive_root, archive_leaf, leaf_name = _archive_leaf_parts(digest)
    try:
        archive_fd = _open_r6_descriptor_directory(
            contract=contract,
            relative_directory=archive_root,
            create=True,
            _workspace_lease=_workspace_lease,
        )
    except R6ArtifactContractError as exc:
        raise R6ArtifactContractError(f"r6 archive containment failed: {exc}") from exc
    try:
        try:
            _write_or_verify_r6_immutable_regular_leaf(
                parent_fd=archive_fd,
                leaf_name=leaf_name,
                raw=raw,
            )
        except R6ArtifactContractError as exc:
            raise R6ArtifactContractError(f"r6 archive containment failed: {exc}") from exc
    finally:
        os.close(archive_fd)
    if (
        read_r6_audio_checkpoint_archive(
            contract=contract, original_sha256=digest, _workspace_lease=_workspace_lease
        )
        != raw
    ):
        raise R6ArtifactContractError("r6 archive bytes did not verify")
    return archive_leaf


def read_r6_audio_checkpoint_archive(
    *,
    contract: R6ArtifactContract,
    original_sha256: object,
    _workspace_lease: object | None = None,
) -> bytes:
    """Read only a regular, no-follow fixed archive leaf for successor lineage."""

    digest = _require_r6_digest(original_sha256, label="original checkpoint digest")
    archive_root, _archive_leaf, leaf_name = _archive_leaf_parts(digest)
    try:
        archive_fd = _open_r6_descriptor_directory(
            contract=contract,
            relative_directory=archive_root,
            create=False,
            _workspace_lease=_workspace_lease,
        )
    except R6ArtifactContractError as exc:
        raise R6ArtifactContractError(f"r6 archive containment failed: {exc}") from exc
    try:
        try:
            raw = _read_r6_single_link_regular_leaf(
                parent_fd=archive_fd,
                leaf_name=leaf_name,
                missing_ok=False,
            )
        except R6ArtifactContractError as exc:
            raise R6ArtifactContractError(f"r6 archive containment failed: {exc}") from exc
        assert raw is not None
        return raw
    finally:
        os.close(archive_fd)


def _require_finite_positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R6ArtifactContractError(f"{label} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise R6ArtifactContractError(f"{label} must be finite and positive")
    return numeric


def canonical_r6_queue(
    *, speech_ids: Sequence[str], ocr_ids: Sequence[str]
) -> tuple[dict[str, object], str]:
    """Build a deterministic declared paid queue without filesystem discovery.

    Queue order is part of the authorization.  We retain the receipt-declared
    order rather than sorting it, rejecting duplicate/corrupt identities before
    a later reservation can consume them.
    """

    queues: dict[str, list[str]] = {}
    for task_id, values in (
        ("transcription.session_speech", speech_ids),
        ("ocr.slide_image_parse", ocr_ids),
    ):
        if isinstance(values, (str, bytes)):
            raise R6ArtifactContractError("r6 queue must be an explicit sequence")
        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value or any(ch in value for ch in "\\/\x00"):
                raise R6ArtifactContractError("r6 queue identity is unsafe")
            normalized.append(value)
        if len(set(normalized)) != len(normalized):
            raise R6ArtifactContractError("r6 queue identities must be unique")
        queues[task_id] = normalized
    queue: dict[str, object] = {"schema_version": 1, "queues": queues}
    return queue, r6_digest(queue)


def _entry_fingerprint_payload(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        key: entry.get(key)
        for key in (
            "run_id",
            "task_id",
            "dispatch_kind",
            "identity",
            "ordinal",
            "maximum_charge_usd",
            "queue_digest",
        )
    }


def build_r6_dispatch_schedule(
    *,
    run_id: str,
    queue: Mapping[str, object],
    queue_digest: str,
    speech_total_usd: float,
    ocr_total_usd: float,
) -> dict[str, object]:
    """Create the complete ordered paid-call authorization for one r6 run."""

    if run_id != R6_RUN_ID:
        raise R6ArtifactContractError("dispatch schedule run id drifted")
    _require_r6_digest(queue_digest, label="queue digest")
    queues = queue.get("queues")
    if not isinstance(queues, Mapping):
        raise R6ArtifactContractError("r6 queue is malformed")
    speech = queues.get("transcription.session_speech")
    ocr = queues.get("ocr.slide_image_parse")
    if not isinstance(speech, list) or not isinstance(ocr, list):
        raise R6ArtifactContractError("r6 queue lacks governed lanes")
    speech_amount = _require_finite_positive(speech_total_usd, label="speech total") if speech else 0.0
    ocr_amount = _require_finite_positive(ocr_total_usd, label="ocr total") if ocr else 0.0
    # Every nonempty paid lane gets one explicit benchmark *before any item*
    # may be scheduled.  This leading fixed task-order prefix means the live
    # owner can run speech and OCR preflight calls in distinct invocations,
    # while a later full invocation has no benchmark branch at all.
    lane_entries: list[tuple[str, list[tuple[str, str, float]]]] = []
    for task_id, values, lane_total in (
        ("transcription.session_speech", speech, speech_amount),
        ("ocr.slide_image_parse", ocr, ocr_amount),
    ):
        if not values:
            continue
        count = len(values) + 1
        base = lane_total / count
        lane: list[tuple[str, str, float]] = [("benchmark", f"benchmark:{task_id}", base)]
        for offset, value in enumerate(values, start=1):
            amount = lane_total - base * (count - 1) if offset == count - 1 else base
            lane.append(("item", f"item:{value}", amount))
        lane_entries.append((task_id, lane))

    ordered_entries: list[tuple[str, str, str, float]] = []
    ordered_entries.extend(
        (task_id, kind, identity, amount)
        for task_id, lane in lane_entries
        for kind, identity, amount in lane[:1]
    )
    ordered_entries.extend(
        (task_id, kind, identity, amount)
        for task_id, lane in lane_entries
        for kind, identity, amount in lane[1:]
    )
    entries: list[dict[str, object]] = []
    for ordinal, (task_id, kind, identity, amount) in enumerate(ordered_entries):
        entry: dict[str, object] = {
            "run_id": run_id,
            "task_id": task_id,
            "dispatch_kind": kind,
            "identity": identity,
            "ordinal": ordinal,
            "maximum_charge_usd": amount,
            "queue_digest": queue_digest,
        }
        entry["entry_fingerprint"] = r6_digest(_entry_fingerprint_payload(entry))
        entries.append(entry)
    totals = {
        "all_in_usd": sum(float(entry["maximum_charge_usd"]) for entry in entries),
        "speech_usd": sum(
            float(entry["maximum_charge_usd"])
            for entry in entries
            if entry["task_id"] == "transcription.session_speech"
        ),
        "ocr_usd": sum(
            float(entry["maximum_charge_usd"])
            for entry in entries
            if entry["task_id"] == "ocr.slide_image_parse"
        ),
    }
    if (
        totals["all_in_usd"] > R6_ALL_IN_CAP_USD
        or totals["speech_usd"] > R6_SPEECH_CAP_USD
        or totals["ocr_usd"] > R6_OCR_CAP_USD
    ):
        raise R6ArtifactContractError("r6 dispatch schedule breaches a finite cap")
    unsigned = {
        "schema_version": 1,
        "run_id": run_id,
        "queue": dict(queue),
        "queue_digest": queue_digest,
        "entries": entries,
        "totals": totals,
        "caps": {
            "all_in_usd": R6_ALL_IN_CAP_USD,
            "speech_usd": R6_SPEECH_CAP_USD,
            "ocr_usd": R6_OCR_CAP_USD,
        },
    }
    schedule_digest = r6_digest(unsigned)
    return {**unsigned, "dispatch_schedule_digest": schedule_digest}


def validate_r6_dispatch_schedule(
    schedule: Mapping[str, object], *, expected_run_id: str = R6_RUN_ID
) -> dict[str, object]:
    """Reject a changed, partial, replayable, or over-cap schedule."""

    if not isinstance(schedule, Mapping) or schedule.get("run_id") != expected_run_id:
        raise R6ArtifactContractError("r6 dispatch schedule run id is invalid")
    supplied_digest = _require_r6_digest(
        schedule.get("dispatch_schedule_digest"), label="dispatch schedule digest"
    )
    unsigned = {key: value for key, value in schedule.items() if key != "dispatch_schedule_digest"}
    if r6_digest(unsigned) != supplied_digest:
        raise R6ArtifactContractError("r6 dispatch schedule digest drifted")
    queue = schedule.get("queue")
    queue_digest = _require_r6_digest(schedule.get("queue_digest"), label="queue digest")
    if not isinstance(queue, Mapping) or r6_digest(queue) != queue_digest:
        raise R6ArtifactContractError("r6 queue digest drifted")
    entries = schedule.get("entries")
    totals = schedule.get("totals")
    caps = schedule.get("caps")
    if not isinstance(entries, list) or not isinstance(totals, Mapping) or not isinstance(caps, Mapping):
        raise R6ArtifactContractError("r6 dispatch schedule is malformed")
    declared_queues = queue.get("queues")
    if not isinstance(declared_queues, Mapping):
        raise R6ArtifactContractError("r6 dispatch queue is malformed")
    expected_identities: list[tuple[str, str, str]] = []
    item_identities: list[tuple[str, str, str]] = []
    for task_id in ("transcription.session_speech", "ocr.slide_image_parse"):
        values = declared_queues.get(task_id)
        if not isinstance(values, list):
            raise R6ArtifactContractError("r6 dispatch queue lane is malformed")
        if values:
            expected_identities.append((task_id, "benchmark", f"benchmark:{task_id}"))
            item_identities.extend((task_id, "item", f"item:{value}") for value in values)
    expected_identities.extend(item_identities)
    if len(entries) != len(expected_identities):
        raise R6ArtifactContractError("r6 dispatch schedule omitted or added an authorized entry")
    seen: set[tuple[str, str]] = set()
    calculated = {"all_in_usd": 0.0, "speech_usd": 0.0, "ocr_usd": 0.0}
    for ordinal, raw in enumerate(entries):
        if not isinstance(raw, Mapping) or raw.get("ordinal") != ordinal:
            raise R6ArtifactContractError("r6 dispatch schedule order drifted")
        if raw.get("run_id") != expected_run_id or raw.get("queue_digest") != queue_digest:
            raise R6ArtifactContractError("r6 dispatch schedule identity drifted")
        task_id = raw.get("task_id")
        kind = raw.get("dispatch_kind")
        identity = raw.get("identity")
        if task_id not in {"transcription.session_speech", "ocr.slide_image_parse"}:
            raise R6ArtifactContractError("r6 dispatch schedule task is invalid")
        if kind not in {"benchmark", "item"} or not isinstance(identity, str):
            raise R6ArtifactContractError("r6 dispatch schedule entry is invalid")
        if (task_id, kind, identity) != expected_identities[ordinal]:
            raise R6ArtifactContractError("r6 dispatch schedule queue/order drifted")
        if (str(task_id), identity) in seen:
            raise R6ArtifactContractError("r6 dispatch schedule entry is duplicated")
        seen.add((str(task_id), identity))
        amount = _require_finite_positive(raw.get("maximum_charge_usd"), label="schedule amount")
        if raw.get("entry_fingerprint") != r6_digest(_entry_fingerprint_payload(raw)):
            raise R6ArtifactContractError("r6 dispatch schedule entry fingerprint drifted")
        calculated["all_in_usd"] += amount
        calculated["speech_usd" if task_id == "transcription.session_speech" else "ocr_usd"] += amount
    for key, value in calculated.items():
        if not isinstance(totals.get(key), (int, float)) or not math.isclose(
            float(totals[key]), value, rel_tol=0.0, abs_tol=1e-9
        ):
            raise R6ArtifactContractError("r6 dispatch schedule totals drifted")
    expected_caps = {
        "all_in_usd": R6_ALL_IN_CAP_USD,
        "speech_usd": R6_SPEECH_CAP_USD,
        "ocr_usd": R6_OCR_CAP_USD,
    }
    if dict(caps) != expected_caps or any(calculated[key] > expected_caps[key] for key in calculated):
        raise R6ArtifactContractError("r6 dispatch schedule cap drifted")
    return dict(schedule)


def build_r6_execution_envelope(
    *,
    kind: str,
    status: str,
    material_policy_digest: str,
    audio_policy_digest: str,
    predecessor_bindings: Mapping[str, str],
    schedule: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the common sanitized estimate/benchmark/provider/no-paid schema."""

    if kind not in {"estimate", "benchmark", "provider", "no_paid_lane"}:
        raise R6ArtifactContractError("r6 envelope kind is invalid")
    if status not in {"succeeded", "terminal", "not_applicable", "blocked"}:
        raise R6ArtifactContractError("r6 envelope status is invalid")
    _require_r6_digest(material_policy_digest, label="material policy digest")
    _require_r6_digest(audio_policy_digest, label="audio policy digest")
    bindings = {str(key): _require_r6_digest(value, label="predecessor digest") for key, value in predecessor_bindings.items()}
    if schedule is None:
        if kind != "no_paid_lane":
            raise R6ArtifactContractError("paid r6 envelope requires a dispatch schedule")
        queue_digest = r6_digest({"schema_version": 1, "queues": {"transcription.session_speech": [], "ocr.slide_image_parse": []}})
        schedule_digest: str | None = None
        totals = {"all_in_usd": 0.0, "speech_usd": 0.0, "ocr_usd": 0.0}
    else:
        checked = validate_r6_dispatch_schedule(schedule)
        queue_digest = str(checked["queue_digest"])
        schedule_digest = str(checked["dispatch_schedule_digest"])
        totals = dict(checked["totals"])
    envelope: dict[str, object] = {
        "schema_version": R6_ENVELOPE_VERSION,
        "receipt_type": f"conference_pipeline.cvpr2026_r6_{kind}",
        "kind": kind,
        "status": status,
        "run_id": R6_RUN_ID,
        # Stable alias for existing receipt consumers.  It is intentionally
        # the material policy only; the separately named audio policy remains
        # independently bound below and can never be substituted for it.
        "policy_digest": material_policy_digest,
        "material_policy_digest": material_policy_digest,
        "audio_policy_digest": audio_policy_digest,
        "predecessor_bindings": bindings,
        "queue_digest": queue_digest,
        "dispatch_schedule_digest": schedule_digest,
        "dispatch_schedule": dict(schedule) if schedule is not None else None,
        "scheduled_totals": totals,
        "secrets_copied": False,
    }
    envelope["envelope_sha256"] = r6_digest(envelope)
    return envelope


@dataclass(frozen=True)
class R6ArtifactContract:
    """Fixed r6 identity and workspace-scoped artifact resolver."""

    workspace: Path
    fixture_injection: bool = False
    _production_anchor: _R6ProductionWorkspaceAnchor | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _workspace_lease: object = field(default_factory=object, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        supplied = Path(self.workspace)
        if self.fixture_injection:
            # Fixtures are intentionally allowed to create their workspace
            # after the contract exists, but may never gain authority by
            # resolving a direct workspace link first.
            fixture_workspace = Path(os.path.abspath(supplied))
            try:
                fixture_metadata = os.lstat(fixture_workspace)
            except FileNotFoundError:
                fixture_metadata = None
            except OSError as exc:
                raise R6ArtifactContractError("r6 fixture workspace is unavailable") from exc
            if fixture_metadata is not None:
                _assert_r6_anchor_directory(fixture_metadata, label="fixture workspace")
            object.__setattr__(self, "workspace", fixture_workspace)
            return

        repository = repository_root()
        expected = Path(repository) / R6_WORKSPACE_RELATIVE_PATH
        if supplied != expected:
            raise R6ArtifactContractError("production r6 contract must use the fixed CVPR workspace")
        anchor = _capture_r6_production_workspace_anchor(Path(repository))
        object.__setattr__(self, "workspace", anchor.workspace)
        object.__setattr__(self, "_production_anchor", anchor)

    @classmethod
    def production(cls) -> R6ArtifactContract:
        return cls(repository_root() / R6_WORKSPACE_RELATIVE_PATH)

    @classmethod
    def for_test_fixture(cls, workspace: Path) -> R6ArtifactContract:
        """Inject a non-production root exclusively for local test fixtures."""

        return cls(Path(workspace), fixture_injection=True)

    def _require_trusted_workspace_anchor(self) -> None:
        """Revalidate the retained production anchor before every consumer use."""

        if self.fixture_injection:
            try:
                metadata = os.lstat(self.workspace)
            except FileNotFoundError:
                return
            except OSError as exc:
                raise R6ArtifactContractError("r6 fixture workspace is unavailable") from exc
            _assert_r6_anchor_directory(metadata, label="fixture workspace")
            return

        anchor = self._production_anchor
        if anchor is None:  # pragma: no cover - frozen dataclass invariant
            raise R6ArtifactContractError("r6 production workspace anchor is unavailable")
        current = _capture_r6_production_workspace_anchor(anchor.repository_root)
        if (
            current.repository_root != anchor.repository_root
            or current.conferences_root != anchor.conferences_root
            or current.workspace != anchor.workspace
            or current.repository_identity != anchor.repository_identity
            or current.conferences_identity != anchor.conferences_identity
            or current.workspace_identity != anchor.workspace_identity
        ):
            raise R6ArtifactContractError("r6 production workspace anchor changed after validation")

    def ensure_workspace_trusted(self) -> Path:
        """Return the only workspace after its no-follow identity check."""

        self._require_trusted_workspace_anchor()
        return self.workspace

    def _acquire_workspace_lease(self) -> object:
        """Return an opaque batch token for a bounded internal loop.

        The token is never a retained path or descriptor authority.  It only
        identifies this contract.  The scoped owner context performs the one
        repository → conferences → workspace validation/open at its entry;
        every raw-audio operation independently performs a fresh equivalent
        validation immediately before descriptor-relative traversal.
        """

        return self._workspace_lease

    def _validate_workspace_lease(self, lease: object) -> None:
        """Reject a foreign opaque lease without granting a lexical path."""

        if lease is not self._workspace_lease:
            raise R6ArtifactContractError("r6 workspace lease does not match the fixed contract")

    def revalidate_workspace_lease(self, lease: object) -> None:
        """Freshly validate a lease anchor without exposing a workspace path.

        Compatibility helper for a one-operation check.  Owners that dispatch
        a child must instead use :meth:`workspace_lease_context`, whose
        descriptors remain open through child selection and dispatch.
        """

        with self.workspace_lease_context(lease):
            return None

    def workspace_lease_context(self, lease: object) -> R6WorkspaceLeaseContext:
        """Acquire one scoped, descriptor-backed owner capability.

        The returned context deliberately has no workspace-path accessor.  It
        is only valid for this contract's opaque lease and must be closed by
        its caller (normally through ``with``).  Raw payload helpers do not
        consume this capability: they retain their separate Plan-65 fresh
        descriptor validation at every access.
        """

        return R6WorkspaceLeaseContext.acquire(contract=self, lease=lease)

    def _open_workspace_for_lease(self, lease: object) -> tuple[int, Path]:
        """Open one fresh, descriptor-backed workspace access for a lease.

        Production accesses are rooted in the retained fixed-anchor inode
        chain.  Fixture accesses remain local-test-only but still use an
        ``O_NOFOLLOW`` descriptor, so callers share one ownership contract.
        The caller owns and must close the returned descriptor.
        """

        self._validate_workspace_lease(lease)
        if not self.fixture_injection:
            anchor = self._production_anchor
            if anchor is None:  # pragma: no cover - frozen dataclass invariant
                raise R6ArtifactContractError("r6 production workspace anchor is unavailable")
            return _open_r6_retained_production_workspace(anchor), anchor.workspace

        try:
            metadata = os.lstat(self.workspace)
        except OSError as exc:
            raise R6ArtifactContractError("r6 fixture workspace is unavailable") from exc
        _assert_r6_anchor_directory(metadata, label="fixture workspace")
        try:
            descriptor = os.open(self.workspace, _r6_anchor_directory_flags())
        except OSError as exc:
            raise R6ArtifactContractError("r6 fixture workspace cannot be opened safely") from exc
        try:
            _assert_r6_anchor_directory(os.fstat(descriptor), label="fixture workspace")
            if _r6_anchor_identity(os.fstat(descriptor)) != _r6_anchor_identity(metadata):
                raise R6ArtifactContractError("r6 fixture workspace changed during validation")
        except Exception:
            os.close(descriptor)
            raise
        return descriptor, self.workspace

    @property
    def run_id(self) -> str:
        return R6_RUN_ID

    @property
    def stage_id(self) -> str:
        return R6_STAGE_ID

    @property
    def r6_root(self) -> Path:
        return self.path("r6_root")

    @property
    def audio_root(self) -> Path:
        return self.path("raw_audio")

    @property
    def audio_checkpoint_archive_root(self) -> Path:
        return self.path("audio_recovery_archive_root")

    @property
    def material_payload_root(self) -> Path:
        return self.path("material_payloads")

    @property
    def speech_root(self) -> Path:
        return self.path("speech_text")

    @property
    def paper_root(self) -> Path:
        return self.path("paper_text")

    @property
    def ocr_root(self) -> Path:
        return self.path("slide_ocr")

    def relative_path(self, key: str) -> Path:
        try:
            return R6_ARTIFACT_PATHS[key]
        except KeyError as exc:
            raise R6ArtifactContractError(f"unknown r6 artifact key: {key}") from exc

    def path(self, key: str) -> Path:
        # This is a pure lexical derivation.  Entry-point consumers call
        # ``require_workspace``/``ensure_workspace_trusted`` before their
        # reader, writer, classifier, registry, or provider seam; repeatedly
        # reopening the anchor here would turn the fixed 2,002-row loop into
        # thousands of unnecessary descriptor walks.
        return self.workspace / self.relative_path(key)

    def require_workspace(self, workspace: Path) -> Path:
        self.ensure_workspace_trusted()
        candidate = Path(workspace)
        if not candidate.is_absolute():
            candidate = Path(os.path.abspath(candidate))
        if candidate != self.workspace:
            raise R6ArtifactContractError("r6 workspace does not match the fixed contract")
        return self.workspace

    def require_path(self, key: str, supplied: object) -> Path:
        self.ensure_workspace_trusted()
        if not isinstance(supplied, (str, Path)):
            raise R6ArtifactContractError("r6 artifact path is required")
        candidate = Path(supplied)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            cwd_relative = candidate.resolve()
            resolved = (
                cwd_relative
                if cwd_relative == self.path(key).resolve()
                else (self.workspace / candidate).resolve()
            )
        expected = self.path(key).resolve()
        if resolved != expected:
            raise R6ArtifactContractError(f"r6 {key} path does not match the fixed contract")
        return expected

    def require_under(self, key: str, supplied: object) -> tuple[Path, Path]:
        """Resolve one receipt-declared payload only below a named contract root."""

        self.ensure_workspace_trusted()
        if not isinstance(supplied, (str, Path)):
            raise R6ArtifactContractError("r6 payload path is required")
        value = Path(supplied)
        if value.is_absolute():
            raise R6ArtifactContractError("r6 payload declarations must be workspace-relative")
        resolved = (self.workspace / value).resolve()
        root = self.path(key).resolve()
        if resolved == root or root not in resolved.parents:
            raise R6ArtifactContractError("r6 payload path is outside its fixed artifact root")
        try:
            return resolved, resolved.relative_to(self.workspace)
        except ValueError as exc:  # Defensive containment assertion.
            raise R6ArtifactContractError("r6 payload path escapes the workspace") from exc


_R6_ACTIVE_LEASE_DESCRIPTOR_COUNT = 0
_ACTIVE_R6_EXECUTABLE_CAPABILITY: ContextVar[R6RepositoryExecutableCapability | None] = (
    ContextVar("active_r6_executable_capability", default=None)
)


def r6_active_workspace_lease_descriptor_count() -> int:
    """Return the deterministic count of live scoped-context descriptors.

    This is intentionally narrow observability for local lifecycle tests.  It
    does not expose a descriptor number, workspace path, or authority to
    callers.
    """

    return _R6_ACTIVE_LEASE_DESCRIPTOR_COUNT


def active_r6_repository_executable_capability() -> R6RepositoryExecutableCapability | None:
    """Return the current scoped child capability, if a caller owns one."""

    capability = _ACTIVE_R6_EXECUTABLE_CAPABILITY.get()
    if capability is not None:
        capability._require_active()
    return capability


def _retain_r6_context_descriptor() -> None:
    global _R6_ACTIVE_LEASE_DESCRIPTOR_COUNT
    _R6_ACTIVE_LEASE_DESCRIPTOR_COUNT += 1


def _release_r6_context_descriptor() -> None:
    global _R6_ACTIVE_LEASE_DESCRIPTOR_COUNT
    if _R6_ACTIVE_LEASE_DESCRIPTOR_COUNT <= 0:  # pragma: no cover - invariant guard
        raise R6ArtifactContractError("r6 scoped lease descriptor accounting underflow")
    _R6_ACTIVE_LEASE_DESCRIPTOR_COUNT -= 1


class R6WorkspaceLeaseContext:
    """Scoped descriptor authority for fixed-owner preflight and metadata.

    It retains production repository → ``conferences`` → workspace
    descriptors.  Immediately before a child dispatch it compares the current
    lexical anchor to those held identities; a replaced lexical chain is
    rejected while the original descriptors are still open.  The context is
    intentionally opaque: consumers cannot obtain a lexical workspace path or
    reuse a closed/foreign capability.
    """

    def __init__(
        self,
        *,
        contract: R6ArtifactContract,
        lease: object,
        repository_fd: int | None,
        conferences_fd: int | None,
        workspace_fd: int,
        fixture_workspace_identity: tuple[int, int] | None,
    ) -> None:
        self._contract = contract
        self._lease = lease
        self._repository_fd = repository_fd
        self._conferences_fd = conferences_fd
        self._workspace_fd = workspace_fd
        self._fixture_workspace_identity = fixture_workspace_identity
        self._closed = False
        self._executable_capabilities: set[R6RepositoryExecutableCapability] = set()

    @classmethod
    def acquire(cls, *, contract: R6ArtifactContract, lease: object) -> R6WorkspaceLeaseContext:
        """Open and identity-check one descriptor chain exactly once."""

        contract._validate_workspace_lease(lease)
        if contract.fixture_injection:
            workspace_fd, _workspace = contract._open_workspace_for_lease(lease)
            _retain_r6_context_descriptor()
            return cls(
                contract=contract,
                lease=lease,
                repository_fd=None,
                conferences_fd=None,
                workspace_fd=workspace_fd,
                fixture_workspace_identity=_r6_anchor_identity(os.fstat(workspace_fd)),
            )
        anchor = contract._production_anchor
        if anchor is None:  # pragma: no cover - frozen contract invariant
            raise R6ArtifactContractError("r6 production workspace anchor is unavailable")
        repository_fd, conferences_fd, workspace_fd = _open_r6_retained_production_workspace_chain(
            anchor
        )
        try:
            _retain_r6_context_descriptor()
            _retain_r6_context_descriptor()
            _retain_r6_context_descriptor()
            return cls(
                contract=contract,
                lease=lease,
                repository_fd=repository_fd,
                conferences_fd=conferences_fd,
                workspace_fd=workspace_fd,
                fixture_workspace_identity=None,
            )
        except Exception:
            os.close(workspace_fd)
            os.close(conferences_fd)
            os.close(repository_fd)
            _release_r6_context_descriptor()
            _release_r6_context_descriptor()
            _release_r6_context_descriptor()
            raise

    def __enter__(self) -> R6WorkspaceLeaseContext:
        self._require_active()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _require_active(self) -> None:
        if self._closed:
            raise R6ArtifactContractError("r6 scoped workspace lease is closed")
        self._contract._validate_workspace_lease(self._lease)

    def _assert_current_anchor(self) -> None:
        """Reject a changed lexical chain while the original descriptors live."""

        self._require_active()
        if self._contract.fixture_injection:
            try:
                current = os.lstat(self._contract.workspace)
            except OSError as exc:
                raise R6ArtifactContractError("r6 fixture workspace is unavailable") from exc
            _assert_r6_anchor_directory(current, label="fixture workspace")
            if _r6_anchor_identity(current) != self._fixture_workspace_identity:
                raise R6ArtifactContractError("r6 fixture workspace changed during scoped lease")
            if _r6_anchor_identity(os.fstat(self._workspace_fd)) != self._fixture_workspace_identity:
                raise R6ArtifactContractError("r6 fixture workspace descriptor changed during scoped lease")
            return
        anchor = self._contract._production_anchor
        if anchor is None or self._repository_fd is None or self._conferences_fd is None:
            raise R6ArtifactContractError("r6 production scoped workspace lease is unavailable")
        checks = (
            (anchor.repository_root, anchor.repository_identity, self._repository_fd, "repository root"),
            (anchor.conferences_root, anchor.conferences_identity, self._conferences_fd, "conferences"),
            (anchor.workspace, anchor.workspace_identity, self._workspace_fd, "cvpr2026 workspace"),
        )
        for path, expected, descriptor, label in checks:
            current = _r6_anchor_lstat(path, label=label)
            _require_r6_anchor_identity(current, expected=expected, label=label)
            _require_r6_anchor_identity(os.fstat(descriptor), expected=expected, label=label)

    def assert_child_dispatch_allowed(self) -> None:
        """Check the held authority immediately before a child seam."""

        self._assert_current_anchor()

    def open_fixed_yt_dlp_capability(self) -> R6RepositoryExecutableCapability:
        """Open the fixed executable only below the held repository descriptor."""

        self._assert_current_anchor()
        if self._contract.fixture_injection or self._repository_fd is None:
            raise R6ArtifactContractError(
                "r6 repository-bound executable capability is unavailable for fixture injection"
            )
        flags = _r6_anchor_directory_flags()
        try:
            venv_metadata = os.lstat(".venv", dir_fd=self._repository_fd)
            _assert_r6_anchor_directory(venv_metadata, label="r6 downloader venv")
            venv_fd = os.open(".venv", flags, dir_fd=self._repository_fd)
        except (OSError, TypeError) as exc:
            raise R6ArtifactContractError("r6 fixed downloader venv is unavailable") from exc
        try:
            opened_venv = os.fstat(venv_fd)
            _assert_r6_anchor_directory(opened_venv, label="r6 downloader venv")
            if _r6_anchor_identity(opened_venv) != _r6_anchor_identity(venv_metadata):
                raise R6ArtifactContractError("r6 fixed downloader venv changed during validation")
            try:
                bin_metadata = os.lstat("bin", dir_fd=venv_fd)
                _assert_r6_anchor_directory(bin_metadata, label="r6 downloader bin")
                bin_fd = os.open("bin", flags, dir_fd=venv_fd)
            except (OSError, TypeError) as exc:
                raise R6ArtifactContractError("r6 fixed downloader bin is unavailable") from exc
            try:
                opened_bin = os.fstat(bin_fd)
                _assert_r6_anchor_directory(opened_bin, label="r6 downloader bin")
                if _r6_anchor_identity(opened_bin) != _r6_anchor_identity(bin_metadata):
                    raise R6ArtifactContractError("r6 fixed downloader bin changed during validation")
                executable_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                executable_metadata = os.lstat("yt-dlp", dir_fd=bin_fd)
                if stat.S_ISLNK(executable_metadata.st_mode) or not stat.S_ISREG(
                    executable_metadata.st_mode
                ):
                    raise R6ArtifactContractError("r6 fixed downloader is unsafe")
                executable_fd = os.open("yt-dlp", executable_flags, dir_fd=bin_fd)
                try:
                    opened = os.fstat(executable_fd)
                    if not stat.S_ISREG(opened.st_mode) or not (opened.st_mode & 0o111):
                        raise R6ArtifactContractError("r6 fixed downloader is unavailable")
                    if _r6_anchor_identity(opened) != _r6_anchor_identity(executable_metadata):
                        raise R6ArtifactContractError("r6 fixed downloader changed during validation")
                    executable_mode = stat.S_IMODE(opened.st_mode)
                    executable_digest = _r6_regular_descriptor_digest(executable_fd)
                    os.set_inheritable(executable_fd, True)
                    self._assert_current_anchor()
                    _retain_r6_context_descriptor()
                    capability = R6RepositoryExecutableCapability(
                        context=self,
                        descriptor=executable_fd,
                        venv_identity=_r6_anchor_identity(venv_metadata),
                        venv_mode=stat.S_IMODE(venv_metadata.st_mode),
                        bin_identity=_r6_anchor_identity(bin_metadata),
                        bin_mode=stat.S_IMODE(bin_metadata.st_mode),
                        executable_identity=_r6_anchor_identity(opened),
                        executable_mode=executable_mode,
                        executable_digest=executable_digest,
                    )
                    self._executable_capabilities.add(capability)
                    return capability
                except Exception:
                    os.close(executable_fd)
                    raise
            finally:
                os.close(bin_fd)
        finally:
            os.close(venv_fd)

    def _release_executable_capability(self, capability: R6RepositoryExecutableCapability) -> None:
        # Keep the revoked object until this scoped owner exits.  A copied
        # ContextVar close can release its descriptor/accounting while leaving
        # the owner ContextVar token stale; the owner context must get one
        # final chance to reset that token before it disappears.
        self._executable_capabilities.add(capability)
        _release_r6_context_descriptor()

    def close(self) -> None:
        """Revoke every child capability and descriptor exactly once."""

        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for capability in tuple(self._executable_capabilities):
            try:
                capability.close()
            except BaseException as exc:  # pragma: no cover - hostile close failure
                first_error = first_error or exc
        for descriptor_name in ("_workspace_fd", "_conferences_fd", "_repository_fd"):
            descriptor = getattr(self, descriptor_name)
            if descriptor is None:
                continue
            setattr(self, descriptor_name, None)
            try:
                os.close(descriptor)
            except OSError as exc:  # pragma: no cover - hostile close failure
                first_error = first_error or exc
            finally:
                _release_r6_context_descriptor()
        if first_error is not None:
            raise R6ArtifactContractError("r6 scoped workspace lease could not close") from first_error


@dataclass(eq=False)
class R6RepositoryExecutableCapability:
    """One retained-anchor executable ticket, scoped to one owner lease.

    The ticket never accepts an executable path from a caller.  It binds the
    no-follow ``.venv/bin/yt-dlp`` walk's device/inode, mode, and digest.  On
    platforms where ``/dev/fd`` execution is not viable, it can mint exactly
    one *verified* fixed-path command from those held identities; the caller
    still cannot select a lexical target or a PATH entry.
    """

    context: R6WorkspaceLeaseContext
    descriptor: int
    venv_identity: tuple[int, int]
    venv_mode: int
    bin_identity: tuple[int, int]
    bin_mode: int
    executable_identity: tuple[int, int]
    executable_mode: int
    executable_digest: str
    _closed: bool = field(default=False, init=False, repr=False)
    _released: bool = field(default=False, init=False, repr=False)
    _activation_token: Token[R6RepositoryExecutableCapability | None] | None = field(
        default=None, init=False, repr=False
    )

    def __enter__(self) -> R6RepositoryExecutableCapability:
        self._require_active()
        if self._activation_token is not None or _ACTIVE_R6_EXECUTABLE_CAPABILITY.get() is not None:
            raise R6ArtifactContractError("r6 repository executable capability is already active")
        self._activation_token = _ACTIVE_R6_EXECUTABLE_CAPABILITY.set(self)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _require_active(self) -> None:
        if self._closed:
            raise R6ArtifactContractError("r6 repository executable capability is closed")
        self.context._require_active()

    @property
    def command_target(self) -> str:
        self.assert_dispatch_allowed()
        if os.name != "posix" or not os.path.isdir("/dev/fd"):
            raise R6ArtifactContractError("r6 descriptor-addressable executable is unavailable")
        return f"/dev/fd/{self.descriptor}"

    @property
    def inherited_fds(self) -> tuple[int, ...]:
        self._require_active()
        return (self.descriptor,)

    def assert_dispatch_allowed(self) -> None:
        self._require_active()
        self.context.assert_child_dispatch_allowed()
        self._assert_fixed_executable_identity()

    def _assert_fixed_executable_identity(self) -> None:
        """Revalidate every retained fixed-downloader identity before/after a child."""

        repository_fd = self.context._repository_fd
        anchor = self.context._contract._production_anchor
        if repository_fd is None or anchor is None:
            raise R6ArtifactContractError("r6 fixed downloader capability is unavailable")
        flags = _r6_anchor_directory_flags()
        try:
            venv_metadata = os.lstat(".venv", dir_fd=repository_fd)
            _assert_r6_anchor_directory(venv_metadata, label="r6 downloader venv")
            if (
                _r6_anchor_identity(venv_metadata) != self.venv_identity
                or stat.S_IMODE(venv_metadata.st_mode) != self.venv_mode
            ):
                raise R6ArtifactContractError("r6 fixed downloader venv changed")
            venv_fd = os.open(".venv", flags, dir_fd=repository_fd)
        except (OSError, TypeError) as exc:
            raise R6ArtifactContractError("r6 fixed downloader venv is unavailable") from exc
        try:
            opened_venv = os.fstat(venv_fd)
            _assert_r6_anchor_directory(opened_venv, label="r6 downloader venv")
            if (
                _r6_anchor_identity(opened_venv) != self.venv_identity
                or stat.S_IMODE(opened_venv.st_mode) != self.venv_mode
            ):
                raise R6ArtifactContractError("r6 fixed downloader venv changed")
            try:
                bin_metadata = os.lstat("bin", dir_fd=venv_fd)
                _assert_r6_anchor_directory(bin_metadata, label="r6 downloader bin")
                if (
                    _r6_anchor_identity(bin_metadata) != self.bin_identity
                    or stat.S_IMODE(bin_metadata.st_mode) != self.bin_mode
                ):
                    raise R6ArtifactContractError("r6 fixed downloader bin changed")
                bin_fd = os.open("bin", flags, dir_fd=venv_fd)
            except (OSError, TypeError) as exc:
                raise R6ArtifactContractError("r6 fixed downloader bin is unavailable") from exc
            try:
                opened_bin = os.fstat(bin_fd)
                _assert_r6_anchor_directory(opened_bin, label="r6 downloader bin")
                if (
                    _r6_anchor_identity(opened_bin) != self.bin_identity
                    or stat.S_IMODE(opened_bin.st_mode) != self.bin_mode
                ):
                    raise R6ArtifactContractError("r6 fixed downloader bin changed")
                leaf_metadata = os.lstat("yt-dlp", dir_fd=bin_fd)
                if stat.S_ISLNK(leaf_metadata.st_mode) or not stat.S_ISREG(leaf_metadata.st_mode):
                    raise R6ArtifactContractError("r6 fixed downloader is unsafe")
                if (
                    _r6_anchor_identity(leaf_metadata) != self.executable_identity
                    or stat.S_IMODE(leaf_metadata.st_mode) != self.executable_mode
                ):
                    raise R6ArtifactContractError("r6 fixed downloader changed")
                leaf_fd = os.open("yt-dlp", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=bin_fd)
                try:
                    opened_leaf = os.fstat(leaf_fd)
                    held_leaf = os.fstat(self.descriptor)
                    if (
                        not stat.S_ISREG(opened_leaf.st_mode)
                        or not stat.S_ISREG(held_leaf.st_mode)
                        or _r6_anchor_identity(opened_leaf) != self.executable_identity
                        or _r6_anchor_identity(held_leaf) != self.executable_identity
                        or stat.S_IMODE(opened_leaf.st_mode) != self.executable_mode
                        or stat.S_IMODE(held_leaf.st_mode) != self.executable_mode
                        or _r6_regular_descriptor_digest(leaf_fd) != self.executable_digest
                        or _r6_regular_descriptor_digest(self.descriptor) != self.executable_digest
                    ):
                        raise R6ArtifactContractError("r6 fixed downloader changed")
                finally:
                    os.close(leaf_fd)
            finally:
                os.close(bin_fd)
        finally:
            os.close(venv_fd)

    def verified_fixed_path_command(self, command: Sequence[str]) -> list[str]:
        """Mint the sole Darwin fallback command from this retained ticket.

        This is deliberately unavailable outside the capability and rejects a
        modified command head.  The returned lexical spelling is only a
        platform dispatch mechanism; both anchor and leaf identities are
        checked immediately before it is produced and again by the caller
        before its child result may be accepted.
        """

        if os.name != "posix" or sys.platform != "darwin":
            raise R6ArtifactContractError("r6 fixed-path downloader dispatch is unavailable")
        expected = f"/dev/fd/{self.descriptor}"
        if not command or command[0] != expected:
            raise R6ArtifactContractError("r6 fixed downloader dispatch command is invalid")
        self.assert_dispatch_allowed()
        anchor = self.context._contract._production_anchor
        if anchor is None:  # pragma: no cover - production-only invariant
            raise R6ArtifactContractError("r6 fixed downloader capability is unavailable")
        return [str(anchor.repository_root / ".venv" / "bin" / "yt-dlp"), *command[1:]]

    def close(self) -> None:
        """Revoke descriptor and ContextVar ownership exactly once.

        ``ContextVar.reset`` intentionally rejects a token from a copied
        context.  That lifecycle error must never skip descriptor/accounting
        release.  The stale owner value remains unusable because this ticket
        is revoked before cleanup begins.
        """

        if self._released:
            if self._activation_token is not None:
                try:
                    _ACTIVE_R6_EXECUTABLE_CAPABILITY.reset(self._activation_token)
                except (LookupError, RuntimeError, ValueError) as exc:
                    raise R6ArtifactContractError(
                        "r6 repository executable capability close failed"
                    ) from exc
                self._activation_token = None
            return
        self._closed = True
        first_error: BaseException | None = None
        activation_reset_failed = False
        try:
            if self._activation_token is not None:
                try:
                    _ACTIVE_R6_EXECUTABLE_CAPABILITY.reset(self._activation_token)
                except (LookupError, RuntimeError, ValueError) as exc:
                    first_error = exc
                    activation_reset_failed = True
        finally:
            # A token created in another Context cannot be reset here.  Keep
            # it solely so the original owner context can clear its stale,
            # already-revoked value during its own deterministic close.
            if not activation_reset_failed:
                self._activation_token = None
            descriptor = self.descriptor
            self.descriptor = -1
            try:
                if descriptor >= 0:
                    os.close(descriptor)
            except OSError as exc:  # pragma: no cover - hostile close failure
                first_error = first_error or exc
            finally:
                self._released = True
                try:
                    self.context._release_executable_capability(self)
                except BaseException as exc:  # pragma: no cover - invariant fault
                    first_error = first_error or exc
        if first_error is not None:
            raise R6ArtifactContractError("r6 repository executable capability close failed") from first_error


_RAW_AUDIO_FORBIDDEN_TOKENS = ("*", "?", "[", "\\")


def _raw_audio_directory_flags() -> int:
    """Return the fail-closed descriptor flags for raw-audio traversal."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if (
        not isinstance(no_follow, int)
        or not isinstance(directory, int)
        or no_follow <= 0
        or directory <= 0
        or os.open not in os.supports_dir_fd
    ):
        raise R6ArtifactContractError("r6 raw audio no-follow containment is unavailable")
    return os.O_RDONLY | directory | no_follow


def _require_raw_audio_realpath_containment(*, workspace: Path, candidate: Path) -> None:
    """Reject a raw-audio component whose real path has escaped the workspace."""

    workspace_real = Path(os.path.realpath(workspace))
    candidate_real = Path(os.path.realpath(candidate))
    if candidate_real != workspace_real and workspace_real not in candidate_real.parents:
        raise R6ArtifactContractError("r6 raw audio path escapes the fixed workspace")


def _raw_audio_relative_path(
    *, contract: R6ArtifactContract, local_media_path: object
) -> Path:
    """Canonicalize a declaration lexically, never by granting ``resolve`` authority."""

    if not isinstance(local_media_path, str) or not local_media_path:
        raise R6ArtifactContractError("r6 raw audio local_media_path is required")
    if "\x00" in local_media_path or "://" in local_media_path or any(
        token in local_media_path for token in _RAW_AUDIO_FORBIDDEN_TOKENS
    ):
        raise R6ArtifactContractError("r6 raw audio declaration is unsafe")
    supplied = Path(local_media_path)
    if supplied.is_absolute():
        try:
            relative = supplied.relative_to(contract.workspace)
        except ValueError as exc:
            raise R6ArtifactContractError(
                "r6 raw audio absolute declaration escapes the fixed workspace"
            ) from exc
    else:
        relative = supplied
    if (
        not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative == contract.relative_path("raw_audio")
    ):
        raise R6ArtifactContractError("r6 raw audio declaration is not a payload leaf")
    raw_root = contract.relative_path("raw_audio")
    if relative.parts[: len(raw_root.parts)] != raw_root.parts:
        raise R6ArtifactContractError("r6 raw audio declaration is outside the fixed raw-audio root")
    return relative


def _raw_audio_lstat(*, parent_fd: int, component: str) -> os.stat_result | None:
    try:
        return os.lstat(component, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except (OSError, TypeError) as exc:
        raise R6ArtifactContractError("r6 raw audio component cannot be inspected") from exc


def _require_raw_audio_directory(metadata: os.stat_result, *, label: str) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise R6ArtifactContractError(f"r6 raw audio {label} is a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise R6ArtifactContractError(f"r6 raw audio {label} is not a directory")


def _open_r6_raw_audio_parent(
    *,
    contract: R6ArtifactContract,
    relative: Path,
    create: bool,
    _workspace_lease: object | None = None,
    _workspace_fd: int | None = None,
    _validated_workspace: Path | None = None,
) -> tuple[int, Path, str]:
    """Walk only descriptor-relative raw-audio parents and return their leaf fd."""

    try:
        directory_flags = _raw_audio_directory_flags()
    except Exception:
        if _workspace_fd is not None:
            os.close(_workspace_fd)
        raise
    if _workspace_fd is not None:
        if _validated_workspace is None:
            os.close(_workspace_fd)
            raise R6ArtifactContractError("r6 raw audio validated workspace is unavailable")
        # The caller has transferred ownership of a descriptor freshly opened
        # by ``_open_workspace_for_lease``.  Do not inspect or reopen the
        # lexical workspace on this branch.
        current_fd = _workspace_fd
        workspace = _validated_workspace
    elif _workspace_lease is None:
        workspace = contract.ensure_workspace_trusted()
        try:
            workspace_metadata = os.lstat(workspace)
        except OSError as exc:
            raise R6ArtifactContractError("r6 raw audio workspace is unavailable") from exc
        _require_raw_audio_directory(workspace_metadata, label="workspace root")
        _require_raw_audio_realpath_containment(workspace=workspace, candidate=workspace)
        try:
            current_fd = os.open(workspace, directory_flags)
        except OSError as exc:
            raise R6ArtifactContractError("r6 raw audio workspace cannot be opened safely") from exc
        try:
            _require_raw_audio_directory(os.fstat(current_fd), label="workspace root")
        except Exception:
            os.close(current_fd)
            raise
    else:
        # Do not lstat or reopen ``contract.workspace`` here.  The lease
        # boundary returns a descriptor only after it has freshly validated
        # the retained production repository chain; that descriptor is the
        # authority for every component below it.
        current_fd, workspace = contract._open_workspace_for_lease(_workspace_lease)
    current_path = workspace
    try:
        for component in relative.parts[:-1]:
            next_path = current_path / component
            _require_raw_audio_realpath_containment(workspace=workspace, candidate=next_path)
            metadata = _raw_audio_lstat(parent_fd=current_fd, component=component)
            if metadata is None:
                if not create:
                    raise R6ArtifactContractError("r6 raw audio parent is unavailable")
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                except OSError as exc:
                    raise R6ArtifactContractError("r6 raw audio parent cannot be created") from exc
                metadata = _raw_audio_lstat(parent_fd=current_fd, component=component)
                if metadata is None:  # pragma: no cover - defensive race guard
                    raise R6ArtifactContractError("r6 raw audio parent did not materialize")
            _require_raw_audio_directory(metadata, label="root or parent")
            try:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except OSError as exc:
                raise R6ArtifactContractError("r6 raw audio parent cannot be opened safely") from exc
            try:
                _require_raw_audio_directory(os.fstat(next_fd), label="root or parent")
            except Exception:
                os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
            current_path = next_path
            _require_raw_audio_realpath_containment(workspace=workspace, candidate=current_path)
        return current_fd, current_path, relative.name
    except Exception:
        os.close(current_fd)
        raise


def _safe_r6_relative_parts(relative: Path, *, label: str) -> tuple[str, ...]:
    """Validate one fixed, descriptor-walkable relative path.

    Production callers use only contract-owned relative paths.  Keeping this
    check next to the descriptor walk prevents a future caller from turning a
    ``dir_fd`` primitive into a general filesystem authority.
    """

    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} or "\x00" in part for part in relative.parts)
    ):
        raise R6ArtifactContractError(f"r6 {label} path is invalid")
    return tuple(relative.parts)


def _open_r6_descriptor_directory(
    *,
    contract: R6ArtifactContract,
    relative_directory: Path,
    create: bool,
    _workspace_lease: object | None = None,
) -> int:
    """Open a contract directory using retained workspace authority only.

    No lexical workspace path is inspected after this function begins.  Each
    component is lstat'ed, opened with ``O_NOFOLLOW`` relative to its retained
    parent descriptor, and identity-compared before the previous descriptor
    is released.  The caller owns the returned directory descriptor.
    """

    components = _safe_r6_relative_parts(relative_directory, label="descriptor directory")
    lease = contract._acquire_workspace_lease() if _workspace_lease is None else _workspace_lease
    current_fd, _workspace = contract._open_workspace_for_lease(lease)
    try:
        flags = _r6_anchor_directory_flags()
        for component in components:
            metadata = _raw_audio_lstat(parent_fd=current_fd, component=component)
            if metadata is None:
                if not create:
                    raise R6ArtifactContractError("r6 descriptor directory is unavailable")
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                except OSError as exc:
                    raise R6ArtifactContractError("r6 descriptor directory cannot be created") from exc
                metadata = _raw_audio_lstat(parent_fd=current_fd, component=component)
                if metadata is None:  # pragma: no cover - defensive race guard
                    raise R6ArtifactContractError("r6 descriptor directory did not materialize")
            _assert_r6_anchor_directory(metadata, label="descriptor directory")
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                raise R6ArtifactContractError("r6 descriptor directory cannot be opened safely") from exc
            try:
                opened = os.fstat(next_fd)
                _assert_r6_anchor_directory(opened, label="descriptor directory")
                if _r6_anchor_identity(opened) != _r6_anchor_identity(metadata):
                    raise R6ArtifactContractError("r6 descriptor directory changed during validation")
            except Exception:
                os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _read_r6_regular_leaf(
    *, parent_fd: int, leaf_name: str, missing_ok: bool
) -> bytes | None:
    """Read one no-follow regular leaf from an already-retained parent."""

    if not leaf_name or "/" in leaf_name or "\\" in leaf_name or "\x00" in leaf_name:
        raise R6ArtifactContractError("r6 descriptor leaf is invalid")
    metadata = _raw_audio_lstat(parent_fd=parent_fd, component=leaf_name)
    if metadata is None:
        if missing_ok:
            return None
        raise R6ArtifactContractError("r6 descriptor leaf is unavailable")
    if stat.S_ISLNK(metadata.st_mode):
        raise R6ArtifactContractError("r6 descriptor leaf is a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise R6ArtifactContractError("r6 descriptor leaf is not a regular file")
    try:
        descriptor = os.open(leaf_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as exc:
        raise R6ArtifactContractError("r6 descriptor leaf cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise R6ArtifactContractError("r6 descriptor leaf is not a regular file")
        if _r6_anchor_identity(opened) != _r6_anchor_identity(metadata):
            raise R6ArtifactContractError("r6 descriptor leaf changed during validation")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_r6_single_link_regular_leaf(
    *, parent_fd: int, leaf_name: str, missing_ok: bool
) -> bytes | None:
    """Read one no-follow leaf while rejecting every hard-link alias.

    Digest-addressed archive evidence is immutable only when the directory
    entry is its sole filesystem name.  The ordinary control readers retain
    their historical behaviour; archive and one-time migration callers use
    this stricter boundary.
    """

    if not leaf_name or "/" in leaf_name or "\\" in leaf_name or "\x00" in leaf_name:
        raise R6ArtifactContractError("r6 descriptor leaf is invalid")
    metadata = _raw_audio_lstat(parent_fd=parent_fd, component=leaf_name)
    if metadata is None:
        if missing_ok:
            return None
        raise R6ArtifactContractError("r6 descriptor leaf is unavailable")
    if stat.S_ISLNK(metadata.st_mode):
        raise R6ArtifactContractError("r6 descriptor leaf is a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise R6ArtifactContractError("r6 descriptor leaf is not a regular file")
    if metadata.st_nlink != 1:
        raise R6ArtifactContractError("r6 descriptor leaf has an unsafe link count")
    try:
        descriptor = os.open(leaf_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as exc:
        raise R6ArtifactContractError("r6 descriptor leaf cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise R6ArtifactContractError("r6 descriptor leaf is not a regular file")
        if opened.st_nlink != 1:
            raise R6ArtifactContractError("r6 descriptor leaf has an unsafe link count")
        if _r6_anchor_identity(opened) != _r6_anchor_identity(metadata):
            raise R6ArtifactContractError("r6 descriptor leaf changed during validation")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_r6_regular_leaf_atomically(
    *, parent_fd: int, leaf_name: str, raw: bytes, replace: bool
) -> None:
    """Publish finite bytes through one retained descriptor parent.

    The temporary and destination leaves never use a path rooted at the
    workspace.  A held descriptor makes an ancestor redirect irrelevant to
    creation, replace, fsync, and cleanup.
    """

    if not isinstance(raw, bytes) or not raw:
        raise R6ArtifactContractError("r6 descriptor bytes are invalid")
    if not leaf_name or "/" in leaf_name or "\\" in leaf_name or "\x00" in leaf_name:
        raise R6ArtifactContractError("r6 descriptor leaf is invalid")
    existing = _raw_audio_lstat(parent_fd=parent_fd, component=leaf_name)
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode):
            raise R6ArtifactContractError("r6 descriptor destination is a symlink")
        if not stat.S_ISREG(existing.st_mode):
            raise R6ArtifactContractError("r6 descriptor destination is not a regular file")
        if not replace:
            raise R6ArtifactContractError("r6 descriptor destination already exists")

    temporary_name = f".{leaf_name}.{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    published = False
    try:
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise R6ArtifactContractError("r6 descriptor temporary leaf cannot be created") from exc
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise R6ArtifactContractError("r6 descriptor temporary leaf is not a regular file")
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive write guard
                raise R6ArtifactContractError("r6 descriptor temporary leaf cannot be written")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if not replace:
            # The O_EXCL temporary is already the committed archive leaf only
            # when its fixed name is the destination; this branch is retained
            # for callers that need immutable evidence below.
            raise AssertionError("immutable publication must use fixed leaf helper")
        destination = _raw_audio_lstat(parent_fd=parent_fd, component=leaf_name)
        if destination is not None and stat.S_ISLNK(destination.st_mode):
            raise R6ArtifactContractError("r6 descriptor destination is a symlink")
        try:
            os.replace(
                temporary_name,
                leaf_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except OSError as exc:
            raise R6ArtifactContractError("r6 descriptor destination cannot be published") from exc
        published = True
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _write_r6_immutable_regular_leaf(*, parent_fd: int, leaf_name: str, raw: bytes) -> None:
    """Create one immutable checkpoint-archive leaf through a retained fd."""

    if not isinstance(raw, bytes) or not raw:
        raise R6ArtifactContractError("r6 descriptor bytes are invalid")
    if _raw_audio_lstat(parent_fd=parent_fd, component=leaf_name) is not None:
        raise R6ArtifactContractError("r6 descriptor destination already exists")
    try:
        descriptor = os.open(
            leaf_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
    except FileExistsError as exc:
        raise R6ArtifactContractError("r6 descriptor destination already exists") from exc
    except OSError as exc:
        raise R6ArtifactContractError("r6 descriptor destination cannot be created") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise R6ArtifactContractError("r6 descriptor destination is not a regular file")
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive write guard
                raise R6ArtifactContractError("r6 descriptor destination cannot be written")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(parent_fd)


def _write_or_verify_r6_immutable_regular_leaf(
    *, parent_fd: int, leaf_name: str, raw: bytes
) -> bool:
    """O_EXCL-create an immutable leaf or accept the exact prior publication.

    ``False`` means an already-published, descriptor-read, single-link regular
    file contained byte-for-byte identical evidence.  Every other preexisting
    filesystem object remains fatal.  This narrow idempotency lets a crashed
    one-time migration resume without weakening immutable archive semantics.
    """

    existing = _read_r6_single_link_regular_leaf(
        parent_fd=parent_fd,
        leaf_name=leaf_name,
        missing_ok=True,
    )
    if existing is not None:
        if existing != raw:
            raise R6ArtifactContractError(
                "r6 descriptor destination already exists with different bytes"
            )
        return False
    _write_r6_immutable_regular_leaf(parent_fd=parent_fd, leaf_name=leaf_name, raw=raw)
    verified = _read_r6_single_link_regular_leaf(
        parent_fd=parent_fd,
        leaf_name=leaf_name,
        missing_ok=False,
    )
    if verified != raw:  # pragma: no cover - fsync/readback defensive guard
        raise R6ArtifactContractError("r6 immutable descriptor bytes did not verify")
    return True


def read_r6_audio_checkpoint(
    *, contract: R6ArtifactContract, _workspace_lease: object | None = None
) -> bytes | None:
    """Read the canonical audio checkpoint only through retained descriptors."""

    relative = contract.relative_path("audio_recovery_receipt")
    try:
        parent_fd = _open_r6_descriptor_directory(
            contract=contract,
            relative_directory=relative.parent,
            create=False,
            _workspace_lease=_workspace_lease,
        )
    except R6ArtifactContractError as exc:
        # A completely absent control tree is the ordinary first-run state;
        # any symlink/type/anchor failure retains its fail-closed error.
        if str(exc) == "r6 descriptor directory is unavailable":
            return None
        raise
    try:
        return _read_r6_regular_leaf(parent_fd=parent_fd, leaf_name=relative.name, missing_ok=True)
    finally:
        os.close(parent_fd)


def write_r6_audio_checkpoint(
    *,
    contract: R6ArtifactContract,
    payload: Mapping[str, object],
    _workspace_lease: object | None = None,
) -> None:
    """Atomically replace the one checkpoint via a retained control fd."""

    try:
        raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R6ArtifactContractError("r6 checkpoint payload is not finite JSON") from exc
    relative = contract.relative_path("audio_recovery_receipt")
    parent_fd = _open_r6_descriptor_directory(
        contract=contract,
        relative_directory=relative.parent,
        create=True,
        _workspace_lease=_workspace_lease,
    )
    try:
        _write_r6_regular_leaf_atomically(
            parent_fd=parent_fd, leaf_name=relative.name, raw=raw, replace=True
        )
    finally:
        os.close(parent_fd)


def write_r6_continuation_diagnostic(
    *,
    contract: R6ArtifactContract,
    payload: Mapping[str, object],
    _workspace_lease: object | None = None,
) -> None:
    """Atomically replace the fixed, non-sensitive continuation outcome.

    Callers must validate the canonical checkpoint before using this writer.
    The writer owns the closed, non-sensitive wire shape and descriptor-safe
    publication only; semantic checkpoint facts remain owner-local.
    """

    required_fields = {
        "schema_version",
        "receipt_type",
        "reason_code",
        "recovery_inventory_sha256",
        "audio_policy_digest",
        "source_scope_sha256",
        "record_count",
        "checkpoint_sha256",
        "lineage_state",
        "historical_checkpoint_lineage_sha256",
        "historical_checkpoint_archive_sha256",
    }
    if set(payload) != required_fields:
        raise R6ArtifactContractError("r6 continuation diagnostic shape is invalid")
    if payload.get("schema_version") != 1 or payload.get("receipt_type") != (
        "conference_pipeline.cvpr2026_r6_continuation_diagnostic"
    ):
        raise R6ArtifactContractError("r6 continuation diagnostic identity is invalid")
    if payload.get("reason_code") not in {
        "already_full",
        "worker_no_progress",
        "continuation_bound_exhausted",
    }:
        raise R6ArtifactContractError("r6 continuation diagnostic reason is invalid")

    def _digest(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    for field in (
        "recovery_inventory_sha256",
        "audio_policy_digest",
        "source_scope_sha256",
        "checkpoint_sha256",
    ):
        if not _digest(payload.get(field)):
            raise R6ArtifactContractError("r6 continuation diagnostic digest is invalid")
    record_count = payload.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 0:
        raise R6ArtifactContractError("r6 continuation diagnostic count is invalid")
    lineage_state = payload.get("lineage_state")
    lineage_digest = payload.get("historical_checkpoint_lineage_sha256")
    archive_digest = payload.get("historical_checkpoint_archive_sha256")
    if lineage_state == "absent":
        if lineage_digest is not None or archive_digest is not None:
            raise R6ArtifactContractError("r6 continuation diagnostic absent lineage is invalid")
    elif lineage_state == "successor":
        if not _digest(lineage_digest) or not _digest(archive_digest):
            raise R6ArtifactContractError("r6 continuation diagnostic successor lineage is invalid")
    else:
        raise R6ArtifactContractError("r6 continuation diagnostic lineage state is invalid")

    try:
        raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R6ArtifactContractError("r6 continuation diagnostic is not finite JSON") from exc
    relative = contract.relative_path("continuation_diagnostic")
    parent_fd = _open_r6_descriptor_directory(
        contract=contract,
        relative_directory=relative.parent,
        create=True,
        _workspace_lease=_workspace_lease,
    )
    try:
        _write_r6_regular_leaf_atomically(
            parent_fd=parent_fd, leaf_name=relative.name, raw=raw, replace=True
        )
    finally:
        os.close(parent_fd)


def write_r6_cli_decision(
    *,
    contract: R6ArtifactContract,
    payload: Mapping[str, object],
    _workspace_lease: object | None = None,
) -> None:
    """Publish one closed, non-sensitive fixed-CLI decision.

    This is intentionally a wire-contract validator rather than a decision
    constructor.  The owner derives all binding facts from descriptor-read
    bytes; callers cannot smuggle paths, exception text, or arbitrary lineage
    into a durable outcome.
    """

    fields = {
        "schema_version", "receipt_type", "phase", "reason_code", "exit_code",
        "binding_state", "recovery_inventory_sha256", "audio_policy_digest",
        "source_scope_sha256", "record_count", "checkpoint_sha256",
        "lineage_state", "historical_checkpoint_lineage_sha256",
        "historical_checkpoint_archive_sha256",
    }
    if set(payload) != fields:
        raise R6ArtifactContractError("r6 CLI decision shape is invalid")
    if payload.get("schema_version") != 1 or payload.get("receipt_type") != (
        "conference_pipeline.cvpr2026_r6_cli_decision"
    ):
        raise R6ArtifactContractError("r6 CLI decision identity is invalid")
    allowed = {
        "cli": {"invalid_cli": 2},
        "pre_admission": {
            "owner_import_failed": 1, "owner_initialization_failed": 1,
            "admission_invalid": 1, "unexpected_owner_failure": 1,
        },
        "workspace_root": {"root_or_lease_unavailable": 1},
        "artifact": {"artifact_unreadable_or_invalid": 1},
        "preflight": {"fixed_preflight_blocked": 1},
        "supervisor": {
            "worker_no_progress": 2, "continuation_bound_exhausted": 2,
            "worker_interrupted": 2,
        },
    }
    phase = payload.get("phase")
    reason = payload.get("reason_code")
    if not isinstance(phase, str) or not isinstance(reason, str) or reason not in allowed.get(phase, {}):
        raise R6ArtifactContractError("r6 CLI decision phase or reason is invalid")
    if payload.get("exit_code") != allowed[phase][reason]:
        raise R6ArtifactContractError("r6 CLI decision exit is invalid")

    def digest(value: object) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    binding_state = payload.get("binding_state")
    bound_fields = (
        "recovery_inventory_sha256", "audio_policy_digest", "source_scope_sha256",
        "checkpoint_sha256",
    )
    lineage_fields = (
        "historical_checkpoint_lineage_sha256", "historical_checkpoint_archive_sha256",
    )
    if binding_state == "unavailable":
        if any(payload.get(key) is not None for key in (*bound_fields, "record_count", "lineage_state", *lineage_fields)):
            raise R6ArtifactContractError("r6 unavailable CLI decision binding is invalid")
    elif binding_state == "reachable":
        if not all(digest(payload.get(key)) for key in bound_fields):
            raise R6ArtifactContractError("r6 reachable CLI decision digest is invalid")
        count = payload.get("record_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise R6ArtifactContractError("r6 reachable CLI decision count is invalid")
        lineage_state = payload.get("lineage_state")
        if lineage_state == "absent":
            if any(payload.get(key) is not None for key in lineage_fields):
                raise R6ArtifactContractError("r6 absent CLI decision lineage is invalid")
        elif lineage_state == "successor":
            if not all(digest(payload.get(key)) for key in lineage_fields):
                raise R6ArtifactContractError("r6 successor CLI decision lineage is invalid")
        else:
            raise R6ArtifactContractError("r6 reachable CLI decision lineage is invalid")
    else:
        raise R6ArtifactContractError("r6 CLI decision binding state is invalid")
    try:
        raw = (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R6ArtifactContractError("r6 CLI decision is not finite JSON") from exc
    relative = contract.relative_path("cli_decision")
    parent_fd = _open_r6_descriptor_directory(
        contract=contract, relative_directory=relative.parent, create=True,
        _workspace_lease=_workspace_lease,
    )
    try:
        _write_r6_regular_leaf_atomically(parent_fd=parent_fd, leaf_name=relative.name, raw=raw, replace=True)
    finally:
        os.close(parent_fd)


def publish_r6_unavailable_owner_decision(*, reason_code: str) -> None:
    """Best-effort publication for the two pre-owner unavailable outcomes.

    This deliberately is not a generic decision writer.  The public CLI can
    select only one of the two closed reasons that exist before the fixed
    owner has been initialized, and this bridge derives every other field.
    A failed attempt to establish the fixed durable destination remains a
    finite public non-success without disclosing an operational detail.
    """

    if reason_code not in {"owner_import_failed", "owner_initialization_failed"}:
        raise R6ArtifactContractError("r6 unavailable owner decision reason is invalid")
    payload: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "phase": "pre_admission",
        "reason_code": reason_code,
        "exit_code": 1,
        "binding_state": "unavailable",
        "recovery_inventory_sha256": None,
        "audio_policy_digest": None,
        "source_scope_sha256": None,
        "record_count": None,
        "checkpoint_sha256": None,
        "lineage_state": None,
        "historical_checkpoint_lineage_sha256": None,
        "historical_checkpoint_archive_sha256": None,
    }
    try:
        contract = R6ArtifactContract.production()
        lease = contract._acquire_workspace_lease()
        with contract.workspace_lease_context(lease):
            write_r6_cli_decision(contract=contract, payload=payload, _workspace_lease=lease)
    except (R6ArtifactContractError, OSError):
        return


@dataclass
class R6RawAudioStagingCapability:
    """A descriptor-addressable, one-shot audio publication capability.

    The downloader never receives a filename or descriptor spelling.  Its
    narrowly-owned in-process sink may write bytes through ``write`` only;
    the parent descriptor and staging name remain private to publication.
    """

    parent_fd: int
    output_fd: int
    staged_name: str
    final_name: str
    relative: Path
    _closed: bool = field(default=False, init=False, repr=False)

    def write(self, payload: bytes) -> None:
        """Append one in-process payload chunk through the held descriptor.

        There is deliberately no public fd number, path, filename template,
        or file-like object here.  A caller that needs one of those has left
        the r6 single-stream contract and must fail before bytes are fetched.
        """

        if self._closed or not isinstance(payload, bytes) or not payload:
            raise R6ArtifactContractError("r6 raw-audio sink payload is invalid")
        view = memoryview(payload)
        while view:
            try:
                written = os.write(self.output_fd, view)
            except OSError as exc:
                raise R6ArtifactContractError("r6 raw-audio sink write failed") from exc
            if written <= 0:
                raise R6ArtifactContractError("r6 raw-audio sink made no progress")
            view = view[written:]

    def discard(self) -> None:
        """Close safely, removing only the held-parent staging leaf."""

        if self._closed:
            return
        try:
            os.close(self.output_fd)
            try:
                os.unlink(self.staged_name, dir_fd=self.parent_fd)
            except FileNotFoundError:
                pass
        finally:
            os.close(self.parent_fd)
            self._closed = True

    def publish_and_read(self) -> bytes:
        """Verify staged regular bytes then atomically publish under held fd."""

        if self._closed:
            raise R6ArtifactContractError("r6 raw-audio capability is closed")
        try:
            os.fsync(self.output_fd)
            os.close(self.output_fd)
            raw = _read_r6_regular_leaf(
                parent_fd=self.parent_fd, leaf_name=self.staged_name, missing_ok=False
            )
            assert raw is not None  # ``missing_ok=False`` is the contract.
            if not raw:
                raise R6ArtifactContractError("r6 raw-audio staging output is empty")
            existing = _raw_audio_lstat(parent_fd=self.parent_fd, component=self.final_name)
            if existing is not None:
                raise R6ArtifactContractError("r6 raw-audio final target already exists")
            try:
                os.replace(
                    self.staged_name,
                    self.final_name,
                    src_dir_fd=self.parent_fd,
                    dst_dir_fd=self.parent_fd,
                )
            except OSError as exc:
                raise R6ArtifactContractError("r6 raw-audio staging output cannot be published") from exc
            os.fsync(self.parent_fd)
            published = _read_r6_regular_leaf(
                parent_fd=self.parent_fd, leaf_name=self.final_name, missing_ok=False
            )
            assert published is not None
            if published != raw:
                raise R6ArtifactContractError("r6 raw-audio published bytes did not verify")
            return published
        finally:
            try:
                os.close(self.output_fd)
            except OSError:
                pass
            os.close(self.parent_fd)
            self._closed = True


def prepare_r6_raw_audio_staging_capability(
    *,
    contract: R6ArtifactContract,
    item_id: object,
    video_id: object,
    _workspace_lease: object | None = None,
) -> R6RawAudioStagingCapability:
    """Create a one-shot descriptor capability after metadata succeeds.

    This intentionally does *not* return an output ``Path``.  The capability
    is acquired immediately before child dispatch so an ancestor redirect
    while metadata runs cannot create an external payload.
    """

    if (
        not isinstance(item_id, str)
        or not item_id
        or not isinstance(video_id, str)
        or not video_id
        or any(token in item_id or token in video_id for token in ("/", "\\", "..", "\x00"))
    ):
        raise R6ArtifactContractError("r6 raw audio target identity is unsafe")
    relative = contract.relative_path("raw_audio") / f"{item_id}-{video_id}.mp3"
    relative = _raw_audio_relative_path(contract=contract, local_media_path=relative.as_posix())
    parent_fd = _open_r6_descriptor_directory(
        contract=contract,
        relative_directory=relative.parent,
        create=True,
        _workspace_lease=_workspace_lease,
    )
    try:
        if _raw_audio_lstat(parent_fd=parent_fd, component=relative.name) is not None:
            raise R6ArtifactContractError("r6 raw audio target already exists")
        staged_name = f".{relative.name}.{secrets.token_hex(16)}.staged.mp3"
        if _raw_audio_lstat(parent_fd=parent_fd, component=staged_name) is not None:
            raise R6ArtifactContractError("r6 raw audio staging target already exists")
        try:
            output_fd = os.open(
                staged_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise R6ArtifactContractError("r6 raw audio staging target cannot be created") from exc
        try:
            if not stat.S_ISREG(os.fstat(output_fd).st_mode):
                raise R6ArtifactContractError("r6 raw audio staging target is not a regular file")
        except Exception:
            os.close(output_fd)
            raise
        return R6RawAudioStagingCapability(
            parent_fd=parent_fd,
            output_fd=output_fd,
            staged_name=staged_name,
            final_name=relative.name,
            relative=relative,
        )
    except Exception:
        os.close(parent_fd)
        raise


def prepare_r6_raw_audio_target(
    *,
    contract: R6ArtifactContract,
    item_id: object,
    video_id: object,
    _workspace_lease: object | None = None,
) -> Path:
    """Prepare one absent producer target through the fixed raw-audio boundary."""

    if (
        not isinstance(item_id, str)
        or not item_id
        or not isinstance(video_id, str)
        or not video_id
        or any(token in item_id or token in video_id for token in ("/", "\\", "..", "\x00"))
    ):
        raise R6ArtifactContractError("r6 raw audio target identity is unsafe")
    workspace_fd: int | None = None
    validated_workspace: Path | None = None
    if _workspace_lease is not None:
        # This must precede even lexical raw-payload derivation.  The opaque
        # lease has no cached-path authority; its first usable operation is a
        # new descriptor-rooted retained-anchor validation.
        workspace_fd, validated_workspace = contract._open_workspace_for_lease(_workspace_lease)
    try:
        relative = contract.relative_path("raw_audio") / f"{item_id}-{video_id}.mp3"
        relative = _raw_audio_relative_path(
            contract=contract, local_media_path=relative.as_posix()
        )
        # ``_open_r6_raw_audio_parent`` owns (and closes on failure) a
        # supplied descriptor as soon as traversal begins.
        transferred_workspace_fd = workspace_fd
        workspace_fd = None
        parent_fd, parent_path, leaf_name = _open_r6_raw_audio_parent(
            contract=contract,
            relative=relative,
            create=True,
            _workspace_lease=_workspace_lease if transferred_workspace_fd is None else None,
            _workspace_fd=transferred_workspace_fd,
            _validated_workspace=validated_workspace,
        )
    except Exception:
        if workspace_fd is not None:
            os.close(workspace_fd)
        raise
    try:
        _require_raw_audio_realpath_containment(
            workspace=contract.workspace, candidate=parent_path / leaf_name
        )
        if _raw_audio_lstat(parent_fd=parent_fd, component=leaf_name) is not None:
            raise R6ArtifactContractError("r6 raw audio target already exists")
    finally:
        os.close(parent_fd)
    # This lexical value is returned only for the downloader handoff.  The
    # target-preparation authority above was the fresh descriptor chain, not
    # this cached path; callers re-enter the descriptor boundary before they
    # publish or read acquired bytes.
    return contract.workspace / relative


def read_r6_raw_audio_payload(
    *,
    contract: R6ArtifactContract,
    local_media_path: object,
    _workspace_lease: object | None = None,
) -> tuple[bytes, Path]:
    """Read a declared raw-audio leaf only through no-follow descriptors."""

    workspace_fd: int | None = None
    validated_workspace: Path | None = None
    if _workspace_lease is not None:
        # See target preparation above: no leased path derivation or raw-root
        # inspection may precede this per-access descriptor validation.
        workspace_fd, validated_workspace = contract._open_workspace_for_lease(_workspace_lease)
    try:
        relative = _raw_audio_relative_path(contract=contract, local_media_path=local_media_path)
        # ``_open_r6_raw_audio_parent`` owns (and closes on failure) a
        # supplied descriptor as soon as traversal begins.
        transferred_workspace_fd = workspace_fd
        workspace_fd = None
        parent_fd, parent_path, leaf_name = _open_r6_raw_audio_parent(
            contract=contract,
            relative=relative,
            create=False,
            _workspace_lease=_workspace_lease if transferred_workspace_fd is None else None,
            _workspace_fd=transferred_workspace_fd,
            _validated_workspace=validated_workspace,
        )
    except Exception:
        if workspace_fd is not None:
            os.close(workspace_fd)
        raise
    try:
        _require_raw_audio_realpath_containment(
            workspace=contract.workspace, candidate=parent_path / leaf_name
        )
        metadata = _raw_audio_lstat(parent_fd=parent_fd, component=leaf_name)
        if metadata is None:
            raise R6ArtifactContractError("r6 raw audio leaf is unavailable")
        if stat.S_ISLNK(metadata.st_mode):
            raise R6ArtifactContractError("r6 raw audio leaf is a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise R6ArtifactContractError("r6 raw audio leaf is not a regular file")
        try:
            descriptor = os.open(leaf_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        except OSError as exc:
            raise R6ArtifactContractError("r6 raw audio leaf cannot be opened safely") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise R6ArtifactContractError("r6 raw audio leaf is not a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks), relative
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def manifest_r6_view() -> dict[str, str]:
    """Return the generated manifest path view for r6 artifact families."""

    return {
        family_id: R6_ARTIFACT_PATHS[key].as_posix()
        for family_id, key in R6_MANIFEST_FAMILY_PATHS.items()
    }


def registry_r6_view() -> dict[str, object]:
    """Return the fixed r6 registry view used by validation and tests."""

    return {
        "required_run_id": R6_RUN_ID,
        "stage_id": R6_STAGE_ID,
        "required_receipts": [R6_ARTIFACT_PATHS[key].as_posix() for key in R6_RECEIPT_KEYS],
        "required_outputs": [
            R6_ARTIFACT_PATHS["recovery_inventory"].as_posix(),
            R6_ARTIFACT_PATHS["processing_inventory"].as_posix(),
        ],
        # An ordinary first attempt never creates this control root.  A
        # successor receipt must bind a digest-addressed immutable leaf below
        # it, which media validation independently rechecks.
        "conditional_control_roots": [
            R6_ARTIFACT_PATHS["audio_recovery_archive_root"].as_posix()
        ],
        "final_receipt": R6_ARTIFACT_PATHS["final_receipt"].as_posix(),
    }


def validate_r6_manifest_view(manifest: Mapping[str, object]) -> None:
    """Reject a hand-copied r6 manifest path that drifts from this authority."""

    families = manifest.get("artifact_families")
    if not isinstance(families, list):
        raise R6ArtifactContractError("storage manifest has no artifact families")
    by_id = {
        row.get("id"): row
        for row in families
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    for family_id, expected_path in manifest_r6_view().items():
        row = by_id.get(family_id)
        if not isinstance(row, Mapping) or row.get("path") != expected_path:
            raise R6ArtifactContractError(f"r6 manifest view drifted for {family_id}")
    successor = by_id.get("r6_cli_decision_successor")
    if not isinstance(successor, Mapping) or any(
        successor.get(key) != value
        for key, value in {
            "storage_class": "backup",
            "share_policy": "private",
            "rebuildable": False,
            "sync_profiles": {"public": False, "backup": True},
        }.items()
    ) or "path_glob" in successor:
        raise R6ArtifactContractError("r6 successor manifest family is not a narrow private backup control")
    notes = successor.get("notes")
    if not isinstance(notes, str) or any(
        forbidden not in notes.lower()
        for forbidden in ("payload", "credential", "cookie", "header", "session", "auth-page", "log")
    ):
        raise R6ArtifactContractError("r6 successor manifest family lacks control-only exclusions")


def validate_r6_registry_view(registry: Mapping[str, object]) -> None:
    """Reject an r6 registry binding that does not derive from this contract."""

    bindings = registry.get("stage_bindings")
    if not isinstance(bindings, Mapping):
        raise R6ArtifactContractError("stage registry has no r6 bindings")
    binding = bindings.get(R6_STAGE_ID)
    if not isinstance(binding, Mapping):
        raise R6ArtifactContractError("stage registry lacks r6 binding")
    workspace_bindings = binding.get("workspace_bindings")
    if not isinstance(workspace_bindings, Mapping):
        raise R6ArtifactContractError("stage registry lacks r6 workspace binding")
    cvpr = workspace_bindings.get(R6_WORKSPACE_RELATIVE_PATH.as_posix())
    if not isinstance(cvpr, Mapping) or not isinstance(cvpr.get("runtime_contract"), Mapping):
        raise R6ArtifactContractError("stage registry r6 workspace binding is invalid")
    runtime = cvpr["runtime_contract"]
    view = registry_r6_view()
    if runtime.get("required_run_id") != view["required_run_id"]:
        raise R6ArtifactContractError("stage registry r6 run id drifted")
    receipts = runtime.get("required_receipts")
    paths = (
        [row.get("path") for row in receipts if isinstance(row, Mapping)]
        if isinstance(receipts, list)
        else []
    )
    if paths != view["required_receipts"]:
        raise R6ArtifactContractError("stage registry r6 receipt view drifted")
    outputs = runtime.get("required_outputs")
    output_paths = (
        [row.get("path") for row in outputs if isinstance(row, Mapping)]
        if isinstance(outputs, list)
        else []
    )
    if output_paths != view["required_outputs"]:
        raise R6ArtifactContractError("stage registry r6 output view drifted")
    control_roots = runtime.get("conditional_control_roots")
    if control_roots != view["conditional_control_roots"]:
        raise R6ArtifactContractError("stage registry r6 conditional control view drifted")
