"""Finite direct-public material acquisition with declared-local outputs.

The module deliberately has no conference host literals, browser/session seam,
or default downloader.  A caller supplies an already-validated finite policy,
one receipt-indexed record, and a narrow transport implementation.  Every
non-successful outcome is a terminal classification record and never exposes a
local payload declaration.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_ARTIFACT_PATHS,
    R6ArtifactContract,
    R6ArtifactContractError,
)
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_RUN_ID as R6_EVENT_MEDIA_RUN_ID,
)
from conference_pipeline.acquisition.cvpr_r6_material_index import (
    R6_MATERIAL_DISCOVERY_RELATIVE_PATH,
    MaterialPolicyError,
    R6MaterialSourcePolicy,
    canonical_route_observation_digest,
    load_r6_material_source_policy,
    validate_material_discovery_receipt,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    source_route_scope_payload,
    validate_r6_source_route_scope,
)
from conference_pipeline.transcription_ocr import atomic_write_json, resolve_workspace_member

R6_MATERIAL_PAYLOAD_ROOT = R6_ARTIFACT_PATHS["material_payloads"]
R6_MATERIAL_ACQUISITION_RELATIVE_PATH = R6_ARTIFACT_PATHS["material_acquisition_receipt"]
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_FORBIDDEN_FIELD_TOKENS = frozenset(
    {"authorization", "cookie", "credential", "header", "password", "secret", "session", "token"}
)
_RECORD_REQUIRED_KEYS = frozenset(
    {
        "record_id",
        "event_id",
        "source_page_url",
        "direct_target_url",
        "content_class",
        "mime_type",
        "byte_count",
        "eligibility",
        "terminal_reason",
        "policy_digest",
    }
)
_ACQUISITION_PAYLOAD_KEYS = frozenset(
    {
        "record_id",
        "event_id",
        "acquisition_state",
        "source_page_provenance",
        "initial_target_url",
        "final_target_url",
        "redirect_hosts",
        "content_class",
        "mime_type",
        "byte_count",
        "sha256",
        "policy_digest",
        "local_payload_path",
        "terminal_reason",
    }
)
_ACQUISITION_TERMINAL_KEYS = _ACQUISITION_PAYLOAD_KEYS - {"sha256", "local_payload_path"}
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "application/pdf": (b"%PDF-",),
    "application/vnd.ms-powerpoint": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": (b"PK\x03\x04",),
    "application/vnd.oasis.opendocument.presentation": (b"PK\x03\x04",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/webp": (b"RIFF",),
}
_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.oasis.opendocument.presentation": ".odp",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class PublicMaterialResponse:
    """Small transport-neutral response contract used by the finite fetcher."""

    status_code: int
    headers: Mapping[str, str]
    body: Iterable[bytes]


class PublicMaterialTransport(Protocol):
    """Transport seam with no cookies, auth headers, or session argument."""

    def open(self, url: str) -> PublicMaterialResponse: ...


class PublicMaterialAcquisitionError(ValueError):
    """Raised only for programmer/receipt-contract faults, never for a fetch outcome."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if str(key).lower() == name.lower():
            return str(value)
    return None


def _safe_public_https_url(value: object) -> tuple[str, str] | None:
    """Normalize a credential-free, query-free HTTPS direct-public URL."""

    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        return None
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port not in (None, 443):
        return None
    host = parsed.hostname.lower().removeprefix("www.")
    return urlunsplit(("https", host, parsed.path or "/", "", "")), host


def _safe_record_url(record: Mapping[str, object], key: str) -> str | None:
    parsed = _safe_public_https_url(record.get(key))
    return parsed[0] if parsed is not None else None


def _forbidden_field_present(record: Mapping[str, object]) -> bool:
    return any(
        any(
            token in str(key).strip().lower().replace("-", "_") for token in _FORBIDDEN_FIELD_TOKENS
        )
        for key in record
    )


def _terminal_record(
    record: Mapping[str, object],
    *,
    policy: R6MaterialSourcePolicy,
    reason: str,
    final_url: str | None = None,
    redirect_hosts: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "record_id": str(record.get("record_id") or ""),
        "event_id": str(record.get("event_id") or ""),
        "acquisition_state": "terminal",
        "source_page_provenance": _safe_record_url(record, "source_page_url"),
        "initial_target_url": _safe_record_url(record, "direct_target_url"),
        "final_target_url": final_url,
        "redirect_hosts": list(redirect_hosts),
        "content_class": str(record.get("content_class") or ""),
        "mime_type": str(record.get("mime_type") or "").lower(),
        "byte_count": 0,
        "policy_digest": policy.policy_digest,
        "terminal_reason": reason,
    }


def _payload_root(workspace: Path, payload_root: Path) -> tuple[Path, Path] | None:
    workspace_root = workspace.resolve()
    candidate = payload_root if payload_root.is_absolute() else workspace_root / payload_root
    resolved, error = resolve_workspace_member(workspace_root, candidate)
    if error or resolved is None:
        return None
    try:
        return resolved, resolved.relative_to(workspace_root)
    except ValueError:
        return None


def _content_signature_matches(mime_type: str, prefix: bytes) -> bool:
    expected = _SIGNATURES.get(mime_type)
    if expected is None or not any(prefix.startswith(value) for value in expected):
        return False
    return mime_type != "image/webp" or len(prefix) >= 12 and prefix[8:12] == b"WEBP"


def _looks_like_authentication_path(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return any(
        segment in {"login", "signin", "sign-in", "auth", "authenticate"}
        for segment in path.split("/")
    )


def _deterministic_payload_path(
    *, workspace: Path, payload_root: Path, record_id: str, mime_type: str
) -> tuple[Path, str]:
    root = _payload_root(workspace, payload_root)
    if root is None:
        raise PublicMaterialAcquisitionError("payload root is outside workspace")
    destination_root, relative_root = root
    suffix = _SUFFIXES.get(mime_type)
    if suffix is None:
        raise PublicMaterialAcquisitionError("payload MIME type has no fixed suffix")
    name = sha256(record_id.encode("utf-8")).hexdigest()[:24] + suffix
    output = destination_root / name
    resolved, error = resolve_workspace_member(workspace.resolve(), output)
    if error or resolved is None:
        raise PublicMaterialAcquisitionError("payload output escapes workspace")
    return resolved, (relative_root / name).as_posix()


def _validate_record_shape(record: Mapping[str, object]) -> str | None:
    if _forbidden_field_present(record):
        return "record_contains_forbidden_field"
    if not _RECORD_REQUIRED_KEYS.issubset(record):
        return "malformed_discovery_record"
    if not isinstance(record.get("record_id"), str) or not str(record["record_id"]).strip():
        return "malformed_discovery_record"
    if not isinstance(record.get("event_id"), str) or not str(record["event_id"]).strip():
        return "malformed_discovery_record"
    if record.get("eligibility") != "eligible_direct_public":
        return "record_not_eligible"
    if record.get("terminal_reason") is not None:
        return "malformed_discovery_record"
    if isinstance(record.get("byte_count"), bool) or not isinstance(record.get("byte_count"), int):
        return "malformed_discovery_record"
    if int(record["byte_count"]) <= 0:
        return "malformed_discovery_record"
    return None


def acquire_declared_public_material(
    *,
    workspace: Path,
    policy: R6MaterialSourcePolicy,
    record: Mapping[str, object],
    transport: PublicMaterialTransport,
    payload_root: Path,
) -> dict[str, Any]:
    """Fetch one receipt-indexed direct-public record under a finite policy.

    All runtime failures become terminal records; successful records are the
    only ones that contain ``local_payload_path`` and can enter processing.
    """

    root = _payload_root(workspace, payload_root)
    if root is None:
        return _terminal_record(record, policy=policy, reason="payload_root_outside_workspace")
    shape_error = _validate_record_shape(record)
    if shape_error is not None:
        return _terminal_record(record, policy=policy, reason=shape_error)
    if record.get("policy_digest") != policy.policy_digest:
        return _terminal_record(record, policy=policy, reason="policy_digest_mismatch")
    source = _safe_public_https_url(record.get("source_page_url"))
    if source is None or source[1] not in policy.source_page_hosts:
        return _terminal_record(record, policy=policy, reason="source_page_host_not_allowed")
    provenance = record.get("source_page_provenance")
    if provenance is not None and provenance != source[0]:
        return _terminal_record(record, policy=policy, reason="source_page_provenance_mismatch")
    initial = _safe_public_https_url(record.get("direct_target_url"))
    if initial is None:
        return _terminal_record(record, policy=policy, reason="initial_target_not_public_https")
    if initial[1] not in policy.direct_target_hosts:
        return _terminal_record(record, policy=policy, reason="initial_target_host_not_allowed")
    if _looks_like_authentication_path(initial[0]):
        return _terminal_record(
            record, policy=policy, reason="initial_target_requires_authentication"
        )
    mime_type = str(record["mime_type"]).split(";", 1)[0].strip().lower()
    content_class = str(record["content_class"])
    if mime_type not in policy.content_class_mime_types.get(content_class, ()):
        return _terminal_record(record, policy=policy, reason="mime_type_not_allowed")
    declared_bytes = int(record["byte_count"])
    if declared_bytes > policy.max_bytes:
        return _terminal_record(record, policy=policy, reason="byte_count_exceeds_policy")

    current_url, current_host = initial
    redirect_hosts: list[str] = []
    visited = {current_url}
    redirect_count = 0
    while True:
        try:
            response = transport.open(current_url)
        except Exception:
            return _terminal_record(
                record,
                policy=policy,
                reason="transport_failure",
                final_url=current_url,
                redirect_hosts=redirect_hosts,
            )
        if response.status_code in _REDIRECT_STATUS_CODES:
            location = _header(response.headers, "location")
            next_url = _safe_public_https_url(urljoin(current_url, location or ""))
            if next_url is None:
                return _terminal_record(
                    record,
                    policy=policy,
                    reason="redirect_not_public_https",
                    final_url=current_url,
                    redirect_hosts=redirect_hosts,
                )
            if next_url[1] not in policy.direct_target_hosts:
                return _terminal_record(
                    record,
                    policy=policy,
                    reason="redirect_host_not_allowed",
                    final_url=current_url,
                    redirect_hosts=redirect_hosts,
                )
            if _looks_like_authentication_path(next_url[0]):
                return _terminal_record(
                    record,
                    policy=policy,
                    reason="redirect_to_authentication",
                    final_url=current_url,
                    redirect_hosts=redirect_hosts,
                )
            redirect_count += 1
            if redirect_count > policy.max_redirects:
                return _terminal_record(
                    record,
                    policy=policy,
                    reason="redirect_limit_exceeded",
                    final_url=current_url,
                    redirect_hosts=redirect_hosts,
                )
            if next_url[0] in visited:
                return _terminal_record(
                    record,
                    policy=policy,
                    reason="redirect_loop",
                    final_url=current_url,
                    redirect_hosts=redirect_hosts,
                )
            redirect_hosts.append(next_url[1])
            visited.add(next_url[0])
            current_url, current_host = next_url
            continue
        if response.status_code != 200:
            return _terminal_record(
                record,
                policy=policy,
                reason="http_status_not_success",
                final_url=current_url,
                redirect_hosts=redirect_hosts,
            )
        if current_host not in policy.direct_target_hosts:
            return _terminal_record(
                record,
                policy=policy,
                reason="final_target_host_not_allowed",
                final_url=current_url,
                redirect_hosts=redirect_hosts,
            )
        response_mime = (
            (_header(response.headers, "content-type") or "").split(";", 1)[0].strip().lower()
        )
        if (
            response_mime != mime_type
            or response_mime not in policy.content_class_mime_types[content_class]
        ):
            return _terminal_record(
                record,
                policy=policy,
                reason="mime_type_not_allowed",
                final_url=current_url,
                redirect_hosts=redirect_hosts,
            )
        try:
            output, output_relative = _deterministic_payload_path(
                workspace=workspace,
                payload_root=payload_root,
                record_id=str(record["record_id"]),
                mime_type=mime_type,
            )
        except PublicMaterialAcquisitionError:
            return _terminal_record(record, policy=policy, reason="payload_root_outside_workspace")

        temporary_path: Path | None = None
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".material-", suffix=".tmp", dir=output.parent
            )
            temporary_path = Path(temporary_name)
            total_bytes = 0
            digest = sha256()
            prefix = b""
            with os.fdopen(descriptor, "wb") as handle:
                for chunk in response.body:
                    if not isinstance(chunk, bytes) or not chunk:
                        if chunk == b"":
                            continue
                        raise ValueError("response body chunk is invalid")
                    total_bytes += len(chunk)
                    if total_bytes > policy.max_bytes:
                        raise OverflowError("stream exceeds policy byte limit")
                    digest.update(chunk)
                    if len(prefix) < 16:
                        prefix += chunk[: 16 - len(prefix)]
                    handle.write(chunk)
            if total_bytes != declared_bytes:
                raise ArithmeticError("response byte count differs from receipt")
            if not _content_signature_matches(mime_type, prefix):
                raise LookupError("content signature differs from declared MIME")
            temporary_path.replace(output)
        except OverflowError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            return _terminal_record(
                record,
                policy=policy,
                reason="byte_count_exceeds_policy",
                final_url=current_url,
                redirect_hosts=redirect_hosts,
            )
        except ArithmeticError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            return _terminal_record(
                record,
                policy=policy,
                reason="byte_count_mismatch",
                final_url=current_url,
                redirect_hosts=redirect_hosts,
            )
        except LookupError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            return _terminal_record(
                record,
                policy=policy,
                reason="content_signature_mismatch",
                final_url=current_url,
                redirect_hosts=redirect_hosts,
            )
        except (OSError, ValueError, TypeError, RuntimeError):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            return _terminal_record(
                record,
                policy=policy,
                reason="payload_write_failed",
                final_url=current_url,
                redirect_hosts=redirect_hosts,
            )
        return {
            "record_id": str(record["record_id"]),
            "event_id": str(record["event_id"]),
            "acquisition_state": "acquired",
            "source_page_provenance": source[0],
            "initial_target_url": initial[0],
            "final_target_url": current_url,
            "redirect_hosts": redirect_hosts,
            "content_class": content_class,
            "mime_type": mime_type,
            "byte_count": declared_bytes,
            "sha256": digest.hexdigest(),
            "policy_digest": policy.policy_digest,
            "local_payload_path": output_relative,
            "terminal_reason": None,
        }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicMaterialAcquisitionError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise PublicMaterialAcquisitionError(f"{label} must be a JSON object")
    return payload


def _relative_workspace_file(workspace: Path, relative: Path, *, label: str) -> Path:
    resolved, error = resolve_workspace_member(workspace.resolve(), workspace / relative)
    if error or resolved is None:
        raise PublicMaterialAcquisitionError(f"{label} escapes workspace")
    return resolved


def validate_material_acquisition_receipt(
    *,
    workspace: Path,
    discovery_receipt: Mapping[str, object],
    receipt: Mapping[str, object],
    recovery_inventory: Mapping[str, Any] | None = None,
    require_source_scope: bool = False,
) -> None:
    """Verify that payload declarations came only from the named discovery receipt."""

    workspace_root = workspace.resolve()
    policy = load_r6_material_source_policy(workspace_root)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_type") != "conference_pipeline.r6_public_material_acquisition"
        or receipt.get("status") != "succeeded"
        or receipt.get("run_id") != R6_EVENT_MEDIA_RUN_ID
        or receipt.get("output_path") != R6_MATERIAL_ACQUISITION_RELATIVE_PATH.as_posix()
        or receipt.get("payload_root") != R6_MATERIAL_PAYLOAD_ROOT.as_posix()
        or receipt.get("policy_digest") != policy.policy_digest
    ):
        raise PublicMaterialAcquisitionError("material acquisition receipt identity is invalid")
    discovery_path = _relative_workspace_file(
        workspace_root, R6_MATERIAL_DISCOVERY_RELATIVE_PATH, label="material discovery receipt"
    )
    if (
        not discovery_path.is_file()
        or receipt.get("discovery_receipt_sha256")
        != sha256(discovery_path.read_bytes()).hexdigest()
    ):
        raise PublicMaterialAcquisitionError(
            "material acquisition receipt discovery binding is invalid"
        )
    try:
        validate_material_discovery_receipt(
            workspace=workspace_root,
            receipt=discovery_receipt,
            recovery_inventory=recovery_inventory,
            require_source_scope=require_source_scope,
        )
    except MaterialPolicyError as exc:
        raise PublicMaterialAcquisitionError("material discovery receipt is invalid") from exc
    discovery_by_id = {
        str(record["record_id"]): record
        for record in discovery_receipt.get("records", [])
        if isinstance(record, Mapping)
    }
    payloads = receipt.get("payloads")
    terminals = receipt.get("terminal_records")
    if not isinstance(payloads, list) or not isinstance(terminals, list):
        raise PublicMaterialAcquisitionError("material acquisition receipt rows are invalid")
    seen: set[str] = set()
    for row in [*payloads, *terminals]:
        if not isinstance(row, Mapping):
            raise PublicMaterialAcquisitionError("material acquisition row is invalid")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id in seen or record_id not in discovery_by_id:
            raise PublicMaterialAcquisitionError("material acquisition record identity is invalid")
        seen.add(record_id)
        if row.get("policy_digest") != policy.policy_digest:
            raise PublicMaterialAcquisitionError("material acquisition row policy is invalid")
        if row.get("acquisition_state") == "acquired":
            if set(row) != _ACQUISITION_PAYLOAD_KEYS or row.get("terminal_reason") is not None:
                raise PublicMaterialAcquisitionError("material payload declaration is invalid")
            local_path = row.get("local_payload_path")
            if not isinstance(local_path, str):
                raise PublicMaterialAcquisitionError("material payload path is invalid")
            payload = _relative_workspace_file(
                workspace_root, Path(local_path), label="material payload"
            )
            relative = payload.relative_to(workspace_root)
            if not (
                relative == R6_MATERIAL_PAYLOAD_ROOT or R6_MATERIAL_PAYLOAD_ROOT in relative.parents
            ):
                raise PublicMaterialAcquisitionError("material payload root is invalid")
            if (
                not payload.is_file()
                or row.get("byte_count") != payload.stat().st_size
                or row.get("sha256") != sha256(payload.read_bytes()).hexdigest()
            ):
                raise PublicMaterialAcquisitionError("material payload digest is invalid")
        elif row.get("acquisition_state") == "terminal":
            if set(row) != _ACQUISITION_TERMINAL_KEYS or not isinstance(
                row.get("terminal_reason"), str
            ):
                raise PublicMaterialAcquisitionError("terminal material acquisition row is invalid")
        else:
            raise PublicMaterialAcquisitionError("material acquisition state is invalid")
    if set(discovery_by_id) != seen:
        raise PublicMaterialAcquisitionError("material acquisition receipt is incomplete")
    expected_counts = {"declared": len(seen), "acquired": len(payloads), "terminal": len(terminals)}
    if receipt.get("counts") != expected_counts:
        raise PublicMaterialAcquisitionError("material acquisition receipt counts are invalid")
    if require_source_scope:
        if recovery_inventory is None:
            raise PublicMaterialAcquisitionError("material acquisition source scope is missing")
        scope = validate_r6_source_route_scope(
            recovery_inventory,
            {
                key: discovery_receipt.get(key)
                for key in ("source_scope_count", "source_scope_sha256", "source_routes")
            },
        )
        expected_scope = source_route_scope_payload(scope)
        if any(
            receipt.get(key) != expected_scope[key]
            for key in ("source_scope_count", "source_scope_sha256")
        ):
            raise PublicMaterialAcquisitionError("material acquisition source scope drifted")
        route_observations = discovery_receipt.get("route_observations")
        if (
            not isinstance(route_observations, list)
            or receipt.get("route_observation_sha256")
            != discovery_receipt.get("route_observation_sha256")
            or receipt.get("route_observation_sha256")
            != canonical_route_observation_digest(route_observations)
        ):
            raise PublicMaterialAcquisitionError("material acquisition route observations drifted")


def run_fixed_r6_material_acquisition(
    *,
    workspace: Path,
    transport: PublicMaterialTransport,
    artifact_contract: R6ArtifactContract | None = None,
    recovery_inventory: Mapping[str, Any] | None = None,
    require_source_scope: bool = False,
) -> dict[str, Any]:
    """Acquire only the named r6 receipt-indexed direct-public records.

    There is intentionally no URL, host, session, cookie, header, or discovery
    argument.  The caller can provide only the workspace and a transport seam.
    """

    contract = artifact_contract or R6ArtifactContract.production()
    try:
        workspace_root = contract.require_workspace(workspace)
    except R6ArtifactContractError as exc:
        raise PublicMaterialAcquisitionError(
            "r6 acquisition workspace does not match the fixed artifact contract"
        ) from exc
    discovery_path = _relative_workspace_file(
        workspace_root, R6_MATERIAL_DISCOVERY_RELATIVE_PATH, label="material discovery receipt"
    )
    discovery = _load_json(discovery_path, label="material discovery receipt")
    if require_source_scope and recovery_inventory is None:
        raise PublicMaterialAcquisitionError("r6 material acquisition requires immutable recovery inventory")
    try:
        validate_material_discovery_receipt(
            workspace=workspace_root,
            receipt=discovery,
            recovery_inventory=recovery_inventory,
            require_source_scope=require_source_scope,
        )
    except MaterialPolicyError as exc:
        raise PublicMaterialAcquisitionError("material discovery receipt is invalid") from exc
    policy = load_r6_material_source_policy(workspace_root)
    payloads: list[dict[str, Any]] = []
    terminals: list[dict[str, Any]] = []
    records = discovery.get("records")
    assert isinstance(records, list)  # validated directly above
    for record in records:
        assert isinstance(record, Mapping)
        if record.get("eligibility") != "eligible_direct_public":
            terminals.append(
                _terminal_record(
                    record,
                    policy=policy,
                    reason=f"discovery_terminal:{record.get('terminal_reason')}",
                )
            )
            continue
        result = acquire_declared_public_material(
            workspace=workspace_root,
            policy=policy,
            record=record,
            transport=transport,
            payload_root=R6_MATERIAL_PAYLOAD_ROOT,
        )
        if result["acquisition_state"] == "acquired":
            payloads.append(result)
        else:
            terminals.append(result)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.r6_public_material_acquisition",
        "status": "succeeded",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "output_path": R6_MATERIAL_ACQUISITION_RELATIVE_PATH.as_posix(),
        "payload_root": R6_MATERIAL_PAYLOAD_ROOT.as_posix(),
        "policy_digest": policy.policy_digest,
        "discovery_receipt_sha256": sha256(discovery_path.read_bytes()).hexdigest(),
        "payloads": payloads,
        "terminal_records": terminals,
        "counts": {"declared": len(records), "acquired": len(payloads), "terminal": len(terminals)},
    }
    if require_source_scope:
        assert recovery_inventory is not None
        scope = validate_r6_source_route_scope(
            recovery_inventory,
            {
                key: discovery.get(key)
                for key in ("source_scope_count", "source_scope_sha256", "source_routes")
            },
        )
        receipt.update(
            {
                "source_scope_count": scope.source_scope_count,
                "source_scope_sha256": scope.source_scope_sha256,
                "route_observation_sha256": discovery.get("route_observation_sha256"),
            }
        )
    output = _relative_workspace_file(
        workspace_root, R6_MATERIAL_ACQUISITION_RELATIVE_PATH, label="material acquisition receipt"
    )
    atomic_write_json(output, receipt)
    validate_material_acquisition_receipt(
        workspace=workspace_root,
        discovery_receipt=discovery,
        receipt=receipt,
        recovery_inventory=recovery_inventory,
        require_source_scope=require_source_scope,
    )
    return receipt
