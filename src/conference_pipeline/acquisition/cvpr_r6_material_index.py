"""Pure CVPR r6 material-observation indexing with finite source policy.

This module deliberately classifies fixture-supplied official-page observations.
It never requests pages, follows a redirect, downloads content, scans storage,
or turns a paper/video into a slide claim.  A later acquisition owner may use
only an eligible, receipt-indexed direct-public record after its own checks.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_ARTIFACT_PATHS,
)
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_RUN_ID as R6_EVENT_MEDIA_RUN_ID,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    R6SourceRouteScope,
    source_route_scope_payload,
    validate_r6_source_route_scope,
)
from conference_pipeline.transcription_ocr import atomic_write_json, resolve_workspace_member

R6_MATERIAL_DISCOVERY_RELATIVE_PATH = R6_ARTIFACT_PATHS["material_discovery_receipt"]
_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "source_page_hosts",
        "direct_target_hosts",
        "max_redirects",
        "max_bytes",
        "content_class_mime_types",
    }
)
_OBSERVATION_KEYS = frozenset(
    {
        "record_id",
        "event_id",
        "source_page_url",
        "observed_at",
        "direct_target_url",
        "link_kind",
        "content_class",
        "mime_type",
        "byte_count",
        "redirect_hops",
    }
)
_CONTENT_CLASSES = frozenset({"paper_pdf", "deck_pdf", "presentation_file", "slide_image"})
_LINK_KINDS = frozenset(
    {"direct_public", "vendor_embed", "static_image_page", "authenticated_only", "absent"}
)
_FORBIDDEN_FIELD_TOKENS = frozenset(
    {"authorization", "cookie", "credential", "header", "password", "secret", "session", "token"}
)
_HISTORICAL_CONTEXT = {
    "workshop_pdf_receipt_backed": 517,
    "workshop_pdf_missing": 60,
    "main_cvpr_pdf_existing": 0,
    "presentation_or_deck_existing": 0,
}
_SOURCE_SCOPE_KEYS = frozenset(
    {
        "source_scope_count",
        "source_scope_sha256",
        "source_routes",
    }
)
_ROUTE_OBSERVATION_KEYS = frozenset(
    {
        "route_id",
        "event_id",
        "source_page_url",
        "record_id",
        "terminal_classification",
    }
)


class MaterialPolicyError(ValueError):
    """Raised when a material observation escapes the finite local contract."""


@dataclass(frozen=True)
class R6MaterialSourcePolicy:
    """Validated finite policy selected by a workspace configuration file."""

    policy_path: Path
    policy_digest: str
    source_page_hosts: tuple[str, ...]
    direct_target_hosts: tuple[str, ...]
    max_redirects: int
    max_bytes: int
    content_class_mime_types: dict[str, tuple[str, ...]]


def _canonical_json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def canonical_route_observation_digest(observations: Sequence[Mapping[str, object]]) -> str:
    """Hash only sorted, secret-free terminal route classifications."""

    canonical = [dict(observation) for observation in observations]
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _scope_from_mapping(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != _SOURCE_SCOPE_KEYS:
        raise MaterialPolicyError("material source-route scope has undeclared fields")
    return dict(value)


def _receipt_scope(receipt: Mapping[str, object]) -> dict[str, object]:
    return _scope_from_mapping({key: receipt.get(key) for key in _SOURCE_SCOPE_KEYS})


def _route_observations_for_records(
    *, scope: R6SourceRouteScope, records: Sequence[Mapping[str, object]]
) -> list[dict[str, str]]:
    """Bind one material classification record to every frozen official route."""

    by_id = {str(record.get("record_id")): record for record in records}
    expected_ids = {str(route["route_id"]) for route in scope.routes}
    if set(by_id) != expected_ids or len(by_id) != len(records):
        raise MaterialPolicyError("material observations do not account for every source route")
    result: list[dict[str, str]] = []
    for route in scope.routes:
        route_id = str(route["route_id"])
        record = by_id[route_id]
        if (
            record.get("event_id") != route["event_id"]
            or record.get("source_page_url") != route["source_page_url"]
        ):
            raise MaterialPolicyError("material observation route provenance does not match scope")
        classification = (
            "eligible_direct_public"
            if record.get("eligibility") == "eligible_direct_public"
            else str(record.get("terminal_reason") or "classification_only")
        )
        result.append(
            {
                "route_id": route_id,
                "event_id": str(route["event_id"]),
                "source_page_url": str(route["source_page_url"]),
                "record_id": route_id,
                "terminal_classification": classification,
            }
        )
    return result


def _validate_route_observations(
    *, scope: R6SourceRouteScope, receipt: Mapping[str, object], records: Sequence[Mapping[str, object]]
) -> None:
    observations = receipt.get("route_observations")
    if not isinstance(observations, list) or len(observations) != scope.source_scope_count:
        raise MaterialPolicyError("material discovery route observations are incomplete")
    expected = _route_observations_for_records(scope=scope, records=records)
    normalized: list[dict[str, str]] = []
    for observation in observations:
        if not isinstance(observation, Mapping) or set(observation) != _ROUTE_OBSERVATION_KEYS:
            raise MaterialPolicyError("material discovery route observation is invalid")
        _forbidden_observation_keys(observation, reject_unexpected=False)
        normalized.append({key: str(observation[key]) for key in sorted(_ROUTE_OBSERVATION_KEYS)})
    if normalized != expected:
        raise MaterialPolicyError("material discovery route observations drifted from scope")
    if receipt.get("route_observation_sha256") != canonical_route_observation_digest(normalized):
        raise MaterialPolicyError("material discovery route observation digest is invalid")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _clean_host(value: object, *, label: str) -> str:
    host = str(value or "").strip().lower().removeprefix("www.")
    if not host or any(token in host for token in (":", "/", "@", "?", "#", " ")):
        raise MaterialPolicyError(f"{label} must be a bare hostname")
    return host


def _unique_hosts(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MaterialPolicyError(f"{label} must be a nonempty finite list")
    hosts = tuple(_clean_host(item, label=label) for item in value)
    if len(set(hosts)) != len(hosts):
        raise MaterialPolicyError(f"{label} cannot contain duplicates")
    return hosts


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MaterialPolicyError(f"{label} must be a positive integer")
    return value


def _resolve_workspace_file(workspace: Path, value: object, *, label: str) -> Path:
    path_text = str(value or "").strip()
    if not path_text:
        raise MaterialPolicyError(f"{label} is required")
    path = Path(path_text)
    if path.is_absolute():
        raise MaterialPolicyError(f"{label} must be workspace-relative")
    resolved, error = resolve_workspace_member(workspace.resolve(), workspace / path)
    if error or resolved is None:
        raise MaterialPolicyError(f"{label} must remain inside the workspace")
    return resolved


def load_r6_material_source_policy(workspace: Path) -> R6MaterialSourcePolicy:
    """Load the one workspace-selected finite policy without any network access."""

    workspace_root = workspace.resolve()
    config_path = workspace_root / "config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MaterialPolicyError("workspace config is unreadable") from exc
    if not isinstance(config, Mapping):
        raise MaterialPolicyError("workspace config must be a mapping")
    workflow = config.get("workflow")
    r6 = workflow.get("r6") if isinstance(workflow, Mapping) else None
    if not isinstance(r6, Mapping) or set(r6) != {"material_source_policy_path"}:
        raise MaterialPolicyError("workspace config must point to exactly one r6 material policy")
    policy_path = _resolve_workspace_file(
        workspace_root, r6.get("material_source_policy_path"), label="r6 material policy path"
    )
    try:
        raw_policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MaterialPolicyError("r6 material policy is unreadable") from exc
    if not isinstance(raw_policy, Mapping) or set(raw_policy) != _POLICY_KEYS:
        raise MaterialPolicyError("r6 material policy must contain only the finite policy fields")
    if raw_policy.get("schema_version") != 1 or not isinstance(raw_policy.get("policy_id"), str):
        raise MaterialPolicyError("r6 material policy schema is invalid")
    source_page_hosts = _unique_hosts(
        raw_policy.get("source_page_hosts"), label="source page hosts"
    )
    direct_target_hosts = _unique_hosts(
        raw_policy.get("direct_target_hosts"), label="direct target hosts"
    )
    max_redirects = _positive_int(raw_policy.get("max_redirects"), label="max redirects")
    max_bytes = _positive_int(raw_policy.get("max_bytes"), label="max bytes")
    raw_mime_types = raw_policy.get("content_class_mime_types")
    if not isinstance(raw_mime_types, Mapping) or set(raw_mime_types) != _CONTENT_CLASSES:
        raise MaterialPolicyError("r6 material policy MIME classes are incomplete")
    content_class_mime_types: dict[str, tuple[str, ...]] = {}
    for content_class in sorted(_CONTENT_CLASSES):
        raw_values = raw_mime_types.get(content_class)
        if not isinstance(raw_values, list) or not raw_values:
            raise MaterialPolicyError("r6 material policy MIME values must be finite")
        values = tuple(str(value or "").strip().lower() for value in raw_values)
        if any("/" not in value or value.startswith("/") for value in values) or len(
            set(values)
        ) != len(values):
            raise MaterialPolicyError("r6 material policy MIME values are invalid")
        content_class_mime_types[content_class] = values
    canonical_policy = {
        "schema_version": raw_policy["schema_version"],
        "policy_id": raw_policy["policy_id"].strip(),
        "source_page_hosts": list(source_page_hosts),
        "direct_target_hosts": list(direct_target_hosts),
        "max_redirects": max_redirects,
        "max_bytes": max_bytes,
        "content_class_mime_types": {
            name: list(content_class_mime_types[name]) for name in sorted(content_class_mime_types)
        },
    }
    return R6MaterialSourcePolicy(
        policy_path=policy_path,
        policy_digest=_canonical_json_digest(canonical_policy),
        source_page_hosts=source_page_hosts,
        direct_target_hosts=direct_target_hosts,
        max_redirects=max_redirects,
        max_bytes=max_bytes,
        content_class_mime_types=content_class_mime_types,
    )


def _forbidden_observation_keys(
    observation: Mapping[str, object], *, reject_unexpected: bool = True
) -> None:
    for key in observation:
        normalized = str(key).strip().lower().replace("-", "_")
        if any(token in normalized for token in _FORBIDDEN_FIELD_TOKENS):
            raise MaterialPolicyError(
                "secret or session material is forbidden in material observations"
            )
    unexpected = set(observation) - _OBSERVATION_KEYS
    if reject_unexpected and unexpected:
        raise MaterialPolicyError("material observation has undeclared discovery authority")


def _safe_https_url(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or any(char.isspace() for char in value)
    ):
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port not in (None, 443):
        return None
    host = parsed.hostname.lower().removeprefix("www.")
    netloc = host
    return urlunsplit(("https", netloc, parsed.path or "/", parsed.query, "")), host


def _required_text(observation: Mapping[str, object], key: str) -> str:
    value = observation.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MaterialPolicyError(f"material observation {key} is required")
    return value.strip()


def _validate_observed_at(value: object) -> str:
    timestamp = _required_text({"observed_at": value}, "observed_at")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MaterialPolicyError("material observation observed_at is invalid") from exc
    return timestamp


def _terminal_record(
    *,
    record_id: str,
    event_id: str,
    source_page_url: str,
    observed_at: str,
    direct_target_url: str | None,
    redirect_hops: list[str],
    host_class: str,
    content_class: str,
    mime_type: str,
    byte_count: int,
    policy: R6MaterialSourcePolicy,
    eligibility: str,
    terminal_reason: str | None,
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "event_id": event_id,
        "source_page_url": source_page_url,
        "source_page_provenance": source_page_url,
        "observed_at": observed_at,
        "direct_target_url": direct_target_url,
        "redirect_hops": redirect_hops,
        "host_class": host_class,
        "content_class": content_class,
        "mime_type": mime_type,
        "byte_count": byte_count,
        "eligibility": eligibility,
        "terminal_reason": terminal_reason,
        "policy_digest": policy.policy_digest,
        "source_sha256": None,
        "payload_sha256": None,
        "local_path": None,
    }


def _classify_observation(
    observation: Mapping[str, object], *, policy: R6MaterialSourcePolicy
) -> dict[str, Any]:
    _forbidden_observation_keys(observation)
    record_id = _required_text(observation, "record_id")
    event_id = _required_text(observation, "event_id")
    observed_at = _validate_observed_at(observation.get("observed_at"))
    source_raw = _required_text(observation, "source_page_url")
    source = _safe_https_url(source_raw)
    if source is None or source[1] not in policy.source_page_hosts:
        raise MaterialPolicyError("material observation source page host is not allowed")
    source_page_url = source[0]
    link_kind = _required_text(observation, "link_kind")
    if link_kind not in _LINK_KINDS:
        raise MaterialPolicyError("material observation link kind is invalid")
    content_class = _required_text(observation, "content_class")
    if content_class not in _CONTENT_CLASSES:
        raise MaterialPolicyError("material observation content class is invalid")
    mime_type = _required_text(observation, "mime_type").lower()
    byte_count = observation.get("byte_count")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise MaterialPolicyError("material observation byte count is invalid")
    raw_hops = observation.get("redirect_hops")
    if not isinstance(raw_hops, list) or not all(isinstance(hop, str) for hop in raw_hops):
        raise MaterialPolicyError("material observation redirect hops are invalid")
    target_raw = observation.get("direct_target_url")
    if link_kind == "absent":
        if target_raw not in (None, ""):
            raise MaterialPolicyError("absent material observation cannot declare a target")
        return _terminal_record(
            record_id=record_id,
            event_id=event_id,
            source_page_url=source_page_url,
            observed_at=observed_at,
            direct_target_url=None,
            redirect_hops=[],
            host_class="no_material",
            content_class=content_class,
            mime_type=mime_type,
            byte_count=byte_count,
            policy=policy,
            eligibility="classification_only",
            terminal_reason="no_material_link",
        )
    target = _safe_https_url(target_raw)
    safe_target_url = target[0] if target else None
    safe_hops: list[str] = []
    for hop in raw_hops:
        parsed_hop = _safe_https_url(hop)
        if parsed_hop is None:
            return _terminal_record(
                record_id=record_id,
                event_id=event_id,
                source_page_url=source_page_url,
                observed_at=observed_at,
                direct_target_url=safe_target_url,
                redirect_hops=safe_hops,
                host_class="unknown_host",
                content_class=content_class,
                mime_type=mime_type,
                byte_count=byte_count,
                policy=policy,
                eligibility="classification_only",
                terminal_reason="malformed_redirect_url",
            )
        safe_hops.append(parsed_hop[0])
    if link_kind != "direct_public":
        return _terminal_record(
            record_id=record_id,
            event_id=event_id,
            source_page_url=source_page_url,
            observed_at=observed_at,
            direct_target_url=safe_target_url,
            redirect_hops=safe_hops,
            host_class=link_kind,
            content_class=content_class,
            mime_type=mime_type,
            byte_count=byte_count,
            policy=policy,
            eligibility="classification_only",
            terminal_reason=link_kind,
        )
    if target is None:
        return _terminal_record(
            record_id=record_id,
            event_id=event_id,
            source_page_url=source_page_url,
            observed_at=observed_at,
            direct_target_url=None,
            redirect_hops=safe_hops,
            host_class="unknown_host",
            content_class=content_class,
            mime_type=mime_type,
            byte_count=byte_count,
            policy=policy,
            eligibility="classification_only",
            terminal_reason="malformed_target_url",
        )
    if target[1] not in policy.direct_target_hosts:
        return _terminal_record(
            record_id=record_id,
            event_id=event_id,
            source_page_url=source_page_url,
            observed_at=observed_at,
            direct_target_url=safe_target_url,
            redirect_hops=safe_hops,
            host_class="unknown_host",
            content_class=content_class,
            mime_type=mime_type,
            byte_count=byte_count,
            policy=policy,
            eligibility="classification_only",
            terminal_reason="target_host_not_allowed",
        )
    if len(safe_hops) > policy.max_redirects:
        return _terminal_record(
            record_id=record_id,
            event_id=event_id,
            source_page_url=source_page_url,
            observed_at=observed_at,
            direct_target_url=safe_target_url,
            redirect_hops=safe_hops,
            host_class="direct_public",
            content_class=content_class,
            mime_type=mime_type,
            byte_count=byte_count,
            policy=policy,
            eligibility="classification_only",
            terminal_reason="redirect_limit_exceeded",
        )
    if any(
        (_safe_https_url(hop) or ("", ""))[1] not in policy.direct_target_hosts for hop in safe_hops
    ):
        return _terminal_record(
            record_id=record_id,
            event_id=event_id,
            source_page_url=source_page_url,
            observed_at=observed_at,
            direct_target_url=safe_target_url,
            redirect_hops=safe_hops,
            host_class="unknown_host",
            content_class=content_class,
            mime_type=mime_type,
            byte_count=byte_count,
            policy=policy,
            eligibility="classification_only",
            terminal_reason="redirect_host_not_allowed",
        )
    if mime_type not in policy.content_class_mime_types[content_class]:
        return _terminal_record(
            record_id=record_id,
            event_id=event_id,
            source_page_url=source_page_url,
            observed_at=observed_at,
            direct_target_url=safe_target_url,
            redirect_hops=safe_hops,
            host_class="direct_public",
            content_class=content_class,
            mime_type=mime_type,
            byte_count=byte_count,
            policy=policy,
            eligibility="classification_only",
            terminal_reason="mime_type_not_allowed",
        )
    if byte_count > policy.max_bytes:
        return _terminal_record(
            record_id=record_id,
            event_id=event_id,
            source_page_url=source_page_url,
            observed_at=observed_at,
            direct_target_url=safe_target_url,
            redirect_hops=safe_hops,
            host_class="direct_public",
            content_class=content_class,
            mime_type=mime_type,
            byte_count=byte_count,
            policy=policy,
            eligibility="classification_only",
            terminal_reason="byte_count_exceeds_policy",
        )
    return _terminal_record(
        record_id=record_id,
        event_id=event_id,
        source_page_url=source_page_url,
        observed_at=observed_at,
        direct_target_url=safe_target_url,
        redirect_hops=safe_hops,
        host_class="direct_public",
        content_class=content_class,
        mime_type=mime_type,
        byte_count=byte_count,
        policy=policy,
        eligibility="eligible_direct_public",
        terminal_reason=None,
    )


def _resolve_declared_output_path(workspace: Path, output_path: object) -> Path:
    resolved, error = resolve_workspace_member(workspace.resolve(), output_path)
    if error or resolved is None:
        raise MaterialPolicyError(
            "declared r6 material-discovery path must remain inside the workspace"
        )
    try:
        relative = resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise MaterialPolicyError(
            "declared r6 material-discovery path must remain inside the workspace"
        ) from exc
    if relative != R6_MATERIAL_DISCOVERY_RELATIVE_PATH:
        raise MaterialPolicyError("declared r6 material-discovery path is invalid")
    return resolved


def build_material_discovery_receipt(
    *,
    workspace: Path,
    observations: Sequence[Mapping[str, object]],
    output_path: object,
    source_scope: Mapping[str, object] | None = None,
    recovery_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete classification receipt from explicitly supplied observations."""

    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise MaterialPolicyError("material observations must be an explicit sequence")
    workspace_root = workspace.resolve()
    _resolve_declared_output_path(workspace_root, output_path)
    policy = load_r6_material_source_policy(workspace_root)
    records = [_classify_observation(observation, policy=policy) for observation in observations]
    record_ids = [record["record_id"] for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise MaterialPolicyError("material observations contain duplicate record identities")
    targets = [record["direct_target_url"] for record in records if record["direct_target_url"]]
    if len(set(targets)) != len(targets):
        raise MaterialPolicyError("material observations contain duplicate direct targets")
    records.sort(key=lambda record: str(record["record_id"]))
    eligible_count = sum(record["eligibility"] == "eligible_direct_public" for record in records)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_material_discovery",
        "status": "succeeded",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "output_path": R6_MATERIAL_DISCOVERY_RELATIVE_PATH.as_posix(),
        "policy_digest": policy.policy_digest,
        "records": records,
        "counts": {
            "declared": len(records),
            "eligible_direct_public": eligible_count,
            "classification_only": len(records) - eligible_count,
        },
        "historical_context": dict(_HISTORICAL_CONTEXT),
    }
    if source_scope is not None or recovery_inventory is not None:
        if source_scope is None or recovery_inventory is None:
            raise MaterialPolicyError("material discovery source scope requires immutable recovery inventory")
        scope = validate_r6_source_route_scope(recovery_inventory, _scope_from_mapping(source_scope))
        route_observations = _route_observations_for_records(scope=scope, records=records)
        receipt.update(
            {
                **source_route_scope_payload(scope),
                "route_observations": route_observations,
                "route_observation_sha256": canonical_route_observation_digest(route_observations),
            }
        )
    validate_material_discovery_receipt(
        workspace=workspace_root,
        receipt=receipt,
        recovery_inventory=recovery_inventory,
        require_source_scope=source_scope is not None,
    )
    return receipt


def validate_material_discovery_receipt(
    *,
    workspace: Path,
    receipt: Mapping[str, object],
    recovery_inventory: Mapping[str, Any] | None = None,
    require_source_scope: bool = False,
) -> None:
    """Validate an untrusted receipt without reading a page or a local payload."""

    if not isinstance(receipt, Mapping):
        raise MaterialPolicyError("material discovery receipt must be a mapping")
    policy = load_r6_material_source_policy(workspace)
    if receipt.get("receipt_type") != "conference_pipeline.cvpr2026_r6_material_discovery":
        raise MaterialPolicyError("material discovery receipt type is invalid")
    if receipt.get("schema_version") != 1 or receipt.get("status") != "succeeded":
        raise MaterialPolicyError("material discovery receipt status is invalid")
    if receipt.get("run_id") != R6_EVENT_MEDIA_RUN_ID:
        raise MaterialPolicyError("material discovery receipt run id is invalid")
    if receipt.get("output_path") != R6_MATERIAL_DISCOVERY_RELATIVE_PATH.as_posix():
        raise MaterialPolicyError("material discovery receipt output path is invalid")
    if receipt.get("policy_digest") != policy.policy_digest:
        raise MaterialPolicyError(
            "material discovery receipt policy digest does not match workspace policy"
        )
    if receipt.get("historical_context") != _HISTORICAL_CONTEXT:
        raise MaterialPolicyError("material discovery receipt historical context is invalid")
    has_scope = any(key in receipt for key in (*_SOURCE_SCOPE_KEYS, "route_observations", "route_observation_sha256"))
    if require_source_scope and (recovery_inventory is None or not has_scope):
        raise MaterialPolicyError("material discovery source-route scope is required")
    if has_scope:
        if recovery_inventory is None:
            raise MaterialPolicyError("material discovery scope cannot be validated without recovery inventory")
        scope = validate_r6_source_route_scope(recovery_inventory, _receipt_scope(receipt))
        if receipt.get("source_scope_count") != scope.source_scope_count:
            raise MaterialPolicyError("material discovery source scope count is invalid")
    records = receipt.get("records")
    if not isinstance(records, list):
        raise MaterialPolicyError("material discovery receipt records are invalid")
    seen_ids: set[str] = set()
    seen_targets: set[str] = set()
    eligible_count = 0
    required = {
        "record_id",
        "event_id",
        "source_page_url",
        "source_page_provenance",
        "observed_at",
        "direct_target_url",
        "redirect_hops",
        "host_class",
        "content_class",
        "mime_type",
        "byte_count",
        "eligibility",
        "terminal_reason",
        "policy_digest",
        "source_sha256",
        "payload_sha256",
        "local_path",
    }
    for record in records:
        if not isinstance(record, Mapping) or set(record) != required:
            raise MaterialPolicyError("material discovery record shape is invalid")
        _forbidden_observation_keys(record, reject_unexpected=False)
        record_id = record.get("record_id")
        target = record.get("direct_target_url")
        if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
            raise MaterialPolicyError("material discovery record identity is invalid")
        seen_ids.add(record_id)
        if target is not None:
            if not isinstance(target, str) or not target or target in seen_targets:
                raise MaterialPolicyError("material discovery record target is invalid")
            seen_targets.add(target)
        if record.get("policy_digest") != policy.policy_digest:
            raise MaterialPolicyError("material discovery record policy digest is invalid")
        if (
            record.get("source_sha256") is not None
            or record.get("payload_sha256") is not None
            or record.get("local_path") is not None
        ):
            raise MaterialPolicyError("material discovery cannot declare payload hashes or paths")
        source = _safe_https_url(record.get("source_page_url"))
        if (
            source is None
            or source[1] not in policy.source_page_hosts
            or record.get("source_page_provenance") != source[0]
        ):
            raise MaterialPolicyError("material discovery record source provenance is invalid")
        if record.get("content_class") not in _CONTENT_CLASSES:
            raise MaterialPolicyError("material discovery record content class is invalid")
        if (
            isinstance(record.get("byte_count"), bool)
            or not isinstance(record.get("byte_count"), int)
            or record["byte_count"] <= 0
        ):
            raise MaterialPolicyError("material discovery record byte count is invalid")
        if not isinstance(record.get("redirect_hops"), list):
            raise MaterialPolicyError("material discovery record redirects are invalid")
        eligibility = record.get("eligibility")
        terminal_reason = record.get("terminal_reason")
        if eligibility == "eligible_direct_public":
            eligible_count += 1
            if record.get("host_class") != "direct_public" or terminal_reason is not None:
                raise MaterialPolicyError("eligible material discovery record is invalid")
            target = _safe_https_url(target)
            if (
                target is None
                or target[1] not in policy.direct_target_hosts
                or len(record["redirect_hops"]) > policy.max_redirects
                or any(
                    (parsed_hop := _safe_https_url(hop)) is None
                    or parsed_hop[1] not in policy.direct_target_hosts
                    for hop in record["redirect_hops"]
                )
                or record.get("mime_type")
                not in policy.content_class_mime_types[record["content_class"]]
                or record["byte_count"] > policy.max_bytes
            ):
                raise MaterialPolicyError("eligible material discovery record escapes policy")
        elif eligibility == "classification_only":
            if not isinstance(terminal_reason, str) or not terminal_reason:
                raise MaterialPolicyError("terminal material discovery record lacks a reason")
        else:
            raise MaterialPolicyError("material discovery eligibility is invalid")
    expected_counts = {
        "declared": len(records),
        "eligible_direct_public": eligible_count,
        "classification_only": len(records) - eligible_count,
    }
    if receipt.get("counts") != expected_counts:
        raise MaterialPolicyError("material discovery receipt counts are invalid")
    if has_scope:
        assert recovery_inventory is not None
        scope = validate_r6_source_route_scope(recovery_inventory, _receipt_scope(receipt))
        typed_records = [record for record in records if isinstance(record, Mapping)]
        _validate_route_observations(scope=scope, receipt=receipt, records=typed_records)
        if scope.source_scope_count > 0 and not typed_records:
            raise MaterialPolicyError("nonzero CVPR source scope cannot publish an empty discovery receipt")


def write_material_discovery_receipt(
    *,
    workspace: Path,
    receipt: Mapping[str, object],
    output_path: object,
    recovery_inventory: Mapping[str, Any] | None = None,
    require_source_scope: bool = False,
) -> Path:
    """Atomically write only the named r6 discovery receipt after local validation."""

    workspace_root = workspace.resolve()
    output = _resolve_declared_output_path(workspace_root, output_path)
    validate_material_discovery_receipt(
        workspace=workspace_root,
        receipt=receipt,
        recovery_inventory=recovery_inventory,
        require_source_scope=require_source_scope,
    )
    atomic_write_json(output, dict(receipt))
    return output
