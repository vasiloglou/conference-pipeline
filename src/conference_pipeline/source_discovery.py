"""Safe, digest-bound primitives for conference source discovery.

The module deliberately separates untrusted network responses from canonical
inventory data.  It does not crawl source links, download a corpus, invoke an
LLM, or retain authentication material.  Stage owners supply a configured
``SourceFamily`` and only persist bounded cache snapshots plus sanitized
metadata.
"""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import math
import os
import re
import socket
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import yaml

from conference_pipeline.acquisition.embedded_media import sanitize_media_url
from conference_pipeline.model_governance.registry import (
    default_model,
    default_model_for_provider,
    default_provider,
    load_registry,
)
from conference_pipeline.openreview_auth import load_openreview_auth
from conference_pipeline.provider_receipts import sanitize_for_receipt
from conference_pipeline.stage_contracts import verify_pre_spend_contract


FAMILY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,80}$")
ALLOWED_FAMILY_STATES = frozenset({"required", "locked_excluded", "not_applicable", "blocked", "unknown"})
ALLOWED_AUTH_POSTURES = frozenset({"public", "required", "optional", "registration_gated"})
LOCKED_AUTOMATIC_EXCLUSIONS = frozenset({"paywalled_official_video"})
DEFAULT_MAX_BYTES = 2 * 1024 * 1024
MAX_DECLARED_FAMILY_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_TIMEOUT_SECONDS = 20.0
CROSS_SOURCE_JOIN_TASK_ID = "source_discovery.cross_source_join"
GOVERNED_JOIN_METHOD = "governed_llm_frontier"
# An "included" candidate pair (see SourceFamily.deterministic_title_join)
# never reaches the LLM boundary; this method name always discloses that in
# the resulting typed edge, never disguised as a governed_llm_frontier call.
DETERMINISTIC_JOIN_METHOD = "deterministic_same_corpus_title_join"
_MATERIAL_CONFLICT_FIELDS = frozenset(
    {
        "identity",
        "event_family_scope",
        "acceptance_status",
        "source_ownership",
        "artifact_association",
    }
)
_FIELD_AUTHORITIES: dict[str, tuple[str, ...]] = {
    "event_structure": ("program-schedule",),
    "timing": ("program-schedule",),
    "room": ("program-schedule",),
    "program_family": ("program-schedule",),
    "event_family_scope": ("program-schedule",),
    "submission_metadata": ("openreview-conference", "openreview-workshop"),
    "forum_metadata": ("openreview-conference", "openreview-workshop"),
    "acceptance_status": ("openreview-conference",),
    "camera_ready_artifact": ("cvf-openaccess",),
    "artifact_association": ("cvf-openaccess",),
    "paper_title": ("cvf-openaccess",),
    "authors": ("cvf-openaccess",),
    "media_identifier": ("youtube",),
    "caption_availability": ("youtube",),
}
_DECLARED_TAXONOMY: dict[str, frozenset[str]] = {
    "program_family": frozenset({"oral", "poster", "session", "workshop", "tutorial", "keynote"}),
    "program_type": frozenset({"oral", "poster", "session", "workshop", "tutorial", "keynote"}),
    "media_family": frozenset({"youtube", "official_video", "virtual_paper"}),
    "source_link_class": frozenset({"official", "openreview", "openaccess", "youtube"}),
}
_FAMILY_LINK_CLASS = {
    "program-schedule": "official",
    "cvf-openaccess": "openaccess",
    "cvf-openaccess-day-2026-06-05": "openaccess",
    "cvf-openaccess-day-2026-06-06": "openaccess",
    "cvf-openaccess-day-2026-06-07": "openaccess",
    "openreview-conference": "openreview",
    "openreview-workshop": "openreview",
    "virtual-paper-media-posture": "official",
    "youtube": "youtube",
}
_MAX_EXTRACTED_RECORDS_PER_FAMILY = 10_000
# These labels identify page chrome rather than conference content.  They are
# intentionally title-key based and are applied only to the candidate pair;
# they neither remove a source record nor create an event-family exclusion.
_STRUCTURAL_NAVIGATION_TITLE_KEYS = frozenset(
    {
        "skip to yearly menu bar",
        "home",
        "login",
        "code of conduct",
    }
)


class SourceDiscoveryError(RuntimeError):
    """Base exception for source-discovery validation failures."""


class SourceFetchError(SourceDiscoveryError):
    """Raised before unsafe or incomplete network evidence is persisted."""


class SourcePolicyError(SourceDiscoveryError):
    """Raised when a source-family policy is incomplete or unsafe."""


@dataclass(frozen=True)
class SourceFamily:
    """Declared route and evidence rules for one source family."""

    id: str
    source_url: str
    allowed_hosts: tuple[str, ...]
    authority_fields: tuple[str, ...]
    expected_completeness: str
    auth_posture: str
    state: str
    source_kind: str = "html"
    max_bytes: int | None = None
    # Opt-in only (default False): declares this family a trusted, single-
    # authority catalog of the SAME underlying paper list as any other family
    # also carrying this flag, so an unambiguous (exactly two records sharing
    # an exact normalized title) cross-family match between two such families
    # can be joined deterministically -- skipping the governed LLM boundary
    # entirely -- without inventing a new title-matching concept. A title
    # shared by 3+ records, or shared with any family that does NOT carry
    # this flag, is never auto-joined and still goes through
    # source_discovery.cross_source_join. No existing conference sets this
    # field, so this is behavior-neutral until a policy opts in.
    deterministic_title_join: bool = False


@dataclass(frozen=True)
class SourcePolicy:
    """Validated, declarative source policy for one conference workspace."""

    schema_version: int
    families: dict[str, SourceFamily]
    locked_exclusions: frozenset[str]


@dataclass(frozen=True)
class FetchResponse:
    """Response supplied by the production or injected test transport."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str | None = None
    redirect_url: str | None = None


class AuthProtocol(Protocol):
    def headers(self) -> dict[str, str]: ...

    def receipt_fields(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class OpenReviewFetchSuccess:
    response: FetchResponse
    auth_receipt: dict[str, Any]


@dataclass(frozen=True)
class OpenReviewAuthBlocker:
    """Secret-free blocker emitted after the single permitted auth retry."""

    code: str
    family_id: str
    auth_receipt: dict[str, Any]
    retry_count: int = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _normalise_host(host: str) -> str:
    return host.rstrip(".").lower()


def _host_is_allowed(host: str, allowed_hosts: Sequence[str]) -> bool:
    normalized = _normalise_host(host)
    return any(
        normalized == _normalise_host(allowed)
        or normalized.endswith("." + _normalise_host(allowed))
        for allowed in allowed_hosts
    )


def _reject_private_or_reserved_host(host: str) -> None:
    """Reject literal IP destinations before allowlist matching or DNS use."""

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if not address.is_global:
        raise SourceFetchError("private, loopback, reserved, or link-local destination is forbidden")


def validate_source_url(url: str, family: SourceFamily) -> str:
    """Validate an HTTPS route before a transport can connect to it.

    We only accept declared source hosts, reject URL userinfo, and reject
    literal non-public IP addresses.  The configured host allowlist means this
    function does not perform attacker-controlled DNS resolution.
    """

    try:
        parsed = urlsplit(url)
        host = parsed.hostname
    except ValueError as exc:
        raise SourceFetchError("malformed source URL") from exc
    if parsed.scheme.lower() != "https":
        raise SourceFetchError("source URL must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise SourceFetchError("credential-bearing URLs are forbidden")
    if not host:
        raise SourceFetchError("source URL must include a host")
    _reject_private_or_reserved_host(host)
    if not _host_is_allowed(host, family.allowed_hosts):
        raise SourceFetchError("source URL host is not in the declared allowlist")
    return url


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:  # type: ignore[override]
        return None


def _read_limited(response: Any, max_bytes: int) -> bytes:
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise SourceFetchError(f"response exceeds byte limit ({max_bytes})")
    return body


def _stdlib_transport(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> FetchResponse:
    """Single-hop production transport; redirects are validated by the caller."""

    request = Request(url, headers=dict(headers or {}), method="GET")
    opener = build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            response_headers = {str(key): str(value) for key, value in response.headers.items()}
            location = response_headers.get("Location")
            return FetchResponse(
                status_code=int(response.status),
                headers=response_headers,
                body=_read_limited(response, max_bytes),
                final_url=response.geturl(),
                redirect_url=(urljoin(url, location) if location else None),
            )
    except HTTPError as exc:
        response_headers = {str(key): str(value) for key, value in exc.headers.items()}
        location = response_headers.get("Location")
        return FetchResponse(
            status_code=int(exc.code),
            headers=response_headers,
            body=_read_limited(exc, max_bytes),
            final_url=url,
            redirect_url=(urljoin(url, location) if location else None),
        )


Transport = Callable[..., FetchResponse]


class SafeSourceFetcher:
    """Bounded, allowlisted fetcher with redirect and response guards."""

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = 1,
        backoff_seconds: float = 0.05,
    ) -> None:
        if max_bytes <= 0 or max_redirects < 0 or timeout <= 0 or retries < 0:
            raise ValueError("fetch bounds must be positive (redirects/retries may be zero)")
        self.transport = transport or _stdlib_transport
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.timeout = timeout
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def _request(
        self,
        url: str,
        headers: Mapping[str, str] | None,
        *,
        max_bytes: int,
    ) -> FetchResponse:
        for attempt in range(self.retries + 1):
            try:
                response = self.transport(
                    url,
                    headers=dict(headers or {}),
                    timeout=self.timeout,
                    max_bytes=max_bytes,
                )
                if not isinstance(response.body, bytes):
                    raise SourceFetchError("transport returned a non-byte response body")
                if len(response.body) > max_bytes:
                    raise SourceFetchError(f"response exceeds byte limit ({max_bytes})")
                return response
            except SourceFetchError:
                raise
            except (OSError, URLError) as exc:
                if attempt >= self.retries:
                    raise SourceFetchError(f"bounded source request failed ({exc.__class__.__name__})") from exc
                time.sleep(self.backoff_seconds * (attempt + 1))
        raise AssertionError("unreachable retry loop")

    def fetch(
        self,
        url: str,
        family: SourceFamily,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> FetchResponse:
        current_url = validate_source_url(url, family)
        max_bytes = family.max_bytes if family.max_bytes is not None else self.max_bytes
        redirects = 0
        while True:
            response = self._request(current_url, headers, max_bytes=max_bytes)
            final_url = validate_source_url(response.final_url or current_url, family)
            redirect_url = response.redirect_url
            if not redirect_url and 300 <= response.status_code < 400:
                location = next(
                    (value for key, value in response.headers.items() if key.lower() == "location"),
                    None,
                )
                redirect_url = urljoin(current_url, location) if location else None
            if redirect_url:
                if redirects >= self.max_redirects:
                    raise SourceFetchError(f"redirect limit ({self.max_redirects}) exceeded")
                current_url = validate_source_url(urljoin(final_url, redirect_url), family)
                redirects += 1
                continue
            if 300 <= response.status_code < 400:
                raise SourceFetchError("redirect response has no safe Location target")
            return FetchResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response.body,
                final_url=final_url,
            )


def _safe_header(headers: Mapping[str, str], name: str) -> str | None:
    return next((str(value) for key, value in headers.items() if key.lower() == name.lower()), None)


def _relative_to_cache_parent(cache_root: Path, path: Path) -> str:
    # The expected cache root is <workspace>/cache/source-discovery.  Keep
    # metadata portable while still allowing isolated test cache roots.
    try:
        return str(path.relative_to(cache_root.parent.parent))
    except ValueError:
        return str(path)


def snapshot_family(
    family: SourceFamily,
    *,
    fetcher: SafeSourceFetcher,
    cache_root: Path,
    run_id: str,
    headers: Mapping[str, str] | None = None,
    response: FetchResponse | None = None,
) -> dict[str, Any]:
    """Fetch one configured family and atomically persist its bounded cache evidence."""

    if not run_id or "/" in run_id or "\\" in run_id:
        raise SourceFetchError("run id must be a non-empty path-safe identifier")
    response = response or fetcher.fetch(family.source_url, family, headers=headers)
    digest = hashlib.sha256(response.body).hexdigest()
    snapshot_id = f"{family.id}-{digest[:16]}"
    family_root = cache_root / run_id / family.id
    family_root.mkdir(parents=True, exist_ok=True)
    target = family_root / snapshot_id
    with tempfile.TemporaryDirectory(prefix=f".{snapshot_id}.", dir=family_root) as temp_dir:
        temp = Path(temp_dir)
        body_path = temp / "body.bin"
        body_path.write_bytes(response.body)
        metadata_path = temp / "snapshot.json"
        metadata = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "family_id": family.id,
            "fetched_at": _utc_now(),
            "requested_url": sanitize_media_url(family.source_url),
            "final_url": sanitize_media_url(str(response.final_url)),
            "status_code": response.status_code,
            "content_type": _safe_header(response.headers, "content-type"),
            "bytes": len(response.body),
            "sha256": digest,
            "redaction_applied": True,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if target.exists():
            # Same digest snapshots are immutable; preserve the original evidence.
            existing = target / "snapshot.json"
            if existing.exists():
                return {
                    **json.loads(existing.read_text(encoding="utf-8")),
                    "body_path": _relative_to_cache_parent(cache_root, target / "body.bin"),
                    "metadata_path": _relative_to_cache_parent(cache_root, existing),
                }
        os.replace(temp, target)
    return {
        **metadata,
        "body_path": _relative_to_cache_parent(cache_root, target / "body.bin"),
        "metadata_path": _relative_to_cache_parent(cache_root, target / "snapshot.json"),
    }


def _names_only_auth_receipt(auth: AuthProtocol) -> dict[str, Any]:
    raw = auth.receipt_fields()
    return {
        "openreview_auth_mode": str(raw.get("openreview_auth_mode", "unknown")),
        "openreview_auth_env_names": [
            str(name) for name in raw.get("openreview_auth_env_names", []) if str(name).isupper()
        ],
        "secret_values_recorded": False,
    }


def reload_worktree_environment(env_path: Path) -> None:
    """Reload simple ``KEY=VALUE`` entries without logging values or sourcing a shell."""

    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            os.environ[key] = value.strip().strip('"').strip("'")


def _is_openreview_auth_failure(response: FetchResponse) -> bool:
    if response.status_code in {401, 403}:
        return True
    text = response.body[:2048].decode("utf-8", errors="ignore").lower()
    return "challenge" in text or "verifying your browser" in text


def fetch_openreview_with_recovery(
    family: SourceFamily,
    *,
    fetcher: SafeSourceFetcher,
    auth_loader: Callable[[], AuthProtocol] | None = None,
    environment_reloader: Callable[[Path], None] = reload_worktree_environment,
    env_path: Path | None = None,
) -> OpenReviewFetchSuccess | OpenReviewAuthBlocker:
    """Perform one non-mutating OpenReview retry after a challenge/auth failure."""

    if family.auth_posture not in {"required", "optional"}:
        raise ValueError("OpenReview recovery only applies to authenticated source families")
    if auth_loader is None:
        first_auth = load_openreview_auth("conference-pipeline-source-discovery", mode=family.auth_posture)
    else:
        first_auth = auth_loader()
    first_response = fetcher.fetch(family.source_url, family, headers=first_auth.headers())
    if not _is_openreview_auth_failure(first_response):
        return OpenReviewFetchSuccess(first_response, _names_only_auth_receipt(first_auth))

    environment_reloader(env_path or (Path(__file__).resolve().parents[2] / ".env"))
    retry_auth = (
        load_openreview_auth(
            "conference-pipeline-source-discovery",
            mode=family.auth_posture,
            force_refresh=True,
        )
        if auth_loader is None
        else auth_loader()
    )
    retry_response = fetcher.fetch(family.source_url, family, headers=retry_auth.headers())
    if not _is_openreview_auth_failure(retry_response):
        return OpenReviewFetchSuccess(retry_response, _names_only_auth_receipt(retry_auth))
    return OpenReviewAuthBlocker(
        code="openreview_auth_recovery_failed",
        family_id=family.id,
        auth_receipt=_names_only_auth_receipt(retry_auth),
    )


def load_source_policy(path: Path) -> SourcePolicy:
    """Load and strictly validate the conference-local source policy."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SourcePolicyError(f"unable to read source policy: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise SourcePolicyError("source policy must declare schema_version: 1")
    exclusions = frozenset(str(value) for value in raw.get("locked_exclusions", []))
    if exclusions != LOCKED_AUTOMATIC_EXCLUSIONS:
        raise SourcePolicyError("only paywalled_official_video may be an automatic exclusion")
    raw_families = raw.get("source_families")
    if not isinstance(raw_families, list) or not raw_families:
        raise SourcePolicyError("source policy must include source_families")
    families: dict[str, SourceFamily] = {}
    for row in raw_families:
        if not isinstance(row, dict):
            raise SourcePolicyError("each source family must be an object")
        family_id = str(row.get("id", ""))
        if not FAMILY_ID_RE.fullmatch(family_id) or family_id in families:
            raise SourcePolicyError(f"invalid or duplicate source family id: {family_id!r}")
        hosts = tuple(str(host).lower() for host in row.get("allowed_hosts", []) if str(host))
        authorities = tuple(str(field) for field in row.get("authority_fields", []) if str(field))
        state = str(row.get("state", ""))
        posture = str(row.get("auth_posture", ""))
        raw_max_bytes = row.get("max_bytes")
        if raw_max_bytes is None:
            max_bytes = None
        elif isinstance(raw_max_bytes, int) and 0 < raw_max_bytes <= MAX_DECLARED_FAMILY_BYTES:
            max_bytes = raw_max_bytes
        else:
            raise SourcePolicyError(
                f"{family_id} max_bytes must be a positive integer no greater than "
                f"{MAX_DECLARED_FAMILY_BYTES}"
            )
        if not hosts or not authorities:
            raise SourcePolicyError(f"{family_id} requires allowlisted hosts and authority fields")
        if state not in ALLOWED_FAMILY_STATES or posture not in ALLOWED_AUTH_POSTURES:
            raise SourcePolicyError(f"{family_id} has an invalid state or auth posture")
        raw_deterministic_title_join = row.get("deterministic_title_join", False)
        if not isinstance(raw_deterministic_title_join, bool):
            raise SourcePolicyError(f"{family_id} deterministic_title_join must be a boolean")
        family = SourceFamily(
            id=family_id,
            source_url=str(row.get("source_url", "")),
            allowed_hosts=hosts,
            authority_fields=authorities,
            expected_completeness=str(row.get("expected_completeness", "")),
            auth_posture=posture,
            state=state,
            source_kind=str(row.get("source_kind", "html")),
            max_bytes=max_bytes,
            deterministic_title_join=raw_deterministic_title_join,
        )
        if not family.expected_completeness:
            raise SourcePolicyError(f"{family_id} requires expected_completeness")
        validate_source_url(family.source_url, family)
        families[family_id] = family
    return SourcePolicy(schema_version=1, families=families, locked_exclusions=exclusions)


def canonical_json_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for structured, non-secret evidence."""

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a small control artifact atomically without ever writing in place."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, path)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SourceDiscoveryError(f"invalid existing source-discovery inventory: {path}") from exc
    if not isinstance(raw, dict):
        raise SourceDiscoveryError("existing source-discovery inventory must be an object")
    return raw


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def publish_family_snapshot(
    inventory_path: Path,
    *,
    family_id: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically promote a family snapshot while preserving superseded IDs.

    This writes a small inventory control projection only; raw body bytes stay
    in the cache family owned by :func:`snapshot_family`.
    """

    if not FAMILY_ID_RE.fullmatch(family_id):
        raise ValueError("family id must be a stable, namespaced-safe identifier")
    snapshot_id = snapshot.get("snapshot_id")
    digest = snapshot.get("sha256")
    state = snapshot.get("status")
    if not isinstance(snapshot_id, str) or not snapshot_id or not _valid_sha256(digest):
        raise ValueError("family snapshots require snapshot_id and a SHA-256 digest")
    if state not in {"live_valid", "locked_excluded", "not_applicable", "blocked", "unknown"}:
        raise ValueError("family snapshot has an invalid completeness state")
    inventory = _load_json_object(inventory_path)
    families = inventory.get("source_families")
    if not isinstance(families, dict):
        families = {}
    prior = families.get(family_id)
    superseded: list[str] = []
    if isinstance(prior, dict):
        superseded = [str(value) for value in prior.get("superseded_snapshot_ids", []) if str(value)]
        previous_id = prior.get("snapshot_id")
        if isinstance(previous_id, str) and previous_id and previous_id != snapshot_id:
            superseded.append(previous_id)
    # Preserve chronology but never duplicate an immutable snapshot reference.
    superseded = list(dict.fromkeys(superseded))
    family_projection = {
        **dict(snapshot),
        "snapshot_id": snapshot_id,
        "sha256": digest,
        "superseded_snapshot_ids": superseded,
    }
    families[family_id] = family_projection
    inventory["schema_version"] = int(inventory.get("schema_version", 1))
    inventory["source_families"] = families
    digest_payload = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    inventory["inventory_sha256"] = canonical_json_digest(digest_payload)
    _atomic_write_json(inventory_path, inventory)
    return family_projection


def _seed_identity(seed: Mapping[str, Any]) -> str:
    for field in ("venueid", "native_id", "note_id", "id"):
        value = seed.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("seed records require a stable native identifier")


def revalidate_seed_records(
    seeds: Sequence[Mapping[str, Any]],
    *,
    current_bindings: Mapping[str, Mapping[str, Any]],
    enumeration_complete: bool,
) -> list[dict[str, Any]]:
    """Keep seed history visible while requiring fresh item or batch evidence."""

    diffs: list[dict[str, Any]] = []
    for seed in seeds:
        native_id = _seed_identity(seed)
        binding = current_bindings.get(native_id)
        if binding is not None:
            source_record_id = binding.get("source_record_id")
            snapshot_digest = binding.get("snapshot_sha256")
            if not isinstance(source_record_id, str) or not _valid_sha256(snapshot_digest):
                binding = None
            else:
                diffs.append(
                    {
                        "native_id": native_id,
                        "state": "retained",
                        "source_record_id": source_record_id,
                        "snapshot_sha256": snapshot_digest,
                    }
                )
                continue
        if enumeration_complete:
            diffs.append(
                {
                    "native_id": native_id,
                    "state": "removed",
                    "absence_proven_by_complete_enumeration": True,
                }
            )
        else:
            diffs.append(
                {
                    "native_id": native_id,
                    "state": "unverified",
                    "blocker": "current_complete_enumeration_not_available",
                }
            )
    return diffs


def scan_source_taxonomy(
    *,
    family_id: str,
    snapshot_sha256: str,
    observed: Mapping[str, Sequence[str]],
    declared: Mapping[str, set[str] | frozenset[str]],
) -> dict[str, Any]:
    """Make every newly observed source/program/media type explicit evidence."""

    if not FAMILY_ID_RE.fullmatch(family_id) or not _valid_sha256(snapshot_sha256):
        raise ValueError("taxonomy scans require a family id and snapshot digest")
    entries: list[dict[str, Any]] = []
    blockers: list[str] = []
    for category in sorted(observed):
        for value in sorted({str(item) for item in observed[category] if str(item)}):
            state = "validated" if value in declared.get(category, set()) else "unknown"
            entry = {
                "family_id": family_id,
                "category": category,
                "value": value,
                "state": state,
                "snapshot_sha256": snapshot_sha256,
            }
            entries.append(entry)
            if state == "unknown":
                blockers.append(f"unknown_taxonomy:{category}:{value}")
    return {
        "entries": entries,
        "blockers": blockers,
        "taxonomy_sha256": canonical_json_digest(entries),
    }


class _SourceHTMLParser(HTMLParser):
    """Extract bounded, data-shaped anchors and explicit taxonomy signals.

    The parser intentionally does not execute scripts or infer an application
    state from a JavaScript shell.  Only rendered markup and explicit data
    attributes become source evidence.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[dict[str, Any]] = []
        self.signals: dict[str, list[str]] = {
            "program_family": [],
            "program_type": [],
            "media_family": [],
        }
        self._anchor: dict[str, Any] | None = None
        self._ignored_depth = 0
        self.visible_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        for attribute, category in (
            ("data-event-family", "program_family"),
            ("data-program-family", "program_family"),
            ("data-session-family", "program_family"),
            ("data-program-type", "program_type"),
            ("data-session-type", "program_type"),
            ("data-media-family", "media_family"),
        ):
            value = _normalise_source_text(attributes.get(attribute, ""))
            if value:
                self.signals[category].append(value.lower())
        if tag.lower() == "a":
            self._anchor = {
                "href": attributes.get("href", ""),
                "native_identifier": next(
                    (
                        attributes[key]
                        for key in ("data-event-id", "data-paper-id", "data-note-id", "data-id", "id")
                        if attributes.get(key)
                    ),
                    "",
                ),
                "label": attributes.get("aria-label", ""),
                "text": [],
            }

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        if tag.lower() == "a" and self._anchor is not None:
            self.anchors.append(self._anchor)
            self._anchor = None

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = _normalise_source_text(data)
        if not text:
            return
        self.visible_text.append(text)
        if self._anchor is not None:
            self._anchor["text"].append(text)


def _normalise_source_text(value: Any, *, limit: int = 512) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _safe_discovered_alias(value: str, *, base_url: str) -> str | None:
    """Return a sanitized HTTP(S) alias, never a script/data/mailto reference."""

    try:
        absolute = urljoin(base_url, value)
        parsed = urlsplit(absolute)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return sanitize_media_url(absolute)


def _stable_native_identifier(value: Any, *, alias: str | None, title: str) -> str:
    candidate = _normalise_source_text(value, limit=160)
    if candidate and re.fullmatch(r"[A-Za-z0-9._:/=-]{1,160}", candidate):
        return f"native:{candidate}"
    if alias:
        return f"alias-sha256:{hashlib.sha256(alias.encode('utf-8')).hexdigest()[:32]}"
    return f"text-sha256:{hashlib.sha256(title.encode('utf-8')).hexdigest()[:32]}"


def _source_record(
    *,
    conference_slug: str,
    family: SourceFamily,
    snapshot: Mapping[str, Any],
    native_identifier: Any,
    title: str,
    aliases: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clean_aliases = sorted({alias for alias in aliases if alias})
    stable_native = _stable_native_identifier(
        native_identifier,
        alias=clean_aliases[0] if clean_aliases else None,
        title=title,
    )
    snapshot_digest = str(snapshot["sha256"])
    record = {
        "source_record_id": immutable_id(f"{conference_slug}:source:{family.id}", stable_native),
        "native_ids": [stable_native],
        "aliases": clean_aliases,
        "family_id": family.id,
        "snapshot_sha256": snapshot_digest,
        "source_evidence": {
            "snapshot_id": snapshot.get("snapshot_id"),
            "snapshot_sha256": snapshot_digest,
            "fetched_at": snapshot.get("fetched_at"),
            "final_url": snapshot.get("final_url"),
            "metadata_path": snapshot.get("metadata_path"),
        },
        "title": title,
    }
    if extra:
        record.update({key: value for key, value in extra.items() if value not in (None, "", [], {})})
    return record


def _extract_html_records(
    *, conference_slug: str,
    family: SourceFamily,
    snapshot: Mapping[str, Any],
    body: bytes,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], bool]:
    parser = _SourceHTMLParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    parser.close()
    base_url = str(snapshot.get("final_url") or family.source_url)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for anchor in parser.anchors[:_MAX_EXTRACTED_RECORDS_PER_FAMILY]:
        href = str(anchor.get("href") or "")
        alias = _safe_discovered_alias(href, base_url=base_url) if href else None
        title = _normalise_source_text(" ".join(anchor.get("text", []))) or _normalise_source_text(anchor.get("label"))
        if not alias and not title:
            continue
        aliases = [alias] if alias else [sanitize_media_url(base_url)]
        record = _source_record(
            conference_slug=conference_slug,
            family=family,
            snapshot=snapshot,
            native_identifier=anchor.get("native_identifier"),
            title=title or aliases[0],
            aliases=aliases,
        )
        if record["source_record_id"] not in seen:
            records.append(record)
            seen.add(record["source_record_id"])
        if alias and _host_is_allowed(urlsplit(alias).hostname or "", ("youtube.com", "youtu.be")):
            parser.signals["media_family"].append("youtube")
    # A visible non-JavaScript page is evidence of a source posture even if it
    # has no anchor records (not a claim of complete enumeration).
    if not records and parser.visible_text:
        records.append(
            _source_record(
                conference_slug=conference_slug,
                family=family,
                snapshot=snapshot,
                native_identifier="page:" + str(snapshot.get("sha256")),
                title=_normalise_source_text(" ".join(parser.visible_text), limit=256),
                aliases=[sanitize_media_url(base_url)],
            )
        )
    # A bounded HTML response does not prove pagination or a rendered API was
    # exhausted.  In particular, never let a static shell become an absence
    # proof for an expected source family.
    return records, parser.signals, False


_CVF_OPENACCESS_DAY_FAMILY_PREFIX = "cvf-openaccess-day-"
# A single prefix covering BOTH the bare "cvf-openaccess" family id and every
# day-suffixed variant ("cvf-openaccess-day-2026-06-05", etc). RESEARCH
# measured that the real corpus's operative family ids are always the
# day-suffixed form (the bare string matches 4 of 4,072 records) -- any site
# that dispatches on this family must prefix-match, not exact-match, or it
# silently does nothing for a real conference run. This is the one
# definition; do not re-derive the literal elsewhere.
_CVF_OPENACCESS_FAMILY_ID_PREFIX = "cvf-openaccess"
_CVF_OPENACCESS_HTML_HREF_RE = re.compile(r"^/content/[^/]+/html/(?P<stem>.+)\.html$")
_CVF_OPENACCESS_PDF_HREF_RE = re.compile(r"^/content/[^/]+/papers/(?P<stem>.+)\.pdf$")
# Bounded, non-nested-quantifier patterns for the per-paper author-search-form
# scan below: a CVF day page's body is already byte-bounded by the family's
# max_bytes fetch limit, and these two additionally cap each captured group's
# length so an adversarial page cannot cause catastrophic regex backtracking.
_MAX_AUTHORS_PER_RECORD = 200
_OPENACCESS_QUERY_AUTHOR_RE = re.compile(r'name="query_author"\s+value="(?P<name>[^"]{0,300})"')
_OPENACCESS_PDF_HREF_IN_TEXT_RE = re.compile(
    r'href="(?P<href>/content/[^"]{1,400}/papers/(?P<stem>[^"/]{1,300})\.pdf)"'
)


def _cvf_openaccess_pdf_alias(href: str, *, base_url: str) -> str | None:
    """Build a validated absolute PDF url without going through
    sanitize_media_url's opaque-token redaction.

    That redaction (``LONG_OPAQUE_RE``) exists to scrub secret-looking
    session tokens from arbitrary discovered urls, and it necessarily
    matches any run of 48+ alnum/underscore/hyphen characters -- exactly
    the shape of a CVF paper filename, which encodes the full human-authored
    title (e.g. ``Xiao_Generalizable_..._CVPR_2026_paper.pdf``). That is
    long but not a secret, and redacting it silently produces an
    unreachable ``.../REDACTED.pdf`` alias. ``href`` is only ever called
    here after matching ``_CVF_OPENACCESS_PDF_HREF_RE`` against a same-site
    relative path scraped from CVF's own public page, so scheme/host/
    credential/query validation below still applies -- only the opaque-
    token rewrite is skipped.
    """

    try:
        absolute = urljoin(base_url, href)
        parsed = urlsplit(absolute)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    if parsed.query or parsed.fragment:
        return None
    return absolute


def _authors_by_pdf_stem(html_text: str) -> dict[str, list[str]]:
    """Group ``query_author`` hidden-input values by the pdf-anchor stem they
    precede on a CVF day-listing page.

    A CVF day page carries one author-search ``<form>`` (with a hidden
    ``query_author`` input) per author, immediately adjacent to that paper's
    title, and each paper ends with its own ``[pdf]`` anchor. Author values
    appearing before the FIRST pdf anchor belong to that first paper; values
    between pdf anchor N and pdf anchor N+1 belong to paper N+1. Both regexes
    scan the same decoded body text once, in position order, so this is a
    single linear pass with no re-parsing of the HTML tree.

    Names are HTML-entity-unescaped and whitespace-normalized; empty names
    are dropped and each paper's list is capped at
    ``_MAX_AUTHORS_PER_RECORD`` authors.
    """

    pdf_matches = list(_OPENACCESS_PDF_HREF_IN_TEXT_RE.finditer(html_text))
    author_matches = list(_OPENACCESS_QUERY_AUTHOR_RE.finditer(html_text))
    result: dict[str, list[str]] = {}
    author_index = 0
    for pdf_match in pdf_matches[:_MAX_EXTRACTED_RECORDS_PER_FAMILY]:
        stem = pdf_match.group("stem")
        boundary = pdf_match.start()
        authors: list[str] = []
        while author_index < len(author_matches) and author_matches[author_index].start() < boundary:
            name = _normalise_source_text(html.unescape(author_matches[author_index].group("name")), limit=160)
            if name and len(authors) < _MAX_AUTHORS_PER_RECORD:
                authors.append(name)
            author_index += 1
        # A stem seen more than once (e.g. a duplicated pdf anchor) keeps its
        # first grouping, matching the same setdefault pattern the caller
        # uses for pdf hrefs -- but author_index still advances above so
        # later stems never re-consume authors already assigned here.
        result.setdefault(stem, authors)
    return result


def _extract_cvf_openaccess_paper_records(
    *, conference_slug: str,
    family: SourceFamily,
    snapshot: Mapping[str, Any],
    body: bytes,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], bool]:
    """Extract one record per real paper on a CVF openaccess day-listing page.

    A CVF day page pairs each paper's title anchor
    (``/content/<venue>/html/<Stem>.html``) with a separate ``[pdf]`` anchor
    (``/content/<venue>/papers/<Stem>.pdf``) sharing the identical filename
    stem.  Both hrefs are read directly from the page and joined by that
    stem -- the PDF url is never derived by string substitution from the
    HTML url -- so a record is only ever produced for a paper whose PDF link
    the page itself declares.  Every other anchor on the page (nav, day
    tabs, author-search forms, supplemental/arXiv links) is ignored.  The
    PDF url is recorded as the record's only alias so downstream selection
    (which picks the lexicographically-first url) always resolves to the
    real PDF, never the HTML abstract page.
    """

    html_text = body.decode("utf-8", errors="replace")
    parser = _SourceHTMLParser()
    parser.feed(html_text)
    parser.close()
    base_url = str(snapshot.get("final_url") or family.source_url)
    titles_by_stem: dict[str, str] = {}
    pdf_hrefs_by_stem: dict[str, str] = {}
    authors_by_stem = _authors_by_pdf_stem(html_text)
    # A CVF day page carries roughly ten anchors per paper (title, pdf, supp,
    # arXiv, bibtex, and one author-search form per author), so slicing the
    # raw anchor list at _MAX_EXTRACTED_RECORDS_PER_FAMILY would silently drop
    # ~30% of the papers on a full day.  Scan every anchor (the response is
    # already byte-bounded by the family's max_bytes), and instead cap the
    # number of emitted paper *records* below.
    for anchor in parser.anchors:
        href = str(anchor.get("href") or "")
        path = urlsplit(href).path
        html_match = _CVF_OPENACCESS_HTML_HREF_RE.match(path)
        if html_match:
            title = _normalise_source_text(" ".join(anchor.get("text", [])))
            if title:
                titles_by_stem.setdefault(html_match.group("stem"), title)
            continue
        pdf_match = _CVF_OPENACCESS_PDF_HREF_RE.match(path)
        if pdf_match:
            pdf_hrefs_by_stem.setdefault(pdf_match.group("stem"), href)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for stem, title in sorted(titles_by_stem.items()):
        if len(records) >= _MAX_EXTRACTED_RECORDS_PER_FAMILY:
            break
        pdf_href = pdf_hrefs_by_stem.get(stem)
        if not pdf_href:
            continue
        pdf_alias = _cvf_openaccess_pdf_alias(pdf_href, base_url=base_url)
        if not pdf_alias or not pdf_alias.lower().endswith(".pdf"):
            continue
        record = _source_record(
            conference_slug=conference_slug,
            family=family,
            snapshot=snapshot,
            native_identifier=f"paper:{stem}",
            title=title,
            aliases=[pdf_alias],
            extra={"authors": authors_by_stem.get(stem, [])},
        )
        if record["source_record_id"] not in seen:
            records.append(record)
            seen.add(record["source_record_id"])
    complete = bool(records)
    return records, parser.signals, complete


def _json_scalar(value: Any) -> str:
    if isinstance(value, Mapping) and "value" in value:
        return _json_scalar(value["value"])
    if isinstance(value, list):
        return ", ".join(_json_scalar(item) for item in value if _json_scalar(item))[:512]
    return _normalise_source_text(value)


def _json_records(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("OpenReview source snapshot must be a JSON object")
    for key in ("notes", "data", "results", "items", "groups"):
        rows = payload.get(key)
        if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows):
            return [row for row in rows if isinstance(row, Mapping)]
    raise ValueError("OpenReview source snapshot has no supported record array")


def _extract_openreview_records(
    *, conference_slug: str,
    family: SourceFamily,
    snapshot: Mapping[str, Any],
    body: bytes,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], bool]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("OpenReview source snapshot is not valid JSON") from exc
    rows = _json_records(payload)
    base_url = str(snapshot.get("final_url") or family.source_url)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    signals: dict[str, list[str]] = {"program_family": [], "program_type": [], "media_family": []}
    for row in rows[:_MAX_EXTRACTED_RECORDS_PER_FAMILY]:
        content = row.get("content") if isinstance(row.get("content"), Mapping) else {}
        native = _json_scalar(row.get("id") or row.get("note_id") or row.get("forum"))
        title = _json_scalar(content.get("title") if isinstance(content, Mapping) else row.get("title"))
        forum = _json_scalar(row.get("forum"))
        venue = _json_scalar(content.get("venue") if isinstance(content, Mapping) else row.get("venue"))
        venue_id = _json_scalar(row.get("venueid") or (content.get("venueid") if isinstance(content, Mapping) else ""))
        if not venue_id and "groups" in payload:
            venue_id = _json_scalar(row.get("id"))
        authors = _json_scalar(content.get("authors") if isinstance(content, Mapping) else row.get("authors"))
        if "groups" in payload and native:
            alias = _safe_discovered_alias(
                f"https://openreview.net/group?id={quote(native, safe='')}", base_url=base_url
            )
        else:
            alias = _safe_discovered_alias(f"https://openreview.net/forum?id={native}", base_url=base_url) if native else None
        record = _source_record(
            conference_slug=conference_slug,
            family=family,
            snapshot=snapshot,
            native_identifier=venue_id or native or forum or title,
            title=title or venue_id or native or "OpenReview record",
            aliases=[alias] if alias else [sanitize_media_url(base_url)],
            extra={"forum": forum, "venue": venue, "venueid": venue_id, "authors": authors},
        )
        if record["source_record_id"] not in seen:
            records.append(record)
            seen.add(record["source_record_id"])
        if "workshop" in (venue_id + " " + venue).lower():
            signals["program_family"].append("workshop")
    complete = (
        isinstance(payload.get("count"), int) and payload["count"] == len(rows) and not payload.get("next")
    ) or (family.state == "not_applicable" and not rows)
    return records, signals, complete


def _field_values_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = {}

    def add(field: str, record: Mapping[str, Any], value: Any) -> None:
        text = _normalise_source_text(value)
        if text:
            values.setdefault(field, []).append(
                {
                    "value": text,
                    "family_id": record["family_id"],
                    "source_record_id": record["source_record_id"],
                }
            )

    for record in records:
        family_id = str(record["family_id"])
        title = record.get("title")
        if family_id == "program-schedule":
            add("event_structure", record, title)
            add("program_family", record, title)
        elif family_id.startswith(_CVF_OPENACCESS_FAMILY_ID_PREFIX):
            add("paper_title", record, title)
            add("authors", record, record.get("authors"))
            add("camera_ready_artifact", record, record.get("aliases", [""])[0])
        elif family_id.startswith("openreview-"):
            add("submission_metadata", record, title)
            add("forum_metadata", record, record.get("forum"))
            add("acceptance_status", record, record.get("venue"))
        elif family_id == "youtube":
            add("media_identifier", record, record.get("aliases", [""])[0])
    return values


def _canonical_program_items(
    records: Sequence[Mapping[str, Any]], *, conference_slug: str
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        native = str(record["native_ids"][0])
        source_record_id = str(record["source_record_id"])
        family_id = str(record["family_id"])
        field_evidence: dict[str, Any] = {
            "title": {
                "authority": family_id,
                "source_record_id": source_record_id,
                "snapshot_sha256": record["snapshot_sha256"],
            }
        }
        # Symmetric with "title" above: the same three-key evidence shape,
        # keyed the same way, so a downstream join can resolve authors with
        # one lookup instead of a second, divergent code path. Omitted
        # entirely (not emitted empty) when the record carries no authors.
        if record.get("authors"):
            field_evidence["authors"] = {
                "authority": family_id,
                "source_record_id": source_record_id,
                "snapshot_sha256": record["snapshot_sha256"],
            }
        items.append(
            {
                "program_item_id": immutable_id(f"{conference_slug}:program-item", f"{family_id}:{native}"),
                "aliases": list(record.get("aliases", [])) or [native],
                "source_record_ids": [source_record_id],
                "field_evidence": field_evidence,
            }
        )
    return items


def _candidate_frontier_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    deterministic_join_family_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Create candidates only from an exact, evidence-preserving title key.

    A title never becomes identity: it is merely a deterministic way to expose
    a bounded cross-family candidate to the governed join boundary.

    ``deterministic_join_family_ids`` names families the policy has declared
    trusted, single-authority catalogs of an identical paper list (see
    ``SourceFamily.deterministic_title_join``); default empty, so by default
    every non-excluded pair is ``pending_llm`` exactly as before.  A pair is
    only ever marked ``included`` (bypassing the governed LLM boundary) when
    BOTH its title group has exactly two records (no 3+-way ambiguity) AND
    both records' families are in this set -- a title shared with any
    non-opted-in family, or by 3+ records even among opted-in families,
    still requires a governed decision.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        title = _normalise_source_text(record.get("title"), limit=256).lower()
        title = re.sub(r"[^a-z0-9]+", " ", title).strip()
        if title:
            grouped.setdefault(title, []).append(record)
    pairs: list[dict[str, Any]] = []
    for title, group in sorted(grouped.items()):
        unambiguous_deterministic_group = (
            len(group) == 2
            and all(str(record["family_id"]) in deterministic_join_family_ids for record in group)
        )
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                if left["family_id"] == right["family_id"]:
                    continue
                record_ids = sorted((str(left["source_record_id"]), str(right["source_record_id"])))
                structural_navigation = title in _STRUCTURAL_NAVIGATION_TITLE_KEYS
                if structural_navigation:
                    inclusion_status = "excluded"
                    reason = (
                        "deterministic structural-navigation title; not a program-item relation or event-family exclusion"
                    )
                elif unambiguous_deterministic_group:
                    inclusion_status = "included"
                    reason = (
                        "deterministic same-corpus title join: both families are declared "
                        "single-authority catalogs of an identical paper list "
                        "(SourceFamily.deterministic_title_join) and this exact normalized "
                        f"title has no 3+-way ambiguity: {title[:160]}"
                    )
                else:
                    inclusion_status = "pending_llm"
                    reason = f"exact normalized source title candidate: {title[:160]}"
                pairs.append(
                    {
                        "canonical_pair_key": "|".join(record_ids),
                        "source_record_ids": record_ids,
                        "source_snapshot_digests": sorted((str(left["snapshot_sha256"]), str(right["snapshot_sha256"]))),
                        "relation_kind": "program_item_cross_source_candidate",
                        "inclusion_status": inclusion_status,
                        "reason": reason,
                    }
                )
    return build_candidate_frontier(pairs)


def _workshop_seed_rows(workspace: Path) -> list[Mapping[str, Any]]:
    path = workspace / "storage" / "workshops" / "inventory.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, Mapping) and isinstance(payload.get("workshops"), list):
        return [row for row in payload["workshops"] if isinstance(row, Mapping)]
    return []


def extract_source_discovery_inventory(
    *,
    workspace: Path,
    run_id: str,
    policy: SourcePolicy,
    family_snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Convert saved snapshots into a bounded, evidence-linked source spine.

    This function performs no network or provider work.  It refuses to turn an
    empty or JavaScript-only response into a complete enumeration, and keeps
    every observed but undeclared taxonomy signal as a blocker.
    """

    resolved_workspace = workspace.resolve()
    # The workspace directory name is this platform's established slug
    # convention (conferences/<slug>/); every generated record/program-item
    # id is namespaced under it so ids never claim a conference other than
    # the one that actually produced them. Sanitized to immutable_id's
    # required namespace shape ([a-z0-9][a-z0-9:-]*) since an arbitrary
    # directory name (a test fixture's tmp_path, for instance) need not
    # already satisfy it.
    conference_slug = re.sub(r"[^a-z0-9-]", "-", resolved_workspace.name.lower()).strip("-") or "conference"
    records: list[dict[str, Any]] = []
    family_rows: dict[str, dict[str, Any]] = {}
    taxonomy_entries: list[dict[str, Any]] = []
    blockers: list[str] = []
    workshop_complete = False
    for family_id, family in sorted(policy.families.items()):
        raw_snapshot = family_snapshots.get(family_id)
        if not isinstance(raw_snapshot, Mapping):
            family_rows[family_id] = {"status": "blocked", "snapshot_id": "missing", "sha256": "0" * 64}
            blockers.append(f"source_snapshot_missing:{family_id}")
            continue
        snapshot = dict(raw_snapshot)
        try:
            digest = snapshot.get("sha256")
            if not _valid_sha256(digest):
                raise ValueError("snapshot digest missing")
            body_path = snapshot.get("body_path")
            if not isinstance(body_path, str) or not body_path:
                raise ValueError("snapshot body path missing")
            candidate = Path(body_path)
            candidate = candidate.resolve() if candidate.is_absolute() else (resolved_workspace / candidate).resolve()
            candidate.relative_to(resolved_workspace)
            body = candidate.read_bytes()
            if hashlib.sha256(body).hexdigest() != digest:
                raise ValueError("snapshot body digest mismatch")
            if family_id.startswith(_CVF_OPENACCESS_DAY_FAMILY_PREFIX):
                extracted, observed, complete = _extract_cvf_openaccess_paper_records(
                    conference_slug=conference_slug, family=family, snapshot=snapshot, body=body
                )
            elif family.source_kind == "json":
                extracted, observed, complete = _extract_openreview_records(
                    conference_slug=conference_slug, family=family, snapshot=snapshot, body=body
                )
            else:
                extracted, observed, complete = _extract_html_records(
                    conference_slug=conference_slug, family=family, snapshot=snapshot, body=body
                )
            observed.setdefault("source_link_class", []).append(_FAMILY_LINK_CLASS.get(family_id, "official"))
            scan = scan_source_taxonomy(
                family_id=family_id,
                snapshot_sha256=str(digest),
                observed=observed,
                declared=_DECLARED_TAXONOMY,
            )
            taxonomy_entries.extend(scan["entries"])
            blockers.extend(scan["blockers"])
            if family.state == "not_applicable":
                if extracted or not complete:
                    status = "blocked"
                    exclusion_id = None
                    blockers.append(f"source_not_applicable_unproven:{family_id}")
                else:
                    status = "not_applicable"
                    exclusion_id = None
            elif family.auth_posture == "registration_gated" and extracted:
                status = "locked_excluded"
                exclusion_id = "paywalled_official_video"
            elif not extracted:
                status = "blocked"
                exclusion_id = None
                blockers.append(f"source_record_extraction_incomplete:{family_id}")
            else:
                status = "live_valid"
                exclusion_id = None
            family_rows[family_id] = {
                **snapshot,
                "status": status,
                "exclusion_id": exclusion_id,
                "complete_enumeration": bool(complete),
                "source_record_count": len(extracted),
                "taxonomy_evidence": scan["taxonomy_sha256"],
            }
            records.extend(extracted)
            workshop_complete = family_id == "openreview-workshop" and bool(complete) and status == "live_valid"
        except (OSError, ValueError, UnicodeError) as exc:
            snapshot.update({"status": "blocked", "complete_enumeration": False, "extraction_error": type(exc).__name__})
            family_rows[family_id] = snapshot
            blockers.append(f"source_record_extraction_failed:{family_id}")

    records.sort(key=lambda record: str(record["source_record_id"]))
    current_bindings: dict[str, dict[str, str]] = {}
    for record in records:
        binding = {
            "source_record_id": str(record["source_record_id"]),
            "snapshot_sha256": str(record["snapshot_sha256"]),
        }
        for native in record.get("native_ids", []):
            if not isinstance(native, str):
                continue
            current_bindings[native] = binding
            if native.startswith("native:"):
                current_bindings[native.removeprefix("native:")] = binding
    seed_diffs = revalidate_seed_records(
        _workshop_seed_rows(resolved_workspace),
        current_bindings=current_bindings,
        enumeration_complete=workshop_complete,
    )
    taxonomy_entries.sort(key=lambda row: (row["family_id"], row["category"], row["value"]))
    taxonomy = {
        "entries": taxonomy_entries,
        "blockers": sorted(set(blockers)),
        "taxonomy_sha256": canonical_json_digest(taxonomy_entries),
    }
    deterministic_join_family_ids = frozenset(
        family_id for family_id, family in policy.families.items() if family.deterministic_title_join
    )
    frontier = _candidate_frontier_from_records(
        records, deterministic_join_family_ids=deterministic_join_family_ids
    )
    field_values = _field_values_from_records(records)
    all_blockers = sorted(set(blockers))
    return {
        "status": "succeeded" if not all_blockers and not any(diff["state"] == "unverified" for diff in seed_diffs) else "blocked",
        "family_snapshots": family_rows,
        "source_records": records,
        "program_items": _canonical_program_items(records, conference_slug=conference_slug),
        "seed_diffs": seed_diffs,
        "taxonomy": taxonomy,
        "candidate_frontier": frontier,
        "field_values": field_values,
        "blockers": all_blockers + [
            f"seed_record_unverified:{diff['native_id']}" for diff in seed_diffs if diff["state"] == "unverified"
        ],
    }


def build_candidate_frontier(
    pairs: Sequence[Mapping[str, Any]],
    *,
    expected_pair_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Create a deterministic, exhaustive ledger of considered cross-source pairs."""

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in pairs:
        pair_key = raw.get("canonical_pair_key")
        relation_kind = raw.get("relation_kind")
        status = raw.get("inclusion_status")
        reason = raw.get("reason")
        record_ids = raw.get("source_record_ids")
        digests = raw.get("source_snapshot_digests")
        if not isinstance(pair_key, str) or not pair_key or pair_key in seen:
            raise ValueError("candidate frontier pairs require unique canonical_pair_key values")
        if not isinstance(relation_kind, str) or not relation_kind:
            raise ValueError("candidate frontier pairs require a relation_kind")
        if status not in {"included", "excluded", "pending_llm", "unresolved"}:
            raise ValueError("candidate frontier pairs require an inclusion_status")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("candidate frontier pairs require an explicit reason")
        if not isinstance(record_ids, Sequence) or isinstance(record_ids, (str, bytes)) or len(record_ids) < 2:
            raise ValueError("candidate frontier pairs require at least two source record ids")
        if not isinstance(digests, Sequence) or isinstance(digests, (str, bytes)) or not all(
            _valid_sha256(value) for value in digests
        ):
            raise ValueError("candidate frontier pairs require source snapshot digests")
        seen.add(pair_key)
        normalized.append(
            {
                "canonical_pair_key": pair_key,
                "source_record_ids": sorted(str(value) for value in record_ids),
                "source_snapshot_digests": sorted(str(value) for value in digests),
                "relation_kind": relation_kind,
                "inclusion_status": status,
                "reason": reason.strip(),
            }
        )
    if expected_pair_keys is not None and set(expected_pair_keys) != seen:
        raise ValueError("candidate frontier is missing or contains unexplained expected pairs")
    normalized.sort(key=lambda pair: pair["canonical_pair_key"])
    return {"pairs": normalized, "frontier_sha256": canonical_json_digest(normalized)}


def _normalized_field_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _bounded_join_text(value: Any, *, limit: int = 512) -> str:
    """Keep candidate metadata data-shaped, bounded, and secret-redacted.

    This is deliberately not prompt concatenation: callers serialize the
    returned mapping as JSON for a structured provider boundary.  The source
    text remains data, while receipt and inventory projections retain only
    the digest of this value.
    """

    text = " ".join(str(value).split())[:limit]
    redacted = sanitize_for_receipt(text)
    return str(redacted)[:limit]


def _frontier_pairs(frontier: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    pairs = frontier.get("pairs")
    digest = frontier.get("frontier_sha256")
    if not isinstance(pairs, list) or not _valid_sha256(digest):
        raise ValueError("governed reconciliation requires a digest-pinned candidate frontier")
    if canonical_json_digest(pairs) != digest:
        raise ValueError("governed reconciliation candidate frontier digest mismatch")
    normalized = [dict(pair) for pair in pairs if isinstance(pair, Mapping)]
    if len(normalized) != len(pairs):
        raise ValueError("governed reconciliation candidate frontier pairs must be objects")
    return normalized, str(digest)


def _normalized_candidate_evidence(
    pair_key: str,
    candidate_evidence: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> list[dict[str, str]]:
    """Return the bounded, canonical evidence rows for one candidate pair."""

    evidence_rows: list[dict[str, str]] = []
    for row in candidate_evidence.get(pair_key, ()) if candidate_evidence else ():
        if not isinstance(row, Mapping):
            raise ValueError("candidate evidence entries must be objects")
        field = row.get("field")
        source_record_id = row.get("source_record_id")
        if not isinstance(field, str) or not field.strip() or not isinstance(source_record_id, str) or not source_record_id:
            raise ValueError("candidate evidence requires field and source_record_id")
        value = _bounded_join_text(row.get("value", ""))
        evidence_rows.append(
            {
                "field": _normalized_field_name(field),
                "source_record_id": source_record_id,
                "value": value,
                "value_sha256": canonical_json_digest(value),
            }
        )
    evidence_rows.sort(key=lambda row: (row["field"], row["source_record_id"], row["value_sha256"]))
    return evidence_rows


def build_candidate_evidence_projection(
    frontier: Mapping[str, Any],
    *,
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Project bounded canonical-record evidence for every retained candidate.

    Source snapshots remain the sole raw-evidence layer.  The provider receives
    only stable titles and aliases already present in the canonical source
    records; the request builder later redacts and hashes this projection.
    """

    pairs, _ = _frontier_pairs(frontier)
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in source_records:
        record_id = record.get("source_record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("canonical source evidence requires source_record_id")
        if record_id in by_id:
            raise ValueError("canonical source evidence contains duplicate source_record_id")
        by_id[record_id] = record

    projection: dict[str, list[dict[str, str]]] = {}
    for pair in pairs:
        if pair.get("inclusion_status") == "excluded":
            continue
        pair_key = str(pair["canonical_pair_key"])
        rows: list[dict[str, str]] = []
        for record_id in pair["source_record_ids"]:
            record = by_id.get(str(record_id))
            if record is None:
                raise ValueError("candidate source evidence is missing a canonical record")
            title = record.get("title")
            if isinstance(title, str) and title.strip():
                rows.append({"field": "title", "source_record_id": str(record_id), "value": title})
            aliases = record.get("aliases")
            if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
                for alias in list(aliases)[:4]:
                    if isinstance(alias, str) and alias.strip():
                        rows.append({"field": "aliases", "source_record_id": str(record_id), "value": alias})
        normalized = _normalized_candidate_evidence(pair_key, {pair_key: rows})
        if not normalized:
            raise ValueError("candidate source evidence is empty")
        projection[pair_key] = normalized
    return projection


def frontier_join_input_digest(
    frontier: Mapping[str, Any],
    pair: Mapping[str, Any],
    *,
    evidence: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Reconstruct a candidate's immutable request-boundary digest."""

    _, frontier_digest = _frontier_pairs(frontier)
    normalized_evidence = _normalized_candidate_evidence(
        str(pair.get("canonical_pair_key") or ""),
        {str(pair.get("canonical_pair_key") or ""): evidence or ()},
    )
    return canonical_json_digest(
        {
            "frontier_sha256": frontier_digest,
            "canonical_pair_key": pair.get("canonical_pair_key"),
            "source_record_ids": pair.get("source_record_ids"),
            "source_snapshot_digests": pair.get("source_snapshot_digests"),
            "relation_kind": pair.get("relation_kind"),
            "evidence": normalized_evidence,
        }
    )


def build_cross_source_join_request(
    frontier: Mapping[str, Any],
    *,
    candidate_evidence: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build a strict, bounded JSON request without retaining source bodies.

    The request is a transport-neutral input contract for Plan 07.  It does
    not invoke a provider and callers must record only its digest in receipts.
    """

    pairs, frontier_digest = _frontier_pairs(frontier)
    candidates: list[dict[str, Any]] = []
    for pair in pairs:
        # Only pending_llm pairs need a governed decision: "excluded" pairs
        # never represent a real relation, and "included" pairs are already
        # deterministically resolved (see SourceFamily.deterministic_title_join)
        # -- sending either to the paid boundary would waste a call on a
        # question that is not actually pending.
        if pair.get("inclusion_status") != "pending_llm":
            continue
        pair_key = pair.get("canonical_pair_key")
        if not isinstance(pair_key, str):
            raise ValueError("candidate frontier pair has no stable key")
        evidence_rows = _normalized_candidate_evidence(pair_key, candidate_evidence)
        candidates.append(
            {
                "canonical_pair_key": pair_key,
                "source_record_ids": list(pair["source_record_ids"]),
                "source_snapshot_digests": list(pair["source_snapshot_digests"]),
                "relation_kind": pair["relation_kind"],
                "input_sha256": frontier_join_input_digest(frontier, pair, evidence=evidence_rows),
                "evidence": evidence_rows,
            }
        )
    request = {
        "schema": "source_discovery.cross_source_join.v1",
        "task_id": CROSS_SOURCE_JOIN_TASK_ID,
        "frontier_sha256": frontier_digest,
        "candidates": candidates,
    }
    request["request_sha256"] = canonical_json_digest(request)
    return request


def _contract_payload(value: Any) -> dict[str, Any]:
    """Normalize a contract result without accepting an arbitrary object."""

    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {
        "status": "blocked",
        "blockers": [{"code": "precondition_contract_unmet", "precondition": "contract_result"}],
    }


def _model_registry_digest() -> str:
    registry_path = Path(__file__).with_name("model_governance") / "registry.yaml"
    return hashlib.sha256(registry_path.read_bytes()).hexdigest()


def _is_valid_substitution_waiver(
    waiver: Mapping[str, Any] | None,
    *,
    run_id: str,
    frontier_sha256: str,
    expected_selection: tuple[str, str],
    actual_selection: tuple[str, str],
) -> tuple[bool, str | None]:
    """Accept only an immutable, current waiver scoped to this exact join."""

    if not isinstance(waiver, Mapping):
        return False, None
    waiver_sha256 = waiver.get("waiver_sha256")
    payload = {key: value for key, value in waiver.items() if key != "waiver_sha256"}
    expected = waiver.get("from")
    actual = waiver.get("to")
    expiry = waiver.get("expiry")
    try:
        expires_at = datetime.fromisoformat(str(expiry).replace("Z", "+00:00"))
    except ValueError:
        return False, None
    if expires_at.tzinfo is None or expires_at <= datetime.now(UTC):
        return False, None
    valid = (
        isinstance(waiver_sha256, str)
        and _valid_sha256(waiver_sha256)
        and canonical_json_digest(payload) == waiver_sha256
        and isinstance(waiver.get("waiver_path"), str)
        and _safe_receipt_path(waiver.get("waiver_path"))
        and waiver.get("task_id") == CROSS_SOURCE_JOIN_TASK_ID
        and waiver.get("run_id") == run_id
        and waiver.get("model_registry_sha256") == _model_registry_digest()
        and waiver.get("candidate_frontier_sha256") == frontier_sha256
        and isinstance(expected, Mapping)
        and isinstance(actual, Mapping)
        and (expected.get("provider"), expected.get("model")) == expected_selection
        and (actual.get("provider"), actual.get("model")) == actual_selection
        and all(isinstance(waiver.get(field), str) and waiver.get(field).strip() for field in ("approver", "rationale", "cost_impact"))
    )
    return valid, str(waiver_sha256) if valid else None


def execute_governed_frontier_joins(
    *,
    workspace: str | Path,
    run_id: str,
    frontier: Mapping[str, Any],
    frontier_authority: Mapping[str, Any] | None,
    base_full_estimate_digest: str,
    volatile_preflight: Mapping[str, Any] | None,
    pre_spend_validator: Callable[..., Any] = verify_pre_spend_contract,
    provider_adapter_factory: Callable[..., Any],
    paid_dispatch: Callable[..., Mapping[str, Any]],
    field_values: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    candidate_evidence: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    substitution_waiver: Mapping[str, Any] | None = None,
    prior_decisions: Sequence[Mapping[str, Any]] = (),
    prior_receipt_paths: Sequence[str] = (),
    prior_actual_cost_usd: float = 0.0,
    decision_checkpoint: Callable[[Sequence[Mapping[str, Any]], Sequence[str], float], None] | None = None,
) -> dict[str, Any]:
    """Dispatch exactly the authorized, receipt-bound joins for one frontier.

    The pre-spend validator deliberately runs before an adapter is constructed.
    ``paid_dispatch`` is the runner's bound ``dispatch_paid_call`` bridge: it
    owns named-cap enforcement and invokes the supplied owner only after its
    own fresh authorization checks pass.  Keeping both boundaries injectable
    makes the security ordering executable without a provider SDK in tests.
    """

    request = build_cross_source_join_request(frontier, candidate_evidence=candidate_evidence)
    candidates = list(request["candidates"])
    if not candidates:
        reconciliation = reconcile_cross_source_frontier(
            frontier=frontier,
            decisions=[],
            field_values=field_values or {},
            candidate_evidence=candidate_evidence,
        )
        return {
            "status": "succeeded" if not reconciliation["blockers"] else "blocked",
            "pre_spend": {"status": "not_required", "provider_calls_made": 0, "actual_cost_usd": 0.0},
            "provider_calls_made": 0,
            "actual_cost_usd": 0.0,
            "decisions": [],
            "provider_receipt_paths": [],
            "boundary_method": GOVERNED_JOIN_METHOD,
            "reconciliation": reconciliation,
            "blockers": [{"code": blocker} for blocker in reconciliation["blockers"]],
        }

    existing_decisions = [dict(decision) for decision in prior_decisions if isinstance(decision, Mapping)]
    if len(existing_decisions) != len(prior_decisions):
        return {
            "status": "blocked",
            "pre_spend": {"status": "not_run"},
            "provider_calls_made": 0,
            "new_provider_calls_made": 0,
            "reused_provider_calls": 0,
            "actual_cost_usd": 0.0,
            "decisions": [],
            "provider_receipt_paths": [],
            "boundary_method": GOVERNED_JOIN_METHOD,
            "reconciliation": None,
            "blockers": [{"code": "partial_decision_ledger_incompatible"}],
        }
    retained_keys = {str(candidate["canonical_pair_key"]) for candidate in candidates}
    existing_keys = [str(decision.get("canonical_pair_key") or "") for decision in existing_decisions]
    if (
        len(set(existing_keys)) != len(existing_keys)
        or any(key not in retained_keys for key in existing_keys)
        or len(prior_receipt_paths) != len(existing_decisions)
        or not isinstance(prior_actual_cost_usd, (int, float))
        or isinstance(prior_actual_cost_usd, bool)
        or not math.isfinite(float(prior_actual_cost_usd))
        or float(prior_actual_cost_usd) < 0
    ):
        return {
            "status": "blocked",
            "pre_spend": {"status": "not_run"},
            "provider_calls_made": 0,
            "new_provider_calls_made": 0,
            "reused_provider_calls": 0,
            "actual_cost_usd": 0.0,
            "decisions": [],
            "provider_receipt_paths": [],
            "boundary_method": GOVERNED_JOIN_METHOD,
            "reconciliation": None,
            "blockers": [{"code": "partial_decision_ledger_incompatible"}],
        }
    pending_candidates = [
        candidate for candidate in candidates if str(candidate["canonical_pair_key"]) not in set(existing_keys)
    ]

    policy = load_registry().get(CROSS_SOURCE_JOIN_TASK_ID)
    provider = default_provider(CROSS_SOURCE_JOIN_TASK_ID)
    model = default_model(CROSS_SOURCE_JOIN_TASK_ID)
    pre_spend = _contract_payload(
        pre_spend_validator(
            workspace=workspace,
            run_id=run_id,
            stage_id="source-discovery",
            mode="live",
            frontier_authority=frontier_authority,
            base_full_estimate_digest=base_full_estimate_digest,
            candidate_frontier_sha256=request["frontier_sha256"],
            retained_join_quantity=len(candidates),
            provider=provider,
            provider_stage=policy.stage,
            volatile_preflight=volatile_preflight,
        )
    )
    if pre_spend.get("status") != "precondition_passed":
        blockers = pre_spend.get("blockers")
        return {
            "status": "blocked",
            "pre_spend": pre_spend,
            "provider_calls_made": 0,
            "actual_cost_usd": 0.0,
            "decisions": [],
            "provider_receipt_paths": [],
            "boundary_method": GOVERNED_JOIN_METHOD,
            "reconciliation": None,
            "blockers": list(blockers) if isinstance(blockers, list) else [{"code": "precondition_contract_unmet"}],
        }

    if not pending_candidates:
        reconciliation = reconcile_cross_source_frontier(
            frontier=frontier,
            decisions=existing_decisions,
            field_values=field_values or {},
            candidate_evidence=candidate_evidence,
        )
        blockers = [{"code": blocker} for blocker in reconciliation["blockers"]]
        return {
            "status": "blocked" if blockers else "succeeded",
            "pre_spend": pre_spend,
            "provider_calls_made": len(existing_decisions),
            "new_provider_calls_made": 0,
            "reused_provider_calls": len(existing_decisions),
            "actual_cost_usd": round(float(prior_actual_cost_usd), 6),
            "decisions": existing_decisions,
            "provider_receipt_paths": [str(path) for path in prior_receipt_paths],
            "boundary_method": GOVERNED_JOIN_METHOD,
            "reconciliation": reconciliation,
            "blockers": blockers,
        }

    adapter = provider_adapter_factory(
        task_id=CROSS_SOURCE_JOIN_TASK_ID,
        provider=provider,
        model=model,
    )
    actual_selection = (getattr(adapter, "provider", None), getattr(adapter, "model", None))
    if not all(isinstance(value, str) and value for value in actual_selection):
        return {
            "status": "blocked",
            "pre_spend": pre_spend,
            "provider_calls_made": 0,
            "actual_cost_usd": 0.0,
            "decisions": [],
            "provider_receipt_paths": [],
            "boundary_method": GOVERNED_JOIN_METHOD,
            "reconciliation": None,
            "blockers": [{"code": "unauthorized_provider_substitution"}],
        }
    actual_selection = (str(actual_selection[0]), str(actual_selection[1]))
    fallback_selection = (
        (policy.fallback_provider, default_model_for_provider(CROSS_SOURCE_JOIN_TASK_ID, policy.fallback_provider))
        if policy.fallback_provider
        else None
    )
    waiver_valid, waiver_digest = _is_valid_substitution_waiver(
        substitution_waiver,
        run_id=run_id,
        frontier_sha256=str(request["frontier_sha256"]),
        expected_selection=(provider, model),
        actual_selection=actual_selection,
    )
    if actual_selection not in {(provider, model), fallback_selection} and not waiver_valid:
        return {
            "status": "blocked",
            "pre_spend": pre_spend,
            "provider_calls_made": 0,
            "actual_cost_usd": 0.0,
            "decisions": [],
            "provider_receipt_paths": [],
            "boundary_method": GOVERNED_JOIN_METHOD,
            "reconciliation": None,
            "blockers": [{"code": "unauthorized_provider_substitution"}],
        }

    decisions: list[dict[str, Any]] = list(existing_decisions)
    receipt_paths: list[str] = [str(path) for path in prior_receipt_paths]
    total_cost = float(prior_actual_cost_usd)
    provider_calls_made = len(existing_decisions)
    new_provider_calls_made = 0
    blockers: list[dict[str, Any]] = []
    for candidate in pending_candidates:
        call_result: dict[str, Any] = {}

        def owner(candidate: Mapping[str, Any] = candidate) -> float:
            response = adapter.invoke(dict(candidate))
            if not isinstance(response, Mapping):
                raise ValueError("governed join adapter must return an object")
            call_result.update(response)
            receipt = response.get("receipt")
            if not isinstance(receipt, Mapping):
                raise ValueError("governed join adapter response requires a receipt")
            cost = receipt.get("estimated_cost_usd", 0.0)
            if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(float(cost)) or float(cost) < 0:
                raise ValueError("governed join receipt requires a finite non-negative actual cost")
            return float(cost)

        dispatch = dict(
            paid_dispatch(
                task_id=CROSS_SOURCE_JOIN_TASK_ID,
                provider=actual_selection[0],
                model=actual_selection[1],
                owner=owner,
            )
        )
        if not dispatch.get("authorized"):
            dispatch_blockers = dispatch.get("blockers")
            blockers.extend(
                list(dispatch_blockers) if isinstance(dispatch_blockers, list) else [{"code": "paid_dispatch_denied"}]
            )
            break
        if not call_result:
            blockers.append({"code": "provider_dispatch_missing_call_result"})
            break
        provider_calls_made += 1
        new_provider_calls_made += 1
        receipt = call_result.get("receipt")
        if not isinstance(receipt, Mapping):
            blockers.append({"code": "provider_receipt_missing"})
            break
        receipt_provider = receipt.get("provider")
        receipt_model = receipt.get("model")
        if (receipt_provider, receipt_model) != actual_selection:
            blockers.append({"code": "provider_receipt_selection_mismatch"})
            break
        receipt_path = receipt.get("receipt_path")
        if not _safe_receipt_path(receipt_path):
            blockers.append({"code": "provider_receipt_path_invalid"})
            break
        output = {
            key: call_result.get(key)
            for key in ("decision", "confidence", "conflict_state")
        }
        decision = {
            "canonical_pair_key": candidate["canonical_pair_key"],
            "source_record_ids": list(candidate["source_record_ids"]),
            "frontier_sha256": request["frontier_sha256"],
            "evidence_digests": list(candidate["source_snapshot_digests"]),
            "join_method": GOVERNED_JOIN_METHOD,
            "boundary_method": GOVERNED_JOIN_METHOD,
            "decision": output["decision"],
            "confidence": output["confidence"],
            "conflict_state": output["conflict_state"],
            "registry_task_id": CROSS_SOURCE_JOIN_TASK_ID,
            "actual_provider": receipt_provider,
            "actual_model": receipt_model,
            "provider_receipt_path": str(receipt_path),
            "provider_receipt_sha256": canonical_json_digest(sanitize_for_receipt(dict(receipt))),
            "input_sha256": candidate["input_sha256"],
            "output_sha256": canonical_json_digest(sanitize_for_receipt(output)),
        }
        if waiver_digest:
            decision["substitution_waiver_sha256"] = waiver_digest
        decisions.append(decision)
        receipt_paths.append(str(receipt_path))
        actual_cost = dispatch.get("actual_cost_usd", 0.0)
        if not isinstance(actual_cost, (int, float)) or isinstance(actual_cost, bool) or not math.isfinite(float(actual_cost)) or float(actual_cost) < 0:
            blockers.append({"code": "actual_cost_invalid"})
            break
        total_cost += float(actual_cost)
        if decision_checkpoint is not None:
            try:
                decision_checkpoint(decisions, receipt_paths, total_cost)
            except (OSError, ValueError, TypeError):
                blockers.append({"code": "partial_decision_ledger_checkpoint_failed"})
                break

    reconciliation = reconcile_cross_source_frontier(
        frontier=frontier,
        decisions=decisions,
        field_values=field_values or {},
        candidate_evidence=candidate_evidence,
        approved_substitution_waiver_digests=frozenset({waiver_digest}) if waiver_digest else frozenset(),
    )
    blockers.extend({"code": blocker} for blocker in reconciliation["blockers"])
    return {
        "status": "blocked" if blockers else "succeeded",
        "pre_spend": pre_spend,
        "provider_calls_made": provider_calls_made,
        "new_provider_calls_made": new_provider_calls_made,
        "reused_provider_calls": len(existing_decisions),
        "actual_cost_usd": round(total_cost, 6),
        "decisions": decisions,
        "provider_receipt_paths": receipt_paths,
        "boundary_method": GOVERNED_JOIN_METHOD,
        "reconciliation": reconciliation,
        "blockers": blockers,
    }


def _resolve_field_authority(
    field_values: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Resolve field winners while retaining every competing value and reason."""

    canonical_fields: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    blockers: set[str] = set()
    for raw_field in sorted(field_values):
        field = _normalized_field_name(raw_field)
        rows = field_values[raw_field]
        candidates: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("field authority candidates must be objects")
            family_id = row.get("family_id")
            source_record_id = row.get("source_record_id")
            if not isinstance(family_id, str) or not isinstance(source_record_id, str):
                raise ValueError("field authority candidates require family_id and source_record_id")
            value = _bounded_join_text(row.get("value", ""))
            candidates.append(
                {
                    "family_id": family_id,
                    "source_record_id": source_record_id,
                    "snapshot_sha256": row.get("snapshot_sha256") if _valid_sha256(row.get("snapshot_sha256")) else None,
                    "value": value,
                    "value_sha256": canonical_json_digest(value),
                }
            )
        if not candidates:
            continue
        authority = _FIELD_AUTHORITIES.get(field, ())
        candidates.sort(
            key=lambda row: (
                0 if row["family_id"] in authority else 1,
                row["family_id"],
                row["source_record_id"],
                row["value_sha256"],
            )
        )
        winner = candidates[0]
        distinct_values = {row["value_sha256"] for row in candidates}
        has_conflict = len(distinct_values) > 1
        material = has_conflict and field in _MATERIAL_CONFLICT_FIELDS
        conflict_state = "material" if material else ("warning" if has_conflict else "none")
        if material:
            blockers.add(f"material_conflict:{field}")
        winner_reason = (
            f"{winner['family_id']} is field-authoritative for {field}"
            if winner["family_id"] in authority
            else f"no configured authority for {field}; stable provenance ordering selected {winner['family_id']}"
        )
        canonical_fields[field] = {**winner, "winner_reason": winner_reason, "conflict_state": conflict_state}
        ledger.append(
            {
                "field": field,
                "authority_families": list(authority),
                "canonical_source_record_id": winner["source_record_id"],
                "canonical_value_sha256": winner["value_sha256"],
                "canonical_value": winner["value"],
                "winner_reason": winner_reason,
                "conflict_state": conflict_state,
                "disposition": "blocked_material_conflict" if material else ("warning_retained" if has_conflict else "no_conflict"),
                "competing_values": candidates,
            }
        )
    return canonical_fields, ledger, blockers


def _safe_receipt_path(value: Any) -> bool:
    path = Path(str(value))
    return bool(str(value)) and not path.is_absolute() and ".." not in path.parts


def reconcile_cross_source_frontier(
    *,
    frontier: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    field_values: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_evidence: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    approved_substitution_waiver_digests: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Validate receipt-bound governed decisions against every frontier row.

    This pure function deliberately stops before transport, authorization, and
    client construction.  Plan 07 supplies actual provider receipts; this
    layer prevents an incomplete, mismatched, low-confidence, or materially
    conflicting decision from changing canonical identity.
    """

    pairs, frontier_digest = _frontier_pairs(frontier)
    canonical_fields, conflict_ledger, material_field_conflict_blockers = _resolve_field_authority(field_values)
    # Keep field-authority material conflicts distinct from process-wide
    # validation blockers.  A rejected decision for one candidate must never
    # make an unrelated, otherwise clean candidate appear materially
    # conflicting; only D-07 material source-field conflicts have that effect.
    blockers = set(material_field_conflict_blockers)
    policy = load_registry().get(CROSS_SOURCE_JOIN_TASK_ID)
    default_selection = (default_provider(CROSS_SOURCE_JOIN_TASK_ID), default_model(CROSS_SOURCE_JOIN_TASK_ID))
    fallback_selection: tuple[str, str] | None = None
    if policy.fallback_provider:
        fallback_selection = (
            policy.fallback_provider,
            default_model_for_provider(CROSS_SOURCE_JOIN_TASK_ID, policy.fallback_provider),
        )
    threshold = policy.confidence_threshold
    if threshold is None or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("cross-source join policy requires a finite confidence threshold")

    retained = {
        str(pair["canonical_pair_key"]): pair
        for pair in pairs
        if pair.get("inclusion_status") != "excluded"
    }
    excluded_pairs = [
        str(pair["canonical_pair_key"])
        for pair in pairs
        if pair.get("inclusion_status") == "excluded"
    ]
    for pair in pairs:
        if pair.get("inclusion_status") == "excluded" and not (
            isinstance(pair.get("reason"), str) and pair["reason"].strip()
        ):
            blockers.add(f"unexplained_excluded_pair:{pair.get('canonical_pair_key', 'unknown')}")
        if pair.get("inclusion_status") == "included" and not (
            isinstance(pair.get("reason"), str) and pair["reason"].strip()
        ):
            blockers.add(f"unexplained_included_pair:{pair.get('canonical_pair_key', 'unknown')}")
    decisions_by_key: dict[str, Mapping[str, Any]] = {}
    for decision in decisions:
        key = decision.get("canonical_pair_key")
        # An "included" pair is already deterministically resolved (see
        # SourceFamily.deterministic_title_join) and must never also receive
        # a governed decision -- that would mean two different mechanisms
        # claim authority over the same identity change.
        if (
            not isinstance(key, str)
            or key not in retained
            or retained[key].get("inclusion_status") == "included"
        ):
            blockers.add(f"unexpected_decision:{key or 'unknown'}")
            continue
        if key in decisions_by_key:
            blockers.add(f"duplicate_decision:{key}")
            continue
        decisions_by_key[key] = decision

    typed_edges: list[dict[str, Any]] = []
    canonical_identity_updates: list[dict[str, Any]] = []
    required_fields = {
        "canonical_pair_key",
        "source_record_ids",
        "frontier_sha256",
        "evidence_digests",
        "join_method",
        "boundary_method",
        "decision",
        "confidence",
        "conflict_state",
        "registry_task_id",
        "actual_provider",
        "actual_model",
        "provider_receipt_path",
        "provider_receipt_sha256",
        "input_sha256",
        "output_sha256",
    }
    allowed_fields = required_fields | {"substitution_waiver_sha256"}
    for pair_key, pair in sorted(retained.items()):
        if pair.get("inclusion_status") == "included":
            # Deterministic same-corpus title join: never reached the LLM
            # boundary, so there is no decision to validate -- canonicalize
            # directly, still respecting a material field-authority conflict
            # exactly like the governed path below.
            material_conflict = bool(material_field_conflict_blockers)
            canonicalized = not material_conflict
            edge_blockers: set[str] = (
                {f"material_conflict_blocks_join:{pair_key}"} if not canonicalized else set()
            )
            blockers.update(edge_blockers)
            typed_edges.append(
                {
                    "canonical_pair_key": pair_key,
                    "source_record_ids": list(pair["source_record_ids"]),
                    "relation_kind": pair["relation_kind"],
                    "frontier_sha256": frontier_digest,
                    "evidence_digests": list(pair["source_snapshot_digests"]),
                    "join_method": DETERMINISTIC_JOIN_METHOD,
                    "boundary_method": DETERMINISTIC_JOIN_METHOD,
                    "confidence": 1.0,
                    "conflict_state": "material" if material_conflict else "none",
                    "registry_task_id": None,
                    "actual_provider": None,
                    "actual_model": None,
                    "provider_receipt_path": None,
                    "provider_receipt_sha256": None,
                    "input_sha256": None,
                    "output_sha256": None,
                    "substitution_waiver_sha256": None,
                    "canonicalized": canonicalized,
                    "blockers": sorted(edge_blockers),
                }
            )
            if canonicalized:
                canonical_identity_updates.append(
                    {
                        "canonical_pair_key": pair_key,
                        "source_record_ids": list(pair["source_record_ids"]),
                    }
                )
            continue
        decision = decisions_by_key.get(pair_key)
        if decision is None:
            blockers.add(f"missing_decision:{pair_key}")
            continue
        decision_blockers: set[str] = set()
        if set(decision) - allowed_fields:
            decision_blockers.add(f"decision_unknown_fields:{pair_key}")
        for field in sorted(required_fields - set(decision)):
            decision_blockers.add(f"decision_missing_{field}")
            decision_blockers.add(f"decision_missing_{field}:{pair_key}")
        if decision.get("frontier_sha256") != frontier_digest:
            decision_blockers.add("frontier_digest_mismatch")
        if sorted(decision.get("source_record_ids", [])) != sorted(pair["source_record_ids"]):
            decision_blockers.add(f"source_record_ids_mismatch:{pair_key}")
        if sorted(decision.get("evidence_digests", [])) != sorted(pair["source_snapshot_digests"]):
            decision_blockers.add(f"evidence_digests_mismatch:{pair_key}")
        evidence_rows = _normalized_candidate_evidence(pair_key, candidate_evidence)
        if decision.get("input_sha256") != frontier_join_input_digest(frontier, pair, evidence=evidence_rows):
            decision_blockers.add(f"input_digest_mismatch:{pair_key}")
        if decision.get("registry_task_id") != CROSS_SOURCE_JOIN_TASK_ID:
            decision_blockers.add(f"registry_task_mismatch:{pair_key}")
        if decision.get("join_method") != GOVERNED_JOIN_METHOD or decision.get("boundary_method") != GOVERNED_JOIN_METHOD:
            decision_blockers.add(f"boundary_method_mismatch:{pair_key}")
        if not _safe_receipt_path(decision.get("provider_receipt_path")):
            decision_blockers.add(f"provider_receipt_path_invalid:{pair_key}")
        for field in ("provider_receipt_sha256", "input_sha256", "output_sha256"):
            if not _valid_sha256(decision.get(field)):
                decision_blockers.add(f"decision_missing_{field}")
                decision_blockers.add(f"decision_missing_{field}:{pair_key}")
        actual = (decision.get("actual_provider"), decision.get("actual_model"))
        selection_is_registered = actual == default_selection or actual == fallback_selection
        waiver_digest = decision.get("substitution_waiver_sha256")
        waiver_valid = isinstance(waiver_digest, str) and waiver_digest in approved_substitution_waiver_digests
        if not selection_is_registered and not waiver_valid:
            decision_blockers.add("selection_mismatch")
        confidence = decision.get("confidence")
        confidence_valid = isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(float(confidence))
        if not confidence_valid or not 0 <= float(confidence) <= 1:
            decision_blockers.add(f"confidence_invalid:{pair_key}")
        if decision.get("conflict_state") not in {"none", "warning", "material"}:
            decision_blockers.add(f"conflict_state_invalid:{pair_key}")
        material_conflict = bool(material_field_conflict_blockers) or decision.get("conflict_state") == "material"
        canonicalized = (
            not decision_blockers
            and decision.get("decision") == "join"
            and float(confidence) >= threshold
            and not material_conflict
        )
        if not decision_blockers and not canonicalized:
            if decision.get("decision") != "join":
                decision_blockers.add(f"unresolved_decision:{pair_key}")
            elif float(confidence) < threshold:
                decision_blockers.add(f"confidence_below_threshold:{pair_key}")
            else:
                decision_blockers.add(f"material_conflict_blocks_join:{pair_key}")
        blockers.update(decision_blockers)
        typed_edges.append(
            {
                "canonical_pair_key": pair_key,
                "source_record_ids": list(pair["source_record_ids"]),
                "relation_kind": pair["relation_kind"],
                "frontier_sha256": frontier_digest,
                "evidence_digests": list(pair["source_snapshot_digests"]),
                "join_method": GOVERNED_JOIN_METHOD,
                "boundary_method": GOVERNED_JOIN_METHOD,
                "confidence": float(confidence) if confidence_valid else None,
                "conflict_state": "material" if material_conflict else decision.get("conflict_state"),
                "registry_task_id": CROSS_SOURCE_JOIN_TASK_ID,
                "actual_provider": decision.get("actual_provider"),
                "actual_model": decision.get("actual_model"),
                "provider_receipt_path": decision.get("provider_receipt_path"),
                "provider_receipt_sha256": decision.get("provider_receipt_sha256"),
                "input_sha256": decision.get("input_sha256"),
                "output_sha256": decision.get("output_sha256"),
                "substitution_waiver_sha256": waiver_digest if waiver_valid else None,
                "canonicalized": canonicalized,
                "blockers": sorted(decision_blockers),
            }
        )
        if canonicalized:
            canonical_identity_updates.append(
                {
                    "canonical_pair_key": pair_key,
                    "source_record_ids": list(pair["source_record_ids"]),
                }
            )
    return {
        "frontier_sha256": frontier_digest,
        "canonical_fields": canonical_fields,
        "conflict_ledger": conflict_ledger,
        "typed_edges": typed_edges,
        "canonical_identity_updates": canonical_identity_updates,
        "excluded_pairs": sorted(excluded_pairs),
        "blockers": sorted(blockers),
    }


def immutable_id(namespace: str, native_identifier: str) -> str:
    """Create an immutable namespaced identifier without relying on titles or URLs."""

    if not re.fullmatch(r"[a-z0-9][a-z0-9:-]{2,120}", namespace):
        raise ValueError("canonical namespace must be lower-case and namespaced")
    if not native_identifier or not native_identifier.strip():
        raise ValueError("native identifier must be non-empty")
    digest = hashlib.sha256(native_identifier.encode("utf-8")).hexdigest()[:20]
    return f"{namespace}:{digest}"


# Structural shape of an id built by immutable_id(f"{slug}:source:...", ...) /
# immutable_id(f"{slug}:program-item", ...) -- deliberately not anchored to
# any specific conference slug, since the slug is whichever workspace
# produced the record (see _source_record / _canonical_program_items).
_SOURCE_RECORD_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*:source:.+:[0-9a-f]{20}$")
_PROGRAM_ITEM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*:program-item:[0-9a-f]{20}$")


def build_canonical_inventory(
    *,
    run_id: str,
    family_snapshots: Mapping[str, Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    program_items: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    seed_diffs: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
    candidate_frontier: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the digest-pinned canonical control inventory for downstream use."""

    if not run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("run id must be a non-empty path-safe identifier")
    source_family_rows: dict[str, dict[str, Any]] = {}
    snapshot_digests: dict[str, str] = {}
    for family_id, snapshot in sorted(family_snapshots.items()):
        if not FAMILY_ID_RE.fullmatch(family_id):
            raise ValueError("family snapshots require valid source-family ids")
        digest = snapshot.get("sha256")
        if not _valid_sha256(digest):
            raise ValueError("family snapshots require a SHA-256 digest")
        source_family_rows[family_id] = dict(snapshot)
        snapshot_digests[family_id] = str(digest)
    frontier_pairs = candidate_frontier.get("pairs")
    frontier_digest = candidate_frontier.get("frontier_sha256")
    if not isinstance(frontier_pairs, list) or not _valid_sha256(frontier_digest):
        raise ValueError("canonical inventory requires a digest-pinned candidate frontier")
    taxonomy_digest = taxonomy.get("taxonomy_sha256")
    if not _valid_sha256(taxonomy_digest):
        raise ValueError("canonical inventory requires a digest-pinned taxonomy scan")
    inventory = {
        "schema_version": 1,
        "run_id": run_id,
        "stage_id": "source-discovery",
        "generated_at": _utc_now(),
        "status": "prepared",
        "source_families": source_family_rows,
        "family_snapshot_digests": snapshot_digests,
        "source_records": [dict(record) for record in source_records],
        "program_items": [dict(item) for item in program_items],
        "edges": [dict(edge) for edge in edges],
        "seed_diffs": [dict(diff) for diff in seed_diffs],
        "taxonomy": dict(taxonomy),
        "taxonomy_sha256": taxonomy_digest,
        "candidate_frontier": {"pairs": frontier_pairs, "frontier_sha256": frontier_digest},
        "candidate_frontier_sha256": frontier_digest,
        "supersession_history": {
            family_id: list(row.get("superseded_snapshot_ids", []))
            for family_id, row in source_family_rows.items()
        },
    }
    return inventory


def validate_canonical_inventory(inventory: Mapping[str, Any]) -> list[str]:
    """Return deterministic blockers for claims that cannot safely be complete."""

    blockers: list[str] = []
    for diff in inventory.get("seed_diffs", []):
        if isinstance(diff, Mapping) and diff.get("state") == "unverified":
            blockers.append(f"seed_record_unverified:{diff.get('native_id', 'unknown')}")
    families = inventory.get("source_families", {})
    if isinstance(families, Mapping):
        for family_id, family in sorted(families.items()):
            if not isinstance(family, Mapping):
                blockers.append(f"family_invalid:{family_id}")
                continue
            status = family.get("status")
            if status not in {"live_valid", "locked_excluded", "not_applicable"}:
                blockers.append(f"family_unresolved:{family_id}")
            if status == "locked_excluded" and family.get("exclusion_id") not in LOCKED_AUTOMATIC_EXCLUSIONS:
                blockers.append(f"family_exclusion_unapproved:{family_id}")
            if status == "not_applicable" and not (
                family.get("complete_enumeration") and family.get("taxonomy_evidence")
            ):
                blockers.append(f"family_not_applicable_unproven:{family_id}")
    taxonomy = inventory.get("taxonomy", {})
    if isinstance(taxonomy, Mapping):
        blockers.extend(str(blocker) for blocker in taxonomy.get("blockers", []) if str(blocker))
    frontier = inventory.get("candidate_frontier")
    if frontier is not None:
        if not isinstance(frontier, Mapping) or not _valid_sha256(frontier.get("frontier_sha256")):
            blockers.append("candidate_frontier_unpinned")
        elif canonical_json_digest(frontier.get("pairs", [])) != frontier.get("frontier_sha256"):
            blockers.append("candidate_frontier_digest_mismatch")
    for record in inventory.get("source_records", []):
        if not isinstance(record, Mapping) or not _valid_sha256(record.get("snapshot_sha256")):
            blockers.append("source_record_missing_snapshot_evidence")
            continue
        record_id = record.get("source_record_id")
        # Namespaced "<conference-slug>:source:...:<digest>" -- the slug is
        # whichever workspace produced this record, never a fixed literal
        # (see _source_record / extract_source_discovery_inventory).
        if not isinstance(record_id, str) or not _SOURCE_RECORD_ID_RE.match(record_id):
            blockers.append("source_record_missing_immutable_id")
    for item in inventory.get("program_items", []):
        if not isinstance(item, Mapping):
            blockers.append("program_item_invalid")
            continue
        item_id = item.get("program_item_id")
        if not isinstance(item_id, str) or not _PROGRAM_ITEM_ID_RE.match(item_id):
            blockers.append("program_item_missing_immutable_id")
        if not item.get("aliases") or not item.get("source_record_ids") or not item.get("field_evidence"):
            blockers.append(f"program_item_incomplete:{item_id or 'unknown'}")
    return sorted(set(blockers))


def publish_canonical_inventory(path: Path, inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and atomically publish a downstream-pinnable inventory digest."""

    payload = dict(inventory)
    blockers = validate_canonical_inventory(payload)
    payload["blockers"] = blockers
    payload["status"] = "blocked" if blockers else "prepared"
    digest_payload = {key: value for key, value in payload.items() if key != "inventory_sha256"}
    payload["inventory_sha256"] = canonical_json_digest(digest_payload)
    _atomic_write_json(path, payload)
    return payload
