"""Registered-media recovery: official event pages behind a registration wall.

The contracts here turn the frozen r3 official-video policy mapping into a
separate, deterministic r4 overlay plan, validate it, open the registered
virtual site with an operator-supplied session (never a credential in a
receipt) and recover direct YouTube links from official event pages.
:func:`run_registration_media_lane` is the Stage 4 lane owner: it gates the
whole recovery on one authenticated sentinel page, checkpoints privately, and
promotes the overlay only after the strict stage contract passes.

Recommended for
---------------
``RECOMMENDED_FOR = ("CVPR",)`` (formerly
``acquisition/cvpr_registration_media.py``; that import path remains a
compatibility alias).

* **CVPR stage and quirk covered:** Stage 4 ``download-corpus`` r4,
  "registration media": CVPR's official paper/poster videos are linked only
  from registration-gated virtual-site pages.
* **May also fit:** other conferences whose virtual site hides talk/poster
  media behind a registration wall (ICCV, WACV; NeurIPS/ICML-style virtual
  sites with a different page parser).

The fixed r4 run id, scope and session environment name below pin the
completed CVPR 2026 run (registry/runtime data, revisited in the registry
batch); a new conference passes its own through the lane settings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from http.cookiejar import CookieJar, MozillaCookieJar
import copy
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPCookieProcessor, HTTPRedirectHandler, Request, build_opener

from conference_pipeline.acquisition.embedded_media import embedded_media_links, sanitize_media_url
from conference_pipeline.acquisition.youtube_talks import (
    DEFAULT_ALLOWED_HOSTS,
    HarvestedVideo,
    caption_sidecar_matches_video_id,
    download_captions,
    extract_video_id,
    url_host_allowed,
)
from conference_pipeline.conference_identity import VirtualSiteBoundary


R3_RUN_ID = "cvpr2026-live-production-r3-20260716"
RECOMMENDED_FOR = ("CVPR",)

R4_RUN_ID = "cvpr2026-live-production-r4-registration-media"
OVERLAY_LEDGER_RELATIVE_PATH = Path(
    "storage/download-corpus/registration-media/registration_media.json"
)
OVERLAY_REPORT_RELATIVE_PATH = Path(
    "storage/download-corpus/registration-media/registration_media_report.md"
)
OVERLAY_CAPTIONS_RELATIVE_PATH = Path("storage/download-corpus/registration-media/captions")
OVERLAY_ARTIFACTS_RELATIVE_PATH = Path("storage/download-corpus/registration-media/artifacts")

_POLICY_EXCLUSION_STATUS = "policy_excluded_official_video"
_BOUNDARY_DISPOSITION = "outside_authenticated_official_event_boundary"
_ALLOWED_EVENT_ROUTE_FAMILIES = frozenset(
    {
        "poster",
        "invited-talk",
        "tutorial",
        "workshop",
        "session",
        "demonstration",
        "events",
        "meeting",
        "mentorship",
        "reception",
        "remarks",
    }
)
# These four exact static CVPR virtual-platform paths are part of the immutable
# r4 overlay's authenticated request surface.  They are not eligible for the
# r5 successor inventory, whose caller passes only its source-backed event
# routes.  Keeping this historic allowlist exact preserves r4 validation
# without reopening the former broad ``/virtual/2026/`` prefix boundary.
_LEGACY_R4_STATIC_EVENT_PATHS = frozenset(
    {
        "/virtual/2026/search",  # conference-literal-ok: immutable r4 historical static authenticated path; validates only the frozen r4 overlay, excluded from the r5 source-backed inventory
        "/virtual/2026/index.html",  # conference-literal-ok: immutable r4 historical static authenticated path; validates only the frozen r4 overlay, excluded from the r5 source-backed inventory
        "/virtual/2026/sponsor_list",  # conference-literal-ok: immutable r4 historical static authenticated path; validates only the frozen r4 overlay, excluded from the r5 source-backed inventory
        "/virtual/2026/eventlistwithbios/2026KeynotesPanels",  # conference-literal-ok: immutable r4 historical static authenticated path; validates only the frozen r4 overlay, excluded from the r5 source-backed inventory
    }
)
_SECRET_QUERY_MARKERS = ("api_key", "apikey", "authorization", "cookie", "token", "secret", "bearer")
_VOLATILE_DIGEST_FIELDS = frozenset({"generated_at", "updated_at", "candidate_plan_sha256", "overlay_sha256"})
_SESSION_ENV_NAME = "CVPR2026_VIRTUAL_COOKIES_FILE"  # conference-literal-ok: pinned CVPR r4 session env name (RECOMMENDED_FOR lane default)
_MAX_AUTHENTICATED_REDIRECTS = 3
# Official CVPR virtual-event pages can legitimately carry an approximately
# 3.25 MiB rendered schedule payload.  Keep a bounded in-memory ceiling above
# that observed response while still rejecting unbounded or media payloads.
_MAX_OFFICIAL_EVENT_PAGE_BYTES = 4 * 1024 * 1024
_NOCOOKIE_EMBED_RE = re.compile(
    r"(?:(?:https?:)?//)(?:www\.)?youtube-nocookie\.com/embed/([A-Za-z0-9_-]+)", re.IGNORECASE
)
_AUTHENTICATED_ACCOUNT_MARKERS = (
    b'data-authenticated-account="true"',
    b"data-authenticated-account='true'",
    b'data-account-state="authenticated"',
    b"data-account-state='authenticated'",
    b'data-authenticated="true"',
    b"data-authenticated='true'",
    b"/logout",
    b"sign out",
)
_REGISTRATION_WALL_MARKERS = (
    b'data-registration-wall="true"',
    b"data-registration-wall='true'",
    b"registration required",
    b"register to watch",
    b"sign in to access",
    b"please log in",
    b"session expired",
)


@dataclass(frozen=True)
class RegistrationMediaScope:
    """Complete frozen-scope accounting required for an r4 overlay."""

    total_program_item_count: int
    candidate_program_item_count: int
    r3_predecessor_program_item_count: int
    policy_request_count: int
    transport_eligible_request_count: int
    transport_eligible_program_item_count: int
    boundary_request_count: int
    boundary_program_item_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total_program_item_count": self.total_program_item_count,
            "candidate_program_item_count": self.candidate_program_item_count,
            "r3_predecessor_program_item_count": self.r3_predecessor_program_item_count,
            "policy_request_count": self.policy_request_count,
            "transport_eligible_request_count": self.transport_eligible_request_count,
            "transport_eligible_program_item_count": self.transport_eligible_program_item_count,
            "boundary_request_count": self.boundary_request_count,
            "boundary_program_item_count": self.boundary_program_item_count,
        }

    def validate(self) -> None:
        values = self.as_dict()
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError("registration-media scope counts must be nonnegative integers")
        if self.total_program_item_count != (
            self.candidate_program_item_count + self.r3_predecessor_program_item_count
        ):
            raise ValueError("registration-media scope does not cover the full corpus")
        if self.candidate_program_item_count != (
            self.transport_eligible_program_item_count + self.boundary_program_item_count
        ):
            raise ValueError("candidate program-item scope does not match its boundary partition")
        if self.policy_request_count != (
            self.transport_eligible_request_count + self.boundary_request_count
        ):
            raise ValueError("policy request scope does not match its boundary partition")


FULL_R4_SCOPE = RegistrationMediaScope(
    total_program_item_count=10_515,
    candidate_program_item_count=5_159,
    r3_predecessor_program_item_count=5_356,
    policy_request_count=2_579,
    transport_eligible_request_count=2_573,
    transport_eligible_program_item_count=5_146,
    boundary_request_count=6,
    boundary_program_item_count=13,
)

# These are immutable r3 evidence values recorded in 04-07-SOURCE-COVERAGE.md.
# They are intentionally CVPR-specific and are not a reusable platform policy.
PINNED_R3_PREDECESSORS: list[dict[str, str]] = [
    {
        "id": "source_discovery_receipt",
        "path": "storage/workflow-runs/cvpr2026-live-production-r3-20260716/stage-02-source-discovery.json",
        "file_sha256": "680e41f5c79128789c79bb0d2bd319570fb7831dbd68b87861cb33c3ec798648",
        "publisher_digest": "49afa2d25f0067a6e61784b9ab3fdd921e5c8ef0d4203a360366ffd78aa5eeb7",
        "publisher_field": "canonical_inventory.inventory_sha256",
    },
    {
        "id": "corpus_freeze_receipt",
        "path": "storage/workflow-runs/cvpr2026-live-production-r3-20260716/stage-03-corpus-freeze.json",
        "file_sha256": "ee198c735e63b8b9a9b12fc50e14d2bc82bd66ad25eb878c1a9254aee0e0328c",
        "publisher_digest": "4a4b8a0b5520aec4e3ad85e40c1099fda77a8f8645c379f7050b70a02bb81d32",
        "publisher_field": "corpus_manifest.selection_sha256",
    },
    {
        "id": "download_corpus_receipt",
        "path": "storage/workflow-runs/cvpr2026-live-production-r3-20260716/stage-04-download-corpus.json",
        "file_sha256": "daea94d6bfaa0cc44e4c789bc7139778713cedf3a50dfaa30c66c0d43ce42dc2",
    },
    {
        "id": "download_corpus_ledger",
        "path": "storage/download-corpus/download_corpus.json",
        "file_sha256": "7b2faa00ad43aaf02655a5e761360088763ce082f1d1d4b663a564d7d652e3cf",
        "publisher_digest": "b68b93ca708496fcba55e23e30c536b2b2e378d42ebd7ffb1990c5a33d06d049",
        "publisher_field": "manifest_sha256",
    },
    {
        "id": "program_spine",
        "path": "storage/download-corpus/program_spine.json",
        "file_sha256": "e8ce044d9d3fc1bc0870912f0d552238a56aaec75d0867c508bacf9ea9edcef2",
        "publisher_digest": "ccfc884392fabefd3ff9fd977f77992d975e438b84537d129764c182bdb76de4",
        "publisher_field": "program_spine_sha256",
    },
]


def _canonical_payload(value: Any, *, exclude_volatile: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_payload(item, exclude_volatile=exclude_volatile)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not (exclude_volatile and str(key) in _VOLATILE_DIGEST_FIELDS)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_payload(item, exclude_volatile=exclude_volatile) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_registration_media_digest(payload: Mapping[str, Any]) -> str:
    """Return a stable digest excluding timestamps and self-referential digests."""

    encoded = json.dumps(
        _canonical_payload(payload, exclude_volatile=True),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _normalize_predecessors(predecessors: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    received = [dict(item) for item in predecessors if isinstance(item, Mapping)]
    if len(received) != len(predecessors) or len(received) != len(PINNED_R3_PREDECESSORS):
        raise ValueError("pinned predecessor set is incomplete")
    expected_by_id = {item["id"]: item for item in PINNED_R3_PREDECESSORS}
    received_by_id = {str(item.get("id") or ""): item for item in received}
    if len(received_by_id) != len(received) or set(received_by_id) != set(expected_by_id):
        raise ValueError("pinned predecessor identifiers do not match")
    normalized: list[dict[str, str]] = []
    for expected in PINNED_R3_PREDECESSORS:
        actual = received_by_id[expected["id"]]
        if actual != expected:
            raise ValueError(f"pinned predecessor mismatch: {expected['id']}")
        normalized.append(copy.deepcopy(expected))
    return normalized


def load_pinned_r3_predecessors(workspace: Path) -> list[dict[str, str]]:
    """Recompute and validate the five immutable r3 predecessor records.

    ``workspace`` is the CVPR workspace (for example ``conferences/cvpr2026``).
    This intentionally reads only frozen receipt/control files; it never loads
    sessions, cookies, browser state, or a live transport input.
    """

    root = workspace.resolve()
    observed: list[dict[str, str]] = []
    for expected in PINNED_R3_PREDECESSORS:
        candidate = (root / expected["path"]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("pinned predecessor path escapes the workspace") from exc
        if not candidate.is_file() or _sha256_file(candidate) != expected["file_sha256"]:
            raise ValueError(f"pinned predecessor file mismatch: {expected['id']}")
        publisher_field = expected.get("publisher_field")
        if publisher_field:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"pinned predecessor is unreadable: {expected['id']}") from exc
            if not isinstance(payload, Mapping) or _get_path(payload, publisher_field) != expected.get(
                "publisher_digest"
            ):
                raise ValueError(f"pinned predecessor publisher digest mismatch: {expected['id']}")
        observed.append(copy.deepcopy(expected))
    return _normalize_predecessors(observed)


def _sanitize_source_url(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("policy request lacks a source URL")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValueError("policy request source URL is invalid") from exc
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("policy request source URL is not an HTTPS origin URL")
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not any(marker in key.lower() for marker in _SECRET_QUERY_MARKERS)
    ]
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path, urlencode(query), ""))


def _boundary_policy(boundary: VirtualSiteBoundary) -> dict[str, Any]:
    """Return the replayable configured route policy without a venue fallback."""

    if not isinstance(boundary, VirtualSiteBoundary):
        raise ValueError("registered-media URL policy requires a virtual-site boundary")
    return {
        "origin": boundary.origin,
        "root_segments": list(boundary.root_segments),
        "outside_boundary_disposition": _BOUNDARY_DISPOSITION,
    }


def _path_segments_under_boundary(parsed: Any, *, boundary: VirtualSiteBoundary) -> tuple[str, ...] | None:
    """Return a safe route suffix only when ``parsed`` remains under the configured root."""

    if not isinstance(boundary, VirtualSiteBoundary):
        return None
    path_segments = parsed.path.split("/")
    if not path_segments or path_segments[0] != "" or path_segments[-1] == "":
        return None
    segments = tuple(path_segments[1:])
    if len(segments) <= len(boundary.root_segments) or segments[: len(boundary.root_segments)] != boundary.root_segments:
        return None
    decoded = [unquote(segment) for segment in segments]
    if any(
        not segment
        or segment != decoded_segment
        or decoded_segment in {".", ".."}
        or "/" in decoded_segment
        or "\\" in decoded_segment
        for segment, decoded_segment in zip(segments, decoded, strict=True)
    ):
        return None
    return segments[len(boundary.root_segments) :]


def is_allowed_cvpr_virtual_event_url(value: str, *, boundary: VirtualSiteBoundary) -> bool:
    """Return whether a URL matches the exact CVPR virtual-event grammar.

    This predicate is intentionally stricter than the historic path prefix:
    a cookie-bearing request may target only an approved four-segment event
    route, never a dot-segment escape, query-bearing URL, or account surface.
    Callers additionally require membership in their frozen URL set.
    """

    if not isinstance(boundary, VirtualSiteBoundary):
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or (parsed.hostname or "").rstrip(".").casefold() != boundary.host
        or parsed.query
        or parsed.fragment
    ):
        return False
    suffix = _path_segments_under_boundary(parsed, boundary=boundary)
    if suffix is None:
        return False
    if parsed.path in _LEGACY_R4_STATIC_EVENT_PATHS:
        return True
    if len(suffix) != 2:
        return False
    route_family, event_id = suffix
    if route_family not in _ALLOWED_EVENT_ROUTE_FAMILIES or not event_id:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", event_id))


def _policy_item_ids(program_item_results: Sequence[Mapping[str, Any]]) -> set[str]:
    item_ids: set[str] = set()
    for row in program_item_results:
        item_id = row.get("program_item_id")
        blocker = row.get("blocker")
        if (
            isinstance(item_id, str)
            and isinstance(blocker, Mapping)
            and blocker.get("code") == _POLICY_EXCLUSION_STATUS
        ):
            item_ids.add(item_id)
    return item_ids


def _validate_r3_program_items(
    program_item_results: Sequence[Mapping[str, Any]], *, expected_scope: RegistrationMediaScope
) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in program_item_results:
        item_id = row.get("program_item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("r3 program-item result lacks a program_item_id")
        if item_id in by_id:
            raise ValueError("duplicate program item in full corpus accounting")
        by_id[item_id] = row
    if len(by_id) != expected_scope.total_program_item_count:
        raise ValueError("full corpus program-item accounting is incomplete")
    return by_id


def _candidate_plan_digest_input(plan: Mapping[str, Any]) -> dict[str, Any]:
    requests = plan.get("request_rows")
    if not isinstance(requests, list):
        raise ValueError("candidate plan has no request rows")
    return {
        "r3_run_id": plan.get("r3_run_id"),
        "predecessors": sorted(plan.get("predecessors", []), key=lambda item: str(item.get("id", ""))),
        "host_policy": plan.get("host_policy"),
        "counts": plan.get("counts"),
        "request_rows": [
            {
                "request_key": row.get("request_key"),
                "linked_program_item_ids": sorted(row.get("linked_program_item_ids", [])),
                "transport_eligible": row.get("transport_eligible"),
                "final_disposition": row.get("final_disposition"),
            }
            for row in sorted(requests, key=lambda item: str(item.get("request_key", "")))
        ],
    }


def build_candidate_plan(
    *,
    r3_run_id: str,
    predecessors: Sequence[Mapping[str, Any]],
    r3_request_results: Sequence[Mapping[str, Any]],
    r3_program_item_results: Sequence[Mapping[str, Any]],
    boundary: VirtualSiteBoundary,
    expected_scope: RegistrationMediaScope = FULL_R4_SCOPE,
) -> dict[str, Any]:
    """Plan the exact r4 official-media candidate set from immutable r3 rows."""

    expected_scope.validate()
    if r3_run_id != R3_RUN_ID:
        raise ValueError("candidate plan requires the pinned r3 run")
    normalized_predecessors = _normalize_predecessors(predecessors)
    by_program_item_id = _validate_r3_program_items(
        r3_program_item_results, expected_scope=expected_scope
    )

    requests: list[dict[str, Any]] = []
    request_keys: set[str] = set()
    candidate_item_ids: set[str] = set()
    for raw_request in r3_request_results:
        if raw_request.get("status") != _POLICY_EXCLUSION_STATUS:
            continue
        request_key = raw_request.get("request_key")
        if not isinstance(request_key, str) or not request_key:
            raise ValueError("policy request lacks a request key")
        if request_key in request_keys:
            raise ValueError("duplicate policy request key")
        request_keys.add(request_key)
        linked = raw_request.get("linked_program_item_ids")
        if not isinstance(linked, list) or not linked or any(not isinstance(item, str) or not item for item in linked):
            raise ValueError("policy request lacks linked program-item ids")
        linked_ids = sorted(linked)
        if len(set(linked_ids)) != len(linked_ids):
            raise ValueError("policy request repeats a linked program-item id")
        if candidate_item_ids.intersection(linked_ids):
            raise ValueError("policy request mapping has duplicate program-item ids")
        candidate_item_ids.update(linked_ids)
        source_url = _sanitize_source_url(raw_request.get("source_url"))
        eligible = is_allowed_cvpr_virtual_event_url(source_url, boundary=boundary)
        request: dict[str, Any] = {
            "request_key": request_key,
            "source_url": source_url,
            "linked_program_item_ids": linked_ids,
            "transport_eligible": eligible,
            "source_evidence": copy.deepcopy(raw_request.get("source_evidence"))
            if isinstance(raw_request.get("source_evidence"), Mapping)
            else None,
        }
        if not eligible:
            request["final_disposition"] = _BOUNDARY_DISPOSITION
        requests.append(request)

    policy_item_ids = _policy_item_ids(r3_program_item_results)
    if candidate_item_ids != policy_item_ids:
        raise ValueError("candidate program-item mapping does not match r3 policy exclusions")
    for request in requests:
        for item_id in request["linked_program_item_ids"]:
            r3_item = by_program_item_id.get(item_id)
            if r3_item is None or r3_item.get("request_key") != request["request_key"]:
                raise ValueError("candidate program-item mapping does not match r3 request evidence")

    transport_requests = [row for row in requests if row["transport_eligible"]]
    boundary_requests = [row for row in requests if not row["transport_eligible"]]
    observed = RegistrationMediaScope(
        total_program_item_count=len(by_program_item_id),
        candidate_program_item_count=len(candidate_item_ids),
        r3_predecessor_program_item_count=len(by_program_item_id) - len(candidate_item_ids),
        policy_request_count=len(requests),
        transport_eligible_request_count=len(transport_requests),
        transport_eligible_program_item_count=sum(
            len(row["linked_program_item_ids"]) for row in transport_requests
        ),
        boundary_request_count=len(boundary_requests),
        boundary_program_item_count=sum(len(row["linked_program_item_ids"]) for row in boundary_requests),
    )
    if observed.as_dict() != expected_scope.as_dict():
        raise ValueError("transport-eligible or full-scope registration-media partition differs from the pinned scope")

    plan: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_registration_media_candidate_plan",
        "r3_run_id": R3_RUN_ID,
        "predecessors": normalized_predecessors,
        "host_policy": _boundary_policy(boundary),
        "request_rows": sorted(requests, key=lambda row: str(row["request_key"])),
        "counts": observed.as_dict(),
    }
    plan["candidate_plan_sha256"] = canonical_registration_media_digest(_candidate_plan_digest_input(plan))
    return plan


def _validate_candidate_plan(
    plan: Mapping[str, Any], *, boundary: VirtualSiteBoundary, expected_scope: RegistrationMediaScope
) -> None:
    if plan.get("r3_run_id") != R3_RUN_ID:
        raise ValueError("candidate plan has a non-pinned r3 run")
    _normalize_predecessors(plan.get("predecessors", []))
    if plan.get("counts") != expected_scope.as_dict():
        raise ValueError("candidate plan scope does not match the full frozen corpus")
    if plan.get("host_policy") != _boundary_policy(boundary):
        raise ValueError("candidate plan virtual-site policy does not match the configured boundary")
    expected_digest = canonical_registration_media_digest(_candidate_plan_digest_input(plan))
    if plan.get("candidate_plan_sha256") != expected_digest:
        raise ValueError("candidate plan digest is invalid")
    request_rows = plan.get("request_rows")
    if not isinstance(request_rows, list):
        raise ValueError("candidate plan request rows are invalid")
    seen_requests: set[str] = set()
    seen_items: set[str] = set()
    for row in request_rows:
        if not isinstance(row, Mapping):
            raise ValueError("candidate plan contains a non-object request")
        request_key = row.get("request_key")
        linked_ids = row.get("linked_program_item_ids")
        if not isinstance(request_key, str) or not isinstance(linked_ids, list):
            raise ValueError("candidate plan request row is invalid")
        if request_key in seen_requests:
            raise ValueError("candidate plan contains a duplicate request")
        seen_requests.add(request_key)
        if len(set(linked_ids)) != len(linked_ids) or seen_items.intersection(linked_ids):
            raise ValueError("candidate plan contains duplicate program-item ids")
        seen_items.update(linked_ids)
        eligible = row.get("transport_eligible") is True
        if eligible != is_allowed_cvpr_virtual_event_url(str(row.get("source_url") or ""), boundary=boundary):
            raise ValueError("candidate plan transport eligibility does not match the host boundary")
        if not eligible and row.get("final_disposition") != _BOUNDARY_DISPOSITION:
            raise ValueError("outside-boundary request lacks final no-network disposition")
    if len(seen_requests) != expected_scope.policy_request_count or len(seen_items) != expected_scope.candidate_program_item_count:
        raise ValueError("candidate plan does not retain the complete policy mapping")


def transport_eligible_requests(
    candidate_plan: Mapping[str, Any], *, boundary: VirtualSiteBoundary
) -> list[dict[str, Any]]:
    """Return the only rows a future authenticated opener may receive."""

    _validate_candidate_plan(candidate_plan, boundary=boundary, expected_scope=FULL_R4_SCOPE)
    request_rows = candidate_plan["request_rows"]
    return [copy.deepcopy(dict(row)) for row in request_rows if row.get("transport_eligible") is True]


def build_registration_media_overlay(
    *,
    candidate_plan: Mapping[str, Any],
    r3_program_item_results: Sequence[Mapping[str, Any]],
    boundary: VirtualSiteBoundary,
    expected_scope: RegistrationMediaScope = FULL_R4_SCOPE,
) -> dict[str, Any]:
    """Create a private r4 overlay without changing or copying the r3 ledger."""

    expected_scope.validate()
    _validate_candidate_plan(candidate_plan, boundary=boundary, expected_scope=expected_scope)
    by_program_item_id = _validate_r3_program_items(
        r3_program_item_results, expected_scope=expected_scope
    )
    request_rows = candidate_plan["request_rows"]
    candidate_by_item_id: dict[str, Mapping[str, Any]] = {}
    for request in request_rows:
        for item_id in request["linked_program_item_ids"]:
            candidate_by_item_id[item_id] = request

    accounting: list[dict[str, Any]] = []
    for item_id in sorted(by_program_item_id):
        r3_row = by_program_item_id[item_id]
        candidate_request = candidate_by_item_id.get(item_id)
        if candidate_request is None:
            accounting.append(
                {
                    "program_item_id": item_id,
                    "result_origin": "r3_predecessor",
                    "r3_request_key": r3_row.get("request_key"),
                    "r3_result_status": (
                        r3_row.get("blocker", {}).get("code")
                        if isinstance(r3_row.get("blocker"), Mapping)
                        else "r3_artifact_or_result"
                    ),
                }
            )
            continue
        row: dict[str, Any] = {
            "program_item_id": item_id,
            "result_origin": "r4_registration_media",
            "r3_request_key": candidate_request["request_key"],
            "transport_eligible": candidate_request["transport_eligible"],
            "result_status": "pending" if candidate_request["transport_eligible"] else _BOUNDARY_DISPOSITION,
        }
        if not candidate_request["transport_eligible"]:
            row["blocker"] = {"code": _BOUNDARY_DISPOSITION}
        accounting.append(row)

    overlay: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_registration_media_overlay",
        "stage_id": "download-corpus",
        "run_id": R4_RUN_ID,
        "status": "pre_request",
        "predecessor_lineage": {
            "r3_run_id": R3_RUN_ID,
            "predecessors": copy.deepcopy(candidate_plan["predecessors"]),
            "candidate_plan_sha256": candidate_plan["candidate_plan_sha256"],
        },
        "candidate_plan": {
            "counts": copy.deepcopy(candidate_plan["counts"]),
            "host_policy": copy.deepcopy(candidate_plan["host_policy"]),
            "candidate_plan_sha256": candidate_plan["candidate_plan_sha256"],
        },
        "official_event_requests": copy.deepcopy(request_rows),
        "official_event_media_records": [],
        "program_item_accounting": accounting,
        "counts": expected_scope.as_dict(),
        "secrets_copied": False,
    }
    overlay["overlay_sha256"] = canonical_registration_media_digest(overlay)
    validate_registration_media_overlay(overlay, boundary=boundary, expected_scope=expected_scope)
    return overlay


def validate_registration_media_overlay(
    overlay: Mapping[str, Any],
    *,
    boundary: VirtualSiteBoundary,
    expected_scope: RegistrationMediaScope = FULL_R4_SCOPE,
) -> None:
    """Fail closed on scope loss, r3 substitution, or boundary-policy drift."""

    expected_scope.validate()
    if overlay.get("run_id") != R4_RUN_ID or overlay.get("stage_id") != "download-corpus":
        raise ValueError("overlay has an invalid r4 identity")
    if overlay.get("counts") != expected_scope.as_dict():
        raise ValueError("overlay does not retain full corpus accounting")
    lineage = overlay.get("predecessor_lineage")
    candidate = overlay.get("candidate_plan")
    if not isinstance(lineage, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("overlay lacks immutable predecessor lineage")
    _normalize_predecessors(lineage.get("predecessors", []))
    if lineage.get("r3_run_id") != R3_RUN_ID:
        raise ValueError("overlay does not bind the pinned r3 run")
    if lineage.get("candidate_plan_sha256") != candidate.get("candidate_plan_sha256"):
        raise ValueError("overlay candidate-plan lineage is inconsistent")
    if candidate.get("host_policy") != _boundary_policy(boundary):
        raise ValueError("overlay virtual-site policy does not match the configured boundary")
    if not isinstance(overlay.get("official_event_media_records"), list):
        raise ValueError("overlay media-record surface is invalid")
    request_rows = overlay.get("official_event_requests")
    accounting = overlay.get("program_item_accounting")
    if not isinstance(request_rows, list) or not isinstance(accounting, list):
        raise ValueError("overlay accounting surfaces are invalid")
    seen_request_keys: set[str] = set()
    candidate_item_ids: set[str] = set()
    transport_item_count = 0
    boundary_item_count = 0
    boundary_request_count = 0
    for request in request_rows:
        if not isinstance(request, Mapping):
            raise ValueError("overlay contains a non-object request row")
        request_key = request.get("request_key")
        linked_ids = request.get("linked_program_item_ids")
        if not isinstance(request_key, str) or not isinstance(linked_ids, list) or not linked_ids:
            raise ValueError("overlay request row is invalid")
        if request_key in seen_request_keys:
            raise ValueError("overlay contains a duplicate request key")
        seen_request_keys.add(request_key)
        if len(set(linked_ids)) != len(linked_ids) or candidate_item_ids.intersection(linked_ids):
            raise ValueError("overlay request mapping duplicates program-item ids")
        candidate_item_ids.update(linked_ids)
        eligible = request.get("transport_eligible") is True
        if eligible != is_allowed_cvpr_virtual_event_url(
            str(request.get("source_url") or ""), boundary=boundary
        ):
            raise ValueError("overlay request violates the authenticated host boundary")
        if eligible:
            transport_item_count += len(linked_ids)
        else:
            boundary_request_count += 1
            boundary_item_count += len(linked_ids)
            if request.get("final_disposition") != _BOUNDARY_DISPOSITION or "opener_input" in request:
                raise ValueError("outside-boundary request is not a final no-network outcome")
    if (
        len(seen_request_keys) != expected_scope.policy_request_count
        or len(candidate_item_ids) != expected_scope.candidate_program_item_count
        or transport_item_count != expected_scope.transport_eligible_program_item_count
        or boundary_request_count != expected_scope.boundary_request_count
        or boundary_item_count != expected_scope.boundary_program_item_count
    ):
        raise ValueError("overlay request partition differs from the pinned r4 scope")

    seen_program_items: set[str] = set()
    origins = {"r4_registration_media": 0, "r3_predecessor": 0}
    for row in accounting:
        if not isinstance(row, Mapping):
            raise ValueError("overlay contains a non-object accounting row")
        item_id = row.get("program_item_id")
        origin = row.get("result_origin")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("overlay accounting row lacks a program item")
        if item_id in seen_program_items:
            raise ValueError("duplicate program item in overlay accounting")
        seen_program_items.add(item_id)
        if origin not in origins:
            raise ValueError("overlay accounting row has an unknown result origin")
        origins[origin] += 1
        if item_id in candidate_item_ids:
            if origin != "r4_registration_media":
                raise ValueError("candidate program item was not retained in r4 accounting")
        elif origin != "r3_predecessor":
            raise ValueError("noncandidate program item is not an immutable r3 predecessor reference")
    if len(seen_program_items) != expected_scope.total_program_item_count:
        raise ValueError("overlay full corpus program-item accounting is incomplete")
    if (
        origins["r4_registration_media"] != expected_scope.candidate_program_item_count
        or origins["r3_predecessor"] != expected_scope.r3_predecessor_program_item_count
    ):
        raise ValueError("overlay result-origin accounting is incomplete")
    if overlay.get("secrets_copied") is not False:
        raise ValueError("overlay must explicitly prove secrets_copied is false")
    if overlay.get("overlay_sha256") != canonical_registration_media_digest(overlay):
        raise ValueError("overlay digest is invalid")


def _safe_auth_preflight(
    *,
    passed: bool,
    candidate_plan: Mapping[str, Any],
    capacity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the sole serializable session-preflight record.

    This record intentionally has no configured environment key, filesystem
    path, metadata, cookie identity, header, token, or profile field.
    """

    candidate_digest = candidate_plan.get("candidate_plan_sha256")
    capacity_passed = _capacity_evidence_is_valid(
        capacity=capacity,
        candidate_plan_digest=candidate_digest if isinstance(candidate_digest, str) else "",
    )
    return {
        "auth_mode": "external_cookie_session",
        "auth_status": "preflight_passed" if passed else "registration_auth_required",
        "preflight_status": "pass" if passed else "blocked",
        "lineage": {
            "r3_run_id": R3_RUN_ID,
            "candidate_plan_sha256": candidate_digest if isinstance(candidate_digest, str) else None,
            "status": "pinned" if passed else "invalid_or_unverified",
        },
        "capacity": {
            "status": "pass" if capacity_passed else "invalid_or_insufficient",
            "positive_available_bytes": capacity.get("positive_available_bytes") is True,
            "no_cap_hits": capacity.get("no_cap_hits") is True,
            "manifest_bound": isinstance(capacity.get("manifest_sha256"), str)
            and len(str(capacity.get("manifest_sha256"))) == 64,
            "candidate_plan_bound": capacity.get("candidate_plan_sha256") == candidate_digest,
        },
        "transport_eligible_request_count": FULL_R4_SCOPE.transport_eligible_request_count,
        "secrets_copied": False,
    }


def _capacity_evidence_is_valid(*, capacity: Mapping[str, Any], candidate_plan_digest: str) -> bool:
    return (
        capacity.get("status") in {"pass", "pass_pre_request"}
        and capacity.get("positive_available_bytes") is True
        and capacity.get("no_cap_hits") is True
        and isinstance(capacity.get("manifest_sha256"), str)
        and len(str(capacity.get("manifest_sha256"))) == 64
        and capacity.get("candidate_plan_sha256") == candidate_plan_digest
    )


def _path_is_outside(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return True
    return False


def _external_session_metadata_is_valid(
    *,
    workspace: Path,
    repository_root: Path,
    environ: Mapping[str, str],
    lstat: Callable[[Path], Any],
    is_readable: Callable[[Path], bool],
    current_uid: Callable[[], int],
) -> tuple[bool, Path | None]:
    """Validate only external owner-only cookie-file metadata; never read bytes."""

    configured = environ.get(_SESSION_ENV_NAME)
    if not isinstance(configured, str) or not configured:
        return False, None
    candidate = Path(configured).expanduser()
    try:
        metadata = lstat(candidate)
    except (OSError, ValueError, TypeError):
        return False, None
    mode = getattr(metadata, "st_mode", None)
    owner = getattr(metadata, "st_uid", None)
    if not isinstance(mode, int) or not isinstance(owner, int):
        return False, None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        return False, None
    if owner != current_uid() or not (mode & stat.S_IRUSR):
        return False, None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        return False, None
    try:
        resolved = candidate.resolve(strict=False)
        workspace_root = workspace.resolve()
        repository = repository_root.resolve()
    except (OSError, RuntimeError):
        return False, None
    if not _path_is_outside(resolved, workspace_root) or not _path_is_outside(resolved, repository):
        return False, None
    try:
        readable = is_readable(resolved)
    except OSError:
        return False, None
    if readable is not True:
        return False, None
    return True, resolved


def preflight_registered_virtual_session(
    *,
    workspace: Path,
    repository_root: Path,
    candidate_plan: Mapping[str, Any],
    boundary: VirtualSiteBoundary,
    verified_predecessors: Sequence[Mapping[str, Any]],
    capacity: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
    lstat: Callable[[Path], Any] = os.lstat,
    is_readable: Callable[[Path], bool] | None = None,
    current_uid: Callable[[], int] = os.geteuid,
) -> dict[str, Any]:
    """Check only nonsecret session metadata before any jar or opener exists."""

    environment = os.environ if environ is None else environ
    readable_check = (lambda path: os.access(path, os.R_OK)) if is_readable is None else is_readable
    lineage_valid = True
    try:
        _validate_candidate_plan(candidate_plan, boundary=boundary, expected_scope=FULL_R4_SCOPE)
        if _normalize_predecessors(verified_predecessors) != _normalize_predecessors(
            candidate_plan.get("predecessors", [])
        ):
            lineage_valid = False
    except (TypeError, ValueError):
        lineage_valid = False
    candidate_digest = candidate_plan.get("candidate_plan_sha256")
    capacity_valid = _capacity_evidence_is_valid(
        capacity=capacity,
        candidate_plan_digest=candidate_digest if isinstance(candidate_digest, str) else "",
    )
    session_valid, _ = _external_session_metadata_is_valid(
        workspace=workspace,
        repository_root=repository_root,
        environ=environment,
        lstat=lstat,
        is_readable=readable_check,
        current_uid=current_uid,
    )
    return _safe_auth_preflight(
        passed=lineage_valid and capacity_valid and session_valid,
        candidate_plan=candidate_plan,
        capacity=capacity,
    )


class _NoRedirectHandler(HTTPRedirectHandler):
    """Force explicit validation before an authenticated redirect can be opened."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _default_cookie_jar_loader(path: Path) -> CookieJar:
    jar = MozillaCookieJar(str(path))
    jar.load(ignore_discard=True, ignore_expires=True)
    return jar


def _default_opener_builder(jar: CookieJar) -> Any:
    return build_opener(HTTPCookieProcessor(jar), _NoRedirectHandler())


def _open_url(opener: Any, url: str) -> Any:
    if callable(opener):
        return opener(url)
    return opener.open(Request(url, method="GET"))


def _response_status(response: Any) -> int | None:
    raw_status = getattr(response, "status", None)
    if raw_status is None and hasattr(response, "getcode"):
        raw_status = response.getcode()
    return raw_status if isinstance(raw_status, int) else None


def _response_url(response: Any, fallback: str) -> str:
    raw_url = getattr(response, "url", None)
    if raw_url is None and hasattr(response, "geturl"):
        raw_url = response.geturl()
    return _sanitize_source_url(raw_url if isinstance(raw_url, str) else fallback)


def _response_location(response: Any) -> str | None:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    value = getter("Location") if callable(getter) else None
    return value if isinstance(value, str) and value else None


def _bounded_response_html(response: Any) -> tuple[bytes | None, str | None]:
    """Read at most one trusted HTML page, retaining it only in memory."""

    headers = getattr(response, "headers", None)
    content_type_getter = getattr(headers, "get", None)
    content_type = content_type_getter("Content-Type") if callable(content_type_getter) else None
    if not isinstance(content_type, str) or content_type.lower().split(";", 1)[0].strip() not in {
        "text/html",
        "application/xhtml+xml",
    }:
        return None, None
    reader = getattr(response, "read", None)
    try:
        body = reader(_MAX_OFFICIAL_EVENT_PAGE_BYTES + 1) if callable(reader) else None
    except Exception:
        return None, None
    if not isinstance(body, bytes) or not body or len(body) > _MAX_OFFICIAL_EVENT_PAGE_BYTES:
        return None, None
    return body, content_type


def _html_access_signals(body: bytes | None) -> tuple[bool, bool]:
    """Derive conservative access state from standard response HTML only."""

    if not isinstance(body, bytes):
        return False, False
    lowered = body.lower()
    authenticated = any(marker in lowered for marker in _AUTHENTICATED_ACCOUNT_MARKERS)
    registration_wall = any(marker in lowered for marker in _REGISTRATION_WALL_MARKERS)
    return authenticated, registration_wall


def _safe_auth_outcome(
    *,
    status: str,
    auth_status: str,
    http_status: int | None = None,
    final_url: str | None = None,
    redirect_count: int = 0,
) -> dict[str, Any]:
    return {
        "auth_mode": "external_cookie_session",
        "auth_status": auth_status,
        "status": status,
        "http_status": http_status,
        "final_url": final_url,
        "redirect_count": redirect_count,
        "secrets_copied": False,
    }


def _is_login_redirect(value: str) -> bool:
    try:
        path = urlsplit(value).path.lower()
    except ValueError:
        return False
    return "login" in path or "signin" in path


def _outcome_from_response(
    *,
    response: Any,
    current_url: str,
    redirect_count: int,
    allowed_final_urls: frozenset[str],
    boundary: VirtualSiteBoundary,
    authenticated_account: bool = False,
    registration_wall: bool = False,
) -> dict[str, Any]:
    status = _response_status(response)
    if registration_wall or status in {401, 407}:
        return _safe_auth_outcome(
            status="registration_auth_required",
            auth_status="registration_auth_required",
            http_status=status,
            redirect_count=redirect_count,
        )
    if status == 403:
        if authenticated_account and not registration_wall:
            return _safe_auth_outcome(
                status="registered_access_denied",
                auth_status="registered_access_denied",
                http_status=status,
                redirect_count=redirect_count,
            )
        return _safe_auth_outcome(
            status="registration_auth_required",
            auth_status="registration_auth_required",
            http_status=status,
            redirect_count=redirect_count,
        )
    if status == 404 and authenticated_account and not registration_wall:
        return _safe_auth_outcome(
            status="official_video_not_published",
            auth_status="authenticated",
            http_status=status,
            redirect_count=redirect_count,
        )
    if status is not None and 200 <= status < 300:
        try:
            final_url = _response_url(response, current_url)
        except ValueError:
            final_url = ""
        if final_url not in allowed_final_urls or not is_allowed_cvpr_virtual_event_url(final_url, boundary=boundary):
            return _safe_auth_outcome(
                status="unreachable_authenticated_media",
                auth_status="authenticated" if authenticated_account else "registration_auth_required",
                http_status=status,
                redirect_count=redirect_count,
            )
        if registration_wall or not authenticated_account:
            return _safe_auth_outcome(
                status="registration_auth_required",
                auth_status="registration_auth_required",
                http_status=status,
                redirect_count=redirect_count,
            )
        return _safe_auth_outcome(
            status="authenticated",
            auth_status="authenticated",
            http_status=status,
            final_url=final_url,
            redirect_count=redirect_count,
        )
    return _safe_auth_outcome(
        status="unreachable_authenticated_media",
        auth_status="authenticated" if authenticated_account else "registration_auth_required",
        http_status=status,
        redirect_count=redirect_count,
    )


def cvpr_authenticated_event_opener_factory(
    *,
    workspace: Path,
    repository_root: Path,
    frozen_urls: Sequence[str],
    boundary: VirtualSiteBoundary,
    environ: Mapping[str, str] | None = None,
    lstat: Callable[[Path], Any] = os.lstat,
    is_readable: Callable[[Path], bool] | None = None,
    current_uid: Callable[[], int] = os.geteuid,
    cookie_jar_loader: Callable[[Path], CookieJar] = _default_cookie_jar_loader,
    opener_builder: Callable[[CookieJar], Any] = _default_opener_builder,
    capture_page_body: bool = False,
) -> tuple[Callable[[str], dict[str, Any]] | None, dict[str, Any]]:
    """Create a bounded authenticated opener for a separately pinned CVPR route set.

    Unlike the r4 factory below, this transport helper has no r3 candidate-plan
    semantics.  It is for successor inventories whose exact source-backed URL
    set has its own receipt contract.  The cookie path remains closure-only and
    the returned preflight contains no filesystem location or cookie metadata.
    """

    environment = os.environ if environ is None else environ
    readable_check = (lambda path: os.access(path, os.R_OK)) if is_readable is None else is_readable
    allowed_urls = frozenset(frozen_urls)
    if not allowed_urls or any(
        not is_allowed_cvpr_virtual_event_url(url, boundary=boundary) for url in allowed_urls
    ):
        return None, {
            "preflight_status": "blocked",
            "auth_status": "registration_auth_required",
            "secrets_copied": False,
        }
    metadata_valid, cookie_path = _external_session_metadata_is_valid(
        workspace=workspace,
        repository_root=repository_root,
        environ=environment,
        lstat=lstat,
        is_readable=readable_check,
        current_uid=current_uid,
    )
    if not metadata_valid or cookie_path is None:
        return None, {
            "preflight_status": "blocked",
            "auth_status": "registration_auth_required",
            "secrets_copied": False,
        }
    try:
        jar = cookie_jar_loader(cookie_path)
        raw_opener = opener_builder(jar)
    except Exception:
        return None, {
            "preflight_status": "blocked",
            "auth_status": "registration_auth_required",
            "secrets_copied": False,
        }

    def open_registered_virtual_page(url: str) -> dict[str, Any]:
        if url not in allowed_urls or not is_allowed_cvpr_virtual_event_url(url, boundary=boundary):
            raise ValueError("authenticated opener accepts only frozen CVPR event URLs")
        current_url = url
        redirect_count = 0
        while True:
            try:
                response = _open_url(raw_opener, current_url)
            except HTTPError as exc:
                body, _content_type = _bounded_response_html(exc)
                authenticated_account, registration_wall = _html_access_signals(body)
                return _outcome_from_response(
                    response=exc,
                    current_url=current_url,
                    redirect_count=redirect_count,
                    allowed_final_urls=allowed_urls,
                    boundary=boundary,
                    authenticated_account=authenticated_account,
                    registration_wall=registration_wall,
                )
            except Exception:
                return _safe_auth_outcome(
                    status="unreachable_authenticated_media",
                    auth_status="registration_auth_required",
                    redirect_count=redirect_count,
                )
            try:
                status = _response_status(response)
                if status is not None and 300 <= status < 400:
                    location = _response_location(response)
                    if not location:
                        return _safe_auth_outcome(
                            status="unreachable_authenticated_media",
                            auth_status="registration_auth_required",
                            http_status=status,
                            redirect_count=redirect_count,
                        )
                    try:
                        next_url = _sanitize_source_url(urljoin(current_url, location))
                    except ValueError:
                        next_url = ""
                    if next_url not in allowed_urls or not is_allowed_cvpr_virtual_event_url(next_url, boundary=boundary):
                        return _safe_auth_outcome(
                            status="registration_auth_required" if _is_login_redirect(location) else "unreachable_authenticated_media",
                            auth_status="registration_auth_required",
                            http_status=status,
                            redirect_count=redirect_count,
                        )
                    redirect_count += 1
                    if redirect_count > _MAX_AUTHENTICATED_REDIRECTS:
                        return _safe_auth_outcome(
                            status="unreachable_authenticated_media",
                            auth_status="registration_auth_required",
                            http_status=status,
                            redirect_count=redirect_count,
                        )
                    current_url = next_url
                    continue
                body, content_type = _bounded_response_html(response)
                authenticated_account, registration_wall = _html_access_signals(body)
                outcome = _outcome_from_response(
                    response=response,
                    current_url=current_url,
                    redirect_count=redirect_count,
                    allowed_final_urls=allowed_urls,
                    boundary=boundary,
                    authenticated_account=authenticated_account,
                    registration_wall=registration_wall,
                )
                if not capture_page_body or outcome.get("status") != "authenticated":
                    return outcome
                if not isinstance(body, bytes) or not isinstance(content_type, str):
                    return _safe_auth_outcome(
                        status="unreachable_authenticated_media",
                        auth_status="authenticated",
                        http_status=outcome.get("http_status") if isinstance(outcome.get("http_status"), int) else None,
                        redirect_count=redirect_count,
                    )
                return {**outcome, "content_type": content_type, "body": body}
            finally:
                closer = getattr(response, "close", None)
                if callable(closer):
                    closer()

    return open_registered_virtual_page, {
        "preflight_status": "pass",
        "auth_status": "preflight_passed",
        "secrets_copied": False,
    }


def cvpr_registered_virtual_opener_factory(
    *,
    workspace: Path,
    repository_root: Path,
    candidate_plan: Mapping[str, Any],
    boundary: VirtualSiteBoundary,
    verified_predecessors: Sequence[Mapping[str, Any]],
    capacity: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
    lstat: Callable[[Path], Any] = os.lstat,
    is_readable: Callable[[Path], bool] | None = None,
    current_uid: Callable[[], int] = os.geteuid,
    cookie_jar_loader: Callable[[Path], CookieJar] = _default_cookie_jar_loader,
    opener_builder: Callable[[CookieJar], Any] = _default_opener_builder,
    capture_page_body: bool = False,
) -> tuple[Callable[[str], dict[str, Any]] | None, dict[str, Any]]:
    """Return a bounded authenticated opener or a sanitized blocked preflight.

    The jar path remains a closure-only runtime input and is never returned or
    written.  A caller receives no opener until metadata, lineage, scope, and
    capacity gates all pass.
    """

    environment = os.environ if environ is None else environ
    readable_check = (lambda path: os.access(path, os.R_OK)) if is_readable is None else is_readable
    preflight = preflight_registered_virtual_session(
        workspace=workspace,
        repository_root=repository_root,
        candidate_plan=candidate_plan,
        boundary=boundary,
        verified_predecessors=verified_predecessors,
        capacity=capacity,
        environ=environment,
        lstat=lstat,
        is_readable=readable_check,
        current_uid=current_uid,
    )
    if preflight["preflight_status"] != "pass":
        return None, preflight
    metadata_valid, cookie_path = _external_session_metadata_is_valid(
        workspace=workspace,
        repository_root=repository_root,
        environ=environment,
        lstat=lstat,
        is_readable=readable_check,
        current_uid=current_uid,
    )
    if not metadata_valid or cookie_path is None:
        return None, _safe_auth_preflight(passed=False, candidate_plan=candidate_plan, capacity=capacity)
    try:
        jar = cookie_jar_loader(cookie_path)
        raw_opener = opener_builder(jar)
    except Exception:
        return None, _safe_auth_preflight(passed=False, candidate_plan=candidate_plan, capacity=capacity)

    frozen_urls = {
        row["source_url"] for row in transport_eligible_requests(candidate_plan, boundary=boundary)
    }

    def open_registered_virtual_page(url: str) -> dict[str, Any]:
        if url not in frozen_urls or not is_allowed_cvpr_virtual_event_url(url, boundary=boundary):
            raise ValueError("authenticated opener accepts only frozen transport-eligible CVPR event URLs")
        current_url = url
        redirect_count = 0
        while True:
            try:
                response = _open_url(raw_opener, current_url)
            except HTTPError as exc:
                body, _content_type = _bounded_response_html(exc)
                authenticated_account, registration_wall = _html_access_signals(body)
                return _outcome_from_response(
                    response=exc,
                    current_url=current_url,
                    redirect_count=redirect_count,
                    allowed_final_urls=frozenset(frozen_urls),
                    boundary=boundary,
                    authenticated_account=authenticated_account,
                    registration_wall=registration_wall,
                )
            except Exception:
                return _safe_auth_outcome(
                    status="unreachable_authenticated_media",
                    auth_status="registration_auth_required",
                    redirect_count=redirect_count,
                )
            try:
                status = _response_status(response)
                if status is not None and 300 <= status < 400:
                    location = _response_location(response)
                    if not location:
                        return _safe_auth_outcome(
                            status="unreachable_authenticated_media",
                            auth_status="registration_auth_required",
                            http_status=status,
                            redirect_count=redirect_count,
                        )
                    try:
                        next_url = _sanitize_source_url(urljoin(current_url, location))
                    except ValueError:
                        next_url = ""
                    if next_url not in frozen_urls or not is_allowed_cvpr_virtual_event_url(next_url, boundary=boundary):
                        return _safe_auth_outcome(
                            status="registration_auth_required"
                            if _is_login_redirect(location)
                            else "unreachable_authenticated_media",
                            auth_status="registration_auth_required",
                            http_status=status,
                            redirect_count=redirect_count,
                        )
                    redirect_count += 1
                    if redirect_count > _MAX_AUTHENTICATED_REDIRECTS:
                        return _safe_auth_outcome(
                            status="unreachable_authenticated_media",
                            auth_status="registration_auth_required",
                            http_status=status,
                            redirect_count=redirect_count,
                        )
                    current_url = next_url
                    continue
                body, content_type = _bounded_response_html(response)
                authenticated_account, registration_wall = _html_access_signals(body)
                outcome = _outcome_from_response(
                    response=response,
                    current_url=current_url,
                    redirect_count=redirect_count,
                    allowed_final_urls=frozenset(frozen_urls),
                    boundary=boundary,
                    authenticated_account=authenticated_account,
                    registration_wall=registration_wall,
                )
                if not capture_page_body or outcome.get("status") != "authenticated":
                    return outcome
                if (
                    not isinstance(body, bytes)
                    or not isinstance(content_type, str)
                ):
                    return _safe_auth_outcome(
                        status="unreachable_authenticated_media",
                        auth_status="authenticated",
                        http_status=outcome.get("http_status") if isinstance(outcome.get("http_status"), int) else None,
                        redirect_count=redirect_count,
                    )
                return {**outcome, "content_type": content_type, "body": body}
            finally:
                closer = getattr(response, "close", None)
                if callable(closer):
                    closer()

    return open_registered_virtual_page, preflight


_TERMINAL_OFFICIAL_PAGE_STATUSES = frozenset(
    {
        "direct_youtube_recovered",
        "registration_auth_required",
        "registered_access_denied",
        "official_video_not_published",
        "unreachable_authenticated_media",
        "no_admissible_official_youtube_link",
        "unsupported_official_embed_host",
        _BOUNDARY_DISPOSITION,
    }
)
_SAFE_AUTH_STATUSES = frozenset(
    {"authenticated", "registration_auth_required", "registered_access_denied", "preflight_passed"}
)


def extract_cvpr_youtube_nocookie_links(
    page_html: str, *, parent_url: str, boundary: VirtualSiteBoundary
) -> list[str]:
    """Canonicalize privacy embeds only from an already validated CVPR event page.

    ``youtube-nocookie.com`` is deliberately absent from the shared embedded
    host policy.  This narrow parser turns a validated id into the normal
    YouTube watch URL before the standard YouTube host/id gate is applied.
    """

    if not is_allowed_cvpr_virtual_event_url(parent_url, boundary=boundary):
        return []
    seen: set[str] = set()
    links: list[str] = []
    for raw_video_id in _NOCOOKIE_EMBED_RE.findall(page_html):
        canonical = f"https://www.youtube.com/watch?v={raw_video_id}"
        video_id = extract_video_id(canonical)
        if video_id is None:
            continue
        safe = f"https://www.youtube.com/watch?v={video_id}"
        if safe not in seen:
            seen.add(safe)
            links.append(safe)
    return links


def existing_youtube_caption_helper(
    *,
    runner: Callable[[list[str]], Any],
    yt_dlp_cmd: Sequence[str],
    langs: Sequence[str] = ("en",),
) -> Callable[[HarvestedVideo, Path], Mapping[str, Any] | None]:
    """Adapt the established caption-first primitive for r4 media records."""

    def helper(video: HarvestedVideo, destination: Path) -> Mapping[str, Any] | None:
        return download_captions(
            video,
            destination,
            runner=runner,
            yt_dlp_cmd=yt_dlp_cmd,
            langs=langs,
        )

    return helper


def _safe_relative_path(workspace: Path, path: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except (OSError, ValueError):
        return None


def _caption_mime(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".vtt":
        return "text/vtt"
    if suffix == ".srt":
        return "application/x-subrip"
    return None


def _caption_record_from_helper(
    *,
    workspace: Path,
    video_id: str,
    helper_result: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(helper_result, Mapping):
        return None
    raw_path = helper_result.get("caption_path")
    if not isinstance(raw_path, (str, Path)):
        return None
    path = Path(raw_path)
    captions_root = (workspace / OVERLAY_CAPTIONS_RELATIVE_PATH).resolve()
    try:
        resolved = path.resolve()
        resolved.relative_to(captions_root)
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        return None
    mime = _caption_mime(resolved)
    local_path = _safe_relative_path(workspace, resolved)
    if mime is None or local_path is None or not caption_sidecar_matches_video_id(resolved, video_id):
        return None
    return {
        "local_path": local_path,
        "mime": mime,
        "byte_count": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "transcript_source": str(helper_result.get("transcript_source") or "youtube_caption_auto"),
        "language": helper_result.get("language") if isinstance(helper_result.get("language"), str) else None,
    }


def _caption_record_is_valid(*, workspace: Path, video_id: str, caption: Any) -> bool:
    if not isinstance(caption, Mapping):
        return False
    local_path = caption.get("local_path")
    if not isinstance(local_path, str):
        return False
    candidate = workspace / local_path
    expected_root = workspace / OVERLAY_CAPTIONS_RELATIVE_PATH
    try:
        resolved = candidate.resolve()
        resolved.relative_to(expected_root.resolve())
    except (OSError, ValueError):
        return False
    if (
        not resolved.is_file()
        or resolved.stat().st_size <= 0
        or not caption_sidecar_matches_video_id(resolved, video_id)
    ):
        return False
    return (
        caption.get("mime") == _caption_mime(resolved)
        and caption.get("byte_count") == resolved.stat().st_size
        and caption.get("sha256") == _sha256_file(resolved)
    )


def _candidate_plan_matches_overlay(
    *,
    overlay: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    boundary: VirtualSiteBoundary,
    expected_scope: RegistrationMediaScope,
) -> None:
    _validate_candidate_plan(candidate_plan, boundary=boundary, expected_scope=expected_scope)
    validate_registration_media_overlay(overlay, boundary=boundary, expected_scope=expected_scope)
    lineage = overlay.get("predecessor_lineage")
    if not isinstance(lineage, Mapping) or lineage.get("candidate_plan_sha256") != candidate_plan.get(
        "candidate_plan_sha256"
    ):
        raise ValueError("overlay candidate-plan lineage is incompatible")
    plan_rows = candidate_plan.get("request_rows")
    overlay_rows = overlay.get("official_event_requests")
    if not isinstance(plan_rows, list) or not isinstance(overlay_rows, list):
        raise ValueError("overlay request mapping is invalid")
    expected = {row.get("request_key"): row for row in plan_rows if isinstance(row, Mapping)}
    observed = {row.get("request_key"): row for row in overlay_rows if isinstance(row, Mapping)}
    if set(expected) != set(observed) or len(expected) != len(plan_rows) or len(observed) != len(overlay_rows):
        raise ValueError("overlay request mapping differs from the frozen candidate plan")
    for request_key, planned in expected.items():
        actual = observed[request_key]
        if any(
            actual.get(field) != planned.get(field)
            for field in ("source_url", "linked_program_item_ids", "transport_eligible")
        ):
            raise ValueError("overlay request provenance differs from the frozen candidate plan")


def _safe_page_outcome(
    raw: Any, *, request_url: str, boundary: VirtualSiteBoundary
) -> tuple[dict[str, Any], bytes | None]:
    """Normalize an injected authenticated-page result without retaining a body."""

    if not isinstance(raw, Mapping):
        return {"result_status": "unreachable_authenticated_media", "auth_status": "registration_auth_required"}, None
    status = raw.get("status")
    auth_status = raw.get("auth_status")
    safe_auth = auth_status if auth_status in _SAFE_AUTH_STATUSES else "registration_auth_required"
    http_status = raw.get("http_status")
    safe_http = http_status if isinstance(http_status, int) else None
    if status != "authenticated":
        final_status = status if status in _TERMINAL_OFFICIAL_PAGE_STATUSES else "unreachable_authenticated_media"
        return {
            "result_status": final_status,
            "auth_status": safe_auth,
            "http_status": safe_http,
            "attempts": [{"attempt": 1, "http_status": safe_http, "reason": "authenticated_page"}],
            "retry_events": [],
        }, None
    try:
        final_url = _sanitize_source_url(raw.get("final_url") if isinstance(raw.get("final_url"), str) else request_url)
    except ValueError:
        final_url = ""
    body = raw.get("body")
    content_type = raw.get("content_type")
    if (
        safe_http is None
        or safe_http < 200
        or safe_http >= 300
        or not is_allowed_cvpr_virtual_event_url(final_url, boundary=boundary)
        or not isinstance(content_type, str)
        or not content_type.lower().split(";", 1)[0].strip() in {"text/html", "application/xhtml+xml"}
        or not isinstance(body, bytes)
        or not body
        or len(body) > _MAX_OFFICIAL_EVENT_PAGE_BYTES
    ):
        return {
            "result_status": "unreachable_authenticated_media",
            "auth_status": safe_auth,
            "http_status": safe_http,
            "attempts": [{"attempt": 1, "http_status": safe_http, "reason": "authenticated_page"}],
            "retry_events": [],
        }, None
    return {
        "result_status": "authenticated",
        "auth_status": "authenticated",
        "http_status": safe_http,
        "final_url": final_url,
        "content_type": content_type.split(";", 1)[0].strip().lower(),
        "page_byte_count": len(body),
        "page_sha256": sha256(body).hexdigest(),
        "attempts": [{"attempt": 1, "http_status": safe_http, "reason": "authenticated_page"}],
        "retry_events": [],
    }, body


def _media_candidates_from_page(
    *, page_html: str, parent_url: str, boundary: VirtualSiteBoundary
) -> tuple[dict[str, str], bool]:
    """Return canonical direct YouTube ids plus whether an unsupported embed appeared."""

    candidates: dict[str, str] = {}
    unsupported_embed_seen = False
    for link in embedded_media_links(page_html, base_url=parent_url):
        # Sanitize only receipt fields.  A normal YouTube ID can contain words
        # such as "token", which the generic URL sanitizer must redact.
        video_id = extract_video_id(link.url)
        if video_id is not None and url_host_allowed(link.url, DEFAULT_ALLOWED_HOSTS):
            candidates.setdefault(video_id, "direct_youtube_reference")
        else:
            unsupported_embed_seen = True
    for safe_url in extract_cvpr_youtube_nocookie_links(
        page_html, parent_url=parent_url, boundary=boundary
    ):
        video_id = extract_video_id(safe_url)
        if video_id is not None and url_host_allowed(safe_url, DEFAULT_ALLOWED_HOSTS):
            candidates.setdefault(video_id, "cvpr_local_youtube_nocookie_embed")
    return candidates, unsupported_embed_seen


def _media_record(
    *,
    request: Mapping[str, Any],
    video_id: str,
    provenance: str,
    workspace: Path,
    caption_helper: Callable[[HarvestedVideo, Path], Mapping[str, Any] | None] | None,
) -> dict[str, Any]:
    video = HarvestedVideo(
        video_id=video_id,
        title="",
        url=f"https://www.youtube.com/watch?v={video_id}",
        source="official_event_page",
    )
    caption = None
    if caption_helper is not None:
        try:
            caption = _caption_record_from_helper(
                workspace=workspace,
                video_id=video_id,
                helper_result=caption_helper(video, workspace / OVERLAY_CAPTIONS_RELATIVE_PATH),
            )
        except Exception:
            caption = None
    return {
        "parent_request_key": request["request_key"],
        "linked_program_item_ids": list(request["linked_program_item_ids"]),
        "video_id": video_id,
        "media_url": video.url,
        "provenance": provenance,
        "caption_status": "youtube_caption_available" if caption is not None else "youtube_caption_unavailable",
        "caption": caption,
        "audio_downloaded": False,
        "video_downloaded": False,
    }


def _refresh_registration_media_accounting(overlay: dict[str, Any]) -> None:
    request_rows = overlay.get("official_event_requests")
    program_rows = overlay.get("program_item_accounting")
    media_rows = overlay.get("official_event_media_records")
    if not isinstance(request_rows, list) or not isinstance(program_rows, list) or not isinstance(media_rows, list):
        raise ValueError("registration-media overlay accounting is malformed")
    by_request = {row.get("request_key"): row for row in request_rows if isinstance(row, Mapping)}
    if len(by_request) != len(request_rows):
        raise ValueError("registration-media overlay request keys are invalid")
    for item in program_rows:
        if not isinstance(item, dict) or item.get("result_origin") != "r4_registration_media":
            continue
        request = by_request.get(item.get("r3_request_key"))
        if not isinstance(request, Mapping):
            raise ValueError("r4 program item has no matching request")
        result_status = request.get("result_status") or "pending"
        item["result_status"] = result_status
        if result_status in {"direct_youtube_recovered", "pending"}:
            item["blocker"] = None
        else:
            item["blocker"] = {"code": result_status}
    completed = sum(
        isinstance(row, Mapping) and row.get("result_status") in _TERMINAL_OFFICIAL_PAGE_STATUSES
        for row in request_rows
    )
    transport = sum(isinstance(row, Mapping) and row.get("transport_eligible") is True for row in request_rows)
    boundary = sum(isinstance(row, Mapping) and row.get("transport_eligible") is False for row in request_rows)
    overlay["recovery_counts"] = {
        "completed_request_count": completed,
        "direct_youtube_media_record_count": len(media_rows),
        "transport_eligible_request_count": transport,
        "boundary_no_network_request_count": boundary,
    }
    overlay["status"] = "registration_media_complete" if completed == len(request_rows) else "registration_media_partial"
    overlay["secrets_copied"] = False
    overlay["overlay_sha256"] = canonical_registration_media_digest(overlay)


def _request_media_records_valid(*, workspace: Path, request: Mapping[str, Any], media_rows: Sequence[Mapping[str, Any]]) -> bool:
    if request.get("result_status") != "direct_youtube_recovered":
        return True
    records = [row for row in media_rows if row.get("parent_request_key") == request.get("request_key")]
    if not records:
        return False
    for record in records:
        if record.get("audio_downloaded") is not False or record.get("video_downloaded") is not False:
            return False
        if record.get("caption_status") == "youtube_caption_available" and not _caption_record_is_valid(
            workspace=workspace,
            video_id=str(record.get("video_id") or ""),
            caption=record.get("caption"),
        ):
            return False
    return True


def resume_registration_media_overlay(
    *,
    fresh_overlay: Mapping[str, Any],
    prior_overlay: Mapping[str, Any],
    workspace: Path,
    boundary: VirtualSiteBoundary,
    expected_scope: RegistrationMediaScope = FULL_R4_SCOPE,
) -> dict[str, Any]:
    """Reuse only compatible terminal r4 facts and verified caption sidecars."""

    validate_registration_media_overlay(fresh_overlay, boundary=boundary, expected_scope=expected_scope)
    validate_registration_media_overlay(prior_overlay, boundary=boundary, expected_scope=expected_scope)
    fresh_lineage = fresh_overlay.get("predecessor_lineage")
    prior_lineage = prior_overlay.get("predecessor_lineage")
    if not isinstance(fresh_lineage, Mapping) or not isinstance(prior_lineage, Mapping):
        raise ValueError("registration-media resume lacks predecessor lineage")
    if fresh_lineage.get("candidate_plan_sha256") != prior_lineage.get("candidate_plan_sha256"):
        raise ValueError("registration-media resume candidate-plan lineage is incompatible")
    fresh = copy.deepcopy(dict(fresh_overlay))
    prior_requests = prior_overlay.get("official_event_requests")
    fresh_requests = fresh.get("official_event_requests")
    prior_media = prior_overlay.get("official_event_media_records")
    if not isinstance(prior_requests, list) or not isinstance(fresh_requests, list) or not isinstance(prior_media, list):
        raise ValueError("registration-media resume rows are invalid")
    prior_by_key = {row.get("request_key"): row for row in prior_requests if isinstance(row, Mapping)}
    fresh_by_key = {row.get("request_key"): row for row in fresh_requests if isinstance(row, dict)}
    if set(prior_by_key) != set(fresh_by_key) or len(prior_by_key) != len(prior_requests):
        raise ValueError("registration-media resume request mapping is incompatible")
    fresh_media: list[dict[str, Any]] = []
    for request_key, target in fresh_by_key.items():
        prior = prior_by_key[request_key]
        if any(
            prior.get(field) != target.get(field)
            for field in ("source_url", "linked_program_item_ids", "transport_eligible")
        ):
            raise ValueError("registration-media resume request provenance is incompatible")
        status = prior.get("result_status")
        request_media = [row for row in prior_media if isinstance(row, Mapping) and row.get("parent_request_key") == request_key]
        if (
            status not in _TERMINAL_OFFICIAL_PAGE_STATUSES
            or status in {"registration_auth_required", "registered_access_denied"}
            or not _request_media_records_valid(
                workspace=workspace,
                request=prior,
                media_rows=request_media,
            )
        ):
            if target.get("transport_eligible") is True:
                target["result_status"] = "pending"
                for field in (
                    "auth_status",
                    "http_status",
                    "final_url",
                    "content_type",
                    "page_byte_count",
                    "page_sha256",
                    "attempts",
                    "retry_events",
                ):
                    target.pop(field, None)
            else:
                target["result_status"] = _BOUNDARY_DISPOSITION
            continue
        for field in (
            "result_status",
            "auth_status",
            "http_status",
            "final_url",
            "content_type",
            "page_byte_count",
            "page_sha256",
            "attempts",
            "retry_events",
        ):
            if field in prior:
                target[field] = copy.deepcopy(prior[field])
        fresh_media.extend(copy.deepcopy(dict(row)) for row in request_media)
    fresh["official_event_media_records"] = fresh_media
    _refresh_registration_media_accounting(fresh)
    validate_registration_media_overlay(fresh, boundary=boundary, expected_scope=expected_scope)
    return fresh


def recover_registered_official_pages(
    *,
    overlay: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    boundary: VirtualSiteBoundary,
    workspace: Path,
    authenticated_page_opener: Callable[[str], Any],
    caption_helper: Callable[[HarvestedVideo, Path], Mapping[str, Any] | None] | None = None,
    checkpoint: Callable[[Mapping[str, Any]], None] | None = None,
    max_attempts: int = 3,
    stop_on_auth_failure: bool = False,
    expected_scope: RegistrationMediaScope = FULL_R4_SCOPE,
) -> dict[str, Any]:
    """Recover direct YouTube provenance from exact registered CVPR event pages.

    ``authenticated_page_opener`` is injected so this pure recovery layer never
    manages browser state or a network transport.  Page bodies are used only in
    local variables for parsing and are never copied into the returned overlay.
    """

    if not callable(authenticated_page_opener) or max_attempts <= 0:
        raise ValueError("registered official-page recovery requires an opener and positive retry bound")
    _candidate_plan_matches_overlay(
        overlay=overlay,
        candidate_plan=candidate_plan,
        boundary=boundary,
        expected_scope=expected_scope,
    )
    result = copy.deepcopy(dict(overlay))
    request_rows = result.get("official_event_requests")
    if not isinstance(request_rows, list):
        raise ValueError("registration-media overlay has no official event request rows")
    media_rows = result.get("official_event_media_records")
    if not isinstance(media_rows, list):
        raise ValueError("registration-media overlay media rows are invalid")
    media_by_pair = {
        (str(row.get("parent_request_key")), str(row.get("video_id"))): row
        for row in media_rows
        if isinstance(row, Mapping)
    }
    for request in request_rows:
        if not isinstance(request, dict):
            raise ValueError("registration-media request row is invalid")
        if request.get("transport_eligible") is not True:
            request["result_status"] = _BOUNDARY_DISPOSITION
            request["final_disposition"] = _BOUNDARY_DISPOSITION
            request.pop("opener_input", None)
            continue
        if request.get("result_status") in _TERMINAL_OFFICIAL_PAGE_STATUSES and _request_media_records_valid(
            workspace=workspace, request=request, media_rows=list(media_by_pair.values())
        ):
            continue
        request_key = request.get("request_key")
        if not isinstance(request_key, str) or not isinstance(request.get("source_url"), str):
            raise ValueError("registration-media request lacks immutable source evidence")
        for pair in [pair for pair in media_by_pair if pair[0] == request_key]:
            media_by_pair.pop(pair, None)
        page_result: dict[str, Any] | None = None
        page_body: bytes | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                raw = authenticated_page_opener(request["source_url"])
                page_result, page_body = _safe_page_outcome(
                    raw, request_url=request["source_url"], boundary=boundary
                )
            except Exception:
                page_result = {
                    "result_status": "unreachable_authenticated_media",
                    "auth_status": "registration_auth_required",
                    "http_status": None,
                    "attempts": [{"attempt": attempt, "http_status": None, "reason": "authenticated_page"}],
                    "retry_events": [],
                }
                page_body = None
            if page_result["result_status"] != "unreachable_authenticated_media" or attempt == max_attempts:
                break
        assert page_result is not None
        request.update(page_result)
        if page_body is not None:
            page_html = page_body.decode("utf-8", errors="replace")
            candidates, unsupported_embed_seen = _media_candidates_from_page(
                page_html=page_html,
                parent_url=str(request.get("final_url") or request["source_url"]),
                boundary=boundary,
            )
            if not candidates:
                request["result_status"] = (
                    "unsupported_official_embed_host"
                    if unsupported_embed_seen
                    else "no_admissible_official_youtube_link"
                )
            else:
                request["result_status"] = "direct_youtube_recovered"
                for video_id, provenance in sorted(candidates.items()):
                    media_by_pair[(request_key, video_id)] = _media_record(
                        request=request,
                        video_id=video_id,
                        provenance=provenance,
                        workspace=workspace,
                        caption_helper=caption_helper,
                    )
        result["official_event_media_records"] = [
            media_by_pair[key] for key in sorted(media_by_pair)
        ]
        if checkpoint is not None:
            _refresh_registration_media_accounting(result)
            checkpoint(result)
        if stop_on_auth_failure and request.get("result_status") in {
            "registration_auth_required",
            "registered_access_denied",
        }:
            break
    result["official_event_media_records"] = [media_by_pair[key] for key in sorted(media_by_pair)]
    _refresh_registration_media_accounting(result)
    validate_registration_media_overlay(result, boundary=boundary, expected_scope=expected_scope)
    return result


# --------------------------------------------------------------------------
# Stage 4 lane owner: registration media (r4)
# --------------------------------------------------------------------------

from conference_pipeline.provider_receipts import sanitize_for_receipt as _sanitize_for_receipt  # noqa: E402

REGISTRATION_MEDIA_LANE_ID = "registration-media"
REGISTRATION_MEDIA_PROFILE = "registration-media"


@dataclass(frozen=True)
class RegistrationMediaLaneSettings:
    """A conference's registration-media lane identity and paths.

    ``r3_run_id`` names the pinned public run whose policy rows define the
    candidate plan; ``execution_status_key`` names the Stage 4 execution
    status check; ``owner_function`` is recorded as the owner's entry point.
    """

    execution_status_key: str
    owner_function: str
    r3_run_id: str
    checkpoint_path: Path
    ledger_path: Path
    run_id: str = R4_RUN_ID
    full_scope: RegistrationMediaScope = FULL_R4_SCOPE
    overlay_path: Path = OVERLAY_LEDGER_RELATIVE_PATH


def registration_media_inputs(
    workspace: Path, *, boundary: VirtualSiteBoundary, r3_run_id: str, ledger_path: Path
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Build r4 inputs only from the immutable pinned r3 source surface."""

    predecessors = load_pinned_r3_predecessors(workspace)
    try:
        r3_ledger = json.loads((workspace / ledger_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("pinned r3 download ledger is unavailable") from exc
    if not isinstance(r3_ledger, Mapping):
        raise ValueError("pinned r3 download ledger must be an object")
    request_rows = r3_ledger.get("request_results")
    program_rows = r3_ledger.get("program_item_results")
    if not isinstance(request_rows, list) or not isinstance(program_rows, list):
        raise ValueError("pinned r3 download ledger lacks complete result rows")
    candidate_plan = build_candidate_plan(
        r3_run_id=r3_run_id,
        predecessors=predecessors,
        r3_request_results=request_rows,
        r3_program_item_results=program_rows,
        boundary=boundary,
    )
    overlay = build_registration_media_overlay(
        candidate_plan=candidate_plan,
        r3_program_item_results=program_rows,
        boundary=boundary,
    )
    return predecessors, candidate_plan, overlay


def registration_media_capacity(
    *,
    workspace: Path,
    candidate_plan: Mapping[str, Any],
    read_json_object: Callable[..., Any],
    available_bytes: Any,
    cap_hits: Sequence[str] = (),
) -> dict[str, Any]:
    """Create only digest-bound, secret-free capacity evidence for r4."""

    from conference_pipeline.production.download_corpus_stage import free_disk_bytes, read_storage_controls
    from conference_pipeline.production_preflight import canonical_json_digest

    storage_manifest, safety_margin, blockers = read_storage_controls(workspace, read_json_object=read_json_object)
    candidate_digest = candidate_plan.get("candidate_plan_sha256")
    available_bytes = free_disk_bytes(workspace, available_bytes)
    valid_available = isinstance(available_bytes, int) and not isinstance(available_bytes, bool)
    no_cap_hits = not cap_hits
    manifest_sha256 = canonical_json_digest(storage_manifest) if isinstance(storage_manifest, Mapping) else None
    status = "pass"
    if blockers or not valid_available or int(available_bytes) <= 0 or not no_cap_hits or not isinstance(manifest_sha256, str):
        status = "blocked"
    elif int(available_bytes) < safety_margin:
        status = "blocked"
    return {
        "status": status,
        "available_bytes": int(available_bytes) if valid_available else None,
        "positive_available_bytes": bool(valid_available and int(available_bytes) > 0),
        "required_bytes": safety_margin,
        "no_cap_hits": no_cap_hits,
        "manifest_sha256": manifest_sha256,
        "candidate_plan_sha256": candidate_digest if isinstance(candidate_digest, str) else None,
        "secrets_copied": False,
    }


def registration_media_checks(
    *,
    overlay: Mapping[str, Any] | None,
    capacity: Mapping[str, Any],
    preflight: Mapping[str, Any],
    settings: RegistrationMediaLaneSettings,
    overlay_file_sha256: str | None = None,
    authenticated_request_count: int = 0,
) -> dict[str, Any]:
    """Project the only r4 facts the strict Stage-4 profile accepts."""

    counts = overlay.get("counts") if isinstance(overlay, Mapping) else {}
    recovery_counts = overlay.get("recovery_counts") if isinstance(overlay, Mapping) else {}
    request_rows = overlay.get("official_event_requests") if isinstance(overlay, Mapping) else []
    counts = counts if isinstance(counts, Mapping) else {}
    recovery_counts = recovery_counts if isinstance(recovery_counts, Mapping) else {}
    request_rows = request_rows if isinstance(request_rows, list) else []
    eligible_auth_failure = any(
        isinstance(row, Mapping)
        and row.get("transport_eligible") is True
        and row.get("result_status") in {"registration_auth_required", "registered_access_denied"}
        for row in request_rows
    )
    complete = (
        isinstance(overlay, Mapping)
        and overlay.get("status") == "registration_media_complete"
        and recovery_counts.get("completed_request_count") == settings.full_scope.policy_request_count
        and not eligible_auth_failure
    )
    request_count = (
        authenticated_request_count
        if isinstance(authenticated_request_count, int) and authenticated_request_count >= 0
        else 0
    )
    return {
        "stage_contract_profile": REGISTRATION_MEDIA_PROFILE,
        settings.execution_status_key: "pass" if complete else "blocked",
        "authenticated_registration_media_recovery_status": "pass" if complete else "blocked",
        "official_event_media_accounting_complete": complete,
        "official_event_request_count": counts.get("policy_request_count", 0),
        "official_event_program_item_count": counts.get("candidate_program_item_count", 0),
        "transport_eligible_request_count": counts.get("transport_eligible_request_count", 0),
        "transport_eligible_program_item_count": counts.get("transport_eligible_program_item_count", 0),
        "boundary_no_network_request_count": counts.get("boundary_request_count", 0),
        "boundary_no_network_program_item_count": counts.get("boundary_program_item_count", 0),
        "program_item_count": counts.get("total_program_item_count", 0),
        "program_item_result_count": len(overlay.get("program_item_accounting", [])) if isinstance(overlay, Mapping) else 0,
        "complete_program_item_accounting": complete,
        "no_cap_hits": capacity.get("no_cap_hits") is True,
        "disk_capacity": {
            "status": capacity.get("status"),
            "positive_available_bytes": capacity.get("positive_available_bytes") is True,
            "manifest_sha256": capacity.get("manifest_sha256"),
            "candidate_plan_sha256": capacity.get("candidate_plan_sha256"),
        },
        "registration_media_overlay_path": str(settings.overlay_path),
        "registration_media_overlay_ledger_sha256": overlay_file_sha256,
        "registration_media_overlay_canonical_sha256": (
            overlay.get("overlay_sha256") if isinstance(overlay, Mapping) else None
        ),
        "direct_youtube_media_record_count": recovery_counts.get("direct_youtube_media_record_count", 0),
        "registration_media_transport_created": request_count > 0,
        "registration_media_authenticated_request_count": request_count,
        "registration_media_session_preflight": _sanitize_for_receipt(dict(preflight)),
        "secrets_copied": False,
    }


def _registration_owner_evidence(settings: RegistrationMediaLaneSettings, request_count: int) -> dict[str, Any]:
    return {
        "owner_function": settings.owner_function,
        "overlay_path": str(settings.overlay_path),
        "transport_created": request_count > 0,
        "authenticated_request_count": request_count,
    }


def registration_media_stage_receipt(
    conventions: Any,
    *,
    workspace: Path,
    run_id: str,
    status: str,
    blockers: list[dict[str, Any]],
    checks: Mapping[str, Any],
    stage_contract: Mapping[str, Any] | None,
    overlay: Mapping[str, Any] | None,
    settings: RegistrationMediaLaneSettings,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the canonical Stage-4 attempt without reusing r3 ledger fields."""

    from conference_pipeline.production.download_corpus_stage import write_stage_receipt

    request_count = checks.get("registration_media_authenticated_request_count")
    request_count = request_count if isinstance(request_count, int) and request_count >= 0 else 0
    body = {
        "owner_execution_evidence": {
            conventions.owner_id: _registration_owner_evidence(settings, request_count)
        },
        "registration_media_overlay": {
            "path": str(settings.overlay_path),
            "overlay_sha256": overlay.get("overlay_sha256") if isinstance(overlay, Mapping) else None,
            "status": overlay.get("status") if isinstance(overlay, Mapping) else "not_written",
        },
        "provider_calls_made": 0,
        "actual_provider_cost_usd": 0.0,
        "downloads_made": 0,
    }
    receipt_path, receipt, summary = write_stage_receipt(
        conventions,
        workspace=workspace,
        run_id=run_id,
        status=status,
        body=body,
        checks=checks,
        blockers=blockers,
        stage_contract=stage_contract,
        lineage=lineage,
        profile=REGISTRATION_MEDIA_PROFILE,
    )
    return {
        "status": status,
        "receipt_path": str(receipt_path),
        "receipt": receipt,
        "summary": summary,
        "overlay": dict(overlay) if isinstance(overlay, Mapping) else None,
    }


def run_registration_media_lane(
    conventions: Any,
    *,
    workspace: Path,
    run_id: str,
    settings: RegistrationMediaLaneSettings,
    boundary_loader: Callable[[Path], VirtualSiteBoundary],
    input_loader: Callable[..., tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]],
    opener_factory: Callable[..., tuple[Callable[[str], Mapping[str, Any]] | None, Mapping[str, Any]]],
    recoverer: Callable[..., dict[str, Any]],
    repository_root: Path,
    cap_hits: Sequence[str] = (),
    resume: bool = False,
    available_bytes: Any = None,
    validate_run_id: Callable[[str], str] = lambda value: value,
) -> dict[str, Any]:
    """Run only the fixed registered-media Stage-4 lane.

    The first frozen eligible page is a sentinel.  It must prove authenticated
    access before any remaining request can enter recovery; a wall or an
    authenticated-account denial stays a blocked receipt rather than a fake
    complete overlay. ``conventions`` is the orchestrator's
    :class:`~conference_pipeline.production.download_corpus_stage.LedgerConventions`.
    """

    from conference_pipeline.production.download_corpus_stage import stage4_blocker, verify_lane_contract

    def receipt(**kwargs: Any) -> dict[str, Any]:
        return registration_media_stage_receipt(conventions, workspace=resolved_workspace, run_id=safe_run_id, settings=settings, **kwargs)

    def checks(**kwargs: Any) -> dict[str, Any]:
        return registration_media_checks(settings=settings, **kwargs)

    blocked_preflight = {
        "preflight_status": "blocked",
        "auth_status": "registration_auth_required",
        "secrets_copied": False,
    }
    safe_run_id = validate_run_id(run_id)
    if safe_run_id != settings.run_id:
        raise ValueError("registration-media mode requires the fixed r4 run id")
    resolved_workspace = workspace.resolve()
    try:
        boundary = boundary_loader(resolved_workspace)
        predecessors, candidate_plan, initial_overlay = input_loader(resolved_workspace, boundary=boundary)
        validate_registration_media_overlay(initial_overlay, boundary=boundary)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return receipt(
            status="blocked",
            blockers=[stage4_blocker("registration_media_lineage_unavailable", reason=type(exc).__name__)],
            checks=checks(overlay=None, capacity={}, preflight=dict(blocked_preflight)),
            stage_contract=None,
            overlay=None,
        )
    checkpoint_path = resolved_workspace / settings.checkpoint_path
    if resume and checkpoint_path.exists():
        try:
            prior_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if not isinstance(prior_checkpoint, Mapping):
                raise ValueError("registration-media checkpoint must be an object")
            initial_overlay = resume_registration_media_overlay(
                fresh_overlay=initial_overlay,
                prior_overlay=prior_checkpoint,
                workspace=resolved_workspace,
                boundary=boundary,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return receipt(
                status="blocked",
                blockers=[stage4_blocker("registration_media_checkpoint_incompatible", reason=type(exc).__name__)],
                checks=checks(overlay=None, capacity={}, preflight=dict(blocked_preflight)),
                stage_contract=None,
                overlay=None,
            )
    capacity = registration_media_capacity(
        workspace=resolved_workspace,
        candidate_plan=candidate_plan,
        read_json_object=conventions.read_json_object,
        available_bytes=available_bytes,
        cap_hits=cap_hits,
    )
    try:
        opener, preflight = opener_factory(
            workspace=resolved_workspace,
            repository_root=repository_root,
            candidate_plan=candidate_plan,
            boundary=boundary,
            verified_predecessors=predecessors,
            capacity=capacity,
            capture_page_body=True,
        )
    except Exception:
        opener, preflight = None, dict(blocked_preflight)
    if preflight.get("preflight_status") != "pass" or not callable(opener):
        return receipt(
            status="blocked",
            blockers=[stage4_blocker("registration_auth_required")],
            checks=checks(overlay=None, capacity=capacity, preflight=preflight),
            stage_contract=None,
            overlay=None,
        )
    eligible = [
        row for row in candidate_plan.get("request_rows", [])
        if isinstance(row, Mapping) and row.get("transport_eligible") is True
    ]
    if len(eligible) != settings.full_scope.transport_eligible_request_count:
        raise ValueError("registration-media candidate plan has an invalid eligible fanout")
    sentinel_url = eligible[0].get("source_url")
    if not isinstance(sentinel_url, str):
        raise ValueError("registration-media sentinel lacks a source URL")
    authenticated_request_count = 0
    try:
        authenticated_request_count += 1
        sentinel = opener(sentinel_url)
    except Exception:
        sentinel = {"status": "registration_auth_required", "auth_status": "registration_auth_required"}
    sentinel_status = sentinel.get("status") if isinstance(sentinel, Mapping) else None
    sentinel_auth = sentinel.get("auth_status") if isinstance(sentinel, Mapping) else None
    if sentinel_status in {"registration_auth_required", "registered_access_denied"}:
        return receipt(
            status="blocked",
            blockers=[stage4_blocker(str(sentinel_status), sentinel_request_count=1)],
            checks={
                **checks(
                    overlay=None,
                    capacity=capacity,
                    preflight=preflight,
                    authenticated_request_count=authenticated_request_count,
                ),
                "registration_media_sentinel_request_count": 1,
                "registration_media_sentinel_status": sentinel_status,
            },
            stage_contract=None,
            overlay=None,
        )
    if sentinel_status != "authenticated" and sentinel_auth != "authenticated":
        return receipt(
            status="blocked",
            blockers=[stage4_blocker("registration_auth_required", sentinel_request_count=1)],
            checks={
                **checks(
                    overlay=None,
                    capacity=capacity,
                    preflight=preflight,
                    authenticated_request_count=authenticated_request_count,
                ),
                "registration_media_sentinel_request_count": 1,
                "registration_media_sentinel_status": "registration_auth_required",
            },
            stage_contract=None,
            overlay=None,
        )
    sentinel_used = False

    def cached_opener(url: str) -> Mapping[str, Any]:
        nonlocal sentinel_used, authenticated_request_count
        if not sentinel_used and url == sentinel_url:
            sentinel_used = True
            return sentinel
        authenticated_request_count += 1
        return opener(url)

    def checkpoint(payload: Mapping[str, Any]) -> None:
        """Persist only a private, digest-validated r4 recovery checkpoint."""

        validate_registration_media_overlay(payload, boundary=boundary)
        conventions.write_json(checkpoint_path, dict(payload))

    overlay = recoverer(
        overlay=initial_overlay,
        candidate_plan=candidate_plan,
        boundary=boundary,
        workspace=resolved_workspace,
        authenticated_page_opener=cached_opener,
        checkpoint=checkpoint,
        stop_on_auth_failure=True,
    )
    validate_registration_media_overlay(overlay, boundary=boundary)
    recovery_counts = overlay.get("recovery_counts") if isinstance(overlay.get("recovery_counts"), Mapping) else {}
    fatal_auth_status = next(
        (
            str(row.get("result_status"))
            for row in overlay.get("official_event_requests", [])
            if isinstance(row, Mapping)
            and row.get("transport_eligible") is True
            and row.get("result_status") in {"registration_auth_required", "registered_access_denied"}
        ),
        None,
    )
    if fatal_auth_status is not None:
        checkpoint(overlay)
        return receipt(
            status="blocked",
            blockers=[stage4_blocker(fatal_auth_status)],
            checks=checks(
                overlay=overlay,
                capacity=capacity,
                preflight=preflight,
                authenticated_request_count=authenticated_request_count,
            ),
            stage_contract=None,
            overlay=overlay,
        )
    if (
        overlay.get("status") != "registration_media_complete"
        or recovery_counts.get("completed_request_count") != settings.full_scope.policy_request_count
    ):
        checkpoint(overlay)
        return receipt(
            status="blocked",
            blockers=[stage4_blocker("registration_media_accounting_incomplete")],
            checks=checks(
                overlay=overlay,
                capacity=capacity,
                preflight=preflight,
                authenticated_request_count=authenticated_request_count,
            ),
            stage_contract=None,
            overlay=overlay,
        )
    overlay_path = resolved_workspace / settings.overlay_path
    try:
        prior_overlay_bytes = overlay_path.read_bytes()
    except OSError:
        prior_overlay_bytes = None
    conventions.write_json(overlay_path, overlay)
    overlay_file_sha256 = _sha256_file(overlay_path)
    final_checks = checks(
        overlay=overlay,
        capacity=capacity,
        preflight=preflight,
        overlay_file_sha256=overlay_file_sha256,
        authenticated_request_count=authenticated_request_count,
    )
    contract = verify_lane_contract(
        conventions,
        workspace=resolved_workspace,
        run_id=safe_run_id,
        owner_evidence=_registration_owner_evidence(settings, authenticated_request_count),
        checks=final_checks,
        profile=REGISTRATION_MEDIA_PROFILE,
        extra={"run_id": safe_run_id, "stage_id": conventions.stage_id, "status": "succeeded", "secrets_copied": False},
    )
    blockers: list[dict[str, Any]] = []
    if contract["status"] != "pass":
        blockers.append(stage4_blocker("final_registration_media_stage_contract_incomplete"))
        # The strict verifier must read the canonical overlay path, but a
        # failed verification cannot promote the candidate overlay. Restore a
        # prior success if present; otherwise leave this run resumable only in
        # its private checkpoint.
        checkpoint(overlay)
        if prior_overlay_bytes is None:
            try:
                overlay_path.unlink()
            except FileNotFoundError:
                pass
        else:
            conventions.write_bytes(overlay_path, prior_overlay_bytes)
    else:
        try:
            checkpoint_path.unlink()
        except FileNotFoundError:
            pass
    return receipt(
        status="succeeded" if not blockers else "blocked",
        blockers=blockers,
        checks=final_checks,
        stage_contract=contract,
        overlay=overlay,
    )
