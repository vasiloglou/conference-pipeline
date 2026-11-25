"""Event-media coverage: a virtual-program census with caption-only recovery.

The authenticated r4 registration-media run recovered direct YouTube links
from only 2,565 of the conference's 5,023 poster pages.  It also did not
traverse the keynote, tutorial, workshop, session, or other named main-event
pages.  The contracts here create one deterministic inventory for the
complete requested virtual-program surface (they have no network, cookie,
downloader, or filesystem-write behavior); :func:`run_event_media_lane` is the
Stage 4 lane owner that walks the frozen event pages behind one
authenticated sentinel, then fetches only caption sidecars for the recovered
YouTube ids (no audio, no video, no ASR/OCR).

Recommended for
---------------
``RECOMMENDED_FOR = ("CVPR",)`` (formerly ``acquisition/cvpr_event_media.py``;
that import path remains a compatibility alias).

* **CVPR stage and quirk covered:** Stage 4 ``download-corpus`` r5, "event
  media": keynote, tutorial, workshop, oral and session pages on the CVPR
  virtual site, plus a reuse of r4's poster links.
* **May also fit:** any conference whose virtual-site event pages embed
  official YouTube recordings (ICCV, WACV, ECCV).

The census counts and the fixed r5 run id below pin the completed CVPR 2026
run (registry/runtime data, revisited in the registry batch).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit

from conference_pipeline.acquisition.embedded_media import embedded_media_links
from conference_pipeline.acquisition.youtube_talks import (
    DEFAULT_ALLOWED_HOSTS,
    caption_sidecar_matches_video_id,
    extract_video_id,
    url_host_allowed,
)
from conference_pipeline.conference_identity import VirtualSiteBoundary


RECOMMENDED_FOR = ("CVPR",)

R5_EVENT_MEDIA_RUN_ID = "cvpr2026-live-production-r5-event-media"
R4_REUSED_POSTER_ROUTE_COUNT = 2_565
EVENT_MEDIA_INVENTORY_RELATIVE_PATH = Path(
    "storage/download-corpus/event-media/event_media_inventory.json"
)
EVENT_MEDIA_CAPTIONS_RELATIVE_PATH = Path("storage/download-corpus/event-media/captions")

# The census is pinned to the succeeded r3 program-schedule snapshot.  It is a
# CVPR-local contract, not a reusable conference taxonomy.
EVENT_CATEGORY_COUNTS: dict[str, int] = {
    "poster": 5_023,
    "keynote": 3,
    "tutorial": 19,
    "workshop": 153,
    "oral": 24,
    "session": 19,
    "main_event": 17,
}

_ROUTE_CATEGORY = {
    "poster": "poster",
    "invited-talk": "keynote",
    "tutorial": "tutorial",
    "workshop": "workshop",
    "demonstration": "main_event",
    "events": "main_event",
    "meeting": "main_event",
    "mentorship": "main_event",
    "reception": "main_event",
    "remarks": "main_event",
}
_EVENT_ROUTE_FAMILIES = frozenset({*_ROUTE_CATEGORY, "session"})
_PENDING_PAGE_STATUS = "pending"
_TERMINAL_PAGE_STATUSES = frozenset(
    {
        "direct_youtube_recovered",
        "no_admissible_official_youtube_link",
        "registration_auth_required",
        "registered_access_denied",
        "official_video_not_published",
        "unreachable_authenticated_media",
        "unsupported_official_embed_host",
        "source_mapping_blocker",
    }
)
_CAPTION_STATUSES = frozenset(
    {"pending", "youtube_caption_available", "youtube_caption_unavailable"}
)
_REUSABLE_R5_PAGE_STATUSES = _TERMINAL_PAGE_STATUSES.difference(
    {"registration_auth_required", "registered_access_denied"}
)
_VOLATILE_FIELDS = frozenset({"event_media_inventory_sha256", "generated_at", "updated_at"})
_NOCOOKIE_EMBED_RE = re.compile(
    r"(?:(?:https?:)?//)(?:www\.)?youtube-nocookie\.com/embed/([A-Za-z0-9_-]+)", re.IGNORECASE
)


def _canonical(value: Any, *, exclude_volatile: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item, exclude_volatile=exclude_volatile)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not (exclude_volatile and str(key) in _VOLATILE_FIELDS)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item, exclude_volatile=exclude_volatile) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_event_media_inventory_digest(payload: Mapping[str, Any]) -> str:
    """Hash stable inventory facts while excluding only self/clock fields."""

    encoded = json.dumps(
        _canonical(payload, exclude_volatile=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _safe_route_path_segments(path: str) -> tuple[str, ...] | None:
    """Reject ambiguous route paths before matching the configured root."""

    if not path.startswith("/"):
        return None
    raw_segments = path.split("/")
    if raw_segments[0] != "":
        return None
    if raw_segments[-1] == "":
        raw_segments = raw_segments[:-1]
    segments: list[str] = []
    for segment in raw_segments[1:]:
        decoded = unquote(segment)
        if (
            not segment
            or segment != decoded
            or decoded in {".", ".."}
            or "/" in decoded
            or "\\" in decoded
        ):
            return None
        segments.append(segment)
    return tuple(segments)


def _source_route(url: Any, *, boundary: VirtualSiteBoundary) -> tuple[str, str] | None:
    """Validate and canonicalize one event route under an explicit site boundary."""

    if not isinstance(boundary, VirtualSiteBoundary):
        return None
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    host = (parsed.hostname or "").rstrip(".").casefold()
    if (
        parsed.scheme.casefold() != boundary.scheme
        or host != boundary.host
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    parts = _safe_route_path_segments(parsed.path)
    if parts is None or len(parts) != len(boundary.root_segments) + 2:
        return None
    if parts[: len(boundary.root_segments)] != boundary.root_segments:
        return None
    route_family, event_id = parts[-2:]
    route_family = route_family.lower()
    if route_family not in _EVENT_ROUTE_FAMILIES:
        return None
    return route_family, boundary.canonical_url(route_family, event_id)


def _category_for_route(*, route_family: str, title: str) -> str:
    if route_family == "session":
        if not title.strip():
            raise ValueError("session route lacks a title required for oral classification")
        return "oral" if "oral session" in title.casefold() else "session"
    return _ROUTE_CATEGORY[route_family]


def _is_poster_route(source_url: str, *, boundary: VirtualSiteBoundary) -> bool:
    route = _source_route(source_url, boundary=boundary)
    return route is not None and route[0] == "poster"


def _new_caption_candidate(*, video_id: str, source_url: str) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "source_event_urls": [source_url],
        "provenance": ["r5_official_event_page"],
        "caption_status": "pending",
        "caption": None,
        "audio_downloaded": False,
        "video_downloaded": False,
    }


def event_page_youtube_candidates(
    *, page_html: str, parent_url: str, boundary: VirtualSiteBoundary
) -> tuple[list[str], bool]:
    """Extract admitted direct YouTube IDs from one already-authenticated event page.

    It never follows a link or fetches a page.  ``youtube-nocookie`` embeds are
    canonicalized locally so the downstream caption queue still has only the
    standard ``youtube.com`` watch form.
    """

    if _source_route(parent_url, boundary=boundary) is None:
        raise ValueError("event media parser requires a configured event URL")
    video_ids: set[str] = set()
    unsupported_embed_seen = False
    for link in embedded_media_links(page_html, base_url=parent_url):
        # Receipt sanitization can treat ordinary text such as "token" in a
        # valid YouTube ID as a secret marker.  Validate the ID from the
        # original in-memory embed and emit only the canonical ID-derived URL.
        video_id = extract_video_id(link.url)
        if video_id is not None and url_host_allowed(link.url, DEFAULT_ALLOWED_HOSTS):
            video_ids.add(video_id)
        else:
            unsupported_embed_seen = True
    for video_id in _NOCOOKIE_EMBED_RE.findall(page_html):
        canonical = f"https://www.youtube.com/watch?v={video_id}"
        if extract_video_id(canonical) == video_id:
            video_ids.add(video_id)
    return sorted(video_ids), unsupported_embed_seen


def _refresh_inventory(inventory: dict[str, Any]) -> None:
    events = inventory.get("events")
    if not isinstance(events, list):
        raise ValueError("event media inventory has no event rows")
    categories: dict[str, list[Mapping[str, Any]]] = {category: [] for category in EVENT_CATEGORY_COUNTS}
    candidates_by_id: dict[str, dict[str, Any]] = {}
    existing_candidates = inventory.get("caption_candidates")
    if isinstance(existing_candidates, list):
        for candidate in existing_candidates:
            if isinstance(candidate, Mapping) and isinstance(candidate.get("video_id"), str):
                candidates_by_id[candidate["video_id"]] = copy.deepcopy(dict(candidate))
    for row in events:
        if not isinstance(row, Mapping):
            raise ValueError("event media inventory has an invalid event row")
        category = row.get("event_category")
        if category not in categories:
            raise ValueError("event media inventory contains an unknown event category")
        categories[category].append(row)
        if row.get("page_status") == "direct_youtube_recovered":
            for video_id in row.get("video_ids", []):
                candidate = candidates_by_id.get(video_id)
                if candidate is None:
                    candidate = _new_caption_candidate(video_id=video_id, source_url=str(row["source_url"]))
                    candidates_by_id[video_id] = candidate
                elif "r5_official_event_page" not in candidate.get("provenance", []):
                    candidate["provenance"] = sorted({*candidate.get("provenance", []), "r5_official_event_page"})
                urls = set(candidate.get("source_event_urls", []))
                urls.add(str(row["source_url"]))
                candidate["source_event_urls"] = sorted(urls)
    coverage: dict[str, dict[str, Any]] = {}
    for category, rows in categories.items():
        terminal_count = sum(row.get("page_status") in _TERMINAL_PAGE_STATUSES for row in rows)
        coverage[category] = {
            "event_route_count": len(rows),
            "terminal_route_count": terminal_count,
            "direct_youtube_route_count": sum(row.get("page_status") == "direct_youtube_recovered" for row in rows),
            "status": "complete" if terminal_count == len(rows) == EVENT_CATEGORY_COUNTS[category] else "incomplete",
        }
    inventory["caption_candidates"] = [candidates_by_id[video_id] for video_id in sorted(candidates_by_id)]
    inventory["counts"] = {
        "event_route_count": len(events),
        "event_categories": {category: len(categories[category]) for category in EVENT_CATEGORY_COUNTS},
        "oral_session_route_count": len(categories["oral"]),
        "terminal_event_route_count": sum(
            row.get("page_status") in _TERMINAL_PAGE_STATUSES for row in events if isinstance(row, Mapping)
        ),
        "direct_youtube_route_count": sum(
            row.get("page_status") == "direct_youtube_recovered" for row in events if isinstance(row, Mapping)
        ),
        "r4_reused_poster_route_count": sum(
            row.get("outcome_origin") == "r4_registration_media"
            for row in events
            if isinstance(row, Mapping)
        ),
        "caption_candidate_count": len(inventory["caption_candidates"]),
    }
    inventory["coverage"] = coverage
    inventory["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(inventory)


def build_event_media_inventory(
    *,
    source_records: Sequence[Mapping[str, Any]],
    source_inventory_sha256: str,
    boundary: VirtualSiteBoundary,
) -> dict[str, Any]:
    """Build the exact, source-backed r5 event route inventory without I/O."""

    if not _is_sha256(source_inventory_sha256):
        raise ValueError("event media inventory requires a source inventory SHA-256")
    events_by_url: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    for raw_record in source_records:
        if not isinstance(raw_record, Mapping) or raw_record.get("family_id") != "program-schedule":
            continue
        source_record_id = raw_record.get("source_record_id")
        if not isinstance(source_record_id, str) or not source_record_id or source_record_id in source_ids:
            if isinstance(source_record_id, str) and source_record_id in source_ids:
                raise ValueError("duplicate source record identifier")
            continue
        source_ids.add(source_record_id)
        title = raw_record.get("title")
        title = title.strip() if isinstance(title, str) else ""
        aliases = raw_record.get("aliases")
        if not isinstance(aliases, list):
            continue
        for alias in aliases:
            route = _source_route(alias, boundary=boundary)
            if route is None:
                continue
            route_family, source_url = route
            event_category = _category_for_route(route_family=route_family, title=title)
            event = {
                "source_record_id": source_record_id,
                "source_url": source_url,
                "route_family": route_family,
                "event_category": event_category,
                "title": title,
                "page_status": _PENDING_PAGE_STATUS,
                "outcome_origin": None,
                "video_ids": [],
                "audio_downloaded": False,
                "video_downloaded": False,
            }
            prior = events_by_url.get(source_url)
            if prior is not None and prior != event:
                raise ValueError("duplicate event route has conflicting source evidence")
            if prior is not None:
                raise ValueError("duplicate event route")
            events_by_url[source_url] = event
    inventory: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_event_media_inventory",
        "run_id": R5_EVENT_MEDIA_RUN_ID,
        "source_lineage": {"source_inventory_sha256": source_inventory_sha256},
        "events": [events_by_url[url] for url in sorted(events_by_url)],
        "caption_candidates": [],
        "boundaries": {
            "audio_downloaded": False,
            "video_downloaded": False,
            "asr_dispatched": False,
            "ocr_dispatched": False,
            "provider_calls_made": 0,
        },
    }
    _refresh_inventory(inventory)
    validate_event_media_inventory(inventory, boundary=boundary)
    return inventory


def apply_event_page_outcome(
    inventory: Mapping[str, Any], *, boundary: VirtualSiteBoundary, source_url: str, status: str, video_ids: Sequence[str] = (), auth_status: str | None = None, http_status: int | None = None,
    outcome_origin: str = "r5_authenticated_event_page",
) -> dict[str, Any]:
    """Return a new inventory with one validated terminal page outcome."""

    return apply_event_page_outcomes(
        inventory,
        boundary=boundary,
        outcomes=[
            {
                "source_url": source_url,
                "status": status,
                "video_ids": list(video_ids),
                "auth_status": auth_status,
                "http_status": http_status,
                "outcome_origin": outcome_origin,
            }
        ],
    )


def apply_event_page_outcomes(
    inventory: Mapping[str, Any], *, boundary: VirtualSiteBoundary, outcomes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply a non-overlapping batch of terminal page facts in one copy.

    The full CVPR virtual-program census has thousands of poster routes.  This
    batch primitive prevents a quadratic cascade of defensive inventory copies
    while preserving the same validation as the single-outcome wrapper.
    """

    validate_event_media_inventory(inventory, boundary=boundary)
    result = copy.deepcopy(dict(inventory))
    events = result.get("events")
    if not isinstance(events, list):
        raise ValueError("event media inventory has no event rows")
    by_url = {row.get("source_url"): row for row in events if isinstance(row, dict)}
    if len(by_url) != len(events):
        raise ValueError("event media inventory has duplicate route rows")
    seen_urls: set[str] = set()
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            raise ValueError("event page outcome must be an object")
        source_url = outcome.get("source_url")
        status = outcome.get("status")
        outcome_origin = outcome.get("outcome_origin", "r5_authenticated_event_page")
        raw_video_ids = outcome.get("video_ids", ())
        auth_status = outcome.get("auth_status")
        http_status = outcome.get("http_status")
        if not isinstance(source_url, str) or source_url in seen_urls or source_url not in by_url:
            raise ValueError("event page outcome does not match exactly one pinned route")
        if status not in _TERMINAL_PAGE_STATUSES:
            raise ValueError("event page outcome must be terminal")
        if outcome_origin not in {"r4_registration_media", "r5_authenticated_event_page"}:
            raise ValueError("event page outcome has an invalid origin")
        if not isinstance(raw_video_ids, Sequence) or isinstance(raw_video_ids, (str, bytes)):
            raise ValueError("event page outcome video IDs are invalid")
        normalized_ids: list[str] = []
        for raw_id in raw_video_ids:
            video_id = extract_video_id(f"https://www.youtube.com/watch?v={raw_id}")
            if video_id is None or video_id in normalized_ids:
                raise ValueError("event page outcome has an invalid or duplicate YouTube video id")
            normalized_ids.append(video_id)
        if (status == "direct_youtube_recovered") != bool(normalized_ids):
            raise ValueError("direct YouTube outcome and video IDs disagree")
        if auth_status is not None and auth_status not in {
            "authenticated",
            "registration_auth_required",
            "registered_access_denied",
        }:
            raise ValueError("event page outcome has an invalid authentication status")
        if http_status is not None and (
            not isinstance(http_status, int) or http_status < 100 or http_status > 599
        ):
            raise ValueError("event page outcome has an invalid HTTP status")
        target = by_url[source_url]
        target["page_status"] = status
        target["outcome_origin"] = outcome_origin
        target["video_ids"] = normalized_ids
        if auth_status is not None:
            target["auth_status"] = auth_status
        if http_status is not None:
            target["http_status"] = http_status
        seen_urls.add(source_url)
    _refresh_inventory(result)
    validate_event_media_inventory(result, boundary=boundary)
    return result


def derive_r4_poster_reuse_outcomes(
    *,
    r4_requests: Sequence[Mapping[str, Any]],
    r4_media_records: Sequence[Mapping[str, Any]],
    boundary: VirtualSiteBoundary,
) -> list[dict[str, Any]]:
    """Derive the exact immutable r4 facts eligible for r5 poster reuse.

    This shared projection is the sole translation between the r4 overlay and
    r5 rows.  Both the initial reuse bridge and the final stage validator use
    it, so a matching digest/count cannot substitute a different terminal
    status, auth fact, HTTP fact, URL, or direct YouTube ID.
    """
    media_by_request: dict[str, list[str]] = {}
    for record in r4_media_records:
        if not isinstance(record, Mapping):
            raise ValueError("r4 media record is invalid")
        request_key = record.get("parent_request_key")
        raw_video_id = record.get("video_id")
        video_id = (
            extract_video_id(f"https://www.youtube.com/watch?v={raw_video_id}")
            if isinstance(raw_video_id, str)
            else None
        )
        if not isinstance(request_key, str) or not request_key or video_id is None:
            raise ValueError("r4 media record lacks valid request provenance")
        media_by_request.setdefault(request_key, []).append(video_id)

    outcomes: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for request in r4_requests:
        if not isinstance(request, Mapping):
            raise ValueError("r4 request is invalid")
        source_url = request.get("source_url")
        if not isinstance(source_url, str) or not _is_poster_route(source_url, boundary=boundary):
            continue
        request_key = request.get("request_key")
        status = request.get("result_status")
        if not isinstance(request_key, str) or not request_key or source_url in seen_urls:
            raise ValueError("r4 poster request provenance is invalid")
        if status not in _TERMINAL_PAGE_STATUSES:
            raise ValueError("r4 poster request is not terminal")
        video_ids = sorted(set(media_by_request.get(request_key, [])))
        if (status == "direct_youtube_recovered") != bool(video_ids):
            raise ValueError("r4 poster request/media outcomes disagree")
        seen_urls.add(source_url)
        outcomes.append(
            {
                "source_url": source_url,
                "status": status,
                "video_ids": video_ids,
                "auth_status": request.get("auth_status") if isinstance(request.get("auth_status"), str) else None,
                "http_status": request.get("http_status") if isinstance(request.get("http_status"), int) else None,
                "outcome_origin": "r4_registration_media",
            }
        )
    if len(seen_urls) != R4_REUSED_POSTER_ROUTE_COUNT:
        raise ValueError("r4 poster reuse count does not match the immutable r4 scope")
    return sorted(outcomes, key=lambda outcome: str(outcome["source_url"]))


def validate_r4_poster_reuse(
    inventory: Mapping[str, Any],
    *,
    r4_requests: Sequence[Mapping[str, Any]],
    r4_media_records: Sequence[Mapping[str, Any]],
    boundary: VirtualSiteBoundary,
) -> None:
    """Require every r4-origin r5 row to equal its immutable overlay fact."""

    validate_event_media_inventory(inventory, boundary=boundary)
    expected_by_url = {
        str(outcome["source_url"]): outcome
        for outcome in derive_r4_poster_reuse_outcomes(
            r4_requests=r4_requests,
            r4_media_records=r4_media_records,
            boundary=boundary,
        )
    }
    events = inventory.get("events")
    if not isinstance(events, list):
        raise ValueError("event media inventory has no event rows")
    actual_by_url = {
        str(row.get("source_url")): row
        for row in events
        if isinstance(row, Mapping) and row.get("outcome_origin") == "r4_registration_media"
    }
    if set(actual_by_url) != set(expected_by_url):
        raise ValueError("event media inventory r4 poster URLs do not match the immutable overlay")
    for source_url, expected in expected_by_url.items():
        actual = actual_by_url[source_url]
        observed = {
            "source_url": actual.get("source_url"),
            "status": actual.get("page_status"),
            "video_ids": actual.get("video_ids"),
            "auth_status": actual.get("auth_status"),
            "http_status": actual.get("http_status"),
            "outcome_origin": actual.get("outcome_origin"),
        }
        if observed != expected:
            raise ValueError("event media inventory r4 poster evidence does not match the immutable overlay")


def apply_r4_poster_outcomes(
    inventory: Mapping[str, Any],
    *,
    r4_requests: Sequence[Mapping[str, Any]],
    r4_media_records: Sequence[Mapping[str, Any]],
    r4_overlay_sha256: str,
    boundary: VirtualSiteBoundary,
) -> dict[str, Any]:
    """Reuse r4's immutable poster-page facts without reopening those pages.

    r4's result is trustworthy only for its fixed 2,565-page poster subset.
    This bridge records each of those terminal outcomes locally, preserves the
    r4 ledger digest in lineage, and leaves the remaining 2,458 poster pages
    pending for r5's own authenticated scan.
    """

    validate_event_media_inventory(inventory, boundary=boundary)
    if not _is_sha256(r4_overlay_sha256):
        raise ValueError("r4 poster reuse requires an overlay SHA-256")
    outcomes = derive_r4_poster_reuse_outcomes(
        r4_requests=r4_requests,
        r4_media_records=r4_media_records,
        boundary=boundary,
    )
    result = apply_event_page_outcomes(inventory, boundary=boundary, outcomes=outcomes)
    lineage = result.get("source_lineage")
    if not isinstance(lineage, dict):
        raise ValueError("event media inventory source lineage is invalid")
    lineage["r4_registration_media_overlay_sha256"] = r4_overlay_sha256
    _refresh_inventory(result)
    validate_event_media_inventory(result, boundary=boundary)
    return result


def apply_caption_outcome(
    inventory: Mapping[str, Any],
    *,
    boundary: VirtualSiteBoundary,
    video_id: str,
    caption_status: str,
    caption: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a terminal caption-only outcome to one deduplicated video ID."""

    return apply_caption_outcomes(
        inventory,
        boundary=boundary,
        outcomes=[
            {
                "video_id": video_id,
                "caption_status": caption_status,
                "caption": caption,
            }
        ],
    )


def apply_caption_outcomes(
    inventory: Mapping[str, Any], *, boundary: VirtualSiteBoundary, outcomes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Attach a non-overlapping batch of terminal caption outcomes.

    Checkpoints can contain thousands of caption records.  Updating them in a
    single defensive copy keeps resume and periodic persistence linear in the
    inventory size while retaining the single-outcome contract above.
    """

    validate_event_media_inventory(inventory, boundary=boundary)
    result = copy.deepcopy(dict(inventory))
    candidates = result.get("caption_candidates")
    if not isinstance(candidates, list):
        raise ValueError("event media inventory caption queue is invalid")
    by_video_id = {
        row.get("video_id"): row for row in candidates if isinstance(row, dict) and isinstance(row.get("video_id"), str)
    }
    if len(by_video_id) != len(candidates):
        raise ValueError("event media inventory caption queue is invalid")
    seen_video_ids: set[str] = set()
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            raise ValueError("caption outcome must be an object")
        video_id = outcome.get("video_id")
        caption_status = outcome.get("caption_status")
        caption = outcome.get("caption")
        if (
            not isinstance(video_id, str)
            or video_id not in by_video_id
            or video_id in seen_video_ids
            or caption_status not in {"youtube_caption_available", "youtube_caption_unavailable"}
            or (caption_status == "youtube_caption_available") != isinstance(caption, Mapping)
        ):
            raise ValueError("caption outcome does not match exactly one candidate")
        target = by_video_id[video_id]
        target["caption_status"] = caption_status
        target["caption"] = copy.deepcopy(dict(caption)) if isinstance(caption, Mapping) else None
        seen_video_ids.add(video_id)
    _refresh_inventory(result)
    validate_event_media_inventory(result, boundary=boundary)
    return result


def resume_event_media_inventory(
    *,
    fresh_inventory: Mapping[str, Any],
    prior_inventory: Mapping[str, Any],
    reusable_caption_outcomes: Sequence[Mapping[str, Any]],
    boundary: VirtualSiteBoundary,
) -> dict[str, Any]:
    """Reuse only compatible terminal r5 facts from a private checkpoint.

    The fresh r3/r4-derived inventory is authoritative.  A checkpoint can only
    contribute terminal r5 page outcomes for identical source rows and caption
    outcomes whose local sidecars were separately verified by the runner.
    Authentication failures are intentionally never reusable terminal facts.
    """

    validate_event_media_inventory(fresh_inventory, boundary=boundary)
    validate_event_media_inventory(prior_inventory, boundary=boundary)
    fresh_lineage = fresh_inventory.get("source_lineage")
    prior_lineage = prior_inventory.get("source_lineage")
    if fresh_lineage != prior_lineage:
        raise ValueError("event media checkpoint lineage is incompatible")

    fresh_events = fresh_inventory.get("events")
    prior_events = prior_inventory.get("events")
    if not isinstance(fresh_events, list) or not isinstance(prior_events, list):
        raise ValueError("event media checkpoint events are invalid")
    fresh_by_url = {row.get("source_url"): row for row in fresh_events if isinstance(row, Mapping)}
    prior_by_url = {row.get("source_url"): row for row in prior_events if isinstance(row, Mapping)}
    if len(fresh_by_url) != len(fresh_events) or set(fresh_by_url) != set(prior_by_url):
        raise ValueError("event media checkpoint route census is incompatible")

    identity_fields = ("source_record_id", "source_url", "route_family", "event_category", "title")
    r5_outcomes: list[dict[str, Any]] = []
    for source_url, fresh_row in fresh_by_url.items():
        prior_row = prior_by_url[source_url]
        if any(prior_row.get(field) != fresh_row.get(field) for field in identity_fields):
            raise ValueError("event media checkpoint source identity is incompatible")
        if fresh_row.get("outcome_origin") == "r4_registration_media":
            if prior_row != fresh_row:
                raise ValueError("event media checkpoint changed immutable r4 evidence")
            continue
        if prior_row.get("page_status") == _PENDING_PAGE_STATUS:
            if prior_row != fresh_row:
                raise ValueError("event media checkpoint changed pending source evidence")
            continue
        if (
            prior_row.get("outcome_origin") != "r5_authenticated_event_page"
            or prior_row.get("page_status") not in _REUSABLE_R5_PAGE_STATUSES
        ):
            raise ValueError("event media checkpoint has a non-reusable page outcome")
        r5_outcomes.append(
            {
                "source_url": source_url,
                "status": prior_row.get("page_status"),
                "video_ids": prior_row.get("video_ids"),
                "auth_status": prior_row.get("auth_status"),
                "http_status": prior_row.get("http_status"),
                "outcome_origin": "r5_authenticated_event_page",
            }
        )
    resumed = apply_event_page_outcomes(fresh_inventory, boundary=boundary, outcomes=r5_outcomes)
    return apply_caption_outcomes(resumed, boundary=boundary, outcomes=reusable_caption_outcomes)


def merge_caption_candidates(
    inventory: Mapping[str, Any], *, r4_media_records: Sequence[Mapping[str, Any]], boundary: VirtualSiteBoundary
) -> dict[str, Any]:
    """Deduplicate immutable r4 IDs into r5's caption-only queue."""

    validate_event_media_inventory(inventory, boundary=boundary)
    result = copy.deepcopy(dict(inventory))
    candidates = result.get("caption_candidates")
    if not isinstance(candidates, list):
        raise ValueError("event media inventory has no caption candidates")
    by_video_id = {str(row.get("video_id")): row for row in candidates if isinstance(row, dict)}
    if len(by_video_id) != len(candidates):
        raise ValueError("event media inventory caption candidates are invalid")
    for record in r4_media_records:
        if not isinstance(record, Mapping):
            raise ValueError("r4 media record is invalid")
        raw_video_id = record.get("video_id")
        video_id = extract_video_id(f"https://www.youtube.com/watch?v={raw_video_id}") if isinstance(raw_video_id, str) else None
        if video_id is None:
            raise ValueError("r4 media record has an invalid YouTube video id")
        candidate = by_video_id.get(video_id)
        if candidate is None:
            candidate = _new_caption_candidate(video_id=video_id, source_url="")
            candidate["source_event_urls"] = []
            by_video_id[video_id] = candidate
        candidate["provenance"] = sorted({*candidate.get("provenance", []), "r4_registration_media"})
        request_key = record.get("parent_request_key")
        if not isinstance(request_key, str) or not request_key:
            raise ValueError("r4 media record lacks a parent request key")
        candidate["r4_parent_request_keys"] = sorted({*candidate.get("r4_parent_request_keys", []), request_key})
    result["caption_candidates"] = [by_video_id[video_id] for video_id in sorted(by_video_id)]
    _refresh_inventory(result)
    validate_event_media_inventory(result, boundary=boundary)
    return result


def validate_event_media_inventory(
    inventory: Mapping[str, Any], *, boundary: VirtualSiteBoundary, require_terminal: bool = False, require_caption_terminal: bool = False
) -> None:
    """Fail closed on category drift, fabricated media, or a misleading claim."""

    if not isinstance(inventory, Mapping):
        raise ValueError("event media inventory must be an object")
    if inventory.get("schema_version") != 1 or inventory.get("run_id") != R5_EVENT_MEDIA_RUN_ID:
        raise ValueError("event media inventory schema or run identity is invalid")
    lineage = inventory.get("source_lineage")
    if not isinstance(lineage, Mapping) or not _is_sha256(lineage.get("source_inventory_sha256")):
        raise ValueError("event media inventory source lineage is invalid")
    if (
        "r4_registration_media_overlay_sha256" in lineage
        and not _is_sha256(lineage.get("r4_registration_media_overlay_sha256"))
    ):
        raise ValueError("event media inventory r4 overlay lineage is invalid")
    if inventory.get("event_media_inventory_sha256") != canonical_event_media_inventory_digest(inventory):
        raise ValueError("event media inventory digest is invalid")
    boundaries = inventory.get("boundaries")
    if not isinstance(boundaries, Mapping) or any(boundaries.get(field) is not False for field in ("audio_downloaded", "video_downloaded", "asr_dispatched", "ocr_dispatched")) or boundaries.get("provider_calls_made") != 0:
        raise ValueError("event media inventory violates its caption-only media boundary")
    events = inventory.get("events")
    if not isinstance(events, list):
        raise ValueError("event media inventory events are invalid")
    urls: set[str] = set()
    source_ids: set[str] = set()
    categories: dict[str, list[Mapping[str, Any]]] = {category: [] for category in EVENT_CATEGORY_COUNTS}
    direct_ids: set[str] = set()
    for row in events:
        if not isinstance(row, Mapping):
            raise ValueError("event media inventory has a non-object event row")
        source_url = row.get("source_url")
        route = _source_route(source_url, boundary=boundary)
        source_record_id = row.get("source_record_id")
        category = row.get("event_category")
        status = row.get("page_status")
        outcome_origin = row.get("outcome_origin")
        video_ids = row.get("video_ids")
        if (
            route is None
            or not isinstance(source_record_id, str)
            or not source_record_id
            or source_url in urls
            or source_record_id in source_ids
            or category not in categories
            or not isinstance(video_ids, list)
            or row.get("audio_downloaded") is not False
            or row.get("video_downloaded") is not False
        ):
            raise ValueError("event media inventory event provenance or media boundary is invalid")
        route_family, _canonical_url = route
        title = row.get("title") if isinstance(row.get("title"), str) else ""
        if row.get("route_family") != route_family or _category_for_route(route_family=route_family, title=title) != category:
            raise ValueError("event media inventory event category is invalid")
        if status != _PENDING_PAGE_STATUS and status not in _TERMINAL_PAGE_STATUSES:
            raise ValueError("event media inventory has an invalid page outcome")
        if (
            (status == _PENDING_PAGE_STATUS and outcome_origin is not None)
            or (
                status in _TERMINAL_PAGE_STATUSES
                and outcome_origin not in {"r4_registration_media", "r5_authenticated_event_page"}
            )
            or (outcome_origin == "r4_registration_media" and route_family != "poster")
        ):
            raise ValueError("event media inventory event outcome origin is invalid")
        normalized_ids = [extract_video_id(f"https://www.youtube.com/watch?v={value}") if isinstance(value, str) else None for value in video_ids]
        if any(value is None for value in normalized_ids) or len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("event media inventory has invalid YouTube video IDs")
        if (status == "direct_youtube_recovered") != bool(video_ids):
            raise ValueError("event media inventory direct YouTube outcome disagrees with video IDs")
        urls.add(source_url)
        source_ids.add(source_record_id)
        categories[category].append(row)
        if status == "direct_youtube_recovered":
            direct_ids.update(str(value) for value in video_ids)
    if {category: len(rows) for category, rows in categories.items()} != EVENT_CATEGORY_COUNTS:
        raise ValueError("event media inventory category census drifted")
    coverage = inventory.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("event media inventory coverage summary is invalid")
    for category, rows in categories.items():
        summary = coverage.get(category)
        terminal_count = sum(row.get("page_status") in _TERMINAL_PAGE_STATUSES for row in rows)
        expected = {
            "event_route_count": len(rows),
            "terminal_route_count": terminal_count,
            "direct_youtube_route_count": sum(row.get("page_status") == "direct_youtube_recovered" for row in rows),
            "status": "complete" if terminal_count == len(rows) else "incomplete",
        }
        if summary != expected:
            raise ValueError("event media inventory category coverage summary is invalid")
        if require_terminal and summary["status"] != "complete":
            raise ValueError("event media inventory category is not terminally accounted")
    candidates = inventory.get("caption_candidates")
    if not isinstance(candidates, list):
        raise ValueError("event media inventory caption queue is invalid")
    candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("event media inventory caption candidate is invalid")
        video_id = candidate.get("video_id")
        expected_id = extract_video_id(f"https://www.youtube.com/watch?v={video_id}") if isinstance(video_id, str) else None
        if (
            expected_id is None
            or expected_id != video_id
            or video_id in candidate_ids
            or candidate.get("video_url") != f"https://www.youtube.com/watch?v={video_id}"
            or candidate.get("audio_downloaded") is not False
            or candidate.get("video_downloaded") is not False
            or candidate.get("caption_status") not in _CAPTION_STATUSES
            or not isinstance(candidate.get("source_event_urls"), list)
            or not isinstance(candidate.get("provenance"), list)
        ):
            raise ValueError("event media inventory caption candidate violates its media boundary")
        caption = candidate.get("caption")
        if candidate.get("caption_status") == "youtube_caption_available":
            required_caption_fields = {"local_path", "mime", "byte_count", "sha256", "transcript_source", "language"}
            if (
                not isinstance(caption, Mapping)
                or set(caption) != required_caption_fields
                or not isinstance(caption.get("local_path"), str)
                or not caption["local_path"].startswith(str(EVENT_MEDIA_CAPTIONS_RELATIVE_PATH))
                or not caption_sidecar_matches_video_id(caption["local_path"], video_id)
                or caption.get("mime") not in {"text/vtt", "application/x-subrip"}
                or not isinstance(caption.get("byte_count"), int)
                or caption["byte_count"] <= 0
                or not _is_sha256(caption.get("sha256"))
                or caption.get("transcript_source") not in {"youtube_caption_manual", "youtube_caption_auto"}
                or not isinstance(caption.get("language"), str)
            ):
                raise ValueError("event media inventory available caption is invalid")
        elif caption is not None:
            raise ValueError("event media inventory unavailable caption has metadata")
        if require_caption_terminal and candidate.get("caption_status") == "pending":
            raise ValueError("event media inventory caption candidate is not terminal")
        candidate_ids.add(video_id)
    if not direct_ids.issubset(candidate_ids):
        raise ValueError("event media inventory is missing a direct-event caption candidate")
    counts = inventory.get("counts")
    expected_counts = {
        "event_route_count": len(events),
        "event_categories": {category: len(categories[category]) for category in EVENT_CATEGORY_COUNTS},
        "oral_session_route_count": len(categories["oral"]),
        "terminal_event_route_count": sum(row.get("page_status") in _TERMINAL_PAGE_STATUSES for row in events),
        "direct_youtube_route_count": sum(row.get("page_status") == "direct_youtube_recovered" for row in events),
        "r4_reused_poster_route_count": sum(
            row.get("outcome_origin") == "r4_registration_media" for row in events
        ),
        "caption_candidate_count": len(candidates),
    }
    if counts != expected_counts:
        raise ValueError("event media inventory counts are invalid")
    if "r4_registration_media_overlay_sha256" in lineage and (
        expected_counts["r4_reused_poster_route_count"] != R4_REUSED_POSTER_ROUTE_COUNT
    ):
        raise ValueError("event media inventory r4 poster reuse count is invalid")


# --------------------------------------------------------------------------
# Stage 4 lane owner: event media (r5)
# --------------------------------------------------------------------------

from collections.abc import Callable  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from conference_pipeline.provider_receipts import sanitize_for_receipt as _sanitize_for_receipt  # noqa: E402

EVENT_MEDIA_LANE_ID = "event-media"
EVENT_MEDIA_PROFILE = "event-media-coverage"
_PAGE_CHECKPOINT_BATCH = 50
_CAPTION_CHECKPOINT_BATCH = 25


@dataclass(frozen=True)
class EventMediaLaneSettings:
    """A conference's event-media lane identity and pinned predecessors.

    ``receipt_type`` names the lane's Stage 4 receipt; ``owner_function`` is
    recorded as the owner's entry point; the source inventory and r4 overlay
    are the pinned predecessors the census is built from.
    """

    receipt_type: str
    owner_function: str
    checkpoint_path: Path
    source_inventory_path: Path
    r4_overlay_path: Path
    pinned_source_run_id: str
    pinned_inventory_sha256: str
    caption_worker_count: int = 4
    run_id: str = R5_EVENT_MEDIA_RUN_ID
    inventory_path: Path = EVENT_MEDIA_INVENTORY_RELATIVE_PATH
    captions_path: Path = EVENT_MEDIA_CAPTIONS_RELATIVE_PATH
    pin_name: str = "PINNED_CORPUS_SOURCE_INVENTORY_SHA256"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_media_inventory_inputs(
    workspace: Path,
    *,
    boundary: VirtualSiteBoundary,
    settings: EventMediaLaneSettings,
    validate_r4_overlay: Callable[..., Any],
) -> dict[str, Any]:
    """Load the fixed r3 schedule surface and immutable r4 link handoff."""

    pinned = settings.pinned_inventory_sha256
    source_payload = json.loads((workspace / settings.source_inventory_path).read_text(encoding="utf-8"))
    r4_payload = json.loads((workspace / settings.r4_overlay_path).read_text(encoding="utf-8"))
    if not isinstance(source_payload, Mapping) or not isinstance(r4_payload, Mapping):
        raise ValueError("event media predecessors must be JSON objects")
    if source_payload.get("run_id") != settings.pinned_source_run_id or source_payload.get("status") != "prepared":
        raise ValueError("event media source inventory is not the pinned r3 schedule")
    observed_inventory_sha256 = source_payload.get("inventory_sha256")
    if observed_inventory_sha256 != pinned:
        raise ValueError(
            "event media source inventory digest does not match the pinned "
            f"{settings.pin_name} constant "
            f"(expected={pinned!r}, "
            f"observed={observed_inventory_sha256!r}). "
            "storage/source-discovery/source_discovery_inventory.json's "
            "inventory_sha256 changed, most likely because "
            "scripts/rederive_source_discovery_authors.py (or another "
            "legitimate rewrite) regenerated it. Remedy: verify the "
            "regeneration was intended, then update "
            f"{settings.pin_name} in the orchestrator that pins it to the new "
            "digest, in the same commit as the rederive -- do not weaken or "
            "remove this comparison."
        )
    records = source_payload.get("source_records")
    if not isinstance(records, list):
        raise ValueError("event media source inventory has no source records")
    validate_r4_overlay(r4_payload, boundary=boundary)
    r4_records = r4_payload.get("official_event_media_records")
    r4_requests = r4_payload.get("official_event_requests")
    r4_overlay_sha256 = r4_payload.get("overlay_sha256")
    if (
        not isinstance(r4_records, list)
        or not isinstance(r4_requests, list)
        or not isinstance(r4_overlay_sha256, str)
    ):
        raise ValueError("event media r4 overlay lacks reusable request evidence")
    inventory = build_event_media_inventory(
        source_records=records,
        source_inventory_sha256=pinned,
        boundary=boundary,
    )
    inventory = apply_r4_poster_outcomes(
        inventory,
        r4_requests=r4_requests,
        r4_media_records=r4_records,
        r4_overlay_sha256=r4_overlay_sha256,
        boundary=boundary,
    )
    return merge_caption_candidates(inventory, r4_media_records=r4_records, boundary=boundary)


def event_media_caption_record(
    *, workspace: Path, video_id: str, value: Mapping[str, Any] | None, captions_path: Path = EVENT_MEDIA_CAPTIONS_RELATIVE_PATH
) -> dict[str, Any] | None:
    """Turn a local caption primitive result into a contained receipt record."""

    if not isinstance(value, Mapping):
        return None
    raw_path = value.get("caption_path")
    if not isinstance(raw_path, (str, Path)):
        return None
    candidate = Path(raw_path)
    root = (workspace / captions_path).resolve()
    try:
        resolved = candidate.resolve()
        relative = resolved.relative_to(workspace.resolve()).as_posix()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    if (
        not resolved.is_file()
        or resolved.stat().st_size <= 0
        or not caption_sidecar_matches_video_id(resolved, video_id)
    ):
        return None
    mime = "text/vtt" if resolved.suffix.lower() == ".vtt" else "application/x-subrip" if resolved.suffix.lower() == ".srt" else None
    transcript_source = value.get("transcript_source")
    language = value.get("language")
    if (
        mime is None
        or transcript_source not in {"youtube_caption_manual", "youtube_caption_auto"}
        or not isinstance(language, str)
        or not language
    ):
        return None
    return {
        "local_path": relative,
        "mime": mime,
        "byte_count": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
        "transcript_source": transcript_source,
        "language": language,
    }


def event_media_authenticated_page_count(inventory: Mapping[str, Any]) -> int:
    """Count checkpointed r5 pages that prove an authenticated transport result.

    The private checkpoint deliberately retains only terminal page outcomes,
    never cookies or page bodies.  Those terminal rows are the durable evidence
    that a previous attempt performed its authenticated page work; a resumed
    caption-only attempt still re-runs the volatile session preflight below.
    """

    events = inventory.get("events")
    if not isinstance(events, list):
        raise ValueError("event media inventory events are invalid")
    return sum(
        row.get("outcome_origin") == "r5_authenticated_event_page"
        and row.get("auth_status") == "authenticated"
        for row in events
        if isinstance(row, Mapping)
    )


def verified_checkpoint_caption_outcomes(
    *, workspace: Path, checkpoint: Mapping[str, Any], caption_record: Callable[..., dict[str, Any] | None]
) -> list[dict[str, Any]]:
    """Return checkpoint caption facts whose local sidecars still match exactly.

    An unavailable caption is a terminal no-sidecar fact.  An available caption
    is reusable only when its confined local path, size, hash, format, language,
    and provenance regenerate the exact metadata held by the checkpoint.
    Invalidated sidecars become pending work rather than evidence of a video.
    """

    candidates = checkpoint.get("caption_candidates")
    if not isinstance(candidates, list):
        raise ValueError("event media checkpoint caption queue is invalid")
    outcomes: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("event media checkpoint caption candidate is invalid")
        video_id = candidate.get("video_id")
        caption_status = candidate.get("caption_status")
        if not isinstance(video_id, str):
            raise ValueError("event media checkpoint caption candidate lacks a video ID")
        if caption_status == "youtube_caption_unavailable":
            outcomes.append({"video_id": video_id, "caption_status": caption_status, "caption": None})
            continue
        if caption_status != "youtube_caption_available":
            continue
        caption = candidate.get("caption")
        if not isinstance(caption, Mapping) or not isinstance(caption.get("local_path"), str):
            continue
        verified = caption_record(
            workspace=workspace,
            video_id=video_id,
            value={
                "caption_path": workspace / str(caption["local_path"]),
                "transcript_source": caption.get("transcript_source"),
                "language": caption.get("language"),
            },
        )
        if verified == dict(caption):
            outcomes.append({"video_id": video_id, "caption_status": "youtube_caption_available", "caption": verified})
    return outcomes


def default_caption_downloader(video_id: str, destination: Path) -> Mapping[str, Any] | None:
    """Use the established subtitle primitive while making no media request."""

    import subprocess

    from conference_pipeline.acquisition.youtube_talks import HarvestedVideo, download_captions, resolve_yt_dlp_cmd

    video = HarvestedVideo(
        video_id=video_id,
        title="",
        url=f"https://www.youtube.com/watch?v={video_id}",
        source="official_event_page",
    )
    return download_captions(
        video,
        destination,
        runner=lambda command: subprocess.run(  # noqa: S603 - resolved project dependency and fixed arguments
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        ),
        yt_dlp_cmd=resolve_yt_dlp_cmd(),
        langs=("en",),
    )


def event_media_checks(
    *,
    inventory: Mapping[str, Any] | None,
    inventory_file_sha256: str | None,
    authenticated_request_count: int,
    resumed_authenticated_request_count: int,
    resume_mode: bool,
    preflight: Mapping[str, Any],
    inventory_path: Path = EVENT_MEDIA_INVENTORY_RELATIVE_PATH,
) -> dict[str, Any]:
    counts = inventory.get("counts") if isinstance(inventory, Mapping) else {}
    counts = counts if isinstance(counts, Mapping) else {}
    coverage = inventory.get("coverage") if isinstance(inventory, Mapping) else {}
    coverage = coverage if isinstance(coverage, Mapping) else {}
    complete = bool(coverage) and all(
        isinstance(value, Mapping) and value.get("status") == "complete" for value in coverage.values()
    )
    captions = inventory.get("caption_candidates") if isinstance(inventory, Mapping) else []
    captions = captions if isinstance(captions, list) else []
    caption_complete = all(
        isinstance(row, Mapping)
        and row.get("caption_status") in {"youtube_caption_available", "youtube_caption_unavailable"}
        and row.get("audio_downloaded") is False
        and row.get("video_downloaded") is False
        for row in captions
    )
    return {
        "stage_contract_profile": EVENT_MEDIA_PROFILE,
        "event_media_coverage_status": "pass" if complete and caption_complete else "blocked",
        "event_route_accounting_complete": complete,
        "caption_candidate_accounting_complete": caption_complete,
        "event_route_count": counts.get("event_route_count", 0),
        "event_category_counts": counts.get("event_categories", {}),
        "oral_session_route_count": counts.get("oral_session_route_count", 0),
        "terminal_event_route_count": counts.get("terminal_event_route_count", 0),
        "direct_youtube_route_count": counts.get("direct_youtube_route_count", 0),
        "r4_reused_poster_route_count": counts.get("r4_reused_poster_route_count", 0),
        "caption_candidate_count": counts.get("caption_candidate_count", 0),
        "event_media_inventory_path": str(inventory_path),
        "event_media_inventory_sha256": inventory_file_sha256,
        "event_media_inventory_canonical_sha256": inventory.get("event_media_inventory_sha256") if isinstance(inventory, Mapping) else None,
        "r4_registration_media_overlay_sha256": (
            inventory.get("source_lineage", {}).get("r4_registration_media_overlay_sha256")
            if isinstance(inventory, Mapping) and isinstance(inventory.get("source_lineage"), Mapping)
            else None
        ),
        "event_media_authenticated_request_count": authenticated_request_count,
        "event_media_resumed_authenticated_request_count": resumed_authenticated_request_count,
        "event_media_resume_mode": resume_mode,
        "event_media_session_preflight": _sanitize_for_receipt(dict(preflight)),
        "audio_downloaded": False,
        "video_downloaded": False,
        "asr_dispatched": False,
        "ocr_dispatched": False,
        "provider_calls_made": 0,
        "secrets_copied": False,
    }


def _event_owner_evidence(settings: EventMediaLaneSettings, checks: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "owner_function": settings.owner_function,
        "inventory_path": str(settings.inventory_path),
        "transport_created": checks.get("event_media_authenticated_request_count", 0) > 0,
        "authenticated_request_count": checks.get("event_media_authenticated_request_count", 0),
        "resumed_authenticated_request_count": checks.get("event_media_resumed_authenticated_request_count", 0),
        "caption_only": True,
    }


def event_media_stage_receipt(
    conventions: Any,
    *,
    workspace: Path,
    run_id: str,
    status: str,
    blockers: list[dict[str, Any]],
    checks: Mapping[str, Any],
    inventory: Mapping[str, Any] | None,
    settings: EventMediaLaneSettings,
    stage_contract: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a small sanitized r5 Stage-4 receipt and summary."""

    from conference_pipeline.production.download_corpus_stage import write_stage_receipt

    body = {
        "owner_execution_evidence": {conventions.owner_id: _event_owner_evidence(settings, checks)},
        "event_media_inventory": {
            "path": str(settings.inventory_path),
            "inventory_sha256": inventory.get("event_media_inventory_sha256") if isinstance(inventory, Mapping) else None,
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
        profile=EVENT_MEDIA_PROFILE,
        receipt_type=settings.receipt_type,
        immutable_attempts=False,
        merge_summary=False,
    )
    return {"status": status, "receipt_path": str(receipt_path), "receipt": receipt, "summary": summary}


def run_event_media_lane(
    conventions: Any,
    *,
    workspace: Path,
    run_id: str,
    settings: EventMediaLaneSettings,
    boundary_loader: Callable[[Path], VirtualSiteBoundary],
    inventory_inputs: Callable[..., dict[str, Any]],
    opener_factory: Callable[..., tuple[Callable[[str], Mapping[str, Any]] | None, Mapping[str, Any]]],
    caption_downloader: Callable[[str, Path], Mapping[str, Any] | None],
    caption_record: Callable[..., dict[str, Any] | None],
    checks_builder: Callable[..., dict[str, Any]],
    repository_root: Path,
    resume: bool = False,
    validate_run_id: Callable[[str], str] = lambda value: value,
) -> dict[str, Any]:
    """Run the fixed event census followed by a subtitle-only queue.

    This successor is deliberately independent of r4 transport scope.  It
    references r4 only as an immutable source of already-known YouTube IDs.
    ``conventions`` is the orchestrator's
    :class:`~conference_pipeline.production.download_corpus_stage.LedgerConventions`.
    """

    from conference_pipeline.production.download_corpus_stage import stage4_blocker, verify_lane_contract

    safe_run_id = validate_run_id(run_id)
    if safe_run_id != settings.run_id:
        raise ValueError("event-media coverage requires the fixed r5 run id")
    resolved_workspace = workspace.resolve()
    checkpoint_path = resolved_workspace / settings.checkpoint_path

    def receipt(**kwargs: Any) -> dict[str, Any]:
        return event_media_stage_receipt(
            conventions, workspace=resolved_workspace, run_id=safe_run_id, settings=settings, **kwargs
        )

    try:
        boundary = boundary_loader(resolved_workspace)
        inventory = inventory_inputs(resolved_workspace, boundary=boundary)
        validate_event_media_inventory(inventory, boundary=boundary)
        if resume:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if not isinstance(checkpoint, Mapping):
                raise ValueError("event media checkpoint is not an object")
            reusable_caption_outcomes = verified_checkpoint_caption_outcomes(
                workspace=resolved_workspace, checkpoint=checkpoint, caption_record=caption_record
            )
            inventory = resume_event_media_inventory(
                fresh_inventory=inventory,
                prior_inventory=checkpoint,
                reusable_caption_outcomes=reusable_caption_outcomes,
                boundary=boundary,
            )
            validate_event_media_inventory(inventory, boundary=boundary)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return receipt(
            status="blocked",
            blockers=[
                {
                    "code": "event_media_checkpoint_incompatible" if resume else "event_media_lineage_unavailable",
                    "reason": type(exc).__name__,
                }
            ],
            checks=checks_builder(
                inventory=None,
                inventory_file_sha256=None,
                authenticated_request_count=0,
                resumed_authenticated_request_count=0,
                resume_mode=resume,
                preflight={},
            ),
            inventory=None,
        )
    events = inventory["events"]
    pending_events = [row for row in events if row.get("page_status") == "pending"]
    frozen_urls = [str(row["source_url"]) for row in pending_events]
    resumed_authenticated_request_count = event_media_authenticated_page_count(inventory) if resume else 0
    authenticated_request_count = resumed_authenticated_request_count
    opener: Callable[[str], Mapping[str, Any]] | None = None
    preflight: Mapping[str, Any] = {
        "preflight_status": "not_required",
        "auth_status": "not_required",
        "secrets_copied": False,
    }

    def blocked(code: str) -> dict[str, Any]:
        return receipt(
            status="blocked",
            blockers=[{"code": code}],
            checks=checks_builder(
                inventory=inventory,
                inventory_file_sha256=None,
                authenticated_request_count=authenticated_request_count,
                resumed_authenticated_request_count=resumed_authenticated_request_count,
                resume_mode=resume,
                preflight=preflight,
            ),
            inventory=inventory,
        )

    # Every resume makes one fresh, bounded authenticated sentinel request.
    # Prefer a terminal r5 row so caption-only recovery never reopens the full
    # page fanout; if no such row exists yet, consume the sentinel's in-memory
    # response for the first pending page rather than issuing it twice.
    preflight_urls = frozen_urls
    sentinel_url: str | None = None
    sentinel_raw: Mapping[str, Any] | None = None
    if resume:
        terminal_r5_urls = [
            str(row["source_url"]) for row in events if row.get("outcome_origin") == "r5_authenticated_event_page"
        ]
        sentinel_url = terminal_r5_urls[0] if terminal_r5_urls else (frozen_urls[0] if frozen_urls else None)
        if sentinel_url is None:
            return blocked("event_media_resume_sentinel_unavailable")
        preflight_urls = sorted({*frozen_urls, sentinel_url})
    if pending_events or resume:
        try:
            opener, preflight = opener_factory(
                workspace=resolved_workspace,
                repository_root=repository_root,
                frozen_urls=preflight_urls,
                boundary=boundary,
                capture_page_body=True,
            )
        except Exception:
            opener, preflight = None, {
                "preflight_status": "blocked",
                "auth_status": "registration_auth_required",
                "secrets_copied": False,
            }
        if preflight.get("preflight_status") != "pass" or not callable(opener):
            return blocked("registration_auth_required")
        if resume:
            try:
                authenticated_request_count += 1
                assert opener is not None and sentinel_url is not None
                raw_sentinel = opener(sentinel_url)
            except Exception:
                raw_sentinel = {
                    "status": "unreachable_authenticated_media",
                    "auth_status": "registration_auth_required",
                }
            raw_status = raw_sentinel.get("status") if isinstance(raw_sentinel, Mapping) else None
            raw_auth_status = raw_sentinel.get("auth_status") if isinstance(raw_sentinel, Mapping) else None
            raw_http_status = raw_sentinel.get("http_status") if isinstance(raw_sentinel, Mapping) else None
            sentinel_authenticated = raw_status == "authenticated" and raw_auth_status == "authenticated"
            preflight = {
                **dict(preflight),
                "preflight_status": "pass" if sentinel_authenticated else "blocked",
                "auth_status": "authenticated" if sentinel_authenticated else "registration_auth_required",
                "resume_sentinel": {
                    "authenticated_request_count": 1,
                    "status": "authenticated" if sentinel_authenticated else "not_authenticated",
                    "auth_status": "authenticated" if sentinel_authenticated else "registration_auth_required",
                    "http_status": raw_http_status if isinstance(raw_http_status, int) else None,
                },
                "secrets_copied": False,
            }
            if not sentinel_authenticated:
                return blocked(
                    "registered_access_denied" if raw_status == "registered_access_denied" else "registration_auth_required"
                )
            if sentinel_url in frozen_urls:
                sentinel_raw = raw_sentinel
    page_outcomes: list[dict[str, Any]] = []

    def checkpoint_page_outcomes() -> None:
        nonlocal inventory, page_outcomes
        if not page_outcomes:
            return
        inventory = apply_event_page_outcomes(inventory, outcomes=page_outcomes, boundary=boundary)
        page_outcomes = []
        conventions.write_json(checkpoint_path, inventory)

    for event in pending_events:
        source_url = str(event["source_url"])
        if sentinel_raw is not None and source_url == sentinel_url:
            raw = sentinel_raw
            sentinel_raw = None
        else:
            try:
                authenticated_request_count += 1
                assert opener is not None
                raw = opener(source_url)
            except Exception:
                raw = {"status": "unreachable_authenticated_media", "auth_status": "registration_auth_required"}
        status = raw.get("status") if isinstance(raw, Mapping) else None
        auth_status = raw.get("auth_status") if isinstance(raw, Mapping) else None
        http_status = raw.get("http_status") if isinstance(raw, Mapping) else None
        if status in {"registration_auth_required", "registered_access_denied"}:
            checkpoint_page_outcomes()
            return blocked(str(status))
        if status == "authenticated":
            body = raw.get("body") if isinstance(raw, Mapping) else None
            content_type = raw.get("content_type") if isinstance(raw, Mapping) else None
            if not isinstance(body, bytes) or not isinstance(content_type, str) or not content_type.lower().startswith(("text/html", "application/xhtml+xml")):
                page_outcomes.append(
                    {
                        "source_url": source_url,
                        "status": "unreachable_authenticated_media",
                        "auth_status": "authenticated",
                        "http_status": http_status if isinstance(http_status, int) else None,
                    }
                )
                if len(page_outcomes) >= _PAGE_CHECKPOINT_BATCH:
                    checkpoint_page_outcomes()
                continue
            video_ids, unsupported_embed = event_page_youtube_candidates(
                page_html=body.decode("utf-8", errors="replace"),
                parent_url=source_url,
                boundary=boundary,
            )
            page_outcomes.append(
                {
                    "source_url": source_url,
                    "status": "direct_youtube_recovered" if video_ids else "unsupported_official_embed_host" if unsupported_embed else "no_admissible_official_youtube_link",
                    "video_ids": video_ids,
                    "auth_status": "authenticated",
                    "http_status": http_status if isinstance(http_status, int) else None,
                }
            )
        else:
            terminal_status = status if isinstance(status, str) and status in {"official_video_not_published", "unreachable_authenticated_media", "unsupported_official_embed_host", "no_admissible_official_youtube_link"} else "unreachable_authenticated_media"
            page_outcomes.append(
                {
                    "source_url": source_url,
                    "status": terminal_status,
                    "auth_status": auth_status if isinstance(auth_status, str) else None,
                    "http_status": http_status if isinstance(http_status, int) else None,
                }
            )
        if len(page_outcomes) >= _PAGE_CHECKPOINT_BATCH:
            checkpoint_page_outcomes()
    checkpoint_page_outcomes()
    validate_event_media_inventory(inventory, boundary=boundary, require_terminal=True)
    captions_root = resolved_workspace / settings.captions_path
    caption_outcomes: list[dict[str, Any]] = []

    def checkpoint_caption_outcomes() -> None:
        nonlocal inventory, caption_outcomes
        if not caption_outcomes:
            return
        inventory = apply_caption_outcomes(inventory, outcomes=caption_outcomes, boundary=boundary)
        caption_outcomes = []
        conventions.write_json(checkpoint_path, inventory)

    pending_video_ids = [
        str(candidate["video_id"])
        for candidate in inventory["caption_candidates"]
        if candidate.get("caption_status") == "pending"
    ]

    def download_one_caption(video_id: str) -> tuple[str, Mapping[str, Any] | None]:
        try:
            return video_id, caption_downloader(video_id, captions_root)
        except Exception:
            return video_id, None

    # Caption jobs are independent by validated YouTube ID and write only an
    # ID-scoped sidecar.  Keep concurrency deliberately small and checkpoint
    # completed jobs without waiting for a slow unrelated subtitle response.
    with ThreadPoolExecutor(max_workers=min(settings.caption_worker_count, max(1, len(pending_video_ids)))) as executor:
        caption_futures = [executor.submit(download_one_caption, video_id) for video_id in pending_video_ids]
        for future in as_completed(caption_futures):
            video_id, raw_caption = future.result()
            caption = caption_record(workspace=resolved_workspace, video_id=video_id, value=raw_caption)
            caption_outcomes.append(
                {
                    "video_id": video_id,
                    "caption_status": "youtube_caption_available" if caption is not None else "youtube_caption_unavailable",
                    "caption": caption,
                }
            )
            if len(caption_outcomes) >= _CAPTION_CHECKPOINT_BATCH:
                checkpoint_caption_outcomes()
    checkpoint_caption_outcomes()
    validate_event_media_inventory(
        inventory,
        boundary=boundary,
        require_terminal=True,
        require_caption_terminal=True,
    )
    inventory_path = resolved_workspace / settings.inventory_path
    try:
        prior_inventory_bytes = inventory_path.read_bytes()
    except OSError:
        prior_inventory_bytes = None
    conventions.write_json(inventory_path, inventory)
    checks = checks_builder(
        inventory=inventory,
        inventory_file_sha256=_file_sha256(inventory_path),
        authenticated_request_count=authenticated_request_count,
        resumed_authenticated_request_count=resumed_authenticated_request_count,
        resume_mode=resume,
        preflight=preflight,
    )
    contract = verify_lane_contract(
        conventions,
        workspace=resolved_workspace,
        run_id=safe_run_id,
        owner_evidence={
            "owner_function": settings.owner_function,
            "inventory_path": str(settings.inventory_path),
            "transport_created": authenticated_request_count > 0,
            "authenticated_request_count": authenticated_request_count,
            "resumed_authenticated_request_count": resumed_authenticated_request_count,
            "caption_only": True,
        },
        checks=checks,
        profile=EVENT_MEDIA_PROFILE,
        extra={
            "run_id": safe_run_id,
            "stage_id": conventions.stage_id,
            "status": "succeeded",
            "downloads_made": 0,
            "provider_calls_made": 0,
            "secrets_copied": False,
        },
    )
    blockers: list[dict[str, Any]] = []
    if contract["status"] != "pass":
        blockers.append(stage4_blocker("final_event_media_stage_contract_incomplete"))
        conventions.write_json(checkpoint_path, inventory)
        if prior_inventory_bytes is None:
            try:
                inventory_path.unlink()
            except FileNotFoundError:
                pass
        else:
            conventions.write_bytes(inventory_path, prior_inventory_bytes)
    else:
        try:
            checkpoint_path.unlink()
        except FileNotFoundError:
            pass
    return receipt(
        status="succeeded" if not blockers else "blocked",
        blockers=blockers,
        checks=checks,
        inventory=inventory,
        stage_contract=contract,
    )
