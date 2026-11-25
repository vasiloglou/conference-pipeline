"""Pure, receipt-bound admission for the CVPR 2026 r6 media overlay.

The r6 overlay intentionally starts from the immutable r5 producer inventory.
It does not discover, download, or process media.  Later owners may consume its
declared queue only after they have independently acquired local payloads.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from conference_pipeline.acquisition.event_media import (
    R5_EVENT_MEDIA_RUN_ID,
    canonical_event_media_inventory_digest,
)
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_ARTIFACT_PATHS,
    R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT,
    R6_PUBLIC_AUDIO_POLICY,
    R6ArtifactContract,
    R6ArtifactContractError,
    archive_r6_audio_checkpoint_relative_path,
    r6_audio_policy_digest,
    read_r6_audio_checkpoint_archive,
    read_r6_raw_audio_payload,
)
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_RUN_ID as R6_EVENT_MEDIA_RUN_ID,
)
from conference_pipeline.acquisition.youtube_audio_recovery import TERMINAL_STATES
from conference_pipeline.acquisition.youtube_talks import (
    DEFAULT_ALLOWED_HOSTS,
    extract_video_id,
    url_host_allowed,
)
from conference_pipeline.transcription_ocr import atomic_write_json

R6_RECOVERY_ROOT_RELATIVE_PATH = R6_ARTIFACT_PATHS["r6_root"]
R6_RECOVERY_INVENTORY_RELATIVE_PATH = R6_ARTIFACT_PATHS["recovery_inventory"]
_R6_VOLATILE_FIELDS = frozenset({"r6_recovery_inventory_sha256", "generated_at", "updated_at"})
_SOURCE_SCOPE_VOLATILE_FIELDS = frozenset({"source_scope_sha256"})
_HISTORICAL_CHECKPOINT_LINEAGE_KEY = "historical_checkpoint_lineage"
_HISTORICAL_CHECKPOINT_RECORD_COUNT = 1_043
# These fields are immutable admission facts, not worker hints.  Keep their
# vocabulary deliberately closed: r6 is a fixed overlay over the 2,002 r5
# caption-unavailable rows, so accepting an omitted/``None`` value would make
# a changed policy indistinguishable from a historical fixture.
R6_REQUIRED_RECOVERY_STATE = "asr_fallback_required"
R6_REQUIRED_SOURCE_CLASSIFICATION = "public_youtube_candidate"


def validate_r6_recovery_classification(row: Mapping[str, Any], *, context: str) -> None:
    """Require the exact immutable r6 admission classification on ``row``.

    This validator is intentionally used before an oracle is evaluated and
    before any owner output is admitted.  ``dict.get`` is safe here only
    because absence, nulls, and all non-exact values fail identically; it is
    never a defaulting mechanism.
    """

    if row.get("state") != R6_REQUIRED_RECOVERY_STATE:
        raise ValueError(f"{context} state classification is invalid")
    if row.get("source_classification") != R6_REQUIRED_SOURCE_CLASSIFICATION:
        raise ValueError(f"{context} source classification is invalid")


@dataclass(frozen=True)
class R6SourceRouteScope:
    """The one immutable official-page observation population for r6.

    The source route scope intentionally comes from the r5-derived recovery
    records rather than from a crawler, local storage scan, or caller-supplied
    URL list.  It is the control-plane counterpart to the 2,002 public-audio
    fallback candidates.
    """

    routes: tuple[dict[str, Any], ...]
    source_scope_count: int
    source_scope_sha256: str


@dataclass(frozen=True)
class R5LineagePins:
    """Exact r5 producer digests required before r6 queue construction."""

    receipt_sha256: str
    inventory_sha256: str
    inventory_canonical_sha256: str


@dataclass(frozen=True)
class R6AudioRecoveryClassification:
    """Immutable acquired/terminal facts for the whole r5-derived r6 queue.

    The counts are derived only after the exact `(r5_record_id, video_id)`
    population has been recomputed from the pinned recovery inventory.  Later
    consumers must use these facts instead of subtraction arithmetic.
    """

    acquired_records: tuple[dict[str, Any], ...]
    terminal_records: tuple[dict[str, Any], ...]
    effective_records: tuple[dict[str, Any], ...]
    expected_identities: frozenset[tuple[str, str]]

    @property
    def acquired_count(self) -> int:
        return len(self.acquired_records)

    @property
    def terminal_count(self) -> int:
        return len(self.terminal_records)

    @property
    def declared_count(self) -> int:
        return len(self.expected_identities)


@dataclass(frozen=True)
class R6AudioRecoveryAuthority:
    """Frozen ordered r5→r6 recovery authority used before any work split.

    This is deliberately distinct from an execution queue.  A caller cannot
    manufacture acceptance by hashing the queue it wishes to process: every
    queue is compared, in order, with this inventory-bound authority first.
    ``rows`` contain only receipt-safe identity and prior decision facts.
    """

    recovery_inventory_sha256: str
    rows: tuple[dict[str, str], ...]
    authority_sha256: str

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _canonical(value: Any, *, exclude_volatile: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item, exclude_volatile=exclude_volatile)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not (exclude_volatile and str(key) in _R6_VOLATILE_FIELDS)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item, exclude_volatile=exclude_volatile) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def canonical_r6_recovery_inventory_digest(payload: Mapping[str, Any]) -> str:
    """Hash stable r6 inventory facts while excluding self/clock fields."""

    encoded = json.dumps(
        _canonical(payload, exclude_volatile=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def build_r6_audio_recovery_authority(
    recovery_inventory: Mapping[str, Any],
    *,
    prior_oracle: Callable[[dict[str, Any]], str],
) -> R6AudioRecoveryAuthority:
    """Bind a frozen row oracle to the immutable ordered r5/r6 inventory.

    This is intentionally a *test/control-plane* helper: live recovery still
    performs one public request per queue member.  It lets the owner
    regression prove all 2,002 historical decision inputs once, then exercise
    one representative for each identical behaviour partition rather than
    accidentally making the test's call count part of the proof.
    """
    inventory_digest = recovery_inventory.get("r6_recovery_inventory_sha256")
    if not _is_sha256(inventory_digest):
        raise ValueError("r6 row authority requires an immutable inventory digest")
    # A full production inventory proves this is not merely a copied digest.
    # Tiny disposable fixtures intentionally omit receipt envelope fields, so
    # preserve that seam for the non-live regression suite only.
    if "receipt_type" in recovery_inventory and canonical_r6_recovery_inventory_digest(recovery_inventory) != inventory_digest:
        raise ValueError("r6 row authority inventory digest is stale")
    rows = recovery_inventory.get("asr_fallback_required")
    if not isinstance(rows, list) or len(rows) != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT:
        raise ValueError("r6 row oracle requires the exact immutable queue")
    frozen: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("r6 row oracle has a malformed source row")
        validate_r6_recovery_classification(raw, context="r6 row oracle")
        item_id = raw.get("r5_record_id")
        video_id = raw.get("video_id")
        source_url = raw.get("source_url")
        if not all(isinstance(value, str) and value for value in (item_id, video_id, source_url)):
            raise ValueError("r6 row oracle source identity is malformed")
        try:
            canonical_url = _canonical_youtube_url({"video_id": video_id, "video_url": source_url})[1]
        except ValueError as exc:
            raise ValueError("r6 row authority source URL is not canonical") from exc
        identity = (item_id, video_id)
        if identity in seen:
            raise ValueError("r6 row oracle has a duplicate source identity")
        seen.add(identity)
        behavior = prior_oracle(dict(raw))
        if not isinstance(behavior, str) or not behavior:
            raise ValueError("r6 row oracle behavior is malformed")
        semantic = {
            "index": str(index), "item_id": item_id, "video_id": video_id,
            "source_url": canonical_url,
            "state": R6_REQUIRED_RECOVERY_STATE,
            "source_classification": R6_REQUIRED_SOURCE_CLASSIFICATION,
            "behavior": behavior,
        }
        semantic_digest = _json_sha256(semantic)
        frozen.append({**semantic, "prior_oracle_sha256": semantic_digest})
    canonical = {
        "recovery_inventory_sha256": inventory_digest,
        "count": len(frozen),
        "rows": frozen,
    }
    digest = _json_sha256(canonical)
    return R6AudioRecoveryAuthority(
        recovery_inventory_sha256=inventory_digest,
        rows=tuple(frozen),
        authority_sha256=digest,
    )


def validate_r6_audio_recovery_authority_queue(
    authority: R6AudioRecoveryAuthority,
    queue: object,
    *,
    recovery_inventory_sha256: str,
) -> tuple[dict[str, str], ...]:
    """Reject any queue drift before partitioning or representative work.

    Order is an intentional part of the authority.  Thus changed URL, record
    or video identity, duplicate/omitted rows, and a reordered but otherwise
    equal set all fail at this one boundary.
    """

    if not isinstance(authority, R6AudioRecoveryAuthority) or not _is_sha256(authority.authority_sha256):
        raise ValueError("r6 row authority is invalid")
    canonical = {
        "recovery_inventory_sha256": authority.recovery_inventory_sha256,
        "count": authority.row_count,
        "rows": [dict(row) for row in authority.rows],
    }
    if _json_sha256(canonical) != authority.authority_sha256:
        raise ValueError("r6 row authority digest drifted")
    if authority.recovery_inventory_sha256 != recovery_inventory_sha256:
        raise ValueError("r6 row authority inventory binding drifted")
    if not isinstance(queue, list) or len(queue) != authority.row_count:
        raise ValueError("r6 row authority queue count drifted")
    validated: list[dict[str, str]] = []
    for index, (expected, raw) in enumerate(zip(authority.rows, queue, strict=True)):
        if not isinstance(raw, Mapping):
            raise ValueError("r6 row authority queue row is malformed")
        # These two fields are immutable admission classifications, not
        # mutable worker hints.  Their validation belongs beside identity and
        # URL comparison so a caller cannot reuse a frozen identity authority
        # while silently retargeting a row to a different recovery policy.
        try:
            validate_r6_recovery_classification(
                raw, context=f"r6 row authority queue at ordered row {index}"
            )
        except ValueError as exc:
            raise ValueError(
                f"r6 row authority queue classification drifted at ordered row {index}"
            ) from exc
        item_id = raw.get("r5_record_id", raw.get("item_id"))
        video_id = raw.get("video_id")
        source_url = raw.get("source_url")
        if not all(isinstance(value, str) and value for value in (item_id, video_id, source_url)):
            raise ValueError("r6 row authority queue identity is malformed")
        try:
            actual_url = _canonical_youtube_url({"video_id": video_id, "video_url": source_url})[1]
        except ValueError as exc:
            raise ValueError(f"r6 row authority queue drifted at ordered row {index}") from exc
        actual = {"item_id": item_id, "video_id": video_id, "source_url": actual_url}
        if actual != {key: expected[key] for key in actual}:
            raise ValueError(f"r6 row authority queue drifted at ordered row {index}")
        validated.append(dict(expected))
    return tuple(validated)


def partition_r6_audio_recovery_authority(
    authority: R6AudioRecoveryAuthority,
    queue: object,
    *,
    recovery_inventory_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Return proven-equivalent partitions only after full queue validation."""

    rows = validate_r6_audio_recovery_authority_queue(
        authority, queue, recovery_inventory_sha256=recovery_inventory_sha256
    )
    partitions: dict[str, dict[str, Any]] = {}
    for row in rows:
        partition = partitions.setdefault(row["behavior"], {"behavior": row["behavior"], "representative_index": int(row["index"]), "row_count": 0})
        partition["row_count"] += 1
    return tuple(partitions[key] for key in sorted(partitions))


def validate_r6_audio_recovery_authority_rows(
    authority: R6AudioRecoveryAuthority,
    rows: object,
    *,
    recovery_inventory_sha256: str,
    allow_prefix: bool,
    start_index: int = 0,
) -> tuple[dict[str, str], ...]:
    """Bind owner output to every frozen seed, not aggregate arithmetic.

    ``rows`` are receipt rows (rather than the raw inventory queue) and thus
    use ``item_id``.  A running checkpoint can be a strict ordered prefix;
    final output must cover every authority row exactly once.  The semantic
    outcome is intentionally *not* predicted here -- whether a public stream
    is still available is volatile -- but its immutable admission identity,
    source URL and bounded retry evidence are not.
    """
    if not isinstance(rows, list):
        raise ValueError("r6 row authority receipt rows are invalid")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or start_index < 0
        or start_index > authority.row_count
    ):
        raise ValueError("r6 row authority receipt offset is invalid")
    remaining_count = authority.row_count - start_index
    if (not allow_prefix and len(rows) != remaining_count) or len(rows) > remaining_count:
        raise ValueError("r6 row authority receipt count drifted")
    queue = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("r6 row authority receipt row is malformed")
        queue.append({
            "r5_record_id": raw.get("item_id"),
            "video_id": raw.get("video_id"),
            "source_url": raw.get("source_url"),
        })
    # The generic queue checker deliberately requires a complete population;
    # make the same ordered comparison for checkpoint prefixes here.
    if authority.recovery_inventory_sha256 != recovery_inventory_sha256:
        raise ValueError("r6 row authority inventory binding drifted")
    checked: list[dict[str, str]] = []
    for index, (expected, raw) in enumerate(
        zip(authority.rows[start_index:], queue, strict=False), start=start_index
    ):
        if not all(isinstance(raw.get(key), str) and raw[key] for key in ("r5_record_id", "video_id", "source_url")):
            raise ValueError("r6 row authority receipt identity is malformed")
        try:
            actual_url = _canonical_youtube_url({"video_id": raw["video_id"], "video_url": raw["source_url"]})[1]
        except ValueError as exc:
            raise ValueError(f"r6 row authority receipt drifted at ordered row {index}") from exc
        if (raw["r5_record_id"], raw["video_id"], actual_url) != (expected["item_id"], expected["video_id"], expected["source_url"]):
            raise ValueError(f"r6 row authority receipt drifted at ordered row {index}")
        checked.append(dict(expected))
    return tuple(checked)


def build_r6_audio_row_oracle_manifest(
    recovery_inventory: Mapping[str, Any], *, prior_oracle: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    """Compatibility projection of the immutable authority for audit tests."""

    authority = build_r6_audio_recovery_authority(
        recovery_inventory, prior_oracle=prior_oracle
    )
    partitions = partition_r6_audio_recovery_authority(
        authority,
        list(recovery_inventory["asr_fallback_required"]),
        recovery_inventory_sha256=authority.recovery_inventory_sha256,
    )
    return {
        "row_count": authority.row_count, "rows": [dict(row) for row in authority.rows],
        "partitions": list(partitions),
        "manifest_sha256": authority.authority_sha256,
    }


def _canonical_source_scope_digest(routes: list[dict[str, Any]]) -> str:
    """Return the stable digest for sorted, sanitized route facts only."""

    payload = {
        "source_scope_count": len(routes),
        "routes": routes,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_cvpr_source_route(value: object) -> str:
    """Accept only an uncredentialed official CVPR 2026 virtual event route."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("r5 recovery record has no source event URL")
    parsed = urlsplit(value.strip())
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("r5 recovery record source route has an invalid port") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname.lower().removeprefix("www.") != "cvpr.thecvf.com"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("r5 recovery record source route is not an official CVPR event URL")
    segments = tuple(segment for segment in parsed.path.split("/") if segment)
    if len(segments) < 3 or segments[:2] != ("virtual", "2026"):
        raise ValueError("r5 recovery record source route is outside the CVPR 2026 virtual boundary")
    return urlunsplit(("https", "cvpr.thecvf.com", "/" + "/".join(segments), "", ""))


def derive_r6_source_route_scope(
    recovery_inventory: Mapping[str, Any], *, verify_inventory_digest: bool = True
) -> R6SourceRouteScope:
    """Recompute the immutable official source-route scope from r6 recovery rows.

    This function is intentionally pure.  Callers must run it before loading
    environment state, constructing an authenticated opener, starting a child
    process, or creating a payload.  A stale serialized scope is rejected by
    :func:`validate_r6_source_route_scope` below rather than trusted.
    """

    if (
        recovery_inventory.get("receipt_type")
        != "conference_pipeline.cvpr2026_r6_media_recovery_inventory"
        or recovery_inventory.get("run_id") != R6_EVENT_MEDIA_RUN_ID
    ):
        raise ValueError("r6 recovery inventory identity is invalid")
    counts = _require_mapping(recovery_inventory.get("counts"), name="recovery inventory counts")
    records = recovery_inventory.get("asr_fallback_required")
    if (
        counts.get("asr_fallback_required") != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT
        or not isinstance(records, list)
        or len(records) != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT
    ):
        raise ValueError("r6 recovery inventory does not contain the exact 2002-record queue")
    actual_digest = canonical_r6_recovery_inventory_digest(recovery_inventory)
    if verify_inventory_digest and recovery_inventory.get("r6_recovery_inventory_sha256") != actual_digest:
        raise ValueError("r6 recovery inventory digest is stale or invalid")

    by_url: dict[str, dict[str, set[str]]] = {}
    seen_video_ids: set[str] = set()
    seen_record_ids: set[str] = set()
    for raw_record in records:
        record = _require_mapping(raw_record, name="recovery record")
        record_id = record.get("r5_record_id")
        video_id = record.get("video_id")
        source_urls = record.get("source_event_urls")
        if (
            not isinstance(record_id, str)
            or not record_id
            or not isinstance(video_id, str)
            or not video_id
            or not isinstance(source_urls, list)
            or not source_urls
            or record_id in seen_record_ids
            or video_id in seen_video_ids
        ):
            raise ValueError("r6 recovery records do not provide unique source-route linkage")
        seen_record_ids.add(record_id)
        seen_video_ids.add(video_id)
        for raw_url in source_urls:
            url = _canonical_cvpr_source_route(raw_url)
            linked = by_url.setdefault(url, {"record_ids": set(), "video_ids": set()})
            linked["record_ids"].add(record_id)
            linked["video_ids"].add(video_id)

    routes: list[dict[str, Any]] = []
    for source_page_url in sorted(by_url):
        linked = by_url[source_page_url]
        route_id = "cvpr2026:r6:route:" + sha256(source_page_url.encode("utf-8")).hexdigest()
        routes.append(
            {
                "route_id": route_id,
                "event_id": route_id,
                "source_page_url": source_page_url,
                "linked_r5_record_ids": sorted(linked["record_ids"]),
                "linked_video_ids": sorted(linked["video_ids"]),
            }
        )
    if not routes:
        raise ValueError("r6 recovery inventory has an empty official source-route scope")
    return R6SourceRouteScope(
        routes=tuple(routes),
        source_scope_count=len(routes),
        source_scope_sha256=_canonical_source_scope_digest(routes),
    )


def source_route_scope_payload(scope: R6SourceRouteScope) -> dict[str, Any]:
    """Return the receipt-safe representation of a recomputed route scope."""

    return {
        "source_scope_count": scope.source_scope_count,
        "source_scope_sha256": scope.source_scope_sha256,
        "source_routes": [dict(route) for route in scope.routes],
    }


def validate_r6_source_route_scope(
    recovery_inventory: Mapping[str, Any], source_scope: Mapping[str, object]
) -> R6SourceRouteScope:
    """Reject a substituted, partial, stale, or expanded serialized scope."""

    expected = derive_r6_source_route_scope(recovery_inventory)
    if (
        source_scope.get("source_scope_count") != expected.source_scope_count
        or source_scope.get("source_scope_sha256") != expected.source_scope_sha256
        or source_scope.get("source_routes") != [dict(route) for route in expected.routes]
    ):
        raise ValueError("r6 source-route scope does not match the immutable recovery inventory")
    return expected


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"r5 {name} is invalid")
    return value


def _fixture_contract(workspace: Path, contract: R6ArtifactContract | None) -> R6ArtifactContract:
    """Resolve a production contract or the explicit fixture injection seam."""

    resolved = contract or R6ArtifactContract.production()
    try:
        resolved.require_workspace(workspace)
    except R6ArtifactContractError as exc:
        raise ValueError("r6 workspace does not match the artifact contract") from exc
    return resolved


def resolve_r6_output_path(
    workspace: Path, output_path: object, *, contract: R6ArtifactContract | None = None
) -> Path:
    """Resolve only the contract-declared recovery inventory checkpoint."""

    resolved_contract = _fixture_contract(workspace, contract)
    try:
        return resolved_contract.require_path("recovery_inventory", output_path)
    except R6ArtifactContractError as exc:
        raise ValueError("r6 output path is not the canonical recovery inventory") from exc


def atomic_write_r6_json(
    *,
    workspace: Path,
    output_path: object,
    payload: dict[str, Any],
    contract: R6ArtifactContract | None = None,
) -> Path:
    """Atomically publish an r6 control JSON only below its separate root."""

    resolved = resolve_r6_output_path(workspace, output_path, contract=contract)
    atomic_write_json(resolved, payload)
    return resolved


def _validate_r5_receipt(
    *, receipt: Mapping[str, Any], receipt_file_sha256: str, pins: R5LineagePins
) -> None:
    if not all(
        _is_sha256(value)
        for value in (pins.receipt_sha256, pins.inventory_sha256, pins.inventory_canonical_sha256)
    ):
        raise ValueError("r5 lineage pins must be SHA-256 digests")
    if receipt_file_sha256 != pins.receipt_sha256:
        raise ValueError("r5 receipt digest does not match the pinned digest")
    if receipt.get("run_id") != R5_EVENT_MEDIA_RUN_ID:
        raise ValueError("r5 receipt run id is invalid")
    if receipt.get("stage_id") != "download-corpus" or receipt.get("mode") != "live":
        raise ValueError("r5 receipt stage identity is invalid")
    if receipt.get("status") != "succeeded":
        raise ValueError("r5 receipt status is not succeeded")
    stage_contract = _require_mapping(receipt.get("stage_contract"), name="receipt stage contract")
    if (
        stage_contract.get("status") != "pass"
        or stage_contract.get("stage_id") != "download-corpus"
    ):
        raise ValueError("r5 receipt stage contract is not passing")
    inventory_binding = _require_mapping(
        receipt.get("event_media_inventory"), name="receipt inventory binding"
    )
    checks = _require_mapping(receipt.get("checks"), name="receipt checks")
    if (
        inventory_binding.get("path")
        != "storage/download-corpus/event-media/event_media_inventory.json"
    ):
        raise ValueError("r5 receipt inventory path is invalid")
    if inventory_binding.get("inventory_sha256") != pins.inventory_canonical_sha256:
        raise ValueError("r5 receipt canonical inventory digest does not match")
    if checks.get("event_media_inventory_path") != inventory_binding.get("path"):
        raise ValueError("r5 receipt inventory path check does not match")
    if checks.get("event_media_inventory_sha256") != pins.inventory_sha256:
        raise ValueError("r5 receipt check digest does not match")
    if checks.get("event_media_inventory_canonical_sha256") != pins.inventory_canonical_sha256:
        raise ValueError("r5 receipt check canonical digest does not match")


def _canonical_youtube_url(candidate: Mapping[str, Any]) -> tuple[str, str]:
    video_id = candidate.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("r5 unavailable-caption candidate has no video id")
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    if extract_video_id(canonical_url) != video_id:
        raise ValueError("r5 unavailable-caption candidate has an invalid YouTube video id")
    video_url = candidate.get("video_url")
    if not isinstance(video_url, str) or not url_host_allowed(video_url, DEFAULT_ALLOWED_HOSTS):
        raise ValueError("r5 unavailable-caption candidate is not an allowed YouTube source")
    if video_url != canonical_url:
        raise ValueError("r5 unavailable-caption candidate has a noncanonical YouTube URL")
    return video_id, canonical_url


def _validated_r5_unavailable_candidates(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    if inventory.get("run_id") != R5_EVENT_MEDIA_RUN_ID:
        raise ValueError("r5 inventory run id is invalid")
    if canonical_event_media_inventory_digest(inventory) != inventory.get(
        "event_media_inventory_sha256"
    ):
        raise ValueError("r5 inventory canonical digest does not match")
    candidates = inventory.get("caption_candidates")
    if not isinstance(candidates, list):
        raise ValueError("r5 inventory caption candidates are invalid")
    counts = _require_mapping(inventory.get("counts"), name="inventory counts")
    if counts.get("caption_candidate_count") != len(candidates):
        raise ValueError("r5 inventory candidate count drifted")

    records: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    for index, raw_candidate in enumerate(candidates):
        candidate = _require_mapping(raw_candidate, name="caption candidate")
        if candidate.get("caption_status") != "youtube_caption_unavailable":
            continue
        if candidate.get("caption") is not None:
            raise ValueError("r5 unavailable-caption candidate has caption metadata")
        if (
            candidate.get("audio_downloaded") is not False
            or candidate.get("video_downloaded") is not False
        ):
            raise ValueError("r5 unavailable-caption candidate violates its media boundary")
        video_id, source_url = _canonical_youtube_url(candidate)
        if video_id in seen_video_ids:
            raise ValueError("r5 unavailable-caption queue has a duplicate YouTube video id")
        source_event_urls = candidate.get("source_event_urls")
        provenance = candidate.get("provenance")
        if (
            not isinstance(source_event_urls, list)
            or not source_event_urls
            or not all(isinstance(value, str) and value for value in source_event_urls)
            or not isinstance(provenance, list)
            or not provenance
            or not all(isinstance(value, str) and value for value in provenance)
        ):
            raise ValueError("r5 unavailable-caption candidate lacks source lineage")
        record_id = candidate.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            record_id = f"r5:caption:{video_id}"
        seen_video_ids.add(video_id)
        records.append(
            {
                "r5_record_id": record_id,
                "video_id": video_id,
                "source_url": source_url,
                "source_event_urls": list(source_event_urls),
                "provenance": list(provenance),
                "source_classification": "public_youtube_candidate",
                "state": "asr_fallback_required",
                "attempt_count": 0,
                "local_media": None,
                "terminal_blocker": None,
            }
        )
    if len(records) != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT:
        raise ValueError("r5 unavailable-caption queue count does not equal 2002")
    return sorted(records, key=lambda record: str(record["video_id"]))


def build_r6_recovery_inventory(
    *,
    r5_receipt: Mapping[str, Any],
    r5_inventory: Mapping[str, Any],
    pins: R5LineagePins,
    workspace: Path,
    output_path: object,
    r5_receipt_file_sha256: str | None = None,
    r5_inventory_file_sha256: str | None = None,
    contract: R6ArtifactContract | None = None,
) -> dict[str, Any]:
    """Build the pure r6 ASR-fallback queue from the immutable r5 producer.

    This function validates every admission condition before returning an
    in-memory control payload.  It deliberately does not write that payload;
    callers must opt into :func:`atomic_write_r6_json` after their own gate.
    """

    receipt_file_sha256 = r5_receipt_file_sha256 or _json_sha256(r5_receipt)
    inventory_file_sha256 = r5_inventory_file_sha256 or _json_sha256(r5_inventory)
    _validate_r5_receipt(receipt=r5_receipt, receipt_file_sha256=receipt_file_sha256, pins=pins)
    if inventory_file_sha256 != pins.inventory_sha256:
        raise ValueError("r5 inventory digest does not match the pinned digest")
    if canonical_event_media_inventory_digest(r5_inventory) != pins.inventory_canonical_sha256:
        raise ValueError("r5 inventory canonical digest does not match the pinned digest")
    artifact_contract = _fixture_contract(workspace, contract)
    resolved_output = resolve_r6_output_path(workspace, output_path, contract=artifact_contract)
    records = _validated_r5_unavailable_candidates(r5_inventory)
    workspace_root = artifact_contract.workspace
    inventory: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_media_recovery_inventory",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "output_path": resolved_output.relative_to(workspace_root).as_posix(),
        "r5_lineage": {
            "run_id": R5_EVENT_MEDIA_RUN_ID,
            "receipt_sha256": pins.receipt_sha256,
            "inventory_sha256": pins.inventory_sha256,
            "inventory_canonical_sha256": pins.inventory_canonical_sha256,
        },
        "counts": {"asr_fallback_required": len(records)},
        "asr_fallback_required": records,
        "boundaries": {
            "external_commands_invoked": False,
            "network_requests_made": 0,
            "provider_calls_made": 0,
            "credentials_read": False,
            "media_created": False,
            "backup_started": False,
        },
    }
    scope = derive_r6_source_route_scope(
        {
            **inventory,
            # The inventory digest is added only after source-route derivation;
            # derive against a temporary self-consistent value to retain one
            # canonical construction path.
            "r6_recovery_inventory_sha256": "",
        },
        verify_inventory_digest=False,
    )
    inventory.update(source_route_scope_payload(scope))
    inventory["r6_recovery_inventory_sha256"] = canonical_r6_recovery_inventory_digest(inventory)
    return inventory


def _contains_secretish_key(value: object) -> bool:
    """Keep credentials and browser/auth state out of r6 control receipts."""

    forbidden = ("cookie", "password", "secret", "token", "authorization", "api_key")
    if isinstance(value, Mapping):
        return any(
            (
                str(key).lower() != "secrets_copied"
                and any(marker in str(key).lower() for marker in forbidden)
            )
            or _contains_secretish_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secretish_key(item) for item in value)
    return False


def _require_audio_identity(
    row: Mapping[str, Any], *, expected: Mapping[tuple[str, str], Mapping[str, Any]]
) -> tuple[str, str]:
    item_id = row.get("item_id")
    video_id = row.get("video_id")
    if not isinstance(item_id, str) or not item_id or not isinstance(video_id, str) or not video_id:
        raise ValueError("r6 audio classification has an invalid seed identity")
    identity = (item_id, video_id)
    if identity not in expected:
        raise ValueError("r6 audio classification has an extra or swapped seed identity")
    source_url = row.get("source_url")
    if source_url != expected[identity].get("source_url"):
        raise ValueError("r6 audio classification source provenance drifted")
    if row.get("provenance") != "direct_public_youtube":
        raise ValueError("r6 audio classification is not direct-public YouTube provenance")
    return identity


def _require_attempt_evidence(row: Mapping[str, Any], *, terminal_state: str) -> None:
    attempt_count = row.get("attempt_count")
    max_attempts = R6_PUBLIC_AUDIO_POLICY["max_attempts"]
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or not 1 <= attempt_count <= max_attempts
    ):
        raise ValueError("r6 audio classification attempt evidence violates public policy")
    # A terminal state records the number of recovery operations actually
    # completed.  In particular, deterministic metadata/format rejection can
    # be terminal after one operation; it is not evidence that three retries
    # occurred.  Resume treats every validated terminal state as complete.


def _validate_acquired_audio_row(
    row: Mapping[str, Any], *, contract: R6ArtifactContract, verify_payloads: bool
) -> dict[str, Any]:
    if "local_path" in row:
        raise ValueError("r6 acquired audio may not carry a legacy local_path alias")
    local_path = row.get("local_media_path")
    if not isinstance(local_path, str) or not local_path:
        raise ValueError("r6 acquired audio lacks a contract-contained local path")
    byte_count = row.get("byte_count")
    digest = row.get("sha256")
    mime_type = row.get("mime_type")
    duration = row.get("duration_seconds")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
        or not _is_sha256(digest)
        or not isinstance(mime_type, str)
        or not mime_type.strip()
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
    ):
        raise ValueError("r6 acquired audio lacks byte, digest, MIME, or duration proof")
    if verify_payloads:
        try:
            content, _relative = read_r6_raw_audio_payload(
                contract=contract, local_media_path=local_path
            )
        except R6ArtifactContractError as exc:
            raise ValueError("r6 raw audio payload violates the fixed containment boundary") from exc
        if len(content) != byte_count or sha256(content).hexdigest() != digest:
            raise ValueError("r6 acquired audio local payload bytes do not match its receipt")
    return dict(row)


def _validate_historical_checkpoint_lineage(
    *,
    recovery_inventory: Mapping[str, Any],
    receipt: Mapping[str, Any],
    contract: R6ArtifactContract,
) -> None:
    """Revalidate the sole archived-prefix successor before it is consumed.

    The archive is history only.  It must never contribute rows to the
    effective current receipt, which is independently classified below.
    """

    lineage = receipt.get(_HISTORICAL_CHECKPOINT_LINEAGE_KEY)
    forbidden_aliases = {"supersedes", "supersession", "predecessor_receipt"}
    if lineage is None:
        if any(key in receipt for key in forbidden_aliases):
            raise ValueError("r6 audio receipt has an unknown supersession shape")
        return
    if not isinstance(lineage, Mapping) or set(lineage) != {
        "archive_path",
        "archive_sha256",
        "archive_byte_count",
        "original",
        "preflight",
        "effective_seed_count",
    }:
        raise ValueError("r6 audio successor lineage is malformed")
    archive_digest = lineage.get("archive_sha256")
    archive_count = lineage.get("archive_byte_count")
    if not _is_sha256(archive_digest) or isinstance(archive_count, bool) or not isinstance(archive_count, int) or archive_count <= 0:
        raise ValueError("r6 audio successor archive binding is invalid")
    expected_relative = archive_r6_audio_checkpoint_relative_path(archive_digest)
    if lineage.get("archive_path") != expected_relative.as_posix():
        raise ValueError("r6 audio successor archive path drifted")
    try:
        raw = read_r6_audio_checkpoint_archive(
            contract=contract,
            original_sha256=archive_digest,
        )
        original = json.loads(raw.decode("utf-8"))
    except (R6ArtifactContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("r6 audio successor archive is unreadable") from exc
    if not isinstance(original, Mapping):
        raise ValueError("r6 audio successor archive is not a receipt object")
    if len(raw) != archive_count or sha256(raw).hexdigest() != archive_digest:
        raise ValueError("r6 audio successor archive bytes drifted")
    expected_original = lineage.get("original")
    if not isinstance(expected_original, Mapping):
        raise ValueError("r6 audio successor original lineage is invalid")
    original_summary = _validate_historical_checkpoint_original(
        recovery_inventory=recovery_inventory,
        original=original,
        expected_original=expected_original,
        contract=contract,
    )
    if original_summary["classification"] == "observed_running_terminal_prefix":
        original_terminal = original.get("terminal_records")
        successor_terminal = receipt.get("terminal_records")
        if (
            not isinstance(original_terminal, list)
            or len(original_terminal) != 1
            or not isinstance(successor_terminal, list)
            or not successor_terminal
            or successor_terminal[0] != original_terminal[0]
        ):
            raise ValueError("r6 audio successor does not preserve observed terminal evidence")
    preflight = lineage.get("preflight")
    try:
        from conference_pipeline.acquisition.youtube_audio_recovery import (
            R6DownloaderPreflightError,
            validate_fixed_r6_yt_dlp_preflight,
        )
        validate_fixed_r6_yt_dlp_preflight(preflight)
    except (R6DownloaderPreflightError, TypeError):
        raise ValueError("r6 audio successor fixed downloader lineage is invalid")
    if lineage.get("effective_seed_count") != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT:
        raise ValueError("r6 audio successor effective seed count drifted")


def _validate_historical_checkpoint_original(
    *,
    recovery_inventory: Mapping[str, Any],
    original: Mapping[str, Any],
    expected_original: Mapping[str, Any] | None,
    contract: R6ArtifactContract,
) -> dict[str, object]:
    """Admit only the frozen 1,043 prefix or observed one-row terminal prefix."""

    if _HISTORICAL_CHECKPOINT_LINEAGE_KEY in original or any(
        key in original for key in ("supersedes", "supersession", "predecessor_receipt")
    ):
        raise ValueError("r6 historical checkpoint has an unknown supersession shape")
    if original.get("status") != "running":
        raise ValueError("r6 historical checkpoint is not running")
    if original.get("receipt_type") != "conference_pipeline.cvpr2026_r6_audio_recovery":
        raise ValueError("r6 historical checkpoint receipt type is invalid")
    if original.get("run_id") != R6_EVENT_MEDIA_RUN_ID:
        raise ValueError("r6 historical checkpoint run identity is invalid")
    if original.get("output_path") != contract.relative_path("audio_recovery_receipt").as_posix():
        raise ValueError("r6 historical checkpoint output path is invalid")
    if original.get("recovery_inventory_sha256") != recovery_inventory.get(
        "r6_recovery_inventory_sha256"
    ):
        raise ValueError("r6 historical checkpoint inventory binding drifted")
    if original.get("audio_policy_digest") != r6_audio_policy_digest() or original.get(
        "audio_policy_max_attempts"
    ) != R6_PUBLIC_AUDIO_POLICY["max_attempts"]:
        raise ValueError("r6 historical checkpoint audio policy drifted")
    scope = derive_r6_source_route_scope(recovery_inventory)
    if (
        original.get("source_scope_count") != scope.source_scope_count
        or original.get("source_scope_sha256") != scope.source_scope_sha256
    ):
        raise ValueError("r6 historical checkpoint source scope drifted")
    records = original.get("records")
    terminal_records = original.get("terminal_records")
    counts = original.get("counts")
    if records != [] or not isinstance(terminal_records, list):
        raise ValueError("r6 historical checkpoint has acquired or malformed evidence")
    prefix_count = len(terminal_records)
    if prefix_count not in {_HISTORICAL_CHECKPOINT_RECORD_COUNT, 1} or counts != {
        "declared": prefix_count, "acquired": 0, "terminal": prefix_count,
    }:
        raise ValueError("r6 historical checkpoint counts are not an allowed terminal prefix")
    # Reuse the ordinary prefix classifier for identity/provenance/attempt
    # checks, then require this exact ordered prefix of the frozen queue.
    classified = _classify_r6_audio_recovery_receipt(
        recovery_inventory=recovery_inventory,
        receipt=original,
        contract=contract,
        require_complete=False,
        verify_payloads=False,
        _skip_lineage=True,
    )
    if classified.acquired_records or len(classified.terminal_records) != prefix_count:
        raise ValueError("r6 historical checkpoint contains acquired evidence")
    expected_rows = recovery_inventory.get("asr_fallback_required")
    if not isinstance(expected_rows, list):  # Defensive; the shared classifier already checked it.
        raise ValueError("r6 historical checkpoint recovery inventory is malformed")
    expected_prefix = [
        (str(row.get("r5_record_id")), str(row.get("video_id"))) for row in expected_rows[:prefix_count]
    ]
    observed_prefix = [
        (str(row.get("item_id")), str(row.get("video_id"))) for row in terminal_records
    ]
    if observed_prefix != expected_prefix or any(row.get("terminal_state") != "download_failed" for row in terminal_records):
        raise ValueError("r6 historical checkpoint is not the exact failed prefix")
    classification = (
        "strict_running_download_failed_prefix"
        if prefix_count == _HISTORICAL_CHECKPOINT_RECORD_COUNT
        else "observed_running_terminal_prefix"
    )
    original_summary: dict[str, object] = {
        "status": "running",
        "receipt_type": "conference_pipeline.cvpr2026_r6_audio_recovery",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "output_path": contract.relative_path("audio_recovery_receipt").as_posix(),
        "recovery_inventory_sha256": str(recovery_inventory["r6_recovery_inventory_sha256"]),
        "audio_policy_digest": r6_audio_policy_digest(),
        "audio_policy_max_attempts": R6_PUBLIC_AUDIO_POLICY["max_attempts"],
        "source_scope_count": scope.source_scope_count,
        "source_scope_sha256": scope.source_scope_sha256,
        "counts": dict(counts),
        "classification": classification,
    }
    if expected_original is not None and dict(expected_original) != original_summary:
        raise ValueError("r6 audio successor original lineage drifted")
    return original_summary


def _classify_r6_audio_recovery_receipt(
    *,
    recovery_inventory: Mapping[str, Any],
    receipt: Mapping[str, Any],
    contract: R6ArtifactContract,
    require_complete: bool,
    verify_payloads: bool,
    _skip_lineage: bool = False,
) -> R6AudioRecoveryClassification:
    """Validate either an exact final receipt or a safe resume prefix.

    The public wrapper below always requests full coverage.  The fixed owner
    uses this same classifier for a resumable checkpoint, where a proper
    prefix is allowed but each row is still an immutable, policy-bound seed.
    """

    try:
        contract.require_workspace(contract.workspace)
    except R6ArtifactContractError as exc:  # pragma: no cover - invariant guard
        raise ValueError("r6 audio classification contract workspace drifted") from exc
    if _contains_secretish_key(receipt):
        raise ValueError("r6 audio receipt contains a secret-like field")
    if not _skip_lineage:
        _validate_historical_checkpoint_lineage(
            recovery_inventory=recovery_inventory,
            receipt=receipt,
            contract=contract,
        )
    if receipt.get("receipt_type") != "conference_pipeline.cvpr2026_r6_audio_recovery":
        raise ValueError("r6 audio receipt type is invalid")
    if receipt.get("run_id") != R6_EVENT_MEDIA_RUN_ID:
        raise ValueError("r6 audio receipt run identity is invalid")
    status = receipt.get("status")
    allowed_statuses = {"succeeded"} if require_complete else {"running", "succeeded"}
    if status not in allowed_statuses:
        raise ValueError("r6 audio receipt status is invalid for its classification")
    if receipt.get("recovery_inventory_sha256") != recovery_inventory.get(
        "r6_recovery_inventory_sha256"
    ):
        raise ValueError("r6 audio receipt recovery inventory binding drifted")
    if receipt.get("audio_policy_digest") != r6_audio_policy_digest():
        raise ValueError("r6 audio receipt public policy digest drifted")
    if receipt.get("audio_policy_max_attempts") != R6_PUBLIC_AUDIO_POLICY["max_attempts"]:
        raise ValueError("r6 audio receipt public retry policy drifted")
    try:
        scope = derive_r6_source_route_scope(recovery_inventory)
        validate_r6_source_route_scope(recovery_inventory, recovery_inventory)
    except ValueError as exc:
        raise ValueError("r6 audio receipt immutable source scope is invalid") from exc
    if (
        receipt.get("source_scope_count") != scope.source_scope_count
        or receipt.get("source_scope_sha256") != scope.source_scope_sha256
    ):
        raise ValueError("r6 audio receipt source scope drifted")
    expected_rows = recovery_inventory.get("asr_fallback_required")
    if not isinstance(expected_rows, list) or len(expected_rows) != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT:
        raise ValueError("r6 audio recovery inventory does not contain the exact seed population")
    expected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for source in expected_rows:
        if not isinstance(source, Mapping):
            raise ValueError("r6 audio recovery inventory seed is malformed")
        record_id = source.get("r5_record_id")
        video_id = source.get("video_id")
        if not isinstance(record_id, str) or not record_id or not isinstance(video_id, str) or not video_id:
            raise ValueError("r6 audio recovery inventory seed identity is malformed")
        identity = (record_id, video_id)
        if identity in expected:
            raise ValueError("r6 audio recovery inventory has duplicate seed identities")
        expected[identity] = source
    if len(expected) != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT:
        raise ValueError("r6 audio recovery inventory exact seed count drifted")
    records = receipt.get("records")
    terminal_records = receipt.get("terminal_records")
    if not isinstance(records, list) or not isinstance(terminal_records, list):
        raise ValueError("r6 audio receipt must declare acquired and terminal record lists")
    seen: set[tuple[str, str]] = set()
    acquired: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, Mapping) or row.get("terminal_state") != "acquired":
            raise ValueError("r6 audio acquired record classification is invalid")
        identity = _require_audio_identity(row, expected=expected)
        if identity in seen:
            raise ValueError("r6 audio receipt duplicates an immutable seed")
        _require_attempt_evidence(row, terminal_state="acquired")
        acquired.append(_validate_acquired_audio_row(row, contract=contract, verify_payloads=verify_payloads))
        seen.add(identity)
    for row in terminal_records:
        if not isinstance(row, Mapping):
            raise ValueError("r6 audio terminal record classification is invalid")
        terminal_state = row.get("terminal_state")
        if terminal_state not in TERMINAL_STATES or terminal_state == "acquired":
            raise ValueError("r6 audio terminal state is not allowlisted")
        identity = _require_audio_identity(row, expected=expected)
        if identity in seen:
            raise ValueError("r6 audio receipt duplicates an immutable seed")
        _require_attempt_evidence(row, terminal_state=str(terminal_state))
        if any(key in row for key in ("local_path", "local_media_path", "sha256", "byte_count", "mime_type")):
            raise ValueError("r6 terminal audio record fabricates a local payload")
        terminal.append(dict(row))
        seen.add(identity)
    counts = receipt.get("counts")
    if not isinstance(counts, Mapping) or counts != {
        "declared": len(seen),
        "acquired": len(acquired),
        "terminal": len(terminal),
    }:
        raise ValueError("r6 audio receipt count evidence does not match classifications")
    if require_complete and seen != set(expected):
        raise ValueError("r6 audio receipt does not classify every immutable seed exactly once")

    # ``records`` and ``terminal_records`` are deliberately separate because
    # their payload contracts differ.  A resume, however, needs one producer
    # order.  Do not silently reconstruct it as ``records + terminal`` when
    # both containers have evidence: that loses the ordering proof required to
    # compare a partial checkpoint with the immutable r5-derived authority.
    order = receipt.get("effective_record_order")
    if order is None:
        if acquired and terminal and not require_complete:
            raise ValueError("r6 resumable receipt lacks effective record order")
        effective = [*acquired, *terminal]
    else:
        if not isinstance(order, list) or len(order) != len(seen):
            raise ValueError("r6 audio receipt effective record order is invalid")
        by_identity = {
            (str(row["item_id"]), str(row["video_id"])): row
            for row in [*acquired, *terminal]
        }
        effective = []
        ordered_identities: set[tuple[str, str]] = set()
        for index, entry in enumerate(order):
            if not isinstance(entry, Mapping):
                raise ValueError("r6 audio receipt effective record order is invalid")
            item_id, video_id = entry.get("item_id"), entry.get("video_id")
            if not isinstance(item_id, str) or not item_id or not isinstance(video_id, str) or not video_id:
                raise ValueError("r6 audio receipt effective record order is invalid")
            identity = (item_id, video_id)
            if identity in ordered_identities or identity not in by_identity:
                raise ValueError(f"r6 audio receipt effective record order drifted at ordered row {index}")
            ordered_identities.add(identity)
            effective.append(by_identity[identity])
        if ordered_identities != seen:
            raise ValueError("r6 audio receipt effective record order is incomplete")
    return R6AudioRecoveryClassification(
        acquired_records=tuple(acquired),
        terminal_records=tuple(terminal),
        effective_records=tuple(effective),
        expected_identities=frozenset(expected),
    )


def validate_r6_audio_recovery_receipt(
    *,
    recovery_inventory: Mapping[str, Any],
    receipt: Mapping[str, Any],
    contract: R6ArtifactContract,
) -> R6AudioRecoveryClassification:
    """Require complete exact r5→r6 acquired-or-terminal classification.

    This is the sole consumer-facing boundary for audio recovery.  It never
    derives terminal totals from a count subtraction, so a no-paid outcome
    requires the same exact 2,002 terminal evidence as any paid run.
    """

    return _classify_r6_audio_recovery_receipt(
        recovery_inventory=recovery_inventory,
        receipt=receipt,
        contract=contract,
        require_complete=True,
        verify_payloads=True,
    )


def validate_r6_audio_recovery_resume(
    *,
    recovery_inventory: Mapping[str, Any],
    receipt: Mapping[str, Any],
    contract: R6ArtifactContract,
) -> R6AudioRecoveryClassification:
    """Use the shared classifier for an unfinished, exact-prefix checkpoint."""

    return _classify_r6_audio_recovery_receipt(
        recovery_inventory=recovery_inventory,
        receipt=receipt,
        contract=contract,
        require_complete=False,
        verify_payloads=True,
    )
