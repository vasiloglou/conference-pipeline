"""Source-bound program-spine controls for Stage 4 (talk spine + focused recovery).

The frozen CVPR inventory is deliberately broad: it includes every source
record, not a pre-joined talk table.  This adapter derives a *separate* talk
spine from the immutable Phase-3 selection without adding, removing, or
rewriting selected program items.  It is intentionally local-only: fetching
schedule pages, searching YouTube, downloading captions, and focused media
recovery are owned by their existing acquisition lanes.

Recommended for
---------------
``RECOMMENDED_FOR = ("CVPR",)`` (formerly ``acquisition/cvpr_program_spine.py``;
that import path remains a compatibility alias).

* **CVPR stage and quirk covered:** Stage 4 ``download-corpus`` r3: the talk
  spine the organizer-channel talk lane
  (:mod:`conference_pipeline.acquisition.youtube_channel_harvest`) searches
  for, and the cached SlidesLive focused-recovery assessment lane
  (:func:`focused_recovery_lane`).
* **May also fit:** other CVF virtual-site conferences (ICCV, WACV) whose
  schedule pages use the same ``/virtual/<year>/<family>/`` event paths; the
  virtual host and year below are the CVPR defaults and would become
  settings for such a conference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from conference_pipeline.slideslive_recovery import extract_slideslive_id, sanitize_url


RECOMMENDED_FOR = ("CVPR",)

CVPR_PROGRAM_SPINE_PATH = Path("storage/download-corpus/program_spine.json")
CVPR_YOUTUBE_ALLOWED_HOSTS: tuple[str, ...] = ("youtube.com", "youtu.be")
# Capability names for the same controls.
PROGRAM_SPINE_PATH = CVPR_PROGRAM_SPINE_PATH
YOUTUBE_ALLOWED_HOSTS = CVPR_YOUTUBE_ALLOWED_HOSTS
# The virtual-site hosts whose schedule/item URLs the spine reads (the CVPR
# default; see the module docstring).
_VIRTUAL_SITE_HOSTS = frozenset({"cvpr.thecvf.com", "www.cvpr.thecvf.com"})  # conference-literal-ok: CVPR-recommended lane default (RECOMMENDED_FOR)

_TALK_FAMILY_PATH_SEGMENTS = {
    "keynote": "keynote",
    "invited-talk": "invited",
    "invited": "invited",
    "oral": "oral",
    "oral-session": "oral_session",
    "poster": "poster",
}
_NON_TALK_SCHEDULE_SEGMENTS = frozenset(
    {"break", "calendar", "event", "session", "workshop", "tutorial", "expo"}
)
_SPINE_VOLATILE_FIELDS = frozenset({"program_spine_sha256", "generated_at"})


def _canonical(value: Any, *, exclude_volatile: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item, exclude_volatile=exclude_volatile)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not (exclude_volatile and str(key) in _SPINE_VOLATILE_FIELDS)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item, exclude_volatile=exclude_volatile) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_program_spine_digest(payload: Mapping[str, Any]) -> str:
    """Hash the stable spine contract while excluding its self-digest."""

    encoded = json.dumps(
        _canonical(payload, exclude_volatile=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def write_cvpr_program_spine(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a deterministic control file beneath the Stage-4 ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_canonical(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_urls(record: Mapping[str, Any]) -> list[str]:
    aliases = record.get("aliases")
    if not isinstance(aliases, list):
        return []
    urls = [str(value) for value in aliases if isinstance(value, str) and value.startswith(("https://", "http://"))]
    return sorted(set(urls))


def _schedule_family(url: str) -> str | None:
    """Read only a CVPR virtual URL's declared event path; never infer a family."""

    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if parsed.hostname not in _VIRTUAL_SITE_HOSTS:
        return None
    parts = [part.lower() for part in parsed.path.split("/") if part]
    try:
        virtual_index = parts.index("virtual")
        if parts[virtual_index + 1] != "2026":
            return None
        segment = parts[virtual_index + 2]
    except (IndexError, ValueError):
        return None
    return _TALK_FAMILY_PATH_SEGMENTS.get(segment)


def _schedule_non_talk_reason(url: str) -> str:
    try:
        parts = [part.lower() for part in urlsplit(url).path.split("/") if part]
        virtual_index = parts.index("virtual")
        segment = parts[virtual_index + 2]
    except (IndexError, ValueError):
        return "program_schedule_url_not_a_talk_event"
    if segment in _NON_TALK_SCHEDULE_SEGMENTS:
        return "schedule_event_not_talk_capable"
    return "program_schedule_url_not_a_talk_event"


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or candidate.startswith(("http://", "https://")):
        return None
    return candidate


def _string_sequence(value: Any) -> list[str]:
    if isinstance(value, str):
        candidate = _string_or_none(value)
        return [candidate] if candidate else []
    if not isinstance(value, (list, tuple)):
        return []
    return [candidate for raw in value if (candidate := _string_or_none(raw)) is not None]


def _field_provenance(record: Mapping[str, Any], *, derivation_url: str) -> dict[str, Any]:
    evidence = record.get("source_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    snapshot_sha256 = evidence.get("snapshot_sha256") or record.get("snapshot_sha256")
    base = {
        "derivation_url": sanitize_url(derivation_url),
        "snapshot_metadata_path": str(evidence.get("metadata_path") or ""),
        "snapshot_sha256": str(snapshot_sha256 or ""),
        "source_record_id": str(record.get("source_record_id") or ""),
    }
    return {field: dict(base) for field in ("title", "speakers", "family", "session", "forum_id")}


def _chosen_source_record(
    row: Mapping[str, Any],
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    source_ids = row.get("source_record_ids")
    if not isinstance(source_ids, list):
        return None
    candidates = [records_by_id[str(record_id)] for record_id in source_ids if str(record_id) in records_by_id]
    if not candidates:
        return None

    def sort_key(record: Mapping[str, Any]) -> tuple[int, str]:
        family = str(record.get("family_id") or "")
        urls = _record_urls(record)
        is_talk_schedule = family == "program-schedule" and any(_schedule_family(url) for url in urls)
        return (0 if is_talk_schedule else 1, str(record.get("source_record_id") or ""))

    return sorted(candidates, key=sort_key)[0]


def _metadata_blocker(
    *, program_item_id: str, source_record_id: str, fields: Sequence[str]
) -> dict[str, Any]:
    return {
        "code": "talk_metadata_missing",
        "fields": sorted(set(fields)),
        "program_item_id": program_item_id,
        "source_record_id": source_record_id,
    }


def build_cvpr_program_spine(
    *,
    selection_rows: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a deterministic CVPR talk spine without changing frozen scope.

    Each selected program item receives exactly one disposition: a talk-spine
    item when its schedule evidence is sufficient for public matching, or a
    source-backed non-talk / metadata-blocker row.  Missing optional speaker
    or session metadata remains visible on usable talk rows; missing title or
    family never becomes an invented match candidate.
    """

    records_by_id: dict[str, Mapping[str, Any]] = {}
    for record in source_records:
        record_id = record.get("source_record_id")
        if not isinstance(record_id, str) or not record_id or record_id in records_by_id:
            raise ValueError("source records must have unique source_record_id values")
        records_by_id[record_id] = record

    selected: dict[str, Mapping[str, Any]] = {}
    for row in selection_rows:
        item_id = row.get("program_item_id")
        if not isinstance(item_id, str) or not item_id or item_id in selected:
            raise ValueError("selection rows must have unique program_item_id values")
        if row.get("selection_status") != "included":
            raise ValueError("program spine accepts only immutable included rows")
        selected[item_id] = row

    items: list[dict[str, Any]] = []
    non_talk_rows: list[dict[str, Any]] = []
    metadata_blockers: list[dict[str, Any]] = []
    for item_id in sorted(selected):
        row = selected[item_id]
        record = _chosen_source_record(row, records_by_id)
        if record is None:
            non_talk_rows.append(
                {
                    "program_item_id": item_id,
                    "source_record_id": None,
                    "reason": "source_record_not_available_in_pinned_inventory",
                }
            )
            continue
        record_id = str(record["source_record_id"])
        family_id = str(record.get("family_id") or "")
        urls = _record_urls(record)
        derivation_url = urls[0] if urls else ""
        if family_id == "youtube":
            non_talk_rows.append(
                {
                    "program_item_id": item_id,
                    "source_record_id": record_id,
                    "reason": "youtube_source_is_not_authoritative_talk_evidence",
                    "field_provenance": _field_provenance(record, derivation_url=derivation_url),
                }
            )
            continue
        if family_id != "program-schedule":
            non_talk_rows.append(
                {
                    "program_item_id": item_id,
                    "source_record_id": record_id,
                    "reason": "source_family_not_talk_schedule",
                    "source_family_id": family_id,
                    "field_provenance": _field_provenance(record, derivation_url=derivation_url),
                }
            )
            continue
        talk_url = next((url for url in urls if _schedule_family(url)), None)
        if talk_url is None:
            non_talk_rows.append(
                {
                    "program_item_id": item_id,
                    "source_record_id": record_id,
                    "reason": _schedule_non_talk_reason(derivation_url),
                    "field_provenance": _field_provenance(record, derivation_url=derivation_url),
                }
            )
            continue
        family = _schedule_family(talk_url)
        title = _string_or_none(record.get("title"))
        missing_required = [field for field, value in (("title", title), ("family", family)) if not value]
        if missing_required:
            blocker = _metadata_blocker(
                program_item_id=item_id,
                source_record_id=record_id,
                fields=missing_required,
            )
            metadata_blockers.append(blocker)
            non_talk_rows.append(
                {
                    "program_item_id": item_id,
                    "source_record_id": record_id,
                    "reason": "talk_metadata_missing",
                    "metadata_blocker": blocker,
                    "field_provenance": _field_provenance(record, derivation_url=talk_url),
                }
            )
            continue
        speakers = _string_sequence(record.get("speakers") or record.get("authors"))
        session = _string_or_none(record.get("session"))
        forum_id = _string_or_none(record.get("forum_id"))
        missing_optional = [
            field for field, value in (("speakers", speakers), ("session", session)) if not value
        ]
        if missing_optional:
            metadata_blockers.append(
                _metadata_blocker(
                    program_item_id=item_id,
                    source_record_id=record_id,
                    fields=missing_optional,
                )
            )
        items.append(
            {
                "item_id": item_id,
                "program_item_id": item_id,
                "source_record_id": record_id,
                "title": title,
                "speakers": speakers,
                "family": family,
                "session": session,
                "forum_id": forum_id,
                "source_url": sanitize_url(talk_url),
                "metadata_blockers": missing_optional,
                "field_provenance": _field_provenance(record, derivation_url=talk_url),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_program_spine",
        "lineage": _canonical(dict(lineage)),
        "items": sorted(items, key=lambda item: str(item["item_id"])),
        "non_talk_rows": sorted(non_talk_rows, key=lambda item: str(item["program_item_id"])),
        "metadata_blockers": sorted(
            metadata_blockers,
            key=lambda item: (str(item["program_item_id"]), ",".join(item["fields"])),
        ),
        "counts": {
            "selected_rows": len(selected),
            "talk_spine_items": len(items),
            "non_talk_rows": len(non_talk_rows),
        },
        "allowed_hosts": list(CVPR_YOUTUBE_ALLOWED_HOSTS),
    }
    payload["program_spine_sha256"] = canonical_program_spine_digest(payload)
    return payload


def _workspace_member(workspace: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else workspace / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(workspace.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def _cvpr_item_page_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.hostname not in _VIRTUAL_SITE_HOSTS:
        return None
    return sanitize_url(value)


def assess_cached_cvpr_slideslive_recovery(
    *, workspace: Path, ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Inspect cached CVPR item-page probes only; never create a probe or fetch.

    The result does not make SlidesLive a CVPR allowed host.  It merely records
    whether existing, local CVPR probe evidence requires the later focused
    cached-page recovery branch.
    """

    artifact_rows = ledger.get("artifacts")
    if not isinstance(artifact_rows, list):
        artifact_rows = ledger.get("local_artifacts")
    if not isinstance(artifact_rows, list):
        artifact_rows = []
    inspected = 0
    embedded_assets: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for artifact in artifact_rows:
        if not isinstance(artifact, Mapping):
            continue
        if artifact.get("artifact_kind") != "item_page_auth_probe" or artifact.get("status") != "succeeded":
            continue
        local_path = artifact.get("local_path")
        path = _workspace_member(workspace, local_path)
        page_url = _cvpr_item_page_url(artifact.get("final_url") or artifact.get("source_url"))
        if path is None or page_url is None or not path.is_file():
            continue
        path_key = path.relative_to(workspace.resolve()).as_posix()
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        inspected += 1
        slideslive_id = extract_slideslive_id(path.read_text(encoding="utf-8", errors="replace"))
        if slideslive_id is None:
            continue
        embedded_assets.append(
            {
                "program_item_id": artifact.get("program_item_id"),
                "source_record_id": artifact.get("source_record_id"),
                "page_artifact_path": path_key,
                "page_url": page_url,
                "slideslive_id": slideslive_id,
                "slideslive_url": f"https://slideslive.com/{slideslive_id}",
            }
        )
    embedded_assets.sort(
        key=lambda item: (
            str(item.get("program_item_id") or ""),
            str(item.get("page_artifact_path") or ""),
        )
    )
    if embedded_assets:
        return {
            "status": "eligible_cached_recovery",
            "reason": "cached_cvpr_item_page_probe_exposes_slideslive_embed",
            "inspected_probe_count": inspected,
            "embedded_assets": embedded_assets,
            "allowed_hosts": list(CVPR_YOUTUBE_ALLOWED_HOSTS),
            "probes_created": False,
        }
    return {
        "status": "not_applicable",
        "reason": "no_cached_cvpr_item_page_auth_probe_with_slideslive_embed",
        "inspected_probe_count": inspected,
        "embedded_assets": [],
        "allowed_hosts": list(CVPR_YOUTUBE_ALLOWED_HOSTS),
        "probes_created": False,
    }


# Capability names for the CVPR-era entry points.
write_program_spine = write_cvpr_program_spine
build_program_spine = build_cvpr_program_spine
assess_cached_slideslive_recovery = assess_cached_cvpr_slideslive_recovery


# --------------------------------------------------------------------------
# Stage 4 lanes on a program-item ledger
# --------------------------------------------------------------------------


def build_pinned_program_spine(
    *,
    workspace: Path,
    run_id: str,
    pinned_run_id: str,
    pinned_inventory_sha256: str,
    corpus_manifest: Mapping[str, Any],
    ledger: dict[str, Any],
    validate_source_lineage: Any,
    blocker: Any,
    spine_path: Path = CVPR_PROGRAM_SPINE_PATH,
    allowed_hosts: Sequence[str] = CVPR_YOUTUBE_ALLOWED_HOSTS,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Build the Stage-4 talk control file from the already-validated pinned lineage.

    ``validate_source_lineage(*, workspace, run_id, pinned_run_id,
    pinned_inventory_sha256)`` returns ``(inventory, records_by_id,
    blockers)``; ``blocker(code, **details)`` builds a Stage 4 blocker.
    """

    inventory, records_by_id, source_blockers = validate_source_lineage(
        workspace=workspace,
        run_id=run_id,
        pinned_run_id=pinned_run_id,
        pinned_inventory_sha256=pinned_inventory_sha256,
    )
    blockers = [
        blocker(str(item.get("code") or "program_spine_source_lineage_invalid")) for item in source_blockers
    ]
    if inventory is None or blockers:
        return None, blockers
    try:
        spine = build_cvpr_program_spine(
            selection_rows=corpus_manifest["program_items"],
            source_records=list(records_by_id.values()),
            lineage={
                "run_id": run_id,
                "phase_02_inventory_sha256": pinned_inventory_sha256,
                "phase_03_selection_sha256": corpus_manifest["selection_sha256"],
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, [blocker("program_spine_build_failed", reason=type(exc).__name__)]
    write_cvpr_program_spine(workspace / spine_path, spine)
    ledger["talk_harvest"] = {
        "program_spine_path": str(spine_path),
        "program_spine_sha256": spine["program_spine_sha256"],
        "allowed_hosts": list(allowed_hosts),
        "status": "prepared",
    }
    return spine, []


FOCUSED_RECOVERY_LANE_ID = "cached-slideslive-focused-recovery"
FOCUSED_RECOVERY_INCOMPLETE = "focused_recovery_lane_incomplete"


def focused_recovery_lane(*, requires: Any, blocker: Any) -> Any:
    """Assess cached item-page probes for SlidesLive embeds after the talk lane.

    Runs only when ``requires(state)`` (the talk lane was attempted). A
    cached embed is a blocker that routes to the focused recovery owner;
    ``not_applicable`` completes the lane.
    """

    from conference_pipeline.production.hooks import AcquisitionLane, LaneResult

    def run(ctx: Any, _frozen: Any, _prior: Any) -> Any:
        state = ctx.state
        if not requires(state):
            return LaneResult(completed=False)
        focused_recovery = assess_cached_cvpr_slideslive_recovery(workspace=ctx.workspace, ledger=state.ledger)
        state.ledger["focused_recovery"] = focused_recovery
        if focused_recovery["status"] == "not_applicable":
            return LaneResult(receipt_fields={"focused_recovery_disposition": focused_recovery["status"]})
        return LaneResult(
            blockers=[
                blocker(
                    "cached_slideslive_embed_requires_focused_recovery",
                    embed_count=len(focused_recovery["embedded_assets"]),
                )
            ],
            receipt_fields={"focused_recovery_disposition": focused_recovery["status"]},
            completed=False,
        )

    return AcquisitionLane(
        lane_id=FOCUSED_RECOVERY_LANE_ID, run=run, incomplete_blocker=FOCUSED_RECOVERY_INCOMPLETE
    )
