"""Organizer-channel YouTube talk harvest as a Stage 4 acquisition lane.

Recommended for
---------------
``RECOMMENDED_FOR = ("CVPR",)``.

* **CVPR stage and quirk covered:** Stage 4 ``download-corpus``, the r3
  "organizer-channel talks" lane. CVPR does not publish talk recordings on
  CVF or SlidesLive; keynotes, invited talks and orals live on organizer
  YouTube channels (the CVF channel plus workshop organizers). This lane
  derives the talk spine from the pinned program, runs the generic
  :func:`conference_pipeline.acquisition.youtube_talks.harvest_youtube_talks`
  owner public-only (no Tavily, no caps, captions only), binds its matches and
  gaps to the Stage 4 program-item ledger and derives the exact
  priority-gap frontier the paid :mod:`talk_gap_search` lane may escalate.
* **May also fit:** any conference that posts talks on public organizer
  YouTube channels (ICCV, WACV, ECCV, workshop-heavy venues).

Conference identity (label, channels, broad queries, allowed hosts, the
forbidden-flag list) comes from :class:`TalkLaneSettings`; this module carries
no conference literal.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.youtube_talks import HarvestConfig, ProgramItem
from conference_pipeline.production.download_corpus_stage import ProgramLedgerState, stage4_blocker
from conference_pipeline.production.hooks import AcquisitionLane, LaneResult
from conference_pipeline.production_preflight import canonical_json_digest
from conference_pipeline.provider_receipts import sanitize_for_receipt

RECOMMENDED_FOR = ("CVPR",)

LANE_ID = "organizer-channel-talks"
INCOMPLETE_BLOCKER = "talk_harvest_lane_incomplete"
PRIORITY_GAP_PENDING = "priority_talk_gaps_pending_governed_tavily_escalation"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIORITY_FAMILIES = ("keynote", "invited", "oral")


@dataclass(frozen=True)
class TalkLaneSettings:
    """A conference's organizer-channel talk harvest inputs."""

    conference_label: str
    sources: tuple[str, ...]
    broad_queries: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    forbidden_flags: tuple[str, ...]
    priority_families: tuple[str, ...] = DEFAULT_PRIORITY_FAMILIES
    max_results_per_query: int = 5


def forbidden_flag_hits(command_args: Sequence[str], forbidden_flags: Sequence[str]) -> list[str]:
    """Talk-lane caps and paid-search switches present on the command line."""

    return sorted(
        {
            flag
            for flag in forbidden_flags
            if any(argument == flag or argument.startswith(f"{flag}=") for argument in command_args)
        }
    )


def audit_yt_dlp_prerequisite(
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    command_resolver: Callable[[str], str | None] = shutil.which,
    module_finder: Callable[[str], Any] = importlib_util.find_spec,
    version_resolver: Callable[[str], str | None] = importlib_metadata.version,
) -> dict[str, Any]:
    """Audit the already-declared yt-dlp dependency without installing it.

    The return value deliberately contains only dependency declaration,
    resolver, and version facts.  The caller records a lane blocker instead of
    attempting a package installation when the active environment is not ready.
    """

    def declared(path: Path) -> bool:
        try:
            return "yt-dlp" in path.read_text(encoding="utf-8")
        except OSError:
            return False

    declaration = {
        "pyproject": declared(repo_root / "pyproject.toml"),
        "requirements": declared(repo_root / "requirements.txt"),
    }
    try:
        command_path = command_resolver("yt-dlp")
    except (OSError, TypeError):
        command_path = None
    try:
        module_resolved = module_finder("yt_dlp") is not None
    except (ImportError, ModuleNotFoundError, ValueError, TypeError):
        module_resolved = False
    try:
        installed_version = version_resolver("yt-dlp")
    except (importlib_metadata.PackageNotFoundError, ImportError, ValueError, TypeError):
        installed_version = None
    if not isinstance(installed_version, str) or not installed_version.strip():
        installed_version = None
    resolution = {
        "command": "yt-dlp",
        "command_path": str(command_path) if command_path else None,
        "module": "yt_dlp",
        "module_resolved": module_resolved,
        "installed_version": installed_version,
    }
    blockers: list[dict[str, str]] = []
    if not declaration["pyproject"]:
        blockers.append({"code": "yt_dlp_not_declared_in_pyproject"})
    if not declaration["requirements"]:
        blockers.append({"code": "yt_dlp_not_declared_in_requirements"})
    if not command_path and not module_resolved:
        blockers.append({"code": "yt_dlp_unavailable"})
    elif installed_version is None:
        blockers.append({"code": "yt_dlp_version_unavailable"})
    return {
        "status": "pass" if not blockers else "blocked",
        "declaration": declaration,
        "resolution": resolution,
        "blockers": blockers,
        "installation_action": "not_attempted",
    }


def merge_talk_harvest_into_ledger(*, ledger: dict[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Bind existing generic-harvest facts to immutable Stage-4 item rows."""

    talks = receipt.get("talks")
    gaps = receipt.get("gaps")
    talks_by_id = {
        str(row.get("item_id")): dict(row)
        for row in talks
        if isinstance(row, Mapping) and isinstance(row.get("item_id"), str)
    } if isinstance(talks, list) else {}
    gaps_by_id = {
        str(row.get("item_id")): dict(row)
        for row in gaps
        if isinstance(row, Mapping) and isinstance(row.get("item_id"), str)
    } if isinstance(gaps, list) else {}
    program_results = ledger.get("program_item_results")
    if isinstance(program_results, list):
        for row in program_results:
            if not isinstance(row, dict):
                continue
            item_id = row.get("program_item_id")
            if item_id in talks_by_id:
                talk = talks_by_id[str(item_id)]
                row["talk_media"] = {
                    "status": "matched",
                    "youtube_video_id": talk.get("youtube_video_id"),
                    "match_direction": talk.get("match_direction"),
                    "match_confidence": talk.get("match_confidence"),
                    "match_method": talk.get("match_method"),
                    "transcript": sanitize_for_receipt(talk.get("transcript")),
                    "audio_downloaded": False,
                    "video_downloaded": False,
                }
            elif item_id in gaps_by_id:
                gap = gaps_by_id[str(item_id)]
                row["talk_media"] = {
                    "status": "gap",
                    "family": gap.get("family"),
                    "must_investigate": bool(gap.get("must_investigate")),
                    "audio_downloaded": False,
                    "video_downloaded": False,
                }
    return {
        "matched_talk_count": len(talks_by_id),
        "gap_count": len(gaps_by_id),
        "priority_gap_count": sum(1 for gap in gaps_by_id.values() if gap.get("must_investigate")),
    }


def build_priority_gap_frontier(
    *,
    spine: Mapping[str, Any],
    harvest_inventory: Mapping[str, Any],
    settings: TalkLaneSettings,
) -> dict[str, Any]:
    """Derive the exact priority-only paid-search frontier from public facts."""

    raw_gaps = harvest_inventory.get("gaps")
    if not isinstance(raw_gaps, list):
        raise ValueError("public harvest inventory has no gap list")
    ids = sorted(
        str(gap["item_id"])
        for gap in raw_gaps
        if isinstance(gap, Mapping)
        and gap.get("must_investigate") is True
        and isinstance(gap.get("item_id"), str)
        and gap.get("item_id")
    )
    if len(ids) != len(set(ids)):
        raise ValueError("public harvest priority gaps are duplicated")
    query_policy = {
        "conference_label": settings.conference_label,
        "allowed_hosts": list(settings.allowed_hosts),
        "query_template": 'quoted_title + optional_first_speaker + conference_label',
        "priority_families_only": True,
        "max_results_per_query": settings.max_results_per_query,
    }
    core = {
        "program_spine_sha256": spine.get("program_spine_sha256"),
        "harvest_inventory_sha256": canonical_json_digest(harvest_inventory),
        "query_policy_sha256": canonical_json_digest(query_policy),
        "priority_program_item_ids": ids,
        "priority_gap_quantity": len(ids),
    }
    if not all(isinstance(core[key], str) and core[key] for key in core if key.endswith("sha256")):
        raise ValueError("public harvest priority-gap lineage is incomplete")
    return {
        **core,
        "query_policy": query_policy,
        "priority_gap_frontier_sha256": canonical_json_digest(core),
    }


def run_program_talk_lane(
    *,
    workspace: Path,
    ledger: dict[str, Any],
    corpus_manifest: Mapping[str, Any],
    run_id: str,
    pinned_run_id: str,
    pinned_inventory_sha256: str,
    command_args: Sequence[str],
    harvester: Callable[[HarvestConfig], Mapping[str, Any]],
    prerequisite_audit: Callable[[], Mapping[str, Any]],
    settings: TalkLaneSettings,
    build_spine: Callable[..., tuple[dict[str, Any] | None, list[dict[str, Any]]]],
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    """Run only the source-bound, no-Tavily public YouTube lane.

    The generic harvester remains the single receipt owner.  This adapter only
    constrains its inputs and carries its existing facts into the Stage-4
    ledger. ``build_spine(**kwargs)`` derives the talk spine from the pinned
    lineage and records it on the ledger. Returns ``(completed, outcome,
    blockers)``.
    """

    blockers: list[dict[str, Any]] = []
    forbidden = forbidden_flag_hits(command_args, settings.forbidden_flags)
    if forbidden:
        return False, {"status": "blocked", "forbidden_flags": forbidden}, [
            stage4_blocker("public_talk_lane_forbidden_flags", flags=forbidden)
        ]
    spine, spine_blockers = build_spine(
        workspace=workspace,
        run_id=run_id,
        pinned_run_id=pinned_run_id,
        pinned_inventory_sha256=pinned_inventory_sha256,
        corpus_manifest=corpus_manifest,
        ledger=ledger,
    )
    blockers.extend(spine_blockers)
    if spine is None:
        return False, {"status": "blocked"}, blockers
    audit = sanitize_for_receipt(dict(prerequisite_audit()))
    talk_harvest = ledger.setdefault("talk_harvest", {})
    if isinstance(talk_harvest, dict):
        talk_harvest["yt_dlp_prerequisite"] = audit
    if audit.get("status") != "pass":
        blockers.extend(
            stage4_blocker(str(row.get("code") or "yt_dlp_unavailable"))
            for row in audit.get("blockers", [])
            if isinstance(row, Mapping)
        )
        return False, {"status": "blocked", "yt_dlp_prerequisite": audit}, blockers
    full_spine_items = [ProgramItem.from_mapping(item, index) for index, item in enumerate(spine["items"])]
    # The Stage-4 authority permits a full source-bound spine for evidence,
    # but targeted public YouTube search is deliberately limited to families
    # expected to have conference recordings.  In particular, a poster page
    # is a valid program record, not authority to issue thousands of unrelated
    # ``ytsearch`` requests.  Its later absence stays visible in the full
    # spine, while the exact paid frontier can contain only priority rows.
    priority_spine_items = [item for item in full_spine_items if item.is_priority_family]
    public_search_scope = {
        "mode": "priority_families_only",
        "full_program_spine_item_count": len(full_spine_items),
        "targeted_search_item_count": len(priority_spine_items),
        "deferred_nonpriority_item_count": len(full_spine_items) - len(priority_spine_items),
        "priority_families": list(settings.priority_families),
    }
    talk_harvest = ledger.setdefault("talk_harvest", {})
    if isinstance(talk_harvest, dict):
        talk_harvest["public_search_scope"] = public_search_scope
    config = HarvestConfig(
        workspace=workspace,
        conference_label=settings.conference_label,
        program_spine=priority_spine_items,
        sources=settings.sources,
        broad_queries=settings.broad_queries,
        allowed_hosts=settings.allowed_hosts,
        per_item_search=True,
        tavily_api_key=None,
        transcripts_enabled=True,
        resume=True,
        pilot_limit=None,
        run_id=run_id,
    )
    try:
        receipt = dict(harvester(config))
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        return False, {"status": "blocked"}, [stage4_blocker("public_talk_harvest_failed", reason=type(exc).__name__)]
    boundaries = receipt.get("boundaries")
    receipt_hosts = receipt.get("allowed_hosts")
    valid = (
        receipt.get("status") in {"succeeded", "partial"}
        and receipt_hosts == list(settings.allowed_hosts)
        and isinstance(boundaries, Mapping)
        and boundaries.get("audio_downloaded") is False
        and boundaries.get("video_downloaded") is False
        and receipt.get("pilot_limit") is None
        and isinstance(receipt.get("tavily_lane"), Mapping)
        and receipt["tavily_lane"].get("enabled") is False
    )
    if not valid:
        return False, {"status": "blocked", "receipt": sanitize_for_receipt(receipt)}, [
            stage4_blocker("public_talk_harvest_contract_invalid")
        ]
    inventory_path_value = receipt.get("inventory_path")
    inventory_path = Path(str(inventory_path_value)) if inventory_path_value else None
    if inventory_path is not None and not inventory_path.is_absolute():
        inventory_path = workspace / inventory_path
    try:
        if inventory_path is None or not inventory_path.resolve().is_relative_to(workspace.resolve()):
            raise ValueError("harvest inventory path must stay in workspace")
        harvest_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False, {"status": "blocked"}, [stage4_blocker("public_talk_harvest_inventory_unavailable")]
    if not isinstance(harvest_inventory, Mapping):
        return False, {"status": "blocked"}, [stage4_blocker("public_talk_harvest_inventory_invalid")]
    merged = merge_talk_harvest_into_ledger(ledger=ledger, receipt=harvest_inventory)
    try:
        priority_gap_frontier = build_priority_gap_frontier(
            spine=spine, harvest_inventory=harvest_inventory, settings=settings
        )
    except (TypeError, ValueError):
        return False, {"status": "blocked"}, [stage4_blocker("priority_gap_frontier_invalid")]
    receipt_path = receipt.get("receipt_path")
    talk_harvest = ledger.setdefault("talk_harvest", {})
    if isinstance(talk_harvest, dict):
        talk_harvest.update(
            {
                "status": "completed_public_only",
                "public_search_scope": public_search_scope,
                "harvest_receipt_path": str(receipt_path) if receipt_path else None,
                "harvest_receipt": sanitize_for_receipt(receipt),
                "priority_gap_frontier": priority_gap_frontier,
                **merged,
            }
        )
    if merged["priority_gap_count"]:
        blockers.append(stage4_blocker(PRIORITY_GAP_PENDING, count=merged["priority_gap_count"]))
    return True, {"status": str(receipt["status"]), **merged}, blockers


# Facts the talk lane leaves on the program-ledger state for later lanes.
TALK_LANE_ATTEMPTED = "talk_lane_attempted"
PRIORITY_GAP_COUNT = "priority_gap_count"


def talk_lane(
    *,
    enabled: bool,
    run_lane: Callable[..., tuple[bool, dict[str, Any], list[dict[str, Any]]]],
    pinned_run_id: str,
    pinned_inventory_sha256: str,
    command_args: Sequence[str],
    harvester: Callable[[HarvestConfig], Mapping[str, Any]],
    prerequisite_audit: Callable[[], Mapping[str, Any]],
    requires_lane_completed: Callable[[ProgramLedgerState], bool],
) -> AcquisitionLane:
    """The organizer-channel talk lane on a program-item ledger.

    It runs only after a complete public acquisition
    (``requires_lane_completed(state)``); ``run_lane(**kwargs)`` is the
    orchestrator's bound :func:`run_program_talk_lane`. A zero priority-gap
    outcome records the paid escalation as not required.
    """

    def run(ctx: Any, frozen_corpus: Mapping[str, Any] | None, _prior: Any) -> LaneResult:
        state: ProgramLedgerState = ctx.state
        if not enabled:
            return LaneResult(completed=False)
        if not requires_lane_completed(state):
            return LaneResult(
                blockers=[stage4_blocker("talk_harvest_requires_complete_public_acquisition")], completed=False
            )
        if frozen_corpus is None:
            return LaneResult(blockers=[stage4_blocker("talk_harvest_corpus_manifest_unavailable")], completed=False)
        state.facts[TALK_LANE_ATTEMPTED] = True
        completed, outcome, blockers = run_lane(
            workspace=ctx.workspace,
            ledger=state.ledger,
            corpus_manifest=frozen_corpus,
            run_id=ctx.run_id,
            pinned_run_id=pinned_run_id,
            pinned_inventory_sha256=pinned_inventory_sha256,
            command_args=command_args,
            harvester=harvester,
            prerequisite_audit=prerequisite_audit,
        )
        priority_gap_count = int(outcome.get("priority_gap_count") or 0)
        state.facts[PRIORITY_GAP_COUNT] = priority_gap_count
        if priority_gap_count == 0:
            state.ledger.setdefault("talk_harvest", {}).update(
                {"tavily_lane": {"status": "not_required", "provider_calls_made": 0}}
            )
        return LaneResult(
            blockers=list(blockers),
            receipt_fields={"talk_harvest_lane_status": outcome},
            completed=bool(completed),
        )

    return AcquisitionLane(lane_id=LANE_ID, run=run, incomplete_blocker=INCOMPLETE_BLOCKER)
