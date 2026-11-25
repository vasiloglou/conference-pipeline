"""Governed paid web search for priority talk gaps (the Tavily priority-gap lane).

After the public organizer-channel harvest
(:mod:`conference_pipeline.acquisition.youtube_channel_harvest`) a few
priority talks (keynotes, invited talks, orals) may still have no public
recording match. This lane escalates *exactly* that frontier to a paid search
provider, and nothing else:

* the frontier is read from the durable Stage 4 ledger (never an in-memory
  object) and digest-checked;
* an immutable, frontier-scoped spend authority is prepared from the current
  named full-run estimate and repriced for the canonical cost task
  (:func:`prepare_authority`); a changed frontier supersedes it;
* every search call passes a fresh zero-spend preflight and the pre-spend
  contract, and is recorded through ``record_provider_call``;
* the outcome becomes final per-item talk evidence on the ledger
  (:func:`finalize_governed_outcome`).

Recommended for
---------------
``RECOMMENDED_FOR = ("CVPR",)``.

* **CVPR stage and quirk covered:** Stage 4 ``download-corpus`` r3, the
  "Tavily priority gap" lane: CVPR keynote/oral recordings missing from the
  public organizer channels.
* **May also fit:** any conference using the organizer-channel talk lane
  whose priority talks are partly missing from public channels.

Conference identity (cost/policy task ids, authority kind, query label, stop
tokens, provider caller) comes from :class:`TalkGapSearchSettings`; this
module carries no conference literal.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.program_spine import CVPR_PROGRAM_SPINE_PATH, CVPR_YOUTUBE_ALLOWED_HOSTS
from conference_pipeline.acquisition.youtube_talks import ProgramItem, match_videos_to_spine, tavily_search_videos
from conference_pipeline.costs.estimator import estimate_workspace
from conference_pipeline.production import preflight_stage
from conference_pipeline.production.download_corpus_stage import (
    DOWNLOAD_CORPUS_INVENTORY,
    ProgramLedgerState,
    stage4_blocker,
)
from conference_pipeline.production.hooks import AcquisitionLane, LaneResult
from conference_pipeline.production.presentation_steps import file_sha256
from conference_pipeline.production_preflight import AUTOMATIC_AUTHORIZATION_THRESHOLD_USD, canonical_json_digest
from conference_pipeline.provider_receipts import provider_receipt_context, record_provider_call, sanitize_for_receipt
from conference_pipeline.stage_contracts import verify_tavily_priority_gap_pre_spend_contract
from conference_pipeline.workflow_runs import validate_run_id

RECOMMENDED_FOR = ("CVPR",)

LANE_ID = "priority-talk-gap-search"
STAGE_ID = "download-corpus"
PROVIDER = "tavily"
PRIORITY_GAP_PENDING = "priority_talk_gaps_pending_governed_tavily_escalation"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
AUTOMATIC_MODE = "automatic_under_usd_800"


@dataclass(frozen=True)
class TalkGapSearchSettings:
    """A conference's governed priority-gap search identity.

    ``now()`` and ``write_json(path, payload)`` are the orchestrator's own
    timestamp source and JSON writer, bound at call time.
    """

    cost_task_id: str
    policy_task_id: str
    authority_filename: str
    authority_kind: str
    conference_label: str
    match_stop_tokens: tuple[str, ...]
    provider_caller: str
    now: Callable[[], str]
    write_json: Callable[[Path, dict[str, Any]], None]
    allowed_hosts: tuple[str, ...] = CVPR_YOUTUBE_ALLOWED_HOSTS
    spine_path: Path = CVPR_PROGRAM_SPINE_PATH
    ledger_path: Path = DOWNLOAD_CORPUS_INVENTORY
    prepared_estimate_filename: str = preflight_stage.PREPARED_ESTIMATE_FILENAME
    repo_root: Path = DEFAULT_REPO_ROOT


def load_priority_gap_frontier(workspace: Path, run_id: str, *, ledger_path: Path = DOWNLOAD_CORPUS_INVENTORY) -> dict[str, Any]:
    """Load the exact public-harvest priority frontier without a paid fallback.

    The public lane persists this non-secret control record.  A missing record
    is not interpreted as "search everything"; the caller must keep the stage
    blocked until public harvest has supplied an auditable answer.
    """

    try:
        ledger = json.loads((workspace / ledger_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Tavily priority-gap ledger is required") from exc
    talk_harvest = ledger.get("talk_harvest") if isinstance(ledger, Mapping) else None
    frontier = talk_harvest.get("priority_gap_frontier") if isinstance(talk_harvest, Mapping) else None
    if not isinstance(frontier, Mapping):
        raise ValueError("Tavily priority-gap frontier is required")
    required = {
        "program_spine_sha256",
        "harvest_inventory_sha256",
        "query_policy_sha256",
        "priority_program_item_ids",
        "priority_gap_quantity",
        "priority_gap_frontier_sha256",
    }
    if not required.issubset(frontier):
        raise ValueError("Tavily priority-gap frontier is incomplete")
    item_ids = frontier.get("priority_program_item_ids")
    quantity = frontier.get("priority_gap_quantity")
    if (
        not isinstance(item_ids, list)
        or not all(isinstance(item_id, str) and item_id for item_id in item_ids)
        or item_ids != sorted(item_ids)
        or len(set(item_ids)) != len(item_ids)
        or not isinstance(quantity, int)
        or isinstance(quantity, bool)
        or quantity != len(item_ids)
        or quantity < 0
    ):
        raise ValueError("Tavily priority-gap frontier quantity is incompatible")
    core = {
        key: frontier[key]
        for key in (
            "program_spine_sha256",
            "harvest_inventory_sha256",
            "query_policy_sha256",
            "priority_program_item_ids",
            "priority_gap_quantity",
        )
    }
    if frontier.get("priority_gap_frontier_sha256") != canonical_json_digest(core):
        raise ValueError("Tavily priority-gap frontier digest is incompatible")
    if run_id != str((ledger.get("lineage") or {}).get("run_id") or run_id):
        raise ValueError("Tavily priority-gap frontier run lineage is incompatible")
    return dict(frontier)


def checkpoint_frontier(
    *,
    workspace: Path,
    ledger: dict[str, Any],
    ledger_path: Path = DOWNLOAD_CORPUS_INVENTORY,
    write_json: Callable[[Path, dict[str, Any]], None],
) -> None:
    """Persist the exact public-harvest frontier before the paid lane reads it.

    The governed authority builder intentionally reads the durable Stage-4
    ledger, never a caller's in-memory object; checkpointing the just-produced
    public-only frontier first makes authority, pre-spend validation, and the
    eventual provider receipt bind the same immutable scope.
    """

    from conference_pipeline.production.download_corpus_stage import canonical_ledger_digest

    ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
    write_json(workspace / ledger_path, ledger)


def priority_gap_estimate(
    workspace: Path,
    *,
    priority_gap_quantity: int,
    cost_task_id: str,
    estimate: Callable[..., Any] = estimate_workspace,
) -> dict[str, Any]:
    """Reprice the one canonical paid-search task for the current exact frontier."""

    report = preflight_stage.as_dict(estimate(workspace, scope="full", mode="estimate"))
    tasks = report.get("tasks")
    matches = [
        task for task in tasks
        if isinstance(task, Mapping) and task.get("task_id") == cost_task_id
    ] if isinstance(tasks, list) else []
    if len(matches) != 1:
        raise ValueError("canonical Tavily priority-gap pricing task is missing")
    task = dict(matches[0])
    if (
        priority_gap_quantity <= 0
        or task.get("enabled") is not True
        or task.get("quantity") != priority_gap_quantity
        or task.get("provider") != PROVIDER
        or task.get("stage") != STAGE_ID
        or not isinstance(task.get("model"), str)
        or not task["model"]
        or not isinstance(task.get("estimated_cost_usd"), (int, float))
        or not math.isfinite(float(task["estimated_cost_usd"]))
        or float(task["estimated_cost_usd"]) < 0
    ):
        raise ValueError("canonical Tavily priority-gap pricing is incomplete or incompatible")
    return task


def authority_core(workspace: Path, run_id: str, *, settings: TalkGapSearchSettings) -> dict[str, Any]:
    """Build immutable provider authority for only the current priority gaps."""

    safe_run_id = validate_run_id(run_id)
    resolved_workspace = workspace.resolve()
    base_snapshot, _prepared, base_blockers = preflight_stage.load_prepared_estimate(
        preflight_stage.named_run_dir(resolved_workspace, safe_run_id) / settings.prepared_estimate_filename,
        workspace=resolved_workspace,
        run_id=safe_run_id,
    )
    if base_blockers or base_snapshot is None:
        raise ValueError("a current named full-run estimate is required before Tavily authority")
    frontier = load_priority_gap_frontier(resolved_workspace, safe_run_id, ledger_path=settings.ledger_path)
    quantity = int(frontier["priority_gap_quantity"])
    if quantity <= 0:
        raise ValueError("Tavily authority is not required for a zero priority-gap frontier")
    task = priority_gap_estimate(resolved_workspace, priority_gap_quantity=quantity, cost_task_id=settings.cost_task_id)
    cost = float(task["estimated_cost_usd"])
    allocations = [{"provider": PROVIDER, "stage": STAGE_ID, "ceiling_usd": cost}]
    base_projection = dict(base_snapshot.provider_stage_costs)
    base_tavily_cost = float(base_projection.get(f"{PROVIDER}:{STAGE_ID}", 0.0))
    if base_tavily_cost not in {0.0, cost}:
        raise ValueError("base estimate Tavily projection is incompatible with the priority-gap frontier")
    total_ceiling = round(
        sum(cost for _key, cost in base_snapshot.provider_stage_costs)
        + (0.0 if base_tavily_cost else cost)
        + base_snapshot.approved_headroom_usd,
        6,
    )
    automatic = math.isfinite(total_ceiling) and 0 <= total_ceiling <= AUTOMATIC_AUTHORIZATION_THRESHOLD_USD
    repo_root = settings.repo_root
    task_specs_path = repo_root / "src" / "conference_pipeline" / "costs" / "task_specs.yaml"
    pricing_catalog_path = repo_root / "src" / "conference_pipeline" / "costs" / "pricing_catalog.json"
    model_registry_path = repo_root / "src" / "conference_pipeline" / "model_governance" / "registry.yaml"
    return {
        "schema_version": 1,
        "run_id": safe_run_id,
        "workspace": str(resolved_workspace),
        "authority_kind": settings.authority_kind,
        "base_full_estimate_digest": base_snapshot.digest,
        "program_spine_sha256": frontier["program_spine_sha256"],
        "harvest_inventory_sha256": frontier["harvest_inventory_sha256"],
        "query_policy_sha256": frontier["query_policy_sha256"],
        "priority_gap_frontier_sha256": frontier["priority_gap_frontier_sha256"],
        "priority_gap_quantity": quantity,
        "priority_program_item_ids": list(frontier["priority_program_item_ids"]),
        "task_id": settings.policy_task_id,
        "cost_task_id": settings.cost_task_id,
        "task_estimate_digest": canonical_json_digest(task),
        "task_specs_sha256": file_sha256(task_specs_path),
        "pricing_catalog_sha256": file_sha256(pricing_catalog_path),
        # The no-model provider still binds the current governance registry as
        # a drift invalidator.  It does not manufacture a Tavily model task.
        "model_registry_sha256": file_sha256(model_registry_path),
        "provider": PROVIDER,
        "pricing_model": task["model"],
        "stage": STAGE_ID,
        "provider_stage_allocations": allocations,
        "base_tavily_stage_projection_usd": base_tavily_cost,
        "frontier_estimated_cost_usd": cost,
        "full_run_total_ceiling_usd": total_ceiling,
        "spend_authority": (
            {
                "mode": AUTOMATIC_MODE,
                "threshold_usd": AUTOMATIC_AUTHORIZATION_THRESHOLD_USD,
                "total_ceiling_usd": total_ceiling,
                "provider_stage_allocations": allocations,
            }
            if automatic
            else {"mode": "explicit_immutable_approval_required"}
        ),
        "fresh_volatile_preflight_required": True,
    }


def prepare_authority(workspace: Path, run_id: str, *, settings: TalkGapSearchSettings) -> dict[str, Any]:
    """Persist a new boundary, preserving superseded evidence if scope drifts."""

    core = authority_core(workspace, run_id, settings=settings)
    target = preflight_stage.named_run_dir(workspace, run_id)
    path = target / settings.authority_filename
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        existing = candidate if isinstance(candidate, dict) else None
    except (OSError, json.JSONDecodeError):
        existing = None
    digest = canonical_json_digest(core)
    if existing is not None and existing.get("authority_digest") == digest:
        return existing
    payload = {**core, "authority_digest": digest, "prepared_at": settings.now()}
    if existing is not None and isinstance(existing.get("authority_digest"), str):
        prior = str(existing["authority_digest"])
        superseded = target / f"{Path(settings.authority_filename).stem}.superseded-{prior[:12]}.json"
        if not superseded.exists():
            settings.write_json(superseded, existing)
        payload["supersedes_authority_digest"] = prior
        payload["superseded_authority_path"] = str(superseded)
    settings.write_json(path, payload)
    return payload


def validate_authority(
    workspace: Path, run_id: str, authority_path: Path, *, settings: TalkGapSearchSettings
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Rebuild all mutable inputs before a preflight or provider call."""

    incompatible = [{"code": "tavily_priority_gap_authority_incompatible", "affected_stage": STAGE_ID}]
    try:
        payload = json.loads(authority_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("authority payload must be an object")
        expected = authority_core(workspace, run_id, settings=settings)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, incompatible
    if payload.get("authority_digest") != canonical_json_digest(expected):
        return None, incompatible
    if any(payload.get(key) != value for key, value in expected.items()):
        return None, incompatible
    return payload, []


def run_governed_search(
    *,
    workspace: Path,
    run_id: str,
    authority_path: Path,
    api_key: str | None,
    http_post: Callable[[str, dict[str, Any]], dict[str, Any]] | None,
    fresh_volatile_preflight: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    settings: TalkGapSearchSettings,
) -> dict[str, Any]:
    """Execute only authority-bound priority gap searches with canonical receipts.

    The function is intentionally an injected production boundary: the generic
    harvester CLI cannot call it, an environment key is not read here, and a
    missing authority/preflight/client seam produces an auditable zero-call
    blocker.  Every actual POST is wrapped by ``record_provider_call`` instead
    of creating another receipt format.
    """

    safe_run_id = validate_run_id(run_id)
    resolved_workspace = workspace.resolve()
    try:
        frontier = load_priority_gap_frontier(resolved_workspace, safe_run_id, ledger_path=settings.ledger_path)
    except ValueError:
        return {"status": "blocked", "provider_calls_made": 0, "blockers": [{"code": "priority_gap_frontier_invalid"}]}
    if int(frontier["priority_gap_quantity"]) == 0:
        return {
            "status": "not_required",
            "provider_calls_made": 0,
            "actual_cost_usd": 0.0,
            "affected_priority_program_item_ids": [],
            "provider_receipt_paths": [],
        }
    authority_path = preflight_stage.require_named_run_input(
        authority_path, workspace=resolved_workspace, run_id=safe_run_id
    )
    authority, blockers = validate_authority(resolved_workspace, safe_run_id, authority_path, settings=settings)
    if authority is None:
        return {"status": "blocked", "provider_calls_made": 0, "blockers": blockers}
    if authority.get("spend_authority", {}).get("mode") != AUTOMATIC_MODE:
        return {
            "status": "blocked",
            "provider_calls_made": 0,
            "affected_priority_program_item_ids": list(frontier["priority_program_item_ids"]),
            "blockers": [{"code": "tavily_explicit_immutable_approval_required"}],
        }
    if not isinstance(api_key, str) or not api_key:
        return {
            "status": "blocked",
            "provider_calls_made": 0,
            "affected_priority_program_item_ids": list(frontier["priority_program_item_ids"]),
            "blockers": [{"code": "tavily_credential_unavailable"}],
        }
    if http_post is None or fresh_volatile_preflight is None:
        return {
            "status": "blocked",
            "provider_calls_made": 0,
            "affected_priority_program_item_ids": list(frontier["priority_program_item_ids"]),
            "blockers": [{"code": "tavily_governed_dispatch_dependencies_unavailable"}],
        }
    try:
        spine_payload = json.loads((resolved_workspace / settings.spine_path).read_text(encoding="utf-8"))
        rows = spine_payload.get("items") if isinstance(spine_payload, Mapping) else None
        items = [ProgramItem.from_mapping(row, index) for index, row in enumerate(rows or []) if isinstance(row, Mapping)]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"status": "blocked", "provider_calls_made": 0, "blockers": [{"code": "program_spine_unavailable"}]}
    by_id = {item.item_id: item for item in items}
    expected_ids = list(frontier["priority_program_item_ids"])
    if any(item_id not in by_id for item_id in expected_ids):
        return {"status": "blocked", "provider_calls_made": 0, "blockers": [{"code": "priority_gap_spine_mismatch"}]}

    calls: list[dict[str, Any]] = []
    remaining = set(expected_ids)
    model = str(authority["pricing_model"])
    for item_id in expected_ids:
        fresh = sanitize_for_receipt(dict(fresh_volatile_preflight(authority)))
        contract = verify_tavily_priority_gap_pre_spend_contract(
            workspace=resolved_workspace,
            run_id=safe_run_id,
            stage_id=STAGE_ID,
            mode="live",
            frontier_authority=authority,
            base_full_estimate_digest=str(authority["base_full_estimate_digest"]),
            program_spine_sha256=str(frontier["program_spine_sha256"]),
            harvest_inventory_sha256=str(frontier["harvest_inventory_sha256"]),
            query_policy_sha256=str(frontier["query_policy_sha256"]),
            priority_gap_frontier_sha256=str(frontier["priority_gap_frontier_sha256"]),
            priority_gap_quantity=int(frontier["priority_gap_quantity"]),
            provider=PROVIDER,
            provider_stage=STAGE_ID,
            volatile_preflight=fresh,
        )
        if contract.status != "precondition_passed":
            return {
                "status": "blocked",
                "provider_calls_made": len(calls),
                "actual_cost_usd": round(sum(float(call["actual_cost_usd"]) for call in calls), 6),
                "affected_priority_program_item_ids": sorted(remaining),
                "provider_receipt_paths": [call["provider_receipt_path"] for call in calls],
                "blockers": [{"code": "tavily_pre_spend_contract_unmet", "pre_spend": contract.to_dict()}],
            }
        item = by_id[item_id]
        speaker_part = f" {item.speakers[0]}" if item.speakers else ""
        query = f'"{item.title}"{speaker_part} {settings.conference_label}'.strip()
        per_call: list[Any] = []

        def receipt_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            with provider_receipt_context(
                workspace_root=resolved_workspace,
                run_id=safe_run_id,
                stage_id=STAGE_ID,
                caller=settings.provider_caller,
                task_id=settings.policy_task_id,
            ):
                result = record_provider_call(
                    lambda: http_post(url, payload),
                    provider=PROVIDER,
                    model=model,
                    operation="search",
                    caller=settings.provider_caller,
                    request_kwargs=payload,
                    credits=1.0,
                )
            per_call.append(result.receipt)
            return dict(result.response)

        try:
            videos = tavily_search_videos(
                query,
                api_key=api_key,
                http_post=receipt_post,
                allowed_hosts=settings.allowed_hosts,
            )
        except Exception as exc:  # pragma: no cover - record_provider_call persisted the failed receipt.
            return {
                "status": "blocked",
                "provider_calls_made": len(calls) + len(per_call),
                "affected_priority_program_item_ids": sorted(remaining),
                "blockers": [{"code": "tavily_search_failed", "reason": type(exc).__name__}],
            }
        if len(per_call) != 1:
            return {
                "status": "blocked",
                "provider_calls_made": len(calls),
                "affected_priority_program_item_ids": sorted(remaining),
                "blockers": [{"code": "tavily_provider_receipt_missing"}],
            }
        receipt = per_call[0]
        receipt_path = Path(str(receipt.receipt_path or ""))
        try:
            relative_receipt_path = str(receipt_path.resolve().relative_to(resolved_workspace))
        except ValueError:
            return {"status": "blocked", "provider_calls_made": len(calls), "blockers": [{"code": "tavily_provider_receipt_path_invalid"}]}
        item_matches = match_videos_to_spine(
            [item], videos, direction="program_to_search", stop_tokens=settings.match_stop_tokens
        )
        calls.append(
            {
                "program_item_id": item_id,
                "query_sha256": canonical_json_digest(query),
                "provider_receipt_path": relative_receipt_path,
                "provider_receipt_sha256": file_sha256(receipt_path),
                "actual_cost_usd": float(receipt.estimated_cost_usd or 0.0),
                "match": (
                    {
                        "youtube_video_id": item_matches[0].video.video_id,
                        "match_method": item_matches[0].method,
                        "match_confidence": item_matches[0].confidence,
                    }
                    if item_matches
                    else None
                ),
            }
        )
        remaining.remove(item_id)
    return {
        "status": "completed",
        "provider_calls_made": len(calls),
        "actual_cost_usd": round(sum(float(call["actual_cost_usd"]) for call in calls), 6),
        "affected_priority_program_item_ids": [],
        "provider_receipt_paths": [call["provider_receipt_path"] for call in calls],
        "calls": calls,
        "secrets_copied": False,
    }


def finalize_governed_outcome(
    *,
    ledger: dict[str, Any],
    governed_tavily: Mapping[str, Any],
    priority_gap_count: int,
) -> list[dict[str, Any]]:
    """Turn an authority-bound search result into final talk evidence.

    A failed or incomplete paid lane remains a Stage-4 blocker.  Conversely,
    a completed call for every exact priority-gap item can exhaust without a
    match: that is a final, receipt-backed per-item talk gap rather than a
    reason to pretend the paid lane itself is still pending.  This preserves
    ``must_investigate`` and the evidence needed for a later source expansion.
    """

    talk_harvest = ledger.setdefault("talk_harvest", {})
    if not isinstance(talk_harvest, dict):
        return [stage4_blocker("priority_talk_gaps_governed_tavily_blocked", count=priority_gap_count)]
    frontier = talk_harvest.get("priority_gap_frontier")
    expected_ids = list(frontier.get("priority_program_item_ids", [])) if isinstance(frontier, Mapping) else []
    expected_ids = [item_id for item_id in expected_ids if isinstance(item_id, str) and item_id]
    calls = governed_tavily.get("calls")
    valid_completed_outcome = (
        governed_tavily.get("status") == "completed"
        and priority_gap_count > 0
        and len(expected_ids) == priority_gap_count
        and isinstance(calls, list)
        and len(calls) == priority_gap_count
        and all(isinstance(call, Mapping) for call in calls)
        and {str(call.get("program_item_id")) for call in calls if isinstance(call, Mapping)} == set(expected_ids)
    )
    if not valid_completed_outcome:
        lane = sanitize_for_receipt(dict(governed_tavily))
        lane["status"] = "blocked"
        talk_harvest["tavily_lane"] = lane
        return [stage4_blocker("priority_talk_gaps_governed_tavily_blocked", count=priority_gap_count)]

    rows = ledger.get("program_item_results")
    rows_by_id = {
        str(row.get("program_item_id")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("program_item_id"), str)
    } if isinstance(rows, list) else {}
    if set(expected_ids) - set(rows_by_id):
        lane = sanitize_for_receipt(dict(governed_tavily))
        lane["status"] = "blocked"
        talk_harvest["tavily_lane"] = lane
        return [stage4_blocker("priority_talk_gaps_governed_tavily_blocked", count=priority_gap_count)]

    unmatched_count = 0
    assert isinstance(calls, list)
    for call in calls:
        assert isinstance(call, Mapping)  # Established by valid_completed_outcome.
        item_id = str(call["program_item_id"])
        row = rows_by_id[item_id]
        provider_evidence = {
            "provider_receipt_path": call.get("provider_receipt_path"),
            "provider_receipt_sha256": call.get("provider_receipt_sha256"),
            "query_sha256": call.get("query_sha256"),
            "actual_cost_usd": call.get("actual_cost_usd"),
        }
        match = call.get("match")
        if isinstance(match, Mapping):
            row["talk_media"] = {
                "status": "matched",
                "youtube_video_id": match.get("youtube_video_id"),
                "match_direction": "program_to_governed_tavily_search",
                "match_confidence": match.get("match_confidence"),
                "match_method": match.get("match_method"),
                "audio_downloaded": False,
                "video_downloaded": False,
                **sanitize_for_receipt(provider_evidence),
            }
            continue
        unmatched_count += 1
        row["talk_media"] = {
            "status": "unmatched_after_governed_tavily",
            "must_investigate": True,
            "final_blocker": {"code": "unmatched_after_governed_tavily", "affected_stage": STAGE_ID},
            "audio_downloaded": False,
            "video_downloaded": False,
            **sanitize_for_receipt(provider_evidence),
        }

    lane = sanitize_for_receipt(dict(governed_tavily))
    lane["governed_search_status"] = "completed"
    lane["status"] = "completed_exhausted" if unmatched_count else "completed"
    lane["unmatched_after_governed_tavily_count"] = unmatched_count
    talk_harvest["tavily_lane"] = lane
    talk_harvest["status"] = "completed_with_final_priority_gaps" if unmatched_count else "completed"
    return []


def http_post_json(url: str, payload: dict[str, Any], *, timeout: float = 60) -> dict[str, Any]:
    """POST one JSON search request (only ever called inside the governed receipt boundary)."""

    from urllib.request import Request, urlopen

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # nosec B310 - governed fixed search endpoint
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Tavily response must be an object")
    return decoded


def talk_gap_lane(
    *,
    requires_talk_lane: Callable[[ProgramLedgerState], bool],
    priority_gap_count: Callable[[ProgramLedgerState], int],
    prepare: Callable[[Path, str], Mapping[str, Any]],
    search: Callable[..., Mapping[str, Any]],
    finalize: Callable[..., list[dict[str, Any]]],
    authority_path: Callable[[Path, str], Path],
    api_key: str | None,
    http_post: Callable[[str, dict[str, Any]], dict[str, Any]] | None,
    fresh_preflight: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
) -> AcquisitionLane:
    """The paid priority-gap lane on a program-item ledger (``paid=True``).

    Runs after the talk lane, only for a non-empty priority frontier: it
    checkpoints the ledger so the authority binds exactly that frontier,
    prepares the authority, searches, and turns the outcome into final talk
    evidence; it resolves the talk lane's pending-escalation blocker.
    ``prepare``, ``search`` and ``finalize`` are the orchestrator's bound
    :func:`prepare_authority`, :func:`run_governed_search` and
    :func:`finalize_governed_outcome`.
    """

    def run(ctx: Any, _frozen: Any, _prior: Any) -> LaneResult:
        state: ProgramLedgerState = ctx.state
        count = priority_gap_count(state)
        if not requires_talk_lane(state) or count == 0:
            return LaneResult()
        authority: Mapping[str, Any] | None = None
        try:
            state.checkpoint_with_digest()
            authority = prepare(ctx.workspace, ctx.run_id)
            governed = search(
                workspace=ctx.workspace,
                run_id=ctx.run_id,
                authority_path=authority_path(ctx.workspace, ctx.run_id),
                api_key=api_key,
                http_post=http_post,
                fresh_volatile_preflight=fresh_preflight,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            governed = {
                "status": "blocked",
                "provider_calls_made": 0,
                "blockers": [{"code": "tavily_authority_preparation_failed", "reason": type(exc).__name__}],
            }
        state.ledger.setdefault("talk_harvest", {}).update(
            {"tavily_authority_digest": authority.get("authority_digest") if authority else None}
        )
        blockers = finalize(ledger=state.ledger, governed_tavily=governed, priority_gap_count=count)
        return LaneResult(
            blockers=list(blockers),
            receipt_fields={
                "tavily_priority_gap_lane": sanitize_for_receipt(
                    state.ledger.get("talk_harvest", {}).get("tavily_lane", {})
                )
            },
            resolves=(PRIORITY_GAP_PENDING,),
        )

    return AcquisitionLane(lane_id=LANE_ID, run=run, paid=True)
