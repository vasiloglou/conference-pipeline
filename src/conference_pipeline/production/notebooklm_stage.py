"""Stage 10.4: per-deck NotebookLM podcasts, shared by every conference runner.

Gates, does not orchestrate: (1) resume short-circuit over an already
succeeded, still-passing receipt; (2) input gate on the Stage 10 receipt
(succeeded + passing contract), the Stage 8 receipt, the analysis manifest,
the executive digests, and every in-scope deck's blueprint; (3) the pre-spend
lineage gate; (4) explicit live acknowledgement (quota-metered Google
account); (5) the owner module's auth preflight, then one notebook per deck
through ``conference_pipeline.podcast_notebooklm.run_stage``; (6) a
``--deck-ids`` canary scope is always ``blocked`` with
``notebooklm_podcast_deck_scope_filtered``, never succeeded; (7) an auth
failure halts, a quota failure halts *resumably*; (8) the final
``verify_stage_contract(strict_execution=True)`` binds the aggregate audio
sha256 as the owner's non-echoed execution evidence.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from conference_pipeline import podcast_notebooklm
from conference_pipeline.model_governance import default_model
from conference_pipeline.production.context import UPSTREAM_LINEAGE_NOT_FRESH, ProductionRunnerContext
from conference_pipeline.production.decks import presentation_deck_scope
from conference_pipeline.production.receipts import StageOutcome, resumed_outcome, stage_blocker, write_stage_receipt
from conference_pipeline.provider_receipts import provider_receipt_context, sanitize_for_receipt, utc_now
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)

STAGE_ID = "notebooklm-podcast"
TASK_ID = podcast_notebooklm.MODEL_TASK_ID
OWNER_ID = "podcast.notebooklm_audio_overview"
EXECUTIVE_DIGESTS_FILENAME = "executive_topic_digests.json"


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load a JSON object, returning ``(payload, error)`` -- never raises."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"expected JSON object, found {type(payload).__name__}"
    return payload, None


def required_input_paths(ctx: ProductionRunnerContext) -> dict[str, Path]:
    """Inputs the per-deck NotebookLM podcast owner gates on before any network call."""

    return {
        "stage10_receipt": ctx.receipt_path("presentation"),
        "stage8_receipt": ctx.receipt_path("qa-gap-fill-synthesis"),
        "analysis_manifest": ctx.workspace / ctx.analysis_manifest,
        "executive_digests": ctx.workspace / ctx.decks_dir / EXECUTIVE_DIGESTS_FILENAME,
    }


def _receipt(
    ctx: ProductionRunnerContext,
    *,
    status: str,
    blockers: list[dict[str, Any]],
    checks: dict[str, Any],
    stage_contract: dict[str, Any] | None,
    started_at: str,
) -> StageOutcome:
    provider_calls = bool(checks.get("provider_calls_made"))
    return write_stage_receipt(
        ctx,
        stage_id=STAGE_ID,
        status=status,
        blockers=sanitize_for_receipt(blockers),
        checks=sanitize_for_receipt(dict(checks)),
        stage_contract=sanitize_for_receipt(stage_contract) if stage_contract else None,
        started_at=started_at,
        receipt_extra={
            "owner_execution_evidence": dict(checks.get("owner_execution_evidence") or {}),
            "provider_calls_made": provider_calls,
            "downloads_made": 0,
        },
    )


def _blocked(
    ctx: ProductionRunnerContext, *, started_at: str, code: str, checks: dict[str, Any], **details: Any
) -> StageOutcome:
    """A zero-spend blocked receipt."""

    provider_calls = bool(checks.get("provider_calls_made"))
    return _receipt(
        ctx,
        status="blocked",
        blockers=[stage_blocker(STAGE_ID, code, **details)],
        checks={
            **checks,
            "blocker_code": code,
            "provider_calls_made": provider_calls,
            "live_provider_calls_made": provider_calls,
        },
        stage_contract=None,
        started_at=started_at,
    )


def run_notebooklm_podcast_stage(
    ctx: ProductionRunnerContext,
    *,
    resume: bool = True,
    env_file: Path | None = None,
    deck_ids: Sequence[str] | None = None,
    ack_live_notebooklm: bool = False,
    notebooklm_profile: str | None = None,
    audio_format: str = "deep-dive",
    audio_length: str | None = None,
    engine: podcast_notebooklm.NotebookLMEngine | None = None,
    transcriber: podcast_notebooklm.Transcriber | None = None,
    accept_review_findings: bool = False,
    force_regenerate: bool = False,
    shared_notebook: str | None = None,
) -> StageOutcome:
    """Run the per-deck NotebookLM podcast stage and write its receipt."""

    started_at = utc_now()
    ctx.run_dir().mkdir(parents=True, exist_ok=True)
    env_file_status = ctx.load_env(env_file)

    if resume:
        prior = resumed_outcome(ctx, STAGE_ID)
        if prior is not None:
            return prior

    checks: dict[str, Any] = {
        "stage_id": STAGE_ID,
        "run_id": ctx.run_id,
        "env_file": env_file_status,
        "notebooklm_profile": notebooklm_profile or "default",
        "notebooklm_shared_notebook": shared_notebook,
        "audio_format": audio_format,
        "audio_length": audio_length or "per-deck default (long; short executive deck: default)",
        "model_governance_task_ids": [TASK_ID, podcast_notebooklm.REVIEW_TASK_ID],
        "notebooklm_model": default_model(TASK_ID),
        "provider_calls_made": False,
    }

    input_paths = required_input_paths(ctx)
    missing = sorted(name for name, path in input_paths.items() if not path.is_file())
    if missing:
        return _blocked(
            ctx, started_at=started_at, code="notebooklm_podcast_input_receipt_missing", checks={**checks, "missing_inputs": missing}
        )
    stage10, stage10_error = _load_json(input_paths["stage10_receipt"])
    stage10_contract = stage10.get("stage_contract") if isinstance(stage10, dict) else None
    if (
        stage10_error
        or not isinstance(stage10, dict)
        or stage10.get("status") != "succeeded"
        or not isinstance(stage10_contract, dict)
        or stage10_contract.get("status") != "pass"
    ):
        return _blocked(
            ctx,
            started_at=started_at,
            code="notebooklm_podcast_stage10_gate_failed",
            checks={
                **checks,
                "stage10_status": stage10.get("status") if isinstance(stage10, dict) else None,
                "stage10_contract_status": stage10_contract.get("status") if isinstance(stage10_contract, dict) else None,
                "stage10_error": stage10_error,
            },
        )
    stage8, stage8_error = _load_json(input_paths["stage8_receipt"])
    if stage8_error or not isinstance(stage8, dict) or stage8.get("status") != "succeeded":
        return _blocked(
            ctx,
            started_at=started_at,
            code="notebooklm_podcast_stage8_gate_failed",
            checks={**checks, "stage8_status": stage8.get("status") if isinstance(stage8, dict) else None, "stage8_error": stage8_error},
        )
    analysis_manifest, manifest_error = _load_json(input_paths["analysis_manifest"])
    if manifest_error or not isinstance(analysis_manifest, dict):
        return _blocked(
            ctx,
            started_at=started_at,
            code="notebooklm_podcast_analysis_manifest_invalid",
            checks={**checks, "analysis_manifest_error": manifest_error},
        )
    checks.update(
        {
            "stage10_receipt_sha256": _file_sha256(input_paths["stage10_receipt"]),
            "stage8_receipt_sha256": _file_sha256(input_paths["stage8_receipt"]),
            "analysis_manifest_sha256": _file_sha256(input_paths["analysis_manifest"]),
            "analysis_manifest_content_sha256": analysis_manifest.get("content_sha256"),
            "executive_digests_sha256": _file_sha256(input_paths["executive_digests"]),
        }
    )

    try:
        deck_scope = presentation_deck_scope(analysis_manifest, ctx.executive_decks)
    except RuntimeError as exc:
        return _blocked(
            ctx,
            started_at=started_at,
            code="notebooklm_podcast_deck_scope_failed",
            checks={**checks, "error": sanitize_for_receipt(str(exc))},
        )
    deck_scope_filter = [str(deck_id) for deck_id in (deck_ids or []) if str(deck_id).strip()]
    checks["deck_scope_full_count"] = len(deck_scope)
    if deck_scope_filter:
        known_ids = {deck["id"] for deck in deck_scope}
        unknown_ids = sorted(set(deck_scope_filter) - known_ids)
        if unknown_ids:
            return _blocked(
                ctx,
                started_at=started_at,
                code="notebooklm_podcast_deck_scope_filter_unknown_deck",
                checks={
                    **checks,
                    "deck_scope_filter": deck_scope_filter,
                    "unknown_deck_ids": unknown_ids,
                    "known_deck_ids": sorted(known_ids),
                },
            )
        checks["deck_scope_filter"] = deck_scope_filter
        deck_scope = [deck for deck in deck_scope if deck["id"] in set(deck_scope_filter)]
    checks["deck_scope"] = [dict(deck) for deck in deck_scope]
    checks["deck_count"] = len(deck_scope)

    blueprint_gate: dict[str, Any] = {}
    missing_blueprints: list[str] = []
    for deck in deck_scope:
        deck_dir = ctx.deck_dir(deck["id"])
        blueprint_path = deck_dir / "presentation_blueprint.json"
        review, _review_error = _load_json(deck_dir / "blueprint_review.json")
        frame_ok = deck["kind"] != "executive" or (deck_dir / "presentation_frame.json").is_file()
        blueprint_gate[deck["id"]] = {
            "blueprint_present": blueprint_path.is_file(),
            "presentation_frame_present": frame_ok,
            "blueprint_review_status": review.get("status") if isinstance(review, dict) else None,
        }
        if not blueprint_path.is_file() or not frame_ok:
            missing_blueprints.append(deck["id"])
    checks["blueprint_gate"] = blueprint_gate
    if missing_blueprints:
        return _blocked(
            ctx,
            started_at=started_at,
            code="notebooklm_podcast_blueprint_missing",
            checks={**checks, "missing_blueprint_deck_ids": missing_blueprints},
        )

    lineage_checks = ctx.upstream_lineage_checks(STAGE_ID)
    if lineage_checks is not None:
        return _blocked(ctx, started_at=started_at, code=UPSTREAM_LINEAGE_NOT_FRESH, checks={**checks, **lineage_checks})

    if not ack_live_notebooklm:
        return _blocked(
            ctx,
            started_at=started_at,
            code="notebooklm_live_ack_required",
            checks={
                **checks,
                "hint": "pass --ack-live-notebooklm to acknowledge quota-metered generation on the logged-in Google account",
            },
        )

    active_engine = engine or podcast_notebooklm.RealNotebookLMEngine(profile=notebooklm_profile)
    checks["engine"] = active_engine.describe()
    checks["engine_version"] = str((checks["engine"] or {}).get("version") or "")
    executive_deck_ids = tuple(deck_id for deck_id, _title in ctx.executive_decks)
    try:
        with provider_receipt_context(
            workspace_root=ctx.workspace, run_id=ctx.run_id, stage_id=STAGE_ID, caller=ctx.receipt_caller
        ):
            owner_summary = podcast_notebooklm.run_stage(
                ctx.workspace,
                deck_ids=deck_scope_filter or None,
                engine=active_engine,
                mode="live",
                audio_format=audio_format,
                audio_length=audio_length,
                resume=resume,
                audio_root=str(ctx.podcast_dir),
                executive_deck_ids=executive_deck_ids,
                review=True,
                transcriber=transcriber,
                accept_review_findings=accept_review_findings,
                force_regenerate=force_regenerate,
                shared_notebook=shared_notebook,
            )
    except podcast_notebooklm.NotebookLMAuthBlocked as exc:
        return _blocked(
            ctx,
            started_at=started_at,
            code="notebooklm_auth_preflight_failed",
            checks={**checks, "storage_state_present": False, "error": sanitize_for_receipt(str(exc)[:400])},
            retryable=False,
        )
    except podcast_notebooklm.NotebookLMPodcastError as exc:
        return _blocked(
            ctx,
            started_at=started_at,
            code="notebooklm_podcast_owner_error",
            checks={**checks, "error": sanitize_for_receipt(str(exc)[:400])},
        )

    preflight = owner_summary.get("preflight") or {}
    deck_records = dict(owner_summary.get("decks") or {})
    deck_blockers = [
        {"deck_id": deck_id, "status": record.get("status"), "blockers": record.get("blockers") or []}
        for deck_id, record in deck_records.items()
        if record.get("status") != "succeeded"
    ]
    review_statuses = dict(owner_summary.get("review_statuses") or {})
    checks.update(
        {
            "preflight": preflight,
            "storage_state_present": bool(preflight.get("storage_state_present")),
            "deck_records": deck_records,
            "deck_blockers": deck_blockers,
            "deck_success_count": int(owner_summary.get("deck_success_count") or 0),
            "deck_blocker_count": len(deck_blockers),
            "notebooklm_podcast_audio_sha256s": dict(owner_summary.get("audio_sha256s") or {}),
            "notebooklm_podcast_aggregate_sha256": owner_summary.get("aggregate_sha256"),
            "source_counts": {deck_id: record.get("source_count") for deck_id, record in deck_records.items()},
            "total_sources_added": sum(int(record.get("sources_added") or 0) for record in deck_records.values()),
            "prompt_variants": {deck_id: record.get("prompt_variant") for deck_id, record in deck_records.items()},
            "review_statuses": review_statuses,
            "review_accepted_deck_ids": sorted(deck_id for deck_id, status in review_statuses.items() if status == "accepted"),
            "review_model_governance_task_id": podcast_notebooklm.REVIEW_TASK_ID,
            "review_model": default_model(podcast_notebooklm.REVIEW_TASK_ID),
            "halt": owner_summary.get("halt"),
            "provider_calls_made": bool(owner_summary.get("network_calls_made")),
            "live_provider_calls_made": bool(owner_summary.get("network_calls_made")),
            "paid_provider_calls_made": False,
            "estimated_cost_usd": 0.0,
            "cost_note": "Google NotebookLM subscription; quota-metered, no per-call billing",
            "secrets_copied": False,
        }
    )

    if deck_scope_filter:
        # Canary scope: interim evidence only, never a succeeded receipt
        # (mirrors presentation_deck_scope_filtered).
        return _receipt(
            ctx,
            status="blocked",
            blockers=[
                stage_blocker(
                    STAGE_ID,
                    "notebooklm_podcast_deck_scope_filtered",
                    deck_ids=deck_scope_filter,
                    deck_blocker_count=len(deck_blockers),
                    blocked_deck_ids=[item["deck_id"] for item in deck_blockers],
                    full_scope_deck_count=checks["deck_scope_full_count"],
                )
            ],
            checks=checks,
            stage_contract=None,
            started_at=started_at,
        )
    halt = owner_summary.get("halt")
    if isinstance(halt, dict):
        return _receipt(
            ctx,
            status="blocked",
            blockers=[
                stage_blocker(
                    STAGE_ID,
                    str(halt.get("code") or "notebooklm_podcast_halted"),
                    deck_id=halt.get("deck_id"),
                    retryable=bool(halt.get("retryable")),
                    resume_hint=halt.get("resume_hint"),
                    error=halt.get("error"),
                )
            ],
            checks=checks,
            stage_contract=None,
            started_at=started_at,
        )
    if deck_blockers:
        return _receipt(
            ctx,
            status="blocked",
            blockers=[
                stage_blocker(
                    STAGE_ID,
                    "notebooklm_podcast_partial_deck_scope_blocked",
                    deck_blocker_count=len(deck_blockers),
                    blocked_deck_ids=[item["deck_id"] for item in deck_blockers],
                )
            ],
            checks=checks,
            stage_contract=None,
            started_at=started_at,
        )

    checks["notebooklm_podcast_status"] = "pass"
    required_script_ids = list(required_script_ids_for_stage(load_stage_script_registry(), STAGE_ID))
    owner_evidence_candidates = {
        OWNER_ID: (
            f"{ctx.podcast_dir}/*/{ctx.notebooklm_subdir}/{ctx.notebooklm_audio_filename}:"
            f"{checks['notebooklm_podcast_aggregate_sha256']}"
        ),
    }
    owner_evidence = {
        owner_id: str(owner_evidence_candidates.get(owner_id) or f"{owner_id}:evidence_unavailable")
        for owner_id in required_script_ids
    }
    checks["owner_execution_evidence"] = owner_evidence
    provisional_receipt = {
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": required_script_ids,
        "owner_execution_evidence": owner_evidence,
        "checks": checks,
    }
    stage_contract = ctx.verify_contract(
        stage_id=STAGE_ID, receipt=provisional_receipt, bind=True, mode="live", strict_execution=True
    ).to_dict()
    final_blockers: list[dict[str, Any]] = []
    if stage_contract["status"] != "pass":
        final_blockers.append(stage_blocker(STAGE_ID, "final_stage_contract_failed"))
    return _receipt(
        ctx,
        status="succeeded" if not final_blockers else "blocked",
        blockers=final_blockers,
        checks=checks,
        stage_contract=stage_contract,
        started_at=started_at,
    )
