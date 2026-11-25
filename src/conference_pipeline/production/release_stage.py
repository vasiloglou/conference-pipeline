"""Stage 11: release bundle and inventory, shared by every conference runner.

Records what a run produced: storage lanes from the workspace's own storage
manifest, provenance from the stage receipts that produced each artifact, a
digest per deliverable, and each deck's review status. $0 and deterministic --
no provider call, no upload. Publication stays BLOCKED unless the operator
supplies both a target and an approval; assembling a bundle is not permission
to publish it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from conference_pipeline.production.context import ProductionRunnerContext
from conference_pipeline.production.decks import presentation_deck_scope
from conference_pipeline.production.receipts import (
    StageOutcome,
    load_stage_receipt,
    resumed_outcome,
    stage_blocker,
    write_stage_receipt,
)
from conference_pipeline.provider_receipts import sanitize_for_receipt, utc_now
from conference_pipeline.release_bundle import build_release_inventory, release_inventory_blockers
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)
from conference_pipeline.workflow_runs import write_json_atomic

STAGE_ID = "release"
RELEASE_INVENTORY_FILENAME = "release_inventory.json"
# Release enumerates the producing stages: analysis and everything after it,
# up to but excluding itself. Earlier acquisition stages are accounted by
# size in the inventory's storage section, not as provenance rows.
RELEASE_PROVENANCE_MIN_ORDER = 7


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def release_deliverables(ctx: ProductionRunnerContext, deck_ids: Sequence[str]) -> list[dict[str, str]]:
    """The artifacts a reviewer of this conference actually opens.

    Deliberately NOT every file in the workspace: a release inventory that
    hashed a multi-gigabyte download corpus to describe a deck would take
    hours and say nothing useful. Bulk lanes are accounted by size in the
    storage section instead. ``storage/storage_manifest.json`` is likewise
    not listed: it is the workspace contract every stage reads, produced by
    none of them, so the inventory hashes it in its storage section.
    """

    items: list[dict[str, str]] = [
        {"id": "analysis_manifest", "path": str(ctx.analysis_manifest), "kind": "manifest", "produced_by": "analysis"},
        {
            "id": "executive_summary_report",
            "path": str(ctx.analysis_executive_summary),
            "kind": "report",
            "produced_by": "qa-gap-fill-synthesis",
        },
        {
            "id": "trend_report",
            "path": str(ctx.analysis_trend_report),
            "kind": "report",
            "produced_by": "qa-gap-fill-synthesis",
        },
    ]
    for deck_id in deck_ids:
        base = f"{ctx.decks_dir}/{deck_id}"
        podcast_base = f"{ctx.podcast_dir}/{deck_id}/{ctx.notebooklm_subdir}"
        items.extend(
            [
                {"id": f"{deck_id}:pdf", "path": f"{base}/presentation.pdf", "kind": "deck_pdf", "produced_by": "presentation"},
                {
                    "id": f"{deck_id}:deck_contract",
                    "path": f"{base}/presentation_deck.json",
                    "kind": "deck_contract",
                    "produced_by": "presentation",
                },
                {
                    "id": f"{deck_id}:video_single_speaker",
                    "path": f"{base}/presentation_video.mp4",
                    "kind": "video",
                    "produced_by": "presentation",
                },
                {
                    "id": f"{deck_id}:video_notebooklm",
                    "path": f"{base}/presentation_video_notebooklm.mp4",
                    "kind": "video",
                    "produced_by": "presentation",
                },
                {
                    "id": f"{deck_id}:podcast_notebooklm",
                    "path": f"{podcast_base}/{ctx.notebooklm_audio_filename}",
                    "kind": "podcast",
                    "produced_by": "notebooklm-podcast",
                },
                {
                    "id": f"{deck_id}:podcast_notebooklm_manifest",
                    "path": f"{podcast_base}/notebook_manifest.json",
                    "kind": "manifest",
                    "produced_by": "notebooklm-podcast",
                },
            ]
        )
    return items


def _blocked(ctx: ProductionRunnerContext, *, started_at: str, code: str, checks: dict[str, Any]) -> StageOutcome:
    return write_stage_receipt(
        ctx,
        stage_id=STAGE_ID,
        status="blocked",
        blockers=[stage_blocker(STAGE_ID, code)],
        checks=checks,
        stage_contract=None,
        started_at=started_at,
    )


def run_release_stage(
    ctx: ProductionRunnerContext,
    *,
    resume: bool = True,
    publish_target: str | None = None,
    publish_approved_by: str | None = None,
) -> StageOutcome:
    """Assemble the release inventory and write ``stage-11-release.json``."""

    started_at = utc_now()
    if resume:
        prior = resumed_outcome(ctx, STAGE_ID)
        if prior is not None:
            return prior

    checks: dict[str, Any] = {"stage_id": STAGE_ID, "run_id": ctx.run_id}

    # Gate: Stage 10 must have genuinely succeeded. A release bundle assembled
    # over a blocked presentation stage would inventory artifacts the pipeline
    # itself refused to certify.
    presentation_payload = load_stage_receipt(ctx, "presentation")
    if presentation_payload is None or presentation_payload.get("status") != "succeeded":
        checks["presentation_receipt_path"] = str(ctx.receipt_path("presentation"))
        checks["presentation_status"] = presentation_payload.get("status") if presentation_payload else None
        return _blocked(ctx, started_at=started_at, code="release_presentation_stage_not_succeeded", checks=checks)

    # Same source of truth as Stage 10: the analysis manifest, never a second
    # list that could drift from it.
    try:
        analysis_manifest = json.loads((ctx.workspace / ctx.analysis_manifest).read_text(encoding="utf-8"))
        deck_scope = presentation_deck_scope(analysis_manifest, ctx.executive_decks)
    except (OSError, ValueError, RuntimeError) as exc:
        checks["error"] = sanitize_for_receipt(str(exc))
        return _blocked(ctx, started_at=started_at, code="release_deck_scope_failed", checks=checks)
    deck_ids = [str(deck["id"]) for deck in deck_scope]

    release_order = ctx.stage(STAGE_ID).order
    inventory = build_release_inventory(
        workspace=ctx.workspace,
        run_id=ctx.run_id,
        conference_name=ctx.conference_display_name(default=""),
        stage_receipts=ctx.implemented_stage_receipts(
            min_order=RELEASE_PROVENANCE_MIN_ORDER, max_order_exclusive=release_order
        ),
        deliverables=release_deliverables(ctx, deck_ids),
        decks_dir=str(ctx.decks_dir),
        deck_ids=deck_ids,
        publish_target=publish_target,
        publish_approved_by=publish_approved_by,
    )
    inventory_path = ctx.run_dir() / RELEASE_INVENTORY_FILENAME
    write_json_atomic(inventory_path, inventory)

    checks.update(
        {
            "release_inventory_path": str(inventory_path),
            # Named for the owner id it evidences (release.bundle), so the
            # stage-contract coverage audit binds this stage's own work to its
            # own artifact rather than to a governed exemption.
            "release_bundle_sha256": _file_sha256(inventory_path),
            "deck_count": len(deck_ids),
            "deliverable_count": len(inventory["deliverables"]),
            "storage_lanes": inventory["storage"].get("lanes"),
            "publication": inventory["publication"],
            "rejected_evidence": inventory["rejected_evidence"],
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        }
    )
    blockers = [stage_blocker(STAGE_ID, item["code"], detail=item["detail"]) for item in release_inventory_blockers(inventory)]

    required_script_ids = list(required_script_ids_for_stage(load_stage_script_registry(), STAGE_ID))
    present_truth = [
        path for path in (ctx.deck_dir(deck_id) / "artifact_truth_report.json" for deck_id in deck_ids) if path.is_file()
    ]
    owner_evidence_candidates = {"release.bundle": f"{inventory_path}:{checks['release_bundle_sha256']}"}
    if present_truth:
        owner_evidence_candidates["presentation.artifact_truth_validation"] = "; ".join(
            f"{path}:{_file_sha256(path)}" for path in present_truth[:3]
        )
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
    if stage_contract["status"] != "pass":
        blockers.append(stage_blocker(STAGE_ID, "release_final_stage_contract_failed"))

    return write_stage_receipt(
        ctx,
        stage_id=STAGE_ID,
        status="succeeded" if not blockers else "blocked",
        blockers=blockers,
        checks=checks,
        stage_contract=stage_contract,
        started_at=started_at,
        summary_extra={
            "release_inventory_path": checks.get("release_inventory_path"),
            "publication_blocked": (checks.get("publication") or {}).get("blocked"),
        },
    )
