"""Stage 12: final proof packet and support-label promotion, shared by runners.

Maps every produced artifact to the receipt that produced it, carries the
measured residual risks forward rather than rounding them away, and decides
each support-label promotion against conference-local receipts only. $0 and
deterministic. This stage promotes nothing in the docs by itself -- it
produces the evidence-backed decision list an operator applies.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from conference_pipeline.final_proof import build_proof_packet, proof_packet_blockers
from conference_pipeline.production.context import ProductionRunnerContext
from conference_pipeline.production.receipts import (
    StageOutcome,
    load_stage_receipt,
    resumed_outcome,
    stage_blocker,
    write_stage_receipt,
)
from conference_pipeline.production.release_stage import RELEASE_INVENTORY_FILENAME
from conference_pipeline.provider_receipts import utc_now
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)
from conference_pipeline.workflow_runs import write_json_atomic

STAGE_ID = "proof"
PROOF_PACKET_FILENAME = "final_proof_packet.json"

# Capabilities a conference milestone can promote, and the stages whose
# CONFERENCE-LOCAL receipts have to prove each one. A promotion is refused
# unless every named stage has a local receipt that succeeded in live mode --
# the label means "proven here", so evidence from another conference's run
# can never grant it. ``current_label`` records the support level the docs
# carried before this conference; the proposed label is derived from the
# workspace slug so no runner hardcodes its own name.
SUPPORT_LABEL_CAPABILITIES: tuple[dict[str, Any], ...] = (
    {
        "capability": "Acquisition stages: discover, download, transcribe",
        "current_label": "neurips-live-proven",
        "requires_stages": ["source-discovery", "download-corpus", "transcription-ocr"],
    },
    {
        "capability": "Analyze stage orchestration",
        "current_label": "neurips-live-proven",
        "requires_stages": ["analysis"],
    },
    {
        "capability": "Traceability, QA, gap fill, and synthesis",
        "current_label": "workspace-local",
        "requires_stages": ["qa-gap-fill-synthesis"],
    },
    {"capability": "Podcast rendered audio", "current_label": "workspace-local", "requires_stages": ["podcast"]},
    {"capability": "Presentation production", "current_label": "workspace-local", "requires_stages": ["presentation"]},
    {
        "capability": "Per-deck NotebookLM podcast production",
        "current_label": "workspace-local",
        "requires_stages": ["notebooklm-podcast"],
    },
    {"capability": "Release bundle and inventory", "current_label": "non-live-general", "requires_stages": ["release"]},
)


def support_label_candidates(slug: str) -> tuple[dict[str, Any], ...]:
    """The promotion candidates for one conference: ``<slug>-live-proven``."""

    proposed = f"{slug}-live-proven"
    return tuple({**row, "proposed_label": proposed} for row in SUPPORT_LABEL_CAPABILITIES)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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


def run_proof_stage(ctx: ProductionRunnerContext, *, resume: bool = True) -> StageOutcome:
    """Assemble the final proof packet and write ``stage-12-proof.json``."""

    started_at = utc_now()
    if resume:
        prior = resumed_outcome(ctx, STAGE_ID)
        if prior is not None:
            return prior

    checks: dict[str, Any] = {"stage_id": STAGE_ID, "run_id": ctx.run_id}

    release_payload = load_stage_receipt(ctx, "release")
    if release_payload is None or release_payload.get("status") != "succeeded":
        checks["release_receipt_path"] = str(ctx.receipt_path("release"))
        checks["release_status"] = release_payload.get("status") if release_payload else None
        return _blocked(ctx, started_at=started_at, code="proof_release_stage_not_succeeded", checks=checks)

    inventory_path = ctx.run_dir() / RELEASE_INVENTORY_FILENAME
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else None
    if not isinstance(inventory, dict):
        checks["release_inventory_path"] = str(inventory_path)
        return _blocked(ctx, started_at=started_at, code="proof_release_inventory_missing", checks=checks)

    proof_order = ctx.stage(STAGE_ID).order
    stage_receipts = {
        stage_id: ctx.run_dir() / entry.receipt_filename
        for stage_id, entry in ctx.stages.items()
        if entry.order < proof_order
    }
    packet = build_proof_packet(
        workspace=ctx.workspace,
        run_id=ctx.run_id,
        conference_name=ctx.conference_display_name(default=""),
        inventory=inventory,
        stage_receipts=stage_receipts,
        support_label_candidates=support_label_candidates(ctx.slug),
    )
    packet_path = ctx.run_dir() / PROOF_PACKET_FILENAME
    write_json_atomic(packet_path, packet)

    promoted = [row for row in packet["support_labels"] if row.get("promoted")]
    checks.update(
        {
            "proof_packet_path": str(packet_path),
            # Named for the owner id it evidences (proof.final_verification).
            "proof_final_verification_sha256": _file_sha256(packet_path),
            "artifact_count": len(packet["artifacts"]),
            "unmapped_artifact_count": len(packet["unmapped_artifacts"]),
            "residual_risk_count": len(packet["residual_risks"]),
            "support_labels_promoted": [row["capability"] for row in promoted],
            "support_labels_refused": [row["capability"] for row in packet["support_labels"] if not row.get("promoted")],
            "publication": packet.get("publication"),
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        }
    )
    blockers = [stage_blocker(STAGE_ID, item["code"], detail=item["detail"]) for item in proof_packet_blockers(packet)]

    required_script_ids = list(required_script_ids_for_stage(load_stage_script_registry(), STAGE_ID))
    truth_reports = sorted((ctx.workspace / ctx.decks_dir).glob("*/artifact_truth_report.json"))
    owner_evidence_candidates = {
        "proof.final_verification": f"{packet_path}:{checks['proof_final_verification_sha256']}",
    }
    if truth_reports:
        owner_evidence_candidates["presentation.artifact_truth_validation"] = "; ".join(
            f"{path}:{_file_sha256(path)}" for path in truth_reports[:3]
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
        blockers.append(stage_blocker(STAGE_ID, "proof_final_stage_contract_failed"))

    return write_stage_receipt(
        ctx,
        stage_id=STAGE_ID,
        status="succeeded" if not blockers else "blocked",
        blockers=blockers,
        checks=checks,
        stage_contract=stage_contract,
        started_at=started_at,
        summary_extra={
            "proof_packet_path": checks.get("proof_packet_path"),
            "support_labels_promoted": checks.get("support_labels_promoted"),
            "residual_risk_count": checks.get("residual_risk_count"),
        },
    )
