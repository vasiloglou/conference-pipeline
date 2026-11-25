"""Assemble a final proof packet and evidence-backed support-label promotions
(Phase 12 / PROD-14).

Conference-general. Consumes the Phase 11 release inventory and the run's own
stage receipts; reads nothing conference-specific except through the workspace
it is given.

Two jobs, both fail-closed:

1. Map every produced artifact to the receipt that produced it, the manifest
   that declares it, its validation result, and any residual risk. An artifact
   with no receipt is a blocker, not a footnote.
2. Propose support-label promotions ONLY where a conference-local receipt
   proves the capability ran live in THIS workspace. A promotion without such
   a receipt is refused and recorded as refused -- the label vocabulary
   (`<conference>-live-proven`) is a claim about evidence, and a claim that
   outruns its evidence is the specific failure this gate exists to prevent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROOF_PACKET_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def map_artifacts_to_receipts(
    inventory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Bind each present deliverable to the stage receipt that produced it.

    Returns (rows, unmapped_paths). A deliverable whose ``produced_by`` stage
    has no succeeded receipt in the inventory's provenance is unmapped.
    """

    receipts_by_stage = {
        str(entry.get("stage_id")): entry
        for entry in inventory.get("provenance", [])
        if isinstance(entry, Mapping)
    }
    rows: list[dict[str, Any]] = []
    unmapped: list[str] = []
    for item in inventory.get("deliverables", []):
        if not isinstance(item, Mapping) or not item.get("present"):
            continue
        stage_id = str(item.get("produced_by") or "")
        receipt = receipts_by_stage.get(stage_id)
        row: dict[str, Any] = {
            "id": item.get("id"),
            "path": item.get("path"),
            "sha256": item.get("sha256"),
            "produced_by": stage_id or None,
            "receipt_path": (receipt or {}).get("receipt_path"),
            "receipt_sha256": (receipt or {}).get("sha256"),
            "receipt_status": (receipt or {}).get("status"),
        }
        if not receipt or receipt.get("status") != "succeeded":
            unmapped.append(str(item.get("path")))
            row["unmapped_reason"] = (
                "no receipt for producing stage"
                if not receipt
                else f"producing stage receipt status is {receipt.get('status')!r}"
            )
        rows.append(row)
    return rows, unmapped


def residual_risks(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Known, measured shortfalls a reader of the proof packet must see.

    Residual risk is not a synonym for failure: these are things that passed
    their gate while still being worth stating plainly.
    """

    risks: list[dict[str, Any]] = []
    for row in inventory.get("review_status", []):
        if not isinstance(row, Mapping):
            continue
        qc = row.get("visual_qc")
        if isinstance(qc, Mapping):
            failed = int(qc.get("fail") or 0)
            total = sum(int(value) for value in qc.values() if isinstance(value, int))
            if failed:
                risks.append(
                    {
                        "kind": "visual_qc_findings_accepted",
                        "deck_id": row.get("deck_id"),
                        "detail": f"{failed} of {total} slides carry an accepted vision-judge finding",
                    }
                )
        for style in ("single_speaker", "notebooklm"):
            if row.get(f"video_{style}") and row.get(f"video_sync_{style}") not in ("pass", "passed"):
                risks.append(
                    {
                        "kind": "video_sync_not_passing",
                        "deck_id": row.get("deck_id"),
                        "detail": f"{style} video exists with sync status "
                        f"{row.get(f'video_sync_{style}')!r}",
                    }
                )
        if "podcast_notebooklm" in row and not row.get("podcast_notebooklm"):
            risks.append(
                {
                    "kind": "notebooklm_podcast_missing",
                    "deck_id": row.get("deck_id"),
                    "detail": "no NotebookLM podcast (audio/<deck>/notebooklm_podcast/podcast.m4a) for this deck",
                }
            )
        elif row.get("podcast_notebooklm") and row.get("podcast_notebooklm_prompt_variant") not in (None, "long"):
            risks.append(
                {
                    "kind": "notebooklm_podcast_short_prompt_used",
                    "deck_id": row.get("deck_id"),
                    "detail": f"NotebookLM podcast was steered by the "
                    f"{row.get('podcast_notebooklm_prompt_variant')!r} prompt variant, not the long one",
                }
            )
    if inventory.get("rejected_evidence"):
        risks.append(
            {
                "kind": "rejected_evidence_present",
                "detail": inventory["rejected_evidence"],
            }
        )
    return risks


def evaluate_support_label_promotions(
    *,
    candidates: Sequence[Mapping[str, Any]],
    stage_receipts: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """Decide each proposed promotion against conference-local receipts.

    A candidate names the capability, the label it would be promoted to, and
    the stage ids whose receipts must prove it. Promotion happens only when
    EVERY named stage has a local receipt whose status is ``succeeded`` and
    whose mode is live. Anything else is refused, with the reason recorded.
    """

    decisions: list[dict[str, Any]] = []
    for candidate in candidates:
        required = [str(item) for item in (candidate.get("requires_stages") or [])]
        evidence: list[dict[str, Any]] = []
        missing: list[str] = []
        for stage_id in required:
            path = stage_receipts.get(stage_id)
            payload = _load_json(path) if path is not None else None
            status = payload.get("status") if isinstance(payload, Mapping) else None
            mode = payload.get("mode") if isinstance(payload, Mapping) else None
            if status == "succeeded" and str(mode).lower() == "live":
                evidence.append(
                    {"stage_id": stage_id, "receipt_path": str(path), "status": status, "mode": mode}
                )
            else:
                missing.append(
                    f"{stage_id}: status={status!r} mode={mode!r}"
                    if payload is not None
                    else f"{stage_id}: no local receipt"
                )
        promoted = not missing and bool(required)
        decisions.append(
            {
                "capability": candidate.get("capability"),
                "current_label": candidate.get("current_label"),
                "proposed_label": candidate.get("proposed_label"),
                "requires_stages": required,
                "promoted": promoted,
                "evidence": evidence,
                "refused_because": missing or None,
            }
        )
    return decisions


def build_proof_packet(
    *,
    workspace: Path,
    run_id: str,
    conference_name: str,
    inventory: Mapping[str, Any],
    stage_receipts: Mapping[str, Path],
    support_label_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifacts, unmapped = map_artifacts_to_receipts(inventory)
    return {
        "schema_version": PROOF_PACKET_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "workspace": str(workspace),
        "conference_name": conference_name,
        "run_id": run_id,
        "release_inventory_generated_at": inventory.get("generated_at"),
        "artifacts": artifacts,
        "unmapped_artifacts": unmapped,
        "stage_map": [
            {
                "stage_id": entry.get("stage_id"),
                "receipt_path": entry.get("receipt_path"),
                "status": entry.get("status"),
                "sha256": entry.get("sha256"),
            }
            for entry in inventory.get("provenance", [])
            if isinstance(entry, Mapping)
        ],
        "review_status": list(inventory.get("review_status", [])),
        "residual_risks": residual_risks(inventory),
        "support_labels": evaluate_support_label_promotions(
            candidates=support_label_candidates, stage_receipts=stage_receipts
        ),
        "publication": inventory.get("publication"),
    }


def proof_packet_blockers(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fail-closed checks over an assembled proof packet (done gates 1 and 2)."""

    blockers: list[dict[str, Any]] = []
    if not packet.get("artifacts"):
        blockers.append(
            {"code": "proof_no_artifacts_mapped", "detail": "the release inventory listed no present deliverable"}
        )
    if packet.get("unmapped_artifacts"):
        blockers.append(
            {"code": "proof_artifact_without_receipt", "detail": packet["unmapped_artifacts"]}
        )
    unproven = [
        row.get("capability")
        for row in packet.get("support_labels", [])
        if isinstance(row, Mapping) and row.get("promoted") and not row.get("evidence")
    ]
    if unproven:
        # Defence in depth: evaluate_support_label_promotions cannot produce
        # this, and if it ever did, the packet must refuse rather than ship a
        # label with no evidence behind it.
        blockers.append({"code": "proof_support_label_without_evidence", "detail": unproven})
    return blockers
