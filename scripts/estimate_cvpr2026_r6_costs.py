#!/usr/bin/env python3
"""Build a finite CVPR r6 speech estimate from one declared audio receipt.

This is intentionally an offline gate.  It does not discover files, invoke a
provider, or dispatch ASR.  The caller must provide the one manifest-declared
r6 audio-recovery receipt and the one manifest-declared estimate output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_OCR_CAP_USD,
    R6ArtifactContract,
    build_r6_dispatch_schedule,
    build_r6_execution_envelope,
    canonical_r6_queue,
    r6_audio_policy_digest,
    r6_digest,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    validate_r6_audio_recovery_receipt,
)
from conference_pipeline.costs.budget_enforcement import _existing_spend, check_preflight
from conference_pipeline.costs.estimator import (
    build_receipt_bound_audio_workload,
    estimate_receipt_bound_audio_workload,
)
from conference_pipeline.model_governance.registry import (
    default_model_for_provider,
    load_registry,
)

TASK_ID = "transcription.session_speech"
USER_ALL_IN_CAP_USD = 800.0


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contains_secretish_key(payload: object) -> bool:
    forbidden = ("cookie", "password", "secret", "token", "authorization", "api_key")
    if isinstance(payload, dict):
        return any(
            (
                str(key).lower() != "secrets_copied"
                and any(marker in str(key).lower() for marker in forbidden)
            )
            or _contains_secretish_key(value)
            for key, value in payload.items()
        )
    if isinstance(payload, list):
        return any(_contains_secretish_key(value) for value in payload)
    return False


def estimate_r6_audio_receipt(
    *,
    contract: R6ArtifactContract,
    all_in_cap_usd: float = USER_ALL_IN_CAP_USD,
) -> dict[str, Any]:
    """Validate one declared receipt and return its finite governed estimate."""

    workspace = contract.ensure_workspace_trusted()
    audio_receipt_path = contract.path("audio_recovery_receipt")
    if not math.isfinite(all_in_cap_usd) or not 0 < all_in_cap_usd <= USER_ALL_IN_CAP_USD:
        raise ValueError(
            "all-in cap must be finite, positive, and no greater than the approved cap"
        )
    raw = audio_receipt_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or _contains_secretish_key(payload):
        raise ValueError("audio receipt must be a sanitized object")
    recovery_path = contract.path("recovery_inventory")
    if not recovery_path.is_file():
        raise ValueError("recovery inventory is required for exact r6 audio admission")
    recovery_payload = json.loads(recovery_path.read_bytes())
    if not isinstance(recovery_payload, dict):
        raise ValueError("recovery inventory must be a sanitized object")
    if not contract.fixture_injection:
        if payload.get("status") != "succeeded" or payload.get("run_id") != contract.run_id:
            raise ValueError("audio receipt must be a succeeded current r6 receipt")
        if payload.get("audio_policy_digest") != r6_audio_policy_digest():
            raise ValueError("audio receipt public policy digest drifted")
    classification = validate_r6_audio_recovery_receipt(
        recovery_inventory=recovery_payload,
        receipt=payload,
        contract=contract,
    )
    rows = list(classification.acquired_records)
    workload = build_receipt_bound_audio_workload(rows, receipt_sha256=_sha256_bytes(raw))
    registry = load_registry()
    policy = registry.get(TASK_ID)
    if policy.budget_cap_usd is None:
        raise ValueError("governed speech task has no finite task cap")
    model = default_model_for_provider(TASK_ID, policy.provider)
    estimate = estimate_receipt_bound_audio_workload(
        workload,
        provider=policy.provider,
        model=model,
    )
    existing_spend = _existing_spend(workspace)
    projected_total = existing_spend + float(estimate["estimated_cost_usd"])
    if projected_total > all_in_cap_usd:
        raise ValueError("finite r6 estimate breaches the all-in budget cap")
    if float(estimate["estimated_cost_usd"]) > float(policy.budget_cap_usd):
        raise ValueError("finite r6 estimate breaches the governed speech task cap")
    preflight = check_preflight(
        workspace,
        TASK_ID,
        float(estimate["estimated_cost_usd"]),
        registry=registry,
    )
    if preflight.breach:
        raise ValueError(f"canonical preflight rejected estimate: {preflight.reason}")
    material_path = contract.path("material_acquisition_receipt")
    discovery_path = contract.path("material_discovery_receipt")
    material_payload: dict[str, Any] | None = None
    discovery_payload: dict[str, Any] | None = None
    if material_path.is_file():
        candidate = json.loads(material_path.read_text(encoding="utf-8"))
        if not isinstance(candidate, dict):
            raise ValueError("material receipt must be a sanitized object")
        material_payload = candidate
    elif not contract.fixture_injection:
        raise ValueError("material receipt is required for a production r6 estimate")
    if discovery_path.is_file():
        candidate = json.loads(discovery_path.read_text(encoding="utf-8"))
        if not isinstance(candidate, dict):
            raise ValueError("material discovery receipt must be a sanitized object")
        discovery_payload = candidate
    elif not contract.fixture_injection:
        raise ValueError("material discovery receipt is required for a production r6 estimate")
    material_policy_digest = (
        str(discovery_payload.get("policy_digest"))
        if discovery_payload and isinstance(discovery_payload.get("policy_digest"), str)
        else (str(material_payload.get("policy_digest")) if material_payload else "0" * 64)
    )
    if len(material_policy_digest) != 64:
        raise ValueError("material policy digest is invalid")
    if material_payload and material_payload.get("policy_digest") != material_policy_digest:
        raise ValueError("material policy digest drifted")
    ocr_ids = [
        str(row["record_id"])
        for row in (material_payload or {}).get("payloads", [])
        if isinstance(row, dict)
        and row.get("acquisition_state") == "acquired"
        and row.get("content_class") == "slide_image"
        and isinstance(row.get("record_id"), str)
    ]
    speech_ids = [
        str(row["item_id"])
        for row in rows
        if isinstance(row, dict)
        and row.get("terminal_state") == "acquired"
        and isinstance(row.get("item_id"), str)
    ]
    queue, queue_digest = canonical_r6_queue(speech_ids=speech_ids, ocr_ids=ocr_ids)
    # OCR is never inferred from a filesystem.  This finite ceiling is a
    # schedule authorization, not a provider quote, and remains below the
    # governed OCR cap until a later owner provides a lower exact quote.
    ocr_total = min(R6_OCR_CAP_USD, 0.01 * len(ocr_ids))
    schedule = build_r6_dispatch_schedule(
        run_id=contract.run_id,
        queue=queue,
        queue_digest=queue_digest,
        speech_total_usd=float(estimate["estimated_cost_usd"]),
        ocr_total_usd=ocr_total,
    )
    scheduled_all_in = float(schedule["totals"]["all_in_usd"])
    if existing_spend + scheduled_all_in > all_in_cap_usd:
        raise ValueError("complete r6 dispatch schedule breaches the all-in budget cap")
    predecessor_bindings = {
        "audio_recovery_receipt_sha256": workload.receipt_sha256,
        "material_acquisition_receipt_sha256": _sha256_bytes(material_path.read_bytes())
        if material_path.is_file() else "0" * 64,
        "material_discovery_receipt_sha256": _sha256_bytes(discovery_path.read_bytes())
        if discovery_path.is_file() else "0" * 64,
        "recovery_inventory_sha256": _sha256_bytes(recovery_path.read_bytes())
        if recovery_path.is_file() else "0" * 64,
    }
    envelope = build_r6_execution_envelope(
        kind="estimate",
        status="succeeded",
        material_policy_digest=material_policy_digest,
        audio_policy_digest=(
            str(payload.get("audio_policy_digest"))
            if isinstance(payload.get("audio_policy_digest"), str)
            else r6_audio_policy_digest()
        ),
        predecessor_bindings=predecessor_bindings,
        schedule=schedule,
    )
    result: dict[str, Any] = {
        **envelope,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cost_estimate",
        "status": "succeeded",
        "run_id": contract.run_id,
        "policy_digest": material_policy_digest,
        "audio_receipt_path": contract.relative_path("audio_recovery_receipt").as_posix(),
        "audio_receipt_sha256": workload.receipt_sha256,
        "estimate": estimate,
        "task_id": TASK_ID,
        "task_cap_usd": float(policy.budget_cap_usd),
        "all_in_cap_usd": all_in_cap_usd,
        "canonical_existing_spend_usd": existing_spend,
        "projected_total_usd": projected_total,
        "preflight": {
            "breach": preflight.breach,
            "reason": preflight.reason,
            "cap_usd": preflight.cap_usd,
        },
    }
    result["maximum_charge_usd_by_task"] = {
        "transcription.session_speech": float(schedule["totals"]["speech_usd"]),
        "ocr.slide_image_parse": float(schedule["totals"]["ocr_usd"]),
    }
    result["envelope_sha256"] = r6_digest(
        {key: value for key, value in result.items() if key != "envelope_sha256"}
    )
    return result


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    contract = R6ArtifactContract.production()
    receipt = estimate_r6_audio_receipt(
        contract=contract,
    )
    output = contract.path("cost_estimate_receipt")
    _atomic_write(output, receipt)
    print(json.dumps({"status": "pass", "output": output.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - command wrapper
    raise SystemExit(main())
