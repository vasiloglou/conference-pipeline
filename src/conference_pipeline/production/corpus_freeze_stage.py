"""Stage 3: corpus freeze, shared by every conference orchestrator.

The base is the program-item corpus freeze, the stricter of the two runner
implementations it replaces (:func:`run_corpus_freeze_stage`). It makes no
provider call and no download:

1. pinned source lineage: the run's succeeded, live, blocker-free Stage 2
   receipt with a passing contract, and the canonical source inventory whose
   self-digest verifies and equals the pinned digest (by default the digest
   that receipt recorded);
2. no silent item-count caps in the invocation;
3. the conference's locked policy exclusions (the ``corpus_exclusions``
   hook), which stay auditable without removing program items;
4. the versioned, source-bound operator exclusion file
   (``storage/corpus-freeze/operator_exclusions.json``), every exclusion
   naming a known program item, its rationale and its source evidence;
5. exactly one selection record per canonical program item, every included
   item carrying a Stage 4 acquisition plan or an unreachable-source path;
6. the manifest, a paged report, the strict stage contract and an immutable
   attempt receipt.

A conference whose Stage 2 wrote a probed source-intake inventory (see
:func:`~conference_pipeline.production.source_discovery_stage.run_intake_probe_discovery`)
freezes at source-family granularity instead
(:func:`run_source_family_freeze`): every discovered family is included unless
explicitly excluded by the operator, each with a planned acquisition.

Conference identity comes from the context and :class:`ProgramItemFreezeSettings`
/ :class:`SourceFamilyFreezeSettings`; the module carries no conference
literal.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production.context import ProductionRunnerContext, StageSpec
from conference_pipeline.production.hooks import NO_HOOKS, ConferenceHooks
from conference_pipeline.production.receipts import DEFAULT_STAGE_IO, StageIO
from conference_pipeline.production.source_discovery_stage import (
    SOURCE_DISCOVERY_INVENTORY,
    _write_bytes,
    attempt_receipt_path,
    stage_owner_id,
    write_attempt_receipt,
)
from conference_pipeline.production_preflight import canonical_json_digest
from conference_pipeline.provider_receipts import BlockerClass, sanitize_for_receipt
from conference_pipeline.receipt_lineage import sha256_file
from conference_pipeline.source_discovery import canonical_json_digest as source_discovery_canonical_json_digest
from conference_pipeline.stage_script_registry import registry_sha256

STAGE_ID = "corpus-freeze"
SOURCE_STAGE_ID = "source-discovery"

CORPUS_FREEZE_PATH = Path("storage/corpus-freeze/corpus_freeze.json")
CORPUS_FREEZE_REPORT_PATH = Path("storage/corpus-freeze/corpus_freeze_report.md")
OPERATOR_EXCLUSIONS_PATH = Path("storage/corpus-freeze/operator_exclusions.json")
REPORT_PAGE_SIZE = 50
OWNER_FUNCTION = "run_corpus_freeze_stage"

# Known truncation controls: any of them in a freeze invocation would make an
# item-count cap the implicit corpus scope.
SILENT_CAP_FLAGS: tuple[str, ...] = (
    "--max-papers",
    "--max-items",
    "--max-sessions",
    "--paper-limit",
    "--item-limit",
    "--limit",
    "--sample",
)

PolicyExclusions = tuple[list[dict[str, Any]], list[dict[str, Any]]]


def corpus_freeze_blocker(code: str, **details: Any) -> dict[str, Any]:
    """Create a compact, secret-free corpus-freeze blocker."""

    return {"code": code, "affected_stage": STAGE_ID, **details}


def read_json_object(path: Path, *, missing_code: str, invalid_code: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Read a local control artifact without accepting a non-object payload."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [corpus_freeze_blocker(missing_code, path=str(path))]
    except (OSError, json.JSONDecodeError):
        return None, [corpus_freeze_blocker(invalid_code, path=str(path))]
    if not isinstance(payload, dict):
        return None, [corpus_freeze_blocker(invalid_code, path=str(path))]
    return payload, []


def corpus_inventory_digest(inventory: Mapping[str, Any]) -> str:
    """Rebuild the source inventory digest using its publication contract."""

    # Stage 2 publishes this control artifact through source_discovery, whose
    # digest contract preserves Unicode rather than the preflight encoder's
    # ASCII-escaped representation.  Verify against the publisher's contract
    # so a valid, pinned source inventory cannot be rejected downstream.
    return source_discovery_canonical_json_digest(
        {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    )


def silent_cap_hits(command_args: Sequence[str], flags: Sequence[str] = SILENT_CAP_FLAGS) -> list[str]:
    """Reject all known truncation controls before they can become implicit scope."""

    return sorted(
        {
            flag
            for flag in flags
            if any(argument == flag or argument.startswith(f"{flag}=") for argument in command_args)
        }
    )


@dataclass(frozen=True)
class ProgramItemFreezeSettings:
    """What a conference decides about the program-item corpus freeze.

    ``pinned_run_id`` / ``pinned_inventory_sha256`` pin the Stage 2 lineage
    the freeze may consume (``None``: this run, and the inventory digest its
    Stage 2 receipt recorded). ``program_item_prefix`` is the canonical
    program-item id prefix (``None``: ``<slug>:program-item:``).
    ``display_name`` is recorded as the manifest's conference (``None``: the
    workspace's conference name).
    """

    owner_id: str | None = None
    pinned_run_id: str | None = None
    pinned_inventory_sha256: str | None = None
    program_item_prefix: str | None = None
    display_name: str | None = None
    report_page_size: int = REPORT_PAGE_SIZE
    cap_flags: tuple[str, ...] = SILENT_CAP_FLAGS


DEFAULT_PROGRAM_ITEM_SETTINGS = ProgramItemFreezeSettings()


def validate_pinned_source_lineage(
    ctx: ProductionRunnerContext,
    *,
    pinned_run_id: str | None = None,
    pinned_inventory_sha256: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load only the pinned Stage 2 inventory after its immutable receipt passes."""

    workspace = ctx.workspace
    run_id = ctx.run_id
    expected_run_id = pinned_run_id or run_id
    if run_id != expected_run_id:
        return None, {}, [corpus_freeze_blocker("pinned_source_run_mismatch")]
    source_stage = ctx.stage(SOURCE_STAGE_ID)
    source_receipt_path = ctx.run_dir() / source_stage.receipt_filename
    source_receipt, blockers = read_json_object(
        source_receipt_path,
        missing_code="source_stage_receipt_missing",
        invalid_code="source_stage_receipt_invalid",
    )
    if source_receipt is None:
        return None, {}, blockers
    source_checks = (
        (source_receipt.get("run_id") == expected_run_id, "source_receipt_run_id_mismatch"),
        (source_receipt.get("stage_id") == SOURCE_STAGE_ID, "source_receipt_stage_identity_invalid"),
        (source_receipt.get("stage_order") == source_stage.order, "source_receipt_stage_order_invalid"),
        (source_receipt.get("status") == "succeeded", "source_receipt_not_succeeded"),
        (source_receipt.get("mode") == "live", "source_receipt_not_live"),
        (not source_receipt.get("blockers"), "source_receipt_has_blockers"),
        (
            isinstance(source_receipt.get("stage_contract"), Mapping)
            and source_receipt["stage_contract"].get("status") == "pass",
            "source_receipt_contract_not_passing",
        ),
    )
    blockers.extend(corpus_freeze_blocker(code) for valid, code in source_checks if not valid)
    inventory_path = workspace / SOURCE_DISCOVERY_INVENTORY
    inventory, inventory_blockers = read_json_object(
        inventory_path,
        missing_code="source_inventory_missing",
        invalid_code="source_inventory_invalid",
    )
    blockers.extend(inventory_blockers)
    if inventory is None:
        return None, {}, blockers
    inventory_digest = inventory.get("inventory_sha256")
    receipt_inventory = source_receipt.get("canonical_inventory")
    receipt_digest = receipt_inventory.get("inventory_sha256") if isinstance(receipt_inventory, Mapping) else None
    pinned_digest = pinned_inventory_sha256 or receipt_digest
    digest_checks = (
        (isinstance(inventory_digest, str), "source_inventory_digest_missing"),
        (inventory_digest == corpus_inventory_digest(inventory), "source_inventory_digest_invalid"),
        (isinstance(pinned_digest, str) and inventory_digest == pinned_digest, "source_inventory_digest_not_pinned"),
        (isinstance(pinned_digest, str) and receipt_digest == pinned_digest, "source_receipt_inventory_digest_mismatch"),
        (inventory.get("run_id") == expected_run_id, "source_inventory_run_id_mismatch"),
        (inventory.get("stage_id") == SOURCE_STAGE_ID, "source_inventory_stage_identity_invalid"),
        (inventory.get("status") == "prepared", "source_inventory_not_prepared"),
        (not inventory.get("blockers"), "source_inventory_has_blockers"),
    )
    blockers.extend(corpus_freeze_blocker(code) for valid, code in digest_checks if not valid)
    records = inventory.get("source_records")
    if not isinstance(records, list):
        blockers.append(corpus_freeze_blocker("source_records_missing"))
        return inventory, {}, blockers
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("source_record_id"), str):
            blockers.append(corpus_freeze_blocker("source_record_identity_invalid"))
            continue
        record_id = str(record["source_record_id"])
        if record_id in records_by_id:
            blockers.append(corpus_freeze_blocker("duplicate_source_record_id", source_record_id=record_id))
            continue
        records_by_id[record_id] = dict(record)
    return inventory, records_by_id, blockers


def load_program_item_exclusions(
    *,
    workspace: Path,
    inventory_sha256: str,
    program_items: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Validate the versioned, source-bound operator exclusion control file."""

    path = workspace / OPERATOR_EXCLUSIONS_PATH
    metadata: dict[str, Any] = {"path": str(OPERATOR_EXCLUSIONS_PATH), "present": path.exists()}
    if not path.exists():
        metadata["validation_status"] = "not_present"
        return [], metadata, []
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [], metadata, [corpus_freeze_blocker("operator_exclusions_invalid_json")]
    metadata["sha256"] = sha256(raw).hexdigest()
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        return [], metadata, [corpus_freeze_blocker("operator_exclusions_schema_invalid")]
    if payload.get("source_inventory_sha256") != inventory_sha256:
        return [], metadata, [corpus_freeze_blocker("operator_exclusions_stale")]
    raw_exclusions = payload.get("exclusions")
    if not isinstance(raw_exclusions, list):
        return [], metadata, [corpus_freeze_blocker("operator_exclusions_schema_invalid")]
    exclusions: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    seen: set[str] = set()
    required_fields = (
        "program_item_id",
        "rationale",
        "decision_timestamp",
        "decision_context",
        "downstream_effect",
    )
    for index, raw_exclusion in enumerate(raw_exclusions):
        if not isinstance(raw_exclusion, Mapping):
            blockers.append(corpus_freeze_blocker("operator_exclusion_invalid", index=index))
            continue
        missing = [
            name for name in required_fields if not isinstance(raw_exclusion.get(name), str) or not raw_exclusion[name].strip()
        ]
        source_evidence = raw_exclusion.get("source_evidence")
        if (
            missing
            or not isinstance(source_evidence, list)
            or not source_evidence
            or not all(isinstance(item, str) and item for item in source_evidence)
        ):
            blockers.append(corpus_freeze_blocker("operator_exclusion_invalid", index=index))
            continue
        item_id = str(raw_exclusion["program_item_id"])
        if item_id in seen:
            blockers.append(corpus_freeze_blocker("operator_exclusion_duplicate_target", program_item_id=item_id))
            continue
        seen.add(item_id)
        item = program_items.get(item_id)
        if item is None:
            blockers.append(corpus_freeze_blocker("operator_exclusion_unknown_target", program_item_id=item_id))
            continue
        source_ids = item.get("source_record_ids")
        if not isinstance(source_ids, list) or not set(source_evidence).issubset(set(source_ids)):
            blockers.append(corpus_freeze_blocker("operator_exclusion_source_evidence_invalid", program_item_id=item_id))
            continue
        exclusions.append({key: raw_exclusion[key] for key in (*required_fields, "source_evidence")})
    metadata["validation_status"] = "pass" if not blockers else "blocked"
    return sorted(exclusions, key=lambda item: str(item["program_item_id"])), metadata, blockers


def acquisition_disposition(item: Mapping[str, Any], records_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Derive a Stage 4 intent from source evidence without claiming acquisition."""

    acquisition: list[dict[str, Any]] = []
    for record_id in sorted(str(value) for value in item.get("source_record_ids", [])):
        record = records_by_id[record_id]
        urls = sorted(
            value for value in record.get("aliases", []) if isinstance(value, str) and value.startswith(("https://", "http://"))
        )
        if urls:
            acquisition.append(
                {
                    "source_record_id": record_id,
                    "source_family_id": str(record.get("family_id") or "unknown"),
                    "source_urls": urls,
                    "phase": "download-corpus",
                    "intent": "acquire the source record and eligible linked artifacts with provenance",
                }
            )
    if acquisition:
        return {"planned_acquisition": acquisition, "unreachable_source_path": None}
    return {
        "planned_acquisition": [],
        "unreachable_source_path": {
            "phase": "download-corpus",
            "reason": "the pinned source record has no admissible URL; record an unreachable-source result before downstream analysis",
            "source_record_ids": sorted(str(value) for value in item.get("source_record_ids", [])),
        },
    }


def build_program_item_selection(
    *,
    item: Mapping[str, Any],
    records_by_id: Mapping[str, Mapping[str, Any]],
    operator_exclusions: Mapping[str, Mapping[str, Any]],
    program_item_prefix: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Create exactly one auditable selection record for a canonical item."""

    item_id = item.get("program_item_id")
    source_ids = item.get("source_record_ids")
    if not isinstance(item_id, str) or not item_id.startswith(program_item_prefix):
        return None, [corpus_freeze_blocker("program_item_identity_invalid")]
    if not isinstance(source_ids, list) or not source_ids or not all(isinstance(value, str) for value in source_ids):
        return None, [corpus_freeze_blocker("program_item_source_record_evidence_missing", program_item_id=item_id)]
    if len(set(source_ids)) != len(source_ids) or any(source_id not in records_by_id for source_id in source_ids):
        return None, [corpus_freeze_blocker("program_item_source_record_evidence_missing", program_item_id=item_id)]
    base = {
        "program_item_id": item_id,
        "source_record_ids": sorted(source_ids),
        "source_family_ids": sorted({str(records_by_id[source_id].get("family_id") or "unknown") for source_id in source_ids}),
    }
    exclusion = operator_exclusions.get(item_id)
    if exclusion is not None:
        return {
            **base,
            "selection_status": "operator_excluded",
            "selection_reason": "explicit, validated operator exclusion",
            "operator_exclusion": dict(exclusion),
            "planned_acquisition": [],
            "unreachable_source_path": None,
        }, []
    disposition = acquisition_disposition(item, records_by_id)
    if not disposition["planned_acquisition"] and not disposition["unreachable_source_path"]:
        return None, [corpus_freeze_blocker("included_item_without_phase4_path", program_item_id=item_id)]
    return {
        **base,
        "selection_status": "included",
        "selection_reason": "default broad corpus policy: include every canonical program item unless explicitly operator excluded",
        **disposition,
    }, []


def program_item_report(manifest: Mapping[str, Any]) -> str:
    """Render one bounded report page; the JSON ledger stays complete."""

    items = manifest["program_items"]
    page_size = int(manifest["report_pagination"]["page_size"])
    visible = items[:page_size]
    lines = [
        f"# {manifest['conference']} Corpus Freeze Report",
        "",
        f"Run id: `{manifest['run_id']}`",
        f"Status: `{manifest['status']}`",
        "",
        "## Accounting",
        "",
        f"- Discovered program items: `{manifest['counts']['discovered_program_items']}`",
        f"- Included program items: `{manifest['counts']['included_program_items']}`",
        f"- Operator-excluded program items: `{manifest['counts']['operator_excluded_program_items']}`",
        f"- Policy-excluded program items: `{manifest['counts']['policy_excluded_program_items']}`",
        "",
        "## Locked Source Policy",
        "",
    ]
    lines.extend(
        f"- `{entry['exclusion_id']}` for `{entry['family_id']}`: {entry['selection_effect']}"
        for entry in manifest["policy_exclusions"]
    )
    lines.extend(
        [
            "",
            "## Visible Program-Item Page",
            "",
            f"Visible program-item IDs: {len(visible)} of {len(items)} (page size {page_size}; complete ledger is JSON).",
            "",
        ]
    )
    lines.extend(f"- `{item['program_item_id']}` — `{item['selection_status']}`" for item in visible)
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No provider calls were made.",
            "- No downloads were made.",
            "- No silent item-count caps were used.",
            "- Phase 4 plans are intents, not acquisition-success claims.",
            "",
        ]
    )
    return "\n".join(lines)


def write_corpus_freeze_receipt(
    ctx: ProductionRunnerContext,
    *,
    owner_id: str,
    status: str,
    blockers: list[dict[str, Any]],
    checks: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    stage_contract: Mapping[str, Any] | None,
    lineage: Mapping[str, Any] | None = None,
    io: StageIO = DEFAULT_STAGE_IO,
    write_bytes: Callable[[Path, bytes], None] = _write_bytes,
) -> dict[str, Any]:
    """Persist one immutable corpus-freeze attempt and preserve prior proof bytes."""

    stage = ctx.stage(STAGE_ID)
    receipt_path, canonical_path, superseded_prior_path = attempt_receipt_path(
        ctx, STAGE_ID, status=status, now=io.now(), write_bytes=write_bytes, unique=True
    )
    owner_evidence = {
        owner_id: {
            "owner_function": OWNER_FUNCTION,
            "manifest_path": str(CORPUS_FREEZE_PATH),
            "provider_calls_made": 0,
            "downloads_made": 0,
        }
    }
    receipt = {
        "schema_version": 1,
        "receipt_type": ctx.receipt_type,
        "receipt_id": f"{ctx.run_id}:{stage.id}:{io.now()}",
        "run_id": ctx.run_id,
        "workspace": str(ctx.workspace),
        "stage_id": stage.id,
        "stage_order": stage.order,
        "mode": "live",
        "status": status,
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": [owner_id],
        "owner_execution_evidence": owner_evidence,
        "corpus_manifest": {
            "path": str(CORPUS_FREEZE_PATH),
            "selection_sha256": manifest.get("selection_sha256") if manifest else None,
        },
        "provider_calls_made": 0,
        "downloads_made": 0,
        "checks": sanitize_for_receipt(dict(checks)),
        "stage_contract": sanitize_for_receipt(stage_contract) if stage_contract else None,
        "blockers": sanitize_for_receipt(blockers),
        "supersedes_receipt_path": str(canonical_path) if superseded_prior_path else None,
        "superseded_prior_receipt_path": str(superseded_prior_path) if superseded_prior_path else None,
        "secrets_copied": False,
    }
    outcome = write_attempt_receipt(ctx, receipt, receipt_path=receipt_path, lineage=lineage, io=io)
    outcome["manifest"] = dict(manifest) if manifest else None
    return outcome


def run_corpus_freeze_stage(
    ctx: ProductionRunnerContext,
    *,
    settings: ProgramItemFreezeSettings = DEFAULT_PROGRAM_ITEM_SETTINGS,
    command_args: Sequence[str] = (),
    hooks: ConferenceHooks = NO_HOOKS,
    io: StageIO = DEFAULT_STAGE_IO,
) -> dict[str, Any]:
    """Freeze every pinned canonical program item without paid or download work."""

    workspace = ctx.workspace
    run_id = ctx.run_id
    owner_id = stage_owner_id(settings.owner_id, STAGE_ID)
    item_prefix = settings.program_item_prefix or f"{ctx.slug}:program-item:"

    def receipt(**kwargs: Any) -> dict[str, Any]:
        return write_corpus_freeze_receipt(ctx, owner_id=owner_id, io=io, **kwargs)

    inventory, records_by_id, blockers = validate_pinned_source_lineage(
        ctx,
        pinned_run_id=settings.pinned_run_id,
        pinned_inventory_sha256=settings.pinned_inventory_sha256,
    )
    cap_hits = silent_cap_hits(tuple(command_args), settings.cap_flags)
    if cap_hits:
        blockers.append(corpus_freeze_blocker("silent_item_cap_rejected", flags=cap_hits))
    checks: dict[str, Any] = {
        "source_lineage_integrity": not blockers,
        "source_inventory_digest_pinned": False,
        "complete_canonical_accounting": False,
        "no_cap_hits": not cap_hits,
        "included_items_planned_or_unreachable": False,
        "provider_calls_made": 0,
        "downloads_made": 0,
    }
    if inventory is None or blockers:
        return receipt(status="blocked", blockers=blockers, checks=checks, manifest=None, stage_contract=None)
    checks["source_lineage_integrity"] = True
    checks["source_inventory_digest_pinned"] = True
    if hooks.corpus_exclusions is not None:
        policies, policy_blockers = hooks.corpus_exclusions(inventory)
        policies, policy_blockers = list(policies), [dict(item) for item in policy_blockers]
    else:
        policies, policy_blockers = [], []
    blockers.extend(policy_blockers)
    raw_program_items = inventory.get("program_items")
    if not isinstance(raw_program_items, list):
        blockers.append(corpus_freeze_blocker("program_items_missing"))
        raw_program_items = []
    program_items_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_item in raw_program_items:
        if not isinstance(raw_item, Mapping) or not isinstance(raw_item.get("program_item_id"), str):
            blockers.append(corpus_freeze_blocker("program_item_identity_invalid"))
            continue
        item_id = str(raw_item["program_item_id"])
        if item_id in program_items_by_id:
            blockers.append(corpus_freeze_blocker("duplicate_program_item_id", program_item_id=item_id))
            continue
        program_items_by_id[item_id] = raw_item
    exclusions, exclusion_metadata, exclusion_blockers = load_program_item_exclusions(
        workspace=workspace,
        inventory_sha256=str(inventory["inventory_sha256"]),
        program_items=program_items_by_id,
    )
    blockers.extend(exclusion_blockers)
    if blockers:
        return receipt(status="blocked", blockers=blockers, checks=checks, manifest=None, stage_contract=None)
    exclusion_by_item = {str(item["program_item_id"]): item for item in exclusions}
    selection: list[dict[str, Any]] = []
    for item_id in sorted(program_items_by_id):
        record, item_blockers = build_program_item_selection(
            item=program_items_by_id[item_id],
            records_by_id=records_by_id,
            operator_exclusions=exclusion_by_item,
            program_item_prefix=item_prefix,
        )
        blockers.extend(item_blockers)
        if record is not None:
            selection.append(record)
    complete_accounting = len(selection) == len(program_items_by_id) == len(raw_program_items)
    all_planned = all(
        item["selection_status"] != "included"
        or bool(item["planned_acquisition"])
        or bool(item["unreachable_source_path"])
        for item in selection
    )
    checks["complete_canonical_accounting"] = complete_accounting
    checks["included_items_planned_or_unreachable"] = all_planned
    if not complete_accounting:
        blockers.append(corpus_freeze_blocker("canonical_program_item_accounting_incomplete"))
    if not all_planned:
        blockers.append(corpus_freeze_blocker("included_item_without_phase4_path"))
    if blockers:
        return receipt(status="blocked", blockers=blockers, checks=checks, manifest=None, stage_contract=None)
    included_count = sum(item["selection_status"] == "included" for item in selection)
    operator_excluded_count = sum(item["selection_status"] == "operator_excluded" for item in selection)
    source_stage = ctx.stage(SOURCE_STAGE_ID)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": f"conference_pipeline.{ctx.runner_id}_corpus_freeze",
        "conference": settings.display_name or ctx.conference_display_name(default=ctx.slug),
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "status": "succeeded",
        "generated_at": io.now(),
        "source_lineage": {
            "source_receipt_path": str(Path("storage/workflow-runs") / run_id / source_stage.receipt_filename),
            "source_inventory_path": str(SOURCE_DISCOVERY_INVENTORY),
            "inventory_sha256": inventory["inventory_sha256"],
            "pinned_run_id": settings.pinned_run_id or run_id,
            "pinned_inventory_sha256": settings.pinned_inventory_sha256 or inventory["inventory_sha256"],
        },
        "policy_exclusions": policies,
        "operator_exclusions": exclusion_metadata,
        "counts": {
            "discovered_program_items": len(raw_program_items),
            "included_program_items": included_count,
            "operator_excluded_program_items": operator_excluded_count,
            "policy_excluded_program_items": 0,
            "policy_exclusion_postures": len(policies),
        },
        "command_policy": {"forbidden_cap_flags": list(settings.cap_flags), "cap_hits": cap_hits},
        "boundaries": {"provider_calls_made": 0, "downloads_made": 0, "secrets_copied": False},
        "report_pagination": {
            "page_size": settings.report_page_size,
            "visible_page_count": 1,
            "total_program_items": len(selection),
        },
        "program_items": selection,
    }
    manifest["selection_sha256"] = canonical_json_digest(
        {key: value for key, value in manifest.items() if key not in {"generated_at", "selection_sha256"}}
    )
    manifest_path = workspace / CORPUS_FREEZE_PATH
    report_path = workspace / CORPUS_FREEZE_REPORT_PATH
    io.write_json(manifest_path, manifest)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(program_item_report(manifest), encoding="utf-8")
    provisional_receipt = {
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": [owner_id],
        "owner_execution_evidence": {
            owner_id: {
                "owner_function": OWNER_FUNCTION,
                "manifest_path": str(CORPUS_FREEZE_PATH),
            }
        },
        "checks": checks,
    }
    contract = ctx.verify_contract(
        bind=True,
        stage_id=STAGE_ID,
        receipt=provisional_receipt,
        required_owner_ids=(owner_id,),
        mode="live",
        strict_execution=True,
    ).to_dict()
    if contract["status"] != "pass":
        blockers.append(corpus_freeze_blocker("final_stage_contract_failed"))
    return receipt(
        status="succeeded" if not blockers else "blocked",
        blockers=blockers,
        checks=checks,
        manifest=manifest,
        stage_contract=contract,
    )


# --------------------------------------------------------------------------
# Source-family freeze (the probed source-intake format)
# --------------------------------------------------------------------------


DEFAULT_FAMILY_PLAN_STAGES: tuple[str, ...] = ("download-corpus",)
DEFAULT_FAMILY_PLAN_NOTES = "Acquire source page and preserve provenance before downstream use."


@dataclass(frozen=True)
class FamilyAcquisitionPlan:
    """The planned acquisition for one group of source families."""

    family_ids: frozenset[str]
    stages: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class SourceFamilyFreezeSettings:
    """What a conference decides about the source-family freeze.

    ``display_name`` titles the report; ``family_plans`` are matched in order
    (first match wins; unmatched families get the default plan);
    ``checked_stages`` are the stage invocations checked for silent caps.
    """

    display_name: str
    family_plans: tuple[FamilyAcquisitionPlan, ...] = ()
    cap_flags: tuple[str, ...] = SILENT_CAP_FLAGS
    checked_stages: tuple[str, ...] = ("source-discovery", "corpus-freeze", "download-corpus")


def source_record_id(source: Mapping[str, Any]) -> str:
    return f"source-family:{source.get('id')}"


def acquisition_plan_for_source(
    source: Mapping[str, Any], plans: Sequence[FamilyAcquisitionPlan] = ()
) -> dict[str, Any]:
    source_id = str(source.get("id") or "")
    stages: list[str] = list(DEFAULT_FAMILY_PLAN_STAGES)
    notes = DEFAULT_FAMILY_PLAN_NOTES
    for plan in plans:
        if source_id in plan.family_ids:
            stages, notes = list(plan.stages), plan.notes
            break
    return {
        "planned_stages": stages,
        "source_url": source.get("final_url") or source.get("source_url"),
        "unreachable_source_path": None,
        "notes": notes,
    }


def command_contains_silent_caps(command: str, flags: Sequence[str] = SILENT_CAP_FLAGS) -> list[str]:
    return [flag for flag in flags if f" {flag} " in f" {command} " or f" {flag}=" in f" {command} "]


def load_family_exclusions(workspace: Path) -> tuple[list[dict[str, Any]], Path, list[str]]:
    """The operator's family/record exclusions, each with reason, decision, evidence and effect."""

    exclusions_path = workspace / OPERATOR_EXCLUSIONS_PATH
    if not exclusions_path.exists():
        return [], exclusions_path, []
    try:
        payload = json.loads(exclusions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], exclusions_path, [f"operator exclusions JSON is invalid: {exc}"]
    raw_items = payload.get("exclusions", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        return [], exclusions_path, ["operator exclusions must be a list or an object with exclusions[]"]
    exclusions: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            errors.append(f"operator exclusion {index} must be an object")
            continue
        target = str(item.get("record_id") or item.get("family_id") or "").strip()
        reason = str(item.get("reason") or "").strip()
        decision = str(item.get("decision_timestamp") or item.get("decision_context") or "").strip()
        source_note = str(item.get("source_note") or item.get("evidence_note") or "").strip()
        downstream_effect = str(item.get("downstream_effect") or "").strip()
        missing = [
            label
            for label, value in (
                ("record_id or family_id", target),
                ("reason", reason),
                ("decision_timestamp or decision_context", decision),
                ("source_note or evidence_note", source_note),
                ("downstream_effect", downstream_effect),
            )
            if not value
        ]
        if missing:
            errors.append(f"operator exclusion {index} missing: {', '.join(missing)}")
            continue
        exclusions.append({**item, "target": target})
    return exclusions, exclusions_path, errors


def source_family_report(selection: Mapping[str, Any], *, display_name: str) -> str:
    lines = [
        f"# {display_name} Corpus Freeze Report",
        "",
        f"Generated: `{selection['generated_at']}`",
        f"Workspace: `{selection['workspace']}`",
        f"Run id: `{selection['run_id']}`",
        "",
        "## Verdict",
        "",
        f"Status: `{selection['status']}`",
        f"Included records: `{selection['included_count']}`",
        f"Excluded records: `{selection['excluded_count']}`",
        f"Explicit exclusions file: `{selection['operator_exclusions_path']}`",
        "",
        "## Included Corpus Records",
        "",
        "| Record | Source family | Acquisition stages | URL | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in selection["included_records"]:
        lines.append(
            "| {record_id} | {family} | {stages} | {url} | {reason} |".format(
                record_id=record["record_id"],
                family=record["source_family_id"],
                stages=", ".join(record["planned_acquisition"]["planned_stages"]),
                url=record["planned_acquisition"]["source_url"],
                reason=record["inclusion_reason"],
            )
        )
    lines.extend(["", "## Excluded Corpus Records", ""])
    if selection["excluded_records"]:
        for record in selection["excluded_records"]:
            lines.append(
                "- `{record_id}`: {reason}".format(
                    record_id=record["record_id"],
                    reason=record.get("exclusion_reason") or "operator exclusion",
                )
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No provider calls were made.",
            "- No download-heavy acquisition was run.",
            "- No silent item-count caps were used.",
            "- Phase 4 must expand included source families into local artifacts with paths and digests.",
            "",
        ]
    )
    return "\n".join(lines)


_ZERO_SPEND = {"live_provider_calls_made": False, "download_heavy_acquisition_made": False, "secrets_copied": False}


def run_source_family_freeze(
    conventions: production_orchestrator.RunConventions,
    *,
    workspace: Path,
    run_id: str,
    stage: StageSpec,
    mode: str,
    env_file_status: Callable[[], Mapping[str, bool]],
    cost_approval: Callable[[], Mapping[str, Any]],
    checked_command: Callable[[str], str],
    settings: SourceFamilyFreezeSettings,
    write_json: Callable[[Path, dict[str, Any]], None],
) -> tuple[int, dict[str, Any]]:
    """Freeze every discovered source family under the orchestrator's own receipt writers.

    ``checked_command(stage_id)`` is the exact invocation recorded and
    checked for silent caps for each of ``settings.checked_stages``.
    Returns ``(exit_code, run_summary)``.
    """

    workspace = workspace.resolve()
    run_id = conventions.validate_run_id(run_id)
    started_at = conventions.now()
    conventions.run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file = dict(env_file_status())
    record = production_orchestrator.StageRecorder(
        conventions, workspace=workspace, run_id=run_id, stage=stage, mode=mode, started_at=started_at
    )
    if conventions.active_blockers(workspace, run_id=run_id):
        return record.halt_for_active_blocker()

    discovery_path = workspace / SOURCE_DISCOVERY_INVENTORY
    discovery = json.loads(discovery_path.read_text(encoding="utf-8")) if discovery_path.exists() else None
    if discovery is None:
        return record.blocked(
            checks={
                "env_file": env_file,
                "source_discovery_inventory_path": str(discovery_path),
                "source_discovery_inventory_present": False,
                **_ZERO_SPEND,
            },
            reason=f"missing source-discovery inventory: {discovery_path}",
            remediation="rerun source-discovery before corpus-freeze",
            blocker_class=BlockerClass.MISSING_RECEIPT,
        )

    sources = discovery.get("sources", [])
    discovery_blockers = discovery.get("blockers", [])
    if discovery.get("status") != "succeeded" or not isinstance(sources, list) or not sources or discovery_blockers:
        return record.blocked(
            checks={
                "env_file": env_file,
                "source_discovery_inventory_path": str(discovery_path),
                "source_discovery_status": discovery.get("status"),
                "source_discovery_blockers": discovery_blockers,
                **_ZERO_SPEND,
            },
            reason="source discovery must be succeeded and blocker-free before corpus freeze",
            remediation="repair source-discovery blockers before corpus-freeze",
        )

    exclusions, exclusions_path, exclusion_errors = load_family_exclusions(workspace)
    exclusion_by_target = {str(item["target"]): item for item in exclusions}
    unknown_targets = sorted(
        set(exclusion_by_target) - {source_record_id(source) for source in sources if isinstance(source, dict)}
    )
    if exclusion_errors or unknown_targets:
        reason_parts = list(exclusion_errors)
        if unknown_targets:
            reason_parts.append("unknown exclusion targets: " + ", ".join(unknown_targets))
        return record.blocked(
            checks={
                "env_file": env_file,
                "source_discovery_inventory_path": str(discovery_path),
                "operator_exclusions_path": str(exclusions_path),
                "operator_exclusion_errors": exclusion_errors,
                "unknown_exclusion_targets": unknown_targets,
                **_ZERO_SPEND,
            },
            reason="; ".join(reason_parts),
            remediation="repair operator_exclusions.json or remove invalid exclusions",
        )

    included_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        record_id = source_record_id(source)
        exclusion = exclusion_by_target.get(record_id)
        base_record = {
            "record_id": record_id,
            "source_family_id": source.get("id"),
            "label": source.get("label"),
            "source_url": source.get("source_url"),
            "final_url": source.get("final_url"),
            "discovery_status": source.get("discovery_status"),
            "auth_required": source.get("auth_required"),
        }
        if exclusion:
            excluded_records.append(
                {
                    **base_record,
                    "selection_status": "excluded",
                    "exclusion_reason": exclusion.get("reason"),
                    "decision_timestamp": exclusion.get("decision_timestamp"),
                    "decision_context": exclusion.get("decision_context"),
                    "source_note": exclusion.get("source_note") or exclusion.get("evidence_note"),
                    "downstream_effect": exclusion.get("downstream_effect"),
                }
            )
        else:
            included_records.append(
                {
                    **base_record,
                    "selection_status": "included",
                    "inclusion_reason": "default broad corpus policy: include all discovered source families unless explicitly excluded by the operator",
                    "planned_acquisition": acquisition_plan_for_source(source, settings.family_plans),
                }
            )

    checked_commands = {stage_id: checked_command(stage_id) for stage_id in settings.checked_stages}
    cap_hits = {
        stage_id: command_contains_silent_caps(command, settings.cap_flags)
        for stage_id, command in checked_commands.items()
    }
    cap_hits = {stage_id: hits for stage_id, hits in cap_hits.items() if hits}
    all_in_scope_planned = all(
        bool(item["planned_acquisition"].get("planned_stages"))
        or bool(item["planned_acquisition"].get("unreachable_source_path"))
        for item in included_records
    )
    selection_payload = {
        "schema_version": 1,
        "receipt_type": f"conference_pipeline.{conventions.context(workspace, run_id).runner_id}_corpus_freeze",
        "conference": discovery.get("conference"),
        "workspace": str(workspace),
        "run_id": run_id,
        "stage_id": stage.id,
        "generated_at": conventions.now(),
        "status": "succeeded",
        "source_discovery_inventory_path": str(discovery_path),
        "source_discovery_inventory_sha256": sha256_file(discovery_path),
        "operator_exclusions_path": str(exclusions_path),
        "operator_exclusions_present": exclusions_path.exists(),
        "policy": {
            "default": "include all discovered conference source families unless explicitly excluded",
            "operator_exclusions_required_for_removal": True,
            "silent_item_count_caps_rejected": True,
        },
        "included_count": len(included_records),
        "excluded_count": len(excluded_records),
        "included_records": included_records,
        "excluded_records": excluded_records,
        "command_policy": {
            "checked_commands": checked_commands,
            "forbidden_cap_flags": list(settings.cap_flags),
            "silent_item_cap_hits": cap_hits,
            "silent_item_caps_present": bool(cap_hits),
        },
        "boundaries": {
            "live_provider_calls_made": False,
            "download_heavy_acquisition_made": False,
            "metadata_only_completion": False,
            "secrets_copied": False,
        },
    }
    selection_path = workspace / CORPUS_FREEZE_PATH
    report_path = workspace / CORPUS_FREEZE_REPORT_PATH
    write_json(selection_path, selection_payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(source_family_report(selection_payload, display_name=settings.display_name), encoding="utf-8")

    checks = {
        "env_file": env_file,
        "source_discovery_inventory_path": str(discovery_path),
        "source_discovery_inventory_sha256": selection_payload["source_discovery_inventory_sha256"],
        "corpus_freeze_path": str(selection_path),
        "corpus_freeze_report_path": str(report_path),
        "included_count": len(included_records),
        "excluded_count": len(excluded_records),
        "operator_exclusions_path": str(exclusions_path),
        "operator_exclusions_present": exclusions_path.exists(),
        "all_in_scope_items_have_planned_acquisition_or_unreachable_path": all_in_scope_planned,
        "silent_item_caps_present": bool(cap_hits),
        "silent_item_cap_hits": cap_hits,
        "provider_cost_approval": dict(cost_approval()),
        **_ZERO_SPEND,
    }
    if not all_in_scope_planned or cap_hits:
        return record.blocked(
            checks=checks,
            reason="corpus freeze failed selection policy checks",
            remediation="repair corpus freeze acquisition plan or remove silent item-count caps",
        )
    return record.succeeded(checks=checks)

