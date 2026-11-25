"""Stage 4: download-corpus, the frame shared by every conference orchestrator.

Every conference acquires its frozen corpus differently (OpenReview auth,
virtual-site crawls, SlidesLive carry-forward, organizer YouTube channels,
registration-gated media, ...). Those differences are *lanes*: declared
:class:`~conference_pipeline.production.hooks.AcquisitionLane` hooks whose
implementations live in reusable modules under
``conference_pipeline.acquisition`` (each annotated with ``RECOMMENDED_FOR``
and listed in :mod:`conference_pipeline.acquisition.lane_catalog`). This
module owns everything around the lanes:

* **lane ordering** -- :func:`run_acquisition_lanes` runs every ``acquire``
  lane in declaration order, then every ``gate`` lane; a ``paid`` lane runs
  only when the caller's preflight authorizes it; each lane's blockers are
  appended in lane order (a lane may *resolve* an earlier lane's pending
  blocker); :func:`incomplete_lane_blockers` records the declared
  ``incomplete_blocker`` of every lane that did not complete;
  :func:`select_lane` picks one of several mutually exclusive lanes.
* **lineage** -- :func:`validate_pinned_program_lineage` accepts only the
  exact Stage 2/3 evidence that defines the Stage 4 scope (pinned source run,
  pinned inventory digest, the corpus-freeze receipt and selection digest).
* **capacity** -- :func:`read_storage_controls` and :func:`evaluate_capacity`
  reserve disk before any request.
* **prior-inventory merge** -- :func:`load_prior_inventory` and
  :func:`carry_forward_prior_artifacts` (a lane receives the prior inventory).
* **the Stage 4 ledger and receipts** -- :class:`FrozenCorpusState` for the
  frozen-corpus format, :func:`write_program_ledger` for the program-item
  ledger, :func:`write_stage_receipt` for immutable Stage 4 attempt receipts.

Two stage formats are supported, matching how a conference's Stage 3 freezes
its corpus:

* :func:`run_frozen_corpus_stage` (the default; a conference with no Stage 4
  hooks runs it with no lanes) binds a source-family corpus-freeze selection,
  runs the declared lanes against it and writes the download inventory,
  report and receipt through the orchestrator's own run conventions.
* :func:`run_program_ledger_stage` binds a pinned program-item corpus freeze,
  reserves capacity, builds the per-item acquisition ledger and runs the
  declared lanes against it before the strict stage contract decides.

Conference identity (receipt types, owner ids, pinned digests, report titles,
lanes) comes from the settings and the hooks; this module carries no
conference literal.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from conference_pipeline.indexing.corpus_vector_store import atomic_write_json
from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production.context import StageSpec
from conference_pipeline.production.hooks import (
    LANE_PHASE_ACQUIRE,
    LANE_PHASE_GATE,
    NO_HOOKS,
    AcquisitionLane,
    ConferenceHooks,
    LaneResult,
)
from conference_pipeline.production_preflight import canonical_json_digest
from conference_pipeline.provider_receipts import BlockerClass, sanitize_for_receipt
from conference_pipeline.receipt_lineage import sha256_file
from conference_pipeline.stage_script_registry import registry_sha256

STAGE_ID = "download-corpus"
SOURCE_DISCOVERY_STAGE_ID = "source-discovery"
CORPUS_FREEZE_STAGE_ID = "corpus-freeze"
DOWNLOAD_CORPUS_INVENTORY = Path("storage/download-corpus/download_corpus.json")
DOWNLOAD_CORPUS_REPORT = Path("storage/download-corpus/download_corpus_report.md")
CORPUS_FREEZE_SELECTION = Path("storage/corpus-freeze/corpus_freeze.json")
STORAGE_MANIFEST = Path("storage/storage_manifest.json")
STORAGE_POLICY = Path("storage/preflight-storage-policy.json")
DEFAULT_REPORT_PAGE_SIZE = 50
UNAUTHORIZED_PAID_LANE = "skipped_unauthorized_paid_lane"

def _ledger_primitives() -> Any:
    """The program-item ledger owner (imported lazily: its lanes import this frame)."""

    from conference_pipeline.acquisition import program_item_acquisition

    return program_item_acquisition


def canonical_ledger_digest(payload: Mapping[str, Any]) -> str:
    return _ledger_primitives().canonical_ledger_digest(payload)


# Passed as ``available_bytes`` to measure the workspace's free disk at call time.
AUTO_CAPACITY: Any = object()


def stage4_blocker(code: str, **details: Any) -> dict[str, Any]:
    """Compact, credential-free Stage 4 blocker data."""

    return sanitize_for_receipt({"code": code, "affected_stage": STAGE_ID, **details})


# --------------------------------------------------------------------------
# Lanes: ordering, paid-lane authorization, completion
# --------------------------------------------------------------------------


@dataclass
class LaneContext:
    """What a lane sees: the run, the invocation's options, the stage state and the auth posture."""

    workspace: Path
    run_id: str
    options: Any = None
    state: Any = None
    auth: AuthPosture | None = None


@dataclass(frozen=True)
class LaneRecord:
    """One lane's outcome in the order it ran (``status`` is ``ran`` or skipped)."""

    lane_id: str
    status: str
    result: LaneResult


def run_acquisition_lanes(
    lanes: Sequence[AcquisitionLane],
    *,
    ctx: Any,
    frozen_corpus: Any,
    prior_inventory: Any,
    blockers: list[dict[str, Any]],
    phase: str | None = None,
    authorize_paid: Callable[[AcquisitionLane], bool] | None = None,
    merge: Callable[[LaneResult], None] | None = None,
) -> list[LaneRecord]:
    """Run ``lanes`` (optionally one ``phase``) in declaration order.

    Each lane's blockers are appended to ``blockers`` as it returns, then the
    blocker codes it ``resolves`` are removed; ``merge(result)`` folds its
    artifacts and receipt fields into the stage state. A ``paid`` lane runs
    only when ``authorize_paid(lane)`` is true; otherwise it is recorded as
    skipped (not completed) and contributes nothing.
    """

    records: list[LaneRecord] = []
    for lane in lanes:
        if phase is not None and lane.phase != phase:
            continue
        if lane.paid and not (authorize_paid is not None and authorize_paid(lane)):
            records.append(LaneRecord(lane.lane_id, UNAUTHORIZED_PAID_LANE, LaneResult(completed=False)))
            continue
        result = lane.run(ctx, frozen_corpus, prior_inventory)
        if not isinstance(result, LaneResult):
            raise TypeError(f"acquisition lane {lane.lane_id!r} must return a LaneResult")
        blockers.extend(result.blockers)
        if result.resolves:
            resolved = set(result.resolves)
            blockers[:] = [blocker for blocker in blockers if blocker.get("code") not in resolved]
        if merge is not None:
            merge(result)
        records.append(LaneRecord(lane.lane_id, "ran", result))
    return records


def incomplete_lane_blockers(
    lanes: Sequence[AcquisitionLane],
    records: Sequence[LaneRecord],
    *,
    blocker: Callable[..., dict[str, Any]] = stage4_blocker,
) -> list[dict[str, Any]]:
    """The declared ``incomplete_blocker`` of every lane that did not complete, in lane order."""

    by_id = {record.lane_id: record for record in records}
    found: list[dict[str, Any]] = []
    for lane in lanes:
        if lane.incomplete_blocker is None:
            continue
        record = by_id.get(lane.lane_id)
        if record is None or not record.result.completed:
            found.append(blocker(lane.incomplete_blocker))
    return found


def select_lane(routes: Sequence[tuple[str, bool]], *, default: str) -> str:
    """The first enabled lane id among mutually exclusive ``(lane_id, enabled)`` routes."""

    for lane_id, enabled in routes:
        if enabled:
            return lane_id
    return default


def run_bounded(
    jobs: Sequence[Any],
    work: Callable[[Any], Any],
    *,
    worker_count: int,
    on_result: Callable[[Any], None],
    after_batch: Callable[[], None] | None = None,
    batch_factor: int | None = 8,
) -> None:
    """Run ``work`` over ``jobs`` on a thread pool with a bounded number of queued futures.

    At most ``worker_count * batch_factor`` futures are queued at a time
    (``batch_factor=None`` queues every job in one batch), so an unresponsive
    batch cannot hold an uncapped corpus in memory, and ``after_batch``
    persists partial progress between batches. Results are handed to
    ``on_result`` in completion order.
    """

    worker_count = max(1, int(worker_count))
    batch_size = len(jobs) if batch_factor is None else max(worker_count, worker_count * batch_factor)
    batch_size = max(1, batch_size)
    for batch_start in range(0, len(jobs), batch_size):
        batch = jobs[batch_start : batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(work, job) for job in batch]
            for future in as_completed(futures):
                on_result(future.result())
        if after_batch is not None:
            after_batch()


# --------------------------------------------------------------------------
# Prior-inventory merge
# --------------------------------------------------------------------------


def load_prior_inventory(path: Path) -> dict[str, Any] | None:
    """The previous Stage 4 inventory, or ``None`` when absent or unreadable."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def carry_forward_prior_artifacts(
    prior_inventory: Mapping[str, Any] | None,
    keep: Callable[[Mapping[str, Any]], bool],
    *,
    key: str = "direct_download_artifacts",
) -> list[dict[str, Any]]:
    """Copies of the prior inventory's ``key`` artifacts that ``keep`` accepts, in order."""

    if not isinstance(prior_inventory, Mapping):
        return []
    previous = prior_inventory.get(key)
    if not isinstance(previous, list):
        return []
    return [dict(artifact) for artifact in previous if isinstance(artifact, dict) and keep(artifact)]


# --------------------------------------------------------------------------
# Capacity
# --------------------------------------------------------------------------


def free_disk_bytes(workspace: Path, available_bytes: Any = AUTO_CAPACITY) -> int | None:
    """``available_bytes`` as given, or the workspace's free disk for :data:`AUTO_CAPACITY`."""

    if available_bytes is AUTO_CAPACITY:
        try:
            return shutil.disk_usage(workspace).free
        except OSError:
            return None
    return available_bytes


def _is_byte_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def evaluate_capacity(
    reservation: Mapping[str, Any],
    available_bytes: Any,
    *,
    blocker: Callable[..., dict[str, Any]] = stage4_blocker,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare a byte reservation to the available bytes before any request."""

    capacity = {
        **reservation,
        "available_bytes": available_bytes if _is_byte_count(available_bytes) else None,
        "positive_available_bytes": _is_byte_count(available_bytes) and available_bytes > 0,
    }
    blockers: list[dict[str, Any]] = []
    if capacity["available_bytes"] is None:
        capacity["status"] = "blocked_unknown"
        blockers.append(blocker("disk_capacity_unknown"))
    elif int(capacity["available_bytes"]) < int(capacity["required_bytes"]):
        capacity["status"] = "blocked_insufficient"
        blockers.append(
            blocker(
                "disk_capacity_insufficient",
                required_bytes=capacity["required_bytes"],
                available_bytes=capacity["available_bytes"],
            )
        )
    else:
        capacity["status"] = "pass_pre_request"
    return capacity, blockers


def read_storage_controls(
    workspace: Path,
    *,
    read_json_object: Callable[..., tuple[dict[str, Any] | None, list[dict[str, Any]]]],
    manifest_path: Path = STORAGE_MANIFEST,
    policy_path: Path = STORAGE_POLICY,
    blocker: Callable[..., dict[str, Any]] = stage4_blocker,
) -> tuple[dict[str, Any] | None, int, list[dict[str, Any]]]:
    """Load the storage manifest and the bounded safety margin used by the local preflight."""

    manifest, manifest_blockers = read_json_object(
        workspace / manifest_path,
        missing_code="storage_manifest_missing",
        invalid_code="storage_manifest_invalid",
    )
    blockers = [blocker(str(item.get("code") or "storage_manifest_invalid")) for item in manifest_blockers]
    if manifest is None:
        return None, 0, blockers
    policy, policy_blockers = read_json_object(
        workspace / policy_path,
        missing_code="storage_policy_missing",
        invalid_code="storage_policy_invalid",
    )
    blockers.extend(blocker(str(item.get("code") or "storage_policy_invalid")) for item in policy_blockers)
    safety_margin = policy.get("safety_margin_bytes") if isinstance(policy, Mapping) else None
    if not isinstance(safety_margin, int) or isinstance(safety_margin, bool) or safety_margin < 0:
        blockers.append(blocker("storage_policy_safety_margin_invalid"))
        safety_margin = 0
    return manifest, safety_margin, blockers


# --------------------------------------------------------------------------
# Lineage: the pinned Stage 2/3 evidence that defines the Stage 4 scope
# --------------------------------------------------------------------------


def receipt_workspace_matches(receipt: Mapping[str, Any], workspace: Path) -> bool:
    """Permit legacy fixtures without workspace data but reject foreign evidence."""

    value = receipt.get("workspace")
    if value is None:
        return True
    try:
        return Path(str(value)).resolve() == workspace.resolve()
    except OSError:
        return False


def corpus_selection_digest(manifest: Mapping[str, Any]) -> str:
    """Recompute the Stage 3 selection digest using its publisher's contract."""

    return canonical_json_digest(
        {key: value for key, value in manifest.items() if key not in {"generated_at", "selection_sha256"}}
    )


@dataclass(frozen=True)
class LedgerConventions:
    """The orchestrator's own Stage 4 receipt, lineage and contract conventions.

    ``finalize_receipt(receipt, *, workspace, run_id, stage_id, lineage,
    profile)`` attaches lineage and output digests under the orchestrator's
    code identity; ``verify_contract(**kwargs)`` is its lineage-requiring
    stage-contract verifier; ``read_json_object(path, *, missing_code,
    invalid_code)`` reads a control artifact; ``validate_source_lineage(*,
    workspace, run_id, pinned_run_id, pinned_inventory_sha256)`` validates the
    pinned Stage 2 inventory. ``receipt_type`` and ``owner_id`` name the
    orchestrator's Stage 4 receipt and owner of record.
    """

    stages: Mapping[str, StageSpec]
    run_dir: Callable[[Path, str], Path]
    now: Callable[[], str]
    write_json: Callable[[Path, dict[str, Any]], None]
    write_bytes: Callable[[Path, bytes], None]
    finalize_receipt: Callable[..., dict[str, Any]]
    verify_contract: Callable[..., Any]
    read_json_object: Callable[..., tuple[dict[str, Any] | None, list[dict[str, Any]]]]
    receipt_type: str
    owner_id: str
    validate_source_lineage: Callable[..., tuple[Any, Any, list[dict[str, Any]]]] | None = None
    stage_id: str = STAGE_ID

    @property
    def stage(self) -> StageSpec:
        return self.stages[self.stage_id]


def validate_pinned_program_lineage(
    conventions: LedgerConventions,
    *,
    workspace: Path,
    run_id: str,
    pinned_run_id: str,
    pinned_inventory_sha256: str,
    expected_program_item_count: int,
    corpus_freeze_path: Path = CORPUS_FREEZE_SELECTION,
    blocker: Callable[..., dict[str, Any]] = stage4_blocker,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Accept only the exact Stage 2/3 evidence that defines the Stage 4 scope.

    Returns ``(corpus_manifest, corpus_freeze_receipt, blockers)``.
    """

    if run_id != pinned_run_id:
        return None, None, [blocker("pinned_source_run_mismatch")]
    if conventions.validate_source_lineage is None:
        raise ValueError("pinned program lineage needs a source-lineage validator")
    _inventory, _records, blockers = conventions.validate_source_lineage(
        workspace=workspace,
        run_id=run_id,
        pinned_run_id=pinned_run_id,
        pinned_inventory_sha256=pinned_inventory_sha256,
    )
    if blockers:
        return None, None, [blocker(str(item.get("code") or "source_lineage_invalid")) for item in blockers]

    read_json_object = conventions.read_json_object
    target = conventions.run_dir(workspace, run_id)
    source_receipt, source_blockers = read_json_object(
        target / conventions.stages[SOURCE_DISCOVERY_STAGE_ID].receipt_filename,
        missing_code="source_stage_receipt_missing",
        invalid_code="source_stage_receipt_invalid",
    )
    blockers.extend(source_blockers)
    if source_receipt is not None and not receipt_workspace_matches(source_receipt, workspace):
        blockers.append(blocker("source_receipt_workspace_mismatch"))

    freeze_stage = conventions.stages[CORPUS_FREEZE_STAGE_ID]
    corpus_receipt, receipt_blockers = read_json_object(
        target / freeze_stage.receipt_filename,
        missing_code="corpus_freeze_receipt_missing",
        invalid_code="corpus_freeze_receipt_invalid",
    )
    blockers.extend(receipt_blockers)
    if corpus_receipt is None:
        return None, None, [blocker(str(item.get("code") or "corpus_freeze_receipt_invalid")) for item in blockers]
    receipt_checks = (
        (corpus_receipt.get("run_id") == run_id, "corpus_freeze_receipt_run_id_mismatch"),
        (corpus_receipt.get("stage_id") == CORPUS_FREEZE_STAGE_ID, "corpus_freeze_receipt_stage_identity_invalid"),
        (corpus_receipt.get("stage_order") == freeze_stage.order, "corpus_freeze_receipt_stage_order_invalid"),
        (corpus_receipt.get("mode") == "live", "corpus_freeze_receipt_not_live"),
        (corpus_receipt.get("status") == "succeeded", "corpus_freeze_receipt_not_succeeded"),
        (not corpus_receipt.get("blockers"), "corpus_freeze_receipt_has_blockers"),
        (
            isinstance(corpus_receipt.get("stage_contract"), Mapping)
            and corpus_receipt["stage_contract"].get("status") == "pass",
            "corpus_freeze_receipt_contract_not_passing",
        ),
        (receipt_workspace_matches(corpus_receipt, workspace), "corpus_freeze_receipt_workspace_mismatch"),
    )
    blockers.extend(blocker(code) for valid, code in receipt_checks if not valid)

    corpus_manifest, manifest_blockers = read_json_object(
        workspace / corpus_freeze_path,
        missing_code="corpus_freeze_manifest_missing",
        invalid_code="corpus_freeze_manifest_invalid",
    )
    blockers.extend(manifest_blockers)
    if corpus_manifest is None:
        return None, None, [blocker(str(item.get("code") or "corpus_freeze_manifest_invalid")) for item in blockers]
    source_lineage = corpus_manifest.get("source_lineage")
    selection_sha256 = corpus_manifest.get("selection_sha256")
    receipt_manifest = corpus_receipt.get("corpus_manifest")
    manifest_checks = (
        (corpus_manifest.get("run_id") == run_id, "corpus_freeze_manifest_run_id_mismatch"),
        (corpus_manifest.get("stage_id") == CORPUS_FREEZE_STAGE_ID, "corpus_freeze_manifest_stage_identity_invalid"),
        (corpus_manifest.get("status") == "succeeded", "corpus_freeze_manifest_not_succeeded"),
        (isinstance(source_lineage, Mapping), "phase_02_source_lineage_missing"),
        (
            isinstance(source_lineage, Mapping) and source_lineage.get("inventory_sha256") == pinned_inventory_sha256,
            "phase_02_inventory_digest_mismatch",
        ),
        (
            isinstance(source_lineage, Mapping) and source_lineage.get("pinned_run_id") == pinned_run_id,
            "phase_02_pinned_run_mismatch",
        ),
        (isinstance(selection_sha256, str) and bool(selection_sha256), "phase_03_selection_digest_missing"),
        (selection_sha256 == corpus_selection_digest(corpus_manifest), "phase_03_selection_digest_invalid"),
        (
            isinstance(receipt_manifest, Mapping) and receipt_manifest.get("selection_sha256") == selection_sha256,
            "corpus_freeze_receipt_selection_digest_mismatch",
        ),
    )
    blockers.extend(blocker(code) for valid, code in manifest_checks if not valid)

    raw_items = corpus_manifest.get("program_items")
    if not isinstance(raw_items, list):
        blockers.append(blocker("corpus_program_items_missing"))
        raw_items = []
    seen_ids: set[str] = set()
    complete_rows = True
    for raw_item in raw_items:
        item_id = raw_item.get("program_item_id") if isinstance(raw_item, Mapping) else None
        if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
            complete_rows = False
            continue
        seen_ids.add(item_id)
        if raw_item.get("selection_status") != "included":
            complete_rows = False
            continue
        planned = raw_item.get("planned_acquisition")
        unreachable = raw_item.get("unreachable_source_path")
        if not (isinstance(planned, list) and planned) and not isinstance(unreachable, Mapping):
            complete_rows = False
    if len(raw_items) != expected_program_item_count or len(seen_ids) != expected_program_item_count:
        blockers.append(
            blocker(
                "complete_program_item_accounting_invalid",
                expected=expected_program_item_count,
                observed=len(seen_ids),
            )
        )
    if not complete_rows:
        blockers.append(blocker("corpus_selection_rows_invalid"))
    operator_exclusions = corpus_manifest.get("operator_exclusions")
    if isinstance(operator_exclusions, Mapping) and operator_exclusions.get("present"):
        blockers.append(blocker("operator_exclusions_not_permitted_for_download_scope"))
    return corpus_manifest, corpus_receipt, blockers


# --------------------------------------------------------------------------
# Immutable Stage 4 attempt receipts
# --------------------------------------------------------------------------


def _numbered_path(target: Path, stem: str, marker: str, suffix: str) -> Path:
    path = target / f"{stem}.{marker}-{suffix}.json"
    sequence = 1
    while path.exists():
        path = target / f"{stem}.{marker}-{suffix}-{sequence}.json"
        sequence += 1
    return path


def write_stage_receipt(
    conventions: LedgerConventions,
    *,
    workspace: Path,
    run_id: str,
    status: str,
    body: Mapping[str, Any],
    checks: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    stage_contract: Mapping[str, Any] | None,
    lineage: Mapping[str, Any] | None = None,
    profile: str | None = None,
    receipt_type: str | None = None,
    immutable_attempts: bool = True,
    merge_summary: bool = True,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Persist one Stage 4 receipt and the run summary; return ``(path, receipt, summary)``.

    With ``immutable_attempts`` a prior canonical receipt is never erased: a
    success first preserves it as a ``superseded-*`` copy, and a blocked
    attempt beside it is written as an ``attempt-*`` receipt. ``body``
    carries the lane's owner evidence and ledger fields. With
    ``merge_summary`` the summary keeps every other stage's prior entry.
    """

    stage = conventions.stage
    target = conventions.run_dir(workspace, run_id)
    canonical_path = target / stage.receipt_filename
    superseded_prior_path: Path | None = None
    receipt_path = canonical_path
    if immutable_attempts:
        try:
            prior_bytes = canonical_path.read_bytes()
        except OSError:
            prior_bytes = None
        suffix = conventions.now().replace(":", "").replace("-", "")
        stem = Path(stage.receipt_filename).stem
        if status == "succeeded" and prior_bytes is not None:
            superseded_prior_path = _numbered_path(target, stem, "superseded", suffix)
            conventions.write_bytes(superseded_prior_path, prior_bytes)
        elif prior_bytes is not None:
            receipt_path = _numbered_path(target, stem, "attempt", suffix)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": receipt_type or conventions.receipt_type,
        "receipt_id": f"{run_id}:{stage.id}:{conventions.now()}",
        "run_id": run_id,
        "workspace": str(workspace.resolve()),
        "stage_id": stage.id,
        "stage_order": stage.order,
        "mode": "live",
        "status": status,
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": [conventions.owner_id],
        **dict(body),
        "checks": sanitize_for_receipt(dict(checks)),
        "stage_contract": sanitize_for_receipt(dict(stage_contract)) if stage_contract else None,
        "blockers": sanitize_for_receipt(blockers),
        "secrets_copied": False,
    }
    if immutable_attempts:
        receipt["supersedes_receipt_path"] = str(canonical_path) if superseded_prior_path else None
        receipt["superseded_prior_receipt_path"] = str(superseded_prior_path) if superseded_prior_path else None
    receipt = conventions.finalize_receipt(
        receipt, workspace=workspace, run_id=run_id, stage_id=stage.id, lineage=lineage, profile=profile
    )
    conventions.write_json(receipt_path, sanitize_for_receipt(receipt))
    stages: list[Any] = []
    if merge_summary:
        try:
            prior_summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_summary = {}
        prior_stages = prior_summary.get("stages") if isinstance(prior_summary, Mapping) else []
        stages = (
            [item for item in prior_stages if not isinstance(item, Mapping) or item.get("stage_id") != stage.id]
            if isinstance(prior_stages, list)
            else []
        )
    stages.append(receipt)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "workspace": str(workspace.resolve()),
        "mode": "live",
        "status": status,
        "halt_status": "halted" if status == "blocked" else None,
        "halt_stage_id": stage.id if status == "blocked" else None,
        "stages": stages,
        "summary_path": str(target / "summary.json"),
    }
    conventions.write_json(target / "summary.json", sanitize_for_receipt(summary))
    return receipt_path, receipt, summary


def verify_lane_contract(
    conventions: LedgerConventions,
    *,
    workspace: Path,
    run_id: str,
    owner_evidence: Mapping[str, Any],
    checks: Mapping[str, Any],
    profile: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind and strictly verify a provisional Stage 4 receipt under the owner of record."""

    provisional = {
        **dict(extra or {}),
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": [conventions.owner_id],
        "owner_execution_evidence": {conventions.owner_id: dict(owner_evidence)},
        "checks": checks,
    }
    kwargs: dict[str, Any] = {}
    if profile is not None:
        kwargs["profile"] = profile
    return conventions.verify_contract(
        bind=True,
        workspace=workspace,
        run_id=run_id,
        stage_id=conventions.stage_id,
        receipt=provisional,
        required_owner_ids=(conventions.owner_id,),
        mode="live",
        strict_execution=True,
        **kwargs,
    ).to_dict()


# --------------------------------------------------------------------------
# Format 1: frozen source-family corpus (the default)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthPosture:
    """A conference's Stage 4 auth posture (the ``download-corpus.auth`` hook result).

    ``check_fields`` are recorded in every Stage 4 receipt's checks,
    ``payload_fields`` in the download inventory; ``blockers`` are added
    before any lane runs; ``credentials`` are lane-facing only and never
    recorded.
    """

    check_fields: Mapping[str, Any] = field(default_factory=dict)
    payload_fields: Mapping[str, Any] = field(default_factory=dict)
    blockers: Sequence[Mapping[str, Any]] = ()
    credentials: Mapping[str, Any] = field(default_factory=dict)


NO_AUTH = AuthPosture()


def target_key(target: Mapping[str, Any]) -> tuple[str, str]:
    """A discovered target's identity: ``(target_kind, target_url)``."""

    return (str(target.get("target_kind")), str(target.get("target_url")))


@dataclass
class FrozenCorpusState:
    """The in-progress Stage 4 download inventory the lanes extend.

    ``payload`` is the inventory written to disk (its artifact, target and
    blocker lists are the same list objects held here); ``counts`` and
    ``flags`` are lane-reported facts the frame folds into the final counts.
    """

    workspace: Path
    run_id: str
    inventory_path: Path
    payload: dict[str, Any]
    local_artifacts: list[dict[str, Any]]
    direct_artifacts: list[dict[str, Any]]
    discovered_targets: list[dict[str, Any]]
    blockers: list[dict[str, Any]]
    now: Callable[[], str]
    write_json: Callable[[Path, dict[str, Any]], None]
    counts: dict[str, int] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    unique_targets: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    downloaded_direct_urls: set[str] = field(default_factory=set)

    def persist(self) -> None:
        self.payload["updated_at"] = self.now()
        self.write_json(self.inventory_path, self.payload)

    def record_artifact(self, artifact: dict[str, Any], *, direct: bool) -> None:
        if direct:
            self.direct_artifacts.append(artifact)
        self.local_artifacts.append(artifact)

    def dedupe_targets(self) -> None:
        """Collapse the discovered targets to their first occurrence per identity (in place)."""

        for target in self.discovered_targets:
            self.unique_targets.setdefault(target_key(target), target)
        self.discovered_targets[:] = list(self.unique_targets.values())

    def add_discovered_target(self, target: dict[str, Any]) -> bool:
        key = target_key(target)
        if key in self.unique_targets:
            return False
        self.unique_targets[key] = target
        self.discovered_targets.append(target)
        return True

    def count_succeeded(self, kind: str) -> int:
        return sum(
            1
            for artifact in self.direct_artifacts
            if artifact.get("artifact_kind") == kind and artifact.get("status") == "succeeded"
        )

    def merge(self, result: LaneResult) -> None:
        for artifact in result.artifacts:
            self.direct_artifacts.append(dict(artifact))
            self.local_artifacts.append(dict(artifact))
        self.payload.update(result.receipt_fields)


# Lane-reported count keys the frame reads.
WORKSHOP_PAPER_PDF_COUNT = "workshop_paper_pdf_count"
PAPER_METADATA_ARTIFACT_COUNT = "paper_metadata_artifact_count"
# Set by a lane that re-derived paper artifacts from its own direct downloads
# (item-page expansion): the paper count then excludes workshop PDFs.
PAPER_COUNT_FROM_DIRECT_ONLY = "paper_count_from_direct_downloads_only"

DEFAULT_TARGET_KINDS = ("paper_page", "session_page", "paper_pdf", "source_archive", "media_asset")


@dataclass(frozen=True)
class FrozenCorpusSettings:
    """Where the frozen-corpus stage reads and writes, and how it words its receipts.

    ``inventory_receipt_type`` names the download inventory; ``report_title``
    heads the review report; ``initial_payload(args, workspace)`` adds
    conference fields to the inventory before any lane runs and
    ``check_payload_keys`` are inventory fields mirrored into the receipt
    checks; ``load_env(env_file)`` returns the env-file status flags and
    ``cost_approval(args)`` the provider-cost approval record.
    """

    inventory_receipt_type: str
    report_title: str
    load_env: Callable[[Path | None], Mapping[str, bool]]
    cost_approval: Callable[[Any], Mapping[str, Any]]
    initial_payload: Callable[[Any, Path], Mapping[str, Any]] | None = None
    check_payload_keys: tuple[str, ...] = ()
    corpus_freeze_selection: Path = CORPUS_FREEZE_SELECTION
    download_inventory: Path = DOWNLOAD_CORPUS_INVENTORY
    download_report: Path = DOWNLOAD_CORPUS_REPORT
    target_kinds: tuple[str, ...] = DEFAULT_TARGET_KINDS
    unreachable_blocker_codes: tuple[str, ...] = ("source_page_download_failed",)
    blocked_remediation: str = "inspect storage/download-corpus/download_corpus.json, resolve the lane blockers, then rerun download-corpus"
    write_json: Callable[[Path, dict[str, Any]], None] = atomic_write_json


def load_frozen_corpus(workspace: Path, path: Path = CORPUS_FREEZE_SELECTION) -> tuple[dict[str, Any] | None, Path]:
    selection_path = workspace / path
    if not selection_path.exists():
        return None, selection_path
    return json.loads(selection_path.read_text(encoding="utf-8")), selection_path


def render_frozen_corpus_report(inventory: Mapping[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"Generated: `{inventory['generated_at']}`",
        f"Workspace: `{inventory['workspace']}`",
        f"Run id: `{inventory['run_id']}`",
        "",
        "## Verdict",
        "",
        f"Status: `{inventory['status']}`",
        f"Source families attempted: `{inventory['source_family_count']}`",
        f"Local artifacts: `{inventory['local_artifact_count']}`",
        f"Direct download artifacts: `{inventory['direct_download_artifact_count']}`",
        f"Media artifacts: `{inventory['media_artifact_count']}`",
        f"Paper artifacts: `{inventory['paper_artifact_count']}`",
        f"Paper metadata artifacts: `{inventory.get('paper_metadata_artifact_count', 0)}`",
        f"Discovered targets: `{inventory['discovered_target_count']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = inventory.get("blockers", [])
    if blockers:
        for blocker in blockers:
            lines.append(
                "- `{code}`: {reason}".format(
                    code=blocker.get("code") or "blocked",
                    reason=blocker.get("reason") or blocker.get("source_url") or "",
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
            "- Secret values, cookies, and auth headers were not recorded.",
            "- Paper PDFs, HTML pages, OpenReview metadata, and source pages do not substitute for required video/media artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def run_frozen_corpus_stage(
    conventions: production_orchestrator.RunConventions,
    args: Any,
    stage: StageSpec,
    *,
    settings: FrozenCorpusSettings,
    hooks: ConferenceHooks = NO_HOOKS,
) -> tuple[int, dict[str, Any]]:
    """Run the declared lanes against a succeeded source-family corpus freeze.

    ``args`` carries ``workspace``, ``run_id``, ``mode`` and ``env_file`` plus
    whatever options the lanes read. ``hooks.auth(args)`` (an
    :class:`AuthPosture`) is resolved first; ``hooks.acquisition_lanes`` run
    in phase order (``acquire`` lanes, the paper-artifact gate, then ``gate``
    lanes). Receipts and summaries are the orchestrator's own
    (``conventions``). Returns ``(exit_code, run_summary)``.
    """

    workspace = Path(args.workspace).resolve()
    run_id = conventions.validate_run_id(args.run_id)
    now = conventions.now
    started_at = now()
    conventions.run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file_status = dict(settings.load_env(args.env_file))
    auth = hooks.auth(args) if hooks.auth is not None else NO_AUTH
    record = production_orchestrator.StageRecorder(
        conventions, workspace=workspace, run_id=run_id, stage=stage, mode=args.mode, started_at=started_at
    )
    if conventions.active_blockers(workspace, run_id=run_id):
        return record.halt_for_active_blocker()

    selection, selection_path = load_frozen_corpus(workspace, settings.corpus_freeze_selection)
    if selection is None:
        return record.blocked(
            checks={
                "env_file": env_file_status,
                **auth.check_fields,
                "corpus_freeze_path": str(selection_path),
                "corpus_freeze_present": False,
                "live_provider_calls_made": False,
                "secrets_copied": False,
            },
            reason=f"missing corpus-freeze selection: {selection_path}",
            remediation=f"rerun corpus-freeze before {stage.id}",
            blocker_class=BlockerClass.MISSING_RECEIPT,
        )
    included_records = selection.get("included_records", [])
    selection_blockers = selection.get("blockers", [])
    if (
        selection.get("status") != "succeeded"
        or not isinstance(included_records, list)
        or not included_records
        or selection_blockers
    ):
        return record.blocked(
            checks={
                "env_file": env_file_status,
                **auth.check_fields,
                "corpus_freeze_path": str(selection_path),
                "corpus_freeze_status": selection.get("status"),
                "corpus_freeze_blockers": selection_blockers,
                "live_provider_calls_made": False,
                "secrets_copied": False,
            },
            reason="corpus freeze must be succeeded and include at least one blocker-free record",
            remediation=f"repair corpus-freeze blockers before {stage.id}",
        )

    inventory_path = workspace / settings.download_inventory
    report_path = workspace / settings.download_report
    prior_inventory = load_prior_inventory(inventory_path)
    local_artifacts: list[dict[str, Any]] = []
    direct_artifacts: list[dict[str, Any]] = []
    discovered_targets: list[dict[str, Any]] = []
    stage_blockers: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": settings.inventory_receipt_type,
        "conference": selection.get("conference"),
        "workspace": str(workspace),
        "run_id": run_id,
        "stage_id": stage.id,
        "generated_at": now(),
        "status": "running",
        "corpus_freeze_path": str(selection_path),
        "corpus_freeze_sha256": sha256_file(selection_path),
        "source_family_count": len(included_records),
        "local_artifacts": local_artifacts,
        "direct_download_artifacts": direct_artifacts,
        "discovered_targets": discovered_targets,
        "blockers": stage_blockers,
        "boundaries": {
            "live_provider_calls_made": False,
            "metadata_only_completion": False,
            "paper_or_html_substitutes_for_media": False,
            "secrets_copied": False,
        },
        **dict(auth.payload_fields),
        **dict(settings.initial_payload(args, workspace) if settings.initial_payload else {}),
    }
    state = FrozenCorpusState(
        workspace=workspace,
        run_id=run_id,
        inventory_path=inventory_path,
        payload=payload,
        local_artifacts=local_artifacts,
        direct_artifacts=direct_artifacts,
        discovered_targets=discovered_targets,
        blockers=stage_blockers,
        now=now,
        write_json=settings.write_json,
    )
    stage_blockers.extend(dict(blocker) for blocker in auth.blockers)
    lane_ctx = LaneContext(workspace=workspace, run_id=run_id, options=args, state=state, auth=auth)
    lanes = tuple(hooks.acquisition_lanes)
    run_acquisition_lanes(
        lanes,
        ctx=lane_ctx,
        frozen_corpus=selection,
        prior_inventory=prior_inventory,
        blockers=stage_blockers,
        phase=LANE_PHASE_ACQUIRE,
        merge=state.merge,
    )

    workshop_paper_pdf_count = int(state.counts.get(WORKSHOP_PAPER_PDF_COUNT, 0))
    media_artifact_count = state.count_succeeded("media_asset")
    paper_artifact_count = state.count_succeeded("paper_pdf")
    if not state.flags.get(PAPER_COUNT_FROM_DIRECT_ONLY):
        paper_artifact_count += workshop_paper_pdf_count
    if paper_artifact_count == 0:
        stage_blockers.append(
            {
                "code": "paper_pdf_downloads_missing",
                "reason": "no paper PDF or source-file artifact was acquired; metadata alone is not sufficient",
                "retryable": True,
            }
        )
    run_acquisition_lanes(
        lanes,
        ctx=lane_ctx,
        frozen_corpus=selection,
        prior_inventory=prior_inventory,
        blockers=stage_blockers,
        phase=LANE_PHASE_GATE,
        merge=state.merge,
    )

    local_success_count = sum(1 for artifact in local_artifacts if artifact.get("status") == "succeeded")
    status = "blocked" if stage_blockers else "succeeded"
    payload.update(
        {
            "status": status,
            "local_artifact_count": local_success_count,
            "direct_download_artifact_count": sum(
                1 for artifact in direct_artifacts if artifact.get("status") == "succeeded"
            ),
            "media_artifact_count": media_artifact_count,
            "paper_artifact_count": paper_artifact_count,
            "workshop_paper_pdf_count": workshop_paper_pdf_count,
            "paper_metadata_artifact_count": int(state.counts.get(PAPER_METADATA_ARTIFACT_COUNT, 0)),
            "unreachable_source_count": sum(
                1 for blocker in stage_blockers if blocker.get("code") in settings.unreachable_blocker_codes
            ),
            "discovered_target_count": len(discovered_targets),
            "target_counts": {
                kind: sum(1 for target in discovered_targets if target.get("target_kind") == kind)
                for kind in settings.target_kinds
            },
            "blockers": stage_blockers,
            "auth_posture": {
                "any_auth_required": any(bool(blocker.get("auth_required")) for blocker in stage_blockers),
                "auth_required_blockers": [
                    blocker for blocker in stage_blockers if bool(blocker.get("auth_required"))
                ],
                "secret_values_recorded": False,
            },
        }
    )
    state.persist()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_frozen_corpus_report(payload, title=settings.report_title), encoding="utf-8")

    checks = {
        "env_file": env_file_status,
        **auth.check_fields,
        "corpus_freeze_path": str(selection_path),
        "corpus_freeze_sha256": payload["corpus_freeze_sha256"],
        "download_corpus_inventory_path": str(inventory_path),
        "download_corpus_report_path": str(report_path),
        "source_family_count": len(included_records),
        "attempted_source_count": len(local_artifacts),
        "local_artifact_count": local_success_count,
        "direct_download_artifact_count": payload["direct_download_artifact_count"],
        "unreachable_source_count": payload["unreachable_source_count"],
        "media_artifact_count": media_artifact_count,
        "paper_artifact_count": paper_artifact_count,
        "workshop_paper_pdf_count": workshop_paper_pdf_count,
        "paper_metadata_artifact_count": payload["paper_metadata_artifact_count"],
        "discovered_target_count": len(discovered_targets),
        "target_counts": payload["target_counts"],
        **{key: payload.get(key) for key in settings.check_payload_keys},
        "blockers": stage_blockers,
        "auth_posture": payload["auth_posture"],
        "provider_cost_approval": dict(settings.cost_approval(args)),
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    if stage_blockers:
        return record.blocked(
            checks=checks,
            reason=f"{stage.id} blocked: "
            + ", ".join(sorted({str(blocker.get("code") or "blocked") for blocker in stage_blockers})),
            remediation=settings.blocked_remediation,
        )
    return record.succeeded(checks=checks)


# --------------------------------------------------------------------------
# Format 2: pinned program-item ledger
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramLedgerSettings:
    """The program-ledger stage's paths, wording and owner evidence.

    ``execution_status_key`` names the check that records the stage
    execution status; ``owner_function`` is recorded as the owner of record's
    entry point; ``report_title`` heads the review report.
    """

    execution_status_key: str
    owner_function: str
    report_title: str
    ledger_path: Path = DOWNLOAD_CORPUS_INVENTORY
    report_path: Path = DOWNLOAD_CORPUS_REPORT
    report_page_size: int = DEFAULT_REPORT_PAGE_SIZE
    corpus_freeze_path: Path = CORPUS_FREEZE_SELECTION
    storage_manifest_path: Path = STORAGE_MANIFEST
    storage_policy_path: Path = STORAGE_POLICY


@dataclass
class ProgramLedgerState:
    """The program-item ledger the lanes extend, plus the receipt checks they report into."""

    workspace: Path
    run_id: str
    ledger: dict[str, Any]
    manifest: dict[str, Any]
    checks: dict[str, Any]
    settings: ProgramLedgerSettings
    write_json: Callable[[Path, dict[str, Any]], None]
    facts: dict[str, Any] = field(default_factory=dict)

    def checkpoint(self, payload: Mapping[str, Any] | None = None) -> None:
        """Persist the ledger as-is (a lane's in-flight checkpoint)."""

        self.write_json(self.workspace / self.settings.ledger_path, dict(payload if payload is not None else self.ledger))

    def checkpoint_with_digest(self) -> None:
        """Re-digest and persist the ledger so a later lane reads exactly this scope."""

        self.ledger["manifest_sha256"] = canonical_ledger_digest(self.ledger)
        self.write_json(self.workspace / self.settings.ledger_path, self.ledger)

    def merge(self, result: LaneResult) -> None:
        self.checks.update(result.receipt_fields)


def render_program_ledger_report(ledger: Mapping[str, Any], *, title: str, default_page_size: int = DEFAULT_REPORT_PAGE_SIZE) -> str:
    """Render a small review page without truncating the JSON ledger."""

    rows = ledger.get("program_item_results")
    rows = rows if isinstance(rows, list) else []
    pagination = ledger.get("report_pagination")
    page_size = int(pagination.get("page_size", default_page_size)) if isinstance(pagination, Mapping) else default_page_size
    counts = ledger.get("counts") if isinstance(ledger.get("counts"), Mapping) else {}
    lines = [
        f"# {title}",
        "",
        f"Run id: `{ledger.get('lineage', {}).get('run_id', '') if isinstance(ledger.get('lineage'), Mapping) else ''}`",
        f"Status: `{ledger.get('status')}`",
        "",
        "## Accounting",
        "",
        f"- Program-item results: `{counts.get('program_item_result_count', 0)}`",
        f"- Deduplicated request results: `{counts.get('request_result_count', 0)}`",
        f"- Validated local artifacts: `{counts.get('artifact_count', 0)}`",
        "",
        "## Visible Program-Item Page",
        "",
        f"Visible program-item IDs: {min(page_size, len(rows))} of {len(rows)} (page size {page_size}; complete ledger is JSON).",
        "",
    ]
    lines.extend(
        f"- `{row.get('program_item_id')}` — `{row.get('blocker', {}).get('code') if isinstance(row, Mapping) and isinstance(row.get('blocker'), Mapping) else 'pending'}`"
        for row in rows[:page_size]
        if isinstance(row, Mapping)
    )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- The complete JSON ledger is not limited by report pagination.",
            "- This pre-request scaffold made no provider call or network download.",
            "- Official paywalled video policy remains visible without deleting program items.",
            "",
        ]
    )
    return "\n".join(lines)


def write_program_ledger(
    conventions: LedgerConventions,
    *,
    workspace: Path,
    ledger: dict[str, Any],
    settings: ProgramLedgerSettings,
    make_report_dir: bool = False,
) -> None:
    """Persist the Stage 4 ledger and its review report."""

    conventions.write_json(workspace / settings.ledger_path, ledger)
    report_path = workspace / settings.report_path
    if make_report_dir:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_program_ledger_report(ledger, title=settings.report_title, default_page_size=settings.report_page_size),
        encoding="utf-8",
    )


def program_ledger_receipt(
    conventions: LedgerConventions,
    *,
    workspace: Path,
    run_id: str,
    status: str,
    blockers: list[dict[str, Any]],
    checks: Mapping[str, Any],
    ledger: Mapping[str, Any] | None,
    stage_contract: Mapping[str, Any] | None,
    settings: ProgramLedgerSettings,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist an immutable program-ledger Stage 4 attempt and return the stage outcome."""

    counts = ledger.get("counts") if isinstance(ledger, Mapping) and isinstance(ledger.get("counts"), Mapping) else {}
    artifact_count = int(counts.get("artifact_count") or 0)
    request_count = int(counts.get("request_result_count") or 0)
    blocker_count = (
        len(ledger.get("blockers", []))
        if isinstance(ledger, Mapping) and isinstance(ledger.get("blockers"), list)
        else 0
    )
    talk_harvest = ledger.get("talk_harvest") if isinstance(ledger, Mapping) else None
    paid_lane = talk_harvest.get("tavily_lane") if isinstance(talk_harvest, Mapping) else None
    provider_calls_made = int(paid_lane.get("provider_calls_made") or 0) if isinstance(paid_lane, Mapping) else 0
    actual_provider_cost_usd = float(paid_lane.get("actual_cost_usd") or 0.0) if isinstance(paid_lane, Mapping) else 0.0
    try:
        report_sha256 = sha256((workspace / settings.report_path).read_bytes()).hexdigest()
    except OSError:
        report_sha256 = None
    lineage_fields = ledger.get("lineage") if isinstance(ledger, Mapping) and isinstance(ledger.get("lineage"), Mapping) else None
    body = {
        "owner_execution_evidence": {
            conventions.owner_id: {
                "owner_function": settings.owner_function,
                "ledger_path": str(settings.ledger_path),
                "provider_calls_made": provider_calls_made,
                "actual_provider_cost_usd": actual_provider_cost_usd,
                "downloads_made": artifact_count,
                "transport_created": bool(ledger.get("public_transport_executed")) if isinstance(ledger, Mapping) else False,
            }
        },
        "download_ledger": {
            "path": str(settings.ledger_path),
            "manifest_sha256": ledger.get("manifest_sha256") if ledger else None,
            "report_sha256": report_sha256,
            "phase_02_inventory_sha256": lineage_fields.get("phase_02_inventory_sha256") if lineage_fields else None,
            "phase_03_selection_sha256": lineage_fields.get("phase_03_selection_sha256") if lineage_fields else None,
            "request_count": request_count,
            "artifact_count": artifact_count,
            "blocker_count": blocker_count,
            "public_item_blocker_count": ledger.get("public_item_blocker_count") if isinstance(ledger, Mapping) else None,
        },
        "provider_calls_made": provider_calls_made,
        "actual_provider_cost_usd": actual_provider_cost_usd,
        "downloads_made": artifact_count,
    }
    receipt_path, receipt, summary = write_stage_receipt(
        conventions,
        workspace=workspace,
        run_id=run_id,
        status=status,
        body=body,
        checks=checks,
        blockers=blockers,
        stage_contract=stage_contract,
        lineage=lineage,
    )
    return {
        "status": status,
        "receipt_path": str(receipt_path),
        "receipt": receipt,
        "summary": summary,
        "ledger": dict(ledger) if ledger else None,
    }


def run_program_ledger_stage(
    conventions: LedgerConventions,
    *,
    workspace: Path,
    run_id: str,
    settings: ProgramLedgerSettings,
    lanes: Sequence[AcquisitionLane] = (),
    pinned_run_id: str,
    pinned_inventory_sha256: str,
    expected_program_item_count: int,
    cap_hits: Sequence[str] = (),
    available_bytes: Any = AUTO_CAPACITY,
    lane_options: Any = None,
    authorize_paid: Callable[[AcquisitionLane], bool] | None = None,
    validate_run_id: Callable[[str], str] = lambda value: value,
) -> dict[str, Any]:
    """Run the declared lanes against a pinned program-item corpus freeze.

    The frame validates the pinned Stage 2/3 lineage, rejects silent item
    caps (``cap_hits``), reserves disk capacity, builds (or resumes) the
    per-item ledger and writes it before any lane runs. Lanes run only on a
    built ledger; every lane that did not complete records its
    ``incomplete_blocker``. The strict stage contract decides success, and
    the receipt is an immutable Stage 4 attempt.
    """

    safe_run_id = validate_run_id(run_id)
    resolved_workspace = workspace.resolve()
    manifest, _corpus_receipt, blockers = validate_pinned_program_lineage(
        conventions,
        workspace=resolved_workspace,
        run_id=safe_run_id,
        pinned_run_id=pinned_run_id,
        pinned_inventory_sha256=pinned_inventory_sha256,
        expected_program_item_count=expected_program_item_count,
        corpus_freeze_path=settings.corpus_freeze_path,
    )
    if cap_hits:
        blockers.append(stage4_blocker("silent_item_cap_rejected", flags=list(cap_hits)))
    checks: dict[str, Any] = {
        settings.execution_status_key: "blocked_pre_request",
        "phase_02_inventory_sha256_pinned": False,
        "phase_02_inventory_sha256": None,
        "phase_03_selection_sha256_pinned": False,
        "phase_03_selection_sha256": None,
        "program_item_count": 0,
        "program_item_result_count": 0,
        "complete_program_item_accounting": False,
        "no_cap_hits": not cap_hits,
        "disk_capacity": {"status": "not_evaluated", "positive_available_bytes": False},
        "storage_manifest_sha256": None,
        "download_corpus_ledger_sha256": None,
        "validated_local_artifacts": False,
        "artifact_payload_digests_verified": False,
        "complete_result_or_blocker_accounting": False,
        "policy_exclusions_visible": False,
        "focused_recovery_disposition": "not_started",
        "provider_and_recovery_evidence_sanitized": True,
        "secrets_copied": False,
    }
    ledger: dict[str, Any] | None = None
    if manifest is not None and not blockers:
        storage_manifest, safety_margin, storage_blockers = read_storage_controls(
            resolved_workspace,
            read_json_object=conventions.read_json_object,
            manifest_path=settings.storage_manifest_path,
            policy_path=settings.storage_policy_path,
        )
        blockers.extend(storage_blockers)
        if storage_manifest is not None and not blockers:
            selection_sha256 = str(manifest["selection_sha256"])
            runs_root = Path("storage/workflow-runs") / safe_run_id
            lineage = {
                "run_id": safe_run_id,
                "stage_02_receipt_path": str(runs_root / conventions.stages[SOURCE_DISCOVERY_STAGE_ID].receipt_filename),
                "phase_02_inventory_sha256": pinned_inventory_sha256,
                "stage_03_receipt_path": str(runs_root / conventions.stages[CORPUS_FREEZE_STAGE_ID].receipt_filename),
                "phase_03_selection_sha256": selection_sha256,
                "storage_manifest_sha256": canonical_json_digest(storage_manifest),
            }
            primitives = _ledger_primitives()
            try:
                fresh_ledger = primitives.build_initial_ledger(
                    lineage=lineage,
                    selection_rows=manifest["program_items"],
                    policy_exclusions=manifest.get("policy_exclusions", []),
                    expected_program_item_count=expected_program_item_count,
                    report_page_size=settings.report_page_size,
                )
                reservation = primitives.derive_required_byte_reservation(
                    ledger=fresh_ledger,
                    storage_manifest=storage_manifest,
                    safety_margin_bytes=safety_margin,
                )
                ledger, _resume = primitives.resume_compatible_ledger(
                    workspace=resolved_workspace,
                    fresh_ledger=fresh_ledger,
                    ledger_path=settings.ledger_path,
                )
            except (TypeError, ValueError) as exc:
                blockers.append(stage4_blocker("pre_request_ledger_invalid", reason=type(exc).__name__))
            else:
                capacity, capacity_blockers = evaluate_capacity(
                    reservation, free_disk_bytes(resolved_workspace, available_bytes)
                )
                blockers.extend(capacity_blockers)
                ledger["capacity_preflight"] = capacity
                ledger["status"] = "blocked_pre_request"
                ledger["blockers"] = list(blockers)
                ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
                write_program_ledger(
                    conventions, workspace=resolved_workspace, ledger=ledger, settings=settings, make_report_dir=True
                )
                checks.update(
                    {
                        "phase_02_inventory_sha256_pinned": True,
                        "phase_02_inventory_sha256": pinned_inventory_sha256,
                        "phase_03_selection_sha256_pinned": True,
                        "phase_03_selection_sha256": selection_sha256,
                        "program_item_count": len(manifest["program_items"]),
                        "program_item_result_count": ledger["counts"]["program_item_result_count"],
                        "complete_program_item_accounting": len(manifest["program_items"])
                        == ledger["counts"]["program_item_result_count"]
                        == expected_program_item_count,
                        "disk_capacity": capacity,
                        "storage_manifest_sha256": lineage["storage_manifest_sha256"],
                        "download_corpus_ledger_sha256": ledger["manifest_sha256"],
                        "policy_exclusions_visible": bool(ledger["policy_exclusions"]),
                    }
                )
    if ledger is not None:
        assert manifest is not None
        state = ProgramLedgerState(
            workspace=resolved_workspace,
            run_id=safe_run_id,
            ledger=ledger,
            manifest=manifest,
            checks=checks,
            settings=settings,
            write_json=conventions.write_json,
        )
        records = run_acquisition_lanes(
            lanes,
            ctx=LaneContext(workspace=resolved_workspace, run_id=safe_run_id, options=lane_options, state=state),
            frozen_corpus=manifest,
            prior_inventory=None,
            blockers=blockers,
            authorize_paid=authorize_paid,
            merge=state.merge,
        )
        ledger = state.ledger
        blockers.extend(incomplete_lane_blockers(lanes, records))
        item_blockers = ledger.get("blockers") if isinstance(ledger.get("blockers"), list) else []
        ledger["blockers"] = [*item_blockers, *blockers]
        execution_ready = not blockers
        checks[settings.execution_status_key] = "pass" if execution_ready else "blocked"
        ledger["status"] = "succeeded" if execution_ready else "blocked_pending_required_lanes"
        ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
        write_program_ledger(conventions, workspace=resolved_workspace, ledger=ledger, settings=settings)
        checks["download_corpus_ledger_sha256"] = ledger["manifest_sha256"]
    contract = verify_lane_contract(
        conventions,
        workspace=resolved_workspace,
        run_id=safe_run_id,
        owner_evidence={
            "owner_function": settings.owner_function,
            "ledger_path": str(settings.ledger_path),
            "transport_created": bool(ledger.get("public_transport_executed")) if ledger else False,
        },
        checks=checks,
    )
    if contract["status"] != "pass":
        blockers.append(stage4_blocker("final_stage_contract_incomplete"))
        if ledger is not None:
            checks[settings.execution_status_key] = "blocked"
            item_blockers = ledger.get("blockers") if isinstance(ledger.get("blockers"), list) else []
            ledger["blockers"] = [*item_blockers, stage4_blocker("final_stage_contract_incomplete")]
            ledger["status"] = "blocked_pending_required_lanes"
            ledger["manifest_sha256"] = canonical_ledger_digest(ledger)
            write_program_ledger(conventions, workspace=resolved_workspace, ledger=ledger, settings=settings)
            checks["download_corpus_ledger_sha256"] = ledger["manifest_sha256"]
    return program_ledger_receipt(
        conventions,
        workspace=resolved_workspace,
        run_id=safe_run_id,
        status="succeeded" if not blockers and contract["status"] == "pass" else "blocked",
        blockers=blockers,
        checks=checks,
        ledger=ledger,
        stage_contract=contract,
        settings=settings,
    )


__all__ = [
    "AUTO_CAPACITY",
    "AuthPosture",
    "FrozenCorpusSettings",
    "FrozenCorpusState",
    "LaneContext",
    "LaneRecord",
    "LedgerConventions",
    "ProgramLedgerSettings",
    "ProgramLedgerState",
    "carry_forward_prior_artifacts",
    "evaluate_capacity",
    "incomplete_lane_blockers",
    "load_prior_inventory",
    "read_storage_controls",
    "run_acquisition_lanes",
    "run_bounded",
    "run_frozen_corpus_stage",
    "run_program_ledger_stage",
    "select_lane",
    "stage4_blocker",
    "validate_pinned_program_lineage",
    "write_program_ledger",
    "write_stage_receipt",
]
