"""W1 characterization: pin the current CVPR 2026 runner Stages 1-9.

Each test drives one stage entry point of ``scripts/run_cvpr2026_production.py``
in-process -- the same function ``main()`` dispatches to, with the kwargs
``main()`` passes plus the stage function's own injection seams for fakes --
against a temp copy of the synthetic ``cvpr_shaped`` fixture (placed at
``tmp_path/conferences/cvpr2026`` so the CVPR stage-owner runtime contracts
bind). Every network/provider/LLM/embedding/subprocess seam is faked
(:func:`tests.orchestrator_characterization.offline_guard` fails the test if a
real one is reached), and a stable snapshot of the receipt and its outputs is
compared with ``tests/golden/orchestrators/cvpr2026/<name>.json``.

The CVPR runner binds several stages to fixed, conference-scale contracts (the
pinned r10 inventory digest, the exact r4/r5 run ids and census counts). Where
a stage's success path cannot run on a small synthetic fixture, its fail-closed
block and/or pure helpers are pinned instead; the gaps are listed in
``CHARACTERIZATION_GAPS`` below.

Fixture builders: the large, stage-specific builders are imported from
``tests/test_cvpr2026_production_runner.py`` (never its ``test_*`` functions);
small helpers are copied here so edits to that file cannot silently shift the
goldens. See ``ORCHESTRATOR_GOLDEN_UPDATE=1`` to re-record.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import tests.orchestrator_characterization as characterization
from conference_pipeline.production_preflight import (
    PLANNED_PROVIDER_PROBES,
    ProviderModelChoice,
    ProviderProbeResult,
    ProviderStageAllocation,
    build_approval_scope_fingerprint,
)
from conference_pipeline.source_discovery import (
    build_candidate_frontier,
    build_canonical_inventory,
    immutable_id,
    publish_canonical_inventory,
)
from conference_pipeline.source_discovery import (
    canonical_json_digest as source_discovery_canonical_json_digest,
)
from tests.lineage_fixtures import stamp_upstream_stage_receipts
from tests.orchestrator_characterization import (
    FROZEN_NOW,
    assert_golden,
    load_runner,
    materialize_conference_workspace,
    offline_guard,
    read_json,
    snapshot_payload,
    stable_snapshot,
    write_json,
)

CONFERENCE = "cvpr2026"
FIXTURE = "cvpr_shaped"
RUN_ID = "w1-cvpr-fixture-run"
FAKE_SECRET = "w1-cvpr-fake-secret-never-recorded"

# Stages whose success path is not reachable on a small synthetic fixture.
CHARACTERIZATION_GAPS: dict[str, str] = {
    "download-corpus": (
        "success unreachable: the CVPR download-corpus runtime contract requires "
        "checks.program_item_count == 14583 (live census); the all-lanes fixture run "
        "is pinned blocked on exactly that check"
    ),
    "download-corpus --registration-media": (
        "success unreachable: the registration-media contract requires the exact pinned r3 "
        "predecessor files and publisher digests; pinned blocked on that check after a "
        "complete zero-link recovery"
    ),
    "download-corpus --event-media-coverage": (
        "success unreachable: the event-media contract requires the immutable live r4 "
        "overlay lineage; pinned blocked on that check after a complete 2,693-route scan"
    ),
    "corpus-freeze / download-corpus kwargs": (
        "main() relies on the default pinned_run_id/pinned_inventory_sha256 (live r10 "
        "inventory); the fixture passes its own run id and digest through those kwargs"
    ),
    "transcription-ocr": (
        "success pinned only after rebinding the pinned r5 Stage-4 digest constants "
        "(runner, adapter, stage_contracts) and a temp registry copy to a synthetic handoff; "
        "the local caption adapter is faked"
    ),
    "podcast": "main() uses the real subprocess command runner; the stage's command_runner seam is faked",
    "runner-preflight resume": "not applicable: main() has no --resume path for runner-preflight",
}


# --------------------------------------------------------------------------
# Small helpers copied from tests/test_cvpr2026_production_runner.py
# --------------------------------------------------------------------------


def overlay_runner_workspace_contract(workspace: Path) -> None:
    """The minimum local CVPR runner contract (copied from write_runner_workspace).

    The cvpr_shaped fixture carries the CVPR quirk records; the runner's own
    preflight additionally needs a virtual-site boundary and a storage policy.
    """

    (workspace / "config.yaml").write_text(
        "conference: {slug: cvpr2026, name: CVPR 2026}\n"
        "urls: {virtual: https://cvpr.thecvf.com/virtual/2026}\n",
        encoding="utf-8",
    )
    write_json(
        workspace / "storage" / "storage_manifest.json",
        {
            # No absolute workspace path: the prepared estimate digests the
            # manifest, and a tmp path would make every estimate digest volatile.
            "artifact_families": [
                {"id": "papers", "path": "storage/papers", "storage_class": "backup"},
                {"id": "media", "path": "storage/media", "storage_class": "backup"},
            ],
        },
    )
    write_json(
        workspace / "storage" / "preflight-storage-policy.json",
        {
            "corpus_scope": "all-discovered-cvpr2026",
            "estimator_version": "fixture-v1",
            "remaining_families": {"paper_pdf": 700, "media": 300},
            "safety_margin_bytes": 100,
        },
    )


def write_immutable_approval(workspace: Path, run_id: str, prepared: dict[str, Any]) -> Path:
    provider_models = tuple(
        ProviderModelChoice(provider=str(item["provider"]), model=str(item["model"]))
        for item in prepared["provider_models"]
    )
    allocations = tuple(
        ProviderStageAllocation(provider=key.split(":", 1)[0], stage=key.split(":", 1)[1], ceiling_usd=float(cost) + 1.0)
        for key, cost in prepared["provider_stage_projection"].items()
    )
    total_ceiling_usd = float(prepared["total_projected_cost_usd"]) + 1.0
    approval = {
        "run_id": run_id,
        "corpus_scope": prepared["corpus_scope"],
        "paid_stages": prepared["paid_stages"],
        "provider_models": [item.__dict__ for item in provider_models],
        "estimate_digest": prepared["estimate_digest"],
        "scope_fingerprint": build_approval_scope_fingerprint(
            run_id=run_id,
            corpus_scope=str(prepared["corpus_scope"]),
            paid_stages=tuple(str(stage) for stage in prepared["paid_stages"]),
            provider_models=provider_models,
            estimate_digest=str(prepared["estimate_digest"]),
            total_ceiling_usd=total_ceiling_usd,
            provider_stage_allocations=allocations,
        ),
        "total_ceiling_usd": total_ceiling_usd,
        "provider_stage_allocations": [item.__dict__ for item in allocations],
    }
    return write_json(workspace / "storage" / "workflow-runs" / run_id / "provider-approval.json", approval)


def write_capacity_estimate(workspace: Path, run_id: str, prepared: dict[str, Any]) -> Path:
    inputs = prepared["storage_estimate_inputs"]
    return write_json(
        workspace / "storage" / "workflow-runs" / run_id / "preflight-storage-estimate.json",
        {
            "corpus_scope": inputs["corpus_scope"],
            "inventory_digest": inputs["inventory_digest"],
            "manifest_digest": inputs["manifest_digest"],
            "prepared_estimate_digest": prepared["estimate_digest"],
            "remaining_families": {"paper_pdf": 700, "media": 300},
            "projected_remaining_bytes": 1000,
            "estimator_version": "fixture-v1",
            "safety_margin_bytes": 100,
        },
    )


def fixture_credentials() -> dict[str, str]:
    return {name: FAKE_SECRET for spec in PLANNED_PROVIDER_PROBES for name in spec.credential_names}


def _must_not_run(label: str):
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"resume re-ran {label} instead of accepting the persisted receipt")

    return _raise


# --------------------------------------------------------------------------
# Path-bound digests
# --------------------------------------------------------------------------

# The CVPR preflight estimate digest is computed over the absolute workspace
# path, so it (and every authority digest derived from it) differs per temp
# dir and can never match a file digest. These keys name such digests in the
# run directory's authority files; their values are tokenized by key name.
PATH_BOUND_DIGEST_KEYS = (
    "estimate_digest",
    "prepared_estimate_digest",
    "scope_fingerprint",
    "authority_digest",
    "frontier_authority_digest",
    "capacity_policy_digest",
)


@contextlib.contextmanager
def _extra_digest_tokens(extra: Mapping[str, str]) -> Iterator[None]:
    """Teach the shared DigestIndex about path-bound digests (test-local, no infra edit)."""

    base = characterization.DigestIndex

    class _IndexWithTokens(base):  # type: ignore[misc, valid-type]
        def __init__(self, workspace: Path, **kwargs: Any) -> None:
            super().__init__(workspace, **kwargs)
            self.tokens.update(extra)

    characterization.DigestIndex = _IndexWithTokens  # type: ignore[misc]
    try:
        yield
    finally:
        characterization.DigestIndex = base  # type: ignore[misc]


# Other files whose self-recorded digest covers absolute paths (and so is
# volatile per temp dir): workspace-relative path -> digest keys.
PATH_BOUND_DIGEST_FILES = {
    "analysis/analysis_manifest.json": ("content_sha256",),
}


def collect_path_bound_digests(workspace: Path) -> dict[str, str]:
    tokens: dict[str, str] = {}

    def walk(node: Any, rel: str, keys: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in keys and isinstance(value, str) and len(value) == 64:
                    tokens.setdefault(value, f"<{key}:{rel}>")
                walk(value, rel, keys)
        elif isinstance(node, list):
            for value in node:
                walk(value, rel, keys)

    def scan(path: Path, keys: tuple[str, ...]) -> None:
        with contextlib.suppress(ValueError, OSError):
            walk(json.loads(path.read_text(encoding="utf-8")), path.relative_to(workspace).as_posix(), keys)

    runs = workspace / "storage" / "workflow-runs"
    if runs.is_dir():
        for path in sorted(runs.rglob("*.json")):
            if any(name in path.name for name in ("estimate", "approval", "authority")):
                scan(path, PATH_BOUND_DIGEST_KEYS)
    for rel, keys in PATH_BOUND_DIGEST_FILES.items():
        if (workspace / rel).is_file():
            scan(workspace / rel, keys)
    return tokens


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


class CvprHarness:
    def __init__(self, runner: ModuleType, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.runner = runner
        self.workspace = workspace
        self.monkeypatch = monkeypatch
        # Blocker/evidence receipt file names embed utc_now(); freeze it.
        monkeypatch.setattr(runner, "utc_now", lambda: FROZEN_NOW)
        # The canonical source inventory stamps generated_at into its own digest.
        import conference_pipeline.source_discovery as source_discovery

        monkeypatch.setattr(source_discovery, "_utc_now", lambda: FROZEN_NOW)

    def run_dir(self, run_id: str = RUN_ID) -> Path:
        return self.runner.run_dir(self.workspace, run_id)

    def paths(self, run_id: str = RUN_ID) -> dict[str, Path]:
        """The three preflight authority paths main() derives for every stage."""

        target = self.run_dir(run_id)
        return {
            "prepared_estimate_path": target / self.runner.PREPARED_ESTIMATE_FILENAME,
            "approval_path": target / self.runner.APPROVAL_FILENAME,
            "capacity_estimate_path": target / self.runner.CAPACITY_ESTIMATE_FILENAME,
        }

    def receipt_file(self, stage: str, run_id: str = RUN_ID) -> Path:
        return self.run_dir(run_id) / self.runner.STAGES[stage].receipt_filename

    def receipt(self, stage: str, run_id: str = RUN_ID) -> dict[str, Any]:
        return read_json(self.receipt_file(stage, run_id))

    def snap(self, stage: str, run_id: str = RUN_ID, **kwargs: Any) -> dict[str, Any]:
        return self.snap_receipt(self.receipt(stage, run_id), **kwargs)

    def snap_receipt(self, receipt: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        with _extra_digest_tokens(collect_path_bound_digests(self.workspace)):
            return stable_snapshot(receipt, workspace=self.workspace, **kwargs)

    def snap_payload(self, payload: Any, **kwargs: Any) -> Any:
        with _extra_digest_tokens(collect_path_bound_digests(self.workspace)):
            return snapshot_payload(payload, workspace=self.workspace, **kwargs)

    def assert_secret_free(self) -> None:
        for path in self.workspace.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".md", ".jsonl", ".txt"}:
                assert FAKE_SECRET not in path.read_text(encoding="utf-8", errors="replace"), path

    # Stage 1 -----------------------------------------------------------------

    def prepare_authority(self, run_id: str = RUN_ID) -> dict[str, Any]:
        prepared = self.runner.prepare_preflight_estimate(self.workspace, run_id)
        write_immutable_approval(self.workspace, run_id, prepared)
        write_capacity_estimate(self.workspace, run_id, prepared)
        return prepared

    def run_preflight(self, run_id: str = RUN_ID, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "credential_values": fixture_credentials(),
            "probe": lambda spec: ProviderProbeResult.succeeded(spec),
            "reload_environment": _must_not_run("preflight environment reload"),
            "rebuild_client": _must_not_run("preflight client rebuild"),
            "available_bytes": 1_100,
        }
        kwargs.update(overrides)
        return self.runner.run_runner_preflight(workspace=self.workspace, run_id=run_id, **self.paths(run_id), **kwargs)


@pytest.fixture
def cvpr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = load_runner(CONFERENCE)
    workspace = materialize_conference_workspace(FIXTURE, tmp_path, CONFERENCE)
    overlay_runner_workspace_contract(workspace)
    with offline_guard(monkeypatch, runner):
        yield CvprHarness(runner, workspace, monkeypatch)


# --------------------------------------------------------------------------
# Stage 1: runner-preflight
# --------------------------------------------------------------------------


def test_cvpr_runner_preflight_success_receipt(cvpr: CvprHarness) -> None:
    cvpr.prepare_authority()
    outcome = cvpr.run_preflight()

    assert outcome["status"] == "succeeded", outcome.get("summary")
    assert_golden(CONFERENCE, "stage01_runner_preflight__succeeded", cvpr.snap("runner-preflight"))
    cvpr.assert_secret_free()


def test_cvpr_runner_preflight_missing_authority_blocks(cvpr: CvprHarness) -> None:
    outcome = cvpr.run_preflight()

    assert outcome["status"] == "blocked"
    assert_golden(CONFERENCE, "stage01_runner_preflight__blocked_missing_authority", cvpr.snap("runner-preflight"))


# --------------------------------------------------------------------------
# Stage 2: source-discovery
# --------------------------------------------------------------------------

SD_SOURCE_ID = immutable_id("cvpr2026:source:program", "program-one")
SD_ITEM_ID = immutable_id("cvpr2026:program-item", "item-one")


def _source_frontier() -> dict[str, Any]:
    return build_candidate_frontier(
        [
            {
                "canonical_pair_key": f"{SD_ITEM_ID}|{SD_SOURCE_ID}",
                "source_record_ids": [SD_ITEM_ID, SD_SOURCE_ID],
                "source_snapshot_digests": ["a" * 64, "b" * 64],
                "relation_kind": "program_item_to_paper",
                "inclusion_status": "pending_llm",
                "reason": "fixture frontier is current and complete",
            }
        ]
    )


def fake_source_collector(calls: list[str]):
    """Collector publishing a one-record canonical inventory (copied from the runner tests)."""

    frontier = _source_frontier()

    def collect(*, workspace: Path, run_id: str) -> dict[str, Any]:
        calls.append("collection")
        inventory = build_canonical_inventory(
            run_id=run_id,
            family_snapshots={
                "program-schedule": {"snapshot_id": "program-fixture", "sha256": "a" * 64, "status": "live_valid"}
            },
            source_records=[
                {
                    "source_record_id": SD_SOURCE_ID,
                    "native_ids": ["program-one"],
                    "aliases": ["https://fixtures.example/program-one"],
                    "family_id": "program-schedule",
                    "snapshot_sha256": "a" * 64,
                }
            ],
            program_items=[
                {
                    "program_item_id": SD_ITEM_ID,
                    "aliases": ["item-one"],
                    "source_record_ids": [SD_SOURCE_ID],
                    "field_evidence": {"title": {"authority": "program-schedule", "snapshot_sha256": "a" * 64}},
                }
            ],
            edges=[],
            seed_diffs=[],
            taxonomy={"taxonomy_sha256": "c" * 64, "blockers": []},
            candidate_frontier=frontier,
        )
        publish_canonical_inventory(
            workspace / "storage" / "source-discovery" / "source_discovery_inventory.json", inventory
        )
        return {
            "status": "succeeded",
            "expected_family_ids": ["program-schedule"],
            "field_values": {},
            "candidate_evidence": {
                frontier["pairs"][0]["canonical_pair_key"]: [
                    {"field": "title", "source_record_id": SD_ITEM_ID, "value": "Fixture program title"},
                    {"field": "title", "source_record_id": SD_SOURCE_ID, "value": "Fixture paper title"},
                ]
            },
        }

    return collect


def fake_join_adapter_factory(workspace: Path, run_id: str, calls: list[str]):
    from conference_pipeline.model_governance import default_model

    class Adapter:
        provider = "openai"
        model = default_model("source_discovery.cross_source_join")

        def invoke(self, _candidate: dict[str, Any]) -> dict[str, Any]:
            calls.append("provider")
            receipt = {
                "receipt_type": "conference_pipeline.provider_call_receipt",
                "run_id": run_id,
                "stage_id": "source-discovery",
                "task_id": "source_discovery.cross_source_join",
                "provider": self.provider,
                "model": self.model,
                "receipt_path": "storage/provider-calls/source-discovery/fixture.json",
                "request_sha256": "d" * 64,
                "estimated_cost_usd": 0.0,
                "secrets_copied": False,
            }
            write_json(workspace / receipt["receipt_path"], receipt)
            return {"decision": "join", "confidence": 0.99, "conflict_state": "none", "receipt": receipt}

    return lambda **_kwargs: Adapter()


def run_source_discovery(cvpr: CvprHarness, calls: list[str], *, resume: bool = False, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "collector": fake_source_collector(calls),
        "fresh_volatile_preflight": lambda authority: calls.append("fresh")
        or {"status": "succeeded", "frontier_authority_digest": authority["authority_digest"]},
        "provider_adapter_factory": fake_join_adapter_factory(cvpr.workspace, RUN_ID, calls),
    }
    kwargs.update(overrides)
    return cvpr.runner.run_source_discovery_stage(
        workspace=cvpr.workspace, run_id=RUN_ID, **cvpr.paths(), resume=resume, **kwargs
    )


def test_cvpr_source_discovery_success_receipt(cvpr: CvprHarness) -> None:
    cvpr.runner.prepare_preflight_estimate(cvpr.workspace, RUN_ID)
    cvpr.runner.prepare_preflight_storage_estimate(cvpr.workspace, RUN_ID)
    calls: list[str] = []
    outcome = run_source_discovery(cvpr, calls)

    assert outcome["status"] == "succeeded", outcome["receipt"].get("blockers")
    assert calls[0] == "collection" and calls.count("provider") == 1, calls
    assert_golden(CONFERENCE, "stage02_source_discovery__succeeded", cvpr.snap("source-discovery"))


def test_cvpr_source_discovery_collection_block_stops_before_authority(cvpr: CvprHarness) -> None:
    outcome = run_source_discovery(
        cvpr,
        [],
        collector=lambda **_kwargs: {"status": "blocked", "blockers": [{"code": "source_collection_blocked"}]},
        fresh_volatile_preflight=_must_not_run("source-discovery fresh preflight"),
        provider_adapter_factory=_must_not_run("source-discovery provider adapter"),
    )

    assert outcome["status"] == "blocked"
    assert_golden(
        CONFERENCE, "stage02_source_discovery__blocked_collection", cvpr.snap("source-discovery")
    )


def test_cvpr_source_discovery_resume_reuses_decision_ledger_without_provider_calls(cvpr: CvprHarness) -> None:
    cvpr.runner.prepare_preflight_estimate(cvpr.workspace, RUN_ID)
    cvpr.runner.prepare_preflight_storage_estimate(cvpr.workspace, RUN_ID)
    assert run_source_discovery(cvpr, [])["status"] == "succeeded"

    calls: list[str] = []
    outcome = run_source_discovery(
        cvpr,
        calls,
        resume=True,
        provider_adapter_factory=lambda **_kwargs: _PoisonedAdapter(),
    )

    # Source discovery always re-collects; resume must reuse the receipt-bound
    # decision ledger instead of re-paying for joins.
    assert "provider" not in calls
    assert outcome["status"] == "succeeded"
    assert outcome["receipt"]["resume"]["reused_decision_count"] == 1
    assert_golden(CONFERENCE, "stage02_source_discovery__resumed", cvpr.snap_receipt(outcome["receipt"]))


class _PoisonedAdapter:
    provider = "openai"
    model = "poisoned"

    def invoke(self, _candidate: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("resume re-paid a join the decision ledger already recorded")


# --------------------------------------------------------------------------
# Stage 3: corpus-freeze
# --------------------------------------------------------------------------
#
# main() dispatches run_corpus_freeze_stage(workspace, run_id, command_args)
# and relies on the defaults pinned_run_id=PINNED_CORPUS_SOURCE_RUN_ID /
# pinned_inventory_sha256=PINNED_CORPUS_SOURCE_INVENTORY_SHA256 (the live r10
# inventory). A synthetic fixture cannot reproduce that digest, so these tests
# pass the fixture's own run id and inventory digest through the same kwargs.


def write_corpus_freeze_fixture(workspace: Path, *, run_id: str = RUN_ID) -> dict[str, Any]:
    """A compact r3-shaped inventory and its succeeded source receipt (copied from the runner tests)."""

    inventory_path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    records = [
        {
            "source_record_id": "cvpr2026:source:program-schedule:fixture-a",
            "family_id": "program-schedule",
            "snapshot_sha256": "a" * 64,
            "aliases": ["https://fixtures.example/program-a"],
        },
        {
            "source_record_id": "cvpr2026:source:virtual-paper-media-posture:fixture-b",
            "family_id": "virtual-paper-media-posture",
            "snapshot_sha256": "b" * 64,
            "aliases": ["https://fixtures.example/paper-b"],
        },
    ]
    inventory: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "stage_id": "source-discovery",
        "status": "prepared",
        "blockers": [],
        "source_families": {
            "program-schedule": {"status": "live_valid", "exclusion_id": None},
            "virtual-paper-media-posture": {"status": "locked_excluded", "exclusion_id": "paywalled_official_video"},
        },
        "source_records": records,
        "program_items": [
            {
                "program_item_id": "cvpr2026:program-item:fixture-a",
                "source_record_ids": [records[0]["source_record_id"]],
                "aliases": ["fixture program a"],
                "field_evidence": {"title": {"source_record_id": records[0]["source_record_id"]}},
            },
            {
                "program_item_id": "cvpr2026:program-item:fixture-b",
                "source_record_ids": [records[1]["source_record_id"]],
                "aliases": ["fixture program b"],
                "field_evidence": {"title": {"source_record_id": records[1]["source_record_id"]}},
            },
        ],
    }
    inventory["inventory_sha256"] = source_discovery_canonical_json_digest(inventory)
    write_json(inventory_path, inventory)
    receipt_path = write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-02-source-discovery.json",
        {
            "run_id": run_id,
            "stage_id": "source-discovery",
            "stage_order": 2,
            "status": "succeeded",
            "mode": "live",
            "blockers": [],
            "canonical_inventory": {"path": str(inventory_path), "inventory_sha256": inventory["inventory_sha256"]},
            "stage_contract": {"status": "pass"},
        },
    )
    stamp_upstream_stage_receipts(receipt_path.parent)
    return inventory


def run_corpus_freeze(cvpr: CvprHarness, inventory: Mapping[str, Any], *, run_id: str = RUN_ID) -> dict[str, Any]:
    return cvpr.runner.run_corpus_freeze_stage(
        workspace=cvpr.workspace,
        run_id=run_id,
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
        command_args=("--stage", "corpus-freeze"),
    )


def test_cvpr_corpus_freeze_success_receipt(cvpr: CvprHarness) -> None:
    inventory = write_corpus_freeze_fixture(cvpr.workspace)
    outcome = run_corpus_freeze(cvpr, inventory)

    assert outcome["status"] == "succeeded", outcome["receipt"].get("blockers")
    assert outcome["receipt"]["provider_calls_made"] == 0
    assert outcome["receipt"]["downloads_made"] == 0
    assert_golden(
        CONFERENCE,
        "stage03_corpus_freeze__succeeded",
        cvpr.snap("corpus-freeze", extra_output_paths=["storage/corpus-freeze"]),
    )


def test_cvpr_corpus_freeze_changed_run_blocks_before_reading_inputs(cvpr: CvprHarness) -> None:
    inventory = write_corpus_freeze_fixture(cvpr.workspace)
    outcome = run_corpus_freeze(cvpr, inventory, run_id="w1-cvpr-wrong-run")

    assert outcome["status"] == "blocked"
    assert_golden(
        CONFERENCE,
        "stage03_corpus_freeze__blocked_pinned_run_mismatch",
        cvpr.snap("corpus-freeze", run_id="w1-cvpr-wrong-run"),
    )


# --------------------------------------------------------------------------
# Stage 4: download-corpus (public r3 lane)
# --------------------------------------------------------------------------


class _PublicPage:
    status = 200
    headers = {"Content-Type": "text/html"}

    def __init__(self, url: str) -> None:
        self.url = url
        self.sent = False

    def read(self, _size: int) -> bytes:
        if self.sent:
            return b""
        self.sent = True
        return b"<main>fixture source page</main>"

    def close(self) -> None:
        return None


def fake_public_opener(calls: list[str]):
    return lambda: (lambda url: calls.append(url) or _PublicPage(url))


def run_download_corpus(cvpr: CvprHarness, inventory: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """main()'s download-corpus kwargs, with pinned lineage and every lane seam faked."""

    kwargs: dict[str, Any] = {
        "execute_public_lane": True,
        "public_worker_count": 1,
        "execute_talk_lane": False,
        "execute_tavily_lane": False,
        "tavily_api_key": None,
        "tavily_http_post": _must_not_run("tavily http"),
        "tavily_fresh_preflight": _must_not_run("tavily fresh preflight"),
        "workshop_owner": _must_not_run("workshop owner"),
        "talk_harvester": _must_not_run("youtube talk harvester"),
        "yt_dlp_audit": _must_not_run("yt-dlp audit"),
        "available_bytes": 1_000_000,
        "expected_program_item_count": 2,
        "command_args": ("--stage", "download-corpus"),
    }
    kwargs.update(overrides)
    return cvpr.runner.run_download_corpus_stage(
        workspace=cvpr.workspace,
        run_id=RUN_ID,
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
        **kwargs,
    )


def _frozen_corpus(cvpr: CvprHarness) -> dict[str, Any]:
    inventory = write_corpus_freeze_fixture(cvpr.workspace)
    assert run_corpus_freeze(cvpr, inventory)["status"] == "succeeded"
    return inventory


DOWNLOAD_OUTPUTS = ["storage/download-corpus"]


def test_cvpr_download_corpus_public_lane_receipt(cvpr: CvprHarness) -> None:
    inventory = _frozen_corpus(cvpr)
    calls: list[str] = []
    outcome = run_download_corpus(cvpr, inventory, opener_factory=fake_public_opener(calls))

    assert calls
    # Talk/recovery lanes are not run in this scenario, so the stage stays blocked.
    assert outcome["status"] == "blocked"
    assert_golden(
        CONFERENCE,
        "stage04_download_corpus__public_lane_only",
        cvpr.snap("download-corpus", extra_output_paths=DOWNLOAD_OUTPUTS),
    )


def test_cvpr_download_corpus_resume_reuses_public_artifacts_without_reopening(cvpr: CvprHarness) -> None:
    inventory = _frozen_corpus(cvpr)
    run_download_corpus(cvpr, inventory, opener_factory=fake_public_opener([]))

    calls: list[str] = []
    run_download_corpus(cvpr, inventory, opener_factory=fake_public_opener(calls))

    assert calls == []
    ledger = read_json(cvpr.workspace / "storage" / "download-corpus" / "download_corpus.json")
    assert ledger["resume"]["status"] == "reused"
    assert_golden(
        CONFERENCE,
        "stage04_download_corpus__public_lane_resumed",
        cvpr.snap("download-corpus", extra_output_paths=DOWNLOAD_OUTPUTS),
    )


def test_cvpr_download_corpus_unknown_capacity_blocks_before_opener(cvpr: CvprHarness) -> None:
    inventory = _frozen_corpus(cvpr)
    outcome = run_download_corpus(cvpr, inventory, available_bytes=None, opener_factory=_must_not_run("opener factory"))

    assert outcome["status"] == "blocked"
    assert_golden(CONFERENCE, "stage04_download_corpus__blocked_capacity_unknown", cvpr.snap("download-corpus"))


def fake_talk_harvester(workspace: Path, calls: list[Any]):
    inventory_path = workspace / "storage" / "youtube-talks" / "harvest_inventory.json"

    def harvest(config: Any) -> dict[str, Any]:
        calls.append([item.item_id for item in config.program_spine])
        write_json(inventory_path, {"talks": [], "gaps": []})
        return {
            "status": "succeeded",
            "allowed_hosts": ["youtube.com", "youtu.be"],
            "boundaries": {"audio_downloaded": False, "video_downloaded": False},
            "pilot_limit": None,
            "tavily_lane": {"enabled": False},
            "inventory_path": str(inventory_path.relative_to(workspace)),
            "receipt_path": "storage/youtube-talks/harvest_receipt.json",
        }

    return harvest


def test_cvpr_download_corpus_all_lanes_receipt(cvpr: CvprHarness) -> None:
    inventory = _frozen_corpus(cvpr)
    harvest_calls: list[Any] = []
    outcome = run_download_corpus(
        cvpr,
        inventory,
        opener_factory=fake_public_opener([]),
        execute_talk_lane=True,
        execute_tavily_lane=True,
        talk_harvester=fake_talk_harvester(cvpr.workspace, harvest_calls),
        yt_dlp_audit=lambda: {"status": "pass", "blockers": []},
    )
    # Every lane completes on the fixture; the only blocker is the CVPR
    # download-corpus runtime contract's pinned checks.program_item_count
    # (the live 14,583-item census), which no synthetic fixture can satisfy.
    assert harvest_calls == [[]]
    assert outcome["status"] == "blocked"
    assert [b["code"] for b in outcome["receipt"]["blockers"]] == ["final_stage_contract_incomplete"]
    assert [b["code"] for b in outcome["receipt"]["stage_contract"]["blockers"]] == [
        "stage_contract_required_check_mismatch"
    ]
    assert_golden(
        CONFERENCE,
        "stage04_download_corpus__all_lanes_blocked_by_pinned_item_count",
        cvpr.snap("download-corpus", extra_output_paths=DOWNLOAD_OUTPUTS),
    )


# --------------------------------------------------------------------------
# Stage 4 entry point: --registration-media (r4)
# --------------------------------------------------------------------------
#
# The r4 census (2,579 official event requests / 5,159 program items) is a
# pinned constant of cvpr_registration_media, so its in-memory inputs are built
# by the runner tests' builders (imported, not copied: they are large and
# census-shaped). They are injected through the owner's own input_loader seam.


def _registration_inputs():
    from tests.test_cvpr2026_production_runner import registration_media_fixture_inputs

    return registration_media_fixture_inputs()


def _authenticated_page(calls: list[str]):
    def opener(url: str) -> dict[str, Any]:
        calls.append(url)
        return {
            "status": "authenticated",
            "auth_status": "authenticated",
            "http_status": 200,
            "final_url": url,
            "content_type": "text/html",
            "body": b"<main></main>",
        }

    return opener


PREFLIGHT_PASSED = {"preflight_status": "pass", "auth_status": "preflight_passed", "secrets_copied": False}


def run_registration_media(cvpr: CvprHarness, **overrides: Any) -> dict[str, Any]:
    predecessors, plan, overlay = _registration_inputs()
    kwargs: dict[str, Any] = {
        "command_args": ("--stage", "download-corpus", "--registration-media"),
        "resume": False,
        "input_loader": lambda _workspace, **_kwargs: (predecessors, plan, overlay),
        "available_bytes": 1_000_000,
    }
    kwargs.update(overrides)
    return cvpr.runner.run_cvpr_registration_media_stage(
        workspace=cvpr.workspace, run_id=cvpr.runner.R4_RUN_ID, **kwargs
    )


def _completed_zero_link(**kwargs: Any) -> dict[str, Any]:
    from tests.test_cvpr2026_production_runner import completed_zero_link_overlay

    return completed_zero_link_overlay(kwargs["overlay"])


REGISTRATION_OUTPUTS = ["storage/download-corpus/registration-media"]


def test_cvpr_registration_media_preflight_block(cvpr: CvprHarness) -> None:
    outcome = run_registration_media(
        cvpr,
        opener_factory=lambda **_kwargs: (
            None,
            {"preflight_status": "blocked", "auth_status": "registration_auth_required", "secrets_copied": False},
        ),
        recoverer=_must_not_run("registration-media recoverer"),
    )

    assert outcome["status"] == "blocked"
    assert_golden(
        CONFERENCE,
        "stage04r4_registration_media__blocked_preflight",
        cvpr.snap_receipt(outcome["receipt"]),
    )


def test_cvpr_registration_media_complete_zero_link_receipt(cvpr: CvprHarness) -> None:
    opened: list[str] = []
    outcome = run_registration_media(
        cvpr,
        opener_factory=lambda **_kwargs: (_authenticated_page(opened), PREFLIGHT_PASSED),
        recoverer=_completed_zero_link,
    )

    assert len(opened) == 1  # the sentinel; the recoverer fake completes the rest
    # Recovery completes; the strict registration-media contract then rejects
    # the synthetic workspace because it lacks the exact pinned r3 predecessor
    # files/publisher digests (live-lineage bound, unreachable on a fixture).
    assert outcome["status"] == "blocked"
    assert [b["code"] for b in outcome["receipt"]["stage_contract"]["blockers"]] == [
        "stage_contract_registration_media_predecessor_invalid"
    ]
    assert_golden(
        CONFERENCE,
        "stage04r4_registration_media__complete_blocked_by_pinned_r3_lineage",
        cvpr.snap_receipt(outcome["receipt"], extra_output_paths=REGISTRATION_OUTPUTS),
    )


# --------------------------------------------------------------------------
# Stage 4 entry point: --event-media-coverage (r5)
# --------------------------------------------------------------------------


def _event_inventory() -> dict[str, Any]:
    from tests.test_cvpr2026_production_runner import event_media_fixture_inventory

    return event_media_fixture_inventory()


def run_event_media(cvpr: CvprHarness, **overrides: Any) -> dict[str, Any]:
    baseline = _event_inventory()
    cvpr.monkeypatch.setattr(
        cvpr.runner, "_event_media_inventory_inputs", lambda _workspace, **_kwargs: json.loads(json.dumps(baseline))
    )
    kwargs: dict[str, Any] = {"resume": False, "caption_downloader": _must_not_run("caption downloader")}
    kwargs.update(overrides)
    return cvpr.runner.run_cvpr_event_media_stage(
        workspace=cvpr.workspace, run_id=cvpr.runner.R5_EVENT_MEDIA_RUN_ID, **kwargs
    )


EVENT_OUTPUTS = ["storage/download-corpus/event-media"]


def test_cvpr_event_media_preflight_block(cvpr: CvprHarness) -> None:
    outcome = run_event_media(
        cvpr,
        opener_factory=lambda **_kwargs: (
            None,
            {"preflight_status": "blocked", "auth_status": "registration_auth_required", "secrets_copied": False},
        ),
    )

    assert outcome["status"] == "blocked"
    assert_golden(CONFERENCE, "stage04r5_event_media__blocked_preflight", cvpr.snap_receipt(outcome["receipt"]))


def test_cvpr_event_media_full_scan_no_links_receipt(cvpr: CvprHarness) -> None:
    calls: list[str] = []

    def opener_factory(**_kwargs: Any):
        def opener(url: str) -> dict[str, Any]:
            calls.append(url)
            return {
                "status": "authenticated",
                "auth_status": "authenticated",
                "http_status": 200,
                "content_type": "text/html",
                "body": b"<main>No official YouTube link</main>",
            }

        return opener, PREFLIGHT_PASSED

    outcome = run_event_media(cvpr, opener_factory=opener_factory, caption_downloader=_must_not_run("captions"))

    assert len(calls) == 2_693
    # The scan completes; the strict event-media contract then requires the
    # immutable r4 registration-media overlay lineage (live-bound).
    assert outcome["status"] == "blocked"
    assert [b["code"] for b in outcome["receipt"]["stage_contract"]["blockers"]] == [
        "stage_contract_event_media_r4_lineage_invalid"
    ]
    assert_golden(
        CONFERENCE,
        "stage04r5_event_media__full_scan_blocked_by_pinned_r4_lineage",
        cvpr.snap_receipt(outcome["receipt"], extra_output_paths=EVENT_OUTPUTS),
    )


# --------------------------------------------------------------------------
# Stage 5: transcription-ocr (r5), incl. the --authorize-r5-transcription-retry entry
# --------------------------------------------------------------------------
#
# Stage 5 is bound to the fixed r5 run id and to pinned r5 Stage-4 digests.
# The runner tests' handoff builders (imported: they are large and r5-census
# shaped) write a synthetic published r5 handoff and rebind the pinned digest
# constants -- on the runner, the adapter module and stage_contracts -- plus a
# temp copy of the stage-owner registry whose strict transcription-ocr
# runtime contract equals those synthetic digests. The adapter (local caption
# materialization) is replaced by the tests' deterministic adapter result.


def prepare_stage5_retry(cvpr: CvprHarness) -> None:
    import conference_pipeline.stage_script_registry as stage_script_registry
    from tests.test_cvpr2026_production_runner import (
        _patch_real_stage5_contract_constants,
        _r5_stage5_adapter_result,
        _write_runner_historical_stage5_block,
        _write_runner_published_r5_handoff,
        _write_runner_strict_registry,
    )

    runner, workspace, monkeypatch = cvpr.runner, cvpr.workspace, cvpr.monkeypatch
    stage4_path = _write_runner_published_r5_handoff(runner, workspace, monkeypatch)
    _write_runner_historical_stage5_block(runner, workspace, stage4_path, monkeypatch)
    _patch_real_stage5_contract_constants(runner, monkeypatch)
    monkeypatch.setattr(
        stage_script_registry, "DEFAULT_STAGE_SCRIPT_REGISTRY_PATH", _write_runner_strict_registry(runner, workspace)
    )
    cvpr.stage5_adapter_calls = []

    def adapter(**_kwargs: Any) -> dict[str, Any]:
        cvpr.stage5_adapter_calls.append("adapter")
        return _r5_stage5_adapter_result(runner, workspace)

    monkeypatch.setattr(runner, "build_cvpr_r5_transcription_ocr_inventory", adapter)
    for forbidden_owner in ("run_cvpr_event_media_stage", "run_source_discovery_stage", "run_download_corpus_stage"):
        monkeypatch.setattr(runner, forbidden_owner, _must_not_run(forbidden_owner))


R5 = "cvpr2026-live-production-r5-event-media"
STAGE5_INVENTORY = "storage/transcription-ocr/cvpr2026/inventory.json"
STAGE5_OUTPUTS = [STAGE5_INVENTORY]


def test_cvpr_authorize_r5_transcription_retry_payload(cvpr: CvprHarness) -> None:
    prepare_stage5_retry(cvpr)
    payload = cvpr.runner.authorize_cvpr_r5_transcription_retry(workspace=cvpr.workspace, run_id=R5)

    assert cvpr.stage5_adapter_calls == []
    assert_golden(CONFERENCE, "stage05_authorize_r5_retry__payload", cvpr.snap_payload(payload))


def test_cvpr_transcription_ocr_retry_success_receipt(cvpr: CvprHarness) -> None:
    prepare_stage5_retry(cvpr)
    authorization = cvpr.runner.authorize_cvpr_r5_transcription_retry(workspace=cvpr.workspace, run_id=R5)
    outcome = cvpr.runner.run_cvpr_transcription_ocr_stage(
        workspace=cvpr.workspace,
        run_id=R5,
        resume=False,
        retry_authorization=Path(authorization["authorization_path"]),
    )

    assert outcome["status"] == "succeeded", outcome["receipt"]["stage_contract"].get("blockers")
    assert cvpr.stage5_adapter_calls == ["adapter"]
    assert_golden(
        CONFERENCE,
        "stage05_transcription_ocr__retry_succeeded",
        cvpr.snap("transcription-ocr", run_id=R5, extra_output_paths=STAGE5_OUTPUTS),
    )


def test_cvpr_transcription_ocr_resume_reuses_exact_result(cvpr: CvprHarness) -> None:
    prepare_stage5_retry(cvpr)
    authorization = cvpr.runner.authorize_cvpr_r5_transcription_retry(workspace=cvpr.workspace, run_id=R5)
    first = cvpr.runner.run_cvpr_transcription_ocr_stage(
        workspace=cvpr.workspace, run_id=R5, retry_authorization=Path(authorization["authorization_path"])
    )
    assert first["status"] == "succeeded"
    before = cvpr.receipt_file("transcription-ocr", R5).read_bytes()
    inventory_path = cvpr.workspace / STAGE5_INVENTORY
    inventory_before = inventory_path.read_bytes()

    def reusing_adapter(**_kwargs: Any) -> dict[str, Any]:
        # Mirrors the real adapter's compatible-prior branch: the persisted
        # inventory is returned in memory with resume.reused_output_digest set
        # and is not rewritten.
        prior = json.loads(inventory_path.read_text(encoding="utf-8"))
        prior["resume"] = {"reused_output_digest": prior["inventory_sha256"], "superseded_attempt_count": 0}
        return prior

    cvpr.monkeypatch.setattr(cvpr.runner, "build_cvpr_r5_transcription_ocr_inventory", reusing_adapter)
    cvpr.monkeypatch.setattr(cvpr.runner, "_transcription_ocr_stage_receipt", _must_not_run("stage 5 receipt writer"))

    resumed = cvpr.runner.run_cvpr_transcription_ocr_stage(workspace=cvpr.workspace, run_id=R5, resume=True)

    assert resumed["status"] == "succeeded"
    assert resumed["summary"] == {"status": "succeeded", "reused_exact_result": True}
    assert cvpr.receipt_file("transcription-ocr", R5).read_bytes() == before
    assert inventory_path.read_bytes() == inventory_before


def test_cvpr_transcription_ocr_consumed_authorization_fails_closed(cvpr: CvprHarness) -> None:
    prepare_stage5_retry(cvpr)
    authorization = Path(
        cvpr.runner.authorize_cvpr_r5_transcription_retry(workspace=cvpr.workspace, run_id=R5)["authorization_path"]
    )
    cvpr.runner.run_cvpr_transcription_ocr_stage(workspace=cvpr.workspace, run_id=R5, retry_authorization=authorization)
    calls_before = list(cvpr.stage5_adapter_calls)

    with pytest.raises(ValueError, match="retry authorization already consumed"):
        cvpr.runner.run_cvpr_transcription_ocr_stage(workspace=cvpr.workspace, run_id=R5, retry_authorization=authorization)
    assert cvpr.stage5_adapter_calls == calls_before


# --------------------------------------------------------------------------
# Stage 6: corpus-inventory-index
# --------------------------------------------------------------------------
#
# Predecessors (Stage 3 freeze, Stage 4 ledger + receipt, placeholder Stage 5
# receipt, r5 caption inventory, workshop paper download receipt, two
# extractable PDFs) come from the runner tests' builder (imported). The
# embedding provider is the local deterministic ``fake`` one; the persisted
# Chroma collection (random ids) is excluded from the snapshot.

STAGE6_SKIP = ("chroma",)


def _deterministic_extractable_pdf(path: Path, text: str) -> bytes:
    """Same PDF as the runner tests' writer, minus the random file id."""

    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes(garbage=4, deflate=True, no_new_id=True)
    document.close()
    path.write_bytes(data)
    return data


def _stage6_builder(cvpr: CvprHarness):
    import tests.test_cvpr2026_production_runner as runner_tests

    # The builder resolves this module global at call time; a random PDF id
    # would make the paper digests (and every digest over them) volatile.
    cvpr.monkeypatch.setattr(runner_tests, "_write_extractable_pdf", _deterministic_extractable_pdf)
    # It also lowers the paper-curation text floor by plain assignment;
    # register the original with monkeypatch so it is restored after the test.
    cvpr.monkeypatch.setattr(cvpr.runner, "PAPER_CURATION_MIN_TEXT_BYTES", cvpr.runner.PAPER_CURATION_MIN_TEXT_BYTES)
    cvpr.monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    cvpr.monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    return runner_tests._write_corpus_inventory_index_predecessors


def prepare_stage6(cvpr: CvprHarness, run_id: str = RUN_ID) -> dict[str, Any]:
    return _stage6_builder(cvpr)(cvpr.runner, cvpr.workspace, run_id)


def run_stage6(cvpr: CvprHarness, *, resume: bool = False, run_id: str = RUN_ID) -> dict[str, Any]:
    return cvpr.runner.run_corpus_inventory_index_stage(
        workspace=cvpr.workspace, run_id=run_id, resume=resume, full_rebuild=False, verify_chunk_cache=True
    )


def test_cvpr_corpus_inventory_index_success_receipt(cvpr: CvprHarness) -> None:
    prepare_stage6(cvpr)
    outcome = run_stage6(cvpr)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    assert_golden(
        CONFERENCE,
        "stage06_corpus_inventory_index__succeeded",
        cvpr.snap("corpus-inventory-index", skip_dirs=STAGE6_SKIP),
    )


def test_cvpr_corpus_inventory_index_resume_is_incremental(cvpr: CvprHarness) -> None:
    prepare_stage6(cvpr)
    assert run_stage6(cvpr)["status"] == "succeeded"

    resumed = run_stage6(cvpr, resume=True)

    checks = resumed["receipt"]["checks"]
    assert resumed["status"] == "succeeded"
    assert checks["index_update"]["mode"] == "incremental"
    assert all(update["chunks_embedded_now"] == 0 for update in checks["embedding_update"].values())
    # The first run's chunk manifests embed text_file_mtime_ns/generated_at and
    # are overwritten by the resumed run, so their raw digests (recorded as
    # previous_chunk_manifest_sha256, and inside index_digest.json) can no
    # longer be tied to a file on disk and vary with wall-clock timing.
    snapshot = cvpr.snap(
        "corpus-inventory-index",
        skip_dirs=STAGE6_SKIP,
        mask=["checks.index_update.stores.*.previous_chunk_manifest_sha256"],
    )
    index_digest = f"storage/vector-store/{RUN_ID}/index_digest.json"
    assert snapshot["output_files"].get(index_digest)
    snapshot["output_files"][index_digest] = "<MASKED: embeds overwritten first-run manifest digests>"
    assert_golden(CONFERENCE, "stage06_corpus_inventory_index__resumed_incremental", snapshot)


def test_cvpr_corpus_inventory_index_missing_stage4_blocks(cvpr: CvprHarness) -> None:
    _stage6_builder(cvpr)(cvpr.runner, cvpr.workspace, RUN_ID, include_download_corpus=False)
    outcome = run_stage6(cvpr)

    assert outcome["status"] == "blocked"
    assert_golden(
        CONFERENCE,
        "stage06_corpus_inventory_index__blocked_missing_stage4",
        cvpr.snap("corpus-inventory-index", skip_dirs=STAGE6_SKIP),
    )


# --------------------------------------------------------------------------
# Stage 7 (14A): exhaustive-topic-extraction
# --------------------------------------------------------------------------


def write_stage6_receipt(workspace: Path, run_id: str = RUN_ID, *, status: str = "succeeded") -> Path:
    """Minimal stamped Stage 6 receipt (copied from the runner tests)."""

    receipt_path = write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-06-corpus-inventory-index.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "stage_id": "corpus-inventory-index",
            "stage_order": 6,
            "mode": "live",
            "status": status,
            "blockers": [],
        },
    )
    stamp_upstream_stage_receipts(receipt_path.parent)
    return receipt_path


def fake_exhaustive_owner(calls: list[dict[str, Any]]):
    """Stands in for the shared (paid, OpenRouter) exhaustive extraction owner."""

    def owner(workspace_root: Path, *, run_id: str, force_rebuild: bool = False) -> dict[str, Any]:
        calls.append({"run_id": run_id, "force_rebuild": force_rebuild})
        root = Path(workspace_root) / "storage" / "extractions" / run_id
        write_json(root / "by-topic" / "01_fixture_topic.json", {"topic_id": "01_fixture_topic", "artifacts": []})
        write_json(root / "by-artifact" / "fixture-artifact.json", {"artifact_id": "fixture-artifact", "topics": []})
        write_json(root / "v15_extraction_coverage.json", {"coverage_ratio": 1.0})
        return {
            "schema_version": 1,
            "run_id": run_id,
            "stage_id": "exhaustive-topic-extraction",
            "status": "succeeded",
            "blockers": [],
        }

    return owner


def run_stage7a(cvpr: CvprHarness, *, resume: bool = False) -> dict[str, Any]:
    return cvpr.runner.run_exhaustive_topic_extraction_stage(
        workspace=cvpr.workspace, run_id=RUN_ID, resume=resume, env_file=None
    )


EXTRACTION_OUTPUTS = [f"storage/extractions/{RUN_ID}"]


def test_cvpr_exhaustive_topic_extraction_success_receipt(cvpr: CvprHarness) -> None:
    write_stage6_receipt(cvpr.workspace)
    calls: list[dict[str, Any]] = []
    cvpr.monkeypatch.setattr(cvpr.runner, "run_exhaustive_topic_extraction", fake_exhaustive_owner(calls))

    outcome = run_stage7a(cvpr)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    assert calls == [{"run_id": RUN_ID, "force_rebuild": True}]
    assert_golden(
        CONFERENCE,
        "stage07a_exhaustive_topic_extraction__succeeded",
        cvpr.snap("exhaustive-topic-extraction", extra_output_paths=EXTRACTION_OUTPUTS),
    )


def test_cvpr_exhaustive_topic_extraction_missing_stage6_blocks(cvpr: CvprHarness) -> None:
    cvpr.monkeypatch.setattr(cvpr.runner, "run_exhaustive_topic_extraction", _must_not_run("exhaustive owner"))

    outcome = run_stage7a(cvpr)

    assert outcome["status"] == "blocked"
    assert_golden(
        CONFERENCE,
        "stage07a_exhaustive_topic_extraction__blocked_stage6_required",
        cvpr.snap("exhaustive-topic-extraction"),
    )


def test_cvpr_exhaustive_topic_extraction_resume_accepts_current_receipt(cvpr: CvprHarness) -> None:
    write_stage6_receipt(cvpr.workspace)
    cvpr.monkeypatch.setattr(cvpr.runner, "run_exhaustive_topic_extraction", fake_exhaustive_owner([]))
    assert run_stage7a(cvpr)["status"] == "succeeded"
    before = cvpr.receipt_file("exhaustive-topic-extraction").read_bytes()
    cvpr.monkeypatch.setattr(cvpr.runner, "run_exhaustive_topic_extraction", _must_not_run("exhaustive owner"))

    resumed = run_stage7a(cvpr, resume=True)

    assert resumed["status"] == "succeeded"
    assert cvpr.receipt_file("exhaustive-topic-extraction").read_bytes() == before


# --------------------------------------------------------------------------
# Stage 7: analysis
# --------------------------------------------------------------------------
#
# The predecessor families run_analysis gates on (Stage 6 chunk sidecars,
# 14A heuristic + full-read extraction, topic strategy, 07B/07C/07D/07E/07H
# receipts, reconciliation outputs) are copied, condensed, from the runner
# tests' test_analysis_fixture_receipt_satisfies_runtime_contract (inline
# there, so not importable). The deep-narrative client and the narrative
# renderer are fakes: zero provider calls.

ANALYSIS_TOPIC_IDS = [f"{index:02d}_topic" for index in range(1, 13)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def prepare_analysis(cvpr: CvprHarness, run_id: str = RUN_ID) -> None:
    runner, workspace = cvpr.runner, cvpr.workspace
    target = runner.run_dir(workspace, run_id)
    target.mkdir(parents=True, exist_ok=True)
    final_topics = [
        {"topic_id": topic_id, "title": f"CVPR Fixture Topic {index}"}
        for index, topic_id in enumerate(ANALYSIS_TOPIC_IDS, start=1)
    ]

    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    artifact_store = vector_root / "stores" / "artifacts"
    _write_jsonl(
        paper_store / "chunks.jsonl",
        [
            {
                "chunk_id": f"paper-chunk-{index}",
                "artifact_id": f"paper_text:fixture-{index}",
                "source_path": f"https://cvpr.thecvf.com/fixture-paper-{index}",
                "text": f"Deterministic CVPR fixture topic evidence sentence {index} about the conference corpus.",
            }
            for index in range(4)
        ],
    )
    _write_jsonl(
        artifact_store / "chunks.jsonl",
        [
            {
                "chunk_id": f"artifact-chunk-{index}",
                "artifact_id": f"speech_transcript:fixture-{index}",
                "source_path": f"https://cvpr.thecvf.com/fixture-artifact-{index}",
                "text": f"Deterministic CVPR fixture topic transcript sentence {index} about the conference corpus.",
            }
            for index in range(4)
        ],
    )
    relpath = runner.platform_corpus_vector_store.workspace_relpath
    index_digest_path = vector_root / "index_digest.json"
    index_digest_path.write_text(
        json.dumps(
            {
                "stores": {
                    "papers": {"chunks_path": relpath(workspace, paper_store / "chunks.jsonl")},
                    "artifacts": {"chunks_path": relpath(workspace, artifact_store / "chunks.jsonl")},
                }
            }
        ),
        encoding="utf-8",
    )
    chunk_counts = {"papers": 4, "artifacts": 4}
    write_json(
        target / runner.STAGES["corpus-inventory-index"].receipt_filename,
        {
            "status": "succeeded",
            "checks": {
                "vector_embeddings_written": True,
                "chunk_counts": chunk_counts,
                "embedded_chunk_counts": chunk_counts,
                "index_digest_sha256": runner._file_sha256(index_digest_path),
                "output_inventory": {"sha256": "a" * 64},
            },
        },
    )

    extractions = workspace / "storage" / "extractions" / run_id
    write_json(target / "14A-exhaustive-topic-extraction-receipt.json", {"status": "succeeded"})
    write_json(extractions / "by-topic" / "01_topic.json", {"topic_id": "01_topic"})
    write_json(extractions / "by-artifact" / "artifact-1.json", {"artifact_id": "artifact-1"})
    for index, topic_id in enumerate(ANALYSIS_TOPIC_IDS, start=1):
        write_json(
            extractions / "llm-full-read" / "by-topic" / f"{topic_id}.json",
            {
                "unit_count": 1,
                "items": [
                    {
                        "unit_id": f"unit-{index}",
                        "input_kind": "paper_abstract",
                        "confidence": 0.9,
                        "important": True,
                        "summary": f"Fixture full-read summary for {topic_id}.",
                        "reason": "fixture reason",
                        "evidence_phrases": ["fixture evidence phrase"],
                        "source_artifact_path": f"storage/corpus/paper-text/fixture-{index}.json",
                        "metadata": {"title": f"Fixture Paper {index}", "paper_id": f"fixture-{index}"},
                    }
                ],
            },
        )
        write_json(
            extractions / "llm-full-read" / "by-unit" / f"unit-{index}.json",
            {"unit_id": f"unit-{index}", "topic_id": topic_id},
        )
    write_json(
        target / "14A-llm-full-read-extraction-receipt.json",
        {
            "status": "succeeded",
            "scope": "all",
            "limit": None,
            "checks": {
                "completion_state": "complete",
                "expected_unit_count": 12,
                "completed_unit_count": 12,
                "failed_count": 0,
            },
        },
    )

    write_json(workspace / "analysis" / "topic_strategy.json", {"final_topics": final_topics})
    write_json(
        workspace / "storage" / "analysis" / "topic-strategy" / run_id / "topic_strategy_receipt.json",
        {
            "status": "succeeded",
            "mode": "live",
            "provider_receipt_paths": ["storage/provider-calls/topic-strategy/fixture.json"],
            "final_topics": final_topics,
        },
    )
    write_json(target / "07D-topic-reconciliation.json", {"status": "succeeded", "source_packs_enriched": True})
    write_json(
        target / "07E-topic-reference-validation.json",
        {
            "status": "succeeded",
            "topic_count": 12,
            "checks": {
                "completed_topic_count": 12,
                "ledger_entry_count": 12,
                "citation_count": 12,
                "invalid_reference_count": 0,
                "unknown_reference_count": 0,
                "title_mismatch_count": 0,
                "bare_reference_warning_count": 0,
            },
        },
    )
    write_json(
        target / "07H-topic-reconciliation-reference-workflow.json",
        {
            "status": "succeeded",
            "checks": {
                "final_validation_status": "succeeded",
                "final_topic_count": 12,
                "final_completed_topic_count": 12,
                "final_ledger_entry_count": 12,
                "final_citation_count": 12,
                "final_invalid_reference_count": 0,
                "final_unknown_reference_count": 0,
                "final_title_mismatch_count": 0,
                "final_bare_reference_warning_count": 0,
            },
        },
    )
    write_json(target / "07B-deep-dive-question-selection.json", {"status": "succeeded", "provider_calls_made": True})
    write_json(target / "07C-dual-deep-dive-research-lanes.json", {"status": "succeeded", "provider_calls_made": True})
    for topic_id in ANALYSIS_TOPIC_IDS:
        reconciliation = workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md"
        reconciliation.parent.mkdir(parents=True, exist_ok=True)
        reconciliation.write_text(
            f"# Fixture Reconciliation {topic_id}\n\nDeterministic fixture reconciliation body.\n", encoding="utf-8"
        )
        write_json(
            workspace / "analysis" / "topic_reference_ledgers" / f"{topic_id}.json",
            {"topic_id": topic_id, "entries": ["fixture-entry-1"]},
        )
        write_json(
            workspace / "analysis" / "topic_reference_validation" / f"{topic_id}.json",
            {"topic_id": topic_id, "status": "succeeded"},
        )
    stamp_upstream_stage_receipts(target)


class _FixtureDeepNarrativeClient:
    model: str = ""
    request_count = 0

    @classmethod
    def from_env(cls) -> _FixtureDeepNarrativeClient:
        return cls()


def fake_narrative_renderer(calls: list[str]):
    from types import SimpleNamespace

    def render(
        workspace_arg: Path,
        *,
        run_id: str,
        llm_client: Any,
        allow_deterministic_fallback: bool,
        narrative_model: str,
        analysis_profile: str,
        reuse_existing_pass_reports: bool,
    ) -> Any:
        assert isinstance(llm_client, _FixtureDeepNarrativeClient)
        calls.append(run_id)
        analysis_dir = Path(workspace_arg) / "analysis"
        paths: dict[str, list[Path]] = {"report": [], "brief": [], "prov": [], "prov_json": []}
        summaries = []
        for topic_id in ANALYSIS_TOPIC_IDS:
            report = analysis_dir / f"{topic_id}.md"
            brief = analysis_dir / "slide_briefs" / f"{topic_id}.md"
            prov = analysis_dir / "provenance" / f"{topic_id}.md"
            prov_json = analysis_dir / "provenance" / f"{topic_id}.json"
            for path, body in (
                (report, f"# Fixture Deep Narrative {topic_id}\n\nDeterministic fixture narrative body for {topic_id}.\n"),
                (brief, f"# Fixture Slide Brief {topic_id}\n\nDeterministic fixture slide brief.\n"),
                (prov, f"# Fixture Provenance {topic_id}\n\nDeterministic fixture provenance body.\n"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")
            write_json(prov_json, {"topic_id": topic_id, "citations": ["fixture-citation-1"]})
            for key, path in (("report", report), ("brief", brief), ("prov", prov), ("prov_json", prov_json)):
                paths[key].append(path)
            summaries.append(
                {
                    "topic_id": topic_id,
                    "extraction_record_ids": [f"fixture-extraction-{topic_id}"],
                    "vector_store_chunk_ids": [f"fixture-chunk-{topic_id}"],
                    "research_ids": [f"fixture-research-{topic_id}"],
                    "provenance_path": f"analysis/provenance/{topic_id}.md",
                    "provenance_json_path": f"analysis/provenance/{topic_id}.json",
                    "slide_brief_path": f"analysis/slide_briefs/{topic_id}.md",
                    "analysis_profile": analysis_profile,
                    "narrative_model": narrative_model,
                    "deep_validation": {"status": "not_applicable_fixture"},
                }
            )
        citation_manifest_path = write_json(analysis_dir / "analysis_citation_manifest.json", {"topics": summaries})
        return SimpleNamespace(
            status="succeeded",
            report_paths=tuple(paths["report"]),
            slide_brief_paths=tuple(paths["brief"]),
            provenance_paths=tuple(paths["prov"]),
            provenance_json_paths=tuple(paths["prov_json"]),
            citation_manifest_path=citation_manifest_path,
            citation_summaries=tuple(summaries),
            blockers=(),
        )

    return render


def install_analysis_fakes(cvpr: CvprHarness, calls: list[str]) -> None:
    client = type("_Client", (_FixtureDeepNarrativeClient,), {"model": cvpr.runner.DEFAULT_DEEP_NARRATIVE_MODEL})
    cvpr.monkeypatch.setattr(cvpr.runner, "OpenAIDeepNarrativeClient", client)
    cvpr.monkeypatch.setattr(cvpr.runner, "render_narrative_outputs_from_source_packs", fake_narrative_renderer(calls))


def run_stage7(cvpr: CvprHarness, *, resume: bool = False) -> dict[str, Any]:
    return cvpr.runner.run_analysis(
        workspace=cvpr.workspace,
        run_id=RUN_ID,
        resume=resume,
        provider_cost_approved=True,
        env_file=None,
        topic_scope=None,
    )


ANALYSIS_OUTPUTS = ["analysis/analysis_manifest.json", "analysis/analysis_citation_manifest.json"]


def test_cvpr_analysis_success_receipt(cvpr: CvprHarness) -> None:
    prepare_analysis(cvpr)
    calls: list[str] = []
    install_analysis_fakes(cvpr, calls)

    outcome = run_stage7(cvpr)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    assert calls == [RUN_ID]
    assert_golden(
        CONFERENCE,
        "stage07_analysis__succeeded",
        cvpr.snap("analysis", extra_output_paths=ANALYSIS_OUTPUTS),
    )


def test_cvpr_analysis_without_cost_approval_blocks(cvpr: CvprHarness) -> None:
    """W5 Phase B: without approval the stage blocks before the renderer is
    touched (it used to call the renderer with no client, which overwrites
    every topic's report with a blocked placeholder)."""

    prepare_analysis(cvpr)
    cvpr.monkeypatch.setattr(cvpr.runner, "OpenAIDeepNarrativeClient", _must_not_run_class())
    cvpr.monkeypatch.setattr(cvpr.runner, "render_narrative_outputs_from_source_packs", _must_not_run("renderer"))

    outcome = cvpr.runner.run_analysis(
        workspace=cvpr.workspace, run_id=RUN_ID, resume=False, provider_cost_approved=False, env_file=None, topic_scope=None
    )

    assert outcome["status"] == "blocked"
    assert [item["code"] for item in outcome["receipt"]["blockers"]] == ["analysis_provider_cost_approval_required"]
    assert not list((cvpr.workspace / "analysis").glob("[0-9][0-9]_*.md"))
    assert_golden(CONFERENCE, "stage07_analysis__blocked_no_cost_approval", cvpr.snap("analysis"))


def test_cvpr_analysis_stale_upstream_blocks_before_client(cvpr: CvprHarness) -> None:
    prepare_analysis(cvpr)
    install_analysis_fakes(cvpr, [])
    assert run_stage7(cvpr, resume=True)["status"] == "succeeded"
    target = cvpr.run_dir()
    extraction_path = target / "14A-exhaustive-topic-extraction-receipt.json"
    stage6_key = f"storage/workflow-runs/{RUN_ID}/{cvpr.runner.STAGES['corpus-inventory-index'].receipt_filename}"
    extraction = read_json(extraction_path)
    extraction["lineage"] = {
        **extraction["lineage"],
        "upstream_receipts": {
            stage6_key: {"stage_id": "corpus-inventory-index", "digest_kind": "output_digest", "digest": "0" * 64}
        },
    }
    write_json(extraction_path, extraction)
    cvpr.monkeypatch.setattr(cvpr.runner, "OpenAIDeepNarrativeClient", _must_not_run_class())
    cvpr.monkeypatch.setattr(cvpr.runner, "render_narrative_outputs_from_source_packs", _must_not_run("renderer"))

    blocked = run_stage7(cvpr, resume=True)

    assert [item["code"] for item in blocked["receipt"]["blockers"]] == ["upstream_lineage_not_fresh"]
    assert_golden(CONFERENCE, "stage07_analysis__blocked_upstream_stale", cvpr.snap("analysis"))


def test_cvpr_analysis_resume_returns_the_verified_receipt_without_rendering(cvpr: CvprHarness) -> None:
    """W5 Phase B: --resume over a succeeded receipt that still verifies, with
    fresh upstreams, is a receipt short-circuit: no client, no renderer, no
    spend, and the receipt is left byte-identical (it used to re-enter the
    renderer with reuse_existing_pass_reports=False, paying again)."""

    prepare_analysis(cvpr)
    install_analysis_fakes(cvpr, [])
    assert run_stage7(cvpr)["status"] == "succeeded"
    before = cvpr.receipt_file("analysis").read_bytes()
    seen: list[dict[str, Any]] = []
    cvpr.monkeypatch.setattr(cvpr.runner, "OpenAIDeepNarrativeClient", _must_not_run_class())
    cvpr.monkeypatch.setattr(cvpr.runner, "render_narrative_outputs_from_source_packs", _must_not_run("renderer"))

    resumed = run_stage7(cvpr, resume=True)

    assert resumed["status"] == "succeeded", resumed["receipt"]["blockers"]
    assert cvpr.receipt_file("analysis").read_bytes() == before
    assert_golden(
        CONFERENCE,
        "stage07_analysis__resumed",
        {"renderer_calls": seen, **cvpr.snap("analysis", extra_output_paths=ANALYSIS_OUTPUTS)},
    )


def _must_not_run_class() -> type:
    class _MustNotConstruct:
        @classmethod
        def from_env(cls) -> Any:
            raise AssertionError("paid analysis client constructed on resume/stale upstream")

    return _MustNotConstruct


# --------------------------------------------------------------------------
# Stage 8: qa-gap-fill-synthesis
# --------------------------------------------------------------------------
#
# Predecessors and the non-network Stage8SynthesisClient stand-in are imported
# from the runner tests (large builders).


def prepare_stage8(cvpr: CvprHarness) -> dict[str, Any]:
    from tests.test_cvpr2026_production_runner import _write_qa_gap_fill_synthesis_predecessors

    return _write_qa_gap_fill_synthesis_predecessors(cvpr.runner, cvpr.workspace, RUN_ID, topic_count=3)


def install_stage8_client(cvpr: CvprHarness, instances: list[Any]) -> None:
    from tests.test_cvpr2026_production_runner import _build_fixture_stage8_synthesis_client_class

    cvpr.monkeypatch.setattr(
        cvpr.runner,
        "Stage8SynthesisClient",
        _build_fixture_stage8_synthesis_client_class(cvpr.runner, instances=instances),
    )


def run_stage8(cvpr: CvprHarness, *, resume: bool = False, provider_cost_approved: bool = True) -> dict[str, Any]:
    return cvpr.runner.run_qa_gap_fill_synthesis_stage(
        workspace=cvpr.workspace,
        run_id=RUN_ID,
        resume=resume,
        provider_cost_approved=provider_cost_approved,
        env_file=None,
    )


class _ExplodingClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("paid synthesis client constructed where it must not be")


def test_cvpr_qa_gap_fill_synthesis_success_receipt(cvpr: CvprHarness) -> None:
    prepare_stage8(cvpr)
    instances: list[Any] = []
    install_stage8_client(cvpr, instances)

    outcome = run_stage8(cvpr)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    assert len(instances) == 1
    assert_golden(
        CONFERENCE,
        "stage08_qa_gap_fill_synthesis__succeeded",
        {"synthesis_calls": list(instances[0].calls), **cvpr.snap("qa-gap-fill-synthesis")},
    )


def test_cvpr_qa_gap_fill_synthesis_missing_stage7_blocks(cvpr: CvprHarness) -> None:
    handles = prepare_stage8(cvpr)
    cvpr.monkeypatch.setattr(cvpr.runner, "Stage8SynthesisClient", _ExplodingClient)
    Path(handles["stage7_receipt_path"]).unlink()

    outcome = run_stage8(cvpr)

    assert outcome["status"] == "blocked"
    assert_golden(
        CONFERENCE, "stage08_qa_gap_fill_synthesis__blocked_missing_stage7", cvpr.snap("qa-gap-fill-synthesis")
    )


def test_cvpr_qa_gap_fill_synthesis_resume_accepts_current_receipt(cvpr: CvprHarness) -> None:
    prepare_stage8(cvpr)
    install_stage8_client(cvpr, [])
    assert run_stage8(cvpr)["status"] == "succeeded"
    before = cvpr.receipt_file("qa-gap-fill-synthesis").read_bytes()
    cvpr.monkeypatch.setattr(cvpr.runner, "Stage8SynthesisClient", _ExplodingClient)

    resumed = run_stage8(cvpr, resume=True)

    assert resumed["status"] == "succeeded"
    assert cvpr.receipt_file("qa-gap-fill-synthesis").read_bytes() == before


# --------------------------------------------------------------------------
# Stage 9: podcast
# --------------------------------------------------------------------------
#
# main() calls run_podcast_stage without command_runner (the real subprocess
# runner for scripts/generate_focused_podcast.py and render_podcast_audio.py);
# here the stage's own command_runner seam gets the runner tests' fake
# (imported), and provider_receipt_summary is replaced by a copied stand-in.


def fake_provider_receipt_summary(workspace: Path, run_id: str | None = None, stage_id: str | None = None) -> dict[str, Any]:
    del workspace, run_id, stage_id
    return {
        "canonical_receipt_files_scanned": 2,
        "canonical_total_estimated_cost_usd": 0.02,
        "canonical_totals_by_provider": {"openai": 0.02},
        "canonical_totals_by_model": {"openai:fixture-tts": 0.02},
        "canonical_totals_by_operation": {"tts": 0.02},
        "failed_receipts": 0,
        "unknown_cost_receipts": 0,
        "receipt_paths": ["provider-call-1.json", "provider-call-2.json"],
    }


def prepare_stage9(cvpr: CvprHarness) -> None:
    from tests.test_cvpr2026_production_runner import _write_podcast_predecessors

    _write_podcast_predecessors(cvpr.runner, cvpr.workspace, RUN_ID, cvpr.monkeypatch, topic_count=3)
    cvpr.monkeypatch.setattr(cvpr.runner, "Stage8SynthesisClient", _ExplodingClient)
    cvpr.monkeypatch.setattr(cvpr.runner, "provider_receipt_summary", fake_provider_receipt_summary)
    cvpr.monkeypatch.setattr(cvpr.runner, "run_podcast_command", _must_not_run("real podcast subprocess"))


def fake_podcast_commands(cvpr: CvprHarness, log: list[str]):
    from tests.test_cvpr2026_production_runner import _build_fake_podcast_command_runner

    return _build_fake_podcast_command_runner(cvpr.runner, call_log=log)


def run_stage9(cvpr: CvprHarness, command_runner: Any, *, resume: bool = False, approved: bool = True) -> dict[str, Any]:
    return cvpr.runner.run_podcast_stage(
        workspace=cvpr.workspace,
        run_id=RUN_ID,
        resume=resume,
        provider_cost_approved=approved,
        env_file=None,
        command_runner=command_runner,
    )


def test_cvpr_podcast_success_receipt(cvpr: CvprHarness) -> None:
    prepare_stage9(cvpr)
    log: list[str] = []

    outcome = run_stage9(cvpr, fake_podcast_commands(cvpr, log))

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    assert log == ["script", "audio"]
    assert_golden(CONFERENCE, "stage09_podcast__succeeded", cvpr.snap("podcast"))


def test_cvpr_podcast_without_cost_approval_blocks(cvpr: CvprHarness) -> None:
    prepare_stage9(cvpr)

    outcome = run_stage9(cvpr, _must_not_run("podcast command runner"), approved=False)

    assert outcome["status"] == "blocked"
    assert_golden(CONFERENCE, "stage09_podcast__blocked_no_cost_approval", cvpr.snap("podcast"))


def test_cvpr_podcast_resume_accepts_current_receipt(cvpr: CvprHarness) -> None:
    prepare_stage9(cvpr)
    assert run_stage9(cvpr, fake_podcast_commands(cvpr, []))["status"] == "succeeded"
    before = cvpr.receipt_file("podcast").read_bytes()

    resumed = run_stage9(cvpr, _must_not_run("podcast command runner"), resume=True)

    assert resumed["status"] == "succeeded"
    assert cvpr.receipt_file("podcast").read_bytes() == before
