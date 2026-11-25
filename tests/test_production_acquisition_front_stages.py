"""Shared Stage 1 (preflight), Stage 2 (source discovery) and Stage 3 (corpus freeze) modules.

These stages used to live in every conference runner. These tests pin what
the shared modules do for ANY runner with no hooks: a minimal new conference
(the ``minimal_new_conference`` fixture, plus a second synthetic slug) runs the
approval / estimate / capacity preflight, the probed source-intake discovery,
the governed source-policy receipts and both freeze formats under its own
identity, on fake estimators, probes, transports and receipt writers. Nothing
touches the network or a provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

import conference_pipeline.stage_script_registry as stage_script_registry
from conference_pipeline.production import (
    ProductionRunnerContext,
    corpus_freeze_stage,
    preflight_stage,
    source_discovery_stage,
    stage_table,
)
from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production.hooks import ConferenceHooks
from conference_pipeline.production.receipts import StageIO
from conference_pipeline.production_preflight import PLANNED_PROVIDER_PROBES, ProviderProbeResult, ProviderProbeSpec
from conference_pipeline.provider_receipts import utc_now
from conference_pipeline.stage_script_registry import load_stage_script_registry, required_script_ids_for_stage
from conference_pipeline.workflow_runs import validate_run_id
from tests.orchestrator_golden import materialize_fixture_workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "newconf-front-stages"
FAKE_SECRET = "fixture-secret-value-never-recorded"
STAGES = stage_table.from_registry(
    next_actions={
        "runner-preflight": "Prepare the estimate, approval and capacity input, then rerun preflight.",
        "source-discovery": "Repair the source-discovery blockers, then rerun source-discovery.",
        "corpus-freeze": "Repair the corpus-freeze blockers, then rerun corpus-freeze.",
    },
)


def _context(workspace: Path) -> ProductionRunnerContext:
    slug = workspace.name
    return ProductionRunnerContext(
        workspace=workspace,
        run_id=RUN_ID,
        stages=STAGES,
        runner_id=slug,
        runner_path=f"scripts/run_{slug}_production.py",
        receipt_type=f"conference_pipeline.{slug}_production_stage_receipt",
        repo_root=REPO_ROOT,
    )


def _second_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conferences" / "otherconf2033"
    (workspace / "storage").mkdir(parents=True)
    (workspace / "config.yaml").write_text(
        'conference:\n  slug: "otherconf2033"\n  name: "Other Conference 2033"\n  year: 2033\n', encoding="utf-8"
    )
    return workspace


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


@pytest.fixture(params=["minimal_new_conference", "otherconf2033"])
def ctx(request: pytest.FixtureRequest, tmp_path: Path) -> ProductionRunnerContext:
    if request.param == "minimal_new_conference":
        workspace = materialize_fixture_workspace("minimal_new_conference", tmp_path / "conferences")
    else:
        workspace = _second_workspace(tmp_path)
    _write_json(
        workspace / "storage" / "storage_manifest.json",
        {"artifact_families": [{"id": "papers", "path": "storage/papers", "storage_class": "backup"}]},
    )
    return _context(workspace)


IO = StageIO(write_json=lambda path, payload: _write_json(path, payload))


# --------------------------------------------------------------------------
# Stage 1: the approval / estimate / capacity preflight
# --------------------------------------------------------------------------


def _task(task_id: str, provider: str, stage: str, cost: Any, *, enabled: bool = True) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "provider": provider,
        "stage": stage,
        "model": f"{provider}-fixture-model",
        "enabled": enabled,
        "estimated_cost_usd": cost,
    }


def fake_estimator(monkeypatch: pytest.MonkeyPatch, tasks: list[dict[str, Any]]) -> None:
    report = {"pricing_refreshed_at": "2099-01-01", "tasks": tasks}
    monkeypatch.setattr(preflight_stage, "estimate_workspace", lambda *_a, **_k: report)
    monkeypatch.setattr(preflight_stage, "collect_workspace_workload", lambda *_a, **_k: {"papers": 1})


PRICED_TASKS = [
    _task("analysis_fixture", "openai", "analysis", 2.5),
    _task("failover_fixture", "openrouter", "exhaustive-topic-extraction", 1.25),
    _task("local_fixture", "local", "corpus-inventory-index", None),
]


def prepare_authority(ctx: ProductionRunnerContext, settings: preflight_stage.PreflightSettings) -> dict[str, Any]:
    prepared = preflight_stage.prepare_preflight_estimate(
        ctx.workspace, ctx.run_id, settings=settings, write_json=IO.write_json
    )
    _write_json(
        ctx.workspace / "storage" / preflight_stage.CAPACITY_POLICY_FILENAME,
        {
            "corpus_scope": settings.corpus_scope,
            "estimator_version": "fixture-v1",
            "remaining_families": {"papers": 700},
            "safety_margin_bytes": 100,
        },
    )
    preflight_stage.prepare_preflight_storage_estimate(ctx.workspace, ctx.run_id, write_json=IO.write_json)
    return prepared


def _must_not_run(label: str):
    def run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"{label} must not run")

    return run


def run_preflight(ctx: ProductionRunnerContext, settings: preflight_stage.PreflightSettings, **overrides: Any) -> dict[str, Any]:
    prepared_path, approval_path, capacity_path = preflight_stage.authority_paths(ctx.workspace, ctx.run_id)
    kwargs: dict[str, Any] = {
        "credential_values": {name: FAKE_SECRET for spec in PLANNED_PROVIDER_PROBES for name in spec.credential_names},
        "probe": lambda spec: ProviderProbeResult.succeeded(spec),
        "reload_environment": _must_not_run("environment reload"),
        "rebuild_client": _must_not_run("client rebuild"),
        "available_bytes": 10_000,
        "owners": preflight_stage.PreflightOwners(io=IO),
    }
    kwargs.update(overrides)
    return preflight_stage.run_runner_preflight(
        ctx,
        settings=settings,
        prepared_estimate_path=prepared_path,
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
        **kwargs,
    )


def test_default_settings_derive_from_the_workspace_slug(ctx: ProductionRunnerContext) -> None:
    settings = preflight_stage.default_settings(ctx)

    assert settings.corpus_scope == f"all-discovered-{ctx.slug}"
    assert settings.label and settings.label.isupper() and not any(ch.isdigit() for ch in settings.label)
    assert settings.owner_binding_code == f"{settings.label.lower()}_runner_owner_binding_invalid"
    assert preflight_stage.preflight_owner_id(settings) == required_script_ids_for_stage(
        load_stage_script_registry(), "runner-preflight"
    )[0]


def test_preflight_succeeds_with_no_hooks_under_the_runner_identity(
    ctx: ProductionRunnerContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_estimator(monkeypatch, PRICED_TASKS)
    settings = preflight_stage.default_settings(ctx)
    prepared = prepare_authority(ctx, settings)
    assert prepared["provider_stage_projection"] == {
        "openai:analysis": 2.5,
        "openrouter:exhaustive-topic-extraction": 1.25,
    }

    outcome = run_preflight(ctx, settings)

    assert outcome["status"] == "succeeded", outcome["receipt"]["checks"]
    receipt = outcome["receipt"]
    owner_id = preflight_stage.preflight_owner_id(settings)
    assert receipt["receipt_type"] == ctx.receipt_type
    assert receipt["lineage"]["code_identity"]["runner"] == ctx.runner_id
    assert receipt["required_script_ids"] == [owner_id]
    assert receipt["owner_execution_evidence"][owner_id]["runner_path"] == ctx.runner_path
    assert receipt["stage_contract"]["status"] == "pass"
    assert receipt["preflight_resume_policy"] == "rerun_required"
    assert receipt["checks"]["spend_authority"]["mode"] == preflight_stage.AUTOMATIC_AUTHORITY_MODE
    assert [probe["provider"] for probe in receipt["checks"]["provider_probes"]] == [
        spec.provider for spec in PLANNED_PROVIDER_PROBES
    ]
    on_disk = json.loads(ctx.receipt_path("runner-preflight").read_text(encoding="utf-8"))
    assert on_disk["output_digest"] == receipt["output_digest"]
    for path in ctx.workspace.rglob("*.json"):
        assert FAKE_SECRET not in path.read_text(encoding="utf-8")


def test_preflight_blocks_before_any_reload_or_probe_without_authority(ctx: ProductionRunnerContext) -> None:
    settings = preflight_stage.default_settings(ctx)

    outcome = run_preflight(
        ctx, settings, credential_values=None, probe=_must_not_run("provider probe")
    )

    assert outcome["status"] == "blocked"
    codes = {item["code"] for item in json.loads(Path(outcome["remediation_report_path"]).read_text())["blockers"]}
    assert {"canonical_estimate_required", "storage_estimate_inputs_unavailable"} <= codes
    assert outcome["receipt"]["blocker_receipt_path"]
    assert "reprepare/reapprove before any environment reload" in outcome["console"]
    markdown = (ctx.run_dir() / preflight_stage.REMEDIATION_MARKDOWN_FILENAME).read_text(encoding="utf-8")
    assert markdown.startswith(f"# {settings.label} production preflight remediation")


def test_unpriced_paid_task_is_never_projected_as_zero(ctx: ProductionRunnerContext, monkeypatch: pytest.MonkeyPatch) -> None:
    """The estimator prices an unmeasured paid task from its declared fallback budget or refuses it."""

    settings = preflight_stage.default_settings(ctx)
    fake_estimator(monkeypatch, [*PRICED_TASKS, _task("unmeasured_fixture", "gemini", "analysis", None)])
    with pytest.raises(ValueError, match="unpriced provider-stage projection: gemini:analysis"):
        preflight_stage.prepare_preflight_estimate(ctx.workspace, ctx.run_id, settings=settings, write_json=IO.write_json)

    def refusing(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("paid task quantity cannot be derived; refusing to price it at $0")

    monkeypatch.setattr(preflight_stage, "estimate_workspace", refusing)
    with pytest.raises(ValueError, match="refusing to price it at \\$0"):
        preflight_stage.prepare_preflight_estimate(ctx.workspace, ctx.run_id, settings=settings, write_json=IO.write_json)
    assert not (ctx.run_dir() / preflight_stage.PREPARED_ESTIMATE_FILENAME).exists()


def test_projection_requires_the_failover_provider(ctx: ProductionRunnerContext, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = preflight_stage.default_settings(ctx)
    fake_estimator(monkeypatch, [_task("analysis_fixture", "openai", "analysis", 2.5)])
    with pytest.raises(ValueError, match="OpenRouter failover"):
        preflight_stage.prepare_preflight_estimate(ctx.workspace, ctx.run_id, settings=settings, write_json=IO.write_json)

    without_failover = preflight_stage.default_settings(ctx, failover_providers=())
    prepared = preflight_stage.prepare_preflight_estimate(
        ctx.workspace, ctx.run_id, settings=without_failover, write_json=IO.write_json
    )
    assert prepared["paid_stages"] == ["analysis"]


def test_preflight_hooks_replace_probes_and_add_checks(ctx: ProductionRunnerContext, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_estimator(monkeypatch, PRICED_TASKS)
    settings = preflight_stage.default_settings(ctx)
    prepare_authority(ctx, settings)
    only_openai = tuple(spec for spec in PLANNED_PROVIDER_PROBES if spec.provider == "openai")
    probed: list[str] = []

    def probe(spec: ProviderProbeSpec) -> ProviderProbeResult:
        probed.append(spec.provider)
        return ProviderProbeResult.succeeded(spec)

    hooks = ConferenceHooks(provider_probes=lambda _ctx: only_openai)
    outcome = run_preflight(ctx, settings, probe=probe, hooks=hooks)
    assert outcome["status"] == "succeeded"
    assert probed == ["openai"]

    blocking = ConferenceHooks(extra_preflight_checks=(lambda _ctx: [{"code": "fixture_quirk_unready"}],))
    outcome = run_preflight(ctx, settings, probe=_must_not_run("provider probe"), hooks=blocking)
    assert outcome["status"] == "blocked"
    codes = [item["code"] for item in json.loads(Path(outcome["remediation_report_path"]).read_text())["blockers"]]
    assert codes == ["fixture_quirk_unready"]


def test_exact_owner_registration_mismatch_blocks(ctx: ProductionRunnerContext, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_estimator(monkeypatch, PRICED_TASKS)
    settings = preflight_stage.default_settings(
        ctx, owner_registration={"path": ctx.runner_path, "scope": "production", "existence": "required", "status": "required"}
    )
    prepare_authority(ctx, settings)

    outcome = run_preflight(ctx, settings, probe=_must_not_run("provider probe"))

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["checks"]["owner_binding"]["status"] == "invalid"
    report = json.loads(Path(outcome["remediation_report_path"]).read_text())
    assert [item["code"] for item in report["blockers"]] == [settings.owner_binding_code]
    assert report["blockers"][0]["clearance_evidence"] == f"current_{settings.label}_registration_and_strict_execution_receipt"


def test_paid_dispatch_refuses_without_authority_and_never_runs_the_owner(ctx: ProductionRunnerContext) -> None:
    prepared_path, approval_path, _capacity = preflight_stage.authority_paths(ctx.workspace, ctx.run_id)

    result = preflight_stage.dispatch_paid_call(
        workspace=ctx.workspace,
        run_id=ctx.run_id,
        prepared_estimate_path=prepared_path,
        approval_path=approval_path,
        provider="openai",
        stage="analysis",
        projected_cost_usd=1.0,
        spent_total_usd=0.0,
        spent_by_provider_stage={},
        owner=_must_not_run("paid owner"),
    )

    assert result["authorized"] is False
    assert {item["code"] for item in result["blockers"]} >= {"canonical_estimate_required"}


def test_authority_inputs_must_stay_inside_the_named_run(ctx: ProductionRunnerContext, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the named run directory"):
        preflight_stage.require_named_run_input(tmp_path / "elsewhere.json", workspace=ctx.workspace, run_id=ctx.run_id)


# --------------------------------------------------------------------------
# Run conventions for the (exit_code, summary) formats
# --------------------------------------------------------------------------


def run_conventions(ctx: ProductionRunnerContext, written_blockers: list[dict[str, Any]]) -> production_orchestrator.RunConventions:
    def build_receipt(*, workspace: Path, run_id: str, stage: Any, mode: str, status: str, started_at: str, checks: dict[str, Any], error: str | None = None, blocker_receipt_path: str | None = None) -> dict[str, Any]:
        receipt = {
            "schema_version": 1,
            "receipt_type": ctx.receipt_type,
            "run_id": run_id,
            "workspace": str(workspace),
            "stage_id": stage.id,
            "stage_order": stage.order,
            "mode": mode,
            "status": status,
            "started_at": started_at,
            "checks": checks,
            "error": error,
            "blocker_receipt_path": blocker_receipt_path,
        }
        return ctx.finalize(receipt, stage_id=stage.id)

    def write_blocker(**kwargs: Any) -> Path:
        written_blockers.append(kwargs)
        return _write_json(ctx.run_dir() / "blocker-receipts" / f"{kwargs['stage_id']}-{len(written_blockers)}.json", kwargs)

    return production_orchestrator.RunConventions(
        build_receipt=build_receipt,
        write_receipt=lambda workspace, run_id, stage, receipt: _write_json(ctx.receipt_path(stage.id), receipt),
        write_summary=lambda **kwargs: production_orchestrator.write_run_summary(
            summary_path=ctx.run_dir() / "summary.json",
            generated_at=utc_now(),
            unblock_receipts=[],
            write_json=_write_json,
            **kwargs,
        ),
        active_blockers=lambda *_a, **_k: [],
        context=lambda workspace, run_id: _context(workspace),
        run_dir=lambda workspace, run_id: workspace / "storage" / "workflow-runs" / validate_run_id(run_id),
        now=utc_now,
        validate_run_id=validate_run_id,
        write_blocker=write_blocker,
    )


# --------------------------------------------------------------------------
# Stage 2: probed source intake, the HTTP probe and the governed policy receipts
# --------------------------------------------------------------------------


def _probe(ok: bool, *, auth: bool = False, final_url: str | None = None) -> dict[str, Any]:
    return {
        "ok": ok,
        "auth_required": auth,
        "status_code": 200 if ok else (403 if auth else 404),
        "final_url": final_url,
        "content_type": "text/html",
        "bytes_sampled": 10,
        "title": "Fixture",
        "link_count": 0,
        "sample_links": [],
        "error": None if ok else "HTTP 404",
    }


def seed_intake(ctx: ProductionRunnerContext) -> None:
    _write_json(
        ctx.workspace / source_discovery_stage.SOURCE_INTAKE_INVENTORY,
        {
            "conference": "Fixture Conference",
            "source_urls": {"proceedings_canonical_candidate": "https://example.test/proceedings-canonical"},
            "source_families": [
                {"id": "papers", "source_url": "https://example.test/papers"},
                {"id": "talks", "source_url": "https://example.test/talks"},
                {"id": "proceedings", "source_url": "https://example.test/proceedings-generated"},
            ],
        },
    )


def fixture_probe(url: str, *, timeout_seconds: float, recovery: bool = False) -> dict[str, Any]:
    del timeout_seconds
    if url.endswith("/talks"):
        return _probe(recovery, auth=not recovery, final_url=url)
    if url.endswith("-generated"):
        return _probe(False)
    return _probe(True, final_url=url)


def run_intake(ctx: ProductionRunnerContext, blockers: list[dict[str, Any]], **settings: Any) -> tuple[int, dict[str, Any]]:
    return source_discovery_stage.run_intake_probe_discovery(
        run_conventions(ctx, blockers),
        workspace=ctx.workspace,
        run_id=ctx.run_id,
        stage=ctx.stage("source-discovery"),
        mode="live",
        env_file_status=lambda: {"attempted": False, "exists": False, "loaded": False},
        timeout_seconds=1.0,
        probe=fixture_probe,
        cost_approval=lambda: {"approved": False},
        settings=source_discovery_stage.IntakeProbeSettings(display_name="Fixture Conference", **settings),
        write_json=_write_json,
    )


def test_intake_probe_discovery_recovers_through_auth_retry_and_canonical_candidate(ctx: ProductionRunnerContext) -> None:
    seed_intake(ctx)
    blockers: list[dict[str, Any]] = []

    exit_code, summary = run_intake(ctx, blockers, canonical_candidates={"proceedings": "proceedings_canonical_candidate"})

    assert (exit_code, summary["status"]) == (0, "succeeded"), summary
    inventory = json.loads((ctx.workspace / source_discovery_stage.SOURCE_DISCOVERY_INVENTORY).read_text())
    assert inventory["receipt_type"] == f"conference_pipeline.{ctx.runner_id}_source_discovery_inventory"
    statuses = {item["id"]: item["discovery_status"] for item in inventory["sources"]}
    assert statuses == {
        "papers": "reachable",
        "talks": "reachable_after_auth_retry",
        "proceedings": "recovered_to_canonical_candidate",
    }
    report = (ctx.workspace / source_discovery_stage.SOURCE_DISCOVERY_REPORT).read_text()
    assert report.startswith("# Fixture Conference Source Discovery Report")
    assert blockers == []


def test_intake_probe_discovery_blocks_without_a_declared_candidate(ctx: ProductionRunnerContext) -> None:
    seed_intake(ctx)
    blockers: list[dict[str, Any]] = []

    exit_code, summary = run_intake(ctx, blockers)

    assert (exit_code, summary["status"], summary["halt_stage_id"]) == (1, "blocked", "source-discovery")
    assert [item["stage_id"] for item in blockers] == ["source-discovery"]
    assert "proceedings" in blockers[0]["reason"]


def test_intake_probe_discovery_blocks_on_missing_intake(ctx: ProductionRunnerContext) -> None:
    blockers: list[dict[str, Any]] = []

    exit_code, summary = run_intake(ctx, blockers)

    assert exit_code == 1
    receipt = json.loads(ctx.receipt_path("source-discovery").read_text())
    assert receipt["checks"]["source_intake_inventory_present"] is False
    assert receipt["checks"]["live_provider_calls_made"] is False
    assert "missing source-intake inventory" in blockers[0]["reason"]


class _FakeResponse:
    status = 200
    headers = {"content-type": "text/html"}

    def __init__(self, url: str, body: bytes) -> None:
        self._url, self._body = url, body

    def read(self, _limit: int) -> bytes:
        return self._body

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None


def test_http_probe_records_title_links_and_auth_posture_without_network() -> None:
    body = b"<html><title>Fixture  Papers</title><a href='/a'>a</a><a href='b'>b</a></html>"
    seen: list[str] = []

    def opener(request: Any, timeout: float) -> _FakeResponse:
        seen.append(request.get_header("User-agent"))
        return _FakeResponse("https://example.test/papers/", body)

    record = source_discovery_stage.http_probe(
        "https://example.test/papers", timeout_seconds=1.0, user_agent="fixture-agent", opener=opener, now=lambda: "T"
    )
    assert (record["ok"], record["title"], record["link_count"]) == (True, "Fixture Papers", 2)
    assert record["sample_links"] == ["https://example.test/a", "https://example.test/papers/b"]
    assert seen == ["fixture-agent"]

    def forbidden(request: Any, timeout: float) -> Any:
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

    record = source_discovery_stage.http_probe(
        "https://example.test/gated", timeout_seconds=1.0, user_agent="fixture-agent", opener=forbidden, now=lambda: "T"
    )
    assert (record["ok"], record["auth_required"], record["error"]) == (False, True, "HTTP 403")


def test_policy_discovery_collection_block_writes_immutable_attempts(ctx: ProductionRunnerContext) -> None:
    owners = source_discovery_stage.SourceDiscoveryOwners(
        prepare_estimate=_must_not_run("estimate"),
        prepare_storage_estimate=_must_not_run("capacity"),
        prepare_authority=_must_not_run("frontier authority"),
        runner_preflight=_must_not_run("fresh preflight"),
        dispatch_paid_call=_must_not_run("paid dispatch"),
        io=IO,
    )
    prepared_path, approval_path, capacity_path = preflight_stage.authority_paths(ctx.workspace, ctx.run_id)

    def run() -> dict[str, Any]:
        return source_discovery_stage.run_source_discovery_stage(
            ctx,
            owners=owners,
            prepared_estimate_path=prepared_path,
            approval_path=approval_path,
            capacity_estimate_path=capacity_path,
            collector=lambda **_kwargs: {"status": "blocked", "blockers": [{"code": "fixture_family_unreachable"}]},
        )

    first = run()
    assert first["status"] == "blocked"
    owner_id = source_discovery_stage.stage_owner_id(None)
    assert first["receipt"]["receipt_type"] == ctx.receipt_type
    assert first["receipt"]["required_script_ids"] == [owner_id]
    assert first["receipt"]["owner_execution_evidence"][owner_id]["provider_calls_made"] == 0
    assert Path(first["receipt_path"]) == ctx.receipt_path("source-discovery")
    canonical_bytes = ctx.receipt_path("source-discovery").read_bytes()

    second = run()
    assert Path(second["receipt_path"]).name.startswith("stage-02-source-discovery.attempt-")
    assert ctx.receipt_path("source-discovery").read_bytes() == canonical_bytes


def test_decision_ledger_type_is_the_runner_artifact_type(ctx: ProductionRunnerContext) -> None:
    assert source_discovery_stage.decision_ledger_type(ctx) == f"conference_pipeline.{ctx.runner_id}_source_discovery_decision_ledger"
    assert ctx.slug not in source_discovery_stage.decision_ledger_type("other")


# --------------------------------------------------------------------------
# Stage 3: program-item freeze and source-family freeze
# --------------------------------------------------------------------------


def seed_policy_discovery(ctx: ProductionRunnerContext, *, program_items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    prefix = f"{ctx.slug}:program-item:"
    inventory: dict[str, Any] = {
        "run_id": ctx.run_id,
        "stage_id": "source-discovery",
        "status": "prepared",
        "blockers": [],
        "source_families": {"papers": {"status": "collected"}},
        "source_records": [
            {"source_record_id": "rec-1", "family_id": "papers", "aliases": ["https://example.test/p/1"]},
            {"source_record_id": "rec-2", "family_id": "papers", "aliases": ["not-a-url"]},
        ],
        "program_items": program_items
        if program_items is not None
        else [
            {"program_item_id": f"{prefix}one", "source_record_ids": ["rec-1"]},
            {"program_item_id": f"{prefix}two", "source_record_ids": ["rec-2"]},
        ],
    }
    inventory["inventory_sha256"] = corpus_freeze_stage.corpus_inventory_digest(inventory)
    _write_json(ctx.workspace / source_discovery_stage.SOURCE_DISCOVERY_INVENTORY, inventory)
    source_discovery_stage.write_source_stage_receipt(
        ctx,
        owner_id=source_discovery_stage.stage_owner_id(None),
        status="succeeded",
        collection={"status": "succeeded"},
        owner_result={"provider_calls_made": 0, "provider_receipt_paths": [], "actual_cost_usd": 0.0},
        inventory=inventory,
        blockers=[],
        frontier_authority=None,
        stage_contract={"status": "pass"},
        io=IO,
    )
    return inventory


@pytest.fixture
def registered(ctx: ProductionRunnerContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProductionRunnerContext:
    """A registry copy with this new workspace's Stage 2/3 bindings: all a new conference adds (batch R)."""

    registry = json.loads(stage_script_registry.DEFAULT_STAGE_SCRIPT_REGISTRY_PATH.read_text(encoding="utf-8"))
    identity = f"conferences/{ctx.workspace.name}"
    for stage_id in ("source-discovery", "corpus-freeze"):
        binding = registry["stage_bindings"][stage_id]
        binding["workspace_bindings"][identity] = {
            "registry_stage_id": f"{stage_id}.{ctx.workspace.name}",
            "required_script_ids": list(binding["required_script_ids"]),
        }
    path = _write_json(tmp_path / "registry" / "stage_owner_registry.json", registry)
    monkeypatch.setattr(stage_script_registry, "DEFAULT_STAGE_SCRIPT_REGISTRY_PATH", path)
    return ctx


def test_program_item_freeze_fails_closed_without_a_workspace_binding(ctx: ProductionRunnerContext) -> None:
    seed_policy_discovery(ctx)

    outcome = corpus_freeze_stage.run_corpus_freeze_stage(ctx, io=IO)

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["blockers"] == [{"code": "final_stage_contract_failed", "affected_stage": "corpus-freeze"}]
    assert {item["code"] for item in outcome["receipt"]["stage_contract"]["blockers"]} == {
        "stage_contract_workspace_binding_missing"
    }


def test_program_item_freeze_runs_with_no_hooks_and_no_pin(registered: ProductionRunnerContext) -> None:
    ctx = registered
    inventory = seed_policy_discovery(ctx)

    outcome = corpus_freeze_stage.run_corpus_freeze_stage(ctx, io=IO)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    manifest = outcome["manifest"]
    assert manifest["receipt_type"] == f"conference_pipeline.{ctx.runner_id}_corpus_freeze"
    assert manifest["source_lineage"]["pinned_inventory_sha256"] == inventory["inventory_sha256"]
    assert manifest["policy_exclusions"] == []
    dispositions = {item["program_item_id"].rsplit(":", 1)[1]: item for item in manifest["program_items"]}
    assert dispositions["one"]["planned_acquisition"][0]["source_urls"] == ["https://example.test/p/1"]
    assert dispositions["two"]["unreachable_source_path"]["phase"] == "download-corpus"
    receipt = outcome["receipt"]
    assert receipt["receipt_type"] == ctx.receipt_type
    assert receipt["stage_contract"]["status"] == "pass"
    assert receipt["corpus_manifest"]["selection_sha256"] == manifest["selection_sha256"]
    report = (ctx.workspace / corpus_freeze_stage.CORPUS_FREEZE_REPORT_PATH).read_text()
    assert report.startswith(f"# {manifest['conference']} Corpus Freeze Report")


def test_program_item_freeze_blocks_on_caps_pins_and_policy_hooks(registered: ProductionRunnerContext) -> None:
    ctx = registered
    seed_policy_discovery(ctx)

    capped = corpus_freeze_stage.run_corpus_freeze_stage(ctx, command_args=("--stage", "corpus-freeze", "--limit=5"), io=IO)
    assert capped["status"] == "blocked"
    assert {"code": "silent_item_cap_rejected", "affected_stage": "corpus-freeze", "flags": ["--limit"]} in capped["receipt"]["blockers"]

    pinned = corpus_freeze_stage.run_corpus_freeze_stage(
        ctx, settings=corpus_freeze_stage.ProgramItemFreezeSettings(pinned_inventory_sha256="0" * 64), io=IO
    )
    assert "source_inventory_digest_not_pinned" in {item["code"] for item in pinned["receipt"]["blockers"]}

    hooks = ConferenceHooks(
        corpus_exclusions=lambda _inventory: (
            [{"exclusion_id": "fixture_policy", "family_id": "papers", "selection_effect": "kept_auditable"}],
            [],
        )
    )
    with_policy = corpus_freeze_stage.run_corpus_freeze_stage(ctx, hooks=hooks, io=IO)
    assert with_policy["status"] == "succeeded"
    assert with_policy["manifest"]["counts"]["policy_exclusion_postures"] == 1


def test_program_item_freeze_rejects_foreign_program_items(ctx: ProductionRunnerContext) -> None:
    seed_policy_discovery(ctx, program_items=[{"program_item_id": "otherconf:program-item:x", "source_record_ids": ["rec-1"]}])

    outcome = corpus_freeze_stage.run_corpus_freeze_stage(ctx, io=IO)

    assert outcome["status"] == "blocked"
    assert "program_item_identity_invalid" in {item["code"] for item in outcome["receipt"]["blockers"]}


def test_source_family_freeze_includes_every_family_with_plans(ctx: ProductionRunnerContext) -> None:
    seed_intake(ctx)
    run_intake(ctx, [], canonical_candidates={"proceedings": "proceedings_canonical_candidate"})
    blockers: list[dict[str, Any]] = []
    settings = corpus_freeze_stage.SourceFamilyFreezeSettings(
        display_name="Fixture Conference",
        family_plans=(
            corpus_freeze_stage.FamilyAcquisitionPlan(frozenset({"talks"}), ("download-corpus", "transcription-ocr"), "talks plan"),
        ),
    )

    def freeze(command_suffix: str = "") -> tuple[int, dict[str, Any]]:
        return corpus_freeze_stage.run_source_family_freeze(
            run_conventions(ctx, blockers),
            workspace=ctx.workspace,
            run_id=ctx.run_id,
            stage=ctx.stage("corpus-freeze"),
            mode="live",
            env_file_status=lambda: {"attempted": False, "exists": False, "loaded": False},
            cost_approval=lambda: {"approved": False},
            checked_command=lambda stage_id: f"run --stage {stage_id}{command_suffix}",
            settings=settings,
            write_json=_write_json,
        )

    exit_code, summary = freeze()
    assert (exit_code, summary["status"]) == (0, "succeeded"), summary
    selection = json.loads((ctx.workspace / corpus_freeze_stage.CORPUS_FREEZE_PATH).read_text())
    assert selection["receipt_type"] == f"conference_pipeline.{ctx.runner_id}_corpus_freeze"
    plans = {item["source_family_id"]: item["planned_acquisition"]["planned_stages"] for item in selection["included_records"]}
    assert plans == {
        "papers": ["download-corpus"],
        "talks": ["download-corpus", "transcription-ocr"],
        "proceedings": ["download-corpus"],
    }

    exit_code, summary = freeze(" --max-items 3")
    assert (exit_code, summary["status"]) == (1, "blocked")
    assert blockers and blockers[-1]["reason"] == "corpus freeze failed selection policy checks"


# --------------------------------------------------------------------------
# Preflight command line (shared by every orchestrator on the approval preflight)
# --------------------------------------------------------------------------


def _preflight_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--stage")
    preflight_stage.add_preflight_cli_args(parser)
    return parser


def test_preflight_authority_args_default_to_the_named_run(ctx: ProductionRunnerContext, tmp_path: Path) -> None:
    parser = _preflight_parser()
    args = parser.parse_args(["--workspace", str(ctx.workspace), "--run-id", ctx.run_id])
    assert preflight_stage.preflight_authority_args(args) == preflight_stage.authority_paths(ctx.workspace, ctx.run_id)

    override = tmp_path / "approval.json"
    args = parser.parse_args(["--workspace", str(ctx.workspace), "--run-id", ctx.run_id, "--approval-file", str(override)])
    assert preflight_stage.preflight_authority_args(args)[1] == override


def test_prepare_commands_reject_stage_flags_and_each_other() -> None:
    parser = _preflight_parser()
    raw = ["--prepare-preflight-estimate", "--stage", "runner-preflight"]
    with pytest.raises(SystemExit):
        preflight_stage.validate_prepare_command(parser, parser.parse_args(raw), raw)
    raw = ["--prepare-preflight-estimate", "--prepare-storage-estimate"]
    with pytest.raises(SystemExit):
        preflight_stage.validate_prepare_command(parser, parser.parse_args(raw), raw)
    commands = preflight_stage.preflight_extra_commands(
        prepare_estimate=lambda workspace, run_id: {"run_id": run_id},
        prepare_storage_estimate=_must_not_run("capacity"),
        label="Fixture",
    )
    assert sorted(commands) == ["prepare_preflight_estimate", "prepare_storage_estimate"]
    assert commands["prepare_storage_estimate"].failure_label == "cannot prepare Fixture storage estimate"
