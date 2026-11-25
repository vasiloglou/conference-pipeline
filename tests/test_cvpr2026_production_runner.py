"""Deterministic policy coverage for the CVPR 2026 production runner.

These tests deliberately exercise only injected fakes and local JSON inputs.  They
must never load an environment file, construct a provider SDK client, or make a
network request.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock, Thread
import time
from types import ModuleType, SimpleNamespace

import fitz
import pytest

from tests.lineage_fixtures import stamp_upstream_stage_receipts

import conference_pipeline.stage_contracts as stage_contracts
import conference_pipeline.stage_script_registry as stage_script_registry
from conference_pipeline.production import decks as production_decks
from conference_pipeline.production import podcast_stage as production_podcast_stage
from conference_pipeline.production import presentation_stage as production_presentation_stage
from conference_pipeline.production import presentation_steps as production_presentation_steps
from conference_pipeline.production import qa_synthesis_stage as production_qa_synthesis_stage
from conference_pipeline.production import release_stage as production_release_stage
from conference_pipeline.production import source_discovery_stage as production_source_discovery_stage
from conference_pipeline.production_preflight import (
    AUTOMATIC_AUTHORIZATION_THRESHOLD_USD,
    ApprovalDeclaration,
    PLANNED_PROVIDER_PROBES,
    ProviderModelChoice,
    ProviderProbeResult,
    ProviderProbeSpec,
    ProviderStageAllocation,
    authorize_prospective_call,
    build_approval_scope_fingerprint,
    build_complete_run_estimate_snapshot,
    build_remediation_report,
    build_static_readiness_fingerprint,
    derive_automatic_approval,
    execute_probe_with_single_recovery,
    evaluate_resume_compatibility,
    normalize_root_causes,
    prepare_preflight_estimate,
    validate_acknowledgement_clearance,
    validate_approval_scope,
    validate_credential_inventory,
    validate_provider_probe_specs,
    validate_storage_estimate,
)
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
)
from conference_pipeline.source_discovery import (
    FetchResponse,
    SafeSourceFetcher,
    build_candidate_frontier,
    build_canonical_inventory,
    canonical_json_digest as source_discovery_canonical_json_digest,
    immutable_id,
    publish_canonical_inventory,
)
from conference_pipeline.acquisition.cvpr_registration_media import (
    FULL_R4_SCOPE,
    OVERLAY_LEDGER_RELATIVE_PATH,
    PINNED_R3_PREDECESSORS,
    R3_RUN_ID,
    build_candidate_plan,
    build_registration_media_overlay,
    canonical_registration_media_digest,
)
from conference_pipeline.acquisition.cvpr_event_media import (
    EVENT_CATEGORY_COUNTS,
    apply_caption_outcomes,
    apply_event_page_outcome,
    apply_event_page_outcomes,
    apply_r4_poster_outcomes,
    build_event_media_inventory,
    canonical_event_media_inventory_digest,
    merge_caption_candidates,
)
from conference_pipeline.acquisition import cvpr_transcription_ocr as cvpr_transcription_ocr_adapter
from conference_pipeline.acquisition.workshop_pipeline import slugify_venueid
from conference_pipeline.conference_identity import parse_virtual_site_boundary
from conference_pipeline.indexing import paper_source_metadata as platform_paper_source_metadata


SECRET_SENTINEL = "cvpr-fixture-secret-must-never-persist"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_cvpr2026_production.py"
FIXTURE_BOUNDARY = parse_virtual_site_boundary("https://cvpr.thecvf.com/virtual/2026")
FIXTURE_MODELS = (
    ProviderModelChoice(provider="openai", model="fixture-analysis-model"),
    ProviderModelChoice(provider="openrouter", model="fixture-failover-model"),
)

_raw_build_candidate_plan = build_candidate_plan
_raw_build_registration_media_overlay = build_registration_media_overlay
_raw_build_event_media_inventory = build_event_media_inventory
_raw_apply_r4_poster_outcomes = apply_r4_poster_outcomes
_raw_apply_event_page_outcome = apply_event_page_outcome
_raw_apply_event_page_outcomes = apply_event_page_outcomes
_raw_apply_caption_outcomes = apply_caption_outcomes
_raw_merge_caption_candidates = merge_caption_candidates



def _opus_model() -> str:
    """The catalog's current Opus model (these tests pin the tier, not a generation)."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def _fixture_boundary_call(function: object, *args: object, **kwargs: object) -> object:
    assert callable(function)
    kwargs.setdefault("boundary", FIXTURE_BOUNDARY)
    return function(*args, **kwargs)


def build_candidate_plan(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_build_candidate_plan, *args, **kwargs)  # type: ignore[return-value]


def build_registration_media_overlay(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_build_registration_media_overlay, *args, **kwargs)  # type: ignore[return-value]


def build_event_media_inventory(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_build_event_media_inventory, *args, **kwargs)  # type: ignore[return-value]


def apply_r4_poster_outcomes(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_apply_r4_poster_outcomes, *args, **kwargs)  # type: ignore[return-value]


def apply_event_page_outcome(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_apply_event_page_outcome, *args, **kwargs)  # type: ignore[return-value]


def apply_event_page_outcomes(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_apply_event_page_outcomes, *args, **kwargs)  # type: ignore[return-value]


def apply_caption_outcomes(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_apply_caption_outcomes, *args, **kwargs)  # type: ignore[return-value]


def merge_caption_candidates(*args: object, **kwargs: object) -> dict[str, object]:
    return _fixture_boundary_call(_raw_merge_caption_candidates, *args, **kwargs)  # type: ignore[return-value]


def write_runner_workspace(tmp_path: Path) -> Path:
    """Create the minimum local CVPR workspace contract used by runner tests."""

    workspace = tmp_path / "cvpr2026"
    (workspace / "storage").mkdir(parents=True)
    (workspace / "config.yaml").write_text(
        "conference: {slug: cvpr2026, name: CVPR 2026}\n"
        "urls: {virtual: https://cvpr.thecvf.com/virtual/2026}\n",
        encoding="utf-8",
    )
    (workspace / "storage" / "storage_manifest.json").write_text(
        json.dumps(
            {
                "workspace": str(workspace),
                "artifact_families": [
                    {"id": "papers", "path": "storage/papers", "storage_class": "backup"},
                    {"id": "media", "path": "storage/media", "storage_class": "backup"},
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "storage" / "preflight-storage-policy.json").write_text(
        json.dumps(
            {
                "corpus_scope": "all-discovered-cvpr2026",
                "estimator_version": "fixture-v1",
                "remaining_families": {"paper_pdf": 700, "media": 300},
                "safety_margin_bytes": 100,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return workspace


def test_r4_and_r5_owners_load_one_configured_boundary_before_any_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both named owners inject the same parsed workspace policy into their seams."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    observed_identities: list[Path] = []
    observed_boundaries: list[object] = []
    real_identity_loader = runner.load_conference_identity

    def tracked_identity_loader(path: Path) -> object:
        observed_identities.append(Path(path))
        return real_identity_loader(path)

    predecessors, plan, overlay = registration_media_fixture_inputs()
    monkeypatch.setattr(runner, "load_conference_identity", tracked_identity_loader)

    def r4_inputs(_workspace: Path, *, boundary: object) -> tuple[list[dict[str, str]], dict[str, object], dict[str, object]]:
        observed_boundaries.append(boundary)
        return predecessors, plan, overlay

    r4 = runner.run_cvpr_registration_media_stage(
        workspace=workspace,
        run_id=runner.R4_RUN_ID,
        input_loader=r4_inputs,
        opener_factory=lambda **_kwargs: (
            None,
            {"preflight_status": "blocked", "auth_status": "registration_auth_required", "secrets_copied": False},
        ),
        available_bytes=1_000_000,
    )
    assert r4["status"] == "blocked"

    baseline = event_media_fixture_inventory()

    def r5_inputs(_workspace: Path, *, boundary: object) -> dict[str, object]:
        observed_boundaries.append(boundary)
        return baseline

    monkeypatch.setattr(runner, "_event_media_inventory_inputs", r5_inputs)
    r5 = runner.run_cvpr_event_media_stage(
        workspace=workspace,
        run_id=runner.R5_EVENT_MEDIA_RUN_ID,
        opener_factory=lambda **_kwargs: (
            None,
            {"preflight_status": "blocked", "auth_status": "registration_auth_required", "secrets_copied": False},
        ),
    )
    assert r5["status"] == "blocked"
    assert observed_identities == [workspace, workspace]
    assert observed_boundaries == [FIXTURE_BOUNDARY, FIXTURE_BOUNDARY]


def event_media_fixture_inventory() -> dict[str, object]:
    """Build the exact r5 census with the immutable 2,565-route r4 reuse set."""

    records: list[dict[str, object]] = []
    specs = (
        ("poster", "Poster", EVENT_CATEGORY_COUNTS["poster"]),
        ("invited-talk", "Keynote", EVENT_CATEGORY_COUNTS["keynote"]),
        ("tutorial", "Tutorial", EVENT_CATEGORY_COUNTS["tutorial"]),
        ("workshop", "Workshop", EVENT_CATEGORY_COUNTS["workshop"]),
        ("session", "Oral Session", EVENT_CATEGORY_COUNTS["oral"]),
        ("session", "Poster Session", EVENT_CATEGORY_COUNTS["session"]),
        ("demonstration", "Demo", 5),
        ("events", "Event", 7),
        ("meeting", "Meeting", 1),
        ("mentorship", "Mentorship", 2),
        ("reception", "Reception", 1),
        ("remarks", "Remarks", 1),
    )
    number = 0
    for route_family, title_prefix, count in specs:
        for _ in range(count):
            number += 1
            records.append(
                {
                    "source_record_id": f"fixture:program-schedule:{number:04d}",
                    "family_id": "program-schedule",
                    "title": f"{title_prefix} {number}",
                    "aliases": [f"https://cvpr.thecvf.com/virtual/2026/{route_family}/{number}"],
                }
            )
    inventory = build_event_media_inventory(
        source_records=records,
        source_inventory_sha256="a" * 64,
    )
    posters = [event for event in inventory["events"] if event["event_category"] == "poster"]
    return apply_r4_poster_outcomes(
        inventory,
        r4_requests=[
            {
                "request_key": f"r4-{index}",
                "source_url": event["source_url"],
                "result_status": "no_admissible_official_youtube_link",
                "auth_status": "authenticated",
                "http_status": 200,
            }
            for index, event in enumerate(posters[:2_565])
        ],
        r4_media_records=[],
        r4_overlay_sha256="b" * 64,
    )


def run_cvpr_runner(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER_PATH), *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def load_runner_module() -> object:
    spec = importlib.util.spec_from_file_location("cvpr2026_runner_under_test", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def registration_media_fixture_inputs() -> tuple[list[dict[str, str]], dict[str, object], dict[str, object]]:
    """Create the complete r4 shape in memory without production payloads."""

    requests: list[dict[str, object]] = []
    items: list[dict[str, object]] = []
    next_item = 0
    for index in range(FULL_R4_SCOPE.transport_eligible_request_count):
        linked = [f"cvpr2026:registration-fixture:{next_item + offset:05d}" for offset in range(2)]
        next_item += 2
        request_key = f"eligible-{index:05d}"
        requests.append(
            {
                "request_key": request_key,
                "source_url": f"https://cvpr.thecvf.com/virtual/2026/poster/{index + 1}",
                "status": "policy_excluded_official_video",
                "linked_program_item_ids": linked,
                "source_evidence": {"source_record_id": f"fixture-source-{index}"},
            }
        )
        items.extend(
            {
                "program_item_id": item_id,
                "request_key": request_key,
                "blocker": {"code": "policy_excluded_official_video"},
            }
            for item_id in linked
        )
    for index, item_count in enumerate((3, 2, 2, 2, 2, 2)):
        linked = [f"cvpr2026:registration-fixture:{next_item + offset:05d}" for offset in range(item_count)]
        next_item += item_count
        request_key = f"boundary-{index:02d}"
        requests.append(
            {
                "request_key": request_key,
                "source_url": f"https://outside.example/virtual/2026/{index}",
                "status": "policy_excluded_official_video",
                "linked_program_item_ids": linked,
                "source_evidence": {"source_record_id": f"fixture-boundary-{index}"},
            }
        )
        items.extend(
            {
                "program_item_id": item_id,
                "request_key": request_key,
                "blocker": {"code": "policy_excluded_official_video"},
            }
            for item_id in linked
        )
    while len(items) < FULL_R4_SCOPE.total_program_item_count:
        item_id = f"cvpr2026:registration-fixture:{next_item:05d}"
        next_item += 1
        items.append(
            {
                "program_item_id": item_id,
                "request_key": f"r3-{next_item:05d}",
                "blocker": {"code": "unreachable_http_error"},
            }
        )
    plan = build_candidate_plan(
        r3_run_id=R3_RUN_ID,
        predecessors=PINNED_R3_PREDECESSORS,
        r3_request_results=requests,
        r3_program_item_results=items,
    )
    overlay = build_registration_media_overlay(
        candidate_plan=plan,
        r3_program_item_results=items,
    )
    return copy.deepcopy(PINNED_R3_PREDECESSORS), plan, overlay


def completed_zero_link_overlay(overlay: dict[str, object]) -> dict[str, object]:
    completed = copy.deepcopy(overlay)
    for request in completed["official_event_requests"]:
        if request["transport_eligible"]:
            request.update(
                {
                    "result_status": "no_admissible_official_youtube_link",
                    "auth_status": "authenticated",
                    "http_status": 200,
                }
            )
        else:
            request["result_status"] = "outside_authenticated_official_event_boundary"
    completed["status"] = "registration_media_complete"
    completed["recovery_counts"] = {
        "completed_request_count": FULL_R4_SCOPE.policy_request_count,
        "direct_youtube_media_record_count": 0,
        "transport_eligible_request_count": FULL_R4_SCOPE.transport_eligible_request_count,
        "boundary_no_network_request_count": FULL_R4_SCOPE.boundary_request_count,
    }
    completed["overlay_sha256"] = canonical_registration_media_digest(completed)
    return completed


def test_live_only_rejects_non_live_modes_before_creating_run_directory(tmp_path: Path) -> None:
    workspace = write_runner_workspace(tmp_path / "conferences")

    for mode in ("dry-run", "mock", "planned"):
        run_id = f"non-live-{mode}"
        result = run_cvpr_runner(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                run_id,
                "--stage",
                "runner-preflight",
                "--mode",
                mode,
            ]
        )

        assert result.returncode != 0
        assert "live-only" in result.stderr
        assert not (workspace / "storage" / "workflow-runs" / run_id).exists()


def test_cvpr_stage_registry_declares_completed_local_stage4_and_r5_stage5_owners_as_implemented() -> None:
    runner = load_runner_module()

    assert runner.STAGES["runner-preflight"].order == 1
    assert runner.STAGES["runner-preflight"].receipt_filename == "stage-01-runner-preflight.json"
    assert runner.STAGES["runner-preflight"].implemented is True
    assert runner.STAGES["source-discovery"].implemented is True
    assert runner.STAGES["corpus-freeze"].implemented is True
    assert runner.STAGES["download-corpus"].implemented is True
    assert runner.STAGES["transcription-ocr"].implemented is True
    assert runner.STAGES["corpus-inventory-index"].implemented is True
    assert runner.STAGES["exhaustive-topic-extraction"].implemented is True
    assert runner.STAGES["analysis"].implemented is True
    assert runner.STAGES["qa-gap-fill-synthesis"].implemented is True
    assert runner.STAGES["podcast"].implemented is True
    assert runner.STAGES["presentation"].implemented is True
    # Phases 11 and 12 are implemented now. The proof stage's id is "proof",
    # matching this milestone's live command (--stage proof), its receipt name
    # (stage-12-proof.json) and the shared stage-owner registry's binding; the
    # earlier "final-proof" spelling matched none of the three.
    assert runner.STAGES["release"].implemented is True
    assert runner.STAGES["release"].receipt_filename == "stage-11-release.json"
    assert runner.STAGES["proof"].implemented is True
    assert runner.STAGES["proof"].receipt_filename == "stage-12-proof.json"
    assert "final-proof" not in runner.STAGES
    assert all(
        not stage.implemented
        for stage_id, stage in runner.STAGES.items()
        if stage_id
        not in {
            "runner-preflight",
            "release",
            "proof",
            "source-discovery",
            "corpus-freeze",
            "download-corpus",
            "transcription-ocr",
            "corpus-inventory-index",
            "exhaustive-topic-extraction",
            "analysis",
            "qa-gap-fill-synthesis",
            "podcast",
            "presentation",
            "notebooklm-podcast",
        }
    )


@pytest.mark.repo_data("conferences/cvpr2026/source-discovery.yaml")
def test_snapshot_collector_extracts_a_canonical_spine_before_paid_stage_work(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    (workspace / "source-discovery.yaml").write_text(
        (REPO_ROOT / "conferences" / "cvpr2026" / "source-discovery.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    policy = runner.load_source_policy(workspace / "source-discovery.yaml")
    bodies = {
        "program-schedule": b'<main data-event-family="oral"><a data-event-id="program-1" href="/event/1">Paper One</a></main>',
        "cvf-openaccess": b'<main><a data-paper-id="paper-1" href="/paper-1.pdf">Paper One</a></main>',
        "cvf-openaccess-day-2026-06-05": (
            b'<dt class="ptitle"><a href="/content/CVPR2026/html/One_CVPR_2026_paper.html">Paper One</a></dt>'
            b'<dd>[<a href="/content/CVPR2026/papers/One_CVPR_2026_paper.pdf">pdf</a>]</dd>'
        ),
        "cvf-openaccess-day-2026-06-06": (
            b'<dt class="ptitle"><a href="/content/CVPR2026/html/Two_CVPR_2026_paper.html">Paper Two</a></dt>'
            b'<dd>[<a href="/content/CVPR2026/papers/Two_CVPR_2026_paper.pdf">pdf</a>]</dd>'
        ),
        "cvf-openaccess-day-2026-06-07": (
            b'<dt class="ptitle"><a href="/content/CVPR2026/html/Three_CVPR_2026_paper.html">Paper Three</a></dt>'
            b'<dd>[<a href="/content/CVPR2026/papers/Three_CVPR_2026_paper.pdf">pdf</a>]</dd>'
        ),
        "openreview-conference": b'{"notes": []}',
        "openreview-workshop": b'{"count": 1, "groups": [{"id": "thecvf.com/CVPR/2026/Workshop/Fixture"}]}',
        "virtual-paper-media-posture": b"<main>Registration required.</main>",
        "youtube": b'<main><a href="/watch?v=abc123">Paper One</a></main>',
    }

    auth_attempts: list[str] = []

    class FakeOpenReviewAuth:
        def headers(self) -> dict[str, str]:
            return {"Authorization": "Bearer test-token"}

        def receipt_fields(self) -> dict[str, object]:
            return {
                "openreview_auth_mode": "authenticated",
                "openreview_auth_env_names": ["OPENREVIEW_USERNAME", "OPENREVIEW_PASSWORD"],
                "secret_values_recorded": False,
            }

    def transport(url: str, *, headers: dict[str, str], **_: object) -> FetchResponse:
        family = next(item for item in policy.families.values() if item.source_url == url)
        if family.auth_posture == "required":
            auth_attempts.append(family.id)
            assert headers == {"Authorization": "Bearer test-token"}
        return FetchResponse(
            200,
            {"Content-Type": "application/json" if family.source_kind == "json" else "text/html"},
            bodies[family.id],
            final_url=url,
        )

    result = runner._collect_cvpr_source_discovery_snapshots(
        workspace=workspace,
        run_id="fixture-source-collector",
        fetcher=SafeSourceFetcher(transport=transport),
        openreview_auth_loader=FakeOpenReviewAuth,
    )
    inventory = json.loads(Path(result["canonical_inventory_path"]).read_text(encoding="utf-8"))

    assert result["status"] == "succeeded"
    assert not any(blocker["code"] == "source_record_extraction_required" for blocker in result["blockers"])
    assert result["source_record_count"] > 0
    assert result["program_item_count"] > 0
    assert inventory["status"] == "prepared"
    assert inventory["candidate_frontier_sha256"] == result["candidate_frontier_sha256"]
    assert auth_attempts == ["openreview-conference", "openreview-workshop"]
    assert all(
        result["family_snapshots"][family_id]["auth_receipt"]["secret_values_recorded"] is False
        for family_id in auth_attempts
    )


def test_runner_uses_the_active_worktree_environment_file() -> None:
    runner = load_runner_module()

    assert runner.CANONICAL_ENV_FILE == REPO_ROOT / ".env"


def test_cvpr_yt_dlp_prerequisite_audit_requires_declared_resolvable_dependency() -> None:
    runner = load_runner_module()

    ready = runner.audit_yt_dlp_prerequisite(
        command_resolver=lambda name: "/fixture/bin/yt-dlp" if name == "yt-dlp" else None,
        module_finder=lambda name: object() if name == "yt_dlp" else None,
        version_resolver=lambda name: "2025.01.01" if name == "yt-dlp" else None,
    )
    unavailable = runner.audit_yt_dlp_prerequisite(
        command_resolver=lambda _name: None,
        module_finder=lambda _name: None,
        version_resolver=lambda _name: None,
    )

    assert ready["status"] == "pass"
    assert ready["declaration"] == {"pyproject": True, "requirements": True}
    assert ready["resolution"] == {
        "command": "yt-dlp",
        "command_path": "/fixture/bin/yt-dlp",
        "module": "yt_dlp",
        "module_resolved": True,
        "installed_version": "2025.01.01",
    }
    assert unavailable["status"] == "blocked"
    assert unavailable["blockers"] == [{"code": "yt_dlp_unavailable"}]


def test_cvpr_public_talk_lane_rejects_caps_and_tavily_before_harvest() -> None:
    runner = load_runner_module()

    hits = runner._cvpr_public_talk_lane_forbidden_hits(
        ("--pilot-limit", "1", "--tavily", "--max-items=2")
    )

    assert hits == ["--max-items", "--pilot-limit", "--tavily"]


def test_google_probe_retains_sdk_client_until_models_request_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    calls: list[str] = []

    class FakeModels:
        def list(self) -> None:
            calls.append("models.list")

    class FakeGoogleClient:
        def __init__(self) -> None:
            calls.append("client.created")
            self.models = FakeModels()

    google_module = ModuleType("google")
    google_module.genai = SimpleNamespace(Client=FakeGoogleClient)
    monkeypatch.setitem(sys.modules, "google", google_module)
    spec = next(item for item in PLANNED_PROVIDER_PROBES if item.provider == "google")

    result = runner._default_probe(spec)

    assert result.status == "succeeded"
    assert calls == ["client.created", "models.list"]


def test_cvpr_runner_rejects_traversal_run_id_before_writing_fixture_workspace(tmp_path: Path) -> None:
    workspace = write_runner_workspace(tmp_path)

    result = run_cvpr_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "../escaped-run",
            "--stage",
            "runner-preflight",
            "--mode",
            "live",
        ]
    )

    assert result.returncode != 0
    assert not (tmp_path / "escaped-run").exists()
    assert not (workspace / "storage" / "workflow-runs").exists()


def test_prepare_preflight_estimate_writes_only_canonical_local_artifact(tmp_path: Path) -> None:
    workspace = write_runner_workspace(tmp_path)

    result = run_cvpr_runner(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "prepare-estimate",
            "--prepare-preflight-estimate",
        ]
    )

    artifact_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / "prepare-estimate"
        / "preflight-full-run-estimate.json"
    )
    assert result.returncode == 0, result.stderr
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["workspace"] == str(workspace.resolve())
    assert payload["run_id"] == "prepare-estimate"
    assert payload["scope"] == "full"
    assert payload["estimate_digest"]
    assert payload["workload_digest"]
    assert payload["pricing_catalog_digest"]
    assert payload["task_spec_digest"]
    assert payload["provider_stage_projection"]
    assert any(key.startswith("openrouter:") for key in payload["provider_stage_projection"])
    assert payload["storage_estimate_inputs"]
    assert not (workspace / "storage" / "workflow-runs" / "prepare-estimate" / "stage-01-runner-preflight.json").exists()
    assert not list((workspace / "storage" / "workflow-runs" / "prepare-estimate").glob("summary.json"))


def test_prepare_storage_estimate_binds_current_named_preflight_estimate_without_live_receipts(tmp_path: Path) -> None:
    workspace = write_runner_workspace(tmp_path)
    run_id = "prepare-capacity"

    prepared = run_cvpr_runner(
        [
            "--workspace", str(workspace), "--run-id", run_id,
            "--prepare-preflight-estimate",
        ]
    )
    result = run_cvpr_runner(
        [
            "--workspace", str(workspace), "--run-id", run_id,
            "--prepare-storage-estimate",
        ]
    )

    artifact_path = workspace / "storage" / "workflow-runs" / run_id / "preflight-storage-estimate.json"
    prepared_path = workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json"
    assert prepared.returncode == 0, prepared.stderr
    assert result.returncode == 0, result.stderr
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    prepared_payload = json.loads(prepared_path.read_text(encoding="utf-8"))
    assert payload["prepared_estimate_digest"] == prepared_payload["estimate_digest"]
    assert payload["projected_remaining_bytes"] == sum(payload["remaining_families"].values())
    assert payload["safety_margin_bytes"] > 0
    assert payload["capacity_policy_digest"]
    assert not (workspace / "storage" / "workflow-runs" / run_id / "stage-01-runner-preflight.json").exists()
    assert not (workspace / "storage" / "workflow-runs" / run_id / "summary.json").exists()


def write_immutable_approval(runner: object, workspace: Path, run_id: str, prepared: dict[str, object]) -> Path:
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
    path = workspace / "storage" / "workflow-runs" / run_id / "provider-approval.json"
    path.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_capacity_estimate(workspace: Path, run_id: str, prepared: dict[str, object]) -> Path:
    inputs = prepared["storage_estimate_inputs"]
    path = workspace / "storage" / "workflow-runs" / run_id / "preflight-storage-estimate.json"
    path.write_text(
        json.dumps(
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
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def fixture_credentials() -> dict[str, str]:
    return {
        name: SECRET_SENTINEL
        for spec in PLANNED_PROVIDER_PROBES
        for name in spec.credential_names
    }


def run_successful_fixture_preflight(
    runner: object,
    workspace: Path,
    run_id: str,
    *,
    stage_registry_path: Path | None = None,
) -> dict[str, object]:
    """Exercise only the injected, local success path for owner-contract tests."""

    prepared = runner.prepare_preflight_estimate(workspace, run_id)
    approval_path = write_immutable_approval(runner, workspace, run_id, prepared)
    capacity_path = write_capacity_estimate(workspace, run_id, prepared)
    return runner.run_runner_preflight(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=(
            workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json"
        ),
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
        credential_values=fixture_credentials(),
        probe=lambda spec: ProviderProbeResult.succeeded(spec),
        reload_environment=lambda: pytest.fail("fixture success must not reload environment"),
        rebuild_client=lambda _spec: pytest.fail("fixture success must not rebuild a client"),
        available_bytes=1_100,
        stage_registry_path=stage_registry_path,
    )


def test_runner_preflight_measures_current_capacity_when_no_override_is_injected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "measured-capacity"
    prepared = runner.prepare_preflight_estimate(workspace, run_id)
    approval_path = write_immutable_approval(runner, workspace, run_id, prepared)
    capacity_path = write_capacity_estimate(workspace, run_id, prepared)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _path: type("DiskUsage", (), {"free": 1_100})())

    result = runner.run_runner_preflight(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
        credential_values=fixture_credentials(),
        probe=lambda spec: ProviderProbeResult.succeeded(spec),
        reload_environment=lambda: pytest.fail("fixture success must not reload environment"),
        rebuild_client=lambda _spec: pytest.fail("fixture success must not rebuild a client"),
    )

    assert result["status"] == "succeeded"


def write_registry_variant(
    tmp_path: Path,
    *,
    mutate: Callable[[dict[str, object]], object],
) -> Path:
    registry_path = REPO_ROOT / "src" / "conference_pipeline" / "stage_owner_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    mutate(registry["scripts"])
    variant_path = tmp_path / "stage-owner-registry.json"
    variant_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return variant_path


def test_runner_preflight_strictly_binds_success_to_the_cvpr_owner_contract(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "owner-contract-success"
    registry = load_stage_script_registry()
    owner = registry["scripts"]["cvpr2026.production_runner"]

    result = run_successful_fixture_preflight(runner, workspace, run_id)

    receipt = json.loads(
        (workspace / "storage" / "workflow-runs" / run_id / "stage-01-runner-preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "succeeded"
    assert receipt["script_registry_sha256"] == registry_sha256()
    assert receipt["required_script_ids"] == ["cvpr2026.production_runner"]
    assert receipt["owner_execution_evidence"]["cvpr2026.production_runner"] != "cvpr2026.production_runner"
    assert receipt["owner_execution_evidence"]["cvpr2026.production_runner"]
    assert receipt["owner_registration"] == {
        "path": owner["path"],
        "scope": owner["scope"],
        "existence": owner["existence"],
        "status": owner["status"],
    }
    assert receipt["stage_contract"]["status"] == "pass"
    assert receipt["stage_contract"]["required_script_ids"] == ["cvpr2026.production_runner"]
    # The persisted receipt -- not only the copy the contract verified --
    # carries lineage and an output digest, so stage-status reads it fresh.
    assert receipt["output_digest"] and receipt["lineage"]["lineage_digest"]
    assert (
        stage_contracts.verify_stage_contract(
            workspace=workspace,
            run_id=run_id,
            stage_id="runner-preflight",
            receipt=receipt,
            mode="live",
            required_owner_ids=("cvpr2026.production_runner",),
            require_lineage=True,
        ).status
        == "pass"
    )
    from conference_pipeline.receipt_lineage import stage_lineage_report

    status = stage_lineage_report(workspace, run_id, layout="production")
    assert status["stages"]["runner-preflight"]["state"] == "fresh", status["stages"]["runner-preflight"]
    remediation_markdown = (
        workspace / "storage" / "workflow-runs" / run_id / "runner-preflight-remediation.md"
    ).read_text(encoding="utf-8")
    assert remediation_markdown.endswith("\n")
    assert not remediation_markdown.endswith("\n\n")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda scripts: scripts.pop("cvpr2026.production_runner"),
        lambda scripts: scripts["cvpr2026.production_runner"].update(
            {"path": "scripts/run_iclr2026_production.py"}
        ),
        lambda scripts: scripts["cvpr2026.production_runner"].update({"scope": "verification"}),
        lambda scripts: scripts["cvpr2026.production_runner"].update({"existence": "allowed"}),
        lambda scripts: scripts["cvpr2026.production_runner"].update({"status": "allowed"}),
    ],
    ids=("missing", "path", "scope", "existence", "status"),
)
def test_runner_preflight_blocks_malformed_cvpr_owner_even_with_receipt_evidence(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "owner-contract-blocked"
    variant_path = write_registry_variant(tmp_path, mutate=mutation)

    result = run_successful_fixture_preflight(
        runner,
        workspace,
        run_id,
        stage_registry_path=variant_path,
    )

    run_path = workspace / "storage" / "workflow-runs" / run_id
    receipt = json.loads((run_path / "stage-01-runner-preflight.json").read_text(encoding="utf-8"))
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((run_path / "runner-preflight-remediation.json").read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert receipt["status"] == "blocked"
    assert summary["status"] == "blocked"
    assert summary["halt_status"] == "halted"
    assert receipt["required_script_ids"] == ["cvpr2026.production_runner"]
    assert receipt["owner_execution_evidence"]["cvpr2026.production_runner"]
    assert any(item["code"] == "cvpr_runner_owner_binding_invalid" for item in report["blockers"])
    assert "repair the CVPR runner registration/evidence" in result["console"]


@pytest.mark.parametrize(
    "evidence",
    [
        {},
        {"cvpr2026.production_runner": "cvpr2026.production_runner"},
        {"cvpr2026.production_runner": ""},
    ],
    ids=("omitted", "owner-id-echo", "empty"),
)
def test_runner_preflight_blocks_missing_or_echoed_cvpr_execution_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence: dict[str, object],
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "owner-evidence-blocked"
    monkeypatch.setattr(runner, "_owner_execution_evidence", lambda **_kwargs: evidence)

    result = run_successful_fixture_preflight(runner, workspace, run_id)

    run_path = workspace / "storage" / "workflow-runs" / run_id
    receipt = json.loads((run_path / "stage-01-runner-preflight.json").read_text(encoding="utf-8"))
    report = json.loads((run_path / "runner-preflight-remediation.json").read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert receipt["status"] == "blocked"
    assert receipt["stage_contract"]["status"] == "blocked"
    assert any(item["code"] == "cvpr_runner_owner_binding_invalid" for item in report["blockers"])
    assert "repair the CVPR runner registration/evidence" in result["console"]


def test_runner_preflight_blocks_invalid_authority_before_env_or_probe_work(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    events: list[str] = []
    adapter = FakeProviderAdapter([])
    run_path = workspace / "storage" / "workflow-runs" / "missing-authority"

    result = runner.run_runner_preflight(
        workspace=workspace,
        run_id="missing-authority",
        prepared_estimate_path=run_path / "missing-estimate.json",
        approval_path=run_path / "missing-approval.json",
        capacity_estimate_path=run_path / "missing-capacity.json",
        credential_values=fixture_credentials(),
        probe=lambda spec: adapter.probe(spec),
        reload_environment=lambda: events.append("reload"),
        rebuild_client=lambda spec: events.append(f"rebuild:{spec.provider}"),
        available_bytes=100_000,
    )

    assert result["status"] == "blocked"
    assert adapter.calls == 0
    assert events == []
    receipt = json.loads((run_path / "stage-01-runner-preflight.json").read_text(encoding="utf-8"))
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((run_path / "runner-preflight-remediation.json").read_text(encoding="utf-8"))
    blockers = list((run_path / "blocker-receipts").rglob("*.json"))
    serialized = json.dumps({"receipt": receipt, "summary": summary, "report": report, "console": result["console"]})
    assert receipt["status"] == "blocked"
    assert summary["halt_status"] == "halted"
    assert blockers
    assert report["blockers"]
    assert "reprepare" in serialized
    assert SECRET_SENTINEL not in serialized


def test_runner_preflight_uses_one_retry_and_persists_only_safe_fake_probe_metadata(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "fixture-preflight"
    prepared = runner.prepare_preflight_estimate(workspace, run_id)
    approval_path = write_immutable_approval(runner, workspace, run_id, prepared)
    capacity_path = write_capacity_estimate(workspace, run_id, prepared)
    retry_spec = next(spec for spec in PLANNED_PROVIDER_PROBES if spec.provider == "openrouter")
    outcomes = {
        spec.provider: [ProviderProbeResult.succeeded(spec)]
        for spec in PLANNED_PROVIDER_PROBES
    }
    outcomes[retry_spec.provider] = [
        ProviderProbeResult.failed(retry_spec, error_type=SECRET_SENTINEL, retryable=True),
        ProviderProbeResult.succeeded(retry_spec),
    ]
    calls: list[tuple[str, str]] = []
    events: list[str] = []

    def probe(spec: ProviderProbeSpec) -> ProviderProbeResult:
        calls.append((spec.provider, spec.operation))
        return outcomes[spec.provider].pop(0)

    result = runner.run_runner_preflight(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
        credential_values=fixture_credentials(),
        probe=probe,
        reload_environment=lambda: events.append("reload"),
        rebuild_client=lambda spec: events.append(f"rebuild:{spec.provider}"),
        available_bytes=1_100,
    )

    receipt_path = workspace / "storage" / "workflow-runs" / run_id / "stage-01-runner-preflight.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    serialized = json.dumps({"receipt": receipt, "result": result})
    assert result["status"] == "succeeded"
    assert calls.count(("openrouter", "GET /api/v1/models")) == 2
    assert len(calls) == len(PLANNED_PROVIDER_PROBES) + 1
    assert events == ["reload", "rebuild:openrouter"]
    assert all({"provider", "operation", "classification", "status", "attempt_count"} <= set(item) for item in receipt["checks"]["provider_probes"])
    assert any(item["provider"] == "openrouter" and item["attempt_count"] == 2 for item in receipt["checks"]["provider_probes"])
    assert "OPENROUTER_API_KEY" in serialized
    assert SECRET_SENTINEL not in serialized


def test_runner_preflight_does_not_retry_nonretryable_fake_probe(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "nonretryable-preflight"
    prepared = runner.prepare_preflight_estimate(workspace, run_id)
    approval_path = write_immutable_approval(runner, workspace, run_id, prepared)
    capacity_path = write_capacity_estimate(workspace, run_id, prepared)
    failed_spec = next(spec for spec in PLANNED_PROVIDER_PROBES if spec.provider == "tavily")
    calls: list[str] = []

    def probe(spec: ProviderProbeSpec) -> ProviderProbeResult:
        calls.append(spec.provider)
        if spec == failed_spec:
            return ProviderProbeResult.failed(spec, error_type="forbidden", retryable=False)
        return ProviderProbeResult.succeeded(spec)

    result = runner.run_runner_preflight(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
        credential_values=fixture_credentials(),
        probe=probe,
        reload_environment=lambda: pytest.fail("nonretryable probe must not reload environment"),
        rebuild_client=lambda spec: pytest.fail("nonretryable probe must not rebuild client"),
        available_bytes=1_100,
    )

    assert result["status"] == "blocked"
    assert calls.count("tavily") == 1
    assert "probe_failed" in result["console"]


def test_scoped_volatile_preflight_preserves_full_entry_evidence_and_probes_only_selected_provider(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "scoped-volatile-preflight"
    prepared = runner.prepare_preflight_estimate(workspace, run_id)
    approval_path = write_immutable_approval(runner, workspace, run_id, prepared)
    capacity_path = write_capacity_estimate(workspace, run_id, prepared)
    full_calls: list[tuple[str, str]] = []

    full = runner.run_runner_preflight(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
        credential_values=fixture_credentials(),
        probe=lambda spec: full_calls.append((spec.provider, spec.stage)) or ProviderProbeResult.succeeded(spec),
        reload_environment=lambda: pytest.fail("fixture must not reload environment"),
        rebuild_client=lambda _spec: pytest.fail("fixture must not rebuild a client"),
        available_bytes=1_100,
    )
    canonical_path = workspace / "storage" / "workflow-runs" / run_id / "stage-01-runner-preflight.json"
    canonical_bytes = canonical_path.read_bytes()
    scoped_calls: list[tuple[str, str]] = []

    scoped = runner.run_runner_preflight(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
        credential_values=fixture_credentials(),
        probe=lambda spec: scoped_calls.append((spec.provider, spec.stage)) or (
            ProviderProbeResult.failed(spec, error_type="tavily_rate_limited")
            if spec.provider == "tavily"
            else ProviderProbeResult.succeeded(spec)
        ),
        reload_environment=lambda: pytest.fail("fixture must not reload environment"),
        rebuild_client=lambda _spec: pytest.fail("fixture must not rebuild a client"),
        available_bytes=1_100,
        provider_probe_scope=("openai", "source-discovery"),
    )

    assert full["status"] == "succeeded"
    assert {provider for provider, _stage in full_calls} == {spec.provider for spec in PLANNED_PROVIDER_PROBES}
    assert scoped["status"] == "succeeded"
    assert scoped_calls == [("openai", "source-discovery")]
    assert canonical_path.read_bytes() == canonical_bytes
    scoped_receipts = [
        path
        for path in (canonical_path.parent / "volatile-preflight").glob("source-discovery-openai.*.json")
        if not path.name.endswith(".remediation.json")
    ]
    assert len(scoped_receipts) == 1
    assert json.loads(scoped_receipts[0].read_text(encoding="utf-8"))["checks"]["provider_probe_scope"] == {
        "provider": "openai",
        "stage": "source-discovery",
    }


def test_scoped_volatile_preflight_blocks_selected_provider_before_paid_owner_dispatch(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "scoped-provider-blocked"
    prepared = runner.prepare_preflight_estimate(workspace, run_id)
    approval_path = write_immutable_approval(runner, workspace, run_id, prepared)
    capacity_path = write_capacity_estimate(workspace, run_id, prepared)
    write_candidate_frontier_inventory(workspace, run_id, pair_keys=("program:one|paper:one",))
    authority = runner.prepare_source_discovery_frontier_authority(workspace, run_id)
    authority_path = workspace / "storage" / "workflow-runs" / run_id / "source-discovery-frontier-authority.json"
    probe_calls: list[str] = []
    owner_calls: list[str] = []

    scoped = runner.run_runner_preflight(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
        credential_values=fixture_credentials(),
        probe=lambda spec: probe_calls.append(spec.provider) or ProviderProbeResult.failed(
            spec, error_type="rate_limited", retryable=False
        ),
        reload_environment=lambda: pytest.fail("non-retryable probe must not reload environment"),
        rebuild_client=lambda _spec: pytest.fail("non-retryable probe must not rebuild a client"),
        available_bytes=1_100,
        provider_probe_scope=("openai", "source-discovery"),
    )

    result = runner.dispatch_paid_call(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=approval_path,
        frontier_authority_path=authority_path,
        provider="openai",
        stage="source-discovery",
        projected_cost_usd=0.0,
        spent_total_usd=0.0,
        spent_by_provider_stage={},
        fresh_volatile_preflight=lambda: {
            "status": scoped["status"],
            "frontier_authority_digest": authority["authority_digest"],
        },
        owner=lambda: owner_calls.append("paid") or 0.0,
        task_id="source_discovery.cross_source_join",
    )

    assert probe_calls == ["openai"]
    assert scoped["status"] == "blocked"
    assert result["authorized"] is False
    assert owner_calls == []


def test_paid_dispatch_rejects_exact_allocation_overage_before_owner(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "paid-dispatch"
    prepared = runner.prepare_preflight_estimate(workspace, run_id)
    approval_path = write_immutable_approval(runner, workspace, run_id, prepared)
    owner_calls: list[str] = []

    result = runner.dispatch_paid_call(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=approval_path,
        provider="openrouter",
        stage="exhaustive-topic-extraction",
        projected_cost_usd=1_000_000.0,
        spent_total_usd=0.0,
        spent_by_provider_stage={},
        owner=lambda: owner_calls.append("called") or 0.0,
    )

    assert result["authorized"] is False
    assert owner_calls == []
    assert any(item["code"] == "provider_stage_allocation_exceeded" for item in result["blockers"])


def _r5_stage5_adapter_result(runner: object, workspace: Path, *, reused: bool = False) -> dict[str, object]:
    """Write a minimal local adapter result for the Stage 5 runner seam."""

    output_path = workspace / "storage/transcription-ocr/cvpr2026/inventory.json"
    input_digests = {
        "stage4_receipt_sha256": runner.R5_STAGE4_RECEIPT_SHA256,
        "event_media_file_sha256": runner.R5_STAGE4_INVENTORY_FILE_SHA256,
        "event_media_canonical_sha256": runner.R5_STAGE4_INVENTORY_CANONICAL_DIGEST,
        "r4_overlay_sha256": runner.R5_STAGE4_R4_OVERLAY_DIGEST,
    }
    # The adapter result must be internally consistent with the synthetic r5
    # event-media inventory the fixture wrote: the strict contract now derives
    # its expected accounting from that digest-bound inventory instead of
    # pinning CVPR 2026's July counts as literals (which is the only reason a
    # 1-candidate inventory paired with a 350-transcript result ever passed).
    event_media_path = workspace / "storage/download-corpus/event-media/event_media_inventory.json"
    if event_media_path.is_file():
        event_media_inventory = json.loads(event_media_path.read_text(encoding="utf-8"))
    else:
        # Tests that stub verify_stage_contract entirely never write an
        # event-media inventory; give them the historical r5-shaped population
        # so their receipt-shape assertions stay meaningful.
        event_media_inventory = {
            "counts": {"terminal_event_route_count": 5_258},
            "caption_candidates": [
                {"video_id": f"fixture-caption-{index:04d}", "caption_status": "youtube_caption_available"}
                for index in range(350)
            ]
            + [
                {"video_id": f"fixture-unavailable-{index:04d}", "caption_status": "youtube_caption_unavailable"}
                for index in range(2_002)
            ],
        }
    candidates = [
        candidate
        for candidate in event_media_inventory.get("caption_candidates", [])
        if isinstance(candidate, dict)
    ]
    caption_transcripts = [
        {
            "source_identity": candidate["video_id"],
            "source_class": "caption_transcript",
            "outcome": "caption_used_no_asr",
            "video_id": candidate["video_id"],
            "language": "en",
            "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "fixture"}],
            "caption_sha256": hashlib.sha256(f"caption-{candidate['video_id']}".encode()).hexdigest(),
            "output_sha256": hashlib.sha256(f"output-{candidate['video_id']}".encode()).hexdigest(),
        }
        for candidate in candidates
        if candidate.get("caption_status") == "youtube_caption_available"
    ]
    caption_blockers = [
        {
            "source_identity": candidate["video_id"],
            "source_class": "caption_blocker",
            "video_id": candidate["video_id"],
            "code": "youtube_caption_unavailable",
            "asr_eligible": False,
        }
        for candidate in candidates
        if candidate.get("caption_status") == "youtube_caption_unavailable"
    ]
    terminal_route_count = int(event_media_inventory.get("counts", {}).get("terminal_event_route_count") or 0)
    payload: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_inventory",
        "run_id": "cvpr2026-live-production-r5-event-media",
        "processor_identity": "cvpr2026.caption_materializer",
        "processor_version": "1",
        "semantic_class": "caption_first_no_asr",
        "compatibility": {
            "run_id": runner.R5_EVENT_MEDIA_RUN_ID,
            **input_digests,
            "processor_identity": "cvpr2026.caption_materializer",
            "processor_version": "1",
            "semantic_class": "caption_first_no_asr",
        },
        "input_digests": input_digests,
        "counts": {
            "terminal_route_count": terminal_route_count,
            "caption_candidate_count": len(candidates),
            "caption_used_no_asr_count": len(caption_transcripts),
            "youtube_caption_unavailable_count": len(caption_blockers),
            "asr_queue_count": 0,
            "ocr_queue_count": 0,
            "provider_call_count": 0,
        },
        "capacity_reservation": {
            "accepted_caption_input_bytes": max(len(caption_transcripts), 1),
            "accepted_caption_count": len(caption_transcripts),
            "reservation_bytes": max(len(caption_transcripts), 1) * 2,
            "safety_margin_bytes": max(len(caption_transcripts) // 10, 1),
        },
        "caption_transcripts": caption_transcripts,
        "blockers": caption_blockers,
        "workshop_pdf_provenance_table": [],
        "workshop_talk_boundary": {
            "status": "not_claimed",
            "reason": "no receipt-backed workshop talk media",
        },
        "asr_queue": [],
        "ocr_queue": [],
        "media_recovery_queue": [],
        "provider_calls": [],
        "resume": {"reused_output_digest": "fixture-reused" if reused else None},
    }
    payload["inventory_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    runner.atomic_write_json(output_path, payload)
    return payload


def _write_runner_caption_sidecar(workspace: Path, video_id: str) -> dict[str, object]:
    """Write one declared local caption sidecar for the unmocked-adapter seam."""

    path = (
        workspace
        / "storage"
        / "download-corpus"
        / "event-media"
        / "captions"
        / f"{video_id}.en.vtt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\ncaption for the local retry fixture\n",
        encoding="utf-8",
    )
    return {
        "local_path": path.relative_to(workspace).as_posix(),
        "mime": "text/vtt",
        "byte_count": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "transcript_source": "youtube_caption_manual",
        "language": "en",
    }


def _write_runner_published_r5_handoff(
    runner: object, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Create a minimal complete published-shape handoff for the real adapter."""

    categories = (
        ("poster", "poster", 5_023),
        ("invited-talk", "keynote", 3),
        ("tutorial", "tutorial", 19),
        ("workshop", "workshop", 153),
        ("session", "oral", 24),
        ("session", "session", 19),
        ("events", "main_event", 17),
    )
    events: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}
    index = 0
    poster_index = 0
    video_id = "v0000000001"
    for route_family, category, count in categories:
        category_counts[category] = count
        for item_index in range(1, count + 1):
            index += 1
            is_direct = index == 1
            if route_family == "session":
                title = "Oral Session" if category == "oral" else "Session"
            else:
                title = f"Fixture {category} {item_index}"
            origin = (
                "r4_registration_media"
                if route_family == "poster" and poster_index < 2_565
                else "r5_authenticated_event_page"
            )
            if route_family == "poster":
                poster_index += 1
            events.append(
                {
                    "source_record_id": f"fixture-r5-{index}",
                    "source_url": f"https://cvpr.thecvf.com/virtual/2026/{route_family}/{index}",
                    "route_family": route_family,
                    "event_category": category,
                    "title": title,
                    "page_status": "direct_youtube_recovered" if is_direct else "official_video_not_published",
                    "outcome_origin": origin,
                    "video_ids": [video_id] if is_direct else [],
                    "audio_downloaded": False,
                    "video_downloaded": False,
                }
            )
    caption = _write_runner_caption_sidecar(workspace, video_id)
    inventory: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_event_media_inventory",
        "run_id": runner.R5_EVENT_MEDIA_RUN_ID,
        "source_lineage": {
            "source_inventory_sha256": "a" * 64,
            "r4_registration_media_overlay_sha256": "b" * 64,
        },
        "events": events,
        "caption_candidates": [
            {
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "source_event_urls": [events[0]["source_url"]],
                "provenance": ["r5_official_event_page"],
                "caption_status": "youtube_caption_available",
                "caption": caption,
                "audio_downloaded": False,
                "video_downloaded": False,
            }
        ],
        "coverage": {
            category: {
                "event_route_count": count,
                "terminal_route_count": count,
                "direct_youtube_route_count": 1 if category == "poster" else 0,
                "status": "complete",
            }
            for _route_family, category, count in categories
        },
        "counts": {
            "event_route_count": 5_258,
            "event_categories": category_counts,
            "oral_session_route_count": 24,
            "terminal_event_route_count": 5_258,
            "direct_youtube_route_count": 1,
            "r4_reused_poster_route_count": 2_565,
            "caption_candidate_count": 1,
        },
        "boundaries": {
            "audio_downloaded": False,
            "video_downloaded": False,
            "asr_dispatched": False,
            "ocr_dispatched": False,
            "provider_calls_made": 0,
        },
    }
    inventory["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(inventory)
    inventory_path = workspace / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    inventory_file_sha256 = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    inventory_canonical_sha256 = canonical_event_media_inventory_digest(inventory)
    stage4_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / runner.R5_EVENT_MEDIA_RUN_ID
        / "stage-04-download-corpus.json"
    )
    stage4_path.parent.mkdir(parents=True, exist_ok=True)
    stage4_path.write_text(
        json.dumps(
            {
                "run_id": runner.R5_EVENT_MEDIA_RUN_ID,
                "stage_id": "download-corpus",
                "stage_order": 4,
                "mode": "live",
                "status": "succeeded",
                "event_media_inventory": {
                    "path": inventory_path.relative_to(workspace).as_posix(),
                    "inventory_sha256": inventory_canonical_sha256,
                },
                "checks": {
                    "event_media_inventory_path": inventory_path.relative_to(workspace).as_posix(),
                    "event_media_inventory_sha256": inventory_file_sha256,
                    "event_media_inventory_canonical_sha256": inventory_canonical_sha256,
                    "r4_registration_media_overlay_sha256": "b" * 64,
                },
                "stage_contract": {"status": "pass", "stage_id": "download-corpus"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    stage4_sha256 = hashlib.sha256(stage4_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        cvpr_transcription_ocr_adapter, "R5_EVENT_MEDIA_RECEIPT_SHA256", stage4_sha256
    )
    monkeypatch.setattr(
        cvpr_transcription_ocr_adapter, "R5_EVENT_MEDIA_FILE_SHA256", inventory_file_sha256
    )
    monkeypatch.setattr(
        cvpr_transcription_ocr_adapter, "R5_EVENT_MEDIA_CANONICAL_DIGEST", inventory_canonical_sha256
    )
    monkeypatch.setattr(cvpr_transcription_ocr_adapter, "R4_OVERLAY_DIGEST", "b" * 64)
    monkeypatch.setattr(runner, "R5_STAGE4_RECEIPT_SHA256", stage4_sha256)
    monkeypatch.setattr(runner, "R5_STAGE4_INVENTORY_FILE_SHA256", inventory_file_sha256)
    monkeypatch.setattr(runner, "R5_STAGE4_INVENTORY_CANONICAL_DIGEST", inventory_canonical_sha256)
    monkeypatch.setattr(runner, "R5_STAGE4_R4_OVERLAY_DIGEST", "b" * 64)
    return stage4_path


def _write_runner_historical_stage5_block(
    runner: object, workspace: Path, stage4_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Create the one sanitized 05-11-shaped record eligible for local repair."""

    receipt_path = stage4_path.with_name("stage-05-transcription-ocr.json")
    receipt = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_stage_receipt",
        "run_id": runner.R5_EVENT_MEDIA_RUN_ID,
        "stage_id": "transcription-ocr",
        "stage_order": 5,
        "mode": "live",
        "status": "blocked",
        "provider_calls_made": 0,
        "downloads_made": 0,
        "secrets_copied": False,
        "checks": {
            "stage_04_run_id": runner.R5_EVENT_MEDIA_RUN_ID,
            "stage_04_receipt_path": stage4_path.relative_to(workspace).as_posix(),
            "stage_04_receipt_sha256": runner.R5_STAGE4_RECEIPT_SHA256,
            "event_media_inventory_path": "storage/download-corpus/event-media/event_media_inventory.json",
            "event_media_inventory_sha256": runner.R5_STAGE4_INVENTORY_FILE_SHA256,
            "event_media_inventory_canonical_sha256": runner.R5_STAGE4_INVENTORY_CANONICAL_DIGEST,
            "r4_registration_media_overlay_sha256": runner.R5_STAGE4_R4_OVERLAY_DIGEST,
            "transcription_ocr_cvpr2026_execution_status": "blocked",
        },
        "blockers": [
            {"code": "transcription_ocr_adapter_result_invalid"},
            {
                "code": "transcription_ocr_local_admission_failed",
                "reason": "CvprTranscriptionOcrAdmissionError",
            },
        ],
        "stage_contract": {
            "status": "blocked",
            "blockers": [
                {"code": "transcription_ocr_adapter_result_invalid"},
                {
                    "code": "transcription_ocr_local_admission_failed",
                    "reason": "CvprTranscriptionOcrAdmissionError",
                },
            ],
        },
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "R5_TRANSCRIPTION_OCR_HISTORICAL_BLOCKED_RECEIPT_SHA256",
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        raising=False,
    )
    return receipt_path


def _patch_real_stage5_contract_constants(
    runner: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind the real strict verifier to the synthetic local r5 handoff bytes."""

    monkeypatch.setattr(
        stage_contracts,
        "R5_EVENT_MEDIA_RECEIPT_SHA256",
        runner.R5_STAGE4_RECEIPT_SHA256,
    )
    monkeypatch.setattr(
        stage_contracts,
        "R5_EVENT_MEDIA_FILE_SHA256",
        runner.R5_STAGE4_INVENTORY_FILE_SHA256,
    )
    monkeypatch.setattr(
        stage_contracts,
        "R5_EVENT_MEDIA_CANONICAL_DIGEST",
        runner.R5_STAGE4_INVENTORY_CANONICAL_DIGEST,
    )
    monkeypatch.setattr(
        stage_contracts,
        "R4_OVERLAY_DIGEST",
        runner.R5_STAGE4_R4_OVERLAY_DIGEST,
    )


def _write_runner_strict_registry(runner: object, workspace: Path) -> Path:
    """Bind the real contract schema to one synthetic temporary handoff."""

    registry_path = workspace.parent / "stage-owner-registry.json"
    registry = json.loads(
        (REPO_ROOT / "src/conference_pipeline/stage_owner_registry.json").read_text(encoding="utf-8")
    )
    contract = registry["stage_bindings"]["transcription-ocr"]["workspace_bindings"][
        "conferences/cvpr2026"
    ]["runtime_contract"]
    expected_values = {
        "checks.stage_04_receipt_sha256": runner.R5_STAGE4_RECEIPT_SHA256,
        "checks.event_media_inventory_sha256": runner.R5_STAGE4_INVENTORY_FILE_SHA256,
        "checks.event_media_inventory_canonical_sha256": runner.R5_STAGE4_INVENTORY_CANONICAL_DIGEST,
        "checks.r4_registration_media_overlay_sha256": runner.R5_STAGE4_R4_OVERLAY_DIGEST,
    }
    for check in contract["required_checks"]:
        if check.get("path") in expected_values:
            check["equals"] = expected_values[check["path"]]
    registry_path.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")
    return registry_path


def _retry_control_digest_map(
    runner: object, workspace: Path
) -> tuple[dict[str, dict[str, str]], dict[str, Path]]:
    """Create only the three structural retry controls in a temporary workspace."""

    paths = runner._stage5_retry_control_paths(
        workspace=workspace,
        run_id=runner.R5_EVENT_MEDIA_RUN_ID,
    )
    for name, path in paths.items():
        runner.atomic_write_json(path, {"control": name, "run_id": runner.R5_EVENT_MEDIA_RUN_ID})
    return (
        {
            name: {
                "path": path.relative_to(workspace).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
        paths,
    )


def test_cvpr_r5_retry_persists_and_reloads_real_strict_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The durable Stage-5 receipt, not a provisional object, must be strict-pass."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    stage4_path = _write_runner_published_r5_handoff(runner, workspace, monkeypatch)
    historical_path = _write_runner_historical_stage5_block(
        runner, workspace, stage4_path, monkeypatch
    )
    _patch_real_stage5_contract_constants(runner, monkeypatch)
    strict_registry_path = _write_runner_strict_registry(runner, workspace)
    monkeypatch.setattr(
        stage_script_registry,
        "DEFAULT_STAGE_SCRIPT_REGISTRY_PATH",
        strict_registry_path,
    )
    monkeypatch.setattr(
        runner,
        "build_cvpr_r5_transcription_ocr_inventory",
        lambda **_kwargs: _r5_stage5_adapter_result(runner, workspace),
    )
    monkeypatch.setattr(
        runner,
        "run_cvpr_event_media_stage",
        lambda **_kwargs: pytest.fail("Stage 4 is forbidden"),
    )
    monkeypatch.setattr(
        runner,
        "run_source_discovery_stage",
        lambda **_kwargs: pytest.fail("discovery is forbidden"),
    )

    authorization = runner.authorize_cvpr_r5_transcription_retry(
        workspace=workspace,
        run_id=runner.R5_EVENT_MEDIA_RUN_ID,
    )
    outcome = runner.run_cvpr_transcription_ocr_stage(
        workspace=workspace,
        run_id=runner.R5_EVENT_MEDIA_RUN_ID,
        retry_authorization=Path(authorization["authorization_path"]),
    )

    assert outcome["status"] == "succeeded", outcome["receipt"]["stage_contract"]["blockers"]
    persisted_path = Path(outcome["receipt_path"])
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "succeeded"
    assert persisted["stage_contract"]["status"] == "pass"
    assert persisted["checks"]["transcription_ocr_cvpr2026_execution_status"] == "pass"
    controls = persisted["checks"]["retry_control_digests"]
    assert set(controls) == {"archive", "consumed_authorization", "attempt"}
    for _name, control in controls.items():
        path = workspace / control["path"]
        assert set(control) == {"path", "sha256"}
        assert control["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    reloaded = json.loads(persisted_path.read_text(encoding="utf-8"))
    strict = stage_contracts.verify_stage_contract(
        workspace=workspace,
        run_id=runner.R5_EVENT_MEDIA_RUN_ID,
        stage_id="transcription-ocr",
        receipt=reloaded,
        strict_execution=True,
    )
    assert strict.status == "pass", strict.blockers
    assert historical_path.is_file()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda controls, paths: controls.pop("attempt"),
        lambda controls, paths: controls.__setitem__(
            "extra", {"path": "storage/foreign.json", "sha256": "0" * 64}
        ),
        lambda controls, paths: controls["archive"].__setitem__("path", "storage/foreign.json"),
        lambda controls, paths: controls["archive"].__setitem__("sha256", "UPPERCASE"),
        lambda controls, paths: controls["archive"].__setitem__("sha256", "0" * 64),
        lambda controls, paths: paths["attempt"].write_text("tampered\n", encoding="utf-8"),
    ],
    ids=("missing", "extra", "foreign-path", "malformed", "stale-hash", "tampered-bytes"),
)
def test_stage5_writer_blocks_invalid_retry_controls_after_sanitization(
    tmp_path: Path, mutate: object
) -> None:
    """Malformed structural maps cannot survive the local post-sanitization seam."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    controls, paths = _retry_control_digest_map(runner, workspace)
    mutate(controls, paths)  # type: ignore[operator]
    control_bytes_before_write = {name: path.read_bytes() for name, path in paths.items()}
    receipt_path = workspace / "storage/workflow-runs/local-fixture/stage-05-transcription-ocr.json"

    outcome = runner._transcription_ocr_stage_receipt(
        workspace=workspace,
        run_id=runner.R5_EVENT_MEDIA_RUN_ID,
        receipt_path=receipt_path,
        status="succeeded",
        checks={
            "transcription_ocr_cvpr2026_execution_status": "pass",
            "retry_control_digests": controls,
        },
        blockers=[],
        stage_contract={"status": "pass", "blockers": []},
        resume=False,
    )

    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert outcome["status"] == "blocked"
    assert persisted["status"] == "blocked"
    assert persisted["checks"]["transcription_ocr_cvpr2026_execution_status"] == "blocked"
    assert "retry_control_digests" not in persisted["checks"]
    assert persisted["stage_contract"]["status"] == "blocked"
    for name, path in paths.items():
        assert path.read_bytes() == control_bytes_before_write[name]


def test_stage5_writer_resets_execution_status_before_persisting_strict_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A strict failure may never publish a durable execution-status pass."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    stage4_path = _write_runner_published_r5_handoff(runner, workspace, monkeypatch)
    _write_runner_historical_stage5_block(runner, workspace, stage4_path, monkeypatch)

    class FailingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "blocked", "blockers": [{"code": "forced-strict-failure"}]}

    monkeypatch.setattr(runner, "verify_stage_contract", lambda **_kwargs: FailingContract())
    authorization = runner.authorize_cvpr_r5_transcription_retry(
        workspace=workspace,
        run_id=runner.R5_EVENT_MEDIA_RUN_ID,
    )
    outcome = runner.run_cvpr_transcription_ocr_stage(
        workspace=workspace,
        run_id=runner.R5_EVENT_MEDIA_RUN_ID,
        retry_authorization=Path(authorization["authorization_path"]),
    )

    persisted = json.loads(Path(outcome["receipt_path"]).read_text(encoding="utf-8"))
    assert outcome["status"] == "blocked"
    assert persisted["status"] == "blocked"
    assert persisted["stage_contract"]["status"] == "blocked"
    assert persisted["checks"]["transcription_ocr_cvpr2026_execution_status"] == "blocked"


def test_cvpr_r5_repair_retry_uses_real_adapter_and_binds_immutable_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The repaired owner consumes its one authorization before local Stage 5 work."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    stage4_path = _write_runner_published_r5_handoff(runner, workspace, monkeypatch)
    historical_path = _write_runner_historical_stage5_block(
        runner, workspace, stage4_path, monkeypatch
    )
    historical_bytes = historical_path.read_bytes()
    verifier_receipts: list[dict[str, object]] = []

    class PassingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "pass", "accepted": ["fixture retry-control contract"]}

    def fake_verifier(**kwargs: object) -> PassingContract:
        verifier_receipts.append(dict(kwargs["receipt"]))
        return PassingContract()

    monkeypatch.setattr(runner, "verify_stage_contract", fake_verifier)
    monkeypatch.setattr(runner, "run_cvpr_event_media_stage", lambda **_kwargs: pytest.fail("Stage 4 is forbidden"))
    monkeypatch.setattr(runner, "run_source_discovery_stage", lambda **_kwargs: pytest.fail("discovery is forbidden"))
    monkeypatch.setattr(runner, "run_download_corpus_stage", lambda **_kwargs: pytest.fail("acquisition is forbidden"))

    authorization = runner.authorize_cvpr_r5_transcription_retry(
        workspace=workspace,
        run_id=runner.R5_EVENT_MEDIA_RUN_ID,
    )
    authorization_path = Path(authorization["authorization_path"])
    archive_path = historical_path.with_name("stage-05-transcription-ocr.pre-repair-blocked.json")
    attempt_path = historical_path.with_name("stage-05-transcription-ocr.retry-attempt.json")
    assert archive_path.read_bytes() == historical_bytes
    assert authorization_path == historical_path.with_name("stage-05-transcription-ocr.retry-authorization.json")

    outcome = runner.run_cvpr_transcription_ocr_stage(
        workspace=workspace,
        run_id=runner.R5_EVENT_MEDIA_RUN_ID,
        retry_authorization=authorization_path,
    )

    assert outcome["status"] == "succeeded"
    assert historical_path.read_bytes() != historical_bytes
    assert (workspace / "storage/transcription-ocr/cvpr2026/inventory.json").is_file()
    assert len(verifier_receipts) == 1
    candidate = verifier_receipts[0]
    assert candidate["checks"]["stage_04_receipt_sha256"] == hashlib.sha256(stage4_path.read_bytes()).hexdigest()
    retry_digests = candidate["checks"]["retry_control_digests"]
    assert set(retry_digests) == {"archive", "consumed_authorization", "attempt"}
    expected_paths = {
        "archive": archive_path,
        "consumed_authorization": authorization_path,
        "attempt": attempt_path,
    }
    for name, control_path in expected_paths.items():
        assert set(retry_digests[name]) == {"path", "sha256"}
        assert retry_digests[name]["path"] == control_path.relative_to(workspace).as_posix()
        assert retry_digests[name]["sha256"] == hashlib.sha256(control_path.read_bytes()).hexdigest()
    assert json.loads(authorization_path.read_text(encoding="utf-8"))["status"] == "consumed"
    assert json.loads(attempt_path.read_text(encoding="utf-8"))["status"] == "candidate_materialized"

    before_retry = (workspace / "storage/transcription-ocr/cvpr2026/inventory.json").read_bytes()
    with pytest.raises(ValueError, match="retry authorization already consumed"):
        runner.run_cvpr_transcription_ocr_stage(
            workspace=workspace,
            run_id=runner.R5_EVENT_MEDIA_RUN_ID,
            retry_authorization=authorization_path,
        )
    assert (workspace / "storage/transcription-ocr/cvpr2026/inventory.json").read_bytes() == before_retry


def test_cvpr_r5_transcription_owner_is_local_only_and_publishes_only_after_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    adapter_calls: list[Path] = []
    verifier_receipts: list[dict[str, object]] = []

    def fake_adapter(*, workspace: Path, receipt_path: Path, **_kwargs: object) -> dict[str, object]:
        adapter_calls.append(receipt_path)
        return _r5_stage5_adapter_result(runner, workspace)

    class PassingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "pass", "accepted": ["fixture strict contract"]}

    def fake_verifier(**kwargs: object) -> PassingContract:
        verifier_receipts.append(dict(kwargs["receipt"]))
        return PassingContract()

    monkeypatch.setattr(runner, "build_cvpr_r5_transcription_ocr_inventory", fake_adapter)
    monkeypatch.setattr(runner, "verify_stage_contract", fake_verifier)
    monkeypatch.setattr(runner, "run_cvpr_event_media_stage", lambda **_kwargs: pytest.fail("Stage 4 is forbidden"))
    monkeypatch.setattr(runner, "run_source_discovery_stage", lambda **_kwargs: pytest.fail("discovery is forbidden"))
    monkeypatch.setattr(runner, "run_download_corpus_stage", lambda **_kwargs: pytest.fail("acquisition is forbidden"))

    outcome = runner.run_cvpr_transcription_ocr_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r5-event-media",
    )

    receipt_path = workspace / "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-05-transcription-ocr.json"
    assert outcome["status"] == "succeeded"
    assert adapter_calls == [
        workspace / "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-04-download-corpus.json"
    ]
    assert len(verifier_receipts) == 1
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["owner_execution_evidence"]["transcription_ocr.cvpr2026"]["owner_function"] == "run_cvpr_transcription_ocr_stage"
    assert receipt["checks"]["event_route_count"] == 5_258
    assert receipt["checks"]["caption_candidate_count"] == 2_352
    assert receipt["checks"]["caption_used_no_asr_count"] == 350
    assert receipt["checks"]["youtube_caption_unavailable_count"] == 2_002
    assert receipt["checks"]["provider_calls_made"] == 0
    assert receipt["checks"]["secrets_copied"] is False


def test_cvpr_r5_transcription_owner_fail_closes_and_resume_preserves_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--workspace",
                str(workspace),
                "--run-id",
                "cvpr2026-live-production-r4-registration-media",
                "--stage",
                "transcription-ocr",
            ]
        )

    class FailingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "blocked", "blockers": [{"code": "fixture_contract_failure"}]}

    monkeypatch.setattr(
        runner,
        "build_cvpr_r5_transcription_ocr_inventory",
        lambda **_kwargs: _r5_stage5_adapter_result(runner, workspace),
    )
    monkeypatch.setattr(runner, "verify_stage_contract", lambda **_kwargs: FailingContract())
    failed = runner.run_cvpr_transcription_ocr_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r5-event-media",
    )
    canonical_path = workspace / "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-05-transcription-ocr.json"
    assert failed["status"] == "blocked"
    assert json.loads(canonical_path.read_text(encoding="utf-8"))["status"] == "blocked"

    class PassingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "pass"}

    monkeypatch.setattr(runner, "verify_stage_contract", lambda **_kwargs: PassingContract())
    succeeded = runner.run_cvpr_transcription_ocr_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r5-event-media",
        resume=True,
    )
    assert succeeded["status"] == "succeeded"
    attempts = list(canonical_path.parent.glob("stage-05-transcription-ocr.attempt-*.json"))
    assert attempts
    assert json.loads(attempts[0].read_text(encoding="utf-8"))["status"] == "blocked"


def test_runner_preflight_rejects_external_immutable_input_paths_before_evidence(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)

    with pytest.raises(ValueError, match="named run directory"):
        runner.run_runner_preflight(
            workspace=workspace,
            run_id="contained-inputs",
            prepared_estimate_path=tmp_path / "external-estimate.json",
            approval_path=tmp_path / "external-approval.json",
            capacity_estimate_path=tmp_path / "external-capacity.json",
            credential_values=fixture_credentials(),
            probe=lambda spec: pytest.fail("external input path must stop before any probe"),
            available_bytes=1_100,
        )

    assert not (workspace / "storage" / "workflow-runs" / "contained-inputs").exists()


class FakeProviderAdapter:
    """A local fake that records calls without creating a provider client."""

    def __init__(self, outcomes: list[ProviderProbeResult]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def probe(self, spec: ProviderProbeSpec) -> ProviderProbeResult:
        self.calls += 1
        return self.outcomes.pop(0)


def test_complete_credential_inventory_blocks_later_stage_credential() -> None:
    present = {
        "OPENAI_API_KEY": SECRET_SENTINEL,
        "ANTHROPIC_API_KEY": SECRET_SENTINEL,
        "GOOGLE_API_KEY": SECRET_SENTINEL,
        "TAVILY_API_KEY": SECRET_SENTINEL,
        "OPENROUTER_API_KEY": SECRET_SENTINEL,
        "OPENREVIEW_USERNAME": "fixture-user",
    }

    blockers = validate_credential_inventory(PLANNED_PROVIDER_PROBES, present)

    assert {blocker["code"] for blocker in blockers} == {"missing_credential"}
    assert blockers[0]["credential_names"] == ["OPENREVIEW_PASSWORD"]
    assert blockers[0]["affected_provider"] == "openreview"
    assert SECRET_SENTINEL not in json.dumps(blockers)


def test_probe_contract_requires_declared_non_mutating_operation() -> None:
    undeclared = ProviderProbeSpec(
        provider="example",
        stage="example-stage",
        credential_names=("EXAMPLE_API_KEY",),
        operation="POST /v1/create",
        classification="paid",
        outcome_schema=("status",),
    )

    blockers = validate_provider_probe_specs((*PLANNED_PROVIDER_PROBES, undeclared))

    assert any(blocker["code"] == "invalid_probe_classification" for blocker in blockers)
    openrouter = next(spec for spec in PLANNED_PROVIDER_PROBES if spec.provider == "openrouter")
    assert openrouter.operation == "GET /api/v1/models"
    assert openrouter.classification == "metadata"


def test_retryable_probe_is_reloaded_rebuilt_and_retried_once() -> None:
    spec = next(spec for spec in PLANNED_PROVIDER_PROBES if spec.provider == "openrouter")
    adapter = FakeProviderAdapter(
        [
            ProviderProbeResult.failed(spec, error_type="timeout", retryable=True),
            ProviderProbeResult.succeeded(spec),
        ]
    )
    events: list[str] = []

    result = execute_probe_with_single_recovery(
        spec,
        adapter.probe,
        reload_environment=lambda: events.append("reload"),
        rebuild_client=lambda: events.append("rebuild"),
    )

    assert result.status == "succeeded"
    assert result.attempt_count == 2
    assert adapter.calls == 2
    assert events == ["reload", "rebuild"]


def test_storage_estimate_requires_complete_consistent_full_run_input(tmp_path: Path) -> None:
    estimate_path = tmp_path / "storage-estimate.json"
    estimate_path.write_text(
        json.dumps(
            {
                "corpus_scope": "all-discovered-cvpr2026",
                "inventory_digest": "inventory-digest",
                "manifest_digest": "manifest-digest",
                "remaining_families": {"paper_pdf": 400, "media": 600},
                "projected_remaining_bytes": 1000,
                "estimator_version": "test-v1",
                "safety_margin_bytes": 100,
            }
        ),
        encoding="utf-8",
    )

    passing = validate_storage_estimate(estimate_path, available_bytes=1100)
    insufficient = validate_storage_estimate(estimate_path, available_bytes=1099)

    assert passing["ready"] is True
    assert passing["required_bytes"] == 1100
    assert insufficient["ready"] is False
    assert insufficient["blockers"][0]["code"] == "insufficient_storage"


def test_remediation_groups_independent_root_causes_without_secrets() -> None:
    blockers = normalize_root_causes(
        [
            {
                "code": "missing_credential",
                "affected_provider": "openreview",
                "credential_names": ["OPENREVIEW_PASSWORD"],
                "detail": SECRET_SENTINEL,
            },
            {
                "code": "probe_failed",
                "affected_provider": "openrouter",
                "stage": "remote-extraction-failover",
                "error_type": "timeout",
                "detail": "dependent credential symptom",
                "depends_on": "missing_credential",
            },
            {
                "code": "approval_required",
                "affected_stage": "topic-analysis",
            },
            {
                "code": "insufficient_storage",
                "affected_stage": "download-corpus",
            },
        ]
    )

    report = build_remediation_report(blockers)
    serialized = json.dumps({"blockers": blockers, "report": report})

    assert [blocker["code"] for blocker in blockers] == [
        "missing_credential",
        "approval_required",
        "insufficient_storage",
    ]
    assert "credential_environment" in report["markdown"]
    assert "clearance_evidence" in report["json"][0]
    assert SECRET_SENTINEL not in serialized


def approval_for(snapshot: object, *, total_ceiling_usd: float = 13.0) -> ApprovalDeclaration:
    return ApprovalDeclaration(
        run_id="cvpr2026-live-production",
        corpus_scope="all-discovered-cvpr2026",
        paid_stages=("topic-analysis", "remote-extraction-failover"),
        provider_models=FIXTURE_MODELS,
        estimate_digest=snapshot.digest,
        scope_fingerprint=build_approval_scope_fingerprint(
            run_id="cvpr2026-live-production",
            corpus_scope="all-discovered-cvpr2026",
            paid_stages=("topic-analysis", "remote-extraction-failover"),
            provider_models=FIXTURE_MODELS,
            estimate_digest=snapshot.digest,
            total_ceiling_usd=total_ceiling_usd,
            provider_stage_allocations=(
                ProviderStageAllocation(provider="openai", stage="topic-analysis", ceiling_usd=8.0),
                ProviderStageAllocation(
                    provider="openrouter",
                    stage="remote-extraction-failover",
                    ceiling_usd=4.0,
                ),
            ),
        ),
        total_ceiling_usd=total_ceiling_usd,
        provider_stage_allocations=(
            ProviderStageAllocation(provider="openai", stage="topic-analysis", ceiling_usd=8.0),
            ProviderStageAllocation(
                provider="openrouter",
                stage="remote-extraction-failover",
                ceiling_usd=4.0,
            ),
        ),
    )


def test_named_approval_requires_exact_complete_scope_and_allocations() -> None:
    current_models = FIXTURE_MODELS
    snapshot = build_complete_run_estimate_snapshot(
        workload={"paper_count": 10},
        estimate={"pricing_digest": "pricing", "task_spec_digest": "tasks"},
        corpus_scope="all-discovered-cvpr2026",
        paid_stages=("topic-analysis", "remote-extraction-failover"),
        provider_models=current_models,
        provider_stage_costs={
            "openai:topic-analysis": 7.0,
            "openrouter:remote-extraction-failover": 3.0,
        },
        approved_headroom_usd=1.0,
    )
    approval = approval_for(snapshot)

    assert validate_approval_scope(approval, snapshot) == []

    changed = replace(
        approval,
        run_id="different-run",
        provider_stage_allocations=approval.provider_stage_allocations[:-1],
    )
    codes = {blocker["code"] for blocker in validate_approval_scope(changed, snapshot)}
    assert {"approval_scope_mismatch", "missing_provider_stage_allocation"} <= codes

    model_drift = replace(
        approval,
        provider_models=(ProviderModelChoice(provider="openai", model="changed-fixture-model"),),
    )
    low_headroom = replace(approval, total_ceiling_usd=10.5)
    assert validate_approval_scope(model_drift, snapshot)
    assert any(
        blocker["code"] == "run_ceiling_exceeded"
        for blocker in validate_approval_scope(low_headroom, snapshot)
    )


def test_estimate_digest_ignores_generated_at_but_binds_full_scope() -> None:
    common = {
        "workload": {"paper_count": 10, "bytes": 500},
        "estimate": {"pricing_digest": "pricing", "task_spec_digest": "tasks"},
        "corpus_scope": "all-discovered-cvpr2026",
        "paid_stages": ("topic-analysis",),
        "provider_stage_costs": {"openai:topic-analysis": 7.0},
    }

    first = build_complete_run_estimate_snapshot(**common, generated_at="2026-07-16T00:00:00Z")
    second = build_complete_run_estimate_snapshot(**common, generated_at="2026-07-16T01:00:00Z")

    assert first.scope == "full"
    assert first.digest == second.digest
    assert first.as_dict()["generated_at"] != second.as_dict()["generated_at"]

    report_a = build_complete_run_estimate_snapshot(
        **{**common, "estimate": {**common["estimate"], "generated_at": "2026-07-16T00:00:00Z"}}
    )
    report_b = build_complete_run_estimate_snapshot(
        **{**common, "estimate": {**common["estimate"], "generated_at": "2026-07-16T01:00:00Z"}}
    )
    assert report_a.digest == report_b.digest


def test_prepared_estimate_uses_canonical_full_scope_collectors(tmp_path: Path) -> None:
    calls: list[tuple[object, ...]] = []

    snapshot = prepare_preflight_estimate(
        tmp_path,
        corpus_scope="all-discovered-cvpr2026",
        paid_stages=("topic-analysis",),
        provider_stage_costs={"openai:topic-analysis": 7.0},
        workload_collector=lambda workspace: calls.append(("collect", workspace)) or {"papers": 10},
        estimator=lambda workspace, *, scope, mode: calls.append(("estimate", workspace, scope, mode))
        or {"pricing_digest": "pricing", "task_spec_digest": "tasks"},
    )

    assert calls == [
        ("collect", tmp_path),
        ("estimate", tmp_path, "full", "estimate"),
    ]
    assert snapshot.scope == "full"


def test_call_authorization_fails_closed_for_unknown_cost_or_any_cap_overage() -> None:
    snapshot = build_complete_run_estimate_snapshot(
        workload={"paper_count": 10},
        estimate={"pricing_digest": "pricing", "task_spec_digest": "tasks"},
        corpus_scope="all-discovered-cvpr2026",
        paid_stages=("topic-analysis", "remote-extraction-failover"),
        provider_models=FIXTURE_MODELS,
        provider_stage_costs={
            "openai:topic-analysis": 7.0,
            "openrouter:remote-extraction-failover": 3.0,
        },
    )
    approval = approval_for(snapshot)

    unknown = authorize_prospective_call(
        approval,
        snapshot,
        provider="openai",
        stage="topic-analysis",
        projected_cost_usd=None,
        spent_total_usd=0.0,
        spent_by_provider_stage={},
    )
    nonfinite = authorize_prospective_call(
        approval,
        snapshot,
        provider="openai",
        stage="topic-analysis",
        projected_cost_usd=float("nan"),
        spent_total_usd=0.0,
        spent_by_provider_stage={},
    )
    provider_over = authorize_prospective_call(
        approval,
        snapshot,
        provider="openrouter",
        stage="remote-extraction-failover",
        projected_cost_usd=1.0,
        spent_total_usd=0.0,
        spent_by_provider_stage={"openrouter:remote-extraction-failover": 3.5},
    )
    total_over = authorize_prospective_call(
        approval,
        snapshot,
        provider="openai",
        stage="topic-analysis",
        projected_cost_usd=1.0,
        spent_total_usd=12.5,
        spent_by_provider_stage={"openai:topic-analysis": 0.0},
    )

    assert unknown["authorized"] is False
    assert unknown["blockers"][0]["code"] == "unpriced_call"
    assert nonfinite["blockers"][0]["code"] == "unpriced_call"
    assert provider_over["blockers"][0]["code"] == "provider_stage_allocation_exceeded"
    assert total_over["blockers"][0]["code"] == "run_ceiling_exceeded"


def readiness_fingerprint(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "runner_implementation": "runner-source-digest",
        "workspace_manifest": "manifest-digest",
        "corpus_scope": "all-discovered-cvpr2026",
        "provider_model_scope": {"openai": "fixture-analysis-model"},
        "approval_scope": "approval-digest",
        "disk_estimate_input": "storage-estimate-digest",
        "workspace_identity": "conferences/cvpr2026",
        "command": "runner-preflight",
    }
    fields.update(overrides)
    return build_static_readiness_fingerprint(**fields)


def test_static_fingerprint_tracks_all_nonsecret_readiness_inputs() -> None:
    baseline = readiness_fingerprint()

    assert baseline["credential_value_marker_persisted"] is False
    assert SECRET_SENTINEL not in json.dumps(baseline)
    for field, changed in {
        "runner_implementation": "changed-runner",
        "workspace_manifest": "changed-manifest",
        "corpus_scope": "different-corpus",
        "provider_model_scope": {"openai": "different-fixture-model"},
        "approval_scope": "changed-approval",
        "disk_estimate_input": "changed-storage",
        "workspace_identity": "other-workspace",
        "command": "source-discovery",
    }.items():
        assert readiness_fingerprint(**{field: changed})["digest"] != baseline["digest"]
    with pytest.raises(ValueError, match="credential"):
        readiness_fingerprint(provider_model_scope={"credential": SECRET_SENTINEL})


def test_resume_requires_fresh_preflight_and_volatile_rechecks() -> None:
    fingerprint = readiness_fingerprint(command="topic-analysis")
    prior = {
        "receipt_id": "old-succeeded-receipt",
        "status": "succeeded",
        "static_fingerprint": fingerprint,
    }

    runner_preflight = evaluate_resume_compatibility(
        prior,
        readiness_fingerprint(command="runner-preflight"),
        stage_id="runner-preflight",
    )
    later_stage = evaluate_resume_compatibility(prior, fingerprint, stage_id="topic-analysis")

    assert runner_preflight["eligible_for_skip"] is False
    assert runner_preflight["reason"] == "runner_preflight_requires_fresh_full_check"
    assert later_stage["eligible_for_skip"] is True
    assert later_stage["required_volatile_rechecks"] == [
        "auth",
        "rate_limit",
        "approval_headroom",
        "storage_capacity",
    ]


def test_incompatible_history_is_immutable_and_new_attempt_links_superseded_receipt() -> None:
    prior = {
        "receipt_id": "immutable-old-receipt",
        "status": "succeeded",
        "static_fingerprint": readiness_fingerprint(command="topic-analysis"),
    }
    original = json.loads(json.dumps(prior))

    result = evaluate_resume_compatibility(
        prior,
        readiness_fingerprint(command="changed-command"),
        stage_id="topic-analysis",
        new_attempt_id="new-attempt",
    )

    assert result["eligible_for_skip"] is False
    assert result["reason"] == "static_fingerprint_mismatch"
    assert result["new_attempt"] == {
        "attempt_id": "new-attempt",
        "supersedes_receipt_id": "immutable-old-receipt",
        "reason": "static_fingerprint_mismatch",
    }
    assert prior == original


def test_acknowledgement_is_history_not_clearance_without_new_passing_recheck() -> None:
    acknowledgement = {"acknowledgement_id": "operator-saw-blocker", "blocker_code": "probe_failed"}

    blocked = validate_acknowledgement_clearance(
        acknowledgement,
        fresh_recheck={"status": "blocked", "check_id": "probe"},
    )
    cleared = validate_acknowledgement_clearance(
        acknowledgement,
        fresh_recheck={"status": "succeeded", "check_id": "probe"},
    )

    assert blocked["cleared"] is False
    assert blocked["acknowledgement_id"] == "operator-saw-blocker"
    assert cleared["cleared"] is True


def test_automatic_authorization_is_limited_to_complete_finite_under_threshold_scope() -> None:
    snapshot = build_complete_run_estimate_snapshot(
        workload={"paper_count": 10},
        estimate={"pricing_digest": "pricing", "task_spec_digest": "tasks"},
        corpus_scope="all-discovered-cvpr2026",
        paid_stages=("topic-analysis", "remote-extraction-failover"),
        provider_models=FIXTURE_MODELS,
        provider_stage_costs={
            "openai:topic-analysis": AUTOMATIC_AUTHORIZATION_THRESHOLD_USD - 100.0,
            "openrouter:remote-extraction-failover": 100.0,
        },
    )

    automatic = derive_automatic_approval(snapshot, run_id="under-threshold")

    assert automatic is not None
    assert automatic.total_ceiling_usd == AUTOMATIC_AUTHORIZATION_THRESHOLD_USD
    assert validate_approval_scope(automatic, snapshot) == []
    assert derive_automatic_approval(
        replace(
            snapshot,
            provider_stage_costs=(
                ("openai:topic-analysis", AUTOMATIC_AUTHORIZATION_THRESHOLD_USD + 0.01),
            ),
        ),
        run_id="over-threshold",
    ) is None
    assert derive_automatic_approval(
        replace(snapshot, provider_stage_costs=(("openai:topic-analysis", float("nan")),)),
        run_id="nonfinite",
    ) is None
    assert derive_automatic_approval(
        replace(snapshot, provider_stage_costs=()), run_id="incomplete"
    ) is None


def test_runner_preflight_uses_automatic_under_threshold_authority_without_approval_file(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "automatic-under-threshold"
    prepared = runner.prepare_preflight_estimate(workspace, run_id)
    capacity_path = write_capacity_estimate(workspace, run_id, prepared)
    approval_path = workspace / "storage" / "workflow-runs" / run_id / "provider-approval.json"

    result = runner.run_runner_preflight(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
        credential_values=fixture_credentials(),
        probe=lambda spec: ProviderProbeResult.succeeded(spec),
        reload_environment=lambda: pytest.fail("fixture authority must not reload environment"),
        rebuild_client=lambda _spec: pytest.fail("fixture authority must not rebuild a client"),
        available_bytes=1_100,
    )

    receipt = result["receipt"]
    assert result["status"] == "succeeded"
    assert receipt["checks"]["spend_authority"] == {
        "mode": "automatic_under_usd_800",
        "estimate_digest": prepared["estimate_digest"],
        "threshold_usd": AUTOMATIC_AUTHORIZATION_THRESHOLD_USD,
        "total_ceiling_usd": prepared["total_projected_cost_usd"],
        "provider_stage_allocations": prepared["provider_stage_projection"],
    }
    assert not approval_path.exists()
    assert SECRET_SENTINEL not in json.dumps(receipt)


def write_candidate_frontier_inventory(
    workspace: Path,
    run_id: str,
    *,
    pair_keys: tuple[str, ...],
    excluded_pair_keys: tuple[str, ...] = (),
) -> Path:
    pairs = build_candidate_frontier(
        [
            {
                "canonical_pair_key": pair_key,
                "source_record_ids": [f"cvpr2026:source:{index}:a", f"cvpr2026:source:{index}:b"],
                "source_snapshot_digests": ["a" * 64, "b" * 64],
                "relation_kind": "program_item_matches_paper",
                "inclusion_status": "excluded" if pair_key in excluded_pair_keys else "pending_llm",
                "reason": "fixture candidate frontier",
            }
            for index, pair_key in enumerate(pair_keys, start=1)
        ]
    )
    path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "candidate_frontier": pairs,
                "candidate_frontier_sha256": pairs["frontier_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_frontier_authority_reprices_retained_candidates_and_preserves_superseded_evidence(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "frontier-authority"
    base = runner.prepare_preflight_estimate(workspace, run_id)
    write_candidate_frontier_inventory(
        workspace,
        run_id,
        pair_keys=("program:one|paper:one", "program:two|paper:two"),
        excluded_pair_keys=("program:two|paper:two",),
    )

    first = runner.prepare_source_discovery_frontier_authority(workspace, run_id)
    authority_path = workspace / "storage" / "workflow-runs" / run_id / "source-discovery-frontier-authority.json"

    assert authority_path.exists()
    assert first["base_full_estimate_digest"] == base["estimate_digest"]
    assert first["retained_join_quantity"] == 1
    assert first["candidate_frontier_sha256"] == json.loads(
        (workspace / "storage" / "source-discovery" / "source_discovery_inventory.json").read_text(encoding="utf-8")
    )["candidate_frontier_sha256"]
    assert first["fresh_volatile_preflight_required"] is True
    assert first["spend_authority"]["mode"] == "automatic_under_usd_800"

    first_bytes = authority_path.read_bytes()
    write_candidate_frontier_inventory(
        workspace,
        run_id,
        pair_keys=("program:one|paper:one", "program:three|paper:three"),
    )
    second = runner.prepare_source_discovery_frontier_authority(workspace, run_id)

    assert second["supersedes_authority_digest"] == first["authority_digest"]
    superseded = list(authority_path.parent.glob("source-discovery-frontier-authority.superseded-*.json"))
    assert len(superseded) == 1
    assert superseded[0].read_bytes() == first_bytes


def test_changed_frontier_blocks_paid_join_before_fresh_check_or_owner_call(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "frontier-mismatch"
    runner.prepare_preflight_estimate(workspace, run_id)
    write_candidate_frontier_inventory(workspace, run_id, pair_keys=("program:one|paper:one",))
    authority = runner.prepare_source_discovery_frontier_authority(workspace, run_id)
    authority_path = workspace / "storage" / "workflow-runs" / run_id / "source-discovery-frontier-authority.json"
    allocation = authority["provider_stage_allocations"][0]
    fresh_checks: list[str] = []
    owner_calls: list[str] = []

    write_candidate_frontier_inventory(workspace, run_id, pair_keys=("program:changed|paper:changed",))
    result = runner.dispatch_paid_call(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=workspace / "storage" / "workflow-runs" / run_id / "provider-approval.json",
        frontier_authority_path=authority_path,
        provider=allocation["provider"],
        stage=allocation["stage"],
        projected_cost_usd=allocation["ceiling_usd"],
        spent_total_usd=0.0,
        spent_by_provider_stage={},
        fresh_volatile_preflight=lambda: fresh_checks.append("called") or {"status": "succeeded"},
        owner=lambda: owner_calls.append("called") or 0.0,
        task_id="source_discovery.cross_source_join",
    )

    assert result["authorized"] is False
    assert {blocker["code"] for blocker in result["blockers"]} >= {"frontier_authority_incompatible"}
    assert fresh_checks == []
    assert owner_calls == []


def test_tavily_priority_gap_authority_is_exact_scoped_and_supersedes_changed_frontier(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "tavily-priority-frontier"
    runner.prepare_preflight_estimate(workspace, run_id)
    core = {
        "program_spine_sha256": "a" * 64,
        "harvest_inventory_sha256": "b" * 64,
        "query_policy_sha256": "c" * 64,
        "priority_program_item_ids": ["cvpr:item:001", "cvpr:item:002"],
        "priority_gap_quantity": 2,
    }
    frontier = {
        **core,
        "priority_gap_frontier_sha256": runner.canonical_json_digest(core),
    }
    ledger_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps(
            {"lineage": {"run_id": run_id}, "talk_harvest": {"priority_gap_frontier": frontier}},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    authority = runner.prepare_tavily_priority_gap_authority(workspace, run_id)
    authority_path = workspace / "storage" / "workflow-runs" / run_id / "tavily-priority-gap-authority.json"

    assert authority["authority_kind"] == "cvpr2026_tavily_priority_gap_bound"
    assert authority["priority_gap_quantity"] == 2
    assert authority["provider_stage_allocations"] == [
        {"provider": "tavily", "stage": "download-corpus", "ceiling_usd": 0.02}
    ]
    assert authority["spend_authority"]["mode"] == "automatic_under_usd_800"
    assert authority_path.exists()

    core["priority_program_item_ids"] = ["cvpr:item:changed"]
    core["priority_gap_quantity"] = 1
    frontier = {**core, "priority_gap_frontier_sha256": runner.canonical_json_digest(core)}
    ledger_path.write_text(
        json.dumps({"lineage": {"run_id": run_id}, "talk_harvest": {"priority_gap_frontier": frontier}}),
        encoding="utf-8",
    )
    current, blockers = runner._validate_tavily_priority_gap_authority(
        workspace, run_id, authority_path
    )

    assert current is None
    assert blockers == [{"code": "tavily_priority_gap_authority_incompatible", "affected_stage": "download-corpus"}]


def test_tavily_priority_gap_lane_blocks_before_http_when_no_credential_or_fresh_preflight(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "tavily-no-bare-key"
    runner.prepare_preflight_estimate(workspace, run_id)
    core = {
        "program_spine_sha256": "a" * 64,
        "harvest_inventory_sha256": "b" * 64,
        "query_policy_sha256": "c" * 64,
        "priority_program_item_ids": ["cvpr:item:001"],
        "priority_gap_quantity": 1,
    }
    frontier = {**core, "priority_gap_frontier_sha256": runner.canonical_json_digest(core)}
    ledger_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps({"lineage": {"run_id": run_id}, "talk_harvest": {"priority_gap_frontier": frontier}}),
        encoding="utf-8",
    )
    runner.prepare_tavily_priority_gap_authority(workspace, run_id)
    http_calls: list[str] = []

    outcome = runner.run_governed_tavily_priority_gap_search(
        workspace=workspace,
        run_id=run_id,
        authority_path=workspace / "storage" / "workflow-runs" / run_id / "tavily-priority-gap-authority.json",
        api_key=None,
        http_post=lambda _url, _payload: http_calls.append("called") or {},
        fresh_volatile_preflight=lambda _authority: pytest.fail("preflight must not run without a credential"),
    )

    assert outcome["status"] == "blocked"
    assert outcome["provider_calls_made"] == 0
    assert outcome["affected_priority_program_item_ids"] == ["cvpr:item:001"]
    assert outcome["blockers"] == [{"code": "tavily_credential_unavailable"}]
    assert http_calls == []

    # Supplying a bare key is still insufficient: the exact live authority
    # must pass the fresh Tavily-scoped preflight before an HTTP dispatch.
    fresh_blocked = runner.run_governed_tavily_priority_gap_search(
        workspace=workspace,
        run_id=run_id,
        authority_path=workspace / "storage" / "workflow-runs" / run_id / "tavily-priority-gap-authority.json",
        api_key="fixture-api-key-secret",
        http_post=lambda _url, _payload: pytest.fail("fresh preflight must block the provider call"),
        fresh_volatile_preflight=lambda authority: {
            "status": "blocked",
            "frontier_authority_digest": authority["authority_digest"],
            "provider_probe_scope": {"provider": "tavily", "stage": "download-corpus"},
        },
    )

    assert fresh_blocked["status"] == "blocked"
    assert fresh_blocked["provider_calls_made"] == 0


def test_tavily_priority_gap_lane_makes_zero_calls_for_a_valid_zero_gap_frontier(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "tavily-zero-gap"
    core = {
        "program_spine_sha256": "a" * 64,
        "harvest_inventory_sha256": "b" * 64,
        "query_policy_sha256": "c" * 64,
        "priority_program_item_ids": [],
        "priority_gap_quantity": 0,
    }
    frontier = {**core, "priority_gap_frontier_sha256": runner.canonical_json_digest(core)}
    ledger_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps({"lineage": {"run_id": run_id}, "talk_harvest": {"priority_gap_frontier": frontier}}),
        encoding="utf-8",
    )

    outcome = runner.run_governed_tavily_priority_gap_search(
        workspace=workspace,
        run_id=run_id,
        authority_path=workspace / "does-not-matter-for-zero-gap.json",
        api_key="fixture-api-key-secret",
        http_post=lambda _url, _payload: pytest.fail("zero frontier must not call Tavily"),
        fresh_volatile_preflight=lambda _authority: pytest.fail("zero frontier must not run a preflight"),
    )

    assert outcome == {
        "status": "not_required",
        "provider_calls_made": 0,
        "actual_cost_usd": 0.0,
        "affected_priority_program_item_ids": [],
        "provider_receipt_paths": [],
    }


def test_tavily_priority_gap_lane_uses_canonical_sanitized_provider_receipt(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "tavily-receipt"
    runner.prepare_preflight_estimate(workspace, run_id)
    core = {
        "program_spine_sha256": "a" * 64,
        "harvest_inventory_sha256": "b" * 64,
        "query_policy_sha256": "c" * 64,
        "priority_program_item_ids": ["cvpr:item:001"],
        "priority_gap_quantity": 1,
    }
    frontier = {**core, "priority_gap_frontier_sha256": runner.canonical_json_digest(core)}
    ledger_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps({"lineage": {"run_id": run_id}, "talk_harvest": {"priority_gap_frontier": frontier}}),
        encoding="utf-8",
    )
    (workspace / runner.CVPR_PROGRAM_SPINE_PATH).write_text(
        json.dumps(
            {
                "program_spine_sha256": "a" * 64,
                "items": [{"item_id": "cvpr:item:001", "title": "Fixture Keynote", "family": "keynote"}],
            }
        ),
        encoding="utf-8",
    )
    authority = runner.prepare_tavily_priority_gap_authority(workspace, run_id)

    outcome = runner.run_governed_tavily_priority_gap_search(
        workspace=workspace,
        run_id=run_id,
        authority_path=workspace / "storage" / "workflow-runs" / run_id / "tavily-priority-gap-authority.json",
        api_key="fixture-api-key-secret",
        http_post=lambda _url, _payload: {
            "results": [{
                "url": "https://www.youtube.com/watch?v=fixture01",
                "title": "Fixture Keynote recording",
            }]
        },
        fresh_volatile_preflight=lambda _authority: {
            "status": "succeeded",
            "frontier_authority_digest": authority["authority_digest"],
            "provider_probe_scope": {"provider": "tavily", "stage": "download-corpus"},
        },
    )

    assert outcome["status"] == "completed"
    assert outcome["provider_calls_made"] == 1
    assert outcome["actual_cost_usd"] == pytest.approx(0.01)
    receipt_path = workspace / outcome["provider_receipt_paths"][0]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rendered = json.dumps(receipt, sort_keys=True)
    assert receipt["receipt_type"] == "conference_pipeline.provider_call_receipt"
    assert receipt["task_id"] == "acquisition.youtube_talk_harvest.tavily_priority_gap_search"
    assert receipt["provider"] == "tavily"
    assert "fixture-api-key-secret" not in rendered
    assert receipt["secrets_copied"] is False


def test_authorized_exhausted_tavily_gaps_are_final_item_blockers_not_stage_blockers() -> None:
    """A completed, receipt-backed no-match search closes the paid lane truthfully."""

    runner = load_runner_module()
    item_ids = ["cvpr:item:one", "cvpr:item:two"]
    ledger: dict[str, object] = {
        "program_item_results": [
            {
                "program_item_id": item_id,
                "talk_media": {
                    "status": "gap",
                    "must_investigate": True,
                    "audio_downloaded": False,
                    "video_downloaded": False,
                },
            }
            for item_id in item_ids
        ],
        "talk_harvest": {
            "priority_gap_frontier": {
                "priority_program_item_ids": item_ids,
                "priority_gap_quantity": 2,
            }
        },
    }
    governed_tavily = {
        "status": "completed",
        "provider_calls_made": 2,
        "actual_cost_usd": 0.02,
        "calls": [
            {
                "program_item_id": item_id,
                "match": None,
                "provider_receipt_path": f"storage/provider-calls/{index}.json",
                "provider_receipt_sha256": f"{index}" * 64,
                "actual_cost_usd": 0.01,
            }
            for index, item_id in enumerate(item_ids, start=1)
        ],
        "secrets_copied": False,
    }

    stage_blockers = runner._finalize_governed_tavily_priority_gap_outcome(
        ledger=ledger,
        governed_tavily=governed_tavily,
        priority_gap_count=2,
    )

    assert stage_blockers == []
    assert ledger["talk_harvest"]["tavily_lane"]["status"] == "completed_exhausted"
    assert ledger["talk_harvest"]["tavily_lane"]["unmatched_after_governed_tavily_count"] == 2
    assert all(
        row["talk_media"]["status"] == "unmatched_after_governed_tavily"
        and row["talk_media"]["must_investigate"] is True
        and row["talk_media"]["provider_receipt_path"].startswith("storage/provider-calls/")
        for row in ledger["program_item_results"]
    )


@pytest.mark.parametrize(
    "status, blockers",
    [
        ("blocked", [{"code": "tavily_credential_unavailable"}]),
        ("blocked", [{"code": "tavily_pricing_unavailable"}]),
    ],
    ids=("unauthorized", "unpriced"),
)
def test_unavailable_or_unpriced_tavily_remains_a_stage_blocker(
    status: str, blockers: list[dict[str, str]]
) -> None:
    runner = load_runner_module()
    ledger: dict[str, object] = {
        "program_item_results": [{"program_item_id": "cvpr:item:one"}],
        "talk_harvest": {
            "priority_gap_frontier": {
                "priority_program_item_ids": ["cvpr:item:one"],
                "priority_gap_quantity": 1,
            }
        },
    }

    stage_blockers = runner._finalize_governed_tavily_priority_gap_outcome(
        ledger=ledger,
        governed_tavily={
            "status": status,
            "provider_calls_made": 0,
            "blockers": blockers,
        },
        priority_gap_count=1,
    )

    assert stage_blockers == [{
        "code": "priority_talk_gaps_governed_tavily_blocked",
        "affected_stage": "download-corpus",
        "count": 1,
    }]
    assert ledger["talk_harvest"]["tavily_lane"]["status"] == "blocked"


def test_cvpr_public_talk_lane_limits_targeted_search_to_priority_families(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "cvpr-priority-search-scope"
    spine = {
        "program_spine_sha256": "a" * 64,
        "items": [
            {"item_id": "cvpr:item:keynote", "title": "Keynote", "family": "keynote"},
            {"item_id": "cvpr:item:poster", "title": "Poster", "family": "poster"},
        ],
    }
    monkeypatch.setattr(runner, "_build_pinned_cvpr_program_spine", lambda **_kwargs: (spine, []))
    captured: dict[str, object] = {}
    inventory_path = workspace / "storage" / "youtube-talks" / "harvest_inventory.json"

    def harvester(config: object) -> dict[str, object]:
        captured["config"] = config
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(
            json.dumps(
                {
                    "talks": [],
                    "gaps": [{"item_id": "cvpr:item:keynote", "must_investigate": True}],
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "partial",
            "allowed_hosts": ["youtube.com", "youtu.be"],
            "boundaries": {"audio_downloaded": False, "video_downloaded": False},
            "pilot_limit": None,
            "tavily_lane": {"enabled": False},
            "inventory_path": str(inventory_path.relative_to(workspace)),
            "receipt_path": "storage/youtube-talks/harvest_receipt.json",
        }

    ledger: dict[str, object] = {}
    completed, outcome, blockers = runner._run_cvpr_public_talk_lane(
        workspace=workspace,
        ledger=ledger,
        corpus_manifest={},
        run_id=run_id,
        pinned_run_id=run_id,
        pinned_inventory_sha256="b" * 64,
        command_args=(),
        harvester=harvester,
        prerequisite_audit=lambda: {"status": "pass", "blockers": []},
    )

    assert completed is True
    assert outcome["priority_gap_count"] == 1
    assert blockers == [{
        "code": "priority_talk_gaps_pending_governed_tavily_escalation",
        "affected_stage": "download-corpus",
        "count": 1,
    }]
    config = captured["config"]
    assert [item.item_id for item in config.program_spine] == ["cvpr:item:keynote"]
    assert config.tavily_api_key is None
    assert ledger["talk_harvest"]["public_search_scope"] == {
        "mode": "priority_families_only",
        "full_program_spine_item_count": 2,
        "targeted_search_item_count": 1,
        "deferred_nonpriority_item_count": 1,
        "priority_families": ["keynote", "invited", "oral"],
    }


def test_talk_harvest_frontier_is_durable_before_governed_tavily_authority(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "tavily-durable-frontier"
    core = {
        "program_spine_sha256": "a" * 64,
        "harvest_inventory_sha256": "b" * 64,
        "query_policy_sha256": "c" * 64,
        "priority_program_item_ids": ["cvpr:item:001"],
        "priority_gap_quantity": 1,
    }
    frontier = {**core, "priority_gap_frontier_sha256": runner.canonical_json_digest(core)}
    ledger = {
        "lineage": {"run_id": run_id},
        "talk_harvest": {"priority_gap_frontier": frontier},
    }

    runner._checkpoint_talk_harvest_frontier(workspace=workspace, ledger=ledger)

    persisted = json.loads((workspace / runner.DOWNLOAD_CORPUS_PATH).read_text(encoding="utf-8"))
    assert persisted["manifest_sha256"] == runner.canonical_ledger_digest(persisted)
    assert runner._tavily_priority_gap_frontier(workspace, run_id) == frontier


def test_source_discovery_owner_runs_collection_authority_fresh_pre_spend_and_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "source-owner-success"
    runner.prepare_preflight_estimate(workspace, run_id)
    runner.prepare_preflight_storage_estimate(workspace, run_id)
    source_id = immutable_id("cvpr2026:source:program", "program-one")
    item_id = immutable_id("cvpr2026:program-item", "item-one")
    frontier = build_candidate_frontier(
        [
            {
                "canonical_pair_key": f"{item_id}|{source_id}",
                "source_record_ids": [item_id, source_id],
                "source_snapshot_digests": ["a" * 64, "b" * 64],
                "relation_kind": "program_item_to_paper",
                "inclusion_status": "pending_llm",
                "reason": "fixture frontier is current and complete",
            }
        ]
    )
    order: list[str] = []
    original_prepare_estimate = runner.prepare_preflight_estimate
    original_prepare_capacity = runner.prepare_preflight_storage_estimate

    def refresh_estimate(workspace: Path, run_id: str) -> dict[str, object]:
        order.append("refresh_estimate")
        return original_prepare_estimate(workspace, run_id)

    def refresh_capacity(workspace: Path, run_id: str) -> dict[str, object]:
        order.append("refresh_capacity")
        return original_prepare_capacity(workspace, run_id)

    monkeypatch.setattr(runner, "prepare_preflight_estimate", refresh_estimate)
    monkeypatch.setattr(runner, "prepare_preflight_storage_estimate", refresh_capacity)

    def collect(*, workspace: Path, run_id: str) -> dict[str, object]:
        order.append("collection")
        inventory = build_canonical_inventory(
            run_id=run_id,
            family_snapshots={
                "program-schedule": {
                    "snapshot_id": "program-fixture",
                    "sha256": "a" * 64,
                    "status": "live_valid",
                }
            },
            source_records=[
                {
                    "source_record_id": source_id,
                    "native_ids": ["program-one"],
                    "aliases": ["https://fixtures.example/program-one"],
                    "family_id": "program-schedule",
                    "snapshot_sha256": "a" * 64,
                }
            ],
            program_items=[
                {
                    "program_item_id": item_id,
                    "aliases": ["item-one"],
                    "source_record_ids": [source_id],
                    "field_evidence": {"title": {"authority": "program-schedule", "snapshot_sha256": "a" * 64}},
                }
            ],
            edges=[],
            seed_diffs=[],
            taxonomy={"taxonomy_sha256": "c" * 64, "blockers": []},
            candidate_frontier=frontier,
        )
        publish_canonical_inventory(
            workspace / "storage" / "source-discovery" / "source_discovery_inventory.json",
            inventory,
        )
        return {
            "status": "succeeded",
            "expected_family_ids": ["program-schedule"],
            "field_values": {},
            "candidate_evidence": {
                frontier["pairs"][0]["canonical_pair_key"]: [
                    {"field": "title", "source_record_id": item_id, "value": "Fixture program title"},
                    {"field": "title", "source_record_id": source_id, "value": "Fixture paper title"},
                ]
            },
        }

    def fresh(authority: dict[str, object]) -> dict[str, object]:
        order.append("fresh")
        return {"status": "succeeded", "frontier_authority_digest": authority["authority_digest"]}

    from conference_pipeline.model_governance import default_model as _governed_model

    class Adapter:
        provider = "openai"
        model = _governed_model("source_discovery.cross_source_join")

        def invoke(self, _candidate: dict[str, object]) -> dict[str, object]:
            order.append("provider")
            assert _candidate["evidence"] == [
                {
                    "field": "title",
                    "source_record_id": item_id,
                    "value": "Fixture program title",
                    "value_sha256": runner.canonical_json_digest("Fixture program title"),
                },
                {
                    "field": "title",
                    "source_record_id": source_id,
                    "value": "Fixture paper title",
                    "value_sha256": runner.canonical_json_digest("Fixture paper title"),
                },
            ]
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
            receipt_path = workspace / receipt["receipt_path"]
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            return {
                "decision": "join",
                "confidence": 0.99,
                "conflict_state": "none",
                "receipt": receipt,
            }

    outcome = runner.run_source_discovery_stage(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=workspace / "storage" / "workflow-runs" / run_id / "provider-approval.json",
        capacity_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-storage-estimate.json",
        collector=collect,
        fresh_volatile_preflight=fresh,
        provider_adapter_factory=lambda **_kwargs: Adapter(),
    )

    assert outcome["status"] == "succeeded"
    receipt = outcome["receipt"]
    assert receipt["stage_contract"]["status"] == "pass"
    assert receipt["pre_spend"]["status"] == "precondition_passed"
    assert receipt["provider_calls_made"] == 1
    assert receipt["frontier_authority_lineage"]["candidate_frontier_sha256"] == frontier["frontier_sha256"]
    assert receipt["canonical_inventory"]["inventory_sha256"]
    assert receipt["required_script_ids"] == ["acquisition.source_discovery.cvpr2026"]
    assert order[0] == "collection"
    assert order.index("refresh_estimate") > order.index("collection")
    assert order.index("refresh_capacity") > order.index("refresh_estimate")
    assert order.index("fresh") > order.index("refresh_capacity")
    assert order.index("provider") > order.index("fresh")


def test_source_discovery_owner_stops_before_authority_when_collection_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "prepare_source_discovery_frontier_authority",
        lambda *_args, **_kwargs: calls.append("authority") or pytest.fail("authority must not run"),
    )

    outcome = runner.run_source_discovery_stage(
        workspace=workspace,
        run_id="collection-blocked",
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / "collection-blocked" / "preflight-full-run-estimate.json",
        approval_path=workspace / "storage" / "workflow-runs" / "collection-blocked" / "provider-approval.json",
        capacity_estimate_path=workspace / "storage" / "workflow-runs" / "collection-blocked" / "preflight-storage-estimate.json",
        collector=lambda **_kwargs: {"status": "blocked", "blockers": [{"code": "source_collection_blocked"}]},
    )

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["provider_calls_made"] == 0
    assert calls == []


def test_source_discovery_resume_fails_closed_when_a_legacy_paid_attempt_has_no_decision_ledger(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "legacy-paid-partial"
    runner.prepare_preflight_estimate(workspace, run_id)
    runner.prepare_preflight_storage_estimate(workspace, run_id)
    write_candidate_frontier_inventory(workspace, run_id, pair_keys=("program:one|paper:one",))
    prior_path = workspace / "storage" / "workflow-runs" / run_id / "stage-02-source-discovery.json"
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(
        json.dumps(
            {
                "stage_id": "source-discovery",
                "status": "blocked",
                "provider_calls_made": 11,
                "provider_receipt_paths": ["storage/workflow-runs/legacy/provider.json"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    provider_calls: list[str] = []

    outcome = runner.run_source_discovery_stage(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-full-run-estimate.json",
        approval_path=workspace / "storage" / "workflow-runs" / run_id / "provider-approval.json",
        capacity_estimate_path=workspace / "storage" / "workflow-runs" / run_id / "preflight-storage-estimate.json",
        collector=lambda **_kwargs: {"status": "succeeded", "expected_family_ids": [], "field_values": {}},
        fresh_volatile_preflight=lambda _authority: pytest.fail("legacy resume must stop before a fresh paid preflight"),
        provider_adapter_factory=lambda **_kwargs: provider_calls.append("constructed"),
        resume=True,
    )

    assert outcome["status"] == "blocked"
    assert {blocker["code"] for blocker in outcome["receipt"]["blockers"]} == {"partial_decision_ledger_unavailable"}
    assert outcome["receipt"]["resume"] == {"requested": True, "reused_decision_count": 0, "status": "blocked"}
    assert provider_calls == []
    assert prior_path.read_text(encoding="utf-8").startswith("{")


def test_receipt_bound_decision_ledger_resumes_without_duplicate_dispatch_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "durable-partial-ledger"
    runner.prepare_preflight_estimate(workspace, run_id)
    write_candidate_frontier_inventory(workspace, run_id, pair_keys=("program:one|paper:one",))
    authority = runner.prepare_source_discovery_frontier_authority(workspace, run_id)
    frontier, _digest, _count = runner._current_candidate_frontier(workspace, run_id)
    candidate = runner.build_cross_source_join_request(frontier)["candidates"][0]
    receipt_path = "storage/provider-calls/source-discovery/resume-fixture.json"
    provider_receipt = {
        "receipt_type": "conference_pipeline.provider_call_receipt",
        "run_id": run_id,
        "stage_id": "source-discovery",
        "task_id": "source_discovery.cross_source_join",
        "provider": authority["provider"],
        "model": authority["model"],
        "receipt_path": receipt_path,
        "request_sha256": "e" * 64,
        "estimated_cost_usd": 0.125,
        "secrets_copied": False,
    }
    path = workspace / receipt_path
    path.parent.mkdir(parents=True, exist_ok=True)
    provider_receipt["receipt_path"] = str(path)
    path.write_text(json.dumps(provider_receipt, sort_keys=True), encoding="utf-8")
    output = {"decision": "join", "confidence": 0.99, "conflict_state": "none"}
    decision = {
        "canonical_pair_key": candidate["canonical_pair_key"],
        "source_record_ids": candidate["source_record_ids"],
        "frontier_sha256": frontier["frontier_sha256"],
        "evidence_digests": candidate["source_snapshot_digests"],
        "join_method": "governed_llm_frontier",
        "boundary_method": "governed_llm_frontier",
        **output,
        "registry_task_id": "source_discovery.cross_source_join",
        "actual_provider": authority["provider"],
        "actual_model": authority["model"],
        "provider_receipt_path": receipt_path,
        "provider_receipt_sha256": runner.canonical_json_digest(
            runner.sanitize_for_receipt({**provider_receipt, "receipt_path": receipt_path})
        ),
        "input_sha256": candidate["input_sha256"],
        "output_sha256": runner.canonical_json_digest(runner.sanitize_for_receipt(output)),
    }
    production_source_discovery_stage.write_decision_ledger(
        runner_id=runner.CVPR_RUNNER_CODE_IDENTITY["runner"],
        io=runner._stage_io(),
        workspace=workspace,
        run_id=run_id,
        frontier_authority=authority,
        decisions=[decision],
        receipt_paths=[receipt_path],
        actual_cost_usd=0.125,
    )

    decisions, paths, cost, blockers = production_source_discovery_stage.load_resumable_decision_ledger(
        runner._production_context(workspace, run_id),
        frontier_authority=authority,
        frontier=frontier,
        candidate_evidence=None,
        substitution_waiver=None,
    )
    result = runner.execute_governed_frontier_joins(
        workspace=workspace,
        run_id=run_id,
        frontier=frontier,
        frontier_authority=authority,
        base_full_estimate_digest=authority["base_full_estimate_digest"],
        volatile_preflight={"status": "succeeded"},
        pre_spend_validator=lambda **_kwargs: {"status": "precondition_passed", "blockers": []},
        provider_adapter_factory=lambda **_kwargs: pytest.fail("matching partial decision must not reconstruct an adapter"),
        paid_dispatch=lambda **_kwargs: pytest.fail("matching partial decision must not dispatch again"),
        field_values={},
        prior_decisions=decisions,
        prior_receipt_paths=paths,
        prior_actual_cost_usd=cost,
    )

    assert blockers == []
    assert paths == [receipt_path]
    assert cost == 0.125
    assert result["status"] == "succeeded"
    assert result["provider_calls_made"] == 1
    assert result["new_provider_calls_made"] == 0
    assert result["reused_provider_calls"] == 1
    assert result["actual_cost_usd"] == 0.125

    provider_receipt["model"] = "tampered-model"
    path.write_text(json.dumps(provider_receipt, sort_keys=True), encoding="utf-8")
    _decisions, _paths, _cost, mutated_blockers = production_source_discovery_stage.load_resumable_decision_ledger(
        runner._production_context(workspace, run_id),
        frontier_authority=authority,
        frontier=frontier,
        candidate_evidence=None,
        substitution_waiver=None,
    )
    assert {blocker["code"] for blocker in mutated_blockers} == {"partial_decision_ledger_incompatible"}


def test_cli_passes_resume_to_source_discovery_owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        runner,
        "run_source_discovery_stage",
        lambda **kwargs: captured.update(kwargs) or {"status": "succeeded", "summary": {}},
    )

    assert runner.main(["--workspace", str(workspace), "--run-id", "resume-propagation", "--stage", "source-discovery", "--resume"]) == 0
    assert captured["resume"] is True


def test_cli_download_corpus_uses_only_the_governed_tavily_bridge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setenv("TAVILY_API_KEY", "fixture-tavily-key")
    monkeypatch.setattr(
        runner,
        "run_download_corpus_stage",
        lambda **kwargs: captured.update(kwargs) or {"status": "succeeded", "summary": {}},
    )
    preflight_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "run_runner_preflight",
        lambda **kwargs: preflight_calls.append(kwargs) or {"status": "succeeded"},
    )

    assert runner.main([
        "--workspace", str(workspace), "--run-id", "governed-tavily-cli", "--stage", "download-corpus", "--resume",
    ]) == 0

    assert captured["execute_tavily_lane"] is True
    assert captured["tavily_api_key"] == "fixture-tavily-key"
    assert captured["tavily_http_post"] is runner._tavily_priority_gap_http_post
    fresh = captured["tavily_fresh_preflight"]
    assert callable(fresh)
    assert fresh({"authority_digest": "a" * 64}) == {
        "status": "succeeded",
        "frontier_authority_digest": "a" * 64,
        "provider_probe_scope": {"provider": "tavily", "stage": "download-corpus"},
    }
    assert preflight_calls[0]["provider_probe_scope"] == ("tavily", "download-corpus")


def _write_event_media_pin_check_fixture(
    workspace: Path, *, inventory_sha256: str
) -> None:
    """Write the minimal on-disk pair ``_event_media_inventory_inputs`` reads
    before it ever reaches ``validate_registration_media_overlay`` -- enough
    to exercise the pinned-digest comparison itself in isolation."""

    runner = load_runner_module()
    source_path = workspace / runner.SOURCE_DISCOVERY_INVENTORY_PATH
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "run_id": runner.PINNED_CORPUS_SOURCE_RUN_ID,
                "status": "prepared",
                "inventory_sha256": inventory_sha256,
                "source_records": [],
            }
        ),
        encoding="utf-8",
    )
    r4_path = workspace / runner.OVERLAY_LEDGER_RELATIVE_PATH
    r4_path.parent.mkdir(parents=True, exist_ok=True)
    r4_path.write_text(json.dumps({}), encoding="utf-8")


def test_event_media_pinned_inventory_digest_mismatch_raises_actionable_error(
    tmp_path: Path,
) -> None:
    """A source inventory whose inventory_sha256 differs from the pin still
    RAISES (the fail-closed guard is not weakened) -- but with a message that
    names PINNED_CORPUS_SOURCE_INVENTORY_SHA256, shows both the expected and
    observed digest, and mentions the rederive-script remedy."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    wrong_sha256 = "1" * 64
    _write_event_media_pin_check_fixture(workspace, inventory_sha256=wrong_sha256)

    with pytest.raises(ValueError) as excinfo:
        runner._event_media_inventory_inputs(workspace, boundary=FIXTURE_BOUNDARY)

    message = str(excinfo.value)
    assert "PINNED_CORPUS_SOURCE_INVENTORY_SHA256" in message
    assert runner.PINNED_CORPUS_SOURCE_INVENTORY_SHA256 in message
    assert wrong_sha256 in message
    assert "rederive_source_discovery_authors.py" in message


@pytest.mark.repo_data("conferences/cvpr2026/storage/source-discovery/source_discovery_inventory.json")
def test_pinned_inventory_sha256_matches_live_post_rederive_inventory_field() -> None:
    """Phase 10.1-10 Task 5: after the real (non-dry-run) rederive that this
    milestone actually executed against the real CVPR2026 workspace,
    PINNED_CORPUS_SOURCE_INVENTORY_SHA256 must equal the LIVE, on-disk
    storage/source-discovery/source_discovery_inventory.json's own
    "inventory_sha256" field -- read directly from the file, never
    hardcoded as a duplicate literal here (a literal would silently drift
    from the real file the next time either one changes, exactly the trap
    this test used to fall into before the rederive)."""

    runner = load_runner_module()
    repo_root = Path(__file__).resolve().parent.parent
    live_inventory_path = (
        repo_root
        / "conferences"
        / "cvpr2026"
        / "storage"
        / "source-discovery"
        / "source_discovery_inventory.json"
    )
    live_inventory = json.loads(live_inventory_path.read_text(encoding="utf-8"))
    assert runner.PINNED_CORPUS_SOURCE_INVENTORY_SHA256 == live_inventory["inventory_sha256"]


def test_event_media_matching_pinned_inventory_digest_does_not_raise(
    tmp_path: Path,
) -> None:
    """A source inventory whose inventory_sha256 DOES match the pin passes the
    digest comparison unchanged -- whatever this call raises next (e.g. an
    incomplete fixture r4 overlay) is NOT the pinned-digest error."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _write_event_media_pin_check_fixture(
        workspace, inventory_sha256=runner.PINNED_CORPUS_SOURCE_INVENTORY_SHA256
    )

    try:
        runner._event_media_inventory_inputs(workspace, boundary=FIXTURE_BOUNDARY)
    except Exception as exc:  # noqa: BLE001 - proving it's a DIFFERENT failure
        assert "PINNED_CORPUS_SOURCE_INVENTORY_SHA256" not in str(exc)
        assert "pinned r3 schedule" not in str(exc)


def test_registration_media_cli_is_narrow_and_never_enters_the_public_r3_lane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        runner,
        "run_cvpr_registration_media_stage",
        lambda **kwargs: captured.update(kwargs) or {"status": "succeeded", "summary": {}},
    )
    monkeypatch.setattr(
        runner,
        "run_download_corpus_stage",
        lambda **_kwargs: pytest.fail("registration-media must not dispatch the r3 public lane"),
    )

    assert runner.main([
        "--workspace", str(workspace),
        "--run-id", "cvpr2026-live-production-r4-registration-media",
        "--stage", "download-corpus",
        "--registration-media",
        "--resume",
    ]) == 0

    assert captured["run_id"] == "cvpr2026-live-production-r4-registration-media"
    assert captured["resume"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--stage", "source-discovery", "--registration-media"],
        ["--stage", "download-corpus", "--registration-media", "--run-id", "wrong-r4-run"],
    ],
)
def test_registration_media_cli_rejects_invalid_stage_or_run_id(argv: list[str]) -> None:
    runner = load_runner_module()

    with pytest.raises(SystemExit):
        runner.parse_args(argv)


def test_registration_media_preflight_block_writes_no_overlay_or_opener_attempt(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    predecessors, plan, overlay = registration_media_fixture_inputs()
    opener_calls: list[str] = []

    outcome = runner.run_cvpr_registration_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r4-registration-media",
        input_loader=lambda _workspace, **_kwargs: (predecessors, plan, overlay),
        opener_factory=lambda **_kwargs: (
            None,
            {
                "preflight_status": "blocked",
                "auth_status": "registration_auth_required",
                "secrets_copied": False,
            },
        ),
        recoverer=lambda **_kwargs: pytest.fail("blocked preflight must not recover candidates"),
        available_bytes=1_000_000,
    )

    receipt = outcome["receipt"]
    assert outcome["status"] == receipt["status"] == "blocked"
    assert opener_calls == []
    assert {blocker["code"] for blocker in receipt["blockers"]} == {"registration_auth_required"}
    owner = receipt["owner_execution_evidence"]["acquisition.download_corpus.cvpr2026"]
    assert owner["transport_created"] is False
    assert owner["authenticated_request_count"] == 0
    assert not (workspace / "storage" / "download-corpus" / "registration-media" / "registration_media.json").exists()
    serialized = json.dumps(receipt, sort_keys=True)
    assert SECRET_SENTINEL not in serialized
    assert "CVPR2026_VIRTUAL_COOKIES_FILE" not in serialized


@pytest.mark.parametrize(
    ("sentinel", "expected_reason"),
    [
        (
            {"status": "registration_auth_required", "auth_status": "registration_auth_required"},
            "registration_auth_required",
        ),
        (
            {"status": "registered_access_denied", "auth_status": "registered_access_denied"},
            "registered_access_denied",
        ),
    ],
    ids=("registration-wall", "validated-account-denied"),
)
def test_registration_media_sentinel_stops_before_fanout_with_precise_reason(
    tmp_path: Path,
    sentinel: dict[str, str],
    expected_reason: str,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    predecessors, plan, overlay = registration_media_fixture_inputs()
    opened: list[str] = []

    def opener(url: str) -> dict[str, str]:
        opened.append(url)
        return sentinel

    outcome = runner.run_cvpr_registration_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r4-registration-media",
        input_loader=lambda _workspace, **_kwargs: (predecessors, plan, overlay),
        opener_factory=lambda **_kwargs: (
            opener,
            {
                "preflight_status": "pass",
                "auth_status": "preflight_passed",
                "secrets_copied": False,
            },
        ),
        recoverer=lambda **_kwargs: pytest.fail("blocked sentinel must not recover remaining candidates"),
        available_bytes=1_000_000,
    )

    receipt = outcome["receipt"]
    assert outcome["status"] == receipt["status"] == "blocked"
    assert len(opened) == 1
    assert receipt["checks"]["registration_media_sentinel_request_count"] == 1
    assert receipt["checks"]["registration_media_sentinel_status"] == expected_reason
    assert {blocker["code"] for blocker in receipt["blockers"]} == {expected_reason}
    assert not (workspace / "storage" / "download-corpus" / "registration-media" / "registration_media.json").exists()


def test_registration_media_complete_zero_link_overlay_uses_r4_profile_and_standard_stage4_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    predecessors, plan, initial_overlay = registration_media_fixture_inputs()
    opened: list[str] = []
    contract_calls: list[dict[str, object]] = []

    class PassingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "pass", "blockers": []}

    monkeypatch.setattr(
        runner,
        "verify_stage_contract",
        lambda **kwargs: contract_calls.append(kwargs) or PassingContract(),
    )

    outcome = runner.run_cvpr_registration_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r4-registration-media",
        input_loader=lambda _workspace, **_kwargs: (predecessors, plan, initial_overlay),
        opener_factory=lambda **_kwargs: (
            lambda url: opened.append(url) or {
                "status": "authenticated",
                "auth_status": "authenticated",
                "http_status": 200,
                "final_url": url,
                "content_type": "text/html",
                "body": b"<main></main>",
            },
            {
                "preflight_status": "pass",
                "auth_status": "preflight_passed",
                "secrets_copied": False,
            },
        ),
        recoverer=lambda **kwargs: completed_zero_link_overlay(kwargs["overlay"]),
        available_bytes=1_000_000,
    )

    receipt = outcome["receipt"]
    overlay_path = workspace / "storage" / "download-corpus" / "registration-media" / "registration_media.json"
    assert outcome["status"] == receipt["status"] == "succeeded"
    assert receipt["stage_id"] == "download-corpus"
    assert receipt["stage_order"] == 4
    assert receipt["run_id"] == "cvpr2026-live-production-r4-registration-media"
    assert receipt["checks"]["direct_youtube_media_record_count"] == 0
    assert receipt["checks"]["official_event_request_count"] == 2_579
    assert receipt["checks"]["official_event_program_item_count"] == 5_159
    assert len(opened) == 1
    assert overlay_path.is_file()
    owner = receipt["owner_execution_evidence"]["acquisition.download_corpus.cvpr2026"]
    assert owner["transport_created"] is True
    assert owner["authenticated_request_count"] == 1
    assert contract_calls[0]["profile"] == "registration-media"
    assert contract_calls[0]["stage_id"] == "download-corpus"


@pytest.mark.parametrize("failure_status", ["registration_auth_required", "registered_access_denied"])
def test_registration_media_post_sentinel_auth_failure_is_a_blocked_attempt_not_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_status: str,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    predecessors, plan, initial_overlay = registration_media_fixture_inputs()
    contract_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runner,
        "verify_stage_contract",
        lambda **kwargs: contract_calls.append(kwargs) or pytest.fail("auth failure must not enter strict success"),
    )

    def recovered_with_expiry(**kwargs: object) -> dict[str, object]:
        overlay = completed_zero_link_overlay(kwargs["overlay"])  # type: ignore[index]
        failed_request = next(
            row
            for row in reversed(overlay["official_event_requests"])
            if row["transport_eligible"]
        )
        failed_request.update({"result_status": failure_status, "auth_status": failure_status})
        overlay["overlay_sha256"] = canonical_registration_media_digest(overlay)
        return overlay

    outcome = runner.run_cvpr_registration_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r4-registration-media",
        input_loader=lambda _workspace, **_kwargs: (predecessors, plan, initial_overlay),
        opener_factory=lambda **_kwargs: (
            lambda url: {
                "status": "authenticated",
                "auth_status": "authenticated",
                "http_status": 200,
                "final_url": url,
                "content_type": "text/html",
                "body": b"<main></main>",
            },
            {"preflight_status": "pass", "auth_status": "preflight_passed", "secrets_copied": False},
        ),
        recoverer=recovered_with_expiry,
        available_bytes=1_000_000,
    )

    assert outcome["status"] == "blocked"
    assert {blocker["code"] for blocker in outcome["receipt"]["blockers"]} == {failure_status}
    assert contract_calls == []
    checks = outcome["receipt"]["checks"]
    assert checks["download_corpus_cvpr2026_execution_status"] == "blocked"
    assert checks["authenticated_registration_media_recovery_status"] == "blocked"
    assert checks["official_event_media_accounting_complete"] is False
    assert checks["complete_program_item_accounting"] is False
    assert not (workspace / "storage" / "download-corpus" / "registration-media" / "registration_media.json").exists()


def test_registration_media_contract_failure_keeps_completed_overlay_private(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    predecessors, plan, initial_overlay = registration_media_fixture_inputs()

    class FailingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "blocked", "blockers": [{"code": "fixture_contract_failure"}]}

    monkeypatch.setattr(runner, "verify_stage_contract", lambda **_kwargs: FailingContract())
    outcome = runner.run_cvpr_registration_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r4-registration-media",
        input_loader=lambda _workspace, **_kwargs: (predecessors, plan, initial_overlay),
        opener_factory=lambda **_kwargs: (
            lambda url: {
                "status": "authenticated",
                "auth_status": "authenticated",
                "http_status": 200,
                "final_url": url,
                "content_type": "text/html",
                "body": b"<main></main>",
            },
            {"preflight_status": "pass", "auth_status": "preflight_passed", "secrets_copied": False},
        ),
        recoverer=lambda **kwargs: completed_zero_link_overlay(kwargs["overlay"]),
        available_bytes=1_000_000,
    )

    assert outcome["status"] == outcome["receipt"]["status"] == "blocked"
    assert "final_registration_media_stage_contract_incomplete" in {
        blocker["code"] for blocker in outcome["receipt"]["blockers"]
    }
    registration_media_dir = workspace / "storage" / "download-corpus" / "registration-media"
    assert not (registration_media_dir / "registration_media.json").exists()
    assert (registration_media_dir / "registration_media.checkpoint.json").is_file()


def test_registration_media_resume_reuses_only_compatible_private_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    predecessors, plan, initial_overlay = registration_media_fixture_inputs()

    class PassingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "pass", "blockers": []}

    monkeypatch.setattr(runner, "verify_stage_contract", lambda **_kwargs: PassingContract())
    checkpoint_seen: list[dict[str, object]] = []

    def partial_recoverer(**kwargs: object) -> dict[str, object]:
        overlay = copy.deepcopy(kwargs["overlay"])  # type: ignore[arg-type]
        first = next(row for row in overlay["official_event_requests"] if row["transport_eligible"])
        first.update({"result_status": "no_admissible_official_youtube_link", "auth_status": "authenticated"})
        overlay["status"] = "registration_media_partial"
        overlay["recovery_counts"] = {
            "completed_request_count": 1 + FULL_R4_SCOPE.boundary_request_count,
            "direct_youtube_media_record_count": 0,
            "transport_eligible_request_count": FULL_R4_SCOPE.transport_eligible_request_count,
            "boundary_no_network_request_count": FULL_R4_SCOPE.boundary_request_count,
        }
        overlay["overlay_sha256"] = canonical_registration_media_digest(overlay)
        kwargs["checkpoint"](overlay)  # type: ignore[index]
        return overlay

    common = {
        "workspace": workspace,
        "run_id": "cvpr2026-live-production-r4-registration-media",
        "input_loader": lambda _workspace, **_kwargs: (predecessors, plan, initial_overlay),
        "opener_factory": lambda **_kwargs: (
            lambda url: {
                "status": "authenticated",
                "auth_status": "authenticated",
                "http_status": 200,
                "final_url": url,
                "content_type": "text/html",
                "body": b"<main></main>",
            },
            {"preflight_status": "pass", "auth_status": "preflight_passed", "secrets_copied": False},
        ),
        "available_bytes": 1_000_000,
    }
    first = runner.run_cvpr_registration_media_stage(**common, recoverer=partial_recoverer)
    assert first["status"] == "blocked"
    checkpoint_path = workspace / "storage" / "download-corpus" / "registration-media" / "registration_media.checkpoint.json"
    assert checkpoint_path.is_file()

    def resumed_recoverer(**kwargs: object) -> dict[str, object]:
        checkpoint_seen.append(copy.deepcopy(kwargs["overlay"]))  # type: ignore[arg-type]
        return completed_zero_link_overlay(kwargs["overlay"])  # type: ignore[index]

    resumed = runner.run_cvpr_registration_media_stage(
        **common,
        resume=True,
        recoverer=resumed_recoverer,
    )
    assert resumed["status"] == "succeeded"
    assert next(
        row["result_status"]
        for row in checkpoint_seen[0]["official_event_requests"]
        if row["transport_eligible"]
    ) == "no_admissible_official_youtube_link"
    assert not checkpoint_path.exists()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("{}\n", encoding="utf-8")
    blocked = runner.run_cvpr_registration_media_stage(
        **common,
        resume=True,
        recoverer=lambda **_kwargs: pytest.fail("mixed checkpoint must fail before recovery"),
    )
    assert blocked["status"] == "blocked"
    assert "registration_media_checkpoint_incompatible" in {
        blocker["code"] for blocker in blocked["receipt"]["blockers"]
    }


def test_event_media_resume_skips_checkpointed_urls_and_keeps_secrets_out_of_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    baseline = event_media_fixture_inventory()
    monkeypatch.setattr(runner, "_event_media_inventory_inputs", lambda _workspace, **_kwargs: copy.deepcopy(baseline))
    first_calls: list[str] = []

    def first_factory(**kwargs: object):
        frozen_urls = kwargs["frozen_urls"]
        assert isinstance(frozen_urls, list)
        assert len(frozen_urls) == 2_693

        def first_opener(url: str) -> dict[str, object]:
            first_calls.append(url)
            if len(first_calls) == 1:
                return {
                    "status": "authenticated",
                    "auth_status": "authenticated",
                    "http_status": 200,
                    "content_type": "text/html",
                    "body": b'<iframe src="https://www.youtube.com/embed/AbcToken123"></iframe>' + SECRET_SENTINEL.encode(),
                    "headers": {"Cookie": SECRET_SENTINEL},
                }
            return {"status": "registration_auth_required", "auth_status": "registration_auth_required"}

        return first_opener, {"preflight_status": "pass", "auth_status": "preflight_passed", "secrets_copied": False}

    first = runner.run_cvpr_event_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r5-event-media",
        opener_factory=first_factory,
        caption_downloader=lambda *_args: pytest.fail("caption queue must wait for terminal page coverage"),
    )

    checkpoint_path = workspace / "storage/download-corpus/event-media/event_media.checkpoint.json"
    assert first["status"] == "blocked"
    assert len(first_calls) == 2
    assert checkpoint_path.is_file()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["counts"]["r4_reused_poster_route_count"] == 2_565
    assert checkpoint["counts"]["terminal_event_route_count"] == 2_566
    assert first_calls[0] not in [row["source_url"] for row in checkpoint["events"] if row["page_status"] == "pending"]
    for artifact in (checkpoint_path, Path(first["receipt_path"])):
        assert SECRET_SENTINEL not in artifact.read_text(encoding="utf-8")

    resumed_calls: list[str] = []

    def resumed_factory(**kwargs: object):
        frozen_urls = kwargs["frozen_urls"]
        assert isinstance(frozen_urls, list)
        # The first completed r5 page is the one bounded resume sentinel; the
        # rest are the still-pending frozen routes.
        assert len(frozen_urls) == 2_693
        assert first_calls[0] in frozen_urls

        def resumed_opener(url: str) -> dict[str, object]:
            resumed_calls.append(url)
            return {"status": "registration_auth_required", "auth_status": "registration_auth_required"}

        return resumed_opener, {"preflight_status": "pass", "auth_status": "preflight_passed", "secrets_copied": False}

    resumed = runner.run_cvpr_event_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r5-event-media",
        resume=True,
        opener_factory=resumed_factory,
        caption_downloader=lambda *_args: pytest.fail("caption queue must wait for terminal page coverage"),
    )

    assert resumed["status"] == "blocked"
    assert resumed_calls == [first_calls[0]]
    sentinel = resumed["receipt"]["checks"]["event_media_session_preflight"]["resume_sentinel"]
    assert sentinel["authenticated_request_count"] == 1
    assert sentinel["status"] == "not_authenticated"


def test_event_media_caption_only_resume_rechecks_session_and_promotes_checkpointed_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A completed page checkpoint is promotable without reopening page URLs."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _, _plan, overlay = registration_media_fixture_inputs()
    completed_overlay = completed_zero_link_overlay(overlay)
    baseline = event_media_fixture_inventory()
    reused_poster_urls = {
        row["source_url"]
        for row in baseline["events"]
        if row["outcome_origin"] == "r4_registration_media"
    }
    # The immutable r4 ledger contains 2,565 poster routes plus eight
    # transport-eligible non-poster virtual endpoints.  Preserve that split so
    # the r5 exact-reuse validator exercises the real overlay boundary.
    transport_poster_rows = [
        row
        for row in completed_overlay["official_event_requests"]
        if row["transport_eligible"]
        and "/poster/" in row["source_url"]
    ]
    assert len(transport_poster_rows) == 2_573
    for row, source_url in zip(transport_poster_rows[:2_565], sorted(reused_poster_urls), strict=True):
        row["source_url"] = source_url
    for index, row in enumerate(transport_poster_rows[2_565:]):
        row["source_url"] = f"https://cvpr.thecvf.com/virtual/2026/events/fixture-{index}"
    completed_overlay["overlay_sha256"] = canonical_registration_media_digest(completed_overlay)
    overlay_path = workspace / OVERLAY_LEDGER_RELATIVE_PATH
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(json.dumps(completed_overlay, sort_keys=True) + "\n", encoding="utf-8")

    baseline["source_lineage"]["r4_registration_media_overlay_sha256"] = completed_overlay["overlay_sha256"]
    baseline["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(baseline)
    monkeypatch.setattr(runner, "_event_media_inventory_inputs", lambda _workspace, **_kwargs: copy.deepcopy(baseline))

    real_verify = runner.verify_stage_contract

    class FailingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "blocked", "blockers": [{"code": "fixture_contract_failure"}]}

    monkeypatch.setattr(runner, "verify_stage_contract", lambda **_kwargs: FailingContract())
    first_page_calls: list[str] = []

    def first_factory(**kwargs: object):
        assert len(kwargs["frozen_urls"]) == 2_693

        def opener(url: str) -> dict[str, object]:
            first_page_calls.append(url)
            return {
                "status": "authenticated",
                "auth_status": "authenticated",
                "http_status": 200,
                "content_type": "text/html",
                "body": b"<main>No direct YouTube link</main>",
            }

        return opener, {
            "preflight_status": "pass",
            "auth_status": "preflight_passed",
            "secrets_copied": False,
        }

    first = runner.run_cvpr_event_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r5-event-media",
        opener_factory=first_factory,
        caption_downloader=lambda *_args: pytest.fail("fixture has no caption candidates"),
    )
    checkpoint_path = workspace / "storage/download-corpus/event-media/event_media.checkpoint.json"
    assert first["status"] == "blocked"
    assert len(first_page_calls) == 2_693
    assert json.loads(checkpoint_path.read_text(encoding="utf-8"))["counts"]["terminal_event_route_count"] == 5_258

    # Restore the strict verifier.  A caption-only resume must perform exactly
    # one fresh authenticated sentinel request and never re-scan the terminal
    # page fanout.
    monkeypatch.setattr(runner, "verify_stage_contract", real_verify)
    resumed_page_calls: list[str] = []

    def resume_factory(**kwargs: object):
        frozen_urls = kwargs["frozen_urls"]
        assert isinstance(frozen_urls, list)
        assert len(frozen_urls) == 1

        def sentinel_open(url: str) -> dict[str, object]:
            resumed_page_calls.append(url)
            return {
                "status": "authenticated",
                "auth_status": "authenticated",
                "http_status": 200,
                "content_type": "text/html",
                "body": b"<main>Fresh authenticated sentinel</main>",
            }

        return sentinel_open, {
            "preflight_status": "pass",
            "auth_status": "preflight_passed",
            "secrets_copied": False,
        }

    resumed = runner.run_cvpr_event_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r5-event-media",
        resume=True,
        opener_factory=resume_factory,
        caption_downloader=lambda *_args: pytest.fail("fixture has no caption candidates"),
    )

    assert resumed["status"] == "succeeded"
    assert resumed["receipt"]["stage_contract"]["status"] == "pass"
    assert len(resumed_page_calls) == 1
    assert resumed["receipt"]["checks"]["event_media_authenticated_request_count"] == 2_694
    assert resumed["receipt"]["checks"]["event_media_resumed_authenticated_request_count"] == 2_693
    assert resumed["receipt"]["checks"]["event_media_session_preflight"]["preflight_status"] == "pass"
    assert resumed["receipt"]["checks"]["event_media_session_preflight"]["resume_sentinel"] == {
        "authenticated_request_count": 1,
        "status": "authenticated",
        "auth_status": "authenticated",
        "http_status": 200,
    }
    assert not checkpoint_path.exists()

    # An inventory can name the right candidate sidecar while the filesystem
    # resolves it to a different video's otherwise valid caption.  The strict
    # stage verifier must reject that mismatch independently of the inventory
    # schema and the correct hash/byte-count facts.
    inventory_path = workspace / "storage/download-corpus/event-media/event_media_inventory.json"
    tampered = json.loads(inventory_path.read_text(encoding="utf-8"))
    r5_row = next(
        row for row in tampered["events"] if row["outcome_origin"] == "r5_authenticated_event_page"
    )
    tampered = apply_event_page_outcome(
        tampered,
        source_url=r5_row["source_url"],
        status="direct_youtube_recovered",
        video_ids=["dQw4w9WgXcQ"],
        auth_status="authenticated",
        http_status=200,
    )
    captions_root = workspace / "storage/download-corpus/event-media/captions"
    captions_root.mkdir(parents=True, exist_ok=True)
    wrong_sidecar = captions_root / "ZZZZZZZZZZZ.en.vtt"
    wrong_sidecar.write_text("WEBVTT\n", encoding="utf-8")
    claimed_sidecar = captions_root / "dQw4w9WgXcQ.en.vtt"
    claimed_sidecar.symlink_to(wrong_sidecar.name)
    tampered = apply_caption_outcomes(
        tampered,
        outcomes=[
            {
                "video_id": "dQw4w9WgXcQ",
                "caption_status": "youtube_caption_available",
                "caption": {
                    "local_path": "storage/download-corpus/event-media/captions/dQw4w9WgXcQ.en.vtt",
                    "mime": "text/vtt",
                    "byte_count": wrong_sidecar.stat().st_size,
                    "sha256": runner._file_sha256(wrong_sidecar),
                    "transcript_source": "youtube_caption_auto",
                    "language": "en",
                },
            }
        ],
    )
    inventory_path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
    checks = runner._event_media_checks(
        inventory=tampered,
        inventory_file_sha256=runner._file_sha256(inventory_path),
        authenticated_request_count=2_694,
        resumed_authenticated_request_count=2_693,
        resume_mode=True,
        preflight=resumed["receipt"]["checks"]["event_media_session_preflight"],
    )
    strict_receipt = copy.deepcopy(resumed["receipt"])
    strict_receipt["checks"] = checks
    strict = runner.verify_stage_contract(
        workspace=workspace,
        run_id="cvpr2026-live-production-r5-event-media",
        stage_id="download-corpus",
        receipt=strict_receipt,
        required_owner_ids=(runner.CVPR_DOWNLOAD_CORPUS_OWNER_ID,),
        mode="live",
        strict_execution=True,
        profile="event-media-coverage",
    )
    assert strict.status == "blocked"
    assert "stage_contract_event_media_inventory_invalid" in {
        blocker["code"] for blocker in strict.blockers
    }


def test_event_media_caption_record_rejects_a_video_id_prefix_sidecar(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    video_id = "ABCDEF12345"
    wrong_sidecar = (
        workspace
        / "storage/download-corpus/event-media/captions"
        / f"{video_id}x.en.vtt"
    )
    wrong_sidecar.parent.mkdir(parents=True, exist_ok=True)
    wrong_sidecar.write_text("WEBVTT\n", encoding="utf-8")

    assert (
        runner._event_media_caption_record(
            workspace=workspace,
            video_id=video_id,
            value={
                "caption_path": wrong_sidecar,
                "transcript_source": "youtube_caption_manual",
                "language": "en",
            },
        )
        is None
    )


def test_event_media_full_scan_is_caption_only_and_rolls_back_on_contract_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    baseline = event_media_fixture_inventory()
    monkeypatch.setattr(runner, "_event_media_inventory_inputs", lambda _workspace, **_kwargs: copy.deepcopy(baseline))
    calls: list[str] = []
    caption_calls: list[str] = []

    class FailingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "blocked", "blockers": [{"code": "fixture_contract_failure"}]}

    monkeypatch.setattr(runner, "verify_stage_contract", lambda **_kwargs: FailingContract())

    def opener_factory(**kwargs: object):
        frozen_urls = kwargs["frozen_urls"]
        assert isinstance(frozen_urls, list)
        assert len(frozen_urls) == 2_693

        def opener(url: str) -> dict[str, object]:
            calls.append(url)
            return {
                "status": "authenticated",
                "auth_status": "authenticated",
                "http_status": 200,
                "content_type": "text/html",
                "body": b"<main>No official YouTube link</main>",
            }

        return opener, {"preflight_status": "pass", "auth_status": "preflight_passed", "secrets_copied": False}

    outcome = runner.run_cvpr_event_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r5-event-media",
        opener_factory=opener_factory,
        caption_downloader=lambda video_id, _destination: caption_calls.append(video_id) or None,
    )

    checkpoint_path = workspace / "storage/download-corpus/event-media/event_media.checkpoint.json"
    inventory_path = workspace / "storage/download-corpus/event-media/event_media_inventory.json"
    assert outcome["status"] == outcome["receipt"]["status"] == "blocked"
    assert len(calls) == 2_693
    assert caption_calls == []
    assert checkpoint_path.is_file()
    assert not inventory_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["counts"]["r4_reused_poster_route_count"] == 2_565
    assert checkpoint["counts"]["terminal_event_route_count"] == 5_258
    assert checkpoint["boundaries"] == {
        "audio_downloaded": False,
        "video_downloaded": False,
        "asr_dispatched": False,
        "ocr_dispatched": False,
        "provider_calls_made": 0,
    }
    assert outcome["receipt"]["checks"]["audio_downloaded"] is False
    assert outcome["receipt"]["checks"]["video_downloaded"] is False
    assert outcome["receipt"]["checks"]["asr_dispatched"] is False
    assert outcome["receipt"]["checks"]["ocr_dispatched"] is False


def test_event_media_caption_queue_uses_a_small_bounded_worker_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    baseline = event_media_fixture_inventory()
    pending_urls = [
        row["source_url"]
        for row in baseline["events"]
        if row["page_status"] == "pending"
    ]
    seeded = apply_event_page_outcomes(
        baseline,
        outcomes=[
            {
                "source_url": source_url,
                "status": "direct_youtube_recovered",
                "video_ids": [f"AbcDef{index:05d}"],
                "auth_status": "authenticated",
                "http_status": 200,
            }
            for index, source_url in enumerate(pending_urls[:4], start=1)
        ],
    )
    seeded = merge_caption_candidates(seeded, r4_media_records=[])
    monkeypatch.setattr(runner, "_event_media_inventory_inputs", lambda _workspace, **_kwargs: copy.deepcopy(seeded))

    class FailingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "blocked", "blockers": [{"code": "fixture_contract_failure"}]}

    monkeypatch.setattr(runner, "verify_stage_contract", lambda **_kwargs: FailingContract())
    active = 0
    peak_active = 0
    lock = Lock()

    def caption_downloader(_video_id: str, _destination: Path) -> None:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return None

    def opener_factory(**_kwargs: object):
        def opener(_url: str) -> dict[str, object]:
            return {
                "status": "authenticated",
                "auth_status": "authenticated",
                "http_status": 200,
                "content_type": "text/html",
                "body": b"<main>No direct YouTube link</main>",
            }

        return opener, {"preflight_status": "pass", "auth_status": "preflight_passed", "secrets_copied": False}

    outcome = runner.run_cvpr_event_media_stage(
        workspace=workspace,
        run_id="cvpr2026-live-production-r5-event-media",
        opener_factory=opener_factory,
        caption_downloader=caption_downloader,
    )

    assert outcome["status"] == "blocked"
    assert peak_active == runner.EVENT_MEDIA_CAPTION_WORKER_COUNT


def test_event_media_caption_pool_checkpoints_completed_jobs_before_slow_jobs_finish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    baseline = event_media_fixture_inventory()
    pending_urls = [
        row["source_url"]
        for row in baseline["events"]
        if row["page_status"] == "pending"
    ]
    completed = apply_event_page_outcomes(
        baseline,
        outcomes=[
            {
                "source_url": source_url,
                "status": "direct_youtube_recovered" if index < 29 else "no_admissible_official_youtube_link",
                "video_ids": [f"AbcDef{index:05d}"] if index < 29 else [],
                "auth_status": "authenticated",
                "http_status": 200,
            }
            for index, source_url in enumerate(pending_urls)
        ],
    )
    completed = merge_caption_candidates(completed, r4_media_records=[])
    monkeypatch.setattr(runner, "_event_media_inventory_inputs", lambda _workspace, **_kwargs: copy.deepcopy(completed))

    checkpoint_path = workspace / "storage/download-corpus/event-media/event_media.checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(completed, sort_keys=True) + "\n", encoding="utf-8")

    class FailingContract:
        def to_dict(self) -> dict[str, object]:
            return {"status": "blocked", "blockers": [{"code": "fixture_contract_failure"}]}

    monkeypatch.setattr(runner, "verify_stage_contract", lambda **_kwargs: FailingContract())
    pending_ids = [row["video_id"] for row in completed["caption_candidates"]]
    slow_ids = set(pending_ids[-4:])
    slow_jobs_started = Event()
    release_slow_jobs = Event()

    def caption_downloader(video_id: str, _destination: Path) -> None:
        if video_id in slow_ids:
            slow_jobs_started.set()
            assert release_slow_jobs.wait(timeout=5)
        return None

    def opener_factory(**kwargs: object):
        assert len(kwargs["frozen_urls"]) == 1

        def opener(_url: str) -> dict[str, object]:
            return {
                "status": "authenticated",
                "auth_status": "authenticated",
                "http_status": 200,
                "content_type": "text/html",
                "body": b"<main>Fresh authenticated sentinel</main>",
            }

        return opener, {"preflight_status": "pass", "auth_status": "preflight_passed", "secrets_copied": False}

    result: dict[str, object] = {}

    def run_stage() -> None:
        result.update(
            runner.run_cvpr_event_media_stage(
                workspace=workspace,
                run_id="cvpr2026-live-production-r5-event-media",
                resume=True,
                opener_factory=opener_factory,
                caption_downloader=caption_downloader,
            )
        )

    worker = Thread(target=run_stage)
    worker.start()
    assert slow_jobs_started.wait(timeout=5)
    for _ in range(100):
        persisted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        terminal_captions = sum(
            row["caption_status"] != "pending" for row in persisted["caption_candidates"]
        )
        if terminal_captions >= 25:
            break
        time.sleep(0.01)
    else:
        pytest.fail("completed caption batch was not checkpointed before slow jobs finished")

    release_slow_jobs.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert result["status"] == "blocked"


def test_successful_source_recovery_promotes_canonical_receipt_and_preserves_prior_bytes(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    run_id = "receipt-promotion"
    blocked = production_source_discovery_stage.write_source_stage_receipt(
        runner._production_context(workspace, run_id),
        owner_id=runner.CVPR_SOURCE_DISCOVERY_OWNER_ID,
        io=runner._stage_io(),
        status="blocked",
        collection={},
        owner_result=None,
        inventory=None,
        blockers=[{"code": "fixture_blocker"}],
        frontier_authority=None,
        stage_contract=None,
    )
    canonical_path = Path(blocked["receipt_path"])
    blocked_bytes = canonical_path.read_bytes()

    succeeded = production_source_discovery_stage.write_source_stage_receipt(
        runner._production_context(workspace, run_id),
        owner_id=runner.CVPR_SOURCE_DISCOVERY_OWNER_ID,
        io=runner._stage_io(),
        status="succeeded",
        collection={},
        owner_result={"provider_calls_made": 0, "provider_receipt_paths": [], "actual_cost_usd": 0.0},
        inventory=None,
        blockers=[],
        frontier_authority=None,
        stage_contract={"status": "pass"},
    )

    assert Path(succeeded["receipt_path"]) == canonical_path
    assert json.loads(canonical_path.read_text(encoding="utf-8"))["status"] == "succeeded"
    superseded = list(canonical_path.parent.glob("stage-02-source-discovery.superseded-*.json"))
    assert len(superseded) == 1
    assert superseded[0].read_bytes() == blocked_bytes


def write_corpus_freeze_fixture(
    runner: object,
    workspace: Path,
    *,
    run_id: str = "cvpr2026-live-production-r3-20260716",
) -> tuple[Path, dict[str, object]]:
    """Write a compact r3-shaped inventory and its succeeded source receipt."""

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
    inventory: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "stage_id": "source-discovery",
        "status": "prepared",
        "blockers": [],
        "source_families": {
            "program-schedule": {"status": "live_valid", "exclusion_id": None},
            "virtual-paper-media-posture": {
                "status": "locked_excluded",
                "exclusion_id": "paywalled_official_video",
            },
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
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = workspace / "storage" / "workflow-runs" / run_id / "stage-02-source-discovery.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage_id": "source-discovery",
                "stage_order": 2,
                "status": "succeeded",
                "mode": "live",
                "blockers": [],
                "canonical_inventory": {
                    "path": str(inventory_path),
                    "inventory_sha256": inventory["inventory_sha256"],
                },
                "stage_contract": {"status": "pass"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stamp_upstream_stage_receipts(receipt_path.parent)
    return inventory_path, inventory


def run_fixture_corpus_freeze(runner: object, workspace: Path, inventory: dict[str, object]) -> dict[str, object]:
    return runner.run_corpus_freeze_stage(
        workspace=workspace,
        run_id=str(inventory["run_id"]),
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
    )


def test_corpus_freeze_verifies_inventory_with_the_phase2_publication_digest_contract() -> None:
    runner = load_runner_module()
    inventory = {
        "title": "R\u00e9sum\u00e9 evidence",
        "source_records": ["cvpr2026:source:fixture"],
    }
    inventory["inventory_sha256"] = source_discovery_canonical_json_digest(inventory)

    assert runner._corpus_inventory_digest(inventory) == inventory["inventory_sha256"]


def test_corpus_freeze_owner_creates_complete_digest_bound_program_item_ledger(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)

    outcome = run_fixture_corpus_freeze(runner, workspace, inventory)
    manifest = json.loads(
        (workspace / "storage" / "corpus-freeze" / "corpus_freeze.json").read_text(encoding="utf-8")
    )
    report = (workspace / "storage" / "corpus-freeze" / "corpus_freeze_report.md").read_text(encoding="utf-8")

    assert outcome["status"] == "succeeded"
    assert outcome["receipt"]["stage_contract"]["status"] == "pass"
    assert outcome["receipt"]["provider_calls_made"] == 0
    assert outcome["receipt"]["downloads_made"] == 0
    assert manifest["source_lineage"]["inventory_sha256"] == inventory["inventory_sha256"]
    assert [item["program_item_id"] for item in manifest["program_items"]] == sorted(
        item["program_item_id"] for item in inventory["program_items"]
    )
    assert {item["selection_status"] for item in manifest["program_items"]} == {"included"}
    assert manifest["counts"] == {
        "discovered_program_items": 2,
        "included_program_items": 2,
        "operator_excluded_program_items": 0,
        "policy_excluded_program_items": 0,
        "policy_exclusion_postures": 1,
    }
    assert manifest["policy_exclusions"] == [
        {
            "exclusion_id": "paywalled_official_video",
            "family_id": "virtual-paper-media-posture",
            "reason": "locked source policy: official virtual-paper video is paywalled",
            "selection_effect": "official_video_acquisition_excluded_without_removing_program_items",
        }
    ]
    assert all(
        item["planned_acquisition"] or item["unreachable_source_path"]
        for item in manifest["program_items"]
    )
    assert "Visible program-item IDs: 2 of 2" in report
    assert "paywalled_official_video" in report


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda workspace, inventory: inventory["program_items"].append(
                dict(inventory["program_items"][0])
            ),
            "duplicate_program_item_id",
        ),
        (
            lambda workspace, inventory: inventory["program_items"][0].__setitem__(
                "source_record_ids", ["cvpr2026:source:program-schedule:unknown"]
            ),
            "program_item_source_record_evidence_missing",
        ),
    ],
    ids=("duplicate-item", "evidence-less-item"),
)
def test_corpus_freeze_blocks_invalid_pinned_inventory_boundaries(
    tmp_path: Path,
    mutate: Callable[[Path, dict[str, object]], object],
    expected_code: str,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)
    mutate(workspace, inventory)
    inventory["inventory_sha256"] = source_discovery_canonical_json_digest(
        {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    )
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_receipt_path = (
        workspace / "storage" / "workflow-runs" / str(inventory["run_id"]) / "stage-02-source-discovery.json"
    )
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    source_receipt["canonical_inventory"]["inventory_sha256"] = inventory["inventory_sha256"]
    source_receipt_path.write_text(json.dumps(source_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outcome = run_fixture_corpus_freeze(runner, workspace, inventory)

    assert outcome["status"] == "blocked"
    assert expected_code in {blocker["code"] for blocker in outcome["receipt"]["blockers"]}
    assert outcome["receipt"]["provider_calls_made"] == 0
    assert outcome["receipt"]["downloads_made"] == 0


def test_corpus_freeze_rejects_a_changed_run_before_reading_any_source_input(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)

    outcome = runner.run_corpus_freeze_stage(
        workspace=workspace,
        run_id="wrong-run",
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
    )

    assert outcome["status"] == "blocked"
    assert {blocker["code"] for blocker in outcome["receipt"]["blockers"]} == {
        "pinned_source_run_mismatch"
    }


def test_corpus_freeze_rejects_invalid_operator_control_and_silent_cap_before_selection(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)
    exclusions_path = workspace / "storage" / "corpus-freeze" / "operator_exclusions.json"
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_path.write_text("{not-json", encoding="utf-8")

    malformed = run_fixture_corpus_freeze(runner, workspace, inventory)
    capped = runner.run_corpus_freeze_stage(
        workspace=workspace,
        run_id=str(inventory["run_id"]),
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
        command_args=("--max-items", "1"),
    )

    assert malformed["status"] == "blocked"
    assert "operator_exclusions_invalid_json" in {
        blocker["code"] for blocker in malformed["receipt"]["blockers"]
    }
    assert capped["status"] == "blocked"
    assert "silent_item_cap_rejected" in {blocker["code"] for blocker in capped["receipt"]["blockers"]}


def test_corpus_freeze_is_deterministic_and_preserves_prior_blocked_receipt(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)

    blocked = runner.run_corpus_freeze_stage(
        workspace=workspace,
        run_id=str(inventory["run_id"]),
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
        command_args=("--limit=1",),
    )
    prior_path = Path(blocked["receipt_path"])
    prior_bytes = prior_path.read_bytes()
    first = run_fixture_corpus_freeze(runner, workspace, inventory)
    second = run_fixture_corpus_freeze(runner, workspace, inventory)
    manifest = json.loads(
        (workspace / "storage" / "corpus-freeze" / "corpus_freeze.json").read_text(encoding="utf-8")
    )

    assert first["status"] == second["status"] == "succeeded"
    assert first["manifest"]["selection_sha256"] == second["manifest"]["selection_sha256"]
    assert manifest["selection_sha256"] == first["manifest"]["selection_sha256"]
    superseded = list(prior_path.parent.glob("stage-03-corpus-freeze.superseded-*.json"))
    assert any(path.read_bytes() == prior_bytes for path in superseded)


def valid_operator_exclusion(inventory: dict[str, object], *, target: str | None = None) -> dict[str, object]:
    item = dict(inventory["program_items"][0])
    if target is not None:
        item["program_item_id"] = target
    return {
        "program_item_id": item["program_item_id"],
        "rationale": "fixture operator scope decision",
        "decision_timestamp": "2026-07-16T00:00:00Z",
        "decision_context": "fixture local test",
        "source_evidence": list(item["source_record_ids"]),
        "downstream_effect": "exclude from Phase 4 acquisition",
    }


def test_corpus_freeze_applies_only_source_bound_operator_exclusions(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)
    exclusion = valid_operator_exclusion(inventory)
    exclusions_path = workspace / "storage" / "corpus-freeze" / "operator_exclusions.json"
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_inventory_sha256": inventory["inventory_sha256"],
                "exclusions": [exclusion],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    outcome = run_fixture_corpus_freeze(runner, workspace, inventory)
    selection = {item["program_item_id"]: item for item in outcome["manifest"]["program_items"]}

    assert outcome["status"] == "succeeded"
    assert selection[exclusion["program_item_id"]]["selection_status"] == "operator_excluded"
    assert selection[exclusion["program_item_id"]]["planned_acquisition"] == []
    assert outcome["manifest"]["operator_exclusions"]["validation_status"] == "pass"


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda control, inventory: control.__setitem__("source_inventory_sha256", "f" * 64),
            "operator_exclusions_stale",
        ),
        (
            lambda control, inventory: control["exclusions"].__setitem__(
                0, valid_operator_exclusion(inventory, target="cvpr2026:program-item:unknown")
            ),
            "operator_exclusion_unknown_target",
        ),
        (
            lambda control, inventory: control.__setitem__(
                "exclusions", [control["exclusions"][0], dict(control["exclusions"][0])]
            ),
            "operator_exclusion_duplicate_target",
        ),
    ],
    ids=("stale", "unknown-target", "duplicate-target"),
)
def test_corpus_freeze_rejects_stale_unknown_or_duplicate_operator_exclusions(
    tmp_path: Path,
    mutate: Callable[[dict[str, object], dict[str, object]], object],
    expected_code: str,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)
    control: dict[str, object] = {
        "schema_version": 1,
        "source_inventory_sha256": inventory["inventory_sha256"],
        "exclusions": [valid_operator_exclusion(inventory)],
    }
    mutate(control, inventory)
    path = workspace / "storage" / "corpus-freeze" / "operator_exclusions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outcome = run_fixture_corpus_freeze(runner, workspace, inventory)

    assert outcome["status"] == "blocked"
    assert expected_code in {blocker["code"] for blocker in outcome["receipt"]["blockers"]}


def test_corpus_freeze_rejects_a_nonpassing_source_stage_receipt(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)
    source_receipt_path = (
        workspace / "storage" / "workflow-runs" / str(inventory["run_id"]) / "stage-02-source-discovery.json"
    )
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    source_receipt["status"] = "blocked"
    source_receipt_path.write_text(json.dumps(source_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outcome = run_fixture_corpus_freeze(runner, workspace, inventory)

    assert outcome["status"] == "blocked"
    assert "source_receipt_not_succeeded" in {blocker["code"] for blocker in outcome["receipt"]["blockers"]}


def test_download_corpus_seeds_a_complete_lineage_bound_ledger_before_any_transport(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)
    freeze = run_fixture_corpus_freeze(runner, workspace, inventory)
    opener_calls: list[str] = []

    outcome = runner.run_download_corpus_stage(
        workspace=workspace,
        run_id=str(inventory["run_id"]),
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
        expected_program_item_count=2,
        available_bytes=1_000_000,
        opener_factory=lambda: opener_calls.append("constructed"),
    )
    ledger = json.loads(
        (workspace / "storage" / "download-corpus" / "download_corpus.json").read_text(encoding="utf-8")
    )

    assert freeze["status"] == "succeeded"
    assert outcome["status"] == "blocked"  # Later public/talk/recovery lanes are intentionally incomplete.
    assert opener_calls == []
    assert outcome["receipt"]["downloads_made"] == 0
    assert outcome["receipt"]["provider_calls_made"] == 0
    assert ledger["lineage"]["phase_02_inventory_sha256"] == inventory["inventory_sha256"]
    assert ledger["lineage"]["phase_03_selection_sha256"] == freeze["manifest"]["selection_sha256"]
    assert len(ledger["program_item_results"]) == 2
    assert ledger["manifest_sha256"]


def test_download_corpus_public_dispatcher_records_local_artifacts_but_keeps_future_lanes_blocked(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)
    run_fixture_corpus_freeze(runner, workspace, inventory)
    opener_calls: list[str] = []

    class Response:
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

    def opener(url: str) -> Response:
        opener_calls.append(url)
        return Response(url)

    outcome = runner.run_download_corpus_stage(
        workspace=workspace,
        run_id=str(inventory["run_id"]),
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
        expected_program_item_count=2,
        available_bytes=1_000_000,
        execute_public_lane=True,
        opener_factory=lambda: opener,
    )
    ledger = json.loads(
        (workspace / "storage" / "download-corpus" / "download_corpus.json").read_text(encoding="utf-8")
    )

    assert opener_calls
    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["downloads_made"] == ledger["counts"]["artifact_count"]
    assert ledger["public_acquisition_complete"] is True
    assert all(
        row.get("artifact_id") or row.get("blocker", {}).get("code") == "policy_excluded_official_video"
        for row in ledger["program_item_results"]
    )
    assert outcome["receipt"]["checks"]["complete_result_or_blocker_accounting"] is True
    assert "talk_harvest_lane_incomplete" in {blocker["code"] for blocker in outcome["receipt"]["blockers"]}
    assert "focused_recovery_lane_incomplete" in {blocker["code"] for blocker in outcome["receipt"]["blockers"]}
    assert outcome["receipt"]["provider_calls_made"] == 0


def test_download_corpus_resume_reuses_valid_artifact_without_reopening_its_public_url(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)
    run_fixture_corpus_freeze(runner, workspace, inventory)

    class Response:
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

    first_calls: list[str] = []
    first = runner.run_download_corpus_stage(
        workspace=workspace,
        run_id=str(inventory["run_id"]),
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
        expected_program_item_count=2,
        available_bytes=1_000_000,
        execute_public_lane=True,
        opener_factory=lambda: lambda url: first_calls.append(url) or Response(url),
    )
    assert first_calls
    assert first["receipt"]["downloads_made"] == len(first_calls)

    resumed_calls: list[str] = []
    resumed = runner.run_download_corpus_stage(
        workspace=workspace,
        run_id=str(inventory["run_id"]),
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
        expected_program_item_count=2,
        available_bytes=1_000_000,
        execute_public_lane=True,
        opener_factory=lambda: lambda url: resumed_calls.append(url) or Response(url),
    )

    assert resumed_calls == []
    assert resumed["receipt"]["checks"]["complete_result_or_blocker_accounting"] is True
    ledger = json.loads((workspace / "storage" / "download-corpus" / "download_corpus.json").read_text(encoding="utf-8"))
    assert ledger["resume"]["status"] == "reused"
    assert ledger["resume"]["reused_artifact_count"] == len(first_calls)


@pytest.mark.parametrize("available_bytes", [None, 0])
def test_download_corpus_blocks_unknown_or_insufficient_capacity_before_opener_creation(
    tmp_path: Path,
    available_bytes: int | None,
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)
    run_fixture_corpus_freeze(runner, workspace, inventory)
    opener_calls: list[str] = []

    outcome = runner.run_download_corpus_stage(
        workspace=workspace,
        run_id=str(inventory["run_id"]),
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
        expected_program_item_count=2,
        available_bytes=available_bytes,
        opener_factory=lambda: opener_calls.append("constructed"),
    )

    assert outcome["status"] == "blocked"
    assert opener_calls == []
    assert outcome["receipt"]["downloads_made"] == 0
    assert outcome["receipt"]["provider_calls_made"] == 0
    assert any(
        blocker["code"] in {"disk_capacity_unknown", "disk_capacity_insufficient"}
        for blocker in outcome["receipt"]["blockers"]
    )


def test_download_corpus_rejects_changed_selection_before_writing_or_opening(tmp_path: Path) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    _inventory_path, inventory = write_corpus_freeze_fixture(runner, workspace)
    freeze = run_fixture_corpus_freeze(runner, workspace, inventory)
    corpus_path = workspace / "storage" / "corpus-freeze" / "corpus_freeze.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["program_items"][0]["program_item_id"] = "cvpr2026:program-item:mutated"
    corpus_path.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    opener_calls: list[str] = []

    outcome = runner.run_download_corpus_stage(
        workspace=workspace,
        run_id=str(inventory["run_id"]),
        pinned_run_id=str(inventory["run_id"]),
        pinned_inventory_sha256=str(inventory["inventory_sha256"]),
        expected_program_item_count=2,
        available_bytes=1_000_000,
        opener_factory=lambda: opener_calls.append("constructed"),
    )

    assert freeze["status"] == "succeeded"
    assert outcome["status"] == "blocked"
    assert not (workspace / "storage" / "download-corpus" / "download_corpus.json").exists()
    assert opener_calls == []
    assert "phase_03_selection_digest_invalid" in {
        blocker["code"] for blocker in outcome["receipt"]["blockers"]
    }


def _write_extractable_pdf(path: Path, text: str) -> bytes:
    """Write a genuinely pypdf-extractable PDF (not a fake ``%PDF-`` header).

    A prior fixture wrote ``b"%PDF-1.4\\n" + b"some text"`` -- not a valid PDF
    at all -- which happened to "work" only because the code under test back
    then never actually opened the file with a PDF parser (it just checked
    for a pre-seeded text sidecar). Now that paper text is genuinely
    extracted from the PDF bytes, the fixture must be a real PDF or the
    extraction call itself would (correctly) fail.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(str(path))
    document.close()
    return path.read_bytes()


def _write_corpus_inventory_index_predecessors(
    runner: object,
    workspace: Path,
    run_id: str,
    *,
    include_download_corpus: bool = True,
) -> dict[str, object]:
    """Write the four digest-bound Stage 6 predecessors for a fixture run.

    Mirrors D-02's admitted input list: the download-corpus Stage-4 receipt +
    ledger, the corpus-freeze ledger, the r5 event-media caption inventory,
    and the workshops paper-download receipt. Also writes a placeholder
    Stage-5 (transcription-ocr) receipt: the stage_owner_registry's
    pre-existing ``required_receipts`` for corpus-inventory-index lists that
    path as a bare existence gate, independent of this owner's own D-02
    admission list, which this plan does not change.
    """
    # Fixture papers hold a few KB of real text; the production floor for
    # image-only PDFs would exclude them all.
    runner.PAPER_CURATION_MIN_TEXT_BYTES = 0

    _inventory_path, source_inventory = write_corpus_freeze_fixture(runner, workspace, run_id=run_id)
    freeze = run_fixture_corpus_freeze(runner, workspace, source_inventory)
    assert freeze["status"] == "succeeded", freeze

    main_paper_pdf_path = workspace / "storage" / "download-corpus" / "artifacts" / "paper_pdf" / "fixture-main-paper.pdf"
    main_paper_pdf_bytes = _write_extractable_pdf(
        main_paper_pdf_path,
        "Fixture Main-Conference Paper\nSection A\n"
        + ("Main-conference evidence covers reasoning agents and benchmark evaluation. " * 20),
    )
    main_paper_sha256 = hashlib.sha256(main_paper_pdf_bytes).hexdigest()
    main_paper_request_key = "fixture-main-paper-request-key"

    if include_download_corpus:
        ledger_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "receipt_type": "conference_pipeline.cvpr2026_download_corpus",
                    "stage_id": "download-corpus",
                    "status": "succeeded",
                    "artifacts": [
                        {
                            "artifact_kind": "paper_pdf",
                            "local_path": str(main_paper_pdf_path.relative_to(workspace)),
                            "sha256": main_paper_sha256,
                            "request_key": main_paper_request_key,
                        }
                    ],
                    # Real binding field (verified against the actual CVPR2026
                    # on-disk artifacts): the download-corpus ledger records
                    # the source-discovery inventory digest it was built
                    # against, which Stage 6's new
                    # source_discovery_inventory_stale gate compares against
                    # the live source_discovery_inventory.json's own
                    # inventory_sha256.
                    "lineage": {
                        "phase_02_inventory_sha256": str(source_inventory["inventory_sha256"])
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        ledger_sha256 = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
        stage4_receipt_path = (
            workspace / "storage" / "workflow-runs" / run_id / "stage-04-download-corpus.json"
        )
        stage4_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        stage4_receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "stage_id": "download-corpus",
                    "stage_order": 4,
                    "mode": "live",
                    "status": "succeeded",
                    "blockers": [],
                    "checks": {"download_corpus_ledger_sha256": ledger_sha256},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    stage5_receipt_path = workspace / "storage" / "workflow-runs" / run_id / "stage-05-transcription-ocr.json"
    stage5_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    stage5_receipt_path.write_text(
        json.dumps(
            {
                "stage_id": "transcription-ocr",
                "stage_order": 5,
                "mode": "live",
                "status": "succeeded",
                "blockers": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    captions_dir = workspace / "storage" / "download-corpus" / "event-media" / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)
    caption_text = "WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n" + (
        "Speaker discusses reasoning agents, evaluation, and benchmark coverage. " * 10
    )
    caption_path = captions_dir / "fixture-video-0001.en.vtt"
    caption_path.write_text(caption_text, encoding="utf-8")
    caption_sha256 = hashlib.sha256(caption_path.read_bytes()).hexdigest()
    candidates: list[dict[str, object]] = [
        {
            "video_id": "fixture-video-0001",
            "caption_status": "youtube_caption_available",
            "caption": {
                "local_path": str(caption_path.relative_to(workspace)),
                "sha256": caption_sha256,
                "byte_count": caption_path.stat().st_size,
                "language": "en",
                "mime": "text/vtt",
            },
            "provenance": ["r5_official_event_page"],
            "source_event_urls": ["https://cvpr.thecvf.com/virtual/2026/poster/1"],
        }
    ]
    for index in range(3):
        candidates.append(
            {
                "video_id": f"fixture-unavailable-{index:04d}",
                "caption_status": "youtube_caption_unavailable",
                "caption": None,
                "provenance": ["r4_registration_media", "r5_official_event_page"],
                "source_event_urls": [
                    f"https://cvpr.thecvf.com/virtual/2026/poster/{index + 2}"
                ],
            }
        )
    event_media_inventory: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_event_media_inventory",
        "run_id": runner.R5_EVENT_MEDIA_RUN_ID,
        "caption_candidates": candidates,
        "counts": {"caption_candidate_count": len(candidates)},
        "coverage": {},
        "boundaries": {},
        "events": [
            {
                "source_url": url,
                "title": f"Fixture poster {url.rsplit('/', 1)[-1]}",
                "event_category": "poster",
                "route_family": "poster",
            }
            for candidate in candidates
            for url in candidate["source_event_urls"]
        ],
        "source_lineage": {},
    }
    event_media_inventory["event_media_inventory_sha256"] = hashlib.sha256(
        json.dumps(event_media_inventory, sort_keys=True).encode("utf-8")
    ).hexdigest()
    event_media_path = (
        workspace / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json"
    )
    event_media_path.parent.mkdir(parents=True, exist_ok=True)
    event_media_path.write_text(
        json.dumps(event_media_inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    venueid = "thecvf.com/CVPR/2026/Workshop/Fixture"
    note_id = "fixturepaper01"
    slug = slugify_venueid(venueid)
    workshop_dir = workspace / "storage" / "workshops" / "papers" / slug
    workshop_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = workshop_dir / f"{note_id}.pdf"
    pdf_bytes = _write_extractable_pdf(
        pdf_path,
        "Fixture Workshop Paper\nSection A\n"
        + ("Workshop evidence covers reasoning agents and benchmark evaluation. " * 20),
    )
    pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    receipts_dir = workshop_dir / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    (receipts_dir / f"{note_id}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.workshop_paper_receipt",
                "note_id": note_id,
                "venueid": venueid,
                "status": "succeeded",
                "sha256": pdf_sha256,
                "title": "Fixture Workshop Paper",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    download_receipt_path = workspace / "storage" / "workshops" / "download_receipt.json"
    download_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    download_receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.workshop_paper_download_receipt",
                "stage_id": "acquisition.workshop_paper_download",
                "status": "succeeded",
                "output_dir": "storage/workshops/papers",
                "per_workshop": [
                    {
                        "venueid": venueid,
                        "status": "succeeded",
                        "paper_count_expected": 1,
                        "downloaded": 1,
                        "resume_skipped": 0,
                        "available": 1,
                        "unreachable": 0,
                    }
                ],
                "totals": {"attempted": 1, "succeeded": 1},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    stamp_upstream_stage_receipts(workspace / "storage" / "workflow-runs" / run_id)
    return {
        "caption_path": caption_path,
        "caption_sha256": caption_sha256,
        "pdf_path": pdf_path,
        "pdf_sha256": pdf_sha256,
        "workshop_dir": workshop_dir,
        "note_id": note_id,
        "venueid": venueid,
        "event_media_inventory": event_media_inventory,
        "main_paper_pdf_path": main_paper_pdf_path,
        "main_paper_sha256": main_paper_sha256,
        "main_paper_request_key": main_paper_request_key,
        "source_inventory": source_inventory,
    }


def test_corpus_inventory_index_blocks_on_missing_download_corpus_receipt(tmp_path: Path) -> None:
    """(a) Blocked when the download-corpus Stage-4 receipt is missing/not-succeeded."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-missing-stage4"
    _write_corpus_inventory_index_predecessors(
        runner, workspace, run_id, include_download_corpus=False
    )

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "blocked"
    assert any(
        blocker["code"] == "download_corpus_stage_not_succeeded"
        for blocker in outcome["receipt"]["blockers"]
    )
    assert not (workspace / "storage" / "corpus" / "corpus_inventory.json").exists()


def _source_discovery_inventory_fixture_path(workspace: Path) -> Path:
    return workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"


def test_corpus_inventory_index_blocks_on_missing_source_discovery_inventory(
    tmp_path: Path,
) -> None:
    """A missing source_discovery_inventory.json fails closed with a named,
    corpus-inventory-index-scoped blocker -- the new Stage 6 input dependency
    (RESEARCH change-point 2) is not silently skipped."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-source-discovery-missing"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)
    _source_discovery_inventory_fixture_path(workspace).unlink()

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "blocked"
    matching = [
        b for b in outcome["receipt"]["blockers"] if b["code"] == "source_discovery_inventory_missing"
    ]
    assert matching, outcome["receipt"]["blockers"]
    assert matching[0]["affected_stage"] == "corpus-inventory-index"


def test_corpus_inventory_index_blocks_on_invalid_source_discovery_inventory(
    tmp_path: Path,
) -> None:
    """An unparseable/non-object source_discovery_inventory.json fails closed
    with its own named blocker, distinct from 'missing'."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-source-discovery-invalid"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)
    _source_discovery_inventory_fixture_path(workspace).write_text("not json {{{", encoding="utf-8")

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "blocked"
    matching = [
        b for b in outcome["receipt"]["blockers"] if b["code"] == "source_discovery_inventory_invalid"
    ]
    assert matching, outcome["receipt"]["blockers"]
    assert matching[0]["affected_stage"] == "corpus-inventory-index"


def test_corpus_inventory_index_blocks_on_stale_source_discovery_inventory(
    tmp_path: Path,
) -> None:
    """Drop-and-fail proof: a source_discovery_inventory.json whose
    inventory_sha256 no longer matches the download-corpus ledger's recorded
    lineage.phase_02_inventory_sha256 binding is rejected as stale -- proves
    the gate can actually fail, not just always pass."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-source-discovery-stale"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)
    inventory_path = _source_discovery_inventory_fixture_path(workspace)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    # Mutate the inventory's content so its real digest changes, without
    # updating the download-corpus ledger's pinned lineage binding -- exactly
    # the shape of a superseded/re-run source-discovery inventory.
    inventory["source_records"].append(
        {
            "source_record_id": "cvpr2026:source:program-schedule:fixture-tampered",
            "family_id": "program-schedule",
            "snapshot_sha256": "c" * 64,
            "aliases": ["https://fixtures.example/tampered"],
        }
    )
    inventory["inventory_sha256"] = "0" * 64  # deliberately wrong, no longer self-consistent either
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "blocked"
    matching = [
        b for b in outcome["receipt"]["blockers"] if b["code"] == "source_discovery_inventory_stale"
    ]
    assert matching, outcome["receipt"]["blockers"]
    assert matching[0]["affected_stage"] == "corpus-inventory-index"
    assert matching[0]["observed_inventory_sha256"] == "0" * 64


def test_corpus_inventory_index_healthy_fixture_adds_source_discovery_sha256_to_receipt_binding(
    tmp_path: Path,
) -> None:
    """A healthy (matching) source_discovery_inventory.json adds its resolved
    path and sha256 to the Stage 6 receipt's input-binding evidence."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-source-discovery-healthy"
    predecessors = _write_corpus_inventory_index_predecessors(runner, workspace, run_id)

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    checks = outcome["receipt"]["checks"]
    assert checks["source_discovery_inventory_status"] == "ok"
    assert checks["source_discovery_inventory_sha256"] == str(
        predecessors["source_inventory"]["inventory_sha256"]
    )
    assert checks["source_discovery_inventory_path"] == str(
        runner.SOURCE_DISCOVERY_INVENTORY_PATH
    )


def test_corpus_inventory_index_receipt_has_source_metadata_coverage_with_three_family_keys(
    tmp_path: Path,
) -> None:
    """The Stage 6 receipt's source_metadata_coverage carries exactly the
    three source-family classes, never a blended aggregate."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-source-metadata-coverage"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    coverage = outcome["receipt"]["checks"]["source_metadata_coverage"]
    assert set(coverage) == {"papers_main", "papers_workshop", "artifacts"}
    for bucket in coverage.values():
        assert "total" in bucket and "title_resolved" in bucket and "authors_resolved" in bucket


def _write_stage4_ledger_and_sibling_receipt(
    runner: object,
    workspace: Path,
    *,
    sibling_run_id: str,
    ledger_sha256: str | None = None,
    phase_02_inventory_sha256: str | None = None,
) -> str:
    """Write the workspace-level Stage-4 ledger plus a receipt under a DIFFERENT run-id.

    Mirrors this milestone's real history: every phase after the initial
    discovery run executed under a freshly-suffixed run-id rather than
    reusing one literal run-id for every stage. Returns the ledger's real
    ``manifest_sha256`` (or the caller-supplied one, for the stale-digest
    negative test) -- the same canonical, volatile-field-excluding digest
    the real Stage-4 receipt declares, never a raw file-bytes hash.
    """

    ledger = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_download_corpus",
        "stage_id": "download-corpus",
        "status": "succeeded",
        "artifacts": [],
    }
    if phase_02_inventory_sha256 is not None:
        ledger["lineage"] = {"phase_02_inventory_sha256": phase_02_inventory_sha256}
    real_ledger_sha256 = runner.canonical_ledger_digest(ledger)
    ledger["manifest_sha256"] = real_ledger_sha256
    ledger_path = workspace / "storage" / "download-corpus" / "download_corpus.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    declared_sha256 = ledger_sha256 if ledger_sha256 is not None else real_ledger_sha256
    sibling_receipt_path = (
        workspace / "storage" / "workflow-runs" / sibling_run_id / "stage-04-download-corpus.json"
    )
    sibling_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    sibling_receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": sibling_run_id,
                "stage_id": "download-corpus",
                "stage_order": 4,
                "mode": "live",
                "status": "succeeded",
                "blockers": [],
                "checks": {"download_corpus_ledger_sha256": declared_sha256},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stamp_upstream_stage_receipts(sibling_receipt_path.parent)
    return real_ledger_sha256


def test_corpus_inventory_index_resolves_stage4_receipt_from_a_sibling_run_id(tmp_path: Path) -> None:
    """A succeeded Stage-4 receipt under a DIFFERENT run-id is admitted when its declared
    ledger digest matches the current workspace ledger -- matching how this milestone has
    actually always executed (a fresh run-id per amendment), without pretending the
    invoked run performed Stage-4 acquisition it did not (the resolved run-id is
    disclosed in the stage-06 receipt's checks, never silently hidden)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-cross-run-stage4"
    sibling_run_id = "test-corpus-inventory-cross-run-stage4-r5-sibling"
    predecessors = _write_corpus_inventory_index_predecessors(
        runner, workspace, run_id, include_download_corpus=False
    )
    _write_stage4_ledger_and_sibling_receipt(
        runner,
        workspace,
        sibling_run_id=sibling_run_id,
        phase_02_inventory_sha256=str(predecessors["source_inventory"]["inventory_sha256"]),
    )

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "succeeded", outcome["receipt"].get("blockers")
    assert not any(
        blocker["code"] == "download_corpus_stage_not_succeeded"
        for blocker in outcome["receipt"].get("blockers", [])
    )
    checks = outcome["receipt"]["checks"]
    assert checks["download_corpus_stage_receipt_resolved_run_id"] == sibling_run_id


def test_corpus_inventory_index_rejects_sibling_receipt_with_stale_ledger_digest(tmp_path: Path) -> None:
    """A sibling-run Stage-4 receipt whose declared ledger digest does NOT match the
    current workspace ledger must still block -- proves the cross-run search only
    accepts evidence describing today's actual corpus, never a stale/superseded one."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-cross-run-stale"
    sibling_run_id = "test-corpus-inventory-cross-run-stale-r5-sibling"
    _write_corpus_inventory_index_predecessors(
        runner, workspace, run_id, include_download_corpus=False
    )
    _write_stage4_ledger_and_sibling_receipt(
        runner, workspace, sibling_run_id=sibling_run_id, ledger_sha256="0" * 64
    )

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "blocked"
    assert any(
        blocker["code"] == "download_corpus_stage_not_succeeded"
        for blocker in outcome["receipt"]["blockers"]
    )


def test_corpus_inventory_index_resolves_transcription_ocr_from_differently_suffixed_sibling(
    tmp_path: Path,
) -> None:
    """The canonical stage-05-transcription-ocr.json is still status:blocked (an
    early, unresolved attempt), but a later repair step recorded the real succeeded
    outcome in a differently-suffixed sibling file without ever overwriting the
    original -- this is the exact live-incident shape. Stage 6 must admit the
    succeeded sibling and disclose exactly which file it resolved to."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-transcription-ocr-sibling"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)

    canonical_path = (
        workspace / "storage" / "workflow-runs" / run_id / "stage-05-transcription-ocr.json"
    )
    canonical_path.write_text(
        json.dumps(
            {
                "stage_id": "transcription-ocr",
                "stage_order": 5,
                "mode": "live",
                "status": "blocked",
                "blockers": [{"code": "transcription_ocr_adapter_result_invalid"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    finalization_path = canonical_path.with_name(
        "stage-05-transcription-ocr.receipt-finalization.json"
    )
    finalization_path.write_text(
        json.dumps(
            {
                "stage_id": "transcription-ocr",
                "stage_order": 5,
                "mode": "live",
                "status": "succeeded",
                "blockers": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "succeeded", outcome["receipt"].get("blockers")
    checks = outcome["receipt"]["checks"]
    assert checks["transcription_ocr_stage_receipt_resolved_run_id"] == run_id
    assert (
        checks["transcription_ocr_stage_receipt_resolved_filename"]
        == "stage-05-transcription-ocr.receipt-finalization.json"
    )


def test_corpus_inventory_index_blocks_when_no_succeeded_transcription_ocr_receipt_exists(
    tmp_path: Path,
) -> None:
    """No succeeded stage-05-transcription-ocr* receipt anywhere in the workspace
    must block, not silently pass -- this is a bare existence/status gate, not
    something Stage 6 can infer or fabricate."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-transcription-ocr-missing"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)

    canonical_path = (
        workspace / "storage" / "workflow-runs" / run_id / "stage-05-transcription-ocr.json"
    )
    canonical_path.write_text(
        json.dumps(
            {
                "stage_id": "transcription-ocr",
                "stage_order": 5,
                "mode": "live",
                "status": "blocked",
                "blockers": [{"code": "transcription_ocr_adapter_result_invalid"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "blocked"
    assert any(
        blocker["code"] == "transcription_ocr_stage_not_succeeded"
        for blocker in outcome["receipt"]["blockers"]
    )


def test_corpus_inventory_index_rejects_out_of_workspace_path_before_file_open(tmp_path: Path) -> None:
    """(b) An out-of-workspace admitted path is rejected before any file open (done gate 9)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    outside_target = tmp_path / "outside-workspace-secret.vtt"
    outside_target.write_text("must never be opened", encoding="utf-8")

    open_calls: list[str] = []
    real_open = Path.open

    def tracked_open(self: Path, *args: object, **kwargs: object) -> object:
        open_calls.append(str(self))
        return real_open(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "open", tracked_open)
        resolved, blockers = runner._validate_admitted_binary_artifact(
            workspace=workspace,
            path_value=str(outside_target),
            expected_sha256="irrelevant",
        )

    assert resolved is None
    assert [blocker["code"] for blocker in blockers] == ["out_of_workspace_path"]
    assert not any(str(outside_target) in call for call in open_calls)


def test_corpus_inventory_index_rejects_zip_impostor_with_mismatched_digest(tmp_path: Path) -> None:
    """(c) A ZIP/non-text impostor with a mismatched digest is rejected as a
    provenance mismatch -- digest verification takes precedence over the
    content-type check, since a mismatched digest is the more fundamental
    tampering/corruption signal regardless of what the bytes look like."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    impostor_path = workspace / "storage" / "impostor.pdf"
    impostor_path.parent.mkdir(parents=True, exist_ok=True)
    impostor_path.write_bytes(b"PK\x03\x04" + b"not a real pdf" * 10)

    resolved, blockers = runner._validate_admitted_binary_artifact(
        workspace=workspace,
        path_value=str(impostor_path.relative_to(workspace)),
        expected_sha256="0" * 64,
        require_magic=b"%PDF-",
    )

    assert resolved is None
    assert [blocker["code"] for blocker in blockers] == ["artifact_provenance_mismatch"]


def test_corpus_inventory_index_rejects_zip_impostor_with_matching_digest(tmp_path: Path) -> None:
    """A ZIP impostor whose digest DOES match the declared value is only then
    classified as impostor_artifact_rejected -- the digest gate guarantees
    this code path never hides a tampered file behind a content-type label."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    impostor_path = workspace / "storage" / "impostor.pdf"
    impostor_path.parent.mkdir(parents=True, exist_ok=True)
    impostor_path.write_bytes(b"PK\x03\x04" + b"not a real pdf" * 10)
    real_sha256 = hashlib.sha256(impostor_path.read_bytes()).hexdigest()

    resolved, blockers = runner._validate_admitted_binary_artifact(
        workspace=workspace,
        path_value=str(impostor_path.relative_to(workspace)),
        expected_sha256=real_sha256,
        require_magic=b"%PDF-",
    )

    assert resolved is None
    assert [blocker["code"] for blocker in blockers] == ["impostor_artifact_rejected"]
    assert blockers[0]["reason"] == "zip_magic_bytes"


def test_build_main_conference_paper_records_extracts_real_text_and_marks_indexable(
    tmp_path: Path,
) -> None:
    """A genuine paper_pdf artifact is admitted with its real text extracted
    locally (no network, no paid provider) and marked indexable."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    pdf_path = workspace / "storage" / "download-corpus" / "artifacts" / "paper_pdf" / "one.pdf"
    pdf_bytes = _write_extractable_pdf(pdf_path, "A Real Paper Title\nGenuinely extractable body text.")
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    ledger = {
        "artifacts": [
            {
                "artifact_kind": "paper_pdf",
                "local_path": str(pdf_path.relative_to(workspace)),
                "sha256": sha256,
                "request_key": "req-one",
            },
            {"artifact_kind": "source_page", "local_path": "irrelevant", "sha256": "x", "request_key": "req-two"},
        ]
    }

    records, blockers = runner._build_main_conference_paper_records(
        workspace=workspace, download_corpus_ledger=ledger
    )

    assert blockers == []
    assert len(records) == 1
    record = records[0]
    assert record["artifact_id"] == "paper:req-one"
    assert record["source_family_id"] == "paper_text"
    assert record["indexable_text"] is True
    text_path = workspace / record["text_path"]
    assert text_path.is_file()
    assert "Genuinely extractable body text" in text_path.read_text(encoding="utf-8")


def test_build_main_conference_paper_records_rejects_digest_mismatch(tmp_path: Path) -> None:
    """A declared sha256 that does not match the real bytes is rejected, never opened as text."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    pdf_path = workspace / "storage" / "download-corpus" / "artifacts" / "paper_pdf" / "tampered.pdf"
    _write_extractable_pdf(pdf_path, "Tampered after the ledger recorded its digest.")
    ledger = {
        "artifacts": [
            {
                "artifact_kind": "paper_pdf",
                "local_path": str(pdf_path.relative_to(workspace)),
                "sha256": "0" * 64,
                "request_key": "req-tampered",
            }
        ]
    }

    records, blockers = runner._build_main_conference_paper_records(
        workspace=workspace, download_corpus_ledger=ledger
    )

    assert records == []
    assert any(b["code"] == "artifact_provenance_mismatch" for b in blockers)


def test_build_main_conference_paper_records_admits_audit_only_when_text_unextractable(
    tmp_path: Path,
) -> None:
    """A structurally valid but textless PDF (e.g. a blank/scanned page with no
    embedded text layer) is admitted as an audit-only entry, never fabricated
    as indexable."""

    from pypdf import PdfWriter

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    pdf_path = workspace / "storage" / "download-corpus" / "artifacts" / "paper_pdf" / "blank.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    ledger = {
        "artifacts": [
            {
                "artifact_kind": "paper_pdf",
                "local_path": str(pdf_path.relative_to(workspace)),
                "sha256": sha256,
                "request_key": "req-blank",
            }
        ]
    }

    records, blockers = runner._build_main_conference_paper_records(
        workspace=workspace, download_corpus_ledger=ledger
    )

    assert blockers == []
    assert len(records) == 1
    assert records[0]["indexable_text"] is False


def test_build_main_conference_paper_records_admits_zip_impostor_as_audit_only(
    tmp_path: Path,
) -> None:
    """A byte-verified artifact whose declared 'pdf' is actually a ZIP (e.g. a
    source-code bundle mistakenly submitted as the paper) is admitted as an
    audit-only record instead of hard-blocking the whole stage -- the digest
    matches exactly what was downloaded, it just isn't a paper."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    zip_path = workspace / "storage" / "download-corpus" / "artifacts" / "paper_pdf" / "bundle.pdf"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(b"PK\x03\x04" + b"not a real pdf, a source bundle" * 4)
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    ledger = {
        "artifacts": [
            {
                "artifact_kind": "paper_pdf",
                "local_path": str(zip_path.relative_to(workspace)),
                "sha256": sha256,
                "request_key": "req-zip-bundle",
            }
        ]
    }

    records, blockers = runner._build_main_conference_paper_records(
        workspace=workspace, download_corpus_ledger=ledger
    )

    assert blockers == []
    assert len(records) == 1
    assert records[0]["indexable_text"] is False
    assert records[0]["text_extraction_error"]["code"] == "impostor_artifact_rejected"


def test_build_workshop_paper_records_blocks_when_output_dir_missing(tmp_path: Path) -> None:
    """A workshop download receipt whose output_dir does not resolve to a real
    directory must raise a blocker, not silently return zero records. This is
    the exact shape of a live incident: the receipt stored a repo-root-relative
    output_dir, which double-prefixed against workspace and pointed nowhere --
    every workshop paper vanished from the corpus with zero blockers raised."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    workshop_receipt = {
        "output_dir": "storage/workshops/does-not-exist",
        "per_workshop": [
            {"venueid": "thecvf.com/CVPR/2026/Workshop/Fixture", "status": "succeeded"}
        ],
    }

    records, blockers = runner._build_workshop_paper_records(
        workspace=workspace, workshop_receipt=workshop_receipt
    )

    assert records == []
    assert any(b["code"] == "workshop_output_dir_missing" for b in blockers)


def test_build_workshop_paper_records_admits_zip_impostor_as_audit_only(tmp_path: Path) -> None:
    """Same audit-only admission as the main-conference case, for a workshop
    paper whose declared 'pdf' is actually a ZIP with a matching digest."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    venueid = "thecvf.com/CVPR/2026/Workshop/Fixture"
    note_id = "zipbundle01"
    slug = slugify_venueid(venueid)
    workshop_dir = workspace / "storage" / "workshops" / "papers" / slug
    workshop_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = workshop_dir / f"{note_id}.pdf"
    pdf_path.write_bytes(b"PK\x03\x04" + b"not a real pdf, a source bundle" * 4)
    pdf_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    receipts_dir = workshop_dir / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    (receipts_dir / f"{note_id}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.workshop_paper_receipt",
                "note_id": note_id,
                "venueid": venueid,
                "status": "succeeded",
                "sha256": pdf_sha256,
                "title": "Fixture Zip Bundle",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    workshop_receipt = {
        "output_dir": "storage/workshops/papers",
        "per_workshop": [{"venueid": venueid, "status": "succeeded"}],
    }

    records, blockers = runner._build_workshop_paper_records(
        workspace=workspace, workshop_receipt=workshop_receipt
    )

    assert blockers == []
    assert len(records) == 1
    assert records[0]["indexable_text"] is False
    assert records[0]["text_extraction_error"]["code"] == "impostor_artifact_rejected"
    assert "text_path" not in records[0]


def _fixture_program_item_index() -> object:
    """A minimal, real ``ProgramItemIndex`` joining ``req-one`` to a resolved
    title/authors pair, for testing the corpus-inventory builders' join."""

    return platform_paper_source_metadata.build_program_item_index(
        download_corpus_ledger={
            "program_item_results": [{"request_key": "req-one", "program_item_id": "pi-one"}]
        },
        source_discovery_inventory={
            "program_items": [
                {
                    "program_item_id": "pi-one",
                    "field_evidence": {"title": {"source_record_id": "sr-one"}},
                }
            ],
            "source_records": [
                {
                    "source_record_id": "sr-one",
                    "title": "Real Joined Source Title",
                    "authors": ["Ada Researcher", "Bo Scientist"],
                }
            ],
        },
    )


def test_build_main_conference_paper_records_joins_title_and_authors_via_index(
    tmp_path: Path,
) -> None:
    """The main-paper join populates paper_title/authors from the fixture
    source-discovery index, through the shared platform resolver."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    pdf_path = workspace / "storage" / "download-corpus" / "artifacts" / "paper_pdf" / "one.pdf"
    pdf_bytes = _write_extractable_pdf(pdf_path, "A Real Paper Title\nGenuinely extractable body text.")
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    ledger = {
        "artifacts": [
            {
                "artifact_kind": "paper_pdf",
                "local_path": str(pdf_path.relative_to(workspace)),
                "sha256": sha256,
                "request_key": "req-one",
            }
        ]
    }

    records, blockers = runner._build_main_conference_paper_records(
        workspace=workspace, download_corpus_ledger=ledger, index=_fixture_program_item_index()
    )

    assert blockers == []
    assert len(records) == 1
    assert records[0]["paper_title"] == "Real Joined Source Title"
    assert records[0]["authors"] == ["Ada Researcher", "Bo Scientist"]
    assert records[0]["source_metadata_provenance"] == {
        "title": platform_paper_source_metadata.PROVENANCE_SOURCE_DISCOVERY_JOIN,
        "authors": platform_paper_source_metadata.PROVENANCE_SOURCE_DISCOVERY_JOIN,
    }


def test_build_main_conference_paper_records_soft_impostor_branch_also_carries_resolved_metadata(
    tmp_path: Path,
) -> None:
    """An unreadable/impostor PDF still carries the resolved title/authors --
    a soft-impostor rejection does not erase known metadata."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    zip_path = workspace / "storage" / "download-corpus" / "artifacts" / "paper_pdf" / "bundle.pdf"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(b"PK\x03\x04" + b"not a real pdf, a source bundle" * 4)
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    ledger = {
        "artifacts": [
            {
                "artifact_kind": "paper_pdf",
                "local_path": str(zip_path.relative_to(workspace)),
                "sha256": sha256,
                "request_key": "req-one",
            }
        ]
    }

    records, blockers = runner._build_main_conference_paper_records(
        workspace=workspace, download_corpus_ledger=ledger, index=_fixture_program_item_index()
    )

    assert blockers == []
    assert len(records) == 1
    assert records[0]["indexable_text"] is False
    assert records[0]["paper_title"] == "Real Joined Source Title"
    assert records[0]["authors"] == ["Ada Researcher", "Bo Scientist"]


def test_build_main_conference_paper_records_unresolvable_request_key_stays_honestly_unresolved(
    tmp_path: Path,
) -> None:
    """A request_key absent from the index yields an unresolved-provenance
    record with no paper_title/authors keys, and raises NO blocker -- an
    individual paper's unresolved metadata is a coverage number, not a stage
    failure."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    pdf_path = workspace / "storage" / "download-corpus" / "artifacts" / "paper_pdf" / "orphan.pdf"
    pdf_bytes = _write_extractable_pdf(pdf_path, "An orphan paper with no source-discovery match.")
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    ledger = {
        "artifacts": [
            {
                "artifact_kind": "paper_pdf",
                "local_path": str(pdf_path.relative_to(workspace)),
                "sha256": sha256,
                "request_key": "req-not-in-index",
            }
        ]
    }

    records, blockers = runner._build_main_conference_paper_records(
        workspace=workspace, download_corpus_ledger=ledger, index=_fixture_program_item_index()
    )

    assert blockers == []
    assert len(records) == 1
    assert "paper_title" not in records[0]
    assert "authors" not in records[0]
    assert records[0]["source_metadata_provenance"] == {
        "title": platform_paper_source_metadata.PROVENANCE_UNRESOLVED,
        "authors": platform_paper_source_metadata.PROVENANCE_UNRESOLVED,
    }


def test_build_workshop_paper_records_populates_authors_in_normal_branch(tmp_path: Path) -> None:
    """The normal (extraction-succeeded) workshop branch populates authors
    from the receipt via the shared resolve_workshop_paper_metadata."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    venueid = "thecvf.com/CVPR/2026/Workshop/Fixture"
    note_id = "normalpaper01"
    slug = slugify_venueid(venueid)
    workshop_dir = workspace / "storage" / "workshops" / "papers" / slug
    pdf_path = workshop_dir / f"{note_id}.pdf"
    pdf_bytes = _write_extractable_pdf(
        pdf_path, "A Real Workshop Paper\nGenuinely extractable body text."
    )
    pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    receipts_dir = workshop_dir / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    (receipts_dir / f"{note_id}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.workshop_paper_receipt",
                "note_id": note_id,
                "venueid": venueid,
                "status": "succeeded",
                "sha256": pdf_sha256,
                "title": "Real Workshop Paper Title",
                "authors": ["Carla Workshop", "Dev Author"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    workshop_receipt = {
        "output_dir": "storage/workshops/papers",
        "per_workshop": [{"venueid": venueid, "status": "succeeded"}],
    }

    records, blockers = runner._build_workshop_paper_records(
        workspace=workspace, workshop_receipt=workshop_receipt
    )

    assert blockers == []
    assert len(records) == 1
    assert records[0]["authors"] == ["Carla Workshop", "Dev Author"]
    assert records[0]["paper_title"] == "Real Workshop Paper Title"


def test_build_workshop_paper_records_admits_cvf_openaccess_download_receipts(tmp_path: Path) -> None:
    """Receipts written by the CVF Open Access workshop downloader are admitted
    by the same builder as OpenReview workshop receipts (shared schema/layout)."""

    from conference_pipeline.acquisition import cvf_openaccess_workshops as cvf

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    pdf_bytes = _write_extractable_pdf(source_pdf, "A CVF Workshop Paper\nGenuinely extractable body text.")
    menu_url = "https://openaccess.thecvf.com/CONF2099_workshops/menu"
    listing_url = "https://openaccess.thecvf.com/CONF2099_workshops/Fixture"
    pdf_url = "https://openaccess.thecvf.com/content/CONF2099W/Fixture/papers/Doe_Paper_CVPRW_2099_paper.pdf"
    pages = {
        menu_url: b'<dd><a href="/CONF2099_workshops/Fixture">Fixture Workshop</a></dd>',
        listing_url: (
            b'<dt class="ptitle"><a href="/content/CONF2099W/Fixture/html/Doe_Paper_CVPRW_2099_paper.html">'
            b'CVF Fixture Paper</a></dt><dd><input type="hidden" name="query_author" value="Jane Doe"></dd>'
            b'<dd>[<a href="/content/CONF2099W/Fixture/papers/Doe_Paper_CVPRW_2099_paper.pdf">pdf</a>]</dd>'
        ),
        pdf_url: pdf_bytes,
    }
    receipt = cvf.run(
        cvf.RunOptions(workspace=workspace, menu_url=menu_url, rate_limit_rps=100.0),
        fetch=lambda url: (200, "application/pdf", pages[url]),
    )
    assert receipt["status"] == "succeeded"

    records, blockers = runner._build_workshop_paper_records(workspace=workspace, workshop_receipt=receipt)

    assert blockers == []
    assert len(records) == 1
    assert records[0]["source_family_id"] == "workshop_paper_text"
    assert records[0]["indexable_text"] is True
    assert records[0]["paper_title"] == "CVF Fixture Paper"
    assert records[0]["authors"] == ["Jane Doe"]
    assert records[0]["workshop_id"] == "openaccess.thecvf.com/CONF2099_workshops/Fixture"


def test_build_workshop_paper_records_populates_authors_in_soft_impostor_branch(
    tmp_path: Path,
) -> None:
    """The soft-impostor workshop branch also populates authors from the
    receipt via the same shared resolver -- not a special case."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)
    venueid = "thecvf.com/CVPR/2026/Workshop/Fixture"
    note_id = "zipbundle02"
    slug = slugify_venueid(venueid)
    workshop_dir = workspace / "storage" / "workshops" / "papers" / slug
    workshop_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = workshop_dir / f"{note_id}.pdf"
    pdf_path.write_bytes(b"PK\x03\x04" + b"not a real pdf, a source bundle" * 4)
    pdf_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    receipts_dir = workshop_dir / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    (receipts_dir / f"{note_id}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.workshop_paper_receipt",
                "note_id": note_id,
                "venueid": venueid,
                "status": "succeeded",
                "sha256": pdf_sha256,
                "title": "Impostor Workshop Paper",
                "authors": ["Ellis Zip", "Fran Bundle"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    workshop_receipt = {
        "output_dir": "storage/workshops/papers",
        "per_workshop": [{"venueid": venueid, "status": "succeeded"}],
    }

    records, blockers = runner._build_workshop_paper_records(
        workspace=workspace, workshop_receipt=workshop_receipt
    )

    assert blockers == []
    assert len(records) == 1
    assert records[0]["indexable_text"] is False
    assert records[0]["authors"] == ["Ellis Zip", "Fran Bundle"]


def test_corpus_inventory_index_carries_unavailable_captions_as_coverage_facts(tmp_path: Path) -> None:
    """(d) youtube_caption_unavailable rows are carried as coverage facts and never chunked."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path)

    records, coverage_facts, blockers = runner._build_caption_transcript_records(
        workspace=workspace,
        event_media_inventory={
            "caption_candidates": [
                {
                    "video_id": "unavailable-0001",
                    "caption_status": "youtube_caption_unavailable",
                    "caption": None,
                    "provenance": ["r5_official_event_page"],
                    "source_event_urls": ["https://cvpr.thecvf.com/virtual/2026/poster/1"],
                }
            ]
        },
    )

    assert blockers == []
    assert records == []
    assert len(coverage_facts) == 1
    assert coverage_facts[0]["video_id"] == "unavailable-0001"
    assert coverage_facts[0]["caption_status"] == "youtube_caption_unavailable"


def test_corpus_inventory_index_happy_path_builds_dual_stores_and_passes_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(e) Happy path over a fixture workspace produces both stores and a passing contract."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-happy-path"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    receipt = outcome["receipt"]
    assert receipt["stage_contract"]["status"] == "pass"
    assert receipt["checks"]["vector_embeddings_written"] is True
    assert receipt["checks"]["chunk_counts"] == receipt["checks"]["embedded_chunk_counts"]
    assert receipt["checks"]["output_inventory"]["path"] == "storage/corpus/corpus_inventory.json"
    assert receipt["checks"]["index_digest_path"] == f"storage/vector-store/{run_id}/index_digest.json"
    for store in receipt["checks"]["vector_materialization"]["stores"].values():
        assert store["embedding_sidecar_manifest_path"]
        assert store["embedding_sidecar_manifest_sha256"]
        assert store["chroma_insert_batch_size"]

    inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["artifact_counts"]["indexable_text"] == 3
    assert any(
        fact["caption_status"] == "youtube_caption_unavailable"
        for fact in inventory["caption_coverage_facts"]
    )

    papers_chunks = (
        workspace / "storage" / "vector-store" / run_id / "stores" / "papers" / "chunks.jsonl"
    )
    artifacts_chunks = (
        workspace / "storage" / "vector-store" / run_id / "stores" / "artifacts" / "chunks.jsonl"
    )
    assert papers_chunks.exists() and papers_chunks.stat().st_size > 0
    assert artifacts_chunks.exists() and artifacts_chunks.stat().st_size > 0
    serialized = json.dumps(outcome, default=str)
    assert SECRET_SENTINEL not in serialized


def test_corpus_inventory_index_rerun_is_incremental_and_full_rebuild_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged corpus re-chunks and re-embeds nothing; --full-rebuild redoes everything."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-incremental"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")

    first = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)
    assert first["status"] == "succeeded", first["receipt"]["blockers"]
    assert first["receipt"]["checks"]["index_update"]["mode"] == "full_rechunk"
    assert first["receipt"]["checks"]["corpus_records_sha256"]

    second = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)
    assert second["status"] == "succeeded", second["receipt"]["blockers"]
    checks = second["receipt"]["checks"]
    assert checks["index_update"]["mode"] == "incremental"
    assert checks["corpus_records_sha256"] == first["receipt"]["checks"]["corpus_records_sha256"]
    for store_name, update in checks["index_update"]["stores"].items():
        assert update["artifacts_rechunked"] == 0, store_name
        assert update["chunks_jsonl_rewritten"] is False, store_name
    for store_name, update in checks["embedding_update"].items():
        assert update["chunks_embedded_now"] == 0, store_name
    assert second["receipt"]["stage_contract"]["status"] == "pass"

    rebuilt = runner.run_corpus_inventory_index_stage(
        workspace=workspace, run_id=run_id, resume=True, full_rebuild=True
    )
    assert rebuilt["status"] == "succeeded", rebuilt["receipt"]["blockers"]
    rebuilt_checks = rebuilt["receipt"]["checks"]
    assert rebuilt_checks["full_rebuild_requested"] is True
    assert rebuilt_checks["index_update"]["mode"] == "full_rebuild"
    for store_name, update in rebuilt_checks["embedding_update"].items():
        assert update["chunks_embedded_now"] == rebuilt_checks["chunk_counts"][store_name]


def _get_by_path(payload: object, path: str) -> object:
    """Resolve a dot-separated path through nested dicts (test-local helper)."""

    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _assert_required_check(payload: dict[str, object], spec: dict[str, object]) -> None:
    """Re-implement stage_contracts._check_required_check as a plain assertion.

    This is intentionally a second, independent implementation of the same
    ``truthy`` / ``equals`` / ``equals_path`` semantics used by
    ``verify_stage_contract`` (T-06V-04): the fixture test below reads
    ``required_checks`` straight from the registry file and asserts each
    path individually, so the coupling cannot silently drift or become
    tautological.
    """

    path = str(spec["path"])
    observed = _get_by_path(payload, path)
    if spec.get("truthy"):
        assert observed, f"{path} must be truthy; observed={observed!r}"
    elif "equals_path" in spec:
        expected = _get_by_path(payload, str(spec["equals_path"]))
        assert observed == expected, (
            f"{path}={observed!r} must equal {spec['equals_path']}={expected!r}"
        )
    elif "equals" in spec:
        assert observed == spec["equals"], f"{path}={observed!r} must equal {spec['equals']!r}"
    else:
        raise AssertionError(f"unsupported required_check spec shape: {spec!r}")


def test_corpus_inventory_index_fixture_receipt_satisfies_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixture/mock end-to-end proof the Stage 6 receipt satisfies its runtime
    contract and every roadmap done gate (3-8, plus 4-5) BEFORE any live run
    (repo rule D-05: validate in dry-run/mock first; T-06V-01).

    VALIDATION-ONLY: this test proves the owner is live-ready against a
    constructed fixture workspace using ``FakeEmbeddingClient`` (no network,
    no provider call, no real acquisition data). Per the plan's own
    objective, a passing result here can never substitute for the live
    Stage 6 receipt -- the live receipt remains the SOLE completion proof
    for PROD-07 (Plan 06-03).
    """

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-contract-fixture"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_MAX_LENGTH", "2048")
    monkeypatch.setenv("CORPUS_VECTOR_EMBED_BATCH_SIZE", "4")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_SENTINEL)

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    receipt = outcome["receipt"]
    checks = receipt["checks"]

    # -- Contract coupling: read required_checks from the registry file
    # itself so this assertion cannot drift from the declared contract (D-04).
    registry_path = REPO_ROOT / "src" / "conference_pipeline" / "stage_owner_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    required_checks = registry["stage_bindings"]["corpus-inventory-index"]["runtime_contract"][
        "required_checks"
    ]
    assert required_checks, "registry must declare at least one required_check"

    for spec in required_checks:
        _assert_required_check(receipt, spec)

    # Prove the coupling is real, not tautological (T-06V-04): dropping any
    # one required_check path from a copy of the receipt must make the SAME
    # assertion function fail.
    for spec in required_checks:
        mutated = copy.deepcopy(receipt)
        path_parts = str(spec["path"]).split(".")
        cursor = mutated
        for part in path_parts[:-1]:
            cursor = cursor[part]
        cursor.pop(path_parts[-1], None)
        with pytest.raises(AssertionError):
            _assert_required_check(mutated, spec)

    # -- Done gates 3, 6, 7, 8: embedding/materialization policy receipt fields --
    assert checks["token_budget_policy_digest"]
    assert checks["embedding_policy_digest"]
    embedding_readiness = checks["embedding_readiness"]
    assert embedding_readiness["client_start_or_load_attempted"] is True
    assert embedding_readiness["embedding_max_length"] == 2048
    assert embedding_readiness["materialization_batch_size"] == 4
    for store in checks["vector_materialization"]["stores"].values():
        assert store["embedding_sidecar_manifest_path"]
        assert store["embedding_sidecar_manifest_sha256"]
        assert store["chroma_insert_batch_size"]
    assert checks["chunk_length_stats"]["total_chunks"] > 0

    # -- Done gates 4-5: dual stores + paper-title/section/subsection prepending --
    papers_chunks_path = (
        workspace / "storage" / "vector-store" / run_id / "stores" / "papers" / "chunks.jsonl"
    )
    artifacts_chunks_path = (
        workspace / "storage" / "vector-store" / run_id / "stores" / "artifacts" / "chunks.jsonl"
    )
    assert papers_chunks_path.exists() and papers_chunks_path.stat().st_size > 0
    assert artifacts_chunks_path.exists() and artifacts_chunks_path.stat().st_size > 0
    papers_rows = [
        json.loads(line)
        for line in papers_chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row["text"].startswith("Paper Title: Fixture Workshop Paper") for row in papers_rows
    )

    # The fixture workshop paper only has a known title (no per-section PDF
    # extraction exists yet for real CVPR workshop papers -- Plan 06-01's
    # decision log). Prove the shared platform chunking contract itself
    # prepends section and subsection too, when they ARE known, feeding a
    # synthetic record through the exact same build_corpus_index_sidecars
    # entry point Stage 6 uses.
    from conference_pipeline.indexing.corpus_vector_store import build_corpus_index_sidecars

    section_fixture_workspace = tmp_path / "section-fixture-workspace"
    section_text_path = section_fixture_workspace / "papers" / "section-demo.txt"
    section_text_path.parent.mkdir(parents=True, exist_ok=True)
    section_text_path.write_text(
        "Evidence body text about reasoning benchmarks and evaluation coverage. " * 20,
        encoding="utf-8",
    )
    section_corpus_inventory_path = (
        section_fixture_workspace / "storage" / "corpus" / "corpus_inventory.json"
    )
    section_corpus_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    section_corpus_inventory_path.write_text("{}\n", encoding="utf-8")
    section_digest = build_corpus_index_sidecars(
        workspace=section_fixture_workspace,
        run_id="section-demo-run",
        artifact_records=[
            {
                "artifact_id": "paper_text:section-demo",
                "source_family_id": "paper_text",
                "source_path": "papers/section-demo.pdf",
                "source_sha256": "section-demo-sha",
                "indexable_text": True,
                "text_path": str(
                    section_text_path.relative_to(section_fixture_workspace)
                ),
                "paper_title": "Known Section Fixture Paper",
                "section_title": "Method",
                "subsection_title": "Ablation Study",
            }
        ],
        corpus_inventory_path=section_corpus_inventory_path,
        corpus_inventory_sha256="section-demo-corpus-sha",
        embedding_readiness=embedding_readiness,
    )
    section_papers_chunks = (
        section_fixture_workspace
        / "storage"
        / "vector-store"
        / "section-demo-run"
        / "stores"
        / "papers"
        / "chunks.jsonl"
    )
    section_rows = [
        json.loads(line)
        for line in section_papers_chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert section_rows, "expected at least one chunk for the known-section fixture paper"
    assert section_rows[0]["text"].startswith(
        "Paper Title: Known Section Fixture Paper\nSection: Method\nSubsection: Ablation Study"
    )
    assert section_digest["stores"]["papers"]["chunk_count"] == len(section_rows)

    # -- youtube_caption_unavailable rows are carried as coverage facts and
    # never chunked (D-01) --
    inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    unavailable_facts = [
        fact
        for fact in inventory["caption_coverage_facts"]
        if fact["caption_status"] == "youtube_caption_unavailable"
    ]
    assert len(unavailable_facts) == 3
    combined_chunk_text = papers_chunks_path.read_text(
        encoding="utf-8"
    ) + artifacts_chunks_path.read_text(encoding="utf-8")
    for fact in unavailable_facts:
        assert str(fact["video_id"]) not in combined_chunk_text

    # -- verify_stage_contract(strict_execution=True) independently over the
    # persisted receipt returns status pass --
    independent_contract = stage_contracts.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="corpus-inventory-index",
        receipt=receipt,
        required_owner_ids=(runner.CVPR_CORPUS_INVENTORY_INDEX_OWNER_ID,),
        mode="live",
        strict_execution=True,
    ).to_dict()
    assert independent_contract["status"] == "pass"
    assert receipt["stage_contract"]["status"] == "pass"

    # -- SECRET_SENTINEL must never appear in any emitted artifact (T-06-03) --
    index_digest_path = workspace / "storage" / "vector-store" / run_id / "index_digest.json"
    for artifact_text in (
        json.dumps(receipt, default=str),
        inventory_path.read_text(encoding="utf-8"),
        index_digest_path.read_text(encoding="utf-8"),
    ):
        assert SECRET_SENTINEL not in artifact_text


def test_exhaustive_topic_extraction_stage_declared_and_env_file_plumbing_exists() -> None:
    """The STAGES entry exists, is implemented, and --env-file is an accepted arg."""

    runner = load_runner_module()

    stage = runner.STAGES["exhaustive-topic-extraction"]
    assert stage.order == 7
    assert stage.receipt_filename == "14A-exhaustive-topic-extraction-receipt.json"
    assert stage.implemented is True
    assert hasattr(runner, "load_env_file")

    args = runner.parse_args(["--stage", "exhaustive-topic-extraction", "--run-id", "x"])
    assert args.stage == "exhaustive-topic-extraction"
    assert hasattr(args, "env_file")
    assert args.env_file is None


def _write_stage6_receipt(workspace: Path, run_id: str, *, status: str = "succeeded") -> Path:
    """Write a minimal Stage 6 (corpus-inventory-index) receipt fixture.

    Only the fields ``run_exhaustive_topic_extraction_stage`` itself reads
    (the receipt's ``status``) are meaningful here; this deliberately does not
    run the real, expensive Stage 6 owner.
    """

    target = workspace / "storage" / "workflow-runs" / run_id
    target.mkdir(parents=True, exist_ok=True)
    receipt_path = target / "stage-06-corpus-inventory-index.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "stage_id": "corpus-inventory-index",
                "stage_order": 6,
                "mode": "live",
                "status": status,
                "blockers": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stamp_upstream_stage_receipts(receipt_path.parent)
    return receipt_path


def test_exhaustive_topic_extraction_blocks_without_calling_owner_when_stage6_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) Blocked -- and the shared owner is never invoked -- when Stage 6 has
    not succeeded (RESEARCH Pitfall 2 / T-07-02)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-exhaustive-missing-stage6"

    def _raise_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("shared exhaustive owner must not be invoked without a succeeded Stage 6 receipt")

    monkeypatch.setattr(runner, "run_exhaustive_topic_extraction", _raise_if_called)

    outcome = runner.run_exhaustive_topic_extraction_stage(
        workspace=workspace, run_id=run_id, resume=True, env_file=None
    )

    assert outcome["status"] == "blocked"
    assert any(
        blocker["code"] == "exhaustive_extraction_stage6_required"
        for blocker in outcome["receipt"]["blockers"]
    )
    # The blocked receipt is persisted (consistent with every other stage),
    # but no by-topic/by-artifact output family was ever created -- proof the
    # shared owner itself was genuinely never invoked.
    assert not (workspace / "storage" / "extractions" / run_id).exists()

    # Also prove a not-succeeded Stage 6 receipt is rejected the same way.
    run_id_blocked_stage6 = "test-exhaustive-blocked-stage6"
    _write_stage6_receipt(workspace, run_id_blocked_stage6, status="blocked")
    outcome_blocked = runner.run_exhaustive_topic_extraction_stage(
        workspace=workspace, run_id=run_id_blocked_stage6, resume=True, env_file=None
    )
    assert outcome_blocked["status"] == "blocked"
    assert any(
        blocker["code"] == "exhaustive_extraction_stage6_required"
        for blocker in outcome_blocked["receipt"]["blockers"]
    )


def test_exhaustive_topic_extraction_happy_path_dispatches_shared_owner_and_passes_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) Succeeded outcome, receipt written, {status, summary, receipt_path}
    shape, when Stage 6 succeeded and the shared owner returns success."""

    runner = load_runner_module()
    assert hasattr(runner, "run_exhaustive_topic_extraction_stage")
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-exhaustive-happy-path"
    _write_stage6_receipt(workspace, run_id)

    extractions_root = workspace / "storage" / "extractions" / run_id
    by_topic_dir = extractions_root / "by-topic"
    by_artifact_dir = extractions_root / "by-artifact"
    by_topic_dir.mkdir(parents=True)
    by_artifact_dir.mkdir(parents=True)

    calls: list[dict[str, object]] = []

    def fake_run_exhaustive_topic_extraction(
        workspace_root: Path, *, run_id: str, force_rebuild: bool = False
    ) -> dict[str, object]:
        calls.append({"workspace_root": workspace_root, "run_id": run_id, "force_rebuild": force_rebuild})
        (by_topic_dir / "fixture-topic.json").write_text(
            json.dumps({"topic_id": "fixture-topic", "artifacts": []}), encoding="utf-8"
        )
        (by_artifact_dir / "fixture-artifact.json").write_text(
            json.dumps({"artifact_id": "fixture-artifact", "topics": []}), encoding="utf-8"
        )
        return {
            "schema_version": 1,
            "run_id": run_id,
            "stage_id": "exhaustive-topic-extraction",
            "status": "succeeded",
            "blockers": [],
        }

    monkeypatch.setattr(
        runner, "run_exhaustive_topic_extraction", fake_run_exhaustive_topic_extraction
    )

    outcome = runner.run_exhaustive_topic_extraction_stage(
        workspace=workspace, run_id=run_id, resume=False, env_file=None
    )

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    assert len(calls) == 1
    assert calls[0]["run_id"] == run_id
    assert calls[0]["force_rebuild"] is True

    assert set(outcome) >= {"status", "summary", "receipt_path"}
    receipt_path = Path(outcome["receipt_path"])
    assert receipt_path.exists()
    assert receipt_path.name == "14A-exhaustive-topic-extraction-receipt.json"
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "succeeded"
    assert persisted["stage_contract"]["status"] == "pass"
    assert persisted["checks"]["env_file"] == {"attempted": False, "exists": False, "loaded": False}
    assert persisted["secrets_copied"] is False

    # Independent re-verification of the persisted receipt against the
    # registry's own runtime contract (mirrors the corpus-inventory-index
    # coupling proof) -- catches drift between this stage and the registry.
    independent_contract = stage_contracts.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="exhaustive-topic-extraction",
        receipt=persisted,
        mode="live",
        strict_execution=True,
    ).to_dict()
    assert independent_contract["status"] == "pass"


def test_load_env_file_records_only_status_flags_never_secret_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-07-01: --env-file plumbing must never leak a secret value into any
    structure this function returns."""

    runner = load_runner_module()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SOME_UNRELATED_VAR", raising=False)

    env_path = tmp_path / "fixture.env"
    env_path.write_text(
        f"OPENAI_API_KEY={SECRET_SENTINEL}\nSOME_UNRELATED_VAR={SECRET_SENTINEL}\n",
        encoding="utf-8",
    )

    status = runner.load_env_file(env_path)

    assert status == {"attempted": True, "exists": True, "loaded": True}
    assert SECRET_SENTINEL not in json.dumps(status)
    assert os.environ.get("OPENAI_API_KEY") == SECRET_SENTINEL
    # Only allowed keys are ever set; an arbitrary unrelated var is ignored.
    assert "SOME_UNRELATED_VAR" not in os.environ

    # Missing/omitted --env-file returns the same status-flags-only shape.
    assert runner.load_env_file(None) == {"attempted": False, "exists": False, "loaded": False}
    assert runner.load_env_file(tmp_path / "does-not-exist.env") == {
        "attempted": True,
        "exists": False,
        "loaded": False,
    }


def test_analysis_helper_cluster_imports_and_provider_cost_approved_flag_exist() -> None:
    """Phase 7 Plan 07-02 Task 1: the ported analysis helper cluster, the
    narrative/manifest imports, and --provider-cost-approved are all present
    (RESEARCH Architecture Patterns; done gates 1, 3, 11)."""

    runner = load_runner_module()

    for name in (
        "load_analysis_topic_specs",
        "build_analysis_source_packs",
        "provider_cost_approval_status",
        "render_narrative_outputs_from_source_packs",
        "OpenAIDeepNarrativeClient",
        "DEFAULT_DEEP_NARRATIVE_MODEL",
        "TOPIC_STRATEGY_JSON",
        "ANALYSIS_MANIFEST",
        "ANALYSIS_SOURCE_PACK_DIR",
        "ANALYSIS_QUERY_PLAN_DIR",
    ):
        assert hasattr(runner, name), f"missing {name}"
    # Thin-orchestrators W5: the gate chain, blocked receipts, manifest
    # emission and chunk selection live in the shared analysis stage.
    shared = runner.production_analysis_stage
    for name in (
        "run_analysis_stage",
        "emit_analysis_manifest",
        "load_and_validate_manifest",
        "DEFAULT_ANALYSIS_PROFILE",
        "select_topic_source_chunks",
    ):
        assert hasattr(shared, name), f"shared analysis stage missing {name}"

    args = runner.parse_args(
        ["--stage", "analysis", "--run-id", "x", "--provider-cost-approved"]
    )
    assert args.provider_cost_approved is True

    args_default = runner.parse_args(["--stage", "analysis", "--run-id", "x"])
    assert args_default.provider_cost_approved is False


def test_provider_cost_approval_status_never_records_a_secret_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--provider-cost-approved / CVPR2026_PROVIDER_COST_APPROVED gate the paid
    Phase 7 narrative call only, and never leak env values into the returned
    status (mirrors load_env_file's own no-secret-value contract)."""

    runner = load_runner_module()
    monkeypatch.delenv("CVPR2026_PROVIDER_COST_APPROVED", raising=False)

    args = runner.parse_args(["--stage", "analysis", "--run-id", "x"])
    status = runner.provider_cost_approval_status(args)
    assert status["approved"] is False
    assert status["approval_sources"] == []
    assert status["required_before_paid_provider_calls"] is True
    assert status["secret_values_recorded"] is False

    approved_args = runner.parse_args(
        ["--stage", "analysis", "--run-id", "x", "--provider-cost-approved"]
    )
    approved_status = runner.provider_cost_approval_status(approved_args)
    assert approved_status["approved"] is True
    assert approved_status["approval_sources"] == ["--provider-cost-approved"]

    monkeypatch.setenv("CVPR2026_PROVIDER_COST_APPROVED", "true")
    env_status = runner.provider_cost_approval_status(args)
    assert env_status["approved"] is True
    assert env_status["approval_sources"] == ["env:CVPR2026_PROVIDER_COST_APPROVED"]


def test_deep_narrative_synthesis_resolves_to_registry_governed_sol_with_terra_fallback() -> None:
    """Done gate 6 guard (RESEARCH Pitfall 3 -- RESOLVED): the runner never
    hardcodes a gpt-5.6 literal; DEFAULT_DEEP_NARRATIVE_MODEL and the deep
    narrative synthesis task must resolve through the model-governance
    registry to openai:gpt-5.6-sol with an openai:gpt-5.6-terra fallback."""

    from conference_pipeline.model_governance import (
        default_fallback_model,
        default_model,
        default_provider,
    )

    task_id = "analysis.deep_narrative_synthesis"
    provider = default_provider(task_id)
    model = default_model(task_id)
    fallback_model = default_fallback_model(task_id)

    from conference_pipeline.model_governance.registry import load_model_catalog

    catalog = load_model_catalog()
    assert provider == "openai"
    assert model == catalog["sol"]["model"]
    assert fallback_model == catalog["terra"]["model"]

    runner = load_runner_module()
    assert runner.DEFAULT_DEEP_NARRATIVE_MODEL == model


def test_analysis_stage_implemented_and_dispatches_from_main(tmp_path: Path) -> None:
    """Task 2: STAGES['analysis'] is flipped to implemented=True and main()
    routes stage.id == 'analysis' to run_analysis (mirrors the
    exhaustive-topic-extraction/corpus-inventory-index dispatch branches)."""

    runner = load_runner_module()
    stage = runner.STAGES["analysis"]
    assert stage.order == 7
    assert stage.receipt_filename == "stage-07-analysis.json"
    assert stage.implemented is True
    assert runner.orchestrator_spec().handlers["analysis"] is runner.dispatch_analysis


def test_analysis_stage_blocks_gracefully_after_stage6_gate(tmp_path: Path) -> None:
    """RESEARCH State-of-the-Art / Pattern 1: run_analysis blocks with a
    zero-spend receipt (never raises) when Stage 6 has not succeeded, and
    again when Stage 6 succeeded but its vector evidence is stale/incomplete."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-analysis-stage6-gate"

    outcome_missing = runner.run_analysis(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=False, env_file=None
    )
    assert outcome_missing["status"] == "blocked"
    assert outcome_missing["receipt"]["checks"]["provider_calls_made"] is False
    assert outcome_missing["receipt"]["blockers"][0]["code"] == "stage6_receipt_missing_or_malformed"

    stage6 = runner.STAGES["corpus-inventory-index"]
    stage6_receipt_path = runner.run_dir(workspace, run_id) / stage6.receipt_filename
    stage6_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    stage6_receipt_path.write_text(
        json.dumps({"status": "blocked", "checks": {"vector_embeddings_written": False}}),
        encoding="utf-8",
    )
    outcome_stale = runner.run_analysis(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=False, env_file=None
    )
    assert outcome_stale["status"] == "blocked"
    assert outcome_stale["receipt"]["checks"]["provider_calls_made"] is False
    assert outcome_stale["receipt"]["blockers"][0]["code"] == "stage6_vector_evidence_required"


def test_analysis_compact_chunk_preserves_title_and_authors() -> None:
    """Phase 10.1-10 Finding A: ``_analysis_compact_chunk`` is the function
    that actually builds the on-disk narrative source packs for CVPR2026 --
    NOT ``compact_chunk_evidence`` (Phase 10.1-03's fix). It must preserve
    the same structured ``paper_title``/``authors`` metadata, in the same
    real shape the vector-store chunks actually carry it (nested under
    ``source_context``), or the ledger repair silently fails to materialize
    even after a full Stage 6 re-embed."""

    runner = load_runner_module()
    chunk = {
        "chunk_id": "workshop_paper:abc:0",
        "artifact_id": "workshop_paper:abc",
        "source_path": "storage/workshops/papers/abc/1.pdf",
        "text": "Paper Title: Real Paper Title Here\n\nBody text follows " + ("x" * 800),
        "source_context": {
            "paper_title": "Real Paper Title Here",
            "authors": ["Jane Doe", "John Smith"],
        },
    }

    compact = runner._analysis_compact_chunk(chunk)

    # The four original bare keys and the 700-char quote bound are unchanged.
    assert compact["chunk_id"] == "workshop_paper:abc:0"
    assert compact["extraction_record_id"] == "workshop_paper:abc"
    assert compact["source_artifact_path"] == "storage/workshops/papers/abc/1.pdf"
    assert len(compact["quote"]) == 700

    # The new preserved metadata survives compaction.
    assert compact["paper_title"] == "Real Paper Title Here"
    assert compact["authors"] == ["Jane Doe", "John Smith"]


def test_analysis_compact_chunk_prefers_top_level_metadata_over_source_context() -> None:
    """Mirrors ``compact_chunk_evidence``'s own fallback order: a top-level
    ``paper_title``/``authors`` key (if a future call site ever sets one)
    takes priority over the nested ``source_context`` copy, never the other
    way around."""

    runner = load_runner_module()
    chunk = {
        "chunk_id": "c1",
        "artifact_id": "a1",
        "source_path": "p1",
        "text": "text",
        "paper_title": "Top Level Title",
        "authors": ["Top Level Author"],
        "source_context": {
            "paper_title": "Nested Title",
            "authors": ["Nested Author"],
        },
    }

    compact = runner._analysis_compact_chunk(chunk)

    assert compact["paper_title"] == "Top Level Title"
    assert compact["authors"] == ["Top Level Author"]


def test_analysis_source_packs_include_llm_full_read_evidence(tmp_path: Path) -> None:
    """build_analysis_source_packs pairs the deterministic keyword-selected
    chunks with the completed Qwen full-read evidence for the same topic
    (RESEARCH Architecture Diagram step [3]; done gates 2, 3, 11)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-analysis-source-pack-full-read"

    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    artifact_store = vector_root / "stores" / "artifacts"
    paper_store.mkdir(parents=True)
    artifact_store.mkdir(parents=True)
    paper_chunks = [
        {
            "chunk_id": f"paper:reasoning:{index}",
            "artifact_id": f"paper_abstract:paper-{index}",
            "source_path": f"https://cvpr.thecvf.com/paper-{index}",
            "text": f"Reasoning and test-time compute paper evidence for topic one, sentence {index}.",
        }
        for index in range(3)
    ]
    with (paper_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in paper_chunks:
            handle.write(json.dumps(chunk) + "\n")
    (artifact_store / "chunks.jsonl").write_text("", encoding="utf-8")
    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, paper_store / "chunks.jsonl"
                )
            },
            "artifacts": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, artifact_store / "chunks.jsonl"
                )
            },
        }
    }
    (vector_root / "index_digest.json").write_text(json.dumps(index_digest), encoding="utf-8")

    topic_specs = ({"id": "01_topic_one", "title": "Topic One", "keywords": ("reasoning",)},)

    full_read_dir = workspace / "storage" / "workflow-runs" / run_id
    full_read_dir.mkdir(parents=True, exist_ok=True)
    (full_read_dir / "14A-llm-full-read-extraction-receipt.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "model": "fixture-model",
                "scope": "all",
                "checks": {"completion_state": "complete"},
            }
        ),
        encoding="utf-8",
    )
    by_topic_dir = workspace / "storage" / "extractions" / run_id / "llm-full-read" / "by-topic"
    by_topic_dir.mkdir(parents=True, exist_ok=True)
    (by_topic_dir / "01_topic_one.json").write_text(
        json.dumps(
            {
                "unit_count": 1,
                "items": [
                    {
                        "unit_id": "unit-1",
                        "input_kind": "paper_abstract",
                        "confidence": 0.9,
                        "important": True,
                        "summary": "Summary text",
                        "reason": "reason text",
                        "evidence_phrases": ["phrase one"],
                        "source_artifact_path": "storage/corpus/paper-text/paper-1.json",
                        "metadata": {"title": "Paper One", "paper_id": "paper-1"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "a" * 64}}},
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    assert result["insufficient_topics"] == []
    source_pack_path = workspace / "analysis" / "narrative_source_packs" / "01_topic_one.json"
    assert source_pack_path.exists()
    source_pack = json.loads(source_pack_path.read_text(encoding="utf-8"))
    evidence = source_pack["llm_full_read_evidence"]
    assert evidence["status"] == "succeeded"
    assert evidence["unit_count"] == 1
    assert evidence["selected_items"][0]["unit_id"] == "unit-1"
    assert "unit-1" in source_pack["extraction_record_ids"]


def test_analysis_source_packs_are_not_paper_starved_by_captions(tmp_path: Path) -> None:
    """Regression: a caption-dense conference must not starve paper chunks to zero.

    The historic single-pass scan read the artifacts (caption) store before the
    papers store and stopped at ANALYSIS_SOURCE_PACK_MAX_CHUNKS per topic, so for
    CVPR (4k+ papers vs ~350 caption transcripts) every topic filled its chunk
    cap with captions and the source packs -- and therefore the deep narrative --
    were 100% caption / 0% paper. build_analysis_source_packs now composes each
    topic papers-first (up to ANALYSIS_SOURCE_PACK_PAPER_TARGET) then fills with
    captions, so both source families are represented even when captions alone
    could fill the cap."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-analysis-source-pack-paper-balance"

    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    artifact_store = vector_root / "stores" / "artifacts"
    paper_store.mkdir(parents=True)
    artifact_store.mkdir(parents=True)

    # Papers and captions both match the topic keyword; captions alone are more
    # than enough to fill the whole cap (reproducing the starvation scenario).
    # Thirty candidates -- rather than twelve -- so the fixture supplies more
    # distinct papers than the runner's raised paper target (10.3-02, PROD-15:
    # ANALYSIS_SOURCE_PACK_PAPER_TARGET 10 -> 26, ANALYSIS_SOURCE_PACK_MAX_CHUNKS
    # 12 -> 28); this is a fixture-capacity fix, not a contract change -- every
    # assertion below is unchanged and the test passes at both budgets.
    paper_chunks = [
        {
            "chunk_id": f"paper:vision-paper-{index}:0",
            "artifact_id": f"paper_abstract:paper-{index}",
            "source_path": f"https://cvpr.thecvf.com/paper-{index}",
            "text": f"Vision paper evidence for topic one, abstract sentence {index}.",
        }
        for index in range(30)
    ]
    caption_chunks = [
        {
            "chunk_id": f"event_media_caption:talk-{index}:0",
            "artifact_id": f"event_media_caption:talk-{index}",
            "source_path": f"storage/download-corpus/event-media/captions/talk-{index}.en.vtt",
            "text": f"Vision talk transcript window {index} discussing the same topic.",
        }
        for index in range(20)
    ]
    with (paper_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in paper_chunks:
            handle.write(json.dumps(chunk) + "\n")
    with (artifact_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in caption_chunks:
            handle.write(json.dumps(chunk) + "\n")

    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, paper_store / "chunks.jsonl"
                )
            },
            "artifacts": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, artifact_store / "chunks.jsonl"
                )
            },
        }
    }
    (vector_root / "index_digest.json").write_text(json.dumps(index_digest), encoding="utf-8")

    topic_specs = ({"id": "01_topic_one", "title": "Topic One", "keywords": ("vision",)},)

    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "a" * 64}}},
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / "01_topic_one.json").read_text(
            encoding="utf-8"
        )
    )
    chunk_ids = [chunk["chunk_id"] for chunk in source_pack["chunks"]]
    paper_ids = [cid for cid in chunk_ids if cid.startswith("paper:")]
    caption_ids = [cid for cid in chunk_ids if cid.startswith("event_media_caption:")]

    # The core regression assertion: papers are present, not starved to zero.
    assert paper_ids, f"source pack has zero paper chunks (paper-starved): {chunk_ids}"
    # Papers-majority target is honored when the papers store has enough matches.
    assert len(paper_ids) == runner.ANALYSIS_SOURCE_PACK_PAPER_TARGET
    # Source-diversity: the paper chunks come from DISTINCT papers, not many
    # chunks of a single early paper monopolizing the quota.
    distinct_papers = {":".join(cid.split(":")[:-1]) for cid in paper_ids}
    assert len(distinct_papers) == len(paper_ids), (
        f"paper chunks should be from distinct papers, got {sorted(paper_ids)}"
    )
    # Captions still fill the remaining slots (both families represented).
    assert caption_ids
    assert len(chunk_ids) == runner.ANALYSIS_SOURCE_PACK_MAX_CHUNKS
    # The query plan mirrors the same balanced selection.
    query_plan = json.loads(
        (workspace / "analysis" / "provenance" / "query_plans" / "01_topic_one.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(cid.startswith("paper:") for cid in query_plan["selected_chunk_ids"])


def test_analysis_source_packs_runner_wired_to_shared_selector_preserves_conference_scope_order_independence(
    tmp_path: Path,
) -> None:
    """Phase 10.1.1-source-pack-family-stratification Plan 03 (PROD-15): proves
    the delegation is REAL through the runner's own call path, not just the
    shared library in isolation (that proof already exists in
    tests/test_analysis_source_pack_selection.py from Plan 02).

    The papers store here stamps every chunk with an explicit
    ``conference_scope`` and places every ``workshop``-scoped chunk BEFORE
    every ``main_conference``-scoped chunk in file order -- reproducing
    Finding D's physical-order dependency. Wiring: build_analysis_source_packs
    must still surface main-conference chunk ids in the written source pack.
    Provenance: the written query plan's ``source_families`` block must
    report a non-zero main_conference count for this same fixture."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-analysis-source-pack-conference-scope-order"

    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    artifact_store = vector_root / "stores" / "artifacts"
    paper_store.mkdir(parents=True)
    artifact_store.mkdir(parents=True)

    # Workshop-scoped chunks occupy every earlier physical file position; only
    # AFTER all of them do main-conference-scoped chunks appear. A scan that
    # depended on file order (the pre-fix behavior) would fill the whole
    # topic quota from workshop alone and never reach a main-conference match.
    workshop_chunks = [
        {
            "chunk_id": f"paper:workshop-{index}:0",
            "artifact_id": f"paper_abstract:workshop-{index}",
            "source_path": f"https://cvpr.thecvf.com/workshop-{index}",
            "text": f"Quantum evidence from a workshop paper, sentence {index}.",
            "conference_scope": "workshop",
        }
        for index in range(20)
    ]
    # Ten main-conference candidates -- rather than six -- so the fixture
    # supplies thirty chunks in total across both scopes, above the runner's
    # raised chunk cap (10.3-02, PROD-15: ANALYSIS_SOURCE_PACK_MAX_CHUNKS
    # 12 -> 28). Capacity only: the workshop chunks still occupy every earlier
    # physical file position, which is the defect this test exists to pin, and
    # every assertion below is unchanged. Verified to pass at BOTH the old
    # (12/10/10) and new (28/26/26) budgets -- a fixture that only passed after
    # the budget moved would be indistinguishable from one rewritten to fit a
    # new answer.
    main_chunks = [
        {
            "chunk_id": f"paper:main-{index}:0",
            "artifact_id": f"paper_abstract:main-{index}",
            "source_path": f"https://cvpr.thecvf.com/main-{index}",
            "text": f"Quantum evidence from a main-conference paper, sentence {index}.",
            "conference_scope": "main_conference",
        }
        for index in range(10)
    ]
    with (paper_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in workshop_chunks + main_chunks:
            handle.write(json.dumps(chunk) + "\n")
    (artifact_store / "chunks.jsonl").write_text("", encoding="utf-8")

    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, paper_store / "chunks.jsonl"
                )
            },
            "artifacts": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, artifact_store / "chunks.jsonl"
                )
            },
        }
    }
    (vector_root / "index_digest.json").write_text(json.dumps(index_digest), encoding="utf-8")

    topic_specs = ({"id": "01_topic_one", "title": "Topic One", "keywords": ("quantum",)},)

    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "a" * 64}}},
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / "01_topic_one.json").read_text(
            encoding="utf-8"
        )
    )
    chunk_ids = [chunk["chunk_id"] for chunk in source_pack["chunks"]]
    main_ids = [cid for cid in chunk_ids if cid.startswith("paper:main-")]
    workshop_ids = [cid for cid in chunk_ids if cid.startswith("paper:workshop-")]

    # Wiring + order independence: main-conference chunks are present despite
    # occupying every later physical file position, and reach at least the
    # guaranteed floor (5) -- the pre-fix single-bucket scan would have
    # filled the entire cap from the 20 earlier workshop chunks and never
    # reached a main-conference match at all.
    assert main_ids, f"main-conference chunks starved despite workshop-first file order: {chunk_ids}"
    assert len(main_ids) >= 5
    assert workshop_ids
    assert len(chunk_ids) == runner.ANALYSIS_SOURCE_PACK_MAX_CHUNKS

    # Provenance: the query plan's source_families block reports a non-zero
    # main_conference count for this same fixture, so per-family
    # representation is inspectable from disk without re-deriving it.
    query_plan = json.loads(
        (workspace / "analysis" / "provenance" / "query_plans" / "01_topic_one.json").read_text(
            encoding="utf-8"
        )
    )
    source_families = query_plan["source_families"]
    assert source_families["main_conference"] > 0
    assert source_families["main_conference"] == len(main_ids)
    assert source_families["workshop"] == len(workshop_ids)


def test_analysis_source_packs_honor_runner_main_conference_floor_above_shared_default(
    tmp_path: Path,
) -> None:
    """10.2-02 (PROD-15) drop-proof test: fails if the runner's explicit
    ``main_conference_floor=ANALYSIS_SOURCE_PACK_PAPER_TARGET`` keyword
    argument at the ``select_topic_source_chunks`` call site is ever removed
    or reverted so that selection silently falls back to the shared module's
    own (much lower) default floor.

    The papers store here reproduces the same workshop-first physical file
    order as the sibling order-independence test, but supplies MORE
    main-conference-scoped chunks than the shared module's own default floor
    would ever admit -- the runner's raised paper target plus two spares.
    If the runner ever stopped passing its own floor explicitly and fell
    back to the shared default, only ``DEFAULT_MAIN_CONFERENCE_FLOOR`` many
    main-conference chunks would surface and this test would fail on the
    first assertion below."""

    from conference_pipeline.analysis import source_pack_selection

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-analysis-source-pack-runner-floor-above-shared-default"

    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    artifact_store = vector_root / "stores" / "artifacts"
    paper_store.mkdir(parents=True)
    artifact_store.mkdir(parents=True)

    # Every workshop-scoped chunk occupies an earlier physical file position
    # than every main-conference-scoped chunk (Finding D order-dependency
    # reproduction). The main-conference family supplies the runner's paper
    # target plus two spares -- strictly more than the shared module's own
    # default floor would ever select if the explicit keyword argument were
    # dropped.
    workshop_chunks = [
        {
            "chunk_id": f"paper:floor-workshop-{index}:0",
            "artifact_id": f"paper_abstract:floor-workshop-{index}",
            "source_path": f"https://cvpr.thecvf.com/floor-workshop-{index}",
            "text": f"Robotics evidence from a workshop paper, sentence {index}.",
            "conference_scope": "workshop",
        }
        for index in range(20)
    ]
    main_chunks = [
        {
            "chunk_id": f"paper:floor-main-{index}:0",
            "artifact_id": f"paper_abstract:floor-main-{index}",
            "source_path": f"https://cvpr.thecvf.com/floor-main-{index}",
            "text": f"Robotics evidence from a main-conference paper, sentence {index}.",
            "conference_scope": "main_conference",
        }
        for index in range(runner.ANALYSIS_SOURCE_PACK_PAPER_TARGET + 2)
    ]
    with (paper_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in workshop_chunks + main_chunks:
            handle.write(json.dumps(chunk) + "\n")
    (artifact_store / "chunks.jsonl").write_text("", encoding="utf-8")

    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, paper_store / "chunks.jsonl"
                )
            },
            "artifacts": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, artifact_store / "chunks.jsonl"
                )
            },
        }
    }
    (vector_root / "index_digest.json").write_text(json.dumps(index_digest), encoding="utf-8")

    topic_specs = ({"id": "01_topic_one", "title": "Topic One", "keywords": ("robotics",)},)

    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "a" * 64}}},
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / "01_topic_one.json").read_text(
            encoding="utf-8"
        )
    )
    chunk_ids = [chunk["chunk_id"] for chunk in source_pack["chunks"]]
    main_ids = [cid for cid in chunk_ids if cid.startswith("paper:floor-main-")]

    # The runner's own floor -- not the shared module's default -- must be
    # what reached the selector: at least the paper-target-many
    # main-conference chunks must be present.
    assert len(main_ids) >= runner.ANALYSIS_SOURCE_PACK_PAPER_TARGET, (
        f"main-conference chunks fell short of the runner's own floor, "
        f"got {len(main_ids)}: {chunk_ids}"
    )
    # Drop-proof by name, not by literal: if the explicit
    # main_conference_floor=ANALYSIS_SOURCE_PACK_PAPER_TARGET keyword
    # argument is ever reverted, the runner's target must still exceed the
    # shared module's own default so this comparison catches the silent
    # fallback rather than merely re-asserting today's numbers.
    assert (
        runner.ANALYSIS_SOURCE_PACK_PAPER_TARGET
        > source_pack_selection.DEFAULT_MAIN_CONFERENCE_FLOOR
    ), (
        "runner paper target no longer exceeds the shared module's default "
        "main-conference floor -- the raised-floor wiring may have reverted"
    )


def test_analysis_source_pack_schema_top_level_keys_unchanged(tmp_path: Path) -> None:
    """Phase 10.1.1-source-pack-family-stratification Plan 03 (PROD-15): the
    runner's own on-disk source-pack schema is byte-shape identical to before
    the shared-selector delegation -- only WHICH chunks are selected changes,
    never what fields are written. An explicit expected key list (rather than
    a diff against a second live run) catches an accidental future key
    addition or removal in the runner's serialization."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-analysis-source-pack-schema-stability"

    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    artifact_store = vector_root / "stores" / "artifacts"
    paper_store.mkdir(parents=True)
    artifact_store.mkdir(parents=True)
    paper_chunks = [
        {
            "chunk_id": f"paper:schema-{index}:0",
            "artifact_id": f"paper_abstract:schema-{index}",
            "source_path": f"https://cvpr.thecvf.com/schema-{index}",
            "text": f"Schema-stability probe evidence, sentence {index}.",
        }
        for index in range(3)
    ]
    with (paper_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in paper_chunks:
            handle.write(json.dumps(chunk) + "\n")
    (artifact_store / "chunks.jsonl").write_text("", encoding="utf-8")

    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, paper_store / "chunks.jsonl"
                )
            },
            "artifacts": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, artifact_store / "chunks.jsonl"
                )
            },
        }
    }
    (vector_root / "index_digest.json").write_text(json.dumps(index_digest), encoding="utf-8")

    topic_specs = ({"id": "01_topic_one", "title": "Topic One", "keywords": ("schema-stability",)},)

    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "a" * 64}}},
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / "01_topic_one.json").read_text(
            encoding="utf-8"
        )
    )
    expected_keys = [
        "chunks",
        "corpus_inventory_sha256",
        "extraction_record_ids",
        "finding_text",
        "generated_at",
        "index_digest_sha256",
        "live_provider_calls_made",
        "llm_full_read_evidence",
        "matched_papers",
        "owner_ref",
        "paper_matching_evidence",
        "provider_calls_made",
        "research_id",
        "research_path",
        "run_id",
        "schema_version",
        "secrets_copied",
        "status",
        "title",
        "topic_id",
    ]
    assert sorted(source_pack.keys()) == expected_keys


def _write_analysis_priority_fixture_store(
    workspace: Path, *, run_id: str, runner: object, count: int = 20, keyword: str = "spatial"
) -> tuple[dict, list[str]]:
    """Write a single-topic papers store of ``count`` main-conference chunks
    all matching ``keyword``, and return (index_digest, hex64 ids in scan
    order). No ``conference_scope`` field is stamped -- these chunks fail
    open to ``main_conference``, mirroring every neighbouring source-pack
    test in this module."""

    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    artifact_store = vector_root / "stores" / "artifacts"
    paper_store.mkdir(parents=True)
    artifact_store.mkdir(parents=True)

    hashes = [hashlib.sha256(f"priority-fixture-{index}".encode()).hexdigest() for index in range(count)]
    chunks = [
        {
            "chunk_id": f"paper:priority-{index}:0",
            "artifact_id": f"paper:{hashes[index]}",
            "source_path": f"https://cvpr.thecvf.com/priority-{index}",
            "text": f"{keyword.capitalize()} evidence for the priority-signal fixture, sentence {index}.",
        }
        for index in range(count)
    ]
    with (paper_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")
    (artifact_store / "chunks.jsonl").write_text("", encoding="utf-8")

    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, paper_store / "chunks.jsonl"
                )
            },
            "artifacts": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, artifact_store / "chunks.jsonl"
                )
            },
        }
    }
    (vector_root / "index_digest.json").write_text(json.dumps(index_digest), encoding="utf-8")
    return index_digest, hashes


def _write_analysis_matched_papers_fixture(
    workspace: Path, *, run_id: str, topic_id: str, papers: list[dict]
) -> None:
    matched_dir = workspace / "storage" / "paper-matching" / run_id / topic_id
    matched_dir.mkdir(parents=True, exist_ok=True)
    (matched_dir / "matched_papers.json").write_text(
        json.dumps({"papers": papers}), encoding="utf-8"
    )


def test_analysis_source_packs_use_matched_papers_priority_signal(tmp_path: Path) -> None:
    """Phase 10.1.1.1 Plan 03 (PROD-15): a topic's already-paid
    ``matched_papers.json`` signal must reach ``select_topic_source_chunks``
    BEFORE selection, not just decorate the persisted provenance afterwards.

    20 main-conference chunks all match the topic keyword equally (one hit
    each); a cap-then-stop-by-file-position scan (the pre-fix behavior) can
    only ever surface the first 8 of them. This fixture names the chunks at
    scan positions 15 and 19 -- positions such a scan never reaches -- as
    priority ids in ``matched_papers.json``. Both must appear in the written
    source pack's ``chunks[]`` if the wiring is real."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-analysis-source-pack-priority-signal"
    topic_id = "03_topic_priority"

    index_digest, hashes = _write_analysis_priority_fixture_store(workspace, run_id=run_id, runner=runner)
    _write_analysis_matched_papers_fixture(
        workspace,
        run_id=run_id,
        topic_id=topic_id,
        papers=[
            {"paper_id": hashes[15], "rank": 1, "title": "Priority Paper Fifteen"},
            {"paper_id": hashes[19], "rank": 2, "title": "Priority Paper Nineteen"},
        ],
    )

    topic_specs = ({"id": topic_id, "title": "Topic Priority", "keywords": ("spatial",)},)

    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "a" * 64}}},
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json").read_text(
            encoding="utf-8"
        )
    )
    extraction_record_ids = [chunk["extraction_record_id"] for chunk in source_pack["chunks"]]
    assert f"paper:{hashes[15]}" in extraction_record_ids, (
        f"priority-named chunk at scan position 15 missing from selection: {extraction_record_ids}"
    )
    assert f"paper:{hashes[19]}" in extraction_record_ids, (
        f"priority-named chunk at scan position 19 missing from selection: {extraction_record_ids}"
    )


def test_analysis_source_packs_respect_matched_papers_rank_order(tmp_path: Path) -> None:
    """The selector orders priority candidates by ``matched_papers.json``'s
    own ``rank`` field, not by the candidate's position within that JSON
    file's ``papers`` list."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-analysis-source-pack-priority-rank-order"
    topic_id = "03_topic_rank_order"

    index_digest, hashes = _write_analysis_priority_fixture_store(workspace, run_id=run_id, runner=runner)
    # The numerically-worse rank (2) is written FIRST in the file's list; the
    # numerically-better rank (1) is written SECOND. Selection order must
    # still place rank 1 before rank 2.
    _write_analysis_matched_papers_fixture(
        workspace,
        run_id=run_id,
        topic_id=topic_id,
        papers=[
            {"paper_id": hashes[19], "rank": 2, "title": "Worse Rank, Written First"},
            {"paper_id": hashes[15], "rank": 1, "title": "Better Rank, Written Second"},
        ],
    )

    topic_specs = ({"id": topic_id, "title": "Topic Rank Order", "keywords": ("spatial",)},)

    runner.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt={"checks": {"output_inventory": {"sha256": "a" * 64}}},
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    source_pack = json.loads(
        (workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json").read_text(
            encoding="utf-8"
        )
    )
    extraction_record_ids = [chunk["extraction_record_id"] for chunk in source_pack["chunks"]]
    rank_1_id = f"paper:{hashes[15]}"
    rank_2_id = f"paper:{hashes[19]}"
    assert rank_1_id in extraction_record_ids
    assert rank_2_id in extraction_record_ids
    assert extraction_record_ids.index(rank_1_id) < extraction_record_ids.index(rank_2_id), (
        f"rank 1 paper must be selected/ordered ahead of rank 2 paper: {extraction_record_ids}"
    )


def test_analysis_source_packs_ignore_non_hex_matched_paper_ids(tmp_path: Path) -> None:
    """arXiv ids, OpenReview note ids, and hand-typed slugs never join --
    only bare 64-char lowercase hex ids do. A ``matched_papers.json``
    containing only such ids must produce the byte-identical selection to
    the file being absent entirely (the five-zero-joinable-topics path)."""

    run_id = "test-analysis-source-pack-non-hex-ids"
    topic_id = "01_topic_non_hex"
    topic_specs = ({"id": topic_id, "title": "Topic Non Hex", "keywords": ("spatial",)},)
    stage6_receipt = {"checks": {"output_inventory": {"sha256": "a" * 64}}}

    runner = load_runner_module()

    # Run A: matched_papers.json present, but every id is a non-hex64 shape.
    workspace_a = write_runner_workspace(tmp_path / "with_odd_file" / "conferences")
    index_digest_a, _hashes_a = _write_analysis_priority_fixture_store(workspace_a, run_id=run_id, runner=runner)
    _write_analysis_matched_papers_fixture(
        workspace_a,
        run_id=run_id,
        topic_id=topic_id,
        papers=[
            {"paper_id": "arXiv:2401.12345", "rank": 1},
            {"paper_id": "H1xJgh2kQ", "rank": 2},
            {"paper_id": "my-hand-typed-slug", "rank": 3},
        ],
    )
    runner.build_analysis_source_packs(
        workspace=workspace_a,
        run_id=run_id,
        stage6_receipt=stage6_receipt,
        index_digest=index_digest_a,
        topic_specs=topic_specs,
    )
    pack_a = json.loads(
        (workspace_a / "analysis" / "narrative_source_packs" / f"{topic_id}.json").read_text(
            encoding="utf-8"
        )
    )

    # Run B: no storage/paper-matching directory at all.
    workspace_b = write_runner_workspace(tmp_path / "without_file" / "conferences")
    index_digest_b, _hashes_b = _write_analysis_priority_fixture_store(workspace_b, run_id=run_id, runner=runner)
    assert not (workspace_b / "storage" / "paper-matching").exists()
    runner.build_analysis_source_packs(
        workspace=workspace_b,
        run_id=run_id,
        stage6_receipt=stage6_receipt,
        index_digest=index_digest_b,
        topic_specs=topic_specs,
    )
    pack_b = json.loads(
        (workspace_b / "analysis" / "narrative_source_packs" / f"{topic_id}.json").read_text(
            encoding="utf-8"
        )
    )

    chunk_ids_a = [chunk["chunk_id"] for chunk in pack_a["chunks"]]
    chunk_ids_b = [chunk["chunk_id"] for chunk in pack_b["chunks"]]
    assert chunk_ids_a == chunk_ids_b, (
        f"non-hex matched_papers.json changed selection vs. absent file: {chunk_ids_a} != {chunk_ids_b}"
    )


def test_analysis_source_packs_tolerate_malformed_matched_papers_json(tmp_path: Path) -> None:
    """A ``matched_papers.json`` containing invalid JSON, or a JSON list at
    the document root instead of an object, must not raise -- and must
    produce the same selection as the absent-file case for each affected
    topic."""

    run_id = "test-analysis-source-pack-malformed-json"
    invalid_json_topic_id = "01_topic_invalid_json"
    wrong_root_type_topic_id = "02_topic_wrong_root_type"
    topic_specs = (
        {"id": invalid_json_topic_id, "title": "Topic Invalid JSON", "keywords": ("spatial",)},
        {"id": wrong_root_type_topic_id, "title": "Topic Wrong Root Type", "keywords": ("spatial",)},
    )
    stage6_receipt = {"checks": {"output_inventory": {"sha256": "a" * 64}}}

    runner = load_runner_module()

    # Run A: one topic's matched_papers.json is invalid JSON, the other's is
    # a JSON list (not an object) at the document root.
    workspace_a = write_runner_workspace(tmp_path / "with_malformed_files" / "conferences")
    index_digest_a, _hashes_a = _write_analysis_priority_fixture_store(workspace_a, run_id=run_id, runner=runner)
    invalid_json_dir = workspace_a / "storage" / "paper-matching" / run_id / invalid_json_topic_id
    invalid_json_dir.mkdir(parents=True, exist_ok=True)
    (invalid_json_dir / "matched_papers.json").write_text("{not valid json", encoding="utf-8")
    wrong_root_type_dir = workspace_a / "storage" / "paper-matching" / run_id / wrong_root_type_topic_id
    wrong_root_type_dir.mkdir(parents=True, exist_ok=True)
    (wrong_root_type_dir / "matched_papers.json").write_text(
        json.dumps(["not", "an", "object"]), encoding="utf-8"
    )

    runner.build_analysis_source_packs(
        workspace=workspace_a,
        run_id=run_id,
        stage6_receipt=stage6_receipt,
        index_digest=index_digest_a,
        topic_specs=topic_specs,
    )

    # Run B: no storage/paper-matching directory at all, for both topics.
    workspace_b = write_runner_workspace(tmp_path / "without_files" / "conferences")
    index_digest_b, _hashes_b = _write_analysis_priority_fixture_store(workspace_b, run_id=run_id, runner=runner)
    assert not (workspace_b / "storage" / "paper-matching").exists()
    runner.build_analysis_source_packs(
        workspace=workspace_b,
        run_id=run_id,
        stage6_receipt=stage6_receipt,
        index_digest=index_digest_b,
        topic_specs=topic_specs,
    )

    for topic_id in (invalid_json_topic_id, wrong_root_type_topic_id):
        pack_a = json.loads(
            (workspace_a / "analysis" / "narrative_source_packs" / f"{topic_id}.json").read_text(
                encoding="utf-8"
            )
        )
        pack_b = json.loads(
            (workspace_b / "analysis" / "narrative_source_packs" / f"{topic_id}.json").read_text(
                encoding="utf-8"
            )
        )
        chunk_ids_a = [chunk["chunk_id"] for chunk in pack_a["chunks"]]
        chunk_ids_b = [chunk["chunk_id"] for chunk in pack_b["chunks"]]
        assert chunk_ids_a == chunk_ids_b, (
            f"malformed matched_papers.json for {topic_id} changed selection vs. absent file: "
            f"{chunk_ids_a} != {chunk_ids_b}"
        )


def test_analysis_fixture_receipt_satisfies_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixture/mock end-to-end proof the stage-07 analysis receipt satisfies
    its full runtime contract and ALL 15 required_outputs entries (Plan
    07-03; repo rule D-05: validate in dry-run/mock before live; PROD-08 done
    gates 4, 5, 6, 16, 20) BEFORE any live provider spend.

    VALIDATION-ONLY: this test seeds every prerequisite receipt family
    ``run_analysis`` gates on plus the 07D/07E reconciliation receipts
    ``verify_stage_contract`` independently requires, then injects a fake
    deep-narrative client and a fake ``render_narrative_outputs_from_source_packs``
    so ZERO network/provider calls occur anywhere in this test -- no OpenAI,
    Anthropic, or Gemini request is made. A passing result here can NEVER
    substitute for the live ``stage-07-analysis.json`` receipt, which remains
    the SOLE completion proof for PROD-08 (a future live-execution plan).
    Live deep narrative synthesis spends up to $120 (registry
    ``budget_cap_usd``) -- this fixture is the mandatory pre-spend safety
    gate for that spend.
    """

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-analysis-fixture-contract"
    target = runner.run_dir(workspace, run_id)
    target.mkdir(parents=True, exist_ok=True)

    topic_ids = [f"{index:02d}_topic" for index in range(1, 13)]
    final_topics = [
        {"topic_id": topic_id, "title": f"CVPR Fixture Topic {index}"}
        for index, topic_id in enumerate(topic_ids, start=1)
    ]

    # -- Stage 6 (corpus-inventory-index): real chunk sidecars, a few cited
    # chunks per topic, matching build_analysis_source_packs' deterministic
    # keyword-substring scan (no embeddings query, no provider call) --
    vector_root = workspace / "storage" / "vector-store" / run_id
    paper_store = vector_root / "stores" / "papers"
    artifact_store = vector_root / "stores" / "artifacts"
    paper_store.mkdir(parents=True)
    artifact_store.mkdir(parents=True)
    paper_chunks = [
        {
            "chunk_id": f"paper-chunk-{index}",
            "artifact_id": f"paper_text:fixture-{index}",
            "source_path": f"https://cvpr.thecvf.com/fixture-paper-{index}",
            "text": f"Deterministic CVPR fixture topic evidence sentence {index} about the conference corpus.",
        }
        for index in range(4)
    ]
    artifact_chunks = [
        {
            "chunk_id": f"artifact-chunk-{index}",
            "artifact_id": f"speech_transcript:fixture-{index}",
            "source_path": f"https://cvpr.thecvf.com/fixture-artifact-{index}",
            "text": f"Deterministic CVPR fixture topic transcript sentence {index} about the conference corpus.",
        }
        for index in range(4)
    ]
    with (paper_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in paper_chunks:
            handle.write(json.dumps(chunk) + "\n")
    with (artifact_store / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in artifact_chunks:
            handle.write(json.dumps(chunk) + "\n")

    index_digest = {
        "stores": {
            "papers": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, paper_store / "chunks.jsonl"
                )
            },
            "artifacts": {
                "chunks_path": runner.platform_corpus_vector_store.workspace_relpath(
                    workspace, artifact_store / "chunks.jsonl"
                )
            },
        }
    }
    index_digest_path = vector_root / "index_digest.json"
    index_digest_path.write_text(json.dumps(index_digest), encoding="utf-8")
    index_digest_sha256 = runner._file_sha256(index_digest_path)

    chunk_counts = {"papers": len(paper_chunks), "artifacts": len(artifact_chunks)}
    stage6 = runner.STAGES["corpus-inventory-index"]
    (target / stage6.receipt_filename).write_text(
        json.dumps(
            {
                "status": "succeeded",
                "checks": {
                    "vector_embeddings_written": True,
                    "chunk_counts": chunk_counts,
                    "embedded_chunk_counts": chunk_counts,
                    "index_digest_sha256": index_digest_sha256,
                    "output_inventory": {"sha256": "a" * 64},
                },
            }
        ),
        encoding="utf-8",
    )

    # -- 14A legacy heuristic exhaustive-topic-extraction (required_outputs
    # by-topic / by-artifact globs, done gates 3, 9) --
    (target / "14A-exhaustive-topic-extraction-receipt.json").write_text(
        json.dumps({"status": "succeeded"}), encoding="utf-8"
    )
    by_topic_dir = workspace / "storage" / "extractions" / run_id / "by-topic"
    by_artifact_dir = workspace / "storage" / "extractions" / run_id / "by-artifact"
    by_topic_dir.mkdir(parents=True, exist_ok=True)
    by_artifact_dir.mkdir(parents=True, exist_ok=True)
    (by_topic_dir / "01_topic.json").write_text(json.dumps({"topic_id": "01_topic"}), encoding="utf-8")
    (by_artifact_dir / "artifact-1.json").write_text(
        json.dumps({"artifact_id": "artifact-1"}), encoding="utf-8"
    )

    # -- 14A Qwen full-read extraction, complete, one file per topic
    # (required_outputs llm-full-read/{by-unit,by-topic} globs, done gates
    # 2, 10) --
    llm_full_read_by_topic_dir = (
        workspace / "storage" / "extractions" / run_id / "llm-full-read" / "by-topic"
    )
    llm_full_read_by_unit_dir = (
        workspace / "storage" / "extractions" / run_id / "llm-full-read" / "by-unit"
    )
    llm_full_read_by_topic_dir.mkdir(parents=True, exist_ok=True)
    llm_full_read_by_unit_dir.mkdir(parents=True, exist_ok=True)
    for index, topic_id in enumerate(topic_ids, start=1):
        (llm_full_read_by_topic_dir / f"{topic_id}.json").write_text(
            json.dumps(
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
                            "metadata": {
                                "title": f"Fixture Paper {index}",
                                "paper_id": f"fixture-{index}",
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (llm_full_read_by_unit_dir / f"unit-{index}.json").write_text(
            json.dumps({"unit_id": f"unit-{index}", "topic_id": topic_id}), encoding="utf-8"
        )
    (target / "14A-llm-full-read-extraction-receipt.json").write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )

    # -- Title-spine topic strategy: exactly 12 final_topics (done gate 1) --
    (workspace / "analysis").mkdir(parents=True, exist_ok=True)
    (workspace / "analysis" / "topic_strategy.json").write_text(
        json.dumps({"final_topics": final_topics}), encoding="utf-8"
    )
    topic_strategy_receipt_dir = workspace / "storage" / "analysis" / "topic-strategy" / run_id
    topic_strategy_receipt_dir.mkdir(parents=True, exist_ok=True)
    (topic_strategy_receipt_dir / "topic_strategy_receipt.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "mode": "live",
                "provider_receipt_paths": ["storage/provider-calls/topic-strategy/fixture.json"],
                "final_topics": final_topics,
            }
        ),
        encoding="utf-8",
    )

    # -- 07D/07E/07H topic reconciliation + reference validation/workflow
    # receipts -- verify_stage_contract's required_receipts independently
    # demand these even though run_analysis's own pre-spend gate only reads
    # 07H (done gates 12, 13, 16) --
    (target / "07D-topic-reconciliation.json").write_text(
        json.dumps({"status": "succeeded", "source_packs_enriched": True}), encoding="utf-8"
    )
    (target / "07E-topic-reference-validation.json").write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )
    (target / "07H-topic-reconciliation-reference-workflow.json").write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )

    # -- 07B/07C deep-dive question selection + dual research lanes
    # (run_analysis's four-receipt pre-spend gate) --
    (target / "07B-deep-dive-question-selection.json").write_text(
        json.dumps({"status": "succeeded", "provider_calls_made": True}), encoding="utf-8"
    )
    (target / "07C-dual-deep-dive-research-lanes.json").write_text(
        json.dumps({"status": "succeeded", "provider_calls_made": True}), encoding="utf-8"
    )

    # -- Reconciliation output artifact families produced live by the
    # 07D->07H script chain this fixture does not invoke -- seeded directly
    # so the runtime contract's required_outputs globs are satisfied (done
    # gates 12, 13) --
    reconciliations_dir = workspace / "analysis" / "topic_reconciliations"
    ledgers_dir = workspace / "analysis" / "topic_reference_ledgers"
    validation_dir = workspace / "analysis" / "topic_reference_validation"
    reconciliations_dir.mkdir(parents=True, exist_ok=True)
    ledgers_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    for topic_id in topic_ids:
        (reconciliations_dir / f"{topic_id}.md").write_text(
            f"# Fixture Reconciliation {topic_id}\n\nDeterministic fixture reconciliation body.\n",
            encoding="utf-8",
        )
        (ledgers_dir / f"{topic_id}.json").write_text(
            json.dumps({"topic_id": topic_id, "entries": ["fixture-entry-1"]}), encoding="utf-8"
        )
        (validation_dir / f"{topic_id}.json").write_text(
            json.dumps({"topic_id": topic_id, "status": "succeeded"}), encoding="utf-8"
        )

    # -- A SECRET sentinel present in the environment (simulating a real
    # credential) must never leak into any emitted artifact --
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_SENTINEL)

    # -- Inject a fake deep-narrative client + a fake render function so ZERO
    # network/provider calls occur anywhere in this test --
    class _FixtureDeepNarrativeClient:
        model = runner.DEFAULT_DEEP_NARRATIVE_MODEL
        request_count = 0

        @classmethod
        def from_env(cls):
            return cls()

    monkeypatch.setattr(runner, "OpenAIDeepNarrativeClient", _FixtureDeepNarrativeClient)

    def _fake_render_narrative_outputs_from_source_packs(
        workspace_arg,
        *,
        run_id,
        llm_client,
        allow_deterministic_fallback,
        narrative_model,
        analysis_profile,
        reuse_existing_pass_reports,
    ):
        assert isinstance(llm_client, _FixtureDeepNarrativeClient), (
            "fake render function must only ever receive the fixture (non-network) client"
        )
        analysis_dir = workspace_arg / "analysis"
        slide_brief_dir = analysis_dir / "slide_briefs"
        provenance_dir = analysis_dir / "provenance"
        slide_brief_dir.mkdir(parents=True, exist_ok=True)
        provenance_dir.mkdir(parents=True, exist_ok=True)
        report_paths = []
        slide_brief_paths = []
        provenance_paths = []
        provenance_json_paths = []
        citation_summaries = []
        for topic_id in topic_ids:
            report_path = analysis_dir / f"{topic_id}.md"
            report_path.write_text(
                f"# Fixture Deep Narrative {topic_id}\n\n"
                f"Deterministic fixture narrative body for {topic_id}.\n",
                encoding="utf-8",
            )
            slide_brief_path = slide_brief_dir / f"{topic_id}.md"
            slide_brief_path.write_text(
                f"# Fixture Slide Brief {topic_id}\n\nDeterministic fixture slide brief.\n",
                encoding="utf-8",
            )
            provenance_path = provenance_dir / f"{topic_id}.md"
            provenance_path.write_text(
                f"# Fixture Provenance {topic_id}\n\nDeterministic fixture provenance body.\n",
                encoding="utf-8",
            )
            provenance_json_path = provenance_dir / f"{topic_id}.json"
            provenance_json_path.write_text(
                json.dumps({"topic_id": topic_id, "citations": ["fixture-citation-1"]}),
                encoding="utf-8",
            )
            report_paths.append(report_path)
            slide_brief_paths.append(slide_brief_path)
            provenance_paths.append(provenance_path)
            provenance_json_paths.append(provenance_json_path)
            citation_summaries.append(
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
        citation_manifest_path = analysis_dir / "analysis_citation_manifest.json"
        citation_manifest_path.write_text(
            json.dumps({"topics": citation_summaries}), encoding="utf-8"
        )
        return SimpleNamespace(
            status="succeeded",
            report_paths=tuple(report_paths),
            slide_brief_paths=tuple(slide_brief_paths),
            provenance_paths=tuple(provenance_paths),
            provenance_json_paths=tuple(provenance_json_paths),
            citation_manifest_path=citation_manifest_path,
            citation_summaries=tuple(citation_summaries),
            blockers=(),
        )

    monkeypatch.setattr(
        runner,
        "render_narrative_outputs_from_source_packs",
        _fake_render_narrative_outputs_from_source_packs,
    )

    stamp_upstream_stage_receipts(target)
    outcome = runner.run_analysis(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
    )

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    receipt = outcome["receipt"]
    checks = receipt["checks"]

    assert checks["source_packs"]["insufficient_topics"] == []
    assert checks["source_packs"]["source_pack_count"] == 12
    assert checks["topic_strategy_topic_count"] == 12

    # -- Contract coupling: read required_checks from the registry file
    # itself so this assertion cannot silently drift from the declared
    # contract (T-07V-02) --
    registry_path = REPO_ROOT / "src" / "conference_pipeline" / "stage_owner_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    analysis_binding = registry["stage_bindings"]["analysis"]["runtime_contract"]
    required_checks = analysis_binding["required_checks"]
    assert required_checks, "registry must declare at least one required_check"
    for spec in required_checks:
        _assert_required_check(receipt, spec)

    # Prove the coupling is real, not tautological: dropping any one
    # required_check path from a copy of the receipt must make the SAME
    # assertion function fail.
    for spec in required_checks:
        mutated = copy.deepcopy(receipt)
        path_parts = str(spec["path"]).split(".")
        cursor = mutated
        for part in path_parts[:-1]:
            cursor = cursor[part]
        cursor.pop(path_parts[-1], None)
        with pytest.raises(AssertionError):
            _assert_required_check(mutated, spec)

    # -- ALL 15 required_outputs entries, iterated straight from the
    # registry file itself (not a hand-picked subset -- T-07V-02) --
    required_outputs = analysis_binding["required_outputs"]
    assert len(required_outputs) == 15
    for spec in required_outputs:
        min_count = int(spec.get("min_count", 1) or 1)
        if "glob" in spec:
            pattern = str(spec["glob"]).replace("<run-id>", run_id)
            matches = sorted(workspace.glob(pattern))
            assert len(matches) >= min_count, (
                f"{pattern} expected >= {min_count} matches, found {len(matches)}"
            )
        else:
            output_path = workspace / str(spec["path"]).replace("<run-id>", run_id)
            assert output_path.is_file(), f"expected required output file at {output_path}"

    # -- verify_stage_contract(strict_execution=True) independently over the
    # persisted receipt returns status pass --
    independent_contract = stage_contracts.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="analysis",
        receipt=receipt,
        mode="live",
        strict_execution=True,
    ).to_dict()
    assert independent_contract["status"] == "pass", independent_contract["blockers"]
    assert receipt["stage_contract"]["status"] == "pass"

    # -- done gate 6 / RESEARCH Pitfall 3 (resolved): the recorded narrative
    # model is the registry-resolved sol value, never a runner-code literal --
    assert checks["narrative_model"] == runner.DEFAULT_DEEP_NARRATIVE_MODEL

    # -- no live provider call occurred anywhere in this fixture --
    assert checks["provider_calls_made"] is False
    assert checks["live_provider_calls_made"] is False

    # -- SECRET_SENTINEL must never appear in the receipt, analysis manifest,
    # or citation manifest (credential-handling regression) --
    analysis_manifest_path = workspace / "analysis" / "analysis_manifest.json"
    citation_manifest_path = workspace / "analysis" / "analysis_citation_manifest.json"
    for artifact_text in (
        json.dumps(receipt, default=str),
        analysis_manifest_path.read_text(encoding="utf-8"),
        citation_manifest_path.read_text(encoding="utf-8"),
    ):
        assert SECRET_SENTINEL not in artifact_text

    # -- Pre-spend lineage gate (v5.1): Stage 6 re-ran after extraction, so
    # 14A is stale. A re-run of analysis must block before the paid client is
    # constructed, not spend and then fail its final contract --
    extraction_path = target / "14A-exhaustive-topic-extraction-receipt.json"
    stage6_key = f"storage/workflow-runs/{run_id}/{stage6.receipt_filename}"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction["lineage"] = {
        **extraction["lineage"],
        "upstream_receipts": {
            stage6_key: {"stage_id": "corpus-inventory-index", "digest_kind": "output_digest", "digest": "0" * 64}
        },
    }
    extraction_path.write_text(json.dumps(extraction), encoding="utf-8")

    class _MustNotConstruct:
        @classmethod
        def from_env(cls):
            raise AssertionError("paid analysis client constructed over a stale upstream")

    monkeypatch.setattr(runner, "OpenAIDeepNarrativeClient", _MustNotConstruct)
    blocked = runner.run_analysis(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )
    assert blocked["status"] == "blocked"
    assert [item["code"] for item in blocked["receipt"]["blockers"]] == ["upstream_lineage_not_fresh"]
    gate = blocked["receipt"]["checks"]["upstream_lineage_blockers"]
    assert [(item["code"], item["receipt"]) for item in gate] == [
        ("stage_contract_upstream_stale", f"storage/workflow-runs/{run_id}/14A-exhaustive-topic-extraction-receipt.json")
    ], gate
    assert blocked["receipt"]["checks"]["provider_calls_made"] is False


# ---------------------------------------------------------------------------
# Phase 8 (qa-gap-fill-synthesis) mock-before-live gate (Plan 08-03).
#
# Every test below is a fixture/mock proof: a fake, non-network
# Stage8SynthesisClient stand-in is injected so no test in this section ever
# imports ``anthropic``/``openai`` or opens a real connection (RESEARCH
# Pattern 1 / T-08-08). A passing result in this section can NEVER
# substitute for the live stage-08-qa-gap-fill-synthesis.json receipt
# (Plan 08-04), which remains the SOLE completion evidence for PROD-09/
# PROD-10 (roadmap Rejected evidence).
# ---------------------------------------------------------------------------


def _write_qa_gap_fill_synthesis_predecessors(
    runner: object, workspace: Path, run_id: str, *, topic_count: int = 3
) -> dict[str, object]:
    """Seed all four upstream Phase 7 receipts, both Stage 7 manifests, and
    the per-topic on-disk artifacts (markdown/provenance/slide-brief files +
    narrative_source_packs) so ``run_qa_gap_fill_synthesis_stage``'s
    fail-closed input gate AND ``build_stage8_traceability_reports`` both
    pass cleanly for every topic. Mirrors
    ``test_analysis_fixture_receipt_satisfies_runtime_contract``'s seeding
    style, adapted to Stage 8's own required inputs (08-VALIDATION.md;
    08-RESEARCH.md Architecture Diagram).

    Returns a dict of paths/ids the calling test can mutate, corrupt, or
    delete to exercise a specific failure mode.
    """

    target = runner.run_dir(workspace, run_id)
    target.mkdir(parents=True, exist_ok=True)
    analysis_dir = workspace / "analysis"
    provenance_dir = analysis_dir / "provenance"
    slide_brief_dir = analysis_dir / "slide_briefs"
    source_pack_dir = workspace / runner.ANALYSIS_SOURCE_PACK_DIR
    for directory in (analysis_dir, provenance_dir, slide_brief_dir, source_pack_dir):
        directory.mkdir(parents=True, exist_ok=True)

    topic_ids = [f"{index:02d}_fixture_topic" for index in range(1, topic_count + 1)]
    manifest_topics: list[dict[str, object]] = []
    citation_topics: list[dict[str, object]] = []
    markdown_paths: dict[str, Path] = {}

    for index, topic_id in enumerate(topic_ids, start=1):
        markdown_path = analysis_dir / f"{topic_id}.md"
        markdown_path.write_text(
            f"# Fixture Topic {index}\n\nDeterministic fixture body for {topic_id}.\n",
            encoding="utf-8",
        )
        (provenance_dir / f"{topic_id}.md").write_text(
            f"# Fixture Provenance {topic_id}\n", encoding="utf-8"
        )
        (provenance_dir / f"{topic_id}.json").write_text(
            json.dumps({"topic_id": topic_id}), encoding="utf-8"
        )
        (slide_brief_dir / f"{topic_id}.md").write_text(
            f"# Fixture Slide Brief {topic_id}\n", encoding="utf-8"
        )
        markdown_paths[topic_id] = markdown_path

        markdown_rel = f"analysis/{topic_id}.md"
        provenance_rel = f"analysis/provenance/{topic_id}.md"
        provenance_json_rel = f"analysis/provenance/{topic_id}.json"
        slide_brief_rel = f"analysis/slide_briefs/{topic_id}.md"

        manifest_topics.append(
            {
                "id": topic_id,
                "title": f"Fixture Topic {index}",
                "markdown_path": markdown_rel,
                "markdown_sha256": runner._file_sha256(markdown_path),
                "provenance_json_path": provenance_json_rel,
                "provenance_path": provenance_rel,
                "slide_brief_path": slide_brief_rel,
                "deep_validation": {"status": "pass"},
                "narrative_model": runner.DEFAULT_DEEP_NARRATIVE_MODEL,
            }
        )
        citation_topics.append(
            {
                "topic_id": topic_id,
                "markdown_path": markdown_rel,
                "provenance_path": provenance_rel,
                "provenance_json_path": provenance_json_rel,
                "slide_brief_path": slide_brief_rel,
                "narrative_model": runner.DEFAULT_DEEP_NARRATIVE_MODEL,
                "analysis_profile": "fixture-profile",
                "deep_validation": {"status": "pass"},
                "extraction_record_ids": [
                    f"fixture-extraction-{topic_id}-{n}" for n in range(production_qa_synthesis_stage.STAGE8_EXTRACTION_ID_FLOOR)
                ],
                "vector_store_chunk_ids": [f"fixture-chunk-{topic_id}-1"],
                "research_ids": [f"fixture-research-{topic_id}-1"],
            }
        )
        (source_pack_dir / f"{topic_id}.json").write_text(
            json.dumps(
                {
                    "topic_id": topic_id,
                    "topic_reconciliation": {
                        "status": "succeeded",
                        "model": "fixture-reconciliation-model",
                        "reference_validation_status": "succeeded",
                        "reference_validation_counts": {
                            "citation_count": 1,
                            "ledger_entry_count": 1,
                            "invalid_reference_count": 0,
                            "unknown_reference_count": 0,
                            "title_mismatch_count": 0,
                            "bare_reference_warning_count": 0,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    analysis_manifest_content_sha256 = hashlib.sha256(
        json.dumps(manifest_topics, sort_keys=True).encode("utf-8")
    ).hexdigest()
    analysis_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "content_sha256": analysis_manifest_content_sha256,
        "topics": manifest_topics,
    }
    analysis_manifest_path = workspace / runner.ANALYSIS_MANIFEST
    analysis_manifest_path.write_text(json.dumps(analysis_manifest, sort_keys=True), encoding="utf-8")

    citation_manifest = {"schema_version": 1, "run_id": run_id, "topics": citation_topics}
    citation_manifest_path = workspace / runner.ANALYSIS_CITATION_MANIFEST
    citation_manifest_path.write_text(json.dumps(citation_manifest, sort_keys=True), encoding="utf-8")

    stage7_receipt_path = target / "stage-07-analysis.json"
    stage7_receipt_path.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "stage_contract": {"status": "pass"},
                "checks": {"analysis_manifest_content_sha256": analysis_manifest_content_sha256},
            }
        ),
        encoding="utf-8",
    )
    topic_reconciliation_path = target / "07D-topic-reconciliation.json"
    topic_reconciliation_path.write_text(
        json.dumps({"status": "succeeded", "topic_count": len(topic_ids)}), encoding="utf-8"
    )
    topic_reference_validation_path = target / "07E-topic-reference-validation.json"
    topic_reference_validation_path.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "checks": {
                    "invalid_reference_count": 0,
                    "unknown_reference_count": 0,
                    "title_mismatch_count": 0,
                    "bare_reference_warning_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    topic_reference_workflow_path = target / "07H-topic-reconciliation-reference-workflow.json"
    topic_reference_workflow_path.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "checks": {
                    "final_validation_status": "succeeded",
                    "final_invalid_reference_count": 0,
                    "final_unknown_reference_count": 0,
                    "final_title_mismatch_count": 0,
                    "final_bare_reference_warning_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    stamp_upstream_stage_receipts(workspace / "storage" / "workflow-runs" / run_id)
    return {
        "topic_ids": topic_ids,
        "target": target,
        "analysis_manifest_path": analysis_manifest_path,
        "citation_manifest_path": citation_manifest_path,
        "stage7_receipt_path": stage7_receipt_path,
        "topic_reconciliation_path": topic_reconciliation_path,
        "topic_reference_validation_path": topic_reference_validation_path,
        "topic_reference_workflow_path": topic_reference_workflow_path,
        "source_pack_dir": source_pack_dir,
        "markdown_paths": markdown_paths,
    }


def _stage8_default_synthesis_provider_model(runner: object) -> tuple[str, str]:
    """Resolve the real registry provider/model for ``SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID``.

    Used only so the fixture ``Stage8SynthesisClient`` stand-in declares a
    provider/model pair ``estimate_stage8_synthesis_cost`` can actually price
    against the real catalog -- an unrecognized provider/model pair raises
    ``KeyError`` from the pricing catalog, an unrelated failure mode this
    fixture must not trip.
    """

    from conference_pipeline.model_governance import default_model, default_provider

    task_id = runner.SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID
    return default_provider(task_id), default_model(task_id)


def _build_fixture_stage8_synthesis_client_class(
    runner: object, *, instances: list[object] | None = None
) -> type:
    """Build a non-network stand-in for ``Stage8SynthesisClient`` (RESEARCH
    Pattern 1 / T-08-08): returns a fixed, sufficiently long markdown body
    with real registry provider/model values, but never imports
    ``anthropic``/``openai`` or opens a network connection. Every constructed
    instance is appended to ``instances`` (if supplied) so a test can assert
    exactly how many times -- and with which ``kind`` values -- it was used.
    """

    provider, model = _stage8_default_synthesis_provider_model(runner)
    tracked = instances if instances is not None else []

    class _FixtureStage8SynthesisClient:
        def __init__(self, *, workspace: Path, run_id: str, **_ignored: object) -> None:
            self.workspace = workspace
            self.run_id = run_id
            self.provider = provider
            self.model = model
            self.request_count = 0
            self.calls: list[str] = []
            tracked.append(self)

        def synthesize(self, *, kind: str, system_prompt: str, user_prompt: str) -> SimpleNamespace:
            del system_prompt, user_prompt
            self.calls.append(kind)
            self.request_count += 1
            body = f"Fixture {kind} synthesis body sentence. " * 200
            return SimpleNamespace(text=body, input_tokens=100, output_tokens=100)

    return _FixtureStage8SynthesisClient


def test_qa_gap_fill_synthesis_blocks_on_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROD-09: the stage fails closed (never fabricates a pass) with zero
    provider spend when one of the 4 upstream Phase 7 receipts is missing --
    here, the ``stage-07-analysis.json`` receipt itself (08-VALIDATION.md row
    08-01-01)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-qa-gap-fill-synthesis-missing-input"
    handles = _write_qa_gap_fill_synthesis_predecessors(runner, workspace, run_id, topic_count=3)

    class _ExplodingStage8SynthesisClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError(
                "Stage8SynthesisClient must never be constructed when a required "
                "upstream receipt is missing"
            )

    monkeypatch.setattr(runner, "Stage8SynthesisClient", _ExplodingStage8SynthesisClient)
    handles["stage7_receipt_path"].unlink()

    outcome = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )

    assert outcome["status"] == "blocked"
    receipt = outcome["receipt"]
    assert receipt["checks"]["provider_calls_made"] is False
    assert receipt["blockers"][0]["code"] == "stage8_required_inputs_blocked"
    input_blockers = receipt["checks"]["input_blockers"]
    assert any(blocker["code"] == "stage8_input_missing" for blocker in input_blockers)
    assert any(blocker.get("input") == "stage_07_analysis_receipt" for blocker in input_blockers)


def test_qa_gap_fill_synthesis_coverage_gap_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROD-09: a topic whose on-disk markdown digest no longer matches its
    manifest-recorded ``markdown_sha256`` is classified into a
    non-``well_covered`` gap class and the corresponding report status is
    not ``pass`` -- never fabricated as passing (mirrors
    ``build_stage8_traceability_reports``'s per-topic digest check;
    08-VALIDATION.md row 08-01-02)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-qa-gap-fill-synthesis-coverage-gap"
    handles = _write_qa_gap_fill_synthesis_predecessors(runner, workspace, run_id, topic_count=3)

    class _ExplodingStage8SynthesisClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError(
                "Stage8SynthesisClient must never be constructed once the deterministic "
                "traceability/coverage-gap gate has already blocked"
            )

    monkeypatch.setattr(runner, "Stage8SynthesisClient", _ExplodingStage8SynthesisClient)

    corrupted_topic_id = handles["topic_ids"][0]
    handles["markdown_paths"][corrupted_topic_id].write_text(
        "# Corrupted fixture body\n\nThis no longer matches the manifest digest.\n",
        encoding="utf-8",
    )

    outcome = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )

    assert outcome["status"] == "blocked"
    receipt = outcome["receipt"]
    assert receipt["checks"]["provider_calls_made"] is False
    assert receipt["checks"]["blocker_code"] == "stage8_traceability_or_coverage_blocked"
    assert receipt["checks"]["traceability_report_status"] != "pass"
    assert receipt["checks"]["coverage_gap_report_status"] != "pass"

    coverage_report = json.loads(
        (workspace / runner.ANALYSIS_COVERAGE_GAP_REPORT).read_text(encoding="utf-8")
    )
    corrupted_entry = next(
        topic for topic in coverage_report["topics"] if topic["topic_id"] == corrupted_topic_id
    )
    assert corrupted_entry["gap_class"] in {"under_covered", "missing_evidence", "gap"}
    assert corrupted_entry["status"] != "pass"
    other_entries = [
        topic for topic in coverage_report["topics"] if topic["topic_id"] != corrupted_topic_id
    ]
    assert other_entries, "fixture must seed more than one topic to prove per-topic isolation"
    assert all(topic["gap_class"] == "well_covered" for topic in other_entries)


def test_qa_gap_fill_synthesis_synthesis_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROD-10: ``validate_stage8_synthesis`` genuinely gates synthesis
    success -- a synthesized body lacking the required citation-floor
    mentions and digest bindings blocks the stage (``executive_summary_status``/
    ``trend_report_status`` not ``pass``), never fabricating a pass. This
    test monkeypatches ``append_stage8_source_anchor`` to identity (no
    anchor) to isolate ``validate_stage8_synthesis``'s own gate from Plan
    08-02's deterministic source-anchor mitigation -- proving the validator
    itself, not just the anchor, is what stands between a bare/uncited model
    response and a fabricated pass (08-VALIDATION.md row 08-01-04)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-qa-gap-fill-synthesis-validation"
    _write_qa_gap_fill_synthesis_predecessors(runner, workspace, run_id, topic_count=3)

    from conference_pipeline.production import qa_synthesis_stage

    monkeypatch.setattr(qa_synthesis_stage, "append_stage8_source_anchor", lambda text, **_kwargs: text)
    fixture_client_class = _build_fixture_stage8_synthesis_client_class(runner)

    class _UncitedStage8SynthesisClient(fixture_client_class):  # type: ignore[misc, valid-type]
        def synthesize(self, *, kind: str, system_prompt: str, user_prompt: str) -> SimpleNamespace:
            del system_prompt, user_prompt
            self.calls.append(kind)
            self.request_count += 1
            return SimpleNamespace(
                text="A bare synthesized body with no citations or digests whatsoever.",
                input_tokens=10,
                output_tokens=10,
            )

    monkeypatch.setattr(runner, "Stage8SynthesisClient", _UncitedStage8SynthesisClient)

    outcome = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )

    assert outcome["status"] == "blocked"
    checks = outcome["receipt"]["checks"]
    assert checks["blocker_code"] == "stage8_synthesis_validation_blocked"
    assert checks.get("executive_summary_status") != "pass"
    assert checks.get("trend_report_status") != "pass"
    synthesis_blocker_codes = {blocker["code"] for blocker in checks["synthesis_blockers"]}
    assert synthesis_blocker_codes & {
        "stage8_synthesis_topic_report_floor_unmet",
        "stage8_synthesis_extraction_floor_unmet",
        "stage8_synthesis_research_floor_unmet",
        "stage8_synthesis_digest_binding_missing",
        "stage8_synthesis_output_too_short",
    }


def test_qa_gap_fill_synthesis_resume_skips_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RESEARCH Pitfall 5: a second ``resume=True`` invocation over an
    already-succeeded, still-passing Stage 8 receipt must never reconstruct
    ``Stage8SynthesisClient`` -- otherwise every repeat ``--resume``
    invocation re-spends money on the paid executive-summary/trend-report
    synthesis (08-VALIDATION.md row 08-01-06)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-qa-gap-fill-synthesis-resume"
    _write_qa_gap_fill_synthesis_predecessors(runner, workspace, run_id, topic_count=3)

    monkeypatch.setattr(
        runner, "Stage8SynthesisClient", _build_fixture_stage8_synthesis_client_class(runner)
    )
    first_outcome = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )
    assert first_outcome["status"] == "succeeded", first_outcome["receipt"]["blockers"]

    class _ExplodingStage8SynthesisClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError(
                "Stage8SynthesisClient must not be constructed on a resume short-circuit "
                "over an already-succeeded, still-passing receipt"
            )

    monkeypatch.setattr(runner, "Stage8SynthesisClient", _ExplodingStage8SynthesisClient)

    second_outcome = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )
    assert second_outcome["status"] == "succeeded"
    assert second_outcome["receipt"]["receipt_id"] == first_outcome["receipt"]["receipt_id"]


def test_qa_gap_fill_synthesis_owner_execution_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RESEARCH Pitfall 3 / Pattern 2 (T-08-01): the ``strict_execution``
    owner-evidence dict has one real, non-echoed token per
    ``required_script_ids`` entry, and a bare-echo variant (token == owner
    id) is rejected by ``verify_stage_contract``'s strict mode
    (08-VALIDATION.md row 08-01-07)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-qa-gap-fill-synthesis-owner-evidence"
    _write_qa_gap_fill_synthesis_predecessors(runner, workspace, run_id, topic_count=3)
    monkeypatch.setattr(
        runner, "Stage8SynthesisClient", _build_fixture_stage8_synthesis_client_class(runner)
    )

    outcome = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )
    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    receipt = outcome["receipt"]
    required_script_ids = list(
        stage_script_registry.required_script_ids_for_stage(
            load_stage_script_registry(), "qa-gap-fill-synthesis"
        )
    )
    assert required_script_ids == [
        "analysis.coverage_gap_fill",
        "analysis.synthesis_trend_report",
    ]
    evidence = receipt["owner_execution_evidence"]
    assert set(evidence.keys()) >= set(required_script_ids)
    for owner_id in required_script_ids:
        token = evidence[owner_id]
        assert token, f"{owner_id} must have a non-empty evidence token"
        assert token != owner_id, f"{owner_id} evidence must not be a bare echo of the owner id"

    independent_contract = stage_contracts.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="qa-gap-fill-synthesis",
        receipt=receipt,
        mode="live",
        strict_execution=True,
    ).to_dict()
    assert independent_contract["status"] == "pass", independent_contract["blockers"]

    # Bare-echo mutation must be rejected by strict_execution mode.
    mutated_receipt = copy.deepcopy(receipt)
    mutated_receipt["owner_execution_evidence"] = {
        owner_id: owner_id for owner_id in required_script_ids
    }
    mutated_contract = stage_contracts.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="qa-gap-fill-synthesis",
        receipt=mutated_receipt,
        mode="live",
        strict_execution=True,
    ).to_dict()
    assert mutated_contract["status"] != "pass"
    assert any(
        blocker["code"] == "stage_contract_owner_execution_evidence_missing"
        for blocker in mutated_contract["blockers"]
    )


def test_qa_gap_fill_synthesis_stage_implemented_and_dispatches_from_main(tmp_path: Path) -> None:
    """STAGES['qa-gap-fill-synthesis'] is flipped to implemented=True and
    main() routes stage.id == 'qa-gap-fill-synthesis' to
    run_qa_gap_fill_synthesis_stage (mirrors
    test_analysis_stage_implemented_and_dispatches_from_main;
    08-VALIDATION.md row 08-01-05)."""

    runner = load_runner_module()
    stage = runner.STAGES["qa-gap-fill-synthesis"]
    assert stage.order == 8
    assert stage.receipt_filename == "stage-08-qa-gap-fill-synthesis.json"
    assert stage.implemented is True
    assert runner.orchestrator_spec().handlers["qa-gap-fill-synthesis"] is runner.dispatch_qa_gap_fill_synthesis


def test_qa_gap_fill_synthesis_fixture_receipt_satisfies_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixture/mock end-to-end proof the stage-08 qa-gap-fill-synthesis
    receipt satisfies its full runtime contract (Plan 08-03; repo rule:
    validate in mock before live; PROD-09/PROD-10 done gates 1-3) BEFORE any
    live provider spend.

    VALIDATION-ONLY: this test seeds all 4 upstream Phase 7 receipts + both
    Stage 7 manifests + the per-topic on-disk artifacts in a tmp_path
    workspace, then injects a fake, non-network Stage8SynthesisClient so ZERO
    provider/network call occurs anywhere in this test -- no Anthropic or
    OpenAI request is made. A passing result here can NEVER substitute for
    the live ``stage-08-qa-gap-fill-synthesis.json`` receipt, which remains
    the SOLE completion proof for PROD-09/PROD-10 (Plan 08-04, a future
    live-execution plan; roadmap Rejected evidence)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-qa-gap-fill-synthesis-fixture-contract"
    _write_qa_gap_fill_synthesis_predecessors(runner, workspace, run_id, topic_count=3)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_SENTINEL)
    provider, model = _stage8_default_synthesis_provider_model(runner)
    instances: list[object] = []
    monkeypatch.setattr(
        runner,
        "Stage8SynthesisClient",
        _build_fixture_stage8_synthesis_client_class(runner, instances=instances),
    )

    outcome = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    receipt = outcome["receipt"]
    checks = receipt["checks"]

    # -- Zero network/provider calls: exactly one fake client instance was
    # constructed and used for both synthesis kinds. --
    assert len(instances) == 1
    assert instances[0].calls == ["executive_summary", "trend_report"]
    assert checks["synthesis_provider"] == provider
    assert checks["synthesis_model"] == model
    assert checks["model_governance_task_id"] == runner.SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID

    # -- Contract coupling: read required_checks from the registry file
    # itself so this assertion cannot silently drift from the declared
    # contract (mirrors test_analysis_fixture_receipt_satisfies_runtime_contract). --
    registry_path = REPO_ROOT / "src" / "conference_pipeline" / "stage_owner_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    binding = registry["stage_bindings"]["qa-gap-fill-synthesis"]["runtime_contract"]
    required_checks = binding["required_checks"]
    assert required_checks, "registry must declare at least one required_check"
    for spec in required_checks:
        _assert_required_check(receipt, spec)

    # Prove the coupling is real, not tautological: dropping any one
    # required_check path from a copy of the receipt must make the SAME
    # assertion function fail.
    for spec in required_checks:
        mutated = copy.deepcopy(receipt)
        path_parts = str(spec["path"]).split(".")
        cursor = mutated
        for part in path_parts[:-1]:
            cursor = cursor[part]
        cursor.pop(path_parts[-1], None)
        with pytest.raises(AssertionError):
            _assert_required_check(mutated, spec)

    # -- ALL 4 required_outputs entries, iterated straight from the registry
    # file itself (not a hand-picked subset). --
    required_outputs = binding["required_outputs"]
    assert len(required_outputs) == 4
    for spec in required_outputs:
        output_path = workspace / str(spec["path"])
        assert output_path.is_file(), f"expected required output file at {output_path}"

    # -- verify_stage_contract(strict_execution=True) independently over the
    # persisted receipt returns status pass. --
    independent_contract = stage_contracts.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="qa-gap-fill-synthesis",
        receipt=receipt,
        mode="live",
        strict_execution=True,
    ).to_dict()
    assert independent_contract["status"] == "pass", independent_contract["blockers"]
    assert receipt["stage_contract"]["status"] == "pass"

    # -- SECRET_SENTINEL must never appear in the receipt, either manifest,
    # or either synthesized output (credential-handling regression). --
    analysis_manifest_path = workspace / runner.ANALYSIS_MANIFEST
    citation_manifest_path = workspace / runner.ANALYSIS_CITATION_MANIFEST
    executive_summary_path = workspace / runner.ANALYSIS_EXECUTIVE_SUMMARY
    trend_report_path = workspace / runner.ANALYSIS_TREND_REPORT
    for artifact_text in (
        json.dumps(receipt, default=str),
        analysis_manifest_path.read_text(encoding="utf-8"),
        citation_manifest_path.read_text(encoding="utf-8"),
        executive_summary_path.read_text(encoding="utf-8"),
        trend_report_path.read_text(encoding="utf-8"),
    ):
        assert SECRET_SENTINEL not in artifact_text


def _write_podcast_predecessors(
    runner: object,
    workspace: Path,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    topic_count: int = 3,
) -> dict[str, object]:
    """Produce a genuine, passing Stage 8 receipt + ``executive_summary.md``/
    ``trend_report.md``/``analysis_manifest.json`` by actually running
    ``run_qa_gap_fill_synthesis_stage`` with the non-network fixture
    ``Stage8SynthesisClient`` stand-in -- so Stage 9's own fail-closed input
    gate (PROD-11 done gate 1) reads real, digest-consistent Phase 8 outputs
    rather than a hand-assembled stand-in receipt. Mirrors ICLR2026's own
    ``write_stage8_fixture`` precedent for exercising this stage's tests
    (09-RESEARCH.md Test-mocking pattern; 09-VALIDATION.md).
    """

    handles = _write_qa_gap_fill_synthesis_predecessors(runner, workspace, run_id, topic_count=topic_count)
    monkeypatch.setattr(
        runner, "Stage8SynthesisClient", _build_fixture_stage8_synthesis_client_class(runner)
    )
    stage8_outcome = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )
    assert stage8_outcome["status"] == "succeeded", stage8_outcome["receipt"]["blockers"]
    handles["stage8_outcome"] = stage8_outcome
    stamp_upstream_stage_receipts(workspace / "storage" / "workflow-runs" / run_id)
    return handles


def _build_fake_podcast_command_runner(
    runner: object, *, call_log: list[str] | None = None
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    """Port ICLR2026's ``fake_command_runner`` (L4210-4257) as the zero-spend
    subprocess-interception template: scans the constructed command list for
    'scripts/generate_focused_podcast.py'/'scripts/render_podcast_audio.py'
    and writes exactly the output files each owner script is expected to
    produce -- no real subprocess, network, LLM, or TTS call occurs. Appends
    'script'/'audio' to ``call_log`` (if supplied) so a test can assert
    exactly how many times -- and in what order -- each was intercepted.
    """

    log = call_log if call_log is not None else []

    def fake_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if "scripts/generate_focused_podcast.py" in command:
            log.append("script")
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            script = {
                "title": "Test podcast",
                "summary": "A source-backed test podcast.",
                "segments": [
                    {
                        "speaker": "Maya",
                        "text": " ".join(["This is a source-backed podcast segment."] * 30),
                    },
                    {
                        "speaker": "Leo",
                        "text": " ".join(["This segment continues the verified analysis."] * 30),
                    },
                ],
                "show_notes": "Generated from verified reports.",
            }
            (output_dir / "script.json").write_text(
                json.dumps(script, indent=2) + "\n", encoding="utf-8"
            )
            (output_dir / "script.md").write_text("# Test podcast\n", encoding="utf-8")
            (output_dir / "show_notes.md").write_text("# Show notes\n", encoding="utf-8")
            (output_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "script ok", "")
        if "scripts/render_podcast_audio.py" in command:
            log.append("audio")
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_path = output_dir / "podcast.mp3"
            audio_path.write_bytes(b"fake-mp3-bytes")
            audio_manifest = {
                "schema_version": 1,
                "script_json_path": str(output_dir / "script.json"),
                "podcast_path": str(audio_path),
                "podcast_sha256": runner._file_sha256(audio_path),
                "podcast_byte_count": audio_path.stat().st_size,
                "segment_count": 2,
                "audio_file_count": 2,
                "provider_call_count": 2,
                "estimated_cost_usd": 0.01,
                "secrets_copied": False,
            }
            (output_dir / "audio_manifest.json").write_text(
                json.dumps(audio_manifest, indent=2) + "\n", encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, "audio ok", "")
        raise AssertionError(command)

    return fake_command_runner


def _fake_podcast_provider_receipt_summary(
    workspace: Path, run_id: str | None = None, stage_id: str | None = None
) -> dict[str, object]:
    """Non-network ``provider_receipt_summary`` stand-in: reports 2 canonical
    provider calls scanned (matching the 2 fake subprocess interceptions)
    without ever scanning a real receipt file or making a network call."""

    del workspace, run_id, stage_id
    return {
        "canonical_receipt_files_scanned": 2,
        "canonical_total_estimated_cost_usd": 0.02,
        "canonical_totals_by_provider": {"openai": 0.02},
        "canonical_totals_by_model": {"openai:gpt-4o-mini-tts": 0.02},
        "canonical_totals_by_operation": {"tts": 0.02},
        "failed_receipts": 0,
        "unknown_cost_receipts": 0,
        "receipt_paths": ["provider-call-1.json", "provider-call-2.json"],
    }


def test_podcast_blocks_on_missing_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PROD-11 done gate 1: the stage fails closed (never fabricates a pass)
    with zero provider spend when the Stage 8 receipt, the analysis
    manifest, the executive summary, or the trend report is absent -- and
    the fake ``command_runner`` (exploding if invoked) proves neither
    subprocess is ever constructed (09-VALIDATION.md row 09-01-01)."""

    runner = load_runner_module()

    def exploding_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(
            f"command_runner must never be invoked when required podcast inputs "
            f"are missing/not-succeeded: {command}"
        )

    for label in ("stage8_receipt", "analysis_manifest", "executive_summary", "trend_report"):
        workspace = write_runner_workspace(tmp_path / f"conferences-missing-{label}")
        run_id = f"test-podcast-missing-{label}"
        _write_podcast_predecessors(runner, workspace, run_id, monkeypatch)
        input_paths = production_podcast_stage.required_input_paths(runner._production_context(workspace, run_id))
        input_paths[label].unlink()

        outcome = runner.run_podcast_stage(
            workspace=workspace,
            run_id=run_id,
            resume=True,
            provider_cost_approved=True,
            env_file=None,
            command_runner=exploding_command_runner,
        )

        assert outcome["status"] == "blocked", label
        receipt = outcome["receipt"]
        assert receipt["status"] == "blocked", label
        assert receipt["checks"]["provider_calls_made"] is False, label
        assert receipt["blockers"][0]["code"] == "podcast_required_inputs_blocked", label
        input_blockers = receipt["checks"]["input_blockers"]
        assert input_blockers, label


def test_podcast_blocks_on_missing_provider_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROD-11: the stage blocks before either subprocess is invoked when
    ``provider_cost_approved`` is False, and never auto-raises the cap
    (09-VALIDATION.md row 09-01-02)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-podcast-missing-approval"
    _write_podcast_predecessors(runner, workspace, run_id, monkeypatch)

    def exploding_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(
            f"command_runner must never be invoked without provider-cost approval: {command}"
        )

    outcome = runner.run_podcast_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=False,
        env_file=None,
        command_runner=exploding_command_runner,
    )

    assert outcome["status"] == "blocked"
    receipt = outcome["receipt"]
    assert receipt["checks"]["provider_calls_made"] is False
    assert receipt["blockers"][0]["code"] == "podcast_provider_cost_not_approved"
    assert receipt["checks"]["provider_cost_approval"]["approved"] is False


def test_podcast_script_only_cannot_satisfy_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROD-11 done gate 2: a fake runner that satisfies the script
    subprocess but fails the audio subprocess (no ``podcast.mp3``) cannot
    produce a succeeded receipt -- ``podcast_audio_status`` is not ``pass``
    and ``podcast_audio_sha256`` is falsy (09-VALIDATION.md row 09-01-04)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-podcast-script-only"
    _write_podcast_predecessors(runner, workspace, run_id, monkeypatch)

    def script_only_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if "scripts/generate_focused_podcast.py" in command:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            script = {
                "title": "Test podcast",
                "summary": "A source-backed test podcast.",
                "segments": [
                    {
                        "speaker": "Maya",
                        "text": " ".join(["This is a source-backed podcast segment."] * 30),
                    },
                    {
                        "speaker": "Leo",
                        "text": " ".join(["This segment continues the verified analysis."] * 30),
                    },
                ],
                "show_notes": "Generated from verified reports.",
            }
            (output_dir / "script.json").write_text(
                json.dumps(script, indent=2) + "\n", encoding="utf-8"
            )
            (output_dir / "script.md").write_text("# Test podcast\n", encoding="utf-8")
            (output_dir / "show_notes.md").write_text("# Show notes\n", encoding="utf-8")
            (output_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "script ok", "")
        if "scripts/render_podcast_audio.py" in command:
            # Simulate a failed TTS render: nonzero returncode, no podcast.mp3.
            return subprocess.CompletedProcess(command, 1, "", "audio render failed (fixture)")
        raise AssertionError(command)

    outcome = runner.run_podcast_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=script_only_command_runner,
    )

    assert outcome["status"] == "blocked"
    receipt = outcome["receipt"]
    assert receipt["blockers"][0]["code"] == "podcast_audio_render_failed"
    assert receipt["checks"].get("podcast_audio_status") != "pass"
    assert not receipt["checks"].get("podcast_audio_sha256")
    assert receipt["checks"]["provider_calls_made"] is False
    output_dir = workspace / runner.PODCAST_OUTPUT_DIR
    assert not (output_dir / "podcast.mp3").exists()


def test_podcast_resume_skips_paid_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second ``resume=True`` invocation over an already-succeeded,
    still-passing stage-09 receipt never invokes the fake ``command_runner``
    (proven by an exploding runner that raises if called) -- RESEARCH
    Pitfall 6 / Pattern 1 (09-VALIDATION.md row 09-01-05)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-podcast-resume"
    _write_podcast_predecessors(runner, workspace, run_id, monkeypatch)
    monkeypatch.setattr(runner, "provider_receipt_summary", _fake_podcast_provider_receipt_summary)

    first_outcome = runner.run_podcast_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_podcast_command_runner(runner),
    )
    assert first_outcome["status"] == "succeeded", first_outcome["receipt"]["blockers"]

    def exploding_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(
            "command_runner must not be invoked on a resume short-circuit over an "
            "already-succeeded, still-passing receipt"
        )

    second_outcome = runner.run_podcast_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=exploding_command_runner,
    )
    assert second_outcome["status"] == "succeeded"
    assert second_outcome["receipt"]["receipt_id"] == first_outcome["receipt"]["receipt_id"]


def test_podcast_owner_execution_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RESEARCH Pattern 2 / Pitfall 3 (T-09-03): the ``strict_execution``
    ``owner_execution_evidence`` dict carries one real, non-echoed token per
    ``['podcast.focused_script', 'podcast.render_audio']``, and a bare-echo
    mutation is rejected by ``verify_stage_contract``'s strict mode
    (09-VALIDATION.md row 09-01-06)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-podcast-owner-evidence"
    _write_podcast_predecessors(runner, workspace, run_id, monkeypatch)
    monkeypatch.setattr(runner, "provider_receipt_summary", _fake_podcast_provider_receipt_summary)

    outcome = runner.run_podcast_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_podcast_command_runner(runner),
    )
    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    receipt = outcome["receipt"]
    required_script_ids = list(
        stage_script_registry.required_script_ids_for_stage(load_stage_script_registry(), "podcast")
    )
    assert required_script_ids == ["podcast.focused_script", "podcast.render_audio"]
    evidence = receipt["owner_execution_evidence"]
    assert set(evidence.keys()) >= set(required_script_ids)
    for owner_id in required_script_ids:
        token = evidence[owner_id]
        assert token, f"{owner_id} must have a non-empty evidence token"
        assert token != owner_id, f"{owner_id} evidence must not be a bare echo of the owner id"

    independent_contract = stage_contracts.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="podcast",
        receipt=receipt,
        mode="live",
        strict_execution=True,
    ).to_dict()
    assert independent_contract["status"] == "pass", independent_contract["blockers"]

    # Bare-echo mutation must be rejected by strict_execution mode.
    mutated_receipt = copy.deepcopy(receipt)
    mutated_receipt["owner_execution_evidence"] = {
        owner_id: owner_id for owner_id in required_script_ids
    }
    mutated_contract = stage_contracts.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="podcast",
        receipt=mutated_receipt,
        mode="live",
        strict_execution=True,
    ).to_dict()
    assert mutated_contract["status"] != "pass"
    assert any(
        blocker["code"] == "stage_contract_owner_execution_evidence_missing"
        for blocker in mutated_contract["blockers"]
    )


def test_podcast_stage_implemented_and_dispatches_from_main(tmp_path: Path) -> None:
    """``STAGES['podcast']`` is flipped to ``implemented=True`` and ``main()``
    routes ``stage.id == 'podcast'`` to ``run_podcast_stage`` (mirrors
    ``test_qa_gap_fill_synthesis_stage_implemented_and_dispatches_from_main``;
    09-VALIDATION.md row 09-01-07)."""

    runner = load_runner_module()
    stage = runner.STAGES["podcast"]
    assert stage.order == 9
    assert stage.receipt_filename == "stage-09-podcast.json"
    assert stage.implemented is True
    assert runner.orchestrator_spec().handlers["podcast"] is runner.dispatch_podcast


def test_podcast_fixture_receipt_satisfies_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixture/mock end-to-end proof the stage-09 podcast receipt satisfies
    its full runtime contract (Plan 09-03; repo rule: validate in mock before
    live; PROD-11 done gates 1-2) BEFORE any live provider spend.

    VALIDATION-ONLY: this test seeds a genuine, passing Stage 8 receipt +
    executive_summary.md/trend_report.md/analysis_manifest.json (via
    ``_write_podcast_predecessors``, which actually runs
    ``run_qa_gap_fill_synthesis_stage`` with a non-network fixture client),
    then injects a fake ``command_runner`` intercepting BOTH subprocess
    commands (``generate_focused_podcast.py`` then ``render_podcast_audio.py``)
    plus a monkeypatched ``provider_receipt_summary`` stand-in -- so ZERO
    provider/network call occurs anywhere in this test -- no Anthropic,
    OpenAI, or TTS request is made. A passing result here can NEVER
    substitute for the live ``stage-09-podcast.json`` receipt, which remains
    the SOLE completion proof for PROD-11 (a future live-execution plan;
    roadmap Rejected evidence)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-podcast-fixture-contract"
    _write_podcast_predecessors(runner, workspace, run_id, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_SENTINEL)
    monkeypatch.setattr(runner, "provider_receipt_summary", _fake_podcast_provider_receipt_summary)

    call_log: list[str] = []
    outcome = runner.run_podcast_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_podcast_command_runner(runner, call_log=call_log),
    )

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    receipt = outcome["receipt"]
    checks = receipt["checks"]

    # -- Zero network/provider calls: exactly the two fake subprocess
    # interceptions occurred, in order (script then audio). --
    assert call_log == ["script", "audio"]

    # -- Contract coupling: read required_checks from the registry file
    # itself so this assertion cannot silently drift from the declared
    # contract (mirrors test_qa_gap_fill_synthesis_fixture_receipt_satisfies_runtime_contract). --
    registry_path = REPO_ROOT / "src" / "conference_pipeline" / "stage_owner_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    binding = registry["stage_bindings"]["podcast"]["runtime_contract"]
    required_checks = binding["required_checks"]
    assert required_checks, "registry must declare at least one required_check"
    for spec in required_checks:
        _assert_required_check(receipt, spec)

    # Prove the coupling is real, not tautological: dropping any one
    # required_check path from a copy of the receipt must make the SAME
    # assertion function fail.
    for spec in required_checks:
        mutated = copy.deepcopy(receipt)
        path_parts = str(spec["path"]).split(".")
        cursor = mutated
        for part in path_parts[:-1]:
            cursor = cursor[part]
        cursor.pop(path_parts[-1], None)
        with pytest.raises(AssertionError):
            _assert_required_check(mutated, spec)

    # -- ALL required_outputs entries, iterated straight from the registry
    # file itself (not a hand-picked subset). --
    required_outputs = binding["required_outputs"]
    assert len(required_outputs) == 5
    for spec in required_outputs:
        matches = sorted(workspace.glob(str(spec["glob"])))
        assert len(matches) >= int(spec.get("min_count", 1)), (
            f"expected >= {spec.get('min_count', 1)} matches for glob {spec['glob']}"
        )

    # -- required_receipts: the Stage 8 dependency's own checks must also
    # be pass, read live from the registry (not hardcoded). --
    for receipt_spec in binding.get("required_receipts", []):
        dependency_path = str(receipt_spec["path"]).replace("<run-id>", run_id)
        dependency_receipt = json.loads((workspace / dependency_path).read_text(encoding="utf-8"))
        for check_spec in receipt_spec.get("checks", []):
            _assert_required_check(dependency_receipt, check_spec)

    # -- verify_stage_contract(strict_execution=True) independently over the
    # persisted receipt returns status pass. --
    independent_contract = stage_contracts.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="podcast",
        receipt=receipt,
        mode="live",
        strict_execution=True,
    ).to_dict()
    assert independent_contract["status"] == "pass", independent_contract["blockers"]
    assert receipt["stage_contract"]["status"] == "pass"

    # -- rendered-audio digest/byte-count fields prove genuine audio, not
    # just a script (PROD-11 done gate 2). --
    assert checks["podcast_audio_sha256"]
    assert checks["podcast_audio_byte_count"] > 0

    # -- SECRET_SENTINEL must never appear in the receipt or either produced
    # podcast artifact (credential-handling regression). --
    output_dir = workspace / runner.PODCAST_OUTPUT_DIR
    for artifact_text in (
        json.dumps(receipt, default=str),
        (output_dir / "script.json").read_text(encoding="utf-8"),
        (output_dir / "audio_manifest.json").read_text(encoding="utf-8"),
        (workspace / runner.PODCAST_AUDIO_PACKAGE).read_text(encoding="utf-8"),
    ):
        assert SECRET_SENTINEL not in artifact_text


# --- Phase 10 (presentation) Wave-0 test scaffolding (10-PLAN 10-01) ---
#
# These four tests prove the presentation stage's gate-only skeleton (dispatch
# wiring, six-input fail-closed gate, legacy-Markdown rejection, provider-cost
# approval gate) at $0 -- ZERO subprocess wiring exists yet for this stage, so
# every block-path test injects an exploding ``command_runner`` that raises if
# ever invoked, proving zero spend (10-VALIDATION.md rows 10-W0-01..10-W0-04).


def _write_presentation_backup_receipt(
    runner: object, workspace: Path, run_id: str, *, status: str = "succeeded"
) -> Path:
    """Write a fixture Phase 9 BK-09 backup receipt.

    ``run_presentation_stage``'s six-input gate treats this receipt as a pure
    SEQUENCING gate (10-RESEARCH.md architecture diagram: "content NOT
    consumed") -- only its presence and top-level ``status`` are read, so a
    minimal fixture (not a full backup-tool run) is sufficient here.
    """

    target = runner.run_dir(workspace, run_id)
    backup_dir = target / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stage9_id = runner.STAGES["podcast"].id
    receipt_path = backup_dir / f"{stage9_id}-backup-write-receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.cvpr2026_backup_write_receipt",
                "status": status,
                "exit_code": 0 if status == "succeeded" else 1,
                "error": None,
                "secrets_copied": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt_path


def _seed_executive_digest_chain_artifacts(runner: object, workspace: Path) -> None:
    """Write what the digest chain produces for the 2 executive decks.

    Executive decks are authored by ``build_executive_summary.py`` ->
    ``build_executive_videos.py``, never by the per-deck chain, so Stage 10
    verifies their artifacts instead of rebuilding them. This fixture models
    that: before, the executive path was stubbed by simply adding the 2 deck
    ids to a set of "already succeeded" ids, so the runner's executive
    handling was never exercised at all -- which is why it shipped unable to
    produce an executive deck for any conference.
    """

    slides = [{"slide_id": f"sl_{index:02d}"} for index in range(1, 6)]
    for deck_id in ("executive_summary_short", "executive_summary_long"):
        deck_dir = workspace / "decks" / deck_id
        deck_dir.mkdir(parents=True, exist_ok=True)
        payloads = {
            "presentation_deck.json": {"presentation_id": deck_id, "slides": slides},
            "render_spec.json": {"presentation_id": deck_id, "slides": slides},
            "slide_frame_manifest.json": {"presentation_id": deck_id, "frames": slides},
            "presentation_blueprint.json": {"blueprint_id": f"{deck_id}-blueprint"},
            "blueprint_review.json": {"approved_for_draft_generation": True},
            "source_binding_report.json": {"presentation_id": deck_id, "slides": slides},
            "visual_plan.json": {"presentation_id": deck_id},
            "presentation_slide_mapping.json": {"slides": slides},
            "figure_svg_validation_report.json": {"status": "pass"},
            "delivery_single_speaker.json": {"presentation_id": deck_id},
            "delivery_notebooklm.json": {"presentation_id": deck_id},
            "video_sync_validation_single_speaker.json": {"status": "pass"},
            "video_sync_validation_notebooklm.json": {"status": "pass"},
        }
        for name, payload in payloads.items():
            (deck_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        (deck_dir / "presentation.html").write_text("<html></html>", encoding="utf-8")
        (deck_dir / "presentation.pdf").write_bytes(b"%PDF-1.4 fixture")
        (deck_dir / "presentation_video.mp4").write_bytes(b"fixture mp4")
        (deck_dir / "presentation_video_notebooklm.mp4").write_bytes(b"fixture mp4")


def _write_presentation_predecessors(
    runner: object,
    workspace: Path,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    topic_count: int = 3,
) -> dict[str, object]:
    """Produce genuine, passing Stage 8 + Stage 9 receipts (by actually
    running ``run_qa_gap_fill_synthesis_stage``/``run_podcast_stage`` with
    non-network fixtures, via ``_write_podcast_predecessors``) plus a fixture
    Phase 9 BK-09 backup receipt, so Stage 10's own fail-closed six-input
    gate (PROD-12, done gate 1) reads real, digest-consistent upstream
    outputs rather than a hand-assembled stand-in receipt.
    """

    handles = _write_podcast_predecessors(runner, workspace, run_id, monkeypatch, topic_count=topic_count)
    monkeypatch.setattr(runner, "provider_receipt_summary", _fake_podcast_provider_receipt_summary)
    podcast_outcome = runner.run_podcast_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_podcast_command_runner(runner),
    )
    assert podcast_outcome["status"] == "succeeded", podcast_outcome["receipt"]["blockers"]
    handles["podcast_outcome"] = podcast_outcome
    handles["backup_receipt_path"] = _write_presentation_backup_receipt(runner, workspace, run_id)
    _seed_executive_digest_chain_artifacts(runner, workspace)
    stamp_upstream_stage_receipts(workspace / "storage" / "workflow-runs" / run_id)
    return handles


def test_presentation_stage_implemented_and_dispatches_from_main(tmp_path: Path) -> None:
    """``STAGES['presentation']`` is flipped to ``implemented=True`` and
    ``main()`` routes ``stage.id == 'presentation'`` to
    ``run_presentation_stage`` (mirrors
    ``test_podcast_stage_implemented_and_dispatches_from_main``;
    10-VALIDATION.md row 10-W0-01)."""

    runner = load_runner_module()
    stage = runner.STAGES["presentation"]
    assert stage.order == 10
    assert stage.receipt_filename == "stage-10-presentation.json"
    assert stage.implemented is True
    assert runner.orchestrator_spec().handlers["presentation"] is runner.dispatch_presentation


def test_presentation_blocks_on_missing_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PROD-12 done gate 1: the stage fails closed (never fabricates a pass)
    with zero provider spend when the Stage 8 receipt, the analysis
    manifest, the executive summary, the trend report, the Stage 9 receipt,
    or the Phase 9 BK-09 backup receipt is absent -- and the fake
    ``command_runner`` (exploding if invoked) proves no subprocess is ever
    constructed (10-VALIDATION.md row 10-W0-02)."""

    runner = load_runner_module()

    def exploding_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(
            f"command_runner must never be invoked when required presentation "
            f"inputs are missing/not-succeeded: {command}"
        )

    for label in (
        "stage8_receipt",
        "analysis_manifest",
        "executive_summary",
        "trend_report",
        "stage9_receipt",
        "stage9_backup_receipt",
    ):
        workspace = write_runner_workspace(tmp_path / f"conferences-presentation-missing-{label}")
        run_id = f"test-presentation-missing-{label}"
        _write_presentation_predecessors(runner, workspace, run_id, monkeypatch)
        input_paths = production_presentation_stage.required_input_paths(runner._production_context(workspace, run_id))
        input_paths[label].unlink()

        outcome = runner.run_presentation_stage(
            workspace=workspace,
            run_id=run_id,
            resume=True,
            provider_cost_approved=True,
            env_file=None,
            command_runner=exploding_command_runner,
        )

        assert outcome["status"] == "blocked", label
        receipt = outcome["receipt"]
        assert receipt["status"] == "blocked", label
        assert receipt["checks"]["provider_calls_made"] is False, label
        assert receipt["blockers"][0]["code"] == "presentation_required_inputs_blocked", label
        input_blockers = receipt["checks"]["input_blockers"]
        assert input_blockers, label


def test_presentation_blocks_on_missing_provider_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROD-12: the stage blocks before any subprocess is invoked when
    ``provider_cost_approved`` is False, and never auto-raises the cap
    (10-VALIDATION.md row 10-W0-03)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-missing-approval"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch)

    def exploding_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(
            f"command_runner must never be invoked without provider-cost approval: {command}"
        )

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=False,
        env_file=None,
        command_runner=exploding_command_runner,
    )

    assert outcome["status"] == "blocked"
    receipt = outcome["receipt"]
    assert receipt["checks"]["provider_calls_made"] is False
    assert receipt["blockers"][0]["code"] == "presentation_provider_cost_not_approved"
    assert receipt["checks"]["provider_cost_approval"]["approved"] is False


def test_presentation_rejects_legacy_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ROADMAP done gate 1: legacy numbered-Markdown evidence offered as
    production proof is rejected with the literal blocker code
    ``PRESENTATION_LEGACY_MARKDOWN_STAGE_USED`` -- zero spend, proven via an
    exploding ``command_runner`` (10-VALIDATION.md row 10-W0-04)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-legacy-markdown"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch)

    legacy_dir = workspace / "decks" / "01_test_topic"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "1_draft_slides.md").write_text("# Legacy slides\n", encoding="utf-8")

    def exploding_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(
            f"command_runner must never be invoked when legacy Markdown production "
            f"evidence is present: {command}"
        )

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=exploding_command_runner,
    )

    assert outcome["status"] == "blocked"
    receipt = outcome["receipt"]
    assert receipt["blockers"][0]["code"] == "PRESENTATION_LEGACY_MARKDOWN_STAGE_USED"
    assert receipt["checks"]["provider_calls_made"] is False
    assert receipt["checks"]["legacy_markdown_paths"]


# --- Phase 10 (presentation) sub-stage A part 1 test scaffolding (10-PLAN 10-02) ---
#
# Task 1 proves the per-deck loop + in-process paper-matching call site
# Plan 10-02 wires: deck-scope building (12 topics + 2 executive decks,
# LOCKED full coverage) and per-deck paper matching with an idempotent
# per-step skip. Task 2/3 extend this section with the 4 early
# Critical-Finding-2 scripts and the blueprint/deck-contract/source-binding
# chain. Every test here is a $0 fixture/mock proof: paper matching is
# monkeypatched (in-process, never a subprocess) -- no real subprocess,
# network, or LLM call ever occurs.


def _fake_presentation_paper_matching_stage(
    *,
    workspace_root: Path,
    run_id: str,
    topic_id: str | None = None,
    config: object | None = None,
) -> dict[str, object]:
    """Non-network stand-in for ``run_paper_matching_stage`` -- writes only
    ``matched_papers.json`` (the real evidence this step verifies);
    ``ranked_papers.json`` is compatibility-only (done gate 3) and is not
    needed for this plan's own wiring proof."""

    del config
    output_dir = Path(workspace_root) / "storage" / "paper-matching" / run_id / str(topic_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matched_papers.json").write_text(
        json.dumps({"schema_version": 1, "topic_id": topic_id, "papers": [{"title": "Fixture Paper"}]})
        + "\n",
        encoding="utf-8",
    )
    return {"status": "succeeded", "run_id": run_id, "topics": [topic_id], "artifacts": []}


def test_presentation_deck_scope_covers_all_topics_and_executive_decks(tmp_path: Path) -> None:
    """The deck-scope builder derives exactly the 12 topic decks from
    ``analysis_manifest.json`` plus the 2 executive-overview decks -- LOCKED
    full-coverage decision, no canary subset (10-VALIDATION.md)."""

    runner = load_runner_module()
    manifest = {
        "topics": [
            {"id": f"{index:02d}_fixture_topic", "title": f"Fixture Topic {index}"}
            for index in range(1, 13)
        ]
    }
    scope = production_decks.presentation_deck_scope(manifest, runner.PRESENTATION_EXECUTIVE_OVERVIEW_DECKS)
    assert len(scope) == 14
    ids = [deck["id"] for deck in scope]
    assert ids[:12] == [f"{index:02d}_fixture_topic" for index in range(1, 13)]
    assert ids[12:] == ["executive_summary_short", "executive_summary_long"]
    kinds = {deck["id"]: deck["kind"] for deck in scope}
    assert kinds["01_fixture_topic"] == "topic"
    assert kinds["executive_summary_short"] == "executive"
    assert kinds["executive_summary_long"] == "executive"


def test_presentation_compiler_figure_route_replaces_codegen_and_the_redraw_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--figure-route compiler`` runs spec authoring plus the deterministic
    compiler INSTEAD of the two drawsvg passes and the post-hoc semantic
    review, and it never enters the bounded pixel-judge redraw loop: a compiled
    figure is a pure function of its spec, so a redraw could only pay to
    reproduce the same pixels."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-figure-route"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    call_log: list[str] = []
    runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=False,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        deck_ids=["01_fixture_topic"],
        figure_route="compiler",
        accept_visual_qc_findings=True,
        command_runner=_build_fake_presentation_command_runner(call_log=call_log),
    )

    assert "generate_figure_specs.py" in call_log
    assert "compile_figures.py" in call_log
    assert (call_log.index("generate_figure_specs.py")
            < call_log.index("review_claim_fidelity.py")
            < call_log.index("compile_figures.py"))
    assert call_log.count("review_claim_fidelity.py") == 1  # deck_figures stops before narration
    report = json.loads((workspace / "decks/01_fixture_topic/claim_fidelity_review_figures.json").read_text())
    assert report["status"] == "passed" and report["phase"] == "figures"
    assert report["task_id"] == "presentation.claim_fidelity_review"
    assert report["findings"] == []
    # The drawsvg route's paid codegen, its post-hoc semantic review and its
    # redraw loop are all absent.
    assert "generate_programmatic_figures.py" not in call_log
    assert "review_programmatic_figures.py" not in call_log
    assert "repair_figure_loop.py" not in call_log


def test_the_compile_step_records_the_spec_files_as_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """patch_figure_spec.py edits ``figure_specs/<slide>-figure.json`` in place. The
    compile step's other inputs are its own aggregate output and the authoring
    receipt, so a patch left the step "current": the compiled figure, and the videos
    built from it, kept the old text (ICML 2026 deck 01, slide 7). The spec files are
    now declared inputs, so the step ledger sees an edit."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-spec-inputs"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    deck_dir = workspace / "decks" / "01_fixture_topic"
    spec_path = deck_dir / "figure_specs" / "sl_01_fixture-figure.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text('{"title": "Untested prediction"}\n', encoding="utf-8")

    runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=False,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        deck_ids=["01_fixture_topic"],
        figure_route="compiler",
        accept_visual_qc_findings=True,
        command_runner=_build_fake_presentation_command_runner(),
    )

    ledger = json.loads((deck_dir / "step_lineage.json").read_text(encoding="utf-8"))
    inputs = ledger["steps"]["compile_figures"]["inputs"]
    assert "ws:decks/01_fixture_topic/figure_specs" in inputs
    assert "Untested" not in json.dumps(inputs)  # digests, never content
    fidelity_inputs = ledger["steps"]["claim_fidelity_review_figures"]["inputs"]
    assert "ws:decks/01_fixture_topic/figure_specs" in fidelity_inputs


def test_presentation_default_figure_route_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No route argument means the drawsvg route, exactly as before: adding the
    compiler must not silently change what an existing command does."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-default-route"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    call_log: list[str] = []
    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=False,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        deck_ids=["01_fixture_topic"],
        accept_visual_qc_findings=True,
        command_runner=_build_fake_presentation_command_runner(call_log=call_log),
    )

    assert "generate_programmatic_figures.py" in call_log
    assert "generate_figure_specs.py" not in call_log
    assert outcome["receipt"]["checks"]["figure_route"] == "drawsvg"


def test_presentation_deck_ids_filter_runs_only_the_canary_subset_and_never_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator-chosen ``deck_ids`` subset runs the per-deck chain for
    exactly those decks and returns an honest ``blocked`` receipt carrying
    ``presentation_deck_scope_filtered`` -- a filtered scope can never be
    reported as ``succeeded``. An unknown deck id blocks with zero spend."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-deck-filter"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)

    call_log: list[str | None] = []

    def fake_paper_matching(
        *, workspace_root: Path, run_id: str, topic_id: str | None = None, config: object | None = None
    ) -> dict[str, object]:
        call_log.append(topic_id)
        return _fake_presentation_paper_matching_stage(
            workspace_root=workspace_root, run_id=run_id, topic_id=topic_id, config=config
        )

    monkeypatch.setattr(runner.presentation_paper_matching, "run_paper_matching_stage", fake_paper_matching)

    unknown = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        deck_ids=["01_fixture_topic", "99_not_a_deck"],
        command_runner=_build_fake_presentation_command_runner(),
    )
    assert unknown["receipt"]["status"] == "blocked"
    assert unknown["receipt"]["blockers"][0]["code"] == "presentation_deck_scope_filter_unknown_deck"
    assert unknown["receipt"]["checks"]["unknown_deck_ids"] == ["99_not_a_deck"]
    assert call_log == []

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        deck_ids=["01_fixture_topic", "02_fixture_topic"],
        command_runner=_build_fake_presentation_command_runner(),
    )
    receipt = outcome["receipt"]
    assert receipt["status"] == "blocked", receipt
    blocker = receipt["blockers"][0]
    assert blocker["code"] == "presentation_deck_scope_filtered"
    assert blocker["deck_ids"] == ["01_fixture_topic", "02_fixture_topic"]
    assert blocker["blocked_deck_ids"] == []
    assert blocker["full_scope_deck_count"] == 14
    checks = receipt["checks"]
    assert checks["deck_scope_filter"] == ["01_fixture_topic", "02_fixture_topic"]
    assert checks["deck_count"] == 2
    assert [deck["id"] for deck in checks["deck_scope"]] == ["01_fixture_topic", "02_fixture_topic"]
    assert len(checks["deck_records"]) == 2
    assert checks["deck_success_count"] == 2
    assert call_log == ["01_fixture_topic", "02_fixture_topic"]
    assert checks["provider_calls_made"] is True
    assert "presentation.build_notebooklm_video" in checks["owner_execution_evidence"]
    assert receipt.get("stage_contract") is None


def test_presentation_blueprint_review_honors_llm_judge_artifact_and_reruns_failed_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blueprint-review step records the LLM-judge gate only when
    ``blueprint_llm_judge.md`` exists on disk, passes the explicit approver
    when one is given (else the automated id), and never resume-skips a
    review whose existing status is not ``pass``."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-blueprint-review-gate"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    judged_deck = workspace / runner.PRESENTATION_DECKS_DIR / "01_fixture_topic"
    judged_deck.mkdir(parents=True, exist_ok=True)
    judge_artifact = judged_deck / "blueprint_llm_judge.md"
    judge_artifact.write_text("# judge\n\nPASS\n", encoding="utf-8")

    stale_deck = workspace / runner.PRESENTATION_DECKS_DIR / "03_fixture_topic"
    stale_deck.mkdir(parents=True, exist_ok=True)
    stale_review = stale_deck / "blueprint_review.json"
    stale_review.write_text(json.dumps({"schema_version": 1, "status": "fail"}), encoding="utf-8")

    review_commands: dict[str, list[str]] = {}
    inner = _build_fake_presentation_command_runner()

    def capturing_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if any(str(part).endswith("review_presentation_blueprint.py") for part in command):
            deck_id = command[command.index("--presentation-id") + 1]
            review_commands[deck_id] = list(command)
        return inner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        deck_ids=["01_fixture_topic", "02_fixture_topic", "03_fixture_topic"],
        blueprint_approved_by="Operator Directive 2026-09-08",
        command_runner=capturing_runner,
    )
    assert outcome["receipt"]["blockers"][0]["code"] == "presentation_deck_scope_filtered"
    assert outcome["receipt"]["blockers"][0]["blocked_deck_ids"] == []

    judged = review_commands["01_fixture_topic"]
    assert "--llm-judge-approved" in judged
    assert judged[judged.index("--llm-judge-artifact") + 1] == str(judge_artifact)
    assert judged[judged.index("--approved-by") + 1] == "Operator Directive 2026-09-08"

    unjudged = review_commands["02_fixture_topic"]
    assert "--llm-judge-approved" not in unjudged
    assert "--llm-judge-artifact" not in unjudged

    # The pre-existing failed review was not resume-skipped: the review
    # script ran again for deck 03 and its fixture output replaced the stale
    # file.
    assert "03_fixture_topic" in review_commands
    assert json.loads(stale_review.read_text(encoding="utf-8"))["status"] == "pass"

    automated = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=False,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        deck_ids=["02_fixture_topic"],
        command_runner=capturing_runner,
    )
    assert automated["receipt"]["blockers"][0]["code"] == "presentation_deck_scope_filtered"
    rerun = review_commands["02_fixture_topic"]
    assert rerun[rerun.index("--approved-by") + 1] == runner.PRESENTATION_AUTOMATED_APPROVER_ID


def test_presentation_per_deck_paper_matching_in_process_with_idempotent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paper matching runs per deck across all 14 in-scope decks as an
    IN-PROCESS call (never a subprocess -- proven by asserting it never
    appears in the fake command runner's call log), and a resumed pass
    never re-invokes paper matching for a deck whose ``matched_papers.json``
    already exists (RESEARCH Critical Finding 3 / Pattern 1;
    10-VALIDATION.md)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-paper-matching"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)

    call_log: list[str | None] = []

    def fake_paper_matching(
        *, workspace_root: Path, run_id: str, topic_id: str | None = None, config: object | None = None
    ) -> dict[str, object]:
        call_log.append(topic_id)
        return _fake_presentation_paper_matching_stage(
            workspace_root=workspace_root, run_id=run_id, topic_id=topic_id, config=config
        )

    monkeypatch.setattr(runner.presentation_paper_matching, "run_paper_matching_stage", fake_paper_matching)
    command_call_log: list[str] = []

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_presentation_command_runner(call_log=command_call_log),
    )

    receipt = outcome["receipt"]
    # -- with Plan 10-04's delivery/video wiring complete, a fully-populated
    # fixture run now reaches a genuine ``succeeded`` strict-contract receipt
    # instead of the old "not_implemented" placeholder; real spend occurred
    # across the whole per-deck chain. --
    assert receipt["status"] == "succeeded", receipt
    checks = receipt["checks"]
    assert checks["deck_count"] == 14
    assert len(checks["deck_scope"]) == 14
    assert len(checks["deck_records"]) == 14
    # Paper matching is a per-deck-chain step, so only the topic decks make it.
    assert call_log == [
        deck["id"] for deck in checks["deck_scope"] if deck["kind"] != "executive"
    ]
    assert not any("paper_matching" in entry or "match_papers" in entry for entry in command_call_log)
    assert checks["provider_calls_made"] is True

    # -- idempotent per-step skip: a resumed pass must not re-invoke paper
    # matching once matched_papers.json exists for every deck. --
    def exploding_paper_matching(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("paper matching must not be re-invoked once matched_papers.json exists")

    monkeypatch.setattr(runner.presentation_paper_matching, "run_paper_matching_stage", exploding_paper_matching)

    def exploding_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"command_runner must not be re-invoked on a per-step resume skip: {command}")

    second_outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=exploding_command_runner,
    )
    # -- the resume short-circuit returns the existing succeeded receipt. --
    assert second_outcome["receipt"]["status"] == "succeeded", second_outcome["receipt"]
    assert len(second_outcome["receipt"]["checks"]["deck_records"]) == 14


# --- Task 2: the 4 early Critical-Finding-2 deterministic scripts ---
#
# Task 2 extends the per-deck chain with source_corpus_inventory ->
# speaker_author_work_ledger -> repair_presentation_sources ->
# research_cluster_plan, wired inline per deck after paper matching. Task 3
# extends the fake command runner below to also cover blueprint/review/
# deck-contract/source-binding.

# --- Task 3: blueprint generation + review, deck-contract generation, and
# source binding (closes sub-stage A part 1) ---

_PRESENTATION_SOURCE_BINDING_SCRIPT_IDS = (
    "presentation.source_corpus_inventory",
    "presentation.speaker_author_work_ledger",
    "presentation.repair_presentation_sources",
    "presentation.research_cluster_plan",
    "presentation.generate_editorial_blueprint",
    "presentation.review_presentation_blueprint",
    "presentation.generate_deck_contract",
    "presentation.source_pack",
    "presentation.bind_presentation_sources",
)


def _presentation_full_chain_model_governance_task_ids(runner: object, *, repair_ran: bool = False) -> list[str]:
    """The exact ``model_governance_task_ids`` list a fully-populated
    fixture run records once Plan 10-04's delivery/video wiring is
    reachable -- ``repair_ran`` inserts ``PRESENTATION_REPAIR_TASK_ID`` in
    its recorded position only when the bounded repair loop genuinely fired
    (not the ``repair_not_required:`` fallback token)."""

    ids = [
        runner.PRESENTATION_BLUEPRINT_GENERATION_TASK_ID,
        runner.PRESENTATION_SVG_FIGURE_GENERATION_TASK_ID,
        runner.PRESENTATION_VISUAL_QC_TASK_ID,
    ]
    if repair_ran:
        ids.append(runner.PRESENTATION_REPAIR_TASK_ID)
    ids.extend(
        [
            runner.PRESENTATION_PRESENTER_SCRIPT_GENERATION_TASK_ID,
            runner.PRESENTATION_PRESENTER_TTS_TASK_ID,
            runner.PRESENTATION_PRESENTER_DIALOGUE_GENERATION_TASK_ID,
            runner.PRESENTATION_REVIEW_TASK_ID,
        ]
    )
    return ids


def _presentation_fixture_arg(command: list[str], flag: str) -> str | None:
    """Read the value following ``flag`` in a subprocess ``command`` list,
    or ``None`` if the flag is absent (never raises -- absence is a normal,
    expected state for the handful of conditionally-passed flags)."""

    if flag in command:
        index = command.index(flag)
        if index + 1 < len(command):
            return command[index + 1]
    return None


def _presentation_fixture_write_spec(
    *,
    kind: str,
    content: object,
    arg: str | None = None,
    resolver: Callable[[list[str]], str] | None = None,
) -> dict[str, object]:
    """One declarative ``{arg/resolver -> kind, content}`` entry describing
    a single output file an owner-script fixture must write. ``arg`` names
    the CLI flag whose value is the output path (skipped, never asserted,
    when the flag is absent from a given call -- mirrors every owner
    script's own genuinely-optional flags); ``resolver`` is used instead of
    ``arg`` only for the one owner script (``build_deck_source_pack.py``)
    whose fixture output path is computed from two OTHER flags rather than
    read directly off one output flag."""

    return {"kind": kind, "content": content, "arg": arg, "resolver": resolver}


def _resolve_source_pack_receipt_path(command: list[str]) -> str:
    """``build_deck_source_pack.py``'s fixture receipt path is derived from
    ``--workspace``/``--presentation-id`` (it has no dedicated output-path
    flag of its own) -- the one genuine exception to the arg-keyed table
    below, preserved as a named resolver rather than an inline branch."""

    workspace_value = _presentation_fixture_arg(command, "--workspace")
    presentation_id = _presentation_fixture_arg(command, "--presentation-id")
    assert workspace_value and presentation_id, command
    return str(Path(workspace_value) / "decks" / presentation_id / "source_pack_receipt.json")


# Single parametrized table keyed by owner-script basename -> the fixture
# output file(s) that basename's real subprocess call site checks for, per
# 10-PATTERNS.md's explicit recommendation (RESEARCH) to avoid ~14+
# hand-written near-duplicate ``if "..." in command:`` branches. Each
# ``writes`` entry is data, not code: ``fake_command_runner`` below is the
# single generic dispatcher that reads this table and performs every write.
_PRESENTATION_FIXTURE_TABLE: dict[str, dict[str, object]] = {
    "build_presentation_source_corpus_inventory.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output-json", kind="json", content={"schema_version": 1, "sources": []}),
            _presentation_fixture_write_spec(arg="--receipt-path", kind="json", content={"status": "pass"}),
        ),
    },
    "build_speaker_author_work_ledger.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output-json", kind="json", content={"schema_version": 1, "entries": []}),
            _presentation_fixture_write_spec(arg="--receipt-path", kind="json", content={"status": "pass"}),
        ),
    },
    "repair_presentation_sources.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output-json", kind="json", content={"schema_version": 1, "findings": []}),
            _presentation_fixture_write_spec(arg="--receipt-path", kind="json", content={"status": "pass"}),
        ),
    },
    "build_research_cluster_plan.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output-json", kind="json", content={"schema_version": 1, "clusters": []}),
            _presentation_fixture_write_spec(arg="--markdown-output", kind="text", content="# Fixture cluster plan\n"),
            _presentation_fixture_write_spec(arg="--receipt-path", kind="json", content={"status": "pass"}),
        ),
    },
    "generate_editorial_blueprint.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output-json", kind="json", content={"schema_version": 1, "blueprint": "fixture"}),
            _presentation_fixture_write_spec(arg="--output-md", kind="text", content="# Fixture blueprint\n"),
            _presentation_fixture_write_spec(arg="--receipt-path", kind="json", content={"status": "pass"}),
        ),
    },
    "review_presentation_blueprint.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output-json", kind="json", content={"schema_version": 1, "status": "pass"}),
            _presentation_fixture_write_spec(arg="--output-md", kind="text", content="# Fixture blueprint review\n"),
        ),
    },
    "generate_deck_contract.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output", kind="json", content={"schema_version": 1, "slides": []}),
            _presentation_fixture_write_spec(arg="--receipt-output", kind="json", content={"status": "pass"}),
        ),
    },
    "build_deck_source_pack.py": {
        "writes": (
            _presentation_fixture_write_spec(kind="json", content={"status": "pass"}, resolver=_resolve_source_pack_receipt_path),
        ),
    },
    "bind_presentation_sources.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output", kind="json", content={"schema_version": 1, "bindings": []}),
        ),
    },
    "generate_programmatic_figures.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--visual-plan", kind="json", content={"schema_version": 2, "presentation_id": "fixture", "slides": []}),
            _presentation_fixture_write_spec(arg="--asset-inventory", kind="json", content={"schema_version": 2, "generated_assets": []}),
            # -- these two flags are only ever present on the PAID second
            # (Phase 16 LLM codegen) pass; the arg-absence no-op below
            # (never write when the flag itself is missing) already
            # reproduces the original "if --provider in command" gate
            # without a separate conditional branch. --
            _presentation_fixture_write_spec(arg="--model-receipt-output", kind="json", content={"schema_version": 1, "status": "pass"}),
            _presentation_fixture_write_spec(arg="--code-review-output", kind="json", content={"schema_version": 1, "status": "pass"}),
        ),
    },
    "review_programmatic_figures.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-output", kind="json", content={"schema_version": 1, "status": "pass"}),
            _presentation_fixture_write_spec(arg="--markdown-output", kind="text", content="# Fixture figure semantic review\n"),
        ),
    },
    "validate_svgs.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-output", kind="json", content={"schema_version": 1, "summary": {"error": 0}}),
        ),
    },
    "generate_render_spec.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output", kind="json", content={"schema_version": 1, "slides": []}),
        ),
    },
    # Idempotent in-place rewrite of render_spec.json / presentation_deck.json
    # (bookend slides); the fixture leaves both files as the prior steps
    # wrote them.
    # Compiler route (--figure-route compiler): the spec author writes its
    # receipt and the deck's numeric-claims ledger; the compiler writes the two
    # renderer contracts and its own receipt.
    "generate_figure_specs.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--receipt", kind="json", content={
                "receipt_type": "conference_pipeline.figures.spec_authoring",
                "status": "pass", "spec_count": 2, "slide_count": 2, "blockers": [],
            }),
            _presentation_fixture_write_spec(
                resolver=lambda command: _presentation_fixture_deck_path(command, "numeric_claims_ledger.json"),
                kind="json",
                content={"ledger_version": "numeric-claims/1", "claim_count": 0, "claims": []},
            ),
        )
    },
    "review_claim_fidelity.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-output", kind="json", content={
                "schema_version": 1, "task_id": "presentation.claim_fidelity_review",
                "status": "passed", "findings": [], "finding_counts": {"blocker": 0, "advisory": 0},
                "ledger_mismatch_findings": [], "provider": "fixture", "model": "fixture",
                "budget_cap_usd": 5.0, "estimated_max_cost_usd": 0.0, "input_hashes": {},
            }),
        ),
    },
    "compile_figures.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--receipt", kind="json", content={
                "receipt_type": "conference_pipeline.figures.compile",
                "status": "pass", "compiled_count": 2, "blockers": [],
            }),
            _presentation_fixture_write_spec(
                resolver=lambda command: _presentation_fixture_deck_path(command, "visual_plan.json"),
                kind="json",
                content={"schema_version": 1, "presentation_id": "fixture", "slides": []},
            ),
            _presentation_fixture_write_spec(
                resolver=lambda command: _presentation_fixture_deck_path(command, "asset_inventory.json"),
                kind="json",
                content={"schema_version": 1, "generated_assets": [], "generation_route": "figure_compiler"},
            ),
            _presentation_fixture_write_spec(
                resolver=lambda command: _presentation_fixture_deck_path(command, "figure_specs.json"),
                kind="json", content={"schema_version": 1, "figures": []},
            ),
            _presentation_fixture_write_spec(
                resolver=lambda command: _presentation_fixture_deck_path(
                    command, "figure_generation_model_receipt.json"),
                kind="json", content={"schema_version": 1, "generation_route": "figure_compiler"},
            ),
            _presentation_fixture_write_spec(
                resolver=lambda command: _presentation_fixture_deck_path(
                    command, "figure_code_review_report.json"),
                kind="json", content={"schema_version": 1, "status": "pass", "findings": []},
            ),
            _presentation_fixture_write_spec(
                resolver=lambda command: _presentation_fixture_deck_path(
                    command, "figure_semantic_review.json"),
                kind="json", content={"schema_version": 1, "status": "pass", "findings": [],
                                      "semantic_review": []},
            ),
        )
    },
    "add_bookend_slides.py": {"writes": ()},
    "md_to_revealjs.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output-html", kind="text", content="<html><body>Fixture deck</body></html>\n"),
            _presentation_fixture_write_spec(arg="--slide-mapping", kind="json", content={"schema_version": 1, "slides": []}),
        ),
    },
    "export_revealjs_pdf.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output-pdf", kind="bytes", content=b"%PDF-1.4\n%%EOF\n"),
            _presentation_fixture_write_spec(arg="--receipt-output", kind="json", content={"status": "pass"}),
        ),
    },
    "md_to_pptx.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output-pdf", kind="bytes", content=b"%PDF-1.4\n%%EOF\n"),
        ),
    },
    "inspect_rendered_slides.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-output", kind="json", content={"schema_version": 1, "errors": []}),
            _presentation_fixture_write_spec(arg="--screenshots-dir", kind="dir_file", content=("slide-001.png", b"\x89PNG\r\n")),
        ),
    },
    "inspect_rendered_slides_llm.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-output", kind="json", content={"schema_version": 1, "status": "passed", "slides": []}),
        ),
    },
    "build_figure_review_page.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--output", kind="bytes", content=b"<!doctype html><title>review</title>"),
        ),
    },
    "run_presentation_repair_loop.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--receipt-output", kind="json", content={"schema_version": 1, "status": "passed"}),
        ),
    },
    "validate_artifact_truth.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-out", kind="json", content={"schema_version": 1, "status": "pass"}),
        ),
    },
    "audit_presentation_links.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--receipt-output", kind="json", content={"schema_version": 1, "status": "passed"}),
        ),
    },
    "generate_presenter_script.py": {
        "writes": (
            _presentation_fixture_write_spec(
                arg="--delivery-single-speaker",
                kind="json",
                content={"schema_version": 2, "style": "single_speaker", "approve_for_tts": False, "segments": []},
            ),
            _presentation_fixture_write_spec(
                arg="--delivery-notebooklm",
                kind="json",
                content={"schema_version": 2, "style": "notebooklm", "approve_for_tts": False, "segments": []},
            ),
            _presentation_fixture_write_spec(arg="--debug-script-md", kind="text", content="# Fixture presenter script\n"),
            _presentation_fixture_write_spec(
                arg="--debug-notebooklm-script-md",
                kind="text",
                content="### [SLIDE 1] Fixture\n**DIALOGUE:**\n- **Speaker 1:** Fixture line.\n",
            ),
        ),
    },
    # Engaging narration writes its two DRAFT scripts next to --deck-dir (no
    # per-output flags); the delivery rebuild and hash reconcile rewrite the
    # delivery contracts / render_spec in place (the fixture leaves them).
    "generate_engaging_narration.py": {
        "writes": (
            _presentation_fixture_write_spec(
                resolver=lambda cmd: str(
                    Path(_presentation_fixture_arg(cmd, "--deck-dir") or "") / "6_presenter_script.ENGAGING.md"
                ),
                kind="text",
                content="# Fixture engaging script\n",
            ),
            _presentation_fixture_write_spec(
                resolver=lambda cmd: str(
                    Path(_presentation_fixture_arg(cmd, "--deck-dir") or "")
                    / "6_presenter_script_notebooklm_2host.ENGAGING.md"
                ),
                kind="text",
                content="### [SLIDE 1] Fixture\n**DIALOGUE:**\n- **Speaker 1:** Fixture engaging line.\n",
            ),
            # The narration source of truth the delivery rebuild reads; a
            # required, operator-editable output of this step (v5.1 ledger).
            _presentation_fixture_write_spec(
                resolver=lambda cmd: str(
                    Path(_presentation_fixture_arg(cmd, "--deck-dir") or "") / "engaging_narration_raw.json"
                ),
                kind="json",
                content={"schema_version": 1, "slides": [{"slide": 1, "single_speaker": "Fixture engaging line."}]},
            ),
        ),
    },
    "rebuild_delivery_engaging.py": {"writes": ()},
    "reconcile_contract_hashes.py": {"writes": ()},
    "lint_presenter_narration.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-out", kind="json", content={"schema_version": 1, "status": "passed", "finding_counts": {}}),
        ),
    },
    "lint_narration_script.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-out", kind="json", content={"schema_version": 1, "status": "passed", "finding_counts": {}}),
        ),
    },
    "review_presenter_narration.py": {
        "writes": (
            _presentation_fixture_write_spec(
                arg="--json-out",
                kind="json",
                content={"schema_version": 1, "status": "passed", "approve_for_tts": True, "finding_counts": {}},
            ),
            _presentation_fixture_write_spec(arg="--review-out", kind="text", content="# Fixture narration review\n"),
        ),
    },
    "review_narration_script.py": {
        "writes": (
            _presentation_fixture_write_spec(
                arg="--json-out",
                kind="json",
                content={"schema_version": 1, "status": "passed", "approve_for_tts": True, "finding_counts": {}},
            ),
            _presentation_fixture_write_spec(arg="--review-out", kind="text", content="# Fixture narration review\n"),
        ),
    },
    "fix_presenter_narration.py": {
        "writes": (_presentation_fixture_write_spec(arg="--report-out", kind="json", content={"schema_version": 1, "stats": {}}),),
    },
    "fix_narration_script.py": {
        "writes": (_presentation_fixture_write_spec(arg="--report-out", kind="json", content={"schema_version": 1, "stats": {}}),),
    },
    "fix_narration_series.py": {
        "writes": (_presentation_fixture_write_spec(arg="--report-out", kind="json", content={"schema_version": 1, "stats": {}}),),
    },
    "review_narration_series.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-out", kind="json", content={"schema_version": 1, "status": "passed", "finding_counts": {}}),
            _presentation_fixture_write_spec(arg="--markdown-out", kind="text", content="# Fixture narration series review\n"),
        ),
    },
    "build_presenter_video.py": {
        "writes": (_presentation_fixture_write_spec(arg="--output", kind="bytes", content=b"\x00\x00\x00\x18ftypmp42fixture"),),
    },
    "build_notebooklm_style_presenter_video.py": {
        "writes": (_presentation_fixture_write_spec(arg="--output", kind="bytes", content=b"\x00\x00\x00\x18ftypmp42fixture"),),
    },
    "validate_video_sync.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--single-speaker-report", kind="json", content={"schema_version": 1, "status": "pass"}),
            _presentation_fixture_write_spec(arg="--notebooklm-report", kind="json", content={"schema_version": 1, "status": "pass"}),
        ),
    },
    "validate_video_delivery.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-out", kind="json", content={"schema_version": 1, "status": "pass", "check_counts": {"pass": 4}, "checks": []}),
        ),
    },
    "presentation_editorial_gate.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-out", kind="json", content={"schema_version": 1, "status": "passed", "advisory": True, "finding_counts": {}, "findings": []}),
        ),
    },
    "review_presentation.py": {
        "writes": (
            _presentation_fixture_write_spec(arg="--report-output", kind="json", content={"schema_version": 1, "status": "passed"}),
            _presentation_fixture_write_spec(arg="--markdown-output", kind="text", content="# Fixture presentation review\n"),
        ),
    },
}


def _presentation_fixture_deck_path(command: list[str], filename: str) -> str | None:
    """A path next to the command's ``--deck-dir``, for scripts that name their
    outputs themselves instead of taking an output flag."""

    deck_dir = _presentation_fixture_arg(command, "--deck-dir")
    return str(Path(deck_dir) / filename) if deck_dir else None


def _presentation_fixture_apply_write(write_spec: dict[str, object], command: list[str]) -> None:
    """Perform one declarative write from ``_PRESENTATION_FIXTURE_TABLE`` --
    the single generic writer every table entry (of every kind) goes
    through; absence of the target flag is a silent no-op (an
    intentionally-optional flag for that call), never an error."""

    resolver = write_spec["resolver"]
    if resolver is not None:
        path_str: str | None = resolver(command)
    else:
        path_str = _presentation_fixture_arg(command, write_spec["arg"])
    if not path_str:
        return
    path = Path(path_str)
    kind = write_spec["kind"]
    if kind == "json":
        path.parent.mkdir(parents=True, exist_ok=True)
        content = write_spec["content"]
        if isinstance(content, dict) and content.get("task_id") == "presentation.claim_fidelity_review":
            content = {**content, "phase": _presentation_fixture_arg(command, "--phase")}
        path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    elif kind == "text":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(write_spec["content"], encoding="utf-8")
    elif kind == "bytes":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(write_spec["content"])
    elif kind == "dir_file":
        filename, payload = write_spec["content"]
        path.mkdir(parents=True, exist_ok=True)
        (path / filename).write_bytes(payload)
    else:
        raise AssertionError(f"unknown presentation fixture write kind: {kind}")


# The per-deck chain runs for the 12 TOPIC decks only. The 2 executive decks
# are authored by the digest chain (build_executive_summary.py ->
# build_executive_videos.py) and Stage 10 verifies their artifacts instead of
# rebuilding them, so they invoke none of these per-deck owner scripts. Deck
# SCOPE and deck RECORDS are still 14 -- only chain invocations are 12.
#
# EXCEPTION (2026-09-15): the two RENDER GATES -- inspect_rendered_slides.py and
# inspect_rendered_slides_llm.py -- now run for all 14 decks. Executive decks
# are still verified rather than rebuilt, but "verified" includes looking at the
# rendered slides. They previously ran on 12, which is how both executive decks
# shipped with an unreadable opening slide. Use _RENDER_GATE_DECKS for those two.
_PER_DECK_CHAIN_DECKS = 12
_RENDER_GATE_DECKS = 14

def _build_fake_presentation_command_runner(
    *, call_log: list[str] | None = None
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    """Parametrized fake ``command_runner`` covering every owner-script
    subprocess the full presentation chain invokes: looks up each command's
    script basename in ``_PRESENTATION_FIXTURE_TABLE`` and performs exactly
    the declared write(s) -- a SINGLE table-driven dispatcher, not ~14+
    hand-written near-duplicate ``if "..." in command:`` branches
    (10-PATTERNS.md / RESEARCH recommendation). No real subprocess, network,
    or LLM call ever occurs. An unrecognized command basename raises
    ``AssertionError`` -- proving zero silent pass-through for any
    unaccounted-for script. Appends the basename to ``call_log`` (if
    supplied) so a test can assert invocation order/count.
    """

    log = call_log if call_log is not None else []

    def fake_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        script = next((part for part in command if isinstance(part, str) and part.endswith(".py")), "")
        basename = Path(script).name if script else ""
        entry = _PRESENTATION_FIXTURE_TABLE.get(basename)
        if entry is None:
            raise AssertionError(f"unexpected presentation subprocess command: {command}")
        log.append(basename)
        writes = entry["writes"]
        if basename == "generate_engaging_narration.py":
            # The $0 modes never regenerate the narration: --sync-raw-from-md
            # edits the raw JSON only from the drafts (a no-op on fixture
            # drafts), and --render-from-raw rewrites only the markdown drafts.
            # Treating them as a full generation would overwrite an operator's
            # raw-narration patch with fixture text.
            if "--sync-raw-from-md" in command:
                writes = ()
            elif "--render-from-raw" in command:
                writes = tuple(spec for spec in writes if str(spec["resolver"](command)).endswith(".md"))
        for write_spec in writes:
            _presentation_fixture_apply_write(write_spec, command)
        return subprocess.CompletedProcess(command, 0, f"{basename} ok", "")

    return fake_command_runner


def test_presentation_source_binding_chain_succeeds_with_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full per-deck loop (paper matching -> 4 early CF2 scripts ->
    blueprint + review -> deck contract -> source binding) advances all 14
    in-scope decks and populates ``owner_execution_evidence`` with a real,
    non-echoed token for each of the 9 script ids this plan wires -- proven
    entirely via fixtures/mocks at $0 (10-VALIDATION.md)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-source-binding"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    call_log: list[str] = []

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_presentation_command_runner(call_log=call_log),
    )

    receipt = outcome["receipt"]
    # -- with Plan 10-04's delivery/video wiring complete, this fully
    # fixtured run reaches a genuine ``succeeded`` strict-contract receipt. --
    assert receipt["status"] == "succeeded", receipt
    checks = receipt["checks"]
    assert checks["deck_count"] == 14
    assert len(checks["deck_scope"]) == 14
    assert len(checks["deck_records"]) == 14

    # -- one real, non-echoed evidence token per script id this plan wires;
    # never a bare echo of the id itself (RESEARCH Pattern 2 / Pitfall 6). --
    owner_evidence = checks["owner_execution_evidence"]
    for script_id in _PRESENTATION_SOURCE_BINDING_SCRIPT_IDS:
        token = owner_evidence[script_id]
        assert token and token != script_id, script_id
        assert ":" in token, script_id

    # -- paper matching never appears as a subprocess call (RESEARCH Critical
    # Finding 3 / in-process call, not command_runner). --
    assert "generate_editorial_blueprint.py" in call_log
    assert not any("paper_matching" in entry or "match_papers" in entry for entry in call_log)

    # -- every one of the 9 owner scripts ran exactly once per deck (14),
    # EXCEPT build_deck_source_pack.py, which runs only for the 12 topic
    # decks: it requires storage/extractions/<run-id>/by-topic/<id>.json,
    # which the extraction stage never produces for the 2 synthetic
    # executive-overview ids, and its output is not consumed by
    # bind_presentation_sources.py (--source-pack is optional and unused
    # here) -- mirroring the executive-deck extraction-record fallback
    # already established for the 4 Critical-Finding-2 scripts. --
    for script_basename in (
        "build_presentation_source_corpus_inventory.py",
        "build_speaker_author_work_ledger.py",
        "repair_presentation_sources.py",
        "build_research_cluster_plan.py",
        "generate_editorial_blueprint.py",
        "review_presentation_blueprint.py",
        "generate_deck_contract.py",
        "bind_presentation_sources.py",
    ):
        assert call_log.count(script_basename) == _PER_DECK_CHAIN_DECKS, script_basename
    assert call_log.count("build_deck_source_pack.py") == 12, "build_deck_source_pack.py"

    # -- the per-deck step ORDER is exactly corpus_inventory -> work_ledger
    # -> source_repair -> cluster_plan -> blueprint -> blueprint_review ->
    # deck_contract -> source_pack -> bind_presentation_sources, for the
    # first deck processed. --
    first_deck_sequence = call_log[:9]
    assert first_deck_sequence == [
        "build_presentation_source_corpus_inventory.py",
        "build_speaker_author_work_ledger.py",
        "repair_presentation_sources.py",
        "build_research_cluster_plan.py",
        "generate_editorial_blueprint.py",
        "review_presentation_blueprint.py",
        "generate_deck_contract.py",
        "build_deck_source_pack.py",
        "bind_presentation_sources.py",
    ]

    # -- model_governance_task_ids records only the resolving ids this chain
    # actually invokes, never a non-resolving workflow-stage id. The fixture
    # vision-QC report always reports "passed", so the repair loop never
    # fires in this fixture and its task id is absent here. --
    assert checks["model_governance_task_ids"] == _presentation_full_chain_model_governance_task_ids(runner)
    assert runner.default_model(runner.PRESENTATION_BLUEPRINT_GENERATION_TASK_ID)
    assert runner.default_model(runner.PRESENTATION_SVG_FIGURE_GENERATION_TASK_ID)
    assert runner.default_model(runner.PRESENTATION_VISUAL_QC_TASK_ID)

    # -- real spend is reported on this genuinely succeeded receipt. --
    assert checks["provider_calls_made"] is True


# --- Plan 10-03 Task 1: programmatic figures (two-pass) + structural
# review/validate + render-spec generation ---

_PRESENTATION_TASK1_SCRIPT_IDS = (
    "presentation.generate_programmatic_figures",
    "presentation.review_programmatic_figures",
    "presentation.validate_svgs",
    "presentation.generate_render_spec",
)


def test_presentation_figures_review_validate_render_spec_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 10-03 Task 1: figures (two-pass: pass 1 $0, pass 2 PAID Phase 16
    drawsvg codegen) -> structural review ($0) -> SVG validation ($0) ->
    render-spec generation ($0) advance every in-scope deck, each with a
    real non-echoed owner-evidence token; only figure generation is wrapped
    in ``provider_receipt_context``; ``render_spec.json`` is produced per
    deck (10-VALIDATION.md)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-figures-render-spec"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    call_log: list[str] = []
    full_command_log: list[list[str]] = []

    base_runner = _build_fake_presentation_command_runner(call_log=call_log)

    def recording_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        full_command_log.append(command)
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=recording_command_runner,
    )

    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt
    checks = receipt["checks"]
    assert len(checks["deck_records"]) == 14

    owner_evidence = checks["owner_execution_evidence"]
    for script_id in _PRESENTATION_TASK1_SCRIPT_IDS:
        token = owner_evidence[script_id]
        assert token and token != script_id, script_id
        assert ":" in token, script_id

    # -- generate_programmatic_figures.py runs exactly twice per deck (pass
    # 1 $0, pass 2 PAID); every other Task 1 script runs exactly once. --
    assert call_log.count("generate_programmatic_figures.py") == 2 * _PER_DECK_CHAIN_DECKS
    for script_basename in (
        "review_programmatic_figures.py",
        "validate_svgs.py",
        "generate_render_spec.py",
    ):
        assert call_log.count(script_basename) == _PER_DECK_CHAIN_DECKS, script_basename

    # -- only the SECOND generate_programmatic_figures.py call (Phase 16 LLM
    # codegen) carries --provider/--model/--provider-cost-approved; the
    # first (deterministic) call, review, validate, and render-spec never
    # do. Renderer allowlist is drawsvg-only; matplotlib/plotly are always
    # explicitly disallowed. --
    def _has_script(cmd: list[str], basename: str) -> bool:
        return any(str(part).endswith(basename) for part in cmd)

    figures_calls = [cmd for cmd in full_command_log if _has_script(cmd, "generate_programmatic_figures.py")]
    assert len(figures_calls) == 2 * _PER_DECK_CHAIN_DECKS
    paid_figure_calls = [cmd for cmd in figures_calls if "--provider" in cmd]
    assert len(paid_figure_calls) == _PER_DECK_CHAIN_DECKS
    for cmd in paid_figure_calls:
        assert "--model" in cmd
        assert cmd[cmd.index("--provider") + 1] == "anthropic"
        assert cmd[cmd.index("--model") + 1] == _opus_model()
        assert "--provider-cost-approved" in cmd
        assert "drawsvg" in cmd
        assert "matplotlib" in cmd  # only ever as a --disallowed-default-renderer value
        assert "plotly" in cmd  # only ever as a --disallowed-default-renderer value
        assert "graphviz" not in cmd
        assert "pycairo" not in cmd
        assert "altair-vl-convert" not in cmd
    deterministic_figure_calls = [cmd for cmd in figures_calls if "--provider" not in cmd]
    assert len(deterministic_figure_calls) == _PER_DECK_CHAIN_DECKS

    for script_basename in ("review_programmatic_figures.py", "validate_svgs.py", "generate_render_spec.py"):
        for cmd in full_command_log:
            if _has_script(cmd, script_basename):
                assert "--provider" not in cmd, script_basename
                assert "--provider-cost-approved" not in cmd, script_basename

    # -- render_spec.json is a real output-existence-checked artifact per
    # deck. --
    for record in checks["deck_records"]:
        assert Path(record["render_spec_path"]).is_file()

    assert checks["model_governance_task_ids"] == _presentation_full_chain_model_governance_task_ids(runner)
    assert checks["provider_calls_made"] is True


# --- Plan 10-03 Task 2: render/export + visual QC + bounded repair loop ---


def test_presentation_render_export_screenshots_visual_qc_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 10-03 Task 2: render html -> export pdf -> pptx -> screenshots
    -> visual QC advance every in-scope deck; only visual QC (vision-LLM) is
    wrapped in ``provider_receipt_context``; ``presentation.html``/
    ``presentation.pdf`` are real output-existence-checked artifacts; the
    fixture's always-passing vision QC never invokes the repair loop
    (10-VALIDATION.md)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-render-export-qc"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    call_log: list[str] = []
    full_command_log: list[list[str]] = []
    base_runner = _build_fake_presentation_command_runner(call_log=call_log)

    def recording_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        full_command_log.append(command)
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=recording_command_runner,
    )

    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt
    checks = receipt["checks"]
    assert len(checks["deck_records"]) == 14

    owner_evidence = checks["owner_execution_evidence"]
    for script_id in (
        "presentation.md_to_revealjs",
        "presentation.export_pdf",
        "presentation.md_to_pptx",
        "presentation.inspect_rendered_slides",
        "presentation.inspect_rendered_slides_llm",
    ):
        token = owner_evidence[script_id]
        assert token and token != script_id, script_id
        assert ":" in token, script_id
    # -- ``presentation.repair_loop`` is a required (though coverage-exempt)
    # id even when the repair loop never fires: a real, non-fabricated
    # "repair_not_required" token referencing the passing visual-QC report
    # is recorded instead of a bare id echo or an absent entry. --
    repair_loop_token = owner_evidence["presentation.repair_loop"]
    assert repair_loop_token.startswith("repair_not_required:")

    def _has_script(cmd: list[str], basename: str) -> bool:
        return any(str(part).endswith(basename) for part in cmd)

    for script_basename in ("md_to_revealjs.py", "export_revealjs_pdf.py", "md_to_pptx.py", "inspect_rendered_slides.py"):
        for cmd in full_command_log:
            if _has_script(cmd, script_basename):
                assert "--provider" not in cmd, script_basename

    visual_qc_calls = [cmd for cmd in full_command_log if _has_script(cmd, "inspect_rendered_slides_llm.py")]
    assert len(visual_qc_calls) == _RENDER_GATE_DECKS
    assert call_log.count("run_presentation_repair_loop.py") == 0

    for record in checks["deck_records"]:
        assert Path(record["presentation_html_path"]).is_file()
        assert Path(record["presentation_pdf_path"]).is_file()
        # Every deck, executive included, is judged on its rendered slides.
        assert Path(record["visual_llm_qc_report_path"]).is_file()
        assert Path(record["inspect_report_path"]).is_file()

    assert checks["model_governance_task_ids"] == _presentation_full_chain_model_governance_task_ids(runner)
    assert checks["provider_calls_made"] is True


def test_presentation_repair_loop_invoked_on_qc_blocker_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the vision-QC report records blocker findings for a deck, the
    bounded repair loop (PAID, wrapped in ``provider_receipt_context``,
    ``--max-attempts 2``) is invoked; a repair receipt reporting
    ``status: passed`` lets that deck advance normally (10-VALIDATION.md)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-repair-loop-recovers"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    call_log: list[str] = []
    base_runner = _build_fake_presentation_command_runner(call_log=call_log)

    def _arg(command: list[str], flag: str) -> str | None:
        if flag in command:
            index = command.index(flag)
            if index + 1 < len(command):
                return command[index + 1]
        return None

    def _has_script(cmd: list[str], basename: str) -> bool:
        return any(str(part).endswith(basename) for part in cmd)

    qc_calls_by_report: dict[str, int] = {}
    review_calls_by_report: dict[str, int] = {}
    figure_commands: list[list[str]] = []
    crashed_once: set[str] = set()

    def flaky_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if _has_script(command, "review_programmatic_figures.py"):
            # The semantic reviewer rejects the FIRST redraw (a required
            # entity label went missing) and accepts the second one.
            report_path = _arg(command, "--report-output")
            assert report_path
            review_calls_by_report[report_path] = review_calls_by_report.get(report_path, 0) + 1
            if review_calls_by_report[report_path] == 2:
                Path(report_path).write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": "fail",
                            "findings": [
                                {
                                    "check_id": "required_entity_coverage",
                                    "slide_id": "sl_01_fixture",
                                    "message": "required_entities absent from SVG text: Closed checkpoint",
                                    "severity": "error",
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 1, "semantic review failed", "")
            return base_runner(command)
        if _has_script(command, "generate_render_spec.py"):
            # A real slide id so the pixel-judge feedback can target it.
            result = base_runner(command)
            output = _arg(command, "--output")
            assert output
            Path(output).write_text(
                json.dumps({"schema_version": 1, "slides": [{"slide_id": "sl_01_fixture"}]}),
                encoding="utf-8",
            )
            return result
        if _has_script(command, "generate_programmatic_figures.py") and "--provider" in command:
            figure_commands.append(list(command))
            assert command[command.index("--codegen-max-attempts") + 1] == str(
                runner.PRESENTATION_FIGURE_CODEGEN_MAX_ATTEMPTS
            )
            deck = _arg(command, "--presentation-id")
            assert deck
            if "--qc-feedback-json" in command and deck not in crashed_once:
                # One hard-failing figure aborts the whole redraw pass (no
                # receipt written); the runner retries the pass once.
                crashed_once.add(deck)
                return subprocess.CompletedProcess(command, 1, "", "RuntimeError: Opus SVG code generation failed")
        if _has_script(command, "inspect_rendered_slides_llm.py"):
            report_path = _arg(command, "--report-output")
            assert report_path
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            # Executive decks are judged too, but have no repair loop (their
            # figures are compiled from the digest plan, so a finding is a plan
            # change, not a redraw). This test exercises the TOPIC repair loop,
            # so let the executive decks pass outright and keep them out of the
            # per-deck round counting below.
            if "executive_summary" in report_path:
                Path(report_path).write_text(
                    json.dumps({"schema_version": 1, "status": "passed", "slides": []}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "qc passed", "")
            qc_calls_by_report[report_path] = qc_calls_by_report.get(report_path, 0) + 1
            if qc_calls_by_report[report_path] == 1:
                # First pixel judgement fails one slide; the redraw round fixes it.
                Path(report_path).write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": "failed",
                            "slides": [
                                {
                                    "slide_number": 1,
                                    "status": "fail",
                                    "issues": [
                                        {
                                            "check_id": "text_clipped",
                                            "severity": "blocker",
                                            "description": "label overflows its box",
                                        }
                                    ],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 1, "qc failed", "")
            Path(report_path).write_text(
                json.dumps({"schema_version": 1, "status": "passed", "slides": []}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "qc passed", "")
        if _has_script(command, "run_presentation_repair_loop.py"):
            receipt_path = _arg(command, "--receipt-output")
            assert receipt_path
            Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
            Path(receipt_path).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "blocked",
                        "attempts": [{"attempt": 1, "llm_qc_rerun_required": True}],
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 1, "fresh vision QC required", "")
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=flaky_command_runner,
    )

    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt
    checks = receipt["checks"]
    owner_evidence = checks["owner_execution_evidence"]
    assert "presentation.repair_loop" in owner_evidence
    assert not owner_evidence["presentation.repair_loop"].startswith("repair_not_required:")
    # -- one redraw round per deck: the vision QC ran twice per deck, the
    # paid figure pass ran twice per deck, and the retry carried the
    # pixel-judge feedback targeting the failing slide's real slide id. --
    assert set(qc_calls_by_report.values()) == {2}
    # -- per deck: first pass + a crashed redraw retried once + one
    # semantic-review-driven redraw --
    assert len(figure_commands) == 4 * _PER_DECK_CHAIN_DECKS
    assert len(crashed_once) == _PER_DECK_CHAIN_DECKS
    assert set(review_calls_by_report.values()) == {3}
    retry_commands = [cmd for cmd in figure_commands if "--qc-feedback-json" in cmd]
    assert len(retry_commands) == 3 * _PER_DECK_CHAIN_DECKS
    feedback_path = Path(retry_commands[0][retry_commands[0].index("--qc-feedback-json") + 1])
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert list(feedback) == ["sl_01_fixture"]
    assert "text_clipped: label overflows its box" in feedback["sl_01_fixture"]
    assert "required_entities absent from SVG text: Closed checkpoint" in feedback["sl_01_fixture"]
    assert runner.default_model(runner.PRESENTATION_REPAIR_TASK_ID)
    assert checks["model_governance_task_ids"] == _presentation_full_chain_model_governance_task_ids(
        runner, repair_ran=True
    )


def test_presentation_repair_loop_bounded_and_blocks_when_still_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A still-failing QC after the bounded repair loop is recorded as a
    per-deck blocker (``presentation_repair_loop_failed``) -- the outer
    redraw/re-render/re-judge loop runs at most
    ``PRESENTATION_REPAIR_MAX_ATTEMPTS`` rounds per deck (one owner-script
    invocation per round) and never auto-raises the cost cap (T-10-04;
    10-VALIDATION.md). Under the continue-past-failures policy (operator
    decision, Plan 10-06), this fixture's QC failure applies to every deck,
    so ALL 14 decks are recorded as blocked (never silently completed) and
    the overall receipt honestly reports ``status: blocked``, never a
    fabricated ``succeeded``."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-repair-loop-blocked"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    base_runner = _build_fake_presentation_command_runner()
    repair_loop_invocations: list[list[str]] = []

    def _arg(command: list[str], flag: str) -> str | None:
        if flag in command:
            index = command.index(flag)
            if index + 1 < len(command):
                return command[index + 1]
        return None

    def _has_script(cmd: list[str], basename: str) -> bool:
        return any(str(part).endswith(basename) for part in cmd)

    review_commands: list[list[str]] = []

    def still_failing_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if _has_script(command, "review_presentation.py"):
            review_commands.append(list(command))
        if _has_script(command, "inspect_rendered_slides_llm.py"):
            report_path = _arg(command, "--report-output")
            assert report_path
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "slides": [{"slide_number": 1, "status": "fail", "issues": []}],
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 1, "qc failed", "")
        if _has_script(command, "run_presentation_repair_loop.py"):
            repair_loop_invocations.append(command)
            assert "--max-attempts" in command
            assert command[command.index("--max-attempts") + 1] == str(
                runner.PRESENTATION_REPAIR_MAX_ATTEMPTS
            )
            receipt_path = _arg(command, "--receipt-output")
            assert receipt_path
            Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
            Path(receipt_path).write_text(
                json.dumps({"schema_version": 1, "status": "blocked", "attempts": []}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 1, "repair still blocked", "")
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        command_runner=still_failing_command_runner,
    )

    receipt = outcome["receipt"]
    assert receipt["status"] == "blocked"
    assert receipt["blockers"][0]["code"] == "presentation_partial_deck_scope_blocked", receipt["blockers"]
    deck_blockers = receipt["checks"]["deck_blockers"]
    # All 14 block: the 12 topic decks exhaust the bounded repair loop, and the
    # 2 executive decks block on the unresolved findings directly. Executive
    # decks run no repair loop -- their figures are compiled from the digest
    # plan, so a finding is a plan change, not a redraw -- so their blocker
    # code differs, which is what distinguishes the two paths here.
    assert len(deck_blockers) == _RENDER_GATE_DECKS
    assert {item["blocker_code"] for item in deck_blockers} == {
        "presentation_repair_loop_failed",
        "presentation_visual_qc_findings_unresolved",
    }
    assert sum(
        1
        for item in deck_blockers
        if item["blocker_code"] == "presentation_visual_qc_findings_unresolved"
    ) == 2
    assert receipt["checks"]["deck_success_count"] == 0
    # -- the repair loop is invoked exactly PRESENTATION_REPAIR_MAX_ATTEMPTS
    # times PER DECK (one per bounded redraw round) across all 14 decks --
    # never an unbounded outer retry loop, and the continue-past-failures
    # policy still visits every deck rather than stopping after the first. --
    assert len(repair_loop_invocations) == _PER_DECK_CHAIN_DECKS * runner.PRESENTATION_REPAIR_MAX_ATTEMPTS
    assert receipt["checks"]["provider_cost_approval"]["approved"] is True

    # -- explicit operator acceptance (review canary): with zero extra
    # redraw rounds the still-failing decks proceed, each record carrying the
    # verbatim findings, and the repair-loop evidence token names the
    # acceptance rather than a pass; a filtered scope still never succeeds. --
    repair_loop_invocations.clear()
    # Sub-stage "all": the Gate A review runs after the presenter script, in
    # sub-stage B, and the assertion below needs to see it invoked.
    accepted = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="all",
        deck_ids=["01_fixture_topic"],
        visual_qc_max_rounds=0,
        accept_visual_qc_findings=True,
        command_runner=still_failing_command_runner,
    )
    accepted_receipt = accepted["receipt"]
    assert accepted_receipt["status"] == "blocked"
    assert accepted_receipt["blockers"][0]["code"] == "presentation_deck_scope_filtered"
    assert accepted_receipt["blockers"][0]["blocked_deck_ids"] == []
    assert repair_loop_invocations == []
    (record,) = accepted_receipt["checks"]["deck_records"]
    acceptance = record["visual_qc_accepted_with_findings"]
    assert acceptance["qc_failing_slide_numbers"] == [1]
    assert acceptance["qc_rounds"] == 0
    assert acceptance["accepted_by"] == "operator:--accept-visual-qc-findings"
    assert accepted_receipt["checks"]["owner_execution_evidence"]["presentation.repair_loop"].startswith(
        "visual_qc_accepted_with_findings:"
    )
    # -- the acceptance is RECORDED where the Gate A reviewers read it: a
    # verbatim findings file next to the QC report, passed to the review.
    # Before this, both reviewers refused every accepted deck on the strength
    # of the failed QC report alone (2026-09-15). --
    acceptance_path = workspace / "decks" / "01_fixture_topic" / "assets" / "generated" / "visual_qc_acceptance.json"
    assert acceptance_path.is_file()
    acceptance_record = json.loads(acceptance_path.read_text(encoding="utf-8"))
    assert acceptance_record["accepted_by"] == "operator:--accept-visual-qc-findings"
    assert [row["slide_number"] for row in acceptance_record["accepted_findings"]] == [1]
    accepted_review_commands = [
        cmd for cmd in review_commands if "--visual-qc-acceptance" in cmd
    ]
    assert accepted_review_commands, "review must be told about the acceptance"
    assert accepted_review_commands[-1][accepted_review_commands[-1].index("--visual-qc-acceptance") + 1] == str(acceptance_path)


# --- Plan 10-03 Task 3: artifact-truth validation (5th/LATE CF2 script) +
# cross-deck link audit (closes sub-stage A) ---


def test_presentation_artifact_truth_and_link_audit_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 10-03 Task 3: ``validate_artifact_truth.py`` runs LATE, per deck
    (after rendering/QC/repair), with a real non-echoed evidence token; the
    cross-deck link audit (``audit_presentation_links.py``) runs exactly
    ONCE, after the full per-deck pass across all 14 in-scope decks. Neither
    is wrapped in ``provider_receipt_context`` (both $0; 10-VALIDATION.md)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-artifact-truth-link-audit"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    call_log: list[str] = []
    full_command_log: list[list[str]] = []
    base_runner = _build_fake_presentation_command_runner(call_log=call_log)

    def recording_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        full_command_log.append(command)
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=recording_command_runner,
    )

    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt
    checks = receipt["checks"]

    owner_evidence = checks["owner_execution_evidence"]
    token = owner_evidence["presentation.artifact_truth_validation"]
    assert token and token != "presentation.artifact_truth_validation"
    assert ":" in token
    link_token = owner_evidence["presentation.link_audit"]
    assert link_token and link_token != "presentation.link_audit"
    assert ":" in link_token

    assert call_log.count("validate_artifact_truth.py") == _PER_DECK_CHAIN_DECKS
    assert call_log.count("audit_presentation_links.py") == 1

    def _has_script(cmd: list[str], basename: str) -> bool:
        return any(str(part).endswith(basename) for part in cmd)

    for cmd in full_command_log:
        if _has_script(cmd, "validate_artifact_truth.py") or _has_script(cmd, "audit_presentation_links.py"):
            assert "--provider" not in cmd
            assert "--provider-cost-approved" not in cmd

    for record in checks["deck_records"]:
        if record["deck_kind"] != "executive":
            assert Path(record["artifact_truth_report_path"]).is_file()

    link_audit_receipt_path = workspace / "decks" / "presentation_link_audit_receipt.json"
    assert link_audit_receipt_path.is_file()

    # -- the link audit runs exactly once AFTER the full per-deck pass: its
    # single call comes after every deck's own artifact_truth_validation
    # call in the log. --
    last_artifact_truth_index = max(
        index for index, name in enumerate(call_log) if name == "validate_artifact_truth.py"
    )
    link_audit_index = call_log.index("audit_presentation_links.py")
    assert link_audit_index > last_artifact_truth_index


def test_presentation_deck_figures_substage_halts_cleanly_with_interim_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``presentation_substage="deck_figures"`` halts cleanly after the link
    audit with an HONEST interim receipt (``status: succeeded``, no
    blockers, ``provider_calls_made`` reflects real spend) -- and never
    invokes any delivery/video step (Plan 10-04's own substage;
    10-VALIDATION.md)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-deck-figures-substage"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    call_log: list[str] = []

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        command_runner=_build_fake_presentation_command_runner(call_log=call_log),
    )

    assert outcome["status"] == "succeeded"
    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded"
    assert receipt["blockers"] == []
    assert receipt["checks"]["presentation_deck_figures_substage_complete"] is True
    assert receipt["checks"]["provider_calls_made"] is True

    # -- never invokes any delivery/video step (Plan 10-04's own substage). --
    delivery_video_scripts = (
        "generate_presenter_script.py",
        "presentation_editorial_gate.py",
        "build_presenter_video.py",
        "build_notebooklm_style_presenter_video.py",
        "validate_video_sync.py",
        "validate_video_delivery.py",
        "review_presentation.py",
    )
    for script in delivery_video_scripts:
        assert script not in call_log

    assert call_log.count("audit_presentation_links.py") == 1


def test_presentation_per_deck_blocker_continues_to_remaining_decks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continue-past-failures policy (operator decision, Plan 10-06): a
    single deck's blocker (here, blueprint generation for one of 14 decks)
    no longer halts the whole stage. It is recorded in the final receipt's
    ``checks["deck_blockers"]`` and the per-deck loop proceeds to the
    remaining decks -- proven by asserting a LATE deck_figures-scope step
    (visual QC) genuinely ran for all 13 other decks, not merely that the
    loop moved past deck 3 without doing real work. The final outcome
    honestly reports ``status: blocked`` (never a fabricated ``succeeded``
    for a scope that is not fully real), with the failing deck's id and
    blocker code visible in the receipt."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-per-deck-continue"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    failing_deck_id = "03_fixture_topic"
    call_log: list[str] = []
    base_runner = _build_fake_presentation_command_runner(call_log=call_log)

    def selectively_failing_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if _presentation_fixture_arg(command, "--presentation-id") == failing_deck_id and any(
            str(part).endswith("generate_editorial_blueprint.py") for part in command
        ):
            # -- log the attempt even though it's intercepted before
            # reaching ``base_runner`` (which is where the table-driven
            # dispatcher normally appends), so the assertion below can
            # confirm the failing deck's step was genuinely ATTEMPTED, not
            # merely that the loop moved past it. --
            call_log.append("generate_editorial_blueprint.py")
            return subprocess.CompletedProcess(command, 1, "", "fixture blueprint failure")
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        command_runner=selectively_failing_command_runner,
    )

    receipt = outcome["receipt"]
    assert outcome["status"] == "blocked"
    assert receipt["status"] == "blocked"
    assert receipt["blockers"][0]["code"] == "presentation_partial_deck_scope_blocked"
    assert receipt["blockers"][0]["deck_ids"] == [failing_deck_id]

    deck_blockers = receipt["checks"]["deck_blockers"]
    assert len(deck_blockers) == 1
    assert deck_blockers[0]["deck_id"] == failing_deck_id
    assert deck_blockers[0]["blocker_code"] == "presentation_blueprint_generation_failed"
    assert receipt["checks"]["deck_blocker_count"] == 1
    assert receipt["checks"]["deck_success_count"] == 13
    assert receipt["checks"]["presentation_deck_figures_substage_complete"] is False
    # -- real spend/work for the 13 succeeded decks is honestly reflected,
    # unlike a zero-spend gate block. --
    assert receipt["checks"]["provider_calls_made"] is True

    # -- the loop attempted blueprint generation for ALL 14 decks (the
    # failing one included), not just up to the first failure. --
    assert call_log.count("generate_editorial_blueprint.py") == _PER_DECK_CHAIN_DECKS
    # -- every OTHER deck genuinely advanced all the way through the
    # deck_figures-scope chain (a LATE step, visual QC) -- proving real
    # continuation, not merely an attempted-but-abandoned next step. --
    assert call_log.count("inspect_rendered_slides_llm.py") == _RENDER_GATE_DECKS - 1
    # -- the cross-deck link audit still runs once over whatever decks
    # genuinely completed. --
    assert call_log.count("audit_presentation_links.py") == 1


def test_presentation_partial_failure_resume_completes_only_the_blocked_deck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``--resume`` re-run after a partial per-deck failure does not
    re-invoke any already-succeeded deck's paid steps (paper matching, or
    any other subprocess) -- only the previously-blocked deck's remaining
    work is genuinely retried, proven by exploding fakes for paper matching
    and for every already-succeeded deck's id appearing in any subprocess
    command (RESEARCH Pitfall 6 / Pattern 1, applied to the new per-deck
    continue-past-failures policy). The previously-blocked deck completes
    and the overall receipt honestly transitions from ``blocked`` to
    ``succeeded``."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-partial-resume"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    failing_deck_id = "03_fixture_topic"
    base_runner = _build_fake_presentation_command_runner()

    def selectively_failing_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if _presentation_fixture_arg(command, "--presentation-id") == failing_deck_id and any(
            str(part).endswith("generate_editorial_blueprint.py") for part in command
        ):
            return subprocess.CompletedProcess(command, 1, "", "fixture blueprint failure")
        return base_runner(command)

    first_outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        command_runner=selectively_failing_command_runner,
    )
    assert first_outcome["receipt"]["status"] == "blocked"
    assert first_outcome["receipt"]["checks"]["deck_blocker_count"] == 1

    # -- paper matching (the true PAID in-process call) must never be
    # re-invoked on resume: matched_papers.json already exists for all 14
    # decks (blueprint generation fails AFTER paper matching in the chain),
    # including the previously-blocked deck. --
    def exploding_paper_matching(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("paper matching must not be re-invoked on resume for any deck")

    monkeypatch.setattr(
        runner.presentation_paper_matching, "run_paper_matching_stage", exploding_paper_matching
    )

    # -- every deck id OTHER than the previously-blocked one must never
    # appear in any subprocess command on resume -- a black-box proof that
    # covers every step (paid or not) regardless of which CLI flag a given
    # owner script uses to carry the deck id. --
    successful_deck_ids = {f"{index:02d}_fixture_topic" for index in range(1, 13)} - {failing_deck_id}
    successful_deck_ids |= {"executive_summary_short", "executive_summary_long"}

    second_call_log: list[str] = []
    second_base_runner = _build_fake_presentation_command_runner(call_log=second_call_log)

    def resumed_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if any(str(part) in successful_deck_ids for part in command):
            raise AssertionError(
                f"already-succeeded deck must not be re-invoked on resume: {command}"
            )
        return second_base_runner(command)

    second_outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        command_runner=resumed_command_runner,
    )

    # -- the previously-blocked deck's remaining chain now genuinely
    # completes, and the overall receipt honestly reflects full
    # completion -- never a fabricated intermediate state. --
    assert second_outcome["receipt"]["status"] == "succeeded", second_outcome["receipt"]
    assert second_outcome["receipt"]["checks"]["presentation_deck_figures_substage_complete"] is True
    assert second_outcome["receipt"]["checks"]["deck_blocker_count"] == 0
    # -- the previously-blocked deck's blueprint generation (and everything
    # downstream of it in its own chain) was genuinely retried -- not
    # skipped -- exactly once. --
    assert second_call_log.count("generate_editorial_blueprint.py") == 1


def test_presentation_source_binding_chain_idempotent_resume_skips_paid_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``resume=True`` pass over a workspace whose per-deck outputs
    already exist never re-invokes paper matching or any owner-script
    subprocess -- proven by exploding fakes that raise if ever called
    (RESEARCH Pitfall 6 / Pattern 1, applied at per-step granularity;
    10-VALIDATION.md)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-source-binding-resume"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    first_outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_presentation_command_runner(),
    )
    # -- with Plan 10-04's delivery/video wiring complete, a fully-populated
    # fixture run now reaches a genuine ``succeeded`` strict-contract receipt
    # instead of the old "not_implemented" placeholder. --
    assert first_outcome["receipt"]["status"] == "succeeded", first_outcome["receipt"]

    def exploding_paper_matching(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("paper matching must not be re-invoked once matched_papers.json exists")

    def exploding_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"command_runner must not be re-invoked on a per-step resume skip: {command}")

    monkeypatch.setattr(
        runner.presentation_paper_matching, "run_paper_matching_stage", exploding_paper_matching
    )

    second_outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=exploding_command_runner,
    )

    # -- the resume short-circuit returns the existing succeeded receipt
    # without re-invoking paper matching or any subprocess. --
    assert second_outcome["receipt"]["status"] == "succeeded", second_outcome["receipt"]
    assert len(second_outcome["receipt"]["checks"]["deck_records"]) == 14


def test_presentation_source_binding_chain_structural_wiring() -> None:
    """Structural (grep-based) proof of 10-VALIDATION.md's static acceptance
    criteria: no ``shell=True`` anywhere in the runner; paper matching is an
    in-process call (never a subprocess); only paper matching and blueprint
    generation are wrapped in ``provider_receipt_context`` within the
    per-deck source-binding chain; the 4 early Critical-Finding-2 scripts
    appear in the exact required order."""

    runner = load_runner_module()
    # The chain, its step helpers and the stage shell live in the shared
    # production package since the 2026-09-27 lift; the runner wraps them.
    production_dir = REPO_ROOT / "src" / "conference_pipeline" / "production"
    source = (production_dir / "presentation_chain.py").read_text(encoding="utf-8")
    steps_source = (production_dir / "presentation_steps.py").read_text(encoding="utf-8")
    stage_source = (production_dir / "presentation_stage.py").read_text(encoding="utf-8")
    for text in (Path(runner.__file__).read_text(encoding="utf-8"), source, steps_source, stage_source):
        assert "shell=True)" not in text
        assert "shell=True," not in text

    chain_start = source.index("def _deck_chain_steps(")
    chain_end = source.index("\ndef deck_figures_record(", chain_start)
    chain_body = source[chain_start:chain_end]

    assert "presentation_paper_matching.run_paper_matching_stage(" in chain_body
    assert "match_papers_from_artifact_keypoints.py" not in chain_body

    # -- 15 paid=True call sites via the shared owner-step / narration
    # content-step helpers: blueprint generation (Plan 10-02); figure-spec
    # authoring on the compiler route (1, Plan 10-03 route B -- one cheap call
    # per slide that decides WHAT a figure claims; the compile step itself is
    # $0 and deterministic); claim_fidelity_review (1 shared call site, run
    # after figure authoring and again after narration fixes); figures pass 2
    # / Phase 16 LLM codegen and its bounded pixel-judge redraw (Plan 10-03,
    # 2 sites sharing one command); the Claude-authored engaging narration
    # (1, mirroring the orchestrator's engaging_narration stage); single- and
    # NotebookLM-mode narration fix + series-fix (4, via
    # _presentation_owner_step) and review + series-review (3, via
    # _presentation_narration_content_step) (Plan 10-04 Task 1); both video
    # builders + the 3-model presentation review (3, Plan 10-04 Task 2).
    # Paper matching, the special-cased vision-QC call, AND the repair-loop
    # owner script (whose receipt, not exit code, is the verdict) wrap
    # provider_receipt_context inline instead of through either helper (each
    # helper's own body wraps paid subprocess calls once, generically, for
    # every ``paid=True`` caller). --
    assert chain_body.count("paid=True") == 15
    assert "scripts/presentations/review_claim_fidelity.py" in chain_body
    assert 'for phase in ("figures", "narration")' in chain_body
    assert 'presentation_review_command.extend(["--claim-fidelity-report", str(report_path)])' in chain_body
    assert 'StepInput(path, optional=True) for path in claim_fidelity_report_paths' in chain_body
    assert (chain_body.index("generate_figure_specs.py")
            < chain_body.index('_claim_fidelity_step("figures")')
            < chain_body.index("scripts/presentations/compile_figures.py"))
    assert (chain_body.index('_rebuild_delivery_and_reconcile("post-gates")')
            < chain_body.index('_claim_fidelity_step("narration")')
            < chain_body.index("scripts/presentations/review_presentation.py"))
    assert chain_body.count("provider_receipt_context(") == 3

    # The executive branch (defined inside the span measured below) wraps its
    # own paid vision-judge call the same way the topic path does.
    executive_start = source.index("def executive_digest_chain(")
    executive_end = source.index("\ndef ", executive_start + 1)
    assert source[executive_start:executive_end].count("provider_receipt_context(") == 1

    # -- owner_step and narration_content_step both run their subprocess
    # through ``run_command``, the one place a ``paid=True`` call is wrapped
    # in provider_receipt_context, generically, for every paid caller. --
    assert steps_source.count("provider_receipt_context(") == 1
    run_command_start = steps_source.index("def run_command(")
    run_command_end = steps_source.index("\ndef ", run_command_start + 1)
    assert "provider_receipt_context(" in steps_source[run_command_start:run_command_end]

    cf2_order = [
        chain_body.index("build_presentation_source_corpus_inventory.py"),
        chain_body.index("build_speaker_author_work_ledger.py"),
        chain_body.index("repair_presentation_sources.py"),
        chain_body.index("build_research_cluster_plan.py"),
    ]
    assert cf2_order == sorted(cf2_order)

    remaining_order = [
        chain_body.index("generate_editorial_blueprint.py"),
        chain_body.index("review_presentation_blueprint.py"),
        chain_body.index("generate_deck_contract.py"),
        chain_body.index("build_deck_source_pack.py"),
        chain_body.index("bind_presentation_sources.py"),
    ]
    assert remaining_order == sorted(remaining_order)
    assert cf2_order[-1] < remaining_order[0]

    # -- Plan 10-03 Task 1: figures (two passes) -> review -> validate ->
    # render-spec, appearing exactly once each and in this relative order,
    # after source binding. --
    figures_order = [
        chain_body.index("generate_programmatic_figures.py"),
        chain_body.index("review_programmatic_figures.py"),
        chain_body.index("validate_svgs.py"),
        chain_body.index("generate_render_spec.py"),
    ]
    assert figures_order == sorted(figures_order)
    assert remaining_order[-1] < figures_order[0]
    # -- appears 3 times: the docstring reference plus the two real call
    # sites (pass 1 $0, pass 2 PAID). --
    assert chain_body.count("generate_programmatic_figures.py") == 3
    assert "matplotlib" not in chain_body
    assert "plotly" not in chain_body
    # Unprefixed in the shared module; the runner re-exports the PRESENTATION_ name.
    assert chain_body.count("PHASE16_DISALLOWED_RENDERERS") >= 1

    # -- Plan 10-03 Task 2: render html -> export pdf -> pptx -> screenshots
    # -> visual QC -> (conditional) bounded repair loop, all after
    # render-spec. --
    render_export_order = [
        chain_body.index("md_to_revealjs.py"),
        chain_body.index("export_revealjs_pdf.py"),
        chain_body.index("md_to_pptx.py"),
        chain_body.index("inspect_rendered_slides.py"),
        chain_body.index("inspect_rendered_slides_llm.py"),
        chain_body.index("run_presentation_repair_loop.py"),
    ]
    assert render_export_order == sorted(render_export_order)
    assert figures_order[-1] < render_export_order[0]
    assert chain_body.count("--max-attempts") == 1
    assert "REPAIR_MAX_ATTEMPTS" in chain_body
    assert runner.PRESENTATION_REPAIR_MAX_ATTEMPTS == runner.production_presentation_chain.REPAIR_MAX_ATTEMPTS

    # -- Plan 10-03 Task 3: artifact-truth validation (5th, LATE CF2 script)
    # runs per deck, after render/QC/repair, inside the chain; the link
    # audit runs ONCE outside the chain (in ``run_presentation_stage``
    # itself), never per-deck. --
    assert "validate_artifact_truth.py" in chain_body
    assert chain_body.index("validate_artifact_truth.py") > render_export_order[-1]
    assert "audit_presentation_links.py" not in chain_body

    stage_start = stage_source.index("def run_presentation_stage(")
    stage_body = stage_source[stage_start:]
    assert "audit_presentation_links.py" in stage_body
    assert stage_body.count("audit_presentation_links.py") == 1
    # -- the link audit's own owner-step call site is NOT paid=True. --
    link_audit_call_start = stage_body.index("audit_presentation_links.py")
    link_audit_call_end = stage_body.index("owner_step(", link_audit_call_start)
    owner_step_call_end = stage_body.index(")\n", link_audit_call_end)
    link_audit_call_region = stage_body[link_audit_call_end : owner_step_call_end + 400]
    assert "paid=False" in link_audit_call_region


# --- Plan 10-04 Task 1: delivery contracts + both-mode narration
# lint/review/fix (pre-TTS) ---


def test_presentation_delivery_and_narration_run_before_video_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 10-04 Task 1: ``generate_presenter_script.py`` writes all three
    delivery outputs (``delivery_single_speaker.json``,
    ``delivery_notebooklm.json``, ``6_presenter_script.md``); both delivery
    modes' narration is lint/review/fixed BEFORE any video/TTS step runs;
    paid narration review/fix + both video builders + the 3-model review
    are wrapped in ``provider_receipt_context``, lint and video-sync
    validation are not (10-VALIDATION.md)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-delivery-narration-order"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    call_log: list[str] = []
    full_command_log: list[list[str]] = []
    base_runner = _build_fake_presentation_command_runner(call_log=call_log)

    def recording_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        full_command_log.append(command)
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=recording_command_runner,
    )

    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt
    checks = receipt["checks"]

    for record in checks["deck_records"]:
        assert Path(record["delivery_single_speaker_path"]).is_file()
        assert Path(record["delivery_notebooklm_path"]).is_file()
        if record["deck_kind"] != "executive":
            assert Path(record["presenter_script_md_path"]).is_file()

    def _has_script(cmd: list[str], basename: str) -> bool:
        return any(str(part).endswith(basename) for part in cmd)

    # -- both-mode narration lint/review/fix (+ NotebookLM series-review/fix)
    # each run exactly once per deck (14). --
    for script_basename in (
        "generate_presenter_script.py",
        "lint_presenter_narration.py",
        "review_presenter_narration.py",
        "fix_presenter_narration.py",
        "lint_narration_script.py",
        "review_narration_script.py",
        "fix_narration_script.py",
        "review_narration_series.py",
        "fix_narration_series.py",
    ):
        assert call_log.count(script_basename) == _PER_DECK_CHAIN_DECKS, script_basename
    # -- generate_engaging_narration.py runs three times per deck: to write
    # the narration, to render the ENGAGING markdown from the raw JSON
    # (--render-from-raw) before the fixers edit it, and to carry the fixers'
    # edits back into the raw JSON TTS is rebuilt from (--sync-raw-from-md). --
    assert call_log.count("generate_engaging_narration.py") == 3 * _PER_DECK_CHAIN_DECKS
    # -- the delivery rebuild + hash reconcile run twice per deck: once
    # right after the engaging narration and once after the narration
    # fixers, immediately before TTS. --
    for script_basename in ("rebuild_delivery_engaging.py", "reconcile_contract_hashes.py"):
        assert call_log.count(script_basename) == 2 * _PER_DECK_CHAIN_DECKS, script_basename

    # -- per-deck ordering: delivery generation -> DECK ACCEPTANCE (the
    # multi-model presentation review) -> engaging narration -> delivery
    # rebuild + hash reconcile -> single-speaker lint/review/fix -> NotebookLM
    # lint/review/fix/series-review/series-fix -> rebuild + reconcile again ->
    # both video builders -> video-sync, for the FIRST deck processed (located
    # dynamically -- deck/figures sub-stage A runs many more steps ahead of
    # delivery generation).
    #
    # The acceptance review runs BEFORE any TTS or video work, and this
    # ordering is the point of the assertion. It used to run last, after both
    # videos: on the first full CVPR 2026 run that cost 24 videos and two and a
    # half hours, because the review then failed all twelve decks on figure
    # defects its reviewers had confirmed in the screenshots, and the figure
    # fix that followed invalidated every video. The review reads the
    # blueprint, deck contract, render spec, source binding, visual-QC report,
    # screenshots and narration script -- never a video -- so nothing
    # expensive needs to exist for it to refuse a deck. --
    first_presenter_script_index = call_log.index("generate_presenter_script.py")
    first_deck_narration_video_sequence = call_log[
        first_presenter_script_index : first_presenter_script_index + 22
    ]
    # Gate A sits after the narration is FINAL (engaging rewrite, lints,
    # fixes, delivery rebuild) and before anything is synthesized or
    # rendered. Reviewing before the engaging rewrite handed the reviewers
    # the mechanical draft next to a delivery contract carrying the engaging
    # text, and Gemini failed deck 02 on the contradiction (2026-09-15).
    assert first_deck_narration_video_sequence == [
        "generate_presenter_script.py",
        "generate_engaging_narration.py",
        "generate_engaging_narration.py",
        "rebuild_delivery_engaging.py",
        "reconcile_contract_hashes.py",
        "lint_presenter_narration.py",
        "review_presenter_narration.py",
        "fix_presenter_narration.py",
        "lint_narration_script.py",
        "review_narration_script.py",
        "fix_narration_script.py",
        "review_narration_series.py",
        "fix_narration_series.py",
        "generate_engaging_narration.py",
        "rebuild_delivery_engaging.py",
        "reconcile_contract_hashes.py",
        "review_presentation.py",
        # Advisory, $0: the narration budget is on the record before TTS.
        "presentation_editorial_gate.py",
        "build_presenter_video.py",
        "build_notebooklm_style_presenter_video.py",
        "validate_video_sync.py",
        # $0 path-based delivery checks on both final MP4s.
        "validate_video_delivery.py",
    ], call_log[:30]

    # -- paid narration review/fix + both video builders + presentation
    # review are wrapped in provider_receipt_context; lint + video-sync are
    # not. --
    paid_scripts = (
        "generate_engaging_narration.py",
        "review_presenter_narration.py",
        "fix_presenter_narration.py",
        "review_narration_script.py",
        "fix_narration_script.py",
        "review_narration_series.py",
        "fix_narration_series.py",
        "build_presenter_video.py",
        "build_notebooklm_style_presenter_video.py",
        "review_presentation.py",
    )
    zero_dollar_scripts = (
        "generate_presenter_script.py",
        "lint_presenter_narration.py",
        "lint_narration_script.py",
        "presentation_editorial_gate.py",
        "validate_video_sync.py",
        "validate_video_delivery.py",
    )
    # The chain lives in the shared production package since the 2026-09-27 lift.
    chain_source = (REPO_ROOT / "src" / "conference_pipeline" / "production" / "presentation_chain.py").read_text(
        encoding="utf-8"
    )
    chain_start = chain_source.index("def _deck_chain_steps(")
    chain_end = chain_source.index("\ndef deck_figures_record(", chain_start)
    chain_body = chain_source[chain_start:chain_end]
    for script in paid_scripts:
        call_index = chain_body.index(f'"scripts/presentations/{script}"')
        following = chain_body[call_index : call_index + 4000]
        assert "paid=True" in following, script
        assert "paid=False" not in following.split("paid=True")[0], script
    for script in zero_dollar_scripts:
        call_index = chain_body.index(f'"scripts/presentations/{script}"')
        following = chain_body[call_index : call_index + 2500]
        assert "paid=False" in following, script
        assert "paid=True" not in following.split("paid=False")[0], script

    # -- model_governance_task_ids records presenter_script_generation only
    # after generate_presenter_script.py genuinely succeeds, and every
    # recorded id resolves. --
    assert runner.PRESENTATION_PRESENTER_SCRIPT_GENERATION_TASK_ID in checks["model_governance_task_ids"]
    for task_id in checks["model_governance_task_ids"]:
        assert runner.default_model(task_id)


# --- Plan 10-04 Task 2: both videos + video-sync validation + the 3-model
# presentation review + executive-deck finalization ---


def test_presentation_video_and_review_wired_executive_decks_included(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 10-04 Task 2: both videos, video-sync validation (both reports),
    and the 3-model presentation review are produced per deck; the 2
    executive-overview decks are driven through the identical chain (deck
    scope stays 14) (10-VALIDATION.md)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-video-review-executive"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    call_log: list[str] = []

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_presentation_command_runner(call_log=call_log),
    )

    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt
    checks = receipt["checks"]
    assert checks["deck_count"] == 14
    assert len(checks["deck_records"]) == 14

    executive_records = [record for record in checks["deck_records"] if record["deck_kind"] == "executive"]
    assert len(executive_records) == 2
    for record in executive_records:
        # Built by the digest chain and VERIFIED here, so every recorded path
        # is a real file. There is no presentation_review.json: the 3-model
        # review reads the screenshots and vision-QC report that the per-deck
        # figure pipeline produces, and an executive deck has neither. Its
        # equivalent evidence is the digest chain's own legibility and
        # duration gates, asserted below.
        assert record["authored_by"] == "digest_chain"
        assert Path(record["presentation_video_path"]).is_file()
        assert Path(record["presentation_video_notebooklm_path"]).is_file()
        assert Path(record["video_sync_validation_single_speaker_path"]).is_file()
        assert Path(record["video_sync_validation_notebooklm_path"]).is_file()
        assert Path(record["video_delivery_validation_path"]).is_file()
        assert Path(record["figure_svg_validation_path"]).is_file()
        assert Path(record["presentation_pdf_path"]).is_file()
        assert "presentation_review_path" not in record

    for script_basename in (
        "build_presenter_video.py",
        "build_notebooklm_style_presenter_video.py",
        "validate_video_sync.py",
        "review_presentation.py",
    ):
        assert call_log.count(script_basename) == _PER_DECK_CHAIN_DECKS, script_basename

    owner_evidence = checks["owner_execution_evidence"]
    for script_id in (
        "presentation.build_presenter_video",
        "presentation.build_notebooklm_video",
        "presentation.video_sync_validation",
        "presentation.video_delivery_validation",
        "presentation.presentation_review",
        "presentation.executive_overview_decks",
    ):
        token = owner_evidence[script_id]
        assert token and token != script_id, script_id


# --- Plan 10-04 Task 3: the complete 35-entry owner-evidence map + final
# strict-contract receipt assembly ---


def test_presentation_owner_execution_evidence_covers_all_35_required_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final receipt's ``owner_execution_evidence`` has exactly one
    real, non-echoed, non-empty token per each of the 35
    ``stage_owner_registry.json`` presentation ``required_script_ids`` --
    including the 29 coverage-exempt ids (still required at runtime by
    ``_check_owner_execution_evidence``) and the 5 ``phase13_*``
    umbrella-alias ids the automated CVPR2026 runner never invokes (each
    referencing its own named aggregate artifact, never a bare id echo or a
    fabricated placeholder). No ``forbidden_script_id`` ever appears
    anywhere in the receipt (10-VALIDATION.md; PLAN.md Task 3)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-owner-execution-evidence"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_presentation_command_runner(),
    )

    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt

    registry = runner.load_stage_script_registry()
    required_ids = list(runner.required_script_ids_for_stage(registry, "presentation"))
    # 36 since presentation.video_delivery_validation joined the binding (2026-09-27).
    assert len(required_ids) == 36
    assert "presentation.video_delivery_validation" in required_ids

    owner_evidence = receipt["owner_execution_evidence"]
    assert set(owner_evidence) >= set(required_ids)
    for owner_id in required_ids:
        token = owner_evidence.get(owner_id)
        assert token, owner_id
        assert token != owner_id, owner_id
        assert isinstance(token, str) and token.strip(), owner_id
        assert "evidence_unavailable" not in token, owner_id

    # -- the 5 phase13_* umbrella-alias ids are never invoked by the
    # automated runner; each still carries a defensible, honest,
    # non-fabricated reference to its own named aggregate artifact. --
    for phase13_id in (
        "presentation.phase13_bulk_readiness",
        "presentation.phase13_canary_review_packet",
        "presentation.phase13_canary_objective_evidence",
        "presentation.phase13_completion",
        "presentation.phase13_rollout",
    ):
        assert owner_evidence[phase13_id].startswith("umbrella_alias_out_of_scope:"), phase13_id

    forbidden_ids = registry["stage_bindings"]["presentation"]["forbidden_script_ids"]
    assert len(forbidden_ids) == 4
    receipt_text = json.dumps(receipt)
    for forbidden_id in forbidden_ids:
        assert forbidden_id not in owner_evidence
        assert forbidden_id not in receipt_text


def test_presentation_final_stage_contract_passes_and_fails_on_dropped_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``verify_stage_contract(strict_execution=True)`` passes for a
    fully-populated fixture deck set; deleting every deck's copy of a
    required glob output flips the SAME receipt to a blocker -- proving the
    coupling is genuine (reads real files on disk), not tautological
    (PLAN.md Task 3 acceptance criteria)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-final-contract"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_presentation_command_runner(),
    )
    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt
    assert receipt["stage_contract"]["status"] == "pass", receipt["stage_contract"]

    # -- deliberately drop every deck's copy of a required glob output
    # (render_spec.json) and re-verify the SAME receipt directly against
    # the real filesystem -- the contract must now fail. --
    dropped = list(workspace.glob("decks/*/render_spec.json"))
    assert len(dropped) == 14
    for path in dropped:
        path.unlink()

    re_verified = runner.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="presentation",
        receipt=receipt,
        mode="live",
        strict_execution=True,
    )
    assert re_verified.status != "pass"
    assert any(
        blocker.get("code") == "stage_contract_required_output_glob_missing" for blocker in re_verified.blockers
    ), re_verified.blockers


# --- Plan 10-05 (Wave 5, mock-before-live fixture gate) Task 2: gate/
# behavior tests -- resume, done-gate-3 (matched vs. ranked papers),
# done-gate-4 (scaffolded decks), and genuine (non-tautological)
# owner-execution-evidence rejection of a bare-echo mutation. ---


def test_presentation_resume_skips_paid_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second ``resume=True`` pass over an already-``succeeded`` receipt
    re-invokes NO subprocess and NO in-process paper matching -- proven by
    an EXPLODING command_runner and an EXPLODING paper-matching stub, both
    of which raise ``AssertionError`` if ever called (10-VALIDATION.md row
    10-W2-04)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-resume-skips-paid-calls"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    first_outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_presentation_command_runner(),
    )
    assert first_outcome["receipt"]["status"] == "succeeded", first_outcome["receipt"]

    def exploding_paper_matching(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("paper matching must never be re-invoked on a resumed succeeded receipt")

    def exploding_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"command_runner must never be re-invoked on a resumed succeeded receipt: {command}")

    monkeypatch.setattr(runner.presentation_paper_matching, "run_paper_matching_stage", exploding_paper_matching)

    second_outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=exploding_command_runner,
    )

    assert second_outcome["status"] == "succeeded", second_outcome
    assert second_outcome["receipt"]["status"] == "succeeded", second_outcome["receipt"]
    assert second_outcome["receipt"]["receipt_id"] == first_outcome["receipt"]["receipt_id"]


def test_presentation_paper_matches_from_matched_papers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROD-12 done gate 3: the deck's paper set is sourced from
    ``matched_papers.json`` (the genuine paper-matching evidence), never
    from ``ranked_papers.json`` -- proven by seeding a DIVERGENT
    ``ranked_papers.json`` per deck (a different, wrong title) and
    confirming ``bind_presentation_sources.py``'s own ``--matched-papers``
    argument resolves to ``matched_papers.json`` (whose seeded title
    survives untouched), while ``ranked_papers.json`` is never referenced
    as an argument to any subprocess in the chain (10-VALIDATION.md row
    10-W2-03)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-matched-papers-source-of-truth"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)

    def fake_paper_matching_with_divergent_ranked(
        *, workspace_root: Path, run_id: str, topic_id: str | None = None, config: object | None = None
    ) -> dict[str, object]:
        result = _fake_presentation_paper_matching_stage(
            workspace_root=workspace_root, run_id=run_id, topic_id=topic_id, config=config
        )
        # -- seed a DIVERGENT ranked_papers.json (compatibility-only, done
        # gate 3) for this deck -- must never be consumed as the source of
        # the deck's paper set. --
        deck_dir = Path(workspace_root) / "decks" / str(topic_id)
        deck_dir.mkdir(parents=True, exist_ok=True)
        (deck_dir / "ranked_papers.json").write_text(
            json.dumps({"schema_version": 1, "papers": [{"title": "WRONG Ranked-Only Paper"}]}) + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        runner.presentation_paper_matching, "run_paper_matching_stage", fake_paper_matching_with_divergent_ranked
    )

    full_command_log: list[list[str]] = []
    base_runner = _build_fake_presentation_command_runner()

    def recording_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        full_command_log.append(command)
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=recording_command_runner,
    )
    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt

    def _has_script(cmd: list[str], basename: str) -> bool:
        return any(str(part).endswith(basename) for part in cmd)

    bind_commands = [cmd for cmd in full_command_log if _has_script(cmd, "bind_presentation_sources.py")]
    assert len(bind_commands) == _PER_DECK_CHAIN_DECKS
    for cmd in bind_commands:
        matched_papers_arg = _presentation_fixture_arg(cmd, "--matched-papers")
        assert matched_papers_arg, cmd
        assert matched_papers_arg.endswith("matched_papers.json"), cmd
        matched_payload = json.loads(Path(matched_papers_arg).read_text(encoding="utf-8"))
        assert matched_payload["papers"][0]["title"] == "Fixture Paper"

    # -- ranked_papers.json is present on disk (compatibility-only --
    # candidate corpus-inventory source-input evidence, unrelated to the
    # deck's own paper set) but is NEVER passed to
    # bind_presentation_sources.py -- the ONE call site that determines the
    # deck's citable paper set only ever receives ``--matched-papers``. --
    ranked_papers_on_disk = list(workspace.glob("decks/*/ranked_papers.json"))
    assert len(ranked_papers_on_disk) == _PER_DECK_CHAIN_DECKS
    for cmd in bind_commands:
        for part in cmd:
            assert "ranked_papers.json" not in str(part), cmd


def test_presentation_scaffolded_deck_cannot_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROD-12 done gate 4: a deck whose ``generate_deck_contract.py``
    output is scaffold-only (a zero-byte file -- the exact shape a stub/
    placeholder script would produce) cannot reach ``status: succeeded`` --
    ``_presentation_owner_step``'s own ``is_file()``/non-empty check catches
    it (10-VALIDATION.md row 10-W2-02). Under the continue-past-failures
    policy (operator decision, Plan 10-06), this fixture's scaffold applies
    to every deck, so ALL 14 decks are recorded as per-deck blockers with
    the genuine, distinct blocker code -- and the overall receipt still
    never reaches a fabricated ``status: succeeded``."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-scaffolded-deck"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    base_runner = _build_fake_presentation_command_runner()

    def scaffolded_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if any(str(part).endswith("generate_deck_contract.py") for part in command):
            output_path = _presentation_fixture_arg(command, "--output")
            assert output_path, command
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            # -- scaffold marker: an empty (zero-byte) file, exactly the
            # shape a stub/incomplete generator would leave behind. --
            Path(output_path).write_text("", encoding="utf-8")
            receipt_path = _presentation_fixture_arg(command, "--receipt-output")
            if receipt_path:
                Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
                Path(receipt_path).write_text(json.dumps({"status": "pass"}), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "deck contract scaffold-only", "")
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="deck_figures",
        command_runner=scaffolded_command_runner,
    )

    assert outcome["status"] == "blocked"
    receipt = outcome["receipt"]
    assert receipt["status"] == "blocked"
    assert receipt["blockers"][0]["code"] == "presentation_partial_deck_scope_blocked", receipt["blockers"]
    deck_blockers = receipt["checks"]["deck_blockers"]
    assert len(deck_blockers) == _PER_DECK_CHAIN_DECKS
    assert {item["blocker_code"] for item in deck_blockers} == {
        "presentation_deck_contract_outputs_missing"
    }
    for item in deck_blockers:
        assert item["detail"]["missing_outputs"], item
    # The 2 executive decks are not scaffolded -- they are verified against the
    # digest chain's real artifacts and succeed.
    assert receipt["checks"]["deck_success_count"] == 2


def test_presentation_owner_execution_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-10-01: the final receipt's ``owner_execution_evidence`` has one
    real, non-empty, non-echoed token per each of the 35
    ``stage_owner_registry.json`` presentation ``required_script_ids`` --
    AND mutating any single token to a bare echo of its own owner id causes
    ``verify_stage_contract(strict_execution=True)`` to block on that exact
    id, proving the check is genuine (reads real per-id evidence), not
    tautological (10-VALIDATION.md row 10-W2-05)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-owner-evidence-genuine"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_presentation_command_runner(),
    )
    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt

    registry = runner.load_stage_script_registry()
    required_ids = list(runner.required_script_ids_for_stage(registry, "presentation"))
    # 36 since presentation.video_delivery_validation joined the binding (2026-09-27).
    assert len(required_ids) == 36
    assert "presentation.video_delivery_validation" in required_ids

    owner_evidence = receipt["owner_execution_evidence"]
    for owner_id in required_ids:
        token = owner_evidence.get(owner_id)
        assert token, owner_id
        assert isinstance(token, str) and token.strip(), owner_id
        assert token != owner_id, owner_id

    baseline_contract = runner.verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="presentation",
        receipt=receipt,
        mode="live",
        strict_execution=True,
    )
    assert baseline_contract.status == "pass", baseline_contract.blockers

    # -- mutate a representative sample of tokens (first, last, and one
    # Critical-Finding-2 id) to a bare echo of their own owner id -- each
    # mutation, independently, must flip the SAME receipt to a blocker. --
    for mutated_id in (
        required_ids[0],
        required_ids[-1],
        "presentation.artifact_truth_validation",
    ):
        mutated_receipt = copy.deepcopy(receipt)
        mutated_receipt["owner_execution_evidence"][mutated_id] = mutated_id
        mutated_contract = runner.verify_stage_contract(
            workspace=workspace,
            run_id=run_id,
            stage_id="presentation",
            receipt=mutated_receipt,
            mode="live",
            strict_execution=True,
        )
        assert mutated_contract.status != "pass", mutated_id
        assert any(
            blocker.get("code") == "stage_contract_owner_execution_evidence_missing"
            for blocker in mutated_contract.blockers
        ), (mutated_id, mutated_contract.blockers)


# --- Plan 10-05 Task 3: the full-contract fixture pass (Critical Finding 2
# closed) with a genuine per-required-output drop-and-fail loop. ---


def test_presentation_fixture_receipt_satisfies_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROD-12: a full $0 fixture run over the seeded 14-deck set produces a
    ``stage-10-presentation.json`` receipt whose
    ``verify_stage_contract(strict_execution=True, mode='live')`` passes --
    covering EVERY ``required_output`` read LIVE off
    ``stage_owner_registry.json``'s own presentation binding (never
    hardcoded), including the 5 Critical-Finding-2 outputs. A per-required-
    output drop-and-fail loop (delete every glob match for that spec,
    re-verify, confirm a blocker, restore byte-identical) proves the
    coupling is genuine, not tautological. No ``forbidden_script_id`` ever
    appears in the receipt; ``mode`` is ``'live'``, never mock/dry-run/
    planned (10-VALIDATION.md row 10-W2-01; PLAN.md Task 3)."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-full-contract-drop-and-fail"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=_build_fake_presentation_command_runner(),
    )
    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt
    assert receipt["stage_contract"]["status"] == "pass", receipt["stage_contract"]
    assert receipt["mode"] == "live"
    for forbidden_mode in ("mock", "dry-run", "planned"):
        assert receipt["mode"] != forbidden_mode

    # -- read every required_output LIVE off the registry file itself --
    # never a hand-picked/hardcoded subset. --
    registry = runner.load_stage_script_registry()
    binding = registry["stage_bindings"]["presentation"]
    required_outputs = binding["runtime_contract"]["required_outputs"]
    # 16 since decks/*/video_delivery_validation.json joined the contract (2026-09-27).
    assert len(required_outputs) == 16

    critical_finding_2_names = {
        "source_corpus_inventory.json",
        "speaker_author_work_ledger.json",
        "source_repair_findings.json",
        "research_cluster_plan.json",
        "artifact_truth_report.json",
    }
    covered_cf2_names: set[str] = set()

    for spec in required_outputs:
        pattern = str(spec.get("glob") or spec.get("path"))
        matches = sorted(workspace.glob(pattern))
        assert matches, pattern  # the fixture run must have produced >=1 match for every spec

        # -- snapshot every matched file's bytes (min_count is 1 per spec,
        # so leaving even one surviving copy would silently mask the
        # drop -- every copy must be removed). --
        snapshots = {path: path.read_bytes() for path in matches}
        for path in matches:
            path.unlink()

        dropped_contract = runner.verify_stage_contract(
            workspace=workspace,
            run_id=run_id,
            stage_id="presentation",
            receipt=receipt,
            mode="live",
            strict_execution=True,
        )
        assert dropped_contract.status != "pass", pattern
        assert any(
            blocker.get("code")
            in ("stage_contract_required_output_glob_missing", "stage_contract_required_output_missing")
            for blocker in dropped_contract.blockers
        ), (pattern, dropped_contract.blockers)

        matched_name = Path(pattern).name
        if matched_name in critical_finding_2_names:
            covered_cf2_names.add(matched_name)

        # -- restore, byte-identical, before moving to the next spec. --
        for path, payload in snapshots.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        restored_contract = runner.verify_stage_contract(
            workspace=workspace,
            run_id=run_id,
            stage_id="presentation",
            receipt=receipt,
            mode="live",
            strict_execution=True,
        )
        assert restored_contract.status == "pass", (pattern, restored_contract.blockers)

    # -- every one of the 5 Critical-Finding-2 outputs was genuinely
    # exercised by the drop-and-fail loop above. --
    assert covered_cf2_names == critical_finding_2_names

    # -- no forbidden_script_id ever appears anywhere in the receipt. --
    forbidden_ids = binding["forbidden_script_ids"]
    assert len(forbidden_ids) == 4
    receipt_text = json.dumps(receipt)
    for forbidden_id in forbidden_ids:
        assert forbidden_id not in receipt_text


# ---------------------------------------------------------------------------
# Phase 10.3-source-metadata-join-and-budget-correction, Plan 02 (PROD-15):
# FIX A -- the wide abstract window must be SELECTION-PRESERVING.
#
# ``quote`` is simultaneously a display field and a selector input: the shared
# ``select_topic_source_chunks`` lowercases it into its keyword-ranking term
# (source_pack_selection.py:287) and uses that same term as its admission
# filter (:315-318). Research measured that widening ``quote`` in place changes
# the selected paper set in 12 of 12 topics -- silently re-grounding every deck
# on papers the already-paid narrative analysis was never written from. The fix
# therefore emits a SEPARATE wider field that the selector never reads, and the
# two tests below pin both halves of that claim: V-10 that the narrow field is
# unchanged and the wide field is distinct and strictly wider, V-09 that the
# real selector's chosen chunk ids are byte-identical with and without it.
# ---------------------------------------------------------------------------

PREPHASE_SOURCE_PACK_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "analysis" / "cvpr2026_prephase_source_pack_chunk_ids.json"
)
LIVE_WORKSPACE = REPO_ROOT / "conferences" / "cvpr2026"
LIVE_RUN_ID = "cvpr2026-live-production"

# Anti-vacuity floor for V-09: the fraction of the with-window run's SELECTED
# chunks that must actually carry a non-empty, strictly-wider abstract window.
# Without this, a compaction that silently stopped emitting the field (or that
# aliased it to ``quote``) would make the byte-identity comparison pass
# trivially -- the two runs would be comparing the same thing. Measured at
# 144/144 (1.0) against the live corpus at the pre-phase budget; the threshold
# is set below that with margin rather than at it.
ABSTRACT_WINDOW_COVERAGE_FLOOR = 0.9


def test_abstract_window_does_not_perturb_selected_chunk_ids() -> None:
    """V-09 (Done Gate 4): selection byte-identity across the wide-window field.

    Runs the REAL shared selector twice over the REAL live vector-store
    sidecars, with identical arguments except the compaction callable -- once
    with the runner's own compaction (which emits the wide abstract window) and
    once with a wrapper that strips that field back off each compacted chunk.
    The two runs' per-topic selected ``chunk_id`` lists must be equal, and both
    must equal the committed pre-phase golden fixture in 12 of 12 topics.

    The budget is PINNED to the fixture's recorded ``budget_at_capture`` rather
    than read from today's constants. The fixture was captured at
    max_chunks=12 / paper_target=10 / main_conference_floor=10; this same plan
    legitimately moves those to 28/26/26, so selection is EXPECTED to differ at
    the new budget. Comparing post-budget selection against pre-budget packs
    would fail for the wrong reason and look like ``abstract_window`` had
    perturbed selection when it had not.

    $0: a deterministic keyword-substring scan over on-disk sidecars. No
    provider client is constructed, no credential is read, nothing is written.
    """

    from conference_pipeline.analysis import source_pack_selection

    if not PREPHASE_SOURCE_PACK_FIXTURE.exists():
        pytest.skip(f"pre-phase golden fixture absent: {PREPHASE_SOURCE_PACK_FIXTURE}")
    fixture = json.loads(PREPHASE_SOURCE_PACK_FIXTURE.read_text(encoding="utf-8"))
    expected_by_topic = fixture["topics"]
    budget = fixture["budget_at_capture"]

    if not LIVE_WORKSPACE.is_dir():
        pytest.skip(f"live workspace absent: {LIVE_WORKSPACE}")

    runner = load_runner_module()

    # -- resolve the Stage 6 receipt, index digest and topic specs exactly the
    # way scripts/regenerate_narrative_ledgers_without_synthesis.py does, so
    # this exercises the real run's inputs rather than a restatement of them.
    stage6 = runner.STAGES["corpus-inventory-index"]
    stage6_receipt_path = runner.run_dir(LIVE_WORKSPACE, LIVE_RUN_ID) / stage6.receipt_filename
    if not stage6_receipt_path.exists():
        pytest.skip(f"stage 6 receipt absent: {stage6_receipt_path}")
    stage6_receipt, stage6_error = runner._analysis_load_json(stage6_receipt_path)
    if stage6_error or not isinstance(stage6_receipt, dict):
        pytest.skip(f"stage 6 receipt unreadable: {stage6_receipt_path} ({stage6_error})")

    index_digest_path = LIVE_WORKSPACE / runner.VECTOR_STORE_DIR / LIVE_RUN_ID / "index_digest.json"
    if not index_digest_path.exists():
        pytest.skip(f"vector-store index digest absent: {index_digest_path}")
    index_digest, index_digest_error = runner._analysis_load_json(index_digest_path)
    if index_digest_error or not isinstance(index_digest, dict):
        pytest.skip(f"vector-store index digest unreadable: {index_digest_path}")

    store_chunk_paths: list[tuple[str, Path]] = []
    for store_name in ("papers", "artifacts"):
        store = (index_digest.get("stores") or {}).get(store_name)
        if not isinstance(store, dict):
            continue
        chunks_path, resolve_error = runner.platform_corpus_vector_store.resolve_workspace_member(
            LIVE_WORKSPACE, store.get("chunks_path")
        )
        if resolve_error or chunks_path is None or not chunks_path.exists():
            pytest.skip(
                f"vector-store {store_name} sidecar absent: "
                f"{chunks_path or store.get('chunks_path')}"
            )
        store_chunk_paths.append((store_name, chunks_path))
    if not store_chunk_paths:
        pytest.skip(f"no vector-store sidecars resolved from {index_digest_path}")

    topic_specs = runner.load_analysis_topic_specs(LIVE_WORKSPACE, run_id=LIVE_RUN_ID)
    priority_source_ranks = runner.production_analysis_stage.priority_source_ranks(
        LIVE_WORKSPACE, run_id=LIVE_RUN_ID, topic_specs=topic_specs
    )

    window_key = runner.ANALYSIS_COMPACT_ABSTRACT_WINDOW_KEY

    def compact_without_wide_window(chunk: object) -> dict[str, object]:
        """The runner's own compaction with the wide window removed again --
        byte-identical to the pre-phase compaction from the selector's point
        of view."""

        compact = runner._analysis_compact_chunk(chunk)
        compact.pop(window_key, None)
        return compact

    def run_selection(compact_chunk: object) -> object:
        return source_pack_selection.select_topic_source_chunks(
            store_chunk_paths=store_chunk_paths,
            topic_specs=topic_specs,
            compact_chunk=compact_chunk,
            # PINNED to the fixture's capture-time budget -- see the docstring.
            max_chunks_per_topic=budget["ANALYSIS_SOURCE_PACK_MAX_CHUNKS"],
            paper_target=budget["ANALYSIS_SOURCE_PACK_PAPER_TARGET"],
            main_conference_floor=budget["main_conference_floor"],
            priority_source_ranks=priority_source_ranks,
        )

    with_window = run_selection(runner._analysis_compact_chunk)
    without_window = run_selection(compact_without_wide_window)

    # -- 1. the wide field does not perturb selection at all. --
    for spec in topic_specs:
        topic_id = str(spec["id"])
        with_ids = [chunk["chunk_id"] for chunk in with_window.selected.get(topic_id, [])]
        without_ids = [chunk["chunk_id"] for chunk in without_window.selected.get(topic_id, [])]
        assert with_ids == without_ids, (
            f"{topic_id}: the wide abstract window changed the selected chunk ids"
        )

    # -- 2. and both reproduce the committed pre-phase selection exactly --
    # but only against the store that selection was made from. The fixture
    # records that store's index digest and must never be regenerated; once
    # the live store grows (CVPR 2026's transcript evidence lanes were indexed
    # on 2026-09-17), selection legitimately differs and this comparison would
    # fail for a reason unrelated to the abstract window. Check 1 above -- the
    # invariant V-09 exists for -- still runs against the live store.
    assert set(expected_by_topic) == {str(spec["id"]) for spec in topic_specs}
    capture_digests = {
        str(item.get("index_digest_sha256") or "")
        for item in (fixture.get("source_pack_digests") or {}).values()
    }
    live_digest = runner._file_sha256(index_digest_path)
    fixture_store_is_live = live_digest in capture_digests
    matched = 0
    for topic_id, expected_ids in expected_by_topic.items():
        if not fixture_store_is_live:
            break
        with_ids = [chunk["chunk_id"] for chunk in with_window.selected.get(topic_id, [])]
        assert with_ids == expected_ids, (
            f"{topic_id}: selection drifted from the committed pre-phase packs at the "
            f"pre-phase budget {budget}"
        )
        matched += 1
    assert matched == (12 if fixture_store_is_live else 0), matched

    # -- 3. anti-vacuity: the comparison above actually exercised the field. --
    selected_chunks = [
        chunk for spec in topic_specs for chunk in with_window.selected.get(str(spec["id"]), [])
    ]
    assert selected_chunks, "no chunks were selected at all"
    non_empty_window = sum(1 for chunk in selected_chunks if chunk.get(window_key))
    strictly_wider = sum(
        1
        for chunk in selected_chunks
        if len(str(chunk.get(window_key) or "")) > len(str(chunk.get("quote") or ""))
    )
    coverage = non_empty_window / len(selected_chunks)
    wider_coverage = strictly_wider / len(selected_chunks)
    assert coverage >= ABSTRACT_WINDOW_COVERAGE_FLOOR, (
        f"only {non_empty_window}/{len(selected_chunks)} selected chunks carried a non-empty "
        f"{window_key}; the byte-identity comparison above would have been vacuous"
    )
    assert wider_coverage >= ABSTRACT_WINDOW_COVERAGE_FLOOR, (
        f"only {strictly_wider}/{len(selected_chunks)} selected chunks carried a "
        f"{window_key} strictly wider than their quote; the field may be aliasing quote"
    )

    # -- 4. and the run never wrote anything or touched the committed packs. --
    assert not (LIVE_WORKSPACE / "analysis" / "narrative_source_packs").is_file()


def test_analysis_compact_chunk_emits_wide_abstract_window_without_widening_quote() -> None:
    """V-10 (Done Gate 4): the narrow selector-visible quote stays frozen while
    a distinct, strictly wider window is emitted alongside it.

    Both widths are read from module-level names rather than restated as bare
    numbers here, so a later width change cannot leave a stale literal behind
    in this test."""

    runner = load_runner_module()
    narrow_width = runner.ANALYSIS_COMPACT_QUOTE_CHARS
    wide_width = runner.ANALYSIS_COMPACT_ABSTRACT_WINDOW_CHARS
    window_key = runner.ANALYSIS_COMPACT_ABSTRACT_WINDOW_KEY

    assert window_key != "quote"
    assert wide_width > narrow_width

    # -- case 1: text far longer than both windows. --
    long_chunk = {
        "chunk_id": "paper:abcdef:0",
        "artifact_id": "paper_abstract:abcdef",
        "source_path": "storage/download-corpus/artifacts/paper_pdf/abcdef.pdf",
        "text": "evidence " * (wide_width * 2),
    }
    compact = runner._analysis_compact_chunk(long_chunk)

    quote = compact["quote"]
    window = compact[window_key]
    # The narrow field's width is UNCHANGED: this is the assertion that fails
    # the moment somebody "just widens quote" instead of adding a field.
    assert len(quote) == narrow_width
    # The wide field is present, bounded by its own width, and strictly wider.
    assert len(window) == wide_width
    assert len(window) > len(quote)
    # It is a genuine extension of the same collapsed text, not a re-derivation.
    assert window[: len(quote)] == quote
    # And it is a distinct key, not an alias of the narrow one.
    assert window_key in compact
    assert compact[window_key] is not compact["quote"]
    assert compact[window_key] != compact["quote"]

    # -- case 2: text shorter than the narrow window -- identical, no padding. --
    short_text = "A single short abstract sentence."
    short_chunk = dict(long_chunk, text=short_text)
    short_compact = runner._analysis_compact_chunk(short_chunk)
    assert short_compact["quote"] == short_text
    assert short_compact[window_key] == short_text
    assert len(short_compact["quote"]) == len(short_text)
    assert len(short_compact[window_key]) == len(short_text)

    # -- case 3: empty text -- both empty, no exception. --
    empty_compact = runner._analysis_compact_chunk(dict(long_chunk, text=""))
    assert empty_compact["quote"] == ""
    assert empty_compact[window_key] == ""

    # -- the additive field did not disturb the structured metadata the
    # Phase 10.1 Finding-A fix preserves through this same compaction. --
    metadata_chunk = dict(
        long_chunk,
        paper_title="A Structured Title That Must Survive Compaction",
        authors=["Ada Lovelace", "Alan Turing"],
    )
    metadata_compact = runner._analysis_compact_chunk(metadata_chunk)
    assert metadata_compact["paper_title"] == "A Structured Title That Must Survive Compaction"
    assert metadata_compact["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert metadata_compact[window_key]


def test_an_executive_deck_is_given_the_topic_reference_ledgers(tmp_path: Path) -> None:
    """Two markdown files cannot feed the ledger, so an executive deck had zero
    source candidates and the chain raised five steps before the blueprint. No
    executive deck had ever been produced through this path, for any conference.
    """

    module = load_runner_module()
    ws = tmp_path / "conf"
    (ws / "analysis" / "topic_reference_ledgers").mkdir(parents=True)
    (ws / "analysis" / "executive_summary.md").write_text("summary", encoding="utf-8")
    (ws / "analysis" / "trend_report.md").write_text("trends", encoding="utf-8")
    for topic in ("01_alpha", "02_beta"):
        (ws / "analysis" / "topic_reference_ledgers" / f"{topic}.json").write_text(
            json.dumps({"topic_id": topic, "entries": [{"canonical_id": f"{topic}-1"}]}),
            encoding="utf-8",
        )

    inputs = production_presentation_steps.corpus_inventory_source_inputs(
        module._production_context(ws, "source-inputs"), "executive_summary_long", "executive"
    )
    assert "analysis/executive_summary.md" in inputs
    assert "analysis/trend_report.md" in inputs
    # The harvester keys on "topic_reference_ledgers" being in the path, so
    # these are what actually produce ledger entries.
    ledgers = [path for path in inputs if "topic_reference_ledgers" in path]
    assert len(ledgers) == 2, inputs
    assert ledgers == sorted(ledgers), "deterministic order"

    # A topic deck's own discovery is untouched.
    (ws / "analysis" / "07_gamma.md").write_text("topic", encoding="utf-8")
    topic_inputs = production_presentation_steps.corpus_inventory_source_inputs(
        module._production_context(ws, "source-inputs"), "07_gamma", "topic"
    )
    assert not [path for path in topic_inputs if "topic_reference_ledgers" in path]


def test_failed_presentation_review_is_superseded_not_reused(tmp_path: Path) -> None:
    """A failed Gate A review is a verdict to re-evaluate, not an output to
    resume past. The existence-only step skip would otherwise carry a failed
    review forward as if it had passed."""

    runner = load_runner_module()
    report = tmp_path / "presentation_review.json"

    report.write_text(json.dumps({"status": "failed", "production_ready": False}), encoding="utf-8")
    moved = runner._presentation_supersede_failed_review(report)
    assert moved is not None and moved.is_file() and not report.exists()
    assert ".superseded-" in moved.name

    report.write_text(json.dumps({"status": "passed", "production_ready": True}), encoding="utf-8")
    assert runner._presentation_supersede_failed_review(report) is None
    assert report.is_file()

    assert runner._presentation_supersede_failed_review(tmp_path / "absent.json") is None


def test_operator_accepted_failed_review_is_recorded_verbatim_only_for_named_decks(tmp_path: Path) -> None:
    runner = load_runner_module()
    report = tmp_path / "presentation_review.json"
    blocker = {"category": "attribution", "slide": 20, "comment": "chip not shown"}
    report.write_text(
        json.dumps({"status": "failed", "aggregated_feedback": {"blockers": [blocker]}}), encoding="utf-8"
    )
    accepted = frozenset({"04_deck"})

    record = runner._presentation_accepted_review(report, deck_id="04_deck", accepted_deck_ids=accepted)
    assert record["accepted_by"] == "operator:--accept-presentation-review-findings"
    assert record["status"] == "failed" and record["accepted_blockers"] == [blocker]
    assert record["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()

    assert runner._presentation_accepted_review(report, deck_id="05_other", accepted_deck_ids=accepted) is None
    report.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    assert runner._presentation_accepted_review(report, deck_id="04_deck", accepted_deck_ids=accepted) is None
    assert runner._presentation_accepted_review(tmp_path / "absent.json", deck_id="04_deck", accepted_deck_ids=accepted) is None


# ---------------------------------------------------------------------------
# Phase 10.4: per-deck NotebookLM podcast stage (PROD-16)
# ---------------------------------------------------------------------------


def _passing_transcriber():
    from tests.test_podcast_notebooklm import passing_transcriber

    return passing_transcriber()


def _notebooklm_workspace(tmp_path: Path) -> Path:
    from tests.test_podcast_notebooklm import make_workspace as _make_notebooklm_workspace

    return _make_notebooklm_workspace(tmp_path, with_executive=True, with_executive_long=True)


def _seed_notebooklm_predecessors(runner: object, workspace: Path, run_id: str) -> None:
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / runner.STAGES["qa-gap-fill-synthesis"].receipt_filename).write_text(
        json.dumps({"stage_id": "qa-gap-fill-synthesis", "status": "succeeded", "mode": "live",
                    "checks": {"executive_summary_status": "pass", "trend_report_status": "pass"}}),
        encoding="utf-8",
    )
    (run_dir / runner.STAGES["presentation"].receipt_filename).write_text(
        json.dumps({"stage_id": "presentation", "status": "succeeded", "mode": "live", "stage_contract": {"status": "pass"}}),
        encoding="utf-8",
    )
    stamp_upstream_stage_receipts(workspace / "storage" / "workflow-runs" / run_id)


def test_notebooklm_podcast_stage_registered_and_dispatches_from_main() -> None:
    runner = load_runner_module()
    stage = runner.STAGES["notebooklm-podcast"]
    assert stage.order == 10
    assert stage.receipt_filename == "stage-10-notebooklm-podcast.json"
    assert stage.implemented is True
    assert runner.orchestrator_spec().handlers["notebooklm-podcast"] is runner.dispatch_notebooklm_podcast
    assert runner.NOTEBOOKLM_PODCAST_TASK_ID == "podcast.notebooklm_audio_overview"


def test_notebooklm_podcast_blocks_on_missing_inputs_without_network(tmp_path: Path) -> None:
    from conference_pipeline import podcast_notebooklm as pn

    runner = load_runner_module()
    workspace = _notebooklm_workspace(tmp_path)
    engine = pn.FakeNotebookLMEngine()
    outcome = runner.run_notebooklm_podcast_stage(
        workspace=workspace, run_id="nb-missing", ack_live_notebooklm=True, engine=engine, transcriber=_passing_transcriber()
    )
    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["blockers"][0]["code"] == "notebooklm_podcast_input_receipt_missing"
    assert engine.calls == []


def test_notebooklm_podcast_blocks_without_live_ack(tmp_path: Path) -> None:
    from conference_pipeline import podcast_notebooklm as pn

    runner = load_runner_module()
    workspace = _notebooklm_workspace(tmp_path)
    run_id = "nb-no-ack"
    _seed_notebooklm_predecessors(runner, workspace, run_id)
    engine = pn.FakeNotebookLMEngine()
    outcome = runner.run_notebooklm_podcast_stage(workspace=workspace, run_id=run_id, engine=engine)
    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["blockers"][0]["code"] == "notebooklm_live_ack_required"
    assert engine.calls == []
    assert outcome["receipt"]["provider_calls_made"] is False


def test_notebooklm_podcast_deck_ids_canary_is_always_blocked(tmp_path: Path) -> None:
    from conference_pipeline import podcast_notebooklm as pn

    runner = load_runner_module()
    workspace = _notebooklm_workspace(tmp_path)
    run_id = "nb-canary"
    _seed_notebooklm_predecessors(runner, workspace, run_id)
    engine = pn.FakeNotebookLMEngine()
    outcome = runner.run_notebooklm_podcast_stage(
        workspace=workspace, run_id=run_id, deck_ids=["01_alpha_topic"], ack_live_notebooklm=True, engine=engine, transcriber=_passing_transcriber()
    )
    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["blockers"][0]["code"] == "notebooklm_podcast_deck_scope_filtered"
    assert outcome["receipt"]["blockers"][0]["full_scope_deck_count"] == 4
    assert (workspace / "audio" / "01_alpha_topic" / "notebooklm_podcast" / "podcast.m4a").is_file()
    assert engine.generate_calls == 1
    assert outcome["receipt"]["stage_contract"] is None


def test_notebooklm_podcast_fake_engine_end_to_end_satisfies_runtime_contract(tmp_path: Path) -> None:
    """VALIDATION-ONLY: the deterministic fake engine proves the receipt shape
    satisfies the registry contract with zero network calls; it can never
    substitute for the live stage-10-notebooklm-podcast.json receipt."""

    from conference_pipeline import podcast_notebooklm as pn

    runner = load_runner_module()
    workspace = _notebooklm_workspace(tmp_path)
    run_id = "nb-full"
    _seed_notebooklm_predecessors(runner, workspace, run_id)
    engine = pn.FakeNotebookLMEngine()
    outcome = runner.run_notebooklm_podcast_stage(
        workspace=workspace, run_id=run_id, ack_live_notebooklm=True, engine=engine, transcriber=_passing_transcriber()
    )
    receipt = outcome["receipt"]
    assert outcome["status"] == "succeeded", receipt["blockers"]
    checks = receipt["checks"]
    assert checks["deck_success_count"] == 4 == checks["deck_scope_full_count"]
    assert checks["deck_blocker_count"] == 0
    assert checks["notebooklm_podcast_status"] == "pass"
    assert checks["notebooklm_podcast_aggregate_sha256"]
    assert checks["storage_state_present"] is True and checks["engine_version"] == "fake"
    assert checks["paid_provider_calls_made"] is False and checks["estimated_cost_usd"] == 0.0
    assert receipt["provider_calls_made"] is True
    assert receipt["stage_contract"]["status"] == "pass", receipt["stage_contract"]["blockers"]
    assert set(checks["prompt_variants"].values()) == {"long"}

    required = list(stage_script_registry.required_script_ids_for_stage(load_stage_script_registry(), "notebooklm-podcast"))
    assert required == ["podcast.notebooklm_audio_overview"]
    token = receipt["owner_execution_evidence"]["podcast.notebooklm_audio_overview"]
    assert token.endswith(checks["notebooklm_podcast_aggregate_sha256"]) and token != required[0]

    independent = stage_contracts.verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="notebooklm-podcast", receipt=receipt, mode="live", strict_execution=True
    ).to_dict()
    assert independent["status"] == "pass", independent["blockers"]
    mutated = copy.deepcopy(receipt)
    mutated["owner_execution_evidence"] = {owner_id: owner_id for owner_id in required}
    assert stage_contracts.verify_stage_contract(
        workspace=workspace, run_id=run_id, stage_id="notebooklm-podcast", receipt=mutated, mode="live", strict_execution=True
    ).to_dict()["status"] != "pass"

    resumed = runner.run_notebooklm_podcast_stage(
        workspace=workspace, run_id=run_id, ack_live_notebooklm=True, engine=pn.FakeNotebookLMEngine(preflight_ok=False), transcriber=_passing_transcriber()
    )
    assert resumed["status"] == "succeeded" and resumed["summary"].get("resumed") is True

    summary = json.loads((workspace / "storage" / "workflow-runs" / run_id / "summary.json").read_text(encoding="utf-8"))
    assert [s["stage_id"] for s in summary["stages"]] == ["notebooklm-podcast"]


def test_notebooklm_podcast_quota_halt_is_blocked_and_retryable(tmp_path: Path) -> None:
    from conference_pipeline import podcast_notebooklm as pn

    runner = load_runner_module()
    workspace = _notebooklm_workspace(tmp_path)
    run_id = "nb-quota"
    _seed_notebooklm_predecessors(runner, workspace, run_id)
    outcome = runner.run_notebooklm_podcast_stage(
        workspace=workspace, run_id=run_id, ack_live_notebooklm=True, engine=pn.FakeNotebookLMEngine(fail_quota_on_call=2), transcriber=_passing_transcriber()
    )
    assert outcome["status"] == "blocked"
    blocker = outcome["receipt"]["blockers"][0]
    assert blocker["code"] == "notebooklm_quota_exhausted" and blocker["retryable"] is True
    assert outcome["receipt"]["checks"]["deck_success_count"] == 1

    auth = runner.run_notebooklm_podcast_stage(
        workspace=workspace, run_id=run_id, ack_live_notebooklm=True, engine=pn.FakeNotebookLMEngine(preflight_ok=False), transcriber=_passing_transcriber()
    )
    assert auth["receipt"]["blockers"][0]["code"] == "notebooklm_auth_preflight_failed"
    assert auth["receipt"]["checks"]["storage_state_present"] is False


def test_release_deliverables_and_proof_candidates_include_notebooklm_podcasts(tmp_path: Path) -> None:
    runner = load_runner_module()
    items = production_release_stage.release_deliverables(
        runner._production_context(tmp_path, "release-deliverables-preview"), ["01_alpha_topic"]
    )
    podcast_rows = [row for row in items if row["produced_by"] == "notebooklm-podcast"]
    assert [row["kind"] for row in podcast_rows] == ["podcast", "manifest"]
    assert podcast_rows[0]["path"] == "audio/01_alpha_topic/notebooklm_podcast/podcast.m4a"
    assert any(
        candidate["requires_stages"] == ["notebooklm-podcast"] for candidate in runner.PROOF_SUPPORT_LABEL_CANDIDATES
    )


def test_presentation_full_substage_blocks_when_a_deck_fails_gate_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full substage must block on a per-deck blocker exactly as the
    deck_figures substage does. On 2026-09-16 a full run with two decks
    refused by the Gate A reviewers returned ``succeeded`` -- the failures
    were listed under ``checks["deck_blockers"]`` and nothing looked at them
    -- which would have let Stage 11 inventory decks the reviewers rejected."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-full-substage-gate-a"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )

    failing_deck_id = "05_fixture_topic"
    call_log: list[str] = []
    base_runner = _build_fake_presentation_command_runner(call_log=call_log)

    def gate_a_refuses_one_deck(command: list[str]) -> subprocess.CompletedProcess[str]:
        if _presentation_fixture_arg(command, "--presentation-id") == failing_deck_id and any(
            str(part).endswith("review_presentation.py") for part in command
        ):
            report_path = _presentation_fixture_arg(command, "--report-output")
            assert report_path
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(
                json.dumps({"status": "failed", "production_ready": False}), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 1, "", "reviewers refused the deck")
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        presentation_substage="all",
        command_runner=gate_a_refuses_one_deck,
    )

    receipt = outcome["receipt"]
    assert receipt["status"] == "blocked", receipt["status"]
    assert receipt["blockers"][0]["code"] == "presentation_partial_deck_scope_blocked"
    assert receipt["blockers"][0]["deck_ids"] == [failing_deck_id]
    (deck_blocker,) = receipt["checks"]["deck_blockers"]
    assert deck_blocker["deck_id"] == failing_deck_id
    assert deck_blocker["blocker_code"] == "presentation_review_failed"
    assert receipt["checks"]["deck_blocker_count"] == 1
    assert receipt["checks"]["deck_success_count"] == 13
    # The other decks were still carried through to the end of the chain.
    assert call_log.count("validate_video_sync.py") == _PER_DECK_CHAIN_DECKS - 1


def test_corpus_inventory_index_admits_registered_transcript_evidence_lanes(tmp_path: Path) -> None:
    """A registered, digest-bound evidence lane fills an r5 caption gap without
    touching the r5 inventory: the lane transcript becomes a speech_transcript
    record, the gap leaves caption_coverage_facts, a base-duplicated video id
    is superseded (r5 wins), and the receipt binding names the lane digest."""

    from conference_pipeline.indexing import transcript_evidence_lanes as lanes

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-evidence-lanes"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)

    recovered = workspace / "storage" / "transcription-ocr" / "late-recovery" / "fixture-unavailable-0001.en.vtt"
    recovered.parent.mkdir(parents=True, exist_ok=True)
    recovered.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n" + ("Recovered keynote on molecular design. " * 10),
        encoding="utf-8",
    )
    duplicate = lanes.write_plain_transcript(workspace, "late-captions", "fixture-video-0001", "asr duplicate of base")
    receipt, blockers = lanes.build_lane_receipt(
        workspace=workspace,
        lane_id="late-captions",
        conference="fixture",
        producer="test",
        transcript_source="youtube_caption_auto",
        records=[
            {
                "video_id": "fixture-unavailable-0001",
                "transcript_path": str(recovered.relative_to(workspace)),
                "title": "Recovered keynote",
            },
            {"video_id": "fixture-video-0001", "transcript_path": str(duplicate.relative_to(workspace))},
        ],
    )
    assert blockers == [], blockers
    lanes.register_lane(workspace=workspace, lane_id="late-captions")

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    inventory = json.loads((workspace / "storage" / "corpus" / "corpus_inventory.json").read_text(encoding="utf-8"))
    speech = [r for r in inventory["artifact_records"] if r["source_family_id"] == "speech_transcript"]
    assert {r["video_id"] for r in speech} == {"fixture-video-0001", "fixture-unavailable-0001"}
    lane_record = next(r for r in speech if r["video_id"] == "fixture-unavailable-0001")
    assert lane_record["artifact_id"] == "event_media_caption:late-captions:fixture-unavailable-0001"
    assert lane_record["evidence_lane_id"] == "late-captions"
    base_record = next(r for r in speech if r["video_id"] == "fixture-video-0001")
    assert "evidence_lane_id" not in base_record  # r5 record kept, lane duplicate superseded
    # The base caption carries its event title, and its chunks embed that title
    # with timed, markup-free caption text rather than raw WebVTT.
    assert base_record["title"] == "Fixture poster 1"
    chunks_path = workspace / "storage" / "vector-store" / run_id / "stores" / "artifacts" / "chunks.jsonl"
    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    base_rows = [row for row in rows if row["artifact_id"] == "event_media_caption:fixture-video-0001"]
    assert base_rows
    for row in base_rows:
        assert "Presentation: Fixture poster 1" in row["text"]
        assert "WEBVTT" not in row["text"] and "-->" not in row["text"]
        assert row["indexed_text_kind"] == "caption_segments_json"
        assert row["locator"]["kind"] == "transcript_time_range"
    gaps = {fact["video_id"] for fact in inventory["caption_coverage_facts"]}
    assert "fixture-unavailable-0001" not in gaps
    assert {"fixture-unavailable-0000", "fixture-unavailable-0002"} <= gaps
    assert inventory["artifact_counts"]["evidence_lane_transcripts"] == 1
    binding = inventory["upstream"]["transcript_evidence_lanes"]
    assert binding[0]["lane_id"] == "late-captions"
    assert binding[0]["admitted_count"] == 1 and binding[0]["superseded_count"] == 1
    assert binding[0]["receipt_sha256"] == receipt_sha256_of(lanes.lane_receipt_path(workspace, "late-captions"))
    assert outcome["receipt"]["checks"]["transcript_evidence_lanes"] == binding


def receipt_sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_corpus_inventory_index_blocks_on_tampered_evidence_lane(tmp_path: Path) -> None:
    """A registered lane whose transcript drifted from its bound digest blocks
    the stage (fail closed) rather than being silently skipped."""

    from conference_pipeline.indexing import transcript_evidence_lanes as lanes

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-evidence-lane-tamper"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)
    transcript = lanes.write_plain_transcript(workspace, "asr", "fixture-unavailable-0002", "bound text")
    _receipt, blockers = lanes.build_lane_receipt(
        workspace=workspace, lane_id="asr", conference="fixture", producer="test", transcript_source="asr_fallback",
        records=[{"video_id": "fixture-unavailable-0002", "transcript_path": str(transcript.relative_to(workspace))}],
    )
    assert blockers == []
    lanes.register_lane(workspace=workspace, lane_id="asr")
    transcript.write_text("drifted after binding", encoding="utf-8")

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "blocked"
    codes = {blocker["code"] for blocker in outcome["receipt"]["blockers"]}
    assert "evidence_lane_transcript_digest_mismatch" in codes


def test_corpus_inventory_index_blocks_on_untitled_transcript(tmp_path: Path) -> None:
    """Transcript chunks are embedded with their title prefix; an admitted
    transcript without a title blocks the stage instead of indexing bare speech."""

    from conference_pipeline.indexing import transcript_evidence_lanes as lanes

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-untitled-transcript"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)
    transcript = lanes.write_plain_transcript(workspace, "asr", "fixture-unavailable-0002", "bound text " * 20)
    _receipt, blockers = lanes.build_lane_receipt(
        workspace=workspace,
        lane_id="asr",
        conference="fixture",
        producer="test",
        transcript_source="asr_fallback",
        records=[{"video_id": "fixture-unavailable-0002", "transcript_path": str(transcript.relative_to(workspace))}],
    )
    assert blockers == []
    lanes.register_lane(workspace=workspace, lane_id="asr")

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "blocked"
    codes = {blocker["code"] for blocker in outcome["receipt"]["blockers"]}
    assert codes == {"speech_transcript_title_missing"}


def test_corpus_inventory_index_admits_registered_proceedings_track(tmp_path: Path) -> None:
    """A registered proceedings track (e.g. CVF Findings) is admitted as
    main-conference papers carrying venue and venue_track, digest-checked."""

    from conference_pipeline.acquisition import proceedings_paper_sources as proceedings

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-proceedings"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)
    pdf = proceedings.source_dir(workspace, "cvf_findings") / "cvf_abc.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    body = _write_extractable_pdf(pdf, "A findings paper about robust depth estimation. " * 60)
    receipt = {
        "receipt_type": proceedings.DOWNLOAD_RECEIPT_TYPE,
        "status": "succeeded",
        "source_id": "cvf_findings",
        "track": "findings",
        "venue": "CVPR 2026 Findings",
        "papers": [
            {
                "paper_id": "cvf_abc",
                "title": "Robust Depth Findings",
                "authors": ["Ada Lovelace"],
                "pdf_url": "https://openaccess.thecvf.com/content/x.pdf",
                "pdf_path": str(pdf.relative_to(workspace)),
                "status": "succeeded",
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ],
    }
    proceedings.receipt_path(workspace, "cvf_findings").write_text(json.dumps(receipt), encoding="utf-8")
    proceedings.register_source(workspace=workspace, source_id="cvf_findings")

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    inventory = json.loads((workspace / "storage" / "corpus" / "corpus_inventory.json").read_text(encoding="utf-8"))
    record = next(r for r in inventory["artifact_records"] if r["artifact_id"] == "proceedings_paper:cvf_findings:cvf_abc")
    assert record["source_family_id"] == "paper_text" and record["venue_track"] == "findings"
    assert record["authors"] == ["Ada Lovelace"]
    assert record["source_metadata_provenance"]["authors"] == "proceedings_listing"
    assert inventory["upstream"]["proceedings_paper_sources"][0]["admitted_count"] == 1
    rows = [
        json.loads(line)
        for line in (workspace / "storage" / "vector-store" / run_id / "stores" / "papers" / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    mine = [row for row in rows if row["artifact_id"] == record["artifact_id"]]
    assert mine and all("Venue: CVPR 2026 Findings" in row["text"] for row in mine)
    assert mine[0]["source_context"]["conference_scope"] == "main_conference"
    assert mine[0]["source_context"]["venue_track"] == "findings"


def test_corpus_inventory_index_excludes_image_only_papers_as_audit_only(tmp_path: Path) -> None:
    """Stage 6 runs paper curation: a watermark-only PDF stays in the inventory
    as an audit-only record with its reason, and never reaches the index."""

    from conference_pipeline.acquisition import proceedings_paper_sources as proceedings

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-corpus-inventory-curation"
    _write_corpus_inventory_index_predecessors(runner, workspace, run_id)
    runner.PAPER_CURATION_MIN_TEXT_BYTES = 2000
    papers = []
    for paper_id, text in (
        ("cvf_text", "A findings paper about robust depth estimation. " * 60),
        ("cvf_scan", "This CVPR Findings paper is the Open Access version. "),
    ):
        pdf = proceedings.source_dir(workspace, "cvf_findings") / f"{paper_id}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        body = _write_extractable_pdf(pdf, text)
        papers.append(
            {
                "paper_id": paper_id,
                "title": f"Paper {paper_id}",
                "authors": ["Ada Lovelace"],
                "pdf_url": f"https://openaccess.thecvf.com/content/{paper_id}.pdf",
                "pdf_path": str(pdf.relative_to(workspace)),
                "status": "succeeded",
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    receipt = {
        "receipt_type": proceedings.DOWNLOAD_RECEIPT_TYPE,
        "status": "succeeded",
        "source_id": "cvf_findings",
        "track": "findings",
        "venue": "CVPR 2026 Findings",
        "papers": papers,
    }
    proceedings.receipt_path(workspace, "cvf_findings").write_text(json.dumps(receipt), encoding="utf-8")
    proceedings.register_source(workspace=workspace, source_id="cvf_findings")

    outcome = runner.run_corpus_inventory_index_stage(workspace=workspace, run_id=run_id, resume=True)

    assert outcome["status"] == "succeeded", outcome["receipt"]["blockers"]
    inventory = json.loads((workspace / "storage" / "corpus" / "corpus_inventory.json").read_text(encoding="utf-8"))
    records = {r["artifact_id"]: r for r in inventory["artifact_records"]}
    scanned = records["proceedings_paper:cvf_findings:cvf_scan"]
    assert scanned["indexable_text"] is False
    assert scanned["index_exclusion"]["reason"] == "image_only_text_layer"
    assert records["proceedings_paper:cvf_findings:cvf_text"]["indexable_text"] is True
    assert inventory["paper_curation"]["excluded_counts"]["image_only_text_layer"] >= 1
    chunks = (workspace / "storage" / "vector-store" / run_id / "stores" / "papers" / "chunks.jsonl").read_text(
        encoding="utf-8"
    )
    assert "cvf_scan" not in chunks and "cvf_text" in chunks


# ---------------------------------------------------------------------------
# v5.1 receipt lineage: per-deck step ledger (decks/<id>/step_lineage.json)
# ---------------------------------------------------------------------------


def _run_presentation_stage_with_fixtures(
    runner: object,
    workspace: Path,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
    **kwargs: object,
) -> dict[str, object]:
    monkeypatch.setattr(
        runner.presentation_paper_matching,
        "run_paper_matching_stage",
        _fake_presentation_paper_matching_stage,
    )
    return runner.run_presentation_stage(
        workspace=workspace,
        run_id=run_id,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
        command_runner=command_runner,
        **kwargs,
    )


def _deck_scripts(call_log: list[list[str]], deck_id: str) -> list[str]:
    return [
        Path(next(part for part in command if str(part).endswith(".py"))).name
        for command in call_log
        if any(str(part) == deck_id for part in command)
        or any(
            f"/decks/{deck_id}/" in str(part) or str(part).rstrip("/").endswith(f"/decks/{deck_id}")
            for part in command
        )
    ]


def test_presentation_step_lineage_receipt_records_per_deck_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every deck gets a ledger; the receipt carries each deck's output digest
    and one stage-level digest over all of them, re-derivable from disk."""

    from conference_pipeline.presentation.step_lineage import (
        DeckStepLineage,
        presentation_output_digest,
    )

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-step-lineage-receipt"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    outcome = _run_presentation_stage_with_fixtures(
        runner, workspace, run_id, monkeypatch, command_runner=_build_fake_presentation_command_runner()
    )
    assert outcome["receipt"]["status"] == "succeeded", outcome["receipt"]
    checks = outcome["receipt"]["checks"]
    step_lineage = checks["step_lineage"]
    assert len(step_lineage) == 14
    for deck_id, entry in step_lineage.items():
        ledger_path = workspace / "decks" / deck_id / "step_lineage.json"
        assert ledger_path.is_file(), deck_id
        assert entry["ledger"].endswith(f"decks/{deck_id}/step_lineage.json")
        assert entry["output_digest"] and entry["steps_run"]
    topic_entry = step_lineage["01_fixture_topic"]
    assert "generate_editorial_blueprint" in topic_entry["steps_run"]
    assert "build_presenter_video" in topic_entry["steps_run"]
    recomputed = presentation_output_digest(
        {
            deck_id: DeckStepLineage.for_deck_dir(workspace / "decks" / deck_id, workspace=workspace)
            for deck_id in step_lineage
        }
    )
    assert checks["step_lineage_output_digest"] == recomputed


def test_presentation_engaging_narration_raw_change_reruns_delivery_review_and_video_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patching ``engaging_narration_raw.json`` (the narration source of truth)
    is adopted -- the paid narration generator is NOT re-run -- while every
    step that consumed the old narration (delivery rebuild, hash reconcile,
    Gate A review, both videos, sync validation) is; slides, figures and the
    blueprint are untouched."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-narration-raw-change"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    base_runner = _build_fake_presentation_command_runner()

    def rebuild_aware_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        # The real rebuild carries the raw narration into both delivery
        # contracts; the table fixture leaves them alone, so mirror that here.
        result = base_runner(command)
        if any(str(part).endswith("rebuild_delivery_engaging.py") for part in command):
            deck_dir = Path(_presentation_fixture_arg(command, "--deck-dir") or "")
            raw = json.loads((deck_dir / "engaging_narration_raw.json").read_text(encoding="utf-8"))
            for name in ("delivery_single_speaker.json", "delivery_notebooklm.json"):
                payload = json.loads((deck_dir / name).read_text(encoding="utf-8"))
                payload["narration_source_digest"] = json.dumps(raw, sort_keys=True)
                (deck_dir / name).write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return result

    first = _run_presentation_stage_with_fixtures(
        runner, workspace, run_id, monkeypatch, command_runner=rebuild_aware_runner
    )
    assert first["receipt"]["status"] == "succeeded", first["receipt"]

    raw_path = workspace / "decks" / "01_fixture_topic" / "engaging_narration_raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["slides"][0]["single_speaker"] = "Operator-patched narration line."
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    call_log: list[list[str]] = []

    def recording_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        call_log.append(list(command))
        return rebuild_aware_runner(command)

    def exploding_paper_matching(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("paper matching must not re-run for a narration edit")

    monkeypatch.setattr(runner.presentation_paper_matching, "run_paper_matching_stage", exploding_paper_matching)
    second = runner.run_presentation_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None,
        command_runner=recording_runner,
    )
    assert second["receipt"]["status"] == "succeeded", second["receipt"]

    invoked = _deck_scripts(call_log, "01_fixture_topic")
    assert "rebuild_delivery_engaging.py" in invoked
    assert "reconcile_contract_hashes.py" in invoked
    assert "review_presentation.py" in invoked
    assert "build_presenter_video.py" in invoked
    assert "build_notebooklm_style_presenter_video.py" in invoked
    assert "validate_video_sync.py" in invoked
    # The paid generator must not re-run; its $0 render/sync modes may.
    assert not [
        c for c in call_log
        if any(str(p).endswith("generate_engaging_narration.py") for p in c)
        and not {"--render-from-raw", "--sync-raw-from-md"} & {str(p) for p in c}
    ], call_log
    for untouched in (
        "generate_editorial_blueprint.py",
        "generate_presenter_script.py",
        "md_to_revealjs.py",
        "export_revealjs_pdf.py",
        "generate_programmatic_figures.py",
        "lint_narration_script.py",
    ):
        assert untouched not in invoked, invoked
    # No other deck ran anything.
    assert not [c for c in call_log if not any("01_fixture_topic" in str(p) for p in c) and not any(str(p).endswith("audit_presentation_links.py") for p in c)]
    ledger = json.loads((workspace / "decks" / "01_fixture_topic" / "step_lineage.json").read_text(encoding="utf-8"))
    raw_key = "ws:decks/01_fixture_topic/engaging_narration_raw.json"
    assert ledger["paths"][raw_key]["history"][-1]["adopted_operator_edit"] is True
    assert ledger["steps"]["build_presenter_video"]["last_decision"].startswith("input_changed:")


def test_presentation_blueprint_hand_edit_is_adopted_and_reruns_its_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-blueprint-edit"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    first = _run_presentation_stage_with_fixtures(
        runner, workspace, run_id, monkeypatch, command_runner=_build_fake_presentation_command_runner()
    )
    assert first["receipt"]["status"] == "succeeded", first["receipt"]

    blueprint_path = workspace / "decks" / "01_fixture_topic" / "presentation_blueprint.json"
    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    blueprint["revision"] = 2
    blueprint_path.write_text(json.dumps(blueprint, indent=2) + "\n", encoding="utf-8")

    call_log: list[list[str]] = []
    base_runner = _build_fake_presentation_command_runner()

    def recording_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        call_log.append(list(command))
        return base_runner(command)

    second = _run_presentation_stage_with_fixtures(
        runner, workspace, run_id, monkeypatch, command_runner=recording_runner
    )
    assert second["receipt"]["status"] == "succeeded", second["receipt"]
    invoked = _deck_scripts(call_log, "01_fixture_topic")
    assert "generate_editorial_blueprint.py" not in invoked, invoked
    assert "review_presentation_blueprint.py" in invoked
    assert "generate_deck_contract.py" in invoked
    assert "build_presentation_source_corpus_inventory.py" not in invoked


def test_presentation_adopt_existing_outputs_bootstraps_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deck built before the ledger existed re-runs everything on resume
    unless ``adopt_existing_outputs`` records it from disk first."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-adopt-existing"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    first = _run_presentation_stage_with_fixtures(
        runner, workspace, run_id, monkeypatch, command_runner=_build_fake_presentation_command_runner()
    )
    assert first["receipt"]["status"] == "succeeded", first["receipt"]
    ledgers = list((workspace / "decks").glob("**/step_lineage.json"))
    assert len(ledgers) == 15  # 14 decks + the decks-directory ledger of the link audit
    for ledger in ledgers:
        ledger.unlink()

    def exploding_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"no subprocess may run when outputs are adopted: {command}")

    # Without adoption the first step to be reached re-runs (and explodes).
    with pytest.raises(AssertionError, match="no subprocess may run"):
        _run_presentation_stage_with_fixtures(
            runner, workspace, run_id, monkeypatch, command_runner=exploding_runner
        )
    # With adoption every step is recorded from disk and nothing re-runs.
    adopted = _run_presentation_stage_with_fixtures(
        runner, workspace, run_id, monkeypatch, command_runner=exploding_runner, adopt_existing_outputs=True
    )
    assert adopted["receipt"]["status"] == "succeeded", adopted["receipt"]
    entry = adopted["receipt"]["checks"]["step_lineage"]["01_fixture_topic"]
    assert "generate_editorial_blueprint" in entry["steps_adopted"]


# ---------------------------------------------------------------------------
# v5.1 receipt lineage: stage-level resume fails closed on changed upstreams
# ---------------------------------------------------------------------------


def _mutate_receipt(path: Path, **fields: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in fields.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _counting_stage8_client_class(runner: object) -> tuple[type, list[int]]:
    base = _build_fixture_stage8_synthesis_client_class(runner)
    constructed: list[int] = []

    class _CountingClient(base):  # type: ignore[misc,valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructed.append(1)
            super().__init__(*args, **kwargs)

    return _CountingClient, constructed


def test_qa_gap_fill_synthesis_resume_reruns_when_stage7_output_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counterpart to the resume-skips test: when Stage 7's output_digest moved,
    the paid client IS constructed again, the receipt is new, and the prior
    receipt is preserved beside it."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-qa-resume-upstream-changed"
    _write_qa_gap_fill_synthesis_predecessors(runner, workspace, run_id, topic_count=3)
    monkeypatch.setattr(runner, "Stage8SynthesisClient", _build_fixture_stage8_synthesis_client_class(runner))
    first = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )
    assert first["status"] == "succeeded", first["receipt"]["blockers"]
    receipt = first["receipt"]
    assert receipt["output_digest"] and receipt["lineage"]["lineage_digest"]
    stage7_key = f"storage/workflow-runs/{run_id}/stage-07-analysis.json"
    assert receipt["lineage"]["upstream_receipts"][stage7_key]["digest_kind"] == "output_digest"

    run_dir = workspace / "storage" / "workflow-runs" / run_id
    _mutate_receipt(run_dir / "stage-07-analysis.json", output_digest="1" * 64)
    counting, constructed = _counting_stage8_client_class(runner)
    monkeypatch.setattr(runner, "Stage8SynthesisClient", counting)
    second = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )
    assert second["status"] == "succeeded", second["receipt"]["blockers"]
    assert constructed, "the stage must re-run (and re-spend) when its upstream output changed"
    assert second["receipt"]["lineage"]["upstream_receipts"][stage7_key]["digest"] == "1" * 64
    assert second["receipt"]["lineage"]["lineage_digest"] != first["receipt"]["lineage"]["lineage_digest"]
    assert list(run_dir.glob("stage-08-qa-gap-fill-synthesis.superseded-*.json")), "prior receipt preserved"


def test_qa_gap_fill_synthesis_blocks_before_spend_when_stage7_is_itself_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 7's own upstream moved (analysis was not re-run): Stage 8 blocks
    before its paid client or its deterministic reports are produced."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-qa-upstream-stale"
    _write_qa_gap_fill_synthesis_predecessors(runner, workspace, run_id, topic_count=3)
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    stage7_path = run_dir / "stage-07-analysis.json"
    stage7 = json.loads(stage7_path.read_text(encoding="utf-8"))
    extraction_key = f"storage/workflow-runs/{run_id}/14A-exhaustive-topic-extraction-receipt.json"
    stage7["lineage"] = {
        **stage7["lineage"],
        "upstream_receipts": {
            extraction_key: {"stage_id": "exhaustive-topic-extraction", "digest_kind": "output_digest", "digest": "0" * 64}
        },
    }
    stage7_path.write_text(json.dumps(stage7), encoding="utf-8")
    traceability = workspace / runner.ANALYSIS_TRACEABILITY_REPORT
    before = traceability.read_bytes() if traceability.exists() else None
    counting, constructed = _counting_stage8_client_class(runner)
    monkeypatch.setattr(runner, "Stage8SynthesisClient", counting)

    outcome = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )

    assert outcome["status"] == "blocked"
    assert [item["code"] for item in outcome["receipt"]["blockers"]] == ["upstream_lineage_not_fresh"]
    assert not constructed, "the paid client must not be constructed over a stale upstream"
    gate = outcome["receipt"]["checks"]["upstream_lineage_blockers"]
    assert [(item["code"], item["upstream_stage_id"]) for item in gate] == [("stage_contract_upstream_stale", "analysis")]
    assert (traceability.read_bytes() if traceability.exists() else None) == before


def test_qa_gap_fill_synthesis_resume_rejects_receipt_without_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-qa-resume-no-lineage"
    _write_qa_gap_fill_synthesis_predecessors(runner, workspace, run_id, topic_count=3)
    monkeypatch.setattr(runner, "Stage8SynthesisClient", _build_fixture_stage8_synthesis_client_class(runner))
    first = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )
    assert first["status"] == "succeeded", first["receipt"]["blockers"]
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    _mutate_receipt(run_dir / "stage-08-qa-gap-fill-synthesis.json", lineage=None)
    counting, constructed = _counting_stage8_client_class(runner)
    monkeypatch.setattr(runner, "Stage8SynthesisClient", counting)
    second = runner.run_qa_gap_fill_synthesis_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None
    )
    assert second["status"] == "succeeded", second["receipt"]["blockers"]
    assert constructed, "a receipt without lineage is never reused"


def test_presentation_stage_resume_refuses_short_circuit_when_stage8_output_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole-stage short-circuit is refused when an upstream stage's output
    moved; the per-deck ledgers then decide that no deck step needs to re-run,
    and a fresh receipt bound to the new upstream digest is written."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-upstream-changed"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    first = _run_presentation_stage_with_fixtures(
        runner, workspace, run_id, monkeypatch, command_runner=_build_fake_presentation_command_runner()
    )
    assert first["receipt"]["status"] == "succeeded", first["receipt"]
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    stage8_key = f"storage/workflow-runs/{run_id}/stage-08-qa-gap-fill-synthesis.json"
    assert first["receipt"]["lineage"]["upstream_receipts"][stage8_key]["digest_kind"] == "output_digest"

    # Direct upstream moved (Stage 9 re-ran with new outputs): the short-circuit
    # is refused, no deck input moved so no step re-runs, and a fresh receipt is
    # bound to the new digest.
    stage9_key = f"storage/workflow-runs/{run_id}/stage-09-podcast.json"
    _mutate_receipt(run_dir / "stage-09-podcast.json", output_digest="2" * 64)
    call_log: list[str] = []
    second = _run_presentation_stage_with_fixtures(
        runner, workspace, run_id, monkeypatch, command_runner=_build_fake_presentation_command_runner(call_log=call_log)
    )
    assert second["receipt"]["status"] == "succeeded", second["receipt"]
    assert second["receipt"]["lineage"]["upstream_receipts"][stage9_key]["digest"] == "2" * 64
    assert second["receipt"]["lineage"]["lineage_digest"] != first["receipt"]["lineage"]["lineage_digest"]
    assert call_log == [], call_log  # no deck input moved: nothing re-ran, only the receipt was re-bound
    assert list(run_dir.glob("stage-10-presentation.superseded-*.json"))

    # An indirect upstream moved (Stage 8) while Stage 9 was NOT re-run: Stage 9
    # is transitively stale, so Stage 10 blocks before any deck command runs
    # (the pre-spend lineage gate) instead of re-binding over it.
    _mutate_receipt(run_dir / "stage-08-qa-gap-fill-synthesis.json", output_digest="3" * 64)
    third_calls: list[str] = []
    third = _run_presentation_stage_with_fixtures(
        runner, workspace, run_id, monkeypatch,
        command_runner=_build_fake_presentation_command_runner(call_log=third_calls),
    )
    assert third["receipt"]["status"] == "blocked"
    assert [item["code"] for item in third["receipt"]["blockers"]] == ["upstream_lineage_not_fresh"]
    assert third_calls == [], third_calls
    gate = third["receipt"]["checks"]["upstream_lineage_blockers"]
    assert [(item["code"], item["upstream_stage_id"]) for item in gate] == [
        ("stage_contract_upstream_stale", "podcast")
    ], gate


def test_exhaustive_topic_extraction_resume_never_rebinds_a_stale_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed extraction verifies the receipt's own recorded lineage: when
    Stage 6's output moved, the stage re-runs instead of healing the receipt."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-extraction-resume-stale"
    _write_stage6_receipt(workspace, run_id)
    calls: list[str] = []

    def fake_owner(workspace_arg: Path, *, run_id: str, force_rebuild: bool) -> dict[str, object]:
        calls.append(run_id)
        root = workspace_arg / "storage" / "extractions" / run_id
        (root / "by-topic").mkdir(parents=True, exist_ok=True)
        (root / "by-artifact").mkdir(parents=True, exist_ok=True)
        (root / "by-topic" / "01.json").write_text("{}\n", encoding="utf-8")
        (root / "by-artifact" / "a.json").write_text("{}\n", encoding="utf-8")
        (root / "v15_extraction_coverage.json").write_text('{"coverage_ratio": 1.0}\n', encoding="utf-8")
        return {"status": "succeeded", "blockers": []}

    monkeypatch.setattr(runner, "run_exhaustive_topic_extraction", fake_owner)
    first = runner.run_exhaustive_topic_extraction_stage(workspace=workspace, run_id=run_id, resume=True, env_file=None)
    assert first["status"] == "succeeded", first["receipt"]["blockers"]
    assert calls == [run_id]
    second = runner.run_exhaustive_topic_extraction_stage(workspace=workspace, run_id=run_id, resume=True, env_file=None)
    assert second["status"] == "succeeded" and calls == [run_id], "unchanged upstream: reused"

    run_dir = workspace / "storage" / "workflow-runs" / run_id
    _mutate_receipt(run_dir / "stage-06-corpus-inventory-index.json", output_digest="3" * 64)
    third = runner.run_exhaustive_topic_extraction_stage(workspace=workspace, run_id=run_id, resume=True, env_file=None)
    assert third["status"] == "succeeded", third["receipt"]["blockers"]
    assert calls == [run_id, run_id], "changed upstream: the owner ran again"
    stage6_key = f"storage/workflow-runs/{run_id}/stage-06-corpus-inventory-index.json"
    assert third["receipt"]["lineage"]["upstream_receipts"][stage6_key]["digest"] == "3" * 64


def test_priority_ranks_keep_papers_an_earlier_matching_run_found(tmp_path: Path) -> None:
    runner = load_runner_module()
    topic_dir = tmp_path / "storage" / "paper-matching" / "run-x" / "07_topic"
    topic_dir.mkdir(parents=True)
    current, older, oldest = ("a" * 64, "b" * 64, "c" * 64)
    (topic_dir / "matched_papers.json").write_text(
        json.dumps({"papers": [{"paper_id": current, "rank": 1}, {"paper_id": "arXiv:2506.1", "rank": 2}]}),
        encoding="utf-8",
    )
    (topic_dir / "matched_papers.superseded-20260920T000000Z.json").write_text(
        json.dumps({"papers": [{"paper_id": oldest, "rank": 1}]}), encoding="utf-8"
    )
    (topic_dir / "matched_papers.superseded-20260922T000000Z.json").write_text(
        json.dumps({"papers": [{"paper_id": older, "rank": 1}, {"paper_id": current, "rank": 2}]}),
        encoding="utf-8",
    )

    ranks = runner.production_analysis_stage.priority_source_ranks(
        tmp_path, run_id="run-x", topic_specs=({"id": "07_topic", "title": "T"},)
    )

    assert ranks["07_topic"] == {f"paper:{current}": 1, f"paper:{older}": 2, f"paper:{oldest}": 3}


def test_the_advisory_visual_qc_runs_for_topic_and_executive_decks_and_never_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conference that configures an advisory second-opinion visual QC gets it on EVERY deck,
    executive decks included (they were left out at first: the K3 trial never covered them, and
    they have no Gate A to hand it to). Each deck records a compact advisory summary. A pass that
    fails for any reason is recorded on the deck and the stage still succeeds."""

    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / "conferences")
    run_id = "test-presentation-advisory-visual-qc"
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(runner.presentation_paper_matching, "run_paper_matching_stage", _fake_presentation_paper_matching_stage)
    from conference_pipeline.production import presentation_chain as chain

    k3 = "openrouter/moonshotai/kimi-k3"
    monkeypatch.setattr(chain, "advisory_visual_qc_selection", lambda _workspace: ("openrouter", k3))
    base_runner = _build_fake_presentation_command_runner()
    advisory_commands: list[list[str]] = []

    def runner_with_advisory(command: list[str]) -> subprocess.CompletedProcess[str]:
        if "--model" in command and command[command.index("--model") + 1] == k3:
            advisory_commands.append(command)
            if "01_fixture_topic" in command:
                raise RuntimeError("the second model's provider is down")  # recorded, never raised
            report = Path(command[command.index("--report-output") + 1])
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({
                "status": "warnings", "provider": "openrouter", "model": k3,
                "slides": [{"slide_number": 1, "status": "warn",
                            "issues": [{"check_id": "other", "severity": "warn", "description": "a nit"}]}],
            }))
            return subprocess.CompletedProcess(command, 1, "", "")
        return base_runner(command)

    outcome = runner.run_presentation_stage(
        workspace=workspace, run_id=run_id, resume=True, provider_cost_approved=True, env_file=None,
        command_runner=runner_with_advisory,
    )
    receipt = outcome["receipt"]
    assert receipt["status"] == "succeeded", receipt
    records = {record["deck_id"]: record for record in receipt["checks"]["deck_records"]}
    assert len(records) == 14 and len(advisory_commands) == 14, "one advisory pass per deck, topic and executive"
    executive_ids = [deck["id"] for deck in receipt["checks"]["deck_scope"] if deck["kind"] == "executive"]
    assert executive_ids, "the fixture includes executive decks"
    for deck_id in executive_ids:
        advisory = records[deck_id]["visual_qc_advisory"]
        assert advisory["status"] == "ok" and advisory["model"] == k3
        assert records[deck_id]["visual_qc_advisory_summary_path"].endswith("visual_llm_qc_advisory_summary.json")


def test_gate_a_command_includes_both_fidelity_reports_and_tracks_them(tmp_path, monkeypatch):
    runner = load_runner_module()
    workspace = write_runner_workspace(tmp_path / 'conferences')
    run_id = 'test-fidelity-gate-a-handoff'
    _write_presentation_predecessors(runner, workspace, run_id, monkeypatch, topic_count=12)
    monkeypatch.setattr(runner.presentation_paper_matching, 'run_paper_matching_stage',
                        _fake_presentation_paper_matching_stage)
    base = _build_fake_presentation_command_runner()
    commands = []
    def command_runner(command):
        commands.append(command)
        return base(command)
    outcome = runner.run_presentation_stage(workspace=workspace, run_id=run_id, resume=False,
                                  provider_cost_approved=True, env_file=None,
                                  presentation_substage='all',
                                  figure_route='compiler', accept_visual_qc_findings=True,
                                  command_runner=command_runner)
    assert any(any(str(part).endswith('/review_presentation.py') for part in command) for command in commands), outcome
    gate_a = next(command for command in commands if any(str(part).endswith('/review_presentation.py') for part in command))
    reports = [gate_a[i + 1] for i, part in enumerate(gate_a) if part == '--claim-fidelity-report']
    deck = workspace / 'decks/01_fixture_topic'
    assert reports == [str(deck / f'claim_fidelity_review_{phase}.json') for phase in ('figures', 'narration')]
    ledger = json.loads((deck / 'step_lineage.json').read_text())
    inputs = ledger['steps']['review_presentation']['inputs']
    for phase in ('figures', 'narration'):
        assert f'ws:decks/01_fixture_topic/claim_fidelity_review_{phase}.json' in inputs
    fidelity_positions = [i for i, c in enumerate(commands) if any(str(p).endswith('/review_claim_fidelity.py') for p in c)
                          and _presentation_fixture_arg(c, '--deck-dir') == str(deck)]
    narration_position = next(i for i, c in enumerate(commands) if any(str(p).endswith('/generate_engaging_narration.py') for p in c))
    assert len(fidelity_positions) == 2
    assert fidelity_positions[0] < narration_position < fidelity_positions[1] < commands.index(gate_a)
