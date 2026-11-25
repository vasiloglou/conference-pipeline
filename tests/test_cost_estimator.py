import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from conference_pipeline.model_governance.registry import load_model_catalog
import yaml

import conference_pipeline.acquisition.cvpr_r6_artifact_contract as artifact_contract_module
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6ArtifactContract,
    R6ArtifactContractError,
    r6_audio_policy_digest,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    canonical_r6_recovery_inventory_digest,
    derive_r6_source_route_scope,
    source_route_scope_payload,
)
from conference_pipeline.costs.estimator import (
    build_receipt_bound_audio_workload,
    compare_workspace,
    estimate_openai_image_tokens,
    estimate_receipt_bound_audio_workload,
    estimate_workspace,
    load_pricing_catalog,
)

ICLR_BASELINE_AUDIO_MINUTES = 24_682.97


def _load_r6_estimator_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "estimate_cvpr2026_r6_costs.py"
    spec = importlib.util.spec_from_file_location("cvpr2026_r6_estimator", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r6_estimator_rejects_a_redirected_production_anchor_before_registry_or_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixed estimator cannot turn an external workspace into authority."""

    module = _load_r6_estimator_script()
    repository = tmp_path / "repo"
    workspace = repository / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    workspace.rmdir()
    workspace.symlink_to(external, target_is_directory=True)
    touched: list[str] = []

    monkeypatch.setattr(artifact_contract_module, "repository_root", lambda: repository)
    monkeypatch.setattr(module, "load_registry", lambda: touched.append("registry"))
    monkeypatch.setattr(module, "check_preflight", lambda *_args, **_kwargs: touched.append("preflight"))
    monkeypatch.setattr(sys, "argv", ["estimate_cvpr2026_r6_costs.py"])

    with pytest.raises(R6ArtifactContractError, match="anchor|workspace|symlink"):
        module.main()

    assert touched == []


def test_cvpr_r6_wrapper_accepts_only_declared_sanitized_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_r6_estimator_script()
    workspace = tmp_path / "workspace"
    contract = R6ArtifactContract.for_test_fixture(workspace)
    r6_root = workspace / "storage/transcription-ocr/cvpr2026/r6"
    r6_root.mkdir(parents=True)
    raw_audio = r6_root / "raw-audio/item-0000.mp3"
    raw_audio.parent.mkdir(parents=True)
    raw_audio.write_bytes(b"fixture audio")
    recovery = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_media_recovery_inventory",
        "run_id": contract.run_id,
        "counts": {"asr_fallback_required": 2_002},
        "asr_fallback_required": [
            {
                "r5_record_id": f"item-{index:04d}",
                "video_id": f"A{index:010d}",
                "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
                "source_event_urls": ["https://cvpr.thecvf.com/virtual/2026/poster/1"],
            }
            for index in range(2_002)
        ],
    }
    scope = derive_r6_source_route_scope(recovery, verify_inventory_digest=False)
    recovery.update(source_route_scope_payload(scope))
    recovery["r6_recovery_inventory_sha256"] = canonical_r6_recovery_inventory_digest(recovery)
    recovery_path = r6_root / "recovery_inventory.json"
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    audio_receipt = r6_root / "audio_recovery_receipt.json"
    audio_receipt.write_text(
        json.dumps(
            {
                "receipt_type": "conference_pipeline.cvpr2026_r6_audio_recovery",
                "status": "succeeded",
                "run_id": contract.run_id,
                "recovery_inventory_sha256": recovery["r6_recovery_inventory_sha256"],
                "audio_policy_digest": r6_audio_policy_digest(),
                "audio_policy_max_attempts": 3,
                "source_scope_count": scope.source_scope_count,
                "source_scope_sha256": scope.source_scope_sha256,
                "records": [
                    {
                        "item_id": "item-0000",
                        "video_id": "A0000000000",
                        "source_url": "https://www.youtube.com/watch?v=A0000000000",
                        "provenance": "direct_public_youtube",
                        "duration_seconds": 60.0,
                        "sha256": hashlib.sha256(raw_audio.read_bytes()).hexdigest(),
                        "byte_count": raw_audio.stat().st_size,
                        "mime_type": "audio/mpeg",
                        "local_media_path": raw_audio.relative_to(workspace).as_posix(),
                        "terminal_state": "acquired",
                        "attempt_count": 1,
                    }
                ],
                "terminal_records": [
                    {
                        "item_id": f"item-{index:04d}",
                        "video_id": f"A{index:010d}",
                        "source_url": f"https://www.youtube.com/watch?v=A{index:010d}",
                        "provenance": "direct_public_youtube",
                        "terminal_state": "download_failed",
                        "attempt_count": 3,
                    }
                    for index in range(1, 2_002)
                ],
                "counts": {"declared": 2_002, "acquired": 1, "terminal": 2_001},
            }
        )
    )

    result = module.estimate_r6_audio_receipt(
        contract=contract,
    )
    assert result["task_id"] == "transcription.session_speech"
    assert result["estimate"]["item_count"] == 1
    assert result["projected_total_usd"] <= result["all_in_cap_usd"]

    assert "--workspace" not in module.main.__code__.co_consts
    assert "--audio-receipt" not in module.main.__code__.co_consts
    assert "--output" not in module.main.__code__.co_consts
    assert "--all-in-cap-usd" not in module.main.__code__.co_consts

    # CR-07: the exact full 2,002-row receipt must not turn a symlinked
    # raw-audio root into an estimate, even when the external bytes retain
    # the declared digest.  This guard must fire before registry/preflight.
    external = tmp_path / "external-sentinel"
    external.mkdir()
    external_audio = external / raw_audio.name
    external_audio.write_bytes(b"fixture audio")
    raw_audio.unlink()
    raw_audio.parent.rmdir()
    raw_audio.parent.symlink_to(external, target_is_directory=True)
    original_read_bytes = Path.read_bytes

    def no_external_read(path: Path) -> bytes:
        if path.resolve() == external_audio.resolve():
            raise AssertionError("external raw-audio sentinel must not be consumed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", no_external_read)
    monkeypatch.setattr(
        module,
        "load_registry",
        lambda: (_ for _ in ()).throw(AssertionError("registry must remain unreachable")),
    )
    with pytest.raises(ValueError, match="raw audio"):
        module.estimate_r6_audio_receipt(contract=contract)

    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)
    raw_audio.parent.unlink()
    raw_audio.parent.mkdir()
    raw_audio.write_bytes(b"fixture audio")

    forged = json.loads(audio_receipt.read_text(encoding="utf-8"))
    forged["terminal_records"] = []
    # Keep a count-only story that once passed admission; exact immutable seed
    # classification must stop before registry/provider-cost construction.
    forged["counts"] = {"declared": 2_002, "acquired": 1, "terminal": 2_001}
    audio_receipt.write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "load_registry",
        lambda: (_ for _ in ()).throw(AssertionError("registry must remain unreachable")),
    )
    with pytest.raises(ValueError, match="count evidence|classify every immutable seed"):
        module.estimate_r6_audio_receipt(contract=contract)


def test_r6_estimator_secretish_key_scan_ignores_the_secrets_copied_flag() -> None:
    """``secrets_copied`` is positive sanitization evidence (a bool the owner
    sets to prove it copied none), not a secret value — it must not trip the
    same substring scan that catches real cookie/token/api_key fields."""

    module = _load_r6_estimator_script()

    assert module._contains_secretish_key({"secrets_copied": False}) is False
    assert module._contains_secretish_key({"secrets_copied": True}) is False
    assert module._contains_secretish_key({"nested": {"secrets_copied": False}}) is False
    assert module._contains_secretish_key([{"secrets_copied": False}]) is False

    # Real secret-shaped keys must still be rejected.
    assert module._contains_secretish_key({"api_key": "sk-live-x"}) is True
    assert module._contains_secretish_key({"cookie": "session=1"}) is True
    assert module._contains_secretish_key({"nested": {"password": "x"}}) is True


def test_receipt_bound_audio_workload_requires_declared_unique_finite_rows() -> None:
    rows = [
        {
            "item_id": "a",
            "video_id": "abcDEF12345",
            "duration_seconds": 30.0,
            "sha256": "a" * 64,
            "terminal_state": "acquired",
        },
        {
            "item_id": "b",
            "video_id": "defGHI67890",
            "duration_seconds": 90.0,
            "sha256": "b" * 64,
            "terminal_state": "acquired",
        },
    ]
    workload = build_receipt_bound_audio_workload(rows, receipt_sha256="c" * 64)
    assert workload.item_count == 2
    assert workload.duration_seconds == pytest.approx(120.0)

    with pytest.raises(ValueError, match="duplicate"):
        build_receipt_bound_audio_workload([*rows, rows[0]], receipt_sha256="c" * 64)
    with pytest.raises(ValueError, match="duration"):
        build_receipt_bound_audio_workload(
            [{**rows[0], "duration_seconds": float("nan")}], receipt_sha256="c" * 64
        )
    with pytest.raises(ValueError, match="receipt"):
        build_receipt_bound_audio_workload(rows, receipt_sha256="not-a-digest")


def test_receipt_bound_audio_estimate_uses_catalog_price_and_rejects_missing_price() -> None:
    catalog = load_pricing_catalog()
    workload = build_receipt_bound_audio_workload(
        [
            {
                "item_id": "a",
                "video_id": "abcDEF12345",
                "duration_seconds": 60.0,
                "sha256": "a" * 64,
                "terminal_state": "acquired",
            }
        ],
        receipt_sha256="c" * 64,
    )
    estimate = estimate_receipt_bound_audio_workload(
        workload,
        provider="openai",
        model="gpt-4o-mini-transcribe",
        catalog=catalog,
    )
    assert estimate["estimated_cost_usd"] > 0
    assert estimate["duration_minutes"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="price"):
        estimate_receipt_bound_audio_workload(
            workload,
            provider="openai",
            model="gpt-4o",
            catalog=catalog,
        )


def test_pricing_catalog_provider_math():
    catalog = load_pricing_catalog()

    openai_cost, _, _ = catalog.estimate_tokens("openai", "gpt-4o", 1_000_000, 1_000_000)
    openai_mini_cost, _, _ = catalog.estimate_tokens("openai", "gpt-5.4-mini", 1_000_000, 1_000_000)
    anthropic_sonnet_46_cost, _, _ = catalog.estimate_tokens(
        "anthropic", "claude-sonnet-4-6", 1_000_000, 1_000_000
    )
    # Sub-threshold (<=200k input) exercises the base tier.
    gemini_cost, _, _ = catalog.estimate_tokens("gemini", "gemini-2.5-pro", 100_000, 100_000)
    gemini_flash_lite_cost, _, _ = catalog.estimate_tokens(
        "gemini", "gemini-3.1-flash-lite-preview", 1_000_000, 1_000_000
    )
    tavily_cost, _, _ = catalog.estimate_credits("tavily", "tavily-search", 12)
    openrouter_cost, _, warnings = catalog.estimate_tokens(
        "openrouter", "openrouter/anthropic/claude-sonnet-4-6", 1_000_000, 1_000_000
    )

    assert openai_cost == pytest.approx(12.5)
    assert openai_mini_cost == pytest.approx(5.25)
    assert anthropic_sonnet_46_cost == pytest.approx(18.0)
    assert gemini_cost == pytest.approx(1.125)
    assert gemini_flash_lite_cost == pytest.approx(1.75)
    assert tavily_cost == pytest.approx(0.12)
    assert openrouter_cost == pytest.approx(18.0)
    assert any("OpenRouter" in warning for warning in warnings)


def test_gemini_large_prompt_tier_crossover():
    """Gemini 2.5/3.x Pro bill requests above 200k input tokens at a higher
    per-token rate for both input and output. The estimator must apply the
    high tier per-request based on that call's input_tokens (regression for
    the ICML 2026 07A rerun, where flat pricing under-estimated ~2x)."""
    catalog = load_pricing_catalog()

    # 200k is the boundary: at-or-below uses base, strictly above uses high.
    base_cost, _, base_warn = catalog.estimate_tokens(
        "gemini", "gemini-3.1-pro-preview", 200_000, 10_000
    )
    high_cost, _, high_warn = catalog.estimate_tokens(
        "gemini", "gemini-3.1-pro-preview", 200_001, 10_000
    )

    # base: 0.2*$2 + 0.01*$12 = 0.52
    assert base_cost == pytest.approx(0.52)
    assert not any("Large-prompt tier" in w for w in base_warn)
    # high: 0.200001*$4 + 0.01*$18 ≈ 0.980
    assert high_cost == pytest.approx(0.200001 * 4.0 + 0.01 * 18.0)
    assert any("Large-prompt tier" in w for w in high_warn)

    # 2.5 Pro tier crossover (1.25/10 -> 2.5/15).
    pro25_high, _, _ = catalog.estimate_tokens("gemini", "gemini-2.5-pro", 300_000, 5_000)
    assert pro25_high == pytest.approx(0.3 * 2.5 + 0.005 * 15.0)


def test_iclr_audio_baseline_transcription_costs():
    catalog = load_pricing_catalog()

    mini_cost, _, _ = catalog.estimate_audio_minutes(
        "openai", "gpt-4o-mini-transcribe", ICLR_BASELINE_AUDIO_MINUTES
    )
    base_cost, _, _ = catalog.estimate_audio_minutes(
        "openai", "gpt-4o-transcribe", ICLR_BASELINE_AUDIO_MINUTES
    )
    diarized_cost, _, _ = catalog.estimate_audio_minutes(
        "openai", "gpt-4o-transcribe-diarize", ICLR_BASELINE_AUDIO_MINUTES
    )

    assert mini_cost == pytest.approx(74.05, abs=0.01)
    assert base_cost == pytest.approx(148.10, abs=0.01)
    assert diarized_cost == pytest.approx(148.10, abs=0.01)


def test_image_token_estimator_uses_openai_high_detail_rule():
    assert estimate_openai_image_tokens(4078, 2880) == 1105
    assert estimate_openai_image_tokens(512, 512, detail="low") == 85


def test_full_scope_includes_whole_pipeline_families(tmp_path: Path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps({"topics": [{"id": "a"}, {"id": "b"}]})
    )
    (analysis_dir / "01_topic.md").write_text("x" * 4000)

    report = estimate_workspace(tmp_path, scope="full", scenario="base")
    families = set(report.task_family_ids())
    task_ids = {task.task_id for task in report.tasks}
    fact_ranking = next(task for task in report.tasks if task.task_id == "fact_ranking")

    assert {
        "asr_transcription",
        "visual_ocr",
        "pdf_text_followup",
        "exhaustive_extraction",
        "gpt_researcher",
        "embeddings",
        "web_search",
        "topic_analysis",
        "rankings",
        "podcast_generation",
        "presentation_generation",
        "video_generation",
        "local_zero_cost",
    }.issubset(families)
    assert "programmatic_figure_codegen" in task_ids
    assert "programmatic_figure_repair" in task_ids
    assert fact_ranking.quantity == 10


def test_full_scope_prices_bounded_source_discovery_cross_source_joins(tmp_path: Path):
    inventory_path = tmp_path / "storage" / "source-discovery" / "source_discovery_inventory.json"
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text(
        json.dumps(
            {
                "candidate_frontier": {
                    "pairs": [
                        {"canonical_pair_key": "program:one|paper:one", "inclusion_status": "pending_llm"},
                        {"canonical_pair_key": "program:two|paper:two", "inclusion_status": "pending_llm"},
                        {
                            "canonical_pair_key": "program:navigation|paper:navigation",
                            "inclusion_status": "excluded",
                        },
                        {
                            # Deterministically resolved (SourceFamily.deterministic_title_join);
                            # never reaches the paid boundary, so it must not be priced either.
                            "canonical_pair_key": "program:three|paper:three",
                            "inclusion_status": "included",
                        },
                    ]
                }
            }
        )
    )

    report = estimate_workspace(tmp_path, scope="full", scenario="base")
    join = next(
        task for task in report.tasks if task.task_id == "source_discovery_cross_source_join"
    )

    assert join.stage == "source-discovery"
    assert join.enabled is True
    assert join.quantity == 2
    assert math.isfinite(join.estimated_cost_usd)
    assert join.estimated_cost_usd > 0


def test_source_discovery_join_is_explicitly_disabled_without_candidates_and_rejects_unpriced_specs(
    tmp_path: Path,
) -> None:
    empty_report = estimate_workspace(tmp_path, scope="full", scenario="base")
    empty_join = next(
        task for task in empty_report.tasks if task.task_id == "source_discovery_cross_source_join"
    )
    assert empty_join.enabled is False
    assert empty_join.quantity == 0
    assert "Disabled because no candidate frontier pairs are available." in empty_join.warnings

    from conference_pipeline.costs.estimator import load_task_specs

    specs = load_task_specs()
    join_spec = next(
        task for task in specs["tasks"] if task["task_id"] == "source_discovery_cross_source_join"
    )
    join_spec["models"]["base"] = "unpriced-source-discovery-model"
    spec_path = tmp_path / "task_specs.yaml"
    spec_path.write_text(yaml.safe_dump(specs, sort_keys=False))

    inventory_path = tmp_path / "storage" / "source-discovery" / "source_discovery_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {"candidate_frontier": {"pairs": [{"canonical_pair_key": "a|b", "inclusion_status": "pending_llm"}]}}
        )
    )
    with pytest.raises(ValueError, match="source-discovery join projection is unpriced"):
        estimate_workspace(tmp_path, scope="full", scenario="base", task_specs_path=spec_path)


def test_cvpr_tavily_priority_gap_task_prices_only_the_current_complete_frontier(
    tmp_path: Path,
) -> None:
    """Stage 4 must never infer paid Tavily scope from a bare environment key."""

    ledger_path = tmp_path / "storage" / "download-corpus" / "download_corpus.json"
    ledger_path.parent.mkdir(parents=True)
    frontier = {
        "program_spine_sha256": "a" * 64,
        "harvest_inventory_sha256": "b" * 64,
        "query_policy_sha256": "c" * 64,
        "priority_program_item_ids": ["cvpr:item:001", "cvpr:item:002"],
        "priority_gap_quantity": 2,
    }
    frontier["priority_gap_frontier_sha256"] = (
        __import__("hashlib")
        .sha256(json.dumps(frontier, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        .hexdigest()
    )
    ledger_path.write_text(json.dumps({"talk_harvest": {"priority_gap_frontier": frontier}}))

    report = estimate_workspace(tmp_path, scope="full", scenario="base")
    task = next(
        task for task in report.tasks if task.task_id == "cvpr2026_tavily_priority_gap_search"
    )

    assert task.stage == "download-corpus"
    assert task.provider == "tavily"
    assert task.enabled is True
    assert task.quantity == 2
    assert task.credits == 2
    assert math.isfinite(task.estimated_cost_usd)
    assert task.estimated_cost_usd > 0

    frontier["priority_program_item_ids"] = ["cvpr:item:001", "cvpr:item:001"]
    ledger_path.write_text(json.dumps({"talk_harvest": {"priority_gap_frontier": frontier}}))
    with pytest.raises(ValueError, match="Tavily priority-gap frontier"):
        estimate_workspace(tmp_path, scope="full", scenario="base")


def test_visual_ocr_model_plan_includes_primary_and_fallback(tmp_path: Path):
    image_dir = tmp_path / "slides"
    image_dir.mkdir()
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a project dependency.
        pytest.skip("Pillow is required for image fixture")
    Image.new("RGB", (1024, 768), color="white").save(image_dir / "slide_001.png")

    report = estimate_workspace(tmp_path, scope="visual-ocr", scenario="base")
    by_task = {task.task_id: task for task in report.tasks}

    assert by_task["slide_poster_visual_ocr"].provider == "openai"
    # Resolved through the catalog aliases, so a model-generation upgrade moves these with it.
    catalog = load_model_catalog()
    assert by_task["slide_poster_visual_ocr"].model == catalog["luna"]["model"]
    assert by_task["slide_poster_visual_ocr_fallback"].provider == "gemini"
    assert by_task["slide_poster_visual_ocr_fallback"].model == catalog["gemini_flash_lite"]["model"]
    assert by_task["slide_poster_visual_ocr_fallback"].quantity == pytest.approx(0.1)


def test_compare_workspace_outputs_three_scenarios(tmp_path: Path):
    payload = compare_workspace(tmp_path, scope="full")

    assert set(payload["scenarios"]) == {"low", "base", "high"}
    assert payload["scenarios"]["base"]["total_estimated_cost_usd"] >= 0


def test_pre_frontier_source_discovery_inventory_prices_no_join(tmp_path: Path) -> None:
    """An inventory written before the join frontier existed has no pending pairs; a malformed one still refuses."""
    from conference_pipeline.costs.estimator import _count_source_discovery_candidate_pairs

    inventory_path = tmp_path / "storage" / "source-discovery" / "source_discovery_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps({"schema_version": 1, "sources": [], "status": "succeeded"}))
    assert _count_source_discovery_candidate_pairs(tmp_path) == 0
    inventory_path.write_text(json.dumps({"candidate_frontier": {"pairs": "not-a-list"}}))
    with pytest.raises(ValueError, match="missing candidate frontier pairs"):
        _count_source_discovery_candidate_pairs(tmp_path)


def _task(report, task_id: str):
    return next(task for task in report.tasks if task.task_id == task_id)


def test_exhaustive_summary_is_priced_per_call_from_the_summary_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "storage/analysis/exhaustive-summaries/run-a/summary_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {"topic_id": "01_a", "category": "transcripts", "input_tokens": 300_000},
                    {"topic_id": "01_a", "category": "papers", "input_tokens": 100_000},
                    {"topic_id": "02_b", "category": "papers", "input_tokens": 0},
                ]
            }
        )
    )

    report = estimate_workspace(tmp_path, scope="exhaustive_topic_summary_synthesis", scenario="base")
    task = _task(report, "exhaustive_topic_summary_synthesis")
    catalog = load_pricing_catalog()
    threshold = catalog.high_tier_input_token_threshold(task.model, task.provider)

    assert task.quantity_basis == "measured"
    assert not any(warning.startswith("UNMEASURED") for warning in task.warnings)
    assert task.estimated_cost_usd > 0
    assert any("Sized from summary_plan: 2 topic/category pairs" in item for item in task.assumptions)
    if threshold is not None:
        # 300k splits into two sub-threshold chunks plus a merge; 100k is one call.
        assert task.quantity == 4
        assert any(item.startswith("High-tier split: 0 of 4 calls") for item in task.assumptions)


def test_exhaustive_summary_is_sized_from_full_read_records_without_a_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "storage/extractions/run-a/llm-full-read"
    (run_dir / "by-unit").mkdir(parents=True)
    (run_dir / "by-topic").mkdir(parents=True)
    items = []
    for index, kind in enumerate(("transcript_segment", "transcript_segment", "paper_abstract")):
        unit_id = f"unit-{index}"
        items.append({"unit_id": unit_id, "input_kind": kind})
        (run_dir / "by-unit" / f"{unit_id}.json").write_text(
            json.dumps({"unit_id": unit_id, "text_length": 40_000, "summary": "s", "reason": "r"})
        )
    (run_dir / "by-topic" / "01_a.json").write_text(json.dumps({"items": items}))

    report = estimate_workspace(tmp_path, scope="exhaustive_topic_summary_synthesis", scenario="base")
    task = _task(report, "exhaustive_topic_summary_synthesis")

    assert report.workload.exhaustive_summary_quantity_basis == "llm_full_read_records"
    assert len(report.workload.exhaustive_summary_pair_tokens) == 2
    papers, transcripts = report.workload.exhaustive_summary_pair_tokens  # sorted by (topic, category)
    assert papers > 10_000  # one 40k-char unit at ~4 chars/token
    assert transcripts > 20_000  # two 40k-char units
    assert task.quantity_basis == "measured"
    assert task.quantity == 2
    assert task.estimated_cost_usd > 0


def test_unsized_paid_tasks_price_their_declared_budget_loudly_instead_of_zero(tmp_path: Path) -> None:
    report = estimate_workspace(tmp_path, scope="full", scenario="base")

    for task_id in ("exhaustive_topic_summary_synthesis", "paper_matching_keypoint_extraction"):
        task = _task(report, task_id)
        assert task.quantity_basis == "declared_fallback_budget"
        assert task.estimated_cost_usd > 0
        assert task.warnings[0].startswith("UNMEASURED QUANTITY")
    assert not any(
        warning.startswith("Unknown quantity source")
        for task in report.tasks
        if task.provider != "local"
        for warning in task.warnings
    )


def test_paper_matching_keypoints_scale_with_topics_and_measured_findings(tmp_path: Path) -> None:
    from conference_pipeline.costs.estimator import PAPER_MATCHING_KEYPOINT_FINDINGS_CHAR_CAP
    from conference_pipeline.paper_matching import matcher

    # The estimator's cap must match the extractor's truncation.
    assert matcher.KEYPOINT_FINDINGS_CHAR_CAP == PAPER_MATCHING_KEYPOINT_FINDINGS_CHAR_CAP

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps({"topics": [{"id": "01_a"}, {"id": "02_b"}, {"id": "03_c"}]})
    )
    capped = _task(
        estimate_workspace(tmp_path, scope="paper-matching", scenario="base"),
        "paper_matching_keypoint_extraction",
    )
    assert capped.quantity == 3
    assert capped.quantity_basis == "measured"
    assert any("topic_count_x_findings_cap" in item for item in capped.assumptions)

    findings = tmp_path / "storage/paper-matching/run-a/01_a/artifact_gpt_researcher_findings.md"
    findings.parent.mkdir(parents=True)
    findings.write_text("x" * 4_000)
    measured = _task(
        estimate_workspace(tmp_path, scope="paper-matching", scenario="base"),
        "paper_matching_keypoint_extraction",
    )
    assert measured.quantity == 3
    assert any("paper_matching_findings" in item for item in measured.assumptions)
    assert 0 < measured.estimated_cost_usd < capped.estimated_cost_usd


def test_unknown_paid_quantity_source_refuses_without_a_declared_budget(tmp_path: Path) -> None:
    from conference_pipeline.costs.estimator import load_task_specs

    specs = load_task_specs()
    base = next(task for task in specs["tasks"] if task["task_id"] == "workshop_talk_boundary_discovery")
    budgeted = dict(base, task_id="paid_unknown_budgeted", quantity_source="not_a_real_source")
    unbudgeted = dict(budgeted, task_id="paid_unknown_unbudgeted")
    unbudgeted.pop("fallback_input_tokens", None)
    local = dict(budgeted, task_id="local_unknown", provider="local")
    local["models"] = {"low": "local-only", "base": "local-only", "high": "local-only"}

    spec_path = tmp_path / "specs.yaml"
    spec_path.write_text(yaml.safe_dump({**specs, "tasks": [budgeted, local]}, sort_keys=False))
    report = estimate_workspace(tmp_path, scope="full", scenario="base", task_specs_path=spec_path)
    priced = _task(report, "paid_unknown_budgeted")
    assert priced.quantity_basis == "declared_fallback_budget"
    assert priced.estimated_cost_usd > 0
    assert "Unknown quantity source: not_a_real_source" in _task(report, "local_unknown").warnings

    spec_path.write_text(yaml.safe_dump({**specs, "tasks": [unbudgeted]}, sort_keys=False))
    with pytest.raises(ValueError, match="refusing to price it at \\$0"):
        estimate_workspace(tmp_path, scope="full", scenario="base", task_specs_path=spec_path)
