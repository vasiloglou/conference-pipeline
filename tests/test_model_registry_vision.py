"""Phase 65 (v1.61) — Visual LLM registry & catalog coverage."""

from __future__ import annotations

import pytest

from conference_pipeline.model_governance.local_runtime import load_profiles
from conference_pipeline.model_governance.registry import (
    LocalCandidateRecord,
    ModelGovernanceError,
    ModelRegistry,
    load_registry,
)


VISION_TASK_IDS = (
    "vision.slide_analysis",
    "vision.figure_caption",
    "vision.chart_summary",
    "presentation.visual_qc",
    "ocr.slide_image_parse",
)

_VLM_RUNTIME_KINDS = {"mlx_vlm", "vllm_mlx", "mlx_native_fastvlm"}


def test_vision_task_entries_exist() -> None:
    registry = load_registry()
    for task_id in VISION_TASK_IDS:
        policy = registry.get(task_id)
        assert policy.task_family in {"vision", "ocr"}, task_id
        assert policy.local_runtime_policy.startswith("deferred"), task_id
        assert policy.local_candidates, f"{task_id} must have local_candidates"
        assert policy.allowed_selections, f"{task_id} must have allowed_selections"


def test_vision_task_entries_satisfy_governance_contract() -> None:
    registry = load_registry()
    for task_id in VISION_TASK_IDS:
        policy = registry.get(task_id)
        assert policy.current_research["required"] is True, task_id
        assert policy.benchmark["required"] is True, task_id
        assert policy.benchmark["receipt_required"] is True, task_id
        assert policy.provenance_required is True, task_id


def test_local_candidates_carry_runtime_kind_metadata() -> None:
    registry = load_registry()
    for task_id in VISION_TASK_IDS:
        policy = registry.get(task_id)
        for candidate in policy.local_candidates:
            assert isinstance(candidate, LocalCandidateRecord)
            assert candidate.status == "benchmark_pending", (
                f"{task_id}: candidate {candidate.model_id} must enter as benchmark_pending"
            )
            assert candidate.runtime_kind in _VLM_RUNTIME_KINDS, (
                f"{task_id}: candidate {candidate.model_id} runtime_kind must be one of "
                f"{_VLM_RUNTIME_KINDS}, got {candidate.runtime_kind!r}"
            )
            assert candidate.parameter_count, (
                f"{task_id}: candidate {candidate.model_id} missing parameter_count"
            )
            assert candidate.recommended_um_floor_gb is not None, (
                f"{task_id}: candidate {candidate.model_id} missing recommended_um_floor_gb"
            )
            assert candidate.multimodal is not None, (
                f"{task_id}: candidate {candidate.model_id} missing multimodal flag"
            )


def test_slide_analysis_includes_full_appendix_a_roster() -> None:
    """vision.slide_analysis is the primary slide-analysis task per VLM-REG-01,
    so it carries the full Appendix A multimodal roster."""
    registry = load_registry()
    candidates = {c.model_id for c in registry.get("vision.slide_analysis").local_candidates}

    expected_substrings = (
        "Qwen3-VL-8B",
        "Qwen3-VL-32B",
        "Qwen3-VL-4B",
        "Qwen3-VL-72B",
        "Qwen2.5-VL-7B",
        "Qwen2.5-VL-3B",
        "Qwen2.5-VL-32B",
        "FastVLM-7B",
        "FastVLM-1.5B",
        "FastVLM-0.5B",
        "Pixtral-12B-4bit",
        "Pixtral-12B-8bit",
        "gemma-3-vision",
        "SmolVLM",
        "Idefics3",
        "Molmo",
        "Phi-3-Vision",
        "llava-1.5-7B",
        "llava-v1.6-mistral",
        "llava-llama-3-8B",
    )
    for needle in expected_substrings:
        assert any(needle in (mid or "") for mid in candidates), (
            f"vision.slide_analysis missing candidate matching {needle!r}; "
            f"have: {sorted(c for c in candidates if c)}"
        )


def test_ocr_task_includes_specialized_ocr_models() -> None:
    registry = load_registry()
    candidates = {c.model_id for c in registry.get("ocr.slide_image_parse").local_candidates}
    for needle in ("GLM-OCR", "dots-ocr", "DeepSeek-OCR-2"):
        assert any(needle in (mid or "") for mid in candidates), (
            f"ocr.slide_image_parse missing OCR-specialized candidate matching {needle!r}; "
            f"have: {sorted(c for c in candidates if c)}"
        )


def test_runtime_kind_validation_rejects_unknown_value() -> None:
    payload = {
        "schema_version": 1,
        "tasks": {
            "vision.test_invalid_runtime_kind": {
                "task_family": "vision",
                "stage": "presentation",
                "model_ref": "gpt54_mini_legacy",
                "current_research": {"required": True},
                "benchmark": {
                    "required": True,
                    "receipt_required": True,
                    "candidates": [],
                },
                "quality_threshold": "test",
                "schema_validity_required": True,
                "provenance_required": True,
                "local_runtime_policy": "deferred_until_test",
                "selection_status": "planning_default_requires_stage_receipt",
                "allowed_selections": [
                    {"provider": "openai", "model": "gpt-5.4-mini", "role": "default"},
                ],
                "local_candidates": [
                    {
                        "model_id": "fake/Vision-Model",
                        "checked_date": "2026-05-03",
                        "runtime_support": "fake",
                        "status": "benchmark_pending",
                        "runtime_kind": "not_a_real_runtime",
                    },
                ],
            },
        },
    }
    with pytest.raises(ModelGovernanceError, match="runtime_kind must be one of"):
        ModelRegistry.from_mapping(payload)


def test_multimodal_flag_must_be_boolean() -> None:
    payload = {
        "schema_version": 1,
        "tasks": {
            "vision.test_invalid_multimodal": {
                "task_family": "vision",
                "stage": "presentation",
                "model_ref": "gpt54_mini_legacy",
                "current_research": {"required": True},
                "benchmark": {
                    "required": True,
                    "receipt_required": True,
                    "candidates": [],
                },
                "quality_threshold": "test",
                "schema_validity_required": True,
                "provenance_required": True,
                "local_runtime_policy": "deferred_until_test",
                "selection_status": "planning_default_requires_stage_receipt",
                "allowed_selections": [
                    {"provider": "openai", "model": "gpt-5.4-mini", "role": "default"},
                ],
                "local_candidates": [
                    {
                        "model_id": "fake/Vision-Model",
                        "checked_date": "2026-05-03",
                        "runtime_support": "fake",
                        "status": "benchmark_pending",
                        "runtime_kind": "vllm_mlx",
                        "multimodal": "yes",
                    },
                ],
            },
        },
    }
    with pytest.raises(ModelGovernanceError, match="multimodal must be bool"):
        ModelRegistry.from_mapping(payload)


def test_pre_v1_61_local_candidates_still_parse_without_new_fields() -> None:
    """Backwards compatibility: legacy entries without runtime_kind/etc. still parse."""
    payload = {
        "schema_version": 1,
        "tasks": {
            "legacy.task": {
                "task_family": "extraction",
                "stage": "analysis",
                "model_ref": "gpt54_mini_legacy",
                "current_research": {"required": True},
                "benchmark": {
                    "required": True,
                    "receipt_required": True,
                    "candidates": [],
                },
                "quality_threshold": "test",
                "schema_validity_required": False,
                "provenance_required": True,
                "local_runtime_policy": "deferred_until_test",
                "selection_status": "planning_default_requires_stage_receipt",
                "allowed_selections": [
                    {"provider": "openai", "model": "gpt-5.4-mini", "role": "default"},
                ],
                "local_candidates": [
                    {
                        "model_id": "legacy/model",
                        "checked_date": "2026-04-27",
                        "runtime_support": "mlx_community_tested",
                        "status": "benchmark_pending",
                    },
                ],
            },
        },
    }
    registry = ModelRegistry.from_mapping(payload)
    candidate = registry.get("legacy.task").local_candidates[0]
    assert candidate.runtime_kind is None
    assert candidate.parameter_count is None
    assert candidate.multimodal is None
    assert candidate.recommended_um_floor_gb is None


def test_vision_runtime_profiles_loaded() -> None:
    profiles = load_profiles()
    assert "mlx_vision_light" in profiles
    assert "mlx_vision_heavy" in profiles
    light = profiles["mlx_vision_light"]
    assert light.runtime_kind == "mlx"
    assert light.base_url.startswith("http://127.")
    assert light.manage_server is True
    assert light.approval_status == "benchmark_pending"
    assert "mlx_vlm" in light.optional_deps

    heavy = profiles["mlx_vision_heavy"]
    assert heavy.manage_server is False
    assert heavy.lock_scope == "heavy"
    assert heavy.max_concurrency == 1
