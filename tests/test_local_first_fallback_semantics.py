"""Local-first paid-fallback semantics: a fallback must NEVER route to local.

Phase 10 (v3.0 / LOCAL-02), plan 10-01 Task 3. Regression guard for the registry
shape ``provider: openai`` + ``fallback_provider: local`` (the real
``transcription.session_speech`` shape): every fallback reason must resolve to a
non-local paid selection. Local is selectable ONLY through
``approved_local_selection()`` after proof + readiness.
"""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.benchmarks.receipt_schema import (
    BenchmarkReceipt,
    receipt_to_dict,
)
from conference_pipeline.model_governance.local_routing import (
    resolve_local_first_selection,
)
from conference_pipeline.model_governance.registry import load_registry

TASK = "transcription.session_speech"


# ---------------------------------------------------------------------------
# Fixtures: the transcription registry shape (provider=openai + fallback=local)
# ---------------------------------------------------------------------------


# Companion catalog covering every model referenced by this file's ad-hoc
# registries (including fake test-only identifiers), so they resolve
# hermetically. registry.py looks for a sibling models.yaml next to a custom
# registry path before falling back to the real production catalog.
_TEST_MODEL_CATALOG_YAML = """\
schema_version: 1
models:
  test_gpt4o_transcribe_diarize:
    provider: openai
    model: gpt-4o-transcribe-diarize
    status: active
  test_whisper_local:
    provider: local
    model: whisper-1
    status: active
  test_some_local:
    provider: local
    model: mlx-community/SomeLocal-4bit
    status: active
  test_gpt55:
    provider: openai
    model: gpt-5.5
    status: active
  test_gpt54_mini:
    provider: openai
    model: gpt-5.4-mini
    status: active
  test_gemini_flash_lite:
    provider: gemini
    model: gemini-3.1-flash-lite-preview
    status: active
"""


def _write_registry(tmp_path: Path, body: str) -> Path:
    (tmp_path / "models.yaml").write_text(_TEST_MODEL_CATALOG_YAML, encoding="utf-8")
    reg_path = tmp_path / "registry.yaml"
    reg_path.write_text(body, encoding="utf-8")
    load_registry.cache_clear()
    return reg_path


def _deferred_local_registry(tmp_path: Path) -> Path:
    """provider=openai, fallback_provider=local, NO approved local proof.

    Mirrors transcription.session_speech: the local candidate is a
    ``deferred_local_candidate`` (not ``approved_local``), local_proof is empty.
    """
    body = f"""\
schema_version: 1
tasks:
  {TASK}:
    task_family: transcription
    stage: transcription-ocr
    model_ref: test_gpt4o_transcribe_diarize
    fallback_model_ref: test_whisper_local
    current_research: {{required: true}}
    benchmark:
      required: true
      candidates: [whisper-1]
    quality_threshold: high
    schema_validity_required: false
    provenance_required: true
    local_runtime_policy: deferred_until_local_runtime_receipts
    selection_status: planning_default_requires_stage_receipt
    local_proof: {{}}
    allowed_selections:
      - {{provider: openai, model: gpt-4o-transcribe-diarize, role: default}}
      - {{provider: local, model: whisper-1, role: deferred_local_candidate}}
"""
    return _write_registry(tmp_path, body)


def _approved_local_registry(tmp_path: Path) -> Path:
    """Same shape but with full local proof + a passing receipt + approved_local.

    Used to isolate the readiness gate: the approved-local gate passes, so a
    failing readiness check is the ONLY reason a fallback fires.
    """
    receipt_output = tmp_path / "local-output.txt"
    receipt_output.write_text("test output", encoding="utf-8")
    receipt = BenchmarkReceipt(
        workspace="test",
        task_id=TASK,
        task_family="transcription",
        stage="transcription-ocr",
        run_id="00000000-0000-0000-0000-000000000000",
        timestamp="2026-06-25T00:00:00+00:00",
        execution_mode="live",
        provider="local",
        model_id="whisper-1",
        runtime_id="mlx",
        hardware_profile="mlx:unit-test",
        prompt_digest="a" * 64,
        config_digest="b" * 64,
        input_sample_ref="fixture://test",
        output_ref=str(receipt_output),
        raw_output_snippet="test output",
        rubric_scores={"quality": 1.0, "schema_ok": 1.0},
        schema_ok=True,
        provenance_ok=True,
        quality_delta=0.0,
        latency_ms=0.0,
        cost_usd=0.0,
        budget_violations=[],
        selection_decision="passed",
        failure_reason=None,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt_to_dict(receipt)), encoding="utf-8")
    body = f"""\
schema_version: 1
tasks:
  {TASK}:
    task_family: transcription
    stage: transcription-ocr
    model_ref: test_gpt4o_transcribe_diarize
    fallback_model_ref: test_whisper_local
    current_research: {{required: true}}
    benchmark:
      required: true
      candidates: [whisper-1]
      authority_selections:
        - {{provider: local, model: whisper-1}}
    quality_threshold: high
    schema_validity_required: false
    provenance_required: true
    local_runtime_policy: approved
    selection_status: active
    local_proof:
      research_ref: "ref/research"
      hardware_fit_ref: "ref/hardware"
      benchmark_receipt_ref: "{receipt_path}"
      operator_approval_ref: "ref/approval"
    allowed_selections:
      - {{provider: openai, model: gpt-4o-transcribe-diarize, role: default}}
      - {{provider: local, model: whisper-1, role: approved_local}}
"""
    return _write_registry(tmp_path, body)


def _assert_not_local(decision) -> None:
    assert decision.used_local is False
    assert decision.fallback_used is True
    assert decision.provider != "local"
    assert decision.model != "whisper-1"
    assert decision.paid_fallback_provider != "local"


# ---------------------------------------------------------------------------
# The three fallback reasons must all resolve NON-local (the regression)
# ---------------------------------------------------------------------------


def test_switch_off_does_not_route_to_local(tmp_path: Path) -> None:
    reg = _deferred_local_registry(tmp_path)
    decision = resolve_local_first_selection(
        TASK, local_first_enabled=False, registry_path=reg
    )
    assert decision.fallback_reason == "local_first_disabled"
    _assert_not_local(decision)
    # The paid fallback is the registry default (openai), never the local fallback.
    assert decision.provider == "openai"
    assert decision.model == "gpt-4o-transcribe-diarize"


def test_missing_approved_local_does_not_route_to_local(tmp_path: Path) -> None:
    reg = _deferred_local_registry(tmp_path)
    decision = resolve_local_first_selection(
        TASK, local_first_enabled=True, registry_path=reg
    )
    assert decision.fallback_reason == "approved_local_unavailable"
    _assert_not_local(decision)
    assert decision.provider == "openai"
    assert decision.model == "gpt-4o-transcribe-diarize"


def test_readiness_false_does_not_route_to_local(tmp_path: Path) -> None:
    """Approved-local proof present, but readiness fails -> non-local fallback."""
    reg = _approved_local_registry(tmp_path)
    decision = resolve_local_first_selection(
        TASK,
        local_first_enabled=True,
        registry_path=reg,
        readiness_check=lambda _task: False,
    )
    assert decision.fallback_reason == "local_readiness_failed"
    _assert_not_local(decision)
    assert decision.provider == "openai"


# ---------------------------------------------------------------------------
# Preference: prefer a non-local fallback over a local default
# ---------------------------------------------------------------------------


def test_local_default_with_nonlocal_fallback_resolves_to_paid(tmp_path: Path) -> None:
    """Mirror analysis.topic_reconciliation: provider=local, fallback=openai."""
    body = """\
schema_version: 1
tasks:
  analysis.topic_reconciliation:
    task_family: analysis
    stage: analysis
    model_ref: test_some_local
    fallback_model_ref: test_gpt55
    current_research: {required: true}
    benchmark:
      required: true
      candidates: [gpt-5.5]
    quality_threshold: high
    schema_validity_required: false
    provenance_required: true
    local_runtime_policy: deferred
    selection_status: active
    local_proof: {}
    allowed_selections:
      - {provider: local, model: mlx-community/SomeLocal-4bit, role: default}
      - {provider: openai, model: gpt-5.5, role: fallback}
"""
    reg = _write_registry(tmp_path, body)
    decision = resolve_local_first_selection(
        "analysis.topic_reconciliation", local_first_enabled=False, registry_path=reg
    )
    assert decision.used_local is False
    assert decision.provider == "openai"
    assert decision.model == "gpt-5.5"


# ---------------------------------------------------------------------------
# No regression: a genuinely-paid fallback still uses the registry fallback
# ---------------------------------------------------------------------------


def test_nonlocal_fallback_preference_preserved(tmp_path: Path) -> None:
    """Mirror ocr.slide_image_parse: default=openai, fallback=gemini (both paid)."""
    body = """\
schema_version: 1
tasks:
  ocr.slide_image_parse:
    task_family: ocr
    stage: transcription-ocr
    model_ref: test_gpt54_mini
    fallback_model_ref: test_gemini_flash_lite
    current_research: {required: true}
    benchmark:
      required: true
      candidates: [gemini-3.1-flash-lite-preview]
    quality_threshold: high
    schema_validity_required: true
    provenance_required: true
    local_runtime_policy: deferred
    selection_status: active
    local_proof: {}
    allowed_selections:
      - {provider: openai, model: gpt-5.4-mini, role: default}
      - {provider: gemini, model: gemini-3.1-flash-lite-preview, role: fallback}
"""
    reg = _write_registry(tmp_path, body)
    decision = resolve_local_first_selection(
        "ocr.slide_image_parse", local_first_enabled=False, registry_path=reg
    )
    assert decision.used_local is False
    # Fallback-first preference preserved for non-local fallbacks.
    assert decision.provider == "gemini"
    assert decision.model == "gemini-3.1-flash-lite-preview"


# ---------------------------------------------------------------------------
# Approved-local happy path is unchanged (fix is confined to paid fallback)
# ---------------------------------------------------------------------------


def test_approved_local_happy_path_unchanged(tmp_path: Path) -> None:
    reg = _approved_local_registry(tmp_path)
    decision = resolve_local_first_selection(
        TASK,
        local_first_enabled=True,
        registry_path=reg,
        readiness_check=lambda _task: True,
    )
    assert decision.used_local is True
    assert decision.provider == "local"
    assert decision.role == "approved_local"
    assert decision.fallback_used is False
    # Even on the happy path the recorded paid fallback is non-local.
    assert decision.paid_fallback_provider == "openai"
