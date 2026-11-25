"""Spine-level local-first routing tests (Phase 10 / LOCAL-02, plan 10-02).

These exercise the REAL resolver against fixture registries (injected via a
monkeypatched registry_path) and assert the spine/factory maps the decision to
the correct concrete service/engine. Offline only — no providers are called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import conference_pipeline.model_governance.local_routing as lr
from conference_pipeline.benchmarks.receipt_schema import (
    BenchmarkReceipt,
    receipt_to_dict,
)
from conference_pipeline.config import PipelineSettings
from conference_pipeline.model_governance.registry import load_registry

TRANSCRIPTION = "transcription.session_speech"
OCR = "ocr.slide_image_parse"
VISION = "vision.slide_analysis"
LOOPBACK = "http://localhost:8080/v1"

# Closed allowlist of fields a LocalFallbackReceipt may serialize (T-10-03).
# Mirrors conference_pipeline.provider_receipts.LocalFallbackReceipt. The spine
# must write ONLY these audit fields — never an api_key or a credentialed
# base_url.
_FALLBACK_RECEIPT_KEYS = {
    "schema_version",
    "receipt_type",
    "task_id",
    "stage_id",
    "run_id",
    "requested_provider",
    "requested_model",
    "fallback_provider",
    "fallback_model",
    "fallback_reason",
    "benchmark_receipt_ref",
    "cost_cap_checked",
    "timestamp",
    "receipt_path",
}


def _assert_receipt_no_secret_leak(receipt: dict) -> None:
    """T-10-03: a spine-written fallback receipt leaks no secret/credential.

    The api-key sentinel and the local base-url host appear nowhere in the
    serialized receipt; no api_key/base_url keys are present; the receipt's keys
    are a subset of the LocalFallbackReceipt allowlist (no unexpected field
    slips through); and cost_cap_checked is True (Phase-02 budget cap intact).
    """
    blob = json.dumps(receipt)
    assert "test-key-not-real" not in blob
    assert "localhost:8080" not in blob
    assert "api_key" not in receipt
    assert "base_url" not in receipt
    assert set(receipt) <= _FALLBACK_RECEIPT_KEYS, (
        f"unexpected receipt fields: {set(receipt) - _FALLBACK_RECEIPT_KEYS}"
    )
    assert receipt["cost_cap_checked"] is True


def _local_receipt(tmp_path: Path, task_id: str, task_family: str, stage: str,
                   model_id: str) -> Path:
    receipt_output = tmp_path / f"{task_id}-out.txt"
    receipt_output.write_text("ok", encoding="utf-8")
    receipt = BenchmarkReceipt(
        workspace="test",
        task_id=task_id,
        task_family=task_family,
        stage=stage,
        run_id="00000000-0000-0000-0000-000000000000",
        timestamp="2026-06-25T00:00:00+00:00",
        execution_mode="live",
        provider="local",
        model_id=model_id,
        runtime_id="mlx",
        hardware_profile="mlx:unit-test",
        prompt_digest="a" * 64,
        config_digest="b" * 64,
        input_sample_ref="fixture://test",
        output_ref=str(receipt_output),
        raw_output_snippet="ok",
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
    receipt_path = tmp_path / f"{task_id}-receipt.json"
    receipt_path.write_text(json.dumps(receipt_to_dict(receipt)), encoding="utf-8")
    return receipt_path


# ---------------------------------------------------------------------------
# Fixture registries (transcription shape: provider=openai + fallback=local)
# ---------------------------------------------------------------------------


# Companion catalog covering every model referenced by this file's ad-hoc
# registries (including local test-only identifiers not yet promoted to a
# real allowed_selections in production), so they resolve hermetically.
# registry.py looks for a sibling models.yaml next to a custom registry path
# before falling back to the real production catalog.
_TEST_MODEL_CATALOG_YAML = """\
schema_version: 1
models:
  gpt4o_transcribe_diarize:
    provider: openai
    model: gpt-4o-transcribe-diarize
    status: active
  whisper_local:
    provider: local
    model: whisper-1
    status: active
  gpt54_mini_legacy:
    provider: openai
    model: gpt-5.4-mini
    status: active
  gpt5_mini_legacy:
    provider: openai
    model: gpt-5-mini
    status: active
  gemini_flash_lite:
    provider: gemini
    model: gemini-3.1-flash-lite-preview
    status: active
  test_whisper_mlx_local:
    provider: local
    model: mlx-community/whisper-large-v3-mlx
    status: active
  test_glm_ocr_local:
    provider: local
    model: mlx-community/GLM-OCR-0.9B-4bit
    status: active
  test_qwen_vl_8b_local:
    provider: local
    model: mlx-community/Qwen3-VL-8B-Instruct-4bit
    status: active
"""


def _write_registry(tmp_path: Path, body: str) -> Path:
    (tmp_path / "models.yaml").write_text(_TEST_MODEL_CATALOG_YAML, encoding="utf-8")
    reg = tmp_path / "registry.yaml"
    reg.write_text(body, encoding="utf-8")
    load_registry.cache_clear()
    return reg


def _transcription_deferred_registry(tmp_path: Path) -> Path:
    body = f"""\
schema_version: 1
tasks:
  {TRANSCRIPTION}:
    task_family: transcription
    stage: transcription-ocr
    model_ref: gpt4o_transcribe_diarize
    fallback_model_ref: whisper_local
    current_research: {{required: true}}
    benchmark: {{required: true, candidates: [whisper-1]}}
    quality_threshold: high
    schema_validity_required: false
    provenance_required: true
    local_runtime_policy: deferred
    selection_status: planning_default_requires_stage_receipt
    local_proof: {{}}
    allowed_selections:
      - {{provider: openai, model: gpt-4o-transcribe-diarize, role: default}}
      - {{provider: local, model: whisper-1, role: deferred_local_candidate}}
"""
    return _write_registry(tmp_path, body)


def _transcription_approved_registry(tmp_path: Path) -> Path:
    receipt_output = tmp_path / "out.txt"
    receipt_output.write_text("ok", encoding="utf-8")
    receipt = BenchmarkReceipt(
        workspace="test",
        task_id=TRANSCRIPTION,
        task_family="transcription",
        stage="transcription-ocr",
        run_id="00000000-0000-0000-0000-000000000000",
        timestamp="2026-06-25T00:00:00+00:00",
        execution_mode="live",
        provider="local",
        model_id="mlx-community/whisper-large-v3-mlx",
        runtime_id="mlx",
        hardware_profile="mlx:unit-test",
        prompt_digest="a" * 64,
        config_digest="b" * 64,
        input_sample_ref="fixture://test",
        output_ref=str(receipt_output),
        raw_output_snippet="ok",
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
  {TRANSCRIPTION}:
    task_family: transcription
    stage: transcription-ocr
    model_ref: gpt4o_transcribe_diarize
    fallback_model_ref: whisper_local
    current_research: {{required: true}}
    benchmark:
      required: true
      candidates: [mlx-community/whisper-large-v3-mlx]
      authority_selections:
        - {{provider: local, model: mlx-community/whisper-large-v3-mlx}}
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
      - {{provider: local, model: mlx-community/whisper-large-v3-mlx, role: approved_local}}
"""
    return _write_registry(tmp_path, body)


@pytest.fixture
def use_fixture_registry(monkeypatch):
    """Make the spine's resolver read a fixture registry via registry_path."""

    def _install(reg_path: Path) -> None:
        real = lr.resolve_local_first_selection

        def _patched(task_id, **kw):
            kw.setdefault("registry_path", reg_path)
            return real(task_id, **kw)

        monkeypatch.setattr(lr, "resolve_local_first_selection", _patched)

    return _install


def _settings(**overrides) -> PipelineSettings:
    base = dict(
        transcription_provider="openai",
        openai_api_key="test-key-not-real",
        local_first_enabled=True,
    )
    base.update(overrides)
    return PipelineSettings(**base)


# ---------------------------------------------------------------------------
# Transcription routing (Task 1)
# ---------------------------------------------------------------------------


def test_transcription_used_local_constructs_local_service(
    tmp_path, use_fixture_registry
) -> None:
    from conference_pipeline.transcription import (
        LocalWhisperTranscriptionService,
        create_transcription_service,
    )

    use_fixture_registry(_transcription_approved_registry(tmp_path))
    service = create_transcription_service(
        _settings(), readiness_check=lambda _t: True
    )
    assert isinstance(service, LocalWhisperTranscriptionService)
    assert service.model_name == "mlx-community/whisper-large-v3-mlx"


def test_transcription_switch_off_constructs_paid_openai(
    tmp_path, use_fixture_registry
) -> None:
    from conference_pipeline.transcription import (
        LocalWhisperTranscriptionService,
        OpenAITranscriptionService,
        create_transcription_service,
    )

    use_fixture_registry(_transcription_deferred_registry(tmp_path))
    service = create_transcription_service(
        _settings(local_first_enabled=False), readiness_check=lambda _t: True
    )
    assert isinstance(service, OpenAITranscriptionService)
    assert not isinstance(service, LocalWhisperTranscriptionService)


def test_transcription_missing_approved_local_constructs_paid(
    tmp_path, use_fixture_registry
) -> None:
    from conference_pipeline.transcription import (
        LocalWhisperTranscriptionService,
        OpenAITranscriptionService,
        create_transcription_service,
    )

    use_fixture_registry(_transcription_deferred_registry(tmp_path))
    service = create_transcription_service(
        _settings(), readiness_check=lambda _t: True
    )
    # Even though the registry fallback_provider is local, the deferred local
    # candidate is NOT selected — paid OpenAI is used.
    assert isinstance(service, OpenAITranscriptionService)
    assert not isinstance(service, LocalWhisperTranscriptionService)


def test_transcription_readiness_false_constructs_paid(
    tmp_path, use_fixture_registry
) -> None:
    from conference_pipeline.transcription import (
        LocalWhisperTranscriptionService,
        OpenAITranscriptionService,
        create_transcription_service,
    )

    use_fixture_registry(_transcription_approved_registry(tmp_path))
    service = create_transcription_service(
        _settings(), readiness_check=lambda _t: False
    )
    assert isinstance(service, OpenAITranscriptionService)
    assert not isinstance(service, LocalWhisperTranscriptionService)


def test_transcription_fallback_writes_nonlocal_receipt(
    tmp_path, use_fixture_registry
) -> None:
    from conference_pipeline.provider_receipts import scan_local_fallback_receipts
    from conference_pipeline.transcription import create_transcription_service

    use_fixture_registry(_transcription_deferred_registry(tmp_path))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    create_transcription_service(
        _settings(local_first_enabled=False),
        readiness_check=lambda _t: True,
        workspace=workspace,
        run_id="run-xyz",
        stage_id="transcription-ocr",
    )
    receipts = scan_local_fallback_receipts(workspace)
    assert len(receipts) == 1
    assert receipts[0]["fallback_reason"] == "local_first_disabled"
    assert receipts[0]["fallback_provider"] != "local"
    assert receipts[0]["fallback_provider"] == "openai"
    _assert_receipt_no_secret_leak(receipts[0])


def test_transcription_no_run_context_writes_no_receipt(
    tmp_path, use_fixture_registry
) -> None:
    from conference_pipeline.provider_receipts import scan_local_fallback_receipts
    from conference_pipeline.transcription import create_transcription_service

    use_fixture_registry(_transcription_deferred_registry(tmp_path))
    workspace = tmp_path / "ws2"
    workspace.mkdir()
    # No workspace passed -> resolver stays pure, no receipt.
    create_transcription_service(
        _settings(local_first_enabled=False), readiness_check=lambda _t: True
    )
    assert scan_local_fallback_receipts(workspace) == []


# ---------------------------------------------------------------------------
# OCR routing on the generic NeuripsPipeline spine (Task 2)
# ---------------------------------------------------------------------------


def _ocr_deferred_registry(tmp_path: Path) -> Path:
    body = f"""\
schema_version: 1
tasks:
  {OCR}:
    task_family: ocr
    stage: transcription-ocr
    model_ref: gpt54_mini_legacy
    fallback_model_ref: gemini_flash_lite
    current_research: {{required: true}}
    benchmark: {{required: true, candidates: [gemini-3.1-flash-lite-preview]}}
    quality_threshold: high
    schema_validity_required: true
    provenance_required: true
    local_runtime_policy: deferred
    selection_status: active
    local_proof: {{}}
    allowed_selections:
      - {{provider: openai, model: gpt-5.4-mini, role: default}}
      - {{provider: gemini, model: gemini-3.1-flash-lite-preview, role: fallback}}
      - {{provider: local, model: mlx-community/GLM-OCR-0.9B-4bit, role: deferred_local_candidate}}
"""
    return _write_registry(tmp_path, body)


def _ocr_approved_registry(tmp_path: Path) -> Path:
    receipt_path = _local_receipt(
        tmp_path, OCR, "ocr", "transcription-ocr", "mlx-community/GLM-OCR-0.9B-4bit"
    )
    body = f"""\
schema_version: 1
tasks:
  {OCR}:
    task_family: ocr
    stage: transcription-ocr
    model_ref: gpt54_mini_legacy
    fallback_model_ref: gemini_flash_lite
    current_research: {{required: true}}
    benchmark:
      required: true
      candidates: [mlx-community/GLM-OCR-0.9B-4bit]
      authority_selections:
        - {{provider: local, model: mlx-community/GLM-OCR-0.9B-4bit}}
    quality_threshold: high
    schema_validity_required: true
    provenance_required: true
    local_runtime_policy: approved
    selection_status: active
    local_proof:
      research_ref: "ref/research"
      hardware_fit_ref: "ref/hardware"
      benchmark_receipt_ref: "{receipt_path}"
      operator_approval_ref: "ref/approval"
    allowed_selections:
      - {{provider: openai, model: gpt-5.4-mini, role: default}}
      - {{provider: gemini, model: gemini-3.1-flash-lite-preview, role: fallback}}
      - {{provider: local, model: mlx-community/GLM-OCR-0.9B-4bit, role: approved_local}}
"""
    return _write_registry(tmp_path, body)


def _ocr_settings(**overrides) -> PipelineSettings:
    base = dict(use_llm_ocr=True, openai_api_key="test-key-not-real",
                local_first_enabled=True)
    base.update(overrides)
    return PipelineSettings(**base)


def test_ocr_used_local_constructs_local_engine(tmp_path, use_fixture_registry) -> None:
    from conference_pipeline.ocr_engine import LLMOcrEngine, create_ocr_engine

    use_fixture_registry(_ocr_approved_registry(tmp_path))
    engine = create_ocr_engine(
        _ocr_settings(), readiness_check=lambda _t: True, local_base_url=LOOPBACK
    )
    assert isinstance(engine, LLMOcrEngine)
    assert engine.provider == "local"
    assert engine.local_model_id == "mlx-community/GLM-OCR-0.9B-4bit"


def test_ocr_switch_off_constructs_paid_engine(tmp_path, use_fixture_registry) -> None:
    from conference_pipeline.ocr_engine import LLMOcrEngine, create_ocr_engine

    use_fixture_registry(_ocr_deferred_registry(tmp_path))
    engine = create_ocr_engine(
        _ocr_settings(local_first_enabled=False),
        readiness_check=lambda _t: True,
        local_base_url=LOOPBACK,
    )
    assert isinstance(engine, LLMOcrEngine)
    assert engine.provider == "openai"


def test_ocr_no_approved_local_constructs_paid_with_receipt(
    tmp_path, use_fixture_registry
) -> None:
    from conference_pipeline.ocr_engine import LLMOcrEngine, create_ocr_engine
    from conference_pipeline.provider_receipts import scan_local_fallback_receipts

    use_fixture_registry(_ocr_deferred_registry(tmp_path))
    workspace = tmp_path / "ocr-ws"
    workspace.mkdir()
    engine = create_ocr_engine(
        _ocr_settings(),
        readiness_check=lambda _t: True,
        local_base_url=LOOPBACK,
        workspace=workspace,
        run_id="ocr-run",
        stage_id="transcription-ocr",
    )
    assert isinstance(engine, LLMOcrEngine)
    assert engine.provider == "openai"
    receipts = scan_local_fallback_receipts(workspace)
    assert len(receipts) == 1
    assert receipts[0]["fallback_reason"] == "approved_local_unavailable"
    assert receipts[0]["fallback_provider"] != "local"
    _assert_receipt_no_secret_leak(receipts[0])


def test_ocr_used_local_without_base_url_falls_back_to_paid(
    tmp_path, use_fixture_registry
) -> None:
    """used_local but no derivable loopback URL -> graceful paid path (no crash)."""
    from conference_pipeline.ocr_engine import LLMOcrEngine, create_ocr_engine

    use_fixture_registry(_ocr_approved_registry(tmp_path))
    engine = create_ocr_engine(
        _ocr_settings(), readiness_check=lambda _t: True, local_base_url=None
    )
    assert isinstance(engine, LLMOcrEngine)
    assert engine.provider == "openai"


# ---------------------------------------------------------------------------
# Vision slide-analysis factory routing (Task 3) — inactive-spine surface.
# create_slide_analysis_engine accepts registry_path directly (no monkeypatch).
# ---------------------------------------------------------------------------

LOCAL_VLM_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"


def _vision_deferred_registry(tmp_path: Path) -> Path:
    body = f"""\
schema_version: 1
tasks:
  {VISION}:
    task_family: vision
    stage: presentation
    model_ref: gpt54_mini_legacy
    current_research: {{required: true}}
    benchmark: {{required: true, candidates: [gpt-5-mini]}}
    quality_threshold: high
    schema_validity_required: true
    provenance_required: true
    local_runtime_policy: deferred
    selection_status: active
    local_proof: {{}}
    allowed_selections:
      - {{provider: openai, model: gpt-5.4-mini, role: default}}
      - {{provider: openai, model: gpt-5-mini, role: cheaper_candidate}}
"""
    return _write_registry(tmp_path, body)


def _vision_approved_registry(tmp_path: Path, runtime_kind: str) -> Path:
    receipt_path = _local_receipt(
        tmp_path, VISION, "vision", "presentation", LOCAL_VLM_MODEL
    )
    body = f"""\
schema_version: 1
tasks:
  {VISION}:
    task_family: vision
    stage: presentation
    model_ref: gpt54_mini_legacy
    current_research: {{required: true}}
    benchmark:
      required: true
      candidates: [{LOCAL_VLM_MODEL}]
      authority_selections:
        - {{provider: local, model: {LOCAL_VLM_MODEL}}}
    quality_threshold: high
    schema_validity_required: true
    provenance_required: true
    local_runtime_policy: approved
    selection_status: active
    local_candidates:
      - model_id: "{LOCAL_VLM_MODEL}"
        checked_date: "2026-05-03"
        runtime_support: "mlx_community_tested"
        runtime_kind: {runtime_kind}
        status: benchmark_pending
    local_proof:
      research_ref: "ref/research"
      hardware_fit_ref: "ref/hardware"
      benchmark_receipt_ref: "{receipt_path}"
      operator_approval_ref: "ref/approval"
    allowed_selections:
      - {{provider: openai, model: gpt-5.4-mini, role: default}}
      - {{provider: local, model: {LOCAL_VLM_MODEL}, role: approved_local}}
"""
    return _write_registry(tmp_path, body)


def _vision_settings(**overrides) -> PipelineSettings:
    base = dict(local_first_enabled=True, openai_api_key="test-key-not-real")
    base.update(overrides)
    return PipelineSettings(**base)


def test_vision_vllm_mlx_maps_to_local_vlm(tmp_path) -> None:
    from conference_pipeline.slide_analysis import create_slide_analysis_engine

    reg = _vision_approved_registry(tmp_path, "vllm_mlx")
    engine = create_slide_analysis_engine(
        _vision_settings(),
        readiness_check=lambda _t: True,
        local_base_url=LOOPBACK,
        registry_path=reg,
    )
    assert engine.provider == "local_vlm"
    assert engine.model_id == LOCAL_VLM_MODEL


def test_vision_fastvlm_maps_to_local_fastvlm(tmp_path) -> None:
    from conference_pipeline.slide_analysis import create_slide_analysis_engine

    reg = _vision_approved_registry(tmp_path, "mlx_native_fastvlm")
    engine = create_slide_analysis_engine(
        _vision_settings(),
        readiness_check=lambda _t: True,
        local_base_url=LOOPBACK,
        registry_path=reg,
    )
    assert engine.provider == "local_fastvlm"
    assert engine.model_id == LOCAL_VLM_MODEL


def test_vision_unsupported_runtime_kind_uses_paid_model(tmp_path) -> None:
    """REGRESSION: approved local + unsupported runtime_kind must choose a PAID
    provider AND a paid/default model — never the local model id."""
    from conference_pipeline.slide_analysis import create_slide_analysis_engine

    # mlx_lm is a valid registry runtime_kind but is NOT a slide-analysis
    # vision provider -> must fall back to paid.
    reg = _vision_approved_registry(tmp_path, "mlx_lm")
    engine = create_slide_analysis_engine(
        _vision_settings(),
        readiness_check=lambda _t: True,
        local_base_url=LOOPBACK,
        registry_path=reg,
    )
    assert engine.provider in ("openai", "anthropic")
    assert engine.model_id != LOCAL_VLM_MODEL
    assert engine.model_id == "gpt-5.4-mini"


def test_vision_switch_off_uses_paid(tmp_path) -> None:
    from conference_pipeline.slide_analysis import create_slide_analysis_engine

    reg = _vision_approved_registry(tmp_path, "vllm_mlx")
    engine = create_slide_analysis_engine(
        _vision_settings(local_first_enabled=False),
        readiness_check=lambda _t: True,
        local_base_url=LOOPBACK,
        registry_path=reg,
    )
    assert engine.provider == "openai"
    assert engine.model_id != LOCAL_VLM_MODEL


def test_vision_readiness_false_uses_paid(tmp_path) -> None:
    from conference_pipeline.slide_analysis import create_slide_analysis_engine

    reg = _vision_approved_registry(tmp_path, "vllm_mlx")
    engine = create_slide_analysis_engine(
        _vision_settings(),
        readiness_check=lambda _t: False,
        local_base_url=LOOPBACK,
        registry_path=reg,
    )
    assert engine.provider == "openai"
    assert engine.model_id != LOCAL_VLM_MODEL


def test_vision_no_approved_local_uses_paid(tmp_path) -> None:
    from conference_pipeline.slide_analysis import create_slide_analysis_engine

    reg = _vision_deferred_registry(tmp_path)
    engine = create_slide_analysis_engine(
        _vision_settings(),
        readiness_check=lambda _t: True,
        local_base_url=LOOPBACK,
        registry_path=reg,
    )
    assert engine.provider == "openai"
    assert engine.model_id == "gpt-5.4-mini"


def test_vision_fallback_writes_nonlocal_receipt(tmp_path) -> None:
    from conference_pipeline.provider_receipts import scan_local_fallback_receipts
    from conference_pipeline.slide_analysis import create_slide_analysis_engine

    reg = _vision_deferred_registry(tmp_path)
    workspace = tmp_path / "vision-ws"
    workspace.mkdir()
    create_slide_analysis_engine(
        _vision_settings(),
        readiness_check=lambda _t: True,
        local_base_url=LOOPBACK,
        registry_path=reg,
        workspace=workspace,
        run_id="vision-run",
        stage_id="presentation",
    )
    receipts = scan_local_fallback_receipts(workspace)
    assert len(receipts) == 1
    assert receipts[0]["fallback_reason"] == "approved_local_unavailable"
    assert receipts[0]["fallback_provider"] != "local"
    _assert_receipt_no_secret_leak(receipts[0])


def test_spine_fallback_receipt_closed_schema_no_leak(
    tmp_path, use_fixture_registry
) -> None:
    """T-10-03 end-to-end: the transcription spine writes a fallback receipt
    that carries only allowlisted audit fields — no api_key, no credentialed
    base_url — even though the settings hold a (sentinel) api key and the local
    loopback URL is in play."""
    from conference_pipeline.provider_receipts import scan_local_fallback_receipts
    from conference_pipeline.transcription import create_transcription_service

    use_fixture_registry(_transcription_deferred_registry(tmp_path))
    workspace = tmp_path / "leak-ws"
    workspace.mkdir()
    create_transcription_service(
        _settings(local_first_enabled=False),
        readiness_check=lambda _t: True,
        workspace=workspace,
        run_id="leak-run",
        stage_id="transcription-ocr",
    )
    receipts = scan_local_fallback_receipts(workspace)
    assert len(receipts) == 1
    _assert_receipt_no_secret_leak(receipts[0])


# ---------------------------------------------------------------------------
# Documentation contract: the two Phase-10 limitations are recorded
# ---------------------------------------------------------------------------


def _model_governance_doc() -> str:
    doc = (
        Path(__file__).resolve().parent.parent
        / "docs" / "reference" / "model-governance.md"
    )
    return doc.read_text(encoding="utf-8")


def test_doc_records_iclr_slide_ocr_paid_only_limitation() -> None:
    text = _model_governance_doc()
    assert "paid-only" in text
    assert "create_slide_ocr_service" in text


def test_doc_records_vision_inactive_spine_limitation() -> None:
    text = _model_governance_doc()
    assert "inactive spine" in text.lower()
    assert "vision.slide_analysis" in text
