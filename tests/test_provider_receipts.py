from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.costs.estimator import estimate_workspace
from conference_pipeline.provider_receipts import (
    BLOCKER_RECEIPT_MARKER,
    BLOCKER_RECEIPT_SCHEMA_VERSION,
    ESCALATION_RECEIPT_MARKER,
    ESCALATION_RECEIPT_SCHEMA_VERSION,
    LOCAL_LLM_RECEIPT_MARKER,
    LOCAL_LLM_RECEIPT_SCHEMA_VERSION,
    UNBLOCK_RECEIPT_MARKER,
    UNBLOCK_RECEIPT_SCHEMA_VERSION,
    BlockerClass,
    BlockerReceipt,
    EscalationReceipt,
    LocalLLMReceipt,
    ProviderCallResult,
    UnblockReceipt,
    active_blockers,
    is_receipt_stale,
    provider_receipt_context,
    provider_receipt_summary,
    record_provider_call,
    sanitize_for_receipt,
    scan_blocker_receipts,
    scan_escalation_receipts,
    scan_local_llm_receipts,
    scan_unblock_receipts,
    utc_now,
    write_blocker_receipt,
    write_escalation_receipt,
    write_local_llm_receipt,
    write_unblock_receipt,
)


class Usage:
    def __init__(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class Response:
    def __init__(self, *, usage: Usage | None = None, response_id: str = "resp_123") -> None:
        self.usage = usage
        self.id = response_id
        self.choices = []


class GeminiResponse:
    def __init__(self) -> None:
        self.usage_metadata = {
            "prompt_token_count": 200,
            "candidates_token_count": 50,
            "total_token_count": 250,
        }
        self.text = "{}"


def _load_receipt(result: ProviderCallResult[object]) -> dict[str, object]:
    assert result.receipt.receipt_path is not None
    return json.loads(Path(result.receipt.receipt_path).read_text(encoding="utf-8"))


def _load_local_llm_receipt(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_token_call_writes_sanitized_cost_receipt(tmp_path: Path):
    with provider_receipt_context(
        workspace_root=tmp_path,
        run_id="run-provider-test",
        stage_id="demo-stage",
        caller="tests.fake",
    ):
        result = record_provider_call(
            lambda: Response(
                usage=Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
            ),
            provider="openai",
            model="gpt-5-mini",
            operation="chat.completions",
            request_kwargs={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "secret prompt text"}],
                "OPENAI_API_KEY": "sk-should-not-be-written",
            },
        )

    payload = _load_receipt(result)

    assert payload["receipt_type"] == "conference_pipeline.provider_call_receipt"
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-5-mini"
    assert payload["input_tokens"] == 1_000_000
    assert payload["output_tokens"] == 500_000
    assert payload["estimated_cost_usd"] == pytest.approx(1.25)
    assert payload["cost_status"] == "estimated"
    assert payload["secrets_copied"] is False
    assert "secret prompt text" not in json.dumps(payload)
    assert "sk-should-not-be-written" not in json.dumps(payload)
    assert "storage/workflow-runs/run-provider-test/provider-calls/demo-stage" in str(
        payload["receipt_path"]
    )


def test_audio_minute_and_gemini_usage_pricing(tmp_path: Path):
    with provider_receipt_context(workspace_root=tmp_path, caller="tests.audio"):
        audio = record_provider_call(
            lambda: Response(usage=None),
            provider="openai",
            model="gpt-4o-mini-transcribe",
            operation="audio_transcription",
            audio_minutes=2.0,
        )
        gemini = record_provider_call(
            lambda: GeminiResponse(),
            provider="gemini",
            model="gemini-2.5-flash",
            operation="generate_content",
        )

    audio_payload = _load_receipt(audio)
    gemini_payload = _load_receipt(gemini)

    assert audio_payload["audio_minutes"] == 2.0
    assert audio_payload["estimated_cost_usd"] == pytest.approx(0.006)
    assert gemini_payload["input_tokens"] == 200
    assert gemini_payload["output_tokens"] == 50
    assert gemini_payload["estimated_cost_usd"] == pytest.approx(0.000185)


def test_unknown_pricing_still_writes_warning_receipt(tmp_path: Path):
    with provider_receipt_context(workspace_root=tmp_path, caller="tests.unknown"):
        result = record_provider_call(
            lambda: Response(usage=Usage(prompt_tokens=10, completion_tokens=5)),
            provider="openai",
            model="not-in-catalog",
            operation="chat.completions",
        )

    payload = _load_receipt(result)

    assert payload["status"] == "succeeded"
    assert payload["cost_status"] == "unknown"
    assert payload["estimated_cost_usd"] is None
    assert payload["warnings"]


def test_local_qwen_mlx_receipt_records_zero_api_cost_without_prompt_text(tmp_path: Path):
    with provider_receipt_context(
        workspace_root=tmp_path,
        run_id="local-qwen-test",
        stage_id="exhaustive-topic-extraction",
        caller="tests.local-qwen",
    ):
        result = record_provider_call(
            lambda: Response(usage=Usage(prompt_tokens=1234, completion_tokens=567)),
            provider="local",
            model="mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit",
            operation="mlx_lm.generate",
            request_kwargs={
                "model": "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit",
                "enable_thinking": False,
                "prompt": "do not persist this extraction prompt",
            },
        )

    payload = _load_receipt(result)
    payload_text = json.dumps(payload)

    assert payload["provider"] == "local"
    assert payload["model"] == "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"
    assert payload["operation"] == "mlx_lm.generate"
    assert payload["estimated_cost_usd"] == pytest.approx(0.0)
    assert payload["cost_status"] == "estimated"
    assert payload["warnings"] == []
    assert payload["request_summary"]["enable_thinking"] is False
    assert payload["request_summary"]["prompt"]["type"] == "str"
    assert "do not persist this extraction prompt" not in payload_text


def test_sanitizer_preserves_traceability_ids_and_redacts_only_secrets():
    source_id = "media:oral-31776:cache_acquisition-downloads_media_oral-31776_audio_raw.m4a"
    long_non_secret = "A" * 100
    payload = sanitize_for_receipt(
        {
            "source_artifact_id": source_id,
            "local_path": "conferences/iclr2025/storage/transcription-ocr/media/oral-31776/transcript.txt",
            "signed_url": "https://example.invalid/file?access_token=secret-token",
            # A long opaque-looking but non-secret value (e.g. a paper filename
            # stem) must round-trip intact -- redaction is by secret pattern,
            # never by length.
            "opaque_blob": long_non_secret,
        }
    )

    assert payload["source_artifact_id"] == source_id
    assert payload["local_path"].endswith("oral-31776/transcript.txt")
    assert "secret-token" not in payload["signed_url"]
    assert payload["opaque_blob"] == long_non_secret


def test_sanitizer_redacts_arbitrary_authorization_values_without_exceptions():
    """A CVPR-local receipt repair must not relax the global secret policy."""

    payload = sanitize_for_receipt(
        {
            "authorization": "Bearer arbitrary-fixture-secret",
            "nested": {"consumed_authorization": "another-arbitrary-secret"},
        }
    )

    assert payload == {
        "authorization": "REDACTED",
        "nested": {"consumed_authorization": "REDACTED"},
    }


def test_local_openai_compatible_receipt_records_no_paid_provider_cost(tmp_path: Path):
    with provider_receipt_context(workspace_root=tmp_path, caller="tests.local"):
        result = record_provider_call(
            lambda: Response(usage=Usage(input_tokens=10, output_tokens=5)),
            provider="local-openai-compatible",
            model="mlx-community/Qwen3-8B-4bit",
            operation="chat.completions",
        )

    payload = _load_receipt(result)

    assert payload["status"] == "succeeded"
    assert payload["estimated_cost_usd"] == 0.0
    assert payload["cost_status"] == "local_no_paid_provider"
    assert payload["pricing_refreshed_at"] is None


def test_sanitize_for_receipt_preserves_long_non_secret_paths():
    """A long, human-readable path segment (e.g. a CVPR paper filename that
    encodes the full title) is data, not a secret, and must survive intact --
    a prior blanket length rule rewrote these to ".../REDACTED/..." and
    produced unreachable paper URLs."""

    url = (
        "https://openaccess.thecvf.com/content/CVPR2026/papers/"
        "Xiao_Generalizable_Structure-Aware_Keypoint_Correspondence_for_"
        "Category-Unified_3D_Single_Object_Tracking_CVPR_2026_paper.pdf"
    )

    assert sanitize_for_receipt(url) == url
    # Nested inside a structure, still intact.
    nested = sanitize_for_receipt({"source_urls": [url]})
    assert nested["source_urls"] == [url]


def test_sanitize_for_receipt_still_redacts_secret_query_values_in_long_urls():
    """Removing the length rule must not weaken precise secret redaction: an
    access_token in a query string is still scrubbed even on a long URL."""

    url = (
        "https://openaccess.thecvf.com/content/CVPR2026/papers/"
        "Some_Very_Long_Real_Paper_Filename_That_Exceeds_Eighty_Characters_CVPR_2026_paper.pdf"
        "?access_token=super-secret-value"
    )
    sanitized = sanitize_for_receipt(url)

    assert "super-secret-value" not in sanitized
    assert "Some_Very_Long_Real_Paper_Filename" in sanitized


def test_provider_receipt_summary_and_actual_mode_use_canonical_receipts(tmp_path: Path):
    with provider_receipt_context(
        workspace_root=tmp_path,
        run_id="summary-run",
        stage_id="summary-stage",
        caller="tests.summary",
    ):
        record_provider_call(
            lambda: Response(usage=Usage(prompt_tokens=1000, completion_tokens=1000)),
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            operation="messages",
        )

    summary = provider_receipt_summary(tmp_path, "summary-run", "summary-stage")
    report = estimate_workspace(tmp_path, mode="actual")

    assert summary["canonical_receipt_files_scanned"] == 1
    assert summary["canonical_totals_by_provider"]["anthropic"] == pytest.approx(0.006)
    assert report.actual_usage is not None
    assert report.actual_usage["canonical_receipt_files_scanned"] == 1
    assert report.actual_usage["canonical_total_estimated_cost_usd"] == pytest.approx(0.006)


def test_aggregate_excludes_failed_call_estimated_cost_but_counts_it():
    """A failed provider call carries a duration/token cost PROJECTION, but no
    charge was incurred (a rejected request never processes its input). The
    aggregate canonical cost must exclude it while still tallying it under
    ``failed_receipts`` for audit — otherwise cumulative-spend cap checks can
    false-halt on money that was never spent."""

    from conference_pipeline.provider_receipts import aggregate_provider_call_receipts

    receipts = [
        {"provider": "openai", "model": "gpt-4o-transcribe-diarize",
         "operation": "audio_transcription", "status": "succeeded", "estimated_cost_usd": 0.03},
        {"provider": "openai", "model": "gpt-4o-transcribe-diarize",
         "operation": "audio_transcription", "status": "failed", "estimated_cost_usd": 0.09},
        {"provider": "openai", "model": "gpt-4o-transcribe-diarize",
         "operation": "audio_transcription", "status": "failed", "estimated_cost_usd": 0.05},
    ]
    agg = aggregate_provider_call_receipts(receipts)

    assert agg["canonical_total_estimated_cost_usd"] == pytest.approx(0.03)
    assert agg["canonical_totals_by_provider"]["openai"] == pytest.approx(0.03)
    assert agg["failed_receipts"] == 2
    assert agg["canonical_receipt_files_scanned"] == 3


# ---------------------------------------------------------------------------
# LocalLLMReceipt — LOCAL-LLM-03, TRACE-PRESERVE-01
# ---------------------------------------------------------------------------


def _make_local_llm_receipt(**kwargs) -> LocalLLMReceipt:
    defaults = dict(
        schema_version=LOCAL_LLM_RECEIPT_SCHEMA_VERSION,
        receipt_type=LOCAL_LLM_RECEIPT_MARKER,
        endpoint_label="ollama-local",
        model="qwen3-8b",
        request_count=0,
        token_count=None,
        confidence_path="heuristic_only",
        skipped_record_count=0,
        paid_cloud_cost=0.0,
        timestamp="2026-04-30T12:00:00Z",
        run_id=None,
        stage_id=None,
        preflight_mode=False,
        quality_sample_mode=False,
        effective_thresholds=None,
    )
    defaults.update(kwargs)
    return LocalLLMReceipt(**defaults)


def test_local_llm_receipt_written_to_disk(tmp_path: Path):
    receipt = _make_local_llm_receipt(preflight_mode=True)
    receipt_path = write_local_llm_receipt(receipt, tmp_path)

    assert receipt_path.exists()
    payload = _load_local_llm_receipt(receipt_path)

    assert payload["receipt_type"] == "conference_pipeline.local_llm_receipt"
    assert payload["paid_cloud_cost"] == 0.0
    assert payload["preflight_mode"] is True
    assert payload["endpoint_label"] == "ollama-local"
    # URL must never be stored
    assert "localhost" not in json.dumps(payload)
    assert "127.0.0.1" not in json.dumps(payload)


def test_local_llm_receipt_redacts_secrets(tmp_path: Path):
    # LocalLLMReceipt has no quality_metadata field — test sanitize_for_receipt
    # directly on a dict that would be produced by to_dict(), injecting a secret key.
    receipt = _make_local_llm_receipt()
    payload = receipt.to_dict()
    # Inject a secret-named key into the payload and re-sanitize (mirrors how
    # sanitize_for_receipt would handle any secret-keyed value in to_dict output)
    payload_with_secret = dict(payload)
    payload_with_secret["OPENAI_API_KEY"] = "sk-fake-secret"
    sanitized = sanitize_for_receipt(payload_with_secret)

    assert "sk-fake-secret" not in json.dumps(sanitized)
    assert sanitized["OPENAI_API_KEY"] == "REDACTED"


def test_local_llm_receipt_preserves_source_ids(tmp_path: Path):
    receipt = _make_local_llm_receipt(run_id="run-123", stage_id="exhaustive-extraction")
    receipt_path = write_local_llm_receipt(receipt, tmp_path)
    payload = _load_local_llm_receipt(receipt_path)

    assert payload["run_id"] == "run-123"
    assert payload["stage_id"] == "exhaustive-extraction"


def test_local_llm_receipt_endpoint_label_not_redacted(tmp_path: Path):
    receipt = _make_local_llm_receipt(endpoint_label="ollama-local")
    receipt_path = write_local_llm_receipt(receipt, tmp_path)
    payload = _load_local_llm_receipt(receipt_path)

    # Plain labels like "ollama-local" must pass through sanitize_for_receipt unchanged
    assert payload["endpoint_label"] == "ollama-local"
    assert "ollama-local" in json.dumps(payload)


def test_local_llm_receipt_supports_backend_endpoint_schemes(tmp_path: Path):
    mlx_receipt = _make_local_llm_receipt(
        endpoint_label="mlx://mlx-community/Qwen3-8B-4bit",
        timestamp="2026-04-30T12:00:01Z",
    )
    http_receipt = _make_local_llm_receipt(
        endpoint_label="http://localhost:11434/v1",
        timestamp="2026-04-30T12:00:02Z",
    )

    mlx_path = write_local_llm_receipt(mlx_receipt, tmp_path)
    http_path = write_local_llm_receipt(http_receipt, tmp_path)

    assert _load_local_llm_receipt(mlx_path)["endpoint_label"].startswith("mlx://")
    assert _load_local_llm_receipt(http_path)["endpoint_label"].startswith(
        "http://localhost"
    )


# ---------------------------------------------------------------------------
# EscalationReceipt — CLOUD-ESC-02
# ---------------------------------------------------------------------------


def test_escalation_receipt_structure(tmp_path: Path):
    with provider_receipt_context(
        workspace_root=tmp_path,
        run_id="run-esc-test",
        stage_id="escalation-stage",
        caller="tests.escalation",
    ):
        result = record_provider_call(
            lambda: Response(usage=Usage(input_tokens=10, output_tokens=5)),
            provider="local-openai-compatible",
            model="qwen3-8b",
            operation="chat.completions",
        )

    escalation = EscalationReceipt(
        schema_version=ESCALATION_RECEIPT_SCHEMA_VERSION,
        receipt_type=ESCALATION_RECEIPT_MARKER,
        wrapped_receipt=result.receipt,
        escalation_reason="low_confidence_after_local_llm",
        affected_record_ids=("rec-001", "rec-002"),
        fail_closed_state=True,
        timestamp=utc_now(),
    )
    receipt_path = write_escalation_receipt(escalation, tmp_path)
    payload = _load_local_llm_receipt(receipt_path)

    assert payload["receipt_type"] == "conference_pipeline.escalation_receipt"
    assert payload["escalation_reason"] == "low_confidence_after_local_llm"
    assert payload["fail_closed_state"] is True
    assert isinstance(payload["affected_record_ids"], list)
    assert "rec-001" in payload["affected_record_ids"]
    assert "rec-002" in payload["affected_record_ids"]
    assert "wrapped_receipt" in payload
    assert payload["wrapped_receipt"]["receipt_type"] == "conference_pipeline.provider_call_receipt"


def test_escalation_receipt_redacts_nested_secrets(tmp_path: Path):
    # Build a ProviderCallReceipt, then wrap it in EscalationReceipt
    # and verify no secrets appear in the serialized output.
    with provider_receipt_context(
        workspace_root=tmp_path,
        run_id="run-esc-redact",
        stage_id="redact-stage",
        caller="tests.escalation.redact",
    ):
        result = record_provider_call(
            lambda: Response(usage=Usage(input_tokens=5, output_tokens=3)),
            provider="local-openai-compatible",
            model="qwen3-8b",
            operation="chat.completions",
            request_kwargs={
                "model": "qwen3-8b",
                "OPENAI_API_KEY": "sk-fake-api-key",
            },
        )

    escalation = EscalationReceipt(
        schema_version=ESCALATION_RECEIPT_SCHEMA_VERSION,
        receipt_type=ESCALATION_RECEIPT_MARKER,
        wrapped_receipt=result.receipt,
        escalation_reason="low_confidence",
        affected_record_ids=("rec-x",),
        fail_closed_state=True,
        timestamp=utc_now(),
    )

    dumped = json.dumps(escalation.to_dict())
    assert "sk-fake-api-key" not in dumped
    assert "low_confidence" in dumped
    assert "rec-x" in dumped


# ---------------------------------------------------------------------------
# Scan helpers — LOCAL-LLM-03, CLOUD-ESC-02
# ---------------------------------------------------------------------------


def test_scan_local_llm_receipts_finds_written_receipt(tmp_path: Path):
    receipt = _make_local_llm_receipt(preflight_mode=True)
    write_local_llm_receipt(receipt, tmp_path)

    result = scan_local_llm_receipts(tmp_path)

    assert len(result) == 1
    assert result[0]["receipt_type"] == LOCAL_LLM_RECEIPT_MARKER


def test_scan_escalation_receipts_finds_written_receipt(tmp_path: Path):
    with provider_receipt_context(
        workspace_root=tmp_path,
        run_id="run-scan-esc",
        stage_id="scan-stage",
        caller="tests.scan",
    ):
        result = record_provider_call(
            lambda: Response(usage=Usage(input_tokens=5, output_tokens=2)),
            provider="local-openai-compatible",
            model="qwen3-8b",
            operation="chat.completions",
        )

    escalation = EscalationReceipt(
        schema_version=ESCALATION_RECEIPT_SCHEMA_VERSION,
        receipt_type=ESCALATION_RECEIPT_MARKER,
        wrapped_receipt=result.receipt,
        escalation_reason="scan_test",
        affected_record_ids=("rec-scan",),
        fail_closed_state=False,
        timestamp=utc_now(),
    )
    write_escalation_receipt(escalation, tmp_path)

    found = scan_escalation_receipts(tmp_path)

    assert len(found) == 1
    assert found[0]["receipt_type"] == ESCALATION_RECEIPT_MARKER


def test_blocker_and_unblock_receipts_round_trip(tmp_path: Path):
    blocker = BlockerReceipt(
        schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
        receipt_type=BLOCKER_RECEIPT_MARKER,
        blocker_class=BlockerClass.LOCAL_LLM_NOT_READY,
        stage_id="discover",
        reason="local endpoint unreachable",
        remediation_path="EXTRACTION_LOCAL_LLM_BASE_URL",
        automated_unblock_available=False,
        endpoint_label="http://localhost:11434/v1",
        timestamp="2026-04-30T12:00:03Z",
        run_id="run-blocked",
    )
    blocker_path = write_blocker_receipt(blocker, tmp_path)

    unblock = UnblockReceipt(
        schema_version=UNBLOCK_RECEIPT_SCHEMA_VERSION,
        receipt_type=UNBLOCK_RECEIPT_MARKER,
        blocker_receipt_path=str(blocker_path),
        unblock_method="manual_ack",
        operator_id="operator-1",
        note="confirmed retry after server restart",
        timestamp="2026-04-30T12:00:04Z",
        run_id="run-blocked",
    )
    write_unblock_receipt(unblock, tmp_path)

    assert len(scan_blocker_receipts(tmp_path)) == 1
    assert len(scan_unblock_receipts(tmp_path)) == 1
    assert active_blockers(tmp_path, run_id="run-blocked") == []


def test_active_blockers_matches_project_relative_blocker_paths(
    tmp_path: Path,
    monkeypatch,
):
    project = tmp_path / "project"
    workspace_root = project / "conferences" / "iclr2026"
    workspace_root.mkdir(parents=True)
    monkeypatch.chdir(project)
    workspace = Path("conferences/iclr2026")
    blocker = BlockerReceipt(
        schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
        receipt_type=BLOCKER_RECEIPT_MARKER,
        blocker_class=BlockerClass.STAGE_BLOCKED,
        stage_id="download-corpus",
        reason="stale failed attempt",
        remediation_path="rerun download-corpus",
        automated_unblock_available=False,
        endpoint_label=None,
        timestamp="2026-04-30T12:00:03Z",
        run_id="run-blocked",
    )
    blocker_path = write_blocker_receipt(blocker, workspace)
    write_unblock_receipt(
        UnblockReceipt(
            schema_version=UNBLOCK_RECEIPT_SCHEMA_VERSION,
            receipt_type=UNBLOCK_RECEIPT_MARKER,
            blocker_receipt_path=blocker_path.relative_to(workspace).as_posix(),
            unblock_method="manual_ack",
            operator_id="operator-1",
            note="acknowledged after rerun",
            timestamp="2026-04-30T12:00:04Z",
            run_id="run-blocked",
        ),
        workspace,
    )

    assert active_blockers(workspace, run_id="run-blocked") == []


def test_active_blockers_returns_only_unacknowledged_receipts(tmp_path: Path):
    active = BlockerReceipt(
        schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
        receipt_type=BLOCKER_RECEIPT_MARKER,
        blocker_class=BlockerClass.THRESHOLD_VIOLATION,
        stage_id="analyze",
        reason="thresholds failed",
        remediation_path="config/extraction-thresholds.json",
        automated_unblock_available=False,
        endpoint_label=None,
        timestamp="2026-04-30T12:00:05Z",
        run_id="run-active",
    )
    acknowledged = BlockerReceipt(
        schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
        receipt_type=BLOCKER_RECEIPT_MARKER,
        blocker_class=BlockerClass.MISSING_RECEIPT,
        stage_id="download",
        reason="missing required receipt",
        remediation_path="storage/workflow-runs/<run-id>/",
        automated_unblock_available=False,
        endpoint_label=None,
        timestamp="2026-04-30T12:00:06Z",
        run_id="run-active",
    )
    active_path = write_blocker_receipt(active, tmp_path)
    acknowledged_path = write_blocker_receipt(acknowledged, tmp_path)
    write_unblock_receipt(
        UnblockReceipt(
            schema_version=UNBLOCK_RECEIPT_SCHEMA_VERSION,
            receipt_type=UNBLOCK_RECEIPT_MARKER,
            blocker_receipt_path=str(acknowledged_path),
            unblock_method="manual_ack",
            operator_id="operator-2",
            note=None,
            timestamp="2026-04-30T12:00:07Z",
            run_id="run-active",
        ),
        tmp_path,
    )

    blockers = active_blockers(tmp_path, run_id="run-active")

    assert len(blockers) == 1
    assert blockers[0]["receipt_path"] == str(active_path)


def test_is_receipt_stale_uses_timestamp_age():
    stale_receipt = {"timestamp": "2026-04-01T00:00:00Z"}
    fresh_receipt = {"timestamp": utc_now()}

    assert is_receipt_stale(stale_receipt, max_age_seconds=60) is True
    assert is_receipt_stale(fresh_receipt, max_age_seconds=86_400) is False


# ---------------------------------------------------------------------------
# v3.0 03-04: cost-avoided fields on local receipts (LOCAL-04 / D-13..D-16)
# ---------------------------------------------------------------------------


def test_local_receipts_record_benchmark_ref_and_cost_avoided(tmp_path: Path) -> None:
    """LOCAL-04: local embedding/vision receipts can carry benchmark_receipt_ref,
    cost_avoided_usd, cost_avoided_status, paid_counterfactual_provider, and
    paid_counterfactual_model fields when provided."""
    from conference_pipeline.provider_receipts import (
        LOCAL_EMBEDDING_RECEIPT_MARKER,
        record_local_embedding_call,
        record_local_vision_call,
    )

    # Local embedding receipt with cost-avoided stamped at call time.
    embed_path = record_local_embedding_call(
        provider="mlx",
        model="mlx-community/bge-m3-mlx-8bit",
        embedding_dim=1024,
        endpoint_label="mlx://mlx-community/bge-m3-mlx-8bit",
        input_count=5,
        total_input_chars=2000,
        request_count=1,
        cost_usd=0.0,
        workspace=tmp_path,
        benchmark_receipt_ref="storage/benchmarks/embed-bge-m3-20260625.json",
        cost_avoided_usd=0.0024,
        cost_avoided_status="priced",
        paid_counterfactual_provider="openai",
        paid_counterfactual_model="text-embedding-3-small",
        needs_pricing_backfill=False,
    )
    assert embed_path is not None and embed_path.exists()
    embed_payload = json.loads(embed_path.read_text())
    assert embed_payload["benchmark_receipt_ref"] == "storage/benchmarks/embed-bge-m3-20260625.json"
    assert embed_payload["cost_avoided_usd"] == pytest.approx(0.0024)
    assert embed_payload["cost_avoided_status"] == "priced"
    assert embed_payload["paid_counterfactual_provider"] == "openai"
    assert embed_payload["paid_counterfactual_model"] == "text-embedding-3-small"
    assert embed_payload["needs_pricing_backfill"] is False

    # Local vision receipt with cost-avoided stamped at call time.
    vision_path = record_local_vision_call(
        provider="local_vlm",
        runtime_kind="mlx_vlm",
        model="Qwen3-VL-8B-Instruct-4bit",
        endpoint_label="mlx-vision://Qwen3-VL-8B-Instruct-4bit",
        vision_task="ocr.slide_image_parse",
        image_count=1,
        image_dimensions=[[1920, 1080]],
        image_bytes_size=50000,
        workspace=tmp_path,
        benchmark_receipt_ref="storage/benchmarks/vlm-qwen3-ocr-20260625.json",
        cost_avoided_usd=0.05,
        cost_avoided_status="priced",
        paid_counterfactual_provider="openai",
        paid_counterfactual_model="gpt-4o",
        needs_pricing_backfill=False,
    )
    assert vision_path is not None and vision_path.exists()
    vision_payload = json.loads(vision_path.read_text())
    assert vision_payload["benchmark_receipt_ref"] == "storage/benchmarks/vlm-qwen3-ocr-20260625.json"
    assert vision_payload["cost_avoided_usd"] == pytest.approx(0.05)
    assert vision_payload["cost_avoided_status"] == "priced"
    assert vision_payload["paid_counterfactual_provider"] == "openai"
    assert vision_payload["needs_pricing_backfill"] is False


def test_unknown_counterfactual_cost_is_flagged_not_zero(tmp_path: Path) -> None:
    """D-16: when paid counterfactual has no pricing, cost_avoided_usd is None and
    cost_avoided_status is 'unknown' with needs_pricing_backfill=True.
    Savings must never be fabricated as $0 or guessed."""
    from conference_pipeline.costs.cost_avoided import estimate_cost_avoided

    # Task ID with no catalog entry for the paid counterfactual
    result = estimate_cost_avoided(
        "no_such_task_id_xyz",
        paid_provider="openai",
        paid_model="not-in-catalog-model",
    )

    assert result["cost_avoided_status"] == "unknown"
    assert result["cost_avoided_usd"] is None
    assert result["needs_pricing_backfill"] is True

    # Embedding receipt with unknown pricing
    from conference_pipeline.provider_receipts import record_local_embedding_call
    embed_path = record_local_embedding_call(
        provider="mlx",
        model="mlx-community/bge-m3-mlx-8bit",
        embedding_dim=1024,
        endpoint_label="mlx://mlx-community/bge-m3-mlx-8bit",
        input_count=3,
        total_input_chars=900,
        request_count=1,
        cost_usd=0.0,
        workspace=tmp_path,
        cost_avoided_usd=None,
        cost_avoided_status="unknown",
        needs_pricing_backfill=True,
    )
    assert embed_path is not None
    embed_payload = json.loads(embed_path.read_text())
    assert embed_payload["cost_avoided_usd"] is None
    assert embed_payload["cost_avoided_status"] == "unknown"
    assert embed_payload["needs_pricing_backfill"] is True
    # Must not have saved $0 as savings
    assert embed_payload.get("cost_avoided_usd") != 0.0


# ---------------------------------------------------------------------------
# Phase 02-01: task_id threading, Tavily credits, TTS minutes (COST-02, COST-03)
# ---------------------------------------------------------------------------


def test_task_id_threading(tmp_path: Path):
    """Task context active -> receipt JSON records task_id."""
    with provider_receipt_context(workspace_root=tmp_path, task_id="t-01"):
        result = record_provider_call(
            lambda: Response(usage=Usage(prompt_tokens=100, completion_tokens=50)),
            provider="openai",
            model="gpt-4o-mini",
            operation="chat.completions",
        )

    payload = _load_receipt(result)

    assert payload["task_id"] == "t-01"


def test_task_id_null_when_no_context(tmp_path: Path):
    """No task context -> receipt JSON task_id is null (unattributed)."""
    with provider_receipt_context(workspace_root=tmp_path):
        result = record_provider_call(
            lambda: Response(usage=Usage(prompt_tokens=10, completion_tokens=5)),
            provider="openai",
            model="gpt-4o-mini",
            operation="chat.completions",
        )

    payload = _load_receipt(result)

    assert payload["task_id"] is None


def test_task_id_env_mirror(tmp_path: Path, monkeypatch):
    """PROVIDER_RECEIPT_TASK_ID env var is picked up by subprocess-style context."""
    import os
    from conference_pipeline.provider_receipts import _env_context

    monkeypatch.setenv("PROVIDER_RECEIPT_TASK_ID", "env-task-01")
    ctx = _env_context()
    assert ctx.task_id == "env-task-01"


def test_vision_symbols_exported_from_all():
    """Local-vision receipt symbols must be in provider_receipts.__all__."""
    from conference_pipeline import provider_receipts

    for name in [
        "LOCAL_VISION_RECEIPT_MARKER",
        "LOCAL_VISION_RECEIPT_SCHEMA_VERSION",
        "LocalVisionCallReceipt",
        "scan_local_vision_receipts",
        "write_local_vision_receipt",
    ]:
        assert name in provider_receipts.__all__, f"{name} missing from __all__"



def test_tts_minutes_receipt(tmp_path: Path):
    """OpenAI TTS call emits receipt with tts_minutes and non-null cost."""
    with provider_receipt_context(workspace_root=tmp_path):
        result = record_provider_call(
            lambda: b"audio-bytes",
            provider="openai",
            model="gpt-4o-mini-tts",
            operation="audio.speech",
            tts_minutes=0.5,
        )

    payload = _load_receipt(result)

    assert payload["tts_minutes"] == pytest.approx(0.5)
    assert payload["estimated_cost_usd"] is not None
    assert payload["estimated_cost_usd"] > 0


# ---------------------------------------------------------------------------
# Phase 02-01 Task 2 RED: Tavily and OpenAI TTS metering (COST-03)
# ---------------------------------------------------------------------------


def test_tavily_receipt(tmp_path: Path, monkeypatch):
    """Mocked Tavily search emits receipt with provider/credits/cost_status."""
    from conference_pipeline.research.researcher import TavilyHTTPClient

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"title": "T1", "url": "https://example.com", "content": "c1", "score": 0.9}
                ],
                "credits_used": 2,
            }

    monkeypatch.setattr(
        "requests.post",
        lambda *a, **kw: FakeResponse(),
    )

    with provider_receipt_context(workspace_root=tmp_path, task_id="t-tavily"):
        client = TavilyHTTPClient(api_key="fake-key")
        results = client.search("test query")

    assert results, "Expected non-empty results"

    receipts_dir = tmp_path / "storage"
    receipt_files = list(receipts_dir.rglob("*.json"))
    assert receipt_files, "No receipt written"

    import json as _json
    receipt = _json.loads(receipt_files[0].read_text(encoding="utf-8"))

    assert receipt["provider"] == "tavily"
    assert receipt["operation"] == "search"
    assert isinstance(receipt["credits"], float)
    assert receipt["cost_status"] in {"priced", "estimated", "unknown"}
    assert receipt["task_id"] == "t-tavily"
    # Threat model T-02-01: no secret material in receipt
    receipt_text = _json.dumps(receipt)
    assert "fake-key" not in receipt_text


# ---------------------------------------------------------------------------
# LOCAL-03 Task 3: paid embedding receipt (D-10) — record_local_embedding_call
# ---------------------------------------------------------------------------


def test_openai_embedding_receipt_records_paid_cost_status(tmp_path: Path):
    """D-10: a paid embedding receipt has provider='openai' and cost_status in
    {'estimated', 'unknown'} and is found by scan_local_embedding_receipts.

    T-03-03 mitigation: receipt records provider/model/dim/cost_status only —
    no API key or secret material.
    """
    from conference_pipeline.provider_receipts import (
        LocalEmbeddingReceipt,
        record_local_embedding_call,
        scan_local_embedding_receipts,
    )

    # Record a paid embedding call (D-10 addition: cost_status field).
    receipt_path = record_local_embedding_call(
        provider="openai",
        model="text-embedding-3-small",
        embedding_dim=1536,
        endpoint_label="openai://text-embedding-3-small",
        input_count=10,
        total_input_chars=500,
        request_count=1,
        cost_usd=0.001,
        workspace=tmp_path,
        stage_id="test-embedding-stage",
        cost_status="estimated",
    )
    assert receipt_path is not None

    # The receipt should be scannable.
    found = scan_local_embedding_receipts(tmp_path)
    assert len(found) == 1, f"Expected 1 receipt, got {len(found)}: {found}"

    receipt = found[0]
    assert receipt["provider"] == "openai"
    assert receipt["cost_status"] in {"estimated", "unknown"}, (
        f"cost_status must be 'estimated' or 'unknown', got {receipt['cost_status']!r}"
    )
    assert receipt["model"] == "text-embedding-3-small"
    assert receipt["embedding_dim"] == 1536

    # T-03-03: no secret material in receipt.
    receipt_text = json.dumps(receipt)
    assert "sk-" not in receipt_text
    assert "OPENAI_API_KEY" not in receipt_text


def test_local_embedding_receipt_cost_status_defaults_to_free(tmp_path: Path):
    """Existing local mlx/ollama receipts still work; cost_status defaults to 'free'."""
    from conference_pipeline.provider_receipts import (
        record_local_embedding_call,
        scan_local_embedding_receipts,
    )

    record_local_embedding_call(
        provider="mlx",
        model="mlx-community/bge-m3-mlx-8bit",
        embedding_dim=1024,
        endpoint_label="mlx://mlx-community/bge-m3-mlx-8bit",
        input_count=5,
        total_input_chars=250,
        request_count=1,
        cost_usd=0.0,
        workspace=tmp_path,
        stage_id="test-mlx-embedding",
        # no cost_status arg — should default gracefully
    )
    found = scan_local_embedding_receipts(tmp_path)
    assert len(found) == 1
    receipt = found[0]
    assert receipt["provider"] == "mlx"
    # cost_status should be present; defaults to "free" or absent for local providers.
    assert receipt.get("cost_status", "free") in {"free", "estimated", "unknown", None}


# ---------------------------------------------------------------------------
# Task 2 (03-02): LocalFallbackReceipt family tests
# ---------------------------------------------------------------------------


def test_record_local_fallback_receipt_is_sanitized_and_scannable(tmp_path: Path) -> None:
    """record_local_fallback writes a receipt that scan_local_fallback_receipts returns
    with marker match, receipt_path attached, and no secrets in the payload. (T-03-02/T-03-05)"""
    from conference_pipeline.provider_receipts import (
        record_local_fallback,
        scan_local_fallback_receipts,
    )

    record_local_fallback(
        task_id="test.task",
        stage_id="test-stage",
        run_id="run-fallback-test",
        requested_provider="local",
        requested_model="mlx-community/TestModel-4bit",
        fallback_provider="anthropic",
        fallback_model="claude-sonnet-4-6-20251001",
        fallback_reason="local_readiness_failed",
        benchmark_receipt_ref="receipts/benchmark.json",
        workspace=tmp_path,
    )

    found = scan_local_fallback_receipts(tmp_path)
    assert len(found) == 1
    receipt = found[0]

    # Marker must match
    assert receipt.get("receipt_type") == "local-fallback-receipt"

    # Required fields
    assert receipt["task_id"] == "test.task"
    assert receipt["fallback_reason"] == "local_readiness_failed"
    assert receipt["fallback_provider"] == "anthropic"
    assert receipt["cost_cap_checked"] is True

    # receipt_path must be attached
    assert receipt.get("receipt_path"), "receipt_path must be attached by scanner"

    # Sanitization: no api_key, authorization, bearer strings
    raw_text = json.dumps(receipt)
    for secret_fragment in ("api_key", "authorization", "Bearer ", "bearer "):
        # Keys named with secret fragments will be REDACTED values; the fragment
        # must not appear as a plain value
        assert secret_fragment.lower() not in raw_text.lower().replace(
            "authorization", ""
        ).replace("api_key", "").replace("bearer", ""), (
            f"Secret fragment {secret_fragment!r} leaked into receipt: {raw_text[:200]}"
        )


# --- the receipt context is shared by threads: its environment must not leak ------------------

_RECEIPT_ENV = (
    "PROVIDER_RECEIPT_WORKSPACE",
    "PROVIDER_RECEIPT_RUN_ID",
    "PROVIDER_RECEIPT_STAGE_ID",
    "PROVIDER_RECEIPT_CALLER",
    "PROVIDER_RECEIPT_TASK_ID",
)


@pytest.fixture
def clean_receipt_env(monkeypatch):
    for key in _RECEIPT_ENV:
        monkeypatch.delenv(key, raising=False)


def _receipt_env() -> dict[str, str | None]:
    import os

    return {key: os.environ.get(key) for key in _RECEIPT_ENV}


def test_contexts_that_exit_out_of_order_leave_no_task_id_behind(clean_receipt_env) -> None:
    """The interleaving that leaked 'presentation.figure_spec_critic' into later tests and, in a
    real run, into later provider calls: A enters, B enters, A exits, B exits. Each context used to
    save and restore 'the previous value', so B restored A's task id and it stuck."""

    import threading

    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen_while_b_alone: list[str | None] = []

    def thread_a() -> None:
        with provider_receipt_context(task_id="task-A"):
            a_in.set()
            b_in.wait(5)
        a_out.set()

    def thread_b() -> None:
        a_in.wait(5)
        with provider_receipt_context(task_id="task-B"):
            b_in.set()
            a_out.wait(5)
            seen_while_b_alone.append(_receipt_env()["PROVIDER_RECEIPT_TASK_ID"])

    threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert seen_while_b_alone == ["task-B"], "the context still active keeps its own attribution"
    assert _receipt_env() == dict.fromkeys(_RECEIPT_ENV), "nothing is left behind once both exit"


def test_many_threads_entering_and_exiting_in_any_order_restore_the_original_environment(monkeypatch) -> None:
    import random
    import threading
    import time

    # Whatever the environment already holds (other tests' scripts set these keys process-wide)
    # must come back exactly: the invariant is restoration, not emptiness.
    monkeypatch.setenv("PROVIDER_RECEIPT_RUN_ID", "ambient-run")
    monkeypatch.setenv("PROVIDER_RECEIPT_STAGE_ID", "ambient-stage")
    monkeypatch.delenv("PROVIDER_RECEIPT_TASK_ID", raising=False)
    original = _receipt_env()
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        rng = random.Random(index)
        try:
            for round_ in range(20):
                with provider_receipt_context(task_id=f"task-{index}", stage_id=f"stage-{index}"):
                    time.sleep(rng.random() * 0.002)
                    assert _receipt_env()["PROVIDER_RECEIPT_TASK_ID"] is not None
                time.sleep(rng.random() * 0.001)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert not errors, errors
    assert _receipt_env() == original, "every key is back to what it was before the first context entered"
    assert original["PROVIDER_RECEIPT_RUN_ID"] == "ambient-run" and original["PROVIDER_RECEIPT_STAGE_ID"] == "ambient-stage"


def test_nested_contexts_in_one_thread_behave_as_before(clean_receipt_env, tmp_path: Path) -> None:
    with provider_receipt_context(workspace_root=tmp_path, run_id="run-1", task_id="outer"):
        outer = _receipt_env()
        with provider_receipt_context(task_id="inner", stage_id="inner-stage"):
            inner = _receipt_env()
        after_inner = _receipt_env()
    assert outer["PROVIDER_RECEIPT_TASK_ID"] == "outer" and outer["PROVIDER_RECEIPT_RUN_ID"] == "run-1"
    assert inner["PROVIDER_RECEIPT_TASK_ID"] == "inner" and inner["PROVIDER_RECEIPT_STAGE_ID"] == "inner-stage"
    assert inner["PROVIDER_RECEIPT_RUN_ID"] == "run-1", "an inner context inherits what it does not set"
    assert after_inner == outer, "leaving the inner context restores the outer one exactly"
    assert _receipt_env() == dict.fromkeys(_RECEIPT_ENV)


def test_an_exception_inside_a_context_still_restores_the_environment(clean_receipt_env) -> None:
    with pytest.raises(RuntimeError):
        with provider_receipt_context(task_id="boom"):
            raise RuntimeError("provider call failed")
    assert _receipt_env() == dict.fromkeys(_RECEIPT_ENV)


def test_a_worker_thread_without_the_contextvar_still_sees_the_context_through_the_environment(
    clean_receipt_env, tmp_path: Path
) -> None:
    """Why the environment mirror exists at all: a pool thread does not inherit a ContextVar."""

    import threading

    from conference_pipeline.provider_receipts import current_provider_receipt_context

    seen: list[str | None] = []
    with provider_receipt_context(workspace_root=tmp_path, task_id="outer-task"):
        thread = threading.Thread(target=lambda: seen.append(current_provider_receipt_context().task_id))
        thread.start()
        thread.join(5)
    assert seen == ["outer-task"]
