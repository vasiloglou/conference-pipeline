"""v1.61 Phase 70 — CLI vision flags + operator status + cost registry rows."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer.testing

from conference_pipeline.cli import app
from conference_pipeline.control_plane.vision_status import (
    _redact_endpoint,
    _registry_status_per_task,
    vision_status_block,
)
from conference_pipeline.costs.estimator import load_task_specs
from conference_pipeline.provider_receipts import record_local_vision_call


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------


def test_smoketest_help_advertises_local_vision_flags():
    runner = typer.testing.CliRunner()
    result = runner.invoke(app, ["conference-smoketest", "--help"])
    assert result.exit_code == 0
    assert "--local-vision-model" in result.output
    assert "--local-vision-provider" in result.output
    assert "--local-vision-base-url" in result.output


def test_smoketest_local_vision_without_local_fails(monkeypatch, tmp_path):
    runner = typer.testing.CliRunner()
    result = runner.invoke(
        app,
        [
            "conference-smoketest",
            "--workspace", str(tmp_path),
            "--mode", "dry-run",
            "--local-vision-model", "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        ],
    )
    assert result.exit_code == 2
    assert "--local-vision-model requires --local" in result.output


def test_smoketest_invalid_local_vision_provider_rejected(monkeypatch, tmp_path):
    """Repro VLM-CLI-01: --local-vision-provider only accepts vllm_mlx | fastvlm.

    Patch ``dotenv.load_dotenv`` so the smoketest does not slurp the dev
    .env back into the environment after the test clears OPENAI_API_KEY.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    runner = typer.testing.CliRunner()
    result = runner.invoke(
        app,
        [
            "conference-smoketest",
            "--workspace", str(tmp_path),
            "--mode", "dry-run",
            "--local",
            "--local-vision-model", "mlx-community/Qwen3-VL-8B-Instruct-4bit",
            "--local-vision-provider", "bogus",
        ],
    )
    assert result.exit_code == 2
    assert "--local-vision-provider must be" in result.output


def test_smoketest_cli_prints_vision_status_when_configured(
    monkeypatch, tmp_path
):
    """v1.61 Phase 72: VLM-CLI-02 — when --local-vision-model is provided,
    the smoketest CLI prints the redacted endpoint and per-task registry
    status under the report path. Stub run_conference_smoketest so the
    test stays scoped to the CLI surface (workflow execution is covered
    by tests/test_smoke_test.py).
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)

    from conference_pipeline import cli as cli_module
    from conference_pipeline.smoke_test import (
        CONFERENCE_SMOKETEST_ID,
        SmokeTestReport,
        SMOKE_TEST_REPORT_SCHEMA_VERSION,
    )

    fake_report = SmokeTestReport(
        schema_version=SMOKE_TEST_REPORT_SCHEMA_VERSION,
        smoke_test_id=CONFERENCE_SMOKETEST_ID,
        run_id="vision-cli-proof",
        workspace=str(tmp_path),
        topic=None,
        mode="dry-run",
        summary_path=tmp_path / "summary.json",
        report_path=tmp_path / "conference-smoketest-report.json",
        expected_stages=["analyze", "deck"],
        stage_results=[],
        assertions=[],
        non_live_guarantees=[],
        local_vision_status={
            "lane": "vision",
            "active": True,
            "model": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
            "runtime_kind": "vllm_mlx",
            "endpoint": "mlx-vision://mlx-community/Qwen3-VL-8B-Instruct-4bit",
            "registry_status_by_task": {
                "vision.slide_analysis": "benchmark_pending",
                "vision.figure_caption": "benchmark_pending",
                "vision.chart_summary": "benchmark_pending",
                "presentation.visual_qc": "benchmark_pending",
                "ocr.slide_image_parse": "benchmark_pending",
            },
            "session_receipt_count": 0,
        },
    )
    monkeypatch.setattr(
        cli_module, "run_conference_smoketest", lambda **_: fake_report
    )

    runner = typer.testing.CliRunner()
    result = runner.invoke(
        app,
        [
            "conference-smoketest",
            "--workspace", str(tmp_path),
            "--mode", "dry-run",
            "--local",
            "--local-vision-model", "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        ],
    )

    assert result.exit_code == 0, result.output
    # Redacted endpoint surfaced; no host/port leak.
    assert "Vision lane: mlx-vision://mlx-community/Qwen3-VL-8B-Instruct-4bit" in result.output
    assert "127." not in result.output
    assert "8090" not in result.output
    # All five vision/OCR tasks reported.
    assert "vision.slide_analysis: benchmark_pending" in result.output
    assert "presentation.visual_qc: benchmark_pending" in result.output
    assert "ocr.slide_image_parse: benchmark_pending" in result.output


def test_smoketest_cli_omits_vision_status_block_when_no_local_vision(
    monkeypatch, tmp_path
):
    """v1.61 Phase 72: when no --local-vision-model is provided, the
    smoketest CLI must not print a Vision lane block (preserves existing
    operator-output shape for cloud-only runs).
    """
    from conference_pipeline import cli as cli_module
    from conference_pipeline.smoke_test import (
        CONFERENCE_SMOKETEST_ID,
        SmokeTestReport,
        SMOKE_TEST_REPORT_SCHEMA_VERSION,
    )

    fake_report = SmokeTestReport(
        schema_version=SMOKE_TEST_REPORT_SCHEMA_VERSION,
        smoke_test_id=CONFERENCE_SMOKETEST_ID,
        run_id="cloud-cli-proof",
        workspace=str(tmp_path),
        topic=None,
        mode="dry-run",
        summary_path=tmp_path / "summary.json",
        report_path=tmp_path / "conference-smoketest-report.json",
        expected_stages=["analyze", "deck"],
        stage_results=[],
        assertions=[],
        non_live_guarantees=[],
        local_vision_status=None,
    )
    monkeypatch.setattr(
        cli_module, "run_conference_smoketest", lambda **_: fake_report
    )

    runner = typer.testing.CliRunner()
    result = runner.invoke(
        app,
        [
            "conference-smoketest",
            "--workspace", str(tmp_path),
            "--mode", "dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Vision lane:" not in result.output


# ---------------------------------------------------------------------------
# Operator status block
# ---------------------------------------------------------------------------


def test_redact_endpoint_returns_none_without_model():
    assert _redact_endpoint(None, "vllm_mlx") is None


def test_redact_endpoint_uses_vllm_label_by_default():
    assert _redact_endpoint("m/x", "vllm_mlx") == "mlx-vision://m/x"
    assert _redact_endpoint("m/x", None) == "mlx-vision://m/x"


def test_redact_endpoint_uses_fastvlm_label_for_fastvlm():
    assert _redact_endpoint("apple/FastVLM-7B-MLX", "fastvlm") == (
        "mlx-fastvlm://apple/FastVLM-7B-MLX"
    )


def test_status_block_inactive_when_no_model_set(monkeypatch):
    monkeypatch.delenv("GPTR_LOCAL_VISION_MODEL", raising=False)
    monkeypatch.delenv("GPTR_LOCAL_VISION_RUNTIME_KIND", raising=False)
    block = vision_status_block(workspace=None)
    assert block["lane"] == "vision"
    assert block["active"] is False
    assert block["model"] is None
    assert block["endpoint"] is None
    assert block["session_receipt_count"] == 0


def test_status_block_active_when_model_set_no_secrets(monkeypatch):
    monkeypatch.setenv(
        "GPTR_LOCAL_VISION_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    )
    monkeypatch.setenv("GPTR_LOCAL_VISION_RUNTIME_KIND", "vllm_mlx")
    block = vision_status_block(workspace=None)
    assert block["active"] is True
    assert block["model"] == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    assert block["runtime_kind"] == "vllm_mlx"
    assert block["endpoint"] == "mlx-vision://mlx-community/Qwen3-VL-8B-Instruct-4bit"
    # No host/port/URL leaks.
    assert "127." not in block["endpoint"]
    assert "8090" not in block["endpoint"]
    assert "v1" not in block["endpoint"].split("://", 1)[-1].split("/")[0]


def test_status_block_counts_session_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("GPTR_LOCAL_VISION_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit")
    monkeypatch.setenv("GPTR_LOCAL_VISION_RUNTIME_KIND", "vllm_mlx")
    record_local_vision_call(
        provider="local_vlm",
        runtime_kind="vllm_mlx",
        model="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        endpoint_label="mlx-vision://mlx-community/Qwen3-VL-8B-Instruct-4bit",
        vision_task="slide_analysis",
        image_count=1,
        image_dimensions=[(2, 2)],
        image_bytes_size=128,
        run_id="r1",
        stage_id="vision",
        workspace=tmp_path,
    )
    block = vision_status_block(workspace=tmp_path)
    assert block["session_receipt_count"] == 1


def test_status_block_resolves_registry_status_for_known_model(monkeypatch):
    monkeypatch.setenv(
        "GPTR_LOCAL_VISION_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    )
    monkeypatch.setenv("GPTR_LOCAL_VISION_RUNTIME_KIND", "vllm_mlx")
    block = vision_status_block(workspace=None)
    by_task = block["registry_status_by_task"]
    # All five vision/OCR tasks should resolve to benchmark_pending now (no
    # promotion has occurred in source-controlled registry).
    assert set(by_task.keys()) == {
        "vision.slide_analysis",
        "vision.figure_caption",
        "vision.chart_summary",
        "presentation.visual_qc",
        "ocr.slide_image_parse",
    }
    for task_id, status in by_task.items():
        assert status == "benchmark_pending", f"{task_id}: expected benchmark_pending, got {status!r}"


def test_status_block_unknown_model_returns_benchmark_pending(monkeypatch):
    monkeypatch.setenv("GPTR_LOCAL_VISION_MODEL", "fake/UnregisteredModel")
    monkeypatch.setenv("GPTR_LOCAL_VISION_RUNTIME_KIND", "vllm_mlx")
    block = vision_status_block(workspace=None)
    for status in block["registry_status_by_task"].values():
        assert status == "benchmark_pending"


# ---------------------------------------------------------------------------
# Cost registry rows
# ---------------------------------------------------------------------------


def test_cost_registry_includes_paid_and_local_lanes_for_each_vision_task():
    specs = load_task_specs()
    task_ids = {t["task_id"] for t in specs["tasks"]}
    expected = {
        "vision_slide_analysis_paid",
        "vision_slide_analysis_local",
        "vision_figure_caption_paid",
        "vision_figure_caption_local",
        "presentation_visual_qc_paid",
        "presentation_visual_qc_local",
        "vision_chart_summary_paid",
        "vision_chart_summary_local",
        "ocr_slide_image_parse_paid",
        "ocr_slide_image_parse_local",
    }
    missing = expected - task_ids
    assert not missing, f"missing v1.61 cost rows: {missing}"


def test_cost_registry_ocr_slide_image_parse_has_paid_and_local_lanes():
    specs = load_task_specs()
    by_id = {t["task_id"]: t for t in specs["tasks"]}
    paid = by_id["ocr_slide_image_parse_paid"]
    local = by_id["ocr_slide_image_parse_local"]
    assert paid["provider"] == "openai"
    assert paid["pricing_unit"] == "tokens"
    assert paid["quantity_source"] == "openai_image_tokens"
    assert local["provider"] == "local"
    assert local["fixed_usd"] == 0.0


def test_cost_registry_local_rows_are_zero_dollars():
    specs = load_task_specs()
    locals_ = [
        t for t in specs["tasks"]
        if t["task_id"].startswith(("vision_", "ocr_")) and t["provider"] == "local"
    ]
    assert locals_, "expected at least one v1.61 local-lane row"
    for task in locals_:
        assert task["pricing_unit"] == "none"
        assert task["fixed_usd"] == 0.0


def test_cost_registry_paid_rows_have_explicit_models():
    specs = load_task_specs()
    paid = [
        t for t in specs["tasks"]
        if t["task_id"].startswith("vision_") and t["provider"] == "openai"
    ]
    assert paid, "expected v1.61 paid-lane rows"
    for task in paid:
        models = task["models"]
        assert "low" in models and "base" in models and "high" in models
        # Models should reference real OpenAI vision-capable models, not local ids.
        assert all("mlx-community" not in str(m) for m in models.values())
        assert all("FastVLM" not in str(m) for m in models.values())


def test_openai_slide_ocr_receipt_observer_receives_the_wrapped_call_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The text-only OCR API still surfaces the provider receipt to r6 owners."""

    from conference_pipeline import slide_ocr
    from conference_pipeline.provider_receipts import ProviderCallResult

    image = tmp_path / "slide.png"
    _write_png = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
        "de0000000c49444154789c6360f8ffff3f0005fe02fea7c5819b0000000049454e44ae426082"
    )
    image.write_bytes(_write_png)
    observed: list[object] = []
    receipt = SimpleNamespace(status="succeeded")
    service = slide_ocr.OpenAISlideOCRService(
        api_key="test-only",
        receipt_observer=observed.append,
    )
    service._client = object()

    async def fake_create(*_args: object, **_kwargs: object) -> ProviderCallResult[object]:
        return ProviderCallResult(
            response=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="receipt-aware OCR"))]
            ),
            receipt=receipt,
        )

    monkeypatch.setattr(slide_ocr, "openai_chat_completions_create_async", fake_create)
    assert asyncio.run(service.ocr_slide(image, "fixture")) == "receipt-aware OCR"
    assert observed == [receipt]
