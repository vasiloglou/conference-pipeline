"""v1.61 Phase 67 — SlideAnalysisEngine + LLMOcrEngine local branch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conference_pipeline.config import PipelineSettings
from conference_pipeline.models import OcrResult, SlideImage
from conference_pipeline.ocr_engine import LLMOcrEngine, OcrError
from conference_pipeline.provider_receipts import (
    LOCAL_VISION_RECEIPT_MARKER,
    scan_local_vision_receipts,
)
from conference_pipeline.slide_analysis import (
    SlideAnalysisEngine,
    SlideAnalysisParseError,
    SlideAnalysisResult,
    SlideFigureRef,
)
from conference_pipeline.slide_analysis.engine import parse_slide_analysis


@pytest.fixture()
def png_bytes() -> bytes:
    """Bytes-only PNG header reporting 4x4 dims; sufficient for engine plumbing."""
    import struct

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 4, 4) + b"\x08\x06\x00\x00\x00"
    return sig + ihdr + b"\x00\x00\x00\x00"


@pytest.fixture()
def png_file(tmp_path, png_bytes) -> Path:
    path = tmp_path / "slide.png"
    path.write_bytes(png_bytes)
    return path


@pytest.fixture()
def settings() -> PipelineSettings:
    return PipelineSettings()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _full_json_payload() -> str:
    return json.dumps(
        {
            "salient_claims": ["a claim", "another claim"],
            "figures": [
                {"caption": "Fig 1", "modality": "chart", "axes": ["x", "y"]},
                {"caption": "Fig 2", "modality": "diagram", "axes": None},
            ],
            "code_blocks": ['print("hi")'],
            "equations": ["E = mc^2"],
            "headings": ["Method"],
            "body_text": "Body content",
            "reading_order_hint": "top-to-bottom",
            "confidence": 0.85,
        }
    )


def test_parse_json_full_round_trip():
    raw = _full_json_payload()
    result = parse_slide_analysis(raw, prefer="json")
    assert result.salient_claims == ("a claim", "another claim")
    assert len(result.figures) == 2
    assert result.figures[0].modality == "chart"
    assert result.figures[0].axes == ("x", "y")
    assert result.figures[1].axes is None
    assert result.code_blocks == ('print("hi")',)
    assert result.equations == ("E = mc^2",)
    assert result.headings == ("Method",)
    assert result.body_text == "Body content"
    assert result.reading_order_hint == "top-to-bottom"
    assert result.confidence == 0.85


def test_parse_json_strips_code_fences():
    raw = "```json\n" + _full_json_payload() + "\n```"
    result = parse_slide_analysis(raw, prefer="json")
    assert result.headings == ("Method",)


def test_parse_json_clamps_confidence_in_range():
    raw = json.dumps({**json.loads(_full_json_payload()), "confidence": 99.0})
    result = parse_slide_analysis(raw, prefer="json")
    assert result.confidence == 1.0


def test_parse_json_invalid_raises_with_truncated_raw():
    raw = "{not valid json"
    with pytest.raises(SlideAnalysisParseError) as excinfo:
        parse_slide_analysis(raw, prefer="json")
    assert excinfo.value.raw_output_truncated.startswith("{not valid")


def test_parse_markdown_fallback_round_trip():
    raw = (
        "## Salient Claims\n"
        "- first claim\n"
        "- second claim\n"
        "## Figures\n"
        "- Fig 1 — chart — x, y\n"
        "## Code Blocks\n"
        "```python\nprint('hi')\n```\n"
        "## Equations\n"
        "- E = mc^2\n"
        "## Headings\n"
        "- Method\n"
        "## Body Text\n"
        "the body\n"
        "## Reading Order\n"
        "top-to-bottom\n"
        "## Confidence\n"
        "0.7\n"
    )
    result = parse_slide_analysis(raw, prefer="markdown")
    assert result.salient_claims == ("first claim", "second claim")
    assert result.figures[0].modality == "chart"
    assert result.figures[0].axes == ("x", "y")
    assert result.code_blocks == ("print('hi')",)
    assert result.equations == ("E = mc^2",)
    assert result.body_text == "the body"
    assert result.reading_order_hint == "top-to-bottom"
    assert result.confidence == 0.7


# ---------------------------------------------------------------------------
# SlideAnalysisEngine — provider-strategy dispatch
# ---------------------------------------------------------------------------


def test_engine_dispatches_to_runner(png_file, settings):
    captured: dict[str, Any] = {}

    def fake_runner(*, image_bytes, media_type, system_prompt, user_prompt, model_id):
        captured["image_bytes_len"] = len(image_bytes)
        captured["media_type"] = media_type
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["model_id"] = model_id
        return _full_json_payload()

    engine = SlideAnalysisEngine(
        settings=settings,
        provider="local_vlm",
        model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        runner=fake_runner,
    )
    result = engine.analyze(png_file, task_id="vision.slide_analysis")
    assert isinstance(result, SlideAnalysisResult)
    assert captured["media_type"] == "image/png"
    assert captured["model_id"] == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    assert "vision.slide_analysis" in captured["user_prompt"]


def test_engine_writes_local_vision_receipt(tmp_path, png_file, settings):
    def fake_runner(**kwargs):
        return _full_json_payload()

    engine = SlideAnalysisEngine(
        settings=settings,
        provider="local_vlm",
        model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        runner=fake_runner,
        workspace=tmp_path,
        run_id="run-001",
        stage_id="presentation",
    )
    result = engine.analyze(png_file, task_id="vision.slide_analysis")
    assert result.salient_claims == ("a claim", "another claim")
    receipts = scan_local_vision_receipts(tmp_path)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["receipt_type"] == LOCAL_VISION_RECEIPT_MARKER
    assert receipt["provider"] == "local_vlm"
    assert receipt["runtime_kind"] == "vllm_mlx"
    assert receipt["model"] == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    assert receipt["vision_task"] == "vision.slide_analysis"
    assert receipt["cost_usd"] == 0.0


def test_engine_local_fastvlm_uses_fastvlm_endpoint_label(tmp_path, png_file, settings):
    def fake_runner(**kwargs):
        return _full_json_payload()

    engine = SlideAnalysisEngine(
        settings=settings,
        provider="local_fastvlm",
        model_id="apple/FastVLM-7B-MLX",
        runner=fake_runner,
        workspace=tmp_path,
        run_id="r2",
        stage_id="vision-bench",
    )
    engine.analyze(png_file, task_id="vision.figure_caption")
    receipts = scan_local_vision_receipts(tmp_path)
    assert len(receipts) == 1
    assert receipts[0]["runtime_kind"] == "mlx_native_fastvlm"
    assert receipts[0]["endpoint_label"].startswith("mlx-fastvlm://")


def test_engine_paid_provider_does_not_write_local_receipt(tmp_path, png_file, settings):
    def fake_runner(**kwargs):
        return _full_json_payload()

    engine = SlideAnalysisEngine(
        settings=settings,
        provider="openai",
        model_id="gpt-5.4-mini",
        runner=fake_runner,
        workspace=tmp_path,
    )
    engine.analyze(png_file, task_id="vision.slide_analysis")
    assert scan_local_vision_receipts(tmp_path) == []


def test_engine_parse_failure_still_writes_receipt(tmp_path, png_file, settings):
    def fake_runner(**kwargs):
        return "this is not JSON or markdown"

    engine = SlideAnalysisEngine(
        settings=settings,
        provider="local_vlm",
        model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        runner=fake_runner,
        workspace=tmp_path,
        run_id="rX",
        stage_id="presentation",
    )
    with pytest.raises(SlideAnalysisParseError) as excinfo:
        engine.analyze(png_file, task_id="vision.slide_analysis")
    assert "this is not JSON" in excinfo.value.raw_output_truncated
    # Receipt should still be written before the parse failure surfaced.
    receipts = scan_local_vision_receipts(tmp_path)
    assert len(receipts) == 1


def test_engine_unknown_provider_raises(png_file, settings):
    engine = SlideAnalysisEngine(
        settings=settings,
        provider="unknown",  # type: ignore[arg-type]
        model_id="x",
    )
    # Bypass the runner to actually exercise the dispatch table.
    with pytest.raises(ValueError, match="unknown SlideAnalysisEngine provider"):
        engine.analyze(png_file, task_id="vision.slide_analysis")


def test_engine_accepts_bytes_input(settings, png_bytes):
    def fake_runner(**kwargs):
        return _full_json_payload()

    engine = SlideAnalysisEngine(
        settings=settings,
        provider="local_vlm",
        model_id="m",
        runner=fake_runner,
    )
    result = engine.analyze(png_bytes, task_id="vision.slide_analysis")
    assert result.confidence == 0.85


def test_engine_rejects_unsupported_extension(tmp_path, settings):
    bad = tmp_path / "slide.gif"
    bad.write_bytes(b"x")
    engine = SlideAnalysisEngine(
        settings=settings,
        provider="local_vlm",
        model_id="m",
        runner=lambda **k: _full_json_payload(),
    )
    with pytest.raises(ValueError, match="unsupported slide image extension"):
        engine.analyze(bad, task_id="vision.slide_analysis")


# ---------------------------------------------------------------------------
# LLMOcrEngine local branch
# ---------------------------------------------------------------------------


def test_ocr_engine_local_provider_requires_model_id(settings):
    with pytest.raises(ValueError, match="local_model_id"):
        LLMOcrEngine(settings, provider="local")


def test_ocr_engine_local_provider_requires_base_url(monkeypatch, settings):
    monkeypatch.delenv("GPTR_LOCAL_VISION_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="GPTR_LOCAL_VISION_BASE_URL|local_base_url"):
        LLMOcrEngine(
            settings,
            provider="local",
            local_model_id="mlx-community/GLM-OCR-0.9B-4bit",
        )


def test_ocr_engine_local_provider_constructs_with_loopback(settings):
    engine = LLMOcrEngine(
        settings,
        provider="local",
        local_model_id="mlx-community/GLM-OCR-0.9B-4bit",
        local_base_url="http://127.0.0.1:8090/v1",
    )
    assert engine.provider == "local"
    assert engine.local_model_id == "mlx-community/GLM-OCR-0.9B-4bit"
    assert engine.local_base_url == "http://127.0.0.1:8090/v1"


def test_ocr_engine_local_provider_rejects_non_loopback(settings):
    from conference_pipeline.model_governance.registry import ModelGovernanceError

    with pytest.raises(ModelGovernanceError, match="loopback"):
        LLMOcrEngine(
            settings,
            provider="local",
            local_model_id="mlx-community/GLM-OCR-0.9B-4bit",
            local_base_url="http://192.168.1.1:8090/v1",
        )


def test_ocr_engine_unknown_provider_rejected(settings):
    with pytest.raises(ValueError, match="unknown provider"):
        LLMOcrEngine(settings, provider="bogus")


def test_ocr_engine_local_run_ocr_writes_receipt_and_returns_text(
    tmp_path, monkeypatch, settings, png_file
):
    """End-to-end: local OCR run uses the injected client, writes a receipt,
    and returns the model's text in OcrResult shape."""

    class FakeClient:
        base_url = "http://127.0.0.1:8090/v1"

        def chat_complete(self, messages, *, model, max_tokens):
            assert max_tokens == 4096
            assert any(b["type"] == "image_url" for b in messages[-1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": "extracted text from slide; source paper",
                        }
                    }
                ]
            }

    engine = LLMOcrEngine(
        settings,
        provider="local",
        local_model_id="mlx-community/GLM-OCR-0.9B-4bit",
        local_base_url="http://127.0.0.1:8090/v1",
        workspace=tmp_path,
        run_id="run-ocr-1",
        stage_id="transcription-ocr",
    )
    engine.client = FakeClient()  # inject fake to bypass real loopback HTTP

    slide = SlideImage(
        presentation_id="p1",
        slide_index=0,
        image_path=png_file,
    )
    result = engine.run_ocr(slide)
    assert isinstance(result, OcrResult)
    assert result.raw_text == "extracted text from slide; source paper"

    receipts = scan_local_vision_receipts(tmp_path)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["model"] == "mlx-community/GLM-OCR-0.9B-4bit"
    assert receipt["vision_task"] == "ocr.slide_image_parse"
    assert receipt["cost_usd"] == 0.0
    assert receipt["image_count"] == 1
    assert receipt["image_dimensions"] == [[4, 4]]


def test_ocr_engine_openai_branch_unchanged_signature(settings):
    """Regression: paid-provider construction signature still works after
    Phase 67 added the local branch."""
    if settings.openai_api_key:
        # Constructing the OpenAI client requires the package; if importable,
        # construction must succeed without raising.
        try:
            engine = LLMOcrEngine(settings, provider="openai")
            assert engine.provider == "openai"
        except ImportError:
            pytest.skip("openai package not installed")
    else:
        with pytest.raises(ValueError, match="OpenAI API key"):
            LLMOcrEngine(settings, provider="openai")
