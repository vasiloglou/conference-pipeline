"""v1.61 Phase 68 — FastVLMNativeClient unit tests."""

from __future__ import annotations

import struct

import pytest

from conference_pipeline.model_governance.fastvlm_native_client import (
    FastVLMNativeClient,
    FastVLMTextOnlyUnsupported,
    LocalVisionRuntimeError,
)
from conference_pipeline.model_governance.registry import ModelGovernanceError


def _png_bytes() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 4, 4) + b"\x08\x06\x00\x00\x00"
    return sig + ihdr + b"\x00\x00\x00\x00"


def test_construction_does_no_io():
    """Construction must not load weights; lazy on first complete()."""
    client = FastVLMNativeClient(model_id="apple/FastVLM-7B-MLX")
    assert client.model_id == "apple/FastVLM-7B-MLX"
    assert client.endpoint_label == "mlx-fastvlm://apple/FastVLM-7B-MLX"
    assert client.runtime_kind == "mlx_native_fastvlm"
    assert client.provider == "local_fastvlm"
    # No loader called yet.
    assert client._model is None
    assert client._processor is None


def test_text_only_call_raises():
    client = FastVLMNativeClient(loader=lambda m: (object(), object()))
    with pytest.raises(FastVLMTextOnlyUnsupported, match="image-only"):
        client.complete(
            image_bytes=b"",
            media_type="image/png",
            user_prompt="describe",
        )


def test_unsupported_media_type_raises():
    client = FastVLMNativeClient(loader=lambda m: (object(), object()))
    with pytest.raises(ValueError, match="media_type"):
        client.complete(
            image_bytes=_png_bytes(),
            media_type="image/gif",
            user_prompt="describe",
        )


def test_empty_prompt_raises():
    client = FastVLMNativeClient(loader=lambda m: (object(), object()))
    with pytest.raises(ValueError, match="non-empty"):
        client.complete(
            image_bytes=_png_bytes(),
            media_type="image/png",
            user_prompt="   ",
        )


def test_missing_mlx_vlm_raises_governance_error():
    """If neither loader nor mlx_vlm is available, surface ModelGovernanceError."""
    client = FastVLMNativeClient()  # no loader injected, mlx_vlm unavailable
    with pytest.raises(ModelGovernanceError, match="mlx_vlm"):
        client.complete(
            image_bytes=_png_bytes(),
            media_type="image/png",
            user_prompt="describe",
        )


def test_lazy_load_calls_loader_once():
    load_calls: list[str] = []

    def fake_loader(model_id: str):
        load_calls.append(model_id)
        return ("FAKE-MODEL", "FAKE-PROCESSOR")

    received: list[dict] = []

    def fake_generator(**kwargs):
        received.append(kwargs)
        return "fastvlm output text"

    client = FastVLMNativeClient(
        model_id="apple/FastVLM-1.5B-MLX",
        loader=fake_loader,
        generator=fake_generator,
    )
    out1 = client.complete(
        image_bytes=_png_bytes(),
        media_type="image/png",
        user_prompt="caption this slide",
    )
    out2 = client.complete(
        image_bytes=_png_bytes(),
        media_type="image/png",
        user_prompt="caption this slide",
    )
    assert out1 == "fastvlm output text"
    assert out2 == "fastvlm output text"
    assert load_calls == ["apple/FastVLM-1.5B-MLX"]
    assert len(received) == 2
    assert received[0]["model"] == "FAKE-MODEL"
    assert received[0]["processor"] == "FAKE-PROCESSOR"
    assert received[0]["model_id"] == "apple/FastVLM-1.5B-MLX"
    assert received[0]["media_type"] == "image/png"


def test_loader_failure_raises_runtime_error():
    def bad_loader(model_id: str):
        raise OSError("disk failed")

    client = FastVLMNativeClient(loader=bad_loader)
    with pytest.raises(OSError, match="disk failed"):
        client._ensure_loaded()


def test_oom_during_inference_surfaces_local_vision_runtime_error():
    def fake_loader(model_id: str):
        return (object(), object())

    def oom_generator(**kwargs):
        raise MemoryError("simulated OOM")

    client = FastVLMNativeClient(loader=fake_loader, generator=oom_generator)
    with pytest.raises(LocalVisionRuntimeError, match="OOM"):
        client.complete(
            image_bytes=_png_bytes(),
            media_type="image/png",
            user_prompt="describe",
        )


def test_generic_inference_failure_wrapped():
    def fake_loader(model_id: str):
        return (object(), object())

    def boom_generator(**kwargs):
        raise RuntimeError("kernel panic")

    client = FastVLMNativeClient(loader=fake_loader, generator=boom_generator)
    with pytest.raises(LocalVisionRuntimeError, match="inference failed"):
        client.complete(
            image_bytes=_png_bytes(),
            media_type="image/png",
            user_prompt="describe",
        )


def test_endpoint_label_matches_fastvlm_prefix():
    """Receipts written by SlideAnalysisEngine compare on this prefix to
    distinguish native FastVLM calls from server-fronted vllm-mlx calls."""
    client = FastVLMNativeClient(model_id="apple/FastVLM-0.5B-MLX")
    assert client.endpoint_label.startswith("mlx-fastvlm://")
    # Never a host or port number.
    assert "127." not in client.endpoint_label
    assert "8090" not in client.endpoint_label


def test_is_available_returns_bool():
    """is_available must not raise even when mlx_vlm is missing."""
    result = FastVLMNativeClient.is_available()
    assert isinstance(result, bool)


def test_slide_analysis_engine_local_fastvlm_dispatches_to_native_client(tmp_path):
    """Phase 67 wiring: SlideAnalysisEngine(provider='local_fastvlm') routes
    through FastVLMNativeClient and writes a LocalVisionCallReceipt with
    runtime_kind='mlx_native_fastvlm'."""
    from conference_pipeline.config import PipelineSettings
    from conference_pipeline.provider_receipts import scan_local_vision_receipts
    from conference_pipeline.slide_analysis import SlideAnalysisEngine
    import conference_pipeline.slide_analysis.engine as engine_module
    from conference_pipeline.slide_analysis.engine import _MEDIA_TYPE_BY_SUFFIX

    captured: dict = {}

    class FakeFastVLM:
        def __init__(self, *, model_id):
            captured["model_id"] = model_id

        def complete(self, **kwargs):
            captured["complete_kwargs"] = kwargs
            return (
                '{"salient_claims": ["x"], "figures": [], "code_blocks": [], '
                '"equations": [], "headings": [], "body_text": "", '
                '"reading_order_hint": "", "confidence": 0.5}'
            )

    # Patch the real client class with our fake
    import conference_pipeline.model_governance.fastvlm_native_client as fastvlm_module

    real_class = fastvlm_module.FastVLMNativeClient
    fastvlm_module.FastVLMNativeClient = FakeFastVLM
    try:
        slide_path = tmp_path / "slide.png"
        slide_path.write_bytes(_png_bytes())
        engine = SlideAnalysisEngine(
            settings=PipelineSettings(),
            provider="local_fastvlm",
            model_id="apple/FastVLM-7B-MLX",
            workspace=tmp_path,
            run_id="rfast",
            stage_id="vision",
        )
        result = engine.analyze(slide_path, task_id="vision.figure_caption")
    finally:
        fastvlm_module.FastVLMNativeClient = real_class

    assert captured["model_id"] == "apple/FastVLM-7B-MLX"
    assert captured["complete_kwargs"]["media_type"] == "image/png"
    assert result.confidence == 0.5

    receipts = scan_local_vision_receipts(tmp_path)
    assert len(receipts) == 1
    assert receipts[0]["runtime_kind"] == "mlx_native_fastvlm"
    assert receipts[0]["endpoint_label"].startswith("mlx-fastvlm://")
