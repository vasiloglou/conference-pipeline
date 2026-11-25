import json
from pathlib import Path

import pytest

from conference_pipeline.model_governance.registry import load_registry
from conference_pipeline.ocr_benchmark import run_benchmark
from conference_pipeline.slide_ocr import FallbackSlideOCRService, service_call_metadata


class FakeOCRService:
    def __init__(self, response: str, model: str = "fake-model"):
        self.response = response
        self.model = model
        self.calls = 0

    async def ocr_slide(
        self,
        image_path: Path,
        title: str,
        transcript_context: str | None = None,
    ) -> str:
        self.calls += 1
        return self.response


def _valid_response(title: str = "A useful slide") -> str:
    return json.dumps(
        {
            "slide_text": {
                "title": title,
                "bullet_points": ["A detailed enough OCR result for confidence checks."],
                "other_text": "This response has enough extracted text to avoid fallback.",
                "urls": [],
            },
            "visual_elements": [],
            "key_concepts": [{"concept": "confidence"}],
            "references": [],
        }
    )


def _write_png(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a project dependency.
        pytest.skip("Pillow is required for image fixture")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), color="white").save(path)


@pytest.mark.asyncio
async def test_fallback_ocr_uses_primary_when_confident(tmp_path: Path):
    primary = FakeOCRService(_valid_response(), model="primary")
    fallback = FakeOCRService(_valid_response("Fallback"), model="fallback")
    service = FallbackSlideOCRService(primary, fallback)
    image_path = tmp_path / "slide.png"

    response = await service.ocr_slide(image_path, "Talk")
    metadata = service_call_metadata(service, image_path)

    assert response == primary.response
    assert primary.calls == 1
    assert fallback.calls == 0
    assert metadata.fallback_used is False
    assert metadata.model == "primary"


@pytest.mark.asyncio
async def test_fallback_ocr_calls_fallback_on_invalid_primary(tmp_path: Path):
    primary = FakeOCRService("not json", model="primary")
    fallback = FakeOCRService(_valid_response("Fallback"), model="fallback")
    service = FallbackSlideOCRService(primary, fallback)
    image_path = tmp_path / "slide.png"

    response = await service.ocr_slide(image_path, "Talk")
    metadata = service_call_metadata(service, image_path)

    assert response == fallback.response
    assert primary.calls == 1
    assert fallback.calls == 1
    assert metadata.fallback_used is True
    assert metadata.fallback_reason == "invalid_json"
    assert metadata.model == "fallback"


def test_ocr_benchmark_dry_run_covers_openai_anthropic_gemini(tmp_path: Path):
    _write_png(tmp_path / "slides" / "slide_001.png")

    payload = run_benchmark(
        tmp_path,
        sample_size=1,
        live=False,
        budget_cap_usd=1.0,
        output_dir=tmp_path / "benchmark",
    )
    models = {item["model"] for item in payload["candidate_summaries"]}
    providers = {item["provider"] for item in payload["candidate_summaries"]}

    assert payload["live"] is False
    assert payload["sample_image_count"] == 1
    assert {"openai", "anthropic", "gemini"}.issubset(providers)
    # Candidates come from the registry's ocr.slide_image_parse benchmark block.
    registered = load_registry().get("ocr.slide_image_parse").benchmark["candidates"]
    assert models == {str(model) for model in registered}
    assert Path(payload["receipt_path"]).exists()
    assert payload["spent_estimate_usd"] > 0
