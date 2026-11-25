from __future__ import annotations
import pytest

from pathlib import Path

from scripts.exhaustive_extraction.models import LOCAL_MLX_QWEN35B_A3B, ModelSelector
from scripts.exhaustive_extraction.processor import include_slides_for_extraction


@pytest.mark.repo_data("conferences/iclr2025/config.yaml")
def test_iclr_exhaustive_extraction_env_defaults_use_local_qwen35b_a3b() -> None:
    from conference_pipeline.iclr2025_live import exhaustive_extraction_env_defaults

    env = exhaustive_extraction_env_defaults(Path("conferences/iclr2025"))

    assert env["EXTRACTION_LOCAL_LLM_ASSIST"] == "1"
    assert env["EXTRACTION_LOCAL_LLM_BACKEND"] == "mlx"
    assert env["EXTRACTION_LOCAL_LLM_MODEL"] == LOCAL_MLX_QWEN35B_A3B
    assert env["EXHAUSTIVE_EXTRACTION_PROVIDER"] == "mlx"
    assert env["EXHAUSTIVE_EXTRACTION_MODEL"] == LOCAL_MLX_QWEN35B_A3B
    assert env["EXHAUSTIVE_EXTRACTION_INCLUDE_SLIDES"] == "false"


def test_model_selector_uses_local_mlx_qwen_when_requested(monkeypatch):
    monkeypatch.setenv("EXHAUSTIVE_EXTRACTION_PROVIDER", "mlx")
    monkeypatch.delenv("EXHAUSTIVE_EXTRACTION_MODEL", raising=False)

    selector = ModelSelector()

    assert selector.select_model(1000) == LOCAL_MLX_QWEN35B_A3B


def test_include_slides_for_extraction_can_disable_slide_semantic_lane(monkeypatch):
    monkeypatch.setenv("EXHAUSTIVE_EXTRACTION_INCLUDE_SLIDES", "false")
    assert include_slides_for_extraction() is False

    monkeypatch.setenv("EXHAUSTIVE_EXTRACTION_INCLUDE_SLIDES", "true")
    assert include_slides_for_extraction() is True
