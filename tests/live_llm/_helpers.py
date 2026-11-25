from __future__ import annotations

import os

import pytest

from conference_pipeline.extractions.backends import BackendKind, select_local_llm_backend
from conference_pipeline.extractions.walker import ExtractionInput


def handle_unreachable_backend(exc: Exception) -> None:
    message = "local LLM endpoint not reachable for live lane: " f"{exc}"
    if os.environ.get("RUN_LIVE_LLM") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture
def real_local_llm_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ASSIST", "1")
    backend = select_local_llm_backend()
    try:
        backend.readiness()
    except Exception as exc:  # noqa: BLE001 - skip or fail based on live-lane mode
        handle_unreachable_backend(exc)
    return backend


@pytest.fixture
def ambiguous_live_record() -> ExtractionInput:
    return ExtractionInput(
        artifact_id="live:ambiguous-1",
        kind="transcript",
        text_path=None,
        digest="live-ambiguous-1",
        status="transcribed",
        source_path="storage/transcription-ocr/media/live-ambiguous-1/transcript.txt",
        topic_hint=None,
        text_content=(
            "This discussion explores tradeoffs, evaluation framing, and "
            "system behavior in broad terms without matching the local "
            "heuristic taxonomy directly."
        )
        * 4,
    )


def assert_real_endpoint_label(label: str) -> None:
    assert label
    assert "mock" not in label.lower()
    assert label.startswith(("mlx://", "http://localhost", "local-"))


def assert_real_backend_kind(kind: BackendKind) -> None:
    assert kind in {BackendKind.MLX, BackendKind.HTTP}
