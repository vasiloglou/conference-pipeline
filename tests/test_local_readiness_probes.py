"""Tests for the injectable per-family local-runtime readiness probes.

Phase 10 (v3.0 / LOCAL-02), plan 10-01 Task 2 / D-06/D-07: per-family probes
return injectable ``Callable[[str], bool]`` callables; healthy -> True,
down/absent -> False (never raises); no credential leakage.
"""

from __future__ import annotations

from conference_pipeline.model_governance.local_readiness import (
    ReadinessProbe,
    make_transcription_probe,
    make_vision_server_probe,
)


# ---------------------------------------------------------------------------
# Vision-server probe (shared by ocr.slide_image_parse + vision.slide_analysis)
# ---------------------------------------------------------------------------


class _FakeHealthyClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def health_check(self) -> bool:
        return True


class _FakeUnhealthyClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def health_check(self) -> bool:
        return False


class _FakeRaisingClient:
    def __init__(self, base_url: str) -> None:
        # Simulate a server that is down: construction or health_check explodes.
        raise ConnectionError("vision server unreachable")


def test_vision_probe_returns_callable() -> None:
    probe = make_vision_server_probe(
        "http://localhost:8080/v1", client_factory=_FakeHealthyClient
    )
    assert callable(probe)
    # Type alias is a plain Callable; the probe accepts a task id and returns bool.
    annotated: ReadinessProbe = probe
    assert isinstance(annotated("ocr.slide_image_parse"), bool)


def test_vision_probe_true_on_healthy_client() -> None:
    probe = make_vision_server_probe(
        "http://localhost:8080/v1", client_factory=_FakeHealthyClient
    )
    assert probe("ocr.slide_image_parse") is True
    assert probe("vision.slide_analysis") is True


def test_vision_probe_false_on_unhealthy_client() -> None:
    probe = make_vision_server_probe(
        "http://localhost:8080/v1", client_factory=_FakeUnhealthyClient
    )
    assert probe("vision.slide_analysis") is False


def test_vision_probe_false_when_client_raises() -> None:
    """A down/absent server makes the factory raise; the probe swallows it."""
    probe = make_vision_server_probe(
        "http://localhost:8080/v1", client_factory=_FakeRaisingClient
    )
    assert probe("ocr.slide_image_parse") is False


def test_vision_probe_false_when_base_url_missing() -> None:
    probe = make_vision_server_probe(None, client_factory=_FakeHealthyClient)
    assert probe("ocr.slide_image_parse") is False


# ---------------------------------------------------------------------------
# Transcription probe (local whisper runtime)
# ---------------------------------------------------------------------------


def test_transcription_probe_true_when_available() -> None:
    probe = make_transcription_probe(availability_check=lambda: True)
    assert probe("transcription.session_speech") is True


def test_transcription_probe_false_when_absent() -> None:
    probe = make_transcription_probe(availability_check=lambda: False)
    assert probe("transcription.session_speech") is False


def test_transcription_probe_false_on_import_error() -> None:
    """An ImportError from the runtime check resolves to False, not a raise."""

    def _boom() -> bool:
        raise ImportError("whisper package required")

    probe = make_transcription_probe(availability_check=_boom)
    assert probe("transcription.session_speech") is False


def test_default_transcription_probe_never_raises() -> None:
    """The default (real) probe returns a bool regardless of host state."""
    probe = make_transcription_probe()
    result = probe("transcription.session_speech")
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Security: no credentialed base URL or API key escapes a probe (T-10-01)
# ---------------------------------------------------------------------------


def test_vision_probe_does_not_leak_credentialed_url(caplog) -> None:
    """A credentialed base URL must never appear in probe output or logs."""
    import logging

    secret_url = "http://user:supersecret@localhost:8080/v1"

    class _RaisingWithUrl:
        def __init__(self, base_url: str) -> None:
            raise ConnectionError(f"failed talking to {base_url}")

    probe = make_vision_server_probe(secret_url, client_factory=_RaisingWithUrl)
    with caplog.at_level(logging.DEBUG):
        result = probe("ocr.slide_image_parse")

    assert result is False
    # The probe returns only a bool; and nothing it logs may carry the secret
    # or a credentialed authority.
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "supersecret" not in joined
    assert "user:" not in joined
    assert "://user" not in joined
