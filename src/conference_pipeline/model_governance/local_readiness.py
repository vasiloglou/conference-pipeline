"""Injectable per-family local-runtime readiness probes (Phase 10, D-06/D-07).

These factories return ``Callable[[str], bool]`` callables suitable for the
``readiness_check`` argument of
:func:`conference_pipeline.model_governance.local_routing.resolve_local_first_selection`.

Design contract:

- **Binary** available-or-fallback (D-06). Memory-aware selection is
  LOCAL-FUTURE-01 and is out of scope here.
- **Automatic, not halt** (D-07): every probe is hardened to return ``False`` on
  ANY failure (server down/absent, import error, timeout, malformed config)
  rather than raising. The resolver also guards, but probes are hardened too so
  they are safe to call anywhere.
- **Injectable** (D-07): each factory accepts its dependency (a client factory or
  an availability check) so the offline suite can pass a fake; production
  defaults to the real health check.
- **Secret-safe** (T-10-01): probes return only ``bool``. Any diagnostic string
  is passed through :func:`_redacted_base_url_identity` so no credentialed base
  URL or API key is ever emitted.

The vision-server probe is shared by ``ocr.slide_image_parse`` and
``vision.slide_analysis`` (they share the mlx vision server). Transcription
probes the local whisper runtime separately (no shared health endpoint).
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

ReadinessProbe = Callable[[str], bool]


def _safe_endpoint_identity(base_url: str | None) -> str:
    """Return a credential-free identity for a base URL, for diagnostics only."""
    if not base_url:
        return "unconfigured"
    try:
        from conference_pipeline.extractions.classifier import (
            _redacted_base_url_identity,
        )

        return _redacted_base_url_identity(base_url)
    except Exception:  # noqa: BLE001 — diagnostics must never raise
        return "redaction-failed"


class _HealthCheckable(Protocol):
    def health_check(self) -> bool: ...


def _default_vision_client_factory(base_url: str) -> _HealthCheckable:
    """Construct the real mlx vision-server client (loopback-enforced)."""
    from conference_pipeline.model_governance.mlx_vision_client import (
        MlxVisionServerClient,
    )

    return MlxVisionServerClient(base_url, timeout_s=2.0)


def make_vision_server_probe(
    base_url: str | None = None,
    *,
    client_factory: Callable[[str], _HealthCheckable] | None = None,
) -> ReadinessProbe:
    """Return a readiness probe for the shared mlx vision server (OCR + vision).

    The probe wraps :meth:`MlxVisionServerClient.health_check`. It returns
    ``False`` — never raises — when ``base_url`` is missing, when client
    construction fails (e.g. non-loopback URL fails closed), or when the health
    check itself errors or reports unhealthy.

    Args:
        base_url: Loopback base URL of the local vision server. When ``None`` the
            probe always reports not-ready (paid fallback).
        client_factory: Injectable factory returning an object with a
            ``health_check() -> bool`` method. Defaults to the real
            :class:`MlxVisionServerClient`.
    """
    factory = client_factory or _default_vision_client_factory

    def _probe(task_id: str) -> bool:
        if not base_url:
            logger.debug(
                "vision readiness: no base_url for %s -> not ready (paid fallback)",
                task_id,
            )
            return False
        try:
            client = factory(base_url)
            ready = bool(client.health_check())
        except Exception as exc:  # noqa: BLE001 — D-07: any failure -> not ready
            logger.debug(
                "vision readiness probe failed for %s at %s: %s",
                task_id,
                _safe_endpoint_identity(base_url),
                type(exc).__name__,
            )
            return False
        return ready

    return _probe


def _default_whisper_available() -> bool:
    """Report whether the local whisper runtime is importable (no model load)."""
    # importlib.util.find_spec only inspects the import system; it does not
    # import or load a model, so this stays a readiness check (D-07), not an
    # inference run.
    try:
        return importlib.util.find_spec("whisper") is not None
    except Exception:  # noqa: BLE001 — a broken import system -> not ready
        return False


def make_transcription_probe(
    *,
    availability_check: Callable[[], bool] | None = None,
) -> ReadinessProbe:
    """Return a readiness probe for the local whisper transcription runtime.

    The probe lazily checks that the whisper runtime is importable. It returns
    ``False`` — never raises — on ImportError, a broken import system, or any
    other failure.

    Args:
        availability_check: Injectable callable returning whether the local
            whisper runtime is ready. Defaults to a lightweight
            ``importlib.util.find_spec("whisper")`` probe (no model load).
    """
    check = availability_check or _default_whisper_available

    def _probe(task_id: str) -> bool:
        try:
            return bool(check())
        except Exception as exc:  # noqa: BLE001 — D-07: any failure -> not ready
            logger.debug(
                "transcription readiness probe failed for %s: %s",
                task_id,
                type(exc).__name__,
            )
            return False

    return _probe
