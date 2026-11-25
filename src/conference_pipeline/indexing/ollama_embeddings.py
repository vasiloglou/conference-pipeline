"""Ollama-backed embedding client for v1.9.

Calls ``POST /api/embeddings`` against a loopback Ollama endpoint
(default ``http://127.0.0.1:11434``). Default model
``nomic-embed-text`` (768-dim).

Loopback-only: the constructor validates the base URL before any HTTP
call. ``manage_server=True`` (the v1.9 default) attempts to spawn
``ollama serve`` if the endpoint is unreachable on first use; the
subprocess is registered for ``atexit`` cleanup. ``manage_server=False``
fails fast with ``EmbeddingError`` when unreachable.

Both the HTTP client and the spawn callable are constructor-injectable
so unit tests run without real network or subprocess activity.

The ``usage()`` shape mirrors :class:`OpenAIEmbeddingClient.usage` and
:class:`MLXEmbeddingClient.usage` for downstream telemetry parity.
``estimated_cost_usd`` is always ``0.0`` and ``provider_receipt_paths``
is always empty (no per-call HTTP receipt — vector-store-level
aggregation lands in v1.9 Phase 45).
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from conference_pipeline._security.loopback import is_loopback_url
from conference_pipeline.indexing.embeddings import (
    EmbeddingError,
    _LegacyEmbedMixin,
    effective_batch_size,
)
from conference_pipeline.provider_receipts import record_local_embedding_call

LOG = logging.getLogger("conference_pipeline.indexing.ollama_embeddings")

_OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "nomic-embed-text"
_DEFAULT_DIM = 768
_OLLAMA_SERVE_LOG_PATH = Path(".planning/logs/ollama-serve.log")

# Tracks subprocesses we own so atexit can terminate them. Keyed by
# normalised base_url so multiple OllamaEmbeddingClient instances sharing
# the same endpoint do not double-spawn.
_MANAGED: dict[str, "subprocess.Popen[bytes]"] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redact_loopback(url: str) -> str:
    """Rewrite host to ``localhost`` for receipt-safe display."""
    parts = urlsplit(url)
    port = parts.port or 11434
    return f"http://localhost:{port}"


def _terminate(proc: "subprocess.Popen[bytes]") -> None:
    """Graceful subprocess teardown invoked from atexit."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:  # noqa: BLE001 - atexit must not raise
        pass


def _default_spawn_ollama_serve(base_url: str) -> "subprocess.Popen[bytes]":
    """Spawn ``ollama serve`` as a managed subprocess.

    Idempotent on ``base_url``: returns the existing managed process if
    one is already tracked. Raises :class:`EmbeddingError` when the
    ``ollama`` binary is not on PATH.

    v1.9 Phase 44 review HI-03: stdout/stderr are captured to
    ``.planning/logs/ollama-serve.log`` (truncated on each spawn) so a
    failed startup (port-in-use, missing model store, GPU init failure)
    is debuggable. Previously these went to ``DEVNULL``.
    """
    existing = _MANAGED.get(base_url)
    if existing is not None and existing.poll() is None:
        return existing
    binary = shutil.which("ollama")
    if not binary:
        raise EmbeddingError(
            "manage_server=True requires the 'ollama' binary on PATH. "
            "Install: `brew install ollama` (macOS) or "
            "`curl -fsSL https://ollama.com/install.sh | sh` (Linux). "
            "Or pass manage_server=False and start `ollama serve` manually."
        )
    _OLLAMA_SERVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # v1.9 review HIGH-3: open the log handle inside a try/except so a
    # Popen failure (PermissionError if the binary loses +x between
    # ``which`` and ``Popen``, OSError on resource exhaustion, etc.)
    # closes the handle before re-raising. Without this, repeated retries
    # leak a file descriptor each time. Also use binary mode since
    # ``stdout=log_handle`` is byte-redirected by ``Popen`` and the
    # ``encoding`` kwarg is ignored on the file-handle path.
    log_handle = _OLLAMA_SERVE_LOG_PATH.open("wb")
    try:
        proc = subprocess.Popen(  # noqa: S603 - binary from shutil.which; args constant
            [binary, "serve"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_handle.close()
        raise
    _MANAGED[base_url] = proc
    atexit.register(_terminate, proc)
    atexit.register(log_handle.close)
    return proc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OllamaEmbeddingClient(_LegacyEmbedMixin):
    """Implements the v1.9 ``EmbeddingClient`` Protocol against Ollama.

    Loopback-only. Constructor performs no HTTP I/O; first
    :meth:`embed_texts` triggers :meth:`_ensure_loaded` which probes
    ``GET /api/tags`` and (when ``manage_server=True``) auto-spawns
    ``ollama serve`` if needed.
    """

    provider: str = "ollama"

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        embedding_dim: int = _DEFAULT_DIM,
        base_url: str | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
        manage_server: bool = True,
        startup_timeout_s: float = 30.0,
        # v1.9 Phase 45 Plan 04: workspace + run_id + stage_id enable
        # per-call LocalEmbeddingReceipt writes; all optional so existing
        # callers and tests work unchanged.
        workspace: Path | None = None,
        run_id: str | None = None,
        stage_id: str | None = None,
        _http: Any | None = None,
        _spawn: Callable[[str], Any] | None = None,
    ) -> None:
        self._model_name = model
        self._embedding_dim = int(embedding_dim)
        self._batch_size_override = batch_size
        self._normalize = bool(normalize)
        self._manage_server = bool(manage_server)
        self._startup_timeout_s = float(startup_timeout_s)
        self._workspace = workspace
        self._run_id = run_id
        self._stage_id = stage_id
        resolved = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or _OLLAMA_DEFAULT_BASE_URL
        ).rstrip("/")
        if not is_loopback_url(resolved):
            raise EmbeddingError(
                "non_loopback_base_url: OllamaEmbeddingClient requires a "
                "loopback endpoint (127.0.0.1, localhost, or ::1); got "
                f"{resolved!r}."
            )
        self._base_url = resolved
        self._endpoint_label = _redact_loopback(resolved)
        self._http_override = _http
        self._spawn_override = _spawn
        self._client: Any | None = None
        self._input_tokens = 0
        self._request_count = 0
        # v1.9 Phase 44 review ME-05: surfaced via usage() and logged once.
        self._actual_dim: int | None = None
        self._dim_mismatch_warned = False

    # ------------------------------------------------------------------
    # Protocol attributes
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def endpoint_label(self) -> str:
        return self._endpoint_label

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return
        client = self._http_override
        # v1.9 Phase 44 review ME-01: ensure the httpx client is closed if
        # the reachability/spawn probe raises (otherwise the socket leaks).
        owns_client = False
        if client is None:
            try:
                import httpx  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - httpx is transitive
                raise EmbeddingError(
                    "OllamaEmbeddingClient requires httpx (transitively "
                    "available via the OpenAI SDK in conference-pipeline)."
                ) from exc
            client = httpx.Client(base_url=self._base_url, timeout=30.0)
            owns_client = True
        try:
            if not self._probe_reachable(client, timeout_s=2.0):
                if self._manage_server:
                    spawn = self._spawn_override or _default_spawn_ollama_serve
                    proc = spawn(self._base_url)
                    deadline = time.monotonic() + self._startup_timeout_s
                    while time.monotonic() < deadline:
                        # v1.9 Phase 44 review HI-04: detect a crashed
                        # subprocess immediately rather than waiting out the
                        # full startup_timeout_s with a misleading
                        # "unreachable" error.
                        if proc is not None and hasattr(proc, "poll") and proc.poll() is not None:
                            raise EmbeddingError(
                                "ollama_serve_crashed: spawned `ollama serve` "
                                f"exited with code {proc.returncode!r} before "
                                "becoming reachable. See "
                                f"{_OLLAMA_SERVE_LOG_PATH} for diagnostics. "
                                f"endpoint_label={self._endpoint_label}."
                            )
                        if self._probe_reachable(client, timeout_s=1.0):
                            break
                        time.sleep(0.5)
                    else:
                        raise EmbeddingError(
                            "ollama_unreachable: spawned `ollama serve` did not "
                            f"become reachable within {self._startup_timeout_s}s "
                            f"at endpoint_label={self._endpoint_label}. "
                            f"See {_OLLAMA_SERVE_LOG_PATH} for diagnostics."
                        )
                else:
                    raise EmbeddingError(
                        "ollama_unreachable: endpoint not reachable and "
                        "manage_server=False. Start `ollama serve` manually or "
                        "pass manage_server=True. "
                        f"endpoint_label={self._endpoint_label}."
                    )
        except Exception:
            if owns_client and hasattr(client, "close"):
                try:
                    client.close()
                except Exception:  # noqa: BLE001 - cleanup best-effort
                    pass
            raise
        self._client = client

    def _probe_reachable(self, client: Any, *, timeout_s: float) -> bool:
        try:
            response = client.get("/api/tags", timeout=timeout_s)
            return bool(getattr(response, "status_code", 200) < 500)
        except Exception:  # noqa: BLE001 - reachability probe is best-effort
            return False

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        assert self._client is not None
        batch_size = effective_batch_size(
            "ollama", self._model_name, override=self._batch_size_override
        )
        # v1.9 review HIGH-2: capture starting counter so the per-call
        # receipt records *this* call's request count, not the cumulative
        # total across the client's lifetime.
        request_count_before = self._request_count
        all_vectors: list[list[float]] = []
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            for text in batch:
                response = self._client.post(
                    "/api/embeddings",
                    json={"model": self._model_name, "prompt": text},
                )
                # raise_for_status if the fake exposes it; real httpx does.
                raise_for_status = getattr(response, "raise_for_status", None)
                if callable(raise_for_status):
                    raise_for_status()
                payload = response.json()
                vec = list(payload.get("embedding") or [])
                if self._normalize and vec:
                    vec = _l2_normalize_one(vec)
                all_vectors.append([float(v) for v in vec])
                self._request_count += 1
                self._input_tokens += max(1, len(text.split()))
        # v1.9 Phase 44 review ME-05: declared embedding_dim is the contract;
        # observed mismatch logs a warning and tracks actual dim out-of-band
        # rather than silently mutating (which would defeat Phase 45's
        # COMPAT-02 dim-mismatch fail-stop). Mirrors MLX-4 fix in
        # mlx_embeddings.py.
        if all_vectors and len(all_vectors[0]) != self._embedding_dim:
            actual_dim = len(all_vectors[0])
            if not self._dim_mismatch_warned:
                LOG.warning(
                    "ollama_embeddings.dim_mismatch declared=%d actual=%d "
                    "model=%s — declared embedding_dim is now stale; update "
                    "config to match.",
                    self._embedding_dim,
                    actual_dim,
                    self._model_name,
                )
                self._dim_mismatch_warned = True
            self._actual_dim = actual_dim
        # v1.9 Phase 45 Plan 04: per-call LocalEmbeddingReceipt write
        # (closes Phase 44 review BL-01 audit-trail gap). No-op when
        # workspace=None.
        record_local_embedding_call(
            provider="ollama",
            model=self._model_name,
            embedding_dim=self._actual_dim or self._embedding_dim,
            endpoint_label=self._endpoint_label,
            input_count=len(texts),
            total_input_chars=sum(len(t) for t in texts),
            request_count=self._request_count - request_count_before,
            cost_usd=0.0,
            run_id=self._run_id,
            stage_id=self._stage_id,
            workspace=self._workspace,
        )
        return all_vectors

    def embed_texts_array(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch through Ollama's multi-input ``/api/embed`` endpoint.

        The legacy ``/api/embeddings`` endpoint accepts one prompt per
        request. Corpus materialization processes hundreds of thousands of
        chunks, so use the batch endpoint when available while retaining the
        existing single-input method for compatibility with older callers and
        tests.
        """

        if not texts:
            return []
        self._ensure_loaded()
        assert self._client is not None
        batch_size = effective_batch_size(
            "ollama", self._model_name, override=self._batch_size_override
        )
        request_count_before = self._request_count
        all_vectors: list[list[float]] = []
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            response = self._client.post(
                "/api/embed",
                json={"model": self._model_name, "input": batch},
            )
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            payload = response.json()
            raw_embeddings = payload.get("embeddings") or []
            if len(raw_embeddings) != len(batch):
                raise EmbeddingError(
                    "ollama_batch_embedding_count_mismatch:"
                    f"{len(raw_embeddings)}/{len(batch)}"
                )
            for raw_vector, text in zip(raw_embeddings, batch):
                vector = [float(value) for value in raw_vector]
                if self._normalize and vector:
                    vector = _l2_normalize_one(vector)
                all_vectors.append(vector)
                self._input_tokens += max(1, len(text.split()))
            self._request_count += 1

        if all_vectors and len(all_vectors[0]) != self._embedding_dim:
            actual_dim = len(all_vectors[0])
            if not self._dim_mismatch_warned:
                LOG.warning(
                    "ollama_embeddings.dim_mismatch declared=%d actual=%d "
                    "model=%s — declared embedding_dim is now stale; update "
                    "config to match.",
                    self._embedding_dim,
                    actual_dim,
                    self._model_name,
                )
                self._dim_mismatch_warned = True
            self._actual_dim = actual_dim
        record_local_embedding_call(
            provider="ollama",
            model=self._model_name,
            embedding_dim=self._actual_dim or self._embedding_dim,
            endpoint_label=self._endpoint_label,
            input_count=len(texts),
            total_input_chars=sum(len(text) for text in texts),
            request_count=self._request_count - request_count_before,
            cost_usd=0.0,
            run_id=self._run_id,
            stage_id=self._stage_id,
            workspace=self._workspace,
        )
        return all_vectors

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def usage(self) -> dict[str, Any]:
        return {
            "provider": "ollama",
            "model": self._model_name,
            "input_tokens": self._input_tokens,
            "request_count": self._request_count,
            "estimated_cost_usd": 0.0,
            "provider_receipt_paths": [],
        }

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 - close is best-effort
                pass
            self._client = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalize_one(vec: list[float]) -> list[float]:
    norm = sum(float(v) * float(v) for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [float(v) / norm for v in vec]


__all__ = ["OllamaEmbeddingClient"]
