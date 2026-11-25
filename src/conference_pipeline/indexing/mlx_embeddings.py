"""MLX-backed embedding client for Apple Silicon (v1.9 Phase 43).

Implements the ``EmbeddingClient`` Protocol established in v1.9 Phase 42
using the ``mlx-embeddings`` PyPI package (NOT ``mlx-lm`` — that's for
text generation). Default model ``mlx-community/bge-m3-mlx-8bit`` (1024-dim).

The constructor performs no I/O — model weights are loaded lazily on
first embedding call via ``_ensure_loaded()``. For tests, a
loader callable can be injected via the constructor's ``_loader``
parameter, mirroring the project's Fake-test-double convention.

L2 normalization is on by default; pass ``normalize=False`` to opt out.

The ``usage()`` shape mirrors :class:`OpenAIEmbeddingClient.usage` so
downstream telemetry / receipt aggregators do not need to special-case
local providers. ``estimated_cost_usd`` is always ``0.0`` (the local
runtime never incurs paid cost) and ``provider_receipt_paths`` is
always empty (no HTTP receipt is written — vector-store-level
aggregation lands in v1.9 Phase 45).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from conference_pipeline.indexing.embeddings import (
    EmbeddingError,
    _LegacyEmbedMixin,
    _redact,
    effective_batch_size,
)
from conference_pipeline.model_governance.registry import model_for_alias
from conference_pipeline.provider_receipts import record_local_embedding_call

LOG = logging.getLogger("conference_pipeline.indexing.mlx_embeddings")

# The catalog alias, not a literal: the same governed bge-m3 entry the public search uses.
_DEFAULT_MODEL = model_for_alias("bge_m3_mlx")
_DEFAULT_DIM = 1024
# MLX's unified-memory buffer cache retains an allocation for every distinct
# tensor shape it evaluates. Because ``_tokenize_batch`` pads each batch to
# that batch's own longest text, a long indexing run performs tens of
# thousands of forward passes with almost all-distinct shapes, so the cache
# grows without bound by default — enough to exhaust system memory and take
# the whole machine down on a large corpus. Bound it, and release it between
# calls. Override with MLX_EMBEDDING_CACHE_LIMIT_BYTES (0 disables caching).
_DEFAULT_CACHE_LIMIT_BYTES = 2 * 1024**3
_CACHE_LIMIT_ENV = "MLX_EMBEDDING_CACHE_LIMIT_BYTES"


def _cached_snapshot_path(model_name: str) -> str | None:
    """The local snapshot of a cached hub model, without touching the network.

    Loading by repo id makes the hub client check online for a newer revision
    on every load. A cached model then fails wherever outbound network is
    blocked (the research engine's loopback-only egress guard) and pays a
    round trip everywhere else. Returns ``None`` when the name is already a
    path or the model is not cached, so the normal (downloading) load runs.
    """

    if not model_name or Path(model_name).exists():
        return None
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        return str(snapshot_download(repo_id=model_name, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached, or no hub client: fall back to the name
        return None


class MLXEmbeddingClient(_LegacyEmbedMixin):
    """Implements the v1.9 ``EmbeddingClient`` Protocol against ``mlx-embeddings``.

    Constructor performs no I/O. Pass ``_loader`` to inject a fake
    `(model, tokenizer)` factory in tests. Production calls import
    ``mlx_embeddings.utils.load`` inside :meth:`_ensure_loaded`.

    Endpoint label convention (consumed by Phase 45 receipts):
    ``mlx://<model_name>``.
    """

    provider: str = "mlx"

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        embedding_dim: int = _DEFAULT_DIM,
        batch_size: int | None = None,
        max_length: int | None = None,
        normalize: bool = True,
        # v1.9 Phase 45 Plan 04: workspace + run_id + stage_id enable
        # per-call LocalEmbeddingReceipt writes. All optional so existing
        # callers (and tests that don't care about receipts) work unchanged.
        workspace: Path | None = None,
        run_id: str | None = None,
        stage_id: str | None = None,
        _loader: Callable[[str], tuple[Any, Any]] | None = None,
    ) -> None:
        self._model_name = model
        self._embedding_dim = int(embedding_dim)
        self._batch_size_override = batch_size
        self._max_length = int(max_length or 2048)
        self._normalize = bool(normalize)
        self._loader_override = _loader
        self._workspace = workspace
        self._run_id = run_id
        self._stage_id = stage_id
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._input_tokens = 0
        self._request_count = 0
        self._endpoint_label = f"mlx://{model}"
        # v1.9 Phase 43 review MLX-4: surfaced via usage() and logged once.
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

    @property
    def max_length(self) -> int:
        return self._max_length

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True iff ``mlx_embeddings`` is importable here."""
        try:
            import mlx_embeddings  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        loader = self._loader_override
        if loader is None:
            try:
                from mlx_embeddings.utils import load  # type: ignore[import-not-found]
            except ImportError as exc:
                raise EmbeddingError(
                    "MLXEmbeddingClient requires the mlx-embeddings package. "
                    'Install with: pip install -e ".[mlx]"'
                ) from exc
            loader = load
            model_ref = _cached_snapshot_path(self._model_name) or self._model_name
        else:
            model_ref = self._model_name
        # v1.9 Phase 43 review MLX-2: real mlx_embeddings.utils.load can
        # raise HuggingFace download errors, OSError (disk full), gated-repo
        # auth errors, etc. — all may include credentials, paths, or URLs.
        # Mirror OpenAIEmbeddingClient / OllamaEmbeddingClient sanitization
        # so receipts can never leak the raw exception text.
        try:
            self._model, self._tokenizer = loader(model_ref)
        except Exception as exc:  # noqa: BLE001 - boundary catch
            raise EmbeddingError(
                _redact(f"mlx_load_failed model={self._model_name}: {exc}")
            ) from None
        self._apply_metal_cache_limit()

    @staticmethod
    def _mlx_core() -> Any | None:
        """Return ``mlx.core`` when importable, else ``None`` (non-Mac CI)."""
        try:
            import mlx.core as mx  # type: ignore[import-not-found]
        except ImportError:
            return None
        return mx

    def _apply_metal_cache_limit(self) -> None:
        """Bound MLX's buffer cache so it cannot grow past a fixed ceiling."""
        mx = self._mlx_core()
        if mx is None:
            return
        raw = os.environ.get(_CACHE_LIMIT_ENV, "").strip()
        limit = _DEFAULT_CACHE_LIMIT_BYTES
        if raw:
            try:
                limit = max(0, int(raw))
            except ValueError:
                LOG.warning(
                    "mlx_embeddings.invalid_cache_limit %s=%r; using default %d",
                    _CACHE_LIMIT_ENV,
                    raw,
                    _DEFAULT_CACHE_LIMIT_BYTES,
                )
        setter = getattr(mx, "set_cache_limit", None)
        if not callable(setter):
            setter = getattr(getattr(mx, "metal", None), "set_cache_limit", None)
        if callable(setter):
            try:
                setter(limit)
            except Exception as exc:  # noqa: BLE001 - cache tuning must not break embedding
                LOG.warning("mlx_embeddings.set_cache_limit_failed: %s", exc)

    def _release_metal_cache(self) -> None:
        """Free cached MLX buffers accumulated by this call's batch shapes."""
        mx = self._mlx_core()
        if mx is None:
            return
        clearer = getattr(mx, "clear_cache", None)
        if not callable(clearer):
            clearer = getattr(getattr(mx, "metal", None), "clear_cache", None)
        if callable(clearer):
            try:
                clearer()
            except Exception as exc:  # noqa: BLE001 - cache release must not break embedding
                LOG.warning("mlx_embeddings.clear_cache_failed: %s", exc)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _tokenize_batch(self, batch: list[str]) -> Any:
        assert self._tokenizer is not None
        return self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=self._max_length,
        )

    def _iter_tokenized_batches(
        self,
        texts: list[str],
        *,
        batch_size: int,
    ) -> Iterator[tuple[list[str], Any]]:
        """Yield tokenized batches, optionally pre-tokenizing one batch ahead."""
        enabled = os.environ.get("MLX_EMBEDDING_PRETOKENIZE_PIPELINE", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if not enabled:
            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start : batch_start + batch_size]
                yield batch, self._tokenize_batch(batch)
            return

        tokenized_queue: queue.Queue[tuple[list[str] | None, Any]] = queue.Queue(maxsize=2)
        sentinel: tuple[list[str] | None, Any] = (None, None)

        def _producer() -> None:
            try:
                for batch_start in range(0, len(texts), batch_size):
                    batch = texts[batch_start : batch_start + batch_size]
                    tokenized_queue.put((batch, self._tokenize_batch(batch)))
            except BaseException as exc:  # noqa: BLE001 - re-raised on consumer thread
                tokenized_queue.put((None, exc))
            finally:
                tokenized_queue.put(sentinel)

        thread = threading.Thread(
            target=_producer,
            name="mlx-embedding-pretokenizer",
            daemon=True,
        )
        thread.start()
        while True:
            batch, tokenized_or_error = tokenized_queue.get()
            if batch is None:
                thread.join()
                if isinstance(tokenized_or_error, BaseException):
                    raise tokenized_or_error
                break
            yield batch, tokenized_or_error

    def embed_texts_array(self, texts: list[str]) -> np.ndarray:
        """Return embeddings as a ``float32`` NumPy array."""
        if not texts:
            return np.empty((0, self._embedding_dim), dtype=np.float32)
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        batch_size = effective_batch_size(
            "mlx", self._model_name, override=self._batch_size_override
        )
        # v1.9 review HIGH-2: capture starting counter so the per-call
        # receipt records *this* call's request count, not the cumulative
        # total across the client's lifetime.
        request_count_before = self._request_count
        batch_arrays: list[np.ndarray] = []
        try:
            for batch, tokenized in self._iter_tokenized_batches(texts, batch_size=batch_size):
                try:
                    import mlx.core as mx  # type: ignore[import-not-found]

                    input_ids = mx.array(tokenized["input_ids"])
                    attention = tokenized.get("attention_mask") if hasattr(tokenized, "get") else None
                    attention_mask = mx.array(attention) if attention is not None else None
                    output = self._model(input_ids, attention_mask=attention_mask)
                except (ImportError, TypeError):
                    # Older tests and some compatible wrappers accept the whole
                    # tokenized mapping. Real mlx-embeddings 0.1.x Qwen models
                    # expect input_ids/attention_mask arrays instead.
                    output = self._model(tokenized)
                embeddings_obj = getattr(output, "text_embeds", output)
                try:
                    embeddings = np.asarray(embeddings_obj, dtype=np.float32)
                except RuntimeError:
                    # MLX bfloat arrays may expose a buffer dtype numpy cannot
                    # consume directly. ``tolist`` gives numpy plain Python floats.
                    embeddings = np.asarray(embeddings_obj.tolist(), dtype=np.float32)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                if self._normalize:
                    embeddings = _l2_normalize(embeddings)
                batch_arrays.append(embeddings.astype(np.float32, copy=False))
                self._request_count += 1
                self._input_tokens += sum(len(text.split()) for text in batch)
        finally:
            # Padded batch shapes are effectively unique per call, so cached
            # buffers rarely get reused across calls — release them instead of
            # letting the cache climb toward the system memory ceiling.
            self._release_metal_cache()
        all_vectors = (
            np.concatenate(batch_arrays, axis=0).astype(np.float32, copy=False)
            if batch_arrays
            else np.empty((0, self._embedding_dim), dtype=np.float32)
        )
        # v1.9 Phase 43 review MLX-4: previously this silently mutated
        # ``self._embedding_dim`` on first call when the real model returned
        # a different dim. That defeats the v1.9 Phase 45 ``COMPAT-02``
        # dim-mismatch fail-stop intent. Now: log a warning and surface the
        # discrepancy via :meth:`usage` instead of mutating in place. The
        # declared ``embedding_dim`` is the contract; if it differs from
        # reality the operator must update config explicitly.
        if len(all_vectors) and int(all_vectors.shape[1]) != self._embedding_dim:
            actual_dim = int(all_vectors.shape[1])
            if not self._dim_mismatch_warned:
                LOG.warning(
                    "mlx_embeddings.dim_mismatch declared=%d actual=%d "
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
        # workspace=None (unit tests + ad-hoc usage).
        record_local_embedding_call(
            provider="mlx",
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

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.embed_texts_array(texts)
        return embeddings.astype(float, copy=False).tolist()

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def usage(self) -> dict[str, Any]:
        return {
            "provider": "mlx",
            "model": self._model_name,
            "input_tokens": self._input_tokens,
            "request_count": self._request_count,
            "estimated_cost_usd": 0.0,
            "provider_receipt_paths": [],
            "max_length": self._max_length,
        }

    def close(self) -> None:
        """Release model + tokenizer references (test cleanup helper)."""
        self._model = None
        self._tokenizer = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize each row of ``arr``; rows with zero norm pass through."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32, copy=False)


__all__ = ["MLXEmbeddingClient"]
