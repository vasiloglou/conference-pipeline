"""OpenAI embedding-client adapter for the enhanced-indexing-vector-store stage.

Refuses to make any provider call when ``OPENAI_API_KEY`` is unset (see
``T-14B-02`` in 14B-01-PLAN.md). All exceptions at the provider boundary are
caught and rewrapped as :class:`EmbeddingError` with a sanitized message so
receipts cannot leak API key fragments.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import warnings
from typing import Any, Literal

from conference_pipeline.iclr2025_live import sanitize_for_receipt
from conference_pipeline.provider_receipts import (
    openai_embeddings_create,
)

LOG = logging.getLogger("conference_pipeline.indexing.embeddings")

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_EMBEDDING_DIM = 1536

Provider = Literal["openai", "mlx", "ollama", "fake"]
"""Embedding-provider identifiers consumed by :func:`build_embedding_client`."""

_VALID_PROVIDERS: tuple[str, ...] = ("openai", "mlx", "ollama", "fake")
_BILLIONS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")
_PROVIDER_DEFAULT_BATCH: dict[str, int] = {
    "mlx": 96,
    "ollama": 32,
    "openai": 128,
    "fake": 64,
}


class EmbeddingError(RuntimeError):
    """Sanitized wrapper raised at the embedding-client boundary.

    The string form is always pre-sanitized via ``sanitize_for_receipt`` so
    callers can record it directly into a receipt without further redaction.
    """


def _redact(message: str | Exception) -> str:
    sanitized = sanitize_for_receipt(str(message))
    return sanitized if isinstance(sanitized, str) else "redacted"


def _resolve_pricing_per_1m(model: str) -> float:
    """Return USD per 1M input tokens for ``model`` from the pricing catalog."""
    try:
        from conference_pipeline.costs.estimator import load_pricing_catalog

        catalog = load_pricing_catalog()
    except Exception:  # noqa: BLE001 - pricing must stay non-blocking
        return 0.02
    providers = catalog.get("providers") if isinstance(catalog, dict) else None
    if not isinstance(providers, dict):
        return 0.02
    openai = providers.get("openai")
    if not isinstance(openai, dict):
        return 0.02
    models = openai.get("models")
    if not isinstance(models, dict):
        return 0.02
    entry = models.get(model)
    if not isinstance(entry, dict):
        return 0.02
    price = entry.get("input_per_1m_usd")
    try:
        return float(price)
    except (TypeError, ValueError):
        return 0.02


class _LegacyEmbedMixin:
    """One-cycle compatibility shim. Removed in v1.10.

    Provides ``embed(texts)`` that emits :class:`DeprecationWarning`
    and delegates to ``embed_texts(texts)``. Concrete embedding
    clients inherit from this mixin so external callers (notebooks,
    integration scripts) that haven't migrated keep working through
    v1.9 with a clear migration warning.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        warnings.warn(
            "EmbeddingClient.embed() is deprecated; use embed_texts() "
            "instead. The shim will be removed in v1.10.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.embed_texts(texts)  # type: ignore[attr-defined]


class OpenAIEmbeddingClient(_LegacyEmbedMixin):
    """Implements ``analysis.steps.vector_store_setup.EmbeddingClient`` Protocol.

    The adapter:

    * Reads ``OPENAI_API_KEY`` from environment; raises
      :class:`EmbeddingError` when unset (no provider call attempted).
    * Bound by ``EMBEDDING_MAX_BATCH`` (default 128).
    * Retries transient errors with exponential backoff (max 3 retries).
    * Sanitizes any exception via :func:`sanitize_for_receipt` before
      re-raising as :class:`EmbeddingError`.
    * Records ``input_tokens``, ``request_count``, ``estimated_cost_usd``
      via :meth:`usage`.
    """

    provider: str = "openai"

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        max_batch: int | None = None,
        max_retries: int = 3,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise EmbeddingError("missing_api_key")
        self._api_key = resolved_key
        self._model = model
        self._embedding_dim = _DEFAULT_EMBEDDING_DIM
        self._max_batch = (
            max_batch
            if max_batch is not None
            else int(os.environ.get("EMBEDDING_MAX_BATCH", "128"))
        )
        self._max_retries = max(0, int(max_retries))
        self._input_tokens = 0
        self._request_count = 0
        self._price_per_1m = _resolve_pricing_per_1m(model)
        self._client: Any | None = None
        self._receipt_paths: list[str] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import-not-found]
            except Exception as exc:  # noqa: BLE001 - SDK import is the boundary
                raise EmbeddingError(_redact(f"openai_sdk_unavailable: {exc}")) from None
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._ensure_client()
        results: list[list[float]] = []
        for batch_start in range(0, len(texts), self._max_batch):
            batch = texts[batch_start : batch_start + self._max_batch]
            results.extend(self._embed_batch(client, batch))
        return results

    def _embed_batch(self, client: Any, batch: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = openai_embeddings_create(client, caller=__file__, model=self._model, input=batch)
                if response.receipt.receipt_path:
                    self._receipt_paths.append(response.receipt.receipt_path)
                self._request_count += 1
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self._input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                vectors: list[list[float]] = []
                for item in response.data:  # type: ignore[attr-defined]
                    vec = list(item.embedding)
                    vectors.append([float(v) for v in vec])
                if vectors and len(vectors[0]) != self._embedding_dim:
                    self._embedding_dim = len(vectors[0])
                return vectors
            except Exception as exc:  # noqa: BLE001 - boundary catch
                last_exc = exc
                LOG.warning(
                    "indexing.embedding_batch_retry attempt=%d redacted=%s",
                    attempt,
                    _redact(exc),
                )
                if attempt >= self._max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise EmbeddingError(_redact(last_exc or "unknown_embedding_error"))

    def usage(self) -> dict[str, Any]:
        cost_usd = round(self._input_tokens * self._price_per_1m / 1_000_000, 6)
        return {
            "provider": "openai",
            "model": self._model,
            "input_tokens": self._input_tokens,
            "request_count": self._request_count,
            "estimated_cost_usd": cost_usd,
            "provider_receipt_paths": list(self._receipt_paths),
        }


class FakeEmbeddingClient(_LegacyEmbedMixin):
    """Deterministic in-process embedding client for tests.

    Returns a 1536-dim vector keyed on a sha256 of the input string. Records
    usage like the real client so tests can assert on receipt fields without
    making a real provider call.
    """

    provider: str = "fake"

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
    ) -> None:
        self._model = model
        self._embedding_dim = embedding_dim
        self._input_tokens = 0
        self._request_count = 0

    @property
    def model(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._request_count += 1
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Tile the 32-byte digest to `embedding_dim` pseudo-floats in [-1, 1].
            tiled = (digest * ((self._embedding_dim // len(digest)) + 1))[
                : self._embedding_dim
            ]
            vec = [(byte - 128) / 128.0 for byte in tiled]
            vectors.append(vec)
            self._input_tokens += max(1, len(text.split()))
        return vectors

    def usage(self) -> dict[str, Any]:
        return {
            "provider": "fake",
            "model": self._model,
            "input_tokens": self._input_tokens,
            "request_count": self._request_count,
            "estimated_cost_usd": 0.0,
        }


def _chroma_embedding_function_base() -> type:
    """Resolve ``chromadb.api.types.EmbeddingFunction`` lazily.

    Lazy import so the module remains importable in environments where
    chromadb is not installed (the embedding-client adapter does not
    require chromadb at import time — only when the wrapped client is
    registered on a Chroma collection). Returns ``object`` as a fallback
    sentinel so static analysis does not break; subclass-only behavior
    (``__init_subclass__`` validation, ``is_legacy`` detection) only
    applies when the real chromadb base is available.
    """
    try:
        from chromadb.api.types import (
            EmbeddingFunction as _ChromaEmbeddingFunction,
        )
    except Exception:  # noqa: BLE001 - missing dep falls back to object
        return object
    return _ChromaEmbeddingFunction


class ChromaEmbeddingAdapter(_chroma_embedding_function_base()):  # type: ignore[misc]
    """chromadb 1.5.x EmbeddingFunction adapter for our embedding clients.

    Wraps any object exposing
    ``embed_texts(list[str]) -> list[list[float]]`` and exposes the
    chromadb >= 0.5 EmbeddingFunction protocol:

      * ``__call__(self, input)`` — chromadb passes documents
        positionally as ``input`` (a Sequence[str]); we coerce to
        ``list`` and delegate to the wrapped client.
      * ``name(self) -> str`` — required by chromadb >= 0.5 so the
        collection metadata can record which adapter produced the
        persisted vectors. Format:
        ``conference_pipeline.<provider>.<model>``.
      * ``get_config()`` / ``build_from_config()`` /
        ``default_space()`` / ``supported_spaces()`` — WR-05 fix:
        implement the minimum protocol so chromadb's ``is_legacy()``
        check returns False, eliminating the recurring
        ``DeprecationWarning("legacy embedding function config")`` AND
        the silent fall-back to the default ``l2`` HNSW space. Our
        embedding providers (OpenAI text-embedding-3-small, MLX BGE-M3,
        Ollama nomic) are normalised and used with **cosine**
        similarity, so we declare cosine as the default space.

    Phase 48 (D-14): registered on BOTH walker write and retriever
    read so the persisted Chroma collection round-trips through the
    same embedding function — chromadb's default 384-dim MiniLM
    never sees a 1024/1536-dim collection.

    Threat T-48-08 mitigation: ``__call__`` validates that every
    element of ``input`` is a ``str`` before delegating; non-str
    elements raise ``TypeError("chroma_embedding_adapter_invalid_input")``
    instead of allowing a malformed input to reach
    ``embed_texts``.
    """

    def __init__(self, client: Any) -> None:
        # Skip the parent's ``__init__`` (which warns about missing
        # constructor) — our concrete implementation accepts a wrapped
        # client and is fully initialized by setting ``_client``.
        self._client = client

    def __call__(self, input: Any) -> list[list[float]]:  # noqa: A002 - chromadb protocol param
        # chromadb passes documents positionally as ``input`` — keep
        # the parameter name verbatim. Coerce to list (chromadb may
        # pass any Sequence[str]).
        texts = list(input)
        for item in texts:
            if not isinstance(item, str):
                raise TypeError(
                    "chroma_embedding_adapter_invalid_input"
                )
        return self._client.embed_texts(texts)

    def embed_query(self, input: Any) -> list[list[float]]:  # noqa: A002 - chromadb protocol param
        """chromadb 1.5.x EmbeddingFunction protocol — ``query_texts=`` path.

        Phase 48 plan 02 task 3 (Rule 3 fix): chromadb 1.5.8 dispatches
        ``collection.query(query_texts=[...])`` through
        ``embedding_function.embed_query`` rather than ``__call__``.
        Without this method the round-trip fails with
        ``'ChromaEmbeddingAdapter' object has no attribute 'embed_query'``.
        Default implementation: delegate to ``__call__`` (the same
        ``embed_texts`` path) — our embedding clients use one model for
        both documents and queries, so document/query asymmetry does
        not apply.
        """
        return self.__call__(input)

    @staticmethod
    def name() -> str:  # type: ignore[override]
        """Return the stable class-level identifier for chromadb registration.

        WR-05: chromadb registers embedding functions via ``cls.name()``
        (not ``self.name()``) so this MUST be a classmethod/staticmethod.
        Provider/model identity is carried in :meth:`get_config` instead
        of in the name (the legacy instance-aware ``conference_pipeline.{provider}.{model}``
        format would fail registration with
        ``TypeError("name() missing 1 required positional argument: 'self'")``).
        """
        return "conference_pipeline.chroma_embedding_adapter"

    def get_config(self) -> dict[str, Any]:
        """Return a serializable adapter config (provider/model identity).

        WR-05: chromadb consults this on collection creation; returning a
        non-``NotImplemented`` value flips ``is_legacy()`` to False and
        avoids the legacy-config DeprecationWarning + silent l2-default.
        """
        provider = getattr(self._client, "provider", "unknown")
        model = getattr(self._client, "model", "unknown")
        return {"provider": str(provider), "model": str(model)}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "ChromaEmbeddingAdapter":
        """Reconstruct an adapter from its serialized config.

        WR-05: chromadb invokes this in two paths:

        1. ``is_legacy()`` validation — calls
           ``self.build_from_config(self.get_config())`` purely to verify
           the round-trip; the result is discarded.
        2. Collection rehydration via the registered EF table — only
           triggered when a caller queries a collection WITHOUT passing
           ``embedding_function=``. In production both walker write and
           retriever read pass the wrapped adapter explicitly (D-14), so
           the rehydrated EF is never actually invoked.

        Because the runtime EF is always supplied externally, we return a
        cheap ``FakeEmbeddingClient``-backed adapter here. This preserves
        the protocol contract (returns an adapter instance, not
        ``NotImplemented``) without requiring provider credentials at
        config-serialization time — credentials may not be available in
        the chromadb worker context, and raising here would re-trigger
        the legacy-config DeprecationWarning we are eliminating.
        """
        # Avoid touching real-provider constructors (which require
        # credentials) — a Fake client is sufficient because chromadb
        # never invokes the rehydrated adapter when the caller passes
        # embedding_function= explicitly (the walker and retriever
        # paths both do).
        provider = str((config or {}).get("provider") or "fake")
        model = (config or {}).get("model")
        fake = FakeEmbeddingClient(
            model=str(model) if model else _DEFAULT_MODEL,
            embedding_dim=_DEFAULT_EMBEDDING_DIM,
        )
        adapter = ChromaEmbeddingAdapter(fake)
        # Preserve the original provider identity in the adapter's
        # serialized config for diagnostic introspection.
        adapter._original_provider = provider  # noqa: SLF001
        return adapter

    def default_space(self) -> str:  # type: ignore[override]
        """Return ``"cosine"`` — our embedding providers are normalised.

        WR-05: without this override chromadb falls back to ``"l2"``
        (the protocol default), silently changing ranking semantics for
        OpenAI/MLX/Ollama-backed collections.
        """
        return "cosine"

    def supported_spaces(self) -> list[str]:  # type: ignore[override]
        """Allow cosine, ip, and l2 — operator may override per-collection."""
        return ["cosine", "ip", "l2"]


def _infer_model_billions(model: str) -> float:
    """Infer parameter count in billions from a model name (regex on ``\\d+b``).

    Returns ``0.0`` when no match — small / unknown models fall through to
    the per-provider default batch size.
    """
    match = _BILLIONS_RE.search(model)
    return float(match.group(1)) if match else 0.0


def effective_batch_size(
    provider: Provider, model: str, *, override: int | None = None
) -> int:
    """Return the effective batch size for ``(provider, model)``.

    Precedence (highest to lowest):

    1. ``override`` argument.
    2. ``EMBEDDING_BATCH_SIZE`` env var (must be a positive integer).
    3. Adaptive heuristic on inferred model size: ``≥8B`` → 2; ``≥3B`` → 8.
    4. Per-provider default (``mlx`` → 96, ``ollama`` → 32,
       ``openai`` → 128, ``fake`` → 64).
    """
    if override is not None and override > 0:
        return int(override)
    env = os.environ.get("EMBEDDING_BATCH_SIZE", "").strip()
    if env.isdigit():
        env_int = int(env)
        if env_int > 0:
            return env_int
    billions = _infer_model_billions(model)
    if billions >= 8:
        return 2
    if billions >= 3:
        return 8
    return _PROVIDER_DEFAULT_BATCH.get(provider, 32)


def build_embedding_client(config: dict) -> Any:
    """Construct the configured embedding client.

    ``config`` schema:

    - ``provider`` (required): one of :data:`Provider`.
    - ``model`` (optional): provider-specific model id; provider defaults
      apply when omitted.
    - ``embedding_dim`` (optional): override the inferred dim.
    - ``batch_size`` (optional): override the adaptive default.
    - ``max_length`` (optional): tokenizer hard max for local MLX inputs.
    - additional provider-specific fields (e.g. ``base_url``,
      ``runtime_profile``) are forwarded to the concrete client.

    Raises:
        ValueError: ``provider`` is missing or not in
            ``('openai', 'mlx', 'ollama', 'fake')``.
        NotImplementedError: ``provider`` lands in a future v1.9 phase
            (``mlx`` → Phase 43, ``ollama`` → Phase 44).
    """
    if not isinstance(config, dict):
        raise ValueError(
            "build_embedding_client(config) requires a dict; "
            f"got {type(config).__name__}"
        )
    provider = config.get("provider")
    if not provider:
        raise ValueError(
            "build_embedding_client requires config['provider']; "
            f"valid: 'openai', 'mlx', 'ollama', 'fake'"
        )
    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"unknown embedding provider {provider!r}; "
            f"valid: 'openai', 'mlx', 'ollama', 'fake'"
        )
    # v1.9 Phase 42 review HIGH-02: validate config['batch_size'] before
    # forwarding to the constructor. Without this, batch_size=-1 silently
    # produces zero embeddings via range(0, n, -1) inside embed_texts.
    raw_batch_size = config.get("batch_size")
    if raw_batch_size is not None and (
        not isinstance(raw_batch_size, int)
        or isinstance(raw_batch_size, bool)
        or raw_batch_size <= 0
    ):
        raise ValueError(
            "build_embedding_client: config['batch_size'] must be a "
            f"positive int; got {raw_batch_size!r}"
        )
    raw_dim = config.get("embedding_dim")
    if raw_dim is not None and (
        not isinstance(raw_dim, int)
        or isinstance(raw_dim, bool)
        or raw_dim <= 0
    ):
        raise ValueError(
            "build_embedding_client: config['embedding_dim'] must be a "
            f"positive int; got {raw_dim!r}"
        )
    model = config.get("model")
    if provider == "openai":
        return OpenAIEmbeddingClient(
            model=model or _DEFAULT_MODEL,
            api_key=config.get("api_key"),
            max_batch=config.get("batch_size"),
        )
    if provider == "fake":
        return FakeEmbeddingClient(
            model=model or _DEFAULT_MODEL,
            embedding_dim=int(config.get("embedding_dim") or _DEFAULT_EMBEDDING_DIM),
        )
    # Validate config['max_length'] before forwarding to MLXEmbeddingClient.
    raw_max_length = config.get("max_length")
    if provider == "mlx" and raw_max_length is not None and (
        not isinstance(raw_max_length, int)
        or isinstance(raw_max_length, bool)
        or raw_max_length <= 0
    ):
        raise ValueError(
            "build_embedding_client: config['max_length'] must be a "
            f"positive int; got {raw_max_length!r}"
        )
    if provider == "mlx":
        from .mlx_embeddings import _DEFAULT_MODEL as MLX_DEFAULT_MODEL, MLXEmbeddingClient

        return MLXEmbeddingClient(
            model=model or MLX_DEFAULT_MODEL,
            embedding_dim=int(config.get("embedding_dim") or 1024),
            batch_size=config.get("batch_size"),
            max_length=config.get("max_length"),
            normalize=bool(config.get("normalize", True)),
            _loader=config.get("_loader"),
        )
    if provider == "ollama":
        from .ollama_embeddings import OllamaEmbeddingClient

        return OllamaEmbeddingClient(
            model=model or "nomic-embed-text",
            embedding_dim=int(config.get("embedding_dim") or 768),
            base_url=config.get("base_url"),
            batch_size=config.get("batch_size"),
            normalize=bool(config.get("normalize", True)),
            manage_server=bool(config.get("manage_server", True)),
            startup_timeout_s=float(config.get("startup_timeout_s") or 30.0),
            _http=config.get("_http"),
            _spawn=config.get("_spawn"),
        )
    # Unreachable — _VALID_PROVIDERS guard above is exhaustive.
    raise ValueError(  # pragma: no cover
        f"unknown embedding provider {provider!r}"
    )


__all__ = [
    "ChromaEmbeddingAdapter",
    "EmbeddingError",
    "FakeEmbeddingClient",
    "OpenAIEmbeddingClient",
    "Provider",
    "build_embedding_client",
    "effective_batch_size",
]
