"""Phase 43 — MLXEmbeddingClient unit tests.

Tests use a deterministic injected ``_loader`` so the suite passes
without the ``mlx-embeddings`` package installed (which is the case in
non-Mac CI). An additional test asserts the import-failure path
produces the documented error message when no loader is injected.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from conference_pipeline.indexing.embeddings import EmbeddingError
from conference_pipeline.indexing.mlx_embeddings import MLXEmbeddingClient


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Records each call so tests can assert per-batch sizes."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.kwargs_calls: list[dict[str, Any]] = []

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool = True,  # noqa: ARG002
        truncation: bool = True,  # noqa: ARG002
        return_tensors: str = "np",  # noqa: ARG002
        max_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        self.calls.append(list(texts))
        self.kwargs_calls.append(
            {
                "padding": padding,
                "truncation": truncation,
                "return_tensors": return_tensors,
                "max_length": max_length,
            }
        )
        return {"input_ids": np.zeros((len(texts), 4), dtype=int)}


class _FakeModel:
    """Returns deterministic per-row vectors of dim 1024.

    Row i contains values [i, i+1, ..., i+1023] / 1000. Distinct rows
    so L2 norms differ when un-normalized.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.invocations = 0

    def __call__(self, tokenized: dict[str, np.ndarray]) -> np.ndarray:
        self.invocations += 1
        n = int(tokenized["input_ids"].shape[0])
        rows = []
        for i in range(n):
            base = float(self.invocations * 100 + i)
            rows.append(np.arange(self.dim, dtype=float) / 1000.0 + base)
        return np.asarray(rows)


def _make_fake_loader() -> tuple[Any, _FakeTokenizer, _FakeModel]:
    """Returns (loader_callable, captured_tokenizer, captured_model)."""
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    def _loader(model_name: str) -> tuple[Any, Any]:  # noqa: ARG001
        return model, tokenizer

    return _loader, tokenizer, model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_constructor_does_no_io():
    """No I/O at construction; model + tokenizer are None until first call."""
    client = MLXEmbeddingClient(model="not-a-real-model-name-zzz")
    assert client._model is None
    assert client._tokenizer is None
    assert client.model == "not-a-real-model-name-zzz"
    assert client.endpoint_label == "mlx://not-a-real-model-name-zzz"
    assert client.provider == "mlx"


def test_ensure_loaded_raises_clear_error_when_mlx_missing_and_no_loader():
    """Default path imports mlx_embeddings; absent → EmbeddingError with [mlx] hint."""
    if MLXEmbeddingClient.is_available():
        pytest.skip("mlx_embeddings is installed; cannot assert missing-package path")
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b")
    with pytest.raises(EmbeddingError) as exc_info:
        client.embed_texts(["hello"])
    msg = str(exc_info.value)
    assert "mlx-embeddings" in msg
    assert '[mlx]' in msg


def test_embed_texts_uses_injected_loader():
    loader, tokenizer, model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=loader)
    vectors = client.embed_texts(["alpha", "beta"])
    assert len(vectors) == 2
    assert all(len(v) == 1024 for v in vectors)
    assert tokenizer.calls == [["alpha", "beta"]]
    assert model.invocations == 1


def test_embed_texts_normalizes_by_default():
    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=loader)
    vectors = client.embed_texts(["a", "b", "c"])
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6


def test_embed_texts_array_returns_normalized_float32_array():
    loader, tokenizer, model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=loader)

    vectors = client.embed_texts_array(["a", "b", "c"])

    assert vectors.dtype == np.float32
    assert vectors.shape == (3, 1024)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(3), rtol=1e-6)
    assert tokenizer.calls == [["a", "b", "c"]]
    assert model.invocations == 1
    assert client.usage()["request_count"] == 1


def test_embed_texts_wraps_array_fast_path():
    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", embedding_dim=4, _loader=loader)

    def fake_array(texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)

    client.embed_texts_array = fake_array  # type: ignore[method-assign]

    assert client.embed_texts(["a", "b"]) == [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]


def test_embed_texts_can_disable_normalization():
    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(
        model="qwen3-embedding-0.6b", normalize=False, _loader=loader
    )
    vectors = client.embed_texts(["a", "b"])
    norms = [math.sqrt(sum(x * x for x in v)) for v in vectors]
    # Fake model returns large unnormalized vectors; norms should be far from 1.
    assert all(n > 10 for n in norms), norms


def test_embed_texts_respects_explicit_batch_size():
    """batch_size=2 over 5 inputs → tokenizer called with sizes [2, 2, 1]."""
    loader, tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(
        model="qwen3-embedding-0.6b", batch_size=2, _loader=loader
    )
    vectors = client.embed_texts(["x1", "x2", "x3", "x4", "x5"])
    assert len(vectors) == 5
    assert [len(call) for call in tokenizer.calls] == [2, 2, 1]


def test_embed_texts_array_supports_optional_pretokenize_pipeline(monkeypatch):
    monkeypatch.setenv("MLX_EMBEDDING_PRETOKENIZE_PIPELINE", "1")
    loader, tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(
        model="qwen3-embedding-0.6b", batch_size=2, _loader=loader
    )

    vectors = client.embed_texts_array(["x1", "x2", "x3", "x4", "x5"])

    assert vectors.shape == (5, 1024)
    assert [len(call) for call in tokenizer.calls] == [2, 2, 1]


def test_embed_texts_passes_default_max_length_to_tokenizer():
    loader, tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=loader)

    client.embed_texts(["alpha"])

    assert tokenizer.kwargs_calls[0]["max_length"] == 2048


def test_embed_texts_passes_explicit_max_length_to_tokenizer():
    loader, tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(
        model="qwen3-embedding-0.6b",
        max_length=1024,
        _loader=loader,
    )

    client.embed_texts(["alpha"])

    assert client.max_length == 1024
    assert tokenizer.kwargs_calls[0]["max_length"] == 1024


def test_usage_shape_parity_with_openai():
    """usage() keys + provider/cost contract match OpenAIEmbeddingClient.usage()."""
    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=loader)
    client.embed_texts(["alpha word", "beta word"])
    usage = client.usage()
    assert set(usage.keys()) == {
        "provider",
        "model",
        "input_tokens",
        "request_count",
        "estimated_cost_usd",
        "provider_receipt_paths",
        "max_length",
    }
    assert usage["provider"] == "mlx"
    assert usage["model"] == "qwen3-embedding-0.6b"
    assert usage["estimated_cost_usd"] == 0.0
    assert usage["provider_receipt_paths"] == []
    assert usage["max_length"] == 2048
    assert usage["request_count"] == 1
    assert usage["input_tokens"] >= 4  # "alpha word" + "beta word" = 4 tokens


def test_is_available_returns_false_when_mlx_embeddings_missing():
    """In the dev env (no mlx_embeddings installed), is_available() is False."""
    # Sanity: confirm the package is genuinely absent.
    try:
        import mlx_embeddings  # noqa: F401
        pytest.skip("mlx_embeddings happens to be installed; cannot assert False")
    except ImportError:
        pass
    assert MLXEmbeddingClient.is_available() is False


def test_legacy_embed_shim_works_on_mlx_client():
    """Phase 42's deprecation shim is inherited by MLXEmbeddingClient."""
    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=loader)
    with pytest.warns(DeprecationWarning, match=r"embed_texts"):
        result = client.embed(["hello"])
    assert len(result) == 1
    assert len(result[0]) == 1024


# ---------------------------------------------------------------------------
# v1.9 Phase 43 review fixes
# ---------------------------------------------------------------------------


def test_loader_exception_wrapped_as_embedding_error_with_redaction():
    """v1.9 Phase 43 review MLX-2: real loader failures (HF download
    error, OSError, gated-repo auth, etc.) must be wrapped in
    EmbeddingError so receipts can never leak the raw message.
    """
    fake_secret = "sk-LEAK_TEST_KEY_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"

    def boom_loader(model_name: str):  # noqa: ARG001
        raise RuntimeError(
            f"HF download failed: 401 Unauthorized auth={fake_secret}"
        )

    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=boom_loader)
    with pytest.raises(EmbeddingError) as exc_info:
        client.embed_texts(["x"])
    msg = str(exc_info.value)
    assert "mlx_load_failed" in msg
    assert fake_secret not in msg, (
        "loader-exception text leaked through unredacted; sanitization missing"
    )


def test_dim_mismatch_does_not_silently_mutate_declared_dim(caplog):
    """v1.9 Phase 43 review MLX-4: declared embedding_dim is the contract.
    If the model returns vectors of a different dim, log a warning and
    track the actual dim out-of-band — do NOT silently rewrite the
    declared value (which would defeat Phase 45's COMPAT-02 dim guard).
    """
    import logging

    # Loader returns 64-dim vectors but client declares 1024-dim.
    class _SmallTokenizer:
        def __call__(self, texts, **_kwargs):
            return {"input_ids": np.zeros((len(texts), 4), dtype=int)}

    class _SmallModel:
        def __call__(self, tokenized):
            n = int(tokenized["input_ids"].shape[0])
            return np.tile(np.arange(64, dtype=float), (n, 1))

    def small_loader(model_name: str):  # noqa: ARG001
        return _SmallModel(), _SmallTokenizer()

    client = MLXEmbeddingClient(
        model="qwen3-embedding-0.6b",
        embedding_dim=1024,  # declared
        _loader=small_loader,
    )
    with caplog.at_level(logging.WARNING):
        vectors = client.embed_texts(["a"])
    # Declared dim is the contract — must NOT mutate.
    assert client.embedding_dim == 1024
    # Actual dim is tracked out-of-band.
    assert client._actual_dim == 64
    # Warning logged exactly once.
    mismatch_records = [r for r in caplog.records if "dim_mismatch" in r.message]
    assert len(mismatch_records) == 1
    # Second call: warning is not repeated.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        client.embed_texts(["b"])
    assert not [r for r in caplog.records if "dim_mismatch" in r.message]
    # Vector shape unchanged.
    assert len(vectors[0]) == 64


def test_dim_match_does_not_log_warning(caplog):
    """Happy path: declared dim matches actual; no warning."""
    import logging

    loader, _tokenizer, _model = _make_fake_loader()  # returns 1024-dim
    client = MLXEmbeddingClient(
        model="qwen3-embedding-0.6b",
        embedding_dim=1024,
        _loader=loader,
    )
    with caplog.at_level(logging.WARNING):
        client.embed_texts(["a"])
    assert client._actual_dim is None
    assert not [r for r in caplog.records if "dim_mismatch" in r.message]


# ---------------------------------------------------------------------------
# v1.9 Phase 45 Plan 04 — LocalEmbeddingReceipt wired in
# ---------------------------------------------------------------------------


def test_embed_texts_writes_local_embedding_receipt_when_workspace_provided(
    tmp_path,
):
    """Phase 45 Plan 04: per-call receipt closes BL-01 audit-trail gap."""
    import json

    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(
        model="qwen3-embedding-0.6b",
        _loader=loader,
        workspace=tmp_path,
        run_id="run-abc",
        stage_id="vector-store-setup",
    )
    client.embed_texts(["alpha word", "beta word"])
    receipt_dir = (
        tmp_path / "storage" / "workflow-runs" / "run-abc"
        / "local-embedding-receipts" / "vector-store-setup"
    )
    receipts = list(receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["provider"] == "mlx"
    assert payload["model"] == "qwen3-embedding-0.6b"
    assert payload["input_count"] == 2
    assert payload["cost_usd"] == 0.0
    assert payload["endpoint_label"] == "mlx://qwen3-embedding-0.6b"


def test_embed_texts_no_receipt_when_workspace_omitted():
    """No-op when workspace=None; no exceptions, no file writes."""
    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(
        model="qwen3-embedding-0.6b", _loader=loader,
    )
    # workspace not passed — receipt write is a no-op.
    vectors = client.embed_texts(["hello"])
    assert len(vectors) == 1


# ---------------------------------------------------------------------------
# Metal buffer-cache management (machine-OOM regression)
# ---------------------------------------------------------------------------


class _FakeMxCore:
    """Stands in for ``mlx.core`` to observe cache-management calls."""

    def __init__(self) -> None:
        self.cache_limits: list[int] = []
        self.clear_calls = 0

    def set_cache_limit(self, limit: int) -> None:
        self.cache_limits.append(int(limit))

    def clear_cache(self) -> None:
        self.clear_calls += 1


def test_load_applies_default_metal_cache_limit(monkeypatch):
    monkeypatch.delenv("MLX_EMBEDDING_CACHE_LIMIT_BYTES", raising=False)
    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=loader)
    fake_mx = _FakeMxCore()
    monkeypatch.setattr(MLXEmbeddingClient, "_mlx_core", staticmethod(lambda: fake_mx))
    client.embed_texts(["alpha"])
    assert fake_mx.cache_limits == [2 * 1024**3]


def test_load_applies_env_metal_cache_limit(monkeypatch):
    monkeypatch.setenv("MLX_EMBEDDING_CACHE_LIMIT_BYTES", "123456")
    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=loader)
    fake_mx = _FakeMxCore()
    monkeypatch.setattr(MLXEmbeddingClient, "_mlx_core", staticmethod(lambda: fake_mx))
    client.embed_texts(["alpha"])
    assert fake_mx.cache_limits == [123456]


def test_embed_releases_metal_cache_once_per_call(monkeypatch):
    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=loader)
    fake_mx = _FakeMxCore()
    monkeypatch.setattr(MLXEmbeddingClient, "_mlx_core", staticmethod(lambda: fake_mx))
    client.embed_texts(["alpha", "beta"])
    assert fake_mx.clear_calls == 1
    client.embed_texts(["gamma"])
    assert fake_mx.clear_calls == 2


def test_embed_releases_metal_cache_even_when_model_raises(monkeypatch):
    class _ExplodingModel:
        def __call__(self, tokenized):  # noqa: ARG002
            raise RuntimeError("boom")

    tokenizer = _FakeTokenizer()
    client = MLXEmbeddingClient(
        model="qwen3-embedding-0.6b",
        _loader=lambda name: (_ExplodingModel(), tokenizer),  # noqa: ARG005
    )
    fake_mx = _FakeMxCore()
    monkeypatch.setattr(MLXEmbeddingClient, "_mlx_core", staticmethod(lambda: fake_mx))
    with pytest.raises(RuntimeError, match="boom"):
        client.embed_texts(["alpha"])
    assert fake_mx.clear_calls == 1


def test_cache_management_tolerates_missing_mlx(monkeypatch):
    loader, _tokenizer, _model = _make_fake_loader()
    client = MLXEmbeddingClient(model="qwen3-embedding-0.6b", _loader=loader)
    monkeypatch.setattr(MLXEmbeddingClient, "_mlx_core", staticmethod(lambda: None))
    vectors = client.embed_texts(["alpha"])
    assert len(vectors) == 1


def test_cached_hub_model_loads_from_its_local_snapshot(tmp_path, monkeypatch) -> None:
    """A cached model id resolves to its snapshot path, so no online revision check runs."""
    import sys
    import types

    from conference_pipeline.indexing import mlx_embeddings

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    calls: list[dict] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("repo_id") == "org/cached":
            return str(snapshot)
        raise FileNotFoundError("not cached")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    assert mlx_embeddings._cached_snapshot_path("org/cached") == str(snapshot)
    assert all(call["local_files_only"] is True for call in calls)
    assert mlx_embeddings._cached_snapshot_path("org/missing") is None
    assert mlx_embeddings._cached_snapshot_path(str(snapshot)) is None
