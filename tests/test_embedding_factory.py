"""Phase 42 Plan 02 — `build_embedding_client` factory + adaptive batch sizing.

Covers v1.9 requirements LOCAL-EMBED-02 and LOCAL-EMBED-04. The factory
dispatches on ``config['provider']`` and gates `mlx` / `ollama` branches
with `NotImplementedError` whose message points at Phases 43 / 44 (which
will flip them to real returns when those phases land).
"""

from __future__ import annotations

import pytest

from conference_pipeline.indexing.embeddings import (
    FakeEmbeddingClient,
    OpenAIEmbeddingClient,
    Provider,
    build_embedding_client,
    effective_batch_size,
)


# ---------------------------------------------------------------------------
# build_embedding_client — dispatch matrix
# ---------------------------------------------------------------------------


def test_build_returns_openai_client_when_api_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = build_embedding_client({"provider": "openai"})
    assert isinstance(client, OpenAIEmbeddingClient)
    assert client.provider == "openai"


def test_build_openai_propagates_model_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = build_embedding_client(
        {"provider": "openai", "model": "text-embedding-3-large"}
    )
    assert client.model == "text-embedding-3-large"


def test_build_returns_fake_client():
    client = build_embedding_client({"provider": "fake"})
    assert isinstance(client, FakeEmbeddingClient)
    assert client.provider == "fake"
    assert client.embedding_dim == 1536


def test_build_fake_with_custom_dim():
    client = build_embedding_client({"provider": "fake", "embedding_dim": 64})
    assert client.embedding_dim == 64
    vectors = client.embed_texts(["hello"])
    assert len(vectors[0]) == 64


def test_build_mlx_branch_returns_mlx_client():
    """v1.9 Phase 43 flipped this branch from NotImplementedError to a real return."""
    from conference_pipeline.indexing.mlx_embeddings import MLXEmbeddingClient

    # Inject a no-op loader so construction does not require mlx-embeddings.
    def _noop_loader(model_name):  # noqa: ARG001
        return object(), object()

    client = build_embedding_client({"provider": "mlx", "_loader": _noop_loader})
    assert isinstance(client, MLXEmbeddingClient)
    assert client.model == "mlx-community/bge-m3-mlx-8bit"
    assert client.embedding_dim == 1024
    assert client.provider == "mlx"


def test_build_mlx_branch_propagates_model_and_dim():
    from conference_pipeline.indexing.mlx_embeddings import MLXEmbeddingClient

    def _noop_loader(model_name):  # noqa: ARG001
        return object(), object()

    client = build_embedding_client(
        {
            "provider": "mlx",
            "model": "BAAI/bge-large-en-v1.5",
            "embedding_dim": 1024,
            "normalize": False,
            "_loader": _noop_loader,
        }
    )
    assert isinstance(client, MLXEmbeddingClient)
    assert client.model == "BAAI/bge-large-en-v1.5"
    assert client.embedding_dim == 1024
    assert client._normalize is False


def test_build_mlx_branch_propagates_max_length():
    from conference_pipeline.indexing.mlx_embeddings import MLXEmbeddingClient

    def _noop_loader(model_name):  # noqa: ARG001
        return object(), object()

    client = build_embedding_client(
        {"provider": "mlx", "max_length": 1024, "_loader": _noop_loader}
    )

    assert isinstance(client, MLXEmbeddingClient)
    assert client.max_length == 1024


def test_build_ollama_branch_returns_ollama_client():
    """v1.9 Phase 44 flipped this branch from NotImplementedError to a real return."""
    from conference_pipeline.indexing.ollama_embeddings import (
        OllamaEmbeddingClient,
    )

    client = build_embedding_client(
        {"provider": "ollama", "manage_server": False}
    )
    assert isinstance(client, OllamaEmbeddingClient)
    assert client.model == "nomic-embed-text"
    assert client.embedding_dim == 768
    assert client.provider == "ollama"
    # endpoint_label always redacted to localhost.
    assert client.endpoint_label == "http://localhost:11434"


def test_build_ollama_branch_propagates_base_url_and_model(monkeypatch):
    from conference_pipeline.indexing.ollama_embeddings import (
        OllamaEmbeddingClient,
    )

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    client = build_embedding_client(
        {
            "provider": "ollama",
            "model": "mxbai-embed-large",
            "embedding_dim": 1024,
            "base_url": "http://localhost:33333",
            "manage_server": False,
            "normalize": False,
        }
    )
    assert isinstance(client, OllamaEmbeddingClient)
    assert client.model == "mxbai-embed-large"
    assert client.embedding_dim == 1024
    assert client._base_url == "http://localhost:33333"
    assert client._normalize is False
    assert client._manage_server is False


def test_build_unknown_provider_raises_value_error_with_valid_set():
    with pytest.raises(ValueError) as exc_info:
        build_embedding_client({"provider": "cohere"})
    msg = str(exc_info.value)
    assert "cohere" in msg
    assert "'openai', 'mlx', 'ollama', 'fake'" in msg


def test_build_missing_provider_field_raises_value_error():
    with pytest.raises(ValueError, match=r"requires config\['provider'\]"):
        build_embedding_client({})


def test_build_non_dict_config_raises_value_error():
    with pytest.raises(ValueError, match=r"requires a dict"):
        build_embedding_client("provider=fake")  # type: ignore[arg-type]


def test_provider_literal_alias_lists_canonical_set():
    # Provider is a Literal[...]; the runtime str values are the canonical set.
    # We check the alias is importable and consistent with the factory's guard.
    from typing import get_args
    assert set(get_args(Provider)) == {"openai", "mlx", "ollama", "fake"}


# ---------------------------------------------------------------------------
# effective_batch_size — heuristic + override precedence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("meta-llama-8b-instruct", 2),
        ("Qwen/Qwen2.5-13B", 2),
        ("mlx-community/llama-70b", 2),
    ],
)
def test_effective_batch_size_8b_or_larger_clamps_to_2(model, expected):
    assert effective_batch_size("mlx", model) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("Qwen/Qwen2.5-3B", 8),
        ("mistral-7b-instruct", 8),
        ("phi-3.5b", 8),
    ],
)
def test_effective_batch_size_3b_to_under_8b_clamps_to_8(model, expected):
    assert effective_batch_size("mlx", model) == expected


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("mlx", 96),
        ("ollama", 32),
        ("openai", 128),
        ("fake", 64),
    ],
)
def test_effective_batch_size_default_per_provider_for_small_model(
    provider, expected
):
    # Small / unknown model name → no billions match → provider default.
    assert effective_batch_size(provider, "Qwen/Qwen3-Embedding-0.6B") == expected


def test_effective_batch_size_explicit_override_wins():
    assert effective_batch_size("mlx", "meta-llama-8b", override=4) == 4


def test_effective_batch_size_env_override_wins_over_heuristic(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "16")
    assert effective_batch_size("mlx", "meta-llama-8b") == 16


def test_effective_batch_size_explicit_override_beats_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "16")
    assert effective_batch_size("mlx", "meta-llama-8b", override=99) == 99


def test_effective_batch_size_ignores_non_positive_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    # Falls through to heuristic / per-provider default.
    assert effective_batch_size("mlx", "Qwen3-Embedding-0.6B") == 96


# ---------------------------------------------------------------------------
# v1.9 Phase 42 review HIGH-02 — config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [-1, 0, "8", 1.5, True])
def test_build_rejects_non_positive_int_batch_size(monkeypatch, bad_value):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    with pytest.raises(ValueError, match="batch_size.*must be a positive int"):
        build_embedding_client(
            {"provider": "openai", "batch_size": bad_value}
        )


@pytest.mark.parametrize("bad_value", [-1, 0, "1024", 1.0, True])
def test_build_rejects_non_positive_int_embedding_dim(bad_value):
    with pytest.raises(ValueError, match="embedding_dim.*must be a positive int"):
        build_embedding_client(
            {"provider": "fake", "embedding_dim": bad_value}
        )


@pytest.mark.parametrize("bad_value", [-1, 0, "2048", 1.5, True])
def test_build_rejects_non_positive_int_max_length(bad_value):
    with pytest.raises(ValueError, match="max_length.*must be a positive int"):
        build_embedding_client({"provider": "mlx", "max_length": bad_value})


def test_build_accepts_explicit_positive_batch_size(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = build_embedding_client(
        {"provider": "openai", "batch_size": 32}
    )
    assert client._max_batch == 32  # noqa: SLF001 - invariant assertion


# ---------------------------------------------------------------------------
# v1.9 Phase 42 review HIGH-03 — missing lazy-load + stacklevel coverage
# ---------------------------------------------------------------------------


def test_openai_constructor_does_no_network_io_with_bogus_model(monkeypatch):
    """LOCAL-EMBED-03 explicit test: bogus model name does not trigger any
    network I/O at construction. The OpenAI HTTP client must be lazy.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = OpenAIEmbeddingClient(model="definitely-not-a-real-model-zzz")
    # _ensure_client has not been called — _client is None until first
    # embed_texts() invocation.
    assert client._client is None  # noqa: SLF001 - invariant assertion
    assert client.model == "definitely-not-a-real-model-zzz"


def test_fake_constructor_does_no_io():
    """LOCAL-EMBED-03 — FakeEmbeddingClient must also be lazy at construction."""
    client = FakeEmbeddingClient(model="bogus-model")
    # FakeEmbeddingClient has no network surface; the test asserts the
    # invariant that no exception is raised regardless of model name.
    assert client.model == "bogus-model"
    assert client.embedding_dim == 1536


def test_legacy_embed_shim_warning_points_at_caller_file():
    """v1.9 Phase 42 review HIGH-03(b): verify stacklevel=2 means the warning
    filename is THIS test file (the caller), not embeddings.py (the shim).
    A regression to stacklevel=1 would point at embeddings.py and this test
    would catch it.
    """
    import warnings as _warnings

    client = FakeEmbeddingClient(embedding_dim=8)
    with _warnings.catch_warnings(record=True) as record:
        _warnings.simplefilter("always")
        client.embed(["hello"])
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    # stacklevel=2 means the warning's filename is THIS test file's path,
    # not the embeddings.py shim file.
    assert deprecations[0].filename.endswith("test_embedding_factory.py"), (
        f"DeprecationWarning fired from {deprecations[0].filename}; "
        "expected it to point at the caller (this test file) — likely "
        "stacklevel regressed to 1."
    )


# ---------------------------------------------------------------------------
# LOCAL-03: embedding_resolver — D-09 single source of truth + D-10 paid opt-in
# ---------------------------------------------------------------------------


def test_embedding_resolver_defaults_to_mlx(monkeypatch):
    """D-09: with no env vars, resolve_embedding_config() returns mlx defaults."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MAX_LENGTH", raising=False)
    monkeypatch.delenv("CONFERENCE_ALLOW_PAID_EMBEDDINGS", raising=False)

    from conference_pipeline.indexing.embedding_resolver import resolve_embedding_config

    config = resolve_embedding_config()
    assert config["provider"] == "mlx"
    assert config["model"] == "mlx-community/bge-m3-mlx-8bit"
    assert config["embedding_dim"] == 1024
    assert config["max_length"] == 2048


def test_embedding_resolver_requires_paid_opt_in_for_openai(monkeypatch):
    """D-10: EMBEDDING_PROVIDER=openai without opt-in raises PaidEmbeddingApprovalRequired."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("CONFERENCE_ALLOW_PAID_EMBEDDINGS", raising=False)

    from conference_pipeline.indexing.embedding_resolver import (
        PaidEmbeddingApprovalRequired,
        resolve_embedding_config,
    )

    with pytest.raises(PaidEmbeddingApprovalRequired, match="paid opt-in"):
        resolve_embedding_config(allow_paid=False)


def test_embedding_resolver_openai_with_allow_paid_flag(monkeypatch):
    """D-10: allow_paid=True bypasses the opt-in gate."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("CONFERENCE_ALLOW_PAID_EMBEDDINGS", raising=False)

    from conference_pipeline.indexing.embedding_resolver import resolve_embedding_config

    config = resolve_embedding_config(allow_paid=True)
    assert config["provider"] == "openai"


def test_embedding_resolver_openai_with_env_opt_in(monkeypatch):
    """D-10: CONFERENCE_ALLOW_PAID_EMBEDDINGS=1 is a valid opt-in signal."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("CONFERENCE_ALLOW_PAID_EMBEDDINGS", "1")

    from conference_pipeline.indexing.embedding_resolver import resolve_embedding_config

    config = resolve_embedding_config(allow_paid=False)
    assert config["provider"] == "openai"


def test_embedding_resolver_ollama_defaults(monkeypatch):
    """D-09: ollama provider gets correct model/dim defaults."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)

    from conference_pipeline.indexing.embedding_resolver import resolve_embedding_config

    config = resolve_embedding_config()
    assert config["provider"] == "ollama"
    assert config["model"] == "nomic-embed-text"
    assert config["embedding_dim"] == 768


def test_embedding_resolver_fake_defaults(monkeypatch):
    """D-09: fake provider gets 1536-dim default."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")

    from conference_pipeline.indexing.embedding_resolver import resolve_embedding_config

    config = resolve_embedding_config()
    assert config["provider"] == "fake"
    assert config["embedding_dim"] == 1536


def test_build_resolved_embedding_client_mlx(monkeypatch):
    """D-12: build_resolved_embedding_client returns (config, client) for mlx."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("CONFERENCE_ALLOW_PAID_EMBEDDINGS", raising=False)

    from conference_pipeline.indexing.embedding_resolver import build_resolved_embedding_client

    def _fake_build(config):
        return FakeEmbeddingClient()

    config, client = build_resolved_embedding_client(build_embedding_client_fn=_fake_build)
    assert config["provider"] == "mlx"
    assert isinstance(client, FakeEmbeddingClient)


def test_build_resolved_embedding_client_openai_requires_opt_in(monkeypatch):
    """D-10: build_resolved_embedding_client blocks openai without opt-in."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("CONFERENCE_ALLOW_PAID_EMBEDDINGS", raising=False)

    from conference_pipeline.indexing.embedding_resolver import (
        PaidEmbeddingApprovalRequired,
        build_resolved_embedding_client,
    )

    with pytest.raises(PaidEmbeddingApprovalRequired):
        build_resolved_embedding_client(allow_paid=False)


def test_local_embedding_client_config_from_env_still_works(monkeypatch):
    """D-09 backward-compat: corpus_vector_store delegate still returns same mlx config."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MAX_LENGTH", raising=False)
    monkeypatch.delenv("CONFERENCE_ALLOW_PAID_EMBEDDINGS", raising=False)

    from conference_pipeline.indexing.corpus_vector_store import (
        local_embedding_client_config_from_env,
    )

    config = local_embedding_client_config_from_env()
    assert config["provider"] == "mlx"
    assert config["model"] == "mlx-community/bge-m3-mlx-8bit"
    assert config["embedding_dim"] == 1024


# ---------------------------------------------------------------------------
# v3.0 03-04: local embedding receipt with cost-avoided fields (LOCAL-04 / D-14)
# ---------------------------------------------------------------------------


def test_local_embedding_receipt_gets_cost_avoided_fields(tmp_path: Path) -> None:
    """LOCAL-04 / D-14: local embedding receipts can be produced with cost-avoided
    fields using an injected fake estimator.  No live provider calls are made."""
    from conference_pipeline.indexing.embedding_resolver import (
        estimate_local_embedding_cost_avoided,
    )
    from conference_pipeline.provider_receipts import (
        record_local_embedding_call,
        scan_local_embedding_receipts,
    )

    # Fake estimator: bypasses catalog lookup, returns a priced result.
    def fake_cost_estimate(
        task_id: str,
        *,
        paid_provider: str,
        paid_model: str,
        quantity: float | None = None,
        catalog=None,
        task_specs=None,
    ) -> dict:
        return {
            "cost_avoided_usd": 0.0035,
            "cost_avoided_status": "priced",
            "paid_counterfactual_provider": paid_provider,
            "paid_counterfactual_model": paid_model,
            "needs_pricing_backfill": False,
        }

    # Use estimate_local_embedding_cost_avoided with injected fake to prove
    # the wiring works without live provider calls (D-04 dry-run-first).
    ca_kwargs = estimate_local_embedding_cost_avoided(
        "vector_store.indexing",
        input_count=10,
        total_input_chars=4000,
        benchmark_receipt_ref="storage/benchmarks/embed-bge-m3-20260625.json",
        estimate_cost_avoided_fn=fake_cost_estimate,
    )

    embed_path = record_local_embedding_call(
        provider="mlx",
        model="mlx-community/bge-m3-mlx-8bit",
        embedding_dim=1024,
        endpoint_label="mlx://mlx-community/bge-m3-mlx-8bit",
        input_count=10,
        total_input_chars=4000,
        request_count=1,
        cost_usd=0.0,
        workspace=tmp_path,
        **ca_kwargs,
    )

    assert embed_path is not None and embed_path.exists(), (
        "record_local_embedding_call must write a receipt when workspace is provided"
    )

    # Verify cost-avoided fields are stamped correctly.
    import json
    payload = json.loads(embed_path.read_text())
    assert payload["benchmark_receipt_ref"] == "storage/benchmarks/embed-bge-m3-20260625.json"
    assert payload["cost_avoided_usd"] == pytest.approx(0.0035)
    assert payload["cost_avoided_status"] == "priced"
    assert payload["paid_counterfactual_provider"] == "openai"
    assert payload["paid_counterfactual_model"] == "text-embedding-3-small"
    assert payload["needs_pricing_backfill"] is False

    # Verify the receipt is discoverable via the scanner.
    receipts = scan_local_embedding_receipts(tmp_path)
    assert len(receipts) >= 1
    found = [r for r in receipts if r.get("cost_avoided_usd") is not None]
    assert found, "scanned receipts must include the one with cost_avoided_usd stamped"

    # D-16 invariant: cost_avoided_usd must never be $0 when status is 'priced'
    # (in this test it is 0.0035, confirming it was from the fake estimator, not coerced to 0).
    assert found[0]["cost_avoided_usd"] != 0.0, (
        "cost_avoided_usd must reflect the actual avoided estimate, not $0"
    )
