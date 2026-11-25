"""Phase 44 — OllamaEmbeddingClient unit tests.

Tests inject a fake HTTP client and (when relevant) a fake spawn so the
suite never touches a real ``ollama serve`` process or non-loopback
network. Real Ollama exercise is deferred to v1.9 Phase 47's
``local_e2e`` opt-in lane (planned, not implemented this session).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from conference_pipeline._security.loopback import is_loopback_url
from conference_pipeline.indexing.embeddings import EmbeddingError
from conference_pipeline.indexing.ollama_embeddings import (
    OllamaEmbeddingClient,
    _redact_loopback,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_payload: dict[str, Any] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = json_payload or {}
        self._raise = raise_exc

    def raise_for_status(self) -> None:
        if self._raise is not None:
            raise self._raise

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttpClient:
    """Records get/post calls; configurable response."""

    def __init__(
        self,
        *,
        post_response: _FakeResponse | None = None,
    ) -> None:
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self.post_calls: list[tuple[str, dict[str, Any]]] = []
        self._post_response = post_response or _FakeResponse(
            json_payload={"embedding": [3.0, 4.0]}
        )

    def get(self, path: str, **kwargs: Any) -> _FakeResponse:
        self.get_calls.append((path, kwargs))
        return _FakeResponse(json_payload={"models": []})

    def post(self, path: str, **kwargs: Any) -> _FakeResponse:
        self.post_calls.append((path, kwargs))
        return self._post_response


class _RaisingHttpClient:
    """Always raises on .get; used to simulate unreachable endpoint."""

    def __init__(self) -> None:
        self.get_calls = 0

    def get(self, path: str, **kwargs: Any) -> _FakeResponse:  # noqa: ARG002
        self.get_calls += 1
        raise ConnectionError("simulated unreachable")

    def post(self, path: str, **kwargs: Any):  # pragma: no cover
        raise AssertionError("post should not be called when unreachable")


class _FakeProc:
    def __init__(self) -> None:
        self.terminated = False

    def poll(self) -> int | None:
        return None  # still running

    def terminate(self) -> None:  # pragma: no cover
        self.terminated = True

    def wait(self, timeout: float = 0) -> None:  # pragma: no cover
        return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_is_loopback_url_handles_common_hosts():
    assert is_loopback_url("http://127.0.0.1:11434") is True
    assert is_loopback_url("http://localhost:11434") is True
    assert is_loopback_url("http://[::1]:11434") is True
    assert is_loopback_url("http://api.openai.com:443") is False


def test_redact_loopback_rewrites_host():
    assert _redact_loopback("http://127.0.0.1:11434") == "http://localhost:11434"
    assert _redact_loopback("http://localhost:9999") == "http://localhost:9999"


# ---------------------------------------------------------------------------
# Constructor + loopback validation
# ---------------------------------------------------------------------------


def test_constructor_does_no_http_io():
    client = OllamaEmbeddingClient(model="nomic-embed-text")
    assert client._client is None
    assert client.model == "nomic-embed-text"
    assert client.embedding_dim == 768
    assert client.provider == "ollama"
    assert client.endpoint_label == "http://localhost:11434"


def test_non_loopback_url_raises_before_http():
    with pytest.raises(EmbeddingError, match="non_loopback_base_url"):
        OllamaEmbeddingClient(
            model="nomic-embed-text", base_url="http://api.openai.com:443"
        )


def test_default_base_url_is_loopback(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    client = OllamaEmbeddingClient(model="nomic-embed-text")
    assert client._base_url == "http://127.0.0.1:11434"


def test_env_base_url_overrides_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:33333")
    client = OllamaEmbeddingClient(model="nomic-embed-text")
    assert client._base_url == "http://localhost:33333"
    assert client.endpoint_label == "http://localhost:33333"


def test_endpoint_label_redacts_host():
    client = OllamaEmbeddingClient(
        model="nomic-embed-text", base_url="http://127.0.0.1:11434"
    )
    assert client.endpoint_label == "http://localhost:11434"


# ---------------------------------------------------------------------------
# embed_texts — happy path
# ---------------------------------------------------------------------------


def test_embed_texts_calls_api_embeddings_per_text():
    fake_http = _FakeHttpClient(
        post_response=_FakeResponse(json_payload={"embedding": [0.1, 0.2, 0.3]})
    )
    client = OllamaEmbeddingClient(
        model="nomic-embed-text",
        normalize=False,
        manage_server=False,
        _http=fake_http,
    )
    vectors = client.embed_texts(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(v == [0.1, 0.2, 0.3] for v in vectors)
    # Verify HTTP shape.
    assert len(fake_http.post_calls) == 3
    path, kwargs = fake_http.post_calls[0]
    assert path == "/api/embeddings"
    assert kwargs["json"]["model"] == "nomic-embed-text"
    assert kwargs["json"]["prompt"] == "a"


def test_embed_texts_normalizes_by_default():
    fake_http = _FakeHttpClient(
        post_response=_FakeResponse(json_payload={"embedding": [3.0, 4.0]})
    )
    client = OllamaEmbeddingClient(
        model="nomic-embed-text", manage_server=False, _http=fake_http
    )
    [vec] = client.embed_texts(["unit"])
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-9


def test_embed_texts_can_disable_normalization():
    fake_http = _FakeHttpClient(
        post_response=_FakeResponse(json_payload={"embedding": [3.0, 4.0]})
    )
    client = OllamaEmbeddingClient(
        model="nomic-embed-text",
        normalize=False,
        manage_server=False,
        _http=fake_http,
    )
    [vec] = client.embed_texts(["unit"])
    assert vec == [3.0, 4.0]


def test_embed_texts_empty_input_returns_empty_list():
    fake_http = _FakeHttpClient()
    client = OllamaEmbeddingClient(
        model="nomic-embed-text", manage_server=False, _http=fake_http
    )
    assert client.embed_texts([]) == []
    # No HTTP probe because lazy_load short-circuits on empty input.
    assert fake_http.get_calls == []


# ---------------------------------------------------------------------------
# manage_server lifecycle
# ---------------------------------------------------------------------------


def test_manage_server_false_fails_when_unreachable():
    fake_http = _RaisingHttpClient()
    client = OllamaEmbeddingClient(
        model="nomic-embed-text", manage_server=False, _http=fake_http
    )
    with pytest.raises(EmbeddingError, match="ollama_unreachable"):
        client.embed_texts(["hello"])
    assert fake_http.get_calls == 1


def test_manage_server_true_spawns_when_unreachable_then_succeeds():
    """Inject _spawn so the test never touches a real `ollama` binary."""
    spawn_calls: list[str] = []

    def fake_spawn(base_url: str) -> _FakeProc:
        spawn_calls.append(base_url)
        return _FakeProc()

    # First /api/tags raises; subsequent calls succeed (post-spawn).
    class _FlakyHttp:
        def __init__(self) -> None:
            self.get_calls = 0
            self.post_response = _FakeResponse(
                json_payload={"embedding": [1.0, 0.0]}
            )

        def get(self, path: str, **kwargs: Any) -> _FakeResponse:  # noqa: ARG002
            self.get_calls += 1
            if self.get_calls == 1:
                raise ConnectionError("first probe fails")
            return _FakeResponse(json_payload={"models": []})

        def post(self, path: str, **kwargs: Any) -> _FakeResponse:  # noqa: ARG002
            return self.post_response

    fake_http = _FlakyHttp()
    client = OllamaEmbeddingClient(
        model="nomic-embed-text",
        manage_server=True,
        startup_timeout_s=2.0,
        _http=fake_http,
        _spawn=fake_spawn,
    )
    [vec] = client.embed_texts(["hello"])
    assert spawn_calls == ["http://127.0.0.1:11434"]
    assert vec == [1.0, 0.0]
    assert fake_http.get_calls >= 2  # initial probe + at least one post-spawn


def test_manage_server_true_raises_if_spawn_does_not_become_ready():
    """Spawn returns but endpoint never reachable → raise after timeout."""
    fake_http = _RaisingHttpClient()

    def fake_spawn(base_url: str) -> _FakeProc:  # noqa: ARG001
        return _FakeProc()

    client = OllamaEmbeddingClient(
        model="nomic-embed-text",
        manage_server=True,
        startup_timeout_s=0.5,  # short timeout for test
        _http=fake_http,
        _spawn=fake_spawn,
    )
    with pytest.raises(EmbeddingError, match="ollama_unreachable"):
        client.embed_texts(["hello"])


# ---------------------------------------------------------------------------
# Telemetry + shim parity
# ---------------------------------------------------------------------------


def test_usage_shape_parity():
    fake_http = _FakeHttpClient(
        post_response=_FakeResponse(json_payload={"embedding": [3.0, 4.0]})
    )
    client = OllamaEmbeddingClient(
        model="nomic-embed-text", manage_server=False, _http=fake_http
    )
    client.embed_texts(["one two", "three"])
    usage = client.usage()
    assert set(usage.keys()) == {
        "provider",
        "model",
        "input_tokens",
        "request_count",
        "estimated_cost_usd",
        "provider_receipt_paths",
    }
    assert usage["provider"] == "ollama"
    assert usage["model"] == "nomic-embed-text"
    assert usage["estimated_cost_usd"] == 0.0
    assert usage["provider_receipt_paths"] == []
    assert usage["request_count"] == 2
    assert usage["input_tokens"] >= 3  # "one two" + "three" = 3 tokens


def test_legacy_embed_shim_works_on_ollama_client():
    fake_http = _FakeHttpClient(
        post_response=_FakeResponse(json_payload={"embedding": [1.0, 0.0]})
    )
    client = OllamaEmbeddingClient(
        model="nomic-embed-text", manage_server=False, _http=fake_http
    )
    with pytest.warns(DeprecationWarning, match=r"embed_texts"):
        result = client.embed(["hello"])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# v1.9 Phase 44 review fixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bypass_url",
    [
        "http://0.0.0.0:11434",            # binds all interfaces!
        "http://0:11434",                   # not a loopback per ipaddress
        "http://[::]:11434",                # IPv6 unspecified (binds all)
        "http://api.openai.com:443",        # public host
        "http://1.2.3.4:11434",             # arbitrary IP
    ],
)
def test_is_loopback_url_rejects_bypass_vectors(bypass_url):
    """v1.9 Phase 44 review HI-01: previously the helper used a string set
    that silently accepted 0.0.0.0, 0, and 127.1 — all bypass the
    loopback-only contract. ipaddress-based check fixes this.

    Note: ``[::ffff:127.0.0.1]`` (IPv4-mapped IPv6 loopback) IS classified
    as loopback by ``ipaddress.is_loopback`` and is semantically equivalent
    to ``127.0.0.1``; that case is in the *valid* set, not the bypass set.
    """
    assert is_loopback_url(bypass_url) is False, (
        f"loopback bypass: {bypass_url!r} should be rejected"
    )


@pytest.mark.parametrize(
    "valid_url",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://127.1.2.3:11434",            # any 127.0.0.0/8 → loopback
        "http://127.255.255.254:8080",       # high end of loopback block
        "http://[::1]:11434",                 # IPv6 loopback
        "http://[::ffff:127.0.0.1]:11434",    # IPv4-mapped IPv6 loopback
    ],
)
def test_is_loopback_url_accepts_valid_loopback(valid_url):
    """All addresses in 127.0.0.0/8 plus IPv6 ::1 plus 'localhost' must
    pass. The strict ``host in {127.0.0.1,localhost,::1}`` check rejected
    127.x.y.z that the kernel resolves to loopback; the new
    ipaddress-based check fixes this.
    """
    assert is_loopback_url(valid_url) is True, (
        f"valid loopback rejected: {valid_url!r}"
    )


@pytest.mark.parametrize(
    "bypass_url",
    [
        "http://0.0.0.0:11434",
        "http://[::]:11434",
        "http://1.2.3.4:11434",
    ],
)
def test_constructor_rejects_bypass_loopback_vectors(bypass_url):
    """v1.9 Phase 44 review HI-01: the bypass vectors must trigger the
    constructor's pre-HTTP loopback validation."""
    with pytest.raises(EmbeddingError, match="non_loopback_base_url"):
        OllamaEmbeddingClient(
            model="nomic-embed-text", base_url=bypass_url, manage_server=False
        )


def test_dim_mismatch_does_not_silently_mutate_declared_dim(caplog):
    """v1.9 Phase 44 review ME-05: declared embedding_dim is the contract.
    Mirrors MLX-4 fix in mlx_embeddings.py.
    """
    import logging

    fake_http = _FakeHttpClient(
        post_response=_FakeResponse(json_payload={"embedding": [1.0, 0.0, 0.0]})
    )
    client = OllamaEmbeddingClient(
        model="nomic-embed-text",
        embedding_dim=768,  # declared
        manage_server=False,
        _http=fake_http,
    )
    with caplog.at_level(logging.WARNING):
        client.embed_texts(["a"])
    # Declared dim is the contract — must NOT mutate.
    assert client.embedding_dim == 768
    # Actual dim is tracked out-of-band.
    assert client._actual_dim == 3
    mismatch_records = [r for r in caplog.records if "dim_mismatch" in r.message]
    assert len(mismatch_records) == 1
    # Second call: warning is not repeated.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        client.embed_texts(["b"])
    assert not [r for r in caplog.records if "dim_mismatch" in r.message]


def test_spawn_crash_detected_immediately_via_poll():
    """v1.9 Phase 44 review HI-04: a crashed `ollama serve` (port-in-use,
    missing model store, etc.) must be detected via proc.poll() and
    surfaced as 'ollama_serve_crashed' immediately, not waited out for
    the full startup_timeout_s with a misleading 'unreachable' error.
    """

    class _CrashedProc:
        returncode = 137  # SIGKILL-style exit

        def poll(self):
            return 137  # already exited

    def fake_spawn_returns_crashed(base_url):  # noqa: ARG001
        return _CrashedProc()

    fake_http = _RaisingHttpClient()
    client = OllamaEmbeddingClient(
        model="nomic-embed-text",
        manage_server=True,
        startup_timeout_s=10.0,  # would be a long wait if poll detection broken
        _http=fake_http,
        _spawn=fake_spawn_returns_crashed,
    )
    started = time.monotonic()
    with pytest.raises(EmbeddingError, match="ollama_serve_crashed"):
        client.embed_texts(["hello"])
    elapsed = time.monotonic() - started
    # Should fail fast (well under startup_timeout_s).
    assert elapsed < 2.0, (
        f"crash detection took {elapsed:.1f}s — should be near-instant"
    )


def test_embed_texts_writes_local_embedding_receipt_when_workspace_provided(
    tmp_path,
):
    """Phase 45 Plan 04: per-call receipt closes BL-01 audit-trail gap."""
    import json

    fake_http = _FakeHttpClient(
        post_response=_FakeResponse(json_payload={"embedding": [3.0, 4.0]})
    )
    client = OllamaEmbeddingClient(
        model="nomic-embed-text",
        manage_server=False,
        _http=fake_http,
        workspace=tmp_path,
        run_id="run-xyz",
        stage_id="vector-store-setup",
    )
    client.embed_texts(["alpha", "beta"])
    receipt_dir = (
        tmp_path / "storage" / "workflow-runs" / "run-xyz"
        / "local-embedding-receipts" / "vector-store-setup"
    )
    receipts = list(receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["provider"] == "ollama"
    assert payload["model"] == "nomic-embed-text"
    assert payload["input_count"] == 2
    assert payload["cost_usd"] == 0.0
    assert payload["endpoint_label"] == "http://localhost:11434"


def test_embed_texts_no_receipt_when_workspace_omitted():
    """No-op when workspace=None."""
    fake_http = _FakeHttpClient(
        post_response=_FakeResponse(json_payload={"embedding": [1.0, 0.0]})
    )
    client = OllamaEmbeddingClient(
        model="nomic-embed-text", manage_server=False, _http=fake_http,
    )
    vectors = client.embed_texts(["hello"])
    assert len(vectors) == 1


def test_spawn_log_path_referenced_in_error_message():
    """v1.9 Phase 44 review HI-03: error messages must point at the
    .planning/logs/ollama-serve.log file so operators can diagnose."""

    class _StaysRunning:
        def poll(self):
            return None

    def fake_spawn(base_url):  # noqa: ARG001
        return _StaysRunning()

    fake_http = _RaisingHttpClient()
    client = OllamaEmbeddingClient(
        model="nomic-embed-text",
        manage_server=True,
        startup_timeout_s=0.5,
        _http=fake_http,
        _spawn=fake_spawn,
    )
    with pytest.raises(EmbeddingError) as exc_info:
        client.embed_texts(["hello"])
    msg = str(exc_info.value)
    assert ".planning/logs/ollama-serve.log" in msg


# Need time at module level for test_spawn_crash_detected_immediately_via_poll
import time  # noqa: E402
