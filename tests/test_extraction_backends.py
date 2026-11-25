from __future__ import annotations

from types import SimpleNamespace

import pytest

from conference_pipeline.extractions import backends
from conference_pipeline.extractions.backends import (
    BackendKind,
    BackendSelectionError,
    select_local_llm_backend,
)


def _set_http_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_MODEL", "qwen2.5-7b")
    monkeypatch.delenv("EXTRACTION_LOCAL_LLM_BACKEND", raising=False)
    monkeypatch.delenv("EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL", raising=False)


def test_select_returns_mlx_on_apple_silicon_with_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_http_env(monkeypatch)
    monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backends, "_try_import_mlx", lambda: True)
    monkeypatch.setattr(backends, "_mlx_has_model_locally", lambda model: True)

    backend = select_local_llm_backend()

    assert backend.kind is BackendKind.MLX
    assert backend.endpoint_label == "mlx://qwen2.5-7b"


def test_select_falls_back_to_http_when_mlx_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_http_env(monkeypatch)
    monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backends, "_try_import_mlx", lambda: True)
    monkeypatch.setattr(backends, "_mlx_has_model_locally", lambda model: False)

    backend = select_local_llm_backend()

    assert backend.kind is BackendKind.HTTP
    assert backend.endpoint_label == "http://localhost:11434/v1"


def test_select_falls_back_to_http_when_mlx_not_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_http_env(monkeypatch)
    monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backends, "_try_import_mlx", lambda: False)

    backend = select_local_llm_backend()

    assert backend.kind is BackendKind.HTTP
    assert backend.endpoint_label == "http://localhost:11434/v1"


def test_select_uses_http_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_http_env(monkeypatch)
    monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
    monkeypatch.setattr(backends.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(backends, "_try_import_mlx", lambda: True)
    monkeypatch.setattr(backends, "_mlx_has_model_locally", lambda model: True)

    backend = select_local_llm_backend()

    assert backend.kind is BackendKind.HTTP
    assert backend.endpoint_label == "http://localhost:11434/v1"


def test_force_mlx_on_non_mac_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_http_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_BACKEND", "mlx")
    # Fake only the backend's own platform check. Patching the global
    # ``platform`` module made torch (imported while the fake was active)
    # look for its Linux library on macOS and crash with dlopen.
    monkeypatch.setattr(backends, "_is_apple_silicon", lambda: False)

    with pytest.raises(BackendSelectionError, match="Apple Silicon"):
        select_local_llm_backend()


def test_force_mlx_without_importable_mlx_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_http_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_BACKEND", "mlx")
    monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backends, "_try_import_mlx", lambda: False)

    with pytest.raises(BackendSelectionError, match="not importable"):
        select_local_llm_backend()


def test_force_http_on_mac_with_mlx_uses_http(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_http_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_BACKEND", "http")
    monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backends, "_try_import_mlx", lambda: True)
    monkeypatch.setattr(backends, "_mlx_has_model_locally", lambda model: True)

    backend = select_local_llm_backend()

    assert backend.kind is BackendKind.HTTP
    assert backend.endpoint_label == "http://localhost:11434/v1"


def test_invalid_forced_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_http_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_BACKEND", "banana")

    with pytest.raises(BackendSelectionError, match="must be 'mlx' or 'http'"):
        select_local_llm_backend()


def test_endpoint_label_redacts_loopback_host_for_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_http_env(monkeypatch)
    monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
    monkeypatch.setattr(backends.platform, "machine", lambda: "x86_64")

    backend = select_local_llm_backend()

    assert backend.endpoint_label == "http://localhost:11434/v1"


def test_endpoint_label_for_mlx_uses_mlx_scheme_with_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_http_env(monkeypatch)
    monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backends, "_try_import_mlx", lambda: True)
    monkeypatch.setattr(backends, "_mlx_has_model_locally", lambda model: True)

    backend = select_local_llm_backend()

    assert backend.endpoint_label == "mlx://qwen2.5-7b"


def test_mlx_readiness_generates_and_chat_reuses_loaded_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from conference_pipeline.provider_receipts import provider_receipt_context

    load_calls: list[str] = []
    generate_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    model_obj = object()
    tokenizer = object()

    def fake_load(model: str) -> tuple[object, object]:
        load_calls.append(model)
        return model_obj, tokenizer

    def fake_generate(*args: object, **kwargs: object) -> SimpleNamespace:
        generate_calls.append((args, kwargs))
        return SimpleNamespace(
            id="mlx-response",
            text='{"topic_ids":[]}',
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )

    def fake_import_module(name: str) -> SimpleNamespace:
        if name == "mlx_lm.utils":
            return SimpleNamespace(load=fake_load, generate=fake_generate)
        if name == "mlx_lm":
            return SimpleNamespace(load=fake_load, generate=fake_generate)
        raise ImportError(name)

    monkeypatch.setattr(backends.importlib, "import_module", fake_import_module)
    backend = backends.MLXBackend(model="qwen-local", endpoint_label="mlx://qwen-local")

    backend.readiness()
    with provider_receipt_context(
        workspace_root=tmp_path,
        run_id="mlx-cache-test",
        stage_id="quality-sample",
    ):
        result = backend.chat_completion(
            messages=[{"role": "user", "content": "return json"}],
            max_tokens=5,
        )

    assert result.choices[0].message.content == '{"topic_ids":[]}'
    assert load_calls == ["qwen-local"]
    assert len(generate_calls) == 2
    assert generate_calls[0][1]["prompt"] == "USER: ping"
    assert generate_calls[0][1]["max_tokens"] == 1
    assert generate_calls[1][1]["max_tokens"] == 5


def test_mlx_chat_uses_template_without_thinking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from conference_pipeline.provider_receipts import provider_receipt_context

    generate_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    template_calls: list[dict[str, object]] = []
    model_obj = object()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            template_calls.append(dict(kwargs))
            return "CHAT_TEMPLATE_PROMPT"

    def fake_load(model: str) -> tuple[object, object]:
        return model_obj, Tokenizer()

    def fake_generate(*args: object, **kwargs: object) -> SimpleNamespace:
        generate_calls.append((args, kwargs))
        return SimpleNamespace(text='{"topic_ids":[]}')

    def fake_import_module(name: str) -> SimpleNamespace:
        if name in {"mlx_lm.utils", "mlx_lm"}:
            return SimpleNamespace(load=fake_load, generate=fake_generate)
        raise ImportError(name)

    monkeypatch.setattr(backends.importlib, "import_module", fake_import_module)
    backend = backends.MLXBackend(model="qwen-local", endpoint_label="mlx://qwen-local")

    with provider_receipt_context(
        workspace_root=tmp_path,
        run_id="mlx-template-test",
        stage_id="quality-sample",
    ):
        backend.chat_completion(
            messages=[{"role": "user", "content": "return json"}],
            max_tokens=5,
        )

    assert generate_calls[0][1]["prompt"] == "CHAT_TEMPLATE_PROMPT"
    assert template_calls[0]["enable_thinking"] is False


def test_live_lane_backend_failure_skips_when_not_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.live_llm import _helpers as live_helpers

    monkeypatch.delenv("RUN_LIVE_LLM", raising=False)

    with pytest.raises(pytest.skip.Exception):
        live_helpers.handle_unreachable_backend(RuntimeError("connection refused"))


def test_live_lane_backend_failure_fails_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.live_llm import _helpers as live_helpers

    monkeypatch.setenv("RUN_LIVE_LLM", "1")

    with pytest.raises(pytest.fail.Exception):
        live_helpers.handle_unreachable_backend(RuntimeError("connection refused"))


# ---------------------------------------------------------------------------
# MLXBackend Metal buffer-cache management (machine-OOM regression)
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


def _fake_mlx_backend(monkeypatch: pytest.MonkeyPatch, fake_mx: _FakeMxCore):
    tokenizer = SimpleNamespace()  # no apply_chat_template -> falls back to plain join
    model_obj = object()
    monkeypatch.setattr(backends, "_mlx_load_model", lambda model: (model_obj, tokenizer))  # noqa: ARG005
    monkeypatch.setattr(backends, "_mlx_generate", lambda *a, **k: "reply text")  # noqa: ARG005
    monkeypatch.setattr(backends, "_mlx_core_module", lambda: fake_mx)
    return backends.MLXBackend(model="qwen3.5-35b", endpoint_label="mlx://qwen3.5-35b")


def test_mlx_backend_applies_default_metal_cache_limit_on_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLX_EXTRACTION_CACHE_LIMIT_BYTES", raising=False)
    fake_mx = _FakeMxCore()
    backend = _fake_mlx_backend(monkeypatch, fake_mx)
    backend.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert fake_mx.cache_limits == [2 * 1024**3]


def test_mlx_backend_applies_env_metal_cache_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLX_EXTRACTION_CACHE_LIMIT_BYTES", "123456")
    fake_mx = _FakeMxCore()
    backend = _fake_mlx_backend(monkeypatch, fake_mx)
    backend.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert fake_mx.cache_limits == [123456]


def test_mlx_backend_releases_metal_cache_once_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mx = _FakeMxCore()
    backend = _fake_mlx_backend(monkeypatch, fake_mx)
    backend.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert fake_mx.clear_calls == 1
    backend.chat_completion(messages=[{"role": "user", "content": "again"}])
    assert fake_mx.clear_calls == 2


def test_mlx_backend_releases_metal_cache_even_when_generate_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace()
    model_obj = object()
    fake_mx = _FakeMxCore()
    monkeypatch.setattr(backends, "_mlx_load_model", lambda model: (model_obj, tokenizer))  # noqa: ARG005

    def _raise(*a, **k):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(backends, "_mlx_generate", _raise)
    monkeypatch.setattr(backends, "_mlx_core_module", lambda: fake_mx)
    backend = backends.MLXBackend(model="qwen3.5-35b", endpoint_label="mlx://qwen3.5-35b")
    with pytest.raises(RuntimeError, match="boom"):
        backend.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert fake_mx.clear_calls == 1


def test_mlx_backend_cache_management_tolerates_missing_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _fake_mlx_backend(monkeypatch, _FakeMxCore())
    monkeypatch.setattr(backends, "_mlx_core_module", lambda: None)
    result = backend.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert result.choices[0].message.content == "reply text"


# ---------------------------------------------------------------------------
# _mlx_has_model_locally against the real mlx_lm.utils API surface
# (regression: mlx_lm removed `get_model_path`; auto backend selection must
# still detect an already-cached model rather than silently falling through
# to an unconfigured HTTPBackend -- see backends.py's resolver fallback)
# ---------------------------------------------------------------------------


def test_mlx_has_model_locally_uses_hf_repo_to_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_hf_repo_to_path(model: str) -> str:
        calls.append(model)
        return "/fake/cache/path"

    fake_module = SimpleNamespace(hf_repo_to_path=fake_hf_repo_to_path)
    monkeypatch.setattr(
        backends.importlib,
        "import_module",
        lambda name: fake_module if name == "mlx_lm.utils" else (_ for _ in ()).throw(
            ImportError(name)
        ),
    )

    assert backends._mlx_has_model_locally("mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit") is True
    assert calls == ["mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"]


def test_mlx_has_model_locally_falls_back_to_get_model_path_for_older_mlx_lm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = SimpleNamespace(get_model_path=lambda model: "/fake/cache/path")  # noqa: ARG005
    monkeypatch.setattr(
        backends.importlib,
        "import_module",
        lambda name: fake_module if name == "mlx_lm.utils" else (_ for _ in ()).throw(
            ImportError(name)
        ),
    )

    assert backends._mlx_has_model_locally("some-model") is True


def test_mlx_has_model_locally_returns_false_when_neither_helper_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = SimpleNamespace()  # neither hf_repo_to_path nor get_model_path
    monkeypatch.setattr(
        backends.importlib,
        "import_module",
        lambda name: fake_module if name == "mlx_lm.utils" else (_ for _ in ()).throw(
            ImportError(name)
        ),
    )

    assert backends._mlx_has_model_locally("some-model") is False


def test_mlx_has_model_locally_returns_false_when_resolver_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raising_resolver(model: str) -> str:
        raise ValueError("not cached locally")

    fake_module = SimpleNamespace(hf_repo_to_path=raising_resolver)
    monkeypatch.setattr(
        backends.importlib,
        "import_module",
        lambda name: fake_module if name == "mlx_lm.utils" else (_ for _ in ()).throw(
            ImportError(name)
        ),
    )

    assert backends._mlx_has_model_locally("some-model") is False
# _mlx_has_model_locally (mlx_lm API drift regression)
# ---------------------------------------------------------------------------


def test_mlx_has_model_locally_true_when_hf_repo_to_path_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = SimpleNamespace(hf_repo_to_path=lambda repo: f"/cache/{repo}")  # noqa: ARG005
    monkeypatch.setattr(backends.importlib, "import_module", lambda name: fake_module)  # noqa: ARG005
    assert backends._mlx_has_model_locally("mlx-community/some-model") is True


def test_mlx_has_model_locally_false_when_hf_repo_to_path_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(repo):  # noqa: ARG001
        raise OSError("not cached")

    fake_module = SimpleNamespace(hf_repo_to_path=_raise)
    monkeypatch.setattr(backends.importlib, "import_module", lambda name: fake_module)  # noqa: ARG005
    assert backends._mlx_has_model_locally("mlx-community/some-model") is False


def test_mlx_has_model_locally_false_when_hf_repo_to_path_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: mlx_lm removed get_model_path; a future removal of
    # hf_repo_to_path itself must degrade to False, not raise.
    fake_module = SimpleNamespace()
    monkeypatch.setattr(backends.importlib, "import_module", lambda name: fake_module)  # noqa: ARG005
    assert backends._mlx_has_model_locally("mlx-community/some-model") is False


def test_http_backend_turns_thinking_off_unless_the_caller_sets_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same template behaviour as the in-process MLX backend (thinking off)."""
    import conference_pipeline.provider_receipts as receipts

    sent: list[dict] = []
    monkeypatch.setattr(receipts, "local_openai_chat_completions_create", lambda client, caller=None, **kw: sent.append(kw))
    backend = backends.HTTPBackend(base_url="http://127.0.0.1:9/v1", model="qwen-local", endpoint_label="http://localhost")
    backend.chat_completion(messages=[{"role": "user", "content": "hi"}])
    backend.chat_completion(messages=[{"role": "user", "content": "hi"}],
                            extra_body={"chat_template_kwargs": {"enable_thinking": True}, "top_k": 5})
    assert sent[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert sent[1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 5}
