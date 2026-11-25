"""Backend selection and adapters for local extraction LLMs."""

from __future__ import annotations

import importlib
import os
import platform
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from conference_pipeline.provider_receipts import record_provider_call


_DEFAULT_LOCAL_LLM_MODEL = "local-extraction-classifier"
_DEFAULT_LOCAL_LLM_ENDPOINT_LABEL = "local-openai-compatible"

# See conference_pipeline.indexing.mlx_embeddings for the full rationale: MLX's
# unified-memory buffer cache retains an allocation for every distinct tensor
# shape it evaluates, and a full-corpus extraction run makes tens of thousands
# of generate() calls over near-unique prompt lengths, so the cache grows
# without bound by default -- this is the exact mechanism that took the
# machine down three times during today's embedding run before that fix
# landed. mlx_lm.generate() manages its own per-call KV cache, but not this
# global buffer cache, so MLXBackend needs the identical bound-and-release
# treatment. Override with MLX_EXTRACTION_CACHE_LIMIT_BYTES (0 disables
# caching).
_DEFAULT_EXTRACTION_CACHE_LIMIT_BYTES = 2 * 1024**3
_EXTRACTION_CACHE_LIMIT_ENV = "MLX_EXTRACTION_CACHE_LIMIT_BYTES"


class BackendKind(str, Enum):
    MLX = "mlx"
    HTTP = "http"


class BackendSelectionError(RuntimeError):
    """Raised when a forced local LLM backend is not valid for this host."""


class Backend(Protocol):
    kind: BackendKind
    endpoint_label: str
    model: str

    def readiness(self) -> None:
        """Raise when the selected backend is not ready."""

    def chat_completion(self, *, caller: str | None = None, **kwargs: Any) -> Any:
        """Run one chat completion through the selected backend."""


def _requested_model(explicit_model: str | None = None) -> str:
    return (
        explicit_model
        or os.environ.get("EXTRACTION_LOCAL_LLM_MODEL")
        or os.environ.get("EXTRACTION_LLM_MODEL")
        or _DEFAULT_LOCAL_LLM_MODEL
    ).strip() or _DEFAULT_LOCAL_LLM_MODEL


def _custom_endpoint_label() -> str:
    return (
        os.environ.get("EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL")
        or os.environ.get("EXTRACTION_LLM_ENDPOINT_LABEL")
        or ""
    ).strip()


def _local_llm_base_url() -> str:
    return (
        os.environ.get("EXTRACTION_LOCAL_LLM_BASE_URL")
        or os.environ.get("EXTRACTION_LLM_BASE_URL")
        or ""
    ).strip()


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _try_import_mlx() -> bool:
    try:
        importlib.import_module("mlx_lm")
    except ImportError:
        return False
    return True


def _mlx_has_model_locally(model: str) -> bool:
    """Return True iff ``model`` is already present in the local HF cache.

    ``mlx_lm.utils.get_model_path`` was removed from the installed mlx_lm
    (0.31.x); calling it always raised AttributeError, which the original
    broad ``except Exception`` swallowed, so this check silently returned
    False for every model regardless of what was actually cached -- routing
    every backend selection to HTTPBackend even on a machine with a fully
    cached model and no local HTTP server running (every call then failed
    with a fast "Connection error"). ``hf_repo_to_path`` is the current
    mlx_lm helper for this: it wraps ``snapshot_download(..., local_files_
    only=True)``, which raises rather than downloading when the model is
    not cached, and returns the local path when it is. The ``get_model_path``
    fallback keeps older installed pins working.
    """
    try:
        module = importlib.import_module("mlx_lm.utils")
        resolver = getattr(module, "hf_repo_to_path", None) or getattr(module, "get_model_path", None)
        if not callable(resolver):
            return False
        path = resolver(model)
    except Exception:
        return False
    return bool(path)


def _redact_loopback(base_url: str) -> str:
    try:
        parts = urlsplit(base_url)
        port_value = parts.port
    except ValueError:
        return base_url
    host = (parts.hostname or "").lower()
    if host not in {"localhost", "::1"} and not host.startswith("127."):
        return base_url
    port = f":{port_value}" if port_value else ""
    return urlunsplit((parts.scheme, f"localhost{port}", parts.path, "", ""))


def _build_http_backend(model: str) -> "HTTPBackend":
    base_url = _local_llm_base_url()
    endpoint_label = _custom_endpoint_label() or _redact_loopback(base_url)
    if not endpoint_label:
        endpoint_label = _DEFAULT_LOCAL_LLM_ENDPOINT_LABEL
    return HTTPBackend(
        base_url=base_url,
        model=model,
        endpoint_label=endpoint_label,
    )


@dataclass(frozen=True)
class HTTPBackend:
    base_url: str
    model: str
    endpoint_label: str
    kind: BackendKind = BackendKind.HTTP

    def readiness(self) -> None:
        import httpx

        from conference_pipeline.extractions.classifier import (
            LocalLLMNotReadyError,
            _is_loopback_base_url,
            _redact,
            _redacted_base_url_identity,
        )

        if not self.base_url or not _is_loopback_base_url(self.base_url):
            raise LocalLLMNotReadyError(
                check_failed="reachability",
                endpoint_label=self.endpoint_label,
                message=(
                    "EXTRACTION_LOCAL_LLM_BASE_URL must be a loopback address "
                    "(localhost, 127.x.x.x, or ::1)."
                ),
                fallback_options=[
                    "Set EXTRACTION_LOCAL_LLM_BASE_URL=http://localhost:11434/v1",
                    "Unset EXTRACTION_LOCAL_LLM_ASSIST to use heuristic-only mode",
                ],
            )

        models_url = self.base_url.rstrip("/") + "/models"
        try:
            response = httpx.get(models_url, timeout=10.0)
            response.raise_for_status()
            models_data = response.json()
            listed_ids = []
            if isinstance(models_data, dict):
                entries = models_data.get("models") or models_data.get("data") or []
                if isinstance(entries, list):
                    listed_ids = [
                        str(entry.get("id") or "")
                        for entry in entries
                        if isinstance(entry, dict)
                    ]
            if self.model and self.model not in listed_ids:
                raise LocalLLMNotReadyError(
                    check_failed="model_not_listed",
                    endpoint_label=self.endpoint_label,
                    message=(
                        f"Model '{self.model}' not found in "
                        f"{_redacted_base_url_identity(self.base_url)}/models "
                        f"response. Available: {listed_ids[:5]}"
                    ),
                    fallback_options=[
                        f"Pull the model: ollama pull {self.model}",
                        "Set EXTRACTION_LOCAL_LLM_MODEL to an available model",
                    ],
                )
        except LocalLLMNotReadyError:
            raise
        except Exception as exc:
            raise LocalLLMNotReadyError(
                check_failed="reachability",
                endpoint_label=self.endpoint_label,
                message=_redact(
                    f"Cannot reach {_redacted_base_url_identity(self.base_url)}/models: {exc}"
                ),
                fallback_options=[
                    "Start the local LLM server (e.g., ollama serve)",
                    "Check EXTRACTION_LOCAL_LLM_BASE_URL is correct",
                ],
            ) from exc

        try:
            self.chat_completion(
                caller=__file__,
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        except Exception as exc:
            raise LocalLLMNotReadyError(
                check_failed="inference",
                endpoint_label=self.endpoint_label,
                message=_redact(
                    f"Sample inference failed for model '{self.model}': {exc}"
                ),
                fallback_options=["Verify the model is loaded and the server is running"],
            ) from exc

    def chat_completion(self, *, caller: str | None = None, **kwargs: Any) -> Any:
        from openai import OpenAI  # type: ignore[import-not-found]

        from conference_pipeline.provider_receipts import (
            local_openai_chat_completions_create,
        )

        client = OpenAI(
            api_key=os.environ.get("EXTRACTION_LOCAL_LLM_API_KEY") or "not-needed",
            base_url=self.base_url,
        )
        payload = dict(kwargs)
        payload.setdefault("model", self.model)
        # Match the in-process MLX backend, which renders its chat template with
        # thinking off: an OpenAI-compatible local server (mlx_lm.server) would
        # otherwise let a hybrid-reasoning model think first -- slower, and its
        # replies would not match earlier in-process reads. A caller may override.
        extra_body = dict(payload.get("extra_body") or {})
        template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        template_kwargs.setdefault("enable_thinking", False)
        extra_body["chat_template_kwargs"] = template_kwargs
        payload["extra_body"] = extra_body
        return local_openai_chat_completions_create(client, caller=caller, **payload)


@dataclass(frozen=True)
class MLXBackend:
    model: str
    endpoint_label: str
    kind: BackendKind = BackendKind.MLX

    def _ensure_loaded(self) -> tuple[Any, Any]:
        model_obj = getattr(self, "_loaded_model", None)
        tokenizer = getattr(self, "_loaded_tokenizer", None)
        if model_obj is not None and tokenizer is not None:
            return model_obj, tokenizer

        model_obj, tokenizer = _mlx_load_model(self.model)
        object.__setattr__(self, "_loaded_model", model_obj)
        object.__setattr__(self, "_loaded_tokenizer", tokenizer)
        _apply_extraction_metal_cache_limit()
        return model_obj, tokenizer

    def readiness(self) -> None:
        from conference_pipeline.extractions.classifier import (
            LocalLLMNotReadyError,
            _redact,
        )

        try:
            model_obj, tokenizer = self._ensure_loaded()
            _mlx_generate(
                model_obj,
                tokenizer,
                prompt="USER: ping",
                max_tokens=1,
                verbose=False,
            )
        except ImportError as exc:
            raise LocalLLMNotReadyError(
                check_failed="reachability",
                endpoint_label=self.endpoint_label,
                message="mlx_lm is not installed on this machine.",
                fallback_options=[
                    "Install the mlx extra on Apple Silicon",
                    "Force the HTTP backend with EXTRACTION_LOCAL_LLM_BACKEND=http",
                ],
            ) from exc
        except Exception as exc:
            raise LocalLLMNotReadyError(
                check_failed="inference",
                endpoint_label=self.endpoint_label,
                message=_redact(
                    f"MLX model '{self.model}' could not generate locally: {exc}"
                ),
                fallback_options=[
                    "Download or repair the MLX model locally",
                    "Force the HTTP backend with EXTRACTION_LOCAL_LLM_BACKEND=http",
                ],
            ) from exc

    def chat_completion(self, *, caller: str | None = None, **kwargs: Any) -> Any:
        max_tokens = int(kwargs.get("max_tokens") or 200)

        def _invoke() -> Any:
            model_obj, tokenizer = self._ensure_loaded()
            prompt = _messages_to_prompt(kwargs.get("messages"), tokenizer=tokenizer)
            try:
                output = _mlx_generate(
                    model_obj,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                )
            finally:
                # Prompt/generation lengths are effectively unique per call
                # across a full-corpus extraction run, so cached buffers are
                # rarely reused across calls anyway -- release them instead
                # of letting the cache climb toward the system memory ceiling.
                _release_extraction_metal_cache()
            content = _mlx_output_to_text(output)
            usage = _mlx_output_to_usage(output)
            response_id = getattr(output, "id", None) or f"mlx-{self.model}"
            message = SimpleNamespace(content=content)
            choice = SimpleNamespace(message=message, text=content)
            return SimpleNamespace(
                id=str(response_id),
                choices=[choice],
                usage=usage,
            )

        return record_provider_call(
            _invoke,
            provider="mlx-local",
            model=self.model,
            operation="chat.completions",
            caller=caller,
            request_kwargs=dict(kwargs),
        )


def _messages_to_prompt(messages: Any, *, tokenizer: Any | None = None) -> str:
    if not isinstance(messages, list):
        return ""
    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        apply_chat_template = getattr(tokenizer, "apply_chat_template")
        try:
            rendered = apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except TypeError:
            try:
                rendered = apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception:
                pass
        except Exception:
            pass
    rendered = [f"{item['role'].upper()}: {item['content']}" for item in normalized]
    return "\n\n".join(rendered)


def _mlx_output_to_text(output: Any) -> str:
    if isinstance(output, str):
        return output.strip()
    for attr in ("text", "response", "content"):
        value = getattr(output, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(output, dict):
        for key in ("text", "response", "content"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(output).strip()


def _mlx_output_to_usage(output: Any) -> Any:
    usage = getattr(output, "usage", None)
    if usage is not None:
        return usage
    if isinstance(output, dict) and isinstance(output.get("usage"), dict):
        return SimpleNamespace(**output["usage"])
    return None


def _mlx_load_function() -> Any:
    first_error: Exception | None = None
    for module_name in ("mlx_lm.utils", "mlx_lm"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            first_error = first_error or exc
            continue
        load_fn = getattr(module, "load", None)
        if callable(load_fn):
            return load_fn
    if first_error is not None:
        raise first_error
    raise RuntimeError("mlx_lm load() is unavailable")


def _mlx_generate_function() -> Any:
    first_error: Exception | None = None
    for module_name in ("mlx_lm.utils", "mlx_lm"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            first_error = first_error or exc
            continue
        generate_fn = getattr(module, "generate", None)
        if callable(generate_fn):
            return generate_fn
    if first_error is not None:
        raise first_error
    raise RuntimeError("mlx_lm generate() is unavailable")


def _mlx_load_model(model: str) -> tuple[Any, Any]:
    loaded = _mlx_load_function()(model)
    if not isinstance(loaded, tuple) or len(loaded) < 2:
        raise RuntimeError("mlx_lm load() did not return (model, tokenizer)")
    return loaded[0], loaded[1]


def _mlx_generate(*args: Any, **kwargs: Any) -> Any:
    return _mlx_generate_function()(*args, **kwargs)


def _mlx_core_module() -> Any | None:
    """Return ``mlx.core`` when importable, else ``None`` (non-Mac CI)."""
    try:
        return importlib.import_module("mlx.core")
    except ImportError:
        return None


def _apply_extraction_metal_cache_limit(mx_module: Any | None = None) -> None:
    """Bound MLX's buffer cache so it cannot grow past a fixed ceiling."""
    mx = mx_module if mx_module is not None else _mlx_core_module()
    if mx is None:
        return
    raw = os.environ.get(_EXTRACTION_CACHE_LIMIT_ENV, "").strip()
    limit = _DEFAULT_EXTRACTION_CACHE_LIMIT_BYTES
    if raw:
        try:
            limit = max(0, int(raw))
        except ValueError:
            limit = _DEFAULT_EXTRACTION_CACHE_LIMIT_BYTES
    setter = getattr(mx, "set_cache_limit", None)
    if not callable(setter):
        setter = getattr(getattr(mx, "metal", None), "set_cache_limit", None)
    if callable(setter):
        try:
            setter(limit)
        except Exception:  # noqa: BLE001 - cache tuning must not break generation
            pass


def _release_extraction_metal_cache(mx_module: Any | None = None) -> None:
    """Free cached MLX buffers accumulated by this call's generation."""
    mx = mx_module if mx_module is not None else _mlx_core_module()
    if mx is None:
        return
    clearer = getattr(mx, "clear_cache", None)
    if not callable(clearer):
        clearer = getattr(getattr(mx, "metal", None), "clear_cache", None)
    if callable(clearer):
        try:
            clearer()
        except Exception:  # noqa: BLE001 - cache release must not break generation
            pass


def select_local_llm_backend(model: str | None = None) -> Backend:
    requested_model = _requested_model(model)
    forced_backend = (
        os.environ.get("EXTRACTION_LOCAL_LLM_BACKEND") or ""
    ).strip().lower()
    is_apple_silicon = _is_apple_silicon()
    mlx_available = _try_import_mlx()
    mlx_reachable = mlx_available and _mlx_has_model_locally(requested_model)

    if forced_backend and forced_backend not in {"mlx", "http"}:
        raise BackendSelectionError(
            "EXTRACTION_LOCAL_LLM_BACKEND must be 'mlx' or 'http'."
        )

    if forced_backend == "mlx":
        if not is_apple_silicon:
            raise BackendSelectionError(
                "EXTRACTION_LOCAL_LLM_BACKEND=mlx requires Apple Silicon."
            )
        if not mlx_available:
            raise BackendSelectionError(
                "EXTRACTION_LOCAL_LLM_BACKEND=mlx but mlx_lm is not importable."
            )
        return MLXBackend(
            model=requested_model,
            endpoint_label=_custom_endpoint_label() or f"mlx://{requested_model}",
        )

    if forced_backend == "http":
        return _build_http_backend(requested_model)

    if is_apple_silicon and mlx_reachable:
        return MLXBackend(
            model=requested_model,
            endpoint_label=_custom_endpoint_label() or f"mlx://{requested_model}",
        )
    return _build_http_backend(requested_model)


__all__ = [
    "Backend",
    "BackendKind",
    "BackendSelectionError",
    "HTTPBackend",
    "MLXBackend",
    "select_local_llm_backend",
]
