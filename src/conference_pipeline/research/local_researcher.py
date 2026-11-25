"""LocalLLMResearcher — synthesis adapter against a local OpenAI-compatible
endpoint (v1.9 Phase 46).

Mirrors :class:`conference_pipeline.research.researcher.OpenAIResearcher`
shape but points the OpenAI SDK at a loopback endpoint
(``http://127.0.0.1:8080/v1`` for ``mlx_lm.server`` or Ollama's
OpenAI-compat endpoint). Loopback-only — the constructor raises
:class:`ResearcherError` before any HTTP if the configured base URL
isn't loopback.

Test injection: pass a fake OpenAI-compatible client via the ``_client``
constructor arg. The fake must expose the chat-completions create method
returning an object with ``.choices[0].message.content`` and an
optional ``.usage`` attribute. This mirrors the project's Fake-test-
double convention (``FakeRetriever``, ``FakeTavilyClient``,
``FakeResearcher``, ``FakeEmbeddingClient``).

Receipts: synthesis calls flow through
:func:`provider_receipts.local_openai_chat_completions_create`, which
records ``provider="local-openai-compatible"`` so receipts on disk are
correctly attributed to the local lane (not paid OpenAI). Cost
tracking returns ``0.0`` (``cost_status="local_no_paid_provider"``) by
contract.

Endpoint label convention: ``http://localhost:<port>`` (host always
redacted to ``localhost`` regardless of input URL, matching the v1.8
Phase 36 backend label format).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from conference_pipeline.analysis.steps.gpt_researcher_hybrid import (
    REFERENCE_QUALITY_PROMPT,
)
from conference_pipeline._security.loopback import is_loopback_url
from conference_pipeline.iclr2025_live import sanitize_for_receipt
from conference_pipeline.indexing.ollama_embeddings import _redact_loopback
from conference_pipeline.provider_receipts import (
    local_openai_chat_completions_create,
)
from conference_pipeline.research.researcher import ResearcherError

LOG = logging.getLogger("conference_pipeline.research.local_researcher")

_DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
_DEFAULT_FAST_MODEL = "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
_DEFAULT_SMART_MODEL = "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
_SENTINEL_API_KEY = "local-only"
# Matches analysis/topic_reconciliation.py's DEFAULT_MAX_OUTPUT_TOKENS for the
# same reasoning-capable local model family: enough headroom for a full
# GPT-Researcher-style findings synthesis even with enable_thinking disabled.
DEFAULT_MAX_OUTPUT_TOKENS = 16_000


class LocalLLMResearcher:
    """Production synthesis adapter against a local OpenAI-compatible endpoint.

    Mirrors :class:`OpenAIResearcher` shape: synthesis-only (no
    retriever — orchestration lives in
    :mod:`conference_pipeline.analysis.steps.gpt_researcher_hybrid`).
    The retrieved context is interpolated into the user prompt by the
    upstream caller before invoking :meth:`run`.

    Loopback-only: constructor validates the base URL via
    :func:`is_loopback_url` and raises :class:`ResearcherError`
    immediately if non-loopback. ``api_key`` is set to a sentinel value
    ``"local-only"`` — the OpenAI SDK requires one but never sends
    anything sensitive to a loopback endpoint in any reasonable
    deployment.
    """

    provider: str = "local"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        fast_model: str | None = None,
        smart_model: str | None = None,
        endpoint_label: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        _client: Any | None = None,
    ) -> None:
        resolved = (
            base_url
            or os.environ.get("GPTR_LOCAL_LLM_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        if not is_loopback_url(resolved):
            raise ResearcherError(
                "non_loopback_base_url: LocalLLMResearcher requires a "
                "loopback endpoint (127.0.0.1, localhost, ::1, or "
                "127.0.0.0/8); got "
                f"{resolved!r}."
            )
        self._base_url = resolved
        self._endpoint_label = (
            endpoint_label
            or os.environ.get("GPTR_LOCAL_LLM_ENDPOINT_LABEL")
            or _redact_loopback(resolved)
        )
        self.fast_model = (
            fast_model
            or os.environ.get("GPTR_LOCAL_LLM_FAST_MODEL")
            or _DEFAULT_FAST_MODEL
        )
        self.smart_model = (
            smart_model
            or os.environ.get("GPTR_LOCAL_LLM_SMART_MODEL")
            or self.fast_model
        )
        # Synthesis uses smart_model by default (mirrors gpt-researcher
        # convention: smart_llm = report-grade synthesis).
        self.model = self.smart_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = _SENTINEL_API_KEY
        self._client_override = _client
        # Telemetry parity with OpenAIResearcher.
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0
        self.estimated_cost_usd = 0.0
        self.provider_receipt_paths: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def endpoint_label(self) -> str:
        return self._endpoint_label

    def _get_client(self) -> Any:
        """Return the test-injected client OR construct an OpenAI SDK client.

        The OpenAI SDK accepts ``base_url`` and ``api_key`` constructor
        args; pointing it at MLX/Ollama loopback is just a config swap.
        """
        if self._client_override is not None:
            return self._client_override
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ResearcherError("openai_not_available") from None  # noqa: B904
        return OpenAI(base_url=self._base_url, api_key=self.api_key)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def run(
        self, query: str, *, context_path: Path | None = None  # noqa: ARG002
    ) -> str:
        """Run a single synthesis call against the local endpoint.

        Returns the model's response content. Tracks usage on the
        instance for downstream :meth:`usage` reporting.

        :param context_path: Unused — interface parity with
            :class:`OpenAIResearcher`. The retrieved context is
            interpolated into ``query`` by the upstream caller.
        """
        try:
            client = self._get_client()
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": REFERENCE_QUALITY_PROMPT,
                    },
                    {"role": "user", "content": query},
                ],
                # Reasoning-capable local models (e.g. Qwen3.6-OptiQ) otherwise
                # spend the entire output budget on a hidden reasoning trace and
                # return an empty message.content, which callers then treat as
                # a genuine "no findings" result rather than a config gap.
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature
            request_kwargs["max_tokens"] = int(self.max_tokens) if self.max_tokens is not None else DEFAULT_MAX_OUTPUT_TOKENS
            response = local_openai_chat_completions_create(
                client,
                caller=__file__,
                **request_kwargs,
            )
            if response.receipt.estimated_cost_usd is not None:
                self.estimated_cost_usd += response.receipt.estimated_cost_usd
            if response.receipt.receipt_path:
                self.provider_receipt_paths.append(response.receipt.receipt_path)
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.output_tokens += int(
                    getattr(usage, "completion_tokens", 0) or 0
                )
            self.request_count += 1
            return content
        except Exception as exc:  # noqa: BLE001 - boundary catch
            sanitized = sanitize_for_receipt(str(exc))
            msg = sanitized if isinstance(sanitized, str) else "local_synthesis_failed"
            raise ResearcherError(msg) from None

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def usage(self) -> dict[str, Any]:
        """Telemetry parity with :meth:`OpenAIResearcher.usage`."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "request_count": self.request_count,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "provider_receipt_paths": list(self.provider_receipt_paths),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


_EXPANSION_USER_TEMPLATE = (
    "Topic: {topic_title}\n\n"
    "Original research questions:\n{questions_text}\n\n"
    "Generate {n} paraphrase or alternate-framing variants of these questions that "
    "could surface additional relevant evidence. Return only the new questions, one "
    "per line, without numbering or bullet prefixes."
)


class LocalLLMExpansionClient:
    """Synchronous query expansion via local LLM paraphrase/alternate framing.

    Implements the ``QueryExpansionLLMClient`` protocol from
    :mod:`conference_pipeline.research.query_planner`. Loopback-only —
    constructor validates the base URL and raises :class:`ResearcherError`
    before any HTTP if it is non-loopback.

    Test injection: pass a fake OpenAI-compatible client via the ``_client``
    constructor arg. The fake must expose ``chat.completions.create``
    returning an object with ``.choices[0].message.content``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_variants: int = 3,
        _client: Any | None = None,
    ) -> None:
        resolved = (
            base_url
            or os.environ.get("GPTR_LOCAL_LLM_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        if not is_loopback_url(resolved):
            raise ResearcherError(
                "non_loopback_base_url: LocalLLMExpansionClient requires a "
                "loopback endpoint; got "
                f"{resolved!r}."
            )
        self._base_url = resolved
        self.model = (
            model
            or os.environ.get("GPTR_LOCAL_LLM_FAST_MODEL")
            or _DEFAULT_FAST_MODEL
        )
        self.temperature = temperature
        self.max_variants = max(1, int(max_variants))
        self.api_key = _SENTINEL_API_KEY
        self._client_override = _client

    def _get_client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError:
            raise ResearcherError("openai_not_available") from None  # noqa: B904
        return OpenAI(base_url=self._base_url, api_key=self.api_key)

    def expand_queries(
        self, topic_title: str, base_questions: list[str]
    ) -> list[str]:
        """Return paraphrase/alternate-framing variant questions.

        On any failure, raises :class:`ResearcherError` so callers can
        apply deterministic fallback per D-10.
        """
        try:
            questions_text = "\n".join(f"- {q}" for q in base_questions[:5])
            prompt = _EXPANSION_USER_TEMPLATE.format(
                topic_title=topic_title,
                questions_text=questions_text,
                n=self.max_variants,
            )
            client = self._get_client()
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature
            response = local_openai_chat_completions_create(
                client, caller=__file__, **request_kwargs
            )
            content = response.choices[0].message.content or ""
            variants: list[str] = []
            for line in content.splitlines():
                line = line.strip().lstrip("-• ").strip()
                if line and len(line) > 5:
                    variants.append(line)
            return variants[: self.max_variants]
        except ResearcherError:
            raise
        except Exception as exc:  # noqa: BLE001 - boundary catch
            sanitized = sanitize_for_receipt(str(exc))
            msg = (
                sanitized if isinstance(sanitized, str) else "local_expansion_failed"
            )
            raise ResearcherError(msg) from None


__all__ = ["LocalLLMExpansionClient", "LocalLLMResearcher"]
