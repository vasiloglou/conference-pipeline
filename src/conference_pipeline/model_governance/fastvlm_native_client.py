"""v1.61 Phase 68 — Native MLX FastVLM adapter.

Apple's FastVLM family (``apple/FastVLM-7B-MLX``, ``apple/FastVLM-1.5B-MLX``,
``apple/FastVLM-0.5B-MLX``) ships native MLX checkpoints with sub-second
TTFT on Apple Silicon. This adapter loads them in-process — no HTTP server
in front — so the lifecycle is just:

* Construction: no I/O. Pass ``loader=`` to inject a fake in tests.
* First ``complete()``: lazy load via ``mlx_vlm.utils.load`` (or whatever
  loader was injected). Subsequent calls reuse the loaded model.

Image-only contract: text-only calls (no image bytes) raise
:class:`FastVLMTextOnlyUnsupported` to fail fast and avoid silently
calling a non-existent text-only path.

Receipts: the ``SlideAnalysisEngine.local_fastvlm`` provider (Phase 67)
writes a :class:`LocalVisionCallReceipt` with
``runtime_kind="mlx_native_fastvlm"`` so cost analysis can distinguish
native FastVLM calls from server-fronted vllm-mlx calls.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from conference_pipeline.model_governance.registry import ModelGovernanceError


LOG = logging.getLogger("conference_pipeline.model_governance.fastvlm_native_client")

_DEFAULT_MODEL_ID = "apple/FastVLM-7B-MLX"
_SUPPORTED_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp")


class FastVLMTextOnlyUnsupported(ModelGovernanceError):
    """Raised when a text-only call is attempted on the image-only FastVLM path."""


class LocalVisionRuntimeError(ModelGovernanceError):
    """Raised when the in-process FastVLM runtime errors (e.g. OOM)."""


class FastVLMNativeClient:
    """In-process MLX FastVLM image-only inference client.

    Constructor performs no I/O — model weights load lazily on first
    :meth:`complete` call. Pass ``loader`` to inject a fake `(model,
    processor)` factory in tests. The image-only contract is enforced:
    callers MUST pass ``image_bytes``; text-only invocation raises
    :class:`FastVLMTextOnlyUnsupported`.
    """

    provider: str = "local_fastvlm"
    runtime_kind: str = "mlx_native_fastvlm"

    def __init__(
        self,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        loader: Callable[[str], tuple[Any, Any]] | None = None,
        generator: Callable[..., str] | None = None,
    ) -> None:
        self._model_id = str(model_id)
        self._loader = loader
        self._generator = generator
        self._model: Any | None = None
        self._processor: Any | None = None

    # --------------------------------------------------------- attributes

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def endpoint_label(self) -> str:
        # Distinct prefix from server-fronted mlx-vision:// so receipt
        # consumers can tell native FastVLM calls apart from vllm-mlx calls.
        return f"mlx-fastvlm://{self._model_id}"

    # --------------------------------------------------------- availability

    @classmethod
    def is_available(cls) -> bool:
        """Return True iff ``mlx_vlm`` is importable here."""
        try:
            import mlx_vlm  # noqa: F401
        except ImportError:
            return False
        return True

    # --------------------------------------------------------- inference

    def complete(
        self,
        *,
        image_bytes: bytes,
        media_type: str,
        user_prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        """Run a single image+text inference and return the model's text output.

        Image bytes are NOT written to disk and are NOT logged. ``media_type``
        must be one of ``image/png``, ``image/jpeg``, ``image/webp``. A
        text-only call (empty ``image_bytes``) fails fast with
        :class:`FastVLMTextOnlyUnsupported`.
        """
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise FastVLMTextOnlyUnsupported(
                "FastVLMNativeClient is image-only; image_bytes must be non-empty. "
                "For text-only completions, use the vllm-mlx server lane "
                "(MlxVisionServerClient or LLMOcrEngine)."
            )
        if media_type not in _SUPPORTED_MEDIA_TYPES:
            raise ValueError(
                f"FastVLMNativeClient.complete: media_type must be one of "
                f"{_SUPPORTED_MEDIA_TYPES}, got {media_type!r}"
            )
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")

        self._ensure_loaded()
        try:
            return self._invoke_generator(
                image_bytes=bytes(image_bytes),
                media_type=media_type,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except (FastVLMTextOnlyUnsupported, ValueError):
            raise
        except MemoryError as exc:
            raise LocalVisionRuntimeError(
                f"FastVLMNativeClient OOM while running {self._model_id!r}"
            ) from exc
        except Exception as exc:
            raise LocalVisionRuntimeError(
                f"FastVLMNativeClient inference failed for {self._model_id!r}"
            ) from exc

    # --------------------------------------------------------- internals

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        if self._loader is not None:
            model, processor = self._loader(self._model_id)
        else:
            try:
                from mlx_vlm.utils import load  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ModelGovernanceError(
                    "mlx_vlm is required for FastVLMNativeClient; "
                    "install via `pip install conference-pipeline[mlx]`"
                ) from exc
            try:
                model, processor = load(self._model_id)
            except Exception as exc:
                raise LocalVisionRuntimeError(
                    f"failed to load FastVLM weights with mlx_vlm for {self._model_id!r}"
                ) from exc
        self._model = model
        self._processor = processor

    def _invoke_generator(
        self,
        *,
        image_bytes: bytes,
        media_type: str,
        user_prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if self._generator is not None:
            return self._generator(
                model=self._model,
                processor=self._processor,
                model_id=self._model_id,
                image_bytes=image_bytes,
                media_type=media_type,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        # Production path: route through mlx_vlm.generate. We materialize the
        # image bytes into a tempfile here because mlx_vlm.generate accepts a
        # filesystem path; the tempfile is created with an O_TMPFILE-style
        # lifecycle (auto-deleted) so bytes never persist across the call.
        try:
            from mlx_vlm import generate  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ModelGovernanceError(
                "mlx_vlm is required for FastVLMNativeClient.complete()"
            ) from exc
        import tempfile

        suffix = ".png"
        if media_type == "image/jpeg":
            suffix = ".jpg"
        elif media_type == "image/webp":
            suffix = ".webp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            prompt = user_prompt
            if system_prompt:
                prompt = f"{system_prompt}\n\n{user_prompt}"
            result = generate(
                self._model,
                self._processor,
                image=tmp.name,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )
        if isinstance(result, str):
            return result.strip()
        # Some mlx_vlm versions return tuples — pluck the text.
        if isinstance(result, tuple) and result and isinstance(result[0], str):
            return result[0].strip()
        return str(result).strip()
