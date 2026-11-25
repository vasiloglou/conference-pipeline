"""Vector-store collection-name derivation + dim-mismatch guard (v1.9 Phase 45).

Provider-aware Chroma collection naming so cloud and local lanes can
coexist in the same persistent directory:

- ``neurips_presentations__openai_text-embedding-3-small_1536``
- ``neurips_presentations__mlx_qwen-qwen3-embedding-0-6b_1024``
- ``neurips_presentations__ollama_nomic-embed-text_768``

The dim-mismatch guard fail-stops with a remediation message that
names the rebuild CLI command, so an operator switching providers
gets a clear next step instead of silent corruption (or worse, a
mid-flight Chroma exception).

Legacy collections built before v1.9 (no ``metadata['dim']``) pass
through the guard unchanged — the first rebuild populates the
metadata dict.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Chroma collection-name constraints: 3..63 chars, [a-zA-Z0-9._-], must
# start/end with alphanumeric. Our slug enforces a stricter subset.
_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_CHROMA_NAME_MAX = 63
_CHROMA_NAME_MIN = 3
# v1.9 review HIGH-7: validate the operator-supplied ``base`` against
# Chroma's documented name regex *before* concatenation so the failure
# mode is a clear ValueError naming the offending parameter, not a
# deep-stack chromadb traceback. The pattern enforces start/end with
# alphanumeric and only ``[a-zA-Z0-9._-]`` between.
_BASE_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]|[a-zA-Z0-9]")


class VectorStoreDimensionMismatch(Exception):
    """Raised when persisted Chroma collection dim disagrees with active client.

    Carries enough info for the receiving handler to display a
    remediation command (the v1.9 ``vector-store rebuild`` CLI) to the
    operator instead of just stack-tracing.
    """

    def __init__(
        self,
        *,
        collection: str,
        persisted_dim: int,
        active_dim: int,
        active_provider: str,
        active_model: str,
    ) -> None:
        self.collection = collection
        self.persisted_dim = int(persisted_dim)
        self.active_dim = int(active_dim)
        self.active_provider = active_provider
        self.active_model = active_model
        super().__init__(self._render_message())

    def _render_message(self) -> str:
        return (
            f"vector store collection {self.collection!r} was built at "
            f"dim {self.persisted_dim}; active embedding provider "
            f"{self.active_provider!r} / model {self.active_model!r} "
            f"produces dim {self.active_dim}. Rebuild with: "
            f"conference-pipeline vector-store rebuild "
            f"--provider {self.active_provider}"
        )


def _slug(model: str, *, max_len: int = 30) -> str:
    """Sanitize a model identifier for use in a Chroma collection name.

    Lowercases, collapses runs of non-``[a-z0-9-]`` to single hyphens,
    strips leading/trailing hyphens, and truncates to ``max_len`` chars.
    """
    if not isinstance(model, str):
        raise ValueError(
            f"_slug requires a str model identifier; got {type(model).__name__}"
        )
    sanitized = _SLUG_RE.sub("-", model.lower()).strip("-")
    return sanitized[:max_len].rstrip("-")


def derive_collection_name(
    base: str, provider: str, model: str, dim: int
) -> str:
    """Return a Chroma-safe suffixed collection name.

    Format: ``<base>__<provider>_<model_slug>_<dim>`` with aggressive
    slug truncation if the total length exceeds Chroma's 63-char limit.
    Always preserves base, provider, and dim verbatim.

    Raises:
        ValueError: derived name violates Chroma constraints (3..63 chars).
    """
    if not base:
        raise ValueError("derive_collection_name: 'base' must be non-empty")
    # v1.9 review HIGH-7: reject operator-supplied base values that would
    # produce a Chroma-rejected collection name. Without this, an operator
    # passing ``--collection-base "foo bar"`` (space) or ``"_leading"``
    # would see the failure surface from chromadb deep in the call stack
    # with no reference back to the parameter that caused it.
    if not _BASE_RE.fullmatch(base):
        raise ValueError(
            "derive_collection_name: 'base' must match Chroma's name regex "
            f"(start/end alphanumeric, [a-zA-Z0-9._-] between); got {base!r}"
        )
    if not provider:
        raise ValueError("derive_collection_name: 'provider' must be non-empty")
    if dim is None or int(dim) <= 0:
        raise ValueError(
            f"derive_collection_name: 'dim' must be positive int; got {dim!r}"
        )
    name = f"{base}__{provider}_{_slug(model)}_{dim}"
    if len(name) > _CHROMA_NAME_MAX:
        # Aggressive trim of the model slug; prefix + dim preserved verbatim.
        prefix_overhead = len(f"{base}__{provider}__{dim}")
        max_slug_len = max(1, _CHROMA_NAME_MAX - prefix_overhead)
        name = f"{base}__{provider}_{_slug(model, max_len=max_slug_len)}_{dim}"
    if not (_CHROMA_NAME_MIN <= len(name) <= _CHROMA_NAME_MAX):
        raise ValueError(
            f"derived collection name {name!r} violates Chroma limits "
            f"(must be {_CHROMA_NAME_MIN}..{_CHROMA_NAME_MAX} chars)"
        )
    return name


def guard_dimension(
    collection: "Any",
    *,
    active_provider: str,
    active_model: str,
    active_dim: int,
) -> None:
    """Raise :class:`VectorStoreDimensionMismatch` when persisted dim differs.

    Reads ``collection.metadata['dim']``. Legacy collections without
    metadata (pre-v1.9) pass through unchanged — the next rebuild will
    populate the metadata dict so subsequent guard calls have something
    to compare against.
    """
    metadata = getattr(collection, "metadata", None)
    persisted_dim_raw = (metadata or {}).get("dim")
    if persisted_dim_raw is None:
        return
    persisted_dim = int(persisted_dim_raw)
    if persisted_dim == int(active_dim):
        return
    raise VectorStoreDimensionMismatch(
        collection=getattr(collection, "name", "<unknown>"),
        persisted_dim=persisted_dim,
        active_dim=active_dim,
        active_provider=active_provider,
        active_model=active_model,
    )


__all__ = [
    "VectorStoreDimensionMismatch",
    "derive_collection_name",
    "guard_dimension",
]
