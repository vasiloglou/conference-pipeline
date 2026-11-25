"""Token-budget policy helpers for conference corpus indexing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from typing import Any

DEFAULT_CHUNK_TOKENS = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS = 2048
DEFAULT_EMBEDDING_TOKENIZER_ID = "tiktoken:text-embedding-3-small"


@dataclass(frozen=True)
class TokenBudgetPolicy:
    """Serializable token budget contract for chunking and embedding inputs."""

    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP
    hard_embedding_tokens: int = DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS
    tokenizer_id: str = DEFAULT_EMBEDDING_TOKENIZER_ID
    tokenizer_version: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("chunk_tokens", "hard_embedding_tokens"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive int")
        overlap = self.chunk_overlap_tokens
        if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
            raise ValueError("chunk_overlap_tokens must be a non-negative int")
        if not self.tokenizer_id:
            raise ValueError("tokenizer_id must be non-empty")

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _approx_token_count(text: str) -> int:
    return max(len((text or "").split()), len(text or "") // 4)


def _tokenizer_version(tokenizer_id: str) -> str | None:
    if tokenizer_id.startswith("tiktoken:"):
        try:
            return importlib_metadata.version("tiktoken")
        except importlib_metadata.PackageNotFoundError:
            return None
    return None


def default_token_budget_policy() -> TokenBudgetPolicy:
    return TokenBudgetPolicy(
        tokenizer_version=_tokenizer_version(DEFAULT_EMBEDDING_TOKENIZER_ID)
    )


def token_count(
    text: str,
    *,
    tokenizer_id: str = DEFAULT_EMBEDDING_TOKENIZER_ID,
) -> int:
    """Best-effort token count for embedding input budgeting."""
    text = text or ""
    if len(text) > 50000:
        return _approx_token_count(text)
    if tokenizer_id.startswith("tiktoken:"):
        try:
            import tiktoken  # type: ignore[import-not-found]

            model = tokenizer_id.split(":", 1)[1]
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:  # noqa: BLE001 - optional dependency/fallback
            return _approx_token_count(text)
    return _approx_token_count(text)


def policy_metadata(policy: TokenBudgetPolicy) -> dict[str, object]:
    metadata: dict[str, Any] = asdict(policy)
    metadata["policy_digest"] = policy.digest()
    return metadata


__all__ = [
    "DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_TOKENS",
    "DEFAULT_EMBEDDING_TOKENIZER_ID",
    "TokenBudgetPolicy",
    "default_token_budget_policy",
    "policy_metadata",
    "token_count",
]
