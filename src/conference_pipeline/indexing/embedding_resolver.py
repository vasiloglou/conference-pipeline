"""Single embedding-provider resolver — LOCAL-03, D-09/D-10/D-12.

This module is the **sole** source of truth for embedding provider
selection. All read sites (corpus_vector_store walker write,
iclr2025_live retriever read, GPT-Researcher embedding calls) delegate
here instead of reading ``EMBEDDING_PROVIDER`` directly.

Design decisions honoured:
- D-09: one resolver; three scattered ``os.environ`` reads replaced.
- D-10: ``openai`` provider requires explicit opt-in via ``allow_paid=True``
  **or** ``CONFERENCE_ALLOW_PAID_EMBEDDINGS=1``; raises
  :class:`PaidEmbeddingApprovalRequired` otherwise.
- D-12: every call site routes through ``build_resolved_embedding_client``.
"""

from __future__ import annotations

import os
from typing import Any, Callable


class PaidEmbeddingApprovalRequired(RuntimeError):
    """Raised when a paid embedding provider is selected without explicit opt-in.

    T-03-07 mitigation: OpenAI embeddings cannot be selected accidentally.
    The operator must pass ``allow_paid=True`` or set
    ``CONFERENCE_ALLOW_PAID_EMBEDDINGS=1``.
    """


def _optional_positive_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSY_ENV_VALUES = {"0", "false", "no", "off"}

# Canonical env var that records WHY a paid embedding provider was allowed.
PAID_OPT_IN_ENV = "CONFERENCE_ALLOW_PAID_EMBEDDINGS"


def _bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value in _TRUTHY_ENV_VALUES:
        return True
    if value in _FALSY_ENV_VALUES:
        return False
    return default


def _paid_opt_in_approved(
    allow_paid: bool,
    env: "os._Environ[str] | dict[str, str] | None" = None,
) -> bool:
    """Return True when a paid embedding provider is approved (D-10, WR-04).

    Single source of truth for the paid-embedding gate so the env-var parsing is
    not copy-pasted across resolve_embedding_config and build_resolved_embedding_
    client (which previously drifted). Approval requires an explicit
    ``allow_paid=True`` OR a truthy ``CONFERENCE_ALLOW_PAID_EMBEDDINGS`` env value.
    """
    if allow_paid:
        return True
    _env = os.environ if env is None else env
    return str(_env.get(PAID_OPT_IN_ENV, "")).strip().lower() in _TRUTHY_ENV_VALUES


def resolve_embedding_config(
    env: dict[str, str] | None = None,
    *,
    allow_paid: bool = False,
) -> dict[str, object]:
    """Return the embedding configuration dict for the current environment.

    Reads ``EMBEDDING_PROVIDER``, ``EMBEDDING_MODEL``, ``EMBEDDING_DIM``,
    ``EMBEDDING_BATCH_SIZE``, and ``EMBEDDING_MAX_LENGTH`` from ``env``
    (defaults to ``os.environ``).

    Defaults (D-09):
    - ``mlx``   → model ``mlx-community/bge-m3-mlx-8bit``, dim 1024, max_length 2048
    - ``ollama`` → model ``nomic-embed-text``, dim 768
    - ``fake``  → dim 1536

    Raises :class:`PaidEmbeddingApprovalRequired` when the resolved
    provider is ``openai`` and neither ``allow_paid=True`` nor
    ``CONFERENCE_ALLOW_PAID_EMBEDDINGS=1`` is present (D-10, T-03-07).
    """
    _env: dict[str, str] = os.environ if env is None else env  # type: ignore[assignment]

    def _get(name: str, default: str = "") -> str:
        return str(_env.get(name, default)).strip()

    def _pos_int(name: str) -> int | None:
        value = _get(name)
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    provider = _get("EMBEDDING_PROVIDER") or "mlx"
    provider = provider.lower()

    config: dict[str, object] = {"provider": provider}

    model = _get("EMBEDDING_MODEL")
    dim = _pos_int("EMBEDDING_DIM")
    batch_size = _pos_int("EMBEDDING_BATCH_SIZE")
    max_length = _pos_int("EMBEDDING_MAX_LENGTH")

    if model:
        config["model"] = model
    if dim is not None:
        config["embedding_dim"] = dim
    if batch_size is not None:
        config["batch_size"] = batch_size
    if max_length is not None:
        config["max_length"] = max_length

    if provider == "mlx":
        from conference_pipeline.model_governance.registry import model_for_alias

        config.setdefault("model", model_for_alias("bge_m3_mlx"))
        config.setdefault("embedding_dim", 1024)
        config.setdefault("max_length", 2048)

    elif provider == "ollama":
        config.setdefault("model", "nomic-embed-text")
        config.setdefault("embedding_dim", 768)
        base_url = _get("OLLAMA_BASE_URL")
        if base_url:
            config["base_url"] = base_url
        manage_server_val = _get("OLLAMA_MANAGE_SERVER", "").lower()
        if manage_server_val in {"1", "true", "yes", "on"}:
            config["manage_server"] = True
        elif manage_server_val in {"0", "false", "no", "off"}:
            config["manage_server"] = False
        else:
            config["manage_server"] = True
        startup_timeout_str = _get("OLLAMA_STARTUP_TIMEOUT_SECONDS")
        if startup_timeout_str:
            try:
                st = float(startup_timeout_str)
                if st > 0:
                    config["startup_timeout_s"] = st
            except ValueError:
                pass

    elif provider == "fake":
        config.setdefault("embedding_dim", 1536)

    elif provider == "openai":
        # D-10 / T-03-07: paid provider requires explicit operator opt-in.
        if not _paid_opt_in_approved(allow_paid, _env):
            raise PaidEmbeddingApprovalRequired(
                "openai embeddings require explicit paid opt-in; "
                "pass allow_paid=True or set CONFERENCE_ALLOW_PAID_EMBEDDINGS=1"
            )
        # Record why paid was allowed so downstream receipts can attribute it.
        config["paid_opt_in_ref"] = (
            "allow_paid" if allow_paid else PAID_OPT_IN_ENV
        )

    return config


def build_resolved_embedding_client(
    config: dict[str, object] | None = None,
    *,
    allow_paid: bool = False,
    build_embedding_client_fn: Callable[[dict[str, object]], Any] | None = None,
) -> tuple[dict[str, object], Any]:
    """Resolve embedding config and construct the client.

    Returns ``(config, client)`` where ``config`` is the resolved dict and
    ``client`` is the constructed embedding client.

    D-10: raises :class:`PaidEmbeddingApprovalRequired` when the provider
    resolves to ``openai`` without an explicit paid opt-in signal.

    D-12: all embedding call sites should use this function so there is
    a single canonical path through the resolver.
    """
    if build_embedding_client_fn is None:
        from conference_pipeline.indexing.embeddings import build_embedding_client as _build

        build_embedding_client_fn = _build

    resolved_config = config if config is not None else resolve_embedding_config(allow_paid=allow_paid)

    # If a pre-resolved config was provided, still enforce the paid gate
    # (WR-04: route through the single helper so this cannot drift from
    # resolve_embedding_config's gate).
    if config is not None and str(resolved_config.get("provider") or "") == "openai":
        if not _paid_opt_in_approved(allow_paid):
            raise PaidEmbeddingApprovalRequired(
                "openai embeddings require explicit paid opt-in; "
                "pass allow_paid=True or set CONFERENCE_ALLOW_PAID_EMBEDDINGS=1"
            )
        # Record why paid was allowed for a pre-resolved openai config so the
        # downstream receipt can attribute the approval source.
        resolved_config.setdefault(
            "paid_opt_in_ref", "allow_paid" if allow_paid else PAID_OPT_IN_ENV
        )

    client = build_embedding_client_fn(resolved_config)
    return resolved_config, client


# ---------------------------------------------------------------------------
# v3.0 03-04: cost-avoided estimate for local embedding substitutions (LOCAL-04)
# ---------------------------------------------------------------------------
#
# DEFAULT_PAID_EMBEDDING_PROVIDER / DEFAULT_PAID_EMBEDDING_MODEL define the
# counterfactual paid provider that would have been used had the operator not
# switched to local (mlx) embeddings.  These defaults match the D-10 opt-in
# paid lane so the counterfactual reflects a real alternative, not a guess.

DEFAULT_PAID_EMBEDDING_PROVIDER = "openai"
DEFAULT_PAID_EMBEDDING_MODEL = "text-embedding-3-small"


def estimate_local_embedding_cost_avoided(
    task_id: str,
    *,
    input_count: int = 0,
    total_input_chars: int = 0,
    paid_provider: str = DEFAULT_PAID_EMBEDDING_PROVIDER,
    paid_model: str = DEFAULT_PAID_EMBEDDING_MODEL,
    benchmark_receipt_ref: str | None = None,
    estimate_cost_avoided_fn: Any = None,
) -> dict[str, object]:
    """Estimate the cost avoided by using a local embedding model.

    Calls ``estimate_cost_avoided`` (or the injected fake) with the paid
    counterfactual and returns a dict suitable for passing to
    ``record_local_embedding_call`` as ``**cost_avoided_kwargs``.

    Parameters
    ----------
    task_id:
        Model-governance task ID for catalog lookup.
    input_count:
        Number of texts embedded.  Recorded for context only; it is NOT used to
        fabricate a token quantity.  When ``total_input_chars`` is absent the
        quantity stays ``None`` (→ unknown / backfill) rather than guessing
        tokens-per-text (WR-06 / D-16).
    total_input_chars:
        Total input character count; proxy for tokens (chars/4).  The sole basis
        for a measured token quantity.
    paid_provider:
        Paid counterfactual provider (default ``"openai"``).
    paid_model:
        Paid counterfactual model (default ``"text-embedding-3-small"``).
    benchmark_receipt_ref:
        Path/ref to the benchmark receipt that approved this local model.
    estimate_cost_avoided_fn:
        Optional injected callable (signature of ``estimate_cost_avoided``).
        When ``None``, the production helper is used.  Inject a fake for
        offline tests so no live provider calls are made.

    Returns
    -------
    dict with keys matching the cost-avoided kwargs of ``record_local_embedding_call``:
        benchmark_receipt_ref, cost_avoided_usd, cost_avoided_status,
        paid_counterfactual_provider, paid_counterfactual_model,
        needs_pricing_backfill.
    """
    if estimate_cost_avoided_fn is None:
        from conference_pipeline.costs.cost_avoided import estimate_cost_avoided
        estimate_cost_avoided_fn = estimate_cost_avoided

    # Derive token quantity from character count (chars/4 ≈ tokens).
    #
    # WR-06 / D-16: only a measured character count yields a real quantity. When
    # total_input_chars is absent we deliberately leave quantity=None instead of
    # fabricating input_count*100 tokens. A guessed token count would be priced
    # and stamped "priced"/needs_pricing_backfill=False, which a downstream
    # auditor could not distinguish from a measured saving. quantity=None routes
    # estimate_cost_avoided to its unknown result (cost_avoided_usd=None,
    # cost_avoided_status="unknown", needs_pricing_backfill=True), so an
    # uncertain counterfactual stays flagged for backfill rather than fabricated.
    quantity: float | None = None
    if total_input_chars > 0:
        quantity = float(total_input_chars) / 4.0

    ca_result = estimate_cost_avoided_fn(
        task_id,
        paid_provider=paid_provider,
        paid_model=paid_model,
        quantity=quantity,
    )

    return {
        "benchmark_receipt_ref": benchmark_receipt_ref,
        "cost_avoided_usd": ca_result.get("cost_avoided_usd"),
        "cost_avoided_status": ca_result.get("cost_avoided_status"),
        "paid_counterfactual_provider": ca_result.get("paid_counterfactual_provider"),
        "paid_counterfactual_model": ca_result.get("paid_counterfactual_model"),
        "needs_pricing_backfill": ca_result.get("needs_pricing_backfill"),
    }


__all__ = [
    "DEFAULT_PAID_EMBEDDING_MODEL",
    "DEFAULT_PAID_EMBEDDING_PROVIDER",
    "PAID_OPT_IN_ENV",
    "PaidEmbeddingApprovalRequired",
    "build_resolved_embedding_client",
    "estimate_local_embedding_cost_avoided",
    "resolve_embedding_config",
]
