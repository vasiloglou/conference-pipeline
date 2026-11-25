"""Counterfactual paid-cost helper for local substitutions — LOCAL-04 / D-13..D-16.

``estimate_cost_avoided`` computes the cost that would have been incurred had
the same workload been routed to the specified paid fallback provider/model,
using the Phase 02 pricing catalog and task_specs.yaml for quantity metadata.

D-13: counterfactual basis = pricing_catalog estimate for the same workload.
D-16: when no catalog price or quantity can be resolved, ``cost_avoided_usd``
      is ``None`` and ``cost_avoided_status`` is ``"unknown"`` with
      ``needs_pricing_backfill=True``. Savings are NEVER fabricated as $0.
"""
from __future__ import annotations

from typing import Any

from conference_pipeline.costs.estimator import PricingCatalog, load_pricing_catalog, load_task_specs


def estimate_cost_avoided(
    task_id: str,
    *,
    paid_provider: str,
    paid_model: str,
    quantity: float | None = None,
    catalog: PricingCatalog | None = None,
    task_specs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate the cost avoided by using a local model instead of a paid one.

    Parameters
    ----------
    task_id:
        Model-governance task ID (e.g. ``"asr_transcription"``).  Used to
        look up the workload quantity from ``task_specs.yaml`` when
        ``quantity`` is not supplied explicitly.
    paid_provider:
        Provider name for the paid counterfactual (e.g. ``"openai"``).
    paid_model:
        Model name for the paid counterfactual (e.g. ``"text-embedding-3-small"``).
    quantity:
        Optional explicit quantity (token count, minutes, etc.).  When
        ``None``, the helper reads ``quantity_source`` from ``task_specs``
        to determine a unit count.  If no quantity can be resolved,
        ``cost_avoided_status`` is ``"unknown"``.
    catalog:
        Pre-loaded :class:`PricingCatalog`.  Defaults to the package default.
    task_specs:
        Pre-loaded task_specs dict.  Defaults to the package default.

    Returns
    -------
    dict with keys:
        ``cost_avoided_usd``          : float | None (None when unknown)
        ``cost_avoided_status``       : ``"priced"`` | ``"unknown"``
        ``paid_counterfactual_provider``: the paid provider string
        ``paid_counterfactual_model`` : the paid model string
        ``needs_pricing_backfill``    : True when status is ``"unknown"``
    """
    _catalog = catalog or load_pricing_catalog()
    _specs = task_specs or load_task_specs()

    # Locate the task spec for this task_id.
    task_spec: dict[str, Any] | None = None
    for task in _specs.get("tasks", []):
        if task.get("task_id") == task_id:
            task_spec = task
            break

    # Resolve the quantity to price.
    effective_quantity = quantity
    pricing_unit: str = "tokens"  # default

    if task_spec is not None:
        pricing_unit = str(task_spec.get("pricing_unit", "tokens"))
        if effective_quantity is None:
            quantity_source = str(task_spec.get("quantity_source", "fixed"))
            # For fixed/unknown sources we cannot resolve a quantity at the
            # per-substitution level without workspace context.  Mark unknown.
            if quantity_source == "fixed":
                effective_quantity = 1.0
            # All other quantity_source values (audio_minutes, text_tokens, etc.)
            # require workspace scanning which is beyond a per-call helper scope.
            # Mark unknown — the rollup can aggregate from actual receipt fields.

    if effective_quantity is None:
        return _unknown_result(paid_provider, paid_model)

    # Look up the paid counterfactual price in the catalog.
    try:
        resolved_provider, resolved_model, model_data = _catalog.resolve_model(
            paid_model, paid_provider
        )
    except KeyError:
        return _unknown_result(paid_provider, paid_model)

    # Compute the cost estimate for the resolved quantity.
    try:
        cost_usd = _price_quantity(
            _catalog,
            resolved_provider,
            resolved_model,
            model_data,
            pricing_unit,
            effective_quantity,
        )
    except Exception:  # noqa: BLE001 — pricing errors must not bubble up
        return _unknown_result(paid_provider, paid_model)

    if cost_usd is None:
        return _unknown_result(paid_provider, paid_model)

    return {
        "cost_avoided_usd": round(cost_usd, 8),
        "cost_avoided_status": "priced",
        "paid_counterfactual_provider": paid_provider,
        "paid_counterfactual_model": paid_model,
        "needs_pricing_backfill": False,
    }


def _unknown_result(paid_provider: str, paid_model: str) -> dict[str, Any]:
    """Return the D-16 unknown result — never fabricate savings."""
    return {
        "cost_avoided_usd": None,
        "cost_avoided_status": "unknown",
        "paid_counterfactual_provider": paid_provider,
        "paid_counterfactual_model": paid_model,
        "needs_pricing_backfill": True,
    }


def _price_quantity(
    catalog: PricingCatalog,
    provider: str,
    model: str,
    model_data: dict[str, Any],
    pricing_unit: str,
    quantity: float,
) -> float | None:
    """Price a quantity using the catalog for the given pricing_unit."""
    if pricing_unit in {"none", "fixed"}:
        cost = float(model_data.get("fixed_usd", 0.0))
        return cost

    if pricing_unit == "audio_minute":
        cost, _, _ = catalog.estimate_audio_minutes(provider, model, quantity)
        return cost

    if pricing_unit == "credits":
        cost, _, _ = catalog.estimate_credits(provider, model, quantity)
        return cost

    if pricing_unit in {"tts_minute_estimate", "tts_minute"}:
        cost, _, _ = catalog.estimate_tts_minutes(provider, model, quantity)
        return cost

    # Token-based pricing: treat quantity as input_tokens (no output for embed),
    # or as a combined token count split 50/50 for LLM tasks.
    if pricing_unit in {"tokens", "embedding_tokens", "text_tokens",
                        "openai_image_tokens", "pdf_pages_as_images",
                        "exhaustive_extraction_tokens",
                        "exhaustive_extraction_failover_tokens",
                        "topic_count", "presentation_targets"}:
        # WR-05: int() truncates toward zero, systematically under-reporting
        # avoided cost for fractional token quantities (e.g. chars/4) and pricing
        # a positive 0<quantity<1 as exactly 0 tokens -> a real-looking $0 saving.
        # Round, and never let a positive quantity collapse to 0 tokens.
        input_tokens = max(1, round(quantity)) if quantity > 0 else 0
        output_tokens = 0
        cost, _, _ = catalog.estimate_tokens(provider, model, input_tokens, output_tokens)
        return cost

    return None


__all__ = ["estimate_cost_avoided"]
