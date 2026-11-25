from __future__ import annotations

from pathlib import Path

from conference_pipeline.publishing.byok_tools import js_text


def test_public_byok_cost_safety_doc_locks_budget_cap_contract() -> None:
    doc = Path("docs/operations/public-byok-cost-safety.md").read_text(encoding="utf-8")

    for expected in (
        "budget_cap_usd",
        "never auto-raised",
        "per-session",
        "global daily circuit-breaker",
        "halt",
        "operator approval",
        "does not deploy a maintainer-funded hosted-key LLM endpoint",
    ):
        assert expected in doc


def test_byok_js_entry_point_references_safety_doc() -> None:
    assert "docs/operations/public-byok-cost-safety.md" in js_text()

