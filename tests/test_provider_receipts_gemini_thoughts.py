"""Gemini bills thinking tokens as output (v6.1 Phase 3 canary, 2026-11-18)."""

from types import SimpleNamespace

from conference_pipeline import provider_receipts as pr


def test_gemini_output_tokens_add_thinking_to_visible_tokens():
    assert pr.gemini_output_tokens(SimpleNamespace(candidates_token_count=317, thoughts_token_count=7683)) == 8000
    assert pr.gemini_output_tokens(SimpleNamespace(candidates_token_count=3)) == 3
    assert pr.gemini_output_tokens(SimpleNamespace()) is None


def test_receipt_usage_counts_gemini_thinking():
    response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=2,
                                                              thoughts_token_count=40, total_token_count=52))
    usage = pr._extract_usage(response, provider="gemini")
    assert (usage["input_tokens"], usage["output_tokens"]) == (10, 42)
