"""Tests for structured_video's _compact_text/_first_sentence truncation fixes.

Live discovery: rebuilding the ICML 2026 executive-summary presentation
canary, visual QC flagged multiple slides with text truncated mid-word/
mid-quote ("models \"as.", "...0 gap, 0 missing-evidence, 0...") despite
ample blank space in the panel below. Root cause was two compounding bugs:

1. first_sentence()/_first_sentence()'s regex required terminal punctuation
   to be followed directly by whitespace, so a sentence ending on a quoted
   phrase ('...outcome."') was skipped past in favor of the NEXT sentence's
   punctuation, producing a "first sentence" far longer than intended.
2. _compact_text() then hard-truncated that overly-long text at a small
   character limit (88/96), cutting mid-word/mid-quote and appending a fake
   trailing period regardless of whether that produced a real sentence.
"""

from __future__ import annotations

from conference_pipeline.presentation.structured_video import (
    _compact_text,
    _first_sentence,
    _shorten_caption_to_clause,
)


def test_first_sentence_stops_at_quote_closed_period():
    text = (
        'alignment "can deteriorate under benign customization and become '
        'difficult to diagnose." Crucially, "greater reasoning capability is '
        'no longer assumed to improve safety."'
    )
    result = _first_sentence(text)
    assert result == (
        'alignment "can deteriorate under benign customization and become '
        'difficult to diagnose."'
    )
    assert "Crucially" not in result


def test_compact_text_preserves_complete_sentence_over_limit():
    """A complete, correctly-punctuated sentence under the hard ceiling must
    not be mid-word truncated just because it exceeds the nominal limit."""
    text = (
        "The leadership agenda distills the synthesis into funding, gating, "
        "and governance moves that follow from the conference evidence."
    )
    assert len(text) > 88
    assert _compact_text(text, limit=88) == text


def test_compact_text_preserves_quote_closed_sentence_over_limit():
    text = (
        'the scaling efficiency foundations report marks the shift "from scaling '
        'by uniform expansion toward scaling by selective allocation and systems '
        'co-design."'
    )
    assert len(text) > 96
    assert _compact_text(text, limit=96) == text


def test_compact_text_truncates_genuine_outlier_beyond_hard_ceiling():
    """Real data point: this conference's synthesis occasionally produces a
    single grammatically-correct compound sentence spanning ~350+ chars (a
    quote-closed clause followed by another clause). Preserving it whole made
    the caption panel too tall to also show the heading/chips/takeaway
    (visual QC: text_clipped/crowded_layout). The hard ceiling intentionally
    caps at 300 so only this genuine minority of outliers falls back to
    limit-based truncation; ordinary complete sentences under the ceiling
    (see the sibling test above) are unaffected."""
    text = (
        'the safety alignment governance report delivers perhaps the most '
        'consequential leadership message: alignment "can deteriorate under '
        'benign customization, be bypassed through optimization or adversarial '
        'fine-tuning, fail differently across languages, and become difficult '
        'to diagnose when an agent conceals failure behind an apparently '
        'successful outcome."'
    )
    assert len(text) > 300
    result = _compact_text(text, limit=180)
    assert len(result) <= 182
    assert result != text


def test_compact_text_still_truncates_genuine_run_on_text():
    """Content with no terminal punctuation at all (a run-on/multi-clause
    blob) must still be truncated to the limit -- the fix only protects
    already-complete sentences, not arbitrarily long unpunctuated text."""
    text = (
        "this is a very long caption without any terminal punctuation that "
        "just keeps going and going and going well past the limit"
    )
    result = _compact_text(text, limit=88)
    assert len(result) <= 90
    assert result.endswith(".")


def test_compact_text_still_enforces_hard_ceiling_on_extreme_length():
    """Even a complete sentence gets truncated if it blows past the hard
    safety ceiling, so a pathological input can't produce an unbounded
    caption."""
    text = "This is one real sentence. " * 30 + "Done."
    result = _compact_text(text, limit=88)
    assert len(result) <= 90


def test_shorten_caption_to_clause_trims_long_outlier():
    """Live discovery: the narrow left rail can't legibly carry a ~290-char
    outlier compound sentence (it wraps to ~12 tiny lines -> vision QC flags
    illegible_text). It's trimmed to a complete-clause excerpt, ending
    cleanly with no ellipsis and no comma stranded before a closing quote."""
    caption = (
        'the social culture law equity report argues fairness must be examined '
        '"across the model lifecycle, including dataset distillation, '
        'representation learning, multi-class decision rules, model merging, and '
        'rank aggregation," and that "language is not a sufficient proxy for culture."'
    )
    result = _shorten_caption_to_clause(caption, max_chars=175)
    assert len(result) <= 175
    assert "..." not in result and "…" not in result
    assert not result.rstrip().endswith(",")
    assert not result.rstrip().endswith(',"')
    assert result in caption or result.rstrip('"') in caption


def test_shorten_caption_to_clause_keeps_short_caption_whole():
    caption = "Agents are a runtime-architecture problem, not a prompting problem."
    assert _shorten_caption_to_clause(caption, max_chars=175) == caption
