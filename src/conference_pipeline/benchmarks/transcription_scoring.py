"""Phase 03-01: transcription-task scoring rubric for the benchmark harness.

Pure WER/CER scoring against a reference transcript. No external dependencies;
deterministic and reproducible offline.

Metrics
-------
wer : float
    Word Error Rate — Levenshtein edit distance over word sequences,
    normalised by the number of words in the reference.  Lower is better.
cer : float
    Character Error Rate — same distance computation over character
    sequences.  Lower is better.
quality : float
    Composite quality score derived from WER/CER: ``1.0 - min(wer, 1.0)``
    blended with ``1.0 - min(cer, 1.0)``.  Range [0, 1]; higher is better.
    Exact match → quality 1.0; completely wrong → quality 0.0.

Usage
-----
>>> from conference_pipeline.benchmarks.transcription_scoring import (
...     score_transcription_output,
... )
>>> scores = score_transcription_output(
...     reference_text="hello world",
...     candidate_text="hello world",
... )
>>> scores["quality"]
1.0
"""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize_words(text: str) -> list[str]:
    """Lower-cased word tokens from text."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


def _levenshtein(seq_a: list, seq_b: list) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n
    if n == 0:
        return m
    # Use two rolling rows to keep memory O(min(m, n)).
    if m < n:
        seq_a, seq_b = seq_b, seq_a
        m, n = n, m
    prev = list(range(n + 1))
    for i, a_el in enumerate(seq_a, start=1):
        curr = [i] + [0] * n
        for j, b_el in enumerate(seq_b, start=1):
            if a_el == b_el:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER (word-level Levenshtein / |reference words|)."""
    ref_words = _tokenize_words(reference)
    hyp_words = _tokenize_words(hypothesis)
    if not ref_words:
        return 0.0
    distance = _levenshtein(ref_words, hyp_words)
    return distance / len(ref_words)


def _char_error_rate(reference: str, hypothesis: str) -> float:
    """Compute CER (char-level Levenshtein / |reference chars after collapse|)."""
    ref_chars = list((reference or "").casefold())
    hyp_chars = list((hypothesis or "").casefold())
    if not ref_chars:
        return 0.0
    distance = _levenshtein(ref_chars, hyp_chars)
    return distance / len(ref_chars)


def score_transcription_output(
    reference_text: str,
    candidate_text: str,
) -> dict[str, float]:
    """Compute WER, CER, and composite quality for one transcription pair.

    Parameters
    ----------
    reference_text:
        Ground-truth transcript used as the reference.
    candidate_text:
        Hypothesis text produced by the local transcription model.

    Returns
    -------
    dict with keys:
        ``wer``     — Word Error Rate (lower is better; may exceed 1.0).
        ``cer``     — Character Error Rate (lower is better; may exceed 1.0).
        ``quality`` — Composite quality in [0, 1] (higher is better).
                      Exact match → 1.0; completely mismatched → 0.0.
    """
    wer = _word_error_rate(reference_text, candidate_text)
    cer = _char_error_rate(reference_text, candidate_text)
    # Quality: equal blend of WER-derived and CER-derived accuracy, clamped to [0, 1].
    wer_accuracy = max(0.0, 1.0 - min(wer, 1.0))
    cer_accuracy = max(0.0, 1.0 - min(cer, 1.0))
    quality = round((wer_accuracy + cer_accuracy) / 2.0, 4)
    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "quality": quality,
    }
