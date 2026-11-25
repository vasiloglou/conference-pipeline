"""v1.61 Phase 69 — vision-task scoring rubric for the benchmark harness.

The scoring rubric is intentionally simple and deterministic:

* ``text_recall``: token-overlap recall of expected text against model output.
* ``figure_caption_similarity``: Jaccard similarity over the n-gram set of
  expected vs. predicted captions.
* ``chart_summary_similarity``: same Jaccard rubric applied to chart-summary
  expected text (axes, units, key trends).
* ``parse_success``: 1.0 if the model output parsed into a
  :class:`SlideAnalysisResult` (or non-empty OCR text); 0.0 otherwise.

This file does NOT depend on any external embedding / sentence-transformer
package — receipts must be reproducible offline. Embedding-based similarity
is an upgrade path for later milestones once an offline embedding model is
on the local lane.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class VisionFixtureGroundTruth:
    """Ground-truth annotations for one vision-task fixture.

    The fields are optional so a single dataclass can carry annotations for
    ``vision.slide_analysis``, ``vision.figure_caption``, ``vision.chart_summary``,
    and ``ocr.slide_image_parse`` fixtures.
    """

    expected_text: str = ""
    expected_figure_captions: tuple[str, ...] = ()
    expected_chart_summary: str = ""
    expected_salient_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisionScores:
    """Per-fixture scores for one model-task pair."""

    text_recall: float
    figure_caption_similarity: float
    chart_summary_similarity: float
    parse_success: float

    def overall(self) -> float:
        # Equal-weighted blend; configurable thresholds live on the promotion CLI.
        return (
            self.text_recall
            + self.figure_caption_similarity
            + self.chart_summary_similarity
            + self.parse_success
        ) / 4.0

    def to_dict(self) -> dict:
        return {
            "text_recall": self.text_recall,
            "figure_caption_similarity": self.figure_caption_similarity,
            "chart_summary_similarity": self.chart_summary_similarity,
            "parse_success": self.parse_success,
            "overall": self.overall(),
        }


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


def _ngrams(tokens: Iterable[str], n: int = 3) -> set[tuple[str, ...]]:
    tokens = list(tokens)
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def text_recall(expected: str, predicted: str) -> float:
    """Token-overlap recall: fraction of expected tokens present in prediction."""
    expected_tokens = set(_tokenize(expected))
    if not expected_tokens:
        return 1.0
    predicted_tokens = set(_tokenize(predicted))
    return len(expected_tokens & predicted_tokens) / len(expected_tokens)


def jaccard_ngram_similarity(expected: str, predicted: str, *, n: int = 3) -> float:
    """Jaccard over n-grams; ``n`` defaults to trigrams."""
    e = _ngrams(_tokenize(expected), n=n)
    p = _ngrams(_tokenize(predicted), n=n)
    if not e and not p:
        return 1.0
    if not e or not p:
        return 0.0
    return len(e & p) / len(e | p)


def figure_caption_similarity(
    expected_captions: Iterable[str],
    predicted_captions: Iterable[str],
) -> float:
    """Best-match average Jaccard between expected and predicted figure captions."""
    expected_list = [c for c in expected_captions if c.strip()]
    predicted_list = [c for c in predicted_captions if c.strip()]
    if not expected_list:
        return 1.0
    if not predicted_list:
        return 0.0
    scores: list[float] = []
    for expected in expected_list:
        scores.append(
            max(
                jaccard_ngram_similarity(expected, predicted, n=2)
                for predicted in predicted_list
            )
        )
    return sum(scores) / len(scores)


def score_vision_output(
    *,
    task_id: str,
    ground_truth: VisionFixtureGroundTruth,
    raw_output: str,
    parsed_ok: bool,
    figure_captions: Iterable[str] = (),
    chart_summary: str = "",
) -> VisionScores:
    """Compute a :class:`VisionScores` for one fixture run.

    The four sub-scores are populated based on which task is being scored:

    * ``vision.slide_analysis``: text_recall on body_text + claims;
      figure_caption_similarity on figures; chart_summary_similarity on
      chart_summary; parse_success from ``parsed_ok``.
    * ``vision.figure_caption``: figure_caption_similarity dominates;
      text_recall is computed against expected_text but weighted equally.
    * ``vision.chart_summary`` and ``presentation.visual_qc``:
      chart_summary_similarity dominates when chart ground truth is present.
    * ``ocr.slide_image_parse``: text_recall dominates; other dimensions
      default to 1.0 when no ground truth is provided.
    """
    text_score = text_recall(ground_truth.expected_text, raw_output)
    if ground_truth.expected_salient_claims:
        # Add salient-claim recall into the text score (averaged).
        claim_scores = [
            text_recall(claim, raw_output)
            for claim in ground_truth.expected_salient_claims
        ]
        text_score = (text_score + sum(claim_scores) / len(claim_scores)) / 2.0

    figure_score = figure_caption_similarity(
        ground_truth.expected_figure_captions,
        figure_captions,
    )
    chart_score = jaccard_ngram_similarity(
        ground_truth.expected_chart_summary, chart_summary
    )
    return VisionScores(
        text_recall=round(text_score, 4),
        figure_caption_similarity=round(figure_score, 4),
        chart_summary_similarity=round(chart_score, 4),
        parse_success=1.0 if parsed_ok else 0.0,
    )
