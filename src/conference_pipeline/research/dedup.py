"""Semantic dedup for retrieved research chunks.

Clusters near-duplicate chunks within source-family groups at a moderate
threshold, keeping the highest-scoring representative per cluster.
Cross-family evidence (transcript vs slide/OCR vs paper) is always preserved
so synthesis has corroborating evidence from all source types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conference_pipeline.research.researcher import RetrievedChunk

_DEFAULT_THRESHOLD = 0.82


def _source_family(chunk: Any) -> str:
    """Map chunk_strategy to a coarse source family for dedup grouping."""
    strategy = (getattr(chunk, "chunk_strategy", "") or "").lower()
    if "transcript" in strategy:
        return "transcript"
    if "slide" in strategy or "image" in strategy or "ocr" in strategy:
        return "slide_ocr"
    if "paper" in strategy:
        return "paper"
    return "generic"


def _word_jaccard(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity; pure Python, no external dependencies."""
    words_a = set((text_a or "").lower().split())
    words_b = set((text_b or "").lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    union = len(words_a | words_b)
    return len(words_a & words_b) / union if union > 0 else 0.0


@dataclass
class DedupDecision:
    """Result of one semantic dedup pass over a list of retrieved chunks."""

    kept_chunks: list
    removed_chunk_ids: list
    removed_source_families: list
    kept_chunk_ids: list
    kept_source_families: list
    dedup_policy: str
    threshold: float


def deduplicate_retrieved_chunks(
    chunks: list,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    enabled: bool = True,
) -> DedupDecision:
    """Deduplicate a list of retrieved chunks, preserving cross-family evidence.

    Within each source family (transcript, slide_ocr, paper, generic), chunks
    whose word-Jaccard similarity exceeds ``threshold`` are collapsed to the
    highest-scoring representative.  Chunks from *different* source families
    are never compared — similar evidence in a transcript and a paper both
    survive (they corroborate each other for synthesis).

    Args:
        chunks: Already exact-chunk-id-deduped list of RetrievedChunk objects.
        threshold: Word-Jaccard similarity above which two same-family chunks
            are considered near-duplicates.  Default 0.82 is "moderate".
        enabled: When False, all chunks pass through unchanged.

    Returns:
        DedupDecision recording kept/removed chunk IDs and source families.
    """
    if not chunks:
        return DedupDecision(
            kept_chunks=[],
            removed_chunk_ids=[],
            removed_source_families=[],
            kept_chunk_ids=[],
            kept_source_families=[],
            dedup_policy="passthrough",
            threshold=threshold,
        )

    if not enabled:
        kept_ids = [c.chunk_id for c in chunks]
        families = [_source_family(c) for c in chunks]
        return DedupDecision(
            kept_chunks=list(chunks),
            removed_chunk_ids=[],
            removed_source_families=[],
            kept_chunk_ids=kept_ids,
            kept_source_families=families,
            dedup_policy="disabled",
            threshold=threshold,
        )

    # Group by source family; record each chunk's family for receipt.
    family_groups: dict[str, list] = {}
    chunk_family: dict[str, str] = {}
    for chunk in chunks:
        fam = _source_family(chunk)
        chunk_family[chunk.chunk_id] = fam
        family_groups.setdefault(fam, []).append(chunk)

    removed_ids: list[str] = []
    removed_families: list[str] = []
    kept_id_set: set[str] = set()

    for family, group in family_groups.items():
        # Greedy: highest-scoring chunk is always a representative.
        sorted_group = sorted(group, key=lambda c: float(getattr(c, "score", 0.0)), reverse=True)
        representative_texts: list[str] = []
        for chunk in sorted_group:
            is_near_dup = any(
                _word_jaccard(chunk.text, rep_text) >= threshold
                for rep_text in representative_texts
            )
            if is_near_dup:
                removed_ids.append(chunk.chunk_id)
                removed_families.append(family)
            else:
                representative_texts.append(chunk.text)
                kept_id_set.add(chunk.chunk_id)

    # Preserve original input ordering for kept chunks.
    kept_ordered = [c for c in chunks if c.chunk_id in kept_id_set]
    kept_ids = [c.chunk_id for c in kept_ordered]
    kept_families = [chunk_family[cid] for cid in kept_ids]

    return DedupDecision(
        kept_chunks=kept_ordered,
        removed_chunk_ids=removed_ids,
        removed_source_families=removed_families,
        kept_chunk_ids=kept_ids,
        kept_source_families=kept_families,
        dedup_policy="semantic",
        threshold=threshold,
    )


__all__ = ["DedupDecision", "deduplicate_retrieved_chunks"]
