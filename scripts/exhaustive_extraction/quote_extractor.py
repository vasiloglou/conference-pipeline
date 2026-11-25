"""
Quote extraction from anchor words.

This module handles extracting verbatim quotes from source text using
start/end anchor words provided by the extraction model.

The anchor-based approach guarantees quotes are exact substrings of
the source text, eliminating hallucinated or paraphrased quotes.
"""

import re
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class QuoteExtractionResult:
    """Result of quote extraction attempt."""
    success: bool
    verbatim_quote: Optional[str] = None
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    error: Optional[str] = None
    method: str = "anchor_match"  # anchor_match, fuzzy_match, llm_fallback


def normalize_text(text: str) -> str:
    """Normalize text for matching (lowercase, collapse whitespace)."""
    return " ".join(text.lower().split())


def find_anchor_in_text(anchor: str, text: str, start_from: int = 0) -> Optional[int]:
    """
    Find anchor phrase in text with fuzzy matching.

    Tries exact match first, then progressively relaxes matching.

    Args:
        anchor: Anchor phrase to find
        text: Source text to search
        start_from: Position to start searching from

    Returns:
        Start position if found, None otherwise
    """
    if not anchor or not text:
        return None

    # Normalize both for comparison
    anchor_norm = normalize_text(anchor)
    text_norm = normalize_text(text)

    # Try exact match first
    pos = text_norm.find(anchor_norm, start_from)
    if pos >= 0:
        return pos

    # Try without punctuation
    anchor_no_punct = re.sub(r'[^\w\s]', '', anchor_norm)
    text_no_punct = re.sub(r'[^\w\s]', '', text_norm)
    pos = text_no_punct.find(anchor_no_punct, start_from)
    if pos >= 0:
        return pos

    # Try matching just the first few words (in case model added extra)
    words = anchor_norm.split()
    if len(words) >= 3:
        # Try first 3 words
        short_anchor = " ".join(words[:3])
        pos = text_norm.find(short_anchor, start_from)
        if pos >= 0:
            return pos

        # Try last 3 words
        short_anchor = " ".join(words[-3:])
        pos = text_norm.find(short_anchor, start_from)
        if pos >= 0:
            return pos

    return None


def extract_quote_from_anchors(
    source_text: str,
    start_anchor: str,
    end_anchor: str,
    max_quote_length: int = 5000
) -> QuoteExtractionResult:
    """
    Extract verbatim quote from source text using start/end anchors.

    Uses non-greedy regex to find the SHORTEST valid quote between anchors.
    This handles cases where anchor phrases appear multiple times in the text.

    Args:
        source_text: Full transcript or abstract text
        start_anchor: First 5-8 words of the quote
        end_anchor: Last 5-8 words of the quote
        max_quote_length: Maximum allowed quote length in characters

    Returns:
        QuoteExtractionResult with extracted quote or error
    """
    if not source_text:
        return QuoteExtractionResult(
            success=False,
            error="Empty source text"
        )

    if not start_anchor or not end_anchor:
        return QuoteExtractionResult(
            success=False,
            error="Missing start or end anchor"
        )

    # Use non-greedy regex to find SHORTEST quote between anchors
    # This handles duplicate anchor phrases correctly
    verbatim_quote = extract_substring_preserving_format(
        source_text, start_anchor, end_anchor, max_length=max_quote_length
    )

    if verbatim_quote is None:
        # Try to determine why it failed for better error messages
        start_pos = find_anchor_in_text(start_anchor, source_text)
        if start_pos is None:
            return QuoteExtractionResult(
                success=False,
                error=f"Start anchor not found: '{start_anchor[:50]}...'"
            )

        end_pos = find_anchor_in_text(end_anchor, source_text, start_from=start_pos)
        if end_pos is None:
            return QuoteExtractionResult(
                success=False,
                error=f"End anchor not found: '{end_anchor[:50]}...'"
            )

        # Anchors found but regex failed - likely too long even with shortest match
        return QuoteExtractionResult(
            success=False,
            error=f"Could not extract quote (shortest match may exceed {max_quote_length} chars)"
        )

    # Validate length
    if len(verbatim_quote) > max_quote_length:
        return QuoteExtractionResult(
            success=False,
            error=f"Quote too long ({len(verbatim_quote)} chars > {max_quote_length} max)"
        )

    if len(verbatim_quote) < 10:
        return QuoteExtractionResult(
            success=False,
            error=f"Quote too short ({len(verbatim_quote)} chars)"
        )

    return QuoteExtractionResult(
        success=True,
        verbatim_quote=verbatim_quote,
        start_pos=0,  # Position tracking simplified since we use regex
        end_pos=len(verbatim_quote),
        method="anchor_match"
    )


def extract_substring_preserving_format(
    source_text: str,
    start_anchor: str,
    end_anchor: str,
    max_length: int = 5000
) -> Optional[str]:
    """
    Extract substring from source text preserving original formatting.

    Uses non-greedy regex to find the SHORTEST match between anchors.
    This handles cases where anchor phrases appear multiple times.

    Args:
        source_text: Original text with formatting
        start_anchor: Start anchor phrase
        end_anchor: End anchor phrase
        max_length: Maximum allowed length for the middle portion

    Returns:
        Extracted substring or None if not found
    """
    # Create regex pattern for flexible matching
    # Escape special regex characters in anchors
    start_escaped = re.escape(start_anchor.strip())
    end_escaped = re.escape(end_anchor.strip())

    # Make whitespace flexible
    start_pattern = re.sub(r'\\ +', r'\\s+', start_escaped)
    end_pattern = re.sub(r'\\ +', r'\\s+', end_escaped)

    # Build full pattern: start_anchor ... end_anchor
    # Use non-greedy match to get shortest valid quote
    # Limit middle portion to max_length to avoid catastrophic backtracking
    pattern = f"({start_pattern})(.{{0,{max_length}}}?)({end_pattern})"

    match = re.search(pattern, source_text, re.IGNORECASE | re.DOTALL)
    if match:
        # Return full match (start + middle + end)
        return match.group(0)

    # Try with more flexible whitespace
    source_collapsed = " ".join(source_text.split())
    match = re.search(pattern, source_collapsed, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0)

    return None


def build_anchor_fallback_prompt(
    source_text: str,
    start_anchor: str,
    end_anchor: str,
    summary: str
) -> str:
    """
    Build prompt for LLM fallback when anchor matching fails.

    Args:
        source_text: Full transcript/abstract
        start_anchor: Start anchor that failed to match
        end_anchor: End anchor that failed to match
        summary: Summary of what the quote should be about

    Returns:
        Prompt for LLM to help locate the quote
    """
    # Show more source text (up to 10K chars) so LLM can find quotes anywhere
    max_source_chars = 10000
    source_preview = source_text[:max_source_chars]
    truncated = len(source_text) > max_source_chars

    return f"""The following anchors failed to match exactly in the source text. Find the APPROXIMATE passage.

SOURCE TEXT:
{source_preview}{"..." if truncated else ""}

APPROXIMATE ANCHORS (may have transcription errors or slight variations):
- Start anchor (approximate): "{start_anchor}"
- End anchor (approximate): "{end_anchor}"

INTENDED CONTENT: {summary}

TASK: Find the passage in the source text that APPROXIMATELY matches the anchors and summary.
The anchors may have minor transcription errors, different punctuation, or slight wording variations.
Look for the passage that is CLOSEST to the intended meaning.

Return ONLY a JSON object with EXACT words copied from the source text above:
{{"corrected_start_anchor": "exact first 5-8 words from source", "corrected_end_anchor": "exact last 5-8 words from source"}}

If no matching passage exists, return: {{"error": "no_match"}}"""


async def extract_quote_with_llm_fallback(
    source_text: str,
    start_anchor: str,
    end_anchor: str,
    summary: str,
    llm_call_fn: callable
) -> QuoteExtractionResult:
    """
    Extract quote with LLM fallback for failed anchor matching.

    Args:
        source_text: Full transcript/abstract
        start_anchor: Start anchor from model
        end_anchor: End anchor from model
        summary: Summary of the content
        llm_call_fn: Async function to call LLM (takes prompt, returns dict)

    Returns:
        QuoteExtractionResult
    """
    # First try direct anchor matching
    result = extract_quote_from_anchors(source_text, start_anchor, end_anchor)
    if result.success:
        return result

    # Anchor matching failed - try LLM fallback
    prompt = build_anchor_fallback_prompt(source_text, start_anchor, end_anchor, summary)

    try:
        llm_response = await llm_call_fn(prompt)

        if "error" in llm_response:
            return QuoteExtractionResult(
                success=False,
                error=f"LLM fallback: {llm_response.get('error', 'unknown')}"
            )

        corrected_start = llm_response.get("corrected_start_anchor")
        corrected_end = llm_response.get("corrected_end_anchor")

        if corrected_start and corrected_end:
            # Try with corrected anchors
            result = extract_quote_from_anchors(source_text, corrected_start, corrected_end)
            if result.success:
                result.method = "llm_fallback"
                return result

        return QuoteExtractionResult(
            success=False,
            error="LLM fallback provided anchors that also failed to match"
        )

    except Exception as e:
        return QuoteExtractionResult(
            success=False,
            error=f"LLM fallback error: {str(e)}"
        )


def extract_quote_sync(
    source_text: str,
    start_anchor: str,
    end_anchor: str,
    summary: str,
    llm_call_fn: Optional[callable] = None
) -> QuoteExtractionResult:
    """
    Synchronous quote extraction with optional LLM fallback.

    Args:
        source_text: Full transcript/abstract
        start_anchor: Start anchor from model
        end_anchor: End anchor from model
        summary: Summary of the content
        llm_call_fn: Optional sync function to call LLM for fallback

    Returns:
        QuoteExtractionResult
    """
    # First try direct anchor matching
    result = extract_quote_from_anchors(source_text, start_anchor, end_anchor)
    if result.success:
        return result

    # If no LLM fallback provided, return the failure
    if llm_call_fn is None:
        return result

    # Try LLM fallback
    prompt = build_anchor_fallback_prompt(source_text, start_anchor, end_anchor, summary)

    try:
        llm_response = llm_call_fn(prompt)

        if "error" in llm_response:
            return QuoteExtractionResult(
                success=False,
                error=f"LLM fallback: {llm_response.get('error', 'unknown')}"
            )

        corrected_start = llm_response.get("corrected_start_anchor")
        corrected_end = llm_response.get("corrected_end_anchor")

        if corrected_start and corrected_end:
            # Try with corrected anchors
            result = extract_quote_from_anchors(source_text, corrected_start, corrected_end)
            if result.success:
                result.method = "llm_fallback"
                return result

        return QuoteExtractionResult(
            success=False,
            error="LLM fallback provided anchors that also failed to match"
        )

    except Exception as e:
        return QuoteExtractionResult(
            success=False,
            error=f"LLM fallback error: {str(e)}"
        )


def extract_quote_with_gpt52_fallback(
    source_text: str,
    start_anchor: str,
    end_anchor: str,
    summary: str,
    topic_name: Optional[str] = None,
    timestamp_start_sec: Optional[float] = None,
    timestamp_end_sec: Optional[float] = None,
    relevance_score: Optional[float] = None
) -> QuoteExtractionResult:
    """
    Extract quote with GPT-5.2 fallback for direct quote finding.

    First tries anchor-based extraction. If that fails, uses GPT-5.2 to
    directly find and return the verbatim quote text (not corrected anchors).

    Args:
        source_text: Full transcript/abstract
        start_anchor: Start anchor from model (may have transcription errors)
        end_anchor: End anchor from model (may have transcription errors)
        summary: Summary of the content (helps GPT-5.2 identify the passage)
        topic_name: Research topic name for semantic context (e.g., "LLM Scaling Laws")
        timestamp_start_sec: Start time in seconds to narrow search window
        timestamp_end_sec: End time in seconds to narrow search window
        relevance_score: Confidence score (0-1) from extraction model

    Returns:
        QuoteExtractionResult with extracted quote or error
    """
    # First try direct anchor matching
    result = extract_quote_from_anchors(source_text, start_anchor, end_anchor)
    if result.success:
        return result

    # Anchor matching failed - try GPT-5.2 direct quote extraction
    from .models import call_gpt52_for_quote

    quote = call_gpt52_for_quote(
        source_text=source_text,
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        summary=summary,
        topic_name=topic_name,
        timestamp_start_sec=timestamp_start_sec,
        timestamp_end_sec=timestamp_end_sec,
        relevance_score=relevance_score
    )

    if quote:
        return QuoteExtractionResult(
            success=True,
            verbatim_quote=quote,
            start_pos=0,
            end_pos=len(quote),
            method="gpt52_fallback"
        )

    return QuoteExtractionResult(
        success=False,
        error=f"Anchor matching failed ({result.error}), GPT-5.2 fallback also failed"
    )
