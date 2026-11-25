"""Prompts for reference normalization using LLMs.

These prompts are used to clean up OCR errors and extract structured
bibliographic information from noisy text.
"""

from textwrap import dedent


REFERENCE_NORMALIZATION_PROMPT = dedent("""
    You are a bibliographic reference normalizer. Given messy or incomplete
    reference information extracted from OCR or speech, normalize it into
    structured JSON.

    Input reference:
    {raw_text}

    Additional context (may be present):
    - Detected arXiv ID: {arxiv_id}
    - Detected DOI: {doi}
    - Detected URL: {url}
    - Authors (if any): {authors}

    Instructions:
    1. Clean up any OCR errors in the text
    2. Extract the paper title if present
    3. Extract author names if present
    4. Identify the publication venue if mentioned
    5. Extract the year if present
    6. Normalize arXiv IDs to standard format (YYMM.NNNNN)
    7. Clean up DOIs to standard format (10.XXXX/...)

    Respond with ONLY a JSON object in this format:
    {{
        "normalized_title": "Paper Title Here (or null if not found)",
        "normalized_authors": ["Author One", "Author Two"],
        "arxiv_id": "YYMM.NNNNN (or null)",
        "doi": "10.XXXX/... (or null)",
        "venue": "Conference/Journal name (or null)",
        "year": 2024,
        "confidence": 0.8
    }}

    The confidence score should reflect how certain you are about the
    extracted information (0.0-1.0).
""").strip()


def format_normalization_prompt(
    raw_text: str,
    arxiv_id: str | None = None,
    doi: str | None = None,
    url: str | None = None,
    authors: list[str] | None = None,
    extra_context: str | None = None,
) -> str:
    """Format the normalization prompt with the given inputs.

    Args:
        raw_text: The raw reference text to normalize.
        arxiv_id: Detected arXiv ID, if any.
        doi: Detected DOI, if any.
        url: Detected URL, if any.
        authors: List of detected author names, if any.
        extra_context: Additional context from slide or surrounding text.

    Returns:
        Formatted prompt string ready for LLM.
    """
    prompt = REFERENCE_NORMALIZATION_PROMPT.format(
        raw_text=raw_text,
        arxiv_id=arxiv_id or "None",
        doi=doi or "None",
        url=url or "None",
        authors=", ".join(authors) if authors else "None",
    )

    if extra_context:
        prompt += f"\n\nAdditional context from slide:\n{extra_context}"

    return prompt
