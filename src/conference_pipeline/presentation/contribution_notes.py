"""Shared attribution guards for abbreviated contribution notes."""
import re

from conference_pipeline.presentation.display_text import strip_markdown

_LEDGER_SECTION_NOTE_RE = re.compile(r"^\s*(?:(?:[-*+]|\d+[.)])\s*)?\*\*[^*]+:\*\*")

_NOTE_STOPWORDS = {"with", "from", "that", "this", "their", "about", "using", "based", "toward", "towards", "into", "while", "where", "which"}


def _note_matches_work(note: str, title: str) -> bool:
    """True when the contribution note plausibly describes this work: the note's
    lead shares at least one distinctive content word with the work's own title.
    Aggregated notes are cluster claims, and a work cited inside another topic's
    cluster otherwise inherits that cluster's claim as its 'detail' — a visible
    misattribution on the bookend slides."""
    words = {
        chunk.lower().strip(".,;:'\"()")
        for chunk in re.split(r"[\s/–—-]+", title)
        if len(chunk.strip(".,;:'\"()")) >= 5
    } - _NOTE_STOPWORDS
    # Judge by the note's FIRST SENTENCE only: aggregated notes lead with the
    # cluster claim ("RLVR is central but brittle."), and topic-wide words like
    # "reasoning" appear later in almost every note regardless of the work.
    lead = re.split(r"(?<=[.!?])\s+", strip_markdown(note).strip(), maxsplit=1)[0].lower()
    return any(word in lead for word in words)

