"""Shared deterministic checks for claims unsupported by abbreviated source notes."""
from __future__ import annotations

import re

# Missing evidence in a deck is not evidence of absence in a work.
_ABSENCE = re.compile(
    r"\b(?:neither|none(?:\s+of)?|no\s+(?:work|paper|source|study)|"
    r"(?:the\s+)?(?:work|paper|source|study|sources|papers|authors|works)\b[^.!?\n]{0,65}"
    r"(?:does(?:n't|\s+not)|do(?:n't|\s+not)|has\s+no|have\s+no|lacks?))"
    r"[^.!?\n]{0,130}\b(?:results?|numbers?|evaluations?|benchmarks?|scores?|failure\s+rates?|quantitative|effect\s+sizes?)\b"
    r"|\bno\s+(?:comparative\s+|quantitative\s+)?(?:scores|results|evaluations)\b[^.!?\n]{0,70}\b(?:works|papers|sources)\b"
    r"|\b(?:without|no)\s+(?:controlled\s+|comparable\s+|quantitative\s+)?(?:numbers|results|evaluations)\b",
    re.I,
)


def absence_claims(text: str) -> list[str]:
    """Return suspect source-absence assertions for repair or fail-closed lint."""
    patterns = (_ABSENCE, re.compile(r"\b(?:no\s+(?:control|method|work|paper|system)\b[^.!?\n]{0,90}\b(?:tested|evaluated|shown|demonstrated)|none\b[^.!?\n]{0,60}\b(?:shown|tested|demonstrated))[^.!?\n]*", re.I))
    return [match.group(0) for pattern in patterns for match in pattern.finditer(text)]


# Source records keep a capped author list, so a spoken list of four or more names
# closed with "and <Name>" reads as the complete authorship when it may be cut short.
_NAME = r"(?:[A-Z][a-zà-ÿ'’-]+|[A-Z]\.)(?:\s+(?:[A-Z][a-zà-ÿ'’-]+|[A-Z]\.|van|de|von|der|da))*"
_CLOSED_AUTHOR_LIST = re.compile(
    rf"\b{_NAME}(?:,\s+{_NAME}){{2,}},?\s+and\s+(?!colleagues|others|team|collaborators|co-?authors){_NAME}"
)


def closed_author_lists(text: str) -> list[str]:
    """Return spoken lists of four or more names presented as a complete author list."""
    return [match.group(0) for match in _CLOSED_AUTHOR_LIST.finditer(text)]


def source_record(row: dict) -> dict:
    """Keep evidence attached to its work; unknown kind never implies results."""
    kind = str(row.get('work_kind') or row.get('paper_type') or '').lower()
    if kind not in {'position', 'method', 'empirical', 'talk'}:
        title = str(row.get('title') or '')
        if re.search(r'\bposition(?:\s+paper)?\b', title + ' ' + str(row.get('contribution_note') or ''), re.I):
            kind = 'position'
        elif row.get('speakers') or row.get('source_type') == 'talk':
            kind = 'talk'
        else:
            kind = 'unknown'
    return {**{key: row.get(key) or ([] if key in {'authors', 'speakers'} else '')
               for key in ('label', 'source_id', 'title', 'authors', 'speakers', 'contribution_note')},
            'title': row.get('title') or (row.get('works') or [''])[0],
            'author_count': row.get('author_count') or len(row.get('authors') or []),
            'work_kind': kind}


def bound_source_records(binding: dict) -> list[dict]:
    return [{'slide_id': slide.get('slide_id'),
             'sources': [source_record(row) for row in slide.get('visible_references', []) if isinstance(row, dict)]}
            for slide in binding.get('slides', []) if isinstance(slide, dict)]
