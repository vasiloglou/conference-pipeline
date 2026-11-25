"""WR-03 regression test: podcast._split_text must split on sentence boundaries.

Offline — no providers, no API key required.
Calls _split_text as an unbound method via a throwaway SimpleNamespace (it never
touches self).

The bug (current code):
  re.split(r"(?<=[.!?])\\s+", text)
  In a raw string, \\s is a literal backslash + s — not the \\s whitespace class.
  The regex therefore never matches, so re.split returns the whole string as one element
  and all chunking falls through to the hard-slice (mid-word) fallback.

After fix: single backslash \\s+ correctly splits on sentence boundaries.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from conference_pipeline import podcast
from conference_pipeline.podcast import AnthropicClient, PodcastGenerator

# Bind _split_text as a plain function (self is never used in its body).
_split_text = PodcastGenerator._split_text
_self = SimpleNamespace()


def test_anthropic_podcast_client_uses_messages_without_temperature(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_messages_create(client, **kwargs):  # noqa: ANN001
        del client
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text='{"segments": []}')])

    monkeypatch.setattr(podcast, "anthropic_messages_create", fake_messages_create)
    client = AnthropicClient.__new__(AnthropicClient)
    client.client = object()

    result = client.chat(
        "claude-opus-4-8",
        [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Prompt"},
        ],
        0.4,
    )

    assert result == '{"segments": []}'
    assert calls[0]["model"] == "claude-opus-4-8"
    assert "temperature" not in calls[0]


def test_split_text_splits_on_sentence_boundaries() -> None:
    """Multi-sentence text longer than max_chars must split on sentence boundaries.

    Each individual sentence is <= max_chars (40) so a correct sentence-aware splitter
    yields exactly one chunk per sentence.  We verify:
    (a) more than one chunk is returned (proving a split occurred)
    (b) every chunk ends with a sentence terminator (., !, or ?) — proving splits
        happened at sentence boundaries, NOT mid-word.

    MUST fail on current code: the broken regex returns the whole string as one element,
    which then falls through to the hard-slice fallback, producing chunks that do NOT end
    on sentence terminators.
    """
    # Four short sentences, each <= 40 chars — total > 40 chars.
    sentences = [
        "The model trains on clean data.",
        "Results improve each epoch!",
        "Losses plateau after round five.",
        "Retrain with more examples?",
    ]
    text = " ".join(sentences)
    assert len(text) > 40, "test setup: total text must exceed max_chars"

    max_chars = 40
    chunks = _split_text(_self, text, max_chars)

    # (a) More than one chunk must be returned.
    assert len(chunks) > 1, (
        f"expected multiple sentence-boundary chunks, got {len(chunks)}: {chunks!r}\n"
        "Likely cause: regex literal \\\\s never splits → whole string falls to hard-slice fallback"
    )

    # (b) Each chunk (stripped) must end with a sentence terminator — confirming that
    #     splits happened at sentence boundaries, not mid-word.
    for chunk in chunks:
        assert chunk.strip()[-1] in {".", "!", "?"}, (
            f"chunk does not end on a sentence terminator: {chunk!r}\n"
            "A hard-slice fallback would produce mid-word endings."
        )


def test_split_text_returns_single_chunk_when_text_fits() -> None:
    """Short text (≤ max_chars) must be returned as a single-element list."""
    short = "Fits in one chunk."
    assert len(short) <= 40
    chunks = _split_text(_self, short, max_chars=40)
    assert chunks == [short], f"expected [{short!r}], got {chunks!r}"
