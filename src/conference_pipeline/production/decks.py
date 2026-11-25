"""Deck scope derived from ``analysis_manifest.json`` plus the executive decks.

The analysis manifest is the single source of truth for which topic decks a
run produces; the runner supplies the executive deck ids through the context.
Every stage that enumerates decks (presentation, podcast, release, proof)
derives its list here so none of them can drift from the others.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def presentation_topics_from_analysis_manifest(analysis_manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Derive the topic decks from the analysis manifest's ``topics`` list.

    Each result row is ``{"id", "title", "markdown_path", "slide_brief_path"}``.
    A manifest with no topics, or a topic without id/title, is an error the
    caller records as a blocker: silently producing zero decks is worse.
    """

    topics = analysis_manifest.get("topics")
    if not isinstance(topics, list) or not topics:
        raise RuntimeError("presentation_analysis_manifest_has_no_topics")
    result: list[dict[str, str]] = []
    for index, raw_topic in enumerate(topics, start=1):
        if not isinstance(raw_topic, dict):
            raise RuntimeError(f"presentation_analysis_manifest_topic_{index}_invalid")
        topic_id = str(raw_topic.get("id") or raw_topic.get("topic_id") or raw_topic.get("slug") or "").strip()
        title = str(raw_topic.get("title") or topic_id).strip()
        if not topic_id or not title:
            raise RuntimeError(f"presentation_analysis_manifest_topic_{index}_missing_id_or_title")
        result.append(
            {
                "id": topic_id,
                "title": title,
                "markdown_path": str(raw_topic.get("markdown_path") or f"analysis/{topic_id}.md"),
                "slide_brief_path": str(
                    raw_topic.get("slide_brief_path") or f"analysis/slide_briefs/{topic_id}.md"
                ),
            }
        )
    return result


def presentation_deck_scope(
    analysis_manifest: dict[str, Any],
    executive_decks: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    """The full in-scope deck list: every topic deck, then the executive decks.

    Each entry is ``{"id", "title", "kind"}`` with ``kind`` in
    ``{"topic", "executive"}``.
    """

    scope: list[dict[str, str]] = [
        {"id": topic["id"], "title": topic["title"], "kind": "topic"}
        for topic in presentation_topics_from_analysis_manifest(analysis_manifest)
    ]
    scope.extend({"id": deck_id, "title": deck_title, "kind": "executive"} for deck_id, deck_title in executive_decks)
    return scope
