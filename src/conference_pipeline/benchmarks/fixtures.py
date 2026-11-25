"""Fixture input samples keyed by task_family for dry-run benchmarking."""
from __future__ import annotations

BENCHMARK_FIXTURES: dict[str, dict] = {
    "transcription": {
        "audio_url": "fixture://mock-session-audio.wav",
        "language": "en",
    },
    "ocr": {
        "image_path": "fixture://mock-slide.png",
        "page": 1,
    },
    "analysis": {
        "text": "A brief abstract about machine learning...",
        "max_length": 500,
    },
    "podcast": {
        "topic": "fixture topic",
        "facts": ["fact 1", "fact 2"],
    },
    "presentation": {
        "title": "fixture presentation",
        "outline": ["slide 1", "slide 2"],
    },
    "ranking": {
        "papers": ["paper A", "paper B"],
        "criterion": "reference frequency",
    },
    "research": {
        "query": "What are recent advances in attention mechanisms?",
        "depth": "standard",
    },
    "qa": {
        "claim": "The paper uses transformer architecture.",
        "source_text": "We use transformer architecture.",
    },
    "vision": {
        "image_path": "fixture://mock-figure.png",
        "task": "caption",
    },
    "synthesis": {
        "topics": ["topic A", "topic B"],
        "style": "executive",
    },
    "default": {
        "input": "fixture://mock-input",
        "task": "generic",
    },
}
