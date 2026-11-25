"""Request-shape facts about model families, decided by generation, not by name.

Six call sites decided "does this OpenAI model take `max_completion_tokens`
and reject `temperature`?" with `model.startswith("gpt-5")`. That was true for
exactly one generation: the 2026-09-24 catalog upgrade to GPT-6 would have
sent every GPT-6 request the legacy `max_tokens` + `temperature` shape. The
rule is a property of the generation (GPT-5 and later, and the o-series), so it
is written once, here, against the version number.
"""

from __future__ import annotations

import re

_GPT_GENERATION_RE = re.compile(r"^gpt-(\d+)")
_O_SERIES_RE = re.compile(r"^o\d")


def uses_openai_reasoning_params(model: str) -> bool:
    """True for OpenAI reasoning-era models: GPT-5 or later, and o1/o3/o4-style.

    These take `max_completion_tokens` and do not accept a custom `temperature`.
    A routed id (`openrouter/openai/<model>`) is judged by its model part.
    """

    name = str(model or "").strip().lower().rsplit("/", 1)[-1]
    generation = _GPT_GENERATION_RE.match(name)
    if generation:
        return int(generation.group(1)) >= 5
    return bool(_O_SERIES_RE.match(name))


_CLAUDE_RE = re.compile(r"claude-(opus|sonnet|haiku|fable)-(\d+)(?:[-.](\d+))?")
# The first generation of each tier that rejects a custom `temperature`.
# Haiku 5.5 answered HTTP 400 "`temperature` is deprecated for this model" (v6.1 Phase 5 pilot,
# 2026-11-21); Haiku 4.5 still takes it.
_CLAUDE_NO_TEMPERATURE_FROM = {"opus": (4, 7), "sonnet": (4, 6), "haiku": (5, 5),
                               "fable": (0, 0)}  # every Fable (2026-11-18)


def anthropic_rejects_temperature(model: str) -> bool:
    """True for Claude generations that do not take a custom `temperature`:
    Opus 4.7+ and Sonnet 4.6+ (and every later major version). A name list
    here missed claude-opus-5-5 on the day the catalog moved to it."""

    match = _CLAUDE_RE.search(str(model or "").lower())
    if not match:
        return False
    tier, major, minor = match.group(1), int(match.group(2)), int(match.group(3) or 0)
    floor = _CLAUDE_NO_TEMPERATURE_FROM.get(tier)
    # A dated snapshot suffix (claude-haiku-4-5-20251001) is not a minor version.
    if minor > 99:
        minor = 0
    return floor is not None and (major, minor) >= floor
