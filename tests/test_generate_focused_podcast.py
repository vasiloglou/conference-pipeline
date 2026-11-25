"""Regression tests for the focused-podcast script generation.

Covers two CVPR2026 Phase 9 defects:

1. Plan 09-04's ``max_tokens`` defect: the prior formula
   (``min(16000, max(4096, target_words * 2))``) clamped to exactly 4096
   tokens for any ``target_minutes`` below ~15.75, silently truncating the
   JSON response for this platform's locked 12-minute podcast convention
   every time. See
   ``.planning/workstreams/cvpr-2026/phases/09-rendered-podcast-production/deferred-items.md``
   for the full incident record.
2. The Plan 09-04 fact-check remediation (this file's addendum): the
   rendered script overstated its own source provenance by claiming the
   source material's "Source Anchors" section enumerates every topic by
   number, when in fact that section lists only a subset of topics -- a
   conflation with the separate, true "all N topics well-covered" claim
   made elsewhere in the same source material. See
   ``.planning/workstreams/cvpr-2026/phases/09-rendered-podcast-production/09-04-ADDENDUM-fact-check-remediation.md``.

``scripts/`` has no ``__init__.py`` (it is not a package), so this module
is loaded via ``importlib`` -- the same pattern
``tests/test_cvpr2026_production_runner.py`` uses for
``scripts/run_cvpr2026_production.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_focused_podcast.py"


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "generate_focused_podcast_under_test", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "target_minutes,expected_max_tokens",
    [
        # The previously-broken case: 12 minutes used to clamp to exactly
        # 4096 tokens (min(16000, max(4096, 1560 * 2)) == 4096), which
        # truncated every real live response (CVPR2026 09-04).
        (12, 8192),
        # Below the new 8192 floor entirely -> floor applies.
        (1, 8192),
        # Right at the boundary where target_words * 4 first exceeds the
        # 8192 floor: target_minutes=16 -> target_words=2080,
        # target_words*4=8320 > 8192.
        (16, 8320),
        # Large target_minutes -> the 16000 ceiling applies.
        (60, 16000),
    ],
)
def test_compute_script_max_tokens(target_minutes: int, expected_max_tokens: int) -> None:
    module = _load_module()
    assert module.compute_script_max_tokens(target_minutes) == expected_max_tokens


def test_compute_script_max_tokens_never_below_prior_floor_for_short_durations() -> None:
    """The new floor (8192) is strictly higher than the old one (4096).

    Guards against silently regressing back to a too-low ceiling for any
    short target duration, not just the exact 12-minute case.
    """
    module = _load_module()
    for target_minutes in (1, 5, 10, 12, 15):
        assert module.compute_script_max_tokens(target_minutes) >= 8192


def test_generate_focused_script_prompt_warns_against_source_anchor_conflation() -> None:
    """Regression test for the Plan 09-04 fact-check remediation.

    The rendered CVPR2026 script claimed "the Source Anchors span topic
    reports one through twelve," but the source material's own "Source
    Anchors" section lists only a subset of topics -- a conflation with the
    separate, true "all N topics well-covered" claim made elsewhere in the
    same source. This test asserts the constructed prompt contains an
    explicit instruction guarding against that conflation, so the model is
    told not to overstate the Source Anchors list's coverage.

    See
    ``.planning/workstreams/cvpr-2026/phases/09-rendered-podcast-production/09-04-ADDENDUM-fact-check-remediation.md``
    for the full incident record.
    """
    import json
    from types import SimpleNamespace

    module = _load_module()
    captured: dict[str, str] = {}

    def fake_anthropic_messages_create(
        client, *, caller, model, max_tokens, system, messages
    ):
        captured["prompt"] = messages[0]["content"]
        fake_script = {
            "title": "t",
            "summary": "s",
            "segments": [{"speaker": "Maya", "text": "word " * 200}],
            "show_notes": "",
        }
        return SimpleNamespace(content=[{"text": json.dumps(fake_script)}])

    module.anthropic_messages_create = fake_anthropic_messages_create

    config = module.ScriptConfig(
        title="Test Podcast",
        host_a="Maya",
        host_b="Leo",
        two_hosts=True,
        target_minutes=12,
        max_segment_chars=1000,
        temperature=0.4,
    )
    facts = [{"tier": 1, "score": 0.9, "text": "A high-priority fact."}]

    script = module.generate_focused_script(
        client=object(),
        provider="anthropic",
        model="claude-opus-4-8",
        source_text="## Source Anchors\n\nTopic reports:\n- one\n- two\n",
        selected_facts=facts,
        config=config,
    )

    assert script is not None
    prompt = captured["prompt"]
    assert "Source Anchors" in prompt
    assert (
        "do not conflate it with the specific anchors/citations actually listed"
        in prompt
    )
