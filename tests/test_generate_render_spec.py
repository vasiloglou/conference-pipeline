"""Tests for generate_render_spec.py's _compact_text truncation fix.

See tests/test_structured_video_compact_text.py for the full root-cause
writeup (live discovery on the ICML 2026 executive-summary presentation
canary rebuild). This is the sibling copy of the same _compact_text
function in the render_spec generation stage, upstream of the HTML
renderer -- both needed the same fix.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "presentations" / "generate_render_spec.py"
    spec = importlib.util.spec_from_file_location("generate_render_spec", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compact_text_preserves_complete_sentence_over_limit():
    module = load_module()
    text = (
        "The leadership agenda distills the synthesis into funding, gating, "
        "and governance moves that follow from the conference evidence."
    )
    assert len(text) > 88
    assert module._compact_text(text, limit=88) == text


def test_compact_text_still_truncates_genuine_run_on_text():
    module = load_module()
    text = (
        "this is a very long caption without any terminal punctuation that "
        "just keeps going and going and going well past the limit"
    )
    result = module._compact_text(text, limit=88)
    assert len(result) <= 90
    assert result.endswith(".")


def test_compact_text_still_enforces_hard_ceiling_on_extreme_length():
    module = load_module()
    text = "This is one real sentence. " * 30 + "Done."
    result = module._compact_text(text, limit=88)
    assert len(result) <= 90
