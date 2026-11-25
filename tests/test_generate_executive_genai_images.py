"""build_prompt must use the invoking conference's own label, never a
hardcoded "ICLR 2026" -- a prior bug that would have put the wrong
conference's name into every generated hero-image prompt (including
CVPR2026's, once Phase 10/11 wires this script up)."""

from __future__ import annotations

from scripts.presentations.generate_executive_genai_images import build_prompt


def test_build_prompt_uses_the_supplied_conference_label_not_iclr() -> None:
    prompt = build_prompt("Reasoning Agents", "a glowing lattice", conference_label="CVPR 2026")

    assert "CVPR 2026" in prompt
    assert "ICLR" not in prompt
    assert "Reasoning Agents" in prompt
    assert "a glowing lattice" in prompt


def test_build_prompt_reflects_a_third_conference_too() -> None:
    """Not just CVPR -- any conference label passed through must appear
    verbatim, proving this isn't a two-conference special case."""

    prompt = build_prompt("Diffusion Models", "a spiral of light", conference_label="ICML 2026")

    assert "ICML 2026" in prompt
    assert "ICLR" not in prompt
