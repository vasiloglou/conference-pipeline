from pathlib import Path

import pytest

from scripts.presentations import figure_quality_panel as panel


def verdict(
    value: str,
    *,
    tags: list[str] | None = None,
    status: str = "ok",
) -> panel.JudgeVerdict:
    return panel.JudgeVerdict(
        verdict=value,  # type: ignore[arg-type]
        failure_modes=tags or [],
        rationale="ok",
        status=status,  # type: ignore[arg-type]
        receipt_path=None,
        faithfulness_verdict=value,  # type: ignore[arg-type]
        faithfulness_tags=tags or [],
        legibility_verdict="pass",
        legibility_tags=[],
        style_verdict="pass",
        style_tags=[],
    )


def test_majority_pass():
    result = panel.aggregate_panel(
        [verdict("pass"), verdict("pass"), verdict("fail", tags=["legibility"])],
        ["gemini", "claude", "qwen"],
    )
    assert result.majority_verdict == "pass"
    assert result.votes == {"gemini": "pass", "claude": "pass", "qwen": "fail"}
    assert result.failure_modes == ["legibility"]


def test_majority_fail():
    result = panel.aggregate_panel(
        [
            verdict("pass"),
            verdict("fail", tags=["legibility"]),
            verdict("fail", tags=["wrong-source"]),
        ],
        ["gemini", "claude", "qwen"],
    )
    assert result.majority_verdict == "fail"
    assert result.failure_modes == ["wrong-source", "legibility"]


def test_tag_union_dedup():
    result = panel.aggregate_panel(
        [
            verdict("pass"),
            verdict("fail", tags=["legibility", "other"]),
            verdict("fail", tags=["legibility"]),
        ],
        ["gemini", "claude", "qwen"],
    )
    assert result.failure_modes == ["legibility", "other"]


def test_composition_and_aesthetic_tags_are_first_class():
    result = panel.aggregate_panel(
        [
            verdict("pass"),
            verdict("fail", tags=["aesthetic-polish"]),
            verdict("fail", tags=["composition"]),
        ],
        ["gemini", "claude", "qwen"],
    )
    assert result.majority_verdict == "fail"
    assert result.failure_modes == ["composition", "aesthetic-polish"]


def test_blocked_judge_counts_as_nonpass():
    result = panel.aggregate_panel(
        [
            verdict("pass"),
            verdict("fail", tags=["style-drift"]),
            verdict("pass", status="blocked", tags=["other"]),
        ],
        ["gemini", "claude", "qwen"],
    )
    assert result.majority_verdict == "fail"
    assert result.votes["qwen"] == "blocked"


def test_unanimous_true():
    result = panel.aggregate_panel(
        [verdict("pass"), verdict("pass"), verdict("pass")],
        ["gemini", "claude", "qwen"],
    )
    assert result.unanimous is True


def test_unanimous_false():
    result = panel.aggregate_panel(
        [verdict("pass"), verdict("pass"), verdict("fail")],
        ["gemini", "claude", "qwen"],
    )
    assert result.unanimous is False


def test_adapter_stub_parses_lenses():
    raw = """
    {
      "verdict": "pass",
      "failure_modes": [],
      "rationale": "clear",
      "faithfulness": {"verdict": "pass", "tags": [], "notes": ""},
      "legibility": {"verdict": "pass", "tags": [], "notes": ""},
      "style": {"verdict": "pass", "tags": [], "notes": ""}
    }
    """
    result = panel.gemini_judge_adapter(
        Path("render.png"),
        None,
        judge_callable=lambda _contents: raw,
    )
    assert result.status == "ok"
    assert result.verdict == "pass"


def test_unknown_tags_normalize_to_other():
    raw = """
    {
      "verdict": "fail",
      "failure_modes": ["mystery"],
      "rationale": "odd",
      "faithfulness": {"verdict": "fail", "tags": ["mystery"], "notes": ""},
      "legibility": {"verdict": "pass", "tags": [], "notes": ""},
      "style": {"verdict": "pass", "tags": [], "notes": ""}
    }
    """
    result = panel.gemini_judge_adapter(
        Path("render.png"),
        None,
        judge_callable=lambda _contents: raw,
    )
    assert result.failure_modes == ["other"]


def test_run_panel_includes_qwen_stub():
    raw_pass = """
    {
      "verdict": "pass",
      "failure_modes": [],
      "rationale": "ok",
      "faithfulness": {"verdict": "pass", "tags": [], "notes": ""},
      "legibility": {"verdict": "pass", "tags": [], "notes": ""},
      "style": {"verdict": "pass", "tags": [], "notes": ""}
    }
    """
    result = panel.run_panel(
        Path("render.png"),
        None,
        judge_callables={
            "gemini": lambda _contents: raw_pass,
            "claude": lambda _contents: raw_pass,
            "qwen": lambda _contents: raw_pass,
        },
    )
    assert result.majority_verdict == "pass"
    assert set(result.votes) == {"gemini", "claude", "qwen"}


# --- any model the task allows may judge; the panel's size is configuration ---

K3 = "openrouter/moonshotai/kimi-k3"
_PASS = """{"verdict":"pass","failure_modes":[],"rationale":"ok",
"faithfulness":{"verdict":"pass","tags":[],"notes":""},
"legibility":{"verdict":"pass","tags":[],"notes":""},
"style":{"verdict":"pass","tags":[],"notes":""}}"""
_FAIL = """{"verdict":"fail","failure_modes":["legibility"],"rationale":"tiny",
"faithfulness":{"verdict":"pass","tags":[],"notes":""},
"legibility":{"verdict":"fail","tags":["legibility"],"notes":""},
"style":{"verdict":"pass","tags":[],"notes":""}}"""


def test_the_panel_defaults_to_the_original_trio(monkeypatch):
    monkeypatch.delenv("FIGURE_PANEL_JUDGES", raising=False)
    assert panel.configured_judges() == ("gemini", "claude", "qwen")


def test_a_judge_can_be_any_model_the_task_allows_and_votes_under_its_slug():
    assert panel.configured_judges("gemini,claude," + K3) == ("gemini", "claude", K3)
    assert panel.judge_name(K3) == "kimi-k3"
    assert panel.judge_name("gemini") == "gemini"


def test_a_judge_listed_twice_is_an_error_never_a_double_vote():
    import pytest

    with pytest.raises(ValueError):
        panel.configured_judges("gemini,gemini")


def test_a_kimi_judge_replaces_the_local_qwen_judge_without_a_local_server():
    """The trio's Qwen member needs a local MLX server and halts the panel when it
    is down. A panel that does not seat it never asks for it."""

    result = panel.run_panel(
        Path("render.png"),
        None,
        judges=("gemini", "claude", K3),
        judge_callables={
            "gemini": lambda _c: _PASS,
            "claude": lambda _c: _PASS,
            "kimi-k3": lambda _c: _PASS,
        },
    )
    assert set(result.votes) == {"gemini", "claude", "kimi-k3"}
    assert result.majority_verdict == "pass"


def test_the_api_judge_sends_the_figure_as_an_image_to_the_governed_model(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    png = tmp_path / "r.png"
    Image.new("RGB", (32, 32)).save(png)
    seen = {}

    def fake_complete(task_id, **kwargs):
        seen.update(kwargs, task_id=task_id)
        return SimpleNamespace(text=_PASS, receipt=None)

    monkeypatch.setattr(panel, "complete", fake_complete)
    verdict = panel.api_judge_adapter(png, None, model=K3, provider="openrouter")
    assert verdict.status == "ok" and verdict.verdict == "pass"
    assert seen["task_id"] == panel.PANEL_TASK_ID and len(seen["images"]) == 1


def test_a_four_judge_panel_needs_a_strict_majority_so_a_tie_fails():
    def verdict(value):
        return panel.JudgeVerdict(
            verdict=value, failure_modes=[] if value == "pass" else ["legibility"], rationale="",
            status="ok", receipt_path=None, faithfulness_verdict="pass", faithfulness_tags=[],
            legibility_verdict=value, legibility_tags=[], style_verdict="pass", style_tags=[],
        )

    names = ["gemini", "claude", "qwen", "kimi-k3"]
    assert panel.aggregate_panel([verdict("pass")] * 2 + [verdict("fail")] * 2, names).majority_verdict == "fail"
    assert panel.aggregate_panel([verdict("pass")] * 3 + [verdict("fail")], names).majority_verdict == "pass"
