from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from conference_pipeline.presentation.vision_qc import (
    _openai_chat_completion_options,
    inspect_screenshots_with_llm,
)



def _alias_model(alias: str) -> str:
    """The catalog's current model for an alias, so model upgrades move these tests with them."""
    from conference_pipeline.model_governance.registry import load_model_catalog

    return load_model_catalog()[alias]["model"]

def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 180), color=(255, 255, 255)).save(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vision_llm_visual_qc_passes_clean_runner_output(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    screenshot = workspace / "decks" / "01_demo" / "assets" / "generated" / "qc" / "screenshots" / "slide_001.png"
    _write_png(screenshot)

    report = inspect_screenshots_with_llm(
        screenshot.parent,
        output_report=workspace / "decks" / "01_demo" / "assets" / "generated" / "visual_llm_qc_report.json",
        workspace=workspace,
        runner=lambda _path, _slide: '{"status":"pass","issues":[],"confidence":0.92}',
    )

    assert report["status"] == "passed"
    assert report["errors"] == []
    on_disk = json.loads(
        (
            workspace
            / "decks"
            / "01_demo"
            / "assets"
            / "generated"
            / "visual_llm_qc_report.json"
        ).read_text(encoding="utf-8")
    )
    assert on_disk["model_task_id"] == "presentation.visual_qc"


def test_vision_llm_visual_qc_blocks_overflow_issue(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    screenshot = workspace / "decks" / "01_demo" / "assets" / "generated" / "qc" / "screenshots" / "slide_001.png"
    _write_png(screenshot)

    report = inspect_screenshots_with_llm(
        screenshot.parent,
        output_report=workspace / "decks" / "01_demo" / "assets" / "generated" / "visual_llm_qc_report.json",
        workspace=workspace,
        runner=lambda _path, _slide: json.dumps(
            {
                "status": "fail",
                "issues": [
                    {
                        "check_id": "text_overflow",
                        "severity": "blocker",
                        "description": "right-side label is clipped",
                        "evidence": "right edge",
                        "suggested_fix": "shorten labels",
                    }
                ],
                "confidence": 0.86,
            }
        ),
    )

    assert report["status"] == "failed"
    assert report["errors"][0]["check_id"] == "text_overflow"


def test_vision_llm_visual_qc_promotes_composition_warnings_to_blockers(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    screenshot = workspace / "decks" / "01_demo" / "assets" / "generated" / "qc" / "screenshots" / "slide_001.png"
    _write_png(screenshot)

    report = inspect_screenshots_with_llm(
        screenshot.parent,
        output_report=workspace / "decks" / "01_demo" / "assets" / "generated" / "visual_llm_qc_report.json",
        workspace=workspace,
        runner=lambda _path, _slide: json.dumps(
            {
                "status": "warn",
                "issues": [
                    {
                        "check_id": "text_overlap",
                        "severity": "warn",
                        "description": "connector arrowheads intrude into the central node label",
                        "evidence": "center of the flow chart",
                        "suggested_fix": "route connectors to the box edge with more spacing",
                    }
                ],
                "confidence": 0.89,
            }
        ),
    )

    assert report["status"] == "failed"
    assert report["errors"][0]["check_id"] == "text_overlap"
    assert report["errors"][0]["severity"] == "blocker"


def test_vision_llm_visual_qc_reuses_prior_successes_and_retries_provider_failures(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    screenshots = workspace / "decks" / "01_demo" / "assets" / "generated" / "qc" / "screenshots"
    first = screenshots / "slide_001.png"
    second = screenshots / "slide_002.png"
    _write_png(first)
    _write_png(second)
    report_path = workspace / "decks" / "01_demo" / "assets" / "generated" / "visual_llm_qc_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "provider": "openai",
                "model": _alias_model("luna"),
                "slides": [
                    {
                        "slide_number": 1,
                        "screenshot_path": first.relative_to(workspace).as_posix(),
                        "status": "fail",
                        "confidence": 0.0,
                        "issues": [
                            {
                                "slide_number": 1,
                                "check_id": "vision_llm_visual_qc_parseable",
                                "severity": "blocker",
                                "description": "vision-LLM visual QC failed: Request timed out.",
                            }
                        ],
                    },
                    {
                        "slide_number": 2,
                        "screenshot_path": second.relative_to(workspace).as_posix(),
                        "screenshot_sha256": _sha256(second),
                        "status": "pass",
                        "confidence": 0.91,
                        "issues": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    called: list[int] = []

    def runner(_path: Path, slide_number: int) -> str:
        called.append(slide_number)
        return '{"status":"pass","issues":[],"confidence":0.88}'

    report = inspect_screenshots_with_llm(
        screenshots,
        output_report=report_path,
        workspace=workspace,
        model=_alias_model("luna"),
        runner=runner,
    )

    assert report["status"] == "passed"
    assert report["reused_slide_count"] == 1
    assert called == [1]
    assert [slide["status"] for slide in report["slides"]] == ["pass", "pass"]


def test_vision_llm_visual_qc_does_not_reuse_stale_screenshot_report(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    screenshots = workspace / "decks" / "01_demo" / "assets" / "generated" / "qc" / "screenshots"
    screenshot = screenshots / "slide_001.png"
    _write_png(screenshot)
    report_path = workspace / "decks" / "01_demo" / "assets" / "generated" / "visual_llm_qc_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "provider": "openai",
                "model": _alias_model("luna"),
                "slides": [
                    {
                        "slide_number": 1,
                        "screenshot_path": screenshot.relative_to(workspace).as_posix(),
                        "screenshot_sha256": "not-current",
                        "status": "fail",
                        "confidence": 0.8,
                        "issues": [
                            {
                                "slide_number": 1,
                                "check_id": "text_clipped",
                                "severity": "blocker",
                                "description": "old screenshot issue",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    called: list[int] = []

    def runner(_path: Path, slide_number: int) -> str:
        called.append(slide_number)
        return '{"status":"pass","issues":[],"confidence":0.93}'

    report = inspect_screenshots_with_llm(
        screenshots,
        output_report=report_path,
        workspace=workspace,
        model=_alias_model("luna"),
        runner=runner,
    )

    assert report["status"] == "passed"
    assert report["reused_slide_count"] == 0
    assert called == [1]
    assert report["slides"][0]["screenshot_sha256"] == _sha256(screenshot)


def test_vision_llm_visual_qc_retries_transient_runner_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PRESENTATION_VISUAL_QC_PROVIDER_ATTEMPTS", "2")
    monkeypatch.setenv("PRESENTATION_VISUAL_QC_RETRY_SLEEP_SECONDS", "0")
    workspace = tmp_path / "conferences" / "demo2026"
    screenshot = workspace / "decks" / "01_demo" / "assets" / "generated" / "qc" / "screenshots" / "slide_001.png"
    _write_png(screenshot)
    calls = 0

    def runner(_path: Path, _slide: int) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary")
        return '{"status":"pass","issues":[],"confidence":0.94}'

    report = inspect_screenshots_with_llm(
        screenshot.parent,
        output_report=workspace / "decks" / "01_demo" / "assets" / "generated" / "visual_llm_qc_report.json",
        workspace=workspace,
        runner=runner,
    )

    assert report["status"] == "passed"
    assert calls == 2


def test_vision_llm_visual_qc_blocks_without_provider_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    workspace = tmp_path / "conferences" / "demo2026"
    screenshot = workspace / "decks" / "01_demo" / "assets" / "generated" / "qc" / "screenshots" / "slide_001.png"
    _write_png(screenshot)

    report = inspect_screenshots_with_llm(
        screenshot.parent,
        output_report=workspace / "decks" / "01_demo" / "assets" / "generated" / "visual_llm_qc_report.json",
        workspace=workspace,
    )

    assert report["status"] == "blocked"
    assert report["errors"][0]["check_id"] == "vision_llm_visual_qc_available"


def test_openai_visual_qc_uses_model_compatible_token_parameter() -> None:
    assert _openai_chat_completion_options(
        "gpt-5.4-mini",
        max_completion_tokens=1200,
    ) == {"max_completion_tokens": 1200}
    assert _openai_chat_completion_options(
        "gpt-4o",
        max_completion_tokens=1200,
    ) == {"max_tokens": 1200, "temperature": 0.0}


def _deck_of_screenshots(tmp_path: Path, count: int) -> tuple[Path, Path]:
    shots = tmp_path / "qc" / "screenshots"
    shots.mkdir(parents=True)
    for i in range(1, count + 1):
        _write_png(shots / f"{i:02d}__sl_{i:02d}.png")
    return shots, tmp_path


def test_slides_are_judged_concurrently_and_reported_in_slide_order(tmp_path: Path) -> None:
    """27 vision calls in a row is the longest block in building a deck, and no
    slide's verdict depends on another's. Returning out of order is the point."""

    import random
    import time as _time

    shots, workspace = _deck_of_screenshots(tmp_path, 12)
    seen: list[int] = []

    def runner(path: Path, slide_number: int) -> str:
        _time.sleep(random.uniform(0.0, 0.05))
        seen.append(slide_number)
        return json.dumps({"status": "pass", "issues": [], "confidence": 0.9})

    report = inspect_screenshots_with_llm(
        shots, output_report=workspace / "qc" / "report.json", workspace=workspace, runner=runner,
    )
    numbers = [slide["slide_number"] for slide in report["slides"]]
    assert numbers == sorted(numbers) == list(range(1, 13))
    assert len(seen) == 12


def test_a_broken_provider_still_stops_the_run_rather_than_being_paid_for_every_slide(
    tmp_path: Path, monkeypatch
) -> None:
    """The failure breaker exists so a broken provider is not called once per
    slide. Judging slides at once must not spend them anyway. Exercised on the
    production path -- the breaker deliberately exempts the runner seam -- with
    the provider call itself stubbed, so nothing leaves the machine."""

    from conference_pipeline.presentation import vision_qc

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PRESENTATION_VISUAL_QC_MAX_PROVIDER_FAILURES", "2")
    monkeypatch.setenv("PRESENTATION_VISUAL_QC_WORKERS", "2")
    shots, workspace = _deck_of_screenshots(tmp_path, 20)
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("provider is down")

    monkeypatch.setattr(vision_qc, "_inspect_screenshot_with_retries", boom)
    report = vision_qc.inspect_screenshots_with_llm(
        shots, output_report=workspace / "qc" / "report.json", workspace=workspace,
    )
    assert report["status"] == "failed"
    # Far short of one call per slide: the breaker tripped and the rest were
    # never attempted. A couple already in flight are allowed to finish.
    assert calls["n"] <= 8, calls["n"]


# --- Visual QC is a governed task, not an OpenAI feature -----------------------

K3 = "openrouter/moonshotai/kimi-k3"
PASS_JSON = '{"status":"pass","issues":[],"confidence":0.9}'


def _fake_chat_client(captured: list[dict], *, content: str = PASS_JSON, finish_reason: str = "stop"):
    from types import SimpleNamespace

    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            id="gen-test",
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason,
            )],
            usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=40, total_tokens=1240),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _one_screenshot(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    screenshot = workspace / "decks" / "01_demo" / "assets" / "generated" / "qc" / "screenshots" / "slide_001.png"
    _write_png(screenshot)
    return screenshot.parent, workspace


def test_selection_defaults_to_the_registered_task_model() -> None:
    from conference_pipeline.model_governance import default_model, default_provider
    from conference_pipeline.presentation.vision_qc import MODEL_TASK_ID, resolve_visual_qc_selection

    assert resolve_visual_qc_selection(None, None) == (
        default_provider(MODEL_TASK_ID), default_model(MODEL_TASK_ID),
    )


def test_a_model_alone_finds_its_provider_and_a_provider_alone_finds_its_model() -> None:
    from conference_pipeline.presentation.vision_qc import resolve_visual_qc_selection

    assert resolve_visual_qc_selection(None, K3) == ("openrouter", K3)
    assert resolve_visual_qc_selection("openrouter", None) == ("openrouter", K3)


def test_governance_still_decides_which_models_may_judge(tmp_path: Path) -> None:
    import pytest

    from conference_pipeline.model_governance import ModelGovernanceError

    shots, workspace = _one_screenshot(tmp_path)
    with pytest.raises(ModelGovernanceError):
        inspect_screenshots_with_llm(
            shots, output_report=workspace / "r.json", workspace=workspace,
            provider="openrouter", model="openrouter/z-ai/glm-5.3",
            runner=lambda _p, _s: PASS_JSON,
        )


def test_a_non_openai_provider_is_no_longer_refused_as_unsupported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    shots, workspace = _one_screenshot(tmp_path)
    report = inspect_screenshots_with_llm(
        shots, output_report=workspace / "r.json", workspace=workspace, provider="openrouter", model=K3,
    )
    assert report["status"] == "blocked"
    reason = report["errors"][0]["description"]
    assert "OPENROUTER_API_KEY" in reason and "unsupported" not in reason


def test_openrouter_judges_a_screenshot_end_to_end(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.model_governance import llm_client

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("PRESENTATION_VISUAL_QC_REASONING_EFFORT", raising=False)
    captured: list[dict] = []
    monkeypatch.setattr(llm_client, "_chat_client", lambda provider, request: _fake_chat_client(captured))
    shots, workspace = _one_screenshot(tmp_path)

    report = inspect_screenshots_with_llm(
        shots, output_report=workspace / "r.json", workspace=workspace, provider="openrouter", model=K3,
    )

    assert report["status"] == "passed"
    assert (report["provider"], report["model"]) == ("openrouter", K3)
    (call,) = captured
    assert call["model"] == "moonshotai/kimi-k3", "OpenRouter wants the bare slug"
    assert "extra_body" not in call, "reasoning effort is opt-in"
    user = call["messages"][-1]["content"]
    assert any(block["type"] == "image_url" for block in user)
    assert call["messages"][0]["role"] == "system"


def test_openrouter_reasoning_effort_is_sent_only_when_asked_for(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.model_governance import llm_client

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("PRESENTATION_VISUAL_QC_REASONING_EFFORT", "low")
    captured: list[dict] = []
    monkeypatch.setattr(llm_client, "_chat_client", lambda provider, request: _fake_chat_client(captured))
    shots, workspace = _one_screenshot(tmp_path)

    inspect_screenshots_with_llm(
        shots, output_report=workspace / "r.json", workspace=workspace, provider="openrouter", model=K3,
    )
    assert captured[0]["extra_body"] == {"reasoning": {"effort": "low"}}


def test_a_verdict_cut_off_by_the_token_cap_says_so(tmp_path: Path, monkeypatch) -> None:
    from conference_pipeline.model_governance import llm_client

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("PRESENTATION_VISUAL_QC_PROVIDER_ATTEMPTS", "1")
    monkeypatch.setattr(
        llm_client, "_chat_client",
        lambda provider, request: _fake_chat_client([], content="", finish_reason="length"),
    )
    shots, workspace = _one_screenshot(tmp_path)

    report = inspect_screenshots_with_llm(
        shots, output_report=workspace / "r.json", workspace=workspace, provider="openrouter", model=K3,
    )
    assert report["status"] == "failed"
    assert "PRESENTATION_VISUAL_QC_MAX_COMPLETION_TOKENS" in report["errors"][0]["description"]


def test_a_reviewer_run_on_another_provider_never_reuses_prior_verdicts(tmp_path: Path, monkeypatch) -> None:
    """Prior-slide reuse is keyed on provider and model, so trying K3 on a deck
    gpt-6-sol already judged re-judges every slide rather than inheriting."""

    from conference_pipeline.model_governance import llm_client

    shots, workspace = _one_screenshot(tmp_path)
    report_path = workspace / "r.json"
    inspect_screenshots_with_llm(
        shots, output_report=report_path, workspace=workspace, runner=lambda _p, _s: PASS_JSON,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured: list[dict] = []
    monkeypatch.setattr(llm_client, "_chat_client", lambda provider, request: _fake_chat_client(captured))
    report = inspect_screenshots_with_llm(
        shots, output_report=report_path, workspace=workspace, provider="openrouter", model=K3,
    )
    assert report["reused_slide_count"] == 0
    assert len(captured) == 1
