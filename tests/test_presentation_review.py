from __future__ import annotations

import importlib
import json
from pathlib import Path

import conference_pipeline.presentation.presentation_review as presentation_review_module
from conference_pipeline.model_governance import model_for_alias
from conference_pipeline.presentation.presentation_review import (
    DEFAULT_REVIEWERS,
    REVIEW_CRITERIA,
    build_review_prompt,
    review_presentation,
)



def _opus_model() -> str:
    """The catalog's current Opus model (these tests pin the tier, not a generation)."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_review_workspace(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    requirements = workspace / "decks" / "01_demo_topic" / "presentation_blueprint.json"
    narration = workspace / "decks" / "01_demo_topic" / "6_presenter_script.md"
    screenshots = workspace / "decks" / "01_demo_topic" / "assets" / "generated" / "qc" / "screenshots"
    write_text(requirements, '{"requirements":["technical audience","clear slides"]}\n')
    write_text(narration, "# Presenter Script\n\nSay something concrete.\n")
    write_text(screenshots / "slide_001.png", "png\n")
    write_text(screenshots / "slide_002.png", "png\n")
    return workspace, requirements, narration, screenshots


def passing_response(summary: str = "Production-ready technical presentation.") -> str:
    return json.dumps(
        {
            "status": "pass",
            "summary": summary,
            "criteria": {
                criterion["id"]: {"status": "pass", "comments": ["Looks good."]}
                for criterion in REVIEW_CRITERIA
            },
            "production_blockers": [],
            "slide_comments": [],
            "suggested_fixes": [],
            "confidence": 0.9,
        }
    )


def test_build_review_prompt_includes_requirements_narration_screenshots_and_criteria(
    tmp_path: Path,
) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)

    prompt = build_review_prompt(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots=sorted(screenshots.glob("*.png")),
    )

    assert "technical audience" in prompt
    assert "Say something concrete." in prompt
    assert "slide_001.png" in prompt
    assert "guideline_adherence" in prompt
    assert "technical_audience_pleasantness" in prompt
    assert "production_readiness" in prompt
    assert "aesthetic_pleasantness" in prompt
    assert "clarity_and_congestion" in prompt
    assert "artifact_contract_adherence" in prompt


def test_build_review_prompt_states_label_namespace_and_approved_expansion(
    tmp_path: Path,
) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)

    prompt = build_review_prompt(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots=sorted(screenshots.glob("*.png")),
    )

    # The reviewer must be told the PLAN-/SRC- namespace split, the approved-expansion
    # mechanism, and that source digests are distinct from blueprint_sha256 — the three
    # facts that previously produced false-positive blockers.
    assert "approved_expansion" in prompt
    assert "PLAN-NNN" in prompt
    assert "SRC-NNN" in prompt
    assert "blueprint_sha256" in prompt
    # The approved source ledger is the binding report / approved source plan,
    # not the blueprint's narrower seed list.
    assert "approved_source_plan_sha256" in prompt
    assert "source_binding_report" in prompt


def test_pre_video_review_tells_reviewers_audio_and_video_come_later(tmp_path: Path) -> None:
    """Without a video receipt the review is the deck-acceptance gate, which runs
    before TTS/video; CVPR 2026 reviewers blocked every deck on leftover builds."""

    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    kwargs = dict(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots=sorted(screenshots.glob("*.png")),
    )
    assert "PRE-VIDEO deck-acceptance review" in build_review_prompt(**kwargs)
    receipt = requirements.parent / "video_receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    assert "PRE-VIDEO" not in build_review_prompt(**kwargs, artifact_paths={"video_receipt": receipt})


def test_build_review_prompt_includes_hash_bound_artifact_context(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    deck_dir = requirements.parent
    render_spec = deck_dir / "render_spec.json"
    render_spec.write_text(
        json.dumps(
            {
                "slides": [{"slide_id": "slide-001"}],
                "delivery": {"notebooklm": {"contract_path": "delivery_notebooklm.json"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prompt = build_review_prompt(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots=sorted(screenshots.glob("*.png")),
        artifact_paths={"render_spec": render_spec},
    )

    assert "Produced artifact contracts and receipts" in prompt
    assert "## render_spec" in prompt
    assert "delivery_notebooklm.json" in prompt
    assert "sha256" in prompt


def test_review_presentation_dry_run_writes_planned_three_reviewer_report(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    report_output = workspace / "decks" / "01_demo_topic" / "presentation_review.json"
    markdown_output = workspace / "decks" / "01_demo_topic" / "presentation_review.md"
    prompt_output = workspace / "decks" / "01_demo_topic" / "presentation_review.prompt.md"

    report = review_presentation(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots_dir=screenshots,
        report_output=report_output,
        markdown_output=markdown_output,
        prompt_output=prompt_output,
        live=False,
    )

    assert report["status"] == "planned"
    assert report["production_ready"] is False
    assert report["inputs"]["screenshot_count"] == 2
    assert report["inputs"]["artifacts"] == {}
    reviewer_models = {reviewer["model"] for reviewer in report["reviewers"]}
    assert {_opus_model(), model_for_alias("sol"), DEFAULT_REVIEWERS[-1].model} == reviewer_models
    assert report_output.is_file()
    assert markdown_output.is_file()
    assert prompt_output.is_file()


def test_review_presentation_missing_input_writes_blocked_receipt(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    requirements.unlink()
    report_output = workspace / "decks" / "01_demo_topic" / "presentation_review.json"
    markdown_output = workspace / "decks" / "01_demo_topic" / "presentation_review.md"
    prompt_output = workspace / "decks" / "01_demo_topic" / "presentation_review.prompt.md"

    report = review_presentation(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots_dir=screenshots,
        report_output=report_output,
        markdown_output=markdown_output,
        prompt_output=prompt_output,
        live=False,
    )

    assert report["status"] == "blocked"
    assert report["production_ready"] is False
    assert "requirements file missing" in report["aggregated_feedback"]["summary"]
    assert report_output.is_file()
    assert markdown_output.is_file()
    assert prompt_output.is_file()
    assert "blocked before prompt construction" in prompt_output.read_text(encoding="utf-8")


def test_review_presentation_live_runner_aggregates_blockers(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)

    def runner(reviewer, prompt, prompt_path, workspace_path):
        assert "slide_001.png" in prompt
        assert prompt_path.is_file()
        assert workspace_path == workspace
        if reviewer.reviewer_id == "codex_5_5":
            return json.dumps(
                {
                    "status": "fail",
                    "summary": "Slides are congested.",
                    "criteria": {
                        criterion["id"]: {"status": "pass", "comments": ["ok"]}
                        for criterion in REVIEW_CRITERIA
                    }
                    | {
                        "clarity_and_congestion": {
                            "status": "fail",
                            "comments": ["Dense text blocks on slide 2."],
                        }
                    },
                    "production_blockers": ["Slide 2 is too congested for production."],
                    "slide_comments": [
                        {"slide": 2, "severity": "fail", "comment": "Too dense."}
                    ],
                    "suggested_fixes": ["Split slide 2."],
                    "confidence": 0.8,
                }
            )
        return passing_response(f"{reviewer.display_name} passed the deck.")

    report = review_presentation(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots_dir=screenshots,
        report_output=workspace / "decks" / "01_demo_topic" / "presentation_review.json",
        markdown_output=workspace / "decks" / "01_demo_topic" / "presentation_review.md",
        live=True,
        runner=runner,
    )

    assert report["status"] == "failed"
    assert report["production_ready"] is False
    blockers = report["aggregated_feedback"]["blockers"]
    assert any(blocker["reviewer_id"] == "codex_5_5" for blocker in blockers)
    assert any("congested" in blocker["comment"] for blocker in blockers)


def test_review_presentation_live_runner_blocks_on_slide_fail_comments(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)

    def runner(reviewer, prompt, prompt_path, workspace_path):  # noqa: ARG001
        payload = json.loads(passing_response(f"{reviewer.display_name} passed the deck."))
        payload["slide_comments"] = [
            {
                "slide": 1,
                "severity": "fail",
                "comment": "Slide 1 text is unreadable in the screenshot.",
            }
        ]
        return json.dumps(payload)

    report = review_presentation(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots_dir=screenshots,
        report_output=workspace / "decks" / "01_demo_topic" / "presentation_review.json",
        markdown_output=workspace / "decks" / "01_demo_topic" / "presentation_review.md",
        live=True,
        runner=runner,
    )

    assert report["status"] == "failed"
    assert report["production_ready"] is False
    blockers = report["aggregated_feedback"]["blockers"]
    assert any(blocker["criterion"] == "slide_comment" for blocker in blockers)
    assert any("unreadable" in blocker["comment"] for blocker in blockers)


def test_review_presentation_live_runner_treats_warnings_as_advisory(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)

    def runner(reviewer, prompt, prompt_path, workspace_path):  # noqa: ARG001
        payload = json.loads(passing_response("Aesthetic polish is close but not final."))
        payload["status"] = "warn"
        return json.dumps(payload)

    report = review_presentation(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots_dir=screenshots,
        report_output=workspace / "decks" / "01_demo_topic" / "presentation_review.json",
        markdown_output=workspace / "decks" / "01_demo_topic" / "presentation_review.md",
        live=True,
        runner=runner,
    )

    # Warnings are advisory: they no longer block readiness, but stay recorded.
    assert report["status"] == "passed"
    assert report["production_ready"] is True
    assert report["aggregated_feedback"]["blockers"] == []
    warnings = report["aggregated_feedback"]["warnings"]
    assert any(warning["criterion"] == "overall" for warning in warnings)


def test_review_presentation_unavailable_reviewer_is_advisory_not_blocking(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)

    def runner(reviewer, prompt, prompt_path, workspace_path):  # noqa: ARG001
        if reviewer.reviewer_id == "codex_5_5":
            raise RuntimeError("reviewer command exited 1: Unknown model codex-5.5")
        return passing_response(f"{reviewer.display_name} passed the deck.")

    report = review_presentation(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots_dir=screenshots,
        report_output=workspace / "decks" / "01_demo_topic" / "presentation_review.json",
        markdown_output=workspace / "decks" / "01_demo_topic" / "presentation_review.md",
        live=True,
        runner=runner,
    )

    # Two of three reviewers ran and passed; the unavailable one is advisory only.
    assert report["status"] == "passed"
    assert report["production_ready"] is True
    assert report["aggregated_feedback"]["blockers"] == []
    assert report["reviewer_coverage"]["ran"] == 2
    assert report["reviewer_coverage"]["unavailable"] == ["codex_5_5"]
    warnings = report["aggregated_feedback"]["warnings"]
    assert any(
        warning["reviewer_id"] == "codex_5_5"
        and warning["criterion"] == "reviewer_available"
        for warning in warnings
    )


def test_review_presentation_blocks_when_too_few_reviewers_run(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)

    def runner(reviewer, prompt, prompt_path, workspace_path):  # noqa: ARG001
        if reviewer.reviewer_id == "gemini":
            return passing_response(f"{reviewer.display_name} passed the deck.")
        raise RuntimeError("reviewer command unavailable")

    report = review_presentation(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots_dir=screenshots,
        report_output=workspace / "decks" / "01_demo_topic" / "presentation_review.json",
        markdown_output=workspace / "decks" / "01_demo_topic" / "presentation_review.md",
        live=True,
        runner=runner,
    )

    # Only one reviewer ran (below the default minimum of two): cannot certify.
    assert report["status"] == "blocked"
    assert report["production_ready"] is False
    assert report["reviewer_coverage"]["ran"] == 1


def test_gemini_model_override_updates_default_command(monkeypatch) -> None:
    monkeypatch.setenv("PRESENTATION_REVIEW_GEMINI_MODEL", "gemini-test-model")
    reloaded = importlib.reload(presentation_review_module)

    try:
        gemini = next(reviewer for reviewer in reloaded.DEFAULT_REVIEWERS if reviewer.provider == "gemini")

        assert gemini.model == "gemini-test-model"
        assert gemini.command() == ("gemini", "--model", "gemini-test-model")
    finally:
        monkeypatch.delenv("PRESENTATION_REVIEW_GEMINI_MODEL", raising=False)
        importlib.reload(presentation_review_module)


def test_build_review_prompt_carries_operator_acceptance_to_reviewers(tmp_path: Path) -> None:
    """Under --accept-visual-qc-findings the runner records the accepted
    findings verbatim; the reviewers must see that record, or they refuse the
    deck on the failed QC report alone -- which is what happened to all
    twelve CVPR 2026 topic decks on 2026-09-15."""

    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    deck_dir = requirements.parent
    acceptance = deck_dir / "assets" / "generated" / "visual_qc_acceptance.json"
    acceptance.parent.mkdir(parents=True, exist_ok=True)
    acceptance.write_text(
        json.dumps(
            {
                "accepted_by": "operator:--accept-visual-qc-findings",
                "accepted_findings": [{"slide_number": 5, "findings": [{"check_id": "other"}]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    common = dict(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        requirements_path=requirements,
        narration_script_path=narration,
        screenshots=sorted(screenshots.glob("*.png")),
    )

    with_acceptance = build_review_prompt(**common, artifact_paths={"visual_qc_acceptance": acceptance})
    assert "OPERATOR ACCEPTANCE ON RECORD" in with_acceptance
    assert "## visual_qc_acceptance" in with_acceptance
    assert "accepted residual risk" in with_acceptance

    without = build_review_prompt(**common, artifact_paths={})
    assert "OPERATOR ACCEPTANCE ON RECORD" not in without


def test_gemini_prompt_neutralizes_at_tokens_but_other_reviewers_get_it_verbatim(monkeypatch) -> None:
    """The Gemini CLI splices files for "@<token>"; "F1@50" in narration made it
    report a leaked receipt path that existed nowhere (CVPR deck 09, 2026-09-16)."""
    from conference_pipeline.presentation import presentation_review as pr

    assert pr.gemini_safe_prompt("a 4.8% F1@50 improvement; mail me@x.org; lone @ sign") == (
        "a 4.8% F1＠50 improvement; mail me＠x.org; lone @ sign"
    )

    seen: dict[str, str] = {}

    class _Done:
        returncode = 0
        stdout = '{"status": "pass"}'
        stderr = ""

    def fake_run(cmd, **kwargs):
        seen[cmd[0]] = kwargs["input"]
        return _Done()

    monkeypatch.setattr(pr.subprocess, "run", fake_run)
    pr._run_reviewer_command(["gemini", "--model", "m"], prompt="F1@50", workspace=Path("."))
    pr._run_reviewer_command(["claude", "--print"], prompt="F1@50", workspace=Path("."))
    assert seen["gemini"] == "F1＠50"
    assert seen["claude"] == "F1@50"


def test_only_factual_and_attribution_findings_block_a_deck() -> None:
    """Operator rule 2026-09-25: lineage, visual-requirement and presentation
    findings are recorded; factual/attribution (and uncategorized) ones block."""

    from conference_pipeline.presentation.presentation_review import _collect_blockers, _collect_recorded_findings

    def review(rid: str, blockers: list, comments: list, status: str = "fail") -> dict:
        return {"reviewer_id": rid, "parsed": {"status": status, "summary": "s",
                "criteria": {"production_readiness": {"status": "fail", "comments": ["c"]}},
                "production_blockers": blockers, "slide_comments": comments}}

    soft = review("codex", [{"category": "lineage", "issue": "receipt stale"},
                            {"category": "visual_requirement", "issue": "no chart"}],
                  [{"slide": 5, "severity": "fail", "category": "presentation", "comment": "dense"}])
    assert _collect_blockers([soft]) == []
    assert len(_collect_recorded_findings([soft])) == 3

    hard = review("claude", [{"category": "attribution", "issue": "wrong paper credited"}],
                  [{"slide": 3, "severity": "fail", "category": "factual", "comment": "800 m is 1,000 m"}])
    assert [b["category"] for b in _collect_blockers([hard])] == ["attribution", "factual"]

    legacy = review("gemini", ["uncategorized blocker"], [])
    assert [b["comment"] for b in _collect_blockers([legacy])] == ["uncategorized blocker"]

    bare_fail = review("x", [], [])
    assert [b["criterion"] for b in _collect_blockers([bare_fail])] == ["overall"]


def test_reviewer_payload_keeps_finding_categories() -> None:
    from conference_pipeline.presentation import presentation_review as pr

    normalize = next(
        getattr(pr, name) for name in dir(pr)
        if name.startswith("_normalize") and name not in {"_normalize_slide_comments", "_normalize_production_blockers"}
        and callable(getattr(pr, name)) and "payload" in (getattr(pr, name).__code__.co_varnames or ())
    )
    parsed = normalize({
        "status": "fail", "summary": "s", "criteria": {},
        "production_blockers": [{"category": "Lineage", "slide": 3, "issue": "receipt stale"}, "plain"],
        "slide_comments": [{"slide": 3, "severity": "fail", "category": "factual", "comment": "wrong number"}],
    })
    assert parsed["production_blockers"] == [{"category": "lineage", "slide": 3, "issue": "receipt stale"}, "plain"]
    assert parsed["slide_comments"][0]["category"] == "factual"
    assert [b["category"] for b in pr._collect_blockers([{"reviewer_id": "r", "parsed": parsed}])] == [None, "factual"]


# --- API-transport reviewers: any model the review task allows may sit on the panel ---

pr = presentation_review_module

def test_the_panel_defaults_to_the_three_cli_members_and_names_no_api_reviewer(monkeypatch) -> None:
    monkeypatch.delenv("PRESENTATION_REVIEW_REVIEWERS", raising=False)
    panel = pr.configured_reviewers()
    assert panel == pr.DEFAULT_REVIEWERS
    assert {member.transport for member in panel} == {"cli"}


def test_a_panel_can_mix_cli_members_with_an_api_reviewer() -> None:
    panel = pr.configured_reviewers("claude_code_opus,codex_5_5,openrouter/moonshotai/kimi-k3")
    assert [member.transport for member in panel] == ["cli", "cli", "api"]
    assert panel[2].provider == "openrouter" and panel[2].command() == ()


def test_an_unknown_or_unallowed_or_repeated_member_is_an_error_not_a_smaller_panel() -> None:
    import pytest

    from conference_pipeline.model_governance import ModelGovernanceError

    with pytest.raises(ModelGovernanceError):
        pr.configured_reviewers("claude_code_opus,openrouter/z-ai/glm-5.3")
    with pytest.raises(ValueError):
        pr.configured_reviewers("gemini,gemini")


def test_an_api_reviewer_is_sent_every_screenshot_as_an_image(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from PIL import Image

    shots = []
    for index in range(3):
        path = tmp_path / f"s{index}.png"
        Image.new("RGB", (64, 36), (255, 255, 255)).save(path)
        shots.append(path)
    seen: dict = {}

    def fake_complete(task_id, **kwargs):
        seen.update(kwargs, task_id=task_id)
        return SimpleNamespace(text='{"status":"pass"}', stop_reason="end", block_types=("text",))

    monkeypatch.setattr(pr, "complete", fake_complete)
    reviewer = pr.api_reviewer(model="openrouter/moonshotai/kimi-k3")
    text = pr._run_api_reviewer(reviewer, prompt="review it", screenshots=shots, workspace=tmp_path)
    assert text == '{"status":"pass"}'
    assert seen["task_id"] == pr.MODEL_TASK_ID and seen["provider"] == "openrouter"
    assert len(seen["images"]) == 3 and "ATTACHED" in seen["prompt"]


def test_an_api_reviewer_that_used_its_whole_ceiling_blocks_with_the_reason(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest
    from PIL import Image

    shot = tmp_path / "s.png"
    Image.new("RGB", (64, 36), (255, 255, 255)).save(shot)
    seen: dict = {}

    def fake_complete(task_id, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            text='{"status": "pass", "summary": "cut', stop_reason="max_tokens",
            block_types=("text",), output_tokens=40000,
        )

    monkeypatch.setattr(pr, "complete", fake_complete)
    reviewer = pr.api_reviewer(model="openrouter/moonshotai/kimi-k3")
    with pytest.raises(RuntimeError, match="whole 40000-token ceiling"):
        pr._run_api_reviewer(reviewer, prompt="x", screenshots=[shot], workspace=tmp_path)
    assert seen["max_tokens"] >= 40000  # reasoning shares the ceiling with the verdict JSON


def test_a_deck_over_the_image_limit_blocks_the_api_reviewer_rather_than_truncating(tmp_path, monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("PRESENTATION_REVIEW_API_MAX_IMAGES", "2")
    reviewer = pr.api_reviewer(model="openrouter/moonshotai/kimi-k3")
    with pytest.raises(RuntimeError, match="exceed"):
        pr._run_api_reviewer(reviewer, prompt="x", screenshots=[tmp_path / f"{i}.png" for i in range(3)], workspace=tmp_path)


def test_review_presentation_runs_an_api_member_beside_a_cli_member(tmp_path, monkeypatch) -> None:
    """End to end through review_presentation: the CLI member answers via the
    runner seam, the API member via the shared client; both count toward quorum."""

    from types import SimpleNamespace

    from PIL import Image

    workspace = tmp_path / "ws"
    shots = workspace / "shots"
    shots.mkdir(parents=True)
    Image.new("RGB", (64, 36)).save(shots / "01.png")
    (workspace / "req.md").write_text("requirements")
    (workspace / "narr.md").write_text("narration")
    monkeypatch.setattr(
        pr, "complete",
        lambda task_id, **kw: SimpleNamespace(text='{"status":"pass","summary":"ok","criteria":{}}', stop_reason="end", block_types=("text",)),
    )
    cli_member = next(r for r in pr.DEFAULT_REVIEWERS if r.reviewer_id == "gemini")
    api_member = pr.api_reviewer(model="openrouter/moonshotai/kimi-k3")

    def runner(reviewer, prompt, prompt_path, ws):
        return '{"status":"pass","summary":"ok","criteria":{}}'

    report = pr.review_presentation(
        workspace=workspace, presentation_id="d", presentation_type="topic",
        requirements_path=workspace / "req.md", narration_script_path=workspace / "narr.md",
        screenshots_dir=shots, report_output=workspace / "report.json",
        reviewers=[cli_member, api_member], live=True,
        runner=lambda reviewer, prompt, prompt_path, ws: runner(reviewer, prompt, prompt_path, ws) if reviewer.transport == "cli" else pr._run_api_reviewer(reviewer, prompt=prompt, screenshots=[shots / "01.png"], workspace=ws),
    )
    assert report["reviewer_coverage"]["ran"] == 2
    assert {r["transport"] for r in report["reviewers"]} == {"cli", "api"}


# --- advisory second-opinion visual QC reaches the reviewers as hints ---------------------

def _prompt_with_artifacts(tmp_path, artifacts):
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    return build_review_prompt(
        workspace=workspace, presentation_id="01_demo", presentation_type="topic", topic=None,
        requirements_path=requirements, narration_script_path=narration,
        screenshots=sorted(screenshots.glob("*.png")),
        artifact_paths={label: workspace / rel for label, rel in artifacts.items()},
    ), workspace


def test_the_prompt_tells_reviewers_how_to_read_an_advisory_second_opinion(tmp_path) -> None:
    workspace, *_ = make_review_workspace(tmp_path)
    advisory = workspace / "decks" / "01_demo_topic" / "advisory_summary.json"
    write_text(advisory, '{"flagged_slides": [{"slide_number": 3}]}\n')
    prompt, _ = _prompt_with_artifacts(tmp_path, {"visual_qc_advisory": "decks/01_demo_topic/advisory_summary.json"})
    assert "ADVISORY SECOND OPINION" in prompt
    assert "Treat each finding as a hint" in prompt
    assert "## visual_qc_advisory" in prompt


def test_without_an_advisory_artifact_the_prompt_is_unchanged(tmp_path) -> None:
    prompt, _ = _prompt_with_artifacts(tmp_path, {})
    assert "ADVISORY SECOND OPINION" not in prompt


def test_the_review_cli_accepts_the_advisory_summary() -> None:
    import importlib.util
    import sys as _sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "presentations" / "review_presentation.py"
    spec = importlib.util.spec_from_file_location("review_presentation_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.ARTIFACT_ARG_LABELS["visual_qc_advisory"] == "visual_qc_advisory"
    args = module.parse_args([
        "--workspace", "/tmp", "--presentation-id", "d", "--presentation-type", "topic",
        "--requirements", "r.json", "--narration-script", "n.md", "--screenshots-dir", "s",
        "--report-output", "out.json", "--visual-qc-advisory", "adv.json",
    ])
    assert args.visual_qc_advisory is not None and args.visual_qc_advisory.name == "adv.json"


def test_gate_a_receives_claim_fidelity_as_unconfirmed_advice(tmp_path):
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    paths = []
    for phase in ('figures', 'narration'):
        path = requirements.parent / f'claim_fidelity_review_{phase}.json'
        path.write_text(json.dumps({'phase': phase, 'status': 'passed', 'finding_counts': {'blocker': 0, 'advisory': 1},
                                   'findings': [{'severity': 'advisory', 'code': 'ledger_mismatch',
                                                 'issue': f'{phase} title and note may describe different works.'}]}))
        paths.append(path)
    prompt = build_review_prompt(workspace=workspace, presentation_id='fixture', presentation_type='topic', topic=None,
                                 requirements_path=requirements, narration_script_path=narration,
                                 screenshots=[], claim_fidelity_report_paths=paths)
    assert 'ADVISORY CLAIM-FIDELITY FINDINGS — confirm or dismiss each; do not treat as established' in prompt
    assert 'LEDGER MISMATCH — verify source records first' in prompt
    assert 'figures title and note' in prompt and 'narration title and note' in prompt


def test_claim_fidelity_cli_option_is_repeatable_and_threaded(tmp_path, monkeypatch):
    import importlib.util
    import sys
    path = Path(__file__).resolve().parents[1] / 'scripts/presentations/review_presentation.py'
    spec = importlib.util.spec_from_file_location('review_presentation_calibration_cli', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    captured = {}
    def fake_review(**kwargs):
        captured.update(kwargs)
        return {'status': 'planned'}
    monkeypatch.setattr(module, 'review_presentation', fake_review)
    assert module.main(['--workspace', str(tmp_path), '--presentation-id', 'fixture', '--presentation-type', 'topic',
                        '--requirements', 'requirements.json', '--narration-script', 'script.md',
                        '--screenshots-dir', 'screenshots', '--report-output', 'review.json', '--dry-run',
                        '--claim-fidelity-report', 'figures.json', '--claim-fidelity-report', 'narration.json']) == 0
    assert captured['claim_fidelity_report_paths'] == [tmp_path / 'figures.json', tmp_path / 'narration.json']


def test_the_prompt_tells_reviewers_the_figure_brief_is_the_plan_not_the_render(tmp_path):
    # ICML 2026 deck 03: Gemini raised the blueprint's visual_figure_brief ("AgentLAB inside
    # AutoControl Arena") as a blocker four rounds after the rendered figure had separated them.
    from conference_pipeline.presentation.presentation_review import build_review_prompt
    req = tmp_path / "blueprint.json"; req.write_text("{}")
    nar = tmp_path / "narration.md"; nar.write_text("Hello.")
    prompt = build_review_prompt(workspace=tmp_path, presentation_id="p", presentation_type="topic-deep-dive",
                                 topic="p", requirements_path=req, narration_script_path=nar, screenshots=[])
    assert "visual_figure_brief" in prompt and "not the rendered figure" in prompt


def _finding_runner(findings_by_reviewer):
    def runner(reviewer, prompt, prompt_path, workspace_path):
        findings = findings_by_reviewer.get(reviewer.reviewer_id) or []
        if not findings:
            return passing_response()
        return json.dumps({"status": "fail", "summary": "Factual issues.",
                           "criteria": {c["id"]: {"status": "pass", "comments": ["ok"]} for c in REVIEW_CRITERIA},
                           "production_blockers": findings, "slide_comments": [], "suggested_fixes": [], "confidence": 0.8})
    return runner


def _review(workspace, requirements, narration, screenshots, runner):
    return review_presentation(
        workspace=workspace, presentation_id="01_demo_topic", presentation_type="topic-deep-dive",
        requirements_path=requirements, narration_script_path=narration, screenshots_dir=screenshots,
        report_output=workspace / "decks" / "01_demo_topic" / "presentation_review.json", live=True, runner=runner,
    )


def _fact(slide, issue="Wrong claim."):
    return {"category": "factual", "issue": issue, "slide": slide}


def test_a_new_finding_on_an_unchanged_settled_slide_is_recorded_not_blocking(tmp_path: Path) -> None:
    # ICML 2026 deck 04: every re-review after an edit elsewhere raised fresh items on untouched slides.
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    first = _review(workspace, requirements, narration, screenshots, _finding_runner({"codex_5_5": [_fact(2)]}))
    assert first["status"] == "failed" and first["settled_slides"] == []
    second = _review(workspace, requirements, narration, screenshots,
                     _finding_runner({"codex_5_5": [_fact(1, "A fresh item on slide 1."), _fact(2)]}))
    assert second["settled_slides"] == [1]
    blockers = second["aggregated_feedback"]["blockers"]
    assert [b["slide"] for b in blockers] == [2], "a slide that was blocked stays judged"
    demoted = [f for f in second["aggregated_feedback"]["recorded_non_blocking_findings"] if f.get("demoted_reason")]
    assert [f["slide"] for f in demoted] == [1] and "fresh item" in demoted[0]["comment"]


def test_a_changed_slide_is_judged_again(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    _review(workspace, requirements, narration, screenshots, _finding_runner({}))
    write_text(screenshots / "slide_001.png", "png changed\n")
    report = _review(workspace, requirements, narration, screenshots, _finding_runner({"codex_5_5": [_fact(1)]}))
    assert report["settled_slides"] == [2]
    assert report["status"] == "failed" and [b["slide"] for b in report["aggregated_feedback"]["blockers"]] == [1]


def test_an_unchanged_passed_deck_is_not_failed_by_a_re_review(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    assert _review(workspace, requirements, narration, screenshots, _finding_runner({}))["status"] == "passed"
    again = _review(workspace, requirements, narration, screenshots, _finding_runner({"claude_code_opus": [_fact(1), _fact(2)]}))
    assert again["status"] == "passed" and again["settled_slides"] == [1, 2]


def test_a_deck_level_blocker_settles_nothing(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    _review(workspace, requirements, narration, screenshots, _finding_runner({"codex_5_5": [{"category": "factual", "issue": "Deck-wide."}]}))
    report = _review(workspace, requirements, narration, screenshots, _finding_runner({"codex_5_5": [_fact(1)]}))
    assert report["settled_slides"] == [] and report["status"] == "failed"


def test_each_slides_own_narration_is_part_of_its_digest(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    deck = narration.parent
    write_text(deck / "presentation_deck.json", json.dumps({"slides": [{"slide_id": "sl_a"}, {"slide_id": "sl_b"}]}))
    raw = {"slides": [{"slide_id": "sl_a", "single": "A.", "two_host": []}, {"slide_id": "sl_b", "single": "B.", "two_host": []}]}
    write_text(deck / "engaging_narration_raw.json", json.dumps(raw))
    _review(workspace, requirements, narration, screenshots, _finding_runner({}))
    raw["slides"][1]["single"] = "B, rewritten."
    write_text(deck / "engaging_narration_raw.json", json.dumps(raw))
    report = _review(workspace, requirements, narration, screenshots, _finding_runner({"codex_5_5": [_fact(1), _fact(2)]}))
    assert report["settled_slides"] == [1]
    assert [b["slide"] for b in report["aggregated_feedback"]["blockers"]] == [2]


def test_a_report_without_slide_digests_bootstraps_from_screenshot_and_script_hashes(tmp_path: Path) -> None:
    workspace, requirements, narration, screenshots = make_review_workspace(tmp_path)
    first = _review(workspace, requirements, narration, screenshots, _finding_runner({}))
    report_path = workspace / "decks" / "01_demo_topic" / "presentation_review.json"
    legacy = {k: v for k, v in first.items() if k not in {"slide_digests", "settled_slides"}}
    report_path.write_text(json.dumps(legacy), encoding="utf-8")
    report = _review(workspace, requirements, narration, screenshots, _finding_runner({"codex_5_5": [_fact(2)]}))
    assert report["settled_slides"] == [1, 2] and report["status"] == "passed"
    report_path.write_text(json.dumps(legacy), encoding="utf-8")
    write_text(narration, "# Presenter Script\n\nSay something else.\n")
    changed = _review(workspace, requirements, narration, screenshots, _finding_runner({"codex_5_5": [_fact(2)]}))
    assert changed["settled_slides"] == [] and changed["status"] == "failed"
