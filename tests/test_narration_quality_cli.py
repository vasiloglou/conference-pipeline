from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "presentations"


def load_script(script_name: str, module_name: str) -> Any:
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_source_script(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "### [SLIDE 1] First",
                "",
                "**SAY:**",
                "The result is accurate but brittle.",
                "",
                '**TRANSITION:** "Next, compare the mitigation choices."',
            ]
        ),
        encoding="utf-8",
    )


def write_generated_script(path: Path, *, transition_marker: bool, final_question: bool) -> None:
    final_text = (
        "So is the benchmark actually enough?"
        if final_question
        else "The benchmark is the starting line, not deployment evidence."
    )
    lines = [
        "# Generated",
        "",
        "### [SLIDE 1] First",
        "",
        "**DIALOGUE:**",
        "- **Iapetus:** The result looks strong, but the source says it gets brittle after deployment.",
        "- **Erinome:** What should we watch for here?",
        f"- **Iapetus:** {final_text}",
    ]
    if transition_marker:
        lines.extend(["", '**TRANSITION MARKER:** Next, compare the mitigation choices.'])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_truncated_script(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Generated",
                "",
                "### [SLIDE 1] Broken",
                "",
                "**DIALOGUE:**",
                "- **Iapetus:** The big signal from ICLR 2026 is that th.",
                "- **Erinome:** So we're measuring 'operational readines.",
                "- **Iapetus:** The Liu et al.",
                "- **Erinome:** A complete sentence lands the idea.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_formulaic_script(path: Path) -> None:
    lines = ["# Generated", ""]
    for slide in range(1, 8):
        lines.extend(
            [
                f"### [SLIDE {slide}] Repeated",
                "",
                "**DIALOGUE:**",
                "- **Iapetus:** The source gives us a concrete failure mode.",
                "- **Erinome:** It changes the deployment threshold. What should we pay closest attention to here?",
                "- **Iapetus:** Exactly. The practical decision is whether the evidence changes the release gate.",
                "- **Erinome:** The observation is now tied to an action.",
                "",
                "**TRANSITION MARKER:** The next slide makes that action more concrete.",
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_series_figure_script(path: Path, *, title: str, body: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Generated",
                "",
                "### [SLIDE 0] Intro",
                "",
                "**DIALOGUE:**",
                "- **Iapetus:** The opening establishes the topic.",
                "- **Erinome:** The setup is clear.",
                "",
                "### [SLIDE 100001] " + title + " — Figure Detail",
                "",
                "**DIALOGUE:**",
                f"- **Iapetus:** {body}",
                "- **Erinome:** The chart makes that design target concrete for this topic.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_lint_narration_script_flags_dropped_transitions_and_question_endings(tmp_path: Path):
    module = load_script("lint_narration_script.py", "phase24_lint_narration")
    source = tmp_path / "6_presenter_script.md"
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    write_source_script(source)
    write_generated_script(generated, transition_marker=False, final_question=True)

    report = module.analyze_narration_script(
        script_md=generated,
        source_script_md=source,
        deck_kind="executive",
        conference_label="ICLR 2026",
    )

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert "dropped_transitions" in rules
    assert "question_ending_rate" in rules
    assert report["metrics"]["question_ending_rate"] == 1.0


def test_lint_narration_script_flags_truncated_and_hanging_fragments(tmp_path: Path):
    module = load_script("lint_narration_script.py", "phase24_lint_narration_truncation")
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    write_truncated_script(generated)

    report = module.analyze_narration_script(script_md=generated, deck_kind="topic")

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert "truncated_or_hanging_dialogue" in rules
    assert report["metrics"]["truncated_or_hanging_fragment_count"] == 3
    assert report["metrics"]["truncated_or_hanging_fragment_examples"][0]["line"] == 6


def test_lint_narration_script_allows_multiple_choice_endings(tmp_path: Path):
    module = load_script("lint_narration_script.py", "phase24_lint_narration_multiple_choice")
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    generated.write_text(
        "\n".join(
            [
                "# Generated",
                "",
                "### [SLIDE 1] Multiple Choice",
                "",
                "**DIALOGUE:**",
                "- **Iapetus:** The benchmark asks the model to create something new, not just pick A, B, or C.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.analyze_narration_script(script_md=generated, deck_kind="topic")

    rules = {finding["rule"] for finding in report["findings"]}
    assert "truncated_or_hanging_dialogue" not in rules
    assert report["metrics"]["truncated_or_hanging_fragment_count"] == 0


def test_lint_narration_script_allows_spoken_idioms_and_letter_endings(tmp_path: Path):
    module = load_script("lint_narration_script.py", "phase24_lint_narration_spoken_endings")
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    generated.write_text(
        "\n".join(
            [
                "# Generated",
                "",
                "### [SLIDE 1] Spoken Endings",
                "",
                "**DIALOGUE:**",
                "- **Iapetus:** Some failure modes sound abstract, but the concrete cases are anything but.",
                "- **Erinome:** The analysis is why this was such a mystery for a while.",
                "- **Iapetus:** That is a much higher bar than single-turn Q&A.",
                "- **Erinome:** The relational path gets lost on the way from A to B.",
                "- **Iapetus:** It chooses the reliable B+ over a lottery ticket that might also be an F.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.analyze_narration_script(script_md=generated, deck_kind="topic")

    rules = {finding["rule"] for finding in report["findings"]}
    assert "truncated_or_hanging_dialogue" not in rules
    assert report["metrics"]["truncated_or_hanging_fragment_count"] == 0


def test_lint_narration_script_flags_formulaic_attention_and_agreement(tmp_path: Path):
    module = load_script("lint_narration_script.py", "phase24_lint_narration_formula")
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    write_formulaic_script(generated)

    report = module.analyze_narration_script(script_md=generated, deck_kind="topic")

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert "formulaic_attention_question" in rules
    assert "formulaic_agreement_opener" in rules
    assert report["metrics"]["formulaic_attention_question_count"] == 7
    assert report["metrics"]["exactly_opener_count"] == 7


def test_lint_narration_script_requires_executive_transition_markers(tmp_path: Path):
    module = load_script("lint_narration_script.py", "phase24_lint_narration_exec_transitions")
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    generated.write_text(
        "\n".join(
            [
                "# Generated",
                "",
                "### [SLIDE 0] Intro",
                "",
                "**DIALOGUE:**",
                "- **Iapetus:** The summary opens with the operating frame.",
                "- **Erinome:** The frame is clear.",
                "",
                "### [SLIDE 1] Next",
                "",
                "**DIALOGUE:**",
                "- **Iapetus:** The next slide turns that frame into a decision.",
                "- **Erinome:** The decision is concrete.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.analyze_narration_script(script_md=generated, deck_kind="executive")

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert "missing_executive_transition_markers" in rules
    assert report["metrics"]["expected_executive_transition_count"] == 1


def test_review_narration_script_blocks_tts_and_writes_artifacts(tmp_path: Path):
    lint_module = load_script("lint_narration_script.py", "phase24_lint_narration_for_review")
    review_module = load_script("review_narration_script.py", "phase24_review_narration")
    source = tmp_path / "6_presenter_script.md"
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    lint_report = tmp_path / "narration_lint_report.json"
    review_md = tmp_path / "narration_review.md"
    review_json = tmp_path / "narration_review.json"
    write_source_script(source)
    write_generated_script(generated, transition_marker=False, final_question=True)
    lint_report.write_text(
        json.dumps(
            lint_module.analyze_narration_script(
                script_md=generated,
                source_script_md=source,
                deck_kind="executive",
                conference_label="ICLR 2026",
            )
        ),
        encoding="utf-8",
    )

    exit_code = review_module.main(
        [
            "--script-md",
            str(generated),
            "--source-script-md",
            str(source),
            "--lint-report",
            str(lint_report),
            "--deck-kind",
            "executive",
            "--review-out",
            str(review_md),
            "--json-out",
            str(review_json),
        ]
    )

    payload = json.loads(review_json.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["approve_for_tts"] is False
    assert payload["accuracy_blockers"]
    assert payload["listener_blockers"]
    assert "Approve for TTS:** no" in review_md.read_text(encoding="utf-8")


def test_review_narration_script_approves_clean_deterministic_sample(tmp_path: Path):
    review_module = load_script("review_narration_script.py", "phase24_review_narration_clean")
    source = tmp_path / "6_presenter_script.md"
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    write_source_script(source)
    write_generated_script(generated, transition_marker=True, final_question=False)

    review = review_module.review_narration_script(
        script_md=generated,
        source_script_md=source,
        deck_kind="topic",
        conference_label="ICLR 2026",
    )

    assert review["approve_for_tts"] is True
    assert not review["accuracy_blockers"]
    assert not review["listener_blockers"]


def test_fix_narration_script_repairs_markdown_and_script_json(tmp_path: Path):
    fix_module = load_script("fix_narration_script.py", "phase24_fix_narration")
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    script_json = tmp_path / "script.json"
    generated.write_text(
        "\n".join(
            [
                "# Generated",
                "",
                "### [SLIDE 0] Intro",
                "",
                "**DIALOGUE:**",
                "- **Iapetus:** The source gives us a concrete failure mode.",
                "- **Erinome:** It changes the deployment threshold. What should we pay closest attention to here?",
                "- **Iapetus:** Exactly. The practical decision is whether the evidence changes the release gate.",
                "- **Erinome:** The observation is now tied to an action.",
                "",
                "---",
                "",
                "### [SLIDE 1] Next",
                "",
                "**DIALOGUE:**",
                "- **Iapetus:** Precisely. The next point is concrete.",
                "- **Erinome:** The summary keeps moving.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_json.write_text(
        json.dumps(
            {
                "hosts": {"host_a": "Iapetus", "host_b": "Erinome"},
                "segments": [
                    {
                        "index": 1,
                        "speaker": "Erinome",
                        "text": "It changes the deployment threshold. What should we pay closest attention to here?",
                        "slide_number": 0,
                        "slide_title": "Intro",
                    },
                    {
                        "index": 2,
                        "speaker": "Iapetus",
                        "text": "Exactly. The practical decision is whether the evidence changes the release gate.",
                        "slide_number": 0,
                        "slide_title": "Intro",
                    },
                ],
                "metadata": {
                    "slides": [
                        {
                            "slide_number": 0,
                            "title": "Intro",
                            "turns": [
                                {
                                    "speaker": "Erinome",
                                    "text": "It changes the deployment threshold. What should we pay closest attention to here?",
                                }
                            ],
                        },
                        {"slide_number": 1, "title": "Next", "turns": []},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = fix_module.main(
        [
            "--script-md",
            str(generated),
            "--script-json",
            str(script_json),
            "--deck-kind",
            "executive",
        ]
    )

    assert exit_code == 0
    repaired_md = generated.read_text(encoding="utf-8")
    repaired_json = json.loads(script_json.read_text(encoding="utf-8"))
    assert "What should we pay closest attention to here?" not in repaired_md
    assert "Exactly." not in repaired_md
    assert "Precisely." not in repaired_md
    assert "**TRANSITION MARKER:**" in repaired_md
    assert "What should we pay closest attention to here?" not in json.dumps(repaired_json)
    assert repaired_json["metadata"]["slides"][0]["transition"]


def test_fix_narration_script_thins_conference_label_in_markdown_and_json(tmp_path: Path):
    fix_module = load_script("fix_narration_script.py", "phase24_fix_narration_label")
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    script_json = tmp_path / "script.json"
    lines = ["# Generated", ""]
    for slide in range(4):
        lines.extend(
            [
                f"### [SLIDE {slide}] Slide {slide}",
                "",
                "**DIALOGUE:**",
                f"- **Iapetus:** ICLR 2026 shows a concrete result on slide {slide}.",
                "- **Erinome:** The result changes the operating question.",
                "",
                "---",
                "",
            ]
        )
    generated.write_text("\n".join(lines), encoding="utf-8")
    script_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "index": slide,
                        "speaker": "Iapetus",
                        "text": f"ICLR 2026 shows a concrete result on slide {slide}.",
                        "slide_number": slide,
                    }
                    for slide in range(4)
                ],
                "metadata": {
                    "slides": [
                        {
                            "slide_number": slide,
                            "title": f"Slide {slide}",
                            "turns": [
                                {
                                    "speaker": "Iapetus",
                                    "text": f"ICLR 2026 shows a concrete result on slide {slide}.",
                                }
                            ],
                        }
                        for slide in range(4)
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = fix_module.main(
        [
            "--script-md",
            str(generated),
            "--script-json",
            str(script_json),
            "--deck-kind",
            "topic",
            "--conference-label",
            "ICLR 2026",
            "--conference-label-limit",
            "1",
            "--conference-label-preserve-every-slides",
            "100",
        ]
    )

    assert exit_code == 0
    repaired_md = generated.read_text(encoding="utf-8")
    repaired_json = json.loads(script_json.read_text(encoding="utf-8"))
    assert repaired_md.count("ICLR 2026") == 2
    assert "the conference shows a concrete result on slide 1" in repaired_md
    repaired_json_text = json.dumps(repaired_json)
    assert repaired_json_text.count("ICLR 2026") == 4
    assert "the conference shows a concrete result on slide 1" in repaired_json_text


def test_review_narration_series_flags_cross_deck_figure_repetition(tmp_path: Path):
    series_module = load_script("review_narration_series.py", "phase24_review_narration_series")
    decks = tmp_path / "decks"
    repeated = (
        "This scaling pressure curve shows the same bottleneck: as context grows from 4K to 128K, "
        "the red line gets steeper and the design target is to flatten the curve with memory-aware architecture."
    )
    for index in range(1, 5):
        deck = decks / f"topic_{index:02d}"
        deck.mkdir(parents=True)
        write_series_figure_script(deck / "6_presenter_script_notebooklm_2host.md", title="Scaling Pressure", body=repeated)

    report = series_module.review_narration_series(
        decks_dir=decks,
        conference_label="ICLR 2026",
        phrase_warn_decks=3,
        phrase_fail_decks=4,
        phrase_occurrence_threshold=3,
    )

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert report["approve_for_tts"] is False
    assert "cross_deck_repeated_figure_phrase" in rules


def test_review_narration_series_can_scan_single_presenter_say_blocks(tmp_path: Path):
    series_module = load_script("review_narration_series.py", "phase24_review_presenter_series")
    decks = tmp_path / "decks"
    repeated = (
        "This scaling pressure curve shows the same bottleneck as context grows from 4K to 128K, "
        "and the target is to flatten the curve with memory-aware architecture."
    )
    for index in range(1, 5):
        deck = decks / f"topic_{index:02d}"
        deck.mkdir(parents=True)
        (deck / "6_presenter_script.md").write_text(
            "\n".join(
                [
                    "# Presenter",
                    "",
                    "### [ZOOM 100001] Scaling Pressure - Figure Detail",
                    "",
                    "**SAY:**",
                    f"> {repeated}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    report = series_module.review_narration_series(
        decks_dir=decks,
        script_name="6_presenter_script.md",
        conference_label="ICLR 2026",
        phrase_warn_decks=3,
        phrase_fail_decks=4,
        phrase_occurrence_threshold=3,
    )

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert report["metrics"]["script_name"] == "6_presenter_script.md"
    assert "cross_deck_repeated_figure_phrase" in rules


def test_fix_narration_series_repairs_markdown_and_json(tmp_path: Path):
    fix_module = load_script("fix_narration_series.py", "phase24_fix_narration_series")
    decks = tmp_path / "decks"
    repeated = (
        "This scaling pressure curve shows the same bottleneck: as context grows from 4K to 128K, "
        "the design target is to flatten the curve with memory-aware architectures."
    )
    deck = decks / "topic_01"
    deck.mkdir(parents=True)
    generated = deck / "6_presenter_script_notebooklm_2host.md"
    write_series_figure_script(generated, title="Scaling Pressure", body=repeated)
    script_json = deck / "video_build_notebooklm" / "script.json"
    script_json.parent.mkdir(parents=True)
    script_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "index": 1,
                        "speaker": "Iapetus",
                        "text": repeated,
                        "slide_number": 100001,
                        "slide_title": "Scaling Pressure - Figure Detail",
                    }
                ],
                "metadata": {
                    "slides": [
                        {
                            "slide_number": 100001,
                            "title": "Scaling Pressure - Figure Detail",
                            "turns": [{"speaker": "Iapetus", "text": repeated}],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = fix_module.main(["--decks-dir", str(decks)])

    assert exit_code == 0
    repaired_md = generated.read_text(encoding="utf-8")
    repaired_json_text = script_json.read_text(encoding="utf-8")
    assert "scaling pressure curve" not in repaired_md.lower()
    assert "the design target" not in repaired_md.lower()
    assert "flatten the curve" not in repaired_md.lower()
    assert "memory-aware architectures" not in repaired_md.lower()
    assert "scaling pressure curve" not in repaired_json_text.lower()
    assert "the design target" not in repaired_json_text.lower()


def test_lint_presenter_narration_flags_script_json_leaks(tmp_path: Path):
    module = load_script("lint_presenter_narration.py", "phase24_lint_presenter_narration")
    script_md = tmp_path / "6_presenter_script.md"
    script_json = tmp_path / "script.json"
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 1] Demo",
                "",
                "**SAY:**",
                "> The spoken line is clean.",
                "",
                '**TRANSITION:** "That sets up the next slide."',
            ]
        ),
        encoding="utf-8",
    )
    script_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "Presenter",
                        "text": "The spoken line is clean. **CONTEXT** (not spoken -- for presenter reference): Paper: Demo [arXiv]",
                    }
                ],
                "metadata": {"slides": [{"slide_number": 1, "title": "Demo", "transition": None}]},
            }
        ),
        encoding="utf-8",
    )

    report = module.analyze_presenter_narration(
        script_md=script_md,
        script_json=script_json,
        deck_kind="topic",
        conference_label="ICLR 2026",
    )

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert "tts_forbidden_markdown_context_marker" in rules
    assert "dropped_presenter_transitions" in rules


def test_lint_presenter_narration_flags_source_markdown_paths_in_json(tmp_path: Path):
    module = load_script("lint_presenter_narration.py", "phase24_lint_presenter_source_paths")
    script_md = tmp_path / "6_presenter_script.md"
    script_json = tmp_path / "script.json"
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 1] Demo",
                "",
                "**SAY:**",
                "> The evaluation report changes the release gate.",
            ]
        ),
        encoding="utf-8",
    )
    script_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "Presenter",
                        "text": "analysis/03_evaluation_benchmarks_data.md changes the release gate.",
                    }
                ],
                "metadata": {"slides": [{"slide_number": 1, "title": "Demo"}]},
            }
        ),
        encoding="utf-8",
    )

    report = module.analyze_presenter_narration(
        script_md=script_md,
        script_json=script_json,
        deck_kind="executive_short",
    )

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert "tts_forbidden_source_markdown_path" in rules


def test_review_presenter_narration_approves_clean_json(tmp_path: Path):
    review_module = load_script("review_presenter_narration.py", "phase24_review_presenter_narration")
    script_md = tmp_path / "6_presenter_script.md"
    script_json = tmp_path / "script.json"
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 0] Executive Summary",
                "",
                "**SAY:**",
                "> The leadership readout starts with the operating thesis.",
                "",
                '**TRANSITION:** "That sets up the evidence map."',
            ]
        ),
        encoding="utf-8",
    )
    script_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "Presenter",
                        "text": "The leadership readout starts with the operating thesis. That sets up the evidence map.",
                    }
                ],
                "metadata": {
                    "slides": [
                        {
                            "slide_number": 0,
                            "title": "Executive Summary",
                            "transition": "That sets up the evidence map.",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    review = review_module.review_presenter_narration(
        script_md=script_md,
        script_json=script_json,
        deck_kind="executive",
        conference_label="ICLR 2026",
    )

    assert review["approve_for_tts"] is True
    assert not review["accuracy_blockers"]


def test_lint_presenter_narration_flags_executive_stitch_template(tmp_path: Path):
    module = load_script("lint_presenter_narration.py", "phase24_lint_presenter_stitch")
    script_md = tmp_path / "6_presenter_script.md"
    script_json = tmp_path / "script.json"
    stitched = (
        "Treat frontier progress as a system. That connects directly to measure operational readiness. "
        "The operational test is prioritize reliability. That sets up Eight Signals. That sets up Reliability."
    )
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 1] Executive",
                "",
                "**SAY:**",
                f"> {stitched}",
                "",
                '**TRANSITION:** "That sets up Eight Signals."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_json.write_text(
        json.dumps(
            {
                "segments": [{"speaker": "Presenter", "text": f"{stitched} That sets up Eight Signals."}],
                "metadata": {"slides": [{"slide_number": 1, "title": "Executive", "transition": "That sets up Eight Signals."}]},
            }
        ),
        encoding="utf-8",
    )

    report = module.analyze_presenter_narration(
        script_md=script_md,
        script_json=script_json,
        deck_kind="executive_short",
    )

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert "stitch_template_repetition" in rules
    assert "imperative_glue_template" in rules


def test_lint_presenter_narration_flags_closing_grammar_glue(tmp_path: Path):
    module = load_script("lint_presenter_narration.py", "phase24_lint_presenter_closing_glue")
    script_md = tmp_path / "6_presenter_script.md"
    script_json = tmp_path / "script.json"
    broken = "The next step is We need to build causal intervention tools before deployment."
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 1] Closing",
                "",
                "**SAY:**",
                f"> {broken}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_json.write_text(
        json.dumps(
            {
                "segments": [{"speaker": "Presenter", "text": broken}],
                "metadata": {"slides": [{"slide_number": 1, "title": "Closing"}]},
            }
        ),
        encoding="utf-8",
    )

    report = module.analyze_presenter_narration(
        script_md=script_md,
        script_json=script_json,
        deck_kind="topic",
    )

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert "closing_grammar_glue" in rules


def test_fix_presenter_narration_removes_formula_and_regenerates_json(tmp_path: Path):
    fix_module = load_script("fix_presenter_narration.py", "phase24_fix_presenter_narration")
    script_md = tmp_path / "6_presenter_script.md"
    script_json = tmp_path / "script.json"
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 1] Demo",
                "",
                "**SAY:**",
                "> Welcome everyone. The result changes the release gate.",
                "",
                "**CONTEXT** (not spoken -- for presenter reference):",
                "- Paper: Demo (Ada, 2026) [arXiv]",
                "",
                '**TRANSITION:** "That sets up the evidence map."',
            ]
        ),
        encoding="utf-8",
    )

    exit_code = fix_module.main(
        [
            "--script-md",
            str(script_md),
            "--script-json",
            str(script_json),
            "--title",
            "Demo",
        ]
    )

    assert exit_code == 0
    repaired_md = script_md.read_text(encoding="utf-8")
    repaired_json = script_json.read_text(encoding="utf-8")
    assert "Welcome everyone" not in repaired_md
    assert "Welcome everyone" not in repaired_json
    assert "**CONTEXT**" not in repaired_json
    assert "[arXiv]" not in repaired_json
    assert "The result changes the release gate. Next, the evidence map." in repaired_json


def test_fix_presenter_narration_rewrites_stitch_and_thins_filler(tmp_path: Path):
    fix_module = load_script("fix_presenter_narration.py", "phase24_fix_presenter_stitch")
    script_md = tmp_path / "6_presenter_script.md"
    script_json = tmp_path / "script.json"
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 1] Executive",
                "",
                "**SAY:**",
                "> Welcome everyone. Treat frontier progress as a system. That connects directly to measure operational readiness. The operational test is prioritize reliability. Critical evidence makes the critical path critical for critical teams with critical risks and critical decisions.",
                "",
                '**TRANSITION:** "That sets up Eight Signals."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = fix_module.main(
        [
            "--script-md",
            str(script_md),
            "--script-json",
            str(script_json),
            "--filler-limit",
            "2",
        ]
    )

    repaired_md = script_md.read_text(encoding="utf-8")
    repaired_json = script_json.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "Welcome everyone" not in repaired_md
    assert "That connects directly to" not in repaired_md
    assert "The operational test is prioritize" not in repaired_md
    assert "That also means measuring operational readiness" in repaired_md
    assert "whether teams can prioritize reliability" in repaired_md
    assert repaired_md.count("Critical") + repaired_md.count("critical") == 2
    assert 'Next, Eight Signals.' in repaired_md
    assert "That connects directly to" not in repaired_json
    assert "Welcome everyone" not in repaired_json


def test_fix_presenter_narration_repairs_closing_grammar_glue(tmp_path: Path):
    fix_module = load_script("fix_presenter_narration.py", "phase24_fix_presenter_closing_glue")
    script_md = tmp_path / "6_presenter_script.md"
    script_json = tmp_path / "script.json"
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 1] Closing",
                "",
                "**SAY:**",
                "> The next research phase is We need fundamental architectural solutions. The next step is We need to build causal intervention tools. The hard part now is First, we need to scale these simulators.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = fix_module.main(
        [
            "--script-md",
            str(script_md),
            "--script-json",
            str(script_json),
        ]
    )

    repaired_md = script_md.read_text(encoding="utf-8")
    repaired_json = script_json.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "is We need" not in repaired_md
    assert "is First, we need" not in repaired_md
    assert "The next research phase needs fundamental architectural solutions" in repaired_md
    assert "The next step is to build causal intervention tools" in repaired_md
    assert "The hard part now is to first scale these simulators" in repaired_md
    assert "is We need" not in repaired_json


def test_lint_narration_script_flags_locator_references(tmp_path: Path):
    module = load_script("lint_narration_script.py", "phase24_lint_narration_locators")
    generated = tmp_path / "6_presenter_script_notebooklm_2host.md"
    generated.write_text(
        "\n".join(
            [
                "# Generated",
                "",
                "### [SLIDE 1] Intro",
                "",
                "**DIALOGUE:**",
                "- **Iapetus:** What did the field learn this year?",
                "- **Erinome:** This poster reframes the whole question around deployment.",
                "- **Iapetus:** And the numbers?",
                "- **Erinome:** Poster 63306 reports a 28-point gap.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.analyze_narration_script(script_md=generated, deck_kind="topic")

    rules = {finding["rule"] for finding in report["findings"]}
    assert report["status"] == "failed"
    assert "locator_reference" in rules
    assert report["metrics"]["locator_reference_count"] == 2

    generated.write_text(
        generated.read_text(encoding="utf-8")
        .replace("This poster reframes", "The Peer-Preservation paper reframes")
        .replace("Poster 63306 reports", "The poster session favourite, Peer-Preservation, reports"),
        encoding="utf-8",
    )
    report = module.analyze_narration_script(script_md=generated, deck_kind="topic")
    assert "locator_reference" not in {finding["rule"] for finding in report["findings"]}


def test_poster_child_idiom_is_not_a_locator_reference():
    module = load_script("lint_narration_script.py", "lint_narration_script_idiom")
    assert not module.LOCATOR_REFERENCE_RE.search(
        "for years the poster child of an adversarial attack was that panda"
    )
    assert module.LOCATOR_REFERENCE_RE.search("this poster shows a panda")


def test_fixer_repairs_engaging_narration_raw_json_the_tts_source(tmp_path: Path) -> None:
    """engaging_narration_raw.json keeps slides at the top level with turns
    under "two_host"; the fixer read only metadata.slides[].turns, so its
    repairs reached the reviewed markdown but never the spoken delivery."""

    module = load_script("fix_narration_script.py", "fix_narration_script_raw")
    raw = {
        "slides": [
            {"slide_number": i, "slide_id": f"sl_{i:02d}", "single": "Exactly. It works.",
             "two_host": [{"speaker": "A", "text": "So it works?"}, {"speaker": "B", "text": "Exactly. It does."}]}
            for i in range(1, 9)
        ]
    }
    stats = module.RepairStats()
    repaired = module.repair_script_json(raw, deck_kind="topic", replacements=[], stats=stats)
    assert stats.json_turns_changed > 0
    openers = [t["text"].split()[0] for s in repaired["slides"] for t in s["two_host"]]
    assert openers.count("Exactly.") < 8


def test_sync_raw_from_md_carries_text_fixes_and_refuses_structural_edits() -> None:
    import pytest

    module = load_script("generate_engaging_narration.py", "generate_engaging_narration_sync")
    raw = {"slides": [{"slide_number": 1, "slide_id": "sl_01", "single": "One read.",
                       "two_host": [{"speaker": "A", "text": "So?"}, {"speaker": "B", "text": "Exactly. Yes."}]}]}
    md = module.render_two_host_md(raw).replace("Exactly. Yes.", "Yes, it does.")
    assert module.sync_raw_from_md(raw, md, module.render_single_md(raw)) == 1
    assert raw["slides"][0]["two_host"][1]["text"] == "Yes, it does."
    dropped_turn = "\n".join(line for line in md.splitlines() if "**A:**" not in line)
    with pytest.raises(ValueError, match="structurally"):
        module.sync_raw_from_md(raw, dropped_turn)
