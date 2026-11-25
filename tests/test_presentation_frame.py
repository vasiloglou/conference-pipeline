from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from conference_pipeline.presentation.presentation_frame import (
    load_presentation_frame_file,
    validate_opening_area_generality,
    validate_presentation_frame_payload,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_editorial_module():
    spec = importlib.util.spec_from_file_location(
        "generate_editorial_blueprint_test",
        REPO_ROOT / "scripts" / "presentations" / "generate_editorial_blueprint.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame() -> dict[str, object]:
    return {
        "schema_version": 1,
        "presentation_id": "05_generative_models",
        "topic_id": "05_generative_models",
        "retrieval_title": "Generative modeling: diffusion, flow, and controllability",
        "audience_title": "Generative AI after sample quality",
        "field_thesis": (
            "The evidence shows generative AI becoming operational infrastructure: "
            "controllable, efficient, auditable, provenance-aware, and valid under domain constraints."
        ),
        "general_questions": [
            "What changed once sample quality stopped being the only scoreboard?",
            "Where do generative systems still fail when outputs look plausible?",
            "Which parts of the stack carry the burden of validity and provenance?",
            "Why are diffusion and flow mechanisms inside the story, not the public frame?",
        ],
        "opening_anchor_policy": {
            "rules": [
                "Open on field-level claims before method-local mechanisms.",
                "Do not let unresolved technical artifacts dominate the first section.",
            ]
        },
        "source_analysis_path": "analysis/05_generative_models.md",
        "analysis_evidence": {
            "executive_summary_terms": ["operational", "validity", "provenance"],
        },
    }


def test_presentation_frame_requires_audience_contract_fields() -> None:
    result = validate_presentation_frame_payload(
        {
            "schema_version": 1,
            "presentation_id": "05_generative_models",
            "topic_id": "05_generative_models",
        }
    )

    assert result.status == "fail"
    assert {error.code for error in result.errors} >= {
        "presentation_frame_missing_field",
        "presentation_frame_questions_missing",
        "presentation_frame_anchor_policy_missing",
    }


def test_missing_presentation_frame_is_only_allowed_when_optional(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "presentation_frame.json"

    assert load_presentation_frame_file(missing) is None
    with pytest.raises(FileNotFoundError, match="presentation_frame.json not found"):
        load_presentation_frame_file(missing, required=True)


def test_opening_area_generality_rejects_mechanism_only_opening() -> None:
    blueprint = {
        "slides": [
            {
                "title": "Diffusion and flow as one transport equation",
                "claims": ["Score, velocity, PDE, Feynman-Kac, copula, and SDE samplers unify generation."],
                "visible_text": ["Score field", "Velocity field", "PDE flow", "Copula transport"],
                "narration_beats": ["Diffusion, flow, Gaussian denoising, ODE transport, and distillation."],
            }
        ]
    }

    result = validate_opening_area_generality(blueprint)

    assert result.status == "fail"
    assert {error.code for error in result.errors} >= {
        "opening_frame_too_specialist",
        "opening_frame_questions_missing",
    }


def test_opening_area_generality_counts_repeated_specialist_terms() -> None:
    blueprint = {
        "slides": [
            {
                "title": "Mechanism loop",
                "claims": [
                    "Diffusion? diffusion? flow? flow? PDE? PDE? ODE? SDE?"
                ],
                "visible_text": ["diffusion", "diffusion", "flow", "flow"],
                "narration_beats": ["diffusion diffusion flow flow PDE ODE SDE sampler"],
            }
        ]
    }

    result = validate_opening_area_generality(blueprint)

    assert result.status == "fail"
    assert "opening_frame_too_specialist" in {error.code for error in result.errors}


def test_opening_area_generality_uses_frame_title_and_questions() -> None:
    frame = _frame()
    blueprint = {
        "slides": [
            {
                "title": "Generative AI after sample quality",
                "claims": [frame["field_thesis"]],
                "visible_text": frame["general_questions"],
                "narration_beats": [
                    "The field is shifting from sample quality to control, efficiency, validity, and provenance."
                ],
                "speaker_author_work_mentions": [{"source_id": "speech_transcript:broad"}],
            }
        ]
    }
    ledger = {
        "entries": [
            {
                "source_id": "speech_transcript:broad",
                "resolution_status": "resolved",
            }
        ]
    }

    result = validate_opening_area_generality(
        blueprint,
        presentation_frame=frame,
        ledger=ledger,
    )

    assert result.status == "pass"


def test_presentation_frame_ids_must_match_requested_deck_and_topic() -> None:
    module = _load_editorial_module()

    assert module.presentation_frame_request_mismatches(
        _frame(),
        presentation_id="05_generative_models",
        topic_id="05_generative_models",
    ) == []
    mismatches = module.presentation_frame_request_mismatches(
        _frame(),
        presentation_id="06_multimodal_spatial_vlm",
        topic_id="05_generative_models",
    )

    assert mismatches == [
        "presentation_id='05_generative_models' expected '06_multimodal_spatial_vlm'"
    ]


def test_editorial_source_catalog_prioritizes_named_resolved_sources() -> None:
    module = _load_editorial_module()
    catalog = module.build_source_catalog(
        {
            "entries": [
                {
                    "source_id": "speech_transcript:missing",
                    "resolved_title": "High confidence artifact with missing people",
                    "authors": [],
                    "speakers": [],
                    "contribution_note": "Tempting but unresolved opening anchor.",
                    "confidence": 1.0,
                    "status": "missing_people",
                    "visible_source_chip_eligible": True,
                },
                {
                    "source_id": "paper_abstract:named",
                    "resolved_title": "Lower confidence named source",
                    "authors": ["A. Researcher"],
                    "speakers": [],
                    "contribution_note": "Resolved field-level evidence.",
                    "confidence": 0.4,
                    "status": "resolved",
                    "visible_source_chip_eligible": True,
                },
            ]
        }
    )

    assert catalog[0]["source_id"] == "paper_abstract:named"
    assert catalog[0]["has_named_people"] is True
    assert catalog[-1]["resolution_status"] == "missing_people"


def test_editorial_source_catalog_preserves_distinct_transcript_artifacts() -> None:
    module = _load_editorial_module()
    broad_title = "ML4RS talks and panels on foundation models, flood mapping, and remote sensing"
    catalog = module.build_source_catalog(
        {
            "entries": [
                {
                    "source_id": "speech_transcript:artifact_a:segment:00034",
                    "resolved_title": broad_title,
                    "authors": [],
                    "speakers": [],
                    "contribution_note": "First workshop artifact.",
                    "confidence": 1.0,
                    "status": "missing_people",
                    "visible_source_chip_eligible": True,
                },
                {
                    "source_id": "speech_transcript:artifact_a:segment:00097",
                    "resolved_title": broad_title,
                    "authors": [],
                    "speakers": [],
                    "contribution_note": "Same workshop artifact later segment.",
                    "confidence": 1.0,
                    "status": "missing_people",
                    "visible_source_chip_eligible": True,
                },
                {
                    "source_id": "speech_transcript:artifact_b:segment:00102",
                    "resolved_title": broad_title,
                    "authors": [],
                    "speakers": [],
                    "contribution_note": "Separate workshop artifact with the same broad label.",
                    "confidence": 1.0,
                    "status": "missing_people",
                    "visible_source_chip_eligible": True,
                },
            ]
        }
    )

    assert [item["source_id"] for item in catalog] == [
        "speech_transcript:artifact_a:segment:00034",
        "speech_transcript:artifact_b:segment:00102",
    ]


def test_editorial_source_catalog_recovers_speakers_from_talk_titles() -> None:
    module = _load_editorial_module()
    catalog = module.build_source_catalog(
        {
            "entries": [
                {
                    "source_id": "speech_transcript:peace:segment:00038",
                    "resolved_title": "Alex Hanna, 'The Uncontroversial Thingness of Computer Vision', AI for Peace workshop",
                    "authors": [],
                    "speakers": [],
                    "contribution_note": "Data labor and computer vision history.",
                    "confidence": 1.0,
                    "status": "missing_people",
                    "visible_source_chip_eligible": True,
                },
                {
                    "source_id": "speech_transcript:postagi:segment:00006",
                    "resolved_title": "Lewis Kirsch and Jeff Clune, RSI and Post-AGI Science and Society workshops",
                    "authors": [],
                    "speakers": [],
                    "contribution_note": "Recursive self-improvement and open-ended AI-generating algorithms.",
                    "confidence": 1.0,
                    "status": "missing_people",
                    "visible_source_chip_eligible": True,
                },
            ]
        }
    )

    people_by_source = {item["source_id"]: item["people"] for item in catalog}
    assert people_by_source["speech_transcript:peace:segment:00038"] == ["Alex Hanna"]
    assert people_by_source["speech_transcript:postagi:segment:00006"] == [
        "Lewis Kirsch",
        "Jeff Clune",
    ]


def test_opening_anchor_rebalancer_adds_named_resolved_sources() -> None:
    module = _load_editorial_module()
    slides = [
        {
            "section_id": "S0_frame",
            "speaker_author_work_mentions": [
                {"source_id": "speech_transcript:missing_a", "label": "SRC-001"},
                {"source_id": "speech_transcript:missing_b", "label": "SRC-002"},
                {
                    "source_id": "paper_abstract:resolved_a",
                    "label": "SRC-003",
                    "authors": ["A. Researcher"],
                },
            ],
            "source_plan": [],
        }
    ]
    mention_plan = [
        {
            "source_id": "paper_abstract:resolved_b",
            "label": "SRC-004",
            "authors": ["B. Researcher"],
            "speakers": [],
        }
    ]
    ledger_index = {
        "speech_transcript:missing_a": {"status": "missing_people"},
        "speech_transcript:missing_b": {"status": "missing_people"},
        "paper_abstract:resolved_a": {"status": "resolved"},
        "paper_abstract:resolved_b": {"status": "resolved"},
    }

    module._rebalance_opening_resolved_anchors(
        slides,
        mention_plan=mention_plan,
        ledger_index=ledger_index,
    )

    opening_ids = {
        mention["source_id"]
        for mention in slides[0]["speaker_author_work_mentions"]
    }
    assert "paper_abstract:resolved_b" in opening_ids
    assert slides[0]["source_plan"][-1]["role"] == "field-level opening anchor"


def test_opening_area_generality_requires_questions_in_opening_text() -> None:
    frame = _frame()
    blueprint = {
        "slides": [
            {
                "title": "Generative AI after sample quality",
                "claims": [frame["field_thesis"]],
                "visible_text": ["Control", "Efficiency", "Validity", "Provenance"],
                "narration_beats": [
                    "The field is shifting from sample quality to operational infrastructure."
                ],
            }
        ]
    }

    result = validate_opening_area_generality(blueprint, presentation_frame=frame)

    assert result.status == "fail"
    assert "opening_frame_questions_missing" in {error.code for error in result.errors}


def test_coverage_stats_counts_content_slides_not_title_slide() -> None:
    module = _load_editorial_module()
    stats = module.coverage_stats(
        {
            "slides": [
                {"section_id": "S0_frame", "speaker_author_work_mentions": []},
                {"section_id": "S1", "speaker_author_work_mentions": []},
                {"section_id": "S2", "speaker_author_work_mentions": []},
            ]
        }
    )

    assert stats["slides"] == 2
    assert stats["total_slides"] == 3


def test_compile_blueprint_prefers_audience_title_when_frame_present() -> None:
    module = _load_editorial_module()
    args = SimpleNamespace(
        presentation_id="05_generative_models",
        presentation_type="topic-deep-dive",
        topic="05_generative_models",
        final_report_url="https://example.com/report",
    )
    ledger_index = {
        "speech_transcript:broad": {
            "source_id": "speech_transcript:broad",
            "resolved_title": "Operational Generative Systems",
            "authors": ["A. Researcher"],
            "speakers": [],
            "contribution_note": "Frames the field-level shift.",
        }
    }
    board = {
        "clusters": [
            {
                "cluster_id": "operational-systems",
                "title": "Operational systems replace sample-only scoreboards",
                "claim": "The field-level claim.",
                "source_ids": ["speech_transcript:broad"],
            }
        ]
    }
    section = {
        "section_id": "S1",
        "title": "Field shift",
        "intent": "Establish the area-level frame.",
    }
    slides = [
        {
            "title": "Operational validity becomes the benchmark",
            "viewer_facing_claim": "Validity and provenance now matter.",
            "visible_text": ["Validity", "Provenance", "Control"],
            "narration_beats": ["A. Researcher frames the field-level shift."],
            "visual_concept": {
                "depiction": "tradeoff_map",
                "summary": "A field-level tradeoff map.",
                "components": ["sample quality", "validity", "provenance"],
            },
            "mentions": [{"source_id": "speech_transcript:broad", "role": "grounds frame"}],
        }
    ]

    blueprint = module.compile_blueprint(
        args=args,
        topic_title="Generative modeling: diffusion, flow, and controllability",
        presentation_frame=_frame(),
        board=board,
        section_slides=[(section, slides)],
        ledger_index=ledger_index,
        input_hashes={},
    )

    assert blueprint["slides"][0]["title"] == "Generative AI after sample quality"
    assert blueprint["audience_title"] == "Generative AI after sample quality"
    assert "operational infrastructure" in blueprint["deck_thesis"]
    assert blueprint["retrieval_title"] == "Generative modeling: diffusion, flow, and controllability"


def test_rebuild_blueprint_derived_metadata_after_repair_updates_source_fields() -> None:
    module = _load_editorial_module()
    old_mention = {
        "label": "SRC-001",
        "source_id": "speech_transcript:broad",
        "title": "Operational Generative Systems",
        "authors": ["A. Researcher"],
        "speakers": [],
        "contribution_note": "Frames the field-level shift.",
    }
    repaired_mention = {
        "label": "SRC-002",
        "source_id": "paper_abstract:new",
        "title": "New Evidence Paper",
        "authors": ["B. Author"],
        "speakers": [],
        "contribution_note": "Adds repaired paper coverage.",
    }
    blueprint = {
        "slides": [
            {
                "slide_id": "sl_01_title",
                "section_id": "S0_frame",
                "speaker_author_work_mentions": [],
            },
            {
                "slide_id": "sl_02_claim",
                "section_id": "S1",
                "speaker_author_work_mentions": [old_mention, repaired_mention],
            },
        ],
        "speaker_author_work_mention_plan": [old_mention],
        "source_paper_plan": [],
        "subtopic_clusters": [
            {
                "cluster_id": "operational-systems",
                "sources": ["SRC-001"],
                "slide_bindings": ["sl_02_claim"],
            }
        ],
        "provenance": {"source_metadata": [{"source_id": "speech_transcript:broad"}]},
    }
    board = {
        "clusters": [
            {
                "cluster_id": "operational-systems",
                "title": "Operational systems",
                "claim": "The field-level claim.",
                "source_ids": ["speech_transcript:broad", "paper_abstract:new"],
            }
        ]
    }
    ledger_index = {
        "speech_transcript:broad": {"source_id": "speech_transcript:broad"},
        "paper_abstract:new": {"source_id": "paper_abstract:new"},
    }

    rebuilt = module.rebuild_blueprint_derived_metadata(
        blueprint,
        board=board,
        ledger_index=ledger_index,
    )

    assert [m["source_id"] for m in rebuilt["speaker_author_work_mention_plan"]] == [
        "speech_transcript:broad",
        "paper_abstract:new",
    ]
    assert [m["source_id"] for m in rebuilt["source_paper_plan"]] == ["paper_abstract:new"]
    assert rebuilt["subtopic_clusters"][0]["sources"] == ["SRC-001", "SRC-002"]
    assert rebuilt["subtopic_clusters"][0]["slide_bindings"] == ["sl_02_claim"]
    assert [m["source_id"] for m in rebuilt["provenance"]["source_metadata"]] == [
        "speech_transcript:broad",
        "paper_abstract:new",
    ]


def test_compile_blueprint_falls_back_to_topic_title_without_frame() -> None:
    module = _load_editorial_module()
    args = SimpleNamespace(
        presentation_id="demo",
        presentation_type="topic-deep-dive",
        topic="demo",
        final_report_url="https://example.com/report",
    )
    blueprint = module.compile_blueprint(
        args=args,
        topic_title="Demo Topic",
        presentation_frame=None,
        board={"clusters": []},
        section_slides=[],
        ledger_index={},
        input_hashes={},
    )

    assert blueprint["slides"][0]["title"] == "Demo Topic"
    assert blueprint["audience_title"] == "Demo Topic"
