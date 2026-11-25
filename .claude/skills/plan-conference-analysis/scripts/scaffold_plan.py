#!/usr/bin/env python3
"""Draft or finalize conference-analysis GSD planning artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from conference_pipeline.model_governance import load_registry
except ImportError:  # pragma: no cover - installed skills may run without repo package.
    load_registry = None  # type: ignore[assignment]

try:
    from conference_pipeline.stage_script_registry import (
        load_stage_script_registry,
        registry_sha256,
    )
except ImportError:  # pragma: no cover - installed skills may run without repo package.
    load_stage_script_registry = None  # type: ignore[assignment]
    registry_sha256 = None  # type: ignore[assignment]


REQUIRED_BRIEF_FIELDS = [
    "conference_identity",
    "source_urls",
    "auth_notes",
    "target_outputs",
    "proof_mode",
    "topic_strategy",
    "provider_preferences",
    "provider_cost_approval_posture",
    "storage_share_targets",
    "acceptance_criteria",
    "platform_version",
    "conference_name",
]
# When `workshops` appears in the brief's target_outputs (or in the
# event-family inventory with status other than `out_of_scope`), a
# `workshop_coverage_posture` field is also required. Allowed values are
# enumerated below.
WORKSHOP_COVERAGE_POSTURES = (
    "all_openreview_with_pdf",
    "all_workshops_best_effort",
    "selective",
    "out_of_scope",
)
WORKSHOP_PDF_AVAILABILITY_CLASSES = (
    "openreview_with_pdf",
    "openreview_metadata_only",
    "no_openreview_record",
)
WORKSHOP_DECISION_VALUES = (
    "approved",
    "metadata_only_acceptable",
    "external_site_required",
    "excluded",
)
WORKSHOP_DISCOVERY_OWNER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/discover_openreview_workshops.py "
    "--venue-prefix <openreview_workshop_parent_venue> "
    "--output conferences/<slug>/storage/workshops/inventory.json "
    "--receipt-path conferences/<slug>/storage/workshops/discovery_receipt.json "
    "--openreview-auth required "
    "--workspace conferences/<slug>"
)
WORKSHOP_PAPER_DOWNLOAD_OWNER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/download_openreview_workshop_papers.py "
    "--inventory conferences/<slug>/storage/workshops/inventory.json "
    "--output-dir conferences/<slug>/storage/workshops/papers "
    "--rate-limit-rps 1.4 "
    "--receipt-path conferences/<slug>/storage/workshops/download_receipt.json "
    "--openreview-auth required "
    "--workspace conferences/<slug>"
)
CVF_OPENACCESS_WORKSHOP_DOWNLOAD_OWNER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/download_cvf_openaccess_workshop_papers.py "
    "--workspace conferences/<slug>"
)
WORKSHOP_TALK_MEDIA_OWNER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/acquire_iclr2026_workshop_talk_media.py "
    "--workspace conferences/<slug> "
    "--run-id <run-id> "
    "--resume "
    "--summary "
    "--live-crawl "
    "--iclr-cookie-file <local-cookie-file> "
    "--embedded-media-host slideslive.com "
    "--workshop-worker-count 3 "
    "--slide-worker-count 8"
)
EMBEDDED_MEDIA_RECOVERY_OWNER_REF = "src/conference_pipeline/acquisition/embedded_media.py"
SLIDESLIVE_MEDIA_RECOVERY_OWNER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/recover_iclr2026_slideslive_media.py "
    "--workspace conferences/<slug> "
    "--run-id <run-id> "
    "--resume "
    "--worker-count 2 "
    "--summary"
)
SLIDESLIVE_SLIDE_DOWNLOAD_OWNER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/recover_slideslive_slides.py "
    "--workspace conferences/<slug> "
    "--run-id <run-id> "
    "--resume "
    "--worker-count 4 "
    "--receipt-path conferences/<slug>/storage/download-corpus/slideslive_slide_recovery.json "
    "--summary"
)
SLIDE_REFERENCE_DETECTION_OWNER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/detect_reference_slides.py "
    "--slides-root conferences/<slug>/cache/acquisition-downloads/media_asset "
    "--output-dir conferences/<slug>/storage/download-corpus/reference_slides "
    "--receipt-path conferences/<slug>/storage/download-corpus/slide_reference_detection.json "
    "--summary"
)
SLIDE_CLEANUP_OWNER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/cleanup_slides.py "
    "--slides-root conferences/<slug>/cache/acquisition-downloads/media_asset "
    "--workspace conferences/<slug> "
    "--download-corpus-path storage/download-corpus/download_corpus.json "
    "--receipt-path conferences/<slug>/storage/download-corpus/slide_cleanup.json "
    "--summary"
)
WORKSHOP_TALK_SEGMENTATION_OWNER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/segment_workshop_talks.py "
    "--workspace conferences/<slug> "
    "--input-transcript-json storage/transcription-ocr/speech-transcripts/<talk>.json "
    "--talks-json storage/transcription-ocr/workshop-talks/talks.json "
    "--output-path storage/transcription-ocr/workshop-talks/annotated.json "
    "--receipt-path storage/transcription-ocr/workshop-talks/receipt.json "
    "--summary"
)
WORKSHOP_TALK_BOUNDARY_DISCOVERY_OWNER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/discover_workshop_talks.py "
    "--workspace conferences/<slug> "
    "--input-transcript-json storage/transcription-ocr/speech-transcripts/<workshop>.json "
    "--slides-json storage/download-corpus/workshop-slides/<workshop>.json "
    "--workshop-id <workshop_id> "
    "--workshop-title <workshop_title> "
    "--output-talks-json storage/transcription-ocr/workshop-talks/<workshop_id>_talks.json "
    "--output-batches-json storage/transcription-ocr/workshop-talks/<workshop_id>_batches.json "
    "--annotated-transcript-json storage/transcription-ocr/workshop-talks/<workshop_id>_annotated.json "
    "--receipt-path storage/transcription-ocr/workshop-talks/<workshop_id>_discovery_receipt.json "
    "--provider gemini "
    "--model gemini-2.5-pro "
    "--provider-cost-approved "
    "--summary"
)
TOKEN_BOUNDED_INDEXING_OWNER_REFS = (
    "src/conference_pipeline/indexing/token_policy.py",
    "src/conference_pipeline/indexing/chunking.py",
    "src/conference_pipeline/indexing/embeddings.py",
    "src/conference_pipeline/indexing/mlx_embeddings.py",
)
REQUIREMENT_IDS = [
    "SRC-01",
    "STORE-01",
    "COST-01",
    "ACQ-01",
    "OCR-01",
    "ANALYSIS-01",
    "ANALYSIS-NARRATIVE-01",
    "PAPER-MATCH-01",
    "QA-01",
    "PODCAST-01",
    "PRESENT-01",
    "VALIDATE-01",
    "PROOF-01",
]
ROADMAP_AREAS = [
    "source intake",
    "workshop discovery & corpus freeze",
    "workspace/storage",
    "provider cost gate",
    "acquisition proof",
    "transcription/OCR",
    "analysis",
    "narrative analysis provenance",
    "dual-store paper matching",
    "QA",
    "podcast",
    "presentation production",
    "validation",
    "proof/closeout",
]
EVENT_FAMILY_INVENTORY = [
    "accepted papers",
    "posters",
    "workshops",
    "keynotes",
    "tutorials",
    "competitions",
    "panels",
    "oral sessions",
    "conference-specific event types",
]
PRODUCTION_HANDOFF_FIELDS = [
    "next_step",
    "evidence_class",
    "corpus_bounds",
    "inventory_status",
    "media_family_scope",
    "interest_group_decisions",
    "exclusions",
    "implementation_inventory",
    "stage_script_registry",
    "workspace_gitignore_policy",
    "backup_posture",
    "provider_model_posture",
    "benchmark_requirements",
    "local_provider_deferrals",
    "presentation_figure_policy",
    "presentation_density_policy",
    "presentation_research_enrichment_policy",
    "presentation_slide_consolidation_policy",
    "presentation_brand_policy",
    "presentation_slide_sync_policy",
    "presentation_pdf_export_policy",
    "presentation_visual_qc_policy",
    "presentation_repair_policy",
    "narrative_analysis_contract",
    "local_proof_posture",
    "provider_contracts",
    "task_model_defaults",
    "workshop_coverage_posture",
    "workshop_inventory_contract",
    "structured_gates",
    "openreview_auth_posture",
    "corpus_source_metadata_gate",
]
MODEL_REGISTRY_TASK_IDS = [
    "transcription.session_speech",
    "ocr.slide_image_parse",
    "interest_group.research",
    "video.session_detection",
    "source_reading.categorization",
    "analysis.source_reading",
    "analysis.llm_full_read_extraction",
    "analysis.exhaustive_topic_summarization",
    "analysis.deep_dive_question_selection",
    "analysis.topic_reconciliation",
    "analysis.topic_reconciliation_reference_workflow",
    "analysis.topic_strategy_generation",
    "analysis.deep_narrative_synthesis",
    "research.paper_matching_keypoints",
    "qa.repair",
    "synthesis.executive_summary",
    "podcast.script_generation",
    "podcast.tts",
    "presentation.deck_generation",
    "presentation.svg_figure_generation",
    "presentation.visual_qc",
    "presentation.repair",
    "vision.figure_caption",
]
CORPUS_SOURCE_METADATA_GATE = {
    "gate_id": "PG-CORPUS-SOURCE-METADATA",
    "stage": "corpus source-metadata enrichment",
    "audit_command": (
        "PYTHONPATH=src .venv/bin/python scripts/audit_corpus_source_metadata.py "
        "--workspace conferences/<slug> --json --write-receipt storage/workflow-runs/<run-id>"
    ),
    "receipt_filename": "corpus_source_metadata_coverage.json",
}


def corpus_source_metadata_gate(slug: str) -> dict[str, str]:
    return {
        **CORPUS_SOURCE_METADATA_GATE,
        "audit_command": CORPUS_SOURCE_METADATA_GATE["audit_command"].replace("<slug>", slug),
    }


PRESENTATION_BLUEPRINT_GENERATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_editorial_blueprint.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--analysis-md analysis/<id>.md "
    "--speaker-author-work-ledger decks/<id>/speaker_author_work_ledger.json "
    "--exhaustive-papers analysis/exhaustive_summaries/<id>__papers.json "
    "--presentation-frame decks/<id>/presentation_frame.json "
    "--output-json decks/<id>/presentation_blueprint.json "
    "--output-md decks/<id>/presentation_blueprint.md "
    "--receipt-path decks/<id>/presentation_blueprint_receipt.json "
    "--final-report-url <url> "
    "--provider-cost-approved "
    "presentation_frame_validation"
)
PRESENTATION_BLUEPRINT_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/review_presentation_blueprint.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-blueprint decks/<id>/presentation_blueprint.json "
    "--blueprint-receipt decks/<id>/presentation_blueprint_receipt.json "
    "--blueprint-review decks/<id>/blueprint_review.json "
    "--output-dir decks/<id> "
    "approved_for_draft_generation blueprint_sha256 "
    "gate type: deterministic gate type: LLM-judge gate type: human"
)
PRESENTATION_BLUEPRINT_REPAIR_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/repair_presentation_blueprint.py "
    "--presentation-blueprint decks/<id>/presentation_blueprint.json "
    "--blueprint-review decks/<id>/blueprint_review.json "
    "--output decks/<id>/presentation_blueprint.repair.json"
)
PRESENTATION_DRAFT_GENERATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_deck_contract.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--presentation-blueprint decks/<id>/presentation_blueprint.json "
    "--blueprint-receipt decks/<id>/presentation_blueprint_receipt.json "
    "--blueprint-review decks/<id>/blueprint_review.json "
    "--output decks/<id>/presentation_deck.json "
    "--receipt-output decks/<id>/presentation_deck_receipt.json"
)
PRESENTATION_EXECUTIVE_OVERVIEW_DECKS_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_executive_overview_decks.py "
    "--workspace <workspace> "
    "--analysis-manifest analysis/analysis_manifest.json "
    "--executive-summary analysis/executive_summary.md "
    "--trend-report analysis/trend_report.md "
    "--short-id executive_summary_short --long-id executive_summary_long "
    "--source-link-map decks/executive_source_link_map.json "
    "--require-source-link-map "
    "--receipt-output decks/executive_overview_decks_receipt.json"
)
PRESENTATION_REVEALJS_RENDER_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/md_to_revealjs.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--render-spec decks/<id>/render_spec.json "
    "--output-html decks/<id>/presentation.html "
    "--slide-mapping decks/<id>/presentation_slide_mapping.json "
    "--require-brand-assets"
)
PRESENTATION_PRESENTER_SCRIPT_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_presenter_script.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--render-spec decks/<id>/render_spec.json "
    "--slide-mapping decks/<id>/presentation_slide_mapping.json "
    "--conference-label <conference> "
    "--delivery-single-speaker decks/<id>/delivery_single_speaker.json "
    "--delivery-notebooklm decks/<id>/delivery_notebooklm.json "
    "--debug-script-md decks/<id>/6_presenter_script.md"
)
PRESENTATION_PRESENTER_NARRATION_JSON_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/build_presenter_video.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--delivery-contract decks/<id>/delivery_single_speaker.json "
    "--render-spec decks/<id>/render_spec.json "
    "--build-dir decks/<id>/video_build "
    "--output decks/<id>/presentation_video.mp4 "
    "--stop-after-script-json"
)
PRESENTATION_PRESENTER_NARRATION_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/lint_presenter_narration.py "
    "--delivery-contract decks/<id>/delivery_single_speaker.json "
    "--script-json decks/<id>/video_build/script.json "
    "--deck-kind <topic|executive_short|executive_long> "
    "--conference-label <conference> "
    "--report-out decks/<id>/presenter_narration_lint_report.json && "
    ".venv/bin/python scripts/presentations/review_presenter_narration.py "
    "--delivery-contract decks/<id>/delivery_single_speaker.json "
    "--script-json decks/<id>/video_build/script.json "
    "--lint-report decks/<id>/presenter_narration_lint_report.json "
    "--deck-kind <topic|executive_short|executive_long> "
    "--conference-label <conference> "
    "--review-out decks/<id>/presenter_narration_review.md "
    "--json-out decks/<id>/presenter_narration_review.json"
)
PRESENTATION_PRESENTER_NARRATION_SERIES_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/review_narration_series.py "
    "--decks-dir decks "
    "--delivery-kind single_speaker "
    "--conference-label <conference> "
    "--report-out decks/presenter_narration_series_review.json "
    "--markdown-out decks/presenter_narration_series_review.md"
)
PRESENTATION_PRESENTER_NARRATION_FIX_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/fix_presenter_narration.py "
    "--delivery-contract decks/<id>/delivery_single_speaker.json "
    "--script-json decks/<id>/video_build/script.json "
    "--report-out decks/<id>/presenter_narration_fix_report.json"
)
PRESENTATION_NARRATION_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/lint_narration_script.py "
    "--delivery-contract decks/<id>/delivery_notebooklm.json "
    "--source-delivery-contract decks/<id>/delivery_single_speaker.json "
    "--deck-kind <topic|executive_short|executive_long> "
    "--conference-label <conference> "
    "--report-out decks/<id>/narration_lint_report.json && "
    ".venv/bin/python scripts/presentations/review_narration_script.py "
    "--delivery-contract decks/<id>/delivery_notebooklm.json "
    "--source-delivery-contract decks/<id>/delivery_single_speaker.json "
    "--lint-report decks/<id>/narration_lint_report.json "
    "--deck-kind <topic|executive_short|executive_long> "
    "--conference-label <conference> "
    "--review-out decks/<id>/narration_review.md "
    "--json-out decks/<id>/narration_review.json"
)
PRESENTATION_NARRATION_SERIES_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/review_narration_series.py "
    "--decks-dir decks "
    "--conference-label <conference> "
    "--report-out decks/narration_series_review.json "
    "--markdown-out decks/narration_series_review.md"
)
PRESENTATION_NARRATION_SERIES_FIX_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/fix_narration_series.py "
    "--decks-dir decks "
    "--report-out decks/narration_series_fix_report.json"
)
PRESENTATION_NARRATION_FIX_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/fix_narration_script.py "
    "--delivery-contract decks/<id>/delivery_notebooklm.json "
    "--script-json decks/<id>/video_build_notebooklm/script.json "
    "--deck-kind <topic|executive_short|executive_long> "
    "--conference-label <conference> "
    "--conference-label-limit <target-mentions> "
    "--replacement-json <replacement-json> "
    "--report-out decks/<id>/narration_fix_report.json"
)
PRESENTATION_VIDEO_GENERATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/build_presenter_video.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--delivery-contract decks/<id>/delivery_single_speaker.json "
    "--render-spec decks/<id>/render_spec.json "
    "--build-dir decks/<id>/video_build "
    "--output decks/<id>/presentation_video.mp4 && "
    ".venv/bin/python scripts/presentations/build_notebooklm_style_presenter_video.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--delivery-contract decks/<id>/delivery_notebooklm.json "
    "--source-delivery-contract decks/<id>/delivery_single_speaker.json "
    "--render-spec decks/<id>/render_spec.json "
    "--build-dir decks/<id>/video_build_notebooklm "
    "--output decks/<id>/presentation_video_notebooklm.mp4 "
    "--duration-reference-script-json decks/<id>/video_build/script.json "
    "--conference-label <conference>"
)
PRESENTATION_PRESENTATION_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/review_presentation.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--requirements decks/<id>/presentation_blueprint.json "
    "--narration-script decks/<id>/6_presenter_script.md "
    "--screenshots-dir decks/<id>/assets/generated/qc/screenshots "
    "--blueprint decks/<id>/presentation_blueprint.json "
    "--blueprint-review decks/<id>/blueprint_review.json "
    "--deck-contract decks/<id>/presentation_deck.json "
    "--render-spec decks/<id>/render_spec.json "
    "--source-binding decks/<id>/source_binding_report.json "
    "--delivery-contract decks/<id>/delivery_notebooklm.json "
    "--video-receipt decks/<id>/video_sync_validation_notebooklm.json "
    "--visual-qc-report decks/<id>/assets/generated/visual_llm_qc_report.json "
    "--contact-sheet decks/<id>/render_contact_sheet.png "
    "--report-output decks/<id>/presentation_review.json "
    "--markdown-output decks/<id>/presentation_review.md "
    "--prompt-output decks/<id>/presentation_review.prompt.md "
    "claude-opus-4-8 codex-5.5 gemini guideline_adherence "
    "technical_audience_pleasantness production_readiness "
    "aesthetic_pleasantness clarity_and_congestion artifact_contract_adherence"
)
PRESENTATION_PDF_EXPORT_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/export_revealjs_pdf.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--input-html decks/<id>/presentation.html "
    "--output-pdf decks/<id>/presentation.pdf"
)
PRESENTATION_LINK_AUDIT_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/audit_presentation_links.py "
    "--workspace <workspace> --decks-dir decks --conference-slug <slug> "
    "--receipt-output decks/presentation_link_audit_receipt.json"
)
PRESENTATION_PROGRAMMATIC_FIGURE_OWNER_COMMAND = (
    "scripts/presentations/generate_programmatic_figures.py"
)
PRESENTATION_PROGRAMMATIC_FIGURE_LIVE_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_programmatic_figures.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--deck-contract decks/<id>/presentation_deck.json "
    "--source-binding-report decks/<id>/source_binding_report.json "
    "--source-pack decks/<id>/source_pack_receipt.json "
    "--ranked-papers decks/<id>/ranked_papers.json "
    "--asset-inventory decks/<id>/asset_inventory.json "
    "--visual-plan decks/<id>/visual_plan.json "
    "--generated-dir decks/<id>/assets/generated "
    "--figure-specs-output decks/<id>/assets/generated/figure_specs.json "
    "--live-sonnet-specs"
)
STRUCTURED_PRESENTATION_COMMAND_BLOCK = """
## Structured Presentation Command Contract

Production presentation plans must use structured artifacts as the source of
truth and must reject old numbered Markdown with
`PRESENTATION_LEGACY_MARKDOWN_STAGE_USED`.

`scripts/presentations/generate_deck_contract.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --presentation-blueprint decks/<id>/presentation_blueprint.json --blueprint-receipt decks/<id>/presentation_blueprint_receipt.json --blueprint-review decks/<id>/blueprint_review.json --output decks/<id>/presentation_deck.json --receipt-output decks/<id>/presentation_deck_receipt.json`

`scripts/presentations/build_deck_source_pack.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --source-pack-output decks/<id>/source_pack_receipt.json`

`scripts/presentations/bind_presentation_sources.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --deck-contract decks/<id>/presentation_deck.json --matched-papers matched_papers.json --source-metadata <source-metadata> --source-pack decks/<id>/source_pack_receipt.json --output decks/<id>/source_binding_report.json`

Before `presentation.add_bookend_slides`, Reveal.js render, PDF export, video
release, or proof, generated plans must include a conference-adaptive
source-link implementation checkpoint. Inspect the actual conference corpus and
source metadata (OpenReview/forum ids, conference program pages,
workshop/media inventories, transcript extraction metadata, poster/media paths,
Drive source maps, and executive source-link maps), then update or extend
resolver/render/audit code such as
`src/conference_pipeline/presentation/source_links.py`,
`scripts/presentations/add_bookend_slides.py`,
`src/conference_pipeline/presentation/structured_video.py`,
`scripts/presentations/audit_presentation_links.py`, or a named conference
adapter/config plus regression tests. Do not assume a universal registry lookup
will resolve every link; unresolved visible references block unless a receipt
records that no public URL exists in the corpus.

`scripts/presentations/generate_programmatic_figures.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --deck-contract decks/<id>/presentation_deck.json --source-binding-report decks/<id>/source_binding_report.json --source-pack decks/<id>/source_pack_receipt.json --ranked-papers decks/<id>/ranked_papers.json --visual-plan decks/<id>/visual_plan.json --asset-inventory decks/<id>/asset_inventory.json --live-sonnet-specs`

`scripts/presentations/generate_render_spec.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --presentation-deck decks/<id>/presentation_deck.json --source-binding-report decks/<id>/source_binding_report.json --visual-plan decks/<id>/visual_plan.json --output decks/<id>/render_spec.json`

`scripts/presentations/md_to_revealjs.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --render-spec decks/<id>/render_spec.json --output-html decks/<id>/presentation.html --slide-mapping decks/<id>/presentation_slide_mapping.json`

`scripts/presentations/md_to_pptx.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --render-spec decks/<id>/render_spec.json --output-pdf decks/<id>/presentation.pdf --slide-mapping decks/<id>/presentation_slide_mapping.json`

`scripts/presentations/generate_presenter_script.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --render-spec decks/<id>/render_spec.json --slide-mapping decks/<id>/presentation_slide_mapping.json --delivery-single-speaker decks/<id>/delivery_single_speaker.json --delivery-notebooklm decks/<id>/delivery_notebooklm.json --debug-script-md decks/<id>/6_presenter_script.md`

`scripts/presentations/validate_video_sync.py --workspace <workspace> --presentation-id <id> --delivery-contract decks/<id>/delivery_single_speaker.json --video decks/<id>/presentation_video.mp4 --receipt-output decks/<id>/video_sync_validation_single_speaker.json`

`scripts/presentations/validate_video_sync.py --workspace <workspace> --presentation-id <id> --delivery-contract decks/<id>/delivery_notebooklm.json --video decks/<id>/presentation_video_notebooklm.mp4 --receipt-output decks/<id>/video_sync_validation_notebooklm.json`

`scripts/presentations/review_presentation.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --requirements decks/<id>/presentation_blueprint.json --narration-script decks/<id>/6_presenter_script.md --screenshots-dir decks/<id>/assets/generated/qc/screenshots --blueprint decks/<id>/presentation_blueprint.json --blueprint-review decks/<id>/blueprint_review.json --deck-contract decks/<id>/presentation_deck.json --render-spec decks/<id>/render_spec.json --source-binding decks/<id>/source_binding_report.json --delivery-contract decks/<id>/delivery_notebooklm.json --video-receipt decks/<id>/video_sync_validation_notebooklm.json --visual-qc-report decks/<id>/assets/generated/visual_llm_qc_report.json --contact-sheet decks/<id>/render_contact_sheet.png --report-output decks/<id>/presentation_review.json --markdown-output decks/<id>/presentation_review.md --prompt-output decks/<id>/presentation_review.prompt.md`

Produced presentation review must cover `claude-opus-4-8`, `codex-5.5`,
`gemini`, `guideline_adherence`, `technical_audience_pleasantness`,
`production_readiness`, `aesthetic_pleasantness`, `clarity_and_congestion`, and
`artifact_contract_adherence`.

After HTML/PDF export, `presentation.link_audit` must enumerate expected public
links from slide-2 events/papers, final-slide resources, content-slide source
chips, and executive summary decks; reject `file:///`, `/Users/...`,
repo-local, and `../analysis/...` hrefs; verify matching HTML hrefs and PDF
`/URI` annotations; and report `unresolved_visible_refs`, `bad_local_hrefs`,
`missing_html`, and `missing_pdf`. If the audit code cannot prove those
dimensions for the current deck format, the plan must patch the resolver/audit
code and rerun export rather than waive the check.
"""
PRESENTATION_FIGURE_SEMANTIC_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/review_programmatic_figures.py "
    "--workspace <workspace> --presentation-id <id> "
    "--dir decks/<id>/assets/generated "
    "--asset-inventory decks/<id>/asset_inventory.json "
    "--report-output decks/<id>/assets/generated/semantic_review.json"
)
PRESENTATION_EDITORIAL_GATE_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/presentation_editorial_gate.py "
    "--deck-dir decks/<id> --deck-kind <topic|executive_short|executive_long> "
    "--report-out decks/<id>/presentation_editorial_gate_report.json "
    "editorial_storyboard.json 40 minutes redundancy visual variety "
    "figure-narration alignment"
)
PRESENTATION_FIGURE_VALIDATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/validate_svgs.py "
    "--workspace <workspace> --presentation-id <id> "
    "--dir decks/<id>/assets/generated "
    "--report-output decks/<id>/assets/generated/validation_report.md"
)
PRESENTATION_SLIDE_CONSOLIDATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_render_spec.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--presentation-deck decks/<id>/presentation_deck.json "
    "--source-binding-report decks/<id>/source_binding_report.json "
    "--visual-plan decks/<id>/visual_plan.json "
    "--output decks/<id>/render_spec.json"
)
PRESENTATION_RESEARCH_ENRICHMENT_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/build_deck_source_pack.py "
    "--workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive "
    "--topic <topic> --extraction-run-id <run-id> "
    "--source-pack-output decks/<id>/source_pack_receipt.json && "
    ".venv/bin/python scripts/presentations/bind_presentation_sources.py "
    "--workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive "
    "--topic <topic> --deck-contract decks/<id>/presentation_deck.json "
    "--matched-papers storage/paper-matching/<run-id>/<topic>/matched_papers.json "
    "--source-metadata <source-metadata> "
    "--source-pack decks/<id>/source_pack_receipt.json "
    "--output decks/<id>/source_binding_report.json"
)
PRESENTATION_VISUAL_QC_OWNER_COMMAND = (
    "scripts/presentations/inspect_rendered_slides.py && "
    "scripts/presentations/inspect_rendered_slides_llm.py --report-output "
    "decks/<id>/assets/generated/visual_llm_qc_report.json --screenshots-dir "
    "decks/<id>/assets/generated/qc/screenshots"
)
PRESENTATION_REPAIR_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/run_presentation_repair_loop.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--render-spec decks/<id>/render_spec.json "
    "--visual-plan decks/<id>/visual_plan.json "
    "--generated-dir decks/<id>/assets/generated "
    "--html-output decks/<id>/presentation.html "
    "--slide-mapping decks/<id>/presentation_slide_mapping.json "
    "--validation-report decks/<id>/assets/generated/validation_report.md "
    "--visual-qc-report decks/<id>/assets/generated/visual_qc_report.md "
    "--llm-qc-report decks/<id>/assets/generated/visual_llm_qc_report.json "
    "--repair-feedback-output decks/<id>/assets/generated/visual_repair_feedback.json "
    "--screenshots-dir decks/<id>/assets/generated/qc/screenshots "
    "--source-pack decks/<id>/source_pack_receipt.json "
    "--ranked-papers decks/<id>/ranked_papers.json "
    "--asset-inventory decks/<id>/asset_inventory.json "
    "--receipt-output decks/<id>/assets/generated/repair_loop_receipt.json && "
    ".venv/bin/python scripts/presentations/inspect_rendered_slides_llm.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--screenshots-dir decks/<id>/assets/generated/qc/screenshots "
    "--report-output decks/<id>/assets/generated/visual_llm_qc_report.after_repair.json"
)
PODCAST_SCRIPT_GENERATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/generate_focused_podcast.py analysis/<id>.md "
    "--fact-ranks audio/<id>/ranks.json --output-dir audio/<id>/focused_podcast"
)
PODCAST_TTS_OWNER_COMMAND = (
    ".venv/bin/python scripts/render_podcast_audio.py "
    "--script-json audio/<id>/focused_podcast/script.json "
    "--output-dir audio/<id>/focused_podcast --model gpt-4o-mini-tts"
)
STAGE_SCRIPT_VERIFIER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py "
    "--plan-root <draft-root> --skills-root .claude/skills --docs-root docs --strict"
)
WORKSPACE_GITIGNORE_WRITE_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/generate_workspace_gitignore.py "
    "--workspace conferences/<slug> --write --check "
    "--receipt-path conferences/<slug>/storage/gitignore_policy_receipt.json"
)
WORKSPACE_GITIGNORE_CHECK_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/generate_workspace_gitignore.py "
    "--workspace conferences/<slug> --check"
)
REJECTED_PRESENTATION_DRAFT_OWNER_COMMANDS = [
    "render_slides(...)",
    "conference-specific scaffold writers",
    "scripts/presentations/generate_slides_simple.py",
    "scripts/generate_slides_agentic.py",
]
PROVIDER_CONTRACT_FIELDS = (
    "contract_id",
    "stage",
    "registry_task_id",
    "cost_task_id",
    "approved_provider_model",
    "fallback_or_escalation",
    "selection_status",
    "benchmark_authority",
    "receipt_requirement",
    "local_posture",
    "source_refs",
)
PROVIDER_CONTRACT_TABLE_HEADER = "| Contract ID | Stage | Registry task ID | Cost task ID | Approved provider/model | Fallback or escalation | Selection status | Benchmark authority | Receipt requirement | Local posture | Sources |"
BENCHMARK_AWARE_PROVIDER_CONTRACTS = [
    {
        "contract_id": "PC-14A-LOCAL-FIRST",
        "stage": "exhaustive extraction local/heuristic default",
        "registry_task_id": "analysis.source_reading; source_reading.categorization",
        "cost_task_id": "exhaustive_extraction_local_heuristic",
        "approved_provider_model": "local heuristic classifier over completed media artifacts and selected papers",
        "fallback_or_escalation": "heuristics serve as prefilter and fallback; cloud LLM escalation only for low-confidence or ambiguous records unresolved by local path; hard unresolved records keep remote escalation",
        "selection_status": "14A-only local-first default",
        "benchmark_authority": "Phase 14A context/research and registry snapshot; benchmark receipt required before expanding local-first beyond exhaustive extraction",
        "receipt_requirement": "receipt records provider, model, endpoint label or redacted local base URL identity, call counts, token counts when available, confidence path, escalation reason, paid-cost cap, estimated paid cost, and missing-server behavior",
        "local_posture": "local LLM is the default exhaustive extraction path; heuristics prefilter; cloud is ambiguity escalation only; local-first scope limited to exhaustive extraction",
        "source_refs": ".planning/phases/14A-exhaustive-topic-extraction-and-evidence-index/14A-CONTEXT.md; src/conference_pipeline/extractions/classifier.py; tests/test_exhaustive_topic_extraction.py",
    },
    {
        "contract_id": "PC-14A-QWEN-FULL-READ",
        "stage": "LLM full-read exhaustive extraction",
        "registry_task_id": "analysis.llm_full_read_extraction",
        "cost_task_id": "exhaustive_extraction_local_full_read",
        "approved_provider_model": "Local Qwen full-read lane: mlx-community/Qwen3.5-35B-A3B-4bit",
        "fallback_or_escalation": "no production fallback; pilot, limited, transcript-only, heuristic-only, or metadata-only receipts are blocker evidence",
        "selection_status": "required before final analysis when exhaustive extraction is in scope",
        "benchmark_authority": "registry snapshot plus conference-local receipt over the current transcript-window and paper-abstract scope",
        "receipt_requirement": "receipt records scope=all, no limit, completion_state=complete, completed_unit_count equals expected_unit_count, failed_count=0, transcript windows first, paper abstracts last, and by-unit/by-topic outputs",
        "local_posture": "local-only Qwen full read complements heuristic extraction; it does not replace final deep narrative synthesis",
        "source_refs": "src/conference_pipeline/extractions/llm_full_read.py; scripts/run_llm_full_read_extraction.py; docs/reference/stage-owner-audit.md#analysis-llm-full-read-extraction",
    },
    {
        "contract_id": "PC-EXHAUSTIVE-SUMMARY-PACKS",
        "stage": "provisional exhaustive topic/category summaries",
        "registry_task_id": "analysis.exhaustive_topic_summarization",
        "cost_task_id": "exhaustive_topic_summary_synthesis",
        "approved_provider_model": "Gemini gemini-3.1-pro-preview over completed LLM full-read by-topic evidence",
        "fallback_or_escalation": "local Qwen 122B is a smaller-chunk critic only until memory/long-context receipts prove safety",
        "selection_status": "experimental before formal stage promotion",
        "benchmark_authority": "registry snapshot plus dry-run summary_plan.json and live provider receipts over the current by-topic evidence",
        "receipt_requirement": "07A-exhaustive-topic-summaries.json records provider/model, pair_count, chunk_count, input tokens, provider receipt paths, and output JSON/Markdown summaries",
        "local_posture": "Gemini long-context is preferred for first-pass topic/category synthesis; local execution is deferred until memory proof receipts pass",
        "source_refs": "src/conference_pipeline/analysis/exhaustive_summaries.py; scripts/summarize_llm_full_read_topics.py; docs/reference/stage-owner-audit.md#analysis-exhaustive-topic-summarization",
    },
    {
        "contract_id": "PC-14B-EMBEDDINGS",
        "stage": "enhanced indexing/vector-store construction",
        "registry_task_id": "n/a: cost task spec authority",
        "cost_task_id": "phase_14b_indexing_embeddings",
        "approved_provider_model": "Local MLX BGE-M3 embeddings: mlx-community/bge-m3-mlx-8bit (1024 dim)",
        "fallback_or_escalation": "OpenAI text-embedding-3-small is a compatibility fallback only when local MLX embeddings are unavailable or receipt validation fails",
        "selection_status": "v1.92 local GPT-Researcher corpus default",
        "benchmark_authority": "registry snapshot plus ICLR 2025 BGE+Qwen benchmark receipt: conferences/iclr2025/storage/experiments/topic2-gptr-hybrid-bge-m3-retrieval-k18-c3-e144/comparison.json",
        "receipt_requirement": "vector-store receipt records EMBEDDING_PROVIDER=mlx, EMBEDDING_MODEL=mlx-community/bge-m3-mlx-8bit, EMBEDDING_DIM=1024, vector_store_run_id, source manifest digest, local-only API cost posture, and embedding_readiness/client_start_or_load_attempted proof before any local embedding blocker is accepted",
        "local_posture": "approved local corpus-retrieval default for GPT-Researcher evidence layer; not a final narrative substitute",
        "source_refs": "src/conference_pipeline/costs/task_specs.yaml; docs/workflows/pipeline-workflow.md; conferences/iclr2025/storage/experiments/topic2-gptr-hybrid-bge-m3-retrieval-k18-c3-e144/variants/i5-b4-d3-s8/config.json",
    },
    {
        "contract_id": "PC-GPT-RESEARCHER-LLM",
        "stage": "GPT Researcher query planning/deep query/synthesis",
        "registry_task_id": "research.gpt_researcher_fast; research.gpt_researcher_smart",
        "cost_task_id": "gpt_researcher_llm",
        "approved_provider_model": "Local Qwen GPT-Researcher corpus synthesis: mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit with WEB_SEARCH_ENABLED=false",
        "fallback_or_escalation": "OpenAI/Tavily is optional only for web-search-backed research, local runtime outage, or explicit operator escalation; final public report remains analysis.deep_narrative_synthesis",
        "selection_status": "v1.92 local corpus evidence default",
        "benchmark_authority": "registry snapshot plus ICLR 2025 BGE+Qwen benchmark receipt: conferences/iclr2025/storage/experiments/topic2-gptr-hybrid-bge-m3-retrieval-k18-c3-e144/comparison.json",
        "receipt_requirement": "receipt records GPTR_LOCAL_LLM_RUNTIME_KIND=mlx, LOCAL_LLM_* model provenance, WEB_SEARCH_ENABLED=false, retrieval_mode=hybrid, MAX_ITERATIONS=5, DEEP_RESEARCH_BREADTH=4, DEEP_RESEARCH_DEPTH=3, MAX_SUBTOPICS=8, RETRIEVAL_K=18, PAPER_MATCHING_RETRIEVAL_K=18, CANDIDATE_MULTIPLIER=3, RRF_K=60, TOTAL_WORDS=3000, token limits, temperature, behavioral/full parameter digests, and for deep dives the heuristic prefilter fields prefilter_terms_sha256, selected_evidence_ids, stores_searched, source_family_scope, and invalid_citation_ids",
        "local_posture": "approved local corpus evidence/draft default for GPT-Researcher findings; not final public narrative synthesis",
        "source_refs": "src/conference_pipeline/model_governance/registry.yaml; src/conference_pipeline/costs/task_specs.yaml; src/conference_pipeline/research/config.py; src/conference_pipeline/research/researcher.py; src/conference_pipeline/research/heuristic_prefilter.py; docs/reference/stage-owner-audit.md#analysis-gpt-researcher-stage; docs/reference/stage-owner-audit.md#analysis-gpt-researcher-heuristic-prefilter",
    },
    {
        "contract_id": "PC-GPT-RESEARCHER-EMBEDDINGS",
        "stage": "GPT Researcher hybrid local corpus retrieval embeddings",
        "registry_task_id": "research.gpt_researcher_fast; research.gpt_researcher_smart",
        "cost_task_id": "gpt_researcher_embeddings",
        "approved_provider_model": "Local MLX BGE-M3 embeddings: mlx-community/bge-m3-mlx-8bit (1024 dim)",
        "fallback_or_escalation": "OpenAI text-embedding-3-small is a compatibility fallback only when local MLX embeddings are unavailable or receipt validation fails",
        "selection_status": "v1.92 BM25+BGE vector hybrid retrieval default; vector-only is explicit override",
        "benchmark_authority": "registry snapshot plus ICLR 2025 BGE+Qwen benchmark receipt: conferences/iclr2025/storage/experiments/topic2-gptr-hybrid-bge-m3-retrieval-k18-c3-e144/comparison.json",
        "receipt_requirement": "query-planning-gpt-researcher receipt records workflow.research.gpt_researcher, retrieval_mode=hybrid, RETRIEVAL_K=18, PAPER_MATCHING_RETRIEVAL_K=18, CANDIDATE_MULTIPLIER=3, RRF_K=60, EMBEDDING_PROVIDER=mlx, EMBEDDING_MODEL=mlx-community/bge-m3-mlx-8bit, EMBEDDING_DIM=1024, vector-store digest, and embedding_readiness/client_start_or_load_attempted proof before any local embedding blocker is accepted",
        "local_posture": "approved local corpus-retrieval default for GPT-Researcher evidence layer; hybrid retrieval remains the default retrieval method",
        "source_refs": "src/conference_pipeline/costs/task_specs.yaml; src/conference_pipeline/research/config.py; src/conference_pipeline/research/researcher.py; docs/reference/stage-owner-audit.md#analysis-gpt-researcher-stage",
    },
    {
        "contract_id": "PC-TOPIC-STRATEGY",
        "stage": "title-spine topic strategy generation",
        "registry_task_id": "analysis.topic_strategy_generation",
        "cost_task_id": "topic_strategy_generation",
        "approved_provider_model": "Three-model consensus: OpenAI gpt-5.6-sol, Anthropic claude-opus-4-8, Gemini gemini-3.1-pro-preview; OpenAI gpt-5.6-sol merge",
        "fallback_or_escalation": "Model specs are CLI-overridable only with updated model-governance evidence; mock mode is planning/test evidence only and cannot close live production",
        "selection_status": "v2.1 title-spine consensus default",
        "benchmark_authority": "registry snapshot plus live topic_strategy_receipt before downstream query planning",
        "receipt_requirement": "receipt records title_spine sha256, source_family_counts, three recommendation model_specs, final_topics count=12, merge_model, provider_receipt_paths for live mode, and analysis/topic_strategy.json path",
        "local_posture": "not local by default; this is a frontier editorial consensus step unless a future benchmark approves a local replacement",
        "source_refs": "src/conference_pipeline/analysis/topic_strategy.py; scripts/build_topic_strategy.py; docs/reference/stage-owner-audit.md#analysis-topic-strategy; src/conference_pipeline/model_governance/registry.yaml",
    },
    {
        "contract_id": "PC-DEEP-NARRATIVE-SYNTHESIS",
        "stage": "deep narrative analysis synthesis",
        "registry_task_id": "analysis.deep_narrative_synthesis",
        "cost_task_id": "analysis_deep_narrative_synthesis",
        "approved_provider_model": "OpenAI gpt-5.6-sol authority with gpt-5.6-terra fallback",
        "fallback_or_escalation": "OpenAI gpt-5.6-terra fallback when gpt-5.6-sol output is unavailable or fails validation; no deterministic production success fallback",
        "selection_status": "v1.63 governed production default",
        "benchmark_authority": "registry snapshot plus deep-analysis validation receipt before replacement",
        "receipt_requirement": "receipt records analysis_profile, narrative_model, deep_validation, source-family coverage, slide_brief_path, and blocked/degraded status on missing LLM evidence",
        "local_posture": "local narrative replacement deferred until approved benchmark receipt exists; local GPT-Researcher/BGE+Qwen findings may feed the source pack but cannot close production-final analysis without deep validation",
        "source_refs": "docs/operations/analysis-contract.md; src/conference_pipeline/analysis/narrative.py; src/conference_pipeline/model_governance/registry.yaml",
    },
    {
        "contract_id": "PC-PAPER-MATCHING-KEYPOINTS",
        "stage": "artifact-derived paper matching keypoint extraction",
        "registry_task_id": "research.paper_matching_keypoints",
        "cost_task_id": "paper_matching_keypoint_extraction",
        "approved_provider_model": "OpenAI gpt-5.6-sol frontier model for keypoint extraction from artifact GPT-Researcher findings",
        "fallback_or_escalation": "replacement requires model-governance approval and benchmark receipt before paper matching production use",
        "selection_status": "v1.91 default when presentation paper matching is in scope",
        "benchmark_authority": "v1.91 dual-store paper matching contract and registry snapshot before frontier-model replacement",
        "receipt_requirement": "receipt records keypoint provider, keypoint model, artifact research source, topic id, prompt digest, and paper-store research linkage",
        "local_posture": "not local; frontier keypoint extraction is not replaced by legacy paper ranking or podcast-frequency scoring",
        "source_refs": "docs/workflows/pipeline-workflow.md; src/conference_pipeline/paper_matching/matcher.py; scripts/match_papers_from_artifact_keypoints.py",
    },
    {
        "contract_id": "PC-TAVILY-SEARCH-EXTRACT",
        "stage": "GPT Researcher search/extract",
        "registry_task_id": "research.gpt_researcher_fast; research.gpt_researcher_smart",
        "cost_task_id": "tavily_search_extract",
        "approved_provider_model": "Tavily tavily-search (default-OFF for local BGE+Qwen corpus-only GPT-Researcher)",
        "fallback_or_escalation": "enable only for explicit web-search-backed research or operator escalation; otherwise WEB_SEARCH_ENABLED=false",
        "selection_status": "tracked optional web-search provider",
        "benchmark_authority": "registry snapshot and benchmark receipt before Tavily replacement",
        "receipt_requirement": "provider/model/cost receipt required when Tavily is enabled; local corpus-only runs record WEB_SEARCH_ENABLED=false",
        "local_posture": "BM25+vector local corpus retrieval supplies the normal evidence path; Tavily remains optional web fallback and is suppressed for strict corpus-only GPT-Researcher runs",
        "source_refs": "src/conference_pipeline/costs/task_specs.yaml; .planning/phases/14C-query-planning-and-gpt-researcher-topic-execution/14C-CONTEXT.md",
    },
    {
        "contract_id": "PC-OCR-AUTHORITY",
        "stage": "OCR benchmark authority",
        "registry_task_id": "ocr.slide_image_parse",
        "cost_task_id": "slide_poster_visual_ocr",
        "approved_provider_model": "OpenAI gpt-5.6-sol authority with gpt-5.6-terra fallback",
        "fallback_or_escalation": "OpenAI gpt-5.6-terra authority fallback when gpt-5.6-sol output is empty or invalid",
        "selection_status": "benchmark authority preserved",
        "benchmark_authority": "registry snapshot and benchmark receipt authority for OCR provider/model selection",
        "receipt_requirement": "receipt requires cost, latency, JSON validity, schema completeness, locator/numeric recall, and text similarity",
        "local_posture": "local OCR replacement deferred until approved local proof and benchmark receipt exist",
        "source_refs": "docs/api/slide_ocr.md; src/conference_pipeline/model_governance/registry.yaml; .planning/milestones/v1.5-ICLR-2025-PRODUCTION-EXECUTION.md",
    },
    {
        "contract_id": "PC-OCR-PRODUCTION",
        "stage": "OCR production execution",
        "registry_task_id": "ocr.slide_image_parse",
        "cost_task_id": "slide_poster_visual_ocr; slide_poster_visual_ocr_fallback",
        "approved_provider_model": "OpenAI gpt-5.6-luna primary",
        "fallback_or_escalation": "Gemini gemini-3.1-flash-lite-preview only for low-confidence or failed image parses",
        "selection_status": "production default preserved",
        "benchmark_authority": "registry snapshot and OCR benchmark receipt before replacement",
        "receipt_requirement": "provider/model/cost receipt required for primary and fallback OCR calls",
        "local_posture": "local OCR replacement deferred until approved local proof and benchmark receipt exist",
        "source_refs": "docs/api/slide_ocr.md; src/conference_pipeline/costs/task_specs.yaml; src/conference_pipeline/model_governance/registry.yaml",
    },
    {
        "contract_id": "PC-EXTRACTION-REMOTE-ESCALATION",
        "stage": "exhaustive extraction remote escalation/failover",
        "registry_task_id": "exhaustive_extraction.claude_haiku; exhaustive_extraction.openrouter_sonnet",
        "cost_task_id": "exhaustive_extraction_direct; exhaustive_extraction_openrouter_failover",
        "approved_provider_model": "Anthropic claude-haiku-4-5-20251001 base; Anthropic claude-sonnet-5 hard-pass escalation",
        "fallback_or_escalation": "OpenRouter openrouter/anthropic/claude-haiku-4-5-20251001 base failover; openrouter/anthropic/claude-sonnet-5 hard-pass escalation",
        "selection_status": "remote extraction escalation/failover preserved separately from 14A local/heuristic default",
        "benchmark_authority": "registry snapshot and benchmark receipt before remote extraction provider replacement",
        "receipt_requirement": "provider/model/cost receipt required for remote extraction escalation/failover",
        "local_posture": "local-first exhaustive extraction remains fenced to 14A; remote escalation/failover stays available",
        "source_refs": "src/conference_pipeline/model_governance/registry.yaml; src/conference_pipeline/costs/task_specs.yaml; .planning/phases/14A-exhaustive-topic-extraction-and-evidence-index/14A-CONTEXT.md",
    },
    {
        "contract_id": "PC-NON14A-REGISTRY-GOVERNED",
        "stage": "non-14A registry-governed outputs",
        "registry_task_id": "analysis.source_reading; model_governance_task_ids",
        "cost_task_id": "registry/task_specs governed",
        "approved_provider_model": "podcast, deck/presentation, TTS, figure captioning, repair, QA, and executive synthesis inherit registry/task_specs defaults",
        "fallback_or_escalation": "replacement requires benchmark receipt, operator/provider approval posture, and audit-visible explanation",
        "selection_status": "registry/task_specs defaults preserved outside 14A extraction",
        "benchmark_authority": "registry snapshot and benchmark receipt before non-14A provider replacement",
        "receipt_requirement": "provider/model/cost receipt required before changing non-14A defaults",
        "local_posture": "local providers deferred unless approved local proof and benchmark receipts exist",
        "source_refs": "src/conference_pipeline/model_governance/registry.yaml; src/conference_pipeline/costs/task_specs.yaml; .planning/milestones/v1.5-ICLR-2025-PRODUCTION-EXECUTION.md",
    },
]
SUPPORT_LABELS = [
    "live",
    "dry-run",
    "mock",
    "unsupported",
    "planned",
    "dry-run/mock only",
    "needs live proof",
]
OFFICIAL_WORKFLOW_ROW_FIELDS = (
    "step_id",
    "official_step",
    "status",
    "phase_name",
    "support_label",
    "receipt_boundary",
    "rationale",
    "source_ref",
)
OFFICIAL_WORKFLOW_MAPPING = [
    {
        "step_id": "WF-01",
        "official_step": "Discover sessions",
        "status": "mapped",
        "phase_name": "Source Intake",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "source discovery receipt or source-intake blocker before acquisition proof",
        "rationale": "Official step 1 owns source discovery and proof labels.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-02",
        "official_step": "Manual filtering",
        "status": "mapped",
        "phase_name": "Source Intake",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "source selection receipt before acquisition proof",
        "rationale": "Official step 2 records manual selection instead of hiding it inside acquisition.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-03",
        "official_step": "Download audio and slide images",
        "status": "mapped",
        "phase_name": "Acquisition Proof",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "download manifest or unreachable-source receipt before transcription/OCR",
        "rationale": "Official step 3 requires acquisition evidence before media processing claims.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-04",
        "official_step": "Transcribe audio",
        "status": "mapped",
        "phase_name": "Transcription/OCR",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "transcription receipt before analysis",
        "rationale": "Official step 4 must have a transcript receipt before analysis support is claimed.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-05",
        "official_step": "Clean slides",
        "status": "mapped",
        "phase_name": "Transcription/OCR",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "clean-slide receipt or explicit no-slide waiver before OCR",
        "rationale": "Official step 5 is optional but must be acknowledged before OCR.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-06",
        "official_step": "OCR slides",
        "status": "mapped",
        "phase_name": "Transcription/OCR",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "OCR receipt before analysis",
        "rationale": "Official step 6 must have OCR evidence before slide-backed analysis claims.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-07",
        "official_step": "Reference extraction + paper download",
        "status": "out_of_scope",
        "phase_name": "",
        "support_label": "planned",
        "receipt_boundary": "explicit waiver in Official Workflow Mapping",
        "rationale": "Official step 7 is optional and not required for the default analysis plan.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-08",
        "official_step": "Enhanced indexing",
        "status": "mapped",
        "phase_name": "Analysis",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "enhanced indexing receipt before query planning",
        "rationale": "Official step 8 needs its own receipt boundary before query planning.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-09",
        "official_step": "Query planning",
        "status": "mapped",
        "phase_name": "Analysis",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "query planning receipt before topic execution",
        "rationale": "Official step 9 needs its own receipt boundary before topic execution.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-10",
        "official_step": "Topic execution",
        "status": "mapped",
        "phase_name": "Analysis",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "topic execution receipt before traceability/QA",
        "rationale": "Official step 10 needs its own receipt boundary before traceability and QA.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-11",
        "official_step": "Traceability + QA",
        "status": "mapped",
        "phase_name": "QA",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "traceability and QA receipt before coverage analysis",
        "rationale": "Official step 11 needs timestamps, slide IDs, and source-gap evidence before coverage repair.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-12",
        "official_step": "Coverage analysis and gap filling",
        "status": "mapped",
        "phase_name": "QA",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "coverage analysis receipt before synthesis",
        "rationale": "Official step 12 needs its own coverage receipt before synthesis.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-13",
        "official_step": "Synthesis",
        "status": "mapped",
        "phase_name": "Analysis",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "synthesis receipt before validation",
        "rationale": "Official step 13 needs an executive-summary or trend-report receipt before validation.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-14",
        "official_step": "Optional pipelines",
        "status": "mapped",
        "phase_name": "Proof/Closeout",
        "support_label": "planned",
        "receipt_boundary": "optional pipeline receipt or explicit waiver before proof/closeout",
        "rationale": "Official step 14 is tracked as a parent row with child rows for each optional pipeline.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-14a",
        "official_step": "Exhaustive extraction",
        "status": "mapped",
        "phase_name": "Analysis",
        "support_label": "planned",
        "receipt_boundary": "exhaustive extraction receipt or explicit waiver before proof/closeout",
        "rationale": "Exhaustive extraction is an optional pipeline and must not be hidden inside broad analysis.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-14b",
        "official_step": "Podcast",
        "status": "mapped",
        "phase_name": "Podcast",
        "support_label": "dry-run/mock only",
        "receipt_boundary": "podcast receipt or explicit waiver before proof/closeout",
        "rationale": "Podcast output is optional but already has a named generated phase.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "step_id": "WF-14c",
        "official_step": "Paper-level analysis",
        "status": "out_of_scope",
        "phase_name": "",
        "support_label": "planned",
        "receipt_boundary": "explicit waiver in Official Workflow Mapping",
        "rationale": "Paper-level analysis is optional and not part of the default generated analysis milestone.",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview",
    },
]
SECRET_PATTERNS = (
    re.compile(r"\b(?:ANTHROPIC_API_KEY|OPENAI_API_KEY|GOOGLE_API_KEY|GEMINI_API_KEY|TAVILY_API_KEY|NEURIPS_PASSWORD)\s*=\s*[^\s,;\"']+", re.IGNORECASE),
    re.compile(r"\bCookie:\s*[^\n\r]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9._-]+\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]+\b"),
)
SECRET_VALUE_PATTERNS = (
    re.compile(
        r"\b[\w.-]*(?:API[_-]?KEY|APIKEY|PASSWORD|TOKEN|SECRET)[\w.-]*\s*[:=]\s*[\"']?[^\s,\"']+",
        re.IGNORECASE,
    ),
    re.compile(r"\bAuthorization:\s*(?:Bearer|Basic)\s+[^\s,\"']+", re.IGNORECASE),
    re.compile(r"\bCookie:\s*[^\n\r]+", re.IGNORECASE),
    re.compile(r"\b(?:sessionid|session|cookie)\s*[:=]\s*[^\s,\"']+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]+\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]+\b"),
)


def load_brief(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    if not slug or "/" in slug or "\\" in slug or ".." in slug:
        raise ValueError("invalid conference slug")
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError("invalid conference slug")
    return slug


def validate_brief(brief: dict[str, object]) -> list[str]:
    return [field for field in REQUIRED_BRIEF_FIELDS if field not in brief or brief[field] in (None, "", [])]


def ensure_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"path escapes root: {path}")
    return resolved


def safe_text(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    for pattern in SECRET_VALUE_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


def safe_list_text(value: object) -> str:
    items = value if isinstance(value, list) else [value]
    return ", ".join(safe_text(item) for item in items)


def milestone_identity(brief: dict[str, object]) -> str:
    return f"{safe_text(brief['platform_version'])}-{safe_text(brief['conference_name'])}"


def conference_slug(brief: dict[str, object]) -> str:
    identity = brief.get("conference_identity")
    if isinstance(identity, dict) and isinstance(identity.get("slug"), str):
        return slugify(safe_text(identity["slug"]))
    return slugify(safe_text(brief["conference_name"]))


def draft_directory(brief: dict[str, object], draft_root: Path, *, new_version: bool) -> Path:
    slug = conference_slug(brief)
    if new_version:
        slug = f"{slug}-{slugify(safe_text(brief['platform_version']))}"
    return ensure_inside(draft_root / slug, draft_root)


def context_filename(brief: dict[str, object]) -> str:
    return f"{milestone_identity(brief)}-CONTEXT.md"


def backup_target_leaf(brief: dict[str, object]) -> str:
    return f"My Drive/{safe_text(brief['conference_name'])}-raw"


def backup_remote_path_placeholder(brief: dict[str, object]) -> str:
    return f"<mounted-drive-root>/{safe_text(brief['conference_name'])}-raw"


def generated_relative_paths(brief: dict[str, object], slug: str) -> list[str]:
    return [
        f".planning/milestones/{context_filename(brief)}",
        ".planning/REQUIREMENTS.md",
        ".planning/ROADMAP.md",
        ".planning/STATE.md",
        f"conferences/{slug}/README.md",
    ]


def render_context(brief: dict[str, object], slug: str) -> str:
    identity = milestone_identity(brief)
    source_urls = "\n".join(f"- {safe_text(url)}" for url in brief["source_urls"])
    target_outputs = safe_list_text(brief["target_outputs"])
    proof_mode = safe_text(brief["proof_mode"])
    inventory_rows = "\n".join(
        f"- {family}: `unverified` until source-backed inventory records evidence."
        for family in EVENT_FAMILY_INVENTORY
    )
    handoff_rows = "\n".join(
        f"- `{field}`: required before live production planning can proceed."
        for field in PRODUCTION_HANDOFF_FIELDS
    )
    registry_rows = "\n".join(f"- `{task_id}`" for task_id in MODEL_REGISTRY_TASK_IDS)
    backup_target = backup_target_leaf(brief)
    backup_remote_path = backup_remote_path_placeholder(brief)
    return f"""---
conference_slug: "{slug}"
conference_name: "{safe_text(brief['conference_name'])}"
platform_version: "{safe_text(brief['platform_version'])}"
conference_milestone_name: "{identity}"
proof_status: "scaffolded"
planning_owner: ".planning/milestones/{context_filename(brief)}"
workspace: "conferences/{slug}"
---

# {identity} Context

## Local Sources And Auth Notes

Sources:
{source_urls}

Auth notes: {safe_text(brief['auth_notes'])}

## Analysis Profile And Topic Strategy

Topic strategy: {safe_text(brief['topic_strategy'])}

Target outputs: {target_outputs}

## Event-Family Inventory Contract

Generated analysis specs must enumerate the full virtual-program scope, not
only accepted papers. Inventory rows start unresolved and must become
source-backed before expensive provider work:

{inventory_rows}

Unknown or uncertain event families must remain `unknown` or `unverified` with
a source note. They must not be silently omitted.

## Media-Family Scope Decision

Media acquisition scope is opt-out, never opt-in: the default is ALL
media-bearing event families in this inventory (tutorials, keynotes/invited
talks, orals, workshops and segmented workshop talks, panels, spotlights,
expo talks/demonstrations/workshops, affinity events, town halls, award
talks, socials with recorded media). Planning must ask the user explicitly
which families, if any, to EXCLUDE, and record the answer here and in the
`media_family_scope` handoff field (`default: all_media_bearing_families`,
`excluded_families`, `decision_source`). Production must re-confirm the same
question before acquisition runs, and the download/corpus stage fails closed
when a declared media-bearing family yields zero discovered item pages
without a receipt-recorded exclusion
(`program_family_media_items_missing`). Acquisition must never allowlist
event families by URL shape; every virtual-site item page (typed or bare-ID)
is a target by default via `conference_pipeline.program_coverage`. A
reachable family index with zero extracted item links is a discovery gap,
never evidence of no media.

## Workshop Coverage Contract

Workshops are a structured forest of sub-venues, not a single event-family
bullet. The brief must record `workshop_coverage_posture` (one of
`all_openreview_with_pdf`, `all_workshops_best_effort`, `selective`,
`out_of_scope`) and the analysis scaffold must produce a per-workshop sub-
inventory at `conferences/{slug}/storage/workshops/inventory.json` via:

```
{WORKSHOP_DISCOVERY_OWNER_COMMAND.replace('<slug>', slug)}
```

Each row carries `venueid`, `paper_count`, `tiers`, and one of the PDF
availability classes `openreview_with_pdf`, `openreview_metadata_only`, or
`no_openreview_record`, plus an explicit decision: `approved`,
`metadata_only_acceptable`, `external_site_required`, or `excluded`. The
follow-on owner

```
{WORKSHOP_PAPER_DOWNLOAD_OWNER_COMMAND.replace('<slug>', slug)}
```

downloads PDFs for `openreview_with_pdf` venues at ≤1.4 rps with HTTP 429
backoff and resume. For `openreview_metadata_only` and `no_openreview_record`
venues the analysis scaffold writes an external-site downloader stub under
`scripts/conference/{slug}/workshop_site_downloaders/<workshop_slug>.py` and
records it in `conferences/{slug}/storage/workshops/workshop_site_tracker.json`.
Production audit rejects stub-only scaffolds for in-scope workshops; live
production must replace each stub with a concrete downloader and a per-paper
receipt schema matching the OpenReview downloader before the workshop
acquisition gates can pass.

For CVF-published conferences (CVPR, ICCV, WACV) the archival workshop
proceedings are public on the CVF Open Access site, so before any stub is
written or counted the phase runs

```
{CVF_OPENACCESS_WORKSHOP_DOWNLOAD_OWNER_COMMAND.replace('<slug>', slug)}
```

(`--dry-run` first; menu URL from `urls.openaccess_workshops` in
`config.yaml`). It skips papers already downloaded from OpenReview and writes
per-paper receipts in the same schema; only workshops with no CVF listing and
non-archival tracks remain `external_site_required`.

Since roughly 2026-07 the OpenReview notes and PDF endpoints are challenge-gated
(HTTP 403 `ChallengeRequiredError`) for unauthenticated non-browser clients, so
both owner commands pass `--openreview-auth required` and mint a Bearer token
from `OPENREVIEW_USERNAME`/`OPENREVIEW_PASSWORD` (repo-root `.env`; values never
recorded — receipts carry only `openreview_auth_mode` and
`openreview_auth_env_names`). Authenticated download runs fetch PDFs from the
`api2.openreview.net/attachment` endpoint the token unlocks. A 403 or zero-note
result from an unauthenticated fetch is a gated result, not evidence of no
papers, and must not be classified `no_openreview_record`;
`openreview_auth_mode: unauthenticated` on a required corpus is a fail-stop
blocker. Conference virtual-site login-gated pages/media use
`scripts/export_browser_cookies.py --browser <browser> --domain <site>
--output <path-outside-repo>` to reuse an existing browser session into a
Netscape cookie file, referenced by the conference's `*_COOKIES_FILE` env var
(names and cookie names only in artifacts, never values).

## SlidesLive Slide Acquisition And Cleanup Contract

SlidesLive audio and SlidesLive slide images are separate acquisition lanes.
When cached `item_page_auth_probe` HTML exposes SlidesLive embeds and
slide/visual/OCR evidence is in scope, plans must include:

```
{SLIDESLIVE_SLIDE_DOWNLOAD_OWNER_COMMAND.replace('<slug>', slug)}
```

The slide owner writes `storage/download-corpus/slideslive_slide_recovery.json`
and merges slide-image artifacts into `download_corpus.json` without
redownloading source pages, papers, workshops, poster images, audio, video, or
unrelated media families. Before visual OCR or indexing, plans must then run:

```
{SLIDE_REFERENCE_DETECTION_OWNER_COMMAND.replace('<slug>', slug)}

{SLIDE_CLEANUP_OWNER_COMMAND.replace('<slug>', slug)}
```

Cleanup removes generic/banner slides and consecutive duplicates from OCR and
indexing with `skip_ocr` markers. Raw slide images remain preserved for audit.
Audio-only recovery, paper PDFs, and poster thumbnails are rejected evidence for
SlidesLive slide acquisition.

## Token-Bounded Indexing And Talk-Aware Transcript Contract

Corpus indexing is a first-class production gate, not a side effect of source
inventory. Production plans must include `PG-CORPUS-TOKEN-INDEXING` and must
prove token-bounded chunking through `conference_pipeline.indexing.token_policy`,
`conference_pipeline.indexing.chunking`,
`src/conference_pipeline/indexing/token_policy.py`, and
`src/conference_pipeline/indexing/chunking.py`. Defaults are 1000 chunk tokens
and 100 overlap tokens after metadata prefixes. Embedding materialization is
owned by `conference_pipeline.indexing.embeddings`,
`conference_pipeline.indexing.mlx_embeddings`,
`src/conference_pipeline/indexing/embeddings.py`, and
`src/conference_pipeline/indexing/mlx_embeddings.py`. The Stage 6 receipt and
index digest must record `token_budget_policy_digest`,
`embedding_policy_digest`, `EMBEDDING_MAX_LENGTH`,
`CORPUS_VECTOR_EMBED_BATCH_SIZE`, `embedding_sidecar_manifest_path`,
`embedding_sidecar_manifest_sha256`, `chroma_insert_batch_size`, and
`vector_embeddings_written: true`, with embedded chunk counts matching chunk
sidecars before query planning or GPT Researcher runs. Embedding sidecars are
written before bulk Chroma load; runtime batch sizes are receipt evidence, not
embedding identity. Rejected evidence includes word-count-only sidecars,
readiness-only probes, partial Chroma rows, conference-local embedding
reimplementations, and historical extraction output without registry receipt.

Workshop speech transcripts must preserve talk context. When workshop audio or
transcripts are in scope and no vetted `talks.json` already exists, production
plans must first prove talk-media acquisition with public-first/authenticated
retry and embedded-player extraction:

```
{WORKSHOP_TALK_MEDIA_OWNER_COMMAND.replace('<slug>', slug)}
```

The required acquisition owner ids are `acquisition.workshop_talk_media` and
`acquisition.embedded_media_recovery`, backed by
`conference_pipeline.acquisition.embedded_media` and
`src/conference_pipeline/acquisition/embedded_media.py`. The owner extracts
embedded media with `embedded_media_links`, strips secrets with
`sanitize_media_url`, uses yt-dlp before accepting inaccessible-media blockers,
records `allowed_hosts`, and records explicit per-workshop blockers for any
workshop that still lacks accessible talk media. ICLR production must keep this
host allowlist SlidesLive-only with `--embedded-media-host slideslive.com`
unless a future conference-specific phase explicitly widens it. Then include
`PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY` and run:

```
{WORKSHOP_TALK_BOUNDARY_DISCOVERY_OWNER_COMMAND.replace('<slug>', slug)}
```

The required discovery owner id is
`transcription_ocr.workshop_talk_boundary_discovery`, backed by
`conference_pipeline.workshop_talk_segmentation` and
`scripts/discover_workshop_talks.py`. It uses Gemini 2.5 Pro
(`gemini-2.5-pro`) by default over long workshop transcripts and slide
timestamps, writes `*_talks.json`, produces 5-minute talk-context batches, and
records `model_governance_task_id`, `provider_receipt_paths`, and
`cost_task_id=workshop_talk_boundary_discovery` before any provider-backed
cost can close.

After discovery or a vetted `talks.json`, production plans must run:

```
{WORKSHOP_TALK_SEGMENTATION_OWNER_COMMAND.replace('<slug>', slug)}
```

The required owner id is `transcription_ocr.workshop_talk_segmentation`, backed
by `conference_pipeline.workshop_talk_segmentation`. It adds `workshop_id`,
`workshop_title`, `talk_id`, `talk_title`, and
`talk_speaker` metadata, and if neither `talks.json` nor governed Gemini
boundary discovery exists it blocks with
`talk_segmentation_provider_not_configured`. audio-only recovery is rejected
evidence for talk-aware transcripts.

## Corpus Source-Metadata Coverage Gate

The `corpus source-metadata enrichment` stage is a first-class stage between
corpus inventory and enhanced indexing, not an implicit side effect of either.
Production plans generated from this analysis specification must include
`PG-CORPUS-SOURCE-METADATA` and run:

```
{corpus_source_metadata_gate(slug)['audit_command']}
```

This writes the coverage receipt `{corpus_source_metadata_gate(slug)['receipt_filename']}`,
which must record per-source-family (`papers_main`, `papers_workshop`, `artifacts`)
title/author coverage at or above each family's justified floor, the
`artifacts` family's `authors_classification: structurally_authorless`
posture, and a non-empty justification for any workspace floor override. A
milestone must not close while this gate fails.

This is a plan-text presence check only: `audit_generated_plan.py` and
`audit_generated_production_plan.py` verify that a generated plan's text
declares this stage and gate. They do not and cannot verify that a future
conference's live run actually produced real coverage -- that stronger,
runtime-scanning guarantee comes only from running the
`corpus-metadata-audit` skill (`scripts/audit_corpus_source_metadata.py`)
against real on-disk artifacts, alongside (not instead of)
`conference-hardcoding-audit` and `model-registry-audit`. See
`.planning/workstreams/icml-2026/audits/skill-enforcement-review-20260713.md`
for the empirical record of a plan-text audit reporting a full pass against a
plan whose live execution failed.

## Interest-Group Review

Interest groups are proposed only after the event-family inventory is complete.
Each proposed group must include evidence notes and an explicit user decision:
approved, edited, rejected, or complete. Incomplete interest-group
approval blocks downstream provider-backed analysis rather than carrying only a
warning.

## Exclusion Records

Exclusions can cover event families, workshops, competitions, sessions, or
individual items. Each exclusion record must preserve scope, reason, timestamp
or decision context, source/evidence note, and downstream effect before
downloads, transcription/OCR, deep research, or provider-backed analysis.

## Production Execution Handoff

The production handoff is a required structured contract:

{handoff_rows}

## Next Step: Production Planning

After approving and finalizing this analysis specification, call
`$plan-conference-production` with the finalized spec root to create the live-only execution phase or milestone that actually produces the conference analysis.
This planning specification is not production execution evidence:
live production still requires approval, credentials/source access, provider
cost gates, and receipt-backed completion.

Model-governance task ids required by this handoff:

{registry_rows}

Model posture must distinguish approved choices, pending research, benchmark
requirements, and `not approved yet` status. Local-provider candidates remain
explicit deferrals until a later local-runtime milestone produces receipts.

## Implementation Inventory Contract

Before production planning names owner commands, each governed task id must map
to an existing executable owner, a planned implementation task, or an explicit
blocker. Existing implementations are preferred when they satisfy the contract;
new scripts require a written no-existing-implementation rationale.

## Manifest-Driven Workspace Gitignore Policy

Conference workspace gitignore rules are generated from
`conferences/{slug}/storage/storage_manifest.json`, not duplicated by hand.
Planning must run:

`{WORKSPACE_GITIGNORE_WRITE_COMMAND.replace("<slug>", slug)}`

Production and closeout plans must verify the same policy with:

`{WORKSPACE_GITIGNORE_CHECK_COMMAND.replace("<slug>", slug)}`

Only `storage_class: git` families are trackable by default. Families marked
`public`, `backup`, or `cache` are Google Drive or cache-owned and must be
ignored unless a small control/proof file is declared in
`gitignore_exceptions`. Payload artifacts such as SVGs, screenshots, videos,
audio, PDFs, generated HTML, and deck intermediates must not be committed when
their artifact family is Drive-owned.

Before manifest-backed `presentation.deck_contract_generation`, generated production
plans must produce and approve `presentation_blueprint`. The registry owner task
ids are `presentation.generate_editorial_blueprint`,
`presentation.review_presentation_blueprint`, and
`presentation.repair_presentation_blueprint`:

`{PRESENTATION_BLUEPRINT_GENERATION_OWNER_COMMAND}`

The blueprint owner implementation also includes
`src/conference_pipeline/presentation/presentation_frame.py`; missing or stale
`presentation_frame.json` blocks deck-contract generation.

`{PRESENTATION_BLUEPRINT_REVIEW_OWNER_COMMAND}`

If review blocks, repair with:

`{PRESENTATION_BLUEPRINT_REPAIR_OWNER_COMMAND}`

The approval token must include `approved_for_draft_generation`, the current
`blueprint_sha256`, gate type: deterministic, gate type: LLM-judge, and gate
type: human evidence. No bulk presentation regeneration before approved 04_safety_alignment_privacy canary. Registry usage docs:
`docs/reference/stage-owner-audit.md#presentation-generate-presentation-blueprint`
and
`docs/reference/stage-owner-audit.md#presentation-review-presentation-blueprint`.

Phase 13 canary and bulk-readiness commands must be present before any bulk
presentation regeneration plan:

`scripts/presentations/build_phase13_canary_objective_evidence.py --deck-dir decks/<id> --validation-summary decks/<id>/presentation_validation_report.json --frames-dir decks/<id>/assets/generated/qc/frames --report-out 13-CANARY-OBJECTIVE-EVIDENCE.json --markdown-out 13-CANARY-OBJECTIVE-EVIDENCE.md`

`scripts/presentations/build_phase13_canary_review_packet.py --deck-dir decks/<id> --validation-summary decks/<id>/presentation_validation_report.json --objective-evidence-report 13-CANARY-OBJECTIVE-EVIDENCE.json --report-out 13-CANARY-REVIEW-PACKET.json --markdown-out 13-CANARY-REVIEW-PACKET.md`

`scripts/presentations/run_phase13_rollout.py --workspace conferences/<slug> --require-approved-canary --canary-review-packet 13-CANARY-REVIEW-PACKET.json --report-out 13-ROLLOUT-PLAN.json --markdown-out 13-ROLLOUT-PLAN.md review_narration_series.py presenter_narration_series_review.json narration_series_review.json`

`scripts/presentations/verify_phase13_bulk_readiness.py --workspace conferences/<slug> --canary-review-packet 13-CANARY-REVIEW-PACKET.json --objective-evidence-report 13-CANARY-OBJECTIVE-EVIDENCE.json --rollout-plan 13-ROLLOUT-PLAN.json --report-out 13-BULK-READINESS.json --markdown-out 13-BULK-READINESS.md`

`scripts/presentations/verify_phase13_completion.py --workspace conferences/<slug> --phase-dir <phase-dir> --final-validation-summary presentation_validation_report.json --report-out 13-COMPLETION-VERIFICATION.json --markdown-out 13-COMPLETION-VERIFICATION.md`

For manifest-backed `presentation.deck_contract_generation`, generated production
plans must then use:

`{PRESENTATION_DRAFT_GENERATION_OWNER_COMMAND}`

When a conference deliverable includes overview or executive-summary
presentations, analysis plans must require both `decks/executive_summary_short`
and `decks/executive_summary_long` as deck artifacts. `analysis/executive_summary.md`
and `analysis/trend_report.md` are source synthesis reports, not deck
completion evidence. Generate the overview deck pair with:

`{PRESENTATION_EXECUTIVE_OVERVIEW_DECKS_OWNER_COMMAND}`

Executive overview structure is operator-locked. Planning may extract up to 24
candidate lessons for coverage, but the produced short deck should select about
8 executive points for roughly 15 minutes of narration. The produced long deck
should use the same 8-point spine plus 3 finer-grained expansions, around 11
points total, for roughly 30 minutes of narration. The long deck deepens the
short deck with evidence and visual explanation; it must not become a separate
topic index or resource inventory.

Reject these as completion owners for manifest-backed deck contract generation:

{chr(10).join(f"- rejected owner, do not use: `{command}`" for command in REJECTED_PRESENTATION_DRAFT_OWNER_COMMANDS)}

For manifest-backed `presentation.render_export`, generated production plans
must render from the structured render specification with the exact RelationalAI
brand reference:

`{PRESENTATION_REVEALJS_RENDER_OWNER_COMMAND}`

Missing title metadata inside `render_spec.json`, missing
`assets/brand/relationalai/` reference assets, generic themes, or approximate
brand colors are blockers.

For manifest-backed `presentation.delivery_generation`, generated production plans
must emit slide-mapping-aware delivery contracts for both video styles:

`{PRESENTATION_PRESENTER_SCRIPT_OWNER_COMMAND}`

Generated production plans must then include dedicated single-presenter and
NotebookLM narration review/fix phases before TTS or video rendering. First
generate reviewable single-presenter TTS JSON and run the presenter review owner
for every topic and executive deck:

`{PRESENTATION_PRESENTER_NARRATION_JSON_OWNER_COMMAND}`

`{PRESENTATION_PRESENTER_NARRATION_REVIEW_OWNER_COMMAND}`

Then run the single-presenter series review owner across the full deck set:

`{PRESENTATION_PRESENTER_NARRATION_SERIES_REVIEW_OWNER_COMMAND}`

If presenter review fails, route through the presenter fix owner or a governed
LLM/manual revision skill, keeping `delivery_single_speaker.json` and
`video_build/script.json` synchronized:

`{PRESENTATION_PRESENTER_NARRATION_FIX_OWNER_COMMAND}`

For NotebookLM, run the review owner for every topic and executive deck:

`{PRESENTATION_NARRATION_REVIEW_OWNER_COMMAND}`

Then run the series review owner across the full deck set:

`{PRESENTATION_NARRATION_SERIES_REVIEW_OWNER_COMMAND}`

TTS is blocked unless every deck's presenter and NotebookLM review JSON records
`approve_for_tts=true` and both series reviews have no unaccepted failure. If
NotebookLM deck-level review fails, route through the fix owner or a governed
LLM/manual revision skill, keeping `delivery_notebooklm.json` and `script.json`
synchronized:

`{PRESENTATION_NARRATION_FIX_OWNER_COMMAND}`

If series review fails, route through the series fix owner or a governed
cross-deck revision pass, then rerun both deck-level and series-level reviews:

`{PRESENTATION_NARRATION_SERIES_FIX_OWNER_COMMAND}`

For manifest-backed `presentation.video_generation`, generated production plans
must run both required video builders:

`{PRESENTATION_VIDEO_GENERATION_OWNER_COMMAND}`

The video builders are required evidence owners because they validate that
the delivery contracts and NotebookLM `script.json` cover every mapped
render-spec slide, including the title slide, before any audio/video sync can
be treated as complete.

Produced presentation review must run before human approval, release promotion,
or closeout:

`{PRESENTATION_PRESENTATION_REVIEW_OWNER_COMMAND}`

The report must pass for every required reviewer and criterion before produced
presentation artifacts can be treated as release-ready.

For manifest-backed `presentation.render_export`, generated production plans must
export PDF/PPTX artifacts from `render_spec.json` with canonical owners:

`{PRESENTATION_PDF_EXPORT_OWNER_COMMAND}`

Manual browser print-to-PDF, ad hoc Chrome commands, or presentation screenshots
cannot satisfy `presentation.render_export` unless the plan records an explicit
registry-approved waiver.

{STRUCTURED_PRESENTATION_COMMAND_BLOCK}

## Programmatic Figure Generation Contract

`presentation.svg_figure_generation` requires programmatic matplotlib, Plotly,
or svgwrite SVG generation plus render/inspect/repair evidence. Programmatic
output is necessary but not sufficient: figures must be content-aware,
slide-specific, and editorial-quality, with template selection tied to slide
intent and rendered inspection rejecting repeated boilerplate or generic
placeholder/scaffold diagrams. The required owner command is
`{PRESENTATION_PROGRAMMATIC_FIGURE_LIVE_OWNER_COMMAND}`.

Generated SVGs must then pass the registry-backed semantic figure review owner:

`{PRESENTATION_FIGURE_SEMANTIC_REVIEW_OWNER_COMMAND}`

Generated SVGs must then pass the registry-backed figure validation owner:

`{PRESENTATION_FIGURE_VALIDATION_OWNER_COMMAND}`

Deprecated paper-figure scripts are forbidden as completion owners:

- deprecated, do not use: `scripts/presentations/extract_topic_figures.py`
- deprecated, do not use: `scripts/presentations/add_images_to_md.py`

Paper-extracted/cropped figures may supplement presentation content, but they
must not satisfy `FIGURE_NEED` or SVG generation requirements.

## Presentation Visual Repair Contract

If governed vision-LLM visual QC produces blocker findings, production must run
the LLM-feedback repair owner:

`{PRESENTATION_REPAIR_OWNER_COMMAND}`

The repair loop must read `visual_llm_qc_report.json`, normalize findings by
markdown slide using `presentation_slide_mapping.json`, write
`visual_repair_feedback.json`, rerun
`scripts/presentations/generate_programmatic_figures.py` with that feedback,
rebuild `presentation.html`, and rerun rendered screenshot QC before the deck can
retry governed vision-LLM QC. A mechanical `validate_svgs.py --fix` pass alone
does not satisfy `presentation.repair`.

## Vision-LLM Visual QC Contract

Presentation visual QC requires both rendered screenshot inspection and a
governed vision-LLM review. Generated production plans must run:

`{PRESENTATION_VISUAL_QC_OWNER_COMMAND}`

The LLM reviewer uses model-governance task `presentation.visual_qc` with
default `openai:gpt-5.6-sol` and fallback `openai:gpt-5.6-terra`. Missing provider/model,
missing `visual_llm_qc_report.json`, unparseable JSON, or any blocker finding
keeps presentation production incomplete.

## Presentation Density Contract

Generated production plans must configure NeurIPS-style density targets in
`workflow.deck` and `presentation_manifest.json`: `target_hyperlinks_per_slide`
defaults to `2.0`, and `target_svgs_per_slide` defaults to `0.70`. Presentation
validation must run `hyperlink_density_meets_target`,
`hyperlink_preservation_4_5_to_5`, and `svg_density_meets_target`; figure
insertion must preserve or increase the hyperlink count from
`4.5_draft_slides_with_hyperlinks.md` into `5_draft_slides_with_images.md`.
Plans must also map `presentation.research_enrichment` to the full source-pack,
paper-enrichment, reflow, sanity-check, and hybrid hyperlink owner chain. The
hybrid hyperlink command must pass `--source-attributions` and
`--matched-papers`; unresolved `[paper](paper_url)` placeholders block
production.

## Narrative Analysis Contract

Public `analysis/<topic>.md` files must be deep analysis documents with
`Executive Summary`, `Key Findings`, `Detailed Analysis`, `References And
Sources`, `Cross-References`, `Recommendations`, `Questions For Further
Investigation`, and a compact `Slide Brief` section. Deck-facing summaries
belong in `analysis/slide_briefs/<topic_id>.md`. Raw extraction, vector chunk,
research, matched-paper, and quote ledgers belong in
`analysis/provenance/<topic_id>.md`, `analysis/provenance/<topic_id>.json`, and
`analysis/analysis_citation_manifest.json`. Production narrative synthesis must
use `analysis.deep_narrative_synthesis` with default `openai:gpt-5.6-sol` and
fallback `openai:gpt-5.6-terra`, iterative GPT-Researcher/deep-query plans, and
deep validation gates. Missing LLM/model/evidence must block production rather
than publish shallow prose. `Source Basis`, `Verbatim transcript/OCR evidence`,
`Matched Paper Evidence`, and `Source Gaps` are rejected as top-level
public-report sections.

Deck/presentation phases must include a presentation narrative-content gate
that verifies draft decks consume `analysis/slide_briefs/<topic_id>.md`, bind
to the current slide-brief/deep-analysis digest, reject stale scaffold/public-ledger
language, and fail identical deck checkpoint files unless an explicit
valid stage waiver exists.

## Backup Posture

- Manifest source: `conferences/{slug}/storage/storage_manifest.json`.
- Expected backup profile: `backup`.
- Target Drive leaf: `{backup_target}`.
- Runtime requirement: production must pass `--remote-path {backup_remote_path}`
  explicitly for every backup sync, setup, restore verification, and prune
  command.
- Drive setup posture: production must run `scripts/sync_to_gdrive.py --profile
  backup --remote-path {backup_remote_path} --init-only` before live artifact
  generation so a new conference gets its mounted Drive leaf and
  manifest-selected family directories without copying payloads.
- Continuous offload posture: every artifact-producing production stage must run
  the manifest-bounded `backup` sync immediately after it writes outputs and
  before downstream stages consume them. Sync is rsync-incremental and profile
  bounded so cache/unselected families do not bloat Drive.
- Restore verification posture: generated production work must record backup
  receipts before verification and use restore evidence before cleanup.
- No-secret rule: backup paths and receipts may name mounted Drive paths and
  redacted target identities, but never credential values, cookies, API keys,
  access tokens, or mounted-drive secrets.

## Supported Pipeline Stages

### Live Support

- live: needs live proof before crawl, transcription, analysis, podcast, or presentation production are claimed.
- unsupported: no live scaffold support is claimed by this generated plan.

### Dry-Run Support

- dry-run: use generic workflow and smoke-test receipts where supported.
- dry-run/mock only: proof mode is {proof_mode}.

### Mock Support

- mock: use mock providers for UAT and planning rehearsal.
- planned: expand to live only after conference-local proof.

## Proof Status

This scaffold is dry-run/mock only and needs live proof before any live support claims.

## Canonical Local Docs And Manifests

- `conferences/{slug}/README.md`
- `conferences/{slug}/analysis/analysis_manifest.json` planned
- `conferences/{slug}/analysis/analysis_citation_manifest.json` planned when narrative source packs exist
- `conferences/{slug}/analysis/provenance/<topic_id>.md` and `.json` planned for production analysis provenance
- `conferences/{slug}/storage/workflow-runs/<run-id>/conference-smoketest-report.json` planned

## Decisions And Limitations

- Provider preferences: {safe_text(brief['provider_preferences'])}
- Provider cost approval posture: {safe_text(brief['provider_cost_approval_posture'])}
- Storage/share targets: {safe_text(brief['storage_share_targets'])}
- Acceptance criteria: {safe_text(brief['acceptance_criteria'])}
"""


def render_requirements(identity: str) -> str:
    rows = "\n".join(f"- [ ] **{req}** - {identity} {req} planning coverage." for req in REQUIREMENT_IDS)
    return f"# Requirements: {identity}\n\n{rows}\n"


def workflow_mapping_for_brief(brief: dict[str, object]) -> list[dict[str, str]]:
    rows = [dict(row) for row in OFFICIAL_WORKFLOW_MAPPING]
    target_outputs = brief.get("target_outputs", [])
    if isinstance(target_outputs, str):
        normalized_targets = [target_outputs.lower()]
    else:
        normalized_targets = [str(item).lower() for item in target_outputs]
    if any("paper-level analysis" in item or "paper analysis" in item for item in normalized_targets):
        for row in rows:
            if row["step_id"] == "WF-14c":
                row.update(
                    {
                        "status": "mapped",
                        "phase_name": "Analysis",
                        "support_label": "dry-run/mock only",
                        "receipt_boundary": "paper-level analysis receipt before synthesis",
                        "rationale": "The brief requested paper-level analysis as an optional pipeline.",
                    }
                )
    if any("reference extraction" in item or "paper download" in item for item in normalized_targets):
        for row in rows:
            if row["step_id"] == "WF-07":
                row.update(
                    {
                        "status": "mapped",
                        "phase_name": "Acquisition Proof",
                        "support_label": "dry-run/mock only",
                        "receipt_boundary": "reference extraction and paper download receipt before indexing",
                        "rationale": "The brief requested reference extraction + paper download as an optional pipeline.",
                    }
                )
    return rows


def render_official_workflow_mapping(rows: list[dict[str, str]]) -> str:
    lines = [
        "## Official Workflow Mapping",
        "",
        "| Step ID | Official workflow step | Status | Phase | Support label | Receipt boundary | Rationale | Source |",
        "|---------|------------------------|--------|-------|---------------|------------------|-----------|--------|",
    ]
    for row in rows:
        cells = []
        for field in OFFICIAL_WORKFLOW_ROW_FIELDS:
            value = row.get(field, "")
            if field == "phase_name" and not value:
                value = "-"
            cells.append(safe_text(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def benchmark_aware_provider_contracts() -> list[dict[str, str]]:
    return [
        {field: safe_text(row[field]) for field in PROVIDER_CONTRACT_FIELDS}
        for row in BENCHMARK_AWARE_PROVIDER_CONTRACTS
    ]


def render_benchmark_aware_provider_contracts(rows: list[dict[str, str]]) -> str:
    lines = [
        "## Benchmark-Aware Provider Contracts",
        "",
        "These rows are also written to approval-summary.json as `benchmark-aware provider contracts` and to `production handoff.provider_contracts`.",
        "",
        PROVIDER_CONTRACT_TABLE_HEADER,
        "|-------------|-------|------------------|--------------|-------------------------|------------------------|------------------|---------------------|---------------------|---------------|---------|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(safe_text(row.get(field, "")) for field in PROVIDER_CONTRACT_FIELDS) + " |")
    return "\n".join(lines) + "\n"


def task_model_default_rows(policies: dict[str, dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task_id in MODEL_REGISTRY_TASK_IDS:
        policy = policies.get(task_id, {})
        provider = safe_text(policy.get("provider", "unresolved"))
        model = safe_text(policy.get("model", "unresolved"))
        fallback_provider = safe_text(policy.get("fallback_provider", "") or "")
        fallback_model = safe_text(policy.get("fallback_model", "") or "")
        benchmark = policy.get("benchmark")
        if isinstance(benchmark, dict):
            benchmark_required = str(bool(benchmark.get("required", True))).lower()
            receipt_required = str(bool(benchmark.get("receipt_required", True))).lower()
        else:
            benchmark_required = "true"
            receipt_required = "true"
        rows.append(
            {
                "task_id": task_id,
                "stage": safe_text(policy.get("stage", "unresolved")),
                "default_provider_model": f"{provider}:{model}",
                "fallback_provider_model": (
                    f"{fallback_provider}:{fallback_model}"
                    if fallback_provider and fallback_model
                    else "-"
                ),
                "selection_status": safe_text(policy.get("selection_status", "unresolved")),
                "benchmark_required": benchmark_required,
                "receipt_required": receipt_required,
            }
        )
    return rows


def render_task_model_defaults(rows: list[dict[str, str]]) -> str:
    lines = [
        "## Task Model Defaults",
        "",
        "This is the operator-facing model/default matrix. Every governed task that may call an LLM, ASR, OCR, embedding, or local model must appear here before production planning; downstream plans must preserve these defaults unless a benchmark receipt approves a replacement.",
        "",
        "| Registry task ID | Stage | Default provider/model | Fallback provider/model | Selection status | Benchmark required | Receipt required |",
        "|------------------|-------|------------------------|-------------------------|------------------|--------------------|------------------|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                safe_text(row.get(field, ""))
                for field in (
                    "task_id",
                    "stage",
                    "default_provider_model",
                    "fallback_provider_model",
                    "selection_status",
                    "benchmark_required",
                    "receipt_required",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_extraction_provider_policy() -> str:
    return (
        "## Extraction Provider Policy\n"
        "\n"
        "Generated exhaustive extraction plans default to the configured local\n"
        "OpenAI-compatible LLM endpoint and model for topic classification and evidence\n"
        "mapping over completed artifacts. Deterministic heuristics serve as prefilter\n"
        "and fallback when the local endpoint is unavailable or when heuristic confidence\n"
        "meets the classification threshold. Cloud LLMs are reserved for low-confidence\n"
        "or ambiguous records that neither heuristics nor the local LLM can resolve with\n"
        "sufficient confidence.\n"
        "\n"
        "The following env vars configure the local extraction LLM when exhaustive\n"
        "extraction is in scope: `EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL` (endpoint label),\n"
        "`EXTRACTION_LOCAL_LLM_BASE_URL` (loopback address; non-local endpoints are\n"
        "rejected), `EXTRACTION_LOCAL_LLM_MODEL` (model name). Plans must declare local\n"
        "LLM readiness as a prerequisite before production extraction begins. If the local\n"
        "endpoint is not configured or not reachable, the plan records this as a blocker\n"
        "rather than silently routing to cloud LLM.\n"
        "\n"
        "Non-extraction generation providers are not covered by this\n"
        "extraction-local-default policy. The approved GPT-Researcher evidence lane\n"
        "uses local Qwen3.6 35B-A3B plus local BGE-M3 embeddings with corpus-only hybrid\n"
        "retrieval; ASR, TTS, ranking, podcast, presentation, OCR, and final report\n"
        "synthesis continue to use their own benchmark-governed contracts.\n"
    )


def render_gpt_researcher_retrieval_policy() -> str:
    return (
        "## GPT-Researcher Retrieval And Parameter Policy\n"
        "\n"
        "Generated analysis plans must use the registry-owned GPT-Researcher stage\n"
        "`analysis.gpt_researcher_stage` backed by\n"
        "`conference_pipeline.iclr2025_live.run_query_planning_gpt_researcher`,\n"
        "`conference_pipeline.research.researcher`,\n"
        "`conference_pipeline.research.query_planner`,\n"
        "`conference_pipeline.research.config`, and\n"
        "`conference_pipeline.research.walker`. The registry usage anchor is\n"
        "`docs/reference/stage-owner-audit.md#analysis-gpt-researcher-stage`.\n"
        "\n"
        "The default GPT-Researcher evidence lane is local/corpus-only:\n"
        "`GPTR_LOCAL_LLM_RUNTIME_KIND=mlx`,\n"
        "`GPTR_LOCAL_LLM_SMART_MODEL=mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`,\n"
        "`GPTR_LOCAL_LLM_FAST_MODEL=mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`,\n"
        "`EMBEDDING_PROVIDER=mlx`, and\n"
        "`EMBEDDING_MODEL=mlx-community/bge-m3-mlx-8bit`. The default config is\n"
        "`workflow.research.gpt_researcher.retrieval_mode: hybrid`, meaning\n"
        "BM25+vector reciprocal-rank fusion. Vector-only retrieval is an explicit\n"
        "override, not the default.\n"
        "\n"
        "Deep-dive topics must first run `analysis.deep_dive_question_selection`\n"
        "with GPT-5.6 Luna to compress exhaustive summary bullets into 4-6 selected\n"
        "questions per topic via `conference_pipeline.research.deep_dive_lanes`,\n"
        "`src/conference_pipeline/research/deep_dive_lanes.py`, and\n"
        "`scripts/select_deep_dive_questions.py`; one\n"
        "GPT-Researcher run per raw key point is rejected.\n"
        "It consumes `analysis.deep_dive_question_selection` and writes\n"
        "`07B-deep-dive-question-selection.json` plus `selected_deep_dives.json`.\n"
        "\n"
        "Each selected deep-dive question must run two evidence lanes before promotion: the global\n"
        "no-topic-id `workflow.research.gpt_researcher` / `global_no_topic_id` lane over the whole corpus\n"
        "or stable source scopes, and the keyword-prefiltered lane owned by\n"
        "`analysis.gpt_researcher_heuristic_prefilter`,\n"
        "`conference_pipeline.research.heuristic_prefilter`,\n"
        "`src/conference_pipeline/research/heuristic_prefilter.py`, and\n"
        "`scripts/run_heuristic_prefiltered_research.py` with\n"
        "`docs/reference/stage-owner-audit.md#analysis-gpt-researcher-heuristic-prefilter`.\n"
        "The heuristic lane records\n"
        "`workflow.research.heuristic_prefiltered_gpt_researcher`,\n"
        "`heuristic_keyword_prefilter`, `keyword_seed_source`,\n"
        "`prefilter_terms_sha256`, `candidate_count`, `selected_evidence_ids`,\n"
        "`stores_searched`, `source_family_scope`, `retrieval_scope`, and\n"
        "`invalid_citation_ids`.\n"
        "Rejected evidence: topic-id filtered deep dives, unrecorded keyword\n"
        "lists, source-family leakage, and citation-invalid heuristic reports.\n"
        "The combined runner is `analysis.dual_deep_dive_research_lanes` with\n"
        "`workflow.research.dual_deep_dive_gpt_researcher_lanes`\n"
        "via `scripts/run_dual_deep_dive_research.py --resume` with local Qwen3.6 35B-A3B.\n"
        "Before final GPT-5.6 Luna narrative synthesis or slide synthesis, plans must run\n"
        "`analysis.topic_reconciliation_reference_workflow` via\n"
        "`scripts/run_topic_reconciliation_reference_workflow.py`; this packages\n"
        "`07D-topic-reconciliation.json`, deterministic `07E-topic-reference-validation.json`,\n"
        "deterministic safety repair, bounded `07G-topic-reference-candidate-resolution.json`,\n"
        "fallback `07F-topic-reference-repair.json`, and a final validator-clean\n"
        "`07H-topic-reconciliation-reference-workflow.json` with zero\n"
        "`final_invalid_reference_count`, `final_unknown_reference_count`,\n"
        "`final_title_mismatch_count`, and `final_bare_reference_warning_count`.\n"
        "Inside that workflow,\n"
        "`analysis.topic_reconciliation` via `scripts/reconcile_topic_analysis.py`\n"
        "using `conference_pipeline.analysis.topic_reconciliation` with local\n"
        "`mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`; it writes\n"
        "`07D-topic-reconciliation.json`, `analysis/topic_reconciliations/<topic_id>.md`,\n"
        "and injects `topic_reconciliation` into `analysis/narrative_source_packs/<topic_id>.json`.\n"
        "Then `analysis.topic_reference_validation` runs `scripts/validate_topic_reconciliation_references.py`\n"
        "to build `analysis/topic_reference_ledgers/<topic_id>.json`, write\n"
        "`analysis/topic_reference_validation/<topic_id>.json`, record\n"
        "`07E-topic-reference-validation.json`, and block empty ledgers, unknown ids, bare ids, or title/id mismatches.\n"
        "If 07E blocks, `analysis.topic_reference_candidate_resolution` first runs\n"
        "`scripts/resolve_topic_reference_candidates.py` with local Qwen3.6, writes\n"
        "`07G-topic-reference-candidate-resolution.json`, candidate packets under\n"
        "`analysis/topic_reference_candidate_resolution/`, records\n"
        "`topic_reconciliation.reference_candidate_resolution`, and applies only\n"
        "`final_validation`-clean repaired markdown. `analysis.topic_reference_repair`\n"
        "is the broad fallback only after unresolved candidate-resolution blockers,\n"
        "records `packet_truncated_count` and `applied_topic_count`,\n"
        "and then 07E validation reruns before final narrative or slide synthesis.\n"
        "Generated plans should point operators at the packaged workflow command\n"
        "rather than relying on manual memory of the 07D/07E/07G/07F order.\n"
        "Slide synthesis must not be the first place raw exhaustive summaries and deep-dive reports are reconciled.\n"
        "\n"
        "Plans must expose and receipt the GPT-Researcher budget knobs:\n"
        "`MAX_ITERATIONS`, `DEEP_RESEARCH_BREADTH`, `DEEP_RESEARCH_DEPTH`,\n"
        "`DEEP_RESEARCH_CONCURRENCY`, `MAX_SUBTOPICS`, `TOTAL_WORDS`, token limits,\n"
        "`RETRIEVAL_K`, `PAPER_MATCHING_RETRIEVAL_K`, `CANDIDATE_MULTIPLIER`,\n"
        "`RRF_K`, and `TEMPERATURE`. Defaults are `MAX_ITERATIONS=5`,\n"
        "`DEEP_RESEARCH_BREADTH=4`, `DEEP_RESEARCH_DEPTH=3`,\n"
        "`DEEP_RESEARCH_CONCURRENCY=2`, `MAX_SUBTOPICS=8`, `TOTAL_WORDS=3000`,\n"
        "`SMART_TOKEN_LIMIT=8192`, `STRATEGIC_TOKEN_LIMIT=8192`,\n"
        "`FAST_TOKEN_LIMIT=4096`, `RETRIEVAL_K=18`,\n"
        "`PAPER_MATCHING_RETRIEVAL_K=18`, `CANDIDATE_MULTIPLIER=3`, `RRF_K=60`,\n"
        "and `TEMPERATURE=0.0`. The behavioral resume digest includes knobs that\n"
        "change execution or the synthesis prompt; concurrency and fast/strategic\n"
        "token limits are receipt-only until separate parallel or fast-model calls\n"
        "exist.\n"
        "\n"
        "The local GPT-Researcher lane must use a loopback-only endpoint, must record\n"
        "actual provider/model provenance, and must suppress Tavily for strict local\n"
        "runs unless web search is explicitly allowed. Local BGE-M3 embeddings plus\n"
        "local Qwen3.6 35B-A3B may own store-grounded retrieval, evidence expansion,\n"
        "parameter sweeps, and draft findings, but production `analysis/<topic>.md`\n"
        "still uses governed GPT-5.6 Luna/GPT-5.4 deep narrative synthesis unless local\n"
        "validation passes.\n"
    )


def render_token_bounded_indexing_policy() -> str:
    owner_refs = ", ".join(f"`{ref}`" for ref in TOKEN_BOUNDED_INDEXING_OWNER_REFS)
    return (
        "## Token-Bounded Corpus Indexing Policy\n"
        "\n"
        "Generated production plans must include `PG-CORPUS-TOKEN-INDEXING` before "
        "query planning, GPT-Researcher retrieval, paper matching, synthesis, "
        "podcast, deck/presentation, release, or proof. The reusable owners are "
        f"{owner_refs}.\n"
        "\n"
        "The Stage 6 receipt and index digest must record "
        "`token_budget_policy_digest`, `embedding_policy_digest`, "
        "`EMBEDDING_MAX_LENGTH`, `CORPUS_VECTOR_EMBED_BATCH_SIZE`, "
        "`embedding_sidecar_manifest_path`, `embedding_sidecar_manifest_sha256`, "
        "and `chroma_insert_batch_size`. Passing evidence requires "
        "`vector_embeddings_written: true` and embedded chunk counts matching "
        "per-store chunk sidecars. Embedding sidecars are written before bulk "
        "Chroma load; batch-size tuning is receipt evidence, not embedding "
        "identity. Word-count-only sidecars, readiness-only probes, partial "
        "Chroma rows, conference-local embedding reimplementations, and historical exhaustive "
        "extraction output without registry receipt are rejected evidence.\n"
        "\n"
        "Stage 6 is resume-only over completed corpus inputs: it must not "
        "redownload source pages, papers, workshop PDFs, audio, slides, video, "
        "poster media, or unrelated asset families to satisfy indexing.\n"
        "\n"
        "Workshop transcripts must carry talk-level metadata before indexing. "
        "Plans must name `transcription_ocr.workshop_talk_segmentation`, record "
        "`talk_segmentation_provider_not_configured` when no `talks.json` or "
        "governed discovery path exists, and state that audio-only recovery is "
        "rejected evidence for talk-aware transcripts.\n"
    )


def render_conditional_slide_ocr() -> str:
    return (
        "## Slide/Visual OCR Assessment\n"
        "\n"
        "Before committing to slide OCR, plans must run or consume a source-coverage\n"
        "assessment comparing these four sources against downstream evidence needs:\n"
        "transcripts, paper text, source files (existing PDF text layer), and available\n"
        "media metadata.\n"
        "\n"
        "OCR is classified as one of three states:\n"
        "\n"
        "- `required`: corpus lacks sufficient text-bearing artifacts, the brief\n"
        "  explicitly requires slide/visual evidence, or downstream deliverables need\n"
        "  slide-level citations.\n"
        "- `skipped_with_rationale`: available text sources provide sufficient coverage.\n"
        "  Record: available_text_sources (list), expected_coverage_impact (str),\n"
        "  reopen_conditions (list).\n"
        "- `deferred_blocked`: OCR decision requires upstream information not yet\n"
        "  available.\n"
        "\n"
        "When OCR is `required`, the plan must state: why_required (one of the three\n"
        "triggers), expected_receipt_paths (list), provider_runtime_contract (str),\n"
        "cost_rate_limits (str), downstream_blockers (list).\n"
    )


def render_dual_store_paper_matching() -> str:
    return (
        "## Dual-Store Paper Matching\n"
        "\n"
        "Presentation paper matching uses two vector stores: `papers` and\n"
        "`artifacts`. The artifact store contains transcripts, slide OCR, visual\n"
        "descriptions, session/talk evidence, and other non-paper artifacts. The\n"
        "paper store contains only `paper_pdf`, `paper_text`, and\n"
        "`paper_metadata_fallback` chunks. Full paper PDF/text chunks prepend paper\n"
        "title, section title, and subsection title when known; title-plus-abstract\n"
        "fallback is recorded in metadata.\n"
        "\n"
        "The required order is artifact-store GPT-Researcher -> frontier-model\n"
        "keypoint extraction -> paper-store GPT-Researcher -> returned paper titles\n"
        "in `matched_papers.json`. The default configuration is\n"
        "`workflow.paper_matching.keypoint_provider: openai`,\n"
        "`workflow.paper_matching.keypoint_model: gpt-5.6-sol`,\n"
        "`artifact_research_store: artifacts`, and `paper_research_store: papers`.\n"
        "`scripts/rank_papers.py` is legacy/deprecated for NeurIPS 2025\n"
        "compatibility only; reusable plans use\n"
        "`scripts/match_papers_from_artifact_keypoints.py`.\n"
    )


def render_fail_stop_pipeline_policy() -> str:
    return (
        "## Fail-Stop Pipeline Policy\n"
        "\n"
        "Required workflow stages fail closed: `status: blocked`,\n"
        "`status: deferred_blocked`, `status: failed`, `status: unavailable`,\n"
        "and `status: unsupported` stop the whole pipeline before downstream\n"
        "podcast, deck, release, proof, or publication work. Blocked and\n"
        "unavailable stages write `BlockerReceipt` evidence instead of being\n"
        "treated as successful completion.\n"
        "\n"
        "Required blocker classes include `ocr_deferred_blocked`,\n"
        "`stage_blocked`, `stage_unavailable`, `local_llm_not_ready`,\n"
        "`missing_receipt`, `threshold_violation`, and `stale_receipt`.\n"
        "Provider availability failures, missing live owner commands,\n"
        "unsupported live stages, missing required receipts, stale receipts, and\n"
        "failed required validation stop the run. Resume remains blocked until an\n"
        "`UnblockReceipt` is written through `workflow unblock` or\n"
        "`conference_pipeline.workflow_unblock`; summaries expose `halt_status`,\n"
        "`halt_stage_id`, `blocker_receipt_path`, and\n"
        "`unblock_acknowledgements`.\n"
    )


def render_pre_spend_contract_gate_policy() -> str:
    return (
        "## Pre-Spend Contract Gate\n"
        "\n"
        "Every paid provider stage must evaluate its stage binding's "
        "`runtime_contract` `required_receipts`/`required_checks` in "
        "precondition mode BEFORE the first provider call -- this is "
        "dependency-order enforcement, distinct from the existing budget/cost "
        "estimate gate. A precondition failure writes a zero-spend blocker "
        "with code `precondition_contract_unmet` and stops the stage before "
        "any provider is invoked; no partial spend is permitted while the "
        "precondition is unmet.\n"
    )


def render_incremental_receipts_and_resume_economics_policy() -> str:
    return (
        "## Incremental Receipts And Resume Economics\n"
        "\n"
        "Multi-unit stages must refresh their top-level receipt after every "
        "completed unit (or write per-unit receipts plus a rollup) so a "
        "mid-run kill cannot mask N/M completed units as 0/M. Paid artifacts "
        "are reuse-by-default on `--resume`: previously produced, still-valid "
        "artifacts are reused rather than regenerated, and any explicit "
        "regeneration must be stated in the plan by name -- a silent "
        "regenerate-all pattern (e.g. `reuse_existing_pass_reports=False` with "
        "no plan-level justification) is forbidden.\n"
    )


def render_provider_substitution_governance_policy() -> str:
    return (
        "## Provider-Substitution Governance And Credential Preflight\n"
        "\n"
        "Every provider-backed stage's `runtime_contract` must equals-check "
        "the receipt's actual `provider` and `model` fields against the "
        "approved PC-* provider contract list and require "
        "`provider_calls_made: true` (and `boundary_method` where "
        "applicable), reusing the existing `equals`/`one_of`/`not_equals` "
        "predicate vocabulary.\n"
        "\n"
        "The ONLY governed path around an approved provider contract is a "
        "`provider_substitution_waiver.json` matching the schema in "
        "`references/provider-substitution-waiver.md` (`task_id`, `from` "
        "{provider, model}, `to` {provider, model}, `approver`, `rationale`, "
        "`cost_impact`, `expiry`). A bare manual-ack UnblockReceipt does NOT "
        "count as a governed substitution.\n"
        "\n"
        "Before a multi-day run, a Phase-1 cloud-credential preflight probe "
        "makes one cheap live call per provider named in Task Model Defaults "
        "and writes a preflight receipt, mirroring the already-mandated "
        "local LLM readiness preflight.\n"
        "\n"
        "Blocker classes `unauthorized_provider_substitution` and "
        "`credential_invalid` fail closed on an unapproved provider swap or "
        "an invalid/expired credential.\n"
    )


def render_gsd_lifecycle_and_orchestrator_conformance_policy() -> str:
    return (
        "## GSD Lifecycle And Orchestrator Conformance\n"
        "\n"
        "Hard rule: no live execution of Phase N without `phases/NN-*/` PLAN "
        "plus plan-review/verification artifacts. STATE.md and this ROADMAP "
        "must never advance a phase to live execution status without those "
        "artifacts present.\n"
        "\n"
        "Any supervisor or orchestration wrapper must be plan-declared and "
        "execute a machine-readable stage sequence emitted into "
        "`approval-summary.json` as `orchestrator_stage_sequence`; it is "
        "forbidden from provider fallbacks outside the approved contract "
        "rows. The proof stage must reject receipts produced by an "
        "undeclared orchestrator.\n"
    )


def render_roadmap(brief: dict[str, object]) -> str:
    identity = milestone_identity(brief)
    slug = conference_slug(brief)
    mapping = render_official_workflow_mapping(workflow_mapping_for_brief(brief))
    registry_policies = _registry_policy_snapshots()
    task_defaults = render_task_model_defaults(task_model_default_rows(registry_policies))
    provider_contracts = render_benchmark_aware_provider_contracts(benchmark_aware_provider_contracts())
    script_registry = stage_script_registry_snapshot()
    script_registry_tool_table = render_stage_script_registry_tool_table(
        script_registry.get("tools")
    )
    script_registry_section = (
        "## Stage Script Registry\n\n"
        "Generated planning and production work must use the unified registry at "
        "`src/conference_pipeline/stage_owner_registry.json` for required, "
        "allowed, deprecated, and forbidden stage scripts. Tool instructions must "
        "cite the registry-owned `usage_doc_refs`; do not copy undocumented script "
        "usage into plans or skills. When a registry tool fails, read its "
        "`diagnostic.next_actions` before retrying or advancing the plan. Live "
        "production stages that can write `status: \"succeeded\"` must be "
        "registry-bound through `stage_bindings.<stage>.runtime_contract`, call "
        "`conference_pipeline.receipt_lineage.finalize_receipt` (v5.1 lineage: bind on the provisional receipt of a fresh run, pass the recorded lineage through on resume) and `conference_pipeline.stage_contracts.verify_stage_contract`, and embed "
        "`stage_contract.accepted_evidence` plus `stage_contract.blockers` in "
        "their receipts; unsupported stages must be explicitly marked "
        "non-production.\n\n"
        f"- schema_version: `{script_registry.get('schema_version')}`\n"
        f"- sha256: `{script_registry.get('sha256')}`\n"
        f"- verification command: `{script_registry.get('verification_command')}`\n"
        f"- workspace gitignore generator: `{WORKSPACE_GITIGNORE_WRITE_COMMAND.replace('<slug>', slug)}`\n"
        f"- workspace gitignore check: `{WORKSPACE_GITIGNORE_CHECK_COMMAND.replace('<slug>', slug)}`\n"
        f"\n{script_registry_tool_table}\n"
    )
    blocks = []
    for index, area in enumerate(ROADMAP_AREAS, start=1):
        req = REQUIREMENT_IDS[min(index - 1, len(REQUIREMENT_IDS) - 1)]
        extra = ""
        if area == "provider cost gate":
            extra = (
                "6. Estimate command is recorded before paid provider work: "
                f"`.venv/bin/python -m conference_pipeline.costs estimate --workspace conferences/{slug} --scope full --scenario base --json`.\n"
            )
        if area == "dual-store paper matching":
            extra = (
                "6. Paper matching uses artifact-store GPT-Researcher findings, frontier-model keypoints, and paper-store GPT-Researcher returned titles.\n"
                "7. `matched_papers.json` is canonical; `ranked_papers.json` is compatibility only; `scripts/rank_papers.py` is deprecated.\n"
            )
        if area == "workshop discovery & corpus freeze":
            extra = (
                "6. Discovery owner records per-workshop inventory at `conferences/"
                + slug
                + "/storage/workshops/inventory.json` via `"
                + WORKSHOP_DISCOVERY_OWNER_COMMAND.replace("<slug>", slug)
                + "`.\n"
                "7. Per-workshop decisions carry one of `approved`, `metadata_only_acceptable`, `external_site_required`, or `excluded` with rationale.\n"
                "8. `openreview_with_pdf` venues are downloaded via `"
                + WORKSHOP_PAPER_DOWNLOAD_OWNER_COMMAND.replace("<slug>", slug)
                + "` at ≤1.4 rps with HTTP 429 backoff and resume.\n"
                "9. `openreview_metadata_only` and `no_openreview_record` venues require an external-site downloader scaffold under `scripts/conference/"
                + slug
                + "/workshop_site_downloaders/<workshop_slug>.py` and a tracker entry; production audit rejects stub-only scaffolds.\n"
            )
        if area == "narrative analysis provenance":
            extra = (
                "6. Public `analysis/<topic>.md` files are deep reports with Executive Summary, Key Findings, Detailed Analysis, References/Sources, Cross-References, Recommendations, Questions, and a compact Slide Brief section.\n"
                "7. Deck-facing summaries are written to `analysis/slide_briefs/<topic_id>.md`, while raw citation ledgers are written to `analysis/provenance/<topic_id>.md`, `analysis/provenance/<topic_id>.json`, and `analysis/analysis_citation_manifest.json`.\n"
                "8. Production narrative synthesis uses `analysis.deep_narrative_synthesis` (`openai:gpt-5.6-sol`, fallback `openai:gpt-5.6-terra`) and fails closed when the LLM/model, iterative GPT-Researcher evidence, or deep validation gates are missing.\n"
                "9. Local Qwen3.6 35B-A3B plus BGE-M3 GPT-Researcher findings are the default evidence-input side when receipts prove corpus-only hybrid retrieval, but cannot replace final narrative synthesis unless they pass the same deep validation gates.\n"
                "10. Topic strategy owner proof records `analysis.topic_strategy_generation`, `conference_pipeline.analysis.topic_strategy`, `main_keynote`, `main_oral`, `session_title`, `workshop_segmented_talk`, `topic_count`, and `analysis/topic_strategy.json`.\n"
                "11. Full-read owner proof records `PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION`, `PG-EXHAUSTIVE-TOPIC-SUMMARIES`, `analysis.llm_full_read_extraction`, `conference_pipeline.extractions.llm_full_read.run_llm_full_read`, `14A-llm-full-read-extraction-receipt.json`, `07A-exhaustive-topic-summaries.json`, `llm-full-read/by-unit`, and `llm-full-read/by-topic`.\n"
                "12. Deep-dive topics first run `analysis.deep_dive_question_selection` through `conference_pipeline.research.deep_dive_lanes`, `src/conference_pipeline/research/deep_dive_lanes.py`, and `scripts/select_deep_dive_questions.py` with GPT-5.6 Luna to select 4-6 questions per topic from exhaustive summaries; one-run-per-key-point expansion is rejected.\n"
                "13. Each selected question runs both the global no-topic-id `workflow.research.gpt_researcher` / `global_no_topic_id` lane and the `workflow.research.heuristic_prefiltered_gpt_researcher` lane owned by `analysis.gpt_researcher_heuristic_prefilter`, `conference_pipeline.research.heuristic_prefilter`, `src/conference_pipeline/research/heuristic_prefilter.py`, and `scripts/run_heuristic_prefiltered_research.py`.\n"
                "14. Heuristic deep-dive evidence records `heuristic_keyword_prefilter`, `keyword_seed_source`, `prefilter_terms_sha256`, `candidate_count`, `selected_evidence_ids`, `stores_searched`, `source_family_scope`, `retrieval_scope`, and `invalid_citation_ids`.\n"
                "15. Rejected evidence: topic-id filtered deep dives, unrecorded keyword lists, source-family leakage, and citation-invalid heuristic reports.\n"
                "16. The dual runner `analysis.dual_deep_dive_research_lanes` records `workflow.research.dual_deep_dive_gpt_researcher_lanes` via `scripts/run_dual_deep_dive_research.py --resume`.\n"
                "17. `analysis.topic_reconciliation_reference_workflow` runs `conference_pipeline.analysis.topic_reconciliation_reference_workflow` via `scripts/run_topic_reconciliation_reference_workflow.py`, packages 07D reconciliation, 07E deterministic validation, deterministic safety repair, bounded 07G candidate resolution, fallback 07F repair, and final 07E validation, and writes `07H-topic-reconciliation-reference-workflow.json` with zero `final_invalid_reference_count`, `final_unknown_reference_count`, `final_title_mismatch_count`, and `final_bare_reference_warning_count` before final narrative or slide synthesis.\n"
                "18. `analysis.topic_reconciliation` runs `conference_pipeline.analysis.topic_reconciliation` via `scripts/reconcile_topic_analysis.py` with `mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`, writes `07D-topic-reconciliation.json` and `analysis/topic_reconciliations/<topic_id>.md`, and injects `topic_reconciliation` into source packs before final narrative or slide synthesis.\n"
                "19. `analysis.topic_reference_validation` runs `conference_pipeline.analysis.topic_reference_validation` via `scripts/validate_topic_reconciliation_references.py`, writes `07E-topic-reference-validation.json`, `analysis/topic_reference_ledgers/<topic_id>.json`, and `analysis/topic_reference_validation/<topic_id>.json`, and requires positive `ledger_entry_count` and `citation_count` plus zero `invalid_reference_count`, `unknown_reference_count`, `title_mismatch_count`, and `bare_reference_warning_count` before final narrative or slide synthesis.\n"
                "20. `analysis.topic_reference_candidate_resolution` runs `conference_pipeline.analysis.topic_reference_candidate_resolution` via `scripts/resolve_topic_reference_candidates.py` when 07E blocks, writes `07G-topic-reference-candidate-resolution.json`, `analysis/topic_reference_candidate_resolution/<topic_id>.json`, candidate packets, and Qwen decision sidecars, records `topic_reconciliation.reference_candidate_resolution`, and applies repaired markdown only when `final_validation` has zero invalid references.\n"
                "21. `analysis.topic_reference_repair` runs `conference_pipeline.analysis.topic_reference_repair` via `scripts/repair_topic_reconciliation_references.py` only after unresolved candidate-resolution blockers, writes `07F-topic-reference-repair.json` and `analysis/topic_reference_repairs/<topic_id>.json`, records `topic_reconciliation.reference_repair`, `packet_truncated_count`, and `applied_topic_count`, and applies repaired markdown only when `repaired_validation` has zero invalid references before rerunning 07E validation.\n"
                "22. `Source Basis`, `Verbatim transcript/OCR evidence`, `Matched Paper Evidence`, and `Source Gaps` are rejected as top-level public-report sections.\n"
            )
        blocks.append(
            f"""### Phase {index}: {area.title()}

**Goal:** Plan {area} for {identity} with dry-run/mock only proof boundaries.
**Requirements:** {req}
**Success Criteria:**
1. Support labels include dry-run/mock only, unsupported, planned, and needs live proof where live proof is absent.
2. Event-family inventory, interest-group decisions, exclusion records, and production handoff fields are explicit or blocked.
3. Model-governance task ids and benchmark requirements are preserved for downstream production planning.
4. Canonical References cite docs/source-of-truth.md and docs/workflows/pipeline-workflow.md.
5. The phase produces auditable planning evidence without live provider calls.
{extra}**Canonical References:** docs/source-of-truth.md; docs/workflows/pipeline-workflow.md; docs/reference/pipeline-capabilities.md; src/conference_pipeline/model_governance/registry.yaml.
"""
        )
    extraction_policy = render_extraction_provider_policy()
    gpt_researcher_policy = render_gpt_researcher_retrieval_policy()
    token_bounded_indexing_policy = render_token_bounded_indexing_policy()
    conditional_ocr = render_conditional_slide_ocr()
    paper_matching = render_dual_store_paper_matching()
    fail_stop_policy = render_fail_stop_pipeline_policy()
    pre_spend_contract_gate_policy = render_pre_spend_contract_gate_policy()
    incremental_receipts_resume_economics_policy = render_incremental_receipts_and_resume_economics_policy()
    provider_substitution_governance_policy = render_provider_substitution_governance_policy()
    gsd_lifecycle_policy = render_gsd_lifecycle_and_orchestrator_conformance_policy()
    return f"# Roadmap: {identity}\n\n{mapping}\n{task_defaults}\n{provider_contracts}\n{script_registry_section}\n{fail_stop_policy}\n{pre_spend_contract_gate_policy}\n{incremental_receipts_resume_economics_policy}\n{provider_substitution_governance_policy}\n{gsd_lifecycle_policy}\n{extraction_policy}\n{gpt_researcher_policy}\n{token_bounded_indexing_policy}\n{paper_matching}\n{conditional_ocr}\n" + "\n".join(blocks) + "\n"


def render_state(brief: dict[str, object]) -> str:
    identity = milestone_identity(brief)
    return f"""---
status: planning
milestone: "{identity}"
---

# State

Current focus: {identity}

Support claims: dry-run/mock only; unsupported live support needs live proof.

Hard rule: no live execution of Phase N without `phases/NN-*/` PLAN plus
plan-review/verification artifacts.
"""


def render_readme(brief: dict[str, object], slug: str) -> str:
    return f"""# {safe_text(brief['conference_name'])} Workspace

Planning context:
`.planning/milestones/{context_filename(brief)}`

This scaffold is dry-run/mock only. Live support needs live proof.
"""


def write_file(root: Path, relative: str, content: str) -> Path:
    target = ensure_inside(root / relative, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _registry_policy_snapshots() -> dict[str, dict[str, object]]:
    if load_registry is None:
        return {
            task_id: {
                "provider": "unresolved",
                "model": "unresolved",
                "fallback_provider": None,
                "fallback_model": None,
                "stage": "unresolved",
                "current_research": {"required": True, "status": "registry unavailable"},
                "benchmark": {"required": True, "receipt_required": True, "status": "registry unavailable"},
                "local_runtime_policy": "deferred_until_registry_available",
                "selection_status": "blocked_until_registry_available",
            }
            for task_id in MODEL_REGISTRY_TASK_IDS
        }
    registry = load_registry()
    snapshots: dict[str, dict[str, object]] = {}
    for task_id in MODEL_REGISTRY_TASK_IDS:
        policy = registry.get(task_id)
        snapshots[task_id] = {
            "provider": policy.provider,
            "model": policy.model,
            "fallback_provider": policy.fallback_provider,
            "fallback_model": policy.fallback_model,
            "stage": policy.stage,
            "current_research": policy.current_research,
            "benchmark": policy.benchmark,
            "local_runtime_policy": policy.local_runtime_policy,
            "selection_status": policy.selection_status,
        }
    return snapshots


def stage_script_registry_snapshot() -> dict[str, object]:
    if load_stage_script_registry is None or registry_sha256 is None:
        return {
            "path": "src/conference_pipeline/stage_owner_registry.json",
            "schema_version": "unavailable",
            "sha256": "registry unavailable",
            "verification_command": STAGE_SCRIPT_VERIFIER_COMMAND,
            "status": "blocked_until_registry_available",
        }
    try:
        registry = load_stage_script_registry()
        return {
            "path": "src/conference_pipeline/stage_owner_registry.json",
            "schema_version": registry.get("schema_version"),
            "sha256": registry_sha256(),
            "script_count": len(registry.get("scripts") or {}),
            "stage_binding_count": len(registry.get("stage_bindings") or {}),
            "task_binding_count": len(registry.get("task_bindings") or {}),
            "planning_required_task_ids": registry.get("planning_required_task_ids", []),
            "tools": stage_script_registry_tool_rows(registry),
            "verification_command": STAGE_SCRIPT_VERIFIER_COMMAND,
            "status": "required_for_plan_and_production_verification",
        }
    except Exception as exc:
        return {
            "path": "src/conference_pipeline/stage_owner_registry.json",
            "schema_version": "error",
            "sha256": "registry unavailable",
            "verification_command": STAGE_SCRIPT_VERIFIER_COMMAND,
            "status": "blocked_until_registry_available",
            "error": str(exc),
        }


def stage_script_registry_tool_rows(registry: dict[str, object]) -> list[dict[str, object]]:
    scripts = registry.get("scripts") if isinstance(registry, dict) else {}
    stage_bindings = registry.get("stage_bindings") if isinstance(registry, dict) else {}
    task_bindings = registry.get("task_bindings") if isinstance(registry, dict) else {}
    rows: list[dict[str, object]] = []
    if not isinstance(scripts, dict):
        return rows
    for script_id, script in sorted(scripts.items()):
        if not isinstance(script, dict):
            continue
        stages = [
            stage_id
            for stage_id, binding in sorted((stage_bindings or {}).items())
            if isinstance(binding, dict)
            and script_id
            in set(binding.get("required_script_ids", []) or [])
            | set(binding.get("optional_script_ids", []) or [])
            | set(binding.get("forbidden_script_ids", []) or [])
        ]
        tasks = [
            task_id
            for task_id, binding in sorted((task_bindings or {}).items())
            if isinstance(binding, dict)
            and script_id
            in set(binding.get("required_script_ids", []) or [])
            | set(binding.get("approved_script_ids", []) or [])
            | set(binding.get("optional_script_ids", []) or [])
            | set(binding.get("forbidden_script_ids", []) or [])
        ]
        rows.append(
            {
                "script_id": script_id,
                "path": script.get("path", ""),
                "status": script.get("status", ""),
                "scope": script.get("scope", ""),
                "stage_bindings": stages,
                "task_bindings": tasks,
                "usage_doc_refs": script.get("usage_doc_refs", []),
                "diagnostic_profile": script.get("diagnostic_profile", {}),
            }
        )
    return rows


def render_stage_script_registry_tool_table(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "_Registry tool table unavailable until the registry loads._\n"
    lines = [
        "| Script ID | Status | Scope | Path | Stage/task bindings | Usage docs | Agent diagnostic |",
        "|-----------|--------|-------|------|---------------------|------------|------------------|",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        bindings = sorted(
            set(row.get("stage_bindings", []) or []) | set(row.get("task_bindings", []) or [])
        )
        usage_refs = row.get("usage_doc_refs", []) or []
        profile = row.get("diagnostic_profile", {}) or {}
        if isinstance(profile, dict):
            agent_diagnostic = str(profile.get("success_summary") or "MISSING")
            blocked_actions = profile.get("blocked_next_actions", []) or []
            if blocked_actions:
                agent_diagnostic += " Blocked: " + str(blocked_actions[0])
        else:
            agent_diagnostic = "MISSING"
        lines.append(
            "| "
            + " | ".join(
                safe_text(str(value))
                for value in (
                    f"`{row.get('script_id', '')}`",
                    f"`{row.get('status', '')}`",
                    f"`{row.get('scope', '')}`",
                    f"`{row.get('path', '')}`",
                    ", ".join(f"`{item}`" for item in bindings) or "-",
                    ", ".join(f"`{item}`" for item in usage_refs) or "MISSING",
                    agent_diagnostic,
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def workspace_gitignore_policy(slug: str) -> dict[str, object]:
    write_command = WORKSPACE_GITIGNORE_WRITE_COMMAND.replace("<slug>", slug)
    check_command = WORKSPACE_GITIGNORE_CHECK_COMMAND.replace("<slug>", slug)
    return {
        "policy_source": f"conferences/{slug}/storage/storage_manifest.json",
        "policy_location": f"conferences/{slug}/.gitignore",
        "generator_command": write_command,
        "verification_command": check_command,
        "receipt_path": f"conferences/{slug}/storage/gitignore_policy_receipt.json",
        "tracked_by_default": ["storage_class: git"],
        "ignored_by_default": ["storage_class: public", "storage_class: backup", "storage_class: cache"],
        "exception_field": "gitignore_exceptions",
        "payload_rule": "Drive-owned public/backup artifacts and cache outputs stay out of git",
    }


def production_next_step(slug: str) -> dict[str, object]:
    return {
        "skill": "plan-conference-production",
        "command": "$plan-conference-production",
        "purpose": "generate live-only GSD production work that actually produces the conference analysis from this finalized analysis specification",
        "input": "finalized plan-conference-analysis spec root",
        "workspace": f"conferences/{slug}",
        "boundary": "creates an approval-gated live execution phase or milestone; rejects analysis planning and non-live artifacts as live production evidence",
        "preconditions": [
            "analysis specification finalized",
            "provider cost posture reviewed",
            "live credentials and source access approved before execution",
            "receipt-backed production gates remain required",
        ],
    }


_LOCAL_PROOF_FIELDS = (
    "research_ref",
    "hardware_fit_ref",
    "benchmark_receipt_ref",
    "operator_approval_ref",
)


def _brief_workshop_payload(brief: dict[str, object]) -> dict[str, object]:
    payload = brief.get("workshop_coverage", {})
    if isinstance(payload, dict):
        return payload
    return {}


def workshop_coverage_posture_block(brief: dict[str, object]) -> dict[str, object]:
    """Return the workshop_coverage_posture entry for the production handoff.

    The brief may declare `workshop_coverage.posture` (one of
    WORKSHOP_COVERAGE_POSTURES). If unset, the block records
    `status: pending_user_decision` so the analysis audit can flag it when
    workshops are referenced in target outputs or roadmap.
    """
    payload = _brief_workshop_payload(brief)
    posture = str(payload.get("posture") or "")
    if posture and posture not in WORKSHOP_COVERAGE_POSTURES:
        posture = ""
    return {
        "posture": posture,
        "allowed_postures": list(WORKSHOP_COVERAGE_POSTURES),
        "openreview_parent_venue": payload.get("openreview_parent_venue") or "",
        "discovery_owner_command_template": WORKSHOP_DISCOVERY_OWNER_COMMAND,
        "download_owner_command_template": WORKSHOP_PAPER_DOWNLOAD_OWNER_COMMAND,
        "status": "approved" if posture else "pending_user_decision",
    }


def workshop_inventory_contract_block(
    brief: dict[str, object], slug: str
) -> dict[str, object]:
    payload = _brief_workshop_payload(brief)
    workshops_input = payload.get("workshops")
    workshops: list[dict[str, object]] = []
    if isinstance(workshops_input, list):
        for row in workshops_input:
            if not isinstance(row, dict):
                continue
            workshops.append(
                {
                    "venueid": str(row.get("venueid") or ""),
                    "pdf_availability_class": str(row.get("pdf_availability_class") or ""),
                    "paper_count": int(row.get("paper_count") or 0),
                    "decision": str(row.get("decision") or ""),
                    "decision_rationale": str(row.get("decision_rationale") or ""),
                    "external_site_url": row.get("external_site_url") or None,
                    "external_scaffold_path": row.get("external_scaffold_path") or None,
                }
            )
    return {
        "inventory_path": f"conferences/{slug}/storage/workshops/inventory.json",
        "tracker_path": f"conferences/{slug}/storage/workshops/workshop_site_tracker.json",
        "pdf_availability_classes": list(WORKSHOP_PDF_AVAILABILITY_CLASSES),
        "allowed_decisions": list(WORKSHOP_DECISION_VALUES),
        "workshops": workshops,
        "status": "populated" if workshops else "empty_until_discovery",
        "external_site_required_class_handling": (
            "openreview_metadata_only and no_openreview_record venues require an "
            "external-site downloader scaffold under "
            "scripts/conference/<slug>/workshop_site_downloaders/<workshop_slug>.py "
            "and a workshop_site_tracker.json entry; production audit rejects "
            "stub-only scaffolds for in-scope workshops."
        ),
    }


def structured_production_gates() -> list[dict[str, object]]:
    """Machine-checkable analysis-phase gates: gate id -> receipt path -> field checks.

    Emitted alongside the existing prose PRODUCTION_WORKFLOW_GATES table (no
    removal). Fixes the review defect where PG-TOPIC-STRATEGY, PG-QUERY,
    PG-GPT-RESEARCHER, and PG-PAPER-MATCHING all pointed at the single
    receipt `stage-07-analysis.json`, making per-gate checking impossible --
    each gate below carries a DISTINCT `receipt_path`.
    """

    return [
        {
            "gate_id": "PG-TOPIC-STRATEGY",
            "receipt_path": "storage/analysis/topic-strategy/<run-id>/topic_strategy_receipt.json",
            "field_checks": [{"path": "status", "equals": "succeeded"}],
        },
        {
            "gate_id": "PG-QUERY",
            "receipt_path": "storage/workflow-runs/<run-id>/07-query-planning.json",
            "field_checks": [{"path": "status", "equals": "succeeded"}],
        },
        {
            "gate_id": "PG-GPT-RESEARCHER",
            "receipt_path": "storage/workflow-runs/<run-id>/07C-dual-deep-dive-research-lanes.json",
            "field_checks": [
                {"path": "status", "equals": "succeeded"},
                {"path": "provider_calls_made", "truthy": True},
            ],
        },
        {
            "gate_id": "PG-PAPER-MATCHING",
            "receipt_path": "decks/<id>/ranked_papers.json",
            "field_checks": [{"path": "status", "equals": "succeeded"}],
        },
    ]


def production_handoff(brief: dict[str, object], slug: str) -> dict[str, object]:
    policies = _registry_policy_snapshots()

    local_proof_posture: dict[str, object] = {}
    if load_registry is not None:
        try:
            registry = load_registry()
            for task_id in MODEL_REGISTRY_TASK_IDS:
                try:
                    policy = registry.get(task_id)
                    proof = getattr(policy, "local_proof", {}) or {}
                    if all(proof.get(f) for f in _LOCAL_PROOF_FIELDS):
                        local_proof_posture[task_id] = {
                            k: proof[k]
                            for k in _LOCAL_PROOF_FIELDS + ("approved_date", "scope")
                            if proof.get(k)
                        }
                    else:
                        local_proof_posture[task_id] = "deferred"
                except Exception:
                    local_proof_posture[task_id] = "deferred"
        except Exception:
            local_proof_posture = {task_id: "deferred" for task_id in MODEL_REGISTRY_TASK_IDS}
    else:
        local_proof_posture = {task_id: "deferred" for task_id in MODEL_REGISTRY_TASK_IDS}

    return {
        "schema_version": 1,
        "next_step": production_next_step(slug),
        "evidence_class": {
            "required": "receipt-backed live production evidence",
            "rejected": [
                "dry-run receipts",
                "mock receipts",
                "metadata-only packets",
                "readiness-only files",
                "needs live proof labels",
            ],
            "status": "blocked_until_live_receipts",
        },
        "corpus_bounds": {
            "default_scope": "all discovered conference items unless explicitly excluded",
            "status": "requires_source_backed_inventory",
        },
        "inventory_status": {
            "status": "unverified",
            "event_families": EVENT_FAMILY_INVENTORY,
        },
        "media_family_scope": {
            "default": "all_media_bearing_families",
            "excluded_families": [],
            "decision_source": "pending_user_exclusion_question",
            "status": "blocking_until_user_answers_exclusion_question",
            "policy": (
                "opt-out only: all media-bearing event families are in scope "
                "unless the user explicitly excludes a family; production "
                "re-confirms before acquisition and the download stage fails "
                "closed on zero item pages for any non-excluded family "
                "(program_family_media_items_missing)"
            ),
        },
        "interest_group_decisions": {
            "status": "blocking_until_user_approval",
            "decisions": [],
            "required_decisions": ["approved", "edited", "rejected", "complete"],
        },
        "exclusions": {
            "status": "none_recorded",
            "records": [],
            "required_fields": ["scope", "reason", "decision context", "source note", "downstream effect"],
        },
        "implementation_inventory": {
            "status": "required_before_production_planning",
            "policy": "map every governed task id to a stage script registry owner command, implementation task, or blocker",
            "required_owner_commands": {
                "planning.workspace_gitignore_policy": WORKSPACE_GITIGNORE_WRITE_COMMAND.replace("<slug>", slug),
                "podcast.script_generation": PODCAST_SCRIPT_GENERATION_OWNER_COMMAND,
                "podcast.tts": PODCAST_TTS_OWNER_COMMAND,
                "presentation.blueprint_generation": PRESENTATION_BLUEPRINT_GENERATION_OWNER_COMMAND,
                "presentation.blueprint_review": PRESENTATION_BLUEPRINT_REVIEW_OWNER_COMMAND,
                "presentation.repair_presentation_blueprint": PRESENTATION_BLUEPRINT_REPAIR_OWNER_COMMAND,
                "presentation.deck_contract_generation": PRESENTATION_DRAFT_GENERATION_OWNER_COMMAND,
                "presentation.executive_overview_decks": PRESENTATION_EXECUTIVE_OVERVIEW_DECKS_OWNER_COMMAND,
                "presentation.source_binding": PRESENTATION_RESEARCH_ENRICHMENT_OWNER_COMMAND,
                "presentation.render_spec_generation": PRESENTATION_SLIDE_CONSOLIDATION_OWNER_COMMAND,
                "presentation.render_export": PRESENTATION_REVEALJS_RENDER_OWNER_COMMAND,
                "presentation.svg_figure_generation": PRESENTATION_PROGRAMMATIC_FIGURE_OWNER_COMMAND,
                "presentation.figure_semantic_review": PRESENTATION_FIGURE_SEMANTIC_REVIEW_OWNER_COMMAND,
                "presentation.figure_validation": PRESENTATION_FIGURE_VALIDATION_OWNER_COMMAND,
                "presentation.delivery_generation": PRESENTATION_PRESENTER_SCRIPT_OWNER_COMMAND,
                "presentation.editorial_gate": PRESENTATION_EDITORIAL_GATE_OWNER_COMMAND,
                "presentation.video_generation": PRESENTATION_VIDEO_GENERATION_OWNER_COMMAND,
                "presentation.presentation_review": PRESENTATION_PRESENTATION_REVIEW_OWNER_COMMAND,
                "presentation.link_audit": PRESENTATION_LINK_AUDIT_OWNER_COMMAND,
                "presentation.visual_qc": PRESENTATION_VISUAL_QC_OWNER_COMMAND,
                "presentation.repair": PRESENTATION_REPAIR_OWNER_COMMAND,
                "acquisition.workshop_discovery": WORKSHOP_DISCOVERY_OWNER_COMMAND.replace("<slug>", slug),
                "acquisition.workshop_paper_download": WORKSHOP_PAPER_DOWNLOAD_OWNER_COMMAND.replace("<slug>", slug),
                "acquisition.workshop_talk_media": WORKSHOP_TALK_MEDIA_OWNER_COMMAND.replace("<slug>", slug),
                "acquisition.embedded_media_recovery": EMBEDDED_MEDIA_RECOVERY_OWNER_REF,
                "acquisition.slideslive_media_recovery": SLIDESLIVE_MEDIA_RECOVERY_OWNER_COMMAND.replace("<slug>", slug),
                "acquisition.slideslive_slide_download": SLIDESLIVE_SLIDE_DOWNLOAD_OWNER_COMMAND.replace("<slug>", slug),
                "acquisition.slide_reference_detection": SLIDE_REFERENCE_DETECTION_OWNER_COMMAND.replace("<slug>", slug),
                "acquisition.slide_cleanup": SLIDE_CLEANUP_OWNER_COMMAND.replace("<slug>", slug),
                "indexing.token_bounded_chunking": " ; ".join(TOKEN_BOUNDED_INDEXING_OWNER_REFS),
                "indexing.local_embedding_materialization_policy": "conference_pipeline.indexing.corpus_vector_store; src/conference_pipeline/indexing/corpus_vector_store.py; src/conference_pipeline/indexing/embeddings.py; src/conference_pipeline/indexing/mlx_embeddings.py",
                "transcription_ocr.workshop_talk_boundary_discovery": WORKSHOP_TALK_BOUNDARY_DISCOVERY_OWNER_COMMAND.replace("<slug>", slug),
                "transcription_ocr.workshop_talk_segmentation": WORKSHOP_TALK_SEGMENTATION_OWNER_COMMAND.replace("<slug>", slug),
                "analysis.topic_reconciliation_reference_workflow": "scripts/run_topic_reconciliation_reference_workflow.py --workspace conferences/<slug> --run-id <run-id> --mode live --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --resume --summary; conference_pipeline.analysis.topic_reconciliation_reference_workflow; 07H-topic-reconciliation-reference-workflow.json; 07D-topic-reconciliation.json; 07E-topic-reference-validation.json; 07G-topic-reference-candidate-resolution.json; 07F-topic-reference-repair.json; final_validation_status; final_invalid_reference_count; final_unknown_reference_count; final_title_mismatch_count; final_bare_reference_warning_count",
                "analysis.topic_reference_validation": "scripts/validate_topic_reconciliation_references.py --workspace conferences/<slug> --run-id <run-id> --summary; conference_pipeline.analysis.topic_reference_validation; 07E-topic-reference-validation.json; analysis/topic_reference_ledgers; analysis/topic_reference_validation; ledger_entry_count; citation_count; bare_reference_warning_count",
                "analysis.topic_reference_candidate_resolution": "scripts/resolve_topic_reference_candidates.py --workspace conferences/<slug> --run-id <run-id> --mode live --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --topic <topic_id> --apply --summary; conference_pipeline.analysis.topic_reference_candidate_resolution; 07G-topic-reference-candidate-resolution.json; analysis/topic_reference_candidate_resolution; reference_candidate_resolution; qwen_candidate_decisions; final_validation",
                "analysis.topic_reference_repair": "scripts/repair_topic_reconciliation_references.py --workspace conferences/<slug> --run-id <run-id> --mode live --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --topic <topic_id> --apply --summary; conference_pipeline.analysis.topic_reference_repair; 07F-topic-reference-repair.json; analysis/topic_reference_repairs; reference_repair; packet_truncated_count; applied_topic_count; repaired_validation; repaired_invalid_reference_count",
            },
            "conditional_acquisition_recovery_policy": (
                "If Stage 4 leaves successful cached item_page_auth_probe pages with "
                "embedded SlidesLive media but no merged local audio/video artifact, "
                "run acquisition.slideslive_media_recovery or a conference-local "
                "equivalent before transcription/OCR. The recovery scans only cached "
                "item pages, writes storage/download-corpus/slideslive_media_recovery.json, "
                "and must not redownload source pages, paper PDFs, workshop PDFs, "
                "poster images, or unrelated media families."
            ),
            "slideslive_slide_acquisition_policy": (
                "If visual/OCR evidence is in scope and cached item_page_auth_probe pages "
                "expose SlidesLive embeds, run acquisition.slideslive_slide_download before "
                "visual OCR/indexing, write storage/download-corpus/slideslive_slide_recovery.json, "
                "then run acquisition.slide_reference_detection and acquisition.slide_cleanup "
                "to mark duplicate and generic/banner slides with skip_ocr without deleting raw files. "
                "Audio-only recovery, paper PDFs, and poster thumbnails do not satisfy SlidesLive slide evidence."
            ),
            "corpus_token_indexing_policy": (
                "Production must include PG-CORPUS-TOKEN-INDEXING before query planning. "
                "Stage 6 receipts must record token_budget_policy_digest, "
                "embedding_policy_digest, EMBEDDING_MAX_LENGTH, "
                "CORPUS_VECTOR_EMBED_BATCH_SIZE, embedding_sidecar_manifest_path, "
                "embedding_sidecar_manifest_sha256, chroma_insert_batch_size, "
                "vector_embeddings_written: true, and embedded chunk counts matching "
                "chunk sidecars. Embedding sidecars are written before bulk Chroma "
                "load; batch-size tuning is receipt evidence, not embedding identity. "
                "Word-count-only sidecars, readiness-only probes, partial Chroma rows, "
                "conference-local embedding reimplementations, and historical "
                "exhaustive extraction output without registry receipt are rejected."
            ),
            "workshop_talk_segmentation_policy": (
                "When workshops are in scope, production must first prove workshop "
                "talk-media acquisition with acquisition.workshop_talk_media plus "
                "acquisition.embedded_media_recovery "
                "(src/conference_pipeline/acquisition/embedded_media.py): use "
                "public-first/authenticated retry, embedded-player extraction, "
                "embedded_media_links, sanitize_media_url, yt-dlp, and a "
                "conference-local embedded host allowlist before accepting "
                "inaccessible-media blockers; for ICLR production use "
                "--embedded-media-host slideslive.com and reject YouTube, Vimeo, "
                "Zoom, social/profile, and unrelated embeds unless a future "
                "phase explicitly widens the allowlist; record explicit "
                "per-workshop blockers for remaining no-media cases, then write "
                "storage/workshops/talk_media_acquisition_receipt.json with "
                "workshop_id, workshop_title, talk_id, talk_title, speaker, media_url, "
                "local_path, sha256, and byte_count for acquired talk media, or a formal "
                "storage/workshops/workshop_talk_media_scope_waiver.json with decision, "
                "rationale, approver, and downstream impact. A zero-count workshop "
                "speech audit is rejected evidence for deciding workshop talks are "
                "out of scope. Workshop transcripts must then run "
                "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY and "
                "transcription_ocr.workshop_talk_boundary_discovery with Gemini 2.5 Pro "
                "(gemini-2.5-pro) when no vetted talks.json exists; it writes *_talks.json, "
                "5-minute talk-context batches, model_governance_task_id, "
                "provider_receipt_paths, and cost_task_id=workshop_talk_boundary_discovery. "
                "Then run transcription_ocr.workshop_talk_segmentation "
                "or block with talk_segmentation_provider_not_configured; "
                "audio-only recovery is rejected evidence for talk-aware transcripts. "
                "Indexed transcript chunks preserve workshop_id, workshop_title, "
                "talk_id, talk_title, and talk_speaker metadata."
            ),
            "rejected_draft_generation_owner_commands": REJECTED_PRESENTATION_DRAFT_OWNER_COMMANDS,
            "deprecated_owner_commands": [
                "scripts/presentations/extract_topic_figures.py",
                "scripts/presentations/add_images_to_md.py",
            ],
        },
        "stage_script_registry": stage_script_registry_snapshot(),
        "workspace_gitignore_policy": workspace_gitignore_policy(slug),
        "backup_posture": {
            "manifest_source": f"conferences/{slug}/storage/storage_manifest.json",
            "expected_profile": "backup",
            "target_drive_leaf": backup_target_leaf(brief),
            "runtime_requirement": (
                f"explicit --remote-path {backup_remote_path_placeholder(brief)}"
            ),
            "drive_setup_command": (
                "scripts/sync_to_gdrive.py --profile backup "
                f"--remote-path '{backup_remote_path_placeholder(brief)}' "
                "--init-only --receipt-path <receipt>"
            ),
            "continuous_offload": "run backup dry-run and write sync immediately after each artifact-producing stage before downstream consumption",
            "bloat_control": "manifest sync_profiles bound what moves to Drive; rsync incremental updates changed files; cache families stay out unless explicitly enabled",
            "restore_verification": "restore evidence required before cleanup",
            "no_secret_rule": True,
        },
        "provider_model_posture": {
            task_id: {
                "provider": policy["provider"],
                "model": policy["model"],
                "fallback_provider": policy["fallback_provider"],
                "fallback_model": policy["fallback_model"],
                "selection_status": policy["selection_status"],
                "stage": policy["stage"],
            }
            for task_id, policy in policies.items()
        },
        "task_model_defaults": task_model_default_rows(policies),
        "benchmark_requirements": {
            task_id: {
                "current_research_required": policy["current_research"].get("required", True)
                if isinstance(policy["current_research"], dict)
                else True,
                "benchmark_required": policy["benchmark"].get("required", True)
                if isinstance(policy["benchmark"], dict)
                else True,
                "receipt_required": policy["benchmark"].get("receipt_required", True)
                if isinstance(policy["benchmark"], dict)
                else True,
                "placement": policy["benchmark"].get("placement", "stage-local realistic inputs")
                if isinstance(policy["benchmark"], dict)
                else "stage-local realistic inputs",
            }
            for task_id, policy in policies.items()
        },
        "local_provider_deferrals": {
            task_id: {
                "status": f"deferred: {policy['local_runtime_policy']}",
                "evidence": "Local provider remains deferred until the follow-on local-runtime milestone records receipts.",
            }
            for task_id, policy in policies.items()
            if local_proof_posture.get(task_id) == "deferred"
        },
        "presentation_figure_policy": {
            "task_id": "presentation.svg_figure_generation",
            "required_renderers": ["matplotlib", "plotly", "svgwrite"],
            "required_live_owner_command": PRESENTATION_PROGRAMMATIC_FIGURE_LIVE_OWNER_COMMAND,
            "semantic_review_task_id": "presentation.figure_semantic_review",
            "semantic_review_owner_command": PRESENTATION_FIGURE_SEMANTIC_REVIEW_OWNER_COMMAND,
            "completion_owner_policy": "rejects generic placeholder/scaffold SVGs; rejects deprecated paper-figure extraction/insertion scripts; owner must be content-aware editorial programmatic SVG generation",
            "visual_quality_profile": "slide-specific editorial SVGs with evidence-bound template selection; generic boilerplate diagrams are blockers",
            "required_quality_gates": [
                "content-aware template selection",
                "slide evidence binding",
                "rendered visual inspection",
                "LLM-feedback repair regeneration",
                "repair and re-render loop",
                "generic scaffold rejection",
            ],
            "waiver_policy": "explicit per-slide waiver required when no generated SVG is produced",
        },
        "presentation_figure_validation_policy": {
            "task_id": "presentation.figure_validation",
            "required_owner_command": PRESENTATION_FIGURE_VALIDATION_OWNER_COMMAND,
            "required_report_path": "decks/<id>/assets/generated/validation_report.md",
            "completion_policy": "blocks before render/QC when generated SVGs are missing, invalid, blank, or not viewBox-safe",
        },
        "presentation_density_policy": {
            "source": "NeurIPS 2025 backup deck density baseline",
            "target_hyperlinks_per_slide": 2.0,
            "target_svgs_per_slide": 0.70,
            "manifest_fields": [
                "target_hyperlinks_per_slide",
                "target_svgs_per_slide",
            ],
            "validation_check_ids": [
                "hyperlink_density_meets_target",
                "source_binding_density_meets_target",
                "svg_density_meets_target",
            ],
            "figure_stage_hyperlink_policy": "source_binding_report.json must preserve or increase approved reference density before render_spec.json is produced",
        },
        "presentation_executive_overview_policy": {
            "task_id": "presentation.executive_overview_decks",
            "required_owner_command": PRESENTATION_EXECUTIVE_OVERVIEW_DECKS_OWNER_COMMAND,
            "required_decks": ["decks/executive_summary_short", "decks/executive_summary_long"],
            "source_reports": ["analysis/executive_summary.md", "analysis/trend_report.md"],
            "completion_policy": "overview/executive-summary deliverables are incomplete until both decks are manifest-registered, rendered, exported to PDF, QC-reviewed, and shaped as an 8-point ~15 minute short deck plus the same 8-point spine with 3 finer-grained expansions for an ~30 minute long deck",
        },
        "presentation_research_enrichment_policy": {
            "task_id": "presentation.source_binding",
            "required_owner_command": PRESENTATION_RESEARCH_ENRICHMENT_OWNER_COMMAND,
            "required_command_terms": [
                "scripts/presentations/build_deck_source_pack.py",
                "scripts/presentations/bind_presentation_sources.py",
                "--deck-contract",
                "decks/<id>/presentation_deck.json",
                "--source-pack",
                "decks/<id>/source_pack_receipt.json",
                "--output",
                "decks/<id>/source_binding_report.json",
                "--matched-papers",
            ],
            "blocked_placeholders": ["[paper](paper_url)", "(paper_url)"],
        },
        "podcast_notebooklm_policy": {
            "task_id": "podcast.notebooklm_audio_overview",
            "scope_flag": "podcast_notebooklm (brief target output; per-deck Google NotebookLM Audio Overviews)",
            "approved_owner_refs": ["scripts/generate_notebooklm_podcast.py", "conference_pipeline.podcast_notebooklm"],
            "required_owner_command": ".venv/bin/python scripts/generate_notebooklm_podcast.py --workspace <workspace> --live --ack-live-notebooklm --format deep-dive --length long --summary",
            "required_command_terms": [
                "scripts/generate_notebooklm_podcast.py",
                "--workspace",
                "--live",
                "--ack-live-notebooklm",
                "--format deep-dive",
                "--length long",
                "audio/<id>/notebooklm_podcast",
            ],
            "required_outputs": [
                "audio/<id>/notebooklm_podcast/podcast.m4a",
                "audio/<id>/notebooklm_podcast/notebook_manifest.json",
                "audio/<id>/notebooklm_podcast/prompt.txt",
                "audio/<id>/notebooklm_podcast/show_notes.md",
            ],
            "receipt": "stage-10-notebooklm-podcast.json",
            "delivery_owner": "scripts/deliver_notebooklm_podcasts.py",
            "auth_policy": "notebooklm login --browser-cookies chrome on a Google account with NotebookLM Pro; storage_state.json is a secret and never enters artifacts or receipts",
            "completion_policy": "one live Audio Overview per topic and executive deck, sourced from the deck's analysis spine plus blueprint-cited paper texts and talk transcripts and bound by notebook_manifest.json; dry-run, fake-engine, and --deck-ids canary manifests are rejected completion evidence; quota halts (notebooklm_quota_exhausted) are resumable",
            "usage_doc_ref": "docs/reference/stage-owner-audit.md#podcast-notebooklm-audio-overview",
        },
        "presentation_slide_consolidation_policy": {
            "task_id": "presentation.render_spec_generation",
            "required_owner_command": PRESENTATION_SLIDE_CONSOLIDATION_OWNER_COMMAND,
            "required_command_terms": [
                "scripts/presentations/generate_render_spec.py",
                "--presentation-deck",
                "decks/<id>/presentation_deck.json",
                "--source-binding-report",
                "decks/<id>/source_binding_report.json",
                "--visual-plan",
                "decks/<id>/visual_plan.json",
                "--output",
                "decks/<id>/render_spec.json",
            ],
            "stage_order": "after source binding and visual plan generation, before render/export",
            "receipt_policy": "missing or stale render_spec receipts block render/export and delivery generation",
        },
        "presentation_brand_policy": {
            "model_governance_task_id": "presentation.render_export",
            "required_owner_command": PRESENTATION_REVEALJS_RENDER_OWNER_COMMAND,
            "required_title_markers": ["render_spec.json", "title slide"],
            "required_brand_reference": "assets/brand/relationalai/",
            "template_resolution_order": [
                "assets/brand/relationalai/",
                "RAI_PRESENTATION_TEMPLATE with matching NeurIPS brand assets",
                "approved local template path with matching NeurIPS brand assets",
            ],
            "fail_closed_on": [
                "missing title slide in render_spec.json",
                "missing RelationalAI NeurIPS 2025 brand assets",
                "generic or approximate brand theme",
                "unbranded render",
            ],
        },
        "presentation_slide_sync_policy": {
            "presenter_script_task_id": "presentation.delivery_generation",
            "presenter_narration_json_task_id": "presentation.build_presenter_video",
            "presenter_narration_review_task_id": "presentation.presenter_narration_review",
            "presenter_narration_fix_task_id": "presentation.presenter_narration_fix",
            "narration_review_task_id": "presentation.narration_script_review",
            "narration_series_review_task_id": "presentation.narration_series_review",
            "narration_series_fix_task_id": "presentation.narration_series_fix",
            "narration_fix_task_id": "presentation.narration_script_fix",
            "video_generation_task_id": "presentation.video_generation",
            "presentation_review_task_id": "presentation.presentation_review",
            "editorial_gate_task_id": "presentation.editorial_gate",
            "editorial_gate_owner_command": PRESENTATION_EDITORIAL_GATE_OWNER_COMMAND,
            "editorial_storyboard_output": "decks/<id>/editorial_storyboard.json",
            "editorial_gate_report": "decks/<id>/presentation_editorial_gate_report.json",
            "duration_target_seconds": 2280,
            "duration_hard_limit_seconds": 2400,
            "compression_policy": "moderate merge of duplicate paper clusters; retain materially distinct papers",
            "zoom_policy": "integrate visual references into parent slide narration; explicit zoom/detail beats only for complex or central figures",
            "presenter_owner_command": PRESENTATION_PRESENTER_SCRIPT_OWNER_COMMAND,
            "presenter_narration_json_owner_command": PRESENTATION_PRESENTER_NARRATION_JSON_OWNER_COMMAND,
            "presenter_narration_review_owner_command": PRESENTATION_PRESENTER_NARRATION_REVIEW_OWNER_COMMAND,
            "presenter_narration_series_review_owner_command": PRESENTATION_PRESENTER_NARRATION_SERIES_REVIEW_OWNER_COMMAND,
            "presenter_narration_fix_owner_command": PRESENTATION_PRESENTER_NARRATION_FIX_OWNER_COMMAND,
            "narration_review_owner_command": PRESENTATION_NARRATION_REVIEW_OWNER_COMMAND,
            "narration_series_review_owner_command": PRESENTATION_NARRATION_SERIES_REVIEW_OWNER_COMMAND,
            "narration_series_fix_owner_command": PRESENTATION_NARRATION_SERIES_FIX_OWNER_COMMAND,
            "narration_fix_owner_command": PRESENTATION_NARRATION_FIX_OWNER_COMMAND,
            "video_owner_command": PRESENTATION_VIDEO_GENERATION_OWNER_COMMAND,
            "presentation_review_owner_command": PRESENTATION_PRESENTATION_REVIEW_OWNER_COMMAND,
            "required_mapping": "decks/<id>/presentation_slide_mapping.json",
            "required_delivery_contracts": [
                "decks/<id>/delivery_single_speaker.json",
                "decks/<id>/delivery_notebooklm.json",
                "decks/<id>/video_build/script.json",
                "decks/<id>/presenter_narration_lint_report.json",
                "decks/<id>/presenter_narration_review.md",
                "decks/<id>/presenter_narration_review.json",
                "decks/<id>/narration_lint_report.json",
                "decks/<id>/narration_review.md",
                "decks/<id>/narration_review.json",
                "decks/presenter_narration_series_review.md",
                "decks/presenter_narration_series_review.json",
                "decks/narration_series_review.md",
                "decks/narration_series_review.json",
                "decks/<id>/video_sync_validation_single_speaker.json",
                "decks/<id>/video_sync_validation_notebooklm.json",
            ],
            "fail_closed_on": [
                "missing mapped Slide 0 narration",
                "missing mapped slide narration",
                "stale delivery contract generated before render_spec.json",
                "missing presenter narration review",
                "presenter_narration_review.json approve_for_tts is not true",
                "missing presenter narration series review",
                "missing narration review",
                "narration_review.json approve_for_tts is not true",
                "missing narration series review",
                "narration_series_review.json status is failed without explicit waiver",
                "fixed delivery contract and script.json are out of sync",
            ],
        },
        "presentation_pdf_export_policy": {
            "model_governance_task_id": "presentation.render_export",
            "required_owner_command": PRESENTATION_PDF_EXPORT_OWNER_COMMAND,
            "required_input": "decks/<id>/presentation.html",
            "required_output": "decks/<id>/presentation.pdf",
            "fail_closed_on": [
                "missing Reveal.js HTML",
                "missing presentation.pdf",
                "manual browser print-to-PDF without registry-approved waiver",
                "ad hoc export command outside scripts/presentations/export_revealjs_pdf.py",
            ],
        },
        "presentation_link_audit_policy": {
            "model_governance_task_id": "presentation.link_audit",
            "required_owner_command": PRESENTATION_LINK_AUDIT_OWNER_COMMAND,
            "required_receipt": "decks/presentation_link_audit_receipt.json",
            "fail_closed_on": [
                "file:/// links in HTML or PDF",
                "/Users/ or repo-local links in HTML or PDF",
                "../analysis links emitted when Drive source map is missing",
                "missing or failed presentation_link_audit_receipt.json",
            ],
        },
        "presentation_visual_qc_policy": {
            "model_governance_task_id": "presentation.visual_qc",
            "default_model": "openai:gpt-5.6-sol",
            "fallback_model": "openai:gpt-5.6-terra",
            "required_owner_commands": [
                "scripts/presentations/inspect_rendered_slides.py",
                "scripts/presentations/inspect_rendered_slides_llm.py",
            ],
            "required_report_path": "decks/<id>/assets/generated/visual_llm_qc_report.json",
            "source_screenshots": "decks/<id>/assets/generated/qc/screenshots",
            "fail_closed_on": [
                "missing provider/model",
                "missing visual_llm_qc_report.json",
                "unparseable JSON",
                "blocker findings",
            ],
        },
        "presentation_repair_policy": {
            "model_governance_task_id": "presentation.repair",
            "required_owner_command": PRESENTATION_REPAIR_OWNER_COMMAND,
            "required_inputs": [
                "decks/<id>/assets/generated/visual_llm_qc_report.json",
                "decks/<id>/presentation_slide_mapping.json",
                "decks/<id>/asset_inventory.json",
                "decks/<id>/visual_plan.json",
                "decks/<id>/render_spec.json",
            ],
            "required_outputs": [
                "decks/<id>/assets/generated/visual_repair_feedback.json",
                "decks/<id>/assets/generated/repair_loop_receipt.json",
                "decks/<id>/presentation.html",
                "decks/<id>/assets/generated/visual_qc_report.md",
            ],
            "must_rerun_after_repair": [
                "scripts/presentations/generate_programmatic_figures.py --repair-feedback",
                "scripts/presentations/md_to_revealjs.py",
                "scripts/presentations/inspect_rendered_slides.py",
                "scripts/presentations/inspect_rendered_slides_llm.py",
            ],
            "mechanical_svg_fix_only": "insufficient",
        },
        "narrative_analysis_contract": {
            "public_report_shape": [
                "Executive Summary",
                "Key Findings",
                "Detailed Analysis",
                "References And Sources",
                "Cross-References",
                "Recommendations",
                "Questions For Further Investigation",
                "Slide Seeds",
            ],
            "slide_brief_sidecars": ["analysis/slide_briefs/<topic_id>.md"],
            "analysis_profile": "deep-narrative-v1",
            "model_governance_task_id": "analysis.deep_narrative_synthesis",
            "default_model": "openai:gpt-5.6-sol",
            "fallback_model": "openai:gpt-5.6-terra",
            "deep_validation_required": True,
            "production_fallback_policy": "block or degraded receipt; deterministic fallback is tests/mock/offline only",
            "provenance_sidecars": [
                "analysis/provenance/<topic_id>.md",
                "analysis/provenance/<topic_id>.json",
                "analysis/analysis_citation_manifest.json",
            ],
            "forbidden_public_sections": [
                "Source Basis",
                "Verbatim transcript/OCR evidence",
                "Matched Paper Evidence",
                "Source Gaps",
            ],
        },
        "presentation_content_gate_policy": {
            "gate": "presentation narrative-content gate",
            "requires_analysis_digest_binding": True,
            "rejects_stale_scaffold_language": True,
            "rejects_public_ledger_language": True,
            "rejects_identical_deck_checkpoint_files_without_waiver": True,
        },
        "local_proof_posture": local_proof_posture,
        "provider_contracts": benchmark_aware_provider_contracts(),
        "model_governance_task_ids": MODEL_REGISTRY_TASK_IDS,
        "workshop_coverage_posture": workshop_coverage_posture_block(brief),
        "workshop_inventory_contract": workshop_inventory_contract_block(brief, slug),
        "structured_gates": structured_production_gates(),
        "openreview_auth_posture": {
            "required_flag": "--openreview-auth required",
            "auth_env_names": ["OPENREVIEW_USERNAME", "OPENREVIEW_PASSWORD"],
            "unauthenticated_is_fail_stop": True,
            "rejected_modes_on_required_corpus": ["unauthenticated"],
        },
        "corpus_source_metadata_gate": corpus_source_metadata_gate(slug),
    }


def _slug_for_workshop(venueid: str) -> str:
    keep = []
    for ch in venueid.replace("/", "_"):
        if ch.isalnum() or ch in "-_":
            keep.append(ch.lower())
        else:
            keep.append("-")
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:80] or "workshop"


WORKSHOP_SITE_DOWNLOADER_TEMPLATE = """#!/usr/bin/env python3
\"\"\"External-site workshop paper downloader stub for {venueid}.

This file is a scaffold emitted by `plan-conference-analysis`. It exists
because the workshop is classified as `{pdf_availability_class}` on
OpenReview, so paper PDFs must be acquired from the workshop's own site
({external_site_url}). Replace the body with a concrete downloader that:

  - Reads a workshop-site config (paper URLs, accept-list, schedule).
  - Downloads each PDF politely (rate-limited; honour robots.txt).
  - Writes per-paper receipts matching the OpenReview downloader contract:
    {{paper_id, venueid, title, authors[], pdf_path, sha256, byte_count,
      content_type, attempted_at, status: succeeded|unreachable,
      failure_reason?, source_url}}.
  - Writes a stage receipt that the production audit can validate.
\"\"\"

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VENUEID = "{venueid}"
WORKSHOP_SLUG = "{workshop_slug}"
EXTERNAL_SITE_URL = "{external_site_url}"
PDF_AVAILABILITY_CLASS = "{pdf_availability_class}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"External-site paper downloader scaffold for {{VENUEID}}."
    )
    parser.add_argument("--workshop-site-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt-path", required=True, type=Path)
    parser.add_argument("--rate-limit-rps", type=float, default=1.0)
    parser.add_argument("--workspace", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raise NotImplementedError(
        f"External-site downloader for {{VENUEID}} is a stub. Workshop site: "
        f"{{EXTERNAL_SITE_URL or '<unknown>'}}. PDF availability class: "
        f"{{PDF_AVAILABILITY_CLASS}}. Fill in this scaffold with a concrete "
        f"workshop-site downloader, write per-paper receipts, and rerun "
        f"production audit."
    )


if __name__ == "__main__":
    sys.exit(main())
"""


def write_workshop_site_scaffolds(
    brief: dict[str, object], slug: str, draft_dir: Path
) -> list[dict[str, object]]:
    """Write external-site downloader stubs for workshops that need them.

    Returns the tracker entries (also persisted as
    ``conferences/<slug>/storage/workshops/workshop_site_tracker.json`` in the
    draft) so callers can audit later.
    """
    payload = _brief_workshop_payload(brief)
    workshops_input = payload.get("workshops") or []
    if not isinstance(workshops_input, list):
        workshops_input = []
    tracker_entries: list[dict[str, object]] = []
    scaffold_dir_rel = Path("scripts") / "conference" / slug / "workshop_site_downloaders"
    for row in workshops_input:
        if not isinstance(row, dict):
            continue
        decision = str(row.get("decision") or "")
        pdf_class = str(row.get("pdf_availability_class") or "")
        if decision != "external_site_required":
            continue
        venueid = str(row.get("venueid") or "")
        if not venueid:
            continue
        workshop_slug = _slug_for_workshop(venueid)
        scaffold_relative = scaffold_dir_rel / f"{workshop_slug}.py"
        body = WORKSHOP_SITE_DOWNLOADER_TEMPLATE.format(
            venueid=venueid,
            workshop_slug=workshop_slug,
            external_site_url=str(row.get("external_site_url") or ""),
            pdf_availability_class=pdf_class or "openreview_metadata_only",
        )
        write_file(draft_dir, scaffold_relative.as_posix(), body)
        tracker_entries.append(
            {
                "venueid": venueid,
                "workshop_slug": workshop_slug,
                "scaffold_path": scaffold_relative.as_posix(),
                "pdf_availability_class": pdf_class,
                "decision": decision,
                "external_site_url": row.get("external_site_url") or None,
                "completion_status": "stub",
            }
        )
    tracker = {
        "schema_version": 1,
        "tracker_type": "conference_pipeline.workshop_site_tracker",
        "workshop_count": len(tracker_entries),
        "stubs": tracker_entries,
        "completion_rule": (
            "production audit rejects any entry with completion_status=stub for"
            " in-scope workshops; replace the scaffold with a concrete downloader"
            " and update completion_status to `complete` once per-paper receipts"
            " are written."
        ),
    }
    write_file(
        draft_dir,
        f"conferences/{slug}/storage/workshops/workshop_site_tracker.json",
        json.dumps(tracker, indent=2, sort_keys=True) + "\n",
    )
    return tracker_entries


def write_draft(brief: dict[str, object], draft_root: Path, *, update_existing: bool, new_version: bool, autonomous: bool) -> Path:
    slug = conference_slug(brief)
    draft_dir = draft_directory(brief, draft_root, new_version=new_version)
    if draft_dir.exists() and any(draft_dir.iterdir()) and not update_existing:
        raise FileExistsError(f"target already exists: {draft_dir}")
    draft_dir.mkdir(parents=True, exist_ok=True)
    changed = generated_relative_paths(brief, slug)
    write_file(draft_dir, changed[0], render_context(brief, slug))
    write_file(draft_dir, ".planning/REQUIREMENTS.md", render_requirements(milestone_identity(brief)))
    write_file(draft_dir, ".planning/ROADMAP.md", render_roadmap(brief))
    write_file(draft_dir, ".planning/STATE.md", render_state(brief))
    write_file(draft_dir, f"conferences/{slug}/README.md", render_readme(brief, slug))
    registry_policies = _registry_policy_snapshots()
    approval = {
        "generated milestone identity": milestone_identity(brief),
        "support claims": SUPPORT_LABELS,
        "roadmap phases": ROADMAP_AREAS,
        "official workflow mapping": workflow_mapping_for_brief(brief),
        "task model defaults": task_model_default_rows(registry_policies),
        "benchmark-aware provider contracts": benchmark_aware_provider_contracts(),
        "dual-store paper matching contract": {
            "vector_store_stores": ["papers", "artifacts"],
            "keypoint_provider": "openai",
            "keypoint_model": "gpt-5.6-sol",
            "keypoint_source": "artifact_gpt_researcher_findings",
            "paper_research_store": "papers",
            "artifact_research_store": "artifacts",
            "canonical_output": "matched_papers.json",
            "legacy_script": "scripts/rank_papers.py deprecated for NeurIPS 2025 compatibility only",
        },
        "fail-stop pipeline policy": {
            "blocking_statuses": [
                "blocked",
                "deferred_blocked",
                "failed",
                "unavailable",
                "unsupported",
            ],
            "blocker_classes": [
                "ocr_deferred_blocked",
                "stage_blocked",
                "stage_unavailable",
                "local_llm_not_ready",
                "missing_receipt",
                "threshold_violation",
                "stale_receipt",
            ],
            "downstream_policy": "provider availability, live-owner availability, missing or stale receipts, and failed required validation halt downstream podcast, deck, release, proof, and publication work",
            "resume_policy": "active blockers require UnblockReceipt via workflow unblock or conference_pipeline.workflow_unblock",
            "summary_fields": [
                "halt_status",
                "halt_stage_id",
                "blocker_receipt_path",
                "unblock_acknowledgements",
            ],
        },
        "backup posture": {
            "manifest_source": f"conferences/{slug}/storage/storage_manifest.json",
            "expected_profile": "backup",
            "target_drive_leaf": backup_target_leaf(brief),
            "runtime_requirement": (
                f"explicit --remote-path {backup_remote_path_placeholder(brief)}"
            ),
            "drive_setup_command": (
                "scripts/sync_to_gdrive.py --profile backup "
                f"--remote-path '{backup_remote_path_placeholder(brief)}' "
                "--init-only --receipt-path <receipt>"
            ),
            "continuous_offload": "run backup dry-run and write sync immediately after each artifact-producing stage before downstream consumption",
            "bloat_control": "manifest sync_profiles bound what moves to Drive; rsync incremental updates changed files; cache families stay out unless explicitly enabled",
            "restore_verification": "backup receipts before verification; restore evidence before cleanup",
            "no_secret_rule": True,
        },
        "gsd lifecycle and orchestrator conformance": {
            "hard_rule": (
                "no live execution of Phase N without phases/NN-*/ PLAN + "
                "plan-review/verification artifacts"
            ),
            "orchestrator_stage_sequence": list(ROADMAP_AREAS),
            "orchestrator_declaration_rule": (
                "any supervisor/orchestration wrapper must be plan-declared and "
                "execute the machine-readable orchestrator_stage_sequence; it is "
                "forbidden from provider fallbacks outside the approved contract rows"
            ),
            "proof_stage_rejection_rule": (
                "the proof stage must reject receipts produced by an undeclared "
                "orchestrator"
            ),
        },
        "workspace gitignore policy": workspace_gitignore_policy(slug),
        "stage script registry": stage_script_registry_snapshot(),
        "provider cost approval posture": safe_text(brief["provider_cost_approval_posture"]),
        "event-family inventory": EVENT_FAMILY_INVENTORY,
        "interest-group decision gate": "explicit approve/edit/reject/complete before provider-backed work",
        "exclusion record fields": ["scope", "reason", "decision context", "source note", "downstream effect"],
        "production handoff fields": PRODUCTION_HANDOFF_FIELDS,
        "next step": production_next_step(slug),
        "production handoff": production_handoff(brief, slug),
        "model governance task ids": MODEL_REGISTRY_TASK_IDS,
        "workshop coverage posture": workshop_coverage_posture_block(brief),
        "workshop inventory contract": workshop_inventory_contract_block(brief, slug),
        "changed files": changed,
        "autonomous": autonomous,
    }
    write_file(draft_dir, "approval-summary.json", json.dumps(approval, indent=2, sort_keys=True) + "\n")
    write_workshop_site_scaffolds(brief, slug, draft_dir)
    return draft_dir


def finalize_draft(brief: dict[str, object], draft_root: Path, repo_root: Path, *, update_existing: bool, new_version: bool) -> list[str]:
    slug = conference_slug(brief)
    draft_dir = draft_directory(brief, draft_root, new_version=new_version)
    if not draft_dir.exists():
        raise FileNotFoundError(f"draft not found: {draft_dir}")
    repo_root = repo_root.resolve()
    copied: list[str] = []
    for relative_name in generated_relative_paths(brief, slug):
        relative = Path(relative_name)
        source = ensure_inside(draft_dir / relative, draft_dir)
        if not source.is_file():
            raise FileNotFoundError(f"generated draft file missing: {source}")
        target = ensure_inside(repo_root / relative, repo_root)
        if target.exists() and not (update_existing or new_version):
            raise FileExistsError(f"target already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative.as_posix())
    return sorted(copied)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft conference-analysis planning artifacts.")
    parser.add_argument("--brief", required=True)
    parser.add_argument("--draft-root", required=True)
    parser.add_argument("--autonomous", action="store_true")
    parser.add_argument("--update-existing", action="store_true")
    parser.add_argument("--new-version", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--approved-finalize", action="store_true")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    brief = load_brief(Path(args.brief))
    missing = validate_brief(brief)
    if missing:
        print("missing required brief fields: " + ", ".join(missing), file=sys.stderr)
        return 2

    draft_root = Path(args.draft_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    try:
        if args.finalize:
            if not args.approved_finalize:
                print("--approved-finalize is required for final writes", file=sys.stderr)
                return 2
            copied = finalize_draft(
                brief,
                draft_root,
                repo_root,
                update_existing=args.update_existing,
                new_version=args.new_version,
            )
            print("finalized planning set")
            for item in copied:
                print(item)
            return 0
        draft_dir = write_draft(
            brief,
            draft_root,
            update_existing=args.update_existing,
            new_version=args.new_version,
            autonomous=args.autonomous,
        )
        print(f"draft planning set: {draft_dir}")
        return 0
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
