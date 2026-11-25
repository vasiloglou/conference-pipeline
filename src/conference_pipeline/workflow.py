"""Canonical workflow registry and workspace-aware stage helpers."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import yaml

from conference_pipeline.config import PipelineSettings, load_settings
from conference_pipeline.model_governance import default_model

DispatchKind = Literal["python_stage", "subprocess_stage"]
CommandRunner = Callable[[Sequence[str], Path], None]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = Path("conferences/neurips2025")
SUBPROCESS_PYTHON = sys.executable
RUN_ALL_PRESENTATION_OWNER_COMMANDS_BEFORE_FAILING = frozenset({"visual_qc"})


@dataclass(frozen=True)
class WorkflowStage:
    """Static metadata for a workflow stage."""

    id: str
    order: int
    description: str
    dispatch_kind: DispatchKind
    implementation_ref: str
    config_key: str
    supports_topic_filter: bool
    requires_workspace_files: tuple[str, ...]


@dataclass(frozen=True)
class PodcastWorkflowConfig:
    """Operator-facing defaults for the podcast stage."""

    target_minutes: int = 30
    max_tier: int = 3
    top_n_facts: int = 100
    host_a: str = "Alex"
    host_b: str = "Sam"
    model: str = default_model("podcast.focused_script_generation")


@dataclass(frozen=True)
class DeckWorkflowConfig:
    """Operator-facing defaults for the deck stage."""

    target_slides: int = 60
    target_hyperlinks_per_slide: float = 2.0
    target_svgs_per_slide: float = 0.70
    reflow_model: str = "gemini"
    max_papers: int = 2
    required_env: tuple[str, ...] = (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
    )


@dataclass(frozen=True)
class WorkflowConfig:
    """Workflow config loaded from the conference workspace."""

    podcast: PodcastWorkflowConfig
    deck: DeckWorkflowConfig


@dataclass(frozen=True)
class WorkflowWorkspace:
    """Validated conference workspace with workflow defaults."""

    root: Path
    config_path: Path
    manifest_path: Path
    config_data: dict[str, Any]
    workflow: WorkflowConfig


@dataclass(frozen=True)
class PresentationSelectedScope:
    """Normalized presentation target recorded on presentation-stage receipts."""

    presentation_id: str
    presentation_type: str
    topic: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "presentation_id": self.presentation_id,
            "presentation_type": self.presentation_type,
            "topic": self.topic,
        }


@dataclass(frozen=True)
class PresentationStageSelection:
    """Manifest-backed selection for a first-class presentation workflow stage."""

    loaded_manifest: Any
    stage_spec: Any
    presentation: Any
    selected_scope: PresentationSelectedScope


LEGACY_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        id="discover",
        order=1,
        description="Discover presentations from the conference source.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.pipeline.NeuripsPipeline.discover_presentations",
        config_key="acquisition",
        supports_topic_filter=False,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json"),
    ),
    WorkflowStage(
        id="download",
        order=2,
        description="Download presentation assets for discovered items.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.pipeline.NeuripsPipeline.download_assets_for_presentation",
        config_key="acquisition",
        supports_topic_filter=False,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json"),
    ),
    WorkflowStage(
        id="transcribe",
        order=3,
        description="Transcribe downloaded presentation audio.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.pipeline.NeuripsPipeline.transcribe_presentation",
        config_key="acquisition",
        supports_topic_filter=False,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json"),
    ),
    WorkflowStage(
        id="analyze",
        order=4,
        description="Run the composable analysis chain and emit analysis_manifest.json.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.analysis.run_analyze_stage@v1",
        config_key="analysis",
        supports_topic_filter=False,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json"),
    ),
    WorkflowStage(
        id="podcast",
        order=5,
        description="Generate focused podcasts from topic analysis docs.",
        dispatch_kind="subprocess_stage",
        implementation_ref="scripts/generate_focused_podcast.py",
        config_key="podcast",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json"),
    ),
    WorkflowStage(
        id="deck",
        order=6,
        description="Generate deck artifacts from topic analysis docs.",
        dispatch_kind="subprocess_stage",
        implementation_ref=(
            "scripts/presentations/generate_slides_agent.py -> "
            "scripts/presentations/enrich_slides_with_papers.py -> "
            "scripts/presentations/reflow_slides.py -> "
            "scripts/presentations/sanity_check_slides.py -> "
            "scripts/presentations/enrich_hyperlinks_hybrid.py -> "
            "scripts/presentations/generate_presenter_script.py"
        ),
        config_key="deck",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json"),
    ),
)

PRESENTATION_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        id="presentation_blueprint",
        order=7,
        description="Plan or run presentation blueprint generation and approval review from manifest scope.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.presentation_blueprint",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="deck_contract_generation",
        order=8,
        description="Plan or run structured deck contract generation from approved blueprint evidence.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.deck_contract_generation",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="source_binding",
        order=9,
        description="Plan or run structured source binding from deck/source contracts.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.source_binding",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="visual_plan_generation",
        order=10,
        description="Plan or run visual plan and asset inventory generation from structured inputs.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.visual_plan_generation",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="render_spec_generation",
        order=11,
        description="Plan or run render_spec.json generation from deck, source, and visual contracts.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.render_spec_generation",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="render_export",
        order=12,
        description="Plan or run HTML/PDF export from render_spec.json.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.render_export",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="visual_qc",
        order=13,
        description="Plan or run presentation visual QC from manifest scope.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.visual_qc",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="delivery_generation",
        order=14,
        description="Plan or run structured single-speaker and NotebookLM delivery contract generation.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.delivery_generation",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="presentation_review",
        order=15,
        description="Plan or run whole-presentation review from requirements, narration, and screenshots.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.presentation_review",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="video_generation",
        order=16,
        description="Plan or run presentation video generation from structured delivery contracts.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.video_generation",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="video_sync_validation",
        order=17,
        description="Plan or run video synchronization validation for both delivery variants.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.video_sync_validation",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
    WorkflowStage(
        id="human_review",
        order=18,
        description="Track the human review checkpoint declared by the presentation manifest.",
        dispatch_kind="subprocess_stage",
        implementation_ref="presentation_manifest.stage_specs.human_review",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "decks/presentation_manifest.json",
        ),
    ),
)

VALIDATION_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        id="presentation_blueprint_validation",
        order=19,
        description="Validate the presentation_blueprint checkpoint approval contract.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="deck_contract_generation_validation",
        order=20,
        description="Validate structural readiness of the deck_contract_generation checkpoint.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="source_binding_validation",
        order=21,
        description="Validate structural readiness of the source_binding checkpoint.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="visual_plan_generation_validation",
        order=22,
        description="Validate structural readiness of the visual_plan_generation checkpoint.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="render_spec_generation_validation",
        order=23,
        description="Validate structural readiness of the render_spec_generation checkpoint.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="render_export_validation",
        order=24,
        description="Validate structural readiness of the render_export checkpoint.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="visual_qc_validation",
        order=25,
        description="Validate structural readiness of the visual_qc checkpoint.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="delivery_generation_validation",
        order=26,
        description="Validate structural readiness of the delivery_generation checkpoint.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="video_generation_validation",
        order=27,
        description="Validate structural readiness of the video_generation checkpoint.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="video_sync_validation_receipt",
        order=28,
        description="Validate structural readiness of the video_sync_validation checkpoint.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="human_review_validation",
        order=29,
        description="Validate structural readiness of the human_review checkpoint.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="slide_consolidation_validation",
        order=30,
        description="Legacy/debug validation for old slide mapping Markdown compatibility fixtures.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="research_enrichment_validation",
        order=31,
        description="Legacy/debug validation for old hyperlink-enriched Markdown compatibility fixtures.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="figures_structural_validation",
        order=32,
        description="Legacy/debug validation for old figure Markdown and SVG compatibility fixtures.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="revealjs_render_validation",
        order=33,
        description="Legacy/debug validation for old RevealJS render compatibility fixtures.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="presenter_script_validation",
        order=34,
        description="Legacy/debug validation for presenter-script Markdown compatibility fixtures.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
    WorkflowStage(
        id="pdf_export_validation",
        order=35,
        description="Legacy/debug validation for old PDF export compatibility fixtures.",
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.presentation.validation.run_validation_stage@v1",
        config_key="presentation",
        supports_topic_filter=True,
        requires_workspace_files=("config.yaml", "storage/storage_manifest.json", "decks/presentation_manifest.json"),
    ),
)

EXTRACTION_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        id="exhaustive-topic-extraction",
        order=31,
        description=(
            "Exhaustive topic extraction across transcripts, OCR, local text, "
            "page-image OCR, and selected paper artifacts."
        ),
        dispatch_kind="python_stage",
        implementation_ref=(
            "conference_pipeline.iclr2025_live.run_exhaustive_topic_extraction@v1"
        ),
        config_key="analysis",
        supports_topic_filter=False,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
            "storage/corpus/v15_corpus_inventory.json",
        ),
    ),
)

INDEXING_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        id="enhanced-indexing-vector-store",
        order=32,
        description=(
            "Enhanced indexing + vector store construction over Phase 14A "
            "extraction artifacts."
        ),
        dispatch_kind="python_stage",
        implementation_ref=(
            "conference_pipeline.iclr2025_live.run_enhanced_indexing_vector_store@v1"
        ),
        config_key="analysis",
        supports_topic_filter=False,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "storage/extractions/v15-iclr2025-production/v15_extraction_coverage.json",
            "storage/workflow-runs/v15-iclr2025-production/14A-exhaustive-topic-extraction-receipt.json",
        ),
    ),
)

RESEARCH_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        id="query-planning-gpt-researcher",
        order=33,
        description=(
            "Query planning + GPT Researcher topic execution over the Phase "
            "14B vector store."
        ),
        dispatch_kind="python_stage",
        implementation_ref=(
            "conference_pipeline.iclr2025_live.run_query_planning_gpt_researcher@v1"
        ),
        config_key="analysis",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "storage/extractions/v15-iclr2025-production/v15_extraction_coverage.json",
            "storage/workflow-runs/v15-iclr2025-production/14A-exhaustive-topic-extraction-receipt.json",
            "storage/workflow-runs/v15-iclr2025-production/14B-enhanced-indexing-and-vector-store-receipt.json",
            "storage/vector-store/v15-iclr2025-production/v15_index_coverage.json",
            "analysis/iclr2025_taxonomy.json",
        ),
    ),
)

PAPER_MATCHING_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        id="paper-matching",
        order=34,
        description=(
            "Artifact-grounded paper matching: artifact-store research, "
            "frontier keypoints, then paper-store research."
        ),
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.paper_matching.run_paper_matching_stage@v1",
        config_key="paper_matching",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "storage/vector-store/<run-id>/stores/artifacts/v15_index_digest.json",
            "storage/vector-store/<run-id>/stores/papers/v15_index_digest.json",
        ),
    ),
)

COVERAGE_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        id="coverage-gap-fill",
        order=35,
        description=(
            "Phase 16A coverage analysis + gap fill over Phase 14A/14B/14C "
            "and Phase 14/16 evidence."
        ),
        dispatch_kind="python_stage",
        implementation_ref=(
            "conference_pipeline.iclr2025_live.run_coverage_gap_fill_stage@v1"
        ),
        config_key="analysis",
        supports_topic_filter=False,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "analysis/analysis_manifest.json",
            "analysis/analysis_citation_manifest.json",
            "storage/extractions/v15-iclr2025-production/v15_extraction_coverage.json",
            "storage/vector-store/v15-iclr2025-production/v15_index_coverage.json",
            "storage/research/v15-iclr2025-production/v15_research_coverage.json",
            "analysis/qa/v15_coverage_report.json",
            "storage/workflow-runs/v15-iclr2025-production/14A-exhaustive-topic-extraction-receipt.json",
            "storage/workflow-runs/v15-iclr2025-production/14B-enhanced-indexing-and-vector-store-receipt.json",
            "storage/workflow-runs/v15-iclr2025-production/14C-query-planning-gpt-researcher-receipt.json",
            "storage/workflow-runs/v15-iclr2025-production/stage-16-qa.json",
        ),
    ),
)

SYNTHESIS_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        id="synthesis-trend-report",
        order=36,
        description=(
            "Phase 19A executive summary + trend report synthesis over the "
            "validated final ICLR topic set."
        ),
        dispatch_kind="python_stage",
        implementation_ref=(
            "conference_pipeline.iclr2025_live.run_synthesis_trend_report_stage@v1"
        ),
        config_key="analysis",
        supports_topic_filter=False,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "analysis/analysis_manifest.json",
            "analysis/analysis_citation_manifest.json",
            "analysis/qa/coverage_gap_report.json",
            "analysis/02_model_architectures.md",
            "analysis/03_training_optimization.md",
            "analysis/04_representation_learning.md",
            "analysis/05_agents_reasoning.md",
            "analysis/06_multimodal_systems.md",
            "analysis/07_evaluation_benchmarks.md",
            "analysis/08_efficiency_scaling.md",
            "analysis/09_safety_alignment.md",
            "analysis/10_scientific_applications.md",
            "analysis/11_datasets_tooling.md",
            "storage/workflow-runs/v15-iclr2025-production/16A-coverage-gap-fill-receipt.json",
        ),
    ),
)

NOTEBOOKLM_PODCAST_WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        id="notebooklm-podcast",
        order=37,
        description=(
            "Per-deck Google NotebookLM Audio Overview podcasts over each topic "
            "and executive deck's analysis spine plus blueprint-cited sources."
        ),
        dispatch_kind="python_stage",
        implementation_ref="conference_pipeline.podcast_notebooklm.run_stage@v1",
        config_key="podcast",
        supports_topic_filter=True,
        requires_workspace_files=(
            "config.yaml",
            "storage/storage_manifest.json",
            "analysis/analysis_manifest.json",
        ),
    ),
)

WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    LEGACY_WORKFLOW_STAGES
    + PRESENTATION_WORKFLOW_STAGES
    + VALIDATION_WORKFLOW_STAGES
    + EXTRACTION_WORKFLOW_STAGES
    + INDEXING_WORKFLOW_STAGES
    + RESEARCH_WORKFLOW_STAGES
    + PAPER_MATCHING_WORKFLOW_STAGES
    + COVERAGE_WORKFLOW_STAGES
    + SYNTHESIS_WORKFLOW_STAGES
    + NOTEBOOKLM_PODCAST_WORKFLOW_STAGES
)
WORKFLOW_STAGE_BY_ID = {stage.id: stage for stage in WORKFLOW_STAGES}
PRESENTATION_STAGE_ID_SET = frozenset(stage.id for stage in PRESENTATION_WORKFLOW_STAGES)
VALIDATION_STAGE_ID_SET = frozenset(stage.id for stage in VALIDATION_WORKFLOW_STAGES)
EXTRACTION_STAGE_ID_SET = frozenset(stage.id for stage in EXTRACTION_WORKFLOW_STAGES)
INDEXING_STAGE_ID_SET = frozenset(stage.id for stage in INDEXING_WORKFLOW_STAGES)
RESEARCH_STAGE_ID_SET = frozenset(stage.id for stage in RESEARCH_WORKFLOW_STAGES)
PAPER_MATCHING_STAGE_ID_SET = frozenset(stage.id for stage in PAPER_MATCHING_WORKFLOW_STAGES)
COVERAGE_STAGE_ID_SET = frozenset(stage.id for stage in COVERAGE_WORKFLOW_STAGES)
SYNTHESIS_STAGE_ID_SET = frozenset(stage.id for stage in SYNTHESIS_WORKFLOW_STAGES)

# Phase 14A allow-list: workspaces routed against the exhaustive-topic-extraction
# stage today. Surfaced as a module-level constant so tests can pin the expected
# set and so a future patch can replace the hardcoded set with a config-sourced
# value without touching call sites.
EXTRACTION_SUPPORTED_WORKSPACE_NAMES: frozenset[str] = frozenset(
    {"iclr2025", "neurips2025"}
)


def default_workflow_config() -> WorkflowConfig:
    """Return default workflow config values."""

    return WorkflowConfig(
        podcast=PodcastWorkflowConfig(),
        deck=DeckWorkflowConfig(),
    )


def get_default_workspace_path() -> Path:
    """Resolve the default NeurIPS workspace path from the repo root."""

    return (PROJECT_ROOT / DEFAULT_WORKSPACE).resolve()


def _has_narrative_source_packs(workspace: WorkflowWorkspace) -> bool:
    """Return whether a workspace has source packs for the generic analyze path."""

    source_pack_dir = workspace.root / "analysis" / "narrative_source_packs"
    if not source_pack_dir.is_dir():
        return False
    return any(path.is_file() for path in source_pack_dir.glob("*.json"))


def _validate_workspace_support(workspace: WorkflowWorkspace, stage: WorkflowStage) -> None:
    """Reject stage/workspace combinations that still depend on NeurIPS-only scripts."""

    if is_presentation_stage_id(stage.id):
        return
    if stage.id == "analyze" and _has_narrative_source_packs(workspace):
        return
    if (
        is_extraction_stage_id(stage.id)
        or is_indexing_stage_id(stage.id)
        or is_research_stage_id(stage.id)
        or is_paper_matching_stage_id(stage.id)
        or is_synthesis_stage_id(stage.id)
    ):
        # Phase 14A / 14B / 14C — allow ICLR (and the canonical NeurIPS
        # workspace) explicitly. Phase 14C reuses the Phase 14A/14B allow-list
        # since the research stage consumes Phase 14A by-topic seeds + Phase
        # 14B vector store artifacts produced for the same set of workspaces.
        # TODO(v1.6+): source supported workspace slugs from a single config key
        # (e.g., workflow.extraction_supported_workspaces) instead of a hardcoded
        # set keyed on Path.name.
        if workspace.root.name in EXTRACTION_SUPPORTED_WORKSPACE_NAMES:
            return
        if workspace.root == get_default_workspace_path():
            return
        raise ValueError(
            f"Workflow stage {stage.id} is only routed against the ICLR 2025 "  # conference-literal-ok: guarded legacy error message; routing uses EXTRACTION_SUPPORTED_WORKSPACE_NAMES
            "workspace today; extend the allow-list when broadening conference support."
        )
    if workspace.root == get_default_workspace_path():
        return
    if stage.id == "podcast":
        return

    raise ValueError(
        f"Workflow stage {stage.id} only supports the canonical NeurIPS workspace "
        "until cross-conference routing is implemented."
    )


def ordered_workflow_stages(
    *,
    include_presentation: bool = True,
) -> tuple[WorkflowStage, ...]:
    """Return the canonical ordered workflow registry."""

    _assert_presentation_stage_catalog_alignment()
    if include_presentation:
        return WORKFLOW_STAGES
    return LEGACY_WORKFLOW_STAGES


def get_workflow_stage(stage_id: str) -> WorkflowStage:
    """Return a workflow stage by id."""

    try:
        return WORKFLOW_STAGE_BY_ID[stage_id]
    except KeyError as exc:
        raise ValueError(f"Unknown workflow stage: {stage_id}") from exc


def parse_stage_selection(
    selection: str | None,
    *,
    include_presentation: bool = True,
) -> list[WorkflowStage]:
    """Parse a comma-separated stage selection into canonical ordered stages."""

    available_stages = ordered_workflow_stages(include_presentation=include_presentation)
    available_stage_by_id = {stage.id: stage for stage in available_stages}
    if selection is None or not selection.strip():
        return list(available_stages)

    requested = {part.strip() for part in selection.split(",") if part.strip()}
    if not requested:
        raise ValueError("At least one workflow stage must be selected.")

    unknown = sorted(stage_id for stage_id in requested if stage_id not in available_stage_by_id)
    if unknown:
        raise ValueError(f"Unknown workflow stage(s): {', '.join(unknown)}")

    return [stage for stage in available_stages if stage.id in requested]


def is_presentation_stage_id(stage_id: str) -> bool:
    """Return whether a workflow stage id is a manifest-backed presentation checkpoint."""

    return stage_id in PRESENTATION_STAGE_ID_SET or stage_id in VALIDATION_STAGE_ID_SET


def is_presentation_validation_stage_id(stage_id: str) -> bool:
    """Return whether a workflow stage id validates a presentation checkpoint."""

    return stage_id in VALIDATION_STAGE_ID_SET


def is_extraction_stage_id(stage_id: str) -> bool:
    """Return whether a workflow stage id is the exhaustive extraction stage (Phase 14A)."""

    return stage_id in EXTRACTION_STAGE_ID_SET


def is_indexing_stage_id(stage_id: str) -> bool:
    """Return whether a workflow stage id is the enhanced-indexing-vector-store stage (Phase 14B)."""

    return stage_id in INDEXING_STAGE_ID_SET


def is_research_stage_id(stage_id: str) -> bool:
    """Return whether a workflow stage id is the query-planning-gpt-researcher stage (Phase 14C)."""

    return stage_id in RESEARCH_STAGE_ID_SET


def is_paper_matching_stage_id(stage_id: str) -> bool:
    """Return whether a workflow stage id is the v1.91 paper matching stage."""

    return stage_id in PAPER_MATCHING_STAGE_ID_SET


def is_coverage_stage_id(stage_id: str) -> bool:
    """Return whether a workflow stage id is the coverage-gap-fill stage (Phase 16A)."""

    return stage_id in COVERAGE_STAGE_ID_SET


def is_synthesis_stage_id(stage_id: str) -> bool:
    """Return whether a workflow stage id is the synthesis-trend-report stage (Phase 19A)."""

    return stage_id in SYNTHESIS_STAGE_ID_SET


def _assert_presentation_stage_catalog_alignment() -> None:
    from conference_pipeline.presentation.manifest import PRESENTATION_STAGE_IDS
    from conference_pipeline.presentation.validation import VALIDATION_STAGE_IDS

    local_ids = tuple(stage.id for stage in PRESENTATION_WORKFLOW_STAGES)
    if local_ids != PRESENTATION_STAGE_IDS:
        raise RuntimeError(
            "Workflow presentation stages must match presentation manifest checkpoint ids: "
            f"{local_ids!r} != {PRESENTATION_STAGE_IDS!r}"
        )
    validation_ids = tuple(stage.id for stage in VALIDATION_WORKFLOW_STAGES)
    if validation_ids != VALIDATION_STAGE_IDS:
        raise RuntimeError(
            "Workflow validation stages must match presentation validation ids: "
            f"{validation_ids!r} != {VALIDATION_STAGE_IDS!r}"
        )


def _resolve_presentation_topic(
    presentations: Sequence[Any],
    topic: str,
) -> str:
    normalized = topic.strip().replace(".md", "")
    known_topics = sorted(
        {
            topic_id
            for presentation in presentations
            for topic_id in presentation.topic_scope.topics
        }
    )
    if normalized in known_topics:
        return normalized

    candidates: list[str] = []
    if normalized.isdigit():
        prefix = normalized.zfill(2) + "_"
        candidates = [topic_id for topic_id in known_topics if topic_id.startswith(prefix)]
    elif len(normalized) >= 2 and normalized[:2].isdigit():
        prefix = normalized[:2] + "_"
        candidates = [topic_id for topic_id in known_topics if topic_id.startswith(prefix)]
    else:
        candidates = [
            topic_id
            for topic_id in known_topics
            if topic_id.split("_", 1)[-1] == normalized
        ]

    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(f"Ambiguous presentation topic: {topic}")
    raise ValueError(f"Unknown presentation topic: {topic}")


def resolve_presentation_selection(
    workspace: WorkflowWorkspace,
    stage: WorkflowStage,
    *,
    presentation_type: str | None = None,
    topic: str | None = None,
) -> PresentationStageSelection:
    """Resolve a first-class presentation stage to one manifest entry and scope."""

    if not is_presentation_stage_id(stage.id):
        raise ValueError(f"Stage {stage.id} is not a presentation workflow stage")
    if presentation_type is None or not presentation_type.strip():
        raise ValueError(f"Presentation stage {stage.id} requires --presentation-type")

    from conference_pipeline.presentation.manifest import (
        PRESENTATION_TYPES,
        load_and_validate_manifest,
    )
    from conference_pipeline.presentation.validation import VALIDATION_STAGE_TO_CHECKPOINT

    normalized_type = presentation_type.strip()
    if normalized_type not in PRESENTATION_TYPES:
        allowed = ", ".join(PRESENTATION_TYPES)
        raise ValueError(f"--presentation-type must be one of: {allowed}")

    loaded = load_and_validate_manifest(workspace, require_path_existence=False)
    stage_spec_by_id = {spec.id: spec for spec in loaded.manifest.stage_specs}
    checkpoint_stage_id = VALIDATION_STAGE_TO_CHECKPOINT.get(stage.id, stage.id)
    stage_spec = stage_spec_by_id[checkpoint_stage_id]
    selected_topic: str | None = None
    if normalized_type == "topic-deep-dive":
        if topic is None or not topic.strip():
            raise ValueError("--topic is required when --presentation-type topic-deep-dive")
        selected_topic = _resolve_presentation_topic(loaded.manifest.presentations, topic)

    matches = [
        presentation
        for presentation in loaded.manifest.presentations
        if presentation.presentation_type == normalized_type
        and (
            normalized_type != "topic-deep-dive"
            or selected_topic in presentation.topic_scope.topics
        )
    ]
    if len(matches) != 1:
        descriptor = normalized_type if selected_topic is None else f"{normalized_type}:{selected_topic}"
        raise ValueError(
            f"Expected exactly one presentation_manifest.json entry for {descriptor}; "
            f"found {len(matches)}"
        )
    presentation = matches[0]
    return PresentationStageSelection(
        loaded_manifest=loaded,
        stage_spec=stage_spec,
        presentation=presentation,
        selected_scope=PresentationSelectedScope(
            presentation_id=presentation.id,
            presentation_type=presentation.presentation_type,
            topic=selected_topic,
        ),
    )


def _resolve_path_inside_workspace(workspace_root: Path, relative_path: str, *, label: str) -> Path:
    """Resolve a single, pipeline-recorded artifact path safely.

    Containment is checked lexically (``os.path.abspath`` normalizes
    ``.``/``..`` without following symlinks), not via a symlink-following
    ``Path.resolve()``, so a workspace-owned symlink -- for example a
    Drive-backed on-demand artifact leaf recorded in a conference's
    storage_manifest.json, such as storage/transcription-ocr -- is not
    rejected just because it resolves outside the repository tree. A ``..``
    escape attempt embedded in ``relative_path`` is still rejected, since
    normpath collapses it before the comparison.
    """
    resolved = Path(os.path.abspath(str(workspace_root / relative_path)))
    workspace_root_abs = Path(os.path.abspath(str(workspace_root)))
    try:
        resolved.relative_to(workspace_root_abs)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the conference workspace") from exc
    return resolved


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed conference config: could not parse YAML in {path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read conference config: {path}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Malformed conference config: expected mapping in {path}")
    return loaded


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_string_tuple(values: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, list):
        return fallback
    normalized = tuple(str(value) for value in values if isinstance(value, str) and value.strip())
    return normalized or fallback


def _as_float(value: Any, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback


def load_workflow_config(config_data: dict[str, Any]) -> WorkflowConfig:
    """Load workflow defaults from a conference config mapping."""

    defaults = default_workflow_config()
    workflow_data = _as_mapping(config_data.get("workflow"))
    podcast_data = _as_mapping(workflow_data.get("podcast"))
    deck_data = _as_mapping(workflow_data.get("deck"))

    return WorkflowConfig(
        podcast=PodcastWorkflowConfig(
            target_minutes=int(podcast_data.get("target_minutes", defaults.podcast.target_minutes)),
            max_tier=int(podcast_data.get("max_tier", defaults.podcast.max_tier)),
            top_n_facts=int(podcast_data.get("top_n_facts", defaults.podcast.top_n_facts)),
            host_a=str(podcast_data.get("host_a", defaults.podcast.host_a)),
            host_b=str(podcast_data.get("host_b", defaults.podcast.host_b)),
            model=str(podcast_data.get("model", defaults.podcast.model)),
        ),
        deck=DeckWorkflowConfig(
            target_slides=int(deck_data.get("target_slides", defaults.deck.target_slides)),
            target_hyperlinks_per_slide=_as_float(
                deck_data.get(
                    "target_hyperlinks_per_slide",
                    defaults.deck.target_hyperlinks_per_slide,
                ),
                defaults.deck.target_hyperlinks_per_slide,
            ),
            target_svgs_per_slide=_as_float(
                deck_data.get("target_svgs_per_slide", defaults.deck.target_svgs_per_slide),
                defaults.deck.target_svgs_per_slide,
            ),
            reflow_model=str(deck_data.get("reflow_model", defaults.deck.reflow_model)),
            max_papers=int(deck_data.get("max_papers", defaults.deck.max_papers)),
            required_env=_as_string_tuple(deck_data.get("required_env"), defaults.deck.required_env),
        ),
    )


def resolve_workspace(workspace: str | Path) -> WorkflowWorkspace:
    """Resolve and validate a conference workspace root."""

    workspace_root = Path(workspace)
    if not workspace_root.is_absolute():
        workspace_root = (PROJECT_ROOT / workspace_root).resolve()
    else:
        workspace_root = workspace_root.resolve()

    if not workspace_root.exists():
        raise ValueError(f"Conference workspace not found: {workspace_root}")
    if not workspace_root.is_dir():
        raise ValueError(f"Conference workspace is not a directory: {workspace_root}")

    config_path = workspace_root / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Conference workspace is missing config.yaml: {workspace_root}")
    if not config_path.is_file():
        raise ValueError(f"Conference workspace config is not a file: {config_path}")

    config_data = _load_yaml_mapping(config_path)
    storage_data = _as_mapping(config_data.get("storage"))
    manifest_rel = str(storage_data.get("manifest_path", "storage/storage_manifest.json"))
    manifest_path = _resolve_path_inside_workspace(
        workspace_root,
        manifest_rel,
        label="storage.manifest_path",
    )

    if not manifest_path.exists():
        raise ValueError(f"Conference workspace is missing storage manifest: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Conference workspace manifest is not a file: {manifest_path}")

    return WorkflowWorkspace(
        root=workspace_root,
        config_path=config_path,
        manifest_path=manifest_path,
        config_data=config_data,
        workflow=load_workflow_config(config_data),
    )


def list_workflow_topics(workspace: WorkflowWorkspace) -> list[str]:
    """Return numbered analysis topic ids from the workspace's manifest."""

    from conference_pipeline.analysis.errors import ManifestValidationError
    from conference_pipeline.analysis.manifest import load_and_validate_manifest

    try:
        loaded = load_and_validate_manifest(workspace, require_path_existence=False)
    except ManifestValidationError:
        return []
    return [topic.id for topic in loaded.manifest.topics]


def resolve_topic_slug(workspace: WorkflowWorkspace, topic: str) -> str:
    """Resolve a topic input like `01` or `01_tools_and_open_source`."""

    topics = list_workflow_topics(workspace)
    if not topics:
        raise ValueError(f"No numbered analysis topics found in {workspace.root / 'analysis'}")

    normalized = topic.strip().replace(".md", "")
    if normalized in topics:
        return normalized

    if normalized.isdigit():
        prefix = normalized.zfill(2) + "_"
        for candidate in topics:
            if candidate.startswith(prefix):
                return candidate

    if len(normalized) >= 2 and normalized[:2].isdigit():
        prefix = normalized[:2] + "_"
        for candidate in topics:
            if candidate.startswith(prefix):
                return candidate

    raise ValueError(f"Unknown workflow topic: {topic}")


def format_workflow_stage(stage: WorkflowStage) -> str:
    """Format a workflow stage for CLI display."""

    return f"{stage.id}\t{stage.description}\t{stage.dispatch_kind}\t{stage.implementation_ref}"


def _run_subprocess(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def _run_presentation_owner_commands(
    stage: WorkflowStage,
    plans: Sequence[Any],
    *,
    command_runner: CommandRunner,
) -> None:
    failures: list[Exception] = []
    run_all_before_failing = stage.id in RUN_ALL_PRESENTATION_OWNER_COMMANDS_BEFORE_FAILING
    for plan in plans:
        if plan.unsupported_reason:
            raise ValueError(
                f"Presentation stage {stage.id} has no live owner command: "
                f"{plan.unsupported_reason}"
            )
        try:
            command_runner(plan.argv, PROJECT_ROOT)
        except Exception as exc:
            if not run_all_before_failing:
                raise
            failures.append(exc)
    if failures:
        raise RuntimeError(
            f"Presentation stage {stage.id} failed after running all "
            f"{len(plans)} owner command(s); {len(failures)} command(s) failed."
        ) from failures[0]


def _select_topic_entries(manifest: "AnalysisManifest", topic: str | None) -> list["TopicEntry"]:
    """Resolve --topic input against manifest topic entries."""

    if topic is None:
        return list(manifest.topics)

    normalized = topic.strip().replace(".md", "")
    by_id = {entry.id: entry for entry in manifest.topics}
    if normalized in by_id:
        return [by_id[normalized]]

    if normalized.isdigit():
        prefix = normalized.zfill(2) + "_"
        for entry in manifest.topics:
            if entry.id.startswith(prefix):
                return [entry]

    if len(normalized) >= 2 and normalized[:2].isdigit():
        prefix = normalized[:2] + "_"
        for entry in manifest.topics:
            if entry.id.startswith(prefix):
                return [entry]

    by_slug = {entry.slug: entry for entry in manifest.topics}
    if normalized in by_slug:
        return [by_slug[normalized]]

    raise ValueError(f"Unknown workflow topic: {topic}")


def _podcast_command_from_entry(workspace: WorkflowWorkspace, entry: "TopicEntry") -> list[str]:
    """Build byte-equivalent podcast argv from a validated manifest topic."""

    from conference_pipeline.analysis.errors import ManifestValidationError

    podcast_config = workspace.workflow.podcast
    if entry.fact_ranks_path is None:
        raise ManifestValidationError(
            f"Topic {entry.id} has no fact_ranks_path; podcast generation requires ranks.json."
        )
    if entry.audio_dir is None:
        raise ManifestValidationError(
            f"Topic {entry.id} has no audio_dir; podcast output directory cannot be resolved."
        )

    analysis_file = workspace.root / entry.markdown_path
    fact_ranks = workspace.root / entry.fact_ranks_path
    output_dir = workspace.root / entry.audio_dir / "focused_podcast"
    return [
        SUBPROCESS_PYTHON,
        "scripts/generate_focused_podcast.py",
        str(analysis_file),
        "--fact-ranks",
        str(fact_ranks),
        "--output-dir",
        str(output_dir),
        "--target-minutes",
        str(podcast_config.target_minutes),
        "--max-tier",
        str(podcast_config.max_tier),
        "--top-n-facts",
        str(podcast_config.top_n_facts),
        "--two-hosts",
        "--host-a",
        podcast_config.host_a,
        "--host-b",
        podcast_config.host_b,
        "--model",
        podcast_config.model,
    ]


def _deck_commands_from_entries(
    workspace: WorkflowWorkspace,
    entries: list["TopicEntry"],
    *,
    topic_selected: bool,
) -> list[list[str]]:
    """Build byte-equivalent deck argv while gating topic selection via manifest."""

    deck_cfg = workspace.workflow.deck
    topic_args = ["--topic", f"{entries[0].number:02d}"] if topic_selected and len(entries) == 1 else ["--all"]
    return [
        [
            SUBPROCESS_PYTHON,
            "scripts/presentations/generate_slides_agent.py",
            *topic_args,
            "--slides",
            str(deck_cfg.target_slides),
            "-v",
        ],
        [
            SUBPROCESS_PYTHON,
            "scripts/presentations/enrich_slides_with_papers.py",
            *topic_args,
            "--slides",
            str(deck_cfg.target_slides),
            "-v",
        ],
        [
            SUBPROCESS_PYTHON,
            "scripts/presentations/reflow_slides.py",
            *topic_args,
            "--model",
            deck_cfg.reflow_model,
            "-v",
        ],
        [SUBPROCESS_PYTHON, "scripts/presentations/sanity_check_slides.py", *topic_args, "-v"],
        [SUBPROCESS_PYTHON, "scripts/presentations/enrich_hyperlinks_hybrid.py", *topic_args, "-v"],
        [SUBPROCESS_PYTHON, "scripts/presentations/generate_presenter_script.py", *topic_args, "-v"],
    ]


def _presentation_command_plans(
    workspace: WorkflowWorkspace,
    stage: WorkflowStage,
    *,
    topic: str | None,
    presentation_type: str | None,
    mode: str = "dry-run",
    run_id: str | None = None,
) -> list[Any]:
    try:
        selection = resolve_presentation_selection(
            workspace,
            stage,
            presentation_type=presentation_type,
            topic=topic,
        )
    except Exception as exc:
        if (
            exc.__class__.__name__ == "PresentationManifestValidationError"
            and "Missing presentation_manifest.json" in str(exc)
        ):
            raise ValueError(
                f"Missing decks/presentation_manifest.json: "
                f"{workspace.root / 'decks' / 'presentation_manifest.json'}"
            ) from exc
        raise

    from conference_pipeline.presentation.commands import build_presentation_command_plans

    return build_presentation_command_plans(
        workspace.root,
        selection,
        python_executable=SUBPROCESS_PYTHON,
        mode=mode,
        run_id=run_id,
    )


def build_stage_commands(
    workspace: WorkflowWorkspace,
    stage: WorkflowStage,
    *,
    topic: str | None = None,
    presentation_type: str | None = None,
    mode: str = "dry-run",
    run_id: str | None = None,
) -> list[list[str]]:
    """Build subprocess commands for a workflow stage."""

    if stage.dispatch_kind != "subprocess_stage":
        raise ValueError(f"Stage {stage.id} is not a subprocess stage")
    _validate_workspace_support(workspace, stage)

    if is_presentation_stage_id(stage.id):
        plans = _presentation_command_plans(
            workspace,
            stage,
            topic=topic,
            presentation_type=presentation_type,
            mode=mode,
            run_id=run_id,
        )
        return [plan.argv for plan in plans]

    if stage.id == "podcast":
        from conference_pipeline.analysis.manifest import load_and_validate_manifest

        loaded = load_and_validate_manifest(workspace, require_path_existence=True)
        entries = _select_topic_entries(loaded.manifest, topic)
        return [_podcast_command_from_entry(workspace, entry) for entry in entries]

    if stage.id == "deck":
        from conference_pipeline.analysis.manifest import load_and_validate_manifest

        loaded = load_and_validate_manifest(workspace, require_path_existence=True)
        entries = _select_topic_entries(loaded.manifest, topic) if topic is not None else list(loaded.manifest.topics)
        return _deck_commands_from_entries(
            workspace,
            entries,
            topic_selected=topic is not None,
        )

    raise ValueError(f"Unhandled subprocess workflow stage: {stage.id}")


def build_stage_command_metadata(
    workspace: WorkflowWorkspace,
    stage: WorkflowStage,
    *,
    topic: str | None = None,
    presentation_type: str | None = None,
    mode: str = "dry-run",
    run_id: str | None = None,
) -> list[dict[str, object]]:
    """Build rich command metadata for first-class presentation stages."""

    if not is_presentation_stage_id(stage.id):
        return []
    if stage.dispatch_kind != "subprocess_stage":
        return []
    _validate_workspace_support(workspace, stage)
    plans = _presentation_command_plans(
        workspace,
        stage,
        topic=topic,
        presentation_type=presentation_type,
        mode=mode,
        run_id=run_id,
    )
    return [plan.to_dict() for plan in plans]


def _pipeline_settings(env_file: Path | None, storage_root: Path | None) -> PipelineSettings:
    settings = load_settings(env_file)
    if storage_root is not None:
        settings.storage_root = storage_root
    return settings


def _validate_required_env(required_env: tuple[str, ...]) -> None:
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(f"Missing required env for deck stage: {missing_csv}")


def _get_pipeline_runtime():
    from conference_pipeline.models import PresentationStatus
    from conference_pipeline.pipeline import NeuripsPipeline

    return NeuripsPipeline, PresentationStatus


async def _run_discover_stage(*, env_file: Path | None, storage_root: Path | None) -> None:
    NeuripsPipeline, _ = _get_pipeline_runtime()
    settings = _pipeline_settings(env_file, storage_root)
    pipeline = NeuripsPipeline(settings)
    await pipeline.initialize()
    try:
        await pipeline.login()
        await pipeline.discover_presentations()
    finally:
        await pipeline.cleanup()


async def _run_download_stage(*, env_file: Path | None, storage_root: Path | None) -> None:
    NeuripsPipeline, PresentationStatus = _get_pipeline_runtime()
    settings = _pipeline_settings(env_file, storage_root)
    pipeline = NeuripsPipeline(settings)
    await pipeline.initialize()
    try:
        presentations = pipeline.storage.load_presentations_by_status(PresentationStatus.DISCOVERED)
        for presentation in presentations:
            await pipeline.download_assets_for_presentation(presentation)
    finally:
        await pipeline.cleanup()


async def _run_transcribe_stage(*, env_file: Path | None, storage_root: Path | None) -> None:
    NeuripsPipeline, PresentationStatus = _get_pipeline_runtime()
    settings = _pipeline_settings(env_file, storage_root)
    pipeline = NeuripsPipeline(settings)
    await pipeline.initialize()
    try:
        presentations = pipeline.storage.load_presentations_by_status(
            PresentationStatus.ASSETS_DOWNLOADED
        )
        for presentation in presentations:
            await pipeline.transcribe_presentation(presentation)
    finally:
        await pipeline.cleanup()


def run_stage(
    workspace: WorkflowWorkspace,
    stage: WorkflowStage,
    *,
    topic: str | None = None,
    presentation_type: str | None = None,
    mode: str = "live",
    run_id: str | None = None,
    env_file: Path | None = None,
    storage_root: Path | None = None,
    command_runner: CommandRunner = _run_subprocess,
    strict_required_artifacts: bool = False,
) -> Any:
    """Run a single workflow stage."""

    _validate_workspace_support(workspace, stage)

    if is_presentation_validation_stage_id(stage.id):
        from conference_pipeline.presentation.validation import run_validation_stage

        return run_validation_stage(
            workspace,
            stage,
            presentation_type=presentation_type,
            topic=topic,
            strict_required_artifacts=strict_required_artifacts,
        )

    if is_presentation_stage_id(stage.id):
        plans = _presentation_command_plans(
            workspace,
            stage,
            topic=topic,
            presentation_type=presentation_type,
            mode=mode,
            run_id=run_id,
        )
        _run_presentation_owner_commands(
            stage,
            plans,
            command_runner=command_runner,
        )
        return

    if stage.dispatch_kind == "python_stage":
        if stage.id == "discover":
            asyncio.run(_run_discover_stage(env_file=env_file, storage_root=storage_root))
            return
        if stage.id == "download":
            asyncio.run(_run_download_stage(env_file=env_file, storage_root=storage_root))
            return
        if stage.id == "transcribe":
            asyncio.run(_run_transcribe_stage(env_file=env_file, storage_root=storage_root))
            return
        if stage.id == "analyze":
            from conference_pipeline.analysis import run_analyze_stage

            return run_analyze_stage(
                workspace,
                run_id=run_id,
                env_file=env_file,
                storage_root=storage_root,
            )
        if stage.id == "exhaustive-topic-extraction":
            from conference_pipeline.iclr2025_live import (
                PRODUCTION_RUN_ID,
                run_exhaustive_topic_extraction,
            )

            run_id_env = os.getenv("EXTRACTION_RUN_ID") or PRODUCTION_RUN_ID
            force_rebuild = bool(os.getenv("EXTRACTION_FORCE_REBUILD"))
            run_exhaustive_topic_extraction(
                workspace.root,
                run_id=run_id_env,
                force_rebuild=force_rebuild,
            )
            return
        if stage.id == "enhanced-indexing-vector-store":
            from conference_pipeline.iclr2025_live import (
                PRODUCTION_RUN_ID,
                run_enhanced_indexing_vector_store,
            )

            run_id_env = (
                os.getenv("INDEXING_RUN_ID")
                or os.getenv("EXTRACTION_RUN_ID")
                or PRODUCTION_RUN_ID
            )
            force_rebuild = bool(os.getenv("INDEXING_FORCE_REBUILD"))
            run_enhanced_indexing_vector_store(
                workspace.root,
                run_id=run_id_env,
                force_rebuild=force_rebuild,
            )
            return
        if stage.id == "query-planning-gpt-researcher":
            from conference_pipeline.iclr2025_live import (
                PRODUCTION_RUN_ID,
                run_query_planning_gpt_researcher,
            )

            run_id_env = (
                os.getenv("RESEARCH_RUN_ID")
                or os.getenv("EXTRACTION_RUN_ID")
                or PRODUCTION_RUN_ID
            )
            force_rebuild = bool(os.getenv("RESEARCH_FORCE_REBUILD"))
            run_query_planning_gpt_researcher(
                workspace.root,
                run_id=run_id_env,
                force_rebuild=force_rebuild,
            )
            return
        if stage.id == "paper-matching":
            from conference_pipeline.iclr2025_live import PRODUCTION_RUN_ID
            from conference_pipeline.paper_matching import run_paper_matching_stage

            run_id_env = (
                run_id
                or os.getenv("PAPER_MATCHING_RUN_ID")
                or os.getenv("RESEARCH_RUN_ID")
                or os.getenv("INDEXING_RUN_ID")
                or os.getenv("EXTRACTION_RUN_ID")
                or PRODUCTION_RUN_ID
            )
            return run_paper_matching_stage(
                workspace_root=workspace.root,
                run_id=run_id_env,
                topic_id=topic,
            )
        if stage.id == "coverage-gap-fill":
            from conference_pipeline.iclr2025_live import (
                PRODUCTION_RUN_ID,
                run_coverage_gap_fill_stage,
            )

            run_id_env = (
                os.getenv("COVERAGE_RUN_ID")
                or os.getenv("EXTRACTION_RUN_ID")
                or PRODUCTION_RUN_ID
            )
            force_rebuild = bool(os.getenv("COVERAGE_FORCE_REBUILD"))
            run_coverage_gap_fill_stage(
                workspace.root,
                run_id=run_id_env,
                force_rebuild=force_rebuild,
            )
            return
        if stage.id == "synthesis-trend-report":
            from conference_pipeline.iclr2025_live import (
                PRODUCTION_RUN_ID,
                run_synthesis_trend_report_stage,
            )

            run_id_env = (
                os.getenv("SYNTHESIS_RUN_ID")
                or os.getenv("EXTRACTION_RUN_ID")
                or PRODUCTION_RUN_ID
            )
            force_rebuild = bool(os.getenv("SYNTHESIS_FORCE_REBUILD"))
            run_synthesis_trend_report_stage(
                workspace.root,
                run_id=run_id_env,
                force_rebuild=force_rebuild,
            )
            return
        raise ValueError(f"Unhandled python workflow stage: {stage.id}")

    if stage.id == "deck":
        _validate_required_env(workspace.workflow.deck.required_env)

    for command in build_stage_commands(
        workspace,
        stage,
        topic=topic,
        presentation_type=presentation_type,
        run_id=run_id,
    ):
        command_runner(command, PROJECT_ROOT)


def run_selected_stages(
    workspace: WorkflowWorkspace,
    selected_stages: Sequence[WorkflowStage],
    *,
    topic: str | None = None,
    env_file: Path | None = None,
    storage_root: Path | None = None,
    mode: str = "live",
    run_id: str | None = None,
    resume: bool = False,
    write_receipts: bool = False,
    command_runner: CommandRunner = _run_subprocess,
    presentation_type: str | None = None,
    strict_required_artifacts: bool = False,
) -> Any:
    """Run selected workflow stages in canonical order."""

    if mode == "live" and run_id is None and not resume and not write_receipts:
        from conference_pipeline.workflow_runs import (
            WorkflowBlockedError,
            blocker_from_stage_result,
        )

        for stage in selected_stages:
            stage_result = run_stage(
                workspace,
                stage,
                topic=topic,
                presentation_type=presentation_type,
                mode=mode,
                run_id=run_id,
                env_file=env_file,
                storage_root=storage_root,
                command_runner=command_runner,
                strict_required_artifacts=strict_required_artifacts,
            )
            blocker = blocker_from_stage_result(stage_result, stage_id=stage.id)
            if blocker is not None:
                raise WorkflowBlockedError(
                    stage_id=stage.id,
                    blocker_class=str(blocker["blocker_class"]),
                    reason=str(blocker["reason"]),
                )
        return None

    from conference_pipeline.workflow_runs import (
        WorkflowExecutionMode,
        WorkflowRunOptions,
        generate_run_id,
        run_workflow_stages,
    )

    execution_mode = WorkflowExecutionMode(mode)
    options = WorkflowRunOptions(
        run_id=run_id or generate_run_id(),
        mode=execution_mode,
        resume=resume,
        topic=topic,
        presentation_type=presentation_type,
        env_file=env_file,
        storage_root=storage_root,
        strict_required_artifacts=strict_required_artifacts,
    )
    return run_workflow_stages(
        workspace,
        selected_stages,
        options,
        command_runner=command_runner,
    )
