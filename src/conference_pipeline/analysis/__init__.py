"""Phase 12 composable analysis pipeline - run_analyze_stage + analysis_manifest.json contract."""

from __future__ import annotations

__version__ = "0.1.0"

from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.analysis.constants import NEURIPS_TOPIC_IDS
from conference_pipeline.analysis.errors import (
    AnalysisContractError,
    ManifestValidationError,
    StepError,
)
from conference_pipeline.analysis.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    MIN_COMPATIBLE_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
)

__all__ = [
    "AnalysisContractError",
    "AnalysisManifest",
    "AnalyzeStageResult",
    "LoadedManifest",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "MIN_COMPATIBLE_SCHEMA_VERSION",
    "ManifestValidationError",
    "NEURIPS_TOPIC_IDS",
    "StepError",
    "SUPPORTED_SCHEMA_VERSIONS",
    "TopicEntry",
    "emit_analysis_manifest",
    "load_and_validate_manifest",
    "render_narrative_topic_report",
    "render_slide_brief",
    "render_topic_provenance_json",
    "run_topic_reconciliation",
    "run_topic_reconciliation_reference_workflow",
    "run_topic_reference_candidate_resolution",
    "run_topic_reference_validation",
    "run_analyze_stage",
    "sha256_file",
    "validate_narrative_topic_report",
]


def __getattr__(name: str):
    if name in {"run_analyze_stage", "AnalyzeStageResult"}:
        from conference_pipeline.analysis.orchestrator import AnalyzeStageResult, run_analyze_stage

        exports = {
            "AnalyzeStageResult": AnalyzeStageResult,
            "run_analyze_stage": run_analyze_stage,
        }
        return exports[name]
    if name in {
        "render_narrative_topic_report",
        "render_slide_brief",
        "render_topic_provenance_json",
        "validate_narrative_topic_report",
    }:
        from conference_pipeline.analysis.narrative import (
            render_narrative_topic_report,
            render_slide_brief,
            render_topic_provenance_json,
            validate_narrative_topic_report,
        )

        exports = {
            "render_narrative_topic_report": render_narrative_topic_report,
            "render_slide_brief": render_slide_brief,
            "render_topic_provenance_json": render_topic_provenance_json,
            "validate_narrative_topic_report": validate_narrative_topic_report,
        }
        return exports[name]
    if name in {"run_topic_reconciliation"}:
        from conference_pipeline.analysis.topic_reconciliation import run_topic_reconciliation

        return run_topic_reconciliation
    if name in {"run_topic_reconciliation_reference_workflow"}:
        from conference_pipeline.analysis.topic_reconciliation_reference_workflow import (
            run_topic_reconciliation_reference_workflow,
        )

        return run_topic_reconciliation_reference_workflow
    if name in {"run_topic_reference_validation"}:
        from conference_pipeline.analysis.topic_reference_validation import (
            run_topic_reference_validation,
        )

        return run_topic_reference_validation
    if name in {"run_topic_reference_candidate_resolution"}:
        from conference_pipeline.analysis.topic_reference_candidate_resolution import (
            resolve_topic_reference_candidates,
        )

        return resolve_topic_reference_candidates
    if name in {
        "AnalysisManifest",
        "TopicEntry",
        "LoadedManifest",
        "load_and_validate_manifest",
        "emit_analysis_manifest",
    }:
        from conference_pipeline.analysis.manifest import (
            AnalysisManifest,
            LoadedManifest,
            TopicEntry,
            emit_analysis_manifest,
            load_and_validate_manifest,
        )

        exports = {
            "AnalysisManifest": AnalysisManifest,
            "LoadedManifest": LoadedManifest,
            "TopicEntry": TopicEntry,
            "emit_analysis_manifest": emit_analysis_manifest,
            "load_and_validate_manifest": load_and_validate_manifest,
        }
        return exports[name]
    raise AttributeError(name)
