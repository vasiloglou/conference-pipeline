"""Presentation contract package for workspace-level presentation manifests."""

from __future__ import annotations

from conference_pipeline.presentation.errors import (
    PresentationContractError,
    PresentationManifestValidationError,
)
from conference_pipeline.presentation.commands import (
    PresentationCommandPlan,
    build_presentation_command_plans,
)
from conference_pipeline.presentation.manifest import (
    DEFAULT_STAGE_SPECS,
    MANIFEST_FILENAME,
    MANIFEST_RELATIVE_PATH,
    MANIFEST_SCHEMA_VERSION,
    PRESENTATION_STAGE_IDS,
    PRESENTATION_TYPES,
    SUPPORTED_SCHEMA_VERSIONS,
    LoadedPresentationManifest,
    PresentationArtifactRef,
    PresentationEntry,
    PresentationManifest,
    PresentationStageSpec,
    PresentationStageState,
    PresentationTopicScope,
    StageModeReadiness,
    build_backfilled_presentations,
    emit_presentation_manifest,
    load_and_validate_manifest,
)
from conference_pipeline.presentation.narrative_gate import (
    DEFAULT_DECK_STAGE_FILENAMES,
    DEFAULT_POST_STAGE14_DECK_STALE_MARKERS,
    DEFAULT_PUBLIC_LEDGER_HEADINGS,
    parse_narrative_analysis,
    validate_presentation_narrative_contract,
)

__all__ = [
    "DEFAULT_STAGE_SPECS",
    "DEFAULT_DECK_STAGE_FILENAMES",
    "DEFAULT_POST_STAGE14_DECK_STALE_MARKERS",
    "DEFAULT_PUBLIC_LEDGER_HEADINGS",
    "LoadedPresentationManifest",
    "MANIFEST_FILENAME",
    "MANIFEST_RELATIVE_PATH",
    "MANIFEST_SCHEMA_VERSION",
    "PRESENTATION_STAGE_IDS",
    "PRESENTATION_TYPES",
    "PresentationCommandPlan",
    "PresentationArtifactRef",
    "PresentationContractError",
    "PresentationEntry",
    "PresentationManifest",
    "PresentationManifestValidationError",
    "PresentationStageSpec",
    "PresentationStageState",
    "PresentationTopicScope",
    "SUPPORTED_SCHEMA_VERSIONS",
    "StageModeReadiness",
    "build_backfilled_presentations",
    "build_presentation_command_plans",
    "emit_presentation_manifest",
    "load_and_validate_manifest",
    "parse_narrative_analysis",
    "validate_presentation_narrative_contract",
]
