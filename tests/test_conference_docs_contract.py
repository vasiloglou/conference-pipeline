from __future__ import annotations
import pytest

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_OF_TRUTH = REPO_ROOT / "docs" / "source-of-truth.md"
CAPABILITIES = REPO_ROOT / "docs" / "reference" / "pipeline-capabilities.md"
PLAYBOOK = REPO_ROOT / "docs" / "workflows" / "conference-analysis-playbook.md"
CLAUDE = REPO_ROOT / "CLAUDE.md"
AGENTS = REPO_ROOT / "AGENTS.md"
MILESTONES = REPO_ROOT / ".planning" / "MILESTONES.md"
NEURIPS_V1_CONTEXT = REPO_ROOT / ".planning" / "milestones" / "v1.0-NeurIPS-2025-CONTEXT.md"
ICLR2025_CONTEXT = REPO_ROOT / ".planning" / "milestones" / "v1.2-ICLR-2025-CONTEXT.md"
NEURIPS_V18_OCR_PROOF_RUN = "ocr-proof-v1.8-neurips2025-20260430"
ICLR2025_V18_OCR_PROOF_RUN = "ocr-proof-v1.8-iclr2025-20260430"
NEURIPS_PRESENTATION_DOC_PATHS = (
    "conferences/neurips2025/decks/Execute_Presentation_Plan.md",
    "conferences/neurips2025/decks/Presentation_building_plan.md",
    "conferences/neurips2025/decks/SLIDE_MAPPING.md",
)
PRESENTATION_CAPABILITY_BOUNDARY_SENTENCE = (
    "Presentation production is not yet `conference-general`: current "
    "reusable support covers `Deck command construction` plus "
    "manifest-backed presentation stage registration and receipts, "
    "workspace-aware command planning, safe dry-run script contracts, "
    "structural validation, and receipt-backed Phase 26 proof reports; live "
    "cross-conference production still requires conference-local proof."
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.repo_data("CLAUDE.md")
def test_root_agent_docs_are_universal_and_delegate_conference_context():
    claude = read(CLAUDE)
    agents = read(AGENTS)

    assert "docs/source-of-truth.md" in claude
    assert "docs/reference/pipeline-capabilities.md" in claude
    assert ".planning/milestones/v1.0-NeurIPS-2025-CONTEXT.md" in claude
    assert ".planning/milestones/v1.2-ICLR-2025-CONTEXT.md" in claude
    assert "read `CLAUDE.md`" in agents
    assert ".planning/milestones/" in agents

    forbidden_root_inventory = (
        "01 - Tools and Open Source",
        "02 - GenAI Architectures",
        "Total Papers at NeurIPS 2025",
        "YoY Attendance Growth",
    )
    for phrase in forbidden_root_inventory:
        assert phrase not in claude


def test_source_of_truth_links_capabilities_and_playbook():
    source = read(SOURCE_OF_TRUTH)
    capabilities = read(CAPABILITIES)
    playbook = read(PLAYBOOK)

    assert "conference_milestone_name" in source
    assert "{platform_version}-{conference name}" in source
    assert "v1.0-NeurIPS 2025" in source
    assert "v1.2-ICLR 2025" in source
    assert "conference-smoketest" in capabilities
    assert "small-neurips-smoketest" in capabilities
    assert "conference-smoketest-report.json" in playbook


@pytest.mark.repo_data(".planning/milestones/v1.0-NeurIPS-2025-CONTEXT.md")
def test_conference_context_packets_declare_milestone_identity():
    contexts = [
        NEURIPS_V1_CONTEXT,
        REPO_ROOT / ".planning" / "milestones" / "v1.2-ICLR-2025-CONTEXT.md",
    ]

    pattern = re.compile(r'conference_milestone_name:\s+"v\d+\.\d+-.+"')
    for context_path in contexts:
        content = read(context_path)
        assert pattern.search(content), context_path
        assert "proof_status:" in content
        assert "planning_owner:" in content

    assert "v1.0-NeurIPS 2025" in read(contexts[0])
    assert "v1.2-ICLR 2025" in read(contexts[1])

    assert not (REPO_ROOT / "conferences" / "neurips2025" / "GSD-CONTEXT.md").exists()
    assert not (REPO_ROOT / "conferences" / "iclr2025" / "GSD-CONTEXT.md").exists()


@pytest.mark.repo_data(".planning/MILESTONES.md")
def test_v1_milestone_display_identity_is_retroactive_without_path_rename():
    content = read(MILESTONES)

    assert "v1.0-NeurIPS 2025" in content
    assert "milestones/v1.0-ROADMAP.md" in content
    assert "milestones/v1.0-REQUIREMENTS.md" in content
    assert "milestones/v1.0-MILESTONE-AUDIT.md" in content


@pytest.mark.repo_data(".planning/milestones/v1.0-NeurIPS-2025-CONTEXT.md")
def test_neurips_context_links_historical_presentation_docs():
    content = read(NEURIPS_V1_CONTEXT)

    for doc_path in NEURIPS_PRESENTATION_DOC_PATHS:
        assert doc_path in content
        assert (REPO_ROOT / doc_path).exists(), doc_path

    assert "historical local presentation-production expertise" in content
    assert "not reusable platform workflow support" in content


def test_capability_catalog_bounds_presentation_production():
    content = read(CAPABILITIES)

    for expected in (
        "Deck command construction",
            "| Presentation production | `workspace-local`;",
            "conferences/neurips2025/decks/",
            "scripts/presentations/",
            "manifest-backed presentation stage registration and receipts",
            "proof receipts",
            PRESENTATION_CAPABILITY_BOUNDARY_SENTENCE,
        ):
        assert expected in content


@pytest.mark.repo_data(".planning/milestones/v1.0-NeurIPS-2025-CONTEXT.md")
def test_conference_contexts_record_v18_ocr_proof_status() -> None:
    contexts_and_runs = (
        (NEURIPS_V1_CONTEXT, NEURIPS_V18_OCR_PROOF_RUN),
        (ICLR2025_CONTEXT, ICLR2025_V18_OCR_PROOF_RUN),
    )

    for context_path, run_id in contexts_and_runs:
        content = read(context_path)
        assert "v1.8 OCR Proof Receipts" in content
        assert run_id in content

    neurips_summary = (
        REPO_ROOT
        / "conferences"
        / "neurips2025"
        / "storage"
        / "workflow-runs"
        / NEURIPS_V18_OCR_PROOF_RUN
        / "summary.json"
    )
    iclr2025_summary = (
        REPO_ROOT
        / "conferences"
        / "iclr2025"
        / "storage"
        / "workflow-runs"
        / ICLR2025_V18_OCR_PROOF_RUN
        / "summary.json"
    )
    assert neurips_summary.exists()
    assert iclr2025_summary.exists()

    assert "Deferred." in read(NEURIPS_V1_CONTEXT)
    assert "required-media-oral-31735.json" in read(ICLR2025_CONTEXT)
    assert "skipped_with_rationale-openreview-2Chkk5Ye2s.json" in read(ICLR2025_CONTEXT)
    assert "deferred_blocked-media-oral-31803.json" in read(ICLR2025_CONTEXT)
