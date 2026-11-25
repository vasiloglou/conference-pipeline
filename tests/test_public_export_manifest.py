"""
Tests for the public export manifest schema (EXPORT-01) and the additive
storage_manifest.schema.json extension (D-12).

Phase 05-01: schema validation, required-field enforcement, redistribution_status
enum constraints, pilot manifest conformance, and storage schema regression.
"""

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).parent.parent
EXPORT_SCHEMA_PATH = REPO_ROOT / "docs" / "public_export_manifest.schema.json"
STORAGE_SCHEMA_PATH = REPO_ROOT / "docs" / "storage_manifest.schema.json"
NEURIPS_EXPORT_MANIFEST_PATH = (
    REPO_ROOT / "conferences" / "neurips2025" / "storage" / "public_export_manifest.json"
)
NEURIPS_STORAGE_MANIFEST_PATH = (
    REPO_ROOT / "conferences" / "neurips2025" / "storage" / "storage_manifest.json"
)
ICLR_EXPORT_MANIFEST_PATH = (
    REPO_ROOT / "conferences" / "iclr2026" / "storage" / "public_export_manifest.json"
)
ICLR_STORAGE_MANIFEST_PATH = (
    REPO_ROOT / "conferences" / "iclr2026" / "storage" / "storage_manifest.json"
)
ICLR_DRY_RUN_RECEIPT_PATH = (
    REPO_ROOT
    / "conferences"
    / "iclr2026"
    / "storage"
    / "workflow-runs"
    / "public-export-iclr2026-v1.0.0-dry-run-receipt.json"
)
ICLR_VALIDATION_RECEIPT_PATH = (
    REPO_ROOT
    / "conferences"
    / "iclr2026"
    / "storage"
    / "workflow-runs"
    / "public-export-iclr2026-v1.0.0-validation-receipt.json"
)
ICLR_GAP_CLOSURE_VALIDATION_RECEIPT_PATH = (
    REPO_ROOT
    / "conferences"
    / "iclr2026"
    / "storage"
    / "workflow-runs"
    / "public-release-iclr2026-v1.0.0-gap-closure-validation-receipt.json"
)
ICLR_GAP_CLOSURE_VALIDATION_JOURNAL_PATH = (
    REPO_ROOT
    / "conferences"
    / "iclr2026"
    / "storage"
    / "workflow-runs"
    / "public-release-iclr2026-v1.0.0-gap-closure-validation-receipt.journal.json"
)
ICLR_GAP_CLOSURE_REBUILD_CHILD_PATH = (
    REPO_ROOT
    / "conferences"
    / "iclr2026"
    / "storage"
    / "workflow-runs"
    / "public-export-iclr2026-v1.0.0-gap-closure-rebuild-receipt.json"
)
ICLR_GAP_CLOSURE_REBUILD_RECEIPT_PATH = (
    REPO_ROOT
    / "conferences"
    / "iclr2026"
    / "storage"
    / "workflow-runs"
    / "public-release-iclr2026-v1.0.0-gap-closure-rebuild-receipt.json"
)
ICLR_GAP_CLOSURE_REBUILD_JOURNAL_PATH = (
    REPO_ROOT
    / "conferences"
    / "iclr2026"
    / "storage"
    / "workflow-runs"
    / "public-release-iclr2026-v1.0.0-gap-closure-rebuild-receipt.journal.json"
)
ICLR_BUNDLE_ROOT = (
    REPO_ROOT
    / "conferences"
    / "iclr2026"
    / "exports"
    / "public-release"
    / "iclr2026-v1.0.0"
)
ICLR_BUNDLE_PROOF_PATH = (
    REPO_ROOT
    / ".planning"
    / "milestones"
    / "v3.1-phases"
    / "31-iclr-public-export-bundle"
    / "31-BUNDLE-PROOF.md"
)
ICLR_MILESTONE_CONTEXT_PATH = (
    REPO_ROOT
    / ".planning"
    / "milestones"
    / "v3.1-ICLR 2026 Public Publishing-CONTEXT.md"
)


def _receipts_were_written_from_this_checkout() -> bool:
    """The ICLR release receipts record absolute paths of the checkout that
    produced them. Tests that compare those paths to REPO_ROOT are only
    meaningful in that checkout; in a sibling worktree every path differs by
    the worktree suffix and the comparison says nothing about the receipts."""
    try:
        text = ICLR_DRY_RUN_RECEIPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return False
    return REPO_ROOT.resolve().as_posix() in text


_requires_producing_checkout = pytest.mark.skipif(
    not _receipts_were_written_from_this_checkout(),
    reason="ICLR release receipts record another checkout's absolute paths",
)
def _bundle_unreadable() -> bool:
    """``exports`` is a symlink into the Google Drive backup leaf. A sandboxed
    run (macOS TCC on ~/Library/CloudStorage) can stat the bundle but not read
    it, and ``rglob`` then silently finds no files."""
    try:
        (ICLR_BUNDLE_ROOT / "RELEASE.json").read_bytes()
    except PermissionError:
        return True
    except OSError:
        return False
    return False


_requires_readable_bundle = pytest.mark.skipif(
    _bundle_unreadable(),
    reason="ICLR 2026 public-release bundle is a Drive symlink this environment cannot read (PermissionError)",
)
_requires_bundle_proof = pytest.mark.skipif(
    not ICLR_BUNDLE_PROOF_PATH.exists(),
    reason="31-BUNDLE-PROOF.md is not present in this checkout",
)


@pytest.fixture(scope="module")
def export_schema():
    return json.loads(EXPORT_SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def storage_schema():
    return json.loads(STORAGE_SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def neurips_export_manifest():
    return json.loads(NEURIPS_EXPORT_MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def neurips_storage_manifest():
    return json.loads(NEURIPS_STORAGE_MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def iclr_export_manifest():
    return json.loads(ICLR_EXPORT_MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def iclr_storage_manifest():
    return json.loads(ICLR_STORAGE_MANIFEST_PATH.read_text())


def _minimal_valid_family(**overrides):
    base = {
        "id": "test_family",
        "private_path": "some/path",
        "public_path": "exports/public-release/{bundle}/test/",
        "storage_manifest_id": "test_family",
        "redistribution_status": "allowed",
        "license": "CC-BY-4.0",
        "provenance": "pipeline-generated",
    }
    base.update(overrides)
    return base


def _minimal_valid_manifest(**overrides):
    base = {
        "schema_version": "1.0.0",
        "conference": "test2025",
        "release": {"source_sha": ""},
        "families": [_minimal_valid_family()],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Task 1 tests: public_export_manifest.schema.json
# ---------------------------------------------------------------------------


def test_schema_is_valid_json_schema(export_schema):
    """Schema must pass jsonschema meta-validation (draft-07)."""
    meta_schema_url = "http://json-schema.org/draft-07/schema#"
    meta_validator = jsonschema.Draft7Validator(
        jsonschema.Draft7Validator.META_SCHEMA
    )
    errors = list(meta_validator.iter_errors(export_schema))
    assert not errors, f"Schema failed meta-validation: {errors}"


def test_schema_accepts_minimal_valid_manifest(export_schema):
    manifest = _minimal_valid_manifest()
    jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_requires_schema_version(export_schema):
    manifest = _minimal_valid_manifest()
    del manifest["schema_version"]
    with pytest.raises(jsonschema.ValidationError, match="schema_version"):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_requires_conference(export_schema):
    manifest = _minimal_valid_manifest()
    del manifest["conference"]
    with pytest.raises(jsonschema.ValidationError, match="conference"):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_requires_release(export_schema):
    manifest = _minimal_valid_manifest()
    del manifest["release"]
    with pytest.raises(jsonschema.ValidationError, match="release"):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_requires_release_source_sha(export_schema):
    manifest = _minimal_valid_manifest()
    manifest["release"] = {}
    with pytest.raises(jsonschema.ValidationError, match="source_sha"):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_rejects_missing_required_safety_fields(export_schema):
    """A family missing redistribution_status must fail validation."""
    family = _minimal_valid_family()
    del family["redistribution_status"]
    manifest = _minimal_valid_manifest(families=[family])
    with pytest.raises(jsonschema.ValidationError, match="redistribution_status"):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_rejects_missing_license(export_schema):
    family = _minimal_valid_family()
    del family["license"]
    manifest = _minimal_valid_manifest(families=[family])
    with pytest.raises(jsonschema.ValidationError, match="license"):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_rejects_missing_provenance(export_schema):
    family = _minimal_valid_family()
    del family["provenance"]
    manifest = _minimal_valid_manifest(families=[family])
    with pytest.raises(jsonschema.ValidationError, match="provenance"):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_rejects_missing_storage_manifest_id(export_schema):
    family = _minimal_valid_family()
    del family["storage_manifest_id"]
    manifest = _minimal_valid_manifest(families=[family])
    with pytest.raises(jsonschema.ValidationError, match="storage_manifest_id"):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_rejects_invalid_redistribution_status(export_schema):
    """redistribution_status: 'unknown' is not in the enum and must fail."""
    family = _minimal_valid_family(redistribution_status="unknown")
    manifest = _minimal_valid_manifest(families=[family])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_rejects_invalid_provenance(export_schema):
    family = _minimal_valid_family(provenance="not-a-real-provenance")
    manifest = _minimal_valid_manifest(families=[family])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_accepts_all_redistribution_status_values(export_schema):
    for status in ("allowed", "caution", "prohibited"):
        family = _minimal_valid_family(redistribution_status=status)
        manifest = _minimal_valid_manifest(families=[family])
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_optional_fields_may_be_absent(export_schema):
    """All optional fields absent — manifest must still be valid."""
    manifest = _minimal_valid_manifest()
    jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_accepts_optional_fields_present(export_schema):
    family = _minimal_valid_family(
        redaction_rules=["strip_internal_fields"],
        large_media_policy={"delivery": "cdn-only"},
        format_contract={"type": "pdf"},
        include_globs=["*.pdf"],
        exclude_globs=["*_draft*"],
        cdn_base_url="https://cdn.example.com/",
        checksum_required=True,
        public_export_notes="Some operator notes.",
        glob_overrides=[
            {
                "glob": "*.raw",
                "redistribution_status": "prohibited",
                "license": "proprietary",
            }
        ],
    )
    manifest = _minimal_valid_manifest(families=[family])
    jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_rejects_extra_top_level_properties(export_schema):
    manifest = _minimal_valid_manifest()
    manifest["unexpected_key"] = "should fail"
    with pytest.raises(jsonschema.ValidationError, match="Additional properties"):
        jsonschema.validate(instance=manifest, schema=export_schema)


def test_schema_rejects_extra_family_properties(export_schema):
    family = _minimal_valid_family()
    family["unknown_field"] = "should fail"
    manifest = _minimal_valid_manifest(families=[family])
    with pytest.raises(jsonschema.ValidationError, match="Additional properties"):
        jsonschema.validate(instance=manifest, schema=export_schema)


# ---------------------------------------------------------------------------
# Task 2 tests: storage_manifest.schema.json additive extension
# ---------------------------------------------------------------------------


def _minimal_storage_family(**overrides):
    base = {
        "id": "test",
        "path": "some/path",
        "purpose": "test purpose",
        "producer": "test-producer",
        "storage_class": "backup",
        "share_policy": "private",
        "rebuildable": True,
        "sync_profiles": {"public": False, "backup": True},
    }
    base.update(overrides)
    return base


def _minimal_storage_manifest(**overrides):
    base = {
        "schema_version": 1,
        "workspace": "conferences/test2025",
        "artifact_families": [_minimal_storage_family()],
    }
    base.update(overrides)
    return base


def test_storage_manifest_schema_accepts_public_export_field(storage_schema):
    """A family with sync_profiles.public_export: true must pass schema validation."""
    family = _minimal_storage_family(
        sync_profiles={"public": True, "backup": True, "public_export": True}
    )
    manifest = _minimal_storage_manifest(artifact_families=[family])
    jsonschema.validate(instance=manifest, schema=storage_schema)


def test_storage_manifest_schema_accepts_public_export_false(storage_schema):
    family = _minimal_storage_family(
        sync_profiles={"public": False, "backup": False, "public_export": False}
    )
    manifest = _minimal_storage_manifest(artifact_families=[family])
    jsonschema.validate(instance=manifest, schema=storage_schema)


def test_storage_manifest_schema_field_optional_when_absent(storage_schema):
    """A family without sync_profiles.public_export must still pass (field is optional)."""
    family = _minimal_storage_family()
    assert "public_export" not in family["sync_profiles"]
    manifest = _minimal_storage_manifest(artifact_families=[family])
    jsonschema.validate(instance=manifest, schema=storage_schema)


def test_storage_manifest_schema_accepts_public_export_notes(storage_schema):
    family = _minimal_storage_family(public_export_notes="Operator notes here.")
    manifest = _minimal_storage_manifest(artifact_families=[family])
    jsonschema.validate(instance=manifest, schema=storage_schema)


@pytest.mark.repo_data("conferences/neurips2025/storage/storage_manifest.json")
def test_storage_manifest_schema_still_validates_existing_neurips_manifest(
    storage_schema, neurips_storage_manifest
):
    """The existing neurips2025 storage_manifest.json must still validate after the additive extension."""
    jsonschema.validate(instance=neurips_storage_manifest, schema=storage_schema)


# ---------------------------------------------------------------------------
# Task 3 tests: NeurIPS 2025 pilot public_export_manifest.json
# ---------------------------------------------------------------------------

# NeurIPS 2025 v1.1.0, as the owner ruled it in Phase 6b (D25: text decks, figures named,
# not copied). The Phase 1 pilot's families (prebuilt index, podcasts, executive maps,
# payloads) are not in this release.
REQUIRED_FAMILY_IDS = {
    "reader_catalog",
    "reader_text",
    "transcripts",
    "presentation_markdown",
    "presentation_manifest",
}

REQUIRED_SAFETY_FIELDS = {"redistribution_status", "license", "provenance", "storage_manifest_id"}


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_neurips_pilot_manifest_validates_against_schema(
    export_schema, neurips_export_manifest
):
    """Pilot manifest must validate against docs/public_export_manifest.schema.json without errors."""
    jsonschema.validate(instance=neurips_export_manifest, schema=export_schema)


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_neurips_pilot_manifest_has_required_family_ids(neurips_export_manifest):
    actual_ids = {f["id"] for f in neurips_export_manifest["families"]}
    assert actual_ids == REQUIRED_FAMILY_IDS


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_neurips_pilot_manifest_has_required_safety_fields_on_all_families(
    neurips_export_manifest,
):
    """Every family must carry all required safety/identity fields."""
    for family in neurips_export_manifest["families"]:
        missing = REQUIRED_SAFETY_FIELDS - set(family.keys())
        assert not missing, (
            f"Family '{family.get('id', '?')}' is missing required safety fields: {missing}"
        )


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_neurips_pilot_manifest_has_source_sha_field(neurips_export_manifest):
    """release.source_sha must be present (may be empty string placeholder)."""
    assert "source_sha" in neurips_export_manifest["release"]


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_transcripts_family_is_caution_not_allowed(neurips_export_manifest):
    """Transcripts family must have redistribution_status=caution per D-09."""
    families_by_id = {f["id"]: f for f in neurips_export_manifest["families"]}
    assert "transcripts" in families_by_id, "transcripts family is missing from pilot manifest"
    transcripts = families_by_id["transcripts"]
    assert transcripts["redistribution_status"] == "caution", (
        f"Expected transcripts.redistribution_status='caution', got '{transcripts['redistribution_status']}'"
    )


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_transcripts_family_license_is_nc_nd(neurips_export_manifest):
    families_by_id = {f["id"]: f for f in neurips_export_manifest["families"]}
    transcripts = families_by_id["transcripts"]
    assert "NC" in transcripts["license"] or "nc" in transcripts["license"].lower(), (
        f"Expected NC license for transcripts, got '{transcripts['license']}'"
    )


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_transcripts_family_provenance_is_talk_derived(neurips_export_manifest):
    families_by_id = {f["id"]: f for f in neurips_export_manifest["families"]}
    transcripts = families_by_id["transcripts"]
    assert transcripts["provenance"] == "talk-derived"


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_no_family_ships_a_prebuilt_index(neurips_export_manifest):
    """Since Phase 6b the release build indexes the bundle it writes, so no family exports a
    prebuilt index, and none reads the old public_index_staging family (was D-11)."""
    for family in neurips_export_manifest["families"]:
        assert family["id"] != "prebuilt_index"
        assert family["storage_manifest_id"] != "public_index_staging"


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_no_family_missing_storage_manifest_id(neurips_export_manifest):
    for family in neurips_export_manifest["families"]:
        assert "storage_manifest_id" in family and family["storage_manifest_id"], (
            f"Family '{family.get('id', '?')}' is missing or has empty storage_manifest_id"
        )


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_schema_version_is_present_and_semver(neurips_export_manifest):
    version = neurips_export_manifest.get("schema_version", "")
    parts = version.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts), (
        f"schema_version '{version}' is not a valid semver string"
    )


# ---------------------------------------------------------------------------
# Phase 30 tests: ICLR 2026 public release manifest safety boundary
# ---------------------------------------------------------------------------

# The families iclr2026-v1.0.0 shipped; its dry-run receipt (below) records them.
ICLR_REQUIRED_FAMILY_IDS = {
    "presentation_markdown",
    "presentation_manifest",
    "presentation_frame_contracts",
    "executive_summary_deck_first_plan",
    "executive_source_link_map",
    "executive_presentation_link_map",
    "presentation_payloads",
    "podcast_payloads",
}

# iclr2026-v1.1.0, as the owner ruled it in Phase 6b (families 1-9; deck review summaries
# and executive control files dropped).
ICLR_V110_FAMILY_IDS = {
    "reader_catalog",
    "reader_text",
    "transcripts",
    "presentation_markdown",
    "presentation_payloads",
    "presentation_pdfs",
    "presentation_control",
    "presentation_manifest",
    "podcasts",
}

# Families whose bytes `release families` writes into the private release-staging folder;
# every other family reads a curated, public-synced storage family.
RELEASE_STAGING_ID = "release_staging"

ICLR_PROHIBITED_STORAGE_IDS = {
    "workshop_talk_media_outputs",
    "workshop_talk_media_inventory",
    "download_corpus_inventory",
    "download_corpus_logs",
    "live_workflow_logs",
    "live_provider_call_receipts",
    "workspace_provider_call_receipts",
    "workflow_run_provider_call_receipts",
    "workspace_logs",
    "paper_matching_outputs",
    "presentation_redesign_snapshots",
    "quality_audit_payloads",
    "transcription_ocr_outputs",
    "corpus_inventory_index_outputs",
    "corpus_vector_store_outputs",
    "acquisition_media_assets",
    "analysis",
    "presentation_video_workdirs",
    "presentation_orchestrator_logs",
    "presentation_generated_figure_code",
    "exports",
    "agent_work",
    "cache",
    "legacy",
}

ICLR_PROHIBITED_PATH_FRAGMENTS = {
    "storage/workshops/talk-media",
    "storage/workshops/talk_media_inventory.json",
    "storage/download-corpus",
    "storage/workflow-runs/iclr2026-live-production/logs",
    "storage/workflow-runs/iclr2026-live-production/provider-calls",
    "storage/provider-calls",
    "storage/logs",
    "storage/paper-matching",
    "storage/presentation-redesign",
    "storage/quality-audits",
    "storage/transcription-ocr",
    "storage/corpus",
    "storage/vector-store",
    "cache/acquisition-downloads/media_asset",
    "analysis",
    "video_build",
    "orchestrator_logs",
    "assets/generated/qc",
    "assets/generated/code",
    "agent-work",
    "cache/",
    "legacy",
}

CAUTION_FAMILY_HINTS = ("analysis", "transcript")


def _storage_families_by_id(storage_manifest):
    return {family["id"]: family for family in storage_manifest["artifact_families"]}


@pytest.mark.repo_data("conferences/iclr2026/storage/public_export_manifest.json")
def test_iclr_manifest_validates_against_schema(export_schema, iclr_export_manifest):
    jsonschema.validate(instance=iclr_export_manifest, schema=export_schema)
    assert iclr_export_manifest["conference"] == "iclr2026"
    assert "source_sha" in iclr_export_manifest["release"]


@pytest.mark.repo_data("conferences/iclr2026/storage/public_export_manifest.json")
def test_iclr_manifest_locks_phase31_cdn_base_and_threshold(iclr_export_manifest):
    expected_family_ids = ICLR_V110_FAMILY_IDS
    expected_cdn_base = (
        "https://pub-0c8355dc0a634ac6985680523ed7f40f.r2.dev/"
        "conferences/iclr2026"
    )
    families = {
        family["id"]: family for family in iclr_export_manifest["families"]
    }

    assert iclr_export_manifest["release"]["source_sha"] == (
        "3a155500a63b4e6094fb28eff3ffd72a4573703e"
    )
    assert set(families) == expected_family_ids
    assert {
        family_id
        for family_id, family in families.items()
        if family.get("cdn_base_url")
    } == {"presentation_payloads", "podcasts"}
    for family_id in ("presentation_payloads", "podcasts"):
        family = families[family_id]
        assert family["cdn_base_url"] == expected_cdn_base
        assert family["large_media_policy"]["delivery"] == "inline-if-small"
        assert family["large_media_policy"]["max_inline_size_mb"] == 20


@pytest.mark.repo_data("conferences/iclr2026/storage/public_export_manifest.json")
def test_iclr_manifest_has_required_public_scope_families(iclr_export_manifest):
    actual_ids = {family["id"] for family in iclr_export_manifest["families"]}
    missing = ICLR_V110_FAMILY_IDS - actual_ids
    assert not missing, f"ICLR public manifest missing expected generated families: {missing}"
    assert "workshop_paper_payloads" not in actual_ids


@pytest.mark.repo_data("conferences/iclr2026/storage/public_export_manifest.json")
def test_iclr_manifest_has_required_safety_fields_on_all_families(iclr_export_manifest):
    for family in iclr_export_manifest["families"]:
        missing = REQUIRED_SAFETY_FIELDS - set(family.keys())
        assert not missing, (
            f"Family '{family.get('id', '?')}' is missing required safety fields: {missing}"
        )
        assert family["public_path"].startswith("exports/public-release/{bundle}/")


@pytest.mark.repo_data("conferences/iclr2026/storage/public_export_manifest.json")
def test_iclr_manifest_references_public_or_curated_storage_families(
    iclr_export_manifest, iclr_storage_manifest
):
    storage_families = _storage_families_by_id(iclr_storage_manifest)
    for family in iclr_export_manifest["families"]:
        storage_id = family["storage_manifest_id"]
        assert storage_id in storage_families, f"Unknown storage manifest id: {storage_id}"
        storage_family = storage_families[storage_id]
        if storage_id == RELEASE_STAGING_ID:
            # Generated for this export by `release families`; private in storage, published
            # only through the bundle the release check passes.
            assert family["private_path"].startswith("storage/release-staging/")
            continue
        assert storage_family.get("share_policy") == "curated"
        assert storage_family.get("sync_profiles", {}).get("public") is True


@pytest.mark.repo_data("conferences/iclr2026/storage/public_export_manifest.json")
def test_iclr_analysis_and_transcripts_are_excluded_or_caution_gated(iclr_export_manifest):
    caution_candidates = [
        family
        for family in iclr_export_manifest["families"]
        if any(hint in f"{family['id']} {family['private_path']}".lower() for hint in CAUTION_FAMILY_HINTS)
    ]
    for family in caution_candidates:
        assert family["redistribution_status"] == "caution"
        assert family["license"]
        assert family["provenance"] in {"pipeline-generated", "talk-derived"}
        notes = family.get("public_export_notes", "")
        assert "ack-caution-family" in notes or "acknowledgement is required on each export run" in notes


@pytest.mark.repo_data("conferences/iclr2026/storage/public_export_manifest.json")
def test_iclr_manifest_excludes_d06_private_source_gated_storage_ids(iclr_export_manifest):
    actual_storage_ids = {family["storage_manifest_id"] for family in iclr_export_manifest["families"]}
    actual_family_ids = {family["id"] for family in iclr_export_manifest["families"]}
    assert not (actual_storage_ids & ICLR_PROHIBITED_STORAGE_IDS)
    assert not (actual_family_ids & ICLR_PROHIBITED_STORAGE_IDS)


@pytest.mark.repo_data("conferences/iclr2026/storage/public_export_manifest.json")
def test_iclr_manifest_excludes_d06_private_source_gated_path_fragments(iclr_export_manifest):
    combined_paths = "\n".join(
        "\n".join(
            [
                family["private_path"],
                family["public_path"],
                *family.get("include_globs", []),
            ]
        )
        for family in iclr_export_manifest["families"]
    )
    leaked = sorted(fragment for fragment in ICLR_PROHIBITED_PATH_FRAGMENTS if fragment in combined_paths)
    assert not leaked, f"ICLR public manifest includes prohibited private/source-gated paths: {leaked}"


@pytest.mark.repo_data("conferences/iclr2026/storage/public_export_manifest.json")
def test_iclr_presentation_markdown_scope_excludes_prompts_and_research_plans(iclr_export_manifest):
    family = next(family for family in iclr_export_manifest["families"] if family["id"] == "presentation_markdown")
    include_globs = set(family.get("include_globs", []))
    exclude_globs = set(family.get("exclude_globs", []))

    import fnmatch

    assert "**/*.md" not in include_globs
    assert "**/*prompt*" in exclude_globs or "**/*prompt*.md" in exclude_globs
    # Matched as scripts/export_for_public.py matches them (fnmatch on the relative path).
    for path in ("01_topic/narration_prompt.md", "01_topic/key_event_synthesis/plan.md",
                 "01_topic/research_cluster_plan.md", "01_topic/presentation_review.md"):
        included = any(fnmatch.fnmatch(path, g) for g in include_globs)
        excluded = any(fnmatch.fnmatch(path, g) for g in exclude_globs)
        assert not included or excluded, path


@pytest.mark.repo_data("conferences/iclr2026/storage/workflow-runs/public-export-iclr2026-v1.0.0-dry-run-receipt.json")
def test_iclr_dry_run_receipt_excludes_prompt_and_research_markdown():
    receipt = json.loads(ICLR_DRY_RUN_RECEIPT_PATH.read_text(encoding="utf-8"))
    planned_paths = [artifact["path"] for artifact in receipt.get("planned_artifacts", [])]
    leaked = [
        path
        for path in planned_paths
        if "prompt" in path.lower()
        or "key_event_synthesis" in path
        or path.endswith("research_cluster_plan.md")
    ]

    assert not leaked, f"ICLR dry-run receipt planned unsafe Markdown artifacts: {leaked[:10]}"


@pytest.mark.repo_data("conferences/iclr2026/storage/workflow-runs/public-export-iclr2026-v1.0.0-dry-run-receipt.json")
def test_iclr_dry_run_receipt_excludes_private_payload_fragments():
    receipt = json.loads(ICLR_DRY_RUN_RECEIPT_PATH.read_text(encoding="utf-8"))
    planned_paths = [artifact["path"] for artifact in receipt.get("planned_artifacts", [])]
    leaked = [
        path
        for path in planned_paths
        for fragment in ICLR_PROHIBITED_PATH_FRAGMENTS
        if fragment in path
    ]

    assert not leaked, f"ICLR dry-run receipt planned prohibited private payloads: {leaked[:10]}"


@_requires_producing_checkout
def test_iclr_gap_closure_validation_receipt_has_finalized_journal():
    workflow_root = (
        REPO_ROOT / "conferences" / "iclr2026" / "storage" / "workflow-runs"
    )
    child_paths = [ICLR_VALIDATION_RECEIPT_PATH, ICLR_DRY_RUN_RECEIPT_PATH]
    evidence_paths = [
        *child_paths,
        ICLR_GAP_CLOSURE_VALIDATION_RECEIPT_PATH,
        ICLR_GAP_CLOSURE_VALIDATION_JOURNAL_PATH,
    ]
    for path in evidence_paths:
        assert path.is_file()
        assert not path.is_symlink()
        assert path.resolve().is_relative_to(workflow_root.resolve())

    top = json.loads(
        ICLR_GAP_CLOSURE_VALIDATION_RECEIPT_PATH.read_text(encoding="utf-8")
    )
    journal = json.loads(
        ICLR_GAP_CLOSURE_VALIDATION_JOURNAL_PATH.read_text(encoding="utf-8")
    )
    dry_run = json.loads(ICLR_DRY_RUN_RECEIPT_PATH.read_text(encoding="utf-8"))

    child_receipt_paths = [path.as_posix() for path in child_paths]
    expected_results = [
        {"stage": "validate_manifest", "exit_code": 0, "receipt_path": child_receipt_paths[0]},
        {"stage": "export_dry_run", "exit_code": 0, "receipt_path": child_receipt_paths[1]},
    ]
    expected_commands = [
        {
            "stage": result["stage"],
            "receipt_path": result["receipt_path"],
            "live": False,
            "shell": False,
        }
        for result in expected_results
    ]

    assert top["receipt_type"] == "conference_pipeline.public_release"
    assert top["status"] == "succeeded"
    assert top["exit_code"] == 0
    assert top["child_receipt_paths"] == child_receipt_paths
    assert top["command_results"] == expected_results
    assert [command["stage"] for command in top["commands"]] == [
        "validate_manifest",
        "export_dry_run",
    ]
    assert all(command["shell"] is False and command["live"] is False for command in top["commands"])
    assert top["blocked_stages"] == []
    assert top["env_loaded_for_live_stages"] is False
    assert top["loaded_env_var_names"] == []
    assert top["secret_values_recorded"] is False
    assert top["journal_path"] == ICLR_GAP_CLOSURE_VALIDATION_JOURNAL_PATH.as_posix()

    assert journal["receipt_type"] == "conference_pipeline.public_release_journal"
    assert journal["status"] == "finalized"
    assert journal["top_level_receipt_path"] == ICLR_GAP_CLOSURE_VALIDATION_RECEIPT_PATH.as_posix()
    assert journal["planned_commands"] == expected_commands
    assert journal["command_results"] == expected_results
    assert journal["last_completed_stage"] == "export_dry_run"
    assert journal["failure_code"] is None
    assert journal["exception_type"] is None
    assert journal["secret_values_recorded"] is False

    assert dry_run["receipt_type"] == "conference_pipeline.public_export"
    assert dry_run["status"] == "succeeded"
    assert dry_run["exit_code"] == 0
    assert dry_run["dry_run"] is True
    assert dry_run["conference"] == "iclr2026"
    assert dry_run["bundle_id"] == "iclr2026-v1.0.0"
    assert dry_run["source_sha"] == "3a155500a63b4e6094fb28eff3ffd72a4573703e"
    assert dry_run["export_manifest_sha256"] == hashlib.sha256(
        ICLR_EXPORT_MANIFEST_PATH.read_bytes()
    ).hexdigest()
    assert dry_run["included_families"] == sorted(ICLR_REQUIRED_FAMILY_IDS)
    assert dry_run["planned_artifact_count"] == 605
    assert len(dry_run["planned_artifacts"]) == 605
    assert sum(
        artifact["delivery"] == "inline" for artifact in dry_run["planned_artifacts"]
    ) == 577
    assert sum(
        artifact["delivery"] == "cdn" for artifact in dry_run["planned_artifacts"]
    ) == 28
    assert len(dry_run["cdn_stubs"]) == 28
    assert len(dry_run["excluded_artifacts"]) == 13082
    assert dry_run["redacted_files"] == []
    assert dry_run["license_gating"]["caution_families"] == []
    gate_receipts = dry_run["license_gating"]["gate_receipts"]
    assert {gate_id: receipt["status"] for gate_id, receipt in gate_receipts.items()} == {
        "redistribution": "passed",
        "secrets": "passed",
    }
    assert dry_run["secret_values_recorded"] is False
    serialized_evidence = json.dumps([top, journal, dry_run], sort_keys=True)
    assert "GoogleDrive" not in serialized_evidence
    assert "My Drive" not in serialized_evidence


@_requires_producing_checkout
@_requires_readable_bundle
def test_iclr_gap_closure_rebuild_receipts_prove_current_contract():
    workflow_root = (
        REPO_ROOT / "conferences" / "iclr2026" / "storage" / "workflow-runs"
    )
    evidence_paths = [
        ICLR_DRY_RUN_RECEIPT_PATH,
        ICLR_GAP_CLOSURE_REBUILD_CHILD_PATH,
        ICLR_GAP_CLOSURE_REBUILD_RECEIPT_PATH,
        ICLR_GAP_CLOSURE_REBUILD_JOURNAL_PATH,
    ]
    for path in evidence_paths:
        assert path.is_file()
        assert not path.is_symlink()
        assert path.resolve().is_relative_to(workflow_root.resolve())

    dry_run = json.loads(ICLR_DRY_RUN_RECEIPT_PATH.read_text(encoding="utf-8"))
    child = json.loads(
        ICLR_GAP_CLOSURE_REBUILD_CHILD_PATH.read_text(encoding="utf-8")
    )
    top = json.loads(
        ICLR_GAP_CLOSURE_REBUILD_RECEIPT_PATH.read_text(encoding="utf-8")
    )
    journal = json.loads(
        ICLR_GAP_CLOSURE_REBUILD_JOURNAL_PATH.read_text(encoding="utf-8")
    )

    assert child["receipt_type"] == "conference_pipeline.public_export"
    assert child["status"] == "succeeded"
    assert child["exit_code"] == 0
    assert child["dry_run"] is False
    assert child["conference"] == "iclr2026"
    assert child["bundle_id"] == "iclr2026-v1.0.0"
    assert child["source_sha"] == "3a155500a63b4e6094fb28eff3ffd72a4573703e"
    assert child["export_manifest_sha256"] == hashlib.sha256(
        ICLR_EXPORT_MANIFEST_PATH.read_bytes()
    ).hexdigest()
    assert child["planned_artifact_count"] == 605
    assert child["actual_artifact_count"] == 607
    assert child["reviewed_dry_run_receipt_path"] == ICLR_DRY_RUN_RECEIPT_PATH.as_posix()
    assert child["secret_values_recorded"] is False

    rows = child["cdn_upload_sources"]
    assert len(rows) == 28
    assert rows == sorted(
        rows,
        key=lambda row: (
            row["family_id"],
            row["path"],
            row["object_key"],
            row["source_path"],
        ),
    )
    required_row_fields = {
        "family_id",
        "path",
        "object_key",
        "source_path",
        "source_sha256",
        "size_bytes",
    }
    for row in rows:
        assert set(row) == required_row_fields
        source_path = Path(row["source_path"])
        assert not source_path.is_absolute()
        assert ".." not in source_path.parts
        assert len(row["source_sha256"]) == 64
        assert row["size_bytes"] > 0
        assert "GoogleDrive" not in row["source_path"]
        assert "My Drive" not in row["source_path"]
    canonical_rows = json.dumps(
        rows,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert child["cdn_upload_sources_sha256"] == hashlib.sha256(
        canonical_rows.encode("utf-8")
    ).hexdigest()

    dry_cdn_contract = {
        (
            row["family_id"],
            row["path"],
            row["object_key"],
            row["source_sha256"],
            row["size_bytes"],
        )
        for row in dry_run["cdn_stubs"]
    }
    current_cdn_contract = {
        (
            row["family_id"],
            row["path"],
            row["object_key"],
            row["source_sha256"],
            row["size_bytes"],
        )
        for row in rows
    }
    assert current_cdn_contract == dry_cdn_contract

    verification = child["bundle_verification"]
    assert verification == {
        "status": "passed",
        "artifact_count": 607,
        "exact_artifact_set": True,
        "checksums_verified": True,
        "portable_regular_files": True,
        "cdn_sidecars_verified": True,
        "manifest_source_identity_verified": True,
        "safety_gates_verified": True,
        "bundle_tree_sha256": verification["bundle_tree_sha256"],
    }
    assert len(verification["bundle_tree_sha256"]) == 64
    publication = child["publication"]
    assert publication["previous_bundle_present"] is True
    assert publication["published_version_evidence"] == []
    assert publication["rollback_performed"] is False
    assert publication["cleanup_pending"] is False
    assert publication["rollback_authority_retained_through_receipt"] is True
    reproducibility = child["reproducibility"]
    assert reproducibility["source_sha"] == child["source_sha"]
    assert reproducibility["export_manifest_sha256"] == child["export_manifest_sha256"]
    assert reproducibility["prior_bundle_tree_sha256"]
    assert reproducibility["prior_bundle_tree_sha256"] == reproducibility[
        "current_bundle_tree_sha256"
    ]
    assert reproducibility["byte_identical"] is True
    assert reproducibility["current_bundle_tree_sha256"] == verification[
        "bundle_tree_sha256"
    ]

    child_path = ICLR_GAP_CLOSURE_REBUILD_CHILD_PATH.as_posix()
    expected_results = [
        {"stage": "export_write", "exit_code": 0, "receipt_path": child_path}
    ]
    assert top["receipt_type"] == "conference_pipeline.public_release"
    assert top["status"] == "succeeded"
    assert top["exit_code"] == 0
    assert top["child_receipt_paths"] == [child_path]
    assert top["command_results"] == expected_results
    assert len(top["commands"]) == 1
    assert top["commands"][0]["stage"] == "export_write"
    assert top["commands"][0]["shell"] is False
    assert top["commands"][0]["live"] is False
    assert top["env_loaded_for_live_stages"] is False
    assert top["loaded_env_var_names"] == []
    assert top["secret_values_recorded"] is False
    assert top["journal_path"] == ICLR_GAP_CLOSURE_REBUILD_JOURNAL_PATH.as_posix()

    assert journal["receipt_type"] == "conference_pipeline.public_release_journal"
    assert journal["status"] == "finalized"
    assert journal["top_level_receipt_path"] == ICLR_GAP_CLOSURE_REBUILD_RECEIPT_PATH.as_posix()
    assert journal["planned_commands"] == [
        {
            "stage": "export_write",
            "receipt_path": child_path,
            "live": False,
            "shell": False,
        }
    ]
    assert journal["command_results"] == expected_results
    assert journal["last_completed_stage"] == "export_write"
    assert journal["failure_code"] is None
    assert journal["exception_type"] is None
    assert journal["secret_values_recorded"] is False

    assert ICLR_BUNDLE_ROOT.is_dir()
    assert not ICLR_BUNDLE_ROOT.is_symlink()
    bundle_files = sorted(path for path in ICLR_BUNDLE_ROOT.rglob("*") if path.is_file())
    # Phase 31's export contract remains 607 artifacts; Phase 32 appends the
    # public index, site, BYOK, and self-host surfaces to the same bundle.
    assert len(bundle_files) >= 607
    # The staged public-repo mirror duplicates the 28 bundle sidecars.
    assert sum(path.name.endswith(".cdn.json") for path in bundle_files) >= 28
    assert (ICLR_BUNDLE_ROOT / "RELEASE.json").is_file()
    assert (ICLR_BUNDLE_ROOT / "CHECKSUMS.json").is_file()
    assert not any(path.is_symlink() for path in ICLR_BUNDLE_ROOT.rglob("*"))
    assert not list(ICLR_BUNDLE_ROOT.parent.glob(".iclr2026-v1.0.0.staging-*"))
    assert not list(ICLR_BUNDLE_ROOT.parent.glob(".iclr2026-v1.0.0.backup-*"))


@_requires_bundle_proof
def test_iclr_bundle_proof_distinguishes_historical_first_write_from_gap_closure():
    proof = ICLR_BUNDLE_PROOF_PATH.read_text(encoding="utf-8")
    milestone = ICLR_MILESTONE_CONTEXT_PATH.read_text(encoding="utf-8")
    combined = f"{proof}\n{milestone}".lower()

    assert "historical first-write evidence" in combined
    assert "historical pre-gap reproducibility evidence" in combined
    for name in (
        "public-release-iclr2026-v1.0.0-gap-closure-validation-receipt.json",
        "public-release-iclr2026-v1.0.0-gap-closure-validation-receipt.journal.json",
        "public-export-iclr2026-v1.0.0-gap-closure-rebuild-receipt.json",
        "public-release-iclr2026-v1.0.0-gap-closure-rebuild-receipt.json",
        "public-release-iclr2026-v1.0.0-gap-closure-rebuild-receipt.journal.json",
    ):
        assert name.lower() in combined
    # The packet said "dry-run-proven" until the live publication of 2026-07-11
    # (2f1c586fa); this test sat behind a wrong path until v6.0 Phase 3 (IF9).
    assert 'proof_status: "complete"' in milestone
    assert "no upload" in combined
    assert "provider" in combined
    assert "public url liveness" in combined
    assert "public repo" in combined
    assert "public index" in combined
    assert "static site" in combined
    assert "byok" in combined
    assert "metadata restamp" in combined
    assert "closeout" in combined
