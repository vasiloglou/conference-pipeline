from __future__ import annotations
import pytest

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "docs" / "storage_manifest.schema.json"
NEURIPS_MANIFEST_PATH = (
    REPO_ROOT / "conferences" / "neurips2025" / "storage" / "storage_manifest.json"
)
TEMPLATE_MANIFEST_PATH = (
    REPO_ROOT / "conferences" / "TEMPLATE" / "storage" / "storage_manifest.json"
)
INVENTORY_PATH = REPO_ROOT / "conferences" / "neurips2025" / "storage" / "inventory.json"
NEURIPS_CONFIG_PATH = REPO_ROOT / "conferences" / "neurips2025" / "config.yaml"
TEMPLATE_CONFIG_PATH = REPO_ROOT / "conferences" / "TEMPLATE" / "config.yaml"
STARTER_MANIFEST_PATHS = (TEMPLATE_MANIFEST_PATH,)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_schema_declares_expected_contract():
    schema = load_json(SCHEMA_PATH)

    assert schema["$schema"].endswith("draft-07/schema#")
    properties = schema["properties"]
    assert "artifact_families" in properties

    artifact_family = schema["definitions"]["artifactFamily"]["properties"]
    assert artifact_family["storage_class"]["enum"] == ["git", "public", "backup", "cache"]
    assert artifact_family["share_policy"]["enum"] == ["none", "curated", "private"]
    assert set(artifact_family["sync_profiles"]["required"]) == {"public", "backup"}
    assert "tracked_samples" in artifact_family
    assert "gitignore_exceptions" in artifact_family
    assert schema["definitions"]["artifactFamily"]["oneOf"] == [
        {"required": ["path"]},
        {"required": ["path_glob"]},
    ]
    assert "anyOf" not in schema["definitions"]["artifactFamily"]


@pytest.mark.repo_data("conferences/neurips2025/storage/storage_manifest.json")
def test_manifests_have_required_top_level_keys():
    for manifest_path in (NEURIPS_MANIFEST_PATH, TEMPLATE_MANIFEST_PATH):
        manifest = load_json(manifest_path)
        assert manifest["schema_version"] == 1
        assert manifest["workspace"]
        assert manifest["artifact_families"]
        assert "inventory_exemptions" in manifest

        for family in manifest["artifact_families"]:
            assert family["id"]
            assert family["purpose"]
            assert family["producer"]
            assert family["storage_class"] in {"git", "public", "backup", "cache"}
            assert family["share_policy"] in {"none", "curated", "private"}
            assert isinstance(family["rebuildable"], bool)
            sp = family["sync_profiles"]
            assert {"public", "backup"}.issubset(sp), f"sync_profiles missing required keys: {sp}"
            assert isinstance(sp["public"], bool)
            assert isinstance(sp["backup"], bool)
            allowed_sp_keys = {"public", "backup", "public_export"}
            assert set(sp).issubset(
                allowed_sp_keys
            ), f"unexpected sync_profiles keys: {set(sp) - allowed_sp_keys}"
            if "public_export" in sp:
                assert isinstance(sp["public_export"], bool)


@pytest.mark.repo_data("conferences/neurips2025/storage/storage_manifest.json")
def test_neurips_manifest_covers_inventory_families():
    manifest = load_json(NEURIPS_MANIFEST_PATH)
    inventory = load_json(INVENTORY_PATH)

    manifest_paths = {
        family["path"] for family in manifest["artifact_families"] if "path" in family
    }
    exemptions = set(manifest["inventory_exemptions"])
    inventory_paths = {family["path"] for family in inventory["artifact_families"]}

    uncovered = sorted(
        path for path in inventory_paths if path not in manifest_paths and path not in exemptions
    )
    assert uncovered == []


@pytest.mark.repo_data("conferences/neurips2025/config.yaml")
def test_configs_reference_manifest_with_plain_text_checks():
    pattern = re.compile(r'manifest_path:\s*"storage/storage_manifest\.json"')
    for config_path in (NEURIPS_CONFIG_PATH, TEMPLATE_CONFIG_PATH):
        content = config_path.read_text(encoding="utf-8")
        assert pattern.search(content)
        assert "GoogleDrive" not in content
        assert "/Users/" not in content


def test_starter_manifests_use_lane_based_paths():
    expected_paths = {
        "analysis",
        "decks",
        "audio",
        "exports/deep-dive",
        # Where the extraction owner writes; exports/exhaustive-extractions is NeurIPS 2025 only.
        "storage/extractions",
        "exports/notebooklm_export",
        "agent-work/claude",
        "agent-work/codex",
        "agent-work/gpt-researcher",
        "agent-work/notebook-lm",
        "cache/vector-store",
        "cache/experimental-videos",
        "legacy",
    }
    obsolete_paths = {
        "presentations",
        "podcasts",
        "export_for_deep_dive",
        "exhaustive_extractions",
        "claude-autonomy",
        "codex-autonomy",
        "gpt_researcher",
        "notebook-lm-findings",
        "vector_store",
        "experimental_videos",
    }

    for manifest_path in STARTER_MANIFEST_PATHS:
        manifest = load_json(manifest_path)
        manifest_paths = {
            family["path"] for family in manifest["artifact_families"] if "path" in family
        }

        assert expected_paths.issubset(manifest_paths)
        assert manifest_paths.isdisjoint(obsolete_paths)


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr_download_corpus_manifest_has_private_payloads_and_small_controls():
    manifest_path = REPO_ROOT / "conferences" / "cvpr2026" / "storage" / "storage_manifest.json"
    manifest = load_json(manifest_path)
    families = {family["id"]: family for family in manifest["artifact_families"]}

    report = families["download_corpus_report"]
    assert report["path"] == "storage/download-corpus/download_corpus_report.md"
    assert report["storage_class"] == "git"
    assert report["share_policy"] == "private"
    assert report["sync_profiles"] == {"public": False, "backup": True}

    expected_private_payloads = {
        "download_corpus_ledger": "storage/download-corpus/download_corpus.json",
        "download_corpus_artifacts": "storage/download-corpus/artifacts",
        "download_corpus_youtube_inventory": "storage/download-corpus/youtube/youtube_talk_harvest.json",
        "download_corpus_youtube_captions": "storage/download-corpus/youtube/captions",
        "download_corpus_focused_recovery": "storage/download-corpus/focused-recovery",
    }
    for family_id, expected_path in expected_private_payloads.items():
        family = families[family_id]
        assert family["path"] == expected_path
        assert family["storage_class"] == "backup"
        assert family["share_policy"] == "private"
        assert family["sync_profiles"] == {"public": False, "backup": True}

    worker_logs = families["download_corpus_worker_logs"]
    assert worker_logs["path"] == "storage/download-corpus/logs"
    assert worker_logs["storage_class"] == "cache"
    assert worker_logs["sync_profiles"] == {"public": False, "backup": False}


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr_registration_media_manifest_has_a_private_overlay_and_no_sync_session_seams():
    manifest_path = REPO_ROOT / "conferences" / "cvpr2026" / "storage" / "storage_manifest.json"
    families = {family["id"]: family for family in load_json(manifest_path)["artifact_families"]}

    report = families["registration_media_report"]
    assert (
        report["path"] == "storage/download-corpus/registration-media/registration_media_report.md"
    )
    assert report["storage_class"] == "git"
    assert report["share_policy"] == "private"
    assert report["sync_profiles"] == {"public": False, "backup": True}

    expected_private_payloads = {
        "registration_media_overlay_ledger": "storage/download-corpus/registration-media/registration_media.json",
        "registration_media_captions": "storage/download-corpus/registration-media/captions",
        "registration_media_artifacts": "storage/download-corpus/registration-media/artifacts",
    }
    for family_id, expected_path in expected_private_payloads.items():
        family = families[family_id]
        assert family["path"] == expected_path
        assert family["storage_class"] == "backup"
        assert family["share_policy"] == "private"
        assert family["sync_profiles"] == {"public": False, "backup": True}

    expected_cache = {
        "registration_media_raw_page_cache": "storage/download-corpus/registration-media/raw-pages",
        "registration_media_worker_logs": "storage/download-corpus/registration-media/logs",
        "registration_media_diagnostics": "storage/download-corpus/registration-media/diagnostics",
        "registration_media_session_cache": "storage/download-corpus/registration-media/session",
    }
    for family_id, expected_path in expected_cache.items():
        family = families[family_id]
        assert family["path"] == expected_path
        assert family["storage_class"] == "cache"
        assert family["share_policy"] == "none"
        assert family["sync_profiles"] == {"public": False, "backup": False}


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr_event_media_manifest_has_private_payloads_and_local_only_seams():
    manifest_path = REPO_ROOT / "conferences" / "cvpr2026" / "storage" / "storage_manifest.json"
    families = {family["id"]: family for family in load_json(manifest_path)["artifact_families"]}

    expected_private_payloads = {
        "event_media_inventory": "storage/download-corpus/event-media/event_media_inventory.json",
        "event_media_captions": "storage/download-corpus/event-media/captions",
    }
    for family_id, expected_path in expected_private_payloads.items():
        family = families[family_id]
        assert family["path"] == expected_path
        assert family["storage_class"] == "backup"
        assert family["share_policy"] == "private"
        assert family["sync_profiles"] == {"public": False, "backup": True}

    expected_cache = {
        "event_media_checkpoint_cache": "storage/download-corpus/event-media/event_media.checkpoint.json",
        "event_media_raw_page_cache": "storage/download-corpus/event-media/raw-pages",
        "event_media_worker_logs": "storage/download-corpus/event-media/logs",
        "event_media_session_cache": "storage/download-corpus/event-media/session",
    }
    for family_id, expected_path in expected_cache.items():
        family = families[family_id]
        assert family["path"] == expected_path
        assert family["storage_class"] == "cache"
        assert family["share_policy"] == "none"
        assert family["sync_profiles"] == {"public": False, "backup": False}


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr_transcription_ocr_manifest_has_bounded_control_and_private_payloads():
    manifest_path = REPO_ROOT / "conferences" / "cvpr2026" / "storage" / "storage_manifest.json"
    families = {family["id"]: family for family in load_json(manifest_path)["artifact_families"]}

    inventory = families["transcription_ocr_inventory"]
    assert inventory["path"] == "storage/transcription-ocr/cvpr2026/inventory.json"
    assert inventory["storage_class"] == "backup"
    assert inventory["share_policy"] == "private"
    assert inventory["sync_profiles"] == {"public": False, "backup": True}
    assert "cvpr2026-live-production-r5-event-media" in inventory["notes"]
    assert "workshop_pdf_provenance_table" in inventory["notes"]
    assert "directory" in inventory["notes"]

    report = families["transcription_ocr_report"]
    assert report["path"] == "storage/transcription-ocr/transcription_ocr_report.md"
    assert report["storage_class"] == "git"
    assert report["share_policy"] == "private"
    assert report["sync_profiles"] == {"public": False, "backup": True}
    assert "aggregate-only" in report["notes"]

    retry_controls = {
        "transcription_ocr_pre_repair_blocked_archive": "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-05-transcription-ocr.pre-repair-blocked.json",
        "transcription_ocr_retry_authorization_record": "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-05-transcription-ocr.retry-authorization.json",
        "transcription_ocr_retry_attempt_record": "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-05-transcription-ocr.retry-attempt.json",
        "transcription_ocr_receipt_finalization": "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-05-transcription-ocr.receipt-finalization.json",
    }
    for family_id, expected_path in retry_controls.items():
        family = families[family_id]
        assert family["path"] == expected_path
        assert "path_glob" not in family
        assert family["storage_class"] == "backup"
        assert family["share_policy"] == "private"
        assert family["sync_profiles"] == {"public": False, "backup": True}
    assert not any(
        "workflow-runs" in str(family.get("path_glob", "")) for family in families.values()
    )

    expected_private_payloads = {
        "transcription_ocr_caption_sidecars": "storage/transcription-ocr/cvpr2026/caption-transcripts",
        "transcription_ocr_paper_text": "storage/transcription-ocr/cvpr2026/paper-text",
        "transcription_ocr_speech_text": "storage/transcription-ocr/cvpr2026/speech-text",
        "transcription_ocr_workshop_talks": "storage/transcription-ocr/cvpr2026/workshop-talks",
        "transcription_ocr_scope_waivers": "storage/transcription-ocr/cvpr2026/scope-waivers",
    }
    for family_id, expected_path in expected_private_payloads.items():
        family = families[family_id]
        assert family["path"] == expected_path
        assert family["storage_class"] == "backup"
        assert family["share_policy"] == "private"
        assert family["sync_profiles"] == {"public": False, "backup": True}

    expected_cache_seams = {
        "transcription_ocr_checkpoint_cache": "storage/transcription-ocr/cvpr2026/inventory.checkpoint.json",
        "transcription_ocr_raw_input_cache": "storage/transcription-ocr/cvpr2026/raw-inputs",
        "transcription_ocr_worker_logs": "storage/transcription-ocr/cvpr2026/logs",
        "transcription_ocr_session_cache": "storage/transcription-ocr/cvpr2026/session",
    }
    for family_id, expected_path in expected_cache_seams.items():
        family = families[family_id]
        assert family["path"] == expected_path
        assert family["storage_class"] == "cache"
        assert family["share_policy"] == "none"
        assert family["sync_profiles"] == {"public": False, "backup": False}


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr_r6_manifest_declares_only_narrow_recovery_and_processing_families():
    manifest_path = REPO_ROOT / "conferences" / "cvpr2026" / "storage" / "storage_manifest.json"
    families = {family["id"]: family for family in load_json(manifest_path)["artifact_families"]}

    durable = {
        "r6_recovery_inventory": "storage/transcription-ocr/cvpr2026/r6/recovery_inventory.json",
        "r6_audio_recovery_receipt": "storage/transcription-ocr/cvpr2026/r6/audio_recovery_receipt.json",
        "r6_material_discovery_receipt": "storage/transcription-ocr/cvpr2026/r6/material_discovery_receipt.json",
        "r6_material_acquisition_receipt": "storage/transcription-ocr/cvpr2026/r6/material_acquisition_receipt.json",
        "r6_material_payloads": "storage/transcription-ocr/cvpr2026/r6/material-payloads",
        "r6_budget_ledger": "storage/transcription-ocr/cvpr2026/r6/budget_ledger.json",
        "r6_processing_inventory": "storage/transcription-ocr/cvpr2026/r6/r6_processing_inventory.json",
        "r6_speech_text": "storage/transcription-ocr/cvpr2026/r6/speech-text",
        "r6_paper_text": "storage/transcription-ocr/cvpr2026/r6/paper-text",
        "r6_slide_ocr": "storage/transcription-ocr/cvpr2026/r6/slide-ocr",
        "r6_final_receipt": "storage/workflow-runs/cvpr2026-live-production-r6-expanded-media/stage-05-transcription-ocr-r6.json",
    }
    for family_id, path in durable.items():
        family = families[family_id]
        assert family["path"] == path
        assert family["storage_class"] == "backup"
        assert family["share_policy"] == "private"
        assert family["sync_profiles"] == {"public": False, "backup": True}
        assert "path_glob" not in family

    excluded = {
        "r6_raw_audio_cache": "storage/transcription-ocr/cvpr2026/r6/raw-audio",
        "r6_auth_page_cache": "storage/transcription-ocr/cvpr2026/r6/auth-pages",
        "r6_logs": "storage/transcription-ocr/cvpr2026/r6/logs",
        "r6_session_cache": "storage/transcription-ocr/cvpr2026/r6/session",
    }
    for family_id, path in excluded.items():
        family = families[family_id]
        assert family["path"] == path
        assert family["storage_class"] == "cache"
        assert family["sync_profiles"] == {"public": False, "backup": False}
    assert not any(
        str(family.get("path_glob", "")).startswith("storage/transcription-ocr/cvpr2026/r6")
        or family.get("path") == "storage/transcription-ocr/cvpr2026/r6"
        for family in families.values()
    )
