from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


def load_sync_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "sync_to_gdrive.py"
    spec = importlib.util.spec_from_file_location("sync_to_gdrive", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    conference_dir = tmp_path / "conferences" / "demo2026"
    conference_dir.mkdir(parents=True)
    config_path = conference_dir / "config.yaml"
    config_path.write_text(
        "conference:\n  name: Demo 2026\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    return conference_dir, config_path


def write_manifest(path: Path, artifact_families: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace": "conferences/demo2026",
                "inventory_exemptions": [],
                "artifact_families": artifact_families,
            }
        ),
        encoding="utf-8",
    )


def make_manifest_workspace(tmp_path: Path) -> tuple[Path, Path]:
    conference_dir, config_path = make_workspace(tmp_path)
    for family_path in ("presentations", "podcasts", "analysis", "vector_store"):
        target = conference_dir / family_path
        target.mkdir(parents=True, exist_ok=True)
        (target / "artifact.txt").write_text(family_path, encoding="utf-8")
    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "presentations",
                "path": "presentations",
                "sync_profiles": {"public": True, "backup": True},
            },
            {
                "id": "podcasts",
                "path": "podcasts",
                "sync_profiles": {"public": True, "backup": True},
            },
            {
                "id": "analysis",
                "path": "analysis",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "vector_store",
                "path": "vector_store",
                "sync_profiles": {"public": False, "backup": False},
            },
        ],
    )
    return conference_dir, config_path


def make_source_discovery_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    conference_dir, config_path = make_workspace(tmp_path)
    inventory_path = conference_dir / "storage" / "source-discovery" / "source_discovery_inventory.json"
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text('{"inventory_version": 1}\n', encoding="utf-8")
    raw_snapshot = conference_dir / "cache" / "source-discovery" / "schedule-response.json"
    raw_snapshot.parent.mkdir(parents=True)
    raw_snapshot.write_text('{"upstream": "raw"}\n', encoding="utf-8")
    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "source_discovery_inventory",
                "path": "storage/source-discovery/source_discovery_inventory.json",
                "storage_class": "git",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "source_discovery_raw_snapshots",
                "path": "cache/source-discovery",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
        ],
    )
    return conference_dir, config_path, inventory_path


def make_corpus_freeze_workspace(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    """Create exact control artifacts alongside backup-ineligible payloads."""
    conference_dir, config_path = make_workspace(tmp_path)
    control_files = {
        "corpus_freeze_ledger": (
            conference_dir / "storage" / "corpus-freeze" / "corpus_freeze.json"
        ),
        "corpus_freeze_report": (
            conference_dir / "storage" / "corpus-freeze" / "corpus_freeze_report.md"
        ),
        "corpus_freeze_operator_exclusions": (
            conference_dir / "storage" / "corpus-freeze" / "operator_exclusions.json"
        ),
    }
    control_files["corpus_freeze_ledger"].parent.mkdir(parents=True)
    control_files["corpus_freeze_ledger"].write_text('{"status": "succeeded"}\n', encoding="utf-8")
    control_files["corpus_freeze_report"].write_text("# Corpus freeze\n", encoding="utf-8")
    control_files["corpus_freeze_operator_exclusions"].write_text("[]\n", encoding="utf-8")

    raw_snapshot = conference_dir / "cache" / "source-discovery" / "schedule-response.json"
    raw_snapshot.parent.mkdir(parents=True)
    raw_snapshot.write_text('{"raw": true}\n', encoding="utf-8")
    credential_path = conference_dir / "agent-work" / "operator" / "credentials.json"
    credential_path.parent.mkdir(parents=True)
    credential_path.write_text('{"placeholder": true}\n', encoding="utf-8")
    generated_media = conference_dir / "decks" / "topic-1" / "slide.png"
    generated_media.parent.mkdir(parents=True)
    generated_media.write_bytes(b"generated-media")
    vector_payload = conference_dir / "cache" / "vector-store" / "vectors.bin"
    vector_payload.parent.mkdir(parents=True)
    vector_payload.write_bytes(b"vector-payload")

    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "corpus_freeze_ledger",
                "path": "storage/corpus-freeze/corpus_freeze.json",
                "storage_class": "git",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "corpus_freeze_report",
                "path": "storage/corpus-freeze/corpus_freeze_report.md",
                "storage_class": "git",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "corpus_freeze_operator_exclusions",
                "path": "storage/corpus-freeze/operator_exclusions.json",
                "storage_class": "git",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "source_discovery_raw_snapshots",
                "path": "cache/source-discovery",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "operator_credentials",
                "path": "agent-work/operator/credentials.json",
                "storage_class": "backup",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "generated_media",
                "path_glob": "decks/**/*.png",
                "storage_class": "public",
                "share_policy": "curated",
                "sync_profiles": {"public": True, "backup": False},
            },
            {
                "id": "vector_store",
                "path": "cache/vector-store",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
        ],
    )
    return conference_dir, config_path, control_files


def make_download_corpus_workspace(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    """Create Stage 4 files plus paths that private backup must exclude."""

    conference_dir, config_path = make_workspace(tmp_path)
    selected_files = {
        "download_corpus_ledger": (
            conference_dir / "storage" / "download-corpus" / "download_corpus.json"
        ),
        "download_corpus_report": (
            conference_dir / "storage" / "download-corpus" / "download_corpus_report.md"
        ),
        "download_corpus_artifacts": (
            conference_dir / "storage" / "download-corpus" / "artifacts" / "paper.pdf"
        ),
        "download_corpus_youtube_inventory": (
            conference_dir
            / "storage"
            / "download-corpus"
            / "youtube"
            / "youtube_talk_harvest.json"
        ),
        "download_corpus_youtube_captions": (
            conference_dir
            / "storage"
            / "download-corpus"
            / "youtube"
            / "captions"
            / "talk.vtt"
        ),
        "download_corpus_focused_recovery": (
            conference_dir
            / "storage"
            / "download-corpus"
            / "focused-recovery"
            / "receipt.json"
        ),
    }
    for path in selected_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name + "\n", encoding="utf-8")

    excluded_paths = (
        conference_dir / "storage" / "download-corpus" / "logs" / "worker.log",
        conference_dir / "cache" / "source-discovery" / "response.json",
        conference_dir / "cache" / "auth" / "cookies.json",
        conference_dir / ".env",
        conference_dir / "cache" / "vector-store" / "vectors.bin",
        conference_dir / "cache" / "experimental-videos" / "preview.mp4",
    )
    for path in excluded_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded\n", encoding="utf-8")

    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "download_corpus_ledger",
                "path": "storage/download-corpus/download_corpus.json",
                "storage_class": "backup",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "download_corpus_report",
                "path": "storage/download-corpus/download_corpus_report.md",
                "storage_class": "git",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "download_corpus_artifacts",
                "path": "storage/download-corpus/artifacts",
                "storage_class": "backup",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "download_corpus_youtube_inventory",
                "path": "storage/download-corpus/youtube/youtube_talk_harvest.json",
                "storage_class": "backup",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "download_corpus_youtube_captions",
                "path": "storage/download-corpus/youtube/captions",
                "storage_class": "backup",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "download_corpus_focused_recovery",
                "path": "storage/download-corpus/focused-recovery",
                "storage_class": "backup",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "download_corpus_worker_logs",
                "path": "storage/download-corpus/logs",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "source_discovery_raw_snapshots",
                "path": "cache/source-discovery",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "auth_cookies",
                "path": "cache/auth/cookies.json",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "workspace_environment",
                "path": ".env",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "vector_store",
                "path": "cache/vector-store",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "experimental_videos",
                "path": "cache/experimental-videos",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
        ],
    )
    return conference_dir, config_path, selected_files


def make_transcription_ocr_workspace(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    """Create declared Stage 5 controls/payloads plus backup-ineligible paths."""

    conference_dir, config_path = make_workspace(tmp_path)
    selected_files = {
        "transcription_ocr_inventory": (
            conference_dir
            / "storage"
            / "transcription-ocr"
            / "cvpr2026"
            / "inventory.json"
        ),
        "transcription_ocr_report": (
            conference_dir
            / "storage"
            / "transcription-ocr"
            / "transcription_ocr_report.md"
        ),
        "transcription_ocr_caption_sidecars": (
            conference_dir
            / "storage"
            / "transcription-ocr"
            / "cvpr2026"
            / "caption-transcripts"
            / "talk.json"
        ),
        "transcription_ocr_paper_text": (
            conference_dir
            / "storage"
            / "transcription-ocr"
            / "cvpr2026"
            / "paper-text"
            / "paper.txt"
        ),
        "transcription_ocr_speech_text": (
            conference_dir
            / "storage"
            / "transcription-ocr"
            / "cvpr2026"
            / "speech-text"
            / "talk.json"
        ),
        "transcription_ocr_workshop_talks": (
            conference_dir
            / "storage"
            / "transcription-ocr"
            / "cvpr2026"
            / "workshop-talks"
            / "talks.json"
        ),
        "transcription_ocr_scope_waivers": (
            conference_dir
            / "storage"
            / "transcription-ocr"
            / "cvpr2026"
            / "scope-waivers"
            / "waiver.json"
        ),
        "transcription_ocr_pre_repair_blocked_archive": (
            conference_dir
            / "storage"
            / "workflow-runs"
            / "cvpr2026-live-production-r5-event-media"
            / "stage-05-transcription-ocr.pre-repair-blocked.json"
        ),
        "transcription_ocr_retry_authorization_record": (
            conference_dir
            / "storage"
            / "workflow-runs"
            / "cvpr2026-live-production-r5-event-media"
            / "stage-05-transcription-ocr.retry-authorization.json"
        ),
        "transcription_ocr_retry_attempt_record": (
            conference_dir
            / "storage"
            / "workflow-runs"
            / "cvpr2026-live-production-r5-event-media"
            / "stage-05-transcription-ocr.retry-attempt.json"
        ),
        "transcription_ocr_receipt_finalization": (
            conference_dir
            / "storage"
            / "workflow-runs"
            / "cvpr2026-live-production-r5-event-media"
            / "stage-05-transcription-ocr.receipt-finalization.json"
        ),
    }
    for path in selected_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name + "\n", encoding="utf-8")

    excluded_paths = (
        conference_dir
        / "storage"
        / "transcription-ocr"
        / "cvpr2026"
        / "inventory.checkpoint.json",
        conference_dir
        / "storage"
        / "transcription-ocr"
        / "cvpr2026"
        / "raw-inputs"
        / "event.html",
        conference_dir / "storage" / "transcription-ocr" / "cvpr2026" / "logs" / "worker.log",
        conference_dir / "storage" / "transcription-ocr" / "cvpr2026" / "session" / "cookies.txt",
        conference_dir / "cache" / "source-discovery" / "response.json",
        conference_dir / "cache" / "vector-store" / "vectors.bin",
        conference_dir / "agent-work" / "operator" / "credentials.json",
        conference_dir / ".env",
    )
    for path in excluded_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded\n", encoding="utf-8")

    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "transcription_ocr_inventory",
                "path": "storage/transcription-ocr/cvpr2026/inventory.json",
                "storage_class": "backup",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            {
                "id": "transcription_ocr_report",
                "path": "storage/transcription-ocr/transcription_ocr_report.md",
                "storage_class": "git",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": True},
            },
            *[
                {
                    "id": family_id,
                    "path": (
                        path.relative_to(conference_dir).as_posix()
                        if family_id
                        in {
                            "transcription_ocr_pre_repair_blocked_archive",
                            "transcription_ocr_retry_authorization_record",
                            "transcription_ocr_retry_attempt_record",
                            "transcription_ocr_receipt_finalization",
                        }
                        else path.relative_to(conference_dir).parent.as_posix()
                    ),
                    "storage_class": "backup",
                    "share_policy": "private",
                    "sync_profiles": {"public": False, "backup": True},
                }
                for family_id, path in selected_files.items()
                if family_id not in {"transcription_ocr_inventory", "transcription_ocr_report"}
            ],
            {
                "id": "transcription_ocr_checkpoint_cache",
                "path": "storage/transcription-ocr/cvpr2026/inventory.checkpoint.json",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "transcription_ocr_raw_input_cache",
                "path": "storage/transcription-ocr/cvpr2026/raw-inputs",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "transcription_ocr_worker_logs",
                "path": "storage/transcription-ocr/cvpr2026/logs",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "transcription_ocr_session_cache",
                "path": "storage/transcription-ocr/cvpr2026/session",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "source_discovery_raw_snapshots",
                "path": "cache/source-discovery",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "vector_store",
                "path": "cache/vector-store",
                "storage_class": "cache",
                "share_policy": "none",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "operator_credentials",
                "path": "agent-work/operator/credentials.json",
                "storage_class": "backup",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": False},
            },
            {
                "id": "workspace_environment",
                "path": ".env",
                "storage_class": "backup",
                "share_policy": "private",
                "sync_profiles": {"public": False, "backup": False},
            },
        ],
    )
    return conference_dir, config_path, selected_files


def test_public_profile_target_resolution_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_sync_module()
    conference_dir, _ = make_workspace(tmp_path)
    public_target = tmp_path / "public-target"
    monkeypatch.setenv("GDRIVE_PUBLIC_PATH", str(public_target))

    args = module.parse_args(
        ["--conference-dir", str(conference_dir), "--profile", "public", "--dry-run"]
    )
    config = module.normalize_args(args)

    assert config.profile == "public"
    assert config.conference_dir == conference_dir.resolve()
    assert config.remote_path == public_target.resolve()
    assert config.dry_run is True


def test_backup_profile_requires_explicit_remote_path(tmp_path: Path):
    module = load_sync_module()
    conference_dir, _ = make_workspace(tmp_path)

    args = module.parse_args(["--conference-dir", str(conference_dir), "--profile", "backup"])

    with pytest.raises(ValueError, match="explicit --remote-path"):
        module.normalize_args(args)


def test_remote_path_override_beats_profile_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    conference_dir, _ = make_workspace(tmp_path)
    monkeypatch.setenv("GDRIVE_PUBLIC_PATH", str(tmp_path / "env-public"))
    override_target = tmp_path / "override-public"

    args = module.parse_args(
        [
            "--conference-dir",
            str(conference_dir),
            "--profile",
            "public",
            "--remote-path",
            str(override_target),
        ]
    )
    config = module.normalize_args(args)

    assert config.remote_path == override_target.resolve()


def test_config_locator_resolves_parent_directory(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    override_target = tmp_path / "backup-target"

    args = module.parse_args(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(override_target),
        ]
    )
    config = module.normalize_args(args)

    assert config.config_path == config_path.resolve()
    assert config.conference_dir == conference_dir.resolve()


def test_missing_profile_target_raises_clear_error(tmp_path: Path):
    module = load_sync_module()
    conference_dir, _ = make_workspace(tmp_path)
    args = module.parse_args(["--conference-dir", str(conference_dir), "--profile", "public"])

    with pytest.raises(ValueError, match="No destination configured for profile"):
        module.normalize_args(args, environ={})


def test_directory_config_path_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = load_sync_module()
    config_dir = tmp_path / "config-dir"
    config_dir.mkdir()

    exit_code = module.main(
        [
            "--config",
            str(config_dir),
            "--profile",
            "public",
            "--remote-path",
            str(tmp_path / "dest"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"Conference config does not exist: {config_dir.resolve()}" in captured.out
    assert captured.err == ""


def test_manifest_path_resolution_uses_yaml_config_pointer(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    config_path.write_text(
        "conference:\n  name: Demo 2026\nstorage:\n  manifest_path: custom/manifest.json\n",
        encoding="utf-8",
    )
    manifest_path = conference_dir / "custom" / "manifest.json"
    write_manifest(
        manifest_path,
        [
            {
                "id": "presentations",
                "path": "presentations",
                "sync_profiles": {"public": True, "backup": True},
            }
        ],
    )

    resolved = module.resolve_manifest_path(conference_dir, config_path)

    assert resolved == manifest_path.resolve()


def test_manifest_path_rejects_parent_traversal(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    config_path.write_text(
        "conference:\n  name: Demo 2026\nstorage:\n  manifest_path: ../outside/manifest.json\n",
        encoding="utf-8",
    )
    outside_manifest = tmp_path / "conferences" / "outside" / "manifest.json"
    write_manifest(
        outside_manifest,
        [
            {
                "id": "presentations",
                "path": "presentations",
                "sync_profiles": {"public": True, "backup": True},
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"storage\.manifest_path must stay inside the conference workspace",
    ):
        module.resolve_manifest_path(conference_dir, config_path)


def test_manifest_family_selection_matches_profile_invariants(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_manifest_workspace(tmp_path)
    args = module.parse_args(
        ["--config", str(config_path), "--profile", "backup", "--remote-path", str(tmp_path / "dest")]
    )
    plan = module.build_sync_plan(module.normalize_args(args))

    assert "presentations" in plan.included_families
    assert "podcasts" in plan.included_families
    assert "analysis" in plan.included_families
    assert "vector_store" in plan.excluded_families


def test_missing_manifest_fails_closed(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    config_path.write_text(
        "conference:\n  name: Demo 2026\nstorage:\n  manifest_path: storage/missing.json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No storage manifest found for conference. Sync aborted for safety."):
        module.resolve_manifest_path(conference_dir, config_path)


def test_manifest_directory_path_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    manifest_dir = conference_dir / "storage" / "storage_manifest.json"
    manifest_dir.mkdir(parents=True)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "public",
            "--remote-path",
            str(tmp_path / "dest"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"Storage manifest is not a file: {manifest_dir.resolve()}" in captured.out
    assert captured.err == ""


def test_malformed_yaml_config_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    config_path.write_text("conference: [demo\nstorage:\n  manifest_path: broken.json\n", encoding="utf-8")

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "public",
            "--remote-path",
            str(conference_dir / "dest"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"Could not read conference config: {config_path.resolve()}" in captured.out
    assert captured.err == ""


def test_malformed_manifest_sync_profiles_fail_loudly(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "presentations",
                "path": "presentations",
                "sync_profiles": {"public": True},
            }
        ],
    )
    args = module.parse_args(
        ["--config", str(config_path), "--profile", "public", "--remote-path", str(tmp_path / "dest")]
    )

    with pytest.raises(ValueError, match="Malformed storage manifest"):
        module.build_sync_plan(module.normalize_args(args))


def test_manifest_family_path_rejects_parent_traversal(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    outside_dir = conference_dir.parent / "outside"
    outside_dir.mkdir(parents=True)
    (outside_dir / "artifact.txt").write_text("outside", encoding="utf-8")
    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "presentations",
                "path": "../outside",
                "sync_profiles": {"public": True, "backup": True},
            }
        ],
    )
    args = module.parse_args(
        ["--config", str(config_path), "--profile", "public", "--remote-path", str(tmp_path / "dest")]
    )

    with pytest.raises(
        ValueError,
        match=r"presentations\.path must stay inside the conference workspace",
    ):
        module.build_sync_plan(module.normalize_args(args))


def test_manifest_family_symlink_escape_fails_closed(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    outside_dir = tmp_path / "outside-analysis"
    outside_dir.mkdir()
    (outside_dir / "artifact.txt").write_text("outside", encoding="utf-8")
    (conference_dir / "analysis").symlink_to(outside_dir, target_is_directory=True)
    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "analysis",
                "path": "analysis",
                "sync_profiles": {"public": True, "backup": True},
            }
        ],
    )
    args = module.parse_args(
        ["--config", str(config_path), "--profile", "public", "--remote-path", str(tmp_path / "dest")]
    )

    with pytest.raises(
        ValueError,
        match=r"analysis\.path resolves outside the conference workspace",
    ):
        module.build_sync_plan(module.normalize_args(args))


def test_manifest_family_drive_backed_symlink_is_admitted(tmp_path: Path):
    """A family explicitly declared storage_class backup/public/cache -- the
    on-demand Drive/cache leaves such as storage/workshops/papers or
    storage/transcription-ocr -- is a deliberate, operator-placed symlink
    whose real target sits outside the repository tree entirely. Unlike
    test_manifest_family_symlink_escape_fails_closed's ungoverned family,
    this must be admitted, not rejected, or the backup profile can never
    sync any of the repo's own Drive-backed storage leaves."""
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    drive_backed_root = tmp_path / "drive-backed-store"
    drive_backed_root.mkdir()
    (drive_backed_root / "paper.pdf").write_text("paper", encoding="utf-8")
    (conference_dir / "storage").mkdir(parents=True, exist_ok=True)
    (conference_dir / "storage" / "workshops").mkdir(parents=True, exist_ok=True)
    (conference_dir / "storage" / "workshops" / "papers").symlink_to(
        drive_backed_root, target_is_directory=True
    )
    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "workshop_paper_payloads",
                "path": "storage/workshops/papers",
                "storage_class": "public",
                "sync_profiles": {"public": True, "backup": True},
            }
        ],
    )
    args = module.parse_args(
        ["--config", str(config_path), "--profile", "backup", "--remote-path", str(tmp_path / "dest")]
    )

    plan = module.build_sync_plan(module.normalize_args(args))

    assert "workshop_paper_payloads" in plan.included_families
    assert any(item.family_id == "workshop_paper_payloads" for item in plan.sync_items)


def test_public_dry_run_report_mentions_presentations_not_analysis(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_manifest_workspace(tmp_path)
    destination_root = tmp_path / "dest"
    args = module.parse_args(
        ["--config", str(config_path), "--profile", "public", "--remote-path", str(destination_root), "--dry-run"]
    )
    plan = module.build_sync_plan(module.normalize_args(args))
    destination_status = module.prepare_destination_root(destination_root, dry_run=True)
    report = module.render_preflight_report(plan, "1M", destination_status)

    assert "Included families:" in report
    assert "Target root status: reachable" in report
    assert "Leaf status: missing" in report
    assert "presentations" in report
    included_line = next(line for line in report.splitlines() if line.startswith("Included families:"))
    assert "analysis" not in included_line


def test_backup_dry_run_report_mentions_presentations_and_analysis(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_manifest_workspace(tmp_path)
    destination_root = tmp_path / "dest"
    args = module.parse_args(
        ["--config", str(config_path), "--profile", "backup", "--remote-path", str(destination_root), "--dry-run"]
    )
    plan = module.build_sync_plan(module.normalize_args(args))
    destination_status = module.prepare_destination_root(destination_root, dry_run=True)
    report = module.render_preflight_report(plan, "1M", destination_status)

    included_line = next(line for line in report.splitlines() if line.startswith("Included families:"))
    assert "Target root status: reachable" in report
    assert "Leaf status: missing" in report
    assert "presentations" in included_line
    assert "analysis" in included_line


def test_missing_root_raises_clear_error(tmp_path: Path):
    module = load_sync_module()
    missing_root_destination = tmp_path / "missing-root" / "conference-leaf"

    with pytest.raises(ValueError, match="Remote root does not exist:"):
        module.prepare_destination_root(missing_root_destination, dry_run=True)


def test_delete_requires_reviewed_dry_run_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    module = load_sync_module()
    conference_dir, config_path = make_manifest_workspace(tmp_path)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(tmp_path / "dest"),
            "--delete",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Refusing destructive sync without reviewed dry run" in captured.out


def test_delete_dry_run_allows_preview_without_reviewed_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    module = load_sync_module()
    _conference_dir, config_path = make_manifest_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    stale_file = remote_leaf / "presentations" / "obsolete.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale", encoding="utf-8")
    rsync_commands: list[list[str]] = []

    def fake_run_rsync(command: list[str]) -> int:
        rsync_commands.append(command)
        return 0

    monkeypatch.setattr(module, "run_rsync", fake_run_rsync)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "public",
            "--remote-path",
            str(remote_leaf),
            "--dry-run",
            "--delete",
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert stale_file.exists()
    assert rsync_commands
    assert all("--dry-run" in command for command in rsync_commands)
    assert all("--delete" in command for command in rsync_commands)
    assert "Delete mode: enabled" in captured.out
    assert "Dry run: enabled" in captured.out


def test_delete_mode_only_targets_included_family_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path = make_manifest_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    stale_sibling = remote_leaf / "legacy" / "obsolete.txt"
    stale_sibling.parent.mkdir(parents=True)
    stale_sibling.write_text("stale", encoding="utf-8")
    rsync_commands: list[list[str]] = []

    def fake_run_rsync(command: list[str]) -> int:
        rsync_commands.append(command)
        return 0

    monkeypatch.setattr(module, "run_rsync", fake_run_rsync)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "public",
            "--remote-path",
            str(remote_leaf),
            "--dry-run",
            "--delete",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert stale_sibling.exists()
    assert rsync_commands
    assert {command[-1] for command in rsync_commands} == {
        f"{remote_leaf / 'podcasts'}/",
        f"{remote_leaf / 'presentations'}/",
    }
    assert all(command[-1] != f"{remote_leaf}/" for command in rsync_commands)


def test_rsync_command_preserves_family_relative_paths(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_manifest_workspace(tmp_path)
    args = module.parse_args(
        ["--config", str(config_path), "--profile", "backup", "--remote-path", str(tmp_path / "dest"), "--dry-run"]
    )
    plan = module.build_sync_plan(module.normalize_args(args))
    analysis_item = next(item for item in plan.sync_items if item.family_id == "analysis")
    rsync_command = module.build_rsync_command(
        analysis_item,
        dry_run=True,
        delete=False,
        quiet=True,
    )

    assert analysis_item.destination_path == (tmp_path / "dest" / "analysis").resolve()
    assert rsync_command[0] == "rsync"
    assert rsync_command[-1].endswith("/analysis/")


def test_dry_run_missing_leaf_does_not_create_remote_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path = make_manifest_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    monkeypatch.delenv("GDRIVE_PUBLIC_PATH", raising=False)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "public",
            "--remote-path",
            str(remote_leaf),
            "--dry-run",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert remote_root.exists()
    assert not remote_leaf.exists()
    assert not (remote_leaf / "presentations").exists()
    assert not (remote_leaf / "podcasts").exists()


def test_write_mode_creates_missing_leaf_before_rsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path = make_manifest_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    rsync_leaf_states: list[bool] = []

    def fake_run_rsync(_command: list[str]) -> int:
        rsync_leaf_states.append(remote_leaf.exists())
        return 0

    monkeypatch.setattr(module, "run_rsync", fake_run_rsync)
    monkeypatch.delenv("GDRIVE_PUBLIC_PATH", raising=False)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "public",
            "--remote-path",
            str(remote_leaf),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert remote_leaf.exists()
    assert rsync_leaf_states
    assert all(rsync_leaf_states)


def test_path_glob_family_syncs_matches_with_relative_destinations(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    slide_dir = conference_dir / "decks" / "topic-a"
    slide_dir.mkdir(parents=True)
    (slide_dir / "presentation.html").write_text("<html></html>", encoding="utf-8")
    (slide_dir / "notes.md").write_text("notes", encoding="utf-8")
    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "rendered_presentations",
                "path_glob": "decks/*/presentation.html",
                "sync_profiles": {"public": True, "backup": True},
            }
        ],
    )

    args = module.parse_args(
        ["--config", str(config_path), "--profile", "public", "--remote-path", str(tmp_path / "dest")]
    )
    plan = module.build_sync_plan(module.normalize_args(args))

    assert plan.included_families == ("rendered_presentations",)
    assert len(plan.sync_items) == 1
    item = plan.sync_items[0]
    assert item.relative_path == Path("decks/topic-a/presentation.html")
    assert item.destination_path == (tmp_path / "dest" / "decks" / "topic-a" / "presentation.html").resolve()


def test_path_glob_family_expands_brace_patterns(tmp_path: Path):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    slide_dir = conference_dir / "decks" / "topic-a"
    slide_dir.mkdir(parents=True)
    (slide_dir / "presentation.html").write_text("<html></html>", encoding="utf-8")
    (slide_dir / "presentation_video.mp4").write_bytes(b"video")
    (slide_dir / "presentation_slide_mapping.json").write_text("{}", encoding="utf-8")
    (slide_dir / "notes.md").write_text("notes", encoding="utf-8")
    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "presentation_payloads",
                "path_glob": "decks/**/*.{html,mp4,json}",
                "sync_profiles": {"public": True, "backup": True},
            }
        ],
    )

    args = module.parse_args(
        ["--config", str(config_path), "--profile", "backup", "--remote-path", str(tmp_path / "dest")]
    )
    plan = module.build_sync_plan(module.normalize_args(args))

    assert plan.included_families == ("presentation_payloads",)
    assert {item.relative_path for item in plan.sync_items} == {
        Path("decks/topic-a/presentation.html"),
        Path("decks/topic-a/presentation_video.mp4"),
        Path("decks/topic-a/presentation_slide_mapping.json"),
    }


def test_secret_like_path_allows_research_token_artifact_names():
    module = load_sync_module()

    assert not module.is_secret_like_path(
        Path("slide_005_comparison_chart_of_fixed_vs_dynamic_token_usage.svg")
    )
    assert not module.is_secret_like_path(
        Path("slide_003_diagram_showing_tokens_taking_different_paths.svg")
    )


def test_secret_like_path_blocks_credential_token_names():
    module = load_sync_module()

    assert module.is_secret_like_path(Path(".env"))
    assert module.is_secret_like_path(Path("access_token.json"))
    assert module.is_secret_like_path(Path("client_secret.json"))
    assert module.is_secret_like_path(Path("service_account_credentials.json"))
    assert module.is_secret_like_path(Path("openai_api_key.txt"))


def test_receipt_records_manifest_and_artifact_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path = make_manifest_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    receipt_path = tmp_path / "receipt.json"

    monkeypatch.setattr(module, "run_rsync", lambda _command: 0)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--receipt-path",
            str(receipt_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 3
    assert receipt["manifest_sha256"]
    assert receipt["artifact_snapshot_sha256"]
    assert receipt["artifact_file_count"] == receipt["digest_count"]
    assert receipt["artifact_size_bytes"] > 0
    assert {item["family_id"] for item in receipt["sync_items"]} >= {
        "analysis",
        "podcasts",
        "presentations",
    }
    assert receipt["bloat_control"]["copy_mode"] == "rsync incremental"


def test_backup_receipt_records_only_exact_git_control_file_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path, inventory_path = make_source_discovery_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    receipt_path = tmp_path / "receipt.json"

    monkeypatch.setattr(module, "run_rsync", lambda _command: 0)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_root / "conference-leaf"),
            "--receipt-path",
            str(receipt_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["included_families"] == ["source_discovery_inventory"]
    assert receipt["selected_control_file_digests"] == [
        {
            "path": "storage/source-discovery/source_discovery_inventory.json",
            "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        }
    ]


@pytest.mark.repo_data("conferences/cvpr2026/storage/storage_manifest.json")
def test_cvpr_manifest_declares_exact_private_corpus_freeze_controls():
    manifest_path = (
        Path(__file__).resolve().parent.parent
        / "conferences"
        / "cvpr2026"
        / "storage"
        / "storage_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    families = {family["id"]: family for family in manifest["artifact_families"]}

    expected_paths = {
        "corpus_freeze_ledger": "storage/corpus-freeze/corpus_freeze.json",
        "corpus_freeze_report": "storage/corpus-freeze/corpus_freeze_report.md",
        "corpus_freeze_operator_exclusions": "storage/corpus-freeze/operator_exclusions.json",
    }
    for family_id, expected_path in expected_paths.items():
        family = families[family_id]
        assert family["path"] == expected_path
        assert "path_glob" not in family
        assert family["storage_class"] == "git"
        assert family["share_policy"] == "private"
        assert family["sync_profiles"] == {"public": False, "backup": True}


def test_corpus_freeze_backup_receipts_bind_only_exact_control_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path, control_files = make_corpus_freeze_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    dry_receipt_path = tmp_path / "corpus-freeze-backup-dry-run.json"
    write_receipt_path = tmp_path / "corpus-freeze-backup-write.json"

    monkeypatch.setattr(module, "run_rsync", lambda _command: 0)

    dry_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--dry-run",
            "--receipt-path",
            str(dry_receipt_path),
            "--quiet",
        ]
    )
    write_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--receipt-path",
            str(write_receipt_path),
            "--quiet",
        ]
    )

    assert dry_exit == 0
    assert write_exit == 0
    dry_receipt = json.loads(dry_receipt_path.read_text(encoding="utf-8"))
    write_receipt = json.loads(write_receipt_path.read_text(encoding="utf-8"))
    expected_families = [
        "corpus_freeze_ledger",
        "corpus_freeze_operator_exclusions",
        "corpus_freeze_report",
    ]
    expected_digests = [
        {
            "path": control_files[family_id].relative_to(
                control_files[family_id].parents[2]
            ).as_posix(),
            "sha256": hashlib.sha256(control_files[family_id].read_bytes()).hexdigest(),
        }
        for family_id in expected_families
    ]
    expected_digests.sort(key=lambda entry: entry["path"])

    for receipt, dry_run in ((dry_receipt, True), (write_receipt, False)):
        assert receipt["status"] == "succeeded"
        assert receipt["dry_run"] is dry_run
        assert receipt["included_families"] == expected_families
        assert receipt["artifact_file_count"] == 3
        assert receipt["digest_count"] == 3
        assert receipt["manifest_sha256"]
        assert receipt["artifact_snapshot_sha256"]
        assert receipt["selected_control_file_digests"] == expected_digests
        assert receipt["secret_like_paths"] == []
        assert receipt["secrets_copied"] is False

    assert dry_receipt["manifest_sha256"] == write_receipt["manifest_sha256"]
    assert dry_receipt["artifact_snapshot_sha256"] == write_receipt["artifact_snapshot_sha256"]
    assert {
        item["relative_path"] for item in write_receipt["sync_items"]
    } == {entry["path"] for entry in expected_digests}
    assert {
        "source_discovery_raw_snapshots",
        "operator_credentials",
        "generated_media",
        "vector_store",
    }.issubset(set(write_receipt["skipped_families"]))


def test_download_corpus_backup_receipts_select_private_families_without_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path, selected_files = make_download_corpus_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    dry_receipt_path = tmp_path / "download-corpus-backup-dry-run.json"
    write_receipt_path = tmp_path / "download-corpus-backup-write.json"

    monkeypatch.setattr(module, "run_rsync", lambda _command: 0)

    dry_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--dry-run",
            "--receipt-path",
            str(dry_receipt_path),
            "--quiet",
        ]
    )
    write_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--receipt-path",
            str(write_receipt_path),
            "--quiet",
        ]
    )

    assert dry_exit == 0
    assert write_exit == 0
    dry_receipt = json.loads(dry_receipt_path.read_text(encoding="utf-8"))
    write_receipt = json.loads(write_receipt_path.read_text(encoding="utf-8"))
    expected_families = sorted(selected_files)
    expected_control_digest = {
        "path": "storage/download-corpus/download_corpus_report.md",
        "sha256": hashlib.sha256(
            selected_files["download_corpus_report"].read_bytes()
        ).hexdigest(),
    }

    for receipt, dry_run in ((dry_receipt, True), (write_receipt, False)):
        assert receipt["status"] == "succeeded"
        assert receipt["dry_run"] is dry_run
        assert receipt["included_families"] == expected_families
        assert receipt["selected_control_file_digests"] == [expected_control_digest]
        assert receipt["secret_like_paths"] == []
        assert receipt["secrets_copied"] is False
        assert receipt["artifact_file_count"] == len(selected_files)
        assert receipt["digest_count"] == len(selected_files)
        assert {
            "download_corpus_worker_logs",
            "source_discovery_raw_snapshots",
            "auth_cookies",
            "workspace_environment",
            "vector_store",
            "experimental_videos",
        }.issubset(set(receipt["skipped_families"]))

    assert dry_receipt["manifest_sha256"] == write_receipt["manifest_sha256"]
    assert dry_receipt["artifact_snapshot_sha256"] == write_receipt["artifact_snapshot_sha256"]
    assert {item["relative_path"] for item in write_receipt["sync_items"]} == {
        "storage/download-corpus/download_corpus.json",
        "storage/download-corpus/download_corpus_report.md",
        "storage/download-corpus/artifacts",
        "storage/download-corpus/youtube/youtube_talk_harvest.json",
        "storage/download-corpus/youtube/captions",
        "storage/download-corpus/focused-recovery",
    }


def test_transcription_ocr_backup_receipts_select_declared_private_families_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path, selected_files = make_transcription_ocr_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    dry_receipt_path = tmp_path / "transcription-ocr-backup-dry-run.json"
    write_receipt_path = tmp_path / "transcription-ocr-backup-write.json"

    monkeypatch.setattr(module, "run_rsync", lambda _command: 0)

    for receipt_path, dry_run in ((dry_receipt_path, True), (write_receipt_path, False)):
        args = [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--receipt-path",
            str(receipt_path),
            "--quiet",
        ]
        if dry_run:
            args.append("--dry-run")
        assert module.main(args) == 0

    dry_receipt = json.loads(dry_receipt_path.read_text(encoding="utf-8"))
    write_receipt = json.loads(write_receipt_path.read_text(encoding="utf-8"))
    expected_families = sorted(selected_files)
    expected_control_digest = {
        "path": "storage/transcription-ocr/transcription_ocr_report.md",
        "sha256": hashlib.sha256(
            selected_files["transcription_ocr_report"].read_bytes()
        ).hexdigest(),
    }

    for receipt, dry_run in ((dry_receipt, True), (write_receipt, False)):
        assert receipt["status"] == "succeeded"
        assert receipt["dry_run"] is dry_run
        assert receipt["included_families"] == expected_families
        assert receipt["selected_control_file_digests"] == [expected_control_digest]
        assert receipt["secret_like_paths"] == []
        assert receipt["secrets_copied"] is False
        assert {
            "transcription_ocr_checkpoint_cache",
            "transcription_ocr_raw_input_cache",
            "transcription_ocr_worker_logs",
            "transcription_ocr_session_cache",
            "source_discovery_raw_snapshots",
            "vector_store",
            "operator_credentials",
            "workspace_environment",
        }.issubset(set(receipt["skipped_families"]))

    assert dry_receipt["manifest_sha256"] == write_receipt["manifest_sha256"]
    assert dry_receipt["artifact_snapshot_sha256"] == write_receipt["artifact_snapshot_sha256"]
    assert {item["relative_path"] for item in write_receipt["sync_items"]} == {
        "storage/transcription-ocr/cvpr2026/inventory.json",
        "storage/transcription-ocr/transcription_ocr_report.md",
        "storage/transcription-ocr/cvpr2026/caption-transcripts",
        "storage/transcription-ocr/cvpr2026/paper-text",
        "storage/transcription-ocr/cvpr2026/speech-text",
        "storage/transcription-ocr/cvpr2026/workshop-talks",
        "storage/transcription-ocr/cvpr2026/scope-waivers",
        "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-05-transcription-ocr.pre-repair-blocked.json",
        "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-05-transcription-ocr.retry-authorization.json",
        "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-05-transcription-ocr.retry-attempt.json",
        "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-05-transcription-ocr.receipt-finalization.json",
    }


def test_init_only_creates_leaf_and_family_dirs_without_rsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path = make_manifest_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    rsync_calls: list[list[str]] = []

    monkeypatch.setattr(module, "run_rsync", lambda command: rsync_calls.append(command) or 0)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--init-only",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert remote_leaf.exists()
    assert (remote_leaf / "analysis").is_dir()
    assert (remote_leaf / "presentations").is_dir()
    assert (remote_leaf / "podcasts").is_dir()
    assert not rsync_calls


def test_init_only_creates_declared_family_dirs_before_local_artifacts_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    conference_dir, config_path = make_workspace(tmp_path)
    write_manifest(
        conference_dir / "storage" / "storage_manifest.json",
        [
            {
                "id": "decks",
                "path": "decks",
                "sync_profiles": {"public": True, "backup": True},
            },
            {
                "id": "audio",
                "path": "audio",
                "sync_profiles": {"public": True, "backup": True},
            },
        ],
    )
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    rsync_calls: list[list[str]] = []

    monkeypatch.setattr(module, "run_rsync", lambda command: rsync_calls.append(command) or 0)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "public",
            "--remote-path",
            str(remote_leaf),
            "--init-only",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert (remote_leaf / "decks").is_dir()
    assert (remote_leaf / "audio").is_dir()
    assert not (conference_dir / "decks").exists()
    assert not rsync_calls


def test_delete_write_can_be_bound_to_matching_dry_run_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path = make_manifest_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    remote_leaf.mkdir()
    dry_receipt = tmp_path / "dry-run-receipt.json"
    write_receipt = tmp_path / "write-receipt.json"
    rsync_commands: list[list[str]] = []

    monkeypatch.setattr(module, "run_rsync", lambda command: rsync_commands.append(command) or 0)

    dry_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--dry-run",
            "--delete",
            "--receipt-path",
            str(dry_receipt),
            "--quiet",
        ]
    )
    write_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--delete",
            "--reviewed-dry-run-receipt",
            str(dry_receipt),
            "--receipt-path",
            str(write_receipt),
            "--quiet",
        ]
    )

    assert dry_exit == 0
    assert write_exit == 0
    assert any("--delete" in command for command in rsync_commands)
    receipt = json.loads(write_receipt.read_text(encoding="utf-8"))
    assert receipt["reviewed_dry_run_receipt"] == str(dry_receipt)


def test_delete_write_rejects_stale_dry_run_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    conference_dir, config_path = make_manifest_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    remote_leaf.mkdir()
    dry_receipt = tmp_path / "dry-run-receipt.json"
    write_receipt = tmp_path / "write-receipt.json"

    monkeypatch.setattr(module, "run_rsync", lambda _command: 0)

    dry_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--dry-run",
            "--delete",
            "--receipt-path",
            str(dry_receipt),
            "--quiet",
        ]
    )
    (conference_dir / "analysis" / "artifact.txt").write_text("changed", encoding="utf-8")
    write_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--delete",
            "--reviewed-dry-run-receipt",
            str(dry_receipt),
            "--receipt-path",
            str(write_receipt),
            "--quiet",
        ]
    )

    assert dry_exit == 0
    assert write_exit != 0
    receipt = json.loads(write_receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "reviewed dry-run receipt does not match" in receipt["error"]


def test_delete_write_rejects_non_delete_dry_run_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path = make_manifest_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    remote_leaf.mkdir()
    dry_receipt = tmp_path / "plain-dry-run-receipt.json"
    write_receipt = tmp_path / "write-receipt.json"

    monkeypatch.setattr(module, "run_rsync", lambda _command: 0)

    dry_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--dry-run",
            "--receipt-path",
            str(dry_receipt),
            "--quiet",
        ]
    )
    write_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--delete",
            "--reviewed-dry-run-receipt",
            str(dry_receipt),
            "--receipt-path",
            str(write_receipt),
            "--quiet",
        ]
    )

    assert dry_exit == 0
    assert write_exit != 0
    receipt = json.loads(write_receipt.read_text(encoding="utf-8"))
    assert "delete" in receipt["error"]


def test_delete_write_rejects_init_only_dry_run_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_sync_module()
    _conference_dir, config_path = make_manifest_workspace(tmp_path)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    remote_leaf = remote_root / "conference-leaf"
    dry_receipt = tmp_path / "init-only-dry-run-receipt.json"
    write_receipt = tmp_path / "write-receipt.json"

    monkeypatch.setattr(module, "run_rsync", lambda _command: 0)

    dry_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--dry-run",
            "--delete",
            "--init-only",
            "--receipt-path",
            str(dry_receipt),
            "--quiet",
        ]
    )
    remote_leaf.mkdir()
    write_exit = module.main(
        [
            "--config",
            str(config_path),
            "--profile",
            "backup",
            "--remote-path",
            str(remote_leaf),
            "--delete",
            "--reviewed-dry-run-receipt",
            str(dry_receipt),
            "--receipt-path",
            str(write_receipt),
            "--quiet",
        ]
    )

    assert dry_exit == 0
    assert write_exit != 0
    receipt = json.loads(write_receipt.read_text(encoding="utf-8"))
    assert "init_only" in receipt["error"]
